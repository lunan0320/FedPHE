

import os
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
import time
import random
from threading import Thread
from utils.util import logging
import tenseal as ts
import pickle
import utils.min_hash as lsh
import time
from functools import reduce
from multiprocessing import Pool,cpu_count
from utils import sampling
from sklearn.cluster import KMeans
from encryption.paillier import paillier_dec,paillier_enc
from encryption.bfv import bfv_dec
from client import params_tolist,params_tomodel
from utils.util import model_init
from client import test_epoch
#######################################
def chunks_idx(l, n):
    d, r = divmod(len(l), n)
    for i in range(n):
        si = (d+1)*(i if i < r else r) + d*(0 if i < r else i - r)
        yield si, si+(d+1 if i < r else d)


def _compress(flatten_array, num_bits):
    res = 0
    l = len(flatten_array)
    for element in flatten_array:
        res <<= num_bits
        res += element

    return res, l

def compress_multi(flatten_array, num_bits):
    l = len(flatten_array)
    MAGIC_N_JOBS = 10
    pool_inputs = []
    sizes = []
    pool = Pool(MAGIC_N_JOBS)
    
    for begin, end in chunks_idx(range(l), MAGIC_N_JOBS):
        sizes.append(end - begin)
        
        pool_inputs.append([flatten_array[begin:end], num_bits])

    pool_outputs = pool.starmap(_compress, pool_inputs)
    pool.close()
    pool.join()
    
    res = 0

    for idx, output in enumerate(pool_outputs):
        res +=  output[0] << (int(np.sum(sizes[idx + 1:])) * num_bits)
    
    num_bytes = (num_bits * l - 1) // 8 + 1
    res = res.to_bytes(num_bytes, 'big')
    return res, l

def device_init(args):
    use_cuda = args.cuda and torch.cuda.is_available()
    use_mps = args.mps and torch.backends.mps.is_available()
    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device     
 
def recv_msg(idx,pipe,lock,recv_list,rec,participation_list):
    recv_list[idx] = 1
    msg = pipe.recv()
    participation_list[idx]+=1
    # lock.acquire()
    # print("Server receive: client{}".format(idx))
    # lock.release()
    rec[str(idx)] = msg
    recv_list[idx] = 0
    return

def recv_acc(idx,pipe,recv_list,rec):
    recv_list[idx] = 1
    msg = pipe.recv()
    rec[str(idx)] = msg
    recv_list[idx] = 0
    return

def test_epoch1(model, device, data_loaders):
    model.to(device)
    model.eval()
    correct = 0
    total_data = 0
    test_loss = 0
    with torch.no_grad():
        for data_loader in data_loaders:
            for data, target in data_loader:
                output = model(data.to(device))
                #test_loss += F.nll_loss(output, target.to(device), reduction='sum').item() # sum up batch loss
                test_loss += torch.nn.CrossEntropyLoss()(output, target.to(device))
                pred = output.max(1)[1] # get the index of the max log-probability
                correct += pred.eq(target.to(device)).sum().item()
            total_data += len(data_loader.dataset)
    test_loss /= total_data
    #print("loss",test_loss)
    # print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    #     test_loss, correct, len(data_loader.dataset),
    #     100. * correct / len(data_loader.dataset)))
    client_acc = correct / total_data
    # print(test_loss.item())
    # print(client_acc)
    test_loss = float(test_loss)
    return client_acc, test_loss

def cipher_size(cipher):
    ciphertext_size = 0
    for batch_cipher in cipher:
        compressed_ciphertext_bytes = pickle.dumps(batch_cipher)
        ciphertext_size += len(compressed_ciphertext_bytes)
    return ciphertext_size

def client_selection( mat, num_clients,train_weights,weights_clusters):      
    client_idxs,rep_num = lsh.clusters_selection_L2(np.array(mat), num_clients, train_weights,weights_clusters)
    return client_idxs,rep_num
    

def aggregatie_weights(rec,recv_list,weights_client,total_sum,batch_num,id_list,args,enc_tools = {},rep_num = []):
    weights = 0
    if args.enc:
        ciphertext_size = 0
        global_cipher = [0] *  batch_num
        if args.algorithm == 'paillier':
            global_cipher1 = [0] *  batch_num
            global_cipher2 = [0] *  batch_num
        if args.isSpars == 'topk':
            sum_mask = [0] * batch_num
    else:
        agg_res = np.zeros(total_sum)
    add_count = 0
    for idx,value in enumerate(rec.values()):
        c_id = value[0]
        if args.enc:
            if recv_list[c_id] != 0:
                continue
            if args.algorithm == 'ckks':
                ckks_file = os.path.join(args.data_dir + 'context_params')
                with open(ckks_file, "rb") as f:
                    params = f.read()
                ckks_ctx = ts.context_from(params) 
                frac = ts.ckks_vector(ckks_ctx,[weights_client[c_id]])
                if  args.isSpars == 'topk':     
                    mask = value[1]
                    cipher = value[2]
                    #print("id:",c_id,"mask:",mask)
                    if args.cipher_count:
                        ciphertext_size += cipher_size(cipher)
                    for batch in range(batch_num):       
                        res = 0
                        
                        if mask[batch]:
                            cnt = 0
                            for i in range(batch):
                                if mask[i]:
                                    cnt += 1
                            res = ts.CKKSVector.load(ckks_ctx,cipher[cnt]) * frac
                            sum_mask[batch] += weights_client[c_id]

                            if global_cipher[batch]:
                                res += ts.CKKSVector.load(ckks_ctx, global_cipher[batch]) 
                            global_cipher[batch] = res.serialize()  
                elif args.isSpars == 'randk' or args.isSpars == 'full':
                    cipher = value[1]
                    if args.cipher_count:
                        ciphertext_size += cipher_size(cipher)
                    for batch in range(len(cipher)):
                        add_cipher_batch = ts.CKKSVector.load(ckks_ctx, cipher[batch]) * frac
                        if global_cipher[batch]:
                            global_cipher_batch = ts.CKKSVector.load(ckks_ctx, global_cipher[batch]) 
                            add_cipher_batch += global_cipher_batch 
                        global_cipher[batch] = add_cipher_batch.serialize() 
                    weights += weights_client[c_id]
            elif args.algorithm == 'paillier':
                mod = enc_tools['mod']
                num_bits_per_batch = enc_tools['num_bits_per_batch']
                if args.isSpars == 'topk':
                    mask = value[1]
                    cipher = value[2]
                    if args.cipher_count:
                        compressed_ciphertext= compress_multi(np.array(cipher).flatten().astype(object), num_bits_per_batch)
                        ciphertext_size += cipher_size(compressed_ciphertext)
                    for batch in range(batch_num):
                        res = 0
                        if mask[batch]:
                            cnt = 0
                            for i in range(batch):
                                if mask[i]:
                                    cnt += 1
                            res = cipher[cnt]
                            sum_mask[batch] += weights_client[c_id]
                            for i in range(rep_num[idx]-1):
                                res = (res * cipher[cnt])%mod
                            if global_cipher[batch]:
                                global_cipher_batch = global_cipher[batch]
                                global_cipher[batch] = (global_cipher_batch * res) % mod
                            else:
                                global_cipher[batch] = res
                else:
                    cipher = value[1]
                    add_count += 1
                    # if args.algorithm == 'paillier' and add_count == 3:
                    #     global_cipher1 = global_cipher 
                    #     global_cipher = global_cipher2
                    if args.cipher_count:
                        compressed_ciphertext= compress_multi(np.array(cipher).flatten().astype(object), num_bits_per_batch)
                        ciphertext_size += cipher_size(compressed_ciphertext)

                    # test code
                    '''
                    cls_paillier = enc_tools['cls_paillier']
                    total_params = enc_tools['total_params']
                    global_weights = paillier_dec(cipher,cls_paillier,total_params,args)
                    print("server: id :",c_id,"global_weights[0]:",global_weights[0])
                    '''
                    for batch in range(len(cipher)):
                        add_cipher_batch = cipher[batch]
                        for i in range(rep_num[idx]-1):
                            add_cipher_batch = (add_cipher_batch * cipher[batch])%mod
                        if global_cipher[batch]:
                            add_cipher_batch = (add_cipher_batch * global_cipher[batch])%mod
                        global_cipher[batch] =  add_cipher_batch
                    weights += weights_client[c_id]
            elif args.algorithm == 'bfv':
                bfv_file = os.path.join(args.data_dir + 'bfv_ctx')
                with open(bfv_file, "rb") as f:
                    params = f.read()
                bfv_ctx = ts.context_from(params) 
                sk = bfv_ctx.secret_key()
                if  args.isSpars == 'topk':
                    mask = value[1]
                    cipher = value[2]
                    # tmp_list = bfv_dec(cipher,bfv_ctx,sk,args.isBatch,args.quan_bits,args.n_clients,batch_size = args.enc_batch_size)    
                    # print("id:",c_id,tmp_list[0],mask)
                    if args.cipher_count:
                        ciphertext_size += cipher_size(cipher)
                    for batch in range(batch_num):       
                        res = 0
                    
                        if mask[batch]:
                            cnt = 0
                            for i in range(batch):
                                if mask[i]:
                                    cnt += 1
                            res = ts.BFVVector.load(bfv_ctx,cipher[cnt])
                            sum_mask[batch] += weights_client[c_id]
 
                            if global_cipher[batch]:
                                res += ts.BFVVector.load(bfv_ctx, global_cipher[batch]) 
                            global_cipher[batch] = res.serialize()  

                else:
                    cipher = value[1]
                    #sk = bfv_ctx.secret_key()
                    # tmp_plain = bfv_dec(cipher,bfv_ctx,sk,args.isBatch,args.quan_bits,args.n_clients,batch_size = args.enc_batch_size)    
                    # print("server dec id:",c_id,tmp_plain[0])
                    if args.cipher_count:
                        ciphertext_size += cipher_size(cipher)
                    for batch in range(len(cipher)):
                        add_cipher_batch = ts.BFVVector.load(bfv_ctx, cipher[batch]) 
                        if global_cipher[batch]:
                            global_cipher_batch = ts.BFVVector.load(bfv_ctx, global_cipher[batch]) 
                            add_cipher_batch += global_cipher_batch 
                        global_cipher[batch] = add_cipher_batch.serialize()       
                    weights += weights_client[c_id]      
            else:
                raise ValueError("invalid enc algorithm",args.algorithm)
        else:
            value = value[1]
            if recv_list[c_id] == 0:
                add_params = np.array(value)*weights_client[c_id]
                weights += weights_client[c_id]
                agg_res += add_params
    if args.enc:
        if args.isSpars == 'topk':
            if args.cipher_count:
                logging('server receive: ciphertext size:{} bytes'.format(ciphertext_size),args)
                return sum_mask, ciphertext_size, global_cipher
            else:
                return sum_mask,global_cipher
        else:
            if args.cipher_count:
                logging('server receive: ciphertext size:{} bytes'.format(ciphertext_size),args)
                return weights, ciphertext_size, global_cipher
            else:
                return weights,global_cipher
    else:
        agg_res = agg_res.tolist()
        return weights,agg_res

def server_process(args,kwargs_IPC,total_sum,batch_num,train_weights,test_weights,server_test_sets,kwargs):
    n_clients = args.n_clients
    rec = {}
    acc_rec = {}
    n_epochs = args.epochs
    queues = kwargs_IPC['queues']
    acc_queue = kwargs_IPC['acc_queue']
    e = kwargs_IPC['e']
    lock = kwargs_IPC['lock']
    recv_list = [0 for i in range(n_clients)]
    recv_acc_list = [0 for i in range(n_clients)]
    pipe = kwargs_IPC['client_pipes']
    pipes = kwargs_IPC['send_pipes']
    send_pipes= [pipes[idx][1] for idx in range(n_clients)]
    server_pipes = [pipe[idx][1] for idx in range(n_clients)]
    flag = kwargs_IPC['flag']
    e_server = kwargs_IPC['e_server']
    if args.enc and args.algorithm == 'paillier':
        enc_tools = kwargs_IPC['enc_tools']
    else:
        enc_tools = {}
    rep_num = [1] * args.n_clients
    select_flag=False
    hash_queue = kwargs_IPC['hash_queue']
        
    participation_list = [0 for _ in range(n_clients)]
    accuracy_list = []
    loss_list = []
    total_ciphertext_size = 0
    cipher_size_list = []
    id_list = [range(n_clients)]
    weights_client = [weight for weight in train_weights]   
    time_list = []
    tmp_len_clusters = []
    # If it is plain text training, the server has a global model
    if args.enc == False:
        device = device_init(args)
        model = model_init(args.dataset,device)
        params_list,params_num,layer_shape = params_tolist(model)
    server_test_sets = torch.utils.data.DataLoader(server_test_sets, **kwargs)
    begin = time.time()
    
    for epoch in range(n_epochs):
        
        if epoch > 0 and epoch % 10 == 0:
            select_flag = True
        e.clear()
        
        threads = []
        for idx in range(n_clients):
            # If the previous listening thread ends or there is no listening thread
            if recv_list[idx] == 0:
                client_pipe = server_pipes[idx]
                thread = Thread(target=recv_msg,args = (idx,client_pipe,lock,recv_list,rec,participation_list))
                threads.append(thread)
                thread.start()
        for thread in threads:
            thread.join(timeout=3)

        if args.isSelection and epoch > 0:
            wait_bound = len(client_selected)
        else:
            wait_bound = n_clients
        wait_time = n_clients
        for i in range(args.n_clients):
            wait_time -= recv_list[i]
        while wait_time != wait_bound:
            time.sleep(1)
            wait_time = n_clients
            for i in range(args.n_clients):
                wait_time -= recv_list[i]


        # average weight
        if not args.weighted:
            weights_client = [1/n_clients for _ in range(n_clients)]
            train_weights =  [1/n_clients for _ in range(n_clients)]
 
        # Encryption weight aggregation
        if args.enc:
            if args.cipher_count:
                weights, *agg_res= aggregatie_weights(rec,recv_list,weights_client,
                                                total_sum,batch_num,id_list,args,enc_tools,rep_num)
                total_ciphertext_size += agg_res[0]
                cipher_size_list.append(agg_res[0])
                agg_res = agg_res[1]             
            else:
                weights, agg_res= aggregatie_weights(rec,recv_list,weights_client,
                                                total_sum,batch_num,id_list,args,enc_tools,rep_num)
        else:
            
            weights, agg_res= aggregatie_weights(rec,recv_list,weights_client,
                                              total_sum,batch_num,id_list,args)
            global_weights = (np.array(agg_res) / weights).tolist()  
            params_list,params_num,layer_shape = params_tolist(model)
            params_tomodel(model,global_weights,params_num,layer_shape,args,params_list)
        lock.acquire()
        logging('server agg: epoch {}.'.format(epoch),args)
        lock.release()

        # The aggregation is completed
        if epoch > 0 and args.isSelection:
            e_server.clear()

        # send to client 
        for queue in queues:
            if queue.empty() == False:
                a = queue.get()
            queue.put([weights,agg_res])
        
        # The aggregated content has been sent and can be read by the client
        e.set()

        if args.enc == True:
            acc_rec = {}
            threads = []
            for idx in range(n_clients):
                if recv_acc_list[idx] == 0:
                    client_pipe = send_pipes[idx]
                    thread = Thread(target=recv_acc,args = (idx,client_pipe,recv_acc_list,acc_rec))
                    threads.append(thread)
                    thread.start()    
            for thread in threads:
                thread.join(timeout = 3)

            # wait for client accuracy
            time.sleep(1)
            acc_epoch_list = []
            acc_weights = 0
            epoch_acc = 0
            epoch_loss = 0
            loss_epoch_list = []
            for idx,value in enumerate(acc_rec.values()):
                id_acc = value
                c_id = id_acc[0]
                acc = id_acc[1]
                loss = id_acc[2]

                # lock.acquire()
                # logging('client:{}, accuracy:{}%.'.format(c_id,acc),args)
                # lock.release()

                acc_weights += test_weights[c_id]
                loss_epoch_list.append(loss*test_weights[c_id])
                acc_epoch_list.append(acc*test_weights[c_id])

            # current epoch accuraacy
            epoch_acc = round(np.sum(np.array(acc_epoch_list)) / acc_weights,2)
            epoch_loss = round(np.sum(np.array(loss_epoch_list)) / acc_weights,2)

            # save each epoch accuracy 
            accuracy_list.append(epoch_acc)
            loss_list.append(epoch_loss)
            lock.acquire()
            logging("***********Server epoch {}, Clients accuracy:{}, loss:{}%***********\n".format(
                epoch,epoch_acc,epoch_loss),args)
            lock.release()
        else:
            server_acc,server_loss = test_epoch(model, device, server_test_sets)
            accuracy_list.append(server_acc)
            loss_list.append(server_loss)
            lock.acquire()
            logging("***********Server epoch {}, Clients accuracy:{}%***********\n".format(epoch,server_acc),args)
            lock.release()
        end = time.time()
        time_cost = round(end-begin,2)
        print("time:{}s".format(time_cost))
        time_list.append(time_cost)

        if args.isSelection:
            time.sleep(1)
            # wait for client sketch
            weights_clusters = [weight for weight in train_weights]
            weights_client = [weight for weight in train_weights]
            hash_list = []
            id_list = []
            while not hash_queue.empty():
                id_hash = hash_queue.get()
                id_list.append(id_hash[0])
                hash_list.append(id_hash[1])

            hash_list = sorted(hash_list, key=lambda x: id_list[hash_list.index(x)])
            id_list = sorted(id_list)
            client_selected,rep_num = client_selection(np.array(hash_list),len(id_list),weights_client,weights_clusters)
            if args.isSelection:
                logging("Num:{} ,Next round Selected clients:{}".format(len(client_selected), client_selected),args)
                tmp_len_clusters.append(len(client_selected))
            weights_client = weights_clusters
            new_weights = []
            for i in client_selected:
                new_weights.append(weights_client[i])
            selected_file = os.path.join(args.data_dir, args.dataset + 'selected')
            with open(selected_file, "wb") as f:
                clients_bytes = pickle.dumps([client_selected,new_weights])
                f.write(clients_bytes)  
            # Set the flag bit to indicate that the client selection is completed
            e_server.set()
        
    logging('server end!',args)
    
    flag.value = True
    e.clear()
    e_server.clear()
    if args.enc and args.cipher_count:
        logging("Total ciphertext size: {} bytes, size list: {}.".format(total_ciphertext_size,cipher_size_list),args)
    logging("Accuracy list: {}%.".format(accuracy_list), args)
    logging("Loss list:{}".format(loss_list),args)
    logging("time list:{}s".format(time_list),args)
    logging("Participate list: {}.".format(participation_list), args)
    logging("tmp_len_clusters:{}".format(tmp_len_clusters),args)

    return
            
