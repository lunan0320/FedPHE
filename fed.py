
import os
import torch
import torch.multiprocessing as mp
import numpy as np

import utils.min_hash as lsh
from utils.util import logging
from client import client_process
from server import server_process
from utils.dataset import load_dataset,load_exist
from utils.util import init_prop
from client import params_tolist
from models.model import LeNet_mnist,CNN_fmnist,resnet20,CNN_cifar
from models.resnet50 import resnet50

def model_init(dataset,device):
    """
    Model initialization.

    Args:
        dataset (`str`):
            Name of dataset.
        device (`str`):
            Training on GPU or MPS or CPU.

    Returns:
        model (`OrderDict`):
            Model for dataset.
    """
    if dataset == 'MNIST': 
        model = LeNet_mnist().to(device)
    elif dataset == 'FashionMNIST':
        model = CNN_fmnist().to(device)
        #model = resnet20(in_channels=1,num_classes=10).to(device) 
    elif dataset == 'CIFAR10':
        model = resnet20(in_channels=3,num_classes=10).to(device)
    elif dataset == 'CIFAR100':
        model = resnet50(in_channels=3,num_classes=100).to(device)
    else:
        raise ValueError("Datset name is invalid, please input MNIST, FashionMNIST, CIFAR10 or CIFAR100")
    return model


def run(args,kwargs_IPC,device):
    """
    Run fucntion to launch server and clients processes.

    Args:
        args (`arg_parse`):
            Hyper-parameters.
        kwargs_IPC (`dict`):
            Parameters for IPC communication.    
        device (`str`):
            Training on GPU or MPS or CPU.
    Returns:
        None.
    """
    train_file = os.path.join(args.data_dir, args.dataset + '_train')
    if not os.path.exists(train_file):
        client_train_datasets, client_test_datasets, data_info,server_test_sets = load_dataset(args)
        print("Generate new files!")
    else:
        client_train_datasets, client_test_datasets, data_info,server_test_sets = load_exist(args)
        print("Load last files!")

    train_weights,test_weights = init_prop(client_train_datasets,client_test_datasets, args.n_clients)

    logging("training weights: {}".format(train_weights),args)
    logging("testing weights:{}".format(test_weights),args)
    for idx in range(args.n_clients):
        logging('client{}, train samples {},test samples {}'.format(
                  idx,len(client_train_datasets[idx]),len(client_test_datasets[idx])),args)
    logging("data split finished!",args)

    kwargs = {'batch_size': args.batch_size,
              'shuffle': True,'drop_last':True}
    if args.cuda and torch.cuda.is_available():
        kwargs.update({'num_workers': 0,
                       'pin_memory': True,
                      })


    model = model_init(args.dataset,device)
    params_list,params_num,layer_shape = params_tolist(model)
    total_sum = sum(params_num.values())

    # enc_tool for paillier algorithm
    if args.enc and args.algorithm == 'paillier':
        enc_tools = kwargs_IPC['enc_tools']
        enc_tools.update({'total_params':total_sum})
        kwargs_IPC.update({'enc_tools':enc_tools})

    # number of batchs for processing
    batch_num = int(np.ceil(total_sum / args.enc_batch_size))
    if args.enc and args.isBatch:
        logging("Batch num:{}".format(batch_num),args)
    
    if args.isSelection:
        random_R = lsh.gen_random_R(input_len = total_sum, sim_len=args.sim_len)
        kwargs_IPC.update({'random_R':random_R,})


    processes = []
    for rank in range(args.n_clients+1):
        # for server
        if rank == 0:
            p = mp.Process(target=server_process,args=(args,kwargs_IPC,total_sum,batch_num,train_weights,test_weights,server_test_sets,kwargs))
        # for clients
        else:
            p = mp.Process(target=client_process, args=(rank-1, args, model, device,
                                           client_train_datasets[rank-1], client_test_datasets[rank-1], kwargs,kwargs_IPC,train_weights))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    

    logging('Final End',args)
