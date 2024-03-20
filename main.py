

import argparse
import os
import torch
import torch.multiprocessing as mp
from utils.util import logging
import random
import numpy as np
import tenseal as ts
from fed import run
from encryption.paillier import PaillierCipher


def arg_parse():
    parser = argparse.ArgumentParser()

    # dataset and parameters
    parser.add_argument('--dataset', type=str, default='MNIST',
                            help='datasets: MNIST, FashionMNIST, CIFAR10, CIFAR100')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--weighted',type=bool,default=True)
    parser.add_argument('--n_clients', type=int, default= 8, metavar='N',
                        help='how many training processes to use (default: 10)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='momentum (default: 0.9)')

    
    # data split 
    parser.add_argument('--n_shards', type=int, default=5,
                        help='number of shards')
    parser.add_argument('--alpha', type=float, default=1,
                        help='parameter of dirichlet')
    parser.add_argument('--sgm', type=float, default=0.3,
                        help='parameter of unbalance')
    parser.add_argument('--split', type=str, default='noniid',
                        help='split method: iid or non-iid')
    parser.add_argument('--noniid_method', type=str, default='dirichlet',
                        help='noniid method: pathological or dirichlet')  
    # modules 
    parser.add_argument('--enc',type=bool,default=True,
                        help='enc or not')
    parser.add_argument('--isSelection',type=bool,default=True, 
                        help='Client selection or not')
    parser.add_argument('--isSpars', type=str, default='topk',
                        help='sparsification method: topk or randk or topk')        
    # sparsification
    parser.add_argument('--topk',type=float,default=0.2,
                        help='sparfication fraction')
                  
    # encryption paramsy
    parser.add_argument('--isBatch',type=bool,default=True,
                        help='Batch HE or not')
    parser.add_argument('--cipher_count',type=bool,default=True,
                        help='ciphertext size')
    parser.add_argument('--algorithm',type=str,default='ckks',
                        help='HE algorithm: paillier,bfv, ckks')
    parser.add_argument('--quan_bits',type=int,default=16,
                        help='quantification bits')
    parser.add_argument('--enc_batch_size',type=int,default=4096,
                        help='Batch Encryption size') 

    # selection
    parser.add_argument('--sim_len',type=int,default=200,
                        help='lsh matrix width')
    
    # device and logdir                   
    parser.add_argument('--cuda', action='store_true', default=True,
                        help='enables CUDA training')
    parser.add_argument('--mps', action='store_true', default=True,
                            help='enables macOS GPU training')
    parser.add_argument("--log_dir", type=str,
                            default="log", help="directory of logs")
    parser.add_argument("--data_dir", type=str,
                            default="data_dir/", help="directory of logs")                        
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--randk_seed', type=int, default=12, help='random k packages seed')

    return parser.parse_args()


def seed_everything(seed,is_cuda):
    """
    Seed function for randomization

    Args:
        seed (`int`):
            The seed in the parameters.
        is_cuda (`bool`):
            Whether to enable CUDA training.
    Returns:
        None
    """ 
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    # initialize the gpu device id
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    if is_cuda:
        torch.cuda.manual_seed_all(is_cuda)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def init_logger(log_dir,dataset):
    """
    Remove the historical log file 

    Args:
        log_dir (`arg_parse` ):
            The directory to save log files.
    Returns:
        None
    """
    log_file = os.path.join(log_dir, dataset + '.log')
    if os.path.exists(log_file):
        os.remove(log_file)

def ckks_init(data_dir):
    """
    Initialize and write context in CKKS encryption mode.

    Args:
        data_dir (`str`):
            Directory for data to store.

    Returns:
        None.
    """  
    ckks_ctx = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=8192, coeff_mod_bit_sizes=[60, 40, 40, 60])
    ckks_ctx.global_scale=2**40  
    ckks_ctx.generate_galois_keys()
    params = ckks_ctx.serialize(save_secret_key=True)
    ckks_file = os.path.join(data_dir + 'context_params')
    with open(ckks_file, "wb") as f:
        f.write(params)

def bfv_init(data_dir):
    """
    Initialize and write context in BFV encryption mode.

    Args:
        data_dir (`str`):
            Directory for data to store.

    Returns:
        None.
    """
    bfv_ctx = ts.context(ts.SCHEME_TYPE.BFV, poly_modulus_degree=8192, plain_modulus=1032193)
    params = bfv_ctx.serialize(save_secret_key=True)
    ckks_file = os.path.join(data_dir + 'bfv_ctx')
    with open(ckks_file, "wb") as f:
        f.write(params)

def paillier_init():
    """
    Initialize context in paillier encryption mode.

    Args:
        None.

    Returns:
        enc_tools (`dict`):
            cls_paillier (`context`):
                PaillierCipher context.
            mod (`int`):
                mod size.
            num_bits_per_batch (`int`):
                bit length for one package.
    """
    enc_tools = {}
    cls_paillier = PaillierCipher()
    cls_paillier.generate_key(n_length=2048)
    mod = pow(cls_paillier.get_n(),2)

    enc_tools['cls_paillier'] = cls_paillier
    enc_tools['mod'] = mod
    enc_tools['num_bits_per_batch'] = (cls_paillier.get_n() ** 2).bit_length()
    return enc_tools


def IPC_init(n_clients):
    """
    IPC communication between processes.

    Args:
        n_clients (`int` ):
            The num of clients to participate.
    Returns:
        `dict`: The locks, pipes, queues, flag, event in multiprocessing communication.
                lock_print, queue_lock (`Lock`): 
                    Process lock for print and logging.
                flag (`Value`):
                    Record whether the iteration is terminated.
                e, e_server (`Event`):
                    Synchronization with clients.
                pipes, send_pipes (`Pipe`):
                    (Encrypted) gradients send to the server.
                queues, acc_queue, hash_queue, clients_queues (`Queue`):
                    Server send (encrypted) aggregated gradients to clients, 
                    clients send local accuracy to the server,
                    clients send hash value to the server,
                    server send clients selected in the next epoch.
    """ 
    lock_print = mp.Lock()
    queue_lock = mp.Lock()

    flag = mp.Value('b', False)

    e = mp.Event()
    e_server = mp.Event()

    pipes = [mp.Pipe() for _ in range(n_clients)]
    send_pipes = [mp.Pipe() for _ in range(n_clients)]

    queues = [mp.Queue(1) for _ in range(n_clients)]
    acc_queue = mp.Queue(n_clients)
    hash_queue = mp.Queue(n_clients)
    clients_queues = mp.Queue(n_clients)
    

    kwargs_IPC = {'lock':lock_print,'e':e,'client_pipes':pipes,'queues':queues,'flag':flag,'e_server':e_server,
               'acc_queue':acc_queue,'hash_queue':hash_queue,'queue_lock':queue_lock,'clients_queues':clients_queues
               ,'send_pipes':send_pipes,}
    return kwargs_IPC

def device_init(is_cuda,is_mps):
    """
    Determine which device to train on.

    Args:
        is_cuda (`bool`):
            Whether to enable CUDA training.
        is_mps (`bool`):
            Whether to enable mps training.
    Returns:
        None
    """
    use_cuda = is_cuda and torch.cuda.is_available()
    use_mps = is_mps and torch.backends.mps.is_available()
    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device     
               
def main():
    """
    Main function.

    """
    mp.set_start_method('spawn', force=True)
    args = arg_parse()

    seed_everything(args.seed, args.cuda)
   
    init_logger(args.log_dir,args.dataset)

    device = device_init(args.cuda,args.mps)

    kwargs_IPC = IPC_init(args.n_clients)

    if args.enc :
        if args.algorithm == 'ckks':
            ckks_init(args.data_dir)
        elif args.algorithm == 'paillier':
            enc_tools = paillier_init()
            kwargs_IPC.update({'enc_tools':enc_tools,})
        elif args.algorithm == 'bfv':
            bfv_init(args.data_dir)
        else:
            raise ValueError("invalid algorithm!")
    
    logging("Basic information: device {}, learning rate {}, num clients {}, epochs {},noniid_method {},\
            isEnc {},isBatch {},sparsification {}, client selection {},topk {}, enc_batch_size {}".format(
        device, args.lr,args.n_clients,args.epochs,args.noniid_method,args.enc,args.isBatch,args.isSpars,args.isSelection,args.topk,args.enc_batch_size),args)

    run(args,kwargs_IPC,device)


if __name__ == '__main__':
    main()