from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
from torch.utils.data import ConcatDataset
import numpy as np
from utils.subset import CustomSubset
from utils.partition import CIFAR10Partitioner,MNISTPartitioner,FMNISTPartitioner,CIFAR100Partitioner 
import pickle
import os

''''
load and split dataset for clients 
'''
def load_dataset(args):
    # dataset load
    if args.dataset == 'MNIST':
        transform = transforms.Compose([ToTensor()])
        train_set = datasets.MNIST(root="./data", download=True, transform=transform, train=True)
        test_set = datasets.MNIST(root="./data", download=True, transform=transform, train=False)
    elif args.dataset == 'FashionMNIST':
        transform = transforms.Compose([ToTensor()])
        train_set = datasets.FashionMNIST(root="./data", download=True, transform=transform, train=True)
        test_set = datasets.FashionMNIST(root="./data", download=True, transform=transform, train=False)        
    elif args.dataset == 'CIFAR10':
        transform = transforms.Compose([ToTensor()])
        train_set = datasets.CIFAR10(root="./data", download=True, transform=transform, train=True)
        test_set = datasets.CIFAR10(root="./data", download=True, transform=transform, train=False)
    elif args.dataset == "CIFAR100":
        transform = transforms.Compose([ToTensor()])
        train_set = datasets.CIFAR100(root="./data", download=True, transform=transform, train=True)
        test_set = datasets.CIFAR100(root="./data", download=True, transform=transform, train=False)  
    else:
        raise ValueError("Please input the correct dataset name, it must be one of:"
                        "MNIST, FashionMNST, CIFAR10, CIFAR100")
    
    # data info
    dataset_info = {}
    dataset_info["classes"] = train_set.classes
    dataset_info["num_classes"] = len(train_set.classes)
    dataset_info["input_size"] = train_set.data[0].shape[0]

    if len(train_set.data[0].shape) == 2:
        dataset_info["num_channels"] = 1
    else:
        dataset_info["num_channels"] = train_set.data[0].shape[-1]

    client_train_idx, client_test_idx = [], []

    # labels for train/test dataset
    train_labels = np.array(train_set.targets)
    test_labels = np.array(test_set.targets)

    # data split method
    if args.split == 'noniid':
        if args.noniid_method == 'pathological':
            if args.dataset == "MNIST":
                train_part = MNISTPartitioner(train_labels,
                                            args.n_clients,
                                            partition="noniid-#label",
                                            major_classes_num=args.n_shards,
                                            seed=args.seed)      
                test_part = MNISTPartitioner(test_labels, 
                                            args.n_clients,
                                            partition="iid",
                                            seed=args.seed)             
            elif args.dataset == "FashionMNIST":
                train_part = FMNISTPartitioner(train_labels,
                                            args.n_clients,
                                            partition="noniid-#label",
                                            major_classes_num=args.n_shards,
                                            seed=args.seed)      
                test_part = FMNISTPartitioner(test_labels, 
                                            args.n_clients,
                                            partition="iid",
                                            seed=args.seed)
            elif args.dataset == 'CIFAR10':
                train_part  = CIFAR10Partitioner(train_labels,
                                args.n_clients,
                                balance=None,
                                partition="shards",
                                num_shards=args.n_shards,
                                seed=args.seed)
                test_part = CIFAR10Partitioner(test_labels,
                                args.n_clients,
                                balance=True,
                                partition="iid",
                                seed=args.seed)
            elif args.dataset == 'CIFAR100':
                train_part = CIFAR100Partitioner(train_labels,
                                args.n_clients,
                                balance=None, 
                                partition="shards",
                                num_shards=args.n_shards,
                                seed=args.seed)
                test_part = CIFAR100Partitioner(test_labels, 
                                        args.n_clients,
                                        balance=True, 
                                        partition="iid",
                                        seed=args.seed) 
            else:
                raise ValueError("Please input the correct dataset name, it must be one of:"
                                 "MNIST, FashionMNIST, CIFAR10, CIFAR100")

        elif args.noniid_method == 'dirichlet':
            if args.dataset == "MNIST":
                train_part = MNISTPartitioner(train_labels,
                                            args.n_clients,
                                            # partition="noniid-labeldir",
                                            partition="unbalance",
                                            dir_alpha=args.alpha,
                                            seed=args.seed)
                test_part = MNISTPartitioner(test_labels,
                                            args.n_clients,
                                            partition="iid",
                                            seed=args.seed)
            elif args.dataset == "FashionMNIST":
                train_part = FMNISTPartitioner(train_labels,
                                            args.n_clients,
                                            #partition="noniid-labeldir",
                                            partition="unbalance",
                                            dir_alpha=args.alpha,
                                            seed=args.seed)
                test_part = FMNISTPartitioner(test_labels, 
                                            args.n_clients,
                                            partition="iid",
                                            seed=args.seed)
            elif args.dataset == 'CIFAR10':
                train_part  = CIFAR10Partitioner(train_labels,
                                args.n_clients,
                                balance=False,
                                partition="dirichlet",
                                unbalance_sgm=args.sgm,
                                dir_alpha=args.alpha,
                                seed=args.seed)
                test_part = CIFAR10Partitioner(test_labels,
                                args.n_clients,
                                balance=True,
                                partition="iid",
                                unbalance_sgm=args.sgm,
                                seed=args.seed)
            elif args.dataset == 'CIFAR100':
                train_part = CIFAR100Partitioner(train_labels, 
                                        args.n_clients,
                                        balance=False, 
                                        unbalance_sgm=args.sgm,
                                        partition="dirichlet",
                                        dir_alpha=args.alpha,
                                        seed=args.seed)
                test_part = CIFAR100Partitioner(test_labels, 
                                        args.n_clients,
                                        balance=True, 
                                        partition="iid",
                                        seed=args.seed)
            else:
                raise ValueError("Please input the correct dataset name, it must be one of:"
                                 "MNIST, FashionMNIST, CIFAR10")
        else:
            raise ValueError("Please input the correct noniid method, it must be one of:"
                            "pathological, dirichlet")
    elif args.split == 'iid':
        if args.dataset == "MNIST":
            train_part = MNISTPartitioner(train_labels,
                                        args.n_clients,
                                        partition="iid",
                                        seed=args.seed)     
            test_part = MNISTPartitioner(test_labels, 
                                        args.n_clients,
                                        partition="iid",
                                        seed=args.seed)             
        elif args.dataset == "FashionMNIST":
            train_part = FMNISTPartitioner(train_labels,
                                        args.n_clients,
                                        partition="iid",
                                        seed=args.seed) 
            test_part = FMNISTPartitioner(test_labels, 
                                        args.n_clients,
                                        partition="iid",
                                        seed=args.seed)
        elif args.dataset == 'CIFAR10':
            train_part = CIFAR10Partitioner(train_labels,
                            args.n_clients,
                            balance=True,
                            partition="iid",
                            seed=args.seed)
            test_part = CIFAR10Partitioner(test_labels,
                            args.n_clients,
                            balance=True,
                            partition="iid",
                            seed=args.seed)
        elif args.dataset == 'CIFAR100':
                train_part = CIFAR100Partitioner(train_labels,
                                args.n_clients,
                                balance=True, 
                                partition="iid",
                                seed=args.seed)
                test_part = CIFAR100Partitioner(test_labels, 
                                        args.n_clients,
                                        balance=True, 
                                        partition="iid",
                                        seed=args.seed)        
        else:
            raise ValueError("Please input the correct dataset name, it must be one of:"
                                 "MNIST, FashionMNIST, CIFAR10, CIFAR100")
    else:
        raise ValueError("Please input the correct split method, it must be one of:"
                         "iid, noniid")
    
    # index to value
    for value in train_part.client_dict.values():
        client_train_idx.append(value)
    for value in test_part.client_dict.values():
        client_test_idx.append(value)

    # subset of the original train/test dataset for each client
    client_train_sets = [CustomSubset(train_set,idx) for idx in client_train_idx]
    client_test_sets = [CustomSubset(test_set, idx) for idx in client_test_idx]

    # save the load_data results
    train_file = os.path.join(args.data_dir, args.dataset + '_train')
    test_file = os.path.join(args.data_dir, args.dataset + '_test')
    with open(train_file, "wb") as f:
        train_bytes = pickle.dumps(client_train_idx)
        f.write(train_bytes)
    with open(test_file, "wb") as f:
        test_bytes = pickle.dumps(client_test_idx)
        f.write(test_bytes)  

    # server dataset for fedavg
    server_test_idxs = []
    for i in range(len(client_test_idx)):
        server_test_idxs += client_test_sets[i]
    server_test_sets = server_test_idxs
    return client_train_sets, client_test_sets, dataset_info, server_test_sets


def load_exist(args):
    '''
    load existing dataset for clients
    '''
    if args.dataset == 'MNIST':
        transform = transforms.Compose([ToTensor()])
        train_set = datasets.MNIST(root="./data", download=True, transform=transform, train=True)
        test_set = datasets.MNIST(root="./data", download=True, transform=transform, train=False)
    elif args.dataset == 'FashionMNIST':
        transform = transforms.Compose([ToTensor()])
        train_set = datasets.FashionMNIST(root="./data", download=True, transform=transform, train=True)
        test_set = datasets.FashionMNIST(root="./data", download=True, transform=transform, train=False)        
    elif args.dataset == 'CIFAR10':
        transform = transforms.Compose([ToTensor()])
        train_set = datasets.CIFAR10(root="./data", download=True, transform=transform, train=True)
        test_set = datasets.CIFAR10(root="./data", download=True, transform=transform, train=False)
    elif args.dataset == "CIFAR100":
        transform = transforms.Compose([ToTensor()])
        train_set = datasets.CIFAR100(root="./data", download=True, transform=transform, train=True)
        test_set = datasets.CIFAR100(root="./data", download=True, transform=transform, train=False)    
    else:
        raise ValueError("Please input the correct dataset name, it must be one of:"
                        "MNIST, FashionMNST, CIFAR10")
    
    # data info
    dataset_info = {}
    dataset_info["classes"] = train_set.classes
    dataset_info["num_classes"] = len(train_set.classes)
    dataset_info["input_size"] = train_set.data[0].shape[0]

    if len(train_set.data[0].shape) == 2:
        dataset_info["num_channels"] = 1
    else:
        dataset_info["num_channels"] = train_set.data[0].shape[-1]

    # read dataset 
    train_file = os.path.join(args.data_dir, args.dataset + '_train')
    test_file = os.path.join(args.data_dir, args.dataset + '_test')
    with open(train_file, "rb") as f:
        train_bytes = f.read()
        client_train_idx = pickle.loads(train_bytes)
    with open(test_file, "rb") as f:
        test_bytes = f.read()
        client_test_idx = pickle.loads(test_bytes)

    
    # subset of the original train/test dataset for each client
    client_train_sets = [CustomSubset(train_set,idx) for idx in client_train_idx]
    client_test_sets = [CustomSubset(test_set, idx) for idx in client_test_idx]

    # server dataset for fedavg
    server_test_idxs = []
    for i in range(len(client_test_idx)):
        server_test_idxs += client_test_sets[i]
    server_test_sets = server_test_idxs

    return client_train_sets, client_test_sets, dataset_info,server_test_sets