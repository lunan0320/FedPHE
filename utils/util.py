from datetime import datetime
import numpy as np
import os
import torch
import random
from models.model import CNN_cifar,LeNet_mnist,CNN_fmnist,resnet20
from sklearn.cluster import KMeans
from models.resnet50 import resnet50

def model_init(dataset,device):
    if dataset=='MNIST':
        model = LeNet_mnist().to(device)
    elif dataset == 'FashionMNIST':
        model = CNN_fmnist().to(device)
    elif dataset == 'CIFAR10':
        model = resnet20(in_channels=3,num_classes=10).to(device)
        #model = CNN_cifar().to(device)
    elif dataset == 'CIFAR100':
        model = resnet50(in_channels=3,num_classes=100).to(device)
    else:
        raise ValueError("Datset name is invalid, please input MNIST, FashionMNIST or CIFAR10")
    return model

def logging(str,args):
    log_file = open(os.path.join(args.log_dir, args.dataset + '.log'), "a+")
    print("{} | {}".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), str))
    print("{} | {}".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), str), file=log_file)


def init_prop(train_dataset,test_dataset,n_clients):
    """
    Initialize weights of aggregation according to the samples of clients.

    Args:
        train_dataset (`dict`):
            Training dataset.
        test_dataset (`dict`):
            Test dataset.    
        n_clients (`int`):
            Number of clients to participate.
    Returns:
        train_props (`list`):
            Training weight for each client.
        test_props (`list`):
            Test weight for each client.
    """
    client_n_samples_train = []
    client_n_samples_test = []
    for idx in range(n_clients):
        client_n_samples_train.append(len(train_dataset[idx]))
        client_n_samples_test.append(len(test_dataset[idx]))
    samples_sum_train = np.sum(client_n_samples_train)
    samples_sum_test = np.sum(client_n_samples_test)
    test_props = []
    train_props = []
    for idx in range(n_clients):
        train_props.append(client_n_samples_train[idx]/samples_sum_train)
        test_props.append(client_n_samples_test[idx]/samples_sum_test)
    return train_props,test_props

def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def pseudo_random(seed, batch_num, topk, round_t):
    random.seed(seed + round_t)  
    number_list = list(range(batch_num))
    topk = int(np.ceil(batch_num * topk))
    # 对数字列表进行洗牌
    random.shuffle(number_list)
    random_list = sorted(number_list[:topk])

    return random_list

def jaccard_similarity(x, y):
    intersection = np.intersect1d(x, y)
    union = np.union1d(x, y)
    return len(intersection) / len(union)

def jaccard_distance_matrix(matrix):
    n = matrix.shape[0]
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                dist_matrix[i][j] = 1 - jaccard_similarity(matrix[i], matrix[j])
    return dist_matrix

def jaccard_kmeans_clustering(matrix, k):
    distance_matrix = jaccard_distance_matrix(matrix)
    clusters = np.zeros(len(matrix))  # Initializing clusters array
    while True:
        kmeans = KMeans(n_clusters=k, random_state=42)
        clusters = kmeans.fit_predict(distance_matrix)
        cluster_indices = [np.where(clusters == i)[0] for i in range(k)]
        empty_clusters = [i for i, indices in enumerate(cluster_indices) if len(indices) == 0]
        
        if len(empty_clusters) == 0:
            break  # No empty clusters, exit loop

    return cluster_indices