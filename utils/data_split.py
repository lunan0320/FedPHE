import numpy as np


def noniid_split_dirichlet(train_labels, alpha, n_clients):
    """
    Splits a list of data indices with corresponding labels
    into subsets according to a dirichlet distribution with parameter
    alpha
    Args:
        train_labels: ndarray of train_labels
        alpha: the parameter of dirichlet distribution
        n_clients: number of clients
    Returns:
        client_idcs: a list containing sample idcs of clients
    """

    n_classes = train_labels.max()+1
    label_distribution = np.random.dirichlet([alpha]*n_clients, n_classes)

    class_idcs = [np.argwhere(train_labels==y).flatten() 
           for y in range(n_classes)]

    
    client_idcs = [[] for _ in range(n_clients)]
    for c, fracs in zip(class_idcs, label_distribution):
        for i, idcs in enumerate(np.split(c, (np.cumsum(fracs)[:-1]*len(c)).astype(int))):
            client_idcs[i] += [idcs]

    client_idcs = [np.concatenate(idcs) for idcs in client_idcs]
  
    return client_idcs

def noniid_split_pathological(train_labels, n_classes_per_client, n_clients):
    n_classes = train_labels.max()+1
    data_idcs = list(range(len(train_labels)))
    label2index = {k: [] for k in range(n_classes)}
    for idx in data_idcs:
        label = train_labels[idx]
        label2index[label].append(idx)

    sorted_idcs = []
    for label in label2index:
        sorted_idcs += label2index[label]

    def iid_divide(l, g):
            num_elems = len(l)
            group_size = int(len(l) / g)
            num_big_groups = num_elems - g * group_size
            num_small_groups = g - num_big_groups
            glist = []
            for i in range(num_small_groups):
                glist.append(l[group_size * i: group_size * (i + 1)])
            bi = group_size * num_small_groups
            group_size += 1
            for i in range(num_big_groups):
                glist.append(l[bi + group_size * i:bi + group_size * (i + 1)])
            return glist


    n_shards = n_clients * n_classes_per_client

    shards = iid_divide(sorted_idcs, n_shards)
    np.random.shuffle(shards)

    tasks_shards = iid_divide(shards, n_clients)

    clients_idcs = [[] for _ in range(n_clients)]
    for client_id in range(n_clients):
        for shard in tasks_shards[client_id]:

            clients_idcs[client_id] += shard 

    return clients_idcs