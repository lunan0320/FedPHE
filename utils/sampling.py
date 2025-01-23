import torch
import random
import numpy as np


# cosine similarity
def importance_get(g1,g2):
    flat_tensor1 = torch.from_numpy(np.array(g1))
    flat_tensor2 = torch.from_numpy(np.array(g2))
    cos_sim = torch.nn.functional.cosine_similarity(flat_tensor1, flat_tensor2, dim=-1)
    return cos_sim


# sort according to importance
def importance_sort(list_imp):
    dict_sort = sorted(enumerate(list_imp), key=lambda list_imp:list_imp[1])  # x[1]是因为在enumerate(a)中，a数值在第1位
    list_sort = sorted(list_imp)
    list_index = [x[0] for x in dict_sort]
    return list_sort,list_index


# dynamic sampling
def dynamic_B(list_imp,threshold,B):
    K = 0
    L = len(list_imp)
    for i in range(L):
        if list_imp[i] > threshold:
            K += 1
    B = max(B-1,L-K)
    return B,K


# prob for client sync
def get_p(list_imp,B,L):
    list_k = []
    list_p = [0] * L
    for i in range(len(list_imp)):
        for k in range(L,0,-1):
            tmp = sum(list_imp[:k])
            if B + k-1 - L <= list_imp[i]/tmp:
                list_k.append(k)
                list_p[i] = ((B+k-L)*list_imp[i]/tmp)
                break
                
    K = max(list_k)
    for i in range(L-1,K-1,-1):
        list_p[i] = 1
    return list_p


def client_sampling(list_p,global_index):
    list_index = []
    for i in range(len(list_p)):
        rand = random.random()
        if list_p[i] >= rand:
            list_index.append(global_index[i])
    return sorted(list_index)


def Adaptive_samping(global_importance,threshold,B):
    global_sort,global_index = importance_sort(global_importance)
    B,K = dynamic_B(global_importance,threshold,B)
    global_p = get_p(global_sort,B,len(global_sort))
    global_index = client_sampling(global_p,global_index)
    B_real_list.append(len(global_index))
    return global_index


def Adaptive_samping_bar(global_importance,threshold,B,B_real_list):
    global_sort,global_index = importance_sort(global_importance)
    B,K = dynamic_B(global_importance,threshold,B)
    global_p = get_p(global_sort,B,len(global_sort))
    global_index = client_sampling(global_p,global_index)
    B_real_list.append(len(global_index))

    return global_index,B,B_real_list
