import random
import numpy as np
import warnings
from sklearn.cluster import KMeans
from sklearn.metrics import jaccard_score
from gap_statistic import OptimalK

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

from sklearn.metrics.pairwise import cosine_similarity
from utils.util import jaccard_kmeans_clustering

def sigGen(matrix,randomSeq):
    """
    * generate the signature vector
    :param matrix: a ndarray var
    :return a signature vector: a list var
    """
    # initialize the sig vector as [-1, -1, ..., -1]
    result = [-1 for i in range(matrix.shape[1])]

    count = 0

    for row in randomSeq:
        for i in range(matrix.shape[1]):
            if matrix[row][i] != 0 and result[i] == -1:
                result[i] = row
                count += 1
        if count == matrix.shape[1]:
            break
    return result

def sigMatrixGen(input_matrix,random_R, n):
    """
    generate the sig matrix
    :param input_matrix: naarray var
    :param n: the row number of sig matrix which we set
    :return sig matrix: ndarray var
    """

    result = []

    for i in range(n):
        sig = sigGen(input_matrix,random_R[i])
        result.append(sig)
    return np.array(result)

def quan_params(input_matrix, threshold):
    for idx, row in enumerate(input_matrix):
        for i in range(len(row)):
            if abs(row[i]) < threshold:
                input_matrix[idx][i] = 0
            else:
                input_matrix[idx][i] = 1
        
    return input_matrix
        
def real_sim(input_matrix):
    row_num = input_matrix.shape[0]
    total = 0
    sim = 0
    for row in range(row_num):
        if input_matrix[row][0] == 1 or input_matrix[row][1] == 1:
            total += 1
            if input_matrix[row][0] == input_matrix[row][1]:
                sim += 1
    return sim / total

def dim_reduce_sim(input_matrix):
    row_num = input_matrix.shape[0]
    sim = 0
    for row in range(row_num):
        if input_matrix[row][0] == input_matrix[row][1]:
            sim += 1
    return sim / row_num


def gen_random_R(input_len, sim_len):
    """
    Random matrix is needed for sketch-based client selection.

    Args:
        input_len (`int`):
            Input dimension.
        sim_len (`int`):
            Output dimension.  
    Returns:
        random_R (`list`):
            Random matrix.
    """

    random_R = []
    for i in range(sim_len):
        seq_list = np.arange(input_len)
        np.random.shuffle(seq_list)
        random_R.append(seq_list)
    return random_R

'''
def gap_statistic(data,max_k):
    optimalK = OptimalK(n_jobs=4, parallel_backend='joblib')
    n_clusters = optimalK(data, cluster_array=np.arange(1, max_k+1))
    return n_clusters
'''

def sse_statistic(X,max_k):
    sse = []
    for k in range(1, max_k+1):
        kmeans = KMeans(n_clusters=k).fit(X)
        sse.append(kmeans.inertia_)
    diff = np.diff(sse)
    diff_r = diff[1:] / diff[:-1]
    k_opt = np.argmax(diff_r) + 2
    return k_opt

def gap_statistic(data,max_k):
    data = np.array(data).reshape(-1,1)
    optimalK = OptimalK(n_jobs=20, parallel_backend='joblib')
    n_clusters = optimalK(data, cluster_array=np.arange(1, max_k+1))
    return n_clusters

def clusters_selection_L2(hash_mat,max_k,train_weights=[],weights_clusters=[]):
    k_opt = gap_statistic(hash_mat.astype(float),max_k)
    clusters = jaccard_kmeans_clustering(hash_mat, k_opt)
    tmp = hash_mat.tolist()
    rep_num = [1] * len(train_weights)
    selected_clients = []
    print("Num:{} ,Next round Selected clients:{}".format(k_opt, clusters))

    for i, indices in enumerate(clusters):
        tmp_weights = [train_weights[i] for i in indices]
        tmp_list = list(indices)
        zipped = zip(tmp_weights, tmp_list)
        max_tuple = max(zipped, key=lambda x: x[0])
        client_index = max_tuple[1]  
        if train_weights != []:
            rep_num[client_index] = len(indices)
            tmp = 0
            for idx in indices:
                tmp += train_weights[idx]
            weights_clusters[client_index] = tmp
        selected_clients.append(client_index)
    return selected_clients,rep_num  

def clusters_selection(hash_mat,max_k,train_weights=[],weights_clusters=[]):
    k_opt = gap_statistic(hash_mat.astype(float),max_k)
    while True:
        kmeans = KMeans(n_clusters=k_opt).fit(hash_mat)  
        cluster_counts = np.unique(kmeans.labels_)
        if len(cluster_counts) == k_opt:
            break
        labels = kmeans.labels_
    unique_labels = np.unique(labels)
    selected_clients = []

    rep_num = [1] * len(train_weights)
    for label in unique_labels:
        indices = np.where(labels == label)[0]  
        tmp_weights = [train_weights[i] for i in indices]
        tmp_list = list(indices)
        zipped = zip(tmp_weights, tmp_list)
        max_tuple = max(zipped, key=lambda x: x[0])
        client_index = max_tuple[1]  

        if train_weights != []:
            rep_num[client_index] = len(indices)
        
        if train_weights != []:
            tmp = 0
            for idx in indices:
                tmp += train_weights[idx]
            weights_clusters[client_index] = tmp
        selected_clients.append(client_index)

    return selected_clients,rep_num

def client_selection(sampled_clients,labels,train_weights=[],weights_clusters=[]):
    unique_labels = np.unique(labels)
    selected_clients = []
    rep_num = [1] * len(train_weights)
    for label in unique_labels:
        indices = np.where(labels == label)[0]  # 找到属于当前标签的数据点索引
        retA = [i for i in indices if i in sampled_clients]
        if retA == []:
            continue
        else:
            client_index = np.random.choice(np.array(retA))

        if train_weights != []:
            rep_num[client_index] = len(indices)
        
        if train_weights != []:
            tmp = 0
            for idx in retA:
                tmp += train_weights[idx]
            weights_clusters[client_index] = tmp
        selected_clients.append(client_index)
    return selected_clients,rep_num
