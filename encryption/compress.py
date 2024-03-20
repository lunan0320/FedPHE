import numpy as np
from multiprocessing import Pool
import sys

N_JOBS = 4


def chunks_idx(l, n):
    d, r = divmod(len(l), n)
    for i in range(n):
        si = (d+1)*(i if i < r else r) + d*(0 if i < r else i - r)
        yield si, si+(d+1 if i < r else d)

#######################################
def _compress(flatten_array, num_bits):
    res = 0
    l = len(flatten_array)
    for element in flatten_array:
        res <<= num_bits
        res += element

    return res, l

def compress_multi(flatten_array, num_bits):
    l = len(flatten_array)
    
    pool_inputs = []
    sizes = []
    pool = Pool(N_JOBS)
    
    for begin, end in chunks_idx(range(l), N_JOBS):
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