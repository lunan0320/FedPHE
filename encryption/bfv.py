import tenseal as ts
import numpy as np
from encryption.quantize import quantize, unquantize
import random


def bfv_enc(plain_list,bfv_ctx,args):
    isBatch=args.isBatch
    batch_size=args.enc_batch_size
    topk=args.topk
    is_spars = args.isSpars

    quan_bits = args.quan_bits 
    plain_quan = quantize(plain_list,quan_bits,args.n_clients).tolist()

    batch_num = int(np.ceil(len(plain_list) / batch_size))
    if isBatch:
        # padding
        if len(plain_list) %  batch_num != 0:
            padding_num = batch_num * batch_size - len(plain_list)
            plain_list.extend([0]*padding_num)
            plain_quan.extend([0]*padding_num)
        if is_spars == 'topk':
            topk = int(np.ceil(batch_num * topk))
            sign = np.sign(np.array(plain_list))
            tmp_list = (np.array(plain_list) * sign).tolist()
            plain_batchs = [tmp_list[i * batch_size : (i+1) * batch_size ]for i in range(batch_num)] 
            avg_list = [np.average(np.abs(batch)) for batch in plain_batchs]    
            max_avg_list = np.sort(avg_list)[::-1][:topk]
            mask_list = []   
            for i in range(len(max_avg_list)):
                mask_list.append(avg_list.index(max_avg_list[i]))
            mask_list.sort()
 
            res_mask = [0  for i in range(batch_num) ]
            for i in range(batch_num):
                if i in mask_list:
                    res_mask[i] = 1
            # batch for encryption
            plain_list = [plain_quan[mask_list[i] * batch_size : (mask_list[i] + 1) * batch_size] for i in range(len(mask_list))]

            cipher_list = []
            for i in range(len(mask_list)):
                cipher = ts.bfv_vector(bfv_ctx,plain_list[i])
                cipher_list.append(cipher.serialize())
            return cipher_list, res_mask
        else:
            cipher_list = []
            for i in range(batch_num):
                cipher = ts.bfv_vector(bfv_ctx, plain_quan[i * batch_size : (i + 1) * batch_size])
                cipher_list.append(cipher.serialize())
            return cipher_list
    else:
        cipher = [ts.bfv_vector(bfv_ctx, [i]).serialize() for i in plain_quan]
        return cipher
  

def bfv_dec(cipher_list,bfv_ctx,sk,isBatch,quan_bits,n_clients,sum_masks = [],batch_size = 0):

    if isBatch:
        plain_list = []
        for idx, cipher_serial in enumerate(cipher_list):
            if cipher_serial == 0:
                zero_pad = [0] * batch_size
                plain_list.extend(zero_pad)
            else:
                plain = ts.BFVVector.load(bfv_ctx, cipher_serial).decrypt(sk)    
                plain = unquantize(plain,quan_bits,n_clients)
                if sum_masks != []:
                    plain = np.array(plain)/sum_masks[idx]
                plain_list.extend(plain)   
            
        return np.array(plain_list)
    else:
        plains = [ts.BFVVector.load(bfv_ctx, i).decrypt() for i in cipher_list]
        plains = [plains.squeeze()]
        res = []
        for plain in plains:
            tmp = unquantize(plain,quan_bits,n_clients)
            res.append(tmp)
        return np.array(res)

 
  