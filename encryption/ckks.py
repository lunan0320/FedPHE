import tenseal as ts
import numpy as np
from encryption.encrypt import Encrypt
from functools import reduce
from utils.util import pseudo_random


class CKKSCipher(Encrypt):
    def __init__(self, poly_modulus_degree=8192,
                 coeff_mod_bit_sizes=None,
                 global_scale=2**40):
        super(CKKSCipher, self).__init__()

        self.context = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=8192, coeff_mod_bit_sizes=[60, 40, 40, 60])
        self.context.global_scale=2**40
        # ckks_sk=ckks_ctx.secret_key()
        # ckks_ctx.make_context_public()

        # self.poly_modulus_degree = poly_modulus_degree
        # if coeff_mod_bit_sizes:
        #     self.coeff_mod_bit_sizes = coeff_mod_bit_sizes
        # else:
        #     self.coeff_mod_bit_sizes = []  # should be this since we do no do any multiplication
        # self.global_scale = global_scale

        # self.context = ts.context(
        #     scheme=ts.SCHEME_TYPE.CKKS,
        #     poly_modulus_degree=self.poly_modulus_degree,
        #     coeff_mod_bit_sizes=self.coeff_mod_bit_sizes,
        #     encryption_type=ts.ENCRYPTION_TYPE.SYMMETRIC
        # )
        # self.context.generate_galois_keys()
        # self.context.global_scale = self.global_scale

    def from_bytes(self, arr):
        if isinstance(arr, list):
            ret = []
            for e in arr:
                ret.append(self.from_bytes(e))
            return ret
        else:
            c = ts.CKKSVector.load(self.context, arr)
            return c

    def to_bytes(self, arr):
        if isinstance(arr, list):
            ret = []
            for e in arr:
                ret.append(self.to_bytes(e))
            return ret
        else:
            return arr.serialize()

    def encrypt(self, value):
        batch_size = []
        # value should be a 1-d np.array
        cipher = ts.ckks_vector(self.context, value)
        cipher_serial = cipher.serialize()
        return cipher_serial

    def enc_batch(self,value):
        batch_size = 50
        batch_num = int(np.ceil(len(value) / batch_size))
        cipher_list = []
        for i in range(batch_num):
            cipher = ts.ckks_vector(self.context, value[i * batch_size : (i + 1) * batch_size])
            cipher_list.append(cipher.serialize())
        return cipher_list
    

    


    def sum(self, arr,idx_weights):
        loaded = [ts.CKKSVector.load(self.context, e)*idx_weights[arr.index(e)] for e in arr]
        res = reduce(lambda x, y: x + y, loaded)
        return res.serialize()

    def sum_batch(self,arr,idx_weights):
        batch_num = len(arr[0])
        res_list = []
        for batch in range(batch_num):
            res = 0
            for client_cipher in range(len(arr)):
                res += ts.CKKSVector.load(self.context, arr[client_cipher][batch]) * idx_weights[client_cipher]
            res_list.append(res.serialize())
        return res_list
    
    def decrypt(self, value):
        return np.array(ts.CKKSVector.load(self.context, value).decrypt())
    

    def encrypt_no_batch(self, value):
        return [ts.ckks_vector(self.context, [i]).serialize() for i in value]
    
    def encrypt_no_batch1(self, value):
        return [ts.ckks_vector(self.context, [i]) for i in value]
    
    def sum_no_batch(self, arr):
        l = len(arr[0])
        result = []
        for i in range(l):
            scalars = [ts.CKKSVector.load(self.context, e[i]) for e in arr]
            
            result.append(reduce(lambda x, y: x + y, scalars).serialize())
        return result
    
    def sum_no_batch1(self, cipher_lists,idx_weights):
        l = len(cipher_lists[0])
        result = []
        for i in range(l):
            scalars = [e[i]*idx_weights[cipher_lists.index(e)] for e in cipher_lists]
            result.append(reduce(lambda x, y: x + y, scalars))
        return result

    def dec_batch(self,serial_list):
        plain_list = []
        for cipher_serial in serial_list:
            plain = ts.CKKSVector.load(self.context, cipher_serial).decrypt()
            plain_list.extend(plain)
        
        return np.array(plain_list)

    def decrypt_no_batch(self, value):
        return np.array([ts.CKKSVector.load(self.context, i).decrypt() for i in value])
    
    def decrypt_no_batch1(self, value):
        return np.array([ts.CKKSVector(self.context, i).decrypt() for i in value])
    
    def set_context(self, bytes):
        context = ts.Context.load(bytes)
        self.context = context

    def get_context(self, save_secret_key=False):
        return self.context.serialize(save_secret_key=save_secret_key,
                                      save_galois_keys=False,
                                      save_relin_keys=False)


def ckks_enc(plain_list,ckks_ctx,isBatch,batch_size,topk,round,randk_seed, is_spars = 'topk'):
    batch_num = int(np.ceil(len(plain_list) / batch_size))
    if isBatch:
        # padding
        if len(plain_list) %  batch_num != 0:
            padding_num = batch_num * batch_size - len(plain_list)
            plain_list.extend([0]*padding_num)
        if is_spars == 'topk':
            topk = int(np.ceil(batch_num * topk))
            plain_batchs = [plain_list[i * batch_size : (i+1) * batch_size ]for i in range(batch_num)] 
            # L2 norm
            #avg_list = ((np.linalg.norm(plain_batchs,axis=1,keepdims=True)).flatten()).tolist()
            # avg
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
            plain_list = [plain_list[mask_list[i] * batch_size : (mask_list[i] + 1) * batch_size] for i in range(len(mask_list))]
            cipher_list = []
            for i in range(len(mask_list)):
                cipher = ts.ckks_vector(ckks_ctx,plain_list[i])
                cipher_list.append(cipher.serialize())
            return cipher_list, res_mask
        elif is_spars == 'randk':
            randk = topk
            plain_batchs = [plain_list[i * batch_size : (i+1) * batch_size ]for i in range(batch_num)] 
            randk_list = pseudo_random(randk_seed,batch_num,randk,round)
            # batch for encryption
            plain_list = [plain_list[randk_list[i] * batch_size : (randk_list[i] + 1) * batch_size] for i in range(len(randk_list))]
            cipher_list = []
            for i in range(len(randk_list)):
                cipher = ts.ckks_vector(ckks_ctx,plain_list[i])
                cipher_list.append(cipher.serialize())
            return cipher_list, randk_list
        else:
            cipher_list = []
            for i in range(batch_num):
                cipher = ts.ckks_vector(ckks_ctx, plain_list[i * batch_size : (i + 1) * batch_size])
                cipher_list.append(cipher.serialize())
            return cipher_list
    else:
        cipher = [ts.ckks_vector(ckks_ctx, [i]).serialize() for i in plain_list]
        return cipher
  

def ckks_dec(cipher_list,ckks_ctx,sk,isBatch,randk_list, sum_masks = [],batch_size = 0):
    if isBatch:
        # randk align
        if randk_list != []:
            tmp_list = [0] * len(cipher_list)
            for i,radnk_idx in enumerate(randk_list):
                tmp_list[radnk_idx] = cipher_list[i]
            cipher_list = tmp_list
        plain_list = []
        for idx, cipher_serial in enumerate(cipher_list):
            if cipher_serial == 0:
                zero_pad = [0] * batch_size
                plain_list.extend(zero_pad)
            else:
                plain = ts.CKKSVector.load(ckks_ctx, cipher_serial).decrypt(sk)    
                if sum_masks != []:
                    plain = np.array(plain)/sum_masks[idx]
                plain_list.extend(plain)   
        return np.array(plain_list)
    else:
        plains = [ts.CKKSVector.load(ckks_ctx, i).decrypt() for i in cipher_list]
        plains = [plains.squeeze()]

        return np.array(plains)

 
  