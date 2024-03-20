import numpy as np
import random
import time
from encryption import gmpy_math

from multiprocessing import cpu_count, Pool
from encryption.encrypt import Encrypt

from encryption.quantize import quantize, unquantize, batch_padding, unbatching_padding



N_JOBS = cpu_count()


def _static_encrypt(value, n):
    pubkey = PaillierPublicKey(n)
    return pubkey.encrypt(value)


def _static_decrypt(value, n, p, q):
    pubkey = PaillierPublicKey(n)
    prikey = PaillierPrivateKey(pubkey, p, q)
    return prikey.decrypt(value)


class PaillierPublicKey(object):
    def __init__(self, n):
        self.g = n + 1
        self.n = n
        self.nsquare = n * n
        self.max_int = n // 3 - 1

    def get_n(self):
        return self.n

    def __repr__(self):
        hashcode = hex(hash(self))[2:]
        return "<PaillierPublicKey {}>".format(hashcode[:10])

    def __eq__(self, other):
        return self.n == other.n

    def __hash__(self):
        return hash(self.n)

    def apply_obfuscator(self, ciphertext):
        r = random.SystemRandom().randrange(1, self.n)
        obfuscator = gmpy_math.powmod(r, self.n, self.nsquare)

        return (ciphertext * obfuscator) % self.nsquare

    def encrypt(self, plaintext):
        if plaintext >= (self.n - self.max_int) and plaintext < self.n:
            # Very large plaintext, take a sneaky shortcut using inverses
            neg_plaintext = self.n - plaintext  # = abs(plaintext - nsquare)
            neg_ciphertext = (self.n * neg_plaintext + 1) % self.nsquare
            ciphertext = gmpy_math.invert(neg_ciphertext, self.nsquare)
        else:
            ciphertext = (self.n * plaintext + 1) % self.nsquare

        ciphertext = self.apply_obfuscator(ciphertext)
        return ciphertext
    

class PaillierPrivateKey(object):
    def __init__(self, public_key, p, q):
        if not p * q == public_key.n:
            raise ValueError("given public key does not match the given p and q")
        if p == q:
            raise ValueError("p and q have to be different")
        self.public_key = public_key
        if q < p:
            self.p = q
            self.q = p
        else:
            self.p = p
            self.q = q
        self.psquare = self.p * self.p
        self.qsquare = self.q * self.q
        self.q_inverse = gmpy_math.invert(self.q, self.p)
        self.hp = self.h_func(self.p, self.psquare)
        self.hq = self.h_func(self.q, self.qsquare)

    def __eq__(self, other):
        return self.p == other.p and self.q == other.q

    def __hash__(self):
        return hash((self.p, self.q))

    def __repr__(self):
        hashcode = hex(hash(self))[2:]

        return "<PaillierPrivateKey {}>".format(hashcode[:10])

    def h_func(self, x, xsquare):
        return gmpy_math.invert(self.l_func(gmpy_math.powmod(self.public_key.g,
                                                                 x - 1, xsquare), x), x)

    def l_func(self, x, p):
        return (x - 1) // p

    def crt(self, mp, mq):
        u = (mp - mq) * self.q_inverse % self.p
        x = (mq + (u * self.q)) % self.public_key.n

        return x

    def decrypt(self, ciphertext):
        mp = self.l_func(gmpy_math.powmod(ciphertext,
                                              self.p-1, self.psquare),
                                              self.p) * self.hp % self.p
        mq = self.l_func(gmpy_math.powmod(ciphertext,
                                              self.q-1, self.qsquare),
                                              self.q) * self.hq % self.q

        plaintext = self.crt(mp, mq)

        return plaintext

    def get_p_q(self):
        return self.p, self.q
    
    
class PaillierKeypair(object):
    def __init__(self):
        pass

    @staticmethod
    def generate_keypair(n_length=1024):
        p = q = n = None
        n_len = 0

        while n_len != n_length:
            p = gmpy_math.getprimeover(n_length // 2)
            q = p
            while q == p:
                q = gmpy_math.getprimeover(n_length // 2)
            n = p * q
            n_len = n.bit_length()

        public_key = PaillierPublicKey(n)
        private_key = PaillierPrivateKey(public_key, p, q)

        return public_key, private_key
    
    
class PaillierCipher(Encrypt):
    def __init__(self):
        super(PaillierCipher, self).__init__()
        self.uuid = None
        self.exchanged_keys = None
        self.n = None
        self.key_length = None

    def set_n(self, n):  # for all (arbiter is necessary, while host and guest is optional since they dont add)
        self.n = n

    def get_n(self):
        return self.n

    def set_self_uuid(self, uuid):
        self.uuid = uuid

    def set_exchanged_keys(self, keys):
        self.exchanged_keys = keys

    def generate_key(self, n_length=2048):
        self.key_length = n_length
        self.public_key, self.privacy_key = \
            PaillierKeypair.generate_keypair(n_length=n_length)
        self.set_n(self.public_key.n)

    def get_key_pair(self):
        return self.public_key, self.privacy_key

    def set_public_key(self, public_key):
        self.public_key = public_key
        # for host
        self.set_n(public_key.n)

    def get_public_key(self):
        return self.public_key

    def set_privacy_key(self, privacy_key):
        self.privacy_key = privacy_key

    def get_privacy_key(self):
        return self.privacy_key

    def _dynamic_encrypt(self, value):
        return self.public_key.encrypt(value)

    def _multiprocessing_encrypt(self, value):
        shape = value.shape
        value_flatten = value.flatten()
        n = self.public_key.get_n()

        pool_inputs = []
        for i in range(len(value_flatten)):
            pool_inputs.append([value_flatten[i], n])

        pool = Pool(N_JOBS)
        ret = pool.starmap(_static_encrypt, pool_inputs)
        pool.close()
        pool.join()

        ret = np.array(ret)
        return ret.reshape(shape)

    def encrypt(self, value):
        if self.public_key is not None:
            if not isinstance(value, np.ndarray):
                return self._dynamic_encrypt(value)
            else:
                return self._multiprocessing_encrypt(value)
        else:
            return None

    def _dynamic_decrypt(self, value):
        return self.privacy_key.decrypt(value)

    def _multiprocessing_decrypt(self, value):
        shape = value.shape
        value_flatten = value.flatten()
        n = self.public_key.get_n()
        p, q = self.privacy_key.get_p_q()

        pool_inputs = []
        for i in range(len(value_flatten)):
            pool_inputs.append(
                [value_flatten[i], n, p, q]
            )

        pool = Pool(N_JOBS)
        ret = pool.starmap(_static_decrypt, pool_inputs)
        pool.close()
        pool.join()

        ret = np.array(ret)
        return ret.reshape(shape)

    def decrypt(self, value):
        if self.privacy_key is not None:
            if not isinstance(value, np.ndarray):
                return self._dynamic_decrypt(value)
            else:
                return self._multiprocessing_decrypt(value)
        else:
            return None


def paillier_enc(plain_list, cls_paillier,args):

    quan_bits = args.quan_bits
    isBatch = args.isBatch
    isSpars = args.isSpars
    num_clients = args.n_clients
    batch_size = args.enc_batch_size
    padding_bits = int(np.ceil(np.log2(num_clients + 1)))
    elem_bits = padding_bits + quan_bits

    quan_begin = time.time()

    quan_list = quantize(plain_list,quan_bits,num_clients)
    plain_shape = len(plain_list)
    quan_end = time.time()
    #print(f"*******************Client {id} Encrypt*******************")
    #print(f"Quantized time:{quan_end-quan_begin:.3f}s",end=" |  ")
    
    if isBatch:
        batch_num = int(np.ceil(len(plain_list) / batch_size))
        # if len(plain_list) %  batch_num != 0:
        #     padding_num = batch_num * batch_size - len(plain_list)
        #     plain_list.extend([0]*padding_num)
        batch_begin = time.time()
        quan_list = batch_padding(quan_list,cls_paillier.key_length,elem_bits,batch_size=batch_size)
        # unquan_list = unbatching_padding(quan_list,elem_bits,batch_size)
        batch_end = time.time()
        #print(f"Batching time:{batch_end-batch_begin:.3f}s",end=" |  ")
        if isSpars == 'topk':
            # 取最大的 topk 个 batch
            topk = int(np.ceil(batch_num * args.topk))
            sign = np.sign(np.array(plain_list))
            tmp_list = (np.array(plain_list) * sign).tolist()
            plain_batchs = [tmp_list[i * batch_size : (i+1) * batch_size ]for i in range(batch_num)] 
            avg_list = [np.average(batch) for batch in plain_batchs]    
            max_avg_list = np.sort(avg_list)[::-1][:topk]
            tmp_list = []
            mask_list = [0] * batch_num
            for i in range(len(max_avg_list)):
                tmp_list.append(avg_list.index(max_avg_list[i]))

            tmp_list = np.sort(tmp_list)
            for i in tmp_list:
                mask_list[i] = 1
            
            quan_list = [quan_list[tmp_list[i]] for i in range(len(tmp_list))]

    # paillier encrypt
    enc_begin = time.time()
    cipher_list = []
    for longint in quan_list:
        tmp_cipher = cls_paillier.encrypt(longint)
        cipher_list.append(tmp_cipher)
    enc_end = time.time()
    #print(f"Encrypt time:{enc_end-enc_begin:.3f}s")

    if isSpars == 'topk':
        return cipher_list,mask_list
    
    return cipher_list

def paillier_dec(cipher_list, cls_paillier,plain_shape,args):
    quan_bits = args.quan_bits
    num_clients = args.n_clients
    isBatch = args.isBatch
    padding_bits = int(np.ceil(np.log2(num_clients + 1)))
    elem_bits = quan_bits + padding_bits
    # paillier decrypt
    dec_begin = time.time()
    decrypted_m = []
    batch_size = args.enc_batch_size
    for cipher in cipher_list:
        if cipher == 0:
            dec_plain = 0
            decrypted_m.append(dec_plain)
        else:
            dec_plain = cls_paillier.decrypt(cipher)
            decrypted_m.append(dec_plain)
    quantized_sum = decrypted_m
    dec_end = time.time()
    #print(f"*******************Client {id} Decrypt*******************")
    #print(f"Decrypt time:{dec_end-dec_begin:.3f}s",end=" |  ")
            
    if isBatch:
        # unbatching
        unbatch_begin = time.time()
        unbatch_plaintext = unbatching_padding(decrypted_m, elem_bits,args.enc_batch_size)[:(int(np.prod(plain_shape)))]
        quantized_sum = unbatch_plaintext.reshape(plain_shape)
        unbatch_end = time.time()
        #print(f"Unbatching time:{unbatch_end - unbatch_begin:.3f}s",end=" |  ")
            
    # unquantized
    unquan_begin = time.time()
    unquan_plaintext = unquantize(quantized_sum,quan_bits,num_clients)
    unquan_end = time.time()
    #print("Quan agg:",unquan_plaintext)
    #print(f"Unquantized time:{unquan_end-unquan_begin:.3f}s")
    return unquan_plaintext