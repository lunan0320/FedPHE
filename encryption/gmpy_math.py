import os
import random
import gmpy2

POWMOD_GMP_SIZE = pow(2, 64)

def powmod(a, b, c):
    """
    return int: (a ** b) % c
    """
    
    if a == 1:
        return 1
    
    if max(a, b, c) < POWMOD_GMP_SIZE:
        return pow(a, b, c)
    
    else:
        return int(gmpy2.powmod(a, b, c))


def invert(a, b):
    """return int: x, where a * x == 1 mod b
    """    
    x = int(gmpy2.invert(a, b))
   
    if x == 0:
        raise ZeroDivisionError('invert(a, b) no inverse exists')
    
    return x
   
   
def getprimeover(n):
    """return a random n-bit prime number
    """     
    r = gmpy2.mpz(random.SystemRandom().getrandbits(n))
    r = gmpy2.bit_set(r, n - 1)
    
    return int(gmpy2.next_prime(r))


def isqrt(n):
    """ return the integer square root of N """
    
    return int(gmpy2.isqrt(n))


