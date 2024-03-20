
# true value trans to 2's complement
def true2two(value, padding_bits, quan_bits):
    if value >= 0:
        binary = format(value,'0{}b'.format(quan_bits))
        return int(binary.zfill(padding_bits + quan_bits),2)
    else:
        binary = format(abs(value),'0{}b'.format(quan_bits))
        inversed = ''.join('1' if bit == '0' else '0' for bit in binary)
        complement = int(bin(int(inversed,2)+1)[2:].zfill(padding_bits + quan_bits),2)
        return complement

# 2's complement trans to true value
def two2true(value, padding_bits, quan_bits):
    #value = int(value )
    mod = pow(2,quan_bits)
    #value = value % mod
    if value > mod:
        value %= mod
    value = bin(value)[2:].zfill(quan_bits)

    if value[0] == '0':
        return int(value,2)
    elif value[0] == '1':
        inversed = ''.join('1' if bit == '0' else '0' for bit in value)
        value = int(inversed,2) + 1
        if value >= pow(2, quan_bits) - 1:
            value -= pow(2, quan_bits)
        return (-1*value)
    else:
        raise ValueError("Overflow {}".format(value))