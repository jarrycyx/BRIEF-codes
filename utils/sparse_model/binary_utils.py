import struct
import numpy as np

def int2bitsarray(n, bitnum=8):
    if n < 2**bitnum:
        binstr = bin(n)[2:]
        bin_array = [bit=="1" for bit in binstr]
        prefix = [False for _ in range(bitnum - len(binstr))]
        prefix.extend(bin_array)
        bin_array = prefix
    else:
        raise Exception("Value Exceeds Limit!")

    return bin_array

def bitsarray2int(arr):
    bitstr = [str(int(bit)) for bit in arr]
    return int("".join(bitstr), 2)

def float2bitsarray(F):
    bit_str = float2binstr(F)
    return [bit=='1' for bit in bit_str]

def bitsarray2float(B):
    bit_str = ""
    for bit in B:
        bit_str += str(int(bit))
    return binstr2float(bit_str)

#转半精度，浮点到二进制
def float2binstr(F):  #F是浮点数
    if F.dtype == np.dtype("float32"):
        return '{:032b}'.format(struct.unpack('<L', np.float32(F).tobytes())[0])
    else:
        raise Exception("Not a Float!") 

#半精度转浮点，二进制到浮点
def binstr2float(B):  #B是二进制字符串
    if len(B) == 32:
        return np.frombuffer(struct.pack('<L',int(B,2)), dtype='<f4')[0]
    else:
        raise Exception("Cannot Convert to Float!") 


if __name__=="__main__":
    bin_str = float2binstr(np.array(0.0123).astype(np.float32))
    f = binstr2float(bin_str)
    print(bin_str)
    print(f)
    
    bit_arr = float2bitsarray(np.array(0.0123).astype(np.float32))
    f = bitsarray2float(bit_arr)
    print(bit_arr)
    print(f)
    
    print(int2bitsarray(12, 7))
    
    print(bitsarray2int([False, True, False]))