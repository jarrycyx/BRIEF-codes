import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(os.path.dirname(__file__))

import torch
import numpy as np
import matplotlib.pyplot as plt
from binary_utils import *
import copy


from os.path import splitext as ops
from os.path import basename as opb
from os.path import dirname as opd
from os.path import join as opj


def calc_stat(module_dict: dict, log_res=False):
    zero_num = 0
    para_num = 0

    for _, value in module_dict.items():
        shape = value.shape
        zero_num += torch.sum(value==0)
        para_num += torch.prod(torch.tensor(shape))
    
    if log_res:
        print("Parameter Number: %d, Zeros Number: %d, Sparsity: %.2f" % (para_num, zero_num, zero_num / para_num))
        print("Parameter dtype: %s" % str(value.dtype))
        
    return para_num, zero_num, zero_num / para_num

def get_pure_array(module_dict: dict):
    number_array = np.array([], dtype=np.float32)
    for _, value in module_dict.items():
        number_array = np.concatenate([number_array, np.array(value).flatten()], axis=0)
    
    # np.save("D:/proj-files-level2/data-compression/NeuralFiledsDataCompression/data/my_mc_test/number_arr", number_array)
    return number_array

def extract_zeros(arr):
    extracted = []
    zero_neighbor = []
    
    zero_num = 0
    for i in range(arr.shape[0]):
        v = arr[i]
        if v == 0:
            zero_num += 1
        else:
            if zero_num > 0:
                extracted.append((0, zero_num))
                zero_neighbor.append(zero_num)
            extracted.append((1, v))
            zero_num = 0
            
    return extracted, zero_neighbor


def esti_storage_size(extracted, zero_block):
    s = 0
    for flag, v in extracted:
        if flag:
            s += 32 + 1
        else:
            s += (zero_block + 1) * np.ceil(v / 2**zero_block)
    return s / 8


def generate_byte_file(extracted, zero_block=3):
    file_bits = []
    for flag, v in extracted:
        if flag:
            bit_array = float2bitsarray(v)
            file_bits.append(True)
            file_bits.extend(bit_array)
        else:
            zero_num = v
            while zero_num > 0:
                zero_num_this_block = min(2**zero_block, zero_num)
                zero_num -= zero_num_this_block
                file_bits.append(False)
                file_bits.extend(int2bitsarray(zero_num_this_block-1, bitnum=zero_block))
    
    bit_num = len(file_bits)
    byte_num = np.ceil(bit_num / 8)
    byte_array = []
    file_bits.extend([False]*8)
    for i in range(int(byte_num)):
        this_byte_arr = file_bits[i*8:(i+1)*8]
        byte = bitsarray2int(this_byte_arr)
        byte_array.append(byte)
        
    return np.array(byte_array).astype(np.uint8)

def parse_byte_file(byte_file, zero_block=3):
    
    bits = []
    for i in range(byte_file.shape[0]):
        byte = byte_file[i]
        bits.extend(int2bitsarray(byte))
    
    parameters = []
    while len(bits) > 0:
        flag = bits.pop(0)
        if flag:
            bit_array = bits[:32]
            bits = bits[32:]
            parameters.append(bitsarray2float(bit_array))
        else:
            bit_array = bits[:zero_block]
            bits = bits[zero_block:]
            parameters.extend([0]*(bitsarray2int(bit_array) + 1))
    
    return parameters
            
def fill_module_parameters(void_module, par_array):
    filled_module = copy.deepcopy(void_module)
    for key, value in void_module.items():
        par_num = np.array(value).flatten().shape[0]
        filled_module[key] = torch.tensor(par_array[:par_num]).view(value.shape)
        par_array = par_array[par_num:]
        
    return filled_module
    

def store_sparse_module(original_module, save_path: str, log=None):
    
    os.makedirs(opd(save_path), exist_ok=True)
    
    calc_stat(original_module, log_res=True)
    number_array = get_pure_array(original_module)
    
    extracted, zero_neighbor = extract_zeros(number_array)
    # print(esti_storage_size(extracted, zero_block=3))
    
    bytes_arr = generate_byte_file(extracted, zero_block=3)
    # print(bytes_arr[:10])
    print("Saved file size: {} Byte".format(len(bytes_arr)))
    
    np.save(save_path, bytes_arr)
    return len(bytes_arr)
    
def read_sparse_module(void_module):
    
    read_path = "D:/proj-files-level2/data-compression/NeuralFiledsDataCompression/data/my_mc_test/mc_res.npy" 
    parameters_array = parse_byte_file(np.load(read_path))
    filled_module = fill_module_parameters(void_module,parameters_array)
    
    return filled_module

def check_integrity(recovered, original):
    cmp = (get_pure_array(recovered) == get_pure_array(original))
    if np.min(cmp) == True:
        print("Module intact!")
    else:
        print("Module damaged!")
        

if __name__=="__main__":
    # path = "D:/proj-files-level2/data-compression/NeuralFiledsDataCompression/outputs/data1_2021_1105_154113/exp/steps80000/compressed/module"
    # path = "D:/proj-files-level2/data-compression/NeuralFiledsDataCompression/outputs/data1_2021_1105_120406/exp/mc_steps10000/compressed/module"
    path = "D:/proj-files-level2/data-compression/NeuralFiledsDataCompression/data/my_mc_test/module_70"
    original_module = torch.load(path)["phi"]
    store_sparse_module(original_module)
    
    void_module = fill_module_parameters(original_module, [0]*10000)
    check_integrity(void_module, original_module)
    recoverd_module = read_sparse_module(void_module)
    
    check_integrity(recoverd_module, original_module)
    
    