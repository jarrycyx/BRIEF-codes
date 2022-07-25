import math
import sys
from utils.io import gen_data_path_list_list, invnormalize_data, normalize_data, read_data_batch, save_data_batch
import numpy as np
import torch
import os
import random
import cv2
import shutil
import copy
from typing import Callable, List, Tuple,Dict, Union
import tifffile

sys.setrecursionlimit(100000)  # 将默认的递归深度修改为3000  16090次递归会报错Segmentation fault (core dumped)

# 开运算 = 先腐蚀运算，再膨胀运算
def open_img(img,iter):
    #iterations为整数，可调，越大效果越强
    img = cv2.erode(img, None, iterations=iter) #腐蚀
    img = cv2.dilate(img, None, iterations=iter)#膨胀
    return img

# 沿z轴开运算
def open3d(data0,iter):
    data  = data0.copy()
    for i in range(data.shape[0]):
        data[i] = open_img(data[i],iter)
    return data

#进度条打印
def progress(per:float,width=50):
    text = ('\r[%%-%ds]'%width)%('#'*int(per*width))
    text += '%3s%%'
    text = text%(round(per*100))
    print('\t'+text,end='')

# 获取方向向量
def get_dir(gap_z:int=1,gap_xy:int=1):
    dir = []
    # 本层
    for i in range(-gap_xy,gap_xy+1):
        for j in range(-gap_xy,gap_xy+1):
            if not(i == 0 and j == 0):
                dir.append([i,j,0])
    # 上下层，因为跨z，所以gap不同
    for i in range(-gap_z,gap_z+1):
        for j in range(-gap_z,gap_z+1):
            dir.append([i,j,1])
            dir.append([i,j,-1])
    return dir

class Point:
    def __init__(self,pos:list,dir:list) -> None:
        self.origin = pos
        self.pos = pos
        self.index = 0
        self.dir = dir
    # 判断是否遍历完成
    def check_index(self):
        if self.index < len(self.dir):
            return True
        else:
            return False
    # 来到下一个pos点
    def next_pos(self):
        self.pos = [self.origin[i] + self.dir[self.index][i] for i in range(3)]
    # 改变遍历方向
    def index_plus(self):
        self.index += 1

class Neuron:
    def __init__(self,origin:list,gap_z:int=1,gap_xy:int=1,name:str='none',data:np.ndarray=None) -> None:
        self.origin = origin
        self.visited = np.zeros_like(data)
        self.path = []
        self.sub = [self.origin]
        self.name = name
        if data is None:
            raise NotImplementedError
        else:
            self.data = data
        self.dir = get_dir(gap_z,gap_xy)
        self.recursion = 0
    # 判断pos是否在data下标范围内
    def check_index(self,pos:list):
        if (pos[0] in range(self.data.shape[0])) and (pos[1] in range(self.data.shape[1])) and (pos[2] in range(self.data.shape[2])):
            return True
        else:
            return False
    # 获取路径长度
    def len(self):
        return len(self.path)
    # 判断pos是否为可行点
    def judge_go(self,pos:list):
        if self.check_index(pos):
            if (self.visited[pos[0],pos[1],pos[2]] == 0 and self.data[pos[0],pos[1],pos[2]] != 0):
                return True
        return False
    # 深度优先遍历寻路(数据稀疏不会造成递归转储报错时)
    def dfs(self,pos:list):
        if not self.check_index(pos) or (self.data[pos[0],pos[1],pos[2]] == 0):
            return
        self.path.append(pos)
        self.visited[pos[0],pos[1],pos[2]] = 1
        origin = pos
        index = 0
        while(index < len(self.dir)):
            pos = [origin[i] + self.dir[index][i] for i in range(3)]
            if self.judge_go(pos):
                self.dfs(pos)
            index += 1
    # 分段深度优先遍历寻路，当快达到递归转储报错数目时，暂存该点，继续遍历
    def dfs_flag(self,pos:list,recursion_thr:int=14000):
        # 下标不满足要求 数据为0 已经遍历过
        if not self.check_index(pos) or (self.data[pos[0],pos[1],pos[2]] == 0) or self.visited[pos[0],pos[1],pos[2]] != 0: 
            return
        # 递归转储报错，暂存该点，下一次再遍历 当gapz=3时14000次就会报错,所以可能需要看情况修改
        if self.recursion>recursion_thr:
            self.sub.append(pos)
            return
        self.path.append(pos)
        self.recursion += 1
        self.visited[pos[0],pos[1],pos[2]] = 1
        # print(len(self.path),self.recursion)
        origin = pos
        index = 0
        while(index < len(self.dir)):
            pos = [origin[i] + self.dir[index][i] for i in range(3)]
            if self.judge_go(pos):
                self.dfs_flag(pos)
            index += 1
    def dfs_sub(self,recursion_thr):
        while (len(self.sub) != 0):
            self.dfs_flag(self.sub[0],recursion_thr)      
            self.sub.pop(0)
            self.recursion = 0

def preprocess(data0,tif_dir,sub,iter):
    origin_path = tif_dir+'/origin.tif'
    save_data_batch(data0,[origin_path])
    data = data0[0,0].copy()
    data[data<sub] = 0
    data = open3d(data,iter)
    data0[0,0] = data
    pre_path = tif_dir+'/preprocess.tif'
    save_data_batch(data0,[pre_path])
    return data0

def creat_project(project,name):
    if not os.path.exists(project):  
        os.mkdir(project)
    project = project + '/' + name
    if not os.path.exists(project):  
        os.mkdir(project)
    files_dir = project+'/TXTS'
    if not os.path.exists(files_dir):  
        os.mkdir(files_dir)
    tif_dir = project + '/TIFS'
    if not os.path.exists(tif_dir):  
        os.mkdir(tif_dir)
    return files_dir,tif_dir

def find_neuron(data0,thr,gap_z:int=1,gap_xy:int=1):
    data = copy.deepcopy(data0[0,0])    # 遍历数据
    flag = np.zeros_like(data)          # 标志数据
    weight = np.zeros_like(data0)       # 权重数据
    data = np.transpose(data,(2,1,0))   # dhw变换为xyz
    shape = data.shape
    count = 0
    neuron_id = 0
    point_count = 0 # 原数据中非零点的个数，验证神经元计数时是否有重复错误
    white_count = 0 # 找到的每个神经元点集合中点的总数
    recursion_thr = 1400
    for x in range(shape[0]):
        for y in range(shape[1]):
            for z in range(shape[2]):
                if data[x,y,z] != 0: 
                    white_count += 1
                if data[x,y,z] != 0 and flag[x,y,z] == 0:
                    count += 1
                    neuron = Neuron([x,y,z],gap_z,gap_xy,str(count),data)
                    neuron.dfs_sub(recursion_thr)
                    flag = flag + neuron.visited    
                    point_count += len(neuron.path)
                    if len(neuron.path) > thr:
                        neuron_id += 1
                        weight[0,0] += neuron.visited
    print('total origin:'+str(white_count)+';total get:'+str(point_count))
    if white_count == point_count:
        print('Completed!')
    else:
        print('Error!')
    weight[0,0] = np.transpose(weight[0,0],(2,1,0))
    return weight

def find_weight(data0,thr,gap_z:int=1,gap_xy:int=1,method:str='FindOrigin'):
    data = copy.deepcopy(data0[0,0])    # 遍历数据
    flag = np.zeros_like(data)          # 标志数据
    weight = np.zeros_like(data0)       # 权重数据
    data = np.transpose(data,(2,1,0))   # dhw变换为xyz
    shape = data.shape
    count = 0
    white_count = (data!=0).sum()       # 原数据中非零点的个数，验证神经元计数时是否有重复错误
    point_count = 0                     # 找到的每个神经元点集合中点的总数
    recursion_thr = 1400                # 设置最大递归次数，防止递归转储报错
    for x in range(shape[0]):
        for y in range(shape[1]):
            for z in range(shape[2]):
                if data[x,y,z] != 0 and flag[x,y,z] == 0:
                    count += 1
                    neuron = Neuron([x,y,z],gap_z,gap_xy,str(count),data)
                    neuron.dfs_sub(recursion_thr)
                    flag = flag + neuron.visited    
                    point_count += len(neuron.path)
                    weight[0,0] += neuron.visited*len(neuron.path)  # weight中记录长度
    if white_count != point_count:
        raise ValueError('The dfs count must be same to the origin white count')
    weight[0,0] = np.transpose(weight[0,0],(2,1,0))
    weight[weight<thr] = 0
    if 'Origin' in method:
        pass
    elif 'Exp' in method:
        weight = (weight-weight.min())/(weight.max()-weight.min())
        weight = np.exp(weight)   # 训练高值部分
        # weight = np.exp(-3*weight)    # 训练低值部分
    else:
        raise NotImplemented
    return weight

# def main():
#     # 要调整的参数主要有preprocess()的sub和iter,find_neuron()的thr、gap_z、gap_xy
#     sub,iter,thr,gap_z,gap_xy = 100,1,20,2,1
#     data_path = 'xtx/mydata/data1_d_256_320_h_256_320_w_256_320.tif'
#     project = 'xtx/weight'
#     name = 'exp_'+str(sub)+'_'+str(iter)+'_'+str(thr)+'_'+str(gap_z)+'_'+str(gap_xy)
#     files_dir,tif_dir = creat_project(project,name)
#     data = read_data_batch([data_path])
#     # data = preprocess(data,tif_dir,sub,iter)
#     data[data<sub] = 0
#     print((data!=0).sum())
#     method = 'FindExp'
#     weight = find_weight(data,thr,gap_z,gap_xy,method)
#     print(weight.max(),weight.min())

# if __name__=='__main__':
#     main()
def cal_ratio(origin,thr):
    if len(origin.shape) == 3:
        data = origin
    elif len(origin.shape) == 5:
        data = origin[0,0]
    else:
        raise NotImplemented
    lower = ((data<thr)*(data!=0)).sum()
    higher = (data>=thr).sum()
    zero = (data==0).sum()
    ratio = higher/lower
    if lower + higher +zero != data.shape[0]*data.shape[1]*data.shape[2]:
        raise ValueError
    return ratio
def dynamic_parse_weight(data:np.ndarray,thr:int):
    if not isinstance(data,np.ndarray):
        data = np.array(data)
    weight = np.ones_like(data).astype(np.float32)
    ratio =cal_ratio(data,thr)
    if ratio < 0.2:
        mid_x,mid_value = thr,0.05
        a = -np.log(mid_value)/mid_x
        weight = np.exp(-a*data)
    elif ratio >= 0.2:
        # gap_z,gap_xy = 2,1
        # method = 'FindExp'
        # weight = find_weight(data,thr,gap_z,gap_xy,method)
        l,h,scale = 2001,65535,0.01
        weight[(data>=l) * (data<=h)] = scale
    else:
        raise NotImplementedError
    return weight
def get_ratios(thr):
    #### 求取动态分配权重方法的临界tatio值
    origin_path = 'xtx/mydata/data1_cropped_320_320_320_preprocessed.tif'
    origin = tifffile.imread(origin_path)
    size = 64
    ratios = []
    for i in range(5):
        for j in range(5):
            for k in range(5):
                d1,d2 = size*i,size*(i+1)
                h1,h2 = size*j,size*(j+1)
                w1,w2 = size*k,size*(k+1)
                origin_chunk = origin[d1:d2,h1:h2,w1:w2]
                ratio = cal_ratio(origin_chunk,thr)
                ratios.append(ratio)
    ratios = np.array(ratios)
    print(ratios.max())
    print(ratios.min())
    print(ratios.mean())