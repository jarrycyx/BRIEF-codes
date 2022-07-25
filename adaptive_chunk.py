import tifffile
from utils.adaptive_blocking import OctTree, QuadTree, adaptive_cal_tree
import cv2
from math import log, sqrt
import os
import numpy as np
import copy

# def adaptive_cal_tree(img_path,img,ratio,var_thr:float=0,e_thr:float=0,gpu_limit:int=858580,maxl:int=-1,minl:int=-1,Nb:int=-1):
#     img_size = os.path.getsize(img_path)
#     para_size = img_size/ratio
#     data = copy.deepcopy(img)
#     if 'tif' in img_path:
#         assert len(data.shape) == 3 or len(data.shape) == 4,"image must be 3d!"
#         if img_size < gpu_limit:
#             if minl == -1:
#                 minl = 0
#         else:
#             minl = int(log((img_size/gpu_limit)**(1/3),2)) 
#         # 以f=20、l=5、float32的siren网络耳朵参数量为分块参数平均值，计算分块数目Nb 3*20+20+3*(20*20+20)+20*1+1=1361
#         if Nb == -1:
#             Nb = int(para_size/(4*1361))
#         # (2^maxl)^3 = Nb为临界情况
#         if maxl == -1:
#             maxl = int(log(Nb**(1/3),2)+1.5)
#         # Create an octree
#         tree = OctTree(data,maxl)
#     elif 'png' in img_path:
#         assert len(data.shape) == 2 or len(data.shape) == 3,"image must be 2d!"
#         if len(data.shape) == 3:
#             data = cv2.cvtColor(data,cv2.COLOR_RGB2GRAY)
#         if img_size < gpu_limit:
#             if minl == -1:
#                 minl = 0
#         else:
#             minl = int(log(sqrt(img_size/gpu_limit),2))
#         # 以f=10、l=5、float32的siren网络耳朵参数量为分块参数平均值，计算分块数目Nb 2*10+10+3*(10*10+10)+10*1+1=371
#         if Nb == -1:
#             Nb = int(para_size/(4*371))
#         # (2^maxl)^2 = Nb为临界情况
#         if maxl == -1:
#             maxl = int(log(Nb**(1/2),2)+1.5)
#         # Create a quadtree
#         tree = QuadTree(data,maxl)
#     tree.prune(var_thr,e_thr)
#     tree.solve_optim(Nb,minl)
#     save_data = copy.deepcopy(img)
#     save_data = tree.draw(save_data)
#     print(maxl,minl,var_thr,e_thr,Nb)
#     return tree, save_data

if __name__=='__main__':
    # Set data path and ratio
    # img_path = 'dataset/origin/data2_cropped_320_320_320.tif'
    img_path = 'dataset/origin/mountain_cropped_1024_1024.png'
    ratio = 100
    
    _, save_data, dimension = adaptive_cal_tree(img_path,ratio,var_thr=0,e_thr=0,maxl=-1,minl=-1,Nb=100,Type=3)
    save_path = 'dataset/output/'+img_path.split('/')[-1]
    if 'tif' in img_path:
        tifffile.imsave(save_path,save_data)
    elif 'png' in img_path:
        cv2.imwrite(save_path,save_data)
    