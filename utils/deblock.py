from matplotlib.image import imsave
import numpy as np
from tool import read_img, save_img
import cv2
import math
import os

def alpha(x):
    return 0.8*(2**(x/6)-1)

def beta(x):
    return 0.5*x-7

def clip(x,min,max):
    if x < min:
        x = min
    if x > max:
        x = max
    return x

# 没有考虑QP,offset,之后看看要不要考虑这个
def judge_filter(p1,p0,q0,q1,index_a,index_b,thres):
    # print(p1,p0,q0,q1)
    # print(p0-q0,abs(p0-q0))
    if (p1+p0+q0+q1)/4 > thres: # The block artifacts of high brightness area is not obvious
        # print((p1+p0+q0+q1)/4,thres)
        return False
    if abs(p0-q0) < alpha(index_a):
        if abs(p1-p0) < beta(index_b) and abs(q1-q0) < beta(index_b):
            return True
    return False

def filter(p2,p1,p0,q0,q1,q2,index_b):
    # basic filter operation
    delta0 = (4*(q0-p0)+(p1-q1)+4)/8
    deltap1 = (p2+(p0+q0+1)/2-2*p1)/2
    deltaq1 = (q2+(q0+p0+1)/2-2*q1)/2
    # clipping
    table = [[20]] # TODO
    c1 = table[0][0]
    c0 = c1
    if abs(p2-p0) < beta(index_b): # luminance
        c0 += 1
    if abs(q2-q0) < beta(index_b): # luminance
        c0 += 1
    delta0 = clip(delta0,-c0,c0)
    deltap1 = clip(deltap1,-c1,c1)
    deltaq1 = clip(deltaq1,-c1,c1)
    # result
    # print(delta0,deltap1)
    p0 += delta0
    q0 -= delta0
    p1 += deltap1
    q1 += deltaq1
    return p0,q0,p1,q1

# (x1,y1)(x2,y2)为右边界线,img为单通道
def filter2d(p,img,index_a,index_b,thres):
    x1,y1,x2,y2 = p
    x1,x2,y1,y2 = int(x1),int(x2),int(y1),int(y2)
    if x1 == x2:
        dir = 0
        if x1 - 3 < 0 or x1 + 3 > img.shape[1] -1:
            return img
    elif y1 == y2:
        dir = 1
        if y1 - 3 < 0 or y1 + 3 > img.shape[0] -1:
            return img
    for y in range(y1,y2+1):
        for x in range(x1,x2+1):
            if dir == 0: # 水平滤波
                p2,p1,p0,q0,q1,q2 = img[y,int(x-3):int(x+3)]
                p2,p1,p0,q0,q1,q2 = float(p2),float(p1),float(p0),float(q0),float(q1),float(q2)
                print(f'1:{p1},{p0},{q0},{q1}')
                if judge_filter(p1,p0,q0,q1,index_a,index_b):
                    p0,q0,p1,q1 = filter(p2,p1,p0,q0,q1,q2,index_b)
                    img[y,int(x-2):int(x+2)] = int(p0),int(q0),int(p1),int(q1)
                    print(f'2:{p1},{p0},{q0},{q1}')
                # img[y1:y2,x1] = 0
            elif dir == 1: # 垂直滤波
                p2,p1,p0,q0,q1,q2 = img[int(y-3):int(y+3),x]
                p2,p1,p0,q0,q1,q2 = float(p2),float(p1),float(p0),float(q0),float(q1),float(q2)
                # print(f'1:{p1},{p0},{q0},{q1}')
                if judge_filter(p1,p0,q0,q1,index_a,index_b,thres):
                    p0,q0,p1,q1 = filter(p2,p1,p0,q0,q1,q2,index_b)
                    img[int(y-2):int(y+2),x] = int(p0),int(q0),int(p1),int(q1)
                    # print(f'2:{p1},{p0},{q0},{q1}')
                # img[y1,x1:x2] = 0
    return img

def read_deblock_img(step_dir:str, index_a:float, index_b:float, thres:float):
    origin_name = os.listdir(os.path.join(step_dir,'decompressed'))[0]
    save_name = origin_name[:-4] + '_deblocked.tif'
    img_path = os.path.join(step_dir,'decompressed',origin_name)
    module_dir = os.path.join(step_dir,'compressed/module')
    # deblocking
    img = read_img(img_path)
    info = f'index_a:{index_a},index_b:{index_b},thres:{thres}'
    print(info)
    block_infos = os.listdir(module_dir)
    lines = []
    if len(img.shape) == 3: # 2d image, hwc
        for block_info in block_infos:
            h, w = block_info.split('-')
            x1, x2 = w.split('_')[1:]
            y1, y2 = h.split('_')[1:]
            x1, x2, y1, y2 = int(x1), int(x2), int(y1), int(y2)
            if not [x1, y1, x1, y2] in lines:
                lines.append([x1, y1, x1, y2])
            if not [x2, y1, x2, y2] in lines:
                lines.append([x2, y1, x2, y2])
            if not [x1, y1, x2, y1] in lines:
                lines.append([x1, y1, x2, y1])
            if not [x1, y2, x2, y2] in lines:
                lines.append([x1, y2, x2, y2])
        for k in range(img.shape[-1]):
            for p in lines:
                img[:,:,k] = filter2d(p,img[:,:,k],index_a,index_b,thres)
    elif len(img.shape) == 4: # 3d image, dhwc
        for block_info in block_infos:
            d, h, w = block_info.split('-')
            z1, z2 = d.split('_')[1:]
            x1, x2 = w.split('_')[1:]
            y1, y2 = h.split('_')[1:]
            x1, x2, y1, y2, z1, z2 = int(x1), int(x2), int(y1), int(y2), int(z1), int(z2)
            for i in range(z1,z2+1):
                if not [i, x1, y1, x1, y2] in lines:
                    lines.append([i, x1, y1, x1, y2])
                if not [i, x2, y1, x2, y2] in lines:
                    lines.append([i, x2, y1, x2, y2])
                if not [i, x1, y1, x2, y1] in lines:
                    lines.append([i, x1, y1, x2, y1])
                if not [i, x1, y2, x2, y2] in lines:
                    lines.append([i, x1, y2, x2, y2])
        for k in range(img.shape[-1]):
            for p in lines:
                img[p[0],:,:,k] = filter2d(p[1:],img[p[0],:,:,k],index_a,index_b,thres)
                
    return img


if __name__ == '__main__':
    # paranmeter set
    save_path = 'dataset/gap/kodim04_result.png'
    img_path = 'Experiments/exp2/kodim04/kodim04-adaptive3_2022_0215_025012_786052/exp_00005/steps20000/decompressed/kodim04_decompressed.png'
    module_dir = 'Experiments/exp2/kodim04/kodim04-adaptive3_2022_0215_025012_786052/exp_00005/steps20000/compressed/module'
    index_a = 51
    index_b = index_a
    # deblocking
    img = read_img(img_path)
    info = f'index_a:{index_a},index_b:{index_b},thres:{thres}'
    print(info)
    block_infos = os.listdir(module_dir)
    lines = []
    if len(img.shape) == 3: # 2d image, hwc
        for block_info in block_infos:
            h, w = block_info.split('-')
            x1, x2 = w.split('_')[1:]
            y1, y2 = h.split('_')[1:]
            x1, x2, y1, y2 = int(x1), int(x2), int(y1), int(y2)
            if not [x1, y1, x1, y2] in lines:
                lines.append([x1, y1, x1, y2])
            if not [x2, y1, x2, y2] in lines:
                lines.append([x2, y1, x2, y2])
            if not [x1, y1, x2, y1] in lines:
                lines.append([x1, y1, x2, y1])
            if not [x1, y2, x2, y2] in lines:
                lines.append([x1, y2, x2, y2])
        for k in range(img.shape[-1]):
            for p in lines:
                img[:,:,k] = filter2d(p,img[:,:,k],index_a,index_b,thres)
        save_img(save_path,img)
    elif len(img.shape) == 4: # 3d image, dhwc
        for block_info in block_infos:
            d, h, w = block_info.split('-')
            z1, z2 = d.split('_')[1:]
            x1, x2 = w.split('_')[1:]
            y1, y2 = h.split('_')[1:]
            x1, x2, y1, y2, z1, z2 = int(x1), int(x2), int(y1), int(y2), int(z1), int(z2)
            for i in range(z1,z2+1):
                if not [i, x1, y1, x1, y2] in lines:
                    lines.append([i, x1, y1, x1, y2])
                if not [i, x2, y1, x2, y2] in lines:
                    lines.append([i, x2, y1, x2, y2])
                if not [i, x1, y1, x2, y1] in lines:
                    lines.append([i, x1, y1, x2, y1])
                if not [i, x1, y2, x2, y2] in lines:
                    lines.append([i, x1, y2, x2, y2])
        for k in range(img.shape[-1]):
            for p in lines:
                img[p[0],:,:,k] = filter2d(p[1:],img[p[0],:,:,k],index_a,index_b,thres)
        save_img(save_path,img)
    
    