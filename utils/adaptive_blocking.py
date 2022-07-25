import gurobipy as gp
import cv2
import numpy as np
import matplotlib.pyplot as plt
import copy
import random
import tifffile
import cv2
from math import log, sqrt, ceil
import os
import numpy as np
import copy
from utils.tool import get_dimension, read_img, save_img
import pywt
import math

def gradient(img,dx:int=0,dy:int=0,dz:int=0):
    sobelx=cv2.Sobel(img,cv2.CV_64F,dx=dx,dy=0)
    sobelx=cv2.convertScaleAbs(sobelx)

    sobely=cv2.Sobel(img,cv2.CV_64F,dx=0,dy=dy)
    sobely=cv2.convertScaleAbs(sobely)

    result=cv2.addWeighted(sobelx,0.5,sobely,0.5,0)

    # result=cv2.Sobel(img,cv2.CV_64F,dx=dx,dy=dy)
    # result=cv2.convertScaleAbs(result)
    return result

def dwt(img, wavelet, mode, level):
    coeffs = pywt.wavedec2(data=img, wavelet=wavelet, mode=mode, level=level)
    high_fre = []
    feature = []
    for i in range(level):
        (H, V, D) = coeffs[i+1]
        high_fre += [H, V, D]
        feature += [(H**2).mean(),(V**2).mean(),(D**2).mean()]
    feature = np.array(feature)
    return feature.max()/feature.sum()

def cal_feature(image):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        f = np.fft.fft(np.fft.fft(gray,axis=0),axis=1)
    elif len(image.shape) == 4:
        f = np.fft.fft(np.fft.fft(np.fft.fft(image,axis=0),axis=1),axis=2)
    f = np.abs(f)
    feature = int(f.max())/int(f.sum())
    return feature

class Patch2d():
    def __init__(self, optim_model, parent, level, orderx, ordery) -> None:
        self.level = level
        self.orderx = orderx
        self.ordery = ordery
        self.parent = parent
        self.children = []
        self.optim_model = optim_model
        self.active = self.optim_model.addVar(vtype=gp.GRB.BINARY,name=f"{self.level}-{self.orderx}-{self.ordery}")
        self.prune = False
    def get_children(self):
        for i in range(2):
            for j in range(2):
                orderx = 2*self.orderx + j
                ordery = 2*self.ordery + i
                child = Patch2d(self.optim_model,parent=self,level=self.level+1,orderx=orderx,ordery=ordery)
                self.children.append(child)
        return self.children
    def init_data(self,data,h,w):
        self.h = h//(2**self.level)
        self.w = w//(2**self.level)
        self.y = self.h*self.ordery
        self.x = self.w*self.orderx
        self.data = copy.deepcopy(data[self.y:self.y+self.h,self.x:self.x+self.w])
        return self.data
    def get_feature(self, Type):
        # self.variance = ((self.data-self.data.mean())**2).mean()
        if Type == 0:
            self.feature = ((self.data-self.data.mean())**2).mean()
        elif Type == 1:
            result = gradient(self.data,dx=1,dy=1)
            self.feature = ((result-result.mean())**2).mean()
        elif Type == 2:
            wavelet = 'sym2'
            mode = 'symmetric'
            level = 4
            self.feature = dwt(self.data, wavelet, mode, level)
        elif Type == 3:
            # f = np.fft.fft(np.fft.fft(self.data,axis=0),axis=1)
            # f = np.abs(f) 
            # self.feature = f[0][0]/f.sum()
            self.feature = cal_feature(self.data)
        else:
            raise NotImplementedError
        return self.feature
    def active(self):
        self.prune = False
        self.active = self.optim_model.addVar(vtype=gp.GRB.BINARY,name=f"{self.level}-{self.orderx}-{self.ordery}-{self.orderz}")
    def deactive(self):
        self.prune = True
        self.optim_model.remove(self.active)

class QuadTree():
    def __init__(self, data, max_level, Type, var_thr, e_thr):
        # data
        self.data = data
        self.Type = Type
        self.h = data.shape[0]  # cv2加载图像为(h,w,c)
        self.w = data.shape[1]
        self.max_level = max_level
        assert len(self.data.shape) == 2,"data must be 2d!"
        assert self.h%(2**max_level) == 0 and self.w%(2**max_level) == 0,"image size error!"
        # optimizer
        self.optim_model = gp.Model()
        # quadtree and addvar
        self.tree = Patch2d(self.optim_model,parent=None,level=0, orderx=0, ordery=0)
        self.patch_list = []    
        self.patch_dict = {}   
        self.init_tree(self.tree,0)
        self.tree2list(self.tree)
        self.tree2dict(self.tree)
        self.init_data()
        self.prune(var_thr,e_thr)
        self.get_feature()
        self.optim_model.update()
    def init_tree(self,parent,level):
        # self.patch_list.append(parent)
        if level < self.max_level:
            children = parent.get_children()    # 只可调用一次
            for child in children:
                self.init_tree(child,level+1)
    def tree2list(self,patch):  # dfs
        self.patch_list.append(patch)
        children = patch.children
        if len(children) != 0:
            for child in children:
                self.tree2list(child)
    def tree2dict(self,patch):  # dfs
        if not (str(patch.level) in self.patch_dict):
            self.patch_dict[str(patch.level)] = [patch]
        else:
            self.patch_dict[str(patch.level)].append(patch)
        children = patch.children
        if len(children) != 0:
            for child in children:
                self.tree2dict(child)
    # def tree2dict1(self):  # dfs
    #     for patch in self.patch_list:
    #         if not (str(patch.level) in self.patch_dict):
    #             self.patch_dict[str(patch.level)] = [patch]
    #         else:
    #             self.patch_dict[str(patch.level)].append(patch)
    def init_data(self):
        for patch in self.patch_list:
            patch.init_data(self.data,self.h,self.w)
    def get_depth(self):
        patch = self.tree
        while len(patch.children) != 0:
            patch = patch.children[0]
        return patch.level
    def get_feature(self):
        for patch in self.patch_list:
            if patch.prune == False:    # 未被裁剪
                patch.get_feature(self.Type)
    def draw(self,data:np.array=None):
        if data.any() == None:
            data = copy.deepcopy(self.data)
        for patch in self.patch_list:
            if patch.prune == False:
                if int(patch.active.x) == 1:
                    x,y,w,h = patch.x,patch.y,patch.w,patch.h
                    cv2.rectangle(data, (x, y), (x+w, y+h), (0, 0, 255), 2)
                    # cv2.putText(data, str(int(patch.variance)), (x+int(w/3), y+int(h/3)), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 0), 1)
                # print('({}, {}), ({}, {})'.format(x,y,x+w,y+h))
            # else:
            #     x,y,w,h = patch.x,patch.y,patch.w,patch.h
            #     data[y:y+h,x:x+w,1] += 10
        return data
    def get_descendants(self,patch):# 获取该节点的子孙节点（不包含自身）
        descendants = []
        children = patch.children
        descendants += children
        if len(children) == 0:
            return []
        for child in children:
            descendants += self.get_descendants(child)
        return descendants
    def get_genealogy(self,patch):  # 获取该节点的所有父系节点 （包括自身）
        genealogy = [patch]
        while (patch.parent != None):
            genealogy.append(patch.parent)
            patch = patch.parent
        return genealogy
    def solve_optim(self, Nb, min_level):
        self.Nb = Nb
        Obj = []
        Constr = []
        for patch in self.patch_list:
            if patch.prune == False:    # 未被裁剪
                Obj.append(patch.feature*patch.active/(4**patch.level))
                # Obj.append(patch.variance*patch.active)
                Constr.append(patch.active)
                # 4.the active chunk's level should larger than the min_level
                if patch.level < min_level:
                    self.optim_model.addConstr(patch.active == 0)
        if self.Type == 0 or self.Type == 1:
            self.optim_model.setObjective(gp.quicksum(Obj), gp.GRB.MINIMIZE)
        elif self.Type == 2 or self.Type == 3:
            self.optim_model.setObjective(gp.quicksum(Obj), gp.GRB.MAXIMIZE)
        else:
            raise NotImplementedError
        # Add constraints
        # 1.the total numbers of the active chunks should not be larger than the set value
        self.optim_model.addConstr(gp.quicksum(Constr) <= self.Nb)
        # 2.only one menber can be active in the same genealogy
        depth = self.get_depth()
        deepest_layer = self.patch_dict[str(depth)]
        for patch in deepest_layer:
            genealogy = self.get_genealogy(patch)
            actives = []
            for patch in genealogy:
                if patch.prune == False: 
                    actives.append(patch.active)
            # 3.if one member is pruned, the numbers of the other active members in the same genealogy should lease than one
            if len(actives) < len(genealogy) and len(actives) >= 2:   # 该条族谱线上至少还有两个未被prune的节点，则需要满足
                self.optim_model.addConstr(gp.quicksum(actives) <= 1)
                # print(len(genealogy)-len(actives))
            elif len(actives) == len(genealogy):
                self.optim_model.addConstr(gp.quicksum(actives) == 1)
        # Solve it!
        self.optim_model.optimize()
        print(f"Optimal objective value: {self.optim_model.objVal}")
    def get_active(self):
        self.active_patch_list = []
        for patch in self.patch_list:
            if patch.prune == False:
                if int(patch.active.x) == 1:
                    self.active_patch_list.append(patch)
        return self.active_patch_list
    def prune(self,var_thr:float=0,e_thr:float=0):
        count = 0
        for patch in self.patch_list:
            if ((patch.data-patch.data.mean())**2).mean() <= var_thr and abs(patch.data.mean())<=e_thr:
                patch.deactive()
                count += 1
                descendants = self.get_descendants(patch)
                for descendant in descendants:
                    descendant.deactive()
                    count += 1
        print(f'prune numbers:{count}')
    def draw_tree(self):    # 必须再求解成功后才能调用
        actives = {}
        for patch in self.patch_list:
            if not (str(patch.level) in actives):
                actives[str(patch.level)] = [patch.active.x]
            else:
                actives[str(patch.level)].append(patch.active.x)
        for key in actives.keys():
            print(actives[key])

# if __name__=='__main__':
#     # Get data
#     img_path = 'mydata/mountain.jpg'
#     # img_path = 'mydata/exp.png'
#     img = cv2.imread(img_path)
#     # print(img.shape)
#     img = img[0:1024,300:1324,0]
#     # img = img[:,:,0]
#     tree = QuadTree(img,5)
#     tree.prune(0)
#     tree.solve_optim(100)
#     img = tree.draw()
#     save_path = 'img/exp.png'
#     cv2.imwrite(save_path,img)

class Patch3d():
    def __init__(self, optim_model, parent, level, orderx, ordery, orderz) -> None:
        self.level = level
        self.orderx = orderx
        self.ordery = ordery
        self.orderz = orderz
        self.parent = parent
        self.children = []
        self.optim_model = optim_model
        self.active()
    def get_children(self):
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    orderx = 2*self.orderx + k
                    ordery = 2*self.ordery + j
                    orderz = 2*self.orderz + i
                    child = Patch3d(self.optim_model,parent=self,level=self.level+1,orderx=orderx,ordery=ordery,orderz=orderz)
                    self.children.append(child)
        return self.children
    def init_data(self,data,d,h,w):
        self.d = d//(2**self.level)
        self.h = h//(2**self.level)
        self.w = w//(2**self.level)
        self.z = self.d*self.orderz
        self.y = self.h*self.ordery
        self.x = self.w*self.orderx
        self.data = copy.deepcopy(data[self.z:self.z+self.d,self.y:self.y+self.h,self.x:self.x+self.w])
        return self.data
    def get_feature(self, Type):
        if Type == 0:
            self.feature = ((self.data-self.data.mean())**2).mean()
        elif Type == 1:
            # f = np.fft.fft(np.fft.fft(np.fft.fft(self.data,axis=0),axis=1),axis=2)
            # f = np.abs(f)
            # self.feature = int(f[0][0][0])/int(f.sum())
            self.feature = cal_feature(self.data)
        else:
            raise NotImplementedError
        return self.feature
    def active(self):
        self.prune = False
        self.active = self.optim_model.addVar(vtype=gp.GRB.BINARY,name=f"{self.level}-{self.orderx}-{self.ordery}-{self.orderz}")
    def deactive(self):
        self.prune = True
        self.optim_model.remove(self.active)

class OctTree():
    def __init__(self, data, max_level, Type, var_thr, e_thr):
        # data (d,h,w)
        self.Type = Type
        self.data = data
        self.d = data.shape[0]  
        self.h = data.shape[1]
        self.w = data.shape[2]
        self.max_level = max_level
        assert len(self.data.shape) == 3 or (len(self.data.shape) == 4 and self.data.shape[-1]==1),"data must be 3d!"
        assert self.d%(2**max_level) ==0 and self.h%(2**max_level) == 0 and self.w%(2**max_level) == 0,"image size error!"
        # optimizer
        self.optim_model = gp.Model()
        # octtree and addvar
        self.tree = Patch3d(self.optim_model,parent=None,level=0, orderx=0, ordery=0, orderz=0)
        self.patch_list = []    
        self.patch_dict = {}   
        self.init_tree(self.tree,0)
        self.tree2list(self.tree)
        self.tree2dict(self.tree)
        self.init_data()
        self.prune(var_thr,e_thr)
        self.get_feature()
        self.optim_model.update()
    def init_tree(self,parent,level):
        if level < self.max_level:
            children = parent.get_children()    # 只可调用一次
            for child in children:
                self.init_tree(child,level+1)
    def tree2list(self,patch):
        self.patch_list.append(patch)
        children = patch.children
        if len(children) != 0:
            for child in children:
                self.tree2list(child)
    def tree2dict(self,patch):
        if not (str(patch.level) in self.patch_dict):
            self.patch_dict[str(patch.level)] = [patch]
        else:
            self.patch_dict[str(patch.level)].append(patch)
        children = patch.children
        if len(children) != 0:
            for child in children:
                self.tree2dict(child)
    def init_data(self):
        for patch in self.patch_list:
            patch.init_data(self.data,self.d,self.h,self.w)
    def get_depth(self):
        patch = self.tree
        while len(patch.children) != 0:
            patch = patch.children[0]
        return patch.level
    def get_feature(self):
        for patch in self.patch_list:
            if patch.prune == False:    # 未被裁剪
                patch.get_feature(self.Type)
    def get_descendants(self,patch):# 获取该节点的子孙节点（不包含自身）
        descendants = []
        children = patch.children
        descendants += children
        if len(children) == 0:
            return []
        for child in children:
            descendants += self.get_descendants(child)
        return descendants
    def get_genealogy(self,patch):  # 获取该节点的所有父系节点 （包括自身）
        genealogy = [patch]
        while (patch.parent != None):
            genealogy.append(patch.parent)
            patch = patch.parent
        return genealogy
    def solve_optim(self, Nb, min_level):
        self.Nb = Nb
        Obj = []
        Constr = []
        for patch in self.patch_list:
            if patch.prune == False:    # 未被裁剪
                Obj.append(patch.feature*patch.active/(8**patch.level))
                Constr.append(patch.active)
                # 4.the active chunk's level should larger than the min_level
                if patch.level < min_level:
                    self.optim_model.addConstr(patch.active == 0)
        if self.Type == 0:
            self.optim_model.setObjective(gp.quicksum(Obj), gp.GRB.MINIMIZE)
        elif self.Type == 1:
            self.optim_model.setObjective(gp.quicksum(Obj), gp.GRB.MAXIMIZE)
        else:
            raise NotImplementedError
        # Add constraints
        # 1.the total numbers of the active chunks should not be larger than the set value
        self.optim_model.addConstr(gp.quicksum(Constr) <= self.Nb)
        # 2.only one member can be active in the same genealogy
        depth = self.get_depth()
        deepest_layer = self.patch_dict[str(depth)]
        for patch in deepest_layer:
            genealogy = self.get_genealogy(patch)
            actives = []
            for patch in genealogy:
                if patch.prune == False: 
                    actives.append(patch.active)
            # 3.if one member is pruned, the numbers of the other active members in the same genealogy should lease than one
            if len(actives) < len(genealogy) and len(actives) >= 2:   # 该条族谱线上至少还有两个未被prune的节点，则需要满足
                self.optim_model.addConstr(gp.quicksum(actives) <= 1)
                # print(len(genealogy)-len(actives))
            elif len(actives) == len(genealogy):
                self.optim_model.addConstr(gp.quicksum(actives) == 1)
        # Solve it!
        self.optim_model.optimize()
        # print(f"Optimal objective value: {self.optim_model.objVal}")
    def prune(self,var_thr:float=0,e_thr:float=0):
        count = 0
        for patch in self.patch_list:
            if ((patch.data-patch.data.mean())**2).mean() <= var_thr and abs(patch.data.mean())<=e_thr:
                # print(patch.variance)
                patch.deactive()
                count += 1
                descendants = self.get_descendants(patch)
                for descendant in descendants:
                    descendant.deactive()
                    count += 1
        print(f'prune numbers:{count}')
    def get_active(self):
        self.active_patch_list = []
        for patch in self.patch_list:
            if patch.prune == False:
                if int(patch.active.x) == 1:
                    self.active_patch_list.append(patch)
        return self.active_patch_list
    def draw(self,data:np.array=None):
        if data.any() == None:
            data = copy.deepcopy(self.data)
        for patch in self.patch_list:
            if patch.prune == False:
                if int(patch.active.x) == 1:
                    x,y,z,w,h,d = patch.x,patch.y,patch.z,patch.w,patch.h,patch.d
                    data[z,y:y+h,x:x+w] = 2000
                    data[z+d-1,y:y+h,x:x+w] = 2000
                    data[z:z+d,y,x:x+w] = 2000
                    data[z:z+d,y+h-1,x:x+w] = 2000
                    data[z:z+d,y:y+h,x] = 2000
                    data[z:z+d,y:y+h,x+w-1] = 2000
        return data
    def draw_tree(self):    # 必须再求解成功后才能调用
        actives = {}
        for patch in self.patch_list:
            if not (str(patch.level) in actives):
                actives[str(patch.level)] = [int(not patch.prune)]
            else:
                actives[str(patch.level)].append(int(not patch.prune))
        for key in actives.keys():
            print(actives[key])

# if __name__=='__main__':
#     # Get data
#     # tif_path = 'data1/data1_cropped_320_320_320_preprocessed.tif'
#     tif_path = 'mydata/data2_cropped_320_320_320.tif'
#     data = tifffile.imread(tif_path)

#     tree = OctTree(data,5)
#     tree.prune(0)
#     tree.solve_optim(250)
#     save_data = tree.draw()

#     save_path = 'img/data2-exp.tif'
#     tifffile.imsave(save_path,save_data)

# gpu_limit: 80*80*80*2 = 1024000
def adaptive_cal_tree(img_path,param_size,var_thr:float=-1,e_thr:float=-1,gpu_limit:int=1024000,maxl:int=-1,minl:int=-1,Nb:int=-1,Type=0):
    dimension = get_dimension(img_path)
    img = read_img(img_path)
    data = copy.deepcopy(img)
    if len(data.shape) == 4:
        if data.shape[-1] == 3:
            gray = np.zeros(data.shape[:-1]).astype(data.dtype.name)
            for i in range(data.shape[0]):
                gray[i] = cv2.cvtColor(data[i],cv2.COLOR_RGB2GRAY)
            data = gray
        # if img_size < gpu_limit:
        #     if minl == -1:
        #         minl = 0
        # else:
        #     minl = int(log((img_size/gpu_limit)**(1/3),2)) 
        # 以f=20、l=5、float32的siren网络耳朵参数量为分块参数平均值，计算分块数目Nb 3*20+20+3*(20*20+20)+20*1+1=1361
        if Nb == -1:
            Nb = int(param_size/(4*1361))
            if Nb <= 0:
                Nb = 1
        # minl尽量均匀
        minl = math.floor(log(Nb,2**dimension))
        maxl = minl + 2
        # (2^maxl)^3 = Nb为临界情况
        # if maxl == -1:
        #     maxl = int(log(Nb**(1/3),2)+1.5)
        # Create an octree
        tree = OctTree(data,maxl,Type,var_thr,e_thr)
    elif len(data.shape) == 3:
        if len(data.shape) == 3:
            data = cv2.cvtColor(data,cv2.COLOR_RGB2GRAY)
        # if img_size < gpu_limit:
        #     if minl == -1:
        #         minl = 0
        # else:
        #     minl = int(log(sqrt(img_size/gpu_limit),2))
        # 以f=10、l=5、float32的siren网络耳朵参数量为分块参数平均值，计算分块数目Nb 2*10+10+3*(10*10+10)+10*1+1=371
        if Nb == -1:
            Nb = int(param_size/(4*1361))
            if Nb <= 0:
                Nb = 1
        minl = math.floor(log(Nb,2**dimension))
        maxl = minl + 2
        # (2^maxl)^2 = Nb为临界情况
        # if maxl == -1:
        #     maxl = int(log(Nb**(1/2),2)+1.5)
        # Create a quadtree
        tree = QuadTree(data,maxl,Type,var_thr,e_thr)
    tree.solve_optim(Nb,minl)
    save_data = copy.deepcopy(img)
    save_data = tree.draw(save_data)
    info = 'maxl:{},minl:{},var_thr:{},e_thr:{},Nb:{}'.format(maxl,minl,var_thr,e_thr,Nb)
    print(info)
    print('number of blocks:{}'.format(len(tree.get_active())))
    return tree, save_data, dimension

# if __name__=='__main__':
#     # Set parameter
#     ratio = 100
#     var_thr,e_thr = 0,0
#     gpu_limit = 858580  # 规定一次放入GPU的最大字节数不得超过该值

#     # Get data
#     img_path = 'dataset/origin/data1_cropped_320_320_320.tif'
#     # img_path = 'dataset/png/kodim17.png'
#     if 'tif' in img_path:
#         origin = tifffile.imread(img_path)
#     elif 'png' in img_path:
#         origin = cv2.imread(img_path)

#     _, save_data = adaptive_cal_tree(img_path,origin,ratio,var_thr,e_thr)

#     save_path = 'dataset/output/'+img_path.split('/')[-1]
#     if 'tif' in img_path:
#         tifffile.imsave(save_path,save_data)
#     elif 'png' in img_path:
#         cv2.imwrite(save_path,save_data)


def cal_factor(n):
    fac = [1] # 定义一个列表存放因子
    for i in range(2, n):
        if n % i == 0:
            fac.append(i)
    return fac

def cal_divide_num(d,h,w,Nb,Type,param_size):
    fac_d = cal_factor(d)
    fac_h = cal_factor(h)
    fac_w = cal_factor(w)
    num_max = 0
    # 以f=20、l=5、float32的siren网络耳朵参数量为分块参数平均值，计算分块数目Nb 3*20+20+3*(20*20+20)+20*1+1=1361
    if Nb <= 0:
        Nb = int(param_size/(4*1361))
        if Nb <= 0:
            Nb = 1
    for nd in fac_d:
        for nh in fac_h:
            for nw in fac_w:
                num = nd*nh*nw
                if num <= Nb:
                    if num > num_max:
                        num_max = num
                        number = np.array([nd,nh,nw])
                        if Type == 0:
                            var_min = ((number - number.mean())**2).mean()
                        elif Type == 1:
                            size = np.array([d/nd,h/nh,w/nw])
                            var_min = ((size - size.mean())**2).mean()
                    elif num == num_max: # 因为是先遍历d,大值优先率w>h>d: [2,4,4] [4,2,4] [4,4,2] 三个中选[2,4,4]
                        number_tem = np.array([nd,nh,nw])
                        if Type == 0:
                            var_tem = ((number_tem - number_tem.mean())**2).mean()
                        elif Type == 1:
                            size_tem = np.array([d/nd,h/nh,w/nw])
                            var_tem = ((size_tem - size_tem.mean())**2).mean()
                        if var_tem < var_min:
                            number = number_tem
                            var_min = var_tem
    return number