from functools import reduce
import sys
from typing import List, Tuple, Dict, Union
from utils.Typing import CompressFrameworkOpt
from utils.io import *
from utils.misc import *
from utils.dataset import *
import torch
import torch.nn as nn
import torch.optim

class _BaseCompressFramerwork:
    def __init__(self,opt: CompressFrameworkOpt) -> None:
        super().__init__()
        self.opt = opt
        self.module ={}
    def init_module(self,):
        """Initialize every Module in self.module.
        """
        raise NotImplementedError
    def load_module(self,module_path:str,serializing_method:str='torchsave'):#module_state_dict:Dict[str,Dict[str,torch.Tensor]]
        """Load every Module in self.module from the trained module.
        """
        if serializing_method in ['torchsave','torchsave_wo_new_zipfile']:
            module_state_dict = torch.load(module_path,map_location=torch.device('cpu'))
        elif serializing_method in ['torchsave_7z','torchsave_wo_new_zipfile_7z']:
            module_state_dict = read_7z(module_path)
            module_state_dict = torch.load(next(iter(module_state_dict.values())),map_location=torch.device('cpu'))  
        elif serializing_method == 'numpysave':
            module_state_dict = {k:self.module[k].cpu().state_dict() for k in self.module.keys()}
            for n in module_state_dict:
                for k in module_state_dict[n]:
                    data = np.load(opj(module_path,n,k,'0.npy'))
                    data = torch.from_numpy(data)
                    module_state_dict[n][k] = data
        elif serializing_method == 'numpysave_7z':
            module_extracted_dir = opj(opd(module_path),'temp_extracted')
            extract_7z(module_path,module_extracted_dir)
            module_state_dict = {k:self.module[k].cpu().state_dict() for k in self.module.keys()}
            for n in module_state_dict:
                for k in module_state_dict[n]:
                    data = np.load(opj(module_extracted_dir,'0',n,k,'0.npy'))
                    data = torch.from_numpy(data)
                    module_state_dict[n][k] = data
            os.system('rm -rf -r {}'.format(module_extracted_dir))#TODO 安全删除
        for k in module_state_dict.keys():
            self.module[k].load_state_dict(module_state_dict[k])
    def save_module(self,save_path:str,serializing_method:str='torchsave'):
        """Save the trained every Module in self.module into save_path e.g. xxx.pt.
        """
        module_state_dict = {k:self.module[k].cpu().state_dict() for k in self.module.keys()}
        if serializing_method == 'torchsave':
            torch.save(module_state_dict,save_path)
            module_size = os.path.getsize(save_path)
        elif serializing_method == 'torchsave_7z':
            torch.save(module_state_dict,save_path)
            write_7z([save_path],save_path+'_7z',[opb(save_path)])
            os.remove(save_path)
            os.rename(save_path+'_7z',save_path)
            module_size = os.path.getsize(save_path)
        elif serializing_method == 'torchsave_wo_new_zipfile':
            torch.save(module_state_dict,save_path,_use_new_zipfile_serialization=False)
            module_size = os.path.getsize(save_path)
        elif serializing_method == 'torchsave_wo_new_zipfile_7z':
            torch.save(module_state_dict,save_path,_use_new_zipfile_serialization=False)
            write_7z([save_path],save_path+'_7z',[opb(save_path)])
            os.remove(save_path)
            os.rename(save_path+'_7z',save_path)
            module_size = os.path.getsize(save_path)        
        elif serializing_method == 'numpysave':
            os.makedirs(save_path)
            for n in module_state_dict:
                for k in module_state_dict[n]:
                    os.makedirs(opj(save_path,n,k))
                    np.save(opj(save_path,n,k,'data.npy'),module_state_dict[n][k].numpy())
            module_size = get_folder_size(save_path) 
        elif serializing_method == 'numpysave_7z':
            os.makedirs(save_path)
            for n in module_state_dict:
                for k in module_state_dict[n]:
                    os.makedirs(opj(save_path,n,k))
                    np.save(opj(save_path,n,k,'0.npy'),module_state_dict[n][k].numpy())
            write_7z([save_path],save_path+'_7z','0')
            os.system('rm -rf -r {}'.format(save_path))#TODO 安全删除
            os.rename(save_path+'_7z',save_path)
            module_size = os.path.getsize(save_path)  
        return module_size
    def move_module_to(self,device:str):
        """Move every Module in self.module to the given device.
        """
        for k in self.module.keys():
            self.module[k] = self.module[k].to(device)
    def set_module_eval(self,):
        """Set every Module in self.module in evaluation mode.
        """
        for k in self.module.keys():
            self.module[k].eval()
    def set_module_train(self,):
        """Set every Module in self.module in training mode.
        """
        for k in self.module.keys():
            self.module[k].train()
    def set_module_no_grad(self,):
        """Let autograd not record operations on parameters in every Module in self.module
        """
        for k in self.module.keys():
            self.module[k].requires_grad_(False)
    def set_module_grad(self,):
        """Let autograd record operations on parameters in every Module in self.module
        """
        for k in self.module.keys():
            self.module[k].requires_grad_(True)
    def module_parameters(self,) -> List[nn.Parameter]:
        """Return parameters in every Module in self.module.
        """
        parameters = []
        for module in self.module.values():
            parameters.extend(list(module.parameters()))
        return parameters
    def sample_nf(self,coords:torch.Tensor) -> torch.Tensor:
        """Given sampled coords, calculate the corresponding sampled data from the Neural Fileds.
        """
        raise NotImplementedError
    def train(self,save_dir:str):
        """Train every Module in self.module.
        Args:
            save_dir (str): save the trained self.module in this dir. save the files while evaluating performance in this dir.
        """
        raise NotImplementedError
    def compress(self,data_path_list:List[str],save_path:str,) -> torch.Tensor:
        """Compress a batch of data read from the given data_path_list, then save the compressed data into save_path.
        """
        raise NotImplementedError
    def decompress(self,compressed_data_path:str,compressed_data:Dict[str,Union[torch.Tensor,str,dict]],save_path_list:List[str],) -> torch.Tensor:
        """Decompress a batch of data from the given compressed_data or compressed_data_path, then save the decompressed datas into save_path_list.
        """
        raise NotImplementedError
    