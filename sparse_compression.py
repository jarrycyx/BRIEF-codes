import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "./nvidia_condensa"))
sys.path.append(os.path.dirname(__file__))

from main_cyx import NFGR, RandomCubeSampler, RandompointSampler
from condensa.schemes import Compose, Prune, Quantize, NeuronPrune, StructurePrune
import condensa
# from utils.nvidia_condensa import condensa
from einops import rearrange
from omegaconf import OmegaConf
import argparse
from os.path import splitext as ops
from os.path import basename as opb
from os.path import dirname as opd
from os.path import join as opj
from utils.Logger import MyLogger
import torch
import numpy as np
from utils.io import normalize_data, read_data_batch, save_data_batch
import random
from utils.misc import eval_performance, mip_ops, parse_weight
from utils.Typing import ReproducOpt, SingleTaskOpt
import time
from utils import sparse_model_storage
from utils.ModelSave import save_model, load_model, CopyDir
from utils.tool import read_img, save_img
import csv
from utils.io import get_folder_size

class NFGR_MC(NFGR):
    
    def __init__(self, opt, log):
        NFGR.__init__(self, opt)
        self.Log = log

    def prepare_model_compress(self, data_path):
        self.data_path = data_path
        if hasattr(self.opt.Module.phi, "features"):
            delattr(self.opt.Module.phi, "features")
        
        self.phi_features,_ = self.prepare_module(self.parse_param_size(data_path))
        # self.load_module(self.opt.Compress.param.init_net_path)
        load_model(self.module['phi'], self.opt.Compress.param.init_net_path, self.device)
        self.data = read_img(data_path)
        self.current_step = 0
        
    def train_status(self, tag, sideinfos):
        '''
        训练进程中保存结果和模型
        '''
        save_dir = opj(self.Log.logdir, tag)
        module_path = opj(save_dir, "compressed", "module")
        os.makedirs(opd(module_path), exist_ok=True)
        # save compressed data
        # actual_module_size = self.save_module(module_path, self.opt.Compress.module_serializing_method)
        save_model(self.module['phi'], module_path)
        sideinfos_path = opj(save_dir, 'sideinfos.yaml')
        OmegaConf.save(sideinfos, sideinfos_path)
        if self.opt.Compress.decompress:
            decompress_path = opj(save_dir, "decompressed")
            os.makedirs(decompress_path, exist_ok=True)
            decompressed_data = self.decompress(args.p, module_path, sideinfos_path)
            # save data
            if self.opt.Decompress.keep_decompressed:
                decompressed_save_dir = opj(save_dir, 'decompressed')
                os.makedirs(decompressed_save_dir, exist_ok=True)
                decompressed_data_save_path = opj(decompressed_save_dir, opb(
                    ops(self.data_path)[0])+'_decompressed'+ops(self.data_path)[-1])
                save_data_batch(decompressed_data, [
                                decompressed_data_save_path])
            # mip
            if self.opt.Decompress.mip:
                mip_save_dir = opj(save_dir, 'mip')
                os.makedirs(mip_save_dir, exist_ok=True)
                orig_data = read_img(self.data_path)
                if len(orig_data.shape) == 4:
                    mip_ops(orig_data, mip_save_dir, opb(ops(self.data_path)[0]), ops(self.data_path)[-1])
                    mip_ops(decompressed_data, mip_save_dir, opb(ops(self.data_path)[0])+'_decompressed', ops(self.data_path)[-1])
            
            # performance
            performance = eval_performance(self.current_step, orig_data, decompressed_data, self.Log, self.opt.Decompress.aoi_thres_list, 
                    self.opt.Decompress.mse_mape_thres_list, self.opt.Decompress.psnr_thres_list, 
                    self.opt.Decompress.iou_acc_pre_thres_list, self.opt.Decompress.ssim)
                    # eval_performance(steps, orig_data, decompressed_data, self.Log, self.opt.Decompress.aoi,
                    #          self.opt.Decompress.mse, self.opt.Decompress.psnr, self.opt.Decompress.ssim, self.opt.Decompress.iou)
            csv_path = os.path.join(save_dir,'performance.csv')
            if not os.path.exists(csv_path):
                out = open(csv_path, "a")
                csv_writer = csv.writer(out, dialect = "excel")
                csv_writer.writerow(performance.keys())
            row = [performance[key] for key in performance.keys()]
            csv_writer.writerow(row)

            self.current_step += 1
            
        self.move_module_to(self.device)

    def compress_model(self, sparsity=0):
        data = self.data
        # normalize
        weight = parse_weight(data, self.opt.Compress.loss.weight)
        data, sideinfos = normalize_data(data, **self.opt.Normalize)
        data = data.to(self.device)
        # sideinfos
        try:
            sideinfos = sideinfos | {'data_shape': list(
                data.shape), 'phi_features': self.phi_features, 'phi_name': self.phi_name}
        except:
            sideinfos = {**sideinfos, **{'data_shape': list(
                data.shape), 'phi_features': self.phi_features, 'phi_name': self.phi_name}}
        weight_thres_normalized, _ = normalize_data(np.array(
            self.opt.Compress.loss.weight_thres), **self.opt.Normalize, max=sideinfos['max'], min=sideinfos['min'])
        weight_thres_normalized = float(weight_thres_normalized)
        # print(weight_thres_normalized)
        # self.train_status("dense", sideinfos)
        sideinfos_path = opj(self.Log.logdir, 'sideinfos.yaml')
        OmegaConf.save(sideinfos, sideinfos_path)

        MEM = Compose([Prune(1-sparsity)])
        max_steps = self.opt.Compress.param.dynamic_step

        lc = condensa.opt.LC(steps=3,                             # L-C iterations
                             l_optimizer=condensa.opt.lc.Adam,      # L-step sub-optimizer
                             # L-step sub-optimizer parameters
                             # l_optimizer_params={'momentum': 0.95},
                             lr=1e-4,                              # Initial learning rate
                             lr_end=1e-4,                          # Final learning rate
                             mb_iterations_per_l=200,             # Mini-batch iterations per L-step
                             mb_iterations_first_l=400,          # Mini-batch iterations for first L-step
                             mu_init=1e-4,                         # Initial value of `mu`
                             mu_multiplier=1.1,                    # Multiplier for `mu`
                             mu_cap=10000,                         # Maximum value of `mu`
                             debugging_flags={'disable_train_stats': True})

        # sampler
        if self.opt.Compress.sampler.name == 'randompoint':
            print("Using randompoint")
            sampler = RandompointSampler(
                data, weight, self.opt.Compress.coords_mode, self.opt.Compress.sampler.sample_size, max_steps, self.device)
        elif self.opt.Compress.sampler.name == 'randomcube':
            print("Using randomcube")
            sampler = RandomCubeSampler(data, weight, self.opt.Compress.coords_mode, self.opt.Compress.sampler.cube_count,
                                        self.opt.Compress.sampler.cube_len, max_steps, self.device, self.opt.Compress.sampler.gpu_force)
        else:
            raise NotImplementedError

        weight_thres_normalized, _ = normalize_data(np.array(
            self.opt.Compress.loss.weight_thres), **self.opt.Normalize, max=sideinfos['max'], min=sideinfos['min'])
        weight_thres_normalized = float(weight_thres_normalized)
        
        para_num, zero_n, sparsity = sparse_model_storage.calc_stat(self.module["phi"].state_dict())
        # self.Log.log_metrics({"sparsity":sparsity}, 0)
        # self.Log.log_metrics({"parameter number":para_num}, 0)

        compressor_MEM = condensa.Compressor(lc,
                                             MEM,
                                             self.module["phi"],
                                             sampler,
                                             None,
                                             None,
                                             self.loss_func)
        w_MEM = compressor_MEM.run()
        self.module["phi"] = w_MEM
        # self.train_status("sparse", sideinfos)

        # # torch.save(self.module["phi"].state_dict(), self.opt.Compress.param.save_net_path)

        ft = condensa.FineTuner(self.module["phi"], [torch.nn.Linear])
        w_ft = ft.run(epochs=10,
                      lr=self.opt.Compress.lr_phi,
                      lr_end=self.opt.Compress.lr_phi*1e-2,
                      momentum=0.95,
                      weight_decay=0,
                      criterion=self.loss_func,
                      trainloader=sampler,
                      testloader=None,
                      valloader=None,
                      debugging_flags={'disable_train_stats': True})

        self.module["phi"] = w_ft
        # self.train_status("final", sideinfos)
        save_model(self.module["phi"], self.opt.Compress.param.save_net_path)
        
        para_num, zero_n, sparsity = sparse_model_storage.calc_stat(self.module["phi"].state_dict())
        # self.Log.log_metrics({"sparsity":sparsity}, 1)
        # self.Log.log_metrics({"parameter number":para_num}, 1)
        
        # original_size = os.path.getsize(self.opt.Compress.param.save_net_path)
        original_size = get_folder_size(self.opt.Compress.param.save_net_path)
        stored_size = sparse_model_storage.store_sparse_module(self.module["phi"].cpu().state_dict(), self.opt.Compress.param.sparse_file_path)
        
        # self.Log.log_metrics({"File size":original_size}, 0)
        # self.Log.log_metrics({"File size":stored_size}, 1)
        info = 'Sparsity:{},Origin Size:{},Final Size:{}'.format(sparsity, original_size, stored_size)
        print(info)
        
        
        



def reproduc(opt: ReproducOpt):
    """Make experiments reproducible
    """
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)
    torch.backends.cudnn.benchmark = opt.benchmark
    torch.backends.cudnn.deterministic = opt.deterministic


def main():
    global Log
    opt: SingleTaskOpt = OmegaConf.load(args.p)

    # opt.self.Log.project_name += time.strftime("_%Y_%m%d_%H%M%S")
    Log = MyLogger(**opt.Log)
    Log.log_opt(opt)
    reproduc(opt.Reproduc)

    origin_size = get_folder_size(opt.CompressFramework.Compress.param.init_net_path)
    # print(opt.CompressFramework.Compress.param.init_net_path,origin_size)
    args.sparsity = 1-opt.CompressFramework.Compress.param.given_size/origin_size
    print(args.sparsity)

    CompressFramework = NFGR_MC(opt.CompressFramework, log=Log)
    CompressFramework.device = 'cuda' if opt.CompressFramework.Compress.gpu else 'cpu'
    CompressFramework.phi_name = opt.CompressFramework.Module.phi.name

    CompressFramework.prepare_model_compress(opt.Dataset.data_path)

    CompressFramework.compress_model(sparsity=args.sparsity)



if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    parser = argparse.ArgumentParser(
        description='single task for datacompress')
    parser.add_argument('-p', type=str, default=opj(opd(__file__),
                        "data/my_mc_test/adaptive.yaml"), help='yaml file path')
    parser.add_argument('-g', help='availabel gpu list', default='0,1,2,3',
                        type=lambda s: [int(item) for item in s.split(',')])
    parser.add_argument('-sparsity', type=float, default=0, 
                        help='percentage of zero parameters numbers')
    args = parser.parse_args()
    gpu_list = args.g
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(i) for i in gpu_list])
    main()
