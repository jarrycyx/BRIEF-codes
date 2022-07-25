from copy import deepcopy
import logging
import time
from urllib import parse

from utils.TasksManager import Queue, Task
from utils.Typing import CompressFrameworkOpt, ReproducOpt, SingleTaskOpt
import shutil
import sys
from typing import Any, Dict, List, Tuple, Union
from tqdm import tqdm
from utils.misc import alloc_param, configure_lr_scheduler, configure_optimizer, divide_data, eval_performance, merge_divided_data, mip_ops, parse_checkpoints, parse_weight, plot_conv3d_weight, preprocess, reconstruct_flattened
from utils.dataset import create_coords, create_flattened_coords
from utils.Networks import ALL_CALC_PHI_FEATURES, ALL_CALC_PHI_PARAM_COUNT, ALL_CHECK_PARAM_COUNT, get_nnmodule_param_count, init_phi
import random
from utils.io import extract_7z, get_folder_size, invnormalize_data, normalize_data, read_data_batch, save_data_batch, write_7z, read_7z
import numpy as np
import torch
from utils.CompressFramework import _BaseCompressFramerwork
from utils.Logger import MyLogger
import os
from os.path import join as opj
from os.path import dirname as opd
from os.path import basename as opb
from os.path import splitext as ops
import argparse
from omegaconf import OmegaConf
import torch.nn.functional as F
from einops import rearrange
from utils.TasksManager import MIAOCODE, USER, Queue, Task, reminding
from utils.find_weight import dynamic_parse_weight
import copy
from math import sqrt
from utils.adaptive_blocking import OctTree, QuadTree, adaptive_cal_tree, cal_divide_num
from utils.tool import get_type_max, read_img, save_img, get_dimension
import tifffile
from utils.ModelSave import save_model, load_model, CopyDir
import logging
import csv


class RandomCubeSampler:
    def __init__(self, data: torch.Tensor, weight: np.ndarray, coords_mode: str, cube_count: int, cube_len: List[int], sample_count: int, device: str = 'cpu', gpu_force: bool = False) -> None:
        self.sample_count = sample_count
        self.device = device
        # 3d-> dhwc or thwc
        if len(data.shape) == 4:
            self.d, self.h, self.w, self.c = data.shape
            self.coords = create_coords(
                (self.d, self.h, self.w), mode=coords_mode)
            self.data = data
            weight = torch.from_numpy(weight)
            self.weight = weight
            for i in range(3):
                cube_len[i] = min(cube_len[i], data.shape[i])
            self.cube_len_d, self.cube_len_h, self.cube_len_w = cube_len[:3]
            self.cube_count = cube_count
            self.gpu_force = gpu_force
            if gpu_force:
                self.coords = self.coords.to(device)
                self.weight = self.weight.to(device)
            else:
                self.data = self.data.cpu()
            self.data_cubes = self.data.unfold(0, cube_len[0], 1).unfold(
                1, cube_len[1], 1).unfold(2, cube_len[2], 1)
            self.coords_cubes = self.coords.unfold(0, cube_len[0], 1).unfold(
                1, cube_len[1], 1).unfold(2, cube_len[2], 1)
            self.weight_cubes = self.weight.unfold(0, cube_len[0], 1).unfold(
                1, cube_len[1], 1).unfold(2, cube_len[2], 1)
            self.data_cubes = rearrange(
                self.data_cubes, 'dc hc wc c ds hs ws -> (dc hc wc) ds hs ws c')
            self.coords_cubes = rearrange(
                self.coords_cubes, 'dc hc wc c ds hs ws -> (dc hc wc) ds hs ws c')
            self.weight_cubes = rearrange(
                self.weight_cubes, 'dc hc wc c ds hs ws -> (dc hc wc) ds hs ws c')
            self.pop_size = self.data_cubes.shape[0]
        # 2d-> hwc
        elif len(data.shape) == 3:
            self.h, self.w, self.c = data.shape
            self.coords = create_coords((self.h, self.w), mode=coords_mode)
            self.data = data
            weight = torch.from_numpy(weight)
            self.weight = weight
            for i in range(2):
                cube_len[i] = min(cube_len[i], data.shape[i])
            self.cube_len_h, self.cube_len_w = cube_len[:2]
            self.cube_count = cube_count
            self.gpu_force = gpu_force
            if gpu_force:
                self.coords = self.coords.to(device)
                self.weight = self.weight.to(device)
            else:
                self.data = self.data.cpu()
            self.data_cubes = self.data.unfold(0, cube_len[0], 1).unfold(
                1, cube_len[1], 1)
            self.coords_cubes = self.coords.unfold(0, cube_len[0], 1).unfold(
                1, cube_len[1], 1)
            self.weight_cubes = self.weight.unfold(0, cube_len[0], 1).unfold(
                1, cube_len[1], 1)
            self.data_cubes = rearrange(
                self.data_cubes, 'hc wc c hs ws-> (hc wc) hs ws c')
            self.coords_cubes = rearrange(
                self.coords_cubes, 'hc wc c hs ws-> (hc wc) hs ws c')
            self.weight_cubes = rearrange(
                self.weight_cubes, 'hc wc c hs ws-> (hc wc) hs ws c')
            self.pop_size = self.data_cubes.shape[0]
        else:
            raise NotImplementedError

    def __len__(self):
        return self.sample_count

    def __iter__(self,):
        self.index = 0
        return self

    def __next__(self,) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.index < self.__len__():
            sampled_idxs = torch.randint(0, self.pop_size, (self.cube_count,))
            sampled_coords = self.coords_cubes[sampled_idxs, :]
            sampled_data = self.data_cubes[sampled_idxs, :]
            sampled_weight = self.weight_cubes[sampled_idxs, :]
            if not self.gpu_force:
                sampled_coords = sampled_coords.to(self.device)
                sampled_data = sampled_data.to(self.device)
                sampled_weight = sampled_weight.to(self.device)
            self.index += 1
            return sampled_coords, sampled_data, sampled_weight
        else:
            raise StopIteration


class RandompointSampler:
    def __init__(self, data: torch.Tensor, weight: np.ndarray, coords_mode: str, sample_size: int, sample_count: int, device: str = 'cpu') -> None:
        self.sample_size = sample_size
        self.sample_count = sample_count
        if len(data.shape) == 4:
            d, h, w, c = data.shape
            self.coords = create_flattened_coords(
                (d, h, w), mode=coords_mode).to(device)
            self.data = rearrange(data, 'd h w c-> (d h w) c')
            weight = torch.from_numpy(weight).to(device)
            self.weight = rearrange(weight, 'd h w c-> (d h w) c')
            self.pop_size = d*h*w
        elif len(data.shape) == 3:
            h, w, c = data.shape
            self.coords = create_flattened_coords(
                (h, w), mode=coords_mode).to(device)
            self.data = rearrange(data, 'h w c-> (h w) c')
            weight = torch.from_numpy(weight).to(device)
            self.weight = rearrange(weight, 'h w c-> (h w) c')
            self.pop_size = h*w
        else:
            raise NotImplementedError

    def __len__(self):
        return self.sample_count

    def __iter__(self,):
        self.index = 0
        return self

    def __next__(self,) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.index < self.__len__():
            sampled_idxs = torch.randint(0, self.pop_size, (self.sample_size,))
            sampled_coords = self.coords[sampled_idxs, :]
            sampled_data = self.data[sampled_idxs, :]
            sampled_weight = self.weight[sampled_idxs, :]
            self.index += 1
            return sampled_coords, sampled_data, sampled_weight
        else:
            raise StopIteration


class NFGR(_BaseCompressFramerwork):
    def __init__(self, opt: CompressFrameworkOpt,) -> None:
        super().__init__(opt)
        self.set_loss()
        self.half = opt.Compress.half
        self.device = 'cuda' if opt.Compress.gpu else 'cpu'

    def set_loss(self):
        loss_name = self.opt.Compress.loss.name
        if 'smoothl1' in loss_name:
            beta = self.opt.Compress.loss.beta

        def datal2(data_gt, data_hat, weight, weight_thres):
            loss = F.mse_loss(data_hat, data_gt, reduction='none')
            if weight_thres:
                weight[data_hat <= weight_thres] = 1
            loss = loss*weight
            loss = loss.mean()
            return loss

        def datasmoothl1(data_gt, data_hat, weight, weight_thres):
            loss = F.smooth_l1_loss(
                data_hat, data_gt, beta=beta, reduction='none')
            if weight_thres:
                weight[data_hat <= weight_thres] = 1
            loss = loss*weight
            loss = loss.mean()
            return loss
        if loss_name == 'datal2':
            self.loss_func = datal2
        elif loss_name == 'datasmoothl1':
            self.loss_func = datasmoothl1
        else:
            raise NotImplementedError

    def parse_param_size(self, data_path: str = None):
        if (self.opt.Compress.param.given_size > 0 and self.opt.Compress.param.filesize_ratio > 0) or (self.opt.Compress.param.given_size == 0 and self.opt.Compress.param.filesize_ratio == 0):
            raise ValueError('There can only be one arg to be used')
        if self.opt.Compress.param.given_size > 0:
            param_size = self.opt.Compress.param.given_size
        else:
            param_size = os.path.getsize(
                data_path)/self.opt.Compress.param.filesize_ratio
        return param_size

    def init_module(self):
        self.module['phi'] = init_phi(self.opt.Module.phi)
        if self.half:
            self.module['phi'].half()

    @staticmethod
    def estimate_module_size(ideal_module_size: float, opt: CompressFrameworkOpt):
        # phi
        if opt.Compress.half:
            ideal_phi_param_count = ideal_module_size / 2.
        else:
            ideal_phi_param_count = ideal_module_size / 4.
        if opt.Module.phi.name == 'SIREN_Pyramid':
            check_param_count = ALL_CHECK_PARAM_COUNT['SIREN_Pyramid']
            if not check_param_count(param_count=ideal_phi_param_count, **opt.Module.phi):
                opt.Module.phi.name = 'SIRENFT'
                opt.Module.phi.features_plus = opt.Module.phi.features_dis
        if opt.Module.phi.name == 'SIRENFT':
            check_param_count = ALL_CHECK_PARAM_COUNT['SIRENFT']
            if not check_param_count(param_count=ideal_phi_param_count, **opt.Module.phi):
                opt.Module.phi.name = 'SIREN'
        if opt.Module.phi.name == 'SIRENPS':
            check_param_count = ALL_CHECK_PARAM_COUNT['SIRENPS']
            if not check_param_count(param_count=ideal_phi_param_count, **opt.Module.phi):
                opt.Module.phi.name = 'SIREN'
        #### xtx: end
        calc_phi_features = ALL_CALC_PHI_FEATURES[opt.Module.phi.name]
        phi_features = calc_phi_features(
            param_count=ideal_phi_param_count, **opt.Module.phi)
        calc_phi_param_count = ALL_CALC_PHI_PARAM_COUNT[opt.Module.phi.name]
        actual_phi_param_count = calc_phi_param_count(
            features=phi_features, **opt.Module.phi)
        # print(ideal_phi_param_count,actual_phi_param_count)
        # actual_module_size
        if opt.Compress.half:
            theory_module_size = actual_phi_param_count * 2.
        else:
            theory_module_size = actual_phi_param_count * 4.
        return phi_features, actual_phi_param_count, theory_module_size

    def prepare_module(self, ideal_module_size: float):
        device = 'cuda' if self.opt.Compress.gpu else 'cpu'
        # calc features
        phi_features, actual_phi_param_count, theory_module_size = NFGR.estimate_module_size(
            ideal_module_size, self.opt)
        error_rate = (theory_module_size-ideal_module_size)/ideal_module_size
        # print(ideal_module_size,theory_module_size)
        if abs(error_rate) > 0.05:
            logging.warning('Error_rate={:.3f}>0.05! ideal_module_size={} theory_module_size={} '.format(
                error_rate, ideal_module_size, theory_module_size))
        # assert abs(error_rate)<=0.15,'Error_rate={:.3f}>0.05! ideal_module_size={} theory_module_size={} '.format(error_rate,ideal_module_size,theory_module_size)
        self.opt.Module.phi.features = phi_features
        self.init_module()
        assert get_nnmodule_param_count(
            self.module['phi']) == actual_phi_param_count, 'calc_phi_param_count mismatch get_nnmodule_param_count !'
        self.move_module_to(device)
        return phi_features, theory_module_size

    def sample_nf(self, coords: torch.Tensor) -> torch.Tensor:
        data_hat = self.module['phi'].forward(coords)
        return data_hat

    @staticmethod
    def decompress(opt_path: str, module_path: str, sideinfos_path: str) -> np.ndarray:
        opt: SingleTaskOpt = OmegaConf.load(opt_path)
        DeCF = NFGR(opt.CompressFramework)
        sideinfos = OmegaConf.load(sideinfos_path)
        data_shape = sideinfos['data_shape']
        phi_features = sideinfos['phi_features']
        phi_name = sideinfos['phi_name']
        DeCF.opt.Module.phi.features = phi_features
        DeCF.opt.Module.phi.name = phi_name
        DeCF.init_module()
        load_model(DeCF.module['phi'], module_path, DeCF.device)
        # device
        decompress_device = 'cuda' if DeCF.opt.Decompress.gpu else 'cpu'
        # half
        if DeCF.half:
            DeCF.module['phi'].half()
        DeCF.move_module_to(decompress_device)
        # sample from nf
        decompressed_data = reconstruct_flattened(data_shape, DeCF.opt.Decompress.sample_size, DeCF.sample_nf,
                                                  device=decompress_device, half=DeCF.half, coords_mode=DeCF.opt.Compress.coords_mode).float()
        decompressed_data = decompressed_data.cpu()
        decompressed_data = invnormalize_data(
            decompressed_data, sideinfos, **DeCF.opt.Normalize)
        decompressed_data = preprocess(decompressed_data, DeCF.opt.Decompress.postprocess.denoise.level,
                                       DeCF.opt.Decompress.postprocess.denoise.close, DeCF.opt.Decompress.postprocess.clip)
        return decompressed_data

    def decompress_divide(self, original_opt, orig_sideinfos_path: str, module_save_dir: str, sideinfos_save_dir: str) -> np.ndarray:
        # orig sideinfos
        orig_sideinfos = OmegaConf.load(orig_sideinfos_path)
        data_shape = orig_sideinfos['data_shape']
        decompressed_data_chunk_list = []
        for chunk_name in os.listdir(module_save_dir):
            if len(data_shape) == 4:
                d = [int(i) for i in chunk_name.split('-')[0].split('_')[1:]]
                h = [int(i) for i in chunk_name.split('-')[1].split('_')[1:]]
                w = [int(i) for i in chunk_name.split('-')[2].split('_')[1:]]
            elif len(data_shape) == 3:
                h = [int(i) for i in chunk_name.split('-')[0].split('_')[1:]]
                w = [int(i) for i in chunk_name.split('-')[1].split('_')[1:]]
            module_path = opj(module_save_dir, chunk_name, 'module')
            sideinfos_path = opj(sideinfos_save_dir,
                                 chunk_name, 'sideinfos.yaml')
            decompressed_data = NFGR.decompress(
                original_opt, module_path, sideinfos_path)
            chunk_dict = {'data': decompressed_data,
                          'name': chunk_name, 'h': h, 'w': w}
            if len(data_shape) == 4:
                chunk_dict['d'] = d
            decompressed_data_chunk_list.append(chunk_dict)
        decompressed_data = merge_divided_data(
            decompressed_data_chunk_list, data_shape)
        return decompressed_data

    def compress(self, data_path: str):
        data = read_img(data_path)
        # size limit: size should not be larger than 80*80*80*1
        cube_len = self.opt.Compress.sampler.cube_len
        if len(data.shape) == 4:
            cube_len = cube_len[0]*cube_len[1]*cube_len[2]
        elif len(data.shape) == 3:
            cube_len = cube_len[1]*cube_len[2]
        if self.opt.Compress.sampler.name == 'randomcube' and min(data.size, cube_len) > 80*80*80*1:
            logging.warning('Chunk size should not be larger than 80*80*80*1!')
            self.opt.Compress.sampler.name = 'randompoint'
        data_preprocessed = preprocess(data, self.opt.Compress.preprocess.denoise.level,
                                       self.opt.Compress.preprocess.denoise.close, self.opt.Compress.preprocess.clip)
        data_preprocessed_save_path = opj(Log.logdir, opb(
            ops(data_path)[0])+'_preprocessed'+ops(data_path)[-1])
        save_img(data_preprocessed_save_path, data_preprocessed)
        weight = parse_weight(data_preprocessed, self.opt.Compress.loss.weight)
        # normalize
        data, sideinfos = normalize_data(
            data_preprocessed, **self.opt.Normalize)
        data = data.to(self.device)
        # module
        if self.opt.Compress.param.init_net_path == 'none':
            ideal_param_size = self.parse_param_size(data_path)
            phi_features, theory_module_size = self.prepare_module(
                ideal_param_size)
        else:
            ideal_param_size = self.parse_param_size(data_path)
            phi_features, theory_module_size = self.prepare_module(
                ideal_param_size)

            module_path = self.opt.Compress.param.init_net_path
            load_model(self.module['phi'], module_path, self.device)
        # sideinfos
        try:
            sideinfos = sideinfos | {'data_shape': list(
                data.shape), 'phi_features': phi_features, 'phi_name': self.opt.Module.phi.name}
        except:
            sideinfos = {**sideinfos, **{'data_shape': list(
                data.shape), 'phi_features': phi_features, 'phi_name': self.opt.Module.phi.name}}
        # sampler
        max_steps = self.opt.Compress.max_steps
        if self.opt.Compress.sampler.name == 'randompoint':
            sampler = RandompointSampler(
                data, weight, self.opt.Compress.coords_mode, self.opt.Compress.sampler.sample_size, max_steps, self.device)
        elif self.opt.Compress.sampler.name == 'randomcube':
            sampler = RandomCubeSampler(data, weight, self.opt.Compress.coords_mode, self.opt.Compress.sampler.cube_count,
                                        self.opt.Compress.sampler.cube_len, max_steps, self.device, self.opt.Compress.sampler.gpu_force)
        else:
            raise NotImplementedError
        # compressing
        checkpoints = parse_checkpoints(
            self.opt.Compress.checkpoints, max_steps)
        optimizer_phi = configure_optimizer(
            self.module['phi'].parameters(), self.opt.Compress.optimizer_name_phi, self.opt.Compress.lr_phi)
        lr_scheduler_phi = configure_lr_scheduler(
            optimizer_phi, self.opt.Compress.lr_scheduler_phi)
        pbar = tqdm(sampler, desc='Compressing', leave=True, file=sys.stdout)
        loss_log_freq = self.opt.Compress.loss_log_freq
        assert self.opt.Compress.loss.weight_thres <= get_type_max(
            data_preprocessed), 'The weight threshold should be less than the data maximum!'
        weight_thres_normalized, _ = normalize_data(np.array(
            self.opt.Compress.loss.weight_thres), **self.opt.Normalize, max=sideinfos['max'], min=sideinfos['min'])
        weight_thres_normalized = float(weight_thres_normalized)
        for steps, (sampled_coords, sampled_data, sampled_weight) in enumerate(pbar):
            steps += 1
            optimizer_phi.zero_grad()
            if self.half:
                self.module['phi'].half()
                sampled_data = sampled_data.half()
                data_hat = self.module['phi'].forward(sampled_coords.half())
            else:
                data_hat = self.module['phi'].forward(sampled_coords)
            loss = self.loss_func(sampled_data, data_hat,
                                  sampled_weight, weight_thres_normalized)
            loss.backward()
            if self.half:
                self.module['phi'].float()
            optimizer_phi.step()
            lr_scheduler_phi.step()
            pbar.set_postfix_str("loss={:.6f}".format(loss.item()))
            if steps % loss_log_freq == 0:
                Log.log_metrics({'loss': loss.item()}, steps)
            if steps in checkpoints:
                save_dir_current_steps = opj(
                    Log.logdir, 'steps{}'.format(steps))
                os.makedirs(save_dir_current_steps, exist_ok=True)
                # dynamic allo param save
                if self.opt.Compress.param.save_net_path == 'none':
                    compressed_save_dir = opj(
                        save_dir_current_steps, 'compressed')
                    os.makedirs(compressed_save_dir, exist_ok=True)
                    module_path = opj(compressed_save_dir, 'module')
                    sideinfos_path = opj(compressed_save_dir, 'sideinfos.yaml')
                else:
                    module_path = self.opt.Compress.param.save_net_path
                    sideinfos_path = self.opt.Compress.param.sideinfos_path
                OmegaConf.save(sideinfos, sideinfos_path)
                save_model(self.module['phi'], module_path, self.device)
                actual_module_size = get_folder_size(module_path)
                theory_compressed_data_bytes = os.path.getsize(
                    sideinfos_path) + theory_module_size
                actual_compressed_data_bytes = os.path.getsize(
                    sideinfos_path) + actual_module_size
                orig_data_bytes = os.path.getsize(data_path)
                Log.log_metrics({'data_bytes/orig': orig_data_bytes, 'data_bytes/theory_compressed': theory_compressed_data_bytes, 'data_bytes/actual_compressed': actual_compressed_data_bytes,
                                 'compress_ratio/theory': orig_data_bytes/theory_compressed_data_bytes, 'compress_ratio/actual': orig_data_bytes/actual_compressed_data_bytes}, steps)
                # decompress
                if self.opt.Compress.decompress:
                    decompressed_data = self.decompress(
                        args.p, module_path, sideinfos_path)
                    # save data
                    if self.opt.Decompress.keep_decompressed:
                        decompressed_save_dir = opj(
                            save_dir_current_steps, 'decompressed')
                        os.makedirs(decompressed_save_dir, exist_ok=True)
                        decompressed_data_save_path = opj(decompressed_save_dir, opb(
                            ops(data_path)[0])+'_decompressed'+ops(data_path)[-1])
                        save_img(decompressed_data_save_path,
                                 decompressed_data)
                    # mip
                    if self.opt.Decompress.mip:
                        mip_save_dir = opj(save_dir_current_steps, 'mip')
                        os.makedirs(mip_save_dir, exist_ok=True)
                        orig_data = read_img(data_path)
                        if len(orig_data.shape) == 4:
                            mip_ops(orig_data, mip_save_dir, opb(
                                ops(data_path)[0]), ops(data_path)[-1])
                            mip_ops(decompressed_data, mip_save_dir, opb(
                                ops(data_path)[0])+'_decompressed', ops(data_path)[-1])
                    # performance
                    performance = eval_performance(steps, orig_data, decompressed_data, Log, self.opt.Decompress.aoi_thres_list,
                                                   self.opt.Decompress.mse_mape_thres_list, self.opt.Decompress.psnr_thres_list,
                                                   self.opt.Decompress.iou_acc_pre_thres_list, self.opt.Decompress.ssim)
                    performance['loss'] = loss.cpu().detach().numpy()
                    csv_path = os.path.join(Log.logdir, 'performance.csv')
                    if not os.path.exists(csv_path):
                        out = open(csv_path, "a")
                        csv_writer = csv.writer(out, dialect="excel")
                        csv_writer.writerow(performance.keys())
                    row = [performance[key] for key in performance.keys()]
                    csv_writer.writerow(row)
                self.move_module_to(self.device)
        Log.close()

    def adaptive_chunk(self, img_path, param_size, divide_type):
        _, max_level, min_level, prune_var_thres, prune_mean_thres, Nb, Type = divide_type.split(
            '_')
        max_level, min_level, prune_var_thres, prune_mean_thres, Nb, Type = int(
            max_level), int(min_level), int(prune_var_thres), int(prune_mean_thres), int(Nb), int(Type)
        tree, save_data, dimension = adaptive_cal_tree(
            img_path, param_size, var_thr=prune_var_thres, e_thr=prune_mean_thres, maxl=max_level, minl=min_level, Nb=Nb, Type=Type)
        patch_list = tree.get_active()
        # adaptive chunks
        data_chunk_list = []
        data = read_img(img_path)
        for patch in patch_list:
            if dimension == 3:
                info = {'data': data[patch.z:patch.z+patch.d, patch.y:patch.y+patch.h, patch.x:patch.x+patch.w], 'd': [
                    patch.z, patch.z+patch.d-1], 'h': [patch.y, patch.y+patch.h-1], 'w': [patch.x, patch.x+patch.w-1]}
            elif dimension == 2:
                info = {'data': data[patch.y:patch.y+patch.h, patch.x:patch.x+patch.w],
                        'h': [patch.y, patch.y+patch.h-1], 'w': [patch.x, patch.x+patch.w-1]}
            data_chunk_list.append(info)
        for data_chunk in data_chunk_list:
            data_chunk['total_size'] = data.size
            data_chunk['size'] = data_chunk['data'].size
            if dimension == 3:
                data_chunk['name'] = 'd_{}_{}-h_{}_{}-w_{}_{}'.format(
                    *data_chunk['d'], *data_chunk['h'], *data_chunk['w'])
            elif dimension == 2:
                data_chunk['name'] = 'h_{}_{}-w_{}_{}'.format(
                    *data_chunk['h'], *data_chunk['w'])
        print('total numbers of the chunks: '+str(len(data_chunk_list)))
        return data_chunk_list, save_data

    def compress_divide(self, data_path: str, opt: SingleTaskOpt):
        # temp_script_path avoids modifying during training cause error
        # if 'temp' in __file__:
        #     raise NotImplemented
        temp_script_path = opj(opd(__file__), 'temp_{}'.format(
            time.strftime("_%Y_%m%d_%H%M%S"))+opb(__file__))
        shutil.copy(__file__, temp_script_path)
        # load and preprocess data
        data = read_img(data_path)
        assert len(data.shape) == self.opt.Module.phi.coords_channel + \
            1, "The data dimension {} is inconsistent with the neural network input {}!".format(
                len(data.shape)-1, self.opt.Module.phi.coords_channel)
        assert data.shape[-1] == self.opt.Module.phi.data_channel, "The number of data channels {} is inconsistent with the output of neural network {}!".format(
            data.shape[-1], self.opt.Module.phi.data_channel)
        orig_sideinfos = {'data_shape': list(data.shape)}
        data_preprocessed = preprocess(data, self.opt.Compress.preprocess.denoise.level,
                                       self.opt.Compress.preprocess.denoise.close, self.opt.Compress.preprocess.clip)
        data_preprocessed_save_path = opj(Log.logdir, opb(
            ops(data_path)[0])+'_preprocessed'+ops(data_path)[-1])
        save_img(data_preprocessed_save_path, data_preprocessed)
        # calculate parameter size
        param_size = self.parse_param_size(data_path)
        # 分配dynamic_ratio%的参数用于动态分配
        # if opt.CompressFramework.Compress.param.dynamic_iter != 0:
        #     dynamic_param_size = param_size*self.opt.Compress.param.dynamic_ratio
        # param_size = param_size - dynamic_param_size
        # adaptive blocking
        if 'adaptive' in self.opt.Compress.divide.divide_type:
            _, max_level, min_level, prune_var_thres, prune_mean_thres, Nb, Type = self.opt.Compress.divide.divide_type.split(
                '_')
            if int(Nb) < 8:
                logging.warning('Nb<8!')
                self.opt.Compress.divide.divide_type = f'adaptotal_-1_-1_-1_{Nb}_1'
            else:
                data_chunk_list, divide_img = self.adaptive_chunk(
                    data_preprocessed_save_path, param_size, self.opt.Compress.divide.divide_type)
                divide_path = os.path.join(
                    Log.logdir, 'adaptive' + os.path.splitext(data_preprocessed_save_path)[-1])
        if 'adaptotal' in self.opt.Compress.divide.divide_type:
            _, d_num, h_num, w_num, Nb, Type = self.opt.Compress.divide.divide_type.split(
                '_')
            d_num, h_num, w_num, Nb, Type = int(d_num), int(
                h_num), int(w_num), int(Nb), int(Type)
            if len(data.shape) == 3:
                if h_num == -1 or w_num == -1:
                    d_num, h_num, w_num = cal_divide_num(
                        1, data.shape[0], data.shape[1], Nb, Type, param_size)
                print(
                    f'2D Image Divide Num:{h_num}-{w_num}, Chunk Size:{data.shape[0]/h_num}-{data.shape[1]/w_num}')
            elif len(data.shape) == 4:
                if d_num == -1 or h_num == -1 or w_num == -1:
                    d_num, h_num, w_num = cal_divide_num(
                        data.shape[0], data.shape[1], data.shape[2], Nb, Type, param_size)
                print(
                    f'3D Image Divide Num:{d_num}-{h_num}-{w_num}, Chunk Size:{data.shape[0]/d_num}-{data.shape[1]/h_num}-{data.shape[2]/w_num}')
            data_chunk_list, divide_img = divide_data(
                data_preprocessed, f'total_{d_num}_{h_num}_{w_num}')
            divide_path = os.path.join(
                Log.logdir, 'every' + os.path.splitext(data_preprocessed_save_path)[-1])
        elif 'every' in self.opt.Compress.divide.divide_type or 'total' in self.opt.Compress.divide.divide_type:
            data_chunk_list, divide_img = divide_data(
                data_preprocessed, self.opt.Compress.divide.divide_type)
            divide_path = os.path.join(
                Log.logdir, 'every' + os.path.splitext(data_preprocessed_save_path)[-1])
        save_img(divide_path, divide_img)
        orig_sideinfos['chunks_numbers'] = len(data_chunk_list)
       # allocate parameter to the divided blocks
        data_chunk_list = alloc_param(
            data_chunk_list, param_size, self.opt.Compress.divide.param_alloc, self.opt.Compress.divide.param_size_thres)
        dynamic_chunk.init_patches(data_chunk_list)
        chunk_name_list = [data_chunk['name']
                           for data_chunk in data_chunk_list]
        # assign task
        task_pdir = 'subexps'
        exception_opt = self.opt.Compress.divide.exception
        if exception_opt == 'none':
            exception_opt = {}
        task_list = []
        data_chunk_save_dir = opj(opd(Log.logdir), 'data_chunks')
        task_opt_save_dir = opj(opd(Log.logdir), 'task_opts')
        os.makedirs(data_chunk_save_dir, exist_ok=True)
        os.makedirs(task_opt_save_dir, exist_ok=True)
        checkpoints = parse_checkpoints(
            self.opt.Compress.checkpoints, self.opt.Compress.max_steps)
        for data_chunk in data_chunk_list:
            task_opt = deepcopy(opt)
            task_opt.Log.logdir = opj(
                opd(Log.logdir), task_pdir, data_chunk['name'])
            task_opt.Log.task_name = data_chunk['name']+'-static'
            task_opt.Log.stdlog = True
            task_opt.Log.tensorboard = False
            task_opt.CompressFramework.Compress.divide.divide_type = 'none'
            task_opt.CompressFramework.Compress.param.filesize_ratio = 0
            task_opt.CompressFramework.Compress.param.given_size = data_chunk['param_size']
            task_opt.CompressFramework.Compress.preprocess.denoise.level = 0
            task_opt.CompressFramework.Compress.preprocess.denoise.close = False
            task_opt.CompressFramework.Compress.decompress = False
            # size limit: size should not be larger than 80*80*80*1
            cube_len = task_opt.CompressFramework.Compress.sampler.cube_len
            if len(data.shape) == 4:
                cube_len = cube_len[0]*cube_len[1]*cube_len[2]
            elif len(data.shape) == 3:
                cube_len = cube_len[1]*cube_len[2]
            if task_opt.CompressFramework.Compress.sampler.name == 'randomcube' and min(data_chunk['size'], cube_len) > 80*80*80*1:
                logging.warning(
                    'Chunk size should not be larger than 80*80*80*1!')
                task_opt.CompressFramework.Compress.sampler.name = 'randompoint'
            # theory_module_size
            _, _, data_chunk['theory_module_size'] = NFGR.estimate_module_size(
                data_chunk['param_size'], task_opt.CompressFramework)
            # save data chunk
            data_chunk_save_path = opj(
                data_chunk_save_dir, data_chunk['name']+ops(data_path)[-1])
            save_img(data_chunk_save_path, data_chunk['data'])
            # save task opt
            task_opt.Dataset.data_path = data_chunk_save_path
            if data_chunk['name'] in exception_opt.keys():
                task_opt = OmegaConf.merge(
                    task_opt, exception_opt[data_chunk['name']])
            task_opt_yaml_path = opj(
                task_opt_save_dir, task_opt.Log.task_name+'.yaml')
            OmegaConf.save(task_opt, task_opt_yaml_path)
            # instance Task
            command = "python {} -p {}".format(temp_script_path,
                                               task_opt_yaml_path)
            task_list.append(Task(command, task_opt.Log.task_name, gpucost,
                             cpucost, cost_variable=str(data_chunk['param_size'])))
        try:
            queue = Queue(task_list, gpu_list)
            queue.init_sharecost_dict()
            queue.start(time_interval, max_task, log=False,
                        remind=False, debug=debug, autogpu=False)
        except:
            pass

        # compress sevenzip decompress eval
        orig_data_bytes = os.path.getsize(data_path)
        for steps in checkpoints:
            # orig data sideinfos
            save_dir_current_steps = opj(Log.logdir, 'steps{}'.format(steps))
            os.makedirs(save_dir_current_steps, exist_ok=True)
            # save compressed data
            compressed_save_dir = opj(save_dir_current_steps, 'compressed')
            os.makedirs(compressed_save_dir, exist_ok=True)
            # save orig sideinfos
            orig_sideinfos_path = opj(compressed_save_dir, 'sideinfos.yaml')
            OmegaConf.save(orig_sideinfos, orig_sideinfos_path)
            # save module sideinfos
            module_save_dir = opj(compressed_save_dir, 'module')
            sideinfos_save_dir = opj(compressed_save_dir, 'sideinfos')
            for chunk_name in chunk_name_list:
                chunk_sideinfos_save_dir = opj(sideinfos_save_dir, chunk_name)
                os.makedirs(chunk_sideinfos_save_dir, exist_ok=True)
                shutil.copy(opj(opd(Log.logdir), task_pdir, chunk_name, 'steps{}'.format(
                    steps), 'compressed', 'sideinfos.yaml'), opj(chunk_sideinfos_save_dir, 'sideinfos.yaml'))
            for chunk_name in chunk_name_list:
                chunk_module_save_dir = opj(module_save_dir, chunk_name)
                os.makedirs(chunk_module_save_dir, exist_ok=True)
                origin_dir = opj(opd(Log.logdir), task_pdir, chunk_name, 'steps{}'.format(
                    steps), 'compressed', 'module')
                new_dir = opj(chunk_module_save_dir, 'module')
                CopyDir(origin_dir, new_dir)
            actual_compressed_data_bytes = get_folder_size(compressed_save_dir)
            theory_compressed_data_bytes = get_folder_size(sideinfos_save_dir) + sum(
                [data_chunk['theory_module_size'] for data_chunk in data_chunk_list])
            Log.log_metrics({'data_bytes/orig': orig_data_bytes, 'data_bytes/theory_compressed': theory_compressed_data_bytes, 'data_bytes/actual_compressed': actual_compressed_data_bytes,
                             'compress_ratio/theory': orig_data_bytes/theory_compressed_data_bytes, 'compress_ratio/actual': orig_data_bytes/actual_compressed_data_bytes}, steps)
            # decompress
            if self.opt.Compress.decompress:
                decompressed_data = self.decompress_divide(
                    args.p, orig_sideinfos_path, module_save_dir, sideinfos_save_dir)
                # save data
                if self.opt.Decompress.keep_decompressed:
                    decompressed_save_dir = opj(
                        save_dir_current_steps, 'decompressed')
                    os.makedirs(decompressed_save_dir, exist_ok=True)
                    decompressed_data_save_path = opj(decompressed_save_dir, opb(
                        ops(data_path)[0])+'_decompressed'+ops(data_path)[-1])
                    save_img(decompressed_data_save_path, decompressed_data)
                # mip
                if self.opt.Decompress.mip:
                    orig_data = read_img(data_path)
                    if len(orig_data.shape) == 4:
                        mip_save_dir = opj(save_dir_current_steps, 'mip')
                        os.makedirs(mip_save_dir, exist_ok=True)
                        mip_ops(orig_data, mip_save_dir, opb(
                            ops(data_path)[0]), ops(data_path)[-1])
                        mip_ops(decompressed_data, mip_save_dir, opb(
                            ops(data_path)[0])+'_decompressed', ops(data_path)[-1])
                # performance
                performance = eval_performance(steps, orig_data, decompressed_data, Log, self.opt.Decompress.aoi_thres_list,
                                               self.opt.Decompress.mse_mape_thres_list, self.opt.Decompress.psnr_thres_list,
                                               self.opt.Decompress.iou_acc_pre_thres_list, self.opt.Decompress.ssim)

                csv_path = os.path.join(Log.logdir, 'performance.csv')
                if not os.path.exists(csv_path):
                    out = open(csv_path, "a")
                    csv_writer = csv.writer(out, dialect="excel")
                    csv_writer.writerow(performance.keys())
                row = [performance[key] for key in performance.keys()]
                csv_writer.writerow(row)

        #################### 根据loss动态分配参数量 ####################
        dynamic_task_list = []
        dynamic_task_pdir = opj(opd(Log.logdir), 'dynamic_exps')
        dynamic_data_chunk_save_dir = opj(opd(Log.logdir), 'data_chunks')
        dynamic_task_opt_save_dir = opj(opd(Log.logdir), 'dynamic_task_opts')
        if opt.CompressFramework.Compress.param.dynamic_iter != 0:
            os.makedirs(dynamic_task_pdir)
            # os.makedirs(dynamic_data_chunk_save_dir)  # 之前已经创建过了，就不用再创建了
            os.makedirs(dynamic_task_opt_save_dir)
        # 初始化模型
        for chunk in dynamic_chunk.patches:
            chunk.last_net_path = opj(module_save_dir, chunk.name, 'module')
            chunk.init_net_path = opj(module_save_dir, chunk.name, 'module')
            chunk.save_net_path = opj(module_save_dir, chunk.name, 'module')
            chunk.opt = deepcopy(opt)
        # 动态训练
        dynamic_iter = opt.CompressFramework.Compress.param.dynamic_iter
        dynamic_step = opt.CompressFramework.Compress.param.dynamic_step
        dynamic_num = int(len(dynamic_chunk.patches) *
                          opt.CompressFramework.Compress.param.dynamic_num_ratio)
        for step in range(dynamic_iter):
            os.makedirs(opj(dynamic_task_pdir, str(step)))
            # update loss
            dynamic_chunk.update_loss(decompressed_data)
            # update module
            for chunk in dynamic_chunk.patches:
                os.makedirs(opj(dynamic_task_pdir, str(step), chunk.name))
                chunk.last_net_path = chunk.save_net_path
                chunk.init_net_path = opj(dynamic_task_pdir, str(
                    step), chunk.name, 'init_module')
                chunk.save_net_path = opj(dynamic_task_pdir, str(
                    step), chunk.name, 'save_module')
                chunk.sideinfos_path = opj(dynamic_task_pdir, str(
                    step), chunk.name, 'sideinfos.yaml')
            dynamic_chunk.update_net(
                dynamic_num, self.opt.Compress.param.dynamic_ratio)
            # train
            for chunk in dynamic_chunk.patches:
                chunk.opt = deepcopy(opt)
                chunk.opt.Log.logdir = opj(dynamic_task_pdir, str(
                    step), chunk.name)
                chunk.opt.Log.task_name = chunk.name + \
                    '-dynamic{}'.format(step)
                if chunk.sparse == False:
                    chunk.opt.Log.stdlog = True
                chunk.opt.Log.tensorboard = False
                chunk.opt.CompressFramework.Compress.divide.divide_type = 'none'
                chunk.opt.CompressFramework.Compress.checkpoints = 'none'
                chunk.opt.CompressFramework.Compress.param.filesize_ratio = 0
                chunk.opt.CompressFramework.Compress.preprocess.denoise.level = 0
                chunk.opt.CompressFramework.Compress.preprocess.denoise.close = False
                chunk.opt.CompressFramework.Compress.decompress = False

                chunk.opt.CompressFramework.Compress.param.dynamic = True             
                chunk.opt.CompressFramework.Compress.max_steps = dynamic_step          
                chunk.opt.CompressFramework.Compress.param.given_size = chunk.param_size
                chunk.opt.CompressFramework.Compress.param.last_net_path = chunk.last_net_path
                chunk.opt.CompressFramework.Compress.param.init_net_path = chunk.init_net_path
                chunk.opt.CompressFramework.Compress.param.save_net_path = chunk.save_net_path
                chunk.opt.CompressFramework.Compress.param.sideinfos_path = chunk.sideinfos_path
                chunk.opt.CompressFramework.Compress.param.sparse_file_path = chunk.sparse_file_path
                _, _, chunk.actual_module_size = NFGR.estimate_module_size(
                    chunk.param_size, chunk.opt.CompressFramework)
                # save data chunk
                data_chunk_save_path = opj(
                    dynamic_data_chunk_save_dir, chunk.name+ops(data_path)[-1])
                # save task opt
                chunk.opt.Dataset.data_path = data_chunk_save_path
                dynamic_task_opt_yaml_path = opj(
                    dynamic_task_opt_save_dir, chunk.opt.Log.task_name+'.yaml')
                OmegaConf.save(chunk.opt, dynamic_task_opt_yaml_path)
                # instance Task
                if chunk.sparse == False:
                    command = "python {} -p {}".format(
                        temp_script_path, dynamic_task_opt_yaml_path)
                    dynamic_task_list.append(Task(
                        command, chunk.opt.Log.task_name, gpucost, cpucost, cost_variable=str(chunk.param_size)))
                else:
                    # if chunk.last_param_size == chunk.param_size:
                    #     pass
                    # else:
                    command = "python {} -p {}".format(
                        args.s, dynamic_task_opt_yaml_path)
                    dynamic_task_list.append(Task(
                        command, chunk.opt.Log.task_name, gpucost, cpucost, cost_variable=str(chunk.param_size)))
            try:
                queue = Queue(dynamic_task_list, gpu_list)
                queue.init_sharecost_dict()
                queue.start(time_interval, max_task, log=False,
                            remind=False, debug=debug)
            except:
                pass

            # eval
            orig_sideinfos = OmegaConf.load(orig_sideinfos_path)
            data_shape = orig_sideinfos['data_shape']
            decompressed_data_chunk_list = []
            for chunk in dynamic_chunk.patches:
                chunk_name = chunk.name
                module_path = chunk.save_net_path
                sideinfos_path = chunk.sideinfos_path
                decompressed_data = NFGR.decompress(
                    args.p, module_path, sideinfos_path)
                chunk_dict = {'data': decompressed_data,
                              'name': chunk_name}
                if dynamic_chunk.dimension == 2:
                    h = [int(i)
                         for i in chunk_name.split('-')[0].split('_')[1:]]
                    w = [int(i)
                         for i in chunk_name.split('-')[1].split('_')[1:]]
                elif dynamic_chunk.dimension == 3:
                    d = [int(i)
                         for i in chunk_name.split('-')[0].split('_')[1:]]
                    h = [int(i)
                         for i in chunk_name.split('-')[1].split('_')[1:]]
                    w = [int(i)
                         for i in chunk_name.split('-')[2].split('_')[1:]]
                    chunk_dict['d'] = d
                chunk_dict['h'] = h
                chunk_dict['w'] = w
                decompressed_data_chunk_list.append(chunk_dict)
            decompressed_data = merge_divided_data(
                decompressed_data_chunk_list, data_shape)

            # save data in the exp dir
            current_steps_dir = opj(Log.logdir, 'steps{}'.format(
                steps)+'-dynamic{}'.format((step+1)*dynamic_step))
            os.makedirs(current_steps_dir, exist_ok=True)
            # compress
            compressed_save_dir = opj(current_steps_dir, 'compressed')
            module_save_dir = opj(compressed_save_dir, 'module')
            sideinfos_save_dir = opj(compressed_save_dir, 'sideinfos')
            orig_sideinfos_path = opj(compressed_save_dir, 'sideinfos.yaml')
            os.makedirs(current_steps_dir, exist_ok=True)
            os.makedirs(module_save_dir, exist_ok=True)
            os.makedirs(sideinfos_save_dir, exist_ok=True)
            OmegaConf.save(orig_sideinfos, orig_sideinfos_path)
            for chunk in dynamic_chunk.patches:
                chunk_sideinfos_save_dir = opj(sideinfos_save_dir, chunk.name)
                os.makedirs(chunk_sideinfos_save_dir, exist_ok=True)
                shutil.copy(chunk.sideinfos_path, opj(
                    chunk_sideinfos_save_dir, 'sideinfos.yaml'))
                chunk_module_save_dir = opj(module_save_dir, chunk.name)
                os.makedirs(chunk_module_save_dir, exist_ok=True)
                if chunk.sparse == False:
                    CopyDir(chunk.save_net_path, opj(
                        chunk_module_save_dir, 'module'))
                else:
                    shutil.copy(chunk.sparse_file_path, opj(
                        chunk_module_save_dir, 'module_sparse.npy'))
            actual_compressed_data_bytes = get_folder_size(compressed_save_dir)
            theory_compressed_data_bytes = get_folder_size(
                sideinfos_save_dir) + sum([chunk.actual_module_size for chunk in dynamic_chunk.patches])
            Log.log_metrics({'data_bytes/orig': orig_data_bytes, 'data_bytes/theory_compressed': theory_compressed_data_bytes, 'data_bytes/actual_compressed': actual_compressed_data_bytes,
                             'compress_ratio/theory': orig_data_bytes/theory_compressed_data_bytes, 'compress_ratio/actual': orig_data_bytes/actual_compressed_data_bytes}, steps+(step+1)*dynamic_step)

            # decompress
            if self.opt.Decompress.keep_decompressed:
                decompressed_save_dir = opj(current_steps_dir, 'decompressed')
                os.makedirs(decompressed_save_dir, exist_ok=True)
                decompressed_data_save_path = opj(decompressed_save_dir, opb(
                    ops(data_path)[0])+'_decompressed'+ops(data_path)[-1])
                save_img(decompressed_data_save_path, decompressed_data)
            # mip
            if self.opt.Decompress.mip:
                orig_data = read_img(data_path)
                if len(orig_data.shape) == 4:
                    mip_save_dir = opj(current_steps_dir, 'mip')
                    os.makedirs(mip_save_dir, exist_ok=True)
                    mip_ops(orig_data, mip_save_dir, opb(
                        ops(data_path)[0]), ops(data_path)[-1])
                    mip_ops(decompressed_data, mip_save_dir, opb(
                        ops(data_path)[0])+'_decompressed', ops(data_path)[-1])
            # performance
            performance = eval_performance(steps+(step+1)*dynamic_step, orig_data, decompressed_data, Log, self.opt.Decompress.aoi_thres_list,
                                           self.opt.Decompress.mse_mape_thres_list, self.opt.Decompress.psnr_thres_list,
                                           self.opt.Decompress.iou_acc_pre_thres_list, self.opt.Decompress.ssim)
            csv_path = os.path.join(Log.logdir, 'performance_dynamic.csv')
            if not os.path.exists(csv_path):
                out = open(csv_path, "a")
                csv_writer = csv.writer(out, dialect="excel")
                csv_writer.writerow(performance.keys())
            row = [performance[key] for key in performance.keys()]
            csv_writer.writerow(row)
        #################### END ####################
        # remove
        if not args.substore:
            shutil.rmtree(opj(opd(Log.logdir), 'subexps'))
            shutil.rmtree(opj(opd(Log.logdir), 'data_chunks'))
            shutil.rmtree(opj(opd(Log.logdir), 'task_opts'))
            if opt.CompressFramework.Compress.param.dynamic_iter != 0:
                shutil.rmtree(opj(opd(Log.logdir), 'dynamic_task_opts'))
                shutil.rmtree(opj(opd(Log.logdir), 'dynamic_exps'))
        os.remove(temp_script_path)



class Chunk:
    def __init__(self, data_chunk, device) -> None:
        self.name = data_chunk['name']
        self.dimension = len(data_chunk['data'].shape)-1
        self.device = device
        if self.dimension == 3:
            d = self.name.split('-')[0].split('_')[1:]
            h = self.name.split('-')[1].split('_')[1:]
            w = self.name.split('-')[2].split('_')[1:]
            self.z = int(d[0])
            self.y = int(h[0])
            self.x = int(w[0])
            self.d = int(d[1])+1-int(d[0])
            self.h = int(h[1])+1-int(h[0])
            self.w = int(w[1])+1-int(w[0])
        elif self.dimension == 2:
            h = self.name.split('-')[0].split('_')[1:]
            w = self.name.split('-')[1].split('_')[1:]
            self.y = int(h[0])
            self.x = int(w[0])
            self.h = int(h[1])+1-int(h[0])
            self.w = int(w[1])+1-int(w[0])
        self.data = data_chunk['data']
        self.loss = None
        self.last_loss = None
        self.param_size = data_chunk['param_size']  # 给定参数量
        self.actual_module_size = None              # 实际参数量
        self.phi = None
        self.last_net_path = 'none'
        self.init_net_path = 'none'
        self.save_net_path = 'none'
        self.sideinfos_path = 'none'
        self.sparse = False                     # 是否还在剪枝
        self.sparse_init_net_path = 'none'      # 剪枝初始化模型
        self.sparse_param_size = 'none'         # 剪枝初始模型所给定的参数量
        self.sparse_file_path = 'none'          # 剪枝并压缩后文件位置

    def param_increase(self):
        """ 
        must update these parameters before using this function：
            self.last_param_size,self.param_size,self.opt,self.last_net_path,self.init_net_path
        output: get the init module and save it in the init_net_path
        """
        if self.sparse == False:  # if the module has not been pruned
            # old module
            # init the constructure
            old_opt = deepcopy(self.opt)
            old_features, _, _ = NFGR.estimate_module_size(
                self.last_param_size, old_opt.CompressFramework)
            old_opt.CompressFramework.Module.phi.features = old_features
            old_net = init_phi(old_opt.CompressFramework.Module.phi)
            if old_opt.CompressFramework.Compress.half:
                old_net.half()
            load_model(old_net, self.last_net_path, self.device)
            # new module
            # init the constructure
            new_opt = deepcopy(self.opt)
            new_features, _, _ = NFGR.estimate_module_size(
                self.param_size, new_opt.CompressFramework)
            new_opt.CompressFramework.Module.phi.features = new_features
            new_net = init_phi(new_opt.CompressFramework.Module.phi)
            if new_opt.CompressFramework.Compress.half:
                new_net.half()
            # init the parameters
            with torch.no_grad():
                for l in range(len(new_net.net)):
                    old_weight_shape = old_net.net[l][0].weight.shape
                    new_net.net[l][0].weight[:] = 0
                    new_net.net[l][0].weight[:old_weight_shape[0],
                                             :old_weight_shape[1]] = old_net.net[l][0].weight
                    new_net.net[l][0].bias[:] = 0
                    new_net.net[l][0].bias[:old_weight_shape[0]
                                           ] = old_net.net[l][0].bias
            save_model(new_net, self.init_net_path, self.device)
        else:   # if the module has been pruned
            if self.param_size >= self.sparse_param_size:   # stop pruning
                self.sparse = False
                self.last_param_size = self.sparse_param_size
                self.last_net_path = self.sparse_init_net_path
                self.init_net_path = self.init_net_path
                self.sparse_file_path = 'none'
                self.param_increase()
            else:
                CopyDir(self.sparse_init_net_path, self.init_net_path)

    def param_decrease(self):
        if self.sparse == False:  # if the module has not been pruned
            self.sparse = True
            self.sparse_init_net_path = self.last_net_path
            self.sparse_param_size = self.last_param_size
            self.sparse_file_path = self.save_net_path + '_sparse.npy'
            CopyDir(self.sparse_init_net_path, self.init_net_path)
        else:
            self.sparse_file_path = self.save_net_path + '_sparse.npy'
            CopyDir(self.sparse_init_net_path, self.init_net_path)


class Dynamic_Chunk:
    def __init__(self, data_path, device) -> None:
        self.data_path = data_path
        self.data = read_img(data_path)
        self.dimension = get_dimension(data_path)
        self.device = device

    def init_patches(self, data_chunk_list):
        self.patches = []   # 激活块
        for data_chunk in data_chunk_list:
            chunk = Chunk(data_chunk, self.device)
            self.patches.append(chunk)

    def update_loss(self, decompressed_data):
        self.loss = []
        assert decompressed_data.shape == self.data.shape, "Inconsistent data dimensions!"
        res = self.data - decompressed_data
        for patch in self.patches:
            if self.dimension == 2:
                patch_res = res[patch.y:patch.y +
                                patch.h, patch.x:patch.x+patch.w]
            elif self.dimension == 3:
                patch_res = res[patch.z:patch.z+patch.d,
                                patch.y:patch.y+patch.h, patch.x:patch.x+patch.w]

            patch.last_loss = patch.loss
            patch.loss = sqrt((patch_res**2).mean())
            self.loss.append(patch.loss)

    def update_net(self, num, exchange_ratio):
        self.min_loss_patches = []
        self.max_loss_patches = []
        loss = copy.deepcopy(self.loss)
        max_loss = max(loss)
        for i in range(num):
            min_loss = min(loss)
            min_index = loss.index(min_loss)
            loss[min_index] = max_loss + 1
            self.min_loss_patches.append(self.patches[min_index])
        loss = copy.deepcopy(self.loss)
        max_loss_total = 0
        for i in range(num):
            max_loss = max(loss)
            max_index = loss.index(max_loss)
            loss[max_index] = -1
            self.max_loss_patches.append(self.patches[max_index])
            max_loss_total += max_loss
        # get the init net
        for patch in self.patches:
            patch.last_param_size = patch.param_size
            if patch.sparse == False:
                CopyDir(patch.last_net_path, patch.init_net_path)
            else:
                CopyDir(patch.sparse_init_net_path, patch.init_net_path)
        if exchange_ratio <= 0:
            return
        exchange_param_size = 0
        for patch in self.min_loss_patches[int(-num*0.2):]:
            patch.param_size -= patch.param_size*exchange_ratio
            patch.param_decrease()
            exchange_param_size += patch.param_size*exchange_ratio
        for patch in self.max_loss_patches:
            patch.param_size += exchange_param_size*patch.loss / \
                max_loss_total  # patch.param_size += exchange_param_size/num
            patch.param_increase()


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
    from datetime import datetime
    opt.Log.project_name += datetime.now().strftime("_%Y_%m%d_%H%M%S_%f")
    Log = MyLogger(**opt.Log)
    Log.log_opt(opt)
    reproduc(opt.Reproduc)
    CompressFramework = NFGR(opt.CompressFramework)
    CompressFramework.device = 'cuda' if opt.CompressFramework.Compress.gpu else 'cpu'
    data_path = opt.Dataset.data_path
    if opt.CompressFramework.Compress.divide.divide_type == 'none':
        CompressFramework.compress(data_path)
    else:
        global dynamic_chunk
        dynamic_chunk = Dynamic_Chunk(
            data_path, CompressFramework.device)  # 动态分块
        CompressFramework.compress_divide(data_path, opt)


if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    parser = argparse.ArgumentParser(
        description='single task for datacompress')
    parser.add_argument('-p', type=str, default=opj(opd(__file__),
                        'opt', 'neurons.yaml'), help='yaml file path')
    parser.add_argument('-s', type=str, default=opj(opd(__file__),
                        'sparse_compression.py'), help='sparse compression file path')
    parser.add_argument('-g', help='availabel gpu list', default='0',
                        type=lambda s: [int(item) for item in s.split(',')])
    parser.add_argument('-gc', type=int, default=8000,
                        help='gpu cost in every task. For compress_divide')
    parser.add_argument('-cc', type=int, default=3000,
                        help='cpu cost in every task. For compress_divide')
    parser.add_argument('-t', type=float, default=2,
                        help='the time interval between each task-assigning loop. For compress_divide')
    parser.add_argument('-m', type=int, default=4,
                        help='the max nums of task in running. For compress_divide')
    parser.add_argument('-dropslice', action='store_true',
                        help='. For compress_divide')
    parser.add_argument('-debug', action='store_true',
                        help='. For compress_divide')
    parser.add_argument('-weightstore', action='store_true',
                        help='save the weight of the first layer')
    parser.add_argument('-substore', action='store_true',
                        help='keep the sub exps and opts')
    args = parser.parse_args()
    gpu_list = args.g
    gpucost = args.gc
    cpucost = args.cc
    time_interval = args.t
    max_task = args.m
    dropslice = args.dropslice
    debug = args.debug
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(i) for i in gpu_list])

    main()
