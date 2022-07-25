import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(os.path.dirname(__file__))

from cyx_exp.utils.img_utils import show_bright_images, show_3d_img
import numpy as np
import argparse
from os.path import join as opj
from os.path import dirname as opd
from os.path import basename as opb
from os.path import splitext as ops
from utils.io import read_data_batch, save_data_batch, read_img


    
if __name__=="__main__":
    parser = argparse.ArgumentParser(
        description='single task for datacompress')
    parser.add_argument('-p', type=str, default="cyx_exp/experiments_outputs/ex0207_lqm_ffmpeg/outputs/sota_ffmpeg_2022_0208_202709/task_00001/decompressed/d_16_31-h_256_511-w_0_255.tif", help='yaml file path')
    # parser.add_argument('p', type=str, help='yaml file path')
    parser.add_argument('-g', type=float, default=1, help='image gain')
    parser.add_argument('-n', action="store_true", help='normalize to full scale')
    args = parser.parse_args()
    print("Reading data...")
    img = read_img(args.p)
    print("Data type: {:s}, Data shape: {:s}, Data range: {:f}-{:f}".format(
        str(img.dtype),
        str(img.shape),
        np.min(img),
        np.max(img)
    ))
    
    data_range = {"uint16":65535.0, "uint8":255.0, "float64":65535.0, "float32":2}[str(img.dtype)]
    show_3d_img(img/data_range, gain=args.g, norm=args.n)