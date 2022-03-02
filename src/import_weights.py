r"""PyTorch Detection Inference.
"""
import datetime
import os
import time

import torch
import torch.utils.data
import torchvision
import torchvision.models.detection
import torchvision.models.detection.mask_rcnn

from coco_utils import get_coco, get_coco_kp, get_coco_api_from_dataset
import matplotlib.pyplot as plt
from PIL import ImageDraw, ImageFont
from PIL import Image

from group_by_aspect_ratio import GroupedBatchSampler, create_aspect_ratio_groups

import presets
import utils

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import torch.hub
import numpy

def download_url_to_file(url, dst, hash_prefix=None, progress=True):
    print((url, dst, hash_prefix, progress))
    
torch.hub.download_url_to_file=download_url_to_file

def get_args_parser(add_help=True):
    import argparse
    parser = argparse.ArgumentParser(description='Import weight-files', add_help=add_help)

    parser.add_argument('--output-dir', default='.', help='path where to save')
    parser.add_argument('--weight-dir', default='./weights/', help='path where to save')
    parser.add_argument('--resume', default='', help='resume from checkpoint')

    return parser

                
def main(args):
    if args.output_dir:
        utils.mkdir(args.output_dir)

    print(args)

    model = torch.load(args.resume)
    weight_dir = args.weight_dir
    model["model"]["roi_heads.box_head.fc6.weight"] = torch.from_numpy(numpy.fromfile(weight_dir + "roi_heads:box_head:fc6:weight_in_12544_out_1024.bin", dtype=numpy.float32).reshape([1024,12544]))
    model["model"]["roi_heads.box_head.fc6.bias"] = torch.from_numpy(numpy.fromfile(weight_dir + "roi_heads:box_head:fc6:bias_in_12544_out_1024.bin", dtype=numpy.float32))
    model["model"]["roi_heads.box_head.fc7.weight"] = torch.from_numpy(numpy.fromfile(weight_dir + "roi_heads:box_head:fc7:weight_in_1024_out_1024.bin", dtype=numpy.float32).reshape([1024,1024]))
    model["model"]["roi_heads.box_head.fc7.bias"] = torch.from_numpy(numpy.fromfile(weight_dir + "roi_heads:box_head:fc7:bias_in_1024_out_1024.bin", dtype=numpy.float32))
    model["model"]["roi_heads.box_predictor.cls_score.weight"] = torch.from_numpy(numpy.fromfile(weight_dir + "roi_heads:box_predictor:cls_score:weight_in_1024_out_14.bin", dtype=numpy.float32).reshape([14,1024]))
    print('Before update')
    print('model["model"]["roi_heads.box_predictor.cls_score.bias"].shape')
    print(model["model"]["roi_heads.box_predictor.cls_score.bias"].shape)
    model["model"]["roi_heads.box_predictor.cls_score.bias"] = torch.from_numpy(numpy.fromfile(weight_dir + "roi_heads:box_predictor:cls_score:bias_in_1024_out_14.bin", dtype=numpy.float32))
    print('After update')
    print('Read ' + (weight_dir + "roi_heads:box_predictor:cls_score:bias_in_1024_out_14.bin"))
    print('model["model"]["roi_heads.box_predictor.cls_score.bias"].shape')
    print(model["model"]["roi_heads.box_predictor.cls_score.bias"].shape)
    utils.save_on_master(
        model,
        os.path.join(args.output_dir, 'checkpoint.pth'))
        
    
if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
