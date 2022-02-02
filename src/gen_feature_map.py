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

#from coco_utils import get_coco, get_coco_kp, get_coco_api_from_dataset
from bdd100k_objects_utils import get_bdd100k_objects
import matplotlib.pyplot as plt
from PIL import ImageDraw, ImageFont
from PIL import Image

from group_by_aspect_ratio import GroupedBatchSampler, create_aspect_ratio_groups

import presets
import utils

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import torch.hub

import pandas as pd

def download_url_to_file(url, dst, hash_prefix=None, progress=True):
    print((url, dst, hash_prefix, progress))
    
torch.hub.download_url_to_file=download_url_to_file

def get_dataset(name, image_set, transform, data_path):
    paths = {
        "bdd100k-objects": (data_path, get_bdd100k_objects, 13+1),
    }
    p, ds_fn, num_classes = paths[name]

    ds = ds_fn(p, image_set=image_set, transforms=transform)
    return ds, num_classes


def get_transform(train, data_augmentation):
    return presets.DetectionPresetTrain(data_augmentation) if train else presets.DetectionPresetEval()


def get_args_parser(add_help=True):
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch Detection Training', add_help=add_help)

    parser.add_argument('--data-path', default='bdd100k-objects', help='dataset')
    parser.add_argument('--dataset', default='bdd100k-objects', help='dataset')
    parser.add_argument('--model', default='fasterrcnn_resnet50_fpn', help='model')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('-b', '--batch-size', default=2, type=int,
                        help='images per gpu, the total batch size is $NGPU x batch_size')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--print-freq', default=20, type=int, help='print frequency')
    parser.add_argument('--output-dir', default='.', help='path where to save')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, help='start epoch')
    parser.add_argument('--aspect-ratio-group-factor', default=3, type=int)
    parser.add_argument('--rpn-score-thresh', default=0.4, type=float, help='rpn score threshold for faster-rcnn')
    parser.add_argument('--rpn-nms-thresh', default=0.5, type=float, help='rpn nms threshold for faster-rcnn')
    parser.add_argument('--box-score-thresh', default=0.4, type=float, help='box score threshold for faster-rcnn')
    parser.add_argument('--box-nms-thresh', default=0.5, type=float, help='box nms threshold for faster-rcnn')
    parser.add_argument('--trainable-backbone-layers', default=None, type=int,
                        help='number of trainable layers of backbone')
    parser.add_argument('--data-augmentation', default="hflip", help='data augmentation policy (default: hflip)')
    parser.add_argument(
        "--sync-bn",
        dest="sync_bn",
        help="Use sync batch norm",
        action="store_true",
    )
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )
    parser.add_argument(
        "--pretrained",
        dest="pretrained",
        help="Use pre-trained models from the modelzoo",
        action="store_true",
    )

    # distributed training parameters
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')

    return parser

roi_head_input=None
box_feature_map=None

@torch.no_grad()
def inference(model,
              data_loaders,
              device,
              dataset_name="bdd100k-objects",
              output_dir="output"
              ):
    model.eval()

    for (data_loader,dataset_dir) in data_loaders:
        for (i,dat) in enumerate(data_loader):
            print(dat["image"].shape)
            images = dat["image"].to(device)
    
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            model_time = time.time()
            outputs = model(images)
            print("outputs")
            print(outputs)
            global box_feature_map
            print(dat["filename"])
            filename=output_dir + "/" + dataset_dir + "/" + dat["filename"][0]+".bin"

            
            print(filename)
            box_feature_map.cpu().numpy().tofile(filename)

            
            
def main(args):
    if args.output_dir:
        utils.mkdir(args.output_dir)

    utils.init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)

    # Data loading code
    print("Loading data")

    dataset, num_classes = get_dataset(args.dataset, "train", get_transform(False, args.data_augmentation), args.data_path)
    dataset_test, _ =      get_dataset(args.dataset, "val", get_transform(False, args.data_augmentation), args.data_path)
    print(dataset[0]['image'].shape)
    print(dataset_test[0]['image'].shape)
    print(num_classes)

    print("Creating data loaders")
    sampler = torch.utils.data.SequentialSampler(dataset)
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)
   # for data in iter(dataset):
   #yield collate_fn(data)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1,
        sampler=sampler, num_workers=args.workers)
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1,
        sampler=test_sampler, num_workers=args.workers)

    print("Creating model")
    kwargs = {
        "trainable_backbone_layers": args.trainable_backbone_layers,
        "rpn_score_thresh": args.rpn_score_thresh,
        "rpn_nms_thresh": args.rpn_nms_thresh,
        "box_score_thresh": args.box_score_thresh,
        "box_nms_thresh": args.box_nms_thresh
    }
    model = torchvision.models.detection.__dict__[args.model](num_classes=num_classes, pretrained=args.pretrained,**kwargs)
    model.eval()
    model.to(device)
    #print(model)

    def hook(module,input,output):
        for (i,(height,width)) in enumerate(input[0].image_sizes):
#            output[0][i]=torch.tensor([[0,0,width,height]],dtype=torch.float32).to(device)
            mergin = 0.2
            w = (width/(1 + mergin*2))
            h = (height/(1 + mergin*2))
            output[0][i]=torch.tensor([[w*mergin,h*mergin,w*mergin+w,h*mergin+h]],dtype=torch.float32).to(device)
            print(output[0][i])
        return output
    def transform_hook(module,input,output):
        # print("transform_hook")
        # print("input")
        # print(input[0].shape)
        # print("output")
        # print(output[0].tensors.shape)
        # print(output[0].image_sizes)
        output[0].tensors = input[0]
        output[0].image_sizes = [tuple(input[0].shape[2:])]
        # print(output[0].tensors.shape)
        # print(output[0].image_sizes)
        return output
    def backbone_hook(module,input,output):
        print("backbone_hook")
        print(input[0].shape)
        
    def roi_head_hook(module,input,output):
        global roi_head_input
        print("roi_head_hook")
        roi_head_input=input[0]["0"]
        print(roi_head_input.shape)
    def box_roi_head_hook(module,input,output):
        global box_feature_map
        print("box_roi_head_hook")
        box_feature_map=output
        print(box_feature_map.shape)
        print(box_feature_map.dtype)
        
    model.transform.register_forward_hook(transform_hook)
    #model.backbone.register_forward_hook(backbone_hook)
    model.rpn.register_forward_hook(hook)
    #model.roi_heads.register_forward_hook(roi_head_hook)
    model.roi_heads.box_roi_pool.register_forward_hook(box_roi_head_hook)
        
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
    
    # model.roi_heads.box_head.fc6.weight.cpu().detach().numpy().tofile(args.output_dir + "/" + "roi_heads:box_head:fc6:weight_in_12544_out_1024.bin")
    # model.roi_heads.box_head.fc6.bias.cpu().detach().numpy().tofile(args.output_dir + "/" + "roi_heads:box_head:fc6:bias_in_12544_out_1024.bin")
    # model.roi_heads.box_head.fc7.weight.cpu().detach().numpy().tofile(args.output_dir + "/" + "roi_heads:box_head:fc7:weight_in_1024_out_1024.bin")
    # model.roi_heads.box_head.fc7.bias.cpu().detach().numpy().tofile(args.output_dir + "/" + "roi_heads:box_head:fc7:bias_in_1024_out_1024.bin")
    # model.roi_heads.box_predictor.cls_score.weight.cpu().detach().numpy().tofile(args.output_dir + "/" + "roi_heads:box_predictor:cls_score:weight_in_1024_out_14.bin")
    # model.roi_heads.box_predictor.cls_score.bias.cpu().detach().numpy().tofile(args.output_dir + "/" + "roi_heads:box_predictor:cls_score:bias_in_1024_out_14.bin")
    
    inference(model,
              [(data_loader,"trains/images"),(data_loader_test,"valids/images")],
              device=device,
              dataset_name= args.data_path,
              output_dir= args.output_dir
              )


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
