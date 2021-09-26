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

def download_url_to_file(url, dst, hash_prefix=None, progress=True):
    print((url, dst, hash_prefix, progress))
    
torch.hub.download_url_to_file=download_url_to_file

def get_dataset(name, image_set, transform, data_path):
    paths = {
        "bdd100k": (data_path, get_coco, 13+1),
        "coco": (data_path, get_coco, 91),
        "coco_kp": (data_path, get_coco_kp, 2)
    }
    p, ds_fn, num_classes = paths[name]

    ds = ds_fn(p, image_set=image_set, transforms=transform)
    return ds, num_classes


def get_transform(train, data_augmentation):
    return presets.DetectionPresetTrain(data_augmentation) if train else presets.DetectionPresetEval()


def get_args_parser(add_help=True):
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch Detection Training', add_help=add_help)

    parser.add_argument('--data-path', default='bdd100k', help='dataset')
    parser.add_argument('--data-dir', default='images/valids', help='dataset')
    parser.add_argument('--dataset', default='bdd100k', help='dataset')
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


@torch.no_grad()
def inference(model,
              data_loader,
              device,
              dataset_name="few-bdd100k",
              dataset_dir="images/valids",
              output_dir="output"
              ):
    coco = get_coco_api_from_dataset(data_loader.dataset)
    catIDs = coco.getCatIds()
    cats = coco.loadCats(catIDs)
    model.eval()

    for images, targets in data_loader:
        images = list(img.to(device) for img in images)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_time = time.time()

        outputs = model(images)

        for (output,image,target) in zip(outputs,images,targets):
            img = Image.open(dataset_name + "/" + dataset_dir + "/" + target["file_name"]).convert('RGB')
            draw = ImageDraw.Draw(img)
            #fnt = ImageFont.truetype("arial.ttf", 10)#40
            #text_w, text_h = fnt.getsize(label)
            for box, label, score  in zip(output["boxes"],output["labels"],output["scores"]):
                print(f"%s, %s, score = %.3f" % (target["file_name"], cats[int(label)]["name"], score))
                draw.rectangle([(box[0], box[1]), (box[2], box[3])], outline="red", width=1)
                draw.text((box[0], box[1]), cats[int(label)]["name"], fill='white')
            img.save(output_dir + "/" + target["file_name"])

def main(args):
    if args.output_dir:
        utils.mkdir(args.output_dir)

    utils.init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)

    # Data loading code
    print("Loading data")

    dataset_test, num_classes = get_dataset(args.dataset, "val", get_transform(False, args.data_augmentation), args.data_path)
    print(num_classes)

    print("Creating data loaders")
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1,
        sampler=test_sampler, num_workers=args.workers,
        collate_fn=utils.collate_fn)

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

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
    
    inference(model,
              data_loader_test,
              device=device,
              dataset_name= args.data_path,
              dataset_dir= args.data_dir,
              output_dir= args.output_dir
              )


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
