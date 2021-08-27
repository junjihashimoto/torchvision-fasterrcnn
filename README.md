# Faster RCNN of Torchvision 

This repo provides CLI commands for Faster RCNN of Torchvision.

Original code is https://github.com/pytorch/vision/tree/19ad0bbc5e26504a501b9be3f0345381d6ba1efc/references/detection .


## Setup bdd100k-dataset

```
nix build github:hasktorch/hasktorch-datasets#datasets-bdd100k-coco -o bdd100k
```


## Training

```
python train.py
```


## Test

```
python test.py
```

## Inference

```
python inference.py
```
