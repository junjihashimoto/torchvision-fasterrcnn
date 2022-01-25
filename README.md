# Faster R-CNN of Torchvision 

This repo provides CLI commands for Faster R-CNN of Torchvision.

Original code is https://github.com/pytorch/vision/tree/19ad0bbc5e26504a501b9be3f0345381d6ba1efc/references/detection .

This repository is also a component for machine learning pipeline managed by nix.


## Setup bdd100k-dataset

```
nix build github:hasktorch/hasktorch-datasets#datasets-bdd100k-coco -o bdd100k
```


## Training

```
nix develop
python src/train.py
```


## Test

```
nix develop
python src/test.py
```

## Inference

```
nix develop
python src/inference.py
```
