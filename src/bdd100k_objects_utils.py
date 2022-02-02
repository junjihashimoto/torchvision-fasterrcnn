import copy
import os
from PIL import Image

import torch
import torch.utils.data
from torch.utils.data import Dataset, DataLoader
import torchvision
import pandas as pd

import transforms as T
import cv2 as io

from torchvision.transforms import functional as F
#from torchvision.transforms import transforms as T


class Bdd100kObjectsDataset(Dataset):
    """Bdd100k Objects dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.classids = pd.read_csv(csv_file,header=None)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.classids)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        filename = self.classids.iloc[idx, 0]
        img_name = os.path.join(self.root_dir,
                                filename)
        image = Image.open(img_name)
        image = F.pil_to_tensor(image)
        image = F.convert_image_dtype(image)
        classid = self.classids.iloc[idx, 1]
        sample = {
            'image': image,
            'classid': classid,
            'filename': filename
        }

        #if self.transform:
        #    sample = self.transform(sample)

        return sample

def get_bdd100k_objects(root, image_set, transforms=None, mode='instances'):
    PATHS = {
        "train": ("trains/images", "trains/labels.csv"),
        "val": ("valids/images", "valids/labels.csv"),
    }
    img_folder, ann_file = PATHS[image_set]
    img_folder = os.path.join(root, img_folder)
    ann_file = os.path.join(root,ann_file)

    return Bdd100kObjectsDataset(ann_file,img_folder,transforms)
