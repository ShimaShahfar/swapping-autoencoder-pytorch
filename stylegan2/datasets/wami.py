"""INRIA Dataset."""

import os
import numpy as np
import torch
import cv2
from PIL import Image
import rasterio as rio
from skimage import img_as_ubyte
import matplotlib.pylab as plt
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from scipy.ndimage.morphology import distance_transform_edt
import torchvision.transforms.functional as TF


class WAMI(torch.utils.data.Dataset):
    def __init__(
        self,
        data_dir
    ):
        """
        Args:
            # cfg: Configs
            data_dir (string): Path to dataset directory.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        # super(INRIA, self).__init__(data_dir)
        self.data_dir = data_dir
        self.image_dir = os.path.join(self.data_dir, "images/")
        self.image_dir_list = os.listdir(self.image_dir)
    def __len__(self):
        return len(self.image_dir_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        PIL_image = Image.open(self.image_dir_list[idx]).convert('RGB')
        image_transform = transforms.Compose(
            [
                # transforms.Resize((256,256)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
            ]
        )
        
        i, j, h, w = transforms.RandomCrop.get_params(PIL_image, output_size=(512, 512))
        PIL_image = TF.crop(PIL_image, i, j, h, w)
        image = image_transform(PIL_image)

        return image.float() 