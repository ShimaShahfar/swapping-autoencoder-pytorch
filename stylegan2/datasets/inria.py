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


class INRIA(torch.utils.data.Dataset):
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
        self.mask_dir = os.path.join(self.data_dir, "gt/")

    def __len__(self):
        return len(self.image_dir_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = self.read_train(idx)
        PIL_image = Image.fromarray(np.uint8(image)).convert('RGB')
        image_transform = transforms.Compose(
            [
                # transforms.Resize((256,256)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
            ]
        )
        
        i, j, h, w = transforms.RandomCrop.get_params(PIL_image, output_size=(256, 256))
        PIL_image = TF.crop(PIL_image, i, j, h, w)
        image = image_transform(PIL_image)

        return image.float() 
    
    def read_train(self, idx):
        create_image_path = os.path.join(
                self.image_dir,
                f"{self.image_dir_list[idx]}",
            )
        # create_mask_path = os.path.join(
        #         self.mask_dir,
        #         f"{self.image_dir_list[idx]}",
        #     )

        with rio.open(create_image_path) as image:
            n_band = image.count
            img = np.zeros((image.shape[0], image.shape[1], n_band))
            for idx in range(n_band):
                img[:, :, idx] = image.read(idx+1)
        # img = img / np.max(img)
        
        # with rio.open(create_mask_path) as gt:
        #     mask = np.zeros((gt.shape[0], gt.shape[1]))
        #     mask[:, :] = gt.read(1)
        
        return img
    
    def read_test(self, idx):
        create_test_path = os.path.join(
                self.test_dir,
                f"{self.test_dir_list[idx]}",
            )

        with rio.open(create_test_path) as image:
            n_band = image.count
            img = np.zeros((image.shape[0], image.shape[1], n_band))
            for idx in range(n_band):
                img[:, :, idx] = image.read(idx+1)
        img = img / np.max(img)

        return img
    

    def onehot_to_binary_edges(self, mask, radius):
        if radius < 0:
            return mask
        mask = mask.numpy()
        mask = mask.astype(np.uint8)
        # We need to pad the borders for boundary conditions
        mask_pad = np.pad(mask, ((1,1),(1,1)), mode="constant", constant_values=0)
        dist = distance_transform_edt(mask_pad)
        dist = dist[1:-1, 1:-1]
        for i in range(dist.shape[0]):
            for j in range(dist.shape[1]):
                if dist[i][j] > radius:
                    dist[i][j] = 0
        # edgemap += dist
#         edgemap = np.expand_dims(edgemap, axis=0)
#         edgemap = (edgemap > 0).astype(np.uint8)
        return dist

    def get_edge(self, idx, mask):
        _edgemap = mask
        _edgemap = self.onehot_to_binary_edges(_edgemap, 1)
        return _edgemap

    def auto_canny(self, image, sigma=0.33):
        # compute the median of the single channel pixel intensities
        image = img_as_ubyte(image)
        v = np.median(image)

        # apply automatic Canny edge detection using the computed median
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        edge = cv2.Canny(image, lower, upper)
        edge = edge[np.newaxis, ...].astype(np.float32) / 255

        return edge
