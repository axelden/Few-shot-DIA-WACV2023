from torch.utils.data import Dataset
import os
from skimage import io
import numpy as np
import torchvision.transforms as T
from PIL import Image
import torch
from skimage.color import rgb2gray, gray2rgb
from skimage.filters import threshold_sauvola, threshold_niblack
from skimage.morphology import dilation, disk, erosion, diameter_closing, closing, binary_opening, opening
from einops import rearrange

import matplotlib.pyplot as plt
class BibDS(Dataset):
    def __init__(self, data_dir, gt_dir, transform=None, gen_crops=False, gen_bin_mask=False, n_crops=0, patch_split=False, patch_size=None):
        # super().__init__(data_dir, transform)
        self.data_dir = data_dir
        self.gt_dir = gt_dir
        self.data_filelist = os.listdir(self.data_dir)
        self.transform = transform
        self.patch_split = patch_split
        self.gen_bin_mask=gen_bin_mask
        if self.patch_split:
            self.patch_h = patch_size[0]
            self.patch_w = patch_size[1]
            self.gen_crops = gen_crops
        else:
            self.gen_crops = False
        if self.gen_crops:
            self.n_crops = n_crops

    def genMask(self, GTImg, dilate=True):
        GTImg = np.asarray(GTImg)
        blue_ch = np.copy(GTImg[:, :, 2])
        red_channel = np.copy(GTImg[:, :, 0])

        GTMask = np.zeros(blue_ch.shape)  # Extract just blue channel
        bg = ((blue_ch % 2) == 1) | (red_channel == 128)
        comment = (blue_ch == 2)
        decoration = (((blue_ch == 4) > 0)| ((blue_ch == 6) > 0) | ((blue_ch == 12) > 0))
        text = ((blue_ch == 8) > 0)| ((blue_ch == 10) > 0)

        GTMask[decoration] = 2

        GTMask[comment] = 1
        GTMask[text] = 3
        GTMask[bg] = 0

        if dilate:
            GTMask = dilation(GTMask.squeeze(), disk(1))
        return GTMask.astype(int)

    def genFilter(self, image, ws=31, k=0.01):

        image = image.permute(1, 2, 0)
        gray = rgb2gray(image)
        thresh = threshold_sauvola(gray, window_size=ws, k=k)
        bin_mask = (gray < thresh)

        return bin_mask

    def genRandomCrops(self, image, gt_image):
        random_crops = torch.tensor([])
        random_crops_gt = torch.tensor([])
        for i in range(self.n_crops):
            h, w, hs, ws = T.RandomCrop.get_params(image, (self.patch_h, self.patch_w))
            r_crop = T.functional.crop(image, h, w, hs, ws)
            r_crop_gt = T.functional.crop(gt_image, h, w, hs, ws)
            random_crops = torch.cat([random_crops, r_crop.unsqueeze(0)])
            random_crops_gt = torch.cat([random_crops_gt, r_crop_gt.unsqueeze(0)])

        return random_crops, random_crops_gt

    def splitIntoPatches(self, img):
        return rearrange(img, "c (h p1) (w p2) -> (h w) c p1 p2", p1=self.patch_h, p2=self.patch_w)

    def __len__(self):
        return len(self.data_filelist)

    def __getitem__(self, idx):
        img_filename = self.data_filelist[idx]

        # loading and transforming the img
        img_path = os.path.join(self.data_dir, img_filename)
        image = Image.open(img_path)

        gt_image_path = os.path.join(self.gt_dir, img_filename)
        gt_image_path = os.path.splitext(gt_image_path)[0] +".png"
        gt_image = io.imread(gt_image_path)

        if self.transform:
            image = self.transform(image)
            GTtf = T.Compose([
                T.ToPILImage(),
                T.Resize((image.shape[1],image.shape[2]), interpolation=T.InterpolationMode.NEAREST)
            ])
            gt_image = GTtf(gt_image)
            # gt_mask = GTtf(gt_mask)

        bin_mask = self.genFilter(image) if self.gen_bin_mask else torch.tensor([])
        gt_mask= self.genMask(gt_image)
        # boundary_mask=np.ones(gt_mask.shape)
        if self.patch_split:
            assert ((image.shape[1] % self.patch_h) + (image.shape[2] % self.patch_w)) == 0, "Image size must be divisible by patch size!"
            instance = self.splitIntoPatches(image)
            gt_instance = self.splitIntoPatches(torch.tensor(gt_mask).unsqueeze(0)).squeeze()
            if self.gen_bin_mask:
                bin_mask = self.splitIntoPatches(torch.tensor(bin_mask).unsqueeze(0)).squeeze()
        else:
            instance = image
            gt_instance = torch.tensor(gt_mask)

        if self.gen_crops:
            random_crops, random_crops_gt = self.genRandomCrops(image, torch.tensor(gt_mask))
            instance = torch.cat([instance, random_crops])
            gt_instance = torch.cat([gt_instance.squeeze(), random_crops_gt])

        sample = {'images': instance, 'gt': gt_instance, 'bin_mask': bin_mask}

        return sample


class Bib_trainset(BibDS):
    def __init__(self, data_dir, gt_dir, transform=None, gen_crops=True, gen_bin_mask=False, n_crops=0, patch_split=False, patch_size=None):
        super().__init__(data_dir, gt_dir, transform, gen_crops, gen_bin_mask,  n_crops, patch_split, patch_size)
        self.data_dir = data_dir+'/training'
        self.gt_dir = gt_dir+'/training'
        self.data_filelist = os.listdir(self.data_dir)

class Bib_validset(BibDS):
    def __init__(self, data_dir, gt_dir, transform=None, gen_crops=True, gen_bin_mask=False, n_crops=0, patch_split=False, patch_size=None):
        super().__init__(data_dir, gt_dir, transform, gen_crops, gen_bin_mask, n_crops, patch_split, patch_size)
        self.data_dir = data_dir+'/validation'
        self.gt_dir = gt_dir+'/validation'
        self.data_filelist = os.listdir(self.data_dir)

class Bib_testset(BibDS):
    def __init__(self, data_dir, gt_dir, transform=None, gen_crops=False, gen_bin_mask=True, n_crops=0, patch_split=False, patch_size=None):
        super().__init__(data_dir, gt_dir, transform, gen_crops, gen_bin_mask, n_crops, patch_split, patch_size)
        self.data_dir = data_dir+'/public-test'
        self.gt_dir = gt_dir+'/public-test'
        self.data_filelist = os.listdir(self.data_dir)

    def genMask(self, GTImg, dilate=False):
        return super().genMask(GTImg, dilate)
