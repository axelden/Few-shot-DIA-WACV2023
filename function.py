
import numpy as np
import time
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score
import cv2
import torch
from skimage.measure import label, regionprops, regionprops_table
import torch.nn as nn

def saveIMGRes(imgs, masks, save_path):
    # creating the colormaps
    colour_table = np.zeros((256, 1, 3), dtype=np.uint8)
    colour_table[0] = [0, 0, 0]
    colour_table[1] = [0, 255, 255]
    colour_table[2] = [255, 255, 0]
    colour_table[3] = [255, 0, 255]

    # iterating throug the images and masks
    for img, mask in zip(imgs, masks):
        date_string = time.strftime("%d%m%Y-%H-%M-%S")
        # applying colormaps to the masks
        colour_mask = cv2.applyColorMap(mask.squeeze().astype(np.uint8), colour_table)
        # formatting image
        np_img = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        # adding generated masks to the images
        img = cv2.addWeighted(np_img, 0.1, colour_mask, 1, 0)
        # saving results
        cv2.imwrite(save_path + f"IMG{date_string}.png", img)

def get_scores( y_true,y_pred, average="macro"):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average=average)
    recall = recall_score(y_true, y_pred, average=average)
    f1_sc = f1_score(y_true, y_pred, average=average)
    iou_sc = jaccard_score(y_true, y_pred, average=average)
    return accuracy, precision, recall, f1_sc, iou_sc


