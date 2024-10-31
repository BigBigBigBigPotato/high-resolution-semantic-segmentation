import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import random
import shutil
import cv2


def img_padding(img_path):
    '''
    图片填充
    Args:
        img_path: example: "E:\DataSet\data\AerialImageDataset\test\images\bellingham1.tif"
    Returns:填充后的图片，type = <class 'numpy.ndarray'>
    '''
    img = cv2.imread(img_path)
    H, W = img.shape[0], img.shape[1]
    diff_height = 5120 - H
    diff_width = 5120 - W
    img_padded = np.pad(img, ((0, diff_height), (0, diff_width), (0, 0)), mode='constant')
    return img_padded


def img_pad_crop(img_path, crop_size, output_path):
    '''
    对单张图片进行裁剪和分割
    Args:
        img_path: example: "E:\DataSet\data\AerialImageDataset\test\images\bellingham1.tif"
        crop_size: example: 512
        output_path: example: "E:\DataSet\data\AerialImageDataset_preprocessed\train\images"
    '''
    img = cv2.imread(img_path)
    H, W = img.shape[0], img.shape[1]
    diff_height = 5120 - H
    diff_width = 5120 - W
    img_padded = np.pad(img, ((0, diff_height), (0, diff_width), (0, 0)), mode='constant')

    H, W = img_padded.shape[0], img_padded.shape[1]
    for i in range(H // crop_size):
        for j in range(W // crop_size):
            img_cropped = img_padded[i * crop_size:(i + 1) * crop_size, j * crop_size:(j + 1) * crop_size]
            cropped_name = os.path.basename(img_path).split(".")[0] + f"_{i}_{j}.tif"
            print(cropped_name)
            cv2.imwrite(os.path.join(output_path, cropped_name), img_cropped)


def img_preprecess(img_path, output_path):
    '''
    输入原数据集和处理后的图片的训练和验证数据集，进行图片处理和划分
    Args:
        img_dir: example:"E:\DataSet\data\AerialImageDataset\train\images"
        train_path: example"E:\DataSet\data\AerialImageDataset_preprocessed\train\images"
    '''
    for f in os.listdir(img_path):
        img_pad_crop(os.path.join(img_path, f), 512, output_path)


if __name__ == "__main__":
    img_path = r"E:\DataSet\data\AerialImageDataset\train\images"
    output_path = r"E:\DataSet\data\AerialImageDataset_preprocessed\train\images"
    img_preprecess(img_path,output_path)