# `.\PaddleOCR\benchmark\PaddleOCR_DBNet\utils\compute_mean_std.py`

```py
# -*- coding: utf-8 -*-
# @Time    : 2019/12/7 14:46
# @Author  : zhoujun

# 导入所需的库
import numpy as np
import cv2
import os
import random
from tqdm import tqdm

# 计算图像数据集的均值和标准差
train_txt_path = './train_val_list.txt'

CNum = 10000  # 挑选多少图片进行计算

img_h, img_w = 640, 640
imgs = np.zeros([img_w, img_h, 3, 1])
means, stdevs = [], []

# 读取训练数据集的文件路径
with open(train_txt_path, 'r') as f:
    lines = f.readlines()
    random.shuffle(lines)  # shuffle , 随机挑选图片

    # 遍历选定数量的图片
    for i in tqdm(range(CNum)):
        img_path = lines[i].split('\t')[0]

        # 读取图像并调整大小
        img = cv2.imread(img_path)
        img = cv2.resize(img, (img_h, img_w))
        img = img[:, :, :, np.newaxis]

        # 将图像数据添加到数组中
        imgs = np.concatenate((imgs, img), axis=3)

# 将图像数据转换为浮点数并归一化
imgs = imgs.astype(np.float32) / 255.

# 计算每个通道的均值和标准差
for i in tqdm(range(3)):
    pixels = imgs[:, :, i, :].ravel()  # 拉成一行
    means.append(np.mean(pixels))
    stdevs.append(np.std(pixels))

# cv2 读取的图像格式为BGR，PIL/Skimage读取到的都是RGB不用转
means.reverse()  # BGR --> RGB
stdevs.reverse()

# 打印计算得到的均值和标准差
print("normMean = {}".format(means))
print("normStd = {}".format(stdevs))
print('transforms.Normalize(normMean = {}, normStd = {})'.format(means, stdevs))
```