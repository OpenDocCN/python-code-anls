# `.\comic-translate\modules\ocr\pororo\pororo\models\brainOCR\imgproc.py`

```py
"""
This is adapted from https://github.com/clovaai/CRAFT-pytorch/blob/master/imgproc.py
Copyright (c) 2019-present NAVER Corp.
MIT License
"""

import cv2
import numpy as np
from skimage import io


def load_image(img_file):
    # 使用 skimage 库读取图像文件，返回的图像数据格式为 RGB 顺序
    img = io.imread(img_file)  # RGB order
    
    # 如果图像高度为2，将其转换为一维数组
    if img.shape[0] == 2:
        img = img[0]
    
    # 如果图像是单通道灰度图，使用 OpenCV 将其转换为三通道 RGB 图像
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    
    # 如果图像包含 alpha 通道，去除 alpha 通道，保留前三个通道
    if img.shape[2] == 4:
        img = img[:, :, :3]
    
    # 将图像转换为 NumPy 数组格式
    img = np.array(img)

    return img


def normalize_mean_variance(
        in_img,
        mean=(0.485, 0.456, 0.406),
        variance=(0.229, 0.224, 0.225),
):
    # 将输入图像复制为 float32 类型
    img = in_img.copy().astype(np.float32)

    # 减去均值并除以方差，注意均值和方差需要乘以255来归一化到0-255范围
    img -= np.array([mean[0] * 255.0, mean[1] * 255.0, mean[2] * 255.0],
                    dtype=np.float32)
    img /= np.array(
        [variance[0] * 255.0, variance[1] * 255.0, variance[2] * 255.0],
        dtype=np.float32,
    )
    return img


def denormalize_mean_variance(
        in_img,
        mean=(0.485, 0.456, 0.406),
        variance=(0.229, 0.224, 0.225),
):
    # 将输入图像复制一份
    img = in_img.copy()
    
    # 恢复图像的均值和方差，将图像还原到0-255的整数范围
    img *= variance
    img += mean
    img *= 255.0
    
    # 将图像像素值裁剪到0-255之间，并转换为 uint8 类型
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def resize_aspect_ratio(
    img: np.ndarray,
    square_size: int,
    interpolation: int,
    mag_ratio: float = 1.0,
):
    # 获取图像的高度、宽度和通道数
    height, width, channel = img.shape

    # 计算放大后的目标尺寸
    target_size = mag_ratio * max(height, width)

    # 如果放大后的尺寸大于指定的正方形尺寸，将目标尺寸设为正方形尺寸
    if target_size > square_size:
        target_size = square_size

    # 计算缩放比例
    ratio = target_size / max(height, width)

    # 计算缩放后的目标高度和宽度
    target_h, target_w = int(height * ratio), int(width * ratio)
    
    # 使用 OpenCV 进行图像缩放
    proc = cv2.resize(img, (target_w, target_h), interpolation=interpolation)

    # 创建一个空白画布并将缩放后的图像粘贴到中心位置
    target_h32, target_w32 = target_h, target_w
    if target_h % 32 != 0:
        target_h32 = target_h + (32 - target_h % 32)
    if target_w % 32 != 0:
        target_w32 = target_w + (32 - target_w % 32)
    
    # 创建一个指定尺寸的空白画布，用于接受缩放后的图像
    resized = np.zeros((target_h32, target_w32, channel), dtype=np.float32)
    resized[0:target_h, 0:target_w, :] = proc
    
    # 更新目标高度和宽度
    target_h, target_w = target_h32, target_w32

    # 返回缩放后的图像、缩放比例和处理后的尺寸
    size_heatmap = (int(target_w / 2), int(target_h / 2))
    return resized, ratio, size_heatmap


def cvt2heatmap_img(img):
    # 将图像像素值裁剪到0-1之间，并转换为 uint8 类型
    img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    
    # 将灰度图像应用热力图映射（COLORMAP_JET）
    img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
    return img
```