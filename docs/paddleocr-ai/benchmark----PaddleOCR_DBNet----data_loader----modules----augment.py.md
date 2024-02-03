# `.\PaddleOCR\benchmark\PaddleOCR_DBNet\data_loader\modules\augment.py`

```py
# -*- coding: utf-8 -*-
# @Time    : 2019/8/23 21:52
# @Author  : zhoujun

# 导入所需的库
import math
import numbers
import random

import cv2
import numpy as np
from skimage.util import random_noise

# 定义一个类，用于给图片添加随机噪声
class RandomNoise:
    def __init__(self, random_rate):
        self.random_rate = random_rate

    def __call__(self, data: dict):
        """
        对图片加噪声
        :param data: {'img':,'text_polys':,'texts':,'ignore_tags':}
        :return:
        """
        # 如果随机数大于设定的随机系数，则直接返回数据
        if random.random() > self.random_rate:
            return data
        # 对图片添加高斯噪声，并将像素值缩放到0-255范围
        data['img'] = (random_noise(
            data['img'], mode='gaussian', clip=True) * 255).astype(im.dtype)
        return data

# 定义一个类，用于对图片和文本框进行随机缩放
class RandomScale:
    def __init__(self, scales, random_rate):
        """
        :param scales: 尺度
        :param ramdon_rate: 随机系数
        :return:
        """
        self.random_rate = random_rate
        self.scales = scales

    def __call__(self, data: dict) -> dict:
        """
        从scales中随机选择一个尺度，对图片和文本框进行缩放
        :param data: {'img':,'text_polys':,'texts':,'ignore_tags':}
        :return:
        """
        # 如果随机数大于设定的随机系数，则直接返回数据
        if random.random() > self.random_rate:
            return data
        # 获取图片和文本框数据
        im = data['img']
        text_polys = data['text_polys']

        tmp_text_polys = text_polys.copy()
        # 从尺度列表中随机选择一个尺度
        rd_scale = float(np.random.choice(self.scales))
        # 对图片进行缩放
        im = cv2.resize(im, dsize=None, fx=rd_scale, fy=rd_scale)
        # 对文本框坐标也进行相应缩放
        tmp_text_polys *= rd_scale

        data['img'] = im
        data['text_polys'] = tmp_text_polys
        return data

# 定义一个类，用于对图片和文本框进行随机旋转
class RandomRotateImgBox:
    # 初始化函数，接受角度、随机系数和是否保持原图大小作为参数
    def __init__(self, degrees, random_rate, same_size=False):
        """
        :param degrees: 角度，可以是一个数值或者list
        :param ramdon_rate: 随机系数
        :param same_size: 是否保持和原图一样大
        :return:
        """
        # 如果 degrees 是一个数字
        if isinstance(degrees, numbers.Number):
            # 如果角度小于0，抛出数值错误异常
            if degrees < 0:
                raise ValueError(
                    "If degrees is a single number, it must be positive.")
            # 将角度转换为元组形式
            degrees = (-degrees, degrees)
        # 如果 degrees 是一个列表、元组或者 numpy 数组
        elif isinstance(degrees, list) or isinstance(
                degrees, tuple) or isinstance(degrees, np.ndarray):
            # 如果角度的长度不为2，抛出数值错误异常
            if len(degrees) != 2:
                raise ValueError(
                    "If degrees is a sequence, it must be of len 2.")
            # 保持角度不变
            degrees = degrees
        else:
            # 如果 degrees 不是数字、列表、元组或者 numpy 数组，抛出异常
            raise Exception(
                'degrees must in Number or list or tuple or np.ndarray')
        # 将处理后的角度、是否保持原图大小和随机系数赋值给对象属性
        self.degrees = degrees
        self.same_size = same_size
        self.random_rate = random_rate
    def __call__(self, data: dict) -> dict:
        """
        从scales中随机选择一个尺度，对图片和文本框进行缩放
        :param data: {'img':,'text_polys':,'texts':,'ignore_tags'}
        :return: 返回处理后的数据字典
        """
        # 如果随机数大于设定的概率，则直接返回原始数据
        if random.random() > self.random_rate:
            return data
        # 获取图像和文本框的数据
        im = data['img']
        text_polys = data['text_polys']

        # ---------------------- 旋转图像 ----------------------
        # 获取图像的宽度和高度
        w = im.shape[1]
        h = im.shape[0]
        # 随机生成旋转角度
        angle = np.random.uniform(self.degrees[0], self.degrees[1])

        if self.same_size:
            nw = w
            nh = h
        else:
            # 将角度转换为弧度
            rangle = np.deg2rad(angle)
            # 计算旋转后图像的宽度和高度
            nw = (abs(np.sin(rangle) * h) + abs(np.cos(rangle) * w))
            nh = (abs(np.cos(rangle) * h) + abs(np.sin(rangle) * w))
        # 构造旋转矩阵
        rot_mat = cv2.getRotationMatrix2D((nw * 0.5, nh * 0.5), angle, 1)
        # 计算图像中心点的偏移量
        rot_move = np.dot(rot_mat, np.array([(nw - w) * 0.5, (nh - h) * 0.5, 0]))
        # 更新旋转矩阵
        rot_mat[0, 2] += rot_move[0]
        rot_mat[1, 2] += rot_move[1]
        # 进行仿射变换
        rot_img = cv2.warpAffine(
            im,
            rot_mat, (int(math.ceil(nw)), int(math.ceil(nh))),
            flags=cv2.INTER_LANCZOS4)

        # ---------------------- 矫正bbox坐标 ----------------------
        # 获取旋转后的文本框坐标
        rot_text_polys = list()
        for bbox in text_polys:
            point1 = np.dot(rot_mat, np.array([bbox[0, 0], bbox[0, 1], 1]))
            point2 = np.dot(rot_mat, np.array([bbox[1, 0], bbox[1, 1], 1]))
            point3 = np.dot(rot_mat, np.array([bbox[2, 0], bbox[2, 1], 1]))
            point4 = np.dot(rot_mat, np.array([bbox[3, 0], bbox[3, 1], 1]))
            rot_text_polys.append([point1, point2, point3, point4])
        # 更新数据字典中的图像和文本框数据
        data['img'] = rot_img
        data['text_polys'] = np.array(rot_text_polys)
        return data
class RandomResize:
    def __init__(self, size, random_rate, keep_ratio=False):
        """
        :param input_size: resize尺寸,数字或者list的形式，如果为list形式，就是[w,h]
        :param ramdon_rate: 随机系数
        :param keep_ratio: 是否保持长宽比
        :return:
        """
        # 初始化函数，设置resize尺寸、随机系数和是否保持长宽比
        if isinstance(size, numbers.Number):
            if size < 0:
                raise ValueError(
                    "If input_size is a single number, it must be positive.")
            size = (size, size)
        elif isinstance(size, list) or isinstance(size, tuple) or isinstance(
                size, np.ndarray):
            if len(size) != 2:
                raise ValueError(
                    "If input_size is a sequence, it must be of len 2.")
            size = (size[0], size[1])
        else:
            raise Exception(
                'input_size must in Number or list or tuple or np.ndarray')
        self.size = size
        self.keep_ratio = keep_ratio
        self.random_rate = random_rate

    def __call__(self, data: dict) -> dict:
        """
        从scales中随机选择一个尺度，对图片和文本框进行缩放
        :param data: {'img':,'text_polys':,'texts':,'ignore_tags':}
        :return:
        """
        # 调用实例时的函数，根据随机系数对图片和文本框进行缩放
        if random.random() > self.random_rate:
            return data
        im = data['img']
        text_polys = data['text_polys']

        if self.keep_ratio:
            # 将图片短边pad到和长边一样
            h, w, c = im.shape
            max_h = max(h, self.size[0])
            max_w = max(w, self.size[1])
            im_padded = np.zeros((max_h, max_w, c), dtype=np.uint8)
            im_padded[:h, :w] = im.copy()
            im = im_padded
        text_polys = text_polys.astype(np.float32)
        h, w, _ = im.shape
        im = cv2.resize(im, self.size)
        w_scale = self.size[0] / float(w)
        h_scale = self.size[1] / float(h)
        text_polys[:, :, 0] *= w_scale
        text_polys[:, :, 1] *= h_scale

        data['img'] = im
        data['text_polys'] = text_polys
        return data
# 定义一个函数，用于调整图像大小至指定短边长度
def resize_image(img, short_size):
    # 获取图像的高度、宽度和通道数
    height, width, _ = img.shape
    # 判断图像的高度和宽度，以确定调整后的新高度和宽度
    if height < width:
        new_height = short_size
        new_width = new_height / height * width
    else:
        new_width = short_size
        new_height = new_width / width * height
    # 将新高度和宽度调整为32的倍数
    new_height = int(round(new_height / 32) * 32)
    new_width = int(round(new_width / 32) * 32)
    # 使用OpenCV的resize函数调整图像大小
    resized_img = cv2.resize(img, (new_width, new_height))
    return resized_img, (new_width / width, new_height / height)


# 定义一个类，用于按照指定短边长度调整图像大小
class ResizeShortSize:
    def __init__(self, short_size, resize_text_polys=True):
        """
        :param size: resize尺寸,数字或者list的形式，如果为list形式，就是[w,h]
        :return:
        """
        # 初始化函数，设置短边长度和是否调整文本框
        self.short_size = short_size
        self.resize_text_polys = resize_text_polys

    def __call__(self, data: dict) -> dict:
        """
        对图片和文本框进行缩放
        :param data: {'img':,'text_polys':,'texts':,'ignore_tags':}
        :return:
        """
        # 获取输入数据中的图像和文本框
        im = data['img']
        text_polys = data['text_polys']

        h, w, _ = im.shape
        # 计算图像的短边长度
        short_edge = min(h, w)
        if short_edge < self.short_size:
            # 如果短边长度小于指定长度，则按比例调整图像大小
            scale = self.short_size / short_edge
            im = cv2.resize(im, dsize=None, fx=scale, fy=scale)
            scale = (scale, scale)
            # 如果需要调整文本框大小，则按比例调整文本框坐标
            if self.resize_text_polys:
                text_polys[:, 0] *= scale[0]
                text_polys[:, 1] *= scale[1]

        # 更新数据中的图像和文本框信息，并返回
        data['img'] = im
        data['text_polys'] = text_polys
        return data


# 定义一个类，用于水平翻转图像
class HorizontalFlip:
    def __init__(self, random_rate):
        """

        :param random_rate: 随机系数
        """
        # 初始化函数，设置水平翻转的随机系数
        self.random_rate = random_rate
    # 定义一个方法，用于对输入的数据进行处理
    def __call__(self, data: dict) -> dict:
        """
        从scales中随机选择一个尺度，对图片和文本框进行缩放
        :param data: {'img':,'text_polys':,'texts':,'ignore_tags'}
        :return: 返回处理后的数据字典
        """
        # 如果生成的随机数大于设定的随机率，则直接返回原始数据
        if random.random() > self.random_rate:
            return data
        # 获取输入数据中的图片和文本框信息
        im = data['img']
        text_polys = data['text_polys']

        # 复制文本框信息
        flip_text_polys = text_polys.copy()
        # 水平翻转图片
        flip_im = cv2.flip(im, 1)
        # 获取翻转后图片的高度和宽度
        h, w, _ = flip_im.shape
        # 更新翻转后文本框的 x 坐标
        flip_text_polys[:, :, 0] = w - flip_text_polys[:, :, 0]

        # 更新数据字典中的图片和文本框信息
        data['img'] = flip_im
        data['text_polys'] = flip_text_polys
        # 返回处理后的数据字典
        return data
class VerticallFlip:
    def __init__(self, random_rate):
        """
        初始化函数，接收一个随机系数参数
        :param random_rate: 随机系数
        """
        self.random_rate = random_rate

    def __call__(self, data: dict) -> dict:
        """
        对传入的数据进行垂直翻转操作
        :param data: {'img':,'text_polys':,'texts':,'ignore_tags'}
        :return: 处理后的数据字典
        """
        # 根据随机系数判断是否进行翻转操作
        if random.random() > self.random_rate:
            return data
        # 获取图像和文本框数据
        im = data['img']
        text_polys = data['text_polys']

        # 复制文本框数据
        flip_text_polys = text_polys.copy()
        # 对图像进行垂直翻转
        flip_im = cv2.flip(im, 0)
        h, w, _ = flip_im.shape
        # 更新文本框的坐标信息
        flip_text_polys[:, :, 1] = h - flip_text_polys[:, :, 1]
        # 更新数据字典中的图像和文本框数据
        data['img'] = flip_im
        data['text_polys'] = flip_text_polys
        return data
```