<!--yml
category: 游戏
date: 2023-09-17 14:44:22
-->

# YOLOv5-6.x源码分析（七）---- 数据增强之augmentations.py

> 来源：[https://blog.csdn.net/weixin_51322383/article/details/130409656](https://blog.csdn.net/weixin_51322383/article/details/130409656)

### 文章目录

*   [前言](#_1)
*   [🚀YOLOv5-6.x源码分析（七）---- 数据增强之augmentations.py](#YOLOv56x_augmentationspy_11)
*   *   [1\. 导包](#1__24)
    *   [2\. 自定义Albumentations](#2__Albumentations_43)
    *   [3\. 归一化和反规范化](#3__90)
    *   [4\. hsv 色调-饱和度-亮度的图像增强](#4_hsv__110)
    *   [5\. 直方图均衡化增强](#5__133)
    *   [6\. 图像框的平移复制增强](#6__153)
    *   [7\. 图片缩放letterbox](#7_letterbox_173)
    *   [8\. 随机透视变换](#8__220)
    *   [9\. cutout](#9_cutout_343)
    *   [10\. mixup](#10_mixup_385)
    *   [11\. box_candidates](#11_box_candidates_400)
    *   [总结](#_413)

# 前言

**今天上午刚回南山本校办了点事情，来回一趟就花了整整一上午，太累了，回到实验室就戴上眼罩睡了十几分钟。刚刚上辅导员的职规课，她讲的一些东西，我也不知道算不算有用吧，无非就是一些企业走访，我感觉没多少用处，下面的同学也没几个听的。包括上周学院带我们班去参观公司也是，我觉得真是纯粹浪费时间，每次一去就是介绍公司文化，感觉学生去就是凑数的。反观，我感觉现在大学里面有用的课真的太少了，就拿求职这方面来说，大学里面根本没有教你如何签合同，而这些才是我们大学生最该上的也最该学的一些技能，这么重要的东西居然仅仅想通过一节课就给我们讲清楚；包括我们如果遇到一些劳务纠纷如何解决，这些都没有。反倒从大一开始的思政课，就有3~4学分，而大一一门C语言专业课才只有3分。今天跟同学谈到，大学教会我们的唯一东西就是告诉我们什么东西都得要自学，学校里根本教不了什么东西。这或许也是我选择要升学的目的，一方面是如今行情太差了， 另一方面我也想通过自己多花点时间多学点知识，并且先多接触下社会，以免到时候一进入社会就被现实敲得粉碎。**

* * *

**说回主题，今天准备剖析的是数据增强部分。**

**我们知道，要完成很多实际的项目，我们都要充足的数据来完成任务，这样才能适应多场景的任务。但是我们的目标应用可能存在于不同的条件，比如在不同的方向、位置、缩放比例、亮度等。而单靠自己寻找数据远远不够，这个时候就需要数据增强。其实就是将数据，通过额外合成的数据来训练神经网络来解释这些情况。**

**什么是数据增强呢？数据增强也叫数据扩增，意思是在不实质性的增加数据的情况下，让有限的数据产生等价于更多数据的价值。数据增强有很多种方法，YOLOv5中就有十几种，下面我会介绍用得比较多的，let’s get it~**

# 🚀YOLOv5-6.x源码分析（七）---- 数据增强之augmentations.py

总的来说，YOLOv5-6.1涉及到的数据增强方法主要有以下几种：

*   对原图做数据增强
    *   像素级：HSV增强、旋转、缩放、平移、剪切、透视、翻转等
    *   图片级：MixUp、Cutout、CutMix、Mosaic、Copy-Paste(Segment)等
*   对标签做同样的增强
    *   变换后的坐标偏移量
    *   防止标签坐标越界

导航：[YOLOv5-6.x源码分析 全流程记录](https://blog.csdn.net/weixin_51322383/article/details/130353834?spm=1001.2014.3001.5502)

* * *

## 1\. 导包

```py
import math
import random

import cv2
import numpy as np
import torch
import torchvision.transforms as T  # 图像预处理工具包
import torchvision.transforms.functional as TF  # 图像变换的函数库

from utils.general import LOGGER, check_version, colorstr, resample_segments, segment2box, xywhn2xyxy   # 常用工具函数
from utils.metrics import bbox_ioa  # 计算IoU与box2面积的比值

IMAGENET_MEAN = 0.485, 0.456, 0.406  # RGB mean RGB均值
IMAGENET_STD = 0.229, 0.224, 0.225  # RGB standard deviation RGB标准偏差 
```

## 2\. 自定义Albumentations

```py
class Albumentations:
    # YOLOv5 Albumentations class (optional, only used if package is installed)
    def __init__(self, size=640):
        self.transform = None
        prefix = colorstr('albumentations: ')
        try:
            import albumentations as A
            check_version(A.__version__, '1.0.3', hard=True)  # version requirement

            T = [
                A.RandomResizedCrop(height=size, width=size, scale=(0.8, 1.0), ratio=(0.9, 1.11), p=0.0),
                A.Blur(p=0.01),
                A.MedianBlur(p=0.01),
                A.ToGray(p=0.01),
                A.CLAHE(p=0.01),
                A.RandomBrightnessContrast(p=0.0),
                A.RandomGamma(p=0.0),
                A.ImageCompression(quality_lower=75, p=0.0)]  # transforms
            # A.Compose 用于图像增强和数据增强
            # https://blog.csdn.net/u014264373/article/details/114144303 最快最好用的数据增强库「albumentations」 一文看懂用法
            # bbox_params参数定义了格式 format; label_fileds:表示自定义的类标签变量的名字，是一个列表，可以放置多个参数名称，表示多标签。
            self.transform = A.Compose(T, bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

            LOGGER.info(prefix + ', '.join(f'{x}'.replace('always_apply=False, ', '') for x in T if x.p))
        # 如果没有安装会跳过
        except ImportError:  # package not installed, skip
            pass
        except Exception as e:
            LOGGER.info(f'{prefix}{e}')

    # 相当于C++的仿函数，可以像函数一样调用该类，接收参数并返回值
    def __call__(self, im, labels, p=1.0):
        if self.transform and random.random() < p:
            # 传入图片，获取到数据增强后的图片
            new = self.transform(image=im, bboxes=labels[:, 1:], class_labels=labels[:, 0])  # transformed
            im, labels = new['image'], np.array([[c, *b] for c, b in zip(new['class_labels'], new['bboxes'])])
        return im, labels 
```

**这个类只会在你安装了albumentations这个库的时候使用，如果没有安装的话，也不会报错,不过都用torchvision.transforms里面的内容，I guess**

这个类主要是重新定义了一下YOLO格式的数据增强，加入了`format`为yolo格式。并且还定义了`__call__`方法，这个相当于C++的仿函数，可以像函数一样调用该类，实际上就是返回图片和标签，不过没做什么数据增强。重点是`A.Compose`，实际上内部还是调用`albumentations`这个类的。

> 具体使用方法可看：[最快最好用的数据增强库「albumentations」 一文看懂用法](https://blog.csdn.net/u014264373/article/details/114144303)

## 3\. 归一化和反规范化

```py
# 归一化
def normalize(x, mean=IMAGENET_MEAN, std=IMAGENET_STD, inplace=False):
    # Denormalize RGB images x per ImageNet stats in BCHW format, i.e. = (x - mean) / std
    return TF.normalize(x, mean, std, inplace=inplace)

# 反规范化
def denormalize(x, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    # Denormalize RGB images x per ImageNet stats in BCHW format, i.e. = x * std + mean
    for i in range(3):
        x[:, i] = x[:, i] * std[i] + mean[i]
    return x 
```

**归一化就是直接调用TF的函数，输入参数mean均值和std方差，inplace:是否就地计算 (相当于x += b和y = x+b,x=y的区别) 默认为False**

**反规范化就是值乘以方差再加上均值**

## 4\. hsv 色调-饱和度-亮度的图像增强

```py
def augment_hsv(im, hgain=0.5, sgain=0.5, vgain=0.5):   # 做h-色调， s-饱和度， v-亮度上面的随机增强
    # HSV color-space augmentation
    if hgain or sgain or vgain: # random gains 生成3个[-1, 1)之间的随机数，分别与hsv相乘后+1 [0,2]之间
        r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
        hue, sat, val = cv2.split(cv2.cvtColor(im, cv2.COLOR_BGR2HSV))
        dtype = im.dtype  # uint8

        x = np.arange(0, 256, dtype=r.dtype)    # [0,1,...,255]
        lut_hue = ((x * r[0]) % 180).astype(dtype)  # 0~180
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)   # 将数组截断至[0, 255]
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        # cv2.LUT lookup-table 查找表方式，即通过lut_hue 这个表对之前hue数值做修正，返回0-255对应位置的lut_hue值  具体： https://blog.csdn.net/Dontla/article/details/103963085
        # cv2.merge 合并三个通道
        cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR, dst=im)  # no return needed 
```

**做一个随机的色调、饱和度、亮度亮度增强**

## 5\. 直方图均衡化增强

```py
def hist_equalize(im, clahe=True, bgr=False):   # 直方图均衡化增强 参考 https://www.cnblogs.com/my-love-is-python/p/10405811.html
    # Equalize histogram on BGR image 'im' with im.shape(n,m,3) and range 0-255
    yuv = cv2.cvtColor(im, cv2.COLOR_BGR2YUV if bgr else cv2.COLOR_RGB2YUV) # bgr -> YUV
    if clahe:
        c = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        # cv2.createCLAHE 实例化自适应直方图均衡化函数 局部直方图均衡化 ，不会使得细节消失
        # c.apply 进行自适应直方图均衡化
        yuv[:, :, 0] = c.apply(yuv[:, :, 0])
    else:
        # cv2.equalizeHist 进行像素点的均衡化 ，即全局均衡化 ，使得整体亮度提升，但是局部会模糊 
        yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])  # equalize Y channel histogram
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR if bgr else cv2.COLOR_YUV2RGB)  # convert YUV image to RGB 
```

**先判断clahe是否为true，如果是就先将图片转化为YUV格式，然后采用`cv.createCLAHE`,这个方法是实例化自适应直方图均衡化函数 局部直方图均衡化 ，不会使得细节消失，然后再用`c.apply`进行自适应直方图均衡化。**

具体可看[直方图均衡化](https://www.cnblogs.com/my-love-is-python/p/10405811.html)

## 6\. 图像框的平移复制增强

```py
def replicate(im, labels):  # 复制，实际上指的是框的平移
    # Replicate labels
    h, w = im.shape[:2] # 获取图像长宽
    boxes = labels[:, 1:].astype(int)   # 获取框的位置和大小
    x1, y1, x2, y2 = boxes.T    # 框的左右和上下位置
    s = ((x2 - x1) + (y2 - y1)) / 2  # side length (pixels)
    for i in s.argsort()[:round(s.size * 0.5)]:  # smallest indices
        x1b, y1b, x2b, y2b = boxes[i]
        bh, bw = y2b - y1b, x2b - x1b
        yc, xc = int(random.uniform(0, h - bh)), int(random.uniform(0, w - bw))  # offset x, y
        x1a, y1a, x2a, y2a = [xc, yc, xc + bw, yc + bh]
        im[y1a:y2a, x1a:x2a] = im[y1b:y2b, x1b:x2b]  # im4[ymin:ymax, xmin:xmax]
        labels = np.append(labels, [[labels[i, 0], x1a, y1a, x2a, y2a]], axis=0)

    return im, labels 
```

## 7\. 图片缩放letterbox

```py
def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):  # 如果是1个数字，默认长宽相等
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old) 计算收缩比，选择较小的
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    # 计算收缩后图片的长宽
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    # 计算需要填充的边的像素
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    # 除以2即最终每边填充的像素
    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        # 先将图片按比例缩放到指定大小
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))    # 上下位置
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))    # 左右位置
    # cv2.copyMakeBorder 对im设置边界框
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh) 
```

**这个函数挺重要的。**

**letterbox的主要思想是尽可能的利用网络感受野的信息特征。比如在YOLOv5中最后一层的Stride=5，即最后一层的特征图中每个点，可以对应原图中32X32的区域信息**，那么只要在保证整体图片变换比例一致的情况下，长宽均可以被32整除，那么就可以有效的利用感受野的信息。

假设图片原来尺寸为（1080， 1920），我们想要resize的尺寸为（640，640）。要想满足收缩的要求，应该选取收缩比例640/1920 = 0.33.则图片被缩放为（360，640）.下一步则要填充灰白边至360可以被32整除，则应该填充至384，最终得到图片尺寸（384，640）

## 8\. 随机透视变换

```py
def random_perspective(im,          # mosaic整合后的图片img4 [2*img_size, 2*img_size]
                       targets=(),  # mosaic整合后图片的所有正常label标签labels4(不正常的会通过segments2boxes将多边形标签转化为正常标签) [N, cls+xyxy]
                       segments=(),  # mosaic整合后图片的所有不正常label信息(包含segments多边形也包含正常gt)  [m, x1y1....]
                       degrees=10,  # 旋转和缩放矩阵参数
                       translate=.1,    # 平移矩阵参数
                       scale=.1,    # 缩放矩阵参数
                       shear=10,    # 剪切矩阵参数
                       perspective=0.0, # 透视变换参数
                       border=(0, 0)):  # 用于确定最后输出的图片大小 一般等于[-img_size, -img_size] 那么最后输出的图片大小为 [img_size, img_size]
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(0.1, 0.1), scale=(0.9, 1.1), shear=(-10, 10))
    # targets = [cls, xyxy]

    height = im.shape[0] + border[0] * 2  # shape(h,w,c)
    width = im.shape[1] + border[1] * 2

    # Center 计算中心点
    C = np.eye(3)   # 生成3*3的对角为1的对角矩阵
    C[0, 2] = -im.shape[1] / 2  # x translation (pixels)
    C[1, 2] = -im.shape[0] / 2  # y translation (pixels)

    # Perspective
    # 透视
    P = np.eye(3)
    P[2, 0] = random.uniform(-perspective, perspective)  # x perspective (about y)
    P[2, 1] = random.uniform(-perspective, perspective)  # y perspective (about x)

    # Rotation and Scale
    # 旋转和缩放
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    s = random.uniform(1 - scale, 1 + scale) #随机生成缩放比例
    # s = 2 ** random.uniform(-scale, scale)
    # 图片旋转得到仿射变化矩阵赋给R的前两行
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear
    # 弯曲角度
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Translation
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width  # x translation (pixels)
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height  # y translation (pixels)

    # Combined rotation matrix
    # 组合旋转矩阵
    M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
    # 通过矩阵乘法组合
    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
        if perspective: # 如果透视
            # cv2.warpPerspective透视变换函数，可保持直线不变形，但是平行线可能不再平行
            im = cv2.warpPerspective(im, M, dsize=(width, height), borderValue=(114, 114, 114))
        else:  # affine
            # cv2.warpAffine放射变换函数，可实现旋转，平移，缩放，并且变换后的平行线依旧平行
            im = cv2.warpAffine(im, M[:2], dsize=(width, height), borderValue=(114, 114, 114))

    # Visualize
    # import matplotlib.pyplot as plt
    # ax = plt.subplots(1, 2, figsize=(12, 6))[1].ravel()
    # ax[0].imshow(im[:, :, ::-1])  # base
    # ax[1].imshow(im2[:, :, ::-1])  # warped

    # Transform label coordinates
    # 变换标签坐标
    n = len(targets)
    if n:
        # 判断segments是否为空或是否全为0（目标像素段）
        use_segments = any(x.any() for x in segments) and len(segments) == n
        new = np.zeros((n, 4))
        # 如果使用的是segments标签(标签中含有多边形gt)
        if use_segments:  # warp segments
            #上采样
            segments = resample_segments(segments)  # upsample
            for i, segment in enumerate(segments):
                xy = np.ones((len(segment), 3))
                xy[:, :2] = segment
                xy = xy @ M.T  # transform
                xy = xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]  # perspective rescale or affine

                # clip
                new[i] = segment2box(xy, width, height)

        else:  # warp boxes
            xy = np.ones((n * 4, 3))
            xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
            xy = xy @ M.T  # transform
            xy = (xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]).reshape(n, 8)  # perspective rescale or affine

            # create new boxes
            x = xy[:, [0, 2, 4, 6]]
            y = xy[:, [1, 3, 5, 7]]
            new = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

            # clip
            new[:, [0, 2]] = new[:, [0, 2]].clip(0, width)
            new[:, [1, 3]] = new[:, [1, 3]].clip(0, height)

        # filter candidates
        i = box_candidates(box1=targets[:, 1:5].T * s, box2=new.T, area_thr=0.01 if use_segments else 0.10)
        targets = targets[i]
        targets[:, 1:5] = new[i]

    return im, targets 
```

**这个函数会用于`load_mosaic`中用在mosaic操作之后，还是蛮重要的。**

**这段代码包括对图片的旋转、缩放、透视、弯曲、放大缩小的随机变化，每一个操作都通过创建一个`3*3`的矩阵，最后相乘，进行变换**

**最后还要调整标签信息，只有多边形gt时才有。**

**Mosaic**
![在这里插入图片描述](img/4d7f4b6545d69835383c6c75982d53ad.png)
**经过mosaic + random_perspective**
![在这里插入图片描述](img/488208f432d24cf9d35001d45ba1328f.png)

![在这里插入图片描述](img/91ac6b698910da1f413c65243b1b9ebe.png)

## 9\. cutout

```py
def cutout(im, labels, p=0.5):
    # Applies image cutout augmentation https://arxiv.org/abs/1708.04552
    """用在LoadImagesAndLabels模块中的__getitem__函数进行cutout增强  v5源码作者默认是没用用这个的 感兴趣的可以测试一下
        cutout数据增强, 给图片随机添加随机大小的方块噪声  目的是提高泛化能力和鲁棒性
        实现：随机选择一个固定大小的正方形区域，然后采用全0填充就OK了，当然为了避免填充0值对训练的影响，应该要对数据进行中心归一化操作，norm到0。
        论文: https://arxiv.org/abs/1708.04552
        :params image: 一张图片 [640, 640, 3] numpy
        :params labels: 这张图片的标签 [N, 5]=[N, cls+x1y1x2y2]
        :return labels: 筛选后的这张图片的标签 [M, 5]=[M, cls+x1y1x2y2]  M<N
                        筛选: 如果随机生成的噪声和原始的gt框相交区域占gt框太大 就筛出这个gt框label
        """
    if random.random() < p:
        h, w = im.shape[:2]
        scales = [0.5] * 1 + [0.25] * 2 + [0.125] * 4 + [0.0625] * 8 + [0.03125] * 16  # image size fraction
        for s in scales:
            mask_h = random.randint(1, int(h * s))  # create random masks
            mask_w = random.randint(1, int(w * s))

            # box 随机生成噪声
            xmin = max(0, random.randint(0, w) - mask_w // 2)
            ymin = max(0, random.randint(0, h) - mask_h // 2)
            xmax = min(w, xmin + mask_w)
            ymax = min(h, ymin + mask_h)

            # apply random color mask 添加随机颜色的噪声 
            im[ymin:ymax, xmin:xmax] = [random.randint(64, 191) for _ in range(3)]

            # return unobscured labels 返回没有噪声的label
            if len(labels) and s > 0.03:
                box = np.array([xmin, ymin, xmax, ymax], dtype=np.float32)
                ioa = bbox_ioa(box, xywhn2xyxy(labels[:, 1:5], w, h))  # intersection over area
                labels = labels[ioa < 0.60]  # remove >60% obscured labels

    return labels 
```

**cutout数据增强，用在LoadImagesAndLabels模块中的__getitem__函数进行cutout增强。给图片随机添加随机大小的方块噪声 ，目的是提高泛化能力和鲁棒性。**

**作者没有使用这个，可以自己试一试。**

## 10\. mixup

```py
def mixup(im, labels, im2, labels2):
    # Applies MixUp augmentation https://arxiv.org/pdf/1710.09412.pdf
    r = np.random.beta(32.0, 32.0)  # mixup ratio, alpha=beta=32.0
    im = (im * r + im2 * (1 - r)).astype(np.uint8)
    labels = np.concatenate((labels, labels2), 0)
    return im, labels 
```

**将两张图片按比例融合起来，labels就相同维度concat起来**

**也用在`LoadImagesAndLabels`的`__getitem__`中，进行数据增强**
![在这里插入图片描述](img/f602b3b524b3cbb7779e3109019e5681.png)

## 11\. box_candidates

```py
def box_candidates(box1, box2, wh_thr=2, ar_thr=100, area_thr=0.1, eps=1e-16):  # box1(4,n), box2(4,n)
    # Compute candidate boxes: box1 before augment, box2 after augment, wh_thr (pixels), aspect_ratio_thr, area_ratio
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    ar = np.maximum(w2 / (h2 + eps), h2 / (w2 + eps))  # aspect ratio
    return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + eps) > area_thr) & (ar < ar_thr)  # candidates 
```

**这个函数用在random_perspective中，是对透视变换后的图片的label进行筛选，增强后w、h要大于2 增强后图像与增强前图像面积比值大于area_thr 宽高比大于ar_thr**

## 总结

**这篇主要讲了YOLOv5中的各种数据增强方法。其中[图片缩放](#7_letterbox_171)和[随机透视](#8__218)变换特别重要，尤其是后者会在`Mosaic`过后用到。其他的了解下即可。**

**References**

> CSDN 嗜睡的篠龙[【YOLOv5-6.x】数据增强代码解析](https://blog.csdn.net/weixin_43799388/article/details/123830587)
> CSDN Tina姐 [最快最好用的数据增强库「albumentations」 一文看懂用法](https://blog.csdn.net/u014264373/article/details/114144303)
> CSDN 满船清梦压星河HK[【YOLOV5-5.x 源码解读】datasets.py](https://blog.csdn.net/qq_38253797/article/details/119904518)