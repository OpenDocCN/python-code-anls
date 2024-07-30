<!--yml
category: 游戏
date: 2023-09-17 14:44:33
-->

# YOLOv5-6.x源码分析（六）---- 数据集创建之dataloaders.py

> 来源：[https://blog.csdn.net/weixin_51322383/article/details/130387945](https://blog.csdn.net/weixin_51322383/article/details/130387945)

### 文章目录

*   [前言](#_1)
*   [🚀YOLOv5-6.x源码分析（六）---- 数据集创建之dataloaders.py](#YOLOv56x_dataloaderspy_6)
*   *   [1\. 导包](#1__7)
    *   [2\. 相机设置](#2__50)
    *   [3\. create_dataloader](#3_create_dataloader_79)
    *   [4\. 自定义DataLoader](#4_DataLoader_141)
    *   [5\. LoadImagesAndLabels](#5_LoadImagesAndLabels_178)
    *   *   [5.1 __init__](#51___init___184)
        *   [5.2 cache_labels](#52_cache_labels_360)
        *   [5.3 len](#53_len_408)
        *   [5.4 getitem](#54_getitem_417)
        *   [5.5 collate_fn](#55_collate_fn_532)
    *   [6\. img2label_paths](#6_img2label_paths_572)
    *   [7\. verify_image_label](#7_verify_image_label_596)
    *   [8\. load_image](#8_load_image_663)
    *   [9\. load_mosaic](#9_load_mosaic_699)
    *   [总结](#_793)

# 前言

这个文件主要是创建数据集+各种数据增强操作。

导航：[YOLOv5-6.x源码分析 全流程记录](https://blog.csdn.net/weixin_51322383/article/details/130353834?spm=1001.2014.3001.5502)

* * *

# 🚀YOLOv5-6.x源码分析（六）---- 数据集创建之dataloaders.py

## 1\. 导包

```py
import cv2
import contextlib
import glob # 文件操作相关模块
import hashlib  # 哈希模块，童工了多种安全方便的hash方法
import json
import math
import os
import random
import shutil   # 文件夹、压缩包处理模块
import time
from itertools import repeat    # 复制模块
from multiprocessing.pool import Pool, ThreadPool   # 多线程模块。线程池
from pathlib import Path
from threading import Thread
from urllib.parse import urlparse
from zipfile import ZipFile

import numpy as np
import torch
import torch.nn.functional as F # 封装了很多卷积、池化函数
import yaml     # yaml文件操作模块
from PIL import ExifTags, Image, ImageOps       # 图片、相机操作模块
from torch.utils.data import DataLoader, Dataset, dataloader, distributed   # 自定义数据集模块
from tqdm import tqdm

from utils.augmentations import Albumentations, augment_hsv, copy_paste, letterbox, mixup, random_perspective   # 数据增强
from utils.general import (DATASETS_DIR, LOGGER, NUM_THREADS, check_dataset, check_requirements, check_yaml, clean_str,
                           cv2, is_colab, is_kaggle, segments2boxes, xyn2xy, xywh2xyxy, xywhn2xyxy, xyxy2xywhn) # 常用的一些工具函数
from utils.torch_utils import torch_distributed_zero_first  # 分布式训练相关

# Parameters
HELP_URL = 'https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data'
IMG_FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp'  # include image suffixes    图片格式
VID_FORMATS = 'asf', 'avi', 'gif', 'm4v', 'mkv', 'mov', 'mp4', 'mpeg', 'mpg', 'ts', 'wmv'  # include video suffixes 视频格式
BAR_FORMAT = '{l_bar}{bar:10}{r_bar}{bar:-10b}'  # tqdm bar format 
LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html 在整个分布式中的序号，每个进程都有一个rank和一个local_rank 
```

关于local_rank的理解：[local_rank，rank，node等理解](https://blog.csdn.net/shenjianhua005/article/details/127318594) 

* * *

## 2\. 相机设置

```py
# 这部分是相机相关设置，当使用相机采样时才会使用。
# Get orientation exif tag
# 可交换图像文件格式 是专门为数码相机的照片设定的，可以记录数码照片的属性信息和拍摄数据
for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == 'Orientation':
        break

# 返回文件列表的hash值
def get_hash(paths):
    # Returns a single hash value of a list of paths (files or dirs)
    size = sum(os.path.getsize(p) for p in paths if os.path.exists(p))  # sizes
    h = hashlib.md5(str(size).encode())  # hash sizes
    h.update(''.join(paths).encode())  # hash paths
    return h.hexdigest()  # return hash

# 获取图片的宽高信息
def exif_size(img):
    # Returns exif-corrected PIL size
    # 获取数码相机的图片宽高信息  并且判断是否需要旋转（数码相机可以多角度拍摄）
    s = img.size  # (width, height)
    with contextlib.suppress(Exception):
        rotation = dict(img._getexif().items())[orientation]    # 调整数码相机照片方向
        if rotation in [6, 8]:  # rotation 270 or 90
            s = (s[1], s[0])
    return s 
```

## 3\. create_dataloader

```py
def create_dataloader(path,         # 图片数据加载路径 train/test
                      imgsz,        # train/test图片尺寸（数据增强后大小） 640
                      batch_size,   # batch size 大小 8/16/32
                      stride,       # 模型最大stride=32   [32 16 8]
                      single_cls=False,     # 数据集是否是单类别 默认False
                      hyp=None,             # 超参列表dict 网络训练时的一些超参数，包括学习率等，这里主要用到里面一些关于数据增强(旋转、平移等)的系数
                      augment=False,        # 是否要进行数据增强  True
                      cache=False,          # 是否cache_images False
                      pad=0.0,      # 设置矩形训练的shape时进行的填充 默认0.0
                      rect=False,   # 是否开启矩形train/test  默认训练集关闭 验证集开启
                      rank=-1,      # 多卡训练时的进程编号 rank为进程编号  -1且gpu=1时不进行分布式  -1且多块gpu使用DataParallel模式  默认-1
                      workers=8,
                      image_weights=False,  # 训练时是否根据图片样本真实框分布权重来选择图片  默认False
                      quad=False,
                      prefix='',            # 显示信息  一个标志 多为train/val，处理标签时保存cache文件会用到
                      shuffle=False):
    # 是否使用矩形训练模式
    if rect and shuffle:    # 做一个保护，rect时不能打乱shuffle（因为序列是固定的）
        LOGGER.warning('WARNING: --rect is incompatible with DataLoader shuffle, setting shuffle=False')
        shuffle = False
    # 主进程实现数据的预读取并缓存，然后其它子进程则从缓存中读取数据并进行一系列运算。
    # 为了完成数据的正常同步, yolov5基于torch.distributed.barrier()函数实现了上下文管理器
    with torch_distributed_zero_first(rank):  # init dataset *.cache only once if DDP   分布式
        # 载入文件数据(增强数据集)
        dataset = LoadImagesAndLabels(
            path,
            imgsz,
            batch_size,
            augment=augment,  # augmentation
            hyp=hyp,  # hyperparameters
            rect=rect,  # rectangular batches
            cache_images=cache,
            single_cls=single_cls,
            stride=int(stride),
            pad=pad,
            image_weights=image_weights,
            prefix=prefix)

    batch_size = min(batch_size, len(dataset))
    nd = torch.cuda.device_count()  # number of CUDA devices
    nw = min([os.cpu_count() // max(nd, 1), batch_size if batch_size > 1 else 0, workers])  # number of workers
    # 分布式采样器DistributedSampler
    sampler = None if rank == -1 else distributed.DistributedSampler(dataset, shuffle=shuffle)
    # 使用InfiniteDataLoader和_RepeatSampler来对DataLoader进行封装, 代替原D先的DataLoader, 能够永久持续的采样数据
    loader = DataLoader if image_weights else InfiniteDataLoader  # only DataLoader allows for attribute updates
    generator = torch.Generator()
    generator.manual_seed(0)
    return loader(dataset,
                  batch_size=batch_size,
                  shuffle=shuffle and sampler is None,
                  num_workers=0,
                  sampler=sampler,
                  pin_memory=True,
                  collate_fn=LoadImagesAndLabels.collate_fn4 if quad else LoadImagesAndLabels.collate_fn,
                  worker_init_fn=seed_worker,
                  generator=generator), dataset 
```

这个函数会在`train.py`中被调用，用于生成`dataloader`,`dataset`

## 4\. 自定义DataLoader

```py
# 当image_weights=False时（不根据图片样本真实框分布权重来选择图片）就会调用这两个函数 进行自定义DataLoader，进行持续性采样。在上面的create_dataloader模块中被调用。
class InfiniteDataLoader(dataloader.DataLoader):
    """ Dataloader that reuses workers

    Uses same syntax as vanilla DataLoader
    """
    # 使用InfiniteDataLoader和_RepeatSampler来对DataLoader进行封装, 代替原先的DataLoader, 能够永久持续的采样数据
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for _ in range(len(self)):
            yield next(self.iterator)

class _RepeatSampler:
    """ Sampler that repeats forever
    这部分是进行持续采样
    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler) 
```

## 5\. LoadImagesAndLabels

**这部分在create_dataloader里面用到。**

**起作用主要是数据加载，也是数据增强部分，即自定义数据集部分，继承自Dataset，主要是重写了`__getitem()__`方法。这个函数非常关键，是理解数据增强的关键。**

### 5.1 **init**

```py
def __init__(self,
             path,  # 数据path
             img_size=640,
             batch_size=16,
             augment=False, # 数据增强
             hyp=None,  # 超惨
             rect=False,
             image_weights=False,   # 图片权重
             cache_images=False,
             single_cls=False,
             stride=32,
             pad=0.0,
             prefix=''):

    # 1、赋值一些基础的self变量 用于后面在__getitem__中调用
    self.img_size = img_size    # 经过数据增强后的数据图片的大小
    self.augment = augment      # 是否启用数据增强
    self.hyp = hyp              # 超参数
    self.image_weights = image_weights  # 图片采样权重
    self.rect = False if image_weights else rect    # 矩阵训练   # 是否启动矩形训练 一般训练时关闭 验证时打开 可以加速
    # mosaic数据增强
    self.mosaic = self.augment and not self.rect  # load 4 images at a time into a mosaic (only during training) 四张图片拼成一张图
    self.mosaic_border = [-img_size // 2, -img_size // 2]
    self.stride = stride    # 模型下采样的步长
    self.path = path
    self.albumentations = Albumentations() if augment else None

    # 2、得到path路径下的所有图片的路径self.img_files
    try:
        f = []  # image files
        for p in path if isinstance(path, list) else [path]:
            # 获取数据集路径path，包含图片路径的txt文件或包含图片的文件夹路径
            # 使用pathlib.Path生成与操作系统无关的路径，因为不同操作系统路径的‘/’会有所不同
            p = Path(p)  # os-agnostic
            if p.is_dir():  # dir
                # glob.glab: 返回所有匹配的文件路径列表  递归获取p路径下所有文件
                f += glob.glob(str(p / '**' / '*.*'), recursive=True)
                # f = list(p.rglob('*.*'))  # pathlib
            elif p.is_file():  # file
                with open(p) as t:
                    t = t.read().strip().splitlines()   # strip:删除前导和尾随空格  splitlines()方法，按行将字符串分为字符串list
                    parent = str(p.parent) + os.sep # 获取数据集路径的上级父目录；os.sep为分隔符（不同操作系统的分隔符不一样）
                    f += [x.replace('./', parent) if x.startswith('./') else x for x in t]  # local to global path
                    # f += [p.parent / x.lstrip(os.sep) for x in t]  # local to global path (pathlib)
            else:
                raise FileNotFoundError(f'{prefix}{p} does not exist')
        self.im_files = sorted(x.replace('/', os.sep) for x in f if x.split('.')[-1].lower() in IMG_FORMATS)
        # self.img_files = sorted([x for x in f if x.suffix[1:].lower() in IMG_FORMATS])  # pathlib
        assert self.im_files, f'{prefix}No images found'
    except Exception as e:
        raise Exception(f'{prefix}Error loading data from {path}: {e}\nSee {HELP_URL}')

    # Check cache
    # 3、根据imgs路径找到labels的路径self.label_files
    self.label_files = img2label_paths(self.im_files)

    # 4、cache label 下次运行这个脚本的时候直接从cache中取label而不是去文件中取label 速度更快
    cache_path = (p if p.is_file() else Path(self.label_files[0]).parent).with_suffix('.cache')
    try:
        cache, exists = np.load(cache_path, allow_pickle=True).item(), True  # load dict
        assert cache['version'] == self.cache_version  # matches current version
        assert cache['hash'] == get_hash(self.label_files + self.im_files)  # identical hash
    except Exception:
        # 否则调用cache_labels缓存标签及标签相关信息
        cache, exists = self.cache_labels(cache_path, prefix), False  # run cache ops

    # Display cache
    # 打印cache的结果 nf nm ne nc n = 找到的标签数量，漏掉的标签数量，空的标签数量，损坏的标签数量，总的标签数量
    nf, nm, ne, nc, n = cache.pop('results')  # found, missing, empty, corrupt, total
    if exists and LOCAL_RANK in {-1, 0}:
        d = f"Scanning '{cache_path}' images and labels... {nf} found, {nm} missing, {ne} empty, {nc} corrupt"
        tqdm(None, desc=prefix + d, total=n, initial=n, bar_format=BAR_FORMAT)  # display cache results
        if cache['msgs']:
            LOGGER.info('\n'.join(cache['msgs']))  # display warnings
    # 数据集没有标签信息 就发出警告并显示标签label下载地址help_url
    assert nf > 0 or not augment, f'{prefix}No labels in {cache_path}. Can not train without labels. See {HELP_URL}'

    # Read cache
    # 5、Read cache  从cache中读出最新变量赋给self  方便给forward中使用
    # cache中的键值对最初有: cache[img_file]=[l, shape, segments] cache[hash] cache[results] cache[msg] cache[version]
    # 先从cache中去除cache文件中其他无关键值如:'hash', 'version', 'msgs'等都删除
    [cache.pop(k) for k in ('hash', 'version', 'msgs')]  # remove items
    # pop掉results、hash、version、msgs后只剩下cache[img_file]=[l, shape, segments]
    # cache.values(): 取cache中所有值 对应所有l, shape, segments
    # labels: 如果数据集所有图片中没有一个多边形label  labels存储的label就都是原始label(都是正常的矩形label)
    #         否则将所有图片正常gt的label存入labels 不正常gt(存在一个多边形)经过segments2boxes转换为正常的矩形label
    # shapes: 所有图片的shape
    # self.segments: 如果数据集所有图片中没有一个多边形label  self.segments=None
    #                否则存储数据集中所有存在多边形gt的图片的所有原始label(肯定有多边形label 也可能有矩形正常label 未知数)
    # zip 是因为cache中所有labels、shapes、segments信息都是按每张img分开存储的, zip是将所有图片对应的信息叠在一起
    labels, shapes, self.segments = zip(*cache.values())
    self.labels = list(labels)      # labels 所有图片的所有gt框的信息
    self.shapes = np.array(shapes, dtype=np.float64)
    self.im_files = list(cache.keys())  # update
    self.label_files = img2label_paths(cache.keys())  # update
    n = len(shapes)  # number of images
    bi = np.floor(np.arange(n) / batch_size).astype(np.int)  # batch index
    nb = bi[-1] + 1  # number of batches
    self.batch = bi  # batch index of image
    self.n = n
    self.indices = range(n)

    # Update labels
    include_class = []  # filter labels to include only these classes (optional)
    include_class_array = np.array(include_class).reshape(1, -1)
    for i, (label, segment) in enumerate(zip(self.labels, self.segments)):
        if include_class:
            j = (label[:, 0:1] == include_class_array).any(1)
            self.labels[i] = label[j]
            if segment:
                self.segments[i] = segment[j]
        if single_cls:  # single-class training, merge all classes into 0
            self.labels[i][:, 0] = 0
            if segment:
                self.segments[i][:, 0] = 0

    # Rectangular Training
    # 6、为Rectangular Training作准备
    # 这里主要是注意shapes的生成 这一步很重要 因为如果采样矩形训练那么整个batch的形状要一样 就要计算这个符合整个batch的shape
    # 而且还要对数据集按照高宽比进行排序 这样才能保证同一个batch的图片的形状差不多相同 再选则一个共同的shape代价也比较小
    if self.rect:
        # Sort by aspect ratio
        s = self.shapes  # wh
        ar = s[:, 1] / s[:, 0]  # aspect ratio
        irect = ar.argsort()
        self.im_files = [self.im_files[i] for i in irect]
        self.label_files = [self.label_files[i] for i in irect]
        self.labels = [self.labels[i] for i in irect]
        self.shapes = s[irect]  # wh
        ar = ar[irect]

        # Set training image shapes
        shapes = [[1, 1]] * nb  # 初始化shapes，nb为一轮批次batch的数量
        for i in range(nb):
            ari = ar[bi == i]
            mini, maxi = ari.min(), ari.max()
            if maxi < 1:
                shapes[i] = [maxi, 1]
            elif mini > 1:
                shapes[i] = [1, 1 / mini]

        self.batch_shapes = np.ceil(np.array(shapes) * img_size / stride + pad).astype(np.int) * stride

    # Cache images into RAM/disk for faster training (WARNING: large datasets may exceed system resources)
    self.ims = [None] * n
    self.npy_files = [Path(f).with_suffix('.npy') for f in self.im_files]
    if cache_images:
        gb = 0  # Gigabytes of cached images
        self.im_hw0, self.im_hw = [None] * n, [None] * n
        fcn = self.cache_images_to_disk if cache_images == 'disk' else self.load_image
        results = ThreadPool(NUM_THREADS).imap(fcn, range(n))
        pbar = tqdm(enumerate(results), total=n, bar_format=BAR_FORMAT, disable=LOCAL_RANK > 0)
        for i, x in pbar:
            if cache_images == 'disk':
                gb += self.npy_files[i].stat().st_size
            else:  # 'ram'
                self.ims[i], self.im_hw0[i], self.im_hw[i] = x  # im, hw_orig, hw_resized = load_image(self, i)
                gb += self.ims[i].nbytes
            pbar.desc = f'{prefix}Caching images ({gb / 1E9:.1f}GB {cache_images})'
        pbar.close() 
```

这段代码几个步骤：

1.  赋值一些基础变量，为后面的函数做准备
2.  获取path路径下所有图片的路径`self.img_files`
3.  根据imgs路径找到labels的路径`self.label_files`，这里用到了`img2label_paths`函数
4.  将label存放到了cache中，这样下次运行这个脚本的时候就可以直接从cache中取出label，速度更快，相当于高速缓存
5.  打印cache中的结果，比如找到的标签数量、漏掉的标签数量灯等
6.  从cache中读取最新变量给self，方便给forward中使用，并将cache中其他无关的hash值删除
7.  为Retangular Training做准备：生成self.batch_shapes
8.  是否需要cache image（太大了，一般false）

### 5.2 cache_labels

```py
# 这个函数用于加载文件路径中的label信息生成cache文件。cache文件中包括的信息有：im_file, l, shape, segments, hash, results, msgs, version等
def cache_labels(self, path=Path('./labels.cache'), prefix=''): # 日志头部信息(彩打高亮部分)
    # Cache dataset labels, check images and read shapes
    x = {}  # dict  初始化最终cache中保存的字典dict
    nm, nf, ne, nc, msgs = 0, 0, 0, 0, []  # number missing, found, empty, corrupt, messages
    desc = f"{prefix}Scanning '{path.parent / path.stem}' images and labels..."
    with Pool(NUM_THREADS) as pool:
        # 定义pbar进度条
        # pool.imap_unordered: 对大量数据遍历多进程计算 返回一个迭代器
        # 把self.img_files, self.label_files, repeat(prefix) list中的值作为参数依次送入(一次送一个)verify_image_label函数
        pbar = tqdm(pool.imap(verify_image_label, zip(self.im_files, self.label_files, repeat(prefix))),
                    desc=desc,
                    total=len(self.im_files),
                    bar_format=BAR_FORMAT)
        for im_file, lb, shape, segments, nm_f, nf_f, ne_f, nc_f, msg in pbar:
            nm += nm_f
            nf += nf_f
            ne += ne_f
            nc += nc_f
            if im_file:
                x[im_file] = [lb, shape, segments]
            if msg:
                msgs.append(msg)
            pbar.desc = f"{desc}{nf} found, {nm} missing, {ne} empty, {nc} corrupt"

    pbar.close()    # 关闭进度条
    if msgs:
        LOGGER.info('\n'.join(msgs))
    if nf == 0:
        LOGGER.warning(f'{prefix}WARNING: No labels found in {path}. See {HELP_URL}')
    x['hash'] = get_hash(self.label_files + self.im_files)
    x['results'] = nf, nm, ne, nc, len(self.im_files)
    x['msgs'] = msgs  # warnings
    x['version'] = self.cache_version  # cache version
    try:
        np.save(path, x)  # save cache for next time
        path.with_suffix('.cache.npy').rename(path)  # remove .npy suffix
        LOGGER.info(f'{prefix}New cache created: {path}')
    except Exception as e:
        LOGGER.warning(f'{prefix}WARNING: Cache directory {path.parent} is not writeable: {e}')  # not writeable
    return x 
```

**这个函数用于加载文件路径中的label信息生成cache文件。cache文件中包括的信息有：im_file, l, shape, segments, hash, results, msgs, version等**

### 5.3 len

```py
def __len__(self):
    return len(self.im_files) 
```

**获取数据集图片的数量**

### 5.4 getitem

```py
# 这部分是数据增强函数，一般一次性执行batch_size次。
def __getitem__(self, index):
    """
           这部分是数据增强函数，一般一次性执行batch_size次。
           训练 数据增强: mosaic(random_perspective) + hsv + 上下左右翻转
           测试 数据增强: letterbox
           :return torch.from_numpy(img): 这个index的图片数据(增强后) [3, 640, 640]
           :return labels_out: 这个index图片的gt label [6, 6] = [gt_num, 0+class+xywh(normalized)]
           :return self.img_files[index]: 这个index图片的路径地址
           :return shapes: 这个batch的图片的shapes 测试时(矩形训练)才有  验证时为None   for COCO mAP rescaling
    """
    index = self.indices[index]  # linear, shuffled, or image_weights  如果存在image_weights，则获取新的下标
    hyp = self.hyp
    mosaic = self.mosaic and random.random() < hyp['mosaic']
    # mosaic增强 对图像进行4张图拼接训练  一般训练时运行
    # mosaic + MixUp
    if mosaic:
        # Load mosaic
        img, labels = self.load_mosaic(index)
        shapes = None

        # MixUp augmentation mixup数据增强
        if random.random() < hyp['mixup']:
            img, labels = mixup(img, labels, *self.load_mosaic(random.randint(0, self.n - 1)))

    # 否则：载入图片 + letterbox（val）
    else:
        # Load image
        # 载入图片  载入图片后还会进行一次resize  将当前图片的最长边缩放到指定的大小(512), 较小边同比例缩放
        # load image img=(343, 512, 3)=(h, w, c)  (h0, w0)=(335, 500)  numpy  index=4
        # img: resize后的图片   (h0, w0): 原始图片的hw  (h, w): resize后的图片的hw
        # 这一步是将(335, 500, 3) resize-> (343, 512, 3)
        img, (h0, w0), (h, w) = self.load_image(index)

        # Letterbox
        # letterbox之前确定这张当前图片letterbox之后的shape  如果不用self.rect矩形训练shape就是self.img_size
        # 如果使用self.rect矩形训练shape就是当前batch的shape 因为矩形训练的话我们整个batch的shape必须统一(在__init__函数第6节内容)
        shape = self.batch_shapes[self.batch[index]] if self.rect else self.img_size  # final letterboxed shape
        # letterbox 这一步将第一步缩放得到的图片再缩放到当前batch所需要的尺度 (343, 512, 3) pad-> (384, 512, 3)
        # (矩形推理需要一个batch的所有图片的shape必须相同，而这个shape在init函数中保持在self.batch_shapes中)
        # 这里没有缩放操作，所以这里的ratio永远都是(1.0, 1.0)  pad=(0.0, 20.5)
        img, ratio, pad = letterbox(img, shape, auto=False, scaleup=self.augment)
        shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling

        # 图片进行letterbox后label的坐标也要相应变化，根据pad调整label坐标 并将归一化的xywh -> 未归一化的xyxy
        labels = self.labels[index].copy()
        if labels.size:  # normalized xywh to pixel xyxy format 根据pad调整框的标签坐标，并从归一化xywh->未归一化的xyxy
            labels[:, 1:] = xywhn2xyxy(labels[:, 1:], ratio[0] * w, ratio[1] * h, padw=pad[0], padh=pad[1])

        # 测试代码 测试letterbox效果
        # cv2.imshow("letterbox", img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # print(img.shape)   # (640, 640, 3)

        if self.augment:
            # 不做mosaic的话就要做random_perspective增强 因为mosaic函数内部执行了random_perspective增强
            # random_perspective增强: 随机对图片进行旋转，平移，缩放，裁剪，透视变换
            img, labels = random_perspective(img,
                                             labels,
                                             degrees=hyp['degrees'],
                                             translate=hyp['translate'],
                                             scale=hyp['scale'],
                                             shear=hyp['shear'],
                                             perspective=hyp['perspective'])

    nl = len(labels)  # number of labels
    if nl:
        labels[:, 1:5] = xyxy2xywhn(labels[:, 1:5], w=img.shape[1], h=img.shape[0], clip=True, eps=1E-3)

    if self.augment:
        # Albumentations
        img, labels = self.albumentations(img, labels)
        nl = len(labels)  # update after albumentations

        # HSV color-space 随机改变图片的色调H、饱和度S、亮度V
        augment_hsv(img, hgain=hyp['hsv_h'], sgain=hyp['hsv_s'], vgain=hyp['hsv_v'])

        # Flip up-down
        if random.random() < hyp['flipud']:
            img = np.flipud(img)
            if nl:
                labels[:, 2] = 1 - labels[:, 2]

        # Flip left-right
        if random.random() < hyp['fliplr']:
            img = np.fliplr(img)
            if nl:
                labels[:, 1] = 1 - labels[:, 1]

        # Cutouts
        # labels = cutout(img, labels, p=0.5)
        # nl = len(labels)  # update after cutout

    labels_out = torch.zeros((nl, 6))
    if nl:
        labels_out[:, 1:] = torch.from_numpy(labels)

    # Convert
    # img[:,:,::-1]的作用是实现BGR到RGB通道的转换，对于列表img进行[:,:,::-1]的作用是列表数组左右翻转
    # channel轴换到前面
    # torch.Tensor 高维矩阵的表示： (nSample)*C*H*W
    # num.ndarry 高维矩阵的表示： H*W*C
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img = np.ascontiguousarray(img)

    return torch.from_numpy(img), labels_out, self.im_files[index], shapes 
```

**相当于重写`[]`，跟数据增强相关，一般一次性执行`batch_size`次。**

### 5.5 collate_fn

```py
@staticmethod
def collate_fn(batch):  # 整理函数：如何取样本的，可以定义自己的函数实现想要的功能
    """这个函数会在create_dataloader中生成dataloader时调用：
            整理函数  将image和label整合到一起
            :return torch.stack(img, 0): 如[16, 3, 640, 640] 整个batch的图片
            :return torch.cat(label, 0): 如[15, 6] [num_target, img_index+class_index+xywh(normalized)] 整个batch的label
            :return path: 整个batch所有图片的路径
            :return shapes: (h0, w0), ((h / h0, w / w0), pad)    for COCO mAP rescaling
            pytorch的DataLoader打包一个batch的数据集时要经过此函数进行打包 通过重写此函数实现标签与图片对应的划分，一个batch中哪些标签属于哪一张图片,形如
                [[0, 6, 0.5, 0.5, 0.26, 0.35],
                 [0, 6, 0.5, 0.5, 0.26, 0.35],
                 [1, 6, 0.5, 0.5, 0.26, 0.35],
                 [2, 6, 0.5, 0.5, 0.26, 0.35],]
               前两行标签属于第一张图片, 第三行属于第二张。。。
    """
    # img: 一个tuple 由batch_size个tensor组成 整个batch中每个tensor表示一张图片
    # label: 一个tuple 由batch_size个tensor组成 每个tensor存放一张图片的所有的target信息
    #        label[6, object_num] 6中的第一个数代表一个batch中的第几张图
    # path: 一个tuple 由4个str组成, 每个str对应一张图片的地址信息
    im, label, path, shapes = zip(*batch)  # transposed
    for i, lb in enumerate(label):
        lb[:, 0] = i  # add target image index for build_targets()
    # 返回的img=[batch_size, 3, 736, 736]
    #      torch.stack(img, 0): 将batch_size个[3, 736, 736]的矩阵拼成一个[batch_size, 3, 736, 736]
    # label=[target_sums, 6]  6：表示当前target属于哪一张图+class+x+y+w+h
    #      torch.cat(label, 0): 将[n1,6]、[n2,6]、[n3,6]...拼接成[n1+n2+n3+..., 6]
    # 这里之所以拼接的方式不同是因为img拼接的时候它的每个部分的形状是相同的，都是[3, 736, 736]
    # 而我label的每个部分的形状是不一定相同的，每张图的目标个数是不一定相同的（label肯定也希望用stack,更方便,但是不能那样拼）
    # 如果每张图的目标个数是相同的，那我们就可能不需要重写collate_fn函数了
    return torch.stack(im, 0), torch.cat(label, 0), path, shapes 
```

**很多人以为写完 init 和 getitem 函数数据增强就做完了，我们在分类任务中的确写完这两个函数就可以了，因为系统中是给我们写好了一个collate_fn函数的，但是在目标检测中我们却需要重写collate_fn函数**

> 注意：这个函数一般是当调用了batch_size次 getitem 函数后才会调用一次这个函数，对batch_size张图片和对应的label进行打包。
> **References**
> CSDN qq_27172615： [精读yolo LoadImagesAndLabels类](https://blog.csdn.net/qq_27172615/article/details/128729070)

## 6\. img2label_paths

```py
def img2label_paths(img_paths):
    '''
        用在LoadImagesAndLabels的init函数中
        根据imgs图片的路径找到对应labels的路径
    Define label paths as a function of image paths
    :params img_paths: {list: 50}  整个数据集的图片相对路径  例如: '..\\datasets\\VOC\\images\\train2007\\000012.jpg'
                                                        =>   '..\\datasets\\VOC\\labels\\train2007\\000012.txt'

    '''
    # Define label paths as a function of image paths
    # os.sep 可以根据所处的平台不同，自适应采用分隔符
    # sa: '\\images\\'    sb: '\\labels\\'
    sa, sb = f'{os.sep}images{os.sep}', f'{os.sep}labels{os.sep}'  # /images/, /labels/ substrings
    # 把img_paths中所有图片路径中的images替换为labels
    return [sb.join(x.rsplit(sa, 1)).rsplit('.', 1)[0] + '.txt' for x in img_paths] 
```

根据源码可知，我们在制作数据集时，图片的文件夹必须设置为`images`，标签的名字必须设为`labels`。

并且放在相同的路径下。在文件夹里面，再分`train`、`val`、`test`等文件夹

## 7\. verify_image_label

```py
# 这个函数用于检查每一张图片和每一张label文件是否完好。
def verify_image_label(args):
    # Verify one image-label pair
    im_file, lb_file, prefix = args
    # segments: 存放这张图所有gt框的信息(包含segments多边形: label某一列数大于8)
    nm, nf, ne, nc, msg, segments = 0, 0, 0, 0, '', []  # number (missing, found, empty, corrupt), message, segments
    try:
        # verify images
        im = Image.open(im_file)    # 打开图片
        im.verify()  # PIL verify   检查图片内容和格式是否正常
        shape = exif_size(im)  # image size
        assert (shape[0] > 9) & (shape[1] > 9), f'image size {shape} <10 pixels'
        assert im.format.lower() in IMG_FORMATS, f'invalid image format {im.format}'
        if im.format.lower() in ('jpg', 'jpeg'):
            with open(im_file, 'rb') as f:
                f.seek(-2, 2)
                if f.read() != b'\xff\xd9':  # corrupt JPEG
                    ImageOps.exif_transpose(Image.open(im_file)).save(im_file, 'JPEG', subsampling=0, quality=100)
                    msg = f'{prefix}WARNING: {im_file}: corrupt JPEG restored and saved'

        # verify labels
        if os.path.isfile(lb_file):
            nf = 1  # label found
            with open(lb_file) as f:
                lb = [x.split() for x in f.read().strip().splitlines() if len(x)]
                if any(len(x) > 6 for x in lb):  # is segment
                    classes = np.array([x[0] for x in lb], dtype=np.float32)
                    # segments(多边形) -> bbox(正方形), 得到新标签  [gt_num, cls+xywh(normalized)]
                    segments = [np.array(x[1:], dtype=np.float32).reshape(-1, 2) for x in lb]  # (cls, xy1...)
                    lb = np.concatenate((classes.reshape(-1, 1), segments2boxes(segments)), 1)  # (cls, xywh)
                lb = np.array(lb, dtype=np.float32)
            nl = len(lb)
            if nl:
                # 判断标签是否有5列
                assert lb.shape[1] == 5, f'labels require 5 columns, {lb.shape[1]} columns detected'
                # 是否全部大于0
                assert (lb >= 0).all(), f'negative label values {lb[lb < 0]}'
                # 判断标签坐标x y w h是否归一化
                assert (lb[:, 1:] <= 1).all(), f'non-normalized or out of bounds coordinates {lb[:, 1:][lb[:, 1:] > 1]}'
                # 判断标签中是否有重复的坐标
                _, i = np.unique(lb, axis=0, return_index=True)
                if len(i) < nl:  # duplicate row check
                    lb = lb[i]  # remove duplicates
                    if segments:
                        segments = segments[i]
                    msg = f'{prefix}WARNING: {im_file}: {nl - len(i)} duplicate labels removed'
            else:
                ne = 1  # label empty
                lb = np.zeros((0, 5), dtype=np.float32)
        else:
            nm = 1  # label missing
            lb = np.zeros((0, 5), dtype=np.float32)
        return im_file, lb, shape, segments, nm, nf, ne, nc, msg
    except Exception as e:
        nc = 1
        msg = f'{prefix}WARNING: {im_file}: ignoring corrupt image/label: {e}'
        return [None, None, None, None, nm, nf, ne, nc, msg] 
```

这一部分是检查每一张图片和每一张label是否完好。

*   **图片：**主要是看格式是否损坏
*   **labels：**看标签是否有5列、归一化、重复等

## 8\. load_image

```py
 def load_image(self, i):
        """用在LoadImagesAndLabels模块的__getitem__函数和load_mosaic模块中
            从self或者从对应图片路径中载入对应index的图片 并将原图中hw中较大者扩展到self.img_size, 较小者同比例扩展
            loads 1 image from dataset, returns img, original hw, resized hw
            :params self: 一般是导入LoadImagesAndLabels中的self
            :param index: 当前图片的index
            :return: img: resize后的图片
                    (h0, w0): hw_original  原图的hw
                    img.shape[:2]: hw_resized resize后的图片hw(hw中较大者扩展到self.img_size, 较小者同比例扩展)
        """
        # Loads 1 image from dataset index 'i', returns (im, original hw, resized hw)
        im, f, fn = self.ims[i], self.im_files[i], self.npy_files[i],
        # 图片是空的，则从对应路径读取
        if im is None:  # not cached in RAM
            if fn.exists():  # load npy
                im = np.load(fn)
            else:  # read image
                im = cv2.imread(f)  # BGR
                assert im is not None, f'Image Not Found {f}'
            h0, w0 = im.shape[:2]  # orig hw
            r = self.img_size / max(h0, w0)  # ratio
            if r != 1:  # if sizes are not equal
                # 不同方式的缩放
                interp = cv2.INTER_LINEAR if (self.augment or r > 1) else cv2.INTER_AREA
                im = cv2.resize(im, (int(w0 * r), int(h0 * r)), interpolation=interp)
            return im, (h0, w0), im.shape[:2]  # im, hw_original, hw_resized
        return self.ims[i], self.im_hw0[i], self.im_hw[i]  # im, hw_original, hw_resized 
```

**这一部分是加載图片并根据设定的输入大小与图片原大小的比例ratio进行resize**

**用在LoadImagesAndLabels模块的__getitem__函数和load_mosaic模块中**

## 9\. load_mosaic

这个板块就是大名鼎鼎的mosaic，训练的时候都会用到它，可以大幅度提升小目标的mAP。非常重要，需要熟练掌握。

```py
# 生成一个mosaic增强的图片
def load_mosaic(self, index):
    """用在LoadImagesAndLabels模块的__getitem__函数 进行mosaic数据增强
        将四张图片拼接在一张马赛克图像中  loads images in a 4-mosaic
        :param index: 需要获取的图像索引
        :return: img4: mosaic和随机透视变换后的一张图片  numpy(640, 640, 3)
                 labels4: img4对应的target  [M, cls+x1y1x2y2]
    """
    # YOLOv5 4-mosaic loader. Loads 1 image + 3 random images into a 4-image mosaic
    # labels4: 用于存放拼接图像（4张图拼成一张）的label信息(不包含segments多边形)
    # segments4: 用于存放拼接图像（4张图拼成一张）的label信息(包含segments多边形)
    labels4, segments4 = [], []
    s = self.img_size
    # 随机初始化拼接图像的中心点坐标  [0, s*2]之间随机取2个数作为拼接图像的中心坐标
    yc, xc = (int(random.uniform(-x, 2 * s + x)) for x in self.mosaic_border)  # mosaic center x, y 取中心点
    # 从dataset中随机寻找额外的三张图像进行拼接 [14, 26, 2, 16] 再随机选三张图片的index
    indices = [index] + random.choices(self.indices, k=3)  # 3 additional image indices
    random.shuffle(indices) # 将列表中元素打乱
    for i, index in enumerate(indices):
        # Load image
        # 每次拿一张图片 并将这张图片resize到self.size(h,w)
        # 加载图片并根据设定的输入大小与图片原大小的比例ratio进行resize
        img, _, (h, w) = self.load_image(index)

        # place img in img4
        if i == 0:  # top left
            # 初始化大图
            img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
            # 设置大图上的位置（左上角）
            x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
            # 选取小图上的位置
            x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
        elif i == 1:  # top right
            # 设置大图上的位置（右上角）
            x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
        elif i == 2:  # bottom left 左下角
            x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
        elif i == 3:  # bottom right 右下角
            x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

        # 将小图上截取的部分贴到大图上
        img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
        # 计算小图到大图时产生的偏移，用来计算mosaic增强后的标签框的位置
        padw = x1a - x1b
        padh = y1a - y1b

        # Labels
        labels, segments = self.labels[index].copy(), self.segments[index].copy()
        # 获取标签
        if labels.size:
            # 将xywh（百分比那些值）标准化为像素xy格式
            labels[:, 1:] = xywhn2xyxy(labels[:, 1:], w, h, padw, padh)  # normalized xywh to pixel xyxy format
            #转为像素段
            segments = [xyn2xy(x, w, h, padw, padh) for x in segments]
        labels4.append(labels)
        # 填进列表
        segments4.extend(segments)

    # Concat/clip labels
    # 调整标签框在图片内部
    labels4 = np.concatenate(labels4, 0)    # 对array进行拼接的函数，以第一维度进行拼接
    for x in (labels4[:, 1:], *segments4):
        np.clip(x, 0, 2 * s, out=x)  # clip when using random_perspective()
    # img4, labels4 = replicate(img4, labels4)  # replicate

    # Augment
    # 进行mosaic的时候将四张图片整合到一起之后shape为[2*img_size,2*img_size]
    # 对mosaic整合的图片进行随机旋转、平移、缩放、裁剪，并resize为输入大小img_size
    img4, labels4, segments4 = copy_paste(img4, labels4, segments4, p=self.hyp['copy_paste'])
    img4, labels4 = random_perspective(img4,
                                       labels4,
                                       segments4,
                                       degrees=self.hyp['degrees'],
                                       translate=self.hyp['translate'],
                                       scale=self.hyp['scale'],
                                       shear=self.hyp['shear'],
                                       perspective=self.hyp['perspective'],
                                       border=self.mosaic_border)  # border to remove

    return img4, labels4 
```

同理，还有个load_mosaic9函数，做法相同，用的好像并不是很多，效果没mosaic4好。

大致步骤：
![在这里插入图片描述](img/ac51e4a38c97b3cd118bdb70033259c1.png)
字有点丑哈哈哈，需要注意的是，labels4也是需要进行相对位置变换的。

## 总结

dataloader部分差不多就是这些了，这部分的内容主要是创建数据集，方便后面训练的时候调用，并且在进行损失函数计算的时候，也需要传入labels。不算很难，最难的应该算是接下来的`loss.py`了。