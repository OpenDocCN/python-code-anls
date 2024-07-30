# `.\yolov8\ultralytics\data\base.py`

```
# Ultralytics YOLO 🚀, AGPL-3.0 license

import glob  # 导入用于获取文件路径的模块
import math  # 导入数学函数模块
import os  # 导入操作系统功能模块
import random  # 导入生成随机数的模块
from copy import deepcopy  # 导入深拷贝函数
from multiprocessing.pool import ThreadPool  # 导入多线程池模块
from pathlib import Path  # 导入处理路径的模块
from typing import Optional  # 导入类型提示模块

import cv2  # 导入OpenCV图像处理库
import numpy as np  # 导入NumPy数值计算库
import psutil  # 导入进程和系统信息获取模块
from torch.utils.data import Dataset  # 导入PyTorch数据集基类

from ultralytics.data.utils import FORMATS_HELP_MSG, HELP_URL, IMG_FORMATS  # 导入自定义数据处理工具
from ultralytics.utils import DEFAULT_CFG, LOCAL_RANK, LOGGER, NUM_THREADS, TQDM  # 导入自定义工具函数


class BaseDataset(Dataset):
    """
    Base dataset class for loading and processing image data.

    Args:
        img_path (str): Path to the folder containing images.
        imgsz (int, optional): Image size. Defaults to 640.
        cache (bool, optional): Cache images to RAM or disk during training. Defaults to False.
        augment (bool, optional): If True, data augmentation is applied. Defaults to True.
        hyp (dict, optional): Hyperparameters to apply data augmentation. Defaults to None.
        prefix (str, optional): Prefix to print in log messages. Defaults to ''.
        rect (bool, optional): If True, rectangular training is used. Defaults to False.
        batch_size (int, optional): Size of batches. Defaults to None.
        stride (int, optional): Stride. Defaults to 32.
        pad (float, optional): Padding. Defaults to 0.0.
        single_cls (bool, optional): If True, single class training is used. Defaults to False.
        classes (list): List of included classes. Default is None.
        fraction (float): Fraction of dataset to utilize. Default is 1.0 (use all data).

    Attributes:
        im_files (list): List of image file paths.
        labels (list): List of label data dictionaries.
        ni (int): Number of images in the dataset.
        ims (list): List of loaded images.
        npy_files (list): List of numpy file paths.
        transforms (callable): Image transformation function.
    """

    def __init__(
        self,
        img_path,
        imgsz=640,
        cache=False,
        augment=True,
        hyp=DEFAULT_CFG,
        prefix="",
        rect=False,
        batch_size=16,
        stride=32,
        pad=0.5,
        single_cls=False,
        classes=None,
        fraction=1.0,
        ):
        # 初始化数据集对象，设置各种参数和属性
        """Initialize BaseDataset with given configuration and options."""
        # 调用父类初始化方法
        super().__init__()
        # 设置图片路径
        self.img_path = img_path
        # 图像大小
        self.imgsz = imgsz
        # 是否进行数据增强
        self.augment = augment
        # 是否单类别
        self.single_cls = single_cls
        # 数据集前缀
        self.prefix = prefix
        # 数据集采样比例
        self.fraction = fraction
        # 获取所有图像文件路径
        self.im_files = self.get_img_files(self.img_path)
        # 获取标签
        self.labels = self.get_labels()
        # 更新标签，根据是否单类别和指定的类别
        self.update_labels(include_class=classes)  # single_cls and include_class
        # 图像数量
        self.ni = len(self.labels)  # number of images
        # 是否使用矩形边界框
        self.rect = rect
        # 批处理大小
        self.batch_size = batch_size
        # 步长
        self.stride = stride
        # 填充
        self.pad = pad
        # 如果使用矩形边界框，确保指定了批处理大小
        if self.rect:
            assert self.batch_size is not None
            # 设置矩形边界框参数
            self.set_rectangle()

        # 用于马赛克图像的缓冲线程
        self.buffer = []  # buffer size = batch size
        # 最大缓冲长度，最小为图像数量、批处理大小的8倍、1000中的最小值（如果进行数据增强）
        self.max_buffer_length = min((self.ni, self.batch_size * 8, 1000)) if self.augment else 0

        # 缓存图像（缓存选项包括 True, False, None, "ram", "disk"）
        self.ims, self.im_hw0, self.im_hw = [None] * self.ni, [None] * self.ni, [None] * self.ni
        # 生成每个图像文件对应的 .npy 文件路径
        self.npy_files = [Path(f).with_suffix(".npy") for f in self.im_files]
        # 设置缓存选项
        self.cache = cache.lower() if isinstance(cache, str) else "ram" if cache is True else None
        # 如果缓存选项是 "ram" 并且内存中已存在缓存，或者缓存选项是 "disk"，则进行图像缓存
        if (self.cache == "ram" and self.check_cache_ram()) or self.cache == "disk":
            self.cache_images()

        # 构建图像转换操作
        self.transforms = self.build_transforms(hyp=hyp)
    def get_img_files(self, img_path):
        """Read image files."""
        try:
            f = []  # image files列表，用于存储图像文件路径
            for p in img_path if isinstance(img_path, list) else [img_path]:
                p = Path(p)  # 将路径转换为Path对象，以保证在不同操作系统上的兼容性
                if p.is_dir():  # 如果是目录
                    f += glob.glob(str(p / "**" / "*.*"), recursive=True)
                    # 获取目录下所有文件的路径，并加入到f列表中
                    # 使用glob模块，支持递归查找
                    # 使用pathlib的方式：F = list(p.rglob('*.*'))  
                elif p.is_file():  # 如果是文件
                    with open(p) as t:
                        t = t.read().strip().splitlines()  # 读取文件内容，并按行分割
                        parent = str(p.parent) + os.sep
                        # 获取文件的父目录，并在每个文件路径前添加父目录路径，处理本地到全局路径的转换
                        f += [x.replace("./", parent) if x.startswith("./") else x for x in t]
                        # 将文件路径添加到f列表中，处理相对路径
                        # 使用pathlib的方式：F += [p.parent / x.lstrip(os.sep) for x in t]
                else:
                    raise FileNotFoundError(f"{self.prefix}{p} does not exist")
                    # 如果既不是文件也不是目录，则抛出文件不存在的异常
            im_files = sorted(x.replace("/", os.sep) for x in f if x.split(".")[-1].lower() in IMG_FORMATS)
            # 对f列表中的文件路径进行筛选，保留符合图像格式的文件路径，并排序
            # 使用pathlib的方式：self.img_files = sorted([x for x in f if x.suffix[1:].lower() in IMG_FORMATS])
            assert im_files, f"{self.prefix}No images found in {img_path}. {FORMATS_HELP_MSG}"
            # 如果im_files为空，则抛出断言错误，表示未找到任何图像文件
        except Exception as e:
            raise FileNotFoundError(f"{self.prefix}Error loading data from {img_path}\n{HELP_URL}") from e
            # 捕获所有异常，并抛出带有详细信息的文件加载错误异常
        if self.fraction < 1:
            im_files = im_files[: round(len(im_files) * self.fraction)]  # 保留数据集的一部分比例
            # 如果fraction小于1，则根据fraction保留im_files中的部分文件路径
        return im_files
        # 返回处理后的图像文件路径列表

    def update_labels(self, include_class: Optional[list]):
        """Update labels to include only these classes (optional)."""
        include_class_array = np.array(include_class).reshape(1, -1)
        # 将include_class转换为NumPy数组，并进行形状重塑
        for i in range(len(self.labels)):
            if include_class is not None:  # 如果include_class不为空
                cls = self.labels[i]["cls"]
                bboxes = self.labels[i]["bboxes"]
                segments = self.labels[i]["segments"]
                keypoints = self.labels[i]["keypoints"]
                j = (cls == include_class_array).any(1)
                # 找到标签中与include_class相匹配的类别索引
                self.labels[i]["cls"] = cls[j]  # 更新类别
                self.labels[i]["bboxes"] = bboxes[j]  # 更新边界框
                if segments:  # 如果存在分割信息
                    self.labels[i]["segments"] = [segments[si] for si, idx in enumerate(j) if idx]
                    # 更新分割信息，只保留与include_class匹配的分割
                if keypoints is not None:  # 如果存在关键点信息
                    self.labels[i]["keypoints"] = keypoints[j]  # 更新关键点信息
            if self.single_cls:  # 如果标签是单类别的
                self.labels[i]["cls"][:, 0] = 0  # 将所有类别标记为0
    def load_image(self, i, rect_mode=True):
        """Loads 1 image from dataset index 'i', returns (im, resized hw)."""
        # 从数据集索引 'i' 加载一张图片，并返回原图和调整大小后的尺寸
        im, f, fn = self.ims[i], self.im_files[i], self.npy_files[i]
        
        if im is None:  # not cached in RAM
            # 如果图像未被缓存在内存中
            if fn.exists():  # load npy
                # 如果存在对应的 *.npy 文件，则加载该文件
                try:
                    im = np.load(fn)
                except Exception as e:
                    # 捕获异常，警告并删除损坏的 *.npy 图像文件
                    LOGGER.warning(f"{self.prefix}WARNING ⚠️ Removing corrupt *.npy image file {fn} due to: {e}")
                    Path(fn).unlink(missing_ok=True)
                    # 从原始图像文件加载图像（BGR格式）
                    im = cv2.imread(f)  # BGR
            else:  # read image
                # 否则，直接从原始图像文件中读取图像（BGR格式）
                im = cv2.imread(f)  # BGR
            
            # 如果未能成功加载图像，则抛出文件未找到异常
            if im is None:
                raise FileNotFoundError(f"Image Not Found {f}")

            h0, w0 = im.shape[:2]  # orig hw
            if rect_mode:  # resize long side to imgsz while maintaining aspect ratio
                # 如果矩形模式为真，则将长边调整到指定的imgsz大小，并保持纵横比
                r = self.imgsz / max(h0, w0)  # ratio
                if r != 1:  # if sizes are not equal
                    # 计算调整后的宽高，并进行插值缩放
                    w, h = (min(math.ceil(w0 * r), self.imgsz), min(math.ceil(h0 * r), self.imgsz))
                    im = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)
            elif not (h0 == w0 == self.imgsz):  # resize by stretching image to square imgsz
                # 否则，将图像拉伸调整到正方形大小imgsz
                im = cv2.resize(im, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)

            # 如果进行数据增强训练，则将处理后的图像数据和原始、调整后的尺寸保存到缓冲区
            if self.augment:
                self.ims[i], self.im_hw0[i], self.im_hw[i] = im, (h0, w0), im.shape[:2]  # im, hw_original, hw_resized
                self.buffer.append(i)
                if 1 < len(self.buffer) >= self.max_buffer_length:  # prevent empty buffer
                    # 如果缓冲区长度超过最大长度限制，则弹出最旧的元素
                    j = self.buffer.pop(0)
                    if self.cache != "ram":
                        # 如果不是RAM缓存，则清空该位置的图像和尺寸数据
                        self.ims[j], self.im_hw0[j], self.im_hw[j] = None, None, None

            # 返回加载的图像、原始尺寸和调整后的尺寸
            return im, (h0, w0), im.shape[:2]

        # 如果图像已缓存在内存中，则直接返回已缓存的图像及其原始和调整后的尺寸
        return self.ims[i], self.im_hw0[i], self.im_hw[i]

    def cache_images(self):
        """Cache images to memory or disk."""
        b, gb = 0, 1 << 30  # bytes of cached images, bytes per gigabytes
        # 根据缓存选项选择不同的缓存函数和存储介质
        fcn, storage = (self.cache_images_to_disk, "Disk") if self.cache == "disk" else (self.load_image, "RAM")
        
        # 使用线程池处理图像缓存操作
        with ThreadPool(NUM_THREADS) as pool:
            # 并行加载图像或执行缓存操作
            results = pool.imap(fcn, range(self.ni))
            # 使用进度条显示缓存进度
            pbar = TQDM(enumerate(results), total=self.ni, disable=LOCAL_RANK > 0)
            for i, x in pbar:
                if self.cache == "disk":
                    # 如果缓存到磁盘，则累加缓存的图像文件大小
                    b += self.npy_files[i].stat().st_size
                else:  # 'ram'
                    # 如果缓存到RAM，则直接将加载的图像和其尺寸保存到相应的位置
                    self.ims[i], self.im_hw0[i], self.im_hw[i] = x
                    b += self.ims[i].nbytes
                # 更新进度条描述信息，显示当前缓存的总量及存储介质
                pbar.desc = f"{self.prefix}Caching images ({b / gb:.1f}GB {storage})"
            pbar.close()
    def cache_images_to_disk(self, i):
        """Saves an image as an *.npy file for faster loading."""
        f = self.npy_files[i]  # 获取第 i 个 *.npy 文件的路径
        if not f.exists():  # 如果该文件不存在
            np.save(f.as_posix(), cv2.imread(self.im_files[i]), allow_pickle=False)  # 将对应图像保存为 *.npy 文件

    def check_cache_ram(self, safety_margin=0.5):
        """Check image caching requirements vs available memory."""
        b, gb = 0, 1 << 30  # 初始化缓存图像占用的字节数和每个 GB 的字节数
        n = min(self.ni, 30)  # 选取 self.ni 和 30 中较小的一个作为采样图片数目
        for _ in range(n):
            im = cv2.imread(random.choice(self.im_files))  # 随机选取一张图片进行读取
            ratio = self.imgsz / max(im.shape[0], im.shape[1])  # 计算图片尺寸与最大宽高之比
            b += im.nbytes * ratio**2  # 计算每张图片占用的内存字节数，并根据比率进行加权求和
        mem_required = b * self.ni / n * (1 + safety_margin)  # 计算需要缓存整个数据集所需的内存大小（GB）
        mem = psutil.virtual_memory()  # 获取系统内存信息
        success = mem_required < mem.available  # 判断是否有足够的内存来缓存数据集
        if not success:  # 如果内存不足
            self.cache = None  # 清空缓存
            LOGGER.info(
                f"{self.prefix}{mem_required / gb:.1f}GB RAM required to cache images "
                f"with {int(safety_margin * 100)}% safety margin but only "
                f"{mem.available / gb:.1f}/{mem.total / gb:.1f}GB available, not caching images ⚠️"
            )  # 记录日志，显示缓存失败的原因和相关内存信息
        return success  # 返回是否成功缓存的布尔值

    def set_rectangle(self):
        """Sets the shape of bounding boxes for YOLO detections as rectangles."""
        bi = np.floor(np.arange(self.ni) / self.batch_size).astype(int)  # 计算每张图片所属的批次索引
        nb = bi[-1] + 1  # 计算总批次数

        s = np.array([x.pop("shape") for x in self.labels])  # 提取标签中的形状信息（宽高）
        ar = s[:, 0] / s[:, 1]  # 计算宽高比
        irect = ar.argsort()  # 对宽高比进行排序的索引
        self.im_files = [self.im_files[i] for i in irect]  # 根据排序后的索引重新排列图像文件路径
        self.labels = [self.labels[i] for i in irect]  # 根据排序后的索引重新排列标签
        ar = ar[irect]  # 根据排序后的索引重新排列宽高比

        # 设置训练图像的形状
        shapes = [[1, 1]] * nb
        for i in range(nb):
            ari = ar[bi == i]  # 找出属于当前批次的所有图片的宽高比
            mini, maxi = ari.min(), ari.max()  # 计算当前批次内宽高比的最小值和最大值
            if maxi < 1:
                shapes[i] = [maxi, 1]  # 如果最大宽高比小于1，则设为最大宽度，高度为1
            elif mini > 1:
                shapes[i] = [1, 1 / mini]  # 如果最小宽高比大于1，则设为宽度1，高度为最小高度的倒数

        self.batch_shapes = np.ceil(np.array(shapes) * self.imgsz / self.stride + self.pad).astype(int) * self.stride  # 计算批次形状，保证整数倍的步长
        self.batch = bi  # 记录每张图像所属的批次索引

    def __getitem__(self, index):
        """Returns transformed label information for given index."""
        return self.transforms(self.get_image_and_label(index))  # 返回给定索引的图像和标签的转换信息
    def get_image_and_label(self, index):
        """Get and return label information from the dataset."""
        label = deepcopy(self.labels[index])  # 创建标签的深层副本，确保不影响原始数据 https://github.com/ultralytics/ultralytics/pull/1948
        label.pop("shape", None)  # 如果存在形状信息，从标签中移除，通常适用于矩形标注数据
        # 载入图像并将相关信息存入标签字典
        label["img"], label["ori_shape"], label["resized_shape"] = self.load_image(index)
        # 计算图像缩放比例，用于评估
        label["ratio_pad"] = (
            label["resized_shape"][0] / label["ori_shape"][0],
            label["resized_shape"][1] / label["ori_shape"][1],
        )
        if self.rect:
            # 如果使用矩形模式，添加批次对应的形状信息到标签中
            label["rect_shape"] = self.batch_shapes[self.batch[index]]
        # 更新标签信息并返回
        return self.update_labels_info(label)

    def __len__(self):
        """Returns the length of the labels list for the dataset."""
        # 返回数据集标签列表的长度
        return len(self.labels)

    def update_labels_info(self, label):
        """Custom your label format here."""
        # 自定义标签格式的方法，直接返回输入的标签
        return label

    def build_transforms(self, hyp=None):
        """
        Users can customize augmentations here.

        Example:
            ```py
            if self.augment:
                # Training transforms
                return Compose([])
            else:
                # Val transforms
                return Compose([])
            ```
        """
        # 用户可以在此处自定义数据增强操作，此处抛出未实现错误，鼓励用户进行定制
        raise NotImplementedError

    def get_labels(self):
        """
        Users can customize their own format here.

        Note:
            Ensure output is a dictionary with the following keys:
            ```py
            dict(
                im_file=im_file,
                shape=shape,  # format: (height, width)
                cls=cls,
                bboxes=bboxes, # xywh
                segments=segments,  # xy
                keypoints=keypoints, # xy
                normalized=True, # or False
                bbox_format="xyxy",  # or xywh, ltwh
            )
            ```
        """
        # 用户可以在此处自定义标签输出格式，此处抛出未实现错误，鼓励用户进行定制
        raise NotImplementedError
```