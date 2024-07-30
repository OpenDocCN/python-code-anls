# `.\yolov8\ultralytics\data\build.py`

```
# Ultralytics YOLO 🚀, AGPL-3.0 license

import os
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import dataloader, distributed

# 导入自定义数据集类
from ultralytics.data.dataset import GroundingDataset, YOLODataset, YOLOMultiModalDataset
# 导入数据加载器
from ultralytics.data.loaders import (
    LOADERS,
    LoadImagesAndVideos,
    LoadPilAndNumpy,
    LoadScreenshots,
    LoadStreams,
    LoadTensor,
    SourceTypes,
    autocast_list,
)
# 导入数据相关的工具函数和常量
from ultralytics.data.utils import IMG_FORMATS, PIN_MEMORY, VID_FORMATS
# 导入辅助工具
from ultralytics.utils import RANK, colorstr
# 导入检查函数
from ultralytics.utils.checks import check_file


class InfiniteDataLoader(dataloader.DataLoader):
    """
    Dataloader that reuses workers.

    Uses same syntax as vanilla DataLoader.
    """

    def __init__(self, *args, **kwargs):
        """Dataloader that infinitely recycles workers, inherits from DataLoader."""
        super().__init__(*args, **kwargs)
        # 使用 _RepeatSampler 来无限循环利用数据加载器的工作线程
        object.__setattr__(self, "batch_sampler", _RepeatSampler(self.batch_sampler))
        # 创建迭代器
        self.iterator = super().__iter__()

    def __len__(self):
        """Returns the length of the batch sampler's sampler."""
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        """Creates a sampler that repeats indefinitely."""
        for _ in range(len(self)):
            yield next(self.iterator)

    def reset(self):
        """
        Reset iterator.

        This is useful when we want to modify settings of dataset while training.
        """
        # 重置迭代器，允许在训练过程中修改数据集设置
        self.iterator = self._get_iterator()


class _RepeatSampler:
    """
    Sampler that repeats forever.

    Args:
        sampler (Dataset.sampler): The sampler to repeat.
    """

    def __init__(self, sampler):
        """Initializes an object that repeats a given sampler indefinitely."""
        self.sampler = sampler

    def __iter__(self):
        """Iterates over the 'sampler' and yields its contents."""
        while True:
            yield from iter(self.sampler)


def seed_worker(worker_id):  # noqa
    """Set dataloader worker seed https://pytorch.org/docs/stable/notes/randomness.html#dataloader."""
    # 设置数据加载器的工作线程种子
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_yolo_dataset(cfg, img_path, batch, data, mode="train", rect=False, stride=32, multi_modal=False):
    """Build YOLO Dataset."""
    # 根据 multi_modal 参数选择 YOLO 单模态或多模态数据集
    dataset = YOLOMultiModalDataset if multi_modal else YOLODataset
    # 返回一个数据集对象，用于训练或推断
    return dataset(
        img_path=img_path,           # 图像路径
        imgsz=cfg.imgsz,             # 图像尺寸
        batch_size=batch,            # 批处理大小
        augment=mode == "train",     # 是否进行数据增强（训练模式下）
        hyp=cfg,                     # 训练超参数配置
        rect=cfg.rect or rect,       # 是否使用矩形批处理（从配置文件或参数中获取）
        cache=cfg.cache or None,     # 是否缓存数据（从配置文件或参数中获取）
        single_cls=cfg.single_cls or False,  # 是否单类别训练（从配置文件或参数中获取，默认为False）
        stride=int(stride),          # 步幅大小（转换为整数）
        pad=0.0 if mode == "train" else 0.5,  # 填充值（训练模式下为0.0，推断模式下为0.5）
        prefix=colorstr(f"{mode}: "),  # 日志前缀，包含模式信息
        task=cfg.task,               # 任务类型（从配置文件中获取）
        classes=cfg.classes,         # 类别列表（从配置文件中获取）
        data=data,                   # 数据集对象
        fraction=cfg.fraction if mode == "train" else 1.0,  # 数据集分数（训练模式下从配置文件获取，推断模式下为1.0）
    )
# 构建用于 YOLO 数据集的数据加载器
def build_grounding(cfg, img_path, json_file, batch, mode="train", rect=False, stride=32):
    """Build YOLO Dataset."""
    # 返回一个 GroundingDataset 对象，用于训练或验证
    return GroundingDataset(
        img_path=img_path,  # 图像文件路径
        json_file=json_file,  # 包含标注信息的 JSON 文件路径
        imgsz=cfg.imgsz,  # 图像尺寸
        batch_size=batch,  # 批处理大小
        augment=mode == "train",  # 是否进行数据增强
        hyp=cfg,  # 配置信息对象，可能需要通过 get_hyps_from_cfg 函数获取
        rect=cfg.rect or rect,  # 是否使用矩形批处理
        cache=cfg.cache or None,  # 是否使用缓存
        single_cls=cfg.single_cls or False,  # 是否为单类别检测
        stride=int(stride),  # 步长
        pad=0.0 if mode == "train" else 0.5,  # 边缘填充
        prefix=colorstr(f"{mode}: "),  # 输出前缀
        task=cfg.task,  # YOLO 的任务类型
        classes=cfg.classes,  # 类别信息
        fraction=cfg.fraction if mode == "train" else 1.0,  # 数据集的使用比例
    )


# 构建用于训练或验证集的 DataLoader
def build_dataloader(dataset, batch, workers, shuffle=True, rank=-1):
    """Return an InfiniteDataLoader or DataLoader for training or validation set."""
    # 限制批处理大小不超过数据集的大小
    batch = min(batch, len(dataset))
    nd = torch.cuda.device_count()  # CUDA 设备数量
    nw = min(os.cpu_count() // max(nd, 1), workers)  # 确定使用的工作线程数量
    sampler = None if rank == -1 else distributed.DistributedSampler(dataset, shuffle=shuffle)
    generator = torch.Generator()
    generator.manual_seed(6148914691236517205 + RANK)  # 设置随机数生成器种子
    # 返回一个 InfiniteDataLoader 或 DataLoader 对象
    return InfiniteDataLoader(
        dataset=dataset,  # 数据集对象
        batch_size=batch,  # 批处理大小
        shuffle=shuffle and sampler is None,  # 是否打乱数据顺序
        num_workers=nw,  # 工作线程数量
        sampler=sampler,  # 分布式采样器
        pin_memory=PIN_MEMORY,  # 是否将数据保存在固定内存中
        collate_fn=getattr(dataset, "collate_fn", None),  # 数据集的整理函数
        worker_init_fn=seed_worker,  # 工作线程初始化函数
        generator=generator,  # 随机数生成器
    )


# 检查输入数据源的类型，并返回相应的标志值
def check_source(source):
    """Check source type and return corresponding flag values."""
    webcam, screenshot, from_img, in_memory, tensor = False, False, False, False, False
    if isinstance(source, (str, int, Path)):  # 检查是否为字符串、整数或路径
        source = str(source)
        is_file = Path(source).suffix[1:] in (IMG_FORMATS | VID_FORMATS)  # 检查是否为支持的图像或视频格式
        is_url = source.lower().startswith(("https://", "http://", "rtsp://", "rtmp://", "tcp://"))  # 检查是否为 URL
        webcam = source.isnumeric() or source.endswith(".streams") or (is_url and not is_file)  # 是否为摄像头
        screenshot = source.lower() == "screen"  # 是否为屏幕截图
        if is_url and is_file:
            source = check_file(source)  # 下载文件
    elif isinstance(source, LOADERS):  # 检查是否为特定加载器类型
        in_memory = True  # 是否在内存中
    elif isinstance(source, (list, tuple)):  # 检查是否为列表或元组
        source = autocast_list(source)  # 转换列表元素为 PIL 图像或 np 数组
        from_img = True  # 是否从图像获取
    elif isinstance(source, (Image.Image, np.ndarray)):  # 检查是否为 PIL 图像或 np 数组
        from_img = True  # 是否从图像获取
    elif isinstance(source, torch.Tensor):  # 检查是否为 PyTorch 张量
        tensor = True  # 是否为张量
    else:
        raise TypeError("Unsupported image type. For supported types see https://docs.ultralytics.com/modes/predict")  # 抛出错误，不支持的图像类型

    return source, webcam, screenshot, from_img, in_memory, tensor  # 返回源数据及相关标志值


# 加载推断数据源，用于目标检测，并应用必要的转换
def load_inference_source(source=None, batch=1, vid_stride=1, buffer=False):
    """
    Loads an inference source for object detection and applies necessary transformations.
    """
    # 返回一个 InfiniteDataLoader 对象，用于推断数据源加载
    return InfiniteDataLoader(
        dataset=dataset,  # 数据集对象
        batch_size=batch,  # 批处理大小
        shuffle=shuffle and sampler is None,  # 是否打乱数据顺序
        num_workers=nw,  # 工作线程数量
        sampler=sampler,  # 分布式采样器
        pin_memory=PIN_MEMORY,  # 是否将数据保存在固定内存中
        collate_fn=getattr(dataset, "collate_fn", None),  # 数据集的整理函数
        worker_init_fn=seed_worker,  # 工作线程初始化函数
        generator=generator,  # 随机数生成器
    )
    Args:
        source (str, Path, Tensor, PIL.Image, np.ndarray): 接收推理输入的源数据类型，可以是文件路径、张量、图像对象等。
        batch (int, optional): 数据加载器的批大小。默认为1。
        vid_stride (int, optional): 视频源的帧间隔。默认为1。
        buffer (bool, optional): 决定流式帧是否缓存。默认为False。

    Returns:
        dataset (Dataset): 返回特定输入源的数据集对象。
    """
    # 检查输入源的类型并进行适配
    source, stream, screenshot, from_img, in_memory, tensor = check_source(source)
    
    # 如果数据源在内存中，则使用其类型；否则根据源的不同选择源类型
    source_type = source.source_type if in_memory else SourceTypes(stream, screenshot, from_img, tensor)

    # 数据加载器选择
    if tensor:
        # 如果输入源是张量，则加载张量数据集
        dataset = LoadTensor(source)
    elif in_memory:
        # 如果输入源在内存中，则直接使用该源作为数据集
        dataset = source
    elif stream:
        # 如果输入源是流式数据（视频流），则加载流数据集
        dataset = LoadStreams(source, vid_stride=vid_stride, buffer=buffer)
    elif screenshot:
        # 如果输入源是截图，则加载截图数据集
        dataset = LoadScreenshots(source)
    elif from_img:
        # 如果输入源是PIL图像或numpy数组，则加载对应数据集
        dataset = LoadPilAndNumpy(source)
    else:
        # 其他情况下（图片或视频文件），加载图片和视频数据集
        dataset = LoadImagesAndVideos(source, batch=batch, vid_stride=vid_stride)

    # 将源类型附加到数据集对象
    setattr(dataset, "source_type", source_type)

    # 返回创建的数据集对象
    return dataset
```