# `.\yolov8\ultralytics\data\dataset.py`

```py
# Ultralytics YOLO , AGPL-3.0 license

# 导入必要的模块和库
import contextlib
import json
from collections import defaultdict
from itertools import repeat
from multiprocessing.pool import ThreadPool
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import ConcatDataset

# 导入 Ultralytics 自定义的工具函数和类
from ultralytics.utils import LOCAL_RANK, NUM_THREADS, TQDM, colorstr
from ultralytics.utils.ops import resample_segments
from ultralytics.utils.torch_utils import TORCHVISION_0_18

# 导入数据增强相关模块
from .augment import (
    Compose,
    Format,
    Instances,
    LetterBox,
    RandomLoadText,
    classify_augmentations,
    classify_transforms,
    v8_transforms,
)
# 导入基础数据集类和工具函数
from .base import BaseDataset
from .utils import (
    HELP_URL,
    LOGGER,
    get_hash,
    img2label_paths,
    load_dataset_cache_file,
    save_dataset_cache_file,
    verify_image,
    verify_image_label,
)

# Ultralytics dataset *.cache version, >= 1.0.0 for YOLOv8
# 数据集缓存版本号
DATASET_CACHE_VERSION = "1.0.3"

# YOLODataset 类，用于加载 YOLO 格式的对象检测和/或分割标签数据集
class YOLODataset(BaseDataset):
    """
    Dataset class for loading object detection and/or segmentation labels in YOLO format.

    Args:
        data (dict, optional): A dataset YAML dictionary. Defaults to None.
        task (str): An explicit arg to point current task, Defaults to 'detect'.

    Returns:
        (torch.utils.data.Dataset): A PyTorch dataset object that can be used for training an object detection model.
    """

    # 初始化方法，设置数据集类型和任务类型
    def __init__(self, *args, data=None, task="detect", **kwargs):
        """Initializes the YOLODataset with optional configurations for segments and keypoints."""
        # 根据任务类型设置是否使用分割标签、关键点标签或旋转矩形标签
        self.use_segments = task == "segment"
        self.use_keypoints = task == "pose"
        self.use_obb = task == "obb"
        self.data = data
        # 断言不能同时使用分割标签和关键点标签
        assert not (self.use_segments and self.use_keypoints), "Can not use both segments and keypoints."
        # 调用父类 BaseDataset 的初始化方法
        super().__init__(*args, **kwargs)
    def cache_labels(self, path=Path("./labels.cache")):
        """
        Cache dataset labels, check images and read shapes.

        Args:
            path (Path): Path where to save the cache file. Default is Path('./labels.cache').

        Returns:
            (dict): labels.
        """
        # 初始化空字典用于存储标签数据
        x = {"labels": []}
        # 初始化计数器和消息列表
        nm, nf, ne, nc, msgs = 0, 0, 0, 0, []  # number missing, found, empty, corrupt, messages
        # 构建描述信息字符串，表示正在扫描路径下的文件
        desc = f"{self.prefix}Scanning {path.parent / path.stem}..."
        # 获取图像文件总数
        total = len(self.im_files)
        # 从数据中获取关键点形状信息
        nkpt, ndim = self.data.get("kpt_shape", (0, 0))
        # 如果使用关键点信息且关键点数量或维度不正确，抛出异常
        if self.use_keypoints and (nkpt <= 0 or ndim not in {2, 3}):
            raise ValueError(
                "'kpt_shape' in data.yaml missing or incorrect. Should be a list with [number of "
                "keypoints, number of dims (2 for x,y or 3 for x,y,visible)], i.e. 'kpt_shape: [17, 3]'"
            )
        # 使用线程池处理图像验证任务
        with ThreadPool(NUM_THREADS) as pool:
            # 并行处理图像验证任务，获取验证结果
            results = pool.imap(
                func=verify_image_label,
                iterable=zip(
                    self.im_files,
                    self.label_files,
                    repeat(self.prefix),
                    repeat(self.use_keypoints),
                    repeat(len(self.data["names"])),
                    repeat(nkpt),
                    repeat(ndim),
                ),
            )
            # 初始化进度条对象
            pbar = TQDM(results, desc=desc, total=total)
            # 遍历进度条以显示验证进度
            for im_file, lb, shape, segments, keypoint, nm_f, nf_f, ne_f, nc_f, msg in pbar:
                # 更新计数器
                nm += nm_f
                nf += nf_f
                ne += ne_f
                nc += nc_f
                # 如果图像文件存在，则添加标签信息到x["labels"]中
                if im_file:
                    x["labels"].append(
                        {
                            "im_file": im_file,
                            "shape": shape,
                            "cls": lb[:, 0:1],  # n, 1
                            "bboxes": lb[:, 1:],  # n, 4
                            "segments": segments,
                            "keypoints": keypoint,
                            "normalized": True,
                            "bbox_format": "xywh",
                        }
                    )
                # 如果有消息，则添加到消息列表中
                if msg:
                    msgs.append(msg)
                # 更新进度条描述信息
                pbar.desc = f"{desc} {nf} images, {nm + ne} backgrounds, {nc} corrupt"
            # 关闭进度条
            pbar.close()

        # 如果有警告消息，则记录日志
        if msgs:
            LOGGER.info("\n".join(msgs))
        # 如果未找到标签，则记录警告日志
        if nf == 0:
            LOGGER.warning(f"{self.prefix}WARNING ⚠️ No labels found in {path}. {HELP_URL}")
        # 计算数据集文件的哈希值并存储在结果字典中
        x["hash"] = get_hash(self.label_files + self.im_files)
        # 将结果相关信息存储在结果字典中
        x["results"] = nf, nm, ne, nc, len(self.im_files)
        # 将警告消息列表存储在结果字典中
        x["msgs"] = msgs  # warnings
        # 保存数据集缓存文件
        save_dataset_cache_file(self.prefix, path, x, DATASET_CACHE_VERSION)
        # 返回结果字典
        return x
    def get_labels(self):
        """Returns dictionary of labels for YOLO training."""
        # 获取图像文件对应的标签文件路径字典
        self.label_files = img2label_paths(self.im_files)
        # 构建缓存文件路径，并尝试加载 *.cache 文件
        cache_path = Path(self.label_files[0]).parent.with_suffix(".cache")
        try:
            # 尝试加载数据集缓存文件
            cache, exists = load_dataset_cache_file(cache_path), True  # attempt to load a *.cache file
            # 检查缓存文件版本与哈希值是否匹配当前要求
            assert cache["version"] == DATASET_CACHE_VERSION  # matches current version
            assert cache["hash"] == get_hash(self.label_files + self.im_files)  # identical hash
        except (FileNotFoundError, AssertionError, AttributeError):
            # 加载失败时，重新生成标签缓存
            cache, exists = self.cache_labels(cache_path), False  # run cache ops

        # 显示缓存信息
        nf, nm, ne, nc, n = cache.pop("results")  # found, missing, empty, corrupt, total
        if exists and LOCAL_RANK in {-1, 0}:
            d = f"Scanning {cache_path}... {nf} images, {nm + ne} backgrounds, {nc} corrupt"
            TQDM(None, desc=self.prefix + d, total=n, initial=n)  # display results
            if cache["msgs"]:
                LOGGER.info("\n".join(cache["msgs"]))  # display warnings

        # 读取缓存内容
        [cache.pop(k) for k in ("hash", "version", "msgs")]  # remove items
        labels = cache["labels"]
        if not labels:
            # 若缓存中无标签信息，则发出警告
            LOGGER.warning(f"WARNING ⚠️ No images found in {cache_path}, training may not work correctly. {HELP_URL}")
        self.im_files = [lb["im_file"] for lb in labels]  # update im_files

        # 检查数据集是否仅含有框或者分段信息
        lengths = ((len(lb["cls"]), len(lb["bboxes"]), len(lb["segments"])) for lb in labels)
        len_cls, len_boxes, len_segments = (sum(x) for x in zip(*lengths))
        if len_segments and len_boxes != len_segments:
            # 若分段数与框数不相等，则发出警告，并移除所有分段信息
            LOGGER.warning(
                f"WARNING ⚠️ Box and segment counts should be equal, but got len(segments) = {len_segments}, "
                f"len(boxes) = {len_boxes}. To resolve this only boxes will be used and all segments will be removed. "
                "To avoid this please supply either a detect or segment dataset, not a detect-segment mixed dataset."
            )
            for lb in labels:
                lb["segments"] = []
        if len_cls == 0:
            # 若标签数量为零，则发出警告
            LOGGER.warning(f"WARNING ⚠️ No labels found in {cache_path}, training may not work correctly. {HELP_URL}")
        return labels
    # 构建并追加变换操作到列表中
    def build_transforms(self, hyp=None):
        """Builds and appends transforms to the list."""
        # 如果启用数据增强
        if self.augment:
            # 设置混合和镶嵌的比例，如果未使用矩形模式则为0.0
            hyp.mosaic = hyp.mosaic if self.augment and not self.rect else 0.0
            hyp.mixup = hyp.mixup if self.augment and not self.rect else 0.0
            # 使用指定的版本和超参数构建变换
            transforms = v8_transforms(self, self.imgsz, hyp)
        else:
            # 否则，使用指定的图像尺寸创建 LetterBox 变换
            transforms = Compose([LetterBox(new_shape=(self.imgsz, self.imgsz), scaleup=False)])
        # 添加格式化变换到变换列表
        transforms.append(
            Format(
                bbox_format="xywh",
                normalize=True,
                return_mask=self.use_segments,
                return_keypoint=self.use_keypoints,
                return_obb=self.use_obb,
                batch_idx=True,
                mask_ratio=hyp.mask_ratio,
                mask_overlap=hyp.overlap_mask,
                bgr=hyp.bgr if self.augment else 0.0,  # 仅影响训练时的图像背景
            )
        )
        return transforms

    # 关闭镶嵌，复制粘贴和混合选项，并构建转换
    def close_mosaic(self, hyp):
        """Sets mosaic, copy_paste and mixup options to 0.0 and builds transformations."""
        # 将镶嵌比例设置为0.0
        hyp.mosaic = 0.0
        # 保持与之前版本v8 close-mosaic相同的行为，复制粘贴比例设置为0.0
        hyp.copy_paste = 0.0
        # 保持与之前版本v8 close-mosaic相同的行为，混合比例设置为0.0
        hyp.mixup = 0.0
        # 使用给定超参数构建转换
        self.transforms = self.build_transforms(hyp)

    def update_labels_info(self, label):
        """
        Custom your label format here.

        Note:
            cls is not with bboxes now, classification and semantic segmentation need an independent cls label
            Can also support classification and semantic segmentation by adding or removing dict keys there.
        """
        # 弹出标签中的边界框信息
        bboxes = label.pop("bboxes")
        # 弹出标签中的分割信息，默认为空列表
        segments = label.pop("segments", [])
        # 弹出标签中的关键点信息，默认为None
        keypoints = label.pop("keypoints", None)
        # 弹出标签中的边界框格式信息
        bbox_format = label.pop("bbox_format")
        # 弹出标签中的归一化信息
        normalized = label.pop("normalized")

        # 如果使用方向框，则设置分割重新采样数为100，否则设置为1000
        segment_resamples = 100 if self.use_obb else 1000
        # 如果存在分割信息
        if len(segments) > 0:
            # 对分割信息进行重采样，返回重采样后的堆栈数组
            segments = np.stack(resample_segments(segments, n=segment_resamples), axis=0)
        else:
            # 否则创建全零数组，形状为(0, 1000, 2)
            segments = np.zeros((0, segment_resamples, 2), dtype=np.float32)
        # 创建实例对象，包含边界框、分割、关键点等信息
        label["instances"] = Instances(bboxes, segments, keypoints, bbox_format=bbox_format, normalized=normalized)
        return label
    # 定义一个函数用于将数据样本整理成批次
    def collate_fn(batch):
        """Collates data samples into batches."""
        # 创建一个新的批次字典
        new_batch = {}
        # 获取批次中第一个样本的所有键
        keys = batch[0].keys()
        # 获取批次中所有样本的值，并转置成列表形式
        values = list(zip(*[list(b.values()) for b in batch]))
        # 遍历所有键值对
        for i, k in enumerate(keys):
            # 获取当前键对应的值列表
            value = values[i]
            # 如果键是 "img"，则将值列表堆叠为张量
            if k == "img":
                value = torch.stack(value, 0)
            # 如果键在 {"masks", "keypoints", "bboxes", "cls", "segments", "obb"} 中，
            # 则将值列表连接为张量
            if k in {"masks", "keypoints", "bboxes", "cls", "segments", "obb"}:
                value = torch.cat(value, 0)
            # 将处理后的值赋给新的批次字典对应的键
            new_batch[k] = value
        # 将新的批次索引列表转换为列表形式
        new_batch["batch_idx"] = list(new_batch["batch_idx"])
        # 为每个批次索引添加目标图像的索引以供 build_targets() 使用
        for i in range(len(new_batch["batch_idx"])):
            new_batch["batch_idx"][i] += i  # add target image index for build_targets()
        # 将处理后的批次索引连接为张量
        new_batch["batch_idx"] = torch.cat(new_batch["batch_idx"], 0)
        # 返回整理好的新批次字典
        return new_batch
class YOLOMultiModalDataset(YOLODataset):
    """
    Dataset class for loading object detection and/or segmentation labels in YOLO format.

    Args:
        data (dict, optional): A dataset YAML dictionary. Defaults to None.
        task (str): An explicit arg to point current task, Defaults to 'detect'.

    Returns:
        (torch.utils.data.Dataset): A PyTorch dataset object that can be used for training an object detection model.
    """

    def __init__(self, *args, data=None, task="detect", **kwargs):
        """Initializes a dataset object for object detection tasks with optional specifications."""
        # 调用父类构造函数初始化对象
        super().__init__(*args, data=data, task=task, **kwargs)

    def update_labels_info(self, label):
        """Add texts information for multi-modal model training."""
        # 调用父类方法更新标签信息
        labels = super().update_labels_info(label)
        # NOTE: some categories are concatenated with its synonyms by `/`.
        # 将数据集中的类别名按照 `/` 分割成列表，添加到标签中
        labels["texts"] = [v.split("/") for _, v in self.data["names"].items()]
        return labels

    def build_transforms(self, hyp=None):
        """Enhances data transformations with optional text augmentation for multi-modal training."""
        # 调用父类方法构建数据转换列表
        transforms = super().build_transforms(hyp)
        if self.augment:
            # NOTE: hard-coded the args for now.
            # 如果开启数据增强，插入一个文本加载的转换操作
            transforms.insert(-1, RandomLoadText(max_samples=min(self.data["nc"], 80), padding=True))
        return transforms


class GroundingDataset(YOLODataset):
    """Handles object detection tasks by loading annotations from a specified JSON file, supporting YOLO format."""

    def __init__(self, *args, task="detect", json_file, **kwargs):
        """Initializes a GroundingDataset for object detection, loading annotations from a specified JSON file."""
        # 断言任务类型为 "detect"
        assert task == "detect", "`GroundingDataset` only support `detect` task for now!"
        self.json_file = json_file
        # 调用父类构造函数初始化对象
        super().__init__(*args, task=task, data={}, **kwargs)

    def get_img_files(self, img_path):
        """The image files would be read in `get_labels` function, return empty list here."""
        # 返回空列表，因为图像文件在 `get_labels` 函数中读取
        return []
    def get_labels(self):
        """Loads annotations from a JSON file, filters, and normalizes bounding boxes for each image."""
        labels = []  # 初始化空列表用于存储标签数据
        LOGGER.info("Loading annotation file...")  # 记录日志，指示正在加载注释文件
        with open(self.json_file, "r") as f:
            annotations = json.load(f)  # 从 JSON 文件中加载注释数据
        images = {f'{x["id"]:d}': x for x in annotations["images"]}  # 创建图像字典，以图像ID为键
        img_to_anns = defaultdict(list)
        for ann in annotations["annotations"]:
            img_to_anns[ann["image_id"]].append(ann)  # 根据图像ID将注释分组到字典中
        for img_id, anns in TQDM(img_to_anns.items(), desc=f"Reading annotations {self.json_file}"):
            img = images[f"{img_id:d}"]  # 获取当前图像的信息
            h, w, f = img["height"], img["width"], img["file_name"]  # 获取图像的高度、宽度和文件名
            im_file = Path(self.img_path) / f  # 构建图像文件的路径
            if not im_file.exists():
                continue  # 如果图像文件不存在，则跳过处理
            self.im_files.append(str(im_file))  # 将图像文件路径添加到实例变量中
            bboxes = []  # 初始化空列表用于存储边界框信息
            cat2id = {}  # 初始化空字典，用于存储类别到ID的映射关系
            texts = []  # 初始化空列表用于存储文本信息
            for ann in anns:
                if ann["iscrowd"]:
                    continue  # 如果注释标记为iscrowd，则跳过处理
                box = np.array(ann["bbox"], dtype=np.float32)  # 获取注释中的边界框信息并转换为numpy数组
                box[:2] += box[2:] / 2  # 将边界框坐标转换为中心点坐标
                box[[0, 2]] /= float(w)  # 归一化边界框的x坐标
                box[[1, 3]] /= float(h)  # 归一化边界框的y坐标
                if box[2] <= 0 or box[3] <= 0:
                    continue  # 如果边界框的宽度或高度小于等于零，则跳过处理

                cat_name = " ".join([img["caption"][t[0]:t[1]] for t in ann["tokens_positive"]])  # 从tokens_positive获取类别名称
                if cat_name not in cat2id:
                    cat2id[cat_name] = len(cat2id)  # 将类别名称映射到唯一的ID
                    texts.append([cat_name])  # 将类别名称添加到文本列表中
                cls = cat2id[cat_name]  # 获取类别的ID
                box = [cls] + box.tolist()  # 将类别ID与边界框信息合并
                if box not in bboxes:
                    bboxes.append(box)  # 将边界框信息添加到列表中
            lb = np.array(bboxes, dtype=np.float32) if len(bboxes) else np.zeros((0, 5), dtype=np.float32)  # 构建边界框数组或者空数组
            labels.append(
                {
                    "im_file": im_file,
                    "shape": (h, w),
                    "cls": lb[:, 0:1],  # 提取类别信息，n行1列
                    "bboxes": lb[:, 1:],  # 提取边界框信息，n行4列
                    "normalized": True,
                    "bbox_format": "xywh",
                    "texts": texts,
                }
            )  # 将图像信息和处理后的标签数据添加到标签列表中
        return labels  # 返回所有图像的标签信息列表

    def build_transforms(self, hyp=None):
        """Configures augmentations for training with optional text loading; `hyp` adjusts augmentation intensity."""
        transforms = super().build_transforms(hyp)  # 调用父类方法，获取基本的数据增强列表
        if self.augment:
            # NOTE: hard-coded the args for now.
            transforms.insert(-1, RandomLoadText(max_samples=80, padding=True))  # 在数据增强列表的倒数第二个位置插入文本加载的随机操作
        return transforms  # 返回配置后的数据增强列表
class YOLOConcatDataset(ConcatDataset):
    """
    Dataset as a concatenation of multiple datasets.

    This class is useful to assemble different existing datasets.
    """

    @staticmethod
    def collate_fn(batch):
        """Collates data samples into batches."""
        return YOLODataset.collate_fn(batch)



# TODO: support semantic segmentation
class SemanticDataset(BaseDataset):
    """
    Semantic Segmentation Dataset.

    This class is responsible for handling datasets used for semantic segmentation tasks. It inherits functionalities
    from the BaseDataset class.

    Note:
        This class is currently a placeholder and needs to be populated with methods and attributes for supporting
        semantic segmentation tasks.
    """

    def __init__(self):
        """Initialize a SemanticDataset object."""
        super().__init__()

class ClassificationDataset:
    """
    Extends torchvision ImageFolder to support YOLO classification tasks, offering functionalities like image
    augmentation, caching, and verification. It's designed to efficiently handle large datasets for training deep
    learning models, with optional image transformations and caching mechanisms to speed up training.

    This class allows for augmentations using both torchvision and Albumentations libraries, and supports caching images
    in RAM or on disk to reduce IO overhead during training. Additionally, it implements a robust verification process
    to ensure data integrity and consistency.

    Attributes:
        cache_ram (bool): Indicates if caching in RAM is enabled.
        cache_disk (bool): Indicates if caching on disk is enabled.
        samples (list): A list of tuples, each containing the path to an image, its class index, path to its .npy cache
                        file (if caching on disk), and optionally the loaded image array (if caching in RAM).
        torch_transforms (callable): PyTorch transforms to be applied to the images.
    """

    def __getitem__(self, i):
        """Returns subset of data and targets corresponding to given indices."""
        f, j, fn, im = self.samples[i]  # filename, index, filename.with_suffix('.npy'), image
        if self.cache_ram:
            if im is None:  # Warning: two separate if statements required here, do not combine this with previous line
                im = self.samples[i][3] = cv2.imread(f)
        elif self.cache_disk:
            if not fn.exists():  # load npy
                np.save(fn.as_posix(), cv2.imread(f), allow_pickle=False)
            im = np.load(fn)
        else:  # read image
            im = cv2.imread(f)  # BGR
        # Convert NumPy array to PIL image
        im = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        sample = self.torch_transforms(im)
        return {"img": sample, "cls": j}

    def __len__(self) -> int:
        """Return the total number of samples in the dataset."""
        return len(self.samples)
    def verify_images(self):
        """Verify all images in dataset."""
        # 构建描述信息，指定要扫描的根目录
        desc = f"{self.prefix}Scanning {self.root}..."
        # 根据根目录生成对应的缓存文件路径
        path = Path(self.root).with_suffix(".cache")  # *.cache file path
        
        # 尝试加载缓存文件，处理可能出现的文件未找到、断言错误和属性错误
        with contextlib.suppress(FileNotFoundError, AssertionError, AttributeError):
            # 加载数据集缓存文件
            cache = load_dataset_cache_file(path)  # attempt to load a *.cache file
            # 断言缓存文件版本与当前版本匹配
            assert cache["version"] == DATASET_CACHE_VERSION  # matches current version
            # 断言缓存文件的哈希与数据集样本的哈希一致
            assert cache["hash"] == get_hash([x[0] for x in self.samples])  # identical hash
            # 解构缓存结果，包括发现的、丢失的、空的、损坏的样本数量以及样本列表
            nf, nc, n, samples = cache.pop("results")  # found, missing, empty, corrupt, total
            # 如果在主机的本地或者单个进程运行时，显示描述信息和进度条
            if LOCAL_RANK in {-1, 0}:
                d = f"{desc} {nf} images, {nc} corrupt"
                TQDM(None, desc=d, total=n, initial=n)
                # 如果存在警告消息，则记录日志显示
                if cache["msgs"]:
                    LOGGER.info("\n".join(cache["msgs"]))  # display warnings
            # 返回样本列表
            return samples
        
        # 如果未能检索到缓存文件，则执行扫描操作
        nf, nc, msgs, samples, x = 0, 0, [], [], {}
        # 使用线程池并发执行图像验证函数
        with ThreadPool(NUM_THREADS) as pool:
            results = pool.imap(func=verify_image, iterable=zip(self.samples, repeat(self.prefix)))
            # 创建进度条并显示扫描描述信息
            pbar = TQDM(results, desc=desc, total=len(self.samples))
            for sample, nf_f, nc_f, msg in pbar:
                # 如果图像未损坏，则将其添加到样本列表中
                if nf_f:
                    samples.append(sample)
                # 如果存在警告消息，则添加到消息列表中
                if msg:
                    msgs.append(msg)
                # 更新发现的和损坏的图像数量
                nf += nf_f
                nc += nc_f
                # 更新进度条的描述信息
                pbar.desc = f"{desc} {nf} images, {nc} corrupt"
            # 关闭进度条
            pbar.close()
        
        # 如果存在警告消息，则记录日志显示
        if msgs:
            LOGGER.info("\n".join(msgs))
        
        # 计算数据集样本的哈希值并保存相关信息到 x 字典
        x["hash"] = get_hash([x[0] for x in self.samples])
        x["results"] = nf, nc, len(samples), samples
        x["msgs"] = msgs  # warnings
        
        # 将数据集缓存信息保存到缓存文件中
        save_dataset_cache_file(self.prefix, path, x, DATASET_CACHE_VERSION)
        
        # 返回发现的样本列表
        return samples
```