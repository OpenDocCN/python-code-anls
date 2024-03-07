# `.\YOLO-World\yolo_world\datasets\mm_dataset.py`

```py
# 导入所需的模块和类
import copy
import json
import logging
from typing import Callable, List, Union

from mmengine.logging import print_log
from mmengine.dataset.base_dataset import (
        BaseDataset, Compose, force_full_init)
from mmyolo.registry import DATASETS

# 注册MultiModalDataset类到DATASETS
@DATASETS.register_module()
class MultiModalDataset:
    """Multi-modal dataset."""

    def __init__(self,
                 dataset: Union[BaseDataset, dict],
                 class_text_path: str = None,
                 test_mode: bool = True,
                 pipeline: List[Union[dict, Callable]] = [],
                 lazy_init: bool = False) -> None:
        # 初始化dataset属性
        self.dataset: BaseDataset
        if isinstance(dataset, dict):
            self.dataset = DATASETS.build(dataset)
        elif isinstance(dataset, BaseDataset):
            self.dataset = dataset
        else:
            raise TypeError(
                'dataset must be a dict or a BaseDataset, '
                f'but got {dataset}')

        # 加载类别文本文件
        if class_text_path is not None:
            self.class_texts = json.load(open(class_text_path, 'r'))
            # ori_classes = self.dataset.metainfo['classes']
            # assert len(ori_classes) == len(self.class_texts), \
            #     ('The number of classes in the dataset and the class text'
            #      'file must be the same.')
        else:
            self.class_texts = None

        # 设置测试模式
        self.test_mode = test_mode
        # 获取数据集的元信息
        self._metainfo = self.dataset.metainfo
        # 初始化数据处理pipeline
        self.pipeline = Compose(pipeline)

        # 标记是否已完全初始化
        self._fully_initialized = False
        # 如果不是延迟初始化，则进行完全初始化
        if not lazy_init:
            self.full_init()

    @property
    def metainfo(self) -> dict:
        # 返回元信息的深拷贝
        return copy.deepcopy(self._metainfo)

    def full_init(self) -> None:
        """``full_init`` dataset."""
        # 如果已经完全初始化，则直接返回
        if self._fully_initialized:
            return

        # 对数据集进行完全初始化
        self.dataset.full_init()
        self._ori_len = len(self.dataset)
        self._fully_initialized = True

    @force_full_init
    # 根据索引获取数据信息，返回一个字典
    def get_data_info(self, idx: int) -> dict:
        """Get annotation by index."""
        # 通过数据集对象获取指定索引的数据信息
        data_info = self.dataset.get_data_info(idx)
        # 如果类别文本不为空，则将其添加到数据信息字典中
        if self.class_texts is not None:
            data_info.update({'texts': self.class_texts})
        return data_info

    # 根据索引获取数据
    def __getitem__(self, idx):
        # 如果数据集未完全初始化，则打印警告信息并手动调用`full_init`方法以加快速度
        if not self._fully_initialized:
            print_log(
                'Please call `full_init` method manually to '
                'accelerate the speed.',
                logger='current',
                level=logging.WARNING)
            self.full_init()

        # 获取数据信息
        data_info = self.get_data_info(idx)

        # 如果数据集具有'test_mode'属性且不为测试模式，则将数据集信息添加到数据信息字典中
        if hasattr(self.dataset, 'test_mode') and not self.dataset.test_mode:
            data_info['dataset'] = self
        # 如果不是测试模式，则将数据集信息添加到数据信息字典中
        elif not self.test_mode:
            data_info['dataset'] = self
        # 返回经过管道处理后的数据信息
        return self.pipeline(data_info)

    # 返回数据集的长度
    @force_full_init
    def __len__(self) -> int:
        return self._ori_len
# 注册 MultiModalMixedDataset 类到 DATASETS 模块
@DATASETS.register_module()
class MultiModalMixedDataset(MultiModalDataset):
    """Multi-modal Mixed dataset.
    mix "detection dataset" and "caption dataset"
    Args:
        dataset_type (str): dataset type, 'detection' or 'caption'
    """

    # 初始化方法，接受多种参数，包括 dataset、class_text_path、dataset_type、test_mode、pipeline 和 lazy_init
    def __init__(self,
                 dataset: Union[BaseDataset, dict],
                 class_text_path: str = None,
                 dataset_type: str = 'detection',
                 test_mode: bool = True,
                 pipeline: List[Union[dict, Callable]] = [],
                 lazy_init: bool = False) -> None:
        # 设置 dataset_type 属性
        self.dataset_type = dataset_type
        # 调用父类的初始化方法
        super().__init__(dataset,
                         class_text_path,
                         test_mode,
                         pipeline,
                         lazy_init)

    # 强制完全初始化装饰器，用于 get_data_info 方法
    @force_full_init
    def get_data_info(self, idx: int) -> dict:
        """Get annotation by index."""
        # 调用 dataset 的 get_data_info 方法获取数据信息
        data_info = self.dataset.get_data_info(idx)
        # 如果 class_texts 不为空，则更新 data_info 中的 'texts' 字段
        if self.class_texts is not None:
            data_info.update({'texts': self.class_texts})
        # 根据 dataset_type 设置 data_info 中的 'is_detection' 字段
        data_info['is_detection'] = 1 \
            if self.dataset_type == 'detection' else 0
        return data_info
```