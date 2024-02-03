# `.\PaddleOCR\benchmark\PaddleOCR_DBNet\data_loader\__init__.py`

```py
# -*- coding: utf-8 -*-
# @Time    : 2019/8/23 21:52
# @Author  : zhoujun
# 导入所需的模块和库
import copy

import PIL
import numpy as np
import paddle
from paddle.io import DataLoader, DistributedBatchSampler, BatchSampler

from paddle.vision import transforms

# 定义一个函数，用于获取训练dataset
def get_dataset(data_path, module_name, transform, dataset_args):
    """
    获取训练dataset
    :param data_path: dataset文件列表，每个文件内以如下格式存储 ‘path/to/img\tlabel’
    :param module_name: 所使用的自定义dataset名称，目前只支持data_loaders.ImageDataset
    :param transform: 该数据集使用的transforms
    :param dataset_args: module_name的参数
    :return: 如果data_path列表不为空，返回对于的ConcatDataset对象，否则None
    """
    # 导入自定义的dataset模块
    from . import dataset
    # 根据module_name获取对应的dataset对象
    s_dataset = getattr(dataset, module_name)(transform=transform,
                                              data_path=data_path,
                                              **dataset_args)
    return s_dataset

# 定义一个函数，用于获取transforms
def get_transforms(transforms_config):
    tr_list = []
    for item in transforms_config:
        if 'args' not in item:
            args = {}
        else:
            args = item['args']
        # 根据配置信息创建transforms对象
        cls = getattr(transforms, item['type'])(**args)
        tr_list.append(cls)
    # 组合transforms对象
    tr_list = transforms.Compose(tr_list)
    return tr_list

# 定义一个类，用于处理ICDAR数据集的数据收集
class ICDARCollectFN:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, batch):
        data_dict = {}
        to_tensor_keys = []
        for sample in batch:
            for k, v in sample.items():
                if k not in data_dict:
                    data_dict[k] = []
                if isinstance(v, (np.ndarray, paddle.Tensor, PIL.Image.Image)):
                    if k not in to_tensor_keys:
                        to_tensor_keys.append(k)
                data_dict[k].append(v)
        for k in to_tensor_keys:
            data_dict[k] = paddle.stack(data_dict[k], 0)
        return data_dict

# 定义一个函数，用于获取dataloader
def get_dataloader(module_config, distributed=False):
    if module_config is None:
        return None
    # 深拷贝module_config
    config = copy.deepcopy(module_config)
    # 从配置中获取数据集参数
    dataset_args = config['dataset']['args']
    # 如果数据集参数中包含transforms，则获取transforms
    if 'transforms' in dataset_args:
        img_transfroms = get_transforms(dataset_args.pop('transforms'))
    else:
        img_transfroms = None
    # 获取数据集类型和数据路径
    dataset_name = config['dataset']['type']
    data_path = dataset_args.pop('data_path')
    # 如果数据路径为None，则返回None
    if data_path == None:
        return None

    # 过滤掉数据路径中为None的部分
    data_path = [x for x in data_path if x is not None]
    # 如果数据路径为空，则返回None
    if len(data_path) == 0:
        return None
    # 检查配置中是否存在collate_fn，如果不存在或者为None，则设置为None
    if 'collate_fn' not in config['loader'] or config['loader'][
            'collate_fn'] is None or len(config['loader']['collate_fn']) == 0:
        config['loader']['collate_fn'] = None
    else:
        # 否则，将collate_fn从字符串转换为函数
        config['loader']['collate_fn'] = eval(config['loader']['collate_fn'])()

    # 获取数据集对象
    _dataset = get_dataset(
        data_path=data_path,
        module_name=dataset_name,
        transform=img_transfroms,
        dataset_args=dataset_args)
    sampler = None
    # 如果是分布式训练
    if distributed:
        # 使用DistributedSampler创建批次采样器
        batch_sampler = DistributedBatchSampler(
            dataset=_dataset,
            batch_size=config['loader'].pop('batch_size'),
            shuffle=config['loader'].pop('shuffle'))
    else:
        # 否则，使用BatchSampler创建批次采样器
        batch_sampler = BatchSampler(
            dataset=_dataset,
            batch_size=config['loader'].pop('batch_size'),
            shuffle=config['loader'].pop('shuffle'))
    # 创建数据加载器
    loader = DataLoader(
        dataset=_dataset, batch_sampler=batch_sampler, **config['loader'])
    return loader
```