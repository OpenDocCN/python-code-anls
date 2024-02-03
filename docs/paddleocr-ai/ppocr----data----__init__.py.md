# `.\PaddleOCR\ppocr\data\__init__.py`

```
# 版权声明，告知代码版权归属
# 根据 Apache 许可证 2.0 版本规定使用该文件
# 获取许可证的链接
# 根据适用法律或书面同意，根据许可证分发的软件基于“原样”分发，没有任何明示或暗示的保证或条件
# 查看特定语言的许可证，以确定权限和限制

# 导入必要的库
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import os
import sys
import numpy as np
import skimage
import paddle
import signal
import random

# 获取当前文件所在目录的绝对路径
__dir__ = os.path.dirname(os.path.abspath(__file__))
# 将上级目录添加到系统路径中
sys.path.append(os.path.abspath(os.path.join(__dir__, '../..')))

# 导入额外的库和模块
import copy
from paddle.io import Dataset, DataLoader, BatchSampler, DistributedBatchSampler
import paddle.distributed as dist

# 导入自定义的数据增强操作
from ppocr.data.imaug import transform, create_operators
# 导入简单数据集类
from ppocr.data.simple_dataset import SimpleDataSet, MultiScaleDataSet
# 导入 LMDB 数据集类
from ppocr.data.lmdb_dataset import LMDBDataSet, LMDBDataSetSR, LMDBDataSetTableMaster
# 导入 PGNet 数据集类
from ppocr.data.pgnet_dataset import PGDataSet
# 导入 PubTab 数据集类
from ppocr.data.pubtab_dataset import PubTabDataSet
# 导入多尺度采样器类
from ppocr.data.multi_scale_sampler import MultiScaleSampler

# 为 PaddleX 数据集类型定义别名
TextDetDataset = SimpleDataSet
TextRecDataset = SimpleDataSet
MSTextRecDataset = MultiScaleDataSet
PubTabTableRecDataset = PubTabDataSet
KieDataset = SimpleDataSet

# 导出的模块和函数
__all__ = [
    'build_dataloader', 'transform', 'create_operators', 'set_signal_handlers'
]

# 信号处理函数，用于终止所有子进程
def term_mp(sig_num, frame):
    """ kill all child processes
    """
    pid = os.getpid()
    pgid = os.getpgid(os.getpid())
    print("main proc {} exit, kill process group " "{}".format(pid, pgid))
    # 使用给定的进程组ID和SIGKILL信号来杀死整个进程组
    os.killpg(pgid, signal.SIGKILL)
# 设置信号处理程序，用于处理进程信号
def set_signal_handlers():
    # 获取当前进程的进程 ID
    pid = os.getpid()
    try:
        # 获取当前进程的进程组 ID
        pgid = os.getpgid(pid)
    except AttributeError:
        # 如果 `os.getpgid` 不可用，则不设置信号处理程序，因为无法进行安全清理
        pass
    else:
        # XXX: `term_mp` 杀死进程组中的所有进程，在某些情况下包括当前进程的父进程，可能导致意外结果。
        # 为了解决这个问题，只有当当前进程是进程组的组长时才设置信号处理程序。
        # 未来，最好考虑仅杀死当前进程的后代。
        if pid == pgid:
            # 支持使用 ctrl+c 退出
            signal.signal(signal.SIGINT, term_mp)
            signal.signal(signal.SIGTERM, term_mp)


# 构建数据加载器
def build_dataloader(config, mode, device, logger, seed=None):
    # 深拷贝配置
    config = copy.deepcopy(config)

    # 支持的数据集类别列表
    support_dict = [
        'SimpleDataSet',
        'LMDBDataSet',
        'PGDataSet',
        'PubTabDataSet',
        'LMDBDataSetSR',
        'LMDBDataSetTableMaster',
        'MultiScaleDataSet',
        'TextDetDataset',
        'TextRecDataset',
        'MSTextRecDataset',
        'PubTabTableRecDataset',
        'KieDataset',
    ]
    # 获取数据集模块名
    module_name = config[mode]['dataset']['name']
    # 断言数据集模块名在支持列表中
    assert module_name in support_dict, Exception(
        'DataSet only support {}'.format(support_dict))
    # 断言模式为训练、评估或测试
    assert mode in ['Train', 'Eval', 'Test'
                    ], "Mode should be Train, Eval or Test."

    # 根据数据集模块名创建数据集对象
    dataset = eval(module_name)(config, mode, logger, seed)
    # 获取数据加载器配置
    loader_config = config[mode]['loader']
    # 获取每个卡的批量大小
    batch_size = loader_config['batch_size_per_card']
    # 是否丢弃最后一个不完整的批次
    drop_last = loader_config['drop_last']
    # 是否打乱数据
    shuffle = loader_config['shuffle']
    # 数据加载器的工作进程数
    num_workers = loader_config['num_workers']
    # 如果加载器配置中存在 'use_shared_memory' 键，则使用共享内存，否则默认为 True
    if 'use_shared_memory' in loader_config.keys():
        use_shared_memory = loader_config['use_shared_memory']
    else:
        use_shared_memory = True
    # 如果模式为"Train"
    if mode == "Train":
        # 将数据分发到多个卡上
        if 'sampler' in config[mode]:
            # 获取配置中的采样器信息
            config_sampler = config[mode]['sampler']
            # 获取采样器名称
            sampler_name = config_sampler.pop("name")
            # 根据采样器名称和配置创建批次采样器
            batch_sampler = eval(sampler_name)(dataset, **config_sampler)
        else:
            # 创建分布式批次采样器
            batch_sampler = DistributedBatchSampler(
                dataset=dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                drop_last=drop_last)
    else:
        # 将数据分发到单个卡上
        batch_sampler = BatchSampler(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last)

    # 如果加载器配置中包含'collate_fn'
    if 'collate_fn' in loader_config:
        # 导入collate_fn模块
        from . import collate_fn
        # 获取collate_fn函数
        collate_fn = getattr(collate_fn, loader_config['collate_fn'])()
    else:
        # 如果没有指定collate_fn，则设为None
        collate_fn = None
    # 创建数据加载器
    data_loader = DataLoader(
        dataset=dataset,
        batch_sampler=batch_sampler,
        places=device,
        num_workers=num_workers,
        return_list=True,
        use_shared_memory=use_shared_memory,
        collate_fn=collate_fn)

    # 返回数据加载器
    return data_loader
```