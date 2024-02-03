# `.\PaddleOCR\test_tipc\supplementary\data_loader.py`

```
# 导入必要的库
import numpy as np
from paddle.vision.datasets import Cifar100
from paddle.vision.transforms import Normalize
import signal
import os
from paddle.io import Dataset, DataLoader, DistributedBatchSampler

# 定义信号处理函数，用于终止所有子进程
def term_mp(sig_num, frame):
    """ kill all child processes
    """
    pid = os.getpid()
    pgid = os.getpgid(os.getpid())
    print("main proc {} exit, kill process group " "{}".format(pid, pgid))
    os.killpg(pgid, signal.SIGKILL)
    return

# 构建数据加载器函数
def build_dataloader(mode,
                     batch_size=4,
                     seed=None,
                     num_workers=0,
                     device='gpu:0'):

    # 定义数据标准化处理
    normalize = Normalize(
        mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], data_format='HWC')

    # 根据模式选择数据集
    if mode.lower() == "train":
        dataset = Cifar100(mode=mode, transform=normalize)
    elif mode.lower() in ["test", 'valid', 'eval']:
        dataset = Cifar100(mode="test", transform=normalize)
    else:
        raise ValueError(f"{mode} should be one of ['train', 'test']")

    # 定义批量采样器
    batch_sampler = DistributedBatchSampler(
        dataset=dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    # 创建数据加载器
    data_loader = DataLoader(
        dataset=dataset,
        batch_sampler=batch_sampler,
        places=device,
        num_workers=num_workers,
        return_list=True,
        use_shared_memory=False)

    # 支持使用 Ctrl+C 终止程序
    signal.signal(signal.SIGINT, term_mp)
    signal.signal(signal.SIGTERM, term_mp)

    return data_loader

# 以下代码段被注释掉，不会执行
# cifar100 = Cifar100(mode='train', transform=normalize)
# data = cifar100[0]
# image, label = data
# reader = build_dataloader('train')
# for idx, data in enumerate(reader):
#     print(idx, data[0].shape, data[1].shape)
#     if idx >= 10:
#         break
```