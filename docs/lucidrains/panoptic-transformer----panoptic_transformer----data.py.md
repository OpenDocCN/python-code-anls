# `.\lucidrains\panoptic-transformer\panoptic_transformer\data.py`

```py
# 导入所需的库
from pathlib import Path
from random import choice
from PIL import Image
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
from torchvision import transforms as T

# 定义一个循环生成器函数，用于循环遍历数据集
def cycle(dl):
    while True:
        for el in dl:
            yield el

# 定义 PathfinderXDataset 类，继承自 Dataset 类
class PathfinderXDataset(Dataset):
    def __init__(
        self,
        folder,
        augment = False
    ):
        super().__init__()
        # 获取文件夹中所有的 .npy 文件
        metadata_files = [*Path(folder).glob(f'**/*.npy')]
        # 断言确保找到了至少一个 metadata 文件
        assert len(metadata_files) > 0, 'not able to find more than 1 metadata file'

        # 获取第一个 metadata 文件
        metadata_file = metadata_files[0]
        # 加载 metadata 文件
        metadata = np.load(str(metadata_file))
        # 获取 metadata 文件的父目录
        root_path = metadata_file.parents[1]

        self.augment = augment
        # 将数据集的路径和标签存储为元组的列表
        self.data = [(str(root_path / m[0] / m[1]), int(m[3])) for m in metadata]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, ind):
        # 获取指定索引的路径和标签
        path, label = self.data[ind]
        # 打开图像文件
        img = Image.open(path)

        # 对图像进行数据增强处理
        img = T.Compose([
            T.RandomHorizontalFlip() if self.augment else nn.Identity(),
            T.RandomVerticalFlip() if self.augment else nn.Identity(),
            T.PILToTensor()
        ])(img)

        # 将标签转换为 torch 张量
        label = torch.tensor(label, dtype = torch.float32)

        if self.augment:
            # 随机选择旋转角度
            rand_rotate = [0, 90, 180, 270]
            img = T.functional.rotate(img, choice(rand_rotate))
            # 随机选择填充方式
            rand_padding = [(0, 0, 0, 0), (1, -1, 0, 0), (-1, 1, 0, 0), (0, 0, 1, -1), (0, 0, -1, 1)]
            img = F.pad(img, choice(rand_padding))

        return img.float(), label

# 获取训练和验证数据加载器函数
def get_dataloaders(
    data_path,
    *,
    augment = True,
    frac_valids = 0.05,
    batch_size
):
    # 创建 PathfinderXDataset 实例
    ds = PathfinderXDataset(data_path, augment = augment)

    total_samples = len(ds)
    # 计算验证集样本数量
    num_valid = int(frac_valids * total_samples)
    # 计算训练集样本数量
    num_train = total_samples - num_valid

    print(f'training with {num_train} samples and validating with {num_valid} samples')

    # 随机划分数据集为训练集和验证集
    train_ds, valid_ds = random_split(ds, [num_train, num_valid])

    # 创建训练数据加载器和验证数据加载器
    train_dl = DataLoader(train_ds, batch_size = batch_size, shuffle = True)
    valid_dl = DataLoader(valid_ds, batch_size = batch_size, shuffle = True)

    return cycle(train_dl), cycle(valid_dl)
```