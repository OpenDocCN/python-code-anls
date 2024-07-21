# `.\pytorch\functorch\examples\maml_omniglot\support\omniglot_loaders.py`

```py
# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# These Omniglot loaders are from Jackie Loong's PyTorch MAML implementation:
#     https://github.com/dragen1860/MAML-Pytorch
#     https://github.com/dragen1860/MAML-Pytorch/blob/master/omniglot.py
#     https://github.com/dragen1860/MAML-Pytorch/blob/master/omniglotNShot.py

import errno
import os
import os.path

import numpy as np
import torchvision.transforms as transforms
from PIL import Image

import torch
import torch.utils.data as data


class Omniglot(data.Dataset):
    urls = [
        "https://github.com/brendenlake/omniglot/raw/master/python/images_background.zip",
        "https://github.com/brendenlake/omniglot/raw/master/python/images_evaluation.zip",
    ]
    raw_folder = "raw"
    processed_folder = "processed"
    training_file = "training.pt"
    test_file = "test.pt"

    """
    The items are (filename,category). The index of all the categories can be found in self.idx_classes
    Args:
    - root: the directory where the dataset will be stored
    - transform: how to transform the input
    - target_transform: how to transform the target
    - download: need to download the dataset
    """

    def __init__(self, root, transform=None, target_transform=None, download=False):
        # 初始化函数，设置数据集的根目录，数据转换方法，以及是否下载数据集
        self.root = root
        self.transform = transform
        self.target_transform = target_transform

        # 如果数据集不存在，根据参数决定是否下载数据集或抛出错误
        if not self._check_exists():
            if download:
                self.download()
            else:
                raise RuntimeError(
                    "Dataset not found." + " You can use download=True to download it"
                )

        # 获取数据集中所有的文件及其类别，并索引类别信息
        self.all_items = find_classes(os.path.join(self.root, self.processed_folder))
        self.idx_classes = index_classes(self.all_items)

    def __getitem__(self, index):
        # 根据索引获取数据集中的文件名和类别
        filename = self.all_items[index][0]
        img = str.join("/", [self.all_items[index][2], filename])

        # 根据类别索引获取目标标签
        target = self.idx_classes[self.all_items[index][1]]
        
        # 如果定义了数据转换方法，则对图像进行转换
        if self.transform is not None:
            img = self.transform(img)
        
        # 如果定义了目标转换方法，则对目标进行转换
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        # 返回数据集中样本的数量
        return len(self.all_items)
    # 检查数据集是否已存在于指定路径下，包括两个子文件夹 images_evaluation 和 images_background
    def _check_exists(self):
        return os.path.exists(
            os.path.join(self.root, self.processed_folder, "images_evaluation")
        ) and os.path.exists(
            os.path.join(self.root, self.processed_folder, "images_background")
        )

    # 下载数据集的方法
    def download(self):
        import urllib  # 导入 urllib 库，用于处理 URL 相关操作
        import zipfile  # 导入 zipfile 库，用于处理 ZIP 文件

        # 如果数据集已存在，则直接返回
        if self._check_exists():
            return

        # 创建目录结构，如果目录已存在则忽略，否则抛出 OSError 异常
        try:
            os.makedirs(os.path.join(self.root, self.raw_folder))
            os.makedirs(os.path.join(self.root, self.processed_folder))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        # 遍历下载 URL 列表中的每个 URL
        for url in self.urls:
            print("== Downloading " + url)
            # 打开 URL 并读取数据
            data = urllib.request.urlopen(url)
            # 从 URL 中提取文件名
            filename = url.rpartition("/")[2]
            # 构建文件保存的完整路径
            file_path = os.path.join(self.root, self.raw_folder, filename)
            # 将下载的数据写入到本地文件
            with open(file_path, "wb") as f:
                f.write(data.read())
            # 指定处理后文件的路径
            file_processed = os.path.join(self.root, self.processed_folder)
            # 打印解压信息，并将 ZIP 文件解压到指定路径下
            print("== Unzip from " + file_path + " to " + file_processed)
            zip_ref = zipfile.ZipFile(file_path, "r")
            zip_ref.extractall(file_processed)
            zip_ref.close()
        # 打印下载完成信息
        print("Download finished.")
# 定义函数 `find_classes`，用于在指定根目录下查找符合条件的文件并返回信息列表
def find_classes(root_dir):
    retour = []  # 返回的结果列表
    # 遍历根目录及其子目录下的文件和文件夹
    for root, dirs, files in os.walk(root_dir):
        # 遍历当前目录中的文件
        for f in files:
            # 如果文件以 "png" 结尾，则符合条件
            if f.endswith("png"):
                # 分割当前目录路径，并获取其长度
                r = root.split("/")
                lr = len(r)
                # 将文件名、上级目录名与当前目录路径添加到结果列表中
                retour.append((f, r[lr - 2] + "/" + r[lr - 1], root))
    # 打印找到的项目数量
    print(f"== Found {len(retour)} items ")
    # 返回结果列表
    return retour


# 定义函数 `index_classes`，用于为查找到的项目创建索引，并返回索引字典
def index_classes(items):
    idx = {}  # 索引字典
    # 遍历输入的项目列表
    for i in items:
        # 如果项目的第二个元素不在索引字典中，则将其添加到索引字典中，并分配一个新的索引值
        if i[1] not in idx:
            idx[i[1]] = len(idx)
    # 打印找到的类别数量
    print(f"== Found {len(idx)} classes")
    # 返回索引字典
    return idx


# 定义类 `OmniglotNShot`，用于处理数据集的标准化及批处理操作
class OmniglotNShot:
    def normalization(self):
        """
        Normalizes our data, to have a mean of 0 and sdt of 1
        标准化数据集，使其均值为0，标准差为1
        """
        # 计算训练集的均值、标准差、最大值和最小值
        self.mean = np.mean(self.x_train)
        self.std = np.std(self.x_train)
        self.max = np.max(self.x_train)
        self.min = np.min(self.x_train)
        
        # 对训练集和测试集进行标准化处理
        self.x_train = (self.x_train - self.mean) / self.std
        self.x_test = (self.x_test - self.mean) / self.std
        
        # 重新计算标准化后训练集的均值、标准差、最大值和最小值
        self.mean = np.mean(self.x_train)
        self.std = np.std(self.x_train)
        self.max = np.max(self.x_train)
        self.min = np.min(self.x_train)

    def next(self, mode="train"):
        """
        Gets next batch from the dataset with name.
        从数据集中获取下一个批次数据。
        :param mode: 数据集划分的名称之一（"train", "val", "test"之一）
        :return: 下一个批次的数据
        """
        # 如果索引超过了缓存数据集的长度，则更新缓存
        if self.indexes[mode] >= len(self.datasets_cache[mode]):
            self.indexes[mode] = 0
            self.datasets_cache[mode] = self.load_data_cache(self.datasets[mode])

        # 获取下一个批次的数据
        next_batch = self.datasets_cache[mode][self.indexes[mode]]
        self.indexes[mode] += 1

        return next_batch
```