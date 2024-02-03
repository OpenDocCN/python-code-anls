# `stable-diffusion-webui\modules\textual_inversion\dataset.py`

```
# 导入必要的库
import os
import numpy as np
import PIL
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision import transforms
from collections import defaultdict
from random import shuffle, choices

import random
import tqdm
from modules import devices, shared
import re

# 定义正则表达式，用于匹配以数字或负号开头的字符串
re_numbers_at_start = re.compile(r"^[-\d]+\s*")

# 定义数据集条目类
class DatasetEntry:
    def __init__(self, filename=None, filename_text=None, latent_dist=None, latent_sample=None, cond=None, cond_text=None, pixel_values=None, weight=None):
        self.filename = filename
        self.filename_text = filename_text
        self.weight = weight
        self.latent_dist = latent_dist
        self.latent_sample = latent_sample
        self.cond = cond
        self.cond_text = cond_text
        self.pixel_values = pixel_values

# 定义个性化基类数据集
class PersonalizedBase(Dataset):
    # 创建文本方法
    def create_text(self, filename_text):
        # 从文本列表中随机选择一条文本
        text = random.choice(self.lines)
        # 将文件名文本按逗号分隔成标签列表
        tags = filename_text.split(',')
        # 如果标签丢弃率不为0，则根据丢弃率随机丢弃标签
        if self.tag_drop_out != 0:
            tags = [t for t in tags if random.random() > self.tag_drop_out]
        # 如果需要打乱标签顺序，则打乱标签列表
        if self.shuffle_tags:
            random.shuffle(tags)
        # 替换文本中的占位符为标签
        text = text.replace("[filewords]", ','.join(tags))
        text = text.replace("[name]", self.placeholder_token)
        return text

    # 返回数据集长度
    def __len__(self):
        return self.length

    # 获取数据集中的单个条目
    def __getitem__(self, i):
        entry = self.dataset[i]
        # 如果标签丢弃率不为0或需要打乱标签顺序，则重新生成条件文本
        if self.tag_drop_out != 0 or self.shuffle_tags:
            entry.cond_text = self.create_text(entry.filename_text)
        # 如果潜变量采样方法为随机，则获取第一阶段编码的潜变量样本
        if self.latent_sampling_method == "random":
            entry.latent_sample = shared.sd_model.get_first_stage_encoding(entry.latent_dist).to(devices.cpu)
        return entry

# 定义分组批次采样器类
class GroupedBatchSampler(Sampler):
    # 初始化方法，接受数据源和批处理大小作为参数
    def __init__(self, data_source: PersonalizedBase, batch_size: int):
        # 调用父类的初始化方法
        super().__init__(data_source)

        # 获取数据源的长度
        n = len(data_source)
        # 将数据源的分组信息赋值给当前对象的groups属性
        self.groups = data_source.groups
        # 计算批次数量
        self.len = n_batch = n // batch_size
        # 计算每个分组期望的批次数量
        expected = [len(g) / n * n_batch * batch_size for g in data_source.groups]
        # 计算每个分组的基础批次数量
        self.base = [int(e) // batch_size for e in expected]
        # 计算随机批次的数量
        self.n_rand_batches = nrb = n_batch - sum(self.base)
        # 计算每个分组的概率
        self.probs = [e%batch_size/nrb/batch_size if nrb>0 else 0 for e in expected]
        # 设置批处理大小
        self.batch_size = batch_size

    # 返回批次的数量
    def __len__(self):
        return self.len

    # 迭代器方法
    def __iter__(self):
        # 获取批处理大小
        b = self.batch_size

        # 对每个分组进行随机打乱
        for g in self.groups:
            shuffle(g)

        # 生成批次
        batches = []
        for g in self.groups:
            batches.extend(g[i*b:(i+1)*b] for i in range(len(g) // b))
        # 添加随机批次
        for _ in range(self.n_rand_batches):
            rand_group = choices(self.groups, self.probs)[0]
            batches.append(choices(rand_group, k=b))

        # 打乱所有批次
        shuffle(batches)

        # 生成器，返回每个批次
        yield from batches
class PersonalizedDataLoader(DataLoader):
    # 定义个性化数据加载器类，继承自DataLoader类
    def __init__(self, dataset, latent_sampling_method="once", batch_size=1, pin_memory=False):
        # 初始化方法，接受数据集、潜在采样方法、批处理大小和是否使用pin_memory作为参数
        super(PersonalizedDataLoader, self).__init__(dataset, batch_sampler=GroupedBatchSampler(dataset, batch_size), pin_memory=pin_memory)
        # 调用父类DataLoader的初始化方法，传入数据集、批处理采样器和是否使用pin_memory
        if latent_sampling_method == "random":
            # 如果潜在采样方法为"random"
            self.collate_fn = collate_wrapper_random
            # 设置collate_fn为collate_wrapper_random函数
        else:
            # 否则
            self.collate_fn = collate_wrapper
            # 设置collate_fn为collate_wrapper函数

class BatchLoader:
    # 定义批处理加载器类
    def __init__(self, data):
        # 初始化方法，接受数据作为参数
        self.cond_text = [entry.cond_text for entry in data]
        # 将数据中每个条目的条件文本存储在cond_text中
        self.cond = [entry.cond for entry in data]
        # 将数据中每个条目的条件存储在cond中
        self.latent_sample = torch.stack([entry.latent_sample for entry in data]).squeeze(1)
        # 将数据中每个条目的潜在样本堆叠成张量，并在第一维上压缩为1
        if all(entry.weight is not None for entry in data):
            # 如果所有数据条目的权重都不为None
            self.weight = torch.stack([entry.weight for entry in data]).squeeze(1)
            # 将数据中每个条目的权重堆叠成张量，并在第一维上压缩为1
        else:
            # 否则
            self.weight = None
            # 将权重设置为None
        #self.emb_index = [entry.emb_index for entry in data]
        #print(self.latent_sample.device)

    def pin_memory(self):
        # 定义pin_memory方法
        self.latent_sample = self.latent_sample.pin_memory()
        # 将潜在样本存储在内存中
        return self

def collate_wrapper(batch):
    # 定义collate_wrapper函数，接受批处理作为参数
    return BatchLoader(batch)
    # 返回BatchLoader对象

class BatchLoaderRandom(BatchLoader):
    # 定义BatchLoaderRandom类，继承自BatchLoader类
    def __init__(self, data):
        # 初始化方法，接受数据作为参数
        super().__init__(data)
        # 调用父类BatchLoader的初始化方法，传入数据

    def pin_memory(self):
        # 定义pin_memory方法
        return self
        # 返回自身对象

def collate_wrapper_random(batch):
    # 定义collate_wrapper_random函数，接受批处理作为参数
    return BatchLoaderRandom(batch)
    # 返回BatchLoaderRandom对象
```