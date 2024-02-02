# `so-vits-svc\diffusion\data_loaders.py`

```py
# 导入所需的库
import os
import random

import librosa
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

# 导入自定义的工具函数
from utils import repeat_expand_2d

# 定义函数，用于遍历指定目录下的文件
def traverse_dir(
        root_dir,
        extensions,
        amount=None,
        str_include=None,
        str_exclude=None,
        is_pure=False,
        is_sort=False,
        is_ext=True):

    # 初始化文件列表和计数器
    file_list = []
    cnt = 0
    # 遍历指定目录下的所有文件和子目录
    for root, _, files in os.walk(root_dir):
        for file in files:
            # 检查文件扩展名是否在指定的扩展名列表中
            if any([file.endswith(f".{ext}") for ext in extensions]):
                # 构建文件路径
                mix_path = os.path.join(root, file)
                # 如果需要纯净路径，则去除根目录部分
                pure_path = mix_path[len(root_dir)+1:] if is_pure else mix_path

                # 如果指定了数量限制，并且已经达到数量限制，则返回文件列表
                if (amount is not None) and (cnt == amount):
                    if is_sort:
                        file_list.sort()
                    return file_list
                
                # 检查路径中是否包含指定的字符串
                if (str_include is not None) and (str_include not in pure_path):
                    continue
                if (str_exclude is not None) and (str_exclude in pure_path):
                    continue
                
                # 如果不需要文件扩展名，则去除扩展名部分
                if not is_ext:
                    ext = pure_path.split('.')[-1]
                    pure_path = pure_path[:-(len(ext)+1)]
                # 将路径添加到文件列表中，并增加计数器
                file_list.append(pure_path)
                cnt += 1
    # 如果需要排序文件列表，则进行排序
    if is_sort:
        file_list.sort()
    # 返回文件列表
    return file_list

# 定义函数，用于获取数据加载器
def get_data_loaders(args, whole_audio=False):
    # 创建训练数据集对象，传入训练文件列表、音频时长、帧移大小、采样率等参数
    data_train = AudioDataset(
        filelists = args.data.training_files,
        waveform_sec=args.data.duration,
        hop_size=args.data.block_size,
        sample_rate=args.data.sampling_rate,
        load_all_data=args.train.cache_all_data,
        whole_audio=whole_audio,
        extensions=args.data.extensions,
        n_spk=args.model.n_spk,
        spk=args.spk,
        device=args.train.cache_device,
        fp16=args.train.cache_fp16,
        unit_interpolate_mode = args.data.unit_interpolate_mode,
        use_aug=True)
    # 创建训练数据加载器，传入训练数据集对象、批量大小、是否打乱数据、工作线程数等参数
    loader_train = torch.utils.data.DataLoader(
        data_train ,
        batch_size=args.train.batch_size if not whole_audio else 1,
        shuffle=True,
        num_workers=args.train.num_workers if args.train.cache_device=='cpu' else 0,
        persistent_workers=(args.train.num_workers > 0) if args.train.cache_device=='cpu' else False,
        pin_memory=True if args.train.cache_device=='cpu' else False
    )
    # 创建验证数据集对象，传入验证文件列表、音频时长、帧移大小、采样率等参数
    data_valid = AudioDataset(
        filelists = args.data.validation_files,
        waveform_sec=args.data.duration,
        hop_size=args.data.block_size,
        sample_rate=args.data.sampling_rate,
        load_all_data=args.train.cache_all_data,
        whole_audio=True,
        spk=args.spk,
        extensions=args.data.extensions,
        unit_interpolate_mode = args.data.unit_interpolate_mode,
        n_spk=args.model.n_spk)
    # 创建验证数据加载器，传入验证数据集对象、批量大小、是否打乱数据、工作线程数等参数
    loader_valid = torch.utils.data.DataLoader(
        data_valid,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    # 返回训练数据加载器和验证数据加载器
    return loader_train, loader_valid 
# 定义一个名为 AudioDataset 的类，继承自 Dataset 类
class AudioDataset(Dataset):
    # 初始化函数，接受多个参数
    def __init__(
        self,
        filelists,
        waveform_sec,
        hop_size,
        sample_rate,
        spk,
        load_all_data=True,
        whole_audio=False,
        extensions=['wav'],
        n_spk=1,
        device='cpu',
        fp16=False,
        use_aug=False,
        unit_interpolate_mode = 'left'
    # 获取指定索引的数据项
    def __getitem__(self, file_idx):
        # 获取文件路径及扩展名
        name_ext = self.paths[file_idx]
        # 获取数据缓冲区
        data_buffer = self.data_buffer[name_ext]
        # 检查音频时长，如果太短则跳过
        if data_buffer['duration'] < (self.waveform_sec + 0.1):
            # 返回下一个索引的数据项
            return self.__getitem__( (file_idx + 1) % len(self.paths))
        # 获取数据项
        return self.get_data(name_ext, data_buffer)

    # 返回数据集的长度
    def __len__(self):
        return len(self.paths)
```