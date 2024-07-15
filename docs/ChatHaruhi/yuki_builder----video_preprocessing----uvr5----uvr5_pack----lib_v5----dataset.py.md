# `.\Chat-Haruhi-Suzumiya\yuki_builder\video_preprocessing\uvr5\uvr5_pack\lib_v5\dataset.py`

```py
# 导入标准库和第三方库
import os
import random

import numpy as np
import torch.utils.data
from tqdm import tqdm

# 从当前包中导入特定模块
from . import spec_utils

# 定义一个数据集类，用于声音分离任务的验证集
class VocalRemoverValidationSet(torch.utils.data.Dataset):
    def __init__(self, patch_list):
        self.patch_list = patch_list  # 初始化数据集列表

    def __len__(self):
        return len(self.patch_list)  # 返回数据集的长度

    def __getitem__(self, idx):
        path = self.patch_list[idx]  # 获取指定索引位置的数据路径
        data = np.load(path)  # 加载数据文件

        X, y = data["X"], data["y"]  # 从加载的数据中分离出输入和目标

        X_mag = np.abs(X)  # 计算输入数据的幅度谱
        y_mag = np.abs(y)  # 计算目标数据的幅度谱

        return X_mag, y_mag  # 返回幅度谱作为训练样本


# 函数：生成音频文件对的列表
def make_pair(mix_dir, inst_dir):
    input_exts = [".wav", ".m4a", ".mp3", ".mp4", ".flac"]  # 支持的音频文件扩展名列表

    # 从混音文件夹中获取所有符合条件的文件路径
    X_list = sorted(
        [
            os.path.join(mix_dir, fname)
            for fname in os.listdir(mix_dir)
            if os.path.splitext(fname)[1] in input_exts
        ]
    )

    # 从乐器文件夹中获取所有符合条件的文件路径
    y_list = sorted(
        [
            os.path.join(inst_dir, fname)
            for fname in os.listdir(inst_dir)
            if os.path.splitext(fname)[1] in input_exts
        ]
    )

    filelist = list(zip(X_list, y_list))  # 将混音文件路径和乐器文件路径组合成列表

    return filelist  # 返回文件路径对的列表


# 函数：根据指定模式和验证比例划分训练集和验证集
def train_val_split(dataset_dir, split_mode, val_rate, val_filelist):
    if split_mode == "random":
        # 生成混音文件和乐器文件的路径对列表
        filelist = make_pair(
            os.path.join(dataset_dir, "mixtures"),
            os.path.join(dataset_dir, "instruments"),
        )

        random.shuffle(filelist)  # 随机打乱文件路径对列表

        if len(val_filelist) == 0:
            val_size = int(len(filelist) * val_rate)  # 计算验证集大小
            train_filelist = filelist[:-val_size]  # 获取训练集文件路径对列表
            val_filelist = filelist[-val_size:]  # 获取验证集文件路径对列表
        else:
            # 使用给定的验证文件列表划分训练集和验证集
            train_filelist = [
                pair for pair in filelist if list(pair) not in val_filelist
            ]
    elif split_mode == "subdirs":
        if len(val_filelist) != 0:
            raise ValueError(
                "The `val_filelist` option is not available in `subdirs` mode"
            )

        # 从训练子目录中生成训练集文件路径对列表
        train_filelist = make_pair(
            os.path.join(dataset_dir, "training/mixtures"),
            os.path.join(dataset_dir, "training/instruments"),
        )

        # 从验证子目录中生成验证集文件路径对列表
        val_filelist = make_pair(
            os.path.join(dataset_dir, "validation/mixtures"),
            os.path.join(dataset_dir, "validation/instruments"),
        )

    return train_filelist, val_filelist  # 返回训练集文件路径对列表和验证集文件路径对列表


# 函数：数据增强操作，包括数据重排、减少率、混合率、混合指数
def augment(X, y, reduction_rate, reduction_mask, mixup_rate, mixup_alpha):
    perm = np.random.permutation(len(X))  # 生成随机排列的索引
    # 使用 tqdm 函数迭代处理 perm 列表的索引和值
    for i, idx in enumerate(tqdm(perm)):
        # 如果随机数小于减少率 reduction_rate，则调用 reduce_vocal_aggressively 函数处理 y[idx] 数据
        if np.random.uniform() < reduction_rate:
            y[idx] = spec_utils.reduce_vocal_aggressively(
                X[idx], y[idx], reduction_mask
            )

        # 如果随机数小于 0.5，则交换信道（channel）
        if np.random.uniform() < 0.5:
            # swap channel
            X[idx] = X[idx, ::-1]
            y[idx] = y[idx, ::-1]

        # 如果随机数小于 0.02，则对 X[idx] 和 y[idx] 执行均值操作，实现单声道效果
        if np.random.uniform() < 0.02:
            # mono
            X[idx] = X[idx].mean(axis=0, keepdims=True)
            y[idx] = y[idx].mean(axis=0, keepdims=True)

        # 如果随机数小于 0.02，则将 y[idx] 赋值给 X[idx]，实现某种特定的操作（这里的意图依赖于具体的上下文）
        if np.random.uniform() < 0.02:
            # inst
            X[idx] = y[idx]

        # 如果随机数小于 mixup_rate 并且索引 i 小于 perm 列表长度减一，则执行 mixup 操作
        if np.random.uniform() < mixup_rate and i < len(perm) - 1:
            # 生成混合比例 lam，并根据 lam 对 X[idx] 和 y[idx] 进行线性混合
            lam = np.random.beta(mixup_alpha, mixup_alpha)
            X[idx] = lam * X[idx] + (1 - lam) * X[perm[i + 1]]
            y[idx] = lam * y[idx] + (1 - lam) * y[perm[i + 1]]

    # 返回处理后的 X 和 y 数据
    return X, y
# 定义一个函数，用于生成左右填充的尺寸和感兴趣区域的大小
def make_padding(width, cropsize, offset):
    # 左侧填充量等于给定的偏移量
    left = offset
    # 感兴趣区域的大小等于裁剪尺寸减去左右两侧的填充量
    roi_size = cropsize - left * 2
    # 如果感兴趣区域大小为0，则将其设置为裁剪尺寸
    if roi_size == 0:
        roi_size = cropsize
    # 右侧填充量为感兴趣区域大小减去宽度模感兴趣区域大小的余数，再加上左侧填充量
    right = roi_size - (width % roi_size) + left

    return left, right, roi_size


# 定义一个函数，用于生成训练集数据
def make_training_set(filelist, cropsize, patches, sr, hop_length, n_fft, offset):
    # 计算训练集的总长度
    len_dataset = patches * len(filelist)

    # 初始化训练集的输入和输出数据数组，复数64位
    X_dataset = np.zeros((len_dataset, 2, n_fft // 2 + 1, cropsize), dtype=np.complex64)
    y_dataset = np.zeros((len_dataset, 2, n_fft // 2 + 1, cropsize), dtype=np.complex64)

    # 遍历文件列表，并显示进度条
    for i, (X_path, y_path) in enumerate(tqdm(filelist)):
        # 缓存或加载音频的频谱数据
        X, y = spec_utils.cache_or_load(X_path, y_path, sr, hop_length, n_fft)
        # 计算归一化系数
        coef = np.max([np.abs(X).max(), np.abs(y).max()])
        X, y = X / coef, y / coef

        # 获取左右填充量和感兴趣区域的大小
        l, r, roi_size = make_padding(X.shape[2], cropsize, offset)
        
        # 在频谱数据的时间轴上进行常数填充
        X_pad = np.pad(X, ((0, 0), (0, 0), (l, r)), mode="constant")
        y_pad = np.pad(y, ((0, 0), (0, 0), (l, r)), mode="constant")

        # 在填充后的数据上随机选择裁剪区域的起始位置
        starts = np.random.randint(0, X_pad.shape[2] - cropsize, patches)
        ends = starts + cropsize
        for j in range(patches):
            idx = i * patches + j
            # 将裁剪后的数据存入训练集数组中
            X_dataset[idx] = X_pad[:, :, starts[j]:ends[j]]
            y_dataset[idx] = y_pad[:, :, starts[j]:ends[j]]

    # 返回输入和输出的训练集数据数组
    return X_dataset, y_dataset


# 定义一个函数，用于生成验证集数据
def make_validation_set(filelist, cropsize, sr, hop_length, n_fft, offset):
    # 初始化补丁列表
    patch_list = []
    # 根据参数生成保存补丁文件的目录名
    patch_dir = "cs{}_sr{}_hl{}_nf{}_of{}".format(cropsize, sr, hop_length, n_fft, offset)
    # 如果目录不存在则创建它
    os.makedirs(patch_dir, exist_ok=True)

    # 遍历文件列表，并显示进度条
    for i, (X_path, y_path) in enumerate(tqdm(filelist)):
        # 获取音频文件的基本名称（不含路径和扩展名）
        basename = os.path.splitext(os.path.basename(X_path))[0]

        # 缓存或加载音频的频谱数据
        X, y = spec_utils.cache_or_load(X_path, y_path, sr, hop_length, n_fft)
        # 计算归一化系数
        coef = np.max([np.abs(X).max(), np.abs(y).max()])
        X, y = X / coef, y / coef

        # 获取左右填充量和感兴趣区域的大小
        l, r, roi_size = make_padding(X.shape[2], cropsize, offset)
        
        # 在频谱数据的时间轴上进行常数填充
        X_pad = np.pad(X, ((0, 0), (0, 0), (l, r)), mode="constant")
        y_pad = np.pad(y, ((0, 0), (0, 0), (l, r)), mode="constant")

        # 计算数据集的长度（向上取整）
        len_dataset = int(np.ceil(X.shape[2] / roi_size))
        for j in range(len_dataset):
            # 构建保存当前补丁数据的文件路径
            outpath = os.path.join(patch_dir, "{}_p{}.npz".format(basename, j))
            start = j * roi_size
            # 如果文件路径不存在，则保存数据到文件中
            if not os.path.exists(outpath):
                np.savez(outpath, X=X_pad[:, :, start:start + cropsize], y=y_pad[:, :, start:start + cropsize])
            # 将当前补丁文件的路径添加到补丁列表中
            patch_list.append(outpath)

    # 返回一个包含补丁文件路径列表的验证集对象
    return VocalRemoverValidationSet(patch_list)
```