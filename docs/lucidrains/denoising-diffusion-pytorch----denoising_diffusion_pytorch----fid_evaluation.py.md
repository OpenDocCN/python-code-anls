# `.\lucidrains\denoising-diffusion-pytorch\denoising_diffusion_pytorch\fid_evaluation.py`

```
# 导入所需的库
import math
import os

import numpy as np
import torch
from einops import rearrange, repeat
from pytorch_fid.fid_score import calculate_frechet_distance
from pytorch_fid.inception import InceptionV3
from torch.nn.functional import adaptive_avg_pool2d
from tqdm.auto import tqdm

# 定义一个函数，将数字分成若干组
def num_to_groups(num, divisor):
    # 计算可以分成多少组
    groups = num // divisor
    # 计算余数
    remainder = num % divisor
    # 创建一个列表，每个元素为 divisor
    arr = [divisor] * groups
    # 如果余数大于0，则添加一个元素为余数的值
    if remainder > 0:
        arr.append(remainder)
    return arr

# 定义 FID 评估类
class FIDEvaluation:
    def __init__(
        self,
        batch_size,
        dl,
        sampler,
        channels=3,
        accelerator=None,
        stats_dir="./results",
        device="cuda",
        num_fid_samples=50000,
        inception_block_idx=2048,
    ):
        # 初始化 FID 评估类的属性
        self.batch_size = batch_size
        self.n_samples = num_fid_samples
        self.device = device
        self.channels = channels
        self.dl = dl
        self.sampler = sampler
        self.stats_dir = stats_dir
        self.print_fn = print if accelerator is None else accelerator.print
        # 确保 inception_block_idx 在 InceptionV3.BLOCK_INDEX_BY_DIM 中
        assert inception_block_idx in InceptionV3.BLOCK_INDEX_BY_DIM
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[inception_block_idx]
        self.inception_v3 = InceptionV3([block_idx]).to(device)
        self.dataset_stats_loaded = False

    # 计算 Inception 特征
    def calculate_inception_features(self, samples):
        # 如果通道数为1，则将 samples 重复为3通道
        if self.channels == 1:
            samples = repeat(samples, "b 1 ... -> b c ...", c=3)

        self.inception_v3.eval()
        features = self.inception_v3(samples)[0]

        # 如果特征的尺寸不是 (1, 1)，则进行自适应平均池化
        if features.size(2) != 1 or features.size(3) != 1:
            features = adaptive_avg_pool2d(features, output_size=(1, 1))
        features = rearrange(features, "... 1 1 -> ...")
        return features

    # 加载或预计算数据集统计信息
    def load_or_precalc_dataset_stats(self):
        path = os.path.join(self.stats_dir, "dataset_stats")
        try:
            ckpt = np.load(path + ".npz")
            self.m2, self.s2 = ckpt["m2"], ckpt["s2"]
            self.print_fn("Dataset stats loaded from disk.")
            ckpt.close()
        except OSError:
            num_batches = int(math.ceil(self.n_samples / self.batch_size))
            stacked_real_features = []
            self.print_fn(
                f"Stacking Inception features for {self.n_samples} samples from the real dataset."
            )
            for _ in tqdm(range(num_batches)):
                try:
                    real_samples = next(self.dl)
                except StopIteration:
                    break
                real_samples = real_samples.to(self.device)
                real_features = self.calculate_inception_features(real_samples)
                stacked_real_features.append(real_features)
            stacked_real_features = (
                torch.cat(stacked_real_features, dim=0).cpu().numpy()
            )
            m2 = np.mean(stacked_real_features, axis=0)
            s2 = np.cov(stacked_real_features, rowvar=False)
            np.savez_compressed(path, m2=m2, s2=s2)
            self.print_fn(f"Dataset stats cached to {path}.npz for future use.")
            self.m2, self.s2 = m2, s2
        self.dataset_stats_loaded = True

    # 进入推断模式
    @torch.inference_mode()
    # 计算 FID 分数
    def fid_score(self):
        # 如果数据集统计信息未加载，则加载或预先计算数据集统计信息
        if not self.dataset_stats_loaded:
            self.load_or_precalc_dataset_stats()
        # 将采样器设置为评估模式
        self.sampler.eval()
        # 将样本数量分成多个批次
        batches = num_to_groups(self.n_samples, self.batch_size)
        # 初始化一个空列表用于存储生成样本的 Inception 特征
        stacked_fake_features = []
        # 打印信息，指示正在为生成的样本堆叠 Inception 特征
        self.print_fn(
            f"Stacking Inception features for {self.n_samples} generated samples."
        )
        # 遍历每个批次
        for batch in tqdm(batches):
            # 从采样器中获取指定数量的生成样本
            fake_samples = self.sampler.sample(batch_size=batch)
            # 计算生成样本的 Inception 特征
            fake_features = self.calculate_inception_features(fake_samples)
            # 将生成样本的 Inception 特征添加到列表中
            stacked_fake_features.append(fake_features)
        # 将所有生成样本的 Inception 特征在维度0上拼接起来，并转换为 NumPy 数组
        stacked_fake_features = torch.cat(stacked_fake_features, dim=0).cpu().numpy()
        # 计算生成样本的均值和协方差矩阵
        m1 = np.mean(stacked_fake_features, axis=0)
        s1 = np.cov(stacked_fake_features, rowvar=False)

        # 返回计算得到的 FID 分数
        return calculate_frechet_distance(m1, s1, self.m2, self.s2)
```