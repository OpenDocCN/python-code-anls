# `.\cogview3-finetune\sat\sgm\modules\diffusionmodules\denoiser.py`

```py
# 导入所需的类型定义，便于后续使用
from typing import Dict, Union

# 导入 PyTorch 及其神经网络模块
import torch
import torch.nn as nn

# 从上层目录导入实用工具函数
from ...util import append_dims, instantiate_from_config


# 定义去噪器类，继承自 nn.Module
class Denoiser(nn.Module):
    # 初始化方法，接受加权配置和缩放配置
    def __init__(self, weighting_config, scaling_config):
        # 调用父类构造函数
        super().__init__()

        # 根据加权配置实例化加权模块
        self.weighting = instantiate_from_config(weighting_config)
        # 根据缩放配置实例化缩放模块
        self.scaling = instantiate_from_config(scaling_config)

    # 可能对 sigma 进行量化的占位符方法
    def possibly_quantize_sigma(self, sigma):
        return sigma  # 返回未修改的 sigma

    # 可能对噪声 c_noise 进行量化的占位符方法
    def possibly_quantize_c_noise(self, c_noise):
        return c_noise  # 返回未修改的 c_noise

    # 计算加权后的 sigma 值
    def w(self, sigma):
        return self.weighting(sigma)  # 返回加权后的 sigma

    # 前向传播方法，定义模型如何处理输入
    def forward(
        self,
        network: nn.Module,  # 网络模型
        input: torch.Tensor,  # 输入张量
        sigma: torch.Tensor,  # sigma 张量
        cond: Dict,  # 条件字典
        **additional_model_inputs,  # 其他模型输入
    ) -> torch.Tensor:
        # 可能对 sigma 进行量化
        sigma = self.possibly_quantize_sigma(sigma)
        # 获取 sigma 的形状
        sigma_shape = sigma.shape
        # 调整 sigma 的维度，以匹配输入的维度
        sigma = append_dims(sigma, input.ndim)
        # 使用缩放模块处理 sigma 和额外模型输入，获取多个输出
        c_skip, c_out, c_in, c_noise = self.scaling(sigma, **additional_model_inputs)
        # 可能对噪声 c_noise 进行量化，并调整形状
        c_noise = self.possibly_quantize_c_noise(c_noise.reshape(sigma_shape))
        # 返回经过网络处理的结果，结合输入和噪声
        return (
            network(input * c_in, c_noise, cond, **additional_model_inputs) * c_out
            + input * c_skip
        )


# 定义离散去噪器类，继承自 Denoiser
class DiscreteDenoiser(Denoiser):
    # 初始化方法，接受多个配置参数
    def __init__(
        self,
        weighting_config,  # 加权配置
        scaling_config,  # 缩放配置
        num_idx,  # 索引数量
        discretization_config,  # 离散化配置
        do_append_zero=False,  # 是否附加零
        quantize_c_noise=True,  # 是否量化 c_noise
        flip=True,  # 是否翻转
    ):
        # 调用父类构造函数
        super().__init__(weighting_config, scaling_config)
        # 根据离散化配置实例化 sigma
        sigmas = instantiate_from_config(discretization_config)(
            num_idx, do_append_zero=do_append_zero, flip=flip
        )
        # 保存 sigma 值
        self.sigmas = sigmas
        # self.register_buffer("sigmas", sigmas)  # （可选）注册一个持久化的缓冲区
        # 设置是否量化 c_noise
        self.quantize_c_noise = quantize_c_noise

    # 将 sigma 转换为索引的方法
    def sigma_to_idx(self, sigma):
        # 计算 sigma 与 sigma 列表的距离
        dists = sigma - self.sigmas.to(sigma.device)[:, None]
        # 返回距离最小的索引
        return dists.abs().argmin(dim=0).view(sigma.shape)

    # 将索引转换为 sigma 的方法
    def idx_to_sigma(self, idx):
        return self.sigmas.to(idx.device)[idx]  # 根据索引返回对应的 sigma

    # 可能对 sigma 进行量化的重写方法
    def possibly_quantize_sigma(self, sigma):
        return self.idx_to_sigma(self.sigma_to_idx(sigma))  # 返回量化后的 sigma

    # 可能对噪声 c_noise 进行量化的重写方法
    def possibly_quantize_c_noise(self, c_noise):
        if self.quantize_c_noise:  # 如果选择量化 c_noise
            return self.sigma_to_idx(c_noise)  # 返回量化后的索引
        else:
            return c_noise  # 返回未修改的 c_noise
```