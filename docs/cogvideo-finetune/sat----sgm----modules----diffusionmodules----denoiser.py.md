# `.\cogvideo-finetune\sat\sgm\modules\diffusionmodules\denoiser.py`

```py
# 从 typing 模块导入字典和联合类型的支持
from typing import Dict, Union

# 导入 PyTorch 库
import torch
import torch.nn as nn

# 从上级模块的 util 中导入 append_dims 和 instantiate_from_config 函数
from ...util import append_dims, instantiate_from_config


# 定义去噪器类，继承自 nn.Module
class Denoiser(nn.Module):
    # 初始化方法，接受权重配置和缩放配置
    def __init__(self, weighting_config, scaling_config):
        # 调用父类的初始化方法
        super().__init__()

        # 根据权重配置实例化权重处理对象
        self.weighting = instantiate_from_config(weighting_config)
        # 根据缩放配置实例化缩放处理对象
        self.scaling = instantiate_from_config(scaling_config)

    # 可能对 sigma 进行量化的函数，当前仅返回原值
    def possibly_quantize_sigma(self, sigma):
        return sigma

    # 可能对 c_noise 进行量化的函数，当前仅返回原值
    def possibly_quantize_c_noise(self, c_noise):
        return c_noise

    # 计算权重的函数，返回处理后的 sigma
    def w(self, sigma):
        return self.weighting(sigma)

    # 前向传播方法，定义了模型的计算过程
    def forward(
        self,
        network: nn.Module,  # 网络模块
        input: torch.Tensor,  # 输入张量
        sigma: torch.Tensor,  # sigma 张量
        cond: Dict,  # 条件字典
        **additional_model_inputs,  # 其他模型输入参数
    ) -> torch.Tensor:  # 返回一个张量
        # 量化 sigma
        sigma = self.possibly_quantize_sigma(sigma)
        # 获取 sigma 的形状
        sigma_shape = sigma.shape
        # 将 sigma 的维度扩展到输入的维度
        sigma = append_dims(sigma, input.ndim)
        # 通过缩放处理得到多个输出
        c_skip, c_out, c_in, c_noise = self.scaling(sigma, **additional_model_inputs)
        # 量化 c_noise，并恢复原来的形状
        c_noise = self.possibly_quantize_c_noise(c_noise.reshape(sigma_shape))
        # 返回通过网络处理的结果，加上输入与 c_skip 的加权和
        return network(input * c_in, c_noise, cond, **additional_model_inputs) * c_out + input * c_skip


# 定义离散去噪器类，继承自 Denoiser
class DiscreteDenoiser(Denoiser):
    # 初始化方法，接受多个参数进行配置
    def __init__(
        self,
        weighting_config,  # 权重配置
        scaling_config,  # 缩放配置
        num_idx,  # 索引数量
        discretization_config,  # 离散化配置
        do_append_zero=False,  # 是否添加零
        quantize_c_noise=True,  # 是否量化 c_noise
        flip=True,  # 是否翻转
    ):
        # 调用父类的初始化方法
        super().__init__(weighting_config, scaling_config)
        # 根据离散化配置实例化 sigma 对象
        sigmas = instantiate_from_config(discretization_config)(num_idx, do_append_zero=do_append_zero, flip=flip)
        # 保存 sigma 对象
        self.sigmas = sigmas
        # self.register_buffer("sigmas", sigmas)  # 可选，注册为缓冲区
        # 保存是否量化 c_noise 的配置
        self.quantize_c_noise = quantize_c_noise

    # 将 sigma 转换为索引的函数
    def sigma_to_idx(self, sigma):
        # 计算 sigma 与每个 sigma 值的距离
        dists = sigma - self.sigmas.to(sigma.device)[:, None]
        # 返回距离最小的索引
        return dists.abs().argmin(dim=0).view(sigma.shape)

    # 将索引转换为 sigma 的函数
    def idx_to_sigma(self, idx):
        # 根据索引返回对应的 sigma 值
        return self.sigmas.to(idx.device)[idx]

    # 可能对 sigma 进行量化的函数
    def possibly_quantize_sigma(self, sigma):
        # 通过索引转换函数进行量化
        return self.idx_to_sigma(self.sigma_to_idx(sigma))

    # 可能对 c_noise 进行量化的函数
    def possibly_quantize_c_noise(self, c_noise):
        # 如果配置为量化，则返回 c_noise 的索引
        if self.quantize_c_noise:
            return self.sigma_to_idx(c_noise)
        else:
            # 否则返回原值
            return c_noise
```