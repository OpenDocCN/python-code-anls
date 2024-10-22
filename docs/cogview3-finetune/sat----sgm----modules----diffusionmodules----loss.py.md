# `.\cogview3-finetune\sat\sgm\modules\diffusionmodules\loss.py`

```py
# 导入所需的标准库和类型提示
import os
import copy
from typing import List, Optional, Union

# 导入 NumPy 和 PyTorch 库
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# 导入 OmegaConf 中的 ListConfig
from omegaconf import ListConfig

# 从自定义模块中导入所需的函数和类
from ...util import append_dims, instantiate_from_config
from ...modules.autoencoding.lpips.loss.lpips import LPIPS
from ...modules.diffusionmodules.wrappers import OPENAIUNETWRAPPER
from ...util import get_obj_from_str, default
from ...modules.diffusionmodules.discretizer import generate_roughly_equally_spaced_steps, sub_generate_roughly_equally_spaced_steps


# 定义标准扩散损失类，继承自 nn.Module
class StandardDiffusionLoss(nn.Module):
    # 初始化方法，设置损失类型和噪声级别等参数
    def __init__(
        self,
        sigma_sampler_config,
        type="l2",
        offset_noise_level=0.0,
        batch2model_keys: Optional[Union[str, List[str], ListConfig]] = None,
    ):
        super().__init__()

        # 确保损失类型有效
        assert type in ["l2", "l1", "lpips"]

        # 根据配置实例化 sigma 采样器
        self.sigma_sampler = instantiate_from_config(sigma_sampler_config)

        # 保存损失类型和噪声级别
        self.type = type
        self.offset_noise_level = offset_noise_level

        # 如果损失类型为 lpips，则初始化 lpips 模块
        if type == "lpips":
            self.lpips = LPIPS().eval()

        # 如果没有提供 batch2model_keys，则设置为空列表
        if not batch2model_keys:
            batch2model_keys = []

        # 如果 batch2model_keys 是字符串，则转为列表
        if isinstance(batch2model_keys, str):
            batch2model_keys = [batch2model_keys]

        # 将 batch2model_keys 转为集合以便于后续处理
        self.batch2model_keys = set(batch2model_keys)

    # 定义调用方法，计算损失
    def __call__(self, network, denoiser, conditioner, input, batch):
        # 使用条件器处理输入批次
        cond = conditioner(batch)
        # 从批次中提取附加模型输入
        additional_model_inputs = {
            key: batch[key] for key in self.batch2model_keys.intersection(batch)
        }

        # 生成 sigma 值
        sigmas = self.sigma_sampler(input.shape[0]).to(input.device)
        # 生成与输入相同形状的随机噪声
        noise = torch.randn_like(input)
        # 如果设置了噪声级别，调整噪声
        if self.offset_noise_level > 0.0:
            noise = noise + append_dims(
                torch.randn(input.shape[0]).to(input.device), input.ndim
            ) * self.offset_noise_level
            # 确保噪声数据类型与输入一致
            noise = noise.to(input.dtype)
        # 将输入与噪声和 sigma 结合，生成有噪声的输入
        noised_input = input.float() + noise * append_dims(sigmas, input.ndim)
        # 使用去噪网络处理有噪声的输入
        model_output = denoiser(
            network, noised_input, sigmas, cond, **additional_model_inputs
        )
        # 将去噪网络的权重调整为与输入相同的维度
        w = append_dims(denoiser.w(sigmas), input.ndim)
        # 返回损失值
        return self.get_loss(model_output, input, w)

    # 定义计算损失的方法
    def get_loss(self, model_output, target, w):
        # 根据损失类型计算 l2 损失
        if self.type == "l2":
            return torch.mean(
                (w * (model_output - target) ** 2).reshape(target.shape[0], -1), 1
            )
        # 根据损失类型计算 l1 损失
        elif self.type == "l1":
            return torch.mean(
                (w * (model_output - target).abs()).reshape(target.shape[0], -1), 1
            )
        # 根据损失类型计算 lpips 损失
        elif self.type == "lpips":
            loss = self.lpips(model_output, target).reshape(-1)
            return loss


# 定义线性中继扩散损失类，继承自 StandardDiffusionLoss
class LinearRelayDiffusionLoss(StandardDiffusionLoss):
    # 初始化方法，设置相关参数
    def __init__(
        self,
        sigma_sampler_config,
        type="l2",
        offset_noise_level=0.0,
        partial_num_steps=500,
        blurring_schedule='linear',
        batch2model_keys: Optional[Union[str, List[str], ListConfig]] = None,
    ):
        # 调用父类构造函数，初始化基本参数
        super().__init__(
            sigma_sampler_config,  # sigma 采样器的配置
            type=type,  # 类型参数
            offset_noise_level=offset_noise_level,  # 偏移噪声水平
            batch2model_keys=batch2model_keys,  # 批次到模型的键映射
        )

        # 设置模糊调度参数
        self.blurring_schedule = blurring_schedule
        # 设置部分步骤数量
        self.partial_num_steps = partial_num_steps

    
    def __call__(self, network, denoiser, conditioner, input, batch):
        # 使用调节器处理批次数据，生成条件
        cond = conditioner(batch)
        # 生成额外的模型输入，筛选出与模型键对应的批次数据
        additional_model_inputs = {
            key: batch[key] for key in self.batch2model_keys.intersection(batch)
        }
        # 从批次中获取低分辨率输入
        lr_input = batch["lr_input"]

        # 生成随机整数，用于选择部分步骤
        rand = torch.randint(0, self.partial_num_steps, (input.shape[0],))
        # 从 sigma 采样器生成 sigma 值，并转换为输入数据类型和设备
        sigmas = self.sigma_sampler(input.shape[0], rand).to(input.dtype).to(input.device)
        # 生成与输入形状相同的随机噪声
        noise = torch.randn_like(input)
        # 如果偏移噪声水平大于0，则添加额外噪声
        if self.offset_noise_level > 0.0:
            # 生成额外随机噪声并调整其维度，乘以偏移噪声水平
            noise = noise + append_dims(
                torch.randn(input.shape[0]).to(input.device), input.ndim
            ) * self.offset_noise_level
            # 转换噪声为输入数据类型
            noise = noise.to(input.dtype)
        # 调整 rand 的维度并转换为输入数据类型和设备
        rand = append_dims(rand, input.ndim).to(input.dtype).to(input.device)
        # 根据模糊调度的不同方式计算模糊输入
        if self.blurring_schedule == 'linear':
            # 线性模糊处理
            blurred_input = input * (1 - rand / self.partial_num_steps) + lr_input * (rand / self.partial_num_steps)
        elif self.blurring_schedule == 'sigma':
            # 使用 sigma 最大值进行模糊处理
            max_sigmas = self.sigma_sampler(input.shape[0], torch.ones(input.shape[0])*self.partial_num_steps).to(input.dtype).to(input.device)
            blurred_input = input * (1 - sigmas / max_sigmas) + lr_input * (sigmas / max_sigmas)
        elif self.blurring_schedule == 'exp':
            # 指数模糊处理
            rand_blurring = (1 - torch.exp(-(torch.sin((rand+1) / self.partial_num_steps * torch.pi / 2)**4))) / (1 - torch.exp(-torch.ones_like(rand)))
            blurred_input = input * (1 - rand_blurring) + lr_input * rand_blurring
        else:
            # 如果模糊调度不被支持，抛出未实现错误
            raise NotImplementedError
        # 将噪声添加到模糊输入中
        noised_input = blurred_input + noise * append_dims(sigmas, input.ndim)
        # 调用去噪声器处理模糊输入，获取模型输出
        model_output = denoiser(
            network, noised_input, sigmas, cond, **additional_model_inputs
        )
        # 调整去噪声器权重的维度
        w = append_dims(denoiser.w(sigmas), input.ndim)
        # 返回模型输出的损失值
        return self.get_loss(model_output, input, w)
# 定义一个名为 ZeroSNRDiffusionLoss 的类，继承自 StandardDiffusionLoss
class ZeroSNRDiffusionLoss(StandardDiffusionLoss):

    # 重载调用方法，接受网络、去噪器、条件、输入和批次作为参数
    def __call__(self, network, denoiser, conditioner, input, batch):
        # 使用条件生成器处理批次，得到条件变量
        cond = conditioner(batch)
        # 从批次中提取与模型键相交的额外输入
        additional_model_inputs = {
            key: batch[key] for key in self.batch2model_keys.intersection(batch)
        }

        # 生成累积的 alpha 值并获取索引
        alphas_cumprod_sqrt, idx = self.sigma_sampler(input.shape[0], return_idx=True)
        # 将 alpha 值移动到输入的设备上
        alphas_cumprod_sqrt = alphas_cumprod_sqrt.to(input.device)
        # 将索引移动到输入的数据类型和设备上
        idx = idx.to(input.dtype).to(input.device)
        # 将索引添加到额外模型输入中
        additional_model_inputs['idx'] = idx

        # 生成与输入形状相同的随机噪声
        noise = torch.randn_like(input)
        # 如果偏移噪声水平大于零，则添加额外噪声
        if self.offset_noise_level > 0.0:
            noise = noise + append_dims(
                # 生成随机噪声并调整维度，乘以偏移噪声水平
                torch.randn(input.shape[0]).to(input.device), input.ndim
            ) * self.offset_noise_level

        # 计算加入噪声的输入
        noised_input = input.float() * append_dims(alphas_cumprod_sqrt, input.ndim) + noise * append_dims((1-alphas_cumprod_sqrt**2)**0.5, input.ndim)
        # 使用去噪器处理带噪声的输入
        model_output = denoiser(
            network, noised_input, alphas_cumprod_sqrt, cond, **additional_model_inputs
        )
        # 计算 v-pred 权重
        w = append_dims(1/(1-alphas_cumprod_sqrt**2), input.ndim) 
        # 返回损失值
        return self.get_loss(model_output, input, w)
    
    # 定义一个获取损失的函数
    def get_loss(self, model_output, target, w):
        # 如果损失类型为 L2，计算 L2 损失
        if self.type == "l2":
            return torch.mean(
                # 计算每个样本的 L2 损失并调整维度
                (w * (model_output - target) ** 2).reshape(target.shape[0], -1), 1
            )
        # 如果损失类型为 L1，计算 L1 损失
        elif self.type == "l1":
            return torch.mean(
                # 计算每个样本的 L1 损失并调整维度
                (w * (model_output - target).abs()).reshape(target.shape[0], -1), 1
            )
        # 如果损失类型为 LPIPS，计算 LPIPS 损失
        elif self.type == "lpips":
            loss = self.lpips(model_output, target).reshape(-1)
            return loss
```