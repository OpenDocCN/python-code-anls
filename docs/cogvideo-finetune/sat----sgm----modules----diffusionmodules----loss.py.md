# `.\cogvideo-finetune\sat\sgm\modules\diffusionmodules\loss.py`

```py
# 从类型提示模块导入相关类型
from typing import List, Optional, Union

# 导入 PyTorch 库及其神经网络模块
import torch
import torch.nn as nn
# 导入 ListConfig 用于配置管理
from omegaconf import ListConfig
# 从工具模块导入辅助函数
from ...util import append_dims, instantiate_from_config
# 从 LPIPS 模块导入损失计算
from ...modules.autoencoding.lpips.loss.lpips import LPIPS
# 导入模型并行模块
from sat import mpu


# 定义标准扩散损失类，继承自 nn.Module
class StandardDiffusionLoss(nn.Module):
    # 初始化方法，设置损失类型和其他参数
    def __init__(
        self,
        sigma_sampler_config,
        type="l2",
        offset_noise_level=0.0,
        batch2model_keys: Optional[Union[str, List[str], ListConfig]] = None,
    ):
        # 调用父类初始化方法
        super().__init__()

        # 确保损失类型是合法的
        assert type in ["l2", "l1", "lpips"]

        # 从配置中实例化 sigma 采样器
        self.sigma_sampler = instantiate_from_config(sigma_sampler_config)

        # 保存损失类型和偏移噪声水平
        self.type = type
        self.offset_noise_level = offset_noise_level

        # 如果损失类型是 lpips，则初始化 LPIPS 实例并设置为评估模式
        if type == "lpips":
            self.lpips = LPIPS().eval()

        # 如果没有提供 batch2model_keys，初始化为空列表
        if not batch2model_keys:
            batch2model_keys = []

        # 如果 batch2model_keys 是字符串，则转换为列表
        if isinstance(batch2model_keys, str):
            batch2model_keys = [batch2model_keys]

        # 将 batch2model_keys 转换为集合以去重
        self.batch2model_keys = set(batch2model_keys)

    # 定义调用方法，计算损失
    def __call__(self, network, denoiser, conditioner, input, batch):
        # 使用调节器处理批次数据
        cond = conditioner(batch)
        # 从批次中提取额外的模型输入
        additional_model_inputs = {key: batch[key] for key in self.batch2model_keys.intersection(batch)}

        # 生成 sigma 值并移至输入设备
        sigmas = self.sigma_sampler(input.shape[0]).to(input.device)
        # 生成与输入形状相同的随机噪声
        noise = torch.randn_like(input)
        # 如果偏移噪声水平大于零，则调整噪声
        if self.offset_noise_level > 0.0:
            noise = (
                noise + append_dims(torch.randn(input.shape[0]).to(input.device), input.ndim) * self.offset_noise_level
            )
            # 确保噪声数据类型与输入一致
            noise = noise.to(input.dtype)
        # 计算加噪输入
        noised_input = input.float() + noise * append_dims(sigmas, input.ndim)
        # 使用去噪器处理加噪输入并生成模型输出
        model_output = denoiser(network, noised_input, sigmas, cond, **additional_model_inputs)
        # 获取权重并调整维度
        w = append_dims(denoiser.w(sigmas), input.ndim)
        # 返回计算的损失
        return self.get_loss(model_output, input, w)

    # 定义获取损失的方法
    def get_loss(self, model_output, target, w):
        # 根据损失类型计算不同类型的损失
        if self.type == "l2":
            return torch.mean((w * (model_output - target) ** 2).reshape(target.shape[0], -1), 1)
        elif self.type == "l1":
            return torch.mean((w * (model_output - target).abs()).reshape(target.shape[0], -1), 1)
        elif self.type == "lpips":
            # 计算 LPIPS 损失并调整维度
            loss = self.lpips(model_output, target).reshape(-1)
            return loss


# 定义视频扩散损失类，继承自标准扩散损失类
class VideoDiffusionLoss(StandardDiffusionLoss):
    # 初始化方法，设置视频相关参数
    def __init__(self, block_scale=None, block_size=None, min_snr_value=None, fixed_frames=0, **kwargs):
        # 保存固定帧数量
        self.fixed_frames = fixed_frames
        # 保存块缩放因子
        self.block_scale = block_scale
        # 保存最小信噪比值
        self.block_size = block_size
        # 保存最小信噪比值
        self.min_snr_value = min_snr_value
        # 调用父类初始化方法
        super().__init__(**kwargs)
    # 定义一个可调用对象，接收网络、去噪器、调节器、输入和批处理作为参数
    def __call__(self, network, denoiser, conditioner, input, batch):
        # 使用调节器对批处理进行处理，生成条件
        cond = conditioner(batch)
        # 从批处理中过滤出与模型输入相关的额外输入
        additional_model_inputs = {key: batch[key] for key in self.batch2model_keys.intersection(batch)}

        # 获取累积的 alpha 的平方根及其索引
        alphas_cumprod_sqrt, idx = self.sigma_sampler(input.shape[0], return_idx=True)
        # 将 alpha 的平方根移动到输入的设备上
        alphas_cumprod_sqrt = alphas_cumprod_sqrt.to(input.device)
        # 将索引移动到输入的设备上
        idx = idx.to(input.device)

        # 生成与输入形状相同的随机噪声
        noise = torch.randn_like(input)

        # 广播噪声
        mp_size = mpu.get_model_parallel_world_size()  # 获取模型并行世界的大小
        global_rank = torch.distributed.get_rank() // mp_size  # 计算全局排名
        src = global_rank * mp_size  # 计算源节点
        # 广播索引到所有相关节点
        torch.distributed.broadcast(idx, src=src, group=mpu.get_model_parallel_group())
        # 广播噪声到所有相关节点
        torch.distributed.broadcast(noise, src=src, group=mpu.get_model_parallel_group())
        # 广播 alpha 的平方根到所有相关节点
        torch.distributed.broadcast(alphas_cumprod_sqrt, src=src, group=mpu.get_model_parallel_group())

        # 将索引添加到额外的模型输入中
        additional_model_inputs["idx"] = idx

        # 如果偏移噪声级别大于 0，则调整噪声
        if self.offset_noise_level > 0.0:
            noise = (
                noise + append_dims(torch.randn(input.shape[0]).to(input.device), input.ndim) * self.offset_noise_level
            )

        # 计算带噪声的输入
        noised_input = input.float() * append_dims(alphas_cumprod_sqrt, input.ndim) + noise * append_dims(
            (1 - alphas_cumprod_sqrt**2) ** 0.5, input.ndim
        )

        # 如果批处理包含拼接图像，则将其添加到条件中
        if "concat_images" in batch.keys():
            cond["concat"] = batch["concat_images"]

        # 调用去噪器处理噪声输入，并获取模型输出
        model_output = denoiser(network, noised_input, alphas_cumprod_sqrt, cond, **additional_model_inputs)
        # 计算加权值（v-pred）
        w = append_dims(1 / (1 - alphas_cumprod_sqrt**2), input.ndim)  

        # 如果设置了最小信噪比值，则取加权值与最小信噪比值的最小值
        if self.min_snr_value is not None:
            w = min(w, self.min_snr_value)
        # 返回损失值
        return self.get_loss(model_output, input, w)

    # 定义获取损失的函数，接收模型输出、目标和权重
    def get_loss(self, model_output, target, w):
        # 如果损失类型为 L2，则计算 L2 损失
        if self.type == "l2":
            return torch.mean((w * (model_output - target) ** 2).reshape(target.shape[0], -1), 1)
        # 如果损失类型为 L1，则计算 L1 损失
        elif self.type == "l1":
            return torch.mean((w * (model_output - target).abs()).reshape(target.shape[0], -1), 1)
        # 如果损失类型为 LPIPS，则计算 LPIPS 损失
        elif self.type == "lpips":
            loss = self.lpips(model_output, target).reshape(-1)  # 计算 LPIPS 损失并调整形状
            return loss
```