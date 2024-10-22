# `.\diffusers\schedulers\scheduling_amused.py`

```py
# 导入数学库
import math
# 从 dataclasses 模块导入 dataclass 装饰器
from dataclasses import dataclass
# 从 typing 模块导入所需的类型
from typing import List, Optional, Tuple, Union

# 导入 PyTorch 库
import torch

# 从配置工具模块导入 ConfigMixin 和 register_to_config 函数
from ..configuration_utils import ConfigMixin, register_to_config
# 从工具模块导入 BaseOutput 类
from ..utils import BaseOutput
# 从调度工具模块导入 SchedulerMixin 类
from .scheduling_utils import SchedulerMixin


# 定义生成 Gumbel 噪声的函数
def gumbel_noise(t, generator=None):
    # 获取生成器设备，或使用输入张量 t 的设备
    device = generator.device if generator is not None else t.device
    # 创建与 t 形状相同的全零张量，并填充均匀分布的随机数
    noise = torch.zeros_like(t, device=device).uniform_(0, 1, generator=generator).to(t.device)
    # 返回 Gumbel 噪声
    return -torch.log((-torch.log(noise.clamp(1e-20))).clamp(1e-20))


# 定义根据随机选择的 top-k 进行掩蔽的函数
def mask_by_random_topk(mask_len, probs, temperature=1.0, generator=None):
    # 计算信心值，将概率取对数并添加 Gumbel 噪声
    confidence = torch.log(probs.clamp(1e-20)) + temperature * gumbel_noise(probs, generator=generator)
    # 对信心值进行排序
    sorted_confidence = torch.sort(confidence, dim=-1).values
    # 根据掩蔽长度获取截止值
    cut_off = torch.gather(sorted_confidence, 1, mask_len.long())
    # 创建掩蔽，标记信心值低于截止值的元素
    masking = confidence < cut_off
    # 返回掩蔽结果
    return masking


# 定义 AmusedSchedulerOutput 数据类
@dataclass
class AmusedSchedulerOutput(BaseOutput):
    """
    调度器 `step` 函数输出的输出类。

    参数:
        prev_sample (`torch.Tensor` 形状为 `(batch_size, num_channels, height, width)` 的图像):
            前一个时间步的计算样本 `(x_{t-1})`。`prev_sample` 应作为下一次模型输入用于去噪循环。
        pred_original_sample (`torch.Tensor` 形状为 `(batch_size, num_channels, height, width)` 的图像):
            基于当前时间步模型输出的预测去噪样本 `(x_{0})`。
            `pred_original_sample` 可用于预览进度或进行引导。
    """

    # 定义前一个样本和预测原始样本
    prev_sample: torch.Tensor
    pred_original_sample: torch.Tensor = None


# 定义 AmusedScheduler 类，继承自 SchedulerMixin 和 ConfigMixin
class AmusedScheduler(SchedulerMixin, ConfigMixin):
    # 设置调度器的顺序
    order = 1

    # 定义温度张量
    temperatures: torch.Tensor

    # 注册到配置的构造函数
    @register_to_config
    def __init__(
        self,
        mask_token_id: int,
        masking_schedule: str = "cosine",
    ):
        # 初始化温度和时间步
        self.temperatures = None
        self.timesteps = None

    # 设置时间步的函数
    def set_timesteps(
        self,
        num_inference_steps: int,
        temperature: Union[int, Tuple[int, int], List[int]] = (2, 0),
        device: Union[str, torch.device] = None,
    ):
        # 生成反向的时间步张量
        self.timesteps = torch.arange(num_inference_steps, device=device).flip(0)

        # 如果温度是元组或列表，生成线性间隔的温度张量
        if isinstance(temperature, (tuple, list)):
            self.temperatures = torch.linspace(temperature[0], temperature[1], num_inference_steps, device=device)
        # 否则生成固定温度到 0.01 的线性间隔温度张量
        else:
            self.temperatures = torch.linspace(temperature, 0.01, num_inference_steps, device=device)

    # 定义执行一步的函数
    def step(
        self,
        model_output: torch.Tensor,
        timestep: torch.long,
        sample: torch.LongTensor,
        starting_mask_ratio: int = 1,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = True,
    # 定义函数返回类型为 AmusedSchedulerOutput 或元组
        ) -> Union[AmusedSchedulerOutput, Tuple]:
            # 检查输入样本和模型输出的维度，判断是否为二维输入
            two_dim_input = sample.ndim == 3 and model_output.ndim == 4
    
            # 如果是二维输入
            if two_dim_input:
                # 解包模型输出的形状信息
                batch_size, codebook_size, height, width = model_output.shape
                # 重新调整样本的形状为 (batch_size, height * width)
                sample = sample.reshape(batch_size, height * width)
                # 重新调整模型输出的形状并转置
                model_output = model_output.reshape(batch_size, codebook_size, height * width).permute(0, 2, 1)
    
            # 创建一个布尔图，标识样本中的未知标记位置
            unknown_map = sample == self.config.mask_token_id
    
            # 对模型输出应用 softmax 函数，得到概率分布
            probs = model_output.softmax(dim=-1)
    
            # 获取概率的设备信息
            device = probs.device
            # 将概率移动到生成器所在的设备上（如果存在生成器）
            probs_ = probs.to(generator.device) if generator is not None else probs  # handles when generator is on CPU
            # 如果概率在 CPU 上且不是浮点32格式，转换为浮点32格式
            if probs_.device.type == "cpu" and probs_.dtype != torch.float32:
                probs_ = probs_.float()  # multinomial is not implemented for cpu half precision
            # 重新调整概率的形状为 (-1, 最后一个维度大小)
            probs_ = probs_.reshape(-1, probs.size(-1))
            # 根据概率分布进行抽样，获取预测的原始样本
            pred_original_sample = torch.multinomial(probs_, 1, generator=generator).to(device=device)
            # 调整预测样本的形状以匹配概率的形状
            pred_original_sample = pred_original_sample[:, 0].view(*probs.shape[:-1])
            # 在未知位置用原样本替换预测样本
            pred_original_sample = torch.where(unknown_map, pred_original_sample, sample)
    
            # 如果时间步为0，设置先前样本为预测的原样本
            if timestep == 0:
                prev_sample = pred_original_sample
            else:
                # 获取样本的序列长度
                seq_len = sample.shape[1]
                # 查找当前时间步的索引
                step_idx = (self.timesteps == timestep).nonzero()
                # 计算时间步的比例
                ratio = (step_idx + 1) / len(self.timesteps)
    
                # 根据遮罩调度的类型计算遮罩比例
                if self.config.masking_schedule == "cosine":
                    mask_ratio = torch.cos(ratio * math.pi / 2)
                elif self.config.masking_schedule == "linear":
                    mask_ratio = 1 - ratio
                else:
                    # 抛出未知的遮罩调度错误
                    raise ValueError(f"unknown masking schedule {self.config.masking_schedule}")
    
                # 计算最终的遮罩比例
                mask_ratio = starting_mask_ratio * mask_ratio
    
                # 计算需要遮罩的长度
                mask_len = (seq_len * mask_ratio).floor()
                # 确保不遮罩超过之前遮罩的数量
                mask_len = torch.min(unknown_map.sum(dim=-1, keepdim=True) - 1, mask_len)
                # 确保至少遮罩一个标记
                mask_len = torch.max(torch.tensor([1], device=model_output.device), mask_len)
    
                # 根据预测样本获取相应的概率
                selected_probs = torch.gather(probs, -1, pred_original_sample[:, :, None])[:, :, 0]
                # 忽略输入中给定的标记，覆盖其置信度
                selected_probs = torch.where(unknown_map, selected_probs, torch.finfo(selected_probs.dtype).max)
    
                # 通过随机 top-k 方法进行遮罩
                masking = mask_by_random_topk(mask_len, selected_probs, self.temperatures[step_idx], generator)
    
                # 用遮罩覆盖置信度较低的标记
                prev_sample = torch.where(masking, self.config.mask_token_id, pred_original_sample)
    
            # 如果是二维输入，调整样本的形状
            if two_dim_input:
                prev_sample = prev_sample.reshape(batch_size, height, width)
                pred_original_sample = pred_original_sample.reshape(batch_size, height, width)
    
            # 如果不返回字典格式，返回先前样本和预测样本的元组
            if not return_dict:
                return (prev_sample, pred_original_sample)
    
            # 返回 AmusedSchedulerOutput 对象
            return AmusedSchedulerOutput(prev_sample, pred_original_sample)
    # 定义一个方法，用于向样本添加噪声
        def add_noise(self, sample, timesteps, generator=None):
            # 获取当前时间步在 timesteps 中的位置索引
            step_idx = (self.timesteps == timesteps).nonzero()
            # 计算掩码比例，根据时间步的位置索引
            ratio = (step_idx + 1) / len(self.timesteps)
    
            # 根据配置选择掩码调度策略
            if self.config.masking_schedule == "cosine":
                # 使用余弦函数计算掩码比例
                mask_ratio = torch.cos(ratio * math.pi / 2)
            elif self.config.masking_schedule == "linear":
                # 线性掩码比例计算
                mask_ratio = 1 - ratio
            else:
                # 如果调度策略未知，抛出错误
                raise ValueError(f"unknown masking schedule {self.config.masking_schedule}")
    
            # 生成随机掩码索引，基于样本形状和设备
            mask_indices = (
                torch.rand(
                    sample.shape, device=generator.device if generator is not None else sample.device, generator=generator
                ).to(sample.device)  # 将随机张量移至样本设备
                < mask_ratio  # 与掩码比例进行比较，生成布尔掩码
            )
    
            # 创建样本的克隆，以便进行掩码操作
            masked_sample = sample.clone()
    
            # 将掩码位置的样本值替换为掩码标记 ID
            masked_sample[mask_indices] = self.config.mask_token_id
    
            # 返回添加噪声后的样本
            return masked_sample
```