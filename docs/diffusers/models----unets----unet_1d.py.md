# `.\diffusers\models\unets\unet_1d.py`

```py
# 版权声明，声明此文件归 HuggingFace 团队所有，所有权利保留。
# 
# 根据 Apache 许可证 2.0 版（“许可证”）进行许可；
# 除非遵循许可证，否则您不得使用此文件。
# 您可以在以下地址获取许可证副本：
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# 除非适用法律或书面协议另有约定，软件按“原样”分发，
# 不提供任何明示或暗示的保证或条件。
# 请参阅许可证以了解有关权限的具体语言和
# 限制条款。
#
# 从 dataclasses 模块导入 dataclass 装饰器，用于创建数据类
from dataclasses import dataclass
# 从 typing 模块导入 Optional, Tuple, Union 类型提示
from typing import Optional, Tuple, Union

# 导入 PyTorch 库，用于深度学习
import torch
# 导入 PyTorch 的神经网络模块
import torch.nn as nn

# 从配置工具模块导入 ConfigMixin 和 register_to_config 以实现配置功能
from ...configuration_utils import ConfigMixin, register_to_config
# 从 utils 模块导入 BaseOutput 类，作为输出基类
from ...utils import BaseOutput
# 从 embeddings 模块导入 GaussianFourierProjection, TimestepEmbedding, Timesteps，用于处理嵌入
from ..embeddings import GaussianFourierProjection, TimestepEmbedding, Timesteps
# 从 modeling_utils 模块导入 ModelMixin，作为模型混合基类
from ..modeling_utils import ModelMixin
# 从 unet_1d_blocks 模块导入用于构建 UNet 1D 的各个块
from .unet_1d_blocks import get_down_block, get_mid_block, get_out_block, get_up_block

# 定义 UNet1DOutput 数据类，继承自 BaseOutput
@dataclass
class UNet1DOutput(BaseOutput):
    """
    [`UNet1DModel`] 的输出。

    参数：
        sample (`torch.Tensor`，形状为 `(batch_size, num_channels, sample_size)`):
            模型最后一层输出的隐藏状态。
    """

    # 模型输出的样本张量
    sample: torch.Tensor

# 定义 UNet1DModel 类，继承自 ModelMixin 和 ConfigMixin
class UNet1DModel(ModelMixin, ConfigMixin):
    r"""
    1D UNet 模型，接收噪声样本和时间步并返回形状的输出样本。

    该模型继承自 [`ModelMixin`]。请查看超类文档，以获取其实现的所有模型的通用方法（例如下载或保存）。
    # 参数说明部分，列出可选参数及其默认值
    Parameters:
        # 默认样本长度，可在运行时适应的整型参数
        sample_size (`int`, *optional*): Default length of sample. Should be adaptable at runtime.
        # 输入样本的通道数，默认值为2
        in_channels (`int`, *optional*, defaults to 2): Number of channels in the input sample.
        # 输出的通道数，默认值为2
        out_channels (`int`, *optional*, defaults to 2): Number of channels in the output.
        # 附加的输入通道数，默认值为0
        extra_in_channels (`int`, *optional*, defaults to 0):
            # 首个下采样块输入中额外通道的数量，用于处理输入数据通道多于模型设计时的情况
            Number of additional channels to be added to the input of the first down block. Useful for cases where the
            input data has more channels than what the model was initially designed for.
        # 时间嵌入类型，默认值为"fourier"
        time_embedding_type (`str`, *optional*, defaults to `"fourier"`): Type of time embedding to use.
        # 傅里叶时间嵌入的频率偏移，默认值为0.0
        freq_shift (`float`, *optional*, defaults to 0.0): Frequency shift for Fourier time embedding.
        # 是否将正弦函数翻转为余弦函数，默认值为False
        flip_sin_to_cos (`bool`, *optional*, defaults to `False`):
            # 对于傅里叶时间嵌入，是否翻转sin为cos。
            Whether to flip sin to cos for Fourier time embedding.
        # 下采样块类型的元组，默认值为指定的块类型
        down_block_types (`Tuple[str]`, *optional*, defaults to  ("DownBlock1DNoSkip", "DownBlock1D", "AttnDownBlock1D")):
            # 下采样块类型的元组。
            Tuple of downsample block types.
        # 上采样块类型的元组，默认值为指定的块类型
        up_block_types (`Tuple[str]`, *optional*, defaults to  ("AttnUpBlock1D", "UpBlock1D", "UpBlock1DNoSkip")):
            # 上采样块类型的元组。
            Tuple of upsample block types.
        # 块输出通道的元组，默认值为(32, 32, 64)
        block_out_channels (`Tuple[int]`, *optional*, defaults to `(32, 32, 64)`):
            # 块输出通道的元组。
            Tuple of block output channels.
        # UNet中间块的类型，默认值为"UNetMidBlock1D"
        mid_block_type (`str`, *optional*, defaults to `"UNetMidBlock1D"`): Block type for middle of UNet.
        # UNet的可选输出处理块，默认值为None
        out_block_type (`str`, *optional*, defaults to `None`): Optional output processing block of UNet.
        # UNet块中的可选激活函数，默认值为None
        act_fn (`str`, *optional*, defaults to `None`): Optional activation function in UNet blocks.
        # 归一化的组数，默认值为8
        norm_num_groups (`int`, *optional*, defaults to 8): The number of groups for normalization.
        # 每个块的层数，默认值为1
        layers_per_block (`int`, *optional*, defaults to 1): The number of layers per block.
        # 每个块是否下采样，默认值为False
        downsample_each_block (`int`, *optional*, defaults to False):
            # 用于不进行上采样的UNet的实验特性。
            Experimental feature for using a UNet without upsampling.
    """    
    # 装饰器，注册到配置中
    @register_to_config
    def __init__(
        # 初始化方法，定义各种参数及其默认值
        self,
        # 默认样本大小为65536
        sample_size: int = 65536,
        # 可选样本速率，默认为None
        sample_rate: Optional[int] = None,
        # 输入通道数，默认值为2
        in_channels: int = 2,
        # 输出通道数，默认值为2
        out_channels: int = 2,
        # 附加输入通道数，默认值为0
        extra_in_channels: int = 0,
        # 时间嵌入类型，默认值为"fourier"
        time_embedding_type: str = "fourier",
        # 是否翻转正弦为余弦，默认值为True
        flip_sin_to_cos: bool = True,
        # 是否使用时间步长嵌入，默认值为False
        use_timestep_embedding: bool = False,
        # 傅里叶时间嵌入的频率偏移，默认值为0.0
        freq_shift: float = 0.0,
        # 下采样块类型的元组，默认值为指定的块类型
        down_block_types: Tuple[str] = ("DownBlock1DNoSkip", "DownBlock1D", "AttnDownBlock1D"),
        # 上采样块类型的元组，默认值为指定的块类型
        up_block_types: Tuple[str] = ("AttnUpBlock1D", "UpBlock1D", "UpBlock1DNoSkip"),
        # 中间块的类型，默认值为"UNetMidBlock1D"
        mid_block_type: Tuple[str] = "UNetMidBlock1D",
        # 可选的输出处理块，默认值为None
        out_block_type: str = None,
        # 块输出通道的元组，默认值为(32, 32, 64)
        block_out_channels: Tuple[int] = (32, 32, 64),
        # 可选激活函数，默认值为None
        act_fn: str = None,
        # 归一化的组数，默认值为8
        norm_num_groups: int = 8,
        # 每个块的层数，默认值为1
        layers_per_block: int = 1,
        # 是否下采样，默认值为False
        downsample_each_block: bool = False,
    # 定义 UNet1DModel 的前向传播方法
    def forward(
            self,
            sample: torch.Tensor,
            timestep: Union[torch.Tensor, float, int],
            return_dict: bool = True,
        ) -> Union[UNet1DOutput, Tuple]:
            r"""
            UNet1DModel 的前向传播方法。
    
            参数:
                sample (`torch.Tensor`):
                    噪声输入张量，形状为 `(batch_size, num_channels, sample_size)`。
                timestep (`torch.Tensor` 或 `float` 或 `int`): 用于去噪输入的时间步数。
                return_dict (`bool`, *可选*, 默认为 `True`):
                    是否返回 [`~models.unets.unet_1d.UNet1DOutput`] 而不是普通元组。
    
            返回:
                [`~models.unets.unet_1d.UNet1DOutput`] 或 `tuple`:
                    如果 `return_dict` 为 True，则返回 [`~models.unets.unet_1d.UNet1DOutput`]，否则返回一个元组，
                    其中第一个元素是样本张量。
            """
    
            # 1. 时间处理
            timesteps = timestep
            # 检查 timesteps 是否为张量，如果不是，则将其转换为张量
            if not torch.is_tensor(timesteps):
                timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
            # 如果 timesteps 是张量且没有形状，则扩展其维度
            elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
                timesteps = timesteps[None].to(sample.device)
    
            # 对时间步进行嵌入处理
            timestep_embed = self.time_proj(timesteps)
            # 如果使用时间步嵌入，则通过 MLP 进行处理
            if self.config.use_timestep_embedding:
                timestep_embed = self.time_mlp(timestep_embed)
            # 否则，调整嵌入的形状
            else:
                timestep_embed = timestep_embed[..., None]
                # 重复嵌入以匹配样本的大小
                timestep_embed = timestep_embed.repeat([1, 1, sample.shape[2]]).to(sample.dtype)
                # 广播嵌入以匹配样本的形状
                timestep_embed = timestep_embed.broadcast_to((sample.shape[:1] + timestep_embed.shape[1:]))
    
            # 2. 向下采样
            down_block_res_samples = ()
            # 遍历下采样块
            for downsample_block in self.down_blocks:
                # 在下采样块中处理样本和时间嵌入
                sample, res_samples = downsample_block(hidden_states=sample, temb=timestep_embed)
                # 收集残差样本
                down_block_res_samples += res_samples
    
            # 3. 中间块处理
            if self.mid_block:
                # 如果存在中间块，则进行处理
                sample = self.mid_block(sample, timestep_embed)
    
            # 4. 向上采样
            for i, upsample_block in enumerate(self.up_blocks):
                # 获取最后一个残差样本
                res_samples = down_block_res_samples[-1:]
                # 移除最后一个残差样本
                down_block_res_samples = down_block_res_samples[:-1]
                # 在上采样块中处理样本和时间嵌入
                sample = upsample_block(sample, res_hidden_states_tuple=res_samples, temb=timestep_embed)
    
            # 5. 后处理
            if self.out_block:
                # 如果存在输出块，则进行处理
                sample = self.out_block(sample, timestep_embed)
    
            # 如果不需要返回字典，则返回样本元组
            if not return_dict:
                return (sample,)
    
            # 返回 UNet1DOutput 对象
            return UNet1DOutput(sample=sample)
```