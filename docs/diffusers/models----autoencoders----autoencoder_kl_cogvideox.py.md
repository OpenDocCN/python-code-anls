# `.\diffusers\models\autoencoders\autoencoder_kl_cogvideox.py`

```py
# Copyright 2024 The CogVideoX team, Tsinghua University & ZhipuAI and The HuggingFace Team.
# All rights reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional, Tuple, Union  # 导入用于类型注解的 Optional、Tuple 和 Union

import numpy as np  # 导入 NumPy 库，用于数组和矩阵操作
import torch  # 导入 PyTorch 库，用于张量操作和深度学习
import torch.nn as nn  # 导入 PyTorch 的神经网络模块
import torch.nn.functional as F  # 导入 PyTorch 的函数式 API

from ...configuration_utils import ConfigMixin, register_to_config  # 从配置工具中导入配置混合类和注册配置的方法
from ...loaders.single_file_model import FromOriginalModelMixin  # 导入处理单文件模型的混合类
from ...utils import logging  # 导入日志工具
from ...utils.accelerate_utils import apply_forward_hook  # 导入用于应用前向钩子的工具
from ..activations import get_activation  # 导入获取激活函数的方法
from ..downsampling import CogVideoXDownsample3D  # 导入3D下采样模块
from ..modeling_outputs import AutoencoderKLOutput  # 导入自编码器KL输出的模块
from ..modeling_utils import ModelMixin  # 导入模型混合类
from ..upsampling import CogVideoXUpsample3D  # 导入3D上采样模块
from .vae import DecoderOutput, DiagonalGaussianDistribution  # 导入变分自编码器相关输出类和分布类


logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器，禁用 pylint 警告

class CogVideoXSafeConv3d(nn.Conv3d):  # 定义一个继承自 nn.Conv3d 的类，代表安全的3D卷积层
    r"""A 3D convolution layer that splits the input tensor into smaller parts to avoid OOM in CogVideoX Model.
    """  # 类文档字符串，描述该卷积层的功能

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # 定义前向传播方法，接收一个张量并返回一个张量
        memory_count = torch.prod(torch.tensor(input.shape)).item() * 2 / 1024**3  # 计算输入张量的内存占用（GB）

        # Set to 2GB, suitable for CuDNN
        if memory_count > 2:  # 如果内存占用超过2GB
            kernel_size = self.kernel_size[0]  # 获取卷积核的大小
            part_num = int(memory_count / 2) + 1  # 计算需要拆分的部分数量
            input_chunks = torch.chunk(input, part_num, dim=2)  # 将输入张量沿着深度维度拆分成多个块

            if kernel_size > 1:  # 如果卷积核大小大于1
                input_chunks = [input_chunks[0]] + [  # 将第一个块保留并处理后续块
                    torch.cat((input_chunks[i - 1][:, :, -kernel_size + 1 :], input_chunks[i]), dim=2)  # 将前一个块和当前块拼接
                    for i in range(1, len(input_chunks))  # 遍历后续块
                ]

            output_chunks = []  # 初始化输出块的列表
            for input_chunk in input_chunks:  # 遍历所有输入块
                output_chunks.append(super().forward(input_chunk))  # 使用父类的前向方法处理每个输入块并保存结果
            output = torch.cat(output_chunks, dim=2)  # 将所有输出块沿着深度维度拼接
            return output  # 返回拼接后的输出
        else:  # 如果内存占用不超过2GB
            return super().forward(input)  # 直接使用父类的前向方法处理输入张量


class CogVideoXCausalConv3d(nn.Module):  # 定义一个3D因果卷积层的类，继承自 nn.Module
    r"""A 3D causal convolution layer that pads the input tensor to ensure causality in CogVideoX Model.
    """  # 类文档字符串，描述该因果卷积层的功能
    # 参数说明文档
    Args:
        in_channels (`int`): 输入张量的通道数。
        out_channels (`int`): 卷积生成的输出通道数。
        kernel_size (`int` or `Tuple[int, int, int]`): 卷积核的大小。
        stride (`int`, defaults to `1`): 卷积的步幅。
        dilation (`int`, defaults to `1`): 卷积的扩张率。
        pad_mode (`str`, defaults to `"constant"`): 填充模式。
    """

    # 初始化方法
    def __init__(
        # 输入通道数
        self,
        in_channels: int,
        # 输出通道数
        out_channels: int,
        # 卷积核大小
        kernel_size: Union[int, Tuple[int, int, int]],
        # 步幅，默认为1
        stride: int = 1,
        # 扩张率，默认为1
        dilation: int = 1,
        # 填充模式，默认为"constant"
        pad_mode: str = "constant",
    ):
        # 调用父类构造函数
        super().__init__()

        # 如果卷积核大小是整数，则扩展为三维元组
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 3

        # 解包卷积核的时间、高度和宽度尺寸
        time_kernel_size, height_kernel_size, width_kernel_size = kernel_size

        # 设置填充模式
        self.pad_mode = pad_mode
        # 计算时间维度的填充量
        time_pad = dilation * (time_kernel_size - 1) + (1 - stride)
        # 计算高度和宽度的填充量
        height_pad = height_kernel_size // 2
        width_pad = width_kernel_size // 2

        # 保存填充量
        self.height_pad = height_pad
        self.width_pad = width_pad
        self.time_pad = time_pad
        # 设置因果填充参数
        self.time_causal_padding = (width_pad, width_pad, height_pad, height_pad, time_pad, 0)

        # 设置时间维度索引
        self.temporal_dim = 2
        # 保存时间卷积核大小
        self.time_kernel_size = time_kernel_size

        # 将步幅和扩张转换为三维元组
        stride = (stride, 1, 1)
        dilation = (dilation, 1, 1)
        # 创建三维卷积层对象
        self.conv = CogVideoXSafeConv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
        )

        # 初始化卷积缓存为None
        self.conv_cache = None

    # 假上下文并行前向传播方法
    def fake_context_parallel_forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # 获取时间卷积核大小
        kernel_size = self.time_kernel_size
        # 如果卷积核大小大于1，进行缓存处理
        if kernel_size > 1:
            # 使用缓存的输入，或者用当前输入的首个切片填充
            cached_inputs = (
                [self.conv_cache] if self.conv_cache is not None else [inputs[:, :, :1]] * (kernel_size - 1)
            )
            # 将缓存输入和当前输入连接在一起
            inputs = torch.cat(cached_inputs + [inputs], dim=2)
        # 返回处理后的输入
        return inputs

    # 清除假上下文并行缓存的方法
    def _clear_fake_context_parallel_cache(self):
        # 删除卷积缓存
        del self.conv_cache
        # 将卷积缓存设置为None
        self.conv_cache = None

    # 前向传播方法
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # 进行假上下文并行前向传播
        inputs = self.fake_context_parallel_forward(inputs)

        # 清除卷积缓存
        self._clear_fake_context_parallel_cache()
        # 注意：可以将这些数据移动到CPU以降低内存使用，但目前仅几百兆，不考虑
        # 缓存输入的最后几帧数据
        self.conv_cache = inputs[:, :, -self.time_kernel_size + 1 :].clone()

        # 设置二维填充参数
        padding_2d = (self.width_pad, self.width_pad, self.height_pad, self.height_pad)
        # 对输入进行填充
        inputs = F.pad(inputs, padding_2d, mode="constant", value=0)

        # 通过卷积层处理输入
        output = self.conv(inputs)
        # 返回卷积结果
        return output
# 定义一个用于空间条件归一化的3D视频处理模型
class CogVideoXSpatialNorm3D(nn.Module):
    r"""
    根据 https://arxiv.org/abs/2209.09002 中定义的空间条件归一化，专门针对3D视频数据的实现。

    使用 CogVideoXSafeConv3d 替代 nn.Conv3d，以避免在 CogVideoX 模型中出现内存不足问题。

    参数：
        f_channels (`int`):
            输入到组归一化层的通道数，以及空间归一化层的输出通道数。
        zq_channels (`int`):
            论文中描述的量化向量的通道数。
        groups (`int`):
            用于将通道分组的组数。
    """

    # 初始化模型
    def __init__(
        self,
        f_channels: int,
        zq_channels: int,
        groups: int = 32,
    ):
        # 调用父类构造函数
        super().__init__()
        # 创建组归一化层
        self.norm_layer = nn.GroupNorm(num_channels=f_channels, num_groups=groups, eps=1e-6, affine=True)
        # 创建因果卷积层用于Y通道
        self.conv_y = CogVideoXCausalConv3d(zq_channels, f_channels, kernel_size=1, stride=1)
        # 创建因果卷积层用于B通道
        self.conv_b = CogVideoXCausalConv3d(zq_channels, f_channels, kernel_size=1, stride=1)

    # 前向传播定义
    def forward(self, f: torch.Tensor, zq: torch.Tensor) -> torch.Tensor:
        # 检查输入形状，确保处理的逻辑正确
        if f.shape[2] > 1 and f.shape[2] % 2 == 1:
            # 分离第一个帧和其余帧
            f_first, f_rest = f[:, :, :1], f[:, :, 1:]
            # 获取各部分的大小
            f_first_size, f_rest_size = f_first.shape[-3:], f_rest.shape[-3:]
            # 分离量化向量
            z_first, z_rest = zq[:, :, :1], zq[:, :, 1:]
            # 进行插值调整大小
            z_first = F.interpolate(z_first, size=f_first_size)
            z_rest = F.interpolate(z_rest, size=f_rest_size)
            # 合并调整后的量化向量
            zq = torch.cat([z_first, z_rest], dim=2)
        else:
            # 对量化向量进行插值以匹配输入形状
            zq = F.interpolate(zq, size=f.shape[-3:])

        # 对输入进行归一化
        norm_f = self.norm_layer(f)
        # 计算新的输出
        new_f = norm_f * self.conv_y(zq) + self.conv_b(zq)
        # 返回处理后的结果
        return new_f


# 定义用于CogVideoX模型的3D ResNet块
class CogVideoXResnetBlock3D(nn.Module):
    r"""
    CogVideoX模型中使用的3D ResNet块。

    参数：
        in_channels (`int`):
            输入通道数。
        out_channels (`int`, *可选*):
            输出通道数。如果为 None，默认与 `in_channels` 相同。
        dropout (`float`, 默认值为 `0.0`):
            Dropout比率。
        temb_channels (`int`, 默认值为 `512`):
            时间嵌入通道数。
        groups (`int`, 默认值为 `32`):
            用于将通道分组的组数。
        eps (`float`, 默认值为 `1e-6`):
            归一化层的 epsilon 值。
        non_linearity (`str`, 默认值为 `"swish"`):
            使用的激活函数。
        conv_shortcut (bool, 默认值为 `False`):
            是否使用卷积快捷连接。
        spatial_norm_dim (`int`, *可选*):
            如果使用空间归一化而非组归一化时的维度。
        pad_mode (str, 默认值为 `"first"`):
            填充模式。
    """
    # 初始化方法，设置神经网络层的参数
    def __init__(
        self,
        in_channels: int,  # 输入通道数
        out_channels: Optional[int] = None,  # 输出通道数（可选，默认为 None）
        dropout: float = 0.0,  # 丢弃率（默认为 0.0）
        temb_channels: int = 512,  # 时间嵌入通道数（默认为 512）
        groups: int = 32,  # 分组数（默认为 32）
        eps: float = 1e-6,  # 为数值稳定性而添加的小常数（默认为 1e-6）
        non_linearity: str = "swish",  # 非线性激活函数的类型（默认为 "swish"）
        conv_shortcut: bool = False,  # 是否使用卷积快捷连接（默认为 False）
        spatial_norm_dim: Optional[int] = None,  # 空间归一化维度（可选）
        pad_mode: str = "first",  # 填充模式（默认为 "first"）
    ):
        # 调用父类初始化方法
        super().__init__()

        # 如果未提供输出通道数，则将其设置为输入通道数
        out_channels = out_channels or in_channels

        # 保存输入和输出通道数
        self.in_channels = in_channels
        self.out_channels = out_channels
        # 获取指定的非线性激活函数
        self.nonlinearity = get_activation(non_linearity)
        # 保存是否使用卷积快捷连接的标志
        self.use_conv_shortcut = conv_shortcut

        # 根据空间归一化维度选择归一化方法
        if spatial_norm_dim is None:
            # 创建第一个归一化层，使用分组归一化
            self.norm1 = nn.GroupNorm(num_channels=in_channels, num_groups=groups, eps=eps)
            # 创建第二个归一化层，使用分组归一化
            self.norm2 = nn.GroupNorm(num_channels=out_channels, num_groups=groups, eps=eps)
        else:
            # 创建第一个归一化层，使用空间归一化
            self.norm1 = CogVideoXSpatialNorm3D(
                f_channels=in_channels,
                zq_channels=spatial_norm_dim,
                groups=groups,
            )
            # 创建第二个归一化层，使用空间归一化
            self.norm2 = CogVideoXSpatialNorm3D(
                f_channels=out_channels,
                zq_channels=spatial_norm_dim,
                groups=groups,
            )

        # 创建第一个卷积层
        self.conv1 = CogVideoXCausalConv3d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=3, pad_mode=pad_mode
        )

        # 如果时间嵌入通道数大于 0，则创建时间嵌入投影层
        if temb_channels > 0:
            self.temb_proj = nn.Linear(in_features=temb_channels, out_features=out_channels)

        # 创建丢弃层
        self.dropout = nn.Dropout(dropout)
        # 创建第二个卷积层
        self.conv2 = CogVideoXCausalConv3d(
            in_channels=out_channels, out_channels=out_channels, kernel_size=3, pad_mode=pad_mode
        )

        # 如果输入通道数与输出通道数不相同，则创建快捷连接
        if self.in_channels != self.out_channels:
            # 如果使用卷积快捷连接
            if self.use_conv_shortcut:
                # 创建卷积快捷连接层
                self.conv_shortcut = CogVideoXCausalConv3d(
                    in_channels=in_channels, out_channels=out_channels, kernel_size=3, pad_mode=pad_mode
                )
            else:
                # 创建安全卷积快捷连接层
                self.conv_shortcut = CogVideoXSafeConv3d(
                    in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0
                )

    # 前向传播方法，定义模型如何处理输入数据
    def forward(
        self,
        inputs: torch.Tensor,  # 输入张量
        temb: Optional[torch.Tensor] = None,  # 可选的时间嵌入张量
        zq: Optional[torch.Tensor] = None,  # 可选的 zq 张量
    # 定义函数的返回类型为 torch.Tensor
        ) -> torch.Tensor:
            # 初始化隐藏状态为输入
            hidden_states = inputs
    
            # 如果 zq 不为 None，则对隐藏状态应用 norm1 归一化
            if zq is not None:
                hidden_states = self.norm1(hidden_states, zq)
            # 否则仅对隐藏状态应用 norm1 归一化
            else:
                hidden_states = self.norm1(hidden_states)
    
            # 应用非线性激活函数
            hidden_states = self.nonlinearity(hidden_states)
            # 通过卷积层 conv1 处理隐藏状态
            hidden_states = self.conv1(hidden_states)
    
            # 如果 temb 不为 None，则将其通过投影与隐藏状态相加
            if temb is not None:
                hidden_states = hidden_states + self.temb_proj(self.nonlinearity(temb))[:, :, None, None, None]
    
            # 如果 zq 不为 None，则对隐藏状态应用 norm2 归一化
            if zq is not None:
                hidden_states = self.norm2(hidden_states, zq)
            # 否则仅对隐藏状态应用 norm2 归一化
            else:
                hidden_states = self.norm2(hidden_states)
    
            # 应用非线性激活函数
            hidden_states = self.nonlinearity(hidden_states)
            # 进行 dropout 操作以防止过拟合
            hidden_states = self.dropout(hidden_states)
            # 通过卷积层 conv2 处理隐藏状态
            hidden_states = self.conv2(hidden_states)
    
            # 如果输入通道数与输出通道数不相等，应用卷积快捷连接
            if self.in_channels != self.out_channels:
                inputs = self.conv_shortcut(inputs)
    
            # 将输入与隐藏状态相加以形成最终的隐藏状态
            hidden_states = hidden_states + inputs
            # 返回最终的隐藏状态
            return hidden_states
# 定义一个用于 CogVideoX 模型的下采样模块
class CogVideoXDownBlock3D(nn.Module):
    r"""
    CogVideoX 模型中使用的下采样块。

    Args:
        in_channels (`int`):
            输入通道数。
        out_channels (`int`, *可选*):
            输出通道数。如果为 None，默认为 `in_channels`。
        temb_channels (`int`, defaults to `512`):
            时间嵌入通道数。
        num_layers (`int`, defaults to `1`):
            ResNet 层数。
        dropout (`float`, defaults to `0.0`):
            Dropout 率。
        resnet_eps (`float`, defaults to `1e-6`):
            归一化层的 epsilon 值。
        resnet_act_fn (`str`, defaults to `"swish"`):
            使用的激活函数。
        resnet_groups (`int`, defaults to `32`):
            用于组归一化的通道组数。
        add_downsample (`bool`, defaults to `True`):
            是否使用下采样层。如果不使用，输出维度将与输入维度相同。
        compress_time (`bool`, defaults to `False`):
            是否在时间维度上进行下采样。
        pad_mode (str, defaults to `"first"`):
            填充模式。
    """

    # 支持梯度检查点
    _supports_gradient_checkpointing = True

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        add_downsample: bool = True,
        downsample_padding: int = 0,
        compress_time: bool = False,
        pad_mode: str = "first",
    ):
        # 初始化父类 nn.Module
        super().__init__()

        resnets = []  # 创建一个空列表以存储 ResNet 层
        for i in range(num_layers):
            # 确定当前层的输入通道数
            in_channel = in_channels if i == 0 else out_channels
            # 将 ResNet 层添加到列表中
            resnets.append(
                CogVideoXResnetBlock3D(
                    in_channels=in_channel,
                    out_channels=out_channels,
                    dropout=dropout,
                    temb_channels=temb_channels,
                    groups=resnet_groups,
                    eps=resnet_eps,
                    non_linearity=resnet_act_fn,
                    pad_mode=pad_mode,
                )
            )

        # 将 ResNet 层列表转换为 nn.ModuleList 以便于管理
        self.resnets = nn.ModuleList(resnets)
        self.downsamplers = None  # 初始化下采样层为 None

        # 如果需要下采样，则添加下采样层
        if add_downsample:
            self.downsamplers = nn.ModuleList(
                [
                    CogVideoXDownsample3D(
                        out_channels, out_channels, padding=downsample_padding, compress_time=compress_time
                    )
                ]
            )

        self.gradient_checkpointing = False  # 初始化梯度检查点为 False

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: Optional[torch.Tensor] = None,
        zq: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:  # 指定返回类型为 torch.Tensor
        for resnet in self.resnets:  # 遍历每个 ResNet 模块
            if self.training and self.gradient_checkpointing:  # 检查是否在训练模式且启用梯度检查点

                def create_custom_forward(module):  # 定义创建自定义前向传播的函数
                    def create_forward(*inputs):  # 定义前向传播的具体实现
                        return module(*inputs)  # 调用传入的模块进行前向传播

                    return create_forward  # 返回前向传播函数

                hidden_states = torch.utils.checkpoint.checkpoint(  # 使用检查点机制计算前向传播
                    create_custom_forward(resnet), hidden_states, temb, zq  # 调用自定义前向函数并传入参数
                )
            else:  # 如果不满足前面条件
                hidden_states = resnet(hidden_states, temb, zq)  # 直接通过 ResNet 模块进行前向传播

        if self.downsamplers is not None:  # 检查是否存在下采样模块
            for downsampler in self.downsamplers:  # 遍历每个下采样模块
                hidden_states = downsampler(hidden_states)  # 通过下采样模块处理隐藏状态

        return hidden_states  # 返回处理后的隐藏状态
# 定义 CogVideoX 模型中的一个中间模块，继承自 nn.Module
class CogVideoXMidBlock3D(nn.Module):
    r"""
    CogVideoX 模型中使用的中间块。

    参数:
        in_channels (`int`):
            输入通道的数量。
        temb_channels (`int`, defaults to `512`):
            时间嵌入通道的数量。
        dropout (`float`, defaults to `0.0`):
            dropout 比率。
        num_layers (`int`, defaults to `1`):
            ResNet 层的数量。
        resnet_eps (`float`, defaults to `1e-6`):
            归一化层的 epsilon 值。
        resnet_act_fn (`str`, defaults to `"swish"`):
            要使用的激活函数。
        resnet_groups (`int`, defaults to `32`):
            用于组归一化的通道分组数量。
        spatial_norm_dim (`int`, *optional*):
            如果使用空间归一化而不是组归一化，则使用的维度。
        pad_mode (str, defaults to `"first"`):
            填充模式。
    """

    # 指示是否支持梯度检查点
    _supports_gradient_checkpointing = True

    # 初始化方法
    def __init__(
        self,
        in_channels: int,  # 输入通道数
        temb_channels: int,  # 时间嵌入通道数
        dropout: float = 0.0,  # dropout 比率
        num_layers: int = 1,  # ResNet 层数
        resnet_eps: float = 1e-6,  # 归一化层的 epsilon 值
        resnet_act_fn: str = "swish",  # 激活函数
        resnet_groups: int = 32,  # 组归一化的组数
        spatial_norm_dim: Optional[int] = None,  # 空间归一化的维度
        pad_mode: str = "first",  # 填充模式
    ):
        super().__init__()  # 调用父类的初始化方法

        resnets = []  # 初始化一个空列表以存储 ResNet 层
        for _ in range(num_layers):  # 根据层数循环
            resnets.append(  # 将新的 ResNet 层添加到列表中
                CogVideoXResnetBlock3D(  # 实例化 ResNet 层
                    in_channels=in_channels,  # 输入通道数
                    out_channels=in_channels,  # 输出通道数与输入相同
                    dropout=dropout,  # dropout 比率
                    temb_channels=temb_channels,  # 时间嵌入通道数
                    groups=resnet_groups,  # 组归一化的组数
                    eps=resnet_eps,  # epsilon 值
                    spatial_norm_dim=spatial_norm_dim,  # 空间归一化的维度
                    non_linearity=resnet_act_fn,  # 激活函数
                    pad_mode=pad_mode,  # 填充模式
                )
            )
        self.resnets = nn.ModuleList(resnets)  # 将 ResNet 层列表转换为 ModuleList

        self.gradient_checkpointing = False  # 初始化梯度检查点标志为 False

    # 前向传播方法
    def forward(
        self,
        hidden_states: torch.Tensor,  # 隐藏状态的输入张量
        temb: Optional[torch.Tensor] = None,  # 可选的时间嵌入张量
        zq: Optional[torch.Tensor] = None,  # 可选的 zq 张量
    ) -> torch.Tensor:  # 返回张量
        for resnet in self.resnets:  # 遍历每个 ResNet 层
            if self.training and self.gradient_checkpointing:  # 如果在训练中且支持梯度检查点

                # 创建一个自定义前向传播的函数
                def create_custom_forward(module):
                    def create_forward(*inputs):  # 定义前向传播函数
                        return module(*inputs)  # 调用模块的前向传播

                    return create_forward  # 返回前向传播函数

                # 使用检查点机制执行前向传播以节省内存
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(resnet),  # 传入自定义前向函数
                    hidden_states,  # 隐藏状态
                    temb,  # 时间嵌入
                    zq  # zq 张量
                )
            else:
                hidden_states = resnet(hidden_states, temb, zq)  # 直接调用 ResNet 层的前向传播

        return hidden_states  # 返回隐藏状态的输出


# 定义 CogVideoX 模型中的一个上采样模块，继承自 nn.Module
class CogVideoXUpBlock3D(nn.Module):
    r"""
    CogVideoX 模型中使用的上采样块。
    # 参数说明
    Args:
        in_channels (`int`):  # 输入通道的数量
            Number of input channels.
        out_channels (`int`, *optional*):  # 输出通道的数量，如果为 None，则默认为 `in_channels`
            Number of output channels. If None, defaults to `in_channels`.
        temb_channels (`int`, defaults to `512`):  # 时间嵌入通道的数量
            Number of time embedding channels.
        dropout (`float`, defaults to `0.0`):  # dropout 率
            Dropout rate.
        num_layers (`int`, defaults to `1`):  # ResNet 层的数量
            Number of resnet layers.
        resnet_eps (`float`, defaults to `1e-6`):  # 归一化层的 epsilon 值
            Epsilon value for normalization layers.
        resnet_act_fn (`str`, defaults to `"swish"`):  # 使用的激活函数
            Activation function to use.
        resnet_groups (`int`, defaults to `32`):  # 用于组归一化的通道组数
            Number of groups to separate the channels into for group normalization.
        spatial_norm_dim (`int`, defaults to `16`):  # 用于空间归一化的维度
            The dimension to use for spatial norm if it is to be used instead of group norm.
        add_upsample (`bool`, defaults to `True`):  # 是否使用上采样层
            Whether or not to use a upsampling layer. If not used, output dimension would be same as input dimension.
        compress_time (`bool`, defaults to `False`):  # 是否在时间维度上进行下采样
            Whether or not to downsample across temporal dimension.
        pad_mode (str, defaults to `"first"`):  # 填充模式
            Padding mode.
    """

    def __init__(  # 初始化方法
        self,
        in_channels: int,  # 输入通道数量
        out_channels: int,  # 输出通道数量
        temb_channels: int,  # 时间嵌入通道数量
        dropout: float = 0.0,  # dropout 率
        num_layers: int = 1,  # ResNet 层数量
        resnet_eps: float = 1e-6,  # 归一化 epsilon 值
        resnet_act_fn: str = "swish",  # 激活函数
        resnet_groups: int = 32,  # 组归一化的组数
        spatial_norm_dim: int = 16,  # 空间归一化维度
        add_upsample: bool = True,  # 是否添加上采样层
        upsample_padding: int = 1,  # 上采样时的填充
        compress_time: bool = False,  # 是否压缩时间维度
        pad_mode: str = "first",  # 填充模式
    ):
        super().__init__()  # 调用父类初始化方法

        resnets = []  # 初始化空列表以存储 ResNet 层
        for i in range(num_layers):  # 遍历每一层
            in_channel = in_channels if i == 0 else out_channels  # 确定当前层的输入通道数量
            resnets.append(  # 将新的 ResNet 块添加到列表中
                CogVideoXResnetBlock3D(
                    in_channels=in_channel,  # 设置输入通道数量
                    out_channels=out_channels,  # 设置输出通道数量
                    dropout=dropout,  # 设置 dropout 率
                    temb_channels=temb_channels,  # 设置时间嵌入通道数量
                    groups=resnet_groups,  # 设置组数量
                    eps=resnet_eps,  # 设置 epsilon 值
                    non_linearity=resnet_act_fn,  # 设置非线性激活函数
                    spatial_norm_dim=spatial_norm_dim,  # 设置空间归一化维度
                    pad_mode=pad_mode,  # 设置填充模式
                )
            )

        self.resnets = nn.ModuleList(resnets)  # 将 ResNet 层列表转换为 ModuleList
        self.upsamplers = None  # 初始化上采样器为 None

        if add_upsample:  # 如果需要添加上采样层
            self.upsamplers = nn.ModuleList(  # 创建上采样器的 ModuleList
                [
                    CogVideoXUpsample3D(  # 添加上采样层
                        out_channels, out_channels, padding=upsample_padding, compress_time=compress_time
                    )
                ]
            )

        self.gradient_checkpointing = False  # 初始化梯度检查点标志为 False

    def forward(  # 前向传播方法
        self,
        hidden_states: torch.Tensor,  # 输入的隐藏状态
        temb: Optional[torch.Tensor] = None,  # 可选的时间嵌入张量
        zq: Optional[torch.Tensor] = None,  # 可选的额外张量
    # CogVideoXUpBlock3D 类的前向传播方法
    ) -> torch.Tensor:
            r"""Forward method of the `CogVideoXUpBlock3D` class."""
            # 遍历类中的每个 ResNet 模块
            for resnet in self.resnets:
                # 如果处于训练模式并且启用了梯度检查点
                if self.training and self.gradient_checkpointing:
                    # 定义一个自定义前向传播函数
                    def create_custom_forward(module):
                        # 创建接受输入的前向传播函数
                        def create_forward(*inputs):
                            return module(*inputs)
                        # 返回自定义的前向传播函数
                        return create_forward
    
                    # 使用梯度检查点机制计算隐藏状态
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(resnet), hidden_states, temb, zq
                    )
                else:
                    # 直接通过 ResNet 模块处理隐藏状态
                    hidden_states = resnet(hidden_states, temb, zq)
    
            # 如果存在上采样器
            if self.upsamplers is not None:
                # 遍历每个上采样器
                for upsampler in self.upsamplers:
                    # 通过上采样器处理隐藏状态
                    hidden_states = upsampler(hidden_states)
    
            # 返回最终的隐藏状态
            return hidden_states
# 定义一个名为 `CogVideoXEncoder3D` 的类，继承自 `nn.Module`，用于变分自编码器
class CogVideoXEncoder3D(nn.Module):
    r"""
    `CogVideoXEncoder3D` 层用于将输入编码为潜在表示的变分自编码器。

    参数:
        in_channels (`int`, *可选*, 默认值为 3):
            输入通道的数量。
        out_channels (`int`, *可选*, 默认值为 3):
            输出通道的数量。
        down_block_types (`Tuple[str, ...]`, *可选*, 默认值为 `("DownEncoderBlock2D",)`):
            使用的下采样块类型。有关可用选项，请参见 `~diffusers.models.unet_2d_blocks.get_down_block`。
        block_out_channels (`Tuple[int, ...]`, *可选*, 默认值为 `(64,)`):
            每个块的输出通道数量。
        act_fn (`str`, *可选*, 默认值为 `"silu"`):
            要使用的激活函数。有关可用选项，请参见 `~diffusers.models.activations.get_activation`。
        layers_per_block (`int`, *可选*, 默认值为 2):
            每个块的层数。
        norm_num_groups (`int`, *可选*, 默认值为 32):
            归一化的组数。
    """

    # 设置类属性以支持梯度检查点
    _supports_gradient_checkpointing = True

    # 初始化方法，设置类的参数
    def __init__(
        self,
        in_channels: int = 3,  # 输入通道数，默认为 3
        out_channels: int = 16,  # 输出通道数，默认为 16
        down_block_types: Tuple[str, ...] = (  # 下采样块类型的元组
            "CogVideoXDownBlock3D",  # 第一个下采样块类型
            "CogVideoXDownBlock3D",  # 第二个下采样块类型
            "CogVideoXDownBlock3D",  # 第三个下采样块类型
            "CogVideoXDownBlock3D",  # 第四个下采样块类型
        ),
        block_out_channels: Tuple[int, ...] = (128, 256, 256, 512),  # 每个块的输出通道数的元组
        layers_per_block: int = 3,  # 每个块的层数，默认为 3
        act_fn: str = "silu",  # 激活函数，默认为 "silu"
        norm_eps: float = 1e-6,  # 归一化的 epsilon 值，默认为 1e-6
        norm_num_groups: int = 32,  # 归一化组数，默认为 32
        dropout: float = 0.0,  # dropout 概率，默认为 0.0
        pad_mode: str = "first",  # 填充模式，默认为 "first"
        temporal_compression_ratio: float = 4,  # 时间压缩比，默认为 4
    ):
        # 调用父类的初始化方法
        super().__init__()

        # 计算时间压缩等级的对数（以2为底）
        temporal_compress_level = int(np.log2(temporal_compression_ratio))

        # 创建一个三维卷积层，输入通道数为in_channels，输出通道数为block_out_channels[0]，卷积核大小为3
        self.conv_in = CogVideoXCausalConv3d(in_channels, block_out_channels[0], kernel_size=3, pad_mode=pad_mode)
        # 初始化一个空的ModuleList，用于存储下采样模块
        self.down_blocks = nn.ModuleList([])

        # 设置初始输出通道数为第一个块的输出通道数
        output_channel = block_out_channels[0]
        # 遍历下采样模块的类型，i为索引，down_block_type为类型
        for i, down_block_type in enumerate(down_block_types):
            # 输入通道数为当前输出通道数
            input_channel = output_channel
            # 更新输出通道数为当前块的输出通道数
            output_channel = block_out_channels[i]
            # 判断是否为最后一个下采样块
            is_final_block = i == len(block_out_channels) - 1
            # 判断当前块是否需要压缩时间
            compress_time = i < temporal_compress_level

            # 如果下采样模块的类型为CogVideoXDownBlock3D
            if down_block_type == "CogVideoXDownBlock3D":
                # 创建下采样块，设置输入输出通道、丢弃率等参数
                down_block = CogVideoXDownBlock3D(
                    in_channels=input_channel,
                    out_channels=output_channel,
                    temb_channels=0,
                    dropout=dropout,
                    num_layers=layers_per_block,
                    resnet_eps=norm_eps,
                    resnet_act_fn=act_fn,
                    resnet_groups=norm_num_groups,
                    # 如果不是最后一个块则添加下采样
                    add_downsample=not is_final_block,
                    compress_time=compress_time,
                )
            else:
                # 如果下采样模块类型无效，则抛出异常
                raise ValueError("Invalid `down_block_type` encountered. Must be `CogVideoXDownBlock3D`")

            # 将创建的下采样块添加到down_blocks列表中
            self.down_blocks.append(down_block)

        # 创建中间块
        self.mid_block = CogVideoXMidBlock3D(
            in_channels=block_out_channels[-1],
            temb_channels=0,
            dropout=dropout,
            num_layers=2,
            resnet_eps=norm_eps,
            resnet_act_fn=act_fn,
            resnet_groups=norm_num_groups,
            pad_mode=pad_mode,
        )

        # 创建归一化层，使用GroupNorm
        self.norm_out = nn.GroupNorm(norm_num_groups, block_out_channels[-1], eps=1e-6)
        # 创建激活函数层，使用SiLU激活函数
        self.conv_act = nn.SiLU()
        # 创建输出卷积层，将最后一个块的输出通道数转换为2倍的out_channels
        self.conv_out = CogVideoXCausalConv3d(
            block_out_channels[-1], 2 * out_channels, kernel_size=3, pad_mode=pad_mode
        )

        # 初始化梯度检查点为False
        self.gradient_checkpointing = False
    # 定义 `CogVideoXEncoder3D` 类的前向传播方法，接收输入样本和可选的时间嵌入
    def forward(self, sample: torch.Tensor, temb: Optional[torch.Tensor] = None) -> torch.Tensor:
        r"""`CogVideoXEncoder3D` 类的前向方法。"""
        # 通过输入样本进行初始卷积，得到隐藏状态
        hidden_states = self.conv_in(sample)

        # 检查是否在训练模式并且启用梯度检查点
        if self.training and self.gradient_checkpointing:

            # 定义一个创建自定义前向传播函数的内部函数
            def create_custom_forward(module):
                # 自定义前向传播，传入可变参数
                def custom_forward(*inputs):
                    return module(*inputs)

                return custom_forward

            # 1. 向下采样
            # 遍历下采样块，并应用检查点以减少内存使用
            for down_block in self.down_blocks:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(down_block), hidden_states, temb, None
                )

            # 2. 中间块
            # 对中间块进行检查点处理
            hidden_states = torch.utils.checkpoint.checkpoint(
                create_custom_forward(self.mid_block), hidden_states, temb, None
            )
        else:
            # 如果不是训练模式，直接执行前向传播
            
            # 1. 向下采样
            # 遍历下采样块，直接应用每个下采样块的前向传播
            for down_block in self.down_blocks:
                hidden_states = down_block(hidden_states, temb, None)

            # 2. 中间块
            # 直接应用中间块的前向传播
            hidden_states = self.mid_block(hidden_states, temb, None)

        # 3. 后处理
        # 对隐藏状态进行归一化处理
        hidden_states = self.norm_out(hidden_states)
        # 应用激活函数
        hidden_states = self.conv_act(hidden_states)
        # 通过最后的卷积层输出结果
        hidden_states = self.conv_out(hidden_states)
        # 返回最终的隐藏状态
        return hidden_states
# 定义一个名为 `CogVideoXDecoder3D` 的类，继承自 `nn.Module`
class CogVideoXDecoder3D(nn.Module):
    r"""
    `CogVideoXDecoder3D` 是一个变分自编码器的层，用于将潜在表示解码为输出样本。

    参数：
        in_channels (`int`, *可选*, 默认为 3):
            输入通道的数量。
        out_channels (`int`, *可选*, 默认为 3):
            输出通道的数量。
        up_block_types (`Tuple[str, ...]`, *可选*, 默认为 `("UpDecoderBlock2D",)`):
            要使用的上采样块类型。请参见 `~diffusers.models.unet_2d_blocks.get_up_block` 获取可用选项。
        block_out_channels (`Tuple[int, ...]`, *可选*, 默认为 `(64,)`):
            每个块的输出通道数量。
        act_fn (`str`, *可选*, 默认为 `"silu"`):
            要使用的激活函数。请参见 `~diffusers.models.activations.get_activation` 获取可用选项。
        layers_per_block (`int`, *可选*, 默认为 2):
            每个块的层数。
        norm_num_groups (`int`, *可选*, 默认为 32):
            归一化的组数。
    """

    # 定义一个类属性，表示支持梯度检查点
    _supports_gradient_checkpointing = True

    # 初始化方法，定义类的构造函数
    def __init__(
        # 输入通道数量，默认为 16
        in_channels: int = 16,
        # 输出通道数量，默认为 3
        out_channels: int = 3,
        # 上采样块类型的元组，包含四个 'CogVideoXUpBlock3D'
        up_block_types: Tuple[str, ...] = (
            "CogVideoXUpBlock3D",
            "CogVideoXUpBlock3D",
            "CogVideoXUpBlock3D",
            "CogVideoXUpBlock3D",
        ),
        # 每个块的输出通道数量，指定为 128, 256, 256, 512
        block_out_channels: Tuple[int, ...] = (128, 256, 256, 512),
        # 每个块的层数，默认为 3
        layers_per_block: int = 3,
        # 激活函数名称，默认为 "silu"
        act_fn: str = "silu",
        # 归一化的 epsilon 值，默认为 1e-6
        norm_eps: float = 1e-6,
        # 归一化的组数，默认为 32
        norm_num_groups: int = 32,
        # dropout 比例，默认为 0.0
        dropout: float = 0.0,
        # 填充模式，默认为 "first"
        pad_mode: str = "first",
        # 时间压缩比，默认为 4
        temporal_compression_ratio: float = 4,
    ):
        # 调用父类的初始化方法
        super().__init__()

        # 反转输出通道列表，以便后续处理
        reversed_block_out_channels = list(reversed(block_out_channels))

        # 创建输入卷积层，使用反转后的输出通道的第一个元素
        self.conv_in = CogVideoXCausalConv3d(
            in_channels, reversed_block_out_channels[0], kernel_size=3, pad_mode=pad_mode
        )

        # 创建中间块
        self.mid_block = CogVideoXMidBlock3D(
            # 使用反转后的输出通道的第一个元素作为输入通道
            in_channels=reversed_block_out_channels[0],
            temb_channels=0,
            num_layers=2,
            resnet_eps=norm_eps,
            resnet_act_fn=act_fn,
            resnet_groups=norm_num_groups,
            spatial_norm_dim=in_channels,
            pad_mode=pad_mode,
        )

        # 初始化上采样块的模块列表
        self.up_blocks = nn.ModuleList([])

        # 设置当前的输出通道为反转后的输出通道的第一个元素
        output_channel = reversed_block_out_channels[0]
        # 计算时间压缩级别
        temporal_compress_level = int(np.log2(temporal_compression_ratio))

        # 遍历每种上采样块类型
        for i, up_block_type in enumerate(up_block_types):
            # 保存前一个输出通道
            prev_output_channel = output_channel
            # 更新当前输出通道为反转后的输出通道
            output_channel = reversed_block_out_channels[i]
            # 判断当前块是否为最后一个块
            is_final_block = i == len(block_out_channels) - 1
            # 判断是否需要时间压缩
            compress_time = i < temporal_compress_level

            # 如果块类型为指定的上采样块类型
            if up_block_type == "CogVideoXUpBlock3D":
                # 创建上采样块
                up_block = CogVideoXUpBlock3D(
                    in_channels=prev_output_channel,
                    out_channels=output_channel,
                    temb_channels=0,
                    dropout=dropout,
                    num_layers=layers_per_block + 1,
                    resnet_eps=norm_eps,
                    resnet_act_fn=act_fn,
                    resnet_groups=norm_num_groups,
                    spatial_norm_dim=in_channels,
                    add_upsample=not is_final_block,
                    compress_time=compress_time,
                    pad_mode=pad_mode,
                )
                # 更新前一个输出通道
                prev_output_channel = output_channel
            else:
                # 如果上采样块类型不合法，抛出错误
                raise ValueError("Invalid `up_block_type` encountered. Must be `CogVideoXUpBlock3D`")

            # 将创建的上采样块添加到模块列表中
            self.up_blocks.append(up_block)

        # 创建输出的空间归一化层
        self.norm_out = CogVideoXSpatialNorm3D(reversed_block_out_channels[-1], in_channels, groups=norm_num_groups)
        # 创建激活函数层
        self.conv_act = nn.SiLU()
        # 创建输出卷积层
        self.conv_out = CogVideoXCausalConv3d(
            reversed_block_out_channels[-1], out_channels, kernel_size=3, pad_mode=pad_mode
        )

        # 初始化梯度检查点标志为 False
        self.gradient_checkpointing = False
    # 定义 `CogVideoXDecoder3D` 类的前向传播方法
    def forward(self, sample: torch.Tensor, temb: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 方法文档字符串，描述该方法的功能
        r"""The forward method of the `CogVideoXDecoder3D` class."""
        # 对输入样本应用初始卷积，生成隐藏状态
        hidden_states = self.conv_in(sample)
    
        # 如果处于训练模式且启用梯度检查点
        if self.training and self.gradient_checkpointing:
    
            # 创建自定义前向传播函数
            def create_custom_forward(module):
                # 定义接受输入并调用模块的函数
                def custom_forward(*inputs):
                    return module(*inputs)
    
                return custom_forward
    
            # 1. 中间块处理
            hidden_states = torch.utils.checkpoint.checkpoint(
                create_custom_forward(self.mid_block), hidden_states, temb, sample
            )
    
            # 2. 上采样块处理
            for up_block in self.up_blocks:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(up_block), hidden_states, temb, sample
                )
        else:
            # 1. 中间块处理
            hidden_states = self.mid_block(hidden_states, temb, sample)
    
            # 2. 上采样块处理
            for up_block in self.up_blocks:
                hidden_states = up_block(hidden_states, temb, sample)
    
        # 3. 后处理
        hidden_states = self.norm_out(hidden_states, sample)  # 归一化输出
        hidden_states = self.conv_act(hidden_states)          # 应用激活函数
        hidden_states = self.conv_out(hidden_states)          # 应用最终卷积
        return hidden_states                                   # 返回处理后的隐藏状态
# 定义一个名为 AutoencoderKLCogVideoX 的类，继承自 ModelMixin、ConfigMixin 和 FromOriginalModelMixin
class AutoencoderKLCogVideoX(ModelMixin, ConfigMixin, FromOriginalModelMixin):
    r"""
    一个具有 KL 损失的变分自编码器（VAE）模型，用于将图像编码为潜在表示并解码潜在表示为图像。用于
    [CogVideoX](https://github.com/THUDM/CogVideo)。

    该模型继承自 [`ModelMixin`]。有关所有模型实现的通用方法（例如下载或保存）的详细信息，请查看父类文档。

    参数：
        in_channels (int, *可选*，默认值为 3)：输入图像的通道数。
        out_channels (int,  *可选*，默认值为 3)：输出的通道数。
        down_block_types (`Tuple[str]`, *可选*，默认值为 `("DownEncoderBlock2D",)`):
            下采样块类型的元组。
        up_block_types (`Tuple[str]`, *可选*，默认值为 `("UpDecoderBlock2D",)`):
            上采样块类型的元组。
        block_out_channels (`Tuple[int]`, *可选*，默认值为 `(64,)`):
            块输出通道数的元组。
        act_fn (`str`, *可选*，默认值为 `"silu"`)：使用的激活函数。
        sample_size (`int`, *可选*，默认值为 `32`)：样本输入大小。
        scaling_factor (`float`, *可选*，默认值为 `1.15258426`):
            使用训练集的第一批计算的训练潜在空间的逐分量标准差。用于在训练扩散模型时将潜在空间缩放到单位方差。潜在表示在传递给扩散模型之前使用公式 `z = z * scaling_factor` 进行缩放。在解码时，潜在表示使用公式 `z = 1 / scaling_factor * z` 缩放回原始比例。有关详细信息，请参阅 [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) 论文的第 4.3.2 节和 D.1 节。
        force_upcast (`bool`, *可选*，默认值为 `True`):
            如果启用，它将强制 VAE 在 float32 中运行，以支持高图像分辨率管道，例如 SD-XL。VAE 可以在不失去太多精度的情况下微调/训练到较低范围，在这种情况下可以将 `force_upcast` 设置为 `False` - 参见：https://huggingface.co/madebyollin/sdxl-vae-fp16-fix
    """

    # 设置支持梯度检查点
    _supports_gradient_checkpointing = True
    # 定义不进行拆分的模块列表
    _no_split_modules = ["CogVideoXResnetBlock3D"]

    # 用于将类注册到配置中的装饰器
    @register_to_config
    # 初始化方法，用于设置类的属性
        def __init__(
            # 输入通道数，默认为3
            in_channels: int = 3,
            # 输出通道数，默认为3
            out_channels: int = 3,
            # 下采样块类型的元组
            down_block_types: Tuple[str] = (
                "CogVideoXDownBlock3D",  # 第一个下采样块类型
                "CogVideoXDownBlock3D",  # 第二个下采样块类型
                "CogVideoXDownBlock3D",  # 第三个下采样块类型
                "CogVideoXDownBlock3D",  # 第四个下采样块类型
            ),
            # 上采样块类型的元组
            up_block_types: Tuple[str] = (
                "CogVideoXUpBlock3D",    # 第一个上采样块类型
                "CogVideoXUpBlock3D",    # 第二个上采样块类型
                "CogVideoXUpBlock3D",    # 第三个上采样块类型
                "CogVideoXUpBlock3D",    # 第四个上采样块类型
            ),
            # 每个块的输出通道数的元组
            block_out_channels: Tuple[int] = (128, 256, 256, 512),
            # 潜在通道数，默认为16
            latent_channels: int = 16,
            # 每个块的层数，默认为3
            layers_per_block: int = 3,
            # 激活函数类型，默认为"silu"
            act_fn: str = "silu",
            # 归一化的epsilon，默认为1e-6
            norm_eps: float = 1e-6,
            # 归一化的组数，默认为32
            norm_num_groups: int = 32,
            # 时间压缩比，默认为4
            temporal_compression_ratio: float = 4,
            # 样本高度，默认为480
            sample_height: int = 480,
            # 样本宽度，默认为720
            sample_width: int = 720,
            # 缩放因子，默认为1.15258426
            scaling_factor: float = 1.15258426,
            # 位移因子，默认为None
            shift_factor: Optional[float] = None,
            # 潜在均值，默认为None
            latents_mean: Optional[Tuple[float]] = None,
            # 潜在标准差，默认为None
            latents_std: Optional[Tuple[float]] = None,
            # 强制上位数，默认为True
            force_upcast: float = True,
            # 是否使用量化卷积，默认为False
            use_quant_conv: bool = False,
            # 是否使用后量化卷积，默认为False
            use_post_quant_conv: bool = False,
        # 设置梯度检查点的方法
        def _set_gradient_checkpointing(self, module, value=False):
            # 检查模块是否为特定类型以设置梯度检查点
            if isinstance(module, (CogVideoXEncoder3D, CogVideoXDecoder3D)):
                # 设置模块的梯度检查点标志
                module.gradient_checkpointing = value
    
        # 清理伪上下文并行缓存的方法
        def _clear_fake_context_parallel_cache(self):
            # 遍历所有命名模块
            for name, module in self.named_modules():
                # 检查模块是否为特定类型
                if isinstance(module, CogVideoXCausalConv3d):
                    # 记录清理操作
                    logger.debug(f"Clearing fake Context Parallel cache for layer: {name}")
                    # 清理模块的伪上下文并行缓存
                    module._clear_fake_context_parallel_cache()
    
        # 启用平铺的方法
        def enable_tiling(
            # 平铺样本最小高度，默认为None
            tile_sample_min_height: Optional[int] = None,
            # 平铺样本最小宽度，默认为None
            tile_sample_min_width: Optional[int] = None,
            # 平铺重叠因子高度，默认为None
            tile_overlap_factor_height: Optional[float] = None,
            # 平铺重叠因子宽度，默认为None
            tile_overlap_factor_width: Optional[float] = None,
    # 该方法用于启用分块的 VAE 解码
    ) -> None:
            r"""
            启用分块 VAE 解码。启用后，VAE 将输入张量分割成多个块来进行解码和编码。
            这有助于节省大量内存并允许处理更大图像。
    
            参数：
                tile_sample_min_height (`int`, *可选*):
                    样本在高度维度上分块所需的最小高度。
                tile_sample_min_width (`int`, *可选*):
                    样本在宽度维度上分块所需的最小宽度。
                tile_overlap_factor_height (`int`, *可选*):
                    两个连续垂直块之间的最小重叠量。以确保在高度维度上没有块状伪影。必须在 0 和 1 之间。设置较高的值可能导致处理更多块，从而减慢解码过程。
                tile_overlap_factor_width (`int`, *可选*):
                    两个连续水平块之间的最小重叠量。以确保在宽度维度上没有块状伪影。必须在 0 和 1 之间。设置较高的值可能导致处理更多块，从而减慢解码过程。
            """
            # 启用分块处理
            self.use_tiling = True
            # 设置最小高度，使用提供的值或默认值
            self.tile_sample_min_height = tile_sample_min_height or self.tile_sample_min_height
            # 设置最小宽度，使用提供的值或默认值
            self.tile_sample_min_width = tile_sample_min_width or self.tile_sample_min_width
            # 计算最小潜在高度，根据配置的块通道数调整
            self.tile_latent_min_height = int(
                self.tile_sample_min_height / (2 ** (len(self.config.block_out_channels) - 1))
            )
            # 计算最小潜在宽度，根据配置的块通道数调整
            self.tile_latent_min_width = int(self.tile_sample_min_width / (2 ** (len(self.config.block_out_channels) - 1)))
            # 设置高度重叠因子，使用提供的值或默认值
            self.tile_overlap_factor_height = tile_overlap_factor_height or self.tile_overlap_factor_height
            # 设置宽度重叠因子，使用提供的值或默认值
            self.tile_overlap_factor_width = tile_overlap_factor_width or self.tile_overlap_factor_width
    
        # 该方法用于禁用分块的 VAE 解码
        def disable_tiling(self) -> None:
            r"""
            禁用分块 VAE 解码。如果之前启用了 `enable_tiling`，该方法将返回到一步解码。
            """
            # 将分块处理状态设置为禁用
            self.use_tiling = False
    
        # 该方法用于启用切片的 VAE 解码
        def enable_slicing(self) -> None:
            r"""
            启用切片 VAE 解码。启用后，VAE 将输入张量切割为切片以进行多步解码。
            这有助于节省内存并允许更大的批处理大小。
            """
            # 启用切片处理
            self.use_slicing = True
    
        # 该方法用于禁用切片的 VAE 解码
        def disable_slicing(self) -> None:
            r"""
            禁用切片 VAE 解码。如果之前启用了 `enable_slicing`，该方法将返回到一步解码。
            """
            # 将切片处理状态设置为禁用
            self.use_slicing = False
    # 定义编码函数，输入为一个 Torch 张量，输出为编码后的 Torch 张量
    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        # 获取输入张量的维度信息，包括批大小、通道数、帧数、高度和宽度
        batch_size, num_channels, num_frames, height, width = x.shape
    
        # 检查是否使用切片和输入图像尺寸是否超过最小切片尺寸
        if self.use_tiling and (width > self.tile_sample_min_width or height > self.tile_sample_min_height):
            # 如果条件满足，则调用切片编码函数
            return self.tiled_encode(x)
    
        # 设置每个批次处理的帧数
        frame_batch_size = self.num_sample_frames_batch_size
        # 计算批次数，期望帧数为 1 或批大小的整数倍
        num_batches = num_frames // frame_batch_size if num_frames > 1 else 1
        # 初始化编码结果列表
        enc = []
        # 遍历每个批次
        for i in range(num_batches):
            # 计算剩余帧数
            remaining_frames = num_frames % frame_batch_size
            # 计算当前批次的起始和结束帧索引
            start_frame = frame_batch_size * i + (0 if i == 0 else remaining_frames)
            end_frame = frame_batch_size * (i + 1) + remaining_frames
            # 从输入张量中提取当前批次的帧
            x_intermediate = x[:, :, start_frame:end_frame]
            # 对当前批次进行编码
            x_intermediate = self.encoder(x_intermediate)
            # 如果存在量化卷积，则对结果进行量化
            if self.quant_conv is not None:
                x_intermediate = self.quant_conv(x_intermediate)
            # 将当前批次的编码结果添加到结果列表中
            enc.append(x_intermediate)
    
        # 清除假上下文的并行缓存
        self._clear_fake_context_parallel_cache()
        # 将所有批次的编码结果沿时间维度连接
        enc = torch.cat(enc, dim=2)
    
        # 返回最终的编码结果
        return enc
    
    # 应用前向钩子装饰器
    @apply_forward_hook
    def encode(
        self, x: torch.Tensor, return_dict: bool = True
    ) -> Union[AutoencoderKLOutput, Tuple[DiagonalGaussianDistribution]]:
        """
        将一批图像编码为潜在表示。
    
        参数:
            x (`torch.Tensor`): 输入图像批次。
            return_dict (`bool`, *可选*, 默认为 `True`):
                是否返回 [`~models.autoencoder_kl.AutoencoderKLOutput`] 而不是普通元组。
    
        返回:
            编码视频的潜在表示。如果 `return_dict` 为 True，则返回一个
            [`~models.autoencoder_kl.AutoencoderKLOutput`]，否则返回普通元组。
        """
        # 如果使用切片且输入批次大于 1，进行切片编码
        if self.use_slicing and x.shape[0] > 1:
            # 针对每个切片调用编码函数，并收集结果
            encoded_slices = [self._encode(x_slice) for x_slice in x.split(1)]
            # 将所有切片的结果连接
            h = torch.cat(encoded_slices)
        else:
            # 否则直接编码整个输入
            h = self._encode(x)
    
        # 使用编码结果创建对角高斯分布
        posterior = DiagonalGaussianDistribution(h)
    
        # 根据返回字典标志决定返回结果的类型
        if not return_dict:
            return (posterior,)
        # 返回编码输出对象
        return AutoencoderKLOutput(latent_dist=posterior)
    # 解码给定的潜在张量 z，并选择返回字典或张量格式
    def _decode(self, z: torch.Tensor, return_dict: bool = True) -> Union[DecoderOutput, torch.Tensor]:
        # 获取输入张量的批量大小、通道数、帧数、高度和宽度
        batch_size, num_channels, num_frames, height, width = z.shape

        # 如果启用平铺解码且宽度或高度超过最小平铺尺寸，则调用平铺解码函数
        if self.use_tiling and (width > self.tile_latent_min_width or height > self.tile_latent_min_height):
            return self.tiled_decode(z, return_dict=return_dict)

        # 设置每批潜在帧的大小
        frame_batch_size = self.num_latent_frames_batch_size
        # 计算总的批次数
        num_batches = num_frames // frame_batch_size
        # 创建用于存储解码结果的列表
        dec = []
        # 遍历每个批次
        for i in range(num_batches):
            # 计算剩余帧数
            remaining_frames = num_frames % frame_batch_size
            # 计算当前批次的起始帧和结束帧
            start_frame = frame_batch_size * i + (0 if i == 0 else remaining_frames)
            end_frame = frame_batch_size * (i + 1) + remaining_frames
            # 获取当前批次的潜在张量
            z_intermediate = z[:, :, start_frame:end_frame]
            # 如果存在后量化卷积，则对当前潜在张量进行处理
            if self.post_quant_conv is not None:
                z_intermediate = self.post_quant_conv(z_intermediate)
            # 将潜在张量解码为输出
            z_intermediate = self.decoder(z_intermediate)
            # 将解码结果添加到列表中
            dec.append(z_intermediate)

        # 清除假上下文并行缓存
        self._clear_fake_context_parallel_cache()
        # 将所有解码结果沿着帧维度拼接
        dec = torch.cat(dec, dim=2)

        # 如果不需要返回字典，直接返回解码结果
        if not return_dict:
            return (dec,)

        # 返回解码结果的字典形式
        return DecoderOutput(sample=dec)

    # 应用前向钩子修饰器
    @apply_forward_hook
    def decode(self, z: torch.Tensor, return_dict: bool = True) -> Union[DecoderOutput, torch.Tensor]:
        """
        解码一批图像。

        参数:
            z (`torch.Tensor`): 输入的潜在向量批次。
            return_dict (`bool`, *可选*, 默认为 `True`):
                是否返回 [`~models.vae.DecoderOutput`] 而不是普通元组。

        返回:
            [`~models.vae.DecoderOutput`] 或 `tuple`:
                如果 return_dict 为 True，返回 [`~models.vae.DecoderOutput`]，否则返回普通元组。
        """
        # 如果启用切片解码且输入的批次大小大于 1
        if self.use_slicing and z.shape[0] > 1:
            # 遍历每个切片并解码，收集解码结果
            decoded_slices = [self._decode(z_slice).sample for z_slice in z.split(1)]
            # 将所有解码结果拼接
            decoded = torch.cat(decoded_slices)
        else:
            # 对整个输入进行解码并获取解码样本
            decoded = self._decode(z).sample

        # 如果不需要返回字典，直接返回解码结果
        if not return_dict:
            return (decoded,)
        # 返回解码结果的字典形式
        return DecoderOutput(sample=decoded)

    # 垂直混合两个张量 a 和 b，并指定混合范围
    def blend_v(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
        # 确定混合范围的最小值
        blend_extent = min(a.shape[3], b.shape[3], blend_extent)
        # 在混合范围内遍历每一行
        for y in range(blend_extent):
            # 混合张量 a 的底部与张量 b 的顶部
            b[:, :, :, y, :] = a[:, :, :, -blend_extent + y, :] * (1 - y / blend_extent) + b[:, :, :, y, :] * (
                y / blend_extent
            )
        # 返回混合后的张量 b
        return b

    # 水平混合两个张量 a 和 b，并指定混合范围
    def blend_h(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
        # 确定混合范围的最小值
        blend_extent = min(a.shape[4], b.shape[4], blend_extent)
        # 在混合范围内遍历每一列
        for x in range(blend_extent):
            # 混合张量 a 的右侧与张量 b 的左侧
            b[:, :, :, :, x] = a[:, :, :, :, -blend_extent + x] * (1 - x / blend_extent) + b[:, :, :, :, x] * (
                x / blend_extent
            )
        # 返回混合后的张量 b
        return b
    # 定义前向传播函数，接收输入样本及其他参数
    def forward(
            self,
            sample: torch.Tensor,  # 输入样本，类型为张量
            sample_posterior: bool = False,  # 是否从后验分布中采样，默认值为 False
            return_dict: bool = True,  # 是否以字典形式返回结果，默认值为 True
            generator: Optional[torch.Generator] = None,  # 可选的随机数生成器
        ) -> Union[torch.Tensor, torch.Tensor]:  # 返回类型为张量
            x = sample  # 将输入样本赋值给变量 x
            posterior = self.encode(x).latent_dist  # 编码输入样本并获取后验分布
            if sample_posterior:  # 检查是否需要从后验分布中采样
                z = posterior.sample(generator=generator)  # 从后验分布中采样
            else:
                z = posterior.mode()  # 使用后验分布的众数作为 z 的值
            dec = self.decode(z)  # 解码 z 得到解码结果 dec
            if not return_dict:  # 检查是否需要返回字典形式的结果
                return (dec,)  # 以元组形式返回解码结果
            return dec  # 返回解码结果
```