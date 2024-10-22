# `.\diffusers\pipelines\wuerstchen\modeling_paella_vq_model.py`

```py
# Copyright (c) 2022 Dominic Rampas MIT License
# Copyright 2024 The HuggingFace Team. All rights reserved.
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

# 从 typing 模块导入 Union 类型，用于类型注解
from typing import Union

# 导入 PyTorch 库及其神经网络模块
import torch
import torch.nn as nn

# 从配置相关的工具模块导入 ConfigMixin 和 register_to_config
from ...configuration_utils import ConfigMixin, register_to_config
# 从 VAE 模型中导入 DecoderOutput 和 VectorQuantizer
from ...models.autoencoders.vae import DecoderOutput, VectorQuantizer
# 从模型工具中导入 ModelMixin
from ...models.modeling_utils import ModelMixin
# 从 VQ 模型中导入 VQEncoderOutput
from ...models.vq_model import VQEncoderOutput
# 从加速工具中导入 apply_forward_hook
from ...utils.accelerate_utils import apply_forward_hook


class MixingResidualBlock(nn.Module):
    """
    Residual block with mixing used by Paella's VQ-VAE.
    """  
    # 定义 MixingResidualBlock 类，继承自 nn.Module

    def __init__(self, inp_channels, embed_dim):
        # 构造函数，初始化输入通道数和嵌入维度
        super().__init__()
        # depthwise
        # 对输入通道进行层归一化，设置为不使用可学习的仿射变换，防止除以零的情况
        self.norm1 = nn.LayerNorm(inp_channels, elementwise_affine=False, eps=1e-6)
        # 使用深度可分离卷积，增加卷积的有效性和计算效率
        self.depthwise = nn.Sequential(
            # 对输入进行填充以保持卷积后的尺寸
            nn.ReplicationPad2d(1), 
            # 创建深度可分离卷积层
            nn.Conv2d(inp_channels, inp_channels, kernel_size=3, groups=inp_channels)
        )

        # channelwise
        # 对输入通道进行第二次层归一化
        self.norm2 = nn.LayerNorm(inp_channels, elementwise_affine=False, eps=1e-6)
        # 定义一个全连接层的序列，用于通道混合
        self.channelwise = nn.Sequential(
            # 第一个线性层将输入通道数映射到嵌入维度
            nn.Linear(inp_channels, embed_dim), 
            # 使用 GELU 激活函数
            nn.GELU(), 
            # 第二个线性层将嵌入维度映射回输入通道数
            nn.Linear(embed_dim, inp_channels)
        )

        # 定义可学习的参数 gammas，初始化为零，允许模型在训练中更新这些值
        self.gammas = nn.Parameter(torch.zeros(6), requires_grad=True)

    def forward(self, x):
        # 定义前向传播函数，接收输入 x
        mods = self.gammas  # 获取可学习的 gammas 参数
        # 对输入进行第一层归一化和变换，并应用 gammas[0] 和 mods[1]
        x_temp = self.norm1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2) * (1 + mods[0]) + mods[1]
        # 将经过深度卷积处理的 x_temp 加入原始输入 x，乘以 gammas[2]
        x = x + self.depthwise(x_temp) * mods[2]
        # 对当前的 x 进行第二层归一化和变换，并应用 gammas[3] 和 mods[4]
        x_temp = self.norm2(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2) * (1 + mods[3]) + mods[4]
        # 将经过通道混合处理的 x_temp 加入当前的 x，乘以 gammas[5]
        x = x + self.channelwise(x_temp.permute(0, 2, 3, 1)).permute(0, 3, 1, 2) * mods[5]
        # 返回处理后的 x
        return x


class PaellaVQModel(ModelMixin, ConfigMixin):
    r"""VQ-VAE model from Paella model.

    This model inherits from [`ModelMixin`]. Check the superclass documentation for the generic methods the library
    implements for all the model (such as downloading or saving, etc.)
    # 参数说明部分，描述构造函数的各个参数
    Parameters:
        # 输入图像的通道数，默认为3（RGB图像）
        in_channels (int, *optional*, defaults to 3): Number of channels in the input image.
        # 输出图像的通道数，默认为3（RGB图像）
        out_channels (int,  *optional*, defaults to 3): Number of channels in the output.
        # 输入图像的上下缩放因子，默认为2
        up_down_scale_factor (int, *optional*, defaults to 2): Up and Downscale factor of the input image.
        # 模型中的层数，默认为2
        levels  (int, *optional*, defaults to 2): Number of levels in the model.
        # 模型中的瓶颈块数，默认为12
        bottleneck_blocks (int, *optional*, defaults to 12): Number of bottleneck blocks in the model.
        # 模型中隐藏通道的数量，默认为384
        embed_dim (int, *optional*, defaults to 384): Number of hidden channels in the model.
        # VQ-VAE模型中的潜在通道数量，默认为4
        latent_channels (int, *optional*, defaults to 4): Number of latent channels in the VQ-VAE model.
        # VQ-VAE中的代码簿向量数量，默认为8192
        num_vq_embeddings (int, *optional*, defaults to 8192): Number of codebook vectors in the VQ-VAE.
        # 潜在空间的缩放因子，默认为0.3764
        scale_factor (float, *optional*, defaults to 0.3764): Scaling factor of the latent space.
    """

    # 初始化方法的装饰器，用于注册配置
    @register_to_config
    # 构造函数，定义模型初始化参数及其默认值
    def __init__(
        # 输入图像的通道数，默认为3
        self,
        in_channels: int = 3,
        # 输出图像的通道数，默认为3
        out_channels: int = 3,
        # 上下缩放因子，默认为2
        up_down_scale_factor: int = 2,
        # 模型的层数，默认为2
        levels: int = 2,
        # 瓶颈块的数量，默认为12
        bottleneck_blocks: int = 12,
        # 隐藏通道的数量，默认为384
        embed_dim: int = 384,
        # 潜在通道的数量，默认为4
        latent_channels: int = 4,
        # VQ-VAE中的代码簿向量数量，默认为8192
        num_vq_embeddings: int = 8192,
        # 潜在空间的缩放因子，默认为0.3764
        scale_factor: float = 0.3764,
    ):
        # 调用父类的初始化方法
        super().__init__()

        # 计算每个层级的通道数，使用倒序以便后续操作
        c_levels = [embed_dim // (2**i) for i in reversed(range(levels))]
        # 创建编码器块
        self.in_block = nn.Sequential(
            # 像素不规则拆分，改变输入的空间分辨率
            nn.PixelUnshuffle(up_down_scale_factor),
            # 1x1卷积，将输入通道数变换为第一个层级的通道数
            nn.Conv2d(in_channels * up_down_scale_factor**2, c_levels[0], kernel_size=1),
        )
        down_blocks = []  # 初始化下采样块列表
        for i in range(levels):  # 遍历每一层级
            if i > 0:  # 如果不是第一层级
                # 添加卷积层，用于下采样，改变通道数
                down_blocks.append(nn.Conv2d(c_levels[i - 1], c_levels[i], kernel_size=4, stride=2, padding=1))
            # 创建混合残差块，增加网络深度
            block = MixingResidualBlock(c_levels[i], c_levels[i] * 4)
            down_blocks.append(block)  # 添加残差块到下采样列表
        down_blocks.append(
            nn.Sequential(
                # 1x1卷积，将最后一层的通道数转变为潜在通道数
                nn.Conv2d(c_levels[-1], latent_channels, kernel_size=1, bias=False),
                # 批归一化，确保数据均值为0，方差为1
                nn.BatchNorm2d(latent_channels),  # then normalize them to have mean 0 and std 1
            )
        )
        # 将下采样块列表封装成序列
        self.down_blocks = nn.Sequential(*down_blocks)

        # 向量量化器，使用指定数量的嵌入向量
        self.vquantizer = VectorQuantizer(num_vq_embeddings, vq_embed_dim=latent_channels, legacy=False, beta=0.25)

        # 创建解码器块
        up_blocks = [nn.Sequential(nn.Conv2d(latent_channels, c_levels[-1], kernel_size=1))]  # 第一层解码
        for i in range(levels):  # 遍历每一层级
            for j in range(bottleneck_blocks if i == 0 else 1):  # 添加瓶颈块
                block = MixingResidualBlock(c_levels[levels - 1 - i], c_levels[levels - 1 - i] * 4)
                up_blocks.append(block)  # 添加混合残差块到上采样列表
            if i < levels - 1:  # 如果不是最后一层级
                up_blocks.append(
                    nn.ConvTranspose2d(
                        # 转置卷积层，用于上采样，改变通道数
                        c_levels[levels - 1 - i], c_levels[levels - 2 - i], kernel_size=4, stride=2, padding=1
                    )
                )
        # 将上采样块列表封装成序列
        self.up_blocks = nn.Sequential(*up_blocks)
        self.out_block = nn.Sequential(
            # 1x1卷积，将第一层的通道数变为输出通道数
            nn.Conv2d(c_levels[0], out_channels * up_down_scale_factor**2, kernel_size=1),
            # 像素重排，恢复到原始的空间分辨率
            nn.PixelShuffle(up_down_scale_factor),
        )

    # 应用前向钩子，定义编码过程
    @apply_forward_hook
    def encode(self, x: torch.Tensor, return_dict: bool = True) -> VQEncoderOutput:
        # 通过输入块处理输入数据
        h = self.in_block(x)
        # 通过下采样块处理数据
        h = self.down_blocks(h)

        # 如果不需要返回字典形式
        if not return_dict:
            return (h,)

        # 返回 VQ 编码输出，包含潜在表示
        return VQEncoderOutput(latents=h)

    # 应用前向钩子，定义解码过程
    @apply_forward_hook
    def decode(
        self, h: torch.Tensor, force_not_quantize: bool = True, return_dict: bool = True
    ) -> Union[DecoderOutput, torch.Tensor]:
        # 如果不强制不量化，使用向量量化器
        if not force_not_quantize:
            quant, _, _ = self.vquantizer(h)
        else:
            # 否则直接使用输入作为量化结果
            quant = h

        # 通过上采样块处理量化结果
        x = self.up_blocks(quant)
        # 通过输出块生成最终解码结果
        dec = self.out_block(x)
        # 如果不需要返回字典形式
        if not return_dict:
            return (dec,)

        # 返回解码输出，包含样本数据
        return DecoderOutput(sample=dec)
    # 定义一个前向传播的方法，接受输入样本并选择返回格式
    def forward(self, sample: torch.Tensor, return_dict: bool = True) -> Union[DecoderOutput, torch.Tensor]:
        # 文档字符串，描述参数及其用途
        r"""
        Args:
            sample (`torch.Tensor`): Input sample.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`DecoderOutput`] instead of a plain tuple.
        """
        # 将输入样本赋值给变量 x
        x = sample
        # 对输入样本进行编码，提取潜在变量
        h = self.encode(x).latents
        # 对潜在变量进行解码，获取样本
        dec = self.decode(h).sample
    
        # 如果不返回字典格式
        if not return_dict:
            # 返回解码样本作为元组
            return (dec,)
    
        # 返回包含解码样本的 DecoderOutput 对象
        return DecoderOutput(sample=dec)
```