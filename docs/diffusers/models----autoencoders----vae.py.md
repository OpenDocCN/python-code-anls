# `.\diffusers\models\autoencoders\vae.py`

```py
# 版权声明，2024年HuggingFace团队保留所有权利
# 
# 根据Apache许可证第2.0版（“许可证”）授权；
# 除非遵循许可证，否则您不得使用此文件。
# 您可以在以下网址获取许可证副本：
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是按“原样”基础提供的，
# 不提供任何形式的保证或条件，无论是明示或暗示的。
# 请参阅许可证以获取有关特定语言的权限和限制的更多信息。
from dataclasses import dataclass  # 导入dataclass装饰器用于简化类的定义
from typing import Optional, Tuple  # 导入可选类型和元组类型，用于类型注释

import numpy as np  # 导入numpy库，用于数值计算
import torch  # 导入PyTorch库，用于构建深度学习模型
import torch.nn as nn  # 导入PyTorch的神经网络模块

from ...utils import BaseOutput, is_torch_version  # 从utils模块导入BaseOutput类和版本检查函数
from ...utils.torch_utils import randn_tensor  # 从torch_utils模块导入随机张量生成函数
from ..activations import get_activation  # 从activations模块导入获取激活函数的函数
from ..attention_processor import SpatialNorm  # 从attention_processor模块导入空间归一化类
from ..unets.unet_2d_blocks import (  # 从unet_2d_blocks模块导入多个网络块
    AutoencoderTinyBlock,  # 导入自动编码器小块
    UNetMidBlock2D,  # 导入UNet中间块
    get_down_block,  # 导入获取下采样块的函数
    get_up_block,  # 导入获取上采样块的函数
)

@dataclass  # 使用dataclass装饰器，简化类的初始化和表示
class DecoderOutput(BaseOutput):  # 定义DecoderOutput类，继承自BaseOutput
    r"""  # 文档字符串，描述解码方法的输出
    Output of decoding method.  # 解码方法的输出

    Args:  # 参数说明
        sample (`torch.Tensor` of shape `(batch_size, num_channels, height, width)`):  # 输入样本的描述
            The decoded output sample from the last layer of the model.  # 模型最后一层的解码输出样本
    """

    sample: torch.Tensor  # 定义样本属性，类型为torch.Tensor
    commit_loss: Optional[torch.FloatTensor] = None  # 定义可选的损失属性，默认为None

class Encoder(nn.Module):  # 定义Encoder类，继承自nn.Module
    r"""  # 文档字符串，描述变分自动编码器的Encoder层
    The `Encoder` layer of a variational autoencoder that encodes its input into a latent representation.  # 变分自动编码器的Encoder层

    Args:  # 参数说明
        in_channels (`int`, *optional*, defaults to 3):  # 输入通道的描述
            The number of input channels.  # 输入通道的数量
        out_channels (`int`, *optional*, defaults to 3):  # 输出通道的描述
            The number of output channels.  # 输出通道的数量
        down_block_types (`Tuple[str, ...]`, *optional*, defaults to `("DownEncoderBlock2D",)`):  # 下采样块类型的描述
            The types of down blocks to use. See `~diffusers.models.unet_2d_blocks.get_down_block` for available  # 使用的下采样块类型，具体可查看相关文档
            options.  # 可选项
        block_out_channels (`Tuple[int, ...]`, *optional*, defaults to `(64,)`):  # 块输出通道的描述
            The number of output channels for each block.  # 每个块的输出通道数量
        layers_per_block (`int`, *optional*, defaults to 2):  # 每个块层数的描述
            The number of layers per block.  # 每个块的层数
        norm_num_groups (`int`, *optional*, defaults to 32):  # 归一化组数的描述
            The number of groups for normalization.  # 归一化的组数
        act_fn (`str`, *optional*, defaults to `"silu"`):  # 激活函数类型的描述
            The activation function to use. See `~diffusers.models.activations.get_activation` for available options.  # 使用的激活函数，具体可查看相关文档
        double_z (`bool`, *optional*, defaults to `True`):  # 最后一个块输出通道双倍化的描述
            Whether to double the number of output channels for the last block.  # 是否将最后一个块的输出通道数量翻倍
    """
    # 初始化方法，设置网络的基本参数
        def __init__(
            self,
            in_channels: int = 3,  # 输入通道数，默认为3（RGB图像）
            out_channels: int = 3,  # 输出通道数，默认为3（RGB图像）
            down_block_types: Tuple[str, ...] = ("DownEncoderBlock2D",),  # 下采样模块类型
            block_out_channels: Tuple[int, ...] = (64,),  # 每个块的输出通道数
            layers_per_block: int = 2,  # 每个下采样块的层数
            norm_num_groups: int = 32,  # 归一化时的组数
            act_fn: str = "silu",  # 激活函数类型，默认为SiLU
            double_z: bool = True,  # 是否双输出通道
            mid_block_add_attention=True,  # 中间块是否添加注意力机制
        ):
            super().__init__()  # 调用父类构造函数
            self.layers_per_block = layers_per_block  # 保存每块的层数
    
            # 定义输入卷积层
            self.conv_in = nn.Conv2d(
                in_channels,  # 输入通道数
                block_out_channels[0],  # 输出通道数
                kernel_size=3,  # 卷积核大小
                stride=1,  # 步幅
                padding=1,  # 填充
            )
    
            self.down_blocks = nn.ModuleList([])  # 初始化下采样块的模块列表
    
            # down
            output_channel = block_out_channels[0]  # 初始化输出通道数
            for i, down_block_type in enumerate(down_block_types):  # 遍历下采样块类型
                input_channel = output_channel  # 当前块的输入通道数
                output_channel = block_out_channels[i]  # 当前块的输出通道数
                is_final_block = i == len(block_out_channels) - 1  # 判断是否为最后一个块
    
                # 获取下采样块并初始化
                down_block = get_down_block(
                    down_block_type,  # 下采样块类型
                    num_layers=self.layers_per_block,  # 下采样块的层数
                    in_channels=input_channel,  # 输入通道数
                    out_channels=output_channel,  # 输出通道数
                    add_downsample=not is_final_block,  # 是否添加下采样
                    resnet_eps=1e-6,  # ResNet的epsilon值
                    downsample_padding=0,  # 下采样的填充
                    resnet_act_fn=act_fn,  # ResNet的激活函数
                    resnet_groups=norm_num_groups,  # ResNet的组数
                    attention_head_dim=output_channel,  # 注意力头的维度
                    temb_channels=None,  # 时间嵌入通道数
                )
                self.down_blocks.append(down_block)  # 将下采样块添加到模块列表中
    
            # mid
            # 定义中间块
            self.mid_block = UNetMidBlock2D(
                in_channels=block_out_channels[-1],  # 中间块的输入通道数
                resnet_eps=1e-6,  # ResNet的epsilon值
                resnet_act_fn=act_fn,  # ResNet的激活函数
                output_scale_factor=1,  # 输出缩放因子
                resnet_time_scale_shift="default",  # ResNet时间缩放偏移
                attention_head_dim=block_out_channels[-1],  # 注意力头的维度
                resnet_groups=norm_num_groups,  # ResNet的组数
                temb_channels=None,  # 时间嵌入通道数
                add_attention=mid_block_add_attention,  # 是否添加注意力机制
            )
    
            # out
            # 定义输出的归一化层
            self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[-1], num_groups=norm_num_groups, eps=1e-6)
            self.conv_act = nn.SiLU()  # 激活函数层
    
            # 根据双输出设置卷积层的输出通道数
            conv_out_channels = 2 * out_channels if double_z else out_channels  
            # 定义输出卷积层
            self.conv_out = nn.Conv2d(block_out_channels[-1], conv_out_channels, 3, padding=1)  
    
            self.gradient_checkpointing = False  # 设置梯度检查点为False
    # 定义 Encoder 类的前向传播方法，接收一个张量作为输入并返回一个张量
    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        r"""Encoder 类的前向方法。"""

        # 使用输入的样本通过初始卷积层进行处理
        sample = self.conv_in(sample)

        # 如果处于训练模式并且开启了梯度检查点
        if self.training and self.gradient_checkpointing:

            # 定义一个创建自定义前向传播的内部函数
            def create_custom_forward(module):
                # 定义自定义前向传播函数，接受任意输入并调用模块
                def custom_forward(*inputs):
                    return module(*inputs)

                return custom_forward

            # 如果 PyTorch 版本大于等于 1.11.0
            if is_torch_version(">=", "1.11.0"):
                # 遍历每个下采样块，应用检查点机制来节省内存
                for down_block in self.down_blocks:
                    sample = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(down_block), sample, use_reentrant=False
                    )
                # 中间块应用检查点机制
                sample = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(self.mid_block), sample, use_reentrant=False
                )
            else:
                # 遍历每个下采样块，应用检查点机制
                for down_block in self.down_blocks:
                    sample = torch.utils.checkpoint.checkpoint(create_custom_forward(down_block), sample)
                # 中间块应用检查点机制
                sample = torch.utils.checkpoint.checkpoint(create_custom_forward(self.mid_block), sample)

        else:
            # 否则，直接通过下采样块处理样本
            for down_block in self.down_blocks:
                sample = down_block(sample)

            # 直接通过中间块处理样本
            sample = self.mid_block(sample)

        # 后处理步骤
        sample = self.conv_norm_out(sample)  # 应用归一化卷积层
        sample = self.conv_act(sample)        # 应用激活函数卷积层
        sample = self.conv_out(sample)        # 应用输出卷积层

        # 返回处理后的样本
        return sample
# 定义变分自编码器的解码层，将潜在表示解码为输出样本
class Decoder(nn.Module):
    r"""
    `Decoder`层的文档字符串，描述其功能及参数

    Args:
        in_channels (`int`, *optional*, defaults to 3):
            输入通道的数量
        out_channels (`int`, *optional*, defaults to 3):
            输出通道的数量
        up_block_types (`Tuple[str, ...]`, *optional*, defaults to `("UpDecoderBlock2D",)`):
            使用的上采样块类型，参考可用选项
        block_out_channels (`Tuple[int, ...]`, *optional*, defaults to `(64,)`):
            每个块的输出通道数量
        layers_per_block (`int`, *optional*, defaults to 2):
            每个块包含的层数
        norm_num_groups (`int`, *optional*, defaults to 32):
            归一化的组数
        act_fn (`str`, *optional*, defaults to `"silu"`):
            使用的激活函数，参考可用选项
        norm_type (`str`, *optional*, defaults to `"group"`):
            使用的归一化类型，可以是"group"或"spatial"
    """

    # 初始化解码器层
    def __init__(
        self,
        in_channels: int = 3,  # 默认输入通道为3
        out_channels: int = 3,  # 默认输出通道为3
        up_block_types: Tuple[str, ...] = ("UpDecoderBlock2D",),  # 默认上采样块类型
        block_out_channels: Tuple[int, ...] = (64,),  # 默认块输出通道
        layers_per_block: int = 2,  # 每个块的默认层数
        norm_num_groups: int = 32,  # 默认归一化组数
        act_fn: str = "silu",  # 默认激活函数为"silu"
        norm_type: str = "group",  # 默认归一化类型为"group"
        mid_block_add_attention=True,  # 中间块是否添加注意力机制，默认为True
    # 初始化父类
        ):
            super().__init__()
            # 设置每个块的层数
            self.layers_per_block = layers_per_block
    
            # 定义输入卷积层，转换输入通道数到最后块的输出通道数
            self.conv_in = nn.Conv2d(
                in_channels,
                block_out_channels[-1],
                kernel_size=3,
                stride=1,
                padding=1,
            )
    
            # 初始化上采样块的模块列表
            self.up_blocks = nn.ModuleList([])
    
            # 根据归一化类型设置时间嵌入通道数
            temb_channels = in_channels if norm_type == "spatial" else None
    
            # 中间块
            self.mid_block = UNetMidBlock2D(
                # 设置中间块的输入通道、eps、激活函数等参数
                in_channels=block_out_channels[-1],
                resnet_eps=1e-6,
                resnet_act_fn=act_fn,
                output_scale_factor=1,
                resnet_time_scale_shift="default" if norm_type == "group" else norm_type,
                attention_head_dim=block_out_channels[-1],
                resnet_groups=norm_num_groups,
                temb_channels=temb_channels,
                add_attention=mid_block_add_attention,
            )
    
            # 上采样
            # 反转输出通道数列表以用于上采样
            reversed_block_out_channels = list(reversed(block_out_channels))
            # 设定当前输出通道数为反转列表的第一个元素
            output_channel = reversed_block_out_channels[0]
            # 遍历上采样块类型并创建相应的上采样块
            for i, up_block_type in enumerate(up_block_types):
                # 存储前一个输出通道数
                prev_output_channel = output_channel
                # 更新当前输出通道数
                output_channel = reversed_block_out_channels[i]
    
                # 判断当前块是否为最后一个块
                is_final_block = i == len(block_out_channels) - 1
    
                # 创建上采样块并传入相关参数
                up_block = get_up_block(
                    up_block_type,
                    num_layers=self.layers_per_block + 1,
                    in_channels=prev_output_channel,
                    out_channels=output_channel,
                    prev_output_channel=None,
                    add_upsample=not is_final_block,
                    resnet_eps=1e-6,
                    resnet_act_fn=act_fn,
                    resnet_groups=norm_num_groups,
                    attention_head_dim=output_channel,
                    temb_channels=temb_channels,
                    resnet_time_scale_shift=norm_type,
                )
                # 将新创建的上采样块添加到模块列表
                self.up_blocks.append(up_block)
                # 更新前一个输出通道数
                prev_output_channel = output_channel
    
            # 输出层
            # 根据归一化类型选择输出卷积层的归一化方法
            if norm_type == "spatial":
                self.conv_norm_out = SpatialNorm(block_out_channels[0], temb_channels)
            else:
                self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[0], num_groups=norm_num_groups, eps=1e-6)
            # 设置输出卷积层的激活函数为 SiLU
            self.conv_act = nn.SiLU()
            # 定义最终输出的卷积层，输出通道数为 out_channels
            self.conv_out = nn.Conv2d(block_out_channels[0], out_channels, 3, padding=1)
    
            # 初始化梯度检查点开关为 False
            self.gradient_checkpointing = False
    
        # 定义前向传播方法
        def forward(
            self,
            sample: torch.Tensor,
            latent_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # `Decoder` 类的前向方法文档字符串

        # 通过输入卷积层处理样本
        sample = self.conv_in(sample)

        # 获取上采样块参数的数据类型
        upscale_dtype = next(iter(self.up_blocks.parameters())).dtype
        # 如果处于训练模式且使用梯度检查点
        if self.training and self.gradient_checkpointing:

            # 创建自定义前向函数
            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)

                return custom_forward

            # 检查 PyTorch 版本
            if is_torch_version(">=", "1.11.0"):
                # 中间处理
                sample = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(self.mid_block),
                    sample,
                    latent_embeds,
                    use_reentrant=False,
                )
                # 转换样本数据类型
                sample = sample.to(upscale_dtype)

                # 上采样处理
                for up_block in self.up_blocks:
                    sample = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(up_block),
                        sample,
                        latent_embeds,
                        use_reentrant=False,
                    )
            else:
                # 中间处理
                sample = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(self.mid_block), sample, latent_embeds
                )
                # 转换样本数据类型
                sample = sample.to(upscale_dtype)

                # 上采样处理
                for up_block in self.up_blocks:
                    sample = torch.utils.checkpoint.checkpoint(create_custom_forward(up_block), sample, latent_embeds)
        else:
            # 中间处理
            sample = self.mid_block(sample, latent_embeds)
            # 转换样本数据类型
            sample = sample.to(upscale_dtype)

            # 上采样处理
            for up_block in self.up_blocks:
                sample = up_block(sample, latent_embeds)

        # 后处理
        if latent_embeds is None:
            # 如果没有潜在嵌入，则直接进行卷积归一化输出
            sample = self.conv_norm_out(sample)
        else:
            # 如果有潜在嵌入，则传入进行卷积归一化输出
            sample = self.conv_norm_out(sample, latent_embeds)
        # 应用激活函数
        sample = self.conv_act(sample)
        # 最终输出卷积层处理样本
        sample = self.conv_out(sample)

        # 返回处理后的样本
        return sample
# 定义一个名为 UpSample 的类，继承自 nn.Module
class UpSample(nn.Module):
    r"""
    `UpSample` 层用于变分自编码器，可以对输入进行上采样。

    参数:
        in_channels (`int`, *可选*, 默认为 3):
            输入通道的数量。
        out_channels (`int`, *可选*, 默认为 3):
            输出通道的数量。
    """

    # 初始化方法，接受输入和输出通道数量
    def __init__(
        self,
        in_channels: int,  # 输入通道数量
        out_channels: int,  # 输出通道数量
    ) -> None:
        super().__init__()  # 调用父类的初始化方法
        self.in_channels = in_channels  # 保存输入通道数量
        self.out_channels = out_channels  # 保存输出通道数量
        # 创建转置卷积层，用于上采样
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)

    # 前向传播方法
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""`UpSample` 类的前向传播方法。"""
        x = torch.relu(x)  # 对输入应用 ReLU 激活函数
        x = self.deconv(x)  # 通过转置卷积层进行上采样
        return x  # 返回上采样后的结果


# 定义一个名为 MaskConditionEncoder 的类，继承自 nn.Module
class MaskConditionEncoder(nn.Module):
    """
    用于 AsymmetricAutoencoderKL
    """

    # 初始化方法，接受多个参数以构建编码器
    def __init__(
        self,
        in_ch: int,  # 输入通道数量
        out_ch: int = 192,  # 输出通道数量，默认值为 192
        res_ch: int = 768,  # 结果通道数量，默认值为 768
        stride: int = 16,  # 步幅，默认值为 16
    ) -> None:
        super().__init__()  # 调用父类的初始化方法

        channels = []  # 初始化通道列表
        # 计算每一层的输入和输出通道数量，直到步幅小于等于 1
        while stride > 1:
            stride = stride // 2  # 将步幅减半
            in_ch_ = out_ch * 2  # 输入通道数量为输出通道的两倍
            if out_ch > res_ch:  # 如果输出通道大于结果通道
                out_ch = res_ch  # 将输出通道设置为结果通道
            if stride == 1:  # 如果步幅为 1
                in_ch_ = res_ch  # 输入通道数量设置为结果通道
            channels.append((in_ch_, out_ch))  # 将输入和输出通道对添加到列表
            out_ch *= 2  # 输出通道数量翻倍

        out_channels = []  # 初始化输出通道列表
        # 从通道列表中提取输出通道数量
        for _in_ch, _out_ch in channels:
            out_channels.append(_out_ch)  # 添加输出通道数量
        out_channels.append(channels[-1][0])  # 添加最后一层的输入通道数量

        layers = []  # 初始化层列表
        in_ch_ = in_ch  # 将输入通道数量赋值给临时变量
        # 根据输出通道数量构建卷积层
        for l in range(len(out_channels)):
            out_ch_ = out_channels[l]  # 当前输出通道数量
            if l == 0 or l == 1:  # 对于前两层
                layers.append(nn.Conv2d(in_ch_, out_ch_, kernel_size=3, stride=1, padding=1))  # 添加 3x3 卷积层
            else:  # 对于后续层
                layers.append(nn.Conv2d(in_ch_, out_ch_, kernel_size=4, stride=2, padding=1))  # 添加 4x4 卷积层
            in_ch_ = out_ch_  # 更新输入通道数量

        self.layers = nn.Sequential(*layers)  # 将所有层组合成一个顺序容器

    # 前向传播方法
    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        r"""`MaskConditionEncoder` 类的前向传播方法。"""
        out = {}  # 初始化输出字典
        # 遍历所有层
        for l in range(len(self.layers)):
            layer = self.layers[l]  # 获取当前层
            x = layer(x)  # 通过当前层处理输入
            out[str(tuple(x.shape))] = x  # 将当前输出的形状作为键，输出张量作为值存入字典
            x = torch.relu(x)  # 对输出应用 ReLU 激活函数
        return out  # 返回输出字典


# 定义一个名为 MaskConditionDecoder 的类，继承自 nn.Module
class MaskConditionDecoder(nn.Module):
    r"""`MaskConditionDecoder` 应与 [`AsymmetricAutoencoderKL`] 一起使用，以增强模型的
    解码器，结合掩膜和被掩膜的图像。
    # 函数参数定义部分
        Args:
            in_channels (`int`, *optional*, defaults to 3):  # 输入通道的数量，默认为3
                The number of input channels.
            out_channels (`int`, *optional*, defaults to 3):  # 输出通道的数量，默认为3
                The number of output channels.
            up_block_types (`Tuple[str, ...]`, *optional*, defaults to `("UpDecoderBlock2D",)`):  # 使用的上采样模块类型，默认为UpDecoderBlock2D
                The types of up blocks to use. See `~diffusers.models.unet_2d_blocks.get_up_block` for available options.
            block_out_channels (`Tuple[int, ...]`, *optional*, defaults to `(64,)`):  # 每个模块的输出通道数量，默认为64
                The number of output channels for each block.
            layers_per_block (`int`, *optional*, defaults to 2):  # 每个模块的层数，默认为2
                The number of layers per block.
            norm_num_groups (`int`, *optional*, defaults to 32):  # 归一化的组数，默认为32
                The number of groups for normalization.
            act_fn (`str`, *optional*, defaults to `"silu"`):  # 使用的激活函数，默认为silu
                The activation function to use. See `~diffusers.models.activations.get_activation` for available options.
            norm_type (`str`, *optional*, defaults to `"group"`):  # 归一化类型，可以是"group"或"spatial"，默认为"group"
                The normalization type to use. Can be either `"group"` or `"spatial"`.
        """
    
        # 初始化方法定义
        def __init__(
            self,
            in_channels: int = 3,  # 输入通道的数量，默认为3
            out_channels: int = 3,  # 输出通道的数量，默认为3
            up_block_types: Tuple[str, ...] = ("UpDecoderBlock2D",),  # 上采样模块类型，默认为UpDecoderBlock2D
            block_out_channels: Tuple[int, ...] = (64,),  # 每个模块的输出通道数量，默认为64
            layers_per_block: int = 2,  # 每个模块的层数，默认为2
            norm_num_groups: int = 32,  # 归一化的组数，默认为32
            act_fn: str = "silu",  # 激活函数，默认为silu
            norm_type: str = "group",  # 归一化类型，默认为"group"
    ):
        # 调用父类构造函数初始化
        super().__init__()
        # 设置每个块的层数
        self.layers_per_block = layers_per_block

        # 初始化输入卷积层，接收输入通道并生成块输出通道
        self.conv_in = nn.Conv2d(
            in_channels,
            block_out_channels[-1],
            kernel_size=3,
            stride=1,
            padding=1,
        )

        # 创建一个空的模块列表，用于存储上采样块
        self.up_blocks = nn.ModuleList([])

        # 根据归一化类型设置时间嵌入通道数
        temb_channels = in_channels if norm_type == "spatial" else None

        # 中间块的初始化
        self.mid_block = UNetMidBlock2D(
            in_channels=block_out_channels[-1],  # 使用最后一个块的输出通道作为输入
            resnet_eps=1e-6,                      # ResNet 的 epsilon 参数
            resnet_act_fn=act_fn,                 # ResNet 的激活函数
            output_scale_factor=1,                # 输出缩放因子
            resnet_time_scale_shift="default" if norm_type == "group" else norm_type,  # 时间缩放偏移
            attention_head_dim=block_out_channels[-1],  # 注意力头的维度
            resnet_groups=norm_num_groups,        # ResNet 的组数
            temb_channels=temb_channels,          # 时间嵌入通道数
        )

        # 初始化上采样块
        reversed_block_out_channels = list(reversed(block_out_channels))  # 反转输出通道列表
        output_channel = reversed_block_out_channels[0]  # 获取第一个输出通道
        for i, up_block_type in enumerate(up_block_types):  # 遍历上采样块类型
            prev_output_channel = output_channel  # 保存上一个输出通道数
            output_channel = reversed_block_out_channels[i]  # 获取当前输出通道数

            is_final_block = i == len(block_out_channels) - 1  # 判断是否为最后一个块

            # 获取上采样块
            up_block = get_up_block(
                up_block_type,  # 上采样块类型
                num_layers=self.layers_per_block + 1,  # 上采样层数
                in_channels=prev_output_channel,  # 输入通道数
                out_channels=output_channel,  # 输出通道数
                prev_output_channel=None,  # 前一个输出通道数
                add_upsample=not is_final_block,  # 是否添加上采样操作
                resnet_eps=1e-6,  # ResNet 的 epsilon 参数
                resnet_act_fn=act_fn,  # ResNet 的激活函数
                resnet_groups=norm_num_groups,  # ResNet 的组数
                attention_head_dim=output_channel,  # 注意力头的维度
                temb_channels=temb_channels,  # 时间嵌入通道数
                resnet_time_scale_shift=norm_type,  # 时间缩放偏移
            )
            self.up_blocks.append(up_block)  # 将上采样块添加到模块列表
            prev_output_channel = output_channel  # 更新前一个输出通道数

        # 条件编码器的初始化
        self.condition_encoder = MaskConditionEncoder(
            in_ch=out_channels,  # 输入通道数
            out_ch=block_out_channels[0],  # 输出通道数
            res_ch=block_out_channels[-1],  # ResNet 通道数
        )

        # 输出层的归一化处理
        if norm_type == "spatial":  # 如果归一化类型为空间归一化
            self.conv_norm_out = SpatialNorm(block_out_channels[0], temb_channels)  # 初始化空间归一化
        else:  # 否则使用组归一化
            self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[0], num_groups=norm_num_groups, eps=1e-6)  # 初始化组归一化
        # 初始化激活函数为 SiLU
        self.conv_act = nn.SiLU()
        # 初始化输出卷积层
        self.conv_out = nn.Conv2d(block_out_channels[0], out_channels, 3, padding=1)

        # 初始化梯度检查点标志为 False
        self.gradient_checkpointing = False

    def forward(
        self,
        z: torch.Tensor,  # 输入的张量 z
        image: Optional[torch.Tensor] = None,  # 可选的输入图像张量
        mask: Optional[torch.Tensor] = None,  # 可选的输入掩码张量
        latent_embeds: Optional[torch.Tensor] = None,  # 可选的潜在嵌入张量
# 定义一个向量量化器类，继承自 nn.Module
class VectorQuantizer(nn.Module):
    """
    Improved version over VectorQuantizer, can be used as a drop-in replacement. Mostly avoids costly matrix
    multiplications and allows for post-hoc remapping of indices.
    """

    # 初始化方法，设置类的基本参数
    def __init__(
        self,
        n_e: int,  # 向量量化的嵌入数量
        vq_embed_dim: int,  # 嵌入的维度
        beta: float,  # beta 参数，用于调节量化误差
        remap=None,  # 用于重映射的可选参数
        unknown_index: str = "random",  # 未知索引的处理方式
        sane_index_shape: bool = False,  # 是否强制索引形状的合理性
        legacy: bool = True,  # 是否使用旧版本的实现
    ):
        # 调用父类的初始化方法
        super().__init__()
        # 保存嵌入数量
        self.n_e = n_e
        # 保存嵌入维度
        self.vq_embed_dim = vq_embed_dim
        # 保存 beta 参数
        self.beta = beta
        # 保存是否使用旧版的标志
        self.legacy = legacy

        # 初始化嵌入层，随机生成嵌入权重
        self.embedding = nn.Embedding(self.n_e, self.vq_embed_dim)
        # 将嵌入权重初始化为[-1/n_e, 1/n_e]的均匀分布
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

        # 处理重映射参数
        self.remap = remap
        if self.remap is not None:  # 如果提供了重映射文件
            # 注册一个缓冲区，加载重映射的数据
            self.register_buffer("used", torch.tensor(np.load(self.remap)))
            self.used: torch.Tensor  # 声明用于重映射的张量
            # 重新嵌入的数量
            self.re_embed = self.used.shape[0]
            # 设置未知索引的方式
            self.unknown_index = unknown_index  # "random"、"extra"或整数
            if self.unknown_index == "extra":  # 如果未知索引为"extra"
                self.unknown_index = self.re_embed  # 设置为重新嵌入数量
                self.re_embed = self.re_embed + 1  # 增加重新嵌入的数量
            # 打印重映射信息
            print(
                f"Remapping {self.n_e} indices to {self.re_embed} indices. "
                f"Using {self.unknown_index} for unknown indices."
            )
        else:
            # 如果没有提供重映射，则重新嵌入数量与嵌入数量相同
            self.re_embed = n_e

        # 保存是否强制索引形状合理的标志
        self.sane_index_shape = sane_index_shape

    # 将索引映射到已使用的索引
    def remap_to_used(self, inds: torch.LongTensor) -> torch.LongTensor:
        # 保存输入张量的形状
        ishape = inds.shape
        # 确保输入张量至少有两个维度
        assert len(ishape) > 1
        # 将输入张量重塑为二维，保持第一个维度不变
        inds = inds.reshape(ishape[0], -1)
        # 将使用的张量转换到相同的设备
        used = self.used.to(inds)
        # 检查 inds 中的元素是否与 used 中的元素匹配
        match = (inds[:, :, None] == used[None, None, ...]).long()
        # 找到匹配的索引
        new = match.argmax(-1)
        # 检查是否有未知索引
        unknown = match.sum(2) < 1
        if self.unknown_index == "random":  # 如果未知索引为"random"
            # 随机生成未知索引
            new[unknown] = torch.randint(0, self.re_embed, size=new[unknown].shape).to(device=new.device)
        else:
            # 否则设置为指定的未知索引
            new[unknown] = self.unknown_index
        # 将结果重塑为原来的形状并返回
        return new.reshape(ishape)

    # 将映射到的索引还原为所有索引
    def unmap_to_all(self, inds: torch.LongTensor) -> torch.LongTensor:
        # 保存输入张量的形状
        ishape = inds.shape
        # 确保输入张量至少有两个维度
        assert len(ishape) > 1
        # 将输入张量重塑为二维，保持第一个维度不变
        inds = inds.reshape(ishape[0], -1)
        # 将使用的张量转换到相同的设备
        used = self.used.to(inds)
        if self.re_embed > self.used.shape[0]:  # 如果有额外的标记
            # 将超出已使用索引的标记设置为零
            inds[inds >= self.used.shape[0]] = 0  # 简单设置为零
        # 根据 inds 从 used 中选择对应的值
        back = torch.gather(used[None, :][inds.shape[0] * [0], :], 1, inds)
        # 将结果重塑为原来的形状并返回
        return back.reshape(ishape)
    # 前向传播方法，接收一个张量 z，返回量化张量、损失和附加信息
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Tuple]:
        # 将 z 重新排列为 (batch, height, width, channel) 的形状，并展平
        z = z.permute(0, 2, 3, 1).contiguous()
        # 将 z 展平为二维张量，维度为 (batch_size * height * width, vq_embed_dim)
        z_flattened = z.view(-1, self.vq_embed_dim)

        # 计算 z 与嵌入 e_j 之间的距离，公式为 (z - e)^2 = z^2 + e^2 - 2 * e * z
        min_encoding_indices = torch.argmin(torch.cdist(z_flattened, self.embedding.weight), dim=1)

        # 根据最小编码索引从嵌入中获取量化的 z，重新调整为原始 z 的形状
        z_q = self.embedding(min_encoding_indices).view(z.shape)
        perplexity = None  # 初始化困惑度为 None
        min_encodings = None  # 初始化最小编码为 None

        # 计算嵌入的损失
        if not self.legacy:
            # 计算损失时考虑 beta 权重
            loss = self.beta * torch.mean((z_q.detach() - z) ** 2) + torch.mean((z_q - z.detach()) ** 2)
        else:
            # 计算损失时考虑不同的权重
            loss = torch.mean((z_q.detach() - z) ** 2) + self.beta * torch.mean((z_q - z.detach()) ** 2)

        # 保持梯度
        z_q: torch.Tensor = z + (z_q - z).detach()

        # 将 z_q 重新排列为与原始输入形状相匹配
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        if self.remap is not None:
            # 如果存在重映射，则调整最小编码索引形状，增加批次维度
            min_encoding_indices = min_encoding_indices.reshape(z.shape[0], -1)  # add batch axis
            # 将索引映射到使用的编码
            min_encoding_indices = self.remap_to_used(min_encoding_indices)
            # 将索引展平
            min_encoding_indices = min_encoding_indices.reshape(-1, 1)  # flatten

        if self.sane_index_shape:
            # 如果需要，调整最小编码索引的形状以匹配 z_q 的形状
            min_encoding_indices = min_encoding_indices.reshape(z_q.shape[0], z_q.shape[2], z_q.shape[3])

        # 返回量化的 z、损失和其他信息的元组
        return z_q, loss, (perplexity, min_encodings, min_encoding_indices)

    # 获取代码簿条目，根据索引返回量化的潜在向量
    def get_codebook_entry(self, indices: torch.LongTensor, shape: Tuple[int, ...]) -> torch.Tensor:
        # shape 指定 (batch, height, width, channel)
        if self.remap is not None:
            # 如果存在重映射，则调整索引形状，增加批次维度
            indices = indices.reshape(shape[0], -1)  # add batch axis
            # 将索引映射回所有编码
            indices = self.unmap_to_all(indices)
            # 将索引展平
            indices = indices.reshape(-1)  # flatten again

        # 获取量化的潜在向量
        z_q: torch.Tensor = self.embedding(indices)

        if shape is not None:
            # 如果形状不为空，将 z_q 重新调整为指定的形状
            z_q = z_q.view(shape)
            # 重新排列以匹配原始输入形状
            z_q = z_q.permute(0, 3, 1, 2).contiguous()

        # 返回量化的潜在向量
        return z_q
# 定义对角高斯分布类
class DiagonalGaussianDistribution(object):
    # 初始化方法，接收参数和是否为确定性分布的标志
    def __init__(self, parameters: torch.Tensor, deterministic: bool = False):
        # 将参数存储在实例中
        self.parameters = parameters
        # 将参数分为均值和对数方差
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        # 将对数方差限制在-30到20之间
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        # 记录是否为确定性分布
        self.deterministic = deterministic
        # 计算标准差
        self.std = torch.exp(0.5 * self.logvar)
        # 计算方差
        self.var = torch.exp(self.logvar)
        # 如果是确定性分布，方差和标准差设为零
        if self.deterministic:
            self.var = self.std = torch.zeros_like(
                self.mean, device=self.parameters.device, dtype=self.parameters.dtype
            )

    # 采样方法，生成符合分布的样本
    def sample(self, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        # 确保样本与参数在相同的设备上且具有相同的数据类型
        sample = randn_tensor(
            self.mean.shape,
            generator=generator,
            device=self.parameters.device,
            dtype=self.parameters.dtype,
        )
        # 根据均值和标准差生成样本
        x = self.mean + self.std * sample
        # 返回生成的样本
        return x

    # 计算与另一个分布的KL散度
    def kl(self, other: "DiagonalGaussianDistribution" = None) -> torch.Tensor:
        # 如果是确定性分布，KL散度为0
        if self.deterministic:
            return torch.Tensor([0.0])
        else:
            # 如果没有提供另一个分布
            if other is None:
                # 计算自身的KL散度
                return 0.5 * torch.sum(
                    torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar,
                    dim=[1, 2, 3],
                )
            else:
                # 计算与另一个分布的KL散度
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var
                    - 1.0
                    - self.logvar
                    + other.logvar,
                    dim=[1, 2, 3],
                )

    # 计算负对数似然
    def nll(self, sample: torch.Tensor, dims: Tuple[int, ...] = [1, 2, 3]) -> torch.Tensor:
        # 如果是确定性分布，负对数似然为0
        if self.deterministic:
            return torch.Tensor([0.0])
        # 计算常数log(2π)
        logtwopi = np.log(2.0 * np.pi)
        # 返回负对数似然的计算结果
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims,
        )

    # 返回分布的众数
    def mode(self) -> torch.Tensor:
        return self.mean


# 定义EncoderTiny类，继承自nn.Module
class EncoderTiny(nn.Module):
    r"""
    `EncoderTiny`层是`Encoder`层的简化版本。

    参数：
        in_channels (`int`):
            输入通道的数量。
        out_channels (`int`):
            输出通道的数量。
        num_blocks (`Tuple[int, ...]`):
            元组中的每个值表示一个Conv2d层后跟随`value`数量的`AutoencoderTinyBlock`。
        block_out_channels (`Tuple[int, ...]`):
            每个块的输出通道数量。
        act_fn (`str`):
            使用的激活函数。请参见`~diffusers.models.activations.get_activation`以获取可用选项。
    """
    # 初始化方法，构造 EncoderTiny 类的实例
        def __init__(
            self,
            in_channels: int,  # 输入通道数
            out_channels: int,  # 输出通道数
            num_blocks: Tuple[int, ...],  # 每个层中块的数量
            block_out_channels: Tuple[int, ...],  # 每个层的输出通道数
            act_fn: str,  # 激活函数的类型
        ):
            # 调用父类的初始化方法
            super().__init__()
    
            layers = []  # 初始化空层列表
            # 遍历每个层的块数量
            for i, num_block in enumerate(num_blocks):
                num_channels = block_out_channels[i]  # 当前层的输出通道数
    
                # 如果是第一个层，创建卷积层
                if i == 0:
                    layers.append(nn.Conv2d(in_channels, num_channels, kernel_size=3, padding=1))
                else:
                    # 创建后续卷积层，包含步幅和无偏置选项
                    layers.append(
                        nn.Conv2d(
                            num_channels,
                            num_channels,
                            kernel_size=3,
                            padding=1,
                            stride=2,
                            bias=False,
                        )
                    )
    
                # 添加指定数量的 AutoencoderTinyBlock
                for _ in range(num_block):
                    layers.append(AutoencoderTinyBlock(num_channels, num_channels, act_fn))
    
            # 添加最后的卷积层，将最后一层输出通道映射到目标输出通道
            layers.append(nn.Conv2d(block_out_channels[-1], out_channels, kernel_size=3, padding=1))
    
            # 将所有层组合为一个顺序模块
            self.layers = nn.Sequential(*layers)
            # 初始化梯度检查点标志为 False
            self.gradient_checkpointing = False
    
        # 前向传播方法
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            r"""EncoderTiny 类的前向方法。"""
            # 如果模型处于训练状态并且启用了梯度检查点
            if self.training and self.gradient_checkpointing:
    
                # 创建自定义前向传播方法
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)
    
                    return custom_forward
    
                # 根据 PyTorch 版本选择检查点方式
                if is_torch_version(">=", "1.11.0"):
                    x = torch.utils.checkpoint.checkpoint(create_custom_forward(self.layers), x, use_reentrant=False)
                else:
                    x = torch.utils.checkpoint.checkpoint(create_custom_forward(self.layers), x)
    
            else:
                # 将图像从 [-1, 1] 线性缩放到 [0, 1]，以匹配 TAESD 规范
                x = self.layers(x.add(1).div(2))
    
            # 返回前向传播的输出
            return x
# 定义一个名为 `DecoderTiny` 的类，继承自 `nn.Module`
class DecoderTiny(nn.Module):
    r"""
    `DecoderTiny` 层是 `Decoder` 层的简化版本。

    参数:
        in_channels (`int`):
            输入通道的数量。
        out_channels (`int`):
            输出通道的数量。
        num_blocks (`Tuple[int, ...]`):
            元组中的每个值表示一个 Conv2d 层后面跟着 `value` 个 `AutoencoderTinyBlock` 的数量。
        block_out_channels (`Tuple[int, ...]`):
            每个块的输出通道数量。
        upsampling_scaling_factor (`int`):
            用于上采样的缩放因子。
        act_fn (`str`):
            使用的激活函数。有关可用选项，请参见 `~diffusers.models.activations.get_activation`。
    """

    # 初始化方法，设置类的基本参数
    def __init__(
        self,
        in_channels: int,  # 输入通道数量
        out_channels: int,  # 输出通道数量
        num_blocks: Tuple[int, ...],  # 每个块的数量
        block_out_channels: Tuple[int, ...],  # 每个块的输出通道数量
        upsampling_scaling_factor: int,  # 上采样缩放因子
        act_fn: str,  # 激活函数名称
        upsample_fn: str,  # 上采样函数名称
    ):
        super().__init__()  # 调用父类的初始化方法

        # 初始化层的列表
        layers = [
            # 添加一个 Conv2d 层，输入通道到第一个块的输出通道
            nn.Conv2d(in_channels, block_out_channels[0], kernel_size=3, padding=1),
            # 添加指定的激活函数
            get_activation(act_fn),
        ]

        # 遍历每个块的数量
        for i, num_block in enumerate(num_blocks):
            is_final_block = i == (len(num_blocks) - 1)  # 判断是否为最后一个块
            num_channels = block_out_channels[i]  # 获取当前块的输出通道数量

            # 对于当前块的数量，添加相应数量的 `AutoencoderTinyBlock`
            for _ in range(num_block):
                layers.append(AutoencoderTinyBlock(num_channels, num_channels, act_fn))

            # 如果不是最后一个块，则添加上采样层
            if not is_final_block:
                layers.append(nn.Upsample(scale_factor=upsampling_scaling_factor, mode=upsample_fn))

            # 设置当前卷积的输出通道数量
            conv_out_channel = num_channels if not is_final_block else out_channels
            # 添加卷积层
            layers.append(
                nn.Conv2d(
                    num_channels,  # 输入通道数量
                    conv_out_channel,  # 输出通道数量
                    kernel_size=3,  # 卷积核大小
                    padding=1,  # 填充大小
                    bias=is_final_block,  # 如果是最后一个块，使用偏置
                )
            )

        # 将所有层组合成一个顺序模型
        self.layers = nn.Sequential(*layers)
        # 初始化梯度检查点标志为 False
        self.gradient_checkpointing = False

    # 前向传播方法
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""`DecoderTiny` 类的前向方法。"""
        # 将输入张量缩放并限制到 [-3, 3] 范围
        x = torch.tanh(x / 3) * 3

        # 如果处于训练状态并且启用了梯度检查点
        if self.training and self.gradient_checkpointing:

            # 创建自定义前向函数
            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)  # 调用模块

                return custom_forward

            # 如果 PyTorch 版本大于等于 1.11.0，使用非重入的检查点
            if is_torch_version(">=", "1.11.0"):
                x = torch.utils.checkpoint.checkpoint(create_custom_forward(self.layers), x, use_reentrant=False)
            else:
                x = torch.utils.checkpoint.checkpoint(create_custom_forward(self.layers), x)  # 使用检查点

        else:
            x = self.layers(x)  # 否则直接通过层处理输入

        # 将图像从 [0, 1] 范围缩放到 [-1, 1]，以匹配 diffusers 的约定
        return x.mul(2).sub(1)  # 缩放并返回结果
```