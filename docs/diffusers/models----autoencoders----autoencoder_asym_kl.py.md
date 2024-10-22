# `.\diffusers\models\autoencoders\autoencoder_asym_kl.py`

```py
# 版权声明，标识该文件的所有权和使用条款
# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# 根据 Apache 许可证第 2.0 版（“许可证”）进行授权；
# 除非遵循许可证，否则您不得使用此文件。
# 您可以在以下地址获取许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律或书面同意，否则根据许可证分发的软件是以“现状”基础提供的，
# 不提供任何形式的明示或暗示的担保或条件。
# 有关许可证所管理的权限和限制的具体信息，请参见许可证。
from typing import Optional, Tuple, Union  # 导入类型提示模块，用于指定可选类型、元组和联合类型

import torch  # 导入 PyTorch 库，用于深度学习
import torch.nn as nn  # 导入 PyTorch 神经网络模块

from ...configuration_utils import ConfigMixin, register_to_config  # 从配置工具中导入配置混合类和注册函数
from ...utils.accelerate_utils import apply_forward_hook  # 从加速工具中导入前向钩子应用函数
from ..modeling_outputs import AutoencoderKLOutput  # 从建模输出模块导入自编码器 KL 输出类
from ..modeling_utils import ModelMixin  # 从建模工具中导入模型混合类
from .vae import DecoderOutput, DiagonalGaussianDistribution, Encoder, MaskConditionDecoder  # 从 VAE 模块导入解码器输出、对角高斯分布、编码器和掩码条件解码器类


class AsymmetricAutoencoderKL(ModelMixin, ConfigMixin):  # 定义不对称自编码器 KL 类，继承模型混合类和配置混合类
    r"""  # 开始文档字符串，描述模型的用途和背景
    设计一个更好的不对称 VQGAN 以用于 StableDiffusion https://arxiv.org/abs/2306.04632。一个具有 KL 损失的 VAE 模型
    用于将图像编码为潜在表示，并将潜在表示解码为图像。

    此模型继承自 [`ModelMixin`]。请查看超类文档以了解其为所有模型实现的通用方法
    （例如下载或保存）。
    # 参数说明部分，描述类或函数的参数及其默认值
    Parameters:
        # 输入图像的通道数，默认值为 3
        in_channels (int, *optional*, defaults to 3): Number of channels in the input image.
        # 输出的通道数，默认值为 3
        out_channels (int,  *optional*, defaults to 3): Number of channels in the output.
        # 下采样块类型的元组，默认值为包含一个元素的元组
        down_block_types (`Tuple[str]`, *optional*, defaults to `("DownEncoderBlock2D",)`):
            Tuple of downsample block types.
        # 下采样块输出通道的元组，默认值为包含一个元素的元组
        down_block_out_channels (`Tuple[int]`, *optional*, defaults to `(64,)`):
            Tuple of down block output channels.
        # 每个下采样块的层数，默认值为 1
        layers_per_down_block (`int`, *optional*, defaults to `1`):
            Number layers for down block.
        # 上采样块类型的元组，默认值为包含一个元素的元组
        up_block_types (`Tuple[str]`, *optional*, defaults to `("UpDecoderBlock2D",)`):
            Tuple of upsample block types.
        # 上采样块输出通道的元组，默认值为包含一个元素的元组
        up_block_out_channels (`Tuple[int]`, *optional*, defaults to `(64,)`):
            Tuple of up block output channels.
        # 每个上采样块的层数，默认值为 1
        layers_per_up_block (`int`, *optional*, defaults to `1`):
            Number layers for up block.
        # 使用的激活函数，默认值为 "silu"
        act_fn (`str`, *optional*, defaults to `"silu"`): The activation function to use.
        # 潜在空间的通道数，默认值为 4
        latent_channels (`int`, *optional*, defaults to 4): Number of channels in the latent space.
        # 输入样本的大小，默认值为 32
        sample_size (`int`, *optional*, defaults to 32): Sample input size.
        # ResNet 块中第一个归一化层使用的组数，默认值为 32
        norm_num_groups (`int`, *optional*, defaults to 32):
            Number of groups to use for the first normalization layer in ResNet blocks.
        # 训练潜在空间的分量标准差，默认值为 0.18215
        scaling_factor (`float`, *optional*, defaults to 0.18215):
            The component-wise standard deviation of the trained latent space computed using the first batch of the
            training set. This is used to scale the latent space to have unit variance when training the diffusion
            model. The latents are scaled with the formula `z = z * scaling_factor` before being passed to the
            diffusion model. When decoding, the latents are scaled back to the original scale with the formula: `z = 1
            / scaling_factor * z`. For more details, refer to sections 4.3.2 and D.1 of the [High-Resolution Image
            Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) paper.
    """
    # 定义初始化方法，无返回值
    ) -> None:
            # 调用父类初始化方法
            super().__init__()
    
            # 将初始化参数传递给编码器
            self.encoder = Encoder(
                # 输入通道数
                in_channels=in_channels,
                # 潜在通道数
                out_channels=latent_channels,
                # 下采样块类型
                down_block_types=down_block_types,
                # 下采样块输出通道数
                block_out_channels=down_block_out_channels,
                # 每个块的层数
                layers_per_block=layers_per_down_block,
                # 激活函数
                act_fn=act_fn,
                # 归一化的组数
                norm_num_groups=norm_num_groups,
                # 设置双重潜变量
                double_z=True,
            )
    
            # 将初始化参数传递给解码器
            self.decoder = MaskConditionDecoder(
                # 输入潜在通道数
                in_channels=latent_channels,
                # 输出通道数
                out_channels=out_channels,
                # 上采样块类型
                up_block_types=up_block_types,
                # 上采样块输出通道数
                block_out_channels=up_block_out_channels,
                # 每个块的层数
                layers_per_block=layers_per_up_block,
                # 激活函数
                act_fn=act_fn,
                # 归一化的组数
                norm_num_groups=norm_num_groups,
            )
    
            # 定义量化卷积层，输入输出通道数相同
            self.quant_conv = nn.Conv2d(2 * latent_channels, 2 * latent_channels, 1)
            # 定义后量化卷积层，输入输出通道数相同
            self.post_quant_conv = nn.Conv2d(latent_channels, latent_channels, 1)
    
            # 禁用切片功能
            self.use_slicing = False
            # 禁用平铺功能
            self.use_tiling = False
    
            # 注册上采样块输出通道数到配置
            self.register_to_config(block_out_channels=up_block_out_channels)
            # 注册强制上溯参数到配置
            self.register_to_config(force_upcast=False)
    
        # 应用前向钩子修饰符
        @apply_forward_hook
        def encode(self, x: torch.Tensor, return_dict: bool = True) -> Union[AutoencoderKLOutput, Tuple[torch.Tensor]]:
            # 使用编码器处理输入数据
            h = self.encoder(x)
            # 通过量化卷积获取时刻
            moments = self.quant_conv(h)
            # 创建对角高斯分布
            posterior = DiagonalGaussianDistribution(moments)
    
            # 检查是否返回字典
            if not return_dict:
                return (posterior,)
    
            # 返回潜在分布输出
            return AutoencoderKLOutput(latent_dist=posterior)
    
        # 定义解码私有方法
        def _decode(
            self,
            z: torch.Tensor,
            image: Optional[torch.Tensor] = None,
            mask: Optional[torch.Tensor] = None,
            return_dict: bool = True,
        ) -> Union[DecoderOutput, Tuple[torch.Tensor]]:
            # 通过后量化卷积处理潜在变量
            z = self.post_quant_conv(z)
            # 使用解码器生成输出
            dec = self.decoder(z, image, mask)
    
            # 检查是否返回字典
            if not return_dict:
                return (dec,)
    
            # 返回解码器输出
            return DecoderOutput(sample=dec)
    
        # 应用前向钩子修饰符
        @apply_forward_hook
        def decode(
            self,
            z: torch.Tensor,
            generator: Optional[torch.Generator] = None,
            image: Optional[torch.Tensor] = None,
            mask: Optional[torch.Tensor] = None,
            return_dict: bool = True,
        ) -> Union[DecoderOutput, Tuple[torch.Tensor]]:
            # 调用解码私有方法并获取样本
            decoded = self._decode(z, image, mask).sample
    
            # 检查是否返回字典
            if not return_dict:
                return (decoded,)
    
            # 返回解码器输出
            return DecoderOutput(sample=decoded)
    
        # 定义前向传播方法
        def forward(
            self,
            sample: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
            sample_posterior: bool = False,
            return_dict: bool = True,
            generator: Optional[torch.Generator] = None,
    # 定义一个函数，返回类型为解码输出或包含张量的元组
    ) -> Union[DecoderOutput, Tuple[torch.Tensor]]:
        # 函数文档字符串，描述输入参数的含义
        r"""
        Args:
            sample (`torch.Tensor`): 输入样本。
            mask (`torch.Tensor`, *optional*, defaults to `None`): 可选的修补掩码。
            sample_posterior (`bool`, *optional*, defaults to `False`):
                是否从后验分布中采样。
            return_dict (`bool`, *optional*, defaults to `True`):
                是否返回解码输出而不是普通元组。
        """
        # 将输入样本赋值给变量 x
        x = sample
        # 对输入样本进行编码，获取潜在分布
        posterior = self.encode(x).latent_dist
        # 根据标志决定是从后验分布中采样还是使用众数
        if sample_posterior:
            z = posterior.sample(generator=generator)
        else:
            z = posterior.mode()
        # 解码潜在变量 z，并获取样本
        dec = self.decode(z, generator, sample, mask).sample
    
        # 检查是否返回字典格式的输出
        if not return_dict:
            # 如果不返回字典，则返回解码样本的元组
            return (dec,)
    
        # 返回解码输出的实例
        return DecoderOutput(sample=dec)
```