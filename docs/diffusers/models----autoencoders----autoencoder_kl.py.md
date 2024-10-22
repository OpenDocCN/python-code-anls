# `.\diffusers\models\autoencoders\autoencoder_kl.py`

```
# 版权声明，表明此文件的版权所有者及其所有权利
# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# 根据 Apache 许可证 2.0 版本进行许可，声明该文件使用条件
# Licensed under the Apache License, Version 2.0 (the "License");
# 只能在符合许可证的情况下使用该文件
# you may not use this file except in compliance with the License.
# 可以在此网址获取许可证副本
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律或书面协议另有约定，软件按 "原样" 提供，且不附带任何形式的保证
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# 不承担任何明示或暗示的担保或条件
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# 查看许可证以了解特定权限和限制
# See the License for the specific language governing permissions and
# limitations under the License.
# 导入所需的类型定义
from typing import Dict, Optional, Tuple, Union

# 导入 PyTorch 库
import torch
import torch.nn as nn

# 导入其他模块中的混合类和工具函数
from ...configuration_utils import ConfigMixin, register_to_config
from ...loaders.single_file_model import FromOriginalModelMixin
from ...utils.accelerate_utils import apply_forward_hook
# 导入注意力处理器相关的类和常量
from ..attention_processor import (
    ADDED_KV_ATTENTION_PROCESSORS,
    CROSS_ATTENTION_PROCESSORS,
    Attention,
    AttentionProcessor,
    AttnAddedKVProcessor,
    AttnProcessor,
    FusedAttnProcessor2_0,
)
# 导入模型输出相关的类
from ..modeling_outputs import AutoencoderKLOutput
# 导入模型工具类
from ..modeling_utils import ModelMixin
# 导入变分自编码器相关的类
from .vae import Decoder, DecoderOutput, DiagonalGaussianDistribution, Encoder

# 定义一个变分自编码器模型，使用 KL 损失编码图像到潜在空间并解码
class AutoencoderKL(ModelMixin, ConfigMixin, FromOriginalModelMixin):
    r"""
    一个带有 KL 损失的 VAE 模型，用于将图像编码为潜在表示，并将潜在表示解码为图像。

    该模型继承自 [`ModelMixin`]。查看超类文档以了解其实现的通用方法
    适用于所有模型（例如下载或保存）。
    # 参数说明
        Parameters:
            # 输入图像的通道数，默认为 3
            in_channels (int, *optional*, defaults to 3): Number of channels in the input image.
            # 输出的通道数，默认为 3
            out_channels (int,  *optional*, defaults to 3): Number of channels in the output.
            # 下采样块类型的元组，默认为 ("DownEncoderBlock2D",)
            down_block_types (`Tuple[str]`, *optional*, defaults to `("DownEncoderBlock2D",)`):
                Tuple of downsample block types.
            # 上采样块类型的元组，默认为 ("UpDecoderBlock2D",)
            up_block_types (`Tuple[str]`, *optional*, defaults to `("UpDecoderBlock2D",)`):
                Tuple of upsample block types.
            # 块输出通道的元组，默认为 (64,)
            block_out_channels (`Tuple[int]`, *optional*, defaults to `(64,)`):
                Tuple of block output channels.
            # 使用的激活函数，默认为 "silu"
            act_fn (`str`, *optional*, defaults to `"silu"`): The activation function to use.
            # 潜在空间的通道数，默认为 4
            latent_channels (`int`, *optional*, defaults to 4): Number of channels in the latent space.
            # 样本输入大小，默认为 32
            sample_size (`int`, *optional*, defaults to 32): Sample input size.
            # 训练潜在空间的分量标准差，默认为 0.18215
            scaling_factor (`float`, *optional*, defaults to 0.18215):
                The component-wise standard deviation of the trained latent space computed using the first batch of the
                training set. This is used to scale the latent space to have unit variance when training the diffusion
                model. The latents are scaled with the formula `z = z * scaling_factor` before being passed to the
                diffusion model. When decoding, the latents are scaled back to the original scale with the formula: `z = 1
                / scaling_factor * z`. For more details, refer to sections 4.3.2 and D.1 of the [High-Resolution Image
                Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) paper.
            # 是否强制使用 float32，以适应高分辨率管道，默认为 True
            force_upcast (`bool`, *optional*, default to `True`):
                If enabled it will force the VAE to run in float32 for high image resolution pipelines, such as SD-XL. VAE
                can be fine-tuned / trained to a lower range without losing too much precision in which case
                `force_upcast` can be set to `False` - see: https://huggingface.co/madebyollin/sdxl-vae-fp16-fix
            # 是否在 Encoder 和 Decoder 的 mid_block 中添加注意力块，默认为 True
            mid_block_add_attention (`bool`, *optional*, default to `True`):
                If enabled, the mid_block of the Encoder and Decoder will have attention blocks. If set to false, the
                mid_block will only have resnet blocks
        """
    
        # 支持梯度检查点
        _supports_gradient_checkpointing = True
        # 不分割的模块列表
        _no_split_modules = ["BasicTransformerBlock", "ResnetBlock2D"]
    
        # 注册到配置中
        @register_to_config
    # 构造函数，初始化模型参数
    def __init__(
        # 输入通道数，默认值为3
        self,
        in_channels: int = 3,
        # 输出通道数，默认值为3
        out_channels: int = 3,
        # 下采样块的类型，默认为包含一个下采样编码块的元组
        down_block_types: Tuple[str] = ("DownEncoderBlock2D",),
        # 上采样块的类型，默认为包含一个上采样解码块的元组
        up_block_types: Tuple[str] = ("UpDecoderBlock2D",),
        # 每个块的输出通道数，默认为包含64的元组
        block_out_channels: Tuple[int] = (64,),
        # 每个块的层数，默认为1
        layers_per_block: int = 1,
        # 激活函数类型，默认为"silu"
        act_fn: str = "silu",
        # 潜在通道数，默认为4
        latent_channels: int = 4,
        # 归一化的组数，默认为32
        norm_num_groups: int = 32,
        # 样本大小，默认为32
        sample_size: int = 32,
        # 缩放因子，默认为0.18215
        scaling_factor: float = 0.18215,
        # 移位因子，默认为None（可选）
        shift_factor: Optional[float] = None,
        # 潜在变量的均值，默认为None（可选）
        latents_mean: Optional[Tuple[float]] = None,
        # 潜在变量的标准差，默认为None（可选）
        latents_std: Optional[Tuple[float]] = None,
        # 强制上溢出，默认为True
        force_upcast: float = True,
        # 使用量化卷积，默认为True
        use_quant_conv: bool = True,
        # 使用后量化卷积，默认为True
        use_post_quant_conv: bool = True,
        # 中间块是否添加注意力机制，默认为True
        mid_block_add_attention: bool = True,
    ):
        # 调用父类构造函数
        super().__init__()

        # 将初始化参数传递给编码器
        self.encoder = Encoder(
            # 输入通道数
            in_channels=in_channels,
            # 输出潜在通道数
            out_channels=latent_channels,
            # 下采样块的类型
            down_block_types=down_block_types,
            # 每个块的输出通道数
            block_out_channels=block_out_channels,
            # 每个块的层数
            layers_per_block=layers_per_block,
            # 激活函数类型
            act_fn=act_fn,
            # 归一化的组数
            norm_num_groups=norm_num_groups,
            # 是否双重潜在变量
            double_z=True,
            # 中间块是否添加注意力机制
            mid_block_add_attention=mid_block_add_attention,
        )

        # 将初始化参数传递给解码器
        self.decoder = Decoder(
            # 潜在通道数作为输入
            in_channels=latent_channels,
            # 输出通道数
            out_channels=out_channels,
            # 上采样块的类型
            up_block_types=up_block_types,
            # 每个块的输出通道数
            block_out_channels=block_out_channels,
            # 每个块的层数
            layers_per_block=layers_per_block,
            # 归一化的组数
            norm_num_groups=norm_num_groups,
            # 激活函数类型
            act_fn=act_fn,
            # 中间块是否添加注意力机制
            mid_block_add_attention=mid_block_add_attention,
        )

        # 根据是否使用量化卷积初始化卷积层
        self.quant_conv = nn.Conv2d(2 * latent_channels, 2 * latent_channels, 1) if use_quant_conv else None
        # 根据是否使用后量化卷积初始化卷积层
        self.post_quant_conv = nn.Conv2d(latent_channels, latent_channels, 1) if use_post_quant_conv else None

        # 是否使用切片，初始值为False
        self.use_slicing = False
        # 是否使用平铺，初始值为False
        self.use_tiling = False

        # 仅在启用VAE平铺时相关
        # 平铺采样的最小大小设置为配置中的样本大小
        self.tile_sample_min_size = self.config.sample_size
        # 获取样本大小，如果是列表或元组则取第一个元素
        sample_size = (
            self.config.sample_size[0]
            if isinstance(self.config.sample_size, (list, tuple))
            else self.config.sample_size
        )
        # 计算平铺潜在变量的最小大小
        self.tile_latent_min_size = int(sample_size / (2 ** (len(self.config.block_out_channels) - 1)))
        # 设置平铺重叠因子
        self.tile_overlap_factor = 0.25

    # 设置梯度检查点的函数
    def _set_gradient_checkpointing(self, module, value=False):
        # 如果模块是编码器或解码器，设置梯度检查点标志
        if isinstance(module, (Encoder, Decoder)):
            module.gradient_checkpointing = value

    # 启用平铺的函数
    def enable_tiling(self, use_tiling: bool = True):
        r"""
        启用平铺VAE解码。当此选项启用时，VAE将输入张量拆分为平铺块，以分步计算解码和编码。
        这对于节省大量内存并允许处理更大图像非常有用。
        """
        # 设置使用平铺的标志
        self.use_tiling = use_tiling
    # 定义一个方法用于禁用瓷砖 VAE 解码
    def disable_tiling(self):
        r""" 
        禁用瓷砖 VAE 解码。如果之前启用了 `enable_tiling`，此方法将恢复到一次性解码计算。
        """
        # 调用设置方法，将瓷砖解码状态设置为 False
        self.enable_tiling(False)
    
    # 定义一个方法用于启用切片 VAE 解码
    def enable_slicing(self):
        r""" 
        启用切片 VAE 解码。当此选项启用时，VAE 将把输入张量分割成切片，以
        多次计算解码。这有助于节省一些内存并允许更大的批量大小。
        """
        # 设置使用切片的标志为 True
        self.use_slicing = True
    
    # 定义一个方法用于禁用切片 VAE 解码
    def disable_slicing(self):
        r""" 
        禁用切片 VAE 解码。如果之前启用了 `enable_slicing`，此方法将恢复到一次性解码计算。
        """
        # 设置使用切片的标志为 False
        self.use_slicing = False
    
    # 定义一个属性，用于返回注意力处理器
    @property
    # 复制自 diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.attn_processors
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        r""" 
        返回：
            `dict` 的注意力处理器：一个字典，包含模型中所有注意力处理器，按权重名称索引。
        """
        # 创建一个空字典用于存储处理器
        processors = {}
    
        # 定义递归函数用于添加处理器
        def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors: Dict[str, AttentionProcessor]):
            # 如果模块具有获取处理器的方法，则将其添加到字典中
            if hasattr(module, "get_processor"):
                processors[f"{name}.processor"] = module.get_processor()
    
            # 遍历模块的所有子模块，并递归调用处理器添加函数
            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)
    
            return processors
    
        # 遍历当前对象的所有子模块，并调用递归函数
        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)
    
        # 返回所有收集到的处理器
        return processors
    
    # 复制自 diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.set_attn_processor
    # 设置用于计算注意力的处理器
        def set_attn_processor(self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]):
            r"""
            设置用于计算注意力的处理器。
    
            参数：
                processor（`dict` of `AttentionProcessor` or only `AttentionProcessor`）：
                    已实例化的处理器类或处理器类的字典，将作为**所有** `Attention` 层的处理器设置。
    
                    如果 `processor` 是字典，则键需要定义对应的交叉注意力处理器的路径。当设置可训练的注意力处理器时，强烈推荐使用这种方式。
    
            """
            # 获取当前注意力处理器的数量
            count = len(self.attn_processors.keys())
    
            # 检查传入的处理器字典长度是否与注意力层数量匹配
            if isinstance(processor, dict) and len(processor) != count:
                raise ValueError(
                    f"传入的处理器字典数量 {len(processor)} 与注意力层数量 {count} 不匹配。请确保传入 {count} 个处理器类。"
                )
    
            # 递归设置注意力处理器的函数
            def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
                # 如果模块有设置处理器的方法
                if hasattr(module, "set_processor"):
                    # 如果处理器不是字典，直接设置
                    if not isinstance(processor, dict):
                        module.set_processor(processor)
                    else:
                        # 从字典中弹出对应的处理器并设置
                        module.set_processor(processor.pop(f"{name}.processor"))
    
                # 遍历模块的子模块，递归调用
                for sub_name, child in module.named_children():
                    fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)
    
            # 对当前实例的每个子模块调用递归函数
            for name, module in self.named_children():
                fn_recursive_attn_processor(name, module, processor)
    
        # 从 diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.set_default_attn_processor 复制
        def set_default_attn_processor(self):
            """
            禁用自定义注意力处理器，并设置默认的注意力实现。
            """
            # 检查所有处理器是否属于添加的 KV 注意力处理器
            if all(proc.__class__ in ADDED_KV_ATTENTION_PROCESSORS for proc in self.attn_processors.values()):
                processor = AttnAddedKVProcessor()
            # 检查所有处理器是否属于交叉注意力处理器
            elif all(proc.__class__ in CROSS_ATTENTION_PROCESSORS for proc in self.attn_processors.values()):
                processor = AttnProcessor()
            else:
                raise ValueError(
                    f"当注意力处理器的类型为 {next(iter(self.attn_processors.values()))} 时，无法调用 `set_default_attn_processor`"
                )
    
            # 调用设置处理器的方法
            self.set_attn_processor(processor)
    
        # 应用前向钩子
        @apply_forward_hook
        def encode(
            self, x: torch.Tensor, return_dict: bool = True
    # 定义返回类型为 AutoencoderKLOutput 或者 DiagonalGaussianDistribution 的函数
    ) -> Union[AutoencoderKLOutput, Tuple[DiagonalGaussianDistribution]]:
            """
            编码一批图像为潜在表示。
    
            参数：
                x (`torch.Tensor`): 输入图像的批次。
                return_dict (`bool`, *可选*, 默认为 `True`):
                    是否返回 [`~models.autoencoder_kl.AutoencoderKLOutput`] 而非简单元组。
    
            返回：
                    编码图像的潜在表示。如果 `return_dict` 为 True，则返回一个
                    [`~models.autoencoder_kl.AutoencoderKLOutput`]，否则返回一个普通的 `tuple`。
            """
            # 检查是否使用平铺，并且输入尺寸超过最小平铺尺寸
            if self.use_tiling and (x.shape[-1] > self.tile_sample_min_size or x.shape[-2] > self.tile_sample_min_size):
                # 使用平铺编码方法处理输入
                return self.tiled_encode(x, return_dict=return_dict)
    
            # 检查是否使用切片，并且输入批次大于1
            if self.use_slicing and x.shape[0] > 1:
                # 对输入的每个切片进行编码，并将结果连接起来
                encoded_slices = [self.encoder(x_slice) for x_slice in x.split(1)]
                h = torch.cat(encoded_slices)
            else:
                # 直接编码整个输入
                h = self.encoder(x)
    
            # 检查量化卷积是否存在
            if self.quant_conv is not None:
                # 使用量化卷积处理编码后的结果
                moments = self.quant_conv(h)
            else:
                # 如果不存在，直接使用编码结果
                moments = h
    
            # 创建对角高斯分布的后验
            posterior = DiagonalGaussianDistribution(moments)
    
            # 如果不返回字典，返回后验分布的元组
            if not return_dict:
                return (posterior,)
    
            # 返回 AutoencoderKLOutput 对象，包含潜在分布
            return AutoencoderKLOutput(latent_dist=posterior)
    
        # 定义解码函数，返回类型为 DecoderOutput 或 torch.Tensor
        def _decode(self, z: torch.Tensor, return_dict: bool = True) -> Union[DecoderOutput, torch.Tensor]:
            # 检查是否使用平铺，并且潜在向量尺寸超过最小平铺尺寸
            if self.use_tiling and (z.shape[-1] > self.tile_latent_min_size or z.shape[-2] > self.tile_latent_min_size):
                # 使用平铺解码方法处理潜在向量
                return self.tiled_decode(z, return_dict=return_dict)
    
            # 检查后量化卷积是否存在
            if self.post_quant_conv is not None:
                # 使用后量化卷积处理潜在向量
                z = self.post_quant_conv(z)
    
            # 通过解码器解码潜在向量
            dec = self.decoder(z)
    
            # 如果不返回字典，返回解码结果的元组
            if not return_dict:
                return (dec,)
    
            # 返回解码结果的 DecoderOutput 对象
            return DecoderOutput(sample=dec)
    
        # 应用前向钩子的解码函数
        @apply_forward_hook
        def decode(
            self, z: torch.FloatTensor, return_dict: bool = True, generator=None
        ) -> Union[DecoderOutput, torch.FloatTensor]:
            """
            解码一批图像。
    
            参数：
                z (`torch.Tensor`): 输入潜在向量的批次。
                return_dict (`bool`, *可选*, 默认为 `True`):
                    是否返回 [`~models.vae.DecoderOutput`] 而非简单元组。
    
            返回：
                [`~models.vae.DecoderOutput`] 或 `tuple`:
                    如果 return_dict 为 True，返回 [`~models.vae.DecoderOutput`]，否则返回普通 `tuple`。
            """
            # 检查是否使用切片，并且潜在向量批次大于1
            if self.use_slicing and z.shape[0] > 1:
                # 对每个切片进行解码，并将结果连接起来
                decoded_slices = [self._decode(z_slice).sample for z_slice in z.split(1)]
                decoded = torch.cat(decoded_slices)
            else:
                # 直接解码整个潜在向量
                decoded = self._decode(z).sample
    
            # 如果不返回字典，返回解码结果的元组
            if not return_dict:
                return (decoded,)
    
            # 返回解码结果的 DecoderOutput 对象
            return DecoderOutput(sample=decoded)
    # 定义一个垂直混合函数，接受两个张量和混合范围，返回混合后的张量
        def blend_v(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
            # 计算实际的混合范围，确保不超过输入张量的尺寸
            blend_extent = min(a.shape[2], b.shape[2], blend_extent)
            # 逐行进行混合操作，根据当前行的比例计算混合值
            for y in range(blend_extent):
                b[:, :, y, :] = a[:, :, -blend_extent + y, :] * (1 - y / blend_extent) + b[:, :, y, :] * (y / blend_extent)
            # 返回混合后的张量
            return b
    
    # 定义一个水平混合函数，接受两个张量和混合范围，返回混合后的张量
        def blend_h(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
            # 计算实际的混合范围，确保不超过输入张量的尺寸
            blend_extent = min(a.shape[3], b.shape[3], blend_extent)
            # 逐列进行混合操作，根据当前列的比例计算混合值
            for x in range(blend_extent):
                b[:, :, :, x] = a[:, :, :, -blend_extent + x] * (1 - x / blend_extent) + b[:, :, :, x] * (x / blend_extent)
            # 返回混合后的张量
            return b
    # 定义一个函数，用于通过平铺编码器对图像批次进行编码
    def tiled_encode(self, x: torch.Tensor, return_dict: bool = True) -> AutoencoderKLOutput:
        # 文档字符串，描述该函数的用途及参数
        r"""Encode a batch of images using a tiled encoder.
    
        当这个选项启用时，VAE 会将输入张量分割成多个小块以进行编码
        步骤。这对于保持内存使用量恒定非常有用。平铺编码的最终结果与非平铺编码不同，
        因为每个小块使用不同的编码器。为了避免平铺伪影，小块之间会重叠并混合在一起
        形成平滑的输出。你可能仍然会看到与小块大小相关的变化，
        但这些变化应该不那么明显。
    
        参数:
            x (`torch.Tensor`): 输入图像批次。
            return_dict (`bool`, *可选*, 默认为 `True`):
                是否返回一个 [`~models.autoencoder_kl.AutoencoderKLOutput`] 而不是一个普通元组。
    
        返回:
            [`~models.autoencoder_kl.AutoencoderKLOutput`] 或 `tuple`:
                如果 return_dict 为 True，则返回 [`~models.autoencoder_kl.AutoencoderKLOutput`]，
                否则返回普通元组。
        """
        # 计算重叠区域的大小
        overlap_size = int(self.tile_sample_min_size * (1 - self.tile_overlap_factor))
        # 计算混合的范围
        blend_extent = int(self.tile_latent_min_size * self.tile_overlap_factor)
        # 计算行限制，确保不会超出范围
        row_limit = self.tile_latent_min_size - blend_extent
    
        # 初始化一个列表以存储每一行的编码结果
        rows = []
        # 遍历输入张量的高度，以重叠的方式进行切片
        for i in range(0, x.shape[2], overlap_size):
            # 初始化当前行的编码结果列表
            row = []
            # 遍历输入张量的宽度，以重叠的方式进行切片
            for j in range(0, x.shape[3], overlap_size):
                # 切割当前小块
                tile = x[:, :, i : i + self.tile_sample_min_size, j : j + self.tile_sample_min_size]
                # 对当前小块进行编码
                tile = self.encoder(tile)
                # 如果配置使用量化卷积，则对小块进行量化处理
                if self.config.use_quant_conv:
                    tile = self.quant_conv(tile)
                # 将编码后的小块添加到当前行中
                row.append(tile)
            # 将当前行的结果添加到 rows 列表中
            rows.append(row)
        # 初始化一个列表以存储最终的结果行
        result_rows = []
        # 遍历所有行以进行混合处理
        for i, row in enumerate(rows):
            result_row = []
            # 遍历当前行的每个小块
            for j, tile in enumerate(row):
                # 将上方小块与当前小块混合
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_extent)
                # 将左侧小块与当前小块混合
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_extent)
                # 将混合后的小块裁剪至指定大小并添加到结果行
                result_row.append(tile[:, :, :row_limit, :row_limit])
            # 将当前行的结果合并并添加到最终结果中
            result_rows.append(torch.cat(result_row, dim=3))
    
        # 将所有结果行合并为一个张量
        moments = torch.cat(result_rows, dim=2)
        # 创建一个对角高斯分布以表示后验分布
        posterior = DiagonalGaussianDistribution(moments)
    
        # 如果不返回字典，则返回后验分布的元组
        if not return_dict:
            return (posterior,)
    
        # 返回包含后验分布的 AutoencoderKLOutput 对象
        return AutoencoderKLOutput(latent_dist=posterior)
    # 定义一个方法，用于解码一批图像，使用平铺解码器
    def tiled_decode(self, z: torch.Tensor, return_dict: bool = True) -> Union[DecoderOutput, torch.Tensor]:
        r"""
        使用平铺解码器解码一批图像。

        参数：
            z (`torch.Tensor`): 输入的潜在向量批次。
            return_dict (`bool`, *可选*, 默认值为 `True`):
                是否返回一个 [`~models.vae.DecoderOutput`] 而不是普通的元组。

        返回：
            [`~models.vae.DecoderOutput`] 或 `tuple`:
                如果 return_dict 为 True，则返回一个 [`~models.vae.DecoderOutput`]，
                否则返回普通的 `tuple`。
        """
        # 计算重叠区域的大小
        overlap_size = int(self.tile_latent_min_size * (1 - self.tile_overlap_factor))
        # 计算混合区域的范围
        blend_extent = int(self.tile_sample_min_size * self.tile_overlap_factor)
        # 计算每行的限制大小
        row_limit = self.tile_sample_min_size - blend_extent

        # 将 z 分割成重叠的 64x64 瓦片，并分别解码
        # 瓦片之间有重叠，以避免瓦片之间的接缝
        rows = []
        # 遍历潜在向量 z 的高度，按重叠大小步进
        for i in range(0, z.shape[2], overlap_size):
            row = []  # 存储当前行的解码结果
            # 遍历潜在向量 z 的宽度，按重叠大小步进
            for j in range(0, z.shape[3], overlap_size):
                # 从 z 中提取当前瓦片
                tile = z[:, :, i : i + self.tile_latent_min_size, j : j + self.tile_latent_min_size]
                # 如果配置中启用了后量化卷积，则对瓦片进行处理
                if self.config.use_post_quant_conv:
                    tile = self.post_quant_conv(tile)
                # 解码当前瓦片
                decoded = self.decoder(tile)
                # 将解码结果添加到当前行中
                row.append(decoded)
            # 将当前行添加到总行列表中
            rows.append(row)
        result_rows = []  # 存储最终结果的行
        # 遍历解码的每一行
        for i, row in enumerate(rows):
            result_row = []  # 存储当前结果行
            # 遍历当前行的瓦片
            for j, tile in enumerate(row):
                # 将上方的瓦片与当前瓦片混合
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_extent)
                # 将左侧的瓦片与当前瓦片混合
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_extent)
                # 将当前瓦片的结果裁剪到限制大小并添加到结果行
                result_row.append(tile[:, :, :row_limit, :row_limit])
            # 将结果行中的瓦片沿着宽度拼接
            result_rows.append(torch.cat(result_row, dim=3))

        # 将所有结果行沿着高度拼接
        dec = torch.cat(result_rows, dim=2)
        # 如果不返回字典，则返回解码结果的元组
        if not return_dict:
            return (dec,)

        # 返回解码结果的 DecoderOutput 对象
        return DecoderOutput(sample=dec)

    # 定义前向传播方法
    def forward(
        # 输入样本的张量
        sample: torch.Tensor,
        # 是否对样本进行后验采样
        sample_posterior: bool = False,
        # 是否返回字典形式的结果
        return_dict: bool = True,
        # 随机数生成器（可选）
        generator: Optional[torch.Generator] = None,
    ) -> Union[DecoderOutput, torch.Tensor]:
        r"""  # 函数的返回类型是 DecoderOutput 或 torch.Tensor 的联合类型
        Args:  # 参数说明
            sample (`torch.Tensor`): Input sample.  # 输入样本，类型为 torch.Tensor
            sample_posterior (`bool`, *optional*, defaults to `False`):  # 是否从后验分布进行采样，默认为 False
                Whether to sample from the posterior.  # 描述参数的用途
            return_dict (`bool`, *optional*, defaults to `True`):  # 是否返回 DecoderOutput 而不是普通元组，默认为 True
                Whether or not to return a [`DecoderOutput`] instead of a plain tuple.  # 描述参数的用途
        """
        x = sample  # 将输入样本赋值给 x
        posterior = self.encode(x).latent_dist  # 对输入样本进行编码，并获取其后验分布
        if sample_posterior:  # 检查是否需要从后验分布中采样
            z = posterior.sample(generator=generator)  # 从后验分布中进行采样
        else:  # 否则
            z = posterior.mode()  # 取后验分布的众数
        dec = self.decode(z).sample  # 解码 z 并获取样本

        if not return_dict:  # 如果不需要返回字典
            return (dec,)  # 返回样本作为元组

        return DecoderOutput(sample=dec)  # 返回 DecoderOutput 对象，包含解码后的样本

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.fuse_qkv_projections
    def fuse_qkv_projections(self):  # 定义融合 QKV 投影的方法
        """  # 方法的文档字符串
        Enables fused QKV projections. For self-attention modules, all projection matrices (i.e., query, key, value)  # 启用融合的 QKV 投影，适用于自注意力模块
        are fused. For cross-attention modules, key and value projection matrices are fused.  # 适用于交叉注意力模块

        <Tip warning={true}>  # 提示标签，表示此 API 为实验性
        This API is 🧪 experimental.  # 提示内容
        </Tip>
        """
        self.original_attn_processors = None  # 初始化原始注意力处理器为 None

        for _, attn_processor in self.attn_processors.items():  # 遍历当前的注意力处理器
            if "Added" in str(attn_processor.__class__.__name__):  # 检查处理器类名中是否包含 "Added"
                raise ValueError("`fuse_qkv_projections()` is not supported for models having added KV projections.")  # 抛出异常，提示不支持融合操作

        self.original_attn_processors = self.attn_processors  # 保存当前的注意力处理器

        for module in self.modules():  # 遍历模型中的所有模块
            if isinstance(module, Attention):  # 如果模块是 Attention 类型
                module.fuse_projections(fuse=True)  # 融合其投影

        self.set_attn_processor(FusedAttnProcessor2_0())  # 设置融合的注意力处理器

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.unfuse_qkv_projections
    def unfuse_qkv_projections(self):  # 定义取消融合 QKV 投影的方法
        """Disables the fused QKV projection if enabled.  # 如果已启用，禁用融合的 QKV 投影

        <Tip warning={true}>  # 提示标签，表示此 API 为实验性
        This API is 🧪 experimental.  # 提示内容
        </Tip>

        """
        if self.original_attn_processors is not None:  # 如果原始注意力处理器不为 None
            self.set_attn_processor(self.original_attn_processors)  # 恢复原始注意力处理器
```