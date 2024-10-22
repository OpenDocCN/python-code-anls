# `.\diffusers\models\autoencoders\vq_model.py`

```py
# 版权声明，指明版权归 HuggingFace 团队所有
# 
# 根据 Apache 许可证第 2.0 版（“许可证”）授权；
# 你不得在不遵守许可证的情况下使用此文件。
# 可以在以下网址获取许可证副本：
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# 除非适用法律要求或书面同意，否则软件在“按原样”基础上分发，
# 不提供任何形式的担保或条件，无论是明示或暗示的。
# 有关许可证的具体条款和条件，请参阅许可证。
# 从 dataclasses 模块导入 dataclass 装饰器
from dataclasses import dataclass
# 从 typing 模块导入可选类型、元组和联合类型
from typing import Optional, Tuple, Union

# 导入 PyTorch 库
import torch
# 导入 PyTorch 的神经网络模块
import torch.nn as nn

# 从配置工具中导入 ConfigMixin 和注册配置的函数
from ...configuration_utils import ConfigMixin, register_to_config
# 从工具模块导入 BaseOutput 类
from ...utils import BaseOutput
# 从加速工具中导入应用前向钩子的函数
from ...utils.accelerate_utils import apply_forward_hook
# 从自动编码器模块导入解码器、解码器输出、编码器和向量量化器
from ..autoencoders.vae import Decoder, DecoderOutput, Encoder, VectorQuantizer
# 从建模工具中导入 ModelMixin 类
from ..modeling_utils import ModelMixin


# 定义一个数据类，用于表示 VQModel 编码方法的输出
@dataclass
class VQEncoderOutput(BaseOutput):
    """
    VQModel 编码方法的输出。

    参数：
        latents （`torch.Tensor`，形状为 `(batch_size, num_channels, height, width)`）：
            模型最后一层的编码输出样本。
    """

    # 定义一个属性 latents，类型为 torch.Tensor
    latents: torch.Tensor


# 定义 VQModel 类，继承自 ModelMixin 和 ConfigMixin
class VQModel(ModelMixin, ConfigMixin):
    r"""
    用于解码潜在表示的 VQ-VAE 模型。

    该模型继承自 [`ModelMixin`]。请查看超类文档，以了解其为所有模型实现的通用方法
    （例如下载或保存）。
    # 函数参数说明部分
    Parameters:
        # 输入图像的通道数，默认为3
        in_channels (int, *optional*, defaults to 3): Number of channels in the input image.
        # 输出图像的通道数，默认为3
        out_channels (int,  *optional*, defaults to 3): Number of channels in the output.
        # 下采样块类型的元组，默认为包含一个类型的元组
        down_block_types (`Tuple[str]`, *optional*, defaults to `("DownEncoderBlock2D",)`):
            Tuple of downsample block types.
        # 上采样块类型的元组，默认为包含一个类型的元组
        up_block_types (`Tuple[str]`, *optional*, defaults to `("UpDecoderBlock2D",)`):
            Tuple of upsample block types.
        # 块输出通道数的元组，默认为包含一个值的元组
        block_out_channels (`Tuple[int]`, *optional*, defaults to `(64,)`):
            Tuple of block output channels.
        # 每个块的层数，默认为1
        layers_per_block (`int`, *optional*, defaults to `1`): Number of layers per block.
        # 激活函数类型，默认为"silu"
        act_fn (`str`, *optional*, defaults to `"silu"`): The activation function to use.
        # 潜在空间的通道数，默认为3
        latent_channels (`int`, *optional*, defaults to `3`): Number of channels in the latent space.
        # 输入样本的大小，默认为32
        sample_size (`int`, *optional*, defaults to `32`): Sample input size.
        # VQ-VAE中的代码本向量数量，默认为256
        num_vq_embeddings (`int`, *optional*, defaults to `256`): Number of codebook vectors in the VQ-VAE.
        # 归一化层的组数，默认为32
        norm_num_groups (`int`, *optional*, defaults to `32`): Number of groups for normalization layers.
        # VQ-VAE中代码本向量的隐藏维度，可选
        vq_embed_dim (`int`, *optional*): Hidden dim of codebook vectors in the VQ-VAE.
        # 缩放因子，默认为0.18215，主要用于训练时的标准化
        scaling_factor (`float`, *optional*, defaults to `0.18215`):
            The component-wise standard deviation of the trained latent space computed using the first batch of the
            training set. This is used to scale the latent space to have unit variance when training the diffusion
            model. The latents are scaled with the formula `z = z * scaling_factor` before being passed to the
            diffusion model. When decoding, the latents are scaled back to the original scale with the formula: `z = 1
            / scaling_factor * z`. For more details, refer to sections 4.3.2 and D.1 of the [High-Resolution Image
            Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) paper.
        # 归一化层的类型，默认为"group"，可选为"group"或"spatial"
        norm_type (`str`, *optional*, defaults to `"group"`):
            Type of normalization layer to use. Can be one of `"group"` or `"spatial"`.
    """

    # 注册配置的构造函数
    @register_to_config
    def __init__(
        # 输入通道参数，默认为3
        self,
        in_channels: int = 3,
        # 输出通道参数，默认为3
        out_channels: int = 3,
        # 下采样块类型，默认为("DownEncoderBlock2D",)
        down_block_types: Tuple[str, ...] = ("DownEncoderBlock2D",),
        # 上采样块类型，默认为("UpDecoderBlock2D",)
        up_block_types: Tuple[str, ...] = ("UpDecoderBlock2D",),
        # 块输出通道参数，默认为(64,)
        block_out_channels: Tuple[int, ...] = (64,),
        # 每块层数参数，默认为1
        layers_per_block: int = 1,
        # 激活函数参数，默认为"silu"
        act_fn: str = "silu",
        # 潜在通道数参数，默认为3
        latent_channels: int = 3,
        # 样本大小参数，默认为32
        sample_size: int = 32,
        # VQ-VAE代码本向量数量，默认为256
        num_vq_embeddings: int = 256,
        # 归一化层组数参数，默认为32
        norm_num_groups: int = 32,
        # VQ-VAE代码本向量隐藏维度，默认为None
        vq_embed_dim: Optional[int] = None,
        # 缩放因子参数，默认为0.18215
        scaling_factor: float = 0.18215,
        # 归一化层类型参数，默认为"group"
        norm_type: str = "group",  # group, spatial
        # 是否在中间块添加注意力，默认为True
        mid_block_add_attention=True,
        # 是否从代码本查找，默认为False
        lookup_from_codebook=False,
        # 是否强制上溯，默认为False
        force_upcast=False,
    # 初始化方法，调用父类构造函数
        ):
            super().__init__()
    
            # 将初始化参数传递给编码器
            self.encoder = Encoder(
                in_channels=in_channels,  # 输入通道数
                out_channels=latent_channels,  # 潜在通道数
                down_block_types=down_block_types,  # 下采样块类型
                block_out_channels=block_out_channels,  # 块输出通道数
                layers_per_block=layers_per_block,  # 每个块的层数
                act_fn=act_fn,  # 激活函数
                norm_num_groups=norm_num_groups,  # 归一化的组数
                double_z=False,  # 是否使用双潜变量
                mid_block_add_attention=mid_block_add_attention,  # 中间块是否添加注意力机制
            )
    
            # 如果未提供，使用潜在通道数作为嵌入维度
            vq_embed_dim = vq_embed_dim if vq_embed_dim is not None else latent_channels
    
            # 创建量化卷积层，将潜在通道数映射到嵌入维度
            self.quant_conv = nn.Conv2d(latent_channels, vq_embed_dim, 1)
            # 初始化向量量化器
            self.quantize = VectorQuantizer(num_vq_embeddings, vq_embed_dim, beta=0.25, remap=None, sane_index_shape=False)
            # 创建后量化卷积层，将嵌入维度映射回潜在通道数
            self.post_quant_conv = nn.Conv2d(vq_embed_dim, latent_channels, 1)
    
            # 将初始化参数传递给解码器
            self.decoder = Decoder(
                in_channels=latent_channels,  # 潜在通道数
                out_channels=out_channels,  # 输出通道数
                up_block_types=up_block_types,  # 上采样块类型
                block_out_channels=block_out_channels,  # 块输出通道数
                layers_per_block=layers_per_block,  # 每个块的层数
                act_fn=act_fn,  # 激活函数
                norm_num_groups=norm_num_groups,  # 归一化的组数
                norm_type=norm_type,  # 归一化类型
                mid_block_add_attention=mid_block_add_attention,  # 中间块是否添加注意力机制
            )
    
        # 应用前向钩子，定义编码方法
        @apply_forward_hook
        def encode(self, x: torch.Tensor, return_dict: bool = True) -> VQEncoderOutput:
            # 将输入 x 传递给编码器以获取潜在表示
            h = self.encoder(x)
            # 通过量化卷积层处理潜在表示
            h = self.quant_conv(h)
    
            # 如果不需要返回字典，返回潜在表示
            if not return_dict:
                return (h,)
    
            # 返回包含潜在表示的自定义输出对象
            return VQEncoderOutput(latents=h)
    
        # 应用前向钩子，定义解码方法
        @apply_forward_hook
        def decode(
            self, h: torch.Tensor, force_not_quantize: bool = False, return_dict: bool = True, shape=None
        ) -> Union[DecoderOutput, torch.Tensor]:
            # 如果不强制不量化，则通过量化层处理潜在表示
            if not force_not_quantize:
                quant, commit_loss, _ = self.quantize(h)
            # 如果从代码本中查找，则获取代码本条目
            elif self.config.lookup_from_codebook:
                quant = self.quantize.get_codebook_entry(h, shape)
                # 初始化承诺损失为零
                commit_loss = torch.zeros((h.shape[0])).to(h.device, dtype=h.dtype)
            else:
                # 否则直接使用输入
                quant = h
                # 初始化承诺损失为零
                commit_loss = torch.zeros((h.shape[0])).to(h.device, dtype=h.dtype)
            # 通过后量化卷积层处理量化结果
            quant2 = self.post_quant_conv(quant)
            # 将量化结果传递给解码器以获取输出
            dec = self.decoder(quant2, quant if self.config.norm_type == "spatial" else None)
    
            # 如果不需要返回字典，返回解码结果和承诺损失
            if not return_dict:
                return dec, commit_loss
    
            # 返回自定义输出对象，包括解码结果和承诺损失
            return DecoderOutput(sample=dec, commit_loss=commit_loss)
    
        # 定义前向传播方法
        def forward(
            self, sample: torch.Tensor, return_dict: bool = True
    ) -> Union[DecoderOutput, Tuple[torch.Tensor, ...]]:
        r"""  # 文档字符串，描述该方法的功能和参数
        The [`VQModel`] forward method.  # 指明这是 VQModel 类的前向传播方法

        Args:  # 参数说明部分
            sample (`torch.Tensor`): Input sample.  # 输入样本，类型为 torch.Tensor
            return_dict (`bool`, *optional*, defaults to `True`):  # 可选参数，指示是否返回字典
                Whether or not to return a [`models.autoencoders.vq_model.VQEncoderOutput`] instead of a plain tuple.  # 说明返回值的类型

        Returns:  # 返回值说明部分
            [`~models.autoencoders.vq_model.VQEncoderOutput`] or `tuple`:  # 返回值可以是 VQEncoderOutput 对象或元组
                If return_dict is True, a [`~models.autoencoders.vq_model.VQEncoderOutput`] is returned, otherwise a  # 如果 return_dict 为 True，则返回 VQEncoderOutput
                plain `tuple` is returned.  # 否则返回一个普通的元组
        """

        h = self.encode(sample).latents  # 调用 encode 方法对输入样本进行编码，并获取其潜在表示
        dec = self.decode(h)  # 调用 decode 方法对潜在表示进行解码，获取解码结果

        if not return_dict:  # 如果 return_dict 为 False
            return dec.sample, dec.commit_loss  # 返回解码结果的样本和承诺损失
        return dec  # 否则返回解码结果对象
```