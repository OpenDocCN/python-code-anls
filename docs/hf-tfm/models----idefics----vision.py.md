# `.\models\idefics\vision.py`

```py
# 设置文件编码为 UTF-8

# 版权声明，声明此代码版权归 OpenAI 团队和 HuggingFace 团队所有，保留所有权利
#
# 根据 Apache 许可证 2.0 版本授权使用本文件
# 除非符合许可证规定，否则不得使用本文件
# 您可以从以下网址获取许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，本软件是基于“按原样”提供的，没有任何形式的明示或暗示担保或条件
# 有关详细信息，请参阅许可证

""" PyTorch IdeficsVision model: a copy of CLIPVisionModel using a simpler config object"""

# 导入必要的库和模块
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn

# 导入模型输出相关的类和函数
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from ...utils import ModelOutput, logging

# 导入 IdeficsVisionConfig 配置类
from .configuration_idefics import IdeficsVisionConfig

# 获取日志记录器对象
logger = logging.get_logger(__name__)

# 数据类，用于表示 IdeficsVision 模型的输出
@dataclass
class IdeficsVisionModelOutput(ModelOutput):
    """
    Base class for vision model's outputs that also contains image embeddings of the pooling of the last hidden states.

    Args:
        image_embeds (`torch.FloatTensor` of shape `(batch_size, output_dim)` *optional* returned when model is initialized with `with_projection=True`):
            The image embeddings obtained by applying the projection layer to the pooler_output.
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True` is set to `True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    image_embeds: Optional[torch.FloatTensor] = None  # 可选的图像嵌入，通过将投影层应用于 pooler_output 而获得
    last_hidden_state: torch.FloatTensor = None  # 最后一层模型输出的隐藏状态序列
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None  # 隐藏状态的元组，每层模型的输出和可选的初始嵌入输出
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None  # 注意力权重的元组，用于计算自注意头中的加权平均值
# Adapted from transformers.models.clip.modeling_clip.CLIPVisionEmbeddings
# 从 transformers.models.clip.modeling_clip.CLIPVisionEmbeddings 改编而来

class IdeficsVisionEmbeddings(nn.Module):
    # 定义 IdeficsVisionEmbeddings 类，继承自 nn.Module
    def __init__(self, config: IdeficsVisionConfig):
        super().__init__()
        # 调用父类构造函数初始化模块
        self.config = config
        # 存储传入的配置对象
        self.embed_dim = config.hidden_size
        # 设置嵌入维度为配置中的隐藏大小
        self.image_size = config.image_size
        # 图像尺寸为配置中的图像大小
        self.patch_size = config.patch_size
        # 补丁大小为配置中的补丁大小

        self.class_embedding = nn.Parameter(torch.randn(self.embed_dim))
        # 初始化类别嵌入为一个随机的可学习参数

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=False,
        )
        # 创建卷积层用于将图像补丁映射到嵌入维度空间，无偏置项

        self.num_patches = (self.image_size // self.patch_size) ** 2
        # 计算图像中的补丁数量

        self.num_positions = self.num_patches + 1
        # 计算位置嵌入的数量，为补丁数量加一

        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        # 创建位置嵌入层，将每个位置映射到嵌入维度空间

        self.register_buffer("position_ids", torch.arange(self.num_positions).expand((1, -1)), persistent=False)
        # 注册位置 ID 缓冲区，用于存储位置索引张量，非持久化

    # Heavily inspired from https://github.com/huggingface/transformers/blob/v4.33.0/src/transformers/models/vit/modeling_vit.py#L82
    # 在很大程度上受到 https://github.com/huggingface/transformers/blob/v4.33.0/src/transformers/models/vit/modeling_vit.py#L82 的启发
    # 对嵌入向量进行插值，以便在更高分辨率的图像上使用预训练的位置编码
    def interpolate_pos_encoding(self, embeddings: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """
        This method allows to interpolate the pre-trained position encodings, to be able to use the model on higher
        resolution images.

        Source:
        https://github.com/facebookresearch/dino/blob/de9ee3df6cf39fac952ab558447af1fa1365362a/vision_transformer.py#L174
        """

        # 计算嵌入向量中的补丁数量
        num_patches = embeddings.shape[1] - 1
        # 获取位置编码
        pos_embed = self.position_embedding(self.position_ids)
        # 获取位置编码的数量
        num_positions = pos_embed.shape[1] - 1

        # 如果补丁数量与位置编码数量相等，并且图像高度与宽度相等，则直接返回位置编码
        if num_patches == num_positions and height == width:
            return pos_embed

        # 提取类别位置编码和补丁位置编码
        class_pos_embed = pos_embed[:, 0]
        patch_pos_embed = pos_embed[:, 1:]

        # 获取嵌入向量的维度
        embed_dim = embeddings.shape[-1]
        # 计算高度和宽度上的补丁数量
        num_h_patches = height // self.config.patch_size
        num_w_patches = width // self.config.patch_size

        # 添加一个小数以避免插值时的浮点错误
        num_h_patches, num_w_patches = num_h_patches + 0.1, num_w_patches + 0.1

        # 计算位置编码中位置的平方根
        sqrt_num_positions = math.sqrt(num_positions)

        # 重塑补丁位置编码的形状以便插值
        patch_pos_embed = patch_pos_embed.reshape(1, int(sqrt_num_positions), int(sqrt_num_positions), embed_dim)
        patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)

        # 如果补丁位置编码的数据类型为 torch.bfloat16，则进行转换为 torch.float，因为 torch.bfloat16 不支持 bicubic 插值
        fp32_upcasting = patch_pos_embed.dtype == torch.bfloat16
        if fp32_upcasting:
            logger.warning_once(
                "Upcasting patch_pos_embed to fp32 for interpolation since `upsample_bicubic2d_out_frame` in nn.functional.interpolate "
                "is not implemented for 'torch.bfloat16' dtype. This will result in a slight overhead."
            )
            patch_pos_embed = patch_pos_embed.to(torch.float)

        # 使用双三次插值对补丁位置编码进行插值
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed,
            scale_factor=(num_h_patches / sqrt_num_positions, num_w_patches / sqrt_num_positions),
            mode="bicubic",
            align_corners=False,
        )

        # 如果之前进行了类型转换，则将补丁位置编码还原为 torch.bfloat16 类型
        if fp32_upcasting:
            patch_pos_embed = patch_pos_embed.to(torch.bfloat16)

        # 检查插值后的补丁位置编码形状是否符合预期
        if int(num_h_patches) != patch_pos_embed.shape[-2] or int(num_w_patches) != patch_pos_embed.shape[-1]:
            raise ValueError(
                f"Number of patches for images ({int(num_h_patches), int(num_w_patches)}) don't match the "
                f"shape of position embedding ({patch_pos_embed.shape[-2], patch_pos_embed.shape[-1]})"
            )

        # 调整补丁位置编码的形状并返回
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, embed_dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)
    # 定义前向传播函数，输入是像素值张量 pixel_values 和是否插值位置编码的标志 interpolate_pos_encoding
    def forward(self, pixel_values: torch.FloatTensor, interpolate_pos_encoding: bool = False) -> torch.Tensor:
        # 获取输入张量的批量大小、通道数、高度和宽度
        batch_size, num_channels, height, width = pixel_values.shape
        
        # 如果不进行位置编码的插值
        if not interpolate_pos_encoding:
            # 检查输入图像的尺寸是否与模型要求的 self.image_size 一致，否则引发值错误
            if height != self.image_size or width != self.image_size:
                raise ValueError(
                    f"Input image size ({height}*{width}) doesn't match model"
                    f" ({self.image_size}*{self.image_size}). You should try to set `interpolate_pos_encoding=True`"
                )

        # 确定目标数据类型为 self.patch_embedding 的权重数据类型
        target_dtype = self.patch_embedding.weight.dtype
        
        # 使用 patch_embedding 将像素值张量转换为补丁嵌入向量，形状为 [*, width, grid, grid]
        patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype))

        # 将 patch_embeds 沿着第三个维度展平，然后交换第一维和第二维
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)

        # 创建类别嵌入向量，扩展到与 batch_size 相同的大小
        class_embeds = self.class_embedding.expand(batch_size, 1, -1)
        
        # 将类别嵌入向量与补丁嵌入向量连接起来
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)

        # 如果插值位置编码为真，则对每个令牌添加位置编码
        if interpolate_pos_encoding:
            embeddings = embeddings + self.interpolate_pos_encoding(embeddings, height, width)
        else:
            # 否则，使用预定义的位置 id 来添加位置编码
            embeddings = embeddings + self.position_embedding(self.position_ids)

        # 返回最终的嵌入向量
        return embeddings
# Copied from transformers.models.clip.modeling_clip.CLIPAttention with CLIP->IdeficsVision
class IdeficsVisionAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size  # 设置嵌入维度为隐藏大小
        self.num_heads = config.num_attention_heads  # 获取注意力头的数量
        self.head_dim = self.embed_dim // self.num_heads  # 计算每个注意力头的维度
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        self.scale = self.head_dim**-0.5  # 缩放因子，根据头的维度计算
        self.dropout = config.attention_dropout  # 注意力部分的丢弃率

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)  # 查询投影层
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)  # 值投影层
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)  # 键投影层
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)  # 输出投影层

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        """重新形状张量以适应多头注意力的结构"""
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,



# Copied from transformers.models.clip.modeling_clip.CLIPMLP with CLIP->IdeficsVision
class IdeficsVisionMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.activation_fn = ACT2FN[config.hidden_act]  # 激活函数从配置中获取
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)  # 第一个全连接层
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)  # 第二个全连接层

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)  # 第一个全连接层计算
        hidden_states = self.activation_fn(hidden_states)  # 应用激活函数
        hidden_states = self.fc2(hidden_states)  # 第二个全连接层计算
        return hidden_states



# Copied from transformers.models.clip.modeling_clip.CLIPEncoderLayer with CLIP->IdeficsVision
class IdeficsVisionEncoderLayer(nn.Module):
    def __init__(self, config: IdeficsVisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size  # 嵌入维度等于隐藏大小
        self.self_attn = IdeficsVisionAttention(config)  # 自注意力机制
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)  # 第一个层归一化
        self.mlp = IdeficsVisionMLP(config)  # 多层感知器网络
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)  # 第二个层归一化

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        causal_attention_mask: torch.Tensor,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
                `(config.encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states  # 保存输入的原始 hidden_states 作为残差连接的一部分

        hidden_states = self.layer_norm1(hidden_states)  # 对输入进行 Layer Normalization

        # 调用 self-attention 层计算新的 hidden_states 和注意力权重
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
        )

        hidden_states = residual + hidden_states  # 将残差与经过 self-attention 后的结果相加

        residual = hidden_states  # 更新残差连接的变量为当前的 hidden_states

        hidden_states = self.layer_norm2(hidden_states)  # 对新的 hidden_states 进行 Layer Normalization

        hidden_states = self.mlp(hidden_states)  # 通过 MLP 进行全连接网络的处理

        hidden_states = residual + hidden_states  # 将残差与经过 MLP 后的结果相加

        outputs = (hidden_states,)  # 将处理后的 hidden_states 包装成 tuple 输出

        if output_attentions:
            outputs += (attn_weights,)  # 如果需要输出注意力权重，将它们也加入输出的 tuple 中

        return outputs  # 返回输出的 tuple，其中包括处理后的 hidden_states 和可能的注意力权重
# 从transformers.models.clip.modeling_clip.CLIPEncoder复制，将CLIP更改为IdeficsVision
class IdeficsVisionEncoder(nn.Module):
    """
    IdeficsVision编码器，由`config.num_hidden_layers`个自注意力层组成。每一层都是一个[`IdeficsVisionEncoderLayer`]。

    Args:
        config: IdeficsVisionConfig
    """

    def __init__(self, config: IdeficsVisionConfig):
        super().__init__()
        self.config = config
        # 创建包含`config.num_hidden_layers`个IdeficsVisionEncoderLayer的模块列表
        self.layers = nn.ModuleList([IdeficsVisionEncoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(
        self,
        inputs_embeds,
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,



# 从transformers.models.clip.modeling_clip.CLIPVisionTransformer调整而来
class IdeficsVisionTransformer(nn.Module):
    def __init__(self, config: IdeficsVisionConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        # 初始化嵌入层和LayerNorm
        self.embeddings = IdeficsVisionEmbeddings(config)
        self.pre_layrnorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        # 初始化IdeficsVision编码器
        self.encoder = IdeficsVisionEncoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

    # 从transformers.models.clip.modeling_clip.CLIPVisionTransformer.forward适应而来
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: Optional[bool] = False,
        return_dict: Optional[bool] = None,
        ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        返回模型的输出结果。

        """
        # 如果未指定output_attentions，则使用配置中的output_attentions参数
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 如果未指定output_hidden_states，则使用配置中的output_hidden_states参数
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 如果未指定return_dict，则使用配置中的use_return_dict参数
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 如果pixel_values为None，则抛出数值错误异常
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # 将像素值嵌入到模型的嵌入层中，如果指定了interpolate_pos_encoding，则插值位置编码
        hidden_states = self.embeddings(pixel_values, interpolate_pos_encoding=interpolate_pos_encoding)
        # 在嵌入层之后应用预层归一化
        hidden_states = self.pre_layrnorm(hidden_states)

        # 使用编码器对嵌入的输入进行编码
        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取编码器输出的最后隐藏状态
        last_hidden_state = encoder_outputs[0]
        # 提取汇聚输出，即取每个样本的第一个位置的隐藏状态
        pooled_output = last_hidden_state[:, 0, :]
        # 在汇聚输出后应用后层归一化
        pooled_output = self.post_layernorm(pooled_output)

        # 如果return_dict为False，则返回元组形式的输出
        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        # 如果return_dict为True，则返回BaseModelOutputWithPooling对象
        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
```