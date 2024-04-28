# `.\models\idefics\vision.py`

```
# 设置文件编码为 utf-8
# 版权声明，版权归 The OpenAI Team Authors 和 The HuggingFace Team 所有
# 根据 Apache 许可证 2.0 版本授权
# 除非符合许可证规定，否则不得使用此文件
# 您可以在以下网址获取许可证的副本：http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于“原样”基础分发的，没有任何明示或暗示的担保或条件
# 请查看许可证以获取有关特定语言的权限和限制
""" PyTorch IdeficsVision model: a copy of CLIPVisionModel using a simpler config object"""

# 导入必要的库
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn

# 导入相关模块
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from ...utils import ModelOutput, logging
from .configuration_idefics import IdeficsVisionConfig

# 获取日志记录器
logger = logging.get_logger(__name__)

# 定义一个数据类，用于存储 IdeficsVision 模型的输出
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
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True` is passed):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    image_embeds: Optional[torch.FloatTensor] = None
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
# 从transformers.models.clip.modeling_clip.CLIPVisionEmbeddings适配而来的类
class IdeficsVisionEmbeddings(nn.Module):
    def __init__(self, config: IdeficsVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        # 初始化类别嵌入向量
        self.class_embedding = nn.Parameter(torch.randn(self.embed_dim))

        # 初始化图像块嵌入层
        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=False,
        )

        # 计算图像中的块数和位置数
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1
        # 初始化位置嵌入层
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        # 注册位置ID张量
        self.register_buffer("position_ids", torch.arange(self.num_positions).expand((1, -1)), persistent=False)

    # 从https://github.com/huggingface/transformers/blob/v4.33.0/src/transformers/models/vit/modeling_vit.py#L82中得到启发的部分
    def interpolate_pos_encoding(self, embeddings: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """
        This method allows to interpolate the pre-trained position encodings, to be able to use the model on higher
        resolution images.

        Source:
        https://github.com/facebookresearch/dino/blob/de9ee3df6cf39fac952ab558447af1fa1365362a/vision_transformer.py#L174
        """

        # 计算嵌入张量中的补丁数量
        num_patches = embeddings.shape[1] - 1
        # 获取位置编码
        pos_embed = self.position_embedding(self.position_ids)
        # 获取位置编码的位置数量
        num_positions = pos_embed.shape[1] - 1
        # 如果补丁数量等于位置数量且高度等于宽度，则直接返回位置编码
        if num_patches == num_positions and height == width:
            return pos_embed
        # 分离类别位置编码和补丁位置编码
        class_pos_embed = pos_embed[:, 0]
        patch_pos_embed = pos_embed[:, 1:]

        # 获取嵌入维度
        embed_dim = embeddings.shape[-1]
        num_h_patches = height // self.config.patch_size
        num_w_patches = width // self.config.patch_size
        # 为了避免插值时的浮点误差，添加一个小数
        num_h_patches, num_w_patches = num_h_patches + 0.1, num_w_patches + 0.1
        sqrt_num_positions = math.sqrt(num_positions)
        # 重塑补丁位置编码的形状
        patch_pos_embed = patch_pos_embed.reshape(1, int(sqrt_num_positions), int(sqrt_num_positions), embed_dim)
        patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)
        fp32_upcasting = patch_pos_embed.dtype == torch.bfloat16
        if fp32_upcasting:
            # 如果是 torch.bfloat16 类型，则将其转换为 torch.float 类型
            logger.warning_once(
                "Upcasting patch_pos_embed to fp32 for interpolation since `upsample_bicubic2d_out_frame` in nn.functional.interpolate "
                "is not implemented for 'torch.bfloat16' dtype. This will result in a slight overhead."
            )
            patch_pos_embed = patch_pos_embed.to(torch.float)
        # 插值补丁位置编码
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed,
            scale_factor=(num_h_patches / sqrt_num_positions, num_w_patches / sqrt_num_positions),
            mode="bicubic",
            align_corners=False,
        )
        if fp32_upcasting:
            # 如果之前转换为 torch.float 类型，则再转换回 torch.bfloat16 类型
            patch_pos_embed = patch_pos_embed.to(torch.bfloat16)
        # 检查插值后的形状是否匹配
        if int(num_h_patches) != patch_pos_embed.shape[-2] or int(num_w_patches) != patch_pos_embed.shape[-1]:
            raise ValueError(
                f"Number of patches for images ({int(num_h_patches), int(num_w_patches)}) don't match the "
                f"shape of position embedding ({patch_pos_embed.shape[-2], patch_pos_embed.shape[-1]})"
            )
        # 重新排列补丁位置编码的形状
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, embed_dim)
        # 返回连接类别位置编码和补丁位置编码的结果
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)
    # 定义一个前向传播函数，接受像素值和是否插值位置编码作为参数，返回处理后的张量
    def forward(self, pixel_values: torch.FloatTensor, interpolate_pos_encoding: bool = False) -> torch.Tensor:
        # 获取输入像素值张量的形状信息
        batch_size, num_channels, height, width = pixel_values.shape
        # 如果不需要插值位置编码
        if not interpolate_pos_encoding:
            # 检查输入图像大小是否与模型要求的大小一致，如果不一致则抛出数值错误
            if height != self.image_size or width != self.image_size:
                raise ValueError(
                    f"Input image size ({height}*{width}) doesn't match model"
                    f" ({self.image_size}*{self.image_size}). You should try to set `interpolate_pos_encoding=True`"
                )

        # 获取目标数据类型为嵌入权重的数据类型
        target_dtype = self.patch_embedding.weight.dtype
        # 使用嵌入权重对像素值进行嵌入，得到补丁嵌入张量，形状为 [*, width, grid, grid]
        patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype))

        # 将补丁嵌入张量展平并转置
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)

        # 扩展类别嵌入以匹配批次大小
        class_embeds = self.class_embedding.expand(batch_size, 1, -1)
        # 将类别嵌入和补丁嵌入拼接在一起
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)

        # 如果需要插值位置编码
        if interpolate_pos_encoding:
            # 将位置编码插值到每个令牌中
            embeddings = embeddings + self.interpolate_pos_encoding(embeddings, height, width)
        else:
            # 将位置编码添加到每个令牌中
            embeddings = embeddings + self.position_embedding(self.position_ids)

        # 返回处理后的张量
        return embeddings
# 从transformers.models.clip.modeling_clip中复制CLIPAttention类，并将CLIP更改为IdeficsVision
class IdeficsVisionAttention(nn.Module):
    """来自'Attention Is All You Need'论文的多头注意力机制"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim必须能被num_heads整除（得到`embed_dim`：{self.embed_dim}和`num_heads`：{self.num_heads}）."
            )
        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
# 从transformers.models.clip.modeling_clip中复制CLIPMLP类，并将CLIP更改为IdeficsVision
class IdeficsVisionMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.activation_fn = ACT2FN[config.hidden_act]
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


# 从transformers.models.clip.modeling_clip中复制CLIPEncoderLayer类，并将CLIP更改为IdeficsVision
class IdeficsVisionEncoderLayer(nn.Module):
    def __init__(self, config: IdeficsVisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = IdeficsVisionAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = IdeficsVisionMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

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
        # 保存输入 hidden_states 作为残差连接的基准
        residual = hidden_states

        # 对 hidden_states 进行 Layer Normalization
        hidden_states = self.layer_norm1(hidden_states)
        # 使用 self-attention 层处理 hidden_states，并返回注意力权重
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
        )
        # 将残差连接和 self-attention 处理后的结果相加
        hidden_states = residual + hidden_states

        # 保存当前 hidden_states 作为残差连接的基准
        residual = hidden_states
        # 对 hidden_states 进行第二次 Layer Normalization
        hidden_states = self.layer_norm2(hidden_states)
        # 使用 MLP 处理 hidden_states
        hidden_states = self.mlp(hidden_states)
        # 将残差连接和 MLP 处理后的结果相加
        hidden_states = residual + hidden_states

        # 将处理后的 hidden_states 存入 outputs
        outputs = (hidden_states,)

        # 如果需要输出注意力权重，则将注意力权重加入 outputs
        if output_attentions:
            outputs += (attn_weights,)

        # 返回 outputs
        return outputs
# 从transformers.models.clip.modeling_clip.CLIPEncoder复制而来，将CLIP替换为IdeficsVision
class IdeficsVisionEncoder(nn.Module):
    """
    由`config.num_hidden_layers`个自注意力层组成的Transformer编码器。每一层都是一个[`IdeficsVisionEncoderLayer`]。

    Args:
        config: IdeficsVisionConfig
    """

    def __init__(self, config: IdeficsVisionConfig):
        super().__init__()
        self.config = config
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

        self.embeddings = IdeficsVisionEmbeddings(config)
        self.pre_layrnorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        self.encoder = IdeficsVisionEncoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

    # 从transformers.models.clip.modeling_clip.CLIPVisionTransformer.forward调整而来
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: Optional[bool] = False,
        return_dict: Optional[bool] = None,
    # 定义函数的返回类型为元组或BaseModelOutputWithPooling类型
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:

        """
        # 如果output_attentions为None，则使用self.config.output_attentions
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 如果output_hidden_states为None，则使用self.config.output_hidden_states
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 如果return_dict为None，则使用self.config.use_return_dict
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 如果pixel_values为None，则抛出数值错误
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # 使用embeddings方法将pixel_values转换为hidden_states
        hidden_states = self.embeddings(pixel_values, interpolate_pos_encoding=interpolate_pos_encoding)
        # 对hidden_states进行预层归一化
        hidden_states = self.pre_layrnorm(hidden_states)

        # 使用encoder对hidden_states进行编码
        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取编码器的最后隐藏状态和池化输出
        last_hidden_state = encoder_outputs[0]
        pooled_output = last_hidden_state[:, 0, :]
        # 对池化输出进行后层归一化
        pooled_output = self.post_layernorm(pooled_output)

        # 如果return_dict为False，则返回元组
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