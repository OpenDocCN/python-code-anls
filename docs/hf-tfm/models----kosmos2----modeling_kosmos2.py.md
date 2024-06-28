# `.\models\kosmos2\modeling_kosmos2.py`

```
# 设置文件编码为 UTF-8
# 版权声明，声明版权归 Microsoft Research 和 HuggingFace Inc. 团队所有
#
# 根据 Apache License, Version 2.0 许可，除非符合许可要求，否则不得使用此文件
# 您可以在以下网址获取许可证的副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则本软件按"原样"分发，不附带任何明示或暗示的担保或条件
# 请参阅许可证以了解特定语言的权限和限制

""" PyTorch KOSMOS-2 model."""

# 导入必要的库和模块
import math
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss

# 导入与激活函数相关的模块
from ...activations import ACT2FN
# 导入不同类型的模型输出
from ...modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPooling,
    CausalLMOutputWithCrossAttentions,
)
# 导入预训练模型的基类
from ...modeling_utils import PreTrainedModel
# 导入实用工具函数
from ...utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
# 导入 KOSMOS-2 的配置类
from .configuration_kosmos2 import Kosmos2Config, Kosmos2TextConfig, Kosmos2VisionConfig

# 获取日志记录器对象
logger = logging.get_logger(__name__)

# 文档用的配置对象
_CONFIG_FOR_DOC = Kosmos2Config

# 预训练模型存档列表
KOSMOS2_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "microsoft/kosmos-2-patch14-224",
    # 可以在 https://huggingface.co/models?filter=kosmos-2 查看所有 KOSMOS-2 模型
]

# 定义函数：将注意力掩码从 `[bsz, seq_len]` 扩展到 `[bsz, 1, tgt_seq_len, src_seq_len]`
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    # 扩展注意力掩码
    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    # 创建反向的掩码
    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


# 定义函数：创建用于双向自注意力的因果掩码
def _make_causal_mask(
    input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)

# 从 transformers.models.roberta.modeling_roberta.create_position_ids_from_input_ids 复制过来的部分
# 定义一个函数，根据输入的 token IDs 创建位置 ID，用于替换非填充符号为它们的位置编号。位置编号从 padding_idx+1 开始计数，忽略填充符号。
def create_position_ids_from_input_ids(input_ids, padding_idx, past_key_values_length=0):
    """
    Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding symbols
    are ignored. This is modified from fairseq's `utils.make_positions`.

    Args:
        input_ids (torch.Tensor): 输入的 token IDs
        padding_idx (int): 填充符号的索引
        past_key_values_length (int, optional): 过去键值长度，用于增量索引计算

    Returns:
        torch.Tensor: 替换后的位置 ID
    """
    # 在这里进行一系列的类型转换和转换，以确保同时支持 ONNX 导出和 XLA。
    mask = input_ids.ne(padding_idx).int()  # 创建一个掩码，标记非填充符号的位置
    incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask  # 计算增量索引
    return incremental_indices.long() + padding_idx  # 返回最终的位置 ID，加上 padding_idx 得到真实的位置编号


KOSMOS2_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`Kosmos2Config`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

KOSMOS2_VISION_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See
            [`CLIPImageProcessor.__call__`] for details.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""

KOSMOS2_TEXT_INPUTS_DOCSTRING = r"""
    Args:
"""

KOSMOS2_INPUTS_DOCSTRING = r"""
    Args:
"""


@dataclass
class Kosmos2ModelOutput(ModelOutput):
    """
    Base class for text model's outputs that also contains a pooling of the last hidden states.
    """
    # 最后一层模型的隐藏状态，形状为(batch_size, sequence_length, hidden_size)
    last_hidden_state: torch.FloatTensor = None
    
    # 过去的键-值对，可选参数，形状为(config.n_layers, 2, batch_size, num_heads, sequence_length, embed_size_per_head)，用于加速顺序解码
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    
    # 模型每一层的隐藏状态的元组，如果模型有嵌入层，则包括嵌入层输出，形状为(batch_size, sequence_length, hidden_size)
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    
    # 自注意力机制每一层的注意力权重的元组，形状为(batch_size, num_heads, sequence_length, sequence_length)
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    
    # 图像嵌入的隐藏状态，形状为(batch_size, latent_query_num, hidden_size)，可选参数
    image_embeds: Optional[torch.FloatTensor] = None
    # 定义一个可选类型的变量 projection_attentions，可能是一个包含 torch.FloatTensor 的元组，初始值为 None
    projection_attentions: Optional[Tuple[torch.FloatTensor]] = None
    
    # 定义一个变量 vision_model_output，类型为 BaseModelOutputWithPooling，初始值为 None
    vision_model_output: BaseModelOutputWithPooling = None
    
    # 定义一个方法 to_tuple，返回一个元组，包含对象所有键对应的值，但对于键为"text_model_output"和"vision_model_output"的情况，返回它们的 to_tuple() 方法的结果
    def to_tuple(self) -> Tuple[Any]:
        return tuple(
            self[k] if k not in ["text_model_output", "vision_model_output"] else getattr(self, k).to_tuple()
            for k in self.keys()
        )
@dataclass
class Kosmos2ForConditionalGenerationModelOutput(ModelOutput):
    """
    Model output class for `Kosmos2ForConditionalGeneration`.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        image_embeds (`torch.FloatTensor` of shape `(batch_size, latent_query_num, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of `Kosmos2ImageToTextProjection`.
        projection_attentions (`tuple(torch.FloatTensor)`, *optional*):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights given by `Kosmos2ImageToTextProjection`, after the attention softmax, used to compute
            the weighted average in the self-attention heads.
        vision_model_output(`BaseModelOutputWithPooling`, *optional*):
            The output of the [`Kosmos2VisionModel`].
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and optionally if
            `config.is_encoder_decoder=True` 2 additional tensors of shape `(batch_size, num_heads,
            encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and optionally if
            `config.is_encoder_decoder=True` in the cross-attention blocks) that can be used (see `past_key_values`
            input) to speed up sequential decoding.
    """
    # 定义可选的损失张量
    loss: Optional[torch.FloatTensor] = None
    # 定义空的 logits 张量
    logits: torch.FloatTensor = None
    # 定义可选的过去键值元组，包含 FloatTensor
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    # 定义可选的隐藏状态元组，包含 FloatTensor
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # 定义可选的注意力元组，包含 FloatTensor
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    # 定义可选的图像嵌入张量
    image_embeds: Optional[torch.FloatTensor] = None
    # 定义可选的投影注意力元组，包含 FloatTensor
    projection_attentions: Optional[Tuple[torch.FloatTensor]] = None
    # 定义空的视觉模型输出，类型为 BaseModelOutputWithPooling
    vision_model_output: BaseModelOutputWithPooling = None

    # 转换为元组的方法，返回包含所有非 "text_model_output" 和 "vision_model_output" 的属性的元组
    def to_tuple(self) -> Tuple[Any]:
        return tuple(
            self[k] if k not in ["text_model_output", "vision_model_output"] else getattr(self, k).to_tuple()
            for k in self.keys()
        )
# 从transformers.models.clip.modeling_clip.CLIPVisionEmbeddings复制而来，修改为Kosmos2
class Kosmos2VisionEmbeddings(nn.Module):
    def __init__(self, config: Kosmos2VisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size  # 设置嵌入维度为配置文件中的隐藏大小
        self.image_size = config.image_size  # 设置图像大小为配置文件中的图像大小
        self.patch_size = config.patch_size  # 设置补丁大小为配置文件中的补丁大小

        self.class_embedding = nn.Parameter(torch.randn(self.embed_dim))  # 定义类别嵌入作为可学习参数

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=False,
        )  # 定义补丁嵌入为二维卷积层，用于从图像像素值生成嵌入向量

        self.num_patches = (self.image_size // self.patch_size) ** 2  # 计算图像中的补丁数量
        self.num_positions = self.num_patches + 1  # 计算位置嵌入的数量
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)  # 定义位置嵌入为一个嵌入层
        self.register_buffer("position_ids", torch.arange(self.num_positions).expand((1, -1)), persistent=False)  # 注册位置 ID，用于序列位置编码

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        batch_size = pixel_values.shape[0]  # 获取批次大小
        target_dtype = self.patch_embedding.weight.dtype  # 获取目标数据类型
        patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype))  # 使用补丁嵌入层处理像素值，生成补丁嵌入向量

        # 展开补丁嵌入向量并进行维度转换
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)

        class_embeds = self.class_embedding.expand(batch_size, 1, -1)  # 扩展类别嵌入以适应批次大小
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)  # 连接类别嵌入和补丁嵌入，形成最终嵌入向量
        embeddings = embeddings + self.position_embedding(self.position_ids)  # 添加位置嵌入到最终嵌入向量中
        return embeddings


# 从transformers.models.clip.modeling_clip.CLIPAttention复制而来，修改为Kosmos2Vision
class Kosmos2VisionAttention(nn.Module):
    """来自 'Attention Is All You Need' 论文的多头注意力机制"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size  # 设置嵌入维度为配置文件中的隐藏大小
        self.num_heads = config.num_attention_heads  # 设置注意力头数为配置文件中的注意力头数
        self.head_dim = self.embed_dim // self.num_heads  # 计算每个注意力头的维度
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: "
                f"{self.num_heads})."
            )
        self.scale = self.head_dim**-0.5  # 缩放因子，用于缩放注意力分数
        self.dropout = config.attention_dropout  # 设置注意力层的dropout比例

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)  # 初始化键的投影层
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)  # 初始化值的投影层
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)  # 初始化查询的投影层
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)  # 初始化输出的投影层

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    # 定义一个方法 `forward`，用于模型前向传播操作
    def forward(
        self,
        # 输入参数：表示模型当前隐藏状态的张量
        hidden_states: torch.Tensor,
        # 输入参数：可选的注意力掩码张量，用于指示哪些位置需要注意
        attention_mask: Optional[torch.Tensor] = None,
        # 输入参数：可选的因果注意力掩码张量，用于自回归任务的自注意力
        causal_attention_mask: Optional[torch.Tensor] = None,
        # 输入参数：是否输出注意力权重，默认为 False
        output_attentions: Optional[bool] = False,
# Copied from transformers.models.clip.modeling_clip.CLIPMLP with CLIP->Kosmos2Vision
class Kosmos2VisionMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.activation_fn = ACT2FN[config.hidden_act]  # 使用配置中的激活函数
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)  # 创建全连接层 fc1
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)  # 创建全连接层 fc2

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)  # 应用 fc1
        hidden_states = self.activation_fn(hidden_states)  # 应用激活函数
        hidden_states = self.fc2(hidden_states)  # 应用 fc2
        return hidden_states


# Copied from transformers.models.clip.modeling_clip.CLIPEncoderLayer with CLIP->Kosmos2Vision
class Kosmos2VisionEncoderLayer(nn.Module):
    def __init__(self, config: Kosmos2VisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size  # 设置嵌入维度为隐藏尺寸
        self.self_attn = Kosmos2VisionAttention(config)  # 创建 Kosmos2VisionAttention 自注意力层
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)  # 创建 LayerNorm 层1
        self.mlp = Kosmos2VisionMLP(config)  # 创建 Kosmos2VisionMLP 多层感知器
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)  # 创建 LayerNorm 层2

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
            causal_attention_mask (`torch.FloatTensor`): mask indicating the causal nature of attention
            output_attentions (`bool`, *optional*): Whether or not to return the attentions tensors of all attention layers.
        """
        residual = hidden_states  # 记录残差连接

        hidden_states = self.layer_norm1(hidden_states)  # 应用 LayerNorm 层1
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
        )  # 应用自注意力机制层，并返回注意力权重
        hidden_states = residual + hidden_states  # 残差连接

        residual = hidden_states  # 记录残差连接
        hidden_states = self.layer_norm2(hidden_states)  # 应用 LayerNorm 层2
        hidden_states = self.mlp(hidden_states)  # 应用 MLP 层
        hidden_states = residual + hidden_states  # 残差连接

        outputs = (hidden_states,)  # 输出结果为 hidden_states

        if output_attentions:
            outputs += (attn_weights,)  # 如果需要输出注意力权重，加入输出结果

        return outputs


# Copied from transformers.models.clip.modeling_clip.CLIPEncoder with CLIP->Kosmos2Vision
class Kosmos2VisionEncoder(nn.Module):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self attention layers. Each layer is a
    """
    # 定义 Kosmos2VisionEncoderLayer 类，用于处理 Kosmos2Vision 模型的编码器层
    class Kosmos2VisionEncoderLayer(nn.Module):

        # 初始化方法，接收一个 Kosmos2VisionConfig 类型的配置对象作为参数
        def __init__(self, config: Kosmos2VisionConfig):
            # 调用父类的初始化方法
            super().__init__()
            # 将传入的配置对象保存到实例变量中
            self.config = config
            # 创建一个包含多个 Kosmos2VisionEncoderLayer 实例的列表，列表长度为 config.num_hidden_layers
            self.layers = nn.ModuleList([Kosmos2VisionEncoderLayer(config) for _ in range(config.num_hidden_layers)])
            # 设置梯度检查点标志为 False
            self.gradient_checkpointing = False

        # 前向传播方法，接收多个参数
        def forward(
            self,
            inputs_embeds,
            attention_mask: Optional[torch.Tensor] = None,
            causal_attention_mask: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
# 类定义，实现了一个类似于 `transformers.models.clip.modeling_clip.CLIPVisionTransformer` 的模型，但没有为 `forward` 方法添加文档字符串
class Kosmos2VisionTransformer(nn.Module):
    # 构造函数，接受一个 `Kosmos2VisionConfig` 类型的参数 `config`
    # 初始化父类 `nn.Module`
    def __init__(self, config: Kosmos2VisionConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        # 实例化 `Kosmos2VisionEmbeddings` 类，用于嵌入层处理
        self.embeddings = Kosmos2VisionEmbeddings(config)
        # LayerNorm 层，对嵌入向量进行归一化
        self.pre_layrnorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        # `Kosmos2VisionEncoder` 类，用于编码器的处理
        self.encoder = Kosmos2VisionEncoder(config)
        # 再次应用 LayerNorm 层，对输出进行归一化
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

    # 前向传播函数，接受多个参数，返回一个元组或者 `BaseModelOutputWithPooling` 类型
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        # 如果 `output_attentions` 未指定，则使用配置中的默认值
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 如果 `output_hidden_states` 未指定，则使用配置中的默认值
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 如果 `return_dict` 未指定，则使用配置中的默认值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 如果 `pixel_values` 为空，则抛出值错误异常
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # 通过嵌入层处理 `pixel_values`，得到隐藏状态
        hidden_states = self.embeddings(pixel_values)
        # 对隐藏状态应用预 LayerNorm 层进行归一化
        hidden_states = self.pre_layrnorm(hidden_states)

        # 将归一化后的隐藏状态传递给编码器 `self.encoder` 进行编码
        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取编码器输出的最后一个隐藏状态
        last_hidden_state = encoder_outputs[0]
        # 从最后一个隐藏状态中提取池化输出，通常是第一个位置的输出
        pooled_output = last_hidden_state[:, 0, :]
        # 对池化输出应用后 LayerNorm 层进行归一化
        pooled_output = self.post_layernorm(pooled_output)

        # 如果不需要返回字典，则返回一个元组，包含最后一个隐藏状态、池化输出以及额外的编码器输出
        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        # 否则，返回一个 `BaseModelOutputWithPooling` 对象，包含最后一个隐藏状态、池化输出、所有隐藏状态以及注意力权重
        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


# 类定义，实现了一个类似于 `transformers.models.m2m_100.modeling_m2m_100.M2M100SinusoidalPositionalEmbedding` 的模块，但允许传递 `position_ids`
class Kosmos2TextSinusoidalPositionalEmbedding(nn.Module):
    """This module produces sinusoidal positional embeddings of any length."""

    # 构造函数，无参数，继承自 `nn.Module`
    # 此处省略了具体的初始化过程
    # 初始化函数，用于设置位置编码的参数
    def __init__(self, num_positions: int, embedding_dim: int, padding_idx: Optional[int] = None):
        super().__init__()
        # 设置位置编码的偏移量为2
        self.offset = 2
        # 设定位置编码的维度
        self.embedding_dim = embedding_dim
        # 可选参数：填充索引
        self.padding_idx = padding_idx
        # 调用make_weights方法生成位置编码权重
        self.make_weights(num_positions + self.offset, embedding_dim, padding_idx)

    # 静态方法：从transformers.models.m2m_100.modeling_m2m_100.M2M100SinusoidalPositionalEmbedding类中复制得到
    # 生成位置编码权重的方法
    def make_weights(self, num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None):
        # 调用get_embedding方法获取嵌入向量权重
        emb_weights = self.get_embedding(num_embeddings, embedding_dim, padding_idx)
        # 如果self中已经有了weights属性，则在forward方法中将权重转换成正确的数据类型和设备
        if hasattr(self, "weights"):
            emb_weights = emb_weights.to(dtype=self.weights.dtype, device=self.weights.device)

        # 将生成的权重注册为缓冲区，不持久化保存
        self.register_buffer("weights", emb_weights, persistent=False)

    # 静态方法：从transformers.models.m2m_100.modeling_m2m_100.M2M100SinusoidalPositionalEmbedding类中复制得到
    # 生成嵌入向量的方法
    @staticmethod
    def get_embedding(num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None):
        """
        构建正弦位置编码的嵌入向量。

        该方法与tensor2tensor中的实现匹配，但与《Attention Is All You Need》中第3.5节的描述略有不同。
        """
        # 计算嵌入向量的半径
        half_dim = embedding_dim // 2
        # 计算正弦函数的周期
        emb = math.log(10000) / (half_dim - 1)
        # 计算正弦位置编码的指数值
        emb = torch.exp(torch.arange(half_dim, dtype=torch.int64).float() * -emb)
        # 计算位置编码张量
        emb = torch.arange(num_embeddings, dtype=torch.int64).float().unsqueeze(1) * emb.unsqueeze(0)
        # 拼接正弦和余弦函数，生成最终的位置编码张量
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
        # 如果嵌入维度是奇数，则在末尾填充零
        if embedding_dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        # 如果指定了填充索引，则将该位置的嵌入向量置为零向量
        if padding_idx is not None:
            emb[padding_idx, :] = 0

        return emb.to(torch.get_default_dtype())

    # 用于前向传播计算的方法，设置位置编码
    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor = None,
        inputs_embeds: torch.Tensor = None,
        past_key_values_length: int = 0,
        position_ids: torch.Tensor = None,
    ):
        # 如果传入了 input_ids 参数
        if input_ids is not None:
            # 获取 batch size 和 sequence length
            bsz, seq_len = input_ids.size()
            # 如果 position_ids 参数为 None
            if position_ids is None:
                # 根据输入的 token ids 创建 position ids。任何填充的 token 保持填充状态。
                position_ids = create_position_ids_from_input_ids(
                    input_ids, self.padding_idx, past_key_values_length
                ).to(input_ids.device)
        else:
            # 获取 batch size 和 sequence length，排除最后一维
            bsz, seq_len = inputs_embeds.size()[:-1]
            # 如果 position_ids 参数为 None
            if position_ids is None:
                # 根据 inputs_embeds 和 past_key_values_length 创建 position ids
                position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds, past_key_values_length)

        # 如果需要扩展 embeddings
        max_pos = self.padding_idx + 1 + seq_len + past_key_values_length
        if max_pos > self.weights.size(0):
            # 根据最大位置和偏移量，以及 embedding 维度和填充索引，创建新的 weights
            self.make_weights(max_pos + self.offset, self.embedding_dim, self.padding_idx)

        # 根据 position_ids 从 weights 中选择对应的 embeddings，并重新组织形状
        return self.weights.index_select(0, position_ids.view(-1)).view(bsz, seq_len, self.weights.shape[-1]).detach()

    # 从 transformers.models.m2m_100.modeling_m2m_100.M2M100SinusoidalPositionalEmbedding.create_position_ids_from_inputs_embeds 复制而来
    def create_position_ids_from_inputs_embeds(self, inputs_embeds, past_key_values_length):
        """
        直接提供 embeddings。无法推断哪些是填充的，因此生成顺序的 position ids。

        Args:
            inputs_embeds: torch.Tensor

        Returns: torch.Tensor
        """
        # 获取输入 embeddings 的形状，排除最后一维
        input_shape = inputs_embeds.size()[:-1]
        # 获取序列长度
        sequence_length = input_shape[1]

        # 根据序列长度、padding_idx 和设备类型，在设备上创建 long 类型的序列 tensor
        position_ids = torch.arange(
            self.padding_idx + 1, sequence_length + self.padding_idx + 1, dtype=torch.long, device=inputs_embeds.device
        )
        # 扩展 position_ids 的形状以匹配 inputs_embeds，并确保连续性，加上 past_key_values_length
        return position_ids.unsqueeze(0).expand(input_shape).contiguous() + past_key_values_length
class KosmosTextAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    # Similar to transformers.models.bart.modeling_bart.BartAttention.__init__ except an additional `inner_attn_ln`.
    def __init__(
        self,
        config,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        add_inner_attn_layernorm: bool = False,
        bias: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim  # 设置模型的嵌入维度
        self.num_heads = num_heads  # 设置注意力头的数量
        self.dropout = dropout  # 设置dropout概率
        self.head_dim = embed_dim // num_heads  # 计算每个注意力头的维度

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5  # 缩放因子
        self.is_decoder = is_decoder  # 是否为解码器

        # 初始化线性投影层
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        # 添加内部注意力层规范化
        self.inner_attn_ln = None
        if add_inner_attn_layernorm:
            self.inner_attn_ln = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

    def _shape(self, projection: torch.Tensor) -> torch.Tensor:
        new_projection_shape = projection.size()[:-1] + (self.num_heads, self.head_dim)
        # 将投影重新形状以适应多头注意力的结构
        # (B, T, H * D) -> (B, T, H, D) -> (B, H, T, D)
        new_projection = projection.view(new_projection_shape).permute(0, 2, 1, 3)
        return new_projection

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    # 定义神经网络的前向传播方法，接收隐藏状态作为输入
    def forward(self, hidden_states):
        # 将隐藏状态输入全连接层 fc1，并应用激活函数 activation_fn
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        # 对隐藏状态进行 dropout 操作，以防止过拟合，根据训练状态决定是否执行
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        # 对经过 dropout 后的隐藏状态进行层归一化处理
        hidden_states = self.ffn_layernorm(hidden_states)
        # 将归一化后的隐藏状态输入全连接层 fc2
        hidden_states = self.fc2(hidden_states)
        # 对最终输出的隐藏状态再次进行 dropout 操作，根据训练状态决定是否执行
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        # 返回经过前向传播后的隐藏状态
        return hidden_states
# 定义一个名为 Kosmos2TextBlock 的神经网络模块，继承自 nn.Module
class Kosmos2TextBlock(nn.Module):
    # 初始化函数，接受一个名为 config 的 Kosmos2TextConfig 类型参数
    def __init__(self, config: Kosmos2TextConfig):
        # 调用父类 nn.Module 的初始化方法
        super().__init__()
        # 设置嵌入维度为 config 中的 embed_dim
        self.embed_dim = config.embed_dim

        # 创建自注意力层 KosmosTextAttention 对象
        self.self_attn = KosmosTextAttention(
            config,
            embed_dim=self.embed_dim,
            num_heads=config.attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
            add_inner_attn_layernorm=True,
        )
        
        # 设置 dropout 概率
        self.dropout = config.dropout
        # 创建自注意力层的 LayerNorm 层
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

        # 如果配置中包含交叉注意力设置
        if config.add_cross_attention:
            # 创建编码器注意力层 KosmosTextAttention 对象
            self.encoder_attn = KosmosTextAttention(
                config,
                embed_dim=self.embed_dim,
                num_heads=config.attention_heads,
                dropout=config.attention_dropout,
                is_decoder=True,
                add_inner_attn_layernorm=False,
            )
            # 创建编码器注意力层的 LayerNorm 层
            self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

        # 创建前馈神经网络对象 Kosmos2TextFFN
        self.ffn = Kosmos2TextFFN(config)
        # 创建最终输出层的 LayerNorm 层
        self.final_layer_norm = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    # 前向传播函数，接受多个输入参数
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        cross_attn_layer_head_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = True,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        # 保存原始的隐藏状态作为残差连接的基础
        residual = hidden_states

        # Self Attention
        # 如果有过去的键/值缓存，从中提取decoder单向self-attention的缓存键/值对，位置为1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None

        # 对隐藏状态进行 layer normalization
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # 使用self-attention机制处理隐藏状态
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=self_attn_past_key_value,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )

        # 对输出的隐藏状态进行dropout处理
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        
        # 将残差连接到当前隐藏状态上
        hidden_states = residual + hidden_states

        # Cross-Attention Block
        cross_attn_present_key_value = None
        cross_attn_weights = None

        # 如果有encoder的隐藏状态
        if encoder_hidden_states is not None:
            # 检查是否存在cross-attention层，若不存在则抛出异常
            if not hasattr(self, "encoder_attn"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers"
                    " by setting `config.add_cross_attention=True`"
                )

            # 保存当前隐藏状态作为残差连接的基础
            residual = hidden_states

            # 对隐藏状态进行 layer normalization
            hidden_states = self.encoder_attn_layer_norm(hidden_states)

            # 如果有过去的键/值缓存，从中提取cross-attention的缓存键/值对，位置为3,4
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None

            # 使用cross-attention机制处理隐藏状态
            hidden_states, cross_attn_weights, cross_attn_present_key_value = self.encoder_attn(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=cross_attn_past_key_value,
                output_attentions=output_attentions,
            )

            # 对输出的隐藏状态进行dropout处理
            hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

            # 将残差连接到当前隐藏状态上
            hidden_states = residual + hidden_states

            # 将cross-attention的键/值对添加到当前的present_key_value中，位置为3,4
            present_key_value = present_key_value + cross_attn_present_key_value

        # Fully Connected
        # 保存当前隐藏状态作为残差连接的基础
        residual = hidden_states

        # 对隐藏状态进行 layer normalization
        hidden_states = self.final_layer_norm(hidden_states)

        # Feed Forward Network (FFN)
        hidden_states = self.ffn(hidden_states)

        # 将残差连接到当前隐藏状态上
        hidden_states = residual + hidden_states

        # 将最终的隐藏状态作为输出
        outputs = (hidden_states,)

        # 如果需要输出注意力权重，将self-attention和cross-attention的权重也添加到输出中
        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)

        # 如果需要使用缓存，将当前的present_key_value添加到输出中
        if use_cache:
            outputs += (present_key_value,)

        return outputs
    """
    Transformer decoder consisting of `config.layers` layers. Each layer is a [`Kosmos2TextBlock`].

    Args:
        config: Kosmos2TextConfig
    """
    
    def __init__(self, config: Kosmos2TextConfig):
        super().__init__()
        self.config = config  # 保存配置对象
        self.dropout = config.dropout  # 设置 dropout 概率
        self.layerdrop = config.layerdrop  # 设置层级 dropout 概率

        self.embed_scale = math.sqrt(config.embed_dim) if config.scale_embedding else 1.0  # 计算嵌入的缩放因子
        self.embed_tokens = nn.Embedding(config.vocab_size, config.embed_dim, padding_idx=config.pad_token_id)  # 嵌入层，根据配置创建

        self.embed_positions = Kosmos2TextSinusoidalPositionalEmbedding(
            num_positions=config.max_position_embeddings,
            embedding_dim=config.embed_dim,
            padding_idx=config.pad_token_id,
        )  # 创建位置嵌入对象

        self.layers = nn.ModuleList([Kosmos2TextBlock(config) for _ in range(config.layers)])  # 创建多层 Transformer 块
        self.layer_norm = nn.LayerNorm(config.embed_dim, config.layer_norm_eps)  # 创建层归一化对象

        self.gradient_checkpointing = False  # 是否使用梯度检查点，默认为 False

    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # 创建因果注意力遮罩
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )  # 调用函数创建因果遮罩

        if attention_mask is not None:
            # 扩展注意力遮罩的维度
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(
                inputs_embeds.device
            )
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )  # 合并注意力遮罩

        return combined_attention_mask

    def forward_embedding(
        self,
        input_ids,
        inputs_embeds: torch.Tensor = None,
        image_embeds: torch.Tensor = None,
        img_input_mask: torch.Tensor = None,
        past_key_values_length: int = 0,
        position_ids: torch.Tensor = None,
        # 如果未提供 `inputs_embeds` 参数，则使用 `input_ids` 生成对应的嵌入表示
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # 如果提供了 `image_embeds` 参数，则将其融合到 `inputs_embeds` 中相应的位置
        if image_embeds is not None:
            # 使用 `img_input_mask` 将 `image_embeds` 插入到 `inputs_embeds` 中对应位置
            inputs_embeds[img_input_mask.to(dtype=torch.bool)] = image_embeds.to(inputs_embeds.device).view(
                -1, image_embeds.size(-1)
            )

        # 将 `inputs_embeds` 缩放乘以 `self.embed_scale`
        inputs_embeds = inputs_embeds * self.embed_scale

        # 嵌入位置信息
        positions = self.embed_positions(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
            position_ids=position_ids,
        )
        # 将位置嵌入张量移到与 `inputs_embeds` 相同的设备上
        positions = positions.to(inputs_embeds.device)

        # 将位置嵌入张量与输入嵌入张量相加，得到隐藏状态张量
        hidden_states = inputs_embeds + positions

        # 在训练过程中进行 dropout 操作
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        # 返回最终的隐藏状态张量作为前向传播的输出
        return hidden_states
class Kosmos2PreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 配置类，用于当前模型的配置信息
    config_class = Kosmos2Config
    # 是否支持梯度检查点（gradient checkpointing）
    supports_gradient_checkpointing = True
    # 不需要分割的模块列表
    _no_split_modules = ["Kosmos2VisionEncoderLayer", "Kosmos2TextBlock"]

class Kosmos2VisionModel(Kosmos2PreTrainedModel):
    # 使用的配置类
    config_class = Kosmos2VisionConfig
    # 主要输入名称为像素值
    main_input_name = "pixel_values"

    # 从 CLIPVisionModel.__init__ 复制而来，修改了命名空间和变量名
    def __init__(self, config: Kosmos2VisionConfig):
        super().__init__(config)
        # 创建视觉模型对象
        self.model = Kosmos2VisionTransformer(config)
        # 初始化权重并进行最终处理
        self.post_init()

    # 从 CLIPVisionModel.get_input_embeddings 复制而来，修改了命名空间和变量名
    def get_input_embeddings(self) -> nn.Module:
        # 返回嵌入层的 patch 嵌入
        return self.model.embeddings.patch_embedding

    # 添加了文档字符串到模型前向方法的装饰器
    # 替换了返回文档字符串，指定了输出类型和配置类
    @add_start_docstrings_to_model_forward(KOSMOS2_VISION_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=Kosmos2VisionConfig)
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        """
        前向传播方法，接受像素值作为输入，可选输出注意力、隐藏状态和返回字典。

        Returns:
            返回模型的输出，可能是元组或带池化的基础模型输出。
        """
        return self.model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


class Kosmos2TextModel(Kosmos2PreTrainedModel):
    # 使用的配置类
    config_class = Kosmos2TextConfig

    def __init__(self, config: Kosmos2TextConfig):
        super().__init__(config)
        # 创建文本模型对象
        self.model = Kosmos2TextTransformer(config)
        # 初始化权重并进行最终处理
        self.post_init()

    # 获取输入嵌入的方法
    def get_input_embeddings(self) -> nn.Module:
        # 返回嵌入令牌（embed_tokens）
        return self.model.embed_tokens

    # 设置输入嵌入的方法
    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    # 添加了文档字符串到模型前向方法的装饰器
    # 替换了返回文档字符串，指定了输出类型和配置类
    @add_start_docstrings_to_model_forward(KOSMOS2_TEXT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPastAndCrossAttentions, config_class=Kosmos2TextConfig)
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_embeds: Optional[torch.Tensor] = None,
        image_embeds_position_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
        r"""
        将输入参数传递给模型，并返回模型的输出。

        Parameters:
        - input_ids (Optional[torch.Tensor]): 输入的 token IDs 序列，默认为 None。
        - attention_mask (Optional[torch.Tensor]): 注意力遮罩张量，默认为 None。
        - image_embeds (Optional[torch.Tensor]): 图像嵌入张量，默认为 None。
        - image_embeds_position_mask (Optional[torch.Tensor]): 图像嵌入的位置遮罩张量，默认为 None。
        - encoder_hidden_states (Optional[torch.Tensor]): 编码器的隐藏状态张量，默认为 None。
        - encoder_attention_mask (Optional[torch.Tensor]): 编码器的注意力遮罩张量，默认为 None。
        - head_mask (Optional[torch.Tensor]): 头部遮罩张量，默认为 None。
        - cross_attn_head_mask (Optional[torch.Tensor]): 跨注意力头部遮罩张量，默认为 None。
        - past_key_values (Optional[List[torch.FloatTensor]]): 过去的键值对列表，默认为 None。
        - inputs_embeds (Optional[torch.Tensor]): 输入的嵌入张量，默认为 None。
        - position_ids (Optional[torch.Tensor]): 位置 ID 张量，默认为 None。
        - use_cache (Optional[bool]): 是否使用缓存，默认为 None。
        - output_attentions (Optional[bool]): 是否输出注意力权重，默认为 None。
        - output_hidden_states (Optional[bool]): 是否输出隐藏状态，默认为 None。
        - return_dict (Optional[bool]): 是否返回字典格式的输出，默认为 None。

        Returns:
        - Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]: 返回模型的输出，可能是一个元组或特定的输出类对象。

        """
        # 调用模型的 forward 方法，将所有参数传递给模型，并返回模型的输出结果
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            image_embeds=image_embeds,
            image_embeds_position_mask=image_embeds_position_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            head_mask=head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
"""
The text model from KOSMOS-2 with a language modeling head on top (linear layer with weights tied to the input
embeddings).
"""
# 基于KOSMOS-2的文本模型，顶部带有语言建模头部（线性层，权重与输入嵌入绑定）。

# 使用装饰器添加文档字符串到类的开头
@add_start_docstrings(
    KOSMOS2_START_DOCSTRING,  # 使用预定义的起始文档字符串
)
class Kosmos2TextForCausalLM(Kosmos2PreTrainedModel):
    config_class = Kosmos2TextConfig  # 设置配置类为Kosmos2TextConfig
    _tied_weights_keys = ["lm_head.weight"]  # 定义权重绑定的键名列表

    def __init__(self, config: Kosmos2TextConfig):
        super().__init__(config)

        # 初始化模型和语言建模头部线性层
        self.model = Kosmos2TextTransformer(config)  # 使用配置初始化文本转换器模型
        self.lm_head = nn.Linear(in_features=config.embed_dim, out_features=config.vocab_size, bias=False)
        # 初始化线性层，输入维度为config.embed_dim，输出维度为config.vocab_size，无偏置

        # 初始化权重并应用最终处理
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.model.embed_tokens  # 返回模型的输入嵌入层

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value  # 设置模型的输入嵌入层为给定的value

    def get_output_embeddings(self) -> nn.Module:
        return self.lm_head  # 返回语言建模头部的输出嵌入层

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings  # 设置语言建模头部的输出嵌入层为给定的new_embeddings

    # 使用装饰器添加文档字符串到模型的前向方法
    @add_start_docstrings_to_model_forward(KOSMOS2_TEXT_INPUTS_DOCSTRING)
    @replace_return_docstrings(
        output_type=CausalLMOutputWithCrossAttentions,  # 替换输出类型为带交叉注意力的因果语言建模输出
        config_class=Kosmos2TextConfig  # 替换配置类为Kosmos2TextConfig
    )
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_embeds: Optional[torch.Tensor] = None,
        image_embeds_position_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        # 前向传播函数，接受多种输入参数并返回模型输出
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the left-to-right language modeling loss (next word prediction). Indices should be in
            `[-100, 0, ..., config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are
            ignored (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`

        Returns:
            Depending on `return_dict`, either a tuple with `loss` and various outputs or an instance of
            `CausalLMOutputWithCrossAttentions` containing `loss`, `logits`, and other relevant model outputs.

        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if use_cache:
                logger.warning("The `use_cache` argument is changed to `False` since `labels` is provided.")
            use_cache = False  # 如果提供了labels，则强制关闭use_cache，避免使用缓存

        outputs = self.model(
            input_ids=input_ids,  # 输入的token IDs
            attention_mask=attention_mask,  # 注意力遮罩，用于指示哪些token需要被注意
            image_embeds=image_embeds,  # 图像嵌入向量，可选
            image_embeds_position_mask=image_embeds_position_mask,  # 图像嵌入位置掩码，可选
            encoder_hidden_states=encoder_hidden_states,  # 编码器的隐藏状态，用于多层编码器
            encoder_attention_mask=encoder_attention_mask,  # 编码器的注意力遮罩
            head_mask=head_mask,  # 多头注意力机制的头部遮罩
            cross_attn_head_mask=cross_attn_head_mask,  # 跨注意力头部的遮罩
            past_key_values=past_key_values,  # 过去的键值，用于生成
            inputs_embeds=inputs_embeds,  # 输入的嵌入向量，用于替代input_ids
            position_ids=position_ids,  # 位置IDs，指定每个token的位置
            use_cache=use_cache,  # 是否使用缓存，根据labels的存在动态设置
            output_attentions=output_attentions,  # 是否输出注意力权重
            output_hidden_states=output_hidden_states,  # 是否输出隐藏状态
            return_dict=return_dict,  # 是否返回字典形式的输出
        )
        lm_logits = self.lm_head(outputs[0])  # 使用语言模型头部预测的logits

        loss = None
        if labels is not None:
            labels = labels.to(lm_logits.device)  # 将labels移到与logits相同的设备上，以支持模型并行处理
            shift_logits = lm_logits[..., :-1, :].contiguous()  # 将logits向左移动一位，用于计算损失
            shift_labels = labels[..., 1:].contiguous()  # 将labels向左移动一位，与shift_logits对齐
            batch_size, seq_length, vocab_size = shift_logits.shape  # 获取logits的形状信息
            loss_fct = CrossEntropyLoss()  # 交叉熵损失函数
            loss = loss_fct(
                shift_logits.view(batch_size * seq_length, vocab_size), shift_labels.view(batch_size * seq_length)
            )  # 计算损失，将logits和labels展平成二维张量进行计算

        if not return_dict:
            output = (lm_logits,) + outputs[1:]  # 如果不返回字典，输出包含logits和其它模型输出
            return (loss,) + output if loss is not None else output  # 如果有损失，则返回损失和输出；否则只返回输出

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )  # 返回一个包含损失、logits和其它相关模型输出的对象
    ):
        input_shape = input_ids.shape
        # 如果模型作为编码器-解码器模型的解码器使用，会动态创建解码器注意力掩码
        if attention_mask is None:
            # 如果没有提供注意力掩码，创建一个全为1的张量，形状与输入张量相同
            attention_mask = input_ids.new_ones(input_shape)

        position_ids = None

        # 如果使用了过去的键值对，根据输入的ID创建位置ID
        if past_key_values is not None:
            position_ids = create_position_ids_from_input_ids(
                input_ids,
                padding_idx=self.config.pad_token_id,
                past_key_values_length=0,
            )[:, -1:]

            # 截取输入的ID，仅保留最后一个
            input_ids = input_ids[:, -1:]
            # 图像信息已经编码到过去的键/值中，因此不需要额外的图像嵌入
            image_embeds = None
            image_embeds_position_mask = None
        elif image_embeds_position_mask is not None:
            # 将`False`追加到`image_embeds_position_mask`（因为在生成过程中`input_ids`会增长）
            batch_size, seq_len = input_ids.size()
            mask_len = image_embeds_position_mask.size()[-1]
            image_embeds_position_mask = torch.cat(
                (
                    image_embeds_position_mask,
                    torch.zeros(size=(batch_size, seq_len - mask_len), dtype=torch.bool, device=input_ids.device),
                ),
                dim=1,
            )

        return {
            "input_ids": input_ids,
            "image_embeds": image_embeds,
            "image_embeds_position_mask": image_embeds_position_mask,
            "past_key_values": past_key_values,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "use_cache": use_cache,
        }

    @staticmethod
    # 从transformers.models.umt5.modeling_umt5.UMT5ForConditionalGeneration._reorder_cache复制过来的方法
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                # 根据beam_idx重新排序过去的状态
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past
class Kosmos2ImageToTextProjection(nn.Module):
    """The layer that transforms the image model's output to part of the text model's input (namely, image features)"""

    def __init__(self, config: Kosmos2Config):
        super().__init__()
        # 定义一个全连接层，将图像模型输出的隐藏状态映射到文本模型的嵌入维度
        self.dense = nn.Linear(config.vision_config.hidden_size, config.text_config.embed_dim)
        # 定义一个可学习的查询参数矩阵，用于文本注意力机制
        self.latent_query = nn.Parameter(torch.randn(config.latent_query_num, config.text_config.embed_dim))

        # 初始化文本注意力机制，用于处理图像到文本的投影
        self.x_attn = KosmosTextAttention(
            config.text_config,
            config.text_config.embed_dim,
            config.text_config.attention_heads,
            dropout=config.text_config.attention_dropout,
            is_decoder=False,
            add_inner_attn_layernorm=False,
        )

    def forward(self, features):
        # 使用全连接层将图像特征转换为隐藏状态
        hidden_states = self.dense(features)

        # shape = [batch, latent_query_num, h_dim]
        # 准备 latent_query，扩展以匹配隐藏状态的形状
        latent_query = self.latent_query.unsqueeze(0).expand(hidden_states.size(0), -1, -1)
        # 将隐藏状态和 latent_query 连接起来，形成键值状态
        key_value_states = torch.cat([hidden_states, latent_query], dim=1)

        # 应用文本注意力机制，处理图像到文本的转换过程
        hidden_states, attn_weights, _ = self.x_attn(
            hidden_states=latent_query,
            encoder_hidden_states=key_value_states,
            past_key_value=None,
            attention_mask=None,
            output_attentions=None,
        )

        # 返回处理后的隐藏状态和注意力权重
        return hidden_states, attn_weights


@add_start_docstrings(
    """
    KOSMOS-2 Model for generating text and image features. The model consists of a vision encoder and a language model.
    """,
    KOSMOS2_START_DOCSTRING,
)
class Kosmos2Model(Kosmos2PreTrainedModel):
    config_class = Kosmos2Config
    main_input_name = "pixel_values"

    def __init__(self, config: Kosmos2Config):
        super().__init__(config)

        # 初始化文本模型、视觉模型和图像到文本的投影层
        self.text_model = Kosmos2TextModel(config.text_config)
        self.vision_model = Kosmos2VisionModel(config.vision_config)
        self.image_to_text_projection = Kosmos2ImageToTextProjection(config)

        # 初始化权重并应用最终处理
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        # 返回文本模型的嵌入层
        return self.text_model.model.embed_tokens

    def set_input_embeddings(self, value):
        # 设置文本模型的嵌入层
        self.text_model.model.embed_tokens = value

    @add_start_docstrings_to_model_forward(KOSMOS2_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Kosmos2ModelOutput, config_class=_CONFIG_FOR_DOC)
    # 定义模型的前向传播方法，接受多个可选的输入参数
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,  # 图像像素值的张量，可选
        input_ids: Optional[torch.Tensor] = None,  # 输入文本的张量表示，可选
        image_embeds_position_mask: Optional[torch.Tensor] = None,  # 图像嵌入位置掩码的张量，可选
        attention_mask: Optional[torch.Tensor] = None,  # 注意力掩码的张量，可选
        head_mask: Optional[torch.Tensor] = None,  # 头部掩码的张量，可选
        past_key_values: Optional[List[torch.FloatTensor]] = None,  # 过去的键值对列表，可选
        image_embeds: Optional[torch.Tensor] = None,  # 图像嵌入的张量表示，可选
        inputs_embeds: Optional[torch.Tensor] = None,  # 输入嵌入的张量表示，可选
        position_ids: Optional[torch.Tensor] = None,  # 位置ID的张量表示，可选
        use_cache: Optional[bool] = None,  # 是否使用缓存，可选
        output_attentions: Optional[bool] = None,  # 是否输出注意力权重，可选
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态，可选
        return_dict: Optional[bool] = None,  # 是否返回字典格式的结果，可选
"""
KOSMOS-2 Model for generating text and bounding boxes given an image. The model consists of a vision encoder and a
language model.
"""
@add_start_docstrings(
    """
    KOSMOS-2 Model for generating text and bounding boxes given an image. The model consists of a vision encoder and a
    language model.
    """,
    KOSMOS2_START_DOCSTRING,
)
class Kosmos2ForConditionalGeneration(Kosmos2PreTrainedModel):
    # 指定配置类
    config_class = Kosmos2Config
    # 主要输入名称为像素值
    main_input_name = "pixel_values"
    # 绑定权重的键列表
    _tied_weights_keys = ["text_model.lm_head.weight"]

    def __init__(self, config: Kosmos2Config):
        # 调用父类初始化方法
        super().__init__(config)

        # 文本模型部分，使用给定的文本配置初始化
        self.text_model = Kosmos2TextForCausalLM(config.text_config)
        # 视觉模型部分，使用给定的视觉配置初始化
        self.vision_model = Kosmos2VisionModel(config.vision_config)

        # 图像到文本投影模块，使用给定的配置初始化
        self.image_to_text_projection = Kosmos2ImageToTextProjection(config)

        # 初始化权重并应用最终处理
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        # 返回文本模型的嵌入层
        return self.text_model.model.embed_tokens

    def set_input_embeddings(self, value):
        # 设置文本模型的嵌入层
        self.text_model.model.embed_tokens = value

    def get_output_embeddings(self) -> nn.Module:
        # 返回文本模型的输出嵌入层
        return self.text_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        # 设置文本模型的输出嵌入层
        self.text_model.set_output_embeddings(new_embeddings)

    @add_start_docstrings_to_model_forward(KOSMOS2_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Kosmos2ForConditionalGenerationModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        image_embeds_position_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        image_embeds: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        # 前向传播方法，详细参数和返回值请参考模型输入和输出文档字符串
        pass

    def generate(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        image_embeds_position_mask: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        # 生成方法，用于生成文本和边界框，接受多种输入参数
        pass
        # 为了允许 `inputs` 参数（如 `GenerationMixin` 中所需）
        inputs = kwargs.pop("inputs", None)
        # 如果 `pixel_values` 不为 None，并且 `inputs` 也不为 None，则抛出 ValueError
        if pixel_values is not None and inputs is not None:
            raise ValueError(
                f"`inputs`: {inputs} were passed alongside `pixel_values` which is not allowed."
                f"Make sure to either pass `inputs` or pixel_values=..."
            )
        # 如果 `pixel_values` 为 None 且 `inputs` 不为 None，则将 `pixel_values` 设置为 `inputs`
        if pixel_values is None and inputs is not None:
            pixel_values = inputs

        # 如果 `image_embeds` 为 None，则进行以下操作
        if image_embeds is None:
            # 使用 `self.vision_model` 处理 `pixel_values` 得到视觉模型的输出
            vision_model_output = self.vision_model(pixel_values)
            # 将整个 `last_hidden_state` 通过 `post_layernorm` 而不是只使用 `pooled_output`
            image_embeds = self.vision_model.model.post_layernorm(vision_model_output[0])
            # 对特征进行归一化处理
            image_embeds = nn.functional.normalize(image_embeds, dim=-1)
            # 将图像嵌入向量转换为文本嵌入向量
            image_embeds, projection_attentions = self.image_to_text_projection(image_embeds)

        # 使用 `self.text_model` 生成文本输出
        output = self.text_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            image_embeds=image_embeds,
            image_embeds_position_mask=image_embeds_position_mask,
            **kwargs,
        )

        # 返回生成的输出结果
        return output
```