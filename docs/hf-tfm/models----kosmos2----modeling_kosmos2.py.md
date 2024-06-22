# `.\models\kosmos2\modeling_kosmos2.py`

```py
# 设置文件编码为 UTF-8
# 版权声明
# 根据 Apache 许可证 2.0 版本，本文件受版权保护
# 除非符合许可证的规定，否则不得使用此文件
# 您可以在以下网址获取许可证的副本
#     http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于"原样"的基础分发的
# 没有任何明示或暗示的担保或条件，无论是明示的还是暗示的
# 请查看许可证以获取有关权限和限制的详细信息
""" PyTorch KOSMOS-2 model."""

# 导入所需的库
import math
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss

# 导入相关模块和类
from ...activations import ACT2FN
from ...modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPooling,
    CausalLMOutputWithCrossAttentions,
)
from ...modeling_utils import PreTrainedModel
from ...utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from .configuration_kosmos2 import Kosmos2Config, Kosmos2TextConfig, Kosmos2VisionConfig

# 获取日志记录器
logger = logging.get_logger(__name__)

# 用于文档的配置信息
_CONFIG_FOR_DOC = Kosmos2Config

# 预训练模型的存档列表
KOSMOS2_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "microsoft/kosmos-2-patch14-224",
    # 查看所有 KOSMOS-2 模型，请访问 https://huggingface.co/models?filter=kosmos-2
]

# 定义函数用于扩展注意力掩码
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)

# 定义函数用于创建因果掩码
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

# 从 transformers.models.roberta.modeling_roberta.create_position_ids_from_input_ids 复制代码
# 从输入的 input_ids 中创建位置 id，非填充符号替换为它们的位置数字。位置数字从 padding_idx+1 开始。填充符号将被忽略。
def create_position_ids_from_input_ids(input_ids, padding_idx, past_key_values_length=0):
    """
    Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding symbols
    are ignored. This is modified from fairseq's `utils.make_positions`.

    Args:
        x: torch.Tensor x:

    Returns: torch.Tensor
    """
    # 创建一个 mask，标记非填充符号为1，填充符号为0
    mask = input_ids.ne(padding_idx).int()
    # 计算递增的索引，将非填充符号替换为它们的位置数字
    incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask
    # 返回最终的位置 id
    return incremental_indices.long() + padding_idx


# Kosmos2ModelOutput 类的文档字符串
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

# Kosmos2ModelOutput 类的文档字符串，用于视觉输入
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

# Kosmos2ModelOutput 类的文档字符串，用于文本输入
KOSMOS2_TEXT_INPUTS_DOCSTRING = r"""
"""

# Kosmos2ModelOutput 类的文档字符串，用于输入
KOSMOS2_INPUTS_DOCSTRING = r"""
"""


# Kosmos2ModelOutput 类，包含文本模型输出和最后隐藏状态的池化
@dataclass
class Kosmos2ModelOutput(ModelOutput):
    """
    Base class for text model's outputs that also contains a pooling of the last hidden states.
    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
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

    # 定义变量并初始化为None
    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    image_embeds: Optional[torch.FloatTensor] = None
    # 定义一个可选的元组类型变量projection_attentions，初始值为None
    projection_attentions: Optional[Tuple[torch.FloatTensor]] = None
    # 定义一个BaseModelOutputWithPooling类型的变量vision_model_output，初始值为None
    vision_model_output: BaseModelOutputWithPooling = None
    
    # 定义一个方法to_tuple，返回一个元组
    def to_tuple(self) -> Tuple[Any]:
        # 返回一个元组，遍历self中的键值对
        return tuple(
            # 如果键不是"text_model_output"和"vision_model_output"，则返回self中对应键的值
            # 否则返回self中对应键的值的to_tuple方法的结果
            self[k] if k not in ["text_model_output", "vision_model_output"] else getattr(self, k).to_tuple()
            for k in self.keys()
        )
# 定义一个数据类，用于存储`Kosmos2ForConditionalGeneration`的模型输出
@dataclass
class Kosmos2ForConditionalGenerationModelOutput(ModelOutput):
    """
    Model output class for `Kosmos2ForConditionalGeneration`.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            训练过程中的语言模型损失（仅在提供`labels`时返回）。
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            语言建模头部的预测分数（SoftMax前每个词汇标记的得分）。
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            `torch.FloatTensor`的元组（如果模型具有嵌入层，则为嵌入层的输出+每层的输出）的形状为`batch_size, sequence_length, hidden_size`。

            模型在每一层的隐藏状态加上可选的初始嵌入输出。
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            `torch.FloatTensor`的元组（每层一个）的形状为`batch_size, num_heads, sequence_length, sequence_length`。

            在注意力SoftMax之后的注意权重，用于计算自注意力头部的加权平均值。
        image_embeds (`torch.FloatTensor` of shape `(batch_size, latent_query_num, hidden_size)`, *optional*):
            `Kosmos2ImageToTextProjection`的输出的隐藏状态序列。
        projection_attentions (`tuple(torch.FloatTensor)`, *optional*):
            `torch.FloatTensor`的元组（每层一个）的形状为`batch_size, num_heads, sequence_length, sequence_length`。

            由`Kosmos2ImageToTextProjection`给出的注意权重，在注意力SoftMax之后使用，用于计算自注意力头部的加权平均值。
        vision_model_output(`BaseModelOutputWithPooling`, *optional*):
            [`Kosmos2VisionModel`]的输出结果。
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            `tuple(torch.FloatTensor)`的长度为`config.n_layers`，每个元组有2个张量的形状为`(batch_size, num_heads, sequence_length, embed_size_per_head)`，以及可选的如果`config.is_encoder_decoder=True`则有2个额外的形状为`(batch_size,num_heads,encoder_sequence_length,embed_size_per_head)`的张量。

            包含预先计算的隐藏状态（自注意力块中的键和值以及可选的如果`config.is_encoder_decoder=True` 则包含交叉注意力块中的键和值），可以用于加速顺序解码。
    """
    # 定义一个可选的 torch.FloatTensor 类型的 loss 变量，初始值为 None
    loss: Optional[torch.FloatTensor] = None
    # 定义一个 torch.FloatTensor 类型的 logits 变量，初始值为 None
    logits: torch.FloatTensor = None
    # 定义一个可选的元组类型的 past_key_values 变量，初始值为 None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    # 定义一个可选的元组类型的 hidden_states 变量，初始值为 None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # 定义一个可选的元组类型的 attentions 变量，初始值为 None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    # 定义一个可选的 torch.FloatTensor 类型的 image_embeds 变量，初始值为 None
    image_embeds: Optional[torch.FloatTensor] = None
    # 定义一个可选的元组类型的 projection_attentions 变量，初始值为 None
    projection_attentions: Optional[Tuple[torch.FloatTensor]] = None
    # 定义一个 BaseModelOutputWithPooling 类型的 vision_model_output 变量，初始值为 None

    def to_tuple(self) -> Tuple[Any]:
        # 返回一个元组，元组的每个元素是 self[k]，若 k 不在 ["text_model_output", "vision_model_output"] 中，则取 self 中的值，否则调用 getattr(self, k).to_tuple() 方法
        return tuple(
            self[k] if k not in ["text_model_output", "vision_model_output"] else getattr(self, k).to_tuple()
            for k in self.keys()
        )
# 从transformers.models.clip.modeling_clip.CLIPVisionEmbeddings复制并更改为Kosmos2VisionEmbeddings类
class Kosmos2VisionEmbeddings(nn.Module):
    def __init__(self, config: Kosmos2VisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.class_embedding = nn.Parameter(torch.randn(self.embed_dim))  # 初始化类别嵌入向量

        self.patch_embedding = nn.Conv2d(  # 初始化用于处理图像块的卷积层
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=False,
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2  # 计算图像块数量
        self.num_positions = self.num_patches + 1
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)  # 初始化位置嵌入
        self.register_buffer("position_ids", torch.arange(self.num_positions).expand((1, -1)), persistent=False)  # 注册位置ID张量

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        batch_size = pixel_values.shape[0]
        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype))  # 获取图像块的嵌入表示形状为[*，宽度，网格，网格]
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)  # 变形并转置嵌入表示

        class_embeds = self.class_embedding.expand(batch_size, 1, -1)  # 扩展类别嵌入
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)  # 拼接类别嵌入和图像块嵌入
        embeddings = embeddings + self.position_embedding(self.position_ids)  # 添加位置嵌入
        return embeddings


# 从transformers.models.clip.modeling_clip.CLIPAttention复制并更改为Kosmos2VisionAttention类
class Kosmos2VisionAttention(nn.Module):
    """从'Attention Is All You Need'论文中的多头注意力机制"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)  # 用于生成键的线性变���层
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)  # 用于生成值的线性变换层
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)  # 用于生成查询的线性变换层
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)  # 用于生成输出的线性变换层

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()  # 重新整形张量
    # 定义一个名为 forward 的方法，用于模型的前向传播
    def forward(
        self,
        hidden_states: torch.Tensor,  # 输入的隐藏状态张量，代表模型的输入
        attention_mask: Optional[torch.Tensor] = None,  # 可选参数，用于屏蔽不需要关注的部分，如填充部分
        causal_attention_mask: Optional[torch.Tensor] = None,  # 可选参数，用于生成自回归（causal）的注意力掩码
        output_attentions: Optional[bool] = False,  # 可选参数，控制是否输出注意力矩阵
# 从transformers.models.clip.modeling_clip.CLIPMLP复制并将CLIP改为Kosmos2Vision
class Kosmos2VisionMLP(nn.Module):
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


# 从transformers.models.clip.modeling_clip.CLIPEncoderLayer复制并将CLIP改为Kosmos2Vision
class Kosmos2VisionEncoderLayer(nn.Module):
    def __init__(self, config: Kosmos2VisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = Kosmos2VisionAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = Kosmos2VisionMLP(config)
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
                是否返回所有注意力层的注意力张量。有关更多详细信息，请参阅返回的张量下的`attentions`。
        """
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


# 从transformers.models.clip.modeling_clip.CLIPEncoder复制并将CLIP改为Kosmos2Vision
class Kosmos2VisionEncoder(nn.Module):
    """
    由`config.num_hidden_layers`个自注意力层组成的Transformer编码器。每一层都是
    # 初始化 Kosmos2VisionEncoderLayer 类
    def __init__(self, config: Kosmos2VisionConfig):
        # 调用父类的初始化方法
        super().__init__()
        # 将参数config赋值给self.config
        self.config = config
        # 创建一个包含多个 Kosmos2VisionEncoderLayer 实例的模块列表
        self.layers = nn.ModuleList([Kosmos2VisionEncoderLayer(config) for _ in range(config.num_hidden_layers)])
        # 设置梯度检查点标志为False
        self.gradient_checkpointing = False

    # 前向传播方法
    def forward(
        self,
        inputs_embeds,
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
# 类定义，类名为Kosmos2VisionTransformer，和transformers.models.clip.modeling_clip.CLIPVisionTransformer类似，但是没有为forward方法添加文档字符串
class Kosmos2VisionTransformer(nn.Module):
    # 构造方法，接受一个Kosmos2VisionConfig类型的参数config
    def __init__(self, config: Kosmos2VisionConfig):
        # 调用父类构造方法
        super().__init__()
        # 从config中获取hidden_size作为嵌入维度
        embed_dim = config.hidden_size

        # 创建Kosmos2VisionEmbeddings对象
        self.embeddings = Kosmos2VisionEmbeddings(config)
        # 创建LayerNorm层，用于嵌入层的数据处理
        self.pre_layrnorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        # 创建Kosmos2VisionEncoder对象
        self.encoder = Kosmos2VisionEncoder(config)
        # 创建LayerNorm层，用于encoder的输出数据处理
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

    # 前向传播方法，接受pixel_values、output_attentions、output_hidden_states和return_dict等参数，返回Tuple或BaseModelOutputWithPooling类型的结果
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        # 如果未指定output_attentions，则使用self.config.output_attentions的值
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 如果未指定output_hidden_states，则使用self.config.output_hidden_states的值
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 如果未指定return_dict，则使用self.config.use_return_dict的值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 如果pixel_values为None，则抛出数值错误异常
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # 使用embeddings处理pixel_values
        hidden_states = self.embeddings(pixel_values)
        # 对处理后的数据进行LayerNorm处理
        hidden_states = self.pre_layrnorm(hidden_states)

        # 使用encoder处理嵌入数据
        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取encoder的最后一个隐藏状态
        last_hidden_state = encoder_outputs[0]
        # 提取池化输出
        pooled_output = last_hidden_state[:, 0, :]
        # 对池化输出进行LayerNorm处理
        pooled_output = self.post_layernorm(pooled_output)

        # 如果return_dict为False，则返回元组
        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        # 如果return_dict为True，则返回BaseModelOutputWithPooling类型的结果
        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


# 类定义，类名为Kosmos2TextSinusoidalPositionalEmbedding，和transformers.models.m2m_100.modeling_m2m_100.M2M100SinusoidalPositionalEmbedding类似，但允许传入position_ids参数
class Kosmos2TextSinusoidalPositionalEmbedding(nn.Module):
    """This module produces sinusoidal positional embeddings of any length."""

    # 未完成的构造方法，用于生成任意长度的正弦位置嵌入
    # 初始化函数，用于创建一个 Sinusoidal 位置嵌入层的实例
    def __init__(self, num_positions: int, embedding_dim: int, padding_idx: Optional[int] = None):
        # 调用父类的初始化函数
        super().__init__()
        # 设置偏移量为2，用于调整位置编码的起始位置
        self.offset = 2
        # 设置嵌入维度
        self.embedding_dim = embedding_dim
        # 设置填充索引
        self.padding_idx = padding_idx
        # 调用 make_weights 方法创建位置嵌入权重
        self.make_weights(num_positions + self.offset, embedding_dim, padding_idx)

    # 从 transformers.models.m2m_100.modeling_m2m_100.M2M100SinusoidalPositionalEmbedding.make_weights 复制的方法
    # 创建位置嵌入权重
    def make_weights(self, num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None):
        # 调用 get_embedding 方法获取嵌入权重
        emb_weights = self.get_embedding(num_embeddings, embedding_dim, padding_idx)
        # 如果存在 "weights" 属性
        if hasattr(self, "weights"):
            # 在前向传播中将权重放置在正确的数据类型和设备上
            emb_weights = emb_weights.to(dtype=self.weights.dtype, device=self.weights.device)

        # 注册嵌入权重为缓冲区
        self.register_buffer("weights", emb_weights, persistent=False)

    # 从 transformers.models.m2m_100.modeling_m2m_100.M2M100SinusoidalPositionalEmbedding.get_embedding 复制的方法
    # 创建 Sinusoidal 位置嵌入
    @staticmethod
    def get_embedding(num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None):
        """
        构建 Sinusoidal 位置嵌入。

        这与 tensor2tensor 中的实现相匹配，但与 "Attention Is All You Need" 第3.5节中的描述略有不同。
        """
        # 计算嵌入维度的一半
        half_dim = embedding_dim // 2
        # 计算 Sinusoidal 嵌入的频率
        emb = math.log(10000) / (half_dim - 1)
        # 计算 Sinusoidal 嵌入
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        # 创建 Sinusoidal 嵌入矩阵
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        # 将 Sinusoidal 嵌入连接为正弦和余弦序列，并将形状调整为(num_embeddings, embedding_dim)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
        # 如果嵌入维度为奇数，则进行零填充
        if embedding_dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        # 如果存在填充索引，则将填充位置的嵌入设置为0
        if padding_idx is not None:
            emb[padding_idx, :] = 0

        return emb.to(torch.get_default_dtype())

    # 前向传播函数
    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor = None,
        inputs_embeds: torch.Tensor = None,
        past_key_values_length: int = 0,
        position_ids: torch.Tensor = None,
    ):
        # 如果输入的标识不为空
        if input_ids is not None:
            # 获取输入标识的批量大小和序列长度
            bsz, seq_len = input_ids.size()
            # 如果位置标识为空
            if position_ids is None:
                # 从输入标识创建位置标识。任何填充的标识保持填充状态
                position_ids = create_position_ids_from_input_ids(
                    input_ids, self.padding_idx, past_key_values_length
                ).to(input_ids.device)
        else:
            # 获取输入嵌入的批量大小和序列长度
            bsz, seq_len = inputs_embeds.size()[:-1]
            # 如果位置标识为空
            if position_ids is None:
                # 从输入嵌入创建位置标识
                position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds, past_key_values_length)

        # 如果需要，扩展嵌入
        max_pos = self.padding_idx + 1 + seq_len + past_key_values_length
        if max_pos > self.weights.size(0):
            self.make_weights(max_pos + self.offset, self.embedding_dim, self.padding_idx)

        # 选择位置标识对应的嵌入，并重新组织成指定形状的张量，然后进行截断获取梯度
        return self.weights.index_select(0, position_ids.view(-1)).view(bsz, seq_len, self.weights.shape[-1]).detach()

    # 从 transformers.models.m2m_100.modeling_m2m_100.M2M100SinusoidalPositionalEmbedding.create_position_ids_from_inputs_embeds 复制而来
    def create_position_ids_from_inputs_embeds(self, inputs_embeds, past_key_values_length):
        """
        We are provided embeddings directly. We cannot infer which are padded so just generate sequential position ids.

        Args:
            inputs_embeds: torch.Tensor

        Returns: torch.Tensor
        """
        # 获取输入嵌入的形状
        input_shape = inputs_embeds.size()[:-1]
        sequence_length = input_shape[1]

        # 生成顺序位置标识
        position_ids = torch.arange(
            self.padding_idx + 1, sequence_length + self.padding_idx + 1, dtype=torch.long, device=inputs_embeds.device
        )
        # 扩展并返回位置标识
        return position_ids.unsqueeze(0).expand(input_shape).contiguous() + past_key_values_length
# 定义了一个名为 KosmosTextAttention 的 PyTorch 模块，实现了多头注意力机制，参考了 'Attention Is All You Need' 论文
class KosmosTextAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    # 初始化函数，类似于 transformers.models.bart.modeling_bart.BartAttention.__init__，但多了一个内部注意力层标准化的参数 `inner_attn_ln`
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
        # 调用父类的初始化方法
        super().__init__()
        # 设置模块的参数
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads

        # 检查 embed_dim 必须能被 num_heads 整除
        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        # 计算缩放系数
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder

        # 定义线性映射层
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        # 定义内部注意力层标准化层
        self.inner_attn_ln = None
        if add_inner_attn_layernorm:
            self.inner_attn_ln = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

    # 将输入张量投影成多头注意力所需的形状
    def _shape(self, projection: torch.Tensor) -> torch.Tensor:
        new_projection_shape = projection.size()[:-1] + (self.num_heads, self.head_dim)
        # 将头部维度移到第二个位置 (B, T, H * D) -> (B, T, H, D) -> (B, H, T, D)
        new_projection = projection.view(new_projection_shape).permute(0, 2, 1, 3)
        return new_projection

    # 前向传播函数
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,



        # 定义了一个名为 Kosmos2TextFFN 的 PyTorch 模块，实现了 Transformer 中的前馈神经网络
class Kosmos2TextFFN(nn.Module):
    def __init__(self, config: Kosmos2TextConfig):
        # 调用父类的初始化方法
        super().__init__()

        # 设置模块的参数
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout

        # 定义两个线性映射层和一个标准化层
        self.fc1 = nn.Linear(config.embed_dim, config.ffn_dim)
        self.fc2 = nn.Linear(config.ffn_dim, config.embed_dim)

        self.ffn_layernorm = nn.LayerNorm(config.ffn_dim, eps=config.layer_norm_eps)
    # 实现神经网络的前向传播函数
    def forward(self, hidden_states):
        # 使用激活函数处理全连接层的输出
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        # 对处理后的隐藏状态进行dropout操作
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        # 使用layernorm对隐藏状态进行正则化处理
        hidden_states = self.ffn_layernorm(hidden_states)
        # 使用全连接层处理隐藏状态
        hidden_states = self.fc2(hidden_states)
        # 对全连接层的输出进行dropout操作
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    
        # 返回处理后的隐藏状态
        return hidden_states
class Kosmos2TextBlock(nn.Module):
    # 定义 Kosmos2TextBlock 类，继承自 nn.Module
    def __init__(self, config: Kosmos2TextConfig):
        # 初始化函数，接收一个 Kosmos2TextConfig 类型的 config 参数
        super().__init__()
        # 调用父类的初始化函数

        # 从 config 中获取 embed_dim 并赋值给 self.embed_dim
        self.embed_dim = config.embed_dim

        # 创建 self-attention 层，使用 KosmosTextAttention 类
        self.self_attn = KosmosTextAttention(
            config,
            embed_dim=self.embed_dim,
            num_heads=config.attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
            add_inner_attn_layernorm=True,
        )
        # 设置 dropout 等参数
        self.dropout = config.dropout
        # 创建 LayerNorm 层，对 self-attention 进行归一化
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

        # 如果需要添加 cross-attention
        if config.add_cross_attention:
            # 创建 cross-attention 层，同样使用 KosmosTextAttention 类
            self.encoder_attn = KosmosTextAttention(
                config,
                embed_dim=self.embed_dim,
                num_heads=config.attention_heads,
                dropout=config.attention_dropout,
                is_decoder=True,
                add_inner_attn_layernorm=False,
            )
            # 创建 LayerNorm 层，对 encoder-attention 进行归一化
            self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

        # 创建线性层，FFN 层
        self.ffn = Kosmos2TextFFN(config)
        # 创建 LayerNorm 层，对最终输出进行归一化
        self.final_layer_norm = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

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
    # 定义函数的输入参数及返回值类型
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        # 保存输入的隐藏状态
        residual = hidden_states

        # Self Attention
        # 如果有过去的键/值缓存，将其提取出来，否则为空
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None

        # 对隐藏状态进行 LayerNormalization
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # 进行自注意力操作，获取输出隐藏状态、注意力权重和当前的键/值缓存
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=self_attn_past_key_value,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        
        # 对输出隐藏状态进行 Dropout
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        
        # 将原始输入的隐藏状态与处理后的隐藏状态相加，得到最终的隐藏状态
        hidden_states = residual + hidden_states

        # Cross-Attention Block
        # 初始化交叉注意力的相关变量
        cross_attn_present_key_value = None
        cross_attn_weights = None
        
        # 如果有编码器的隐藏状态
        if encoder_hidden_states is not None:
            # 如果模型中没有编码器的注意力层，抛出数值错误
            if not hasattr(self, "encoder_attn"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers"
                    " by setting `config.add_cross_attention=True`"
                )
            
            # 保存输入的隐藏状态
            residual = hidden_states
            
            # 对隐藏状态进行 LayerNormalization
            hidden_states = self.encoder_attn_layer_norm(hidden_states)

            # 如果有过去的交叉注意力的键/值缓存，将其提取出来，否则为空
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None

            # 进行交叉注意力操作，获取输出隐藏状态、注意力权重和当前的键/值缓存
            hidden_states, cross_attn_weights, cross_attn_present_key_value = self.encoder_attn(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=cross_attn_past_key_value,
                output_attentions=output_attentions,
            )
            
            # 对输出隐藏状态进行 Dropout
            hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
            
            # 将原始输入的隐藏状态与处理后的隐藏状态相加，得到最终的隐藏状态
            hidden_states = residual + hidden_states
            
            # 更新当前的键/值缓存
            present_key_value = present_key_value + cross_attn_present_key_value

        # Fully Connected
        # 保存输入的隐藏状态
        residual = hidden_states
        
        # 对隐藏状态进行 LayerNormalization
        hidden_states = self.final_layer_norm(hidden_states)

        # 进行 FFN（Feed-Forward Network）操作
        hidden_states = self.ffn(hidden_states)
        
        # 将原始输入的隐藏状态与处理后的隐藏状态相加，得到最终的隐藏状态
        hidden_states = residual + hidden_states

        # 将最终的隐藏状态保存到输出中
        outputs = (hidden_states,)

        # 如果需要输出注意力权重信息，则将其添加到输出中
        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)

        # 如果需要使用缓存，则将当前的键/值缓存添加到输出中
        if use_cache:
            outputs += (present_key_value,)

        # 返回最终的输出
        return outputs
class Kosmos2TextTransformer(nn.Module):
    """
    Transformer decoder consisting of `config.layers` layers. Each layer is a [`Kosmos2TextBlock`].

    Args:
        config: Kosmos2TextConfig
    """

    def __init__(self, config: Kosmos2TextConfig):
        super().__init__()
        # 初始化模型配置
        self.config = config
        # 设置丢弃率
        self.dropout = config.dropout
        # 设置层丢弃率
        self.layerdrop = config.layerdrop

        # 计算嵌入尺度
        self.embed_scale = math.sqrt(config.embed_dim) if config.scale_embedding else 1.0
        # 初始化嵌入层
        self.embed_tokens = nn.Embedding(config.vocab_size, config.embed_dim, padding_idx=config.pad_token_id)

        # 初始化位置编码
        self.embed_positions = Kosmos2TextSinusoidalPositionalEmbedding(
            num_positions=config.max_position_embeddings,
            embedding_dim=config.embed_dim,
            padding_idx=config.pad_token_id,
        )

        # 初始化多层 Transformer 块
        self.layers = nn.ModuleList([Kosmos2TextBlock(config) for _ in range(config.layers)])
        # 初始化层归一化
        self.layer_norm = nn.LayerNorm(config.embed_dim, config.layer_norm_eps)

        # 是否启用梯度检查点
        self.gradient_checkpointing = False

    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # 创建自回归掩码
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(
                inputs_embeds.device
            )
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask

    def forward_embedding(
        self,
        input_ids,
        inputs_embeds: torch.Tensor = None,
        image_embeds: torch.Tensor = None,
        img_input_mask: torch.Tensor = None,
        past_key_values_length: int = 0,
        position_ids: torch.Tensor = None,
```  
        # 如果未提供输入嵌入，则使用模型的嵌入函数根据输入的 ID 生成输入嵌入
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # 如果提供了图像嵌入，则将其应用到相应位置
        if image_embeds is not None:
            # 将图像嵌入应用到输入的嵌入中，并根据图像嵌入的位置掩码进行调整
            inputs_embeds[img_input_mask.to(dtype=torch.bool)] = image_embeds.to(inputs_embeds.device).view(
                -1, image_embeds.size(-1)
            )

        # 将输入嵌入乘以嵌入尺度
        inputs_embeds = inputs_embeds * self.embed_scale

        # 嵌入位置信息
        positions = self.embed_positions(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
            position_ids=position_ids,
        )
        # 将位置嵌入移到与输入嵌入相同的设备上
        positions = positions.to(inputs_embeds.device)

        # 将输入嵌入与位置嵌入相加以获取最终的隐藏状态
        hidden_states = inputs_embeds + positions

        # 对隐藏状态应用丢弃，以防止过拟合
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        # 返回最终的隐藏状态
        return hidden_states

    # 定义前向传播函数
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
```py  
class Kosmos2PreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 设置默认配置类
    config_class = Kosmos2Config
    # 支持梯度检查点
    supports_gradient_checkpointing = True
    # 不进行模块拆分的名称列表
    _no_split_modules = ["Kosmos2VisionEncoderLayer", "Kosmos2TextBlock"]

class Kosmos2VisionModel(Kosmos2PreTrainedModel):
    # 设置视觉模型的配置类
    config_class = Kosmos2VisionConfig
    # 主输入名称为像素值
    main_input_name = "pixel_values"

    # 从CLIPVisionModel.__init__中复制的代码，初始化Kosmos2VisionModel
    def __init__(self, config: Kosmos2VisionConfig):
        # 调用父类的初始化方法
        super().__init__(config)
        # 创建Kosmos2VisionTransformer模型
        self.model = Kosmos2VisionTransformer(config)
        # 初始化权重并应用最终处理
        self.post_init()

    # 从CLIPVisionModel.get_input_embeddings中复制的代码，返回输入嵌入
    def get_input_embeddings(self) -> nn.Module:
        return self.model.embeddings.patch_embedding

    # 添加到模型前向传播的文档字符串
    @add_start_docstrings_to_model_forward(KOSMOS2_VISION_INPUTS_DOCSTRING)
    # 替换返回值的文档字符串
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=Kosmos2VisionConfig)
    # 模型的前向传播方法
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:

        """
        # 调用Kosmos2VisionTransformer模型的前向传播方法
        return self.model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


class Kosmos2TextModel(Kosmos2PreTrainedModel):
    # 设置文本模型的配置类
    config_class = Kosmos2TextConfig

    # 初始化Kosmos2TextModel
    def __init__(self, config: Kosmos2TextConfig):
        # 调用父类的初始化方法
        super().__init__(config)
        # 创建Kosmos2TextTransformer模型
        self.model = Kosmos2TextTransformer(config)
        # 初始化权重并应用最终处理
        self.post_init()

    # 返回输入嵌入
    def get_input_embeddings(self) -> nn.Module:
        return self.model.embed_tokens

    # 设置输入嵌入
    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    # 添加到模型前向传播的文档字符串
    @add_start_docstrings_to_model_forward(KOSMOS2_TEXT_INPUTS_DOCSTRING)
    # 替换返回值的文档字符串
    @replace_return_docstrings(output_type=BaseModelOutputWithPastAndCrossAttentions, config_class=Kosmos2TextConfig)
    # 此方法用于模型的前向传播
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,  # 输入的token ID序列，默认为None
        attention_mask: Optional[torch.Tensor] = None,  # 注意力掩码，默认为None
        image_embeds: Optional[torch.Tensor] = None,  # 图像嵌入，默认为None
        image_embeds_position_mask: Optional[torch.Tensor] = None,  # 图像嵌入位置掩码，默认为None
        encoder_hidden_states: Optional[torch.Tensor] = None,  # 编码器隐藏状态，默认为None
        encoder_attention_mask: Optional[torch.Tensor] = None,  # 编码器注意力掩码，默认为None
        head_mask: Optional[torch.Tensor] = None,  # 头部掩码，默认为None
        cross_attn_head_mask: Optional[torch.Tensor] = None,  # 跨注意力头部掩码，默认为None
        past_key_values: Optional[List[torch.FloatTensor]] = None,  # 过去的键值对，默认为None
        inputs_embeds: Optional[torch.Tensor] = None,  # 输入的嵌入，默认为None
        position_ids: Optional[torch.Tensor] = None,  # 位置ID，默认为None
        use_cache: Optional[bool] = None,  # 是否使用缓存，默认为None
        output_attentions: Optional[bool] = None,  # 是否输出注意力，默认为None
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态，默认为None
        return_dict: Optional[bool] = None,  # 是否返回字典，默认为None
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
        r"""
        Returns:
        此方法返回模型的前向传播结果
        """
        # 调用模型的forward方法，传入参数并返回结果
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
# 使用装饰器为该类添加文档字符串，在 KOSMOS-2 模型的基础上添加了一个语言建模头部（线性层，其权重与输入嵌入绑定）
@add_start_docstrings(
    """
    The text model from KOSMOS-2 with a language modeling head on top (linear layer with weights tied to the input
    embeddings).
    """,
    KOSMOS2_START_DOCSTRING,
)
# 定义了 Kosmos2TextForCausalLM 类，继承自 Kosmos2PreTrainedModel 类
class Kosmos2TextForCausalLM(Kosmos2PreTrainedModel):
    # 类变量，指向 Kosmos2TextConfig 类
    config_class = Kosmos2TextConfig
    # 权重绑定的键列表
    _tied_weights_keys = ["lm_head.weight"]

    # 初始化方法，接受一个 Kosmos2TextConfig 对象作为参数
    def __init__(self, config: Kosmos2TextConfig):
        # 调用父类的初始化方法
        super().__init__(config)

        # 创建 Kosmos2TextTransformer 模型对象
        self.model = Kosmos2TextTransformer(config)
        # 创建一个线性层，输入特征维度为 config.embed_dim，输出特征维度为 config.vocab_size，无偏置
        self.lm_head = nn.Linear(in_features=config.embed_dim, out_features=config.vocab_size, bias=False)

        # 初始化权重并应用最终处理
        self.post_init()

    # 获取输入嵌入的方法
    def get_input_embeddings(self) -> nn.Module:
        return self.model.embed_tokens

    # 设置输入嵌入的方法
    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    # 获取输出嵌入的方法
    def get_output_embeddings(self) -> nn.Module:
        return self.lm_head

    # 设置输出嵌入的方法
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    # 前向传播方法，接受多个输入参数，返回一个输出结果
    @add_start_docstrings_to_model_forward(KOSMOS2_TEXT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithCrossAttentions, config_class=Kosmos2TextConfig)
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
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional`):
            Labels for computing the left-to-right language modeling loss (next word prediction). Indices should be in
            `[-100, 0, ..., config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are
            ignored (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`

        Returns:

        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if use_cache:
                logger.warning("The `use_cache` argument is changed to `False` since `labels` is provided.")
            use_cache = False

        outputs = self.model(
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
        lm_logits = self.lm_head(outputs[0])

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(lm_logits.device)
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            batch_size, seq_length, vocab_size = shift_logits.shape
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(batch_size * seq_length, vocab_size), shift_labels.view(batch_size * seq_length)
            )

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

# 返回 CausalLMOutputWithCrossAttentions 类型的对象，包括 loss、logits等信息
        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        image_embeds=None,
        image_embeds_position_mask=None,
        past_key_values=None,
        attention_mask=None,
        use_cache=None,
        **model_kwargs,
        ):
            # 获取输入张量的形状
            input_shape = input_ids.shape
            # 如果模型作为编码器在编码器-解码器模型中使用，则动态创建解码器注意力掩码
            if attention_mask is None:
                # 如果注意力掩码为None，则创建全为1的张量作为注意力掩码
                attention_mask = input_ids.new_ones(input_shape)

            position_ids = None

            # 如果使用了过去的键值，则截取输入的input_ids
            if past_key_values is not None:
                position_ids = create_position_ids_from_input_ids(
                    input_ids,
                    padding_idx=self.config.pad_token_id,
                    past_key_values_length=0,
                )[:, -1:]

                input_ids = input_ids[:, -1:]
                # 图像信息已编码到过去的键/值中
                image_embeds = None
                image_embeds_position_mask = None
            elif image_embeds_position_mask is not None:
                # 将 `False` 添加到 `image_embeds_position_mask` 中（因为 `input_ids` 在生成过程中会增长）
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
        # 从transformers.models.umt5.modeling_umt5.UMT5ForConditionalGeneration._reorder_cache中复制的方法
        def _reorder_cache(past_key_values, beam_idx):
            # 重新排列过去的键/值对
            reordered_past = ()
            for layer_past in past_key_values:
                reordered_past += (
                    tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
                )
            return reordered_past
class Kosmos2ImageToTextProjection(nn.Module):
    """The layer that transforms the image model's output to part of the text model's input (namely, image features)"""

    def __init__(self, config: Kosmos2Config):
        # 初始化函数，设置图像特征向文本模型输入转换的线性层
        super().__init__()
        # 线性层，将图像特征的隐藏大小转换为文本嵌入的大小
        self.dense = nn.Linear(config.vision_config.hidden_size, config.text_config.embed_dim)
        # 存储用于生成潜在查询的参数
        self.latent_query = nn.Parameter(torch.randn(config.latent_query_num, config.text_config.embed_dim))

        # 初始化文本注意力机制
        self.x_attn = KosmosTextAttention(
            config.text_config,
            config.text_config.embed_dim,
            config.text_config.attention_heads,
            dropout=config.text_config.attention_dropout,
            is_decoder=False,
            add_inner_attn_layernorm=False,
        )

    def forward(self, features):
        # 线性转换图像特征向量
        hidden_states = self.dense(features)

        # 创建潜在查询，将其与隐藏状态拼接作为键值对的状态
        latent_query = self.latent_query.unsqueeze(0).expand(hidden_states.size(0), -1, -1)
        key_value_states = torch.cat([hidden_states, latent_query], dim=1)

        # 进行文本注意力机制
        hidden_states, attn_weights, _ = self.x_attn(
            hidden_states=latent_query,
            encoder_hidden_states=key_value_states,
            past_key_value=None,
            attention_mask=None,
            output_attentions=None,
        )

        return hidden_states, attn_weights


@add_start_docstrings(
    """
    KOSMOS-2 Model for generating text and image features. The model consists of a vision encoder and a language model.
    """,
    KOSMOS2_START_DOCSTRING,
)
class Kosmos2Model(Kosmos2PreTrainedModel):
    # 定义 Kosmos2Model 类，生成文本和图像特征的模型
    config_class = Kosmos2Config
    main_input_name = "pixel_values"

    def __init__(self, config: Kosmos2Config):
        # 初始化函数，构建 Kosmos2Model
        super().__init__(config)

        # 初始化文本模型、视觉模型和图像到文本投影模型
        self.text_model = Kosmos2TextModel(config.text_config)
        self.vision_model = Kosmos2VisionModel(config.vision_config)
        self.image_to_text_projection = Kosmos2ImageToTextProjection(config)

        # 初始化权重并应用最终处理
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        # 返回文本模型的嵌入层
        return self.text_model.model.embed_tokens

    def set_input_embeddings(self, value):
        # 设置文本模型的��入层
        self.text_model.model.embed_tokens = value

    @add_start_docstrings_to_model_forward(KOSMOS2_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Kosmos2ModelOutput, config_class=_CONFIG_FOR_DOC)
    # 前向传播方法，用于模型推理或训练中的前向计算
    def forward(
        # 输入像素值的张量，可选
        pixel_values: Optional[torch.Tensor] = None,
        # 输入 token ID 的张量，可选
        input_ids: Optional[torch.Tensor] = None,
        # 图像嵌入位置掩码的张量，可选
        image_embeds_position_mask: Optional[torch.Tensor] = None,
        # 注意力掩码的张量，可选
        attention_mask: Optional[torch.Tensor] = None,
        # 头部掩码的张量，可选
        head_mask: Optional[torch.Tensor] = None,
        # 用于存储过去的键值的列表，可选
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        # 图像嵌入的张量，可选
        image_embeds: Optional[torch.Tensor] = None,
        # 输入嵌入的张量，可选
        inputs_embeds: Optional[torch.Tensor] = None,
        # 位置 ID 的张量，可选
        position_ids: Optional[torch.Tensor] = None,
        # 是否使用缓存，可选
        use_cache: Optional[bool] = None,
        # 是否输出注意力权重，可选
        output_attentions: Optional[bool] = None,
        # 是否输出隐藏状态，可选
        output_hidden_states: Optional[bool] = None,
        # 是否返回字典形式的输出，可选
        return_dict: Optional[bool] = None,
# KOSMOS-2 模型用于生成给定图像的文本和边界框。该模型包含视觉编码器和语言模型。
@add_start_docstrings(
    """
    KOSMOS-2 Model for generating text and bounding boxes given an image. The model consists of a vision encoder and a
    language model.
    """,
    KOSMOS2_START_DOCSTRING,
)
class Kosmos2ForConditionalGeneration(Kosmos2PreTrainedModel):
    
    # 设置配置类
    config_class = Kosmos2Config
    # 设置主要的输入名称
    main_input_name = "pixel_values"
    # 被绑定权重的键
    _tied_weights_keys = ["text_model.lm_head.weight"]

    def __init__(self, config: Kosmos2Config):
        super().__init__(config)

        # 初始化文本模型和视觉模型
        self.text_model = Kosmos2TextForCausalLM(config.text_config)
        self.vision_model = Kosmos2VisionModel(config.vision_config)

        # 图像到文本的映射
        self.image_to_text_projection = Kosmos2ImageToTextProjection(config)

        # 初始化权重并进行最终处理
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        # 获取输入嵌入层对象
        return self.text_model.model.embed_tokens

    def set_input_embeddings(self, value):
        # 设置输入嵌入层对象
        self.text_model.model.embed_tokens = value

    def get_output_embeddings(self) -> nn.Module:
        # 获取输出嵌入层对象
        return self.text_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        # 设置输出嵌入层对象
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
        """
        前向传播函数
        """
        ...

    def generate(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        image_embeds_position_mask: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """
        生成函数
        """
        ...
        # 允许使用 `inputs` 参数（如 `GenerationMixin` 中所用）
        inputs = kwargs.pop("inputs", None)
        # 如果 `pixel_values` 和 `inputs` 都不为空，则抛出 ValueError
        if pixel_values is not None and inputs is not None:
            raise ValueError(
                f"`inputs`: {inputs} were passed alongside `pixel_values` which is not allowed."
                f"Make sure to either pass `inputs` or pixel_values=..."
            )
        # 如果 `pixel_values` 为空且 `inputs` 不为空，则将 `pixel_values` 赋值为 `inputs`
        if pixel_values is None and inputs is not None:
            pixel_values = inputs

        # 如果 `image_embeds` 为空
        if image_embeds is None:
            # 通过视觉模型获取视觉输出
            vision_model_output = self.vision_model(pixel_values)
            # 通过 `post_layernorm` 获取整个 `last_hidden_state` 而不只是 `pooled_output`
            image_embeds = self.vision_model.model.post_layernorm(vision_model_output[0])
            # 对特征进行归一化
            image_embeds = nn.functional.normalize(image_embeds, dim=-1)
            # 将图像嵌入向文本的投影
            image_embeds, projection_attentions = self.image_to_text_projection(image_embeds)

        # 生成文本模型的输出
        output = self.text_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            image_embeds=image_embeds,
            image_embeds_position_mask=image_embeds_position_mask,
            **kwargs,
        )

        # 返回输出结果
        return output
```