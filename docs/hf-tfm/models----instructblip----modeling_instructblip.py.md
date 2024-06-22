# `.\models\instructblip\modeling_instructblip.py`

```py
# 设置文件编码为 utf-8
# 版权声明，版权归 Salesforce 作者和 HuggingFace 团队所有
# 根据 Apache 许可证 2.0 版本授权使用本文件
# 只有在遵守许可证的情况下才能使用本文件
# 可以在以下网址获取许可证的副本：http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于“原样”分发的，没有任何明示或暗示的担保或条件
# 请查看许可证以获取有关特定语言的权限和限制
""" PyTorch InstructBLIP model."""

# 导入所需的库
import math
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss

# 导入自定义的模块
from ...activations import ACT2FN
from ...modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPooling,
    BaseModelOutputWithPoolingAndCrossAttentions,
)
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from ..auto import AutoModelForCausalLM, AutoModelForSeq2SeqLM
from .configuration_instructblip import InstructBlipConfig, InstructBlipQFormerConfig, InstructBlipVisionConfig

# 获取日志记录器
logger = logging.get_logger(__name__)

# 用于文档的检查点
_CHECKPOINT_FOR_DOC = "Salesforce/instructblip-flan-t5-xl"

# InstructBLIP 预训练模型存档列表
INSTRUCTBLIP_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "Salesforce/instructblip-flan-t5-xl",
    # 查看所有 InstructBLIP 模型：https://huggingface.co/models?filter=instructblip
]

# 数据类，定义了 InstructBlipForConditionalGenerationModelOutput 的输出
@dataclass
# 从 transformers.models.blip_2.modeling_blip_2.Blip2ForConditionalGenerationModelOutput 复制并将 Blip2 改为 InstructBlip
class InstructBlipForConditionalGenerationModelOutput(ModelOutput):
    """
    Class defining the outputs of [`InstructBlipForConditionalGeneration`].

    Args:
        loss (`torch.FloatTensor`, *optional*, returned when `labels` is provided, `torch.FloatTensor` of shape `(1,)`):
            Language modeling loss from the language model.
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head of the language model.
        vision_outputs (`BaseModelOutputWithPooling`):
            Outputs of the vision encoder.
        qformer_outputs (`BaseModelOutputWithPoolingAndCrossAttentions`):
            Outputs of the Q-Former (Querying Transformer).
        language_model_outputs (`CausalLMOutputWithPast` or `Seq2SeqLMOutput`):
            Outputs of the language model.
    """

    loss: Optional[Tuple[torch.FloatTensor]] = None
    logits: Optional[Tuple[torch.FloatTensor]] = None
    # 定义三个可选的 torch.FloatTensor 类型变量，分别用于存储视觉输出、qformer 输出和语言模型输出
    vision_outputs: Optional[torch.FloatTensor] = None
    qformer_outputs: Optional[Tuple[torch.FloatTensor]] = None
    language_model_outputs: Optional[Tuple[torch.FloatTensor]] = None
    
    # 定义一个方法，将当前对象转换为元组
    def to_tuple(self) -> Tuple[Any]:
        # 返回一个元组，其中包含当前对象的所有属性值
        return tuple(
            # 如果属性不是视觉输出、qformer 输出或语言模型输出，则直接取其值
            self[k]
            if k not in ["vision_outputs", "qformer_outputs", "language_model_outputs"]
            # 如果属性是视觉输出、qformer 输出或语言模型输出，则调用其 to_tuple() 方法
            else getattr(self, k).to_tuple()
            for k in self.keys()
        )
# 从transformers.models.blip.modeling_blip.InstructBlipVisionEmbeddings类复制而来，用于处理视觉嵌入
class InstructBlipVisionEmbeddings(nn.Module):
    def __init__(self, config: InstructBlipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.class_embedding = nn.Parameter(torch.randn(1, 1, self.embed_dim))  # 定义类别嵌入参数

        self.patch_embedding = nn.Conv2d(
            in_channels=3, out_channels=self.embed_dim, kernel_size=self.patch_size, stride=self.patch_size
        )  # 定义图像块嵌入卷积层

        self.num_patches = (self.image_size // self.patch_size) ** 2  # 计算图像块数量
        self.num_positions = self.num_patches + 1  # 计算位置嵌入数量

        self.position_embedding = nn.Parameter(torch.randn(1, self.num_positions, self.embed_dim))  # 定义位置嵌入参数

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        batch_size = pixel_values.shape[0]
        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype))  # 获取图像块嵌入
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)  # 对图像块嵌入进行展平和转置

        class_embeds = self.class_embedding.expand(batch_size, 1, -1).to(target_dtype)  # 扩展类别嵌入
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)  # 拼接类别嵌入和图像块嵌入
        embeddings = embeddings + self.position_embedding[:, : embeddings.size(1), :].to(target_dtype)  # 添加位置嵌入
        return embeddings


# 从transformers.models.blip_2.modeling_blip_2.InstructBlipAttention类复制而来，用于处理注意力机制
class InstructBlipAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

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
        self.dropout = nn.Dropout(config.attention_dropout)

        # small tweak here compared to CLIP, no bias here
        self.qkv = nn.Linear(self.embed_dim, 3 * self.embed_dim, bias=False)  # 定义查询、键、值的线性层

        if config.qkv_bias:
            q_bias = nn.Parameter(torch.zeros(self.embed_dim))
            v_bias = nn.Parameter(torch.zeros(self.embed_dim))
        else:
            q_bias = None
            v_bias = None

        if q_bias is not None:
            qkv_bias = torch.cat((q_bias, torch.zeros_like(v_bias, requires_grad=False), v_bias))
            self.qkv.bias = nn.Parameter(qkv_bias)  # 设置查询、键、值的偏置

        self.projection = nn.Linear(self.embed_dim, self.embed_dim)  # 定义投影线性层
    # 定义一个私有方法，用于对输入的张量进行形状变换，将其转换为适合多头注意力计算的形状
    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    # 前向传播函数，接收隐藏状态张量和其他可选参数，返回输出张量和注意力概率
    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        # 获取隐藏状态张量的形状信息
        bsz, tgt_len, embed_dim = hidden_states.size()

        # 通过全连接层将隐藏状态张量映射为混合的查询、键、值张量
        mixed_qkv = self.qkv(hidden_states)

        # 重塑混合的查询、键、值张量的形状以便进行多头注意力计算
        mixed_qkv = mixed_qkv.reshape(bsz, tgt_len, 3, self.num_heads, embed_dim // self.num_heads).permute(
            2, 0, 3, 1, 4
        )
        query_states, key_states, value_states = mixed_qkv[0], mixed_qkv[1], mixed_qkv[2]

        # 计算原始注意力分数，即查询和键的点积
        attention_scores = torch.matmul(query_states, key_states.transpose(-1, -2))

        # 缩放注意力分数
        attention_scores = attention_scores * self.scale

        # 将注意力分数归一化为概率
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # 对注意力概率进行dropout操作
        attention_probs = self.dropout(attention_probs)

        # 如果需要，对注意力概率进行头部掩码操作
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        # 计算上下文张量，即注意力概率与值的加权和
        context_layer = torch.matmul(attention_probs, value_states).permute(0, 2, 1, 3)

        # 重塑上下文张量的形状
        new_context_layer_shape = context_layer.size()[:-2] + (self.embed_dim,)
        context_layer = context_layer.reshape(new_context_layer_shape)

        # 通过投影层得到最终输出
        output = self.projection(context_layer)

        # 根据是否需要输出注意力概率，返回不同的结果
        outputs = (output, attention_probs) if output_attentions else (output, None)

        return outputs
# 从transformers.models.blip.modeling_blip.BlipMLP复制过来的类
class InstructBlipMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.activation_fn = ACT2FN[config.hidden_act]  # 从配置中获取激活函数
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)  # 创建全连接层1
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)  # 创建全连接层2

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)  # 全连接层1的前向传播
        hidden_states = self.activation_fn(hidden_states)  # 激活函数的应用
        hidden_states = self.fc2(hidden_states)  # 全连接层2的前向传播
        return hidden_states  # 返回隐藏状态


# 从transformers.models.blip.modeling_blip.BlipEncoderLayer复制过来的类，将Blip->InstructBlip
class InstructBlipEncoderLayer(nn.Module):
    def __init__(self, config: InstructBlipConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = InstructBlipAttention(config)  # 创建自注意力层
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)  # 创建LayerNorm层1
        self.mlp = InstructBlipMLP(config)  # 创建MLP层
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)  # 创建LayerNorm层2

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
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
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)  # LayerNorm层1的应用
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            head_mask=attention_mask,
            output_attentions=output_attentions,
        )  # 自注意力层的前向传播
        hidden_states = hidden_states + residual
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)  # LayerNorm层2的应用
        hidden_states = self.mlp(hidden_states)  # MLP层的前向传播

        hidden_states = hidden_states + residual

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)  # 如果需要输出注意力权重，则添加到输出中

        return outputs  # 返回输出结果


class InstructBlipPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = InstructBlipConfig
    base_model_prefix = "blip"
    supports_gradient_checkpointing = True
    # 不需要拆分的模块列表
    _no_split_modules = [
        "InstructBlipQFormerEmbeddings",
        "InstructBlipAttention",
        "InstructBlipQFormerMultiHeadAttention",
        "InstructBlipQFormerSelfOutput",
    ]
    # 需要保持为32位浮点数的模块列表

    _keep_in_fp32_modules = []

    # 从transformers.models.blip_2.modeling_blip_2.Blip2PreTrainedModel._init_weights复制而来，用InstructBlip替换Blip2
    def _init_weights(self, module):
        """初始化权重"""
        factor = self.config.initializer_range
        # 如果是卷积层、嵌入层或线性层，初始化权重和偏置
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Embedding) or isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=factor)
            if hasattr(module, "bias") and module.bias is not None:
                module.bias.data.zero_()

        # 如果是InstructBlipVisionEmbeddings，根据配置初始化位置嵌入和类别嵌入
        if isinstance(module, InstructBlipVisionEmbeddings):
            if hasattr(self.config, "vision_config"):
                factor = self.config.vision_config.initializer_range
            nn.init.trunc_normal_(module.position_embedding, mean=0.0, std=factor)
            nn.init.trunc_normal_(module.class_embedding, mean=0.0, std=factor)

        # 如果是LayerNorm层，初始化偏置和权重
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        # 如果是线性层且有偏置，初始化偏置
        elif isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
# 定义模型文档字符串的起始部分，包含了模型的继承关系和参数说明
INSTRUCTBLIP_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`InstructBlipConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

# 定义模型输入文档字符串的起始部分，包含了输入参数的说明
INSTRUCTBLIP_VISION_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`InstructBlipProcessor`]. See
            [`InstructBlipProcessor.__call__`] for details.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""

# 定义模型输入文档字符串的起始部分，暂时为空
INSTRUCTBLIP_INPUTS_DOCSTRING = r"""
"""

# 定义 InstructBlipEncoder 类，继承自 nn.Module，用于实现 Transformer 编码器
class InstructBlipEncoder(nn.Module):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self attention layers. Each layer is a
    [`InstructBlipEncoderLayer`].

    Args:
        config (`InstructBlipConfig`):
            The corresponding vision configuration for the `InstructBlipEncoder`.
    """

    def __init__(self, config: InstructBlipConfig):
        super().__init__()
        self.config = config
        # 创建包含多个 InstructBlipEncoderLayer 的 ModuleList
        self.layers = nn.ModuleList([InstructBlipEncoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(
        self,
        inputs_embeds,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        r"""
        Args:
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Embedded representation of the inputs. Should be float, not int tokens.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        # 设置输出的注意力张量是否包含所有注意力层的信息
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 设置输出的隐藏状态是否包含所有层的信息
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 设置是否返回一个 ModelOutput 对象而不是一个普通元组
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 如果不输出隐藏状态，则初始化为空
        encoder_states = () if output_hidden_states else None
        # 如果不输出注意力，则初始化为空
        all_attentions = () if output_attentions else None

        # 将输入的嵌入表示设置为隐藏状态
        hidden_states = inputs_embeds
        # 遍历每个编码层
        for idx, encoder_layer in enumerate(self.layers):
            # 如果输出隐藏状态，则将当前隐藏状态添加到 encoder_states 中
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            # 如果启用渐变检查点且处于训练模式，则使用渐变检查点函数
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    encoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    output_attentions,
                )
            else:
                # 否则直接调用编码层
                layer_outputs = encoder_layer(
                    hidden_states,
                    attention_mask,
                    output_attentions=output_attentions,
                )

            # 更新隐藏状态为编码层的输出
            hidden_states = layer_outputs[0]

            # 如果输出注意力，则将当前层的注意力添加到 all_attentions 中
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # 如果输出隐藏状态，则将最终隐藏状态添加到 encoder_states 中
        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        # 如果不返回字典，则返回包含隐藏状态、编码状态和注意力的元组
        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        # 否则返回一个 BaseModelOutput 对象
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )
# 从transformers.models.blip.modeling_blip.BlipVisionModel复制而来，将Blip->InstructBlip, BLIP->INSTRUCTBLIP
class InstructBlipVisionModel(InstructBlipPreTrainedModel):
    # 主要输入名称为"pixel_values"
    main_input_name = "pixel_values"
    # 配置类为InstructBlipVisionConfig

    def __init__(self, config: InstructBlipVisionConfig):
        # 调用父类的构造函数，传入配置
        super().__init__(config)
        self.config = config
        embed_dim = config.hidden_size

        # 初始化嵌入层
        self.embeddings = InstructBlipVisionEmbeddings(config)
        # 初始化编码器
        self.encoder = InstructBlipEncoder(config)
        # 初始化后层归一化
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

        # 调用后初始化函数
        self.post_init()

    # 前向传播函数
    @add_start_docstrings_to_model_forward(INSTRUCTBLIP_VISION_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=InstructBlipVisionConfig)
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
        # 如果未指定pixel_values，则引发值错误
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # 将pixel_values传入嵌入层
        hidden_states = self.embeddings(pixel_values)

        # 将嵌入层的输出传入编码器
        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取最后一个隐藏状态并进行后层归一化
        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.post_layernorm(last_hidden_state)

        # 提取池化输出并进行后层归一化
        pooled_output = last_hidden_state[:, 0, :]
        pooled_output = self.post_layernorm(pooled_output)

        # 如果不使用返回字典，则返回元组
        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        # 使用BaseModelOutputWithPooling返回结果
        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

    # 获取输入嵌入层
    def get_input_embeddings(self):
        return self.embeddings


class InstructBlipQFormerMultiHeadAttention(nn.Module):
    # 初始化函数，接受配置和是否跨注意力的参数
    def __init__(self, config, is_cross_attention=False):
        # 调用父类的初始化函数
        super().__init__()
        # 保存配置
        self.config = config
        # 检查隐藏层大小是否是注意力头数的整数倍，如果不是则抛出数值错误
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention heads (%d)"
                % (config.hidden_size, config.num_attention_heads)
            )

        # 保存注意力头数和每个头的大小
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # 创建查询、键和值的线性层
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        if is_cross_attention:
            self.key = nn.Linear(config.encoder_hidden_size, self.all_head_size)
            self.value = nn.Linear(config.encoder_hidden_size, self.all_head_size)
        else:
            self.key = nn.Linear(config.hidden_size, self.all_head_size)
            self.value = nn.Linear(config.hidden_size, self.all_head_size)

        # 创建丢弃层
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        # 获取位置嵌入类型
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        # 如果位置嵌入类型是相对键或相对键查询，则创建距离嵌入层
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)
        # 保存注意力标志
        self.save_attention = False

    # 保存注意力梯度
    def save_attn_gradients(self, attn_gradients):
        self.attn_gradients = attn_gradients

    # 获取注意力梯度
    def get_attn_gradients(self):
        return self.attn_gradients

    # 保存注意力映射
    def save_attention_map(self, attention_map):
        self.attention_map = attention_map

    # 获取注意力映射
    def get_attention_map(self):
        return self.attention_map

    # 调整形状以便计算分数
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    # 前向传播函数
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
# 从transformers.models.bert.modeling_bert.BertSelfOutput复制代码，并将Bert->InstructBlipQFormer
class InstructBlipQFormerSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 创建一个全连接层，输入和输出维度都是config.hidden_size
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 创建LayerNorm层，输入维度是config.hidden_size，eps为config.layer_norm_eps
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 创建Dropout层，概率为config.hidden_dropout_prob
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 通过全连接层处理hidden_states
        hidden_states = self.dense(hidden_states)
        # 使用Dropout层处理hidden_states
        hidden_states = self.dropout(hidden_states)
        # 使用LayerNorm层处理hidden_states和input_tensor的和
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


# 从transformers.models.blip_2.modeling_blip_2.Blip2QFormerAttention复制代码，并将Blip2->InstructBlip
class InstructBlipQFormerAttention(nn.Module):
    def __init__(self, config, is_cross_attention=False):
        super().__init__()
        # 创建InstructBlipQFormerMultiHeadAttention对象
        self.attention = InstructBlipQFormerMultiHeadAttention(config, is_cross_attention)
        # 创建InstructBlipQFormerSelfOutput对象
        self.output = InstructBlipQFormerSelfOutput(config)
        # 初始化pruned_heads为空集合
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        # 找到可剪枝的头部和索引
        heads, index = find_pruneable_heads_and_indices(
            heads, self.attention.num_attention_heads, self.attention.attention_head_size, self.pruned_heads
        )

        # 剪枝线性层
        self.attention.query = prune_linear_layer(self.attention.query, index)
        self.attention.key = prune_linear_layer(self.attention.key, index)
        self.attention.value = prune_linear_layer(self.attention.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # 更新超参数并存储剪枝的头部
        self.attention.num_attention_heads = self.attention.num_attention_heads - len(heads)
        self.attention.all_head_size = self.attention.attention_head_size * self.attention.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        # 使用attention处理hidden_states等参数
        self_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        # 使用output处理attention输出和hidden_states
        attention_output = self.output(self_outputs[0], hidden_states)
        # 如果需要输出attentions，则将其添加到outputs中
        outputs = (attention_output,) + self_outputs[1:]
        return outputs
# 从transformers.models.bert.modeling_bert.BertIntermediate复制并将Bert->InstructBlipQFormer
class InstructBlipQFormerIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 创建一个全连接层，将输入维度为config.hidden_size，输出维度为config.intermediate_size
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        # 如果config.hidden_act是字符串类型，则使用ACT2FN字典中对应的激活函数，否则使用config.hidden_act作为激活函数
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 将输入hidden_states通过全连接层dense
        hidden_states = self.dense(hidden_states)
        # 将全连接层的输出应用激活函数
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


# 从transformers.models.bert.modeling_bert.BertOutput复制并将Bert->InstructBlipQFormer
class InstructBlipQFormerOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 创建一个全连接层，将输入维度为config.intermediate_size，输出维度为config.hidden_size
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        # 创建LayerNorm层，对隐藏状态进行归一化
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 创建Dropout层，用于随机失活
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 将输入hidden_states通过全连接层dense
        hidden_states = self.dense(hidden_states)
        # 对全连接层的输出进行随机失活
        hidden_states = self.dropout(hidden_states)
        # 对全连接层的输出和输入tensor进行残差连接，并进行LayerNorm
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class InstructBlipQFormerLayer(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        # 设置feed forward的chunk大小
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        # 设置序列长度维度
        self.seq_len_dim = 1
        # 创建自注意力层InstructBlipQFormerAttention
        self.attention = InstructBlipQFormerAttention(config)

        self.layer_idx = layer_idx

        # 如果当前层的索引能被config.cross_attention_frequency整除，则创建跨层注意力层InstructBlipQFormerAttention
        if layer_idx % config.cross_attention_frequency == 0:
            self.crossattention = InstructBlipQFormerAttention(config, is_cross_attention=True)
            self.has_cross_attention = True
        else:
            self.has_cross_attention = False

        # 创建中间层InstructBlipQFormerIntermediate
        self.intermediate = InstructBlipQFormerIntermediate(config)
        # 创建输出层InstructBlipQFormerOutput
        self.output = InstructBlipQFormerOutput(config)

        # 创建查询中间层InstructBlipQFormerIntermediate
        self.intermediate_query = InstructBlipQFormerIntermediate(config)
        # 创建查询输出层InstructBlipQFormerOutput
        self.output_query = InstructBlipQFormerOutput(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
        query_length=0,
        # 如果存在过去的键/值缓存元组，则取出前两个元素作为自注意力的过去键/值
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        # 进行自注意力计算
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        # 获取自注意力输出
        attention_output = self_attention_outputs[0]
        # 获取除了自注意力输出之外的其他输出
        outputs = self_attention_outputs[1:-1]

        # 获取当前自注意力的键/值缓存
        present_key_value = self_attention_outputs[-1]

        # 如果查询长度大于0
        if query_length > 0:
            # 截取注意力输出中的查询部分
            query_attention_output = attention_output[:, :query_length, :]

            # 如果存在跨注意力
            if self.has_cross_attention:
                # 如果编码器隐藏状态为空，则抛出异常
                if encoder_hidden_states is None:
                    raise ValueError("encoder_hidden_states must be given for cross-attention layers")
                # 进行跨注意力计算
                cross_attention_outputs = self.crossattention(
                    query_attention_output,
                    attention_mask,
                    head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    output_attentions=output_attentions,
                )
                # 获取跨注意力的输出
                query_attention_output = cross_attention_outputs[0]
                # 如果输出注意力权重，则添加跨注意力
                outputs = outputs + cross_attention_outputs[1:-1]

            # 对查询注意力输出应用分块前向传播
            layer_output = apply_chunking_to_forward(
                self.feed_forward_chunk_query,
                self.chunk_size_feed_forward,
                self.seq_len_dim,
                query_attention_output,
            )

            # 如果注意力输出的长度大于查询长度
            if attention_output.shape[1] > query_length:
                # 对注意力输出中剩余部分应用分块前向传播
                layer_output_text = apply_chunking_to_forward(
                    self.feed_forward_chunk,
                    self.chunk_size_feed_forward,
                    self.seq_len_dim,
                    attention_output[:, query_length:, :],
                )
                # 拼接查询部分和剩余部分的输出
                layer_output = torch.cat([layer_output, layer_output_text], dim=1)
        else:
            # 对注意力输出应用分块前向传播
            layer_output = apply_chunking_to_forward(
                self.feed_forward_chunk,
                self.chunk_size_feed_forward,
                self.seq_len_dim,
                attention_output,
            )
        # 将当前层输出添加到输出中
        outputs = (layer_output,) + outputs

        # 将当前层的键/值缓存添加到输出中
        outputs = outputs + (present_key_value,)

        # 返回输出
        return outputs

    # 前向传播分块函数，用于处理注意力输出
    def feed_forward_chunk(self, attention_output):
        # 中间层计算
        intermediate_output = self.intermediate(attention_output)
        # 输出层计算
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output

    # 前向传播分块函数，用于处理查询注意力输出
    def feed_forward_chunk_query(self, attention_output):
        # 查询中间层计算
        intermediate_output = self.intermediate_query(attention_output)
        # 查询输出层计算
        layer_output = self.output_query(intermediate_output, attention_output)
        return layer_output
# 从transformers.models.blip_2.modeling_blip_2.Blip2QFormerEncoder复制代码，并将Blip2->InstructBlip
class InstructBlipQFormerEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # 创建包含多个InstructBlipQFormerLayer层的ModuleList
        self.layer = nn.ModuleList(
            [InstructBlipQFormerLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        # 是否启用梯度检查点
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
        query_length=0,
    ):  # 定义函数
        all_hidden_states = () if output_hidden_states else None  # 如果output_hidden_states为True，则创建一个空元组，否则为None
        all_self_attentions = () if output_attentions else None  # 如果output_attentions为True，则创建一个空元组，否则为None
        all_cross_attentions = () if output_attentions else None  # 如果output_attentions为True，则创建一个空元组，否则为None

        next_decoder_cache = () if use_cache else None  # 如果use_cache为True，则创建一个空元组，否则为None

        for i in range(self.config.num_hidden_layers):  # 遍历self.config.num_hidden_layers次的循环
            layer_module = self.layer[i]  # 获取第i层的层模块
            if output_hidden_states:  # 如果output_hidden_states为True
                all_hidden_states = all_hidden_states + (hidden_states,)  # 将hidden_states添加到all_hidden_states中的元组中

            layer_head_mask = head_mask[i] if head_mask is not None else None  # 如果head_mask不为None，则取head_mask[i]，否则为None
            past_key_value = past_key_values[i] if past_key_values is not None else None  # 如果past_key_values不为None，则取past_key_values[i]，否则为None

            if getattr(self.config, "gradient_checkpointing", False) and self.training:  # 如果self.config中存在"gradient_checkpointing"属性且为True，并且模型处于训练状态
                if use_cache:  # 如果use_cache为True
                    logger.warning("`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`...")  # 输出警告信息
                    use_cache = False  # 将use_cache设为False
                layer_outputs = self._gradient_checkpointing_func(  # 调用_gradient_checkpointing_func方法
                    layer_module.__call__,  # 传入layer_module.__call__方法
                    hidden_states,  # 传入hidden_states
                    attention_mask,  # 传入attention_mask
                    layer_head_mask,  # 传入layer_head_mask
                    encoder_hidden_states,  # 传入encoder_hidden_states
                    encoder_attention_mask,  # 传入encoder_attention_mask
                )
            else:  # 如果不满足上述条件
                layer_outputs = layer_module(  # 调用layer_module方法
                    hidden_states,  # 传入hidden_states
                    attention_mask,  # 传入attention_mask
                    layer_head_mask,  # 传入layer_head_mask
                    encoder_hidden_states,  # 传入encoder_hidden_states
                    encoder_attention_mask,  # 传入encoder_attention_mask
                    past_key_value,  # 传入past_key_value
                    output_attentions,  # 传入output_attentions
                    query_length,  # 传入query_length
                )

            hidden_states = layer_outputs[0]  # 获取layer_outputs中索引为0的值并赋给hidden_states
            if use_cache:  # 如果use_cache为True
                next_decoder_cache += (layer_outputs[-1],)  # 将layer_outputs中索引为-1的值添加到next_decoder_cache中的元组中
            if output_attentions:  # 如果output_attentions为True
                all_self_attentions = all_self_attentions + (layer_outputs[1],)  # 将layer_outputs中索引为1的值添加到all_self_attentions中的元组中
                if layer_module.has_cross_attention:  # 如果layer_module具有has_cross_attention属性
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)  # 将layer_outputs中索引为2的值添加到all_cross_attentions中的元组中

        if output_hidden_states:  # 如果output_hidden_states为True
            all_hidden_states = all_hidden_states + (hidden_states,)  # 将hidden_states添加到all_hidden_states中的元组中

        if not return_dict:  # 如果return_dict为False
            return tuple(  # 返回一个元组
                v  # 元组中的元素
                for v in [  # 遍历列表中的元素
                    hidden_states,  # 将hidden_states添加到元组中
                    next_decoder_cache,  # 将next_decoder_cache添加到元组中
                    all_hidden_states,  # 将all_hidden_states添加到元组中
                    all_self_attentions,  # 将all_self_attentions添加到元组中
                    all_cross_attentions,  # 将all_cross_attentions添加到元组中
                ]
                if v is not None  # 如果v不为None
            )
        return BaseModelOutputWithPastAndCrossAttentions(  # 返回BaseModelOutputWithPastAndCrossAttentions对象
            last_hidden_state=hidden_states,  # 设置last_hidden_state属性为hidden_states
            past_key_values=next_decoder_cache,  # 设置past_key_values属性为next_decoder_cache
            hidden_states=all_hidden_states,  # 设置hidden_states属性为all_hidden_states
            attentions=all_self_attentions,  # 设置attentions属性为all_self_attentions
            cross_attentions=all_cross_attentions,  # 设置cross_attentions属性为all_cross_attentions
        )
class InstructBlipQFormerEmbeddings(nn.Module):
    """构建单词和位置嵌入的嵌入层。"""

    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()
        # 创建一个单词嵌入层，根据词汇表的大小和隐藏层大小来初始化，设置padding的index为config中的pad_token_id
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        # 创建一个位置嵌入层，根据最大位置嵌入的长度和隐藏层大小来初始化
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)

        # 使用LayerNorm对隐藏层进行归一化处理
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 使用Dropout对隐藏层进行随机失活处理
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # position_ids (1, len position emb) 在序列化时是内存中连续的，并在导出时被导出
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )
        # 设置位置嵌入的类型为绝对位置嵌入
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")

        # 保存配置信息
        self.config = config

    def forward(
        self,
        input_ids=None,
        position_ids=None,
        query_embeds=None,
        past_key_values_length=0,
    ):
        # 如果input_ids不为空，则获取序列的长度
        if input_ids is not None:
            seq_length = input_ids.size()[1]
        else:
            seq_length = 0

        # 如果位置ids为空，则从position_ids中获取位置信息
        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length].clone()

        # 如果input_ids不为空，则获取单词嵌入并根据位置嵌入类型进行处理
        if input_ids is not None:
            embeddings = self.word_embeddings(input_ids)
            if self.position_embedding_type == "absolute":
                position_embeddings = self.position_embeddings(position_ids.to(embeddings.device))
                embeddings = embeddings + position_embeddings

            # 如果query_embeds不为空，则将其与embeddings拼接起来
            if query_embeds is not None:
                embeddings = torch.cat((query_embeds, embeddings), dim=1)
        else:
            embeddings = query_embeds

        # 将embeddings转移到与layernorm权重相同的设备上
        embeddings = embeddings.to(self.layernorm.weight.dtype)
        # 对embeddings进行layernorm处理
        embeddings = self.layernorm(embeddings)
        # 对embeddings进行dropout处理
        embeddings = self.dropout(embeddings)
        # 返回处理后的embeddings
        return embeddings


class InstructBlipQFormerModel(InstructBlipPreTrainedModel):
    """
    查询变换器（Q-Former），在InstructBLIP中使用。与BLIP-2略有修改，因为它还将指令作为输入。
    """

    def __init__(self, config: InstructBlipQFormerConfig):
        # 调用父类的初始化方法
        super().__init__(config)
        # 保存配置信息
        self.config = config

        # 创建嵌入层
        self.embeddings = InstructBlipQFormerEmbeddings(config)

        # 创建编码器
        self.encoder = InstructBlipQFormerEncoder(config)

        # 执行初始化后的操作
        self.post_init()

    def get_input_embeddings(self):
        # 返回单词嵌入层
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        # 设置单词嵌入层的值
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        修剪模型的头部。heads_to_prune: {layer_num: 要在此层中修剪的头部列表} 参见基类PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            # 对模型进行头部修剪
            self.encoder.layer[layer].attention.prune_heads(heads)
    def get_extended_attention_mask(
        self,
        attention_mask: torch.Tensor,
        input_shape: Tuple[int],
        device: torch.device,
        has_query: bool = False,
    ) -> torch.Tensor:
        """
        Makes broadcastable attention and causal masks so that future and masked tokens are ignored.

        Arguments:
            attention_mask (`torch.Tensor`):
                Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
            input_shape (`Tuple[int]`):
                The shape of the input to the model.
            device: (`torch.device`):
                The device of the input to the model.

        Returns:
            `torch.Tensor` The extended attention mask, with a the same dtype as `attention_mask.dtype`.
        """
        # 如果注意力掩码的维度为3，则表示提供了自定义的自注意力掩码，形状为 [batch_size, from_seq_length, to_seq_length]
        # 我们只需要将其广播到所有注意力头部即可
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        # 如果注意力掩码的维度为2，则表示提供了形状为 [batch_size, seq_length] 的填充掩码
        # - 模型是一个编码器，因此将掩码广播到 [batch_size, num_heads, seq_length, seq_length]
        elif attention_mask.dim() == 2:
            extended_attention_mask = attention_mask[:, None, None, :]
        else:
            # 抛出异常，提示输入的input_ids形状（shape {input_shape}）或attention_mask形状（shape {attention_mask.shape}）错误
            raise ValueError(
                f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})",
            )

        # 由于注意力掩码中1.0表示要注意的位置，0.0表示掩盖的位置，因此此操作将创建一个张量，
        # 其中0.0表示要注意的位置，-10000.0表示掩盖的位置。
        # 由于我们在softmax之前将其添加到原始分数中，这实际上等同于完全删除这些位置。
        extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        query_embeds: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
# 使用add_start_docstrings装饰器添加模型文档字符串，介绍了InstructBLIP模型的组成和功能
# 此类继承自InstructBlipPreTrainedModel，是用于根据图像和可选文本提示生成文本的模型
class InstructBlipForConditionalGeneration(InstructBlipPreTrainedModel):
    # 配置类为InstructBlipConfig
    config_class = InstructBlipConfig
    # 主要输入名称为"pixel_values"
    main_input_name = "pixel_values"

    # 初始化方法，接收一个InstructBlipConfig类型的配置对象
    def __init__(self, config: InstructBlipConfig):
        # 调用父类的初始化方法
        super().__init__(config)
        # 使用InstructBlipVisionModel类创建一个视觉模型
        self.vision_model = InstructBlipVisionModel(config.vision_config)
        # 创建一个可查询的token参数
        self.query_tokens = nn.Parameter(torch.zeros(1, config.num_query_tokens, config.qformer_config.hidden_size))
        # 使用InstructBlipQFormerModel类创建一个Q-Former模型
        self.qformer = InstructBlipQFormerModel(config.qformer_config)
        # 创建一个线性层，用于将Q-Former的隐藏状态映射到文本模型的隐藏状态
        self.language_projection = nn.Linear(config.qformer_config.hidden_size, config.text_config.hidden_size)
        
        # 根据配置选择性地使用decoder-only的语言模型或者seq2seq的语言模型
        if config.use_decoder_only_language_model:
            # 根据配置创建一个AutoModelForCausalLM类的语言模型
            language_model = AutoModelForCausalLM.from_config(config.text_config)
        else:
            # 根据配置创建一个AutoModelForSeq2SeqLM类的语言模型
            language_model = AutoModelForSeq2SeqLM.from_config(config.text_config)
        
        # 如果语言模型中有不拆分的模块，则将其添加到不拆分模块列表中
        if language_model._no_split_modules is not None:
            self._no_split_modules.extend(language_model._no_split_modules)
        
        # 如果语言模型中有需要保持在fp32精度的模块，则将其添加到保持在fp32精度模块列表中
        if language_model._keep_in_fp32_modules is not None:
            self._keep_in_fp32_modules.extend(language_model._keep_in_fp32_modules)
        
        # 将创建的语言模型赋值给self.language_model
        self.language_model = language_model

        # 初始化权重并进行最终处理
        self.post_init()

    # 获取输入嵌入层
    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    # 设置输入嵌入层
    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    # 设置输出嵌入层
    def set_output_embeddings(self, new_embeddings):
        self.language_model.set_output_embeddings(new_embeddings)

    # 获取输出嵌入层
    def get_output_embeddings(self) -> nn.Module:
        return self.language_model.get_output_embeddings()

    # 获取编码器
    def get_encoder(self):
        return self.language_model.get_encoder()

    # 获取解码器
    def get_decoder(self):
        return self.language_model.get_decoder()

    # 将权重绑定
    def _tie_weights(self):
        # 如果不是仅使用解码器的语言模型，则将编码器和解码器的嵌入层绑定到共享的嵌入层
        if not self.config.use_decoder_only_language_model:
            self.language_model.encoder.embed_tokens = self.language_model.shared
            self.language_model.decoder.embed_tokens = self.language_model.shared
    # 定义一个内部方法，用于预处理加速模型所需的一些特殊处理
    def _preprocess_accelerate(self):
        r"""
        Some pre-processing hacks to make the model `accelerate` compatible. Check
        https://github.com/huggingface/transformers/pull/21707 for more details.
        """
        # 获取模型的设备映射
        hf_device_map = self.hf_device_map

        # 如果设备映射中包含多个设备，并且不包含“language_model”，并且 CUDA 设备数量大于1
        if len(hf_device_map) > 1 and "language_model" not in hf_device_map and torch.cuda.device_count() > 1:
            # 警告用户在使用多GPU + InstructBLIP + `accelerate`时可能出现意外行为
            logger.warning(
                "The `language_model` is not in the `hf_device_map` dictionary and you are running your script"
                " in a multi-GPU environment. this may lead to unexpected behavior when using `accelerate`."
                " Please pass a `device_map` that contains `language_model` to remove this warning."
                " Please refer to https://github.com/huggingface/blog/blob/main/accelerate-large-models.md for"
                " more details on creating a `device_map` for large models.",
            )

        # 如果语言模型具有“_hf_hook”属性
        if hasattr(self.language_model, "_hf_hook"):
            self.language_model._hf_hook.io_same_device = True  # 为了 `generate` 兼容性

    # 添加模型前向方法的参数注释和描述
    @add_start_docstrings_to_model_forward(INSTRUCTBLIP_INPUTS_DOCSTRING)
    @replace_return_docstrings(
        output_type=InstructBlipForConditionalGenerationModelOutput, config_class=InstructBlipVisionConfig
    )
    # 定义模型的前向方法
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        qformer_input_ids: torch.FloatTensor,
        qformer_attention_mask: Optional[torch.LongTensor] = None,
        input_ids: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
    # 禁用梯度计算的装饰器
    @torch.no_grad()
    # 定义 generate 方法
    def generate(
        self,
        pixel_values: torch.FloatTensor,
        qformer_input_ids: Optional[torch.LongTensor] = None,
        qformer_attention_mask: Optional[torch.LongTensor] = None,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        **generate_kwargs,
```