# `.\transformers\models\blip_2\modeling_blip_2.py`

```py
# 设置文件编码为 UTF-8
# 版权声明及许可协议
# 注意：该代码涉及版权声明和许可协议，要确保在使用时遵循相应的法律规定和许可协议。
# 若使用该代码，需要遵循 Apache 许可证 2.0 版本，可以从给定网址获取许可证文本
# 详细许可信息可以在上述网址找到
""" PyTorch BLIP-2 model."""
# 导入模块
import math
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss

# 导入其他模块中的类和函数
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
from .configuration_blip_2 import Blip2Config, Blip2QFormerConfig, Blip2VisionConfig

# 获取日志记录器
logger = logging.get_logger(__name__)

# 文档中使用的检查点
_CHECKPOINT_FOR_DOC = "Salesforce/blip2-opt-2.7b"

# BLIP-2 预训练模型存档列表
BLIP_2_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "Salesforce/blip2-opt-2.7b",
    # 查看所有 BLIP-2 模型：https://huggingface.co/models?filter=blip
]

# 定义一个数据类，用于存储 `Blip2ForConditionalGeneration` 的输出
@dataclass
class Blip2ForConditionalGenerationModelOutput(ModelOutput):
    """
    Class defining the outputs of [`Blip2ForConditionalGeneration`].

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

    # 语言模型的语言建模损失，当提供了 `labels` 参数时返回
    loss: Optional[Tuple[torch.FloatTensor]] = None
    # 语言模型头的预测得分，形状为 `(batch_size, sequence_length, config.vocab_size)`
    logits: Optional[Tuple[torch.FloatTensor]] = None
    # 视觉编码器的输出
    vision_outputs: Optional[torch.FloatTensor] = None
    # Q-Former（查询转换器）的输出
    qformer_outputs: Optional[Tuple[torch.FloatTensor]] = None
    # 语言模型的输出
    language_model_outputs: Optional[Tuple[torch.FloatTensor]] = None
    # 将对象转换为元组的方法，返回包含对象所有键值的元组
    def to_tuple(self) -> Tuple[Any]:
        # 使用生成器表达式构建元组
        return tuple(
            # 如果键不是特定的键，则直接将对应的值添加到元组中
            self[k]
            # 如果键是特定的键之一，则调用相应属性的to_tuple()方法，并将结果添加到元组中
            if k not in ["vision_outputs", "qformer_outputs", "language_model_outputs"]
            else getattr(self, k).to_tuple()  # 调用特定属性的to_tuple()方法，并将结果添加到元组中
            for k in self.keys()  # 遍历对象的所有键
        )
# 从transformers.models.blip.modeling_blip.BlipVisionEmbeddings复制代码，并将Blip->Blip2
class Blip2VisionEmbeddings(nn.Module):
    def __init__(self, config: Blip2VisionConfig):
        # 初始化Blip2VisionEmbeddings类
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        # 创建表示类别嵌入的可学习参数
        self.class_embedding = nn.Parameter(torch.randn(1, 1, self.embed_dim))

        # 创建用于将像素值转换为嵌入向量的卷积层
        self.patch_embedding = nn.Conv2d(
            in_channels=3, out_channels=self.embed_dim, kernel_size=self.patch_size, stride=self.patch_size
        )

        # 计算图像中补丁的数量和位置的嵌入维度
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1

        # 创建位置嵌入的可学习参数
        self.position_embedding = nn.Parameter(torch.randn(1, self.num_positions, self.embed_dim))

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        # 获取输入张量的批次大小和目标数据类型
        batch_size = pixel_values.shape[0]
        target_dtype = self.patch_embedding.weight.dtype
        # 将输入像素值转换为嵌入向量
        patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype))  # shape = [*, width, grid, grid]
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)

        # 扩展类别嵌入以匹配批次大小，并转换为目标数据类型
        class_embeds = self.class_embedding.expand(batch_size, 1, -1).to(target_dtype)
        # 拼接类别嵌入和补丁嵌入以获得最终嵌入表示
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
        # 添加位置嵌入到最终嵌入表示中
        embeddings = embeddings + self.position_embedding[:, : embeddings.size(1), :].to(target_dtype)
        return embeddings


class Blip2Attention(nn.Module):
    """从'Attention Is All You Need'论文中的多头注意力机制"""

    def __init__(self, config):
        # 初始化Blip2Attention类
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

        # 线性变换用于计算查询、键和值
        self.qkv = nn.Linear(self.embed_dim, 3 * self.embed_dim, bias=False)

        if config.qkv_bias:
            q_bias = nn.Parameter(torch.zeros(self.embed_dim))
            v_bias = nn.Parameter(torch.zeros(self.embed_dim))
        else:
            q_bias = None
            v_bias = None

        if q_bias is not None:
            # 如果设置了qkv_bias，则设置偏置参数
            qkv_bias = torch.cat((q_bias, torch.zeros_like(v_bias, requires_grad=False), v_bias))
            self.qkv.bias = nn.Parameter(qkv_bias)

        # 线性变换用于投影注意力输出
        self.projection = nn.Linear(self.embed_dim, self.embed_dim)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        # 重塑张量形状以便用于注意力计算
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        # 获取输入张量的形状信息：Batch size、Time steps、Channel（通道数）
        bsz, tgt_len, embed_dim = hidden_states.size()

        # 将输入张量通过 QKV 线性变换矩阵进行混合
        mixed_qkv = self.qkv(hidden_states)

        # 重塑混合后的张量以便分离 Q、K、V，并调整维度顺序
        mixed_qkv = mixed_qkv.reshape(bsz, tgt_len, 3, self.num_heads, embed_dim // self.num_heads).permute(
            2, 0, 3, 1, 4
        )
        query_states, key_states, value_states = mixed_qkv[0], mixed_qkv[1], mixed_qkv[2]

        # 计算原始的注意力分数，即Q与K的点积
        attention_scores = torch.matmul(query_states, key_states.transpose(-1, -2))

        # 缩放注意力分数
        attention_scores = attention_scores * self.scale

        # 将注意力分数归一化为概率
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # 对注意力分数进行 dropout 操作
        attention_probs = self.dropout(attention_probs)

        # 如果指定了头部掩码，则应用头部掩码
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        # 计算上下文张量，即注意力分数加权的V
        context_layer = torch.matmul(attention_probs, value_states).permute(0, 2, 1, 3)

        # 重塑上下文张量以便输出
        new_context_layer_shape = context_layer.size()[:-2] + (self.embed_dim,)
        context_layer = context_layer.reshape(new_context_layer_shape)

        # 通过投影层得到输出
        output = self.projection(context_layer)

        # 若需要输出注意力分数，则包含在输出中
        outputs = (output, attention_probs) if output_attentions else (output, None)

        return outputs
# 定义一个名为 Blip2MLP 的类，继承自 nn.Module
class Blip2MLP(nn.Module):
    def __init__(self, config):
        # 调用父类的构造函数
        super().__init__()
        # 将传入的配置保存到实例中
        self.config = config
        # 根据配置中的隐藏激活函数选择对应的激活函数
        self.activation_fn = ACT2FN[config.hidden_act]
        # 创建第一个全连接层，输入大小为隐藏大小，输出大小为中间大小
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        # 创建第二个全连接层，输入大小为中间大小，输出大小为隐藏大小
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    # 定义前向传播方法
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 输入数据经过第一个全连接层
        hidden_states = self.fc1(hidden_states)
        # 使用激活函数激活
        hidden_states = self.activation_fn(hidden_states)
        # 再经过第二个全连接层
        hidden_states = self.fc2(hidden_states)
        # 返回处理后的隐藏状态
        return hidden_states


# 定义一个名为 Blip2EncoderLayer 的类，继承自 nn.Module
class Blip2EncoderLayer(nn.Module):
    # 定义初始化方法
    def __init__(self, config: Blip2Config):
        # 调用父类的构造函数
        super().__init__()
        # 设置嵌入维度为隐藏大小
        self.embed_dim = config.hidden_size
        # 创建注意力机制模块
        self.self_attn = Blip2Attention(config)
        # 创建第一个层规范化层，输入大小为嵌入维度
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        # 创建 MLP 模块
        self.mlp = Blip2MLP(config)
        # 创建第二个层规范化层
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    # 定义前向传播方法
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
        # 将输入状态作为残差连接的一部分
        residual = hidden_states

        # 应用第一个层规范化层
        hidden_states = self.layer_norm1(hidden_states)
        # 应用注意力机制
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            head_mask=attention_mask,
            output_attentions=output_attentions,
        )
        # 将注意力机制的输出与残差连接
        hidden_states = hidden_states + residual
        # 保存残差以备后续使用
        residual = hidden_states
        # 应用第二个层规范化层
        hidden_states = self.layer_norm2(hidden_states)
        # 经过 MLP 模块
        hidden_states = self.mlp(hidden_states)

        # 将 MLP 模块的输出与残差连接
        hidden_states = hidden_states + residual

        # 将输出打包为元组
        outputs = (hidden_states,)

        # 如果需要返回注意力权重
        if output_attentions:
            outputs += (attn_weights,)

        # 返回输出
        return outputs


# 定义一个名为 Blip2PreTrainedModel 的类，继承自 PreTrainedModel
class Blip2PreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 配置类为 Blip2Config
    config_class = Blip2Config
    # 模型前缀为 "blip"
    base_model_prefix = "blip"
    # 支持梯度检查点
    supports_gradient_checkpointing = True
    # 需要跳过的模块名称列表
    _no_split_modules = ["Blip2Attention", "T5Block", "OPTDecoderLayer"]
    # 在设备中跳过的键名
    _skip_keys_device_placement = "past_key_values"
    # 定义需要保持在 FP32 模块中的模块名称列表
    _keep_in_fp32_modules = ["wo"]

    # 初始化模型权重
    def _init_weights(self, module):
        """Initialize the weights"""
        # 获取初始化因子
        factor = self.config.initializer_range
        # 如果模块是卷积层、嵌入层或全连接层
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Embedding) or isinstance(module, nn.Linear):
            # 使用正态分布初始化权重
            module.weight.data.normal_(mean=0.0, std=factor)
            # 如果模块有偏置项且不为None，则将偏置项初始化为0
            if hasattr(module, "bias") and module.bias is not None:
                module.bias.data.zero_()

        # 如果模块是Blip2VisionEmbeddings类型
        if isinstance(module, Blip2VisionEmbeddings):
            # 如果配置中有vision_config，则使用其初始化范围
            if hasattr(self.config, "vision_config"):
                factor = self.config.vision_config.initializer_range
            # 使用截断正态分布初始化位置嵌入和类别嵌入
            nn.init.trunc_normal_(module.position_embedding, mean=0.0, std=factor)
            nn.init.trunc_normal_(module.class_embedding, mean=0.0, std=factor)

        # 如果模块是LayerNorm类型
        elif isinstance(module, nn.LayerNorm):
            # 将偏置项初始化为0
            module.bias.data.zero_()
            # 将权重初始化为1
            module.weight.data.fill_(1.0)
        # 如果模块是全连接层且有偏置项
        elif isinstance(module, nn.Linear) and module.bias is not None:
            # 将偏置项初始化为0
            module.bias.data.zero_()
# BLIP_2_START_DOCSTRING 是一个包含模型说明文档的字符串常量
BLIP_2_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`Blip2Config`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

# BLIP_2_VISION_INPUTS_DOCSTRING 是一个包含视觉输入说明文档的字符串常量
BLIP_2_VISION_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`Blip2Processor`]. See [`Blip2Processor.__call__`] for
            details.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""

# BLIP_2_TEXT_INPUTS_DOCSTRING 是一个包含文本输入说明文档的字符串常量
BLIP_2_TEXT_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            # 输入序列标记在词汇表中的索引。默认情况下，将忽略填充。可以使用 AutoTokenizer 获取这些索引。
            # 有关输入 ID 的详细信息，请参阅 PreTrainedTokenizer.encode 和 PreTrainedTokenizer.__call__。
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            # 避免在填充标记索引上执行注意力的掩码。掩码值选在 `[0, 1]`：
            # - 1 表示**未被掩码**的标记，
            # - 0 表示**被掩码**的标记。
        decoder_input_ids (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*):
            # 解码器输入序列标记在词汇表中的索引。
            # 可以使用 AutoTokenizer 获取这些索引。
            # T5 使用 `pad_token_id` 作为 `decoder_input_ids` 生成的起始标记。
            # 如果使用 `past_key_values`，可以选择仅输入最后的 `decoder_input_ids`。
        decoder_attention_mask (`torch.BoolTensor` of shape `(batch_size, target_sequence_length)`, *optional*):
            # 默认行为：生成一个忽略 `decoder_input_ids` 中填充标记的张量。默认还将使用因果掩码。
        output_attentions (`bool`, *optional*):
            # 是否返回所有注意力层的注意力张量。有关更多详细信息，请参阅返回张量中的 `attentions`。
        output_hidden_states (`bool`, *optional*):
            # 是否返回所有层的隐藏状态。有关更多详细信息，请参阅返回张量中的 `hidden_states`。
        return_dict (`bool`, *optional*):
            # 是否返回一个 `~utils.ModelOutput` 而不是一个普通元组。
"""

BLIP_2_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`Blip2Processor`]. See [`Blip2Processor.__call__`] for
            details.

        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of input sequence tokens in the vocabulary of the language model. Input tokens can optionally be
            provided to serve as text prompt, which the language model can continue.

            Indices can be obtained using [`Blip2Processor`]. See [`Blip2Processor.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

        decoder_input_ids (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*):
            Indices of decoder input sequence tokens in the vocabulary of the language model. Only relevant in case an
            encoder-decoder language model (like T5) is used.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details. [What are decoder input IDs?](../glossary#decoder-input-ids)

        decoder_attention_mask (`torch.BoolTensor` of shape `(batch_size, target_sequence_length)`, *optional*):
            Default behavior: generate a tensor that ignores pad tokens in `decoder_input_ids`. Causal mask will also
            be used by default.

            Only relevant in case an encoder-decoder language model (like T5) is used.

        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


# Copied from transformers.models.blip.modeling_blip.BlipEncoder with Blip->Blip2
class Blip2Encoder(nn.Module):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self attention layers. Each layer is a
    [`Blip2EncoderLayer`].

    Args:
        config (`Blip2Config`):
            The corresponding vision configuration for the `Blip2Encoder`.
    """
    # 初始化函数，接受一个Blip2Config类型的参数config
    def __init__(self, config: Blip2Config):
        # 调用父类的初始化函数
        super().__init__()
        # 将传入的config参数保存到self.config中
        self.config = config
        # 创建包含多个Blip2EncoderLayer对象的列表，列表长度为config.num_hidden_layers
        self.layers = nn.ModuleList([Blip2EncoderLayer(config) for _ in range(config.num_hidden_layers)])
        # 是否启用梯度检查点，默认为False
        self.gradient_checkpointing = False

    # 前向传播函数
    def forward(
        self,
        inputs_embeds,
        # 可选参数，注意力掩码，默认为None
        attention_mask: Optional[torch.Tensor] = None,
        # 可选参数，是否输出注意力权重，默认为None
        output_attentions: Optional[bool] = None,
        # 可选参数，是否输出隐藏状态，默认为None
        output_hidden_states: Optional[bool] = None,
        # 可选参数，是否返回字典格式的输出，默认为None
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
        # 确定是否输出注意力张量
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 确定是否输出隐藏状态张量
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 确定是否返回字典形式的结果
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 如果不输出隐藏状态，则将 encoder_states 设为空元组
        encoder_states = () if output_hidden_states else None
        # 如果不输出注意力张量，则将 all_attentions 设为空元组
        all_attentions = () if output_attentions else None

        # 初始隐藏状态为输入的嵌入表示
        hidden_states = inputs_embeds
        # 遍历每个编码器层
        for idx, encoder_layer in enumerate(self.layers):
            # 如果要输出隐藏状态，则记录当前隐藏状态
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            # 如果启用渐变检查点且处于训练模式，则使用渐变检查点函数调用编码器层
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    encoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    output_attentions,
                )
            else:
                # 否则直接调用编码器层
                layer_outputs = encoder_layer(
                    hidden_states,
                    attention_mask,
                    output_attentions=output_attentions,
                )

            # 更新隐藏状态为编码器层的输出的第一个元素（通常为最终隐藏状态）
            hidden_states = layer_outputs[0]

            # 如果要输出注意力张量，则记录当前层的注意力张量
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # 如果要输出隐藏状态，则记录最终隐藏状态
        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        # 如果不返回字典形式的结果，则返回一个包含隐藏状态、隐藏状态序列和注意力张量序列的元组
        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        # 否则返回一个 BaseModelOutput 对象，包含最终隐藏状态、隐藏状态序列和注意力张量序列
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )
# 从 transformers.models.blip.modeling_blip.BlipVisionModel 复制代码，并将 Blip->Blip2, BLIP->BLIP_2
class Blip2VisionModel(Blip2PreTrainedModel):
    # 主要输入名称为 "pixel_values"
    main_input_name = "pixel_values"
    # 配置类为 Blip2VisionConfig
    config_class = Blip2VisionConfig

    def __init__(self, config: Blip2VisionConfig):
        # 调用父类的构造函数
        super().__init__(config)
        # 存储配置信息
        self.config = config
        # 嵌入维度为配置中的隐藏大小
        embed_dim = config.hidden_size

        # 创建 Blip2VisionEmbeddings 对象
        self.embeddings = Blip2VisionEmbeddings(config)
        # 创建 Blip2Encoder 对象
        self.encoder = Blip2Encoder(config)
        # 创建 LayerNorm 层，用于后处理
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

        # 执行后处理初始化操作
        self.post_init()

    @add_start_docstrings_to_model_forward(BLIP_2_VISION_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=Blip2VisionConfig)
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        """
        前向传播函数

        Args:
            pixel_values (Optional[torch.FloatTensor]): 输入像素值，可选
            output_attentions (Optional[bool]): 是否输出注意力权重，可选
            output_hidden_states (Optional[bool]): 是否输出隐藏状态，可选
            return_dict (Optional[bool]): 是否返回字典格式输出，可选

        Returns:
            Union[Tuple, BaseModelOutputWithPooling]: 返回模型输出
        """
        # 如果未提供像素值，则引发 ValueError
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # 将像素值传入嵌入层
        hidden_states = self.embeddings(pixel_values)

        # 将嵌入向量传入编码器
        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取最后一个隐藏状态，并对其进行后处理
        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.post_layernorm(last_hidden_state)

        # 获取池化输出，并对其进行后处理
        pooled_output = last_hidden_state[:, 0, :]
        pooled_output = self.post_layernorm(pooled_output)

        # 如果不返回字典，则返回元组形式的输出
        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        # 返回字典形式的输出
        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

    def get_input_embeddings(self):
        # 获取输入嵌入层
        return self.embeddings


class Blip2QFormerMultiHeadAttention(nn.Module):
    # 在此处添加 Blip2QFormerMultiHeadAttention 类的注释
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
# 从transformers.models.bert.modeling_bert.BertSelfOutput复制并修改为Blip2QFormerSelfOutput类
class Blip2QFormerSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 初始化线性层，用于将隐藏状态转换为相同维度的输出
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 初始化 LayerNorm 层，用于对输出进行归一化
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 初始化 Dropout 层，用于随机失活以防止过拟合
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 使用线性层对隐藏状态进行转换
        hidden_states = self.dense(hidden_states)
        # 对转换后的隐藏状态进行随机失活
        hidden_states = self.dropout(hidden_states)
        # 将随机失活后的隐藏状态与输入张量相加，再进行 LayerNorm 归一化
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        # 返回归一化后的输出张量
        return hidden_states


# 从transformers.models.bert.modeling_bert.BertAttention复制并修改为Blip2QFormerAttention类
class Blip2QFormerAttention(nn.Module):
    def __init__(self, config, is_cross_attention=False):
        super().__init__()
        # 初始化注意力机制层，使用Blip2QFormerMultiHeadAttention类
        self.attention = Blip2QFormerMultiHeadAttention(config, is_cross_attention)
        # 初始化输出层，使用Blip2QFormerSelfOutput类
        self.output = Blip2QFormerSelfOutput(config)
        # 初始化一个集合，用于存储被剪枝的注意力头
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        # 使用帮助函数查找可剪枝的注意力头及其索引
        heads, index = find_pruneable_heads_and_indices(
            heads, self.attention.num_attention_heads, self.attention.attention_head_size, self.pruned_heads
        )

        # 剪枝线性层
        self.attention.query = prune_linear_layer(self.attention.query, index)
        self.attention.key = prune_linear_layer(self.attention.key, index)
        self.attention.value = prune_linear_layer(self.attention.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # 更新超参数并存储被剪枝的头
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
        # 通过注意力机制层获取自注意力的输出
        self_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        # 将自注意力的输出与输入张量进行连接并通过输出层得到最终输出
        attention_output = self.output(self_outputs[0], hidden_states)
        # 如果需要输出注意力权重，则在输出中添加注意力权重
        outputs = (attention_output,) + self_outputs[1:]  # 如果输出注意力，则添加
        # 返回输出
        return outputs


# 从transformers.models.bert.modeling_bert.BertIntermediate复制并修改为Blip2QFormerIntermediate类
class Blip2QFormerIntermediate(nn.Module):
    # 初始化函数，用于创建一个新的实例
    def __init__(self, config):
        # 调用父类的初始化函数
        super().__init__()
        # 创建一个全连接层，输入大小为 config.hidden_size，输出大小为 config.intermediate_size
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        # 判断 config.hidden_act 是否是字符串类型
        if isinstance(config.hidden_act, str):
            # 如果是字符串类型，则使用预定义的激活函数字典 ACT2FN 中对应的激活函数
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            # 如果不是字符串类型，则直接使用 config.hidden_act 作为激活函数
            self.intermediate_act_fn = config.hidden_act
    
    # 前向传播函数，用于执行模型的前向计算
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 将输入张量通过全连接层进行线性变换
        hidden_states = self.dense(hidden_states)
        # 将线性变换后的结果通过激活函数进行非线性变换
        hidden_states = self.intermediate_act_fn(hidden_states)
        # 返回变换后的结果张量
        return hidden_states
# 从 transformers.models.bert.modeling_bert.BertOutput 复制并修改为 Blip2QFormerOutput 类
class Blip2QFormerOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 线性变换层，将隐藏状态的维度从 intermediate_size 转换为 hidden_size
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        # Layer normalization 层，用于规范化隐藏状态
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # Dropout 层，用于防止过拟合
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 全连接层，将隐藏状态进行线性变换
        hidden_states = self.dense(hidden_states)
        # 使用 Dropout 进行正则化
        hidden_states = self.dropout(hidden_states)
        # 对线性变换结果进行 Layer normalization，并将其与输入张量相加
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        # 返回处理后的隐藏状态
        return hidden_states


# Blip2QFormerLayer 类，表示 Blip2QFormer 模型的一个层
class Blip2QFormerLayer(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        # 设置 Feed Forward 层的分块大小
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        # 序列长度维度索引
        self.seq_len_dim = 1
        # 注意力机制，用于处理注意力权重
        self.attention = Blip2QFormerAttention(config)

        # 当前层的索引
        self.layer_idx = layer_idx

        # 若当前层需要跨层注意力，创建跨层注意力机制
        if layer_idx % config.cross_attention_frequency == 0:
            self.crossattention = Blip2QFormerAttention(config, is_cross_attention=True)
            self.has_cross_attention = True
        else:
            self.has_cross_attention = False

        # 中间层，用于转换注意力输出
        self.intermediate_query = Blip2QFormerIntermediate(config)
        # 输出层，将中间层的输出转换为最终输出
        self.output_query = Blip2QFormerOutput(config)

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
# 定义一个名为Blip2QFormerEncoder的类，继承自nn.Module
class Blip2QFormerEncoder(nn.Module):
    # 初始化方法，接受config参数
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()
        # 将传入的config参数保存到self.config中
        self.config = config
        # 创建一个包含多个Blip2QFormerLayer对象的ModuleList，数量由config.num_hidden_layers确定
        self.layer = nn.ModuleList(
            [Blip2QFormerLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        # 初始化梯度检查点为False
        self.gradient_checkpointing = False

    # 前向传播方法
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
        # 如果不输出隐藏状态，则初始化所有隐藏状态为一个空元组，否则初始化为 None
        all_hidden_states = () if output_hidden_states else None
        # 如果不输出注意力权重，则初始化所有自注意力权重和交叉注意力权重为一个空元组，否则初始化为 None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions else None

        # 如果不使用缓存，则初始化下一个解码器缓存为一个空元组，否则初始化为 None
        next_decoder_cache = () if use_cache else None

        # 遍历每个编码器层
        for i in range(self.config.num_hidden_layers):
            # 获取当前层的模块
            layer_module = self.layer[i]
            # 如果输出隐藏状态，则将当前隐藏状态加入所有隐藏状态元组中
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 获取当前层的头部遮罩
            layer_head_mask = head_mask[i] if head_mask is not None else None
            # 获取过去的键值对（用于缓存）
            past_key_value = past_key_values[i] if past_key_values is not None else None

            # 如果启用梯度检查点且在训练中，则使用梯度检查点函数计算当前层的输出
            if getattr(self.config, "gradient_checkpointing", False) and self.training:
                # 如果使用缓存，警告不兼容，并将 use_cache 设置为 False
                if use_cache:
                    logger.warning(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False
                # 使用梯度检查点函数计算当前层的输出
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                # 否则直接调用当前层的前向传播函数计算当前层的输出
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                    query_length,
                )

            # 更新隐藏状态为当前层的输出的第一个元素
            hidden_states = layer_outputs[0]
            # 如果使用缓存，则将当前层的输出（缓存）加入下一个解码器缓存元组中
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            # 如果输出注意力权重，则将当前层的自注意力权重加入所有自注意力权重元组中
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                # 如果当前层具有交叉注意力，则将当前层的交叉注意力权重加入所有交叉注意力权重元组中
                if layer_module.has_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        # 如果输出隐藏状态，则将当前隐藏状态加入所有隐藏状态元组中
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 如果不返回字典，则返回一个元组，其中包含隐藏状态、下一个解码器缓存、所有隐藏状态、所有自注意力权重和所有交叉注意力权重
        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        # 如果返回字典，则返回一个带有过去键值对和交叉注意力的基本模型输出
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )
class Blip2QFormerModel(Blip2PreTrainedModel):
    """
    Querying Transformer (Q-Former), used in BLIP-2.
    """

    def __init__(self, config: Blip2QFormerConfig):
        # 初始化 Q-Former 模型，继承自 Blip2PreTrainedModel
        super().__init__(config)
        self.config = config

        # 初始化 LayerNorm 层和 Dropout 层
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # 初始化 Q-Former 编码器
        self.encoder = Blip2QFormerEncoder(config)

        # 调用后续初始化方法
        self.post_init()

    def get_input_embeddings(self):
        # 获取输入嵌入层的词嵌入
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        # 设置输入嵌入层的词嵌入
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        # 剪枝模型的注意力头
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def get_extended_attention_mask(
        self,
        attention_mask: torch.Tensor,
        input_shape: Tuple[int],
        device: torch.device,
        has_query: bool = False,
    def forward(
        self,
        query_embeds: torch.FloatTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,



        """
        Makes broadcastable attention and causal masks so that future and masked tokens are ignored.

        Arguments:
            attention_mask (`torch.Tensor`):
                Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
            input_shape (`Tuple[int]`):
                The shape of the input to the model.
            device (`torch.device`):
                The device of the input to the model.

        Returns:
            `torch.Tensor` The extended attention mask, with a the same dtype as `attention_mask.dtype`.
        """



        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - the model is an encoder, so make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
            extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                "Wrong shape for input_ids (shape {}) or attention_mask (shape {})".format(
                    input_shape, attention_mask.shape
                )
            )



        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask
# 使用装饰器为类添加文档字符串，描述 BLIP-2 模型生成文本和图像特征的结构
@add_start_docstrings(
    """
    BLIP-2 Model for generating text and image features. The model consists of a vision encoder, Querying Transformer
    (Q-Former) and a language model.
    """,
    BLIP_2_START_DOCSTRING,
)
# 定义 Blip2Model 类，继承自 Blip2PreTrainedModel
class Blip2Model(Blip2PreTrainedModel):
    # 指定配置类
    config_class = Blip2Config
    # 主输入名称为 "pixel_values"
    main_input_name = "pixel_values"

    # 初始化方法
    def __init__(self, config: Blip2Config):
        # 调用父类的初始化方法
        super().__init__(config)

        # 初始化视觉模型
        self.vision_model = Blip2VisionModel(config.vision_config)

        # 创建可学习的查询令牌参数
        self.query_tokens = nn.Parameter(torch.zeros(1, config.num_query_tokens, config.qformer_config.hidden_size))
        # 初始化查询 Transformer 模型
        self.qformer = Blip2QFormerModel(config.qformer_config)

        # 创建语言投影层，将查询 Transformer 的隐藏状态映射到文本配置的隐藏大小
        self.language_projection = nn.Linear(config.qformer_config.hidden_size, config.text_config.hidden_size)
        # 根据配置选择加载语言模型，支持从 config.text_config 创建自回归语言模型或序列到序列语言模型
        if config.use_decoder_only_language_model:
            language_model = AutoModelForCausalLM.from_config(config.text_config)
        else:
            language_model = AutoModelForSeq2SeqLM.from_config(config.text_config)

        # 根据基础模型更新 _tied_weights_keys
        if language_model._tied_weights_keys is not None:
            self._tied_weights_keys = [f"language_model.{k}" for k in language_model._tied_weights_keys]

        # 设置语言模型
        self.language_model = language_model

        # 初始化权重并应用最终处理
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

    # 绑定权重
    def _tie_weights(self):
        if not self.config.use_decoder_only_language_model:
            self.language_model.encoder.embed_tokens = self.language_model.shared
            self.language_model.decoder.embed_tokens = self.language_model.shared

    # 添加模型前向传播的文档字符串
    @add_start_docstrings_to_model_forward(BLIP_2_TEXT_INPUTS_DOCSTRING)
    def get_text_features(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        r"""
        Returns:
            text_outputs (`CausalLMOutputWithPast`, or `tuple(torch.FloatTensor)` if `return_dict=False`):
                The language model outputs. If `return_dict=True`, the output is a [`CausalLMOutputWithPast`] that
                contains the language model logits, the past key values and the hidden states if
                `output_hidden_states=True`.
        Examples:
        ```python
        >>> import torch
        >>> from transformers import AutoTokenizer, Blip2Model

        >>> model = Blip2Model.from_pretrained("Salesforce/blip2-opt-2.7b")

        >>> tokenizer = AutoTokenizer.from_pretrained("Salesforce/blip2-opt-2.7b")
        >>> inputs = tokenizer(["a photo of a cat"], padding=True, return_tensors="pt")
        >>> text_features = model.get_text_features(**inputs)
        ```py"""
        # 如果未提供值，则使用配置中的参数值
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 如果配置为仅使用解码器，只使用语言模型
        if self.config.use_decoder_only_language_model:
            text_outputs = self.language_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        else:
            # 获取输入嵌入
            inputs_embeds = self.language_model.get_input_embeddings()(input_ids)

            # 使用语言模型
            text_outputs = self.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                labels=labels,
            )

        # 返回文本输出
        return text_outputs

    @add_start_docstrings_to_model_forward(BLIP_2_VISION_INPUTS_DOCSTRING)
    def get_image_features(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        r"""
        返回：
            vision_outputs (`BaseModelOutputWithPooling` 或者 `torch.FloatTensor` 的元组):
                视觉模型的输出。如果 `return_dict=True`，输出是一个包含图像特征、汇聚图像特征和隐藏状态（如果`output_hidden_states=True`）的 [`BaseModelOutputWithPooling`]。
        示例:
        ```python
        >>> import torch
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, Blip2Model

        >>> model = Blip2Model.from_pretrained("Salesforce/blip2-opt-2.7b")

        >>> processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)
        >>> inputs = processor(images=image, return_tensors="pt")
        >>> image_outputs = model.get_image_features(**inputs)
        ```py"""
        # 如果未提供output_attentions参数，则使用配置中的output_attentions参数
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 如果未提供output_hidden_states参数，则使用配置中的output_hidden_states参数
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 如果未提供return_dict参数，则使用配置中的use_return_dict参数
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 使用视觉模型获取特征
        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 返回视觉模型的输出
        return vision_outputs

    @add_start_docstrings_to_model_forward(BLIP_2_INPUTS_DOCSTRING)
    def get_qformer_features(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        r"""
        Returns:
            vision_outputs (`BaseModelOutputWithPooling` or tuple of `torch.FloatTensor`):
                The vision model outputs. If `return_dict=True`, the output is a [`BaseModelOutputWithPooling`] that
                contains the image features, the pooled image features and the hidden states if
                `output_hidden_states=True`.
        Examples:
        ```python
        >>> import torch
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import Blip2Processor, Blip2Model

        >>> processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        >>> model = Blip2Model.from_pretrained("Salesforce/blip2-opt-2.7b")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)
        >>> inputs = processor(images=image, return_tensors="pt")
        >>> qformer_outputs = model.get_qformer_features(**inputs)
        ```py"""
        # 设置输出注意力权重，默认为模型配置中的设置
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 设置输出隐藏状态，默认为模型配置中的设置
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 设置是否返回字典，默认为模型配置中的设置
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 获取视觉模型的输出
        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 提取图像嵌入
        image_embeds = vision_outputs[0]

        # step 2: forward the query tokens through the QFormer, using the image embeddings for cross-attention
        # 创建图像注意力掩码，全为1，大小与图像嵌入的形状相同
        image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)

        # 扩展查询令牌以匹配图像嵌入的形状
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        # 通过 QFormer 前向传播查询令牌，使用图像嵌入进行跨注意力
        query_outputs = self.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 返回查询输出
        return query_outputs

    # 添加模型前向传播的注释文档字符串
    @add_start_docstrings_to_model_forward(BLIP_2_INPUTS_DOCSTRING)
    # 替换返回值的文档字符串
    @replace_return_docstrings(output_type=Blip2ForConditionalGenerationModelOutput, config_class=Blip2VisionConfig)
    # 定义一个前向传播函数，用于模型推断
    def forward(
        # 输入像素值的张量，通常是图像数据，类型为浮点张量
        self,
        pixel_values: torch.FloatTensor,
        # 输入序列的标识符张量，通常是输入文本的编码，类型为浮点张量
        input_ids: torch.FloatTensor,
        # 注意力掩码张量，用于指示哪些标记是填充的，哪些是真实的，类型为可选的长整型张量
        attention_mask: Optional[torch.LongTensor] = None,
        # 解码器输入序列的标识符张量，类型为可选的长整型张量
        decoder_input_ids: Optional[torch.LongTensor] = None,
        # 解码器注意力掩码张量，用于指示哪些标记是填充的，哪些是真实的，类型为可选的长整型张量
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        # 是否返回注意力权重张量的标志，类型为可选的布尔值
        output_attentions: Optional[bool] = None,
        # 是否返回隐藏状态张量的标志，类型为可选的布尔值
        output_hidden_states: Optional[bool] = None,
        # 标签张量，通常用于计算损失，类型为可选的长整型张量
        labels: Optional[torch.LongTensor] = None,
        # 是否返回字典格式的输出，类型为可选的布尔值
        return_dict: Optional[bool] = None,
# 导入所需模块或函数
@add_start_docstrings(
    """
    BLIP-2 模型用于根据图像和可选文本提示生成文本。该模型包括一个视觉编码器、查询变换器（Q-Former）和一个语言模型。

    可选择向模型传递 `input_ids`，作为文本提示，以使语言模型继续提示。否则，语言模型将从 [BOS]（序列开始）标记开始生成文本。

    <Tip>

    注意，Flan-T5 检查点无法转换为 float16。它们是使用 bfloat16 进行预训练的。

    </Tip>
    """,
    BLIP_2_START_DOCSTRING,
)
# 定义 BLIP-2 用于条件生成的类，继承自 Blip2PreTrainedModel
class Blip2ForConditionalGeneration(Blip2PreTrainedModel):
    # 配置类
    config_class = Blip2Config
    # 主要输入名称为 "pixel_values"
    main_input_name = "pixel_values"

    # 初始化方法
    def __init__(self, config: Blip2Config):
        # 调用父类的初始化方法
        super().__init__(config)

        # 创建视觉模型
        self.vision_model = Blip2VisionModel(config.vision_config)

        # 创建查询令牌
        self.query_tokens = nn.Parameter(torch.zeros(1, config.num_query_tokens, config.qformer_config.hidden_size))
        # 创建 Q-Former 模型
        self.qformer = Blip2QFormerModel(config.qformer_config)

        # 创建语言模型的投影层
        self.language_projection = nn.Linear(config.qformer_config.hidden_size, config.text_config.hidden_size)
        # 根据配置创建语言模型
        if config.use_decoder_only_language_model:
            # 如果只使用解码器语言模型，则创建自回归语言模型
            language_model = AutoModelForCausalLM.from_config(config.text_config)
        else:
            # 否则，创建序列到序列的语言模型
            language_model = AutoModelForSeq2SeqLM.from_config(config.text_config)

        # 根据使用的基础模型更新 _tied_weights_keys
        if language_model._tied_weights_keys is not None:
            self._tied_weights_keys = [f"language_model.{k}" for k in language_model._tied_weights_keys]

        # 设置语言模型
        self.language_model = language_model

        # 初始化权重并应用最终处理
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

    # 绑定权重
    def _tie_weights(self):
        if not self.config.use_decoder_only_language_model:
            self.language_model.encoder.embed_tokens = self.language_model.shared
            self.language_model.decoder.embed_tokens = self.language_model.shared
    # 定义一个私有方法，用于预处理，以使模型与 `accelerate` 兼容
    def _preprocess_accelerate(self):
        r"""
        Some pre-processing hacks to make the model `accelerate` compatible. Check
        https://github.com/huggingface/transformers/pull/21707 for more details.
        """
        # 获取模型的 HF 设备映射
        hf_device_map = self.hf_device_map

        # 如果 HF 设备映射中的条目数量大于 1，且不包含 "language_model"，同时 CUDA 设备数量大于 1
        if len(hf_device_map) > 1 and "language_model" not in hf_device_map and torch.cuda.device_count() > 1:
            # 提示用户在使用多 GPU + BLIP-2 + `accelerate` 时可能会出现意外行为
            logger.warning(
                "The `language_model` is not in the `hf_device_map` dictionary and you are running your script"
                " in a multi-GPU environment. this may lead to unexpected behavior when using `accelerate`."
                " Please pass a `device_map` that contains `language_model` to remove this warning."
                " Please refer to https://github.com/huggingface/blog/blob/main/accelerate-large-models.md for"
                " more details on creating a `device_map` for large models.",
            )

        # 如果模型具有 `_hf_hook` 属性
        if hasattr(self.language_model, "_hf_hook"):
            # 设置 `_hf_hook` 的 `io_same_device` 属性为 True，以保证兼容 `generate` 方法
            self.language_model._hf_hook.io_same_device = True  # For `generate` compatibility

    # 为前向传播方法添加文档字符串，使用 BLIP_2_INPUTS_DOCSTRING
    # 替换返回值文档字符串，输出类型为 Blip2ForConditionalGenerationModelOutput，配置类为 Blip2VisionConfig
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        input_ids: torch.FloatTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
    # 禁用梯度追踪的上下文管理器
    @torch.no_grad()
    # 定义生成方法，接受像素值、输入 ID、注意力掩码等参数，以及其他生成方法的关键字参数
    def generate(
        self,
        pixel_values: torch.FloatTensor,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        **generate_kwargs,
```  
    ) -> torch.LongTensor:
        """
        Overrides `generate` function to be able to use the model as a conditional generator.

        Args:
            pixel_values (`torch.FloatTensor` of shape (batch_size, num_channels, height, width)):
                Input images to be processed.
            input_ids (`torch.LongTensor` of shape (batch_size, sequence_length), *optional*):
                The sequence used as a prompt for the generation.
            attention_mask (`torch.LongTensor` of shape (batch_size, sequence_length), *optional*):
                Mask to avoid performing attention on padding token indices

        Returns:
            captions (list): A list of strings of length batch_size * num_captions.
        """
        if hasattr(self, "hf_device_map"):
            # 检查是否存在属性"hf_device_map"，用于加速处理
            self._preprocess_accelerate()  # 预处理以加速处理流程

        batch_size = pixel_values.shape[0]  # 获取批处理大小
        image_embeds = self.vision_model(pixel_values, return_dict=True).last_hidden_state  # 通过视觉模型处理像素值，获取图像嵌入表示
        image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)  # 创建图像注意力掩码，默认全1

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)  # 通过复制扩展查询标记，以匹配图像嵌入维度
        query_outputs = self.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
            return_dict=True,
        )  # 使用查询标记和图像嵌入作为输入，执行查询转换操作
        query_output = query_outputs.last_hidden_state  # 获取查询转换的输出

        language_model_inputs = self.language_projection(query_output)  # 使用查询转换的输出作为输入，执行语言模型投影操作
        language_attention_mask = torch.ones(
            language_model_inputs.size()[:-1], dtype=torch.long, device=language_model_inputs.device
        )  # 创建语言注意力掩码，默认全1
        if input_ids is None:
            input_ids = (
                torch.LongTensor([[self.config.text_config.bos_token_id]])  # 如果未提供输入标记，则使用特殊的起始标记
                .repeat(batch_size, 1)  # 在批次维度上复制起始标记
                .to(image_embeds.device)  # 将起始标记移动到与图像嵌入相同的设备上
            )
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)  # 如果未提供注意力掩码，则创建一个与输入标记相同形状的全1张量
        attention_mask = torch.cat([language_attention_mask, attention_mask.to(language_attention_mask.device)], dim=1)  # 将语言注意力掩码与输入标记的注意力掩码连接在一起

        # 将查询嵌入与提示嵌入连接起来
        inputs_embeds = self.get_input_embeddings()(input_ids)  # 获取输入标记的嵌入表示
        inputs_embeds = torch.cat([language_model_inputs, inputs_embeds.to(language_model_inputs.device)], dim=1)  # 将语言模型输入嵌入与输入标记嵌入连接在一起

        outputs = self.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **generate_kwargs,
        )  # 使用语言模型生成序列

        return outputs  # 返回生成的序列
```