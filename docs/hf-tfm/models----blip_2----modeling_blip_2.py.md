# `.\models\blip_2\modeling_blip_2.py`

```
# 设置文件编码为 UTF-8
# 版权声明，指出版权归 Salesforce 作者和 HuggingFace 团队所有
#
# 根据 Apache 许可证版本 2.0 进行许可
# 除非符合许可证规定，否则不得使用本文件
# 可以在以下网址获取许可证的副本：http://www.apache.org/licenses/LICENSE-2.0
#
# 如果适用法律要求或书面同意，本软件是基于"按原样提供"的基础分发的
# 没有任何明示或暗示的担保或条件
# 请参阅许可证以了解详细的法律条款
""" PyTorch BLIP-2 model."""

import math  # 导入数学函数库
from dataclasses import dataclass  # 导入用于数据类的 dataclass 装饰器
from typing import Any, Optional, Tuple, Union  # 导入类型提示相关的模块

import torch  # 导入 PyTorch 深度学习框架
import torch.utils.checkpoint  # 导入用于检查点的实用工具
from torch import nn  # 从 PyTorch 导入神经网络模块
from torch.nn import CrossEntropyLoss  # 导入交叉熵损失函数

# 从相关模块导入各种类和函数
from ...activations import ACT2FN
from ...modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPooling,
    BaseModelOutputWithPoolingAndCrossAttentions,
)
from ...modeling_utils import PreTrainedModel  # 导入预训练模型的实用函数
from ...pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer  # 导入 PyTorch 实用工具函数
from ...utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from ..auto import AutoModelForCausalLM, AutoModelForSeq2SeqLM  # 从自动化模块导入自动模型类
from .configuration_blip_2 import Blip2Config, Blip2QFormerConfig, Blip2VisionConfig  # 导入 BLIP-2 配置类

logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器

_CHECKPOINT_FOR_DOC = "Salesforce/blip2-opt-2.7b"  # 预训练模型的检查点名称用于文档

# BLIP-2 模型的预训练模型存档列表
BLIP_2_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "Salesforce/blip2-opt-2.7b",
    # 可以在以下网址查看所有 BLIP-2 模型：https://huggingface.co/models?filter=blip
]


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

    loss: Optional[Tuple[torch.FloatTensor]] = None  # 语言模型的损失，当提供了标签时返回，为一个张量元组
    logits: Optional[Tuple[torch.FloatTensor]] = None  # 语言模型头部的预测分数，形状为(batch_size, sequence_length, config.vocab_size)的张量
    vision_outputs: Optional[torch.FloatTensor] = None  # 视觉编码器的输出
    qformer_outputs: Optional[Tuple[torch.FloatTensor]] = None  # Q-Former (查询变压器)的输出，包含池化和交叉注意力
    language_model_outputs: Optional[Tuple[torch.FloatTensor]] = None  # 语言模型的输出，包含过去的 CausalLMOutputWithPast 或者 Seq2SeqLMOutput
    # 将对象转换为元组的方法，返回对象的各个属性组成的元组
    def to_tuple(self) -> Tuple[Any]:
        # 使用生成器表达式生成元组
        return tuple(
            # 如果键不是特定的输出属性，则直接取对象的属性值
            self[k]
            # 否则，调用对象的相应属性的to_tuple方法来获取其元组表示
            if k not in ["vision_outputs", "qformer_outputs", "language_model_outputs"]
            else getattr(self, k).to_tuple()
            # 对于对象的所有键进行遍历
            for k in self.keys()
        )
# 从transformers.models.blip.modeling_blip.BlipVisionEmbeddings复制并修改为Blip2VisionEmbeddings类
class Blip2VisionEmbeddings(nn.Module):
    def __init__(self, config: Blip2VisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size  # 设置嵌入维度为配置中的隐藏大小
        self.image_size = config.image_size  # 图像尺寸来自配置
        self.patch_size = config.patch_size  # 补丁尺寸来自配置

        self.class_embedding = nn.Parameter(torch.randn(1, 1, self.embed_dim))  # 类别嵌入参数，形状为(1, 1, embed_dim)

        self.patch_embedding = nn.Conv2d(
            in_channels=3, out_channels=self.embed_dim, kernel_size=self.patch_size, stride=self.patch_size
        )  # 使用卷积层作为补丁嵌入器，输入通道为3，输出通道为embed_dim，卷积核大小和步长为patch_size

        self.num_patches = (self.image_size // self.patch_size) ** 2  # 计算图像中的补丁数量
        self.num_positions = self.num_patches + 1  # 位置嵌入数量为补丁数量加1

        self.position_embedding = nn.Parameter(torch.randn(1, self.num_positions, self.embed_dim))
        # 位置嵌入参数，形状为(1, num_positions, embed_dim)

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        batch_size = pixel_values.shape[0]  # 获取批次大小
        target_dtype = self.patch_embedding.weight.dtype  # 目标数据类型与补丁嵌入的权重数据类型一致
        patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype))  # 得到补丁嵌入，形状为[*, embed_dim, grid, grid]
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)  # 展平并转置补丁嵌入的维度

        class_embeds = self.class_embedding.expand(batch_size, 1, -1).to(target_dtype)  # 扩展类别嵌入以适应批次大小
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)  # 拼接类别嵌入和补丁嵌入
        embeddings = embeddings + self.position_embedding[:, : embeddings.size(1), :].to(target_dtype)
        # 加上位置嵌入（仅限于嵌入的数量），使用与目标数据类型一致的参数
        return embeddings


class Blip2Attention(nn.Module):
    """来自'Attention Is All You Need'论文的多头注意力模块"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size  # 设置嵌入维度为配置中的隐藏大小
        self.num_heads = config.num_attention_heads  # 设置注意力头的数量
        self.head_dim = self.embed_dim // self.num_heads  # 每个注意力头的维度
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim必须能够被num_heads整除（得到`embed_dim`:{self.embed_dim}和`num_heads`:{self.num_heads}）."
            )
        self.scale = self.head_dim**-0.5  # 缩放因子，根据头维度设置
        self.dropout = nn.Dropout(config.attention_dropout)  # 注意力模型的dropout率

        # 对比于CLIP，这里做了一个小调整，不使用偏置
        self.qkv = nn.Linear(self.embed_dim, 3 * self.embed_dim, bias=False)
        # QKV线性变换层，输出维度为3 * embed_dim，没有偏置

        if config.qkv_bias:
            q_bias = nn.Parameter(torch.zeros(self.embed_dim))
            v_bias = nn.Parameter(torch.zeros(self.embed_dim))
        else:
            q_bias = None
            v_bias = None

        if q_bias is not None:
            qkv_bias = torch.cat((q_bias, torch.zeros_like(v_bias, requires_grad=False), v_bias))
            self.qkv.bias = nn.Parameter(qkv_bias)  # 设置QKV的偏置参数

        self.projection = nn.Linear(self.embed_dim, self.embed_dim)
        # 线性投影层，用于最终的嵌入维度映射
    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        # 获取输入张量的维度信息
        bsz, tgt_len, embed_dim = hidden_states.size()

        # 使用 self.qkv 对象对隐藏状态进行线性变换，生成混合的查询、键、值张量
        mixed_qkv = self.qkv(hidden_states)

        # 将混合的查询、键、值张量重塑为合适的形状，并重新排列维度顺序
        mixed_qkv = mixed_qkv.reshape(bsz, tgt_len, 3, self.num_heads, embed_dim // self.num_heads).permute(
            2, 0, 3, 1, 4
        )
        query_states, key_states, value_states = mixed_qkv[0], mixed_qkv[1], mixed_qkv[2]

        # 计算注意力分数，即查询和键的点积
        attention_scores = torch.matmul(query_states, key_states.transpose(-1, -2))

        # 缩放注意力分数
        attention_scores = attention_scores * self.scale

        # 对注意力分数进行归一化，转换为注意力概率
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # 使用 dropout 对注意力概率进行随机置零，防止过拟合
        attention_probs = self.dropout(attention_probs)

        # 如果提供了 head_mask，则对注意力概率进行掩码操作
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        # 计算加权后的值张量，再重新排列维度顺序
        context_layer = torch.matmul(attention_probs, value_states).permute(0, 2, 1, 3)

        # 重新调整上下文层的形状，以匹配预期的输出维度
        new_context_layer_shape = context_layer.size()[:-2] + (self.embed_dim,)
        context_layer = context_layer.reshape(new_context_layer_shape)

        # 对上下文层进行投影，生成最终的输出
        output = self.projection(context_layer)

        # 根据是否需要输出注意力概率，选择性返回结果
        outputs = (output, attention_probs) if output_attentions else (output, None)

        return outputs
# Copied from transformers.models.blip.modeling_blip.BlipMLP
# 定义一个名为 Blip2MLP 的类，继承自 nn.Module
class Blip2MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 初始化函数，保存配置信息
        self.config = config
        # 选择激活函数，根据配置中的隐藏层激活函数选择对应的函数
        self.activation_fn = ACT2FN[config.hidden_act]
        # 第一个全连接层，输入为隐藏大小，输出为中间大小
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        # 第二个全连接层，输入为中间大小，输出为隐藏大小
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    # 前向传播函数，接受隐藏状态张量并返回处理后的张量
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 第一层全连接操作
        hidden_states = self.fc1(hidden_states)
        # 应用激活函数
        hidden_states = self.activation_fn(hidden_states)
        # 第二层全连接操作
        hidden_states = self.fc2(hidden_states)
        # 返回处理后的隐藏状态张量
        return hidden_states


# Copied from transformers.models.blip.modeling_blip.BlipEncoderLayer with Blip->Blip2
# 定义一个名为 Blip2EncoderLayer 的类，继承自 nn.Module
class Blip2EncoderLayer(nn.Module):
    def __init__(self, config: Blip2Config):
        super().__init__()
        # 设置嵌入维度为隐藏大小
        self.embed_dim = config.hidden_size
        # 定义自注意力层，使用 Blip2Attention 类
        self.self_attn = Blip2Attention(config)
        # 第一个层归一化层，输入维度为嵌入维度
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        # 定义 MLP 层，使用 Blip2MLP 类
        self.mlp = Blip2MLP(config)
        # 第二个层归一化层，输入维度为嵌入维度
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    # 前向传播函数，接受隐藏状态、注意力掩码及是否输出注意力权重，并返回元组
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
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers.
        """
        # 保存残差连接
        residual = hidden_states

        # 应用第一个层归一化层
        hidden_states = self.layer_norm1(hidden_states)
        # 调用自注意力层的前向传播
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            head_mask=attention_mask,
            output_attentions=output_attentions,
        )
        # 添加残差连接
        hidden_states = hidden_states + residual
        # 保存更新后的残差连接
        residual = hidden_states
        # 应用第二个层归一化层
        hidden_states = self.layer_norm2(hidden_states)
        # 调用 MLP 层的前向传播
        hidden_states = self.mlp(hidden_states)
        # 添加残差连接
        hidden_states = hidden_states + residual

        # 准备输出，仅包含隐藏状态
        outputs = (hidden_states,)

        # 如果需要输出注意力权重，则添加到输出中
        if output_attentions:
            outputs += (attn_weights,)

        # 返回输出元组
        return outputs


# 定义一个名为 Blip2PreTrainedModel 的抽象类，继承自 PreTrainedModel
class Blip2PreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 指定配置类为 Blip2Config
    config_class = Blip2Config
    # 指定基础模型前缀为 "blip"
    base_model_prefix = "blip"
    # 支持梯度检查点
    supports_gradient_checkpointing = True
    # 不需要拆分的模块列表
    _no_split_modules = ["Blip2Attention", "T5Block", "OPTDecoderLayer"]
    # 跳过键的设备放置
    _skip_keys_device_placement = "past_key_values"
    # 初始化要保持为单精度浮点数的模块列表，包括名为"wo"的模块
    _keep_in_fp32_modules = ["wo"]
    
    # 初始化模块的权重
    def _init_weights(self, module):
        """Initialize the weights"""
        # 获取初始化因子
        factor = self.config.initializer_range
        
        # 如果模块是卷积层、嵌入层或线性层，对其权重进行正态分布初始化，偏置初始化为零
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Embedding) or isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=factor)
            if hasattr(module, "bias") and module.bias is not None:
                module.bias.data.zero_()
        
        # 如果模块是特定类型的自定义嵌入层Blip2VisionEmbeddings
        elif isinstance(module, Blip2VisionEmbeddings):
            # 如果配置中有视觉配置，则使用视觉配置中的初始化因子
            if hasattr(self.config, "vision_config"):
                factor = self.config.vision_config.initializer_range
            # 对位置嵌入和类别嵌入进行截断正态分布初始化
            nn.init.trunc_normal_(module.position_embedding, mean=0.0, std=factor)
            nn.init.trunc_normal_(module.class_embedding, mean=0.0, std=factor)
        
        # 如果模块是归一化层，初始化偏置为零，权重为1.0
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        
        # 如果模块是线性层且具有偏置项，初始化偏置为零
        elif isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
# BLIP_2_START_DOCSTRING 字符串变量，包含关于模型继承和用法的文档信息
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

# BLIP_2_VISION_INPUTS_DOCSTRING 字符串变量，包含关于视觉输入参数的文档信息
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

# BLIP_2_TEXT_INPUTS_DOCSTRING 字符串变量，尚未提供具体的文档信息，待补充
BLIP_2_TEXT_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            # 输入序列 token 在词汇表中的索引。默认情况下会忽略填充部分。可以使用 `AutoTokenizer` 获取索引。详见 `PreTrainedTokenizer.encode` 和 `PreTrainedTokenizer.__call__`。
            [什么是输入 ID？](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            # 避免在填充 token 索引上执行注意力操作的掩码。掩码取值为 `[0, 1]`：
            - 1 表示**未被掩蔽**的 token，
            - 0 表示**被掩蔽**的 token。
            [什么是注意力掩码？](../glossary#attention-mask)
        decoder_input_ids (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*):
            # 解码器输入序列 token 在词汇表中的索引。
            # 可以使用 `AutoTokenizer` 获取索引。详见 `PreTrainedTokenizer.encode` 和 `PreTrainedTokenizer.__call__`。
            [什么是解码器输入 ID？](../glossary#decoder-input-ids)

            T5 使用 `pad_token_id` 作为 `decoder_input_ids` 生成的起始 token。如果使用了 `past_key_values`，可以选择仅输入最后的 `decoder_input_ids`（参见 `past_key_values`）。

            欲了解更多有关预训练的 `decoder_input_ids` 准备工作，请查看 [T5 Training](./t5#training)。
        decoder_attention_mask (`torch.BoolTensor` of shape `(batch_size, target_sequence_length)`, *optional*):
            # 默认行为：生成一个忽略 `decoder_input_ids` 中填充 token 的张量。默认情况下也会使用因果掩码。
        output_attentions (`bool`, *optional*):
            # 是否返回所有注意力层的注意力张量。详见返回张量中的 `attentions` 获取更多细节。
        output_hidden_states (`bool`, *optional*):
            # 是否返回所有层的隐藏状态。详见返回张量中的 `hidden_states` 获取更多细节。
        return_dict (`bool`, *optional*):
            # 是否返回一个 `~utils.ModelOutput` 而非简单的元组。
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
    # 初始化函数，用于创建一个新的 Blip2Encoder 对象
    def __init__(self, config: Blip2Config):
        # 调用父类的初始化方法
        super().__init__()
        # 将传入的配置对象保存到实例变量中
        self.config = config
        # 创建一个包含多个 Blip2EncoderLayer 对象的列表，列表长度为 config.num_hidden_layers
        self.layers = nn.ModuleList([Blip2EncoderLayer(config) for _ in range(config.num_hidden_layers)])
        # 设置梯度检查点标记为 False
        self.gradient_checkpointing = False

    # 前向传播函数，用于定义模型的前向计算过程
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
        # Determine if output_attentions should be overridden by input or default configuration
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # Determine if output_hidden_states should be overridden by input or default configuration
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # Determine if return_dict should be overridden by input or default configuration
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Initialize empty tuple for encoder_states if output_hidden_states is False
        encoder_states = () if output_hidden_states else None
        # Initialize empty tuple for all_attentions if output_attentions is False
        all_attentions = () if output_attentions else None

        # Set initial hidden_states to inputs_embeds
        hidden_states = inputs_embeds
        # Iterate through each encoder layer
        for idx, encoder_layer in enumerate(self.layers):
            # Append current hidden_states to encoder_states if output_hidden_states is True
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            # Perform gradient checkpointing if enabled during training
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    encoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    output_attentions,
                )
            else:
                # Execute encoder_layer with current hidden_states and attention_mask
                layer_outputs = encoder_layer(
                    hidden_states,
                    attention_mask,
                    output_attentions=output_attentions,
                )

            # Update hidden_states to the first output of encoder_layer
            hidden_states = layer_outputs[0]

            # Append attentions of current layer to all_attentions if output_attentions is True
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # Append final hidden_states to encoder_states if output_hidden_states is True
        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        # Return either a tuple or a BaseModelOutput based on return_dict flag
        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )
# 从 transformers.models.blip.modeling_blip.BlipVisionModel 复制而来，将 Blip -> Blip2, BLIP -> BLIP_2
class Blip2VisionModel(Blip2PreTrainedModel):
    # 主要输入名称为 "pixel_values"
    main_input_name = "pixel_values"
    # 使用 Blip2VisionConfig 作为配置类
    config_class = Blip2VisionConfig

    def __init__(self, config: Blip2VisionConfig):
        super().__init__(config)
        self.config = config
        embed_dim = config.hidden_size

        # 初始化嵌入层
        self.embeddings = Blip2VisionEmbeddings(config)
        # 初始化编码器
        self.encoder = Blip2Encoder(config)
        # 初始化 LayerNorm 层
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

        # 执行额外的初始化操作
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
        r"""
        前向传播函数

        Returns:
            根据 return_dict 的值返回不同的输出格式
        """
        # 如果未指定 pixel_values，抛出 ValueError
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # 将像素值传入嵌入层
        hidden_states = self.embeddings(pixel_values)

        # 使用编码器进行编码
        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取最后一个隐藏状态并进行 LayerNorm 处理
        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.post_layernorm(last_hidden_state)

        # 从最后隐藏状态中提取池化输出，并再次进行 LayerNorm 处理
        pooled_output = last_hidden_state[:, 0, :]
        pooled_output = self.post_layernorm(pooled_output)

        # 如果 return_dict 为 False，则返回元组形式的输出
        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        # 如果 return_dict 为 True，则返回 BaseModelOutputWithPooling 类型的输出
        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

    def get_input_embeddings(self):
        # 返回嵌入层对象
        return self.embeddings


class Blip2QFormerMultiHeadAttention(nn.Module):
    # 初始化函数，接受配置对象和是否跨注意力的标志
    def __init__(self, config, is_cross_attention=False):
        # 调用父类初始化函数
        super().__init__()
        # 将配置对象保存到实例变量中
        self.config = config
        # 检查隐藏大小是否能被注意力头数整除，如果不能且没有嵌入大小属性，则抛出错误
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention heads (%d)"
                % (config.hidden_size, config.num_attention_heads)
            )

        # 保存注意力头的数量和每个头的大小
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # 创建查询、键和值的线性变换层
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        if is_cross_attention:
            # 如果是跨注意力，则使用编码器隐藏大小进行线性变换
            self.key = nn.Linear(config.encoder_hidden_size, self.all_head_size)
            self.value = nn.Linear(config.encoder_hidden_size, self.all_head_size)
        else:
            # 否则使用隐藏大小进行线性变换
            self.key = nn.Linear(config.hidden_size, self.all_head_size)
            self.value = nn.Linear(config.hidden_size, self.all_head_size)

        # 创建一个丢弃层，用于注意力概率的丢弃
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        
        # 保存位置嵌入类型，如果是相对键或相对键查询，则创建距离嵌入
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)
        
        # 默认不保存注意力
        self.save_attention = False

    # 保存注意力梯度
    def save_attn_gradients(self, attn_gradients):
        self.attn_gradients = attn_gradients

    # 获取保存的注意力梯度
    def get_attn_gradients(self):
        return self.attn_gradients

    # 保存注意力映射
    def save_attention_map(self, attention_map):
        self.attention_map = attention_map

    # 获取保存的注意力映射
    def get_attention_map(self):
        return self.attention_map

    # 对输入张量进行维度转换以匹配多头注意力的形状
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    # 前向传播函数，接收隐藏状态和各种掩码、键值对等参数
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
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)  # 创建线性层，用于变换隐藏状态的维度
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)  # 创建层归一化层，用于归一化隐藏状态
        self.dropout = nn.Dropout(config.hidden_dropout_prob)  # 创建dropout层，用于随机丢弃部分神经元输出

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)  # 线性变换
        hidden_states = self.dropout(hidden_states)  # dropout处理
        hidden_states = self.LayerNorm(hidden_states + input_tensor)  # 添加残差连接并归一化
        return hidden_states


# 从transformers.models.bert.modeling_bert.BertSelfOutput复制并修改为Blip2QFormerAttention类
class Blip2QFormerAttention(nn.Module):
    def __init__(self, config, is_cross_attention=False):
        super().__init__()
        self.attention = Blip2QFormerMultiHeadAttention(config, is_cross_attention)  # 初始化多头注意力机制
        self.output = Blip2QFormerSelfOutput(config)  # 初始化自我输出层
        self.pruned_heads = set()  # 初始化被修剪的注意力头集合

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.attention.num_attention_heads, self.attention.attention_head_size, self.pruned_heads
        )

        # 修剪线性层
        self.attention.query = prune_linear_layer(self.attention.query, index)
        self.attention.key = prune_linear_layer(self.attention.key, index)
        self.attention.value = prune_linear_layer(self.attention.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # 更新超参数并存储修剪的注意力头
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
        self_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        attention_output = self.output(self_outputs[0], hidden_states)  # 调用自我输出层处理注意力输出
        outputs = (attention_output,) + self_outputs[1:]  # 如果需要输出注意力权重，则添加到输出中
        return outputs


# 从transformers.models.bert.modeling_bert.BertIntermediate复制并修改为Blip2QFormerIntermediate类
class Blip2QFormerIntermediate(nn.Module):
    # 初始化方法，用于初始化对象
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()
        # 创建一个全连接层，输入大小为 config.hidden_size，输出大小为 config.intermediate_size
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        # 判断 config.hidden_act 是否为字符串类型
        if isinstance(config.hidden_act, str):
            # 如果是字符串，则从预定义的映射 ACT2FN 中获取对应的激活函数
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            # 否则直接使用 config.hidden_act 作为激活函数
            self.intermediate_act_fn = config.hidden_act

    # 前向传播方法，定义了数据流向
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 使用全连接层进行线性变换
        hidden_states = self.dense(hidden_states)
        # 使用激活函数对线性变换的结果进行非线性变换
        hidden_states = self.intermediate_act_fn(hidden_states)
        # 返回变换后的结果
        return hidden_states
# 从 transformers.models.bert.modeling_bert.BertOutput 复制并修改为 Blip2QFormerOutput 类
class Blip2QFormerOutput(nn.Module):
    # 初始化方法，接受一个配置对象并设置模型的基本结构
    def __init__(self, config):
        super().__init__()
        # 创建一个全连接层，将 intermediate_size 转换为 hidden_size
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        # 创建 LayerNorm 层，对 hidden_size 进行归一化，eps 是归一化层的小数值
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 创建 Dropout 层，以 hidden_dropout_prob 的概率丢弃神经元
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    # 前向传播方法，接受隐藏状态和输入张量，返回处理后的张量
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 全连接层处理隐藏状态
        hidden_states = self.dense(hidden_states)
        # 使用 Dropout 层丢弃部分神经元
        hidden_states = self.dropout(hidden_states)
        # 将归一化层应用于加上输入张量的结果
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        # 返回处理后的隐藏状态张量
        return hidden_states


# Blip2QFormerLayer 类，继承自 nn.Module
class Blip2QFormerLayer(nn.Module):
    # 初始化方法，接受配置对象和层索引
    def __init__(self, config, layer_idx):
        super().__init__()
        # 设置前馈块的分块大小
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        # 设置序列长度的维度
        self.seq_len_dim = 1
        # 创建 Blip2QFormerAttention 对象
        self.attention = Blip2QFormerAttention(config)
        
        # 记录当前层的索引
        self.layer_idx = layer_idx

        # 根据 config 中的跨注意力频率设置是否包含跨注意力机制
        if layer_idx % config.cross_attention_frequency == 0:
            # 如果层索引可以被跨注意力频率整除，创建跨注意力对象
            self.crossattention = Blip2QFormerAttention(config, is_cross_attention=True)
            self.has_cross_attention = True
        else:
            # 否则不包含跨注意力机制
            self.has_cross_attention = False

        # 创建中间查询层
        self.intermediate_query = Blip2QFormerIntermediate(config)
        # 创建输出查询层
        self.output_query = Blip2QFormerOutput(config)

    # 前向传播方法，接受多个输入参数并返回处理后的隐藏状态张量
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
        ):
        ):
        # 如果过去的键/值元组不为空，则将自注意力的过去键/值截取到位置1和2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        # 对自注意力层进行前向传播
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        # 获取自注意力的输出
        attention_output = self_attention_outputs[0]
        # 输出不包括注意力
        outputs = self_attention_outputs[1:-1]

        # 获取当前注意力的键/值
        present_key_value = self_attention_outputs[-1]

        if query_length > 0:
            # 截取注意力输出的查询长度部分
            query_attention_output = attention_output[:, :query_length, :]

            if self.has_cross_attention:
                if encoder_hidden_states is None:
                    raise ValueError("encoder_hidden_states must be given for cross-attention layers")
                # 进行交叉注意力的前向传播
                cross_attention_outputs = self.crossattention(
                    query_attention_output,
                    attention_mask,
                    head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    output_attentions=output_attentions,
                )
                # 获取交叉注意力的输出
				query_attention_output = cross_attention_outputs[0]
                # 如果输出注意力权重，则添加交叉注意力
                outputs = outputs + cross_attention_outputs[1:-1]

            # 对前向传播应用分块处理
            layer_output = apply_chunking_to_forward(
                self.feed_forward_chunk_query,
                self.chunk_size_feed_forward,
                self.seq_len_dim,
                query_attention_output,
            )

            if attention_output.shape[1] > query_length:
                # 对注意力输出的查询长度之后的部分进行分块处理
                layer_output_text = apply_chunking_to_forward(
                    self.feed_forward_chunk,
                    self.chunk_size_feed_forward,
                    self.seq_len_dim,
                    attention_output[:, query_length:, :],
                )
                # 将处理后的两部分拼接起来
                layer_output = torch.cat([layer_output, layer_output_text], dim=1)
        else:
            # 对注意力输出进行分块处理
            layer_output = apply_chunking_to_forward(
                self.feed_forward_chunk,
                self.chunk_size_feed_forward,
                self.seq_len_dim,
                attention_output,
            )
        # 更新输出
        outputs = (layer_output,) + outputs

        outputs = outputs + (present_key_value,)

        return outputs

    # 定义前馈层的分块处理函数
    def feed_forward_chunk(self, attention_output):
        # 对注意力输出进行前馈处理
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output

    # 定义交叉注意力的前馈层的分块处理函数
    def feed_forward_chunk_query(self, attention_output):
        # 对注意力输出进行前馈处理
        intermediate_output = self.intermediate_query(attention_output)
        layer_output = self.output_query(intermediate_output, attention_output)
        return layer_output
# 定义一个名为 Blip2QFormerEncoder 的类，继承自 nn.Module
class Blip2QFormerEncoder(nn.Module):
    # 初始化方法，接受一个参数 config
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()
        # 将传入的 config 参数保存到实例变量 self.config 中
        self.config = config
        # 创建一个 nn.ModuleList，其中包含多个 Blip2QFormerLayer 对象
        # 每个 Blip2QFormerLayer 对象通过 config 和 layer_idx 创建，layer_idx 在 0 到 config.num_hidden_layers 之间循环
        self.layer = nn.ModuleList(
            [Blip2QFormerLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        # 设置梯度检查点为 False
        self.gradient_checkpointing = False

    # 定义前向传播方法
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
        ):
        # 如果输出隐藏状态为真，则初始化一个空元组用于存储所有隐藏状态
        all_hidden_states = () if output_hidden_states else None
        # 如果输出注意力权重为真，则初始化一个空元组用于存储所有自注意力权重
        all_self_attentions = () if output_attentions else None
        # 如果输出交叉注意力权重为真，则初始化一个空元组用于存储所有交叉注意力权重
        all_cross_attentions = () if output_attentions else None

        # 如果使用缓存，则初始化一个空元组用于存储下一个解码器缓存
        next_decoder_cache = () if use_cache else None

        # 循环遍历所有的层
        for i in range(self.config.num_hidden_layers):
            # 获取当前层的模块
            layer_module = self.layer[i]
            # 如果输出隐藏状态为真，则将当前隐藏状态加入到所有隐藏状态中
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 获取当前层的头部掩码，如果没有则为 None
            layer_head_mask = head_mask[i] if head_mask is not None else None
            # 获取过去的键值对，如果没有则为 None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            # 如果启用了梯度检查点且处于训练模式
            if getattr(self.config, "gradient_checkpointing", False) and self.training:
                # 如果使用了缓存，则发出警告并设置 use_cache=False
                if use_cache:
                    logger.warning(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False
                # 调用梯度检查点函数，用于在计算中断时进行检查点
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                # 否则，正常调用当前层的模块进行前向传播
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

            # 更新当前隐藏状态为当前层的输出的第一个元素
            hidden_states = layer_outputs[0]
            # 如果使用缓存，则将当前层的输出的最后一个元素加入到下一个解码器缓存中
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            # 如果输出注意力权重为真，则将当前层的输出的第二个元素加入到所有自注意力权重中
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                # 如果当前层有交叉注意力，将其输出的第三个元素加入到所有交叉注意力权重中
                if layer_module.has_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        # 如果输出隐藏状态为真，则将最终的隐藏状态加入到所有隐藏状态中
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 如果不返回字典形式的结果，则以元组形式返回多个非空值
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
        # 否则，以特定格式返回包含各种输出的对象
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )
    """
    Querying Transformer (Q-Former), used in BLIP-2.
    """

    def __init__(self, config: Blip2QFormerConfig):
        # 调用父类的初始化方法，传入配置对象
        super().__init__(config)
        # 将配置对象保存在实例中
        self.config = config

        # 初始化 LayerNorm 层，用于规范化隐藏状态向量
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 初始化 Dropout 层，用于在训练时随机失活一部分神经元
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # 初始化编码器，使用 BLIP-2 Q-Former 的编码器
        self.encoder = Blip2QFormerEncoder(config)

        # 执行额外的初始化步骤
        self.post_init()

    def get_input_embeddings(self):
        # 返回输入嵌入层的单词嵌入
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        # 设置输入嵌入层的单词嵌入为给定的值
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        # 遍历需要修剪的层和头部
        for layer, heads in heads_to_prune.items():
            # 调用编码器中的注意力机制的修剪方法，修剪指定层中的指定注意力头
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
            device (`torch.device`):
                The device of the input to the model.

        Returns:
            `torch.Tensor` The extended attention mask, with the same dtype as `attention_mask.dtype`.
        """
        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            # Extend the attention mask to have an additional dimension for heads
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            # Extend the attention mask for an encoder model to include heads and both sequence lengths
            extended_attention_mask = attention_mask[:, None, None, :]
        else:
            # Raise an error if the shape of attention_mask doesn't match expected shapes
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
        
        # Convert extended_attention_mask to the same dtype as self.dtype (for FP16 compatibility)
        extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)
        # Apply the masking: set positions to -10000.0 where extended_attention_mask is 0.0
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask
# 使用装饰器添加文档字符串到类 Blip2Model，描述了该模型的组成部分和功能
@add_start_docstrings(
    """
    BLIP-2 Model for generating text and image features. The model consists of a vision encoder, Querying Transformer
    (Q-Former) and a language model.
    """,
    BLIP_2_START_DOCSTRING,
)
# 定义 Blip2Model 类，继承自 Blip2PreTrainedModel 类
class Blip2Model(Blip2PreTrainedModel):
    
    # 引用 Blip2Config 作为配置类
    config_class = Blip2Config
    # 主输入名称为 "pixel_values"
    main_input_name = "pixel_values"

    # 初始化方法，接受一个 Blip2Config 类型的参数 config
    def __init__(self, config: Blip2Config):
        # 调用父类的初始化方法
        super().__init__(config)

        # 使用 Blip2VisionModel 类根据 vision_config 初始化 vision_model
        self.vision_model = Blip2VisionModel(config.vision_config)

        # 创建一个可训练参数，形状为 (1, num_query_tokens, hidden_size)，用于查询 Transformer 模型
        self.query_tokens = nn.Parameter(torch.zeros(1, config.num_query_tokens, config.qformer_config.hidden_size))
        # 使用 Blip2QFormerModel 类根据 qformer_config 初始化 qformer
        self.qformer = Blip2QFormerModel(config.qformer_config)

        # 使用 nn.Linear 创建一个线性层，将 qformer 的隐藏大小映射到 text_config 的隐藏大小
        self.language_projection = nn.Linear(config.qformer_config.hidden_size, config.text_config.hidden_size)
        
        # 根据配置选择使用语言模型，可能是 AutoModelForCausalLM 或 AutoModelForSeq2SeqLM
        if config.use_decoder_only_language_model:
            language_model = AutoModelForCausalLM.from_config(config.text_config)
        else:
            language_model = AutoModelForSeq2SeqLM.from_config(config.text_config)

        # 如果语言模型有 tied weights keys，则更新 _tied_weights_keys 属性
        if language_model._tied_weights_keys is not None:
            self._tied_weights_keys = [f"language_model.{k}" for k in language_model._tied_weights_keys]

        # 将初始化后的语言模型赋值给 self.language_model
        self.language_model = language_model

        # 初始化权重并进行最终处理
        self.post_init()

    # 返回语言模型的输入嵌入层
    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    # 设置语言模型的输入嵌入层
    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    # 设置语言模型的输出嵌入层
    def set_output_embeddings(self, new_embeddings):
        self.language_model.set_output_embeddings(new_embeddings)

    # 获取语言模型的输出嵌入层
    def get_output_embeddings(self) -> nn.Module:
        return self.language_model.get_output_embeddings()

    # 获取语言模型的编码器
    def get_encoder(self):
        return self.language_model.get_encoder()

    # 获取语言模型的解码器
    def get_decoder(self):
        return self.language_model.get_decoder()

    # 如果未使用仅解码器语言模型，则将语言模型的编码器和解码器的 embed_tokens 与 shared 属性绑定
    def _tie_weights(self):
        if not self.config.use_decoder_only_language_model:
            self.language_model.encoder.embed_tokens = self.language_model.shared
            self.language_model.decoder.embed_tokens = self.language_model.shared

    # 使用装饰器添加文档字符串到 get_text_features 方法，描述其输入参数和功能
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
    @add_start_docstrings_to_model_forward(BLIP_2_VISION_INPUTS_DOCSTRING)
    # 将 BLIP_2_VISION_INPUTS_DOCSTRING 添加到模型前向方法的文档字符串中
    def get_image_features(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
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
        ```"""
        # 如果未指定 output_attentions，则使用 self.config.output_attentions
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 如果未指定 output_hidden_states，则使用 self.config.output_hidden_states
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 如果未指定 return_dict，则使用 self.config.use_return_dict
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 如果配置为仅使用解码器语言模型
        if self.config.use_decoder_only_language_model:
            # 调用语言模型的前向方法，传入参数
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

            # 调用语言模型的前向方法，传入参数
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

        # 返回语言模型的输出
        return text_outputs
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
        >>> from transformers import AutoProcessor, Blip2Model

        >>> model = Blip2Model.from_pretrained("Salesforce/blip2-opt-2.7b")

        >>> processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)
        >>> inputs = processor(images=image, return_tensors="pt")
        >>> image_outputs = model.get_image_features(**inputs)
        ```"""
        # 如果没有明确指定，则从配置中获取是否返回注意力权重
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 如果没有明确指定，则从配置中获取是否返回隐藏状态
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 如果没有明确指定，则从配置中获取是否使用返回字典
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用 vision_model 方法，传入像素值、注意力权重、隐藏状态和返回字典参数
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
        ```"""

        # Initialize optional variables or use values from model configuration if not provided
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Forward pass through the vision model to obtain image embeddings
        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # Extract image embeddings from the vision model outputs
        image_embeds = vision_outputs[0]

        # Step 2: Forward the query tokens through the QFormer with cross-attention to image embeddings

        # Create attention mask for the image embeddings
        image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)

        # Expand the query tokens to match the batch size of image embeddings
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)

        # Forward pass through the QFormer model
        query_outputs = self.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # Return the outputs from the QFormer model
        return query_outputs
    # 定义模型的前向传播方法，用于推断或训练过程中的数据传递
    def forward(
        self,
        # 输入的像素值张量，作为模型的输入之一
        pixel_values: torch.FloatTensor,
        # 编码器的输入标识符张量，描述输入序列
        input_ids: torch.FloatTensor,
        # 可选的注意力掩码张量，用于指示哪些元素需要被注意到
        attention_mask: Optional[torch.LongTensor] = None,
        # 可选的解码器输入标识符张量，用于解码器输入
        decoder_input_ids: Optional[torch.LongTensor] = None,
        # 可选的解码器注意力掩码张量，用于解码器的注意力掩码
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        # 是否输出注意力权重的标志，可选
        output_attentions: Optional[bool] = None,
        # 是否输出隐藏状态的标志，可选
        output_hidden_states: Optional[bool] = None,
        # 标签张量，用于训练时的目标标签
        labels: Optional[torch.LongTensor] = None,
        # 是否返回字典形式的输出，可选
        return_dict: Optional[bool] = None,
"""
BLIP-2 Model for generating text given an image and an optional text prompt. The model consists of a vision
encoder, Querying Transformer (Q-Former) and a language model.

One can optionally pass `input_ids` to the model, which serve as a text prompt, to make the language model continue
the prompt. Otherwise, the language model starts generating text from the [BOS] (beginning-of-sequence) token.

<Tip>

Note that Flan-T5 checkpoints cannot be cast to float16. They are pre-trained using bfloat16.

</Tip>
"""
@add_start_docstrings(
    """
    BLIP-2 Model for generating text given an image and an optional text prompt. The model consists of a vision
    encoder, Querying Transformer (Q-Former) and a language model.

    One can optionally pass `input_ids` to the model, which serve as a text prompt, to make the language model continue
    the prompt. Otherwise, the language model starts generating text from the [BOS] (beginning-of-sequence) token.

    <Tip>

    Note that Flan-T5 checkpoints cannot be cast to float16. They are pre-trained using bfloat16.

    </Tip>
    """,
    BLIP_2_START_DOCSTRING,
)
class Blip2ForConditionalGeneration(Blip2PreTrainedModel):
    config_class = Blip2Config
    main_input_name = "pixel_values"

    def __init__(self, config: Blip2Config):
        super().__init__(config)

        # Initialize vision model using Blip2VisionModel with given configuration
        self.vision_model = Blip2VisionModel(config.vision_config)

        # Initialize query tokens as a learnable parameter tensor for Q-Former
        self.query_tokens = nn.Parameter(torch.zeros(1, config.num_query_tokens, config.qformer_config.hidden_size))
        
        # Initialize Q-Former model using Blip2QFormerModel with given configuration
        self.qformer = Blip2QFormerModel(config.qformer_config)

        # Linear projection layer to map Q-Former's output to language model's input size
        self.language_projection = nn.Linear(config.qformer_config.hidden_size, config.text_config.hidden_size)

        # Conditionally select between AutoModelForCausalLM and AutoModelForSeq2SeqLM based on config
        if config.use_decoder_only_language_model:
            language_model = AutoModelForCausalLM.from_config(config.text_config)
        else:
            language_model = AutoModelForSeq2SeqLM.from_config(config.text_config)

        # Update _tied_weights_keys using the base model used
        if language_model._tied_weights_keys is not None:
            self._tied_weights_keys = [f"language_model.{k}" for k in language_model._tied_weights_keys]

        # Set the language model based on the condition above
        self.language_model = language_model

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def set_output_embeddings(self, new_embeddings):
        self.language_model.set_output_embeddings(new_embeddings)

    def get_output_embeddings(self) -> nn.Module:
        return self.language_model.get_output_embeddings()

    def get_encoder(self):
        return self.language_model.get_encoder()

    def get_decoder(self):
        return self.language_model.get_decoder()

    def _tie_weights(self):
        # Tie weights between encoder and decoder if not using decoder-only language model
        if not self.config.use_decoder_only_language_model:
            self.language_model.encoder.embed_tokens = self.language_model.shared
            self.language_model.decoder.embed_tokens = self.language_model.shared
    def _preprocess_accelerate(self):
        r"""
        Some pre-processing hacks to make the model `accelerate` compatible. Check
        https://github.com/huggingface/transformers/pull/21707 for more details.
        """
        # 将当前对象的 hf_device_map 属性赋值给局部变量 hf_device_map
        hf_device_map = self.hf_device_map

        # 如果 hf_device_map 中有多个设备，并且不包含 "language_model"，并且当前有多个 CUDA 设备可用
        if len(hf_device_map) > 1 and "language_model" not in hf_device_map and torch.cuda.device_count() > 1:
            # 发出警告，提示用户在使用多 GPU + BLIP-2 + `accelerate` 时可能会出现意外行为
            logger.warning(
                "The `language_model` is not in the `hf_device_map` dictionary and you are running your script"
                " in a multi-GPU environment. this may lead to unexpected behavior when using `accelerate`."
                " Please pass a `device_map` that contains `language_model` to remove this warning."
                " Please refer to https://github.com/huggingface/blog/blob/main/accelerate-large-models.md for"
                " more details on creating a `device_map` for large models.",
            )

        # 如果当前对象的 language_model 属性具有 _hf_hook 方法
        if hasattr(self.language_model, "_hf_hook"):
            # 设置 language_model._hf_hook.io_same_device 为 True，以保证 `generate` 方法的兼容性
            self.language_model._hf_hook.io_same_device = True  # For `generate` compatibility

    @add_start_docstrings_to_model_forward(BLIP_2_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Blip2ForConditionalGenerationModelOutput, config_class=Blip2VisionConfig)
    # 定义 forward 方法，接收多个输入参数并返回特定类型的输出
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
    ):
        # forward 方法的具体实现在这里被定义

    @torch.no_grad()
    # 定义 generate 方法，用于生成模型输出，不进行梯度计算
    def generate(
        self,
        pixel_values: torch.FloatTensor,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        **generate_kwargs,
    ):
        # generate 方法的具体实现在这里被定义
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
            # 如果模型有 `hf_device_map` 属性，进行 `accelerate` 预处理
            self._preprocess_accelerate()

        # 获取输入的批次大小
        batch_size = pixel_values.shape[0]
        # 提取图像特征向量，使用 `vision_model` 模型，并返回最后一个隐藏状态
        image_embeds = self.vision_model(pixel_values, return_dict=True).last_hidden_state
        # 创建图像的注意力掩码，全为 1，形状与 `image_embeds` 的前几维相同
        image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)

        # 扩展查询 tokens 到与 `image_embeds` 相同的批次大小
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        # 使用 `qformer` 模型处理查询 tokens 和图像特征向量，返回一个字典
        query_outputs = self.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
            return_dict=True,
        )
        # 获取查询输出的最后一个隐藏状态
        query_output = query_outputs.last_hidden_state

        # 对查询输出应用语言模型的投影层
        language_model_inputs = self.language_projection(query_output)
        # 创建语言模型的注意力掩码，全为 1，形状与 `language_model_inputs` 的前几维相同
        language_attention_mask = torch.ones(
            language_model_inputs.size()[:-1], dtype=torch.long, device=language_model_inputs.device
        )

        # 如果 `input_ids` 为空，则使用配置文件中的 `bos_token_id` 重复生成一个张量，并移到与 `image_embeds` 相同的设备上
        if input_ids is None:
            input_ids = (
                torch.LongTensor([[self.config.text_config.bos_token_id]])
                .repeat(batch_size, 1)
                .to(image_embeds.device)
            )
        # 如果 `attention_mask` 为空，则创建一个与 `input_ids` 相同形状的全 1 张量作为注意力掩码
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        # 拼接语言模型的注意力掩码和输入的注意力掩码，以便考虑到 padding 的处理
        attention_mask = torch.cat([language_attention_mask, attention_mask.to(language_attention_mask.device)], dim=1)

        # 将输入的 tokens embeddings 与语言模型的输入 embeddings 进行拼接
        inputs_embeds = self.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat([language_model_inputs, inputs_embeds.to(language_model_inputs.device)], dim=1)

        # 如果语言模型不是编码器-解码器结构，则增加 `max_length` 和 `min_length` 的值，确保最终计数仅在 token embeddings 上
        if not self.language_model.config.is_encoder_decoder:
            generate_kwargs["max_length"] = generate_kwargs.get("max_length", 20) + language_model_inputs.shape[1]
            generate_kwargs["min_length"] = generate_kwargs.get("min_length", 0) + language_model_inputs.shape[1]

        # 使用语言模型生成文本输出
        outputs = self.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **generate_kwargs,
        )

        # 返回生成的文本输出
        return outputs
```