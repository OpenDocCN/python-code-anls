# `.\models\instructblip\modeling_instructblip.py`

```py
# 设置文件编码为 UTF-8
# 版权声明，标明版权归 Salesforce 作者和 HuggingFace 团队所有，保留所有权利
#
# 根据 Apache 许可证 2.0 版本授权使用本文件；除非遵守许可证的要求，否则不得使用本文件
# 您可以在以下网址获取许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则本软件按“原样”提供，无任何明示或暗示的担保或条件
# 请参阅许可证以获取具体语言规定的权限以及限制
""" PyTorch InstructBLIP model."""

import math  # 导入数学库，用于数学运算
from dataclasses import dataclass  # 导入 dataclass 用于创建数据类
from typing import Any, Optional, Tuple, Union  # 导入类型提示

import torch  # 导入 PyTorch 库
import torch.utils.checkpoint  # 导入 PyTorch 的 checkpoint 模块
from torch import nn  # 从 PyTorch 导入神经网络模块
from torch.nn import CrossEntropyLoss  # 导入交叉熵损失函数

from ...activations import ACT2FN  # 导入激活函数
from ...modeling_outputs import (  # 导入模型输出相关类
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPooling,
    BaseModelOutputWithPoolingAndCrossAttentions,
)
from ...modeling_utils import PreTrainedModel  # 导入预训练模型工具类
from ...pytorch_utils import (  # 导入 PyTorch 工具类
    apply_chunking_to_forward,
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)
from ...utils import (  # 导入工具函数
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from ..auto import AutoModelForCausalLM, AutoModelForSeq2SeqLM  # 导入自动模型类
from .configuration_instructblip import InstructBlipConfig, InstructBlipQFormerConfig, InstructBlipVisionConfig  # 导入配置类


logger = logging.get_logger(__name__)  # 获取 logger 对象用于日志记录

_CHECKPOINT_FOR_DOC = "Salesforce/instructblip-flan-t5-xl"  # 指定用于文档的检查点名称

INSTRUCTBLIP_PRETRAINED_MODEL_ARCHIVE_LIST = [  # 预训练模型存档列表
    "Salesforce/instructblip-flan-t5-xl",
    # 查看所有 InstructBLIP 模型：https://huggingface.co/models?filter=instructblip
]


@dataclass  # 使用 dataclass 装饰器创建数据类
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

    loss: Optional[Tuple[torch.FloatTensor]] = None  # 语言模型损失，当提供标签时返回
    logits: Optional[Tuple[torch.FloatTensor]] = None  # 语言模型头部的预测分数
    vision_outputs: Optional[BaseModelOutputWithPooling] = None  # 视觉编码器的输出
    qformer_outputs: Optional[BaseModelOutputWithPoolingAndCrossAttentions] = None  # Q-Former 的输出（包含交叉注意力）
    language_model_outputs: Optional[Union[CausalLMOutputWithPast, Seq2SeqLMOutput]] = None  # 语言模型的输出
    # 声明并初始化可选的 torch.FloatTensor 类型的变量，分别用于视觉、问题转换器和语言模型的输出
    vision_outputs: Optional[torch.FloatTensor] = None
    qformer_outputs: Optional[Tuple[torch.FloatTensor]] = None
    language_model_outputs: Optional[Tuple[torch.FloatTensor]] = None

    # 定义一个方法，将对象转换为元组
    def to_tuple(self) -> Tuple[Any]:
        # 返回对象的元组形式，其中包括所有属性的值，但排除了特定的输出属性
        return tuple(
            self[k]  # 如果属性 k 不是特定的输出属性，则直接取其值
            if k not in ["vision_outputs", "qformer_outputs", "language_model_outputs"]
            else getattr(self, k).to_tuple()  # 如果属性 k 是特定的输出属性，则调用其 to_tuple() 方法
            for k in self.keys()  # 对于对象的所有属性进行处理
        )
# Copied from transformers.models.blip.modeling_blip.BlipVisionEmbeddings with Blip->InstructBlip
class InstructBlipVisionEmbeddings(nn.Module):
    def __init__(self, config: InstructBlipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size  # 设置嵌入维度为隐藏大小
        self.image_size = config.image_size  # 图像大小配置
        self.patch_size = config.patch_size  # 图像块大小配置

        self.class_embedding = nn.Parameter(torch.randn(1, 1, self.embed_dim))  # 类别嵌入参数，形状为[1, 1, embed_dim]

        # 图像块嵌入，使用2D卷积进行处理
        self.patch_embedding = nn.Conv2d(
            in_channels=3, out_channels=self.embed_dim, kernel_size=self.patch_size, stride=self.patch_size
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2  # 计算图像中的块数
        self.num_positions = self.num_patches + 1  # 加上一个位置用于类别嵌入

        self.position_embedding = nn.Parameter(torch.randn(1, self.num_positions, self.embed_dim))  # 位置嵌入参数

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        batch_size = pixel_values.shape[0]  # 获取批量大小
        target_dtype = self.patch_embedding.weight.dtype  # 获取目标数据类型
        patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype))  # 对输入像素值进行块嵌入处理
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)  # 展平并转置以适应后续的拼接操作

        class_embeds = self.class_embedding.expand(batch_size, 1, -1).to(target_dtype)  # 扩展类别嵌入以适应批量处理
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)  # 拼接类别嵌入和块嵌入
        embeddings = embeddings + self.position_embedding[:, : embeddings.size(1), :].to(target_dtype)  # 添加位置嵌入
        return embeddings


# Copied from transformers.models.blip_2.modeling_blip_2.Blip2Attention with Blip2->InstructBlip
class InstructBlipAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size  # 设置嵌入维度为隐藏大小
        self.num_heads = config.num_attention_heads  # 注意力头的数量
        self.head_dim = self.embed_dim // self.num_heads  # 每个注意力头的维度
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: "
                f"{self.num_heads})."
            )
        self.scale = self.head_dim**-0.5  # 缩放因子
        self.dropout = nn.Dropout(config.attention_dropout)  # 注意力机制的dropout率设置

        # 无偏置的线性层用于计算Q, K, V
        self.qkv = nn.Linear(self.embed_dim, 3 * self.embed_dim, bias=False)

        if config.qkv_bias:
            q_bias = nn.Parameter(torch.zeros(self.embed_dim))  # Q偏置参数
            v_bias = nn.Parameter(torch.zeros(self.embed_dim))  # V偏置参数
        else:
            q_bias = None
            v_bias = None

        if q_bias is not None:
            qkv_bias = torch.cat((q_bias, torch.zeros_like(v_bias, requires_grad=False), v_bias))
            self.qkv.bias = nn.Parameter(qkv_bias)  # 设置QKV线性层的偏置参数

        self.projection = nn.Linear(self.embed_dim, self.embed_dim)  # 线性投影层用于最终输出
    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        # 将输入张量重新形状为 (bsz, seq_len, num_heads, head_dim)，并交换维度顺序
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        # 获取隐藏状态张量的维度信息
        bsz, tgt_len, embed_dim = hidden_states.size()

        # 使用 self.qkv 对隐藏状态张量进行线性变换，产生混合的查询、键、值张量
        mixed_qkv = self.qkv(hidden_states)

        # 将混合的查询、键、值张量重新形状为 (bsz, tgt_len, 3, num_heads, embed_dim // num_heads) 并置换维度
        mixed_qkv = mixed_qkv.reshape(bsz, tgt_len, 3, self.num_heads, embed_dim // self.num_heads).permute(
            2, 0, 3, 1, 4
        )
        query_states, key_states, value_states = mixed_qkv[0], mixed_qkv[1], mixed_qkv[2]

        # 计算注意力分数，采用查询张量和键张量的点积
        attention_scores = torch.matmul(query_states, key_states.transpose(-1, -2))

        # 缩放注意力分数
        attention_scores = attention_scores * self.scale

        # 对注意力分数进行 softmax 归一化，得到注意力概率
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # 对注意力概率进行 dropout 处理
        attention_probs = self.dropout(attention_probs)

        # 如果给定了 head_mask，则将注意力概率与 head_mask 相乘
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        # 计算上下文层，采用注意力概率与值张量的乘积，然后置换维度
        context_layer = torch.matmul(attention_probs, value_states).permute(0, 2, 1, 3)

        # 将上下文层重新形状为与原始嵌入维度相匹配的形状
        new_context_layer_shape = context_layer.size()[:-2] + (self.embed_dim,)
        context_layer = context_layer.reshape(new_context_layer_shape)

        # 将上下文层传递给投影层进行线性变换，得到最终输出
        output = self.projection(context_layer)

        # 如果需要输出注意力分数，则将其包含在输出元组中
        outputs = (output, attention_probs) if output_attentions else (output, None)

        return outputs
# Copied from transformers.models.blip.modeling_blip.BlipMLP
class InstructBlipMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # 选择激活函数，从配置中获取并赋值给对象
        self.activation_fn = ACT2FN[config.hidden_act]
        # 创建第一个全连接层，输入维度为隐藏大小，输出维度为中间大小
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        # 创建第二个全连接层，输入维度为中间大小，输出维度为隐藏大小
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 将输入的隐藏状态传递给第一个全连接层
        hidden_states = self.fc1(hidden_states)
        # 应用预先选择的激活函数到第一个全连接层的输出
        hidden_states = self.activation_fn(hidden_states)
        # 将激活后的输出传递给第二个全连接层
        hidden_states = self.fc2(hidden_states)
        return hidden_states


# Copied from transformers.models.blip.modeling_blip.BlipEncoderLayer with Blip->InstructBlip
class InstructBlipEncoderLayer(nn.Module):
    def __init__(self, config: InstructBlipConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        # 初始化自注意力层，使用给定的配置对象
        self.self_attn = InstructBlipAttention(config)
        # 初始化第一个层归一化层，输入维度为嵌入维度，使用给定的层归一化参数
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        # 初始化MLP模块，使用给定的配置对象
        self.mlp = InstructBlipMLP(config)
        # 初始化第二个层归一化层，输入维度为嵌入维度，使用给定的层归一化参数
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

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
        residual = hidden_states

        # 对输入进行第一个层归一化操作
        hidden_states = self.layer_norm1(hidden_states)
        # 将归一化后的输入传递给自注意力层，并返回注意力权重
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            head_mask=attention_mask,
            output_attentions=output_attentions,
        )
        # 将自注意力层的输出与残差连接
        hidden_states = hidden_states + residual
        residual = hidden_states
        # 对连接后的输出进行第二个层归一化操作
        hidden_states = self.layer_norm2(hidden_states)
        # 将第二层归一化后的输出传递给MLP模块
        hidden_states = self.mlp(hidden_states)

        # 将MLP的输出与残差连接
        hidden_states = hidden_states + residual

        outputs = (hidden_states,)

        # 如果需要输出注意力权重，将其添加到输出中
        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class InstructBlipPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 指定用于配置的类
    config_class = InstructBlipConfig
    # 指定基础模型前缀名称
    base_model_prefix = "blip"
    # 支持梯度检查点
    supports_gradient_checkpointing = True
    _no_split_modules = [
        "InstructBlipQFormerEmbeddings",  # 列出不需要拆分的模块名称
        "InstructBlipAttention",          # 列出不需要拆分的模块名称
        "InstructBlipQFormerMultiHeadAttention",  # 列出不需要拆分的模块名称
        "InstructBlipQFormerSelfOutput",  # 列出不需要拆分的模块名称
    ]
    _keep_in_fp32_modules = []  # 留空列表，用于保留需要在FP32精度下操作的模块
    
    # 从 transformers.models.blip_2.modeling_blip_2.Blip2PreTrainedModel._init_weights 复制而来，将Blip2替换为InstructBlip
    def _init_weights(self, module):
        """初始化权重"""
        factor = self.config.initializer_range
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Embedding) or isinstance(module, nn.Linear):
            # 如果是卷积层、嵌入层或线性层，使用正态分布初始化权重，偏置置零
            module.weight.data.normal_(mean=0.0, std=factor)
            if hasattr(module, "bias") and module.bias is not None:
                module.bias.data.zero_()
    
        if isinstance(module, InstructBlipVisionEmbeddings):
            if hasattr(self.config, "vision_config"):
                factor = self.config.vision_config.initializer_range
            # 如果是InstructBlipVisionEmbeddings类型，使用截断正态分布初始化位置嵌入和类别嵌入
            nn.init.trunc_normal_(module.position_embedding, mean=0.0, std=factor)
            nn.init.trunc_normal_(module.class_embedding, mean=0.0, std=factor)
    
        elif isinstance(module, nn.LayerNorm):
            # 如果是LayerNorm层，偏置置零，权重置为1.0
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Linear) and module.bias is not None:
            # 如果是线性层且有偏置，偏置置零
            module.bias.data.zero_()
# 定义文档字符串，描述从 PreTrainedModel 继承的通用方法和本模型作为 PyTorch Module 的使用说明
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

# 定义输入说明文档字符串，描述 InstructBlipEncoder 类中 forward 方法的参数及其用途
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

# 空的输入文档字符串，暂无特定内容描述，可能是为了后续扩展或未完全实现的功能
INSTRUCTBLIP_INPUTS_DOCSTRING = r"""
"""


# 从 transformers.models.blip.modeling_blip.BlipEncoder 复制并重命名为 InstructBlipEncoder
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
        # 设置当前实例的配置参数
        self.config = config
        # 创建包含 config.num_hidden_layers 个 InstructBlipEncoderLayer 实例的模块列表
        self.layers = nn.ModuleList([InstructBlipEncoderLayer(config) for _ in range(config.num_hidden_layers)])
        # 梯度检查点标志，默认为 False
        self.gradient_checkpointing = False

    def forward(
        self,
        inputs_embeds,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        # forward 方法的具体参数说明详见 INSTRUCTBLIP_VISION_INPUTS_DOCSTRING
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
        # 根据传入的参数或者模型配置，确定是否输出注意力权重
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 根据传入的参数或者模型配置，确定是否输出各层隐藏状态
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 根据传入的参数或者模型配置，确定是否返回一个字典格式的输出
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 如果不需要输出隐藏状态，则设置为空元组
        encoder_states = () if output_hidden_states else None
        # 如果不需要输出注意力权重，则设置为空元组
        all_attentions = () if output_attentions else None

        # 将输入的嵌入向量作为初始隐藏状态
        hidden_states = inputs_embeds
        # 遍历所有编码器层
        for idx, encoder_layer in enumerate(self.layers):
            # 如果需要输出隐藏状态，则将当前隐藏状态加入到encoder_states中
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            # 如果启用渐变检查点且处于训练状态，则使用渐变检查点函数计算编码器层的输出
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    encoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    output_attentions,
                )
            else:
                # 否则直接调用编码器层计算输出
                layer_outputs = encoder_layer(
                    hidden_states,
                    attention_mask,
                    output_attentions=output_attentions,
                )

            # 更新隐藏状态为编码器层的输出的第一个元素
            hidden_states = layer_outputs[0]

            # 如果需要输出注意力权重，则将当前层的注意力权重加入到all_attentions中
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # 如果需要输出隐藏状态，则将最终的隐藏状态加入到encoder_states中
        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        # 如果不返回字典格式的输出，则返回一个包含非空值的元组
        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        # 否则返回一个BaseModelOutput对象，包含最终的隐藏状态、所有隐藏状态和注意力权重
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )
# 从transformers.models.blip.modeling_blip.BlipVisionModel复制而来，将Blip->InstructBlip，BLIP->INSTRUCTBLIP
class InstructBlipVisionModel(InstructBlipPreTrainedModel):
    # 主要输入名称为"pixel_values"
    main_input_name = "pixel_values"
    # 使用的配置类为InstructBlipVisionConfig
    config_class = InstructBlipVisionConfig

    def __init__(self, config: InstructBlipVisionConfig):
        super().__init__(config)
        self.config = config
        embed_dim = config.hidden_size

        # 初始化嵌入层
        self.embeddings = InstructBlipVisionEmbeddings(config)
        # 初始化编码器
        self.encoder = InstructBlipEncoder(config)
        # 初始化后层归一化层
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

        # 执行初始化后操作
        self.post_init()

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
        前向传播函数:
        
        Args:
            pixel_values (Optional[torch.FloatTensor], optional): 像素值. Defaults to None.
            output_attentions (Optional[bool], optional): 是否输出注意力权重. Defaults to None.
            output_hidden_states (Optional[bool], optional): 是否输出隐藏状态. Defaults to None.
            return_dict (Optional[bool], optional): 是否返回字典格式结果. Defaults to None.

        Returns:
            Union[Tuple, BaseModelOutputWithPooling]: 返回值包含最后隐藏状态、池化输出以及可能的其他结果.
        """
        # 如果未提供像素值，则抛出数值错误
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # 将像素值转换为嵌入向量
        hidden_states = self.embeddings(pixel_values)

        # 使用编码器处理嵌入向量
        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取编码器输出的最后隐藏状态，并进行后层归一化
        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.post_layernorm(last_hidden_state)

        # 对最后隐藏状态进行池化，获取池化输出，并再次进行后层归一化
        pooled_output = last_hidden_state[:, 0, :]
        pooled_output = self.post_layernorm(pooled_output)

        # 如果不返回字典格式的结果，则返回元组格式的结果
        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        # 返回带有池化输出和其他信息的 BaseModelOutputWithPooling 对象
        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

    def get_input_embeddings(self):
        return self.embeddings


class InstructBlipQFormerMultiHeadAttention(nn.Module):
    # 初始化函数，用于创建一个新的注意力模型层
    def __init__(self, config, is_cross_attention=False):
        super().__init__()
        # 将配置信息保存到实例中
        self.config = config
        # 检查隐藏层大小是否能够被注意力头的数量整除，或者是否有嵌入大小属性
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            # 如果不能整除且没有嵌入大小属性，则抛出异常
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention heads (%d)"
                % (config.hidden_size, config.num_attention_heads)
            )

        # 设置注意力头的数量
        self.num_attention_heads = config.num_attention_heads
        # 计算每个注意力头的大小
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        # 计算所有头的总大小
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # 创建查询线性层，用于生成查询矩阵
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        
        # 根据是否是交叉注意力模型，选择创建键和值的线性层
        if is_cross_attention:
            # 如果是交叉注意力，使用编码器隐藏层大小创建键和值的线性层
            self.key = nn.Linear(config.encoder_hidden_size, self.all_head_size)
            self.value = nn.Linear(config.encoder_hidden_size, self.all_head_size)
        else:
            # 如果不是交叉注意力，使用隐藏层大小创建键和值的线性层
            self.key = nn.Linear(config.hidden_size, self.all_head_size)
            self.value = nn.Linear(config.hidden_size, self.all_head_size)

        # 创建一个dropout层，用于注意力概率的dropout
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        
        # 设置位置嵌入类型，默认为绝对位置嵌入
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        
        # 如果位置嵌入类型为相对键或者相对键查询，则创建距离嵌入层
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            # 设置最大位置嵌入数
            self.max_position_embeddings = config.max_position_embeddings
            # 创建距离嵌入的Embedding层
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)
        
        # 初始化保存注意力权重标志为False
        self.save_attention = False
# Copied from transformers.models.bert.modeling_bert.BertSelfOutput with Bert->InstructBlipQFormer
class InstructBlipQFormerSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 初始化线性层，用于变换隐藏状态的维度
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 初始化层归一化，对输出进行标准化
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 初始化 dropout，以防止过拟合
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 对隐藏状态应用线性层变换
        hidden_states = self.dense(hidden_states)
        # 应用 dropout
        hidden_states = self.dropout(hidden_states)
        # 将层归一化应用到变换后的隐藏状态和输入张量之和上
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


# Copied from transformers.models.blip_2.modeling_blip_2.Blip2QFormerAttention with Blip2->InstructBlip
class InstructBlipQFormerAttention(nn.Module):
    def __init__(self, config, is_cross_attention=False):
        super().__init__()
        # 初始化自注意力机制和输出层
        self.attention = InstructBlipQFormerMultiHeadAttention(config, is_cross_attention)
        self.output = InstructBlipQFormerSelfOutput(config)
        # 初始化一个空集合，用于存储需要剪枝的注意力头
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        # 找到需要剪枝的注意力头并返回索引
        heads, index = find_pruneable_heads_and_indices(
            heads, self.attention.num_attention_heads, self.attention.attention_head_size, self.pruned_heads
        )

        # 剪枝线性层
        self.attention.query = prune_linear_layer(self.attention.query, index)
        self.attention.key = prune_linear_layer(self.attention.key, index)
        self.attention.value = prune_linear_layer(self.attention.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # 更新超参数并存储已剪枝的注意力头
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
        # 执行自注意力机制的前向传播
        self_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        # 将自注意力输出应用到自输出层，并与原始隐藏状态相加
        attention_output = self.output(self_outputs[0], hidden_states)
        # 如果需要输出注意力权重，则将它们添加到输出中
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs
# 从 transformers.models.bert.modeling_bert.BertIntermediate 复制并修改为 InstructBlipQFormerIntermediate 类
class InstructBlipQFormerIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 创建一个全连接层，输入大小为 config.hidden_size，输出大小为 config.intermediate_size
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        # 根据配置选择激活函数，将字符串形式的激活函数映射到对应的函数，存储在 self.intermediate_act_fn 中
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 使用全连接层进行前向传播，输入为 hidden_states，输出为经过全连接层处理后的 hidden_states
        hidden_states = self.dense(hidden_states)
        # 对经过全连接层处理后的 hidden_states 应用预先选择的激活函数
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


# 从 transformers.models.bert.modeling_bert.BertOutput 复制并修改为 InstructBlipQFormerOutput 类
class InstructBlipQFormerOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 创建一个全连接层，输入大小为 config.intermediate_size，输出大小为 config.hidden_size
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        # 创建 LayerNorm 层，对输入大小为 config.hidden_size 的数据进行归一化
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 创建 Dropout 层，以指定的概率 config.hidden_dropout_prob 随机丢弃输入
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 使用全连接层处理 hidden_states，将其转换为大小为 config.hidden_size 的张量
        hidden_states = self.dense(hidden_states)
        # 对全连接层输出应用 Dropout 操作，以减少过拟合风险
        hidden_states = self.dropout(hidden_states)
        # 将经过 Dropout 处理后的 hidden_states 与输入张量 input_tensor 相加，并对结果进行 LayerNorm 处理
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


# InstructBlipQFormerLayer 类定义
class InstructBlipQFormerLayer(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        # 设置前向传播中的分块大小和序列长度维度
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        # 创建注意力层对象 InstructBlipQFormerAttention
        self.attention = InstructBlipQFormerAttention(config)
        
        # 记录当前层的索引
        self.layer_idx = layer_idx

        # 如果当前层索引能够整除 config.cross_attention_frequency，说明需要进行跨层注意力操作
        if layer_idx % config.cross_attention_frequency == 0:
            # 创建跨层注意力对象 InstructBlipQFormerAttention
            self.crossattention = InstructBlipQFormerAttention(config, is_cross_attention=True)
            self.has_cross_attention = True  # 标记该层有跨层注意力
        else:
            self.has_cross_attention = False  # 标记该层没有跨层注意力

        # 创建中间层对象 InstructBlipQFormerIntermediate 和输出层对象 InstructBlipQFormerOutput
        self.intermediate = InstructBlipQFormerIntermediate(config)
        self.output = InstructBlipQFormerOutput(config)

        # 创建用于查询的中间层对象和输出层对象
        self.intermediate_query = InstructBlipQFormerIntermediate(config)
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
        ):
        # 省略了 forward 方法的其余部分，需要根据具体情况补充完整
        pass  # 这里只是为了注释需要包含完整的代码块
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        # 解码器单向自注意力的缓存键/值元组位于位置 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        # 使用保存的过去键/值元组（如果存在）进行自注意力计算
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        # 获取自注意力计算的输出
        attention_output = self_attention_outputs[0]
        # 获取除了第一个和最后一个元素之外的所有输出（主要用于输出注意力权重）
        outputs = self_attention_outputs[1:-1]

        # 获取当前的键/值元组，以备将来的计算使用
        present_key_value = self_attention_outputs[-1]

        # 如果查询长度大于 0，则截取注意力输出的一部分
        if query_length > 0:
            query_attention_output = attention_output[:, :query_length, :]

            # 如果模型包含交叉注意力，执行交叉注意力的计算
            if self.has_cross_attention:
                # 如果缺少编码器的隐藏状态，则引发错误
                if encoder_hidden_states is None:
                    raise ValueError("encoder_hidden_states must be given for cross-attention layers")
                # 执行交叉注意力计算
                cross_attention_outputs = self.crossattention(
                    query_attention_output,
                    attention_mask,
                    head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    output_attentions=output_attentions,
                )
                # 获取交叉注意力计算的输出
                query_attention_output = cross_attention_outputs[0]
                # 如果需要输出注意力权重，则将其添加到已有的输出中
                outputs = outputs + cross_attention_outputs[1:-1]

            # 将查询注意力输出传递给前馈网络的函数，可能会对其进行分块处理
            layer_output = apply_chunking_to_forward(
                self.feed_forward_chunk_query,
                self.chunk_size_feed_forward,
                self.seq_len_dim,
                query_attention_output,
            )

            # 如果注意力输出的形状大于查询长度，则对其余部分进行前馈网络计算
            if attention_output.shape[1] > query_length:
                layer_output_text = apply_chunking_to_forward(
                    self.feed_forward_chunk,
                    self.chunk_size_feed_forward,
                    self.seq_len_dim,
                    attention_output[:, query_length:, :],
                )
                # 将计算得到的结果拼接到之前的输出结果中
                layer_output = torch.cat([layer_output, layer_output_text], dim=1)
        else:
            # 对注意力输出进行前馈网络计算，可能会对其进行分块处理
            layer_output = apply_chunking_to_forward(
                self.feed_forward_chunk,
                self.chunk_size_feed_forward,
                self.seq_len_dim,
                attention_output,
            )
        # 将最终的层输出添加到总体的输出中
        outputs = (layer_output,) + outputs

        # 将当前的键/值元组添加到总体的输出中
        outputs = outputs + (present_key_value,)

        # 返回最终的输出结果
        return outputs
# 从 transformers.models.blip_2.modeling_blip_2.Blip2QFormerEncoder 复制并修改为 InstructBlipQFormerEncoder
class InstructBlipQFormerEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # 创建一个 nn.ModuleList，包含 config.num_hidden_layers 个 InstructBlipQFormerLayer 对象
        self.layer = nn.ModuleList(
            [InstructBlipQFormerLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
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
        ):
            # 如果输出隐藏状态为真，则初始化一个空元组
            all_hidden_states = () if output_hidden_states else None
            # 如果输出注意力权重为真，则初始化一个空元组
            all_self_attentions = () if output_attentions else None
            # 如果输出交叉注意力权重为真，则初始化一个空元组
            all_cross_attentions = () if output_attentions else None

            # 如果使用缓存，则初始化一个空元组
            next_decoder_cache = () if use_cache else None

            # 遍历所有 Transformer 层
            for i in range(self.config.num_hidden_layers):
                # 获取当前层的模块
                layer_module = self.layer[i]
                # 如果输出隐藏状态为真，则将当前隐藏状态添加到所有隐藏状态元组中
                if output_hidden_states:
                    all_hidden_states = all_hidden_states + (hidden_states,)

                # 获取当前层的注意力头掩码
                layer_head_mask = head_mask[i] if head_mask is not None else None
                # 获取当前层的过去键值对，用于跨层信息传递
                past_key_value = past_key_values[i] if past_key_values is not None else None

                # 如果配置启用梯度检查点且当前处于训练状态
                if getattr(self.config, "gradient_checkpointing", False) and self.training:
                    # 如果使用缓存为真，警告并设置为假，因为与梯度检查点不兼容
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
                    # 否则直接调用当前层模块计算当前层的输出
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
                # 如果使用缓存为真，则将当前层的输出的最后一个元素添加到下一个解码器缓存中
                if use_cache:
                    next_decoder_cache += (layer_outputs[-1],)
                # 如果输出注意力权重为真，则将当前层的注意力权重添加到所有自注意力权重元组中
                if output_attentions:
                    all_self_attentions = all_self_attentions + (layer_outputs[1],)
                    # 如果当前层有交叉注意力，将当前层的交叉注意力权重添加到所有交叉注意力权重元组中
                    if layer_module.has_cross_attention:
                        all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

            # 如果输出隐藏状态为真，则将最终隐藏状态添加到所有隐藏状态元组中
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 如果不返回字典，则返回包含非空元素的元组
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
            # 否则返回带有过去和交叉注意力权重的基础模型输出对象
            return BaseModelOutputWithPastAndCrossAttentions(
                last_hidden_state=hidden_states,
                past_key_values=next_decoder_cache,
                hidden_states=all_hidden_states,
                attentions=all_self_attentions,
                cross_attentions=all_cross_attentions,
            )
# 定义一个模块类 InstructBlipQFormerEmbeddings，用于构建来自单词和位置嵌入的嵌入向量。
class InstructBlipQFormerEmbeddings(nn.Module):
    """Construct the embeddings from word and position embeddings."""

    # 初始化函数，接受一个配置对象 config 作为参数
    def __init__(self, config):
        super().__init__()
        # 创建一个单词嵌入层，vocab_size 是词汇表大小，hidden_size 是隐藏层大小，padding_idx 是填充标记的索引
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        # 创建一个位置嵌入层，max_position_embeddings 是最大位置嵌入数，hidden_size 是隐藏层大小
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)

        # LayerNorm 层，用于归一化隐藏层的输出
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # Dropout 层，用于在训练过程中随机失活隐藏层的输出，以防止过拟合
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # 注册一个缓冲区 position_ids，这个是位置标识符，是一个长度为 max_position_embeddings 的张量
        # 在序列化时它是内存连续的，并且当持久性设置为 False 时不会被导出
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )
        # 设置位置嵌入的类型，默认是绝对位置编码
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")

        self.config = config

    # 前向传播函数，接收输入参数并计算输出
    def forward(
        self,
        input_ids=None,
        position_ids=None,
        query_embeds=None,
        past_key_values_length=0,
    ):
        # 如果输入 input_ids 不为空，则获取序列长度 seq_length
        if input_ids is not None:
            seq_length = input_ids.size()[1]
        else:
            seq_length = 0

        # 如果位置标识符 position_ids 为空，则从预定义的 position_ids 中复制相应位置的标识符
        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length].clone()

        # 如果输入 input_ids 不为空，则计算单词嵌入
        if input_ids is not None:
            embeddings = self.word_embeddings(input_ids)
            # 如果位置嵌入类型为绝对位置编码，则计算位置嵌入并将其加到单词嵌入中
            if self.position_embedding_type == "absolute":
                position_embeddings = self.position_embeddings(position_ids.to(embeddings.device))
                embeddings = embeddings + position_embeddings

            # 如果存在查询嵌入 query_embeds，则将其与计算得到的嵌入拼接起来
            if query_embeds is not None:
                embeddings = torch.cat((query_embeds, embeddings), dim=1)
        else:
            # 如果没有输入 input_ids，则使用查询嵌入作为嵌入向量
            embeddings = query_embeds

        # 将嵌入向量转换为与 layernorm 权重相同的数据类型
        embeddings = embeddings.to(self.layernorm.weight.dtype)
        # 应用 layernorm 对嵌入向量进行归一化
        embeddings = self.layernorm(embeddings)
        # 对归一化后的向量应用 dropout 操作
        embeddings = self.dropout(embeddings)
        # 返回最终的嵌入向量作为输出
        return embeddings


# 定义一个模型类 InstructBlipQFormerModel，继承自 InstructBlipPreTrainedModel 类
class InstructBlipQFormerModel(InstructBlipPreTrainedModel):
    """
    Querying Transformer (Q-Former), used in InstructBLIP. Slightly modified from BLIP-2 as it also takes the
    instruction as input.
    """

    # 初始化函数，接受一个配置对象 config 作为参数
    def __init__(self, config: InstructBlipQFormerConfig):
        super().__init__(config)
        self.config = config

        # 创建嵌入层对象
        self.embeddings = InstructBlipQFormerEmbeddings(config)

        # 创建编码器对象
        self.encoder = InstructBlipQFormerEncoder(config)

        # 调用初始化后的处理方法
        self.post_init()

    # 获取输入嵌入层的函数，返回嵌入层的单词嵌入部分
    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    # 设置输入嵌入层的函数，设置嵌入层的单词嵌入部分
    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    # 剪枝模型头部的方法，接受一个 heads_to_prune 字典作为参数，用于指定要剪枝的头部
    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        # 遍历 heads_to_prune 字典的每一项，获取层号和要剪枝的头部列表
        for layer, heads in heads_to_prune.items():
            # 对指定层的注意力头部进行剪枝操作
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
            `torch.Tensor` The extended attention mask, with the same dtype as `attention_mask.dtype`.
        """
        # 如果 attention_mask 的维度是 [batch_size, from_seq_length, to_seq_length]
        # 则将其扩展为 [batch_size, 1, from_seq_length, to_seq_length]，以便广播到所有的注意力头上
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            # 如果提供的是维度为 [batch_size, seq_length] 的填充遮罩
            # 模型是编码器，因此将遮罩扩展为 [batch_size, 1, 1, seq_length]，使其可广播到 [batch_size, num_heads, seq_length, seq_length]
            extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})",
            )

        # 由于 attention_mask 中 1.0 表示我们要关注的位置，0.0 表示被屏蔽的位置，
        # 这个操作将创建一个张量，对于我们要关注的位置是 0.0，对于被屏蔽的位置是 -10000.0
        # 在 softmax 之前将其添加到原始分数中，实际上等同于完全删除这些位置的影响
        extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)  # 用于 fp16 兼容性
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask
"""
为生成文本的 InstructBLIP 模型提供说明，该模型基于图像和可选文本提示生成文本。模型包括视觉编码器、查询变压器（Q-Former）和语言模型。

可以选择向模型传递 `input_ids`，作为文本提示，以便让语言模型继续提示。否则，语言模型将从 [BOS]（序列开始）标记开始生成文本。
"""
@add_start_docstrings(
    """
    InstructBLIP Model for generating text given an image and an optional text prompt. The model consists of a vision
    encoder, Querying Transformer (Q-Former) and a language model.

    One can optionally pass `input_ids` to the model, which serve as a text prompt, to make the language model continue
    the prompt. Otherwise, the language model starts generating text from the [BOS] (beginning-of-sequence) token.
    """,
    INSTRUCTBLIP_START_DOCSTRING,
)
class InstructBlipForConditionalGeneration(InstructBlipPreTrainedModel):
    config_class = InstructBlipConfig
    main_input_name = "pixel_values"

    def __init__(self, config: InstructBlipConfig):
        super().__init__(config)

        # 初始化视觉模型
        self.vision_model = InstructBlipVisionModel(config.vision_config)

        # 初始化查询令牌
        self.query_tokens = nn.Parameter(torch.zeros(1, config.num_query_tokens, config.qformer_config.hidden_size))
        # 初始化查询变压器模型
        self.qformer = InstructBlipQFormerModel(config.qformer_config)

        # 配置语言投影层，将查询变压器的隐藏状态映射到文本配置的隐藏大小
        self.language_projection = nn.Linear(config.qformer_config.hidden_size, config.text_config.hidden_size)

        # 根据配置选择语言模型：CausalLM 或 Seq2SeqLM
        if config.use_decoder_only_language_model:
            language_model = AutoModelForCausalLM.from_config(config.text_config)
        else:
            language_model = AutoModelForSeq2SeqLM.from_config(config.text_config)

        # 如果语言模型有不需拆分的模块，则扩展这些模块
        if language_model._no_split_modules is not None:
            self._no_split_modules.extend(language_model._no_split_modules)

        # 如果语言模型有需保持在 FP32 的模块，则扩展这些模块
        if language_model._keep_in_fp32_modules is not None:
            self._keep_in_fp32_modules.extend(language_model._keep_in_fp32_modules)

        # 设置实例的语言模型
        self.language_model = language_model

        # 初始化权重并应用最终处理
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
        # 如果不是仅使用解码器语言模型，则绑定权重
        if not self.config.use_decoder_only_language_model:
            self.language_model.encoder.embed_tokens = self.language_model.shared
            self.language_model.decoder.embed_tokens = self.language_model.shared
    def _preprocess_accelerate(self):
        r"""
        Some pre-processing hacks to make the model `accelerate` compatible. Check
        https://github.com/huggingface/transformers/pull/21707 for more details.
        """
        # 将 self.hf_device_map 赋值给 hf_device_map 变量
        hf_device_map = self.hf_device_map

        # 如果 hf_device_map 的长度大于 1，且不包含 "language_model" 键，并且 CUDA 设备数量大于 1
        if len(hf_device_map) > 1 and "language_model" not in hf_device_map and torch.cuda.device_count() > 1:
            # 发出警告，提示用户在使用多 GPU + InstructBLIP + `accelerate` 时可能会遇到意外行为
            logger.warning(
                "The `language_model` is not in the `hf_device_map` dictionary and you are running your script"
                " in a multi-GPU environment. this may lead to unexpected behavior when using `accelerate`."
                " Please pass a `device_map` that contains `language_model` to remove this warning."
                " Please refer to https://github.com/huggingface/blog/blob/main/accelerate-large-models.md for"
                " more details on creating a `device_map` for large models.",
            )

        # 如果 self.language_model 具有属性 "_hf_hook"
        if hasattr(self.language_model, "_hf_hook"):
            # 设置 self.language_model._hf_hook.io_same_device 为 True，以便与 `generate` 兼容
            self.language_model._hf_hook.io_same_device = True  # For `generate` compatibility

    @add_start_docstrings_to_model_forward(INSTRUCTBLIP_INPUTS_DOCSTRING)
    @replace_return_docstrings(
        output_type=InstructBlipForConditionalGenerationModelOutput, config_class=InstructBlipVisionConfig
    )
    # 定义 forward 方法，接受多个参数并返回指定类型的输出
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
    ):
        # 方法级装饰器，标记 forward 方法为添加了特定文档字符串的模型前向方法

    @torch.no_grad()
    # 方法级装饰器，标记 generate 方法为不需要梯度的方法
    def generate(
        self,
        pixel_values: torch.FloatTensor,
        qformer_input_ids: Optional[torch.LongTensor] = None,
        qformer_attention_mask: Optional[torch.LongTensor] = None,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        **generate_kwargs,
    ):
        # 定义 generate 方法，接受多个参数，用于生成模型的输出
```