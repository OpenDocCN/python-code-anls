# `.\transformers\models\mpt\modeling_mpt.py`

```
# 设置文件编码为 utf-8
# 版权声明：2023年由 HuggingFace Inc. 团队和 MosaicML NLP 团队拥有
# 根据 Apache 许可证 2.0 版本授权，除非在符合许可证的情况下不得使用此文件
# 您可以在以下网址获取许可证的副本：http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于"AS IS"基础分发的，没有任何种类的保证或条件
# 请参阅许可证获取有关权限的具体说明和限制
"""PyTorch MPT model."""

import math
from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, LayerNorm, MSELoss
from torch.nn import functional as F

from ...file_utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward
from ...modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from ...modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from ...modeling_utils import PreTrainedModel
from ...utils import logging
from .configuration_mpt import MptConfig

# 获取日志记录器对象
logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "mosaicml/mpt-7b"
_CONFIG_FOR_DOC = "MptConfig"

# 预训练模型列表
MPT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "mosaicml/mpt-7b",
    "mosaicml/mpt-7b-storywriter",
    "mosaicml/mpt-7b-instruct",
    "mosaicml/mpt-7b-8k",
    "mosaicml/mpt-7b-8k-instruct",
    "mosaicml/mpt-7b-8k-chat",
    "mosaicml/mpt-30b",
    "mosaicml/mpt-30b-instruct",
    "mosaicml/mpt-30b-chat",
    # 查看所有的 MPT 模型：https://huggingface.co/models?filter=mpt
]

def build_mpt_alibi_tensor(num_heads, sequence_length, alibi_bias_max=8, device=None):
    r"""
    Link to paper: https://arxiv.org/abs/2108.12409 - Alibi tensor is not causal as the original paper mentions, it
    relies on a translation invariance of softmax for quick implementation. This implementation has been copied from
    the alibi implementation of MPT source code that led to slightly different results than the Bloom alibi:
    https://huggingface.co/mosaicml/mpt-7b/blob/main/attention.py#L292
    """
    # 创建 Alibi 张量，详情参考论文链接
    alibi = torch.arange(1 - sequence_length, 1, dtype=torch.int32, device=device).view(1, 1, 1, sequence_length)
    num_heads_power_of_2 = 2 ** math.ceil(math.log2(num_heads))

    base = torch.arange(1, num_heads_power_of_2 + 1, dtype=torch.float32, device=device)
    base = base * (alibi_bias_max / num_heads_power_of_2)

    slopes = 1.0 / torch.pow(2, base)
    slopes = slopes.view(1, num_heads_power_of_2, 1, 1)

    if num_heads_power_of_2 != num_heads:
        slopes = torch.concat([slopes[:, 1::2, ...], slopes[:, ::2, ...]], dim=1)[:, :num_heads, ...]
    # 将alibi和slopes进行逐元素相乘
    alibi = alibi * slopes
    # 压缩alibi的第一个维度为1
    return alibi.squeeze(0)
class MptAttention(nn.Module):
    """定义多头自注意力模块。
    使用 torch 或 triton 的注意力实现，使用户还可以使用加性偏置。
    """

    def __init__(self, config: MptConfig):
        # 初始化函数，设置模块的各种参数
        super().__init__()
        self.hidden_size = config.hidden_size
        self.n_heads = config.n_heads
        self.max_seq_length = config.max_seq_len
        self.head_dim = self.hidden_size // self.n_heads
        self.softmax_scale = config.attn_config.softmax_scale
        if self.softmax_scale is None:
            self.softmax_scale = 1 / math.sqrt(self.hidden_size / self.n_heads)

        self.attn_dropout_p = config.attn_config.attn_pdrop
        self.Wqkv = nn.Linear(self.hidden_size, 3 * self.hidden_size, bias=False)
        self.out_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_bias: torch.Tensor,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        # 获取隐藏状态张量的批量大小和序列长度
        batch_size, seq_length = hidden_states.shape[:2]

        # 通过线性变换将隐藏状态张量映射为混合的查询、键和值
        mixed_qkv = self.Wqkv(hidden_states)
        # 将混合的查询、键和值张量分成三个部分
        query_states, key_states, value_states = mixed_qkv.chunk(3, dim=2)
        # 将查询张量重塑为(batch_size, n_heads, seq_length, head_dim)，并转置维度以便于后续计算
        query_states = query_states.reshape(batch_size, seq_length, self.n_heads, self.head_dim).transpose(1, 2)
        # 将键张量重塑为(batch_size, n_heads, seq_length, head_dim)，并转置维度以便于后续计算
        key_states = key_states.reshape(batch_size, seq_length, self.n_heads, self.head_dim).transpose(1, 2)
        # 将值张量重塑为(batch_size, n_heads, seq_length, head_dim)，并转置维度以便于后续计算
        value_states = value_states.reshape(batch_size, seq_length, self.n_heads, self.head_dim).transpose(1, 2)

        # 如果过去的键值不为空
        if past_key_value is not None:
            # 如果过去的键值长度不为零
            if len(past_key_value) != 0:
                # 将当前的键张量与过去的键张量连接在一起
                key_states = torch.cat([past_key_value[0], key_states], dim=2)
                # 将当前的值张量与过去的值张量连接在一起
                value_states = torch.cat([past_key_value[1], value_states], dim=2)
            # 更新过去的键值为当前的键值
            past_key_value = (key_states, value_states)
        else:
            # 如果过去的键值为空，则将当前的键值设置为过去的键值
            past_key_value = (key_states, value_states)

        # 计算注意力分数
        attention_scores = torch.matmul(query_states, key_states.transpose(-1, -2)) * self.softmax_scale

        # 获取查询长度
        query_length = seq_length if past_key_value is None else seq_length + past_key_value[0].shape[2]

        # 如果存在位置偏置
        if position_bias is not None:
            # 如果位置偏置张量的维度不为3
            if len(position_bias.shape) != 3:
                # 抛出值错误异常
                raise ValueError(f"Expecting position_bias shape to be 3 dimensions, got {len(position_bias.shape)}")
            # 获取键长度
            key_length = key_states.shape[-2]

            # 获取位置偏置的查询索引和键索引
            position_bias_query_index = max(0, position_bias.size(1) - query_length)
            position_bias_key_index = max(0, position_bias.size(2) - key_length)

            # 截取位置偏置张量的子张量以匹配当前查询和键的长度
            position_bias = position_bias[:, position_bias_query_index:, position_bias_key_index:]

            # 将位置偏置加到注意力分数上
            attention_scores = attention_scores + position_bias

        # 如果存在注意力遮罩
        if attention_mask is not None:
            # 使用遮罩填充注意力分数中的无效位置
            attention_scores = attention_scores.masked_fill(attention_mask, torch.finfo(query_states.dtype).min)

        # 计算注意力权重并进行 dropout
        attn_weights = nn.functional.softmax(attention_scores.float(), dim=-1).to(value_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attn_dropout_p, training=self.training)

        # 计算上下文张量
        context_states = torch.matmul(attn_weights, value_states)
        # 将上下文张量的维度重新排列以匹配输出维度
        context_states = context_states.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_length, -1)
        # 对上下文张量应用输出投影
        attn_output = self.out_proj(context_states)

        # 返回注意力输出、注意力权重和更新的过去键值
        return attn_output, attn_weights, past_key_value
# 定义一个名为 MptMLP 的类，继承自 nn.Module 类
class MptMLP(nn.Module):
    # 初始化函数，接受一个类型为 MptConfig 的参数 config
    def __init__(self, config: MptConfig):
        # 调用父类的初始化函数
        super().__init__()
        # 从 config 对象中获取 hidden_size 参数
        hidden_size = config.hidden_size

        # 定义一个线性变换层，输入维度为 hidden_size，输出维度为 4 * hidden_size，不使用偏置
        self.up_proj = nn.Linear(hidden_size, 4 * hidden_size, bias=False)
        # 定义 GELU 激活函数，似乎传入了不支持的参数 "none"
        self.act = nn.GELU(approximate="none")
        # 定义一个线性变换层，输入维度为 4 * hidden_size，输出维度为 hidden_size，不使用偏置
        self.down_proj = nn.Linear(4 * hidden_size, hidden_size, bias=False)
        # 从 config 对象的 attn_config 属性中获取 attn_pdrop 参数作为隐藏层的dropout概率
        self.hidden_dropout = config.attn_config.attn_pdrop

    # 前向传播函数，接受类型为 torch.Tensor 的 hidden_states 和 residual 参数，返回类型为 torch.Tensor
    def forward(self, hidden_states: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        # 使用 GELU 激活函数处理 up_proj 层的输出，得到 hidden_states
        hidden_states = self.act(self.up_proj(hidden_states))

        # 使用 down_proj 层处理 hidden_states，得到 intermediate_output
        intermediate_output = self.down_proj(hidden_states)

        # 对 intermediate_output 执行 dropout，使用隐藏层的dropout概率，训练时启用
        output = F.dropout(intermediate_output, p=self.hidden_dropout, training=self.training)
        
        # 将 output 与 residual 相加作为返回值
        output = output + residual

        # 返回 output
        return output


# 定义一个名为 MptBlock 的类，继承自 nn.Module 类
class MptBlock(nn.Module):
    # 初始化函数，接受一个类型为 MptConfig 的参数 config
    def __init__(self, config: MptConfig):
        # 调用父类的初始化函数
        super().__init__()
        # 从 config 对象中获取 hidden_size 参数
        hidden_size = config.hidden_size

        # 初始化 LayerNorm 层，输入维度为 hidden_size，eps 参数为 config 对象的 layer_norm_epsilon 属性
        self.norm_1 = LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        # 为了向后兼容，对加载的权重使用 Hub 上的权重值
        self.norm_1.bias = None

        # 从 config 对象中获取 n_heads 参数作为 self.num_heads
        self.num_heads = config.n_heads
        # 初始化 MptAttention 层
        self.attn = MptAttention(config)

        # 初始化 LayerNorm 层，输入维度为 hidden_size，eps 参数为 config 对象的 layer_norm_epsilon 属性
        self.norm_2 = LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        # 为了向后兼容，对加载的权重使用 Hub 上的权重值
        self.norm_2.bias = None

        # 初始化 MptMLP 层
        self.ffn = MptMLP(config)

        # 从 attn_config 属性中获取 attn_pdrop 参数作为 dropout_rate
        self.dropout_rate = config.attn_config.attn_pdrop
        # 使用 dropout_rate 初始化 dropout 层
        self.resid_attn_dropout = nn.Dropout(self.dropout_rate)

    # 前向传播函数
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_bias: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        output_attentions: bool = False
    ):
        # 对 hidden_states 执行 layer_norm，得到 layernorm_output
        layernorm_output = self.norm_1(hidden_states)
        
        # 保存 hidden_states 作为 residual
        residual = hidden_states

        # 调用 self.attn 进行 self attention 计算，得到 attn_outputs, attn_weights, past_key_value
        attn_outputs, attn_weights, past_key_value = self.attn(
            layernorm_output,
            position_bias=position_bias,
            attention_mask=attention_mask,
            past_key_value=layer_past
        )

        # 对 attn_outputs 执行 dropout，使用 dropout_rate，训练时启用
        hidden_states = self.resid_attn_dropout(attn_outputs) + residual

        # 对 hidden_states 执行 layer_norm，得到 layernorm_output
        layernorm_output = self.norm_2(hidden_states)
        
        # 保存 hidden_states 作为 residual
        residual = hidden_states

        # 调用 self.ffn 进行前向传播，得到 output
        output = self.ffn(layernorm_output, residual)
        outputs = (output,)

        # 如果 use_cache 为真，将 past_key_value 加入到 outputs
        if use_cache:
            outputs += (past_key_value,)

        # 如果 output_attentions 为真，将 attn_weights 加入到 outputs
        if output_attentions:
            outputs += (attn_weights,)

        # 返回 outputs，即 (output,) 或 (output, past_key_value) 或 (output, attn_weights) 或 其组合
        return outputs  # hidden_states, present, attentions


# 定义一个名为 MptPreTrainedModel 的类，继承自 PreTrainedModel 类
class MptPreTrainedModel(PreTrainedModel):
    # 类属性，指定配置类为 MptConfig
    config_class = MptConfig
    # 类属性，指定基础模型前缀为 "transformer"
    base_model_prefix = "transformer"
    # 类属性，支持梯度检查点
    supports_gradient_checkpointing = True
    # 类属性，不对 "MptBlock" 模块进行切分
    _no_split_modules = ["MptBlock"]
    # 类属性，忽略加载时缺失的键值
    _keys_to_ignore_on_load_missing = [r"lm_head.*."
    # 初始化函数，接受任意数量的输入参数和关键字参数，调用父类的初始化函数
    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

    # 初始化模型的权重
    def _init_weights(self, module: nn.Module):
        """Initialize the weights."""
        # 如果是线性层
        if isinstance(module, nn.Linear):
            # 用正态分布初始化权重
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果存在偏置，初始化为0
            if module.bias is not None:
                module.bias.data.zero_()
        # 如果是嵌入层
        elif isinstance(module, nn.Embedding):
            # 用正态分布初始化权重
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果存在填充索引，将对应位置置为0
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        # 如果是 LayerNorm 层
        elif isinstance(module, LayerNorm):
            # 如果存在偏置，初始化为0
            if module.bias is not None:
                module.bias.data.zero_()
            # 权重初始化为1
            module.weight.data.fill_(1.0)

    # 将缓存转换为 MPT 所需的格式
    @staticmethod
    def _convert_to_mpt_cache(
        past_key_value: Tuple[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Converts the cache to the format expected by Mpt, i.e. to tuple(tuple([batch_size * num_heads, ...]))
        """
        # 获取维度信息
        batch_size, num_heads, head_dim, seq_length = past_key_value[0][0].shape
        batch_size_times_num_heads = batch_size * num_heads
        # 格式转换
        return tuple(
            (
                layer_past[0].reshape(batch_size_times_num_heads, head_dim, seq_length),
                layer_past[1].reshape(batch_size_times_num_heads, seq_length, head_dim),
            )
            for layer_past in past_key_value
        )
# MPT_START_DOCSTRING的值为模型文档字符串的起始部分，包含了模型的继承关系、PyTorch Module的描述、参数说明等
MPT_START_DOCSTRING = r"""

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`MptConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

# MPT_INPUTS_DOCSTRING的值为空，作为输入文档字符串的起始部分，可以添加模型输入相关的说明
MPT_INPUTS_DOCSTRING = r"""
    # 参数说明部分
        Args:
            # `input_ids` 是一个形状为 `(batch_size, input_ids_length)` 的 `torch.LongTensor` 对象，
            # `input_ids_length` 是 `sequence_length`，如果没有提供 `past_key_values`，否则为 `past_key_values[0][0].shape[2]`
            # （输入过去键值状态的序列长度）。词汇表中输入序列标记的索引。
            input_ids (`torch.LongTensor` of shape `(batch_size, input_ids_length)`):
                # 如果使用了 `past_key_values`，那么应该只传递还没有计算过的 `input_ids`
                If `past_key_values` is used, only `input_ids` that do not have their past calculated should be passed as
                `input_ids`.
    
                # 索引可以通过 [`AutoTokenizer`] 获取，详见 [`PreTrainedTokenizer.encode`] 和
                # [`PreTrainedTokenizer.__call__`]。
                Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
                [`PreTrainedTokenizer.__call__`] for details.
    
                # 关于输入ID的更多信息可查看链接
                [What are input IDs?](../glossary#input-ids)
            # `past_key_values` 是一个长为 `config.n_layers` 的 `Tuple[Tuple[torch.Tensor]]` 对象：
            # 包含由模型计算的隐藏状态（注意力块中的键和值），可以用来加速序列解码。已经计算过的 `input_ids` 不应再传递，
            # 因为它们已经被计算过。
            past_key_values (`Tuple[Tuple[torch.Tensor]]` of length `config.n_layers`):
                # 每一个元素 `past_key_values` 都是一个元组 (past_key, past_value)：
                # - past_key: [batch_size * num_heads, head_dim, kv_length]
                # - past_value: [batch_size * num_heads, kv_length, head_dim]
                Each element of `past_key_values` is a tuple (past_key, past_value):
            # `attention_mask` 是一个形状为 `(batch_size, sequence_length)` 的 `torch.FloatTensor`，可选：
            # 用于避免对填充标记索引执行注意力操作。遮罩值选在 `[0, 1]`：
            # - 1 表示 **未遮罩** 的标记，
            # - 0 表示 **被遮罩** 的标记。
            attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
                # 关于注意力掩码的更多信息可查看链接
                [What are attention masks?](../glossary#attention-mask)
    
            # `inputs_embeds` 是一个形状为 `(batch_size, sequence_length, hidden_size)` 的 `torch.FloatTensor`，可选：
            # 你可以选择直接传递一个嵌入表示，而不是 `input_ids`。这在你想要更多控制如何将 `input_ids` 索引转换为关联向量时很有用，
            # 而不是使用模型内部的嵌入查找矩阵。
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                # 如果使用 `past_key_values`，可以选择只输入最后的 `inputs_embeds`（见 `past_key_values`）
                If `past_key_values` is used, optionally only the last `inputs_embeds` have to be input (see
                `past_key_values`).
            # `use_cache` 是一个布尔值，可选：
            # 如果设置为 `True`，则返回 `past_key_values` 键值状态，可用于加速解码（见 `past_key_values`）
            use_cache (`bool`, *optional*):
            # `output_attentions` 是一个布尔值，可选：
            # 是否返回所有注意力层的注意力张量。详情请查看返回的张量中的 `attentions`
            output_attentions (`bool`, *optional*):
            # `output_hidden_states` 是一个布尔值，可选：
            # 是否返回所有层的隐藏状态。详情请查看返回的张量中的 `hidden_states`
            output_hidden_states (`bool
# 定义 MptModel 类，继承自 MptPreTrainedModel 类
@add_start_docstrings(
    "The bare Mpt Model transformer outputting raw hidden-states without any specific head on top.",
    MPT_START_DOCSTRING,
)
class MptModel(MptPreTrainedModel):
    # 初始化函数，接受 MptConfig 类型的参数
    def __init__(self, config: MptConfig):
        # 调用父类的初始化函数
        super().__init__(config)

        # 设置隐藏层大小和头数
        self.hidden_size = config.hidden_size
        self.num_heads = config.n_heads

        # 定义词嵌入和 LN 词嵌入
        self.wte = nn.Embedding(config.vocab_size, self.hidden_size)

        # 定义 Transformer 块
        self.blocks = nn.ModuleList([MptBlock(config) for _ in range(config.n_layers)])

        # 定义最终的 Layer Norm
        self.norm_f = LayerNorm(self.hidden_size, eps=config.layer_norm_epsilon)
        # 与 Hub 上的权重兼容
        self.norm_f.bias = None

        # 初始化梯度检查点为 False
        self.gradient_checkpointing = False

        # 初始化权重并应用最终处理
        self.post_init()

    # 返回输入词嵌入
    def get_input_embeddings(self):
        return self.wte

    # 构建 MPT Alibi Tensor
    def build_mpt_alibi_tensor(self, num_heads, sequence_length, alibi_bias_max=8, device=None):
        return build_mpt_alibi_tensor(num_heads, sequence_length, alibi_bias_max, device)

    # 设置输入词嵌入
    def set_input_embeddings(self, new_embeddings: torch.Tensor):
        self.wte = new_embeddings

    # 前向传播函数
    @add_start_docstrings_to_model_forward(MPT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPastAndCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    # 准备生成输入数据的函数
    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor, # 输入ID序列
        past_key_values: Optional[torch.Tensor] = None, # 上次生成的密钥值
        attention_mask: Optional[torch.Tensor] = None, # 注意力掩码
        inputs_embeds: Optional[torch.Tensor] = None, # 输入的词嵌入
        use_cache: Optional[bool] = None, # 是否使用缓存
        **kwargs,
    ) -> dict:
        # 如果有上次生成的密钥值
        if past_key_values is not None:
            # 获取上次生成的序列长度
            past_length = past_key_values[0][0].shape[2]
            
            # 如果当前输入ID序列长度大于上次长度, 则只保留最后的部分
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # 否则只保留最后一个ID
                remove_prefix_length = input_ids.shape[1] - 1

            # 截取输入ID序列
            input_ids = input_ids[:, remove_prefix_length:]

        # 如果提供了输入词嵌入, 且没有上次生成的密钥值
        if inputs_embeds is not None and past_key_values is None:
            # 使用输入词嵌入作为模型输入
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            # 否则使用输入ID序列作为模型输入
            model_inputs = {"input_ids": input_ids}

        # 更新模型输入, 包括上次生成的密钥值、是否使用缓存、注意力掩码
        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    # 前向传播函数
    @add_start_docstrings_to_model_forward(MPT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=CausalLMOutputWithCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None, # 输入ID序列
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None, # 上次生成的密钥值
        attention_mask: Optional[torch.Tensor] = None, # 注意力掩码
        inputs_embeds: Optional[torch.Tensor] = None, # 输入的词嵌入
        labels: Optional[torch.Tensor] = None, # 标签
        use_cache: Optional[bool] = None, # 是否使用缓存
        output_attentions: Optional[bool] = None, # 是否输出注意力权重
        output_hidden_states: Optional[bool] = None, # 是否输出隐藏状态
        return_dict: Optional[bool] = None, # 是否返回字典格式的输出
    ) -> Union[Tuple[torch.Tensor], CausalLMOutputWithCrossAttentions]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional`):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        # 若return_dict未设置，则使用config里的默认值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 使用transformer处理输入数据
        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        # 通过lm_head获取LM预测的logits
        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # 将labels转移到正确的设备以启用模型并行化
            labels = labels.to(lm_logits.device)
            # 移动logits以使得tokens < n 预测n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            batch_size, seq_length, vocab_size = shift_logits.shape
            # 展平tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(batch_size * seq_length, vocab_size), shift_labels.view(batch_size * seq_length)
            )

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

    def _reorder_cache(
        self, past: Tuple[Tuple[torch.Tensor, torch.Tensor], ...], beam_idx: torch.LongTensor
``` 
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], ...]:
        """
        This function is used to re-order the `past_key_values` cache if [`~PreTrainedModel.beam_search`] or
        [`~PreTrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
        beam_idx at every generation step.

        Output shares the same memory storage as `past`.
        """
        # 在调用 [`~PreTrainedModel.beam_search`] 或 [`~PreTrainedModel.beam_sample`] 时，重新排列 `past_key_values` 缓存。
        # 这是为了在每个生成步骤中将 `past_key_values` 与正确的 beam_idx 匹配。

        # 在所有需要这些索引的设备上，获取 `beam_idx` 的副本。
        device_to_beam_idx = {
            past_state.device: beam_idx.to(past_state.device) for layer_past in past for past_state in layer_past
        }
        # 重新排列 `past`，使其与 `beam_idx` 对应。
        reordered_past = tuple(
            (
                layer_past[0].index_select(0, device_to_beam_idx[layer_past[0].device]),
                layer_past[1].index_select(0, device_to_beam_idx[layer_past[0].device]),
            )
            for layer_past in past
        )
        # 返回重新排列后的 `past`
        return reordered_past
# 这是一个包含序列分类任务的 MPT 模型的定义
@add_start_docstrings(
    """
    The MPT Model transformer with a sequence classification head on top (linear layer).

    [`MptForSequenceClassification`] uses the last token in order to do the classification, as other causal models
    (e.g. GPT-1) do.

    Since it does classification on the last token, it requires to know the position of the last token. If a
    `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If
    no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
    padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
    each row of the batch).
    """,
    MPT_START_DOCSTRING,
)
class MptForSequenceClassification(MptPreTrainedModel):
    # 初始化方法，设置配置参数
    def __init__(self, config: MptConfig):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.transformer = MptModel(config)
        self.score = nn.Linear(config.hidden_size, config.num_labels, bias=False)

        # 初始化权重并应用最终处理
        self.post_init()

    # 前向传播方法
    @add_start_docstrings_to_model_forward(MPT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=SequenceClassifierOutputWithPast,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,

# 这是一个包含 token 分类任务的 MPT 模型的定义
@add_start_docstrings(
    """
    MPT Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
    Named-Entity-Recognition (NER) tasks.
    """,
    MPT_START_DOCSTRING,
)
class MptForTokenClassification(MptPreTrainedModel):
    # 初始化方法，设置配置参数
    def __init__(self, config: MptConfig):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.transformer = MptModel(config)
        if hasattr(config, "classifier_dropout") and config.classifier_dropout is not None:
            classifier_dropout = config.classifier_dropout
        elif hasattr(config, "hidden_dropout") and config.hidden_dropout is not None:
            classifier_dropout = config.hidden_dropout
        else:
            classifier_dropout = 0.1
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # 初始化权重并应用最终处理
        self.post_init()

    # 前向传播方法
    @add_start_docstrings_to_model_forward(MPT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TokenClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    # 定义前向传播函数，接收模型输入并返回输出结果
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,  # 输入的 token ID
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,  # 过去的 key-value 对
        attention_mask: Optional[torch.Tensor] = None,  # 注意力掩码
        inputs_embeds: Optional[torch.Tensor] = None,  # 输入的嵌入向量
        labels: Optional[torch.Tensor] = None,  # 标签数据
        use_cache: Optional[bool] = None,  # 是否使用缓存
        output_attentions: Optional[bool] = None,  # 是否输出注意力权重
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态
        return_dict: Optional[bool] = None,  # 是否以字典形式返回结果
        **deprecated_arguments,  # 其他过时参数
    ) -> Union[Tuple[torch.Tensor], TokenClassifierOutput]:  # 返回值的类型注释
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 运行 transformer 模型并获取输出
        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = transformer_outputs[0]  # 获取 transformer 输出中的隐藏状态
        hidden_states = self.dropout(hidden_states)  # 对隐藏状态应用 dropout
        logits = self.classifier(hidden_states)  # 使用分类器获取 logits

        loss = None  # 初始化损失为 None
        if labels is not None:  # 如果存在标签数据
            # 将标签数据移到正确的设备以启用模型并行计算
            labels = labels.to(logits.device)
            batch_size, seq_length = labels.shape  # 获取标签数据的形状
            loss_fct = CrossEntropyLoss()  # 使用交叉熵损失函数
            # 计算损失
            loss = loss_fct(
                logits.view(batch_size * seq_length, self.num_labels),
                labels.view(batch_size * seq_length)
            )

        if not return_dict:  # 如果不返回字典形式的结果
            output = (logits,) + transformer_outputs[2:]  # 构建输出元组
            return ((loss,) + output) if loss is not None else output

        # 返回 TokenClassifierOutput 类型的结果
        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )
# 用于提取式问答任务的 MPT 模型变压器，在隐藏状态的顶部，添加一个用于分类跨度的头部（在隐藏状态的顶部添加线性层，用于计算`起始标记`和`结束标记`）。
# 这是一个引用了 MPT_START_DOCSTRING 的文档字符串
class MptForQuestionAnswering(MptPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        # 初始化 MPT 模型
        self.transformer = MptModel(config)
        # 初始化线性层，输出维度为 2
        self.qa_outputs = nn.Linear(config.hidden_size, 2)

        # 初始化权重并应用最终处理
        self.post_init()

    # 将 MPT_INPUTS_DOCSTRING 的格式应用到模型前向方法上
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        start_positions: Optional[torch.LongTensor] = None,
        end_positions: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, QuestionAnsweringModelOutput]:
        r"""
        start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        """
        # 设置是否返回字典结果，默认为配置文件中的值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 使用transformer处理输入序列
        outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取transformer输出的序列输出
        sequence_output = outputs[0]

        # 使用qa_outputs层来得到logits
        logits = self.qa_outputs(sequence_output)
        # 将logits在最后一个维度分割成start和end logits
        start_logits, end_logits = logits.split(1, dim=-1)
        # 压缩维度并保持连续性
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # 如果在多GPU上运行，则添加一个维度
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # 有时起始/结束位置超出模型输入范围，忽略这些值
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            # 定义交叉熵损失函数
            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        # 返回问题回答模型输出
        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
```