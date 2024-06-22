# `.\transformers\models\xlm_prophetnet\modeling_xlm_prophetnet.py`

```py
# 指定编码为 UTF-8
# 版权声明
# 使用 Apache 许可证 2.0 版本
# 获取许可证的网址
# 如果符合许可证要求，可以使用此文件；否则，禁止使用
""" PyTorch XLM-ProphetNet 模型。"""


# 导入必要的库
import copy
import math
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import Tensor, nn
from torch.nn import LayerNorm

# 导入激活函数映射表和模型输出
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
# 导入 XLM-ProphetNet 的配置类
from .configuration_xlm_prophetnet import XLMProphetNetConfig


# 获取日志记录器
logger = logging.get_logger(__name__)


# 用于文档的配置信息
_CONFIG_FOR_DOC = "XLMProphetNetConfig"

# 预训练模型的存档列表
XLM_PROPHETNET_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "microsoft/xprophetnet-large-wiki100-cased",
    # 查看所有 XLMProphetNet 模型 https://huggingface.co/models?filter=xprophetnet
]

# XLM-ProphetNet 模型文档的起始字符串
XLM_PROPHETNET_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    Original ProphetNet code can be found [here](https://github.com/microsoft/ProphetNet). Checkpoints were converted
    from original Fairseq checkpoints. For more information on the checkpoint conversion, please take a look at the
    file `convert_prophetnet_original_pytorch_checkpoint_to_pytorch.py`.

    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matters related to general usage and
    behavior.

    Parameters:
        config ([`XLMProphetNetConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

# XLM-ProphetNet 模型输入文档字符串
XLM_PROPHETNET_INPUTS_DOCSTRING = r"""
# 这段注释包含了关于 XLM-ProphetNet 模型的输入参数说明。这些输入参数包括:
# 1. input_ids: 输入序列的词汇表索引
# 2. attention_mask: 用于避免在填充标记上进行注意力计算的掩码
# 3. head_mask: 用于nullify选定的编码器注意力模块头部的掩码
# 4. output_attentions: 是否返回所有注意力层的注意力张量
# 5. output_hidden_states: 是否返回所有层的隐藏状态
# 6. return_dict: 是否返回一个 ModelOutput 对象而不是元组

# 这个函数用于计算 softmax 输出, 其中:
# 1. 如果 onnx_trace 为 True, 则使用 nn.functional.softmax 计算 softmax
# 2. 否则, 使用带有 dtype=torch.float32 的 nn.functional.softmax 计算 softmax

# 此函数用于计算 n-gram 注意力偏差, 其中:
# 1. 根据 n-gram 和序列长度计算左右两个注意力偏差块
# 2. 左边的注意力偏差块设置为负无穷大, 右边的注意力偏差块设置为 0 在对角线上
# 3. 最后将左右两个注意力偏差块拼接起来返回

# 这个函数用于计算相对位置编码的 bucket 索引, 具体实现没有给出
# 计算相对位置桶的各个部分
def compute_relative_buckets(num_buckets, max_distance, relative_positions, is_bidirectional=False):
    """
    This function computes individual parts of the relative position buckets. For more detail, see paper.
    """
    # 相对位置的相反数
    inv_relative_positions = -relative_positions
    rel_positions_bucket = 0

    # 如果是双向的，则重新计算 num_buckets，并根据条件计算 rel_positions_bucket
    if is_bidirectional:
        num_buckets = num_buckets // 2
        rel_positions_bucket = (
            rel_positions_bucket
            + torch.lt(inv_relative_positions, torch.zeros_like(inv_relative_positions)).int() * num_buckets
        )
        inv_relative_positions = torch.abs(inv_relative_positions)
    else:
        # 如果不是双向，则将负的相对位置调整为 0
        inv_relative_positions = torch.max(inv_relative_positions, torch.zeros_like(inv_relative_positions))

    max_exact = num_buckets // 2
    # 判断是否在 max_exact 范围内，根据条件进行不同的计算
    is_small = torch.lt(inv_relative_positions, max_exact)
    val_if_large = max_exact + torch.log(inv_relative_positions.float() / max_exact) / math.log(
        max_distance / max_exact
    ) * (num_buckets - max_exact)
    val_if_large = torch.min(val_if_large, torch.ones_like(val_if_large) * (num_buckets - 1)).int()
    rel_positions_bucket = rel_positions_bucket + torch.where(is_small, inv_relative_positions.int(), val_if_large)
    return rel_positions_bucket


# 从 transformers.models.prophetnet.modeling_prophetnet.compute_all_stream_relative_buckets 复制而来
def compute_all_stream_relative_buckets(num_buckets, max_distance, position_ids):
    """
    This function computes both main and predict relative position buckets. For more detail, see paper.
    """
    # 主流位置
    main_stream_relative_positions = position_ids.unsqueeze(1).repeat(1, position_ids.size(-1), 1)
    main_stream_relative_positions = main_stream_relative_positions - position_ids.unsqueeze(-1)

    # 预测流位置
    predicting_stream_relative_positions = torch.cat((position_ids - 1, position_ids), dim=-1).unsqueeze(1)
    predicting_stream_relative_positions = predicting_stream_relative_positions.repeat(1, position_ids.size(-1), 1)
    predicting_stream_relative_positions = predicting_stream_relative_positions - position_ids.unsqueeze(-1)

    # 计算主要和预测相对位置桶
    main_relative_position_buckets = compute_relative_buckets(
        num_buckets, max_distance, main_stream_relative_positions, is_bidirectional=False
    )
    predict_relative_position_buckets = compute_relative_buckets(
        num_buckets, max_distance, predicting_stream_relative_positions, is_bidirectional=False
    )
    return main_relative_position_buckets, predict_relative_position_buckets


@dataclass
# 从 transformers.models.prophetnet.modeling_prophetnet.ProphetNetSeq2SeqLMOutput 复制而来，仅将 ProphetNet->XLMProphetNet 全部大写
class XLMProphetNetSeq2SeqLMOutput(ModelOutput):
    """
    Base class for sequence-to-sequence language models outputs.

    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    logits_ngram: Optional[torch.FloatTensor] = None
    # 初始化过去的键值（以 torch.FloatTensor 类型存储），初始值为 None
    past_key_values: Optional[Tuple[torch.FloatTensor]] = None
    # 初始化解码器隐藏状态（以 torch.FloatTensor 类型存储），初始值为 None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # 初始化解码器 n-gram 隐藏状态（以 torch.FloatTensor 类型存储），初始值为 None
    decoder_ngram_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # 初始化解码器注意力权重（以 torch.FloatTensor 类型存储），初始值为 None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    # 初始化解码器 n-gram 注意力权重（以 torch.FloatTensor 类型存储），初始值为 None
    decoder_ngram_attentions: Optional[Tuple[torch.FloatTensor]] = None
    # 初始化交叉注意力权重（以 torch.FloatTensor 类型存储），初始值为 None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    # 初始化编码器最后的隐藏状态（以 torch.FloatTensor 类型存储），初始值为 None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    # 初始化编码器隐藏状态（以 torch.FloatTensor 类型存储的元组），初始值为 None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # 初始化编码器注意力权重（以 torch.FloatTensor 类型存储的元组），初始值为 None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    
    # 定义属性方法，返回解码器的交叉注意力权重
    @property
    def decoder_cross_attentions(self):
        # 发出警告，说明 `decoder_cross_attentions` 属性已经弃用，并将很快移除。建议使用 `cross_attentions` 属性代替
        warnings.warn(
            "`decoder_cross_attentions` is deprecated and will be removed soon. Please use `cross_attentions`"
            " instead.",
            FutureWarning,
        )
        # 返回解码器的交叉注意力权重
        return self.cross_attentions
# 使用dataclass装饰器定义一个名为XLMProphetNetSeq2SeqModelOutput的类，作为ModelOutput的子类
class XLMProphetNetSeq2SeqModelOutput(ModelOutput):
    """
    model encoder的输出基类，也包含了可以加速顺序解码的预计算隐藏状态。
    """

    last_hidden_state: torch.FloatTensor
    last_hidden_state_ngram: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Tuple[torch.FloatTensor]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_ngram_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    decoder_ngram_attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None

    @property
    def decoder_cross_attentions(self):
        warnings.warn(
            "`decoder_cross_attentions` is deprecated and will be removed soon. Please use `cross_attentions`"
            " instead.",
            FutureWarning,
        )
        return self.cross_attentions


# 使用dataclass装饰器定义一个名为XLMProphetNetDecoderModelOutput的类，作为ModelOutput的子类
class XLMProphetNetDecoderModelOutput(ModelOutput):
    """
    model输出的基类，也可能包含过去的键/值（用于加速顺序解码）。
    """

    last_hidden_state: torch.FloatTensor
    last_hidden_state_ngram: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Tuple[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    hidden_states_ngram: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    ngram_attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None


# 使用dataclass装饰器定义一个名为XLMProphetNetDecoderLMOutput的类，作为ModelOutput的子类
class XLMProphetNetDecoderLMOutput(ModelOutput):
    """
    model输出的基类，也可能包含过去的键/值（用于加速顺序解码）。
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    logits_ngram: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Tuple[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    hidden_states_ngram: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    ngram_attentions: Optional[Tuple[torch.FloatTensor]] = None
    # 定义一个名为cross_attentions的变量，类型为可选的元组，元组内包含一个torch.FloatTensor类型的值，默认为None
# 从 transformers.models.prophetnet.modeling_prophetnet.ProphetNetPreTrainedModel 复制代码，并将 ProphetNet 改为 XLMProphetNet
class XLMProphetNetPreTrainedModel(PreTrainedModel):
    # 设置配置类为 XLMProphetNetConfig
    config_class = XLMProphetNetConfig
    # 设置基础模型前缀为 "prophetnet"
    base_model_prefix = "prophetnet"
    # 支持梯度检查点
    supports_gradient_checkpointing = True

    # 初始化权重函数
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # 初始化 Linear 层权重
            module.weight.data.normal_(mean=0.0, std=self.config.init_std)
            if module.bias is not None:
                # 若有偏置，则初始化为零
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            # 初始化嵌入层权重
            module.weight.data.normal_(mean=0.0, std=self.config.init_std)
            if module.padding_idx is not None:
                # 若有 padding_idx，则将对应位置设置为零
                module.weight.data[module.padding_idx].zero_()

    # 右移输入函数
    def _shift_right(self, input_ids):
        # 获取decoder起始标记ID和pad标记ID
        decoder_start_token_id = self.config.decoder_start_token_id
        pad_token_id = self.config.pad_token_id

        assert decoder_start_token_id is not None, (
            "self.model.config.decoder_start_token_id has to be defined. In XLMProphetNet it is usually set to the"
            " pad_token_id. See XLMProphetNet docs for more information"
        )

        # 将输入向右移动
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
        shifted_input_ids[..., 0] = decoder_start_token_id

        assert pad_token_id is not None, "self.model.config.pad_token_id has to be defined."
        # 将标签中可能存在的-100值替换为`pad_token_id`
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

        assert torch.all(shifted_input_ids >= 0).item(), "Verify that `shifted_input_ids` has only positive values"

        return shifted_input_ids

# 从 transformers.models.prophetnet.modeling_prophetnet.ProphetNetPositionalEmbeddings 复制代码，并将 ProphetNet 改为 XLMProphetNet
class XLMProphetNetPositionalEmbeddings(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size. Padding ids are ignored by either offsetting
    based on padding_idx or by setting padding_idx to None and ensuring that the appropriate position ids are passed to
    the forward function.
    """

    def __init__(self, config: XLMProphetNetConfig) -> None:
        # 设置最大长度为��置的最大位置嵌入长度
        self.max_length = config.max_position_embeddings
        # 调用父类初始化方法
        super().__init__(config.max_position_embeddings, config.hidden_size, config.pad_token_id)
    # 前向传播方法，接受输入形状、设备、注意力掩码、过去键值对、位置 ID
    def forward(self, inputs_shape, device, attention_mask=None, past_key_values=None, position_ids=None):
        # 确保 position_ids 为空或 padding_idx 未设置
        assert (position_ids is None) or (
            self.padding_idx is None
        ), "If position_ids is pre-computed then padding_idx should not be set."

        # 如果 position_ids 未设置
        if position_ids is None:
            # 如果 past_key_values 存在，说明是在解码单个步骤
            if past_key_values is not None:
                # 获取之前输入的数量和当前输入的数量
                prev_num_input_ids = past_key_values[0][0].shape[2]
                num_input_ids = inputs_shape[1] + prev_num_input_ids
                # 计算 position_ids，使用 padding_idx 加上总输入数量
                position_ids = torch.ones((1, 1), dtype=torch.long, device=device) * (
                    int(self.padding_idx + num_input_ids)
                )
            else:
                # 如果 attention_mask 未设置，则默认为全 1 的掩码
                if attention_mask is None:
                    attention_mask = torch.ones(inputs_shape, dtype=torch.long, device=device)

                # 从 input_ids 或 attention_mask 中获取 position_ids
                position_ids = (
                    torch.cumsum(attention_mask, dim=1).type_as(attention_mask) * attention_mask
                ).long() + self.padding_idx

                # 确保 position_ids 不超过最大长度
                position_ids = position_ids.clamp(0, self.max_length - 1)

        # 调用父类的 forward 方法并返回结果
        return super().forward(position_ids), position_ids

    # 私有的前向传播方法，接受 position_ids 参数
    def _forward(self, position_ids):
        # 调用父类的 forward 方法并返回结果
        return super().forward(position_ids)
# 从transformers.models.prophetnet.modeling_prophetnet.ProphetNetAttention复制并修改为XLMProphetNetAttention
class XLMProphetNetAttention(nn.Module):
    """来自'Attention Is All You Need'论文的多头注意力机制"""

    def __init__(
        self,
        config: XLMProphetNetConfig,  # 使用XLMProphetNet的配置
        num_attn_heads: int,  # 注意力头的数量
    ):
        super().__init__()
        hidden_size = config.hidden_size

        self.attention_dropout = config.attention_dropout  # 注意力层的dropout率
        self.dropout = config.dropout  # dropout率
        self.num_attn_heads = num_attn_heads  # 注意力头的数量
        self.head_dim = hidden_size // num_attn_heads  # 每个注意力头的维度

        assert self.head_dim * num_attn_heads == hidden_size, (
            "`config.hidden_size`必须可以被`config.num_encoder_attention_heads`和"
            "`config.num_decoder_attention_heads`整除"
        )

        # 对键、值和查询进行线性投影
        self.key_proj = nn.Linear(hidden_size, hidden_size)
        self.value_proj = nn.Linear(hidden_size, hidden_size)
        self.query_proj = nn.Linear(hidden_size, hidden_size)

        # 输出投影层
        self.out_proj = nn.Linear(hidden_size, hidden_size)

    # 调整张量的形状
    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_attn_heads, self.head_dim).transpose(1, 2).contiguous()

    # 前向传播函数
    def forward(
        self,
        hidden_states,  # 输入的隐藏状态
        key_value_states: Optional[Tensor] = None,  # 键值状态（可选）
        attention_mask: Optional[Tensor] = None,  # 注意力掩码（可选）
        layer_head_mask: Optional[Tensor] = None,  # 层头掩码（可选）
        past_key_value: Optional[Tuple[Tensor]] = None,  # 上一次的键值对（可选）
        output_attentions: bool = False,  # 是否输出注意力权重（默认为False）



# 从transformers.models.prophetnet.modeling_prophetnet.ProphetNetFeedForward复制并修改为XLMProphetNetFeedForward
class XLMProphetNetFeedForward(nn.Module):
    """
    这是基于原始Transformer实现的残差式两层前馈网络块。
    """

    def __init__(self, config: XLMProphetNetConfig, ffn_dim: int):
        super().__init__()
        self.activation_fn = ACT2FN[config.activation_function]  # 激活函数
        self.intermediate = nn.Linear(config.hidden_size, ffn_dim)  # 中间线性层
        self.output = nn.Linear(ffn_dim, config.hidden_size)  # 输出线性层
        self.activation_dropout = config.activation_dropout  # 激活函数的dropout率
        self.dropout = config.dropout  # dropout率

    # 前向传播函数
    def forward(self, hidden_states):  # 输入的隐藏状态
        hidden_states = self.intermediate(hidden_states)  # 中间线性层的计算
        hidden_states = self.activation_fn(hidden_states)  # 应用激活函数

        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)  # 应用激活函数的dropout
        hidden_states = self.output(hidden_states)  # 输出线性层的计算
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)  # 输出dropout
        return hidden_states


# 从transformers.models.prophetnet.modeling_prophetnet.ProphetNetNgramSelfAttention复制并修改为XLMProphetNetNgramSelfAttention
class XLMProphetNetNgramSelfAttention(nn.Module):
    # 初始化函数，接受一个XLMProphetNetConfig对象作为配置参数
    def __init__(self, config: XLMProphetNetConfig):
        # 调用父类的初始化函数
        super().__init__()
        # 从配置参数中获取并设置隐藏层大小
        self.hidden_size = config.hidden_size
    
        # 从配置参数中获取并设置桶的数量
        self.num_buckets = config.num_buckets
        # 从配置参数中获取并设置相对最大距离
        self.relative_max_distance = config.relative_max_distance
        # 从配置参数中获取并设置注意力头的数量
        self.num_attn_heads = config.num_decoder_attention_heads
        # 从配置参数中获取并设置全连接层的丢弃率
        self.dropout = config.dropout
        # 从配置参数中获取并设置注意力机制的丢弃率
        self.attention_dropout = config.attention_dropout
        # 计算每个注意力头的维度
        self.head_dim = config.hidden_size // self.num_attn_heads
        # 从配置参数中获取并设置ngram值
        self.ngram = config.ngram
    
        # 断言确保每个注意力头的维度乘以注意力头的数量等于隐藏层大小
        assert (
            self.head_dim * self.num_attn_heads == config.hidden_size
        ), "config.hidden_size must be divisible by num_attn_heads"
        
        # 创建键的线性投影层
        self.key_proj = nn.Linear(config.hidden_size, config.hidden_size)
        # 创建值的线性投影层
        self.value_proj = nn.Linear(config.hidden_size, config.hidden_size)
        # 创建查询的线性投影层
        self.query_proj = nn.Linear(config.hidden_size, config.hidden_size)
    
        # 创建输出的线性投影层
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)
    
        # 创建相对位置编码的线性层
        self.relative_pos_embeddings = nn.Linear(config.hidden_size, self.num_buckets * self.num_attn_heads)
    
        # 设置onnx运行时的跟踪标志
        self.onnx_trace = False
    
    # 对张量进行重塑，用于形状转换
    def _shape(self, tensor, seq_len, batch_size):
        return tensor.view(batch_size, seq_len, self.num_attn_heads, self.head_dim).transpose(1, 2).contiguous()
    
    # 为onnx导出做准备
    def prepare_for_onnx_export_(self):
        self.onnx_trace = True
    
    # 前向传播函数
    def forward(
        self,
        hidden_states,
        past_key_value: Optional[Tuple[Tensor]] = None,
        attention_mask=None,
        layer_head_mask=None,
        extended_predict_attention_mask=None,
        main_relative_position_buckets=None,
        predict_relative_position_buckets=None,
        position_ids=None,
    # 获取主要相对位置编码的函数
    def get_main_relative_pos_embeddings(
        self, hidden_states, attn_weights, position_ids, main_relative_position_buckets
    ):
        # input hidden_states [batch_size, sequence_length, hidden_size]
        # input attn_weights [batch_size, num_heads, sequence_length, sequence_length]
        # input position_ids [batch_size, sequence_length] or [1,1]
        # 从输入的注意力权重张量中获取 batch_size, num_heads, tgt_len, src_len
        batch_size, num_attn_heads, tgt_len, src_len = attn_weights.shape
        # 重新调整注意力权重张量的形状
        attn_weights = attn_weights.view(batch_size, num_attn_heads, tgt_len, src_len)
        if main_relative_position_buckets is None:
            batch_size, sequence_length = hidden_states.shape[:2]
            # 计算相对位置
            relative_positions = (
                torch.arange(1, attn_weights.shape[-1] + 1)
                .unsqueeze(0)
                .unsqueeze(0)
                .repeat(batch_size, sequence_length, 1)
                .to(position_ids.device)
            )
            # 计算相对位置之间的距离
            relative_positions = relative_positions - position_ids.unsqueeze(0).repeat(batch_size, sequence_length, 1)
            # 根据相对位置计算相对桶
            main_relative_position_buckets = compute_relative_buckets(
                self.num_buckets, self.relative_max_distance, relative_positions, False
            )

        # 计算相对位置嵌入
        rel_pos_embeddings = self.relative_pos_embeddings(hidden_states)
        rel_pos_embeddings = rel_pos_embeddings.view(
            rel_pos_embeddings.shape[:2] + (self.num_buckets, self.num_attn_heads)
        )
        rel_pos_embeddings = rel_pos_embeddings.permute(0, 3, 1, 2)
        # 重新调整相对位置嵌入的形状
        rel_pos_embeddings = rel_pos_embeddings.reshape(attn_weights.shape[:3] + (-1,))

        main_relative_position_buckets = main_relative_position_buckets.repeat(1, self.num_attn_heads, 1)
        # 重新调整主要相对位置桶的形状
        main_relative_position_buckets = main_relative_position_buckets.view(
            -1, main_relative_position_buckets.shape[-1]
        )
        main_relative_position_buckets = main_relative_position_buckets.long()
        # 重新调整相对位置嵌入的形状
        rel_pos_embeddings = rel_pos_embeddings.reshape(-1, rel_pos_embeddings.size(-1))

        # 选择出主要相对位置嵌入
        main_relative_pos_embeddings = torch.gather(rel_pos_embeddings, dim=1, index=main_relative_position_buckets)
        main_relative_pos_embeddings = main_relative_pos_embeddings.view(batch_size, num_attn_heads, tgt_len, -1)
        return main_relative_pos_embeddings

    def get_predict_relative_pos_embeddings(
        self, hidden_states, attn_weights, position_ids, predict_relative_position_buckets
        # 计算隐藏状态的批量大小和序列长度
        # 隐藏状态的维度是 [batch_size, sequence_length, ngram, hidden_size]
        batch_size, sequence_length = hidden_states.shape[0:2]

        # 检查是否没有预测相对位置桶
        if predict_relative_position_buckets is None:
            # 获取注意力权重的序列长度
            key_sequence_length = attn_weights.shape[-1]
            # 确保位置 ID 是正确的，应该是 1 2 3 4 5 ... (key_sequence_length - 1)
            assert (
                position_ids[0][0] == key_sequence_length - 1
            ), "`position_ids` are incorrect. They should be of the format 1 2 3 4 5 ... (key_sequence_length - 1)"

            # 计算相对位置
            relative_positions = (
                torch.arange(0, key_sequence_length)
                .unsqueeze(0)
                .unsqueeze(0)
                .repeat(batch_size, sequence_length, 1)
                .to(position_ids.device)
            )

            # 根据位置 ID 调整相对位置
            relative_positions = relative_positions - position_ids.unsqueeze(0).repeat(batch_size, sequence_length, 1)

            # 计算预测相对位置桶
            predict_relative_position_buckets = compute_relative_buckets(
                self.num_buckets, self.relative_max_distance, relative_positions, False
            )

        # 调整隐藏状态的维度
        # [batch_size, ngram, sequence_length, hidden_size]
        hidden_states = hidden_states.transpose(1, 2)

        # 计算相对位置的嵌入
        rel_pos_embeddings = self.relative_pos_embeddings(hidden_states)

        # 调整相对位置嵌入的形状
        # [batch_size, ngram, sequence_length, num_buckets, num_heads]
        rel_pos_embeddings = rel_pos_embeddings.view(
            hidden_states.shape[:-1] + (self.num_buckets, self.num_attn_heads)
        )
        rel_pos_embeddings = rel_pos_embeddings.permute(0, 2, 1, 4, 3)

        # 调整相对位置嵌入的形状
        # [batch_size * ngram * sequence_length * num_heads, num_buckets]
        rel_pos_embeddings = rel_pos_embeddings.reshape(-1, self.num_buckets)

        # 重复预测相对位置桶以匹配维度
        predict_relative_position_buckets = predict_relative_position_buckets.unsqueeze(0)
        predict_relative_position_buckets = predict_relative_position_buckets.repeat(
            self.ngram, 1, self.num_attn_heads, 1
        )

        # 调整预测相对位置桶的形状
        # [ngram * batch_size * num_heads * sequence_length, -1]
        predict_relative_position_buckets = predict_relative_position_buckets.view(
            -1, predict_relative_position_buckets.size(-1)
        ).long()

        # 提取相对位置嵌入
        predict_relative_pos_embeddings = torch.gather(
            rel_pos_embeddings, dim=1, index=predict_relative_position_buckets
        )

        # 调整预测相对位置嵌入的形状
        # [batch_size, gram, num_heads, sequence_length, -1]
        predict_relative_pos_embeddings = predict_relative_pos_embeddings.view(
            batch_size, self.ngram, self.num_attn_heads, sequence_length, -1
        )

        # 返回预测相对位置嵌入
        return predict_relative_pos_embeddings
# XLMProphetNetEncoderLayer 类实现了编码器层的功能
class XLMProphetNetEncoderLayer(nn.Module):
    """
    Encoder block for XLMProphetnet
    """

    def __init__(self, config: XLMProphetNetConfig):
        super().__init__()
        # 初始化第一个残差块
        # 第一个残差块包含自注意力机制
        self.self_attn = XLMProphetNetAttention(config, config.num_encoder_attention_heads)
        self.self_attn_layer_norm = LayerNorm(config.hidden_size)

        # 初始化第二个残差块
        # 第二个残差块包含前馈神经网络
        self.feed_forward = XLMProphetNetFeedForward(config, config.encoder_ffn_dim)
        self.feed_forward_layer_norm = LayerNorm(config.hidden_size)

    def forward(
        self,
        hidden_states,
        attention_mask,
        layer_head_mask,
        output_attentions: bool = False,
    ):
        # 第一个残差块
        # 计算自注意力输出，返回注意力权重
        attention_output, attn_weights, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        # 将自注意力输出与原始输入相加并标准化
        hidden_states = self.self_attn_layer_norm(attention_output + hidden_states)

        # 第二个残差块
        # 计算前馈神经网络输出
        feed_forward_output = self.feed_forward(hidden_states)
        # 将前馈神经网络输出与原始输入相加并标准化
        hidden_states = self.feed_forward_layer_norm(feed_forward_output + hidden_states)

        # 返回最终的隐藏状态
        outputs = (hidden_states,)

        # 如果需要输出注意力权重，则将其添加到输出中
        if output_attentions:
            outputs += (attn_weights,)

        return outputs


# XLMProphetNetDecoderLayer 类实现了解码器层的功能
class XLMProphetNetDecoderLayer(nn.Module):
    """
    Decoder block for XLMProphetnet
    """

    def __init__(self, config: XLMProphetNetConfig):
        super().__init__()
        # 初始化第一个残差块
        # 第一个残差块包含 N-gram 自注意力机制
        self.self_attn = XLMProphetNetNgramSelfAttention(config)
        self.self_attn_layer_norm = LayerNorm(config.hidden_size)

        # 初始化第二个残差块
        # 第二个残差块包含交叉注意力机制
        if config.add_cross_attention:
            self.cross_attn = XLMProphetNetAttention(config, config.num_decoder_attention_heads)
            self.cross_attn_layer_norm = LayerNorm(config.hidden_size)

        # 初始化第三个残差块
        # 第三个残差块包含前馈神经网络
        self.feed_forward = XLMProphetNetFeedForward(config, config.decoder_ffn_dim)
        self.feed_forward_layer_norm = LayerNorm(config.hidden_size)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attn_mask=None,
        layer_head_mask=None,
        cross_attn_layer_head_mask=None,
        extended_predict_attention_mask=None,
        main_relative_position_buckets=None,
        predict_relative_position_buckets=None,
        position_ids=None,
        past_key_value=None,
        use_cache: bool = True,
        output_attentions: bool = False,
    ):
        # 1st residual block
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        # 进行自注意力机制计算，包括 ngram 及 self-attention，返回输出、权重和键值对
        ngram_attention_output, self_attn_weights, self_attn_weights_ngram, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=self_attn_past_key_value,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            extended_predict_attention_mask=extended_predict_attention_mask,
            main_relative_position_buckets=main_relative_position_buckets,
            predict_relative_position_buckets=predict_relative_position_buckets,
            position_ids=position_ids,
        )
        hidden_states = self.self_attn_layer_norm(hidden_states + ngram_attention_output)

        # cross_attn cached key/values tuple is at positions 3,4 of present_key_value tuple
        cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
        cross_attn_weights = None
        if encoder_hidden_states is not None:
            # 2nd residual block
            # 进行跨注意力机制计算，与编码器隐藏状态进行注意力交互，返回输出、权重和键值对
            attention_output, cross_attn_weights, cross_attn_present_key_value = self.cross_attn(
                hidden_states=hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attn_mask,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=cross_attn_past_key_value,
                output_attentions=output_attentions,
            )
            hidden_states = self.cross_attn_layer_norm(attention_output + hidden_states)

            # add cross-attn to positions 3,4 of present_key_value tuple
            present_key_value = present_key_value + cross_attn_present_key_value

        # 3rd residual block
        # 经过前馈神经网络 (feed-forward) 处理得到输出
        feed_forward_output = self.feed_forward(hidden_states)
        hidden_states = self.feed_forward_layer_norm(feed_forward_output + hidden_states)

        # 将结果存入输出元组
        outputs = (hidden_states,)

        # 如果需要输出注意力权重信息，将相应的权重信息存入输出元组
        if output_attentions:
            outputs += (self_attn_weights, self_attn_weights_ngram, cross_attn_weights)

        # 如果需要缓存过去的键值对信息，将其存入输出元组
        if use_cache:
            outputs += (present_key_value,)

        # 返回处理后的结果
        return outputs
# 在XLMProphetNetEncoder类上方添加文档字符串
@add_start_docstrings(
    "The standalone encoder part of the XLMProphetNetModel.",
    XLM_PROPHETNET_START_DOCSTRING,
)
# 从transformers.models.prophetnet.modeling_prophetnet.ProphetNetEncoder复制到XLMProphetNet，microsoft/prophetnet-large-uncased复制到patrickvonplaten/xprophetnet-large-uncased-standalone, ProphetNet复制到XLMProphetNet, PROPHETNET复制到XLM_PROPHETNET
class XLMProphetNetEncoder(XLMProphetNetPreTrainedModel):
    r"""
    word_embeddings  (`torch.nn.Embeddings` of shape `(config.vocab_size, config.hidden_size)`, *optional*):
        The word embedding parameters. This can be used to initialize [`XLMProphetNetEncoder`] with pre-defined word
        embeddings instead of randomly initialized word embeddings.
    """

    def __init__(self, config: XLMProphetNetConfig, word_embeddings: nn.Embedding = None):
        super().__init__(config)

        # 如果word_embeddings不为None，则使用给定的word_embeddings参数，否则随机初始化word_embeddings
        self.word_embeddings = (
            word_embeddings
            if word_embeddings is not None
            else nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        )
        # 初始化位置嵌入
        self.position_embeddings = XLMProphetNetPositionalEmbeddings(config)
        # 初始化嵌入层的LayerNorm
        self.embeddings_layer_norm = LayerNorm(config.hidden_size)

        # 初始化编码层
        self.layers = nn.ModuleList([XLMProphetNetEncoderLayer(config) for _ in range(config.num_encoder_layers)])

        # 梯度检查点设为False
        self.gradient_checkpointing = False
        # 初始化权重并应用最终处理
        self.post_init()

    # 获取输入嵌入
    def get_input_embeddings(self):
        return self.word_embeddings

    # 设置输入嵌入
    def set_input_embeddings(self, value):
        self.word_embeddings = value

    @add_start_docstrings_to_model_forward(XLM_PROPHETNET_STANDALONE_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,



# 在XLMProphetNetDecoder类上方添加文档字符串
@add_start_docstrings(
    "The standalone decoder part of the XLMProphetNetModel.",
    XLM_PROPHETNET_START_DOCSTRING,
)
# 从transformers.models.prophetnet.modeling_prophetnet.ProphetNetDecoder复制到XLMProphetNet，microsoft/prophetnet-large-uncased复制到patrickvonplaten/xprophetnet-large-uncased-standalone, ProphetNet复制到XLMProphetNet, PROPHETNET复制到XLM_PROPHETNET
class XLMProphetNetDecoder(XLMProphetNetPreTrainedModel):
    r"""
    word_embeddings  (`torch.nn.Embeddings` of shape `(config.vocab_size, config.hidden_size)`, *optional*):
        The word embedding parameters. This can be used to initialize [`XLMProphetNetEncoder`] with pre-defined word
        embeddings instead of randomly initialized word embeddings.
    """
    # 初始化方法，接受 XLMProphetNetConfig 对象和可选的词嵌入参数
    def __init__(self, config: XLMProphetNetConfig, word_embeddings: Optional[nn.Embedding] = None):
        # 调用父类的初始化方法
        super().__init__(config)

        # 从配置中获取参数，并赋值给对应的实例变量
        self.ngram = config.ngram
        self.num_buckets = config.num_buckets
        self.relative_max_distance = config.relative_max_distance
        self.dropout = config.dropout
        self.max_target_positions = config.max_position_embeddings

        # 根据是否提供词嵌入参数，初始化 word_embeddings 实例变量
        self.word_embeddings = (
            word_embeddings
            if word_embeddings is not None
            else nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        )
        # 初始化位置嵌入实例
        self.position_embeddings = XLMProphetNetPositionalEmbeddings(config)

        # 初始化 ngram 嵌入实例
        self.ngram_embeddings = nn.Embedding(self.ngram, config.hidden_size, None)
        # 初始化一系列解码器层
        self.layers = nn.ModuleList([XLMProphetNetDecoderLayer(config) for _ in range(config.num_decoder_layers)])
        # 初始化嵌入层的 LayerNorm
        self.embeddings_layer_norm = LayerNorm(config.hidden_size)

        # 初始化梯度检查
        self.gradient_checkpointing = False
        # 初始化权重并应用最终处理
        self.post_init()

    # 获取输入词嵌入
    def get_input_embeddings(self):
        return self.word_embeddings

    # 设置输入词嵌入
    def set_input_embeddings(self, value):
        self.word_embeddings = value

    # 前向传播方法，接受一系列输入参数
    @add_start_docstrings_to_model_forward(XLM_PROPHETNET_STANDALONE_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=XLMProphetNetDecoderModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    # 计算缓冲的相对存储桶
    def compute_buffered_relative_buckets(self, position_ids):
        # 获取批次大小和序列长度
        batch_size, sequence_length = position_ids.shape
    
        # 创建相对位置 ID，范围是 1 到 self.max_target_positions
        position_ids = torch.arange(1, self.max_target_positions).to(position_ids.device).repeat(1, 1)
        # 计算主要和预测的相对存储桶
        main_relative_buckets, predict_relative_buckets = compute_all_stream_relative_buckets(
            self.num_buckets, self.relative_max_distance, position_ids
        )
    
        # 缓冲相对存储桶
        # 主要相对存储桶: 截取序列长度部分，并重复到批次大小
        main_relative_buckets = main_relative_buckets[:, :sequence_length, :sequence_length].repeat(batch_size, 1, 1)
        # 预测相对存储桶: 截取序列长度部分，并在最后一个维度上拼接预测范围，然后重复到批次大小
        predict_relative_buckets = torch.cat(
            [
                predict_relative_buckets[:, :sequence_length, :sequence_length],
                predict_relative_buckets[
                    :, :sequence_length, self.max_target_positions : self.max_target_positions + sequence_length
                ],
            ],
            2,
        ).repeat(batch_size, 1, 1)
    
        return main_relative_buckets, predict_relative_buckets
    
    # 准备注意力掩码
    def prepare_attention_mask(self, hidden_states, attention_mask):
        # 获取批次大小和序列长度
        batch_size, seq_length = hidden_states.shape[:2]
    
        # 创建因果掩码
        causal_mask = torch.full(
            (seq_length, seq_length),
            torch.finfo(hidden_states.dtype).min,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        causal_mask = torch.triu(causal_mask, 1)
    
        # 扩展因果掩码到批次大小和注意力头数
        extended_causal_mask = causal_mask[:seq_length, :seq_length][None, None, :, :].expand(
            (batch_size, self.config.num_decoder_attention_heads) + causal_mask.shape
        )
    
        # 添加常规注意力掩码
        if attention_mask is not None:
            # 扩展注意力掩码
            extended_attention_mask = (1.0 - attention_mask[:, None, None, :]) * torch.finfo(self.dtype).min
            # 将扩展的因果掩码和扩展的注意力掩码相加
            extended_attention_mask = extended_causal_mask + extended_attention_mask
        else:
            extended_attention_mask = extended_causal_mask
        # 将注意力掩码转换为隐藏状态的数据类型
        return extended_attention_mask.to(hidden_states.dtype)
    # 准备预测时的注意力遮罩，根据隐藏状态和注意力遮罩
    def prepare_predict_attention_mask(self, hidden_states, attention_mask):
        # 获取隐藏状态的批处理大小和序列长度
        batch_size, seq_length = hidden_states.shape[:2]

        # 获取因果遮罩
        predict_causal_mask = ngram_attention_bias(
            self.max_target_positions, self.ngram, hidden_states.device, hidden_states.dtype
        )
        # 将因果遮罩拼接在一起，分为两部分
        predict_causal_mask = torch.cat(
            [
                predict_causal_mask[:, :seq_length, :seq_length],
                predict_causal_mask[
                    :, :seq_length, self.max_target_positions : self.max_target_positions + seq_length
                ],
            ],
            dim=-1,
        )
        extended_predict_causal_mask = predict_causal_mask[None, None, :, :, :].expand(
            (batch_size, self.config.num_decoder_attention_heads) + predict_causal_mask.shape
        )

        # 添加常规的注意力遮罩
        if attention_mask is not None:
            extended_attention_mask = (1.0 - attention_mask[:, None, None, None, :]) * torch.finfo(self.dtype).min
            extended_attention_mask = extended_attention_mask.expand(
                (batch_size, self.config.num_decoder_attention_heads, self.ngram, seq_length, seq_length)
            )
            # 预测的流的注意力遮罩应该始终为0
            extended_attention_mask = torch.cat(
                [extended_attention_mask, torch.zeros_like(extended_attention_mask)], dim=-1
            )
            extended_predict_attention_mask = extended_predict_causal_mask + extended_attention_mask
        else:
            extended_predict_attention_mask = extended_predict_causal_mask
        # 将结果转换为隐藏状态的数据类型并返回
        return extended_predict_attention_mask.to(hidden_states.dtype)
# 导入函数和类装饰器，用于添加模型文档字符串
from transformers.file_utils import add_start_docstrings, add_start_docstrings_to_model_forward, replace_return_docstrings
from transformers.models.prophetnet.modeling_prophetnet import XLMProphetNetPreTrainedModel, XLMProphetNetConfig, XLM_PROPHETNET_START_DOCSTRING, XLM_PROPHETNET_INPUTS_DOCSTRING, XLMProphetNetSeq2SeqModelOutput, _CONFIG_FOR_DOC, XLMProphetNetEncoder, XLMProphetNetDecoder
import torch
import torch.nn as nn
import copy
from typing import Optional, Tuple

@add_start_docstrings(
    "The bare XLMProphetNet Model outputting raw hidden-states without any specific head on top.",  # 添加模型的基础文档字符串
    XLM_PROPHETNET_START_DOCSTRING,  # 添加 XLMProphetNet 特有的文档字符串
)
# 定义 XLMProphetNetModel 类，继承自 XLMProphetNetPreTrainedModel 类
class XLMProphetNetModel(XLMProphetNetPreTrainedModel):
    # 定义需要共享权重的键值列表
    _tied_weights_keys = ["encoder.word_embeddings.weight", "decoder.word_embeddings.weight"]

    # 初始化方法
    def __init__(self, config: XLMProphetNetConfig):
        # 调用父类初始化方法
        super().__init__(config)
        # 初始化词嵌入层
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)

        # 克隆编码器配置，并设置为非编码-解码器模式，关闭缓存
        encoder_config = copy.deepcopy(config)
        encoder_config.is_encoder_decoder = False
        encoder_config.use_cache = False
        # 初始化编码器
        self.encoder = XLMProphetNetEncoder(encoder_config, self.word_embeddings)

        # 克隆解码器配置，并设置为解码器模式，非编码-解码器模式，关闭缓存
        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        # 初始化解码器
        self.decoder = XLMProphetNetDecoder(decoder_config, self.word_embeddings)

        # 初始化权重并应用最终处理
        self.post_init()

    # 获取输入词嵌入
    def get_input_embeddings(self):
        return self.word_embeddings

    # 设置输入词嵌入
    def set_input_embeddings(self, value):
        self.word_embeddings = value
        self.encoder.word_embeddings = self.word_embeddings
        self.decoder.word_embeddings = self.word_embeddings

    # 绑定权重
    def _tie_weights(self):
        if self.config.tie_word_embeddings:
            self._tie_or_clone_weights(self.encoder.word_embeddings, self.word_embeddings)
            self._tie_or_clone_weights(self.decoder.word_embeddings, self.word_embeddings)

    # 获取编码器
    def get_encoder(self):
        return self.encoder

    # 获取解码器
    def get_decoder(self):
        return self.decoder

    @add_start_docstrings_to_model_forward(XLM_PROPHETNET_INPUTS_DOCSTRING)  # 添加输入文档字符串
    @replace_return_docstrings(output_type=XLMProphetNetSeq2SeqModelOutput, config_class=_CONFIG_FOR_DOC)  # 替换输出文档字符串
    # 前向传播方法
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        decoder_inputs_embeds: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
    # 设置模型的描述字符串
    "The XLMProphetNet Model with a language modeling head. Can be used for sequence generation tasks."
    # 设置XLMProphetNet模型的docstring起始部分
    XLM_PROPHETNET_START_DOCSTRING,
# 从transformers.models.prophetnet.modeling_prophetnet.ProphetNetForConditionalGeneration中复制而来，将microsoft/prophetnet-large-uncased->patrickvonplaten/xprophetnet-large-uncased-standalone，将ProphetNet->XLMProphetNet，将PROPHETNET->XLM_PROPHETNET
class XLMProphetNetForConditionalGeneration(XLMProphetNetPreTrainedModel):
    # 绑定权重的键列表
    _tied_weights_keys = ["encoder.word_embeddings.weight", "decoder.word_embeddings.weight", "lm_head.weight"]

    def __init__(self, config: XLMProphetNetConfig):
        # 调用父类构造函数
        super().__init__(config)
        # 创建XLMProphetNetModel
        self.prophetnet = XLMProphetNetModel(config)
        # 获取填充索引
        self.padding_idx = config.pad_token_id
        # 禁用ngram损失
        self.disable_ngram_loss = config.disable_ngram_loss

        # 初始化线性层（语言模型的输出层）
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # 初始化权重并应用最终处理
        self.post_init()

    # 获取输出嵌入
    def get_output_embeddings(self):
        return self.lm_head

    # 设置输出嵌入
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    # 绑定权重
    def _tie_weights(self):
        if self.config.tie_word_embeddings:
            self._tie_or_clone_weights(self.prophetnet.word_embeddings, self.lm_head)

    # 获取输入嵌入
    def get_input_embeddings(self):
        return self.prophetnet.word_embeddings

    # 前向传播方法
    @add_start_docstrings_to_model_forward(XLM_PROPHETNET_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=XLMProphetNetSeq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        decoder_inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    # 计算损失函数
    def _compute_loss(self, logits, labels, ignore_index=-100):
        # 创建与标签相同形状的零张量，填充值为ignore_index，这里用于扩展目标标签
        expend_targets = labels.new_zeros(self.config.ngram, labels.size(0), labels.size(1)).fill_(ignore_index)

        # 循环创建多个相同的目标标签，用于计算Ngram损失
        for i in range(self.config.ngram):
            if i > 0 and self.disable_ngram_loss:  # 如果i>0且disable_ngram_loss为真，则跳出循环
                break
            expend_targets[i, :, :] = labels

        # 转置logits，使得维度为（时间步，批量大小，类别数）
        logits = logits.transpose(0, 1).contiguous()
        # 对logits进行log_softmax操作，返回对数概率
        lprobs = nn.functional.log_softmax(
            logits.view(-1, logits.size(-1)),  # 对logits进行reshape操作
            dim=-1,  # 沿着最后一个维度进行操作
            dtype=torch.float32,  # 数据类型为torch.float32
        )

        # 计算负对数损失
        loss = nn.functional.nll_loss(lprobs, expend_targets.view(-1), reduction="mean")

        # 如果设置了eps超参数，则进行平滑处理
        if self.config.eps > 0.0:
            smooth_loss = -lprobs.sum(dim=-1, keepdim=True)  # 计算平滑损失的负对数概率和
            non_masked_tokens = expend_targets.ne(ignore_index).view(-1)  # 获取非屏蔽的标记
            smooth_loss = smooth_loss[non_masked_tokens]  # 获取非屏蔽的平滑损失
            smooth_loss = smooth_loss.mean()  # 计算平滑损失的均值

            eps_i = self.config.eps / lprobs.size(-1)  # 计算eps_i的值
            loss = (1.0 - self.config.eps) * loss + eps_i * smooth_loss  # 最终的损失是原始损失和平滑损失的加权和

        return loss  # 返回计算得到的损失值

    # 为生成准备输入参数
    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past_key_values=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs,
    ):
        assert encoder_outputs is not None, "`encoder_outputs` have to be passed for generation."  # 断言encoder_outputs必须存在

        if past_key_values:  # 如果存在过去的关键值
            decoder_input_ids = decoder_input_ids[:, -1:]  # 则只取decoder_input_ids的最后一个时间步的输入
        # 返回准备好的输入参数
        return {
            "input_ids": None,  # encoder_outputs已经定义，因此input_ids不需要
            "encoder_outputs": encoder_outputs,  # 编码器输出
            "past_key_values": past_key_values,  # 过去的关键值
            "decoder_input_ids": decoder_input_ids,  # 解码器的输入id
            "attention_mask": attention_mask,  # 注意力遮罩
            "head_mask": head_mask,  # 头部遮罩
            "decoder_head_mask": decoder_head_mask,  # 解码器头部遮罩
            "cross_attn_head_mask": cross_attn_head_mask,  # 交叉注意力头部遮罩
            "use_cache": use_cache,  # 是否使用缓存
        }

    # 从标签中准备解码器的输入id
    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return self._shift_right(labels)  # 返回右移标签后的结果作为解码器的输入id

    @staticmethod
    # 重排序缓存
    # 从transformers.models.bart.modeling_bart.BartForConditionalGeneration中复制的_reorder_cache方法
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            # 缓存的交叉注意力状态不需要重新排序 -> 它们始终相同
            # 对每一层的过去关键值进行重新排序，返回的是一个元组
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past[:2])
                + layer_past[2:],
            )
        return reordered_past  # 返回经重新排序后的过去关键值元组

    # 获取编码器
    def get_encoder(self):
        return self.prophetnet.encoder  # 返回prophetnet的编码器

    # 获取解码器
    def get_decoder(self):
        return self.prophetnet.decoder  # 返回prophetnet的解码器
# 为 XLMProphetNetForCausalLM 类添加文档字符串，描述该模型用于语言建模
# 从 transformers.models.prophetnet.modeling_prophetnet.ProphetNetForCausalLM 复制代码，并做出相应修改
# 修改模型命名，引入模型，修改配置

class XLMProphetNetForCausalLM(XLMProphetNetPreTrainedModel):
    # 定义需要共享权重的键值
    _tied_weights_keys = [
        "prophetnet.word_embeddings.weight",
        "prophetnet.decoder.word_embeddings.weight",
        "lm_head.weight",
    ]

    # 初始化 XLMProphetNetForCausalLM 类
    def __init__(self, config: XLMProphetNetConfig):
        # 复制配置
        config = copy.deepcopy(config)
        # 设置为解码器
        config.is_decoder = True
        config.is_encoder_decoder = False
        # 调用父类初始化方法
        super().__init__(config)
        # 初始化 XLMProphetNetDecoderWrapper 类
        self.prophetnet = XLMProphetNetDecoderWrapper(config)

        self.padding_idx = config.pad_token_id
        self.disable_ngram_loss = config.disable_ngram_loss

        # 初始化线性层，用于LM头部
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # 初始化权重并进行最终处理
        self.post_init()

    # 返回输入嵌入
    def get_input_embeddings(self):
        return self.prophetnet.decoder.word_embeddings

    # 设置输入嵌入
    def set_input_embeddings(self, value):
        self.prophetnet.decoder.word_embeddings = value

    # 返回输出嵌入
    def get_output_embeddings(self):
        return self.lm_head

    # 设置输出嵌入
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    # 绑定权重
    def _tie_weights(self):
        if self.config.tie_word_embeddings:
            self._tie_or_clone_weights(self.prophetnet.decoder.word_embeddings, self.lm_head)

    # 设置解码器
    def set_decoder(self, decoder):
        self.prophetnet.decoder = decoder

    # 返回解码器
    def get_decoder(self):
        return self.prophetnet.decoder

    # 前向传播函数
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    # 计算损失函数，输入为模型的输出 logits、标签 labels，以及忽略的索引 ignore_index，默认为 -100
    def _compute_loss(self, logits, labels, ignore_index=-100):
        # 创建一个与 labels 相同大小的全零张量，用于扩展标签以匹配 logits 的维度
        expend_targets = labels.new_zeros(self.config.ngram, labels.size(0), labels.size(1)).fill_(ignore_index)

        # 根据 ngram 参数将标签扩展到不同的维度，若开启了 disable_ngram_loss 则只扩展第一个维度
        for i in range(self.config.ngram):
            if i > 0 and self.disable_ngram_loss:
                break
            expend_targets[i, :, :] = labels

        # 调整 logits 的维度顺序，并转换为连续内存
        logits = logits.transpose(0, 1).contiguous()
        # 对 logits 进行 log_softmax 操作，计算对数概率
        lprobs = nn.functional.log_softmax(
            logits.view(-1, logits.size(-1)),  # 将 logits 展平为二维张量
            dim=-1,  # 沿着最后一个维度计算 softmax
            dtype=torch.float32,  # 输出的数据类型为 float32
        )

        # 使用负对数似然损失函数计算损失
        loss = nn.functional.nll_loss(lprobs, expend_targets.view(-1), reduction="mean")

        # 如果 eps 大于 0，则进行 label 平滑处理
        if self.config.eps > 0.0:
            # 计算平滑损失
            smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
            non_masked_tokens = expend_targets.ne(ignore_index).view(-1)
            smooth_loss = smooth_loss[non_masked_tokens]
            smooth_loss = smooth_loss.mean()

            # 计算 epsilon 平滑项
            eps_i = self.config.eps / lprobs.size(-1)
            # 组合损失函数
            loss = (1.0 - self.config.eps) * loss + eps_i * smooth_loss

        # 返回计算得到的损失
        return loss

    # 为生成准备输入，接收输入的 token ids，过去的键值对（用于缓存）、注意力掩码、头部掩码、是否使用缓存等参数
    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        head_mask=None,
        use_cache=None,
        **kwargs,
    ):
        # 如果没有提供注意力掩码，则创建一个全 1 的掩码
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_ids.shape)

        # 如果存在过去的键值对，则只保留最后一个 token 的输入
        if past_key_values:
            input_ids = input_ids[:, -1:]
        # 对于第一个步骤，decoder_cached_states 是空的
        # 返回模型生成所需的输入信息
        return {
            "input_ids": input_ids,  # 编码器输出已经定义，不需要 input_ids
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
        }

    @staticmethod
    # 从给定的 past_key_values 中重新排列缓存，用于束搜索
    # 该方法来自于 transformers.models.bart.modeling_bart.BartForCausalLM._reorder_cache
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past
# 从transformers.models.prophetnet.modeling_prophetnet.ProphetNetDecoderWrapper中复制代码，将ProphetNet->XLMProphetNet, prophetnet->XLMProphetNet
# 创建XLMProphetNetDecoderWrapper类，用于正确加载预训练的XLMProphetNet类
class XLMProphetNetDecoderWrapper(XLMProphetNetPreTrainedModel):
    """
    This is a wrapper class, so that [`XLMProphetNetForCausalLM`] can correctly be loaded from pretrained XLMProphetNet
    classes.
    """

    def __init__(self, config: XLMProphetNetConfig):
        # 调用父类构造函数初始化配置
        super().__init__(config)

        # 初始化词嵌入层，用于将词汇id映射为隐藏状态向量，并指定填充id
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        # 创建XLMProphetNet解码器实例
        self.decoder = XLMProphetNetDecoder(config, word_embeddings=self.word_embeddings)

        # 初始化权重并应用最终处理
        self.post_init()

    # 绑定词嵌入层权重
    def _tie_weights(self):
        self._tie_or_clone_weights(self.word_embeddings, self.decoder.get_input_embeddings())

    # 前向传播函数
    def forward(self, *args, **kwargs):
        return self.decoder(*args, **kwargs)
```