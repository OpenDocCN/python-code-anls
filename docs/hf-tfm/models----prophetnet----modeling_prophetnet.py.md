# `.\models\prophetnet\modeling_prophetnet.py`

```
# 导入所需模块和库
import copy
import math
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import Tensor, nn
from torch.nn import LayerNorm

# 导入相关的自定义模块和函数
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
from .configuration_prophetnet import ProphetNetConfig

# 获取 logger 对象用于日志记录
logger = logging.get_logger(__name__)

# 配置和检查点信息，用于文档和模型加载
_CONFIG_FOR_DOC = "ProphenetConfig"
_CHECKPOINT_FOR_DOC = "microsoft/prophetnet-large-uncased"

# 预训练模型存档列表
PROPHETNET_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "microsoft/prophetnet-large-uncased",
    # 可在 https://huggingface.co/models?filter=prophetnet 查看所有 ProphetNet 模型
]

# ProphetNet 模型开始文档字符串
PROPHETNET_START_DOCSTRING = r"""
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
        config ([`ProphetNetConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

# ProphetNet 输入文档字符串
PROPHETNET_INPUTS_DOCSTRING = r"""
"""

# 独立输入文档字符串
PROPHETNET_STANDALONE_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            # 输入序列标记在词汇表中的索引。默认情况下，会忽略填充标记。
            # 可以使用`AutoTokenizer`获取这些索引。详见`PreTrainedTokenizer.encode`和`PreTrainedTokenizer.__call__`。

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            # 遮罩，用于避免在填充标记的位置进行注意力计算。遮罩值在 `[0, 1]` 范围内：

            - 1 表示**不被遮罩**的标记，
            - 0 表示**被遮罩**的标记。

            [What are attention masks?](../glossary#attention-mask)
        head_mask (`torch.Tensor` of shape `(encoder_layers, encoder_attention_heads)`, *optional*):
            # 用于在编码器的注意力模块中屏蔽选定头部的遮罩。遮罩值在 `[0, 1]` 范围内：

            - 1 表示**不被遮罩**的头部，
            - 0 表示**被遮罩**的头部。

        output_attentions (`bool`, *optional*):
            # 是否返回所有注意力层的注意力张量。返回的张量中会有关于注意力的更多细节。

        output_hidden_states (`bool`, *optional*):
            # 是否返回所有层的隐藏状态。返回的张量中会有关于隐藏状态的更多细节。

        return_dict (`bool`, *optional*):
            # 是否返回一个[`~utils.ModelOutput`]而不是普通的元组。
    # 计算主流相对位置
    main_stream_relative_positions = position_ids.unsqueeze(1).repeat(1, position_ids.size(-1), 1)
    # 从主流位置中减去每个位置的索引，得到相对位置
    main_stream_relative_positions = main_stream_relative_positions - position_ids.unsqueeze(-1)

    # 预测流相对位置
    predicting_stream_relative_positions = torch.cat((position_ids - 1, position_ids), dim=-1).unsqueeze(1)
    # 将预测流位置重复以匹配主流位置数目，并计算相对位置
    predicting_stream_relative_positions = predicting_stream_relative_positions.repeat(1, position_ids.size(-1), 1)
    predicting_stream_relative_positions = predicting_stream_relative_positions - position_ids.unsqueeze(-1)

    # 获取主流和预测流的位置桶
    # 计算主要流相对位置的桶
    main_relative_position_buckets = compute_relative_buckets(
        num_buckets,                   # 桶的数量
        max_distance,                  # 最大距离
        main_stream_relative_positions, # 主要流的相对位置
        is_bidirectional=False         # 是否双向（这里为单向，即不考虑双向）
    )
    
    # 计算预测流相对位置的桶
    predict_relative_position_buckets = compute_relative_buckets(
        num_buckets,                          # 桶的数量
        max_distance,                         # 最大距离
        predicting_stream_relative_positions, # 预测流的相对位置
        is_bidirectional=False                # 是否双向（这里为单向，即不考虑双向）
    )
    
    # 返回计算得到的主要流和预测流的相对位置桶
    return main_relative_position_buckets, predict_relative_position_buckets
@dataclass
class ProphetNetSeq2SeqLMOutput(ModelOutput):
    """
    Base class for sequence-to-sequence language models outputs.

    """

    # 损失值，可选的浮点张量
    loss: Optional[torch.FloatTensor] = None
    # 模型输出的 logits，浮点张量
    logits: torch.FloatTensor = None
    # ngram 模型输出的 logits，可选的浮点张量
    logits_ngram: Optional[torch.FloatTensor] = None
    # 过去的键/值对，可选的张量元组，用于加速顺序解码
    past_key_values: Optional[Tuple[torch.FloatTensor]] = None
    # 解码器隐藏状态的元组，可选的浮点张量元组
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # ngram 解码器隐藏状态的元组，可选的浮点张量元组
    decoder_ngram_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # 解码器注意力权重的元组，可选的浮点张量元组
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    # ngram 解码器注意力权重的元组，可选的浮点张量元组
    decoder_ngram_attentions: Optional[Tuple[torch.FloatTensor]] = None
    # 交叉注意力权重的元组，可选的浮点张量元组
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    # 编码器最后隐藏状态，可选的浮点张量
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    # 编码器隐藏状态的元组，可选的浮点张量元组
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # 编码器注意力权重的元组，可选的浮点张量元组
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None

    @property
    def decoder_cross_attentions(self):
        # 警告信息，提示 'decoder_cross_attentions' 将被移除，请使用 'cross_attentions' 替代
        warnings.warn(
            "`decoder_cross_attentions` is deprecated and will be removed soon. Please use `cross_attentions`"
            " instead.",
            FutureWarning,
        )
        # 返回交叉注意力权重的元组
        return self.cross_attentions


@dataclass
class ProphetNetSeq2SeqModelOutput(ModelOutput):
    """
    Base class for model encoder's outputs that also contains : pre-computed hidden states that can speed up sequential
    decoding.

    """

    # 最后一个隐藏状态的浮点张量
    last_hidden_state: torch.FloatTensor
    # ngram 模型的最后一个隐藏状态，可选的浮点张量
    last_hidden_state_ngram: Optional[torch.FloatTensor] = None
    # 过去的键/值对，可选的张量元组，用于加速顺序解码
    past_key_values: Optional[Tuple[torch.FloatTensor]] = None
    # 解码器隐藏状态的元组，可选的浮点张量元组
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # ngram 解码器隐藏状态的元组，可选的浮点张量元组
    decoder_ngram_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # 解码器注意力权重的元组，可选的浮点张量元组
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    # ngram 解码器注意力权重的元组，可选的浮点张量元组
    decoder_ngram_attentions: Optional[Tuple[torch.FloatTensor]] = None
    # 交叉注意力权重的元组，可选的浮点张量元组
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    # 编码器最后隐藏状态，可选的浮点张量
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    # 编码器隐藏状态的元组，可选的浮点张量元组
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # 编码器注意力权重的元组，可选的浮点张量元组
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None

    @property
    def decoder_cross_attentions(self):
        # 警告信息，提示 'decoder_cross_attentions' 将被移除，请使用 'cross_attentions' 替代
        warnings.warn(
            "`decoder_cross_attentions` is deprecated and will be removed soon. Please use `cross_attentions`"
            " instead.",
            FutureWarning,
        )
        # 返回交叉注意力权重的元组
        return self.cross_attentions


@dataclass
class ProphetNetDecoderModelOutput(ModelOutput):
    """
    Base class for model's outputs that may also contain a past key/values (to speed up sequential decoding).

    """

    # 最后一个隐藏状态的浮点张量
    last_hidden_state: torch.FloatTensor
    # ngram 模型的最后一个隐藏状态，可选的浮点张量
    last_hidden_state_ngram: Optional[torch.FloatTensor] = None
    # 过去的键/值对，可选的张量元组，用于加速顺序解码
    past_key_values: Optional[Tuple[torch.FloatTensor]] = None
    # 隐藏状态的元组，可选的浮点张量元组
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # ngram 隐藏状态的元组，可选的浮点张量元组
    hidden_states_ngram: Optional[Tuple[torch.FloatTensor]] = None
    # 注意力权重的元组，可选的浮点张量元组
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    # ngram 注意力权重的元组，可选的浮点张量元组
    ngram_attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
@dataclass
class ProphetNetDecoderLMOutput(ModelOutput):
    """
    Model output class for the ProphetNet decoder, inheriting from ModelOutput.
    Contains various tensors representing model predictions and intermediate states.
    """

    loss: Optional[torch.FloatTensor] = None  # Optional tensor for model training loss
    logits: torch.FloatTensor = None  # Tensor containing logits (predictions) from the decoder
    logits_ngram: Optional[torch.FloatTensor] = None  # Optional tensor for n-gram logits
    past_key_values: Optional[Tuple[torch.FloatTensor]] = None  # Optional tuple of past key/values for fast decoding
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None  # Optional tuple of hidden states
    hidden_states_ngram: Optional[Tuple[torch.FloatTensor]] = None  # Optional tuple of n-gram hidden states
    attentions: Optional[Tuple[torch.FloatTensor]] = None  # Optional tuple of attention tensors
    ngram_attentions: Optional[Tuple[torch.FloatTensor]] = None  # Optional tuple of n-gram attention tensors
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None  # Optional tuple of cross-attention tensors


class ProphetNetPreTrainedModel(PreTrainedModel):
    """
    Base class for all models in the ProphetNet series, inheriting from PreTrainedModel.
    """

    config_class = ProphetNetConfig  # Configuration class for ProphetNet models
    base_model_prefix = "prophetnet"  # Prefix used for the base model
    supports_gradient_checkpointing = True  # Indicates whether the model supports gradient checkpointing

    def _init_weights(self, module):
        """
        Initialize weights of linear and embedding modules based on configuration.
        """
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.init_std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.init_std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def _shift_right(self, input_ids):
        """
        Shift input ids to the right for autoregressive decoding.
        """
        decoder_start_token_id = self.config.decoder_start_token_id
        pad_token_id = self.config.pad_token_id

        assert decoder_start_token_id is not None, (
            "self.model.config.decoder_start_token_id has to be defined. In ProphetNet it is usually set to the"
            " pad_token_id. See ProphetNet docs for more information"
        )

        # shift inputs to the right
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
        shifted_input_ids[..., 0] = decoder_start_token_id

        assert pad_token_id is not None, "self.model.config.pad_token_id has to be defined."
        # replace possible -100 values in labels by `pad_token_id`
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

        assert torch.all(shifted_input_ids >= 0).item(), "Verify that `shifted_input_ids` has only positive values"

        return shifted_input_ids


class ProphetNetPositionalEmbeddings(nn.Embedding):
    """
    Positional embedding module for ProphetNet models.
    Learns positional embeddings up to a fixed maximum size, handling padding ids.
    """

    def __init__(self, config: ProphetNetConfig) -> None:
        self.max_length = config.max_position_embeddings
        super().__init__(config.max_position_embeddings, config.hidden_size, config.pad_token_id)
    # 定义前向传播函数，接受输入形状、设备信息，可选的注意力掩码、过去的键值对和位置 ID
    def forward(self, inputs_shape, device, attention_mask=None, past_key_values=None, position_ids=None):
        # 断言：如果位置 ID 已预先计算，则填充索引不应设置
        assert (position_ids is None) or (
            self.padding_idx is None
        ), "If position_ids is pre-computed then padding_idx should not be set."

        # 如果位置 ID 未提供
        if position_ids is None:
            # 如果有过去的键值对
            if past_key_values is not None:
                # 位置 ID 在解码单步时对每个令牌相同
                # 在导出到 ONNX 时，如果没有 int() 转换，在某些情况下可能无法正常工作
                prev_num_input_ids = past_key_values[0][0].shape[2]
                num_input_ids = inputs_shape[1] + prev_num_input_ids
                # 计算位置 ID，确保它为填充索引加上输入令牌数
                position_ids = torch.ones((1, 1), dtype=torch.long, device=device) * (
                    int(self.padding_idx + num_input_ids)
                )
            else:
                # 如果没有过去的键值对，并且没有提供注意力掩码，则创建全一的注意力掩码
                if attention_mask is None:
                    attention_mask = torch.ones(inputs_shape, dtype=torch.long, device=device)

                # 从输入令牌 / 注意力掩码中检索位置 ID
                position_ids = (
                    torch.cumsum(attention_mask, dim=1).type_as(attention_mask) * attention_mask
                ).long() + self.padding_idx

                # 确保位置 ID 不超过最大长度减一
                position_ids = position_ids.clamp(0, self.max_length - 1)

        # 调用父类的前向传播函数，返回结果和计算得到的位置 ID
        return super().forward(position_ids), position_ids

    # 私有方法 _forward，接受位置 ID 参数
    def _forward(self, position_ids):
        # 调用父类的前向传播函数，传递位置 ID 参数
        return super().forward(position_ids)
class ProphetNetAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        config: ProphetNetConfig,
        num_attn_heads: int,
    ):
        super().__init__()
        hidden_size = config.hidden_size

        self.attention_dropout = config.attention_dropout  # 从配置中获取注意力丢弃率
        self.dropout = config.dropout  # 从配置中获取全连接层输出的丢弃率
        self.num_attn_heads = num_attn_heads  # 设置注意力头的数量
        self.head_dim = hidden_size // num_attn_heads  # 计算每个注意力头的维度

        assert self.head_dim * num_attn_heads == hidden_size, (
            "`config.hidden_size` must be divisible by `config.num_encoder_attention_heads` and"
            " `config.num_decoder_attention_heads`"
        )

        self.key_proj = nn.Linear(hidden_size, hidden_size)  # 初始化键的投影矩阵
        self.value_proj = nn.Linear(hidden_size, hidden_size)  # 初始化值的投影矩阵
        self.query_proj = nn.Linear(hidden_size, hidden_size)  # 初始化查询的投影矩阵

        self.out_proj = nn.Linear(hidden_size, hidden_size)  # 初始化输出投影矩阵

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_attn_heads, self.head_dim).transpose(1, 2).contiguous()
        # 重新形状张量以便进行多头注意力计算

    def forward(
        self,
        hidden_states,
        key_value_states: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        layer_head_mask: Optional[Tensor] = None,
        past_key_value: Optional[Tuple[Tensor]] = None,
        output_attentions: bool = False,
    ):
        # 前向传播函数定义，执行注意力计算
    # 初始化函数，接受一个ProphetNetConfig类型的参数config
    def __init__(self, config: ProphetNetConfig):
        # 调用父类的初始化函数
        super().__init__()
        # 设置隐藏层大小
        self.hidden_size = config.hidden_size

        # 设置桶的数量
        self.num_buckets = config.num_buckets
        # 设置相对位置的最大距离
        self.relative_max_distance = config.relative_max_distance
        # 设置注意力头的数量
        self.num_attn_heads = config.num_decoder_attention_heads
        # 设置dropout率
        self.dropout = config.dropout
        # 设置注意力dropout率
        self.attention_dropout = config.attention_dropout
        # 计算每个注意力头的维度
        self.head_dim = config.hidden_size // self.num_attn_heads
        # 设置ngram
        self.ngram = config.ngram

        # 断言确保隐藏层大小能够被注意力头的数量整除
        assert (
            self.head_dim * self.num_attn_heads == config.hidden_size
        ), "config.hidden_size must be divisible by num_attn_heads"

        # key, value, query的投影层
        self.key_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.value_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.query_proj = nn.Linear(config.hidden_size, config.hidden_size)

        # 输出投影层
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)

        # 相对位置嵌入
        self.relative_pos_embeddings = nn.Linear(config.hidden_size, self.num_buckets * self.num_attn_heads)

        # 用于ONNX运行时的标志
        self.onnx_trace = False

    # 将张量形状重新整理为(batch_size, seq_len, num_attn_heads, head_dim)，并进行转置
    def _shape(self, tensor, seq_len, batch_size):
        return tensor.view(batch_size, seq_len, self.num_attn_heads, self.head_dim).transpose(1, 2).contiguous()

    # 准备用于导出到ONNX的设置
    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    # 前向传播函数，接受一系列输入参数，并返回输出结果
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
    ):
        # 省略部分前向传播函数的具体实现
    ):
        # input hidden_states [batch_size, sequence_length, hidden_size]
        # input attn_weights [batch_size, num_heads, sequence_length, sequence_length]
        # input position_ids [batch_size, sequence_length] or [1,1]
        
        # 解构输入参数中的维度信息
        batch_size, num_attn_heads, tgt_len, src_len = attn_weights.shape
        
        # 调整注意力权重张量的形状，以匹配后续计算需求
        attn_weights = attn_weights.view(batch_size, num_attn_heads, tgt_len, src_len)
        
        # 如果未提供主要相对位置桶，则计算默认相对位置信息
        if main_relative_position_buckets is None:
            # 获取隐藏状态张量的形状信息
            batch_size, sequence_length = hidden_states.shape[:2]
            
            # 计算相对位置张量，减去给定的位置标识
            relative_positions = (
                torch.arange(1, attn_weights.shape[-1] + 1)
                .unsqueeze(0)
                .unsqueeze(0)
                .repeat(batch_size, sequence_length, 1)
                .to(position_ids.device)
            )
            relative_positions = relative_positions - position_ids.unsqueeze(0).repeat(batch_size, sequence_length, 1)
            
            # 计算主要相对位置桶，用于多头注意力机制
            main_relative_position_buckets = compute_relative_buckets(
                self.num_buckets, self.relative_max_distance, relative_positions, False
            )

        # 计算相对位置编码张量
        rel_pos_embeddings = self.relative_pos_embeddings(hidden_states)
        
        # 调整相对位置编码张量的形状，以匹配注意力权重和多头数目
        rel_pos_embeddings = rel_pos_embeddings.view(
            rel_pos_embeddings.shape[:2] + (self.num_buckets, self.num_attn_heads)
        )
        rel_pos_embeddings = rel_pos_embeddings.permute(0, 3, 1, 2)
        
        # 将相对位置编码张量重塑为适合注意力权重形状的张量
        rel_pos_embeddings = rel_pos_embeddings.reshape(attn_weights.shape[:3] + (-1,))

        # 复制主要相对位置桶以适配多头数目
        main_relative_position_buckets = main_relative_position_buckets.repeat(1, self.num_attn_heads, 1)
        
        # 将主要相对位置桶的形状调整为适合索引操作的形式
        main_relative_position_buckets = main_relative_position_buckets.view(
            -1, main_relative_position_buckets.shape[-1]
        )
        
        # 将主要相对位置桶转换为长整型
        main_relative_position_buckets = main_relative_position_buckets.long()
        
        # 将相对位置编码张量重塑为适合索引操作的形式
        rel_pos_embeddings = rel_pos_embeddings.reshape(-1, rel_pos_embeddings.size(-1))

        # 使用索引操作从相对位置编码张量中获取主要相对位置编码
        main_relative_pos_embeddings = torch.gather(rel_pos_embeddings, dim=1, index=main_relative_position_buckets)
        
        # 将获取的主要相对位置编码重新调整为原始形状
        main_relative_pos_embeddings = main_relative_pos_embeddings.view(batch_size, num_attn_heads, tgt_len, -1)
        
        # 返回主要相对位置编码张量
        return main_relative_pos_embeddings

    def get_predict_relative_pos_embeddings(
        self, hidden_states, attn_weights, position_ids, predict_relative_position_buckets
    ):
        # input hidden_states [batch_size, sequence_length, ngram, hidden_size]
        # input attn_weights [batch_size, ngram, num_heads, sequence_length, 2*sequence_length]
        # input position_ids [batch_size, sequence_length] or [1,1]
        # input predict_relative_position_buckets [batch_size, sequence_length, 2*sequence_length] or None
        
        # 获取 batch_size 和 sequence_length
        batch_size, sequence_length = hidden_states.shape[0:2]

        # 如果 predict_relative_position_buckets 为 None，则计算相对位置信息
        if predict_relative_position_buckets is None:
            # 获取 attn_weights 的 key_sequence_length
            key_sequence_length = attn_weights.shape[-1]
            # 检查 position_ids 的有效性，确保格式为 1 2 3 4 5 ... (key_sequence_length - 1)
            assert (
                position_ids[0][0] == key_sequence_length - 1
            ), "`position_ids` are incorrect. They should be of the format 1 2 3 4 5 ... (key_sequence_length - 1)"
            
            # 生成相对位置信息 relative_positions
            relative_positions = (
                torch.arange(0, key_sequence_length)
                .unsqueeze(0)
                .unsqueeze(0)
                .repeat(batch_size, sequence_length, 1)
                .to(position_ids.device)
            )

            # 计算相对位置差值
            relative_positions = relative_positions - position_ids.unsqueeze(0).repeat(batch_size, sequence_length, 1)
            
            # 计算预测相对位置桶
            predict_relative_position_buckets = compute_relative_buckets(
                self.num_buckets, self.relative_max_distance, relative_positions, False
            )

        # 将 hidden_states 的维度 [batch_size, sequence_length, ngram, hidden_size] 转置为 [batch_size, ngram, sequence_length, hidden_size]
        hidden_states = hidden_states.transpose(1, 2)
        
        # 计算相对位置嵌入 rel_pos_embeddings
        rel_pos_embeddings = self.relative_pos_embeddings(hidden_states)

        # 将 rel_pos_embeddings 的维度调整为 [batch_size, ngram, sequence_length, num_buckets, num_heads]
        rel_pos_embeddings = rel_pos_embeddings.view(
            hidden_states.shape[:-1] + (self.num_buckets, self.num_attn_heads)
        )
        
        # 将 rel_pos_embeddings 的维度重新排列为 [batch_size, ngram, num_heads, sequence_length, num_buckets]
        rel_pos_embeddings = rel_pos_embeddings.permute(0, 2, 1, 4, 3)
        
        # 将 rel_pos_embeddings 的形状调整为 [batch_size * ngram * sequence_length * num_heads, num_buckets]
        rel_pos_embeddings = rel_pos_embeddings.reshape(-1, self.num_buckets)
        
        # 将 predict_relative_position_buckets 的形状调整为 [ngram, batch_size, num_heads, sequence_length, -1]
        predict_relative_position_buckets = predict_relative_position_buckets.unsqueeze(0)
        predict_relative_position_buckets = predict_relative_position_buckets.repeat(
            self.ngram, 1, self.num_attn_heads, 1
        )
        
        # 将 predict_relative_position_buckets 的形状调整为 [ngram * batch_size * num_heads * sequence_length, -1]
        predict_relative_position_buckets = predict_relative_position_buckets.view(
            -1, predict_relative_position_buckets.size(-1)
        ).long()
        
        # 使用 torch.gather 获取预测的相对位置嵌入 predict_relative_pos_embeddings
        predict_relative_pos_embeddings = torch.gather(
            rel_pos_embeddings, dim=1, index=predict_relative_position_buckets
        )

        # 将 predict_relative_pos_embeddings 的形状调整为 [batch_size, gram, num_heads, sequence_length, -1]
        predict_relative_pos_embeddings = predict_relative_pos_embeddings.view(
            batch_size, self.ngram, self.num_attn_heads, sequence_length, -1
        )

        # 返回预测的相对位置嵌入 predict_relative_pos_embeddings
        return predict_relative_pos_embeddings
class ProphetNetEncoderLayer(nn.Module):
    """
    Encoder block for Prophetnet
    """

    def __init__(self, config: ProphetNetConfig):
        super().__init__()
        # 1st residual block
        # 创建自注意力机制模块，使用 ProphetNetAttention
        self.self_attn = ProphetNetAttention(config, config.num_encoder_attention_heads)
        # 创建自注意力机制的 LayerNorm 层
        self.self_attn_layer_norm = LayerNorm(config.hidden_size)

        # 2nd residual block
        # 创建前馈神经网络模块，使用 ProphetNetFeedForward
        self.feed_forward = ProphetNetFeedForward(config, config.encoder_ffn_dim)
        # 创建前馈神经网络的 LayerNorm 层
        self.feed_forward_layer_norm = LayerNorm(config.hidden_size)

    def forward(
        self,
        hidden_states,
        attention_mask,
        layer_head_mask,
        output_attentions: bool = False,
    ):
        # 1st residual block
        # 执行自注意力机制，得到注意力输出、注意力权重和无用的变量 _
        attention_output, attn_weights, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        # 应用 LayerNorm 到注意力输出和输入状态的残差连接
        hidden_states = self.self_attn_layer_norm(attention_output + hidden_states)

        # 2nd residual block
        # 执行前馈神经网络
        feed_forward_output = self.feed_forward(hidden_states)
        # 应用 LayerNorm 到前馈神经网络输出和输入状态的残差连接
        hidden_states = self.feed_forward_layer_norm(feed_forward_output + hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class ProphetNetDecoderLayer(nn.Module):
    """
    Decoder block for Prophetnet
    """

    def __init__(self, config: ProphetNetConfig):
        super().__init__()
        # 1st residual block
        # 创建 N-gram 自注意力机制模块，使用 ProphetNetNgramSelfAttention
        self.self_attn = ProphetNetNgramSelfAttention(config)
        # 创建自注意力机制的 LayerNorm 层
        self.self_attn_layer_norm = LayerNorm(config.hidden_size)

        # 2nd residual block
        # 如果配置要求添加跨注意力机制
        if config.add_cross_attention:
            # 创建跨注意力机制模块，使用 ProphetNetAttention
            self.cross_attn = ProphetNetAttention(config, config.num_decoder_attention_heads)
            # 创建跨注意力机制的 LayerNorm 层
            self.cross_attn_layer_norm = LayerNorm(config.hidden_size)

        # 3rd residual block
        # 创建解码器前馈神经网络模块，使用 ProphetNetFeedForward
        self.feed_forward = ProphetNetFeedForward(config, config.decoder_ffn_dim)
        # 创建前馈神经网络的 LayerNorm 层
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
        ):
            # 1st residual block
            # 如果过去的键/值对存在，则从中选择前两个作为自注意力的过去键/值对；否则为 None
            self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
            # 调用自注意力机制，计算输出和注意力权重
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
            # 应用 Layer Normalization 到自注意力输出和原始隐藏状态的和
            hidden_states = self.self_attn_layer_norm(hidden_states + ngram_attention_output)

            # 如果过去的键/值对存在，则从中选择后两个作为跨注意力的过去键/值对；否则为 None
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            cross_attn_weights = None
            if encoder_hidden_states is not None:
                # 2nd residual block
                # 调用跨注意力机制，计算输出、注意力权重和当前键/值对
                attention_output, cross_attn_weights, cross_attn_present_key_value = self.cross_attn(
                    hidden_states=hidden_states,
                    key_value_states=encoder_hidden_states,
                    attention_mask=encoder_attn_mask,
                    layer_head_mask=cross_attn_layer_head_mask,
                    past_key_value=cross_attn_past_key_value,
                    output_attentions=output_attentions,
                )
                # 应用 Layer Normalization 到跨注意力输出和原始隐藏状态的和
                hidden_states = self.cross_attn_layer_norm(attention_output + hidden_states)

                # 将跨注意力的当前键/值对添加到当前键/值对中的第三、第四个位置
                present_key_value = present_key_value + cross_attn_present_key_value

            # 3rd residual block
            # 应用前馈网络层到隐藏状态，得到前馈网络输出
            feed_forward_output = self.feed_forward(hidden_states)
            # 应用 Layer Normalization 到前馈网络输出和原始隐藏状态的和
            hidden_states = self.feed_forward_layer_norm(feed_forward_output + hidden_states)

            # 输出结果初始化为包含隐藏状态的元组
            outputs = (hidden_states,)

            # 如果需要输出注意力权重，则将自注意力和跨注意力的权重添加到输出结果中
            if output_attentions:
                outputs += (self_attn_weights, self_attn_weights_ngram, cross_attn_weights)

            # 如果需要使用缓存，则将当前键/值对添加到输出结果中
            if use_cache:
                outputs += (present_key_value,)

            # 返回最终的输出结果
            return outputs
# 定义封装了ProphetNet模型的独立编码器部分 的类

@add_start_docstrings(
    "The standalone encoder part of the ProphetNetModel.",
    PROPHETNET_START_DOCSTRING,
)
class ProphetNetEncoder(ProphetNetPreTrainedModel):
    """
    代表ProphetNet编码器部分，用于封装模型的编码器参数。这个类可以使用预定义的词嵌入初始化，而不是随机初始化的词嵌入。
    """

    def __init__(self, config: ProphetNetConfig, word_embeddings: nn.Embedding = None):
        # 初始化ProphetNet编码器
        super().__init__(config)

        # 根据传入的参数，创建词嵌入层。如果没有提供，使用预设的随机初始化和填充索引。
        self.word_embeddings = (
            word_embeddings
            if word_embeddings is not None
            else nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        )

        # 初始化位置嵌入层
        self.position_embeddings = ProphetNetPositionalEmbeddings(config)

        # 初始化嵌入层归一化层
        self.embeddings_layer_norm = LayerNorm(config.hidden_size)

        # 创建编码器层列表
        self.layers = nn.ModuleList([
            ProphetNetEncoderLayer(config)
            for _ in range(config.num_encoder_layers)
        ])

        # 初始化梯度检查点
        self.gradient_checkpointing = False

        # 执行最后的操作，初始化权重和处理
        self.post_init()

    # 获取输入嵌入的函数
    def get_input_embeddings(self):
        return self.word_embeddings

    # 设置输入嵌入的函数
    def set_input_embeddings(self, value):
        self.word_embeddings = value

    # 定义处理输入的数据格式和参数的函数，实现前馈操作
    @add_start_docstrings_to_model_forward(PROPHETNET_STANDALONE_INPUTS_DOCSTRING)
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
    ):
        return MethodName(...)

# 定义封装了ProphetNet模型的独立解码器部分 的类
@add_start_docstrings(
    "The standalone decoder part of the ProphetNetModel.",
    PROPHETNET_START_DOCSTRING,
)
class ProphetNetDecoder(ProphetNetPreTrainedModel):
    """
    用于封装ProphetNet模型的独立解码器部分。此类可用于通过提供预定义的词嵌入初始化模型，而不是随机初始化词嵌入。
    """

    def __init__(self, config: ProphetNetConfig, word_embeddings: nn.Embedding = None):
        # 初始化ProphetNet解码器
        super().__init__(config)

        # 根据传入的参数，创建词嵌入层。如果没有提供，使用预设的随机初始化和填充索引。
        self.word_embeddings = (
            word_embeddings
            if word_embeddings is not None
            else nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        )

        # 这里可添加额外的初始化操作和参数初始化
    # 初始化函数，用于初始化ProphetNetDecoder模型的各种参数和组件
    def __init__(self, config: ProphetNetConfig, word_embeddings: Optional[nn.Embedding] = None):
        # 调用父类的初始化函数，初始化模型的基本配置
        super().__init__(config)

        # 设置模型使用的N-gram大小
        self.ngram = config.ngram
        # 设置模型使用的桶（bucket）数量
        self.num_buckets = config.num_buckets
        # 设置模型使用的相对最大距离
        self.relative_max_distance = config.relative_max_distance
        # 设置模型使用的dropout比例
        self.dropout = config.dropout
        # 设置模型允许的最大目标位置
        self.max_target_positions = config.max_position_embeddings

        # 初始化词嵌入层，如果给定了外部的词嵌入则使用外部的，否则创建新的词嵌入
        self.word_embeddings = (
            word_embeddings
            if word_embeddings is not None
            else nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        )
        
        # 初始化位置编码层
        self.position_embeddings = ProphetNetPositionalEmbeddings(config)

        # 初始化N-gram编码层
        self.ngram_embeddings = nn.Embedding(self.ngram, config.hidden_size, None)
        
        # 初始化多层ProphetNet解码器层
        self.layers = nn.ModuleList([ProphetNetDecoderLayer(config) for _ in range(config.num_decoder_layers)])
        
        # 初始化嵌入层的LayerNorm层
        self.embeddings_layer_norm = LayerNorm(config.hidden_size)

        # 设置梯度检查点为False，通常用于内存优化
        self.gradient_checkpointing = False
        
        # 执行后续的初始化和权重设置
        self.post_init()

    # 返回模型的输入词嵌入层
    def get_input_embeddings(self):
        return self.word_embeddings

    # 设置模型的输入词嵌入层
    def set_input_embeddings(self, value):
        self.word_embeddings = value

    # ProphetNetDecoder模型的前向传播函数，接受多个输入参数并返回相应的输出
    @add_start_docstrings_to_model_forward(PROPHETNET_STANDALONE_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=ProphetNetDecoderModelOutput, config_class=_CONFIG_FOR_DOC)
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
        # 输入参数的详细描述已经通过装饰器添加到模型前向函数的文档中
    def compute_buffered_relative_buckets(self, position_ids):
        # 获取批处理大小和序列长度
        batch_size, sequence_length = position_ids.shape

        # 创建位置ID序列，范围从1到self.max_target_positions，复制到当前设备
        position_ids = torch.arange(1, self.max_target_positions).to(position_ids.device).repeat(1, 1)
        
        # 计算主要和预测相对桶
        main_relative_buckets, predict_relative_buckets = compute_all_stream_relative_buckets(
            self.num_buckets, self.relative_max_distance, position_ids
        )

        # 缓冲相对桶
        main_relative_buckets = main_relative_buckets[:, :sequence_length, :sequence_length].repeat(batch_size, 1, 1)
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

    def prepare_attention_mask(self, hidden_states, attention_mask):
        # 获取批处理大小和序列长度
        batch_size, seq_length = hidden_states.shape[:2]

        # 获取因果遮罩
        causal_mask = torch.full(
            (seq_length, seq_length),
            torch.finfo(hidden_states.dtype).min,  # 用隐藏状态的最小浮点数填充
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        causal_mask = torch.triu(causal_mask, 1)  # 获取因果上三角遮罩

        extended_causal_mask = causal_mask[:seq_length, :seq_length][None, None, :, :].expand(
            (batch_size, self.config.num_decoder_attention_heads) + causal_mask.shape
        )

        # 添加常规的注意力遮罩
        if attention_mask is not None:
            extended_attention_mask = (1.0 - attention_mask[:, None, None, :]) * torch.finfo(hidden_states.dtype).min
            extended_attention_mask = extended_causal_mask + extended_attention_mask
        else:
            extended_attention_mask = extended_causal_mask

        return extended_attention_mask.to(hidden_states.dtype)
    # 定义一个方法用于准备预测时的注意力掩码
    def prepare_predict_attention_mask(self, hidden_states, attention_mask):
        # 获取批处理大小和序列长度
        batch_size, seq_length = hidden_states.shape[:2]

        # 获取因果掩码
        predict_causal_mask = ngram_attention_bias(
            self.max_target_positions, self.ngram, hidden_states.device, hidden_states.dtype
        )
        # 将因果掩码按列连接，形成预测时的完整因果掩码
        predict_causal_mask = torch.cat(
            [
                predict_causal_mask[:, :seq_length, :seq_length],
                predict_causal_mask[
                    :, :seq_length, self.max_target_positions : self.max_target_positions + seq_length
                ],
            ],
            dim=-1,
        )
        # 扩展因果掩码以适应批处理维度和注意力头的数量
        extended_predict_causal_mask = predict_causal_mask[None, None, :, :, :].expand(
            (batch_size, self.config.num_decoder_attention_heads) + predict_causal_mask.shape
        )

        # 添加普通的注意力掩码
        if attention_mask is not None:
            # 创建扩展的注意力掩码，并确保预测流的注意力掩码始终为0
            extended_attention_mask = (1.0 - attention_mask[:, None, None, None, :]) * torch.finfo(self.dtype).min
            extended_attention_mask = extended_attention_mask.expand(
                (batch_size, self.config.num_decoder_attention_heads, self.ngram, seq_length, seq_length)
            )
            extended_attention_mask = torch.cat(
                [extended_attention_mask, torch.zeros_like(extended_attention_mask)], dim=-1
            )
            extended_predict_attention_mask = extended_predict_causal_mask + extended_attention_mask
        else:
            extended_predict_attention_mask = extended_predict_causal_mask
        
        # 将最终的扩展预测注意力掩码转换为隐藏状态的数据类型并返回
        return extended_predict_attention_mask.to(hidden_states.dtype)
@add_start_docstrings(
    "The bare ProphetNet Model outputting raw hidden-states without any specific head on top.",
    PROPHETNET_START_DOCSTRING,
)
# 定义 ProphetNetModel 类，继承自 ProphetNetPreTrainedModel
class ProphetNetModel(ProphetNetPreTrainedModel):
    # 定义 tied_weights_keys 列表，用于存储需要绑定权重的键名
    _tied_weights_keys = ["encoder.word_embeddings.weight", "decoder.word_embeddings.weight"]

    # 初始化方法，接收 ProphetNetConfig 类型的 config 参数
    def __init__(self, config: ProphetNetConfig):
        # 调用父类 ProphetNetPreTrainedModel 的初始化方法
        super().__init__(config)
        
        # 创建词嵌入层，使用 nn.Embedding 类，设置词汇量大小、隐藏层大小和填充标记ID
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)

        # 复制 config 以创建编码器的配置，设置为不是编码-解码器模式且不使用缓存
        encoder_config = copy.deepcopy(config)
        encoder_config.is_encoder_decoder = False
        encoder_config.use_cache = False
        
        # 创建编码器实例，使用 ProphetNetEncoder 类，并传入配置和词嵌入层
        self.encoder = ProphetNetEncoder(encoder_config, self.word_embeddings)

        # 复制 config 以创建解码器的配置，设置为解码器模式且不是编码-解码器模式
        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        
        # 创建解码器实例，使用 ProphetNetDecoder 类，并传入配置和词嵌入层
        self.decoder = ProphetNetDecoder(decoder_config, self.word_embeddings)

        # 初始化权重并应用最终处理
        self.post_init()

    # 获取输入词嵌入层的方法
    def get_input_embeddings(self):
        return self.word_embeddings

    # 设置输入词嵌入层的方法，接收 value 参数
    def set_input_embeddings(self, value):
        # 设置词嵌入层为 value
        self.word_embeddings = value
        # 设置编码器和解码器的词嵌入层为相同的 value
        self.encoder.word_embeddings = self.word_embeddings
        self.decoder.word_embeddings = self.word_embeddings

    # 绑定权重的私有方法
    def _tie_weights(self):
        # 如果配置中指定了绑定词嵌入层的权重
        if self.config.tie_word_embeddings:
            # 将编码器和解码器的词嵌入层权重绑定到同一个实例
            self._tie_or_clone_weights(self.encoder.word_embeddings, self.word_embeddings)
            self._tie_or_clone_weights(self.decoder.word_embeddings, self.word_embeddings)

    # 获取编码器实例的方法
    def get_encoder(self):
        return self.encoder

    # 获取解码器实例的方法
    def get_decoder(self):
        return self.decoder

    # 前向传播方法，接收多个输入参数，并设置了输出文档的注释和返回值类型
    @add_start_docstrings_to_model_forward(PROPHETNET_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=ProphetNetSeq2SeqModelOutput, config_class=_CONFIG_FOR_DOC)
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
    # 定义一个类变量，包含需要共享权重的模型层的名称列表
    _tied_weights_keys = ["encoder.word_embeddings.weight", "decoder.word_embeddings.weight", "lm_head.weight"]

    # 初始化方法，接收一个ProphetNetConfig类型的配置对象作为参数
    def __init__(self, config: ProphetNetConfig):
        # 调用父类初始化方法，传入配置对象
        super().__init__(config)
        # 创建ProphetNetModel对象，并将其保存在self.prophetnet中
        self.prophetnet = ProphetNetModel(config)
        # 设置padding_idx为配置对象中的pad_token_id属性
        self.padding_idx = config.pad_token_id
        # 根据配置对象的disable_ngram_loss属性设置self.disable_ngram_loss
        self.disable_ngram_loss = config.disable_ngram_loss

        # 创建一个线性层，将输入维度设为配置对象的hidden_size，输出维度设为配置对象的vocab_size
        # 不使用偏置项（bias=False）
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # 调用初始化权重并应用最终处理方法
        self.post_init()

    # 返回lm_head作为输出的嵌入层对象
    def get_output_embeddings(self):
        return self.lm_head

    # 将新的嵌入层对象赋值给lm_head
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    # 如果配置对象指定了tie_word_embeddings，则共享权重
    def _tie_weights(self):
        if self.config.tie_word_embeddings:
            self._tie_or_clone_weights(self.prophetnet.word_embeddings, self.lm_head)

    # 返回prophetnet模型中的word_embeddings作为输入嵌入层对象
    def get_input_embeddings(self):
        return self.prophetnet.word_embeddings

    # 前向传播方法，接受一系列可能为空的张量作为输入参数
    @add_start_docstrings_to_model_forward(PROPHETNET_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=ProphetNetSeq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
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
    # 定义一个方法用于计算损失，输入参数包括 logits（预测值）、labels（真实标签）、ignore_index（忽略的索引，默认为-100）
    def _compute_loss(self, logits, labels, ignore_index=-100):
        # 创建一个与 labels 维度相同的全零张量，用于存储扩展后的标签，填充值为 ignore_index
        expend_targets = labels.new_zeros(self.config.ngram, labels.size(0), labels.size(1)).fill_(ignore_index)

        # 根据 config 中的 ngram 参数，扩展标签，将 labels 复制到 expend_targets 的不同维度中
        for i in range(self.config.ngram):
            if i > 0 and self.disable_ngram_loss:
                break
            expend_targets[i, :, :] = labels

        # 调整 logits 的维度顺序，并确保其连续性
        logits = logits.transpose(0, 1).contiguous()
        # 计算 log_softmax，得到 lprobs，用于后续的负对数似然损失计算
        lprobs = nn.functional.log_softmax(
            logits.view(-1, logits.size(-1)),  # 展平 logits 张量的前两个维度
            dim=-1,
            dtype=torch.float32,
        )

        # 使用负对数似然损失函数计算损失值，reduction 参数为 "mean" 表示计算平均损失
        loss = nn.functional.nll_loss(lprobs, expend_targets.view(-1), reduction="mean")

        # 如果 config 中的 eps 大于 0.0，则执行 label 平滑操作
        if self.config.eps > 0.0:
            # 计算平滑损失
            smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
            non_masked_tokens = expend_targets.ne(ignore_index).view(-1)
            smooth_loss = smooth_loss[non_masked_tokens]
            smooth_loss = smooth_loss.mean()

            # 计算 eps_i
            eps_i = self.config.eps / lprobs.size(-1)
            # 结合 label 平滑和原始损失值，得到最终的损失值
            loss = (1.0 - self.config.eps) * loss + eps_i * smooth_loss

        # 返回计算得到的损失值
        return loss

    # 为生成准备输入的方法，返回一个包含所需输入的字典
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
        # 断言 encoder_outputs 参数不为 None，确保其在生成时被传递
        assert encoder_outputs is not None, "`encoder_outputs` have to be passed for generation."

        # 如果 past_key_values 存在，仅保留 decoder_input_ids 的最后一个 token
        if past_key_values:
            decoder_input_ids = decoder_input_ids[:, -1:]

        # 返回包含生成所需输入的字典
        return {
            "input_ids": None,  # encoder_outputs 已定义，不需要 input_ids
            "encoder_outputs": encoder_outputs,
            "past_key_values": past_key_values,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
        }

    # 根据标签准备 decoder_input_ids 的静态方法
    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return self._shift_right(labels)

    # 从 past_key_values 中重新排序缓存的静态方法，用于 beam search 生成
    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            # 对每一层的过去状态执行重新排序，以适应 beam search 的索引变化
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past[:2])
                + layer_past[2:],
            )
        return reordered_past

    # 获取 encoder 的方法，返回 prophetnet 模型的 encoder 部分
    def get_encoder(self):
        return self.prophetnet.encoder

    # 获取 decoder 的方法，返回 prophetnet 模型的 decoder 部分
    def get_decoder(self):
        return self.prophetnet.decoder
# 为 ProphetNetForCausalLM 类添加文档字符串，描述其作为 ProphetNetModel 的解码器部分，用于有因果关系的语言建模
@add_start_docstrings(
    "The standalone decoder part of the ProphetNetModel with a lm head on top. The model can be used for causal"
    " language modeling.",
    PROPHETNET_START_DOCSTRING,
)
class ProphetNetForCausalLM(ProphetNetPreTrainedModel):
    # 定义绑定权重的关键词列表，用于共享或复制权重
    _tied_weights_keys = [
        "prophetnet.word_embeddings.weight",
        "prophetnet.decoder.word_embeddings.weight",
        "lm_head.weight",
    ]

    # 初始化方法，接收 ProphetNetConfig 类型的配置参数
    def __init__(self, config: ProphetNetConfig):
        # 深拷贝配置对象，设置为解码器模式，关闭编码-解码模式
        config = copy.deepcopy(config)
        config.is_decoder = True
        config.is_encoder_decoder = False
        # 调用父类初始化方法
        super().__init__(config)
        # 创建 ProphetNetDecoderWrapper 对象
        self.prophetnet = ProphetNetDecoderWrapper(config)

        # 设置填充 token 的索引
        self.padding_idx = config.pad_token_id
        # 是否禁用 ngram 损失的标志
        self.disable_ngram_loss = config.disable_ngram_loss

        # 创建线性层 lm_head，用于预测词汇表中词的概率分布
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # 调用初始化权重和应用最终处理的方法
        self.post_init()

    # 获取输入嵌入的方法，返回 ProphetNet 解码器的词嵌入层
    def get_input_embeddings(self):
        return self.prophetnet.decoder.word_embeddings

    # 设置输入嵌入的方法，设置 ProphetNet 解码器的词嵌入层
    def set_input_embeddings(self, value):
        self.prophetnet.decoder.word_embeddings = value

    # 获取输出嵌入的方法，返回 lm_head 线性层，用于预测词汇表中词的概率分布
    def get_output_embeddings(self):
        return self.lm_head

    # 设置输出嵌入的方法，设置 lm_head 线性层
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    # 绑定权重的方法，如果配置指定了共享词嵌入，则共享 ProphetNet 解码器的词嵌入层和 lm_head 线性层
    def _tie_weights(self):
        if self.config.tie_word_embeddings:
            self._tie_or_clone_weights(self.prophetnet.decoder.word_embeddings, self.lm_head)

    # 设置解码器的方法，用给定的解码器替换当前的 ProphetNet 解码器
    def set_decoder(self, decoder):
        self.prophetnet.decoder = decoder

    # 获取解码器的方法，返回当前 ProphetNet 解码器
    def get_decoder(self):
        return self.prophetnet.decoder

    # 前向传播方法，执行 ProphetNet 解码器的前向传播，预测下一个词的分布
    @add_start_docstrings_to_model_forward(PROPHETNET_STANDALONE_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=ProphetNetDecoderLMOutput, config_class=_CONFIG_FOR_DOC)
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
        # 前向传播的参数列表，支持 ProphetNetDecoderLMOutput 类型的输出
        **kwargs,
    ):
    def _compute_loss(self, logits, labels, ignore_index=-100):
        # 创建一个与labels具有相同大小的张量，填充为ignore_index，用于扩展目标张量
        expend_targets = labels.new_zeros(self.config.ngram, labels.size(0), labels.size(1)).fill_(ignore_index)

        for i in range(self.config.ngram):
            # 如果当前ngram大于0并且禁用了ngram损失，则退出循环
            if i > 0 and self.disable_ngram_loss:
                break
            # 将labels复制到扩展目标张量的第i层
            expend_targets[i, :, :] = labels

        # 调整logits的维度顺序，并确保内存连续
        logits = logits.transpose(0, 1).contiguous()
        # 计算log_softmax以获取概率对数
        lprobs = nn.functional.log_softmax(
            logits.view(-1, logits.size(-1)),
            dim=-1,
            dtype=torch.float32,
        )

        # 计算负对数似然损失
        loss = nn.functional.nll_loss(lprobs, expend_targets.view(-1), reduction="mean")

        if self.config.eps > 0.0:
            # 计算平滑损失，排除掩码标记，并计算平均值
            smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
            non_masked_tokens = expend_targets.ne(ignore_index).view(-1)
            smooth_loss = smooth_loss[non_masked_tokens]
            smooth_loss = smooth_loss.mean()

            # 计算eps_i
            eps_i = self.config.eps / lprobs.size(-1)
            # 应用平滑损失到总损失中
            loss = (1.0 - self.config.eps) * loss + eps_i * smooth_loss

        # 返回最终的损失值
        return loss

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        head_mask=None,
        use_cache=None,
        **kwargs,
    ):
        # 如果attention_mask为空，则创建全为1的张量，表示所有token都被attention
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_ids.shape)

        if past_key_values:
            # 如果past_key_values存在，则仅使用最后一个token作为输入
            input_ids = input_ids[:, -1:]
        
        # 返回用于生成的输入字典
        return {
            "input_ids": input_ids,  # encoder_outputs is defined. input_ids not needed
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
        }

    @staticmethod
    # 重新排序缓存中的过去键值，以匹配beam search的顺序
    # 从transformers.models.bart.modeling_bart.BartForCausalLM._reorder_cache复制而来
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            # 根据beam_idx重新排序每一层的过去状态
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past
# 定义一个名为 ProphetNetDecoderWrapper 的类，继承自 ProphetNetPreTrainedModel 类
class ProphetNetDecoderWrapper(ProphetNetPreTrainedModel):
    """
    This is a wrapper class, so that [`ProphetNetForCausalLM`] can correctly be loaded from pretrained prophetnet
    classes.
    """

    # 初始化方法，接受一个 ProphetNetConfig 类型的参数 config
    def __init__(self, config: ProphetNetConfig):
        # 调用父类的初始化方法，传入 config 参数
        super().__init__(config)

        # 创建一个 nn.Embedding 对象，用于词嵌入，参数包括词汇表大小、隐藏层大小和填充标记的索引
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        
        # 创建 ProphetNetDecoder 对象，传入 config 参数和之前创建的词嵌入对象
        self.decoder = ProphetNetDecoder(config, word_embeddings=self.word_embeddings)

        # 初始化权重并应用最终处理
        self.post_init()

    # 方法，用于将词嵌入层的权重与解码器的输入词嵌入层权重相绑定
    def _tie_weights(self):
        self._tie_or_clone_weights(self.word_embeddings, self.decoder.get_input_embeddings())

    # 前向传播方法，将调用解码器的前向传播方法
    def forward(self, *args, **kwargs):
        return self.decoder(*args, **kwargs)
```