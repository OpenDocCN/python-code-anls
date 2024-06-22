# `.\transformers\models\prophetnet\modeling_prophetnet.py`

```py
# 设定文件编码为 UTF-8
# 版权声明，指出代码的版权及使用许可
# 此处代码使用 Apache License 2.0 授权
# 详细许可内容请查阅 http://www.apache.org/licenses/LICENSE-2.0
# 如果没有遵守许可条款，除非法律要求或书面同意，否则不得使用此文件
""" PyTorch ProphetNet 模型，从 ProphetNet 仓库（fairseq 版本）移植而来。"""

# 导入所需的库
import copy  # 导入 copy 库，用于深度拷贝对象
import math  # 导入 math 库，用于数学计算
import warnings  # 导入 warnings 库，用于处理警告信息
from dataclasses import dataclass  # 导入 dataclasses 库中的 dataclass 装饰器
from typing import Optional, Tuple, Union  # 导入 typing 库中的类型标注

# 导入 PyTorch 相关库
import torch  # 导入 PyTorch 库
import torch.utils.checkpoint  # 导入 PyTorch 库中的 checkpoint 模块
from torch import Tensor, nn  # 从 PyTorch 中导入 Tensor 和 nn（神经网络）模块
from torch.nn import LayerNorm  # 从 PyTorch 中导入 LayerNorm 层

# 导入 Hugging Face 相关库
from ...activations import ACT2FN  # 从 Hugging Face 库中导入 ACT2FN 激活函数
from ...modeling_outputs import BaseModelOutput  # 从 Hugging Face 库中导入 BaseModelOutput 类
from ...modeling_utils import PreTrainedModel  # 从 Hugging Face 库中导入 PreTrainedModel 类
from ...utils import (  # 从 Hugging Face 库中导入一系列工具函数和类
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from .configuration_prophetnet import ProphetNetConfig  # 从当前目录中导入 ProphetNetConfig 配置类

# 获取日志记录器
logger = logging.get_logger(__name__)

# 用于文档的配置和模型检查点信息
_CONFIG_FOR_DOC = "ProphenetConfig"
_CHECKPOINT_FOR_DOC = "microsoft/prophetnet-large-uncased"

# ProphetNet 预训练模型存档列表
PROPHETNET_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "microsoft/prophetnet-large-uncased",
    # 查看所有 ProphetNet 模型，请访问 https://huggingface.co/models?filter=prophetnet
]

# ProphetNet 文档的起始字符串
PROPHETNET_START_DOCSTRING = r"""
    此模型继承自 [`PreTrainedModel`]。请查看超类文档以了解库实现的通用方法（如下载或保存、调整输入嵌入、修剪头等）。

    原始 ProphetNet 代码可在 [这里](https://github.com/microsoft/ProphetNet) 找到。检查点是从原始 Fairseq 检查点转换而来的。有关检查点转换的更多信息，请查看文件 `convert_prophetnet_original_pytorch_checkpoint_to_pytorch.py`。

    此模型是 PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) 的子类。将其用作常规 PyTorch 模块，并参考 PyTorch 文档以获取有关一般用法和行为的所有相关信息。

    参数:
        config ([`ProphetNetConfig`]): 带有模型所有参数的配置类。
            使用配置文件初始化不会加载与模型相关的权重，只加载配置。请查看 [`~PreTrainedModel.from_pretrained`] 方法以加载模型权重。
"""

# ProphetNet 输入文档字符串
PROPHETNET_INPUTS_DOCSTRING = r"""
"""

# ProphetNet 独立输入文档字符串
PROPHETNET_STANDALONE_INPUTS_DOCSTRING = r"""
    # 处理输入的参数
    Args:
        # 输入序列的tokens在词汇表中的索引，padding的部分将被忽略
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.
    
            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.
    
            [What are input IDs?](../glossary#input-ids)
        # 注意力mask，避免在padding token上进行注意力计算
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
    
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
    
            [What are attention masks?](../glossary#attention-mask)
        # 编码器层注意力头的掩码，用于将某些头置为0
        head_mask (`torch.Tensor` of shape `(encoder_layers, encoder_attention_heads)`, *optional*):
            Mask to nullify selected heads of the attention modules in the encoder. Mask values selected in `[0, 1]`:
    
            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
    
        # 是否返回所有注意力层的张量
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        # 是否返回所有层的隐藏状态
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        # 是否返回ModelOutput对象
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
# 定义 softmax 函数，用于计算概率分布
def softmax(hidden_state, dim, onnx_trace=False):
    # 如果处于 ONNX 追踪状态，则将隐藏状态转换为 float 类型后进行 softmax
    if onnx_trace:
        return nn.functional.softmax(hidden_state.float(), dim=dim)
    else:
        # 否则直接对隐藏状态进行 softmax，指定数据类型为 torch.float32
        return nn.functional.softmax(hidden_state, dim=dim, dtype=torch.float32)


# 计算 n-gram 注意力偏置
def ngram_attention_bias(sequence_length, ngram, device, dtype):
    """
    This function computes the bias for the predict stream
    """
    # 创建左侧和右侧的矩阵块
    left_block = (
        torch.ones((ngram, sequence_length, sequence_length), device=device, dtype=dtype) * torch.finfo(dtype).min
    )
    right_block = left_block.detach().clone()
    # 生成偏置
    for stream_idx in range(ngram):
        right_block[stream_idx].fill_diagonal_(0, wrap=False)
        left_block[stream_idx].triu_(-stream_idx + 1)

    # 左侧矩阵的第一列设为零
    left_block[:, :, 0] = 0
    # 将左右两个块连接起来形成最终的注意力偏置矩阵
    return torch.cat([left_block, right_block], dim=2)


# 计算相对位置桶的各个部分
def compute_relative_buckets(num_buckets, max_distance, relative_positions, is_bidirectional=False):
    """
    This function computes individual parts of the relative position buckets. For more detail, see paper.
    """
    # 计算相对位置的负值
    inv_relative_positions = -relative_positions
    rel_positions_bucket = 0

    # 如果是双向的注意力，则处理负数相对位置
    if is_bidirectional:
        num_buckets = num_buckets // 2
        # 对于负数相对位置，分配到前一半的桶中
        rel_positions_bucket = (
            rel_positions_bucket
            + torch.lt(inv_relative_positions, torch.zeros_like(inv_relative_positions)).int() * num_buckets
        )
        inv_relative_positions = torch.abs(inv_relative_positions)
    else:
        # 对于单向注意力，将负数相对位置转换为非负数
        inv_relative_positions = torch.max(inv_relative_positions, torch.zeros_like(inv_relative_positions))

    max_exact = num_buckets // 2
    is_small = torch.lt(inv_relative_positions, max_exact)
    # 计算相对位置的桶索引
    val_if_large = max_exact + torch.log(inv_relative_positions.float() / max_exact) / math.log(
        max_distance / max_exact
    ) * (num_buckets - max_exact)
    val_if_large = torch.min(val_if_large, torch.ones_like(val_if_large) * (num_buckets - 1)).int()
    rel_positions_bucket = rel_positions_bucket + torch.where(is_small, inv_relative_positions.int(), val_if_large)
    return rel_positions_bucket


# 计算所有流的相对位置桶
def compute_all_stream_relative_buckets(num_buckets, max_distance, position_ids):
    """
    This function computes both main and predict relative position buckets. For more detail, see paper.
    """
    # 主流
    main_stream_relative_positions = position_ids.unsqueeze(1).repeat(1, position_ids.size(-1), 1)
    main_stream_relative_positions = main_stream_relative_positions - position_ids.unsqueeze(-1)

    # 预测流
    predicting_stream_relative_positions = torch.cat((position_ids - 1, position_ids), dim=-1).unsqueeze(1)
    predicting_stream_relative_positions = predicting_stream_relative_positions.repeat(1, position_ids.size(-1), 1)
    predicting_stream_relative_positions = predicting_stream_relative_positions - position_ids.unsqueeze(-1)

    # 获取主流和预测流的相对位置桶
    # 计算主流相对位置的桶（buckets），主要应用于非双向的情况
    main_relative_position_buckets = compute_relative_buckets(
        num_buckets, max_distance, main_stream_relative_positions, is_bidirectional=False
    )
    # 计算预测流相对位置的桶（buckets），主要应用于非双向的情况
    predict_relative_position_buckets = compute_relative_buckets(
        num_buckets, max_distance, predicting_stream_relative_positions, is_bidirectional=False
    )
    # 返回主流相对位置的桶和预测流相对位置的桶
    return main_relative_position_buckets, predict_relative_position_buckets
# 定义一个具有数据层属性的类，用于存储ProphetNet序列到序列语言模型的输出
@dataclass
class ProphetNetSeq2SeqLMOutput(ModelOutput):
    """
    Base class for sequence-to-sequence language models outputs.

    """

    # 损失值，默认为None
    loss: Optional[torch.FloatTensor] = None
    # 对数值，表示输出的概率分布，默认为None
    logits: torch.FloatTensor = None
    # 对数值，表示输出的ngram概率分布，默认为None
    logits_ngram: Optional[torch.FloatTensor] = None
    # 过去的键值对，用于加速顺序解码，默认为None
    past_key_values: Optional[Tuple[torch.FloatTensor]] = None
    # 解码器的隐藏状态，默认为None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # 解码器的ngram隐藏状态，默认为None
    decoder_ngram_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # 解码器的注意力，默认为None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    # 解码器的ngram注意力，默认为None
    decoder_ngram_attentions: Optional[Tuple[torch.FloatTensor]] = None
    # 交叉注意力，默认为None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    # 编码器的最终隐藏状态，默认为None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    # 编码器的隐藏状态，默认为None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # 编码器的注意力，默认为None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None

    # 解码器的交叉注意力，返回交叉注意力并发出警告
    @property
    def decoder_cross_attentions(self):
        warnings.warn(
            "`decoder_cross_attentions` is deprecated and will be removed soon. Please use `cross_attentions`"
            " instead.",
            FutureWarning,
        )
        return self.cross_attentions


@dataclass
# 定义一个具有数据层属性的类，用于存储ProphetNet序列到序列模型的输出
class ProphetNetSeq2SeqModelOutput(ModelOutput):
    """
    Base class for model encoder's outputs that also contains : pre-computed hidden states that can speed up sequential
    decoding.

    """

    # 最终的隐藏状态，默认为None
    last_hidden_state: torch.FloatTensor
    # ngram的最终隐藏状态，默认为None
    last_hidden_state_ngram: Optional[torch.FloatTensor] = None
    # 过去的键值对，用于加速顺序解码，默认为None
    past_key_values: Optional[Tuple[torch.FloatTensor]] = None
    # 解码器的隐藏状态，默认为None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # 解码器的ngram隐藏状态，默认为None
    decoder_ngram_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # 解码器的注意力，默认为None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    # 解码器的ngram注意力，默认为None
    decoder_ngram_attentions: Optional[Tuple[torch.FloatTensor]] = None
    # 交叉注意力，默认为None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    # 编码器的最终隐藏状态，默认为None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    # 编码器的隐藏状态，默认为None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # 编码器的注意力，默认为None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None

    # 解码器的交叉注意力，返回交叉注意力并发出警告
    @property
    def decoder_cross_attentions(self):
        warnings.warn(
            "`decoder_cross_attentions` is deprecated and will be removed soon. Please use `cross_attentions`"
            " instead.",
            FutureWarning,
        )
        return self.cross_attentions


@dataclass
# 为模型的输出定义一个具有数据层属性的类，也可能包含过去的键/值（以加速顺序解码）
class ProphetNetDecoderModelOutput(ModelOutput):
    """
    Base class for model's outputs that may also contain a past key/values (to speed up sequential decoding).

    """

    # 最终的隐藏状态，默认为None
    last_hidden_state: torch.FloatTensor
    # ngram的最终隐藏状态，默认为None
    last_hidden_state_ngram: Optional[torch.FloatTensor] = None
    # 过去的键值对，用于加速顺序解码，默认为None
    past_key_values: Optional[Tuple[torch.FloatTensor]] = None
    # 隐藏状态，默认为None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # ngram的隐藏状态，默认为None
    hidden_states_ngram: Optional[Tuple[torch.FloatTensor]] = None
    # 注意力，默认为None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    # ngram的注意力，默认为None
    ngram_attentions: Optional[Tuple[torch.FloatTensor]] = None
    # 定义名为cross_attentions的可选元组变量，类型为torch.FloatTensor，初始值为None。
# 创建一个数据类，用于存储模型的输出
@dataclass
class ProphetNetDecoderLMOutput(ModelOutput):
    """
    Base class for model's outputs that may also contain a past key/values (to speed up sequential decoding).

    """
    # 损失值，可选的浮点数张量
    loss: Optional[torch.FloatTensor] = None
    # 输出的logits张量
    logits: torch.FloatTensor = None
    # ngram相关的logits张量，可选的浮点数张量
    logits_ngram: Optional[torch.FloatTensor] = None
    # 预测键/值对，以加速顺序解码的模型输出，可选的一对张量
    past_key_values: Optional[Tuple[torch.FloatTensor]] = None
    # 隐藏状态的元组，可选的一对张量
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # ngram隐藏状态的元组，可选的一对张量
    hidden_states_ngram: Optional[Tuple[torch.FloatTensor]] = None
    # 注意力分布的元组，可选的一对张量
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    # ngram的注意力分布的元组，可选的一对张量
    ngram_attentions: Optional[Tuple[torch.FloatTensor]] = None
    # 交叉注意力分布的元组，可选的一对张量
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None


# ProphetNet模型的预训练类的定义
class ProphetNetPreTrainedModel(PreTrainedModel):
    # ProphetNet配置的类
    config_class = ProphetNetConfig
    # 模型前缀
    base_model_prefix = "prophetnet"
    # 是否支持梯度检查点
    supports_gradient_checkpointing = True

    # 初始化权重的方法
    def _init_weights(self, module):
        # 如果是线性层，初始化其权重和偏置
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.init_std)
            if module.bias is not None:
                module.bias.data.zero_()
        # 如果是嵌入层，初始化其权重
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.init_std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    # 将输入向右移动的方法
    def _shift_right(self, input_ids):
        decoder_start_token_id = self.config.decoder_start_token_id
        pad_token_id = self.config.pad_token_id

        assert decoder_start_token_id is not None, (
            "self.model.config.decoder_start_token_id has to be defined. In ProphetNet it is usually set to the"
            " pad_token_id. See ProphetNet docs for more information"
        )

        # 将输入向右移动
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
        shifted_input_ids[..., 0] = decoder_start_token_id

        assert pad_token_id is not None, "self.model.config.pad_token_id has to be defined."
        # 用`pad_token_id`替换标签中可能的-100值
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

        assert torch.all(shifted_input_ids >= 0).item(), "Verify that `shifted_input_ids` has only positive values"

        return shifted_input_ids


# ProphetNet定位嵌入的定义
class ProphetNetPositionalEmbeddings(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size. Padding ids are ignored by either offsetting
    based on padding_idx or by setting padding_idx to None and ensuring that the appropriate position ids are passed to
    the forward function.
    """
    # 初始化方法
    def __init__(self, config: ProphetNetConfig) -> None:
        # 最大长度为ProphetNet配置的最大位置嵌入
        self.max_length = config.max_position_embeddings
        super().__init__(config.max_position_embeddings, config.hidden_size, config.pad_token_id)
    # 定义前向传播函数，用于模型的正向推理
    def forward(self, inputs_shape, device, attention_mask=None, past_key_values=None, position_ids=None):
        # 断言：如果position_ids为None，则padding_idx也必须为None；确保推理时position_ids是预先计算的
        assert (position_ids is None) or (
            self.padding_idx is None
        ), "If position_ids is pre-computed then padding_idx should not be set."

        # 如果position_ids为None
        if position_ids is None:
            # 如果past_key_values不为None，则在解码单个步骤时position_ids对每个标记是相同的
            if past_key_values is not None:
                # 获取先前输入id的数量，用于计算新的position_ids
                prev_num_input_ids = past_key_values[0][0].shape[2]
                num_input_ids = inputs_shape[1] + prev_num_input_ids
                # 计算position_ids，加上padding_idx保证唯一性，并转换为长整型
                position_ids = torch.ones((1, 1), dtype=torch.long, device=device) * (
                    int(self.padding_idx + num_input_ids)
                )
            else:
                # 如果attention_mask为None，则初始化一个全1的attention_mask
                if attention_mask is None:
                    attention_mask = torch.ones(inputs_shape, dtype=torch.long, device=device)

                # 从input_ids/attention_mask中检索position_ids
                # 通过对attention_mask进行累积求和，并转换为与attention_mask相同的类型，再加上padding_idx
                position_ids = (
                    torch.cumsum(attention_mask, dim=1).type_as(attention_mask) * attention_mask
                ).long() + self.padding_idx

                # 确保position_ids不大于最大长度max_length-1
                position_ids = position_ids.clamp(0, self.max_length - 1)

        # 调用父类的forward方法进行前向传播，并返回结果和计算得到的position_ids
        return super().forward(position_ids), position_ids

    # 私有方法，用于执行模型的前向传播
    def _forward(self, position_ids):
        # 调用父类的forward方法执行前向传播，并返回结果
        return super().forward(position_ids)
# 定义了一个ProphetNetAttention类，用于实现多头注意力机制
class ProphetNetAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        config: ProphetNetConfig,
        num_attn_heads: int,
    ):
        super().__init__()
        hidden_size = config.hidden_size

        self.attention_dropout = config.attention_dropout  # 设置注意力机制的dropout
        self.dropout = config.dropout  # 设置dropout
        self.num_attn_heads = num_attn_heads  # 设置注意力头数
        self.head_dim = hidden_size // num_attn_heads  # 计算每个注意力头的维度

        assert self.head_dim * num_attn_heads == hidden_size, (  # 断言确保hidden_size能被num_attn_heads整除
            "`config.hidden_size` must be divisible by `config.num_encoder_attention_heads` and"
            " `config.num_decoder_attention_heads`"
        )

        self.key_proj = nn.Linear(hidden_size, hidden_size)  # key的线性变换
        self.value_proj = nn.Linear(hidden_size, hidden_size)  # value的线性变换
        self.query_proj = nn.Linear(hidden_size, hidden_size)  # query的线性变换

        self.out_proj = nn.Linear(hidden_size, hidden_size)  # 输出的线性变换

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_attn_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states,
        key_value_states: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        layer_head_mask: Optional[Tensor] = None,
        past_key_value: Optional[Tuple[Tensor]] = None,
        output_attentions: bool = False,



# 定义了一个ProphetNetFeedForward类，用于实现Transformer中的前馈神经网络层
class ProphetNetFeedForward(nn.Module):
    """
    This is the residual two feed-forward layer block based on the original Transformer implementation.
    """

    def __init__(self, config: ProphetNetConfig, ffn_dim: int):
        super().__init__()
        self.activation_fn = ACT2FN[config.activation_function]  # 设置激活函数
        self.intermediate = nn.Linear(config.hidden_size, ffn_dim)  # 中间层的线性变换
        self.output = nn.Linear(ffn_dim, config.hidden_size)  # 输出层的线性变换
        self.activation_dropout = config.activation_dropout  # 设置激活函数的dropout
        self.dropout = config.dropout  # 设置dropout

    def forward(self, hidden_states):
        hidden_states = self.intermediate(hidden_states)  # 中间层的线性变换
        hidden_states = self.activation_fn(hidden_states)  # 激活函数
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)  # 激活函数的dropout
        hidden_states = self.output(hidden_states)  # 输出层的线性变换
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)  # dropout
        return hidden_states
    # 初始化函数，接受一个ProphetNetConfig对象作为参数
    def __init__(self, config: ProphetNetConfig):
        # 调用父类的初始化函数
        super().__init__()
        # 从配置对象中获取隐藏层大小
        self.hidden_size = config.hidden_size
    
        # 从配置对象中获取桶的数量
        self.num_buckets = config.num_buckets
        # 从配置对象中获取相对位置的最大距离
        self.relative_max_distance = config.relative_max_distance
        # 从配置对象中获取解码器注意力头的数量
        self.num_attn_heads = config.num_decoder_attention_heads
        # 从配置对象中获取普通的dropout率
        self.dropout = config.dropout
        # 从配置对象中获取注意力机制的dropout率
        self.attention_dropout = config.attention_dropout
        # 计算每个注意力头的维度
        self.head_dim = config.hidden_size // self.num_attn_heads
        # 从配置对象中获取ngram值
        self.ngram = config.ngram
    
        # 断言确保隐藏层大小能被注意力头的数量整除
        assert (
            self.head_dim * self.num_attn_heads == config.hidden_size
        ), "config.hidden_size must be divisible by num_attn_heads"
        # 创建线性层，用于键的投影
        self.key_proj = nn.Linear(config.hidden_size, config.hidden_size)
        # 创建线性层，用于值的投影
        self.value_proj = nn.Linear(config.hidden_size, config.hidden_size)
        # 创建线性层，用于查询的投影
        self.query_proj = nn.Linear(config.hidden_size, config.hidden_size)
    
        # 创建输出投影线性层
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)
    
        # 创建相对位置嵌入的线性层
        self.relative_pos_embeddings = nn.Linear(config.hidden_size, self.num_buckets * self.num_attn_heads)
    
        # 用于ONNX运行时的标志
        self.onnx_trace = False
    
    # 将张量重新形状，以适应多头注意力的输入
    def _shape(self, tensor, seq_len, batch_size):
        return tensor.view(batch_size, seq_len, self.num_attn_heads, self.head_dim).transpose(1, 2).contiguous()
    
    # 准备导出为ONNX格式
    def prepare_for_onnx_export_(self):
        self.onnx_trace = True
    
    # 前向传播函数，接受隐藏状态、过去键值对、注意力掩码等作为输入
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
    # 获取主要相对位置嵌入
    def get_main_relative_pos_embeddings(
        self, hidden_states, attn_weights, position_ids, main_relative_position_buckets
    ):
        # input hidden_states [batch_size, sequence_length, hidden_size]
        # input attn_weights [batch_size, num_heads, sequence_length, sequence_length]
        # input position_ids [batch_size, sequence_length] or [1,1]
        batch_size, num_attn_heads, tgt_len, src_len = attn_weights.shape
        # 重塑注意力权重的形状
        attn_weights = attn_weights.view(batch_size, num_attn_heads, tgt_len, src_len)
        # 如果主要相对位置桶为空，计算相对位置
        if main_relative_position_buckets is None:
            batch_size, sequence_length = hidden_states.shape[:2]
            relative_positions = (
                torch.arange(1, attn_weights.shape[-1] + 1)
                .unsqueeze(0)
                .unsqueeze(0)
                .repeat(batch_size, sequence_length, 1)
                .to(position_ids.device)
            )
            # [batch_size, sequence_length, sequence_length+1]
            relative_positions = relative_positions - position_ids.unsqueeze(0).repeat(batch_size, sequence_length, 1)
            main_relative_position_buckets = compute_relative_buckets(
                self.num_buckets, self.relative_max_distance, relative_positions, False
            )

        # 计算相对位置嵌入
        # [batch_size, sequence_length, num_buckets * num_heads]
        rel_pos_embeddings = self.relative_pos_embeddings(hidden_states)
        rel_pos_embeddings = rel_pos_embeddings.view(
            rel_pos_embeddings.shape[:2] + (self.num_buckets, self.num_attn_heads)
        )
        rel_pos_embeddings = rel_pos_embeddings.permute(0, 3, 1, 2)
        # [batch_size, num_heads, sequence_length, num_buckets]
        rel_pos_embeddings = rel_pos_embeddings.reshape(attn_weights.shape[:3] + (-1,))

        main_relative_position_buckets = main_relative_position_buckets.repeat(1, self.num_attn_heads, 1)
        # [batch_size * num_heads * sequence_length, sequence_length]
        main_relative_position_buckets = main_relative_position_buckets.view(
            -1, main_relative_position_buckets.shape[-1]
        )
        main_relative_position_buckets = main_relative_position_buckets.long()
        # [batch_size * num_heads * sequence_length, sequence_length]
        rel_pos_embeddings = rel_pos_embeddings.reshape(-1, rel_pos_embeddings.size(-1))

        main_relative_pos_embeddings = torch.gather(rel_pos_embeddings, dim=1, index=main_relative_position_buckets)
        main_relative_pos_embeddings = main_relative_pos_embeddings.view(batch_size, num_attn_heads, tgt_len, -1)
        return main_relative_pos_embeddings

    def get_predict_relative_pos_embeddings(
        self, hidden_states, attn_weights, position_ids, predict_relative_position_buckets
    # 输入隐藏状态为 [batch_size, sequence_length, ngram, hidden_size]
    # 输入注意力权重为 [batch_size, ngram, num_heads, sequence_length, 2*sequence_length]
    # 输入位置ID为 [batch_size, sequence_length] 或 [1,1]
    # 输入预测相对位置桶为 [batch_size, sequence_length, 2*sequence_length] 或 None
    batch_size, sequence_length = hidden_states.shape[0:2]
    
    # 如果预测相对位置桶为None，则根据注意力权重的形状创建
    if predict_relative_position_buckets is None:
        # 获取注意力权重的长度
        key_sequence_length = attn_weights.shape[-1]
        # 确保位置ID的格式为 1 2 3 4 5 ... (key_sequence_length - 1)
        assert (
            position_ids[0][0] == key_sequence_length - 1
        ), "`position_ids` are incorrect. They should be of the format 1 2 3 4 5 ... (key_sequence_length - 1)"
        # 创建相对位置张量，形状为 [batch_size, sequence_length, key_sequence_length]
        relative_positions = (
            torch.arange(0, key_sequence_length)
            .unsqueeze(0)
            .unsqueeze(0)
            .repeat(batch_size, sequence_length, 1)
            .to(position_ids.device)
        )
    
        # 计算相对位置
        relative_positions = relative_positions - position_ids.unsqueeze(0).repeat(batch_size, sequence_length, 1)
        # 计算预测的相对位置桶
        predict_relative_position_buckets = compute_relative_buckets(
            self.num_buckets, self.relative_max_distance, relative_positions, False
        )
    
    # 将隐藏状态转置为 [batch_size, ngram, sequence_length, hidden_size]
    hidden_states = hidden_states.transpose(1, 2)
    # 使用相对位置嵌入层计算相对位置嵌入
    rel_pos_embeddings = self.relative_pos_embeddings(hidden_states)
    
    # 重新形状相对位置嵌入为 [batch_size, ngram, sequence_length, num_buckets, num_heads]
    rel_pos_embeddings = rel_pos_embeddings.view(
        hidden_states.shape[:-1] + (self.num_buckets, self.num_attn_heads)
    )
    # 交换维度为 [batch_size, ngram, sequence_length, num_heads, num_buckets]
    rel_pos_embeddings = rel_pos_embeddings.permute(0, 2, 1, 4, 3)
    # 重新形状为 [batch_size * ngram * sequence_length * num_heads, num_buckets]
    rel_pos_embeddings = rel_pos_embeddings.reshape(-1, self.num_buckets)
    # 重新形状预测的相对位置桶为 [ngram, batch_size, num_heads * sequence_length, -1]
    predict_relative_position_buckets = predict_relative_position_buckets.unsqueeze(0)
    predict_relative_position_buckets = predict_relative_position_buckets.repeat(
        self.ngram, 1, self.num_attn_heads, 1
    )
    # 重新形状为 [ngram * batch_size * num_heads * sequence_length, -1]
    predict_relative_position_buckets = predict_relative_position_buckets.view(
        -1, predict_relative_position_buckets.size(-1)
    ).long()
    
    # 使用torch.gather函数根据相对位置桶索引获取预测的相对位置嵌入
    predict_relative_pos_embeddings = torch.gather(
        rel_pos_embeddings, dim=1, index=predict_relative_position_buckets
    )
    
    # 重新形状为 [batch_size, gram, num_heads, sequence_length, -1]
    predict_relative_pos_embeddings = predict_relative_pos_embeddings.view(
        batch_size, self.ngram, self.num_attn_heads, sequence_length, -1
    )
    
    # 返回预测的相对位置嵌入
    return predict_relative_pos_embeddings
class ProphetNetEncoderLayer(nn.Module):
    """
    Encoder block for Prophetnet
    """

    def __init__(self, config: ProphetNetConfig):
        super().__init__()
        # 1st residual block
        # 定义自注意力机制模块
        self.self_attn = ProphetNetAttention(config, config.num_encoder_attention_heads)
        # 定义自注意力机制后的 LayerNorm 层
        self.self_attn_layer_norm = LayerNorm(config.hidden_size)

        # 2nd residual block
        # 定义前馈神经网络模块
        self.feed_forward = ProphetNetFeedForward(config, config.encoder_ffn_dim)
        # 定义前馈神经网络后的 LayerNorm 层
        self.feed_forward_layer_norm = LayerNorm(config.hidden_size)

    def forward(
        self,
        hidden_states,
        attention_mask,
        layer_head_mask,
        output_attentions: bool = False,
    ):
        # 1st residual block
        # 使用自注意力机制模块进行计算
        attention_output, attn_weights, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        # 将自注意力机制模块的输出与输入相加，并通过 LayerNorm 层进行归一化
        hidden_states = self.self_attn_layer_norm(attention_output + hidden_states)

        # 2nd residual block
        # 使用前馈神经网络模块进行计算
        feed_forward_output = self.feed_forward(hidden_states)
        # 将前馈神经网络模块的输出与输入相加，并通过 LayerNorm 层进行归一化
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
        # 定义 n-gram 自注意力机制模块
        self.self_attn = ProphetNetNgramSelfAttention(config)
        # 定义 n-gram 自注意力机制后的 LayerNorm 层
        self.self_attn_layer_norm = LayerNorm(config.hidden_size)

        # 2nd residual block
        # 如果配置中包含跨层注意力机制，则定义跨层注意力机制模块
        if config.add_cross_attention:
            self.cross_attn = ProphetNetAttention(config, config.num_decoder_attention_heads)
            # 定义跨层注意力机制后的 LayerNorm 层
            self.cross_attn_layer_norm = LayerNorm(config.hidden_size)

        # 3rd residual block
        # 定义解码器的前馈神经网络模块
        self.feed_forward = ProphetNetFeedForward(config, config.decoder_ffn_dim)
        # 定义前馈神经网络后的 LayerNorm 层
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
        # 提取缓存的自注意力键/值元组，位置在1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        # 调用自注意力机制
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
        # 更新隐藏状态
        hidden_states = self.self_attn_layer_norm(hidden_states + ngram_attention_output)

        # 提取缓存的交叉注意力键/值元组，位置在3,4
        cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
        cross_attn_weights = None
        if encoder_hidden_states is not None:
            # 2nd residual block
            # 调用交叉注意力机制
            attention_output, cross_attn_weights, cross_attn_present_key_value = self.cross_attn(
                hidden_states=hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attn_mask,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=cross_attn_past_key_value,
                output_attentions=output_attentions,
            )
            # 更新隐藏状态
            hidden_states = self.cross_attn_layer_norm(attention_output + hidden_states)

            # 将交叉-注意力加到 present_key_value元组的位置 3,4
            present_key_value = present_key_value + cross_attn_present_key_value

        # 3rd residual block
        # 调用前馈神经网络
        feed_forward_output = self.feed_forward(hidden_states)
        # 更新隐藏状态
        hidden_states = self.feed_forward_layer_norm(feed_forward_output + hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            # 将自注意力权重、自注意力权重-ngram、交叉注意力权重添加到输出中
            outputs += (self_attn_weights, self_attn_weights_ngram, cross_attn_weights)

        if use_cache:
            # 将present_key_value元组添加到输出中
            outputs += (present_key_value,)

        return outputs
@add_start_docstrings(
    "The standalone encoder part of the ProphetNetModel.",  # 添加起始文档字符串，说明这是 ProphetNetModel 的独立编码器部分
    PROPHETNET_START_DOCSTRING,  # 添加 ProphetNetModel 的起始文档字符串
)
class ProphetNetEncoder(ProphetNetPreTrainedModel):  # 定义 ProphetNetEncoder 类，继承自 ProphetNetPreTrainedModel
    r"""
    word_embeddings  (`torch.nn.Embeddings` of shape `(config.vocab_size, config.hidden_size)`, *optional*):
        The word embedding parameters. This can be used to initialize [`ProphetNetEncoder`] with pre-defined word
        embeddings instead of randomly initialized word embeddings.
    """
    
    def __init__(self, config: ProphetNetConfig, word_embeddings: nn.Embedding = None):  # 初始化方法，接受 ProphetNetConfig 和 word_embeddings 作为参数
        super().__init__(config)  # 调用父类的初始化方法

        self.word_embeddings = (  # 定义词嵌入层，如果提供了 word_embeddings 则使用该值，否则随机初始化词嵌入
            word_embeddings
            if word_embeddings is not None
            else nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        )
        self.position_embeddings = ProphetNetPositionalEmbeddings(config)  # 位置编码层
        self.embeddings_layer_norm = LayerNorm(config.hidden_size)  # 归一化层

        self.layers = nn.ModuleList([ProphetNetEncoderLayer(config) for _ in range(config.num_encoder_layers)])  # 多层编码器层列表

        self.gradient_checkpointing = False  # 是否使用渐变检查点
        # Initialize weights and apply final processing
        self.post_init()  # 初始化权重并进行最终处理

    def get_input_embeddings(self):  # 获取输入词嵌入层
        return self.word_embeddings

    def set_input_embeddings(self, value):  # 设置输入词嵌入层
        self.word_embeddings = value

    @add_start_docstrings_to_model_forward(PROPHETNET_STANDALONE_INPUTS_DOCSTRING)  # 添加起始文档字符串到模型前向传播方法
    @replace_return_docstrings(output_type=BaseModelOutput, config_class=_CONFIG_FOR_DOC)  # 替换返回字符串的文档注释
    def forward(  # 前向传播方法
        self,
        input_ids: Optional[torch.Tensor] = None,  # 输入的词 ID
        attention_mask: Optional[torch.Tensor] = None,  # 注意力掩码
        head_mask: Optional[torch.Tensor] = None,  # 头部掩码
        inputs_embeds: Optional[torch.Tensor] = None,  # 输入的嵌入向量
        output_attentions: Optional[bool] = None,  # 是否输出注意力值
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态
        return_dict: Optional[bool] = None,  # 是否返回字典
```  


@add_start_docstrings(
    "The standalone decoder part of the ProphetNetModel.",  # 添加起始文档字符串，说明这是 ProphetNetModel 的独立解码器部分
    PROPHETNET_START_DOCSTRING,  # 添加 ProphetNetModel 的起始文档字符串
)
class ProphetNetDecoder(ProphetNetPreTrainedModel):  # 定义 ProphetNetDecoder 类，继承自 ProphetNetPreTrainedModel
    r"""
    word_embeddings  (`torch.nn.Embeddings` of shape `(config.vocab_size, config.hidden_size)`, *optional*):
        The word embedding parameters. This can be used to initialize [`ProphetNetEncoder`] with pre-defined word
        embeddings instead of randomly initialized word embeddings.
    """

```py  
    # ProphetNetDecoder 的初始化函数，接受 ProphetNetConfig 和可选的词嵌入作为参数
    def __init__(self, config: ProphetNetConfig, word_embeddings: Optional[nn.Embedding] = None):
        # 调用父类的初始化函数
        super().__init__(config)

        # 从配置中获取并设置以下属性
        self.ngram = config.ngram  # ngram 模型的参数
        self.num_buckets = config.num_buckets  # 用于相对位置编码的桶的数量
        self.relative_max_distance = config.relative_max_distance  # 相对位置编码的最大距离
        self.dropout = config.dropout  # 用于 dropout 的概率
        self.max_target_positions = config.max_position_embeddings  # 解码器的最大位置编码数

        # 初始化词嵌入层
        self.word_embeddings = (
            word_embeddings  # 如果传入了词嵌入，则使用传入的词嵌入
            if word_embeddings is not None
            else nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)  # 否则创建一个新的词嵌入
        )
        # 初始化位置编码层
        self.position_embeddings = ProphetNetPositionalEmbeddings(config)

        # 初始化 ngram 编码层
        self.ngram_embeddings = nn.Embedding(self.ngram, config.hidden_size, None)

        # 初始化多个 ProphetNetDecoderLayer 组成的层列表
        self.layers = nn.ModuleList([ProphetNetDecoderLayer(config) for _ in range(config.num_decoder_layers)])

        # 初始化 embeddings 层的 LayerNorm 层
        self.embeddings_layer_norm = LayerNorm(config.hidden_size)

        # 初始化梯度检查点标志
        self.gradient_checkpointing = False

        # 初始化权重并应用最终处理
        self.post_init()

    # 获取输入词嵌入层
    def get_input_embeddings(self):
        return self.word_embeddings

    # 设置输入词嵌入层
    def set_input_embeddings(self, value):
        self.word_embeddings = value

    # ProphetNetDecoder 的前向传播函数
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
    # 计算缓冲的相对位置桶
    def compute_buffered_relative_buckets(self, position_ids):
        # 获取批次大小和序列长度
        batch_size, sequence_length = position_ids.shape

        # 创建位置 ID 张量，从 1 到 max_target_positions，并复制 1 次
        position_ids = torch.arange(1, self.max_target_positions).to(position_ids.device).repeat(1, 1)
        # 计算所有流的相对位置桶
        main_relative_buckets, predict_relative_buckets = compute_all_stream_relative_buckets(
            self.num_buckets, self.relative_max_distance, position_ids
        )

        # 缓冲相对位置桶
        # 选取 main_relative_buckets 的前 sequence_length 行和列，并复制 batch_size 次
        main_relative_buckets = main_relative_buckets[:, :sequence_length, :sequence_length].repeat(batch_size, 1, 1)
        # 拼接 predict_relative_buckets 的前 sequence_length 行和 max_target_positions 到 max_target_positions + sequence_length 列，并复制 batch_size 次
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
            device=hidden_states.dtype,
        )
        causal_mask = torch.triu(causal_mask, 1)

        # 扩展因果掩码到批次大小和注意力头数
        extended_causal_mask = causal_mask[:seq_length, :seq_length][None, None, :, :].expand(
            (batch_size, self.config.num_decoder_attention_heads) + causal_mask.shape
        )

        # 添加常规注意力掩码
        if attention_mask is not None:
            # 计算扩展注意力掩码
            extended_attention_mask = (1.0 - attention_mask[:, None, None, :]) * torch.finfo(self.dtype).min
            # 将扩展注意力掩码与扩展因果掩码相加
            extended_attention_mask = extended_causal_mask + extended_attention_mask
        else:
            extended_attention_mask = extended_causal_mask
        return extended_attention_mask.to(hidden_states.dtype)
    # 准备预测时使用的注意力掩码
    def prepare_predict_attention_mask(self, hidden_states, attention_mask):
        # 获取批次大小和序列长度
        batch_size, seq_length = hidden_states.shape[:2]
    
        # 获取因果掩码
        predict_causal_mask = ngram_attention_bias(
            self.max_target_positions, self.ngram, hidden_states.device, hidden_states.dtype
        )
        # 拼接掩码，形成最终的因果掩码
        predict_causal_mask = torch.cat(
            [
                predict_causal_mask[:, :seq_length, :seq_length],  # 前部分掩码
                predict_causal_mask[
                    :, :seq_length, self.max_target_positions : self.max_target_positions + seq_length
                ],  # 追加到目标位置的掩码
            ],
            dim=-1,  # 在最后一维拼接
        )
    
        # 扩展因果掩码以适应批次和注意力头的形状
        extended_predict_causal_mask = predict_causal_mask[None, None, :, :, :].expand(
            (batch_size, self.config.num_decoder_attention_heads) + predict_causal_mask.shape
        )
    
        # 加入常规的注意力掩码
        if attention_mask is not None:
            # 根据给定的注意力掩码创建扩展掩码，并与极小值相乘（未激活位置）
            extended_attention_mask = (1.0 - attention_mask[:, None, None, None, :]) * torch.finfo(self.dtype).min
            # 扩展掩码以适应批次和注意力头的形状
            extended_attention_mask = extended_attention_mask.expand(
                (batch_size, self.config.num_decoder_attention_heads, self.ngram, seq_length, seq_length)
            )
            # 预测流的注意力掩码应始终为0
            extended_attention_mask = torch.cat(
                [extended_attention_mask, torch.zeros_like(extended_attention_mask)], dim=-1
            )
            # 合并扩展的因果掩码和常规掩码
            extended_predict_attention_mask = extended_predict_causal_mask + extended_attention_mask
        else:
            # 如果没有注意力掩码，使用扩展的因果掩码
            extended_predict_attention_mask = extended_predict_causal_mask
    
        # 返回转换为与隐藏状态相同数据类型的最终注意力掩码
        return extended_predict_attention_mask.to(hidden_states.dtype)
#导入相应依赖库
@add_start_docstrings(
    "The bare ProphetNet Model outputting raw hidden-states without any specific head on top.",
    PROPHETNET_START_DOCSTRING,
)

#定义ProphetNetModel类，集成ProphetNetPreTrainedModel类
class ProphetNetModel(ProphetNetPreTrainedModel):

    # 设定encoder和decoder中共享的权重
    _tied_weights_keys = ["encoder.word_embeddings.weight", "decoder.word_embeddings.weight"]

    #初始化ProphetNetModel类，获取相应的配置config
    def __init__(self, config: ProphetNetConfig):
        super().__init__(config)
        
        #定义词嵌入层，使用config中的vocab_size和hidden_size，并设定padding_id为config中的pad_token_id
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_encoder_decoder = False
        encoder_config.use_cache = False
        
        #定义encoder层，使用ProphetNetEncoder类，并传入encoder_config和self.word_embeddings
        self.encoder = ProphetNetEncoder(encoder_config, self.word_embeddings)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        
        #定义decoder层，使用ProphetNetDecoder类，并传入decoder_config和self.word_embeddings
        self.decoder = ProphetNetDecoder(decoder_config, self.word_embeddings)

        # 根据权重初始化并应用最终处理
        self.post_init()

    # 获取输入的词嵌入
    def get_input_embeddings(self):
        return self.word_embeddings

    # 设定输入的词嵌入
    def set_input_embeddings(self, value):
        self.word_embeddings = value
               
        # 绑定encoder中的词嵌入层和self.word_embeddings
        self.encoder.word_embeddings = self.word_embeddings
        
        # 绑定decoder中的词嵌入层和self.word_embeddings
        self.decoder.word_embeddings = self.word_embeddings

    # 绑定权重
    def _tie_weights(self):
        if self.config.tie_word_embeddings:
            
            # 在encoder中绑定词嵌入层和self.word_embeddings
            self._tie_or_clone_weights(self.encoder.word_embeddings, self.word_embeddings)
            
            # 在decoder中绑定词嵌入层和self.word_embeddings
            self._tie_or_clone_weights(self.decoder.word_embeddings, self.word_embeddings)

    # 获取encoder层实例
    def get_encoder(self):
        return self.encoder

    # 获取decoder层实例
    def get_decoder(self):
        return self.decoder

    #前向传播方法
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

#定义ProphetNetForConditionalGeneration类，继承ProphetNetPreTrainedModel类
@add_start_docstrings(
    "The ProphetNet Model with a language modeling head. Can be used for sequence generation tasks.",
    PROPHETNET_START_DOCSTRING,
)
class ProphetNetForConditionalGeneration(ProphetNetPreTrainedModel):
    # _tied_weights_keys 是用于绑定权重的关键词列表
    _tied_weights_keys = ["encoder.word_embeddings.weight", "decoder.word_embeddings.weight", "lm_head.weight"]

    # 初始化模型，设置参数并进行后续处理
    def __init__(self, config: ProphetNetConfig):
        # 调用父类的初始化方法
        super().__init__(config)
        # 创建 ProphetNetModel 模型对象
        self.prophetnet = ProphetNetModel(config)
        # 设置填充标记的索引
        self.padding_idx = config.pad_token_id
        # 禁用 ngram 损失
        self.disable_ngram_loss = config.disable_ngram_loss

        # 初始化一个线性层，用于预测
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # 初始化权重并进行后处理
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

    # 前向传播方法，接收一系列输入参数
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
    # 计算损失函数
    def _compute_loss(self, logits, labels, ignore_index=-100):
        # 创建一个 Tensor，大小为(self.config.ngram, labels.size(0), labels.size(1))，用 ignore_index 填充
        expend_targets = labels.new_zeros(self.config.ngram, labels.size(0), labels.size(1)).fill_(ignore_index)
    
        # 遍历 ngram 的范围，如果 i > 0 且 disable_ngram_loss 为 True，则停止循环
        for i in range(self.config.ngram):
            if i > 0 and self.disable_ngram_loss:
                break
            # 将 labels 复制到 expend_targets 的对应位置
            expend_targets[i, :, :] = labels
    
        # 将 logits 的维度顺序调整为(1, 0, 2)
        logits = logits.transpose(0, 1).contiguous()
        # 计算 log_softmax 值
        lprobs = nn.functional.log_softmax(
            logits.view(-1, logits.size(-1)),
            dim=-1,
            dtype=torch.float32,
        )
    
        # 计算 NLL 损失
        loss = nn.functional.nll_loss(lprobs, expend_targets.view(-1), reduction="mean")
    
        # 如果 eps 大于 0，则计算平滑损失并加权
        if self.config.eps > 0.0:
            smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
            non_masked_tokens = expend_targets.ne(ignore_index).view(-1)
            smooth_loss = smooth_loss[non_masked_tokens]
            smooth_loss = smooth_loss.mean()
    
            eps_i = self.config.eps / lprobs.size(-1)
            loss = (1.0 - self.config.eps) * loss + eps_i * smooth_loss
    
        return loss
    
    # 准备用于生成的输入数据
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
        # 确保提供了 encoder_outputs
        assert encoder_outputs is not None, "`encoder_outputs` have to be passed for generation."
    
        # 如果存在 past_key_values，则只保留 decoder_input_ids 的最后一个token
        if past_key_values:
            decoder_input_ids = decoder_input_ids[:, -1:]
    
        # 返回一个字典，包含生成所需的所有输入数据
        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past_key_values,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
        }
    
    # 从标签中准备解码器输入ID
    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return self._shift_right(labels)
    
    # 重新排序缓存
    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            # cached cross_attention states don't have to be reordered -> they are always the same
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past[:2])
                + layer_past[2:],
            )
        return reordered_past
    
    # 获取编码器
    def get_encoder(self):
        return self.prophetnet.encoder
    
    # 获取解码器
    def get_decoder(self):
        return self.prophetnet.decoder
# 这是 ProphetNetForCausalLM 类的定义，它是 ProphetNetModel 的一个子类
@add_start_docstrings(
    "The standalone decoder part of the ProphetNetModel with a lm head on top. The model can be used for causal"
    " language modeling.",
    PROPHETNET_START_DOCSTRING,
)
class ProphetNetForCausalLM(ProphetNetPreTrainedModel):
    # 定义需要绑定的权重的键名
    _tied_weights_keys = [
        "prophetnet.word_embeddings.weight",
        "prophetnet.decoder.word_embeddings.weight",
        "lm_head.weight",
    ]

    def __init__(self, config: ProphetNetConfig):
        # 深拷贝 config 并修改部分参数
        config = copy.deepcopy(config)
        config.is_decoder = True
        config.is_encoder_decoder = False
        # 调用父类的初始化方法
        super().__init__(config)
        # 创建 ProphetNetDecoderWrapper 对象
        self.prophetnet = ProphetNetDecoderWrapper(config)

        # 设置填充 token ID 和 n-gram 损失禁用标志
        self.padding_idx = config.pad_token_id
        self.disable_ngram_loss = config.disable_ngram_loss

        # 创建输出层（语言模型头）
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # 初始化权重并应用最终处理
        self.post_init()

    # 获取输入词嵌入
    def get_input_embeddings(self):
        return self.prophetnet.decoder.word_embeddings

    # 设置输入词嵌入
    def set_input_embeddings(self, value):
        self.prophetnet.decoder.word_embeddings = value

    # 获取输出层（语言模型头）
    def get_output_embeddings(self):
        return self.lm_head

    # 设置输出层（语言模型头）
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    # 绑定词嵌入和输出层（语言模型头）的权重
    def _tie_weights(self):
        if self.config.tie_word_embeddings:
            self._tie_or_clone_weights(self.prophetnet.decoder.word_embeddings, self.lm_head)

    # 设置解码器
    def set_decoder(self, decoder):
        self.prophetnet.decoder = decoder

    # 获取解码器
    def get_decoder(self):
        return self.prophetnet.decoder

    # 定义前向传播方法
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
    # 计算损失函数，接受logits（预测值）、标签、ignore_index（默认值为-100）
    def _compute_loss(self, logits, labels, ignore_index=-100):
        # 创建与labels相同shape的tensor，并用ignore_index填充
        expend_targets = labels.new_zeros(self.config.ngram, labels.size(0), labels.size(1)).fill_(ignore_index)

        # 遍历ngram，将labels赋值给expend_targets
        for i in range(self.config.ngram):
            if i > 0 and self.disable_ngram_loss:
                break
            expend_targets[i, :, :] = labels

        # 转置logits，并确保内存连续性
        logits = logits.transpose(0, 1).contiguous()
        # 对logits进行log_softmax操作
        lprobs = nn.functional.log_softmax(
            logits.view(-1, logits.size(-1)),
            dim=-1,
            dtype=torch.float32,
        )

        # 计算负对数似然损失
        loss = nn.functional.nll_loss(lprobs, expend_targets.view(-1), reduction="mean")

        # 如果配置中的eps大于0
        if self.config.eps > 0.0:
            # 计算平滑损失
            smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
            non_masked_tokens = expend_targets.ne(ignore_index).view(-1)
            smooth_loss = smooth_loss[non_masked_tokens]
            smooth_loss = smooth_loss.mean()

            # 计算平滑损失的eps值
            eps_i = self.config.eps / lprobs.size(-1)
            # 重新计算损失
            loss = (1.0 - self.config.eps) * loss + eps_i * smooth_loss

        # 返回损失值
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
        # 如果注意力掩码为空，创建形状相同的全1张量作为注意力掩码
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_ids.shape)

        # 如果过去的键值对存在，只保留最后一个位置的input_ids
        if past_key_values:
            input_ids = input_ids[:, -1:]
        # 返回准备好的输入，用于生成文本
        return {
            "input_ids": input_ids,  # encoder_outputs is defined. input_ids not needed
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
        }

    @staticmethod
    # 重新排列缓存，用于beam search过程
    # 从transformers.models.bart.modeling_bart.BartForCausalLM._reorder_cache复制
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            # 用beam_idx重新排列过去的���态
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past
class ProphetNetDecoderWrapper(ProphetNetPreTrainedModel):
    """
    This is a wrapper class, so that [`ProphetNetForCausalLM`] can correctly be loaded from pretrained prophetnet
    classes.
    """

    def __init__(self, config: ProphetNetConfig):
        # 调用父类的初始化方法
        super().__init__(config)

        # 初始化词嵌入层，使用配置中的词汇表大小和隐藏大小，设置填充标记的索引
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        # 创建解码器实例
        self.decoder = ProphetNetDecoder(config, word_embeddings=self.word_embeddings)

        # 初始化权重并应用最终处理
        self.post_init()

    def _tie_weights(self):
        # 将词嵌入层的权重与解码器的输入嵌入层权重进行绑定或克隆
        self._tie_or_clone_weights(self.word_embeddings, self.decoder.get_input_embeddings())

    def forward(self, *args, **kwargs):
        # 前向传播方法，调用解码器的前向传播方法
        return self.decoder(*args, **kwargs)
```