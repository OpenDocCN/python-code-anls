# `.\transformers\models\longformer\modeling_longformer.py`

```
# 设定脚本编码格式为 UTF-8
# 版权声明，分别标注 AI 研究所团队和 HuggingFace 公司团队的版权
# 根据 Apache 2.0 版本授权许可，禁止未授权使用此文件
# 可以在下方链接获取许可证的副本：http://www.apache.org/licenses/LICENSE-2.0
# 除非被适用法律要求或经书面同意，否则不能使用本文件
# 分发的软件将基于“按原样提供”的基础分发，没有任何担保或条件，无论是明示的还是暗示的
# 关于许可下的特定语言可以查看特定语言所控制的权限以及许可的具体限制，请查看许可证
"""PyTorch Longformer model."""

# 导入所需的包和模块
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

# 从 Transformers 中导入所需要的激活函数和辅助函数
from ...activations import ACT2FN, gelu
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
# 从 Longformer 的配置文件中导入 LongformerConfig 类
from .configuration_longformer import LongformerConfig

# 初始化 logger 对象
logger = logging.get_logger(__name__)

# 以下为用于文档说明的一些常量和列表
_CHECKPOINT_FOR_DOC = "allenai/longformer-base-4096"
_CONFIG_FOR_DOC = "LongformerConfig"

# 预训练模型的归档列表
LONGFORMER_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "allenai/longformer-base-4096",
    "allenai/longformer-large-4096",
    "allenai/longformer-large-4096-finetuned-triviaqa",
    "allenai/longformer-base-4096-extra.pos.embd.only",
    "allenai/longformer-large-4096-extra.pos.embd.only",
    # 可在链接 https://huggingface.co/models?filter=longformer 查看所有 Longformer 模型
]

# 定义一个数据类 LongformerBaseModelOutput，用于作为 Longformer 的输出基类，包括隐藏状态、本地和全局的注意力
@dataclass
class LongformerBaseModelOutput(ModelOutput):
    """
    Base class for Longformer's outputs, with potential hidden states, local and global attentions.
    # 接受参数last_hidden_state，类型为torch.FloatTensor，shape为(batch_size, sequence_length, hidden_size)，代表模型最后一层的隐藏状态序列
    last_hidden_state: torch.FloatTensor
    # 接受参数hidden_states，默认为None，类型为tuple(torch.FloatTensor)，当传入output_hidden_states=True或者config.output_hidden_states=True时会返回此参数
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # 初始化一个可选的 torch.FloatTensor 元组变量 attentions，用于存储注意力机制的输出
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    # 初始化一个可选的 torch.FloatTensor 元组变量 global_attentions，用于存储全局注意力机制的输出
    global_attentions: Optional[Tuple[torch.FloatTensor]] = None
# 使用 dataclass 装饰器定义 LongformerBaseModelOutputWithPooling 类，该类用于存储 Longformer 模型输出以及最后一层隐藏状态的池化结果
class LongformerBaseModelOutputWithPooling(ModelOutput):
    """
    Base class for Longformer's outputs that also contains a pooling of the last hidden states.

    """

    # 存储最后一层的隐藏状态
    last_hidden_state: torch.FloatTensor
    # 存储池化层的输出
    pooler_output: torch.FloatTensor = None
    # 存储所有隐藏状态的元组
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # 存储注意力分布的元组
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    # 存储全局注意力的元组
    global_attentions: Optional[Tuple[torch.FloatTensor]] = None


# 使用 dataclass 装饰器定义 LongformerMaskedLMOutput 类，该类用于存储掩码语言模型的输出
@dataclass
class LongformerMaskedLMOutput(ModelOutput):
    """
    Base class for masked language models outputs.

    """

    # 存储损失值的可选项
    loss: Optional[torch.FloatTensor] = None
    # 存储逻辑回归输出的张量
    logits: torch.FloatTensor = None
    # 存储所有隐藏状态的元组
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # 存储注意力分布的元组
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    # 存储全局注意力的元组
    global_attentions: Optional[Tuple[torch.FloatTensor]] = None


# 使用 dataclass 装饰器定义 LongformerQuestionAnsweringModelOutput 类，该类用于存储问答 Longformer 模型的输出
@dataclass
class LongformerQuestionAnsweringModelOutput(ModelOutput):
    """
    Base class for outputs of question answering Longformer models.

    """

    # 存储损失值的可选项
    loss: Optional[torch.FloatTensor] = None
    # 存储起始位置的逻辑回归输出的张量
    start_logits: torch.FloatTensor = None
    # 存储结束位置的逻辑回归输出的张量
    end_logits: torch.FloatTensor = None
    # 存储所有隐藏状态的元组
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # 存储注意力分布的元组
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    # 存储全局注意力的元组
    global_attentions: Optional[Tuple[torch.FloatTensor]] = None


# 使用 dataclass 装饰器定义 LongformerSequenceClassifierOutput 类，该类用于存储句子分类模型的输出
@dataclass
class LongformerSequenceClassifierOutput(ModelOutput):
    """
    Base class for outputs of sentence classification models.
    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            分类（或回归，如果`config.num_labels==1`）损失。
        logits (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`):
            分类（或回归，如果`config.num_labels==1`）得分（SoftMax 之前）。
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            元组的 `torch.FloatTensor`（一个用于嵌入的输出 + 一个用于每个层的输出），形状为 `(batch_size, sequence_length, hidden_size)`。

            模型在每个层的输出以及初始嵌入输出的隐藏状态。
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            元组的 `torch.FloatTensor`（每个层一个）的形状为 `(batch_size, num_heads, sequence_length, x + attention_window + 1)`，其中 `x` 是具有全局注意力掩码的令牌数量。

            在注意力 softmax 之后的局部注意力权重，用于计算自注意力头中的加权平均值。这些是从序列中的每个令牌到具有全局注意力的每个令牌（前 `x` 个值）以及到注意力窗口中的每个令牌（剩余 `attention_window + 1` 个值）的注意力权重。注意，前 `x` 个值是指文本中具有固定位置的令牌，但剩余的 `attention_window + 1` 个值是指具有相对位置的令牌：令牌到自身的注意力权重位于索引 `x + attention_window / 2` 处，前 `attention_window / 2`（后 `attention_window / 2`）个值是令牌到前 `attention_window / 2`（后 `attention_window / 2`）个令牌的注意力权重。如果注意力窗口包含具有全局注意力的令牌，则相应索引处的注意力权重设置为 0；该值应从前 `x` 个注意力权重中访问。如果一个令牌具有全局注意力，则 `attentions` 中对所有其他令牌的注意力权重设置为 0，值应从 `global_attentions` 中访问。
        global_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            元组的 `torch.FloatTensor`（每个层一个）的形状为 `(batch_size, num_heads, sequence_length, x)`，其中 `x` 是具有全局注意力掩码的令牌数量。

            在注意力 softmax 之后的全局注意力权重，用于计算自注意力头中的加权平均值。这些是具有全局注意力的每个令牌到序列中的每个令牌的注意力权重。
    # 定义一个可选的浮点类型变量 loss，初始值为 None
    loss: Optional[torch.FloatTensor] = None
    # 定义一个浮点类型变量 logits，初始值为 None
    logits: torch.FloatTensor = None
    # 定义一个可选的元组类型变量 hidden_states，元组内的元素为浮点类型，初始值为 None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # 定义一个可选的元组类型变量 attentions，元组内的元素为浮点类型，初始值为 None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    # 定义一个可选的元组类型变量 global_attentions，元组内的元素为浮点类型，初始值为 None
    global_attentions: Optional[Tuple[torch.FloatTensor]] = None
@dataclass
class LongformerMultipleChoiceModelOutput(ModelOutput):
    """
    Longformer多项选择模型输出的基类。

    """

    loss: Optional[torch.FloatTensor] = None  # 损失值，可选的浮点张量，默认为None
    logits: torch.FloatTensor = None  # 预测的逻辑回归值，张量
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None  # 隐藏状态，可选的张量元组，默认为None
    attentions: Optional[Tuple[torch.FloatTensor]] = None  # 注意力张量，可选的张量元组，默认为None
    global_attentions: Optional[Tuple[torch.FloatTensor]] = None  # 全局注意力张量，可选的张量元组，默认为None


@dataclass
class LongformerTokenClassifierOutput(ModelOutput):
    """
    分词分类模型输出的基类。

    """
    # 定义函数参数和返回值的说明
    Args:
        # 分类损失，是一个形状为`(1,)`的`torch.FloatTensor`张量，当`labels`提供时返回
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) :
            Classification loss.
        
        # 分类得分（SoftMax之前）的张量，具有形状为`(batch_size, sequence_length, config.num_labels)`
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.num_labels)`):
            Classification scores (before SoftMax).

        # 模型隐藏状态的元组，包含每一层输出和初始嵌入输出
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.
        
        # 本地注意力权重的元组，用于计算自注意力头中的加权平均值
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, x +
            attention_window + 1)`, where `x` is the number of tokens with global attention mask.

            # 在注意力 softmax 之后的本地注意力权重，用于计算自注意力头中的加权平均值
            # 这些是从序列中的每个标记到具有全局注意力的每个标记的注意力权重
            # 如果注意力窗口包含具有全局注意力的标记，则相应索引处的注意力权重设置为0；该值应从前`x`个注意力权重中获得
            # 如果一个标记具有全局注意力，那么`attentions`中对所有其他标记的关注权重被设置为0，值应从`global_attentions`中获取
            # 具体的计算规则和细节解释
            Local attentions weights after the attention softmax, used to compute the weighted average in the
            self-attention heads. Those are the attention weights from every token in the sequence to every token with
            global attention (first `x` values) and to every token in the attention window (remaining `attention_window
            + 1` values). Note that the first `x` values refer to tokens with fixed positions in the text, but the
            remaining `attention_window + 1` values refer to tokens with relative positions: the attention weight of a
            token to itself is located at index `x + attention_window / 2` and the `attention_window / 2` preceding
            (succeeding) values are the attention weights to the `attention_window / 2` preceding (succeeding) tokens.
            If the attention window contains a token with global attention, the attention weight at the corresponding
            index is set to 0; the value should be accessed from the first `x` attention weights. If a token has global
            attention, the attention weights to all other tokens in `attentions` is set to 0, the values should be
            accessed from `global_attentions`.
        
        # 全局注意力权重的元组，用于计算自注意力头中的加权平均值
        global_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, x)`,
            where `x` is the number of tokens with global attention mask.

            # 在注意力 softmax 之后的全局注意力权重，用于自注意力头中的加权平均值的计算
            # 这些是从具有全局注意力的每个标记到序列中的每个标记的注意力权重
    """

    # 定义变量 loss 为可选的`torch.FloatTensor`类型，默认值为 None
    loss: Optional[torch.FloatTensor] = None
    # 定义一个类型为torch.FloatTensor的logits变量，并初始化为None
    logits: torch.FloatTensor = None
    # 定义一个可选的类型为Tuple[torch.FloatTensor]的hidden_states变量，并初始化为None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # 定义一个可选的类型为Tuple[torch.FloatTensor]的attentions变量，并初始化为None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    # 定义一个可选的类型为Tuple[torch.FloatTensor]的global_attentions变量，并初始化为None
    global_attentions: Optional[Tuple[torch.FloatTensor]] = None
def _get_question_end_index(input_ids, sep_token_id):
    """
    Computes the index of the first occurrence of `sep_token_id`.
    """

    # Find the indices of separator tokens in the input
    sep_token_indices = (input_ids == sep_token_id).nonzero()
    batch_size = input_ids.shape[0]

    # Ensure the shape of sep_token_indices is appropriate
    assert sep_token_indices.shape[1] == 2, "`input_ids` should have two dimensions"
    # Ensure there are exactly three separator tokens in every sample for question answering
    assert sep_token_indices.shape[0] == 3 * batch_size, (
        f"There should be exactly three separator tokens: {sep_token_id} in every sample for questions answering. You"
        " might also consider to set `global_attention_mask` manually in the forward function to avoid this error."
    )
    # Extract the index of the first occurrence of the separator token for each sample
    return sep_token_indices.view(batch_size, 3, 2)[:, 0, 1]


def _compute_global_attention_mask(input_ids, sep_token_id, before_sep_token=True):
    """
    Computes global attention mask by putting attention on all tokens before `sep_token_id` if `before_sep_token is
    True` else after `sep_token_id`.
    """
    # Get the index of the first occurrence of the separator token for each sample
    question_end_index = _get_question_end_index(input_ids, sep_token_id)
    question_end_index = question_end_index.unsqueeze(dim=1)  # size: batch_size x 1
    # Create a boolean attention mask with True in locations of global attention
    attention_mask = torch.arange(input_ids.shape[1], device=input_ids.device)
    if before_sep_token is True:
        # Set True for tokens before the separator token
        attention_mask = (attention_mask.expand_as(input_ids) < question_end_index).to(torch.bool)
    else:
        # Set True for tokens after the separator token
        attention_mask = (attention_mask.expand_as(input_ids) > (question_end_index + 1)).to(torch.bool) * (
            attention_mask.expand_as(input_ids) < input_ids.shape[-1]
        ).to(torch.bool)

    return attention_mask


def create_position_ids_from_input_ids(input_ids, padding_idx):
    """
    Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding symbols
    are ignored. This is modified from fairseq's `utils.make_positions`.

    Args:
        x: torch.Tensor x:

    Returns: torch.Tensor
    """
    # Create a mask for non-padding symbols
    mask = input_ids.ne(padding_idx).int()
    # Compute incremental indices for non-padding symbols
    incremental_indices = torch.cumsum(mask, dim=1).type_as(mask) * mask
    # Add padding index to the incremental indices
    return incremental_indices.long() + padding_idx


class LongformerEmbeddings(nn.Module):
    """
    Same as BertEmbeddings with a tiny tweak for positional embeddings indexing.
    """
    def __init__(self, config):
        # 初始化函数，继承父类的初始化方法
        super().__init__()
        # 创建词嵌入矩阵，vocab_size为词汇表大小，hidden_size为隐藏层大小，padding_idx指定填充的词索引
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        # 创建token类型嵌入矩阵，type_vocab_size为类型词汇表大小，hidden_size为隐藏层大小
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # 使用LayerNorm来规范化数据，保持与TensorFlow模型变量名一致，以便能够加载任何TensorFlow检查点文件
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 创建dropout层，按照指定的概率丢弃神经元
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # 设定填充索引为pad_token_id
        self.padding_idx = config.pad_token_id
        # 创建位置嵌入矩阵，max_position_embeddings为最大位置嵌入数量，padding_idx指定填充的位置
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size, padding_idx=self.padding_idx
        )

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None):
        # 如果位置ids为空
        if position_ids is None:
            # 如果输入ids不为空
            if input_ids is not None:
                # 从输入的token ids中创建位置ids。任何填充的token仍然保持填充状态
                position_ids = create_position_ids_from_input_ids(input_ids, self.padding_idx).to(input_ids.device)
            else:
                # 如果输入ids为空，则从inputs_embeds中创建位置ids
                position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds)

        # 如果输入ids不为空
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        # 如果token类型ids为空，则创建全零的token类型ids
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=position_ids.device)

        # 如果inputs_embeds为空，则使用word_embeddings获取embeddings
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        # 获取位置嵌入
        position_embeddings = self.position_embeddings(position_ids)
        # 获取token类型嵌入
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        # 将embeddings组合起来
        embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

    def create_position_ids_from_inputs_embeds(self, inputs_embeds):
        """
        We are provided embeddings directly. We cannot infer which are padded so just generate sequential position ids.

        Args:
            inputs_embeds: torch.Tensor inputs_embeds:

        Returns: torch.Tensor
        """
        # 获取输入形状
        input_shape = inputs_embeds.size()[:-1]
        # 获取序列长度
        sequence_length = input_shape[1]

        # 生成顺序位置ids
        position_ids = torch.arange(
            self.padding_idx + 1, sequence_length + self.padding_idx + 1, dtype=torch.long, device=inputs_embeds.device
        )
        return position_ids.unsqueeze(0).expand(input_shape)
```  
# 定义一个名为LongformerSelfAttention的类，继承自nn.Module
class LongformerSelfAttention(nn.Module):
    # 初始化方法，接收配置参数config和层索引layer_id
    def __init__(self, config, layer_id):
        super().__init__()
        # 如果隐藏大小不能被注意力头数整除，则引发数值错误
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )
        # 设置属性值
        self.num_heads = config.num_attention_heads
        self.head_dim = int(config.hidden_size / config.num_attention_heads)
        self.embed_dim = config.hidden_size

        # 创建查询、键和值的线性层
        self.query = nn.Linear(config.hidden_size, self.embed_dim)
        self.key = nn.Linear(config.hidden_size, self.embed_dim)
        self.value = nn.Linear(config.hidden_size, self.embed_dim)

        # 为具有全局注意力的标记创建不同的投影层
        self.query_global = nn.Linear(config.hidden_size, self.embed_dim)
        self.key_global = nn.Linear(config.hidden_size, self.embed_dim)
        self.value_global = nn.Linear(config.hidden_size, self.embed_dim)

        # 设置注意力概率的丢弃率
        self.dropout = config.attention_probs_dropout_prob

        self.layer_id = layer_id
        attention_window = config.attention_window[self.layer_id]
        # 确保attention_window是偶数
        assert (
            attention_window % 2 == 0
        ), f"`attention_window` for layer {self.layer_id} has to be an even value. Given {attention_window}"
        # 确保attention_window是正数
        assert (
            attention_window > 0
        ), f"`attention_window` for layer {self.layer_id} has to be positive. Given {attention_window}"

        # 计算单侧注意力窗口大小
        self.one_sided_attn_window_size = attention_window // 2

        self.config = config

    # 前向传播方法
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        layer_head_mask=None,
        is_index_masked=None,
        is_index_global_attn=None,
        is_global_attn=None,
        output_attentions=False,
    # 静态方法，用于填充并转置最后两个维度
    @staticmethod
    def _pad_and_transpose_last_two_dims(hidden_states_padded, padding):
        """pads rows and then flips rows and columns"""
        # 对hidden_states_padded进行填充
        hidden_states_padded = nn.functional.pad(
            hidden_states_padded, padding
        )  # padding value is not important because it will be overwritten
        # 转置最后两个维度
        hidden_states_padded = hidden_states_padded.view(
            *hidden_states_padded.size()[:-2], hidden_states_padded.size(-1), hidden_states_padded.size(-2)
        )
        return hidden_states_padded

    @staticmethod
    # 定义一个内部方法，用于填充和对角化分块隐藏状态
    def _pad_and_diagonalize(chunked_hidden_states):
        """
        shift every row 1 step right, converting columns into diagonals.
        将每一行向右移动1步，将列转换为对角线。

        Example:

        ```python
        chunked_hidden_states: [
            0.4983,
            2.6918,
            -0.0071,
            1.0492,
            -1.8348,
            0.7672,
            0.2986,
            0.0285,
            -0.7584,
            0.4206,
            -0.0405,
            0.1599,
            2.0514,
            -1.1600,
            0.5372,
            0.2629,
        ]
        window_overlap = num_rows = 4
        ```

                     (pad & diagonalize) => [ 0.4983, 2.6918, -0.0071, 1.0492, 0.0000, 0.0000, 0.0000
                       0.0000, -1.8348, 0.7672, 0.2986, 0.0285, 0.0000, 0.0000 0.0000, 0.0000, -0.7584, 0.4206,
                       -0.0405, 0.1599, 0.0000 0.0000, 0.0000, 0.0000, 2.0514, -1.1600, 0.5372, 0.2629 ]
        """

        # 获取输入张量的维度信息
        total_num_heads, num_chunks, window_overlap, hidden_dim = chunked_hidden_states.size()
        # 对输入张量进行填充，使得维度扩展到 window_overlap+1
        chunked_hidden_states = nn.functional.pad(
            chunked_hidden_states, (0, window_overlap + 1)
        )  # total_num_heads x num_chunks x window_overlap x (hidden_dim+window_overlap+1). Padding value is not important because it'll be overwritten
        # 对填充后的张量进行维度变换
        chunked_hidden_states = chunked_hidden_states.view(
            total_num_heads, num_chunks, -1
        )  # total_num_heads x num_chunks x window_overlap*window_overlap+window_overlap
        # 截取填充后的张量的一部分维度
        chunked_hidden_states = chunked_hidden_states[
            :, :, :-window_overlap
        ]  # total_num_heads x num_chunks x window_overlap*window_overlap
        # 再次对张量进行维度变换
        chunked_hidden_states = chunked_hidden_states.view(
            total_num_heads, num_chunks, window_overlap, window_overlap + hidden_dim
        )
        # 对填充后的张量进行尺寸裁剪
        chunked_hidden_states = chunked_hidden_states[:, :, :, :-1]
        # 返回结果张量
        return chunked_hidden_states

    @staticmethod
    def _chunk(hidden_states, window_overlap, onnx_export: bool = False):
        """将隐藏状态转换为重叠的块。块大小 = 2w，重叠大小 = w"""
        if not onnx_export:
            # 将隐藏状态划分为大小为2w的非重叠块
            hidden_states = hidden_states.view(
                hidden_states.size(0),
                torch.div(hidden_states.size(1), (window_overlap * 2), rounding_mode="trunc"),
                window_overlap * 2,
                hidden_states.size(2),
            )
            # 使用 `as_trided` 让块重叠，重叠大小为 window_overlap
            chunk_size = list(hidden_states.size())
            chunk_size[1] = chunk_size[1] * 2 - 1

            chunk_stride = list(hidden_states.stride())
            chunk_stride[1] = chunk_stride[1] // 2
            return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)

        # 在导出到 ONNX 时，使用单独的逻辑
        # 因为在 ONNX 导出中不支持 as_strided、unfold 和 2D-张量索引（尚未支持）

        # TODO 用这个替代
        # > return hidden_states.unfold(dimension=1, size=window_overlap * 2, step=window_overlap).transpose(2, 3)
        # 一旦 `unfold` 支持
        # 当 hidden_states.size(1) == window_overlap * 2 时，也可以简单地返回 hidden_states.unsqueeze(1)，但那是控制流

        chunk_size = [
            hidden_states.size(0),
            torch.div(hidden_states.size(1), window_overlap, rounding_mode="trunc") - 1,
            window_overlap * 2,
            hidden_states.size(2),
        ]

        overlapping_chunks = torch.empty(chunk_size, device=hidden_states.device)
        for chunk in range(chunk_size[1]):
            overlapping_chunks[:, chunk, :, :] = hidden_states[
                :, chunk * window_overlap : chunk * window_overlap + 2 * window_overlap, :
            ]
        return overlapping_chunks

    @staticmethod
    def _mask_invalid_locations(input_tensor, affected_seq_len) -> torch.Tensor:
        beginning_mask_2d = input_tensor.new_ones(affected_seq_len, affected_seq_len + 1).tril().flip(dims=[0])
        beginning_mask = beginning_mask_2d[None, :, None, :]
        ending_mask = beginning_mask.flip(dims=(1, 3))
        beginning_input = input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1]
        beginning_mask = beginning_mask.expand(beginning_input.size())
        input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
            beginning_input, -float("inf")
        ).where(beginning_mask.bool(), beginning_input)
        ending_input = input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :]
        ending_mask = ending_mask.expand(ending_input.size())
        input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
            ending_input, -float("inf")
        ).where(ending_mask.bool(), ending_input)
    # 该函数计算注意力得分和价值的矩阵乘积并返回
    def _sliding_chunks_matmul_attn_probs_value(
        self, attn_probs: torch.Tensor, value: torch.Tensor, window_overlap: int
    ):
        # 获取输入张量的各维度大小
        batch_size, seq_len, num_heads, head_dim = value.size()
    
        # 确保序列长度是窗口重叠长度的整数倍
        assert seq_len % (window_overlap * 2) == 0
        # 确保注意力概率和价值张量有相同的前三个维度
        assert attn_probs.size()[:3] == value.size()[:3]
        # 确保注意力概率张量的第四个维度等于 2 * window_overlap + 1
        assert attn_probs.size(3) == 2 * window_overlap + 1
        # 计算块的数量
        chunks_count = torch.div(seq_len, window_overlap, rounding_mode="trunc") - 1
    
        # 将批次大小和头数量合并成一个维度，然后将序列长度分块
        chunked_attn_probs = attn_probs.transpose(1, 2).reshape(
            batch_size * num_heads,
            torch.div(seq_len, window_overlap, rounding_mode="trunc"),
            window_overlap,
            2 * window_overlap + 1,
        )
    
        # 将批次大小和头数量合并成一个维度
        value = value.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
    
        # 在序列开头和结尾添加窗口重叠长度的填充
        padded_value = nn.functional.pad(value, (0, 0, window_overlap, window_overlap), value=-1)
    
        # 将填充后的值分块
        chunked_value_size = (batch_size * num_heads, chunks_count + 1, 3 * window_overlap, head_dim)
        chunked_value_stride = padded_value.stride()
        chunked_value_stride = (
            chunked_value_stride[0],
            window_overlap * chunked_value_stride[1],
            chunked_value_stride[1],
            chunked_value_stride[2],
        )
        chunked_value = padded_value.as_strided(size=chunked_value_size, stride=chunked_value_stride)
    
        # 对chunked_attn_probs进行填充和对角线化
        chunked_attn_probs = self._pad_and_diagonalize(chunked_attn_probs)
    
        # 计算结果张量
        context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
        return context.view(batch_size, num_heads, seq_len, head_dim).transpose(1, 2)
    # 该函数计算全局注意力权重索引，这些索引在整个前向传递过程中都需要使用
    def _get_global_attn_indices(is_index_global_attn):
        # 帮助变量：计算每个样本中全局注意力权重的数量
        num_global_attn_indices = is_index_global_attn.long().sum(dim=1)
        
        # 批量中最大的全局注意力权重数量
        max_num_global_attn_indices = num_global_attn_indices.max()
        
        # 获取全局注意力权重的索引
        is_index_global_attn_nonzero = is_index_global_attn.nonzero(as_tuple=True)
        
        # 帮助变量：判断局部索引是否属于全局注意力权重
        is_local_index_global_attn = torch.arange(
            max_num_global_attn_indices, device=is_index_global_attn.device
        ) < num_global_attn_indices.unsqueeze(dim=-1)
        
        # 获取全局注意力权重的非填充值位置
        is_local_index_global_attn_nonzero = is_local_index_global_attn.nonzero(as_tuple=True)
        
        # 获取全局注意力权重的填充值位置
        is_local_index_no_global_attn_nonzero = (is_local_index_global_attn == 0).nonzero(as_tuple=True)
        
        return (
            max_num_global_attn_indices,
            is_index_global_attn_nonzero,
            is_local_index_global_attn_nonzero,
            is_local_index_no_global_attn_nonzero,
        )
    
    # 该函数将全局注意力权重概率与局部注意力权重概率拼接在一起
    def _concat_with_global_key_attn_probs(
        self,
        key_vectors,
        query_vectors,
        max_num_global_attn_indices,
        is_index_global_attn_nonzero,
        is_local_index_global_attn_nonzero,
        is_local_index_no_global_attn_nonzero,
    ):
        batch_size = key_vectors.shape[0]
        
        # 创建只包含全局注意力权重的 key 向量
        key_vectors_only_global = key_vectors.new_zeros(
            batch_size, max_num_global_attn_indices, self.num_heads, self.head_dim
        )
        key_vectors_only_global[is_local_index_global_attn_nonzero] = key_vectors[is_index_global_attn_nonzero]
        
        # 计算 query 向量与全局 key 向量的注意力权重
        attn_probs_from_global_key = torch.einsum("blhd,bshd->blhs", (query_vectors, key_vectors_only_global))
        
        # 需要进行转置，因为 ONNX 导出只支持连续索引
        attn_probs_from_global_key = attn_probs_from_global_key.transpose(1, 3)
        attn_probs_from_global_key[
            is_local_index_no_global_attn_nonzero[0], is_local_index_no_global_attn_nonzero[1], :, :
        ] = torch.finfo(attn_probs_from_global_key.dtype).min
        attn_probs_from_global_key = attn_probs_from_global_key.transpose(1, 3)
        
        return attn_probs_from_global_key
    
    def _compute_attn_output_with_global_indices(
        self,
        value_vectors,
        attn_probs,
        max_num_global_attn_indices,
        is_index_global_attn_nonzero,
        is_local_index_global_attn_nonzero,
    ):
    # 计算全局注意力输出
    def _compute_global_attn_output(
        self,
        attn_probs,
        value_vectors,
        max_num_global_attn_indices,
        is_local_index_global_attn_nonzero,
        is_index_global_attn_nonzero,
    ):
        # 获取批次大小
        batch_size = attn_probs.shape[0]
    
        # 仅保留全局注意力权重
        attn_probs_only_global = attn_probs.narrow(-1, 0, max_num_global_attn_indices)
        # 获取全局注意力对应的值向量
        value_vectors_only_global = value_vectors.new_zeros(
            batch_size, max_num_global_attn_indices, self.num_heads, self.head_dim
        )
        value_vectors_only_global[is_local_index_global_attn_nonzero] = value_vectors[is_index_global_attn_nonzero]
    
        # 使用 matmul 计算全局注意力输出
        attn_output_only_global = torch.matmul(
            attn_probs_only_global.transpose(1, 2).clone(), value_vectors_only_global.transpose(1, 2).clone()
        ).transpose(1, 2)
    
        # 提取非全局注意力部分的注意力权重
        attn_probs_without_global = attn_probs.narrow(
            -1, max_num_global_attn_indices, attn_probs.size(-1) - max_num_global_attn_indices
        ).contiguous()
    
        # 计算非全局注意力部分的注意力输出
        attn_output_without_global = self._sliding_chunks_matmul_attn_probs_value(
            attn_probs_without_global, value_vectors, self.one_sided_attn_window_size
        )
    
        # 返回全局和非全局注意力输出的和
        return attn_output_only_global + attn_output_without_global
    
    # 从隐藏状态计算全局注意力输出
    def _compute_global_attn_output_from_hidden(
        self,
        hidden_states,
        max_num_global_attn_indices,
        layer_head_mask,
        is_local_index_global_attn_nonzero,
        is_index_global_attn_nonzero,
        is_local_index_no_global_attn_nonzero,
        is_index_masked,
    ):
        # 函数实现省略
        pass
# 从 transformers.models.bert.modeling_bert.BertSelfOutput 中复制得到 LongformerSelfOutput 类
class LongformerSelfOutput(nn.Module):
    # 初始化函数，接收配置参数
    def __init__(self, config):
        super().__init__()
        # 创建一个全链接层，输入尺寸为配置中的隐藏层尺寸，输出尺寸为配置中的隐藏层尺寸
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 创建一个 LayerNorm 层，输入尺寸为配置中的隐藏层尺寸，使用配置中的 layer_norm_eps 作为 epsilon 值
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 创建一个 Dropout 层，使用配置中的 hidden_dropout_prob 作为 dropout 概率值
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    # 前向传播函数，接收隐状态张量和输入张量，并返回张量
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 使用全链接层处理隐状态
        hidden_states = self.dense(hidden_states)
        # 使用 Dropout 处理隐状态
        hidden_states = self.dropout(hidden_states)
        # 使用 LayerNorm 处理隐状态和输入张量的和
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        # 返回处理后的张量
        return hidden_states


# 从 transformers.models.bert.modeling_bert.BertIntermediate 中复制得到 LongformerIntermediate 类
class LongformerIntermediate(nn.Module):
    # 初始化函数，接收配置参数
    def __init__(self, config):
        super().__init__()
        # 创建一个全链接层，输入尺寸为配置中的隐藏层尺寸，输出尺寸为配置中的 intermediate_size
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        # 如果配置中的 hidden_act 是字符串类型，则使用对应的激活函数；否则直接使用配置中的 hidden_act 作为激活函数
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act
    # 前向传播函数，接受隐藏状态张量作为输入，返回处理后的隐藏状态张量
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 使用全连接层处理隐藏状态张量
        hidden_states = self.dense(hidden_states)
        # 使用激活函数处理全连接层的输出
        hidden_states = self.intermediate_act_fn(hidden_states)
        # 返回处理后的隐藏状态张量
        return hidden_states
# 从transformers.models.bert.modeling_bert.BertOutput中复制得到的LongformerOutput类
class LongformerOutput(nn.Module):
    # 初始化函数
    def __init__(self, config):
        super().__init__()
        # 创建一个全连接层，将输入的维度从config.intermediate_size转换为config.hidden_size
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        # 创建一个LayerNorm层，用于对隐藏状态进行归一化
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 创建一个dropout层，用于随机丢弃隐藏状态中的一部分数据
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    # 前向传播函数
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 将隐藏状态输入到全连接层中
        hidden_states = self.dense(hidden_states)
        # 对全连接层的输出进行随机丢弃
        hidden_states = self.dropout(hidden_states)
        # 将丢弃后的隐藏状态与输入张量进行相加，并进行归一化
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        # 返回归一化后的隐藏状态
        return hidden_states

# LongformerLayer类，继承自nn.Module类
class LongformerLayer(nn.Module):
    # 初始化函数
    def __init__(self, config, layer_id=0):
        super().__init__()
        # 创建LongformerAttention对象
        self.attention = LongformerAttention(config, layer_id)
        # 创建LongformerIntermediate对象
        self.intermediate = LongformerIntermediate(config)
        # 创建LongformerOutput对象
        self.output = LongformerOutput(config)
        # 分块前向传播过程使用的块大小
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        # 序列长度维度
        self.seq_len_dim = 1

    # 前向传播函数
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        layer_head_mask=None,
        is_index_masked=None,
        is_index_global_attn=None,
        is_global_attn=None,
        output_attentions=False,
    ):
        # 使用LongformerAttention对象进行注意力计算
        self_attn_outputs = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            is_index_masked=is_index_masked,
            is_index_global_attn=is_index_global_attn,
            is_global_attn=is_global_attn,
            output_attentions=output_attentions,
        )
        # 获取注意力计算的输出
        attn_output = self_attn_outputs[0]
        # 获取其它输出
        outputs = self_attn_outputs[1:]

        # 对注意力计算的输出进行分块
        layer_output = apply_chunking_to_forward(
            self.ff_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attn_output
        )
        # 将分块后的输出与其它输出组成一个元组返回
        outputs = (layer_output,) + outputs
        return outputs

    # 分块的前向传播函数
    def ff_chunk(self, attn_output):
        # 使用LongformerIntermediate对象对注意力计算的输出进行前向传播
        intermediate_output = self.intermediate(attn_output)
        # 使用LongformerOutput对象对输出的中间结果进行前向传播
        layer_output = self.output(intermediate_output, attn_output)
        # 返回输出结果
        return layer_output

# LongformerEncoder类，继承自nn.Module类
class LongformerEncoder(nn.Module):
    # 初始化函数
    def __init__(self, config):
        super().__init__()
        self.config = config
        # 创建长度为config.num_hidden_layers的nn.ModuleList，每个元素都是一个LongformerLayer对象
        self.layer = nn.ModuleList([LongformerLayer(config, layer_id=i) for i in range(config.num_hidden_layers)])
        # 梯度检查点在这里是否激活
        self.gradient_checkpointing = False

    # 前向传播函数
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        padding_len=0,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        # 循环遍历每个LongformerLayer对象，并进行前向传播
        # 将前一个LongformerLayer对象的输出作为当前对象的输入
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask, head_mask, padding_len, output_attentions)
        # 返回结果
        return hidden_states

# 从transformers.models.bert.modeling_bert.BertPooler中复制得到的LongformerPooler类
class LongformerPooler(nn.Module):
    # 初始化函数
    def __init__(self, config):
        super().__init__()
        # 创建一个全连接层，将输入的维度从config.hidden_size转换为config.hidden_size
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 创建一个Tanh层，对输入进行激活函数处理
        self.activation = nn.Tanh()
    # 前向传播函数，接受隐藏状态张量作为输入，并返回处理后的张量
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 我们通过简单地选择对应于第一个标记的隐藏状态来“汇聚”模型。
        # 选择隐藏状态张量的所有行（样本）的第一个元素作为汇聚的隐藏状态
        first_token_tensor = hidden_states[:, 0]
        # 将第一个标记的隐藏状态传递给全连接层，用于降维
        pooled_output = self.dense(first_token_tensor)
        # 应用激活函数
        pooled_output = self.activation(pooled_output)
        # 返回汇聚后的输出张量
        return pooled_output
# 从 transformers.models.roberta.modeling_roberta.RobertaLMHead 复制并修改为 LongformerLMHead 类
class LongformerLMHead(nn.Module):
    """Longformer Head for masked language modeling."""

    def __init__(self, config):
        # 初始化函数，接受配置参数
        super().__init__()
        # 创建线性层，用于将输入特征映射到相同大小的特征空间
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 创建 LayerNorm 层，用于归一化输入特征
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # 创建线性层，用于将特征映射回词汇表大小的空间
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size)
        # 创建偏置参数，并将其作为 decoder 的偏置
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.decoder.bias = self.bias

    def forward(self, features, **kwargs):
        # 将输入特征通过 dense 层
        x = self.dense(features)
        # 应用 GELU 激活函数
        x = gelu(x)
        # 应用 LayerNorm 归一化
        x = self.layer_norm(x)

        # 通过 decoder 层将特征映射回词汇表大小的空间
        x = self.decoder(x)

        # 返回结果
        return x

    def _tie_weights(self):
        # 如果 decoder 的偏置所在设备类型为 "meta"，则将其与 bias 参数绑定，否则将 bias 与 decoder 的偏置绑定
        if self.decoder.bias.device.type == "meta":
            self.decoder.bias = self.bias
        else:
            self.bias = self.decoder.bias


class LongformerPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 设置 LongformerPreTrainedModel 的配置类为 LongformerConfig
    config_class = LongformerConfig
    # 设置基础模型前缀为 "longformer"
    base_model_prefix = "longformer"
    # 支持梯度检查点
    supports_gradient_checkpointing = True
    # 不分割的模块列表
    _no_split_modules = ["LongformerSelfAttention"]

    def _init_weights(self, module):
        """Initialize the weights"""
        # 初始化权重函数
        if isinstance(module, nn.Linear):
            # 如果是线性层，使用正态分布初始化权重，偏置初始化为零
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            # 如果是嵌入层，使用正态分布初始化权重，padding_idx 对应的权重初始化为零
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            # 如果是 LayerNorm 层，偏置初始化为零，权重初始化为 1
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


LONGFORMER_START_DOCSTRING = r"""

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.


"""
    Parameters:
        config ([`LongformerConfig`]): 模型配置类，包含模型的所有参数。使用配置文件初始化不会加载与模型相关的权重，只会加载配置。查看 [`~PreTrainedModel.from_pretrained`] 方法以加载模型权重。
"""
# 定义了长形Transformer模型，继承自LongformerPreTrainedModel
@add_start_docstrings(
    "The bare Longformer Model outputting raw hidden-states without any specific head on top.",
    LONGFORMER_START_DOCSTRING,
)
class LongformerModel(LongformerPreTrainedModel):
    """
    This class copied code from [`RobertaModel`] and overwrote standard self-attention with longformer self-attention
    to provide the ability to process long sequences following the self-attention approach described in [Longformer:
    the Long-Document Transformer](https://arxiv.org/abs/2004.05150) by Iz Beltagy, Matthew E. Peters, and Arman Cohan.
    Longformer self-attention combines a local (sliding window) and global attention to extend to long documents
    without the O(n^2) increase in memory and compute.

    The self-attention module `LongformerSelfAttention` implemented here supports the combination of local and global
    attention but it lacks support for autoregressive attention and dilated attention. Autoregressive and dilated
    attention are more relevant for autoregressive language modeling than finetuning on downstream tasks. Future
    release will add support for autoregressive attention, but the support for dilated attention requires a custom CUDA
    kernel to be memory and compute efficient.
    """

    # 初始化函数
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        # 判断config.attention_window的类型，并做相应处理
        if isinstance(config.attention_window, int):
            assert config.attention_window % 2 == 0, "`config.attention_window` has to be an even value"
            assert config.attention_window > 0, "`config.attention_window` has to be positive"
            config.attention_window = [config.attention_window] * config.num_hidden_layers  # one value per layer
        else:
            assert len(config.attention_window) == config.num_hidden_layers, (
                "`len(config.attention_window)` should equal `config.num_hidden_layers`. "
                f"Expected {config.num_hidden_layers}, given {len(config.attention_window)}"
            )

        # 初始化LongformerEmbeddings、LongformerEncoder和LongformerPooler
        self.embeddings = LongformerEmbeddings(config)
        self.encoder = LongformerEncoder(config)
        self.pooler = LongformerPooler(config) if add_pooling_layer else None

        # 初始化权重并应用最终处理
        self.post_init()

    # 获取输入的嵌入
    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    # 设置输入的嵌入
    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    # 剪枝模型的attention头部
    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)
    # 辅助函数，用于填充标记和掩码以便与Longformer自注意力实现一起工作
    def _pad_to_window_size(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
        position_ids: torch.Tensor,
        inputs_embeds: torch.Tensor,
        pad_token_id: int,
    ):
        """A helper function to pad tokens and mask to work with implementation of Longformer self-attention."""
        # 确定用于填充的大小，即 `attention_window` 参数值设为偶数
        attention_window = (
            self.config.attention_window
            if isinstance(self.config.attention_window, int)
            else max(self.config.attention_window)
        )

        assert attention_window % 2 == 0, f"`attention_window` should be an even value. Given {attention_window}"
        input_shape = input_ids.shape if input_ids is not None else inputs_embeds.shape
        batch_size, seq_len = input_shape[:2]

        # 计算要填充的长度，以使序列长度为 `attention_window` 的整数倍
        padding_len = (attention_window - seq_len % attention_window) % attention_window

        # 如果需要填充
        if padding_len > 0:
            # 输出警告信息，显示自动填充的信息
            logger.warning_once(
                f"Input ids are automatically padded from {seq_len} to {seq_len + padding_len} to be a multiple of "
                f"`config.attention_window`: {attention_window}"
            )
            # 如果存在 input_ids，则用 pad_token_id 填充
            if input_ids is not None:
                input_ids = nn.functional.pad(input_ids, (0, padding_len), value=pad_token_id)
            # 如果存在 position_ids，则用 pad_token_id 填充
            if position_ids is not None:
                # 和 modeling_roberta.RobertaEmbeddings 中一样，用 pad_token_id 填充
                position_ids = nn.functional.pad(position_ids, (0, padding_len), value=pad_token_id)
            # 如果存在 inputs_embeds，则进行填充
            if inputs_embeds is not None:
                input_ids_padding = inputs_embeds.new_full(
                    (batch_size, padding_len),
                    self.config.pad_token_id,
                    dtype=torch.long,
                )
                inputs_embeds_padding = self.embeddings(input_ids_padding)
                inputs_embeds = torch.cat([inputs_embeds, inputs_embeds_padding], dim=-2)

            # attention_mask 填充，填充值为 0，填充部分的注意力值为 0
            attention_mask = nn.functional.pad(
                attention_mask, (0, padding_len), value=0
            )  # no attention on the padding tokens
            # token_type_ids 填充，填充值为 0
            token_type_ids = nn.functional.pad(token_type_ids, (0, padding_len), value=0)  # pad with token_type_id = 0

        # 返回填充长度、填充后的 input_ids、attention_mask、token_type_ids、position_ids、inputs_embeds
        return padding_len, input_ids, attention_mask, token_type_ids, position_ids, inputs_embeds
    # 将局部注意力和全局注意力融合到注意力掩码中
    def _merge_to_attention_mask(self, attention_mask: torch.Tensor, global_attention_mask: torch.Tensor):
        # longformer 自注意力期望注意力掩码为 0（无注意力），1（局部注意力），2（全局注意力）
        # (global_attention_mask + 1) => 1 代表局部注意力，2 代表全局注意力
        # 最终的 attention_mask => 0 代表无注意力，1 代表局部注意力，2 代表全局注意力
        if attention_mask is not None:
            # 如果存在局部注意力掩码，将其与全局注意力掩码相乘并加 1，得到融合后的掩码
            attention_mask = attention_mask * (global_attention_mask + 1)
        else:
            # 如果没有给定局部注意力掩码，则直接使用全局注意力掩码加 1 作为注意力掩码
            attention_mask = global_attention_mask + 1
        return attention_mask

    # 将长形式模型的输入文档字符串添加到模型的前向方法中
    @add_start_docstrings_to_model_forward(LONGFORMER_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    # 替换返回文档字符串的输出类型为带池化的 Longformer 基础模型输出
    @replace_return_docstrings(output_type=LongformerBaseModelOutputWithPooling, config_class=_CONFIG_FOR_DOC)
    # 模型的前向传播方法
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        global_attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
# 为LongformerForMaskedLM模型添加文档字符串
@add_start_docstrings("""Longformer Model with a `language modeling` head on top.""", LONGFORMER_START_DOCSTRING)
class LongformerForMaskedLM(LongformerPreTrainedModel):
    # 设置共享权重的键值对
    _tied_weights_keys = ["lm_head.decoder"]

    # 初始化函数，传入配置参数
    def __init__(self, config):
        super().__init__(config)

        # 初始化Longformer模型，不添加池化层
        self.longformer = LongformerModel(config, add_pooling_layer=False)
        # 初始化LongformerLMHead层
        self.lm_head = LongformerLMHead(config)

        # 初始化权重并应用最终处理
        self.post_init()

    # 获取输出嵌入
    def get_output_embeddings(self):
        return self.lm_head.decoder

    # 设置输出嵌入
    def set_output_embeddings(self, new_embeddings):
        self.lm_head.decoder = new_embeddings

    # 前向传播函数，接受多种输入参数
    @add_start_docstrings_to_model_forward(LONGFORMER_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=LongformerMaskedLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        global_attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, LongformerMaskedLMOutput]:
        r"""
        定义函数的返回类型注解为 Tuple 或 LongformerMaskedLMOutput
        
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            用于计算掩码语言模型损失的标签。索引应该在 `[-100, 0, ..., config.vocab_size]`内（参见 `input_ids` 文档）。索引设置为`-100`的标记将被忽略（掩码），损失仅对标签为`[0, ..., config.vocab_size]`内的标记进行计算。
            
        kwargs (`Dict[str, any]`, optional, defaults to *{}*):
            用于隐藏已经被弃用的传统参数。

        Returns:
        返回结果：
        
        Mask filling example:
        掩码填充示例：

        ```python
        >>> from transformers import AutoTokenizer, LongformerForMaskedLM

        >>> tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096")
        >>> model = LongformerForMaskedLM.from_pretrained("allenai/longformer-base-4096")
        ```

        Let's try a very long input.
        让我们尝试一个非常长的输入。

        ```python
        >>> TXT = (
        ...     "My friends are <mask> but they eat too many carbs."
        ...     + " That's why I decide not to eat with them." * 300
        ... )
        >>> input_ids = tokenizer([TXT], return_tensors="pt")["input_ids"]
        >>> logits = model(input_ids).logits

        >>> masked_index = (input_ids[0] == tokenizer.mask_token_id).nonzero().item()
        >>> probs = logits[0, masked_index].softmax(dim=0)
        >>> values, predictions = probs.topk(5)

        >>> tokenizer.decode(predictions).split()
        ['healthy', 'skinny', 'thin', 'good', 'vegetarian']
        ```"""

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        如果 return_dict 不为空，则使用 return_dict 值；否则使用 self.config.use_return_dict 的值

        outputs = self.longformer(
            input_ids,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask,
            head_mask=head_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        使用输入参数调用 self.longformer 方法，并将结果赋给 outputs
        sequence_output = outputs[0]
        从 outputs 中获取第一个元素并赋给 sequence_output
        prediction_scores = self.lm_head(sequence_output)
        使用 sequence_output 调用 self.lm_head 方法并将结果赋给 prediction_scores

        masked_lm_loss = None
        初始化 masked_lm_loss 为 None
        if labels is not None:
            如果 labels 不为空：
            创建交叉熵损失函数实例
            loss_fct = CrossEntropyLoss()

            将 labels 移动到 prediction_scores 的设备上
            labels = labels.to(prediction_scores.device)
            使用 loss_fct 计算损失，并将结果赋给 masked_lm_loss
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        如果 return_dict 为假：
        如果 labels 不为空，则返回 masked_lm_loss 和 output 的元组；否则返回 output
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        如果 return_dict 为真：
        返回 LongformerMaskedLMOutput 对象，包括损失、预测分数、隐藏状态、关注力和全局关注力
        return LongformerMaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            global_attentions=outputs.global_attentions,
        )
# 定义一个Longformer模型，该模型在顶部具有序列分类/回归头部（在池化输出之上的线性层），例如用于GLUE任务
@add_start_docstrings(
    """
    Longformer Model transformer with a sequence classification/regression head on top (a linear layer on top of the
    pooled output) e.g. for GLUE tasks.
    """,
    LONGFORMER_START_DOCSTRING,
)
class LongformerForSequenceClassification(LongformerPreTrainedModel):
    # 初始化函数
    def __init__(self, config):
        super().__init__(config)
        # 初始化类别数量
        self.num_labels = config.num_labels
        self.config = config

        # 创建LongformerModel对象
        self.longformer = LongformerModel(config, add_pooling_layer=False)
        # 创建LongformerClassificationHead对象
        self.classifier = LongformerClassificationHead(config)

        # 初始化权重并应用最终处理
        self.post_init()

    # 前向传播函数
    @add_start_docstrings_to_model_forward(LONGFORMER_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint="jpwahle/longformer-base-plagiarism-detection",
        output_type=LongformerSequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output="'ORIGINAL'",
        expected_loss=5.44,
    )
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        global_attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        # 输入参数及其类型注释
    # 定义返回类型，可以是元组或 LongformerSequenceClassifierOutput 对象
        ) -> Union[Tuple, LongformerSequenceClassifierOutput]:
            r"""
            # 文档字符串解释 labels 参数，用于计算分类/回归损失
            labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
                # labels 的格式，及其值范围
                Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
                config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
                `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
            """
            # 根据 return_dict 是否为 None 设置返回字典的标志
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    
            # 如果全局注意力掩码为 None，初始化全局注意力并设置警告
            if global_attention_mask is None:
                logger.warning_once("Initializing global attention on CLS token...")
                # 创建与输入 ID 相同形状的全零张量
                global_attention_mask = torch.zeros_like(input_ids)
                # 将第一列设置为 1，表示对 CLS 标记进行全局注意力
                global_attention_mask[:, 0] = 1
    
            # 调用 Longformer 模型，传入各种参数，包括输入 ID、掩码、头部掩码、等
            outputs = self.longformer(
                input_ids,
                attention_mask=attention_mask,
                global_attention_mask=global_attention_mask,
                head_mask=head_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
    
            # 获取 Longformer 模型输出的第一个元素，通常是序列输出
            sequence_output = outputs[0]
            # 通过分类器对序列输出进行分类，得到 logits
            logits = self.classifier(sequence_output)
    
            # 初始化损失为 None
            loss = None
            # 如果提供了 labels，则计算损失
            if labels is not None:
                # 确保 labels 在与 logits 相同的设备上
                labels = labels.to(logits.device)
    
                # 如果 problem_type 未设置，根据条件自动确定
                if self.config.problem_type is None:
                    if self.num_labels == 1:
                        # 如果只有一个标签，则为回归问题
                        self.config.problem_type = "regression"
                    elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                        # 如果多个标签且 labels 类型为整数，则为单标签分类
                        self.config.problem_type = "single_label_classification"
                    else:
                        # 否则为多标签分类
                        self.config.problem_type = "multi_label_classification"
    
                # 根据 problem_type 选择适当的损失函数
                if self.config.problem_type == "regression":
                    # 使用均方误差损失
                    loss_fct = MSELoss()
                    if self.num_labels == 1:
                        # 单标签回归
                        loss = loss_fct(logits.squeeze(), labels.squeeze())
                    else:
                        # 多标签回归
                        loss = loss_fct(logits, labels)
                elif self.config.problem_type == "single_label_classification":
                    # 使用交叉熵损失
                    loss_fct = CrossEntropyLoss()
                    loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                elif self.config.problem_type == "multi_label_classification":
                    # 使用带 Logits 的二元交叉熵损失
                    loss_fct = BCEWithLogitsLoss()
                    loss = loss_fct(logits, labels)
    
            # 如果不返回字典，输出为元组
            if not return_dict:
                # 合并损失和其他输出，并返回
                output = (logits,) + outputs[2:]
                return ((loss,) + output) if loss is not None else output
    
            # 如果返回字典，创建 LongformerSequenceClassifierOutput 对象
            return LongformerSequenceClassifierOutput(
                loss=loss,
                logits=logits,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
                global_attentions=outputs.global_attentions,
            )
class LongformerClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        # 创建全连接层，输入维度为config.hidden_size，输出维度为config.hidden_size
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 随机失活层，参数为config.hidden_dropout_prob
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # 输出层，输入维度为config.hidden_size，输出维度为config.num_labels
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, hidden_states, **kwargs):
        # 取出hidden_states的第一个位置的token作为输出
        hidden_states = hidden_states[:, 0, :]  # take <s> token (equiv. to [CLS])
        # 对输出进行随机失活
        hidden_states = self.dropout(hidden_states)
        # 通过全连接层进行线性变换
        hidden_states = self.dense(hidden_states)
        # 对输出进行tanh激活
        hidden_states = torch.tanh(hidden_states)
        # 通过全连接层进行线性变换
        hidden_states = self.dropout(hidden_states)
        # 通过输出层得到最终输出
        output = self.out_proj(hidden_states)
        return output

# 根据SQuAD / TriviaQA类型问题回答任务，创建Longformer模型和跨度分类头部的模型
@add_start_docstrings(
    """
    Longformer Model with a span classification head on top for extractive question-answering tasks like SQuAD /
    TriviaQA (a linear layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    LONGFORMER_START_DOCSTRING,
)
class LongformerForQuestionAnswering(LongformerPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        # 定义类别数量
        self.num_labels = config.num_labels
        # 创建Longformer模型，不添加池化层
        self.longformer = LongformerModel(config, add_pooling_layer=False)
        # 创建全连接层，输入维度为config.hidden_size，输出维度为config.num_labels
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        # 初始化权重并进行最终处理
        self.post_init()

    # 对Longformer模型进行前向传播
    @add_start_docstrings_to_model_forward(LONGFORMER_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=LongformerQuestionAnsweringModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        global_attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        start_positions: Optional[torch.Tensor] = None,
        end_positions: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        
# 创建Longformer模型和标记分类头部的模型
@add_start_docstrings(
    """
    Longformer Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g.
    for Named-Entity-Recognition (NER) tasks.
    """,
LONGFORMER_START_DOCSTRING,
)
class LongformerForTokenClassification(LongformerPreTrainedModel):
    # 构造函数，初始化模型组件
    def __init__(self, config):
        # 调用父类构造函数
        super().__init__(config)
        # 从配置中获取标签数量并存储
        self.num_labels = config.num_labels

        # 创建 Longformer 模型实例，不添加池化层
        self.longformer = LongformerModel(config, add_pooling_layer=False)
        # 创建 Dropout 层，用配置中的 dropout 概率
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # 创建一个全连接层，将 hidden_size 映射到标签数量
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # 初始化权重并进行后续处理
        self.post_init()

    # 使用装饰器添加文档字符串，定义模型输入格式
    @add_start_docstrings_to_model_forward(LONGFORMER_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    # 使用装饰器添加代码样例文档，提供模型输出类型，配置类，期望输出及损失
    @add_code_sample_docstrings(
        checkpoint="brad1141/Longformer-finetuned-norm",
        output_type=LongformerTokenClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output=(
            "['Evidence', 'Evidence', 'Evidence', 'Evidence', 'Evidence', 'Evidence', 'Evidence', 'Evidence',"
            " 'Evidence', 'Evidence', 'Evidence', 'Evidence']"
        ),
        expected_loss=0.63,
    )
    # 定义前向传播函数
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,  # 可选的输入 ID
        attention_mask: Optional[torch.Tensor] = None,  # 可选的注意力掩码
        global_attention_mask: Optional[torch.Tensor] = None,  # 可选的全局注意力掩码
        head_mask: Optional[torch.Tensor] = None,  # 可选的头部掩码
        token_type_ids: Optional[torch.Tensor] = None,  # 可选的令牌类型 ID
        position_ids: Optional[torch.Tensor] = None,  # 可选的位置 ID
        inputs_embeds: Optional[torch.Tensor] = None,  # 可选的输入嵌入
        labels: Optional[torch.Tensor] = None,  # 可选的标签
        output_attentions: Optional[bool] = None,  # 是否输出注意力权重
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态
        return_dict: Optional[bool] = None,  # 是否以字典形式返回输出
    # 该函数用于计算Longformer模型的输出和损失函数
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        global_attention_mask=None,
        head_mask=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ) -> Union[Tuple, LongformerTokenClassifierOutput]:
        # 如果未指定return_dict，则使用模型默认配置
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # 通过Longformer模型获得输出结果
        outputs = self.longformer(
            input_ids,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask,
            head_mask=head_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
    
        # 获取输出的序列编码
        sequence_output = outputs[0]
    
        # 对输出序列编码应用dropout
        sequence_output = self.dropout(sequence_output)
        
        # 通过分类器层计算logits
        logits = self.classifier(sequence_output)
    
        # 如果存在标签，则计算损失函数
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            
            # 将标签转移到logits的设备上
            labels = labels.to(logits.device)
            
            # 计算交叉熵损失
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
    
        # 如果不使用返回字典，则返回logits以及其他输出
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
    
        # 否则返回LongformerTokenClassifierOutput
        return LongformerTokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            global_attentions=outputs.global_attentions,
        )
# 引入自定义文档字符串的装饰器，描述了 Longformer 模型结构及其在多选分类任务中的应用
@add_start_docstrings(
    """
    Longformer Model with a multiple choice classification head on top (a linear layer on top of the pooled output and
    a softmax) e.g. for RocStories/SWAG tasks.
    """,
    LONGFORMER_START_DOCSTRING,  # 引入 Longformer 模型的基本文档字符串
)
# 定义 Longformer 多选分类模型，继承自 Longformer 预训练模型
class LongformerForMultipleChoice(LongformerPreTrainedModel):
    # 初始化方法
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)
        # 创建 Longformer 模型实例
        self.longformer = LongformerModel(config)
        # 定义 dropout 层，用于随机失活
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # 定义分类器，使用线性层进行多选分类
        self.classifier = nn.Linear(config.hidden_size, 1)

        # 初始化权重并应用最终处理
        self.post_init()

    # 前向传播方法
    @add_start_docstrings_to_model_forward(
        LONGFORMER_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length")  # 引入输入文档字符串
    )
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,  # 引入检查点文档字符串
        output_type=LongformerMultipleChoiceModelOutput,  # 引入输出类型文档字符串
        config_class=_CONFIG_FOR_DOC,  # 引入配置类文档字符串
    )
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,  # 输入的 token ID
        token_type_ids: Optional[torch.Tensor] = None,  # token 类型 ID
        attention_mask: Optional[torch.Tensor] = None,  # 注意力掩码
        global_attention_mask: Optional[torch.Tensor] = None,  # 全局注意力掩码
        head_mask: Optional[torch.Tensor] = None,  # 头注意力掩码
        labels: Optional[torch.Tensor] = None,  # 标签
        position_ids: Optional[torch.Tensor] = None,  # 位置 ID
        inputs_embeds: Optional[torch.Tensor] = None,  # 输入的嵌入向量
        output_attentions: Optional[bool] = None,  # 是否输出注意力权重
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态
        return_dict: Optional[bool] = None,  # 是否返回字典
```