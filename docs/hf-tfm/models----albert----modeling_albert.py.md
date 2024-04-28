# `.\transformers\models\albert\modeling_albert.py`

```py
# 设置文件编码为 UTF-8
# 版权声明，版权归 Google AI、Google Brain 和 HuggingFace Inc. 团队所有
# 根据 Apache 许可证 2.0 版本授权，除非符合许可证的规定，否则不得使用此文件
# 可以在以下网址获取许可证的副本：http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于“原样”分发的，没有任何明示或暗示的保证或条件
# 请查看许可证以获取有关特定语言的权限和限制
"""PyTorch ALBERT 模型。"""

# 导入所需的库
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

# 导入自定义的激活函数和模型输出
from ...activations import ACT2FN
from ...modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPooling,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
# 导入模型工具函数
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
# 导入 ALBERT 配置
from .configuration_albert import AlbertConfig

# 获取日志记录器
logger = logging.get_logger(__name__)

# 用于文档的检查点和配置
_CHECKPOINT_FOR_DOC = "albert-base-v2"
_CONFIG_FOR_DOC = "AlbertConfig"

# ALBERT 预训练模型存档列表
ALBERT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "albert-base-v1",
    "albert-large-v1",
    "albert-xlarge-v1",
    "albert-xxlarge-v1",
    "albert-base-v2",
    "albert-large-v2",
    "albert-xlarge-v2",
    "albert-xxlarge-v2",
    # 查看所有 ALBERT 模型：https://huggingface.co/models?filter=albert
]

# 加载 TensorFlow 模型权重到 PyTorch 模型
def load_tf_weights_in_albert(model, config, tf_checkpoint_path):
    """Load tf checkpoints in a pytorch model."""
    try:
        import re
        import numpy as np
        import tensorflow as tf
    except ImportError:
        logger.error(
            "Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions."
        )
        raise
    # 获取 TensorFlow 检查点的绝对路径
    tf_path = os.path.abspath(tf_checkpoint_path)
    logger.info(f"Converting TensorFlow checkpoint from {tf_path}")
    # 从 TF 模型加载权重
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    for name, shape in init_vars:
        logger.info(f"Loading TF weight {name} with shape {shape}")
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array)

    for name, array in zip(names, arrays):
        print(name)

    return model

# ALBERT 嵌入层模块
class AlbertEmbeddings(nn.Module):
    """
    # 构建嵌入层，由单词、位置和令牌类型嵌入组成
    def __init__(self, config: AlbertConfig):
        # 调用父类构造函数
        super().__init__()
        # 初始化单词嵌入层，vocab_size 表示词汇表大小，embedding_size 表示嵌入维度，padding_idx 表示填充标记的索引
        self.word_embeddings = nn.Embedding(config.vocab_size, config.embedding_size, padding_idx=config.pad_token_id)
        # 初始化位置嵌入层，max_position_embeddings 表示最大位置嵌入数目
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.embedding_size)
        # 初始化令牌类型嵌入层，type_vocab_size 表示令牌类型数目
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.embedding_size)

        # self.LayerNorm 没有采用蛇形命名以保持与 TensorFlow 模型变量名一致，并且能够加载任何 TensorFlow 检查点文件
        # 初始化 LayerNorm 层，embedding_size 表示嵌入维度，eps 表示 LayerNorm 的 epsilon 值
        self.LayerNorm = nn.LayerNorm(config.embedding_size, eps=config.layer_norm_eps)
        # 初始化 Dropout 层，hidden_dropout_prob 表示隐藏层的 dropout 概率
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # position_ids (1, len position emb) 在内存中是连续的，并在序列化时导出
        # 注册 position_ids 缓冲区，torch.arange 生成从 0 到 max_position_embeddings-1 的序列
        # persistent=False 表示该缓冲区不会被 torch.nn.Module.state_dict() 方法保存
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )
        # 获取配置中的 position_embedding_type，表示位置嵌入的类型，默认为 "absolute"
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        # 初始化 token_type_ids 缓冲区，与 position_ids 相同形状的零张量
        self.register_buffer(
            "token_type_ids", torch.zeros(self.position_ids.size(), dtype=torch.long), persistent=False
        )

    # 从 transformers.models.bert.modeling_bert.BertEmbeddings.forward 复制而来
    # 前向传播函数
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values_length: int = 0,
    # 定义函数的输入参数和返回类型
    ) -> torch.Tensor:
        # 如果输入的 input_ids 不为空，则获取其形状
        if input_ids is not None:
            input_shape = input_ids.size()
        # 如果 input_ids 为空，则获取 inputs_embeds 的形状，去掉最后一个维度
        else:
            input_shape = inputs_embeds.size()[:-1]

        # 获取序列长度
        seq_length = input_shape[1]

        # 如果 position_ids 为空，则从 self.position_ids 中获取对应位置的位置编码
        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]

        # 如果 token_type_ids 为空，则根据模型是否有 token_type_ids 属性来处理
        if token_type_ids is None:
            # 如果模型有 token_type_ids 属性，则使用其中的值，扩展到与输入形状相同
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            # 如果模型没有 token_type_ids 属性，则创建全零的 token_type_ids
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        # 如果 inputs_embeds 为空，则根据 input_ids 获取对应的词嵌入
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        # 获取 token_type_ids 对应的 token 类型嵌入
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        # 将词嵌入和 token 类型嵌入相加得到 embeddings
        embeddings = inputs_embeds + token_type_embeddings
        # 如果位置嵌入类型为 "absolute"，则获取对应的位置嵌入并加到 embeddings 中
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        # 对 embeddings 进行 LayerNorm 处理
        embeddings = self.LayerNorm(embeddings)
        # 对 embeddings 进行 dropout 处理
        embeddings = self.dropout(embeddings)
        # 返回处理后的 embeddings
        return embeddings
class AlbertAttention(nn.Module):
    def __init__(self, config: AlbertConfig):
        super().__init__()
        # 检查隐藏层大小是否能被注意力头数整除，如果不能整除并且配置中没有嵌入大小，则引发值错误
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads}"
            )

        # 初始化 AlbertAttention 类
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # 定义查询、键和值的线性层
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        # 定义注意力和输出的 dropout 层
        self.attention_dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.output_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.pruned_heads = set()

        # 设置位置嵌入类型，默认为绝对位置嵌入
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        # 如果是相对位置嵌入，则初始化距离嵌入层
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

    # 从 Transformers 的 BertSelfAttention 中复制 transpose_for_scores 方法
    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        # 调整张量形状以匹配注意力头的形状
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    # 剪枝指定的注意力头
    def prune_heads(self, heads: List[int]) -> None:
        if len(heads) == 0:
            return
        # 查找可剪枝的注意力头并获取索引
        heads, index = find_pruneable_heads_and_indices(
            heads, self.num_attention_heads, self.attention_head_size, self.pruned_heads
        )

        # 剪枝线性层
        self.query = prune_linear_layer(self.query, index)
        self.key = prune_linear_layer(self.key, index)
        self.value = prune_linear_layer(self.value, index)
        self.dense = prune_linear_layer(self.dense, index, dim=1)

        # 更新超参数并存储已剪枝的注意力头
        self.num_attention_heads = self.num_attention_heads - len(heads)
        self.all_head_size = self.attention_head_size * self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    # 前向传播函数
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: bool = False,
# 定义 AlbertLayer 类，继承自 nn.Module
class AlbertLayer(nn.Module):
    # 初始化函数，接受一个 AlbertConfig 对象作为参数
    def __init__(self, config: AlbertConfig):
        super().__init__()

        # 将配置对象保存在实例中
        self.config = config
        # 设置前向传播中用于分块的大小
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        # 设置序列长度的维度
        self.seq_len_dim = 1
        # 创建用于全连接层的 LayerNorm 层
        self.full_layer_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 创建注意力机制层
        self.attention = AlbertAttention(config)
        # 创建前馈神经网络层
        self.ffn = nn.Linear(config.hidden_size, config.intermediate_size)
        # 创建前馈神经网络输出层
        self.ffn_output = nn.Linear(config.intermediate_size, config.hidden_size)
        # 获取激活函数
        self.activation = ACT2FN[config.hidden_act]
        # 创建用于 dropout 的层
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    # 前向传播函数，接受隐藏状态、注意力掩码等参数，并返回隐藏状态和可能的注意力信息
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 调用注意力层的前向传播函数，获取注意力输出
        attention_output = self.attention(hidden_states, attention_mask, head_mask, output_attentions)

        # 对前向传播过程中的注意力输出应用分块机制
        ffn_output = apply_chunking_to_forward(
            self.ff_chunk,
            self.chunk_size_feed_forward,
            self.seq_len_dim,
            attention_output[0],
        )
        # 使用 LayerNorm 对注意力输出和前馈神经网络输出进行残差连接并归一化
        hidden_states = self.full_layer_layer_norm(ffn_output + attention_output[0])

        # 返回隐藏状态和可能的注意力信息
        return (hidden_states,) + attention_output[1:]  # add attentions if we output them

    # 前馈神经网络的分块函数，接受注意力输出，并返回前馈神经网络的输出
    def ff_chunk(self, attention_output: torch.Tensor) -> torch.Tensor:
        # 前馈神经网络的计算过程
        ffn_output = self.ffn(attention_output)
        ffn_output = self.activation(ffn_output)
        ffn_output = self.ffn_output(ffn_output)
        # 返回前馈神经网络的输出
        return ffn_output


# 定义 AlbertLayerGroup 类，继承自 nn.Module
class AlbertLayerGroup(nn.Module):
    # 初始化函数，接受一个 AlbertConfig 对象作为参数
    def __init__(self, config: AlbertConfig):
        super().__init__()

        # 创建 AlbertLayer 对象的列表，数量为配置中的内部组数
        self.albert_layers = nn.ModuleList([AlbertLayer(config) for _ in range(config.inner_group_num)])

    # 前向传播函数，接受隐藏状态、注意力掩码等参数，并返回隐藏状态和可能的注意力信息
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    # 定义一个空的元组，用于存储每个层的隐藏状态
    layer_hidden_states = ()
    # 定义一个空的元组，用于存储每个层的注意力分数
    layer_attentions = ()

    # 遍历所有的 ALBERT 层
    for layer_index, albert_layer in enumerate(self.albert_layers):
        # 调用当前 ALBERT 层的前向传播方法，得到该层的输出
        layer_output = albert_layer(hidden_states, attention_mask, head_mask[layer_index], output_attentions)
        # 更新隐藏状态为当前层的输出
        hidden_states = layer_output[0]

        # 如果需要输出注意力分数
        if output_attentions:
            # 将当前层的注意力分数添加到存储注意力分数的元组中
            layer_attentions = layer_attentions + (layer_output[1],)

        # 如果需要输出隐藏状态
        if output_hidden_states:
            # 将当前层的隐藏状态添加到存储隐藏状态的元组中
            layer_hidden_states = layer_hidden_states + (hidden_states,)

    # 构建模型输出
    outputs = (hidden_states,)
    # 如果需要输出隐藏状态，则将所有层的隐藏状态添加到模型输出中
    if output_hidden_states:
        outputs = outputs + (layer_hidden_states,)
    # 如果需要输出注意力分数，则将所有层的注意力分数添加到模型输出中
    if output_attentions:
        outputs = outputs + (layer_attentions,)
    # 返回模型输出，包括最后一层的隐藏状态，所有层的隐藏状态（如果需要），所有层的注意力分数（如果需要）
    return outputs  # 最后一层的隐藏状态，（所有层的隐藏状态），（所有层的注意力）
class AlbertTransformer(nn.Module):
    # 定义 AlbertTransformer 类，继承自 nn.Module
    def __init__(self, config: AlbertConfig):
        # 初始化函数，接受一个 AlbertConfig 类型的参数 config
        super().__init__()
        # 调用父类的初始化函数

        self.config = config
        # 将传入的 config 参数保存到实例变量中
        self.embedding_hidden_mapping_in = nn.Linear(config.embedding_size, config.hidden_size)
        # 创建一个线性层，用于将输入的 embedding 映射到隐藏层的维度
        self.albert_layer_groups = nn.ModuleList([AlbertLayerGroup(config) for _ in range(config.num_hidden_groups)])
        # 创建一个包含多个 AlbertLayerGroup 实例的 ModuleList

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Union[BaseModelOutput, Tuple]:
        # 前向传播函数，接受多个参数并返回一个 BaseModelOutput 或 Tuple 类型的结果
        hidden_states = self.embedding_hidden_mapping_in(hidden_states)
        # 将输入的 hidden_states 经过 embedding_hidden_mapping_in 层映射到隐藏层维度

        all_hidden_states = (hidden_states,) if output_hidden_states else None
        # 如果需要输出隐藏状态，则初始化一个包含 hidden_states 的元组，否则为 None
        all_attentions = () if output_attentions else None
        # 如果需要输出注意力权重，则初始化一个空元组，否则为 None

        head_mask = [None] * self.config.num_hidden_layers if head_mask is None else head_mask
        # 如果 head_mask 为 None，则初始化一个包含 None 的列表，长度为 num_hidden_layers

        for i in range(self.config.num_hidden_layers):
            # 遍历每个隐藏层
            layers_per_group = int(self.config.num_hidden_layers / self.config.num_hidden_groups)
            # 计算每个隐藏组中的层数
            group_idx = int(i / (self.config.num_hidden_layers / self.config.num_hidden_groups))
            # 计算当前隐藏层所在的隐藏组索引

            layer_group_output = self.albert_layer_groups[group_idx](
                hidden_states,
                attention_mask,
                head_mask[group_idx * layers_per_group : (group_idx + 1) * layers_per_group],
                output_attentions,
                output_hidden_states,
            )
            # 调用当前隐藏组的 AlbertLayerGroup 实例进行前向传播
            hidden_states = layer_group_output[0]
            # 更新隐藏状态为当前隐藏组的输出

            if output_attentions:
                all_attentions = all_attentions + layer_group_output[-1]
                # 如果需要输出注意力权重，则将当前隐藏组的注意力权重添加到 all_attentions 中

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
                # 如果需要输出隐藏状态，则将当前隐藏状态添加到 all_hidden_states 中

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_attentions] if v is not None)
            # 如果不需要返回字典形式的结果，则返回包含非 None 值的元组
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_attentions
        )
        # 返回一个 BaseModelOutput 实例

class AlbertPreTrainedModel(PreTrainedModel):
    # 定义 AlbertPreTrainedModel 类，继承自 PreTrainedModel
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    # 一个抽象类，用于处理权重初始化以及下载和加载预训练模型

    config_class = AlbertConfig
    # 设置 config_class 为 AlbertConfig 类
    load_tf_weights = load_tf_weights_in_albert
    # 设置 load_tf_weights 为 load_tf_weights_in_albert 函数
    base_model_prefix = "albert"
    # 设置 base_model_prefix 为 "albert"
    def _init_weights(self, module):
        """初始化权重"""
        # 如果是线性层
        if isinstance(module, nn.Linear):
            # 使用正态分布初始化权重，均值为0，标准差为配置中的initializer_range
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果存在偏置项，将其初始化为0
            if module.bias is not None:
                module.bias.data.zero_()
        # 如果是嵌入层
        elif isinstance(module, nn.Embedding):
            # 使用正态分布初始化权重，均值为0，标准差为配置中的initializer_range
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果存在padding_idx，将其对应的权重初始化为0
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        # 如果是LayerNorm层
        elif isinstance(module, nn.LayerNorm):
            # 将偏置项初始化为0
            module.bias.data.zero_()
            # 将权重初始化为1
            module.weight.data.fill_(1.0)
@dataclass
class AlbertForPreTrainingOutput(ModelOutput):
    """
    Output type of [`AlbertForPreTraining`].

    Args:
        loss (*optional*, returned when `labels` is provided, `torch.FloatTensor` of shape `(1,)`):
            Total loss as the sum of the masked language modeling loss and the next sequence prediction
            (classification) loss.
        prediction_logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        sop_logits (`torch.FloatTensor` of shape `(batch_size, 2)`):
            Prediction scores of the next sequence prediction (classification) head (scores of True/False continuation
            before SoftMax).
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None
    prediction_logits: torch.FloatTensor = None
    sop_logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


ALBERT_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Args:
        config ([`AlbertConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

ALBERT_INPUTS_DOCSTRING = r"""
    """
    Args:
        input_ids (`torch.LongTensor` of shape `({0})`):
            # 输入序列中单词在词汇表中的索引
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.__call__`] and
            [`PreTrainedTokenizer.encode`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.FloatTensor` of shape `({0})`, *optional*):
            # 避免在填充标记上执行注意力的遮罩，值为 1 表示不遮蔽，值为 0 表示遮蔽
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        token_type_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            # 指示输入的第一部分和第二部分的段标记索引，0 代表第一部分，1 代表第二部分
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0,
            1]`:

            - 0 corresponds to a *sentence A* token,
            - 1 corresponds to a *sentence B* token.

            [What are token type IDs?](../glossary#token-type-ids)
        position_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            # 每个输入序列标记在位置嵌入中的位置索引
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.max_position_embeddings - 1]`.

            [What are position IDs?](../glossary#position-ids)
        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            # 用于将自注意力模块的选定头部置零的掩码，值为 1 表示不遮蔽，值为 0 表示遮蔽
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        inputs_embeds (`torch.FloatTensor` of shape `({0}, hidden_size)`, *optional*):
            # 如果不想传递 input_ids，可以直接传递嵌入表示，以获得对输入序列索引转换为关联向量的更多控制权
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        output_attentions (`bool`, *optional*):
            # 是否返回所有注意力层的注意力张量
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            # 是否返回所有层的隐藏状态
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            # 是否返回一个 ModelOutput 而不是一个普通的元组
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""
定义 ALBERT 模型类，输出原始隐藏状态而不带任何特定的头部
"""
@add_start_docstrings(
    "The bare ALBERT Model transformer outputting raw hidden-states without any specific head on top.",
    ALBERT_START_DOCSTRING,
)
class AlbertModel(AlbertPreTrainedModel):
    config_class = AlbertConfig
    base_model_prefix = "albert"

    def __init__(self, config: AlbertConfig, add_pooling_layer: bool = True):
        super().__init__(config)

        self.config = config
        self.embeddings = AlbertEmbeddings(config)
        self.encoder = AlbertTransformer(config)
        if add_pooling_layer:
            self.pooler = nn.Linear(config.hidden_size, config.hidden_size)
            self.pooler_activation = nn.Tanh()
        else:
            self.pooler = None
            self.pooler_activation = None

        # 初始化权重并应用最终处理
        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune: Dict[int, List[int]]) -> None:
        """
        剪枝模型的头部。heads_to_prune: {layer_num: 要在此层中剪枝的头部列表} ALBERT 具有不同的架构，其层在组之间共享，然后有内部组。
        如果一个 ALBERT 模型有 12 个隐藏层和 2 个隐藏组，每个组有两个内部组，则总共有 4 个不同的层。

        这些层是扁平化的：索引 [0,1] 对应于第一个隐藏层的两个内部组，而 [2,3] 对应于第二个隐藏层的两个内部组。

        任何索引不在 [0,1,2,3] 中的层都会导致错误。有关剪枝头部的更多信息，请参见基类 PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            group_idx = int(layer / self.config.inner_group_num)
            inner_group_idx = int(layer - group_idx * self.config.inner_group_num)
            self.encoder.albert_layer_groups[group_idx].albert_layers[inner_group_idx].attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(ALBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPooling,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        # 设置输出注意力权重，默认为配置中的值
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 设置输出隐藏状态，默认为配置中的值
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 设置返回字典，默认为配置中的值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 检查输入参数，不能同时指定 input_ids 和 inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            # 如果指定了 input_ids，则检查填充和注意力掩码
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        # 获取输入的 batch_size 和 seq_length
        batch_size, seq_length = input_shape
        # 获取设备信息
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # 如果没有指定注意力掩码，则创建一个全为1的注意力掩码
        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        # 如果没有指定 token_type_ids，则根据情况创建
        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # 扩展注意力掩码的维度
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(self.dtype).min
        # 获取头部掩码
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        # 获取嵌入输出
        embedding_output = self.embeddings(
            input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
        )
        # 获取编码器输出
        encoder_outputs = self.encoder(
            embedding_output,
            extended_attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取序列输出
        sequence_output = encoder_outputs[0]

        # 获取池化输出
        pooled_output = self.pooler_activation(self.pooler(sequence_output[:, 0])) if self.pooler is not None else None

        # 如果不返回字典，则返回元组
        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        # 返回带池化的基础模型输出
        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
# 使用两个头部的 Albert 模型，包括一个用于掩码语言建模的头部和一个用于句子顺序预测（分类）的头部
@add_start_docstrings(
    """
    Albert Model with two heads on top as done during the pretraining: a `masked language modeling` head and a
    `sentence order prediction (classification)` head.
    """,
    ALBERT_START_DOCSTRING,
)
class AlbertForPreTraining(AlbertPreTrainedModel):
    # 定义与预测头部权重相连接的键列表
    _tied_weights_keys = ["predictions.decoder.bias", "predictions.decoder.weight"]

    # 初始化函数
    def __init__(self, config: AlbertConfig):
        # 调用父类初始化函数
        super().__init__(config)

        # 实例化 AlbertModel 对象
        self.albert = AlbertModel(config)
        # 实例化 AlbertMLMHead 对象
        self.predictions = AlbertMLMHead(config)
        # 实例化 AlbertSOPHead 对象
        self.sop_classifier = AlbertSOPHead(config)

        # 初始化权重并应用最终处理
        self.post_init()

    # 获取输出嵌入层
    def get_output_embeddings(self) -> nn.Linear:
        return self.predictions.decoder

    # 设置输出嵌入层
    def set_output_embeddings(self, new_embeddings: nn.Linear) -> None:
        self.predictions.decoder = new_embeddings

    # 获取输入嵌入层
    def get_input_embeddings(self) -> nn.Embedding:
        return self.albert.embeddings.word_embeddings

    # 正向传播函数
    @add_start_docstrings_to_model_forward(ALBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=AlbertForPreTrainingOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        sentence_order_label: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[AlbertForPreTrainingOutput, Tuple]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        sentence_order_label (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the next sequence prediction (classification) loss. Input should be a sequence pair
            (see `input_ids` docstring) Indices should be in `[0, 1]`. `0` indicates original order (sequence A, then
            sequence B), `1` indicates switched order (sequence B, then sequence A).

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, AlbertForPreTraining
        >>> import torch

        >>> tokenizer = AutoTokenizer.from_pretrained("albert-base-v2")
        >>> model = AlbertForPreTraining.from_pretrained("albert-base-v2")

        >>> input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)
        >>> # Batch size 1
        >>> outputs = model(input_ids)

        >>> prediction_logits = outputs.prediction_logits
        >>> sop_logits = outputs.sop_logits
        ```py"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用 Albert 模型进行预测
        outputs = self.albert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output, pooled_output = outputs[:2]

        # 生成预测的 token 序列
        prediction_scores = self.predictions(sequence_output)
        # 生成句子顺序预测
        sop_scores = self.sop_classifier(pooled_output)

        total_loss = None
        # 计算总损失
        if labels is not None and sentence_order_label is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
            sentence_order_loss = loss_fct(sop_scores.view(-1, 2), sentence_order_label.view(-1))
            total_loss = masked_lm_loss + sentence_order_loss

        if not return_dict:
            output = (prediction_scores, sop_scores) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        # 返回 AlbertForPreTrainingOutput 对象
        return AlbertForPreTrainingOutput(
            loss=total_loss,
            prediction_logits=prediction_scores,
            sop_logits=sop_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
class AlbertMLMHead(nn.Module):
    def __init__(self, config: AlbertConfig):
        super().__init__()
        
        # 初始化 LayerNorm 层，用于规范化输入张量
        self.LayerNorm = nn.LayerNorm(config.embedding_size, eps=config.layer_norm_eps)
        # 初始化偏置参数，用于生成预测分数
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        # 初始化全连接层，用于将隐藏状态映射到嵌入维度
        self.dense = nn.Linear(config.hidden_size, config.embedding_size)
        # 初始化线性层，用于生成词汇表大小的预测分数
        self.decoder = nn.Linear(config.embedding_size, config.vocab_size)
        # 初始化激活函数，根据配置选择激活函数类型
        self.activation = ACT2FN[config.hidden_act]
        # 将解码器的偏置设置为之前初始化的偏置参数
        self.decoder.bias = self.bias

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 将隐藏状态映射到嵌入维度
        hidden_states = self.dense(hidden_states)
        # 应用激活函数
        hidden_states = self.activation(hidden_states)
        # 使用 LayerNorm 对隐藏状态进行规范化
        hidden_states = self.LayerNorm(hidden_states)
        # 生成预测分数
        hidden_states = self.decoder(hidden_states)

        # 返回预测分数
        prediction_scores = hidden_states
        return prediction_scores

    def _tie_weights(self) -> None:
        # 如果两个权重被分开（在TPU上或偏置被重新调整大小时），则将它们绑定在一起
        self.bias = self.decoder.bias


class AlbertSOPHead(nn.Module):
    def __init__(self, config: AlbertConfig):
        super().__init__()
        
        # 初始化 dropout 层，用于防止过拟合
        self.dropout = nn.Dropout(config.classifier_dropout_prob)
        # 初始化线性分类器层，用于进行下游任务的分类
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, pooled_output: torch.Tensor) -> torch.Tensor:
        # 应用 dropout 层
        dropout_pooled_output = self.dropout(pooled_output)
        # 通过分类器层生成 logits
        logits = self.classifier(dropout_pooled_output)
        return logits


@add_start_docstrings(
    "Albert Model with a `language modeling` head on top.",
    ALBERT_START_DOCSTRING,
)
class AlbertForMaskedLM(AlbertPreTrainedModel):
    _tied_weights_keys = ["predictions.decoder.bias", "predictions.decoder.weight"]

    def __init__(self, config):
        super().__init__(config)
        
        # 初始化 Albert 模型，不添加池化层
        self.albert = AlbertModel(config, add_pooling_layer=False)
        # 初始化 Masked LM 头部
        self.predictions = AlbertMLMHead(config)

        # 初始化权重并应用最终处理
        self.post_init()

    def get_output_embeddings(self) -> nn.Linear:
        # 返回输出嵌入层（预测分数层）
        return self.predictions.decoder

    def set_output_embeddings(self, new_embeddings: nn.Linear) -> None:
        # 设置输出嵌入层（预测分数层）的新权重
        self.predictions.decoder = new_embeddings

    def get_input_embeddings(self) -> nn.Embedding:
        # 返回输入嵌入层（Albert 模型中的词嵌入层）
        return self.albert.embeddings.word_embeddings

    @add_start_docstrings_to_model_forward(ALBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=MaskedLMOutput, config_class=_CONFIG_FOR_DOC)
    # 此方法用于模型的前向传播，接受各种输入参数并返回相应的输出
    def forward(
        self,
        # 输入的 token IDs，类型为 LongTensor，可选参数，默认为 None
        input_ids: Optional[torch.LongTensor] = None,
        # 注意力掩码，类型为 FloatTensor，可选参数，默认为 None
        attention_mask: Optional[torch.FloatTensor] = None,
        # token 类型 IDs，类型为 LongTensor，可选参数，默认为 None
        token_type_ids: Optional[torch.LongTensor] = None,
        # 位置 IDs，类型为 LongTensor，可选参数，默认为 None
        position_ids: Optional[torch.LongTensor] = None,
        # 头部掩码，类型为 FloatTensor，可选参数，默认为 None
        head_mask: Optional[torch.FloatTensor] = None,
        # 输入的嵌入向量，类型为 FloatTensor，可选参数，默认为 None
        inputs_embeds: Optional[torch.FloatTensor] = None,
        # 标签，类型为 LongTensor，可选参数，默认为 None
        labels: Optional[torch.LongTensor] = None,
        # 是否返回注意力权重，类型为 bool，可选参数，默认为 None
        output_attentions: Optional[bool] = None,
        # 是否返回隐藏状态，类型为 bool，可选参数，默认为 None
        output_hidden_states: Optional[bool] = None,
        # 是否以字典形式返回结果，类型为 bool，可选参数，默认为 None
        return_dict: Optional[bool] = None,
    ) -> Union[MaskedLMOutput, Tuple]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`

        Returns:
            返回值:

        Example:
            示例:

        ```python
        >>> import torch
        >>> from transformers import AutoTokenizer, AlbertForMaskedLM

        >>> tokenizer = AutoTokenizer.from_pretrained("albert-base-v2")
        >>> model = AlbertForMaskedLM.from_pretrained("albert-base-v2")

        >>> # add mask_token
        >>> inputs = tokenizer("The capital of [MASK] is Paris.", return_tensors="pt")
        >>> with torch.no_grad():
        ...     logits = model(**inputs).logits

        >>> # retrieve index of [MASK]
        >>> mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
        >>> predicted_token_id = logits[0, mask_token_index].argmax(axis=-1)
        >>> tokenizer.decode(predicted_token_id)
        'france'
        ```py

        ```python
        >>> labels = tokenizer("The capital of France is Paris.", return_tensors="pt")["input_ids"]
        >>> labels = torch.where(inputs.input_ids == tokenizer.mask_token_id, labels, -100)
        >>> outputs = model(**inputs, labels=labels)
        >>> round(outputs.loss.item(), 2)
        0.81
        ```py
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.albert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_outputs = outputs[0]

        prediction_scores = self.predictions(sequence_outputs)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 使用带有顶部序列分类/回归头的 Albert 模型变压器（在池化输出的顶部有一个线性层），例如用于 GLUE 任务
class AlbertForSequenceClassification(AlbertPreTrainedModel):
    def __init__(self, config: AlbertConfig):
        super().__init__(config)
        # 配置模型的标签数量
        self.num_labels = config.num_labels
        # 保存配置
        self.config = config

        # Albert 模型
        self.albert = AlbertModel(config)
        # 丢弃层，用于防止过拟合
        self.dropout = nn.Dropout(config.classifier_dropout_prob)
        # 分类器线性层，用于序列分类/回归
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)

        # 初始化权重并应用最终处理
        self.post_init()

    # Albert 模型的前向传播函数
    # 输入参数详见 ALBERT_INPUTS_DOCSTRING
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
```  
    ) -> Union[SequenceClassifierOutput, Tuple]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        # 设置返回字典，如果未指定则使用配置中的返回字典
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用 ALBERT 模型进行前向传播
        outputs = self.albert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取池化后的输出
        pooled_output = outputs[1]

        # 对池化后的输出进行 dropout
        pooled_output = self.dropout(pooled_output)
        # 将池化后的输出传入分类器得到 logits
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            # 判断问题类型，如果未指定则根据标签类型和数量确定问题类型
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            # 根据问题类型计算损失
            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        # 如果不需要返回字典，则返回输出
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 返回序列分类器输出对象
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 基于 ALBERT 模型的标记分类头部，用于命名实体识别（NER）等任务
@add_start_docstrings(
    """
    Albert Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
    Named-Entity-Recognition (NER) tasks.
    """,
    ALBERT_START_DOCSTRING,
)
class AlbertForTokenClassification(AlbertPreTrainedModel):
    def __init__(self, config: AlbertConfig):
        # 调用父类的构造函数
        super().__init__(config)
        # 获取标签数量
        self.num_labels = config.num_labels

        # 实例化 ALBERT 模型，不添加池化层
        self.albert = AlbertModel(config, add_pooling_layer=False)
        # 分类器的丢弃概率，若未提供则使用隐藏层的丢弃概率
        classifier_dropout_prob = (
            config.classifier_dropout_prob
            if config.classifier_dropout_prob is not None
            else config.hidden_dropout_prob
        )
        # 初始化丢弃层
        self.dropout = nn.Dropout(classifier_dropout_prob)
        # 全连接层，将隐藏状态映射到标签数量上
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)

        # 初始化权重并应用最终处理
        self.post_init()

    @add_start_docstrings_to_model_forward(ALBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TokenClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[TokenClassifierOutput, Tuple]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional`):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        # 如果 return_dict 为 None，则使用 self.config.use_return_dict 的值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用 ALBERT 模型进行前向传播
        outputs = self.albert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取 ALBERT 模型输出的序列输出
        sequence_output = outputs[0]

        # 对序列输出进行 dropout 处理
        sequence_output = self.dropout(sequence_output)
        # 将处理后的序列输出传入分类器，得到分类结果 logits
        logits = self.classifier(sequence_output)

        # 初始化损失值为 None
        loss = None
        # 如果存在标签，则计算交叉熵损失
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        # 如果 return_dict 为 False，则返回输出结果
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 返回 TokenClassifierOutput 对象，包含损失值、logits、隐藏状态和注意力权重
        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 使用 Albert 模型进行抽取式问答任务的模型，包含一个用于分类的线性层来计算“起始位置对数”和“结束位置对数”的模型
class AlbertForQuestionAnswering(AlbertPreTrainedModel):
    def __init__(self, config: AlbertConfig):
        # 调用父类的初始化方法
        super().__init__(config)
        # 获取标签数量
        self.num_labels = config.num_labels

        # 创建 Albert 模型，不添加池化层
        self.albert = AlbertModel(config, add_pooling_layer=False)
        # 创建一个线性层用于输出问题回答
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        # 初始化权重并应用最终处理
        self.post_init()

    # 前向传播方法
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        start_positions: Optional[torch.LongTensor] = None,
        end_positions: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[AlbertForPreTrainingOutput, Tuple]:
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
        # Decide whether to return a dictionary of outputs based on the configuration
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Pass input data to Albert model for processing
        outputs = self.albert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # Get the output tensor from the model
        sequence_output = outputs[0]

        # Obtain logits for question answering task
        logits: torch.Tensor = self.qa_outputs(sequence_output)
        
        # Split logits into start and end logits
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        # Initialize total loss variable
        total_loss = None
        
        # Calculate total loss if start and end positions are provided
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # Clamp positions to ignore terms outside the model inputs
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            # Define loss function and compute start and end losses
            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        # If return_dict is False, return output tuple
        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output
        
        # If return_dict is True, return QuestionAnsweringModelOutput object
        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 使用 ALBERT 模型进行多选分类任务的模型定义，包含一个线性层（用于池化输出之上的线性层和 softmax）例如 RocStories/SWAG 任务
@add_start_docstrings(
    """
    Albert Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
    softmax) e.g. for RocStories/SWAG tasks.
    """,
    ALBERT_START_DOCSTRING,
)
class AlbertForMultipleChoice(AlbertPreTrainedModel):
    # 初始化函数，接受 ALBERT 配置参数
    def __init__(self, config: AlbertConfig):
        # 调用父类的初始化函数
        super().__init__(config)

        # 创建 ALBERT 模型
        self.albert = AlbertModel(config)
        # 添加 dropout 层
        self.dropout = nn.Dropout(config.classifier_dropout_prob)
        # 添加分类器线性层
        self.classifier = nn.Linear(config.hidden_size, 1)

        # 初始化权重并应用最终处理
        self.post_init()

    # 前向传播函数
    @add_start_docstrings_to_model_forward(ALBERT_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=MultipleChoiceModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        ) -> Union[AlbertForPreTrainingOutput, Tuple]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the multiple choice classification loss. Indices should be in `[0, ...,
            num_choices-1]` where *num_choices* is the size of the second dimension of the input tensors. (see
            *input_ids* above)
        """
        # 确定是否返回字典
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 获取选择数量
        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

        # 重塑输入数据
        input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )
        # 调用 Albert 模型
        outputs = self.albert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取池化输出
        pooled_output = outputs[1]

        # 对池化输出进行 dropout
        pooled_output = self.dropout(pooled_output)
        # 通过分类器获取 logits
        logits: torch.Tensor = self.classifier(pooled_output)
        # 重塑 logits
        reshaped_logits = logits.view(-1, num_choices)

        loss = None
        # 计算损失
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)

        # 如果不返回字典，则返回输出
        if not return_dict:
            output = (reshaped_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 返回多选模型输出
        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
```