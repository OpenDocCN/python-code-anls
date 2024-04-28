# `.\transformers\models\tapas\modeling_tapas.py`

```
# 设置文件编码为 UTF-8

# 导入需要的库
import enum  # 用于定义枚举类型
import math  # 数学函数库
import os  # 系统操作库
from dataclasses import dataclass  # 用于定义数据类
from typing import Optional, Tuple, Union  # 用于类型提示

import torch  # PyTorch 库
import torch.utils.checkpoint  # 用于模型检查点
from torch import nn  # PyTorch 神经网络模块
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss  # 损失函数

# 从 HuggingFace 库中导入模型输出类和工具函数
from ...activations import ACT2FN  # 激活函数
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling, MaskedLMOutput, SequenceClassifierOutput  # 模型输出类
from ...modeling_utils import PreTrainedModel  # 预训练模型基类
from ...pytorch_utils import (  # PyTorch 工具函数
    apply_chunking_to_forward,  # 应用分块到前向传播
    find_pruneable_heads_and_indices,  # 找到可剪枝的头和索引
    is_torch_greater_or_equal_than_1_12,  # 判断 PyTorch 版本是否大于等于 1.12
    prune_linear_layer,  # 剪枝线性层
)
from ...utils import (  # 工具函数
    ModelOutput,  # 模型输出
    add_start_docstrings,  # 添加文档字符串
    add_start_docstrings_to_model_forward,  # 为模型前向传播添加文档字符串
    logging,  # 日志模块
    replace_return_docstrings,  # 替换返回值文档字符串
)
from .configuration_tapas import TapasConfig  # 导入 TAPAS 配置类


logger = logging.get_logger(__name__)  # 获取日志记录器

# 如果 PyTorch 版本小于 1.12，则发出警告提示用户升级 PyTorch
if not is_torch_greater_or_equal_than_1_12:
    logger.warning(
        f"You are using torch=={torch.__version__}, but torch>=1.12.0 is required to use "
        "TapasModel. Please upgrade torch."
    )

_CONFIG_FOR_DOC = "TapasConfig"  # 用于文档的配置类名称
_CHECKPOINT_FOR_DOC = "google/tapas-base"  # 用于文档的检查点模型名称

# TAPAS 预训练模型存档列表
TAPAS_PRETRAINED_MODEL_ARCHIVE_LIST = [
    # 大型模型
    "google/tapas-large",
    "google/tapas-large-finetuned-sqa",
    "google/tapas-large-finetuned-wtq",
    "google/tapas-large-finetuned-wikisql-supervised",
    "google/tapas-large-finetuned-tabfact",
    # 基础模型
    "google/tapas-base",
    "google/tapas-base-finetuned-sqa",
    "google/tapas-base-finetuned-wtq",
    "google/tapas-base-finetuned-wikisql-supervised",
    "google/tapas-base-finetuned-tabfact",
    # 小型模型
    "google/tapas-small",
    "google/tapas-small-finetuned-sqa",
    "google/tapas-small-finetuned-wtq",
    "google/tapas-small-finetuned-wikisql-supervised",
    "google/tapas-small-finetuned-tabfact",
    # 迷你模型
    "google/tapas-mini",
    "google/tapas-mini-finetuned-sqa",
    "google/tapas-mini-finetuned-wtq",
    "google/tapas-mini-finetuned-wikisql-supervised",
    "google/tapas-mini-finetuned-tabfact",
    # 极小型模型
    "google/tapas-tiny",
    "google/tapas-tiny-finetuned-sqa",
    "google/tapas-tiny-finetuned-wtq",
    "google/tapas-tiny-finetuned-wikisql-supervised",
    "google/tapas-tiny-finetuned-tabfact",
    # 查看所有 TAPAS 模型请访问 https://huggingface.co/models?filter=tapas
]
# 定义训练模型时可能出现的除零错误阈值
EPSILON_ZERO_DIVISION = 1e-10
# 定义接近对数零的值
CLOSE_ENOUGH_TO_LOG_ZERO = -10000.0

# 定义表格问题回答模型的输出类型
@dataclass
class TableQuestionAnsweringOutput(ModelOutput):
    """
    Output type of [`TapasForQuestionAnswering`].

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` (and possibly `answer`, `aggregation_labels`, `numeric_values` and `numeric_values_scale` are provided)):
            Total loss as the sum of the hierarchical cell selection log-likelihood loss and (optionally) the
            semi-supervised regression loss and (optionally) supervised loss for aggregations.
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
            Prediction scores of the cell selection head, for every token.
        logits_aggregation (`torch.FloatTensor`, *optional*, of shape `(batch_size, num_aggregation_labels)`):
            Prediction scores of the aggregation head, for every aggregation operator.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the model at the output of each layer
            plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
            the self-attention heads.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    logits_aggregation: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

# 定义从 TensorFlow 模型加载权重到 PyTorch 模型的函数
def load_tf_weights_in_tapas(model, config, tf_checkpoint_path):
    """
    Load tf checkpoints in a PyTorch model. This is an adaptation from load_tf_weights_in_bert

    - add cell selection and aggregation heads
    - take into account additional token type embedding layers
    """
    # 导入所需的库
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
    # 从 TensorFlow 检查点读取权重变量名和对应的值
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    # 遍历初始化变量的元组列表，其中每个元组包含变量名和形状信息
    for name, shape in init_vars:
        # 记录日志，显示正在加载的 TensorFlow 权重的名称和形状
        logger.info(f"Loading TF weight {name} with shape {shape}")
        # 使用 TensorFlow 提供的函数加载指定路径下的变量数据，返回 NumPy 数组
        array = tf.train.load_variable(tf_path, name)
        # 将变量名添加到列表中
        names.append(name)
        # 将加载的变量数据添加到列表中
        arrays.append(array)
    
    # 返回加载的模型
    return model
class TapasEmbeddings(nn.Module):
    """
    Construct the embeddings from word, position and token_type embeddings. Same as BertEmbeddings but with a number of
    additional token type embeddings to encode tabular structure.
    """

    def __init__(self, config):
        # 调用父类的构造函数
        super().__init__()
        # 创建词嵌入层，使用nn.Embedding类，vocab_size为词汇表大小，hidden_size为隐藏层大小，padding_idx为填充标记的索引
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        # 创建位置嵌入层，使用nn.Embedding类，max_position_embeddings为最大位置嵌入数，hidden_size为隐藏层大小
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        # 创建类型嵌入层，遍历type_vocab_sizes，使用nn.Embedding类，每个类型嵌入层大小为hidden_size
        for i, type_vocab_sizes in enumerate(config.type_vocab_sizes):
            name = f"token_type_embeddings_{i}"  # 根据索引创建嵌入层的名称
            setattr(self, name, nn.Embedding(type_vocab_sizes, config.hidden_size))  # 动态创建类型嵌入层

        self.number_of_token_type_embeddings = len(config.type_vocab_sizes)  # 记录类型嵌入层的数量

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        # 创建LayerNorm层，用于对隐藏层进行归一化，eps为LayerNorm的epsilon值
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 创建dropout层，用于随机失活
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # 保存配置信息
        self.config = config
    # 定义前向传播函数，接受输入的标识符、标记类型ID、位置ID和输入嵌入
    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None):
        # 如果输入标识符不为空
        if input_ids is not None:
            # 获取输入标识符的形状
            input_shape = input_ids.size()
        else:
            # 否则获取输入嵌入的形状，不包括最后一个维度
            input_shape = inputs_embeds.size()[:-1]

        # 获取序列长度
        seq_length = input_shape[1]
        # 获取设备信息，如果输入标识符不为空则使用其设备信息，否则使用输入嵌入的设备信息
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # 如果位置ID为空
        if position_ids is None:
            # 创建绝对位置嵌入
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(input_shape)
            # 当self.config.reset_position_index_per_cell设置为True时，创建相对位置嵌入
            if self.config.reset_position_index_per_cell:
                # 形状为（batch_size，seq_len）的列索引
                col_index = IndexMap(token_type_ids[:, :, 1], self.config.type_vocab_sizes[1], batch_dims=1)
                # 形状为（batch_size，seq_len）的行索引
                row_index = IndexMap(token_type_ids[:, :, 2], self.config.type_vocab_sizes[2], batch_dims=1)
                # 形状为（batch_size，seq_len）的完整索引
                full_index = ProductIndexMap(col_index, row_index)
                # 形状为（max_rows * max_columns，）的每个单元格的第一个绝对位置
                first_position_per_segment = reduce_min(position_ids, full_index)[0]
                # ? 形状为（batch_size，seq_len）的每个标记的单元格的第一个绝对位置
                first_position = gather(first_position_per_segment, full_index)
                # 形状为（1，seq_len）的位置
                position = torch.arange(seq_length, dtype=torch.long, device=device).unsqueeze(0)
                position_ids = torch.min(
                    torch.as_tensor(self.config.max_position_embeddings - 1, device=device), position - first_position
                )

        # 如果标记类型ID为空
        if token_type_ids is None:
            # 创建形状为（输入形状 + 标记类型嵌入数）的零张量
            token_type_ids = torch.zeros(
                (input_shape + self.number_of_token_type_embeddings), dtype=torch.long, device=device
            )

        # 如果输入嵌入为空
        if inputs_embeds is None:
            # 使用输入标识符获取词嵌入
            inputs_embeds = self.word_embeddings(input_ids)

        # 获取位置嵌入
        position_embeddings = self.position_embeddings(position_ids)

        # 将输入嵌入和位置嵌入相加
        embeddings = inputs_embeds + position_embeddings

        # 对于每个标记类型嵌入
        for i in range(self.number_of_token_type_embeddings):
            # 获取名称
            name = f"token_type_embeddings_{i}"
            # 将标记类型嵌入相加到嵌入中
            embeddings += getattr(self, name)(token_type_ids[:, :, i])

        # 应用层归一化
        embeddings = self.LayerNorm(embeddings)
        # 应用dropout
        embeddings = self.dropout(embeddings)
        # 返回嵌入
        return embeddings
# 定义自注意力机制的类，继承自 nn.Module
class TapasSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 如果隐藏层大小不是注意力头数的倍数并且配置中没有嵌入大小，抛出错误
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size {config.hidden_size} is not a multiple of the number of attention "
                f"heads {config.num_attention_heads}"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # 初始化 query、key、value 线性层
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        # 初始化 dropout
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        # 是否是解码器
        self.is_decoder = config.is_decoder

    # 改变张量维度以便进行矩阵乘法
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
        ):
            # 根据隐状态进行查询，得到混合查询层
            mixed_query_layer = self.query(hidden_states)

            # 如果是作为交叉注意力模块实例化的，那么键和值来自编码器；
            # 注意力掩码需要使编码器的填充标记不被关注
            is_cross_attention = encoder_hidden_states is not None

            # 如果是交叉关注并且过去的键值不为空，则重用 k、v 和 cross_attentions
            if is_cross_attention and past_key_value is not None:
                key_layer = past_key_value[0]
                value_layer = past_key_value[1]
                attention_mask = encoder_attention_mask
            # 如果是交叉关注
            elif is_cross_attention:
                key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
                value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
                attention_mask = encoder_attention_mask
            # 如果过去的键值不为空
            elif past_key_value is not None:
                key_layer = self.transpose_for_scores(self.key(hidden_states))
                value_layer = self.transpose_for_scores(self.value(hidden_states))
                key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
                value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
            else:
                key_layer = self.transpose_for_scores(self.key(hidden_states))
                value_layer = self.transpose_for_scores(self.value(hidden_states))

            # 将混合查询层转置，以便计算得到原始的注意力得分
            query_layer = self.transpose_for_scores(mixed_query_layer)

            # 如果是解码器
            if self.is_decoder:
                past_key_value = (key_layer, value_layer)

            # 计算"查询"和"键"之间的点积，以获得原始的注意力得分
            attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
            attention_scores = attention_scores / math.sqrt(self.attention_head_size)
            # 如果存在注意力掩码
            if attention_mask is not None:
                # 应用注意力掩码
                attention_scores = attention_scores + attention_mask

            # 将注意力得分归一化为概率
            attention_probs = nn.functional.softmax(attention_scores, dim=-1)

            # 实际上，这是删除整个标记以进行关注，这可能看起来有点不同寻常，但来自原始Transformer的论文
            attention_probs = self.dropout(attention_probs)

            # 如果需要，掩盖头
            if head_mask is not None:
                attention_probs = attention_probs * head_mask

            # 对值层应用注意力概率，得到上下文层
            context_layer = torch.matmul(attention_probs, value_layer)

            context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
            new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
            context_layer = context_layer.view(*new_context_layer_shape)

            # 如果需要输出注意力则输出上下文层和注意力概率，否则只输出上下文层
            outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
            # 如果是解码器，输出还需包括过去的键值
            if self.is_decoder:
                outputs = outputs + (past_key_value,)
            return outputs
# 从transformers.models.bert.modeling_bert.BertSelfOutput中复制代码
# 定义TapasSelfOutput类，继承于nn.Module类
class TapasSelfOutput(nn.Module):
    # 初始化方法，接收config参数
    def __init__(self, config):
        super().__init__()
        # 创建一个全连接层，输入大小为config.hidden_size，输出大小为config.hidden_size
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 创建LayerNorm层，输入大小为config.hidden_size，同时设定eps参数为config.layer_norm_eps
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 创建一个Dropout层，概率为config.hidden_dropout_prob
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    # 前向传播方法，接收hidden_states和input_tensor参数
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 将hidden_states通过全连接层dense
        hidden_states = self.dense(hidden_states)
        # 通过Dropout层
        hidden_states = self.dropout(hidden_states)
        # 通过LayerNorm层，将结果与input_tensor相加
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        # 返回结果
        return hidden_states


# 定义TapasAttention类，继承于nn.Module类
class TapasAttention(nn.Module):
    # 初始化方法，接收config参数
    def __init__(self, config):
        super().__init__()
        # 创建TapasSelfAttention对象
        self.self = TapasSelfAttention(config)
        # 创建TapasSelfOutput对象
        self.output = TapasSelfOutput(config)
        # 创建一个set对象pruned_heads
        self.pruned_heads = set()

    # 从transformers.models.bert.modeling_bert.BertAttention.prune_heads中复制代码
    # 定义prune_heads方法，接收heads参数
    def prune_heads(self, heads):
        # 如果heads为空集合，则直接返回
        if len(heads) == 0:
            return
        # 调用find_pruneable_heads_and_indices函数，返回heads和index
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # 剪枝线性层
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # 更新超参数并存储被剪枝的头部
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    # 从transformers.models.bert.modeling_bert.BertAttention.forward中复制代码
    # 前向传播方法，接收各种参数
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
        # 调用self.self的前向传播方法
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        # 通过self.output层处理self.self的结果，再与原始hidden_states相加
        attention_output = self.output(self_outputs[0], hidden_states)
        # 添加attention_output到outputs中，如果需要输出attention的话
        outputs = (attention_output,) + self_outputs[1:]
        # 返回结果
        return outputs


# 从transformers.models.bert.modeling_bert.BertIntermediate中复制代码
# 定义TapasIntermediate类，继承于nn.Module类
class TapasIntermediate(nn.Module):
    # 初始化方法，接受配置参数
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()
        # 创建一个线性层，输入大小为 config.hidden_size，输出大小为 config.intermediate_size
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        # 如果配置中隐藏激活函数是字符串类型，则使用预定义的 ACT2FN 字典中对应的激活函数
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            # 否则直接使用配置中的隐藏激活函数
            self.intermediate_act_fn = config.hidden_act

    # 前向传播方法，输入 hidden_states 为 torch.Tensor 类型，输出也为 torch.Tensor
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 将输入 hidden_states 输入到 self.dense 线性层中
        hidden_states = self.dense(hidden_states)
        # 使用 intermediate_act_fn 对隐藏状态进行激活
        hidden_states = self.intermediate_act_fn(hidden_states)
        # 返回处理后的隐藏状态
        return hidden_states
# 从transformers.models.bert.modeling_bert.BertOutput中复制而来的TapasOutput类
class TapasOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 初始化一个全连接层，将输入特征维度转换为隐藏层维度
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        # 初始化一个LayerNorm层，用于归一化隐藏状态
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 初始化一个Dropout层，用于随机置零部分神经元，防止过拟合
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 使用全连接层处理隐藏状态
        hidden_states = self.dense(hidden_states)
        # 对全连接层的输出进行Dropout操作
        hidden_states = self.dropout(hidden_states)
        # 对Dropout后的结果进行LayerNorm归一化，加上输入张量，得到最终输出
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class TapasLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 设定前向传播中用于分块的大小
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        # 序列长度维度
        self.seq_len_dim = 1
        # 初始化一个TapasAttention层
        self.attention = TapasAttention(config)
        # 是否为解码器
        self.is_decoder = config.is_decoder
        # 是否添加跨注意力
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            # 如果添加了跨注意力，且不是解码器模型，引发错误
            if not self.is_decoder:
                raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
            # 初始化一个TapasAttention层，用于跨注意力
            self.crossattention = TapasAttention(config)
        # 初始化一个TapasIntermediate层，用于中间层操作
        self.intermediate = TapasIntermediate(config)
        # 初始化一个TapasOutput层，用于输出
        self.output = TapasOutput(config)

    # 从transformers.models.bert.modeling_bert.BertLayer.forward中复制而来的前向传播函数
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
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        # 使用过去的键/值对，进行自注意力计算
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        attention_output = self_attention_outputs[0]

        # if decoder, the last output is tuple of self-attn cache
        # 如果是解码器，最后的输出是自注意力缓存的元组
        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        cross_attn_present_key_value = None
        if self.is_decoder and encoder_hidden_states is not None:
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers"
                    " by setting `config.add_cross_attention=True`"
                )

            # cross_attn cached key/values tuple is at positions 3,4 of past_key_value tuple
            # 交叉注意力缓存的键/值对元组在过去的键/值对元组的位置3,4处
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                cross_attn_past_key_value,
                output_attentions,
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:-1]  # add cross attentions if we output attention weights

            # add cross-attn cache to positions 3,4 of present_key_value tuple
            # 将交叉注意力缓存添加到现有的键/值对元组的位置3,4处
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs

        # if decoder, return the attn key/values as the last output
        # 如果是解码器，返回注意力键/值作为最后的输出
        if self.is_decoder:
            outputs = outputs + (present_key_value,)

        return outputs

    # Copied from transformers.models.bert.modeling_bert.BertLayer.feed_forward_chunk
    # 从transformers.models.bert.modeling_bert.BertLayer.feed_forward_chunk中复制过来
    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output
# 定义 TapasEncoder 类继承自 nn.Module
class TapasEncoder(nn.Module):
    # 初始化函数，接受一个 config 参数
    def __init__(self, config):
        # 调用父类的初始化函数
        super().__init__()
        # 将传入的 config 参数赋值给 self.config
        self.config = config
        # 使用列表推导式创建 nn.ModuleList，包含 config.num_hidden_layers 个 TapasLayer 对象
        self.layer = nn.ModuleList([TapasLayer(config) for _ in range(config.num_hidden_layers)])
        # 初始化梯度检查点变量为 False
        self.gradient_checkpointing = False

    # 前向传播函数
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
    ):
        # 如果需要输出隐藏状态，则初始化 all_hidden_states 为一个空元组，否则为 None
        all_hidden_states = () if output_hidden_states else None
        # 如果需要输出注意力权重，则初始化 all_attentions 为一个空元组，否则为 None
        all_attentions = () if output_attentions else None
        
        # 遍历 self.layer 中的 TapasLayer
        for i, layer_module in enumerate(self.layer):
            # 如果需要输出隐藏状态，则将当前的 hidden_states 加入 all_hidden_states
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 获取当前层的头部掩码
            layer_head_mask = head_mask[i] if head_mask is not None else None

            # 如果启用梯度检查点且处于训练模式
            if self.gradient_checkpointing and self.training:
                # 调用梯度检查点函数
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_values,
                    output_attentions,
                )
            else:
                # 调用当前层的前向传播函数
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_values,
                    output_attentions,
                )
            # 更新隐藏状态为当前层的输出
            hidden_states = layer_outputs[0]
            # 如果需要输出注意力权重，则将当前层的注意力权重加入 all_attentions
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # 如果需要输出隐藏状态，则将最终的隐藏状态加入 all_hidden_states
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 如果不需要返回字典形式的输出
        if not return_dict:
            # 返回需要输出的元组
            return tuple(v for v in [hidden_states, all_hidden_states, all_attentions] if v is not None)
        # 返回 BaseModelOutput 对象
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_attentions
        )


# 定义 TapasPooler 类，继承自 nn.Module
# 从 transformers.models.bert.modeling_bert.BertPooler 复制而来
class TapasPooler(nn.Module):
    # 初始化函数，接受一个 config 参数
    def __init__(self, config):
        # 调用父类的初始化函数
        super().__init__()
        # 创建一个线性层，输入和输出维度都为 config.hidden_size
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 创建一个 Tanh 激活函数
        self.activation = nn.Tanh()

    # 前向传播函数，接受隐藏状态张量作为输入，返回池化后的输出
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 通过取第一个 token 对应的隐藏状态，实现对模型的“池化”
        first_token_tensor = hidden_states[:, 0]
        # 将第一个 token 对应的隐藏状态通过线性层并激活函数得到最终的池化输出
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        # 返回池化输出
        return pooled_output
# 从transformers.models.bert.modeling_bert.BertPredictionHeadTransform复制并修改为TapasPredictionHeadTransform
class TapasPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 初始化一个全连接层，将输入的特征维度转换为相同的维度
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 如果隐藏激活函数是字符串，则从预定义的激活函数字典中获取对应的函数；否则直接使用给定的激活函数
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        # 初始化一个LayerNorm层，用于归一化隐藏状态的特征
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    # 正向传播函数
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 通过全连接层进行特征转换
        hidden_states = self.dense(hidden_states)
        # 应用激活函数对转换后的特征进行非线性变换
        hidden_states = self.transform_act_fn(hidden_states)
        # 对变换后的特征进行LayerNorm
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


# 从transformers.models.bert.modeling_bert.BertLMPredictionHead复制并修改为TapasLMPredictionHead
class TapasLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 初始化一个TapasPredictionHeadTransform层，用于特征转换
        self.transform = TapasPredictionHeadTransform(config)

        # 输出权重与输入嵌入相同，但是每个标记都有一个输出偏置
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # 初始化一个偏置参数，用于输出的偏置
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # 需要一个链接变量以确保偏置能够正确调整大小以适应`resize_token_embeddings`
        self.decoder.bias = self.bias

    # 正向传播函数
    def forward(self, hidden_states):
        # 对输入的隐藏状态进行特征转换
        hidden_states = self.transform(hidden_states)
        # 通过线性层进行预测
        hidden_states = self.decoder(hidden_states)
        return hidden_states


# 从transformers.models.bert.modeling_bert.BertOnlyMLMHead复制并修改为TapasOnlyMLMHead
class TapasOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 初始化一个TapasLMPredictionHead层，用于预测
        self.predictions = TapasLMPredictionHead(config)

    # 正向传播函数
    def forward(self, sequence_output: torch.Tensor) -> torch.Tensor:
        # 通过预测头进行预测
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


# TapasPreTrainedModel类，用于处理权重初始化以及下载和加载预训练模型的简单接口
class TapasPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 模型的配置类
    config_class = TapasConfig
    # 模型的基础名称前缀
    base_model_prefix = "tapas"
    # 是否支持梯度检查点
    supports_gradient_checkpointing = True

    # 从transformers.models.bert.modeling_bert.BertPreTrainedModel._init_weights复制
    def _init_weights(self, module):
        """Initialize the weights"""
        # 如果 module 是线性层 nn.Linear 类型
        if isinstance(module, nn.Linear):
            # 对权重进行初始化，使用正态分布，均值为 0，标准差为模型配置中的初始化范围 self.config.initializer_range
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果该线性层有偏置项，则将其初始化为零
            if module.bias is not None:
                module.bias.data.zero_()
        # 如果 module 是嵌入层 nn.Embedding 类型
        elif isinstance(module, nn.Embedding):
            # 对权重进行初始化，使用正态分布，均值为 0，标准差为模型配置中的初始化范围 self.config.initializer_range
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果该嵌入层有填充索引，则将填充索引位置的权重初始化为零
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        # 如果 module 是归一化层 nn.LayerNorm 类型
        elif isinstance(module, nn.LayerNorm):
            # 将归一化层的偏置项初始化为零
            module.bias.data.zero_()
            # 将归一化层的权重初始化为 1
            module.weight.data.fill_(1.0)
# 定义TAPAS_START_DOCSTRING字符串，用于说明Tapas模型的继承关系和参数说明
TAPAS_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its models (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`TapasConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

# 定义TAPAS_INPUTS_DOCSTRING字符串，用于说明Tapas模型的输入参数
TAPAS_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `({0})`):
            # 输入序列标记在词汇表中的索引。可以使用 [`AutoTokenizer`] 在 [`PreTrainedTokenizer.encode`] 和 [`PreTrainedTokenizer.__call__`] 中获取。
            # [什么是输入 ID？](../glossary#input-ids)
        attention_mask (`torch.FloatTensor` of shape `({0})`, *optional*):
            # 遮挡掩码，避免对填充标记索引执行注意力。选择在 `[0, 1]` 范围内的遮挡值：
            # - 1 代表 **未被遮挡** 的标记，
            # - 0 代表 **被遮挡** 的标记。
            # [什么是注意力遮挡？](../glossary#attention-mask)
        token_type_ids (`torch.LongTensor` of shape `({0}, 7)`, *optional*):
            # 编码表格结构的标记索引。可以使用 [`AutoTokenizer`] 获取。参考该类了解更多信息。
            # [什么是标记类型 ID？](../glossary#token-type-ids)
        position_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            # 输入序列标记在位置嵌入中的位置索引。如果 [`TapasConfig`] 的 `reset_position_index_per_cell` 设置为 `True`，将使用相对位置嵌入。选择在范围 `[0, config.max_position_embeddings - 1]` 内的值。
            # [什么是位置 ID？](../glossary#position-ids)
        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            # 用于将自注意力模块中的选定头部置零的掩码。选择的掩码值在 `[0, 1]` 范围内：- 1 表示头部 **未被遮挡**，- 0 表示头部 **被遮挡**。
        inputs_embeds (`torch.FloatTensor` of shape `({0}, hidden_size)`, *optional*):
            # 可选地，可以选择直接传递嵌入表示而不是传递 `input_ids`。如果要更好地控制如何将 `input_ids` 索引转换为相关向量，比模型的内部嵌入查找矩阵更有用。
        output_attentions (`bool`, *optional*):
            # 是否返回所有注意力层的注意力张量。有关更多详细信息，请查看返回张量中的 `attentions`。
        output_hidden_states (`bool`, *optional*):
            # 是否返回所有层的隐藏状态。有关更多详细信息，请查看返回张量中的 `hidden_states`。
        return_dict (`bool`, *optional*):
            # 是否返回 [`~utils.ModelOutput`] 而不是普通元组。
# 导入必要的库
@add_start_docstrings(
    "The bare Tapas Model transformer outputting raw hidden-states without any specific head on top.",
    TAPAS_START_DOCSTRING,
)
# 定义 TapasModel 类，继承自 TapasPreTrainedModel
class TapasModel(TapasPreTrainedModel):
    """
    This class is a small change compared to [`BertModel`], taking into account the additional token type ids.

    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in [Attention is
    all you need](https://arxiv.org/abs/1706.03762) by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
    Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

    """
    # 初始化方法，接受配置和是否添加汇聚层的参数
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        # 创建 TapasEmbeddings 对象
        self.embeddings = TapasEmbeddings(config)
        # 创建 TapasEncoder 对象
        self.encoder = TapasEncoder(config)

        # 如果需要添加汇聚层，则创建 TapasPooler 对象
        self.pooler = TapasPooler(config) if add_pooling_layer else None

        # 初始化权重并进行最终处理
        self.post_init()

    # 获取输入词嵌入
    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    # 设置输入词嵌入
    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    # 剪枝模型的头部
    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    # 前向传播方法
    @add_start_docstrings_to_model_forward(TAPAS_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=_CONFIG_FOR_DOC)
    def forward(
        # 输入参数
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
# 添加注释
@add_start_docstrings("""Tapas Model with a `language modeling` head on top.""", TAPAS_START_DOCSTRING)
# 定义 TapasForMaskedLM 类，继承自 TapasPreTrainedModel
class TapasForMaskedLM(TapasPreTrainedModel):
    _tied_weights_keys = ["cls.predictions.decoder.weight", "cls.predictions.decoder.bias"]
    # TapasForMaskedLM 的配置类
    config_class = TapasConfig
    # 基础模型名前缀
    base_model_prefix = "tapas"
    # 初始化方法，接受一个配置参数
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)

        # 初始化 Tapas 模型
        self.tapas = TapasModel(config, add_pooling_layer=False)
        # 初始化 Tapas 的 MLM 头部
        self.cls = TapasOnlyMLMHead(config)

        # 初始化权重并应用最终处理
        self.post_init()

    # 获取输出嵌入
    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    # 设置输出嵌入
    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    # 前向传播方法，接受多个输入参数
    @add_start_docstrings_to_model_forward(TAPAS_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=MaskedLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, MaskedLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional`):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`

        Returns:

        Examples:

        ```python
        >>> from transformers import AutoTokenizer, TapasForMaskedLM
        >>> import pandas as pd

        >>> tokenizer = AutoTokenizer.from_pretrained("google/tapas-base")
        >>> model = TapasForMaskedLM.from_pretrained("google/tapas-base")

        >>> data = {
        ...     "Actors": ["Brad Pitt", "Leonardo Di Caprio", "George Clooney"],
        ...     "Age": ["56", "45", "59"],
        ...     "Number of movies": ["87", "53", "69"],
        ... }
        >>> table = pd.DataFrame.from_dict(data)

        >>> inputs = tokenizer(
        ...     table=table, queries="How many [MASK] has George [MASK] played in?", return_tensors="pt"
        ... )
        >>> labels = tokenizer(
        ...     table=table, queries="How many movies has George Clooney played in?", return_tensors="pt"
        ... )["input_ids"]

        >>> outputs = model(**inputs, labels=labels)
        >>> logits = outputs.logits
        ```"""
        # 根据 return_dict 是否为 None 来决定是否使用配置中的 return_dict
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 通过 Tapas 模型获取输出
        outputs = self.tapas(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 从输出中提取序列输出
        sequence_output = outputs[0]
        # 通过分类层获取预测分数
        prediction_scores = self.cls(sequence_output)

        masked_lm_loss = None
        # 如果存在 labels，则计算 masked LM 损失
        if labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        # 如果 return_dict 为 False，则返回不同的输出格式
        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        # 返回 MaskedLMOutput 格式的输出
        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 使用 Tapas 模型进行表格问答任务，包含选择单元格头和可选的汇聚头（线性层在隐藏状态输出之上计算 'logits' 和可选 'logits_aggregation'），例如用于 SQA、WTQ 或 WikiSQL 监督任务
@add_start_docstrings(
    """
    Tapas Model with a cell selection head and optional aggregation head on top for question-answering tasks on tables
    (linear layers on top of the hidden-states output to compute `logits` and optional `logits_aggregation`), e.g. for
    SQA, WTQ or WikiSQL-supervised tasks.
    """,
    TAPAS_START_DOCSTRING,
)
class TapasForQuestionAnswering(TapasPreTrainedModel):
    # 初始化方法
    def __init__(self, config: TapasConfig):
        # 调用父类初始化方法
        super().__init__(config)

        # 基础模型
        self.tapas = TapasModel(config)

        # 丢弃率（仅在训练时使用）
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # 单元格选择头
        if config.init_cell_selection_weights_to_zero:
            # 是否将初始权重设置为0，以确保所有标记具有相同的先验概率
            self.output_weights = nn.Parameter(torch.zeros(config.hidden_size))
            self.column_output_weights = nn.Parameter(torch.zeros(config.hidden_size))
        else:
            self.output_weights = nn.Parameter(torch.empty(config.hidden_size))
            # 在原始实现中使用截断正态分布
            nn.init.normal_(
                self.output_weights, std=config.initializer_range
            )  
            self.column_output_weights = nn.Parameter(torch.empty(config.hidden_size))
            # 在原始实现中使用截断正态分布
            nn.init.normal_(
                self.column_output_weights, std=config.initializer_range
            )  
        self.output_bias = nn.Parameter(torch.zeros([]))
        self.column_output_bias = nn.Parameter(torch.zeros([]))

        # 汇聚头
        if config.num_aggregation_labels > 0:
            self.aggregation_classifier = nn.Linear(config.hidden_size, config.num_aggregation_labels)

        # 初始化权重并应用最终处理
        self.post_init()

    # 更新前向传播的文档字符串
    @add_start_docstrings_to_model_forward(TAPAS_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=TableQuestionAnsweringOutput, config_class=_CONFIG_FOR_DOC)
``` 
    # 定义一个名为 forward 的方法，用于向前传播模型
    def forward(
        # 输入的标识符张量，类型为长整型，可以为空
        input_ids: Optional[torch.LongTensor] = None,
        # 注意力掩码张量，类型为浮点数，可以为空
        attention_mask: Optional[torch.FloatTensor] = None,
        # 标记类型标识符张量，类型为长整型，可以为空
        token_type_ids: Optional[torch.LongTensor] = None,
        # 位置标识符张量，类型为长整型，可以为空
        position_ids: Optional[torch.LongTensor] = None,
        # 头掩码张量，类型为浮点数，可以为空
        head_mask: Optional[torch.FloatTensor] = None,
        # 输入嵌入张量，类型为浮点数，可以为空
        inputs_embeds: Optional[torch.FloatTensor] = None,
        # 表格掩码张量，类型为长整型，可以为空
        table_mask: Optional[torch.LongTensor] = None,
        # 标签张量，类型为长整型，可以为空
        labels: Optional[torch.LongTensor] = None,
        # 聚合标签张量，类型为长整型，可以为空
        aggregation_labels: Optional[torch.LongTensor] = None,
        # 浮点数答案张量，类型为浮点数，可以为空
        float_answer: Optional[torch.FloatTensor] = None,
        # 数值张量，类型为浮点数，可以为空
        numeric_values: Optional[torch.FloatTensor] = None,
        # 数值尺度张量，类型为浮点数，可以为空
        numeric_values_scale: Optional[torch.FloatTensor] = None,
        # 输出注意力张量的布尔值，可以为空
        output_attentions: Optional[bool] = None,
        # 输出隐藏状态的布尔值，可以为空
        output_hidden_states: Optional[bool] = None,
        # 返回字典的布尔值，可以为空
        return_dict: Optional[bool] = None,
# 使用自定义的文档字符串添加到 TapasForSequenceClassification 类
@add_start_docstrings(
    """
    Tapas Model with a sequence classification head on top (a linear layer on top of the pooled output), e.g. for table
    entailment tasks, such as TabFact (Chen et al., 2020).
    """,
    TAPAS_START_DOCSTRING,
)
# 创建 TapasForSequenceClassification 类，继承自 TapasPreTrainedModel
class TapasForSequenceClassification(TapasPreTrainedModel):
    # 初始化方法
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)
        # 将 config 中的 num_labels 赋值给 self.num_labels
        self.num_labels = config.num_labels

        # 初始化 TapasModel，dropout 和分类器
        self.tapas = TapasModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # 初始化权重并应用最终处理
        self.post_init()

    # 前向传播方法
    @add_start_docstrings_to_model_forward(TAPAS_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=SequenceClassifierOutput, config_class=_CONFIG_FOR_DOC)
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
        # TAPAS 实用工具。




# 平均逼近函数的枚举类型
class AverageApproximationFunction(str, enum.Enum):
    RATIO = "ratio"
    FIRST_ORDER = "first_order"
    SECOND_ORDER = "second_order"


# 分段张量的相关内容的开始


# 用于对张量的条目进行索引分组的 IndexMap 类
class IndexMap(object):
    """Index grouping entries within a tensor."""

    def __init__(self, indices, num_segments, batch_dims=0):
        """
        创建一个索引

        Args:
            indices (`torch.LongTensor`, same shape as a *values* Tensor to which the indices refer):
                Tensor containing the indices.
            num_segments (`torch.LongTensor`):
                Scalar tensor, the number of segments. All elements in a batched segmented tensor must have the same
                number of segments (although many segments can be empty).
            batch_dims (`int`, *optional*, defaults to 0):
                The number of batch dimensions. The first *batch_dims* dimensions of a SegmentedTensor are treated as
                batch dimensions. Segments in different batch elements are always distinct even if they have the same
                index.
        """
        self.indices = torch.as_tensor(indices)
        self.num_segments = torch.as_tensor(num_segments, device=indices.device)
        self.batch_dims = batch_dims

    # 返回索引的批次形状
    def batch_shape(self):
        return self.indices.size()[: self.batch_dims]  # returns a torch.Size object


# 两个索引的乘积的 ProductIndexMap 类
class ProductIndexMap(IndexMap):
    """The product of two indices."""
    def __init__(self, outer_index, inner_index):
        """
        Combines indices i and j into pairs (i, j). The result is an index where each segment (i, j) is the
        intersection of segments i and j. For example if the inputs represent table cells indexed by respectively rows
        and columns the output will be a table indexed by (row, column) pairs, i.e. by cell. The implementation
        combines indices {0, .., n - 1} and {0, .., m - 1} into {0, .., nm - 1}. The output has *num_segments* equal to
        *outer_index.num_segments* * *inner_index.num_segments*

        Args:
            outer_index (`IndexMap`):
                IndexMap.
            inner_index (`IndexMap`):
                IndexMap, must have the same shape as *outer_index*.
        """
        # 检查外部索引和内部索引的批次维度是否相同，若不同则引发错误
        if outer_index.batch_dims != inner_index.batch_dims:
            raise ValueError("outer_index.batch_dims and inner_index.batch_dims must be the same.")
        
        # 调用父类的构造函数，将两个索引结合成新的索引
        super().__init__(
            indices=(inner_index.indices + outer_index.indices * inner_index.num_segments),
            num_segments=inner_index.num_segments * outer_index.num_segments,
            batch_dims=inner_index.batch_dims,
        )
        # 保存外部索引和内部索引
        self.outer_index = outer_index
        self.inner_index = inner_index

    def project_outer(self, index):
        """Projects an index with the same index set onto the outer components."""
        # 投影出与外部组件相同的索引
        indices = torch.div(index.indices, self.inner_index.num_segments, rounding_mode="floor").type(torch.long)
        return IndexMap(indices=indices, num_segments=self.outer_index.num_segments, batch_dims=index.batch_dims)

    def project_inner(self, index):
        """Projects an index with the same index set onto the inner components."""
        # 投影出与内部组件相同的索引
        return IndexMap(
            indices=torch.fmod(index.indices, self.inner_index.num_segments)
            .type(torch.float)
            .floor()
            .type(torch.long),
            num_segments=self.inner_index.num_segments,
            batch_dims=index.batch_dims,
        )
# 定义 gather 函数，用于从 values 根据 index map 进行数据聚合
def gather(values, index, name="segmented_gather"):
    """
    Gathers from *values* using the index map. For each element in the domain of the index map this operation looks up
    a value for that index in *values*. Two elements from the same segment always get assigned the same value.

    Args:
        values (`torch.Tensor` of shape (B1, ..., Bn, num_segments, V1, ...)):
            Tensor with segment values.
        index (`IndexMap` of shape (B1, ..., Bn, I1, ..., Ik)):
            IndexMap.
        name (`str`, *optional*, defaults to 'segmented_gather'):
            Name for the operation. Currently not used

    Returns:
        `tuple(torch.Tensor)`: Tensor of shape (B1, ..., Bn, I1, ..., Ik, V1, ...) with the gathered values.
    """
    # 获取索引映射中的索引
    indices = index.indices
    # 首先，检查索引的维度是否是标量（即非向量化）
    if len(values.shape[index.batch_dims :]) < 2:
        # 如果是标量，则使用 torch.gather 函数根据索引聚合数据，并将维度转化为与索引相同
        return torch.gather(
            values,
            index.batch_dims,
            indices.view(
                values.size()[0], -1
            ),  # torch.gather expects index to have the same number of dimensions as values
        ).view(indices.size())
    else:
        # 如果是向量化，则需要调整索引的维度
        # 扩展索引的维度使其与 values 的维度一致，并通过 unsqueeze 函数在最后一维进行扩展操作
        indices = indices.unsqueeze(-1).expand(values.shape)
        # 使用 torch.gather 函数根据索引聚合数据，并返回结果
        return torch.gather(values, index.batch_dims, indices)


def flatten(index, name="segmented_flatten"):
    """
    Flattens a batched index map (which is typically of shape batch_size, seq_length) to a 1d index map. This operation
    relabels the segments to keep batch elements distinct. The k-th batch element will have indices shifted by
    *num_segments* * (k - 1). The result is a tensor with *num_segments* multiplied by the number of elements in the
    batch.

    Args:
        index (`IndexMap`):
            IndexMap to flatten.
        name (`str`, *optional*, defaults to 'segmented_flatten'):
            Name for the operation. Currently not used

    Returns:
        (`IndexMap`): The flattened IndexMap.
    """
    # 首先，使用 torch.prod 函数计算索引映射的批次大小 batch_size
    batch_size = torch.prod(torch.tensor(list(index.batch_shape())))
    # 计算偏移量 offset，其为长度为 batch_size 的一维张量，
    # 并通过元素逐个相乘乘以 num segments，以偏移批次中不同的元素，即如果 batch size 为 2，则为 [0, 64]
    offset = torch.arange(start=0, end=batch_size, device=index.num_segments.device) * index.num_segments
    # 重新调整 offset 的维度与索引一致
    offset = offset.view(index.batch_shape())
    # 使用 unsqueeze 函数在索引后面的维度上执行循环次数为索引批次维度到索引维度大小之间的循环
    for _ in range(index.batch_dims, len(index.indices.size())):  # typically range(1,2)
        offset = offset.unsqueeze(-1)

    # 使用偏移量 offset 与索引 index 进行按元素相加操作
    indices = offset + index.indices
    # 返回新的 IndexMap，其中包含了重新调整的索引，并更新 num segments 和 batch dims
    return IndexMap(indices=indices.view(-1), num_segments=index.num_segments * batch_size, batch_dims=0)


def range_index_map(batch_shape, num_segments, name="range_index_map"):
    """
    Constructs an index map equal to range(num_segments).
    """
    # 创建比索引大小为 range(num_segments) 的 IndexMap
    return IndexMap(
        indices=torch.arange(0, num_segments).unsqueeze(0).expand(batch_shape + [num_segments]),
        num_segments=num_segments,
        batch_dims=len(batch_shape),
    )
    Args:
        batch_shape (`torch.Size`):
            Batch shape
        num_segments (`int`):
            Number of segments
        name (`str`, *optional*, defaults to 'range_index_map'):
            Name for the operation. Currently not used

    Returns:
        (`IndexMap`): IndexMap of shape batch_shape with elements equal to range(num_segments).
    """
    # 将 batch_shape 转换为 long 类型的张量
    batch_shape = torch.as_tensor(
        batch_shape, dtype=torch.long
    )  # create a rank 1 tensor vector containing batch_shape (e.g. [2])
    # 确保 batch_shape 的维度为1
    assert len(batch_shape.size()) == 1
    # 将 num_segments 转换为张量
    num_segments = torch.as_tensor(num_segments)  # create a rank 0 tensor (scalar) containing num_segments (e.g. 64)
    # 确保 num_segments 为标量
    assert len(num_segments.size()) == 0

    # 生成 0 到 num_segments-1 的索引
    indices = torch.arange(
        start=0, end=num_segments, device=num_segments.device
    )  # create a rank 1 vector with num_segments elements
    # 创建新的张量，用于拼接成新的形状
    new_tensor = torch.cat(
        [torch.ones_like(batch_shape, dtype=torch.long, device=num_segments.device), num_segments.unsqueeze(dim=0)],
        dim=0,
    )
    # 将 new_tensor 转换为列表，并保存为 new_shape
    new_shape = [int(x) for x in new_tensor.tolist()]
    # 将 indices 重塑成新的形状
    indices = indices.view(new_shape)

    # 创建一个倍数张量，用于重复 indices
    multiples = torch.cat([batch_shape, torch.as_tensor([1])], dim=0)
    # 重复 indices，并保存为新的 indices
    indices = indices.repeat(multiples.tolist())
    # 返回包含索引、num_segments 和 batch_dims 的 IndexMap 对象
    return IndexMap(indices=indices, num_segments=num_segments, batch_dims=list(batch_shape.size())[0])
def _segment_reduce(values, index, segment_reduce_fn, name):
    # 定义内部函数，对分片进行规约
    """
    Applies a segment reduction segment-wise.

    Args:
        values (`torch.Tensor`):
            Tensor with segment values. 分片值的张量
        index (`IndexMap`):
            IndexMap. 索引映射
        segment_reduce_fn (`str`):
            Name for the reduce operation. One of "sum", "mean", "max" or "min".
            规约的名称，可以是 "sum", "mean", "max" 或 "min" 中的一个
        name (`str`):
            Name for the operation. Currently not used
            操作的名称，当前未使用

    Returns:
        (`IndexMap`): IndexMap of shape batch_shape with elements equal to range(num_segments).
        索引映射，其形状为批处理形状，其中的元素等于 segment 的数量
    """
    # Flatten the batch dimensions, as segments ops (scatter) do not support batching.
    # However if `values` has extra dimensions to the right keep them
    # unflattened. Segmented ops support vector-valued operations.
    # 压扁批处理维度，因为分段操作（scatter）不支持批处理。
    # 但是如果 `values` 的右边有额外的维度，保持它们是未压扁的。分段操作支持矢量值操作。
    flat_index = flatten(index)
    vector_shape = values.size()[len(index.indices.size()) :]  # torch.Size object
    flattened_shape = torch.cat(
        [torch.as_tensor([-1], dtype=torch.long), torch.as_tensor(vector_shape, dtype=torch.long)], dim=0
    )
    # changed "view" by "reshape" in the following line
    # 将 values 重塑成压扁的形状
    flat_values = values.reshape(flattened_shape.tolist())

    out = torch.zeros(int(flat_index.num_segments), dtype=torch.float, device=flat_values.device)
    segment_means = out.scatter_reduce(
        dim=0, index=flat_index.indices.long(), src=flat_values.float(), reduce=segment_reduce_fn, include_self=False
    )

    # Unflatten the values.
    new_shape = torch.cat(
        [
            torch.as_tensor(index.batch_shape(), dtype=torch.long),
            torch.as_tensor([index.num_segments], dtype=torch.long),
            torch.as_tensor(vector_shape, dtype=torch.long),
        ],
        dim=0,
    )

    output_values = segment_means.clone().view(new_shape.tolist()).to(values.dtype)
    output_index = range_index_map(index.batch_shape(), index.num_segments)
    return output_values, output_index


def reduce_sum(values, index, name="segmented_reduce_sum"):
    # 边界分割方式，对张量进行求和

    """
    Sums a tensor over its segments.

    Outputs 0 for empty segments.

    This operations computes the sum over segments, with support for:

        - Batching using the first dimensions [B1, B2, ..., Bn]. Each element in a batch can have different indices.
        - Vectorization using the last dimension [V1, V2, ...]. If they are present, the output will be a sum of
          vectors rather than scalars. Only the middle dimensions [I1, ..., Ik] are reduced by the operation.

    Args:
        values (`torch.Tensor` of shape [B1, B2, ..., Bn, I1, .., Ik, V1, V2, ..]):
            Tensor containing the values of which the sum must be taken segment-wise.
            包含了分段求和的张量值
        index (`IndexMap`, indices are of shape [B1, B2, ..., Bn, I1, .., Ik].):
            Index defining the segments. 确定分段的索引的位置
        name (`str`, *optional*, defaults to 'segmented_reduce_sum'):
            Name for the operation. Currently not used
            操作的名称，当前未使用
    # 返回值:
    # output_values (`torch.Tensor`of shape [B1, B2, ..., Bn, num_segments, V1, V2, ..]): 包含输出值的张量，形状为[B1, B2, ..., Bn, num_segments, V1, V2, ..]。
    # output_index (`IndexMap`): 具有形状[B1, B2, ..., Bn, num_segments]的IndexMap。
    # 使用_segment_reduce函数对values和index进行求和操作，并返回结果
    return _segment_reduce(values, index, "sum", name)
def reduce_mean(values, index, name="segmented_reduce_mean"):
    """
    Averages a tensor over its segments.

    Outputs 0 for empty segments.

    This operations computes the mean over segments, with support for:

        - Batching using the first dimensions [B1, B2, ..., Bn]. Each element in a batch can have different indices.
        - Vectorization using the last dimension [V1, V2, ...]. If they are present, the output will be a mean of
          vectors rather than scalars.

    Only the middle dimensions [I1, ..., Ik] are reduced by the operation.

    Args:
        values (`torch.Tensor` of shape [B1, B2, ..., Bn, I1, .., Ik, V1, V2, ..]):
            Tensor containing the values of which the mean must be taken segment-wise.
        index (`IndexMap`, indices are of shape [B1, B2, ..., Bn, I1, .., Ik].):
            Index defining the segments.
        name (`str`, *optional*, defaults to 'segmented_reduce_sum'):
            Name for the operation. Currently not used

    Returns:
        output_values (`torch.Tensor`of shape [B1, B2, ..., Bn, num_segments, V1, V2, ..]): Tensor containing the
        output values. output_index (`IndexMap`): IndexMap with shape [B1, B2, ..., Bn, num_segments].
    """
    return _segment_reduce(values, index, "mean", name)


def reduce_max(values, index, name="segmented_reduce_max"):
    """
    Computes the maximum over segments.

    This operation computes the maximum over segments, with support for:

        - Batching using the first dimensions [B1, B2, ..., Bn]. Each element in a batch can have different indices.
        - Vectorization using the last dimension [V1, V2, ...]. If they are present, the output will be an element-wise
          maximum of vectors rather than scalars.

    Only the middle dimensions [I1, ..., Ik] are reduced by the operation.

    Args:
        values (`torch.Tensor` of shape [B1, B2, ..., Bn, I1, .., Ik, V1, V2, ..]):
            Tensor containing the values of which the max must be taken segment-wise.
        index (`IndexMap`, indices are of shape [B1, B2, ..., Bn, I1, .., Ik].):
            Index defining the segments.
        name (`str`, *optional*, defaults to 'segmented_reduce_sum'):
            Name for the operation. Currently not used

    Returns:
        output_values (`torch.Tensor`of shape [B1, B2, ..., Bn, num_segments, V1, V2, ..]): Tensor containing the
        output values. output_index (`IndexMap`): IndexMap with shape [B1, B2, ..., Bn, num_segments].
    """
    return _segment_reduce(values, index, "amax", name)


def reduce_min(values, index, name="segmented_reduce_min"):
    """
    Computes the minimum over segments.
    """
    # 这个操作计算各个段的最小值，支持以下功能：
    
        - 使用前面的维度 [B1, B2, ..., Bn] 进行批处理。批中的每个元素可以具有不同的索引。
        - 使用最后一个维度 [V1, V2, ...] 进行向量化。如果存在这些维度，则输出将是向量而不是标量的逐元素最小值。
    
        该操作只会对中间的维度 [I1, ..., Ik] 进行缩减。
    
        Args:
            values (`torch.Tensor` 的形状为 [B1, B2, ..., Bn, I1, .., Ik, V1, V2, ..]):
                包含需要按段取最小值的值的张量。
            index (`IndexMap`，索引的形状为 [B1, B2, ..., Bn, I1, .., Ik]。):
                定义段的索引。
            name (`str`，*可选*，默认为 'segmented_reduce_sum')：
                操作的名称。目前未使用
    
        Returns:
            output_values (`torch.Tensor` 的形状为 [B1, B2, ..., Bn, num_segments, V1, V2, ..])：
                包含输出值的张量。
            output_index (`IndexMap`)：
                形状为 [B1, B2, ..., Bn, num_segments] 的 IndexMap。
    """
    # 调用内部函数 `_segment_reduce`，执行按段取最小值操作，传入参数为 values、index、"amin"，可选参数为 name，默认为 "segmented_reduce_sum"
    return _segment_reduce(values, index, "amin", name)
# End of everything related to segmented tensors


def compute_column_logits(
    sequence_output, column_output_weights, column_output_bias, cell_index, cell_mask, allow_empty_column_selection
):
    """
    Computes the column logits.

    Args:
        sequence_output (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Also known as last_hidden_state. Sequence of hidden-states at the output of the last layer of the model.
        column_output_weights (`torch.FloatTensor` of shape `(hidden_size)`):
            Weights of the linear layer for column selection.
        column_output_bias (`torch.FloatTensor` of shape `()`):
            Bias of the linear layer for column selection.
        cell_index (`ProductIndexMap`):
            Index that groups tokens into cells.
        cell_mask (`torch.FloatTensor` of shape `(batch_size, max_num_rows * max_num_cols)`):
            Mask for cells that exist in the table (i.e. that are not padding).
        allow_empty_column_selection (`bool`):
            Whether to allow not to select any column

    Returns:
        column_logits (`torch.FloatTensor`of shape `(batch_size, max_num_cols)`): Tensor containing the column logits
        for every example in the batch.
    """

    # First, compute the token logits (batch_size, seq_len) - without temperature
    token_logits = torch.einsum("bsj,j->bs", sequence_output, column_output_weights) + column_output_bias

    # Next, average the logits per cell (batch_size, max_num_cols*max_num_rows)
    cell_logits, cell_logits_index = reduce_mean(token_logits, cell_index)

    # Finally, average the logits per column (batch_size, max_num_cols)
    column_index = cell_index.project_inner(cell_logits_index)
    column_logits, out_index = reduce_sum(cell_logits * cell_mask, column_index)

    cell_count, _ = reduce_sum(cell_mask, column_index)
    column_logits /= cell_count + EPSILON_ZERO_DIVISION

    # Mask columns that do not appear in the example.
    is_padding = torch.logical_and(cell_count < 0.5, ~torch.eq(out_index.indices, 0))
    column_logits += CLOSE_ENOUGH_TO_LOG_ZERO * torch.as_tensor(
        is_padding, dtype=torch.float32, device=is_padding.device
    )

    if not allow_empty_column_selection:
        column_logits += CLOSE_ENOUGH_TO_LOG_ZERO * torch.as_tensor(
            torch.eq(out_index.indices, 0), dtype=torch.float32, device=out_index.indices.device
        )

    return column_logits


def _single_column_cell_selection_loss(token_logits, column_logits, labels, cell_index, col_index, cell_mask):
    """
    Computes the loss for cell selection constrained to a single column. The loss is a hierarchical log-likelihood. The
    model first predicts a column and then selects cells within that column (conditioned on the column). Cells outside
    the selected column are never selected.
    # 第一个部分：列损失

    # 首先找到我们应该选择的列。我们使用具有最多选定单元格的列。
    labels_per_column, _ = reduce_sum(torch.as_tensor(labels, dtype=torch.float32, device=labels.device), col_index)
    # labels_per_column的形状是(batch_size, max_num_cols)，它包含每个示例的每个列的标签数。
    column_label = torch.argmax(labels_per_column, dim=-1)  # 形状为(batch_size,)
    # 检查该列中是否没有选择的单元格。在这种情况下，模型应该预测特殊的列ID 0，表示“不选择任何内容”。
    no_cell_selected = torch.eq(
        torch.max(labels_per_column, dim=-1)[0], 0
    )  # no_cell_selected的形状是(batch_size,)且等于True，
    # 如果批次中的示例未选择单元格（即如果没有为该示例设置为1的标签）
    column_label = torch.where(
        no_cell_selected.view(column_label.size()), torch.zeros_like(column_label), column_label
    )

    column_dist = torch.distributions.Categorical(logits=column_logits)  # 形状为(batch_size, max_num_cols)
    column_loss_per_example = -column_dist.log_prob(column_label)

    # 第二个部分：单元格损失

    # 将标签和logits从每个标记减少到每个单元格。
    # logits_per_cell: 形状为(batch_size, max_num_rows*max_num_cols)，即(batch_size, 64*32)
    logits_per_cell, _ = reduce_mean(token_logits, cell_index)
    # labels_per_cell: 形状为(batch_size, 64*32)，指示每个单元格是否应该选择（1）或不选择（0）
    labels_per_cell, labels_index = reduce_max(
        torch.as_tensor(labels, dtype=torch.long, device=labels.device), cell_index
    )

    # 选定列的掩码。
    # column_id_for_cells: 形状为(batch_size, 64*32)，指示每个单元格属于哪一列
    # 从cell_index中获取与labels_index匹配的列标识，返回一个数组
    column_id_for_cells = cell_index.project_inner(labels_index).indices  
    
    # 创建一个column_mask张量，用于确定单元格是否属于要选择的列
    # 其形状为(batch_size, 64*32)，如果单元格属于要选择的列，则等于1
    column_mask = torch.as_tensor(
        torch.eq(column_id_for_cells, torch.unsqueeze(column_label, dim=-1)),
        dtype=torch.float32,
        device=cell_mask.device,
    )
    
    # 计算单元格的对数似然，但只针对所选列
    cell_dist = torch.distributions.Bernoulli(logits=logits_per_cell)  # 形状为(batch_size, 64*32)
    cell_log_prob = cell_dist.log_prob(labels_per_cell.type(torch.float32))  # 形状为(batch_size, 64*32)
    
    # 计算单元格损失，乘以column_mask和cell_mask
    cell_loss = -torch.sum(cell_log_prob * column_mask * cell_mask, dim=1)
    
    # 需要通过所选列中的单元格数量对损失进行标准化
    cell_loss /= torch.sum(column_mask * cell_mask, dim=1) + EPSILON_ZERO_DIVISION
    
    # 每个示例的选择损失等于列损失，再加上单元格损失
    selection_loss_per_example = column_loss_per_example
    selection_loss_per_example += torch.where(
        no_cell_selected.view(selection_loss_per_example.size()),
        torch.zeros_like(selection_loss_per_example),
        cell_loss,
    )
    
    # 将所选列以外的概率设置为0，确保与选择来自多个列的模型的后向兼容性
    selected_column_id = torch.as_tensor(
        torch.argmax(column_logits, dim=-1), dtype=torch.long, device=column_logits.device
    )  # 形状为(batch_size,)
    
    # 创建一个selected_column_mask张量，用于确定单元格是否属于模型选择的列
    # 其形状为(batch_size, 64*32)，如果单元格属于模型选择的列，则等于1
    selected_column_mask = torch.as_tensor(
        torch.eq(column_id_for_cells, torch.unsqueeze(selected_column_id, dim=-1)),
        dtype=torch.float32,
        device=selected_column_id.device,
    )
    
    # 永远不要选择具有特殊列id 0的单元格
    selected_column_mask = torch.where(
        torch.eq(column_id_for_cells, 0).view(selected_column_mask.size()),
        torch.zeros_like(selected_column_mask),
        selected_column_mask,
    )
    
    # 创建一个new_logits_per_cell张量，用于在所选列外附加一个小值，以避免对数值为0的对数
    new_logits_per_cell = logits_per_cell + CLOSE_ENOUGH_TO_LOG_ZERO * (1.0 - cell_mask * selected_column_mask)
    logits = gather(new_logits_per_cell, cell_index)
    
    # 返回选择损失和logits
    return selection_loss_per_example, logits
# 计算每个 token 的 logits
def compute_token_logits(sequence_output, temperature, output_weights, output_bias):
    """
    计算每个 token 的 logits
    
    参数:
        sequence_output (torch.FloatTensor of shape (batch_size, sequence_length, hidden_size)):
            也称为 last_hidden_state。模型最后一层的隐藏状态序列。
        temperature (float):
            伯努利分布的温度。
        output_weights (torch.FloatTensor of shape (hidden_size,)):
            用于单元选择的线性层的权重。
        output_bias (torch.FloatTensor of shape ()):
            用于单元选择的线性层的偏差。

    返回:
        logits (torch.FloatTensor of shape (batch_size, sequence_length)): 每个 token 的 logits。
    """
    # 计算 logits
    logits = (torch.einsum("bsj,j->bs", sequence_output, output_weights) + output_bias) / temperature

    return logits


# 计算聚合掩码
def _calculate_aggregate_mask(answer, pooled_output, cell_selection_preference, labels, aggregation_classifier):
    """
    确定哪些示例需要使用无聚合的方式选择单元。
    
    返回一个掩码,确定哪些示例应该直接从表中选择答案,而不使用任何聚合函数。如果答案是文本,那么这种情况很明确,因为聚合函数只适用于数字。如果答案是数字但不出现在表中,则必须使用某种聚合方式。模棱两可的情况是答案是一个出现在表中的数字。在这种情况下,我们使用模型预测的聚合函数概率来决定是选择还是聚合。这个阈值是一个超参数 cell_selection_preference。

    参数:
        answer (torch.FloatTensor of shape (batch_size,)):
            每个示例的答案。如果没有标量答案,则为 NaN。
        pooled_output (torch.FloatTensor of shape (batch_size, hidden_size)):
            编码器层顶部的池化器 (BertPooler) 的输出。
        cell_selection_preference (float):
            模棱两可情况下的单元选择偏好。
        labels (torch.LongTensor of shape (batch_size, sequence_length)):
            每个 token 的标签。
        aggregation_classifier (torch.nn.Linear): 聚合头

    返回:
        aggregate_mask (torch.FloatTensor of shape (batch_size,)): 对于应该使用聚合函数的示例设置为 1 的掩码。
    """
    # torch.FloatTensor(batch_size,)
    # 如果答案不是 NaN,则初始化 aggregate_mask 为 0
    aggregate_mask_init = torch.logical_not(torch.isnan(answer)).type(torch.FloatTensor).to(answer.device)
    # 计算聚合分类器的 logits
    logits_aggregation = aggregation_classifier(pooled_output)
    # 根据 logits 创建分类分布
    dist_aggregation = torch.distributions.categorical.Categorical(logits=logits_aggregation)
    # 索引 0 对应于"无聚合"
    # 计算除"无聚合"之外的聚合操作的总质量
    aggregation_ops_total_mass = torch.sum(dist_aggregation.probs[:, 1:], dim=1)
    # Cell selection examples according to current model.
    # 根据当前模型的预测结果，确定哪些样本的细胞被选中
    is_pred_cell_selection = aggregation_ops_total_mass <= cell_selection_preference
    
    # Examples with non-empty cell selection supervision.
    # 确定哪些样本具有可用的细胞选择监督信息
    is_cell_supervision_available = torch.sum(labels, dim=1) > 0
    
    # torch.where is not equivalent to tf.where (in tensorflow 1)
    # hence the added .view on the condition to match the shape of the first tensor
    # 使用 torch.where 函数根据条件选择 aggregate_mask_init 中的值
    # 条件为 is_pred_cell_selection 和 is_cell_supervision_available 的逻辑与
    # 结果tensor的shape与 aggregate_mask_init 保持一致
    aggregate_mask = torch.where(
        torch.logical_and(is_pred_cell_selection, is_cell_supervision_available).view(aggregate_mask_init.size()),
        torch.zeros_like(aggregate_mask_init, dtype=torch.float32),
        aggregate_mask_init,
    )
    
    # 将 aggregate_mask 设置为不可梯度
    aggregate_mask = aggregate_mask.detach()
    
    # 返回 aggregate_mask
    return aggregate_mask
def _calculate_aggregation_loss_known(
    logits_aggregation, aggregate_mask, aggregation_labels, use_answer_as_supervision, num_aggregation_labels
):
    """
    Calculates aggregation loss when its type is known during training.

    In the weakly supervised setting, the only known information is that for cell selection examples, "no aggregation"
    should be predicted. For other examples (those that require aggregation), no loss is accumulated. In the setting
    where aggregation type is always known, standard cross entropy loss is accumulated for all examples

    Args:
        logits_aggregation (`torch.FloatTensor` of shape `(batch_size, num_aggregation_labels)`):
            Logits per aggregation operation.
        aggregate_mask (`torch.FloatTensor` of shape `(batch_size, )`):
            A mask set to 1 for examples that should use aggregation functions.
        aggregation_labels (`torch.LongTensor` of shape `(batch_size, )`):
            Aggregation function id for every example in the batch.
        use_answer_as_supervision (`bool`, *optional*):
            Whether to use the answer as the only supervision for aggregation examples.
        num_aggregation_labels (`int`, *optional*, defaults to 0):
            The number of aggregation operators to predict.

    Returns:
        aggregation_loss_known (`torch.FloatTensor` of shape `(batch_size,)`): Aggregation loss (when its type is known
        during training) per example.
    """
    if use_answer_as_supervision:
        # Prepare "no aggregation" targets for cell selection examples.
        target_aggregation = torch.zeros_like(aggregate_mask, dtype=torch.long)
    else:
        # Use aggregation supervision as the target.
        target_aggregation = aggregation_labels

    one_hot_labels = nn.functional.one_hot(target_aggregation, num_classes=num_aggregation_labels).type(torch.float32)
    log_probs = nn.functional.log_softmax(logits_aggregation, dim=-1)

    # torch.FloatTensor[batch_size]
    per_example_aggregation_intermediate = -torch.sum(one_hot_labels * log_probs, dim=-1)
    if use_answer_as_supervision:
        # Accumulate loss only for examples requiring cell selection
        # (no aggregation).
        return per_example_aggregation_intermediate * (1 - aggregate_mask)
    else:
        return per_example_aggregation_intermediate


def _calculate_aggregation_loss_unknown(logits_aggregation, aggregate_mask):
    """
    Calculates aggregation loss in the case of answer supervision.

    Args:
        logits_aggregation (`torch.FloatTensor` of shape `(batch_size, num_aggregation_labels)`):
            Logits per aggregation operation.
        aggregate_mask (`torch.FloatTensor` of shape `(batch_size, )`):
            A mask set to 1 for examples that should use aggregation functions

    Returns:
        aggregation_loss_unknown (`torch.FloatTensor` of shape `(batch_size,)`): Aggregation loss (in case of answer
        supervision) per example.
    """
    # 创建一个分类分布对象，根据给定的 logits 参数
    dist_aggregation = torch.distributions.categorical.Categorical(logits=logits_aggregation)
    # 计算除 "no aggregation" 外其他所有聚合函数的总质量
    aggregation_ops_total_mass = torch.sum(dist_aggregation.probs[:, 1:], dim=1)
    # 在需要聚合的答案情况下预测一些聚合操作
    # 这会增加所有聚合函数的概率，类似于最大边际似然（MML），但不考虑函数是否给出正确答案
    # 返回对数损失乘以聚合掩码的负值
    return -torch.log(aggregation_ops_total_mass) * aggregate_mask
def _calculate_aggregation_loss(
    logits_aggregation,
    aggregate_mask,
    aggregation_labels,
    use_answer_as_supervision,
    num_aggregation_labels,
    aggregation_loss_weight,
):
    """
    Calculates the aggregation loss per example.

    Args:
        logits_aggregation (`torch.FloatTensor` of shape `(batch_size, num_aggregation_labels)`):
            Logits per aggregation operation.
        aggregate_mask (`torch.FloatTensor` of shape `(batch_size, )`):
            A mask set to 1 for examples that should use aggregation functions.
        aggregation_labels (`torch.LongTensor` of shape `(batch_size, )`):
            Aggregation function id for every example in the batch.
        use_answer_as_supervision (`bool`, *optional*):
            Whether to use the answer as the only supervision for aggregation examples.
        num_aggregation_labels (`int`, *optional*, defaults to 0):
            The number of aggregation operators to predict.
        aggregation_loss_weight (`float`, *optional*, defaults to 1.0):
            Importance weight for the aggregation loss.

    Returns:
        aggregation_loss (`torch.FloatTensor` of shape `(batch_size,)`): Aggregation loss per example.
    """
    # Calculate aggregation loss for known aggregation functions
    per_example_aggregation_loss = _calculate_aggregation_loss_known(
        logits_aggregation, aggregate_mask, aggregation_labels, use_answer_as_supervision, num_aggregation_labels
    )

    if use_answer_as_supervision:
        # Add aggregation loss for numeric answers that need aggregation
        per_example_aggregation_loss += _calculate_aggregation_loss_unknown(logits_aggregation, aggregate_mask)
    return aggregation_loss_weight * per_example_aggregation_loss


def _calculate_expected_result(
    dist_per_cell, numeric_values, numeric_values_scale, input_mask_float, logits_aggregation, config
):
    """
    Calculates the expected result given cell and aggregation probabilities.

    Args:
        dist_per_cell (`torch.distributions.Bernoulli`):
            Cell selection distribution for each cell.
        numeric_values (`torch.FloatTensor` of shape `(batch_size, seq_length)`):
            Numeric values of every token. Nan for tokens which are not numeric values.
        numeric_values_scale (`torch.FloatTensor` of shape `(batch_size, seq_length)`):
            Scale of the numeric values of every token.
        input_mask_float (`torch.FloatTensor` of shape `(batch_size, seq_length)`):
            Mask for the table, without question tokens and table headers.
        logits_aggregation (`torch.FloatTensor` of shape `(batch_size, num_aggregation_labels)`):
            Logits per aggregation operation.
        config ([`TapasConfig`]):
            Model configuration class with all the hyperparameters of the model

    Returns:
        expected_result (`torch.FloatTensor` of shape `(batch_size,)`): The expected result per example.
    """
    # 如果配置中使用 Gumbel 分布来计算单元格概率
    if config.use_gumbel_for_cells:
        # 创建一个 RelaxedBernoulli 分布，用来采样
        gumbel_dist = torch.distributions.RelaxedBernoulli(
            # 注意：标记的logits已经被温度除过，用于计算单元格选择错误，所以这里需要再乘回去
            temperature=config.temperature,
            logits=dist_per_cell.logits * config.temperature,
        )
        # 从 Gumbel 分布中采样得到的概率
        scaled_probability_per_cell = gumbel_dist.sample()
    else:
        # 如果不使用 Gumbel 分布，则直接使用原始的概率
        scaled_probability_per_cell = dist_per_cell.probs

    # 缩放每个单元格的概率，然后和输入掩码相乘
    scaled_probability_per_cell = (scaled_probability_per_cell / numeric_values_scale) * input_mask_float
    # 对每个样本计算概率之和
    count_result = torch.sum(scaled_probability_per_cell, dim=1)
    # 创建一个只包含零的张量，用于遮盖非数值的表格数值
    numeric_values_masked = torch.where(
        torch.isnan(numeric_values), torch.zeros_like(numeric_values), numeric_values
    )  # 将非数值的表格数值遮盖为零
    # 对每个样本计算加权数值之和
    sum_result = torch.sum(scaled_probability_per_cell * numeric_values_masked, dim=1)
    # 获取平均值的近似方法
    avg_approximation = config.average_approximation_function
    if avg_approximation == AverageApproximationFunction.RATIO:
        # 计算比率近似的平均值
        average_result = sum_result / (count_result + EPSILON_ZERO_DIVISION)
    elif avg_approximation == AverageApproximationFunction.FIRST_ORDER:
        # 计算一阶近似的平均值
        # 期望值ex，表示除了对应其它单元格外所有概率的和
        ex = torch.sum(scaled_probability_per_cell, dim=1, keepdim=True) - scaled_probability_per_cell + 1
        # 计算基于ex的平均值
        average_result = torch.sum(numeric_values_masked * scaled_probability_per_cell / ex, dim=1)
    elif avg_approximation == AverageApproximationFunction.SECOND_ORDER:
        # 计算二阶近似的平均值
        # 期望值ex，表示除了对应其它单元格外所有概率的和
        ex = torch.sum(scaled_probability_per_cell, dim=1, keepdim=True) - scaled_probability_per_cell + 1
        # 计算点乘的方差
        pointwise_var = scaled_probability_per_cell * (1 - scaled_probability_per_cell)
        var = torch.sum(pointwise_var, dim=1, keepdim=True) - pointwise_var
        # 计算一个乘数，用于计算加权平均值
        multiplier = (var / torch.square(ex) + 1) / ex
        # 计算基于乘数的平均值
        average_result = torch.sum(numeric_values_masked * scaled_probability_per_cell * multiplier, dim=1)
    else:
        # 抛出异常，说明平均值近似方法无效
        raise ValueError(f"Invalid average_approximation_function: {config.average_approximation_function}")

    # 如果配置中使用 Gumbel 分布来进行聚合
    if config.use_gumbel_for_aggregation:
        # 创建一个 RelaxedOneHotCategorical 分布，用来采样
        gumbel_dist = torch.distributions.RelaxedOneHotCategorical(
            config.aggregation_temperature, logits=logits_aggregation[:, 1:]
        )
        # 从 Gumbel 分布中采样得到的聚合操作的概率
        aggregation_op_only_probs = gumbel_dist.sample()
    else:
        # 如果不是第一个条件，执行以下操作
        # 计算聚合操作标签之外的概率，使用softmax函数对logits_aggregation[:, 1:]进行处理
        # 维度为 [batch_size, num_aggregation_labels - 1]
        aggregation_op_only_probs = nn.functional.softmax(
            logits_aggregation[:, 1:] / config.aggregation_temperature, dim=-1
        )

    # 将三个结果合并成一个张量
    all_results = torch.cat(
        [
            # 对sum_result在维度1上增加一个维度
            torch.unsqueeze(sum_result, dim=1),
            # 对average_result在维度1上增加一个维度
            torch.unsqueeze(average_result, dim=1),
            # 对count_result在维度1上增加一个维度
            torch.unsqueeze(count_result, dim=1),
        ],
        # 在维度1上进行拼接
        dim=1,
    )

    # 通过加权求和的方式计算预期结果
    expected_result = torch.sum(all_results * aggregation_op_only_probs, dim=1)
    # 返回预期结果
    return expected_result
# PyTorch 目前不支持具有自定义 delta 的 Huber 损失函数，因此我们自己定义它
def huber_loss(input, target, delta: float = 1.0):
    # 计算输入和目标之间的绝对误差，形状为 (batch_size,)
    errors = torch.abs(input - target)  
    # 返回 Huber 损失函数的计算结果，如果误差小于 delta，则采用 0.5 * errors**2，否则采用 errors * delta - (0.5 * delta**2)
    return torch.where(errors < delta, 0.5 * errors**2, errors * delta - (0.5 * delta**2))


def _calculate_regression_loss(
    answer,
    aggregate_mask,
    dist_per_cell,
    numeric_values,
    numeric_values_scale,
    input_mask_float,
    logits_aggregation,
    config,
):
    """
    计算每个样本的回归损失。

    Args:
        answer (`torch.FloatTensor` of shape `(batch_size,)`):
            批次中每个样本的答案。如果没有标量答案，则为 NaN。
        aggregate_mask (`torch.FloatTensor` of shape `(batch_size,)`):
            对应于应使用聚合函数的示例的掩码，设为 1。
        dist_per_cell (`torch.distributions.Bernoulli`):
            每个单元格的单元格选择分布。
        numeric_values (`torch.FloatTensor` of shape `(batch_size, seq_length)`):
            每个标记的数字值。对于不是数字值的标记，为 NaN。
        numeric_values_scale (`torch.FloatTensor` of shape `(batch_size, seq_length)`):
            每个标记的数字值的规模。
        input_mask_float (`torch.FloatTensor` of shape `(batch_size, seq_length)`):
            表格的掩码，不包括问题标记和表头。
        logits_aggregation (`torch.FloatTensor` of shape `(batch_size, num_aggregation_labels)`):
            每个聚合操作的 logits。
        config ([`TapasConfig`]):
            具有模型所有参数的模型配置类。

    Returns:
        per_example_answer_loss_scaled (`torch.FloatTensor` of shape `(batch_size,)`): 批次中每个样本的答案损失，缩放后。
        large_answer_loss_mask (`torch.FloatTensor` of shape `(batch_size,)`): 对于答案损失大于 answer_loss_cutoff 的示例，为 1 的掩码。
    """
    # 计算期望结果，形状为 float32 (batch_size,)
    expected_result = _calculate_expected_result(
        dist_per_cell, numeric_values, numeric_values_scale, input_mask_float, logits_aggregation, config
    )

    # 答案掩码，如果为 NaN，则替换为 0，形状为 float32 (batch_size,)
    answer_masked = torch.where(torch.isnan(answer), torch.zeros_like(answer), answer)

    if config.use_normalized_answer_loss:
        # 正规化因子，避免零除错误
        normalizer = (torch.max(torch.abs(expected_result), torch.abs(answer_masked)) + EPSILON_ZERO_DIVISION).detach()

        # 标准化后的答案和期望结果
        normalized_answer_masked = answer_masked / normalizer
        normalized_expected_result = expected_result / normalizer
        # 使用标准化后的值计算 Huber 损失
        per_example_answer_loss = huber_loss(
            normalized_expected_result * aggregate_mask, normalized_answer_masked * aggregate_mask
        )
    else:
        # 使用配置中指定的 delta 或默认值计算 Huber 损失
        per_example_answer_loss = huber_loss(
            expected_result * aggregate_mask, answer_masked * aggregate_mask, delta=config.huber_loss_delta
        )
    # 如果配置中未设置答案损失的截断值，则创建一个与每个示例的答案损失形状相同的张量，全部填充为1
    if config.answer_loss_cutoff is None:
        large_answer_loss_mask = torch.ones_like(per_example_answer_loss, dtype=torch.float32)

    # 如果配置中设置了答案损失的截断值
    else:
        # 使用 torch.where 函数根据答案损失是否大于答案损失截断值来创建一个大答案损失掩码
        # 大于答案损失截断值的位置为0，否则为1
        large_answer_loss_mask = torch.where(
            per_example_answer_loss > config.answer_loss_cutoff,
            torch.zeros_like(per_example_answer_loss, dtype=torch.float32),
            torch.ones_like(per_example_answer_loss, dtype=torch.float32),
        )
    
    # 对每个示例的答案损失进行缩放，并与聚合掩码相乘，得到每个示例的答案损失的缩放值
    per_example_answer_loss_scaled = config.answer_loss_importance * (per_example_answer_loss * aggregate_mask)
    
    # 返回每个示例的答案损失的缩放值以及大答案损失掩码
    return per_example_answer_loss_scaled, large_answer_loss_mask
```