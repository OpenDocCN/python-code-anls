# `.\models\tapas\modeling_tapas.py`

```py
# coding=utf-8
# 声明版权和许可信息，该文件使用 Apache License, Version 2.0 授权
# 详细许可信息可在 http://www.apache.org/licenses/LICENSE-2.0 获取
#
# 如果没有适用法律要求或书面同意，本软件按 "原样" 提供，不提供任何形式的保证或条件
# 详见许可文件以了解更多信息。

"""PyTorch TAPAS model."""
# 导入必要的库和模块
import enum
import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

# 导入 TAPAS 模型相关的组件和功能
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling, MaskedLMOutput, SequenceClassifierOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import (
    apply_chunking_to_forward,
    find_pruneable_heads_and_indices,
    is_torch_greater_or_equal_than_1_12,
    prune_linear_layer,
)
from ...utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
# 导入 TAPAS 的配置文件
from .configuration_tapas import TapasConfig

# 获取日志记录器
logger = logging.get_logger(__name__)

# 检查 torch 版本是否大于等于 1.12.0，如果不是则发出警告
if not is_torch_greater_or_equal_than_1_12:
    logger.warning(
        f"You are using torch=={torch.__version__}, but torch>=1.12.0 is required to use "
        "TapasModel. Please upgrade torch."
    )

# 以下是为文档和模型下载提供的配置和检查点信息
_CONFIG_FOR_DOC = "TapasConfig"
_CHECKPOINT_FOR_DOC = "google/tapas-base"

# 定义预训练的 TAPAS 模型存档列表，包括大、基础、小、迷你和微型模型
TAPAS_PRETRAINED_MODEL_ARCHIVE_LIST = [
    # large models
    "google/tapas-large",
    "google/tapas-large-finetuned-sqa",
    "google/tapas-large-finetuned-wtq",
    "google/tapas-large-finetuned-wikisql-supervised",
    "google/tapas-large-finetuned-tabfact",
    # base models
    "google/tapas-base",
    "google/tapas-base-finetuned-sqa",
    "google/tapas-base-finetuned-wtq",
    "google/tapas-base-finetuned-wikisql-supervised",
    "google/tapas-base-finetuned-tabfact",
    # small models
    "google/tapas-small",
    "google/tapas-small-finetuned-sqa",
    "google/tapas-small-finetuned-wtq",
    "google/tapas-small-finetuned-wikisql-supervised",
    "google/tapas-small-finetuned-tabfact",
    # mini models
    "google/tapas-mini",
    "google/tapas-mini-finetuned-sqa",
    "google/tapas-mini-finetuned-wtq",
    "google/tapas-mini-finetuned-wikisql-supervised",
    "google/tapas-mini-finetuned-tabfact",
    # tiny models
    "google/tapas-tiny",
    "google/tapas-tiny-finetuned-sqa",
    "google/tapas-tiny-finetuned-wtq",
    "google/tapas-tiny-finetuned-wikisql-supervised",
    "google/tapas-tiny-finetuned-tabfact",
    # 查看所有 TAPAS 模型：https://huggingface.co/models?filter=tapas
]
# 设置一个小值，用于避免零除错误
EPSILON_ZERO_DIVISION = 1e-10
# 定义一个足够接近负无穷大的值，用于表示对数零
CLOSE_ENOUGH_TO_LOG_ZERO = -10000.0


@dataclass
class TableQuestionAnsweringOutput(ModelOutput):
    """
    [`TapasForQuestionAnswering`] 的输出类型。

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` (and possibly `answer`, `aggregation_labels`, `numeric_values` and `numeric_values_scale` are provided)):
            总损失，包括层次单元选择对数似然损失和（可选）半监督回归损失，以及（可选）聚合操作的监督损失。
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
            单元选择头部的预测分数，每个标记的分数。
        logits_aggregation (`torch.FloatTensor`, *optional*, of shape `(batch_size, num_aggregation_labels)`):
            聚合头部的预测分数，每种聚合运算符的分数。
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            模型在每一层输出的隐藏状态，包括每一层的输出和初始嵌入输出。
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            self-attention 头部中的注意力权重，用于计算加权平均值。

    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    logits_aggregation: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


def load_tf_weights_in_tapas(model, config, tf_checkpoint_path):
    """
    将 TensorFlow 的检查点加载到 PyTorch 模型中。这是从 load_tf_weights_in_bert 改编而来的函数。

    - 添加单元选择和聚合头部
    - 考虑额外的标记类型嵌入层
    """
    try:
        import re  # 导入正则表达式模块
        import numpy as np  # 导入 NumPy 库
        import tensorflow as tf  # 导入 TensorFlow 库
    except ImportError:
        logger.error(
            "Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions."
        )
        raise
    tf_path = os.path.abspath(tf_checkpoint_path)  # 获取 TensorFlow 检查点文件的绝对路径
    logger.info(f"Converting TensorFlow checkpoint from {tf_path}")  # 记录日志，显示正在从 TensorFlow 检查点进行转换
    # 从 TF 模型中加载权重
    init_vars = tf.train.list_variables(tf_path)
    names = []  # 初始化空列表，用于存储变量名
    arrays = []  # 初始化空列表，用于存储变量值
    # 对于每个初始化变量的名称和形状，在日志中记录加载 TensorFlow 权重的信息
    for name, shape in init_vars:
        # 使用 logger.info 输出日志信息，显示正在加载的 TensorFlow 权重的名称和形状
        logger.info(f"Loading TF weight {name} with shape {shape}")
        # 使用 TensorFlow 的 API 从指定路径 tf_path 中加载变量 name 的值，并将其存入 array
        array = tf.train.load_variable(tf_path, name)
        # 将当前加载的变量名 name 添加到 names 列表中
        names.append(name)
        # 将加载的权重数组 array 添加到 arrays 列表中
        arrays.append(array)
    
    # 返回已加载了初始变量的模型
    return model
# TapasEmbeddings 类定义，用于构建从单词、位置和标记类型嵌入生成的嵌入层。
class TapasEmbeddings(nn.Module):
    """
    Construct the embeddings from word, position and token_type embeddings. Same as BertEmbeddings but with a number of
    additional token type embeddings to encode tabular structure.
    """

    def __init__(self, config):
        # 调用父类构造函数初始化模块
        super().__init__()
        
        # 不包括 config.disabled_features 和 config.disable_position_embeddings 这两个原始实现中的特性
        # 单词嵌入层，vocab_size 是词汇表大小，hidden_size 是嵌入的维度，padding_idx 是填充标记的索引
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        
        # 位置嵌入层，max_position_embeddings 是最大位置编码数量，hidden_size 是嵌入的维度
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        
        # 标记类型嵌入层，根据配置中的 type_vocab_sizes 列表创建多个嵌入层
        for i, type_vocab_sizes in enumerate(config.type_vocab_sizes):
            name = f"token_type_embeddings_{i}"
            # 动态设置模块属性，创建名为 token_type_embeddings_i 的嵌入层
            setattr(self, name, nn.Embedding(type_vocab_sizes, config.hidden_size))

        # 记录标记类型嵌入层的数量
        self.number_of_token_type_embeddings = len(config.type_vocab_sizes)

        # LayerNorm 不使用蛇形命名以保持与 TensorFlow 模型变量名一致，可以加载任何 TensorFlow 检查点文件
        # LayerNorm 层，将隐藏层的输出进行归一化，eps 是用于数值稳定性的小值
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # Dropout 层，用于在训练过程中随机断开神经元连接，防止过拟合
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # 存储配置信息，方便模型加载和保存时使用
        self.config = config
    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None):
        # 如果传入了 input_ids 参数，则获取其形状；否则获取 inputs_embeds 的形状去掉最后一个维度
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        # 获取序列长度
        seq_length = input_shape[1]
        # 获取输入数据所在设备
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # 如果未提供 position_ids 参数，则创建绝对位置嵌入
        if position_ids is None:
            # 创建长度为 seq_length 的长整型张量，设备为指定设备
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
            # 将位置张量扩展到与输入形状相同
            position_ids = position_ids.unsqueeze(0).expand(input_shape)
            
            # 当 self.config.reset_position_index_per_cell 设置为 True 时，创建相对位置嵌入
            if self.config.reset_position_index_per_cell:
                # 获取 token_type_ids 的第二个维度作为列索引，形状为 (batch_size, seq_len)
                col_index = IndexMap(token_type_ids[:, :, 1], self.config.type_vocab_sizes[1], batch_dims=1)
                # 获取 token_type_ids 的第三个维度作为行索引，形状为 (batch_size, seq_len)
                row_index = IndexMap(token_type_ids[:, :, 2], self.config.type_vocab_sizes[2], batch_dims=1)
                # 计算全局索引，形状为 (batch_size, seq_len)
                full_index = ProductIndexMap(col_index, row_index)
                # 计算每个单元格的第一个绝对位置，形状为 (max_rows * max_columns,)
                first_position_per_segment = reduce_min(position_ids, full_index)[0]
                # 获取每个 token 的单元格的第一个绝对位置，形状为 (batch_size, seq_len)
                first_position = gather(first_position_per_segment, full_index)
                # 创建长度为 seq_length 的长整型张量，设备为指定设备，表示相对位置
                position = torch.arange(seq_length, dtype=torch.long, device=device).unsqueeze(0)
                # 计算相对位置并限制在最大位置嵌入之内
                position_ids = torch.min(
                    torch.as_tensor(self.config.max_position_embeddings - 1, device=device), position - first_position
                )

        # 如果未提供 token_type_ids 参数，则创建全零张量作为 token_type_ids
        if token_type_ids is None:
            token_type_ids = torch.zeros(
                (input_shape + self.number_of_token_type_embeddings), dtype=torch.long, device=device
            )

        # 如果未提供 inputs_embeds 参数，则使用 input_ids 获取词嵌入
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        # 获取位置嵌入
        position_embeddings = self.position_embeddings(position_ids)

        # 计算总的嵌入向量
        embeddings = inputs_embeds + position_embeddings

        # 添加每个 token 类型的嵌入向量
        for i in range(self.number_of_token_type_embeddings):
            name = f"token_type_embeddings_{i}"
            embeddings += getattr(self, name)(token_type_ids[:, :, i])

        # 应用 LayerNormalization
        embeddings = self.LayerNorm(embeddings)
        # 应用 Dropout
        embeddings = self.dropout(embeddings)
        # 返回结果嵌入向量
        return embeddings
# 定义自注意力机制的类，继承自 nn.Module
class TapasSelfAttention(nn.Module):
    # 初始化函数，接受一个配置对象 config
    def __init__(self, config):
        super().__init__()
        # 检查隐藏层大小是否能被注意力头数整除，若不能且配置对象没有嵌入大小属性，则抛出异常
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size {config.hidden_size} is not a multiple of the number of attention "
                f"heads {config.num_attention_heads}"
            )

        # 设置注意力头数和每个注意力头的大小
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # 创建用于查询、键和值的线性映射层
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        # 使用配置中的注意力概率丢弃率创建 dropout 层
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        # 标记是否为解码器（用于 Transformer 模型）
        self.is_decoder = config.is_decoder

    # 将输入张量转换为注意力分数矩阵的形状
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    # 前向传播函数，接受隐藏状态和其他可选参数
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
        # 从隐藏状态生成查询层
        mixed_query_layer = self.query(hidden_states)

        # 如果这是一个跨注意力模块，keys和values来自一个编码器；
        # 注意力遮罩应确保编码器的填充标记不被注意到。
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # 重复使用过去的键值和跨注意力
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            # 从编码器隐藏状态生成keys和values，并转置以供得分计算
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            # 从隐藏状态生成keys和values，并与过去的键值连接起来
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            # 从隐藏状态生成keys和values
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        # 将mixed_query_layer转置以供得分计算
        query_layer = self.transpose_for_scores(mixed_query_layer)

        if self.is_decoder:
            # 如果是解码器，保存当前键值对
            past_key_value = (key_layer, value_layer)

        # 计算"query"和"key"之间的点积，得到原始注意力分数
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # 应用注意力遮罩（预先计算好的，用于TapasModel的所有层的forward()函数）
            attention_scores = attention_scores + attention_mask

        # 将注意力分数归一化为概率
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # 使用dropout随机丢弃整个token的注意力权重，这在原始Transformer论文中有描述
        attention_probs = self.dropout(attention_probs)

        # 如果需要，对注意力头进行掩码处理
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        # 计算上下文向量，通过注意力概率和values的加权和
        context_layer = torch.matmul(attention_probs, value_layer)

        # 调整上下文向量的形状以适应后续处理
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        # 准备输出结果，包括上下文向量和可能的注意力权重
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs
# Copied from transformers.models.bert.modeling_bert.BertSelfOutput
class TapasSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 初始化一个全连接层，输入和输出维度都是config.hidden_size
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 初始化一个LayerNorm层，对输入进行归一化，eps是归一化过程中的稳定性参数
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 初始化一个Dropout层，用于随机置0输入张量的元素以防止过拟合
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 对输入的hidden_states进行全连接操作
        hidden_states = self.dense(hidden_states)
        # 对全连接后的结果进行dropout操作
        hidden_states = self.dropout(hidden_states)
        # 对dropout后的结果和输入input_tensor进行残差连接并归一化
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class TapasAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 初始化自注意力层和输出层
        self.self = TapasSelfAttention(config)
        self.output = TapasSelfOutput(config)
        # 初始化一个集合用于存储需要剪枝的注意力头
        self.pruned_heads = set()

    # Copied from transformers.models.bert.modeling_bert.BertAttention.prune_heads
    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        # 调用剪枝函数，找到需要剪枝的头部并返回索引
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # 对自注意力层的query、key、value部分进行线性层剪枝
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        # 对输出层的全连接层进行剪枝
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # 更新超参数并记录剪枝的头部
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    # Copied from transformers.models.bert.modeling_bert.BertAttention.forward
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
        # 调用自注意力层的forward方法
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        # 将自注意力层的输出作为输入，调用输出层的forward方法
        attention_output = self.output(self_outputs[0], hidden_states)
        # 如果需要输出注意力信息，则将其添加到输出中
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


# Copied from transformers.models.bert.modeling_bert.BertIntermediate
class TapasIntermediate(nn.Module):
    # 初始化方法，接受一个配置对象作为参数
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()
        # 创建一个线性层，输入维度为 config.hidden_size，输出维度为 config.intermediate_size
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        
        # 检查 config.hidden_act 是否为字符串类型，如果是则从预定义的映射 ACT2FN 中获取对应的激活函数，赋值给 self.intermediate_act_fn
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            # 如果 config.hidden_act 不是字符串类型，则直接将其赋值给 self.intermediate_act_fn
            self.intermediate_act_fn = config.hidden_act

    # 前向传播方法，接受一个 torch.Tensor 类型的隐藏状态作为输入，返回一个 torch.Tensor 类型的隐藏状态
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 将输入的隐藏状态经过线性层 self.dense 处理，得到新的隐藏状态
        hidden_states = self.dense(hidden_states)
        # 将经过线性层处理后的隐藏状态再经过激活函数 self.intermediate_act_fn 处理，得到最终的隐藏状态
        hidden_states = self.intermediate_act_fn(hidden_states)
        # 返回最终的隐藏状态
        return hidden_states
# Copied from transformers.models.bert.modeling_bert.BertOutput
# 定义一个名为 TapasOutput 的类，继承自 nn.Module
class TapasOutput(nn.Module):
    # 初始化方法，接收一个 config 对象作为参数
    def __init__(self, config):
        super().__init__()
        # 创建一个全连接层，输入大小为 config.intermediate_size，输出大小为 config.hidden_size
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        # 创建一个 LayerNorm 层，输入大小为 config.hidden_size，eps 为 config.layer_norm_eps
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 创建一个 Dropout 层，丢弃概率为 config.hidden_dropout_prob
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    # 前向传播方法，接收 hidden_states 和 input_tensor 作为输入，返回 torch.Tensor
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 将 hidden_states 输入到全连接层 dense 中
        hidden_states = self.dense(hidden_states)
        # 对全连接层的输出进行 Dropout
        hidden_states = self.dropout(hidden_states)
        # 将 Dropout 后的输出与 input_tensor 相加，并输入 LayerNorm 层中
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        # 返回 LayerNorm 层的输出作为最终的隐藏状态输出
        return hidden_states


# 定义一个名为 TapasLayer 的类，继承自 nn.Module
class TapasLayer(nn.Module):
    # 初始化方法，接收一个 config 对象作为参数
    def __init__(self, config):
        super().__init__()
        # 设置 chunk_size_feed_forward 属性为 config.chunk_size_feed_forward
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        # 设置 seq_len_dim 属性为 1
        self.seq_len_dim = 1
        # 创建一个 TapasAttention 对象，并传入 config 对象
        self.attention = TapasAttention(config)
        # 设置 is_decoder 属性为 config.is_decoder
        self.is_decoder = config.is_decoder
        # 设置 add_cross_attention 属性为 config.add_cross_attention
        self.add_cross_attention = config.add_cross_attention
        # 如果 add_cross_attention 为 True，则执行以下操作
        if self.add_cross_attention:
            # 如果不是 decoder 模型，则抛出 ValueError 异常
            if not self.is_decoder:
                raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
            # 创建一个 TapasAttention 对象作为 crossattention，并传入 config 对象
            self.crossattention = TapasAttention(config)
        # 创建一个 TapasIntermediate 对象，并传入 config 对象
        self.intermediate = TapasIntermediate(config)
        # 创建一个 TapasOutput 对象，并传入 config 对象
        self.output = TapasOutput(config)

    # Copied from transformers.models.bert.modeling_bert.BertLayer.forward
    # 前向传播方法，接收多个输入参数，返回 torch.Tensor
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
        # Perform self-attention on the input hidden states with optional cached key/values
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        # Extract the attention output from self-attention outputs
        attention_output = self_attention_outputs[0]

        # if decoder, the last output is tuple of self-attn cache
        if self.is_decoder:
            # Extract all outputs except the first (attention_output) and the last (present_key_value)
            outputs = self_attention_outputs[1:-1]
            # Extract the present key/value tuple from self-attention outputs
            present_key_value = self_attention_outputs[-1]
        else:
            # Include self attentions if attention weights are output
            outputs = self_attention_outputs[1:]

        cross_attn_present_key_value = None
        if self.is_decoder and encoder_hidden_states is not None:
            if not hasattr(self, "crossattention"):
                # Raise an error if cross-attention is expected but not instantiated
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers"
                    " by setting `config.add_cross_attention=True`"
                )

            # cross_attn cached key/values tuple is at positions 3,4 of past_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            # Perform cross-attention using self-attention output and encoder hidden states
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                cross_attn_past_key_value,
                output_attentions,
            )
            # Extract the attention output from cross-attention outputs
            attention_output = cross_attention_outputs[0]
            # Add cross attentions to the existing outputs
            outputs = outputs + cross_attention_outputs[1:-1]

            # Add cross-attn cache to positions 3,4 of present_key_value tuple
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        # Apply feed-forward chunking to the attention output
        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        # Append layer_output to outputs
        outputs = (layer_output,) + outputs

        # if decoder, return the attn key/values as the last output
        if self.is_decoder:
            # Append present_key_value to outputs as the last element
            outputs = outputs + (present_key_value,)

        # Return the final outputs
        return outputs

    # Copied from transformers.models.bert.modeling_bert.BertLayer.feed_forward_chunk
    def feed_forward_chunk(self, attention_output):
        # Apply intermediate layer to attention_output
        intermediate_output = self.intermediate(attention_output)
        # Apply output layer to intermediate_output and attention_output
        layer_output = self.output(intermediate_output, attention_output)
        # Return the final layer output
        return layer_output
# 定义一个名为 TapasEncoder 的神经网络模块类，继承自 nn.Module
class TapasEncoder(nn.Module):
    # 初始化方法，接受一个 config 参数
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()
        # 将传入的 config 参数保存为类的一个属性
        self.config = config
        # 创建一个由多个 TapasLayer 组成的层列表，列表长度为 config.num_hidden_layers
        self.layer = nn.ModuleList([TapasLayer(config) for _ in range(config.num_hidden_layers)])
        # 设置梯度检查点标志为 False
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
    ):
        # 如果设置了输出隐藏状态，则初始化一个空元组用于存储所有隐藏状态
        all_hidden_states = () if output_hidden_states else None
        # 如果设置了输出注意力权重，则初始化一个空元组用于存储所有注意力权重
        all_attentions = () if output_attentions else None
        
        # 遍历每个层模块
        for i, layer_module in enumerate(self.layer):
            # 如果需要输出隐藏状态，则将当前的隐藏状态添加到 all_hidden_states 中
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 获取当前层对应的头部掩码（如果有的话）
            layer_head_mask = head_mask[i] if head_mask is not None else None

            # 如果启用梯度检查点且正在训练阶段，则使用梯度检查点功能进行前向传播计算
            if self.gradient_checkpointing and self.training:
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
                # 否则直接调用当前层的前向传播方法进行计算
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_values,
                    output_attentions,
                )
            
            # 更新隐藏状态为当前层的输出的第一个元素
            hidden_states = layer_outputs[0]
            
            # 如果需要输出注意力权重，则将当前层的注意力权重添加到 all_attentions 中
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)
        
        # 如果需要输出隐藏状态，则将最终的隐藏状态添加到 all_hidden_states 中
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        
        # 如果 return_dict 参数为 False，则返回一个包含所有非空结果的元组
        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_attentions] if v is not None)
        
        # 否则，返回一个 BaseModelOutput 对象，包含最终的隐藏状态、所有隐藏状态和所有注意力权重
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_attentions
        )


# 从 transformers.models.bert.modeling_bert.BertPooler 复制而来的 TapasPooler 类
class TapasPooler(nn.Module):
    # 初始化方法，接受一个 config 参数
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()
        # 创建一个线性层，输入维度和输出维度均为 config.hidden_size
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 创建一个 Tanh 激活函数
        self.activation = nn.Tanh()

    # 前向传播方法，接收一个名为 hidden_states 的输入张量
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 取出隐藏状态张量的第一个 token 对应的向量
        first_token_tensor = hidden_states[:, 0]
        # 将该向量输入全连接层进行线性变换
        pooled_output = self.dense(first_token_tensor)
        # 将线性变换的结果输入 Tanh 激活函数
        pooled_output = self.activation(pooled_output)
        # 返回池化后的输出张量
        return pooled_output
# Copied from transformers.models.bert.modeling_bert.BertPredictionHeadTransform with Bert->Tapas
class TapasPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 定义一个全连接层，输入和输出维度都是config.hidden_size
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 根据配置选择激活函数，如果是字符串则从预定义映射表中获取对应函数，否则直接使用配置中的函数
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        # Layer normalization 层，输入维度为config.hidden_size，设置epsilon为config.layer_norm_eps
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 输入hidden_states经过全连接层dense，输出维度不变
        hidden_states = self.dense(hidden_states)
        # 应用激活函数transform_act_fn到全连接层输出上
        hidden_states = self.transform_act_fn(hidden_states)
        # 对输出进行Layer normalization
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


# Copied from transformers.models.bert.modeling_bert.BertLMPredictionHead with Bert->Tapas
class TapasLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 使用TapasPredictionHeadTransform对应的配置初始化transform属性
        self.transform = TapasPredictionHeadTransform(config)

        # 输出权重与输入embedding相同，但每个token有一个单独的偏置
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # 定义一个偏置参数，维度为config.vocab_size
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # 需要链接两个变量以便偏置在resize_token_embeddings时被正确调整大小
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        # 输入hidden_states经过transform层处理
        hidden_states = self.transform(hidden_states)
        # 经过线性层decoder，输出维度为config.vocab_size
        hidden_states = self.decoder(hidden_states)
        return hidden_states


# Copied from transformers.models.bert.modeling_bert.BertOnlyMLMHead with Bert->Tapas
class TapasOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 使用TapasLMPredictionHead对应的配置初始化predictions属性
        self.predictions = TapasLMPredictionHead(config)

    def forward(self, sequence_output: torch.Tensor) -> torch.Tensor:
        # 输入sequence_output经过predictions层处理，得到预测分数
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class TapasPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 使用TapasConfig作为配置类
    config_class = TapasConfig
    # base_model_prefix为模型前缀字符串
    base_model_prefix = "tapas"
    # 支持梯度检查点
    supports_gradient_checkpointing = True

    # Copied from transformers.models.bert.modeling_bert.BertPreTrainedModel._init_weights
    # 初始化神经网络模块的权重
    def _init_weights(self, module):
        """Initialize the weights"""
        # 如果模块是线性层（全连接层）
        if isinstance(module, nn.Linear):
            # 使用正态分布随机初始化权重，均值为0，标准差为self.config.initializer_range
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果存在偏置项，将其初始化为零
            if module.bias is not None:
                module.bias.data.zero_()
        # 如果模块是嵌入层
        elif isinstance(module, nn.Embedding):
            # 使用正态分布随机初始化权重，均值为0，标准差为self.config.initializer_range
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果定义了填充索引，将填充索引对应的权重初始化为零
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        # 如果模块是层归一化层
        elif isinstance(module, nn.LayerNorm):
            # 初始化偏置项为零
            module.bias.data.zero_()
            # 初始化权重为全1
            module.weight.data.fill_(1.0)
# TAPAS_START_DOCSTRING 的文档字符串，用于说明 TapasModel 类的继承关系和用法
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



# TAPAS_INPUTS_DOCSTRING 的文档字符串，暂未提供具体内容
TAPAS_INPUTS_DOCSTRING = r"""
    
"""
    Args:
        input_ids (`torch.LongTensor` of shape `({0})`):
            # 输入序列的标记索引，在词汇表中的索引。可以使用 `AutoTokenizer` 获取这些索引。参见 `PreTrainedTokenizer.encode` 和 `PreTrainedTokenizer.__call__` 获取更多详情。
            # [什么是输入 ID？](../glossary#input-ids)
        attention_mask (`torch.FloatTensor` of shape `({0})`, *optional*):
            # 遮罩，用于在填充标记索引上避免执行注意力操作。遮罩的值选择在 `[0, 1]`：

            # - 1 表示那些**未被遮罩**的标记，
            # - 0 表示那些**被遮罩**的标记。

            # [什么是注意力遮罩？](../glossary#attention-mask)
        token_type_ids (`torch.LongTensor` of shape `({0}, 7)`, *optional*):
            # 编码表格结构的标记索引。可以使用 `AutoTokenizer` 获取这些索引。参见该类获取更多信息。

            # [什么是标记类型 ID？](../glossary#token-type-ids)
        position_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            # 每个输入序列标记在位置嵌入中的位置索引。如果 `TapasConfig` 的 `reset_position_index_per_cell` 设置为 `True`，将使用相对位置嵌入。选择范围在 `[0, config.max_position_embeddings - 1]`。

            # [什么是位置 ID？](../glossary#position-ids)
        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            # 用于置零自注意力模块中选择的头部的遮罩。遮罩的值选择在 `[0, 1]`：

            # - 1 表示该头部**未被遮罩**，
            # - 0 表示该头部**被遮罩**。
        inputs_embeds (`torch.FloatTensor` of shape `({0}, hidden_size)`, *optional*):
            # 可选地，您可以选择直接传递嵌入表示而不是传递 `input_ids`。如果您希望更精确地控制如何将 `input_ids` 索引转换为相关联的向量，这将非常有用，胜过模型内部的嵌入查找矩阵。
        output_attentions (`bool`, *optional*):
            # 是否返回所有注意力层的注意力张量。更多细节请参见返回张量中的 `attentions`。
        output_hidden_states (`bool`, *optional*):
            # 是否返回所有层的隐藏状态。更多细节请参见返回张量中的 `hidden_states`。
        return_dict (`bool`, *optional*):
            # 是否返回 [`~utils.ModelOutput`] 而不是普通的元组。
"""
@add_start_docstrings(
    "The bare Tapas Model transformer outputting raw hidden-states without any specific head on top.",
    TAPAS_START_DOCSTRING,
)
class TapasModel(TapasPreTrainedModel):
    """
    This class defines the Tapas Model, which extends TapasPreTrainedModel.

    It can function as an encoder (self-attention only) or a decoder, incorporating cross-attention layers between
    self-attention layers, following the architecture in the paper "Attention is All You Need" by Ashish Vaswani et al.

    """

    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        # Initialize TapasEmbeddings and TapasEncoder based on provided configuration
        self.embeddings = TapasEmbeddings(config)
        self.encoder = TapasEncoder(config)

        # Optionally initialize TapasPooler for pooling layer
        self.pooler = TapasPooler(config) if add_pooling_layer else None

        # Perform any additional initialization tasks
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        # Iterates over specified layers and prunes heads accordingly
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(TAPAS_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=_CONFIG_FOR_DOC)
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
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    # 初始化函数，用于创建一个新的TapasOnlyMLMHead对象，并初始化权重
    def __init__(self, config):
        # 调用父类的初始化方法，传入配置参数
        super().__init__(config)

        # 创建一个TapasModel对象，不添加池化层
        self.tapas = TapasModel(config, add_pooling_layer=False)
        # 创建一个TapasOnlyMLMHead对象
        self.cls = TapasOnlyMLMHead(config)

        # 调用后续处理函数，初始化权重并进行最终处理
        self.post_init()

    # 返回MLM头部的输出嵌入
    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    # 设置MLM头部的输出嵌入为新的嵌入
    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    # 重写的前向传播函数，接受多个输入参数，并返回MaskedLMOutput对象
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
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`

        Returns:
            Depending on `return_dict`:
            - If `return_dict` is `False`, returns a tuple with `prediction_scores` followed by additional outputs.
            - If `return_dict` is `True`, returns a `MaskedLMOutput` object containing `loss`, `logits`, `hidden_states`, and `attentions`.

        Examples:

        ```
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
        ```

        Determines the return type based on `return_dict`. If `labels` are provided, computes the masked language modeling loss using `CrossEntropyLoss`.
        Returns either a tuple or a `MaskedLMOutput` object depending on `return_dict`.

        ```
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        ```

        Passes input arguments to the Tapas model and retrieves the outputs, including sequence output and prediction scores.

        ```
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
        ```

        Retrieves the sequence output from the Tapas model's outputs and computes prediction scores using a classifier layer.

        ```
        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)
        ```

        If `labels` are provided, calculates the masked language modeling loss using `CrossEntropyLoss`.

        ```
        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
        ```

        Constructs the output based on whether `return_dict` is `False`, returning a tuple of outputs or including `masked_lm_loss` in a `MaskedLMOutput` object.

        ```
        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        ```
"""
Tapas Model with a cell selection head and optional aggregation head on top for question-answering tasks on tables
(linear layers on top of the hidden-states output to compute `logits` and optional `logits_aggregation`), e.g. for
SQA, WTQ or WikiSQL-supervised tasks.
"""
# 使用 TapasStartDocstring 和 TAPAS_START_DOCSTRING 定义的文档字符串来注释 TapasForQuestionAnswering 类
@add_start_docstrings(
    """
    Tapas Model with a cell selection head and optional aggregation head on top for question-answering tasks on tables
    (linear layers on top of the hidden-states output to compute `logits` and optional `logits_aggregation`), e.g. for
    SQA, WTQ or WikiSQL-supervised tasks.
    """,
    TAPAS_START_DOCSTRING,
)
class TapasForQuestionAnswering(TapasPreTrainedModel):
    
    def __init__(self, config: TapasConfig):
        super().__init__(config)
        
        # base model
        self.tapas = TapasModel(config)
        
        # dropout (only used when training)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # cell selection heads
        if config.init_cell_selection_weights_to_zero:
            # init_cell_selection_weights_to_zero: Whether the initial weights should be
            # set to 0. This ensures that all tokens have the same prior probability.
            self.output_weights = nn.Parameter(torch.zeros(config.hidden_size))
            self.column_output_weights = nn.Parameter(torch.zeros(config.hidden_size))
        else:
            self.output_weights = nn.Parameter(torch.empty(config.hidden_size))
            nn.init.normal_(
                self.output_weights, std=config.initializer_range
            )  # here, a truncated normal is used in the original implementation
            self.column_output_weights = nn.Parameter(torch.empty(config.hidden_size))
            nn.init.normal_(
                self.column_output_weights, std=config.initializer_range
            )  # here, a truncated normal is used in the original implementation
        
        self.output_bias = nn.Parameter(torch.zeros([]))
        self.column_output_bias = nn.Parameter(torch.zeros([]))
        
        # aggregation head
        if config.num_aggregation_labels > 0:
            self.aggregation_classifier = nn.Linear(config.hidden_size, config.num_aggregation_labels)
        
        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(TAPAS_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=TableQuestionAnsweringOutput, config_class=_CONFIG_FOR_DOC)
    # 定义一个方法，用于模型的前向传播
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,  # 输入的 token IDs，类型为可选的长整型张量
        attention_mask: Optional[torch.FloatTensor] = None,  # 注意力掩码，类型为可选的浮点数张量
        token_type_ids: Optional[torch.LongTensor] = None,  # token 类型 IDs，类型为可选的长整型张量
        position_ids: Optional[torch.LongTensor] = None,  # 位置 IDs，类型为可选的长整型张量
        head_mask: Optional[torch.FloatTensor] = None,  # 头部掩码，类型为可选的浮点数张量
        inputs_embeds: Optional[torch.FloatTensor] = None,  # 输入的嵌入表示，类型为可选的浮点数张量
        table_mask: Optional[torch.LongTensor] = None,  # 表格掩码，类型为可选的长整型张量
        labels: Optional[torch.LongTensor] = None,  # 标签，类型为可选的长整型张量
        aggregation_labels: Optional[torch.LongTensor] = None,  # 聚合标签，类型为可选的长整型张量
        float_answer: Optional[torch.FloatTensor] = None,  # 浮点型答案，类型为可选的浮点数张量
        numeric_values: Optional[torch.FloatTensor] = None,  # 数值，类型为可选的浮点数张量
        numeric_values_scale: Optional[torch.FloatTensor] = None,  # 数值的比例，类型为可选的浮点数张量
        output_attentions: Optional[bool] = None,  # 是否输出注意力信息，类型为可选的布尔值
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态，类型为可选的布尔值
        return_dict: Optional[bool] = None,  # 是否返回字典形式的输出，类型为可选的布尔值
"""
Tapas Model with a sequence classification head on top (a linear layer on top of the pooled output), e.g. for table
entailment tasks, such as TabFact (Chen et al., 2020).
"""
@add_start_docstrings(
    TAPAS_START_DOCSTRING,
)
class TapasForSequenceClassification(TapasPreTrainedModel):
    def __init__(self, config):
        """
        Initializes TapasForSequenceClassification model.

        Args:
            config (TapasConfig): Configuration object specifying the model architecture and parameters.
        """
        super().__init__(config)
        self.num_labels = config.num_labels

        # Initialize Tapas model
        self.tapas = TapasModel(config)
        # Dropout layer
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # Linear layer for classification
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

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
    ):
        """
        Forward pass of the TapasForSequenceClassification model.

        Args:
            input_ids (torch.LongTensor, optional): Input IDs of the sequence.
            attention_mask (torch.FloatTensor, optional): Mask to avoid performing attention on padding token indices.
            token_type_ids (torch.LongTensor, optional): Segment token indices.
            position_ids (torch.LongTensor, optional): Indices of positions of each input sequence tokens in the model.
            head_mask (torch.FloatTensor, optional): Mask to nullify selected heads of the self-attention modules.
            inputs_embeds (torch.FloatTensor, optional): Embedded representations of input sequences.
            labels (torch.LongTensor, optional): Labels for computing the sequence classification loss.
            output_attentions (bool, optional): Whether to return attentions weights.
            output_hidden_states (bool, optional): Whether to return hidden states.
            return_dict (bool, optional): Whether to return a dictionary as output.

        Returns:
            SequenceClassifierOutput: Output of the sequence classification, including loss, logits, and optional hidden states and attentions.
        """
        """ TAPAS utilities."""


class AverageApproximationFunction(str, enum.Enum):
    """
    Enum defining average approximation functions.

    Includes:
    - RATIO: ratio approximation
    - FIRST_ORDER: first order approximation
    - SECOND_ORDER: second order approximation
    """
    RATIO = "ratio"
    FIRST_ORDER = "first_order"
    SECOND_ORDER = "second_order"


# Beginning of everything related to segmented tensors


class IndexMap(object):
    """
    Index grouping entries within a tensor.

    Attributes:
        indices (torch.LongTensor): Tensor containing the indices.
        num_segments (torch.LongTensor): Scalar tensor specifying the number of segments.
        batch_dims (int): Number of batch dimensions.
    """

    def __init__(self, indices, num_segments, batch_dims=0):
        """
        Creates an IndexMap instance.

        Args:
            indices (torch.LongTensor): Tensor containing the indices.
            num_segments (torch.LongTensor): Scalar tensor specifying the number of segments.
            batch_dims (int, optional): Number of batch dimensions. Defaults to 0.
        """
        self.indices = torch.as_tensor(indices)
        self.num_segments = torch.as_tensor(num_segments, device=indices.device)
        self.batch_dims = batch_dims

    def batch_shape(self):
        """
        Returns the batch shape of the indices tensor.

        Returns:
            torch.Size: Size object representing the shape of the indices tensor up to batch dimensions.
        """
        return self.indices.size()[: self.batch_dims]


class ProductIndexMap(IndexMap):
    """
    Index map representing the product of two indices.
    """
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
        # 检查两个索引对象的批处理维度是否相同
        if outer_index.batch_dims != inner_index.batch_dims:
            raise ValueError("outer_index.batch_dims and inner_index.batch_dims must be the same.")

        # 调用父类的构造函数来初始化对象
        super().__init__(
            indices=(inner_index.indices + outer_index.indices * inner_index.num_segments),
            num_segments=inner_index.num_segments * outer_index.num_segments,
            batch_dims=inner_index.batch_dims,
        )
        # 存储外部索引对象和内部索引对象
        self.outer_index = outer_index
        self.inner_index = inner_index

    def project_outer(self, index):
        """Projects an index with the same index set onto the outer components."""
        # 将索引映射到外部组件
        indices = torch.div(index.indices, self.inner_index.num_segments, rounding_mode="floor").type(torch.long)
        return IndexMap(indices=indices, num_segments=self.outer_index.num_segments, batch_dims=index.batch_dims)

    def project_inner(self, index):
        """Projects an index with the same index set onto the inner components."""
        # 将索引映射到内部组件
        return IndexMap(
            indices=torch.fmod(index.indices, self.inner_index.num_segments)
            .type(torch.float)
            .floor()
            .type(torch.long),
            num_segments=self.inner_index.num_segments,
            batch_dims=index.batch_dims,
        )
# 构造一个索引映射，其值为范围内的连续整数，用于表示段的编号
def range_index_map(batch_shape, num_segments, name="range_index_map"):
    """
    Constructs an index map equal to range(num_segments).

    Args:
        batch_shape (tuple): Shape of the batch dimensions.
        num_segments (int): Number of segments.
        name (str, optional): Name for the operation. Currently not used.

    Returns:
        IndexMap: An IndexMap object containing the constructed indices.
    """
    # 计算批量大小作为标量张量
    batch_size = torch.prod(torch.tensor(list(batch_shape)))
    # 创建偏移量作为长度为批量大小的一维张量，
    # 逐元素与段数相乘（以偏移批次中的不同元素）例如，如果批量大小为2：[0, 64]
    offset = torch.arange(start=0, end=batch_size, device='cpu') * num_segments
    offset = offset.view(batch_shape)
    # 根据索引映射的维度数范围（通常是range(1,2)）多次展开偏移量
    for _ in range(1, len(batch_shape)):  # 通常范围为(1, 2)
        offset = offset.unsqueeze(-1)

    # 计算最终的索引，为偏移量加上原始索引
    indices = offset + torch.arange(num_segments, device='cpu').view(-1)
    return IndexMap(indices=indices.view(-1), num_segments=num_segments * batch_size, batch_dims=0)
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
    # 将 batch_shape 转换为 long 类型的张量，创建一个包含 batch_shape 的一维张量（例如 [2]）
    batch_shape = torch.as_tensor(
        batch_shape, dtype=torch.long
    )  # create a rank 1 tensor vector containing batch_shape (e.g. [2])
    assert len(batch_shape.size()) == 1  # 断言 batch_shape 的维度为 1

    # 将 num_segments 转换为张量，创建一个包含 num_segments 的标量张量（例如 64）
    num_segments = torch.as_tensor(num_segments)  # create a rank 0 tensor (scalar) containing num_segments (e.g. 64)
    assert len(num_segments.size()) == 0  # 断言 num_segments 的维度为 0，即为标量

    # 创建一个从 0 到 num_segments-1 的张量，设备与 num_segments 相同
    indices = torch.arange(
        start=0, end=num_segments, device=num_segments.device
    )  # create a rank 1 vector with num_segments elements

    # 创建一个新的张量，形状为 [1, num_segments]，其中第一个元素为 1，其余维度与 batch_shape 相同
    new_tensor = torch.cat(
        [torch.ones_like(batch_shape, dtype=torch.long, device=num_segments.device), num_segments.unsqueeze(dim=0)],
        dim=0,
    )
    # new_tensor is just a vector of [1 64] for example (assuming only 1 batch dimension)

    # 将 new_tensor 转换为 Python 列表，并将其元素转换为整数，得到新的形状 new_shape
    new_shape = [int(x) for x in new_tensor.tolist()]
    # 通过重塑 indices 张量，使其形状为 new_shape
    indices = indices.view(new_shape)

    # 创建一个倍增张量，其形状为 [batch_shape, 1]，将 indices 张量按 multiples 张量指定的次数重复
    multiples = torch.cat([batch_shape, torch.as_tensor([1])], dim=0)
    indices = indices.repeat(multiples.tolist())
    # equivalent (in Numpy:)
    # indices = torch.as_tensor(np.tile(indices.numpy(), multiples.tolist()))

    # 返回 IndexMap 对象，包含 indices 张量、num_segments 和 batch_shape 的长度（作为 batch_dims）
    return IndexMap(indices=indices, num_segments=num_segments, batch_dims=list(batch_shape.size())[0])
# 对输入的张量进行分段约简操作。
def _segment_reduce(values, index, segment_reduce_fn, name):
    """
    Applies a segment reduction segment-wise.

    Args:
        values (`torch.Tensor`):
            Tensor with segment values. 包含分段值的张量。
        index (`IndexMap`):
            IndexMap. 索引映射对象。
        segment_reduce_fn (`str`):
            Name for the reduce operation. One of "sum", "mean", "max" or "min". 约简操作的名称，可以是"sum"、"mean"、"max"或"min"之一。
        name (`str`):
            Name for the operation. Currently not used. 操作的名称，目前未使用

    Returns:
        (`IndexMap`): IndexMap of shape batch_shape with elements equal to range(num_segments).
        返回值：形状为 batch_shape 的 IndexMap，其元素等于 range(num_segments)。
    """
    # Flatten the batch dimensions, as segments ops (scatter) do not support batching.
    # However if `values` has extra dimensions to the right keep them
    # unflattened. Segmented ops support vector-valued operations.
    # 压平批处理维度，因为分段操作（scatter）不支持批处理。
    # 如果 `values` 的右侧有额外的维度，则保持未压平。分段操作支持矢量值操作。
    flat_index = flatten(index)
    vector_shape = values.size()[len(index.indices.size()) :]  # torch.Size object
    flattened_shape = torch.cat(
        [torch.as_tensor([-1], dtype=torch.long), torch.as_tensor(vector_shape, dtype=torch.long)], dim=0
    )
    # 将 `values` 重塑为压平后的形状
    flat_values = values.reshape(flattened_shape.tolist())

    # Create a tensor filled with zeros for output
    # 为输出创建一个用零填充的张量
    out = torch.zeros(int(flat_index.num_segments), dtype=torch.float, device=flat_values.device)
    # 在指定维度上进行分段约简操作，使用给定的约简函数 `segment_reduce_fn`
    segment_means = out.scatter_reduce(
        dim=0, index=flat_index.indices.long(), src=flat_values.float(), reduce=segment_reduce_fn, include_self=False
    )

    # Unflatten the values.
    # 将值重新恢复为原始形状。
    new_shape = torch.cat(
        [
            torch.as_tensor(index.batch_shape(), dtype=torch.long),
            torch.as_tensor([index.num_segments], dtype=torch.long),
            torch.as_tensor(vector_shape, dtype=torch.long),
        ],
        dim=0,
    )

    # Clone segment_means and reshape it to match the original `values` shape
    # 克隆 segment_means，并将其重塑以匹配原始 `values` 的形状
    output_values = segment_means.clone().view(new_shape.tolist()).to(values.dtype)
    # 创建并返回输出索引对象
    output_index = range_index_map(index.batch_shape(), index.num_segments)
    return output_values, output_index


def reduce_sum(values, index, name="segmented_reduce_sum"):
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
            包含需要进行分段求和的值的张量。
        index (`IndexMap`, indices are of shape [B1, B2, ..., Bn, I1, .., Ik].):
            Index defining the segments. 定义分段的索引对象。
        name (`str`, *optional*, defaults to 'segmented_reduce_sum'):
            Name for the operation. Currently not used
            操作的名称，默认为 'segmented_reduce_sum'，目前未使用
    """
    # 调用 _segment_reduce 函数，用于对输入的 values 和 index 进行分段求和操作
    # 返回的结果包括两部分：
    #   - output_values: 形状为 [B1, B2, ..., Bn, num_segments, V1, V2, ..] 的张量，包含了求和后的输出值
    #   - output_index: 类型为 IndexMap，形状为 [B1, B2, ..., Bn, num_segments]，表示每个分段的索引映射
    return _segment_reduce(values, index, "sum", name)
# 对输入的张量在其各个段上求平均值，空段返回0
def reduce_mean(values, index, name="segmented_reduce_mean"):
    # 调用内部函数 _segment_reduce，执行平均值操作
    return _segment_reduce(values, index, "mean", name)


# 对输入的张量在其各个段上求最大值
def reduce_max(values, index, name="segmented_reduce_max"):
    # 调用内部函数 _segment_reduce，执行最大值操作
    return _segment_reduce(values, index, "amax", name)


# 对输入的张量在其各个段上求最小值
def reduce_min(values, index, name="segmented_reduce_min"):
    # 此函数未完成，仅有文档字符串作为占位符
    # 使用 `_segment_reduce` 函数计算各段的最小值。该函数支持以下特性：
    #
    # - 使用第一维度 [B1, B2, ..., Bn] 进行批处理。每个批次中的元素可以具有不同的索引。
    # - 使用最后一维度 [V1, V2, ...] 进行向量化。如果存在这些维度，则输出将是向量的逐元素最小值，而不是标量。
    #
    # 只有中间维度 [I1, ..., Ik] 会被操作减少。
    #
    # Args:
    #     values (`torch.Tensor` of shape [B1, B2, ..., Bn, I1, .., Ik, V1, V2, ..]):
    #         包含要在每段中取最小值的张量。
    #     index (`IndexMap`, 索引的形状为 [B1, B2, ..., Bn, I1, .., Ik]):
    #         定义段的索引。
    #     name (`str`, *optional*, 默认为 'segmented_reduce_sum'):
    #         操作的名称。当前未使用。
    #
    # Returns:
    #     output_values (`torch.Tensor` 的形状为 [B1, B2, ..., Bn, num_segments, V1, V2, ..]):
    #         包含输出值的张量。
    #     output_index (`IndexMap`):
    #         形状为 [B1, B2, ..., Bn, num_segments] 的索引映射。
# 计算列 logits

def compute_column_logits(
    sequence_output, column_output_weights, column_output_bias, cell_index, cell_mask, allow_empty_column_selection
):
    """
    计算列的 logits。

    Args:
        sequence_output (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            模型最后一层的隐藏状态输出，也称为 last_hidden_state。
        column_output_weights (`torch.FloatTensor` of shape `(hidden_size)`):
            列选择线性层的权重。
        column_output_bias (`torch.FloatTensor` of shape `()`):
            列选择线性层的偏置。
        cell_index (`ProductIndexMap`):
            将标记分组为单元格的索引。
        cell_mask (`torch.FloatTensor` of shape `(batch_size, max_num_rows * max_num_cols)`):
            表格中存在的单元格的掩码（即非填充单元格）。
        allow_empty_column_selection (`bool`):
            是否允许不选择任何列

    Returns:
        column_logits (`torch.FloatTensor`of shape `(batch_size, max_num_cols)`):
            包含每个示例在批次中的列 logits 的张量。
    """

    # 首先，计算标记的 logits（batch_size, seq_len）- 不考虑温度
    token_logits = torch.einsum("bsj,j->bs", sequence_output, column_output_weights) + column_output_bias

    # 接下来，按单元格平均 logits（batch_size, max_num_cols*max_num_rows）
    cell_logits, cell_logits_index = reduce_mean(token_logits, cell_index)

    # 最后，按列平均 logits（batch_size, max_num_cols）
    column_index = cell_index.project_inner(cell_logits_index)
    column_logits, out_index = reduce_sum(cell_logits * cell_mask, column_index)

    # 计算每列的单元格数
    cell_count, _ = reduce_sum(cell_mask, column_index)
    column_logits /= cell_count + EPSILON_ZERO_DIVISION

    # 掩盖不在示例中出现的列
    is_padding = torch.logical_and(cell_count < 0.5, ~torch.eq(out_index.indices, 0))
    column_logits += CLOSE_ENOUGH_TO_LOG_ZERO * torch.as_tensor(
        is_padding, dtype=torch.float32, device=is_padding.device
    )

    # 如果不允许空列选择，则将 logits 加上一个小量，以表示选择空列的代价
    if not allow_empty_column_selection:
        column_logits += CLOSE_ENOUGH_TO_LOG_ZERO * torch.as_tensor(
            torch.eq(out_index.indices, 0), dtype=torch.float32, device=out_index.indices.device
        )

    return column_logits
    Args:
        token_logits (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
            每个标记的对数概率值的张量。
        column_logits (`torch.FloatTensor` of shape `(batch_size, max_num_cols)`):
            每个列的对数概率值的张量。
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            每个标记的标签。
        cell_index (`ProductIndexMap`):
            将标记分组为单元格的索引映射。
        col_index (`IndexMap`):
            将标记分组为列的索引映射。
        cell_mask (`torch.FloatTensor` of shape `(batch_size, max_num_rows * max_num_cols)`):
            表中存在的单元格的掩码（即不是填充的部分）。

    Returns:
        selection_loss_per_example (`torch.FloatTensor` of shape `(batch_size,)`):
            每个示例的损失值。
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
            新的对数概率值，只允许选择单个列中的单元格。
            根据 *column_logits* 最可能的列之外的对数概率设为非常低的值（概率为0）。
    """
    # Part 1: column loss

    # 首先确定应选择的列。我们使用具有最大选定单元格数的列。
    labels_per_column, _ = reduce_sum(torch.as_tensor(labels, dtype=torch.float32, device=labels.device), col_index)
    # labels_per_column 的形状是 (batch_size, max_num_cols)。
    # 它包含每个示例每列的标签数量。
    column_label = torch.argmax(labels_per_column, dim=-1)  # 形状为 (batch_size,)
    # 检查列中是否没有选定的单元格。在这种情况下，模型应该预测特殊的列 id 0，表示“不选择任何内容”。
    no_cell_selected = torch.eq(
        torch.max(labels_per_column, dim=-1)[0], 0
    )  # no_cell_selected 的形状为 (batch_size,)，值为 True
    # 如果批处理中的某个示例没有选定单元格（即没有为该示例设置为1的标签），则将 column_label 设置为0。
    column_label = torch.where(
        no_cell_selected.view(column_label.size()), torch.zeros_like(column_label), column_label
    )

    column_dist = torch.distributions.Categorical(logits=column_logits)  # 形状为 (batch_size, max_num_cols)
    column_loss_per_example = -column_dist.log_prob(column_label)

    # Part 2: cell loss

    # 将标签和对数概率从每个标记减少到每个单元格。
    # logits_per_cell: 形状为 (batch_size, max_num_rows*max_num_cols) 即 (batch_size, 64*32)
    logits_per_cell, _ = reduce_mean(token_logits, cell_index)
    # labels_per_cell: 形状为 (batch_size, 64*32)，指示每个单元格是否应该被选择（1）或不被选择（0）
    labels_per_cell, labels_index = reduce_max(
        torch.as_tensor(labels, dtype=torch.long, device=labels.device), cell_index
    )

    # 选择所选列的掩码。
    # column_id_for_cells: 形状为 (batch_size, 64*32)，指示每个单元格属于哪一列。
    # 使用 `cell_index` 对 `labels_index` 进行投影操作，并获取投影后的列索引
    column_id_for_cells = cell_index.project_inner(labels_index).indices

    # 创建一个形状为 (batch_size, 64*32) 的张量 `column_mask`，
    # 如果单元格属于要选择的列，则该值等于1
    column_mask = torch.as_tensor(
        torch.eq(column_id_for_cells, torch.unsqueeze(column_label, dim=-1)),
        dtype=torch.float32,
        device=cell_mask.device,
    )

    # 使用 Bernoulli 分布生成 `cell_dist`，logits_per_cell 的形状为 (batch_size, 64*32)
    cell_dist = torch.distributions.Bernoulli(logits=logits_per_cell)
    # 计算每个单元格的对数似然，仅针对所选列
    cell_log_prob = cell_dist.log_prob(labels_per_cell.type(torch.float32))  # 形状为 (batch_size, 64*32)

    # 计算单元格损失，乘以列掩码和单元格掩码
    cell_loss = -torch.sum(cell_log_prob * column_mask * cell_mask, dim=1)

    # 将损失标准化为列中的单元格数目
    cell_loss /= torch.sum(column_mask * cell_mask, dim=1) + EPSILON_ZERO_DIVISION

    # 将列损失初始化为每个示例的选择损失
    selection_loss_per_example = column_loss_per_example
    # 如果没有选择单元格，则设置损失为零，否则使用计算的 `cell_loss`
    selection_loss_per_example += torch.where(
        no_cell_selected.view(selection_loss_per_example.size()),
        torch.zeros_like(selection_loss_per_example),
        cell_loss,
    )

    # 通过对 `column_logits` 的最大值获取模型选择的列 ID
    selected_column_id = torch.as_tensor(
        torch.argmax(column_logits, dim=-1), dtype=torch.long, device=column_logits.device
    )  # 形状为 (batch_size,)

    # 创建一个形状为 (batch_size, 64*32) 的 `selected_column_mask`，
    # 如果单元格属于模型选择的列，则该值等于1
    selected_column_mask = torch.as_tensor(
        torch.eq(column_id_for_cells, torch.unsqueeze(selected_column_id, dim=-1)),
        dtype=torch.float32,
        device=selected_column_id.device,
    )

    # 不选择特殊列 ID 为 0 的单元格
    selected_column_mask = torch.where(
        torch.eq(column_id_for_cells, 0).view(selected_column_mask.size()),
        torch.zeros_like(selected_column_mask),
        selected_column_mask,
    )

    # 调整 `logits_per_cell`，确保在模型选择的列之外将概率设为0
    new_logits_per_cell = logits_per_cell + CLOSE_ENOUGH_TO_LOG_ZERO * (1.0 - cell_mask * selected_column_mask)
    # 使用 `cell_index` 收集新的 `new_logits_per_cell`，形状由 `cell_index` 决定
    logits = gather(new_logits_per_cell, cell_index)

    # 返回选择损失和调整后的 `logits`
    return selection_loss_per_example, logits
    """
    Computes logits per token

    Args:
        sequence_output (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Also known as last_hidden_state. Sequence of hidden-states at the output of the last layer of the model.
        temperature (`float`):
            Temperature for the Bernoulli distribution.
        output_weights (`torch.FloatTensor` of shape `(hidden_size,)`):
            Weights of the linear layer for cell selection.
        output_bias (`torch.FloatTensor` of shape `()`):
            Bias of the linear layer for cell selection

    Returns:
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length)`): Logits per token.
    """
    # 计算每个 token 的 logits
    logits = (torch.einsum("bsj,j->bs", sequence_output, output_weights) + output_bias) / temperature

    return logits


def _calculate_aggregate_mask(answer, pooled_output, cell_selection_preference, labels, aggregation_classifier):
    """
    Finds examples where the model should select cells with no aggregation.

    Returns a mask that determines for which examples should the model select answers directly from the table, without
    any aggregation function. If the answer is a piece of text the case is unambiguous as aggregation functions only
    apply to numbers. If the answer is a number but does not appear in the table then we must use some aggregation
    case. The ambiguous case is when the answer is a number that also appears in the table. In this case we use the
    aggregation function probabilities predicted by the model to decide whether to select or aggregate. The threshold
    for this is a hyperparameter *cell_selection_preference*

    Args:
        answer (`torch.FloatTensor` of shape `(batch_size, )`):
            Answer for every example in the batch. Nan if there is no scalar answer.
        pooled_output (`torch.FloatTensor` of shape `(batch_size, hidden_size)`):
            Output of the pooler (BertPooler) on top of the encoder layer.
        cell_selection_preference (`float`):
            Preference for cell selection in ambiguous cases.
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Labels per token.
        aggregation_classifier (`torch.nn.Linear`): Aggregation head

    Returns:
        aggregate_mask (`torch.FloatTensor` of shape `(batch_size,)`): A mask set to 1 for examples that should use
        aggregation functions.
    """
    # 创建一个初始的聚合掩码，根据答案是否为数值来确定是否需要聚合
    aggregate_mask_init = torch.logical_not(torch.isnan(answer)).type(torch.FloatTensor).to(answer.device)
    
    # 使用汇聚分类器计算聚合操作的 logits
    logits_aggregation = aggregation_classifier(pooled_output)
    
    # 使用分类分布创建一个分布对象
    dist_aggregation = torch.distributions.categorical.Categorical(logits=logits_aggregation)
    
    # 计算除了“不进行聚合”操作之外的所有聚合操作的总质量
    aggregation_ops_total_mass = torch.sum(dist_aggregation.probs[:, 1:], dim=1)
    # 根据当前模型的选择例子进行细胞选择。
    is_pred_cell_selection = aggregation_ops_total_mass <= cell_selection_preference

    # 存在非空细胞选择监督的例子。
    is_cell_supervision_available = torch.sum(labels, dim=1) > 0

    # torch.where 与 tf.where（在 tensorflow 1 中）不等价，
    # 因此在条件上添加 .view，以匹配第一个张量的形状。
    aggregate_mask = torch.where(
        torch.logical_and(is_pred_cell_selection, is_cell_supervision_available).view(aggregate_mask_init.size()),
        torch.zeros_like(aggregate_mask_init, dtype=torch.float32),
        aggregate_mask_init,
    )

    # 分离张量，使其不再跟踪梯度。
    aggregate_mask = aggregate_mask.detach()

    # 返回聚合掩码。
    return aggregate_mask
# 计算已知情况下的聚合损失
def _calculate_aggregation_loss_known(
    logits_aggregation, aggregate_mask, aggregation_labels, use_answer_as_supervision, num_aggregation_labels
):
    """
    在训练过程中，当聚合类型已知时计算聚合损失。

    在弱监督设置中，唯一已知的信息是对于单元格选择示例，应预测“无聚合”。对于其他需要聚合的示例，不会累积损失。在总是已知聚合类型的设置中，将为所有示例累积标准交叉熵损失。

    Args:
        logits_aggregation (`torch.FloatTensor` of shape `(batch_size, num_aggregation_labels)`):
            每个聚合操作的logits。
        aggregate_mask (`torch.FloatTensor` of shape `(batch_size, )`):
            对应于应使用聚合函数的示例的掩码，设为1。
        aggregation_labels (`torch.LongTensor` of shape `(batch_size, )`):
            每个示例的聚合函数 id。
        use_answer_as_supervision (`bool`, *可选*):
            是否将答案作为聚合示例的唯一监督。
        num_aggregation_labels (`int`, *可选*, 默认为0):
            要预测的聚合运算符数量。

    Returns:
        aggregation_loss_known (`torch.FloatTensor` of shape `(batch_size,)`): 每个示例的聚合损失（在已知类型的情况下）。
    """
    if use_answer_as_supervision:
        # 为单元格选择示例准备“无聚合”目标。
        target_aggregation = torch.zeros_like(aggregate_mask, dtype=torch.long)
    else:
        # 使用聚合监督作为目标。
        target_aggregation = aggregation_labels

    one_hot_labels = nn.functional.one_hot(target_aggregation, num_classes=num_aggregation_labels).type(torch.float32)
    log_probs = nn.functional.log_softmax(logits_aggregation, dim=-1)

    # torch.FloatTensor[batch_size]
    per_example_aggregation_intermediate = -torch.sum(one_hot_labels * log_probs, dim=-1)
    if use_answer_as_supervision:
        # 仅累积需要单元格选择（无聚合）的示例的损失。
        return per_example_aggregation_intermediate * (1 - aggregate_mask)
    else:
        return per_example_aggregation_intermediate


# 计算未知情况下的聚合损失
def _calculate_aggregation_loss_unknown(logits_aggregation, aggregate_mask):
    """
    在答案监督情况下计算聚合损失。

    Args:
        logits_aggregation (`torch.FloatTensor` of shape `(batch_size, num_aggregation_labels)`):
            每个聚合操作的logits。
        aggregate_mask (`torch.FloatTensor` of shape `(batch_size, )`):
            对应于应使用聚合函数的示例的掩码，设为1。

    Returns:
        aggregation_loss_unknown (`torch.FloatTensor` of shape `(batch_size,)`): 每个示例的聚合损失（在答案监督情况下）。
    """
    # 创建一个分类分布对象，基于给定的 logits 参数
    dist_aggregation = torch.distributions.categorical.Categorical(logits=logits_aggregation)
    # 计算除了第一个索引外的所有聚合操作的总概率质量
    aggregation_ops_total_mass = torch.sum(dist_aggregation.probs[:, 1:], dim=1)
    # 在需要聚合答案的情况下预测一些聚合操作。
    # 这会增加所有聚合函数的概率，类似于最大边际似然(MML)，但不考虑函数是否给出正确答案。
    # 返回负对数似然乘以聚合掩码，用于损失函数计算
    return -torch.log(aggregation_ops_total_mass) * aggregate_mask
# 计算每个样本的聚合损失
def _calculate_aggregation_loss(
    logits_aggregation,
    aggregate_mask,
    aggregation_labels,
    use_answer_as_supervision,
    num_aggregation_labels,
    aggregation_loss_weight,
):
    """
    计算每个样本的聚合损失。

    Args:
        logits_aggregation (`torch.FloatTensor` of shape `(batch_size, num_aggregation_labels)`):
            每个聚合操作的logits。
        aggregate_mask (`torch.FloatTensor` of shape `(batch_size, )`):
            对应于应使用聚合函数的样本的掩码，为1。
        aggregation_labels (`torch.LongTensor` of shape `(batch_size, )`):
            每个样本的聚合函数 ID。
        use_answer_as_supervision (`bool`, *optional*):
            是否将答案作为聚合样本的唯一监督。
        num_aggregation_labels (`int`, *optional*, defaults to 0):
            预测的聚合操作数目。
        aggregation_loss_weight (`float`, *optional*, defaults to 1.0):
            聚合损失的权重。

    Returns:
        aggregation_loss (`torch.FloatTensor` of shape `(batch_size,)`): 每个样本的聚合损失。
    """
    # 使用已知的聚合损失计算函数
    per_example_aggregation_loss = _calculate_aggregation_loss_known(
        logits_aggregation, aggregate_mask, aggregation_labels, use_answer_as_supervision, num_aggregation_labels
    )

    if use_answer_as_supervision:
        # 对需要聚合的数值答案增加聚合损失
        per_example_aggregation_loss += _calculate_aggregation_loss_unknown(logits_aggregation, aggregate_mask)
    return aggregation_loss_weight * per_example_aggregation_loss


# 计算给定单元格和聚合概率时的期望结果
def _calculate_expected_result(
    dist_per_cell, numeric_values, numeric_values_scale, input_mask_float, logits_aggregation, config
):
    """
    计算给定单元格和聚合概率时的期望结果。

    Args:
        dist_per_cell (`torch.distributions.Bernoulli`):
            每个单元格的选择分布。
        numeric_values (`torch.FloatTensor` of shape `(batch_size, seq_length)`):
            每个标记的数值。对于非数值标记为NaN。
        numeric_values_scale (`torch.FloatTensor` of shape `(batch_size, seq_length)`):
            每个标记数值的缩放。
        input_mask_float (`torch.FloatTensor` of shape `(batch_size, seq_length)`):
            表的掩码，不包括问题标记和表头。
        logits_aggregation (`torch.FloatTensor` of shape `(batch_size, num_aggregation_labels)`):
            每个聚合操作的logits。
        config ([`TapasConfig`]):
            包含模型所有超参数的配置类。

    Returns:
        expected_result (`torch.FloatTensor` of shape `(batch_size,)`): 每个样本的期望结果。
    """
    # 如果配置中使用 Gumbel 分布来处理单元格，则创建一个 RelaxedBernoulli 分布对象
    gumbel_dist = torch.distributions.RelaxedBernoulli(
        # 由于标记的 logit 已经被温度除过并用于计算单元格选择误差，因此这里需要再次乘以温度
        temperature=config.temperature,
        logits=dist_per_cell.logits * config.temperature,
    )
    # 从 Gumbel 分布中采样得到每个单元格的概率值
    scaled_probability_per_cell = gumbel_dist.sample()
else:
    # 如果配置中未使用 Gumbel 分布，则直接使用每个单元格的概率值
    scaled_probability_per_cell = dist_per_cell.probs

# <float32>[batch_size, seq_length]，将每个单元格的概率按照数值缩放比例和输入掩码进行调整
scaled_probability_per_cell = (scaled_probability_per_cell / numeric_values_scale) * input_mask_float
# 计算每个批次中单元格概率的总和
count_result = torch.sum(scaled_probability_per_cell, dim=1)
# 将非数值表格值置零，使用 torch.where 进行条件替换
numeric_values_masked = torch.where(
    torch.isnan(numeric_values), torch.zeros_like(numeric_values), numeric_values
)  # Mask non-numeric table values to zero.
# 计算加权和结果，使用数值表格值乘以单元格概率和数值掩码
sum_result = torch.sum(scaled_probability_per_cell * numeric_values_masked, dim=1)
# 选择平均近似函数配置
avg_approximation = config.average_approximation_function
if avg_approximation == AverageApproximationFunction.RATIO:
    # 使用比率近似方法计算平均值结果
    average_result = sum_result / (count_result + EPSILON_ZERO_DIVISION)
elif avg_approximation == AverageApproximationFunction.FIRST_ORDER:
    # 使用一阶方法计算平均值结果，基于 TAPAS 论文附录 D 中的公式 X_c
    ex = torch.sum(scaled_probability_per_cell, dim=1, keepdim=True) - scaled_probability_per_cell + 1
    average_result = torch.sum(numeric_values_masked * scaled_probability_per_cell / ex, dim=1)
elif avg_approximation == AverageApproximationFunction.SECOND_ORDER:
    # 使用二阶方法计算平均值结果，基于 TAPAS 论文附录 D 中的公式
    ex = torch.sum(scaled_probability_per_cell, dim=1, keepdim=True) - scaled_probability_per_cell + 1
    pointwise_var = scaled_probability_per_cell * (1 - scaled_probability_per_cell)
    var = torch.sum(pointwise_var, dim=1, keepdim=True) - pointwise_var
    multiplier = (var / torch.square(ex) + 1) / ex
    average_result = torch.sum(numeric_values_masked * scaled_probability_per_cell * multiplier, dim=1)
else:
    # 如果配置中的平均近似函数配置无效，则抛出异常
    raise ValueError(f"Invalid average_approximation_function: {config.average_approximation_function}")

if config.use_gumbel_for_aggregation:
    # 如果配置中使用 Gumbel 分布来处理聚合操作，则创建一个 RelaxedOneHotCategorical 分布对象
    gumbel_dist = torch.distributions.RelaxedOneHotCategorical(
        config.aggregation_temperature, logits=logits_aggregation[:, 1:]
    )
    # <float32>[batch_size, num_aggregation_labels - 1]，从 Gumbel 分布中采样得到聚合操作的概率值
    aggregation_op_only_probs = gumbel_dist.sample()
    else:
        # 计算去除第一列后的 logits 的 softmax 操作，用于聚合操作概率计算
        aggregation_op_only_probs = nn.functional.softmax(
            logits_aggregation[:, 1:] / config.aggregation_temperature, dim=-1
        )

    # 将三个结果张量按列拼接成一个张量
    all_results = torch.cat(
        [
            torch.unsqueeze(sum_result, dim=1),
            torch.unsqueeze(average_result, dim=1),
            torch.unsqueeze(count_result, dim=1),
        ],
        dim=1,
    )

    # 计算期望的聚合结果，通过加权求和得到
    expected_result = torch.sum(all_results * aggregation_op_only_probs, dim=1)
    # 返回期望的结果张量
    return expected_result
# PyTorch 目前不支持带有自定义 delta 的 Huber 损失函数，因此我们自己定义它
def huber_loss(input, target, delta: float = 1.0):
    # 计算输入和目标之间的绝对误差，形状为 (batch_size,)
    errors = torch.abs(input - target)
    # 根据误差是否小于 delta，选择计算平方误差的一半或线性误差减去常量
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
            每个样本的答案。如果没有标量答案则为 NaN。
        aggregate_mask (`torch.FloatTensor` of shape `(batch_size,)`):
            对应需要使用聚合函数的样本的掩码，为1。
        dist_per_cell (`torch.distributions.Bernoulli`):
            每个单元格的选择分布。
        numeric_values (`torch.FloatTensor` of shape `(batch_size, seq_length)`):
            每个标记的数值。对于非数值标记为 NaN。
        numeric_values_scale (`torch.FloatTensor` of shape `(batch_size, seq_length)`):
            每个标记数值的规模。
        input_mask_float (`torch.FloatTensor` of shape `(batch_size, seq_length)`):
            表格的掩码，不包括问题标记和表头。
        logits_aggregation (`torch.FloatTensor` of shape `(batch_size, num_aggregation_labels)`):
            每种聚合操作的对数。
        config ([`TapasConfig`]):
            模型配置类，包含模型的所有参数。

    Returns:
        per_example_answer_loss_scaled (`torch.FloatTensor` of shape `(batch_size,)`): 每个样本的答案损失（已缩放）。
        large_answer_loss_mask (`torch.FloatTensor` of shape `(batch_size,)`): 一个掩码，对于损失超过 answer_loss_cutoff 的样本为 1。
    """
    # float32 (batch_size,)
    # 计算预期结果
    expected_result = _calculate_expected_result(
        dist_per_cell, numeric_values, numeric_values_scale, input_mask_float, logits_aggregation, config
    )

    # float32 (batch_size,)
    # 将答案中的 NaN 替换为零
    answer_masked = torch.where(torch.isnan(answer), torch.zeros_like(answer), answer)

    if config.use_normalized_answer_loss:
        # 计算归一化因子，避免零除错误
        normalizer = (torch.max(torch.abs(expected_result), torch.abs(answer_masked)) + EPSILON_ZERO_DIVISION).detach()

        # 对答案和预期结果进行归一化
        normalized_answer_masked = answer_masked / normalizer
        normalized_expected_result = expected_result / normalizer

        # 使用 Huber 损失函数计算答案损失
        per_example_answer_loss = huber_loss(
            normalized_expected_result * aggregate_mask, normalized_answer_masked * aggregate_mask
        )
    else:
        # 使用 Huber 损失函数计算答案损失，使用配置中的 delta
        per_example_answer_loss = huber_loss(
            expected_result * aggregate_mask, answer_masked * aggregate_mask, delta=config.huber_loss_delta
        )
    # 如果配置中的答案损失截断为 None，则创建一个全为1的张量，与 per_example_answer_loss 的形状相同
    if config.answer_loss_cutoff is None:
        large_answer_loss_mask = torch.ones_like(per_example_answer_loss, dtype=torch.float32)

    else:
        # 否则，根据答案损失是否大于答案损失截断值，创建一个掩码张量
        large_answer_loss_mask = torch.where(
            per_example_answer_loss > config.answer_loss_cutoff,  # 条件：答案损失大于答案损失截断值
            torch.zeros_like(per_example_answer_loss, dtype=torch.float32),  # 答案损失大于截断值时的值
            torch.ones_like(per_example_answer_loss, dtype=torch.float32),  # 答案损失不大于截断值时的值
        )
    
    # 计算每个样本的答案损失按比例缩放后的值，乘以聚合掩码
    per_example_answer_loss_scaled = config.answer_loss_importance * (per_example_answer_loss * aggregate_mask)

    # 返回按比例缩放后的答案损失和大答案损失掩码
    return per_example_answer_loss_scaled, large_answer_loss_mask
```