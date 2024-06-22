# `.\transformers\models\lxmert\modeling_lxmert.py`

```py
# coding=utf-8
# 版权声明
# Copyright 2018 Hao Tan, Mohit Bansal, and the HuggingFace team
# 
# 根据 Apache 许可证版本 2.0 使用本文件，除非您遵守许可证规定，否则您不得使用此文件。
# 您可以在以下网址获取许可证副本
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# 除非适用法律要求或书面同意，否则本软件按"原样"分发，不提供任何形式的明示或暗示担保或条件。
# 请参阅许可证了解特定语言的管理权限和限制。
""" PyTorch LXMERT model."""

# 导入所需的库和模块
import math
import os
import warnings
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

# 导入 PyTorch 库
import torch
from torch import nn
from torch.nn import CrossEntropyLoss, SmoothL1Loss

# 导入模型相关的工具和配置
from ...activations import ACT2FN, gelu
from ...modeling_utils import PreTrainedModel
from ...utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from .configuration_lxmert import LxmertConfig

# 获取日志记录器
logger = logging.get_logger(__name__)

# 文档中使用的模型检查点和配置文件
_CHECKPOINT_FOR_DOC = "unc-nlp/lxmert-base-uncased"
_CONFIG_FOR_DOC = "LxmertConfig"

# 预训练模型的存档列表
LXMERT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "unc-nlp/lxmert-base-uncased",
]

# GeLU 激活函数的定义
class GeLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return gelu(x)

# LxmertModelOutput 类的数据结构定义，包含语言、视觉和跨模态编码器的最后隐藏状态、池化输出以及注意力概率。
@dataclass
class LxmertModelOutput(ModelOutput):
    """
    Lxmert's outputs that contain the last hidden states, pooled outputs, and attention probabilities for the language,
    visual, and, cross-modality encoders. (note: the visual encoder in Lxmert is referred to as the "relation-ship"
    encoder")
    Args:
        language_output (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            语言编码器最后一层的隐藏状态序列。
        vision_output (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            视觉编码器最后一层的隐藏状态序列。
        pooled_output (`torch.FloatTensor` of shape `(batch_size, hidden_size)`):
            序列第一个标记（分类，CLS标记）的最后一层隐藏状态，经过一个线性层和一个Tanh激活函数进一步处理。
        language_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            元组，包含一个输入特征的`torch.FloatTensor`（每个交叉模态层的输出也是一个）的形状为`(batch_size, sequence_length, hidden_size)`。
        vision_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            元组，包含一个输入特征的`torch.FloatTensor`（每个交叉模态层的输出也是一个）的形状为`(batch_size, sequence_length, hidden_size)`。
        language_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            元组，包含每个层的`torch.FloatTensor`（每个头的一个）的形状为`(batch_size, num_heads, sequence_length, sequence_length)`。注意力softmax之后的注意力权重，用于在自注意力头中计算加权平均值。
        vision_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            元组，包含每个层的`torch.FloatTensor`（每个头的一个）的形状为`(batch_size, num_heads, sequence_length, sequence_length)`。注意力softmax之后的注意力权重，用于在自注意力头中计算加权平均值。
        cross_encoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            元组，包含每个层的`torch.FloatTensor`（每个头的一个）的形状为`(batch_size, num_heads, sequence_length, sequence_length)`。注意力softmax之后的注意力权重，用于在自注意力头中计算加权平均值。
    """

    language_output: Optional[torch.FloatTensor] = None
    vision_output: Optional[torch.FloatTensor] = None
    pooled_output: Optional[torch.FloatTensor] = None
    language_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    vision_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # 定义一个可选的元组变量，用于存储语言注意力信息
    language_attentions: Optional[Tuple[torch.FloatTensor]] = None
    # 定义一个可选的元组变量，用于存储视觉注意力信息
    vision_attentions: Optional[Tuple[torch.FloatTensor]] = None
    # 定义一个可选的元组变量，用于存储交叉编码器的注意力信息
    cross_encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
# 基于dataclass装饰器创建一个名为LxmertForQuestionAnsweringOutput的类
class LxmertForQuestionAnsweringOutput(ModelOutput):
    """
    Output type of [`LxmertForQuestionAnswering`].

    Args:
        loss (*optional*, returned when `labels` is provided, `torch.FloatTensor` of shape `(1,)`):
            Total loss as the sum of the masked language modeling loss and the next sequence prediction
            (classification) loss.k.
        question_answering_score (`torch.FloatTensor` of shape `(batch_size, n_qa_answers)`, *optional*):
            Prediction scores of question answering objective (classification).
        language_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for input features + one for the output of each cross-modality layer) of
            shape `(batch_size, sequence_length, hidden_size)`.
        vision_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for input features + one for the output of each cross-modality layer) of
            shape `(batch_size, sequence_length, hidden_size)`.
        language_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
            the self-attention heads.
        vision_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
            the self-attention heads.
        cross_encoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
            the self-attention heads.
    """

    # 初始化参数，可选参数，当提供'labels'时返回，返回一个形状为`(1,)`的torch.FloatTensor
    loss: Optional[torch.FloatTensor] = None
    # 问题回答目标分类的预测得分，形状为`(batch_size, n_qa_answers)`的torch.FloatTensor
    question_answering_score: Optional[torch.FloatTensor] = None
    # 语言隐藏状态的元组，元组中包含`torch.FloatTensor`，当`output_hidden_states=True`时返回，或者当`config.output_hidden_states=True`时返回，每个元素形状为`(batch_size, sequence_length, hidden_size)`
    language_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # 视觉隐藏状态的元组，元组中包含`torch.FloatTensor`，当`output_hidden_states=True`时返回，或者当`config.output_hidden_states=True`时返回，每个元素形状为`(batch_size, sequence_length, hidden_size)`
    vision_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # 语言注意力的元组，元组中包含`torch.FloatTensor`，当`output_attentions=True`时返回，或者当`config.output_attentions=True`时返回，每个元素形状为`(batch_size, num_heads, sequence_length, sequence_length)`，注意力softmax后的权重，用于计算自注意力头中的加权平均值
    language_attentions: Optional[Tuple[torch.FloatTensor]] = None
    # 视觉注意力的元组，元组中包含`torch.FloatTensor`，当`output_attentions=True`时返回，或者当`config.output_attentions=True`时返回，每个元素形状为`(batch_size, num_heads, sequence_length, sequence_length)`，注意力softmax后的权重，用于计算自注意力头中的加权平均值
    vision_attentions: Optional[Tuple[torch.FloatTensor]] = None
    # 定义 cross_encoder_attentions 变量，类型为可选的元组，元素为 torch 的 FloatTensor 类型
    cross_encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
@dataclass
class LxmertForPreTrainingOutput(ModelOutput):
    """
    LxmertForPreTrainingOutput 类的输出类型。

    Args:
        loss (*optional*, returned when `labels` is provided, `torch.FloatTensor` of shape `(1,)`):
            总损失，由掩码语言建模损失和下一个序列预测（分类）损失之和组成。
        prediction_logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            语言建模头部的预测分数（SoftMax 前每个词汇标记的分数）。
        cross_relationship_score (`torch.FloatTensor` of shape `(batch_size, 2)`):
            文本匹配目标（分类）头部的预测分数（SoftMax 前的 True/False 继续分数）。
        question_answering_score (`torch.FloatTensor` of shape `(batch_size, n_qa_answers)`):
            问题回答目标的预测分数（分类）。
        language_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            `torch.FloatTensor` 的元组（一个用于输入特征 + 一个用于每个交叉模态层的输出），形状为 `(batch_size, sequence_length, hidden_size)`。
        vision_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            `torch.FloatTensor` 的元组（一个用于输入特征 + 一个用于每个交叉模态层的输出），形状为 `(batch_size, sequence_length, hidden_size)`。
        language_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            `torch.FloatTensor` 的元组（每层一个）形状为 `(batch_size, num_heads, sequence_length, sequence_length)`。注意力 softmax 后的注意力权重，用于计算自注意力头中的加权平均值。
        vision_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            `torch.FloatTensor` 的元组（每层一个）形状为 `(batch_size, num_heads, sequence_length, sequence_length)`。注意力 softmax 后的注意力权重，用于计算自注意力头中的加权平均值。
        cross_encoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            `torch.FloatTensor` 的元组（每层一个）形状为 `(batch_size, num_heads, sequence_length, sequence_length)`。注意力 softmax 后的注意力权重，用于计算自注意力头中的加权平均值。

    """
    # 初始化损失值为 None
    loss: Optional[torch.FloatTensor] = None
    # 初始化预测 logits 为 None
    prediction_logits: Optional[torch.FloatTensor] = None
    # 初始化跨关系分数为 None
    cross_relationship_score: Optional[torch.FloatTensor] = None
    # 初始化问答分数为 None
    question_answering_score: Optional[torch.FloatTensor] = None
    # 初始化语言模型隐藏状态为 None
    language_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # 初始化视觉模型隐藏状态为 None
    vision_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # 初始化语言模型注意力分布为 None
    language_attentions: Optional[Tuple[torch.FloatTensor]] = None
    # 初始化视觉模型注意力分布为 None
    vision_attentions: Optional[Tuple[torch.FloatTensor]] = None
    # 初始化跨编码器注意力分布为 None
    cross_encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
# 将 TensorFlow 权重加载到 PyTorch 模型中的函数
def load_tf_weights_in_lxmert(model, config, tf_checkpoint_path):
    """Load tf checkpoints in a pytorch model."""
    # 导入必要的库
    try:
        import re  # 导入正则表达式模块
        import numpy as np  # 导入 NumPy 库
        import tensorflow as tf  # 导入 TensorFlow 库
    except ImportError:
        # 如果导入失败，记录错误信息并抛出 ImportError
        logger.error(
            "Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions."
        )
        raise
    
    # 获取 TensorFlow 检查点文件的绝对路径
    tf_path = os.path.abspath(tf_checkpoint_path)
    # 记录日志，指示正在转换 TensorFlow 检查点
    logger.info(f"Converting TensorFlow checkpoint from {tf_path}")
    
    # 从 TF 模型加载权重
    init_vars = tf.train.list_variables(tf_path)  # 列出 TensorFlow 模型中的变量名和形状
    names = []  # 保存变量名的列表
    arrays = []  # 保存变量值的列表
    for name, shape in init_vars:
        # 记录日志，指示正在加载 TF 权重和其形状
        logger.info(f"Loading TF weight {name} with shape {shape}")
        # 加载 TensorFlow 模型中的变量
        array = tf.train.load_variable(tf_path, name)
        names.append(name)  # 将变量名添加到列表中
        arrays.append(array)  # 将变量值添加到列表中
    
    # 遍历变量名和变量值的列表
    for name, array in zip(names, arrays):
        name = name.split("/")  # 将变量名按斜杠分割成部分
        # 排除不需要加载的变量，如优化器中的变量和全局步数变量等
        if any(
            n
            in [
                "adam_v",
                "adam_m",
                "AdamWeightDecayOptimizer",
                "AdamWeightDecayOptimizer_1",
                "global_step",
            ]
            for n in name
        ):
            # 如果变量名匹配到不需要加载的模式，记录日志并跳过此变量的加载
            logger.info(f"Skipping {'/'.join(name)}")
            continue
        
        pointer = model  # 设置指针指向模型
        for m_name in name:
            if re.fullmatch(r"[A-Za-z]+_\d+", m_name):
                scope_names = re.split(r"_(\d+)", m_name)  # 使用正则表达式将变量名拆分成作用域名称和索引
            else:
                scope_names = [m_name]  # 如果变量名不匹配模式，作用域名称为变量名本身
            # 根据作用域名称更新指针
            if scope_names[0] == "kernel" or scope_names[0] == "gamma":
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "output_bias" or scope_names[0] == "beta":
                pointer = getattr(pointer, "bias")
            elif scope_names[0] == "output_weights":
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "squad":
                pointer = getattr(pointer, "classifier")
            else:
                try:
                    pointer = getattr(pointer, scope_names[0])
                except AttributeError:
                    # 如果找不到对应的属性，记录日志并跳过加载此变量
                    logger.info(f"Skipping {'/'.join(name)}")
                    continue
            # 如果作用域名称包含索引，更新指针到指定索引处
            if len(scope_names) >= 2:
                num = int(scope_names[1])
                pointer = pointer[num]
        # 处理特殊的变量名，如嵌入层权重和 kernel，进行相应的操作
        if m_name[-11:] == "_embeddings":
            pointer = getattr(pointer, "weight")
        elif m_name == "kernel":
            array = np.transpose(array)  # 转置数组（通常是为了适配 TensorFlow 和 PyTorch 的权重格式）
        try:
            assert pointer.shape == array.shape  # 断言指针形状和数组形状相同
        except AssertionError as e:
            # 如果断言失败，抛出 AssertionError，附带指针和数组的形状信息
            e.args += (pointer.shape, array.shape)
            raise
        # 记录日志，指示正在初始化 PyTorch 权重
        logger.info(f"Initialize PyTorch weight {name}")
        pointer.data = torch.from_numpy(array)  # 使用 NumPy 数组初始化 PyTorch 权重
    return model

# 定义 LXMERT 模型的嵌入层类
class LxmertEmbeddings(nn.Module):
    # 构建从单词、位置和标记类型嵌入中构造嵌入向量。

    def __init__(self, config):
        # 继承父类构造函数
        super().__init__()
        # 定义单词嵌入层，用于将单词索引映射为隐藏表示，padding_idx=0表示填充索引为0
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        # 定义位置嵌入层，用于将位置索引映射为隐藏表示，padding_idx=0表示填充索引为0
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size, padding_idx=0)
        # 定义标记类型嵌入层，用于将标记类型索引映射为隐藏表示，padding_idx=0表示填充索引为0

        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size, padding_idx=0)

        # self.LayerNorm不使用蛇式命名，以保持与 TensorFlow 模型变量名称一致，并能够加载任何 TensorFlow 检查点文件
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        # 定义 LayerNorm 层，用于对隐藏表示进行归一化
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # 定义 Dropout 层，用于在训练中进行随机失活

    def forward(self, input_ids, token_type_ids=None, inputs_embeds=None):
        if input_ids is not None:
            input_shape = input_ids.size()
            device = input_ids.device
        else:
            input_shape = inputs_embeds.size()[:-1]
            device = inputs_embeds.device
        # 获取输入数据的形状信息

        seq_length = input_shape[1]
        # 获取序列长度

        position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
        # 生成位置索引
        position_ids = position_ids.unsqueeze(0).expand(input_shape)

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)
        # 如果标记类型索引为None，则创建全零的标记类型索引张量

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        # 如果输入嵌入为None，则使用单词嵌入层将输入单词索引转换为嵌入表示
        position_embeddings = self.position_embeddings(position_ids)
        # 使用位置嵌入层获取位置嵌入
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        # 使用标记类型嵌入层获取标记类型嵌入

        embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        # 将单词嵌入、位置嵌入和标记类型嵌入进行相加得到最终嵌入向量
        embeddings = self.LayerNorm(embeddings)
        # 对嵌入向量进行 LayerNorm 归一化处理
        embeddings = self.dropout(embeddings)
        # 对嵌入向量进行随机失活处理
        return embeddings
# 创建一个名为 LxmertAttention 的类，继承自 nn.Module
class LxmertAttention(nn.Module):
    # 类的初始化方法，接收 config 和 ctx_dim 参数
    def __init__(self, config, ctx_dim=None):
        # 调用父类的初始化方法
        super().__init__()
        # 如果 hidden_size 不能被 num_attention_heads 整除，引发数值错误
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )
        # 设置属性值
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.head_size = self.num_attention_heads * self.attention_head_size

        # 如果 ctx_dim 为 None，则设为 config.hidden_size
        if ctx_dim is None:
            ctx_dim = config.hidden_size
        # 创建 query、key、value 三个线性层
        self.query = nn.Linear(config.hidden_size, self.head_size)
        self.key = nn.Linear(ctx_dim, self.head_size)
        self.value = nn.Linear(ctx_dim, self.head_size)

        # 创建 dropout 层
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    # 定义 transpose_for_scores 方法，用于对输入张量进行形状转换
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    # 定义 forward 方法，实现注意力机制
    def forward(self, hidden_states, context, attention_mask=None, output_attentions=False):
        # 执行 query、key、value 的线性变换
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(context)
        mixed_value_layer = self.value(context)

        # 对变换后的结果执行形状转换
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # 计算点积得到原始的注意力分数
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # 如果存在 attention_mask，则应用到注意力分数上
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # 对注意力分数进行归一化，得到注意力概率
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # 对注意力概率进行dropout
        attention_probs = self.dropout(attention_probs)

        # 计算上下文表示
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        # 返回结果张量和注意力概率（可选）
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs


# 定义 LxmertAttentionOutput 类（未提供代码）
class LxmertAttentionOutput(nn.Module):
    # 初始化函数，用于初始化模型的参数
    def __init__(self, config):
        # 调用父类的初始化函数
        super().__init__()
        # 创建一个全连接层，输入和输出维度都是隐藏层的大小
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 创建一个 LayerNormalization 层，对隐藏状态进行归一化处理
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        # 创建一个 Dropout 层，用于在训练中随机丢弃部分隐藏状态
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    # 前向传播函数，定义了模型的前向计算逻辑
    def forward(self, hidden_states, input_tensor):
        # 使用全连接层对隐藏状态进行线性变换
        hidden_states = self.dense(hidden_states)
        # 对线性变换后的隐藏状态进行随机丢弃
        hidden_states = self.dropout(hidden_states)
        # 将丢弃后的隐藏状态与输入张量相加，并进行 LayerNormalization 处理
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        # 返回处理后的隐藏状态
        return hidden_states
class LxmertCrossAttentionLayer(nn.Module):
    def __init__(self, config):  
        # 初始化 LxmertCrossAttentionLayer 类
        super().__init__()
        # 初始化 LxmertAttention 层
        self.att = LxmertAttention(config)
        # 初始化 LxmertAttentionOutput 层
        self.output = LxmertAttentionOutput(config)

    def forward(self, input_tensor, ctx_tensor, ctx_att_mask=None, output_attentions=False):
        # 进行前向传播
        output = self.att(input_tensor, ctx_tensor, ctx_att_mask, output_attentions=output_attentions)
        if output_attentions:
            attention_probs = output[1]
        # LxmertAttentionOutput 层的前向传播
        attention_output = self.output(output[0], input_tensor)
        outputs = (attention_output, attention_probs) if output_attentions else (attention_output,)
        return outputs


class LxmertSelfAttentionLayer(nn.Module):
    def __init__(self, config):  
        # 初始化 LxmertSelfAttentionLayer 类
        super().__init__()
        # 初始化 LxmertAttention 层
        self.self = LxmertAttention(config)
        # 初始化 LxmertAttentionOutput 层
        self.output = LxmertAttentionOutput(config)

    def forward(self, input_tensor, attention_mask, output_attentions=False):
        # Self attention attends to itself, thus keys and queries are the same (input_tensor).
        # 进行前向传播
        output = self.self(
            input_tensor,
            input_tensor,
            attention_mask,
            output_attentions=output_attentions,
        )
        if output_attentions:
            attention_probs = output[1]
        # LxmertAttentionOutput 层的前向传播
        attention_output = self.output(output[0], input_tensor)
        outputs = (attention_output, attention_probs) if output_attentions else (attention_output,)
        return outputs


class LxmertIntermediate(nn.Module):
    def __init__(self, config):  
        # 初始化 LxmertIntermediate 类
        super().__init__()
        # 初始化线性层
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        # 初始化激活函数
        self.intermediate_act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states):  
        # 进行前向传播
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class LxmertOutput(nn.Module):
    def __init__(self, config):  
        # 初始化 LxmertOutput 类
        super().__init__()
        # 初始化线性层
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        # 初始化 LayerNormalization 层
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        # 初始化 Dropout 层
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):  
        # 进行前向传播
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class LxmertLayer(nn.Module):
    def __init__(self, config):  
        # 初始化 LxmertLayer 类
        super().__init__()
        # 初始化 LxmertSelfAttentionLayer 层
        self.attention = LxmertSelfAttentionLayer(config)
        # 初始化 LxmertIntermediate 层
        self.intermediate = LxmertIntermediate(config)
        # 初始化 LxmertOutput 层
        self.output = LxmertOutput(config)
    # 定义前向传播方法，接收隐藏状态、注意力掩码和是否输出注意力权重
    def forward(self, hidden_states, attention_mask=None, output_attentions=False):
        # 使用注意力层计算隐藏状态的注意力输出
        outputs = self.attention(hidden_states, attention_mask, output_attentions=output_attentions)
        # 获取注意力输出
        attention_output = outputs[0]
        # 使用中间层处理注意力输出
        intermediate_output = self.intermediate(attention_output)
        # 使用输出层处理中间输出和注意力输出
        layer_output = self.output(intermediate_output, attention_output)
        # 如果需要输出注意力权重，则将其加入到输出中
        outputs = (layer_output,) + outputs[1:]  
        # 返回输出结果
        return outputs
class LxmertXLayer(nn.Module):
    # 定义 LxmertXLayer 类，作为 nn.Module 的子类
    def __init__(self, config):
        # 初始化方法，接收一个配置参数 config
        super().__init__()
        # 调用父类的初始化方法

        # 网络结构定义
        self.visual_attention = LxmertCrossAttentionLayer(config)
        # 定义跨注意力层对象 visual_attention

        self.lang_self_att = LxmertSelfAttentionLayer(config)
        # 定义语言自注意力层对象 lang_self_att

        self.visn_self_att = LxmertSelfAttentionLayer(config)
        # 定义视觉自注意力层对象 visn_self_att

        self.lang_inter = LxmertIntermediate(config)
        # 定义语言中间层对象 lang_inter

        self.lang_output = LxmertOutput(config)
        # 定义语言输出层对象 lang_output

        self.visn_inter = LxmertIntermediate(config)
        # 定义视觉中间层对象 visn_inter

        self.visn_output = LxmertOutput(config)
        # 定义视觉输出层对象 visn_output

    def cross_att(
        self,
        lang_input,
        lang_attention_mask,
        visual_input,
        visual_attention_mask,
        output_x_attentions=False,
    ):
        # 定义跨注意力层的前向传播方法 cross_att
        lang_att_output = self.visual_attention(
            lang_input,
            visual_input,
            ctx_att_mask=visual_attention_mask,
            output_attentions=output_x_attentions,
        )
        # 使用 visual_attention 进行跨注意力操作，得到语言的注意力输出
        visual_att_output = self.visual_attention(
            visual_input,
            lang_input,
            ctx_att_mask=lang_attention_mask,
            output_attentions=False,
        )
        # 使用 visual_attention 进行跨注意力操作，得到视觉的注意力输出
        return lang_att_output, visual_att_output

    def self_att(self, lang_input, lang_attention_mask, visual_input, visual_attention_mask):
        # 定义自注意力层的前向传播方法 self_att
        lang_att_output = self.lang_self_att(lang_input, lang_attention_mask, output_attentions=False)
        # 使用 lang_self_att 进行语言自注意力操作
        visual_att_output = self.visn_self_att(visual_input, visual_attention_mask, output_attentions=False)
        # 使用 visn_self_att 进行视觉自注意力操作
        return lang_att_output[0], visual_att_output[0]

    def output_fc(self, lang_input, visual_input):
        # 定义输出全连接层的前向传播方法 output_fc
        lang_inter_output = self.lang_inter(lang_input)
        # 使用 lang_inter 进行语言中间层操作
        visual_inter_output = self.visn_inter(visual_input)
        # 使用 visn_inter 进行视觉中间层操作

        # Layer output
        lang_output = self.lang_output(lang_inter_output, lang_input)
        # 使用 lang_output 进行语言输出操作
        visual_output = self.visn_output(visual_inter_output, visual_input)
        # 使用 visn_output 进行视觉输出操作

        return lang_output, visual_output

    def forward(
        self,
        lang_feats,
        lang_attention_mask,
        visual_feats,
        visual_attention_mask,
        output_attentions=False,
        ):
            # 使用跨模态注意力机制，传入语言特征、语言注意力掩码、视觉特征、视觉注意力掩码，输出语言注意力输出和视觉注意力输出
            lang_att_output, visual_att_output = self.cross_att(
                lang_input=lang_feats,
                lang_attention_mask=lang_attention_mask,
                visual_input=visual_feats,
                visual_attention_mask=visual_attention_mask,
                output_x_attentions=output_attentions,
            )
            # 从语言注意力输出中获取注意力概率
            attention_probs = lang_att_output[1:]
            # 使用自注意力机制，传入语言注意力输出、语言注意力掩码、视觉注意力输出、视觉注意力掩码
            lang_att_output, visual_att_output = self.self_att(
                lang_att_output[0],
                lang_attention_mask,
                visual_att_output[0],
                visual_attention_mask,
            )

            # 使用输出全连接层，传入语言注意力输出、视觉注意力输出，输出语言输出、视觉输出
            lang_output, visual_output = self.output_fc(lang_att_output, visual_att_output)
            # 返回语言输出、视觉输出、注意力概率（如果有）、如果有输出注意力则输出所有
            return (
                (
                    lang_output,
                    visual_output,
                    attention_probs[0],
                )
                if output_attentions
                else (lang_output, visual_output)
            )
# 定义一个类LxmertVisualFeatureEncoder，继承于nn.Module类
class LxmertVisualFeatureEncoder(nn.Module):
    # 初始化函数，接受config参数
    def __init__(self, config):
        # 根据配置获取特征维度和位置维度
        feat_dim = config.visual_feat_dim
        pos_dim = config.visual_pos_dim

        # 对象特征编码
        self.visn_fc = nn.Linear(feat_dim, config.hidden_size)  # 使用线性层进行特征编码
        self.visn_layer_norm = nn.LayerNorm(config.hidden_size, eps=1e-12)  # 使用LayerNorm进行标准化

        # 盒子位置编码
        self.box_fc = nn.Linear(pos_dim, config.hidden_size)  # 使用线性层进行位置编码
        self.box_layer_norm = nn.LayerNorm(config.hidden_size, eps=1e-12)  # 使用LayerNorm进行标准化

        self.dropout = nn.Dropout(config.hidden_dropout_prob)  # 使用Dropout进行随机失活

    # 前向传播函数，接受visual_feats和visual_pos参数
    def forward(self, visual_feats, visual_pos):
        x = self.visn_fc(visual_feats)  # 对visual_feats进行特征编码
        x = self.visn_layer_norm(x)  # 对编码后的特征进行标准化
        y = self.box_fc(visual_pos)  # 对visual_pos进行位置编码
        y = self.box_layer_norm(y)  # 对编码后的位置进行标准化
        output = (x + y) / 2  # 计算特征和位置编码的平均值作为输出

        output = self.dropout(output)  # 对输出进行随机失活
        return output  # 返回输出结果


class LxmertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        # Obj-level image embedding layer
        self.visn_fc = LxmertVisualFeatureEncoder(config)  # 对象级别图像嵌入层的实例化
        self.config = config  # 保存配置信息

        # Number of layers
        self.num_l_layers = config.l_layers  # 获取L层的数量
        self.num_x_layers = config.x_layers  # 获取X层的数量
        self.num_r_layers = config.r_layers  # 获取R层的数量

        # Layers
        # Using self.layer instead of self.l_layer to support loading BERT weights.
        self.layer = nn.ModuleList([LxmertLayer(config) for _ in range(self.num_l_layers)])  # 创建L层的ModuleList
        self.x_layers = nn.ModuleList([LxmertXLayer(config) for _ in range(self.num_x_layers)])  # 创建X层的ModuleList
        self.r_layers = nn.ModuleList([LxmertLayer(config) for _ in range(self.num_r_layers)])  # 创建R层的ModuleList


    # 前向传播函数，接受多个参数
    def forward(
        self,
        lang_feats,
        lang_attention_mask,
        visual_feats,
        visual_pos,
        visual_attention_mask=None,
        output_attentions=None,
    # 初始化变量
    # 存储图像层的隐藏状态
    vision_hidden_states = ()
    # 存储语言层的隐藏状态
    language_hidden_states = ()
    # 存储图像层的注意力分布（如果需要输出注意力分布）
    vision_attentions = () if output_attentions or self.config.output_attentions else None
    # 存储语言层的注意力分布（如果需要输出注意力分布）
    language_attentions = () if output_attentions or self.config.output_attentions else None
    # 存储跨模态层的注意力分布（如果需要输出注意力分布）
    cross_encoder_attentions = () if output_attentions or self.config.output_attentions else None
    
    # 将视觉特征通过全连接层进行处理
    visual_feats = self.visn_fc(visual_feats, visual_pos)
    
    # 运行语言层
    for layer_module in self.layer:
        # 调用每个语言层模块进行处理，获取语言特征和对应的注意力分布
        l_outputs = layer_module(lang_feats, lang_attention_mask, output_attentions=output_attentions)
        lang_feats = l_outputs[0]
        # 存储当前语言层的隐藏状态
        language_hidden_states = language_hidden_states + (lang_feats,)
        if language_attentions is not None:
            # 存储当前语言层的注意力分布
            language_attentions = language_attentions + (l_outputs[1],)
    
    # 运行关系层
    for layer_module in self.r_layers:
        # 调用每个关系层模块进行处理，获取图像特征和对应的注意力分布
        v_outputs = layer_module(visual_feats, visual_attention_mask, output_attentions=output_attentions)
        visual_feats = v_outputs[0]
        # 存储当前图像层的隐藏状态
        vision_hidden_states = vision_hidden_states + (visual_feats,)
        if vision_attentions is not None:
            # 存储当前图像层的注意力分布
            vision_attentions = vision_attentions + (v_outputs[1],)
    
    # 运行跨模态层
    for layer_module in self.x_layers:
        # 调用每个跨模态层模块进行处理，获取语言特征、图像特征和对应的注意力分布
        x_outputs = layer_module(
            lang_feats,
            lang_attention_mask,
            visual_feats,
            visual_attention_mask,
            output_attentions=output_attentions,
        )
        lang_feats, visual_feats = x_outputs[:2]
        # 存储当前跨模态层的图像特征和语言特征的隐藏状态
        vision_hidden_states = vision_hidden_states + (visual_feats,)
        language_hidden_states = language_hidden_states + (lang_feats,)
        if cross_encoder_attentions is not None:
            # 存储当前跨模态层的注意力分布
            cross_encoder_attentions = cross_encoder_attentions + (x_outputs[2],)
    
    # 存储最终的图像编码器输出，包括隐藏状态和注意力分布（如果需要输出注意力分布）
    visual_encoder_outputs = (
        vision_hidden_states,
        vision_attentions if output_attentions else None,
    )
    # 存储最终的语言编码器输出，包括隐藏状态和注意力分布（如果需要输出注意力分布）
    lang_encoder_outputs = (
        language_hidden_states,
        language_attentions if output_attentions else None,
    )
    
    # 返回编码器的输出，注意力分布（如果需要输出），以元组形式返回
    return (
        visual_encoder_outputs,
        lang_encoder_outputs,
        cross_encoder_attentions if output_attentions else None,
    )
# 定义一个 LxmertPooler 类，继承自 nn.Module 类
class LxmertPooler(nn.Module):
    # 初始化方法，接受参数 config
    def __init__(self, config):
        # 调用父类的初始化方法
        super(LxmertPooler, self).__init__()
        # 创建一个全连接层，输入维度为 config.hidden_size，输出维度为 config.hidden_size
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 创建一个激活函数为 Tanh 的激活层
        self.activation = nn.Tanh()

    # 前向传播方法，接受参数 hidden_states
    def forward(self, hidden_states):
        # 从 hidden_states 中取出第一个 token 对应的隐藏状态
        first_token_tensor = hidden_states[:, 0]
        # 将取出的隐藏状态传入全连接层进行计算
        pooled_output = self.dense(first_token_tensor)
        # 通过激活层处理得到最终的池化输出
        pooled_output = self.activation(pooled_output)
        # 返回池化输出
        return pooled_output


# 定义一个 LxmertPredictionHeadTransform 类，继承自 nn.Module 类
class LxmertPredictionHeadTransform(nn.Module):
    # 初始化方法，接受参数 config
    def __init__(self, config):
        # 调用父类的初始化方法
        super(LxmertPredictionHeadTransform, self).__init__()
        # 创建一个全连接层，输入维度为 config.hidden_size，输出维度为 config.hidden_size
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 获取对应激活函数的函数对象
        self.transform_act_fn = ACT2FN[config.hidden_act]
        # 创建一个 LayerNorm 层，输入维度为 config.hidden_size
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=1e-12)

    # 前向传播方法，接受参数 hidden_states
    def forward(self, hidden_states):
        # 通过全连接层计算得到新的隐藏状态
        hidden_states = self.dense(hidden_states)
        # 通过激活函数处理隐藏状态
        hidden_states = self.transform_act_fn(hidden_states)
        # 通过 LayerNorm 处理隐藏状态
        hidden_states = self.LayerNorm(hidden_states)
        # 返回处理后的隐藏状态
        return hidden_states


# 定义一个 LxmertLMPredictionHead 类，继承自 nn.Module 类
class LxmertLMPredictionHead(nn.Module):
    # 初始化方法，接受参数 config 和 lxmert_model_embedding_weights
    def __init__(self, config, lxmert_model_embedding_weights):
        # 调用父类的初始化方法
        super(LxmertLMPredictionHead, self).__init__()
        # 创建一个 LxmertPredictionHeadTransform 对象
        self.transform = LxmertPredictionHeadTransform(config)

        # 输出权重与输入嵌入层相同，但每个 token 都有一个输出偏置
        # 创建一个���性层，输入维度为 lxmert_model_embedding_weights 的列数，输出维度为 lxmert_model_embedding_weights 的行数，不使用偏置
        self.decoder = nn.Linear(
            lxmert_model_embedding_weights.size(1),
            lxmert_model_embedding_weights.size(0),
            bias=False,
        )
        # 将 decoder 层的权重设置为 lxmert_model_embedding_weights
        self.decoder.weight = lxmert_model_embedding_weights
        # 创建一个用于存储偏置的参数
        self.bias = nn.Parameter(torch.zeros(lxmert_model_embedding_weights.size(0)))

    # 前向传播方法，接受参数 hidden_states
    def forward(self, hidden_states):
        # 通过 transform 处理隐藏状态
        hidden_states = self.transform(hidden_states)
        # 通过 decoder 层计算得到预测结果并加上偏置
        hidden_states = self.decoder(hidden_states) + self.bias
        # 返回处理后的隐藏状态
        return hidden_states


# 定义一个 LxmertVisualAnswerHead 类，继承自 nn.Module 类
class LxmertVisualAnswerHead(nn.Module):
    # 初始化方法，接受参数 config 和 num_labels
    def __init__(self, config, num_labels):
        # 调用父类的初始化方法
        super().__init__()
        # 获取隐藏层维度
        hid_dim = config.hidden_size
        # 创建一个包含多个层的线性层对象，包括两个全连接层、GeLU 激活函数、LayerNorm 层和最终的全连接层
        self.logit_fc = nn.Sequential(
            nn.Linear(hid_dim, hid_dim * 2),
            GeLU(),
            nn.LayerNorm(hid_dim * 2, eps=1e-12),
            nn.Linear(hid_dim * 2, num_labels),
        )

    # 前向传播方法，接受参数 hidden_states
    def forward(self, hidden_states):
        # 通过 logit_fc 进行前向传播得到预测结果
        return self.logit_fc(hidden_states)


# 定义一个 LxmertVisualObjHead 类，继承自 nn.Module 类
    # 初始化函数，接收一个配置参数
    def __init__(self, config):
        # 调用父类的初始化函数
        super().__init__()
        # 创建一个 LxmertPredictionHeadTransform 对象并赋值给 self.transform
        self.transform = LxmertPredictionHeadTransform(config)
        # 决定是否使用视觉损失
        visual_losses = {}
        # 如果配置中包含视觉对象损失，则创建对应的损失字典并添加到 visual_losses 中
        if config.visual_obj_loss:
            visual_losses["obj"] = {"shape": (-1,), "num": config.num_object_labels}
        # 如果配置中包含视觉属性损失，则创建对应的损失字典并添加到 visual_losses 中
        if config.visual_attr_loss:
            visual_losses["attr"] = {"shape": (-1,), "num": config.num_attr_labels}
        # 如果配置中包含视觉特征损失，则创建对应的损失字典并添加到 visual_losses 中
        if config.visual_feat_loss:
            visual_losses["feat"] = {
                "shape": (-1, config.visual_feat_dim),
                "num": config.visual_feat_dim,
            }
        # 将视觉损失字典赋值给 self.visual_losses
        self.visual_losses = visual_losses

        # 输出权重与输入嵌入相同，但每个令牌都有一个输出偏差
        # 创建一个包含多个 nn.Linear 对象的字典，并赋值给 self.decoder_dict
        self.decoder_dict = nn.ModuleDict(
            {key: nn.Linear(config.hidden_size, self.visual_losses[key]["num"]) for key in self.visual_losses}
        )

    # 前向传播函数
    def forward(self, hidden_states):
        # 使用 self.transform 对隐藏状态进行转换
        hidden_states = self.transform(hidden_states)
        output = {}
        # 遍历视觉损失字典中的每一个损失类型
        for key in self.visual_losses:
            # 使用 self.decoder_dict 中对应的线性层处理隐藏状态，并将结果添加到 output 字典中
            output[key] = self.decoder_dict[key](hidden_states)
        # 返回输出字典
        return output
class LxmertPreTrainingHeads(nn.Module):
    # 定义 Lxmert 模型的预训练头部
    def __init__(self, config, lxmert_model_embedding_weights):
        # 初始化方法，接收配置和嵌入权重
        super(LxmertPreTrainingHeads, self).__init__()
        # 调用父类的初始化方法
        self.predictions = LxmertLMPredictionHead(config, lxmert_model_embedding_weights)
        # 创建预测输出层对象
        self.seq_relationship = nn.Linear(config.hidden_size, 2)
        # 创建序列关系分类层对象

    def forward(self, sequence_output, pooled_output):
        # 前向传播方法，接收序列输出和池化后的输出
        prediction_scores = self.predictions(sequence_output)
        # 获取预测分数
        seq_relationship_score = self.seq_relationship(pooled_output)
        # 获取序列关系得分
        return prediction_scores, seq_relationship_score
        # 返回预测分数和序列关系得分


class LxmertPreTrainedModel(PreTrainedModel):
    # Lxmert 预训练模型类
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    # 一个抽象类，用于处理权重初始化、预训练模型的下载和加载的简单接口

    config_class = LxmertConfig
    # 配置类为 LxmertConfig
    load_tf_weights = load_tf_weights_in_lxmert
    # 加载 TensorFlow 权重
    base_model_prefix = "lxmert"
    # 基础模型前缀为 "lxmert"

    def _init_weights(self, module):
        # 初始化权重方法
        """Initialize the weights"""
        # 初始化权重
        if isinstance(module, nn.Linear):
            # 若模块是线性层
            # Slightly different from the TF version which uses truncated_normal for initialization
            # 略有不同于 TensorFlow 版本，TensorFlow 使用截尾正态分布进行初始化
            # cf https://github.com/pytorch/pytorch/pull/5617
            # 参考链接
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 初始化权重的数据
            if module.bias is not None:
                # 若有偏置项
                module.bias.data.zero_()
                # 偏置项数据初始化为零
        elif isinstance(module, nn.Embedding):
            # 若模块是嵌入层
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 初始化权重的数据
            if module.padding_idx is not None:
                # 若有填充索引
                module.weight.data[module.padding_idx].zero_()
                # 填充索引对应的权重初始化为零
        elif isinstance(module, nn.LayerNorm):
            # 若模块是 LayerNorm 层
            module.bias.data.zero_()
            # 偏置项初始化为零
            module.weight.data.fill_(1.0)
            # 权重项初始化为 1.0

LXMERT_START_DOCSTRING = r"""

    The LXMERT model was proposed in [LXMERT: Learning Cross-Modality Encoder Representations from
    Transformers](https://arxiv.org/abs/1908.07490) by Hao Tan and Mohit Bansal. It's a vision and language transformer
    model, pretrained on a variety of multi-modal datasets comprising of GQA, VQAv2.0, MSCOCO captions, and Visual
    genome, using a combination of masked language modeling, region of interest feature regression, cross entropy loss
    for question answering attribute prediction, and object tag prediction.

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.
    # 定义一个"配置"参数，其类型为 LxmertConfig 类型
    # LxmertConfig 类包含了模型的所有参数配置
    Parameters:
        config ([`LxmertConfig`]): Model configuration class with all the parameters of the model.
            # 使用配置文件初始化模型只会加载模型的配置信息，
            # 而不会加载与模型相关联的权重。
            # 要加载模型权重，需要使用 [`~PreTrainedModel.from_pretrained`] 方法。
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

# 定义 LXMERT 模型，输出未经特定头部处理的原始隐藏状态
# LXMERT_START_DOCSTRING 中包含 LXMERT 模型的详细信息

@add_start_docstrings(
    "The bare Lxmert Model transformer outputting raw hidden-states without any specific head on top.",
    LXMERT_START_DOCSTRING,
)
class LxmertModel(LxmertPreTrainedModel):
    # 初始化函数
    def __init__(self, config):
        # 调用父类初始化函数
        super().__init__(config)
        # 实例化 LxmertEmbeddings 对象
        self.embeddings = LxmertEmbeddings(config)
        # 实例化 LxmertEncoder 对象
        self.encoder = LxmertEncoder(config)
        # 实例化 LxmertPooler 对象
        self.pooler = LxmertPooler(config)
        # 初始化权重并应用最终处理
        self.post_init()

    # 获取输入嵌入
    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    # 设置输入嵌入
    def set_input_embeddings(self, new_embeddings):
        self.embeddings.word_embeddings = new_embeddings

    # 前向传播函数，包含参数和返回类型的描述信息
    # 同时包含代码示例的描述信息
    @add_start_docstrings_to_model_forward(LXMERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=LxmertModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        visual_feats: Optional[torch.FloatTensor] = None,
        visual_pos: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        visual_attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
)

# 定义具有指定预训练头的 LXMERT 模型
# LXMERT_START_DOCSTRING 包含了 LXMERT 模型的详细信息
class LxmertForPreTraining(LxmertPreTrainedModel):
    _tied_weights_keys = ["cls.predictions.decoder.weight"]
    # 定义 LxmertPreTraining 类，继承自 nn.Module
    def __init__(self, config):
        # 调用父类的构造方法
        super().__init__(config)
        # 保存配置参数
        self.config = config
        # 设置问答标签数量
        self.num_qa_labels = config.num_qa_labels
        # 设置视觉损失归一化因子
        self.visual_loss_normalizer = config.visual_loss_normalizer
    
        # 是否使用 mask language model 预训练任务
        self.task_mask_lm = config.task_mask_lm
        # 是否使用目标预测预训练任务
        self.task_obj_predict = config.task_obj_predict
        # 是否使用匹配预训练任务
        self.task_matched = config.task_matched
        # 是否使用问答预训练任务
        self.task_qa = config.task_qa
    
        # 创建 LxmertModel 实例
        self.lxmert = LxmertModel(config)
    
        # 创建预训练头部
        self.cls = LxmertPreTrainingHeads(config, self.lxmert.embeddings.word_embeddings.weight)
        # 如果使用目标预测预训练任务，创建相应的头部
        if self.task_obj_predict:
            self.obj_predict_head = LxmertVisualObjHead(config)
        # 如果使用问答预训练任务，创建相应的头部
        if self.task_qa:
            self.answer_head = LxmertVisualAnswerHead(config, self.num_qa_labels)
    
        # 进行权重初始化
        self.post_init()
    
        # 定义损失函数
        self.loss_fcts = {
            "l2": SmoothL1Loss(reduction="none"),
            "visual_ce": CrossEntropyLoss(reduction="none"),
            "ce": CrossEntropyLoss(),
        }
    
        # 定义视觉损失
        visual_losses = {}
        # 如果使用目标预测视觉损失
        if config.visual_obj_loss:
            visual_losses["obj"] = {
                "shape": (-1,),
                "num": config.num_object_labels,
                "loss": "visual_ce",
            }
        # 如果使用属性预测视觉损失
        if config.visual_attr_loss:
            visual_losses["attr"] = {
                "shape": (-1,),
                "num": config.num_attr_labels,
                "loss": "visual_ce",
            }
        # 如果使用特征回归视觉损失
        if config.visual_feat_loss:
            visual_losses["feat"] = {
                "shape": (-1, config.visual_feat_dim),
                "num": config.visual_feat_dim,
                "loss": "l2",
            }
        # 保存视觉损失配置
        self.visual_losses = visual_losses
    # 调整问答任务标签数量的方法
    def resize_num_qa_labels(self, num_labels):
        """
        从提供的新线性层构建调整大小的问答任务线性层模块。增加大小将添加新初始化的权重。减少大小将从末尾删除权重
    
        参数:
            num_labels (`int`, *optional*):
                线性层权重矩阵中新的标签数量。增加大小将在末尾添加新初始化的权重。减少大小将从末尾删除权重。如果未提供或为 `None`，只返回模型的问答标签 ``torch.nn.Linear``` 模块的指针，而不执行任何操作。
    
        返回:
            `torch.nn.Linear`: 调整后的线性层或旧线性层的指针
        """
        # 获取当前的问答任务逻辑层
        cur_qa_logit_layer = self.get_qa_logit_layer()
        # 如果没有提供新的标签数量或当前的问答任务逻辑层为 None，则返回
        if num_labels is None or cur_qa_logit_layer is None:
            return
        # 调整问答任务标签数量
        new_qa_logit_layer = self._resize_qa_labels(num_labels)
        # 更新配置和标签数量
        self.config.num_qa_labels = num_labels
        self.num_qa_labels = num_labels
    
        return new_qa_logit_layer
    
    # 实际调整问答任务标签数量的内部方法
    def _resize_qa_labels(self, num_labels):
        # 获取当前的问答任务逻辑层
        cur_qa_logit_layer = self.get_qa_logit_layer()
        # 创建调整后的问答任务逻辑层
        new_qa_logit_layer = self._get_resized_qa_labels(cur_qa_logit_layer, num_labels)
        # 设置调整后的问答任务逻辑层
        self._set_qa_logit_layer(new_qa_logit_layer)
        # 返回调整后的问答任务逻辑层
        return self.get_qa_logit_layer()
    
    # 获取问答任务逻辑层的方法
    def get_qa_logit_layer(self) -> nn.Module:
        """
        返回生成问答任务逻辑输出的线性层。
    
        返回:
            `nn.Module`: 一个将问答预测隐藏状态映射到输出的 PyTorch 模块，如果 LXMERT 没有视觉答案 head，则返回 `None`。
        """
        # 如果有 answer_head 属性，则返回其最后一个线性层
        if hasattr(self, "answer_head"):
            return self.answer_head.logit_fc[-1]
    
    # 设置问答任务逻辑层的方法
    def _set_qa_logit_layer(self, qa_logit_layer):
        # 设置 answer_head 的最后一个线性层
        self.answer_head.logit_fc[-1] = qa_logit_layer
    # 根据当前 QA 标签层和目标标签数量获取调整后的 QA 标签层
    def _get_resized_qa_labels(self, cur_qa_logit_layer, num_labels):
        # 如果目标标签数量为 None，则返回当前 QA 标签层
        if num_labels is None:
            return cur_qa_logit_layer

        # 获取当前 QA 标签层的标签数量和隐藏维度
        cur_qa_labels, hidden_dim = cur_qa_logit_layer.weight.size()
        # 如果当前标签数量等于目标标签数量，则返回当前 QA 标签层
        if cur_qa_labels == num_labels:
            return cur_qa_logit_layer

        # 构建新的线性输出层
        if getattr(cur_qa_logit_layer, "bias", None) is not None:
            new_qa_logit_layer = nn.Linear(hidden_dim, num_labels)
        else:
            new_qa_logit_layer = nn.Linear(hidden_dim, num_labels, bias=False)

        # 将新的 QA 标签层移动到与当前权重相同的设备上
        new_qa_logit_layer.to(cur_qa_logit_layer.weight.device)

        # 初始化所有新标签的权重
        self._init_weights(new_qa_logit_layer)

        # 从先前的权重复制标签
        num_labels_to_copy = min(cur_qa_labels, num_labels)
        new_qa_logit_layer.weight.data[:num_labels_to_copy, :] = cur_qa_logit_layer.weight.data[:num_labels_to_copy, :]
        if getattr(cur_qa_logit_layer, "bias", None) is not None:
            new_qa_logit_layer.bias.data[:num_labels_to_copy] = cur_qa_logit_layer.bias.data[:num_labels_to_copy]

        return new_qa_logit_layer

    # 用于模型前向传播的函数，添加了 LXMERT 输入文档字符串的起始部分
    @add_start_docstrings_to_model_forward(LXMERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    # 将返回文档字符串中的输出类型替换为 LxmertForPreTrainingOutput，配置类替换为 _CONFIG_FOR_DOC
    @replace_return_docstrings(output_type=LxmertForPreTrainingOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        visual_feats: Optional[torch.FloatTensor] = None,
        visual_pos: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        visual_attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        obj_labels: Optional[Dict[str, Tuple[torch.FloatTensor, torch.FloatTensor]]] = None,
        matched_label: Optional[torch.LongTensor] = None,
        ans: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
```py  
# 使用 add_start_docstrings() 装饰器为 LxmertForQuestionAnswering 类添加文档字符串，说明该类是带有视觉问答头的 Lxmert 模型，用于下游 QA 任务
# 调用父类的初始化方法，传入配置参数
class LxmertForQuestionAnswering(LxmertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        # 保存配置参数
        self.config = config
        # 保存 QA 标签数量
        self.num_qa_labels = config.num_qa_labels
        # 保存视觉损失标准化器
        self.visual_loss_normalizer = config.visual_loss_normalizer

        # 创建 Lxmert 模型
        self.lxmert = LxmertModel(config)

        # 创建视觉问答头
        self.answer_head = LxmertVisualAnswerHead(config, self.num_qa_labels)

        # 初始化权重
        # 初始化权重并应用最终处理
        self.post_init()

        # 损失函数
        self.loss = CrossEntropyLoss()

    def resize_num_qa_labels(self, num_labels):
        """
        构建一个调整的问题回答线性层模块，增加大小将会添加新初始化的权重，减小大小将会移除权重

        Args:
            num_labels (`int`, *optional*):
                线性层权重矩阵中的新标签数量。增加大小将会在末尾添加新初始化的权重。减小大小将会从末尾移除权重。如果未提供或为 `None`，则仅返回模型的 qa 标签 ``torch.nn.Linear``` 模块的指针而不执行任何操作。

        Return:
            `torch.nn.Linear`: 调整大小后的线性层的指针或旧线性层
        """
        
        cur_qa_logit_layer = self.get_qa_logit_layer()
        if num_labels is None or cur_qa_logit_layer is None:
            return
        new_qa_logit_layer = self._resize_qa_labels(num_labels)
        self.config.num_qa_labels = num_labels
        self.num_qa_labels = num_labels

        return new_qa_logit_layer

    def _resize_qa_labels(self, num_labels):
        cur_qa_logit_layer = self.get_qa_logit_layer()
        new_qa_logit_layer = self._get_resized_qa_labels(cur_qa_logit_layer, num_labels)
        self._set_qa_logit_layer(new_qa_logit_layer)
        return self.get_qa_logit_layer()

    def get_qa_logit_layer(self) -> nn.Module:
        """
        返回用于产生问题回答对数的线性层

        Returns:
            `nn.Module`: 一个将问题回答预测隐藏状态映射的 torch 模块。`None`: 如果 Lxmert 没有视觉问答头。
        """

        if hasattr(self, "answer_head"):
            return self.answer_head.logit_fc[-1]

    def _set_qa_logit_layer(self, qa_logit_layer):
        self.answer_head.logit_fc[-1] = qa_logit_layer
    # 获取调整大小后的问答标签
    def _get_resized_qa_labels(self, cur_qa_logit_layer, num_labels):
        # 如果标签数量为None，直接返回当前的问答逻辑层
        if num_labels is None:
            return cur_qa_logit_layer

        # 获取当前问答标签的数量和隐藏维度
        cur_qa_labels, hidden_dim = cur_qa_logit_layer.weight.size()
        # 如果当前标签数量等于要求的标签数量，直接返回当前的问答逻辑层
        if cur_qa_labels == num_labels:
            return cur_qa_logit_layer

        # 构建新的线性输出层
        # 如果当前的逻辑层有偏置项，则创建新的线性层含有偏置项
        if getattr(cur_qa_logit_layer, "bias", None) is not None:
            new_qa_logit_layer = nn.Linear(hidden_dim, num_labels)
        # 如果当前的逻辑层没有偏置项，则创建新的线性层不含偏置项
        else:
            new_qa_logit_layer = nn.Linear(hidden_dim, num_labels, bias=False)

        # 将新的问答逻辑层移到与当前问答逻辑层相同的设备上
        new_qa_logit_layer.to(cur_qa_logit_layer.weight.device)

        # 初始化所有的新标签
        self._init_weights(new_qa_logit_layer)

        # 从之前的权重中复制标签
        num_labels_to_copy = min(cur_qa_labels, num_labels)
        new_qa_logit_layer.weight.data[:num_labels_to_copy, :] = cur_qa_logit_layer.weight.data[:num_labels_to_copy, :]
        # 如果当前逻辑层有偏置项，则复制偏置项
        if getattr(cur_qa_logit_layer, "bias", None) is not None:
            new_qa_logit_layer.bias.data[:num_labels_to_copy] = cur_qa_logit_layer.bias.data[:num_labels_to_copy]

        return new_qa_logit_layer

    # 定义前向传播函数
    @add_start_docstrings_to_model_forward(LXMERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=LxmertForQuestionAnsweringOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        visual_feats: Optional[torch.FloatTensor] = None,
        visual_pos: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        visual_attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    # 定义一个方法，参数为input_ids, visual_feats, visual_pos, token_type_ids, attention_mask, visual_attention_mask, inputs_embeds, output_hidden_states, output_attentions, return_dict, labels
    def forward(
        self,
        input_ids,
        visual_feats,
        visual_pos,
        token_type_ids,
        attention_mask,
        visual_attention_mask,
        inputs_embeds,
        output_hidden_states,
        output_attentions,
        return_dict,
        labels
    ) -> Union[LxmertForQuestionAnsweringOutput, Tuple[torch.FloatTensor]]:
        r"""
        labels (`Torch.Tensor` of shape `(batch_size)`, *optional`):
            A one-hot representation of the correct answer
        """
        # 如果return_dict为None，使用self.config.use_return_dict的值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    
        # 调用lxmert方法，传入参数并获取输出
        lxmert_output = self.lxmert(
            input_ids=input_ids,
            visual_feats=visual_feats,
            visual_pos=visual_pos,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            visual_attention_mask=visual_attention_mask,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            return_dict=return_dict,
        )
    
        # 从lxmert输出中获取pooled_output
        pooled_output = lxmert_output[2]
        # 使用answer_head方法对pooled_output进行处理，得到answer_score
        answer_score = self.answer_head(pooled_output)
        # 初始化loss为None
        loss = None
        # 如果labels不为None，使用self.loss计算loss
        if labels is not None:
            loss = self.loss(answer_score.view(-1, self.num_qa_labels), labels.view(-1))
    
        # 如果return_dict为False，返回answer_score和lxmert_output的其余部分
        if not return_dict:
            output = (answer_score,) + lxmert_output[3:]
            return (loss,) + output if loss is not None else output
    
        # 返回LxmertForQuestionAnsweringOutput对象，包含loss，answer_score，language_hidden_states，vision_hidden_states，language_attentions，vision_attentions，cross_encoder_attentions
        return LxmertForQuestionAnsweringOutput(
            loss=loss,
            question_answering_score=answer_score,
            language_hidden_states=lxmert_output.language_hidden_states,
            vision_hidden_states=lxmert_output.vision_hidden_states,
            language_attentions=lxmert_output.language_attentions,
            vision_attentions=lxmert_output.vision_attentions,
            cross_encoder_attentions=lxmert_output.cross_encoder_attentions,
        )
```