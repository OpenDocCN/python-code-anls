# `.\models\qdqbert\modeling_qdqbert.py`

```
# coding=utf-8
# Copyright 2021 NVIDIA Corporation and The HuggingFace Team.
# Copyright (c) 2018-2021, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
PyTorch QDQBERT model.
"""


import math  # 导入数学库
import os  # 导入操作系统功能库
import warnings  # 导入警告模块
from typing import Dict, List, Optional, Tuple, Union  # 导入类型提示相关模块

import torch  # 导入PyTorch库
import torch.utils.checkpoint  # 导入PyTorch的checkpoint功能
from torch import nn  # 导入PyTorch的神经网络模块
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss  # 导入损失函数

from ...activations import ACT2FN  # 导入激活函数
from ...modeling_outputs import (  # 导入模型输出相关类
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    NextSentencePredictorOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from ...modeling_utils import PreTrainedModel  # 导入预训练模型基类
from ...pytorch_utils import (  # 导入PyTorch工具函数
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)
from ...utils import (  # 导入通用工具函数
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_pytorch_quantization_available,
    logging,
    replace_return_docstrings,
    requires_backends,
)
from .configuration_qdqbert import QDQBertConfig  # 导入QDQBERT模型配置

logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器

# soft dependency
if is_pytorch_quantization_available():  # 检查是否支持PyTorch量化
    try:
        from pytorch_quantization import nn as quant_nn  # 导入PyTorch量化模块
        from pytorch_quantization.nn.modules.tensor_quantizer import TensorQuantizer  # 导入量化张量模块
    except OSError:
        logger.error(
            "QDQBERT model are not usable since `pytorch_quantization` can't be loaded. Please try to reinstall it"
            " following the instructions here:"
            " https://github.com/NVIDIA/TensorRT/tree/master/tools/pytorch-quantization."
        )

_CHECKPOINT_FOR_DOC = "google-bert/bert-base-uncased"  # 定义用于文档的检查点名称
_CONFIG_FOR_DOC = "QDQBertConfig"  # 定义用于文档的配置名称

QDQBERT_PRETRAINED_MODEL_ARCHIVE_LIST = [  # 预训练模型存档列表
    "google-bert/bert-base-uncased",
    # See all BERT models at https://huggingface.co/models?filter=bert
]


def load_tf_weights_in_qdqbert(model, tf_checkpoint_path):
    """Load tf checkpoints in a pytorch model."""
    try:
        import re  # 导入正则表达式模块
        import numpy as np  # 导入NumPy库
        import tensorflow as tf  # 导入TensorFlow库
    except ImportError:
        logger.error(
            "Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions."
        )
        raise
    tf_path = os.path.abspath(tf_checkpoint_path)
    # 使用给定的 TensorFlow 检查点路径获取其绝对路径
    logger.info(f"Converting TensorFlow checkpoint from {tf_path}")
    # 记录日志，指示正在转换 TensorFlow 检查点

    # 从 TensorFlow 模型中加载权重变量
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []

    # 遍历初始化的变量名和形状
    for name, shape in init_vars:
        logger.info(f"Loading TF weight {name} with shape {shape}")
        # 记录日志，指示正在加载 TensorFlow 权重，并记录其形状
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array)

    # 遍历加载的变量名和数组
    for name, array in zip(names, arrays):
        # 将变量名按斜杠划分为子路径
        name = name.split("/")

        # 跳过特定的变量名，这些变量不需要在预训练模型中使用
        if any(
            n in ["adam_v", "adam_m", "AdamWeightDecayOptimizer", "AdamWeightDecayOptimizer_1", "global_step"]
            for n in name
        ):
            logger.info(f"Skipping {'/'.join(name)}")
            continue
        
        # 设置指针初始位置为模型对象
        pointer = model
        
        # 遍历变量名的各个部分
        for m_name in name:
            # 如果变量名匹配字母加下划线加数字的模式，则按下划线划分
            if re.fullmatch(r"[A-Za-z]+_\d+", m_name):
                scope_names = re.split(r"_(\d+)", m_name)
            else:
                scope_names = [m_name]
            
            # 根据变量名的首部设置指针指向相应的模型部分
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
                    logger.info(f"Skipping {'/'.join(name)}")
                    continue
            
            # 如果变量名有多个部分，则继续在模型中深入
            if len(scope_names) >= 2:
                num = int(scope_names[1])
                pointer = pointer[num]
        
        # 如果变量名以 "_embeddings" 结尾，则将指针指向权重部分
        if m_name[-11:] == "_embeddings":
            pointer = getattr(pointer, "weight")
        elif m_name == "kernel":
            array = np.transpose(array)  # 转置数组（针对 kernel 变量）

        # 检查指针和加载的数组形状是否匹配，如果不匹配则抛出异常
        try:
            if pointer.shape != array.shape:
                raise ValueError(f"Pointer shape {pointer.shape} and array shape {array.shape} mismatched")
        except AssertionError as e:
            e.args += (pointer.shape, array.shape)
            raise
        
        # 记录日志，指示初始化 PyTorch 权重
        logger.info(f"Initialize PyTorch weight {name}")
        # 将加载的 NumPy 数组转换为 PyTorch 的 Tensor，并赋值给指针
        pointer.data = torch.from_numpy(array)

    # 返回转换后的 PyTorch 模型
    return model
# Copied from transformers.models.bert.modeling_bert.BertEmbeddings with Bert -> QDQBert
# 定义了一个名为 QDQBertEmbeddings 的类，用于构建从词嵌入、位置嵌入和标记类型嵌入生成的嵌入向量。

class QDQBertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""
    # 类的初始化函数，接受一个 config 参数
    def __init__(self, config):
        super().__init__()
        # 词嵌入层，使用 nn.Embedding 创建，参数为词汇表大小、隐藏层大小和填充标记索引
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        # 位置嵌入层，使用 nn.Embedding 创建，参数为最大位置嵌入数和隐藏层大小
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        # 标记类型嵌入层，使用 nn.Embedding 创建，参数为标记类型数和隐藏层大小
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # LayerNorm 层，用于归一化隐藏层输出，参数为隐藏层大小和层标准化的 epsilon 值
        # 这里的 LayerNorm 命名不使用蛇形命名法，以保持与 TensorFlow 模型变量名称的一致性，可以加载任何 TensorFlow 检查点文件
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # Dropout 层，用于随机失活以防止过拟合，参数为隐藏层的丢弃概率
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # position_embedding_type 属性，默认为 "absolute"，表示位置嵌入类型为绝对位置编码
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        # 注册一个持久化张量 "position_ids"，其值为从 0 到 max_position_embeddings-1 的序列张量，形状为 (1, max_position_embeddings)
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )
        # 注册一个持久化张量 "token_type_ids"，其值为全零的张量，形状与 "position_ids" 相同，数据类型为长整型
        self.register_buffer(
            "token_type_ids", torch.zeros(self.position_ids.size(), dtype=torch.long), persistent=False
        )

    # 前向传播函数，接受多个输入参数，并返回嵌入向量
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values_length: int = 0,
        # 省略了函数剩余部分的参数和逻辑，不在这里进行注释
        # 定义函数forward的输入参数及其类型注解，返回torch.Tensor类型的张量
        if input_ids is not None:
            # 如果input_ids不为None，则获取其形状（尺寸）
            input_shape = input_ids.size()
        else:
            # 如果input_ids为None，则获取inputs_embeds的形状，但去掉最后一维
            input_shape = inputs_embeds.size()[:-1]

        # 获取序列长度，即input_shape的第二个维度
        seq_length = input_shape[1]

        # 如果position_ids为None，则从self.position_ids中选择对应长度的位置编码
        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]

        # 设置token_type_ids为构造函数中注册的缓冲区，通常是全零向量，在模型追踪时帮助解决不传递token_type_ids的问题
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                # 如果self中有token_type_ids属性，则使用其缓冲区中的值，并进行扩展以匹配input_shape
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                # 否则，创建全零的token_type_ids张量，与input_shape相同的形状
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        # 如果inputs_embeds为None，则通过word_embeddings层获取input_ids的嵌入表示
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        
        # 使用token_type_ids获取token_type的嵌入表示
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        # 将input embeddings与token_type embeddings相加，得到最终的嵌入表示
        embeddings = inputs_embeds + token_type_embeddings

        # 如果position_embedding_type为"absolute"，则加上位置编码的嵌入表示
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        
        # 对嵌入表示进行LayerNorm归一化
        embeddings = self.LayerNorm(embeddings)
        
        # 对归一化后的嵌入表示进行dropout处理
        embeddings = self.dropout(embeddings)
        
        # 返回最终的嵌入表示张量
        return embeddings
# 定义一个名为 QDQBertSelfOutput 的类，继承自 nn.Module
class QDQBertSelfOutput(nn.Module):
    # 初始化方法，接收一个 config 参数
    def __init__(self, config):
        super().__init__()
        
        # 创建一个 QuantLinear 对象，用于量化线性层的输入和输出
        self.dense = quant_nn.QuantLinear(config.hidden_size, config.hidden_size)
        
        # 创建一个 LayerNorm 层，用于归一化隐藏状态的特征
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # 创建一个 Dropout 层，用于随机失活隐藏状态的特征
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # 创建一个用于量化局部输入的 TensorQuantizer 对象
        self.add_local_input_quantizer = TensorQuantizer(quant_nn.QuantLinear.default_quant_desc_input)
        
        # 创建一个用于量化残差输入的 TensorQuantizer 对象
        self.add_residual_input_quantizer = TensorQuantizer(quant_nn.QuantLinear.default_quant_desc_input)
    # 定义模型的前向传播函数，接收隐藏状态和输入张量作为参数
    def forward(self, hidden_states, input_tensor):
        # 将隐藏状态通过全连接层 dense 进行线性变换
        hidden_states = self.dense(hidden_states)
        # 对线性变换后的隐藏状态进行 dropout 操作，以防止过拟合
        hidden_states = self.dropout(hidden_states)
        # 对输入进行局部加法的量化处理
        add_local = self.add_local_input_quantizer(hidden_states)
        # 对输入张量进行残差加法的量化处理
        add_residual = self.add_residual_input_quantizer(input_tensor)
        # 将量化后的局部加法和残差加法结果进行 LayerNorm 归一化
        hidden_states = self.LayerNorm(add_local + add_residual)
        # 返回处理后的隐藏状态作为输出
        return hidden_states
# 基于 transformers.models.bert.modeling_bert.BertAttention 更改为 QDQBert
class QDQBertAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 初始化自注意力层和自注意力输出层
        self.self = QDQBertSelfAttention(config)
        self.output = QDQBertSelfOutput(config)
        # 存储被修剪的注意力头的集合
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        # 查找可修剪的注意力头并返回索引
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # 修剪线性层
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # 更新超参数并存储修剪的注意力头
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

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
        # 执行自注意力计算
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        # 使用自注意力输出层处理自注意力结果和隐藏状态
        attention_output = self.output(self_outputs[0], hidden_states)
        # 如果需要输出注意力，将其添加到输出中
        outputs = (attention_output,) + self_outputs[1:]
        return outputs


class QDQBertIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 量化线性层
        self.dense = quant_nn.QuantLinear(config.hidden_size, config.intermediate_size)
        # 初始化中间激活函数
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        # 通过量化线性层处理隐藏状态
        hidden_states = self.dense(hidden_states)
        # 应用中间激活函数
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class QDQBertOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Quantize Linear layer
        # 使用量化的神经网络层来定义一个线性层，输入大小为config.intermediate_size，输出大小为config.hidden_size
        self.dense = quant_nn.QuantLinear(config.intermediate_size, config.hidden_size)
        
        # Layer normalization 层，对隐藏状态进行归一化处理
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # Dropout 层，以config.hidden_dropout_prob的概率对输入进行随机置零，用于防止过拟合
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # Quantize the inputs to the residual add
        # 对残差加法的输入进行量化处理，使用默认的输入量化描述符
        self.add_local_input_quantizer = TensorQuantizer(quant_nn.QuantLinear.default_quant_desc_input)
        self.add_residual_input_quantizer = TensorQuantizer(quant_nn.QuantLinear.default_quant_desc_input)

    def forward(self, hidden_states, input_tensor):
        # 将隐藏状态传入量化的线性层进行处理
        hidden_states = self.dense(hidden_states)
        
        # 对处理后的隐藏状态进行dropout操作，以减少过拟合
        hidden_states = self.dropout(hidden_states)
        
        # Quantize the inputs to the residual add
        # 对残差加法的输入进行量化处理
        add_local = self.add_local_input_quantizer(hidden_states)
        add_residual = self.add_residual_input_quantizer(input_tensor)
        
        # 对量化后的本地加法和残差加法进行 Layer normalization 处理
        hidden_states = self.LayerNorm(add_local + add_residual)
        
        # 返回处理后的隐藏状态
        return hidden_states
# 根据 transformers.models.bert.modeling_bert.BertLayer 修改为 QDQBertLayer，是 QDQ 模型的 Bert 层
class QDQBertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 设置序列长度维度为 1
        self.seq_len_dim = 1
        # 初始化 QDQBertAttention 层
        self.attention = QDQBertAttention(config)
        # 标记是否为解码器模型
        self.is_decoder = config.is_decoder
        # 标记是否添加跨层注意力
        self.add_cross_attention = config.add_cross_attention
        # 如果添加了跨层注意力
        if self.add_cross_attention:
            # 如果不是解码器模型，抛出异常
            if not self.is_decoder:
                raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
            # 初始化跨层注意力 QDQBertAttention 层
            self.crossattention = QDQBertAttention(config)
        # 初始化 QDQBertIntermediate 层
        self.intermediate = QDQBertIntermediate(config)
        # 初始化 QDQBertOutput 层
        self.output = QDQBertOutput(config)

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
        # 如果过去的键/值元组不为空，则从中提取解码器单向自注意力的缓存键/值，位置为1和2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        # 使用自注意力层处理隐藏状态，生成自注意力输出
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        # 提取自注意力输出的注意力部分
        attention_output = self_attention_outputs[0]

        # 如果是解码器，最后一个输出是自注意力缓存的元组
        if self.is_decoder:
            # 去除第一个和最后一个元素（自注意力输出中的自注意力元组和最后一个是自注意力缓存）
            outputs = self_attention_outputs[1:-1]
            # 获取当前自注意力的键/值元组
            present_key_value = self_attention_outputs[-1]
        else:
            # 如果不是解码器，从自注意力输出中去除第一个元素（自注意力输出中的自注意力元组）
            outputs = self_attention_outputs[1:]  # 如果输出注意力权重，则添加自注意力
        

        cross_attn_present_key_value = None
        # 如果是解码器且有编码器隐藏状态
        if self.is_decoder and encoder_hidden_states is not None:
            # 如果模型没有交叉注意力层，则引发错误
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers"
                    " by setting `config.add_cross_attention=True`"
                )

            # 如果过去的键/值元组不为空，则从中提取交叉注意力的缓存键/值，位置为3和4
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            # 使用交叉注意力层处理自注意力输出和编码器隐藏状态，生成交叉注意力输出
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                cross_attn_past_key_value,
                output_attentions,
            )
            # 提取交叉注意力输出的注意力部分
            attention_output = cross_attention_outputs[0]
            # 将交叉注意力输出中的除了第一个和最后一个元素以外的部分添加到输出中（如果输出注意力权重）
            outputs = outputs + cross_attention_outputs[1:-1]

            # 将交叉注意力的当前键/值添加到当前键/值中
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        # 使用前馈网络块处理注意力输出
        layer_output = self.feed_forward_chunk(attention_output)
        # 将前馈网络块的输出作为第一个元素，连接到输出中
        outputs = (layer_output,) + outputs

        # 如果是解码器，将注意力键/值作为最后一个输出返回
        if self.is_decoder:
            outputs = outputs + (present_key_value,)

        return outputs

    # 定义前馈网络块函数，输入为注意力输出
    def feed_forward_chunk(self, attention_output):
        # 使用中间层处理注意力输出
        intermediate_output = self.intermediate(attention_output)
        # 使用输出层处理中间输出和注意力输出，生成层输出
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output
# 根据 transformers.models.bert.modeling_bert.BertEncoder 修改为 QDQBertEncoder，表示这是一个基于 QDQBert 的编码器类
class QDQBertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config  # 初始化时保存配置参数
        # 创建多个 QDQBertLayer 层组成的列表，数量等于配置中指定的隐藏层数量
        self.layer = nn.ModuleList([QDQBertLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False  # 初始化梯度检查点标志为 False

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
        # 如果不需要输出隐藏状态，则初始化一个空元组
        all_hidden_states = () if output_hidden_states else None
        # 如果不需要输出注意力权重，则初始化一个空元组
        all_self_attentions = () if output_attentions else None
        # 如果不需要输出交叉注意力权重，且配置允许，则初始化一个空元组
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        # 如果不使用缓存，则初始化一个空元组
        next_decoder_cache = () if use_cache else None
        # 遍历每一个解码器层
        for i, layer_module in enumerate(self.layer):
            # 如果需要输出隐藏状态，则将当前隐藏状态添加到所有隐藏状态中
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 如果指定了解码器头部掩码，则获取当前层的掩码；否则为None
            layer_head_mask = head_mask[i] if head_mask is not None else None
            # 如果指定了过去的键值对，则获取当前层的过去键值对；否则为None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            # 如果启用了梯度检查点且处于训练模式
            if self.gradient_checkpointing and self.training:
                # 如果同时使用缓存，则发出警告并设置不使用缓存
                if use_cache:
                    logger.warning_once(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False
                # 调用梯度检查点函数，计算当前层的输出
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )
            else:
                # 否则，直接调用当前层模块计算当前层的输出
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )

            # 更新当前隐藏状态为当前层的输出的第一个元素
            hidden_states = layer_outputs[0]
            # 如果使用缓存，则将当前层的输出的最后一个元素添加到下一个解码器缓存中
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            # 如果需要输出注意力权重，则将当前层的注意力权重添加到所有自注意力权重中
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                # 如果配置允许且需要添加交叉注意力权重，则将当前层的交叉注意力权重添加到所有交叉注意力权重中
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        # 如果需要输出隐藏状态，则将最终隐藏状态添加到所有隐藏状态中
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 如果不返回字典形式的结果，则按顺序返回非空对象
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
        # 否则，返回带过去键值对和交叉注意力权重的基础模型输出对象
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )
# 从transformers.models.bert.modeling_bert.BertPooler复制过来，将Bert改为QDQBert
class QDQBertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 密集连接层，输入和输出大小都是config.hidden_size
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 激活函数Tanh
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 只使用第一个token对应的隐藏状态来“池化”模型
        first_token_tensor = hidden_states[:, 0]
        # 将第一个token的隐藏状态输入密集连接层
        pooled_output = self.dense(first_token_tensor)
        # 应用激活函数Tanh
        pooled_output = self.activation(pooled_output)
        return pooled_output


# 从transformers.models.bert.modeling_bert.BertPredictionHeadTransform复制过来，将Bert改为QDQBert
class QDQBertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 密集连接层，输入和输出大小都是config.hidden_size
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 根据配置选择激活函数，支持字符串或函数形式
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        # LayerNorm层，输入大小为config.hidden_size，epsilon为config.layer_norm_eps
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 将隐藏状态输入密集连接层
        hidden_states = self.dense(hidden_states)
        # 应用激活函数
        hidden_states = self.transform_act_fn(hidden_states)
        # 应用LayerNorm
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


# 基于transformers.models.bert.modeling_bert.BertLMPredictionHead，将Bert改为QDQBert
class QDQBertLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 使用QDQBertPredictionHeadTransform处理隐藏状态
        self.transform = QDQBertPredictionHeadTransform(config)

        # 输出权重与输入嵌入相同，但每个token有一个仅输出的偏置
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # 偏置参数，与resize_token_embeddings正确调整大小的链接
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # 需要连接这两个变量，以便偏置与`resize_token_embeddings`正确调整大小
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


# 基于transformers.models.bert.modeling_bert.BertOnlyMLMHead，将Bert改为QDQBert
class QDQBertOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 使用QDQBertLMPredictionHead进行预测
        self.predictions = QDQBertLMPredictionHead(config)

    def forward(self, sequence_output):
        # 对序列输出进行预测
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


# 从transformers.models.bert.modeling_bert.BertOnlyNSPHead复制过来，将Bert改为QDQBert
class QDQBertOnlyNSPHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 用于二分类的线性层，输入大小为config.hidden_size，输出大小为2
        self.seq_relationship = nn.Linear(config.hidden_size, 2)
    # 定义一个方法 `forward`，接受参数 `pooled_output`
    def forward(self, pooled_output):
        # 调用模型中的 `seq_relationship` 方法，传入 `pooled_output` 参数，计算序列关系得分
        seq_relationship_score = self.seq_relationship(pooled_output)
        # 返回计算得到的序列关系得分
        return seq_relationship_score
# 根据 transformers.models.bert.modeling_bert.BertPreTrainingHeads 更改为 QDQBertPreTrainingHeads，并将 Bert 替换为 QDQBert
class QDQBertPreTrainingHeads(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 使用 QDQBertLMPredictionHead 初始化预测头部
        self.predictions = QDQBertLMPredictionHead(config)
        # 使用线性层初始化序列关系头部，输出维度为2
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, sequence_output, pooled_output):
        # 对序列输出进行预测
        prediction_scores = self.predictions(sequence_output)
        # 对汇总输出进行序列关系预测
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


# 根据 transformers.models.bert.modeling_bert.BertPreTrainedModel 更改为 QDQBertPreTrainedModel，并将 Bert 替换为 QDQBert
class QDQBertPreTrainedModel(PreTrainedModel):
    """
    一个抽象类，处理权重初始化以及下载和加载预训练模型的简单接口。
    """

    # 使用 QDQBertConfig 作为配置类
    config_class = QDQBertConfig
    # 使用 load_tf_weights_in_qdqbert 来加载 TF 权重
    load_tf_weights = load_tf_weights_in_qdqbert
    # 模型基础名称前缀设置为 "bert"
    base_model_prefix = "bert"
    # 支持梯度检查点
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """初始化权重"""
        if isinstance(module, nn.Linear):
            # 稍微不同于 TF 版本，使用正态分布初始化权重
            # 参考 https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


QDQBERT_START_DOCSTRING = r"""
    此模型继承自 [`PreTrainedModel`]。查看超类文档以了解库实现的通用方法（例如下载或保存模型、调整输入嵌入、修剪头等）。

    此模型还是 PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) 的子类。
    将其用作常规的 PyTorch 模块，并参考 PyTorch 文档以获取所有与一般使用和行为相关的事项。

    参数:
        config ([`QDQBertConfig`]): 包含模型所有参数的配置类。
            使用配置文件初始化不会加载与模型相关的权重，只加载配置。查看 [`~PreTrainedModel.from_pretrained`] 方法以加载模型权重。
"""

QDQBERT_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `({0})`):
            # 输入序列中的token索引，用于词汇表中的位置。

            # 可以使用`AutoTokenizer`获得这些索引。详情请参见`PreTrainedTokenizer.encode`和`PreTrainedTokenizer.__call__`。

            # [什么是input IDs?](../glossary#input-ids)
        attention_mask (`torch.FloatTensor` of shape `({0})`, *optional*):
            # 遮盖掩模，用于避免在填充token索引上进行注意力计算。遮盖值为0或1：

            # - 1 表示对**未遮盖**的token进行注意力计算，
            # - 0 表示对**遮盖**的token进行注意力计算。

            # [什么是attention masks?](../glossary#attention-mask)
        token_type_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            # 段落token索引，指示输入中第一部分和第二部分。索引为0或1：

            # - 0 对应*句子A*的token，
            # - 1 对应*句子B*的token。

            # [什么是token type IDs?](../glossary#token-type-ids)
        position_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            # 输入序列token在位置嵌入中的位置索引。选择范围为`[0, config.max_position_embeddings - 1]`。

            # [什么是position IDs?](../glossary#position-ids)
        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            # 用于屏蔽自注意力模块中选定头部的遮盖。遮盖值为0或1：

            # - 1 表示该头部**未遮盖**，
            # - 0 表示该头部**遮盖**。

        inputs_embeds (`torch.FloatTensor` of shape `({0}, hidden_size)`, *optional*):
            # 可选地，您可以直接传递嵌入表示而不是`input_ids`。如果您想对如何将`input_ids`索引转换为相关向量有更多控制权，这很有用。

        output_attentions (`bool`, *optional*):
            # 是否返回所有注意力层的注意力张量。返回的张量中的`attentions`部分会有更多细节。

        output_hidden_states (`bool`, *optional*):
            # 是否返回所有层的隐藏状态。返回的张量中的`hidden_states`部分会有更多细节。

        return_dict (`bool`, *optional*):
            # 是否返回[`~utils.ModelOutput`]而不是普通元组。
    """

    @add_start_docstrings(
        "The bare QDQBERT Model transformer outputting raw hidden-states without any specific head on top.",
        QDQBERT_START_DOCSTRING,
    )
    class QDQBertModel(QDQBertPreTrainedModel):
        """

        The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
        cross-attention is added between the self-attention layers, following the architecture described in [Attention is
        all you need](https://arxiv.org/abs/1706.03762) by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
        Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

        To behave as an decoder the model needs to be initialized with the `is_decoder` argument of the configuration set
        to `True`. To be used in a Seq2Seq model, the model needs to initialized with both `is_decoder` argument and
        `add_cross_attention` set to `True`; an `encoder_hidden_states` is then expected as an input to the forward pass.
        """

        def __init__(self, config, add_pooling_layer: bool = True):
            requires_backends(self, "pytorch_quantization")
            super().__init__(config)
            self.config = config

            # Initialize the embeddings layer using QDQBertEmbeddings class
            self.embeddings = QDQBertEmbeddings(config)
            # Initialize the encoder layer using QDQBertEncoder class
            self.encoder = QDQBertEncoder(config)

            # Initialize the pooler layer if add_pooling_layer is set to True
            self.pooler = QDQBertPooler(config) if add_pooling_layer else None

            # Initialize weights and apply final processing
            self.post_init()

        def get_input_embeddings(self):
            # Return the word embeddings from the embeddings layer
            return self.embeddings.word_embeddings

        def set_input_embeddings(self, value):
            # Set the word embeddings in the embeddings layer to the given value
            self.embeddings.word_embeddings = value

        def _prune_heads(self, heads_to_prune: Dict[int, List[int]]):
            """
            Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
            class PreTrainedModel
            """
            for layer, heads in heads_to_prune.items():
                # Prune heads specified in heads_to_prune for the attention layer in each encoder layer
                self.encoder.layer[layer].attention.prune_heads(heads)

        @add_start_docstrings_to_model_forward(QDQBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
        @add_code_sample_docstrings(
            checkpoint=_CHECKPOINT_FOR_DOC,
            output_type=BaseModelOutputWithPoolingAndCrossAttentions,
            config_class=_CONFIG_FOR_DOC,
        )
    ```
    # 正向传播函数，用于模型的前向推理过程
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,  # 输入的token id序列，数据类型为长整型Tensor，可选参数
        attention_mask: Optional[torch.FloatTensor] = None,  # 注意力掩码，数据类型为浮点型Tensor，可选参数
        token_type_ids: Optional[torch.LongTensor] = None,  # token类型id，数据类型为长整型Tensor，可选参数
        position_ids: Optional[torch.LongTensor] = None,  # 位置id，数据类型为长整型Tensor，可选参数
        head_mask: Optional[torch.FloatTensor] = None,  # 头部掩码，数据类型为浮点型Tensor，可选参数
        inputs_embeds: Optional[torch.FloatTensor] = None,  # 输入的嵌入向量，数据类型为浮点型Tensor，可选参数
        encoder_hidden_states: Optional[torch.FloatTensor] = None,  # 编码器的隐藏状态，数据类型为浮点型Tensor，可选参数
        encoder_attention_mask: Optional[torch.FloatTensor] = None,  # 编码器的注意力掩码，数据类型为浮点型Tensor，可选参数
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,  # 过去的键值对，数据类型为嵌套元组的浮点型Tensor，可选参数
        use_cache: Optional[bool] = None,  # 是否使用缓存，数据类型为布尔型，可选参数
        output_attentions: Optional[bool] = None,  # 是否输出注意力权重，数据类型为布尔型，可选参数
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态，数据类型为布尔型，可选参数
        return_dict: Optional[bool] = None,  # 是否返回字典格式结果，数据类型为布尔型，可选参数
# 使用装饰器为模型添加文档字符串，指定了其用途为在 CLM 微调中使用语言建模头部的 QDQBERT 模型
@add_start_docstrings(
    """QDQBERT Model with a `language modeling` head on top for CLM fine-tuning.""", QDQBERT_START_DOCSTRING
)
# 定义 QDQBertLMHeadModel 类，继承自 QDQBertPreTrainedModel
class QDQBertLMHeadModel(QDQBertPreTrainedModel):
    # 定义了一组关键字列表，用于指定需要共享权重的参数键名
    _tied_weights_keys = ["predictions.decoder.weight", "predictions.decoder.bias"]

    # 初始化方法，接受配置参数 config
    def __init__(self, config):
        # 调用父类初始化方法
        super().__init__(config)

        # 如果配置中未指定为解码器，则发出警告
        if not config.is_decoder:
            logger.warning("If you want to use `QDQBertLMHeadModel` as a standalone, add `is_decoder=True.`")

        # 创建 QDQBertModel 实例，并禁用添加池化层
        self.bert = QDQBertModel(config, add_pooling_layer=False)
        
        # 创建 QDQBertOnlyMLMHead 实例
        self.cls = QDQBertOnlyMLMHead(config)

        # 执行后续的初始化和权重处理
        self.post_init()

    # 返回输出嵌入层的方法
    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    # 设置输出嵌入层的方法
    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    # forward 方法用于模型前向传播
    @add_start_docstrings_to_model_forward(QDQBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=CausalLMOutputWithCrossAttentions, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.LongTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        # 略

    # prepare_inputs_for_generation 方法准备生成的输入
    def prepare_inputs_for_generation(
        self,
        input_ids: Optional[torch.LongTensor],
        past_key_values=None,
        attention_mask: Optional[torch.Tensor] = None,
        **model_kwargs,
    ):
        # 获取输入张量的形状
        input_shape = input_ids.shape
        
        # 如果没有给定注意力遮罩，则创建全为1的遮罩张量
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_shape)

        # 如果给定了过去的键值对，则根据过去的键值对调整输入的 input_ids
        if past_key_values is not None:
            # 获取过去键值对的长度
            past_length = past_key_values[0][0].shape[2]

            # 如果输入的 input_ids 长度大于过去的长度，则截取掉前面的部分
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # 否则，默认只保留最后一个输入 ID
                remove_prefix_length = input_ids.shape[1] - 1

            input_ids = input_ids[:, remove_prefix_length:]

        # 返回一个包含更新后的 input_ids、attention_mask 和 past_key_values 的字典
        return {"input_ids": input_ids, "attention_mask": attention_mask, "past_key_values": past_key_values}

    # 重新排序缓存中的过去键值对，以便与 beam 搜索索引对应
    def _reorder_cache(self, past_key_values, beam_idx):
        # 初始化一个空的重新排序后的过去键值对元组
        reordered_past = ()
        
        # 对每一层的过去状态进行重新排序
        for layer_past in past_key_values:
            reordered_past += (
                # 将每个过去状态按照 beam_idx 列表重新排序，并转移到正确的设备上
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        
        # 返回重新排序后的过去键值对元组
        return reordered_past
@add_start_docstrings("""QDQBERT Model with a `language modeling` head on top.""", QDQBERT_START_DOCSTRING)
class QDQBertForMaskedLM(QDQBertPreTrainedModel):
    _tied_weights_keys = ["predictions.decoder.weight", "predictions.decoder.bias"]

    def __init__(self, config):
        super().__init__(config)

        if config.is_decoder:
            # 如果配置为解码器，警告用户使用双向自注意力时需将 `config.is_decoder` 设为 False
            logger.warning(
                "If you want to use `QDQBertForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        # 使用配置初始化 QDQBERT 模型，禁用添加池化层
        self.bert = QDQBertModel(config, add_pooling_layer=False)
        # 初始化 MLM 头部
        self.cls = QDQBertOnlyMLMHead(config)

        # 初始化权重并进行最终处理
        self.post_init()

    def get_output_embeddings(self):
        # 返回 MLM 头部的解码器权重
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        # 设置 MLM 头部的解码器权重
        self.cls.predictions.decoder = new_embeddings

    @add_start_docstrings_to_model_forward(QDQBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=MaskedLMOutput,
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
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        """
        # 初始化 return_dict 变量，如果 return_dict 参数非空则使用其值，否则使用 self.config.use_return_dict 的值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用 BERT 模型进行前向传播
        outputs = self.bert(
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

        # 获取 BERT 输出的序列输出
        sequence_output = outputs[0]
        
        # 使用分类层进行预测得分的计算
        prediction_scores = self.cls(sequence_output)

        masked_lm_loss = None
        # 如果 labels 参数不为空，则计算 masked language modeling 的损失
        if labels is not None:
            loss_fct = CrossEntropyLoss()  # 定义交叉熵损失函数，用于计算损失
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        # 如果 return_dict 为 False，则返回 tuple 类型的输出
        if not return_dict:
            output = (prediction_scores,) + outputs[2:]  # 构建输出元组
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        # 如果 return_dict 为 True，则返回 MaskedLMOutput 对象
        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids: torch.LongTensor, attention_mask: Optional[torch.FloatTensor] = None, **model_kwargs
    ):
        # 获取输入张量的形状和有效的 batch 大小
        input_shape = input_ids.shape
        effective_batch_size = input_shape[0]

        # 如果配置文件中的 pad_token_id 为空，则抛出 ValueError 异常
        if self.config.pad_token_id is None:
            raise ValueError("The PAD token should be defined for generation")

        # 扩展 attention_mask，增加一个全零列
        attention_mask = torch.cat([attention_mask, attention_mask.new_zeros((attention_mask.shape[0], 1))], dim=-1)
        
        # 创建一个全为 pad_token_id 的 dummy_token 张量，并将其连接到 input_ids 后面
        dummy_token = torch.full(
            (effective_batch_size, 1), self.config.pad_token_id, dtype=torch.long, device=input_ids.device
        )
        input_ids = torch.cat([input_ids, dummy_token], dim=1)

        # 返回包含输入张量和 attention_mask 的字典
        return {"input_ids": input_ids, "attention_mask": attention_mask}
# 使用指定的文档字符串初始化一个带有“下一个句子预测（分类）”头部的Bert模型。
@add_start_docstrings(
    """Bert Model with a `next sentence prediction (classification)` head on top.""",
    QDQBERT_START_DOCSTRING,
)
# 创建一个QDQBertForNextSentencePrediction类，继承自QDQBertPreTrainedModel类。
class QDQBertForNextSentencePrediction(QDQBertPreTrainedModel):
    # 初始化方法，接受一个配置对象作为参数。
    def __init__(self, config):
        # 调用父类的初始化方法。
        super().__init__(config)

        # 实例化一个QDQBertModel对象，作为BERT模型的主体。
        self.bert = QDQBertModel(config)
        # 实例化一个QDQBertOnlyNSPHead对象，作为只包含NSP头部的模型组件。
        self.cls = QDQBertOnlyNSPHead(config)

        # 调用自定义的初始化方法，用于初始化权重并进行最终的处理。
        self.post_init()

    # 前向传播方法，接受多个输入参数和关键字参数。
    @add_start_docstrings_to_model_forward(QDQBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=NextSentencePredictorOutput, config_class=_CONFIG_FOR_DOC)
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
        **kwargs,
        ) -> Union[Tuple, NextSentencePredictorOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the next sequence prediction (classification) loss. Input should be a sequence pair
            (see `input_ids` docstring). Indices should be in `[0, 1]`:

            - 0 indicates sequence B is a continuation of sequence A,
            - 1 indicates sequence B is a random sequence.

        Returns:
            If `return_dict=True`, returns a `NextSentencePredictorOutput` object containing:
                - loss (`torch.FloatTensor`, *optional*): Next sentence prediction loss.
                - logits (`torch.FloatTensor` of shape `(batch_size, 2)`): Scores for next sentence prediction.
                - hidden_states (`Optional[Tuple[torch.FloatTensor]]`): Tuple of hidden states at each layer of the model.
                - attentions (`Optional[Tuple[torch.FloatTensor]]`): Tuple of attention tensors at each layer of the model.

            If `return_dict=False`, returns a tuple:
                - next_sentence_loss (`Optional[torch.FloatTensor]`): Next sentence prediction loss.
                - seq_relationship_scores (`torch.FloatTensor`): Scores for next sentence prediction.
                - hidden_states (`Optional[Tuple[torch.FloatTensor]]`): Tuple of hidden states at each layer of the model.
                - attentions (`Optional[Tuple[torch.FloatTensor]]`): Tuple of attention tensors at each layer of the model.

        Example:

        ```python
        >>> from transformers import AutoTokenizer, QDQBertForNextSentencePrediction
        >>> import torch

        >>> tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
        >>> model = QDQBertForNextSentencePrediction.from_pretrained("google-bert/bert-base-uncased")

        >>> prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
        >>> next_sentence = "The sky is blue due to the shorter wavelength of blue light."
        >>> encoding = tokenizer(prompt, next_sentence, return_tensors="pt")

        >>> outputs = model(**encoding, labels=torch.LongTensor([1]))
        >>> logits = outputs.logits
        >>> assert logits[0, 0] < logits[0, 1]  # next sentence was random
        ```

        Check if `next_sentence_label` is provided in `kwargs`; issue a warning and use `labels` instead if found.
        """
        
        if "next_sentence_label" in kwargs:
            warnings.warn(
                "The `next_sentence_label` argument is deprecated and will be removed in a future version, use"
                " `labels` instead.",
                FutureWarning,
            )
            labels = kwargs.pop("next_sentence_label")

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
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

        pooled_output = outputs[1]

        seq_relationship_scores = self.cls(pooled_output)

        next_sentence_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            next_sentence_loss = loss_fct(seq_relationship_scores.view(-1, 2), labels.view(-1))

        if not return_dict:
            output = (seq_relationship_scores,) + outputs[2:]
            return ((next_sentence_loss,) + output) if next_sentence_loss is not None else output

        return NextSentencePredictorOutput(
            loss=next_sentence_loss,
            logits=seq_relationship_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
@add_start_docstrings(
    """
    Bert Model transformer with a sequence classification/regression head on top (a linear layer on top of the pooled
    output) e.g. for GLUE tasks.
    """,
    QDQBERT_START_DOCSTRING,
)
class QDQBertForSequenceClassification(QDQBertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels  # 从配置中获取标签数量
        self.config = config

        self.bert = QDQBertModel(config)  # 使用给定配置初始化 QDQBertModel
        self.dropout = nn.Dropout(config.hidden_dropout_prob)  # 根据配置设置 dropout 层
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)  # 创建一个线性层用于分类，输入维度为隐藏大小，输出维度为标签数量
        # 初始化权重并进行最终处理
        self.post_init()

    @add_start_docstrings_to_model_forward(QDQBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=SequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,  # 输入的token IDs
        attention_mask: Optional[torch.FloatTensor] = None,  # 注意力掩码
        token_type_ids: Optional[torch.LongTensor] = None,  # token 类型 IDs
        position_ids: Optional[torch.LongTensor] = None,  # 位置 IDs
        head_mask: Optional[torch.FloatTensor] = None,  # 头部掩码
        inputs_embeds: Optional[torch.FloatTensor] = None,  # 嵌入输入
        labels: Optional[torch.LongTensor] = None,  # 标签
        output_attentions: Optional[bool] = None,  # 是否输出注意力
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态
        return_dict: Optional[bool] = None,  # 是否返回字典形式结果
        ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        # 如果 return_dict 不为 None，则使用其值；否则使用 self.config.use_return_dict 的值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 将输入传递给 BERT 模型进行处理，获取模型的输出
        outputs = self.bert(
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

        # 从 BERT 模型的输出中获取池化后的特征表示
        pooled_output = outputs[1]

        # 对池化后的特征表示应用 dropout 操作
        pooled_output = self.dropout(pooled_output)
        
        # 将 dropout 后的特征表示输入分类器，得到 logits（预测值）
        logits = self.classifier(pooled_output)

        # 初始化损失为 None
        loss = None

        # 如果传入了 labels，则计算损失
        if labels is not None:
            # 如果问题类型未指定，则根据 num_labels 和 labels 的数据类型进行判断
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            # 根据问题类型计算相应的损失
            if self.config.problem_type == "regression":
                loss_fct = MSELoss()  # 使用均方误差损失函数
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()  # 使用交叉熵损失函数
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()  # 使用带 logits 的二元交叉熵损失函数
                loss = loss_fct(logits, labels)

        # 如果 return_dict 为 False，则输出格式为元组
        if not return_dict:
            output = (logits,) + outputs[2:]  # 包括 logits 和其他输出状态
            return ((loss,) + output) if loss is not None else output

        # 如果 return_dict 为 True，则输出格式为 SequenceClassifierOutput 对象
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 使用装饰器为类添加文档字符串，描述了该类的功能和用途，特别是在多选分类任务中使用 BERT 模型的情况
@add_start_docstrings(
    """
    Bert Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
    softmax) e.g. for RocStories/SWAG tasks.
    """,
    QDQBERT_START_DOCSTRING,
)
class QDQBertForMultipleChoice(QDQBertPreTrainedModel):
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)

        # 初始化 BERT 模型
        self.bert = QDQBertModel(config)
        # 添加 dropout 层，以防止过拟合
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # 添加一个线性层作为分类器，输入尺寸为隐藏状态的尺寸，输出尺寸为1（用于二分类）
        self.classifier = nn.Linear(config.hidden_size, 1)

        # 初始化权重并进行最终处理
        self.post_init()

    # 使用装饰器为 forward 方法添加文档字符串，描述了该方法的输入和输出
    @add_start_docstrings_to_model_forward(QDQBERT_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length"))
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
        # forward 方法的输入参数，用于多选分类任务的 BERT 模型
        ) -> Union[Tuple, MultipleChoiceModelOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the multiple choice classification loss. Indices should be in `[0, ...,
            num_choices-1]` where `num_choices` is the size of the second dimension of the input tensors. (See
            `input_ids` above)
        """
        # 确定是否返回字典类型的输出，若未指定则使用模型配置中的默认设置
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 获取输入张量的第二维大小，即选项的数量
        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

        # 重塑输入张量以便进行批处理处理
        input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )

        # 使用BERT模型进行前向传播计算
        outputs = self.bert(
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

        # 提取汇聚后的输出
        pooled_output = outputs[1]

        # 对汇聚输出进行dropout处理
        pooled_output = self.dropout(pooled_output)
        # 使用分类器对处理后的特征进行分类预测
        logits = self.classifier(pooled_output)
        # 重塑logits张量以匹配多选项问题的形状
        reshaped_logits = logits.view(-1, num_choices)

        # 初始化损失为None
        loss = None
        # 如果提供了标签，则计算交叉熵损失
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)

        # 如果不要求返回字典类型的输出，则返回一个元组
        if not return_dict:
            output = (reshaped_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 如果要求返回字典类型的输出，则返回MultipleChoiceModelOutput对象
        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
QDQBERT Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
Named-Entity-Recognition (NER) tasks.
"""
QDQBERT_START_DOCSTRING,
)
class QDQBertForTokenClassification(QDQBertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        # Initialize QDQBertModel with provided configuration
        self.bert = QDQBertModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(QDQBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
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
    ) -> Union[Tuple, TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Perform forward pass through QDQBertModel
        outputs = self.bert(
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

        sequence_output = outputs[0]

        # Apply dropout to the sequence output
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            # Calculate CrossEntropyLoss if labels are provided
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            # Prepare output tuple if return_dict is False
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # Return TokenClassifierOutput if return_dict is True
        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    QDQBERT Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
    layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    QDQBERT_START_DOCSTRING,


# QDQBERT 模型，顶部带有用于类似 SQuAD 的抽取式问答任务的跨度分类头部（在隐藏状态输出之上的线性层，用于计算“跨度起始对数”和“跨度终止对数”）。
# QDQBERT_START_DOCSTRING 是用于文档字符串的起始标记。
)
# 结束 QDQBertForQuestionAnswering 类的定义

class QDQBertForQuestionAnswering(QDQBertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        # 初始化 BERT 模型，不添加池化层
        self.bert = QDQBertModel(config, add_pooling_layer=False)
        # 线性层，用于答案抽取任务的输出
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        # 初始化权重并应用最终处理
        self.post_init()

    @add_start_docstrings_to_model_forward(QDQBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=QuestionAnsweringModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    # 前向传播函数，接受多个输入参数
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
        # 默认情况下，如果 return_dict 为 None，则使用 self.config.use_return_dict
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 使用 BERT 模型处理输入数据，输出包括 sequence_output 和其他附加信息
        outputs = self.bert(
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

        # 取出 BERT 输出的 sequence_output，即模型最后一层的输出
        sequence_output = outputs[0]

        # 将 sequence_output 传入 QA 输出层，得到起始位置和结束位置的 logits
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()  # 去除多余的维度，使得数据连续
        end_logits = end_logits.squeeze(-1).contiguous()  # 去除多余的维度，使得数据连续

        total_loss = None
        # 如果给定了起始和结束位置，则计算损失
        if start_positions is not None and end_positions is not None:
            # 如果在多 GPU 环境中，添加一个维度
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # 将超出模型输入的起始和结束位置修正到有效范围内
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            # 使用交叉熵损失函数计算起始位置和结束位置的损失
            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        # 如果不需要返回字典格式的输出，则直接返回 logits 和其他附加信息
        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]  # 包括除了 sequence_output 外的其他输出
            return ((total_loss,) + output) if total_loss is not None else output

        # 返回格式化的 QuestionAnsweringModelOutput 对象，包括损失、起始和结束 logits，以及其他附加信息
        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
```