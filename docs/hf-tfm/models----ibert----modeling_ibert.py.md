# `.\models\ibert\modeling_ibert.py`

```
# coding=utf-8
# 版权声明，版权归作者及 HuggingFace 公司所有，保留一切权利
# Copyright 2021 The I-BERT Authors (Sehoon Kim, Amir Gholami, Zhewei Yao,
# Michael Mahoney, Kurt Keutzer - UC Berkeley) and The HuggingFace Inc. team.
# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

"""PyTorch I-BERT model."""

import math
from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from ...activations import gelu
from ...modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging

# 导入 logging 模块
logger = logging.get_logger(__name__)

# 以下是用于文档的定义
_CHECKPOINT_FOR_DOC = "kssteven/ibert-roberta-base"
_CONFIG_FOR_DOC = "IBertConfig"

# 预训练模型的存档列表
IBERT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "kssteven/ibert-roberta-base",
    "kssteven/ibert-roberta-large",
    "kssteven/ibert-roberta-large-mnli",
]


class IBertEmbeddings(nn.Module):
    """
    Same as BertEmbeddings with a tiny tweak for positional embeddings indexing.
    
    IBertEmbeddings 类，与 BertEmbeddings 相同，但稍作调整以支持位置嵌入索引。
    """
    # 初始化函数，接受一个配置参数对象作为输入
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()
        
        # 设置量化模式，从配置对象中获取
        self.quant_mode = config.quant_mode
        
        # 设置嵌入比特位数
        self.embedding_bit = 8
        self.embedding_act_bit = 16
        self.act_bit = 8
        self.ln_input_bit = 22
        self.ln_output_bit = 32
    
        # 创建词嵌入对象，使用QuantEmbedding进行量化
        self.word_embeddings = QuantEmbedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx=config.pad_token_id,
            weight_bit=self.embedding_bit,
            quant_mode=self.quant_mode,
        )
        
        # 创建token类型嵌入对象，使用QuantEmbedding进行量化
        self.token_type_embeddings = QuantEmbedding(
            config.type_vocab_size, config.hidden_size, weight_bit=self.embedding_bit, quant_mode=self.quant_mode
        )
    
        # 注册位置ID张量为缓冲区，使用torch.arange生成连续的位置ID
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )
        
        # 设置位置嵌入的类型，默认为绝对位置嵌入
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
    
        # 设置填充索引，并创建位置嵌入对象，使用QuantEmbedding进行量化
        self.padding_idx = config.pad_token_id
        self.position_embeddings = QuantEmbedding(
            config.max_position_embeddings,
            config.hidden_size,
            padding_idx=self.padding_idx,
            weight_bit=self.embedding_bit,
            quant_mode=self.quant_mode,
        )
    
        # 创建嵌入激活函数对象，使用QuantAct进行量化
        self.embeddings_act1 = QuantAct(self.embedding_act_bit, quant_mode=self.quant_mode)
        self.embeddings_act2 = QuantAct(self.embedding_act_bit, quant_mode=self.quant_mode)
    
        # 创建层归一化对象，使用IntLayerNorm进行量化，保持与TensorFlow模型变量名一致
        self.LayerNorm = IntLayerNorm(
            config.hidden_size,
            eps=config.layer_norm_eps,
            output_bit=self.ln_output_bit,
            quant_mode=self.quant_mode,
            force_dequant=config.force_dequant,
        )
        
        # 创建输出激活函数对象，使用QuantAct进行量化
        self.output_activation = QuantAct(self.act_bit, quant_mode=self.quant_mode)
        
        # 创建Dropout对象，使用配置中的隐藏层dropout概率
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        ):
            # 如果没有给定位置编码，根据输入的 token ids 创建位置编码，保留任何填充的 token 的填充状态
            if position_ids is None:
                if input_ids is not None:
                    # 从输入的 token ids 创建位置编码
                    position_ids = create_position_ids_from_input_ids(
                        input_ids, self.padding_idx, past_key_values_length
                    ).to(input_ids.device)
                else:
                    # 根据输入的嵌入向量创建位置编码
                    position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds)

            # 如果给定了 input_ids，则获取其形状；否则获取 inputs_embeds 的形状去掉最后一个维度
            if input_ids is not None:
                input_shape = input_ids.size()
            else:
                input_shape = inputs_embeds.size()[:-1]

            # 如果没有给定 token_type_ids，则创建一个全零张量作为 token_type_ids
            if token_type_ids is None:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

            # 如果没有给定 inputs_embeds，则通过 word_embeddings 获取输入的嵌入向量及其缩放因子
            if inputs_embeds is None:
                inputs_embeds, inputs_embeds_scaling_factor = self.word_embeddings(input_ids)
            else:
                # 否则设定 inputs_embeds_scaling_factor 为 None
                inputs_embeds_scaling_factor = None

            # 根据 token_type_ids 获取 token 类型的嵌入向量及其缩放因子
            token_type_embeddings, token_type_embeddings_scaling_factor = self.token_type_embeddings(token_type_ids)

            # 将 inputs_embeds 和 token_type_embeddings 组合并通过 embeddings_act1 处理得到嵌入向量及其缩放因子
            embeddings, embeddings_scaling_factor = self.embeddings_act1(
                inputs_embeds,
                inputs_embeds_scaling_factor,
                identity=token_type_embeddings,
                identity_scaling_factor=token_type_embeddings_scaling_factor,
            )

            # 如果 position_embedding_type 是 "absolute"，则根据 position_ids 获取位置嵌入向量及其缩放因子
            if self.position_embedding_type == "absolute":
                position_embeddings, position_embeddings_scaling_factor = self.position_embeddings(position_ids)
                # 将 embeddings 和 position_embeddings 组合并通过 embeddings_act1 处理得到最终的嵌入向量及其缩放因子
                embeddings, embeddings_scaling_factor = self.embeddings_act1(
                    embeddings,
                    embeddings_scaling_factor,
                    identity=position_embeddings,
                    identity_scaling_factor=position_embeddings_scaling_factor,
                )

            # 对最终的嵌入向量进行 LayerNorm 处理，并返回处理后的嵌入向量及其缩放因子
            embeddings, embeddings_scaling_factor = self.LayerNorm(embeddings, embeddings_scaling_factor)
            embeddings = self.dropout(embeddings)
            # 对嵌入向量应用 output_activation，并返回处理后的嵌入向量及其缩放因子
            embeddings, embeddings_scaling_factor = self.output_activation(embeddings, embeddings_scaling_factor)
            return embeddings, embeddings_scaling_factor

        def create_position_ids_from_inputs_embeds(self, inputs_embeds):
            """
            We are provided embeddings directly. We cannot infer which are padded so just generate sequential position ids.

            Args:
                inputs_embeds: torch.Tensor

            Returns: torch.Tensor
            """
            # 获取输入嵌入向量的形状，并计算序列长度
            input_shape = inputs_embeds.size()[:-1]
            sequence_length = input_shape[1]

            # 根据序列长度生成从 padding_idx + 1 开始的连续位置编码
            position_ids = torch.arange(
                self.padding_idx + 1, sequence_length + self.padding_idx + 1, dtype=torch.long, device=inputs_embeds.device
            )
            return position_ids.unsqueeze(0).expand(input_shape)
# 定义 IBertSelfAttention 类，继承自 nn.Module，实现自注意力机制部分
class IBertSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 检查 hidden_size 是否能被 num_attention_heads 整除，同时不应有 embedding_size 属性
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )
        
        # 初始化量化模式和量化位数设置
        self.quant_mode = config.quant_mode
        self.weight_bit = 8
        self.bias_bit = 32
        self.act_bit = 8

        # 设置注意力头数和每个注意力头的大小
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # 初始化 Q、K、V 的线性层，进行量化
        self.query = QuantLinear(
            config.hidden_size,
            self.all_head_size,
            bias=True,
            weight_bit=self.weight_bit,
            bias_bit=self.bias_bit,
            quant_mode=self.quant_mode,
            per_channel=True,
        )
        self.key = QuantLinear(
            config.hidden_size,
            self.all_head_size,
            bias=True,
            weight_bit=self.weight_bit,
            bias_bit=self.bias_bit,
            quant_mode=self.quant_mode,
            per_channel=True,
        )
        self.value = QuantLinear(
            config.hidden_size,
            self.all_head_size,
            bias=True,
            weight_bit=self.weight_bit,
            bias_bit=self.bias_bit,
            quant_mode=self.quant_mode,
            per_channel=True,
        )

        # 初始化 Q、K、V 的激活函数，进行量化
        self.query_activation = QuantAct(self.act_bit, quant_mode=self.quant_mode)
        self.key_activation = QuantAct(self.act_bit, quant_mode=self.quant_mode)
        self.value_activation = QuantAct(self.act_bit, quant_mode=self.quant_mode)
        self.output_activation = QuantAct(self.act_bit, quant_mode=self.quant_mode)

        # Dropout 层，用于注意力概率的 dropout
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        
        # 位置嵌入类型设置为绝对位置编码
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        if self.position_embedding_type != "absolute":
            raise ValueError("I-BERT only supports 'absolute' for `config.position_embedding_type`")

        # 定义 Softmax 层，用于计算注意力权重
        self.softmax = IntSoftmax(self.act_bit, quant_mode=self.quant_mode, force_dequant=config.force_dequant)

    # 将输入张量 x 转换为注意力分数张量的形状
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    # 前向传播函数，实现自注意力机制的计算过程
    def forward(
        self,
        hidden_states,
        hidden_states_scaling_factor,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
    # 初始化函数，接收配置参数 config
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()
        # 设置量化模式
        self.quant_mode = config.quant_mode
        # 设置激活位数为 8
        self.act_bit = 8
        # 设置权重位数为 8
        self.weight_bit = 8
        # 设置偏置位数为 32
        self.bias_bit = 32
        # 设置输入层归一化的位数为 22
        self.ln_input_bit = 22
        # 设置输出层归一化的位数为 32
        self.ln_output_bit = 32

        # 创建一个量化线性层对象，用于神经网络的量化线性变换
        self.dense = QuantLinear(
            config.hidden_size,
            config.hidden_size,
            bias=True,
            weight_bit=self.weight_bit,
            bias_bit=self.bias_bit,
            quant_mode=self.quant_mode,
            per_channel=True,
        )
        # 创建一个输入层激活函数的量化对象
        self.ln_input_act = QuantAct(self.ln_input_bit, quant_mode=self.quant_mode)
        # 创建一个整数型层归一化对象，用于神经网络的整数型层次归一化
        self.LayerNorm = IntLayerNorm(
            config.hidden_size,
            eps=config.layer_norm_eps,
            output_bit=self.ln_output_bit,
            quant_mode=self.quant_mode,
            force_dequant=config.force_dequant,
        )
        # 创建一个输出激活函数的量化对象
        self.output_activation = QuantAct(self.act_bit, quant_mode=self.quant_mode)
        # 创建一个 Dropout 层，用于随机置零输入张量的元素，防止过拟合
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    # 前向传播函数，接收隐藏状态、缩放因子、输入张量和其缩放因子作为输入
    def forward(self, hidden_states, hidden_states_scaling_factor, input_tensor, input_tensor_scaling_factor):
        # 使用量化线性层进行隐藏状态的线性变换
        hidden_states, hidden_states_scaling_factor = self.dense(hidden_states, hidden_states_scaling_factor)
        # 对变换后的隐藏状态应用 Dropout 操作
        hidden_states = self.dropout(hidden_states)
        # 使用输入层激活函数的量化对象，对隐藏状态进行激活函数操作
        hidden_states, hidden_states_scaling_factor = self.ln_input_act(
            hidden_states,
            hidden_states_scaling_factor,
            identity=input_tensor,
            identity_scaling_factor=input_tensor_scaling_factor,
        )
        # 使用整数型层归一化对象，对处理后的隐藏状态进行归一化操作
        hidden_states, hidden_states_scaling_factor = self.LayerNorm(hidden_states, hidden_states_scaling_factor)

        # 使用输出激活函数的量化对象，对归一化后的隐藏状态进行激活函数操作
        hidden_states, hidden_states_scaling_factor = self.output_activation(
            hidden_states, hidden_states_scaling_factor
        )
        # 返回处理后的隐藏状态和相应的缩放因子
        return hidden_states, hidden_states_scaling_factor
# 定义 IBertAttention 类，继承自 nn.Module，实现自注意力机制
class IBertAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 从配置中获取量化模式
        self.quant_mode = config.quant_mode
        # 初始化 IBertSelfAttention 层和 IBertSelfOutput 层
        self.self = IBertSelfAttention(config)
        self.output = IBertSelfOutput(config)
        # 初始化头部剪枝集合
        self.pruned_heads = set()

    # 剪枝指定的注意力头
    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        # 调用辅助函数找到可剪枝的头部索引
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # 对自注意力机制的查询、键、值进行剪枝
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        # 对输出层的稠密层进行剪枝
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # 更新超参数并记录剪枝的头部
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    # 前向传播函数
    def forward(
        self,
        hidden_states,
        hidden_states_scaling_factor,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
    ):
        # 调用自注意力层的前向传播
        self_outputs, self_outputs_scaling_factor = self.self(
            hidden_states,
            hidden_states_scaling_factor,
            attention_mask,
            head_mask,
            output_attentions,
        )
        # 调用自注意力输出层的前向传播
        attention_output, attention_output_scaling_factor = self.output(
            self_outputs[0], self_outputs_scaling_factor[0], hidden_states, hidden_states_scaling_factor
        )
        # 如果输出注意力矩阵，添加到输出中
        outputs = (attention_output,) + self_outputs[1:]
        outputs_scaling_factor = (attention_output_scaling_factor,) + self_outputs_scaling_factor[1:]
        return outputs, outputs_scaling_factor


# 定义 IBertIntermediate 类，继承自 nn.Module，实现中间层的量化操作
class IBertIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 从配置中获取量化模式
        self.quant_mode = config.quant_mode
        # 设置激活位数和权重位数
        self.act_bit = 8
        self.weight_bit = 8
        self.bias_bit = 32
        # 创建量化线性层
        self.dense = QuantLinear(
            config.hidden_size,
            config.intermediate_size,
            bias=True,
            weight_bit=self.weight_bit,
            bias_bit=self.bias_bit,
            quant_mode=self.quant_mode,
            per_channel=True,
        )
        # 检查隐藏激活函数是否为 "gelu"
        if config.hidden_act != "gelu":
            raise ValueError("I-BERT only supports 'gelu' for `config.hidden_act`")
        # 初始化中间激活函数为 IntGELU
        self.intermediate_act_fn = IntGELU(quant_mode=self.quant_mode, force_dequant=config.force_dequant)
        # 初始化输出激活函数为 QuantAct
        self.output_activation = QuantAct(self.act_bit, quant_mode=self.quant_mode)
    # 前向传播函数，接受隐藏状态和隐藏状态缩放因子作为输入参数
    def forward(self, hidden_states, hidden_states_scaling_factor):
        # 将隐藏状态和缩放因子传递给稠密层进行处理
        hidden_states, hidden_states_scaling_factor = self.dense(hidden_states, hidden_states_scaling_factor)
        # 将稠密层输出的隐藏状态和缩放因子传递给中间激活函数进行处理
        hidden_states, hidden_states_scaling_factor = self.intermediate_act_fn(
            hidden_states, hidden_states_scaling_factor
        )

        # 重新量化步骤：从32位转换为8位
        hidden_states, hidden_states_scaling_factor = self.output_activation(
            hidden_states, hidden_states_scaling_factor
        )
        # 返回处理后的隐藏状态和缩放因子
        return hidden_states, hidden_states_scaling_factor
class IBertOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.quant_mode = config.quant_mode  # 从配置中获取量化模式
        self.act_bit = 8  # 激活函数的位数设定为8位
        self.weight_bit = 8  # 权重的位数设定为8位
        self.bias_bit = 32  # 偏置的位数设定为32位
        self.ln_input_bit = 22  # LayerNorm输入的位数设定为22位
        self.ln_output_bit = 32  # LayerNorm输出的位数设定为32位

        # 创建量化线性层，指定输入大小、输出大小，并设定权重、偏置的位数，使用量化模式
        self.dense = QuantLinear(
            config.intermediate_size,
            config.hidden_size,
            bias=True,
            weight_bit=self.weight_bit,
            bias_bit=self.bias_bit,
            quant_mode=self.quant_mode,
            per_channel=True,
        )

        # 创建量化激活函数，设定输入位数和量化模式
        self.ln_input_act = QuantAct(self.ln_input_bit, quant_mode=self.quant_mode)

        # 创建整数化LayerNorm，指定输入大小、输出位数、量化模式和是否强制反量化
        self.LayerNorm = IntLayerNorm(
            config.hidden_size,
            eps=config.layer_norm_eps,
            output_bit=self.ln_output_bit,
            quant_mode=self.quant_mode,
            force_dequant=config.force_dequant,
        )

        # 创建量化激活函数，设定激活位数和量化模式
        self.output_activation = QuantAct(self.act_bit, quant_mode=self.quant_mode)

        # 创建Dropout层，设定丢弃率为配置中的隐藏层dropout概率
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, hidden_states_scaling_factor, input_tensor, input_tensor_scaling_factor):
        # 应用量化线性层，处理隐藏状态和其缩放因子
        hidden_states, hidden_states_scaling_factor = self.dense(hidden_states, hidden_states_scaling_factor)
        
        # 应用Dropout层，处理隐藏状态
        hidden_states = self.dropout(hidden_states)
        
        # 应用量化激活函数，处理隐藏状态，同时传入输入张量和其缩放因子作为辅助信息
        hidden_states, hidden_states_scaling_factor = self.ln_input_act(
            hidden_states,
            hidden_states_scaling_factor,
            identity=input_tensor,
            identity_scaling_factor=input_tensor_scaling_factor,
        )
        
        # 应用整数化LayerNorm，处理隐藏状态和其缩放因子
        hidden_states, hidden_states_scaling_factor = self.LayerNorm(hidden_states, hidden_states_scaling_factor)

        # 应用输出激活函数，处理隐藏状态和其缩放因子
        hidden_states, hidden_states_scaling_factor = self.output_activation(
            hidden_states, hidden_states_scaling_factor
        )
        
        # 返回处理后的隐藏状态和其缩放因子
        return hidden_states, hidden_states_scaling_factor


class IBertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.quant_mode = config.quant_mode  # 从配置中获取量化模式
        self.act_bit = 8  # 激活函数的位数设定为8位

        self.seq_len_dim = 1  # 序列长度维度设定为1

        # 创建IBertAttention层，使用给定的配置
        self.attention = IBertAttention(config)
        
        # 创建IBertIntermediate层，使用给定的配置
        self.intermediate = IBertIntermediate(config)
        
        # 创建IBertOutput层，使用给定的配置
        self.output = IBertOutput(config)

        # 创建量化激活函数，设定输入位数和量化模式
        self.pre_intermediate_act = QuantAct(self.act_bit, quant_mode=self.quant_mode)
        
        # 创建量化激活函数，设定输入位数和量化模式
        self.pre_output_act = QuantAct(self.act_bit, quant_mode=self.quant_mode)

    def forward(
        self,
        hidden_states,
        hidden_states_scaling_factor,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
        # 继续编写其他参数
    ):
        self_attention_outputs, self_attention_outputs_scaling_factor = self.attention(
            hidden_states,
            hidden_states_scaling_factor,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
        )
        # 获取自注意力机制的输出和相应的缩放因子
        attention_output = self_attention_outputs[0]
        attention_output_scaling_factor = self_attention_outputs_scaling_factor[0]

        outputs = self_attention_outputs[1:]  # 如果输出注意力权重，则添加自注意力权重

        # 将注意力输出作为输入，应用前馈网络
        layer_output, layer_output_scaling_factor = self.feed_forward_chunk(
            attention_output, attention_output_scaling_factor
        )
        outputs = (layer_output,) + outputs

        return outputs

    def feed_forward_chunk(self, attention_output, attention_output_scaling_factor):
        # 应用预激活函数到注意力输出和缩放因子
        attention_output, attention_output_scaling_factor = self.pre_intermediate_act(
            attention_output, attention_output_scaling_factor
        )
        # 将注意力输出传递给中间层前馈网络
        intermediate_output, intermediate_output_scaling_factor = self.intermediate(
            attention_output, attention_output_scaling_factor
        )

        # 应用预输出激活函数到中间层输出和缩放因子
        intermediate_output, intermediate_output_scaling_factor = self.pre_output_act(
            intermediate_output, intermediate_output_scaling_factor
        )
        # 应用输出层到中间层输出和相应的注意力输出及缩放因子
        layer_output, layer_output_scaling_factor = self.output(
            intermediate_output, intermediate_output_scaling_factor, attention_output, attention_output_scaling_factor
        )
        return layer_output, layer_output_scaling_factor
# 定义一个名为 IBertEncoder 的类，继承自 nn.Module 类，用于实现 BERT 编码器模型
class IBertEncoder(nn.Module):
    # 初始化方法，接收一个配置参数 config
    def __init__(self, config):
        super().__init__()  # 调用父类的初始化方法
        self.config = config  # 将传入的配置参数保存到对象的属性中
        self.quant_mode = config.quant_mode  # 从配置中获取量化模式设置
        # 创建一个由多个 IBertLayer 实例组成的模块列表，列表长度由配置中的 num_hidden_layers 决定
        self.layer = nn.ModuleList([IBertLayer(config) for _ in range(config.num_hidden_layers)])

    # 前向传播方法定义
    def forward(
        self,
        hidden_states,  # 输入的隐藏状态张量
        hidden_states_scaling_factor,  # 隐藏状态的缩放因子
        attention_mask=None,  # 注意力掩码，默认为 None
        head_mask=None,  # 头部掩码，默认为 None
        output_attentions=False,  # 是否输出注意力矩阵，默认为 False
        output_hidden_states=False,  # 是否输出所有隐藏状态，默认为 False
        return_dict=True,  # 是否以字典形式返回，默认为 True
    ):
        # 如果需要输出隐藏状态，则初始化一个空元组用于存储所有的隐藏状态张量
        all_hidden_states = () if output_hidden_states else None
        # 如果需要输出注意力矩阵，则初始化一个空元组用于存储所有的自注意力矩阵
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = None  # 不支持交叉注意力，置为 None
        next_decoder_cache = None  # 不支持缓存，置为 None

        # 遍历每一个 IBertLayer 模块进行处理
        for i, layer_module in enumerate(self.layer):
            # 如果需要输出隐藏状态，则将当前的隐藏状态张量添加到 all_hidden_states 元组中
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 如果存在头部掩码，则获取当前层的头部掩码
            layer_head_mask = head_mask[i] if head_mask is not None else None

            # 调用当前层的前向传播方法，得到该层的输出
            layer_outputs = layer_module(
                hidden_states,
                hidden_states_scaling_factor,
                attention_mask,
                layer_head_mask,
                output_attentions,
            )

            # 更新隐藏状态张量为当前层的输出的第一个元素
            hidden_states = layer_outputs[0]

            # 如果需要输出注意力矩阵，则将当前层的自注意力矩阵添加到 all_self_attentions 元组中
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        # 如果需要输出隐藏状态，则将最终的隐藏状态张量添加到 all_hidden_states 元组中
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 如果不需要以字典形式返回结果，则返回一个元组，包含所有非 None 的值
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

        # 如果需要以字典形式返回结果，则创建一个 BaseModelOutputWithPastAndCrossAttentions 实例作为返回值
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


# 定义一个名为 IBertPooler 的类，继承自 nn.Module 类，用于实现 BERT 池化器模型
class IBertPooler(nn.Module):
    # 初始化方法，接收一个配置参数 config
    def __init__(self, config):
        super().__init__()  # 调用父类的初始化方法
        self.quant_mode = config.quant_mode  # 从配置中获取量化模式设置
        # 创建一个线性层，将输入特征大小映射到相同的输出特征大小
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()  # 定义 Tanh 激活函数

    # 前向传播方法定义
    def forward(self, hidden_states):
        # 只取第一个 token 对应的隐藏状态张量作为池化输出
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)  # 通过线性层映射
        pooled_output = self.activation(pooled_output)  # 应用 Tanh 激活函数
        return pooled_output


# 定义一个名为 IBertPreTrainedModel 的类，继承自 PreTrainedModel 类
class IBertPreTrainedModel(PreTrainedModel):
    """
    一个抽象类，用于处理权重初始化和简单的接口，用于下载和加载预训练模型。
    """
    # 定义配置类为 IBertConfig
    config_class = IBertConfig
    # 定义基础模型前缀为 "ibert"
    base_model_prefix = "ibert"

    def _init_weights(self, module):
        """初始化权重"""
        # 如果模块是 QuantLinear 或 nn.Linear 类型
        if isinstance(module, (QuantLinear, nn.Linear)):
            # 使用正态分布初始化权重数据，均值为 0.0，标准差为配置中的初始化范围
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果模块有偏置，则将偏置数据初始化为零
            if module.bias is not None:
                module.bias.data.zero_()
        # 如果模块是 QuantEmbedding 或 nn.Embedding 类型
        elif isinstance(module, (QuantEmbedding, nn.Embedding)):
            # 使用正态分布初始化权重数据，均值为 0.0，标准差为配置中的初始化范围
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果模块有填充索引，则将填充索引处的权重数据初始化为零
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        # 如果模块是 IntLayerNorm 或 nn.LayerNorm 类型
        elif isinstance(module, (IntLayerNorm, nn.LayerNorm)):
            # 将模块的偏置数据初始化为零
            module.bias.data.zero_()
            # 将模块的权重数据填充为 1.0
            module.weight.data.fill_(1.0)

    def resize_token_embeddings(self, new_num_tokens=None):
        # 抛出未实现错误，因为 I-BERT 不支持调整 token embeddings
        raise NotImplementedError("`resize_token_embeddings` is not supported for I-BERT.")
IBERT_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`IBertConfig`]): Model configuration class with all the parameters of the
            model. Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""



IBERT_INPUTS_DOCSTRING = r"""
    This string contains the docstring for explaining the inputs accepted by the IBERT model.

    This docstring should describe the expected inputs for the model, such as input tensors or data structures,
    their types, shapes, and any preprocessing requirements.

    It provides guidance on how to format and prepare data for the model's forward pass, ensuring compatibility
    with the model's architecture and requirements.

    This documentation helps users understand how to correctly interface with the model, ensuring inputs are
    correctly formatted to achieve expected results.
"""
    Args:
        input_ids (`torch.LongTensor` of shape `({0})`):
            # 输入序列标记在词汇表中的索引。
            # 可以使用 `AutoTokenizer` 获取这些索引。参见 `PreTrainedTokenizer.encode` 和 `PreTrainedTokenizer.__call__`。
            # [什么是输入 ID？](../glossary#input-ids)
        attention_mask (`torch.FloatTensor` of shape `({0})`, *optional*):
            # 遮罩，避免对填充的标记索引执行注意力操作。
            # 遮罩值在 `[0, 1]` 范围内：
            # - 1 表示**未遮罩**的标记，
            # - 0 表示**已遮罩**的标记。
            # [什么是注意力遮罩？](../glossary#attention-mask)
        token_type_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            # 分段标记索引，指示输入的第一和第二部分。
            # 索引选在 `[0, 1]` 范围内：
            # - 0 对应*句子 A* 的标记，
            # - 1 对应*句子 B* 的标记。
            # [什么是分段标记 ID？](../glossary#token-type-ids)
        position_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            # 每个输入序列标记在位置嵌入中的位置索引。
            # 索引选在 `[0, config.max_position_embeddings - 1]` 范围内。
            # [什么是位置 ID？](../glossary#position-ids)
        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            # 用于空置自注意力模块中选定头部的遮罩。
            # 遮罩值在 `[0, 1]` 范围内：
            # - 1 表示**未遮罩**的头部，
            # - 0 表示**已遮罩**的头部。
        inputs_embeds (`torch.FloatTensor` of shape `({0}, hidden_size)`, *optional*):
            # 可选参数，可以直接传递嵌入表示，而不是传递 `input_ids`。
            # 如果要控制如何将 `input_ids` 索引转换为相关联的向量，这很有用。
            # 这比模型内部的嵌入查找矩阵更灵活。
        output_attentions (`bool`, *optional*):
            # 是否返回所有注意力层的注意力张量。
            # 查看返回的张量中的 `attentions` 以获取更多详细信息。
        output_hidden_states (`bool`, *optional*):
            # 是否返回所有层的隐藏状态。
            # 查看返回的张量中的 `hidden_states` 以获取更多详细信息。
        return_dict (`bool`, *optional*):
            # 是否返回 [`~utils.ModelOutput`] 而不是普通元组。
"""
@add_start_docstrings(
    "The bare I-BERT Model transformer outputting raw hidden-states without any specific head on top.",
    IBERT_START_DOCSTRING,
)
class IBertModel(IBertPreTrainedModel):
    """

    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in [Attention is
    all you need](https://arxiv.org/abs/1706.03762) by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
    Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

    """

    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config
        self.quant_mode = config.quant_mode

        # Initialize the embeddings layer for the IBERT model
        self.embeddings = IBertEmbeddings(config)
        
        # Initialize the encoder layer for the IBERT model
        self.encoder = IBertEncoder(config)

        # Initialize the pooling layer if specified
        self.pooler = IBertPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        # Return the word embeddings from the embeddings layer
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        # Set new word embeddings to the embeddings layer
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        # Iterate over layers and prune specific attention heads in each layer
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(IBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPoolingAndCrossAttentions,
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
    ):
        # Forward pass through the IBERT model
        # Detailed arguments are passed to handle different configurations
        pass


@add_start_docstrings(
    "I-BERT Model with a `language modeling` head on top.",
    IBERT_START_DOCSTRING
)
class IBertForMaskedLM(IBertPreTrainedModel):
    _tied_weights_keys = ["lm_head.decoder.bias", "lm_head.decoder.weight"]

    def __init__(self, config):
        super().__init__(config)

        # Initialize the IBERT model without a pooling layer
        self.ibert = IBertModel(config, add_pooling_layer=False)
        
        # Initialize the language modeling head for IBERT
        self.lm_head = IBertLMHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        # Return the decoder weights from the language modeling head
        return self.lm_head.decoder
    def set_output_embeddings(self, new_embeddings):
        # 将语言模型头部的解码器层替换为新的嵌入层
        self.lm_head.decoder = new_embeddings

    @add_start_docstrings_to_model_forward(IBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=MaskedLMOutput,
        config_class=_CONFIG_FOR_DOC,
        mask="<mask>",
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
    ) -> Union[MaskedLMOutput, Tuple[torch.FloatTensor]]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        kwargs (`Dict[str, any]`, optional, defaults to *{}*):
            Used to hide legacy arguments that have been deprecated.
        """
        # 确定是否返回字典格式的输出
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用iBERT模型进行前向传播
        outputs = self.ibert(
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
        # 获取序列输出
        sequence_output = outputs[0]
        # 将序列输出传递给语言模型头部以获取预测分数
        prediction_scores = self.lm_head(sequence_output)

        masked_lm_loss = None
        # 如果存在标签，则计算掩码语言建模损失
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        # 如果不需要返回字典格式的输出，则组装最终输出
        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        # 如果需要返回字典格式的输出，则创建MaskedLMOutput对象
        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
class IBertLMHead(nn.Module):
    """I-BERT Head for masked language modeling."""

    def __init__(self, config):
        super().__init__()
        # 初始化一个全连接层，输入和输出维度都是 config.hidden_size
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 初始化 LayerNorm 层，对隐藏层进行归一化，eps 是归一化过程中的小数值
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # 初始化一个全连接层，输入维度是 config.hidden_size，输出维度是 config.vocab_size
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size)
        # 初始化一个偏置参数，大小是 config.vocab_size
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        # 将偏置参数赋给 decoder 层的偏置
        self.decoder.bias = self.bias

    def forward(self, features, **kwargs):
        # 将输入 features 输入全连接层 dense
        x = self.dense(features)
        # 使用 GELU 激活函数处理全连接层输出
        x = gelu(x)
        # 对处理后的结果进行 LayerNorm 归一化
        x = self.layer_norm(x)

        # 使用全连接层 decoder 将结果映射回词汇表大小，加上偏置
        x = self.decoder(x)

        return x

    def _tie_weights(self):
        # 如果两个权重被分离（在TPU上或者当偏置被重新调整大小时），将偏置与 decoder 的偏置相连
        self.bias = self.decoder.bias


@add_start_docstrings(
    """
    I-BERT Model transformer with a sequence classification/regression head on top (a linear layer on top of the pooled
    output) e.g. for GLUE tasks.
    """,
    IBERT_START_DOCSTRING,
)
class IBertForSequenceClassification(IBertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        # 设置分类任务的类别数
        self.num_labels = config.num_labels

        # 初始化 IBertModel，不添加池化层
        self.ibert = IBertModel(config, add_pooling_layer=False)
        # 初始化 IBertClassificationHead
        self.classifier = IBertClassificationHead(config)

        # 初始化权重并进行最终处理
        self.post_init()

    @add_start_docstrings_to_model_forward(IBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=SequenceClassifierOutput,
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
    ) -> Union[SequenceClassifierOutput, Tuple[torch.FloatTensor]]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        # 如果 return_dict 不为 None，则使用指定的 return_dict 值；否则使用 self.config.use_return_dict 的设定
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用预训练模型 `ibert` 进行处理，获取输出结果
        outputs = self.ibert(
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
        # 从模型输出中提取序列输出（一般是经过分类器之前的最后一层隐藏状态）
        sequence_output = outputs[0]
        # 将序列输出传入分类器，得到 logits（预测的分类/回归结果）
        logits = self.classifier(sequence_output)

        # 初始化损失为 None
        loss = None
        # 如果有提供标签 labels
        if labels is not None:
            # 如果问题类型未定义
            if self.config.problem_type is None:
                # 根据 num_labels 的情况设置问题类型
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            # 根据问题类型计算损失
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

        # 如果 return_dict 为 False，返回带有 logits 和其他输出的元组
        if not return_dict:
            output = (logits,) + outputs[2:]  # 将 logits 和额外的输出合并为元组
            return ((loss,) + output) if loss is not None else output

        # 如果 return_dict 为 True，返回 SequenceClassifierOutput 对象
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
@add_start_docstrings(
    """
    I-BERT Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
    softmax) e.g. for RocStories/SWAG tasks.
    """,
    IBERT_START_DOCSTRING,
)
class IBertForMultipleChoice(IBertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        # 初始化 I-BERT 模型
        self.ibert = IBertModel(config)
        # Dropout 层，用于随机失活
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # 分类器，线性层，将隐藏状态映射到单个输出值
        self.classifier = nn.Linear(config.hidden_size, 1)

        # 初始化权重并进行最终处理
        self.post_init()

    @add_start_docstrings_to_model_forward(IBERT_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=MultipleChoiceModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
):
        ) -> Union[MultipleChoiceModelOutput, Tuple[torch.FloatTensor]]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the multiple choice classification loss. Indices should be in `[0, ...,
            num_choices-1]` where `num_choices` is the size of the second dimension of the input tensors. (See
            `input_ids` above)
        """
        # 根据 `return_dict` 是否为 `None` 确定是否使用配置中的 `use_return_dict`
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 获取输入 `input_ids` 的第二维度大小作为 `num_choices`
        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

        # 将 `input_ids`, `position_ids`, `token_type_ids`, `attention_mask`, `inputs_embeds` 扁平化处理
        flat_input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        flat_position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        flat_inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )

        # 调用 `ibert` 模型，传入扁平化的参数，返回模型的输出结果
        outputs = self.ibert(
            flat_input_ids,
            position_ids=flat_position_ids,
            token_type_ids=flat_token_type_ids,
            attention_mask=flat_attention_mask,
            head_mask=head_mask,
            inputs_embeds=flat_inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # 获取汇聚的输出
        pooled_output = outputs[1]

        # 对汇聚的输出应用 dropout
        pooled_output = self.dropout(pooled_output)
        # 使用分类器得出 logits
        logits = self.classifier(pooled_output)
        # 重塑 logits 的形状，以适应多项选择的结构
        reshaped_logits = logits.view(-1, num_choices)

        loss = None
        # 如果存在 `labels`，计算交叉熵损失
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)

        # 如果 `return_dict` 为 False，则返回扁平化后的输出和额外的隐藏状态
        if not return_dict:
            output = (reshaped_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 否则，返回一个包含损失、重塑后的 logits、隐藏状态和注意力的 `MultipleChoiceModelOutput` 对象
        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
@add_start_docstrings(
    """
    I-BERT Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
    Named-Entity-Recognition (NER) tasks.
    """,
    IBERT_START_DOCSTRING,
)
class IBertForTokenClassification(IBertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels  # 从配置中获取标签数量

        self.ibert = IBertModel(config, add_pooling_layer=False)  # 初始化基于IBert的模型，不包含池化层
        self.dropout = nn.Dropout(config.hidden_dropout_prob)  # 使用配置中的dropout概率初始化dropout层
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)  # 使用隐藏层大小和标签数量初始化分类器线性层

        # 初始化权重并进行最终处理
        self.post_init()

    @add_start_docstrings_to_model_forward(IBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
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
    ) -> Union[TokenClassifierOutput, Tuple[torch.FloatTensor]]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用IBert模型的forward方法，传递参数并获取输出
        outputs = self.ibert(
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

        sequence_output = outputs[0]  # 获取IBert模型的输出序列

        sequence_output = self.dropout(sequence_output)  # 应用dropout层到序列输出上
        logits = self.classifier(sequence_output)  # 应用分类器线性层到序列输出上，得到logits

        loss = None
        if labels is not None:
            # 如果提供了标签，计算交叉熵损失
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            # 如果不要求返回字典形式的输出，按原始格式输出
            output = (logits,) + outputs[2:]  # 将logits和其他输出状态组合起来
            return ((loss,) + output) if loss is not None else output

        # 如果要求返回字典形式的输出，构建TokenClassifierOutput对象并返回
        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
class IBertClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        # 定义一个全连接层，输入和输出维度都是 config.hidden_size
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 定义一个 dropout 层，用于防止过拟合，dropout 概率为 config.hidden_dropout_prob
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # 定义一个全连接层，输入维度为 config.hidden_size，输出维度为 config.num_labels
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        # 从 features 中获取每个样本的第一个 token 的隐藏状态，相当于取 [CLS] token
        hidden_states = features[:, 0, :]
        # 对隐藏状态进行 dropout
        hidden_states = self.dropout(hidden_states)
        # 将 dropout 后的隐藏状态输入全连接层进行线性变换
        hidden_states = self.dense(hidden_states)
        # 对全连接层的输出应用 tanh 激活函数
        hidden_states = torch.tanh(hidden_states)
        # 再次对隐藏状态进行 dropout
        hidden_states = self.dropout(hidden_states)
        # 将 dropout 后的隐藏状态输入最终的全连接层进行线性变换，得到模型的输出
        hidden_states = self.out_proj(hidden_states)
        return hidden_states


@add_start_docstrings(
    """
    I-BERT Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
    layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    IBERT_START_DOCSTRING,
)
class IBertForQuestionAnswering(IBertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        # 保存标签数量
        self.num_labels = config.num_labels

        # 初始化 I-BERT 模型，不加入 pooling 层
        self.ibert = IBertModel(config, add_pooling_layer=False)
        # 定义一个全连接层，用于生成问题回答的输出
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        # 初始化模型权重并应用最终处理
        self.post_init()

    @add_start_docstrings_to_model_forward(IBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=QuestionAnsweringModelOutput,
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
        start_positions: Optional[torch.LongTensor] = None,
        end_positions: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[QuestionAnsweringModelOutput, Tuple[torch.FloatTensor]]:
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
        # 设置返回字典是否已经指定，如果未指定则使用模型配置中的设置
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用模型的前向传播，获取模型输出
        outputs = self.ibert(
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

        # 从模型输出中获取序列输出
        sequence_output = outputs[0]

        # 将序列输出传入问答头部，获取起始和结束 logits
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # 如果 start_positions 或 end_positions 是多维的，在第一个维度上进行压缩
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # 忽略超出模型输入的起始/结束位置
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            # 定义交叉熵损失函数，忽略指定的索引
            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            # 计算起始和结束位置的平均损失
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            # 如果不返回字典，则输出损失和 logits 等信息
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        # 返回一个 QuestionAnsweringModelOutput 对象，包括损失、起始和结束 logits，以及其他隐藏状态和注意力信息
        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 根据输入的 `input_ids` 生成对应的位置标识符。非填充符号被替换为它们的位置数字，位置数字从 `padding_idx+1` 开始计数。
# 填充符号被忽略。此函数改编自 fairseq 的 *utils.make_positions*。

def create_position_ids_from_input_ids(input_ids, padding_idx, past_key_values_length=0):
    """
    Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding symbols
    are ignored. This is modified from fairseq's *utils.make_positions*.

    Args:
    input_ids (`torch.LongTensor`):
           Indices of input sequence tokens in the vocabulary.

    Returns: torch.Tensor
    """
    # 使用 input_ids.ne(padding_idx) 生成一个 mask，标记非填充符号为 1，填充符号为 0
    mask = input_ids.ne(padding_idx).int()
    # 在每行中计算累积的非填充符号数量，类型转换为与 mask 相同的类型，然后加上 past_key_values_length
    incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask
    # 将 incremental_indices 转换为长整型（torch.long），然后加上 padding_idx 得到最终的位置标识符
    return incremental_indices.long() + padding_idx
```