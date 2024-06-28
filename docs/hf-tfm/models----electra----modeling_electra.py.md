# `.\models\electra\modeling_electra.py`

```
# coding=utf-8
# Copyright 2019 The Google AI Language Team Authors and The HuggingFace Inc. team.
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
"""PyTorch ELECTRA model."""

import math
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from ...activations import ACT2FN, get_activation
from ...modeling_outputs import (
    BaseModelOutputWithCrossAttentions,
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from ...modeling_utils import PreTrainedModel, SequenceSummary
from ...pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from .configuration_electra import ElectraConfig


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "google/electra-small-discriminator"
_CONFIG_FOR_DOC = "ElectraConfig"

ELECTRA_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "google/electra-small-generator",
    "google/electra-base-generator",
    "google/electra-large-generator",
    "google/electra-small-discriminator",
    "google/electra-base-discriminator",
    "google/electra-large-discriminator",
    # See all ELECTRA models at https://huggingface.co/models?filter=electra
]

def load_tf_weights_in_electra(model, config, tf_checkpoint_path, discriminator_or_generator="discriminator"):
    """Load tf checkpoints in a pytorch model."""
    try:
        import re  # 导入正则表达式模块，用于处理 TensorFlow checkpoint 中的变量名
        import numpy as np  # 导入 NumPy 模块，用于处理数值数据
        import tensorflow as tf  # 导入 TensorFlow 模块，用于加载 TF checkpoint
    except ImportError:
        logger.error(
            "Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions."
        )
        raise
    tf_path = os.path.abspath(tf_checkpoint_path)  # 获取 TF checkpoint 文件的绝对路径
    logger.info(f"Converting TensorFlow checkpoint from {tf_path}")  # 记录日志信息，显示正在转换的 TF checkpoint 路径
    # 从 TF 模型加载权重
    init_vars = tf.train.list_variables(tf_path)  # 获取 TF checkpoint 中的所有变量列表
    names = []  # 初始化空列表，用于存储变量名
    arrays = []  # 初始化空列表，用于存储变量值数组
    # 遍历初始变量列表，每个元素包含变量名和形状信息
    for name, shape in init_vars:
        # 记录日志，显示正在加载的 TensorFlow 权重的名称和形状
        logger.info(f"Loading TF weight {name} with shape {shape}")
        # 使用 TensorFlow 提供的 API 加载指定路径下的变量数据
        array = tf.train.load_variable(tf_path, name)
        # 将变量名添加到名称列表
        names.append(name)
        # 将加载的数组数据添加到数组列表
        arrays.append(array)
    
    # 遍历名称列表和数组列表，这两个列表应该是一一对应的
    for name, array in zip(names, arrays):
        # 原始变量名称，用于异常处理和日志记录
        original_name: str = name

        try:
            # 如果模型是 ElectraForMaskedLM 类型，则更新变量名
            if isinstance(model, ElectraForMaskedLM):
                name = name.replace("electra/embeddings/", "generator/embeddings/")

            # 如果是生成器，更新变量名以匹配生成器的路径
            if discriminator_or_generator == "generator":
                name = name.replace("electra/", "discriminator/")
                name = name.replace("generator/", "electra/")

            # 对一些特定的变量名进行替换，以匹配 PyTorch 模型的命名规则
            name = name.replace("dense_1", "dense_prediction")
            name = name.replace("generator_predictions/output_bias", "generator_lm_head/bias")

            # 按斜杠分割变量名
            name = name.split("/")
            
            # 检查是否有特定的变量名需要跳过处理
            if any(n in ["global_step", "temperature"] for n in name):
                # 记录日志，跳过当前变量的处理
                logger.info(f"Skipping {original_name}")
                continue
            
            # 初始化指针指向模型
            pointer = model
            
            # 遍历变量名的各个部分
            for m_name in name:
                # 如果变量名匹配形如 A_1 的模式，按照下划线分割
                if re.fullmatch(r"[A-Za-z]+_\d+", m_name):
                    scope_names = re.split(r"_(\d+)", m_name)
                else:
                    scope_names = [m_name]
                
                # 根据不同的变量名部分更新指针
                if scope_names[0] == "kernel" or scope_names[0] == "gamma":
                    pointer = getattr(pointer, "weight")
                elif scope_names[0] == "output_bias" or scope_names[0] == "beta":
                    pointer = getattr(pointer, "bias")
                elif scope_names[0] == "output_weights":
                    pointer = getattr(pointer, "weight")
                elif scope_names[0] == "squad":
                    pointer = getattr(pointer, "classifier")
                else:
                    pointer = getattr(pointer, scope_names[0])
                
                # 如果变量名包含索引，则更新指针到具体索引位置
                if len(scope_names) >= 2:
                    num = int(scope_names[1])
                    pointer = pointer[num]
            
            # 如果变量名以 "_embeddings" 结尾，指针更新到嵌入权重
            if m_name.endswith("_embeddings"):
                pointer = getattr(pointer, "weight")
            # 如果变量名为 "kernel"，需要对数组进行转置操作
            elif m_name == "kernel":
                array = np.transpose(array)
            
            # 检查指针和数组的形状是否匹配
            try:
                if pointer.shape != array.shape:
                    raise ValueError(f"Pointer shape {pointer.shape} and array shape {array.shape} mismatched")
            except ValueError as e:
                # 如果形状不匹配，抛出异常
                e.args += (pointer.shape, array.shape)
                raise
            
            # 记录日志，显示正在初始化的 PyTorch 权重的名称和原始名称
            print(f"Initialize PyTorch weight {name}", original_name)
            # 将 NumPy 数组转换为 PyTorch 张量，并赋值给指针指向的属性
            pointer.data = torch.from_numpy(array)
        
        except AttributeError as e:
            # 捕获属性错误异常，记录日志，跳过当前变量的处理
            print(f"Skipping {original_name}", name, e)
            continue
    
    # 返回更新后的模型
    return model
# ElectraEmbeddings 类，用于构建来自单词、位置和标记类型嵌入的嵌入层。
class ElectraEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    # 初始化方法，接收一个 config 参数
    def __init__(self, config):
        super().__init__()
        # 单词嵌入层，根据词汇表大小、嵌入大小和填充标记ID创建 Embedding 对象
        self.word_embeddings = nn.Embedding(config.vocab_size, config.embedding_size, padding_idx=config.pad_token_id)
        # 位置嵌入层，根据最大位置嵌入数量和嵌入大小创建 Embedding 对象
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.embedding_size)
        # 标记类型嵌入层，根据类型词汇表大小和嵌入大小创建 Embedding 对象
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.embedding_size)

        # LayerNorm 没有使用蛇形命名，以便与 TensorFlow 模型变量名保持一致，并能够加载任何 TensorFlow 检查点文件
        self.LayerNorm = nn.LayerNorm(config.embedding_size, eps=config.layer_norm_eps)
        # Dropout 层，使用指定的丢弃概率
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # position_ids 是在内存中连续的，并在序列化时导出
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )
        # 位置嵌入类型，默认为 "absolute"
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        # token_type_ids 初始化为与 position_ids 相同形状的零张量
        self.register_buffer(
            "token_type_ids", torch.zeros(self.position_ids.size(), dtype=torch.long), persistent=False
        )

    # 前向传播方法，接收多个输入参数，并返回嵌入后的张量
    # 代码复制自 transformers.models.bert.modeling_bert.BertEmbeddings.forward
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values_length: int = 0,
    # 定义一个方法，接受输入参数 input_ids（可选），token_type_ids（可选），inputs_embeds（可选），position_ids（可选），past_key_values_length 和返回一个 torch.Tensor 对象
    def forward(
        self,
        input_ids=None,  # 输入的 token IDs
        token_type_ids=None,  # token 类型 IDs，指示每个 token 的类型（如 segment A 或 segment B）
        inputs_embeds=None,  # 输入的嵌入向量
        position_ids=None,  # 位置 IDs，指示每个 token 在序列中的位置
        past_key_values_length=0,  # 过去的键值对长度，用于注意力机制
    ) -> torch.Tensor:
        # 如果给定 input_ids，则获取其形状；否则获取 inputs_embeds 的形状去除最后一维
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        # 获取序列长度
        seq_length = input_shape[1]

        # 如果未提供 position_ids，则从 self.position_ids 中切片获取位置 IDs
        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]

        # 如果未提供 token_type_ids，则检查是否已定义 self.token_type_ids，若已定义则扩展为与输入形状相匹配的全零 tensor
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        # 如果未提供 inputs_embeds，则使用 self.word_embeddings 获取 input_ids 的嵌入向量
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        # 使用 token_type_ids 获取 token 类型的嵌入向量
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        # 将输入的嵌入向量和 token 类型的嵌入向量相加作为最终的嵌入向量
        embeddings = inputs_embeds + token_type_embeddings

        # 如果位置嵌入类型是 "absolute"，则加上位置嵌入向量
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings

        # 对最终的嵌入向量进行 LayerNorm 处理
        embeddings = self.LayerNorm(embeddings)

        # 对处理后的嵌入向量进行 dropout 处理
        embeddings = self.dropout(embeddings)

        # 返回最终的嵌入向量作为输出
        return embeddings
# Copied from transformers.models.bert.modeling_bert.BertSelfAttention with Bert->Electra
class ElectraSelfAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        # 检查隐藏大小是否能被注意力头数整除，若不能且没有embedding_size属性则引发错误
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # 创建查询、键、值的线性层
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        # Dropout 层，用于注意力概率的随机失活
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        
        # 位置嵌入类型，默认为绝对位置嵌入
        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )
        # 如果位置嵌入类型为相对键或相对键查询，则创建距离嵌入层
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

        # 是否作为解码器使用
        self.is_decoder = config.is_decoder

    # 将输入张量 x 转置以适应多头注意力的形状
    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    # 前向传播函数
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ):
        # 这里将输入隐藏状态、注意力掩码等作为参数
        pass

# Copied from transformers.models.bert.modeling_bert.BertSelfOutput
class ElectraSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 全连接层，用于变换隐藏状态的维度
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # LayerNorm 层，用于归一化隐藏状态
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # Dropout 层，用于随机失活隐藏状态
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    # 前向传播函数
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 将输入隐藏状态先通过全连接层、dropout、LayerNorm层，然后与输入张量相加作为输出
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

# Copied from transformers.models.bert.modeling_bert.BertAttention with Bert->Electra
# ElectraAttention 类定义，继承自 nn.Module
class ElectraAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        # 初始化 self 层，即自注意力层，使用 ElectraSelfAttention 类
        self.self = ElectraSelfAttention(config, position_embedding_type=position_embedding_type)
        # 初始化 output 层，即自注意力层输出层，使用 ElectraSelfOutput 类
        self.output = ElectraSelfOutput(config)
        # 初始化一个空集合用于存储已经裁剪的注意力头
        self.pruned_heads = set()

    # 头裁剪方法，用于删除指定的注意力头
    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        # 寻找可以裁剪的注意力头及其索引
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # 裁剪线性层
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # 更新超参数并存储裁剪的注意力头
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    # 前向传播方法
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
        # 使用 self 层进行自注意力计算
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        # 使用 output 层处理自注意力层的输出
        attention_output = self.output(self_outputs[0], hidden_states)
        # 如果需要输出注意力，将注意力张量添加到输出中
        outputs = (attention_output,) + self_outputs[1:]  # 如果有的话，添加注意力
        return outputs


# Copied from transformers.models.bert.modeling_bert.BertIntermediate
# ElectraIntermediate 类定义，继承自 nn.Module
class ElectraIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 线性层，将隐藏状态转换为中间状态，尺寸由 config.hidden_size 到 config.intermediate_size
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        # 如果隐藏激活函数是字符串，使用对应的激活函数；否则使用配置中指定的激活函数
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    # 前向传播方法
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 使用线性层进行转换
        hidden_states = self.dense(hidden_states)
        # 使用中间激活函数处理转换后的隐藏状态
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


# Copied from transformers.models.bert.modeling_bert.BertOutput
# ElectraOutput 类定义，继承自 nn.Module
class ElectraOutput(nn.Module):
    # 类定义略过，因为没有在提供的代码段中完整展示
    # 初始化方法，用于初始化对象
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()
        # 创建一个全连接层，输入大小为config.intermediate_size，输出大小为config.hidden_size
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        # 创建一个 LayerNorm 层，对隐藏状态进行归一化，设置 epsilon 为config.layer_norm_eps
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 创建一个 Dropout 层，用于随机失活一部分神经元，概率为config.hidden_dropout_prob
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    # 前向传播方法，定义了模型的计算过程
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 将隐藏状态通过全连接层进行线性变换
        hidden_states = self.dense(hidden_states)
        # 对线性变换后的结果进行随机失活
        hidden_states = self.dropout(hidden_states)
        # 将随机失活后的结果与输入张量进行残差连接，并对结果进行 LayerNorm 归一化处理
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        # 返回处理后的隐藏状态张量作为输出
        return hidden_states
# 从transformers.models.bert.modeling_bert.BertLayer复制并修改为使用Electra模型
class ElectraLayer(nn.Module):
    # ElectraLayer类的初始化函数，接受一个config参数
    def __init__(self, config):
        super().__init__()
        # 设置前向传播中的块大小（用于分块的前馈网络）
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        # 序列长度的维度，默认为1
        self.seq_len_dim = 1
        # 创建ElectraAttention对象，根据给定的config参数
        self.attention = ElectraAttention(config)
        # 是否作为解码器使用
        self.is_decoder = config.is_decoder
        # 是否添加交叉注意力
        self.add_cross_attention = config.add_cross_attention
        # 如果添加了交叉注意力
        if self.add_cross_attention:
            # 如果不是解码器，抛出错误
            if not self.is_decoder:
                raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
            # 创建带有绝对位置嵌入类型的ElectraAttention对象
            self.crossattention = ElectraAttention(config, position_embedding_type="absolute")
        # 创建ElectraIntermediate对象
        self.intermediate = ElectraIntermediate(config)
        # 创建ElectraOutput对象
        self.output = ElectraOutput(config)

    # 前向传播函数，接收多个Tensor类型的输入参数
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
        # Perform self-attention on the input hidden states using the attention module
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        attention_output = self_attention_outputs[0]

        # if decoder, the last output is tuple of self-attn cache
        if self.is_decoder:
            # Extract all outputs except the last (which is the present key/value) for decoder
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            # Include self-attentions in outputs if we are outputting attention weights
            outputs = self_attention_outputs[1:]

        cross_attn_present_key_value = None
        if self.is_decoder and encoder_hidden_states is not None:
            if not hasattr(self, "crossattention"):
                # Raise error if cross-attention is expected but not defined in the model
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers"
                    " by setting `config.add_cross_attention=True`"
                )

            # cross_attn cached key/values tuple is at positions 3,4 of past_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            # Perform cross-attention using crossattention module
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
            # Append cross-attentions outputs to existing outputs
            outputs = outputs + cross_attention_outputs[1:-1]

            # add cross-attn cache to positions 3,4 of present_key_value tuple
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        # Apply chunking strategy to the feed forward computation for potentially large inputs
        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs

        # if decoder, return the attn key/values as the last output
        if self.is_decoder:
            outputs = outputs + (present_key_value,)

        return outputs

    def feed_forward_chunk(self, attention_output):
        # Process the attention output through intermediate and output layers of the feed forward network
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output
# 从 transformers.models.bert.modeling_bert.BertEncoder 复制代码，并将其中的 "Bert" 替换为 "Electra"
class ElectraEncoder(nn.Module):
    # ElectraEncoder 类的初始化方法
    def __init__(self, config):
        # 调用父类 nn.Module 的初始化方法
        super().__init__()
        # 将传入的配置参数 config 存储到实例变量 self.config 中
        self.config = config
        # 使用列表推导式创建一个 nn.ModuleList，其中包含 config.num_hidden_layers 个 ElectraLayer 实例
        self.layer = nn.ModuleList([ElectraLayer(config) for _ in range(config.num_hidden_layers)])
        # 将梯度检查点功能设为 False
        self.gradient_checkpointing = False

    # ElectraEncoder 类的前向传播方法
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPastAndCrossAttentions]:
        # 如果输出隐藏状态，初始化空元组；否则设为 None
        all_hidden_states = () if output_hidden_states else None
        # 如果输出注意力权重，初始化空元组；否则设为 None
        all_self_attentions = () if output_attentions else None
        # 如果输出交叉注意力权重且配置允许，初始化空元组；否则设为 None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        # 如果开启梯度检查点且在训练中
        if self.gradient_checkpointing and self.training:
            # 如果 use_cache 为 True，给出警告并设置为 False
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # 如果不使用缓存，初始化空元组；否则设为 None
        next_decoder_cache = () if use_cache else None
        # 遍历每个 Transformer 层
        for i, layer_module in enumerate(self.layer):
            # 如果输出隐藏状态，将当前隐藏状态添加到 all_hidden_states 元组中
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 如果有头部掩码，根据索引获取；否则设为 None
            layer_head_mask = head_mask[i] if head_mask is not None else None
            # 如果有过去的键值对，根据索引获取；否则设为 None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            # 如果开启梯度检查点且在训练中，调用梯度检查点函数处理当前层
            if self.gradient_checkpointing and self.training:
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
                # 否则正常调用当前层模块处理输入数据
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )

            # 更新隐藏状态为当前层输出的第一个元素
            hidden_states = layer_outputs[0]
            # 如果使用缓存，将当前层的缓存信息添加到 next_decoder_cache 中
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            # 如果输出注意力权重，将当前层的注意力权重信息添加到 all_self_attentions 中
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                # 如果配置允许，将当前层的交叉注意力权重信息添加到 all_cross_attentions 中
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        # 如果输出隐藏状态，将最终隐藏状态添加到 all_hidden_states 中
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 如果不返回字典形式的输出结果
        if not return_dict:
            # 返回元组形式的结果，排除其中为 None 的元素
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
        # 返回字典形式的 BaseModelOutputWithPastAndCrossAttentions 结果
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )
class ElectraDiscriminatorPredictions(nn.Module):
    """Prediction module for the discriminator, made up of two dense layers."""

    def __init__(self, config):
        super().__init__()
        
        # 初始化第一个全连接层，输入和输出维度都是 config.hidden_size
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        
        # 根据配置获取激活函数，并初始化激活层
        self.activation = get_activation(config.hidden_act)
        
        # 初始化第二个全连接层，输入维度是 config.hidden_size，输出维度是 1
        self.dense_prediction = nn.Linear(config.hidden_size, 1)
        
        # 保存配置信息
        self.config = config

    def forward(self, discriminator_hidden_states):
        # 经过第一个全连接层
        hidden_states = self.dense(discriminator_hidden_states)
        
        # 应用激活函数
        hidden_states = self.activation(hidden_states)
        
        # 经过第二个全连接层得到 logits，并进行压缩
        logits = self.dense_prediction(hidden_states).squeeze(-1)

        return logits


class ElectraGeneratorPredictions(nn.Module):
    """Prediction module for the generator, made up of two dense layers."""

    def __init__(self, config):
        super().__init__()
        
        # 获取激活函数，并初始化激活层
        self.activation = get_activation("gelu")
        
        # 初始化 LayerNorm 层，输入维度是 config.embedding_size
        self.LayerNorm = nn.LayerNorm(config.embedding_size, eps=config.layer_norm_eps)
        
        # 初始化全连接层，输入维度是 config.hidden_size，输出维度是 config.embedding_size
        self.dense = nn.Linear(config.hidden_size, config.embedding_size)

    def forward(self, generator_hidden_states):
        # 经过全连接层
        hidden_states = self.dense(generator_hidden_states)
        
        # 应用激活函数
        hidden_states = self.activation(hidden_states)
        
        # 应用 LayerNorm
        hidden_states = self.LayerNorm(hidden_states)

        return hidden_states


class ElectraPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 使用的配置类
    config_class = ElectraConfig
    
    # 加载 TensorFlow 权重的方法
    load_tf_weights = load_tf_weights_in_electra
    
    # 模型的前缀名称
    base_model_prefix = "electra"
    
    # 是否支持梯度检查点
    supports_gradient_checkpointing = True

    # 来自 transformers.models.bert.modeling_bert.BertPreTrainedModel._init_weights 的方法
    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # 对线性层的权重进行初始化，使用正态分布，标准差为 self.config.initializer_range
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                # 如果存在偏置，则将偏置初始化为零
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            # 对嵌入层的权重进行初始化，使用正态分布，标准差为 self.config.initializer_range
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                # 如果存在 padding_idx，则将对应位置的权重初始化为零
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            # 对 LayerNorm 层的权重初始化，偏置初始化为零，权重初始化为 1.0
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


@dataclass
class ElectraForPreTrainingOutput(ModelOutput):
    """
    Output type of [`ElectraForPreTraining`].
    """
    Args:
        loss (*optional*, returned when `labels` is provided, `torch.FloatTensor` of shape `(1,)`):
            ELECTRA 目标函数的总损失。
            如果提供了 `labels`，则返回此损失。
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
            头部预测分数（SoftMax 前每个标记的分数）。
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            元组包含 `torch.FloatTensor` 类型的张量（当 `output_hidden_states=True` 或 `config.output_hidden_states=True` 时返回）。
            形状为 `(batch_size, sequence_length, hidden_size)`。

            模型每一层的隐藏状态加上初始嵌入输出。
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            元组包含 `torch.FloatTensor` 类型的张量（每层一个）。
            形状为 `(batch_size, num_heads, sequence_length, sequence_length)`。

            自注意力机制中注意力权重经过 softmax 后的结果，用于计算自注意力头的加权平均值。
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
# ELECTRA 模型的文档字符串，描述了模型继承自 PreTrainedModel，并提供了一些通用方法的描述和链接
ELECTRA_START_DOCSTRING = r"""

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`ElectraConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

# ELECTRA 模型的输入文档字符串，当前为空，通常应包含有关输入参数的描述
ELECTRA_INPUTS_DOCSTRING = r"""
"""


@add_start_docstrings(
    "The bare Electra Model transformer outputting raw hidden-states without any specific head on top. Identical to "
    "the BERT model except that it uses an additional linear layer between the embedding layer and the encoder if the "
    "hidden size and embedding size are different. "
    ""
    "Both the generator and discriminator checkpoints may be loaded into this model.",
    ELECTRA_START_DOCSTRING,
)
# ElectraModel 类的定义，继承自 ElectraPreTrainedModel
class ElectraModel(ElectraPreTrainedModel):
    # ElectraModel 类的初始化方法
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)
        # 初始化词嵌入层
        self.embeddings = ElectraEmbeddings(config)

        # 如果 embedding_size 与 hidden_size 不同，添加一个线性映射层
        if config.embedding_size != config.hidden_size:
            self.embeddings_project = nn.Linear(config.embedding_size, config.hidden_size)

        # 初始化编码器
        self.encoder = ElectraEncoder(config)
        self.config = config
        # 初始化权重并应用最终处理
        self.post_init()

    # 获取输入的词嵌入
    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    # 设置输入的词嵌入
    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    # 剪枝模型中的注意力头
    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    # 向模型的前向方法添加文档字符串，描述了输入参数的格式
    @add_start_docstrings_to_model_forward(ELECTRA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    # 向模型的前向方法添加代码示例的文档字符串，包括了加载检查点、输出类型和配置类
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    # 定义 Transformer 模型的前向传播方法，接收多个输入参数
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,  # 输入的 token IDs，可以为空
        attention_mask: Optional[torch.Tensor] = None,  # 注意力 mask，可以为空
        token_type_ids: Optional[torch.Tensor] = None,  # token 类型 IDs，可以为空
        position_ids: Optional[torch.Tensor] = None,  # 位置 IDs，可以为空
        head_mask: Optional[torch.Tensor] = None,  # 头部 mask，可以为空
        inputs_embeds: Optional[torch.Tensor] = None,  # 输入的嵌入表示，可以为空
        encoder_hidden_states: Optional[torch.Tensor] = None,  # 编码器隐藏状态，可以为空
        encoder_attention_mask: Optional[torch.Tensor] = None,  # 编码器注意力 mask，可以为空
        past_key_values: Optional[List[torch.FloatTensor]] = None,  # 历史的键值对，可以为空
        use_cache: Optional[bool] = None,  # 是否使用缓存，可以为空
        output_attentions: Optional[bool] = None,  # 是否输出注意力权重，可以为空
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态，可以为空
        return_dict: Optional[bool] = None,  # 是否返回字典格式的结果，可以为空
# ElectraClassificationHead 类定义，用于处理句子级别的分类任务
class ElectraClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        # 一个全连接层，输入和输出维度都是 config.hidden_size
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 分类器的 dropout 率，如果没有指定，则使用 config.hidden_dropout_prob
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        # 激活函数为 GELU
        self.activation = get_activation("gelu")
        # Dropout 层，使用指定的 dropout 率
        self.dropout = nn.Dropout(classifier_dropout)
        # 输出层全连接层，输入维度为 config.hidden_size，输出维度为 config.num_labels
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        # 取 <s> 标记对应的特征 (等效于 [CLS] 标记)
        x = features[:, 0, :]
        # 应用 dropout
        x = self.dropout(x)
        # 经过全连接层
        x = self.dense(x)
        # 应用激活函数 GELU
        x = self.activation(x)
        # 再次应用 dropout
        x = self.dropout(x)
        # 经过输出全连接层
        x = self.out_proj(x)
        # 返回分类结果
        return x


# ElectraForSequenceClassification 类，继承自 ElectraPreTrainedModel 类，用于序列分类任务
@add_start_docstrings(
    """
    ELECTRA Model transformer with a sequence classification/regression head on top (a linear layer on top of the
    pooled output) e.g. for GLUE tasks.
    """,
    ELECTRA_START_DOCSTRING,
)
class ElectraForSequenceClassification(ElectraPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        # 标签数量
        self.num_labels = config.num_labels
        # 配置
        self.config = config
        # Electra 模型
        self.electra = ElectraModel(config)
        # 序列分类器头部
        self.classifier = ElectraClassificationHead(config)

        # 初始化权重并进行最终处理
        self.post_init()

    @add_start_docstrings_to_model_forward(ELECTRA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint="bhadresh-savani/electra-base-emotion",
        output_type=SequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output="'joy'",
        expected_loss=0.06,
    )
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        # 根据需要确定是否使用返回字典，如果未指定则根据配置决定
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 使用 ELECTRA 模型进行前向传播，获取鉴别器的隐藏状态
        discriminator_hidden_states = self.electra(
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

        # 获取鉴别器输出的序列特征向量
        sequence_output = discriminator_hidden_states[0]

        # 使用分类器对序列特征向量进行分类预测
        logits = self.classifier(sequence_output)

        # 初始化损失为 None
        loss = None

        # 如果存在标签，则计算损失
        if labels is not None:
            # 根据配置确定问题类型，如果未指定则根据标签类型确定
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            # 根据问题类型选择相应的损失函数
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

        # 如果不需要返回字典，则构造输出元组
        if not return_dict:
            output = (logits,) + discriminator_hidden_states[1:]
            return ((loss,) + output) if loss is not None else output

        # 返回带有损失、预测 logits、隐藏状态和注意力权重的 SequenceClassifierOutput 对象
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=discriminator_hidden_states.hidden_states,
            attentions=discriminator_hidden_states.attentions,
        )
@add_start_docstrings(
    """
    Electra model with a binary classification head on top as used during pretraining for identifying generated tokens.

    It is recommended to load the discriminator checkpoint into that model.
    """,
    ELECTRA_START_DOCSTRING,
)
class ElectraForPreTraining(ElectraPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.electra = ElectraModel(config)  # 初始化 Electra 模型
        self.discriminator_predictions = ElectraDiscriminatorPredictions(config)  # 初始化判别器预测组件
        # 初始化权重并进行最终处理
        self.post_init()

    @add_start_docstrings_to_model_forward(ELECTRA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=ElectraForPreTrainingOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,



@add_start_docstrings(
    """
    Electra model with a language modeling head on top.

    Even though both the discriminator and generator may be loaded into this model, the generator is the only model of
    the two to have been trained for the masked language modeling task.
    """,
    ELECTRA_START_DOCSTRING,
)
class ElectraForMaskedLM(ElectraPreTrainedModel):
    _tied_weights_keys = ["generator_lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)

        self.electra = ElectraModel(config)  # 初始化 Electra 模型
        self.generator_predictions = ElectraGeneratorPredictions(config)  # 初始化生成器预测组件

        self.generator_lm_head = nn.Linear(config.embedding_size, config.vocab_size)  # 初始化生成器的语言建模头部
        # 初始化权重并进行最终处理
        self.post_init()

    def get_output_embeddings(self):
        return self.generator_lm_head

    def set_output_embeddings(self, word_embeddings):
        self.generator_lm_head = word_embeddings

    @add_start_docstrings_to_model_forward(ELECTRA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint="google/electra-small-generator",
        output_type=MaskedLMOutput,
        config_class=_CONFIG_FOR_DOC,
        mask="[MASK]",
        expected_output="'paris'",
        expected_loss=1.22,
    )
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,  # 输入的token IDs，可选参数
        attention_mask: Optional[torch.Tensor] = None,  # 注意力掩码，可选参数
        token_type_ids: Optional[torch.Tensor] = None,  # token 类型 IDs，可选参数
        position_ids: Optional[torch.Tensor] = None,  # 位置 IDs，可选参数
        head_mask: Optional[torch.Tensor] = None,  # 头部掩码，可选参数
        inputs_embeds: Optional[torch.Tensor] = None,  # 嵌入的输入，可选参数
        labels: Optional[torch.Tensor] = None,  # 用于计算MLM损失的标签，可选参数
        output_attentions: Optional[bool] = None,  # 是否输出注意力权重，可选参数
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态，可选参数
        return_dict: Optional[bool] = None,  # 是否返回字典形式的输出，可选参数
    ) -> Union[Tuple[torch.Tensor], MaskedLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 通过Electra模型生成隐状态
        generator_hidden_states = self.electra(
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
        generator_sequence_output = generator_hidden_states[0]  # 获取生成器的序列输出

        # 使用生成器预测MLM任务的分数
        prediction_scores = self.generator_predictions(generator_sequence_output)
        prediction_scores = self.generator_lm_head(prediction_scores)  # 应用MLM的softmax层

        loss = None
        # 如果提供了标签，则计算MLM损失
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()  # 交叉熵损失函数，-100索引表示填充标记
            loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        # 如果不要求返回字典形式的输出，则返回元组形式的输出
        if not return_dict:
            output = (prediction_scores,) + generator_hidden_states[1:]
            return ((loss,) + output) if loss is not None else output

        # 返回MaskedLMOutput对象，包含损失、预测logits、隐藏状态和注意力权重
        return MaskedLMOutput(
            loss=loss,
            logits=prediction_scores,
            hidden_states=generator_hidden_states.hidden_states,
            attentions=generator_hidden_states.attentions,
        )
# 定义一个基于 Electra 模型的标记分类器模型
@add_start_docstrings(
    """
    Electra model with a token classification head on top.

    Both the discriminator and generator may be loaded into this model.
    """,
    ELECTRA_START_DOCSTRING,
)
class ElectraForTokenClassification(ElectraPreTrainedModel):
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)
        # 设置分类器的标签数量
        self.num_labels = config.num_labels

        # 加载 Electra 模型
        self.electra = ElectraModel(config)
        # 获取分类器的 dropout 配置，如果未指定则使用隐藏层 dropout 的配置
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        # 定义一个 dropout 层
        self.dropout = nn.Dropout(classifier_dropout)
        # 定义一个线性层，用于分类
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        # 初始化权重并进行最终的处理
        self.post_init()

    # 增加输入文档字符串到模型的前向传播方法
    @add_start_docstrings_to_model_forward(ELECTRA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    # 增加示例代码文档字符串到模型的前向传播方法
    @add_code_sample_docstrings(
        checkpoint="bhadresh-savani/electra-base-discriminator-finetuned-conll03-english",
        output_type=TokenClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output="['B-LOC', 'B-ORG', 'O', 'O', 'O', 'O', 'O', 'B-LOC', 'O', 'B-LOC', 'I-LOC']",
        expected_loss=0.11,
    )
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        # 前向传播方法，接收多种输入参数，返回模型的输出
        # 可选参数包括输入的张量、注意力掩码、token 类型 ID、位置 ID、头部掩码、嵌入的输入张量、标签、是否输出注意力、是否输出隐藏状态、是否返回字典
        ) -> Union[Tuple[torch.Tensor], TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        # 根据 return_dict 参数确定是否返回字典形式的输出
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 使用 Electra 模型进行推断
        discriminator_hidden_states = self.electra(
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
        # 获取 discriminator_hidden_states 的输出序列
        discriminator_sequence_output = discriminator_hidden_states[0]

        # 对输出序列应用 dropout
        discriminator_sequence_output = self.dropout(discriminator_sequence_output)
        # 使用分类器生成 logits
        logits = self.classifier(discriminator_sequence_output)

        # 初始化损失值为 None
        loss = None
        # 如果提供了标签，则计算交叉熵损失
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        # 如果 return_dict 为 False，则按非字典形式返回输出
        if not return_dict:
            output = (logits,) + discriminator_hidden_states[1:]
            return ((loss,) + output) if loss is not None else output

        # 如果 return_dict 为 True，则按 TokenClassifierOutput 对象形式返回输出
        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=discriminator_hidden_states.hidden_states,
            attentions=discriminator_hidden_states.attentions,
        )
@add_start_docstrings(
    """
    ELECTRA Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
    layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    ELECTRA_START_DOCSTRING,
)
# 定义用于问答任务的 ELECTRA 模型，包含一个用于提取式问答任务（如 SQuAD）的跨度分类头部（在隐藏状态输出之上的线性层，
# 用于计算 `span start logits` 和 `span end logits`）。
class ElectraForQuestionAnswering(ElectraPreTrainedModel):
    # 指定配置类
    config_class = ElectraConfig
    # 基础模型前缀
    base_model_prefix = "electra"

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        # ELECTRA 模型
        self.electra = ElectraModel(config)
        # 问答输出层，用于预测答案的开始和结束位置
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        # 初始化权重并应用最终处理
        self.post_init()

    @add_start_docstrings_to_model_forward(ELECTRA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint="bhadresh-savani/electra-base-squad2",
        output_type=QuestionAnsweringModelOutput,
        config_class=_CONFIG_FOR_DOC,
        qa_target_start_index=11,
        qa_target_end_index=12,
        expected_output="'a nice puppet'",
        expected_loss=2.64,
    )
    # 前向传播方法，接收一系列输入并返回预测的答案开始和结束位置的 logit
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        start_positions: Optional[torch.Tensor] = None,
        end_positions: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,



@add_start_docstrings(
    """
    ELECTRA Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
    softmax) e.g. for RocStories/SWAG tasks.
    """,
    ELECTRA_START_DOCSTRING,
)
# 定义用于多选分类任务的 ELECTRA 模型，包含一个用于多选分类任务（如 RocStories/SWAG）的分类头部（线性层放置在池化输出之上，
# 并应用 softmax 操作）。
class ElectraForMultipleChoice(ElectraPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        # ELECTRA 模型
        self.electra = ElectraModel(config)
        # 序列汇总层
        self.sequence_summary = SequenceSummary(config)
        # 分类器层，用于多选任务的分类
        self.classifier = nn.Linear(config.hidden_size, 1)

        # 初始化权重并应用最终处理
        self.post_init()

    @add_start_docstrings_to_model_forward(ELECTRA_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=MultipleChoiceModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,  # 输入的token ids，类型为可选的PyTorch张量
        attention_mask: Optional[torch.Tensor] = None,  # 注意力掩码，类型为可选的PyTorch张量
        token_type_ids: Optional[torch.Tensor] = None,  # token类型 ids，类型为可选的PyTorch张量
        position_ids: Optional[torch.Tensor] = None,  # 位置 ids，类型为可选的PyTorch张量
        head_mask: Optional[torch.Tensor] = None,  # 头部掩码，类型为可选的PyTorch张量
        inputs_embeds: Optional[torch.Tensor] = None,  # 嵌入的输入，类型为可选的PyTorch张量
        labels: Optional[torch.Tensor] = None,  # 标签，用于多项选择分类损失计算的PyTorch张量
        output_attentions: Optional[bool] = None,  # 是否输出注意力权重，类型为可选的布尔值
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态，类型为可选的布尔值
        return_dict: Optional[bool] = None,  # 是否返回字典格式的输出，类型为可选的布尔值
    ) -> Union[Tuple[torch.Tensor], MultipleChoiceModelOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the multiple choice classification loss. Indices should be in `[0, ...,
            num_choices-1]` where `num_choices` is the size of the second dimension of the input tensors. (See
            `input_ids` above)
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict  # 确定是否返回字典格式的输出

        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]  # 获取选择的数量

        input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None  # 将输入token ids重新视图化
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None  # 将注意力掩码重新视图化
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None  # 将token类型 ids重新视图化
        position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None  # 将位置 ids重新视图化
        inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )  # 将嵌入的输入重新视图化

        discriminator_hidden_states = self.electra(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )  # 使用Electra模型进行前向传播，获取鉴别器的隐藏状态

        sequence_output = discriminator_hidden_states[0]  # 获取鉴别器的序列输出

        pooled_output = self.sequence_summary(sequence_output)  # 序列总结，获取池化输出
        logits = self.classifier(pooled_output)  # 分类器，计算logits

        reshaped_logits = logits.view(-1, num_choices)  # 重新形状化logits

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()  # 定义交叉熵损失函数
            loss = loss_fct(reshaped_logits, labels)  # 计算损失

        if not return_dict:
            output = (reshaped_logits,) + discriminator_hidden_states[1:]  # 构造非字典格式的输出
            return ((loss,) + output) if loss is not None else output  # 返回损失和输出，如果损失不为None的话

        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=discriminator_hidden_states.hidden_states,
            attentions=discriminator_hidden_states.attentions,
        )  # 返回字典格式的多项选择模型输出
# 继承自 ElectraPreTrainedModel 类的 ELECTRA 语言模型，添加了用于条件语言建模 fine-tuning 的头部
@add_start_docstrings(
    """ELECTRA Model with a `language modeling` head on top for CLM fine-tuning.""", ELECTRA_START_DOCSTRING
)
class ElectraForCausalLM(ElectraPreTrainedModel):
    # 用于指定权重共享的键名列表
    _tied_weights_keys = ["generator_lm_head.weight"]

    # 初始化方法，接受配置参数 config
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)

        # 如果配置中未指定为解码器，发出警告
        if not config.is_decoder:
            logger.warning("If you want to use `ElectraForCausalLM` as a standalone, add `is_decoder=True.`")

        # 创建 ELECTRA 模型
        self.electra = ElectraModel(config)
        # 创建用于生成预测的预测器对象
        self.generator_predictions = ElectraGeneratorPredictions(config)
        # 创建用于语言建模的线性层，设置输入维度和词汇表大小
        self.generator_lm_head = nn.Linear(config.embedding_size, config.vocab_size)

        # 初始化权重
        self.init_weights()

    # 返回语言建模头部的输出嵌入
    def get_output_embeddings(self):
        return self.generator_lm_head

    # 设置语言建模头部的输出嵌入
    def set_output_embeddings(self, new_embeddings):
        self.generator_lm_head = new_embeddings

    # 前向传播方法，接受多个输入参数，并返回预测的条件语言建模输出
    @add_start_docstrings_to_model_forward(ELECTRA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=CausalLMOutputWithCrossAttentions, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.Tensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        # 此处为 forward 方法的详细文档注释
        """
        Replace these docstrings with ones in transformers.models.**

        """
    
        # 根据输入准备生成过程中需要的输入参数
        def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, **model_kwargs):
            input_shape = input_ids.shape
            # 如果未提供注意力掩码，则创建全 1 的注意力掩码
            if attention_mask is None:
                attention_mask = input_ids.new_ones(input_shape)

            # 如果传入了过去的键值，则根据它们调整输入的 input_ids
            if past_key_values is not None:
                past_length = past_key_values[0][0].shape[2]

                # 某些生成方法可能只传入最后一个输入 ID
                if input_ids.shape[1] > past_length:
                    remove_prefix_length = past_length
                else:
                    # 默认行为：仅保留最后一个 ID
                    remove_prefix_length = input_ids.shape[1] - 1

                input_ids = input_ids[:, remove_prefix_length:]

            # 返回准备好的输入字典
            return {"input_ids": input_ids, "attention_mask": attention_mask, "past_key_values": past_key_values}
    # 定义方法 `_reorder_cache`，用于重新排序模型的缓存 `past_key_values`，以便与给定的 `beam_idx` 对齐
    def _reorder_cache(self, past_key_values, beam_idx):
        # 初始化一个空元组 `reordered_past`，用于存储重新排序后的缓存
        reordered_past = ()
        # 遍历 `past_key_values` 中的每个层的缓存
        for layer_past in past_key_values:
            # 对当前层的每个缓存状态 `past_state` 执行索引选择操作，
            # 根据 `beam_idx` 对应的索引进行选择，并将结果转移到与 `past_state` 相同的设备上
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        # 返回重新排序后的缓存 `reordered_past`
        return reordered_past
```