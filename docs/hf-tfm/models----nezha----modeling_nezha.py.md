# `.\models\nezha\modeling_nezha.py`

```py
# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
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
"""PyTorch Nezha model."""


import math                           # 导入数学库
import os                             # 导入操作系统库
import warnings                      # 导入警告模块
from dataclasses import dataclass     # 导入数据类装饰器
from typing import List, Optional, Tuple, Union   # 导入类型提示工具

import torch                         # 导入PyTorch
import torch.utils.checkpoint         # 导入PyTorch的checkpoint工具
from torch import nn                  # 导入神经网络模块
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss   # 导入损失函数

from ...activations import ACT2FN    # 从相对路径导入激活函数映射
from ...modeling_outputs import (    # 从相对路径导入模型输出类
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    NextSentencePredictorOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from ...modeling_utils import PreTrainedModel   # 从相对路径导入预训练模型类
from ...pytorch_utils import (      # 从相对路径导入PyTorch工具函数
    apply_chunking_to_forward,
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)
from ...utils import (               # 从相对路径导入工具函数和类
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from .configuration_nezha import NezhaConfig   # 从相对路径导入配置文件类


logger = logging.get_logger(__name__)   # 获取当前模块的日志记录器

_CHECKPOINT_FOR_DOC = "sijunhe/nezha-cn-base"   # 文档中使用的检查点模型名称
_CONFIG_FOR_DOC = "NezhaConfig"                # 文档中使用的配置文件名称

NEZHA_PRETRAINED_MODEL_ARCHIVE_LIST = [         # Nezha预训练模型的模型存档列表
    "sijunhe/nezha-cn-base",
    "sijunhe/nezha-cn-large",
    "sijunhe/nezha-base-wwm",
    "sijunhe/nezha-large-wwm",
    # See all Nezha models at https://huggingface.co/models?filter=nezha
]


def load_tf_weights_in_nezha(model, config, tf_checkpoint_path):
    """Load tf checkpoints in a pytorch model."""
    try:
        import re                      # 导入正则表达式模块
        import numpy as np             # 导入NumPy库
        import tensorflow as tf        # 导入TensorFlow库
    except ImportError:
        logger.error(
            "Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions."
        )
        raise
    tf_path = os.path.abspath(tf_checkpoint_path)   # 获取TensorFlow检查点路径的绝对路径
    logger.info(f"Converting TensorFlow checkpoint from {tf_path}")   # 记录日志，显示正在转换的TensorFlow检查点路径
    # 从TF模型加载权重
    init_vars = tf.train.list_variables(tf_path)   # 列出TF模型的所有变量
    names = []                                     # 初始化变量名列表
    arrays = []                                    # 初始化数组列表
    for name, shape in init_vars:                  # 遍历所有初始化的变量和形状
        logger.info(f"Loading TF weight {name} with shape {shape}")   # 记录日志，显示加载的TF权重和其形状
        array = tf.train.load_variable(tf_path, name)   # 加载TF变量
        names.append(name)                           # 将变量名添加到列表
        arrays.append(array)                         # 将加载的数组添加到列表
    for name, array in zip(names, arrays):
        # 将每个变量名按 '/' 分割成列表
        name = name.split("/")
        # 检查变量名中是否包含不需要的项，如 'adam_v', 'adam_m', 'AdamWeightDecayOptimizer' 等
        if any(
            n in ["adam_v", "adam_m", "AdamWeightDecayOptimizer", "AdamWeightDecayOptimizer_1", "global_step"]
            for n in name
        ):
            # 如果包含不需要的项，记录日志并跳过当前循环
            logger.info(f"Skipping {'/'.join(name)}")
            continue
        # 设置指针指向模型的根部
        pointer = model
        # 遍历变量名列表中的每个名称
        for m_name in name:
            # 如果名称匹配类似 'xxx_0' 的格式，按 '_' 分割成列表
            if re.fullmatch(r"[A-Za-z]+_\d+", m_name):
                scope_names = re.split(r"_(\d+)", m_name)
            else:
                scope_names = [m_name]
            # 根据名称的第一个部分决定指针移动的方式
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
                    # 尝试获取指针指向对象的属性
                    pointer = getattr(pointer, scope_names[0])
                except AttributeError:
                    # 如果属性不存在，记录日志并跳过当前循环
                    logger.info(f"Skipping {'/'.join(name)}")
                    continue
            # 如果名称列表长度大于等于2，将第二部分转换为整数，用于索引指针对象
            if len(scope_names) >= 2:
                num = int(scope_names[1])
                pointer = pointer[num]
        # 如果变量名以 '_embeddings' 结尾，设置指针指向权重
        if m_name[-11:] == "_embeddings":
            pointer = getattr(pointer, "weight")
        elif m_name == "kernel":
            # 如果变量名为 'kernel'，对数组进行转置操作
            array = np.transpose(array)
        # 检查指针对象的形状是否与数组的形状匹配
        try:
            if pointer.shape != array.shape:
                # 如果形状不匹配，抛出异常
                raise ValueError(f"Pointer shape {pointer.shape} and array shape {array.shape} mismatched")
        except AssertionError as e:
            # 捕获断言错误，添加指针对象和数组的形状信息，并重新抛出异常
            e.args += (pointer.shape, array.shape)
            raise
        # 记录日志，表示正在初始化 PyTorch 权重
        logger.info(f"Initialize PyTorch weight {name}")
        # 将数组转换为 Torch 张量，并赋值给指针对象的数据属性
        pointer.data = torch.from_numpy(array)
    # 返回更新后的模型对象
    return model
    """Implement the Functional Relative Position Encoding"""
    # 实现函数式相对位置编码

    def __init__(self, length, depth, max_relative_position=127):
        super().__init__()
        # 获得词汇表大小
        vocab_size = max_relative_position * 2 + 1
        # 创建长度范围向量
        range_vec = torch.arange(length)
        # 创建长度范围矩阵
        range_mat = range_vec.repeat(length).view(length, length)
        # 创建距离矩阵
        distance_mat = range_mat - torch.t(range_mat)
        # 将距离矩阵裁剪在[-max_relative_position, max_relative_position]范围内
        distance_mat_clipped = torch.clamp(distance_mat, -max_relative_position, max_relative_position)
        # 最终的相对位置矩阵
        final_mat = distance_mat_clipped + max_relative_position

        # 创建空的嵌入表
        embeddings_table = torch.zeros(vocab_size, depth)
        # 创建位置矩阵
        position = torch.arange(0, vocab_size, dtype=torch.int64).float().unsqueeze(1)
        # 创建除数项
        div_term = torch.exp(torch.arange(0, depth, 2).float() * (-math.log(10000.0) / depth))
        # 使用正弦函数填充偶数列
        embeddings_table[:, 0::2] = torch.sin(position * div_term)
        # 使用余弦函数填充奇数列
        embeddings_table[:, 1::2] = torch.cos(position * div_term)

        # 将最终的相对位置矩阵展平
        flat_relative_positions_matrix = final_mat.view(-1)
        # 创建独热编码的相对位置矩阵
        one_hot_relative_positions_matrix = torch.nn.functional.one_hot(
            flat_relative_positions_matrix, num_classes=vocab_size
        ).float()
        # 计算位置编码矩阵
        positions_encoding = torch.matmul(one_hot_relative_positions_matrix, embeddings_table)
        # 调整位置编码矩阵形状
        my_shape = list(final_mat.size())
        my_shape.append(depth)
        positions_encoding = positions_encoding.view(my_shape)
        # 将位置编码矩阵注册为模型的缓冲区
        self.register_buffer("positions_encoding", positions_encoding, persistent=False)

    def forward(self, length):
        # 返回指定长度的位置编码矩阵
        return self.positions_encoding[:length, :length, :]


class NezhaEmbeddings(nn.Module):
    """Construct the embeddings from word and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        # 创建词嵌入层
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        # 创建标记类型嵌入层
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # 与 TensorFlow 模型变量名保持一致以便加载任何 TensorFlow 检查点文件
        # 这里的 self.LayerNorm 不使用蛇形命名法
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 创建丢弃层
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # 注册一个零张量作为标记类型的缓冲区
        self.register_buffer(
            "token_type_ids", torch.zeros((1, config.max_position_embeddings), dtype=torch.long), persistent=False
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
    ) -> torch.Tensor:
        # 如果给定了 input_ids，则获取其形状作为 input_shape
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            # 否则，从 inputs_embeds 中获取形状，去掉最后一个维度
            input_shape = inputs_embeds.size()[:-1]

        # 获取序列长度，这里假定 input_shape 是 (batch_size, seq_length)
        seq_length = input_shape[1]

        # 如果 inputs_embeds 为 None，则使用 word_embeddings 对 input_ids 进行嵌入
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        # 设置 token_type_ids，如果未提供，则使用在构造函数中注册的缓冲区，通常为全零。这种设置通常在模型跟踪时帮助用户，避免了手动传入 token_type_ids 的问题，解决了 issue #5664
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                # 从模型中注册的缓冲区获取 token_type_ids，并截取到当前序列长度
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                # 将 buffered_token_type_ids 扩展为 input_shape[0] 行，seq_length 列的张量
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                # 如果模型中未定义 token_type_ids，创建一个全零张量作为 token_type_ids
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=inputs_embeds.device)

        # 根据 token_type_ids 获取 token_type_embeddings
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        # 将 inputs_embeds 和 token_type_embeddings 相加得到最终的 embeddings
        embeddings = inputs_embeds + token_type_embeddings
        # 使用 LayerNorm 对 embeddings 进行归一化
        embeddings = self.LayerNorm(embeddings)
        # 对 embeddings 进行 dropout 处理
        embeddings = self.dropout(embeddings)
        # 返回最终的 embeddings 张量
        return embeddings
# 定义 NezhaSelfAttention 类，继承自 nn.Module
class NezhaSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 检查隐藏层大小是否是注意力头数的整数倍，如果不是则抛出数值错误
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        # 初始化注意力头数和每个头的大小
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # 初始化查询、键、值的线性层
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        # 初始化 dropout 层
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        
        # 初始化相对位置编码
        self.relative_positions_encoding = NezhaRelativePositionsEncoding(
            length=config.max_position_embeddings,
            depth=self.attention_head_size,
            max_relative_position=config.max_relative_position,
        )
        
        # 是否作为解码器的标志
        self.is_decoder = config.is_decoder

    # 将输入张量变换为注意力分数张量的形状
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
        # 这里的前向传播功能将根据输入计算注意力分数并返回相应输出
        # 具体实现需要查看具体的代码逻辑和数学计算过程



# 从 transformers.models.bert.modeling_bert.BertSelfOutput 复制过来，修改为 NezhaSelfOutput 类
class NezhaSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 全连接层，将隐藏状态映射回原始大小
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # LayerNorm 层，归一化隐藏状态
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # dropout 层
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    # 前向传播函数，计算隐藏状态的输出
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 全连接层映射
        hidden_states = self.dense(hidden_states)
        # dropout
        hidden_states = self.dropout(hidden_states)
        # LayerNorm 归一化并添加输入张量
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states



# 定义 NezhaAttention 类，继承自 nn.Module
class NezhaAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 初始化自注意力层和自注意力输出层
        self.self = NezhaSelfAttention(config)
        self.output = NezhaSelfOutput(config)
        # 初始化剪枝的注意力头集合
        self.pruned_heads = set()
    # 剪枝注意力头部，排除头部列表为空的情况
    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        
        # 调用辅助函数找到可剪枝的注意力头部及其索引
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # 剪枝线性层的查询、键、值和输出层
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # 更新超参数并记录剪枝的注意力头部
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    # 前向传播函数，接收多个参数并返回一个元组
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
        # 调用 self 层的前向传播方法，得到 self_outputs
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        
        # 将 self_outputs 的第一个元素与原始隐藏状态传递给输出层，得到注意力输出
        attention_output = self.output(self_outputs[0], hidden_states)
        
        # 构造输出元组，包含注意力输出和可能的其他输出
        outputs = (attention_output,) + self_outputs[1:]  # 如果有输出注意力，将其加入输出元组
        
        # 返回所有输出
        return outputs
# 从transformers.models.bert.modeling_bert.BertIntermediate复制而来，将Bert替换为Nezha
class NezhaIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 创建一个线性层，将输入的隐藏大小转换为中间大小
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        # 如果配置中的隐藏激活函数是字符串，则使用ACT2FN字典中对应的函数，否则直接使用配置中的激活函数
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 通过线性层转换隐藏状态的维度
        hidden_states = self.dense(hidden_states)
        # 应用中间激活函数到转换后的隐藏状态
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


# 从transformers.models.bert.modeling_bert.BertOutput复制而来，将Bert替换为Nezha
class NezhaOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 创建一个线性层，将中间大小的特征映射回隐藏大小
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        # 创建一个LayerNorm层，用于标准化隐藏状态
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 创建一个Dropout层，用于随机置零隐藏状态的部分单元
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 通过线性层将中间大小的特征映射回隐藏大小
        hidden_states = self.dense(hidden_states)
        # 对映射后的隐藏状态应用Dropout
        hidden_states = self.dropout(hidden_states)
        # 对映射后的隐藏状态应用LayerNorm，并将输入张量与其相加
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


# NezhaLayer类定义，用于构建Nezha模型的一个层
class NezhaLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 设置前向传播中的分块大小
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        # 设置序列长度维度
        self.seq_len_dim = 1
        # 创建NezhaAttention层对象
        self.attention = NezhaAttention(config)
        # 检查是否为解码器模型，如果是，则添加跨注意力
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            if not self.is_decoder:
                # 如果非解码器模型且添加了跨注意力，抛出错误
                raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
            # 创建另一个NezhaAttention层对象，用于跨注意力
            self.crossattention = NezhaAttention(config)
        # 创建NezhaIntermediate层对象，用于转换隐藏状态到中间状态
        self.intermediate = NezhaIntermediate(config)
        # 创建NezhaOutput层对象，用于将中间状态转换回隐藏状态
        self.output = NezhaOutput(config)

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
        # NezhaLayer的前向传播方法，将输入隐藏状态传递给Nezha模型的每个组件，并返回隐藏状态的转换结果
    ) -> Tuple[torch.Tensor]:
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        # Perform self-attention operation using the attention module
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        # Extract the self-attention output tensor
        attention_output = self_attention_outputs[0]

        # if decoder, the last output is tuple of self-attn cache
        if self.is_decoder:
            # Extract all outputs except the last one which is the self-attention cache
            outputs = self_attention_outputs[1:-1]
            # Extract the present key/value tuple
            present_key_value = self_attention_outputs[-1]
        else:
            # Include self attentions in outputs if attention weights are to be output
            outputs = self_attention_outputs[1:]

        cross_attn_present_key_value = None
        if self.is_decoder and encoder_hidden_states is not None:
            if not hasattr(self, "crossattention"):
                # Raise error if cross-attention layers are not instantiated when needed
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers"
                    " by setting `config.add_cross_attention=True`"
                )

            # cross_attn cached key/values tuple is at positions 3,4 of past_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            # Perform cross-attention operation using the crossattention module
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                cross_attn_past_key_value,
                output_attentions,
            )
            # Extract the cross-attention output tensor
            attention_output = cross_attention_outputs[0]
            # Append cross-attention outputs to the existing outputs list
            outputs = outputs + cross_attention_outputs[1:-1]

            # add cross-attn cache to positions 3,4 of present_key_value tuple
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        # Apply chunking to the feed forward step and compute layer output
        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        # Append layer_output to outputs tuple
        outputs = (layer_output,) + outputs

        # if decoder, return the attn key/values as the last output
        if self.is_decoder:
            # Append present key/values to outputs if decoder
            outputs = outputs + (present_key_value,)

        # Return the final outputs tuple
        return outputs

    def feed_forward_chunk(self, attention_output):
        # Compute intermediate output using the intermediate module
        intermediate_output = self.intermediate(attention_output)
        # Compute final layer output using the output module
        layer_output = self.output(intermediate_output, attention_output)
        # Return the final layer output
        return layer_output
# 从transformers.models.bert.modeling_bert.BertEncoder复制过来，并将Bert->Nezha
class NezhaEncoder(nn.Module):
    # 初始化方法，接受一个config对象作为参数
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()
        # 将传入的config对象保存到实例变量self.config中
        self.config = config
        # 创建一个由NezhaLayer对象组成的ModuleList，列表长度为config.num_hidden_layers
        self.layer = nn.ModuleList([NezhaLayer(config) for _ in range(config.num_hidden_layers)])
        # 设置梯度检查点为False
        self.gradient_checkpointing = False

    # 前向传播方法定义
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
        all_hidden_states = () if output_hidden_states else None
        # 如果输出隐藏状态为真，则初始化空元组以存储所有隐藏状态，否则设为None
        all_self_attentions = () if output_attentions else None
        # 如果输出自注意力权重为真，则初始化空元组以存储所有自注意力权重，否则设为None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        # 如果输出自注意力和交叉注意力为真且模型配置中包含交叉注意力，则初始化空元组以存储所有交叉注意力，否则设为None

        if self.gradient_checkpointing and self.training:
            # 如果启用了梯度检查点并且处于训练状态
            if use_cache:
                # 如果使用缓存，则发出警告并设置use_cache为False，因为与梯度检查点不兼容
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        next_decoder_cache = () if use_cache else None
        # 如果使用缓存，则初始化空元组以存储下一个解码器缓存，否则设为None
        for i, layer_module in enumerate(self.layer):
            # 遍历每个层模块

            if output_hidden_states:
                # 如果需要输出隐藏状态，则将当前隐藏状态添加到all_hidden_states元组中
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            # 获取当前层的头部掩码，如果头部掩码存在的话

            past_key_value = past_key_values[i] if past_key_values is not None else None
            # 获取过去的键值对，如果过去的键值对存在的话

            if self.gradient_checkpointing and self.training:
                # 如果启用了梯度检查点并且处于训练状态
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
                # 否则正常调用当前层模块
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )

            hidden_states = layer_outputs[0]
            # 更新当前隐藏状态为当前层的输出的第一个元素（即隐藏状态）

            if use_cache:
                # 如果使用缓存，则将当前层的输出的最后一个元素（即下一个解码器缓存）添加到next_decoder_cache中
                next_decoder_cache += (layer_outputs[-1],)

            if output_attentions:
                # 如果需要输出注意力权重
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                # 将当前层的输出的第二个元素（自注意力权重）添加到all_self_attentions中
                if self.config.add_cross_attention:
                    # 如果模型配置中包含交叉注意力
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)
                    # 将当前层的输出的第三个元素（交叉注意力权重）添加到all_cross_attentions中

        if output_hidden_states:
            # 如果需要输出隐藏状态，则将最终的当前隐藏状态添加到all_hidden_states中
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            # 如果不返回字典形式的结果
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
            # 返回包含非空值的元组作为输出

        return BaseModelOutputWithPastAndCrossAttentions(
            # 否则返回基础模型输出与过去和交叉注意力
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )
# Copied from transformers.models.bert.modeling_bert.BertPooler with Bert->Nezha
class NezhaPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Initialize a fully connected layer for pooling hidden states
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # Activation function used after pooling
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Pooling operation: take the hidden state corresponding to the first token
        first_token_tensor = hidden_states[:, 0]
        # Pass through the pooling dense layer
        pooled_output = self.dense(first_token_tensor)
        # Apply activation function
        pooled_output = self.activation(pooled_output)
        return pooled_output


# Copied from transformers.models.bert.modeling_bert.BertPredictionHeadTransform with Bert->Nezha
class NezhaPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Initialize a fully connected layer for transformation
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # Activation function determined by the config
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        # Layer normalization
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Transform the hidden states using the dense layer
        hidden_states = self.dense(hidden_states)
        # Apply activation function
        hidden_states = self.transform_act_fn(hidden_states)
        # Apply layer normalization
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


# Copied from transformers.models.bert.modeling_bert.BertLMPredictionHead with Bert->Nezha
class NezhaLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Initialize prediction head transformation module
        self.transform = NezhaPredictionHeadTransform(config)

        # Decoder layer: projects hidden states to output vocab size
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Bias parameter for output layer
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # Link bias to the decoder to adjust with token embeddings resizing
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        # Transform hidden states
        hidden_states = self.transform(hidden_states)
        # Decode to get prediction scores
        hidden_states = self.decoder(hidden_states)
        return hidden_states


# Copied from transformers.models.bert.modeling_bert.BertOnlyMLMHead with Bert->Nezha
class NezhaOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        # MLM predictions using NezhaLMPredictionHead
        self.predictions = NezhaLMPredictionHead(config)

    def forward(self, sequence_output: torch.Tensor) -> torch.Tensor:
        # Get MLM prediction scores from sequence output
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


# Copied from transformers.models.bert.modeling_bert.BertOnlyNSPHead with Bert->Nezha
class NezhaOnlyNSPHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Next sentence prediction using linear layer
        self.seq_relationship = nn.Linear(config.hidden_size, 2)
    # 定义一个类方法 `forward`，用于执行模型的前向传播
    def forward(self, pooled_output):
        # 调用 `seq_relationship` 方法，传入 `pooled_output` 参数，计算序列关系分数
        seq_relationship_score = self.seq_relationship(pooled_output)
        # 返回计算得到的序列关系分数作为方法的输出结果
        return seq_relationship_score
# 从 transformers.models.bert.modeling_bert.BertPreTrainingHeads 复制并修改为 Nezha 模型的预训练头部
class NezhaPreTrainingHeads(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 创建 NezhaLMPredictionHead 对象用于预测下一个词的概率分布
        self.predictions = NezhaLMPredictionHead(config)
        # 创建一个线性层用于序列关系预测，输出维度为 2
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, sequence_output, pooled_output):
        # 通过预测头部获取预测分数，用于预测下一个词的概率分布
        prediction_scores = self.predictions(sequence_output)
        # 通过线性层获取序列关系分数，用于预测句子关系
        seq_relationship_score = self.seq_relationship(pooled_output)
        # 返回预测分数和序列关系分数作为输出
        return prediction_scores, seq_relationship_score


class NezhaPreTrainedModel(PreTrainedModel):
    """
    一个处理权重初始化、下载和加载预训练模型的抽象类。
    """

    # Nezha 模型的配置类
    config_class = NezhaConfig
    # 加载 TensorFlow 权重的函数
    load_tf_weights = load_tf_weights_in_nezha
    # Nezha 模型的前缀
    base_model_prefix = "nezha"
    # 支持梯度检查点
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """初始化权重"""
        if isinstance(module, nn.Linear):
            # 使用正态分布初始化线性层的权重，均值为 0，标准差为配置中的初始化范围
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                # 如果有偏置项，初始化为零
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            # 使用正态分布初始化嵌入层的权重，均值为 0，标准差为配置中的初始化范围
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                # 如果有填充索引，将填充索引位置的权重初始化为零
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            # 初始化 LayerNorm 层的偏置为零，权重为 1
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


@dataclass
class NezhaForPreTrainingOutput(ModelOutput):
    """
    NezhaForPreTraining 的输出类型。
    """
    """
    loss: Optional[torch.FloatTensor] = None
    prediction_logits: torch.FloatTensor = None
    seq_relationship_logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    """
"""
    This constant defines a docstring for Nezha models, providing an overview and usage guidelines.

    It inherits from `PreTrainedModel`, indicating that it leverages methods defined in the superclass 
    for tasks like downloading, saving, resizing embeddings, and pruning heads.

    Additionally, it specifies that the model is a subclass of PyTorch's `torch.nn.Module`, implying 
    that it can be used as a regular PyTorch module. Users are directed to consult the PyTorch 
    documentation for general usage and behavior details.

    Parameters:
        config (`NezhaConfig`): A configuration object holding all model parameters. When initializing 
            with a config file, only configuration settings are loaded, not model weights. Refer to 
            `PreTrainedModel.from_pretrained` to load both configuration and weights.
"""
    Args:
        input_ids (`torch.LongTensor` of shape `({0})`):
            # 输入序列标记在词汇表中的索引。

            # 可以使用 [`AutoTokenizer`] 获取这些索引。有关详细信息，请参阅 [`PreTrainedTokenizer.encode`] 和
            # [`PreTrainedTokenizer.__call__`]。

            # [什么是输入 ID？](../glossary#input-ids)
        attention_mask (`torch.FloatTensor` of shape `({0})`, *optional*):
            # 遮罩，用于避免在填充的标记索引上执行注意力操作。遮罩的值为 `[0, 1]`：

            # - 1 表示**不遮罩**的标记，
            # - 0 表示**遮罩**的标记。

            # [什么是注意力遮罩？](../glossary#attention-mask)
        token_type_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            # 段标记索引，指示输入的第一部分和第二部分。索引在 `[0, 1]` 中选择：

            # - 0 对应于*句子 A*的标记，
            # - 1 对应于*句子 B*的标记。

            # [什么是标记类型 ID？](../glossary#token-type-ids)
        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            # 用于将自注意力模块的选定头部置零的遮罩。遮罩的值在 `[0, 1]` 中选择：

            # - 1 表示头部**不遮罩**，
            # - 0 表示头部**遮罩**。

        inputs_embeds (`torch.FloatTensor` of shape `({0}, hidden_size)`, *optional*):
            # 可选，您可以选择直接传递嵌入表示，而不是传递 `input_ids`。如果您想对如何将 `input_ids` 索引转换为关联向量
            # 有更多控制权，则这非常有用，而不是使用模型的内部嵌入查找矩阵。
        output_attentions (`bool`, *optional*):
            # 是否返回所有注意力层的注意力张量。有关详细信息，请参阅返回张量中的 `attentions`。
        output_hidden_states (`bool`, *optional*):
            # 是否返回所有层的隐藏状态。有关详细信息，请参阅返回张量中的 `hidden_states`。
        return_dict (`bool`, *optional*):
            # 是否返回 [`~utils.ModelOutput`] 而不是普通元组。
# 定义 NezhaModel 类，继承自 NezhaPreTrainedModel
@add_start_docstrings(
    "The bare Nezha Model transformer outputting raw hidden-states without any specific head on top.",
    NEZHA_START_DOCSTRING,
)
class NezhaModel(NezhaPreTrainedModel):
    """
    # 初始化函数
    def __init__(self, config, add_pooling_layer=True):
        # 调用父类初始化函数
        super().__init__(config)
        # 保存配置
        self.config = config
        # 初始化嵌入层
        self.embeddings = NezhaEmbeddings(config)
        # 初始化编码器
        self.encoder = NezhaEncoder(config)
        # 如果需要添加池化层，则初始化池化层
        self.pooler = NezhaPooler(config) if add_pooling_layer else None
        # 初始化权重并应用最终处理
        self.post_init()

    # 获取输入嵌入层
    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    # 设置输入嵌入层
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

    # 前向传播函数
    @add_start_docstrings_to_model_forward(NEZHA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPoolingAndCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
@add_start_docstrings(
    """
    Nezha Model with two heads on top as done during the pretraining: a `masked language modeling` head and a `next
    sentence prediction (classification)` head.
    """
    这是一个多行字符串（docstring），通常用来描述函数或类的作用、参数、返回值等信息。
    在这里，它描述的是一个叫做 "sentence prediction (classification)" 的部分的头部信息。
    """
    NEZHA_START_DOCSTRING,
# 定义 NezhaForPreTraining 类，继承自 NezhaPreTrainedModel
class NezhaForPreTraining(NezhaPreTrainedModel):
    # 定义 tied_weights_keys 类属性，指定可共享权重的键名
    _tied_weights_keys = ["cls.predictions.decoder"]

    # 初始化方法，接收一个配置对象 config
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)

        # 创建 NezhaModel 实例，并保存到 self.nezha 属性中
        self.nezha = NezhaModel(config)
        # 创建 NezhaPreTrainingHeads 实例，并保存到 self.cls 属性中
        self.cls = NezhaPreTrainingHeads(config)

        # 执行后续初始化和处理步骤
        self.post_init()

    # 获取输出嵌入的方法，返回预测层的解码器
    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    # 设置输出嵌入的方法，用新的嵌入替换预测层的解码器
    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    # 前向传播方法，接收多个输入参数，并返回模型输出
    @add_start_docstrings_to_model_forward(NEZHA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=NezhaForPreTrainingOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        next_sentence_label: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,



# 定义 NezhaForMaskedLM 类，继承自 NezhaPreTrainedModel
@add_start_docstrings("""Nezha Model with a `language modeling` head on top.""", NEZHA_START_DOCSTRING)
class NezhaForMaskedLM(NezhaPreTrainedModel):
    # 定义 tied_weights_keys 类属性，指定可共享权重的键名
    _tied_weights_keys = ["cls.predictions.decoder"]

    # 初始化方法，接收一个配置对象 config
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)

        # 如果配置中标明是解码器，则发出警告
        if config.is_decoder:
            logger.warning(
                "If you want to use `NezhaForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        # 创建 NezhaModel 实例，设置不添加池化层，并保存到 self.nezha 属性中
        self.nezha = NezhaModel(config, add_pooling_layer=False)
        # 创建 NezhaOnlyMLMHead 实例，并保存到 self.cls 属性中
        self.cls = NezhaOnlyMLMHead(config)

        # 执行后续初始化和处理步骤
        self.post_init()

    # 获取输出嵌入的方法，返回预测层的解码器
    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    # 设置输出嵌入的方法，用新的嵌入替换预测层的解码器
    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    # 前向传播方法，接收多个输入参数，并返回模型输出
    @add_start_docstrings_to_model_forward(NEZHA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=MaskedLMOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,  # 输入的token IDs序列，可以为空
        attention_mask: Optional[torch.Tensor] = None,  # 注意力掩码，用于指示哪些token是真实值，哪些是填充值
        token_type_ids: Optional[torch.Tensor] = None,  # token类型IDs，例如segment IDs对于BERT模型
        head_mask: Optional[torch.Tensor] = None,  # 多头注意力掩码，用于遮蔽某些注意力头
        inputs_embeds: Optional[torch.Tensor] = None,  # 嵌入输入的张量表示
        encoder_hidden_states: Optional[torch.Tensor] = None,  # 编码器隐藏状态，用于Transformer类模型
        encoder_attention_mask: Optional[torch.Tensor] = None,  # 编码器的注意力掩码
        labels: Optional[torch.Tensor] = None,  # 用于计算MLM损失的标签
        output_attentions: Optional[bool] = None,  # 是否输出注意力权重
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态
        return_dict: Optional[bool] = None,  # 是否以字典格式返回输出
    ) -> Union[Tuple[torch.Tensor], MaskedLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict  # 确定是否使用字典格式返回输出结果

        # 调用NeZha模型的前向传播函数，传入各种参数
        outputs = self.nezha(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]  # 取出模型输出的序列输出
        prediction_scores = self.cls(sequence_output)  # 使用分类头对序列输出进行预测得分计算

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()  # 定义交叉熵损失函数，用于计算MLM损失
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))  # 计算MLM损失

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]  # 如果不返回字典，则构造输出元组
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output  # 返回带有损失的输出元组或者纯输出元组

        # 返回MaskedLMOutput对象，其中包括损失、logits、隐藏状态和注意力权重
        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    # 准备生成模型输入，处理输入的标识符（token）和注意力掩码
    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, **model_kwargs):
        # 获取输入标识符的形状信息
        input_shape = input_ids.shape
        # 获取有效的批次大小
        effective_batch_size = input_shape[0]

        # 添加一个虚拟标记（dummy token）
        # 如果配置中未定义PAD标记，则抛出数值错误异常
        if self.config.pad_token_id is None:
            raise ValueError("The PAD token should be defined for generation")

        # 将注意力掩码末尾添加一个全零列，扩展其形状
        attention_mask = torch.cat([attention_mask, attention_mask.new_zeros((attention_mask.shape[0], 1))], dim=-1)
        
        # 创建一个全为PAD标记的虚拟标记，形状为（有效批次大小，1），并放置在与输入标识符相同的设备上
        dummy_token = torch.full(
            (effective_batch_size, 1), self.config.pad_token_id, dtype=torch.long, device=input_ids.device
        )
        
        # 将虚拟标记添加到输入标识符的末尾，扩展其长度
        input_ids = torch.cat([input_ids, dummy_token], dim=1)

        # 返回处理后的输入字典，包括更新后的输入标识符和注意力掩码
        return {"input_ids": input_ids, "attention_mask": attention_mask}
# 使用特定的文档字符串描述 Nezha 模型，该模型在顶部包含一个用于下一句预测（分类）的头部。
# 引用了 NEZHA_START_DOCSTRING 中定义的文档字符串。
@add_start_docstrings(
    """Nezha Model with a `next sentence prediction (classification)` head on top.""",
    NEZHA_START_DOCSTRING,
)
class NezhaForNextSentencePrediction(NezhaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        # 初始化 Nezha 模型，使用给定的配置
        self.nezha = NezhaModel(config)
        # 初始化仅包含 NSP（Next Sentence Prediction）头部的组件
        self.cls = NezhaOnlyNSPHead(config)

        # 初始化权重并应用最终处理
        self.post_init()

    @add_start_docstrings_to_model_forward(NEZHA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=NextSentencePredictorOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
):
        ) -> Union[Tuple[torch.Tensor], NextSentencePredictorOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the next sequence prediction (classification) loss. Input should be a sequence pair
            (see `input_ids` docstring). Indices should be in `[0, 1]`:

            - 0 indicates sequence B is a continuation of sequence A,
            - 1 indicates sequence B is a random sequence.

        Returns:
            Depending on `return_dict`, either a tuple with `NextSentencePredictorOutput` or separate elements.

        Example:

        ```
        >>> from transformers import AutoTokenizer, NezhaForNextSentencePrediction
        >>> import torch

        >>> tokenizer = AutoTokenizer.from_pretrained("sijunhe/nezha-cn-base")
        >>> model = NezhaForNextSentencePrediction.from_pretrained("sijunhe/nezha-cn-base")

        >>> prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
        >>> next_sentence = "The sky is blue due to the shorter wavelength of blue light."
        >>> encoding = tokenizer(prompt, next_sentence, return_tensors="pt")

        >>> outputs = model(**encoding, labels=torch.LongTensor([1]))
        >>> logits = outputs.logits
        >>> assert logits[0, 0] < logits[0, 1]  # next sentence was random
        ```
        """

        if "next_sentence_label" in kwargs:
            # 发出警告，告知用户 `next_sentence_label` 参数即将被弃用，应使用 `labels`
            warnings.warn(
                "The `next_sentence_label` argument is deprecated and will be removed in a future version, use"
                " `labels` instead.",
                FutureWarning,
            )
            # 将 `next_sentence_label` 赋值给 `labels` 变量，并从 `kwargs` 中删除该参数
            labels = kwargs.pop("next_sentence_label")

        # 确定是否返回字典形式的输出，如果未指定则使用配置中的默认设置
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用预训练模型的主要处理逻辑 `nezha` 方法，传入各类参数
        outputs = self.nezha(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 从模型输出中获取池化后的输出
        pooled_output = outputs[1]

        # 将池化输出传递给分类层 `cls`，得到序列关系分数
        seq_relationship_scores = self.cls(pooled_output)

        next_sentence_loss = None
        # 如果 `labels` 不为空，则计算下一个句子预测的损失
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            next_sentence_loss = loss_fct(seq_relationship_scores.view(-1, 2), labels.view(-1))

        # 根据 `return_dict` 决定返回的格式，如果不返回字典，则返回元组形式的输出
        if not return_dict:
            output = (seq_relationship_scores,) + outputs[2:]
            return ((next_sentence_loss,) + output) if next_sentence_loss is not None else output

        # 返回包含损失、预测分数、隐藏状态和注意力权重的 `NextSentencePredictorOutput` 对象
        return NextSentencePredictorOutput(
            loss=next_sentence_loss,
            logits=seq_relationship_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 使用 Nezha 模型来实现序列分类或回归任务的模型转换器，顶部是一个线性层（放置在池化输出之上），例如用于 GLUE 任务。
@add_start_docstrings(
    """
    Nezha Model transformer with a sequence classification/regression head on top (a linear layer on top of the pooled
    output) e.g. for GLUE tasks.
    """,
    NEZHA_START_DOCSTRING,
)
class NezhaForSequenceClassification(NezhaPreTrainedModel):
    def __init__(self, config):
        # 初始化函数，接受一个配置对象并调用父类的初始化方法
        super().__init__(config)
        # 设置类别数量
        self.num_labels = config.num_labels
        # 保存配置对象
        self.config = config

        # 创建 Nezha 模型实例
        self.nezha = NezhaModel(config)
        # 设置分类器的 dropout 概率，如果未提供则使用隐藏层 dropout 概率
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        # 定义一个 dropout 层
        self.dropout = nn.Dropout(classifier_dropout)
        # 定义一个线性层，用于分类任务，输入尺寸是隐藏层尺寸，输出尺寸是类别数量
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # 初始化权重并应用最终处理
        self.post_init()

    @add_start_docstrings_to_model_forward(NEZHA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=SequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
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
        # 根据函数定义，返回值类型可以是包含Tensor的元组或者SequenceClassifierOutput对象
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用NeZha模型进行前向传播
        outputs = self.nezha(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 提取经过池化层后的输出
        pooled_output = outputs[1]

        # 对池化后的输出应用dropout
        pooled_output = self.dropout(pooled_output)

        # 使用分类器模型进行分类预测
        logits = self.classifier(pooled_output)

        # 初始化损失值
        loss = None

        # 如果给定了标签，计算损失函数
        if labels is not None:
            # 根据配置动态确定问题类型
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            # 根据问题类型选择对应的损失函数
            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    # 对于单标签回归问题，计算均方误差损失
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    # 对于多标签回归问题，同样计算均方误差损失
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                # 对于单标签分类问题，使用交叉熵损失函数
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                # 对于多标签分类问题，使用带Logits的二元交叉熵损失函数
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        # 如果不需要返回字典格式的结果，则返回元组形式的输出
        if not return_dict:
            output = (logits,) + outputs[2:]  # 包含分类预测和其他输出状态
            return ((loss,) + output) if loss is not None else output

        # 如果需要返回字典格式的结果，则返回SequenceClassifierOutput对象
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 为 Nezha 模型添加一个用于多选分类任务的头部（在池化输出的基础上添加一个线性层和 softmax 函数），例如用于 RocStories/SWAG 任务
@add_start_docstrings(
    """
    Nezha Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
    softmax) e.g. for RocStories/SWAG tasks.
    """,
    NEZHA_START_DOCSTRING,
)
# 定义 NezhaForMultipleChoice 类，继承自 NezhaPreTrainedModel
class NezhaForMultipleChoice(NezhaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        # 初始化 Nezha 模型
        self.nezha = NezhaModel(config)
        # 根据配置获取分类器的 dropout 概率，若未设置则使用隐藏层的 dropout 概率
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        # 使用 dropout 层
        self.dropout = nn.Dropout(classifier_dropout)
        # 分类器线性层，输入维度为隐藏层大小，输出维度为1
        self.classifier = nn.Linear(config.hidden_size, 1)

        # 初始化权重并进行最终处理
        self.post_init()

    # 添加输入说明文档到模型前向传播方法
    @add_start_docstrings_to_model_forward(NEZHA_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length"))
    # 添加代码示例文档到模型前向传播方法
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=MultipleChoiceModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    # 定义前向传播方法
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], MultipleChoiceModelOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the multiple choice classification loss. Indices should be in `[0, ...,
            num_choices-1]` where `num_choices` is the size of the second dimension of the input tensors. (See
            `input_ids` above)
        """
        # 如果 return_dict 为 None，则使用配置中的 use_return_dict 设置
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 计算 num_choices，根据 input_ids 或 inputs_embeds 的第二维大小确定
        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]
        # 将 input_ids 重新视图为二维形状，如果 input_ids 不为 None
        input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        # 将 attention_mask 重新视图为二维形状，如果 attention_mask 不为 None
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        # 将 token_type_ids 重新视图为二维形状，如果 token_type_ids 不为 None
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        # 将 inputs_embeds 重新视图为三维形状，如果 inputs_embeds 不为 None
        inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )

        # 调用 self.nezha 方法，传入各种参数，获取输出
        outputs = self.nezha(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 从输出中获取 pooled_output
        pooled_output = outputs[1]
        print(pooled_output.shape)
        # 对 pooled_output 进行 dropout
        pooled_output = self.dropout(pooled_output)
        # 将 dropout 后的 pooled_output 输入分类器，获取 logits
        logits = self.classifier(pooled_output)
        print(logits.shape)
        print(num_choices)
        # 将 logits 重新视图为二维形状，形状为 (batch_size, num_choices)
        reshaped_logits = logits.view(-1, num_choices)

        # 初始化 loss 为 None
        loss = None
        # 如果 labels 不为 None，则计算交叉熵损失
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)

        # 如果 return_dict 为 False，则输出包含 reshaped_logits 和额外输出的元组
        if not return_dict:
            output = (reshaped_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 如果 return_dict 为 True，则返回 MultipleChoiceModelOutput 对象
        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
@add_start_docstrings(
    """
    Nezha Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
    Named-Entity-Recognition (NER) tasks.
    """,
    NEZHA_START_DOCSTRING,
)
class NezhaForTokenClassification(NezhaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels  # 从配置中获取标签数量

        self.nezha = NezhaModel(config, add_pooling_layer=False)  # 初始化 Nezha 模型，不添加池化层
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)  # 定义一个 Dropout 层，用于分类器
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)  # 定义一个线性层作为分类器，输入维度是隐藏层大小，输出维度是标签数量

        # 初始化权重并进行最终处理
        self.post_init()

    @add_start_docstrings_to_model_forward(NEZHA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TokenClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict  # 如果未指定返回字典，使用配置中的默认设置

        outputs = self.nezha(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )  # 将输入传递给 Nezha 模型并获取输出

        sequence_output = outputs[0]  # 获取模型输出的序列输出

        sequence_output = self.dropout(sequence_output)  # 应用 Dropout 到序列输出上
        logits = self.classifier(sequence_output)  # 通过分类器线性层获取 logits

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))  # 计算标签分类损失

        if not return_dict:
            output = (logits,) + outputs[2:]  # 构建输出元组，包括 logits 和可能的其他输出
            return ((loss,) + output) if loss is not None else output  # 如果有损失，包括损失在内并返回，否则只返回输出

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )  # 返回 TokenClassifierOutput 对象，包含损失、logits、隐藏状态和注意力权重
"""
Nezha Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
"""
# 使用 Nezha 模型，在其顶部添加一个用于抽取式问答任务（如 SQuAD）的跨度分类头部（在隐藏状态输出之上的线性层，用于计算“起始位置对数”和“结束位置对数”）。

@add_start_docstrings(NEZHA_START_DOCSTRING)
# 添加起始文档字符串，继承自 NEZHA_START_DOCSTRING 预定义的文档字符串内容
class NezhaForQuestionAnswering(NezhaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        # 初始化 Nezha 模型，不包含池化层
        self.nezha = NezhaModel(config, add_pooling_layer=False)
        # QA 输出层，线性变换的输出大小为配置中的隐藏大小和标签数目
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        # 初始化权重并应用最终处理
        self.post_init()

    @add_start_docstrings_to_model_forward(NEZHA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=QuestionAnsweringModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    # 添加开始文档字符串到模型前向函数中，格式化 NEZHA_INPUTS_DOCSTRING 包含 "batch_size, sequence_length"
    # 添加代码示例的文档字符串，包括检查点、输出类型和配置类
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        start_positions: Optional[torch.Tensor] = None,
        end_positions: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
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
        # 根据需要决定是否返回字典形式的输出
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 使用 NEZHA 模型进行前向传播
        outputs = self.nezha(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取序列输出
        sequence_output = outputs[0]

        # 将序列输出传递给 QA 输出层得到 logits
        logits = self.qa_outputs(sequence_output)
        
        # 将 logits 沿着最后一个维度分割为 start_logits 和 end_logits
        start_logits, end_logits = logits.split(1, dim=-1)
        
        # 去除不必要的维度并确保连续性
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # 如果在多 GPU 环境下，可能需要扩展维度
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            
            # 忽略超出模型输入范围的 start/end 位置
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            # 定义交叉熵损失函数，忽略指定索引处的预测
            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        # 如果不要求返回字典形式的输出，则返回元组
        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        # 返回包含 loss、start_logits、end_logits 等内容的 QuestionAnsweringModelOutput 对象
        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
```