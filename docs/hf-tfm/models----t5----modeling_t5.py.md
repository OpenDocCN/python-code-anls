# `.\models\t5\modeling_t5.py`

```py
# coding=utf-8
# Copyright 2018 Mesh TensorFlow authors, T5 Authors and HuggingFace Inc. team.
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
""" PyTorch T5 model."""


import copy
import math
import os
import warnings
from typing import List, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from ...activations import ACT2FN
from ...modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
    Seq2SeqQuestionAnsweringModelOutput,
    Seq2SeqSequenceClassifierOutput,
    TokenClassifierOutput,
)
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import ALL_LAYERNORM_LAYERS, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
    DUMMY_INPUTS,
    DUMMY_MASK,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_torch_fx_proxy,
    logging,
    replace_return_docstrings,
)
from ...utils.model_parallel_utils import assert_device_map, get_device_map
from .configuration_t5 import T5Config


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "T5Config"
_CHECKPOINT_FOR_DOC = "google-t5/t5-small"

####################################################
# This dict contains ids and associated url
# for the pretrained weights provided with the models
####################################################
T5_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "google-t5/t5-small",
    "google-t5/t5-base",
    "google-t5/t5-large",
    "google-t5/t5-3b",
    "google-t5/t5-11b",
    # See all T5 models at https://huggingface.co/models?filter=t5
]


####################################################
# This is a conversion method from TF 1.0 to PyTorch
# More details: https://medium.com/huggingface/from-tensorflow-to-pytorch-265f40ef2a28
####################################################
def load_tf_weights_in_t5(model, config, tf_checkpoint_path):
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
    tf_path = os.path.abspath(tf_checkpoint_path)
    logger.info(f"Converting TensorFlow checkpoint from {tf_path}")
    # 从 TensorFlow 模型加载权重

    # 获取 TensorFlow 模型中所有变量的名称和形状
    init_vars = tf.train.list_variables(tf_path)

    # 初始化空列表，用于存储变量名称
    names = []

    # 初始化空字典，用于存储 TensorFlow 权重数组
    tf_weights = {}

    # 遍历 TensorFlow 模型中的每个变量名和形状
    for name, shape in init_vars:
        # 记录日志，显示当前加载的 TensorFlow 权重的名称和形状
        logger.info(f"Loading TF weight {name} with shape {shape}")

        # 使用 TensorFlow API 加载指定名称的变量数据
        array = tf.train.load_variable(tf_path, name)

        # 将当前变量名称添加到名称列表中
        names.append(name)

        # 将加载的 TensorFlow 权重数据存储到字典中，以变量名称作为键
        tf_weights[name] = array

    # 记录日志，显示未复制到 PyTorch 模型的 TensorFlow 权重的名称列表
    logger.info(f"Weights not copied to PyTorch model: {', '.join(tf_weights.keys())}.")

    # 返回 PyTorch 模型对象
    return model
####################################################
# PyTorch Models are constructed by sub-classing
# - torch.nn.Module for the layers and
# - PreTrainedModel for the models (it-self a sub-class of nn.Module)
####################################################

# 定义了一个原始字符串常量，用于并行处理和取消并行处理模型时的文档字符串说明
PARALLELIZE_DOCSTRING = r"""
    This is an experimental feature and is a subject to change at a moment's notice.

    Uses a device map to distribute attention modules of the model across several devices. If no device map is given,
    it will evenly distribute blocks across all devices.

    Args:
        device_map (`Dict[int, list]`, optional, defaults to None):
            A dictionary that maps attention modules to devices. Note that the embedding module and LMHead are always
            automatically mapped to the first device (for esoteric reasons). That means that the first device should
            have fewer attention modules mapped to it than other devices. For reference, the t5 models have the
            following number of attention modules:

                - google-t5/t5-small: 6
                - google-t5/t5-base: 12
                - google-t5/t5-large: 24
                - google-t5/t5-3b: 24
                - google-t5/t5-11b: 24

    Example:

    ```
    # Here is an example of a device map on a machine with 4 GPUs using google-t5/t5-3b, which has a total of 24 attention modules:
    model = T5ForConditionalGeneration.from_pretrained("google-t5/t5-3b")
    device_map = {
        0: [0, 1, 2],
        1: [3, 4, 5, 6, 7, 8, 9],
        2: [10, 11, 12, 13, 14, 15, 16],
        3: [17, 18, 19, 20, 21, 22, 23],
    }
    model.parallelize(device_map)
    ```
"""

# 定义了一个原始字符串常量，用于取消模型并行处理时的文档字符串说明
DEPARALLELIZE_DOCSTRING = r"""
    Moves the model to cpu from a model parallel state.

    Example:

    ```
    # On a 4 GPU machine with google-t5/t5-3b:
    model = T5ForConditionalGeneration.from_pretrained("google-t5/t5-3b")
    device_map = {
        0: [0, 1, 2],
        1: [3, 4, 5, 6, 7, 8, 9],
        2: [10, 11, 12, 13, 14, 15, 16],
        3: [17, 18, 19, 20, 21, 22, 23],
    }
    model.parallelize(device_map)  # Splits the model across several devices
    model.deparallelize()  # Put the model back on cpu and cleans memory by calling torch.cuda.empty_cache()
    ```
"""


class T5LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Construct a layernorm module in the T5 style. No bias and no subtraction of mean.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))  # 初始化权重参数为1，用于层归一化
        self.variance_epsilon = eps  # 初始化方差 epsilon 参数
    def forward(self, hidden_states):
        # 计算隐藏状态的方差，转换为 float32 类型，然后沿着最后一个维度计算均值
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        
        # 使用 rsqrt 函数计算标准差的倒数，对隐藏状态进行 layer normalization
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        # 如果权重的数据类型是半精度浮点数（float16 或 bfloat16），则将隐藏状态转换为相同的数据类型
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        # 返回经过权重调整后的隐藏状态
        return self.weight * hidden_states
try:
    # 尝试导入来自apex.normalization的FusedRMSNorm模块
    from apex.normalization import FusedRMSNorm

    # 将FusedRMSNorm赋值给T5LayerNorm，并禁止flake8检查
    T5LayerNorm = FusedRMSNorm  # noqa

    # 打印信息日志，表明发现了apex.normalization.FusedRMSNorm，将使用它代替T5LayerNorm
    logger.info("Discovered apex.normalization.FusedRMSNorm - will use it instead of T5LayerNorm")
except ImportError:
    # 如果导入失败，则使用普通的T5LayerNorm
    pass
except Exception:
    # 如果导入过程中出现任何异常，则记录警告日志
    logger.warning("discovered apex but it failed to load, falling back to T5LayerNorm")
    pass

# 将T5LayerNorm添加到ALL_LAYERNORM_LAYERS列表中
ALL_LAYERNORM_LAYERS.append(T5LayerNorm)


class T5DenseActDense(nn.Module):
    def __init__(self, config: T5Config):
        super().__init__()
        # 初始化权重为config.d_model到config.d_ff的线性层，没有偏置
        self.wi = nn.Linear(config.d_model, config.d_ff, bias=False)
        # 初始化权重为config.d_ff到config.d_model的线性层，没有偏置
        self.wo = nn.Linear(config.d_ff, config.d_model, bias=False)
        # 根据config.dropout_rate初始化Dropout层
        self.dropout = nn.Dropout(config.dropout_rate)
        # 根据配置选择激活函数，存储在self.act中
        self.act = ACT2FN[config.dense_act_fn]

    def forward(self, hidden_states):
        # 输入hidden_states经过self.wi线性层
        hidden_states = self.wi(hidden_states)
        # 使用self.act激活函数处理hidden_states
        hidden_states = self.act(hidden_states)
        # 对hidden_states应用Dropout
        hidden_states = self.dropout(hidden_states)
        # 如果self.wo.weight是Tensor类型，并且hidden_states的dtype不等于self.wo.weight的dtype，并且self.wo.weight的dtype不是torch.int8
        if (
            isinstance(self.wo.weight, torch.Tensor)
            and hidden_states.dtype != self.wo.weight.dtype
            and self.wo.weight.dtype != torch.int8
        ):
            # 将hidden_states转换到self.wo.weight的dtype
            hidden_states = hidden_states.to(self.wo.weight.dtype)
        # 输入hidden_states经过self.wo线性层
        hidden_states = self.wo(hidden_states)
        # 返回处理后的hidden_states
        return hidden_states


class T5DenseGatedActDense(nn.Module):
    def __init__(self, config: T5Config):
        super().__init__()
        # 初始化两个权重为config.d_model到config.d_ff的线性层，没有偏置
        self.wi_0 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wi_1 = nn.Linear(config.d_model, config.d_ff, bias=False)
        # 初始化权重为config.d_ff到config.d_model的线性层，没有偏置
        self.wo = nn.Linear(config.d_ff, config.d_model, bias=False)
        # 根据config.dropout_rate初始化Dropout层
        self.dropout = nn.Dropout(config.dropout_rate)
        # 根据配置选择激活函数，存储在self.act中
        self.act = ACT2FN[config.dense_act_fn]

    def forward(self, hidden_states):
        # 输入hidden_states经过self.wi_0线性层后使用self.act激活函数处理
        hidden_gelu = self.act(self.wi_0(hidden_states))
        # 输入hidden_states经过self.wi_1线性层
        hidden_linear = self.wi_1(hidden_states)
        # 将hidden_gelu和hidden_linear相乘得到hidden_states
        hidden_states = hidden_gelu * hidden_linear
        # 对hidden_states应用Dropout
        hidden_states = self.dropout(hidden_states)

        # 为了使8位量化在google/flan-t5-xxl中起作用，self.wo被保持为float32。
        # 参见https://github.com/huggingface/transformers/issues/20287
        # 同时确保权重不是`int8`，以防用户强制将`_keep_in_fp32_modules`设为`None`
        if (
            isinstance(self.wo.weight, torch.Tensor)
            and hidden_states.dtype != self.wo.weight.dtype
            and self.wo.weight.dtype != torch.int8
        ):
            # 将hidden_states转换到self.wo.weight的dtype
            hidden_states = hidden_states.to(self.wo.weight.dtype)

        # 输入hidden_states经过self.wo线性层
        hidden_states = self.wo(hidden_states)
        # 返回处理后的hidden_states
        return hidden_states


class T5LayerFF(nn.Module):
    def __init__(self, config: T5Config):
        super().__init__()
        # 如果config.is_gated_act为True，则使用T5DenseGatedActDense，否则使用T5DenseActDense
        if config.is_gated_act:
            self.DenseReluDense = T5DenseGatedActDense(config)
        else:
            self.DenseReluDense = T5DenseActDense(config)

        # 初始化Layer Norm层，参数为config.d_model和config.layer_norm_epsilon
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        # 根据config.dropout_rate初始化Dropout层
        self.dropout = nn.Dropout(config.dropout_rate)
    #`
    # 定义一个前向传播函数，接受隐藏状态作为输入
    def forward(self, hidden_states):
        # 对输入的隐藏状态进行层归一化处理
        forwarded_states = self.layer_norm(hidden_states)
        # 将归一化后的隐藏状态输入到一个全连接层+ReLU激活函数+全连接层的组合中
        forwarded_states = self.DenseReluDense(forwarded_states)
        # 对第二个全连接层的输出进行dropout处理，并将结果加回到原始的隐藏状态中
        hidden_states = hidden_states + self.dropout(forwarded_states)
        # 返回更新后的隐藏状态作为输出
        return hidden_states
# 定义一个名为 T5Attention 的类，继承自 nn.Module，表示它是一个PyTorch模型组件
class T5Attention(nn.Module):
    # 构造方法，初始化注意力机制的各种参数和组件
    def __init__(self, config: T5Config, has_relative_attention_bias=False):
        super().__init__()
        # 是否为解码器
        self.is_decoder = config.is_decoder
        # 是否包含相对注意力偏置
        self.has_relative_attention_bias = has_relative_attention_bias
        # 相对注意力的桶数
        self.relative_attention_num_buckets = config.relative_attention_num_buckets
        # 相对注意力的最大距离
        self.relative_attention_max_distance = config.relative_attention_max_distance
        # 模型的维度
        self.d_model = config.d_model
        # 键值投影的维度
        self.key_value_proj_dim = config.d_kv
        # 注意力头的数量
        self.n_heads = config.num_heads
        # Dropout率
        self.dropout = config.dropout_rate
        # 内部维度，即注意力头数乘以键值投影维度
        self.inner_dim = self.n_heads * self.key_value_proj_dim

        # 初始化注意力计算的线性层，用于查询、键、值和输出
        self.q = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.k = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.v = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.o = nn.Linear(self.inner_dim, self.d_model, bias=False)

        # 如果有相对注意力偏置，则初始化相对注意力偏置的嵌入层
        if self.has_relative_attention_bias:
            self.relative_attention_bias = nn.Embedding(self.relative_attention_num_buckets, self.n_heads)
        
        # 初始化一个集合，用于存储被剪枝的注意力头的索引
        self.pruned_heads = set()
        # 是否启用梯度检查点
        self.gradient_checkpointing = False

    # 方法：剪枝指定的注意力头
    def prune_heads(self, heads):
        # 如果没有要剪枝的头，则直接返回
        if len(heads) == 0:
            return
        # 找到可剪枝的注意力头和对应的索引
        heads, index = find_pruneable_heads_and_indices(
            heads, self.n_heads, self.key_value_proj_dim, self.pruned_heads
        )
        # 剪枝线性层
        self.q = prune_linear_layer(self.q, index)
        self.k = prune_linear_layer(self.k, index)
        self.v = prune_linear_layer(self.v, index)
        self.o = prune_linear_layer(self.o, index, dim=1)
        # 更新超参数
        self.n_heads = self.n_heads - len(heads)
        self.inner_dim = self.key_value_proj_dim * self.n_heads
        # 将剪枝的头添加到集合中
        self.pruned_heads = self.pruned_heads.union(heads)

    # 静态方法，用于其它辅助功能或算法的实现，这里没有具体实现给出
    @staticmethod
    def _relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        """
        Adapted from Mesh Tensorflow:
        https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593

        Translate relative position to a bucket number for relative attention. The relative position is defined as
        memory_position - query_position, i.e. the distance in tokens from the attending position to the attended-to
        position. If bidirectional=False, then positive relative positions are invalid. We use smaller buckets for
        small absolute relative_position and larger buckets for larger absolute relative_positions. All relative
        positions >=max_distance map to the same bucket. All relative positions <=-max_distance map to the same bucket.
        This should allow for more graceful generalization to longer sequences than the model has been trained on

        Args:
            relative_position: an int32 Tensor - the difference in positions between memory and query
            bidirectional: a boolean - whether the attention is bidirectional or not
            num_buckets: an integer - number of buckets to categorize relative positions into
            max_distance: an integer - maximum distance for categorizing relative positions

        Returns:
            a Tensor with the same shape as relative_position, containing int32 values in the range [0, num_buckets)
        """
        # Initialize relative_buckets to 0
        relative_buckets = 0
        
        # Adjust num_buckets if bidirectional is True
        if bidirectional:
            num_buckets //= 2
            # Calculate relative_buckets based on whether relative_position > 0
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
            # Take absolute value of relative_position
            relative_position = torch.abs(relative_position)
        else:
            # Set relative_position to negative of its minimum value or 0
            relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
        
        # now relative_position is in the range [0, inf)
        
        # Determine if relative_position is small (less than max_exact)
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact
        
        # Calculate relative_position_if_large for larger relative positions
        relative_position_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)
        
        # Clamp relative_position_if_large to num_buckets - 1
        relative_position_if_large = torch.min(
            relative_position_if_large, torch.full_like(relative_position_if_large, num_buckets - 1)
        )
        
        # Determine final relative_buckets using conditional assignment based on is_small
        relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)
        
        return relative_buckets
    def compute_bias(self, query_length, key_length, device=None):
        """Compute binned relative position bias"""
        # 如果未提供设备，则使用相对注意力偏置权重的设备
        if device is None:
            device = self.relative_attention_bias.weight.device
        # 创建一个张量，表示查询序列的位置索引，形状为 (query_length, 1)
        context_position = torch.arange(query_length, dtype=torch.long, device=device)[:, None]
        # 创建一个张量，表示键序列的位置索引，形状为 (1, key_length)
        memory_position = torch.arange(key_length, dtype=torch.long, device=device)[None, :]
        # 计算相对位置，形状为 (query_length, key_length)
        relative_position = memory_position - context_position
        # 将相对位置映射到相对位置桶中，形状仍为 (query_length, key_length)
        relative_position_bucket = self._relative_position_bucket(
            relative_position,
            bidirectional=(not self.is_decoder),
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance,
        )
        # 使用相对位置桶来获取相对注意力偏置值，形状为 (query_length, key_length, num_heads)
        values = self.relative_attention_bias(relative_position_bucket)
        # 对值张量进行维度置换和扩展，形状为 (1, num_heads, query_length, key_length)
        values = values.permute([2, 0, 1]).unsqueeze(0)
        # 返回计算得到的相对位置偏置值
        return values

    def forward(
        self,
        hidden_states,
        mask=None,
        key_value_states=None,
        position_bias=None,
        past_key_value=None,
        layer_head_mask=None,
        query_length=None,
        use_cache=False,
        output_attentions=False,
# 定义 T5 模型的自注意力层
class T5LayerSelfAttention(nn.Module):
    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__()
        # 初始化自注意力机制
        self.SelfAttention = T5Attention(config, has_relative_attention_bias=has_relative_attention_bias)
        # 初始化层归一化
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        # 初始化 dropout
        self.dropout = nn.Dropout(config.dropout_rate)

    # 前向传播函数，接受一些参数和张量 hidden_states
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        output_attentions=False,
    ):
        # 对输入的 hidden_states 进行层归一化处理
        normed_hidden_states = self.layer_norm(hidden_states)
        # 将归一化后的 hidden_states 输入到 SelfAttention 层
        attention_output = self.SelfAttention(
            normed_hidden_states,
            mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        # 将原始 hidden_states 和经过 dropout 处理后的注意力输出相加
        hidden_states = hidden_states + self.dropout(attention_output[0])
        # 输出包括更新后的 hidden_states 和额外的注意力信息（如果有的话）
        outputs = (hidden_states,) + attention_output[1:]  # 如果需要，添加注意力信息
        return outputs


# 定义 T5 模型的跨注意力层
class T5LayerCrossAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 初始化编码-解码注意力机制
        self.EncDecAttention = T5Attention(config, has_relative_attention_bias=False)
        # 初始化层归一化
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        # 初始化 dropout
        self.dropout = nn.Dropout(config.dropout_rate)

    # 前向传播函数，接受一些参数和张量 hidden_states，key_value_states
    def forward(
        self,
        hidden_states,
        key_value_states,
        attention_mask=None,
        position_bias=None,
        layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        query_length=None,
        output_attentions=False,
    ):
        # 对输入的 hidden_states 进行层归一化处理
        normed_hidden_states = self.layer_norm(hidden_states)
        # 将归一化后的 hidden_states 输入到 EncDecAttention 层
        attention_output = self.EncDecAttention(
            normed_hidden_states,
            mask=attention_mask,
            key_value_states=key_value_states,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            query_length=query_length,
            output_attentions=output_attentions,
        )
        # 将原始 hidden_states 和经过 dropout 处理后的注意力输出相加
        layer_output = hidden_states + self.dropout(attention_output[0])
        # 输出包括更新后的 hidden_states 和额外的注意力信息（如果有的话）
        outputs = (layer_output,) + attention_output[1:]  # 如果需要，添加注意力信息
        return outputs


# 定义 T5 模型的块
class T5Block(nn.Module):
    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__()
        # 标记该块是否为解码器块
        self.is_decoder = config.is_decoder
        # 初始化层列表
        self.layer = nn.ModuleList()
        # 添加自注意力层到层列表
        self.layer.append(T5LayerSelfAttention(config, has_relative_attention_bias=has_relative_attention_bias))
        # 如果是解码器，添加编码-解码注意力层到层列表
        if self.is_decoder:
            self.layer.append(T5LayerCrossAttention(config))

        # 添加前馈神经网络层到层列表
        self.layer.append(T5LayerFF(config))
    # 定义 Transformer 模型的前向传播函数，接受多个参数：
    # - hidden_states: 输入的隐藏状态
    # - attention_mask: 可选参数，用于屏蔽不需要关注的位置
    # - position_bias: 可选参数，用于位置偏置
    # - encoder_hidden_states: 可选参数，编码器的隐藏状态
    # - encoder_attention_mask: 可选参数，编码器的注意力屏蔽
    # - encoder_decoder_position_bias: 可选参数，编码器到解码器的位置偏置
    # - layer_head_mask: 可选参数，用于层头的屏蔽
    # - cross_attn_layer_head_mask: 可选参数，用于交叉注意力的层头屏蔽
    # - past_key_value: 可选参数，过去的键值对，用于生成缓存
    # - use_cache: 是否使用缓存，默认为 False
    # - output_attentions: 是否输出注意力权重，默认为 False
    # - return_dict: 是否返回结果字典，默认为 True
class T5ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config: T5Config):
        super().__init__()
        # 定义一个全连接层，输入维度为config.d_model，输出维度为config.d_model
        self.dense = nn.Linear(config.d_model, config.d_model)
        # 定义一个Dropout层，概率为config.classifier_dropout
        self.dropout = nn.Dropout(p=config.classifier_dropout)
        # 定义一个全连接层，输入维度为config.d_model，输出维度为config.num_labels
        self.out_proj = nn.Linear(config.d_model, config.num_labels)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 对输入的hidden_states进行dropout处理
        hidden_states = self.dropout(hidden_states)
        # 通过全连接层self.dense进行线性变换
        hidden_states = self.dense(hidden_states)
        # 对线性变换的结果进行tanh激活函数处理
        hidden_states = torch.tanh(hidden_states)
        # 再次对处理后的hidden_states进行dropout处理
        hidden_states = self.dropout(hidden_states)
        # 通过全连接层self.out_proj进行线性变换，得到最终的输出
        hidden_states = self.out_proj(hidden_states)
        return hidden_states


class T5PreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = T5Config
    load_tf_weights = load_tf_weights_in_t5
    base_model_prefix = "transformer"
    is_parallelizable = True
    supports_gradient_checkpointing = True
    _no_split_modules = ["T5Block"]
    _keep_in_fp32_modules = ["wo"]

    @property
    def dummy_inputs(self):
        # 创建一个包含虚拟输入数据的字典
        input_ids = torch.tensor(DUMMY_INPUTS)
        input_mask = torch.tensor(DUMMY_MASK)
        dummy_inputs = {
            "decoder_input_ids": input_ids,
            "input_ids": input_ids,
            "decoder_attention_mask": input_mask,
        }
        return dummy_inputs

    def _shift_right(self, input_ids):
        # 获取decoder起始标记的ID和pad标记的ID
        decoder_start_token_id = self.config.decoder_start_token_id
        pad_token_id = self.config.pad_token_id

        if decoder_start_token_id is None:
            raise ValueError(
                "self.model.config.decoder_start_token_id has to be defined. In T5 it is usually set to the pad_token_id. "
                "See T5 docs for more information."
            )

        # 将输入向右移动一位
        if is_torch_fx_proxy(input_ids):
            # 对于代理对象，不支持原生的项目赋值操作
            shifted_input_ids = torch.full(input_ids.shape[:-1] + (1,), decoder_start_token_id)
            shifted_input_ids = torch.cat([shifted_input_ids, input_ids[..., :-1]], dim=-1)
        else:
            # 创建一个与input_ids形状相同的全零张量
            shifted_input_ids = input_ids.new_zeros(input_ids.shape)
            # 将input_ids的内容向右移动一位，并将decoder起始标记填充到第一位
            shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
            shifted_input_ids[..., 0] = decoder_start_token_id

        if pad_token_id is None:
            raise ValueError("self.model.config.pad_token_id has to be defined.")
        # 将标签中可能存在的-100值替换为pad_token_id
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

        return shifted_input_ids


class T5Stack(T5PreTrainedModel):
    pass
    # 使用给定的配置和嵌入令牌（如果提供），初始化一个 T5Stack 对象
    def __init__(self, config, embed_tokens=None):
        # 调用父类的初始化方法
        super().__init__(config)

        # 将嵌入令牌保存到对象属性中
        self.embed_tokens = embed_tokens
        # 检查配置中是否设置了解码器标志，并保存到对象属性中
        self.is_decoder = config.is_decoder

        # 创建包含多个 T5Block 的模块列表（每个 T5Block 对象表示 T5 模型的一个层）
        self.block = nn.ModuleList(
            [T5Block(config, has_relative_attention_bias=bool(i == 0)) for i in range(config.num_layers)]
        )
        # 初始化最终层归一化对象，用于处理模型的输出
        self.final_layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        # 创建一个 dropout 层，用于随机失活
        self.dropout = nn.Dropout(config.dropout_rate)

        # 初始化权重并进行最终的处理
        self.post_init()

        # Model parallel （模型并行设置）
        self.model_parallel = False  # 默认情况下不使用模型并行
        self.device_map = None  # 设备映射初始化为 None
        self.gradient_checkpointing = False  # 梯度检查点设置为 False

    @add_start_docstrings(PARALLELIZE_DOCSTRING)
    # 并行化方法，用于将模型放置到多个设备上
    def parallelize(self, device_map=None):
        # 发出警告，表明此方法即将被移除
        warnings.warn(
            "`T5Stack.parallelize` is deprecated and will be removed in v5 of Transformers, you should load your model"
            " with `device_map='balanced'` in the call to `from_pretrained`. You can also provide your own"
            " `device_map` but it needs to be a dictionary module_name to device, so for instance {'block.0': 0,"
            " 'block.1': 1, ...}",
            FutureWarning,
        )

        # 检查设备映射的有效性
        self.device_map = (
            get_device_map(len(self.block), range(torch.cuda.device_count())) if device_map is None else device_map
        )
        # 确保设备映射与层的数量匹配
        assert_device_map(self.device_map, len(self.block))

        # 标记模型已启用模型并行
        self.model_parallel = True

        # 确定第一个设备和最后一个设备的名称
        self.first_device = "cpu" if "cpu" in self.device_map.keys() else "cuda:" + str(min(self.device_map.keys()))
        self.last_device = "cuda:" + str(max(self.device_map.keys()))

        # 将每个层移动到对应的设备
        for k, v in self.device_map.items():
            for layer in v:
                cuda_device = "cuda:" + str(k)
                self.block[layer] = self.block[layer].to(cuda_device)

        # 将嵌入令牌移到第一个设备
        self.embed_tokens = self.embed_tokens.to(self.first_device)
        # 将最终层归一化移到最后一个设备
        self.final_layer_norm = self.final_layer_norm.to(self.last_device)

    @add_start_docstrings(DEPARALLELIZE_DOCSTRING)
    # 反并行化方法，用于将模型从多个设备恢复到单设备
    def deparallelize(self):
        # 发出警告，表明此方法即将被移除
        warnings.warn(
            "Like `parallelize`, `deparallelize` is deprecated and will be removed in v5 of Transformers.",
            FutureWarning,
        )

        # 将模型并行标志设置为 False
        self.model_parallel = False
        # 设备映射置为 None
        self.device_map = None
        # 将第一个设备和最后一个设备都设置为 "cpu"
        self.first_device = "cpu"
        self.last_device = "cpu"

        # 将每个层移回到 CPU
        for i in range(len(self.block)):
            self.block[i] = self.block[i].to("cpu")

        # 将嵌入令牌和最终层归一化层移回到 CPU
        self.embed_tokens = self.embed_tokens.to("cpu")
        self.final_layer_norm = self.final_layer_norm.to("cpu")

        # 清空 CUDA 缓存
        torch.cuda.empty_cache()

    # 获取输入嵌入层对象
    def get_input_embeddings(self):
        return self.embed_tokens

    # 设置输入嵌入层对象
    def set_input_embeddings(self, new_embeddings):
        self.embed_tokens = new_embeddings
    # 定义模型的前向传播方法，接受多个输入参数，用于处理输入序列的各种信息
    def forward(
        self,
        input_ids=None,  # 输入的 token IDs，用于表示输入序列
        attention_mask=None,  # 注意力遮罩，指定哪些位置需要参与注意力计算
        encoder_hidden_states=None,  # 编码器隐藏状态，用于某些模型的特定任务
        encoder_attention_mask=None,  # 编码器注意力遮罩，指定哪些编码器隐藏状态需要注意
        inputs_embeds=None,  # 输入的嵌入表示，代替 input_ids 使用
        head_mask=None,  # 头部遮罩，用于指定哪些注意力头部需要被屏蔽
        cross_attn_head_mask=None,  # 跨注意力头部遮罩，类似于 head_mask，但用于跨注意力机制
        past_key_values=None,  # 过去的键-值对，用于支持增量式生成的情况
        use_cache=None,  # 是否使用缓存，用于存储中间计算结果以加速推理
        output_attentions=None,  # 是否输出注意力权重
        output_hidden_states=None,  # 是否输出所有隐藏状态
        return_dict=None,  # 是否返回一个字典作为输出
# T5 模型的文档字符串，用于说明该模型的提出背景和特性
T5_START_DOCSTRING = r"""

    The T5 model was proposed in [Exploring the Limits of Transfer Learning with a Unified Text-to-Text
    Transformer](https://arxiv.org/abs/1910.10683) by Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan
    Narang, Michael Matena, Yanqi Zhou, Wei Li, Peter J. Liu. It's an encoder decoder transformer pre-trained in a
    text-to-text denoising generative setting.

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`T5Config`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

# T5 模型的输入文档字符串，暂未提供具体内容，保留空字符串
T5_INPUTS_DOCSTRING = r"""
"""

# T5 编码器输入的文档字符串，暂未提供具体内容，保留空字符串
T5_ENCODER_INPUTS_DOCSTRING = r"""
"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            # 输入序列的标记索引，在词汇表中。对于 T5 模型，相对位置嵌入使得可以在输入的左右两侧进行填充。
            # 可以使用 `AutoTokenizer` 获取这些索引。参见 `PreTrainedTokenizer.encode` 和 `PreTrainedTokenizer.__call__`。
            # 如何准备 `input_ids` 进行预训练，请查看[T5 Training](./t5#training)。

        attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            # 遮罩，用于避免对填充的标记索引执行注意力操作。遮罩的取值范围为 `[0, 1]`：
            # - 1 表示对应的标记**未被遮罩**，
            # - 0 表示对应的标记**被遮罩**。
            # [什么是注意力遮罩？](../glossary#attention-mask)

        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            # 用于将自注意力模块中的部分头部置零的遮罩。遮罩的取值范围为 `[0, 1]`：
            # - 1 表示该头部**未被遮罩**，
            # - 0 表示该头部**被遮罩**。

        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            # 可选参数，可以直接传递嵌入表示而不是 `input_ids`。如果希望对如何将 `input_ids` 索引转换为相关联向量有更多控制权，那么这很有用。
            # 这对于超越模型内部嵌入查找矩阵有更多控制的情况很有用。

        output_attentions (`bool`, *optional*):
            # 是否返回所有注意力层的注意力张量。查看返回的张量中的 `attentions` 以获取更多细节。

        output_hidden_states (`bool`, *optional*):
            # 是否返回所有层的隐藏状态。查看返回的张量中的 `hidden_states` 以获取更多细节。

        return_dict (`bool`, *optional*):
            # 是否返回一个 `~utils.ModelOutput` 而不是一个普通的元组。
"""

# 警告消息，用于将来的警告：head_mask 参数已分成两个输入参数 - head_mask 和 decoder_head_mask
__HEAD_MASK_WARNING_MSG = """
The input argument `head_mask` was split into two arguments `head_mask` and `decoder_head_mask`. Currently,
`decoder_head_mask` is set to copy `head_mask`, but this feature is deprecated and will be removed in future versions.
If you do not want to use any `decoder_head_mask` now, please set `decoder_head_mask = torch.ones(num_layers,
num_heads)`.
"""


@add_start_docstrings(
    "The bare T5 Model transformer outputting raw hidden-states without any specific head on top.",
    T5_START_DOCSTRING,
)
class T5Model(T5PreTrainedModel):
    # 在模型加载时忽略的意外键列表
    _keys_to_ignore_on_load_unexpected = [
        "decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.weight",
    ]
    # 共享权重的键列表
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight"]

    def __init__(self, config: T5Config):
        super().__init__(config)
        # 创建共享的嵌入层，用于处理词汇大小和模型维度的embedding
        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        # 复制配置并设置编码器
        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5Stack(encoder_config, self.shared)

        # 复制配置并设置解码器
        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = T5Stack(decoder_config, self.shared)

        # 初始化权重并应用最终处理
        self.post_init()

        # 模型并行计算标志
        self.model_parallel = False
        self.device_map = None

    @add_start_docstrings(PARALLELIZE_DOCSTRING)
    def parallelize(self, device_map=None):
        # 发出警告，提醒方法即将被弃用
        warnings.warn(
            "`T5Model.parallelize` is deprecated and will be removed in v5 of Transformers, you should load your model"
            " with `device_map='balanced'` in the call to `from_pretrained`. You can also provide your own"
            " `device_map` but it needs to be a dictionary module_name to device, so for instance {'encoder.block.0':"
            " 0, 'encoder.block.1': 1, ...}",
            FutureWarning,
        )
        # 根据传入的设备映射或自动生成平衡设备映射
        self.device_map = (
            get_device_map(len(self.encoder.block), range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        # 检查设备映射的有效性
        assert_device_map(self.device_map, len(self.encoder.block))
        # 对编码器和解码器进行并行化处理
        self.encoder.parallelize(self.device_map)
        self.decoder.parallelize(self.device_map)
        self.model_parallel = True

    @add_start_docstrings(DEPARALLELIZE_DOCSTRING)
    # 发出一个关于函数过时的警告，提示使用者此功能将在 Transformers 的 v5 版本中移除
    def deparallelize(self):
        warnings.warn(
            "Like `parallelize`, `deparallelize` is deprecated and will be removed in v5 of Transformers.",
            FutureWarning,
        )
        # 调用编码器对象的 deparallelize 方法，取消并行化设置
        self.encoder.deparallelize()
        # 调用解码器对象的 deparallelize 方法，取消并行化设置
        self.decoder.deparallelize()
        # 将编码器移动到 CPU 上执行
        self.encoder = self.encoder.to("cpu")
        # 将解码器移动到 CPU 上执行
        self.decoder = self.decoder.to("cpu")
        # 禁用模型并行化设置
        self.model_parallel = False
        # 将设备映射设置为空
        self.device_map = None
        # 清空 CUDA 缓存
        torch.cuda.empty_cache()

    # 获取输入嵌入层对象的方法
    def get_input_embeddings(self):
        return self.shared

    # 设置输入嵌入层对象的方法
    def set_input_embeddings(self, new_embeddings):
        # 更新共享的嵌入层对象
        self.shared = new_embeddings
        # 更新编码器的输入嵌入层对象
        self.encoder.set_input_embeddings(new_embeddings)
        # 更新解码器的输入嵌入层对象
        self.decoder.set_input_embeddings(new_embeddings)

    # 内部方法，用于绑定权重（如果配置允许）
    def _tie_weights(self):
        if self.config.tie_word_embeddings:
            # 绑定或克隆编码器的词嵌入权重与共享的嵌入层对象
            self._tie_or_clone_weights(self.encoder.embed_tokens, self.shared)
            # 绑定或克隆解码器的词嵌入权重与共享的嵌入层对象
            self._tie_or_clone_weights(self.decoder.embed_tokens, self.shared)

    # 获取编码器对象的方法
    def get_encoder(self):
        return self.encoder

    # 获取解码器对象的方法
    def get_decoder(self):
        return self.decoder

    # 内部方法，用于剪枝模型的注意力头
    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        # 遍历需要剪枝的层及其对应的头信息
        for layer, heads in heads_to_prune.items():
            # 对编码器的某一层的注意力模型进行头剪枝操作
            self.encoder.layer[layer].attention.prune_heads(heads)

    # 此函数用于模型的前向传播
    @add_start_docstrings_to_model_forward(T5_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        decoder_inputs_embeds: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
@add_start_docstrings("""T5 Model with a `language modeling` head on top.""", T5_START_DOCSTRING)
class T5ForConditionalGeneration(T5PreTrainedModel):
    # 忽略加载时不期望的键列表
    _keys_to_ignore_on_load_unexpected = [
        "decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.weight",
    ]
    # 被绑定权重的键列表
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight", "lm_head.weight"]

    def __init__(self, config: T5Config):
        super().__init__(config)
        # 模型维度
        self.model_dim = config.d_model

        # 共享的嵌入层，用于输入词汇表大小和模型维度
        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        # 复制编码器配置，并设置为非解码器模式
        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        # 创建编码器实例
        self.encoder = T5Stack(encoder_config, self.shared)

        # 复制解码器配置，并设置为解码器模式
        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        # 创建解码器实例
        self.decoder = T5Stack(decoder_config, self.shared)

        # 线性层，用于语言模型的输出，输入维度为模型维度，输出维度为词汇表大小，无偏置
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # 初始化权重并应用最终处理
        self.post_init()

        # 模型并行化
        self.model_parallel = False
        self.device_map = None

    @add_start_docstrings(PARALLELIZE_DOCSTRING)
    def parallelize(self, device_map=None):
        # 发出警告，此方法即将弃用
        warnings.warn(
            "`T5ForConditionalGeneration.parallelize` is deprecated and will be removed in v5 of Transformers, you"
            " should load your model with `device_map='balanced'` in the call to `from_pretrained`. You can also"
            " provide your own `device_map` but it needs to be a dictionary module_name to device, so for instance"
            " {'encoder.block.0': 0, 'encoder.block.1': 1, ...}",
            FutureWarning,
        )
        # 获取设备映射，如果未提供则使用均衡映射
        self.device_map = (
            get_device_map(len(self.encoder.block), range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        # 验证设备映射的有效性
        assert_device_map(self.device_map, len(self.encoder.block))
        # 将编码器并行化
        self.encoder.parallelize(self.device_map)
        # 将解码器并行化
        self.decoder.parallelize(self.device_map)
        # 将语言模型头移到解码器的第一个设备上
        self.lm_head = self.lm_head.to(self.decoder.first_device)
        # 设置模型为模型并行化状态
        self.model_parallel = True

    @add_start_docstrings(DEPARALLELIZE_DOCSTRING)
    def deparallelize(self):
        # 发出警告，此方法即将弃用
        warnings.warn(
            "Like `parallelize`, `deparallelize` is deprecated and will be removed in v5 of Transformers.",
            FutureWarning,
        )
        # 取消编码器的并行化
        self.encoder.deparallelize()
        # 取消解码器的并行化
        self.decoder.deparallelize()
        # 将编码器移到CPU
        self.encoder = self.encoder.to("cpu")
        # 将解码器移到CPU
        self.decoder = self.decoder.to("cpu")
        # 将语言模型头移到CPU
        self.lm_head = self.lm_head.to("cpu")
        # 设置模型为非模型并行化状态
        self.model_parallel = False
        self.device_map = None
        # 清空CUDA缓存
        torch.cuda.empty_cache()

    def get_input_embeddings(self):
        # 返回共享的嵌入层
        return self.shared
    # 设置模型的输入词嵌入
    def set_input_embeddings(self, new_embeddings):
        # 将新的词嵌入赋给共享的嵌入层
        self.shared = new_embeddings
        # 设置编码器的输入词嵌入
        self.encoder.set_input_embeddings(new_embeddings)
        # 设置解码器的输入词嵌入
        self.decoder.set_input_embeddings(new_embeddings)

    # 绑定权重（或克隆）以确保编码器和解码器共享相同的词嵌入
    def _tie_weights(self):
        if self.config.tie_word_embeddings:
            # 绑定或克隆编码器的嵌入层与共享的嵌入层
            self._tie_or_clone_weights(self.encoder.embed_tokens, self.shared)
            # 绑定或克隆解码器的嵌入层与共享的嵌入层
            self._tie_or_clone_weights(self.decoder.embed_tokens, self.shared)

    # 设置模型的输出词嵌入
    def set_output_embeddings(self, new_embeddings):
        # 设置语言模型头部的新词嵌入
        self.lm_head = new_embeddings

    # 返回模型的输出词嵌入
    def get_output_embeddings(self):
        return self.lm_head

    # 返回编码器实例
    def get_encoder(self):
        return self.encoder

    # 返回解码器实例
    def get_decoder(self):
        return self.decoder

    # 模型前向传播方法，用于执行T5模型的输入到输出的转换
    @add_start_docstrings_to_model_forward(T5_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        # 省略T5模型的前向传播逻辑，由装饰器管理

    # 准备用于生成的输入，这里主要用于生成文本
    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        decoder_attention_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs,
    ):
        # 省略准备生成文本输入的逻辑，可以传递各种参数给模型
    ):
        # 如果使用了过去的键值（past_key_values），则裁剪decoder_input_ids
        if past_key_values is not None:
            # 获取过去键值的长度
            past_length = past_key_values[0][0].shape[2]

            # 一些生成方法已经只传递最后一个输入ID
            if input_ids.shape[1] > past_length:
                # 如果输入的input_ids长度大于过去的长度，裁剪掉前面的部分
                remove_prefix_length = past_length
            else:
                # 默认旧的行为：保留最后一个ID
                remove_prefix_length = input_ids.shape[1] - 1

            input_ids = input_ids[:, remove_prefix_length:]

        return {
            "decoder_input_ids": input_ids,  # 返回裁剪后的decoder_input_ids
            "past_key_values": past_key_values,  # 返回过去的键值
            "encoder_outputs": encoder_outputs,  # 返回编码器的输出
            "attention_mask": attention_mask,  # 返回注意力掩码
            "head_mask": head_mask,  # 返回头部掩码
            "decoder_head_mask": decoder_head_mask,  # 返回解码器头部掩码
            "decoder_attention_mask": decoder_attention_mask,  # 返回解码器的注意力掩码
            "cross_attn_head_mask": cross_attn_head_mask,  # 返回交叉注意力头部掩码
            "use_cache": use_cache,  # 返回是否使用缓存的标志
        }

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        # 根据标签准备解码器的输入ids，将标签向右移动一位
        return self._shift_right(labels)

    def _reorder_cache(self, past_key_values, beam_idx):
        # 如果解码器的过去状态未包含在输出中
        # 快速解码被禁用，无需重新排序
        if past_key_values is None:
            logger.warning("You might want to consider setting `use_cache=True` to speed up decoding")
            return past_key_values

        reordered_decoder_past = ()
        for layer_past_states in past_key_values:
            # 从层过去状态中获取正确的批次索引
            # past的批次维度在第二个位置
            reordered_layer_past_states = ()
            for layer_past_state in layer_past_states:
                # 需要为每个四个键/值状态设置正确的past
                reordered_layer_past_states = reordered_layer_past_states + (
                    layer_past_state.index_select(0, beam_idx.to(layer_past_state.device)),
                )

            if reordered_layer_past_states[0].shape != layer_past_states[0].shape:
                raise ValueError(
                    f"reordered_layer_past_states[0] shape {reordered_layer_past_states[0].shape} and layer_past_states[0] shape {layer_past_states[0].shape} mismatched"
                )
            if len(reordered_layer_past_states) != len(layer_past_states):
                raise ValueError(
                    f"length of reordered_layer_past_states {len(reordered_layer_past_states)} and length of layer_past_states {len(layer_past_states)} mismatched"
                )

            reordered_decoder_past = reordered_decoder_past + (reordered_layer_past_states,)
        return reordered_decoder_past
# 添加模型的文档字符串，描述了这个类是一个 T5 编码器模型，输出编码器的原始隐藏状态而不带任何特定的头部
@add_start_docstrings(
    "The bare T5 Model transformer outputting encoder's raw hidden-states without any specific head on top.",
    T5_START_DOCSTRING,
)
class T5EncoderModel(T5PreTrainedModel):
    # 在加载模型时需要保持权重一致的键列表
    _tied_weights_keys = ["encoder.embed_tokens.weight"]
    # 加载时需要忽略的意外键列表，这里排除了包含"decoder"的键
    _keys_to_ignore_on_load_unexpected = [r"decoder"]

    def __init__(self, config: T5Config):
        super().__init__(config)
        # 共享的嵌入层，根据配置创建一个词汇表大小为config.vocab_size，维度为config.d_model的嵌入层
        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        # 复制配置以配置编码器，并设置一些属性
        encoder_config = copy.deepcopy(config)
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        # 创建 T5 堆栈编码器，使用共享的嵌入层
        self.encoder = T5Stack(encoder_config, self.shared)

        # 初始化权重并应用最终处理
        self.post_init()

        # 模型并行处理相关属性初始化
        self.model_parallel = False
        self.device_map = None

    # 添加模型并行化的文档字符串，警告此方法在后续版本中将被移除
    @add_start_docstrings(PARALLELIZE_DOCSTRING)
    def parallelize(self, device_map=None):
        warnings.warn(
            "`T5EncoderModel.parallelize` is deprecated and will be removed in v5 of Transformers, you should load"
            " your model with `device_map='balanced'` in the call to `from_pretrained`. You can also provide your own"
            " `device_map` but it needs to be a dictionary module_name to device, so for instance {'block.0': 0,"
            " 'block.1': 1, ...}",
            FutureWarning,
        )
        # 获取设备映射，如果未提供设备映射，则默认使用均衡的设备映射
        self.device_map = (
            get_device_map(len(self.encoder.block), range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        # 检查设备映射的合法性
        assert_device_map(self.device_map, len(self.encoder.block))
        # 调用编码器的并行化方法，设置模型并行为真
        self.encoder.parallelize(self.device_map)
        self.model_parallel = True

    # 添加反并行化的文档字符串，警告此方法在后续版本中将被移除
    @add_start_docstrings(DEPARALLELIZE_DOCSTRING)
    def deparallelize(self):
        warnings.warn(
            "Like `parallelize`, `deparallelize` is deprecated and will be removed in v5 of Transformers.",
            FutureWarning,
        )
        # 调用编码器的反并行化方法，将编码器转移到 CPU 上，并设置模型并行为假
        self.encoder.deparallelize()
        self.encoder = self.encoder.to("cpu")
        self.model_parallel = False
        self.device_map = None
        # 清空 CUDA 缓存
        torch.cuda.empty_cache()

    # 返回共享的嵌入层
    def get_input_embeddings(self):
        return self.shared

    # 设置新的输入嵌入层，并更新编码器的输入嵌入层
    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)

    # 绑定权重的私有方法，如果配置中要求绑定词嵌入权重，则绑定编码器的嵌入词汇表和共享的嵌入层
    def _tie_weights(self):
        if self.config.tie_word_embeddings:
            self._tie_or_clone_weights(self.encoder.embed_tokens, self.shared)

    # 返回编码器对象
    def get_encoder(self):
        return self.encoder

    # 剪枝模型中的注意力头，heads_to_prune 是一个字典，表示需要在每层剪枝的注意力头的列表
    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            # 调用编码器堆栈中每层的自注意力模块的剪枝头方法
            self.encoder.block[layer].layer[0].SelfAttention.prune_heads(heads)

    # 添加 T5 编码器模型前向传播的文档字符串
    @add_start_docstrings_to_model_forward(T5_ENCODER_INPUTS_DOCSTRING)
    # 将函数返回值的文档字符串中的输出类型替换为BaseModelOutput，配置类为_CONFIG_FOR_DOC
    @replace_return_docstrings(output_type=BaseModelOutput, config_class=_CONFIG_FOR_DOC)
    # 前向传播函数，接受多个输入参数，并返回Union类型的torch.FloatTensor或BaseModelOutput
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,  # 输入的token IDs，类型为可选的长整型张量
        attention_mask: Optional[torch.FloatTensor] = None,  # 注意力遮罩张量，类型为可选的浮点数张量
        head_mask: Optional[torch.FloatTensor] = None,  # 头部遮罩张量，类型为可选的浮点数张量
        inputs_embeds: Optional[torch.FloatTensor] = None,  # 嵌入输入张量，类型为可选的浮点数张量
        output_attentions: Optional[bool] = None,  # 是否输出注意力张量，类型为可选的布尔值
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态张量，类型为可选的布尔值
        return_dict: Optional[bool] = None,  # 是否返回字典格式的输出，类型为可选的布尔值
    ) -> Union[Tuple[torch.FloatTensor], BaseModelOutput]:
        r"""
        返回值：
            如果return_dict不为None，则返回return_dict；否则返回self.config.use_return_dict。

        示例：

        ```
        >>> from transformers import AutoTokenizer, T5EncoderModel

        >>> tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small")
        >>> model = T5EncoderModel.from_pretrained("google-t5/t5-small")
        >>> input_ids = tokenizer(
        ...     "Studies have been shown that owning a dog is good for you", return_tensors="pt"
        ... ).input_ids  # Batch size 1
        >>> outputs = model(input_ids=input_ids)
        >>> last_hidden_states = outputs.last_hidden_state
        ```
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 使用编码器模型处理输入，并获取编码器的输出结果
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 返回编码器的输出结果
        return encoder_outputs
"""
T5 model with a sequence classification/head on top (a linear layer on top of the pooled output) e.g. for GLUE
tasks.
"""
# 定义 T5 序列分类模型，其顶部有一个线性层（位于汇聚输出之上），用于例如 GLUE 任务
@add_start_docstrings(
    """
    T5 Encoder Model with a token classification head on top (a linear layer on top of the hidden-states output)
    e.g. for Named-Entity-Recognition (NER) tasks.
    """,
    T5_START_DOCSTRING,
)
class T5ForTokenClassification(T5PreTrainedModel):
    # 指定权重共享的关键键列表
    _tied_weights_keys = ["transformer.encoder.embed_tokens.weight"]

    def __init__(self, config: T5Config):
        super().__init__(config)
        # 初始化 T5 配置
        self.num_labels = config.num_labels

        # 创建 T5 编码器模型
        self.transformer = T5EncoderModel(config)
        # 添加一个丢弃层
        self.dropout = nn.Dropout(config.classifier_dropout)
        # 添加一个线性分类器层
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # 初始化权重并执行最终处理
        self.post_init()

    @add_start_docstrings_to_model_forward(T5_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TokenClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,  # 输入的token IDs张量，可选
        attention_mask: Optional[torch.Tensor] = None,  # 注意力掩码张量，可选
        head_mask: Optional[torch.Tensor] = None,  # 头部掩码张量，可选
        inputs_embeds: Optional[torch.Tensor] = None,  # 输入的嵌入张量，可选
        labels: Optional[torch.Tensor] = None,  # 用于计算标记分类损失的标签张量，可选
        output_attentions: Optional[bool] = None,  # 是否输出注意力权重，可选
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态，可选
        return_dict: Optional[bool] = None,  # 是否返回字典格式的输出，可选
    ) -> Union[Tuple[torch.Tensor], TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        Returns:
        """
        # 确定是否使用配置中的返回字典设置
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 将输入传递给transformer模型，并获取输出
        outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 从输出中获取隐藏状态并应用dropout
        hidden_states = outputs[0]
        hidden_states = self.dropout(hidden_states)
        
        # 将隐藏状态传递给分类器，获取预测的逻辑回归输出
        logits = self.classifier(hidden_states)

        # 如果提供了标签，则计算损失
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        # 根据return_dict标志，返回不同的输出格式
        if not return_dict:
            output = (logits, outputs[2:-1])  # 仅在不返回字典时输出隐藏状态
            return ((loss,) + output) if loss is not None else output

        # 返回TokenClassifierOutput对象，包含损失、预测的逻辑回归、隐藏状态和注意力权重
        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 使用特定的文档字符串初始化模型，该模型包含一个用于提取问答任务（如SQuAD）的跨度分类头部（在隐藏状态输出之上的线性层，用于计算“跨度起始logits”和“跨度结束logits”）。
@add_start_docstrings(
    """
    T5 Model with a span classification head on top for extractive question-answering tasks like SQuAD (linear layers
    on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    T5_START_DOCSTRING,
)
class T5ForQuestionAnswering(T5PreTrainedModel):
    # 在加载时忽略的键列表，遇到不期待的键
    _keys_to_ignore_on_load_unexpected = ["decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.weight"]
    # 权重共享的键列表
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight"]

    # 初始化方法，接受一个T5Config对象作为参数
    def __init__(self, config: T5Config):
        super().__init__(config)
        # 模型维度设为配置文件中的d_model值
        self.model_dim = config.d_model

        # 共享的词嵌入层，使用配置文件中的vocab_size和d_model创建
        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        # 复制配置以创建编码器
        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        # 使用T5Stack模块和共享的词嵌入层创建编码器
        self.encoder = T5Stack(encoder_config, self.shared)

        # 复制配置以创建解码器
        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        # 使用T5Stack模块和共享的词嵌入层创建解码器
        self.decoder = T5Stack(decoder_config, self.shared)

        # 输出的标签数量为配置文件中的num_labels
        self.num_labels = config.num_labels
        # 线性层，将隐藏大小映射到标签数量
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        # 初始化权重并进行最终处理
        self.post_init()

        # 模型并行设置为False
        self.model_parallel = False

    # 返回共享的词嵌入层对象
    def get_input_embeddings(self):
        return self.shared

    # 设置新的输入词嵌入层
    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        # 同时更新编码器和解码器的输入词嵌入层
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    # 绑定权重，如果配置中设置了tie_word_embeddings为True
    def _tie_weights(self):
        if self.config.tie_word_embeddings:
            # 绑定或克隆权重以使编码器和解码器共享词嵌入层的权重
            self._tie_or_clone_weights(self.encoder.embed_tokens, self.shared)
            self._tie_or_clone_weights(self.decoder.embed_tokens, self.shared)

    # 返回编码器对象
    def get_encoder(self):
        return self.encoder

    # 返回解码器对象
    def get_decoder(self):
        return self.decoder

    # 用于模型前向传播方法的装饰器，添加了T5输入文档字符串
    @add_start_docstrings_to_model_forward(T5_INPUTS_DOCSTRING)
    # 替换返回值文档字符串为Seq2SeqQuestionAnsweringModelOutput类型，使用配置类_CONFIG_FOR_DOC
    @replace_return_docstrings(output_type=Seq2SeqQuestionAnsweringModelOutput, config_class=_CONFIG_FOR_DOC)
    # 定义神经网络模型的前向传播函数，接受多个可选的输入参数，类型为 PyTorch 的张量或布尔值
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,  # 输入文本的词编号张量，可选
        attention_mask: Optional[torch.FloatTensor] = None,  # 输入文本的注意力掩码张量，可选
        decoder_input_ids: Optional[torch.LongTensor] = None,  # 解码器输入的词编号张量，可选
        decoder_attention_mask: Optional[torch.BoolTensor] = None,  # 解码器的注意力掩码张量，可选
        head_mask: Optional[torch.FloatTensor] = None,  # 多头注意力的头部掩码张量，可选
        decoder_head_mask: Optional[torch.FloatTensor] = None,  # 解码器多头注意力的头部掩码张量，可选
        cross_attn_head_mask: Optional[torch.Tensor] = None,  # 交叉注意力的头部掩码张量，可选
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,  # 编码器的输出，包含多个张量的元组，可选
        start_positions: Optional[torch.LongTensor] = None,  # 起始位置张量，用于损失计算，可选
        end_positions: Optional[torch.LongTensor] = None,  # 结束位置张量，用于损失计算，可选
        inputs_embeds: Optional[torch.FloatTensor] = None,  # 输入的嵌入向量张量，可选
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,  # 解码器输入的嵌入向量张量，可选
        use_cache: Optional[bool] = None,  # 是否使用缓存，用于解码器的 Transformer 模型，可选
        output_attentions: Optional[bool] = None,  # 是否输出注意力权重，可选
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态，可选
        return_dict: Optional[bool] = None,  # 是否以字典形式返回输出，可选
```