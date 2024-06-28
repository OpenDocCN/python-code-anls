# `.\models\mt5\modeling_mt5.py`

```py
# coding=utf-8
# Copyright 2020 Mesh TensorFlow authors, T5 Authors and HuggingFace Inc. team.
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
""" PyTorch mT5 model."""

# 导入所需的模块和类
import copy
import math
import os
import warnings
from typing import List, Optional, Tuple, Union

# 导入PyTorch库
import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

# 导入各种输出类和模型基类
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
# 导入模型工具函数和常用函数
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
    DUMMY_INPUTS,
    DUMMY_MASK,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_torch_fx_proxy,
    logging,
    replace_return_docstrings,
)
# 导入模型并行处理相关的函数
from ...utils.model_parallel_utils import assert_device_map, get_device_map
# 导入mT5模型配置
from .configuration_mt5 import MT5Config

# 获取日志记录器
logger = logging.get_logger(__name__)

# mT5预训练模型存档列表，包含预训练模型的标识和URL
MT5_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "google/mt5-small",
    "google/mt5-base",
    "google/mt5-large",
    "google/mt5-xl",
    "google/mt5-xxl",
    # 查看所有mT5模型：https://huggingface.co/models?filter=mt5
]

# 并行化文档字符串，描述了一个实验性功能，随时可能变更
PARALLELIZE_DOCSTRING = r"""
    This is an experimental feature and is a subject to change at a moment's notice.

    Uses a device map to distribute attention modules of the model across several devices. If no device map is given,
    it will evenly distribute blocks across all devices.
"""
    Args:
        device_map (`Dict[int, list]`, optional, defaults to None):
            A dictionary that maps attention modules to devices. Note that the embedding module and LMHead are always
            automatically mapped to the first device (for esoteric reasons). That means that the first device should
            have fewer attention modules mapped to it than other devices. For reference, the mt5 models have the
            following number of attention modules:

                - mt5-small: 6
                - mt5-base: 12
                - mt5-large: 24
                - mt5-xl: 24
                - mt5-xxl: 24

    Example:

    ```
    # Here is an example of a device map on a machine with 4 GPUs using mt5-xl, which has a total of 24 attention modules:
    model = MT5ForConditionalGeneration.from_pretrained("mt5-xl")
    创建一个 MT5 模型实例，使用预训练的 "mt5-xl" 模型
    device_map = {
        0: [0, 1, 2],
        将 attention 模块映射到四个 GPU 设备上的示例映射表
        1: [3, 4, 5, 6, 7, 8, 9],
        2: [10, 11, 12, 13, 14, 15, 16],
        3: [17, 18, 19, 20, 21, 22, 23],
    }
    使用给定的设备映射表将模型并行化处理
    model.parallelize(device_map)
    ```
"""
DEPARALLELIZE_DOCSTRING = r"""
    Moves the model to cpu from a model parallel state.

    Example:

    ```
    # On a 4 GPU machine with mt5-xl:
    model = MT5ForConditionalGeneration.from_pretrained("Mt5-xl")
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


# Copied from transformers.models.t5.modeling_t5.T5LayerNorm with T5->MT5
class MT5LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Construct a layernorm module in the MT5 style. No bias and no subtraction of mean.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))  # 初始化权重参数为全一张量
        self.variance_epsilon = eps  # 设置方差的 epsilon 值

    def forward(self, hidden_states):
        # MT5 uses a layer_norm which only scales and doesn't shift, which is also known as Root Mean
        # Square Layer Normalization https://arxiv.org/abs/1910.07467 thus varience is calculated
        # w/o mean and there is no bias. Additionally we want to make sure that the accumulation for
        # half-precision inputs is done in fp32

        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)  # 计算输入张量的方差
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)  # 根据方差进行 layer normalization

        # convert into half-precision if necessary
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)  # 如果权重数据类型为半精度，则将隐藏状态转换为相同精度

        return self.weight * hidden_states  # 返回经过权重调整后的隐藏状态


# Copied from transformers.models.t5.modeling_t5.T5DenseActDense with T5->MT5
class MT5DenseActDense(nn.Module):
    def __init__(self, config: MT5Config):
        super().__init__()
        self.wi = nn.Linear(config.d_model, config.d_ff, bias=False)  # 带有线性变换的全连接层，无偏置
        self.wo = nn.Linear(config.d_ff, config.d_model, bias=False)  # 带有线性变换的全连接层，无偏置
        self.dropout = nn.Dropout(config.dropout_rate)  # 随机丢弃层，使用指定的 dropout 率
        self.act = ACT2FN[config.dense_act_fn]  # 激活函数从配置中获取

    def forward(self, hidden_states):
        hidden_states = self.wi(hidden_states)  # 输入经过第一个线性层变换
        hidden_states = self.act(hidden_states)  # 应用激活函数
        hidden_states = self.dropout(hidden_states)  # 应用 dropout
        if (
            isinstance(self.wo.weight, torch.Tensor)
            and hidden_states.dtype != self.wo.weight.dtype
            and self.wo.weight.dtype != torch.int8
        ):
            hidden_states = hidden_states.to(self.wo.weight.dtype)  # 根据权重数据类型调整隐藏状态的数据类型
        hidden_states = self.wo(hidden_states)  # 输入经过第二个线性层变换
        return hidden_states


# Copied from transformers.models.t5.modeling_t5.T5DenseGatedActDense with T5->MT5
class MT5DenseGatedActDense(nn.Module):
    # 初始化方法，接受一个 MT5Config 对象作为参数
    def __init__(self, config: MT5Config):
        # 调用父类的初始化方法
        super().__init__()
        # 创建一个线性层，输入维度为 config.d_model，输出维度为 config.d_ff，无偏置
        self.wi_0 = nn.Linear(config.d_model, config.d_ff, bias=False)
        # 创建一个线性层，输入维度为 config.d_model，输出维度为 config.d_ff，无偏置
        self.wi_1 = nn.Linear(config.d_model, config.d_ff, bias=False)
        # 创建一个线性层，输入维度为 config.d_ff，输出维度为 config.d_model，无偏置
        self.wo = nn.Linear(config.d_ff, config.d_model, bias=False)
        # 创建一个 Dropout 层，使用给定的 dropout 率
        self.dropout = nn.Dropout(config.dropout_rate)
        # 根据配置中指定的激活函数名称，选择对应的激活函数
        self.act = ACT2FN[config.dense_act_fn]

    # 前向传播方法，接受输入 hidden_states
    def forward(self, hidden_states):
        # 将输入 hidden_states 经过激活函数 act 和线性层 wi_0 得到 hidden_gelu
        hidden_gelu = self.act(self.wi_0(hidden_states))
        # 将输入 hidden_states 经过线性层 wi_1 得到 hidden_linear
        hidden_linear = self.wi_1(hidden_states)
        # 将 hidden_gelu 和 hidden_linear 逐元素相乘得到 hidden_states
        hidden_states = hidden_gelu * hidden_linear
        # 对 hidden_states 应用 dropout 操作
        hidden_states = self.dropout(hidden_states)

        # 为了让 8 位量化适用于 google/flan-t5-xxl，self.wo 保持为 float32 类型。
        # 参考 https://github.com/huggingface/transformers/issues/20287
        # 同时确保权重不是 `int8` 类型，以防止用户强制设置 `_keep_in_fp32_modules` 为 `None`
        if (
            isinstance(self.wo.weight, torch.Tensor)
            and hidden_states.dtype != self.wo.weight.dtype
            and self.wo.weight.dtype != torch.int8
        ):
            # 将 hidden_states 转换为 self.wo.weight 的数据类型
            hidden_states = hidden_states.to(self.wo.weight.dtype)

        # 将 hidden_states 经过线性层 self.wo 得到输出 hidden_states
        hidden_states = self.wo(hidden_states)
        # 返回最终的 hidden_states 结果
        return hidden_states
# 从 transformers.models.t5.modeling_t5.T5LayerFF 复制并改为 T5->MT5
class MT5LayerFF(nn.Module):
    # 初始化函数，接受一个 MT5Config 对象作为参数
    def __init__(self, config: MT5Config):
        super().__init__()
        # 根据配置选择不同的 DenseReluDense 模块
        if config.is_gated_act:
            self.DenseReluDense = MT5DenseGatedActDense(config)
        else:
            self.DenseReluDense = MT5DenseActDense(config)

        # 初始化 LayerNorm 模块，设定 epsilon 值
        self.layer_norm = MT5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        # 初始化 Dropout 模块，设定 dropout 率
        self.dropout = nn.Dropout(config.dropout_rate)

    # 前向传播函数，接受隐藏状态作为输入，返回更新后的隐藏状态
    def forward(self, hidden_states):
        # 对隐藏状态进行 LayerNorm 处理
        forwarded_states = self.layer_norm(hidden_states)
        # 通过 DenseReluDense 模块处理规范化后的隐藏状态
        forwarded_states = self.DenseReluDense(forwarded_states)
        # 使用 Dropout 处理得到的前向传播状态，并与原始隐藏状态相加
        hidden_states = hidden_states + self.dropout(forwarded_states)
        # 返回更新后的隐藏状态
        return hidden_states


# 从 transformers.models.t5.modeling_t5.T5Attention 复制并改为 T5->MT5
class MT5Attention(nn.Module):
    # 初始化函数，接受一个 MT5Config 对象和是否包含相对注意力偏置的标志作为参数
    def __init__(self, config: MT5Config, has_relative_attention_bias=False):
        super().__init__()
        # 是否为解码器
        self.is_decoder = config.is_decoder
        # 是否包含相对注意力偏置
        self.has_relative_attention_bias = has_relative_attention_bias
        # 相对注意力偏置的桶数
        self.relative_attention_num_buckets = config.relative_attention_num_buckets
        # 相对注意力的最大距离
        self.relative_attention_max_distance = config.relative_attention_max_distance
        # 模型的隐藏状态维度
        self.d_model = config.d_model
        # 键值投影维度
        self.key_value_proj_dim = config.d_kv
        # 注意力头的数量
        self.n_heads = config.num_heads
        # Dropout 率
        self.dropout = config.dropout_rate
        # 内部维度，即头数乘以键值投影维度
        self.inner_dim = self.n_heads * self.key_value_proj_dim

        # 初始化查询、键、值和输出的线性变换层，无偏置
        self.q = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.k = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.v = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.o = nn.Linear(self.inner_dim, self.d_model, bias=False)

        # 如果有相对注意力偏置，初始化相对注意力偏置的嵌入层
        if self.has_relative_attention_bias:
            self.relative_attention_bias = nn.Embedding(self.relative_attention_num_buckets, self.n_heads)
        # 初始化被剪枝的注意力头集合和梯度检查点标志
        self.pruned_heads = set()
        self.gradient_checkpointing = False

    # 静态方法：剪枝注意力头
    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        # 找到可剪枝的注意力头和对应索引
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
        self.pruned_heads = self.pruned_heads.union(heads)
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
            relative_position: an int32 Tensor - 相对位置，表示从当前位置到关注位置的距离
            bidirectional: a boolean - 是否为双向注意力
            num_buckets: an integer - 桶的数量，用于将相对位置映射到桶编号
            max_distance: an integer - 最大距离，超过此距离的相对位置映射到同一个桶

        Returns:
            a Tensor with the same shape as relative_position, containing int32 values in the range [0, num_buckets)
            返回一个与 relative_position 形状相同的张量，包含范围在 [0, num_buckets) 内的整数值
        """
        relative_buckets = 0  # 初始化相对位置桶号为0

        # 如果是双向注意力，则将桶数减半，并根据 relative_position 的正负分别计算桶号偏移
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            # 如果是单向注意力，将 relative_position 转换为非正的数值
            relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))

        # 现在 relative_position 范围在 [0, inf)

        # 小于 max_exact 的相对位置使用线性增量的桶
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        # 大于 max_exact 的相对位置使用对数增量的桶，映射到 [max_exact, num_buckets-1] 范围内
        relative_position_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)
        relative_position_if_large = torch.min(
            relative_position_if_large, torch.full_like(relative_position_if_large, num_buckets - 1)
        )

        # 根据相对位置大小选择合适的桶号
        relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)

        return relative_buckets  # 返回计算得到的相对位置桶号张量
    def compute_bias(self, query_length, key_length, device=None):
        """Compute binned relative position bias"""
        # 如果未指定设备，则使用相对注意力偏置权重张量的设备
        if device is None:
            device = self.relative_attention_bias.weight.device
        # 创建表示上下文位置的张量，范围为[0, query_length-1]
        context_position = torch.arange(query_length, dtype=torch.long, device=device)[:, None]
        # 创建表示记忆位置的张量，范围为[0, key_length-1]
        memory_position = torch.arange(key_length, dtype=torch.long, device=device)[None, :]
        # 计算相对位置偏差，形状为(query_length, key_length)
        relative_position = memory_position - context_position
        # 将相对位置映射到桶中，返回形状为(query_length, key_length)的桶索引张量
        relative_position_bucket = self._relative_position_bucket(
            relative_position,
            bidirectional=(not self.is_decoder),
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance,
        )
        # 使用相对位置桶索引获取相对注意力偏置值，形状为(query_length, key_length, num_heads)
        values = self.relative_attention_bias(relative_position_bucket)
        # 调整张量维度顺序以匹配Transformer的注意力头结构，形状为(1, num_heads, query_length, key_length)
        values = values.permute([2, 0, 1]).unsqueeze(0)
        # 返回相对位置注意力偏置张量
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
# Copied from transformers.models.t5.modeling_t5.T5LayerSelfAttention with T5->MT5
class MT5LayerSelfAttention(nn.Module):
    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__()
        # 初始化自注意力层对象，使用MT5Attention进行自注意力计算
        self.SelfAttention = MT5Attention(config, has_relative_attention_bias=has_relative_attention_bias)
        # 初始化层归一化对象，用于规范化隐藏状态
        self.layer_norm = MT5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        # 初始化Dropout层，用于随机失活以防止过拟合
        self.dropout = nn.Dropout(config.dropout_rate)

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
        # 对输入的隐藏状态进行层归一化处理
        normed_hidden_states = self.layer_norm(hidden_states)
        # 使用SelfAttention对象计算自注意力，得到注意力输出
        attention_output = self.SelfAttention(
            normed_hidden_states,
            mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        # 将原始隐藏状态与注意力输出相加，并且应用Dropout
        hidden_states = hidden_states + self.dropout(attention_output[0])
        # 准备输出，如果需要返回注意力权重，则包含在输出中
        outputs = (hidden_states,) + attention_output[1:]  # add attentions if we output them
        return outputs


# Copied from transformers.models.t5.modeling_t5.T5LayerCrossAttention with T5->MT5
class MT5LayerCrossAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 初始化跨注意力层对象，使用MT5Attention进行编码-解码注意力计算
        self.EncDecAttention = MT5Attention(config, has_relative_attention_bias=False)
        # 初始化层归一化对象，用于规范化隐藏状态
        self.layer_norm = MT5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        # 初始化Dropout层，用于随机失活以防止过拟合
        self.dropout = nn.Dropout(config.dropout_rate)

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
        # 对输入的隐藏状态进行层归一化处理
        normed_hidden_states = self.layer_norm(hidden_states)
        # 使用EncDecAttention对象计算编码-解码注意力，得到注意力输出
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
        # 将原始隐藏状态与注意力输出相加，并且应用Dropout
        layer_output = hidden_states + self.dropout(attention_output[0])
        # 准备输出，如果需要返回注意力权重，则包含在输出中
        outputs = (layer_output,) + attention_output[1:]  # add attentions if we output them
        return outputs


# Copied from transformers.models.t5.modeling_t5.T5Block with T5->MT5
class MT5Block(nn.Module):
    # 初始化方法，用于创建一个 MT5Model 的实例
    def __init__(self, config, has_relative_attention_bias=False):
        # 调用父类的初始化方法
        super().__init__()
        # 根据配置设置是否为解码器
        self.is_decoder = config.is_decoder
        # 创建一个空的模块列表用于存储层的组件
        self.layer = nn.ModuleList()
        # 向模块列表中添加自注意力层，并传入配置和是否有相对注意力偏置的参数
        self.layer.append(MT5LayerSelfAttention(config, has_relative_attention_bias=has_relative_attention_bias))
        # 如果是解码器，再向模块列表中添加跨注意力层
        if self.is_decoder:
            self.layer.append(MT5LayerCrossAttention(config))

        # 向模块列表中添加前馈神经网络层
        self.layer.append(MT5LayerFF(config))

    # 前向传播方法，用于计算模型的输出
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        encoder_decoder_position_bias=None,
        layer_head_mask=None,
        cross_attn_layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        output_attentions=False,
        return_dict=True,
# 导入必要的模块和库
try:
    import re
    import numpy as np
    import tensorflow as tf
except ImportError:
    # 如果导入失败，记录错误信息并抛出异常
    logger.error(
        "Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see "
        "https://www.tensorflow.org/install/ for installation instructions."
    )
    raise

# 获取 TensorFlow checkpoint 文件的绝对路径
tf_path = os.path.abspath(tf_checkpoint_path)

# 打印日志，显示正在转换的 TensorFlow checkpoint 的路径
logger.info(f"Converting TensorFlow checkpoint from {tf_path}")

# 从 TensorFlow 模型中加载权重
init_vars = tf.train.list_variables(tf_path)
names = []
tf_weights = {}

# 遍历初始化变量列表，加载每个权重并存储到字典中
for name, shape in init_vars:
    logger.info(f"Loading TF weight {name} with shape {shape}")
    array = tf.train.load_variable(tf_path, name)
    names.append(name)
    tf_weights[name] = array

# 打印日志，显示未复制到 PyTorch 模型的权重名称
logger.info(f"Weights not copied to PyTorch model: {', '.join(tf_weights.keys())}.")

# 返回加载权重后的 PyTorch 模型
return model
    # 定义一个方法 `_shift_right`，接受一个输入的张量 `input_ids`
    def _shift_right(self, input_ids):
        # 从配置中获取解码器起始标记的 ID
        decoder_start_token_id = self.config.decoder_start_token_id
        # 从配置中获取填充标记的 ID
        pad_token_id = self.config.pad_token_id

        # 如果解码器起始标记的 ID 未定义，则抛出数值错误
        if decoder_start_token_id is None:
            raise ValueError(
                "self.model.config.decoder_start_token_id has to be defined. In MT5 it is usually set to the pad_token_id. "
                "See MT5 docs for more information."
            )

        # 将输入向右移动一位
        if is_torch_fx_proxy(input_ids):
            # 对于 Torch FX 代理，不支持原生的项目赋值
            # 创建一个全是解码器起始标记 ID 的张量，并连接到输入张量的末尾
            shifted_input_ids = torch.full(input_ids.shape[:-1] + (1,), decoder_start_token_id)
            shifted_input_ids = torch.cat([shifted_input_ids, input_ids[..., :-1]], dim=-1)
        else:
            # 使用 `new_zeros` 创建与输入张量相同形状的零张量
            shifted_input_ids = input_ids.new_zeros(input_ids.shape)
            # 将输入张量向右移动一位，并将解码器起始标记 ID 放在开头
            shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
            shifted_input_ids[..., 0] = decoder_start_token_id

        # 如果填充标记 ID 未定义，则抛出数值错误
        if pad_token_id is None:
            raise ValueError("self.model.config.pad_token_id has to be defined.")
        
        # 将标签中可能存在的 -100 值替换为 `pad_token_id`
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

        # 返回向右移动后的输入张量
        return shifted_input_ids
# Copied from transformers.models.t5.modeling_t5.T5Stack with T5->MT5
class MT5Stack(MT5PreTrainedModel):
    def __init__(self, config, embed_tokens=None):
        super().__init__(config)

        # 初始化 MT5Stack 类的实例
        self.embed_tokens = embed_tokens  # 嵌入令牌，用于输入的嵌入表示
        self.is_decoder = config.is_decoder  # 是否为解码器模式

        # 创建由多个 MT5Block 组成的模块列表，每个块具有相对注意力偏置（仅第一个块）
        self.block = nn.ModuleList(
            [MT5Block(config, has_relative_attention_bias=bool(i == 0)) for i in range(config.num_layers)]
        )
        self.final_layer_norm = MT5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)  # 最终的层归一化
        self.dropout = nn.Dropout(config.dropout_rate)  # 随机失活率

        # 初始化权重并应用最终处理
        self.post_init()

        # 模型并行化相关设置
        self.model_parallel = False  # 模型是否并行化
        self.device_map = None  # 设备映射表
        self.gradient_checkpointing = False  # 梯度检查点

    @add_start_docstrings(PARALLELIZE_DOCSTRING)
    def parallelize(self, device_map=None):
        warnings.warn(
            "`MT5Stack.parallelize` is deprecated and will be removed in v5 of Transformers, you should load your model"
            " with `device_map='balanced'` in the call to `from_pretrained`. You can also provide your own"
            " `device_map` but it needs to be a dictionary module_name to device, so for instance {'block.0': 0,"
            " 'block.1': 1, ...}",
            FutureWarning,
        )
        # 检查设备映射的有效性
        self.device_map = (
            get_device_map(len(self.block), range(torch.cuda.device_count())) if device_map is None else device_map
        )
        assert_device_map(self.device_map, len(self.block))  # 断言设备映射合法性
        self.model_parallel = True  # 开启模型并行化
        self.first_device = "cpu" if "cpu" in self.device_map.keys() else "cuda:" + str(min(self.device_map.keys()))
        self.last_device = "cuda:" + str(max(self.device_map.keys()))

        # 将每个块加载到指定设备
        for k, v in self.device_map.items():
            for layer in v:
                cuda_device = "cuda:" + str(k)
                self.block[layer] = self.block[layer].to(cuda_device)

        # 将嵌入令牌加载到第一个设备
        self.embed_tokens = self.embed_tokens.to(self.first_device)
        # 将最终层归一化加载到最后一个设备
        self.final_layer_norm = self.final_layer_norm.to(self.last_device)

    @add_start_docstrings(DEPARALLELIZE_DOCSTRING)
    def deparallelize(self):
        warnings.warn(
            "Like `parallelize`, `deparallelize` is deprecated and will be removed in v5 of Transformers.",
            FutureWarning,
        )
        self.model_parallel = False  # 关闭模型并行化
        self.device_map = None  # 清空设备映射表
        self.first_device = "cpu"  # 第一个设备设置为 CPU
        self.last_device = "cpu"  # 最后一个设备设置为 CPU

        # 将每个块加载到 CPU
        for i in range(len(self.block)):
            self.block[i] = self.block[i].to("cpu")
        self.embed_tokens = self.embed_tokens.to("cpu")  # 将嵌入令牌加载到 CPU
        self.final_layer_norm = self.final_layer_norm.to("cpu")  # 将最终层归一化加载到 CPU
        torch.cuda.empty_cache()  # 清空 CUDA 缓存

    def get_input_embeddings(self):
        return self.embed_tokens  # 返回嵌入令牌
    # 设置模型输入的嵌入向量
    def set_input_embeddings(self, new_embeddings):
        self.embed_tokens = new_embeddings

    # 定义模型的前向传播函数，接收多个参数用于推理或训练
    def forward(
        self,
        input_ids=None,  # 输入的token IDs
        attention_mask=None,  # 注意力掩码，指示模型在计算注意力时忽略某些token
        encoder_hidden_states=None,  # 编码器的隐藏状态，用于注意力机制
        encoder_attention_mask=None,  # 编码器的注意力掩码，指示编码器在计算注意力时忽略某些token
        inputs_embeds=None,  # 替代input_ids的嵌入向量输入
        head_mask=None,  # 头部掩码，用于遮蔽某些注意力头部的输出
        cross_attn_head_mask=None,  # 用于跨注意力的头部掩码
        past_key_values=None,  # 用于存储过去的键值对，以便支持自回归生成
        use_cache=None,  # 控制是否使用缓存
        output_attentions=None,  # 是否输出注意力权重
        output_hidden_states=None,  # 是否输出所有隐藏状态
        return_dict=None,  # 是否以字典形式返回输出
# MT5_START_DOCSTRING 是一个长字符串，用来描述 MT5 模型的相关信息和特性，包括其论文引用、模型结构等详细信息。
MT5_START_DOCSTRING = r"""

    The MT5 model was proposed in [Exploring the Limits of Transfer Learning with a Unified Text-to-Text
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
        config ([`MT5Config`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

# MT5_INPUTS_DOCSTRING 是一个空字符串，可能是为了后续补充描述输入的相关文档信息。
MT5_INPUTS_DOCSTRING = r"""
"""

# MT5_ENCODER_INPUTS_DOCSTRING 是另一个字符串，可能用来描述 MT5 模型编码器相关的输入信息。
MT5_ENCODER_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            # 输入序列标记在词汇表中的索引。MT5 模型具有相对位置嵌入，因此可以在右侧和左侧都进行填充。

            # 可以使用 [`AutoTokenizer`] 获取索引。详见 [`PreTrainedTokenizer.encode`] 和 [`PreTrainedTokenizer.__call__`]。

            # 想要了解如何为预训练准备 `input_ids`，请参考 [MT5 Training](./mt5#training)。

        attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            # 遮盖掩码，避免在填充标记索引上执行注意力操作。遮盖值在 `[0, 1]` 中选择：

            # - 1 表示**未遮盖**的标记，
            # - 0 表示**遮盖**的标记。

            # [什么是注意力遮盖？](../glossary#attention-mask)

        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            # 遮头掩码，用于使自注意力模块的特定头部失效。遮盖值在 `[0, 1]` 中选择：

            # - 1 表示头部**未遮盖**，
            # - 0 表示头部**遮盖**。

        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            # 可选地，您可以直接传递嵌入表示，而不是传递 `input_ids`。如果您希望更多控制如何将 `input_ids` 索引转换为关联向量，
            # 则这很有用，而不是使用模型的内部嵌入查找矩阵。

        output_attentions (`bool`, *optional*):
            # 是否返回所有注意力层的注意力张量。有关详细信息，请参见返回张量下的 `attentions`。

        output_hidden_states (`bool`, *optional*):
            # 是否返回所有层的隐藏状态。有关详细信息，请参见返回张量下的 `hidden_states`。

        return_dict (`bool`, *optional*):
            # 是否返回 [`~utils.ModelOutput`] 而不是普通元组。
# 用于将来的警告消息：head_mask 参数已分成两个参数 - head_mask 和 decoder_head_mask
__HEAD_MASK_WARNING_MSG = """
The input argument `head_mask` was split into two arguments `head_mask` and `decoder_head_mask`. Currently,
`decoder_head_mask` is set to copy `head_mask`, but this feature is deprecated and will be removed in future versions.
If you do not want to use any `decoder_head_mask` now, please set `decoder_head_mask = torch.ones(num_layers,
num_heads)`.
"""

# 定义 MT5Model 类，用于输出没有特定输出头的原始隐藏状态
@add_start_docstrings(
    "The bare MT5 Model transformer outputting raw hidden-states without any specific head on top.",
    MT5_START_DOCSTRING,
)
class MT5Model(MT5PreTrainedModel):
    r"""
    Examples:

    ```
    >>> from transformers import MT5Model, AutoTokenizer

    >>> model = MT5Model.from_pretrained("google/mt5-small")
    >>> tokenizer = AutoTokenizer.from_pretrained("google/mt5-small")
    >>> article = "UN Offizier sagt, dass weiter verhandelt werden muss in Syrien."
    >>> summary = "Weiter Verhandlung in Syrien."
    >>> inputs = tokenizer(article, return_tensors="pt")
    >>> labels = tokenizer(text_target=summary, return_tensors="pt")

    >>> outputs = model(input_ids=inputs["input_ids"], decoder_input_ids=labels["input_ids"])
    >>> hidden_states = outputs.last_hidden_state
    ```
    """

    # 模型类型为 "mt5"
    model_type = "mt5"
    # 配置类为 MT5Config
    config_class = MT5Config
    # 在加载时忽略的意外键列表
    _keys_to_ignore_on_load_unexpected = ["decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.weight"]
    # 共享权重键的列表
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight"]

    # 从 transformers.models.t5.modeling_t5.T5Model.__init__ 复制并修改为 MT5Model
    def __init__(self, config: MT5Config):
        super().__init__(config)
        # 创建一个共享的嵌入层
        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        # 复制并修改编码器配置
        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        # 创建编码器实例
        self.encoder = MT5Stack(encoder_config, self.shared)

        # 复制并修改解码器配置
        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        # 创建解码器实例
        self.decoder = MT5Stack(decoder_config, self.shared)

        # 初始化权重并应用最终处理
        self.post_init()

        # 模型并行设置
        self.model_parallel = False
        self.device_map = None

    @add_start_docstrings(PARALLELIZE_DOCSTRING)
    # 从 transformers.models.t5.modeling_t5.T5Model.parallelize 复制
    @add_start_docstrings(DEPARALLELIZE_DOCSTRING)
    # 使用给定的 DEPARALLELIZE_DOCSTRING 添加文档字符串，这是从 transformers.models.t5.modeling_t5.T5Model.deparallelize 复制过来的
    def deparallelize(self):
        # 发出警告，说明此方法即将在 Transformers 的 v5 版本中删除
        warnings.warn(
            "Like `parallelize`, `deparallelize` is deprecated and will be removed in v5 of Transformers.",
            FutureWarning,
        )
        # 调用编码器的 deparallelize 方法
        self.encoder.deparallelize()
        # 调用解码器的 deparallelize 方法
        self.decoder.deparallelize()
        # 将编码器移动到 CPU
        self.encoder = self.encoder.to("cpu")
        # 将解码器移动到 CPU
        self.decoder = self.decoder.to("cpu")
        # 将 model_parallel 标志设置为 False
        self.model_parallel = False
        # 将 device_map 设置为 None
        self.device_map = None
        # 清空 CUDA 缓存
        torch.cuda.empty_cache()

    # Copied from transformers.models.t5.modeling_t5.T5Model.get_input_embeddings
    # 从 transformers.models.t5.modeling_t5.T5Model.get_input_embeddings 复制而来
    def get_input_embeddings(self):
        # 返回共享的输入嵌入层
        return self.shared

    # Copied from transformers.models.t5.modeling_t5.T5Model.set_input_embeddings
    # 从 transformers.models.t5.modeling_t5.T5Model.set_input_embeddings 复制而来
    def set_input_embeddings(self, new_embeddings):
        # 设置共享的输入嵌入层为新的嵌入
        self.shared = new_embeddings
        # 调用编码器的 set_input_embeddings 方法设置新的嵌入
        self.encoder.set_input_embeddings(new_embeddings)
        # 调用解码器的 set_input_embeddings 方法设置新的嵌入

    # Copied from transformers.models.t5.modeling_t5.T5Model.get_encoder
    # 从 transformers.models.t5.modeling_t5.T5Model.get_encoder 复制而来
    def get_encoder(self):
        # 返回编码器
        return self.encoder

    # Copied from transformers.models.t5.modeling_t5.T5Model.get_decoder
    # 从 transformers.models.t5.modeling_t5.T5Model.get_decoder 复制而来
    def get_decoder(self):
        # 返回解码器
        return self.decoder

    # Copied from transformers.models.t5.modeling_t5.T5Model._prune_heads
    # 从 transformers.models.t5.modeling_t5.T5Model._prune_heads 复制而来
    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        # 遍历需要修剪的层和头部的字典
        for layer, heads in heads_to_prune.items():
            # 在编码器的特定层的注意力头部上执行修剪操作
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(MT5_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqModelOutput, config_class=_CONFIG_FOR_DOC)
    # 从 transformers.models.t5.modeling_t5.T5Model.forward 复制过来，但将 T5->MT5, t5->mt5
    # 添加开始的文档字符串和替换返回文档字符串的注解
    # 定义一个方法 `forward`，用于模型的前向传播
    def forward(
        # 输入序列的标识符，可以是一个长整型张量，可选参数
        input_ids: Optional[torch.LongTensor] = None,
        # 注意力掩码，可以是一个浮点数张量，可选参数
        attention_mask: Optional[torch.FloatTensor] = None,
        # 解码器的输入序列的标识符，可以是一个长整型张量，可选参数
        decoder_input_ids: Optional[torch.LongTensor] = None,
        # 解码器的注意力掩码，可以是一个布尔张量，可选参数
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        # 头部掩码，可以是一个浮点数张量，可选参数
        head_mask: Optional[torch.FloatTensor] = None,
        # 解码器的头部掩码，可以是一个浮点数张量，可选参数
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        # 跨注意力头部掩码，可以是一个张量，可选参数
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        # 编码器的输出，可以是一系列浮点数张量的元组，可选参数
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        # 过去键值对，可以是一系列浮点数张量的元组，可选参数
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        # 输入嵌入，可以是一个张量，可选参数
        inputs_embeds: Optional[torch.Tensor] = None,
        # 解码器的输入嵌入，可以是一个张量，可选参数
        decoder_inputs_embeds: Optional[torch.Tensor] = None,
        # 是否使用缓存，布尔值，可选参数
        use_cache: Optional[bool] = None,
        # 是否输出注意力，布尔值，可选参数
        output_attentions: Optional[bool] = None,
        # 是否输出隐藏状态，布尔值，可选参数
        output_hidden_states: Optional[bool] = None,
        # 是否返回字典，布尔值，可选参数
        return_dict: Optional[bool] = None,
# 使用装饰器为类添加文档字符串，描述其作为基于 MT5 模型的带有语言建模头部的条件生成模型的特性
@add_start_docstrings("""MT5 Model with a `language modeling` head on top.""", MT5_START_DOCSTRING)
class MT5ForConditionalGeneration(MT5PreTrainedModel):
    r"""
    Examples:

    ```
    >>> from transformers import MT5ForConditionalGeneration, AutoTokenizer

    >>> model = MT5ForConditionalGeneration.from_pretrained("google/mt5-small")
    >>> tokenizer = AutoTokenizer.from_pretrained("google/mt5-small")
    >>> article = "UN Offizier sagt, dass weiter verhandelt werden muss in Syrien."
    >>> summary = "Weiter Verhandlung in Syrien."
    >>> inputs = tokenizer(article, text_target=summary, return_tensors="pt")

    >>> outputs = model(**inputs)
    >>> loss = outputs.loss
    ```"""

    # 模型类型设定为 "mt5"
    model_type = "mt5"
    # 配置类设定为 MT5Config
    config_class = MT5Config
    # 加载时忽略的键列表，用于处理未预期的键
    _keys_to_ignore_on_load_unexpected = ["decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.weight"]
    # 共享权重的键列表
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight", "lm_head.weight"]

    # 从 transformers.models.t5.modeling_t5.T5ForConditionalGeneration.__init__ 复制并替换 T5 为 MT5
    def __init__(self, config: MT5Config):
        super().__init__(config)
        # 设置模型维度为 config.d_model
        self.model_dim = config.d_model

        # 创建共享的嵌入层，用于词汇表大小和模型维度
        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        # 复制编码器配置，将其设定为非解码器
        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        # 创建 MT5 编码器堆栈
        self.encoder = MT5Stack(encoder_config, self.shared)

        # 复制解码器配置，将其设定为解码器
        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        # 创建 MT5 解码器堆栈
        self.decoder = MT5Stack(decoder_config, self.shared)

        # 创建线性层用于语言建模头部，输入维度为 config.d_model，输出维度为 config.vocab_size，无偏置
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # 初始化权重并应用最终处理
        self.post_init()

        # 模型并行设定为 False
        self.model_parallel = False
        # 设备映射设定为 None
        self.device_map = None

    # 使用装饰器添加并行化文档字符串
    @add_start_docstrings(PARALLELIZE_DOCSTRING)
    # 从 transformers.models.t5.modeling_t5.T5ForConditionalGeneration.parallelize 复制
    def parallelize(self, device_map=None):
        # 发出警告，提醒 `T5ForConditionalGeneration.parallelize` 方法将在 Transformers v5 中移除
        warnings.warn(
            "`T5ForConditionalGeneration.parallelize` is deprecated and will be removed in v5 of Transformers, you"
            " should load your model with `device_map='balanced'` in the call to `from_pretrained`. You can also"
            " provide your own `device_map` but it needs to be a dictionary module_name to device, so for instance"
            " {'encoder.block.0': 0, 'encoder.block.1': 1, ...}",
            FutureWarning,
        )
        # 根据 encoder.block 的数量和当前 CUDA 设备数量生成设备映射，如果未提供 device_map 则使用生成的映射
        self.device_map = (
            get_device_map(len(self.encoder.block), range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        # 检查设备映射的有效性
        assert_device_map(self.device_map, len(self.encoder.block))
        # 并行化编码器
        self.encoder.parallelize(self.device_map)
        # 并行化解码器
        self.decoder.parallelize(self.device_map)
        # 将语言模型头部移动到解码器的第一个设备上
        self.lm_head = self.lm_head.to(self.decoder.first_device)
        # 设置模型并行化标志为 True
        self.model_parallel = True

    @add_start_docstrings(DEPARALLELIZE_DOCSTRING)
    # 从 transformers.models.t5.modeling_t5.T5ForConditionalGeneration.deparallelize 复制而来
    def deparallelize(self):
        # 发出警告，提醒 `deparallelize` 方法将在 Transformers v5 中移除
        warnings.warn(
            "Like `parallelize`, `deparallelize` is deprecated and will be removed in v5 of Transformers.",
            FutureWarning,
        )
        # 反并行化编码器
        self.encoder.deparallelize()
        # 反并行化解码器
        self.decoder.deparallelize()
        # 将编码器移动到 CPU
        self.encoder = self.encoder.to("cpu")
        # 将解码器移动到 CPU
        self.decoder = self.decoder.to("cpu")
        # 将语言模型头部移动到 CPU
        self.lm_head = self.lm_head.to("cpu")
        # 设置模型并行化标志为 False
        self.model_parallel = False
        # 清空 CUDA 缓存
        torch.cuda.empty_cache()

    # 从 transformers.models.t5.modeling_t5.T5ForConditionalGeneration.get_input_embeddings 复制而来
    def get_input_embeddings(self):
        # 返回共享的输入嵌入
        return self.shared

    # 从 transformers.models.t5.modeling_t5.T5ForConditionalGeneration.set_input_embeddings 复制而来
    def set_input_embeddings(self, new_embeddings):
        # 设置共享的输入嵌入
        self.shared = new_embeddings
        # 设置编码器的输入嵌入
        self.encoder.set_input_embeddings(new_embeddings)
        # 设置解码器的输入嵌入
        self.decoder.set_input_embeddings(new_embeddings)

    # 从 transformers.models.t5.modeling_t5.T5ForConditionalGeneration.set_output_embeddings 复制而来
    def set_output_embeddings(self, new_embeddings):
        # 设置语言模型头部的输出嵌入
        self.lm_head = new_embeddings

    # 从 transformers.models.t5.modeling_t5.T5ForConditionalGeneration.get_output_embeddings 复制而来
    def get_output_embeddings(self):
        # 返回语言模型头部的输出嵌入
        return self.lm_head

    # 从 transformers.models.t5.modeling_t5.T5ForConditionalGeneration.get_encoder 复制而来
    def get_encoder(self):
        # 返回编码器
        return self.encoder

    # 从 transformers.models.t5.modeling_t5.T5ForConditionalGeneration.get_decoder 复制而来
    def get_decoder(self):
        # 返回解码器
        return self.decoder

    @add_start_docstrings_to_model_forward(MT5_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    # 从 transformers.models.t5.modeling_t5.T5ForConditionalGeneration.forward 复制而来，定义了 MT5 模型的前向传播方法
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,  # 输入的 token IDs，类型为可选的长整型张量
        attention_mask: Optional[torch.FloatTensor] = None,  # 注意力掩码，类型为可选的浮点数张量
        decoder_input_ids: Optional[torch.LongTensor] = None,  # 解码器的输入 token IDs，类型为可选的长整型张量
        decoder_attention_mask: Optional[torch.BoolTensor] = None,  # 解码器的注意力掩码，类型为可选的布尔张量
        head_mask: Optional[torch.FloatTensor] = None,  # 头部掩码，类型为可选的浮点数张量
        decoder_head_mask: Optional[torch.FloatTensor] = None,  # 解码器头部掩码，类型为可选的浮点数张量
        cross_attn_head_mask: Optional[torch.Tensor] = None,  # 跨注意力头部掩码，类型为可选的张量
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,  # 编码器的输出，类型为可选的张量元组
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,  # 过去的键值对，类型为可选的张量元组
        inputs_embeds: Optional[torch.FloatTensor] = None,  # 输入的嵌入，类型为可选的浮点数张量
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,  # 解码器输入的嵌入，类型为可选的浮点数张量
        labels: Optional[torch.LongTensor] = None,  # 标签，类型为可选的长整型张量
        use_cache: Optional[bool] = None,  # 是否使用缓存，类型为可选的布尔值
        output_attentions: Optional[bool] = None,  # 是否输出注意力，类型为可选的布尔值
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态，类型为可选的布尔值
        return_dict: Optional[bool] = None,  # 是否返回字典形式的输出，类型为可选的布尔值
    ):
    
    # 从 transformers.models.t5.modeling_t5.T5ForConditionalGeneration.prepare_inputs_for_generation 复制而来，准备生成过程中的输入
    def prepare_inputs_for_generation(
        self,
        input_ids,  # 输入的 token IDs
        past_key_values=None,  # 过去的键值对，默认为 None
        attention_mask=None,  # 注意力掩码，默认为 None
        head_mask=None,  # 头部掩码，默认为 None
        decoder_head_mask=None,  # 解码器头部掩码，默认为 None
        decoder_attention_mask=None,  # 解码器的注意力掩码，默认为 None
        cross_attn_head_mask=None,  # 跨注意力头部掩码，默认为 None
        use_cache=None,  # 是否使用缓存，默认为 None
        encoder_outputs=None,  # 编码器的输出，默认为 None
        **kwargs,  # 其他关键字参数
    ):
        # 如果使用了过去的键值对
        if past_key_values is not None:
            # 获取过去键值对的长度
            past_length = past_key_values[0][0].shape[2]
    
            # 如果输入的 token IDs 的长度大于过去键值对的长度
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length  # 移除前缀的长度设为过去键值对的长度
            else:
                # 否则，默认采用旧的行为：只保留最后一个输入 ID
                remove_prefix_length = input_ids.shape[1] - 1
    
            # 将输入的 token IDs 裁剪为移除前缀长度后的部分
            input_ids = input_ids[:, remove_prefix_length:]
    
        # 返回准备好的输入字典
        return {
            "decoder_input_ids": input_ids,
            "past_key_values": past_key_values,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "decoder_attention_mask": decoder_attention_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
        }
    
    # 从 transformers.models.t5.modeling_t5.T5ForConditionalGeneration.prepare_decoder_input_ids_from_labels 复制而来，准备从标签生成解码器输入 token IDs
    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return self._shift_right(labels)
    
    # 从 transformers.models.t5.modeling_t5.T5ForConditionalGeneration._reorder_cache 复制而来，重新排序缓存
    # 重新排列缓存中的过去键值，以便与beam索引对应
    def _reorder_cache(self, past_key_values, beam_idx):
        # 如果过去的键值未包含在输出中
        # 禁用快速解码，无需重新排序
        if past_key_values is None:
            # 提示用户可能需要设置`use_cache=True`来加快解码速度
            logger.warning("You might want to consider setting `use_cache=True` to speed up decoding")
            return past_key_values

        # 重新排序后的解码器过去状态
        reordered_decoder_past = ()
        for layer_past_states in past_key_values:
            # 从层过去状态中获取正确的批次索引，批次维度在第二个位置
            reordered_layer_past_states = ()
            for layer_past_state in layer_past_states:
                # 需要为每个四个键/值状态设置正确的`past`
                reordered_layer_past_states = reordered_layer_past_states + (
                    layer_past_state.index_select(0, beam_idx.to(layer_past_state.device)),
                )

            # 检查重新排序后的第一个层过去状态的形状与原始的是否匹配
            if reordered_layer_past_states[0].shape != layer_past_states[0].shape:
                raise ValueError(
                    f"reordered_layer_past_states[0] shape {reordered_layer_past_states[0].shape} and layer_past_states[0] shape {layer_past_states[0].shape} mismatched"
                )
            # 检查重新排序后的过去状态列表长度与原始列表是否匹配
            if len(reordered_layer_past_states) != len(layer_past_states):
                raise ValueError(
                    f"length of reordered_layer_past_states {len(reordered_layer_past_states)} and length of layer_past_states {len(layer_past_states)} mismatched"
                )

            # 将重新排序后的层过去状态添加到重新排序后的解码器过去状态中
            reordered_decoder_past = reordered_decoder_past + (reordered_layer_past_states,)
        
        # 返回重新排序后的解码器过去状态
        return reordered_decoder_past
# 使用装饰器为类添加文档字符串，描述了该类的基本信息和使用示例
@add_start_docstrings(
    "The bare MT5 Model transformer outputting encoder's raw hidden-states without any specific head on top.",
    MT5_START_DOCSTRING,
)
class MT5EncoderModel(MT5PreTrainedModel):
    r"""
    Examples:

    ```
    >>> from transformers import MT5EncoderModel, AutoTokenizer

    >>> model = MT5EncoderModel.from_pretrained("google/mt5-small")
    >>> tokenizer = AutoTokenizer.from_pretrained("google/mt5-small")
    >>> article = "UN Offizier sagt, dass weiter verhandelt werden muss in Syrien."
    >>> input_ids = tokenizer(article, return_tensors="pt").input_ids
    >>> outputs = model(input_ids)
    >>> hidden_state = outputs.last_hidden_state
    ```"""

    # 设置模型类型为 "mt5"
    model_type = "mt5"
    # 指定配置类为 MT5Config
    config_class = MT5Config
    # 定义了需要绑定权重的键列表
    _tied_weights_keys = ["encoder.embed_tokens.weight"]

    # 从 transformers.models.t5.modeling_t5.T5EncoderModel.__init__ 复制并修改为 MT5EncoderModel
    def __init__(self, config: MT5Config):
        super().__init__(config)
        # 创建共享的嵌入层，使用配置中的词汇表大小和模型维度
        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        # 复制配置以便修改而不影响原始配置，设置不使用缓存和不是编码器-解码器模型
        encoder_config = copy.deepcopy(config)
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        # 创建 MT5 堆栈编码器
        self.encoder = MT5Stack(encoder_config, self.shared)

        # 初始化权重并应用最终处理
        self.post_init()

        # 模型并行设置
        self.model_parallel = False
        self.device_map = None

    # 从 transformers.models.t5.modeling_t5.T5EncoderModel.parallelize 复制而来
    @add_start_docstrings(PARALLELIZE_DOCSTRING)
    def parallelize(self, device_map=None):
        # 发出警告，说明方法已弃用，将在 Transformers v5 版本中删除
        warnings.warn(
            "`T5EncoderModel.parallelize` is deprecated and will be removed in v5 of Transformers, you should load"
            " your model with `device_map='balanced'` in the call to `from_pretrained`. You can also provide your own"
            " `device_map` but it needs to be a dictionary module_name to device, so for instance {'block.0': 0,"
            " 'block.1': 1, ...}",
            FutureWarning,
        )
        # 根据传入的 device_map 参数设置设备映射
        self.device_map = (
            get_device_map(len(self.encoder.block), range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        # 断言设备映射的有效性
        assert_device_map(self.device_map, len(self.encoder.block))
        # 将编码器对象分布到多个设备上
        self.encoder.parallelize(self.device_map)
        self.model_parallel = True

    # 从 transformers.models.t5.modeling_t5.T5EncoderModel.deparallelize 复制而来
    @add_start_docstrings(DEPARALLELIZE_DOCSTRING)
    def deparallelize(self):
        # 发出警告，说明方法已弃用，将在 Transformers v5 版本中删除
        warnings.warn(
            "Like `parallelize`, `deparallelize` is deprecated and will be removed in v5 of Transformers.",
            FutureWarning,
        )
        # 取消编码器对象的并行化
        self.encoder.deparallelize()
        # 将编码器对象移回 CPU
        self.encoder = self.encoder.to("cpu")
        self.model_parallel = False
        self.device_map = None
        # 清空 CUDA 缓存
        torch.cuda.empty_cache()

    # 从 transformers.models.t5.modeling_t5.T5EncoderModel.get_input_embeddings 复制而来
    # 返回当前模型共享的输入嵌入向量
    def get_input_embeddings(self):
        return self.shared

    # 从给定的新嵌入向量设置模型共享的输入嵌入向量，并更新编码器的输入嵌入
    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)

    # 返回当前模型的编码器
    def get_encoder(self):
        return self.encoder

    # 剪枝模型中编码器的注意力头
    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.block[layer].layer[0].SelfAttention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(MT5_ENCODER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutput, config_class=_CONFIG_FOR_DOC)
    # 重写的前向传播函数，用于MT5模型，接受多种输入并返回编码器的输出
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.FloatTensor], BaseModelOutput]:
        r"""
        Returns:

        Example:

        ```
        >>> from transformers import AutoTokenizer, MT5EncoderModel

        >>> tokenizer = AutoTokenizer.from_pretrained("google-mt5/mt5-small")
        >>> model = MT5EncoderModel.from_pretrained("google-mt5/mt5-small")
        >>> input_ids = tokenizer(
        ...     "Studies have been shown that owning a dog is good for you", return_tensors="pt"
        ... ).input_ids  # Batch size 1
        >>> outputs = model(input_ids=input_ids)
        >>> last_hidden_states = outputs.last_hidden_state
        ```"""
        # 如果return_dict未指定，则根据配置确定是否使用返回字典
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用编码器的前向传播，传递输入参数并返回编码器的输出
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        return encoder_outputs
"""
MT5 model with a sequence classification/head on top (a linear layer on top of the pooled output) e.g. for GLUE
tasks.
"""
@add_start_docstrings(
    """
    MT5 model with a sequence classification/head on top (a linear layer on top of the pooled output) e.g. for GLUE
    tasks.
    """,
    MT5_START_DOCSTRING,
)
class MT5ForSequenceClassification(MT5PreTrainedModel):
    _keys_to_ignore_on_load_unexpected = ["decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.weight"]
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight"]

    # Copied from transformers.models.t5.modeling_t5.T5ForSequenceClassification.__init__ with T5->MT5
    def __init__(self, config: MT5Config):
        super().__init__(config)
        self.transformer = MT5Model(config)  # 初始化MT5模型
        self.classification_head = MT5ClassificationHead(config)  # 初始化分类头部

        # Initialize weights and apply final processing
        self.post_init()  # 初始化后处理步骤

        self.model_parallel = False  # 设置模型并行为False

    @add_start_docstrings_to_model_forward(MT5_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqSequenceClassifierOutput, config_class=_CONFIG_FOR_DOC)
    # Copied from transformers.models.t5.modeling_t5.T5ForSequenceClassification.forward
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        """
        Perform forward pass of the MT5 model for sequence classification.
        """
        # Forward pass through MT5 model and classification head
        # 正向传播通过MT5模型和分类头部
        # 详细参数说明参见MT5_INPUTS_DOCSTRING
        pass


"""
MT5 Encoder Model with a token classification head on top (a linear layer on top of the hidden-states output)
e.g. for Named-Entity-Recognition (NER) tasks.
"""
@add_start_docstrings(
    """
    MT5 Encoder Model with a token classification head on top (a linear layer on top of the hidden-states output)
    e.g. for Named-Entity-Recognition (NER) tasks.
    """,
    MT5_START_DOCSTRING,
)
class MT5ForTokenClassification(MT5PreTrainedModel):
    _tied_weights_keys = ["transformer.encoder.embed_tokens.weight"]

    # Copied from transformers.models.t5.modeling_t5.T5ForTokenClassification.__init__ with T5->MT5
    def __init__(self, config: MT5Config):
        super().__init__(config)
        self.num_labels = config.num_labels  # 设置标签数量

        self.transformer = MT5EncoderModel(config)  # 初始化MT5编码器模型
        self.dropout = nn.Dropout(config.classifier_dropout)  # 初始化Dropout层
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)  # 初始化线性分类器

        # Initialize weights and apply final processing
        self.post_init()  # 初始化后处理步骤

    @add_start_docstrings_to_model_forward(MT5_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TokenClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        """
        Perform forward pass of the MT5 model for token classification.
        """
        # Forward pass through MT5 model and token classification head
        # 正向传播通过MT5模型和标记分类头部
        # 详细参数说明参见MT5_INPUTS_DOCSTRING
        pass
    # 从transformers.models.mt5.modeling_mt5.MT5ForTokenClassification.forward中复制而来
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            计算标记分类损失的标签。索引应在 `[0, ..., config.num_labels - 1]` 范围内。
        Returns:
            返回一个元组或者TokenClassifierOutput对象。
        """
        # 确定是否返回字典格式的输出
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 使用transformer模型处理输入
        outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取模型输出中的隐藏状态
        hidden_states = outputs[0]
        # 对隐藏状态应用dropout层
        hidden_states = self.dropout(hidden_states)
        # 将处理后的隐藏状态传入分类器得到logits
        logits = self.classifier(hidden_states)

        # 初始化损失值为None
        loss = None
        # 如果有标签，则计算损失值
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # 计算交叉熵损失
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        # 如果不要求返回字典格式的输出
        if not return_dict:
            # 构建输出元组
            output = (logits, outputs[2:-1])
            # 如果损失不为None，则将损失值加入输出元组中
            return ((loss,) + output) if loss is not None else output

        # 返回TokenClassifierOutput对象，包含损失、logits、隐藏状态和注意力值
        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
@add_start_docstrings(
    """
    MT5 Model with a span classification head on top for extractive question-answering tasks like SQuAD (linear layers
    on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    MT5_START_DOCSTRING,
)
class MT5ForQuestionAnswering(MT5PreTrainedModel):
    _keys_to_ignore_on_load_unexpected = ["decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.weight"]
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight"]

    # Copied from transformers.models.t5.modeling_t5.T5ForQuestionAnswering.__init__ with T5->MT5
    def __init__(self, config: MT5Config):
        super().__init__(config)
        self.model_dim = config.d_model

        # Embedding layer shared between encoder and decoder
        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        # Initialize encoder with MT5Stack
        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = MT5Stack(encoder_config, self.shared)

        # Initialize decoder with MT5Stack
        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = MT5Stack(decoder_config, self.shared)

        # Output layer for question answering logits
        self.num_labels = config.num_labels
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

        self.model_parallel = False

    # Copied from transformers.models.t5.modeling_t5.T5ForQuestionAnswering.get_input_embeddings
    def get_input_embeddings(self):
        return self.shared

    # Copied from transformers.models.t5.modeling_t5.T5ForQuestionAnswering.set_input_embeddings
    def set_input_embeddings(self, new_embeddings):
        # Set new embeddings for shared layer and update encoder and decoder embeddings
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    # Copied from transformers.models.t5.modeling_t5.T5ForQuestionAnswering.get_encoder
    def get_encoder(self):
        return self.encoder

    # Copied from transformers.models.t5.modeling_t5.T5ForQuestionAnswering.get_decoder
    def get_decoder(self):
        return self.decoder

    @add_start_docstrings_to_model_forward(MT5_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqQuestionAnsweringModelOutput, config_class=_CONFIG_FOR_DOC)
    # Copied from transformers.models.t5.modeling_t5.T5ForQuestionAnswering.forward
    # 定义模型的前向传播方法，接受多个可选的输入参数
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,  # 输入序列的token IDs，可选的长整型张量
        attention_mask: Optional[torch.FloatTensor] = None,  # 输入序列的注意力掩码，可选的浮点数张量
        decoder_input_ids: Optional[torch.LongTensor] = None,  # 解码器输入序列的token IDs，可选的长整型张量
        decoder_attention_mask: Optional[torch.BoolTensor] = None,  # 解码器输入序列的注意力掩码，可选的布尔张量
        head_mask: Optional[torch.FloatTensor] = None,  # 多头注意力机制的头掩码，可选的浮点数张量
        decoder_head_mask: Optional[torch.FloatTensor] = None,  # 解码器的多头注意力机制的头掩码，可选的浮点数张量
        cross_attn_head_mask: Optional[torch.Tensor] = None,  # 交叉注意力机制的头掩码，可选的张量
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,  # 编码器输出的元组，可选的张量元组
        start_positions: Optional[torch.LongTensor] = None,  # 开始位置的token IDs，可选的长整型张量
        end_positions: Optional[torch.LongTensor] = None,  # 结束位置的token IDs，可选的长整型张量
        inputs_embeds: Optional[torch.FloatTensor] = None,  # 输入嵌入的张量，可选的浮点数张量
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,  # 解码器输入嵌入的张量，可选的浮点数张量
        use_cache: Optional[bool] = None,  # 是否使用缓存，可选的布尔值
        output_attentions: Optional[bool] = None,  # 是否输出注意力权重，可选的布尔值
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态，可选的布尔值
        return_dict: Optional[bool] = None,  # 是否返回字典格式的结果，可选的布尔值
```