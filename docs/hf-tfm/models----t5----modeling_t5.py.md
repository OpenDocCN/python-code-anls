# `.\transformers\models\t5\modeling_t5.py`

```py
# 设置文件编码为 UTF-8
# 版权声明，包括 Mesh TensorFlow 作者、T5 作者以及 HuggingFace Inc. 团队
# 根据 Apache 许可证 2.0 版本http://www.apache.org/licenses/LICENSE-2.0，授权范围内使用此文件
# 不得在未遵守许可证的情况下使用此文件
# 可以在预先警告的情况下写入文件
# 根据许可证的规定，按“现状”提供软件，没有任何明示或暗示的担保或条件
# 详细了解许可证条款，控制特定语言的授权，限制
""" PyTorch T5 model."""

# 导入所需库
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

# 设置日志记录器
logger = logging.get_logger(__name__)

# 针对文档提供的 T5 配置和检查点的说明
_CONFIG_FOR_DOC = "T5Config"
_CHECKPOINT_FOR_DOC = "t5-small"

# 预训练权重字典，包含各模型的ID和相关URL
####################################################
# This dict contains ids and associated url
# for the pretrained weights provided with the models
####################################################
T5_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "t5-small",
    "t5-base",
    "t5-large",
    "t5-3b",
    "t5-11b",
    # 查看所有 T5 模型：https://huggingface.co/models?filter=t5
]

# TF 1.0 到 PyTorch 的转换方法
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
    # 获取 TF 检查点的绝对路径
    tf_path = os.path.abspath(tf_checkpoint_path)
    logger.info(f"Converting TensorFlow checkpoint from {tf_path}")
    # 从 TF 模型加载权重
    init_vars = tf.train.list_variables(tf_path)
    names = []
    # 初始化一个空的字典，用于存储 TensorFlow 的权重
    tf_weights = {}
    # 遍历初始化变量列表，其中包含了权重的名称和形状
    for name, shape in init_vars:
        # 记录日志，显示正在加载的 TensorFlow 权重的名称和形状
        logger.info(f"Loading TF weight {name} with shape {shape}")
        # 从 TensorFlow 路径加载指定名称的变量，并存储到数组中
        array = tf.train.load_variable(tf_path, name)
        # 将权重名称添加到列表中
        names.append(name)
        # 将加载的权重数组存储到字典中，键为权重名称
        tf_weights[name] = array
    
    # 记录日志，显示未复制到 PyTorch 模型的权重名称
    logger.info(f"Weights not copied to PyTorch model: {', '.join(tf_weights.keys())}.")
    # 返回模型
    return model
# 定义了一个字符串变量，用于并行处理的文档字符串说明
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

                - t5-small: 6
                - t5-base: 12
                - t5-large: 24
                - t5-3b: 24
                - t5-11b: 24

    Example:

    ```python
    # Here is an example of a device map on a machine with 4 GPUs using t5-3b, which has a total of 24 attention modules:
    model = T5ForConditionalGeneration.from_pretrained("t5-3b")
    device_map = {
        0: [0, 1, 2],
        1: [3, 4, 5, 6, 7, 8, 9],
        2: [10, 11, 12, 13, 14, 15, 16],
        3: [17, 18, 19, 20, 21, 22, 23],
    }
    model.parallelize(device_map)
    ```py
"""
# 定义了另一个字符串变量，用于取消并行处理的文档字符串说明
DEPARALLELIZE_DOCSTRING = r"""
    Moves the model to cpu from a model parallel state.

    Example:

    ```python
    # On a 4 GPU machine with t5-3b:
    model = T5ForConditionalGeneration.from_pretrained("t5-3b")
    device_map = {
        0: [0, 1, 2],
        1: [3, 4, 5, 6, 7, 8, 9],
        2: [10, 11, 12, 13, 14, 15, 16],
        3: [17, 18, 19, 20, 21, 22, 23],
    }
    model.parallelize(device_map)  # Splits the model across several devices
    model.deparallelize()  # Put the model back on cpu and cleans memory by calling torch.cuda.empty_cache()
    ```py
"""

# 定义了一个 T5LayerNorm 类，继承自 nn.Module
class T5LayerNorm(nn.Module):
    # 初始化方法，构造 T5 风格的 LayerNormalization 模块，没有偏置和均值减法
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
    def forward(self, hidden_states):
        # T5模型使用层归一化（Layer Normalization）来规模化张量，不进行平移操作，这种归一化方法有时被称为均值平方归一化（Root Mean Square Layer Normalization）
        # 参考论文：https://arxiv.org/abs/1910.07467
        # 在此操作中，方差是在不使用均值的情况下计算的，并且没有偏置。此外，我们需要确保对于半精度输入的累积使用浮点32位进行操作。

        # 计算隐藏状态的方差，将其转换为浮点型，并在最后一维上进行平均操作
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        # 对隐藏状态进行归一化处理，使用方差加上一个微小的ε来防止除以0的错误
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        # 如果权重的数据类型为半精度浮点数或者双精度浮点数，则将隐藏状态转换为相同的精度
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        # 返回应用权重的隐藏状态
        return self.weight * hidden_states
try:
    # 尝试导入apex.normalization模块的FusedRMSNorm类
    from apex.normalization import FusedRMSNorm

    # 将FusedRMSNorm类赋值给T5LayerNorm
    T5LayerNorm = FusedRMSNorm  # noqa

    # 输出日志信息
    logger.info("Discovered apex.normalization.FusedRMSNorm - will use it instead of T5LayerNorm")
except ImportError:
    # 如果导入失败，则使用普通的T5LayerNorm
    pass
except Exception:
    # 如果导入apex失败，则输出警告信息，然后使用T5LayerNorm
    logger.warning("discovered apex but it failed to load, falling back to T5LayerNorm")
    pass

# 将T5LayerNorm添加到ALL_LAYERNORM_LAYERS列表中
ALL_LAYERNORM_LAYERS.append(T5LayerNorm)


class T5DenseActDense(nn.Module):
    def __init__(self, config: T5Config):
        super().__init__()
        # 初始化wi线性层，输入维度为config.d_model，输出维度为config.d_ff
        self.wi = nn.Linear(config.d_model, config.d_ff, bias=False)
        # 初始化wo线性层，输入维度为config.d_ff，输出维度为config.d_model
        self.wo = nn.Linear(config.d_ff, config.d_model, bias=False)
        # 初始化dropout层，丢弃概率为config.dropout_rate
        self.dropout = nn.Dropout(config.dropout_rate)
        # 根据配置选择相应的激活函数
        self.act = ACT2FN[config.dense_act_fn]

    def forward(self, hidden_states):
        # 将输入数据通过wi线性层
        hidden_states = self.wi(hidden_states)
        # 将线性层的输出应用激活函数
        hidden_states = self.act(hidden_states)
        # 对激活函数的输出应用dropout
        hidden_states = self.dropout(hidden_states)
        # 如果wo的权重为torch.Tensor类型，并且hidden_states和wo的权重数据类型不一致，并且wo的权重数据类型不为torch.int8类型
        if (
            isinstance(self.wo.weight, torch.Tensor)
            and hidden_states.dtype != self.wo.weight.dtype
            and self.wo.weight.dtype != torch.int8
        ):
            # 将hidden_states的数据类型转换为wo的权重数据类型
            hidden_states = hidden_states.to(self.wo.weight.dtype)
        # 将hidden_states通过wo线性层
        hidden_states = self.wo(hidden_states)
        # 返回输出结果
        return hidden_states


class T5DenseGatedActDense(nn.Module):
    def __init__(self, config: T5Config):
        super().__init__()
        # 初始化wi_0线性层，输入维度为config.d_model，输出维度为config.d_ff
        self.wi_0 = nn.Linear(config.d_model, config.d_ff, bias=False)
        # 初始化wi_1线性层，输入维度为config.d_model，输出维度为config.d_ff
        self.wi_1 = nn.Linear(config.d_model, config.d_ff, bias=False)
        # 初始化wo线性层，输入维度为config.d_ff，输出维度为config.d_model
        self.wo = nn.Linear(config.d_ff, config.d_model, bias=False)
        # 初始化dropout层，丢弃概率为config.dropout_rate
        self.dropout = nn.Dropout(config.dropout_rate)
        # 根据配置选择相应的激活函数
        self.act = ACT2FN[config.dense_act_fn]

    def forward(self, hidden_states):
        # 将输入数据通过wi_0线性层
        hidden_gelu = self.act(self.wi_0(hidden_states))
        # 将输入数据通过wi_1线性层
        hidden_linear = self.wi_1(hidden_states)
        # 使用element-wise乘法将hidden_gelu和hidden_linear的结果相乘
        hidden_states = hidden_gelu * hidden_linear
        # 对相乘结果应用dropout
        hidden_states = self.dropout(hidden_states)

        # 为了使8位量化适用于google/flan-t5-xxl，self.wo保持为float32类型
        # 参考链接：https://github.com/huggingface/transformers/issues/20287
        # 同时，确保权重不为`int8`类型，以防用户将`_keep_in_fp32_modules`强制为`None`
        if (
            isinstance(self.wo.weight, torch.Tensor)
            and hidden_states.dtype != self.wo.weight.dtype
            and self.wo.weight.dtype != torch.int8
        ):
            # 将hidden_states的数据类型转换为wo的权重数据类型
            hidden_states = hidden_states.to(self.wo.weight.dtype)

        # 将hidden_states通过wo线性层
        hidden_states = self.wo(hidden_states)
        # 返回输出结果
        return hidden_states


class T5LayerFF(nn.Module):
    def __init__(self, config: T5Config):
        super().__init__()
        if config.is_gated_act:
            # 如果配置为gated_act，则使用T5DenseGatedActDense层
            self.DenseReluDense = T5DenseGatedActDense(config)
        else:
            # 否则使用T5DenseActDense层
            self.DenseReluDense = T5DenseActDense(config)

        # 初始化layer_norm层，输入维度为config.d_model，epsilon为config.layer_norm_epsilon
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        # 初始化dropout层，丢弃概率为config.dropout_rate
        self.dropout = nn.Dropout(config.dropout_rate)
    # 通过正向传播计算隐藏状态
    def forward(self, hidden_states):
        # 将隐藏状态进行层归一化处理
        forwarded_states = self.layer_norm(hidden_states)
        # 将归一化后的隐藏状态输入到DenseReluDense层中进行处理
        forwarded_states = self.DenseReluDense(forwarded_states)
        # 将原始隐藏状态与处理后的隐藏状态相加，并进行dropout处理
        hidden_states = hidden_states + self.dropout(forwarded_states)
        # 返回计算后的隐藏状态
        return hidden_states
    # 定义 T5Attention 类，继承自 nn.Module
class T5Attention(nn.Module):
    # 初始化函数，接受一个 T5Config 对象和一个布尔类型的参数
    def __init__(self, config: T5Config, has_relative_attention_bias=False):
        # 调用父类的初始化函数
        super().__init__()
        # 根据 T5Config 对象初始化一系列属性
        self.is_decoder = config.is_decoder
        self.has_relative_attention_bias = has_relative_attention_bias
        self.relative_attention_num_buckets = config.relative_attention_num_buckets
        self.relative_attention_max_distance = config.relative_attention_max_distance
        self.d_model = config.d_model
        self.key_value_proj_dim = config.d_kv
        self.n_heads = config.num_heads
        self.dropout = config.dropout_rate
        self.inner_dim = self.n_heads * self.key_value_proj_dim

        # 使用 nn.Linear 定义线性变换，将 d_model 映射成 inner_dim，不使用偏置
        self.q = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.k = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.v = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.o = nn.Linear(self.inner_dim, self.d_model, bias=False)

        # 如果 has_relative_attention_bias 为 True，则初始化相对注意力偏置的 Embedding 层
        if self.has_relative_attention_bias:
            self.relative_attention_bias = nn.Embedding(self.relative_attention_num_buckets, self.n_heads)
        # 初始化一个空集合，用于存储剪枝的头部
        self.pruned_heads = set()
        # 默认不使用梯度检查点
        self.gradient_checkpointing = False

    # 定义剪枝头部的方法
    def prune_heads(self, heads):
        # 如果传入的头部列表为空，则直接返回
        if len(heads) == 0:
            return
        # 调用辅助函数 find_pruneable_heads_and_indices 定位可剪枝的头部和它们的索引
        heads, index = find_pruneable_heads_and_indices(
            heads, self.n_heads, self.key_value_proj_dim, self.pruned_heads
        )
        # 对线性层进行剪枝
        self.q = prune_linear_layer(self.q, index)
        self.k = prune_linear_layer(self.k, index)
        self.v = prune_linear_layer(self.v, index)
        self.o = prune_linear_layer(self.o, index, dim=1)
        # 更新超参数
        self.n_heads = self.n_heads - len(heads)
        self.inner_dim = self.key_value_proj_dim * self.n_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    # 定义一个静态方法
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
            relative_position: an int32 Tensor
            bidirectional: a boolean - whether the attention is bidirectional
            num_buckets: an integer
            max_distance: an integer

        Returns:
            a Tensor with the same shape as relative_position, containing int32 values in the range [0, num_buckets)
        """
        # 初始化相对位置的bucket
        relative_buckets = 0
        
        # 如果是双向的相对位置，将bucket数量减半
        if bidirectional:
            num_buckets //= 2
            # 根据相对位置是否大于零，进行计算相对的bucket位置
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            # 如果不是双向相对位置，则直接取绝对值
            relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
        # 现在相对位置在[0,∞)范围内

        # 一半的bucket用于精确的位置增量
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        # 另外一半的bucket用于位置在max_distance范围内的对数增量
        relative_position_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)
        # 限制相对位置bucket不超过最大值
        relative_position_if_large = torch.min(
            relative_position_if_large, torch.full_like(relative_position_if_large, num_buckets - 1)
        )

        relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)
        return relative_buckets
    def compute_bias(self, query_length, key_length, device=None):
        """计算分桶相对位置偏差"""
        # 如果设备为空，则使用self.relative_attention_bias.weight.device作为设备
        if device is None:
            device = self.relative_attention_bias.weight.device
        # 创建一个形状为(query_length, 1)的长整型张量，作为上下文位置
        context_position = torch.arange(query_length, dtype=torch.long, device=device)[:, None]
        # 创建一个形状为(1, key_length)的长整型张量，作为记忆位置
        memory_position = torch.arange(key_length, dtype=torch.long, device=device)[None, :]
        # 计算相对位置，即记忆位置减去上下文位置，形状为(query_length, key_length)
        relative_position = memory_position - context_position
        # 将相对位置分到不同的桶中，形状为(query_length, key_length)
        relative_position_bucket = self._relative_position_bucket(
            relative_position,
            bidirectional=(not self.is_decoder),
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance,
        )
        # 使用相对位置的桶对应的权重来计算相对位置偏差，形状为(query_length, key_length, num_heads)
        values = self.relative_attention_bias(relative_position_bucket)
        # 调整形状为(1, num_heads, query_length, key_length)，并在最前面添加一个维度
        values = values.permute([2, 0, 1]).unsqueeze(0)
        # 返回相对位置偏差
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


注释：
# 定义 T5 模型的自注意力层
class T5LayerSelfAttention(nn.Module):
    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__()
        # 初始化自注意力层对象
        self.SelfAttention = T5Attention(config, has_relative_attention_bias=has_relative_attention_bias)
        # 初始化层归一化对象
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        # 初始化丢弃层对象
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
        # 对隐藏状态进行层归一化
        normed_hidden_states = self.layer_norm(hidden_states)
        # 调用自注意力层计算注意力输出
        attention_output = self.SelfAttention(
            normed_hidden_states,
            mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        # 计算加上注意力输出的隐藏状态
        hidden_states = hidden_states + self.dropout(attention_output[0])
        # 返回结果，包括隐藏状态和注意力输出（如果有）
        outputs = (hidden_states,) + attention_output[1:]  # add attentions if we output them
        return outputs


# 定义 T5 模型的交叉注意力层
class T5LayerCrossAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 初始化交叉注意力层对象
        self.EncDecAttention = T5Attention(config, has_relative_attention_bias=False)
        # 初始化层归一化对象
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
         # 初始化丢弃层对象
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
        # 对隐藏状态进行层归一化
        normed_hidden_states = self.layer_norm(hidden_states)
        # 调用交叉注意力层计算注意力输出
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
        # 计算加上注意力输出的隐藏状态
        layer_output = hidden_states + self.dropout(attention_output[0])
        # 返回结果，包括隐藏状态和注意力输出（如果有）
        outputs = (layer_output,) + attention_output[1:]  # add attentions if we output them
        return outputs


# 定义 T5 模型的块
class T5Block(nn.Module):
    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__()
        # 判断是否为解码器
        self.is_decoder = config.is_decoder
        # 初始化块的层列表
        self.layer = nn.ModuleList()
        # 添加自注意力层
        self.layer.append(T5LayerSelfAttention(config, has_relative_attention_bias=has_relative_attention_bias))
        # 如果是解码器，再添加交叉注意力层
        if self.is_decoder:
            self.layer.append(T5LayerCrossAttention(config))
        # 添加前馈层
        self.layer.append(T5LayerFF(config))
    # 定义一个前向传播函数，接收多个输入参数
    def forward(
        self,
        hidden_states,  # 输入的隐藏状态
        attention_mask=None,  # 注意力掩码，默认为None
        position_bias=None,  # 位置偏置，默认为None
        encoder_hidden_states=None,  # 编码器隐藏状态，默认为None
        encoder_attention_mask=None,  # 编码器的注意力掩码，默认为None
        encoder_decoder_position_bias=None,  # 编码器到解码器的位置偏置，默认为None
        layer_head_mask=None,  # 层头掩码，默认为None
        cross_attn_layer_head_mask=None,  # 跨注意力头的层头掩码，默认为None
        past_key_value=None,  # 过去的键值，默认为None
        use_cache=False,  # 是否使用缓存，默认为False
        output_attentions=False,  # 是否输出注意力，默认为False
        return_dict=True,  # 是否返回字典形式的结果，默认为True
class T5ClassificationHead(nn.Module):
    """用于句子级分类任务的头部。"""

    def __init__(self, config: T5Config):
        super().__init__()
        self.dense = nn.Linear(config.d_model, config.d_model)  # 用于变换隐藏状态维度的全连接层
        self.dropout = nn.Dropout(p=config.classifier_dropout)  # 用于在训练过程中随机关闭隐藏状态的一部分单元
        self.out_proj = nn.Linear(config.d_model, config.num_labels)  # 输出层，用于将隐藏状态映射到标签空间

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dropout(hidden_states)  # 在训练过程中随机关闭隐藏状态的一部分单元
        hidden_states = self.dense(hidden_states)  # 使用全连接层进行隐藏状态变换
        hidden_states = torch.tanh(hidden_states)  # 对隐藏状态进行双曲正切激活
        hidden_states = self.dropout(hidden_states)  # 在训练过程中随机关闭隐藏状态的一部分单元
        hidden_states = self.out_proj(hidden_states)  # 输出层映射到标签空间
        return hidden_states  # 返回隐藏状态


class T5PreTrainedModel(PreTrainedModel):
    """
    用于处理权重初始化和下载以及加载预训练模型的抽象类。
    """

    config_class = T5Config  # T5 模型的配置类
    load_tf_weights = load_tf_weights_in_t5  # 加载 TensorFlow 权重的方法
    base_model_prefix = "transformer"  # 基础模型的前缀
    is_parallelizable = True  # 模型是否可并行
    supports_gradient_checkpointing = True  # 是否支持梯度检查点
    _no_split_modules = ["T5Block"]  # 不需拆分的模块
    _keep_in_fp32_modules = ["wo"]  # 需要保持在 fp32 中的模块

    @property
    def dummy_inputs(self):
        input_ids = torch.tensor(DUMMY_INPUTS)  # 输入标识符
        input_mask = torch.tensor(DUMMY_MASK)  # 输入掩码
        dummy_inputs = {
            "decoder_input_ids": input_ids,  # 解码器输入标识符
            "input_ids": input_ids,  # 输入标识符
            "decoder_attention_mask": input_mask,  # 解码器注意力掩码
        }
        return dummy_inputs  # 返回虚拟输入

    def _shift_right(self, input_ids):
        decoder_start_token_id = self.config.decoder_start_token_id  # 解码器起始标记的标识符
        pad_token_id = self.config.pad_token_id  # 填充标记的标识符

        if decoder_start_token_id is None:
            raise ValueError(
                "self.model.config.decoder_start_token_id has to be defined. In T5 it is usually set to the pad_token_id. "
                "See T5 docs for more information."
            )

        # 向右移动输入
        if is_torch_fx_proxy(input_ids):
            # 在代理��不支持原生项目分配
            shifted_input_ids = torch.full(input_ids.shape[:-1] + (1,), decoder_start_token_id)
            shifted_input_ids = torch.cat([shifted_input_ids, input_ids[..., :-1]], dim=-1)
        else:
            shifted_input_ids = input_ids.new_zeros(input_ids.shape)
            shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
            shifted_input_ids[..., 0] = decoder_start_token_id

        if pad_token_id is None:
            raise ValueError("self.model.config.pad_token_id has to be defined.")
        # 用填充标记替换标签中的可能的 -100 值
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

        return shifted_input_ids
    # 初始化 T5Stack 类的实例
    def __init__(self, config, embed_tokens=None):
        # 调用父类的初始化方法
        super().__init__(config)

        # 设置嵌入标记和是否为解码器
        self.embed_tokens = embed_tokens
        self.is_decoder = config.is_decoder

        # 创建 T5Block 模块列表
        self.block = nn.ModuleList(
            [T5Block(config, has_relative_attention_bias=bool(i == 0)) for i in range(config.num_layers)]
        )
        # 初始化最终层归一化和丢弃
        self.final_layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

        # 初始化权重并进行最终处理
        self.post_init()
        # 模型并行
        self.model_parallel = False
        self.device_map = None
        self.gradient_checkpointing = False

    # 并行化方法
    @add_start_docstrings(PARALLELIZE_DOCSTRING)
    def parallelize(self, device_map=None):
        # 发出警告
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
        assert_device_map(self.device_map, len(self.block))
        self.model_parallel = True
        self.first_device = "cpu" if "cpu" in self.device_map.keys() else "cuda:" + str(min(self.device_map.keys()))
        self.last_device = "cuda:" + str(max(self.device_map.keys()))
        # 加载到设备上
        for k, v in self.device_map.items():
            for layer in v:
                cuda_device = "cuda:" + str(k)
                self.block[layer] = self.block[layer].to(cuda_device)

        # 设置嵌入标记到第一个层
        self.embed_tokens = self.embed_tokens.to(self.first_device)
        # 设置最终层归一化到最后一个设备
        self.final_layer_norm = self.final_layer_norm.to(self.last_device)

    # 反并行化方法
    @add_start_docstrings(DEPARALLELIZE_DOCSTRING)
    def deparallelize(self):
        # 发出警告
        warnings.warn(
            "Like `parallelize`, `deparallelize` is deprecated and will be removed in v5 of Transformers.",
            FutureWarning,
        )
        self.model_parallel = False
        self.device_map = None
        self.first_device = "cpu"
        self.last_device = "cpu"
        for i in range(len(self.block)):
            self.block[i] = self.block[i].to("cpu")
        self.embed_tokens = self.embed_tokens.to("cpu")
        self.final_layer_norm = self.final_layer_norm.to("cpu")
        torch.cuda.empty_cache()

    # 获取输入嵌入
    def get_input_embeddings(self):
        return self.embed_tokens

    # 设置输入嵌入
    def set_input_embeddings(self, new_embeddings):
        self.embed_tokens = new_embeddings
    # 定义 Transformer 模型的前向传播方法，接受一系列输入参数
    def forward(
        self,
        input_ids=None,  # 输入序列的 token IDs
        attention_mask=None,  # 注意力掩码，指示模型在哪些位置进行注意力计算
        encoder_hidden_states=None,  # 编码器的隐藏状态
        encoder_attention_mask=None,  # 编码器的注意力掩码，指示编码器在哪些位置进行注意力计算
        inputs_embeds=None,  # 输入嵌入向量
        head_mask=None,  # 头部掩码，用于遮蔽特定注意力头部
        cross_attn_head_mask=None,  # 跨注意力头部的掩码，用于遮蔽特定跨注意力头部
        past_key_values=None,  # 上下文缓存的键值对，用于加速生成
        use_cache=None,  # 是否使用缓存以加速生成
        output_attentions=None,  # 是否输出注意力权重
        output_hidden_states=None,  # 是否输出隐藏状态
        return_dict=None,  # 是否返回结果字典
```  
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

T5_INPUTS_DOCSTRING = r"""
"""

T5_ENCODER_INPUTS_DOCSTRING = r"""

"""
    # 输入参数:
    # input_ids (torch.LongTensor of shape (batch_size, sequence_length)):
    #   输入序列的词语索引，T5模型使用相对位置编码，所以可以在左右两侧进行填充。
    #   使用 AutoTokenizer 可以获取索引，参考 PreTrainedTokenizer.encode 和 PreTrainedTokenizer.__call__
    #   关于准备 input_ids 进行预训练的更多信息,可以参考 T5 Training 文档。
    # attention_mask (torch.FloatTensor of shape (batch_size, sequence_length), optional):
    #   注意力掩码,用于避免在填充标记上进行注意力计算。
    #   掩码值选择在 [0, 1] 范围内:
    #   - 1 表示 token 未被掩码
    #   - 0 表示 token 被掩码
    #   更多关于注意力掩码的信息,请参考 attention-mask 词汇表
    # head_mask (torch.FloatTensor of shape (num_heads,) or (num_layers, num_heads), optional):
    #   用于屏蔽自注意力模块中的特定头。
    #   掩码值选择在 [0, 1] 范围内:
    #   - 1 表示头未被掩码
    #   - 0 表示头被掩码
    # inputs_embeds (torch.FloatTensor of shape (batch_size, sequence_length, hidden_size), optional):
    #   直接传入嵌入表示,而不是 input_ids,这在您希望更好地控制 input_ids 索引到关联向量的转换时很有用。
    # output_attentions (bool, optional):
    #   是否返回所有注意力层的注意力张量。
    #   更多信息请参考 returned tensors 中的 attentions。
    # output_hidden_states (bool, optional):
    #   是否返回所有层的隐藏状态。
    #   更多信息请参考 returned tensors 中的 hidden_states。
    # return_dict (bool, optional):
    #   是否返回 ModelOutput 而不是普通元组。
"""
# FutureWarning 的警告信息: head_mask 被分成两个输入参数 - head_mask 和 decoder_head_mask
__HEAD_MASK_WARNING_MSG = """
输入参数 `head_mask` 被分成两个参数 `head_mask` 和 `decoder_head_mask`。当前，
`decoder_head_mask` 被设置为复制 `head_mask`，但此功能已被弃用，将在将来的版本中移除。
如果您现在不想使用任何 `decoder_head_mask`，请设置 `decoder_head_mask = torch.ones(num_layers,
num_heads)`。
"""


@add_start_docstrings(
    "The bare T5 Model transformer outputting raw hidden-states without any specific head on top.",
    T5_START_DOCSTRING,
)
# T5Model 类，继承自 T5PreTrainedModel
class T5Model(T5PreTrainedModel):
    # 在加载过程中需要忽略的键列表
    _keys_to_ignore_on_load_unexpected = [
        "decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.weight",
    ]
    # 绑定权重的键列表
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight"]

    # 初始化函数，接受一个 T5Config 对象作为参数
    def __init__(self, config: T5Config):
        # 调用父类构造函数
        super().__init__(config)
        # 共享的嵌入层，使用了词汇表大小和模型维度作为参数
        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        # 复制配置，设定编码器的配置
        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        # 创建编码器对象，传入复制后的配置和共享嵌入层
        self.encoder = T5Stack(encoder_config, self.shared)

        # 复制配置，设定解码器的配置
        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        # 创建解码器对象，传入复制后的配置和共享嵌入层
        self.decoder = T5Stack(decoder_config, self.shared)

        # 初始化权重并应用最终处理
        self.post_init()

        # 模型并行化
        self.model_parallel = False
        self.device_map = None

    # 并行化函数，接受一个设备映射参数
    @add_start_docstrings(PARALLELIZE_DOCSTRING)
    def parallelize(self, device_map=None):
        warnings.warn(
            "`T5Model.parallelize` is deprecated and will be removed in v5 of Transformers, you should load your model"
            " with `device_map='balanced'` in the call to `from_pretrained`. You can also provide your own"
            " `device_map` but it needs to be a dictionary module_name to device, so for instance {'encoder.block.0':"
            " 0, 'encoder.block.1': 1, ...}",
            FutureWarning,
        )
        # 设置设备映射
        self.device_map = (
            get_device_map(len(self.encoder.block), range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        # 断言设备映射的正确性
        assert_device_map(self.device_map, len(self.encoder.block))
        # 编码器并行化
        self.encoder.parallelize(self.device_map)
        # 解码器并行化
        self.decoder.parallelize(self.device_map)
        # 设置模型为并行模式
        self.model_parallel = True

    # 解除并行化函数
    @add_start_docstrings(DEPARALLELIZE_DOCSTRING)
"""
    # 发出警告，提示`deparallelize`过期，将在 Transformers 的 v5 版本中删除
    def deparallelize(self):
        warnings.warn(
            "Like `parallelize`, `deparallelize` is deprecated and will be removed in v5 of Transformers.",
            FutureWarning,
        )
        # 解除 encoder 的并行状态
        self.encoder.deparallelize()
        # 解除 decoder 的并行状态
        self.decoder.deparallelize()
        # 将 encoder 移动到 CPU
        self.encoder = self.encoder.to("cpu")
        # 将 decoder 移动到 CPU
        self.decoder = self.decoder.to("cpu")
        # 关闭模型的并行状态
        self.model_parallel = False
        # 清空 GPU 缓存
        self.device_map = None
        torch.cuda.empty_cache()

    # 获取输入的嵌入层
    def get_input_embeddings(self):
        return self.shared

    # 设置输入的嵌入层
    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    # 绑定权重
    def _tie_weights(self):
        if self.config.tie_word_embeddings:
            # 对 encoder 和 shared 进行权重绑定或克隆
            self._tie_or_clone_weights(self.encoder.embed_tokens, self.shared)
            # 对 decoder 和 shared 进行权重绑定或克隆
            self._tie_or_clone_weights(self.decoder.embed_tokens, self.shared)

    # 获取编码器
    def get_encoder(self):
        return self.encoder

    # 获取解码器
    def get_decoder(self):
        return self.decoder

    # 剪枝模型的头
    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            # 对指定层的注意力头进行剪枝
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(T5_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqModelOutput, config_class=_CONFIG_FOR_DOC)
    # 前向传播函数，包含各种输入参数
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
# 使用 T5 预训练模型作为基类，添加了一个顶部的语言建模头部
@add_start_docstrings("""T5 Model with a `language modeling` head on top.""", T5_START_DOCSTRING)
class T5ForConditionalGeneration(T5PreTrainedModel):
    # 加载时忽略的键列表
    _keys_to_ignore_on_load_unexpected = [
        "decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.weight",
    ]
    # 权重绑定的键列表
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight", "lm_head.weight"]

    # 初始化函数，接受一个 T5Config 类型的参数
    def __init__(self, config: T5Config):
        # 调用父类的初始化函数
        super().__init__(config)
        # 模型维度
        self.model_dim = config.d_model

        # 共享的嵌入层，将词汇表大小映射到模型维度
        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        # 复制编码器配置，将其用作编码器的配置，同时调整一些参数
        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        # 创建编码器实例
        self.encoder = T5Stack(encoder_config, self.shared)

        # 复制解码器配置，将其用作解码器的配置，同时调整一些参数
        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        # 创建解码器实例
        self.decoder = T5Stack(decoder_config, self.shared)

        # 语言建模头部，线性层将模型维度映射到词汇表大小
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # 初始化权重并进行最终处理
        self.post_init()

        # 模型并行化标志
        self.model_parallel = False
        # 设备映射
        self.device_map = None

    # 并行化函数，用于将模型的编码器和解码器部分分布到多个设备上
    @add_start_docstrings(PARALLELIZE_DOCSTRING)
    def parallelize(self, device_map=None):
        # 发出警告，该方法将在 v5 版本中删除
        warnings.warn(
            "`T5ForConditionalGeneration.parallelize` is deprecated and will be removed in v5 of Transformers, you"
            " should load your model with `device_map='balanced'` in the call to `from_pretrained`. You can also"
            " provide your own `device_map` but it needs to be a dictionary module_name to device, so for instance"
            " {'encoder.block.0': 0, 'encoder.block.1': 1, ...}",
            FutureWarning,
        )
        # 如果未提供设备映射，则使用均衡的映射
        self.device_map = (
            get_device_map(len(self.encoder.block), range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        # 检查设备映射是否正确
        assert_device_map(self.device_map, len(self.encoder.block))
        # 编码器部分并行化
        self.encoder.parallelize(self.device_map)
        # 解码器部分并行化
        self.decoder.parallelize(self.device_map)
        # 将语言建模头部移动到解码器的第一个设备上
        self.lm_head = self.lm_head.to(self.decoder.first_device)
        # 设置模型并行标志为真
        self.model_parallel = True

    # 反并行化函数，用于将并行化的模型恢复为单设备模型
    @add_start_docstrings(DEPARALLELIZE_DOCSTRING)
    def deparallelize(self):
        # 发出警告，该方法将在 v5 版本中删除
        warnings.warn(
            "Like `parallelize`, `deparallelize` is deprecated and will be removed in v5 of Transformers.",
            FutureWarning,
        )
        # 解码器部分反并行化
        self.encoder.deparallelize()
        # 解码器部分反并行化
        self.decoder.deparallelize()
        # 将编码器移动到 CPU 上
        self.encoder = self.encoder.to("cpu")
        # 将解码器移动到 CPU 上
        self.decoder = self.decoder.to("cpu")
        # 将语言建模头部移动到 CPU 上
        self.lm_head = self.lm_head.to("cpu")
        # 设置模型并行标志为假
        self.model_parallel = False
        # 清空 CUDA 缓存
        torch.cuda.empty_cache()

    # 获取输入嵌入层
    def get_input_embeddings(self):
        return self.shared
    # 设置模型的输入嵌入层
    def set_input_embeddings(self, new_embeddings):
        # 共享输入嵌入层
        self.shared = new_embeddings
        # 设置编码器的输入嵌入层
        self.encoder.set_input_embeddings(new_embeddings)
        # 设置解码器的输入嵌入层
        self.decoder.set_input_embeddings(new_embeddings)

    # 绑定权重（用于权重共享）
    def _tie_weights(self):
        # 如果配置中要求词嵌入层共享
        if self.config.tie_word_embeddings:
            # 绑定编码器的词嵌入层与共享的输入嵌入层
            self._tie_or_clone_weights(self.encoder.embed_tokens, self.shared)
            # 绑定解码器的词嵌入层与共享的输入嵌入层
            self._tie_or_clone_weights(self.decoder.embed_tokens, self.shared)

    # 设置模型的输出嵌入层
    def set_output_embeddings(self, new_embeddings):
        # 设置语言模型头部的输出嵌入层
        self.lm_head = new_embeddings

    # 获取模型的输出嵌入层
    def get_output_embeddings(self):
        # 返回语言模型头部的输出嵌入层
        return self.lm_head

    # 获取编码器
    def get_encoder(self):
        # 返回编码器
        return self.encoder

    # 获取解码器
    def get_decoder(self):
        # 返回解码器
        return self.decoder

    # 前向传播函数，用于模型推理
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
        # forward函数的详细注释已在模型源代码中给出，这里不再重复

    # 生成模式下准备输入的函数
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
```py  
    # 如果在使用 past_key_values 的情况下，修剪 decoder_input_ids
    if past_key_values is not None:
        # 获取过去键值的长度
        past_length = past_key_values[0][0].shape[2]
    
        # 一些生成方法可能已经只传递了最后一个输入ID
        if input_ids.shape[1] > past_length:
            remove_prefix_length = past_length
        else:
            # 默认保留最后一个ID
            remove_prefix_length = input_ids.shape[1] - 1
    
        input_ids = input_ids[:, remove_prefix_length:]
    
    # 返回包含不同参数的字典
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
    
    # 从标签准备解码器输入ID
    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return self._shift_right(labels)
    
    # 重新排列缓存
    def _reorder_cache(self, past_key_values, beam_idx):
        # 如果 decoder past 不包含在输出中
        # 禁用了快速解码，无需重新排序
        if past_key_values is None:
            logger.warning("You might want to consider setting `use_cache=True` to speed up decoding")
            return past_key_values
    
        reordered_decoder_past = ()
        for layer_past_states in past_key_values:
            # 从每个层的过去状态中获取正确的批次索引
            # `past` 的批次维度在第二个位置
            reordered_layer_past_states = ()
            for layer_past_state in layer_past_states:
                # 需要为四个键/值状态中的每一个设置正确的 `past`
                reordered_layer_past_states = reordered_layer_past_states + (
                    layer_past_state.index_select(0, beam_idx.to(layer_past_state.device)),
                )
    
            # 检查重排序后的状态和原状态的形状是否匹配
            if reordered_layer_past_states[0].shape != layer_past_states[0].shape:
                raise ValueError(
                    f"reordered_layer_past_states[0] shape {reordered_layer_past_states[0].shape} and layer_past_states[0] shape {layer_past_states[0].shape} mismatched"
                )
            # 检查重排序后的状态和原状态的长度是否匹配
            if len(reordered_layer_past_states) != len(layer_past_states):
                raise ValueError(
                    f"length of reordered_layer_past_states {len(reordered_layer_past_states)} and length of layer_past_states {len(layer_past_states)} mismatched"
                )
    
            reordered_decoder_past = reordered_decoder_past + (reordered_layer_past_states,)
    
        return reordered_decoder_past
# 导入模块，添加文档字符串
@add_start_docstrings(
    "The bare T5 Model transformer outputting encoder's raw hidden-states without any specific head on top.",
    T5_START_DOCSTRING,
)
# T5 编码器模型类
class T5EncoderModel(T5PreTrainedModel):
    # 需要绑定权重的键名
    _tied_weights_keys = ["encoder.embed_tokens.weight"]
    # 加载时需要忽略的键名
    _keys_to_ignore_on_load_unexpected = [r"decoder"]

    # 初始化方法
    def __init__(self, config: T5Config):
        # 调用父类初始化方法
        super().__init__(config)
        # 共享的嵌入层
        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        # 复制配置对象并设置相关属性
        encoder_config = copy.deepcopy(config)
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        # 创建 T5 编码器
        self.encoder = T5Stack(encoder_config, self.shared)

        # 初始化权重并应用最终处理
        self.post_init()

        # 模型并行设置为 False
        self.model_parallel = False
        # 设备映射为 None

    # 并行化方法
    @add_start_docstrings(PARALLELIZE_DOCSTRING)
    def parallelize(self, device_map=None):
        # 发出未来已弃用警告
        warnings.warn(
            "`T5EncoderModel.parallelize` is deprecated and will be removed in v5 of Transformers, you should load"
            " your model with `device_map='balanced'` in the call to `from_pretrained`. You can also provide your own"
            " `device_map` but it needs to be a dictionary module_name to device, so for instance {'block.0': 0,"
            " 'block.1': 1, ...}",
            FutureWarning,
        )
        # 根据设备映射或默认映射返回设备映射字典
        self.device_map = (
            get_device_map(len(self.encoder.block), range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        # 断言设备映射合法
        assert_device_map(self.device_map, len(self.encoder.block))
        # 并行化编码器
        self.encoder.parallelize(self.device_map)
        # 模型并行设置为 True

    # 反并行化方法
    @add_start_docstrings(DEPARALLELIZE_DOCSTRING)
    def deparallelize(self):
        # 发出未来已弃用警告
        warnings.warn(
            "Like `parallelize`, `deparallelize` is deprecated and will be removed in v5 of Transformers.",
            FutureWarning,
        )
        # 反并行化编码器
        self.encoder.deparallelize()
        # 编码器转移到 CPU
        self.encoder = self.encoder.to("cpu")
        # 模型并行设置为 False
        self.model_parallel = False
        # 清空 CUDA 缓存
        torch.cuda.empty_cache()

    # 获取输入嵌入层
    def get_input_embeddings(self):
        return self.shared

    # 设置输入嵌入层
    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)

    # 绑定权重
    def _tie_weights(self):
        if self.config.tie_word_embeddings:
            self._tie_or_clone_weights(self.encoder.embed_tokens, self.shared)

    # 获取编码器
    def get_encoder(self):
        return self.encoder

    # 剪枝方法
    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.block[layer].layer[0].SelfAttention.prune_heads(heads)

    # 添加文档字符串到模型前向方法
    @add_start_docstrings_to_model_forward(T5_ENCODER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutput, config_class=_CONFIG_FOR_DOC)
    # 使用指定的函数装饰器替换返回文档字符串的参数，设置输出类型和配置类
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
        返回值说明:
    
        Example:
        示例:
    
        ```python
        >>> from transformers import AutoTokenizer, T5EncoderModel
    
        >>> tokenizer = AutoTokenizer.from_pretrained("t5-small")
        >>> model = T5EncoderModel.from_pretrained("t5-small")
        >>> input_ids = tokenizer(
        ...     "Studies have been shown that owning a dog is good for you", return_tensors="pt"
        ... ).input_ids  # Batch size 1
        >>> outputs = model(input_ids=input_ids)
        >>> last_hidden_states = outputs.last_hidden_state
        ```py"""
        # 如果 return_dict 不是 None，则使用其值；否则使用配置中的 use_return_dict 值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    
        # 使用编码器处理输入，获取编码器的输出
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
    
        # 返回编码器的输出
        return encoder_outputs
# 在 T5 模型的基础上增加一个序列分类的头部（在池化输出的顶部添加一个线性层），用于 GLUE 任务等
@add_start_docstrings(
    """
    T5 model with a sequence classification/head on top (a linear layer on top of the pooled output) e.g. for GLUE
    tasks.
    """,
    T5_START_DOCSTRING,
)
class T5ForSequenceClassification(T5PreTrainedModel):
    # 加载时忽略的键值列表
    _keys_to_ignore_on_load_unexpected = ["decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.weight"]
    # 绑定权重的键值列表
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight"]

    # 初始化方法
    def __init__(self, config: T5Config):
        # 调用父类的初始化方法
        super().__init__(config)
        # 创建 T5 模型
        self.transformer = T5Model(config)
        # 创建分类头部
        self.classification_head = T5ClassificationHead(config)

        # 初始化权重并应用最终处理
        self.post_init()

        self.model_parallel = False

    # 前向传播方法
    @add_start_docstrings_to_model_forward(T5_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqSequenceClassifierOutput, config_class=_CONFIG_FOR_DOC)
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


# 在 T5 模型的基础上增加一个用于抽取式问答任务（例如 SQuAD）的跨度分类头部（在隐藏状态输出的顶部添加线性层来计算 `span start logits` 和 `span end logits`）
@add_start_docstrings(
    """
    T5 Model with a span classification head on top for extractive question-answering tasks like SQuAD (linear layers
    on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    T5_START_DOCSTRING,
)
class T5ForQuestionAnswering(T5PreTrainedModel):
    # 加载时忽略的键值列表
    _keys_to_ignore_on_load_unexpected = ["decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.weight"]
    # 绑定权重的键值列表
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight"]
    # 初始化方法，接受一个T5Config类型的参数
    def __init__(self, config: T5Config):
        # 调用父类的初始化方法
        super().__init__(config)
        # 获取模型维度
        self.model_dim = config.d_model

        # 创建共享的嵌入层，将词汇表大小和模型维度传入
        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        # 复制编码器配置
        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        # 创建编码器
        self.encoder = T5Stack(encoder_config, self.shared)

        # 复制解码器配置
        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        # 创建解码器
        self.decoder = T5Stack(decoder_config, self.shared)

        # 设置标签数
        self.num_labels = config.num_labels
        # 创建一个全连接层，将隐藏层大小映射到标签数
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        # 初始化权重并应用最终处理
        self.post_init()

        # 设置模型并行为False
        self.model_parallel = False

    # 获取输入嵌入
    def get_input_embeddings(self):
        return self.shared

    # 设置输入嵌入
    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        # 更新编码器和解码器的输入嵌入
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    # 联结权重
    def _tie_weights(self):
        if self.config.tie_word_embeddings:
            # 如果设置了词嵌入联结，将编码器和解码器的词嵌入权重联结
            self._tie_or_clone_weights(self.encoder.embed_tokens, self.shared)
            self._tie_or_clone_weights(self.decoder.embed_tokens, self.shared)

    # 获取编码器
    def get_encoder(self):
        return self.encoder

    # 获取解码器
    def get_decoder(self):
        return self.decoder

    # 前向传播函数，接受多个可选的输入和输出
    @add_start_docstrings_to_model_forward(T5_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqQuestionAnsweringModelOutput, config_class=_CONFIG_FOR_DOC)
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
        start_positions: Optional[torch.LongTensor] = None,
        end_positions: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
```