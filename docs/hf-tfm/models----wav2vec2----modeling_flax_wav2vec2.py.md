# `.\models\wav2vec2\modeling_flax_wav2vec2.py`

```
# coding=utf-8
# Copyright 2021 The Fairseq Authors and the HuggingFace Inc. team. All rights reserved.
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
Flax Wav2Vec2 model.
"""

# 导入必要的模块和库
from functools import partial  # 导入 partial 函数，用于创建偏函数
from typing import Optional, Tuple, Union  # 导入类型提示所需的类型

import flax  # 导入 Flax 模块
import flax.linen as nn  # 导入 Flax 的线性层模块
import jax  # 导入 JAX 模块
import jax.numpy as jnp  # 导入 JAX 的 NumPy 接口
import numpy as np  # 导入 NumPy 库
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze  # 导入 Flax 的 FrozenDict 相关函数
from flax.linen.attention import dot_product_attention_weights  # 导入注意力权重计算函数
from flax.traverse_util import flatten_dict, unflatten_dict  # 导入字典扁平化和反扁平化函数
from jax import lax  # 导入 JAX 的 lax 模块

# 导入相关输出、工具类和配置
from ...modeling_flax_outputs import FlaxBaseModelOutput, FlaxCausalLMOutput  # 导入输出类
from ...modeling_flax_utils import (  # 导入工具函数和基类
    ACT2FN,
    FlaxPreTrainedModel,
    append_replace_return_docstrings,
    overwrite_call_docstring,
)
from ...utils import ModelOutput, add_start_docstrings, add_start_docstrings_to_model_forward, logging  # 导入实用工具和日志模块
from .configuration_wav2vec2 import Wav2Vec2Config  # 导入 Wav2Vec2 的配置类

# 获取日志记录器
logger = logging.get_logger(__name__)
    # 定义变量 `last_hidden_state`，用于存储 JAX NumPy 数组（jnp.ndarray），初始值为 None
    last_hidden_state: jnp.ndarray = None
    # 定义变量 `extract_features`，用于存储 JAX NumPy 数组（jnp.ndarray），初始值为 None
    extract_features: jnp.ndarray = None
    # 定义变量 `hidden_states`，用于存储一个元组，其中元素是 JAX NumPy 数组（jnp.ndarray），可选类型（Optional）表示可以为 None
    hidden_states: Optional[Tuple[jnp.ndarray]] = None
    # 定义变量 `attentions`，用于存储一个元组，其中元素是 JAX NumPy 数组（jnp.ndarray），可选类型（Optional）表示可以为 None
    attentions: Optional[Tuple[jnp.ndarray]] = None
# 定义一个数据类，用于存储 FlaxWav2Vec2 模型预训练的输出结果，继承自 ModelOutput
@flax.struct.dataclass
class FlaxWav2Vec2ForPreTrainingOutput(ModelOutput):
    """
    Output type of [`FlaxWav2Vec2ForPreTrainingOutput`], with potential hidden states and attentions.

    Args:
        loss (*optional*, returned when model is in train mode, `jnp.ndarray` of shape `(1,)`):
            Total loss as the sum of the contrastive loss (L_m) and the diversity loss (L_d) as stated in the [official
            paper](https://arxiv.org/pdf/2006.11477.pdf) . (classification) loss.
        projected_states (`jnp.ndarray` of shape `(batch_size, sequence_length, config.proj_codevector_dim)`):
            Hidden-states of the model projected to *config.proj_codevector_dim* that can be used to predict the masked
            projected quantized states.
        projected_quantized_states (`jnp.ndarray` of shape `(batch_size, sequence_length, config.proj_codevector_dim)`):
            Quantized extracted feature vectors projected to *config.proj_codevector_dim* representing the positive
            target vectors for contrastive loss.
        hidden_states (`tuple(jnp.ndarray)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `jnp.ndarray` (one for the output of the embeddings + one for the output of each layer) of shape
            `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(jnp.ndarray)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `jnp.ndarray` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    # 定义属性：模型预测的状态向量，形状为 jnp.ndarray 或者 None
    projected_states: jnp.ndarray = None
    # 定义属性：量化后的状态向量，形状为 jnp.ndarray 或者 None
    projected_quantized_states: jnp.ndarray = None
    # 定义属性：码本的困惑度，形状为 jnp.ndarray 或者 None
    codevector_perplexity: jnp.ndarray = None
    # 定义属性：隐藏状态的元组，包含 jnp.ndarray 或者 None
    hidden_states: Optional[Tuple[jnp.ndarray]] = None
    # 定义属性：注意力的元组，包含 jnp.ndarray 或者 None
    attentions: Optional[Tuple[jnp.ndarray]] = None


# 定义一个函数，用于计算给定形状的随机掩码段落，用于实现 SpecAugment 数据增强方法，参考了 ASR 领域的论文
def _compute_mask_indices(
    shape: Tuple[int, int],
    mask_prob: float,
    mask_length: int,
    attention_mask: Optional[np.ndarray] = None,
    min_masks: int = 0,
) -> np.ndarray:
    """
    Computes random mask spans for a given shape. Used to implement [SpecAugment: A Simple Data Augmentation Method for
    ASR](https://arxiv.org/abs/1904.08779). Note that this method is not optimized to run on TPU and should be run on
    CPU as part of the preprocessing during training.
    """
    Args:
        shape: the shape for which to compute masks.
            should be of size 2 where first element is batch size and 2nd is timesteps
        mask_prob:
            probability for each token to be chosen as start of the span to be masked. this will be multiplied by
            number of timesteps divided by length of mask span to mask approximately this percentage of all elements.
            however due to overlaps, the actual number will be smaller (unless no_overlap is True)
        mask_length: size of the mask
        min_masks: minimum number of masked spans

    """
    # 解包形状参数，batch_size 为批次大小，sequence_length 为时间步长
    batch_size, sequence_length = shape

    # 如果 mask_length 小于 1，则引发值错误
    if mask_length < 1:
        raise ValueError("`mask_length` has to be bigger than 0.")

    # 如果 mask_length 大于 sequence_length，则引发值错误
    if mask_length > sequence_length:
        raise ValueError(
            f"`mask_length` has to be smaller than `sequence_length`, but got `mask_length`: {mask_length} and"
            f" `sequence_length`: {sequence_length}`"
        )

    # 计算每批次中需要掩蔽的区间数目
    num_masked_spans = int(mask_prob * sequence_length / mask_length + np.random.rand(1).item())
    # 确保 num_masked_spans 不小于 min_masks
    num_masked_spans = max(num_masked_spans, min_masks)

    # 确保掩蔽的索引数不超过 sequence_length
    if num_masked_spans * mask_length > sequence_length:
        num_masked_spans = sequence_length // mask_length

    # 初始化一个形状为 (batch_size, sequence_length) 的布尔类型的掩蔽数组
    spec_aug_mask = np.zeros((batch_size, sequence_length), dtype=bool)

    # 随机生成要掩蔽的起始索引
    spec_aug_mask_idxs = np.array(
        [
            np.random.choice(np.arange(sequence_length - (mask_length - 1)), num_masked_spans, replace=False)
            for _ in range(batch_size)
        ]
    )

    # 将掩蔽的索引扩展为掩蔽的区间
    spec_aug_mask_idxs = np.broadcast_to(spec_aug_mask_idxs[:, :, None], (batch_size, num_masked_spans, mask_length))
    spec_aug_mask_idxs = spec_aug_mask_idxs.reshape(batch_size, num_masked_spans * mask_length)

    # 创建一个偏移数组以便扩展掩蔽的区间
    offsets = np.arange(mask_length)[None, None, :]
    offsets = np.broadcast_to(offsets, (batch_size, num_masked_spans, mask_length)).reshape(
        batch_size, num_masked_spans * mask_length
    )
    spec_aug_mask_idxs = spec_aug_mask_idxs + offsets

    # 在掩蔽数组中填充掩蔽的索引
    np.put_along_axis(spec_aug_mask, spec_aug_mask_idxs, 1, -1)

    # 如果存在 attention_mask，则确保填充的输入 ID 不能被掩蔽
    if attention_mask is not None:
        spec_aug_mask = np.where(attention_mask, spec_aug_mask, False)

    # 返回生成的掩蔽数组
    return spec_aug_mask
def _sample_negative_indices(features_shape: Tuple, num_negatives: int, attention_mask: Optional[np.ndarray] = None):
    """
    Sample `num_negatives` vectors from feature vectors.
    """
    # 解析输入参数的形状信息
    batch_size, sequence_length, hidden_size = features_shape

    # 检查序列长度是否小于等于1，如果是则引发异常
    if sequence_length <= 1:
        raise ValueError(
            "`features should have `sequence_length` > 1, but are of shape "
            f"(batch_size, sequence_length, hidden_size) = ({batch_size, sequence_length, hidden_size})."
        )

    # 从同一个语句中随机选择 `num_negatives` 个向量索引
    sampled_negative_indices = []
    for batch_idx in range(batch_size):
        # 根据注意力掩码确定可用索引的上限，或者使用序列长度的上限
        high = attention_mask[batch_idx].sum() - 1 if attention_mask is not None else sequence_length - 1
        # 随机抽样索引，数量为 `num_negatives * sequence_length`
        sampled_indices_slice = np.random.randint(0, high, size=(num_negatives * sequence_length,))
        sampled_negative_indices.append(sampled_indices_slice)

    sampled_negative_indices = np.asarray(sampled_negative_indices, dtype=np.int32)

    # 生成正向量的索引，将其重复 `num_negatives` 次
    feature_indices = np.broadcast_to(np.arange(sequence_length)[:, None], (sequence_length, num_negatives)).flatten()

    # 避免抽样到相同的正向量索引，同时保持均匀分布
    sampled_negative_indices[sampled_negative_indices >= feature_indices] += 1

    # 调整索引以匹配批次大小
    for batch_idx in range(1, batch_size):
        sampled_negative_indices[batch_idx] += batch_idx * sequence_length

    return sampled_negative_indices


WAV_2_VEC_2_START_DOCSTRING = r"""
    Wav2Vec2 was proposed in [wav2vec 2.0: A Framework for Self-Supervised Learning of Speech
    Representations](https://arxiv.org/abs/2006.11477) by Alexei Baevski, Henry Zhou, Abdelrahman Mohamed, Michael
    Auli.

    This model inherits from [`FlaxPreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a Flax Linen
    [flax.nn.Module](https://flax.readthedocs.io/en/latest/_autosummary/flax.nn.module.html) subclass. Use it as a
    regular Flax Module and refer to the Flax documentation for all matter related to general usage and behavior.

    Finally, this model supports inherent JAX features such as:

    - [Just-In-Time (JIT) compilation](https://jax.readthedocs.io/en/latest/jax.html#just-in-time-compilation-jit)
    - [Automatic Differentiation](https://jax.readthedocs.io/en/latest/jax.html#automatic-differentiation)
    - [Vectorization](https://jax.readthedocs.io/en/latest/jax.html#vectorization-vmap)
    - [Parallelization](https://jax.readthedocs.io/en/latest/jax.html#parallelization-pmap)


"""
    Parameters:
        config ([`Wav2Vec2Config`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~FlaxPreTrainedModel.from_pretrained`] method to load the model weights.
        dtype (`jax.numpy.dtype`, *optional*, defaults to `jax.numpy.float32`):
            The data type of the computation. Can be one of `jax.numpy.float32`, `jax.numpy.float16` (on GPUs) and
            `jax.numpy.bfloat16` (on TPUs).
            
            This specifies the data type used for computations, allowing for mixed-precision training or
            half-precision inference on GPUs or TPUs. If specified, all computations within the model will be
            performed with the specified `dtype`.

            **Note that this setting affects only the computation dtype and not the dtype of model parameters.**

            To change the dtype of model parameters, refer to [`~FlaxPreTrainedModel.to_fp16`] and
            [`~FlaxPreTrainedModel.to_bf16`].
"""
定义一个类 `FlaxWav2Vec2LayerNormConvLayer`，继承自 `nn.Module`，用于实现基于 Flax 的 Wav2Vec2 模型的一层。
"""
class FlaxWav2Vec2LayerNormConvLayer(nn.Module):
    # 设置类属性 `config` 为 `Wav2Vec2Config` 类型，用于配置模型参数
    config: Wav2Vec2Config
    # 设置类属性 `layer_id` 为整数，表示当前层的标识，默认为 0
    layer_id: int = 0
    # 设置类属性 `dtype` 为 `jnp.float32`，表示数据类型为 32 位浮点数
    dtype: jnp.dtype = jnp.float32
    # 设置函数，用于初始化网络层参数
    def setup(self):
        # 如果当前层不是第一层，设置输入卷积维度为指定的卷积维度列表中对应层的值，否则设为1
        self.in_conv_dim = self.config.conv_dim[self.layer_id] if self.layer_id > 0 else 1
        # 设置输出卷积维度为指定的卷积维度列表中对应层的值
        self.out_conv_dim = self.config.conv_dim[self.layer_id]

        # 初始化卷积层
        self.conv = nn.Conv(
            features=self.config.conv_dim[self.layer_id],  # 卷积层输出特征维度
            kernel_size=(self.config.conv_kernel[self.layer_id],),  # 卷积核大小
            strides=(self.config.conv_stride[self.layer_id],),  # 卷积步长
            use_bias=self.config.conv_bias,  # 是否使用偏置
            kernel_init=jax.nn.initializers.he_normal(),  # 卷积核初始化方法
            padding="VALID",  # 卷积填充方式
            dtype=self.dtype,  # 数据类型
        )
        # 初始化层归一化层
        self.layer_norm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)
        # 初始化激活函数，根据配置选择相应的激活函数
        self.activation = ACT2FN[self.config.feat_extract_activation]

    # 定义调用函数，用于前向传播计算
    def __call__(self, hidden_states):
        # 卷积操作，计算特征提取后的隐藏状态
        hidden_states = self.conv(hidden_states)
        # 层归一化操作，对卷积输出进行归一化处理
        hidden_states = self.layer_norm(hidden_states)
        # 激活函数操作，对归一化后的输出应用激活函数
        hidden_states = self.activation(hidden_states)
        # 返回处理后的隐藏状态
        return hidden_states
# 定义一个自定义的 Flax 模块，用于卷积操作并包含权重归一化
class FlaxConvWithWeightNorm(nn.Module):
    # 配置信息，指定为 Wav2Vec2Config 类型
    config: Wav2Vec2Config
    # 数据类型，默认为 jnp.float32
    dtype: jnp.dtype = jnp.float32

    # 模块设置方法，用于初始化模块的各个部分
    def setup(self):
        # 创建卷积层，设置特征数为 hidden_size，卷积核大小为 num_conv_pos_embeddings
        self.conv = nn.Conv(
            features=self.config.hidden_size,
            kernel_size=(self.config.num_conv_pos_embeddings,),
            kernel_init=jax.nn.initializers.he_normal(),
            padding="VALID",
            feature_group_count=self.config.num_conv_pos_embedding_groups,
            dtype=self.dtype,
        )
        # 定义权重形状，与卷积层特征数及分组数有关
        weight_shape = (
            self.conv.features,
            self.conv.features // self.conv.feature_group_count,
            self.conv.kernel_size[0],
        )
        # 初始化并定义权重 v 作为模型参数，使用 he_normal 初始化器
        self.weight_v = self.param("weight_v", jax.nn.initializers.he_normal(), weight_shape)
        # 计算权重 v 的 L2 范数，并初始化权重 g 作为模型参数
        self.weight_g = self.param("weight_g", lambda _: jnp.linalg.norm(self.weight_v, axis=(0, 1))[None, None, :])
        # 初始化偏置参数，特征数与卷积层相同
        self.bias = self.param("bias", jax.nn.initializers.zeros, (self.conv.features,))
        # 计算用于填充输入的前置填充数
        self.prev_padding = self.conv.kernel_size[0] // 2

    # 内部方法，用于获取归一化后的权重
    def _get_normed_weights(self):
        # 计算权重 v 的归一化形式
        weight_v_norm = jnp.linalg.norm(self.weight_v, axis=(0, 1))[None, None, :]
        normed_weight_v = jnp.divide(self.weight_v, weight_v_norm)
        # 计算归一化后的卷积核
        normed_kernel = jnp.multiply(normed_weight_v, self.weight_g)
        return normed_kernel

    # 模块的调用方法，执行卷积操作并返回结果
    def __call__(self, hidden_states):
        # 获取归一化后的卷积核
        kernel = self._get_normed_weights()
        # 对输入进行前置填充，保证卷积输出尺寸与输入相同
        hidden_states = jnp.pad(hidden_states, ((0, 0), (self.prev_padding, self.prev_padding), (0, 0)))
        # 应用卷积操作到输入上，使用归一化后的卷积核和偏置
        hidden_states = self.conv.apply({"params": {"kernel": kernel.T, "bias": self.bias}}, hidden_states)
        return hidden_states


# 定义一个 Flax 模块，用于处理位置卷积嵌入
class FlaxWav2Vec2PositionalConvEmbedding(nn.Module):
    # 配置信息，指定为 Wav2Vec2Config 类型
    config: Wav2Vec2Config
    # 数据类型，默认为 jnp.float32
    dtype: jnp.dtype = jnp.float32

    # 模块设置方法，用于初始化模块的各个部分
    def setup(self):
        # 创建包含权重归一化的卷积层模块
        self.conv = FlaxConvWithWeightNorm(self.config, dtype=self.dtype)
        # 设置激活函数为配置文件中指定的函数
        self.activation = ACT2FN[self.config.feat_extract_activation]
        # 根据卷积核大小决定需要移除的填充数量
        self.num_pad_remove = 1 if self.config.num_conv_pos_embeddings % 2 == 0 else 0

    # 模块的调用方法，执行位置卷积嵌入操作并返回结果
    def __call__(self, hidden_states):
        # 调整输入张量的维度顺序
        hidden_states = hidden_states.transpose((0, 1, 2))
        # 应用包含权重归一化的卷积操作到输入上
        hidden_states = self.conv(hidden_states)
        # 根据需要移除的填充数量截取卷积输出
        if self.num_pad_remove > 0:
            hidden_states = hidden_states[:, : -self.num_pad_remove, :]
        # 应用激活函数到卷积输出上
        hidden_states = self.activation(hidden_states)
        # 恢复张量的原始维度顺序并返回结果
        hidden_states = hidden_states.transpose((0, 1, 2))
        return hidden_states


# 定义一个 Flax 模块，用于包含一系列卷积层的集合
class FlaxConvLayersCollection(nn.Module):
    # 配置信息，指定为 Wav2Vec2Config 类型
    config: Wav2Vec2Config
    # 数据类型，默认为 jnp.float32
    dtype: jnp.dtype = jnp.float32
    # 初始化方法，用于设置对象的初始状态
    def setup(self):
        # 如果配置要求特征提取的归一化方式为 "layer"
        if self.config.feat_extract_norm == "layer":
            # 创建一系列 FlaxWav2Vec2LayerNormConvLayer 对象作为 self.layers 列表的元素，
            # 每个对象对应一个特征提取层
            self.layers = [
                FlaxWav2Vec2LayerNormConvLayer(self.config, layer_id=i, name=str(i), dtype=self.dtype)
                for i in range(self.config.num_feat_extract_layers)
            ]
        # 如果配置要求特征提取的归一化方式为 "group"，暂时不支持这种方式
        elif self.config.feat_extract_norm == "group":
            # 抛出 NotImplementedError 异常，提醒暂时只支持 "layer" 形式的特征提取归一化
            raise NotImplementedError("At the moment only ``config.feat_extact_norm == 'layer'`` is supported")
        # 如果配置的特征提取归一化方式既不是 "layer" 也不是 "group"，则抛出 ValueError 异常
        else:
            # 抛出 ValueError 异常，指明配置中的 feat_extract_norm 值不合法
            raise ValueError(
                f"`config.feat_extract_norm` is {self.config.feat_extract_norm}, but has to be one of ['group',"
                " 'layer']"
            )

    # 对象被调用时执行的方法，用于处理输入的隐藏状态数据
    def __call__(self, hidden_states):
        # 遍历 self.layers 中的每个 conv_layer，依次对 hidden_states 进行处理
        for i, conv_layer in enumerate(self.layers):
            hidden_states = conv_layer(hidden_states)  # 调用 conv_layer 对象处理 hidden_states
        # 返回处理后的 hidden_states
        return hidden_states
class FlaxWav2Vec2FeatureEncoder(nn.Module):
    """从原始音频波形中构建特征"""

    config: Wav2Vec2Config  # 引用Wav2Vec2Config配置对象
    dtype: jnp.dtype = jnp.float32  # 计算时使用的数据类型，默认为单精度浮点数

    def setup(self):
        self.conv_layers = FlaxConvLayersCollection(self.config, dtype=self.dtype)
        # 初始化卷积层集合，使用配置对象和指定数据类型

    def __call__(self, input_values, freeze_feature_encoder=False):
        hidden_states = input_values[:, :, None]
        # 在最后添加一个维度，将形状从[batch_size, seq_len]变为[batch_size, seq_len, 1]
        hidden_states = self.conv_layers(hidden_states)
        # 经过卷积层处理，处理后形状为[batch_size, seq_len, hidden_size]
        if freeze_feature_encoder:
            hidden_states = jax.lax.stop_gradient(hidden_states)
            # 如果需要冻结特征编码器，则停止梯度传播
        return hidden_states


class FlaxWav2Vec2FeatureProjection(nn.Module):
    config: Wav2Vec2Config  # 引用Wav2Vec2Config配置对象
    dtype: jnp.dtype = jnp.float32  # 计算时使用的数据类型，默认为单精度浮点数

    def setup(self):
        self.layer_norm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)
        # 初始化层归一化，使用指定的epsilon值和数据类型
        self.projection = nn.Dense(
            self.config.hidden_size,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            dtype=self.dtype,
        )
        # 初始化全连接层，设置隐藏大小、权重初始化方法和数据类型
        self.dropout = nn.Dropout(rate=self.config.feat_proj_dropout)
        # 初始化dropout层，设置丢弃率为配置中的特征投影dropout率

    def __call__(self, hidden_states, deterministic=True):
        norm_hidden_states = self.layer_norm(hidden_states)
        # 对隐藏状态进行层归一化处理
        hidden_states = self.projection(norm_hidden_states)
        # 应用投影层
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        # 应用dropout，如果确定性为True，则使用确定性dropout
        return hidden_states, norm_hidden_states


class FlaxWav2Vec2Attention(nn.Module):
    config: Wav2Vec2Config  # 引用Wav2Vec2Config配置对象
    embed_dim: int  # 嵌入维度
    num_heads: int  # 头的数量
    dropout: float = 0.0  # dropout率，默认为0.0
    bias: bool = True  # 是否使用偏置，默认为True
    dtype: jnp.dtype = jnp.float32  # 计算时使用的数据类型，默认为单精度浮点数

    def setup(self) -> None:
        self.head_dim = self.embed_dim // self.num_heads
        # 计算每个头的维度
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
            # 检查embed_dim必须能够被num_heads整除的条件，否则引发错误

        dense = partial(
            nn.Dense,
            self.embed_dim,
            use_bias=self.bias,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
        )
        # 创建一个部分应用了参数的全连接层函数

        self.q_proj, self.k_proj, self.v_proj = dense(), dense(), dense()
        # 使用dense函数初始化查询、键、值投影层
        self.out_proj = dense()
        # 使用dense函数初始化输出投影层

        self.dropout_layer = nn.Dropout(rate=self.dropout)
        # 初始化dropout层，设置丢弃率为配置中的dropout率

    def _split_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.num_heads, self.head_dim))
        # 将隐藏状态切分成多个头

    def _merge_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.embed_dim,))
        # 合并多个头的隐藏状态

    def __call__(
        self,
        hidden_states: jnp.ndarray,
        key_value_states: Optional[jnp.ndarray] = None,
        attention_mask: Optional[jnp.ndarray] = None,
        deterministic: bool = True,
        # 定义Attention层的调用方式，包括隐藏状态、键值状态、注意力掩码和确定性
    ) -> Tuple[jnp.ndarray]:
        """Input shape: Batch x Time x Channel"""
        
        # 获取查询投影
        query_states = self.q_proj(hidden_states)

        # 获取键投影和值投影
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # 将查询投影、键投影和值投影按照头的数量进行分割
        query_states = self._split_heads(query_states)
        key_states = self._split_heads(key_states)
        value_states = self._split_heads(value_states)

        # 如果存在注意力掩码，则扩展维度以匹配张量形状
        if attention_mask is not None:
            attention_mask = jnp.expand_dims(attention_mask, axis=(-3, -2))

        # 将布尔类型的注意力掩码转换为注意力偏置
        if attention_mask is not None:
            # 注意力掩码转换为注意力偏置
            attention_bias = lax.select(
                attention_mask > 0,
                jnp.full(attention_mask.shape, 0.0).astype(self.dtype),
                jnp.full(attention_mask.shape, jnp.finfo(self.dtype).min).astype(self.dtype),
            )
        else:
            attention_bias = None

        # 如果不是确定性计算且具有非零的 dropout 率，则创建 dropout 随机数生成器
        dropout_rng = None
        if not deterministic and self.dropout > 0.0:
            dropout_rng = self.make_rng("dropout")

        # 计算注意力权重
        attn_weights = dot_product_attention_weights(
            query_states,
            key_states,
            bias=attention_bias,
            dropout_rng=dropout_rng,
            dropout_rate=self.dropout,
            broadcast_dropout=True,
            deterministic=deterministic,
            dtype=self.dtype,
            precision=None,
        )

        # 计算注意力输出，使用 einsum 实现批量矩阵乘法
        attn_output = jnp.einsum("...hqk,...khd->...qhd", attn_weights, value_states)
        attn_output = self._merge_heads(attn_output)  # 合并注意力头
        attn_output = self.out_proj(attn_output)  # 输出投影

        return attn_output, attn_weights
# 定义一个名为 FlaxWav2Vec2FeedForward 的自定义神经网络模块，继承自 nn.Module
class FlaxWav2Vec2FeedForward(nn.Module):
    # 类属性：配置信息，类型为 Wav2Vec2Config
    config: Wav2Vec2Config
    # 类属性：数据类型，默认为 jnp.float32
    dtype: jnp.dtype = jnp.float32

    # 初始化方法，设置网络结构
    def setup(self):
        # 定义中间层的 dropout 操作，使用配置中的激活函数的 dropout 率
        self.intermediate_dropout = nn.Dropout(rate=self.config.activation_dropout)

        # 定义中间层的全连接层，输入大小为配置中的 intermediate_size
        # 初始化方式为正态分布，范围为配置中的 initializer_range
        self.intermediate_dense = nn.Dense(
            self.config.intermediate_size,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            dtype=self.dtype,
        )

        # 根据配置选择激活函数，如果是字符串则从预定义的映射中获取，否则直接使用配置中的激活函数
        if isinstance(self.config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[self.config.hidden_act]
        else:
            self.intermediate_act_fn = self.config.hidden_act

        # 定义输出层的全连接层，输出大小为配置中的 hidden_size
        # 初始化方式为正态分布，范围为配置中的 initializer_range
        self.output_dense = nn.Dense(
            self.config.hidden_size,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            dtype=self.dtype,
        )

        # 定义输出层的 dropout 操作，使用配置中的隐藏层 dropout 率
        self.output_dropout = nn.Dropout(rate=self.config.hidden_dropout)

    # 前向传播方法，接收隐藏状态和是否确定性的标志，返回最终的隐藏状态
    def __call__(self, hidden_states, deterministic=True):
        # 中间层的全连接操作
        hidden_states = self.intermediate_dense(hidden_states)
        # 中间层的激活函数
        hidden_states = self.intermediate_act_fn(hidden_states)
        # 中间层的 dropout 操作
        hidden_states = self.intermediate_dropout(hidden_states, deterministic=deterministic)

        # 输出层的全连接操作
        hidden_states = self.output_dense(hidden_states)
        # 输出层的 dropout 操作
        hidden_states = self.output_dropout(hidden_states, deterministic=deterministic)
        # 返回最终的隐藏状态
        return hidden_states


# 定义一个名为 FlaxWav2Vec2EncoderLayerStableLayerNorm 的自定义神经网络模块，继承自 nn.Module
class FlaxWav2Vec2EncoderLayerStableLayerNorm(nn.Module):
    # 类属性：配置信息，类型为 Wav2Vec2Config
    config: Wav2Vec2Config
    # 类属性：数据类型，默认为 jnp.float32
    dtype: jnp.dtype = jnp.float32

    # 初始化方法，设置网络结构
    def setup(self):
        # 定义注意力层
        self.attention = FlaxWav2Vec2Attention(
            config=self.config,
            embed_dim=self.config.hidden_size,
            num_heads=self.config.num_attention_heads,
            dropout=self.config.attention_dropout,
            dtype=self.dtype,
        )
        # 定义隐藏层的 dropout 操作
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout)
        # 定义层归一化操作，使用配置中的 epsilon
        self.layer_norm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)
        # 定义前馈网络层
        self.feed_forward = FlaxWav2Vec2FeedForward(self.config, dtype=self.dtype)
        # 定义最终的层归一化操作
        self.final_layer_norm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)

    # 前向传播方法，接收隐藏状态、注意力掩码、是否确定性的标志和是否输出注意力权重的标志，返回输出
    def __call__(self, hidden_states, attention_mask=None, deterministic=True, output_attentions=False):
        # 记录注意力残差连接
        attn_residual = hidden_states
        # 应用层归一化操作
        hidden_states = self.layer_norm(hidden_states)
        # 注意力层的前向传播
        hidden_states, attn_weights = self.attention(
            hidden_states, attention_mask=attention_mask, deterministic=deterministic
        )
        # 应用隐藏层的 dropout 操作
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        # 加上注意力残差连接
        hidden_states = attn_residual + hidden_states
        # 应用前馈网络层
        hidden_states = hidden_states + self.feed_forward(
            self.final_layer_norm(hidden_states), deterministic=deterministic
        )

        # 输出结果
        outputs = (hidden_states,)

        # 如果需要输出注意力权重，则添加到输出中
        if output_attentions:
            outputs += (attn_weights,)

        return outputs


# 定义一个名为 FlaxWav2Vec2EncoderLayerStableLayerNormCollection 的自定义神经网络模块，继承自 nn.Module
class FlaxWav2Vec2EncoderLayerStableLayerNormCollection(nn.Module):
    # 类属性：配置信息，类型为 Wav2Vec2Config
    config: Wav2Vec2Config
    # 定义数据类型为 jnp.float32，默认为浮点数类型
    dtype: jnp.dtype = jnp.float32
    
    # 定义初始化方法，创建多个编码层对象并存储在列表 self.layers 中
    def setup(self):
        self.layers = [
            # 使用 FlaxWav2Vec2EncoderLayerStableLayerNorm 类创建编码层对象，编号从 '0' 到 str(num_hidden_layers-1)，并指定数据类型为 self.dtype
            FlaxWav2Vec2EncoderLayerStableLayerNorm(self.config, name=str(i), dtype=self.dtype)
            for i in range(self.config.num_hidden_layers)
        ]
    
    # 定义调用方法，接受输入 hidden_states 和多个可选参数，并根据参数返回结果
    def __call__(
        self,
        hidden_states,  # 输入的隐藏状态张量
        attention_mask=None,  # 可选的注意力掩码张量，默认为 None
        deterministic: bool = True,  # 是否确定性推断，默认为 True
        output_attentions: bool = False,  # 是否输出注意力张量，默认为 False
        output_hidden_states: bool = False,  # 是否输出所有隐藏状态，默认为 False
        return_dict: bool = True,  # 是否以字典形式返回结果，默认为 True
    ):
        # 初始化空的元组变量 all_attentions 和 all_hidden_states，根据参数决定是否存储相应的输出
        all_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
    
        # 遍历 self.layers 中的编码层，并依次处理隐藏状态
        for i, layer in enumerate(self.layers):
            if output_hidden_states:
                # 如果需要输出隐藏状态，则将当前隐藏状态存入 all_hidden_states 中
                all_hidden_states += (hidden_states,)
    
            # 调用当前层的 __call__ 方法，处理隐藏状态和注意力掩码，根据参数确定是否输出注意力张量
            layer_outputs = layer(
                hidden_states, attention_mask, deterministic=deterministic, output_attentions=output_attentions
            )
    
            # 更新隐藏状态为当前层的输出的第一个元素
            hidden_states = layer_outputs[0]
    
            if output_attentions:
                # 如果需要输出注意力张量，则将当前层的注意力张量存入 all_attentions 中
                all_attentions += (layer_outputs[1],)
    
        if output_hidden_states:
            # 如果需要输出隐藏状态，则将最终的隐藏状态存入 all_hidden_states 中
            all_hidden_states += (hidden_states,)
    
        # 按照设定的返回方式构建输出元组 outputs
        outputs = (hidden_states, all_hidden_states, all_attentions)
    
        if not return_dict:
            # 如果不需要以字典形式返回，则返回一个去除 None 值后的元组
            return tuple(v for v in outputs if v is not None)
    
        # 如果需要以字典形式返回，则返回一个包含最终隐藏状态、所有隐藏状态和所有注意力张量的 FlaxBaseModelOutput 对象
        return FlaxBaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_attentions
        )
class FlaxWav2Vec2StableLayerNormEncoder(nn.Module):
    # Wav2Vec2Config类型的配置对象
    config: Wav2Vec2Config
    # 数据类型，默认为32位浮点数
    dtype: jnp.dtype = jnp.float32

    # 模块设置方法，初始化各个子模块
    def setup(self):
        # 位置卷积嵌入层对象，使用Wav2Vec2Config配置和指定数据类型
        self.pos_conv_embed = FlaxWav2Vec2PositionalConvEmbedding(self.config, dtype=self.dtype)
        # 层归一化对象，使用指定的epsilon值和数据类型
        self.layer_norm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)
        # 丢弃层对象，使用指定的丢弃率
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout)
        # 编码器层集合对象，使用Wav2Vec2Config配置和指定数据类型
        self.layers = FlaxWav2Vec2EncoderLayerStableLayerNormCollection(self.config, dtype=self.dtype)

    # 对象调用方法，实现编码器的前向计算
    def __call__(
        self,
        hidden_states,
        attention_mask=None,
        deterministic=True,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        # 如果存在注意力掩码，则确保填充的令牌不被注意到
        if attention_mask is not None:
            hidden_states = jnp.where(
                # 根据注意力掩码扩展到hidden_states的形状，将未被掩盖的位置置为0
                jnp.broadcast_to(attention_mask[:, :, None], hidden_states.shape), hidden_states, 0
            )

        # 计算位置嵌入
        position_embeddings = self.pos_conv_embed(hidden_states)

        # 将位置嵌入加到hidden_states中
        hidden_states = hidden_states + position_embeddings
        # 对加了位置嵌入的hidden_states进行丢弃操作
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)

        # 调用编码器层集合对象进行编码器层的前向计算
        outputs = self.layers(
            hidden_states,
            attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 对编码器输出的最后一个隐藏状态进行层归一化处理
        last_hidden_state = self.layer_norm(outputs[0])

        # 如果需要返回隐藏状态历史，更新最后一个`hidden_states`元素
        hidden_states = None
        if output_hidden_states:
            hidden_states = outputs[1]
            hidden_states = hidden_states[:-1] + (last_hidden_state,)

        # 如果不返回字典格式的结果，则展开outputs并返回非空值
        if not return_dict:
            outputs = (last_hidden_state, hidden_states) + (outputs[2:] if output_hidden_states else outputs[1:])
            return tuple(v for v in outputs if v is not None)

        # 返回FlaxBaseModelOutput对象，包括最后的隐藏状态、隐藏状态历史和注意力信息
        return FlaxBaseModelOutput(
            last_hidden_state=last_hidden_state, hidden_states=hidden_states, attentions=outputs.attentions
        )


class FlaxWav2Vec2GumbelVectorQuantizer(nn.Module):
    """
    Vector quantization using gumbel softmax. See [CATEGORICAL REPARAMETERIZATION WITH
    GUMBEL-SOFTMAX](https://arxiv.org/pdf/1611.01144.pdf) for more information.
    """

    # Wav2Vec2Config类型的配置对象
    config: Wav2Vec2Config
    # 数据类型，默认为32位浮点数
    dtype: jnp.dtype = jnp.float32
    # 在设置方法中初始化类的一些属性
    def setup(self):
        # 将配置中的参数赋值给实例属性
        self.num_groups = self.config.num_codevector_groups
        self.num_vars = self.config.num_codevectors_per_group

        # 检查是否能够均匀分割 codevector_dim
        if self.config.codevector_dim % self.num_groups != 0:
            # 如果不能整除，抛出数值错误异常
            raise ValueError(
                f"`config.codevector_dim {self.config.codevector_dim} must be divisible by"
                f" `config.num_codevector_groups` {self.num_groups} for concatenation"
            )

        # 为存储码书变量（码字）预留空间
        self.codevectors = self.param(
            "codevectors",
            jax.nn.initializers.uniform(),
            (1, self.num_groups * self.num_vars, self.config.codevector_dim // self.num_groups),
        )
        
        # 设置权重投影层
        self.weight_proj = nn.Dense(
            self.num_groups * self.num_vars,
            kernel_init=jax.nn.initializers.normal(1.0),
            dtype=self.dtype,
        )

    # 静态方法：计算困惑度
    @staticmethod
    def _compute_perplexity(probs, mask=None):
        # 如果有掩码，扩展掩码并应用到概率矩阵上
        if mask is not None:
            mask_extended = jnp.broadcast_to(mask.flatten()[:, None, None], probs.shape)
            probs = jnp.where(mask_extended, probs, jnp.zeros_like(probs))
            marginal_probs = probs.sum(axis=0) / mask.sum()
        else:
            # 否则，计算概率矩阵的平均值
            marginal_probs = probs.mean(axis=0)

        # 计算困惑度
        perplexity = jnp.exp(-jnp.sum(marginal_probs * jnp.log(marginal_probs + 1e-7), axis=-1)).sum()
        return perplexity
    def __call__(self, hidden_states, mask_time_indices=None, deterministic=True, temperature=1):
        batch_size, sequence_length, hidden_size = hidden_states.shape

        # 将隐藏状态投影到代码向量维度
        hidden_states = self.weight_proj(hidden_states)
        hidden_states = hidden_states.reshape(batch_size * sequence_length * self.num_groups, -1)

        if not deterministic:
            # 使用古贝尔分布在可区分的方式中采样代码向量概率
            gumbel_rng = self.make_rng("gumbel")
            gumbels = jax.random.gumbel(gumbel_rng, hidden_states.shape)
            codevector_probs = nn.softmax((hidden_states + gumbels) / temperature)

            # 计算困惑度
            codevector_soft_dist = nn.softmax(
                hidden_states.reshape(batch_size * sequence_length, self.num_groups, -1), axis=-1
            )
            perplexity = self._compute_perplexity(codevector_soft_dist, mask_time_indices)
        else:
            # 以非可区分的方式取 argmax
            # 计算硬代码向量分布（one-hot）
            codevector_idx = hidden_states.argmax(axis=-1)
            codevector_probs = jax.nn.one_hot(codevector_idx, hidden_states.shape[-1]) * 1.0
            codevector_probs = codevector_probs.reshape(batch_size * sequence_length, self.num_groups, -1)
            perplexity = self._compute_perplexity(codevector_probs, mask_time_indices)

        codevector_probs = codevector_probs.reshape(batch_size * sequence_length, -1)
        # 使用概率值检索代码向量
        codevectors_per_group = jnp.expand_dims(codevector_probs, axis=-1) * self.codevectors
        codevectors = codevectors_per_group.reshape(batch_size * sequence_length, self.num_groups, self.num_vars, -1)
        codevectors = codevectors.sum(-2).reshape(batch_size, sequence_length, -1)

        return codevectors, perplexity
class FlaxWav2Vec2Adapter(nn.Module):
    config: Wav2Vec2Config
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        # hidden_states require down-projection if feature dims don't match
        # 如果特征维度不匹配，则需要对隐藏状态进行降维投影
        if self.config.output_hidden_size != self.config.hidden_size:
            # Initialize a Dense layer for projection with normal distribution initialization
            # 初始化一个用于投影的稠密层，使用正态分布进行初始化
            self.proj = nn.Dense(
                self.config.output_hidden_size,
                kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
                dtype=self.dtype,
            )
            # Layer normalization for the projection layer
            # 投影层的层归一化
            self.proj_layer_norm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)
        else:
            self.proj = self.proj_layer_norm = None

        # Initialize the collection of adapter layers
        # 初始化适配器层集合
        self.layers = FlaxWav2Vec2AdapterLayersCollection(self.config, dtype=self.dtype)

    def __call__(self, hidden_states, deterministic=True):
        # down-project hidden_states if required
        # 如果需要，则对隐藏状态进行降维投影
        if self.proj is not None and self.proj_layer_norm is not None:
            hidden_states = self.proj(hidden_states)
            hidden_states = self.proj_layer_norm(hidden_states)

        # Pass hidden_states through adapter layers
        # 通过适配器层处理隐藏状态
        hidden_states = self.layers(hidden_states)

        return hidden_states


class FlaxWav2Vec2AdapterLayer(nn.Module):
    config: Wav2Vec2Config
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        # Initialize a convolutional layer for the adapter layer
        # 初始化适配器层的卷积层
        self.conv = nn.Conv(
            features=2 * self.config.output_hidden_size,
            kernel_size=(self.config.adapter_kernel_size,),
            strides=(self.config.adapter_stride,),
            padding=((1, 1),),
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            dtype=self.dtype,
        )

    def __call__(self, hidden_states):
        # Apply convolution to hidden_states
        # 将卷积应用于隐藏状态
        hidden_states = self.conv(hidden_states)
        # Apply gated linear unit (GLU) activation along axis 2
        # 沿着轴 2 应用门控线性单元（GLU）激活函数
        hidden_states = nn.glu(hidden_states, axis=2)

        return hidden_states


class FlaxWav2Vec2AdapterLayersCollection(nn.Module):
    config: Wav2Vec2Config
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        # Initialize a list of adapter layers
        # 初始化适配器层的列表
        self.layers = [
            FlaxWav2Vec2AdapterLayer(self.config, name=str(i), dtype=self.dtype)
            for i in range(self.config.num_adapter_layers)
        ]

    def __call__(self, hidden_states):
        # Iterate through each adapter layer and apply it to hidden_states
        # 遍历每个适配器层，并将其应用于隐藏状态
        for conv_layer in self.layers:
            hidden_states = conv_layer(hidden_states)

        return hidden_states


class FlaxWav2Vec2PreTrainedModel(FlaxPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = Wav2Vec2Config
    base_model_prefix: str = "wav2vec2"
    main_input_name = "input_values"
    module_class: nn.Module = None

    def __init__(
        self,
        config: Wav2Vec2Config,
        input_shape: Tuple = (1, 1024),
        seed: int = 0,
        dtype: jnp.dtype = jnp.float32,
        _do_init: bool = True,
        **kwargs,
    ):
        # 使用配置和数据类型初始化模块对象
        module = self.module_class(config=config, dtype=dtype, **kwargs)
        # 调用父类初始化方法，传递配置、模块对象、输入形状、随机种子、数据类型和是否执行初始化的标志
        super().__init__(config, module, input_shape=input_shape, seed=seed, dtype=dtype, _do_init=_do_init)

    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict = None) -> FrozenDict:
        # 初始化输入张量
        input_values = jnp.zeros(input_shape, dtype="i4")
        # 创建一个与输入值形状相同的全1张量作为注意力掩码
        attention_mask = jnp.ones_like(input_values)
        # 拆分随机数生成器为两部分，一个用于参数，一个用于dropout
        params_rng, dropout_rng = jax.random.split(rng, 2)
        rngs = {"params": params_rng, "dropout": dropout_rng}

        # 使用模块的初始化方法初始化参数，返回参数字典
        random_params = self.module.init(rngs, input_values, attention_mask, return_dict=False)["params"]

        if params is not None:
            # 如果传入了额外的参数，将随机生成的参数与传入的参数合并
            random_params = flatten_dict(unfreeze(random_params))
            params = flatten_dict(unfreeze(params))
            for missing_key in self._missing_keys:
                params[missing_key] = random_params[missing_key]
            self._missing_keys = set()
            return freeze(unflatten_dict(params))
        else:
            # 否则直接返回随机生成的参数
            return random_params

    @add_start_docstrings_to_model_forward(WAV_2_VEC_2_INPUTS_DOCSTRING)
    def __call__(
        self,
        input_values,
        attention_mask=None,
        mask_time_indices=None,
        params: dict = None,
        dropout_rng: jax.random.PRNGKey = None,
        train: bool = False,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        freeze_feature_encoder: bool = False,
        return_dict: Optional[bool] = None,
    ):
        # 如果输出注意力没有明确指定，则使用配置中的设置
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 如果输出隐藏状态没有明确指定，则使用配置中的设置
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 如果返回字典没有明确指定，则使用配置中的设置
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        # 获取输入数据的批量大小和序列长度
        batch_size, sequence_length = input_values.shape

        # 如果没有提供注意力掩码，则创建一个全1的注意力掩码
        if attention_mask is None:
            attention_mask = jnp.ones((batch_size, sequence_length))

        # 处理可能存在的随机数生成器
        rngs = {}
        if dropout_rng is not None:
            rngs["dropout"] = dropout_rng

        # 构建输入参数字典，如果未提供params则使用self.params
        inputs = {"params": params or self.params}

        # 调用模块的应用方法，执行模型前向传播
        return self.module.apply(
            inputs,
            jnp.array(input_values, dtype="f4"),
            jnp.array(attention_mask, dtype="i4"),
            mask_time_indices,
            not train,
            output_attentions,
            output_hidden_states,
            freeze_feature_encoder,
            return_dict,
            rngs=rngs,
        )

    def _get_feat_extract_output_lengths(
        self, input_lengths: Union[jnp.ndarray, int], add_adapter: Optional[bool] = None
    ):
        # 调用模块的特征提取方法，获取输出长度
        return self.module._get_feat_extract_output_lengths(input_lengths, add_adapter=add_adapter)
# 定义一个名为 FlaxWav2Vec2Module 的 PyTorch 模块
class FlaxWav2Vec2Module(nn.Module):
    # 类型注解：配置信息为 Wav2Vec2Config 类型
    config: Wav2Vec2Config
    # 数据类型，默认为 jnp.float32
    dtype: jnp.dtype = jnp.float32

    # 模块初始化方法
    def setup(self):
        # 初始化特征提取器，使用配置信息和指定数据类型
        self.feature_extractor = FlaxWav2Vec2FeatureEncoder(self.config, dtype=self.dtype)
        # 初始化特征投影器，使用配置信息和指定数据类型
        self.feature_projection = FlaxWav2Vec2FeatureProjection(self.config, dtype=self.dtype)
        # 初始化掩码后的谱图嵌入参数，形状为 (hidden_size,)
        self.masked_spec_embed = self.param(
            "masked_spec_embed", jax.nn.initializers.uniform(), (self.config.hidden_size,)
        )

        # 如果配置指定使用稳定层归一化
        if self.config.do_stable_layer_norm:
            # 初始化编码器，使用配置信息和指定数据类型
            self.encoder = FlaxWav2Vec2StableLayerNormEncoder(self.config, dtype=self.dtype)
        else:
            # 抛出错误，暂不支持稳定层归一化未启用的情况
            raise NotImplementedError("``config.do_stable_layer_norm is False`` is currently not supported.")

        # 如果配置指定添加适配器，初始化适配器
        self.adapter = FlaxWav2Vec2Adapter(self.config, dtype=self.dtype) if self.config.add_adapter else None

    # 模块的调用方法，用于执行模型前向传播
    def __call__(
        self,
        input_values,
        attention_mask=None,
        mask_time_indices=None,
        deterministic=True,
        output_attentions=None,
        output_hidden_states=None,
        freeze_feature_encoder=False,
        return_dict=None,
    ):
        # 提取特征向量
        extract_features = self.feature_extractor(input_values, freeze_feature_encoder=freeze_feature_encoder)

        # 如果有注意力掩码
        if attention_mask is not None:
            # 计算对应于特征向量的减少注意力掩码
            attention_mask = self._get_feature_vector_attention_mask(
                extract_features.shape[1], attention_mask, add_adapter=False
            )

        # 特征投影
        hidden_states, extract_features = self.feature_projection(extract_features, deterministic=deterministic)
        
        # 如果有时间轴索引的掩码
        if mask_time_indices is not None:
            # 在时间轴上应用 SpecAugment，并使用给定的索引
            hidden_states = jnp.where(
                jnp.broadcast_to(mask_time_indices[:, :, None], hidden_states.shape),
                jnp.broadcast_to(self.masked_spec_embed[None, None, :], hidden_states.shape),
                hidden_states,
            )

        # 编码器的输出
        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=attention_mask,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 编码器的隐藏状态
        hidden_states = encoder_outputs[0]

        # 如果有适配器，应用适配器
        if self.adapter is not None:
            hidden_states = self.adapter(hidden_states)

        # 如果不返回字典形式的结果
        if not return_dict:
            # 返回元组形式的结果：(隐藏状态, 提取的特征) + 编码器输出中的其余部分
            return (hidden_states, extract_features) + encoder_outputs[1:]

        # 返回 FlaxWav2Vec2BaseModelOutput 类的实例，包括最后的隐藏状态、提取的特征、隐藏状态和注意力权重
        return FlaxWav2Vec2BaseModelOutput(
            last_hidden_state=hidden_states,
            extract_features=extract_features,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

    # 辅助方法：获取特征提取器的输出长度
    def _get_feat_extract_output_lengths(
        self, input_lengths: Union[jnp.ndarray, int], add_adapter: Optional[bool] = None
    ):
        """
        Computes the output length of the convolutional layers
        计算卷积层的输出长度
        """

        add_adapter = self.config.add_adapter if add_adapter is None else add_adapter

        def _conv_out_length(input_length, kernel_size, stride):
            # 1D convolutional layer output length formula taken
            # from https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
            # 1D卷积层输出长度的计算公式，参考自 https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
            return (input_length - kernel_size) // stride + 1

        for kernel_size, stride in zip(self.config.conv_kernel, self.config.conv_stride):
            input_lengths = _conv_out_length(input_lengths, kernel_size, stride)

        if add_adapter:
            for _ in range(self.config.num_adapter_layers):
                input_lengths = _conv_out_length(input_lengths, 1, self.config.adapter_stride)

        return input_lengths

    def _get_feature_vector_attention_mask(
        self, feature_vector_length: int, attention_mask: jnp.ndarray, add_adapter=None
    ):
        # Effectively attention_mask.sum(-1), but not inplace to be able to run
        # on inference mode.
        # 实际上是 attention_mask.sum(-1)，但不是原地操作，以便在推断模式下运行。
        non_padded_lengths = attention_mask.cumsum(axis=-1)[:, -1]

        output_lengths = self._get_feat_extract_output_lengths(non_padded_lengths, add_adapter=add_adapter)

        batch_size = attention_mask.shape[0]

        attention_mask = jnp.zeros((batch_size, feature_vector_length), dtype=attention_mask.dtype)
        # these two operations makes sure that all values
        # before the output lengths indices are attended to
        # 这两个操作确保所有输出长度索引之前的值都被关注到
        attention_mask = attention_mask.at[jnp.arange(attention_mask.shape[0]), output_lengths - 1].set(1)
        attention_mask = jnp.flip(jnp.flip(attention_mask, -1).cumsum(-1), -1).astype("bool")
        return attention_mask
# 添加函数文档字符串和装饰器，描述此类作为没有特定输出头部的裸Wav2Vec2模型转换器
@add_start_docstrings(
    "The bare Wav2Vec2 Model transformer outputting raw hidden-states without any specific head on top.",
    WAV_2_VEC_2_START_DOCSTRING,
)
# 定义 FlaxWav2Vec2Model 类，继承自 FlaxWav2Vec2PreTrainedModel 类
class FlaxWav2Vec2Model(FlaxWav2Vec2PreTrainedModel):
    module_class = FlaxWav2Vec2Module  # 设置模块类为 FlaxWav2Vec2Module


# 定义 FLAX_WAV2VEC2_MODEL_DOCSTRING 作为模型的文档字符串，描述返回值和示例用法
FLAX_WAV2VEC2_MODEL_DOCSTRING = """
    Returns:

    Example:

    ```python
    >>> from transformers import AutoProcessor, FlaxWav2Vec2Model
    >>> from datasets import load_dataset
    >>> import soundfile as sf

    >>> processor = AutoProcessor.from_pretrained("facebook/wav2vec2-large-lv60")
    >>> model = FlaxWav2Vec2Model.from_pretrained("facebook/wav2vec2-large-lv60")


    >>> def map_to_array(batch):
    ...     speech, _ = sf.read(batch["file"])
    ...     batch["speech"] = speech
    ...     return batch


    >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    >>> ds = ds.map(map_to_array)

    >>> input_values = processor(
    ...     ds["speech"][0], sampling_rate=16_000, return_tensors="np"
    ... ).input_values  # Batch size 1
    >>> hidden_states = model(input_values).last_hidden_state
    ```
"""

# 调用 overwrite_call_docstring 函数，将输入的文档字符串添加到 FlaxWav2Vec2Model 类的文档字符串中
overwrite_call_docstring(
    FlaxWav2Vec2Model,
    WAV_2_VEC_2_INPUTS_DOCSTRING + FLAX_WAV2VEC2_MODEL_DOCSTRING,
)

# 调用 append_replace_return_docstrings 函数，为 FlaxWav2Vec2Model 类添加返回值文档字符串
append_replace_return_docstrings(
    FlaxWav2Vec2Model, output_type=FlaxWav2Vec2BaseModelOutput, config_class=Wav2Vec2Config
)


# 定义 FlaxWav2Vec2ForCTCModule 类，继承自 nn.Module
class FlaxWav2Vec2ForCTCModule(nn.Module):
    config: Wav2Vec2Config
    dtype: jnp.dtype = jnp.float32

    # 初始化函数，设置模块及其成员
    def setup(self):
        self.wav2vec2 = FlaxWav2Vec2Module(self.config, dtype=self.dtype)  # 初始化 wav2vec2 模块
        self.dropout = nn.Dropout(rate=self.config.final_dropout)  # 初始化 dropout 层
        self.lm_head = nn.Dense(  # 初始化语言模型头部 Dense 层
            self.config.vocab_size,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            dtype=self.dtype,
        )

    # 调用函数，定义模型的前向传播逻辑
    def __call__(
        self,
        input_values,
        attention_mask=None,
        mask_time_indices=None,
        deterministic=True,
        output_attentions=None,
        output_hidden_states=None,
        freeze_feature_encoder=False,
        return_dict=None,
    ):
        # 调用 wav2vec2 模块进行前向传播，获取输出
        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            mask_time_indices=mask_time_indices,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            freeze_feature_encoder=freeze_feature_encoder,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]  # 获取隐藏状态
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)  # 应用 dropout

        logits = self.lm_head(hidden_states)  # 计算 logits

        if not return_dict:
            return (logits,) + outputs[2:]  # 返回 logits 和其他输出

        # 返回包含 logits、隐藏状态和注意力的 FlaxCausalLMOutput 对象
        return FlaxCausalLMOutput(logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)
    def _get_feat_extract_output_lengths(
        self,
        input_lengths: Union[jnp.ndarray, int],
        add_adapter: Optional[bool] = None,
    ):
        """
        Computes the output length of the convolutional layers
        """

        # 如果 add_adapter 未提供，则使用配置中的默认值
        add_adapter = self.config.add_adapter if add_adapter is None else add_adapter

        def _conv_out_length(input_length, kernel_size, stride):
            # 从 https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html 获取的
            # 1维卷积层输出长度计算公式
            return (input_length - kernel_size) // stride + 1

        # 遍历每个卷积核大小和步长，并计算每层卷积的输出长度
        for kernel_size, stride in zip(self.config.conv_kernel, self.config.conv_stride):
            input_lengths = _conv_out_length(input_lengths, kernel_size, stride)

        # 如果需要添加适配器层，根据配置中的适配器层数量和步长进行计算
        if add_adapter:
            for _ in range(self.config.num_adapter_layers):
                input_lengths = _conv_out_length(input_lengths, 1, self.config.adapter_stride)

        # 返回最终计算得到的输出长度
        return input_lengths
# 使用装饰器为 FlaxWav2Vec2ForCTC 类添加文档字符串，描述其为在 Connectionist Temporal Classification (CTC) 上加有语言建模头部的 Wav2Vec2 模型
@add_start_docstrings(
    "Wav2Vec2 Model with a `language modeling` head on top for Connectionist Temporal Classification (CTC).",
    WAV_2_VEC_2_START_DOCSTRING,
)
# 定义 FlaxWav2Vec2ForCTC 类，继承自 FlaxWav2Vec2PreTrainedModel 类
class FlaxWav2Vec2ForCTC(FlaxWav2Vec2PreTrainedModel):
    # 将 module_class 属性指定为 FlaxWav2Vec2ForCTCModule
    module_class = FlaxWav2Vec2ForCTCModule


# FLAX_WAV2VEC2_FOR_CTC_DOCSTRING 是一个长字符串，描述了 FlaxWav2Vec2ForCTC 类的返回值和示例用法

# 调用 overwrite_call_docstring 函数，为 FlaxWav2Vec2ForCTC 类的文档字符串添加输入参数文档和 FLAX_WAV2VEC2_FOR_CTC_DOCSTRING 内容
overwrite_call_docstring(
    FlaxWav2Vec2ForCTC,
    WAV_2_VEC_2_INPUTS_DOCSTRING + FLAX_WAV2VEC2_FOR_CTC_DOCSTRING,
)

# 调用 append_replace_return_docstrings 函数，为 FlaxWav2Vec2ForCTC 类添加输出类型文档，并指定 output_type 和 config_class 参数
append_replace_return_docstrings(FlaxWav2Vec2ForCTC, output_type=FlaxCausalLMOutput, config_class=Wav2Vec2Config)


# 定义 FlaxWav2Vec2ForPreTrainingModule 类，继承自 nn.Module 类
class FlaxWav2Vec2ForPreTrainingModule(nn.Module):
    # 设置 config 属性为 Wav2Vec2Config 类型，dtype 属性默认为 jnp.float32
    config: Wav2Vec2Config
    dtype: jnp.dtype = jnp.float32

    # 定义 setup 方法，初始化模块
    def setup(self):
        # 实例化 FlaxWav2Vec2Module 类，并存储在 self.wav2vec2 属性中
        self.wav2vec2 = FlaxWav2Vec2Module(self.config, dtype=self.dtype)
        # 使用 self.config.feat_quantizer_dropout 参数初始化 nn.Dropout 类，存储在 self.dropout_features 属性中
        self.dropout_features = nn.Dropout(self.config.feat_quantizer_dropout)

        # 实例化 FlaxWav2Vec2GumbelVectorQuantizer 类，并存储在 self.quantizer 属性中
        self.quantizer = FlaxWav2Vec2GumbelVectorQuantizer(self.config, dtype=self.dtype)
        # 使用 self.config.proj_codevector_dim 参数初始化 nn.Dense 类，存储在 self.project_q 属性中
        self.project_q = nn.Dense(
            self.config.proj_codevector_dim,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            dtype=self.dtype,
        )
        # 使用 self.config.proj_codevector_dim 参数初始化 nn.Dense 类，存储在 self.project_hid 属性中
        self.project_hid = nn.Dense(
            self.config.proj_codevector_dim,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            dtype=self.dtype,
        )

    # 定义 __call__ 方法，实现对象的可调用性
    def __call__(
        self,
        input_values,
        attention_mask=None,
        mask_time_indices=None,
        gumbel_temperature: int = 1,
        deterministic: bool = True,
        output_attentions=None,
        output_hidden_states=None,
        freeze_feature_encoder=False,
        return_dict=None,
        # 函数参数的注释可以在文档字符串中找到
        **kwargs,
    ):
        # 省略方法内部的具体实现，不在注释范围内
        ):
        r"""
        Returns:

        Example:

        ```python

        ```"""

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 使用给定的参数调用wav2vec2模型，获取输出
        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            mask_time_indices=mask_time_indices,
            deterministic=deterministic,
            freeze_feature_encoder=freeze_feature_encoder,
            return_dict=return_dict,
        )

        # 将所有转换后的特征（包括被掩码的）投影到最终的向量量化维度
        transformer_features = self.project_hid(outputs[0])

        # 量化所有（未被掩码的）提取特征并投影到最终的向量量化维度
        extract_features = self.dropout_features(outputs[1], deterministic=deterministic)
        quantized_features, codevector_perplexity = self.quantizer(
            extract_features, mask_time_indices, deterministic=deterministic, temperature=gumbel_temperature
        )
        quantized_features = self.project_q(quantized_features)

        # 如果不使用返回字典，则返回元组形式的输出
        if not return_dict:
            return (transformer_features, quantized_features, codevector_perplexity) + outputs[2:]

        # 使用FlaxWav2Vec2ForPreTrainingOutput类封装输出，包括所有相关信息
        return FlaxWav2Vec2ForPreTrainingOutput(
            projected_states=transformer_features,
            projected_quantized_states=quantized_features,
            codevector_perplexity=codevector_perplexity,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def _get_feat_extract_output_lengths(
        self, input_lengths: Union[jnp.ndarray, int], add_adapter: Optional[bool] = None
    ):
        """
        计算卷积层的输出长度
        """

        add_adapter = self.config.add_adapter if add_adapter is None else add_adapter

        def _conv_out_length(input_length, kernel_size, stride):
            # 从 https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html 中得到的一维卷积层输出长度公式
            return (input_length - kernel_size) // stride + 1

        # 遍历配置的卷积核大小和步幅，计算每一层卷积层的输出长度
        for kernel_size, stride in zip(self.config.conv_kernel, self.config.conv_stride):
            input_lengths = _conv_out_length(input_lengths, kernel_size, stride)

        # 如果需要添加适配器层，则计算适配器层的输出长度
        if add_adapter:
            for _ in range(self.config.num_adapter_layers):
                input_lengths = _conv_out_length(input_lengths, 1, self.config.adapter_stride)

        return input_lengths
@add_start_docstrings("""Wav2Vec2 Model with a quantizer and `VQ` head on top.""", WAV_2_VEC_2_START_DOCSTRING)
class FlaxWav2Vec2ForPreTraining(FlaxWav2Vec2PreTrainedModel):
    module_class = FlaxWav2Vec2ForPreTrainingModule

    @add_start_docstrings_to_model_forward(WAV_2_VEC_2_INPUTS_DOCSTRING)
    # 覆盖原始定义，添加了 `gumbel_temperature` 输入参数
    def __call__(
        self,
        input_values,
        attention_mask=None,
        mask_time_indices=None,
        gumbel_temperature: int = 1,
        params: dict = None,
        dropout_rng: jax.random.PRNGKey = None,
        gumbel_rng: jax.random.PRNGKey = None,
        train: bool = False,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        freeze_feature_encoder: bool = False,
        return_dict: Optional[bool] = None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        batch_size, sequence_length = input_values.shape

        # 如果未提供注意力掩码，则创建一个全为1的注意力掩码
        if attention_mask is None:
            attention_mask = jnp.ones((batch_size, sequence_length))

        # 处理可能需要的任何伪随机数生成器
        rngs = {}
        if dropout_rng is not None:
            rngs["dropout"] = dropout_rng

        if gumbel_rng is not None:
            rngs["gumbel"] = gumbel_rng

        # 准备模型输入
        inputs = {"params": params or self.params}

        # 调用模块的前向方法
        return self.module.apply(
            inputs,
            jnp.array(input_values, dtype="f4"),
            jnp.array(attention_mask, dtype="i4"),
            mask_time_indices,
            gumbel_temperature,
            not train,
            output_attentions,
            output_hidden_states,
            freeze_feature_encoder,
            return_dict,
            rngs=rngs,
        )


FLAX_WAV2VEC2_FOR_PRETRAINING_DOCSTRING = """
    Returns:

    Example:

    ```python
    >>> import optax
    >>> import numpy as np
    >>> import jax.numpy as jnp
    >>> from transformers import AutoFeatureExtractor, FlaxWav2Vec2ForPreTraining
    >>> from transformers.models.wav2vec2.modeling_flax_wav2vec2 import _compute_mask_indices
    >>> from datasets import load_dataset
    >>> import soundfile as sf

    >>> feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-large-lv60")
    >>> model = FlaxWav2Vec2ForPreTraining.from_pretrained("facebook/wav2vec2-large-lv60")


    >>> def map_to_array(batch):
    ...     speech, _ = sf.read(batch["file"])
    ...     batch["speech"] = speech
    ...     return batch


    >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    >>> ds = ds.map(map_to_array)
    >>> input_values = feature_extractor(ds["speech"][0], return_tensors="np").input_values  # 获取输入特征向量值，批大小为1

    >>> # 计算掩码索引
    >>> batch_size, raw_sequence_length = input_values.shape  # 获取批大小和原始序列长度
    >>> sequence_length = model._get_feat_extract_output_lengths(raw_sequence_length)  # 根据模型获取特征提取后的序列长度
    >>> mask_time_indices = _compute_mask_indices((batch_size, sequence_length), mask_prob=0.2, mask_length=2)  # 计算掩码时间点的索引

    >>> outputs = model(input_values, mask_time_indices=mask_time_indices)  # 使用模型进行推理，传入掩码时间点索引

    >>> # 计算预测状态(outputs.projected_states)与目标状态(outputs.projected_quantized_states)之间的余弦相似度
    >>> cosine_sim = optax.cosine_similarity(outputs.projected_states, outputs.projected_quantized_states)

    >>> # 确保余弦相似度在掩码时间点上的平均值高于0.5
    >>> assert np.asarray(cosine_sim)[mask_time_indices].mean() > 0.5
"""
为 `FlaxWav2Vec2ForPreTraining` 类的 `__call__` 方法覆盖文档字符串，
使用 `WAV_2_VEC_2_INPUTS_DOCSTRING` 和 `FLAX_WAV2VEC2_FOR_PRETRAINING_DOCSTRING` 进行替换。
"""
overwrite_call_docstring(
    FlaxWav2Vec2ForPreTraining,
    WAV_2_VEC_2_INPUTS_DOCSTRING + FLAX_WAV2VEC2_FOR_PRETRAINING_DOCSTRING,
)

"""
为 `FlaxWav2Vec2ForPreTraining` 类附加和替换返回值文档字符串，
使用 `FlaxWav2Vec2ForPreTrainingOutput` 作为输出类型，`Wav2Vec2Config` 作为配置类。
"""
append_replace_return_docstrings(
    FlaxWav2Vec2ForPreTraining, output_type=FlaxWav2Vec2ForPreTrainingOutput, config_class=Wav2Vec2Config
)
```