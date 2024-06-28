# `.\models\wav2vec2\modeling_tf_wav2vec2.py`

```py
# 设定代码文件的字符编码为 UTF-8
# 版权声明和许可信息，表明此代码的使用受 Apache 许可证 2.0 版本的约束
#
# 警告：此文件涉及 Fairseq 作者和 HuggingFace Inc. 团队的版权，保留所有权利。

""" TensorFlow Wav2Vec2 模型。"""

from __future__ import annotations  # 允许在类型注解中使用字符串以及类型本身的声明

import warnings  # 引入警告模块
from dataclasses import dataclass  # 导入 dataclass 用于数据类的定义
from typing import Any, Optional, Tuple, Union  # 引入类型提示的模块

import numpy as np  # 引入 NumPy 库
import tensorflow as tf  # 导入 TensorFlow 库

from ...activations_tf import get_tf_activation  # 从本地相对路径导入 TensorFlow 激活函数
from ...modeling_tf_outputs import TFBaseModelOutput, TFCausalLMOutput, TFSequenceClassifierOutput  # 导入 TensorFlow 模型输出类
from ...modeling_tf_utils import (  # 导入 TensorFlow 模型工具函数
    TFPreTrainedModel,
    get_initializer,
    keras,
    keras_serializable,
    unpack_inputs,
)
from ...tf_utils import shape_list, stable_softmax  # 从 TensorFlow 实用工具模块导入函数
from ...utils import (  # 从通用工具模块导入多个实用函数和类
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from .configuration_wav2vec2 import Wav2Vec2Config  # 从本地相对路径导入 Wav2Vec2 的配置类

logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器

_HIDDEN_STATES_START_POSITION = 2  # 设置隐藏状态的起始位置索引为2

_CHECKPOINT_FOR_DOC = "facebook/wav2vec2-base-960h"  # 预训练模型的检查点名称，用于文档
_CONFIG_FOR_DOC = "Wav2Vec2Config"  # Wav2Vec2 配置文件的名称，用于文档

# 预训练模型存档列表，包含多个预训练模型的名称
TF_WAV_2_VEC_2_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/wav2vec2-base-960h",
    "facebook/wav2vec2-large-960h",
    "facebook/wav2vec2-large-960h-lv60",
    "facebook/wav2vec2-large-960h-lv60-self",
    # 查看所有 Wav2Vec2 模型：https://huggingface.co/models?filter=wav2vec2
]

LARGE_NEGATIVE = -1e8  # 定义一个较大的负数常量，用于特定目的

@dataclass
class TFWav2Vec2BaseModelOutput(ModelOutput):
    """
    [`TFWav2Vec2BaseModelOutput`] 的输出类型，包含潜在的隐藏状态和注意力。
    继承自 ModelOutput 类。
    """
    """
    Args:
        last_hidden_state (`tf.Tensor` of shape `(batch_size, sequence_length, hidden_size)`):
            模型最后一层输出的隐藏状态序列。
        extract_features (`tf.Tensor` of shape `(batch_size, sequence_length, conv_dim[-1])`):
            模型最后一个卷积层提取的特征向量序列。
        hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            包含模型每一层输出的隐藏状态的元组。形状为 `(batch_size, sequence_length, hidden_size)`。

            模型每一层的隐藏状态，以及初始嵌入输出。
        attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            包含注意力权重的元组。形状为 `(batch_size, num_heads, sequence_length, sequence_length)`。

            经过注意力 softmax 后的注意力权重，用于计算自注意力头中的加权平均值。
    """

    last_hidden_state: tf.Tensor = None  # 初始化最后一层隐藏状态为 None
    extract_features: tf.Tensor = None  # 初始化提取的特征向量为 None
    hidden_states: Tuple[tf.Tensor] | None = None  # 初始化隐藏状态元组为 None
    attentions: Tuple[tf.Tensor] | None = None  # 初始化注意力权重元组为 None
def _sample_without_replacement(distribution, num_samples):
    """
    Categorical sampling without replacement is currently not implemented. The gumbel-max trick will do for now - see
    https://github.com/tensorflow/tensorflow/issues/9260 for more info
    """
    # 使用负数对数的随机数作为采样分布
    z = -tf.math.log(tf.random.uniform(shape_list(distribution), 0, 1))
    # 对分布加上 gumbel-max 技巧后，取前 num_samples 个最高分布的索引
    _, indices = tf.nn.top_k(distribution + z, num_samples)
    return indices


def _scatter_values_on_batch_indices(values, batch_indices, output_shape):
    """
    Scatter function as in PyTorch with indices in format (batch_dim, indixes)
    """
    # 获取 batch_indices 的形状
    indices_shape = shape_list(batch_indices)
    # 扩展 batch 维度到 indices_shape 形状
    broad_casted_batch_dims = tf.reshape(
        tf.broadcast_to(tf.expand_dims(tf.range(indices_shape[0]), axis=-1), indices_shape), [1, -1]
    )
    # 将 batch_indices 转换为成对的 indices
    pair_indices = tf.transpose(tf.concat([broad_casted_batch_dims, tf.reshape(batch_indices, [1, -1])], 0))
    # 将 values 根据 pair_indices 散布到指定的 output_shape 上
    return tf.scatter_nd(pair_indices, tf.reshape(values, [-1]), output_shape)


def _compute_mask_indices(
    shape: Tuple[int, int],
    mask_prob: float,
    mask_length: int,
    min_masks: int = 0,
) -> tf.Tensor:
    """
    Computes random mask spans for a given shape

    Args:
        shape: the shape for which to compute masks.
            should be of size 2 where first element is batch size and 2nd is timesteps
        attention_mask: optional padding mask of the same size as shape, which will prevent masking padded elements
        mask_prob:
            probability for each token to be chosen as start of the span to be masked. this will be multiplied by
            number of timesteps divided by length of mask span to mask approximately this percentage of all elements.
            however due to overlaps, the actual number will be smaller (unless no_overlap is True)
        mask_length: size of the mask
        min_masks: minimum number of masked spans

    Adapted from [fairseq's
    data_utils.py](https://github.com/pytorch/fairseq/blob/e0788f7007a8473a76db573985031f3c94201e79/fairseq/data/data_utils.py#L376).
    """
    batch_size, sequence_length = shape

    if mask_length < 1:
        raise ValueError("`mask_length` has to be bigger than 0.")

    tf.debugging.assert_less(
        mask_length,
        sequence_length,
        message=(
            f"`mask_length` has to be smaller than `sequence_length`, but got `mask_length`: {mask_length} and"
            f" `sequence_length`: {sequence_length}`"
        ),
    )

    # 计算批次中的被遮罩索引数目
    num_masked_spans = mask_prob * tf.cast(sequence_length, tf.float32) / mask_length + tf.random.uniform((1,))
    num_masked_spans = tf.maximum(num_masked_spans, min_masks)
    num_masked_spans = tf.cast(num_masked_spans, tf.int32)

    # 确保被遮罩的索引数目不超过 sequence_length
    # 计算允许的最大掩码数量，确保不超过序列长度的最大掩码数量
    num_masked_spans = tf.math.minimum(sequence_length // mask_length, num_masked_spans)
    # 去除可能存在的多余维度，确保得到一个标量值
    num_masked_spans = tf.squeeze(num_masked_spans)

    # 创建一个全零的张量作为 SpecAugment 掩码的初始模板
    spec_aug_mask = tf.zeros((batch_size, sequence_length), dtype=tf.int32)

    # 创建一个均匀分布的张量，用于采样掩码的起始索引，确保采样的索引不超过序列长度
    uniform_dist = tf.ones((batch_size, sequence_length - (mask_length - 1)))

    # 获取随机的索引位置，用于创建掩码
    spec_aug_mask_idxs = _sample_without_replacement(uniform_dist, num_masked_spans)

    # 将掩码的索引扩展到掩码跨度
    spec_aug_mask_idxs = tf.expand_dims(spec_aug_mask_idxs, -1)
    spec_aug_mask_idxs = tf.tile(spec_aug_mask_idxs, (1, 1, mask_length))
    spec_aug_mask_idxs = tf.reshape(spec_aug_mask_idxs, (batch_size, num_masked_spans * mask_length))

    # 创建偏移量，用于将掩码的索引扩展到每个掩码的具体位置
    offsets = tf.range(mask_length)[tf.newaxis, tf.newaxis, :]
    offsets = tf.tile(offsets, (batch_size, num_masked_spans, 1))
    offsets = tf.reshape(offsets, (batch_size, num_masked_spans * mask_length))

    # 将偏移量加到掩码的索引上，得到最终的掩码位置
    spec_aug_mask_idxs = spec_aug_mask_idxs + offsets

    # 将掩码应用到 spec_aug_mask 上，使用 _scatter_values_on_batch_indices 函数
    spec_aug_mask = _scatter_values_on_batch_indices(
        tf.ones_like(spec_aug_mask_idxs), spec_aug_mask_idxs, tf.shape(spec_aug_mask)
    )

    # 返回生成的 SpecAugment 掩码
    return spec_aug_mask
# Copied from transformers.models.bart.modeling_tf_bart._expand_mask
def _expand_mask(mask: tf.Tensor, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    # 获取输入张量的第二个维度，即序列长度
    src_len = shape_list(mask)[1]
    # 如果没有提供目标长度，使用源长度作为目标长度
    tgt_len = tgt_len if tgt_len is not None else src_len
    # 创建常数张量，值为1.0，数据类型与输入张量相同
    one_cst = tf.constant(1.0)
    # 将输入张量转换为浮点数类型
    mask = tf.cast(mask, dtype=one_cst.dtype)
    # 在第二维度上复制输入张量，使其形状变为 [bsz, 1, tgt_seq_len, src_seq_len]
    expanded_mask = tf.tile(mask[:, None, None, :], (1, 1, tgt_len, 1))

    # 返回扩展后的注意力掩码，乘以一个大负数，用于模型中的无效化处理
    return (one_cst - expanded_mask) * LARGE_NEGATIVE


class TFWav2Vec2GroupNorm(keras.layers.Layer):
    """
    From tensorflow-addons https://www.tensorflow.org/addons/api_docs/python/tfa/layers/GroupNormalization
    """

    def __init__(
        self,
        groups: int = 32,
        axis: int = -1,
        epsilon: float = 1e-3,
        center: bool = True,
        scale: bool = True,
        beta_initializer: keras.initializers.Initializer = "zeros",
        gamma_initializer: keras.initializers.Initializer = "ones",
        beta_regularizer: keras.regularizers.Regularizer = None,
        gamma_regularizer: keras.regularizers.Regularizer = None,
        beta_constraint: keras.constraints.Constraint = None,
        gamma_constraint: keras.constraints.Constraint = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.supports_masking = True
        # 分组数
        self.groups = groups
        # 归一化的轴
        self.axis = axis
        # 小数项，防止分母为零
        self.epsilon = epsilon
        # 是否包含中心参数
        self.center = center
        # 是否包含缩放参数
        self.scale = scale
        # beta 初始化器
        self.beta_initializer = keras.initializers.get(beta_initializer)
        # gamma 初始化器
        self.gamma_initializer = keras.initializers.get(gamma_initializer)
        # beta 正则化器
        self.beta_regularizer = keras.regularizers.get(beta_regularizer)
        # gamma 正则化器
        self.gamma_regularizer = keras.regularizers.get(gamma_regularizer)
        # beta 约束条件
        self.beta_constraint = keras.constraints.get(beta_constraint)
        # gamma 约束条件
        self.gamma_constraint = keras.constraints.get(gamma_constraint)
        # 检查归一化轴
        self._check_axis()

    def build(self, input_shape):
        # 检查输入形状是否为 None
        self._check_if_input_shape_is_none(input_shape)
        # 设置实例归一化的组数
        self._set_number_of_groups_for_instance_norm(input_shape)
        # 检查维度大小
        self._check_size_of_dimensions(input_shape)
        # 创建输入规范
        self._create_input_spec(input_shape)

        # 添加 gamma 权重
        self._add_gamma_weight(input_shape)
        # 添加 beta 权重
        self._add_beta_weight(input_shape)
        self.built = True
        super().build(input_shape)

    def call(self, inputs):
        # 获取输入张量的形状
        input_shape = keras.backend.int_shape(inputs)
        # 获取输入张量的 TensorFlow 形状
        tensor_input_shape = tf.shape(inputs)

        # 重塑输入张量为分组形状，返回重塑后的张量及其形状
        reshaped_inputs, group_shape = self._reshape_into_groups(inputs, input_shape, tensor_input_shape)

        # 应用归一化操作
        normalized_inputs = self._apply_normalization(reshaped_inputs, input_shape)

        # 如果是实例归一化，将张量展平为原始形状
        is_instance_norm = (input_shape[self.axis] // self.groups) == 1
        if not is_instance_norm:
            outputs = tf.reshape(normalized_inputs, tensor_input_shape)
        else:
            outputs = normalized_inputs

        # 返回归一化后的输出张量
        return outputs
    # 获取配置信息的方法，返回一个包含当前层配置信息的字典
    def get_config(self):
        # 构建配置字典，包括各种属性和超参数
        config = {
            "groups": self.groups,
            "axis": self.axis,
            "epsilon": self.epsilon,
            "center": self.center,
            "scale": self.scale,
            "beta_initializer": keras.initializers.serialize(self.beta_initializer),
            "gamma_initializer": keras.initializers.serialize(self.gamma_initializer),
            "beta_regularizer": keras.regularizers.serialize(self.beta_regularizer),
            "gamma_regularizer": keras.regularizers.serialize(self.gamma_regularizer),
            "beta_constraint": keras.constraints.serialize(self.beta_constraint),
            "gamma_constraint": keras.constraints.serialize(self.gamma_constraint),
        }
        # 调用父类的获取配置方法，获取基础配置信息
        base_config = super().get_config()
        # 合并基础配置和当前层配置，返回完整的配置字典
        return {**base_config, **config}

    # 计算输出形状的方法，直接返回输入的形状
    def compute_output_shape(self, input_shape):
        return input_shape

    # 将输入重塑为分组的形状的方法
    def _reshape_into_groups(self, inputs, input_shape, tensor_input_shape):
        # 计算分组的形状
        group_shape = [tensor_input_shape[i] for i in range(len(input_shape))]
        # 检查是否为实例归一化
        is_instance_norm = (input_shape[self.axis] // self.groups) == 1
        if not is_instance_norm:
            # 如果不是实例归一化，重新设置分组的轴
            group_shape[self.axis] = input_shape[self.axis] // self.groups
            group_shape.insert(self.axis, self.groups)
            group_shape = tf.stack(group_shape)
            # 重新形状化输入数据
            reshaped_inputs = tf.reshape(inputs, group_shape)
            return reshaped_inputs, group_shape
        else:
            # 如果是实例归一化，直接返回输入数据和分组的形状
            return inputs, group_shape

    # 应用归一化操作的方法
    def _apply_normalization(self, reshaped_inputs, input_shape):
        # 获取重塑后输入数据的形状
        group_shape = keras.backend.int_shape(reshaped_inputs)
        # 计算需要归并的轴
        group_reduction_axes = list(range(1, len(group_shape)))
        # 检查是否为实例归一化
        is_instance_norm = (input_shape[self.axis] // self.groups) == 1
        if not is_instance_norm:
            # 如果不是实例归一化，确定归并的轴
            axis = -2 if self.axis == -1 else self.axis - 1
        else:
            # 如果是实例归一化，确定归并的轴
            axis = -1 if self.axis == -1 else self.axis - 1
        group_reduction_axes.pop(axis)

        # 计算均值和方差
        mean, variance = tf.nn.moments(reshaped_inputs, group_reduction_axes, keepdims=True)

        # 获取重塑后的权重 gamma 和 beta
        gamma, beta = self._get_reshaped_weights(input_shape)

        # 应用批归一化
        normalized_inputs = tf.nn.batch_normalization(
            reshaped_inputs,
            mean=mean,
            variance=variance,
            scale=gamma,
            offset=beta,
            variance_epsilon=self.epsilon,
        )
        # 返回归一化后的数据
        return normalized_inputs

    # 获取重塑后的权重 gamma 和 beta 的方法
    def _get_reshaped_weights(self, input_shape):
        # 创建广播形状
        broadcast_shape = self._create_broadcast_shape(input_shape)
        gamma = None
        beta = None
        # 如果开启了 scale 参数，重塑 gamma
        if self.scale:
            gamma = tf.reshape(self.gamma, broadcast_shape)

        # 如果开启了 center 参数，重塑 beta
        if self.center:
            beta = tf.reshape(self.beta, broadcast_shape)

        # 返回重塑后的 gamma 和 beta
        return gamma, beta
    # 检查输入形状中指定轴的维度是否为 None，如果是则引发 ValueError 异常
    def _check_if_input_shape_is_none(self, input_shape):
        dim = input_shape[self.axis]
        if dim is None:
            raise ValueError(
                "Axis "
                + str(self.axis)
                + " of input tensor should have a defined dimension but the layer received an input with shape "
                + str(input_shape)
                + "."
            )

    # 设置 InstanceNormalization 层的分组数目，若分组数为 -1，则设置为输入的维度数
    def _set_number_of_groups_for_instance_norm(self, input_shape):
        dim = input_shape[self.axis]

        if self.groups == -1:
            self.groups = dim

    # 检查维度大小，确保分组数不大于通道数，并且分组数必须是通道数的整数倍
    def _check_size_of_dimensions(self, input_shape):
        dim = input_shape[self.axis]
        if dim < self.groups:
            raise ValueError(
                "Number of groups ("
                + str(self.groups)
                + ") cannot be more than the number of channels ("
                + str(dim)
                + ")."
            )

        if dim % self.groups != 0:
            raise ValueError(
                "Number of groups ("
                + str(self.groups)
                + ") must be a multiple of the number of channels ("
                + str(dim)
                + ")."
            )

    # 检查轴的值，如果为 0，则引发 ValueError 异常，建议使用 tf.layer.batch_normalization
    def _check_axis(self):
        if self.axis == 0:
            raise ValueError(
                "You are trying to normalize your batch axis. Do you want to use tf.layer.batch_normalization instead"
            )

    # 创建输入规范（InputSpec），用于指定输入的维度和轴信息
    def _create_input_spec(self, input_shape):
        dim = input_shape[self.axis]
        self.input_spec = keras.layers.InputSpec(ndim=len(input_shape), axes={self.axis: dim})

    # 添加 gamma 权重，如果启用 scale，则创建 gamma 权重变量，否则设为 None
    def _add_gamma_weight(self, input_shape):
        dim = input_shape[self.axis]
        shape = (dim,)

        if self.scale:
            self.gamma = self.add_weight(
                shape=shape,
                name="gamma",
                initializer=self.gamma_initializer,
                regularizer=self.gamma_regularizer,
                constraint=self.gamma_constraint,
            )
        else:
            self.gamma = None

    # 添加 beta 权重，如果启用 center，则创建 beta 权重变量，否则设为 None
    def _add_beta_weight(self, input_shape):
        dim = input_shape[self.axis]
        shape = (dim,)

        if self.center:
            self.beta = self.add_weight(
                shape=shape,
                name="beta",
                initializer=self.beta_initializer,
                regularizer=self.beta_regularizer,
                constraint=self.beta_constraint,
            )
        else:
            self.beta = None

    # 创建广播形状，用于 InstanceNormalization 层的归一化操作
    def _create_broadcast_shape(self, input_shape):
        broadcast_shape = [1] * len(input_shape)
        is_instance_norm = (input_shape[self.axis] // self.groups) == 1
        if not is_instance_norm:
            broadcast_shape[self.axis] = input_shape[self.axis] // self.groups
            broadcast_shape.insert(self.axis, self.groups)
        else:
            broadcast_shape[self.axis] = self.groups
        return broadcast_shape
class TFWav2Vec2WeightNormConv1D(keras.layers.Conv1D):
    """Adapted from https://www.tensorflow.org/probability/api_docs/python/tfp/layers/weight_norm/WeightNorm"""

    def __init__(self, filters, kernel_size, groups, explicit_padding, **kwargs):
        # 调用父类构造函数初始化卷积层
        super().__init__(
            filters=filters,
            kernel_size=kernel_size,
            groups=groups,
            padding="valid",
            use_bias=True,
            bias_initializer="he_normal",
            **kwargs,
        )
        self.explicit_padding = explicit_padding  # 设置是否使用显式填充
        self.filter_axis = 2  # 卷积核在权重张量中的轴索引
        self.kernel_norm_axes = tf.constant([0, 1])  # 计算卷积核标准化时的轴索引

    def _init_norm(self):
        """Set the norm of the weight vector."""
        # 计算权重向量的范数，用于初始化权重标准化的参数
        kernel_norm = tf.sqrt(tf.reduce_sum(tf.square(self.weight_v), axis=self.kernel_norm_axes))
        self.weight_g.assign(kernel_norm[:, tf.newaxis, tf.newaxis])  # 将计算得到的范数赋值给权重标准化的参数

    def _normalize_kernel(self):
        """Generate normalized weights."""
        # 标准化卷积核的权重
        kernel = tf.nn.l2_normalize(self.weight_v, axis=self.kernel_norm_axes) * tf.transpose(self.weight_g)
        self.kernel = tf.transpose(kernel)  # 转置得到标准化后的卷积核权重

    def build(self, input_shape):
        if not self.built:
            super().build(input_shape)

            # 初始化权重向量并赋值给self.weight_v
            self.kernel = tf.Variable(tf.transpose(self.kernel), name="weight_v", trainable=True)
            self.weight_v = self.kernel

            # 添加权重参数weight_g，用于存储卷积核标准化的参数
            self.weight_g = self.add_weight(
                name="weight_g",
                shape=(int(self.weight_v.shape[self.filter_axis]), 1, 1),
                initializer="ones",
                dtype=self.weight_v.dtype,
                trainable=True,
            )
            self._init_norm()  # 初始化权重标准化参数
            self.bias = self.add_weight(name="bias", shape=(self.filters,), initializer="zeros", trainable=True)

    def call(self, inputs):
        # 在call方法中标准化卷积核的权重
        self._normalize_kernel()

        # 对输入进行显式填充
        padded_inputs = tf.pad(inputs, ((0, 0), (self.explicit_padding, self.explicit_padding), (0, 0)))
        # 调用父类的call方法进行卷积操作
        output = super().call(padded_inputs)

        return output


class TFWav2Vec2NoLayerNormConvLayer(keras.layers.Layer):
    def __init__(self, config: Wav2Vec2Config, layer_id: int = 0, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        # 根据配置文件初始化输入和输出的卷积维度
        self.in_conv_dim = config.conv_dim[layer_id] if layer_id > 0 else 1
        self.out_conv_dim = config.conv_dim[layer_id]

        # 初始化卷积层，根据配置文件中的参数设置
        self.conv = keras.layers.Conv1D(
            filters=self.out_conv_dim,
            kernel_size=config.conv_kernel[layer_id],
            strides=config.conv_stride[layer_id],
            use_bias=config.conv_bias,
            name="conv",
        )
        # 根据配置文件获取激活函数，并赋值给self.activation
        self.activation = get_tf_activation(config.feat_extract_activation)
    # 定义一个方法用于调用卷积层和激活函数处理隐藏状态张量，并返回处理后的张量
    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        # 使用定义好的卷积层处理隐藏状态张量
        hidden_states = self.conv(hidden_states)
        # 使用定义好的激活函数处理卷积后的张量
        hidden_states = self.activation(hidden_states)
        # 返回处理后的张量
        return hidden_states

    # 定义一个方法用于构建模型，初始化卷积层
    def build(self, input_shape=None):
        # 如果模型已经构建过，则直接返回
        if self.built:
            return
        # 标记模型已构建
        self.built = True
        # 如果存在卷积层，则在命名作用域内构建卷积层
        if getattr(self, "conv", None) is not None:
            with tf.name_scope(self.conv.name):
                # 构建卷积层，指定输入形状为 [None, None, self.in_conv_dim]
                self.conv.build([None, None, self.in_conv_dim])
class TFWav2Vec2LayerNormConvLayer(keras.layers.Layer):
    # 初始化函数，设置层的参数和配置
    def __init__(self, config: Wav2Vec2Config, layer_id: int = 0, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        # 根据层 ID 设置输入和输出的卷积维度
        self.in_conv_dim = config.conv_dim[layer_id] if layer_id > 0 else 1
        self.out_conv_dim = config.conv_dim[layer_id]

        # 创建卷积层对象
        self.conv = keras.layers.Conv1D(
            filters=self.out_conv_dim,
            kernel_size=config.conv_kernel[layer_id],
            strides=config.conv_stride[layer_id],
            use_bias=config.conv_bias,
            name="conv",
        )
        # 创建层归一化对象
        self.layer_norm = keras.layers.LayerNormalization(name="layer_norm", epsilon=config.layer_norm_eps)
        # 获取激活函数对象
        self.activation = get_tf_activation(config.feat_extract_activation)

    # 前向传播函数，定义层的计算逻辑
    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        # 卷积操作
        hidden_states = self.conv(hidden_states)
        # 层归一化操作
        hidden_states = self.layer_norm(hidden_states)
        # 激活函数操作
        hidden_states = self.activation(hidden_states)
        return hidden_states

    # 构建函数，用于构建层内的各个子层
    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 构建卷积层
        if getattr(self, "conv", None) is not None:
            with tf.name_scope(self.conv.name):
                self.conv.build([None, None, self.in_conv_dim])
        # 构建层归一化层
        if getattr(self, "layer_norm", None) is not None:
            with tf.name_scope(self.layer_norm.name):
                self.layer_norm.build([None, None, self.out_conv_dim])


class TFWav2Vec2GroupNormConvLayer(keras.layers.Layer):
    # 初始化函数，设置层的参数和配置
    def __init__(self, config: Wav2Vec2Config, layer_id: int = 0, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        # 根据层 ID 设置输入和输出的卷积维度
        self.in_conv_dim = config.conv_dim[layer_id] if layer_id > 0 else 1
        self.out_conv_dim = config.conv_dim[layer_id]

        # 创建卷积层对象
        self.conv = keras.layers.Conv1D(
            filters=self.out_conv_dim,
            kernel_size=config.conv_kernel[layer_id],
            strides=config.conv_stride[layer_id],
            use_bias=config.conv_bias,
            name="conv",
        )
        # 获取激活函数对象
        self.activation = get_tf_activation(config.feat_extract_activation)
        # 创建分组归一化层对象
        self.layer_norm = TFWav2Vec2GroupNorm(
            groups=self.out_conv_dim, epsilon=config.layer_norm_eps, name="layer_norm"
        )

    # 前向传播函数，定义层的计算逻辑
    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        # 卷积操作
        hidden_states = self.conv(hidden_states)
        # 分组归一化操作
        hidden_states = self.layer_norm(hidden_states)
        # 激活函数操作
        hidden_states = self.activation(hidden_states)
        return hidden_states

    # 构建函数，用于构建层内的各个子层
    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 构建卷积层
        if getattr(self, "conv", None) is not None:
            with tf.name_scope(self.conv.name):
                self.conv.build([None, None, self.in_conv_dim])
        # 构建分组归一化层
        if getattr(self, "layer_norm", None) is not None:
            with tf.name_scope(self.layer_norm.name):
                self.layer_norm.build([None, None, self.out_conv_dim])
class TFWav2Vec2PositionalConvEmbedding(keras.layers.Layer):
    # 定义 TF Wav2Vec2 的位置卷积嵌入层，继承自 Keras 的层
    def __init__(self, config: Wav2Vec2Config, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        # 初始化函数，接收配置对象和其他关键字参数
        self.conv = TFWav2Vec2WeightNormConv1D(
            filters=config.hidden_size,
            kernel_size=config.num_conv_pos_embeddings,
            groups=config.num_conv_pos_embedding_groups,
            explicit_padding=config.num_conv_pos_embeddings // 2,
            name="conv",
        )
        # 设置卷积层，使用权重归一化的 TF Wav2Vec2 卷积层
        self.padding = TFWav2Vec2SamePadLayer(config.num_conv_pos_embeddings)
        # 设置填充层，用于保持卷积输出的长度
        self.activation = get_tf_activation(config.feat_extract_activation)
        # 获取激活函数并设置为实例的属性
        self.config = config
        # 保存配置对象到实例的属性中

    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        # 定义调用函数，处理输入的隐藏状态张量并返回处理后的张量
        hidden_states = self.conv(hidden_states)
        # 经过卷积层处理
        hidden_states = self.padding(hidden_states)
        # 经过填充层处理
        hidden_states = self.activation(hidden_states)
        # 经过激活函数处理
        return hidden_states
        # 返回处理后的隐藏状态张量

    def build(self, input_shape=None):
        # 构建函数，在第一次调用时构建层的变量
        if self.built:
            return
        self.built = True
        if getattr(self, "conv", None) is not None:
            with tf.name_scope(self.conv.name):
                self.conv.build([None, None, self.config.hidden_size])
                # 使用配置的隐藏大小构建卷积层



class TFWav2Vec2SamePadLayer(keras.layers.Layer):
    # 定义 TF Wav2Vec2 的同填充层，继承自 Keras 的层
    def __init__(self, num_conv_pos_embeddings, **kwargs):
        super().__init__(**kwargs)
        # 初始化函数，接收卷积位置嵌入数目和其他关键字参数
        self.num_pad_remove = 1 if num_conv_pos_embeddings % 2 == 0 else 0
        # 计算需要移除的填充数目，根据卷积位置嵌入的奇偶性确定

    def call(self, hidden_states):
        # 定义调用函数，处理输入的隐藏状态张量并返回处理后的张量
        if self.num_pad_remove > 0:
            hidden_states = hidden_states[:, : -self.num_pad_remove, :]
            # 如果需要移除填充，则在最后一个维度上移除相应数量的填充
        return hidden_states
        # 返回处理后的隐藏状态张量



class TFWav2Vec2FeatureEncoder(keras.layers.Layer):
    # 定义 TF Wav2Vec2 的特征编码器层，继承自 Keras 的层
    def __init__(self, config: Wav2Vec2Config, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        # 初始化函数，接收配置对象和其他关键字参数
        if config.feat_extract_norm == "group":
            # 如果特征提取归一化方式为 group
            conv_layers = [TFWav2Vec2GroupNormConvLayer(config, layer_id=0, name=f"conv_layers.{0}")] + [
                TFWav2Vec2NoLayerNormConvLayer(config, layer_id=i + 1, name=f"conv_layers.{i+1}")
                for i in range(config.num_feat_extract_layers - 1)
            ]
            # 创建一组带有组归一化的卷积层
        elif config.feat_extract_norm == "layer":
            # 如果特征提取归一化方式为 layer
            conv_layers = [
                TFWav2Vec2LayerNormConvLayer(config, layer_id=i, name=f"conv_layers.{i}")
                for i in range(config.num_feat_extract_layers)
            ]
            # 创建一组带有层归一化的卷积层
        else:
            # 如果特征提取归一化方式既不是 group 也不是 layer，则抛出异常
            raise ValueError(
                f"`config.feat_extract_norm` is {config.feat_extract_norm}, but has to be one of ['group', 'layer']"
            )
        self.conv_layers = conv_layers
        # 保存创建的卷积层列表到实例的属性中

    def call(self, input_values):
        # 定义调用函数，处理输入值并返回处理后的张量
        hidden_states = tf.expand_dims(input_values, -1)
        # 在最后一个维度上扩展输入张量
        for conv_layer in self.conv_layers:
            hidden_states = conv_layer(hidden_states)
            # 通过每个卷积层处理隐藏状态
        return hidden_states
        # 返回处理后的隐藏状态张量
    # 定义神经网络层的构建方法，接收输入形状作为参数，如果已经构建过则直接返回
    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回，不再重复构建
        if self.built:
            return
        # 标记该神经网络层已经构建
        self.built = True
        # 如果存在卷积层列表，则逐个构建每个卷积层
        if getattr(self, "conv_layers", None) is not None:
            for conv_layer in self.conv_layers:
                # 使用 TensorFlow 的命名作用域，将当前卷积层的名称作为作用域名称
                with tf.name_scope(conv_layer.name):
                    # 调用卷积层的 build 方法来构建该层
                    conv_layer.build(None)
# 定义 TFWav2Vec2FeatureExtractor 类，继承自 TFWav2Vec2FeatureEncoder 类
class TFWav2Vec2FeatureExtractor(TFWav2Vec2FeatureEncoder):
    
    # 初始化方法，接受 config 和额外的关键字参数
    def __init__(self, config, **kwargs):
        # 调用父类 TFWav2Vec2FeatureEncoder 的初始化方法
        super().__init__(config, **kwargs)
        
        # 发出警告信息，提示该类即将被废弃
        warnings.warn(
            f"The class `{self.__class__.__name__}` has been depreciated "
            "and will be removed in Transformers v5. "
            f"Use `{self.__class__.__bases__[0].__name__}` instead.",
            FutureWarning,
        )


# 定义 TFWav2Vec2FeatureProjection 类，继承自 keras 的 Layer 类
class TFWav2Vec2FeatureProjection(keras.layers.Layer):
    
    # 初始化方法，接受 config 参数和其他关键字参数
    def __init__(self, config: Wav2Vec2Config, **kwargs):
        # 调用父类的初始化方法
        super().__init__(**kwargs)

        # 使用 config 中的参数创建 LayerNormalization 层，设置 epsilon 和名称
        self.layer_norm = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="layer_norm")
        
        # 创建 Dense 层作为投影层，设置单元数、初始化器和偏置初始化器
        self.projection = keras.layers.Dense(
            units=config.hidden_size,
            kernel_initializer=get_initializer(config.initializer_range),
            bias_initializer="zeros",
            name="projection",
        )
        
        # 创建 Dropout 层，设置丢弃率
        self.dropout = keras.layers.Dropout(rate=config.feat_proj_dropout)
        
        # 保存 config 参数
        self.config = config

    # 调用方法，接受 hidden_states 和 training 参数，返回 Tensor
    def call(self, hidden_states: tf.Tensor, training: bool = False) -> tf.Tensor:
        # 对 hidden_states 进行 LayerNormalization 处理
        norm_hidden_states = self.layer_norm(hidden_states)
        
        # 将处理后的 hidden_states 投影到新的维度空间
        hidden_states = self.projection(norm_hidden_states)
        
        # 根据 training 参数应用 Dropout
        hidden_states = self.dropout(hidden_states, training=training)
        
        # 返回处理后的 hidden_states
        return hidden_states, norm_hidden_states

    # 构建方法，用于构建层结构
    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回
        if self.built:
            return
        
        # 标记已经构建
        self.built = True
        
        # 如果存在 layer_norm 属性
        if getattr(self, "layer_norm", None) is not None:
            # 在 layer_norm 的名称空间下构建该层，传入形状参数
            with tf.name_scope(self.layer_norm.name):
                self.layer_norm.build([None, None, self.config.conv_dim[-1]])
        
        # 如果存在 projection 属性
        if getattr(self, "projection", None) is not None:
            # 在 projection 的名称空间下构建该层，传入形状参数
            with tf.name_scope(self.projection.name):
                self.projection.build([None, None, self.config.conv_dim[-1]])


# 从 transformers.models.bart.modeling_tf_bart.TFBartAttention 复制而来，修改为 TFWav2Vec2Attention
class TFWav2Vec2Attention(keras.layers.Layer):
    """Multi-headed attention from "Attention Is All You Need"""

    # 初始化方法，接受多个参数包括 embed_dim, num_heads, dropout 等
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
        **kwargs,
        # 调用父类初始化函数，传入指定的关键字参数
        super().__init__(**kwargs)
        # 初始化嵌入维度
        self.embed_dim = embed_dim

        # 初始化注意力头数
        self.num_heads = num_heads
        # 初始化dropout层
        self.dropout = keras.layers.Dropout(dropout)
        # 初始化头部维度
        self.head_dim = embed_dim // num_heads
        # 如果头部维度乘以注意力头数不等于嵌入维度，抛出数值错误异常
        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        # 初始化缩放因子
        self.scaling = self.head_dim**-0.5
        # 初始化是否是解码器的标志
        self.is_decoder = is_decoder

        # 初始化k投影层
        self.k_proj = keras.layers.Dense(embed_dim, use_bias=bias, name="k_proj")
        # 初始化q投影层
        self.q_proj = keras.layers.Dense(embed_dim, use_bias=bias, name="q_proj")
        # 初始化v投影层
        self.v_proj = keras.layers.Dense(embed_dim, use_bias=bias, name="v_proj")
        # 初始化输出投影层
        self.out_proj = keras.layers.Dense(embed_dim, use_bias=bias, name="out_proj")

    # 定义变形函数，接收张量、序列长度和批大小作为输入，返回变形后的张量
    def _shape(self, tensor: tf.Tensor, seq_len: int, bsz: int):
        return tf.transpose(tf.reshape(tensor, (bsz, seq_len, self.num_heads, self.head_dim)), (0, 2, 1, 3))

    # 定义调用函数，接收隐藏状态、键值状态、过去的键值对、注意力掩码、层头遮罩和训练标志作为输入
    def call(
        self,
        hidden_states: tf.Tensor,
        key_value_states: tf.Tensor | None = None,
        past_key_value: Tuple[Tuple[tf.Tensor]] | None = None,
        attention_mask: tf.Tensor | None = None,
        layer_head_mask: tf.Tensor | None = None,
        training: Optional[bool] = False,
    # 定义构建函数，接收输入形状作为输入，并在已构建时返回
    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 如果存在k_proj，则构建k_proj
        if getattr(self, "k_proj", None) is not None:
            with tf.name_scope(self.k_proj.name):
                self.k_proj.build([None, None, self.embed_dim])
        # 如果存在q_proj，则构建q_proj
        if getattr(self, "q_proj", None) is not None:
            with tf.name_scope(self.q_proj.name):
                self.q_proj.build([None, None, self.embed_dim])
        # 如果存在v_proj，则构建v_proj
        if getattr(self, "v_proj", None) is not None:
            with tf.name_scope(self.v_proj.name):
                self.v_proj.build([None, None, self.embed_dim])
        # 如果存在out_proj，则构建out_proj
        if getattr(self, "out_proj", None) is not None:
            with tf.name_scope(self.out_proj.name):
                self.out_proj.build([None, None, self.embed_dim])
# 定义一个自定义的 Transformer 编码器层，基于 Keras 的 Layer 类
class TFWav2Vec2EncoderLayer(keras.layers.Layer):
    # 初始化方法，接收 Wav2Vec2Config 类型的配置参数和其他关键字参数
    def __init__(self, config: Wav2Vec2Config, **kwargs):
        super().__init__(**kwargs)
        
        # 初始化注意力机制模块，使用 TFWav2Vec2Attention 类
        self.attention = TFWav2Vec2Attention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=False,
            name="attention",
        )
        
        # Dropout 层，用于隐藏状态的随机失活
        self.dropout = keras.layers.Dropout(config.hidden_dropout)
        
        # LayerNormalization 层，用于归一化层输入，防止梯度爆炸
        self.layer_norm = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="layer_norm")
        
        # 基于配置参数初始化前馈神经网络层
        self.feed_forward = TFWav2Vec2FeedForward(config, name="feed_forward")
        
        # 最终的 LayerNormalization 层，用于归一化前馈神经网络的输出
        self.final_layer_norm = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="final_layer_norm")
        
        # 保存配置参数
        self.config = config

    # 前向传播方法，接收隐藏状态张量和训练标志作为输入，返回处理后的张量
    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: tf.Tensor | None = None,
        output_attentions: Optional[bool] = False,
        training: bool = False,
    ) -> tf.Tensor:
        # 使用注意力机制进行处理
        hidden_states = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            training=training,
        )
        
        # 使用 Dropout 对注意力机制输出进行随机失活
        hidden_states = self.dropout(hidden_states, training=training)
        
        # 应用 LayerNormalization 对随机失活后的隐藏状态进行归一化
        hidden_states = self.layer_norm(hidden_states)
        
        # 使用前馈神经网络处理归一化后的隐藏状态
        hidden_states = self.feed_forward(hidden_states, training=training)
        
        # 最终使用 LayerNormalization 对前馈神经网络的输出进行归一化
        hidden_states = self.final_layer_norm(hidden_states)
        
        # 返回处理后的张量作为编码器层的输出
        return hidden_states
    # 定义函数，该函数接受隐藏状态、注意力掩码和训练标志作为输入，返回包含注意力权重的元组
    ) -> Tuple[tf.Tensor]:
        # 将原始的隐藏状态保存到变量 attn_residual 中
        attn_residual = hidden_states
        # 调用 self.attention 对象进行注意力计算，返回新的隐藏状态、注意力权重和占位符
        hidden_states, attn_weights, _ = self.attention(
            hidden_states, attention_mask=attention_mask, training=training
        )
        # 对隐藏状态应用 dropout 操作，用于正则化
        hidden_states = self.dropout(hidden_states, training=training)
        # 将原始的隐藏状态与新的隐藏状态相加，得到残差连接的结果
        hidden_states = attn_residual + hidden_states

        # 应用层归一化操作
        hidden_states = self.layer_norm(hidden_states)
        # 经过前馈神经网络处理隐藏状态
        hidden_states = hidden_states + self.feed_forward(hidden_states)
        # 再次进行最终层的归一化操作
        hidden_states = self.final_layer_norm(hidden_states)

        # 构建输出元组，只包含隐藏状态
        outputs = (hidden_states,)

        # 如果需要输出注意力权重，则将注意力权重加入输出元组中
        if output_attentions:
            outputs += (attn_weights,)

        # 返回最终的输出元组
        return outputs

    # 定义 build 方法，用于构建模型的层次结构
    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回，避免重复构建
        if self.built:
            return
        # 设置标志位表示已经构建过
        self.built = True

        # 如果存在 self.attention 属性，则构建 attention 层
        if getattr(self, "attention", None) is not None:
            with tf.name_scope(self.attention.name):
                self.attention.build(None)
        
        # 如果存在 self.layer_norm 属性，则构建 layer_norm 层
        if getattr(self, "layer_norm", None) is not None:
            with tf.name_scope(self.layer_norm.name):
                self.layer_norm.build([None, None, self.config.hidden_size])
        
        # 如果存在 self.feed_forward 属性，则构建 feed_forward 层
        if getattr(self, "feed_forward", None) is not None:
            with tf.name_scope(self.feed_forward.name):
                self.feed_forward.build(None)
        
        # 如果存在 self.final_layer_norm 属性，则构建 final_layer_norm 层
        if getattr(self, "final_layer_norm", None) is not None:
            with tf.name_scope(self.final_layer_norm.name):
                self.final_layer_norm.build([None, None, self.config.hidden_size])
class TFWav2Vec2EncoderLayerStableLayerNorm(keras.layers.Layer):
    # TFWav2Vec2EncoderLayerStableLayerNorm 类，继承自 keras.layers.Layer

    def __init__(self, config: Wav2Vec2Config, **kwargs):
        # 初始化函数，接受一个 Wav2Vec2Config 类型的 config 对象和其他关键字参数

        super().__init__(**kwargs)
        # 调用父类的初始化函数

        # 创建注意力层对象
        self.attention = TFWav2Vec2Attention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=False,
            name="attention",
        )
        # 创建 Dropout 层
        self.dropout = keras.layers.Dropout(config.hidden_dropout)
        # 创建 LayerNormalization 层
        self.layer_norm = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="layer_norm")
        # 创建前馈网络层
        self.feed_forward = TFWav2Vec2FeedForward(config, name="feed_forward")
        # 创建最终 LayerNormalization 层
        self.final_layer_norm = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="final_layer_norm")

        # 保存配置对象
        self.config = config

    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: tf.Tensor | None = None,
        output_attentions: Optional[bool] = False,
        training: bool = False,
    ) -> Tuple[tf.Tensor]:
        # 定义 call 方法用于前向传播

        # 保存注意力层的残差连接
        attn_residual = hidden_states
        # LayerNormalization 层
        hidden_states = self.layer_norm(hidden_states)
        # 注意力计算
        hidden_states, attn_weights, _ = self.attention(
            hidden_states, attention_mask=attention_mask, training=training
        )
        # Dropout 层
        hidden_states = self.dropout(hidden_states, training=training)
        # 残差连接
        hidden_states = attn_residual + hidden_states
        # 前馈网络计算
        hidden_states = hidden_states + self.feed_forward(self.final_layer_norm(hidden_states))

        # 返回输出结果
        outputs = (hidden_states,)

        # 如果需要输出注意力权重
        if output_attentions:
            outputs += (attn_weights,)

        return outputs

    def build(self, input_shape=None):
        # 构建函数，用于构建层的参数

        if self.built:
            return

        self.built = True

        # 构建注意力层
        if getattr(self, "attention", None) is not None:
            with tf.name_scope(self.attention.name):
                self.attention.build(None)

        # 构建 LayerNormalization 层
        if getattr(self, "layer_norm", None) is not None:
            with tf.name_scope(self.layer_norm.name):
                self.layer_norm.build([None, None, self.config.hidden_size])

        # 构建前馈网络层
        if getattr(self, "feed_forward", None) is not None:
            with tf.name_scope(self.feed_forward.name):
                self.feed_forward.build(None)

        # 构建最终 LayerNormalization 层
        if getattr(self, "final_layer_norm", None) is not None:
            with tf.name_scope(self.final_layer_norm.name):
                self.final_layer_norm.build([None, None, self.config.hidden_size])


class TFWav2Vec2Encoder(keras.layers.Layer):
    # TFWav2Vec2Encoder 类，继承自 keras.layers.Layer

    def __init__(self, config: Wav2Vec2Config, **kwargs):
        # 初始化函数，接受一个 Wav2Vec2Config 类型的 config 对象和其他关键字参数

        super().__init__(**kwargs)
        # 调用父类的初始化函数

        # 保存配置对象
        self.config = config

        # 创建位置卷积嵌入层
        self.pos_conv_embed = TFWav2Vec2PositionalConvEmbedding(config, name="pos_conv_embed")
        # 创建 LayerNormalization 层
        self.layer_norm = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="layer_norm")
        # 创建 Dropout 层
        self.dropout = keras.layers.Dropout(config.hidden_dropout)

        # 创建多层编码器层列表
        self.layer = [TFWav2Vec2EncoderLayer(config, name=f"layers.{i}") for i in range(config.num_hidden_layers)]
    # 定义一个方法用于处理模型调用过程中的输入和输出
    def call(
        self,
        hidden_states: tf.Tensor,  # 输入的隐藏状态张量
        attention_mask: tf.Tensor | None = None,  # 注意力掩码张量，默认为None
        output_attentions: Optional[bool] = False,  # 是否输出注意力权重，默认为False
        output_hidden_states: Optional[bool] = False,  # 是否输出隐藏状态，默认为False
        return_dict: Optional[bool] = True,  # 是否返回字典格式的输出，默认为True
        training: Optional[bool] = False,  # 是否处于训练模式，默认为False
    ) -> Union[TFBaseModelOutput, Tuple[tf.Tensor]]:
        all_hidden_states = () if output_hidden_states else None  # 如果需要输出隐藏状态，则初始化一个空元组，否则为None
        all_self_attentions = () if output_attentions else None  # 如果需要输出注意力权重，则初始化一个空元组，否则为None

        if attention_mask is not None:
            hidden_states = hidden_states * tf.expand_dims(attention_mask, -1)  # 对隐藏状态应用注意力掩码
            attention_mask = _expand_mask(attention_mask)  # 扩展注意力掩码的维度
        else:
            attention_mask = None  # 如果没有提供注意力掩码，则置为None

        position_embeddings = self.pos_conv_embed(hidden_states)  # 使用位置卷积嵌入处理隐藏状态
        hidden_states = hidden_states + position_embeddings  # 加上位置嵌入的结果
        hidden_states = self.layer_norm(hidden_states)  # 使用层归一化处理隐藏状态
        hidden_states = self.dropout(hidden_states, training=training)  # 应用丢弃操作

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)  # 如果需要输出隐藏状态，则将当前隐藏状态添加到元组中

            # 添加层丢弃（详见 https://arxiv.org/abs/1909.11556）
            dropout_probability = np.random.uniform(0, 1)
            if training and (dropout_probability < self.config.layerdrop):  # 如果处于训练状态且随机数小于层丢弃概率，则跳过当前层
                continue

            layer_outputs = layer_module(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                training=training,
            )
            hidden_states = layer_outputs[0]  # 更新隐藏状态为当前层的输出

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)  # 如果需要输出注意力权重，则将当前层的注意力权重添加到元组中

        # 添加最后一层的隐藏状态输出
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 根据return_dict的设置返回相应的输出格式
        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        return TFBaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )

    # 构建模型，初始化各个组件
    def build(self, input_shape=None):
        if self.built:
            return  # 如果模型已构建，则直接返回
        self.built = True  # 标记模型已构建
        if getattr(self, "pos_conv_embed", None) is not None:
            with tf.name_scope(self.pos_conv_embed.name):
                self.pos_conv_embed.build(None)  # 构建位置卷积嵌入层
        if getattr(self, "layer_norm", None) is not None:
            with tf.name_scope(self.layer_norm.name):
                self.layer_norm.build([None, None, self.config.hidden_size])  # 构建层归一化层
        if getattr(self, "layer", None) is not None:
            for layer in self.layer:
                with tf.name_scope(layer.name):
                    layer.build(None)  # 逐层构建模型的层
class TFWav2Vec2EncoderStableLayerNorm(keras.layers.Layer):
    # 初始化函数，接收配置参数 config 和其他关键字参数
    def __init__(self, config: Wav2Vec2Config, **kwargs):
        super().__init__(**kwargs)
        # 保存配置参数
        self.config = config
        # 创建位置编码卷积嵌入层对象，命名为 pos_conv_embed
        self.pos_conv_embed = TFWav2Vec2PositionalConvEmbedding(config, name="pos_conv_embed")
        # 创建层归一化对象，使用配置中的 epsilon，命名为 layer_norm
        self.layer_norm = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="layer_norm")
        # 创建 dropout 层，使用配置中的隐藏层 dropout 率
        self.dropout = keras.layers.Dropout(config.hidden_dropout)
        # 创建多个编码器层，列表中包含 config.num_hidden_layers 个 TFWav2Vec2EncoderLayerStableLayerNorm 实例
        self.layer = [
            TFWav2Vec2EncoderLayerStableLayerNorm(config, name=f"layers.{i}") for i in range(config.num_hidden_layers)
        ]

    # 前向传播函数，接收隐藏状态、注意力掩码和其他控制参数，返回 TFBaseModelOutput 或元组 tf.Tensor
    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: tf.Tensor | None = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
        training: Optional[bool] = False,
    ) -> Union[TFBaseModelOutput, Tuple[tf.Tensor]]:
        # 如果需要输出隐藏状态，则初始化空元组 all_hidden_states，否则设为 None
        all_hidden_states = () if output_hidden_states else None
        # 如果需要输出注意力权重，则初始化空元组 all_self_attentions，否则设为 None
        all_self_attentions = () if output_attentions else None

        # 如果存在 attention_mask，则将隐藏状态与 attention_mask 相乘，实现掩码效果，并使用 _expand_mask 对 attention_mask 进行扩展
        if attention_mask is not None:
            hidden_states = hidden_states * tf.expand_dims(attention_mask, -1)
            attention_mask = _expand_mask(attention_mask)
        else:
            attention_mask = None

        # 计算位置编码并添加到隐藏状态中
        position_embeddings = self.pos_conv_embed(hidden_states)
        hidden_states = hidden_states + position_embeddings
        # 对隐藏状态应用 dropout，根据训练状态进行区分
        hidden_states = self.dropout(hidden_states, training=training)

        # 遍历每个编码器层
        for i, layer_module in enumerate(self.layer):
            # 如果需要输出隐藏状态，则将当前隐藏状态加入 all_hidden_states
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 添加 LayerDrop 机制，根据配置中的 layerdrop 参数跳过某些层
            dropout_probability = np.random.uniform(0, 1)
            if training and (dropout_probability < self.config.layerdrop):  # 根据概率跳过该层
                continue

            # 调用当前编码器层进行前向传播
            layer_outputs = layer_module(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                training=training,
            )
            # 更新隐藏状态为编码器层的输出
            hidden_states = layer_outputs[0]

            # 如果需要输出注意力权重，则将当前层的注意力权重加入 all_self_attentions
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        # 对最终的隐藏状态进行层归一化
        hidden_states = self.layer_norm(hidden_states)

        # 如果需要输出隐藏状态，则将最终的隐藏状态加入 all_hidden_states
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 如果 return_dict 为 False，则返回非 None 的结果元组
        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        # 否则，返回 TFBaseModelOutput 对象，包含最终隐藏状态、所有隐藏状态和所有注意力权重
        return TFBaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )
    # 定义神经网络层的构建方法，如果已经构建过则直接返回
    def build(self, input_shape=None):
        if self.built:
            return
        # 设置标志位，表示网络已经构建
        self.built = True

        # 如果存在位置卷积嵌入层，则构建该层
        if getattr(self, "pos_conv_embed", None) is not None:
            # 在 TensorFlow 中设置命名空间为位置卷积嵌入层的名称，并进行构建
            with tf.name_scope(self.pos_conv_embed.name):
                self.pos_conv_embed.build(None)

        # 如果存在层归一化层，则构建该层
        if getattr(self, "layer_norm", None) is not None:
            # 在 TensorFlow 中设置命名空间为层归一化层的名称，并进行构建
            with tf.name_scope(self.layer_norm.name):
                self.layer_norm.build([None, None, self.config.hidden_size])

        # 如果存在多个子层，则依次构建每个子层
        if getattr(self, "layer", None) is not None:
            for layer in self.layer:
                # 在 TensorFlow 中设置命名空间为子层的名称，并进行构建
                with tf.name_scope(layer.name):
                    layer.build(None)
# 使用 Keras 序列化装饰器标记该类可以被序列化
@keras_serializable
class TFWav2Vec2MainLayer(keras.layers.Layer):
    # 指定配置类为 Wav2Vec2Config
    config_class = Wav2Vec2Config

    # 初始化函数，接受配置对象作为参数，初始化各个子层
    def __init__(self, config: Wav2Vec2Config, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        # 创建特征提取器对象，使用给定的配置对象，并命名为 "feature_extractor"
        self.feature_extractor = TFWav2Vec2FeatureEncoder(config, name="feature_extractor")
        # 创建特征投影对象，使用给定的配置对象，并命名为 "feature_projection"
        self.feature_projection = TFWav2Vec2FeatureProjection(config, name="feature_projection")

        # 根据配置选择稳定层归一化编码器或一般编码器
        if config.do_stable_layer_norm:
            self.encoder = TFWav2Vec2EncoderStableLayerNorm(config, name="encoder")
        else:
            self.encoder = TFWav2Vec2Encoder(config, name="encoder")

    # 构建函数，构建该层的权重和子层
    def build(self, input_shape=None):
        # 如果已经构建过，直接返回
        if self.built:
            return
        self.built = True

        # 如果配置中设置了时间掩码或特征掩码的概率大于0，则添加用于掩码的权重
        if self.config.mask_time_prob > 0.0 or self.config.mask_feature_prob > 0.0:
            self.masked_spec_embed = self.add_weight(
                shape=(self.config.hidden_size,),  # 形状为隐藏尺寸大小的一维向量
                initializer="uniform",  # 使用均匀分布初始化权重
                trainable=True,  # 可训练
                name="masked_spec_embed"  # 权重的名称为 "masked_spec_embed"
            )

        # 如果存在特征提取器对象，则构建特征提取器
        if getattr(self, "feature_extractor", None) is not None:
            with tf.name_scope(self.feature_extractor.name):
                self.feature_extractor.build(None)

        # 如果存在特征投影对象，则构建特征投影
        if getattr(self, "feature_projection", None) is not None:
            with tf.name_scope(self.feature_projection.name):
                self.feature_projection.build(None)

        # 如果存在编码器对象，则构建编码器
        if getattr(self, "encoder", None) is not None:
            with tf.name_scope(self.encoder.name):
                self.encoder.build(None)

    # 计算卷积层的输出长度
    def _get_feat_extract_output_lengths(self, input_lengths: tf.Tensor):
        """
        Computes the output length of the convolutional layers
        """

        def _conv_out_length(input_length, kernel_size, stride):
            # 1D 卷积层的输出长度公式，参考自 https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
            return (input_length - kernel_size) // stride + 1

        # 对于每个卷积核大小和步长，依次计算输出长度
        for kernel_size, stride in zip(self.config.conv_kernel, self.config.conv_stride):
            input_lengths = _conv_out_length(input_lengths, kernel_size, stride)

        return input_lengths
        # 掩盖隐藏状态，根据时间索引和/或特征索引掩盖提取的特征，根据[SpecAugment](https://arxiv.org/abs/1904.08779)进行操作
        def _mask_hidden_states(self, hidden_states: tf.Tensor, mask_time_indices: tf.Tensor | None = None):
            """
            Masks extracted features along time axis and/or along feature axis according to
            [SpecAugment](https://arxiv.org/abs/1904.08779).
            """
            # 获取隐藏状态的形状
            batch_size, sequence_length, hidden_size = shape_list(hidden_states)

            # 如果config.apply_spec_augment设置为False，则不进行掩盖操作
            if not getattr(self.config, "apply_spec_augment", True):
                return hidden_states

            # 如果传入了mask_time_indices
            if mask_time_indices is not None:
                # 根据给定的mask_time_indices沿时间轴应用SpecAugment掩盖
                hidden_states = tf.where(
                    tf.cast(mask_time_indices[:, :, tf.newaxis], tf.bool),
                    self.masked_spec_embed[tf.newaxis, tf.newaxis, :],
                    hidden_states,
                )

            # 如果未传入mask_time_indices，并且mask_time_prob大于0
            elif self.config.mask_time_prob > 0:
                # 生成索引并沿时间轴应用SpecAugment
                mask_time_indices = _compute_mask_indices(
                    (batch_size, sequence_length),
                    mask_prob=self.config.mask_time_prob,
                    mask_length=self.config.mask_time_length,
                    min_masks=2,
                )
                hidden_states = tf.where(
                    tf.cast(mask_time_indices[:, :, tf.newaxis], tf.bool),
                    self.masked_spec_embed[tf.newaxis, tf.newaxis, :],
                    hidden_states,
                )

            # 沿特征轴应用SpecAugment
            if self.config.mask_feature_prob > 0:
                mask_feature_indices = _compute_mask_indices(
                    (batch_size, hidden_size),
                    mask_prob=self.config.mask_feature_prob,
                    mask_length=self.config.mask_feature_length,
                )
                hidden_states = tf.where(mask_feature_indices[:, tf.newaxis, :], hidden_states, 0)

            return hidden_states

        # 解包输入参数并调用模型
        @unpack_inputs
        def call(
            self,
            input_values: tf.Tensor,
            attention_mask: tf.Tensor | None = None,
            token_type_ids: tf.Tensor | None = None,
            position_ids: tf.Tensor | None = None,
            head_mask: tf.Tensor | None = None,
            inputs_embeds: tf.Tensor | None = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            training: bool = False,
            **kwargs: Any,
        ):
        # 使用特征提取器从输入值中提取特征，返回特征张量
        extract_features = self.feature_extractor(tf.cast(input_values, tf.float32), training=training)
        # 如果需要，可以转置提取的特征张量的维度顺序
        # extract_features = tf.transpose(extract_features, perm=(0, 2, 1))

        if attention_mask is not None:
            # 根据卷积公式计算真实的输出长度
            output_lengths = self._get_feat_extract_output_lengths(tf.reduce_sum(attention_mask, -1))

            # 根据计算得到的长度创建注意力掩码
            attention_mask = tf.sequence_mask(
                output_lengths, maxlen=shape_list(extract_features)[1], dtype=extract_features.dtype
            )

        # 将提取的特征张量投影到隐藏状态空间中
        hidden_states, extract_features = self.feature_projection(extract_features, training=training)

        # 获取可选参数中的时间索引屏蔽信息
        mask_time_indices = kwargs.get("mask_time_indices", None)
        if training:
            # 如果处于训练模式，则对隐藏状态进行时间屏蔽处理
            hidden_states = self._mask_hidden_states(hidden_states, mask_time_indices=mask_time_indices)

        # 将隐藏状态输入到编码器中进行编码
        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )
        # 从编码器输出中获取隐藏状态
        hidden_states = encoder_outputs[0]

        # 如果不返回字典形式的结果，则返回一个包含隐藏状态、提取的特征和其他编码器输出的元组
        if not return_dict:
            return (hidden_states, extract_features) + encoder_outputs[1:]

        # 返回一个包含 TF Wav2Vec2 模型输出的命名元组
        return TFWav2Vec2BaseModelOutput(
            last_hidden_state=hidden_states,
            extract_features=extract_features,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 使用 Wav2Vec2Config 类作为配置类
    config_class = Wav2Vec2Config
    # 基础模型的前缀字符串
    base_model_prefix = "wav2vec2"
    # 主输入的名称
    main_input_name = "input_values"

    @property
    def input_signature(self):
        # 定义模型输入的签名，包括 input_values 和 attention_mask
        return {
            "input_values": tf.TensorSpec((None, None), tf.float32, name="input_values"),
            "attention_mask": tf.TensorSpec((None, None), tf.float32, name="attention_mask"),
        }

    @property
    def dummy_inputs(self):
        # 返回一个示例的输入字典，包含随机生成的 input_values 和全为1的 attention_mask
        return {
            "input_values": tf.random.uniform(shape=(1, 500), dtype=tf.float32),
            "attention_mask": tf.ones(shape=(1, 500), dtype=tf.float32),
        }

    def __init__(self, config, *inputs, **kwargs):
        # 调用父类的构造方法，并打印警告信息，指出CPU上不支持反向传播操作，需要使用GPU或TPU进行训练/微调
        super().__init__(config, *inputs, **kwargs)
        logger.warning(
            f"\n{self.__class__.__name__} has backpropagation operations that are NOT supported on CPU. If you wish "
            "to train/fine-tune this model, you need a GPU or a TPU"
        )

    def _get_feat_extract_output_lengths(self, input_lengths, add_adapter=None):
        """
        Computes the output length of the convolutional layers
        """
        # 如果 add_adapter 未提供，则使用配置中的 add_adapter 参数
        add_adapter = self.config.add_adapter if add_adapter is None else add_adapter

        def _conv_out_length(input_length, kernel_size, stride):
            # 计算卷积层的输出长度
            return tf.math.floordiv(input_length - kernel_size, stride) + 1

        # 对每个卷积核和步长进行迭代，更新输入长度
        for kernel_size, stride in zip(self.config.conv_kernel, self.config.conv_stride):
            input_lengths = _conv_out_length(input_lengths, kernel_size, stride)

        # 如果配置中启用了 adapter layers，则对每个 adapter layer 同样计算输出长度
        if add_adapter:
            for _ in range(self.config.num_adapter_layers):
                input_lengths = _conv_out_length(input_lengths, 1, self.config.adapter_stride)
        return input_lengths

    def _get_feature_vector_attention_mask(
        self, feature_vector_length: int, attention_mask: tf.Tensor, add_adapter=None
        )
            # 计算非填充长度，即每个样本序列的实际长度
            non_padded_lengths = tf.math.cumsum(attention_mask, axis=-1)[:, -1]
            # 获取特征提取器输出的长度，考虑是否添加适配器
            output_lengths = self._get_feat_extract_output_lengths(non_padded_lengths, add_adapter=add_adapter)
            output_lengths = tf.cast(output_lengths, tf.int32)
            batch_size = tf.shape(attention_mask)[0]
            # 检查设备位置
            attention_mask = tf.zeros(
                (batch_size, feature_vector_length), dtype=attention_mask.dtype, name="attention_mask"
            )  # 这两个操作确保输出长度之前的所有位置都被注意到
            ## 检查设备
            attention_mask = tf.tensor_scatter_nd_update(
                attention_mask,
                indices=tf.stack([tf.range(batch_size), output_lengths - 1], axis=1),
                updates=tf.ones([batch_size], dtype=attention_mask.dtype),
            )
            attention_mask = tf.reverse(attention_mask, axis=[-1])
            attention_mask = tf.cumsum(attention_mask, axis=-1)
            attention_mask = tf.reverse(attention_mask, axis=[-1])
            attention_mask = tf.cast(attention_mask, tf.bool)
            return attention_mask
"""
This model inherits from `TFPreTrainedModel`. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a `keras.Model` subclass. Use it as a regular TF 2.0 Keras Model and refer to the TF 2.0
documentation for all matters related to general usage and behavior.

<Tip>

TensorFlow models and layers in `transformers` accept two formats as input:

- having all inputs as keyword arguments (like PyTorch models), or
- having all inputs as a list, tuple or dict in the first positional argument.

The reason the second format is supported is that Keras methods prefer this format when passing inputs to models
and layers. Because of this support, when using methods like `model.fit()` things should "just work" for you - just
pass your inputs and labels in any format that `model.fit()` supports! If, however, you want to use the second
format outside of Keras methods like `fit()` and `predict()`, such as when creating your own layers or models with
the Keras `Functional` API, there are three possibilities you can use to gather all the input Tensors in the first
positional argument:

- a single Tensor with `input_values` only and nothing else: `model(input_values)`
- a list of varying length with one or several input Tensors IN THE ORDER given in the docstring:
`model([input_values, attention_mask])` or `model([input_values, attention_mask, token_type_ids])`
- a dictionary with one or several input Tensors associated to the input names given in the docstring:
`model({"input_values": input_values, "token_type_ids": token_type_ids})`

Note that when creating models and layers with
[subclassing](https://keras.io/guides/making_new_layers_and_models_via_subclassing/) then you don't need to worry
about any of this, as you can just pass inputs like you would to any other Python function!

</Tip>

Args:
    config ([`Wav2Vec2Config`]): Model configuration class with all the parameters of the model.
        Initializing with a config file does not load the weights associated with the model, only the
        configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""
WAV_2_VEC_2_START_DOCSTRING = r"""

    This model inherits from [`TFPreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a [keras.Model](https://www.tensorflow.org/api_docs/python/tf/keras/Model) subclass. Use it
    as a regular TF 2.0 Keras Model and refer to the TF 2.0 documentation for all matter related to general usage and
    behavior.

    <Tip>

    TensorFlow models and layers in `transformers` accept two formats as input:

    - having all inputs as keyword arguments (like PyTorch models), or
    - having all inputs as a list, tuple or dict in the first positional argument.

    The reason the second format is supported is that Keras methods prefer this format when passing inputs to models
    and layers. Because of this support, when using methods like `model.fit()` things should "just work" for you - just
    pass your inputs and labels in any format that `model.fit()` supports! If, however, you want to use the second
    format outside of Keras methods like `fit()` and `predict()`, such as when creating your own layers or models with
    the Keras `Functional` API, there are three possibilities you can use to gather all the input Tensors in the first
    positional argument:

    - a single Tensor with `input_values` only and nothing else: `model(input_values)`
    - a list of varying length with one or several input Tensors IN THE ORDER given in the docstring:
    `model([input_values, attention_mask])` or `model([input_values, attention_mask, token_type_ids])`
    - a dictionary with one or several input Tensors associated to the input names given in the docstring:
    `model({"input_values": input_values, "token_type_ids": token_type_ids})`

    Note that when creating models and layers with
    [subclassing](https://keras.io/guides/making_new_layers_and_models_via_subclassing/) then you don't need to worry
    about any of this, as you can just pass inputs like you would to any other Python function!

    </Tip>

    Args:
        config ([`Wav2Vec2Config`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

"""
"""

@add_start_docstrings(
    "The bare TFWav2Vec2 Model transformer outputing raw hidden-states without any specific head on top.",
    WAV_2_VEC_2_START_DOCSTRING,
)
class TFWav2Vec2Model(TFWav2Vec2PreTrainedModel):
    """
    The bare TFWav2Vec2 Model transformer outputting raw hidden-states without any specific head on top.

    This class inherits from `TFWav2Vec2PreTrainedModel` and includes additional documentation provided by
    `WAV_2_VEC_2_START_DOCSTRING`.

    Args:
        config (Wav2Vec2Config): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the `PreTrainedModel.from_pretrained` method to load the model weights.
    """

    def __init__(self, config: Wav2Vec2Config, *inputs, **kwargs):
        """
        Initializes a TFWav2Vec2Model instance.

        Args:
            config (Wav2Vec2Config): Model configuration class with all the parameters of the model.
                Initializing with a config file does not load the weights associated with the model, only the
                configuration. Check out the PreTrainedModel.from_pretrained method to load the model weights.
            *inputs: Additional positional arguments to be passed.
            **kwargs: Additional keyword arguments to be passed.
        """
        super().__init__(config, *inputs, **kwargs)
        self.config = config
        self.wav2vec2 = TFWav2Vec2MainLayer(config, name="wav2vec2")
    # 将模型的文档字符串添加到前向方法中，用于描述模型输入
    @add_start_docstrings_to_model_forward(WAV_2_VEC_2_INPUTS_DOCSTRING)
    # 替换返回值的文档字符串，指定输出类型和配置类
    @replace_return_docstrings(output_type=TFBaseModelOutput, config_class=_CONFIG_FOR_DOC)
    # 解包输入参数，使其作为独立参数传递给 call 方法
    @unpack_inputs
    def call(
        self,
        input_values: tf.Tensor,
        attention_mask: tf.Tensor | None = None,
        token_type_ids: tf.Tensor | None = None,
        position_ids: tf.Tensor | None = None,
        head_mask: tf.Tensor | None = None,
        inputs_embeds: tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
    ) -> Union[TFBaseModelOutput, Tuple[tf.Tensor]]:
        """

        Returns:

        Example:

        ```
        >>> from transformers import AutoProcessor, TFWav2Vec2Model
        >>> from datasets import load_dataset
        >>> import soundfile as sf

        >>> processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base-960h")
        >>> model = TFWav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")


        >>> def map_to_array(batch):
        ...     speech, _ = sf.read(batch["file"])
        ...     batch["speech"] = speech
        ...     return batch


        >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        >>> ds = ds.map(map_to_array)

        >>> input_values = processor(ds["speech"][0], return_tensors="tf").input_values  # Batch size 1
        >>> hidden_states = model(input_values).last_hidden_state
        ```"""

        # 设置是否输出隐藏状态，默认使用配置类中的设定
        output_hidden_states = output_hidden_states if output_hidden_states else self.config.output_hidden_states
        # 设置是否输出注意力权重，默认使用配置类中的设定
        output_attentions = output_attentions if output_attentions else self.config.output_attentions
        # 设置是否返回字典形式的输出，默认使用配置类中的设定
        return_dict = return_dict if return_dict else self.config.return_dict

        # 调用 wav2vec2 模型的前向计算方法，传递所有参数
        outputs = self.wav2vec2(
            input_values=input_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        # 返回模型输出
        return outputs

    def build(self, input_shape=None):
        # 如果已经构建，则直接返回
        if self.built:
            return
        # 标记模型已构建
        self.built = True
        # 如果存在 wav2vec2 模型，使用其名称为命名空间构建模型
        if getattr(self, "wav2vec2", None) is not None:
            with tf.name_scope(self.wav2vec2.name):
                self.wav2vec2.build(None)
@add_start_docstrings(
    """TFWav2Vec2 Model with a `language modeling` head on top for Connectionist Temporal Classification (CTC).""",
    WAV_2_VEC_2_START_DOCSTRING,
)
class TFWav2Vec2ForCTC(TFWav2Vec2PreTrainedModel):
    def __init__(self, config: Wav2Vec2Config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        self.wav2vec2 = TFWav2Vec2MainLayer(config, name="wav2vec2")  # 初始化 TF-Wav2Vec2 主层
        self.dropout = keras.layers.Dropout(config.final_dropout)  # 添加丢弃层，使用给定的丢弃率
        self.lm_head = keras.layers.Dense(config.vocab_size, name="lm_head")  # 初始化语言模型头部密集层
        self.output_hidden_size = (
            config.output_hidden_size if hasattr(config, "add_adapter") and config.add_adapter else config.hidden_size
        )  # 设置输出隐藏尺寸为配置中的特定值或者隐藏尺寸

    def freeze_feature_extractor(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameters will
        not be updated during training.
        """
        warnings.warn(
            "The method `freeze_feature_extractor` is deprecated and will be removed in Transformers v5. "
            "Please use the equivalent `freeze_feature_encoder` method instead.",
            FutureWarning,
        )
        self.freeze_feature_encoder()  # 警告过时方法，调用等效的特征编码器冻结方法

    def freeze_feature_encoder(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        self.wav2vec2.feature_extractor.trainable = False  # 冻结特征编码器，禁止在训练过程中更新其参数

    @unpack_inputs
    @add_start_docstrings_to_model_forward(WAV_2_VEC_2_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFCausalLMOutput, config_class=_CONFIG_FOR_DOC)
    def call(
        self,
        input_values: tf.Tensor,
        attention_mask: tf.Tensor | None = None,
        token_type_ids: tf.Tensor | None = None,
        position_ids: tf.Tensor | None = None,
        head_mask: tf.Tensor | None = None,
        inputs_embeds: tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        labels: tf.Tensor | None = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: Optional[bool] = False,
    ):
        """
        Call function to perform forward pass of the model. This function integrates with the `transformers` library's
        `add_start_docstrings_to_model_forward` decorator to provide structured documentation for inputs and outputs.
        """
        # 实现模型的前向传播，结合 `transformers` 库的 `add_start_docstrings_to_model_forward` 装饰器以提供输入和输出的结构化文档

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "wav2vec2", None) is not None:
            with tf.name_scope(self.wav2vec2.name):
                self.wav2vec2.build(None)  # 构建 TF-Wav2Vec2 主层
        if getattr(self, "lm_head", None) is not None:
            with tf.name_scope(self.lm_head.name):
                self.lm_head.build([None, None, self.output_hidden_size])  # 构建语言模型头部密集层
    def __init__(self, config):
        super().__init__(config)
        # 初始化函数，调用父类构造函数，并初始化相关属性
        self.wav2vec2 = TFWav2Vec2MainLayer(config, name="wav2vec2")
        # 创建一个名为wav2vec2的TFWav2Vec2MainLayer实例，并赋给self.wav2vec2
        self.num_layers = config.num_hidden_layers + 1
        # 设置self.num_layers为config.num_hidden_layers加一
        with tf.name_scope(self._name_scope()):
            # 使用当前对象的命名空间创建一个上下文管理器
            if config.use_weighted_layer_sum:
                # 如果配置中使用加权层求和
                self.layer_weights = self.add_weight(
                    shape=(self.num_layers,), initializer="ones", trainable=True, name="layer_weights"
                )
                # 添加名为layer_weights的权重，形状为(self.num_layers,)，初始化为全1，可训练
        self.config = config
        # 将配置对象保存在self.config中
        self.projector = keras.layers.Dense(units=config.classifier_proj_size, name="projector")
        # 创建一个全连接层，单元数为config.classifier_proj_size，名为projector
        self.classifier = keras.layers.Dense(units=config.num_labels, activation=None, name="classifier")
        # 创建一个全连接层，单元数为config.num_labels，激活函数为None，名为classifier

    def freeze_feature_extractor(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameters will
        not be updated during training.
        """
        # 弃用警告：freeze_feature_extractor方法将在Transformers v5中移除，请使用等效的freeze_feature_encoder方法
        warnings.warn(
            "The method `freeze_feature_extractor` is deprecated and will be removed in Transformers v5. "
            "Please use the equivalent `freeze_feature_encoder` method instead.",
            FutureWarning,
        )
        # 发出警告信息，提醒用户方法即将被移除
        self.freeze_feature_encoder()
        # 调用freeze_feature_encoder方法，禁用特征编码器的梯度计算，使其参数在训练过程中不更新

    def freeze_feature_encoder(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        # 调用此函数将禁用特征编码器的梯度计算，使其参数在训练过程中不更新
        self.wav2vec2.feature_extractor.trainable = False

    def freeze_base_model(self):
        """
        Calling this function will disable the gradient computation for the base model so that its parameters will not
        be updated during training. Only the classification head will be updated.
        """
        # 调用此函数将禁用基础模型的梯度计算，使其参数在训练过程中不更新，只有分类头将被更新
        for layer in self.wav2vec2.layers:
            # 遍历self.wav2vec2的所有层
            layer.trainable = False
            # 设置每一层的trainable属性为False，即不可训练状态

    @unpack_inputs
    def call(
        self,
        input_values: tf.Tensor,
        attention_mask: tf.Tensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        labels: tf.Tensor | None = None,
        training: bool = False,
    ) -> TFSequenceClassifierOutput | Tuple[tf.Tensor]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 如果 return_dict 为 None，则使用 self.config.use_return_dict 的值
        output_hidden_states = True if self.config.use_weighted_layer_sum else output_hidden_states
        # 如果 self.config.use_weighted_layer_sum 为 True，则设置 output_hidden_states 为 True

        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )
        # 调用 self.wav2vec2 模型，传入参数并获取输出

        if self.config.use_weighted_layer_sum:
            hidden_states = outputs[_HIDDEN_STATES_START_POSITION]
            # 获取权重层求和后的隐藏状态
            hidden_states = tf.stack(hidden_states, axis=1)
            # 在第二个维度上堆叠隐藏状态
            norm_weights = tf.nn.softmax(self.layer_weights, axis=-1)
            # 对权重进行 softmax 归一化
            hidden_states = tf.reduce_sum(hidden_states * tf.reshape(norm_weights, [-1, 1, 1]), axis=1)
            # 使用归一化的权重对隐藏状态进行加权求和
        else:
            hidden_states = outputs[0]
            # 否则直接使用模型输出的第一个元素作为隐藏状态

        hidden_states = self.projector(hidden_states)
        # 将隐藏状态投影到指定维度

        if attention_mask is None:
            pooled_output = tf.reduce_mean(hidden_states, axis=1)
            # 如果注意力掩码为 None，则对隐藏状态进行平均池化
        else:
            padding_mask = self._get_feature_vector_attention_mask(shape_list(hidden_states)[1], attention_mask)
            # 获取特征向量注意力掩码
            padding_mask_float = tf.cast(padding_mask, hidden_states.dtype)
            # 将掩码转换为浮点类型
            hidden_states = tf.multiply(hidden_states, tf.expand_dims(padding_mask_float, axis=-1))
            # 使用掩码进行元素级乘法
            pooled_output = tf.divide(
                tf.reduce_sum(hidden_states, axis=1), tf.expand_dims(tf.reduce_sum(padding_mask_float, axis=1), axis=1)
            )
            # 使用掩码对隐藏状态进行加权求和并进行平均池化

        logits = self.classifier(pooled_output)
        # 使用分类器对池化输出进行分类预测

        loss = None
        if labels is not None:
            loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
            # 使用稀疏分类交叉熵作为损失函数
            loss = loss_fn(tf.reshape(labels, [-1]), tf.reshape(logits, [-1, self.config.num_labels]))
            # 计算损失值

        if not return_dict:
            output = (logits,) + outputs[_HIDDEN_STATES_START_POSITION:]
            # 构建输出元组
            return ((loss,) + output) if loss is not None else output
            # 返回损失和输出元组，如果没有损失则返回输出元组

        return TFSequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        # 返回 TFSequenceClassifierOutput 对象，包括损失、预测 logits、隐藏状态和注意力
```