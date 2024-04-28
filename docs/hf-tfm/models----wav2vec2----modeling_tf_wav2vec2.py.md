# `.\transformers\models\wav2vec2\modeling_tf_wav2vec2.py`

```
# 导入必要的库
from __future__ import annotations
import warnings
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union
import numpy as np
import tensorflow as tf

# 导入相关模块和函数
from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import TFBaseModelOutput, TFCausalLMOutput, TFSequenceClassifierOutput
from ...modeling_tf_utils import (
    TFPreTrainedModel,
    get_initializer,
    keras_serializable,
    unpack_inputs,
)
from ...tf_utils import shape_list, stable_softmax
from ...utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from .configuration_wav2vec2 import Wav2Vec2Config

# 设置日志记录器
logger = logging.get_logger(__name__)

# 隐藏状态的开始位置，默认为2
_HIDDEN_STATES_START_POSITION = 2

# 用于文档的checkpoint和配置
_CHECKPOINT_FOR_DOC = "facebook/wav2vec2-base-960h"
_CONFIG_FOR_DOC = "Wav2Vec2Config"

# 预训练模型的存档列表
TF_WAV_2_VEC_2_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/wav2vec2-base-960h",
    "facebook/wav2vec2-large-960h",
    "facebook/wav2vec2-large-960h-lv60",
    "facebook/wav2vec2-large-960h-lv60-self",
    # See all Wav2Vec2 models at https://huggingface.co/models?filter=wav2vec2
]

# 定义一个数据类，表示模型输出
@dataclass
class TFWav2Vec2BaseModelOutput(ModelOutput):
    """
    Output type of [`TFWav2Vec2BaseModelOutput`], with potential hidden states and attentions.
    # 定义函数参数说明
    Args:
        # 最后一层模型的隐藏状态，形状为(batch_size, sequence_length, hidden_size)
        last_hidden_state (`tf.Tensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        # 模型最后一个卷积层提取的特征向量序列，形状为(batch_size, sequence_length, conv_dim[-1])
        extract_features (`tf.Tensor` of shape `(batch_size, sequence_length, conv_dim[-1])`):
            Sequence of extracted feature vectors of the last convolutional layer of the model.
        # 当`output_hidden_states=True`或`config.output_hidden_states=True`时返回的隐藏状态，形状为(batch_size, sequence_length, hidden_size)
        hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `tf.Tensor` (one for the output of the embeddings + one for the output of each layer) of shape
            `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        # 当`output_attentions=True`或`config.output_attentions=True`时返回的注意力权重，形状为(batch_size, num_heads, sequence_length, sequence_length)
        attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    # 最后一层隐藏状态，默认为None
    last_hidden_state: tf.Tensor = None
    # 提取的特征向量，默认为None
    extract_features: tf.Tensor = None
    # 隐藏状态，默认为None
    hidden_states: Tuple[tf.Tensor] | None = None
    # 注意力权重，默认为None
    attentions: Tuple[tf.Tensor] | None = None
# 从分布中进行无替换的类别抽样，目前尚未实现。暂时使用 Gumbel-max 技巧 - 参见 https://github.com/tensorflow/tensorflow/issues/9260 了解更多信息
def _sample_without_replacement(distribution, num_samples):
    # 通过 Gumbel-max 技巧进行采样
    z = -tf.math.log(tf.random.uniform(shape_list(distribution), 0, 1))
    # 获取分布和 z 的和的前 num_samples 个最大值的索引
    _, indices = tf.nn.top_k(distribution + z, num_samples)
    # 返回索引
    return indices


# 在批次索引上散布值，类似于 PyTorch 的 scatter 函数，其中索引格式为 (batch_dim, indices)
def _scatter_values_on_batch_indices(values, batch_indices, output_shape):
    # 获取批次索引的形状
    indices_shape = shape_list(batch_indices)
    # 将批次维度广播到 indices_shape
    broad_casted_batch_dims = tf.reshape(
        tf.broadcast_to(tf.expand_dims(tf.range(indices_shape[0]), axis=-1), indices_shape), [1, -1]
    )
    # 将 batch_indices 转换为 pair_indices
    pair_indices = tf.transpose(tf.concat([broad_casted_batch_dims, tf.reshape(batch_indices, [1, -1])], 0))
    # 将值散布到 pair_indices
    return tf.scatter_nd(pair_indices, tf.reshape(values, [-1]), output_shape)


# 计算给定形状的随机掩码范围
def _compute_mask_indices(
    shape: Tuple[int, int],
    mask_prob: float,
    mask_length: int,
    min_masks: int = 0,
) -> tf.Tensor:
    # 获取形状的批次大小和序列长度
    batch_size, sequence_length = shape

    # 如果 mask_length 小于 1，则引发 ValueError
    if mask_length < 1:
        raise ValueError("`mask_length` has to be bigger than 0.")

    # 断言 mask_length 小于 sequence_length
    tf.debugging.assert_less(
        mask_length,
        sequence_length,
        message=(
            f"`mask_length` has to be smaller than `sequence_length`, but got `mask_length`: {mask_length} and"
            f" `sequence_length`: {sequence_length}`"
        ),
    )

    # 计算批次中的掩码范围数量
    num_masked_spans = mask_prob * tf.cast(sequence_length, tf.float32) / mask_length + tf.random.uniform((1,))
    num_masked_spans = tf.maximum(num_masked_spans, min_masks)
    num_masked_spans = tf.cast(num_masked_spans, tf.int32)

    # 确保掩码索引数小于等于序列长度
    # 计算需要被mask的区间数量，取sequence_length除以mask_length的商和num_masked_spans中较小的值
    num_masked_spans = tf.math.minimum(sequence_length // mask_length, num_masked_spans)
    # 去除维度为1的维度
    num_masked_spans = tf.squeeze(num_masked_spans)

    # 创建用于SpecAugment的mask矩阵，全零矩阵，维度为(batch_size, sequence_length)
    spec_aug_mask = tf.zeros((batch_size, sequence_length), dtype=tf.int32)

    # 创建一个全为1的矩阵，用于采样随机索引，确保offset的采样结果小于sequence_length
    uniform_dist = tf.ones((batch_size, sequence_length - (mask_length - 1)))

    # 根据uniform_dist的分布进行随机采样，获取要mask的随机索引
    spec_aug_mask_idxs = _sample_without_replacement(uniform_dist, num_masked_spans)

    # 将mask的索引扩展为mask的区间
    spec_aug_mask_idxs = tf.expand_dims(spec_aug_mask_idxs, -1)
    spec_aug_mask_idxs = tf.tile(spec_aug_mask_idxs, (1, 1, mask_length))
    spec_aug_mask_idxs = tf.reshape(spec_aug_mask_idxs, (batch_size, num_masked_spans * mask_length))

    # 生成mask的offset
    offsets = tf.range(mask_length)[tf.newaxis, tf.newaxis, :]
    offsets = tf.tile(offsets, (batch_size, num_masked_spans, 1))
    offsets = tf.reshape(offsets, (batch_size, num_masked_spans * mask_length))

    # 根据offset添加到mask的索引上
    spec_aug_mask_idxs = spec_aug_mask_idxs + offsets

    # 将mask索引按指定规则散播到spec_aug_mask上
    spec_aug_mask = _scatter_values_on_batch_indices(
        tf.ones_like(spec_aug_mask_idxs), spec_aug_mask_idxs, tf.shape(spec_aug_mask)
    )

    # 返回生成的spec_aug_mask
    return spec_aug_mask
# 从transformers.models.bart.modeling_tf_bart._expand_mask复制过来的函数
def _expand_mask(mask: tf.Tensor, tgt_len: Optional[int] = None):
    """
    将注意力掩码从`[bsz, seq_len]`扩展到`[bsz, 1, tgt_seq_len, src_seq_len]`。
    """
    # 获取掩码的源序列长度
    src_len = shape_list(mask)[1]
    # 如果未提供目标长度，则使用源序列长度
    tgt_len = tgt_len if tgt_len is not None else src_len
    #创建一个常数张量，值为1.0
    one_cst = tf.constant(1.0)
    #将掩码转换为常数张量的数据类型
    mask = tf.cast(mask, dtype=one_cst.dtype)
    #将掩码沿指定维度进行复制扩展
    expanded_mask = tf.tile(mask[:, None, None, :], (1, 1, tgt_len, 1))

    # 返回扩展后的掩码，乘以一个大的负数，用于处理填充位置
    return (one_cst - expanded_mask) * LARGE_NEGATIVE


class TFWav2Vec2GroupNorm(tf.keras.layers.Layer):
    """
    从tensorflow-addons https://www.tensorflow.org/addons/api_docs/python/tfa/layers/GroupNormalization 复制过来的
    """

    def __init__(
        self,
        groups: int = 32,
        axis: int = -1,
        epsilon: float = 1e-3,
        center: bool = True,
        scale: bool = True,
        beta_initializer: tf.keras.initializers.Initializer = "zeros",
        gamma_initializer: tf.keras.initializers.Initializer = "ones",
        beta_regularizer: tf.keras.regularizers.Regularizer = None,
        gamma_regularizer: tf.keras.regularizers.Regularizer = None,
        beta_constraint: tf.keras.constraints.Constraint = None,
        gamma_constraint: tf.keras.constraints.Constraint = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.supports_masking = True
        self.groups = groups
        self.axis = axis
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = tf.keras.initializers.get(beta_initializer)
        self.gamma_initializer = tf.keras.initializers.get(gamma_initializer)
        self.beta_regularizer = tf.keras.regularizers.get(beta_regularizer)
        self.gamma_regularizer = tf.keras.regularizers.get(gamma_regularizer)
        self.beta_constraint = tf.keras.constraints.get(beta_constraint)
        self.gamma_constraint = tf.keras.constraints.get(gamma_constraint)
        self._check_axis()

    def build(self, input_shape):
        self._check_if_input_shape_is_none(input_shape)
        self._set_number_of_groups_for_instance_norm(input_shape)
        self._check_size_of_dimensions(input_shape)
        self._create_input_spec(input_shape)

        self._add_gamma_weight(input_shape)
        self._add_beta_weight(input_shape)
        self.built = True
        super().build(input_shape)

    def call(self, inputs):
        input_shape = tf.keras.backend.int_shape(inputs)
        tensor_input_shape = tf.shape(inputs)

        reshaped_inputs, group_shape = self._reshape_into_groups(inputs, input_shape, tensor_input_shape)

        normalized_inputs = self._apply_normalization(reshaped_inputs, input_shape)

        is_instance_norm = (input_shape[self.axis] // self.groups) == 1
        if not is_instance_norm:
            outputs = tf.reshape(normalized_inputs, tensor_input_shape)
        else:
            outputs = normalized_inputs

        return outputs
    # 获取配置信息，包括groups、axis、epsilon、center、scale等参数
    def get_config(self):
        config = {
            "groups": self.groups,
            "axis": self.axis,
            "epsilon": self.epsilon,
            "center": self.center,
            "scale": self.scale,
            "beta_initializer": tf.keras.initializers.serialize(self.beta_initializer),
            "gamma_initializer": tf.keras.initializers.serialize(self.gamma_initializer),
            "beta_regularizer": tf.keras.regularizers.serialize(self.beta_regularizer),
            "gamma_regularizer": tf.keras.regularizers.serialize(self.gamma_regularizer),
            "beta_constraint": tf.keras.constraints.serialize(self.beta_constraint),
            "gamma_constraint": tf.keras.constraints.serialize(self.gamma_constraint),
        }
        # 调用父类的方法获取基础配置信息
        base_config = super().get_config()
        # 返回整合的配置信息
        return {**base_config, **config}

    # 计算输出的形状
    def compute_output_shape(self, input_shape):
        return input_shape

    # 重塑为分组形状
    def _reshape_into_groups(self, inputs, input_shape, tensor_input_shape):
        group_shape = [tensor_input_shape[i] for i in range(len(input_shape))]
        # 判断是否为实例标准化
        is_instance_norm = (input_shape[self.axis] // self.groups) == 1
        if not is_instance_norm:
            # 重塑输入为分组形状并返回
            group_shape[self.axis] = input_shape[self.axis] // self.groups
            group_shape.insert(self.axis, self.groups)
            group_shape = tf.stack(group_shape)
            reshaped_inputs = tf.reshape(inputs, group_shape)
            return reshaped_inputs, group_shape
        else:
            # 返回原始输入和分组形状
            return inputs, group_shape

    # 应用标准化
    def _apply_normalization(self, reshaped_inputs, input_shape):
        group_shape = tf.keras.backend.int_shape(reshaped_inputs)
        # 计算分组减少的轴
        group_reduction_axes = list(range(1, len(group_shape)))
        is_instance_norm = (input_shape[self.axis] // self.groups) == 1
        if not is_instance_norm:
            axis = -2 if self.axis == -1 else self.axis - 1
        else:
            axis = -1 if self.axis == -1 else self.axis - 1
        group_reduction_axes.pop(axis)

        # 计算均值和方差
        mean, variance = tf.nn.moments(reshaped_inputs, group_reduction_axes, keepdims=True)

        # 获取重塑后的权重
        gamma, beta = self._get_reshaped_weights(input_shape)
        # 批量标准化
        normalized_inputs = tf.nn.batch_normalization(
            reshaped_inputs,
            mean=mean,
            variance=variance,
            scale=gamma,
            offset=beta,
            variance_epsilon=self.epsilon,
        )
        return normalized_inputs

    # 获取重塑后的权重
    def _get_reshaped_weights(self, input_shape):
        broadcast_shape = self._create_broadcast_shape(input_shape)
        gamma = None
        beta = None
        # 如果启用可以缩放，则重塑gamma权重
        if self.scale:
            gamma = tf.reshape(self.gamma, broadcast_shape)

        # 如果启用中心化，则重塑beta权重
        if self.center:
            beta = tf.reshape(self.beta, broadcast_shape)
        return gamma, beta
    # 检查输入形状是否为 None
    def _check_if_input_shape_is_none(self, input_shape):
        # 获取指定轴的维度
        dim = input_shape[self.axis]
        # 如果维度为 None，抛出数值错误异常
        if dim is None:
            raise ValueError(
                "Axis "
                + str(self.axis)
                + " of input tensor should have a defined dimension but the layer received an input with shape "
                + str(input_shape)
                + "."
            )

    # 设置 instance normalization 的分组数
    def _set_number_of_groups_for_instance_norm(self, input_shape):
        dim = input_shape[self.axis]

        # 如果分组数为 -1，则将其设置为维度值
        if self.groups == -1:
            self.groups = dim

    # 检查维度大小
    def _check_size_of_dimensions(self, input_shape):
        dim = input_shape[self.axis]
        # 如果维度小于分组数，抛出数值错误异常
        if dim < self.groups:
            raise ValueError(
                "Number of groups ("
                + str(self.groups)
                + ") cannot be more than the number of channels ("
                + str(dim)
                + ")."
            )
        # 如果维度除以分组数的余数不为 0，抛出数值错误异常
        if dim % self.groups != 0:
            raise ValueError(
                "Number of groups ("
                + str(self.groups)
                + ") must be a multiple of the number of channels ("
                + str(dim)
                + ")."
            )

    # 检查轴是否合法
    def _check_axis(self):
        # 如果轴为 0，抛出数值错误异常
        if self.axis == 0:
            raise ValueError(
                "You are trying to normalize your batch axis. Do you want to use tf.layer.batch_normalization instead"
            )

    # 创建输入规范
    def _create_input_spec(self, input_shape):
        dim = input_shape[self.axis]
        # 创建输入规范
        self.input_spec = tf.keras.layers.InputSpec(ndim=len(input_shape), axes={self.axis: dim})

    # 添加 gamma 权重
    def _add_gamma_weight(self, input_shape):
        dim = input_shape[self.axis]
        shape = (dim,)

        # 如果需要缩放，则添加 gamma 权重
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

    # 添加 beta 权重
    def _add_beta_weight(self, input_shape):
        dim = input_shape[self.axis]
        shape = (dim,)

        # 如果需要中心化，则添加 beta 权重
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

    # 创建广播形状
    def _create_broadcast_shape(self, input_shape):
        # 初始化广播形状列表为输入形状各维度的 1 数组
        broadcast_shape = [1] * len(input_shape)
        # 判断是否为 instance normalization
        is_instance_norm = (input_shape[self.axis] // self.groups) == 1
        # 如果不是 instance normalization
        if not is_instance_norm:
            broadcast_shape[self.axis] = input_shape[self.axis] // self.groups
            broadcast_shape.insert(self.axis, self.groups)
        else:
            broadcast_shape[self.axis] = self.groups
        # 返回广播形状
        return broadcast_shape
class TFWav2Vec2WeightNormConv1D(tf.keras.layers.Conv1D):
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
        # 设置是否显式填充
        self.explicit_padding = explicit_padding
        # 定义过滤器轴
        self.filter_axis = 2
        # 定义核归一化的轴
        self.kernel_norm_axes = tf.constant([0, 1])

    def _init_norm(self):
        """Set the norm of the weight vector."""
        # 计算权重向量的范数
        kernel_norm = tf.sqrt(tf.reduce_sum(tf.square(self.weight_v), axis=self.kernel_norm_axes))
        # 将范数值赋给权重的归一化因子
        self.weight_g.assign(kernel_norm[:, tf.newaxis, tf.newaxis])

    def _normalize_kernel(self):
        """Generate normalized weights."""
        # 计算归一化后的权重
        kernel = tf.nn.l2_normalize(self.weight_v, axis=self.kernel_norm_axes) * tf.transpose(self.weight_g)
        self.kernel = tf.transpose(kernel)

    def build(self, input_shape):
        if not self.built:
            # 调用父类的 build 方法
            super().build(input_shape)

            # 初始化权重向量
            self.kernel = tf.Variable(tf.transpose(self.kernel), name="weight_v", trainable=True)
            self.weight_v = self.kernel

            # 添加权重归一化因子
            self.weight_g = self.add_weight(
                name="weight_g",
                shape=(int(self.weight_v.shape[self.filter_axis]), 1, 1),
                initializer="ones",
                dtype=self.weight_v.dtype,
                trainable=True,
            )
            # 初始化权重范数
            self._init_norm()
            # 添加偏置
            self.bias = self.add_weight(name="bias", shape=(self.filters,), initializer="zeros", trainable=True)

    def call(self, inputs):
        # TODO Matt: Assigning to attributes in call() is deeply sinful in TensorFlow, as it should be idempotent.
        #            This whole layer should be replaced by a layer that doesn't inherit from Conv1D, but instead calls
        #            a functional 1d convolution with normalized weights that it generates (but does not store!)
        # 归一化权重
        self._normalize_kernel()

        # 填充输入
        padded_inputs = tf.pad(inputs, ((0, 0), (self.explicit_padding, self.explicit_padding), (0, 0)))
        # 调用父类的 call 方法执行卷积操作
        output = super().call(padded_inputs)

        return output


class TFWav2Vec2NoLayerNormConvLayer(tf.keras.layers.Layer):
    # 初始化方法，接收配置和层索引作为参数
    def __init__(self, config: Wav2Vec2Config, layer_id: int = 0, **kwargs: Any) -> None:
        super().__init__(**kwargs)  # 调用父类的初始化方法
        # 设置输入卷积维度为配置里指定索引的卷积维度，如果索引小于等于0则设置为1
        self.in_conv_dim = config.conv_dim[layer_id] if layer_id > 0 else 1
        # 设置输出卷积维度为配置里指定索引的卷积维度
        self.out_conv_dim = config.conv_dim[layer_id]

        # 创建一个一维卷积层
        self.conv = tf.keras.layers.Conv1D(
            filters=self.out_conv_dim,  # 指定卷积核数量
            kernel_size=config.conv_kernel[layer_id],  # 指定卷积核大小
            strides=config.conv_stride[layer_id],  # 指定卷积步长
            use_bias=config.conv_bias,  # 指定是否使用偏置
            name="conv",  # 设置层的名称
        )
        # 获取激活函数
        self.activation = get_tf_activation(config.feat_extract_activation)

    # 前向传播方法，接收一个张量作为输入，返回一个张量
    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        # 对输入张量进行卷积操作
        hidden_states = self.conv(hidden_states)
        # 对卷积后的张量进行激活操作
        hidden_states = self.activation(hidden_states)
        # 返回处理后的张量
        return hidden_states

    # 构建方法，接收输入形状参数
    def build(self, input_shape=None):
        # 如果已经构建过了，则直接返回
        if self.built:
            return
        # 设置为已构建状态
        self.built = True
        # 如果卷积层已经创建
        if getattr(self, "conv", None) is not None:
            # 在命名空间内构建卷积层
            with tf.name_scope(self.conv.name):
                self.conv.build([None, None, self.in_conv_dim])  # 构建卷积层
# 定义一个自定义的层 TFWav2Vec2LayerNormConvLayer，继承自 tf.keras.layers.Layer
class TFWav2Vec2LayerNormConvLayer(tf.keras.layers.Layer):

    # 初始化方法，接受配置和层编号作为参数
    def __init__(self, config: Wav2Vec2Config, layer_id: int = 0, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        
        # 获取输入卷积维度
        self.in_conv_dim = config.conv_dim[layer_id] if layer_id > 0 else 1
        # 获取输出卷积维度
        self.out_conv_dim = config.conv_dim[layer_id]

        # 创建一个 1D 卷积层
        self.conv = tf.keras.layers.Conv1D(
            filters=self.out_conv_dim,
            kernel_size=config.conv_kernel[layer_id],
            strides=config.conv_stride[layer_id],
            use_bias=config.conv_bias,
            name="conv",
        )
        
        # 创建一个 LayerNormalization 层
        self.layer_norm = tf.keras.layers.LayerNormalization(name="layer_norm", epsilon=config.layer_norm_eps)
        
        # 获取激活函数
        self.activation = get_tf_activation(config.feat_extract_activation)

    # 前向传播方法，接受输入张量并返回输出张量
    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        # 使用卷积层处理输入张量
        hidden_states = self.conv(hidden_states)
        # 使用 LayerNormalization 处理卷积层输出张量
        hidden_states = self.layer_norm(hidden_states)
        # 使用激活函数处理 LayerNormalization 的输出张量
        hidden_states = self.activation(hidden_states)
        # 返回处理后的张量
        return hidden_states

    # 构建方法，用于构建层的内部结构
    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 构建卷积层
        if getattr(self, "conv", None) is not None:
            with tf.name_scope(self.conv.name):
                self.conv.build([None, None, self.in_conv_dim])
        # 构建 LayerNormalization 层
        if getattr(self, "layer_norm", None) is not None:
            with tf.name_scope(self.layer_norm.name):
                self.layer_norm.build([None, None, self.out_conv_dim])


# 定义一个自定义的层 TFWav2Vec2GroupNormConvLayer，继承自 tf.keras.layers.Layer
class TFWav2Vec2GroupNormConvLayer(tf.keras.layers.Layer):

    # 初始化方法，接受配置和层编号作为参数
    def __init__(self, config: Wav2Vec2Config, layer_id: int = 0, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        
        # 获取输入卷积维度
        self.in_conv_dim = config.conv_dim[layer_id] if layer_id > 0 else 1
        # 获取输出卷积维度
        self.out_conv_dim = config.conv_dim[layer_id]

        # 创建一个 1D 卷积层
        self.conv = tf.keras.layers.Conv1D(
            filters=self.out_conv_dim,
            kernel_size=config.conv_kernel[layer_id],
            strides=config.conv_stride[layer_id],
            use_bias=config.conv_bias,
            name="conv",
        )
        
        # 获取激活函数
        self.activation = get_tf_activation(config.feat_extract_activation)
        
        # 创建一个自定义的 GroupNormalization 层
        self.layer_norm = TFWav2Vec2GroupNorm(
            groups=self.out_conv_dim, epsilon=config.layer_norm_eps, name="layer_norm"
        )

    # 前向传播方法，接受输入张量并返回输出张量
    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        # 使用卷积层处理输入张量
        hidden_states = self.conv(hidden_states)
        # 使用 GroupNormalization 处理卷积层输出张量
        hidden_states = self.layer_norm(hidden_states)
        # 使用激活函数处理 GroupNormalization 的输出张量
        hidden_states = self.activation(hidden_states)
        # 返回处理后的张量
        return hidden_states

    # 构建方法，用于构建层的内部结构
    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 构建卷积层
        if getattr(self, "conv", None) is not None:
            with tf.name_scope(self.conv.name):
                self.conv.build([None, None, self.in_conv_dim])
        # 构建 GroupNormalization 层
        if getattr(self, "layer_norm", None) is not None:
            with tf.name_scope(self.layer_norm.name):
                self.layer_norm.build([None, None, self.out_conv_dim])
class TFWav2Vec2PositionalConvEmbedding(tf.keras.layers.Layer):
    # TFWav2Vec2PositionalConvEmbedding 类的初始化函数
    def __init__(self, config: Wav2Vec2Config, **kwargs: Any) -> None:
        # 调用父类的初始化函数
        super().__init__(**kwargs)
        # 创建权重归一化的一维卷积层对象
        self.conv = TFWav2Vec2WeightNormConv1D(
            filters=config.hidden_size,
            kernel_size=config.num_conv_pos_embeddings,
            groups=config.num_conv_pos_embedding_groups,
            explicit_padding=config.num_conv_pos_embeddings // 2,
            name="conv",
        )
        # 创建同样大小的填充层对象
        self.padding = TFWav2Vec2SamePadLayer(config.num_conv_pos_embeddings)
        # 获取激活函数
        self.activation = get_tf_activation(config.feat_extract_activation)
        # 保存配置信息
        self.config = config

    # TFWav2Vec2PositionalConvEmbedding 类的调用方法
    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        # 通过卷积层处理隐藏状态
        hidden_states = self.conv(hidden_states)
        # 对处理后的隐藏状态进行填充
        hidden_states = self.padding(hidden_states)
        # 使用激活函数处理填充后的隐藏状态
        hidden_states = self.activation(hidden_states)
        # 返回处理后的隐藏状态
        return hidden_states

    # TFWav2Vec2PositionalConvEmbedding 类的构建方法
    def build(self, input_shape=None):
        # 如果已经构建过则直接返回
        if self.built:
            return
        # 设置已经构建标志为真
        self.built = True
        # 如果卷积层已经存在，则构建卷积层
        if getattr(self, "conv", None) is not None:
            with tf.name_scope(self.conv.name):
                self.conv.build([None, None, self.config.hidden_size])


class TFWav2Vec2SamePadLayer(tf.keras.layers.Layer):
    # TFWav2Vec2SamePadLayer 类的初始化函数
    def __init__(self, num_conv_pos_embeddings, **kwargs):
        # 调用父类的初始化函数
        super().__init__(**kwargs)
        # 计算需要移除的填充数量
        self.num_pad_remove = 1 if num_conv_pos_embeddings % 2 == 0 else 0

    # TFWav2Vec2SamePadLayer 类的调用方法
    def call(self, hidden_states):
        # 如果需要移除填充，则进行相应的处理
        if self.num_pad_remove > 0:
            hidden_states = hidden_states[:, : -self.num_pad_remove, :]
        # 返回处理后的隐藏状态
        return hidden_states


class TFWav2Vec2FeatureEncoder(tf.keras.layers.Layer):
    # TFWav2Vec2FeatureEncoder 类的初始化函数
    def __init__(self, config: Wav2Vec2Config, **kwargs: Any) -> None:
        # 调用父类的初始化函数
        super().__init__(**kwargs)
        # 根据配置信息创建卷积层对象列表
        if config.feat_extract_norm == "group":
            conv_layers = [TFWav2Vec2GroupNormConvLayer(config, layer_id=0, name=f"conv_layers.{0}")] + [
                TFWav2Vec2NoLayerNormConvLayer(config, layer_id=i + 1, name=f"conv_layers.{i+1}")
                for i in range(config.num_feat_extract_layers - 1)
            ]
        elif config.feat_extract_norm == "layer":
            conv_layers = [
                TFWav2Vec2LayerNormConvLayer(config, layer_id=i, name=f"conv_layers.{i}")
                for i in range(config.num_feat_extract_layers)
            ]
        else:
            raise ValueError(
                f"`config.feat_extract_norm` is {config.feat_extract_norm}, but has to be one of ['group', 'layer']"
            )
        # 保存卷积层对象列表
        self.conv_layers = conv_layers

    # TFWav2Vec2FeatureEncoder 类的调用方法
    def call(self, input_values):
        # 将输入值扩展一个维度
        hidden_states = tf.expand_dims(input_values, -1)
        # 对于每个卷积层对象，依次进行处理
        for conv_layer in self.conv_layers:
            hidden_states = conv_layer(hidden_states)
        # 返回处理后的隐藏状态
        return hidden_states
    # 定义 build 方法，用于构建神经网络模型
    def build(self, input_shape=None):
        # 如果模型已经构建过，则直接返回
        if self.built:
            return
        # 将模型标记为已构建
        self.built = True
        # 如果存在卷积层列表，则对每一层进行构建
        if getattr(self, "conv_layers", None) is not None:
            for conv_layer in self.conv_layers:
                # 使用命名空间为卷积层命名
                with tf.name_scope(conv_layer.name):
                    # 构建卷积层
                    conv_layer.build(None)
class TFWav2Vec2FeatureExtractor(TFWav2Vec2FeatureEncoder):
    # TFWav2Vec2FeatureExtractor 类继承自 TFWav2Vec2FeatureEncoder 类
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        # 发出警告信息，提醒该类已经被弃用，将在 Transformers v5 中移除，建议使用其父类代替
        warnings.warn(
            f"The class `{self.__class__.__name__}` has been depreciated "
            "and will be removed in Transformers v5. "
            f"Use `{self.__class__.__bases__[0].__name__}` instead.",
            FutureWarning,
        )


class TFWav2Vec2FeatureProjection(tf.keras.layers.Layer):
    # TFWav2Vec2FeatureProjection 类继承自 tf.keras.layers.Layer 类
    def __init__(self, config: Wav2Vec2Config, **kwargs):
        super().__init__(**kwargs)
        
        # 初始化层归一化（Layer Normalization）层
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="layer_norm")
        # 初始化全连接（Dense）层
        self.projection = tf.keras.layers.Dense(
            units=config.hidden_size,
            kernel_initializer=get_initializer(config.initializer_range),
            bias_initializer="zeros",
            name="projection",
        )
        # 初始化 Dropout 层
        self.dropout = tf.keras.layers.Dropout(rate=config.feat_proj_dropout)
        self.config = config

    def call(self, hidden_states: tf.Tensor, training: bool = False) -> tf.Tensor:
        norm_hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.projection(norm_hidden_states)
        hidden_states = self.dropout(hidden_states, training=training)
        return hidden_states, norm_hidden_states

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "layer_norm", None) is not None:
            with tf.name_scope(self.layer_norm.name):
                self.layer_norm.build([None, None, self.config.conv_dim[-1]])
        if getattr(self, "projection", None) is not None:
            with tf.name_scope(self.projection.name):
                self.projection.build([None, None, self.config.conv_dim[-1]])


# Copied from transformers.models.bart.modeling_tf_bart.TFBartAttention with TFBart->TFWav2Vec2
class TFWav2Vec2Attention(tf.keras.layers.Layer):
    # TFWav2Vec2Attention 类继承自 tf.keras.layers.Layer 类
    """Multi-headed attention from "Attention Is All You Need"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
        **kwargs,
    def __init__(self, embed_dim: int = 768, num_heads: int = 12, dropout: float = 0.0, bias: bool = True, is_decoder: bool = False, **kwargs):
        # 调用父类的构造函数
        super().__init__(**kwargs)
        # 设置嵌入维度
        self.embed_dim = embed_dim
    
        self.num_heads = num_heads
        # 创建一个丢弃层
        self.dropout = tf.keras.layers.Dropout(dropout)
        # 计算每个注意力头的维度
        self.head_dim = embed_dim // num_heads
        # 判断嵌入维度是否可以被头数整除
        if (self.head_dim * num_heads) != self.embed_dim:
            # 如果不能整除，抛出异常
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        # 缩放系数
        self.scaling = self.head_dim**-0.5
        # 是否为解码器
        self.is_decoder = is_decoder
    
        # 创建一个全连接层，用于将k变换为embed_dim维
        self.k_proj = tf.keras.layers.Dense(embed_dim, use_bias=bias, name="k_proj")
        # 创建一个全连接层，用于将q变换为embed_dim维
        self.q_proj = tf.keras.layers.Dense(embed_dim, use_bias=bias, name="q_proj")
        # 创建一个全连接层，用于将v变换为embed_dim维
        self.v_proj = tf.keras.layers.Dense(embed_dim, use_bias=bias, name="v_proj")
        # 创建一个全连接层，用于将输出结果变换为embed_dim维
        self.out_proj = tf.keras.layers.Dense(embed_dim, use_bias=bias, name="out_proj")
    
    def _shape(self, tensor: tf.Tensor, seq_len: int, bsz: int):
        # 将tensor重塑为(bsz, seq_len, num_heads, head_dim)的形状，并进行转置
        return tf.transpose(tf.reshape(tensor, (bsz, seq_len, self.num_heads, self.head_dim)), (0, 2, 1, 3))
    
    def call(
        self,
        hidden_states: tf.Tensor,
        key_value_states: tf.Tensor | None = None,
        past_key_value: Tuple[Tuple[tf.Tensor]] | None = None,
        attention_mask: tf.Tensor | None = None,
        layer_head_mask: tf.Tensor | None = None,
        training: Optional[bool] = False,
        ):
        # ...
        # 这里是函数体的具体实现，暂时无法对其进行注释
    
    def build(self, input_shape=None):
        # 如果已经构建过了，则直接返回
        if self.built:
            return
        # 设置为已构建
        self.built = True
        # 构建全连接层
        if getattr(self, "k_proj", None) is not None:
            with tf.name_scope(self.k_proj.name):
                self.k_proj.build([None, None, self.embed_dim])
        if getattr(self, "q_proj", None) is not None:
            with tf.name_scope(self.q_proj.name):
                self.q_proj.build([None, None, self.embed_dim])
        if getattr(self, "v_proj", None) is not None:
            with tf.name_scope(self.v_proj.name):
                self.v_proj.build([None, None, self.embed_dim])
        if getattr(self, "out_proj", None) is not None:
            with tf.name_scope(self.out_proj.name):
                self.out_proj.build([None, None, self.embed_dim])
class TFWav2Vec2FeedForward(tf.keras.layers.Layer):
    def __init__(self, config: Wav2Vec2Config, **kwargs):
        super().__init__(**kwargs)

        # 定义中间层的 Dropout 层，使用配置中的激活函数 dropout 比例
        self.intermediate_dropout = tf.keras.layers.Dropout(config.activation_dropout)

        # 定义中间层的全连接 Dense 层，根据配置设置神经元数量、初始化方式和偏置
        self.intermediate_dense = tf.keras.layers.Dense(
            units=config.intermediate_size,
            kernel_initializer=get_initializer(config.initializer_range),
            bias_initializer="zeros",
            name="intermediate_dense",
        )
        # 获取配置中定义的激活函数
        self.intermediate_act_fn = get_tf_activation(config.hidden_act)

        # 定义输出层的全连接 Dense 层，根据配置设置神经元数量、初始化方式和偏置
        self.output_dense = tf.keras.layers.Dense(
            units=config.hidden_size,
            kernel_initializer=get_initializer(config.initializer_range),
            bias_initializer="zeros",
            name="output_dense",
        )
        # 定义输出层的 Dropout 层，使用配置中输出层的 dropout 比例
        self.output_dropout = tf.keras.layers.Dropout(config.hidden_dropout)
        self.config = config

    def call(self, hidden_states: tf.Tensor, training: bool = False) -> tf.Tensor:
        # 使用中间层的全连接 Dense 层处理隐藏状态
        hidden_states = self.intermediate_dense(hidden_states)
        # 使用中间层的激活函数处理中间结果
        hidden_states = self.intermediate_act_fn(hidden_states)
        # 使用中间层的 Dropout 处理中间结果
        hidden_states = self.intermediate_dropout(hidden_states, training=training)

        # 使用输出层的全连接 Dense 层处理中间结果
        hidden_states = self.output_dense(hidden_states)
        # 使用输出层的 Dropout 处理输出结果
        hidden_states = self.output_dropout(hidden_states, training=training)
        return hidden_states

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "intermediate_dense", None) is not None:
            with tf.name_scope(self.intermediate_dense.name):
                # 构建中间层的全连接 Dense 层
                self.intermediate_dense.build([None, None, self.config.hidden_size])
        if getattr(self, "output_dense", None) is not None:
            with tf.name_scope(self.output_dense.name):
                # 构建输出层的全连接 Dense 层
                self.output_dense.build([None, None, self.config.intermediate_size])


class TFWav2Vec2EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, config: Wav2Vec2Config, **kwargs):
        super().__init__(**kwargs)
        
        # 定义注意力机制层
        self.attention = TFWav2Vec2Attention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=False,
            name="attention",
        )
        # 定义 Dropout 层
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout)
        # 定义 Layer Normalization 层
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="layer_norm")
        # 定义 Feed Forward 层
        self.feed_forward = TFWav2Vec2FeedForward(config, name="feed_forward")
        # 定义最终的 Layer Normalization 层
        self.final_layer_norm = tf.keras.layers.LayerNormalization(
            epsilon=config.layer_norm_eps, name="final_layer_norm"
        )
        self.config = config

    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: tf.Tensor | None = None,
        output_attentions: Optional[bool] = False,
        training: bool = False,
    # 定义一个函数，返回类型为Tuple[tf.Tensor]，输入参数为隐藏状态和注意力掩码
    def call(self, hidden_states: tf.Tensor, attention_mask: Optional[tf.Tensor] = None, training: bool = False, output_attentions: bool = False) -> Tuple[tf.Tensor]:
        # 保存注意力机制之前的隐藏状态，用于残差连接
        attn_residual = hidden_states
        # 对隐藏状态进行注意力计算
        hidden_states, attn_weights, _ = self.attention(
            hidden_states, attention_mask=attention_mask, training=training
        )
        # 对注意力计算得到的隐藏状态进行Dropout操作
        hidden_states = self.dropout(hidden_states, training=training)
        # 残差连接，得到最终的隐藏状态
        hidden_states = attn_residual + hidden_states

        # 对最终隐藏状态进行Layer Norm操作
        hidden_states = self.layer_norm(hidden_states)
        # 对Layer Norm后的隐藏状态进行前馈神经网络计算
        hidden_states = hidden_states + self.feed_forward(hidden_states)
        # 对前馈神经网络计算后的隐藏状态进行Layer Norm操作
        hidden_states = self.final_layer_norm(hidden_states)

        # 构建输出结果，包括隐藏状态和注意力权重（如果需要）
        outputs = (hidden_states,)

        # 如果需要输出注意力权重，则将注意力权重加入到输出结果中
        if output_attentions:
            outputs += (attn_weights,)

        return outputs

    # 构建模型，初始化各个子层
    def build(self, input_shape=None):
        # 如果已经构建过，直接返回
        if self.built:
            return
        self.built = True
        # 初始化注意力机制层
        if getattr(self, "attention", None) is not None:
            with tf.name_scope(self.attention.name):
                self.attention.build(None)
        # 初始化Layer Norm层
        if getattr(self, "layer_norm", None) is not None:
            with tf.name_scope(self.layer_norm.name):
                self.layer_norm.build([None, None, self.config.hidden_size])
        # 初始化前馈神经网络层
        if getattr(self, "feed_forward", None) is not None:
            with tf.name_scope(self.feed_forward.name):
                self.feed_forward.build(None)
        # 初始化最终Layer Norm层
        if getattr(self, "final_layer_norm", None) is not None:
            with tf.name_scope(self.final_layer_norm.name):
                self.final_layer_norm.build([None, None, self.config.hidden_size])
class TFWav2Vec2EncoderLayerStableLayerNorm(tf.keras.layers.Layer):
    # 初始化函数，接受配置参数，并设置各个层和模块
    def __init__(self, config: Wav2Vec2Config, **kwargs):
        # 调用父类的初始化函数
        super().__init__(**kwargs)
        # 创建注意力层对象
        self.attention = TFWav2Vec2Attention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=False,
            name="attention",
        )
        # 创建 dropout 层
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout)
        # 创建稳定层归一化层
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="layer_norm")
        # 创建前馈神经网络层
        self.feed_forward = TFWav2Vec2FeedForward(config, name="feed_forward")
        # 创建最终的层归一化层
        self.final_layer_norm = tf.keras.layers.LayerNormalization(
            epsilon=config.layer_norm_eps, name="final_layer_norm"
        )
        # 保存配置参数
        self.config = config

    # 调用函数，对输入进行编码处理
    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: tf.Tensor | None = None,
        output_attentions: Optional[bool] = False,
        training: bool = False,
    ) -> Tuple[tf.Tensor]:
        # 保留注意力层之前的隐藏状态
        attn_residual = hidden_states
        # 使用稳定层归一化
        hidden_states = self.layer_norm(hidden_states)
        # 使用注意力层
        hidden_states, attn_weights, _ = self.attention(
            hidden_states, attention_mask=attention_mask, training=training
        )
        # 使用 dropout 层
        hidden_states = self.dropout(hidden_states, training=training)
        # 将注意力层之前的状态与当前状态相加
        hidden_states = attn_residual + hidden_states
        # 将当前状态与前馈神经网络处理的结果相加
        hidden_states = hidden_states + self.feed_forward(self.final_layer_norm(hidden_states))

        # 将处理结果保存在 outputs 中
        outputs = (hidden_states,)

        # 如果需要输出注意力权重，则保存到 outputs 中
        if output_attentions:
            outputs += (attn_weights,)

        return outputs

    # 构建函数，用于构建各个子层
    def build(self, input_shape=None):
        # 如果已经构建过，直接返回
        if self.built:
            return
        self.built = True
        # 构建注意力层
        if getattr(self, "attention", None) is not None:
            with tf.name_scope(self.attention.name):
                self.attention.build(None)
        # 构建稳定层归一化层
        if getattr(self, "layer_norm", None) is not None:
            with tf.name_scope(self.layer_norm.name):
                self.layer_norm.build([None, None, self.config.hidden_size])
        # 构建前馈神经网络层
        if getattr(self, "feed_forward", None) is not None:
            with tf.name_scope(self.feed_forward.name):
                self.feed_forward.build(None)
        # 构建最终层归一化层
        if getattr(self, "final_layer_norm", None) is not None:
            with tf.name_scope(self.final_layer_norm.name):
                self.final_layer_norm.build([None, None, self.config.hidden_size])


class TFWav2Vec2Encoder(tf.keras.layers.Layer):
    def __init__(self, config: Wav2Vec2Config, **kwargs):
        # 初始化函数，接受Wav2Vec2Config配置对象和其他可变参数
        super().__init__(**kwargs)
        # 调用父类的初始化函数

        # 将配置对象保存到self.config
        self.config = config
        # 创建pos_conv_embed对象，用于位置编码的卷积嵌入
        self.pos_conv_embed = TFWav2Vec2PositionalConvEmbedding(config, name="pos_conv_embed")
        # 创建layer_norm层，用于归一化
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="layer_norm")
        # 创建dropout层，用于随机失活
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout)
        # 创建多个encoder层，保存到self.layer列表中
        self.layer = [TFWav2Vec2EncoderLayer(config, name=f"layers.{i}") for i in range(config.num_hidden_layers)]

    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: tf.Tensor | None = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
        training: Optional[bool] = False,
    ) -> Union[TFBaseModelOutput, Tuple[tf.Tensor]]:
        # 前向传播函数，接受输入hidden_states和其他可选参数

        # 初始化all_hidden_states和all_self_attentions为空元组，用于保存每个encoder层的hidden_states和attention
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        if attention_mask is not None:
            # 如果attention_mask不为空，则将hidden_states与attention_mask逐元素相乘
            hidden_states = hidden_states * tf.expand_dims(attention_mask, -1)
            # 将attention_mask扩展为与hidden_states同样的维度
            attention_mask = _expand_mask(attention_mask)
        else:
            attention_mask = None

        # 通过pos_conv_embed将hidden_states进行位置编码并得到position_embeddings
        position_embeddings = self.pos_conv_embed(hidden_states)
        # 将hidden_states与position_embeddings相加
        hidden_states = hidden_states + position_embeddings
        # 对hidden_states进行归一化处理
        hidden_states = self.layer_norm(hidden_states)
        # 对hidden_states进行dropout处理
        hidden_states = self.dropout(hidden_states, training=training)

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                # 如果需要输出hidden_states，则将当前的hidden_states添加到all_hidden_states中
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 添加LayerDrop（参见https://arxiv.org/abs/1909.11556的描述）
            dropout_probability = np.random.uniform(0, 1)
            if training and (dropout_probability < self.config.layerdrop):  # skip the layer
                # 如果处于训练模式，并且dropout_probability小于配置文件中的layerdrop值，则跳过该层
                continue

            # 调用layer_module的前向传播函数，得到layer_outputs，并更新hidden_states
            layer_outputs = layer_module(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                training=training,
            )
            hidden_states = layer_outputs[0]

            if output_attentions:
                # 如果需要输出attention，则将layer_outputs中的attention添加到all_self_attentions中
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        # 添加最后一层的hidden_states到all_hidden_states中
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            # 如果不需要返回字典，则返回元组形式的结果
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        # 返回TFBaseModelOutput对象形式的结果
        return TFBaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )
    # 构建函数用于构建模型层，如果已经构建过了，则直接返回
    def build(self, input_shape=None):
        if self.built:
            return
        # 标记模型为已构建状态
        self.built = True
        # 如果存在位置卷积嵌入，则构建位置卷积嵌入层
        if getattr(self, "pos_conv_embed", None) is not None:
            # 使用命名空间指定位置卷积嵌入层的命名范围，并构建该层
            with tf.name_scope(self.pos_conv_embed.name):
                self.pos_conv_embed.build(None)
        # 如果存在层归一化，则构建层归一化层
        if getattr(self, "layer_norm", None) is not None:
            # 使用命名空间指定层归一化层的命名范围，并构建该层
            with tf.name_scope(self.layer_norm.name):
                self.layer_norm.build([None, None, self.config.hidden_size])
        # 如果存在多层自注意力层，则逐层构建每一层
        if getattr(self, "layer", None) is not None:
            for layer in self.layer:
                # 使用命名空间指定每一层的命名范围，并构建该层
                with tf.name_scope(layer.name):
                    layer.build(None)
class TFWav2Vec2EncoderStableLayerNorm(tf.keras.layers.Layer):
    # 构造函数，初始化编码器参数
    def __init__(self, config: Wav2Vec2Config, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        # 创建位置卷积嵌入层
        self.pos_conv_embed = TFWav2Vec2PositionalConvEmbedding(config, name="pos_conv_embed")
        # 创建层归一化
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="layer_norm")
        # 创建dropout层
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout)
        # 创建编码器层列表
        self.layer = [
            TFWav2Vec2EncoderLayerStableLayerNorm(config, name=f"layers.{i}") for i in range(config.num_hidden_layers)
        ]

    # 调用函数，进行编码器计算
    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: tf.Tensor | None = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
        training: Optional[bool] = False,
    ) -> Union[TFBaseModelOutput, Tuple[tf.Tensor]]:

        # 如果输出隐藏层信息，则初始化空的所有隐藏状态
        all_hidden_states = () if output_hidden_states else None
        # 如果输出注意力信息，则初始化空的所有自注意力
        all_self_attentions = () if output_attentions else None

        # 如果存在注意力遮罩
        if attention_mask is not None:
            # 对隐藏状态进行注意力遮罩处理
            hidden_states = hidden_states * tf.expand_dims(attention_mask, -1)
            # 对注意力遮罩进行扩展
            attention_mask = _expand_mask(attention_mask)
        else:
            attention_mask = None

        # 计算位置嵌入
        position_embeddings = self.pos_conv_embed(hidden_states)
        # 将位置嵌入加到隐藏状态上
        hidden_states = hidden_states + position_embeddings
        # 对隐藏状态进行dropout
        hidden_states = self.dropout(hidden_states, training=training)

        # 遍历编码器层
        for i, layer_module in enumerate(self.layer):
            # 如果输出隐藏层信息
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 添加LayerDrop（参见https://arxiv.org/abs/1909.11556进行描述）
            dropout_probability = np.random.uniform(0, 1)
            # 如果处于训练阶段且dropout概率小于配置的layerdrop值
            if training and (dropout_probability < self.config.layerdrop):  # skip the layer
                # 跳过当前层
                continue

            # 调用当前编码器层进行计算
            layer_outputs = layer_module(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                training=training,
            )
            hidden_states = layer_outputs[0]

            # 如果输出注意力信息
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        # 对隐藏状态进行层归一化
        hidden_states = self.layer_norm(hidden_states)

        # 如果输出隐藏层信息
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 如果不返回字典
        if not return_dict:
            # 返回计算结果
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        # 返回TFBaseModelOutput对象
        return TFBaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )
    # 如果模型已经构建完成，则直接返回，不进行重复构建
    if self.built:
        return
    # 设置模型已构建标志为 True
    self.built = True
    # 如果存在位置卷积嵌入层
    if getattr(self, "pos_conv_embed", None) is not None:
        # 在命名空间 self.pos_conv_embed.name 下构建位置卷积嵌入层
        with tf.name_scope(self.pos_conv_embed.name):
            self.pos_conv_embed.build(None)
    # 如果存在层标准化层
    if getattr(self, "layer_norm", None) is not None:
        # 在命名空间 self.layer_norm.name 下构建层标准化层
        with tf.name_scope(self.layer_norm.name):
            self.layer_norm.build([None, None, self.config.hidden_size])
    # 如果存在多层自注意力机制层
    if getattr(self, "layer", None) is not None:
        # 遍历每一层
        for layer in self.layer:
            # 在每一层的命名空间 layer.name 下构建该层
            with tf.name_scope(layer.name):
                layer.build(None)
# 使用 keras_serializable 装饰器将类标记为可序列化
@keras_serializable
class TFWav2Vec2MainLayer(tf.keras.layers.Layer):
    # 指定配置类
    config_class = Wav2Vec2Config

    def __init__(self, config: Wav2Vec2Config, **kwargs):
        super().__init__(**kwargs)
        # 初始化配置
        self.config = config
        # 创建特征提取器对象
        self.feature_extractor = TFWav2Vec2FeatureEncoder(config, name="feature_extractor")
        # 创建特征投影对象
        self.feature_projection = TFWav2Vec2FeatureProjection(config, name="feature_projection")

        # 根据配置创建编码器对象
        if config.do_stable_layer_norm:
            self.encoder = TFWav2Vec2EncoderStableLayerNorm(config, name="encoder")
        else:
            self.encoder = TFWav2Vec2Encoder(config, name="encoder")

    def build(self, input_shape=None):
        # 如果已经构建过，直接返回
        if self.built:
            return
        self.built = True
        # 如果配置中的时间和特征遮罩概率大于 0，创建一个可训练的掩码特征嵌入
        if self.config.mask_time_prob > 0.0 or self.config.mask_feature_prob > 0.0:
            self.masked_spec_embed = self.add_weight(
                shape=(self.config.hidden_size,), initializer="uniform", trainable=True, name="masked_spec_embed"
            )
        # 构建特征提取器
        if getattr(self, "feature_extractor", None) is not None:
            with tf.name_scope(self.feature_extractor.name):
                self.feature_extractor.build(None)
        # 构建特征投影
        if getattr(self, "feature_projection", None) is not None:
            with tf.name_scope(self.feature_projection.name):
                self.feature_projection.build(None)
        # 构建编码器
        if getattr(self, "encoder", None) is not None:
            with tf.name_scope(self.encoder.name):
                self.encoder.build(None)

    def _get_feat_extract_output_lengths(self, input_lengths: tf.Tensor):
        """
        Computes the output length of the convolutional layers
        """

        def _conv_out_length(input_length, kernel_size, stride):
            # 计算 1D 卷积层的输出长度公式，参考 https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
            return (input_length - kernel_size) // stride + 1

        # 遍历卷积核大小和步幅，计算输出长度
        for kernel_size, stride in zip(self.config.conv_kernel, self.config.conv_stride):
            input_lengths = _conv_out_length(input_lengths, kernel_size, stride)

        return input_lengths
    def _mask_hidden_states(self, hidden_states: tf.Tensor, mask_time_indices: tf.Tensor | None = None):
        """
        Masks extracted features along time axis and/or along feature axis according to
        [SpecAugment](https://arxiv.org/abs/1904.08779).
        """
        # 获取隐藏状态的形状参数
        batch_size, sequence_length, hidden_size = shape_list(hidden_states)

        # `config.apply_spec_augment` 可以设置为 False，表示不进行掩码处理
        if not getattr(self.config, "apply_spec_augment", True):
            return hidden_states

        if mask_time_indices is not None:
            # 根据给定的 mask_time_indices 对时间轴进行 SpecAugment 处理
            hidden_states = tf.where(
                tf.cast(mask_time_indices[:, :, tf.newaxis], tf.bool),
                self.masked_spec_embed[tf.newaxis, tf.newaxis, :],
                hidden_states,
            )

        elif self.config.mask_time_prob > 0:
            # 生成mask_time_indices并根据其进行 SpecAugment 处理
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

        # 根据配置的 mask_feature_prob 参数，对特征轴进行 SpecAugment 处理
        if self.config.mask_feature_prob > 0:
            mask_feature_indices = _compute_mask_indices(
                (batch_size, hidden_size),
                mask_prob=self.config.mask_feature_prob,
                mask_length=self.config.mask_feature_length,
            )
            hidden_states = tf.where(mask_feature_indices[:, tf.newaxis, :], hidden_states, 0)

        return hidden_states

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
        # 使用特征提取器提取输入值的特征，并进行训练
        extract_features = self.feature_extractor(tf.cast(input_values, tf.float32), training=training)
        # 将提取的特征进行转置操作
        # extract_features = tf.transpose(extract_features, perm=(0, 2, 1))

        if attention_mask is not None:
            # 根据卷积公式计算真实的输出长度
            output_lengths = self._get_feat_extract_output_lengths(tf.reduce_sum(attention_mask, -1))

            # 根据卷积公式计算的输出长度创建一个序列掩码
            attention_mask = tf.sequence_mask(
                output_lengths, maxlen=shape_list(extract_features)[1], dtype=extract_features.dtype
            )

        # 使用特征投影方法处理提取的特征
        hidden_states, extract_features = self.feature_projection(extract_features, training=training)

        # 从关键字参数中获取时间索引掩码
        mask_time_indices = kwargs.get("mask_time_indices", None)
        if training:
            # 如果是训练模式，对隐藏状态进行掩码操作
            hidden_states = self._mask_hidden_states(hidden_states, mask_time_indices=mask_time_indices)

        # 使用编码器对隐藏状态进行编码
        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )
        hidden_states = encoder_outputs[0]

        if not return_dict:
            # 如果不返回字典，返回隐藏状态、提取的特征和编码器输出的其它信息
            return (hidden_states, extract_features) + encoder_outputs[1:]

        # 如果返回字典，则返回基于 TF Wav2Vec2 模型的输出
        return TFWav2Vec2BaseModelOutput(
            last_hidden_state=hidden_states,
            extract_features=extract_features,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
# 创建一个 TFPreTrainedModel 的子类 TFWav2Vec2PreTrainedModel
class TFWav2Vec2PreTrainedModel(TFPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 设置配置类为 Wav2Vec2Config
    config_class = Wav2Vec2Config
    # 设置基础模型前缀为 "wav2vec2"
    base_model_prefix = "wav2vec2"
    # 设置主输入名称为 "input_values"
    main_input_name = "input_values"

    # 定义输入签名，返回包含 "input_values" 和 "attention_mask" 的字典
    @property
    def input_signature(self):
        return {
            "input_values": tf.TensorSpec((None, None), tf.float32, name="input_values"),
            "attention_mask": tf.TensorSpec((None, None), tf.float32, name="attention_mask"),
        }

    # 定义虚拟输入，返回包含随机值的 "input_values" 和全 1 的 "attention_mask" 的字典
    @property
    def dummy_inputs(self):
        return {
            "input_values": tf.random.uniform(shape=(1, 500), dtype=tf.float32),
            "attention_mask": tf.ones(shape=(1, 500), dtype=tf.float32),
        }

    # 初始化方法
    def __init__(self, config, *inputs, **kwargs):
        # 调用父类的初始化方法
        super().__init__(config, *inputs, **kwargs)
        # 输出警告信息，内容为当前类的名称，并提示不支持在 CPU 上进行反向传播操作
        logger.warning(
            f"\n{self.__class__.__name__} has backpropagation operations that are NOT supported on CPU. If you wish "
            "to train/fine-tune this model, you need a GPU or a TPU"
        )

    # 计算卷积层的输出长度
    def _get_feat_extract_output_lengths(self, input_lengths, add_adapter=None):
        """
        Computes the output length of the convolutional layers
        """
        # 设置默认的 add_adapter 为配置里面的 add_adapter
        add_adapter = self.config.add_adapter if add_adapter is None else add_adapter

        # 定义计算卷积层输出长度的函数
        def _conv_out_length(input_length, kernel_size, stride):
            return tf.math.floordiv(input_length - kernel_size, stride) + 1

        # 遍历配置中的卷积核大小和步长，更新输入长度
        for kernel_size, stride in zip(self.config.conv_kernel, self.config.conv_stride):
            input_lengths = _conv_out_length(input_lengths, kernel_size, stride)

        # 如果有 adapter 层，再循环计算 adapter 层的卷积输出长度
        if add_adapter:
            for _ in range(self.config.num_adapter_layers):
                input_lengths = _conv_out_length(input_lengths, 1, self.config.adapter_stride)
        # 返回计算结果
        return input_lengths

    # 获取特征向量的注意力掩码
    def _get_feature_vector_attention_mask(
        self, feature_vector_length: int, attention_mask: tf.Tensor, add_adapter=None
    # 计算非填充长度，对注意力掩码进行累积和并取最后一个值
    non_padded_lengths = tf.math.cumsum(attention_mask, axis=-1)[:, -1]
    # 使用非填充长度计算特征提取器输出长度，并根据需要添加适配器
    output_lengths = self._get_feat_extract_output_lengths(non_padded_lengths, add_adapter=add_adapter)
    output_lengths = tf.cast(output_lengths, tf.int32)
    # 获取批次大小
    batch_size = tf.shape(attention_mask)[0]
    # 在这里检查设备
    attention_mask = tf.zeros(
        (batch_size, feature_vector_length), dtype=attention_mask.dtype, name="attention_mask"
    )  # 这两个操作确保输出长度索引之前的所有值都被关注
    ## 检查设备
    # 使用索引和更新值来更新注意力掩码，确保输出长度索引之前的所有值都被关注
    attention_mask = tf.tensor_scatter_nd_update(
        attention_mask,
        indices=tf.stack([tf.range(batch_size), output_lengths - 1], axis=1),
        updates=tf.ones([batch_size], dtype=attention_mask.dtype),
    )
    # 反转注意力掩码
    attention_mask = tf.reverse(attention_mask, axis=[-1])
    # 对注意力掩码进行累积和
    attention_mask = tf.cumsum(attention_mask, axis=-1)
    # 再次反转注意力掩码
    attention_mask = tf.reverse(attention_mask, axis=[-1])
    # 将注意力掩码转换为布尔类型
    attention_mask = tf.cast(attention_mask, tf.bool)
    # 返回最终的注意力掩码
    return attention_mask
# WAV_2_VEC_2_START_DOCSTRING: WAV_2_VEC_2_START_DOCSTRING是模型的类文档字符串，提供了关于模型的详细说明和用法提示
WAV_2_VEC_2_START_DOCSTRING = r"""

    This model inherits from [`TFPreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a [tf.keras.Model](https://www.tensorflow.org/api_docs/python/tf/keras/Model) subclass. Use it
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

# WAV_2_VEC_2_INPUTS_DOCSTRING: WAV_2_VEC_2_INPUTS_DOCSTRING是模型输入文档字符串
WAV_2_VEC_2_INPUTS_DOCSTRING = r"""
"""

# TFWav2Vec2Model类和其构造方法的说明
@add_start_docstrings(
    "The bare TFWav2Vec2 Model transformer outputing raw hidden-states without any specific head on top.",
    WAV_2_VEC_2_START_DOCSTRING,
)
class TFWav2Vec2Model(TFWav2Vec2PreTrainedModel):
    # TFWav2Vec2Model类的构造方法，初始化模型参数和相关对象
    def __init__(self, config: Wav2Vec2Config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.config = config
        self.wav2vec2 = TFWav2Vec2MainLayer(config, name="wav2vec2")
    # 添加模型前向传播中的文档字符串
    @add_start_docstrings_to_model_forward(WAV_2_VEC_2_INPUTS_DOCSTRING)
    # 替换返回文档字符串
    @replace_return_docstrings(output_type=TFBaseModelOutput, config_class=_CONFIG_FOR_DOC)
    # 解包输入变量
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

        ```python
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

        # 如果未指定output_hidden_states，则使用配置中的值
        output_hidden_states = output_hidden_states if output_hidden_states else self.config.output_hidden_states
        # 如果未指定output_attentions，则使用配置中的值
        output_attentions = output_attentions if output_attentions else self.config.output_attentions
        # 如果未指定return_dict，则使用配置中的值
        return_dict = return_dict if return_dict else self.config.return_dict

        # 调用Wav2Vec2模型的前向传播
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

    # 构建模型
    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回
        if self.built:
            return
        self.built = True
        # 检查wav2vec2是否存在，然后添加名称作用域并构建其结构
        if getattr(self, "wav2vec2", None) is not None:
            with tf.name_scope(self.wav2vec2.name):
                self.wav2vec2.build(None)
# 为TFWav2Vec2ForCTC类添加文档字符串和预训练模型的配置
@add_start_docstrings(
    """TFWav2Vec2 Model with a `language modeling` head on top for Connectionist Temporal Classification (CTC).""",
    WAV_2_VEC_2_START_DOCSTRING,
)
class TFWav2Vec2ForCTC(TFWav2Vec2PreTrainedModel):
    # 初始化函数
    def __init__(self, config: Wav2Vec2Config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        # 创建Wav2Vec2主层
        self.wav2vec2 = TFWav2Vec2MainLayer(config, name="wav2vec2")
        # 创建dropout层
        self.dropout = tf.keras.layers.Dropout(config.final_dropout)
        # 创建lm_head的全连接层
        self.lm_head = tf.keras.layers.Dense(config.vocab_size, name="lm_head")
        # 输出隐藏大小
        self.output_hidden_size = (
            config.output_hidden_size if hasattr(config, "add_adapter") and config.add_adapter else config.hidden_size
        )

    # 冻结特征提取器，不再更新特征提取器参数
    def freeze_feature_extractor(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameters will
        not be updated during training.
        """
        # 警告，freeze_feature_extractor方法即将被移除
        warnings.warn(
            "The method `freeze_feature_extractor` is deprecated and will be removed in Transformers v5. "
            "Please use the equivalent `freeze_feature_encoder` method instead.",
            FutureWarning,
        )
        self.freeze_feature_encoder()  # 调用freeze_feature_encoder方法

    # 冻结特征编码器，不再更新特征编码器参数
    def freeze_feature_encoder(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        self.wav2vec2.feature_extractor.trainable = False

    # 对模型的前向传播进行注释
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
    # 重写build方法
    def build(self, input_shape=None):
        # 如果已经构建过则直接返回
        if self.built:
            return
        self.built = True
        # 构建wav2vec2层
        if getattr(self, "wav2vec2", None) is not None:
            with tf.name_scope(self.wav2vec2.name):
                self.wav2vec2.build(None)
        # 构建lm_head层
        if getattr(self, "lm_head", None) is not None:
            with tf.name_scope(self.lm_head.name):
                self.lm_head.build([None, None, self.output_hidden_size])


class TFWav2Vec2ForSequenceClassification(TFWav2Vec2PreTrainedModel):
    # 初始化函数，接受一个配置对象作为参数
    def __init__(self, config):
        # 调用父类的初始化函数
        super().__init__(config)
        # 创建一个名为wav2vec2的TFWav2Vec2MainLayer对象
        self.wav2vec2 = TFWav2Vec2MainLayer(config, name="wav2vec2")
        # 设置隐藏层的数量
        self.num_layers = config.num_hidden_layers + 1
        # 使用tf.name_scope创建名为_name_scope的作用域
        with tf.name_scope(self._name_scope()):
            # 如果配置要求使用加权层求和
            if config.use_weighted_layer_sum:
                # 添加一个名为layer_weights的可训练权重，初始值为1
                self.layer_weights = self.add_weight(
                    shape=(self.num_layers,), initializer="ones", trainable=True, name="layer_weights"
                )
        # 保存配置对象
        self.config = config
        # 创建一个全连接层，单元数为配置中指定的分类器投影大小
        self.projector = tf.keras.layers.Dense(units=config.classifier_proj_size, name="projector")
        # 创建一个全连接层，单元数为配置中指定的标签数量，激活函数为空
        self.classifier = tf.keras.layers.Dense(units=config.num_labels, activation=None, name="classifier")

    # 冻结特征提取器的函数
    def freeze_feature_extractor(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameters will
        not be updated during training.
        """
        # 给出未来版本移除该函数的警告
        warnings.warn(
            "The method `freeze_feature_extractor` is deprecated and will be removed in Transformers v5. "
            "Please use the equivalent `freeze_feature_encoder` method instead.",
            FutureWarning,
        )
        # 调用freeze_feature_encoder函数
        self.freeze_feature_encoder()

    # 冻结特征编码器的函数
    def freeze_feature_encoder(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        # 将wav2vec2对象的特征提取器设置为不可训练

        self.wav2vec2.feature_extractor.trainable = False

    # 冻结基础模型的函数
    def freeze_base_model(self):
        """
        Calling this function will disable the gradient computation for the base model so that its parameters will not
        be updated during training. Only the classification head will be updated.
        """
        # 遍历wav2vec2对象的每一层，设置为不可训练
        for layer in self.wav2vec2.layers:
            layer.trainable = False

    # 调用函数
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
    # 定义方法，接受输入并返回 TFSequenceClassifierOutput 或 Tuple[tf.Tensor] 类型的结果
    def call(
        self,
        input_values: tf.Tensor,
        attention_mask: Optional[tf.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: Optional[bool] = None,
        labels: Optional[tf.Tensor] = None,
    ) -> TFSequenceClassifierOutput | Tuple[tf.Tensor]:
        # 如果 return_dict 为 None，则使用配置中的 use_return_dict
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 如果使用加权层求和，则将输出隐藏状态设置为 True
        output_hidden_states = True if self.config.use_weighted_layer_sum else output_hidden_states

        # 调用 wav2vec2 模型，获取输出
        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )
        # 如果使用加权层求和，则重新计算隐藏状态
        if self.config.use_weighted_layer_sum:
            hidden_states = outputs[_HIDDEN_STATES_START_POSITION]
            hidden_states = tf.stack(hidden_states, axis=1)
            norm_weights = tf.nn.softmax(self.layer_weights, axis=-1)
            hidden_states = tf.reduce_sum(hidden_states * tf.reshape(norm_weights, [-1, 1, 1]), axis=1)
        else:
            hidden_states = outputs[0]

        # 使用 projector 对隐藏状态进行投影
        hidden_states = self.projector(hidden_states)
        # 计算池化输出
        if attention_mask is None:
            # 如果没有注意力掩码，则求取平均隐藏状态
            pooled_output = tf.reduce_mean(hidden_states, axis=1)
        else:
            # 否则，根据注意力掩码对隐藏状态进行加权求和
            padding_mask = self._get_feature_vector_attention_mask(shape_list(hidden_states)[1], attention_mask)
            padding_mask_float = tf.cast(padding_mask, hidden_states.dtype)
            hidden_states = tf.multiply(hidden_states, tf.expand_dims(padding_mask_float, axis=-1))
            pooled_output = tf.divide(
                tf.reduce_sum(hidden_states, axis=1), tf.expand_dims(tf.reduce_sum(padding_mask_float, axis=1), axis=1)
            )
        # 将池化输出传递给分类器，得到预测 logits
        logits = self.classifier(pooled_output)
        loss = None
        # 如果存在标签，则计算损失
        if labels is not None:
            loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
            loss = loss_fn(tf.reshape(labels, [-1]), tf.reshape(logits, [-1, self.config.num_labels]))
        # 如果不需要返回字典，则返回结果元组
        if not return_dict:
            output = (logits,) + outputs[_HIDDEN_STATES_START_POSITION:]
            return ((loss,) + output) if loss is not None else output

        # 否则，返回 TFSequenceClassifierOutput 对象
        return TFSequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    # 构建模型
    def build(self, input_shape=None):
        # 如果已经构建，则直接返回
        if self.built:
            return
        self.built = True
        # 构建 wav2vec2 模型
        if getattr(self, "wav2vec2", None) is not None:
            with tf.name_scope(self.wav2vec2.name):
                self.wav2vec2.build(None)
        # 构建投影器
        if getattr(self, "projector", None) is not None:
            with tf.name_scope(self.projector.name):
                self.projector.build([None, None, self.config.hidden_size])
        # 构建分类器
        if getattr(self, "classifier", None) is not None:
            with tf.name_scope(self.classifier.name):
                self.classifier.build([None, None, self.config.classifier_proj_size])
```