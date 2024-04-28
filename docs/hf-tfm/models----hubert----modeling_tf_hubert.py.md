# `.\models\hubert\modeling_tf_hubert.py`

```
# 设置文件编码为 UTF-8
# 版权声明
# 根据 Apache 许可证版本 2.0 授权使用此文件
# 除非符合许可证的规定，否则不得使用此文件
# 您可以在以下网址获取许可证的副本
#     http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于"原样"的基础分发的，
# 没有任何明示或暗示的保证或条件，包括但不限于特定用途的适用性和
# 适销性。请查看许可证以获取特定语言的权限和
# 许可证下的限制
""" TensorFlow Hubert 模型。"""

from __future__ import annotations

import warnings
from typing import Any, Optional, Tuple, Union

import numpy as np
import tensorflow as tf

from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import TFBaseModelOutput, TFCausalLMOutput
from ...modeling_tf_utils import (
    TFPreTrainedModel,
    get_initializer,
    keras_serializable,
    unpack_inputs,
)
from ...tf_utils import shape_list, stable_softmax
from ...utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from .configuration_hubert import HubertConfig

# 获取 logger 对象
logger = logging.get_logger(__name__)

# 用于文档的配置
_CONFIG_FOR_DOC = "HubertConfig"

# 预训练模型存档列表
TF_HUBERT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/hubert-base-ls960",
    # 查看所有 Hubert 模型 https://huggingface.co/models?filter=hubert
]

# 定义一个大负数
LARGE_NEGATIVE = -1e8

# 从 transformers.models.wav2vec2.modeling_tf_wav2vec2._sample_without_replacement 复制的函数
def _sample_without_replacement(distribution, num_samples):
    """
    Categorical sampling without replacement is currently not implemented. The gumbel-max trick will do for now - see
    https://github.com/tensorflow/tensorflow/issues/9260 for more info
    """
    z = -tf.math.log(tf.random.uniform(shape_list(distribution), 0, 1))
    _, indices = tf.nn.top_k(distribution + z, num_samples)
    return indices

# 从 transformers.models.wav2vec2.modeling_tf_wav2vec2._scatter_values_on_batch_indices 复制的函数
def _scatter_values_on_batch_indices(values, batch_indices, output_shape):
    """
    Scatter function as in PyTorch with indices in format (batch_dim, indixes)
    """
    indices_shape = shape_list(batch_indices)
    # broadcast batch dim to indices_shape
    broad_casted_batch_dims = tf.reshape(
        tf.broadcast_to(tf.expand_dims(tf.range(indices_shape[0]), axis=-1), indices_shape), [1, -1]
    )
    # transform batch_indices to pair_indices
    pair_indices = tf.transpose(tf.concat([broad_casted_batch_dims, tf.reshape(batch_indices, [1, -1])], 0))
    # scatter values to pair indices
    return tf.scatter_nd(pair_indices, tf.reshape(values, [-1]), output_shape)

# 从 transformers.models.wav2vec2.modeling_tf_wav2vec2._compute_mask_indices 复制的函数
def _compute_mask_indices(
    shape: Tuple[int, int],
    # 定义一个浮点型变量，表示掩码的概率
    mask_prob: float,
    # 定义一个整型变量，表示掩码的长度
    mask_length: int,
    # 定义一个整型变量，表示最小掩码数量，默认为0
    min_masks: int = 0,
# 定义一个函数，用于计算给定形状的随机掩码范围
def compute_random_mask(shape: Tuple[int, int], attention_mask: Optional[tf.Tensor], mask_prob: float, mask_length: int, min_masks: int) -> tf.Tensor:
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
    # 获取批次大小和序列长度
    batch_size, sequence_length = shape

    # 如果掩码长度小于1，则引发值错误
    if mask_length < 1:
        raise ValueError("`mask_length` has to be bigger than 0.")

    # 断言掩码长度小于序列长度
    tf.debugging.assert_less(
        mask_length,
        sequence_length,
        message=(
            f"`mask_length` has to be smaller than `sequence_length`, but got `mask_length`: {mask_length} and"
            f" `sequence_length`: {sequence_length}`"
        ),
    )

    # 计算批次中掩码范围的数量
    num_masked_spans = mask_prob * tf.cast(sequence_length, tf.float32) / mask_length + tf.random.uniform((1,))
    num_masked_spans = tf.maximum(num_masked_spans, min_masks)
    num_masked_spans = tf.cast(num_masked_spans, tf.int32)

    # 确保掩码索引数量小于等于序列长度
    num_masked_spans = tf.math.minimum(sequence_length // mask_length, num_masked_spans)
    num_masked_spans = tf.squeeze(num_masked_spans)

    # 创建用于SpecAugment的掩码
    spec_aug_mask = tf.zeros((batch_size, sequence_length), dtype=tf.int32)

    # 创建均匀分布以进行采样，确保偏移样本小于序列长度
    uniform_dist = tf.ones((batch_size, sequence_length - (mask_length - 1)))

    # 获取要掩码的随机索引
    spec_aug_mask_idxs = _sample_without_replacement(uniform_dist, num_masked_spans)

    # 将掩码索引扩展为掩码范围
    spec_aug_mask_idxs = tf.expand_dims(spec_aug_mask_idxs, -1)
    spec_aug_mask_idxs = tf.tile(spec_aug_mask_idxs, (1, 1, mask_length))
    spec_aug_mask_idxs = tf.reshape(spec_aug_mask_idxs, (batch_size, num_masked_spans * mask_length))

    offsets = tf.range(mask_length)[tf.newaxis, tf.newaxis, :]
    offsets = tf.tile(offsets, (batch_size, num_masked_spans, 1))
    offsets = tf.reshape(offsets, (batch_size, num_masked_spans * mask_length))

    spec_aug_mask_idxs = spec_aug_mask_idxs + offsets

    # 将索引散布到掩码中
    spec_aug_mask = _scatter_values_on_batch_indices(
        tf.ones_like(spec_aug_mask_idxs), spec_aug_mask_idxs, tf.shape(spec_aug_mask)
    )

    return spec_aug_mask
# 从transformers.models.bart.modeling_tf_bart._expand_mask中复制过来的函数
def _expand_mask(mask: tf.Tensor, tgt_len: Optional[int] = None):
    """
    将注意力掩码从`[bsz, seq_len]`扩展为`[bsz, 1, tgt_seq_len, src_seq_len]`。
    """
    # 获取掩码的源序列长度
    src_len = shape_list(mask)[1]
    # 如果未提供目标序列长度，则使用源序列长度
    tgt_len = tgt_len if tgt_len is not None else src_len
    # 创建一个常数张量，值为1.0
    one_cst = tf.constant(1.0)
    # 将掩码转换为与one_cst相同数据类型的张量
    mask = tf.cast(mask, dtype=one_cst.dtype)
    # 在第二维度上复制掩码，扩展为`[bsz, 1, tgt_seq_len, src_seq_len]`
    expanded_mask = tf.tile(mask[:, None, None, :], (1, 1, tgt_len, 1)

    return (one_cst - expanded_mask) * LARGE_NEGATIVE


# 从transformers.models.wav2vec2.modeling_tf_wav2vec2.TFWav2Vec2GroupNorm中复制过来的类，将Wav2Vec2替换为Hubert
class TFHubertGroupNorm(tf.keras.layers.Layer):
    """
    从tensorflow-addons中复制 https://www.tensorflow.org/addons/api_docs/python/tfa/layers/GroupNormalization
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
    # 定义一个方法用于调用自定义层，接受输入并返回处理后的输出
    def call(self, inputs):
        # 获取输入的形状信息
        input_shape = tf.keras.backend.int_shape(inputs)
        # 获取输入张量的形状信息
        tensor_input_shape = tf.shape(inputs)

        # 将输入重塑为分组形式，并返回重塑后的输入和分组形状
        reshaped_inputs, group_shape = self._reshape_into_groups(inputs, input_shape, tensor_input_shape)

        # 对重塑后的输入应用归一化处理
        normalized_inputs = self._apply_normalization(reshaped_inputs, input_shape)

        # 判断是否为实例归一化
        is_instance_norm = (input_shape[self.axis] // self.groups) == 1
        if not is_instance_norm:
            # 如果不是实例归一化，则将输出重塑回原始形状
            outputs = tf.reshape(normalized_inputs, tensor_input_shape)
        else:
            outputs = normalized_inputs

        return outputs

    # 获取自定义层的配置信息
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
        base_config = super().get_config()
        return {**base_config, **config}

    # 计算输出形状
    def compute_output_shape(self, input_shape):
        return input_shape

    # 将输入重塑为分组形式
    def _reshape_into_groups(self, inputs, input_shape, tensor_input_shape):
        group_shape = [tensor_input_shape[i] for i in range(len(input_shape))]
        is_instance_norm = (input_shape[self.axis] // self.groups) == 1
        if not is_instance_norm:
            group_shape[self.axis] = input_shape[self.axis] // self.groups
            group_shape.insert(self.axis, self.groups)
            group_shape = tf.stack(group_shape)
            reshaped_inputs = tf.reshape(inputs, group_shape)
            return reshaped_inputs, group_shape
        else:
            return inputs, group_shape
    # 对输入数据进行归一化处理
    def _apply_normalization(self, reshaped_inputs, input_shape):
        # 获取重塑后输入数据的形状
        group_shape = tf.keras.backend.int_shape(reshaped_inputs)
        # 确定需要进行归一化的轴
        group_reduction_axes = list(range(1, len(group_shape)))
        # 判断是否为实例归一化
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
        # 进行批量归一化
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
        if self.scale:
            gamma = tf.reshape(self.gamma, broadcast_shape)
    
        if self.center:
            beta = tf.reshape(self.beta, broadcast_shape)
        return gamma, beta
    
    # 检查输入形状是否为None
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
    
    # 设置实例归一化的组数
    def _set_number_of_groups_for_instance_norm(self, input_shape):
        dim = input_shape[self.axis]
    
        if self.groups == -1:
            self.groups = dim
    
    # 检查维度大小
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
    
    # 检查轴
    def _check_axis(self):
        if self.axis == 0:
            raise ValueError(
                "You are trying to normalize your batch axis. Do you want to use tf.layer.batch_normalization instead"
            )
    
    # 创建输入规范
    def _create_input_spec(self, input_shape):
        dim = input_shape[self.axis]
        self.input_spec = tf.keras.layers.InputSpec(ndim=len(input_shape), axes={self.axis: dim})
    # 添加 gamma 权重参数，用于缩放
    def _add_gamma_weight(self, input_shape):
        # 获取输入形状中指定轴的维度
        dim = input_shape[self.axis]
        # 创建形状元组
        shape = (dim,)

        # 如果需要缩放
        if self.scale:
            # 添加 gamma 权重参数
            self.gamma = self.add_weight(
                shape=shape,
                name="gamma",
                initializer=self.gamma_initializer,
                regularizer=self.gamma_regularizer,
                constraint=self.gamma_constraint,
            )
        else:
            # 如果不需要缩放，gamma 设置为 None
            self.gamma = None

    # 添加 beta 权重参数，用于偏移
    def _add_beta_weight(self, input_shape):
        # 获取输入形状中指定轴的维度
        dim = input_shape[self.axis]
        # 创建形状元组
        shape = (dim,)

        # 如果需要中心化
        if self.center:
            # 添加 beta 权重参数
            self.beta = self.add_weight(
                shape=shape,
                name="beta",
                initializer=self.beta_initializer,
                regularizer=self.beta_regularizer,
                constraint=self.beta_constraint,
            )
        else:
            # 如果不需要中心化，beta 设置为 None
            self.beta = None

    # 创建广播形状
    def _create_broadcast_shape(self, input_shape):
        # 创建广播形状列表，初始化为全 1
        broadcast_shape = [1] * len(input_shape)
        # 判断是否为实例归一化
        is_instance_norm = (input_shape[self.axis] // self.groups) == 1
        # 如果不是实例归一化
        if not is_instance_norm:
            # 设置广播形状中指定轴的维度
            broadcast_shape[self.axis] = input_shape[self.axis] // self.groups
            # 在指定轴前插入分组数
            broadcast_shape.insert(self.axis, self.groups)
        else:
            # 如果是实例归一化，设置广播形状中指定轴的维度为分组数
            broadcast_shape[self.axis] = self.groups
        # 返回广播形状
        return broadcast_shape
# 从transformers.models.wav2vec2.modeling_tf_wav2vec2.TFWav2Vec2WeightNormConv1D复制而来，将Wav2Vec2改为Hubert
class TFHubertWeightNormConv1D(tf.keras.layers.Conv1D):
    """Adapted from https://www.tensorflow.org/probability/api_docs/python/tfp/layers/weight_norm/WeightNorm"""

    def __init__(self, filters, kernel_size, groups, explicit_padding, **kwargs):
        # 调用父类的构造函数，设置卷积层的参数
        super().__init__(
            filters=filters,
            kernel_size=kernel_size,
            groups=groups,
            padding="valid",
            use_bias=True,
            bias_initializer="he_normal",
            **kwargs,
        )
        self.explicit_padding = explicit_padding
        self.filter_axis = 2
        self.kernel_norm_axes = tf.constant([0, 1])

    def _init_norm(self):
        """Set the norm of the weight vector."""
        # 计算权重向量的范数
        kernel_norm = tf.sqrt(tf.reduce_sum(tf.square(self.weight_v), axis=self.kernel_norm_axes))
        self.weight_g.assign(kernel_norm[:, tf.newaxis, tf.newaxis])

    def _normalize_kernel(self):
        """Generate normalized weights."""
        # 归一化权重
        kernel = tf.nn.l2_normalize(self.weight_v, axis=self.kernel_norm_axes) * tf.transpose(self.weight_g)
        self.kernel = tf.transpose(kernel)

    def build(self, input_shape):
        if not self.built:
            # 构建层
            super().build(input_shape)

            # 初始化权重向量
            self.kernel = tf.Variable(tf.transpose(self.kernel), name="weight_v", trainable=True)
            self.weight_v = self.kernel

            # 添加权重g
            self.weight_g = self.add_weight(
                name="weight_g",
                shape=(int(self.weight_v.shape[self.filter_axis]), 1, 1),
                initializer="ones",
                dtype=self.weight_v.dtype,
                trainable=True,
            )
            self._init_norm()
            # 添加偏置
            self.bias = self.add_weight(name="bias", shape=(self.filters,), initializer="zeros", trainable=True)

    def call(self, inputs):
        # 在call()中对属性进行赋值在TensorFlow中是不推荐的，应该是幂等的
        # 应该替换整个层，使用不继承Conv1D的层，而是调用生成的归一化权重的功能性1D卷积
        self._normalize_kernel()

        # 对输入进行填充
        padded_inputs = tf.pad(inputs, ((0, 0), (self.explicit_padding, self.explicit_padding), (0, 0)))
        # 调用父类的call方法进行卷���操作
        output = super().call(padded_inputs)

        return output


# 从transformers.models.wav2vec2.modeling_tf_wav2vec2.TFWav2Vec2NoLayerNormConvLayer复制而来，将Wav2Vec2改为Hubert
class TFHubertNoLayerNormConvLayer(tf.keras.layers.Layer):
    # 初始化函数，接受配置信息和层编号作为参数
    def __init__(self, config: HubertConfig, layer_id: int = 0, **kwargs: Any) -> None:
        # 调用父类的初始化函数
        super().__init__(**kwargs)
        # 根据层编号获取输入卷积维度
        self.in_conv_dim = config.conv_dim[layer_id] if layer_id > 0 else 1
        # 获取输出卷积维度
        self.out_conv_dim = config.conv_dim[layer_id]

        # 创建一维卷积层
        self.conv = tf.keras.layers.Conv1D(
            filters=self.out_conv_dim,
            kernel_size=config.conv_kernel[layer_id],
            strides=config.conv_stride[layer_id],
            use_bias=config.conv_bias,
            name="conv",
        )
        # 获取激活函数
        self.activation = get_tf_activation(config.feat_extract_activation)

    # 前向传播函数，接受隐藏状态张量作为输入
    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        # 经过卷积层处理隐藏状态
        hidden_states = self.conv(hidden_states)
        # 经过激活函数处理隐藏状态
        hidden_states = self.activation(hidden_states)
        # 返回处理后的隐藏状态
        return hidden_states

    # 构建函数，用于构建卷积层
    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回
        if self.built:
            return
        # 标记为已构建
        self.built = True
        # 如果卷积层存在
        if getattr(self, "conv", None) is not None:
            # 在命名空间下构建卷积层
            with tf.name_scope(self.conv.name):
                # 构建卷积层，指定输入形状
                self.conv.build([None, None, self.in_conv_dim])
# 从transformers.models.wav2vec2.modeling_tf_wav2vec2.TFWav2Vec2LayerNormConvLayer复制代码，并将Wav2Vec2->Hubert
class TFHubertLayerNormConvLayer(tf.keras.layers.Layer):
    def __init__(self, config: HubertConfig, layer_id: int = 0, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        # 初始化输入和输出卷积维度
        self.in_conv_dim = config.conv_dim[layer_id] if layer_id > 0 else 1
        self.out_conv_dim = config.conv_dim[layer_id]

        # 创建卷积层
        self.conv = tf.keras.layers.Conv1D(
            filters=self.out_conv_dim,
            kernel_size=config.conv_kernel[layer_id],
            strides=config.conv_stride[layer_id],
            use_bias=config.conv_bias,
            name="conv",
        )
        # 创建层归一化层
        self.layer_norm = tf.keras.layers.LayerNormalization(name="layer_norm", epsilon=config.layer_norm_eps)
        # 获取激活函数
        self.activation = get_tf_activation(config.feat_extract_activation)

    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        # 进行卷积操作
        hidden_states = self.conv(hidden_states)
        # 进行层归一化操作
        hidden_states = self.layer_norm(hidden_states)
        # 使用激活函数
        hidden_states = self.activation(hidden_states)
        return hidden_states

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "conv", None) is not None:
            with tf.name_scope(self.conv.name):
                # 构建卷积层
                self.conv.build([None, None, self.in_conv_dim])
        if getattr(self, "layer_norm", None) is not None:
            with tf.name_scope(self.layer_norm.name):
                # 构建层归一化层

# 从transformers.models.wav2vec2.modeling_tf_wav2vec2.TFWav2Vec2GroupNormConvLayer复制代码，并将Wav2Vec2->Hubert
class TFHubertGroupNormConvLayer(tf.keras.layers.Layer):
    def __init__(self, config: HubertConfig, layer_id: int = 0, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        # 初始化输入和输出卷积维度
        self.in_conv_dim = config.conv_dim[layer_id] if layer_id > 0 else 1
        self.out_conv_dim = config.conv_dim[layer_id]

        # 创建卷积层
        self.conv = tf.keras.layers.Conv1D(
            filters=self.out_conv_dim,
            kernel_size=config.conv_kernel[layer_id],
            strides=config.conv_stride[layer_id],
            use_bias=config.conv_bias,
            name="conv",
        )
        # 获取激活函数
        self.activation = get_tf_activation(config.feat_extract_activation)
        # 创建组归一化层
        self.layer_norm = TFHubertGroupNorm(groups=self.out_conv_dim, epsilon=config.layer_norm_eps, name="layer_norm")

    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        # 进行卷积操作
        hidden_states = self.conv(hidden_states)
        # 进行组归一化操作
        hidden_states = self.layer_norm(hidden_states)
        # 使用激活函数
        hidden_states = self.activation(hidden_states)
        return hidden_states
    # 构建神经网络模型
    def build(self, input_shape=None):
        # 如果模型已经构建过，则直接返回
        if self.built:
            return
        # 标记模型已经构建
        self.built = True
        # 如果存在卷积层，则构建卷积层
        if getattr(self, "conv", None) is not None:
            # 使用卷积层的名称创建命名空间
            with tf.name_scope(self.conv.name):
                # 构建卷积层，指定输入形状为[None, None, self.in_conv_dim]
                self.conv.build([None, None, self.in_conv_dim])
        # 如果存在层归一化层，则构建层归一化层
        if getattr(self, "layer_norm", None) is not None:
            # 使用层归一化层的名称创建命名空间
            with tf.name_scope(self.layer_norm.name):
                # 构建层归一化层，指定输入形状为[None, None, self.out_conv_dim]
                self.layer_norm.build([None, None, self.out_conv_dim])
# 定义 TFHubertPositionalConvEmbedding 类，继承自 tf.keras.layers.Layer
class TFHubertPositionalConvEmbedding(tf.keras.layers.Layer):
    # 初始化方法，接受 HubertConfig 类型的 config 参数和任意关键字参数
    def __init__(self, config: HubertConfig, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        # 创建 TFHubertWeightNormConv1D 对象，用于卷积操作
        self.conv = TFHubertWeightNormConv1D(
            filters=config.hidden_size,
            kernel_size=config.num_conv_pos_embeddings,
            groups=config.num_conv_pos_embedding_groups,
            explicit_padding=config.num_conv_pos_embeddings // 2,
            name="conv",
        )
        # 创建 TFHubertSamePadLayer 对象，用于填充操作
        self.padding = TFHubertSamePadLayer(config.num_conv_pos_embeddings)
        # 获取激活函数
        self.activation = get_tf_activation(config.feat_extract_activation)
        self.config = config

    # 调用方法，接受 tf.Tensor 类型的 hidden_states 参数，返回 tf.Tensor 类型的结果
    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        # 对隐藏状态进行卷积操作
        hidden_states = self.conv(hidden_states)
        # 对卷积结果进行填充操作
        hidden_states = self.padding(hidden_states)
        # 对填充后的结果应用激活函数
        hidden_states = self.activation(hidden_states)
        return hidden_states

    # 构建方法，接受输入形状 input_shape，默认为 None
    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回
        if self.built:
            return
        self.built = True
        # 如果存在卷积对象，则构建卷积层
        if getattr(self, "conv", None) is not None:
            with tf.name_scope(self.conv.name):
                self.conv.build([None, None, self.config.hidden_size])


# 定义 TFHubertSamePadLayer 类，继承自 tf.keras.layers.Layer
class TFHubertSamePadLayer(tf.keras.layers.Layer):
    # 初始化方法，接受 num_conv_pos_embeddings 参数和任意关键字参数
    def __init__(self, num_conv_pos_embeddings, **kwargs):
        super().__init__(**kwargs)
        # 计算需要移除的填充数量
        self.num_pad_remove = 1 if num_conv_pos_embeddings % 2 == 0 else 0

    # 调用方法，接受 hidden_states 参数，返回处理后的 hidden_states
    def call(self, hidden_states):
        # 如果需要移除填充，则进行截取操作
        if self.num_pad_remove > 0:
            hidden_states = hidden_states[:, : -self.num_pad_remove, :]
        return hidden_states


# 定义 TFHubertFeatureEncoder 类，继承自 tf.keras.layers.Layer
class TFHubertFeatureEncoder(tf.keras.layers.Layer):
    # 初始化方法，接受 HubertConfig 类型的 config 参数和任意关键字参数
    def __init__(self, config: HubertConfig, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        # 根据特征提取的规范类型，创建不同的卷积层列表
        if config.feat_extract_norm == "group":
            conv_layers = [TFHubertGroupNormConvLayer(config, layer_id=0, name=f"conv_layers.{0}")] + [
                TFHubertNoLayerNormConvLayer(config, layer_id=i + 1, name=f"conv_layers.{i+1}")
                for i in range(config.num_feat_extract_layers - 1)
            ]
        elif config.feat_extract_norm == "layer":
            conv_layers = [
                TFHubertLayerNormConvLayer(config, layer_id=i, name=f"conv_layers.{i}")
                for i in range(config.num_feat_extract_layers)
            ]
        else:
            raise ValueError(
                f"`config.feat_extract_norm` is {config.feat_extract_norm}, but has to be one of ['group', 'layer']"
            )
        self.conv_layers = conv_layers

    # 调用方法，接受输入值 input_values，返回处理后的隐藏状态
    def call(self, input_values):
        # 在最后一个维度上扩展输入值
        hidden_states = tf.expand_dims(input_values, -1)
        # 遍历卷积层列表，对隐藏状态进行卷积操作
        for conv_layer in self.conv_layers:
            hidden_states = conv_layer(hidden_states)
        return hidden_states
    # 构建神经网络模型
    def build(self, input_shape=None):
        # 如果模型已经构建完成，则直接返回
        if self.built:
            return
        # 将模型标记为已构建
        self.built = True
        # 遍历所有卷积层
        for conv_layer in self.conv_layers:
            # 使用命名空间为当前卷积层设置名称
            with tf.name_scope(conv_layer.name):
                # 构建当前卷积层，传入输入形状为None
                conv_layer.build(None)
# 定义 TFHubertFeatureExtractor 类，继承自 TFHubertFeatureEncoder 类
class TFHubertFeatureExtractor(TFHubertFeatureEncoder):
    # 初始化方法，接受配置参数和其他关键字参数
    def __init__(self, config, **kwargs):
        # 调用父类的初始化方法
        super().__init__(config, **kwargs)
        # 发出警告，提示该类已被弃用，将在 Transformers v5 中移除，建议使用父类代替
        warnings.warn(
            f"The class `{self.__class__.__name__}` has been depreciated "
            "and will be removed in Transformers v5. "
            f"Use `{self.__class__.__bases__[0].__name__}` instead.",
            FutureWarning,
        )

# 定义 TFHubertFeatureProjection 类，继承自 tf.keras.layers.Layer 类
class TFHubertFeatureProjection(tf.keras.layers.Layer):
    # 初始化方法，接受 HubertConfig 配置参数和其他关键字参数
    def __init__(self, config: HubertConfig, **kwargs):
        # 调用父类的初始化方法
        super().__init__(**kwargs)
        # 创建 LayerNormalization 层
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="layer_norm")
        # 创建 Dense 层，用于特征投影
        self.projection = tf.keras.layers.Dense(
            units=config.hidden_size,
            kernel_initializer=get_initializer(config.initializer_range),
            bias_initializer="zeros",
            name="projection",
        )
        # 创建 Dropout 层，用于特征投影的 dropout
        self.dropout = tf.keras.layers.Dropout(rate=config.feat_proj_dropout)
        # 保存配置参数
        self.config = config

    # 前向传播方法，接受隐藏状态和训练标志，返回特征投影后的隐藏状态
    def call(self, hidden_states: tf.Tensor, training: bool = False) -> tf.Tensor:
        # 对隐藏状态进行 LayerNormalization
        hidden_states = self.layer_norm(hidden_states)
        # 对 LayerNormalization 后的隐藏状态进行投影
        hidden_states = self.projection(hidden_states)
        # 对投影后的隐藏状态进行 dropout
        hidden_states = self.dropout(hidden_states, training=training)
        return hidden_states

    # 构建方法，用于构建层的结构
    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回
        if self.built:
            return
        self.built = True
        # 如果存在 LayerNormalization 层，则构建该层
        if getattr(self, "layer_norm", None) is not None:
            with tf.name_scope(self.layer_norm.name):
                self.layer_norm.build([None, None, self.config.conv_dim[-1]])
        # 如果存在投影层，则构建该层
        if getattr(self, "projection", None) is not None:
            with tf.name_scope(self.projection.name):
                self.projection.build([None, None, self.config.conv_dim[-1]])

# 从 transformers.models.bart.modeling_tf_bart.TFBartAttention 复制的 TFHubertAttention 类
class TFHubertAttention(tf.keras.layers.Layer):
    """Multi-headed attention from "Attention Is All You Need"""

    # 初始化方法，接受嵌入维度、头数、dropout 等参数
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
        **kwargs,
    # 初始化函数，继承父类的初始化方法，并设置一些参数
    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=True, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim

        self.num_heads = num_heads
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.head_dim = embed_dim // num_heads
        # 检查 embed_dim 是否可以被 num_heads 整除
        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder

        # 初始化一些 Dense 层
        self.k_proj = tf.keras.layers.Dense(embed_dim, use_bias=bias, name="k_proj")
        self.q_proj = tf.keras.layers.Dense(embed_dim, use_bias=bias, name="q_proj")
        self.v_proj = tf.keras.layers.Dense(embed_dim, use_bias=bias, name="v_proj")
        self.out_proj = tf.keras.layers.Dense(embed_dim, use_bias=bias, name="out_proj")

    # 将输入张量重塑为指定形状
    def _shape(self, tensor: tf.Tensor, seq_len: int, bsz: int):
        return tf.transpose(tf.reshape(tensor, (bsz, seq_len, self.num_heads, self.head_dim)), (0, 2, 1, 3))

    # 模型调用函数，接受一些输入参数
    def call(
        self,
        hidden_states: tf.Tensor,
        key_value_states: tf.Tensor | None = None,
        past_key_value: Tuple[Tuple[tf.Tensor]] | None = None,
        attention_mask: tf.Tensor | None = None,
        layer_head_mask: tf.Tensor | None = None,
        training: Optional[bool] = False,
    # 构建模型，设置输入形状
    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 构建 k_proj 层
        if getattr(self, "k_proj", None) is not None:
            with tf.name_scope(self.k_proj.name):
                self.k_proj.build([None, None, self.embed_dim])
        # 构建 q_proj 层
        if getattr(self, "q_proj", None) is not None:
            with tf.name_scope(self.q_proj.name):
                self.q_proj.build([None, None, self.embed_dim])
        # 构建 v_proj 层
        if getattr(self, "v_proj", None) is not None:
            with tf.name_scope(self.v_proj.name):
                self.v_proj.build([None, None, self.embed_dim])
        # 构建 out_proj 层
        if getattr(self, "out_proj", None) is not None:
            with tf.name_scope(self.out_proj.name):
                self.out_proj.build([None, None, self.embed_dim])
# 从transformers.models.wav2vec2.modeling_tf_wav2vec2.TFWav2Vec2FeedForward复制代码，并将Wav2Vec2->Hubert
class TFHubertFeedForward(tf.keras.layers.Layer):
    def __init__(self, config: HubertConfig, **kwargs):
        super().__init__(**kwargs)

        # 中间层的dropout
        self.intermediate_dropout = tf.keras.layers.Dropout(config.activation_dropout)

        # 中间层的全连接层
        self.intermediate_dense = tf.keras.layers.Dense(
            units=config.intermediate_size,
            kernel_initializer=get_initializer(config.initializer_range),
            bias_initializer="zeros",
            name="intermediate_dense",
        )
        # 中间层的激活函数
        self.intermediate_act_fn = get_tf_activation(config.hidden_act)

        # 输出层的全连接层
        self.output_dense = tf.keras.layers.Dense(
            units=config.hidden_size,
            kernel_initializer=get_initializer(config.initializer_range),
            bias_initializer="zeros",
            name="output_dense",
        )
        # 输出层的dropout
        self.output_dropout = tf.keras.layers.Dropout(config.hidden_dropout)
        self.config = config

    def call(self, hidden_states: tf.Tensor, training: bool = False) -> tf.Tensor:
        # 中间层的全连接层
        hidden_states = self.intermediate_dense(hidden_states)
        # 中间层的激活函数
        hidden_states = self.intermediate_act_fn(hidden_states)
        # 中间层的dropout
        hidden_states = self.intermediate_dropout(hidden_states, training=training)

        # 输出层的全连接层
        hidden_states = self.output_dense(hidden_states)
        # 输出层的dropout
        hidden_states = self.output_dropout(hidden_states, training=training)
        return hidden_states

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "intermediate_dense", None) is not None:
            with tf.name_scope(self.intermediate_dense.name):
                self.intermediate_dense.build([None, None, self.config.hidden_size])
        if getattr(self, "output_dense", None) is not None:
            with tf.name_scope(self.output_dense.name):
                self.output_dense.build([None, None, self.config.intermediate_size])


# 从transformers.models.wav2vec2.modeling_tf_wav2vec2.TFWav2Vec2EncoderLayer复制代码，并将Wav2Vec2->Hubert
class TFHubertEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, config: HubertConfig, **kwargs):
        super().__init__(**kwargs)
        # 注意力机制
        self.attention = TFHubertAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=False,
            name="attention",
        )
        # dropout
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout)
        # 层归一化
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="layer_norm")
        # 前馈神经网络
        self.feed_forward = TFHubertFeedForward(config, name="feed_forward")
        # 最终的层归一化
        self.final_layer_norm = tf.keras.layers.LayerNormalization(
            epsilon=config.layer_norm_eps, name="final_layer_norm"
        )
        self.config = config
    # 定义一个方法，用于处理模型的前向传播
    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: tf.Tensor | None = None,
        output_attentions: Optional[bool] = False,
        training: bool = False,
    ) -> Tuple[tf.Tensor]:
        # 保存注意力机制之前的隐藏状态，用于残差连接
        attn_residual = hidden_states
        # 调用注意力机制处理隐藏状态
        hidden_states, attn_weights, _ = self.attention(
            hidden_states, attention_mask=attention_mask, training=training
        )
        # 对隐藏状态进行 dropout 处理
        hidden_states = self.dropout(hidden_states, training=training)
        # 将残差连接的隐藏状态和注意力机制处理后的隐藏状态相加
        hidden_states = attn_residual + hidden_states

        # 对隐藏状态进行 Layer Normalization 处理
        hidden_states = self.layer_norm(hidden_states)
        # 将隐藏状态与前馈神经网络的输出相加
        hidden_states = hidden_states + self.feed_forward(hidden_states)
        # 对最终的隐藏状态进行 Layer Normalization 处理
        hidden_states = self.final_layer_norm(hidden_states)

        # 将隐藏状态作为输出
        outputs = (hidden_states,)

        # 如果需要输出注意力权重，则将注意力权重添加到输出中
        if output_attentions:
            outputs += (attn_weights,)

        return outputs

    # 构建模型
    def build(self, input_shape=None):
        # 如果模型已经构建过，则直接返回
        if self.built:
            return
        self.built = True
        # 如果存在注意力机制，则构建注意力机制
        if getattr(self, "attention", None) is not None:
            with tf.name_scope(self.attention.name):
                self.attention.build(None)
        # 如果存在 Layer Normalization，则构建 Layer Normalization
        if getattr(self, "layer_norm", None) is not None:
            with tf.name_scope(self.layer_norm.name):
                self.layer_norm.build([None, None, self.config.hidden_size])
        # 如果存在前馈神经网络，则构建前馈神经网络
        if getattr(self, "feed_forward", None) is not None:
            with tf.name_scope(self.feed_forward.name):
                self.feed_forward.build(None)
        # 如果存在最终的 Layer Normalization，则构建最终的 Layer Normalization
        if getattr(self, "final_layer_norm", None) is not None:
            with tf.name_scope(self.final_layer_norm.name):
                self.final_layer_norm.build([None, None, self.config.hidden_size])
# 从transformers.models.wav2vec2.modeling_tf_wav2vec2.TFWav2Vec2EncoderLayerStableLayerNorm复制代码，并将Wav2Vec2->Hubert
class TFHubertEncoderLayerStableLayerNorm(tf.keras.layers.Layer):
    def __init__(self, config: HubertConfig, **kwargs):
        super().__init__(**kwargs)
        # 初始化注意力层
        self.attention = TFHubertAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=False,
            name="attention",
        )
        # 初始化dropout层
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout)
        # 初始化LayerNormalization层
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="layer_norm")
        # 初始化前馈神经网络层
        self.feed_forward = TFHubertFeedForward(config, name="feed_forward")
        # 初始化最终LayerNormalization层
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
    ) -> Tuple[tf.Tensor]:
        # 保存注意力层的残差连接
        attn_residual = hidden_states
        # LayerNormalization
        hidden_states = self.layer_norm(hidden_states)
        # 注意力计算
        hidden_states, attn_weights, _ = self.attention(
            hidden_states, attention_mask=attention_mask, training=training
        )
        # dropout
        hidden_states = self.dropout(hidden_states, training=training)
        # 残差连接
        hidden_states = attn_residual + hidden_states
        # 前馈神经网络
        hidden_states = hidden_states + self.feed_forward(self.final_layer_norm(hidden_states))

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 构建注意力层
        if getattr(self, "attention", None) is not None:
            with tf.name_scope(self.attention.name):
                self.attention.build(None)
        # 构建LayerNormalization层
        if getattr(self, "layer_norm", None) is not None:
            with tf.name_scope(self.layer_norm.name):
                self.layer_norm.build([None, None, self.config.hidden_size])
        # 构建前馈神经网络层
        if getattr(self, "feed_forward", None) is not None:
            with tf.name_scope(self.feed_forward.name):
                self.feed_forward.build(None)
        # 构建最终LayerNormalization层
        if getattr(self, "final_layer_norm", None) is not None:
            with tf.name_scope(self.final_layer_norm.name):
                self.final_layer_norm.build([None, None, self.config.hidden_size])


# 从transformers.models.wav2vec2.modeling_tf_wav2vec2.TFWav2Vec2Encoder复制代码，并将Wav2Vec2->Hubert
class TFHubertEncoder(tf.keras.layers.Layer):
    # 初始化方法，接受配置参数和其他关键字参数
    def __init__(self, config: HubertConfig, **kwargs):
        # 调用父类的初始化方法
        super().__init__(**kwargs)
        # 保存配置参数
        self.config = config
        # 创建位置卷积嵌入层
        self.pos_conv_embed = TFHubertPositionalConvEmbedding(config, name="pos_conv_embed")
        # 创建 LayerNormalization 层
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="layer_norm")
        # 创建 Dropout 层
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout)
        # 创建多个 Hubert 编码器层
        self.layer = [TFHubertEncoderLayer(config, name=f"layers.{i}") for i in range(config.num_hidden_layers)]

    # 前向传播方法，接受隐藏状态、注意力掩码等参数，返回模型输出
    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: tf.Tensor | None = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
        training: Optional[bool] = False,
    ) -> Union[TFBaseModelOutput, Tuple[tf.Tensor]]:
        # 初始化存储隐藏状态和注意力权重的变量
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        # 如果存在注意力掩码，则将隐藏状态乘以掩码
        if attention_mask is not None:
            hidden_states = hidden_states * tf.expand_dims(attention_mask, -1)
            attention_mask = _expand_mask(attention_mask)
        else:
            attention_mask = None

        # 计算位置嵌入并与隐藏状态相加
        position_embeddings = self.pos_conv_embed(hidden_states)
        hidden_states = hidden_states + position_embeddings
        # LayerNormalization
        hidden_states = self.layer_norm(hidden_states)
        # Dropout
        hidden_states = self.dropout(hidden_states, training=training)

        # 遍历每个编码器层
        for i, layer_module in enumerate(self.layer):
            # 如果需要输出隐藏状态，则保存当前隐藏状态
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 添加 LayerDrop 机制，根据概率跳过编码器层
            dropout_probability = np.random.uniform(0, 1)
            if training and (dropout_probability < self.config.layerdrop):  # 跳过该层
                continue

            # 调用编码器层的前向传播方法
            layer_outputs = layer_module(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                training=training,
            )
            hidden_states = layer_outputs[0]

            # 如果需要输出注意力权重，则保存当前层的注意力权重
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        # 添加最后一层的隐藏状态到输出中
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 根据返回字典标志决定返回结果类型
        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        return TFBaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )
    # 构建模型，如果已经构建过则直接返回
    def build(self, input_shape=None):
        if self.built:
            return
        # 标记模型已经构建
        self.built = True
        # 如果存在位置卷积嵌入层，则构建该层
        if getattr(self, "pos_conv_embed", None) is not None:
            with tf.name_scope(self.pos_conv_embed.name):
                self.pos_conv_embed.build(None)
        # 如果存在层归一化层，则构建该层
        if getattr(self, "layer_norm", None) is not None:
            with tf.name_scope(self.layer_norm.name):
                self.layer_norm.build([None, None, self.config.hidden_size])
        # 如果存在多层，则逐层构建
        if getattr(self, "layer", None) is not None:
            for layer in self.layer:
                with tf.name_scope(layer.name):
                    layer.build(None)
# 从transformers.models.wav2vec2.modeling_tf_wav2vec2.TFWav2Vec2EncoderStableLayerNorm复制代码，并将Wav2Vec2->Hubert
class TFHubertEncoderStableLayerNorm(tf.keras.layers.Layer):
    def __init__(self, config: HubertConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.pos_conv_embed = TFHubertPositionalConvEmbedding(config, name="pos_conv_embed")  # 初始化位置卷积嵌入层
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="layer_norm")  # 初始化层归一化
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout)  # 初始化dropout层
        self.layer = [
            TFHubertEncoderLayerStableLayerNorm(config, name=f"layers.{i}") for i in range(config.num_hidden_layers)
        ]  # 初始化编码器层

    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: tf.Tensor | None = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
        training: Optional[bool] = False,
    ) -> Union[TFBaseModelOutput, Tuple[tf.Tensor]]:
        all_hidden_states = () if output_hidden_states else None  # 初始化所有隐藏状态
        all_self_attentions = () if output_attentions else None  # 初始化所有自注意力

        if attention_mask is not None:
            hidden_states = hidden_states * tf.expand_dims(attention_mask, -1)  # 根据注意力掩码调整隐藏状态
            attention_mask = _expand_mask(attention_mask)  # 扩展注意力掩码
        else:
            attention_mask = None

        position_embeddings = self.pos_conv_embed(hidden_states)  # 获取位置嵌入
        hidden_states = hidden_states + position_embeddings  # 添加位置嵌入到隐藏状态
        hidden_states = self.dropout(hidden_states, training=training)  # 应用dropout

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)  # 添加隐藏状态到所有隐藏状态

            # 添加LayerDrop（参见https://arxiv.org/abs/1909.11556的描述）
            dropout_probability = np.random.uniform(0, 1)
            if training and (dropout_probability < self.config.layerdrop):  # 如果在训练中且随机概率小于layerdrop，则跳过该层
                continue

            layer_outputs = layer_module(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                training=training,
            )  # 获取编码器层输出
            hidden_states = layer_outputs[0]  # 更新隐藏状态

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)  # 添加自注意力到所有自注意力

        hidden_states = self.layer_norm(hidden_states)  # 应用层归一化

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)  # 添加隐藏状态到所有隐藏状态

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)  # 如果不返回字典，则返回元组
        return TFBaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )  # 返回TFBaseModelOutput对象
    # 构建模型，如果已经构建过则直接返回
    def build(self, input_shape=None):
        if self.built:
            return
        # 标记模型已经构建
        self.built = True
        # 如果存在位置卷积嵌入层，则构建该层
        if getattr(self, "pos_conv_embed", None) is not None:
            with tf.name_scope(self.pos_conv_embed.name):
                self.pos_conv_embed.build(None)
        # 如果存在层归一化层，则构建该层
        if getattr(self, "layer_norm", None) is not None:
            with tf.name_scope(self.layer_norm.name):
                self.layer_norm.build([None, None, self.config.hidden_size])
        # 如果存在多层，则逐层构建
        if getattr(self, "layer", None) is not None:
            for layer in self.layer:
                with tf.name_scope(layer.name):
                    layer.build(None)
# 使用 keras_serializable 装饰器将类 TFHubertMainLayer 序列化
@keras_serializable
class TFHubertMainLayer(tf.keras.layers.Layer):
    # 设置配置类为 HubertConfig
    config_class = HubertConfig

    # 初始化方法，接受配置参数 config 和其他关键字参数
    def __init__(self, config: HubertConfig, **kwargs):
        # 调用父类的初始化方法
        super().__init__(**kwargs)
        # 将传入的配置参数赋值给 self.config
        self.config = config
        # 创建 TFHubertFeatureEncoder 对象并赋值给 self.feature_extractor
        self.feature_extractor = TFHubertFeatureEncoder(config, name="feature_extractor")
        # 创建 TFHubertFeatureProjection 对象并赋值给 self.feature_projection
        self.feature_projection = TFHubertFeatureProjection(config, name="feature_projection")

        # 根据配置参数中的 do_stable_layer_norm 判断创建 TFHubertEncoderStableLayerNorm 或 TFHubertEncoder 对象
        if config.do_stable_layer_norm:
            self.encoder = TFHubertEncoderStableLayerNorm(config, name="encoder")
        else:
            self.encoder = TFHubertEncoder(config, name="encoder")

    # 构建方法，用于构建层
    def build(self, input_shape=None):
        # 添加一个形状为 (self.config.hidden_size,) 的可训练权重 masked_spec_embed
        self.masked_spec_embed = self.add_weight(
            shape=(self.config.hidden_size,), initializer="uniform", trainable=True, name="masked_spec_embed"
        )

        # 如果已经构建过，则直接返回
        if self.built:
            return
        # 标记为已构建
        self.built = True
        # 如果存在 feature_extractor 属性，则构建 feature_extractor
        if getattr(self, "feature_extractor", None) is not None:
            with tf.name_scope(self.feature_extractor.name):
                self.feature_extractor.build(None)
        # 如果存在 feature_projection 属性，则构建 feature_projection
        if getattr(self, "feature_projection", None) is not None:
            with tf.name_scope(self.feature_projection.name):
                self.feature_projection.build(None)
        # 如果存在 encoder 属性，则构建 encoder
        if getattr(self, "encoder", None) is not None:
            with tf.name_scope(self.encoder.name):
                self.encoder.build(None)

    # 获取特征提取器输出长度的方法
    def _get_feat_extract_output_lengths(self, input_lengths: tf.Tensor):
        """
        Computes the output length of the convolutional layers
        """

        # 定义计算卷积层输出长度的函数
        def _conv_out_length(input_length, kernel_size, stride):
            # 1D 卷积层输出长度的公式取自 https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
            return (input_length - kernel_size) // stride + 1

        # 遍历配置参数中的卷积核大小和步长，计算输出长度
        for kernel_size, stride in zip(self.config.conv_kernel, self.config.conv_stride):
            input_lengths = _conv_out_length(input_lengths, kernel_size, stride)

        return input_lengths
    def _mask_hidden_states(self, hidden_states: tf.Tensor, mask_time_indices: tf.Tensor | None = None):
        """
        Masks extracted features along time axis and/or along feature axis according to
        [SpecAugment](https://arxiv.org/abs/1904.08779).
        """
        # 获取隐藏状态的形状信息
        batch_size, sequence_length, hidden_size = shape_list(hidden_states)

        # 检查是否需要应用 SpecAugment
        if not getattr(self.config, "apply_spec_augment", True):
            return hidden_states

        if mask_time_indices is not None:
            # 根据给定的 mask_time_indices 在时间轴上应用 SpecAugment
            hidden_states = tf.where(
                tf.cast(mask_time_indices[:, :, tf.newaxis], tf.bool),
                self.masked_spec_embed[tf.newaxis, tf.newaxis, :],
                hidden_states,
            )

        elif self.config.mask_time_prob > 0:
            # 生成索引并在时间轴上应用 SpecAugment
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

        # 在特征轴上应用 SpecAugment
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
        output_attentions: tf.Tensor | None = None,
        output_hidden_states: tf.Tensor | None = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
        **kwargs: Any,
        ):
        # 使用特征提取器提取特征，将输入值转换为浮点数类型
        hidden_states = self.feature_extractor(tf.cast(input_values, tf.float32), training=training)

        if attention_mask is not None:
            # 根据卷积公式计算真实输出长度
            output_lengths = self._get_feat_extract_output_lengths(tf.reduce_sum(attention_mask, -1))

            # 根据输出长度创建序列掩码
            attention_mask = tf.sequence_mask(
                output_lengths, maxlen=shape_list(hidden_states)[1], dtype=hidden_states.dtype
            )

        # 使用特征投影将隐藏状态投影到指定维度
        hidden_states = self.feature_projection(hidden_states, training=training)

        # 获取关键字参数中的时间索引掩码
        mask_time_indices = kwargs.get("mask_time_indices", None)
        if training:
            # 在训练模式下对隐藏状态进行掩码处理
            hidden_states = self._mask_hidden_states(hidden_states, mask_time_indices=mask_time_indices)

        # 使用编码器处理隐藏状态
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
            # 如果不返回字典，则返回隐藏状态和其他输出
            return (hidden_states,) + encoder_outputs[1:]

        # 返回 TFBaseModelOutput 对象，包含最后隐藏状态、隐藏状态和注意力权重
        return TFBaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
class TFHubertPreTrainedModel(TFPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 设置配置类为HubertConfig
    config_class = HubertConfig
    # 设置基础模型前缀为"hubert"
    base_model_prefix = "hubert"
    # 设置主输入名称为"input_values"
    main_input_name = "input_values"

    @property
    # 定义输入签名，指定输入的形状和数据类型
    def input_signature(self):
        return {
            "input_values": tf.TensorSpec((None, 16000), tf.float32, name="input_values"),
            "attention_mask": tf.TensorSpec((None, None), tf.int32, name="attention_mask"),
            "token_type_ids": tf.TensorSpec((None, None), tf.int32, name="token_type_ids"),
        }

    def __init__(self, config, *inputs, **kwargs):
        # 调用父类的初始化方法
        super().__init__(config, *inputs, **kwargs)
        # 输出警告信息，指出该模型不支持在CPU上进行反向传播操作，需要使用GPU或TPU
        logger.warning(
            f"\n{self.__class__.__name__} has backpropagation operations that are NOT supported on CPU. If you wish "
            "to train/fine-tune this model, you need a GPU or a TPU"
        )


HUBERT_START_DOCSTRING = r"""

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
        config ([`HubertConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
# 定义 Hubert 模型的输入文档字符串
HUBERT_INPUTS_DOCSTRING = r"""
"""

# 定义 TFHubertModel 类，继承自 TFHubertPreTrainedModel
@add_start_docstrings(
    "The bare TFHubert Model transformer outputing raw hidden-states without any specific head on top.",
    HUBERT_START_DOCSTRING,
)
class TFHubertModel(TFHubertPreTrainedModel):
    # 初始化方法
    def __init__(self, config: HubertConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.config = config
        self.hubert = TFHubertMainLayer(config, name="hubert")

    # 定义 call 方法，用于模型前向传播
    @add_start_docstrings_to_model_forward(HUBERT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFBaseModelOutput, config_class=_CONFIG_FOR_DOC)
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
        >>> from transformers import AutoProcessor, TFHubertModel
        >>> from datasets import load_dataset
        >>> import soundfile as sf

        >>> processor = AutoProcessor.from_pretrained("facebook/hubert-large-ls960-ft")
        >>> model = TFHubertModel.from_pretrained("facebook/hubert-large-ls960-ft")

        >>> def map_to_array(batch):
        ...     speech, _ = sf.read(batch["file"])
        ...     batch["speech"] = speech
        ...     return batch

        >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        >>> ds = ds.map(map_to_array)

        >>> input_values = processor(ds["speech"][0], return_tensors="tf").input_values  # Batch size 1
        >>> hidden_states = model(input_values).last_hidden_state
        ```"""

        # 设置输出隐藏状态、输出注意力、返回字典的默认值
        output_hidden_states = output_hidden_states if output_hidden_states else self.config.output_hidden_states
        output_attentions = output_attentions if output_attentions else self.config.output_attentions
        return_dict = return_dict if return_dict else self.config.return_dict

        # 调用 Hubert 模型的前向传播
        outputs = self.hubert(
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

        return outputs
    # 构建模型，如果已经构建过则直接返回
    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回
        if self.built:
            return
        # 标记为已构建
        self.built = True
        # 如果存在 hubert 属性
        if getattr(self, "hubert", None) is not None:
            # 在指定的命名空间下构建 hubert 属性
            with tf.name_scope(self.hubert.name):
                self.hubert.build(None)
# 为 TFHubertForCTC 类添加文档字符串，描述其为带有 Connectionist Temporal Classification (CTC) 的语言建模头部的 TFHubert 模型
@add_start_docstrings(
    """TFHubert Model with a `language modeling` head on top for Connectionist Temporal Classification (CTC).""",
    HUBERT_START_DOCSTRING,
)

# 定义 TFHubertForCTC 类，继承自 TFHubertPreTrainedModel 类
class TFHubertForCTC(TFHubertPreTrainedModel):
    # 初始化方法，接受 HubertConfig 类型的配置参数
    def __init__(self, config: HubertConfig, *inputs, **kwargs):
        # 调用父类的初始化方法
        super().__init__(config, *inputs, **kwargs)

        # 创建 TFHubertMainLayer 对象，命名为 "hubert"
        self.hubert = TFHubertMainLayer(config, name="hubert")
        # 创建 Dropout 层，使用配置中的 final_dropout 参数
        self.dropout = tf.keras.layers.Dropout(config.final_dropout)
        # 创建 Dense 层，输出大小为配置中的 vocab_size 参数，命名为 "lm_head"
        self.lm_head = tf.keras.layers.Dense(config.vocab_size, name="lm_head")
        # 设置输出隐藏大小为配置中的 output_hidden_size 参数，如果配置中有 "add_adapter" 属性且为真，则使用 output_hidden_size，否则使用 hidden_size

    # 冻结特征提取器，禁用特征编码器的梯度计算，使其在训练期间不会更新参数
    def freeze_feature_extractor(self):
        warnings.warn(
            "The method `freeze_feature_extractor` is deprecated and will be removed in Transformers v5. "
            "Please use the equivalent `freeze_feature_encoder` method instead.",
            FutureWarning,
        )
        self.freeze_feature_encoder()

    # 冻结特征编码器，禁用特征编码器的梯度计算，使其在训练期间不会更新参数
    def freeze_feature_encoder(self):
        self.hubert.feature_extractor.trainable = False

    # 调用方法，接受多个输入参数，包括输入值、注意力掩码、标记类型 ID 等
    @add_start_docstrings_to_model_forward(HUBERT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFCausalLMOutput, config_class=_CONFIG_FOR_DOC)
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
        labels: tf.Tensor | None = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: Optional[bool] = False,
    # 构建方法，用于构建模型
    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回
        if self.built:
            return
        self.built = True
        # 如果存在 "hubert" 属性，则构建 "hubert" 对象
        if getattr(self, "hubert", None) is not None:
            with tf.name_scope(self.hubert.name):
                self.hubert.build(None)
        # 如果存在 "lm_head" 属性，则构建 "lm_head" 对象
        if getattr(self, "lm_head", None) is not None:
            with tf.name_scope(self.lm_head.name):
                self.lm_head.build([None, None, self.output_hidden_size])
```