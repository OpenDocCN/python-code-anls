# `.\models\hubert\modeling_tf_hubert.py`

```
# 设置编码为 UTF-8
# 版权声明，指明版权归 Fairseq 作者和 HuggingFace Inc. 团队所有
#
# 根据 Apache License, Version 2.0 许可证，除非符合许可证要求，否则不得使用此文件
# 您可以在以下网址获取许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则按"现状"分发软件
# 没有任何明示或暗示的担保或条件。有关详细信息，请参阅许可证

""" TensorFlow Hubert 模型."""

from __future__ import annotations

# 引入警告模块
import warnings
# 引入类型提示
from typing import Any, Optional, Tuple, Union

# 引入 numpy 库，并命名为 np
import numpy as np
# 引入 TensorFlow 库，并命名为 tf
import tensorflow as tf

# 引入相关模块和函数
from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import TFBaseModelOutput, TFCausalLMOutput
from ...modeling_tf_utils import (
    TFPreTrainedModel,
    get_initializer,
    keras,
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
# 引入 Hubert 模型的配置文件
from .configuration_hubert import HubertConfig

# 获取日志记录器对象
logger = logging.get_logger(__name__)

# 用于文档的配置名
_CONFIG_FOR_DOC = "HubertConfig"

# 预训练模型存档列表
TF_HUBERT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/hubert-base-ls960",
    # 查看所有 Hubert 模型，请访问 https://huggingface.co/models?filter=hubert
]

# 定义一个大负数常量
LARGE_NEGATIVE = -1e8


# 从 transformers.models.wav2vec2.modeling_tf_wav2vec2._sample_without_replacement 复制而来
def _sample_without_replacement(distribution, num_samples):
    """
    未实现的无重复分类抽样。目前可以使用 Gumbel-max 技巧代替 - 参见
    https://github.com/tensorflow/tensorflow/issues/9260 了解更多信息
    """
    # 使用 Gumbel-max 技巧进行抽样
    z = -tf.math.log(tf.random.uniform(shape_list(distribution), 0, 1))
    _, indices = tf.nn.top_k(distribution + z, num_samples)
    return indices


# 从 transformers.models.wav2vec2.modeling_tf_wav2vec2._scatter_values_on_batch_indices 复制而来
def _scatter_values_on_batch_indices(values, batch_indices, output_shape):
    """
    类似于 PyTorch 中的 scatter 函数，使用格式为 (batch_dim, indices) 的索引
    """
    indices_shape = shape_list(batch_indices)
    # 将批次维度广播到 indices_shape
    broad_casted_batch_dims = tf.reshape(
        tf.broadcast_to(tf.expand_dims(tf.range(indices_shape[0]), axis=-1), indices_shape), [1, -1]
    )
    # 将 batch_indices 转换为 pair_indices
    pair_indices = tf.transpose(tf.concat([broad_casted_batch_dims, tf.reshape(batch_indices, [1, -1])], 0))
    # 将值 values 散布到 pair_indices 上
    return tf.scatter_nd(pair_indices, tf.reshape(values, [-1]), output_shape)


# 从 transformers.models.wav2vec2.modeling_tf_wav2vec2._compute_mask_indices 复制而来
def _compute_mask_indices(
    shape: Tuple[int, int],
    # 定义一个名为 shape 的变量，其类型为元组，包含两个整数值，分别表示形状的尺寸
    mask_prob: float,
    # 定义一个名为 mask_prob 的变量，其类型为浮点数，表示掩码生成的概率
    mask_length: int,
    # 定义一个名为 mask_length 的变量，其类型为整数，表示每个掩码的长度
    min_masks: int = 0,
    # 定义一个名为 min_masks 的变量，其类型为整数，默认值为 0，表示最少需要的掩码数量
def compute_random_mask_spans(shape: Tuple[int, int],
                              attention_mask: Optional[tf.Tensor] = None,
                              mask_prob: float = 0.15,
                              mask_length: int = 10,
                              min_masks: int = 0) -> tf.Tensor:
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

    Adapted from fairseq's data_utils.py.
    """

    # Extract batch size and sequence length from the shape tuple
    batch_size, sequence_length = shape

    # Check if mask length is valid
    if mask_length < 1:
        raise ValueError("`mask_length` has to be bigger than 0.")

    # Assert that mask length is smaller than sequence length
    tf.debugging.assert_less(
        mask_length,
        sequence_length,
        message=(
            f"`mask_length` has to be smaller than `sequence_length`, but got `mask_length`: {mask_length} and"
            f" `sequence_length`: {sequence_length}`"
        ),
    )

    # Compute the number of masked spans in the batch
    num_masked_spans = mask_prob * tf.cast(sequence_length, tf.float32) / mask_length + tf.random.uniform((1,))
    num_masked_spans = tf.maximum(num_masked_spans, min_masks)
    num_masked_spans = tf.cast(num_masked_spans, tf.int32)

    # Ensure num masked indices <= sequence length
    num_masked_spans = tf.math.minimum(sequence_length // mask_length, num_masked_spans)
    num_masked_spans = tf.squeeze(num_masked_spans)

    # Initialize the specAugment mask
    spec_aug_mask = tf.zeros((batch_size, sequence_length), dtype=tf.int32)

    # Create a uniform distribution to sample from, ensuring offset samples are < sequence_length
    uniform_dist = tf.ones((batch_size, sequence_length - (mask_length - 1)))

    # Get random indices to mask using _sample_without_replacement function
    spec_aug_mask_idxs = _sample_without_replacement(uniform_dist, num_masked_spans)

    # Expand masked indices to masked spans
    spec_aug_mask_idxs = tf.expand_dims(spec_aug_mask_idxs, -1)
    spec_aug_mask_idxs = tf.tile(spec_aug_mask_idxs, (1, 1, mask_length))
    spec_aug_mask_idxs = tf.reshape(spec_aug_mask_idxs, (batch_size, num_masked_spans * mask_length))

    # Create offsets for each mask span
    offsets = tf.range(mask_length)[tf.newaxis, tf.newaxis, :]
    offsets = tf.tile(offsets, (batch_size, num_masked_spans, 1))
    offsets = tf.reshape(offsets, (batch_size, num_masked_spans * mask_length))

    # Apply offsets to the mask indices
    spec_aug_mask_idxs = spec_aug_mask_idxs + offsets

    # Scatter indices to mask using _scatter_values_on_batch_indices function
    spec_aug_mask = _scatter_values_on_batch_indices(
        tf.ones_like(spec_aug_mask_idxs), spec_aug_mask_idxs, tf.shape(spec_aug_mask)
    )

    return spec_aug_mask
# Copied from transformers.models.bart.modeling_tf_bart._expand_mask
def _expand_mask(mask: tf.Tensor, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    # 获取输入张量的第二维长度，即序列长度
    src_len = shape_list(mask)[1]
    # 如果未提供目标长度，则默认使用源长度
    tgt_len = tgt_len if tgt_len is not None else src_len
    # 创建常量张量，数值为1.0
    one_cst = tf.constant(1.0)
    # 将输入的 mask 转换为浮点型张量
    mask = tf.cast(mask, dtype=one_cst.dtype)
    # 在第二维和第三维上复制 mask 张量，扩展为 `[bsz, 1, tgt_len, src_len]`
    expanded_mask = tf.tile(mask[:, None, None, :], (1, 1, tgt_len, 1))

    return (one_cst - expanded_mask) * LARGE_NEGATIVE


# Copied from transformers.models.wav2vec2.modeling_tf_wav2vec2.TFWav2Vec2GroupNorm with Wav2Vec2->Hubert
class TFHubertGroupNorm(keras.layers.Layer):
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
        # 设置 GroupNormalization 的参数
        self.groups = groups
        self.axis = axis
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = keras.initializers.get(beta_initializer)
        self.gamma_initializer = keras.initializers.get(gamma_initializer)
        self.beta_regularizer = keras.regularizers.get(beta_regularizer)
        self.gamma_regularizer = keras.regularizers.get(gamma_regularizer)
        self.beta_constraint = keras.constraints.get(beta_constraint)
        self.gamma_constraint = keras.constraints.get(gamma_constraint)
        self._check_axis()

    def build(self, input_shape):
        # 检查输入张量的形状是否为 None
        self._check_if_input_shape_is_none(input_shape)
        # 设置实例标准化中的组数
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
    # 定义一个方法，用于处理输入数据
    def call(self, inputs):
        # 获取输入数据的静态形状
        input_shape = keras.backend.int_shape(inputs)
        # 获取输入数据的动态形状
        tensor_input_shape = tf.shape(inputs)

        # 调用内部方法对输入数据进行分组重塑操作
        reshaped_inputs, group_shape = self._reshape_into_groups(inputs, input_shape, tensor_input_shape)

        # 对重塑后的数据应用规范化操作
        normalized_inputs = self._apply_normalization(reshaped_inputs, input_shape)

        # 判断是否为实例规范化
        is_instance_norm = (input_shape[self.axis] // self.groups) == 1
        if not is_instance_norm:
            # 如果不是实例规范化，将规范化后的数据重新整形为原始输入数据的形状
            outputs = tf.reshape(normalized_inputs, tensor_input_shape)
        else:
            # 如果是实例规范化，则直接使用规范化后的数据作为输出
            outputs = normalized_inputs

        # 返回处理后的输出数据
        return outputs

    # 获取当前层的配置信息，用于模型保存和加载时使用
    def get_config(self):
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
        # 调用父类方法获取基础配置信息，并合并当前层的配置信息
        base_config = super().get_config()
        return {**base_config, **config}

    # 计算输出形状，这里直接返回输入形状
    def compute_output_shape(self, input_shape):
        return input_shape

    # 内部方法：将输入数据重塑为分组形式
    def _reshape_into_groups(self, inputs, input_shape, tensor_input_shape):
        # 复制输入数据的形状作为分组形状的基础
        group_shape = [tensor_input_shape[i] for i in range(len(input_shape))]
        # 判断是否为实例规范化
        is_instance_norm = (input_shape[self.axis] // self.groups) == 1
        if not is_instance_norm:
            # 如果不是实例规范化，根据分组数调整分组形状
            group_shape[self.axis] = input_shape[self.axis] // self.groups
            group_shape.insert(self.axis, self.groups)
            group_shape = tf.stack(group_shape)
            # 对输入数据进行形状重塑操作
            reshaped_inputs = tf.reshape(inputs, group_shape)
            return reshaped_inputs, group_shape
        else:
            # 如果是实例规范化，则直接返回原始输入数据和分组形状
            return inputs, group_shape

    # 内部方法：对重塑后的数据应用规范化操作
    def _apply_normalization(self, reshaped_inputs, input_shape):
        # 获取分组后数据的形状
        group_shape = keras.backend.int_shape(reshaped_inputs)
        # 确定规范化操作的约简轴
        group_reduction_axes = list(range(1, len(group_shape)))
        # 判断是否为实例规范化
        is_instance_norm = (input_shape[self.axis] // self.groups) == 1
        if not is_instance_norm:
            # 如果不是实例规范化，调整约简轴的位置
            axis = -2 if self.axis == -1 else self.axis - 1
        else:
            axis = -1 if self.axis == -1 else self.axis - 1
        group_reduction_axes.pop(axis)

        # 计算分组均值和方差
        mean, variance = tf.nn.moments(reshaped_inputs, group_reduction_axes, keepdims=True)

        # 获取调整后的权重参数
        gamma, beta = self._get_reshaped_weights(input_shape)

        # 对重塑后的数据应用批量规范化操作
        normalized_inputs = tf.nn.batch_normalization(
            reshaped_inputs,
            mean=mean,
            variance=variance,
            scale=gamma,
            offset=beta,
            variance_epsilon=self.epsilon,
        )
        return normalized_inputs
    # 获取重塑后的权重，根据输入形状创建广播形状
    def _get_reshaped_weights(self, input_shape):
        broadcast_shape = self._create_broadcast_shape(input_shape)
        gamma = None
        beta = None
        # 如果启用了标准化参数，将 gamma 重塑为广播形状
        if self.scale:
            gamma = tf.reshape(self.gamma, broadcast_shape)

        # 如果启用了中心化参数，将 beta 重塑为广播形状
        if self.center:
            beta = tf.reshape(self.beta, broadcast_shape)
        return gamma, beta

    # 检查输入形状是否有未定义的维度
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

    # 为实例标准化设置组数
    def _set_number_of_groups_for_instance_norm(self, input_shape):
        dim = input_shape[self.axis]

        # 如果未指定组数，将组数设置为输入张量的维度
        if self.groups == -1:
            self.groups = dim

    # 检查维度的大小是否符合要求
    def _check_size_of_dimensions(self, input_shape):
        dim = input_shape[self.axis]
        # 检查组数是否超过通道数
        if dim < self.groups:
            raise ValueError(
                "Number of groups ("
                + str(self.groups)
                + ") cannot be more than the number of channels ("
                + str(dim)
                + ")."
            )

        # 检查组数是否是通道数的倍数
        if dim % self.groups != 0:
            raise ValueError(
                "Number of groups ("
                + str(self.groups)
                + ") must be a multiple of the number of channels ("
                + str(dim)
                + ")."
            )

    # 检查是否尝试标准化批处理轴
    def _check_axis(self):
        if self.axis == 0:
            raise ValueError(
                "You are trying to normalize your batch axis. Do you want to use tf.layer.batch_normalization instead"
            )

    # 创建输入规范
    def _create_input_spec(self, input_shape):
        dim = input_shape[self.axis]
        # 根据输入形状创建输入规范
        self.input_spec = keras.layers.InputSpec(ndim=len(input_shape), axes={self.axis: dim})

    # 添加 gamma 权重
    def _add_gamma_weight(self, input_shape):
        dim = input_shape[self.axis]
        shape = (dim,)
        
        # 如果启用了标准化，添加 gamma 权重
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

        # 如果启用了中心化，添加 beta 权重
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
    # 定义一个方法用于创建广播形状，根据输入的形状来确定广播后的形状
    def _create_broadcast_shape(self, input_shape):
        # 创建一个与输入形状长度相同的列表，初始值全部为1，用于构建广播形状
        broadcast_shape = [1] * len(input_shape)
        # 判断是否是实例归一化，这里通过检查特定轴上的尺寸是否等于组数来确定
        is_instance_norm = (input_shape[self.axis] // self.groups) == 1
        # 如果不是实例归一化
        if not is_instance_norm:
            # 将广播形状中特定轴的尺寸设置为输入形状中特定轴的尺寸除以组数的结果
            broadcast_shape[self.axis] = input_shape[self.axis] // self.groups
            # 在特定轴前插入组数，以便于构建正确的广播形状
            broadcast_shape.insert(self.axis, self.groups)
        else:
            # 如果是实例归一化，则直接将广播形状中特定轴的尺寸设置为组数
            broadcast_shape[self.axis] = self.groups
        # 返回构建好的广播形状
        return broadcast_shape
# 从 transformers.models.wav2vec2.modeling_tf_wav2vec2.TFWav2Vec2WeightNormConv1D 复制而来，将 Wav2Vec2 改为 Hubert
class TFHubertWeightNormConv1D(keras.layers.Conv1D):
    """从 https://www.tensorflow.org/probability/api_docs/python/tfp/layers/weight_norm/WeightNorm 改编"""

    def __init__(self, filters, kernel_size, groups, explicit_padding, **kwargs):
        # 调用 Conv1D 的初始化方法，设定卷积核参数
        super().__init__(
            filters=filters,
            kernel_size=kernel_size,
            groups=groups,
            padding="valid",  # 使用有效填充方式
            use_bias=True,  # 使用偏置
            bias_initializer="he_normal",  # 偏置初始化方式为 he_normal
            **kwargs,
        )
        # 设置显式填充和卷积的通道方向
        self.explicit_padding = explicit_padding
        self.filter_axis = 2  # 卷积核的轴数
        self.kernel_norm_axes = tf.constant([0, 1])  # 卷积核的归一化轴

    def _init_norm(self):
        """设置权重向量的范数。"""
        kernel_norm = tf.sqrt(tf.reduce_sum(tf.square(self.weight_v), axis=self.kernel_norm_axes))
        self.weight_g.assign(kernel_norm[:, tf.newaxis, tf.newaxis])

    def _normalize_kernel(self):
        """生成归一化的权重。"""
        kernel = tf.nn.l2_normalize(self.weight_v, axis=self.kernel_norm_axes) * tf.transpose(self.weight_g)
        self.kernel = tf.transpose(kernel)

    def build(self, input_shape):
        if not self.built:
            super().build(input_shape)

            # 初始化权重向量并设为可训练
            self.kernel = tf.Variable(tf.transpose(self.kernel), name="weight_v", trainable=True)
            self.weight_v = self.kernel

            # 添加权重 g，初始化为全1，设为可训练
            self.weight_g = self.add_weight(
                name="weight_g",
                shape=(int(self.weight_v.shape[self.filter_axis]), 1, 1),
                initializer="ones",
                dtype=self.weight_v.dtype,
                trainable=True,
            )
            # 初始化权重向量的范数
            self._init_norm()
            # 添加偏置，并初始化为0，设为可训练
            self.bias = self.add_weight(name="bias", shape=(self.filters,), initializer="zeros", trainable=True)

    def call(self, inputs):
        # TODO Matt: 在 call() 中对属性进行赋值在 TensorFlow 中是不正确的，应该保持幂等性。
        #            这整个层应该被替换为一个不继承 Conv1D 的层，而是调用一个生成归一化权重的函数性1D卷积。
        self._normalize_kernel()

        # 对输入进行显式填充
        padded_inputs = tf.pad(inputs, ((0, 0), (self.explicit_padding, self.explicit_padding), (0, 0)))
        # 调用父类 Conv1D 的 call 方法进行卷积运算
        output = super().call(padded_inputs)

        return output
    # 初始化方法，用于设置对象的初始状态
    def __init__(self, config: HubertConfig, layer_id: int = 0, **kwargs: Any) -> None:
        # 调用父类的初始化方法，传递额外的关键字参数
        super().__init__(**kwargs)
        
        # 设置输入卷积维度为配置对象中的 conv_dim[layer_id]，若 layer_id > 0 则取对应的值，否则设为 1
        self.in_conv_dim = config.conv_dim[layer_id] if layer_id > 0 else 1
        
        # 设置输出卷积维度为配置对象中的 conv_dim[layer_id]
        self.out_conv_dim = config.conv_dim[layer_id]

        # 创建一个 1D 卷积层对象，设置滤波器数量、卷积核大小、步长、是否使用偏置，并命名为 "conv"
        self.conv = keras.layers.Conv1D(
            filters=self.out_conv_dim,
            kernel_size=config.conv_kernel[layer_id],
            strides=config.conv_stride[layer_id],
            use_bias=config.conv_bias,
            name="conv",
        )
        
        # 获取激活函数，根据配置中的 feat_extract_activation 来选择
        self.activation = get_tf_activation(config.feat_extract_activation)

    # 调用方法，用于执行前向传播计算
    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        # 对输入张量进行一维卷积操作
        hidden_states = self.conv(hidden_states)
        
        # 应用激活函数到卷积后的张量
        hidden_states = self.activation(hidden_states)
        
        # 返回处理后的张量作为输出
        return hidden_states

    # 构建方法，用于构建层的变量和权重，确保在首次调用 call 方法时已经构建
    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回
        if self.built:
            return
        
        # 标记为已构建状态
        self.built = True
        
        # 如果存在卷积层对象，则在名称作用域下构建卷积层，指定输入形状为 [None, None, self.in_conv_dim]
        if getattr(self, "conv", None) is not None:
            with tf.name_scope(self.conv.name):
                self.conv.build([None, None, self.in_conv_dim])
# 从transformers.models.wav2vec2.modeling_tf_wav2vec2.TFWav2Vec2LayerNormConvLayer复制代码，将Wav2Vec2改为Hubert
class TFHubertLayerNormConvLayer(keras.layers.Layer):
    def __init__(self, config: HubertConfig, layer_id: int = 0, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        # 初始化卷积层的输入维度和输出维度
        self.in_conv_dim = config.conv_dim[layer_id] if layer_id > 0 else 1
        self.out_conv_dim = config.conv_dim[layer_id]

        # 创建一个一维卷积层对象
        self.conv = keras.layers.Conv1D(
            filters=self.out_conv_dim,
            kernel_size=config.conv_kernel[layer_id],
            strides=config.conv_stride[layer_id],
            use_bias=config.conv_bias,
            name="conv",
        )
        # 创建一个层归一化层对象
        self.layer_norm = keras.layers.LayerNormalization(name="layer_norm", epsilon=config.layer_norm_eps)
        # 获取激活函数对象
        self.activation = get_tf_activation(config.feat_extract_activation)

    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        # 执行一维卷积操作
        hidden_states = self.conv(hidden_states)
        # 执行层归一化操作
        hidden_states = self.layer_norm(hidden_states)
        # 执行激活函数操作
        hidden_states = self.activation(hidden_states)
        return hidden_states

    def build(self, input_shape=None):
        # 如果已经构建则直接返回
        if self.built:
            return
        self.built = True
        # 构建卷积层对象
        if getattr(self, "conv", None) is not None:
            with tf.name_scope(self.conv.name):
                self.conv.build([None, None, self.in_conv_dim])
        # 构建层归一化层对象
        if getattr(self, "layer_norm", None) is not None:
            with tf.name_scope(self.layer_norm.name):
                self.layer_norm.build([None, None, self.out_conv_dim])


# 从transformers.models.wav2vec2.modeling_tf_wav2vec2.TFWav2Vec2GroupNormConvLayer复制代码，将Wav2Vec2改为Hubert
class TFHubertGroupNormConvLayer(keras.layers.Layer):
    def __init__(self, config: HubertConfig, layer_id: int = 0, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        # 初始化卷积层的输入维度和输出维度
        self.in_conv_dim = config.conv_dim[layer_id] if layer_id > 0 else 1
        self.out_conv_dim = config.conv_dim[layer_id]

        # 创建一个一维卷积层对象
        self.conv = keras.layers.Conv1D(
            filters=self.out_conv_dim,
            kernel_size=config.conv_kernel[layer_id],
            strides=config.conv_stride[layer_id],
            use_bias=config.conv_bias,
            name="conv",
        )
        # 获取激活函数对象
        self.activation = get_tf_activation(config.feat_extract_activation)
        # 创建一个组归一化层对象
        self.layer_norm = TFHubertGroupNorm(groups=self.out_conv_dim, epsilon=config.layer_norm_eps, name="layer_norm")

    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        # 执行一维卷积操作
        hidden_states = self.conv(hidden_states)
        # 执行组归一化操作
        hidden_states = self.layer_norm(hidden_states)
        # 执行激活函数操作
        hidden_states = self.activation(hidden_states)
        return hidden_states
    # 定义一个方法 `build`，用于构建神经网络层的结构
    def build(self, input_shape=None):
        # 如果已经构建过，直接返回，不重复构建
        if self.built:
            return
        # 将标记设置为已构建
        self.built = True
        
        # 如果存在 `conv` 属性，执行以下操作
        if getattr(self, "conv", None) is not None:
            # 使用 `tf.name_scope` 创建名为 `self.conv.name` 的命名空间
            with tf.name_scope(self.conv.name):
                # 使用 `self.in_conv_dim` 参数构建 `conv` 层
                self.conv.build([None, None, self.in_conv_dim])
        
        # 如果存在 `layer_norm` 属性，执行以下操作
        if getattr(self, "layer_norm", None) is not None:
            # 使用 `tf.name_scope` 创建名为 `self.layer_norm.name` 的命名空间
            with tf.name_scope(self.layer_norm.name):
                # 使用 `self.out_conv_dim` 参数构建 `layer_norm` 层
                self.layer_norm.build([None, None, self.out_conv_dim])
# 定义一个名为 TFHubertPositionalConvEmbedding 的自定义层，继承自 keras 的 Layer 类
class TFHubertPositionalConvEmbedding(keras.layers.Layer):
    # 初始化方法，接受一个 HubertConfig 对象作为参数
    def __init__(self, config: HubertConfig, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        # 创建一个 TFHubertWeightNormConv1D 类对象，用于卷积操作
        self.conv = TFHubertWeightNormConv1D(
            filters=config.hidden_size,  # 卷积输出的维度大小
            kernel_size=config.num_conv_pos_embeddings,  # 卷积核的大小
            groups=config.num_conv_pos_embedding_groups,  # 卷积操作时的组数
            explicit_padding=config.num_conv_pos_embeddings // 2,  # 明确的填充大小
            name="conv",  # 层的名称
        )
        # 创建一个 TFHubertSamePadLayer 类对象，用于进行相同的填充操作
        self.padding = TFHubertSamePadLayer(config.num_conv_pos_embeddings)
        # 获取激活函数，根据配置参数中的 feat_extract_activation 设置
        self.activation = get_tf_activation(config.feat_extract_activation)
        self.config = config  # 保存配置对象

    # 定义 call 方法，接受输入的 hidden_states 张量，返回处理后的张量
    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        hidden_states = self.conv(hidden_states)  # 进行卷积操作
        hidden_states = self.padding(hidden_states)  # 进行填充操作
        hidden_states = self.activation(hidden_states)  # 应用激活函数
        return hidden_states  # 返回处理后的张量

    # build 方法，用于构建层，根据输入形状 input_shape 构建 conv 层
    def build(self, input_shape=None):
        if self.built:  # 如果已经构建过，则直接返回
            return
        self.built = True  # 将 built 标记为 True，表示已构建
        if getattr(self, "conv", None) is not None:
            with tf.name_scope(self.conv.name):
                self.conv.build([None, None, self.config.hidden_size])
                # 使用配置中的 hidden_size 构建 conv 层的形状
    # 定义模型的构建方法，用于构建模型的层次结构
    def build(self, input_shape=None):
        # 如果模型已经构建完成，则直接返回，不进行重复构建
        if self.built:
            return
        # 将模型标记为已构建状态
        self.built = True
        # 遍历模型中的每个卷积层
        for conv_layer in self.conv_layers:
            # 使用 TensorFlow 的命名空间为当前卷积层命名
            with tf.name_scope(conv_layer.name):
                # 构建当前卷积层，input_shape=None 表示使用默认输入形状
                conv_layer.build(None)
# 定义 TFHubertFeatureExtractor 类，继承自 TFHubertFeatureEncoder 类
class TFHubertFeatureExtractor(TFHubertFeatureEncoder):
    def __init__(self, config, **kwargs):
        # 调用父类 TFHubertFeatureEncoder 的构造函数
        super().__init__(config, **kwargs)
        # 发出警告，提醒该类已被弃用，并将在 Transformers v5 版本中移除，建议使用其基类代替
        warnings.warn(
            f"The class `{self.__class__.__name__}` has been depreciated "
            "and will be removed in Transformers v5. "
            f"Use `{self.__class__.__bases__[0].__name__}` instead.",
            FutureWarning,
        )


# 定义 TFHubertFeatureProjection 类，继承自 keras.layers.Layer 类
class TFHubertFeatureProjection(keras.layers.Layer):
    def __init__(self, config: HubertConfig, **kwargs):
        # 调用父类的构造函数
        super().__init__(**kwargs)

        # 初始化层归一化模块，使用给定的 epsilon 值
        self.layer_norm = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="layer_norm")
        
        # 初始化全连接层，用于特征投影
        self.projection = keras.layers.Dense(
            units=config.hidden_size,
            kernel_initializer=get_initializer(config.initializer_range),
            bias_initializer="zeros",
            name="projection",
        )
        
        # 初始化 Dropout 层，用于在训练时进行随机失活
        self.dropout = keras.layers.Dropout(rate=config.feat_proj_dropout)
        
        # 保存配置信息
        self.config = config

    # 定义调用方法，实现特征投影过程
    def call(self, hidden_states: tf.Tensor, training: bool = False) -> tf.Tensor:
        # 应用层归一化
        hidden_states = self.layer_norm(hidden_states)
        
        # 应用特征投影
        hidden_states = self.projection(hidden_states)
        
        # 应用 Dropout
        hidden_states = self.dropout(hidden_states, training=training)
        
        return hidden_states

    # 定义构建方法，用于构建层对象
    def build(self, input_shape=None):
        # 如果已经构建过，直接返回
        if self.built:
            return
        
        # 标记为已构建
        self.built = True
        
        # 构建层归一化模块，使用输入形状和配置的最后一个维度
        if getattr(self, "layer_norm", None) is not None:
            with tf.name_scope(self.layer_norm.name):
                self.layer_norm.build([None, None, self.config.conv_dim[-1]])
        
        # 构建特征投影层，使用输入形状和配置的最后一个维度
        if getattr(self, "projection", None) is not None:
            with tf.name_scope(self.projection.name):
                self.projection.build([None, None, self.config.conv_dim[-1]])


# 从 transformers.models.bart.modeling_tf_bart.TFBartAttention 复制并改名为 TFHubertAttention
class TFHubertAttention(keras.layers.Layer):
    """Multi-headed attention from "Attention Is All You Need"""

    # 初始化多头注意力层
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim

        self.num_heads = num_heads
        self.dropout = keras.layers.Dropout(dropout)
        self.head_dim = embed_dim // num_heads
        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder

        self.k_proj = keras.layers.Dense(embed_dim, use_bias=bias, name="k_proj")
        self.q_proj = keras.layers.Dense(embed_dim, use_bias=bias, name="q_proj")
        self.v_proj = keras.layers.Dense(embed_dim, use_bias=bias, name="v_proj")
        self.out_proj = keras.layers.Dense(embed_dim, use_bias=bias, name="out_proj")


        # 初始化函数，用于初始化模型的参数和属性
        super().__init__(**kwargs)
        # 设置嵌入维度
        self.embed_dim = embed_dim

        # 设置注意力头的数量
        self.num_heads = num_heads
        # 设置 dropout 层，用于在训练过程中随机丢弃部分神经元，防止过拟合
        self.dropout = keras.layers.Dropout(dropout)
        # 计算每个注意力头的维度
        self.head_dim = embed_dim // num_heads
        # 检查 embed_dim 是否可以被 num_heads 整除
        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        # 缩放因子，用于缩放注意力分数
        self.scaling = self.head_dim**-0.5
        # 是否为解码器的标志位
        self.is_decoder = is_decoder

        # 初始化键、查询、值以及输出的投影层
        self.k_proj = keras.layers.Dense(embed_dim, use_bias=bias, name="k_proj")
        self.q_proj = keras.layers.Dense(embed_dim, use_bias=bias, name="q_proj")
        self.v_proj = keras.layers.Dense(embed_dim, use_bias=bias, name="v_proj")
        self.out_proj = keras.layers.Dense(embed_dim, use_bias=bias, name="out_proj")


    def _shape(self, tensor: tf.Tensor, seq_len: int, bsz: int):
        return tf.transpose(tf.reshape(tensor, (bsz, seq_len, self.num_heads, self.head_dim)), (0, 2, 1, 3))


        # 重新塑造张量的形状，以适应多头注意力的需求
        def _shape(self, tensor: tf.Tensor, seq_len: int, bsz: int):
            return tf.transpose(tf.reshape(tensor, (bsz, seq_len, self.num_heads, self.head_dim)), (0, 2, 1, 3))


    def call(
        self,
        hidden_states: tf.Tensor,
        key_value_states: tf.Tensor | None = None,
        past_key_value: Tuple[Tuple[tf.Tensor]] | None = None,
        attention_mask: tf.Tensor | None = None,
        layer_head_mask: tf.Tensor | None = None,
        training: Optional[bool] = False,
    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
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


        # 模型的调用方法，定义了模型的前向传播逻辑
        def call(
            self,
            hidden_states: tf.Tensor,
            key_value_states: tf.Tensor | None = None,
            past_key_value: Tuple[Tuple[tf.Tensor]] | None = None,
            attention_mask: tf.Tensor | None = None,
            layer_head_mask: tf.Tensor | None = None,
            training: Optional[bool] = False,
        # 模型的构建方法，用于构建模型的层次结构
        def build(self, input_shape=None):
            if self.built:
                return
            self.built = True
            # 构建键、查询、值以及输出的投影层
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
# 从 transformers.models.wav2vec2.modeling_tf_wav2vec2.TFWav2Vec2FeedForward 复制代码，将 Wav2Vec2 替换为 Hubert
class TFHubertFeedForward(keras.layers.Layer):
    def __init__(self, config: HubertConfig, **kwargs):
        super().__init__(**kwargs)

        # 中间层的 Dropout，使用给定的激活 dropout 率
        self.intermediate_dropout = keras.layers.Dropout(config.activation_dropout)

        # 中间层的全连接层，设置单元数、权重和偏置的初始化方式，并命名为 "intermediate_dense"
        self.intermediate_dense = keras.layers.Dense(
            units=config.intermediate_size,
            kernel_initializer=get_initializer(config.initializer_range),
            bias_initializer="zeros",
            name="intermediate_dense",
        )
        # 中间层的激活函数，根据配置选择 Tensorflow 的激活函数
        self.intermediate_act_fn = get_tf_activation(config.hidden_act)

        # 输出层的全连接层，设置单元数、权重和偏置的初始化方式，并命名为 "output_dense"
        self.output_dense = keras.layers.Dense(
            units=config.hidden_size,
            kernel_initializer=get_initializer(config.initializer_range),
            bias_initializer="zeros",
            name="output_dense",
        )
        # 输出层的 Dropout，使用给定的隐藏 dropout 率
        self.output_dropout = keras.layers.Dropout(config.hidden_dropout)
        self.config = config

    # 调用函数，实现前向传播
    def call(self, hidden_states: tf.Tensor, training: bool = False) -> tf.Tensor:
        # 中间层全连接操作
        hidden_states = self.intermediate_dense(hidden_states)
        # 中间层激活函数操作
        hidden_states = self.intermediate_act_fn(hidden_states)
        # 中间层 Dropout 操作，根据训练模式决定是否启用
        hidden_states = self.intermediate_dropout(hidden_states, training=training)

        # 输出层全连接操作
        hidden_states = self.output_dense(hidden_states)
        # 输出层 Dropout 操作，根据训练模式决定是否启用
        hidden_states = self.output_dropout(hidden_states, training=training)
        return hidden_states

    # 构建层，初始化中间层和输出层的权重和偏置
    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 如果中间层已存在，则构建中间层
        if getattr(self, "intermediate_dense", None) is not None:
            with tf.name_scope(self.intermediate_dense.name):
                self.intermediate_dense.build([None, None, self.config.hidden_size])
        # 如果输出层已存在，则构建输出层
        if getattr(self, "output_dense", None) is not None:
            with tf.name_scope(self.output_dense.name):
                self.output_dense.build([None, None, self.config.intermediate_size])


# 从 transformers.models.wav2vec2.modeling_tf_wav2vec2.TFWav2Vec2EncoderLayer 复制代码，将 Wav2Vec2 替换为 Hubert
class TFHubertEncoderLayer(keras.layers.Layer):
    def __init__(self, config: HubertConfig, **kwargs):
        super().__init__(**kwargs)

        # 使用 HubertConfig 初始化注意力机制层
        self.attention = TFHubertAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=False,
            name="attention",
        )
        # dropout 层，使用给定的隐藏 dropout 率
        self.dropout = keras.layers.Dropout(config.hidden_dropout)
        # 层归一化，设置 epsilon 值并命名为 "layer_norm"
        self.layer_norm = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="layer_norm")
        # 前馈神经网络层，使用给定的 HubertConfig 配置并命名为 "feed_forward"
        self.feed_forward = TFHubertFeedForward(config, name="feed_forward")
        # 最终层归一化，设置 epsilon 值并命名为 "final_layer_norm"
        self.final_layer_norm = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="final_layer_norm")
        self.config = config
    # 定义一个方法 `call`，用于执行 Transformer 层的前向传播
    def call(
        self,
        hidden_states: tf.Tensor,  # 输入张量 hidden_states，表示输入的隐藏状态
        attention_mask: tf.Tensor | None = None,  # 注意力掩码张量，默认为 None
        output_attentions: Optional[bool] = False,  # 是否输出注意力权重，默认为 False
        training: bool = False,  # 是否处于训练模式，默认为 False
    ) -> Tuple[tf.Tensor]:  # 返回一个元组，包含类型为 tf.Tensor 的 hidden_states

        # 复制隐藏状态作为注意力残差
        attn_residual = hidden_states
        # 调用 self.attention 对象的前向传播方法，获取更新后的 hidden_states、注意力权重 attn_weights 和一个占位符 _
        hidden_states, attn_weights, _ = self.attention(
            hidden_states, attention_mask=attention_mask, training=training
        )
        # 在训练中使用 dropout 处理 hidden_states
        hidden_states = self.dropout(hidden_states, training=training)
        # 将注意力残差与更新后的 hidden_states 相加，得到新的 hidden_states
        hidden_states = attn_residual + hidden_states

        # 使用层归一化层处理 hidden_states
        hidden_states = self.layer_norm(hidden_states)
        # 将隐藏状态输入到 feed_forward 网络中，再将结果与原始 hidden_states 相加
        hidden_states = hidden_states + self.feed_forward(hidden_states)
        # 最终再次进行层归一化处理
        hidden_states = self.final_layer_norm(hidden_states)

        # 构建输出元组，初始包含更新后的 hidden_states
        outputs = (hidden_states,)

        # 如果设置输出注意力权重，则将 attn_weights 加入到输出元组中
        if output_attentions:
            outputs += (attn_weights,)

        # 返回输出元组
        return outputs

    # 定义 build 方法，用于构建层次结构
    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回
        if self.built:
            return
        # 标记当前对象为已构建状态
        self.built = True
        
        # 如果 self.attention 存在，则构建 self.attention 层
        if getattr(self, "attention", None) is not None:
            with tf.name_scope(self.attention.name):
                self.attention.build(None)
        
        # 如果 self.layer_norm 存在，则构建 self.layer_norm 层
        if getattr(self, "layer_norm", None) is not None:
            with tf.name_scope(self.layer_norm.name):
                self.layer_norm.build([None, None, self.config.hidden_size])
        
        # 如果 self.feed_forward 存在，则构建 self.feed_forward 层
        if getattr(self, "feed_forward", None) is not None:
            with tf.name_scope(self.feed_forward.name):
                self.feed_forward.build(None)
        
        # 如果 self.final_layer_norm 存在，则构建 self.final_layer_norm 层
        if getattr(self, "final_layer_norm", None) is not None:
            with tf.name_scope(self.final_layer_norm.name):
                self.final_layer_norm.build([None, None, self.config.hidden_size])
# 从 transformers.models.wav2vec2.modeling_tf_wav2vec2.TFWav2Vec2EncoderLayerStableLayerNorm 复制过来，将 Wav2Vec2 替换为 Hubert
class TFHubertEncoderLayerStableLayerNorm(keras.layers.Layer):
    def __init__(self, config: HubertConfig, **kwargs):
        super().__init__(**kwargs)
        # 初始化自注意力层，使用 HubertConfig 中定义的参数
        self.attention = TFHubertAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=False,
            name="attention",
        )
        # 随机失活层，使用隐藏层失活率来初始化
        self.dropout = keras.layers.Dropout(config.hidden_dropout)
        # 层归一化，使用 HubertConfig 中定义的 epsilon 来初始化
        self.layer_norm = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="layer_norm")
        # 前馈网络，使用 HubertConfig 初始化
        self.feed_forward = TFHubertFeedForward(config, name="feed_forward")
        # 最终的层归一化，使用 HubertConfig 中定义的 epsilon 来初始化
        self.final_layer_norm = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="final_layer_norm")
        self.config = config

    # 定义前向传播函数，接受隐藏状态、注意力掩码等输入，并返回一个元组
    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: tf.Tensor | None = None,
        output_attentions: Optional[bool] = False,
        training: bool = False,
    ) -> Tuple[tf.Tensor]:
        # 复制注意力层之前的隐藏状态，用于残差连接
        attn_residual = hidden_states
        # 应用层归一化到隐藏状态
        hidden_states = self.layer_norm(hidden_states)
        # 调用自注意力层，得到更新的隐藏状态和注意力权重
        hidden_states, attn_weights, _ = self.attention(
            hidden_states, attention_mask=attention_mask, training=training
        )
        # 应用随机失活到更新的隐藏状态
        hidden_states = self.dropout(hidden_states, training=training)
        # 残差连接：原始隐藏状态 + 更新的隐藏状态
        hidden_states = attn_residual + hidden_states
        # 应用前馈网络和最终的层归一化到更新的隐藏状态
        hidden_states = hidden_states + self.feed_forward(self.final_layer_norm(hidden_states))

        # 构建输出元组，包含更新的隐藏状态
        outputs = (hidden_states,)

        # 如果需要输出注意力权重，将注意力权重加入输出元组
        if output_attentions:
            outputs += (attn_weights,)

        return outputs

    # 构建层，确保所有子层都被构建
    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 如果注意力层存在，则构建注意力层
        if getattr(self, "attention", None) is not None:
            with tf.name_scope(self.attention.name):
                self.attention.build(None)
        # 如果层归一化存在，则根据输入形状构建层归一化
        if getattr(self, "layer_norm", None) is not None:
            with tf.name_scope(self.layer_norm.name):
                self.layer_norm.build([None, None, self.config.hidden_size])
        # 如果前馈网络存在，则构建前馈网络
        if getattr(self, "feed_forward", None) is not None:
            with tf.name_scope(self.feed_forward.name):
                self.feed_forward.build(None)
        # 如果最终的层归一化存在，则根据输入形状构建最终的层归一化
        if getattr(self, "final_layer_norm", None) is not None:
            with tf.name_scope(self.final_layer_norm.name):
                self.final_layer_norm.build([None, None, self.config.hidden_size])


# 从 transformers.models.wav2vec2.modeling_tf_wav2vec2.TFWav2Vec2Encoder 复制过来，将 Wav2Vec2 替换为 Hubert
class TFHubertEncoder(keras.layers.Layer):
    # 初始化方法，用于创建一个 Hubert 模型实例
    def __init__(self, config: HubertConfig, **kwargs):
        # 调用父类的初始化方法
        super().__init__(**kwargs)
        # 保存传入的配置对象
        self.config = config
        # 创建位置卷积嵌入层，命名为 pos_conv_embed
        self.pos_conv_embed = TFHubertPositionalConvEmbedding(config, name="pos_conv_embed")
        # 创建 LayerNormalization 层，使用给定的 epsilon 值
        self.layer_norm = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="layer_norm")
        # 创建 Dropout 层，使用给定的 dropout 率
        self.dropout = keras.layers.Dropout(config.hidden_dropout)
        # 创建多个 HubertEncoderLayer 层，根据配置中的层数进行命名
        self.layer = [TFHubertEncoderLayer(config, name=f"layers.{i}") for i in range(config.num_hidden_layers)]

    # 模型调用方法，实现了 Hubert 模型的前向传播
    def call(
        self,
        hidden_states: tf.Tensor,  # 输入的隐藏状态张量
        attention_mask: tf.Tensor | None = None,  # 注意力遮罩张量，默认为 None
        output_attentions: Optional[bool] = False,  # 是否输出注意力权重，默认为 False
        output_hidden_states: Optional[bool] = False,  # 是否输出隐藏状态，默认为 False
        return_dict: Optional[bool] = True,  # 是否以字典形式返回输出，默认为 True
        training: Optional[bool] = False,  # 是否处于训练模式，默认为 False
    ) -> Union[TFBaseModelOutput, Tuple[tf.Tensor]]:
        # 如果要输出隐藏状态，则初始化 all_hidden_states 为空元组
        all_hidden_states = () if output_hidden_states else None
        # 如果要输出注意力权重，则初始化 all_self_attentions 为空元组
        all_self_attentions = () if output_attentions else None

        # 如果传入了 attention_mask，则将隐藏状态张量与 attention_mask 进行逐元素乘法
        if attention_mask is not None:
            hidden_states = hidden_states * tf.expand_dims(attention_mask, -1)
            # 对 attention_mask 进行扩展，用于后续处理
            attention_mask = _expand_mask(attention_mask)
        else:
            # 否则 attention_mask 为空
            attention_mask = None

        # 使用位置卷积嵌入层处理隐藏状态张量，加上位置嵌入
        position_embeddings = self.pos_conv_embed(hidden_states)
        hidden_states = hidden_states + position_embeddings
        # 对加和后的隐藏状态进行 LayerNormalization 处理
        hidden_states = self.layer_norm(hidden_states)
        # 对 LayerNormalization 后的隐藏状态应用 Dropout 处理
        hidden_states = self.dropout(hidden_states, training=training)

        # 遍历每一个 HubertEncoderLayer 层
        for i, layer_module in enumerate(self.layer):
            # 如果需要输出隐藏状态，则将当前的隐藏状态添加到 all_hidden_states 中
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 添加 LayerDrop（参见 https://arxiv.org/abs/1909.11556 ）
            dropout_probability = np.random.uniform(0, 1)
            # 如果处于训练状态并且随机数小于配置中的 layerdrop 率，则跳过当前层
            if training and (dropout_probability < self.config.layerdrop):
                continue

            # 调用当前层的 forward 方法，得到输出
            layer_outputs = layer_module(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                training=training,
            )
            # 更新隐藏状态为当前层的输出
            hidden_states = layer_outputs[0]

            # 如果需要输出注意力权重，则将当前层的注意力权重输出添加到 all_self_attentions 中
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        # 添加最后一层的隐藏状态到 all_hidden_states 中
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 如果不需要以字典形式返回结果，则按顺序返回隐藏状态、隐藏状态序列、注意力权重序列中的非空元素
        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        # 否则，返回 TFBaseModelOutput 对象，包括最后的隐藏状态、隐藏状态序列和注意力权重序列
        return TFBaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )
    # 构建模型的方法，用于定义模型的输入形状和层次结构
    def build(self, input_shape=None):
        # 如果模型已经构建，则直接返回，避免重复构建
        if self.built:
            return
        # 将模型标记为已构建状态
        self.built = True
        
        # 如果存在位置卷积嵌入层，构建该层
        if getattr(self, "pos_conv_embed", None) is not None:
            # 使用位置卷积嵌入层的名称作为命名空间
            with tf.name_scope(self.pos_conv_embed.name):
                # 调用位置卷积嵌入层的构建方法，传入None作为输入形状
                self.pos_conv_embed.build(None)
        
        # 如果存在层归一化层，构建该层
        if getattr(self, "layer_norm", None) is not None:
            # 使用层归一化层的名称作为命名空间
            with tf.name_scope(self.layer_norm.name):
                # 调用层归一化层的构建方法，传入形状为 [None, None, self.config.hidden_size]
                self.layer_norm.build([None, None, self.config.hidden_size])
        
        # 如果存在多个层，依次构建每一层
        if getattr(self, "layer", None) is not None:
            for layer in self.layer:
                # 使用当前层的名称作为命名空间
                with tf.name_scope(layer.name):
                    # 调用当前层的构建方法，传入None作为输入形状
                    layer.build(None)
# 从 transformers.models.wav2vec2.modeling_tf_wav2vec2.TFWav2Vec2EncoderStableLayerNorm 复制代码，并将 Wav2Vec2 改为 Hubert
class TFHubertEncoderStableLayerNorm(keras.layers.Layer):
    # 初始化函数，接收 HubertConfig 类型的 config 参数，并调用父类的初始化方法
    def __init__(self, config: HubertConfig, **kwargs):
        super().__init__(**kwargs)
        # 将传入的 config 参数保存为对象的属性
        self.config = config
        # 创建 TFHubertPositionalConvEmbedding 对象，命名为 pos_conv_embed
        self.pos_conv_embed = TFHubertPositionalConvEmbedding(config, name="pos_conv_embed")
        # 创建 LayerNormalization 层，epsilon 参数使用 config 中的 layer_norm_eps，命名为 layer_norm
        self.layer_norm = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="layer_norm")
        # 创建 Dropout 层，dropout 率使用 config 中的 hidden_dropout
        self.dropout = keras.layers.Dropout(config.hidden_dropout)
        # 创建 TFHubertEncoderLayerStableLayerNorm 层列表，命名为 layers，根据 config.num_hidden_layers 数量生成多个层对象
        self.layer = [
            TFHubertEncoderLayerStableLayerNorm(config, name=f"layers.{i}") for i in range(config.num_hidden_layers)
        ]

    # 定义 call 方法，接收多个参数，返回 TFBaseModelOutput 或 Tuple[tf.Tensor] 类型
    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: tf.Tensor | None = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
        training: Optional[bool] = False,
    ) -> Union[TFBaseModelOutput, Tuple[tf.Tensor]]:
        # 初始化 all_hidden_states 和 all_self_attentions 变量，根据输出标志确定是否初始化空元组
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        # 如果 attention_mask 不为 None，则将 hidden_states 加上 attention_mask 的扩展维度乘积
        if attention_mask is not None:
            hidden_states = hidden_states * tf.expand_dims(attention_mask, -1)
            # 调用 _expand_mask 函数扩展 attention_mask
            attention_mask = _expand_mask(attention_mask)
        else:
            attention_mask = None

        # 计算位置编码并将其加到 hidden_states 上
        position_embeddings = self.pos_conv_embed(hidden_states)
        hidden_states = hidden_states + position_embeddings
        # 使用 dropout 对 hidden_states 进行处理，根据 training 参数确定是否启用训练模式
        hidden_states = self.dropout(hidden_states, training=training)

        # 遍历 self.layer 中的每个层对象
        for i, layer_module in enumerate(self.layer):
            # 如果输出隐藏状态，则将当前 hidden_states 加入 all_hidden_states
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 添加 LayerDrop 功能，根据论文中描述的概率决定是否跳过当前层
            dropout_probability = np.random.uniform(0, 1)
            if training and (dropout_probability < self.config.layerdrop):  # 如果处于训练状态且概率小于 layerdrop 参数，则跳过该层
                continue

            # 调用当前层对象的 call 方法，处理 hidden_states 和 attention_mask 等参数
            layer_outputs = layer_module(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                training=training,
            )
            # 更新 hidden_states 为当前层的输出的第一个元素
            hidden_states = layer_outputs[0]

            # 如果输出注意力权重，则将当前层的注意力权重加入 all_self_attentions
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        # 对最终的 hidden_states 应用 layer_norm
        hidden_states = self.layer_norm(hidden_states)

        # 如果输出隐藏状态，则将最终的 hidden_states 加入 all_hidden_states
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 如果 return_dict 参数为 False，则返回非空值的元组
        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        # 否则，返回 TFBaseModelOutput 对象，包含最终的隐藏状态、所有隐藏状态和注意力权重
        return TFBaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )
    # 如果已经构建过模型，则直接返回，避免重复构建
    if self.built:
        return

    # 将模型标记为已构建状态
    self.built = True

    # 如果存在位置编码的卷积嵌入层，则构建该层
    if getattr(self, "pos_conv_embed", None) is not None:
        with tf.name_scope(self.pos_conv_embed.name):
            self.pos_conv_embed.build(None)

    # 如果存在 Layer Normalization 层，则构建该层
    if getattr(self, "layer_norm", None) is not None:
        with tf.name_scope(self.layer_norm.name):
            # 构建 Layer Normalization 层，指定输入形状为 [None, None, self.config.hidden_size]
            self.layer_norm.build([None, None, self.config.hidden_size])

    # 如果存在多个层，则逐个构建这些层
    if getattr(self, "layer", None) is not None:
        for layer in self.layer:
            with tf.name_scope(layer.name):
                # 构建当前层，输入形状为 None，表示不限定输入维度
                layer.build(None)
@keras_serializable
class TFHubertMainLayer(keras.layers.Layer):
    # 设置配置类
    config_class = HubertConfig

    def __init__(self, config: HubertConfig, **kwargs):
        # 调用父类初始化方法
        super().__init__(**kwargs)
        # 设置配置属性
        self.config = config
        # 创建特征提取器对象
        self.feature_extractor = TFHubertFeatureEncoder(config, name="feature_extractor")
        # 创建特征投影对象
        self.feature_projection = TFHubertFeatureProjection(config, name="feature_projection")

        # 根据配置选择稳定层归一化编码器或一般编码器
        if config.do_stable_layer_norm:
            self.encoder = TFHubertEncoderStableLayerNorm(config, name="encoder")
        else:
            self.encoder = TFHubertEncoder(config, name="encoder")

    def build(self, input_shape=None):
        # 添加权重，用于掩码特定嵌入
        self.masked_spec_embed = self.add_weight(
            shape=(self.config.hidden_size,), initializer="uniform", trainable=True, name="masked_spec_embed"
        )

        # 如果已经建立过，直接返回
        if self.built:
            return
        self.built = True

        # 如果存在特征提取器，构建其结构
        if getattr(self, "feature_extractor", None) is not None:
            with tf.name_scope(self.feature_extractor.name):
                self.feature_extractor.build(None)
        
        # 如果存在特征投影器，构建其结构
        if getattr(self, "feature_projection", None) is not None:
            with tf.name_scope(self.feature_projection.name):
                self.feature_projection.build(None)
        
        # 如果存在编码器，构建其结构
        if getattr(self, "encoder", None) is not None:
            with tf.name_scope(self.encoder.name):
                self.encoder.build(None)

    def _get_feat_extract_output_lengths(self, input_lengths: tf.Tensor):
        """
        计算卷积层的输出长度
        """

        def _conv_out_length(input_length, kernel_size, stride):
            # 从 https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html 获取的一维卷积层输出长度公式
            return (input_length - kernel_size) // stride + 1

        # 遍历配置中的卷积核大小和步幅，计算每一层的输出长度
        for kernel_size, stride in zip(self.config.conv_kernel, self.config.conv_stride):
            input_lengths = _conv_out_length(input_lengths, kernel_size, stride)

        return input_lengths
    def _mask_hidden_states(self, hidden_states: tf.Tensor, mask_time_indices: tf.Tensor | None = None):
        """
        Masks extracted features along time axis and/or along feature axis according to
        [SpecAugment](https://arxiv.org/abs/1904.08779).
        """
        # 获取 hidden_states 的形状信息：batch_size, sequence_length, hidden_size
        batch_size, sequence_length, hidden_size = shape_list(hidden_states)

        # 检查是否禁用了 SpecAugment 的应用
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
            # 生成 mask_time_indices 并在时间轴上应用 SpecAugment
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

        # 返回经过 SpecAugment 处理后的 hidden_states
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
            # 使用特征提取器提取特征，将输入转换为浮点数类型并进行训练
            hidden_states = self.feature_extractor(tf.cast(input_values, tf.float32), training=training)

            if attention_mask is not None:
                # 根据卷积公式计算真实的输出长度
                output_lengths = self._get_feat_extract_output_lengths(tf.reduce_sum(attention_mask, -1))

                # 根据计算得到的长度创建序列掩码，最大长度为隐藏状态的长度，数据类型与隐藏状态一致
                attention_mask = tf.sequence_mask(
                    output_lengths, maxlen=shape_list(hidden_states)[1], dtype=hidden_states.dtype
                )

            # 使用特征投影器进行特征投影，同时根据是否训练状态进行操作
            hidden_states = self.feature_projection(hidden_states, training=training)

            # 获取参数中的时间索引掩码，如果处于训练状态
            mask_time_indices = kwargs.get("mask_time_indices", None)
            if training:
                # 根据时间索引掩码对隐藏状态进行掩码处理
                hidden_states = self._mask_hidden_states(hidden_states, mask_time_indices=mask_time_indices)

            # 将隐藏状态传入编码器进行编码，同时传递相关参数和是否返回字典
            encoder_outputs = self.encoder(
                hidden_states,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                training=training,
            )
            # 从编码器输出中获取最后的隐藏状态
            hidden_states = encoder_outputs[0]

            if not return_dict:
                # 如果不返回字典，则返回元组形式的隐藏状态和其他编码器输出
                return (hidden_states,) + encoder_outputs[1:]

            # 如果返回字典，则创建 TFBaseModelOutput 对象，并包含相应的属性
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

    # 指定配置类为 HubertConfig
    config_class = HubertConfig
    # 基础模型前缀为 "hubert"
    base_model_prefix = "hubert"
    # 主输入名称为 "input_values"
    main_input_name = "input_values"

    @property
    def input_signature(self):
        # 定义输入签名，指定输入参数的形状和数据类型
        return {
            "input_values": tf.TensorSpec((None, 16000), tf.float32, name="input_values"),
            "attention_mask": tf.TensorSpec((None, None), tf.int32, name="attention_mask"),
            "token_type_ids": tf.TensorSpec((None, None), tf.int32, name="token_type_ids"),
        }

    def __init__(self, config, *inputs, **kwargs):
        # 初始化方法，调用父类的初始化函数
        super().__init__(config, *inputs, **kwargs)
        # 发出警告，说明在 CPU 上不支持后向传播操作
        logger.warning(
            f"\n{self.__class__.__name__} has backpropagation operations that are NOT supported on CPU. If you wish "
            "to train/fine-tune this model, you need a GPU or a TPU"
        )



HUBERT_START_DOCSTRING = r"""

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
        config ([`HubertConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

HUBERT_INPUTS_DOCSTRING = r"""
"""


@add_start_docstrings(
    "The bare TFHubert Model transformer outputing raw hidden-states without any specific head on top.",
    HUBERT_START_DOCSTRING,
)
class TFHubertModel(TFHubertPreTrainedModel):
    def __init__(self, config: HubertConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.config = config
        # 初始化 TFHubertMainLayer 对象，用于处理 Hubert 模型的主要逻辑
        self.hubert = TFHubertMainLayer(config, name="hubert")

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
        根据给定的输入执行模型的前向传播，返回模型输出。

        Args:
            input_values (tf.Tensor): 输入张量，代表输入特征。
            attention_mask (tf.Tensor, optional): 注意力掩码张量，用于控制注意力分配。默认为 None。
            token_type_ids (tf.Tensor, optional): 标记类型 ID 张量，用于多序列输入。默认为 None。
            position_ids (tf.Tensor, optional): 位置 ID 张量，用于指示输入中每个位置的位置信息。默认为 None。
            head_mask (tf.Tensor, optional): 头部掩码张量，用于控制多头注意力中每个头的重要性。默认为 None。
            inputs_embeds (tf.Tensor, optional): 嵌入输入张量，用于直接提供输入的嵌入表示。默认为 None。
            output_attentions (bool, optional): 是否输出注意力权重。默认为 None。
            output_hidden_states (bool, optional): 是否输出隐藏状态。默认为 None。
            return_dict (bool, optional): 是否以字典形式返回结果。默认为 None。
            training (bool, optional): 是否处于训练模式。默认为 False。

        Returns:
            Union[TFBaseModelOutput, Tuple[tf.Tensor]]: 模型的输出结果，包含隐藏状态和/或注意力权重，具体取决于参数设置。

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
        ```
        """

        # 设置输出的隐藏状态、注意力权重和返回字典形式的结果
        output_hidden_states = output_hidden_states if output_hidden_states else self.config.output_hidden_states
        output_attentions = output_attentions if output_attentions else self.config.output_attentions
        return_dict = return_dict if return_dict else self.config.return_dict

        # 调用 TFHubertMainLayer 对象进行前向传播
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
    # 构建模型的方法，在此方法中进行模型的初始化和构建
    def build(self, input_shape=None):
        # 如果模型已经构建过，直接返回，避免重复构建
        if self.built:
            return
        # 将模型标记为已构建状态
        self.built = True
        
        # 检查是否存在名为"hubert"的属性，并且该属性不为None
        if getattr(self, "hubert", None) is not None:
            # 使用"hubert"属性的名称作为命名空间
            with tf.name_scope(self.hubert.name):
                # 调用"hubert"对象的build方法，传入None作为输入形状
                self.hubert.build(None)
@add_start_docstrings(
    """TFHubert Model with a `language modeling` head on top for Connectionist Temporal Classification (CTC).""",
    HUBERT_START_DOCSTRING,
)
class TFHubertForCTC(TFHubertPreTrainedModel):
    def __init__(self, config: HubertConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        # 初始化 TFHubert 主层，使用给定的配置和名称
        self.hubert = TFHubertMainLayer(config, name="hubert")
        # 添加 dropout 层，使用给定的最终 dropout 率
        self.dropout = keras.layers.Dropout(config.final_dropout)
        # 添加全连接层 lm_head，输出大小为词汇表大小
        self.lm_head = keras.layers.Dense(config.vocab_size, name="lm_head")
        # 确定输出隐藏大小，如果配置中存在 `add_adapter` 并且为真，则使用 `output_hidden_size`，否则使用 `hidden_size`
        self.output_hidden_size = (
            config.output_hidden_size if hasattr(config, "add_adapter") and config.add_adapter else config.hidden_size
        )

    def freeze_feature_extractor(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameters will
        not be updated during training.
        """
        # 发出警告，告知方法即将被弃用，建议使用 `freeze_feature_encoder` 方法
        warnings.warn(
            "The method `freeze_feature_extractor` is deprecated and will be removed in Transformers v5. "
            "Please use the equivalent `freeze_feature_encoder` method instead.",
            FutureWarning,
        )
        # 调用 `freeze_feature_encoder` 方法来冻结特征编码器的梯度计算
        self.freeze_feature_encoder()

    def freeze_feature_encoder(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        # 将特征提取器的可训练属性设置为 False，禁止在训练过程中更新其参数
        self.hubert.feature_extractor.trainable = False

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
    ):
        """
        Call method to process inputs and return outputs, adhering to Hubert model's forward function.
        """
        # 省略了具体的前向传播逻辑，由装饰器 `add_start_docstrings_to_model_forward` 和 `replace_return_docstrings` 指定
        pass

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "hubert", None) is not None:
            with tf.name_scope(self.hubert.name):
                # 构建 `hubert` 层，输入形状为 None
                self.hubert.build(None)
        if getattr(self, "lm_head", None) is not None:
            with tf.name_scope(self.lm_head.name):
                # 构建 `lm_head` 层，输入形状为 [None, None, self.output_hidden_size]
                self.lm_head.build([None, None, self.output_hidden_size])
```