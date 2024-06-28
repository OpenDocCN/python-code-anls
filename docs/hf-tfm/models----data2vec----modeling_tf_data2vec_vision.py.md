# `.\models\data2vec\modeling_tf_data2vec_vision.py`

```py
# coding=utf-8
# 版权 2022 Meta Platforms 和 The HuggingFace Inc. 团队。保留所有权利。
#
# 根据 Apache 许可证 2.0 版本（"许可证"）获得许可；
# 除非符合许可证的规定，否则不得使用此文件。
# 您可以在以下网址获取许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则本软件按"原样"分发，
# 没有任何明示或暗示的担保或条件。
# 有关特定语言的权限，请参阅许可证。
""" TF 2.0 Data2Vec Vision model."""

from __future__ import annotations

import collections.abc
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf

from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import (
    TFBaseModelOutput,
    TFBaseModelOutputWithPooling,
    TFSemanticSegmenterOutput,
    TFSequenceClassifierOutput,
)
from ...modeling_tf_utils import (
    TFModelInputType,
    TFPreTrainedModel,
    TFSequenceClassificationLoss,
    get_initializer,
    keras,
    keras_serializable,
    unpack_inputs,
)
from ...tf_utils import shape_list, stable_softmax
from ...utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from .configuration_data2vec_vision import Data2VecVisionConfig

# 获取 logger 对象
logger = logging.get_logger(__name__)

# 用于文档的通用变量
_CONFIG_FOR_DOC = "Data2VecVisionConfig"

# 用于文档的基础检查点信息
_CHECKPOINT_FOR_DOC = "facebook/data2vec-vision-base"
_EXPECTED_OUTPUT_SHAPE = [1, 197, 768]

# 图像分类模型的检查点和预期输出信息
_IMAGE_CLASS_CHECKPOINT = "facebook/data2vec-vision-base-ft1k"
_IMAGE_CLASS_EXPECTED_OUTPUT = "remote control, remote"

# Data2VecVision 模型的预训练模型存档列表
TF_DATA2VEC_VISION_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/data2vec-vision-base-ft1k",
    # 查看所有 Data2VecVision 模型：https://huggingface.co/models?filter=data2vec-vision
]

@dataclass
class TFData2VecVisionModelOutputWithPooling(TFBaseModelOutputWithPooling):
    """
    [`TFData2VecVisionModel`] 的输出类。
    """
    pass
    # 定义函数参数及其类型注解，说明函数接受的输入和返回的输出
    Args:
        last_hidden_state (`tf.Tensor` of shape `(batch_size, sequence_length, hidden_size)`):
            模型最后一层的隐藏状态序列。
        pooler_output (`tf.Tensor` of shape `(batch_size, hidden_size)`):
            如果 *config.use_mean_pooling* 设置为 True，则为所有补丁令牌的最后一层隐藏状态的平均值（不包括 *[CLS]* 令牌）。
            如果设置为 False，则返回 *[CLS]* 令牌的最终隐藏状态。
        hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            一个元组，包含模型每一层的隐藏状态的 `tf.Tensor`（包括初始嵌入输出）。
        attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            一个元组，包含注意力权重的 `tf.Tensor`（每层一个），形状为 `(batch_size, num_heads, sequence_length, sequence_length)`。
            这些权重是注意力 softmax 后的结果，用于计算自注意力头中的加权平均值。
    """

    # 初始化函数参数默认值为 None
    last_hidden_state: tf.Tensor = None
    pooler_output: tf.Tensor = None
    hidden_states: Tuple[tf.Tensor] | None = None
    attentions: Tuple[tf.Tensor] | None = None
class TFData2VecVisionDropPath(keras.layers.Layer):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    References:
        (1) github.com:rwightman/pytorch-image-models
    """

    def __init__(self, drop_path, **kwargs):
        super().__init__(**kwargs)
        self.drop_path = drop_path  # 初始化方法，设置了一个名为 drop_path 的属性

    def call(self, x, training=None):
        if training:
            keep_prob = 1 - self.drop_path  # 如果处于训练模式，计算保留概率
            shape = (tf.shape(x)[0],) + (1,) * (len(tf.shape(x)) - 1)  # 计算随机张量的形状
            random_tensor = keep_prob + tf.random.uniform(shape, 0, 1)  # 生成随机张量
            random_tensor = tf.floor(random_tensor)  # 取下界，得到二元随机张量
            return (x / keep_prob) * random_tensor  # 应用随机张量的 drop path 操作
        return x  # 如果不处于训练模式，直接返回输入张量


class TFData2VecVisionEmbeddings(keras.layers.Layer):
    """
    Construct the CLS token, position and patch embeddings. Optionally, also the mask token.
    """

    def __init__(self, config: Data2VecVisionConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config  # 初始化方法，设置了一个名为 config 的属性

        self.patch_embeddings = TFData2VecVisionPatchEmbeddings(config, name="patch_embeddings")  # 创建 TFData2VecVisionPatchEmbeddings 对象
        self.num_patches = self.patch_embeddings.num_patches  # 获取 patch embeddings 的数量
        self.config = config  # 再次设置 config 属性，这可能是重复的操作

        self.dropout = keras.layers.Dropout(config.hidden_dropout_prob)  # 创建一个 dropout 层

    def build(self, input_shape=None):
        self.cls_token = self.add_weight(
            shape=(1, 1, self.config.hidden_size),
            initializer=tf.random_normal_initializer(stddev=self.config.initializer_range),
            trainable=True,
            name="cls_token",
        )  # 构建 CLS token 的权重

        if self.config.use_mask_token:
            self.mask_token = self.add_weight(
                shape=(1, 1, self.config.hidden_size),
                initializer=tf.random_normal_initializer(stddev=self.config.initializer_range),
                trainable=True,
                name="mask_token",
            )  # 如果配置中使用 mask token，则构建 mask token 的权重
        else:
            self.mask_token = None  # 否则，设置 mask token 为 None

        if self.config.use_absolute_position_embeddings:
            self.position_embeddings = self.add_weight(
                shape=(1, self.num_patches + 1, self.config.hidden_size),
                initializer=tf.random_normal_initializer(stddev=self.config.initializer_range),
                trainable=True,
                name="position_embeddings",
            )  # 如果配置中使用绝对位置 embeddings，则构建位置 embeddings 的权重
        else:
            self.position_embeddings = None  # 否则，设置位置 embeddings 为 None

        if self.built:
            return
        self.built = True  # 标记为已构建状态
        if getattr(self, "patch_embeddings", None) is not None:
            with tf.name_scope(self.patch_embeddings.name):
                self.patch_embeddings.build(None)  # 构建 patch embeddings
    # 定义一个方法`call`，接受两个参数`pixel_values`和`bool_masked_pos`，返回一个张量
    def call(self, pixel_values: tf.Tensor, bool_masked_pos: tf.Tensor | None = None) -> tf.Tensor:
        # 使用`patch_embeddings`方法将输入的像素值转换成嵌入向量
        embeddings = self.patch_embeddings(pixel_values)
        # 获取嵌入向量的形状信息：批大小、序列长度和投影维度
        batch_size, seq_len, projection_dim = shape_list(embeddings)

        # 创建一个形状为(batch_size, 1, 1)的张量，其中每个元素都是`cls_token`
        cls_tokens = tf.tile(self.cls_token, (batch_size, 1, 1))

        # 如果`bool_masked_pos`不为`None`，则执行以下操作
        if bool_masked_pos is not None:
            # 创建一个形状与`embeddings`相同的张量，每个元素都是`self.mask_token`
            mask_tokens = tf.broadcast_to(self.mask_token, (batch_size, seq_len, projection_dim))
            # 将被掩盖的视觉标记替换为`mask_tokens`
            w = bool_masked_pos[..., None]
            # 将`w`转换为与`mask_tokens`相同的数据类型
            w = tf.cast(w, mask_tokens.dtype)
            # 由于TF不支持即时张量赋值，使用加法和乘法来实现掩盖操作
            embeddings = embeddings * (1 - w) + mask_tokens * w

        # 将`cls_tokens`和`embeddings`沿着序列长度的方向连接起来
        embeddings = tf.concat([cls_tokens, embeddings], axis=1)
        # 如果存在`position_embeddings`，将其加到`embeddings`上
        if self.position_embeddings is not None:
            embeddings = embeddings + self.position_embeddings
        # 对`embeddings`应用dropout操作
        embeddings = self.dropout(embeddings)

        # 返回处理后的`embeddings`张量作为方法的输出
        return embeddings
class TFData2VecVisionPatchEmbeddings(keras.layers.Layer):
    """
    Image to Patch Embedding.
    """

    def __init__(self, config: Data2VecVisionConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config

        # 从配置中获取图像大小和补丁大小
        image_size, patch_size = config.image_size, config.patch_size
        # 获取通道数和隐藏层大小
        num_channels, hidden_size = config.num_channels, config.hidden_size

        # 将图像大小和补丁大小转换为迭代对象（如果它们不是），确保它们是元组
        image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        # 计算图像中的补丁数
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        # 计算补丁的形状
        patch_shape = (image_size[0] // patch_size[0], image_size[1] // patch_size[1])

        # 设置对象的属性
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.patch_shape = patch_shape
        self.num_channels = num_channels

        # 创建卷积层，用于将像素值投影到隐藏空间
        self.projection = keras.layers.Conv2D(
            filters=hidden_size,
            kernel_size=patch_size,
            strides=patch_size,
            padding="valid",
            data_format="channels_last",
            kernel_initializer="glorot_uniform",  # 使用glorot_uniform初始化权重，类似于torch.nn.Linear
            bias_initializer="zeros",
            name="projection",
        )

    def call(self, pixel_values: tf.Tensor, training: bool = False) -> tf.Tensor:
        # 获取像素值张量的形状信息
        batch_size, num_channels, height, width = shape_list(pixel_values)
        
        # 在动态执行模式下，验证像素值的通道数是否与配置中设置的通道数匹配
        if tf.executing_eagerly():
            if num_channels != self.num_channels:
                raise ValueError(
                    "Make sure that the channel dimension of the pixel values match with the one set in the"
                    " configuration."
                )
            # 验证输入图像的高度和宽度是否与配置中设置的图像大小匹配
            if height != self.image_size[0] or width != self.image_size[1]:
                raise ValueError(
                    f"Input image size ({height}*{width}) doesn't match model"
                    f" ({self.image_size[0]}*{self.image_size[1]})."
                )

        # 当在CPU上运行时，`keras.layers.Conv2D`不支持`NCHW`格式，所以将输入格式从`NCHW`转换为`NHWC`
        # shape = (batch_size, in_height, in_width, in_channels=num_channels)
        pixel_values = tf.transpose(pixel_values, perm=(0, 2, 3, 1))

        # 将像素值投影到隐藏空间
        projection = self.projection(pixel_values)

        # 将2D空间维度变换为单个时间维度，即将投影结果reshape成(batch_size, num_patches, -1)
        num_patches = (width // self.patch_size[1]) * (height // self.patch_size[0])

        return tf.reshape(tensor=projection, shape=(batch_size, num_patches, -1))
    # 定义一个方法 `build`，用于构建神经网络层的参数
    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回，不再重复构建
        if self.built:
            return
        # 标记为已经构建
        self.built = True
        # 如果存在投影层 `projection`，则构建该投影层
        if getattr(self, "projection", None) is not None:
            # 在 TensorFlow 中创建一个命名空间 `self.projection.name`
            with tf.name_scope(self.projection.name):
                # 构建投影层，指定输入形状为 [None, None, None, self.num_channels]
                self.projection.build([None, None, None, self.num_channels])
    class TFData2VecVisionSelfAttention(keras.layers.Layer):
        def __init__(self, config: Data2VecVisionConfig, window_size: Optional[tuple] = None, **kwargs):
            super().__init__(**kwargs)
    
            # 检查隐藏大小是否是注意力头数的整数倍
            if config.hidden_size % config.num_attention_heads != 0:
                raise ValueError(
                    f"The hidden size ({config.hidden_size}) is not a multiple of the number "
                    f"of attention heads ({config.num_attention_heads})"
                )
    
            self.num_attention_heads = config.num_attention_heads
            self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
            self.all_head_size = self.num_attention_heads * self.attention_head_size
            self.sqrt_att_head_size = math.sqrt(self.attention_head_size)
    
            # 创建用于查询的全连接层
            self.query = keras.layers.Dense(
                units=self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name="query"
            )
            # 创建用于键的全连接层，不使用偏置项
            self.key = keras.layers.Dense(
                units=self.all_head_size,
                kernel_initializer=get_initializer(config.initializer_range),
                name="key",
                use_bias=False,
            )
            # 创建用于值的全连接层
            self.value = keras.layers.Dense(
                units=self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name="value"
            )
            # Dropout 层，用于注意力概率的丢弃
            self.dropout = keras.layers.Dropout(rate=config.attention_probs_dropout_prob)
    
            # 如果给定了窗口大小，则创建相对位置偏置层
            if window_size:
                self.relative_position_bias = TFData2VecVisionRelativePositionBias(
                    config, window_size=window_size, name="relative_position_bias"
                )
            else:
                self.relative_position_bias = None
            self.config = config
    
        def transpose_for_scores(self, tensor: tf.Tensor, batch_size: int) -> tf.Tensor:
            # 将张量从 [batch_size, seq_length, all_head_size] 重塑为 [batch_size, seq_length, num_attention_heads, attention_head_size]
            tensor = tf.reshape(tensor=tensor, shape=(batch_size, -1, self.num_attention_heads, self.attention_head_size))
    
            # 将张量从 [batch_size, seq_length, num_attention_heads, attention_head_size] 转置为 [batch_size, num_attention_heads, seq_length, attention_head_size]
            return tf.transpose(tensor, perm=[0, 2, 1, 3])
    
        def call(
            self,
            hidden_states: tf.Tensor,
            head_mask: tf.Tensor,
            output_attentions: bool,
            relative_position_bias: Optional["TFData2VecVisionRelativePositionBias"] = None,
            training: bool = False,
    ) -> Tuple[tf.Tensor]:
        # 获取隐藏状态的批量大小
        batch_size = shape_list(hidden_states)[0]
        # 通过self.query对隐藏状态进行查询操作
        mixed_query_layer = self.query(inputs=hidden_states)
        # 通过self.key对隐藏状态进行键操作
        mixed_key_layer = self.key(inputs=hidden_states)
        # 通过self.value对隐藏状态进行值操作
        mixed_value_layer = self.value(inputs=hidden_states)
        # 将混合的查询层转置以便进行注意力计算
        query_layer = self.transpose_for_scores(mixed_query_layer, batch_size)
        # 将混合的键层转置以便进行注意力计算
        key_layer = self.transpose_for_scores(mixed_key_layer, batch_size)
        # 将混合的值层转置以便进行注意力计算
        value_layer = self.transpose_for_scores(mixed_value_layer, batch_size)

        # 对"查询"和"键"进行点积操作以获得原始注意力分数
        # 结果形状为(batch size, num_heads, seq_len_q, seq_len_k)
        attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
        # 对注意力分数进行除以平方根(num_heads)的缩放操作
        attention_scores = attention_scores / self.sqrt_att_head_size

        # 如果存在相对位置偏置，则添加到注意力分数中
        if self.relative_position_bias is not None:
            # 传递0.0给relative_position_bias()层，因为在这种情况下，该输入不会参与任何计算
            attention_scores = attention_scores + self.relative_position_bias(0.0)[None, ...]

        # 如果提供了共享的相对位置偏置，则添加到注意力分数中
        if relative_position_bias is not None:
            attention_scores = attention_scores + relative_position_bias

        # 将注意力分数归一化为概率
        attention_probs = stable_softmax(logits=attention_scores, axis=-1)

        # 使用dropout随机丢弃整个注意力概率矩阵中的元素，这在Transformer中是标准做法
        attention_probs = self.dropout(inputs=attention_probs, training=training)

        # 如果有头部掩码(head_mask)，则应用头部掩码到注意力概率中
        if head_mask is not None:
            attention_probs = tf.multiply(attention_probs, head_mask)

        # 计算注意力输出，将注意力概率乘以值层
        attention_output = tf.matmul(attention_probs, value_layer)
        # 调整注意力输出的维度顺序
        attention_output = tf.transpose(attention_output, perm=[0, 2, 1, 3])

        # 将注意力输出重塑为(batch_size, seq_len_q, all_head_size)
        attention_output = tf.reshape(tensor=attention_output, shape=(batch_size, -1, self.all_head_size))
        # 如果需要输出注意力矩阵，则将注意力概率包含在输出中
        outputs = (attention_output, attention_probs) if output_attentions else (attention_output,)

        return outputs
    # 构建方法用于初始化模型结构，如果已经构建过则直接返回
    def build(self, input_shape=None):
        if self.built:
            return
        # 将标志设置为已构建
        self.built = True
        # 如果存在查询（query）属性，则构建查询的神经网络层
        if getattr(self, "query", None) is not None:
            # 在 TensorFlow 中使用名称作用域来管理操作，这里是为查询层创建名称作用域
            with tf.name_scope(self.query.name):
                # 构建查询层，输入形状为 [None, None, self.config.hidden_size]
                self.query.build([None, None, self.config.hidden_size])
        # 如果存在键（key）属性，则构建键的神经网络层
        if getattr(self, "key", None) is not None:
            # 在 TensorFlow 中使用名称作用域来管理操作，这里是为键层创建名称作用域
            with tf.name_scope(self.key.name):
                # 构建键层，输入形状为 [None, None, self.config.hidden_size]
                self.key.build([None, None, self.config.hidden_size])
        # 如果存在值（value）属性，则构建值的神经网络层
        if getattr(self, "value", None) is not None:
            # 在 TensorFlow 中使用名称作用域来管理操作，这里是为值层创建名称作用域
            with tf.name_scope(self.value.name):
                # 构建值层，输入形状为 [None, None, self.config.hidden_size]
                self.value.build([None, None, self.config.hidden_size])
        # 如果存在相对位置偏置（relative_position_bias）属性，则构建该偏置
        if getattr(self, "relative_position_bias", None) is not None:
            # 在 TensorFlow 中使用名称作用域来管理操作，这里是为相对位置偏置层创建名称作用域
            with tf.name_scope(self.relative_position_bias.name):
                # 构建相对位置偏置层，输入形状为 None（形状由数据决定）
                self.relative_position_bias.build(None)
class TFData2VecVisionSelfOutput(keras.layers.Layer):
    """
    The residual connection is defined in TFData2VecVisionLayer instead of here (as is the case with other models), due
    to the layernorm applied before each block.
    """

    def __init__(self, config: Data2VecVisionConfig, **kwargs):
        super().__init__(**kwargs)

        # 初始化一个全连接层，用于变换隐藏状态到指定大小
        self.dense = keras.layers.Dense(
            units=config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )
        # 初始化一个dropout层，用于在训练时随机丢弃部分神经元，防止过拟合
        self.dropout = keras.layers.Dropout(rate=config.hidden_dropout_prob)
        self.config = config

    def call(self, hidden_states: tf.Tensor, input_tensor: tf.Tensor, gamma=None, training: bool = False) -> tf.Tensor:
        # 使用全连接层变换隐藏状态
        hidden_states = self.dense(inputs=hidden_states)
        # 在训练时应用dropout层
        hidden_states = self.dropout(inputs=hidden_states, training=training)

        return hidden_states

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 如果已经构建，直接返回
        if getattr(self, "dense", None) is not None:
            # 在名为dense的作用域内构建dense层
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.hidden_size])


class TFData2VecVisionAttention(keras.layers.Layer):
    def __init__(self, config: Data2VecVisionConfig, window_size: Optional[tuple] = None, **kwargs):
        super().__init__(**kwargs)

        # 初始化自注意力层，使用Data2VecVisionSelfAttention
        self.attention = TFData2VecVisionSelfAttention(config, window_size=window_size, name="attention")
        # 初始化输出层，使用TFData2VecVisionSelfOutput
        self.dense_output = TFData2VecVisionSelfOutput(config, name="output")

    def prune_heads(self, heads):
        # 留空，抛出未实现错误，暂不实现头部修剪功能
        raise NotImplementedError

    def call(
        self,
        input_tensor: tf.Tensor,
        head_mask: tf.Tensor,
        output_attentions: bool,
        relative_position_bias: Optional["TFData2VecVisionRelativePositionBias"] = None,
        training: bool = False,
    ) -> Tuple[tf.Tensor]:
        # 调用自注意力层处理输入张量
        self_outputs = self.attention(
            hidden_states=input_tensor,
            head_mask=head_mask,
            output_attentions=output_attentions,
            relative_position_bias=relative_position_bias,
            training=training,
        )
        # 使用输出层处理自注意力层的输出结果
        attention_output = self.dense_output(
            hidden_states=self_outputs[0], input_tensor=input_tensor, training=training
        )
        # 将处理后的结果打包成元组输出，如果需要输出注意力权重，则附加到输出元组中
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them

        return outputs

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 如果未构建，构建自注意力层
        if getattr(self, "attention", None) is not None:
            with tf.name_scope(self.attention.name):
                self.attention.build(None)
        # 如果未构建，构建输出层
        if getattr(self, "dense_output", None) is not None:
            with tf.name_scope(self.dense_output.name):
                self.dense_output.build(None)


# Copied from transformers.models.vit.modeling_tf_vit.TFViTIntermediate with ViT->Data2VecVision
class TFData2VecVisionIntermediate(keras.layers.Layer):
    # 初始化函数，用于创建一个新的Data2VecVisionConfig对象，并设置网络层的一些参数
    def __init__(self, config: Data2VecVisionConfig, **kwargs):
        # 调用父类的初始化方法，传递额外的关键字参数
        super().__init__(**kwargs)

        # 创建一个全连接层，设置单元数为config.intermediate_size，使用指定的初始化器初始化权重
        self.dense = keras.layers.Dense(
            units=config.intermediate_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )

        # 如果config.hidden_act是字符串类型，将其转换为对应的激活函数
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = get_tf_activation(config.hidden_act)
        else:
            # 否则直接使用配置中指定的激活函数
            self.intermediate_act_fn = config.hidden_act
        
        # 保存配置对象到当前实例中
        self.config = config

    # 调用函数，实现对输入隐藏状态的前向传播
    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        # 将输入隐藏状态传递给全连接层，并获取输出
        hidden_states = self.dense(inputs=hidden_states)
        # 将全连接层的输出应用中间激活函数
        hidden_states = self.intermediate_act_fn(hidden_states)

        # 返回处理后的隐藏状态作为输出
        return hidden_states

    # 构建函数，用于在第一次调用call函数时构建网络层
    def build(self, input_shape=None):
        # 如果已经构建过，直接返回
        if self.built:
            return
        
        # 设置标志位表示已经构建过网络层
        self.built = True
        
        # 如果存在全连接层dense，则开始构建dense层，指定输入形状为[None, None, self.config.hidden_size]
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.hidden_size])
class TFData2VecVisionOutput(keras.layers.Layer):
    def __init__(self, config: Data2VecVisionConfig, **kwargs):
        super().__init__(**kwargs)

        # 创建一个全连接层，用于处理隐藏状态
        self.dense = keras.layers.Dense(
            units=config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )
        # 创建一个dropout层，用于在训练时随机断开连接以防止过拟合
        self.dropout = keras.layers.Dropout(rate=config.hidden_dropout_prob)
        self.config = config

    def call(self, hidden_states: tf.Tensor, training: bool = False) -> tf.Tensor:
        # 输入隐藏状态到全连接层进行处理
        hidden_states = self.dense(inputs=hidden_states)
        # 根据训练状态应用dropout操作
        hidden_states = self.dropout(inputs=hidden_states, training=training)

        return hidden_states

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                # 根据配置构建全连接层，输入形状为[None, None, self.config.intermediate_size]
                self.dense.build([None, None, self.config.intermediate_size])


class TFData2VecVisionLayer(keras.layers.Layer):
    """This corresponds to the Block class in the timm implementation."""

    def __init__(
        self, config: Data2VecVisionConfig, window_size: Optional[tuple] = None, drop_path_rate: float = 0.0, **kwargs
    ):
        super().__init__(**kwargs)
        self.config = config

        # 创建注意力层对象，用于处理视觉特征
        self.attention = TFData2VecVisionAttention(config, window_size=window_size, name="attention")
        # 创建中间层对象，用于进一步处理注意力层的输出
        self.intermediate = TFData2VecVisionIntermediate(config, name="intermediate")
        # 创建数据2向量输出层对象，用于最终的输出
        self.data2vec_output = TFData2VecVisionOutput(config, name="output")

        # 创建LayerNormalization层，用于在每个子层之前应用
        self.layernorm_before = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="layernorm_before")
        # 创建LayerNormalization层，用于在每个子层之后应用
        self.layernorm_after = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="layernorm_after")
        # 根据drop_path_rate值选择不同的激活层对象
        self.drop_path = (
            TFData2VecVisionDropPath(drop_path_rate, name="drop_path")
            if drop_path_rate > 0.0
            else keras.layers.Activation("linear", name="drop_path")
        )
        # 初始化层标度的初始值
        self.init_values = config.layer_scale_init_value
    # 定义模型的 build 方法，用于构建模型结构
    def build(self, input_shape: tf.TensorShape = None):
        # 如果指定了初始化值，则创建 lambda_1 和 lambda_2 权重，并赋予初始值
        if self.init_values > 0:
            # 创建 lambda_1 权重，形状为隐藏层大小，初始化为全1，可训练
            self.lambda_1 = self.add_weight(
                shape=(self.config.hidden_size),
                initializer="ones",
                trainable=True,
                name="lambda_1",
            )
            # 创建 lambda_2 权重，形状为隐藏层大小，初始化为全1，可训练
            self.lambda_2 = self.add_weight(
                shape=(self.config.hidden_size),
                initializer="ones",
                trainable=True,
                name="lambda_2",
            )
            # 使用初始化值乘以全1向量，赋值给 lambda_1 和 lambda_2
            self.lambda_1.assign(self.init_values * tf.ones((self.config.hidden_size)))
            self.lambda_2.assign(self.init_values * tf.ones((self.config.hidden_size)))
        else:
            # 如果没有指定初始化值，则 lambda_1 和 lambda_2 设为 None
            self.lambda_1, self.lambda_2 = None, None

        # 如果模型已经构建完成，则直接返回
        if self.built:
            return
        self.built = True
        
        # 如果定义了 attention 属性，则构建 attention 模块
        if getattr(self, "attention", None) is not None:
            with tf.name_scope(self.attention.name):
                self.attention.build(None)
        
        # 如果定义了 intermediate 属性，则构建 intermediate 模块
        if getattr(self, "intermediate", None) is not None:
            with tf.name_scope(self.intermediate.name):
                self.intermediate.build(None)
        
        # 如果定义了 data2vec_output 属性，则构建 data2vec_output 模块
        if getattr(self, "data2vec_output", None) is not None:
            with tf.name_scope(self.data2vec_output.name):
                self.data2vec_output.build(None)
        
        # 如果定义了 layernorm_before 属性，则构建 layernorm_before 模块
        if getattr(self, "layernorm_before", None) is not None:
            with tf.name_scope(self.layernorm_before.name):
                self.layernorm_before.build([None, None, self.config.hidden_size])
        
        # 如果定义了 layernorm_after 属性，则构建 layernorm_after 模块
        if getattr(self, "layernorm_after", None) is not None:
            with tf.name_scope(self.layernorm_after.name):
                self.layernorm_after.build([None, None, self.config.hidden_size])
        
        # 如果定义了 drop_path 属性，则构建 drop_path 模块
        if getattr(self, "drop_path", None) is not None:
            with tf.name_scope(self.drop_path.name):
                self.drop_path.build(None)

    # 定义模型的 call 方法，用于执行前向传播
    def call(
        self,
        hidden_states: tf.Tensor,
        head_mask: tf.Tensor,
        output_attentions: bool,
        relative_position_bias: Optional["TFData2VecVisionRelativePositionBias"] = None,
        training: bool = False,
    ) -> Tuple[tf.Tensor]:  # 定义函数返回类型为包含一个 TensorFlow 张量的元组
        self_attention_outputs = self.attention(
            # 在 Data2VecVision 中，在自注意力之前应用层归一化
            input_tensor=self.layernorm_before(inputs=hidden_states),  # 对输入张量进行层归一化处理
            head_mask=head_mask,  # 头部遮罩，用于指定屏蔽哪些注意力头
            output_attentions=output_attentions,  # 是否输出注意力权重
            relative_position_bias=relative_position_bias,  # 相对位置偏置，用于自注意力中的位置编码
            training=training,  # 是否在训练模式下
        )
        attention_output = self_attention_outputs[0]  # 获取自注意力模块的输出张量
        outputs = self_attention_outputs[1:]  # 如果需要输出注意力权重，则包含在输出中

        # 如果存在 lambda_1，则应用到注意力输出上
        if self.lambda_1 is not None:
            attention_output = self.lambda_1 * attention_output

        # 第一个残差连接
        hidden_states = self.drop_path(attention_output) + hidden_states  # 使用 drop_path 函数进行残差连接

        # 在 Data2VecVision 中，还会在自注意力之后应用层归一化
        layer_output = self.layernorm_after(hidden_states)  # 对残差连接后的张量进行层归一化处理

        layer_output = self.intermediate(layer_output)  # 中间层转换
        layer_output = self.data2vec_output(layer_output)  # Data2Vec 输出层

        # 如果存在 lambda_2，则应用到最终输出层上
        if self.lambda_2 is not None:
            layer_output = self.lambda_2 * layer_output

        # 第二个残差连接
        layer_output = self.drop_path(layer_output) + hidden_states  # 使用 drop_path 函数进行残差连接到原始隐藏状态上

        outputs = (layer_output,) + outputs  # 构建最终的输出元组

        return outputs  # 返回输出元组
# Taken and modified from here:
# https://github.com/leondgarse/keras_cv_attention_models/blob/main/keras_cv_attention_models/beit/beit.py#L28
# 定义一个自定义的Keras层，用于处理数据2Vec视觉任务的相对位置偏置
class TFData2VecVisionRelativePositionBias(keras.layers.Layer):
    def __init__(self, config: Data2VecVisionConfig, window_size: tuple, **kwargs) -> None:
        super().__init__(**kwargs)
        self.config = config

        self.window_size = window_size
        # 计算相对距离的数量，加上3用于处理特殊标记（cls_token_pos_len）
        # window_size可以是类似于(14, 14)的元组
        self.num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3

        # 创建相对位置索引
        self.relative_position_index = self.get_position_index()

    def build(self, input_shape):
        # 添加一个可训练的权重矩阵，用于存储相对位置偏置表
        self.relative_position_bias_table = self.add_weight(
            shape=(self.num_relative_distance, self.config.num_attention_heads),
            initializer="zeros",
            trainable=True,
            name="relative_position_bias_table",
        )  # [2*Wh-1 * 2*Ww-1, nH]
        # cls to token & token 2 cls & cls to cls

        super().build(input_shape)

    def get_position_index(self):
        # 获取窗口内每个标记的成对相对位置索引
        xx, yy = tf.meshgrid(range(self.window_size[0]), range(self.window_size[1]))
        coords = tf.stack([yy, xx], axis=0)  # [2, Wh, Ww]
        coords_flatten = tf.reshape(coords, [2, -1])  # [2, Wh*Ww]

        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # [2, Wh*Ww, Wh*Ww]
        relative_coords = tf.transpose(relative_coords, perm=[1, 2, 0])  # [Wh*Ww, Wh*Ww, 2]

        xx = (relative_coords[:, :, 0] + self.window_size[0] - 1) * (2 * self.window_size[1] - 1)
        yy = relative_coords[:, :, 1] + self.window_size[1] - 1
        relative_coords = tf.stack([xx, yy], axis=-1)

        relative_position_index = tf.reduce_sum(relative_coords, axis=-1)  # [Wh*Ww, Wh*Ww]

        # 添加特殊标记，表示cls到token、token到cls、cls到cls的相对位置
        top = tf.ones((1, relative_position_index.shape[1]), dtype=relative_position_index.dtype) * (
            self.num_relative_distance - 3
        )
        left = tf.ones((relative_position_index.shape[0], 1), dtype=relative_position_index.dtype) * (
            self.num_relative_distance - 2
        )
        corner = tf.ones((1, 1), dtype=relative_position_index.dtype) * (self.num_relative_distance - 1)

        left_corner = tf.concat([corner, left], axis=0)
        relative_position_index = tf.concat([top, relative_position_index], axis=0)
        relative_position_index = tf.concat([left_corner, relative_position_index], axis=1)  # [Wh*Ww + 1, Wh*Ww + 1]
        return relative_position_index

    def call(self, inputs=None) -> tf.Tensor:
        # 根据相对位置索引从相对位置偏置表中获取相对位置偏置
        relative_position_bias = tf.gather(self.relative_position_bias_table, self.relative_position_index, axis=0)
        return tf.transpose(relative_position_bias, [2, 0, 1])
    def __init__(self, config: Data2VecVisionConfig, window_size: Optional[tuple] = None, **kwargs):
        super().__init__(**kwargs)
        self.config = config  # 初始化对象的配置信息

        # 根据配置决定是否创建相对位置偏置对象
        if config.use_shared_relative_position_bias:
            self.relative_position_bias = TFData2VecVisionRelativePositionBias(
                config, window_size=window_size, name="relative_position_bias"
            )
        else:
            self.relative_position_bias = None

        # 根据层的数量创建 TFData2VecVisionLayer 对象的列表
        # 每层具有不同的 drop path rate，并且根据配置选择是否使用相对位置偏置
        dpr = list(tf.linspace(0.0, config.drop_path_rate, config.num_hidden_layers))
        self.layer = [
            TFData2VecVisionLayer(
                config,
                window_size=window_size if config.use_relative_position_bias else None,
                drop_path_rate=dpr[i],
                name=f"layer_._{i}",
            )
            for i in range(config.num_hidden_layers)
        ]

    def call(
        self,
        hidden_states: tf.Tensor,
        head_mask: tf.Tensor | None = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Union[tuple, TFBaseModelOutput]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        # 遍历所有层，对每一层进行前向传播
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 获取当前层的头部掩码
            layer_head_mask = head_mask[i] if head_mask is not None else None

            # 获取相对位置偏置对象，如果没有则为 None
            relative_position_bias = (
                self.relative_position_bias(0.0) if self.relative_position_bias is not None else None
            )

            # 调用当前层的前向传播方法
            layer_outputs = layer_module(hidden_states, layer_head_mask, output_attentions, relative_position_bias)

            # 更新隐藏状态为当前层的输出
            hidden_states = layer_outputs[0]

            # 如果需要输出注意力权重，记录当前层的注意力权重输出
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        # 如果需要输出所有隐藏状态，记录最终的隐藏状态
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 根据 return_dict 决定返回值的格式
        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)

        # 返回 TFBaseModelOutput 对象，包含最终的隐藏状态、所有隐藏状态和注意力权重
        return TFBaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )
    # 定义 build 方法，用于构建神经网络层
    def build(self, input_shape=None):
        # 如果已经构建过，直接返回，避免重复构建
        if self.built:
            return
        # 设置标志位，表示已经构建
        self.built = True
        
        # 检查是否存在相对位置偏置项，如果有，则构建其相应的层
        if getattr(self, "relative_position_bias", None) is not None:
            # 使用相对位置偏置项的名称作为命名空间
            with tf.name_scope(self.relative_position_bias.name):
                # 调用该项的 build 方法构建
                self.relative_position_bias.build(None)
        
        # 检查是否存在层列表，如果有，则逐层构建
        if getattr(self, "layer", None) is not None:
            # 遍历层列表
            for layer in self.layer:
                # 使用层的名称作为命名空间
                with tf.name_scope(layer.name):
                    # 调用层的 build 方法构建
                    layer.build(None)
# 声明一个自定义层 TFData2VecVisionMainLayer，使用 keras_serializable 装饰器标记为可序列化
@keras_serializable
class TFData2VecVisionMainLayer(keras.layers.Layer):
    # 设置类属性 config_class 为 Data2VecVisionConfig，指定配置类
    config_class = Data2VecVisionConfig

    # 初始化方法，接受 Data2VecVisionConfig 实例和一个布尔型参数 add_pooling_layer
    def __init__(self, config: Data2VecVisionConfig, add_pooling_layer: bool = True, **kwargs):
        # 调用父类的初始化方法
        super().__init__(**kwargs)

        # 设置实例属性 config 为传入的 config 参数
        self.config = config
        # 设置实例属性 add_pooling_layer 为传入的 add_pooling_layer 参数
        self.add_pooling_layer = add_pooling_layer

        # 创建 TFData2VecVisionEmbeddings 实例，命名为 embeddings
        self.embeddings = TFData2VecVisionEmbeddings(config, name="embeddings")
        # 创建 TFData2VecVisionEncoder 实例，命名为 encoder，传入 window_size 参数
        self.encoder = TFData2VecVisionEncoder(
            config, window_size=self.embeddings.patch_embeddings.patch_shape, name="encoder"
        )

        # 根据配置中的 use_mean_pooling 属性选择性地初始化 layernorm 层
        self.layernorm = (
            tf.identity  # 如果 use_mean_pooling 为 True，使用 tf.identity
            if config.use_mean_pooling
            else keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="layernorm")
            # 如果 use_mean_pooling 为 False，使用 LayerNormalization，并指定 epsilon 和名字
        )

        # 如果 add_pooling_layer 为 True，则创建 TFData2VecVisionPooler 实例，命名为 pooler
        # 否则设为 None
        self.pooler = TFData2VecVisionPooler(config, name="pooler") if add_pooling_layer else None

    # 返回嵌入层的输入 embeddings.patch_embeddings
    def get_input_embeddings(self) -> keras.layers.Layer:
        return self.embeddings.patch_embeddings

    # 未实现的方法，用于剪枝模型中的注意力头
    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        raise NotImplementedError

    # 使用 unpack_inputs 装饰器处理输入参数，定义模型的调用方法
    def call(
        self,
        pixel_values: tf.Tensor | None = None,
        bool_masked_pos: tf.Tensor | None = None,
        head_mask: tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
        **kwargs
    ):
        # 这里未完整显示，继续下面的函数参数和逻辑
    ) -> Union[tuple, TFData2VecVisionModelOutputWithPooling]:
        # 设置输出注意力权重，默认为模型配置中的设定
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 设置输出隐藏状态，默认为模型配置中的设定
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 设置是否返回字典格式的输出，默认为模型配置中的设定
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            # 如果未提供像素值，抛出数值错误
            raise ValueError("You have to specify pixel_values")

        # 如果需要，准备头部掩码
        # head_mask 中 1.0 表示保留该头部
        # attention_probs 的形状为 bsz x n_heads x N x N
        # 输入的 head_mask 形状为 [num_heads] 或 [num_hidden_layers x num_heads]
        # head_mask 被转换为 [num_hidden_layers x batch x num_heads x seq_length x seq_length] 的形状
        if head_mask is not None:
            # 如果有头部掩码，则抛出未实现错误
            raise NotImplementedError
        else:
            # 否则，使用 None 初始化 head_mask 列表，长度为模型中的隐藏层数
            head_mask = [None] * self.config.num_hidden_layers

        # 使用 embeddings 方法生成嵌入输出
        embedding_output = self.embeddings(pixel_values, bool_masked_pos, training=training)

        # 使用 encoder 方法对嵌入输出进行编码
        encoder_outputs = self.encoder(
            embedding_output,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        # 获取编码器的序列输出
        sequence_output = encoder_outputs[0]
        # 应用 layernorm 层
        sequence_output = self.layernorm(sequence_output)
        # 如果存在池化层，对序列输出进行池化
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            # 如果不要求返回字典格式的输出，则返回元组形式的输出
            head_outputs = (sequence_output, pooled_output) if pooled_output is not None else (sequence_output,)
            return head_outputs + encoder_outputs[1:]

        # 如果要求返回字典格式的输出，则构建 TFData2VecVisionModelOutputWithPooling 对象返回
        return TFData2VecVisionModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

    def build(self, input_shape=None):
        # 如果已经构建过模型，则直接返回
        if self.built:
            return
        self.built = True
        # 如果存在 embeddings 属性，则构建 embeddings 层
        if getattr(self, "embeddings", None) is not None:
            with tf.name_scope(self.embeddings.name):
                self.embeddings.build(None)
        # 如果存在 encoder 属性，则构建 encoder 层
        if getattr(self, "encoder", None) is not None:
            with tf.name_scope(self.encoder.name):
                self.encoder.build(None)
        # 如果存在 layernorm 属性，则构建 layernorm 层
        if getattr(self, "layernorm", None) is not None:
            if hasattr(self.layernorm, "name"):
                with tf.name_scope(self.layernorm.name):
                    self.layernorm.build((None, self.config.hidden_size))
        # 如果存在 pooler 属性，则构建 pooler 层
        if getattr(self, "pooler", None) is not None:
            with tf.name_scope(self.pooler.name):
                self.pooler.build(None)
class TFData2VecVisionPooler(keras.layers.Layer):
    def __init__(self, config: Data2VecVisionConfig, **kwargs):
        super().__init__(**kwargs)
        # 如果配置要求使用均值池化，则初始化 LayerNormalization 层
        self.layernorm = (
            keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="layernorm")
            if config.use_mean_pooling
            else None
        )
        self.config = config

    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        if self.layernorm is not None:
            # 对补丁令牌的最终隐藏状态进行均值池化
            patch_tokens = hidden_states[:, 1:, :]
            pooled_output = self.layernorm(tf.reduce_mean(patch_tokens, axis=1))
        else:
            # 通过仅获取 [CLS] 令牌的最终隐藏状态进行池化
            pooled_output = hidden_states[:, 0]

        return pooled_output

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 如果存在 layernorm 层，构建它并指定其输入形状
        if getattr(self, "layernorm", None) is not None:
            if hasattr(self.layernorm, "name"):
                with tf.name_scope(self.layernorm.name):
                    self.layernorm.build((None, self.config.hidden_size))


class TFData2VecVisionPreTrainedModel(TFPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = Data2VecVisionConfig
    base_model_prefix = "data2vec_vision"
    main_input_name = "pixel_values"
    _keys_to_ignore_on_load_unexpected = [r"relative_position_index"]


DATA2VEC_VISION_START_DOCSTRING = r"""
    This model inherits from [`TFPreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.).

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
    - a single Tensor with `pixel_values` only and nothing else: `model(pixel_values)`
    仅传入 `pixel_values` 张量，没有其他参数，用法示例为 `model(pixel_values)`。

    - a list of varying length with one or several input Tensors IN THE ORDER given in the docstring:
    `model([pixel_values, attention_mask])` or `model([pixel_values, attention_mask, token_type_ids])`
    传入一个长度可变的列表，按照文档字符串中给定的顺序包含一个或多个输入张量，例如 `model([pixel_values, attention_mask])` 或 `model([pixel_values, attention_mask, token_type_ids])`。

    - a dictionary with one or several input Tensors associated to the input names given in the docstring:
    `model({"pixel_values": pixel_values, "token_type_ids": token_type_ids})`
    传入一个字典，其中键是文档字符串中指定的输入名称，对应的值是相应的输入张量，例如 `model({"pixel_values": pixel_values, "token_type_ids": token_type_ids})`。

    Note that when creating models and layers with
    [subclassing](https://keras.io/guides/making_new_layers_and_models_via_subclassing/) then you don't need to worry
    about any of this, as you can just pass inputs like you would to any other Python function!

    </Tip>

    Args:
        config ([`Data2VecVisionConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~TFPreTrainedModel.from_pretrained`] method to load the model weights.
"""

DATA2VEC_VISION_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`np.ndarray`, `tf.Tensor`, `List[tf.Tensor]` `Dict[str, tf.Tensor]` or `Dict[str, np.ndarray]` and each example must have the shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See
            [`BeitImageProcessor.__call__`] for details.

        head_mask (`np.ndarray` or `tf.Tensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.

        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.

        return_dict (`bool`, *optional*):
            Whether or not to return a [`~file_utils.ModelOutput`] instead of a plain tuple. This argument can be used
            in eager mode, in graph mode the value will always be set to True.

        training (`bool`, *optional*, defaults to `False``):
            Whether or not to use the model in training mode (some modules like dropout modules have different
            behaviors between training and evaluation).
"""


@add_start_docstrings(
    "The bare Data2VecVision Model transformer outputting raw hidden-states without any specific head on top.",
    DATA2VEC_VISION_START_DOCSTRING,
)
class TFData2VecVisionModel(TFData2VecVisionPreTrainedModel):
    def __init__(self, config: Data2VecVisionConfig, add_pooling_layer: bool = False, *inputs, **kwargs):
        # 调用父类构造函数初始化模型
        super().__init__(config, *inputs, **kwargs)
        # 将传入的配置参数保存在实例变量中
        self.config = config

        # 创建 Data2VecVisionMainLayer 的实例作为模型的核心层
        self.data2vec_vision = TFData2VecVisionMainLayer(
            config, add_pooling_layer=add_pooling_layer, name="data2vec_vision"
        )

    # 获取输入嵌入的方法
    def get_input_embeddings(self):
        return self.data2vec_vision.get_input_embeddings()

    # 模型的调用方法，接受多个输入参数并返回模型输出
    @unpack_inputs
    @add_start_docstrings_to_model_forward(DATA2VEC_VISION_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFData2VecVisionModelOutputWithPooling,
        config_class=_CONFIG_FOR_DOC,
        modality="vision",
        expected_output=_EXPECTED_OUTPUT_SHAPE,
    )
    def call(
        self,
        pixel_values: TFModelInputType | None = None,
        bool_masked_pos: tf.Tensor | None = None,
        head_mask: np.ndarray | tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
        # 剩余未列出的参数由装饰器处理
    ) -> Union[tuple, TFData2VecVisionModelOutputWithPooling]:
        r"""
        bool_masked_pos (`tf.Tensor` of shape `(batch_size, num_patches)`, *optional*):
            Boolean masked positions. Indicates which patches are masked (1) and which aren't (0).
        """
        # 调用 self.data2vec_vision 方法，传入参数并获取输出
        outputs = self.data2vec_vision(
            pixel_values=pixel_values,
            bool_masked_pos=bool_masked_pos,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        # 返回 self.data2vec_vision 方法的输出作为结果
        return outputs

    def build(self, input_shape=None):
        # 如果已经构建过模型，则直接返回
        if self.built:
            return
        
        # 标记模型已经构建
        self.built = True
        
        # 如果 self.data2vec_vision 方法存在
        if getattr(self, "data2vec_vision", None) is not None:
            # 在命名空间 self.data2vec_vision.name 下构建模型
            with tf.name_scope(self.data2vec_vision.name):
                self.data2vec_vision.build(None)
@add_start_docstrings(
    """
    Data2VecVision Model transformer with an image classification head on top (a linear layer`
# 添加文档字符串，以描述 Data2VecVision 模型的初始化细节
@add_start_docstrings(
    """
    Data2VecVision Model transformer with an image classification head on top (a linear layer on top of the average of
    the final hidden states of the patch tokens) e.g. for ImageNet.
    """,
    DATA2VEC_VISION_START_DOCSTRING,
)
# 定义 TFData2VecVisionForImageClassification 类，继承自 TFData2VecVisionPreTrainedModel 和 TFSequenceClassificationLoss
class TFData2VecVisionForImageClassification(TFData2VecVisionPreTrainedModel, TFSequenceClassificationLoss):
    # 初始化方法，接收配置参数 config 以及其他输入参数
    def __init__(self, config: Data2VecVisionConfig, *inputs, **kwargs):
        # 调用父类初始化方法
        super().__init__(config, *inputs, **kwargs)

        # 设置类别数量
        self.num_labels = config.num_labels
        # 初始化 Data2VecVision 主层，传入配置，添加池化层，并指定名称
        self.data2vec_vision = TFData2VecVisionMainLayer(config, add_pooling_layer=True, name="data2vec_vision")

        # 初始化分类器头部，使用 Dense 层，设置单位数为类别数量，权重初始化方法为配置中的初始化范围，命名为 "classifier"
        self.classifier = keras.layers.Dense(
            units=config.num_labels,
            kernel_initializer=get_initializer(config.initializer_range),
            name="classifier",
        )
        # 存储配置
        self.config = config

    # 解包输入参数装饰器
    @unpack_inputs
    # 添加文档字符串到模型的前向方法
    @add_start_docstrings_to_model_forward(DATA2VEC_VISION_INPUTS_DOCSTRING)
    # 添加代码示例的文档字符串，包括检查点、输出类型、配置类和预期输出
    @add_code_sample_docstrings(
        checkpoint=_IMAGE_CLASS_CHECKPOINT,
        output_type=TFSequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_IMAGE_CLASS_EXPECTED_OUTPUT,
    )
    # 定义模型的前向传播方法，接收多个输入参数
    def call(
        self,
        pixel_values: TFModelInputType | None = None,  # 输入的像素值，类型可以是 TFModelInputType 或 None
        head_mask: np.ndarray | tf.Tensor | None = None,  # 头部掩码，类型可以是 ndarray、tf.Tensor 或 None
        output_attentions: Optional[bool] = None,  # 是否输出注意力权重，默认 None
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态，默认 None
        return_dict: Optional[bool] = None,  # 是否返回字典形式的输出，默认 None
        labels: np.ndarray | tf.Tensor | None = None,  # 标签，类型可以是 ndarray、tf.Tensor 或 None
        training: Optional[bool] = False,  # 是否在训练模式下，默认 False
        r"""
        labels (`tf.Tensor` or `np.ndarray` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        # 确定是否返回字典格式的输出，如果未指定则使用配置中的默认设置
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 将像素值和其他参数传递给数据转换函数data2vec_vision，获取其输出
        outputs = self.data2vec_vision(
            pixel_values=pixel_values,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        # 根据return_dict决定选取的输出方式，获取池化后的输出或者特定位置的输出向量
        pooled_output = outputs.pooler_output if return_dict else outputs[1]

        # 将池化输出通过分类器模型进行分类得到logits
        logits = self.classifier(pooled_output)

        # 如果提供了标签，计算损失，否则损失为None
        loss = None if labels is None else self.hf_compute_loss(labels=labels, logits=logits)

        # 如果不是以字典格式返回结果，按照元组格式构建输出
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 如果需要以TFSequenceClassifierOutput对象格式返回结果，构建对象并返回
        return TFSequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def build(self, input_shape=None):
        # 如果模型已经构建过，直接返回
        if self.built:
            return
        # 标记模型为已构建状态
        self.built = True

        # 如果data2vec_vision函数存在，使用tf.name_scope构建其模型
        if getattr(self, "data2vec_vision", None) is not None:
            with tf.name_scope(self.data2vec_vision.name):
                self.data2vec_vision.build(None)

        # 如果classifier函数存在，使用tf.name_scope构建其模型
        if getattr(self, "classifier", None) is not None:
            with tf.name_scope(self.classifier.name):
                self.classifier.build([None, None, self.config.hidden_size])
# 定义一个自定义的 Keras 层，用于创建包含卷积、归一化和激活层的卷积块。这个块简化了卷积层的使用，
# 这些卷积层通常与归一化层（如 BatchNorm）和激活层（如 ReLU）一起使用。
class TFData2VecVisionConvModule(keras.layers.Layer):
    """
    A convolutional block that bundles conv/norm/activation layers. This block simplifies the usage of convolution
    layers, which are commonly used with a norm layer (e.g., BatchNorm) and activation layer (e.g., ReLU).

    Based on OpenMMLab's implementation, found in https://github.com/open-mmlab/mmsegmentation.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        padding: str = "valid",
        bias: bool = False,
        dilation: Union[int, Tuple[int, int]] = 1,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        # 创建一个二维卷积层
        self.conv = keras.layers.Conv2D(
            filters=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            use_bias=bias,
            dilation_rate=dilation,
            name="conv",
        )
        # 创建一个批归一化层
        self.bn = keras.layers.BatchNormalization(name="bn", momentum=0.9, epsilon=1e-5)
        # 设置激活函数为 ReLU
        self.activation = tf.nn.relu
        self.in_channels = in_channels
        self.out_channels = out_channels

    def call(self, input: tf.Tensor) -> tf.Tensor:
        # 前向传播函数，依次对输入进行卷积、归一化和激活操作
        output = self.conv(input)
        output = self.bn(output)
        output = self.activation(output)
        return output

    def build(self, input_shape=None):
        # 在首次调用 build 方法时构建层
        if self.built:
            return
        self.built = True
        # 如果存在卷积层，根据输入形状构建卷积层
        if getattr(self, "conv", None) is not None:
            with tf.name_scope(self.conv.name):
                self.conv.build([None, None, None, self.in_channels])
        # 如果存在归一化层，根据输出通道数构建归一化层
        if getattr(self, "bn", None) is not None:
            with tf.name_scope(self.bn.name):
                self.bn.build((None, None, None, self.out_channels))


class TFAdaptiveAvgPool2D(keras.layers.Layer):
    # 定义一个自适应平均池化层，根据给定的输出维度和输入数据的顺序（NHWC 或者 NCHW）
    def __init__(self, output_dims: Tuple[int, int], input_ordering: str = "NHWC", **kwargs):
        super().__init__(**kwargs)
        self.output_dims = output_dims
        self.input_ordering = input_ordering
        # 如果输入数据顺序不是 'NCHW' 或者 'NHWC'，则抛出异常
        if input_ordering not in ("NCHW", "NHWC"):
            raise ValueError("Unrecognized input_ordering, should be 'NCHW' or 'NHWC'!")
        # 获取输入数据中高度和宽度的索引位置
        self.h_axis = input_ordering.index("H")
        self.w_axis = input_ordering.index("W")
    # 定义一个方法 `call`，接受一个 TensorFlow 张量作为输入
    def call(self, inputs: tf.Tensor):
        # 根据输入顺序确定输入的形状
        if self.input_ordering == "NHWC":
            # 如果输入顺序是 NHWC，则提取高度和宽度信息
            input_shape = inputs.shape[1:3]
        else:
            # 如果输入顺序不是 NHWC，则提取剩余维度的信息
            input_shape = inputs.shape[2:]

        # 将任务分解为每种可能的情况
        # 首先，如果输出维度为1，则直接使用 tf.reduce_mean
        if self.output_dims[0] == self.output_dims[1] == 1:
            if self.input_ordering == "NHWC":
                reduce_dims = [1, 2]
            else:
                reduce_dims = [2, 3]
            return tf.reduce_mean(inputs, axis=reduce_dims, keepdims=True)
        
        # 其次，如果在两个维度上以整数因子进行调整，则可以使用快捷方式
        elif input_shape[0] % self.output_dims[0] == 0 and input_shape[1] % self.output_dims[1] == 0:
            # 计算高度和宽度的调整因子
            h_resize = int(input_shape[0] // self.output_dims[0])
            w_resize = int(input_shape[1] // self.output_dims[1])
            # 使用 tf.nn.avg_pool2d 进行平均池化操作
            return tf.nn.avg_pool2d(
                inputs,
                ksize=(h_resize, w_resize),
                strides=(h_resize, w_resize),
                padding="VALID",
                data_format=self.input_ordering,
            )
        
        # 最后，如果不能采用快捷方式，则在每个轴上进行一维池化
        else:
            # 对于无法使用整数因子调整大小的维度，使用伪一维池化方法
            h_pooled = self.pseudo_1d_pool(inputs, h_pooling=True)
            return self.pseudo_1d_pool(h_pooled, h_pooling=False)
class TFData2VecVisionPyramidPoolingModule(keras.layers.Layer):
    """
    Pyramid Pooling Module (PPM) used in PSPNet.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module.
        channels (int): Channels after modules, before conv_seg.

    Based on OpenMMLab's implementation, found in https://github.com/open-mmlab/mmsegmentation.
    """

    def __init__(self, pool_scales: Tuple[int, ...], in_channels: int, out_channels: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.pool_scales = pool_scales  # 设置池化尺度
        self.in_channels = in_channels  # 输入通道数
        self.out_channels = out_channels  # 输出通道数

        self.layer_list = []  # 初始化层列表
        for idx, pool_scale in enumerate(pool_scales):
            pool_scale = pool_scale if isinstance(pool_scale, collections.abc.Iterable) else (pool_scale, pool_scale)
            self.layer_list.append(  # 向层列表添加池化层和卷积层组成的模块
                [
                    TFAdaptiveAvgPool2D(output_dims=pool_scale),  # 自适应平均池化层
                    TFData2VecVisionConvModule(  # 自定义的卷积模块
                        in_channels=in_channels, out_channels=self.out_channels, kernel_size=1, name=f"{idx}.1"
                    ),
                ]
            )

    def call(self, x: tf.Tensor) -> List[tf.Tensor]:
        ppm_outs = []  # 初始化池化模块输出列表
        inputs = x  # 保存输入张量

        for ppm in self.layer_list:
            for layer_module in ppm:
                ppm_out = layer_module(x)  # 对输入应用每个模块
                x = ppm_out  # 更新输入为当前模块的输出

            upsampled_ppm_out = tf.image.resize(ppm_out, size=shape_list(inputs)[1:-1], method="bilinear")  # 双线性插值上采样
            ppm_outs.append(upsampled_ppm_out)  # 将上采样后的结果添加到输出列表
        return ppm_outs  # 返回所有池化模块的输出列表

    def build(self, input_shape=None):
        for layer in self.layer_list:
            for layer_module in layer:
                with tf.name_scope(layer_module.name):
                    layer_module.build(None)  # 构建每个层模块


class TFData2VecVisionUperHead(keras.layers.Layer):
    """
    Unified Perceptual Parsing for Scene Understanding. This head is the implementation of
    [UPerNet](https://arxiv.org/abs/1807.10221).

    Based on OpenMMLab's implementation, found in https://github.com/open-mmlab/mmsegmentation.
    """
    # 初始化函数，用于初始化类的实例
    def __init__(self, config: Data2VecVisionConfig, **kwargs) -> None:
        # 调用父类的初始化方法
        super().__init__(**kwargs)

        # 设置池化尺度，例如 (1, 2, 3, 6)
        self.pool_scales = config.pool_scales
        # 设置输入通道数列表，例如 [768, 768, 768, 768]
        self.in_channels = [config.hidden_size] * 4
        # 设置通道数
        self.channels = config.hidden_size
        # 创建一个卷积层作为分类器
        self.classifier = keras.layers.Conv2D(config.num_labels, kernel_size=1, name="classifier")

        # PSP模块
        # 创建一个金字塔池化模块
        self.psp_modules = TFData2VecVisionPyramidPoolingModule(
            self.pool_scales, self.in_channels[-1], self.channels, name="psp_modules"
        )
        # 创建一个卷积模块作为瓶颈层
        self.bottleneck = TFData2VecVisionConvModule(
            self.in_channels[-1] + len(self.pool_scales) * self.channels,
            self.channels,
            kernel_size=3,
            padding="same",
            name="bottleneck",
        )

        # FPN模块
        self.lateral_convs = []
        self.fpn_convs = []
        # 遍历输入通道数列表，创建侧边卷积和FPN卷积模块
        for idx, in_channels in enumerate(self.in_channels[:-1]):  # 跳过顶层
            l_conv = TFData2VecVisionConvModule(
                in_channels, out_channels=self.channels, kernel_size=1, name=f"lateral_convs.{idx}"
            )
            fpn_conv = TFData2VecVisionConvModule(
                in_channels=self.channels,
                out_channels=self.channels,
                kernel_size=3,
                padding="same",
                name=f"fpn_convs.{idx}",
            )
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        # 创建一个FPN瓶颈层
        self.fpn_bottleneck = TFData2VecVisionConvModule(
            in_channels=len(self.in_channels) * self.channels,
            out_channels=self.channels,
            kernel_size=3,
            padding="same",
            name="fpn_bottleneck",
        )

    # PSP模块的前向传播方法
    def psp_forward(self, inputs):
        # 获取输入的最后一个元素
        x = inputs[-1]
        # 将输入的最后一层作为初始输出
        psp_outs = [x]
        # 对PSP模块进行操作，并将结果拼接在一起
        psp_outs.extend(self.psp_modules(x))
        psp_outs = tf.concat(psp_outs, axis=-1)
        # 使用瓶颈层处理PSP模块的输出
        output = self.bottleneck(psp_outs)

        return output
    def call(self, encoder_hidden_states: tf.Tensor) -> tf.Tensor:
        # 构建侧向连接
        laterals = [lateral_conv(encoder_hidden_states[i]) for i, lateral_conv in enumerate(self.lateral_convs)]

        # 将PSP模块的输出添加到侧向连接中
        laterals.append(self.psp_forward(encoder_hidden_states))

        # 构建自顶向下路径
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            # 获取前一层特征图的形状（不包括批次和通道维度）
            prev_shape = shape_list(laterals[i - 1])[1:-1]
            # 使用双线性插值将当前层特征图调整到前一层特征图的大小，并与前一层特征图相加
            laterals[i - 1] = laterals[i - 1] + tf.image.resize(laterals[i], size=prev_shape, method="bilinear")

        # 构建FPN的输出
        fpn_outs = [self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels - 1)]
        # 将PSP特征添加到FPN的输出中
        fpn_outs.append(laterals[-1])

        # 使用双线性插值将所有层的FPN输出调整为与第一层相同大小
        for i in range(used_backbone_levels - 1, 0, -1):
            fpn_outs[i] = tf.image.resize(fpn_outs[i], size=shape_list(fpn_outs[0])[1:-1], method="bilinear")
        # 将所有层的FPN输出连接在一起
        fpn_outs = tf.concat(fpn_outs, axis=-1)
        # 过FPN的瓶颈层
        output = self.fpn_bottleneck(fpn_outs)
        # 使用分类器输出最终结果
        output = self.classifier(output)

        return output

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 如果已定义分类器，则构建分类器
        if getattr(self, "classifier", None) is not None:
            with tf.name_scope(self.classifier.name):
                self.classifier.build([None, None, None, self.channels])
        # 如果已定义PSP模块，则构建PSP模块
        if getattr(self, "psp_modules", None) is not None:
            with tf.name_scope(self.psp_modules.name):
                self.psp_modules.build(None)
        # 如果已定义瓶颈层，则构建瓶颈层
        if getattr(self, "bottleneck", None) is not None:
            with tf.name_scope(self.bottleneck.name):
                self.bottleneck.build(None)
        # 如果已定义FPN瓶颈层，则构建FPN瓶颈层
        if getattr(self, "fpn_bottleneck", None) is not None:
            with tf.name_scope(self.fpn_bottleneck.name):
                self.fpn_bottleneck.build(None)
        # 遍历所有侧向卷积层，并构建它们
        for layer in self.lateral_convs:
            with tf.name_scope(layer.name):
                layer.build(None)
        # 遍历所有FPN卷积层，并构建它们
        for layer in self.fpn_convs:
            with tf.name_scope(layer.name):
                layer.build(None)
class TFData2VecVisionFCNHead(keras.layers.Layer):
    """
    Fully Convolution Networks for Semantic Segmentation. This head is implemented from
    [FCNNet](https://arxiv.org/abs/1411.4038).

    Args:
        config (Data2VecVisionConfig): Configuration.
        kernel_size (int): The kernel size for convs in the head. Default: 3.
        dilation (int): The dilation rate for convs in the head. Default: 1.

    Based on OpenMMLab's implementation, found in https://github.com/open-mmlab/mmsegmentation.
    """

    def __init__(
        self,
        config: Data2VecVisionConfig,
        in_index: int = 2,
        kernel_size: int = 3,
        dilation: Union[int, Tuple[int, int]] = 1,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        # 设置输入通道数为模型配置的隐藏层大小
        self.in_channels = config.hidden_size
        # 设置通道数为辅助通道数
        self.channels = config.auxiliary_channels
        # 设置卷积层数为辅助卷积层数
        self.num_convs = config.auxiliary_num_convs
        # 设置是否连接输入的标志为模型配置的辅助连接输入
        self.concat_input = config.auxiliary_concat_input
        # 设置输入索引为给定的索引
        self.in_index = in_index

        convs = []
        # 添加第一个卷积模块到列表中
        convs.append(
            TFData2VecVisionConvModule(
                in_channels=self.in_channels,
                out_channels=self.channels,
                kernel_size=kernel_size,
                padding="same",
                dilation=dilation,
                name="convs.0",
            )
        )
        # 循环添加剩余的卷积模块到列表中
        for i in range(self.num_convs - 1):
            convs.append(
                TFData2VecVisionConvModule(
                    in_channels=self.channels,
                    out_channels=self.channels,
                    kernel_size=kernel_size,
                    padding="same",
                    dilation=dilation,
                    name=f"conv_module_{i+2}",
                )
            )
        # 如果卷积层数为0，则设置卷积模块列表为tf.identity
        if self.num_convs == 0:
            self.convs = [tf.identity]
        else:
            self.convs = convs
        # 如果设置了连接输入，则创建连接输入的卷积模块
        if self.concat_input:
            self.conv_cat = TFData2VecVisionConvModule(
                self.in_channels + self.channels,
                out_channels=self.channels,
                kernel_size=kernel_size,
                padding="same",
                name="conv_cat",
            )

        # 设置分类器为卷积层，输出类别数为模型配置的类别数，卷积核大小为1x1
        self.classifier = keras.layers.Conv2D(config.num_labels, kernel_size=1, name="classifier")

    def call(self, encoder_hidden_states: tf.Tensor) -> tf.Tensor:
        # 从编码器隐藏状态中取出指定索引的特征映射
        hidden_states = encoder_hidden_states[self.in_index]
        output = hidden_states
        # 逐层应用卷积模块列表中的卷积操作
        for layer_module in self.convs:
            output = layer_module(output)
        # 如果设置了连接输入，则将原始输入与最终输出进行连接并应用连接输入的卷积模块
        if self.concat_input:
            output = self.conv_cat(tf.concat([hidden_states, output], axis=-1))
        # 应用分类器卷积层，最终输出预测结果
        output = self.classifier(output)
        return output
    # 如果模型已经构建完成，则直接返回，不进行重复构建
    if self.built:
        return
    
    # 将模型标记为已构建状态
    self.built = True
    
    # 如果模型中包含分类器（classifier）属性，则构建分类器模型
    if getattr(self, "classifier", None) is not None:
        # 使用分类器的名称作为命名空间，构建分类器模型
        with tf.name_scope(self.classifier.name):
            self.classifier.build([None, None, None, self.channels])
    
    # 如果模型中包含卷积层（conv_cat）属性，则构建卷积层模型
    if getattr(self, "conv_cat", None) is not None:
        # 使用卷积层的名称作为命名空间，构建卷积层模型
        with tf.name_scope(self.conv_cat.name):
            self.conv_cat.build(None)
@add_start_docstrings(
    """
    Data2VecVision Model transformer with a semantic segmentation head on top e.g. for ADE20k, CityScapes.
    """,
    DATA2VEC_VISION_START_DOCSTRING,
)
class TFData2VecVisionForSemanticSegmentation(TFData2VecVisionPreTrainedModel):
    def __init__(self, config: Data2VecVisionConfig, *inputs, **kwargs) -> None:
        super().__init__(config, *inputs, **kwargs)
        self.num_labels = config.num_labels
        self.data2vec_vision = TFData2VecVisionMainLayer(config, add_pooling_layer=False, name="data2vec_vision")

        # FPNs (Feature Pyramid Networks)
        self.fpn1 = [
            # First upsample layer of FPN
            keras.layers.Conv2DTranspose(config.hidden_size, kernel_size=2, strides=2, name="fpn1.0"),
            keras.layers.BatchNormalization(name="fpn1.1", momentum=0.9, epsilon=1e-5),
            keras.layers.Activation("gelu"),
            # Second upsample layer of FPN
            keras.layers.Conv2DTranspose(config.hidden_size, kernel_size=2, strides=2, name="fpn1.3"),
        ]

        self.fpn2 = [
            # Third upsample layer of FPN
            keras.layers.Conv2DTranspose(config.hidden_size, kernel_size=2, strides=2, name="fpn2.0"),
        ]

        # Identity function for FPN3
        self.fpn3 = tf.identity

        # Max pooling layer for FPN4
        self.fpn4 = keras.layers.MaxPool2D(pool_size=2, strides=2)

        # Semantic segmentation head(s)
        self.decode_head = TFData2VecVisionUperHead(config, name="decode_head")
        self.auxiliary_head = (
            TFData2VecVisionFCNHead(config, name="auxiliary_head") if config.use_auxiliary_head else None
        )

    def compute_loss(self, logits, auxiliary_logits, labels):
        # upsample logits to the images' original size
        if len(shape_list(labels)) > 3:
            label_interp_shape = shape_list(labels)[1:-1]
        else:
            label_interp_shape = shape_list(labels)[-2:]

        upsampled_logits = tf.image.resize(logits, size=label_interp_shape, method="bilinear")
        if auxiliary_logits is not None:
            upsampled_auxiliary_logits = tf.image.resize(auxiliary_logits, size=label_interp_shape, method="bilinear")

        # compute weighted loss
        loss_fct = keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction="none")

        # Copied from https://www.tensorflow.org/text/tutorials/transformer#loss_and_metrics.
        # Utility to mask the index to ignore during computing the loss.
        def masked_loss(real, pred):
            mask = tf.math.logical_not(tf.math.equal(real, self.config.semantic_loss_ignore_index))
            loss_ = loss_fct(real, pred)
            mask = tf.cast(mask, dtype=loss_.dtype)
            loss_ *= mask
            reduced_masked_loss = tf.reduce_sum(loss_) / tf.reduce_sum(mask)
            return tf.reshape(reduced_masked_loss, (1,))

        main_loss = masked_loss(labels, upsampled_logits)
        auxiliary_loss = masked_loss(labels, upsampled_auxiliary_logits)

        # Total loss combining main and auxiliary losses with weights
        loss = main_loss + self.config.auxiliary_loss_weight * auxiliary_loss

        return loss

    @unpack_inputs
    @add_start_docstrings_to_model_forward(DATA2VEC_VISION_INPUTS_DOCSTRING)
    # 使用装饰器替换函数返回值的文档字符串，指定输出类型为TFSemanticSegmenterOutput，并指定配置类为_CONFIG_FOR_DOC
    @replace_return_docstrings(output_type=TFSemanticSegmenterOutput, config_class=_CONFIG_FOR_DOC)
    # 定义call方法，接受多个参数作为输入
    def call(
        self,
        pixel_values: tf.Tensor | None = None,  # 表示像素值的张量，可选参数，默认为None
        head_mask: tf.Tensor | None = None,  # 表示头部遮罩的张量，可选参数，默认为None
        labels: tf.Tensor | None = None,  # 表示标签的张量，可选参数，默认为None
        output_attentions: Optional[bool] = None,  # 是否输出注意力信息的布尔值，可选参数，默认为None
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态的布尔值，可选参数，默认为None
        return_dict: Optional[bool] = None,  # 是否返回字典格式的输出，可选参数，默认为None
    ):
    
    # 构建函数，用于构建模型的结构
    def build(self, input_shape=None):
        # 如果已经构建过，直接返回
        if self.built:
            return
        # 将构建状态标记为已完成
        self.built = True
        
        # 如果存在data2vec_vision属性，则构建data2vec_vision模块
        if getattr(self, "data2vec_vision", None) is not None:
            with tf.name_scope(self.data2vec_vision.name):
                self.data2vec_vision.build(None)
        
        # 如果存在decode_head属性，则构建decode_head模块
        if getattr(self, "decode_head", None) is not None:
            with tf.name_scope(self.decode_head.name):
                self.decode_head.build(None)
        
        # 如果存在auxiliary_head属性，则构建auxiliary_head模块
        if getattr(self, "auxiliary_head", None) is not None:
            with tf.name_scope(self.auxiliary_head.name):
                self.auxiliary_head.build(None)
        
        # 如果存在fpn1属性，则分别构建fpn1的各个子模块
        if getattr(self, "fpn1", None) is not None:
            with tf.name_scope(self.fpn1[0].name):
                self.fpn1[0].build([None, None, None, self.config.hidden_size])
            with tf.name_scope(self.fpn1[1].name):
                self.fpn1[1].build((None, None, None, self.config.hidden_size))
            with tf.name_scope(self.fpn1[3].name):
                self.fpn1[3].build([None, None, None, self.config.hidden_size])
        
        # 如果存在fpn2属性，则构建fpn2的子模块
        if getattr(self, "fpn2", None) is not None:
            with tf.name_scope(self.fpn2[0].name):
                self.fpn2[0].build([None, None, None, self.config.hidden_size])
```