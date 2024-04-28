# `.\models\data2vec\modeling_tf_data2vec_vision.py`

```
# 指定编码格式为 UTF-8
# 版权声明，该模型代码的版权归 Meta Platforms 和 The HuggingFace Inc. 团队所有
#
# 根据 Apache 许可证 2.0 版本授权
# 除非符合许可证规定，否则您不得使用此文件
# 您可以在以下网址获取许可证副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则本软件基于"原样"提供，不附带任何明示或暗示的担保或条件
# 有关详细信息，请参阅许可证
""" TF 2.0 Data2Vec Vision 模型。"""

# 导入所需的库和模块
from __future__ import annotations

import collections.abc
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf

# 导入额外的模块和函数
# 这些函数和模块包括在 HuggingFace 库中，可以在模型训练和评估中使用
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

# 设置日志记录器
logger = logging.get_logger(__name__)

# 模型预训练配置的文档字符串
_CONFIG_FOR_DOC = "Data2VecVisionConfig"

# 检查点（模型权重）的文档字符串
_CHECKPOINT_FOR_DOC = "facebook/data2vec-vision-base"
# 预期输出形状的文档字符串
_EXPECTED_OUTPUT_SHAPE = [1, 197, 768]

# 图像分类模型的检查点
_IMAGE_CLASS_CHECKPOINT = "facebook/data2vec-vision-base-ft1k"
# 预期的图像分类输出
_IMAGE_CLASS_EXPECTED_OUTPUT = "remote control, remote"

# 预训练模型存档列表
TF_DATA2VEC_VISION_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/data2vec-vision-base-ft1k",
    # 在 https://huggingface.co/models?filter=data2vec-vision 查看所有 Data2VecVision 模型
]


@dataclass
class TFData2VecVisionModelOutputWithPooling(TFBaseModelOutputWithPooling):
    """
    [`TFData2VecVisionModel`] 的输出类。
```  
    Args:
        last_hidden_state (`tf.Tensor` of shape `(batch_size, sequence_length, hidden_size)`):
            最后一层模型输出的隐藏状态序列。
        pooler_output (`tf.Tensor` of shape `(batch_size, hidden_size)`):
            如果 *config.use_mean_pooling* 被设置为 True，则为补丁标记（不包括 *[CLS]* 标记）的最后一层隐藏状态的平均值。
            如果设置为 False，则返回 *[CLS]* 标记的最终隐藏状态。
        hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            形状为 `(batch_size, sequence_length, hidden_size)` 的 `tf.Tensor` 元组（当传递了 `output_hidden_states=True` 或 `config.output_hidden_states=True` 时返回）。

            每一层模型输出的隐藏状态，以及初始嵌入输出。
        attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            形状为 `(batch_size, num_heads, sequence_length, sequence_length)` 的 `tf.Tensor` 元组（当传递了 `output_attentions=True` 或 `config.output_attentions=True` 时返回）。

            在自注意力头中使用的注意力 softmax 后的注意力权重，用于计算加权平均值。
    """

    last_hidden_state: tf.Tensor = None
    pooler_output: tf.Tensor = None
    hidden_states: Tuple[tf.Tensor] | None = None
    attentions: Tuple[tf.Tensor] | None = None
class TFData2VecVisionDropPath(tf.keras.layers.Layer):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    References:
        (1) github.com:rwightman/pytorch-image-models
    """

    def __init__(self, drop_path, **kwargs):
        super().__init__(**kwargs)
        self.drop_path = drop_path  # 初始化对象时设置 drop_path 参数

    def call(self, x, training=None):
        if training:  # 如果处于训练模式
            keep_prob = 1 - self.drop_path  # 计算保留概率
            shape = (tf.shape(x)[0],) + (1,) * (len(tf.shape(x)) - 1)  # 计算随机张量的形状
            random_tensor = keep_prob + tf.random.uniform(shape, 0, 1)  # 生成随机张量
            random_tensor = tf.floor(random_tensor)  # 对随机张量进行向下取整
            return (x / keep_prob) * random_tensor  # 应用随机 dropout
        return x  # 如果不处于训练模式，则不做任何修改


class TFData2VecVisionEmbeddings(tf.keras.layers.Layer):
    """
    Construct the CLS token, position and patch embeddings. Optionally, also the mask token.

    """

    def __init__(self, config: Data2VecVisionConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config  # 初始化对象时设置 config 参数

        self.patch_embeddings = TFData2VecVisionPatchEmbeddings(config, name="patch_embeddings")  # 创建 TFData2VecVisionPatchEmbeddings 对象
        self.num_patches = self.patch_embeddings.num_patches  # 获取 patch_embeddings 对象的 patch 数量
        self.config = config  # 再次设置 config 参数，但与前面重复了

        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout_prob)  # 创建 Dropout 层

    def build(self, input_shape=None):
        self.cls_token = self.add_weight(
            shape=(1, 1, self.config.hidden_size),
            initializer=tf.random_normal_initializer(stddev=self.config.initializer_range),
            trainable=True,
            name="cls_token",
        )  # 创建 CLS token 的权重参数

        if self.config.use_mask_token:
            self.mask_token = self.add_weight(
                shape=(1, 1, self.config.hidden_size),
                initializer=tf.random_normal_initializer(stddev=self.config.initializer_range),
                trainable=True,
                name="mask_token",
            )  # 如果使用 mask token，则创建 mask token 的权重参数
        else:
            self.mask_token = None  # 否则，设置 mask token 为 None

        if self.config.use_absolute_position_embeddings:
            self.position_embeddings = self.add_weight(
                shape=(1, self.num_patches + 1, self.config.hidden_size),
                initializer=tf.random_normal_initializer(stddev=self.config.initializer_range),
                trainable=True,
                name="position_embeddings",
            )  # 如果使用绝对位置嵌入，则创建位置嵌入的权重参数
        else:
            self.position_embeddings = None  # 否则，设置位置嵌入为 None

        if self.built:
            return
        self.built = True
        if getattr(self, "patch_embeddings", None) is not None:
            with tf.name_scope(self.patch_embeddings.name):
                self.patch_embeddings.build(None)  # 构建 patch_embeddings 对象
    # 定义一个方法，用于对输入的像素数值进行处理，返回嵌入向量
    def call(self, pixel_values: tf.Tensor, bool_masked_pos: tf.Tensor | None = None) -> tf.Tensor:
        # 使用patch_embeddings方法将像素值转换为嵌入向量
        embeddings = self.patch_embeddings(pixel_values)
        # 获取嵌入向量的形状信息
        batch_size, seq_len, projection_dim = shape_list(embeddings)

        # 为每个batch复制一个特殊符号的嵌入向量
        cls_tokens = tf.tile(self.cls_token, (batch_size, 1, 1))

        # 如果有bool_masked_pos参数
        if bool_masked_pos is not None:
            # 创建一个与embeddings形状相同的mask_tokens
            mask_tokens = tf.broadcast_to(self.mask_token, (batch_size, seq_len, projection_dim))
            # 用mask_tokens替换被掩盖的可视标记
            w = bool_masked_pos[..., None]
            w = tf.cast(w, mask_tokens.dtype)
            # 由于TF不支持即时张量赋值，使用乘法和加法实现替换操作
            embeddings = embeddings * (1 - w) + mask_tokens * w

        # 在嵌入向量的第一维度上连接cls_tokens和embeddings
        embeddings = tf.concat([cls_tokens, embeddings], axis=1)
        # 如果存在位置嵌入信息，则将其加到嵌入向量中
        if self.position_embeddings is not None:
            embeddings = embeddings + self.position_embeddings
        # 对嵌入向量进行dropout操作
        embeddings = self.dropout(embeddings)

        # 返回处理后的嵌入向量
        return embeddings
class TFData2VecVisionPatchEmbeddings(tf.keras.layers.Layer):
    """
    Image to Patch Embedding.
    """

    def __init__(self, config: Data2VecVisionConfig, **kwargs):
        # 初始化函数，接受配置参数并初始化图像到补丁嵌入层
        super().__init__(**kwargs)
        # 存储配置参数
        self.config = config

        # 设置图像大小、补丁大小、通道数和隐藏层大小
        image_size, patch_size = config.image_size, config.patch_size
        num_channels, hidden_size = config.num_channels, config.hidden_size

        # 判断图像大小和补丁大小是否是可迭代对象，如果不是，转换成元组
        image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        # 计算补丁数量和形状
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        patch_shape = (image_size[0] // patch_size[0], image_size[1] // patch_size[1])
        # 存储计算结果
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.patch_shape = patch_shape
        self.num_channels = num_channels

        # 创建投影层，使用卷积操作将输入图像映射到隐藏层维度
        self.projection = tf.keras.layers.Conv2D(
            filters=hidden_size,
            kernel_size=patch_size,
            strides=patch_size,
            padding="valid",
            data_format="channels_last",
            kernel_initializer="glorot_uniform",  # 使用glorot_uniform初始化卷积核权重，与torch.nn.Linear兼容
            bias_initializer="zeros",
            name="projection",
        )

    def call(self, pixel_values: tf.Tensor, training: bool = False) -> tf.Tensor:
        # 定义 call 方法，用于正向传播计算
        batch_size, num_channels, height, width = shape_list(pixel_values)
        # 检查通道数和图像大小是否符合配置中的要求
        if tf.executing_eagerly():
            if num_channels != self.num_channels:
                raise ValueError(
                    "Make sure that the channel dimension of the pixel values match with the one set in the"
                    " configuration."
                )
            if height != self.image_size[0] or width != self.image_size[1]:
                raise ValueError(
                    f"Input image size ({height}*{width}) doesn't match model"
                    f" ({self.image_size[0]}*{self.image_size[1]})."
                )

        # 在CPU上运行时，`tf.keras.layers.Conv2D` 不支持 `NCHW` 格式，需将输入格式转换成 `NHWC`
        # shape = (batch_size, in_height, in_width, in_channels=num_channels)
        pixel_values = tf.transpose(pixel_values, perm=(0, 2, 3, 1))

        # 使用投影层进行映射操作
        projection = self.projection(pixel_values)

        # 将2D空间维度变换成一个时间维度
        # shape = (batch_size, num_patches, out_channels=embed_dim)
        num_patches = (width // self.patch_size[1]) * (height // self.patch_size[0])

        return tf.reshape(tensor=projection, shape=(batch_size, num_patches, -1))
    # 对神经网络模型进行构建，如果已构建过则直接返回
    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回
        if self.built:
            return
        # 标记为已经构建
        self.built = True
        # 如果模型中存在投影层
        if getattr(self, "projection", None) is not None:
            # 在命名域内创建投影层
            with tf.name_scope(self.projection.name):
                # 构建投影层，输入维度为 [None, None, None, self.num_channels]
                self.projection.build([None, None, None, self.num_channels])
    ```  
class TFData2VecVisionSelfAttention(tf.keras.layers.Layer):
    # 定义 TFData2VecVisionSelfAttention 类，继承自 tf.keras.layers.Layer

    def __init__(self, config: Data2VecVisionConfig, window_size: Optional[tuple] = None, **kwargs):
        # 初始化方法，接受 Data2VecVisionConfig 类型的 config 参数和可选的 window_size 参数

        super().__init__(**kwargs)
        # 调用父类的初始化方法

        if config.hidden_size % config.num_attention_heads != 0:
            # 如果 hidden_size 不能被 num_attention_heads 整除，抛出 ValueError
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number "
                f"of attention heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        # 将 config 中的 num_attention_heads 赋值给实例变量
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        # 计算 attention_head_size
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        # 计算 all_head_size
        self.sqrt_att_head_size = math.sqrt(self.attention_head_size)
        # 计算 sqrt_att_head_size

        self.query = tf.keras.layers.Dense(
            units=self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name="query"
        )
        # 创建 Dense 层，用于计算查询矩阵
        self.key = tf.keras.layers.Dense(
            units=self.all_head_size,
            kernel_initializer=get_initializer(config.initializer_range),
            name="key",
            use_bias=False,
        )
        # 创建 Dense 层，用于计算键矩阵，不使用偏置项
        self.value = tf.keras.layers.Dense(
            units=self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name="value"
        )
        # 创建 Dense 层，用于计算值矩阵
        self.dropout = tf.keras.layers.Dropout(rate=config.attention_probs_dropout_prob)
        # 创建 Dropout 层，用于计算 attention_probs_dropout_prob

        if window_size:
            # 如果存在 window_size
            self.relative_position_bias = TFData2VecVisionRelativePositionBias(
                config, window_size=window_size, name="relative_position_bias"
            )
            # 创建 TFData2VecVisionRelativePositionBias 实例，用于计算相对位置偏置
        else:
            # 如果不存在 window_size
            self.relative_position_bias = None
            # 相对位置偏置设置为 None
        self.config = config
        # 将 config 赋值给实例变量

    def transpose_for_scores(self, tensor: tf.Tensor, batch_size: int) -> tf.Tensor:
        # 定义 transpose_for_scores 方法，接受 tensor 和 batch_size 参数，并返回 tf.Tensor

        # Reshape from [batch_size, seq_length, all_head_size] to [batch_size, seq_length, num_attention_heads, attention_head_size]
        # 将 tensor 重塑为 [batch_size, seq_length, num_attention_heads, attention_head_size]
        tensor = tf.reshape(tensor=tensor, shape=(batch_size, -1, self.num_attention_heads, self.attention_head_size)

        # Transpose the tensor from [batch_size, seq_length, num_attention_heads, attention_head_size] to [batch_size, num_attention_heads, seq_length, attention_head_size]
        # 将 tensor 从 [batch_size, seq_length, num_attention_heads, attention_head_size] 转置为 [batch_size, num_attention_heads, seq_length, attention_head_size]
        return tf.transpose(tensor, perm=[0, 2, 1, 3])

    def call(
        self,
        hidden_states: tf.Tensor,
        head_mask: tf.Tensor,
        output_attentions: bool,
        relative_position_bias: Optional["TFData2VecVisionRelativePositionBias"] = None,
        training: bool = False,
    def call(self, inputs: Tuple[tf.Tensor]) -> Tuple[tf.Tensor]:
            # 获取隐藏状态的批处理大小
            batch_size = shape_list(hidden_states)[0]
            # 使用隐藏状态查询创建混合查询层
            mixed_query_layer = self.query(inputs=hidden_states)
            # 使用隐藏状态键创建混合键层
            mixed_key_layer = self.key(inputs=hidden_states)
            # 使用隐藏状态值创建混合值层
            mixed_value_layer = self.value(inputs=hidden_states)
            # 调整查询层的形状以便进行点积操作
            query_layer = self.transpose_for_scores(mixed_query_layer, batch_size)
            # 调整键层的形状以便进行点积操作
            key_layer = self.transpose_for_scores(mixed_key_layer, batch_size)
            # 调整值层的形状以便进行点积操作
            value_layer = self.transpose_for_scores(mixed_value_layer, batch_size)
    
            # 通过“查询”和“键”之间的点积获取原始注意力分数
            # (批处理大小，头数，查询序列长度，键序列长度)
            attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
            attention_scores = attention_scores / self.sqrt_att_head_size
    
            # 如果存在相对位置偏差，则添加相对位置偏差
            if self.relative_position_bias is not None:
                # 传入`0.0`到`relative_position_bias()`层，因为否则Keras可能会抱怨`Layer.call()`未正确调用
                # 在这种情况下，这个输入，即0.0不会在任何计算中使用，所以我们是安全的
                attention_scores = attention_scores + self.relative_position_bias(0.0)[None, ...]
    
            # 如果提供了共享相对位置偏差，则添加共享相对位置偏差
            if relative_position_bias is not None:
                attention_scores = attention_scores + relative_position_bias
    
            # 将注意力分数归一化为概率
            attention_probs = stable_softmax(logits=attention_scores, axis=-1)
    
            # 这实际上是进行dropout整个 token 来进行注意，可能看起来有点不寻常，但来自原始 Transformer 论文
            attention_probs = self.dropout(inputs=attention_probs, training=training)
    
            # 如果需要，对头进行蒙版
            if head_mask is not None:
                attention_probs = tf.multiply(attention_probs, head_mask)
    
            # 计算注意力输出
            attention_output = tf.matmul(attention_probs, value_layer)
            attention_output = tf.transpose(attention_output, perm=[0, 2, 1, 3])
    
            # (批处理大小，查询序列长度，全部头大小)
            attention_output = tf.reshape(tensor=attention_output, shape=(batch_size, -1, self.all_head_size))
            outputs = (attention_output, attention_probs) if output_attentions else (attention_output,)
    
            return outputs
    # 构建自定义层的方法，用于构建层的参数和内部结构
    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回，避免重复构建
        if self.built:
            return
        # 将标志位设置为已构建
        self.built = True
        # 如果存在查询向量，则构建查询向量
        if getattr(self, "query", None) is not None:
            # 使用 TensorFlow 的命名空间为查询向量构建层
            with tf.name_scope(self.query.name):
                # 构建查询向量的参数
                self.query.build([None, None, self.config.hidden_size])
        # 如果存在键向量，则构建键向量
        if getattr(self, "key", None) is not None:
            # 使用 TensorFlow 的命名空间为键向量构建层
            with tf.name_scope(self.key.name):
                # 构建键向量的参数
                self.key.build([None, None, self.config.hidden_size])
        # 如果存在值向量，则构建值向量
        if getattr(self, "value", None) is not None:
            # 使用 TensorFlow 的命名空间为值向量构建层
            with tf.name_scope(self.value.name):
                # 构建值向量的参数
                self.value.build([None, None, self.config.hidden_size])
        # 如果存在相对位置偏置，则构建相对位置偏置
        if getattr(self, "relative_position_bias", None) is not None:
            # 使用 TensorFlow 的命名空间为相对位置偏置构建层
            with tf.name_scope(self.relative_position_bias.name):
                # 构建相对位置偏置的参数
                self.relative_position_bias.build(None)
# 定义了一个名为 TFData2VecVisionSelfOutput 的自定义层
class TFData2VecVisionSelfOutput(tf.keras.layers.Layer):
    """
    The residual connection is defined in TFData2VecVisionLayer instead of here (as is the case with other models), due
    to the layernorm applied before each block.
    """

    def __init__(self, config: Data2VecVisionConfig, **kwargs):
        super().__init__(**kwargs)

        # 创建一个全连接层，用于将输入数据转换为指定维度的输出
        self.dense = tf.keras.layers.Dense(
            units=config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )
        # 创建一个dropout层，用于在训练过程中随机丢弃部分输入数据，防止过拟合
        self.dropout = tf.keras.layers.Dropout(rate=config.hidden_dropout_prob)
        self.config = config

    # 定义该自定义层的前向传播逻辑
    def call(self, hidden_states: tf.Tensor, input_tensor: tf.Tensor, gamma=None, training: bool = False) -> tf.Tensor:
        # 将输入数据经过全连接层转换
        hidden_states = self.dense(inputs=hidden_states)
        # 在训练模式下对转换后的数据进行dropout处理
        hidden_states = self.dropout(inputs=hidden_states, training=training)

        # 返回处理后的数据
        return hidden_states

    # 构建该自定义层
    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 如果已定义全连接层，调用全连接层的build方法进行构建
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.hidden_size])


# 定义了一个名为 TFData2VecVisionAttention 的自定义层
class TFData2VecVisionAttention(tf.keras.layers.Layer):
    def __init__(self, config: Data2VecVisionConfig, window_size: Optional[tuple] = None, **kwargs):
        super().__init__(**kwargs)

        # 创建一个 SelfAttention 层，用于计算注意力得分并进行加权求和
        self.attention = TFData2VecVisionSelfAttention(config, window_size=window_size, name="attention")
        # 创建一个 SelfOutput 层，用于对经过注意力计算后的结果进行处理
        self.dense_output = TFData2VecVisionSelfOutput(config, name="output")

    # 定义该自定义层的前向传播逻辑
    def call(
        self,
        input_tensor: tf.Tensor,
        head_mask: tf.Tensor,
        output_attentions: bool,
        relative_position_bias: Optional["TFData2VecVisionRelativePositionBias"] = None,
        training: bool = False,
    ) -> Tuple[tf.Tensor]:
        # 使用 SelfAttention 层对输入数据进行注意力计算
        self_outputs = self.attention(
            hidden_states=input_tensor,
            head_mask=head_mask,
            output_attentions=output_attentions,
            relative_position_bias=relative_position_bias,
            training=training,
        )
        # 对 SelfAttention 层输出的结果进行处理
        attention_output = self.dense_output(
            hidden_states=self_outputs[0], input_tensor=input_tensor, training=training
        )
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them

        # 返回处理后的结果
        return outputs

    # 构建该自定义层
    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 如果已定义 SelfAttention 层，调用 SelfAttention 层的build方法进行构建
        if getattr(self, "attention", None) is not None:
            with tf.name_scope(self.attention.name):
                self.attention.build(None)
        # 如果已定义 SelfOutput 层，调用 SelfOutput 层的build方法进行构建
        if getattr(self, "dense_output", None) is not None:
            with tf.name_scope(self.dense_output.name):
                self.dense_output.build(None)


# 从 transformers.models.vit.modeling_tf_vit.TFViTIntermediate 复制并修改为 TFData2VecVisionSelfOutput
# 定义一个 TFData2VecVisionIntermediate 类，继承自 tf.keras.layers.Layer 类
class TFData2VecVisionIntermediate(tf.keras.layers.Layer):
    # 初始化方法，接收一个 Data2VecVisionConfig 类型的参数和其他关键字参数
    def __init__(self, config: Data2VecVisionConfig, **kwargs):
        # 调用父类的初始化方法
        super().__init__(**kwargs)

        # 创建一个全连接层，设置单元数为 config.intermediate_size，权重初始化方式为 config.initializer_range
        self.dense = tf.keras.layers.Dense(
            units=config.intermediate_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )

        # 如果 config.hidden_act 是字符串类型，调用 get_tf_activation 方法得到激活函数并赋给 self.intermediate_act_fn
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = get_tf_activation(config.hidden_act)
        # 否则直接将 config.hidden_act 赋给 self.intermediate_act_fn
        else:
            self.intermediate_act_fn = config.hidden_act
        # 将 config 赋给 self.config
        self.config = config

    # 定义 call 方法，接收一个 tf.Tensor 类型的 hidden_states 参数，返回一个 tf.Tensor 类型的结果
    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        # 通过全连接层处理输入 hidden_states，得到输出 hidden_states
        hidden_states = self.dense(inputs=hidden_states)
        # 通过激活函数处理输出 hidden_states
        hidden_states = self.intermediate_act_fn(hidden_states)

        # 返回处理后的 hidden_states
        return hidden_states

    # 定义 build 方法，接收 input_shape 参数，默认为 None
    def build(self, input_shape=None):
        # 如果已经构建过，直接返回
        if self.built:
            return
        # 设置为已构建
        self.built = True
        # 如果 self.dense 存在
        if getattr(self, "dense", None) is not None:
            # 在命名空间 self.dense.name 下构建全连接层，输入形状为 [None, None, self.config.hidden_size]
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.hidden_size])


# 定义一个 TFData2VecVisionOutput 类，继承自 tf.keras.layers.Layer 类
class TFData2VecVisionOutput(tf.keras.layers.Layer):
    # 初始化方法，接收一个 Data2VecVisionConfig 类型的参数和其他关键字参数
    def __init__(self, config: Data2VecVisionConfig, **kwargs):
        # 调用父类的初始化方法
        super().__init__(**kwargs)

        # 创建一个全连接层，设置单元数为 config.hidden_size，权重初始化方式为 config.initializer_range
        self.dense = tf.keras.layers.Dense(
            units=config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )
        # 创建一个随机失活层，失活概率为 config.hidden_dropout_prob
        self.dropout = tf.keras.layers.Dropout(rate=config.hidden_dropout_prob)
        # 将 config 赋给 self.config
        self.config = config

    # 定义 call 方法，接收一个 tf.Tensor 类型的 hidden_states 参数和一个布尔类型的 training 参数，默认为 False，返回一个 tf.Tensor 类型的结果
    def call(self, hidden_states: tf.Tensor, training: bool = False) -> tf.Tensor:
        # 通过全连接层处理输入 hidden_states，得到输出 hidden_states
        hidden_states = self.dense(inputs=hidden_states)
        # 通过随机失活层处理输出 hidden_states
        hidden_states = self.dropout(inputs=hidden_states, training=training)

        # 返回处理后的 hidden_states
        return hidden_states

    # 定义 build 方法，接收 input_shape 参数，默认为 None
    def build(self, input_shape=None):
        # 如果已经构建过，直接返回
        if self.built:
            return
        # 设置为已构建
        self.built = True
        # 如果 self.dense 存在
        if getattr(self, "dense", None) is not None:
            # 在命名空间 self.dense.name 下构建全连接层，输入形状为 [None, None, self.config.intermediate_size]
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.intermediate_size])


# 定义一个 TFData2VecVisionLayer 类，继承自 tf.keras.layers.Layer 类
class TFData2VecVisionLayer(tf.keras.layers.Layer):
    """This corresponds to the Block class in the timm implementation."""
    
    # 初始化方法，接收一个 Data2VecVisionConfig 类型的参数、一个可选的元组类型的 window_size 参数，默认为 None，一个浮点类型的 drop_path_rate 参数，默认为 0.0，以及其他关键字参数
    def __init__(
        self, config: Data2VecVisionConfig, window_size: Optional[tuple] = None, drop_path_rate: float = 0.0, **kwargs
    ):
        # 调用父类的构造函数，并传递关键字参数
        super().__init__(**kwargs)
        # 将配置信息保存到实例变量中
        self.config = config

        # 创建注意力层对象
        self.attention = TFData2VecVisionAttention(config, window_size=window_size, name="attention")
        # 创建中间层对象
        self.intermediate = TFData2VecVisionIntermediate(config, name="intermediate")
        # 创建输出层对象
        self.data2vec_output = TFData2VecVisionOutput(config, name="output")

        # 创建 LayerNormalization 层对象，用于归一化前
        self.layernorm_before = tf.keras.layers.LayerNormalization(
            epsilon=config.layer_norm_eps, name="layernorm_before"
        )
        # 创建 LayerNormalization 层对象，用于归一化后
        self.layernorm_after = tf.keras.layers.LayerNormalization(
            epsilon=config.layer_norm_eps, name="layernorm_after"
        )
        # 使用 `layers.Activation` 代替 `tf.identity` 来更好地控制 `training` 行为
        # 如果 drop_path_rate 大于 0，则创建 DropPath 层对象，否则创建线性激活层对象
        self.drop_path = (
            TFData2VecVisionDropPath(drop_path_rate, name="drop_path")
            if drop_path_rate > 0.0
            else tf.keras.layers.Activation("linear", name="drop_path")
        )
        # 初始化参数值
        self.init_values = config.layer_scale_init_value

    def build(self, input_shape: tf.TensorShape = None):
        # 如果初始化参数值大于 0，则初始化 lambda_1 和 lambda_2
        if self.init_values > 0:
            self.lambda_1 = self.add_weight(
                shape=(self.config.hidden_size),
                initializer="ones",
                trainable=True,
                name="lambda_1",
            )
            self.lambda_2 = self.add_weight(
                shape=(self.config.hidden_size),
                initializer="ones",
                trainable=True,
                name="lambda_2",
            )
            self.lambda_1.assign(self.init_values * tf.ones((self.config.hidden_size)))
            self.lambda_2.assign(self.init_values * tf.ones((self.config.hidden_size)))
        else:
            self.lambda_1, self.lambda_2 = None, None

        # 如果已经构建过，则直接返回
        if self.built:
            return
        self.built = True
        # 构建各个子层
        if getattr(self, "attention", None) is not None:
            with tf.name_scope(self.attention.name):
                self.attention.build(None)
        if getattr(self, "intermediate", None) is not None:
            with tf.name_scope(self.intermediate.name):
                self.intermediate.build(None)
        if getattr(self, "data2vec_output", None) is not None:
            with tf.name_scope(self.data2vec_output.name):
                self.data2vec_output.build(None)
        if getattr(self, "layernorm_before", None) is not None:
            with tf.name_scope(self.layernorm_before.name):
                self.layernorm_before.build([None, None, self.config.hidden_size])
        if getattr(self, "layernorm_after", None) is not None:
            with tf.name_scope(self.layernorm_after.name):
                self.layernorm_after.build([None, None, self.config.hidden_size])
        if getattr(self, "drop_path", None) is not None:
            with tf.name_scope(self.drop_path.name):
                self.drop_path.build(None)
    # 定义一个方法，用于处理自注意力机制
    def call(
        self,
        hidden_states: tf.Tensor,  # 输入张量，包含了模型的隐藏状态信息
        head_mask: tf.Tensor,  # 头部遮罩张量，用于掩盖某些注意力头部
        output_attentions: bool,  # 是否输出注意力权重
        relative_position_bias: Optional["TFData2VecVisionRelativePositionBias"] = None,  # 相对位置偏置张量，用于自注意力机制
        training: bool = False,  # 是否处于训练模式
    ) -> Tuple[tf.Tensor]:  # 返回类型为包含张量的元组
        # 使用层归一化处理输入张量后，传入注意力层
        self_attention_outputs = self.attention(
            # 在 Data2VecVision 中，自注意力之前会应用层归一化
            input_tensor=self.layernorm_before(inputs=hidden_states),  # 在隐藏状态上应用层归一化
            head_mask=head_mask,  # 传入头部遮罩
            output_attentions=output_attentions,  # 是否输出注意力权重
            relative_position_bias=relative_position_bias,  # 传入相对位置偏置张量
            training=training,  # 传入训练模式标志
        )
        attention_output = self_attention_outputs[0]  # 获取自注意力输出
        outputs = self_attention_outputs[1:]  # 如果要输出注意力权重，则将注意力权重添加到输出中

        # 如果存在 lambda_1，则应用 lambda_1 乘以注意力输出
        if self.lambda_1 is not None:
            attention_output = self.lambda_1 * attention_output

        # 第一个残差连接
        hidden_states = self.drop_path(attention_output) + hidden_states

        # 在 Data2VecVision 中，自注意力之后同样会应用层归一化
        layer_output = self.layernorm_after(hidden_states)  # 在隐藏状态上应用层归一化

        # 经过一个中间层
        layer_output = self.intermediate(layer_output)

        # 经过 Data2Vec 输出层
        layer_output = self.data2vec_output(layer_output)

        # 如果存在 lambda_2，则应用 lambda_2 乘以输出
        if self.lambda_2 is not None:
            layer_output = self.lambda_2 * layer_output

        # 第二个残差连接
        layer_output = self.drop_path(layer_output) + hidden_states

        outputs = (layer_output,) + outputs  # 将输出重新组织为元组形式

        return outputs  # 返回最终输出元组
# 从给定链接中获取，稍作修改
# https://github.com/leondgarse/keras_cv_attention_models/blob/main/keras_cv_attention_models/beit/beit.py#L28
# 创建一个 TFData2VecVisionRelativePositionBias 类，继承自 tf.keras.layers.Layer 类
class TFData2VecVisionRelativePositionBias(tf.keras.layers.Layer):
    # 初始化方法，接收一个 Data2VecVisionConfig 的实例和一个窗口大小的元组
    def __init__(self, config: Data2VecVisionConfig, window_size: tuple, **kwargs) -> None:
        super().__init__(**kwargs)
        self.config = config

        self.window_size = window_size
        # +3 for cls_token_pos_len
        # window_size can be something like (14, 14)
        # num_relative_distance 表示相对距离的数量，它等于 (2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3
        self.num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3

        # 调用 get_position_index 方法，获取相对位置索引矩阵
        self.relative_position_index = self.get_position_index()

    # 构建方法，用于构建并初始化模型的权重
    def build(self, input_shape):
        # 创建一个名为 relative_position_bias_table 的可训练权重，形状为 (num_relative_distance, num_attention_heads)
        # 使用 "zeros" 初始化权重的值
        self.relative_position_bias_table = self.add_weight(
            shape=(self.num_relative_distance, self.config.num_attention_heads),
            initializer="zeros",
            trainable=True,
            name="relative_position_bias_table",
        )  # [2*Wh-1 * 2*Ww-1, nH]
        # cls to token & token 2 cls & cls to cls

        # 调用父类的 build 方法
        super().build(input_shape)

    # 获取相对位置索引矩阵的方法
    def get_position_index(self):
        # 根据窗口大小创建二维网格的坐标矩阵 xx 和 yy
        xx, yy = tf.meshgrid(range(self.window_size[0]), range(self.window_size[1]))
        # 创建坐标矩阵的堆叠，形状为 (2, Wh, Ww)
        coords = tf.stack([yy, xx], axis=0)  # [2, Wh, Ww]
        # 将坐标矩阵进行扁平化，形状变为 (2, Wh*Ww)
        coords_flatten = tf.reshape(coords, [2, -1])  # [2, Wh*Ww]

        # 计算相对坐标矩阵，形状为 (2, Wh*Ww, Wh*Ww)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # [2, Wh*Ww, Wh*Ww]
        # 将相对坐标矩阵的维度进行转置，形状变为 (Wh*Ww, Wh*Ww, 2)
        relative_coords = tf.transpose(relative_coords, perm=[1, 2, 0])  # [Wh*Ww, Wh*Ww, 2]

        # 计算相对位置索引矩阵的前半部分，形状为 (Wh*Ww, Wh*Ww)
        xx = (relative_coords[:, :, 0] + self.window_size[0] - 1) * (2 * self.window_size[1] - 1)
        yy = relative_coords[:, :, 1] + self.window_size[1] - 1
        relative_coords = tf.stack([xx, yy], axis=-1)

        # 计算最终的相对位置索引矩阵，形状为 (Wh*Ww, Wh*Ww)
        relative_position_index = tf.reduce_sum(relative_coords, axis=-1)  # [Wh*Ww, Wh*Ww]

        # 创建三个填充值，用于表示 cls_token_pos_len、token 2 cls 和 cls to cls
        top = tf.ones((1, relative_position_index.shape[1]), dtype=relative_position_index.dtype) * (
            self.num_relative_distance - 3
        )
        left = tf.ones((relative_position_index.shape[0], 1), dtype=relative_position_index.dtype) * (
            self.num_relative_distance - 2
        )
        corner = tf.ones((1, 1), dtype=relative_position_index.dtype) * (self.num_relative_distance - 1)

        # 将填充值与相对位置索引矩阵进行拼接，形状变为 (Wh*Ww + 1, Wh*Ww + 1)
        left_corner = tf.concat([corner, left], axis=0)
        relative_position_index = tf.concat([top, relative_position_index], axis=0)
        relative_position_index = tf.concat([left_corner, relative_position_index], axis=1)  # [Wh*Ww + 1, Wh*Ww + 1]
        return relative_position_index

    # 调用层的方法
    def call(self, inputs=None) -> tf.Tensor:
        # 根据相对位置索引矩阵从 relative_position_bias_table 中获取相对位置偏置矩阵
        relative_position_bias = tf.gather(self.relative_position_bias_table, self.relative_position_index, axis=0)
        # 对相对位置偏置矩阵进行转置，形状变为 (num_attention_heads, num_relative_distance)
        return tf.transpose(relative_position_bias, [2, 0, 1])


class TFData2VecVisionEncoder(tf.keras.layers.Layer):
    # 初始化函数，接受Data2VecVisionConfig配置和窗口大小作为输入参数
    def __init__(self, config: Data2VecVisionConfig, window_size: Optional[tuple] = None, **kwargs):
        # 调用父类的初始化函数
        super().__init__(**kwargs)
        # 保存配置信息到对象属性中
        self.config = config
        # 如果配置要使用共享的相对位置偏置，则创建TFData2VecVisionRelativePositionBias对象
        if config.use_shared_relative_position_bias:
            self.relative_position_bias = TFData2VecVisionRelativePositionBias(
                config, window_size=window_size, name="relative_position_bias"
            )
        else:
            # 否则相对位置偏置设为None
            self.relative_position_bias = None

        # 创建存储所有层的列表
        dpr = list(tf.linspace(0.0, config.drop_path_rate, config.num_hidden_layers))  # 创建随机深度衰减规则
        self.layer = [
            # 为每一层创建TFData2VecVisionLayer对象，并存储在列表中
            TFData2VecVisionLayer(
                config,
                window_size=window_size if config.use_relative_position_bias else None,
                drop_path_rate=dpr[i],
                name=f"layer_._{i}",
            )
            for i in range(config.num_hidden_layers)
        ]

    # 前向传播函数
    def call(
        self,
        hidden_states: tf.Tensor,  # 输入的隐藏状态张量
        head_mask: tf.Tensor | None = None,  # 头部掩码，可选
        output_attentions: bool = False,  # 是否输出注意力信息，缺省为False
        output_hidden_states: bool = False,  # 是否输出隐藏状态信息，缺省为False
        return_dict: bool = True,  # 是否返回字典格式的结果，缺省为True
    ) -> Union[tuple, TFBaseModelOutput]:  # 返回结果的数据类型
        # 根据是否输出隐藏状态和注意力信息，创建空的元组
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        # 遍历所有层
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 获取当前层的头部掩码
            layer_head_mask = head_mask[i] if head_mask is not None else None
            # 获取相对位置偏置，如果存在的话
            relative_position_bias = (
                self.relative_position_bias(0.0) if self.relative_position_bias is not None else None
            )
            # 调用层的前向传播函数
            layer_outputs = layer_module(hidden_states, layer_head_mask, output_attentions, relative_position_bias)

            hidden_states = layer_outputs[0]  # 更新隐藏状态为当前层的输出

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 如果不需要返回字典格式的结果，则返回合并后的元组
        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)

        # 返回字典格式的结果
        return TFBaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )
    # 定义网络层的构建方法，构建网络层时需要指定输入的形状
    def build(self, input_shape=None):
        # 如果网络层已经构建过，则直接返回，不执行后续构建操作
        if self.built:
            return
        # 标记网络层已经构建过
        self.built = True
        # 如果存在相对位置偏差属性，则构建相对位置偏差
        if getattr(self, "relative_position_bias", None) is not None:
            # 在相对位置偏差属性下添加一个命名空间
            with tf.name_scope(self.relative_position_bias.name):
                # 构建相对位置偏差
                self.relative_position_bias.build(None)
        # 如果存在网络层属性，则构建网络层
        if getattr(self, "layer", None) is not None:
            # 遍历网络层列表中的每一个层
            for layer in self.layer:
                # 在每个网络层的名称下添加一个命名空间
                with tf.name_scope(layer.name):
                    # 构建每个网络层
                    layer.build(None)
# 声明一个 Keras 序列化类装饰器，用于标记该类可以被序列化
@keras_serializable
# 定义 TFData2VecVisionMainLayer 类，继承自 tf.keras.layers.Layer
class TFData2VecVisionMainLayer(tf.keras.layers.Layer):
    # 指定配置类为 Data2VecVisionConfig
    config_class = Data2VecVisionConfig

    # 初始化函数，接收配置对象 config 和一个布尔值参数 add_pooling_layer，默认为 True
    def __init__(self, config: Data2VecVisionConfig, add_pooling_layer: bool = True, **kwargs):
        # 调用父类的初始化函数
        super().__init__(**kwargs)

        # 将传入的配置对象和布尔值参数保存为对象属性
        self.config = config
        self.add_pooling_layer = add_pooling_layer

        # 创建 TFData2VecVisionEmbeddings 实例对象，传入配置对象 config，命名为 "embeddings"
        self.embeddings = TFData2VecVisionEmbeddings(config, name="embeddings")
        # 创建 TFData2VecVisionEncoder 实例对象，传入配置对象 config 和窗口大小（patch shape），命名为 "encoder"
        self.encoder = TFData2VecVisionEncoder(
            config, window_size=self.embeddings.patch_embeddings.patch_shape, name="encoder"
        )
        # 根据配置决定是否创建 LayerNormalization 层，命名为 "layernorm"
        self.layernorm = (
            tf.identity
            if config.use_mean_pooling
            else tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="layernorm")
        )

        # 如果 add_pooling_layer 为 True，则创建 TFData2VecVisionPooler 实例对象，命名为 "pooler"
        # 否则将 pooler 设为 None
        self.pooler = TFData2VecVisionPooler(config, name="pooler") if add_pooling_layer else None

    # 获取输入嵌入层的函数
    def get_input_embeddings(self) -> tf.keras.layers.Layer:
        return self.embeddings.patch_embeddings

    # 用于裁剪模型的头部（heads）的方法
    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        # 抛出未实现错误，由子类实现具体逻辑
        raise NotImplementedError

    # call 方法用于模型的前向传播
    # 使用 unpack_inputs 装饰器来自动解包输入参数
    @unpack_inputs
    def call(
        self,
        pixel_values: tf.Tensor | None = None,
        bool_masked_pos: tf.Tensor | None = None,
        head_mask: tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
    ) -> Union[tuple, TFData2VecVisionModelOutputWithPooling]:
        # 如果未指定 output_attentions，则使用配置中的值
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 如果未指定 output_hidden_states，则使用配置中的值
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 如果未指定 return_dict，则使用配置中的值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 如果未提供 pixel_values，则引发 ValueError
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # 准备头部遮罩（如果需要）
        # head_mask 中的 1.0 表示保留该头部
        # attention_probs 的形状为 bsz x n_heads x N x N
        # 输入的 head_mask 形状为 [num_heads] 或 [num_hidden_layers x num_heads]
        # 并且 head_mask 被转换为形状 [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        if head_mask is not None:
            # 如果提供了 head_mask，则抛出 NotImplementedError
            raise NotImplementedError
        else:
            # 否则，使用 None 初始化 head_mask
            head_mask = [None] * self.config.num_hidden_layers

        # 使用像素值和可选的 bool_masked_pos（布尔掩码）来嵌入输入
        embedding_output = self.embeddings(pixel_values, bool_masked_pos, training=training)

        # 对嵌入后的输入进行编码
        encoder_outputs = self.encoder(
            embedding_output,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        # 获取编码器的输出序列
        sequence_output = encoder_outputs[0]
        # 应用 layernorm 到序列输出
        sequence_output = self.layernorm(sequence_output)
        # 如果存在池化器，则将序列输出池化
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        # 如果不要求返回字典，则返回编码器的输出
        if not return_dict:
            head_outputs = (sequence_output, pooled_output) if pooled_output is not None else (sequence_output,)
            return head_outputs + encoder_outputs[1:]

        # 如果要求返回字典，则构建相应的输出对象
        return TFData2VecVisionModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回
        if self.built:
            return
        # 标记为已构建
        self.built = True
        # 构建嵌入层
        if getattr(self, "embeddings", None) is not None:
            with tf.name_scope(self.embeddings.name):
                self.embeddings.build(None)
        # 构建编码器层
        if getattr(self, "encoder", None) is not None:
            with tf.name_scope(self.encoder.name):
                self.encoder.build(None)
        # 构建 layernorm 层
        if getattr(self, "layernorm", None) is not None:
            if hasattr(self.layernorm, "name"):
                with tf.name_scope(self.layernorm.name):
                    self.layernorm.build((None, self.config.hidden_size))
        # 构建池化器层
        if getattr(self, "pooler", None) is not None:
            with tf.name_scope(self.pooler.name):
                self.pooler.build(None)
class TFData2VecVisionPooler(tf.keras.layers.Layer):
    def __init__(self, config: Data2VecVisionConfig, **kwargs):
        super().__init__(**kwargs)
        # 初始化层归一化（如果使用平均池化则进行层归一化，否则为None）
        self.layernorm = (
            tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="layernorm")
            if config.use_mean_pooling
            else None
        )
        # 保存配置信息
        self.config = config

    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        if self.layernorm is not None:
            # 对patch tokens的最终隐藏状态进行平均池化
            patch_tokens = hidden_states[:, 1:, :]
            pooled_output = self.layernorm(tf.reduce_mean(patch_tokens, axis=1))
        else:
            # 通过简单地取[CLS] token的最终隐藏状态进行池化
            pooled_output = hidden_states[:, 0]

        return pooled_output

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "layernorm", None) is not None:
            if hasattr(self.layernorm, "name"):
                with tf.name_scope(self.layernorm.name):
                    self.layernorm.build((None, self.config.hidden_size))


class TFData2VecVisionPreTrainedModel(TFPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    # 指定配置类
    config_class = Data2VecVisionConfig
    # 基础模型前缀
    base_model_prefix = "data2vec_vision"
    # 主输入名称
    main_input_name = "pixel_values"
    # 加载时要忽略的关键字
    _keys_to_ignore_on_load_unexpected = [r"relative_position_index"]


DATA2VEC_VISION_START_DOCSTRING = r"""
    This model inherits from [`TFPreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.).
    ...（后续文档内容太长，省略）
    # 模型输入参数说明：
    # 一个仅包含`pixel_values`的单个张量，其余参数为空：`model(pixel_values)`
    # 一个长度不等的列表，其中包含一个或多个输入张量，按照文档字符串中给定的顺序：
    # `model([pixel_values, attention_mask])` 或 `model([pixel_values, attention_mask, token_type_ids])`
    # 一个字典，其中包含一个或多个与文档字符串中给定输入名称相关联的输入张量:
    # `model({"pixel_values": pixel_values, "token_type_ids": token_type_ids})`

    # 当使用子类化创建模型和层时
    # [subclassing](https://keras.io/guides/making_new_layers_and_models_via_subclassing/)，
    # 则不需要担心任何此类细节，可以像将输入传递给任何其他Python函数一样传递！

    # 参数:
    # config ([`Data2VecVisionConfig`])：模型配置类，包含模型的所有参数。
    # 通过配置文件初始化不会加载与模型相关的权重，只加载配置。查看[`~TFPreTrainedModel.from_pretrained`]方法以加载模型权重。
# 定义了一个包含文档字符串的全局常量，描述了Data2VecVision模型的输入参数

# 定义了Data2VecVision模型的TF版本
class TFData2VecVisionModel(TFData2VecVisionPreTrainedModel):

    # 初始化方法
    def __init__(self, config: Data2VecVisionConfig, add_pooling_layer: bool = False, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.config = config

        # 创建Data2VecVisionMainLayer的实例作为模型的核心层
        self.data2vec_vision = TFData2VecVisionMainLayer(
            config, add_pooling_layer=add_pooling_layer, name="data2vec_vision"
        )

    # 返回输入的嵌入表示
    def get_input_embeddings(self):
        return self.data2vec_vision.get_input_embeddings()

    # 模型的前向传播方法
    def call(
        self,
        pixel_values: TFModelInputType | None = None,
        bool_masked_pos: tf.Tensor | None = None,
        head_mask: np.ndarray | tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
    ) -> Union[tuple, TFData2VecVisionModelOutputWithPooling]:
        r"""
        bool_masked_pos (`tf.Tensor` of shape `(batch_size, num_patches)`, *optional*):
            Boolean masked positions. Indicates which patches are masked (1) and which aren't (0).
        """
        # 调用 data2vec_vision 方法，并将参数传递给它
        outputs = self.data2vec_vision(
            pixel_values=pixel_values,
            bool_masked_pos=bool_masked_pos,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        # 返回 data2vec_vision 方法的输出结果
        return outputs

    def build(self, input_shape=None):
        # 如果已经构建，则直接返回
        if self.built:
            return
        # 将构建标志设置为 True
        self.built = True
        # 如果 data2vec_vision 方法已定义
        if getattr(self, "data2vec_vision", None) is not None:
            # 使用 data2vec_vision 方法的名称作为命名空间
            with tf.name_scope(self.data2vec_vision.name):
                # 构建 data2vec_vision 方法
                self.data2vec_vision.build(None)
# 定义一个 Data2VecVision 模型，带有一个图像分类头部（将最终隐藏状态的平均值作为输入，然后连接一个线性层）用于 ImageNet 等任务
@add_start_docstrings(
    """
    Data2VecVision Model transformer with an image classification head on top (a linear layer on top of
    the average of the final hidden states of the patch tokens) e.g. for ImageNet.
    """,
    DATA2VEC_VISION_START_DOCSTRING,
)
class TFData2VecVisionForImageClassification(TFData2VecVisionPreTrainedModel, TFSequenceClassificationLoss):
    # 初始化函数，接受 Data2VecVisionConfig 配置
    def __init__(self, config: Data2VecVisionConfig, *inputs, **kwargs):
        # 调用父类的初始化函数
        super().__init__(config, *inputs, **kwargs)

        # 设置模型的标签数量
        self.num_labels = config.num_labels
        # 创建 Data2VecVisionMainLayer 层
        self.data2vec_vision = TFData2VecVisionMainLayer(config, add_pooling_layer=True, name="data2vec_vision")

        # 分类器头部
        self.classifier = tf.keras.layers.Dense(
            units=config.num_labels,  # 单元数量为标签数量
            kernel_initializer=get_initializer(config.initializer_range),  # 设置初始化方式
            name="classifier",
        )
        self.config = config

    # call 函数，接受多个参数，用于前向传播计算
    @unpack_inputs
    @add_start_docstrings_to_model_forward(DATA2VEC_VISION_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_IMAGE_CLASS_CHECKPOINT,
        output_type=TFSequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_IMAGE_CLASS_EXPECTED_OUTPUT,
    )
    def call(
        self,
        pixel_values: TFModelInputType | None = None,
        head_mask: np.ndarray | tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: np.ndarray | tf.Tensor | None = None,
        training: Optional[bool] = False,
    ) -> Union[TFSequenceClassifierOutput, tuple]:
        r"""
        labels (`tf.Tensor` or `np.ndarray` of shape `(batch_size,)`, *optional`):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        # 设置默认的返回字典，如果未指定则使用模型配置的返回字典
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 使用数据2向量模型对图像进行处理
        outputs = self.data2vec_vision(
            pixel_values=pixel_values,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        # 如果使用返回字典，则获取输出的池化向量；否则获取输出的 logit
        pooled_output = outputs.pooler_output if return_dict else outputs[1]
        logits = self.classifier(pooled_output)
        loss = None if labels is None else self.hf_compute_loss(labels=labels, logits=logits)

        # 如果不使用返回字典，则返回 logits 和 outputs 的结果，可能包含 loss
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 使用返回字典，则返回 TFSequenceClassifierOutput 对象
        return TFSequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 如果 data2vec_vision 模型存在，则构建该模型
        if getattr(self, "data2vec_vision", None) is not None:
            with tf.name_scope(self.data2vec_vision.name):
                self.data2vec_vision.build(None)
        # 如果 classifier 模型存在，则构建该模型
        if getattr(self, "classifier", None) is not None:
            with tf.name_scope(self.classifier.name):
                self.classifier.build([None, None, self.config.hidden_size])
class TFData2VecVisionConvModule(tf.keras.layers.Layer):
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
        # 初始化函数，定义卷积块的各个参数
        super().__init__(**kwargs)
        # 创建卷积层，设置卷积核数、核大小、填充方式、是否使用偏置、膨胀率等
        self.conv = tf.keras.layers.Conv2D(
            filters=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            use_bias=bias,
            dilation_rate=dilation,
            name="conv",
        )
        # 创建批归一化层，用于规范化卷积层的输出
        self.bn = tf.keras.layers.BatchNormalization(name="bn", momentum=0.9, epsilon=1e-5)
        # 激活函数设置为ReLU
        self.activation = tf.nn.relu
        # 记录输入通道数和输出通道数
        self.in_channels = in_channels
        self.out_channels = out_channels

    def call(self, input: tf.Tensor) -> tf.Tensor:
        # 前向传播函数，执行卷积、批归一化和激活操作
        output = self.conv(input)
        output = self.bn(output)
        output = self.activation(output)
        return output

    def build(self, input_shape=None):
        # 构建函数，用于构建卷积层和批归一化层
        if self.built:
            return
        self.built = True
        # 如果已经构建了卷积层，则直接返回
        if getattr(self, "conv", None) is not None:
            with tf.name_scope(self.conv.name):
                # 构建卷积层，根据输入形状和通道数构建
                self.conv.build([None, None, None, self.in_channels])
        # 如果已经构建了批归一化层，则直接返回
        if getattr(self, "bn", None) is not None:
            with tf.name_scope(self.bn.name):
                # 构建批归一化层，根据输出通道数构建
                self.bn.build((None, None, None, self.out_channels))


class TFAdaptiveAvgPool2D(tf.keras.layers.Layer):
    # 自定义的二维自适应平均池化层
    def __init__(self, output_dims: Tuple[int, int], input_ordering: str = "NHWC", **kwargs):
        # 初始化函数，定义输出维度和输入数据的顺序
        super().__init__(**kwargs)
        self.output_dims = output_dims
        self.input_ordering = input_ordering
        # 检查输入数据的顺序是否合法，只能是"NCHW"或"NHWC"
        if input_ordering not in ("NCHW", "NHWC"):
            raise ValueError("Unrecognized input_ordering, should be 'NCHW' or 'NHWC'!")
        # 获取高度和宽度轴的索引
        self.h_axis = input_ordering.index("H")
        self.w_axis = input_ordering.index("W")
    # 定义一个方法，用于调整输入张量的大小
    def call(self, inputs: tf.Tensor):
        # 检查输入张量的顺序，确定输入形状
        if self.input_ordering == "NHWC":
            # 如果输入顺序为NHWC，提取出除了batch大小外的高度和宽度
            input_shape = inputs.shape[1:3]
        else:
            # 如果输入顺序为NCHW，提取出除了batch大小外的高度和宽度
            input_shape = inputs.shape[2:]

        # 将任务拆分为每种可能的情况
        # 首先，如果我们要调整大小到1，只需使用tf.reduce_mean
        if self.output_dims[0] == self.output_dims[1] == 1:
            # 确定要进行平均值池化的维度
            if self.input_ordering == "NHWC":
                reduce_dims = [1, 2]
            else:
                reduce_dims = [2, 3]
            # 对输入张量进行平均值池化，保持维度
            return tf.reduce_mean(inputs, axis=reduce_dims, keepdims=True)
        # 其次，如果我们在两个维度上按整数因子调整大小，我们可以采取快速捷径
        elif input_shape[0] % self.output_dims[0] == 0 and input_shape[1] % self.output_dims[1] == 0:
            # 计算高度和宽度的调整因子
            h_resize = int(input_shape[0] // self.output_dims[0])
            w_resize = int(input_shape[1] // self.output_dims[1])
            # 使用平均值池化进行快速调整大小
            return tf.nn.avg_pool2d(
                inputs,
                ksize=(h_resize, w_resize),  # 池化窗口大小
                strides=(h_resize, w_resize),  # 池化步长
                padding="VALID",  # 不填充
                data_format=self.input_ordering,  # 数据格式
            )
        else:
            # 最后，如果我们无法采取快捷方式，我们在每个轴上进行一维池化。pseudo_1d_pool 将针对可以进行整数调整大小的维度采取快捷方式。
            # 它还可以处理放大。
            # 在水平方向进行一维池化
            h_pooled = self.pseudo_1d_pool(inputs, h_pooling=True)
            # 在垂直方向进行一维池化
            return self.pseudo_1d_pool(h_pooled, h_pooling=False)
# 定义一个名为TFData2VecVisionPyramidPoolingModule的类，继承自tf.keras.layers.Layer，用于实现PSPNet中的金字塔池化模块
class TFData2VecVisionPyramidPoolingModule(tf.keras.layers.Layer):
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
        self.pool_scales = pool_scales
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.layer_list = []
        # 遍历pool_scales，生成对应的AdaptiveAvgPool2D和Data2VecVisionConvModule模块，并放入layer_list中
        for idx, pool_scale in enumerate(pool_scales):
            pool_scale = pool_scale if isinstance(pool_scale, collections.abc.Iterable) else (pool_scale, pool_scale)
            self.layer_list.append(
                [
                    TFAdaptiveAvgPool2D(output_dims=pool_scale),
                    TFData2VecVisionConvModule(
                        in_channels=in_channels, out_channels=self.out_channels, kernel_size=1, name=f"{idx}.1"
                    ),
                ]
            )

    # 定义模型的前向传播过程
    def call(self, x: tf.Tensor) -> List[tf.Tensor]:
        ppm_outs = []
        inputs = x

        # 遍历layer_list中的模块，进行前向传播运算，并将结果添加到ppm_outs列表中
        for ppm in self.layer_list:
            for layer_module in ppm:
                ppm_out = layer_module(x)
                x = ppm_out

            # 将ppm_out上采样到输入x的尺寸，并将结果添加到ppm_outs列表中
            upsampled_ppm_out = tf.image.resize(ppm_out, size=shape_list(inputs)[1:-1], method="bilinear")
            ppm_outs.append(upsampled_ppm_out)
        return ppm_outs

    # 构建模型的权重
    def build(self, input_shape=None):
        # 遍历layer_list中的模块，调用build方法构建权重
        for layer in self.layer_list:
            for layer_module in layer:
                with tf.name_scope(layer_module.name):
                    layer_module.build(None)


class TFData2VecVisionUperHead(tf.keras.layers.Layer):
    """
    Unified Perceptual Parsing for Scene Understanding. This head is the implementation of
    [UPerNet](https://arxiv.org/abs/1807.10221).

    Based on OpenMMLab's implementation, found in https://github.com/open-mmlab/mmsegmentation.
    """
    # Data2VecVisionModel 类的初始化方法
    def __init__(self, config: Data2VecVisionConfig, **kwargs) -> None:
        # 调用父类 (Model) 的初始化方法
        super().__init__(**kwargs)
        
        # 根据配置信息设置池化尺度
        self.pool_scales = config.pool_scales  # e.g. (1, 2, 3, 6)
        
        # 设置每个层的输入通道数
        self.in_channels = [config.hidden_size] * 4  # e.g. [768, 768, 768, 768]
        
        # 设置模型内部通道数
        self.channels = config.hidden_size
        
        # 创建分类器，用于输出类别预测
        self.classifier = tf.keras.layers.Conv2D(config.num_labels, kernel_size=1, name="classifier")
    
        # 创建 PSP (Pyramid Scene Parsing) 模块
        self.psp_modules = TFData2VecVisionPyramidPoolingModule(
            self.pool_scales, self.in_channels[-1], self.channels, name="psp_modules"
        )
        
        # 创建瓶颈模块，用于整合 PSP 模块的输出
        self.bottleneck = TFData2VecVisionConvModule(
            self.in_channels[-1] + len(self.pool_scales) * self.channels,
            self.channels,
            kernel_size=3,
            padding="same",
            name="bottleneck",
        )
    
        # 创建 FPN (Feature Pyramid Network) 模块
        self.lateral_convs = []
        self.fpn_convs = []
        for idx, in_channels in enumerate(self.in_channels[:-1]):  # skip the top layer
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
    
        # 创建 FPN 瓶颈模块，用于整合 FPN 模块的输出
        self.fpn_bottleneck = TFData2VecVisionConvModule(
            in_channels=len(self.in_channels) * self.channels,
            out_channels=self.channels,
            kernel_size=3,
            padding="same",
            name="fpn_bottleneck",
        )
    
    # PSP 模块的前向传播过程
    def psp_forward(self, inputs):
        # 获取输入的最后一个特征图
        x = inputs[-1]
        
        # 将原始特征图添加到输出列表
        psp_outs = [x]
        
        # 将 PSP 模块的输出添加到输出列表
        psp_outs.extend(self.psp_modules(x))
        
        # 将所有输出特征图拼接在一起
        psp_outs = tf.concat(psp_outs, axis=-1)
        
        # 通过瓶颈模块得到最终输出
        output = self.bottleneck(psp_outs)
    
        return output
    def call(self, encoder_hidden_states: tf.Tensor) -> tf.Tensor:
        # 构建横向连接
        laterals = [lateral_conv(encoder_hidden_states[i]) for i, lateral_conv in enumerate(self.lateral_convs)]

        # 将 PSP 模块的输出添加到横向连接结果中
        laterals.append(self.psp_forward(encoder_hidden_states))

        # 构建自顶向下路径
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = shape_list(laterals[i - 1])[1:-1]
            laterals[i - 1] = laterals[i - 1] + tf.image.resize(laterals[i], size=prev_shape, method="bilinear")

        # 构建输出
        fpn_outs = [self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels - 1)]
        # 将 PSP 特征添加到输出中
        fpn_outs.append(laterals[-1])

        # 调整大小以匹配最底层特征的大小
        for i in range(used_backbone_levels - 1, 0, -1):
            fpn_outs[i] = tf.image.resize(fpn_outs[i], size=shape_list(fpn_outs[0])[1:-1], method="bilinear")
        # 合并所有特征
        fpn_outs = tf.concat(fpn_outs, axis=-1)
        # 经过 FPN 瓶颈层
        output = self.fpn_bottleneck(fpn_outs)
        # 经过分类器
        output = self.classifier(output)

        return output

    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回
        if self.built:
            return
        # 设置构建标志
        self.built = True
        # 构建分类器
        if getattr(self, "classifier", None) is not None:
            with tf.name_scope(self.classifier.name):
                self.classifier.build([None, None, None, self.channels])
        # 构建 PSP 模块
        if getattr(self, "psp_modules", None) is not None:
            with tf.name_scope(self.psp_modules.name):
                self.psp_modules.build(None)
        # 构建瓶颈层
        if getattr(self, "bottleneck", None) is not None:
            with tf.name_scope(self.bottleneck.name):
                self.bottleneck.build(None)
        # 构建 FPN 瓶颈层
        if getattr(self, "fpn_bottleneck", None) is not None:
            with tf.name_scope(self.fpn_bottleneck.name):
                self.fpn_bottleneck.build(None)
        # 构建横向连接中的每一层
        for layer in self.lateral_convs:
            with tf.name_scope(layer.name):
                layer.build(None)
        # 构建 FPN 网络中的每一层
        for layer in self.fpn_convs:
            with tf.name_scope(layer.name):
                layer.build(None)
class TFData2VecVisionFCNHead(tf.keras.layers.Layer):
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
        self.in_channels = config.hidden_size  # 设置输入通道数为配置文件中的隐藏层大小
        self.channels = config.auxiliary_channels  # 设置通道数为配置文件中的辅助通道大小
        self.num_convs = config.auxiliary_num_convs  # 设置卷积层数为配置文件中的辅助卷积数量
        self.concat_input = config.auxiliary_concat_input  # 设置是否将输入与输出级联为配置文件中的辅助级联输入
        self.in_index = in_index  # 设置输入索引为给定的索引值

        convs = []  # 创建一个空列表用于存储卷积模块
        convs.append(  # 添加一个卷积模块到列表中
            TFData2VecVisionConvModule(
                in_channels=self.in_channels,  # 输入通道设置为设定的输入通道大小
                out_channels=self.channels,  # 输出通道设置为设定的辅助通道大小
                kernel_size=kernel_size,  # 设置卷积核大小为给定的大小
                padding="same",  # 设置填充模式为相同
                dilation=dilation,  # 设置膨胀率为给定的膨胀率
                name="convs.0",  # 设置名称
            )
        )
        for i in range(self.num_convs - 1):  # 循环辅助卷积次数减1次
            convs.append(  # 添加一个卷积模块到列表中
                TFData2VecVisionConvModule(
                    in_channels=self.channels,  # 输入通道设置为辅助通道大小
                    out_channels=self.channels,  # 输出通道设置为辅助通道大小
                    kernel_size=kernel_size,  # 设置卷积核大小为给定的大小
                    padding="same",  # 设置填充模式为相同
                    dilation=dilation,  # 设置膨胀率为给定的膨胀率
                    name=f"conv_module_{i+2}",  # 设置名称
                )
            )
        if self.num_convs == 0:  # 如果辅助卷积次数为0
            self.convs = [tf.identity]  # 将卷积模块设置为恒等映射
        else:
            self.convs = convs  # 否则将卷积模块设置为之前创建的卷积列表
        if self.concat_input:  # 如果进行级联输入
            self.conv_cat = TFData2VecVisionConvModule(  # 创建一个级联卷积模块
                self.in_channels + self.channels,  # 设置输入通道大小为输入通道大小与辅助通道大小之和
                out_channels=self.channels,  # 输出通道设置为辅助通道大小
                kernel_size=kernel_size,  # 设置卷积核大小为给定的大小
                padding="same",  # 设置填充模式为相同
                name="conv_cat",  # 设置名称
            )

        self.classifier = tf.keras.layers.Conv2D(config.num_labels, kernel_size=1, name="classifier")  # 创建一个卷积分类器，输出通道数为类别数，卷积
    # 定义构建函数，用于构建模型结构
    def build(self, input_shape=None):
        # 如果模型已经构建好了，则直接返回，不再重复构建
        if self.built:
            return
        # 标记模型已经构建
        self.built = True
        # 检查是否存在分类器，如果存在则构建分类器
        if getattr(self, "classifier", None) is not None:
            # 使用分类器的名称创建一个命名空间
            with tf.name_scope(self.classifier.name):
                # 根据输入形状构建分类器
                self.classifier.build([None, None, None, self.channels])
        # 检查是否存在卷积连接器，如果存在则构建卷积连接器
        if getattr(self, "conv_cat", None) is not None:
            # 使用卷积连接器的名称创建一个命名空间
            with tf.name_scope(self.conv_cat.name):
                # 根据输入形状构建卷积连接器
                self.conv_cat.build(None)
# 使用装饰器添加模型的文档字符串，介绍了这个模型是一个带有语义分割头部的 Data2VecVision 模型，例如用于 ADE20k、CityScapes 等任务
@add_start_docstrings(
    """
    Data2VecVision Model transformer with a semantic segmentation head on top e.g. for ADE20k, CityScapes.
    """,
    DATA2VEC_VISION_START_DOCSTRING,
)
# 定义 TFData2VecVisionForSemanticSegmentation 类，继承自 TFData2VecVisionPreTrainedModel
class TFData2VecVisionForSemanticSegmentation(TFData2VecVisionPreTrainedModel):
    # 初始化方法
    def __init__(self, config: Data2VecVisionConfig, *inputs, **kwargs) -> None:
        # 调用父类的初始化方法
        super().__init__(config, *inputs, **kwargs)
        # 获取标签的数量
        self.num_labels = config.num_labels
        # 创建 Data2VecVision 主层，并设置不添加池化层，命名为"data2vec_vision"
        self.data2vec_vision = TFData2VecVisionMainLayer(config, add_pooling_layer=False, name="data2vec_vision")

        # FPNs
        # 创建 FPN1，包括转置卷积、批归一化和激活函数
        self.fpn1 = [
            tf.keras.layers.Conv2DTranspose(config.hidden_size, kernel_size=2, strides=2, name="fpn1.0"),
            tf.keras.layers.BatchNormalization(name="fpn1.1", momentum=0.9, epsilon=1e-5),
            tf.keras.layers.Activation("gelu"),
            tf.keras.layers.Conv2DTranspose(config.hidden_size, kernel_size=2, strides=2, name="fpn1.3"),
        ]
        # 创建 FPN2，包括转置卷积
        self.fpn2 = [tf.keras.layers.Conv2DTranspose(config.hidden_size, kernel_size=2, strides=2, name="fpn2.0")]

        # FPN3 和 FPN4 使用标识函数和最大池化层，分别对应语义分割任务中的上采样和下采样
        self.fpn3 = tf.identity
        self.fpn4 = tf.keras.layers.MaxPool2D(pool_size=2, strides=2)

        # 语义分割头部
        # 创建解码头部
        self.decode_head = TFData2VecVisionUperHead(config, name="decode_head")
        # 如果使用辅助头部，则创建辅助头部，否则设为 None
        self.auxiliary_head = (
            TFData2VecVisionFCNHead(config, name="auxiliary_head") if config.use_auxiliary_head else None
        )

    # 计算损失函数
    def compute_loss(self, logits, auxiliary_logits, labels):
        # 将 logits 上采样到原始图像大小
        if len(shape_list(labels)) > 3:
            label_interp_shape = shape_list(labels)[1:-1]
        else:
            label_interp_shape = shape_list(labels)[-2:]

        upsampled_logits = tf.image.resize(logits, size=label_interp_shape, method="bilinear")
        if auxiliary_logits is not None:
            upsampled_auxiliary_logits = tf.image.resize(auxiliary_logits, size=label_interp_shape, method="bilinear")
        # 计算加权损失
        loss_fct = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction="none")

        # 从 https://www.tensorflow.org/text/tutorials/transformer#loss_and_metrics 复制的代码
        # 创建函数以对计算损失时要忽略的索引进行掩码处理
        def masked_loss(real, pred):
            mask = tf.math.logical_not(tf.math.equal(real, self.config.semantic_loss_ignore_index))
            loss_ = loss_fct(real, pred)
            mask = tf.cast(mask, dtype=loss_.dtype)
            loss_ *= mask
            reduced_masked_loss = tf.reduce_sum(loss_) / tf.reduce_sum(mask)
            return tf.reshape(reduced_masked_loss, (1,))

        # 计算主损失和辅助损失
        main_loss = masked_loss(labels, upsampled_logits)
        auxiliary_loss = masked_loss(labels, upsampled_auxiliary_logits)
        # 组合主损失和辅助损失
        loss = main_loss + self.config.auxiliary_loss_weight * auxiliary_loss

        return loss

    # 解压输入
    @unpack_inputs
    @add_start_docstrings_to_model_forward(DATA2VEC_VISION_INPUTS_DOCSTRING)
    # 将DATA2VEC_VISION_INPUTS_DOCSTRING添加到模型前向方法的文档字符串中

    @replace_return_docstrings(output_type=TFSemanticSegmenterOutput, config_class=_CONFIG_FOR_DOC)
    # 替换返回值的文档字符串为指定的类型和配置类

    def call(
        self,
        pixel_values: tf.Tensor | None = None,
        head_mask: tf.Tensor | None = None,
        labels: tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    # 定义模型的前向方法，接受输入参数pixel_values, head_mask, labels, output_attentions, output_hidden_states, return_dict

    def build(self, input_shape=None):
    # 定义模型的build方法，用于构建模型

        if self.built:
            return
        # 如果模型已经构建完成，则直接返回

        self.built = True
        # 将模型标记为已构建的状态

        if getattr(self, "data2vec_vision", None) is not None:
            # 如果存在data2vec_vision属性，则执行下面代码块
            with tf.name_scope(self.data2vec_vision.name):
                self.data2vec_vision.build(None)
        # 使用tf.name_scope创建命名空间，然后构建data2vec_vision

        if getattr(self, "decode_head", None) is not None:
            # 如果存在decode_head属性，则执行下面代码块
            with tf.name_scope(self.decode_head.name):
                self.decode_head.build(None)
        # 使用tf.name_scope创建命名空间，然后构建decode_head

        if getattr(self, "auxiliary_head", None) is not None:
            # 如果存在auxiliary_head属性，则执行下面代码块
            with tf.name_scope(self.auxiliary_head.name):
                self.auxiliary_head.build(None)
        # 使用tf.name_scope创建命名空间，然后构建auxiliary_head

        if getattr(self, "fpn1", None) is not None:
            # 如果存在fpn1属性，则执行下面代码块
            with tf.name_scope(self.fpn1[0].name):
                self.fpn1[0].build([None, None, None, self.config.hidden_size])
            # 使用tf.name_scope创建命名空间，然后构建fpn1的第一个元素

            with tf.name_scope(self.fpn1[1].name):
                self.fpn1[1].build((None, None, None, self.config.hidden_size))
            # 使用tf.name_scope创建命名空间，然后构建fpn1的第二个元素

            with tf.name_scope(self.fpn1[3].name):
                self.fpn1[3].build([None, None, None, self.config.hidden_size])
            # 使用tf.name_scope创建命名空间，然后构建fpn1的第三个元素

        if getattr(self, "fpn2", None) is not None:
            # 如果存在fpn2属性，则执行下面代码块
            with tf.name_scope(self.fpn2[0].name):
                self.fpn2[0].build([None, None, None, self.config.hidden_size])
            # 使用tf.name_scope创建命名空间，然后构建fpn2的第一个元素
```