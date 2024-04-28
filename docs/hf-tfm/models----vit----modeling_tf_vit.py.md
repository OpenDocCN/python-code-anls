# `.\transformers\models\vit\modeling_tf_vit.py`

```py
# 设置文件编码为 utf-8
# 版权声明
# 根据 Apache 许可证 2.0 版本，除非符合许可证，否则不得使用此文件
# 可以在以下网址获取许可证副本
# http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于“原样”分发的，没有任何明示或暗示的保证或条件
# 请查看许可证以获取有关特定语言的权限和限制
""" TF 2.0 ViT model."""

# 导入必要的库
from __future__ import annotations
import collections.abc
import math
from typing import Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import TFBaseModelOutput, TFBaseModelOutputWithPooling, TFSequenceClassifierOutput
from ...modeling_tf_utils import (
    TFModelInputType,
    TFPreTrainedModel,
    TFSequenceClassificationLoss,
    get_initializer,
    keras_serializable,
    unpack_inputs,
)
from ...tf_utils import shape_list, stable_softmax
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging

# 获取日志记录器
logger = logging.get_logger(__name__)

# 通用文档字符串
_CONFIG_FOR_DOC = "ViTConfig"

# 基本文档字符串
_CHECKPOINT_FOR_DOC = "google/vit-base-patch16-224-in21k"
_EXPECTED_OUTPUT_SHAPE = [1, 197, 768]

# 图像分类文档字符串
_IMAGE_CLASS_CHECKPOINT = "google/vit-base-patch16-224"
_IMAGE_CLASS_EXPECTED_OUTPUT = "Egyptian cat"

# 定义 TFViTEmbeddings 类，用于构建 CLS token、位置和补丁嵌入
class TFViTEmbeddings(tf.keras.layers.Layer):
    
    def __init__(self, config: ViTConfig, **kwargs):
        super().__init__(**kwargs)
        
        # 初始化补丁嵌入
        self.patch_embeddings = TFViTPatchEmbeddings(config, name="patch_embeddings")
        # 添加丢弃层
        self.dropout = tf.keras.layers.Dropout(rate=config.hidden_dropout_prob)
        self.config = config

    def build(self, input_shape=None):
        # 获取补丁数量
        num_patches = self.patch_embeddings.num_patches
        # 添加 CLS token
        self.cls_token = self.add_weight(
            shape=(1, 1, self.config.hidden_size),
            initializer=get_initializer(self.config.initializer_range),
            trainable=True,
            name="cls_token",
        )
        # 添加位置嵌入
        self.position_embeddings = self.add_weight(
            shape=(1, num_patches + 1, self.config.hidden_size),
            initializer=get_initializer(self.config.initializer_range),
            trainable=True,
            name="position_embeddings",
        )

        if self.built:
            return
        self.built = True
        if getattr(self, "patch_embeddings", None) is not None:
            with tf.name_scope(self.patch_embeddings.name):
                self.patch_embeddings.build(None)
    # 插值预训练位置编码，以便在更高分辨率图像上使用模型
    def interpolate_pos_encoding(self, embeddings, height, width) -> tf.Tensor:
        """
        This method allows to interpolate the pre-trained position encodings, to be able to use the model on higher
        resolution images.

        Source:
        https://github.com/facebookresearch/dino/blob/de9ee3df6cf39fac952ab558447af1fa1365362a/vision_transformer.py#L174
        """

        # 获取嵌入的形状信息
        batch_size, seq_len, dim = shape_list(embeddings)
        num_patches = seq_len - 1

        # 获取位置编码的形状信息
        _, num_positions, _ = shape_list(self.position_embeddings)
        num_positions -= 1

        # 如果补丁数等于位置数且高度等于宽度，则直接返回位置编码
        if num_patches == num_positions and height == width:
            return self.position_embeddings
        # 分别获取类别位置编码和补丁位置编码
        class_pos_embed = self.position_embeddings[:, :1]
        patch_pos_embed = self.position_embeddings[:, 1:]
        h0 = height // self.config.patch_size
        w0 = width // self.config.patch_size
        # 对补丁位置编码进行插值
        patch_pos_embed = tf.image.resize(
            images=tf.reshape(
                patch_pos_embed, shape=(1, int(math.sqrt(num_positions)), int(math.sqrt(num_positions)), dim)
            ),
            size=(h0, w0),
            method="bicubic",
        )

        shape = shape_list(patch_pos_embed)
        assert h0 == shape[-3] and w0 == shape[-2]
        patch_pos_embed = tf.reshape(tensor=patch_pos_embed, shape=(1, -1, dim))
        # 将类别位置编码和补丁位置编码拼接在一起
        return tf.concat(values=(class_pos_embed, patch_pos_embed), axis=1)

    # 模型调用方法，传入像素值张量，是否插值位置编码，是否训练
    def call(
        self, pixel_values: tf.Tensor, interpolate_pos_encoding: bool = False, training: bool = False
    ) -> tf.Tensor:
        # 获取像素值张量的形状信息
        batch_size, num_channels, height, width = shape_list(pixel_values)
        # 获取嵌入
        embeddings = self.patch_embeddings(
            pixel_values, interpolate_pos_encoding=interpolate_pos_encoding, training=training
        )

        # 将[CLS]标记添加到嵌入的补丁标记中
        cls_tokens = tf.repeat(self.cls_token, repeats=batch_size, axis=0)
        embeddings = tf.concat((cls_tokens, embeddings), axis=1)

        # 为每个标记添加位置编码
        if interpolate_pos_encoding:
            embeddings = embeddings + self.interpolate_pos_encoding(embeddings, height, width)
        else:
            embeddings = embeddings + self.position_embeddings

        embeddings = self.dropout(embeddings, training=training)

        return embeddings
# 基于 timm 实现的代码，可以在以下链接找到：
# https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
class TFViTPatchEmbeddings(tf.keras.layers.Layer):
    """
    这个类将形状为 `(batch_size, num_channels, height, width)` 的 `pixel_values` 转换为形状为 `(batch_size, seq_length, hidden_size)` 的初始
    `hidden_states`（patch embeddings），以供 Transformer 消费。
    """

    def __init__(self, config: ViTConfig, **kwargs):
        super().__init__(**kwargs)
        image_size, patch_size = config.image_size, config.patch_size
        num_channels, hidden_size = config.num_channels, config.hidden_size

        image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.num_channels = num_channels
        self.config = config

        # 创建一个卷积层，用于将输入的像素值投影到隐藏空间
        self.projection = tf.keras.layers.Conv2D(
            filters=hidden_size,
            kernel_size=patch_size,
            strides=patch_size,
            padding="valid",
            data_format="channels_last",
            use_bias=True,
            kernel_initializer=get_initializer(self.config.initializer_range),
            bias_initializer="zeros",
            name="projection",
        )

    def call(
        self, pixel_values: tf.Tensor, interpolate_pos_encoding: bool = False, training: bool = False
    ) -> tf.Tensor:
        # 从输入张量的形状中获取批大小、通道数、高度和宽度
        batch_size, num_channels, height, width = shape_list(pixel_values)
        # 如果执行即时执行模式并且通道数与配置中设置的不匹配，则引发值错误
        if tf.executing_eagerly() and num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
            )
        # 如果不插值位置编码
        if not interpolate_pos_encoding:
            # 如果执行即时执行模式并且高度或宽度与模型配置不匹配，则引发值错误
            if tf.executing_eagerly():
                if height != self.image_size[0] or width != self.image_size[1]:
                    raise ValueError(
                        f"Input image size ({height}*{width}) doesn't match model"
                        f" ({self.image_size[0]}*{self.image_size[1]})."
                    )

        # 在 CPU 上运行时，`tf.keras.layers.Conv2D` 不支持 `NCHW` 格式。
        # 所以将输入格式从 `NCHW` 更改为 `NHWC`。
        # 形状 = (batch_size, in_height, in_width, in_channels=num_channels)
        pixel_values = tf.transpose(pixel_values, perm=(0, 2, 3, 1))

        # 对输入张量进行投影
        projection = self.projection(pixel_values)

        # 将二维空间维度转换为单个时间维度。
        # 形状 = (batch_size, num_patches, out_channels=embed_dim)
        num_patches = (width // self.patch_size[1]) * (height // self.patch_size[0])
        embeddings = tf.reshape(tensor=projection, shape=(batch_size, num_patches, -1))

        return embeddings

    def build(self, input_shape=None):
        # 如果已经构建，则返回
        if self.built:
            return
        # 设置已构建标志为 True
        self.built = True
        # 如果投影层已存在
        if getattr(self, "projection", None) is not None:
            # 在投影层名称范围内构建投影层
            with tf.name_scope(self.projection.name):
                self.projection.build([None, None, None, self.num_channels])
```  
class TFViTSelfAttention(tf.keras.layers.Layer):
    # 定义一个自注意力层，继承自 tf.keras.layers.Layer
    def __init__(self, config: ViTConfig, **kwargs):
        # 初始化函数，接受 ViTConfig 类型的参数 config 和其他关键字参数
        super().__init__(**kwargs)
        # 调用父类的初始化函数

        if config.hidden_size % config.num_attention_heads != 0:
            # 如果隐藏层大小不能被注意力头数整除，抛出数值错误
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number "
                f"of attention heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        # 设置注意力头数
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        # 计算每个注意力头的大小
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        # 计算所有注意力头的总大小
        self.sqrt_att_head_size = math.sqrt(self.attention_head_size)
        # 计算注意力头大小的平方根

        self.query = tf.keras.layers.Dense(
            units=self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name="query"
        )
        # 创建一个全连接层用于查询
        self.key = tf.keras.layers.Dense(
            units=self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name="key"
        )
        # 创建一个全连接层用于键
        self.value = tf.keras.layers.Dense(
            units=self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name="value"
        )
        # 创建一个全连接层用于值
        self.dropout = tf.keras.layers.Dropout(rate=config.attention_probs_dropout_prob)
        # 创建一个丢弃层，用于注意力概率的丢弃
        self.config = config
        # 保存配置信息

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
        training: bool = False,
    # 定义一个方法，用于计算注意力机制的输出
    ) -> Tuple[tf.Tensor]:
        # 获取隐藏状态的批量大小
        batch_size = shape_list(hidden_states)[0]
        # 使用 query 网络对隐藏状态进行处理
        mixed_query_layer = self.query(inputs=hidden_states)
        # 使用 key 网络对隐藏状态进行处理
        mixed_key_layer = self.key(inputs=hidden_states)
        # 使用 value 网络对隐藏状态进行处理
        mixed_value_layer = self.value(inputs=hidden_states)
        # 对 query 进行转置以便计算注意力分数
        query_layer = self.transpose_for_scores(mixed_query_layer, batch_size)
        # 对 key 进行转置以便计算注意力分数
        key_layer = self.transpose_for_scores(mixed_key_layer, batch_size)
        # 对 value 进行转置以便计算注意力分数
        value_layer = self.transpose_for_scores(mixed_value_layer, batch_size)

        # 计算 "query" 和 "key" 的点积，得到原始注意力分数
        # (batch size, num_heads, seq_len_q, seq_len_k)
        attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
        # 计算缩放因子 dk
        dk = tf.cast(self.sqrt_att_head_size, dtype=attention_scores.dtype)
        attention_scores = tf.divide(attention_scores, dk)

        # 将注意力分数归一化为概率
        attention_probs = stable_softmax(logits=attention_scores, axis=-1)

        # 使用 dropout 进行注意力概率的处理
        attention_probs = self.dropout(inputs=attention_probs, training=training)

        # 如果需要，对头部进行掩码处理
        if head_mask is not None:
            attention_probs = tf.multiply(attention_probs, head_mask)

        # 计算注意力输出
        attention_output = tf.matmul(attention_probs, value_layer)
        attention_output = tf.transpose(attention_output, perm=[0, 2, 1, 3])

        # 将注意力输出进行形状调整
        # (batch_size, seq_len_q, all_head_size)
        attention_output = tf.reshape(tensor=attention_output, shape=(batch_size, -1, self.all_head_size))
        # 根据需要返回注意力输出和注意力概率
        outputs = (attention_output, attention_probs) if output_attentions else (attention_output,)

        return outputs

    # 构建注意力层
    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回
        if self.built:
            return
        self.built = True
        # 如果存在 query 层，则构建 query 层
        if getattr(self, "query", None) is not None:
            with tf.name_scope(self.query.name):
                self.query.build([None, None, self.config.hidden_size])
        # 如果存在 key 层，则构建 key 层
        if getattr(self, "key", None) is not None:
            with tf.name_scope(self.key.name):
                self.key.build([None, None, self.config.hidden_size])
        # 如果存在 value 层，则构建 value 层
        if getattr(self, "value", None) is not None:
            with tf.name_scope(self.value.name):
                self.value.build([None, None, self.config.hidden_size])
class TFViTSelfOutput(tf.keras.layers.Layer):
    """
    The residual connection is defined in TFViTLayer instead of here (as is the case with other models), due to the
    layernorm applied before each block.
    """

    def __init__(self, config: ViTConfig, **kwargs):
        super().__init__(**kwargs)

        # 定义一个全连接层，输出维度为config.hidden_size，初始化方式为config中指定的初始化方法
        self.dense = tf.keras.layers.Dense(
            units=config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )
        # 定义一个dropout层，丢弃率为config.hidden_dropout_prob
        self.dropout = tf.keras.layers.Dropout(rate=config.hidden_dropout_prob)
        self.config = config

    def call(self, hidden_states: tf.Tensor, input_tensor: tf.Tensor, training: bool = False) -> tf.Tensor:
        # 将输入通过全连接层
        hidden_states = self.dense(inputs=hidden_states)
        # 对全连接层的输出进行dropout
        hidden_states = self.dropout(inputs=hidden_states, training=training)

        return hidden_states

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                # 构建全连接层
                self.dense.build([None, None, self.config.hidden_size])


class TFViTAttention(tf.keras.layers.Layer):
    def __init__(self, config: ViTConfig, **kwargs):
        super().__init__(**kwargs)

        # 创建self attention层和输出层
        self.self_attention = TFViTSelfAttention(config, name="attention")
        self.dense_output = TFViTSelfOutput(config, name="output")

    def prune_heads(self, heads):
        raise NotImplementedError

    def call(
        self,
        input_tensor: tf.Tensor,
        head_mask: tf.Tensor,
        output_attentions: bool,
        training: bool = False,
    ) -> Tuple[tf.Tensor]:
        # 调用self attention层和输出层
        self_outputs = self.self_attention(
            hidden_states=input_tensor, head_mask=head_mask, output_attentions=output_attentions, training=training
        )
        attention_output = self.dense_output(
            hidden_states=self_outputs[0], input_tensor=input_tensor, training=training
        )
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them

        return outputs

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "self_attention", None) is not None:
            with tf.name_scope(self.self_attention.name):
                # 构建self attention层
                self.self_attention.build(None)
        if getattr(self, "dense_output", None) is not None:
            with tf.name_scope(self.dense_output.name):
                # 构建输出层
                self.dense_output.build(None)


class TFViTIntermediate(tf.keras.layers.Layer):
    # 初始化函数，接受ViTConfig类型的配置和其他关键字参数
    def __init__(self, config: ViTConfig, **kwargs):
        # 调用父类的初始化函数
        super().__init__(**kwargs)

        # 创建一个全连接层，设置单元数为config.intermediate_size，初始化方式为config.initializer_range
        self.dense = tf.keras.layers.Dense(
            units=config.intermediate_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )

        # 判断config.hidden_act是否为字符串类型，如果是则获取对应的激活函数，否则直接使用config.hidden_act
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = get_tf_activation(config.hidden_act)
        else:
            self.intermediate_act_fn = config.hidden_act
        # 保存配置信息
        self.config = config

    # 调用函数，接受tf.Tensor类型的hidden_states，返回tf.Tensor类型的hidden_states
    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        # 将hidden_states输入到全连接层中
        hidden_states = self.dense(inputs=hidden_states)
        # 使用中间激活函数对hidden_states进行激活
        hidden_states = self.intermediate_act_fn(hidden_states)

        # 返回处理后的hidden_states
        return hidden_states

    # 构建函数，接受输入形状input_shape，默认为None
    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回
        if self.built:
            return
        # 标记为已构建
        self.built = True
        # 如果存在dense层，则构建dense层，输入形状为[None, None, self.config.hidden_size]
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.hidden_size])
class TFViTOutput(tf.keras.layers.Layer):
    # 定义 TFViTOutput 类，继承自 tf.keras.layers.Layer
    def __init__(self, config: ViTConfig, **kwargs):
        # 初始化函数，接受 ViTConfig 类型的 config 参数和其他关键字参数
        super().__init__(**kwargs)

        # 创建一个全连接层，units 参数为 config.hidden_size，kernel 初始化器为 config.initializer_range，名称为 "dense"
        self.dense = tf.keras.layers.Dense(
            units=config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )
        # 创建一个 Dropout 层，丢弃率为 config.hidden_dropout_prob
        self.dropout = tf.keras.layers.Dropout(rate=config.hidden_dropout_prob)
        # 保存 config 参数
        self.config = config

    def call(self, hidden_states: tf.Tensor, input_tensor: tf.Tensor, training: bool = False) -> tf.Tensor:
        # 定义 call 方法，接受 hidden_states、input_tensor 和 training 参数，返回 tf.Tensor 类型的数据
        hidden_states = self.dense(inputs=hidden_states)  # 使用全连接层处理 hidden_states
        hidden_states = self.dropout(inputs=hidden_states, training=training)  # 使用 Dropout 处理 hidden_states
        hidden_states = hidden_states + input_tensor  # 将处理后的 hidden_states 与 input_tensor 相加

        return hidden_states  # 返回处理后的 hidden_states

    def build(self, input_shape=None):
        # 构建方法，如果已经构建过则直接返回
        if self.built:
            return
        self.built = True
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                # 使用 dense 层的名称创建一个命名空间，并构建 dense 层
                self.dense.build([None, None, self.config.intermediate_size])


class TFViTLayer(tf.keras.layers.Layer):
    # 定义 TFViTLayer 类，对应 timm 实现中的 Block 类
    def __init__(self, config: ViTConfig, **kwargs):
        # 初始化函数，接受 ViTConfig 类型的 config 参数和其他关键字参数
        super().__init__(**kwargs)

        # 创建 TFViTAttention、TFViTIntermediate 和 TFViTOutput 层
        self.attention = TFViTAttention(config, name="attention")
        self.intermediate = TFViTIntermediate(config, name="intermediate")
        self.vit_output = TFViTOutput(config, name="output")

        # 创建 LayerNormalization 层，epsilon 参数为 config.layer_norm_eps，名称为 "layernorm_before" 和 "layernorm_after"
        self.layernorm_before = tf.keras.layers.LayerNormalization(
            epsilon=config.layer_norm_eps, name="layernorm_before"
        )
        self.layernorm_after = tf.keras.layers.LayerNormalization(
            epsilon=config.layer_norm_eps, name="layernorm_after"
        )
        # 保存 config 参数
        self.config = config

    def call(
        self,
        hidden_states: tf.Tensor,
        head_mask: tf.Tensor,
        output_attentions: bool,
        training: bool = False,
    ) -> Tuple[tf.Tensor]:
        # 定义 call 方法，接受 hidden_states、head_mask、output_attentions 和 training 参数，返回 tf.Tensor 类型的元组
        attention_outputs = self.attention(
            input_tensor=self.layernorm_before(inputs=hidden_states),  # 在 self-attention 前应用 layernorm
            head_mask=head_mask,
            output_attentions=output_attentions,
            training=training,
        )
        attention_output = attention_outputs[0]

        # 第一个残差连接
        hidden_states = attention_output + hidden_states

        # 在 self-attention 后也应用 layernorm
        layer_output = self.layernorm_after(inputs=hidden_states)

        intermediate_output = self.intermediate(hidden_states=layer_output)

        # 第二个残差连接
        layer_output = self.vit_output(
            hidden_states=intermediate_output, input_tensor=hidden_states, training=training
        )
        outputs = (layer_output,) + attention_outputs[1:]  # 如果需要输出注意力，添加到 outputs 中

        return outputs  # 返回 outputs
    # 构建函数用于构建模型的层，如果已经构建过，则直接返回
    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回
        if self.built:
            return
        # 将标志位设置为已构建
        self.built = True
        # 如果存在注意力层，则构建注意力层
        if getattr(self, "attention", None) is not None:
            # 使用注意力层的名称作为命名空间
            with tf.name_scope(self.attention.name):
                # 构建注意力层
                self.attention.build(None)
        # 如果存在中间层，则构建中间层
        if getattr(self, "intermediate", None) is not None:
            # 使用中间层的名称作为命名空间
            with tf.name_scope(self.intermediate.name):
                # 构建中间层
                self.intermediate.build(None)
        # 如果存在 Vision Transformer 输出层，则构建该层
        if getattr(self, "vit_output", None) is not None:
            # 使用 Vision Transformer 输出层的名称作为命名空间
            with tf.name_scope(self.vit_output.name):
                # 构建 Vision Transformer 输出层
                self.vit_output.build(None)
        # 如果存在层归一化层（LN）在自注意力层之前，则构建该层
        if getattr(self, "layernorm_before", None) is not None:
            # 使用层归一化层（LN）在自注意力层之前的名称作为命名空间
            with tf.name_scope(self.layernorm_before.name):
                # 构建层归一化层（LN），输入形状为 [None, None, self.config.hidden_size]
                self.layernorm_before.build([None, None, self.config.hidden_size])
        # 如果存在层归一化层（LN）在自注意力层之后，则构建该层
        if getattr(self, "layernorm_after", None) is not None:
            # 使用层归一化层（LN）在自注意力层之后的名称作为命名空间
            with tf.name_scope(self.layernorm_after.name):
                # 构建层归一化层（LN），输入形状为 [None, None, self.config.hidden_size]
                self.layernorm_after.build([None, None, self.config.hidden_size])
# 定义 TFViTEncoder 类，继承自 tf.keras.layers.Layer
class TFViTEncoder(tf.keras.layers.Layer):
    # 初始化方法
    def __init__(self, config: ViTConfig, **kwargs):
        # 调用父类初始化方法
        super().__init__(**kwargs)

        # 创建一个包含 TFViTLayer 对象的列表，列表长度为 config.num_hidden_layers
        self.layer = [TFViTLayer(config, name=f"layer_._{i}") for i in range(config.num_hidden_layers)]

    # 定义 call 方法
    def call(
        self,
        hidden_states: tf.Tensor,
        head_mask: tf.Tensor,
        output_attentions: bool,
        output_hidden_states: bool,
        return_dict: bool,
        training: bool = False,
    ) -> Union[TFBaseModelOutput, Tuple[tf.Tensor]]:
        # 如果需要输出隐藏状态，则创建一个空元组 all_hidden_states
        all_hidden_states = () if output_hidden_states else None
        # 如果需要输出注意力矩阵，则创建一个空元组 all_attentions
        all_attentions = () if output_attentions else None

        # 遍历 self.layer 列表
        for i, layer_module in enumerate(self.layer):
            # 如果需要输出隐藏状态，则将当前隐藏状态添加到 all_hidden_states
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 调用 layer_module 的 call 方法，得到该层的输出
            layer_outputs = layer_module(
                hidden_states=hidden_states,
                head_mask=head_mask[i],
                output_attentions=output_attentions,
                training=training,
            )
            # 更新隐藏状态为该层的输出
            hidden_states = layer_outputs[0]

            # 如果需要输出注意力矩阵，则将当前层的注意力矩阵添加到 all_attentions
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # 如果需要输出隐藏状态，则将最后一层的隐藏状态添加到 all_hidden_states
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 如果不需要返回字典，则返回一个元组，包含隐藏状态、所有隐藏状态和所有注意力矩阵
        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_attentions] if v is not None)

        # 如果需要返回字典，则创建一个 TFBaseModelOutput 对象，包含最后的隐藏状态、所有隐藏状态和所有注意力矩阵
        return TFBaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_attentions
        )

    # 定义 build 方法
    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回
        if self.built:
            return
        # 标记已经构建
        self.built = True
        # 如果存在 self.layer 属性，则遍历该属性
        if getattr(self, "layer", None) is not None:
            for layer in self.layer:
                # 对每个层添加名称空间，并调用 build 方法
                with tf.name_scope(layer.name):
                    layer.build(None)


# 定义 TFViTMainLayer 类，继承自 tf.keras.layers.Layer
@keras_serializable
class TFViTMainLayer(tf.keras.layers.Layer):
    # 配置类
    config_class = ViTConfig

    # 初始化方法
    def __init__(self, config: ViTConfig, add_pooling_layer: bool = True, **kwargs):
        # 调用父类初始化方法
        super().__init__(**kwargs)

        # 保存配置信息
        self.config = config

        # 创建 TFViTEmbeddings 对象
        self.embeddings = TFViTEmbeddings(config, name="embeddings")
        # 创建 TFViTEncoder 对象
        self.encoder = TFViTEncoder(config, name="encoder")
        # 创建 LayerNormalization 层
        self.layernorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="layernorm")
        # 如果需要添加池化层，则创建 TFViTPooler 对象
        self.pooler = TFViTPooler(config, name="pooler") if add_pooling_layer else None

    # 获取输入嵌入
    def get_input_embeddings(self) -> tf.keras.layers.Layer:
        return self.embeddings.patch_embeddings

    # 剪枝模型的头部
    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        raise NotImplementedError

    # unpack_inputs 类方法
    @unpack_inputs
    # 定义一个方法，用于处理模型的调用，接受一系列参数并返回模型输出
    def call(
        self,
        pixel_values: TFModelInputType | None = None,  # 像素数值，可以为空
        head_mask: np.ndarray | tf.Tensor | None = None,  # 头部掩码，可以为空
        output_attentions: Optional[bool] = None,  # 是否输出注意力权重信息，可选
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态信息，可选
        interpolate_pos_encoding: Optional[bool] = None,  # 是否插值位置编码，可选
        return_dict: Optional[bool] = None,  # 是否以字典形式返回结果，可选
        training: bool = False,  # 是否处于训练模式，默认为False
    ) -> Union[TFBaseModelOutputWithPooling, Tuple[tf.Tensor]]:  # 返回值可以是指定的类型
        # 如果像素数值为空，则抛出数值错误
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # 使用嵌入层处理像素数值，根据需要插值位置编码和是否处于训练模式
        embedding_output = self.embeddings(
            pixel_values=pixel_values,
            interpolate_pos_encoding=interpolate_pos_encoding,
            training=training,
        )

        # 准备头部掩码
        # 如果头部掩码不为空，则抛出未实现错误，否则将头部掩码设为None
        if head_mask is not None:
            raise NotImplementedError
        else:
            head_mask = [None] * self.config.num_hidden_layers

        # 使用编码器处理嵌入输出，根据需要输出注意力权重和隐藏状态信息，返回值形式由return_dict决定
        encoder_outputs = self.encoder(
            hidden_states=embedding_output,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        # 获取编码器输出的序列信息，并对其进行 LayerNormalization
        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm(inputs=sequence_output)
        # 如果存在池化层，则对序列信息进行池化处理，否则将其设为None
        pooled_output = self.pooler(hidden_states=sequence_output) if self.pooler is not None else None

        # 如果不需要以字典形式返回结果，则返回指定的元组形式结果
        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        # 如果需要以字典形式返回结果，则返回指定的TFBaseModelOutputWithPooling对象
        return TFBaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

    # 定义一个方法，用于构建模型
    def build(self, input_shape=None):  # 输入形状为可选参数
        # 如果已构建过模型，则直接返回
        if self.built:
            return
        # 设置模型为已构建状态
        self.built = True
        # 如果嵌入层存在，则构建嵌入层
        if getattr(self, "embeddings", None) is not None:
            with tf.name_scope(self.embeddings.name):
                self.embeddings.build(None)
        # 如果编码器存在，则构建编码器
        if getattr(self, "encoder", None) is not None:
            with tf.name_scope(self.encoder.name):
                self.encoder.build(None)
        # 如果LayerNormalization存在，则构建LayerNormalization
        if getattr(self, "layernorm", None) is not None:
            with tf.name_scope(self.layernorm.name):
                self.layernorm.build([None, None, self.config.hidden_size])
        # 如果池化层存在，则构建池化层
        if getattr(self, "pooler", None) is not None:
            with tf.name_scope(self.pooler.name):
                self.pooler.build(None)
class TFViTPreTrainedModel(TFPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 作为 ViT 模型的配置类
    config_class = ViTConfig
    # ViT 模型的前缀
    base_model_prefix = "vit"
    # 主输入名称
    main_input_name = "pixel_values"


VIT_START_DOCSTRING = r"""

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

    - a single Tensor with `pixel_values` only and nothing else: `model(pixel_values)`
    - a list of varying length with one or several input Tensors IN THE ORDER given in the docstring:
    `model([pixel_values, attention_mask])` or `model([pixel_values, attention_mask, token_type_ids])`
    - a dictionary with one or several input Tensors associated to the input names given in the docstring:
    `model({"pixel_values": pixel_values, "token_type_ids": token_type_ids})`

    Note that when creating models and layers with
    [subclassing](https://keras.io/guides/making_new_layers_and_models_via_subclassing/) then you don't need to worry
    about any of this, as you can just pass inputs like you would to any other Python function!

    </Tip>

    Args:
        config ([`ViTConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~TFPreTrainedModel.from_pretrained`] method to load the model weights.
"""

VIT_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`np.ndarray`, `tf.Tensor`, `List[tf.Tensor]` ``Dict[str, tf.Tensor]` or `Dict[str, np.ndarray]` and each example must have the shape `(batch_size, num_channels, height, width)`):
            # 像素值。像素值可以使用 [`AutoImageProcessor`] 获得。查看 [`ViTImageProcessor.__call__`] 了解详情。

        head_mask (`np.ndarray` or `tf.Tensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            # 用于对自注意力模块中的选定头部进行屏蔽的掩码。掩码值范围在 `[0, 1]` 之间：

            - 1 表示头部**未屏蔽**，
            - 0 表示头部**已屏蔽**。

        output_attentions (`bool`, *optional*):
            # 是否返回所有注意力层的注意力张量。详细信息请参见返回的张量中的 `attentions`。此参数只能在急切模式下使用，在图模式下将使用配置中的值。

        output_hidden_states (`bool`, *optional*):
            # 是否返回所有层的隐藏状态。详细信息请参见返回的张量中的 `hidden_states`。此参数只能在急切模式下使用，在图模式下将使用配置中的值。

        interpolate_pos_encoding (`bool`, *optional*):
            # 是否对预训练的位置编码进行插值。

        return_dict (`bool`, *optional*):
            # 是否返回 [`~utils.ModelOutput`] 而不是普通元组。此参数可以在急切模式下使用，在图模式下该值将始终设置为True。

        training (`bool`, *optional*, defaults to `False``):
            # 是否以训练模式使用模型（一些模块如丢弃模块在训练和评估之间有不同的行为）。
"""
模型架构，使用ViT Model transformer进行图像分类，顶部有一个线性层（位于[CLS]令牌的最后隐藏状态之上），例如用于ImageNet。
"""


@add_start_docstrings(
    "The bare ViT Model transformer outputting raw hidden-states without any specific head on top.",
    VIT_START_DOCSTRING,
)
class TFViTModel(TFViTPreTrainedModel):
    # 初始化函数
    def __init__(self, config: ViTConfig, *inputs, add_pooling_layer=True, **kwargs):
        # 调用父类的初始化函数
        super().__init__(config, *inputs, **kwargs)

        # 创建ViT模型的主体层
        self.vit = TFViTMainLayer(config, add_pooling_layer=add_pooling_layer, name="vit")

    # 前馈函数
    @unpack_inputs
    @add_start_docstrings_to_model_forward(VIT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFBaseModelOutputWithPooling,
        config_class=_CONFIG_FOR_DOC,
        modality="vision",
        expected_output=_EXPECTED_OUTPUT_SHAPE,
    )
    def call(
        self,
        pixel_values: TFModelInputType | None = None,
        head_mask: np.ndarray | tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
    ) -> Union[TFBaseModelOutputWithPooling, Tuple[tf.Tensor]]:
        # 调用ViT模型的前馈函数
        outputs = self.vit(
            pixel_values=pixel_values,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            interpolate_pos_encoding=interpolate_pos_encoding,
            return_dict=return_dict,
            training=training,
        )

        # 返回模型的输出
        return outputs

    # 构建函数
    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回
        if self.built:
            return
        self.built = True
        if getattr(self, "vit", None) is not None:
            with tf.name_scope(self.vit.name):
                # 构建ViT模型的主体层
                self.vit.build(None)


class TFViTPooler(tf.keras.layers.Layer):
    # 初始化函数
    def __init__(self, config: ViTConfig, **kwargs):
        # 调用父类的初始化函数
        super().__init__(**kwargs)

        # 创建全连接层
        self.dense = tf.keras.layers.Dense(
            units=config.hidden_size,
            kernel_initializer=get_initializer(config.initializer_range),
            activation="tanh",
            name="dense",
        )
        self.config = config

    # 前馈函数
    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        # 通过取第一个令牌对应的隐藏状态来“池化”模型
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(inputs=first_token_tensor)

        # 返回池化的输出
        return pooled_output

    # 构建函数
    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回
        if self.built:
            return
        self.built = True
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                # 构建全连接层
                self.dense.build([None, None, self.config.hidden_size])


@add_start_docstrings(
    """
    ViT Model transformer with an image classification head on top (a linear layer on top of the final hidden state of
    the [CLS] token) e.g. for ImageNet.
    """
)
    <Tip>
    
        提醒：可以通过在模型的前向传播中将`interpolate_pos_encoding`设置为`True`，来在比模型训练分辨率更高的图像上对 ViT 进行微调。这将会对预训练的位置嵌入进行插值，适应更高的分辨率。
    
    </Tip>
    """,
    VIT_START_DOCSTRING,
# 定义 TFViTForImageClassification 类，继承自 TFViTPreTrainedModel 和 TFSequenceClassificationLoss 类
class TFViTForImageClassification(TFViTPreTrainedModel, TFSequenceClassificationLoss):
    # 初始化方法，接受 ViTConfig 实例和其他输入参数
    def __init__(self, config: ViTConfig, *inputs, **kwargs):
        # 调用父类的初始化方法
        super().__init__(config, *inputs, **kwargs)

        # 设置分类器的类别数
        self.num_labels = config.num_labels
        # 初始化 ViT 主层，不添加池化层
        self.vit = TFViTMainLayer(config, add_pooling_layer=False, name="vit")

        # 分类器头部
        self.classifier = tf.keras.layers.Dense(
            units=config.num_labels,  # 输出单元数为类别数
            kernel_initializer=get_initializer(config.initializer_range),  # 使用指定初始化器初始化权重矩阵
            name="classifier",  # 设置层名称
        )
        # 保存配置
        self.config = config

    # 调用方法，处理输入数据，返回分类器输出结果或损失
    @unpack_inputs
    @add_start_docstrings_to_model_forward(VIT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_IMAGE_CLASS_CHECKPOINT,  # 提供代码示例的检查点
        output_type=TFSequenceClassifierOutput,  # 输出类型为 TFSequenceClassifierOutput
        config_class=_CONFIG_FOR_DOC,  # 提供代码示例的配置类
        expected_output=_IMAGE_CLASS_EXPECTED_OUTPUT,  # 期望输出
    )
    def call(
        self,
        pixel_values: TFModelInputType | None = None,  # 输入像素值，可选
        head_mask: np.ndarray | tf.Tensor | None = None,  # 头部屏蔽，可选
        output_attentions: Optional[bool] = None,  # 是否输出注意力权重，可选
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态，可选
        interpolate_pos_encoding: Optional[bool] = None,  # 是否插值位置编码，可选
        return_dict: Optional[bool] = None,  # 是否返回字典格式，可选
        labels: np.ndarray | tf.Tensor | None = None,  # 标签，用于计算分类/回归损失，可选
        training: Optional[bool] = False,  # 是否处于训练模式，可选
    ) -> Union[TFSequenceClassifierOutput, Tuple[tf.Tensor]]:
        r"""
        labels (`tf.Tensor` or `np.ndarray` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """

        # 调用 ViT 主层的call方法处理输入数据
        outputs = self.vit(
            pixel_values=pixel_values,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            interpolate_pos_encoding=interpolate_pos_encoding,
            return_dict=return_dict,
            training=training,
        )
        # 获取序列输出
        sequence_output = outputs[0]
        # 将序列输出的第一个时间步的输出作为输入传递给分类器
        logits = self.classifier(inputs=sequence_output[:, 0, :])
        # 如果有标签，则计算损失
        loss = None if labels is None else self.hf_compute_loss(labels=labels, logits=logits)

        # 如果不返回字典，则以元组形式返回结果
        if not return_dict:
            output = (logits,) + outputs[2:]  # 构造输出元组
            return ((loss,) + output) if loss is not None else output  # 返回损失和输出元组或仅输出元组

        # 返回 TFSequenceClassifierOutput 对象
        return TFSequenceClassifierOutput(
            loss=loss,  # 损失
            logits=logits,  # 预测值
            hidden_states=outputs.hidden_states,  # 隐藏状态
            attentions=outputs.attentions,  # 注意力权重
        )
    # 构建整个模型
    def build(self, input_shape=None):
        # 如果模型已经构建好了，直接返回
        if self.built:
            return
        # 标记模型已经构建好了
        self.built = True
        # 如果模型包含 ViT 部分
        if getattr(self, "vit", None) is not None:
            # 使用 ViT 的名称作为 TensorFlow 命名作用域
            with tf.name_scope(self.vit.name):
                # 构建 ViT 部分
                self.vit.build(None)
        # 如果模型包含分类器部分
        if getattr(self, "classifier", None) is not None:
            # 使用分类器的名称作为 TensorFlow 命名作用域
            with tf.name_scope(self.classifier.name):
                # 构建分类器部分，输入形状为 [None, None, self.config.hidden_size]
                self.classifier.build([None, None, self.config.hidden_size])
```