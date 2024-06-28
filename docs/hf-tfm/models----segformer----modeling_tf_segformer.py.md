# `.\models\segformer\modeling_tf_segformer.py`

```py
# 设置文件编码为 UTF-8

# 版权声明和许可信息，告知此代码的版权归属和使用许可
# 根据 Apache License, Version 2.0 许可证，除非符合许可证的规定，否则不得使用本文件
# 可以在以下网址获取许可证的副本：http://www.apache.org/licenses/LICENSE-2.0

# 引入必要的库和模块
# 引入 math 库用于数学运算
# 引入 typing 库中的一些类型注解
# 引入 TensorFlow 库
from __future__ import annotations
import math
from typing import Optional, Tuple, Union
import tensorflow as tf

# 引入其他模块和函数
# 从本地路径中引入活化函数模块
from ...activations_tf import get_tf_activation
# 从文件工具模块中引入一些函数
from ...file_utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
# 从模型输出的 TensorFlow 版本中引入输出类型
from ...modeling_tf_outputs import TFBaseModelOutput, TFSemanticSegmenterOutput, TFSequenceClassifierOutput
# 从 TensorFlow 工具模块中引入一些实用函数和类
from ...modeling_tf_utils import (
    TFPreTrainedModel,
    TFSequenceClassificationLoss,
    keras,
    keras_serializable,
    unpack_inputs,
)
# 从 TensorFlow 实用工具模块中引入形状列表和稳定 softmax 函数
from ...tf_utils import shape_list, stable_softmax
# 引入日志记录工具
from ...utils import logging
# 从 Segformer 配置模块中引入 SegformerConfig 类
from .configuration_segformer import SegformerConfig

# 获取日志记录器对象
logger = logging.get_logger(__name__)

# 用于文档的配置信息
_CONFIG_FOR_DOC = "SegformerConfig"

# 用于文档的检查点信息
_CHECKPOINT_FOR_DOC = "nvidia/mit-b0"
_EXPECTED_OUTPUT_SHAPE = [1, 256, 16, 16]

# 图像分类文档信息
_IMAGE_CLASS_CHECKPOINT = "nvidia/mit-b0"
_IMAGE_CLASS_EXPECTED_OUTPUT = "tabby, tabby cat"

# 预训练模型存档列表
TF_SEGFORMER_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "nvidia/segformer-b0-finetuned-ade-512-512",
    # 可以在 https://huggingface.co/models?filter=segformer 查看所有 SegFormer 模型
]

# 从 transformers.models.convnext.modeling_tf_convnext.TFConvNextDropPath 复制的类，现在是 SegformerDropPath
# 实现了在残差块的主路径中应用的样本级别的 Drop Path (Stochastic Depth)
# 参考来源：github.com:rwightman/pytorch-image-models
class TFSegformerDropPath(keras.layers.Layer):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    References:
        (1) github.com:rwightman/pytorch-image-models
    """

    def __init__(self, drop_path: float, **kwargs):
        super().__init__(**kwargs)
        self.drop_path = drop_path

    def call(self, x: tf.Tensor, training=None):
        if training:
            keep_prob = 1 - self.drop_path
            shape = (tf.shape(x)[0],) + (1,) * (len(tf.shape(x)) - 1)
            random_tensor = keep_prob + tf.random.uniform(shape, 0, 1)
            random_tensor = tf.floor(random_tensor)
            return (x / keep_prob) * random_tensor
        return x

# 构建重叠的补丁嵌入层
class TFSegformerOverlapPatchEmbeddings(keras.layers.Layer):
    """Construct the overlapping patch embeddings."""
    # 初始化方法，设置类的初始参数及网络层结构
    def __init__(self, patch_size, stride, num_channels, hidden_size, **kwargs):
        # 调用父类的初始化方法
        super().__init__(**kwargs)
        # 创建一个 ZeroPadding2D 层，用于填充输入的像素值
        self.padding = keras.layers.ZeroPadding2D(padding=patch_size // 2)
        # 创建一个 Conv2D 层，用于进行卷积操作，生成特征映射
        self.proj = keras.layers.Conv2D(
            filters=hidden_size, kernel_size=patch_size, strides=stride, padding="VALID", name="proj"
        )

        # 创建一个 LayerNormalization 层，用于归一化特征映射
        self.layer_norm = keras.layers.LayerNormalization(epsilon=1e-05, name="layer_norm")
        # 设置输入通道数和隐藏单元数
        self.num_channels = num_channels
        self.hidden_size = hidden_size

    # 定义调用方法，实现数据流向及数据处理
    def call(self, pixel_values: tf.Tensor) -> Tuple[tf.Tensor, int, int]:
        # 对输入像素值进行填充和卷积操作，生成特征映射
        embeddings = self.proj(self.padding(pixel_values))
        # 获取特征映射的高度、宽度和深度信息
        height = shape_list(embeddings)[1]
        width = shape_list(embeddings)[2]
        hidden_dim = shape_list(embeddings)[3]
        # 重新调整特征映射的形状，将其转换为(batch_size, height*width, hidden_dim)的形式
        embeddings = tf.reshape(embeddings, (-1, height * width, hidden_dim))
        # 对重新调整的特征映射进行层归一化处理
        embeddings = self.layer_norm(embeddings)
        # 返回归一化后的特征映射、高度和宽度信息
        return embeddings, height, width

    # 构建方法，用于构建层次结构及初始化层内部的权重
    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回
        if self.built:
            return
        # 标记该层已经构建
        self.built = True
        # 如果 proj 层存在，则构建 proj 层
        if getattr(self, "proj", None) is not None:
            with tf.name_scope(self.proj.name):
                self.proj.build([None, None, None, self.num_channels])
        # 如果 layer_norm 层存在，则构建 layer_norm 层
        if getattr(self, "layer_norm", None) is not None:
            with tf.name_scope(self.layer_norm.name):
                self.layer_norm.build([None, None, self.hidden_size])
        height: int,
        width: int,
        output_attentions: bool = False,
        training: bool = False,


        # 定义 call 方法用于执行层的前向传播
        hidden_states: tf.Tensor,
        # height 和 width 表示输入张量的高度和宽度
        height: int,
        width: int,
        # output_attentions 表示是否输出注意力权重，默认为 False
        output_attentions: bool = False,
        # training 表示是否处于训练模式，默认为 False
        training: bool = False,
    ) -> Union[tf.Tensor, Tuple[tf.Tensor, tf.Tensor]]:
        # 获取隐藏状态的批量大小
        batch_size = shape_list(hidden_states)[0]
        # 获取隐藏状态的通道数
        num_channels = shape_list(hidden_states)[2]

        # 使用 self.query 对隐藏状态进行查询操作，并调整维度以进行注意力计算
        query_layer = self.transpose_for_scores(self.query(hidden_states))

        if self.sr_ratio > 1:
            # 将隐藏状态重塑为 (batch_size, height, width, num_channels) 的形状
            hidden_states = tf.reshape(hidden_states, (batch_size, height, width, num_channels))
            # 应用序列减少操作
            hidden_states = self.sr(hidden_states)
            # 将隐藏状态重新调整为 (batch_size, seq_len, num_channels) 的形状
            hidden_states = tf.reshape(hidden_states, (batch_size, -1, num_channels))
            # 对调整后的隐藏状态进行层归一化
            hidden_states = self.layer_norm(hidden_states)

        # 使用 self.key 对隐藏状态进行查询操作，并调整维度以进行注意力计算
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        # 使用 self.value 对隐藏状态进行查询操作，并调整维度以进行注意力计算
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        # 计算 "query" 和 "key" 的点积，得到原始的注意力分数
        attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)

        # 缩放注意力分数
        scale = tf.cast(self.sqrt_att_head_size, dtype=attention_scores.dtype)
        attention_scores = tf.divide(attention_scores, scale)

        # 将注意力分数归一化为概率
        attention_probs = stable_softmax(logits=attention_scores, axis=-1)

        # 对注意力概率进行 dropout 操作，训练时使用
        attention_probs = self.dropout(attention_probs, training=training)

        # 计算加权后的 value 层作为上下文层
        context_layer = tf.matmul(attention_probs, value_layer)

        # 调整上下文层的维度顺序
        context_layer = tf.transpose(context_layer, perm=[0, 2, 1, 3])
        # 将上下文层重塑为 (batch_size, seq_len_q, all_head_size) 的形状
        context_layer = tf.reshape(context_layer, (batch_size, -1, self.all_head_size))

        # 如果需要输出注意力分数，则将其包含在输出中
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs

    def build(self, input_shape=None):
        # 如果模型已经构建，直接返回
        if self.built:
            return
        self.built = True
        # 如果存在 self.query 属性，则构建 self.query 层
        if getattr(self, "query", None) is not None:
            with tf.name_scope(self.query.name):
                self.query.build([None, None, self.hidden_size])
        # 如果存在 self.key 属性，则构建 self.key 层
        if getattr(self, "key", None) is not None:
            with tf.name_scope(self.key.name):
                self.key.build([None, None, self.hidden_size])
        # 如果存在 self.value 属性，则构建 self.value 层
        if getattr(self, "value", None) is not None:
            with tf.name_scope(self.value.name):
                self.value.build([None, None, self.hidden_size])
        # 如果存在 self.sr 属性，则构建 self.sr 层
        if getattr(self, "sr", None) is not None:
            with tf.name_scope(self.sr.name):
                self.sr.build([None, None, None, self.hidden_size])
        # 如果存在 self.layer_norm 属性，则构建 self.layer_norm 层
        if getattr(self, "layer_norm", None) is not None:
            with tf.name_scope(self.layer_norm.name):
                self.layer_norm.build([None, None, self.hidden_size])
# Segformer 模型的自定义输出层，用于处理隐藏状态
class TFSegformerSelfOutput(keras.layers.Layer):
    def __init__(self, config: SegformerConfig, hidden_size: int, **kwargs):
        super().__init__(**kwargs)
        # 创建一个全连接层，将输入转换为指定的隐藏大小
        self.dense = keras.layers.Dense(hidden_size, name="dense")
        # Dropout 层，用于在训练过程中随机丢弃部分神经元，以防止过拟合
        self.dropout = keras.layers.Dropout(config.hidden_dropout_prob)
        self.hidden_size = hidden_size

    def call(self, hidden_states: tf.Tensor, training: bool = False) -> tf.Tensor:
        # 将隐藏状态传递给全连接层进行线性变换
        hidden_states = self.dense(hidden_states)
        # 如果在训练模式下，对全连接层的输出进行 Dropout 处理
        hidden_states = self.dropout(hidden_states, training=training)
        return hidden_states

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 如果层已经构建，则直接返回；否则，构建全连接层
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.hidden_size])


# Segformer 模型的注意力层，包含自注意力机制和输出处理
class TFSegformerAttention(keras.layers.Layer):
    def __init__(
        self,
        config: SegformerConfig,
        hidden_size: int,
        num_attention_heads: int,
        sequence_reduction_ratio: int,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # 自注意力机制，用于处理输入的隐藏状态
        self.self = TFSegformerEfficientSelfAttention(
            config=config,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            sequence_reduction_ratio=sequence_reduction_ratio,
            name="self",
        )
        # 输出处理层，负责处理自注意力机制的输出隐藏状态
        self.dense_output = TFSegformerSelfOutput(config, hidden_size=hidden_size, name="output")

    def call(
        self, hidden_states: tf.Tensor, height: int, width: int, output_attentions: bool = False
    ) -> Union[tf.Tensor, Tuple[tf.Tensor, tf.Tensor]]:
        # 使用自注意力机制处理隐藏状态
        self_outputs = self.self(hidden_states, height, width, output_attentions)

        # 将自注意力机制的输出传递给输出处理层进行处理
        attention_output = self.dense_output(self_outputs[0])
        outputs = (attention_output,) + self_outputs[1:]  # 如果需要输出注意力权重，则添加到输出中
        return outputs

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 如果层已经构建，则直接返回；否则，构建自注意力和输出处理层
        if getattr(self, "self", None) is not None:
            with tf.name_scope(self.self.name):
                self.self.build(None)
        if getattr(self, "dense_output", None) is not None:
            with tf.name_scope(self.dense_output.name):
                self.dense_output.build(None)


# Segformer 模型的深度可分离卷积层，用于特征提取
class TFSegformerDWConv(keras.layers.Layer):
    def __init__(self, dim: int = 768, **kwargs):
        super().__init__(**kwargs)
        # 深度可分离卷积层，用于对输入进行特征提取
        self.depthwise_convolution = keras.layers.Conv2D(
            filters=dim, kernel_size=3, strides=1, padding="same", groups=dim, name="dwconv"
        )
        self.dim = dim
    # 定义一个方法 `call`，接受三个参数：隐藏状态（张量）、高度和宽度，并返回一个张量
    def call(self, hidden_states: tf.Tensor, height: int, width: int) -> tf.Tensor:
        # 获取隐藏状态的批量大小
        batch_size = shape_list(hidden_states)[0]
        # 获取隐藏状态的通道数
        num_channels = shape_list(hidden_states)[-1]
        # 将隐藏状态重塑为四维张量，形状为(batch_size, height, width, num_channels)
        hidden_states = tf.reshape(hidden_states, (batch_size, height, width, num_channels))
        # 对重塑后的隐藏状态进行深度可分离卷积操作
        hidden_states = self.depthwise_convolution(hidden_states)

        # 获取卷积后张量的新高度、宽度和通道数
        new_height = shape_list(hidden_states)[1]
        new_width = shape_list(hidden_states)[2]
        num_channels = shape_list(hidden_states)[3]
        # 将卷积后的张量再次重塑为三维张量，形状为(batch_size, new_height * new_width, num_channels)
        hidden_states = tf.reshape(hidden_states, (batch_size, new_height * new_width, num_channels))
        # 返回处理后的张量
        return hidden_states

    # 定义一个方法 `build`，用于构建模型层次结构
    def build(self, input_shape=None):
        # 如果模型已经构建，则直接返回
        if self.built:
            return
        self.built = True
        # 检查是否存在深度可分离卷积层，如果有，则构建该层
        if getattr(self, "depthwise_convolution", None) is not None:
            with tf.name_scope(self.depthwise_convolution.name):
                # 调用深度可分离卷积层的构建方法，并传入期望的输入形状
                self.depthwise_convolution.build([None, None, None, self.dim])
class TFSegformerMixFFN(keras.layers.Layer):
    # Segformer 模型中的混合 FeedForward Network 层
    def __init__(
        self,
        config: SegformerConfig,
        in_features: int,
        hidden_features: int = None,
        out_features: int = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        out_features = out_features or in_features
        # 第一个全连接层，输入特征数为 hidden_features，输出特征数为 hidden_features
        self.dense1 = keras.layers.Dense(hidden_features, name="dense1")
        # 深度可分离卷积层，处理 hidden_features 维度的数据
        self.depthwise_convolution = TFSegformerDWConv(hidden_features, name="dwconv")
        # 中间激活函数，根据配置选择
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = get_tf_activation(config.hidden_act)
        else:
            self.intermediate_act_fn = config.hidden_act
        # 第二个全连接层，输入特征数为 hidden_features，输出特征数为 out_features
        self.dense2 = keras.layers.Dense(out_features, name="dense2")
        # Dropout 层，根据配置的概率进行 dropout
        self.dropout = keras.layers.Dropout(config.hidden_dropout_prob)
        self.hidden_features = hidden_features
        self.in_features = in_features

    def call(self, hidden_states: tf.Tensor, height: int, width: int, training: bool = False) -> tf.Tensor:
        # 前向传播过程
        # 全连接层 1，将 hidden_states 映射到 hidden_features 维度
        hidden_states = self.dense1(hidden_states)
        # 深度可分离卷积层，处理 hidden_features 维度的数据，输入图片的高度和宽度
        hidden_states = self.depthwise_convolution(hidden_states, height, width)
        # 中间激活函数，对 hidden_states 应用激活函数
        hidden_states = self.intermediate_act_fn(hidden_states)
        # Dropout 层，对 hidden_states 进行随机失活
        hidden_states = self.dropout(hidden_states, training=training)
        # 全连接层 2，将 hidden_states 映射回 out_features 维度
        hidden_states = self.dense2(hidden_states)
        # 再次应用 Dropout 层
        hidden_states = self.dropout(hidden_states, training=training)
        return hidden_states

    def build(self, input_shape=None):
        # 构建层的方法
        if self.built:
            return
        self.built = True
        # 如果 dense1 层已经定义，则构建 dense1 层
        if getattr(self, "dense1", None) is not None:
            with tf.name_scope(self.dense1.name):
                self.dense1.build([None, None, self.in_features])
        # 如果 depthwise_convolution 层已经定义，则构建 depthwise_convolution 层
        if getattr(self, "depthwise_convolution", None) is not None:
            with tf.name_scope(self.depthwise_convolution.name):
                self.depthwise_convolution.build(None)
        # 如果 dense2 层已经定义，则构建 dense2 层
        if getattr(self, "dense2", None) is not None:
            with tf.name_scope(self.dense2.name):
                self.dense2.build([None, None, self.hidden_features])


class TFSegformerLayer(keras.layers.Layer):
    """This corresponds to the Block class in the original implementation."""
    # Segformer 模型中的一个层，对应原始实现中的 Block 类

    def __init__(
        self,
        config,
        hidden_size: int,
        num_attention_heads: int,
        drop_path: float,
        sequence_reduction_ratio: int,
        mlp_ratio: int,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.layer_norm_1 = keras.layers.LayerNormalization(epsilon=1e-05, name="layer_norm_1")
        self.attention = TFSegformerAttention(
            config,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            sequence_reduction_ratio=sequence_reduction_ratio,
            name="attention",
        )
        self.drop_path = TFSegformerDropPath(drop_path) if drop_path > 0.0 else keras.layers.Activation("linear")
        self.layer_norm_2 = keras.layers.LayerNormalization(epsilon=1e-05, name="layer_norm_2")
        mlp_hidden_size = int(hidden_size * mlp_ratio)
        self.mlp = TFSegformerMixFFN(config, in_features=hidden_size, hidden_features=mlp_hidden_size, name="mlp")
        self.hidden_size = hidden_size


# 初始化方法，用于创建一个新的 Segformer 层
def __init__(
    self,
    config,
    hidden_size: int,
    num_attention_heads: int,
    sequence_reduction_ratio: int,
    mlp_ratio: float,
    drop_path: float,
    **kwargs
):
    # 调用父类的初始化方法
    super().__init__(**kwargs)
    # 第一个 LayerNormalization 层，用于在自注意力机制之前进行层归一化
    self.layer_norm_1 = keras.layers.LayerNormalization(epsilon=1e-05, name="layer_norm_1")
    # Segformer 的注意力机制层
    self.attention = TFSegformerAttention(
        config,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        sequence_reduction_ratio=sequence_reduction_ratio,
        name="attention",
    )
    # DropPath 层，根据概率 drop_path 进行路径下降或线性激活
    self.drop_path = TFSegformerDropPath(drop_path) if drop_path > 0.0 else keras.layers.Activation("linear")
    # 第二个 LayerNormalization 层，用于在 MLP 之前进行层归一化
    self.layer_norm_2 = keras.layers.LayerNormalization(epsilon=1e-05, name="layer_norm_2")
    # MLP 层，用于非线性变换
    mlp_hidden_size = int(hidden_size * mlp_ratio)
    self.mlp = TFSegformerMixFFN(config, in_features=hidden_size, hidden_features=mlp_hidden_size, name="mlp")
    # 隐藏大小
    self.hidden_size = hidden_size



    def call(
        self,
        hidden_states: tf.Tensor,
        height: int,
        width: int,
        output_attentions: bool = False,
        training: bool = False,
    ) -> Tuple:
        # 使用自注意力机制层处理输入的隐藏状态并返回输出
        self_attention_outputs = self.attention(
            self.layer_norm_1(hidden_states),  # in Segformer, layernorm is applied before self-attention
            height,
            width,
            output_attentions=output_attentions,
            training=training,
        )

        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        # 第一个残差连接（带有随机深度）
        attention_output = self.drop_path(attention_output, training=training)
        hidden_states = attention_output + hidden_states
        # 使用 MLP 层处理归一化后的隐藏状态并返回输出
        mlp_output = self.mlp(self.layer_norm_2(hidden_states), height, width)

        # 第二个残差连接（带有随机深度）
        mlp_output = self.drop_path(mlp_output, training=training)
        layer_output = mlp_output + hidden_states

        outputs = (layer_output,) + outputs

        return outputs


    def build(self, input_shape=None):
        # 如果已经构建过网络层，则直接返回
        if self.built:
            return
        self.built = True
        # 构建第一个层归一化层
        if getattr(self, "layer_norm_1", None) is not None:
            with tf.name_scope(self.layer_norm_1.name):
                self.layer_norm_1.build([None, None, self.hidden_size])
        # 构建注意力机制层
        if getattr(self, "attention", None) is not None:
            with tf.name_scope(self.attention.name):
                self.attention.build(None)
        # 构建第二个层归一化层
        if getattr(self, "layer_norm_2", None) is not None:
            with tf.name_scope(self.layer_norm_2.name):
                self.layer_norm_2.build([None, None, self.hidden_size])
        # 构建 MLP 层
        if getattr(self, "mlp", None) is not None:
            with tf.name_scope(self.mlp.name):
                self.mlp.build(None)
# 定义一个自定义层 TFSegformerEncoder，继承自 keras.layers.Layer 类
class TFSegformerEncoder(keras.layers.Layer):
    
    # 初始化方法，接受一个 SegformerConfig 对象作为参数，并调用父类的初始化方法
    def __init__(self, config: SegformerConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config

        # 计算使用 stochastic depth 策略的衰减率列表
        drop_path_decays = [x.numpy() for x in tf.linspace(0.0, config.drop_path_rate, sum(config.depths))]

        # 创建 patch embeddings 列表
        embeddings = []
        # 根据 num_encoder_blocks 的数量循环创建 TFSegformerOverlapPatchEmbeddings 对象
        for i in range(config.num_encoder_blocks):
            embeddings.append(
                TFSegformerOverlapPatchEmbeddings(
                    patch_size=config.patch_sizes[i],
                    stride=config.strides[i],
                    num_channels=config.num_channels if i == 0 else config.hidden_sizes[i - 1],
                    hidden_size=config.hidden_sizes[i],
                    name=f"patch_embeddings.{i}",
                )
            )
        self.embeddings = embeddings

        # 创建 Transformer blocks 列表
        blocks = []
        cur = 0
        # 根据 num_encoder_blocks 的数量循环创建 Transformer blocks
        for i in range(config.num_encoder_blocks):
            # 每个 block 包含多个 layers
            layers = []
            if i != 0:
                cur += config.depths[i - 1]
            # 根据 depths[i] 的数量循环创建 TFSegformerLayer 对象
            for j in range(config.depths[i]):
                layers.append(
                    TFSegformerLayer(
                        config,
                        hidden_size=config.hidden_sizes[i],
                        num_attention_heads=config.num_attention_heads[i],
                        drop_path=drop_path_decays[cur + j],
                        sequence_reduction_ratio=config.sr_ratios[i],
                        mlp_ratio=config.mlp_ratios[i],
                        name=f"block.{i}.{j}",
                    )
                )
            blocks.append(layers)

        self.block = blocks

        # 创建 Layer norms 列表
        self.layer_norms = [
            keras.layers.LayerNormalization(epsilon=1e-05, name=f"layer_norm.{i}")
            for i in range(config.num_encoder_blocks)
        ]

    # 定义 call 方法，实现层的调用逻辑
    def call(
        self,
        pixel_values: tf.Tensor,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
        training: bool = False,
        # 参数说明：输入的像素值张量，是否输出注意力权重，是否输出隐藏状态，是否返回字典形式的输出，训练模式标志位
    ) -> Union[Tuple, TFBaseModelOutput]:
        # 如果输出隐藏状态为真，则初始化空元组，否则为 None
        all_hidden_states = () if output_hidden_states else None
        # 如果输出注意力权重为真，则初始化空元组，否则为 None
        all_self_attentions = () if output_attentions else None

        # 获取批量大小
        batch_size = shape_list(pixel_values)[0]

        # 初始隐藏状态为输入像素值
        hidden_states = pixel_values
        # 遍历嵌入层、块、层归一化器的组合
        for idx, x in enumerate(zip(self.embeddings, self.block, self.layer_norms)):
            embedding_layer, block_layer, norm_layer = x
            # 第一步，获取图像块的嵌入表示
            hidden_states, height, width = embedding_layer(hidden_states)

            # 第二步，将嵌入表示通过块处理
            # （每个块包含多个层，即层的列表）
            for i, blk in enumerate(block_layer):
                # 调用块的前向传播
                layer_outputs = blk(
                    hidden_states,
                    height,
                    width,
                    output_attentions,
                    training=training,
                )
                hidden_states = layer_outputs[0]
                # 如果需要输出注意力权重，则将它们累加到 all_self_attentions 中
                if output_attentions:
                    all_self_attentions = all_self_attentions + (layer_outputs[1],)

            # 第三步，应用层归一化
            hidden_states = norm_layer(hidden_states)

            # 第四步，可选地将隐藏状态重塑为 (batch_size, height, width, num_channels)
            if idx != len(self.embeddings) - 1 or (idx == len(self.embeddings) - 1 and self.config.reshape_last_stage):
                num_channels = shape_list(hidden_states)[-1]
                hidden_states = tf.reshape(hidden_states, (batch_size, height, width, num_channels))

            # 如果需要输出隐藏状态，则将当前隐藏状态累加到 all_hidden_states 中
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

        # 如果不需要返回字典格式的结果，则返回非 None 的元组
        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        # 否则，返回 TFBaseModelOutput 类型的结果
        return TFBaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_self_attentions
        )

    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回
        if self.built:
            return
        self.built = True
        # 如果存在层归一化器，则对每个层归一化器进行构建
        if getattr(self, "layer_norms", None) is not None:
            for layer, shape in zip(self.layer_norms, self.config.hidden_sizes):
                with tf.name_scope(layer.name):
                    layer.build([None, None, shape])
        # 如果存在块，则对每个块中的每个层进行构建
        if getattr(self, "block", None) is not None:
            for block in self.block:
                for layer in block:
                    with tf.name_scope(layer.name):
                        layer.build(None)
        # 如果存在嵌入层，则对每个嵌入层进行构建
        if getattr(self, "embeddings", None) is not None:
            for layer in self.embeddings:
                with tf.name_scope(layer.name):
                    layer.build(None)
@keras_serializable
class TFSegformerMainLayer(keras.layers.Layer):
    config_class = SegformerConfig

    def __init__(self, config: SegformerConfig, **kwargs):
        super().__init__(**kwargs)

        self.config = config
        # hierarchical Transformer encoder
        self.encoder = TFSegformerEncoder(config, name="encoder")

    @unpack_inputs
    def call(
        self,
        pixel_values: tf.Tensor,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
    ) -> Union[Tuple, TFBaseModelOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # When running on CPU, `keras.layers.Conv2D` doesn't support `NCHW` format.
        # So change the input format from `NCHW` to `NHWC`.
        # shape = (batch_size, in_height, in_width, in_channels=num_channels)
        pixel_values = tf.transpose(pixel_values, perm=(0, 2, 3, 1))

        encoder_outputs = self.encoder(
            pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )
        sequence_output = encoder_outputs[0]
        # Change to NCHW output format to have uniformity in the modules
        sequence_output = tf.transpose(sequence_output, perm=[0, 3, 1, 2])

        # Change the other hidden state outputs to NCHW as well
        if output_hidden_states:
            hidden_states = tuple([tf.transpose(h, perm=(0, 3, 1, 2)) for h in encoder_outputs[1]])

        if not return_dict:
            if tf.greater(len(encoder_outputs[1:]), 0):
                transposed_encoder_outputs = tuple(tf.transpose(v, perm=[0, 3, 1, 2]) for v in encoder_outputs[1:][0])
                return (sequence_output,) + (transposed_encoder_outputs,)
            else:
                return (sequence_output,) + encoder_outputs[1:]

        return TFBaseModelOutput(
            last_hidden_state=sequence_output,
            hidden_states=hidden_states if output_hidden_states else encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "encoder", None) is not None:
            with tf.name_scope(self.encoder.name):
                self.encoder.build(None)


class TFSegformerPreTrainedModel(TFPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """


注释：

# 定义一个可序列化的 Keras 层 `TFSegformerMainLayer`，用于分割器模型
@keras_serializable
class TFSegformerMainLayer(keras.layers.Layer):
    # 配置类为 SegformerConfig
    config_class = SegformerConfig

    def __init__(self, config: SegformerConfig, **kwargs):
        super().__init__(**kwargs)

        self.config = config
        # hierarchical Transformer encoder
        # 使用给定的配置参数创建名为 "encoder" 的 Segformer 编码器
        self.encoder = TFSegformerEncoder(config, name="encoder")

    @unpack_inputs
    def call(
        self,
        pixel_values: tf.Tensor,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
    ) -> Union[Tuple, TFBaseModelOutput]:
        # 设置输出注意力，默认为配置中的设置
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 设置输出隐藏状态，默认为配置中的设置
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 设置是否返回字典，默认为配置中的设置
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 当在 CPU 上运行时，`keras.layers.Conv2D` 不支持 `NCHW` 格式。
        # 因此将输入格式从 `NCHW` 转换为 `NHWC`。
        # 形状为 (batch_size, in_height, in_width, in_channels=num_channels)
        pixel_values = tf.transpose(pixel_values, perm=(0, 2, 3, 1))

        # 使用编码器处理输入数据
        encoder_outputs = self.encoder(
            pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )
        # 获取序列输出（通常是最后一个隐藏状态的输出）
        sequence_output = encoder_outputs[0]
        # 将输出格式转换为 `NCHW`，以保持模块的一致性
        sequence_output = tf.transpose(sequence_output, perm=[0, 3, 1, 2])

        # 如果需要输出隐藏状态，则将它们也转换为 `NCHW` 格式
        if output_hidden_states:
            hidden_states = tuple([tf.transpose(h, perm=(0, 3, 1, 2)) for h in encoder_outputs[1]])

        # 如果不需要返回字典，则返回调整后的编码器输出
        if not return_dict:
            if tf.greater(len(encoder_outputs[1:]), 0):
                # 转换所有其他隐藏状态输出为 `NCHW` 格式
                transposed_encoder_outputs = tuple(tf.transpose(v, perm=[0, 3, 1, 2]) for v in encoder_outputs[1:][0])
                return (sequence_output,) + (transposed_encoder_outputs,)
            else:
                return (sequence_output,) + encoder_outputs[1:]

        # 否则返回 TFBaseModelOutput 类的实例，其中包含最后隐藏状态、隐藏状态和注意力权重
        return TFBaseModelOutput(
            last_hidden_state=sequence_output,
            hidden_states=hidden_states if output_hidden_states else encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

    def build(self, input_shape=None):
        # 如果已经构建，则直接返回
        if self.built:
            return
        # 设置已构建标志
        self.built = True
        # 如果存在编码器，则在其命名范围内构建编码器
        if getattr(self, "encoder", None) is not None:
            with tf.name_scope(self.encoder.name):
                self.encoder.build(None)

# 定义一个基于 TFPreTrainedModel 的抽象类 TFSegformerPreTrainedModel，
# 用于处理权重初始化以及下载和加载预训练模型的简单接口
class TFSegformerPreTrainedModel(TFPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    # 定义一个变量，指定配置类为SegformerConfig
    config_class = SegformerConfig
    # 定义一个字符串变量，作为基础模型前缀，设为"segformer"
    base_model_prefix = "segformer"
    # 定义一个字符串变量，表示主要输入的名称，设为"pixel_values"
    main_input_name = "pixel_values"
    
    # 定义一个属性方法，用于返回输入的签名信息
    @property
    def input_signature(self):
        # 返回一个字典，键为"pixel_values"，值为一个 TensorFlow 张量规格，
        # 其形状为(None, self.config.num_channels, 512, 512)，数据类型为 tf.float32
        return {"pixel_values": tf.TensorSpec(shape=(None, self.config.num_channels, 512, 512), dtype=tf.float32)}
"""
定义了 SEGFORMER_START_DOCSTRING，包含了关于模型继承和参数配置的详细描述文档。
"""
SEGFORMER_START_DOCSTRING = r"""
    This model inherits from [`TFPreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a [keras.Model](https://www.tensorflow.org/api_docs/python/tf/keras/Model) subclass. Use it
    as a regular TF 2.0 Keras Model and refer to the TF 2.0 documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`SegformerConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~TFPreTrainedModel.from_pretrained`] method to load the model weights.
"""

"""
定义了 SEGFORMER_INPUTS_DOCSTRING，包含了模型输入参数的详细描述文档。
"""
SEGFORMER_INPUTS_DOCSTRING = r"""

    Args:
        pixel_values (`np.ndarray`, `tf.Tensor`, `List[tf.Tensor]` ``Dict[str, tf.Tensor]` or `Dict[str, np.ndarray]` and each example must have the shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See
            [`SegformerImageProcessor.__call__`] for details.

        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail. This argument can be used only in eager mode, in graph mode the value in the
            config will be used instead.

        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail. This argument can be used only in eager mode, in graph mode the value in the config will be
            used instead.

        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple. This argument can be used in
            eager mode, in graph mode the value will always be set to True.

        training (`bool`, *optional*, defaults to `False``):
            Whether or not to use the model in training mode (some modules like dropout modules have different
            behaviors between training and evaluation).
"""

"""
添加了模型描述的文档字符串，并调用了 `add_start_docstrings` 装饰器，将模型简介和参数文档串联起来。
"""
@add_start_docstrings(
    "The bare SegFormer encoder (Mix-Transformer) outputting raw hidden-states without any specific head on top.",
    SEGFORMER_START_DOCSTRING,
)
class TFSegformerModel(TFSegformerPreTrainedModel):
    """
    TFSegformerModel 类继承自 TFSegformerPreTrainedModel，表示一个基础的 SegFormer 编码器（混合Transformer），
    输出未经特定顶部处理的原始隐藏状态。

    Args:
        config (SegformerConfig): 包含模型所有参数的配置类。使用配置文件初始化时，不会加载与模型关联的权重，只会加载配置。
            查看 `~TFPreTrainedModel.from_pretrained` 方法以加载模型权重。
    """
    def __init__(self, config: SegformerConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.config = config

        # hierarchical Transformer encoder
        # 层次化Transformer编码器
        self.segformer = TFSegformerMainLayer(config, name="segformer")

    """
    添加了文档字符串到模型的前向传播方法，描述了输入参数的详细信息。
    """
    @unpack_inputs
    @add_start_docstrings_to_model_forward(SEGFORMER_INPUTS_DOCSTRING.format("(batch_size, sequence_length)"))
    # 使用装饰器为此方法添加文档字符串，指定了一些参数和预期输出类型等信息
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFBaseModelOutput,
        config_class=_CONFIG_FOR_DOC,
        modality="vision",
        expected_output=_EXPECTED_OUTPUT_SHAPE,
    )
    # 定义一个方法 `call`，接受多个参数并返回 `TFBaseModelOutput` 类型或其元组
    def call(
        self,
        pixel_values: tf.Tensor,  # 输入像素值的张量
        output_attentions: Optional[bool] = None,  # 控制是否输出注意力权重的布尔值
        output_hidden_states: Optional[bool] = None,  # 控制是否输出隐藏状态的布尔值
        return_dict: Optional[bool] = None,  # 控制是否以字典形式返回输出的布尔值
        training: bool = False,  # 控制是否处于训练模式的布尔值，默认为 False
    ) -> Union[Tuple, TFBaseModelOutput]:  # 返回类型可以是元组或 `TFBaseModelOutput` 类型

        # 调用 `self.segformer` 方法，传递相应参数，并接收输出结果
        outputs = self.segformer(
            pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )
        
        # 返回 `segformer` 方法的输出结果
        return outputs

    # 定义 `build` 方法
    def build(self, input_shape=None):
        # 如果已经构建过，直接返回，避免重复构建
        if self.built:
            return
        
        # 将标志 `self.built` 设置为 True，表示已经构建
        self.built = True
        
        # 如果存在 `self.segformer` 属性，则在名称作用域内构建 `segformer`
        if getattr(self, "segformer", None) is not None:
            with tf.name_scope(self.segformer.name):
                self.segformer.build(None)
"""
SegFormer Model transformer with an image classification head on top (a linear layer on top of the final hidden
states) e.g. for ImageNet.
"""
# 使用 SegFormer 模型进行图像分类，顶部是一个线性层，放置在最终隐藏状态之上，例如用于 ImageNet 数据集。

class TFSegformerForImageClassification(TFSegformerPreTrainedModel, TFSequenceClassificationLoss):
    def __init__(self, config: SegformerConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        self.num_labels = config.num_labels
        # 初始化 SegFormer 主体模型层
        self.segformer = TFSegformerMainLayer(config, name="segformer")

        # 分类器头部
        self.classifier = keras.layers.Dense(config.num_labels, name="classifier")
        self.config = config

    @unpack_inputs
    # 将起始文档字符串添加到模型的前向传播方法上，描述输入的格式
    @add_start_docstrings_to_model_forward(SEGFORMER_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    # 添加代码示例的文档字符串
    @add_code_sample_docstrings(
        checkpoint=_IMAGE_CLASS_CHECKPOINT,
        output_type=TFSequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_IMAGE_CLASS_EXPECTED_OUTPUT,
    )
    def call(
        self,
        pixel_values: tf.Tensor | None = None,
        labels: tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, TFSequenceClassifierOutput]:
        outputs = self.segformer(
            pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        # 将最后的隐藏状态转换为 (batch_size, height*width, hidden_size) 的形状
        batch_size = shape_list(sequence_output)[0]
        sequence_output = tf.transpose(sequence_output, perm=[0, 2, 3, 1])
        sequence_output = tf.reshape(sequence_output, (batch_size, -1, self.config.hidden_sizes[-1]))

        # 全局平均池化
        sequence_output = tf.reduce_mean(sequence_output, axis=1)

        logits = self.classifier(sequence_output)

        # 如果没有标签，损失为 None；否则计算损失
        loss = None if labels is None else self.hf_compute_loss(labels=labels, logits=logits)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return TFSequenceClassifierOutput(
            loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions
        )

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "segformer", None) is not None:
            with tf.name_scope(self.segformer.name):
                self.segformer.build(None)
        if getattr(self, "classifier", None) is not None:
            with tf.name_scope(self.classifier.name):
                self.classifier.build([None, None, self.config.hidden_sizes[-1]])
    """
    
    def __init__(self, input_dim: int, config: SegformerConfig, **kwargs):
        super().__init__(**kwargs)  # 调用父类的构造函数，传递任何额外的关键字参数
        self.proj = keras.layers.Dense(config.decoder_hidden_size, name="proj")  # 初始化一个全连接层 Dense 对象，用于投影
        self.input_dim = input_dim  # 设置输入维度

    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        height = shape_list(hidden_states)[1]  # 获取 hidden_states 张量的高度维度
        width = shape_list(hidden_states)[2]   # 获取 hidden_states 张量的宽度维度
        hidden_dim = shape_list(hidden_states)[-1]  # 获取 hidden_states 张量的最后一个维度（隐藏维度）
        hidden_states = tf.reshape(hidden_states, (-1, height * width, hidden_dim))  # 对 hidden_states 张量进行重新形状操作
        hidden_states = self.proj(hidden_states)  # 应用定义的投影层到 hidden_states 张量上
        return hidden_states  # 返回变换后的 hidden_states 张量

    def build(self, input_shape=None):
        if self.built:  # 如果模型已经构建过，则直接返回
            return
        self.built = True  # 将模型标记为已构建
        if getattr(self, "proj", None) is not None:  # 如果投影层存在
            with tf.name_scope(self.proj.name):  # 使用投影层的名字作为命名空间
                self.proj.build([None, None, self.input_dim])  # 构建投影层，指定输入维度
# 定义一个继承自TFSegformerPreTrainedModel的解码头类，用于Segformer模型
class TFSegformerDecodeHead(TFSegformerPreTrainedModel):
    # 初始化方法，接受一个SegformerConfig对象和其他关键字参数
    def __init__(self, config: SegformerConfig, **kwargs):
        # 调用父类的初始化方法
        super().__init__(config, **kwargs)
        
        # 初始化一个空列表，用于存储多个MLP模块
        mlps = []
        # 根据配置中的encoder块数量迭代创建MLP模块
        for i in range(config.num_encoder_blocks):
            # 创建一个TFSegformerMLP对象，设置输入维度和名称
            mlp = TFSegformerMLP(config=config, input_dim=config.hidden_sizes[i], name=f"linear_c.{i}")
            # 将创建的MLP模块添加到mlps列表中
            mlps.append(mlp)
        # 将创建的MLP模块列表赋值给当前对象的mlps属性
        self.mlps = mlps

        # 创建线性融合层，实现原始实现中的ConvModule
        self.linear_fuse = keras.layers.Conv2D(
            filters=config.decoder_hidden_size, kernel_size=1, use_bias=False, name="linear_fuse"
        )
        # 创建批标准化层
        self.batch_norm = keras.layers.BatchNormalization(epsilon=1e-5, momentum=0.9, name="batch_norm")
        # 创建激活函数层，使用ReLU激活函数
        self.activation = keras.layers.Activation("relu")

        # 创建dropout层，使用配置中的分类器dropout概率
        self.dropout = keras.layers.Dropout(config.classifier_dropout_prob)
        # 创建分类器层，输出通道数为配置中的标签数量
        self.classifier = keras.layers.Conv2D(filters=config.num_labels, kernel_size=1, name="classifier")

        # 将配置对象保存到当前对象的config属性中
        self.config = config

    # 定义call方法，接受encoder_hidden_states和training两个参数，返回一个Tensor对象
    def call(self, encoder_hidden_states: tf.Tensor, training: bool = False) -> tf.Tensor:
        # 初始化一个空元组，用于存储所有隐藏状态
        all_hidden_states = ()
        # 迭代encoder_hidden_states和mlps列表
        for encoder_hidden_state, mlp in zip(encoder_hidden_states, self.mlps):
            # 如果reshape_last_stage为False且encoder_hidden_state的形状长度为3
            if self.config.reshape_last_stage is False and len(shape_list(encoder_hidden_state)) == 3:
                # 计算height和width，并将encoder_hidden_state重塑为四维张量
                height = tf.math.sqrt(tf.cast(shape_list(encoder_hidden_state)[1], tf.float32))
                height = width = tf.cast(height, tf.int32)
                channel_dim = shape_list(encoder_hidden_state)[-1]
                encoder_hidden_state = tf.reshape(encoder_hidden_state, (-1, height, width, channel_dim))

            # 将encoder_hidden_state的通道维度移动到最后一个维度
            encoder_hidden_state = tf.transpose(encoder_hidden_state, perm=[0, 2, 3, 1])
            # 获取当前encoder_hidden_state的height和width
            height, width = shape_list(encoder_hidden_state)[1:3]
            # 将encoder_hidden_state传入mlp模块中进行处理
            encoder_hidden_state = mlp(encoder_hidden_state)
            # 获取处理后的encoder_hidden_state的通道维度
            channel_dim = shape_list(encoder_hidden_state)[-1]
            # 将encoder_hidden_state重塑为四维张量
            encoder_hidden_state = tf.reshape(encoder_hidden_state, (-1, height, width, channel_dim))

            # 上采样
            temp_state = tf.transpose(encoder_hidden_states[0], perm=[0, 2, 3, 1])
            upsample_resolution = shape_list(temp_state)[1:-1]
            encoder_hidden_state = tf.image.resize(encoder_hidden_state, size=upsample_resolution, method="bilinear")
            # 将处理后的encoder_hidden_state添加到all_hidden_states元组中
            all_hidden_states += (encoder_hidden_state,)

        # 对所有隐藏状态进行拼接并通过线性融合层处理
        hidden_states = self.linear_fuse(tf.concat(all_hidden_states[::-1], axis=-1))
        # 对处理后的隐藏状态进行批标准化
        hidden_states = self.batch_norm(hidden_states, training=training)
        # 对批标准化后的隐藏状态进行ReLU激活
        hidden_states = self.activation(hidden_states)
        # 对激活后的隐藏状态进行dropout
        hidden_states = self.dropout(hidden_states, training=training)

        # 计算分类器的logits，形状为(batch_size, height/4, width/4, num_labels)
        logits = self.classifier(hidden_states)

        # 返回logits
        return logits
    # 定义模型的构建方法，用于初始化模型结构
    def build(self, input_shape=None):
        # 如果模型已经构建过，则直接返回，避免重复构建
        if self.built:
            return
        # 标记模型已经构建
        self.built = True
        
        # 如果存在线性融合层（linear_fuse），则构建该层
        if getattr(self, "linear_fuse", None) is not None:
            # 使用线性融合层的名称作为命名空间
            with tf.name_scope(self.linear_fuse.name):
                # 使用给定形状构建线性融合层，形状包括 None 表示任意大小
                self.linear_fuse.build(
                    [None, None, None, self.config.decoder_hidden_size * self.config.num_encoder_blocks]
                )
        
        # 如果存在批量归一化层（batch_norm），则构建该层
        if getattr(self, "batch_norm", None) is not None:
            # 使用批量归一化层的名称作为命名空间
            with tf.name_scope(self.batch_norm.name):
                # 使用给定形状构建批量归一化层，形状中的 None 表示任意大小
                self.batch_norm.build([None, None, None, self.config.decoder_hidden_size])
        
        # 如果存在分类器（classifier），则构建该层
        if getattr(self, "classifier", None) is not None:
            # 使用分类器的名称作为命名空间
            with tf.name_scope(self.classifier.name):
                # 使用给定形状构建分类器，形状中的 None 表示任意大小
                self.classifier.build([None, None, None, self.config.decoder_hidden_size])
        
        # 如果存在多层感知机（mlps），则逐层构建每个多层感知机层
        if getattr(self, "mlps", None) is not None:
            for layer in self.mlps:
                # 使用每层多层感知机层的名称作为命名空间
                with tf.name_scope(layer.name):
                    # 每层多层感知机层不需要特定的输入形状，因此传入 None
                    layer.build(None)
# 使用特定的文档字符串初始化一个 SegFormer 模型，该模型在顶部具有全MLP解码头，例如用于 ADE20k、CityScapes 数据集。
# 继承自 TFSegformerPreTrainedModel 类
class TFSegformerForSemanticSegmentation(TFSegformerPreTrainedModel):
    def __init__(self, config: SegformerConfig, **kwargs):
        # 调用父类的初始化方法
        super().__init__(config, **kwargs)
        # 初始化 SegFormer 主层，并命名为 "segformer"
        self.segformer = TFSegformerMainLayer(config, name="segformer")
        # 初始化 SegFormer 解码头，并命名为 "decode_head"
        self.decode_head = TFSegformerDecodeHead(config, name="decode_head")

    def hf_compute_loss(self, logits, labels):
        # 将 logits 插值（上采样）到原始图像尺寸
        # `labels` 的形状为 (batch_size, height, width)
        label_interp_shape = shape_list(labels)[1:]

        upsampled_logits = tf.image.resize(logits, size=label_interp_shape, method="bilinear")
        # 定义加权损失函数
        loss_fct = keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction="none")

        def masked_loss(real, pred):
            # 计算未屏蔽的损失
            unmasked_loss = loss_fct(real, pred)
            # 创建掩码，排除标签为 self.config.semantic_loss_ignore_index 的位置
            mask = tf.cast(real != self.config.semantic_loss_ignore_index, dtype=unmasked_loss.dtype)
            masked_loss = unmasked_loss * mask
            # 通过加权损失计算减少的掩码损失
            reduced_masked_loss = tf.reduce_sum(masked_loss) / tf.reduce_sum(mask)
            return tf.reshape(reduced_masked_loss, (1,))

        return masked_loss(labels, upsampled_logits)

    @unpack_inputs
    # 添加前向模型调用的文档字符串，使用 SEGFORMER_INPUTS_DOCSTRING 模板，指定输入的格式为 "batch_size, sequence_length"
    @add_start_docstrings_to_model_forward(SEGFORMER_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    # 替换返回值文档字符串，指定输出类型为 TFSemanticSegmenterOutput，使用 _CONFIG_FOR_DOC 类配置
    @replace_return_docstrings(output_type=TFSemanticSegmenterOutput, config_class=_CONFIG_FOR_DOC)
    def call(
        self,
        pixel_values: tf.Tensor,
        labels: tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 如果 return_dict 不为 None，则使用其当前值；否则使用 self.config.use_return_dict 的值

        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 如果 output_hidden_states 不为 None，则使用其当前值；否则使用 self.config.output_hidden_states 的值

        outputs = self.segformer(
            pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=True,  # we need the intermediate hidden states
            return_dict=return_dict,
        )
        # 调用 self.segformer 进行语义分割模型的前向传播计算，传入像素值 pixel_values，
        # 设置 output_attentions 和 return_dict 参数，确保返回中间隐藏状态

        encoder_hidden_states = outputs.hidden_states if return_dict else outputs[1]
        # 如果 return_dict 为 True，则使用 outputs 的 hidden_states；否则使用 outputs 的第二个元素作为编码器的隐藏状态

        logits = self.decode_head(encoder_hidden_states)
        # 根据编码器的隐藏状态计算预测 logits

        loss = None
        if labels is not None:
            if not self.config.num_labels > 1:
                raise ValueError("The number of labels should be greater than one")
            else:
                loss = self.hf_compute_loss(logits=logits, labels=labels)
        # 如果 labels 不为 None，则计算损失值，确保标签数量大于1，否则抛出 ValueError

        # 调整 logits 的形状为 (batch_size, num_labels, height, width)，以保持 API 一致性
        logits = tf.transpose(logits, perm=[0, 3, 1, 2])

        if not return_dict:
            if output_hidden_states:
                output = (logits,) + outputs[1:]
            else:
                output = (logits,) + outputs[2:]
            # 如果 return_dict 为 False，根据 output_hidden_states 的值选择返回 logits 和隐藏状态列表或注意力列表
            return ((loss,) + output) if loss is not None else output
        # 返回包含损失值和输出内容的元组，如果损失值为 None，则返回输出内容

        return TFSemanticSegmenterOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=outputs.attentions,
        )
        # 如果 return_dict 为 True，则返回 TFSemanticSegmenterOutput 对象，包含损失值、logits、隐藏状态和注意力信息
    # 定义模型构建方法，接受输入形状参数，默认为None
    def build(self, input_shape=None):
        # 如果模型已经构建过，则直接返回，不再重复构建
        if self.built:
            return
        # 设置模型已构建标志为True
        self.built = True
        
        # 检查是否存在名为"segformer"的属性，并且该属性不为None
        if getattr(self, "segformer", None) is not None:
            # 在TensorFlow的命名作用域内，使用self.segformer.name作为作用域名
            with tf.name_scope(self.segformer.name):
                # 调用self.segformer对象的build方法，传入None作为输入形状参数
                self.segformer.build(None)
        
        # 检查是否存在名为"decode_head"的属性，并且该属性不为None
        if getattr(self, "decode_head", None) is not None:
            # 在TensorFlow的命名作用域内，使用self.decode_head.name作为作用域名
            with tf.name_scope(self.decode_head.name):
                # 调用self.decode_head对象的build方法，传入None作为输入形状参数
                self.decode_head.build(None)
```