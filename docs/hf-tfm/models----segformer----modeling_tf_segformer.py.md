# `.\transformers\models\segformer\modeling_tf_segformer.py`

```py
# 设置文件编码为 UTF-8
# 版权声明
# 声明代码使用 Apache 许可证 2.0
# 可以在遵守许可证的情况下使用此文件
# 详细许可证信息请查看 http://www.apache.org/licenses/LICENSE-2.0
# 在适用法律要求或书面同意的情况下，根据许可证分发的软件以"原样"的基础分发
# 没有任何担保或条件，无论是明示的还是暗示的
# 查看许可证以了解特定语言的执行情况和权利
""" TensorFlow SegFormer model."""

# 导入所需模块
from __future__ import annotations
import math
from typing import Optional, Tuple, Union
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...file_utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from ...modeling_tf_outputs import TFBaseModelOutput, TFSemanticSegmenterOutput, TFSequenceClassifierOutput
from ...modeling_tf_utils import TFPreTrainedModel, TFSequenceClassificationLoss, keras_serializable, unpack_inputs
from ...tf_utils import shape_list, stable_softmax
from ...utils import logging

from .configuration_segformer import SegformerConfig

logger = logging.get_logger(__name__)

# 用于文档中的常规描述
_CONFIG_FOR_DOC = "SegformerConfig"

# 用于文档中基本描述
_CHECKPOINT_FOR_DOC = "nvidia/mit-b0"
_EXPECTED_OUTPUT_SHAPE = [1, 256, 16, 16]

# 用于图像分类文档
_IMAGE_CLASS_CHECKPOINT = "nvidia/mit-b0"
_IMAGE_CLASS_EXPECTED_OUTPUT = "tabby, tabby cat"

TF_SEGFORMER_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "nvidia/segformer-b0-finetuned-ade-512-512",
    # 在 https://huggingface.co/models?filter=segformer 查看所有 SegFormer 模型
]

# 从 transformers.models.convnext.modeling_tf_convnext.TFConvNextDropPath 复制的 TFSegformerDropPath 类
class TFSegformerDropPath(tf.keras.layers.Layer):
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

# 构建重叠补丁嵌入的 TFSegformerOverlapPatchEmbeddings 类
class TFSegformerOverlapPatchEmbeddings(tf.keras.layers.Layer):
    """Construct the overlapping patch embeddings."""
    # 初始化方法，接收补丁大小、步长、通道数、隐藏单元数等参数
    def __init__(self, patch_size, stride, num_channels, hidden_size, **kwargs):
        # 调用父类初始化方法
        super().__init__(**kwargs)
        # 创建一个大小为 patch_size//2 的零填充层
        self.padding = tf.keras.layers.ZeroPadding2D(padding=patch_size // 2)
        # 创建一个卷积层，用于项目数据到隐藏层大小
        self.proj = tf.keras.layers.Conv2D(
            filters=hidden_size, kernel_size=patch_size, strides=stride, padding="VALID", name="proj"
        )

        # 创建一个归一化层，用于对输入数据进行归一化处理
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-05, name="layer_norm")
        # 记录通道数和隐藏单元数
        self.num_channels = num_channels
        self.hidden_size = hidden_size

    # 调用方法，接收像素值张量，返回处理后的嵌入张量、高度、宽度
    def call(self, pixel_values: tf.Tensor) -> Tuple[tf.Tensor, int, int]:
        # 将输入数据经过填充和投影两个层处理得到嵌入张量
        embeddings = self.proj(self.padding(pixel_values))
        # 获取嵌入张量的高度、宽度和隐藏维度
        height = shape_list(embeddings)[1]
        width = shape_list(embeddings)[2]
        hidden_dim = shape_list(embeddings)[3]
        # 将嵌入张量reshape成(batch_size, height*width, hidden_dim)的形状
        embeddings = tf.reshape(embeddings, (-1, height * width, hidden_dim))
        # 对嵌入张量进行归一化处理
        embeddings = self.layer_norm(embeddings)
        # 返回处理后的嵌入张量、高度、宽度
        return embeddings, height, width

    # 构建方法，用于构建卷积层和归一化层
    def build(self, input_shape=None):
        # 如果已经构建过，直接返回
        if self.built:
            return
        # 设置构建标志为 True
        self.built = True
        # 构建卷积层
        if getattr(self, "proj", None) is not None:
            with tf.name_scope(self.proj.name):
                self.proj.build([None, None, None, self.num_channels])
        # 构建归一化层
        if getattr(self, "layer_norm", None) is not None:
            with tf.name_scope(self.layer_norm.name):
                self.layer_norm.build([None, None, self.hidden_size])
class TFSegformerEfficientSelfAttention(tf.keras.layers.Layer):
    """SegFormer's efficient self-attention mechanism. Employs the sequence reduction process introduced in the [PvT
    paper](https://arxiv.org/abs/2102.12122)."""

    def __init__(
        self,
        config: SegformerConfig,  # 定义初始化函数，需传入SegformerConfig配置参数
        hidden_size: int,  # 隐藏层大小
        num_attention_heads: int,  # 注意力头的数量
        sequence_reduction_ratio: int,  # 序列压缩率
        **kwargs,  # 其他关键字参数
    ):
        super().__init__(**kwargs)  # 调用父类的初始化函数
        self.hidden_size = hidden_size  # 初始化隐藏层大小
        self.num_attention_heads = num_attention_heads  # 初始化注意力头数量

        if self.hidden_size % self.num_attention_heads != 0:  # 如果隐藏层大小不能被注意力头数量整除
            raise ValueError(  # 抛出数值错误
                f"The hidden size ({self.hidden_size}) is not a multiple of the number of attention "
                f"heads ({self.num_attention_heads})"
            )

        self.attention_head_size = self.hidden_size // self.num_attention_heads  # 计算每个注意力头的大小
        self.all_head_size = self.num_attention_heads * self.attention_head_size  # 计算所有注意力头的总大小
        self.sqrt_att_head_size = math.sqrt(self.attention_head_size)  # 计算注意力头大小的平方根

        self.query = tf.keras.layers.Dense(self.all_head_size, name="query")  # 创建密集层，用于查询
        self.key = tf.keras.layers.Dense(self.all_head_size, name="key")  # 创建密集层，用于密钥
        self.value = tf.keras.layers.Dense(self.all_head_size, name="value")  # 创建密集层，用于值

        self.dropout = tf.keras.layers.Dropout(config.attention_probs_dropout_prob)  # 创建丢弃层，用于注意力概率丢弃

        self.sr_ratio = sequence_reduction_ratio  # 初始化序列压缩率
        if sequence_reduction_ratio > 1:  # 如果序列压缩率大于1
            self.sr = tf.keras.layers.Conv2D(  # 创建二维卷积层
                filters=hidden_size, kernel_size=sequence_reduction_ratio, strides=sequence_reduction_ratio, name="sr"
            )
            self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-05, name="layer_norm")  # 创建层归一化层

    def transpose_for_scores(self, tensor: tf.Tensor) -> tf.Tensor:  # 定义转置函数，输入和输出类型均为tf.Tensor
        # Reshape from [batch_size, seq_length, all_head_size]
        # to [batch_size, seq_length, num_attention_heads, attention_head_size]
        batch_size = shape_list(tensor)[0]  # 获取张量的批量大小
        tensor = tf.reshape(tensor=tensor, shape=(batch_size, -1, self.num_attention_heads, self.attention_head_size))  # 重塑张量的形状

        # Transpose the tensor from [batch_size, seq_length, num_attention_heads, attention_head_size]
        # to [batch_size, num_attention_heads, seq_length, attention_head_size]
        return tf.transpose(tensor, perm=[0, 2, 1, 3])  # 转置张量的形状

    def call(
        self,
        hidden_states: tf.Tensor,  # 隐藏状态张量
        height: int,  # 高度
        width: int,  # 宽度
        output_attentions: bool = False,  # 输出注意力
        training: bool = False,  # 是否在训练
    ) -> Union[tf.Tensor, Tuple[tf.Tensor, tf.Tensor]]:
        # 获取隐藏状态的批量大小
        batch_size = shape_list(hidden_states)[0]
        # 获取隐藏状态的通道数
        num_channels = shape_list(hidden_states)[2]

        # 对隐藏状态进行查询操作，将结果进行转置
        query_layer = self.transpose_for_scores(self.query(hidden_states))

        if self.sr_ratio > 1:
            # 将隐藏状态重塑为(batch_size, height, width, num_channels)
            hidden_states = tf.reshape(hidden_states, (batch_size, height, width, num_channels))
            # 应用序列减少
            hidden_states = self.sr(hidden_states)
            # 将隐藏状态重新塑造为(batch_size, seq_len, num_channels)
            hidden_states = tf.reshape(hidden_states, (batch_size, -1, num_channels))
            hidden_states = self.layer_norm(hidden_states)

        # 对隐藏状态进行键值操作，将结果进行转置
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        # 计算"查询"和"键"之间的点积，得到原始注意力分数
        attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)

        scale = tf.cast(self.sqrt_att_head_size, dtype=attention_scores.dtype)
        attention_scores = tf.divide(attention_scores, scale)

        # 将注意力分数标准化为概率
        attention_probs = stable_softmax(logits=attention_scores, axis=-1)

        # 这实际上是将整个令牌丢弃以进行注意力的操作，这可能看起来有点不寻常，但来自原始Transformer论文
        attention_probs = self.dropout(attention_probs, training=training)

        # 将注意力分数与值进行乘积，得到上下文层
        context_layer = tf.matmul(attention_probs, value_layer)

        context_layer = tf.transpose(context_layer, perm=[0, 2, 1, 3])
        # (batch_size, seq_len_q, all_head_size)
        context_layer = tf.reshape(context_layer, (batch_size, -1, self.all_head_size))

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 如果已经构建了模型，则直接返回
        if getattr(self, "query", None) is not None:
            with tf.name_scope(self.query.name):
                self.query.build([None, None, self.hidden_size])
        if getattr(self, "key", None) is not None:
            with tf.name_scope(self.key.name):
                self.key.build([None, None, self.hidden_size])
        if getattr(self, "value", None) is not None:
            with tf.name_scope(self.value.name):
                self.value.build([None, None, self.hidden_size])
        if getattr(self, "sr", None) is not None:
            with tf.name_scope(self.sr.name):
                self.sr.build([None, None, None, self.hidden_size])
        if getattr(self, "layer_norm", None) is not None:
            with tf.name_scope(self.layer_norm.name):
                self.layer_norm.build([None, None, self.hidden_size])
# 定义一个用于输出的自定义层，继承自 tf.keras.layers.Layer 类
class TFSegformerSelfOutput(tf.keras.layers.Layer):
    # 初始化方法，接受 SegformerConfig 对象和隐藏层大小参数
    def __init__(self, config: SegformerConfig, hidden_size: int, **kwargs):
        super().__init__(**kwargs)
        # 创建一个全连接层，将输入的隐藏状态转换为指定大小的输出
        self.dense = tf.keras.layers.Dense(hidden_size, name="dense")
        # 创建一个 Dropout 层，用于随机失活以防止过拟合
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout_prob)
        self.hidden_size = hidden_size

    # 调用方法，对输入的隐藏状态进行处理并返回处理后的结果
    def call(self, hidden_states: tf.Tensor, training: bool = False) -> tf.Tensor:
        # 通过全连接层对隐藏状态进行转换
        hidden_states = self.dense(hidden_states)
        # 在训练时使用 Dropout 层
        hidden_states = self.dropout(hidden_states, training=training)
        # 返回处理后的结果
        return hidden_states

    # 构建方法，用于构建层的内部变量
    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回
        if self.built:
            return
        self.built = True
        # 如果存在全连接层，则构建该全连接层
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.hidden_size])


# 定义一个用于自注意力层的自定义层
class TFSegformerAttention(tf.keras.layers.Layer):
    # 初始化方法，接受 SegformerConfig 对象和其他参数
    def __init__(
        self,
        config: SegformerConfig,
        hidden_size: int,
        num_attention_heads: int,
        sequence_reduction_ratio: int,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # 创建一个自注意力层
        self.self = TFSegformerEfficientSelfAttention(
            config=config,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            sequence_reduction_ratio=sequence_reduction_ratio,
            name="self",
        )
        # 创建一个自定义输出层
        self.dense_output = TFSegformerSelfOutput(config, hidden_size=hidden_size, name="output")

    # 调用方法，对输入的隐藏状态进行处理并返回处理后的结果
    def call(
        self, hidden_states: tf.Tensor, height: int, width: int, output_attentions: bool = False
    ) -> Union[tf.Tensor, Tuple[tf.Tensor, tf.Tensor]]:
        # 调用自注意力层的 call 方法
        self_outputs = self.self(hidden_states, height, width, output_attentions)

        # 将自注意力层的输出通过自定义输出层进行处理
        attention_output = self.dense_output(self_outputs[0])
        outputs = (attention_output,) + self_outputs[1:]  # 如果需要输出注意力，将注意力也一并输出
        return outputs

    # 构建方法，用于构建层的内部变量
    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回
        if self.built:
            return
        self.built = True
        # 如果存在自注意力层，则构建该自注意力层
        if getattr(self, "self", None) is not None:
            with tf.name_scope(self.self.name):
                self.self.build(None)
        # 如果存在自定义输出层，则构建该自定义输出层
        if getattr(self, "dense_output", None) is not None:
            with tf.name_scope(self.dense_output.name):
                self.dense_output.build(None)


# 定义一个深度可分离卷积的自定义层
class TFSegformerDWConv(tf.keras.layers.Layer):
    # 初始化方法，接受隐藏状态的维度参数
    def __init__(self, dim: int = 768, **kwargs):
        super().__init__(**kwargs)
        # 创建一个深度可分离卷积层
        self.depthwise_convolution = tf.keras.layers.Conv2D(
            filters=dim, kernel_size=3, strides=1, padding="same", groups=dim, name="dwconv"
        )
        self.dim = dim
    # 定义一个方法，用于对输入的隐藏状态进行处理并返回处理后的结果
    def call(self, hidden_states: tf.Tensor, height: int, width: int) -> tf.Tensor:
        # 获取隐藏状态张量的批量大小
        batch_size = shape_list(hidden_states)[0]
        # 获取隐藏状态张量的通道数
        num_channels = shape_list(hidden_states)[-1]
        # 将隐藏状态张量重新调整形状为 (batch_size, height, width, num_channels)
        hidden_states = tf.reshape(hidden_states, (batch_size, height, width, num_channels))
        # 对重新调整形状后的隐藏状态进行深度卷积操作
        hidden_states = self.depthwise_convolution(hidden_states)

        # 获取深度卷积后隐藏状态的新高度
        new_height = shape_list(hidden_states)[1]
        # 获取深度卷积后隐藏状态的新宽度
        new_width = shape_list(hidden_states)[2]
        # 获取深度卷积后隐藏状态的通道数
        num_channels = shape_list(hidden_states)[3]
        # 将深度卷积后的隐藏状态重新调整形状为 (batch_size, new_height * new_width, num_channels)
        hidden_states = tf.reshape(hidden_states, (batch_size, new_height * new_width, num_channels))
        # 返回处理后的隐藏状态张量
        return hidden_states

    # 定义一个方法，用于构建深度卷积层
    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回
        if self.built:
            return
        # 将标志位标记为已构建
        self.built = True
        # 如果深度卷积层已经存在
        if getattr(self, "depthwise_convolution", None) is not None:
            # 在指定名称空间下构建深度卷积层
            with tf.name_scope(self.depthwise_convolution.name):
                # 构建深度卷积层，设置输入形状为 [None, None, None, self.dim]
                self.depthwise_convolution.build([None, None, None, self.dim])
class TFSegformerMixFFN(tf.keras.layers.Layer):
    def __init__(
        self,
        config: SegformerConfig,
        in_features: int,
        hidden_features: int = None,
        out_features: int = None,
        **kwargs,
    ):
        # 调用父类的初始化方法
        super().__init__(**kwargs)
        # 如果输出特征数未指定，则默认与输入特征数相同
        out_features = out_features or in_features
        # 创建一个全连接层，将输入特征转换为隐藏特征
        self.dense1 = tf.keras.layers.Dense(hidden_features, name="dense1")
        # 创建一个深度可分离卷积层
        self.depthwise_convolution = TFSegformerDWConv(hidden_features, name="dwconv")
        # 根据配置确定中间激活函数
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = get_tf_activation(config.hidden_act)
        else:
            self.intermediate_act_fn = config.hidden_act
        # 创建一个全连接层，将隐藏特征转换为输出特征
        self.dense2 = tf.keras.layers.Dense(out_features, name="dense2")
        # 创建一个 dropout 层，用于隐藏特征
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout_prob)
        # 记录隐藏特征数和输入特征数
        self.hidden_features = hidden_features
        self.in_features = in_features

    def call(self, hidden_states: tf.Tensor, height: int, width: int, training: bool = False) -> tf.Tensor:
        # 通过第一个全连接层进行特征转换
        hidden_states = self.dense1(hidden_states)
        # 通过深度可分离卷积层进行特征转换
        hidden_states = self.depthwise_convolution(hidden_states, height, width)
        # 应用中间激活函数
        hidden_states = self.intermediate_act_fn(hidden_states)
        # 在训练时使用 dropout
        hidden_states = self.dropout(hidden_states, training=training)
        # 通过第二个全连接层进行特征转换
        hidden_states = self.dense2(hidden_states)
        # 再次应用 dropout
        hidden_states = self.dropout(hidden_states, training=training)
        # 返回特征张量
        return hidden_states

    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回
        if self.built:
            return
        self.built = True
        # 构建第一个全连接层
        if getattr(self, "dense1", None) is not None:
            with tf.name_scope(self.dense1.name):
                self.dense1.build([None, None, self.in_features])
        # 构建深度可分离卷积层
        if getattr(self, "depthwise_convolution", None) is not None:
            with tf.name_scope(self.depthwise_convolution.name):
                self.depthwise_convolution.build(None)
        # 构建第二个全连接层
        if getattr(self, "dense2", None) is not None:
            with tf.name_scope(self.dense2.name):
                self.dense2.build([None, None, self.hidden_features])


class TFSegformerLayer(tf.keras.layers.Layer):
    """This corresponds to the Block class in the original implementation."""

    def __init__(
        self,
        config,
        hidden_size: int,
        num_attention_heads: int,
        drop_path: float,
        sequence_reduction_ratio: int,
        mlp_ratio: int,
        **kwargs,
    # 初始化继承自父类，传入父类初始化参数
    super().__init__(**kwargs)

    # 定义第一个 LayerNormalization 层，设置微小的 epsilon 值和名称
    self.layer_norm_1 = tf.keras.layers.LayerNormalization(epsilon=1e-05, name="layer_norm_1")

    # 定义注意力层，包含各种注意力配置参数
    self.attention = TFSegformerAttention(
        config,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        sequence_reduction_ratio=sequence_reduction_ratio,
        name="attention",
    )

    # 定义下路径损失层，如果 drop_path 参数大于 0.0 则使用 TFSegformerDropPath，否则使用线性激活层
    self.drop_path = TFSegformerDropPath(drop_path) if drop_path > 0.0 else tf.keras.layers.Activation("linear")

    # 定义第二个 LayerNormalization 层
    self.layer_norm_2 = tf.keras.layers.LayerNormalization(epsilon=1e-05, name="layer_norm_2")

    # 定义 MLP 的隐藏层大小，并创建 MLP 层
    mlp_hidden_size = int(hidden_size * mlp_ratio)
    self.mlp = TFSegformerMixFFN(config, in_features=hidden_size, hidden_features=mlp_hidden_size, name="mlp")

    # 存储隐藏层大小
    self.hidden_size = hidden_size

    # 定义模型的 call 方法，处理输入数据
    def call(
        self,
        hidden_states: tf.Tensor,
        height: int,
        width: int,
        output_attentions: bool = False,
        training: bool = False,
    ) -> Tuple:

        # 应用第一个 LayerNormalization 层，然后通过注意力层获取输出
        self_attention_outputs = self.attention(
            self.layer_norm_1(hidden_states),
            height,
            width,
            output_attentions=output_attentions,
            training=training,
        )

        # 提取注意力输出，包含主要输出和附加的自注意力信息
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # 如果输出注意力权重，则保留

        # 使用 drop_path 实现第一个残差连接（可能带有随机深度）
        attention_output = self.drop_path(attention_output, training=training)
        hidden_states = attention_output + hidden_states  # 残差加法

        # 通过 MLP 层处理并执行第二个 LayerNormalization
        mlp_output = self.mlp(self.layer_norm_2(hidden_states), height, width)

        # 使用 drop_path 实现第二个残差连接（可能带有随机深度）
        mlp_output = self.drop_path(mlp_output, training=training)
        layer_output = mlp_output + hidden_states  # 残差加法

        # 返回输出元组，包括主要输出和附加的自注意力信息
        outputs = (layer_output,) + outputs

        return outputs

    # 定义构建函数，用于在必要时初始化相关层
    def build(self, input_shape=None):
        # 检查是否已构建，如果已构建则返回
        if self.built:
            return
        # 标记已构建
        self.built = True
        
        # 构建第一个 LayerNormalization 层的相关参数和结构
        if getattr(self, "layer_norm_1", None) is not None:
            with tf.name_scope(self.layer_norm_1.name):
                self.layer_norm_1.build([None, None, self.hidden_size])

        # 构建注意力层的相关参数和结构
        if getattr(self, "attention", None) is not None:
            with tf.name_scope(self.attention.name):
                self.attention.build(None)

        # 构建第二个 LayerNormalization 层的相关参数和结构
        if getattr(self, "layer_norm_2", None) is not None:
            with tf.name_scope(self.layer_norm_2.name):
                self.layer_norm_2.build([None, None, self.hidden_size])

        # 构建 MLP 层的相关参数和结构
        if getattr(self, "mlp", None) is not None:
            with tf.name_scope(self.mlp.name):
                self.mlp.build(None)
# 定义 TFSegformerEncoder 类，继承自 tf.keras.layers.Layer
class TFSegformerEncoder(tf.keras.layers.Layer):
    # 初始化方法
    def __init__(self, config: SegformerConfig, **kwargs):
        # 调用父类的初始化方法
        super().__init__(**kwargs)
        # 将输入的 config 对象保存为实例变量
        self.config = config

        # 计算 stochastic depth 的 decay 规则
        drop_path_decays = [x.numpy() for x in tf.linspace(0.0, config.drop_path_rate, sum(config.depths))]

        # 创建 patch embeddings 列表
        embeddings = []
        for i in range(config.num_encoder_blocks):
            # 创建 TFSegformerOverlapPatchEmbeddings 对象，并添加到 embeddings 列表中
            embeddings.append(
                TFSegformerOverlapPatchEmbeddings(
                    patch_size=config.patch_sizes[i],
                    stride=config.strides[i],
                    num_channels=config.num_channels if i == 0 else config.hidden_sizes[i - 1],
                    hidden_size=config.hidden_sizes[i],
                    name=f"patch_embeddings.{i}",
                )
            )
        # 将 embeddings 列表保存为实例变量
        self.embeddings = embeddings

        # 创建 Transformer blocks 列表
        blocks = []
        cur = 0
        for i in range(config.num_encoder_blocks):
            # 每个 block 包含多个 layers
            layers = []
            if i != 0:
                cur += config.depths[i - 1]
            for j in range(config.depths[i]):
                # 创建 TFSegformerLayer 对象，并添加到 layers 列表中
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
            # 将 layers 列表添加到 blocks 列表中
            blocks.append(layers)

        # 将 blocks 列表保存为实例变量
        self.block = blocks

        # 创建 Layer norms 列表
        self.layer_norms = [
            tf.keras.layers.LayerNormalization(epsilon=1e-05, name=f"layer_norm.{i}")
            for i in range(config.num_encoder_blocks)
        ]

    # 前向传播方法
    def call(
        self,
        pixel_values: tf.Tensor,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
        training: bool = False,
    # 函数定义，接受像素值作为输入，输出模型预测的元组或模型输出
    ) -> Union[Tuple, TFBaseModelOutput]:
        # 如果不输出隐藏层状态，则初始化空元组
        all_hidden_states = () if output_hidden_states else None
        # 如果不输出注意力权重，则初始化空元组
        all_self_attentions = () if output_attentions else None

        # 获取批处理大小
        batch_size = shape_list(pixel_values)[0]

        # 初始化隐藏状态为像素值
        hidden_states = pixel_values
        # 遍历嵌入层、块、层归一化模块
        for idx, x in enumerate(zip(self.embeddings, self.block, self.layer_norms)):
            embedding_layer, block_layer, norm_layer = x
            # 调用嵌入层获取补丁嵌入和新的高度宽度
            hidden_states, height, width = embedding_layer(hidden_states)

            # 将补丁嵌入传递给块
            # （每个块由多个层组成，即图层列表）
            for i, blk in enumerate(block_layer):
                # 调用块层获取输出
                layer_outputs = blk(
                    hidden_states,
                    height,
                    width,
                    output_attentions,
                    training=training,
                )
                # 更新隐藏状态为块输出的第一个元素
                hidden_states = layer_outputs[0]
                # 如果需要输出注意力权重，则更新所有注意力权重
                if output_attentions:
                    all_self_attentions = all_self_attentions + (layer_outputs[1],)

            # 应用层归一化
            hidden_states = norm_layer(hidden_states)

            # 可选地将形状重新调整为（batch_size，height，width，num_channels）
            if idx != len(self.embeddings) - 1 or (idx == len(self.embeddings) - 1 and self.config.reshape_last_stage):
                num_channels = shape_list(hidden_states)[-1]
                hidden_states = tf.reshape(hidden_states, (batch_size, height, width, num_channels))

            # 如果需要输出隐藏状态，则更新所有隐藏状态
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

        # 如果不返回字典，则返回包含非空值的元组
        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        # 否则，返回 TFBaseModelOutput 实例
        return TFBaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_self_attentions
        )

    # 构建模型
    def build(self, input_shape=None):
        # 如果模型已构建，则直接返回
        if self.built:
            return
        self.built = True
        # 遍历层归一化模块，根据配置文件中的隐藏大小创建模块
        if getattr(self, "layer_norms", None) is not None:
            for layer, shape in zip(self.layer_norms, self.config.hidden_sizes):
                with tf.name_scope(layer.name):
                    layer.build([None, None, shape])
        # 遍历块，为每个图层构建模块
        if getattr(self, "block", None) is not None:
            for block in self.block:
                for layer in block:
                    with tf.name_scope(layer.name):
                        layer.build(None)
        # 遍历嵌入层，为每个图层构建模块
        if getattr(self, "embeddings", None) is not None:
            for layer in self.embeddings:
                with tf.name_scope(layer.name):
                    layer.build(None)
# 使用keras_serializable装饰器标记类TFSegformerMainLayer是可序列化的
@keras_serializable
class TFSegformerMainLayer(tf.keras.layers.Layer):
    # 将SegformerConfig类赋值给config_class属性
    config_class = SegformerConfig

    # 初始化方法，接受config参数和其他关键字参数
    def __init__(self, config: SegformerConfig, **kwargs):
        # 调用父类的初始化方法
        super().__init__(**kwargs)

        # 将config参数赋值给实例变量config
        self.config = config
        # 创建TFSegformerEncoder实例，并赋值给实例变量encoder
        self.encoder = TFSegformerEncoder(config, name="encoder")

    # call方法，接受多个参数并返回Union[Tuple, TFBaseModelOutput]类型的结果
    @unpack_inputs
    def call(
        self,
        pixel_values: tf.Tensor,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
    ) -> Union[Tuple, TFBaseModelOutput]:
        # 设置output_attentions, output_hidden_states, return_dict的默认值
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 将输入格式从NCHW转换为NHWC
        pixel_values = tf.transpose(pixel_values, perm=(0, 2, 3, 1))

        # 调用encoder对象的__call__方法
        encoder_outputs = self.encoder(
            pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )
        # 获取encoder输出的第一个元素，并转置为NCHW格式
        sequence_output = tf.transpose(encoder_outputs[0], perm=[0, 3, 1, 2])

        # 如果output_hidden_states为True，将encoder_outputs中的hidden_states元组中的每个元素转置为NCHW格式
        if output_hidden_states:
            hidden_states = tuple([tf.transpose(h, perm=(0, 3, 1, 2)) for h in encoder_outputs[1]])

        # 如果return_dict为False，转置encoder_outputs中隐藏状态输出的格式
        if not return_dict:
            if tf.greater(len(encoder_outputs[1:]), 0):
                transposed_encoder_outputs = tuple(tf.transpose(v, perm=[0, 3, 1, 2]) for v in encoder_outputs[1:][0])
                return (sequence_output,) + (transposed_encoder_outputs,)
            else:
                return (sequence_output,) + encoder_outputs[1:]

        # 返回TFBaseModelOutput对象，转置输出结果的格式
        return TFBaseModelOutput(
            last_hidden_state=sequence_output,
            hidden_states=hidden_states if output_hidden_states else encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

    # 构建模型，根据输入形状构建模型
    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "encoder", None) is not None:
            with tf.name_scope(self.encoder.name):
                self.encoder.build(None)


# TFSegformerPreTrainedModel类，用于处理权重初始化和预训练模型的下载和加载
class TFSegformerPreTrainedModel(TFPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    # 定义配置类为SegformerConfig
    config_class = SegformerConfig
    # 定义基础模型前缀为"segformer"
    base_model_prefix = "segformer"
    # 定义主输入名称为"pixel_values"
    main_input_name = "pixel_values"

    # 定义输入签名，包含一个像素值的张量规格
    @property
    def input_signature(self):
        return {"pixel_values": tf.TensorSpec(shape=(None, self.config.num_channels, 512, 512), dtype=tf.float32)}
# 定义 SEGFORMER_START_DOCSTRING 字符串，包含有关模型的文档说明
SEGFORMER_START_DOCSTRING = r"""
    This model inherits from [`TFPreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a [tf.keras.Model](https://www.tensorflow.org/api_docs/python/tf/keras/Model) subclass. Use it
    as a regular TF 2.0 Keras Model and refer to the TF 2.0 documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`SegformerConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~TFPreTrainedModel.from_pretrained`] method to load the model weights.
"""

# 定义 SEGFORMER_INPUTS_DOCSTRING 字符串，包含有关输入参数的文档说明
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

# 使用装饰器添加文档说明到 TFSegformerModel 类
@add_start_docstrings(
    "The bare SegFormer encoder (Mix-Transformer) outputting raw hidden-states without any specific head on top.",
    SEGFORMER_START_DOCSTRING,
)
# 定义 TFSegformerModel 类，继承自 TFSegformerPreTrainedModel
class TFSegformerModel(TFSegformerPreTrainedModel):
    # 初始化方法
    def __init__(self, config: SegformerConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.config = config

        # hierarchical Transformer encoder
        self.segformer = TFSegformerMainLayer(config, name="segformer")

    # 使用装饰器添加文档说明到模型前向传播方法
    @unpack_inputs
    @add_start_docstrings_to_model_forward(SEGFORMER_INPUTS_DOCSTRING.format("(batch_size, sequence_length)"))
    # 添加代码示例的文档字符串，包括检查点、输出类型、配置类、模态、预期输出形状等信息
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFBaseModelOutput,
        config_class=_CONFIG_FOR_DOC,
        modality="vision",
        expected_output=_EXPECTED_OUTPUT_SHAPE,
    )
    # 定义一个函数，接受像素值和一些可选参数，返回一个元组或 TFBaseModelOutput 对象
    def call(
        self,
        pixel_values: tf.Tensor,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
    ) -> Union[Tuple, TFBaseModelOutput]:
        # 调用 segformer 模型，传入像素值和一些可选参数
        outputs = self.segformer(
            pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )
        # 返回 segformer 模型的输出
        return outputs

    # 构建模型
    def build(self, input_shape=None):
        # 如果已经构建过模型，则直接返回
        if self.built:
            return
        # 标记模型已经构建
        self.built = True
        # 如果 segformer 模型存在
        if getattr(self, "segformer", None) is not None:
            # 在命名空间下构建 segformer 模型
            with tf.name_scope(self.segformer.name):
                self.segformer.build(None)
# 定义一个 SegFormer 模型转换器，顶部带有一个图像分类头部（在最终隐藏状态之上的线性层），例如用于 ImageNet
class TFSegformerForImageClassification(TFSegformerPreTrainedModel, TFSequenceClassificationLoss):
    def __init__(self, config: SegformerConfig, *inputs, **kwargs):
        # 调用父类的初始化方法
        super().__init__(config, *inputs, **kwargs)

        # 设置标签数量
        self.num_labels = config.num_labels
        # 创建 SegFormer 主层
        self.segformer = TFSegformerMainLayer(config, name="segformer")

        # 分类器头部
        self.classifier = tf.keras.layers.Dense(config.num_labels, name="classifier")
        self.config = config

    # 定义模型的前向传播方法
    @unpack_inputs
    @add_start_docstrings_to_model_forward(SEGFORMER_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
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
        # 调用 SegFormer 主层的前向传播方法
        outputs = self.segformer(
            pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        # 将最后的隐藏状态转换为 (batch_size, height*width, hidden_size)
        batch_size = shape_list(sequence_output)[0]
        sequence_output = tf.transpose(sequence_output, perm=[0, 2, 3, 1])
        sequence_output = tf.reshape(sequence_output, (batch_size, -1, self.config.hidden_sizes[-1]))

        # 全局平均池化
        sequence_output = tf.reduce_mean(sequence_output, axis=1)

        # 通过分类器获取 logits
        logits = self.classifier(sequence_output)

        # 如果没有标签，则损失为 None，否则计算损失
        loss = None if labels is None else self.hf_compute_loss(labels=labels, logits=logits)

        # 如果不返回字典，则返回输出元组
        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        # 返回 TFSequenceClassifierOutput 对象
        return TFSequenceClassifierOutput(
            loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions
        )

    # 构建模型
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


# 定义一个 SegFormer MLP 层
class TFSegformerMLP(tf.keras.layers.Layer):
    """
    Linear Embedding.
    """

    # 初始化函数，接受输入维度和配置参数
    def __init__(self, input_dim: int, config: SegformerConfig, **kwargs):
        # 调用父类的初始化函数
        super().__init__(**kwargs)
        # 创建一个全连接层，将输入维度映射到解码器隐藏层大小
        self.proj = tf.keras.layers.Dense(config.decoder_hidden_size, name="proj")
        # 保存输入维度
        self.input_dim = input_dim

    # 前向传播函数，接受隐藏状态张量并返回处理后的张量
    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        # 获取隐藏状态张量的高度、宽度和隐藏维度
        height = shape_list(hidden_states)[1]
        width = shape_list(hidden_states)[2]
        hidden_dim = shape_list(hidden_states)[-1]
        # 重塑隐藏状态张量的形状
        hidden_states = tf.reshape(hidden_states, (-1, height * width, hidden_dim))
        # 通过全连接层处理隐藏状态张量
        hidden_states = self.proj(hidden_states)
        # 返回处理后的隐藏状态张量
        return hidden_states

    # 构建函数，用于构建模型
    def build(self, input_shape=None):
        # 如果模型已经构建过，则直接返回
        if self.built:
            return
        self.built = True
        # 如果存在全连接层proj，则构建该层
        if getattr(self, "proj", None) is not None:
            with tf.name_scope(self.proj.name):
                self.proj.build([None, None, self.input_dim])
class TFSegformerDecodeHead(TFSegformerPreTrainedModel):
    # 定义一个继承自TFSegformerPreTrainedModel的类TFSegformerDecodeHead
    def __init__(self, config: SegformerConfig, **kwargs):
        # 初始化函数，接受SegformerConfig类型的config参数和其他关键字参数
        super().__init__(config, **kwargs)
        # 调用父类的初始化函数

        # 创建线性层，用于统一每个编码器块的通道维度到相同的config.decoder_hidden_size
        mlps = []
        for i in range(config.num_encoder_blocks):
            mlp = TFSegformerMLP(config=config, input_dim=config.hidden_sizes[i], name=f"linear_c.{i}")
            mlps.append(mlp)
        self.mlps = mlps

        # 实现原始实现中的ConvModule的以下3层
        self.linear_fuse = tf.keras.layers.Conv2D(
            filters=config.decoder_hidden_size, kernel_size=1, use_bias=False, name="linear_fuse"
        )
        self.batch_norm = tf.keras.layers.BatchNormalization(epsilon=1e-5, momentum=0.9, name="batch_norm")
        self.activation = tf.keras.layers.Activation("relu")

        self.dropout = tf.keras.layers.Dropout(config.classifier_dropout_prob)
        self.classifier = tf.keras.layers.Conv2D(filters=config.num_labels, kernel_size=1, name="classifier")

        self.config = config

    def call(self, encoder_hidden_states: tf.Tensor, training: bool = False) -> tf.Tensor:
        # 定义call方法，接受encoder_hidden_states和training参数，返回tf.Tensor类型的结果
        all_hidden_states = ()
        for encoder_hidden_state, mlp in zip(encoder_hidden_states, self.mlps):
            # 遍历encoder_hidden_states和mlps

            if self.config.reshape_last_stage is False and len(shape_list(encoder_hidden_state)) == 3:
                # 如果reshape_last_stage为False且encoder_hidden_state的维度为3
                height = tf.math.sqrt(tf.cast(shape_list(encoder_hidden_state)[1], tf.float32))
                height = width = tf.cast(height, tf.int32)
                channel_dim = shape_list(encoder_hidden_state)[-1]
                encoder_hidden_state = tf.reshape(encoder_hidden_state, (-1, height, width, channel_dim))

            # 统一通道维度
            encoder_hidden_state = tf.transpose(encoder_hidden_state, perm=[0, 2, 3, 1])
            height, width = shape_list(encoder_hidden_state)[1:3]
            encoder_hidden_state = mlp(encoder_hidden_state)
            channel_dim = shape_list(encoder_hidden_state)[-1]
            encoder_hidden_state = tf.reshape(encoder_hidden_state, (-1, height, width, channel_dim))

            # 上采样
            temp_state = tf.transpose(encoder_hidden_states[0], perm=[0, 2, 3, 1])
            upsample_resolution = shape_list(temp_state)[1:-1]
            encoder_hidden_state = tf.image.resize(encoder_hidden_state, size=upsample_resolution, method="bilinear")
            all_hidden_states += (encoder_hidden_state,)

        hidden_states = self.linear_fuse(tf.concat(all_hidden_states[::-1], axis=-1))
        hidden_states = self.batch_norm(hidden_states, training=training)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.dropout(hidden_states, training=training)

        # logits的形状为(batch_size, height/4, width/4, num_labels)
        logits = self.classifier(hidden_states)

        return logits
    # 构建神经网络模型
    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回
        if self.built:
            return
        # 标记为已构建
        self.built = True
        # 如果存在线性融合层，则构建线性融合层
        if getattr(self, "linear_fuse", None) is not None:
            # 使用线性融合层的名称作为命名空间
            with tf.name_scope(self.linear_fuse.name):
                # 构建线性融合层
                self.linear_fuse.build(
                    [None, None, None, self.config.decoder_hidden_size * self.config.num_encoder_blocks]
                )
        # 如果存在批量归一化层，则构建批量归一化层
        if getattr(self, "batch_norm", None) is not None:
            # 使用批量归一化层的名称作为命名空间
            with tf.name_scope(self.batch_norm.name):
                # 构建批量归一化层
                self.batch_norm.build([None, None, None, self.config.decoder_hidden_size])
        # 如果存在分类器，则构建分类器
        if getattr(self, "classifier", None) is not None:
            # 使用分类器的名称作为命名空间
            with tf.name_scope(self.classifier.name):
                # 构建分类器
                self.classifier.build([None, None, None, self.config.decoder_hidden_size])
        # 如果存在多层感知机，则逐层构建
        if getattr(self, "mlps", None) is not None:
            for layer in self.mlps:
                # 使用每一层的名称作为命名空间
                with tf.name_scope(layer.name):
                    # 构建当前层
                    layer.build(None)
# 使用装饰器添加模型文档字符串，描述 SegFormer 模型的特点和用途
# 继承自 TFSegformerPreTrainedModel 类
class TFSegformerForSemanticSegmentation(TFSegformerPreTrainedModel):
    # 初始化方法，接受 SegformerConfig 对象和其他关键字参数
    def __init__(self, config: SegformerConfig, **kwargs):
        # 调用父类的初始化方法
        super().__init__(config, **kwargs)
        # 创建 Segformer 主层对象
        self.segformer = TFSegformerMainLayer(config, name="segformer")
        # 创建 Segformer 解码头对象
        self.decode_head = TFSegformerDecodeHead(config, name="decode_head")

    # 计算损失函数的方法
    def hf_compute_loss(self, logits, labels):
        # 将 logits 上采样到原始图像大小
        # labels 的形状为 (batch_size, height, width)
        label_interp_shape = shape_list(labels)[1:]

        upsampled_logits = tf.image.resize(logits, size=label_interp_shape, method="bilinear")
        # 计算加权损失
        loss_fct = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction="none")

        def masked_loss(real, pred):
            unmasked_loss = loss_fct(real, pred)
            mask = tf.cast(real != self.config.semantic_loss_ignore_index, dtype=unmasked_loss.dtype)
            masked_loss = unmasked_loss * mask
            # 采用类似的减少策略
            reduced_masked_loss = tf.reduce_sum(masked_loss) / tf.reduce_sum(mask)
            return tf.reshape(reduced_masked_loss, (1,))

        return masked_loss(labels, upsampled_logits)

    # 使用装饰器添加模型前向传播的文档字符串
    def call(
        self,
        pixel_values: tf.Tensor,
        labels: tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        ) -> Union[Tuple, TFSemanticSegmenterOutput]:
        r"""
        labels (`tf.Tensor` of shape `(batch_size, height, width)`, *optional`):
            Ground truth semantic segmentation maps for computing the loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels > 1`, a (per-pixel) classification loss is computed
            (Cross-Entropy).

        Returns:

        Examples:

        ```python
        >>> from transformers import AutoImageProcessor, TFSegformerForSemanticSegmentation
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> image_processor = AutoImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
        >>> model = TFSegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")

        >>> inputs = image_processor(images=image, return_tensors="tf")
        >>> outputs = model(**inputs, training=False)
        >>> # logits are of shape (batch_size, num_labels, height/4, width/4)
        >>> logits = outputs.logits
        >>> list(logits.shape)
        [1, 150, 128, 128]
        ```py"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        outputs = self.segformer(
            pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=True,  # we need the intermediate hidden states
            return_dict=return_dict,
        )

        encoder_hidden_states = outputs.hidden_states if return_dict else outputs[1]

        logits = self.decode_head(encoder_hidden_states)

        loss = None
        if labels is not None:
            if not self.config.num_labels > 1:
                raise ValueError("The number of labels should be greater than one")
            else:
                loss = self.hf_compute_loss(logits=logits, labels=labels)

        # make logits of shape (batch_size, num_labels, height, width) to
        # keep them consistent across APIs
        logits = tf.transpose(logits, perm=[0, 3, 1, 2])

        if not return_dict:
            if output_hidden_states:
                output = (logits,) + outputs[1:]
            else:
                output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TFSemanticSegmenterOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=outputs.attentions,
        )
    # 构建模型，如果已经构建过则直接返回
    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回
        if self.built:
            return
        # 标记为已构建
        self.built = True
        # 如果存在 segformer 属性，则构建 segformer
        if getattr(self, "segformer", None) is not None:
            # 在命名空间下构建 segformer
            with tf.name_scope(self.segformer.name):
                self.segformer.build(None)
        # 如果存在 decode_head 属性，则构建 decode_head
        if getattr(self, "decode_head", None) is not None:
            # 在命名空间下构建 decode_head
            with tf.name_scope(self.decode_head.name):
                self.decode_head.build(None)
```