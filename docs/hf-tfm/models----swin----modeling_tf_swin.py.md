# `.\transformers\models\swin\modeling_tf_swin.py`

```
# 设置文件编码格式为 UTF-8

# 版权声明，版权归 Microsoft Research 和 The HuggingFace Inc. 团队所有
# 根据 Apache 许可证 2.0 进行许可
# 除非符合许可证的规定，否则不得使用该文件
# 可以在以下网址获取许可证的副本
# http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或经书面同意，否则按“原样”基础分发软件，无论是明示的还是暗示的
# 请参阅许可证以获取特定语言的权限和限制

""" TF 2.0 Swin Transformer model."""  # TF 2.0 Swin Transformer 模型的说明文档

from __future__ import annotations  # 引入 Python 3.x 特性，以便在注释类型中使用 forward declarations

# 引入所需的库和模块
import collections.abc
import math
import warnings
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
# 引入 TensorFlow 库
import tensorflow as tf

# 导入 HuggingFace 库中的部分模块和工具函数
from ...activations_tf import ACT2FN
from ...modeling_tf_utils import (
    TFPreTrainedModel,
    TFSequenceClassificationLoss,
    get_initializer,
    keras_serializable,
    unpack_inputs,
)
from ...tf_utils import shape_list
from ...utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from .configuration_swin import SwinConfig  # 导入 SwinConfig 配置模块

logger = logging.get_logger(__name__)  # 获取 logger 对象，用于记录日志信息

# General docstring
_CONFIG_FOR_DOC = "SwinConfig"  # 一般文档说明的配置

# Base docstring
_CHECKPOINT_FOR_DOC = "microsoft/swin-tiny-patch4-window7-224"  # 基本文档说明的检查点信息
_EXPECTED_OUTPUT_SHAPE = [1, 49, 768]  # 期望的输出形状为 [1, 49, 768]

# Image classification docstring
_IMAGE_CLASS_CHECKPOINT = "microsoft/swin-tiny-patch4-window7-224"  # 图像分类文档的检查点信息
_IMAGE_CLASS_EXPECTED_OUTPUT = "tabby, tabby cat"  # 期望的图像分类输出

# Swin 模型的预训练模型列表
TF_SWIN_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "microsoft/swin-tiny-patch4-window7-224",
    # 查看所有 Swin 模型，请访问 https://huggingface.co/models?filter=swin
]

# drop_path, TFSwinPatchEmbeddings, TFSwinPatchMerging and TFSwinDropPath are tensorflow
# implementations of PyTorch functionalities in the timm library.

# TFSwinEncoderOutput 类，表示 Swin 编码器的输出，包括潜在的隐藏状态和注意力信息
@dataclass
class TFSwinEncoderOutput(ModelOutput):
    """
    Swin encoder's outputs, with potential hidden states and attentions.
    Args:
        last_hidden_state (`tf.Tensor` of shape `(batch_size, sequence_length, hidden_size)`):
            模型最后一层的输出的隐藏状态序列。
        hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            元组类型的张量，形状为 `(batch_size, sequence_length, hidden_size)`。

            模型在每个层的输出隐藏状态，以及初始嵌入的输出。
        attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            元组类型的张量，形状为 `(batch_size, num_heads, sequence_length, sequence_length)`。

            注意力机制 softmax 后的注意力权重，用于在自注意力头中计算加权平均。
        reshaped_hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            元组类型的张量，形状为 `(batch_size, hidden_size, height, width)`。

            模型在每个层的输出隐藏状态，以及初始嵌入的输出，重塑为包含空间维度的形状。
    """

    last_hidden_state: tf.Tensor = None  # 最后一层的隐藏状态，默认为 None
    hidden_states: Tuple[tf.Tensor] | None = None  # 每个层的隐藏状态，默认为 None
    attentions: Tuple[tf.Tensor] | None = None  # 注意力权重，默认为 None
    reshaped_hidden_states: Tuple[tf.Tensor] | None = None  # 重塑后的隐藏状态，默认为 None
# 使用 dataclass 装饰器声明 TFSwinModelOutput 类，该类继承自 ModelOutput 类
@dataclass
class TFSwinModelOutput(ModelOutput):
    """
    Swin model's outputs that also contains a pooling of the last hidden states.

    Args:
        last_hidden_state (`tf.Tensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        pooler_output (`tf.Tensor` of shape `(batch_size, hidden_size)`, *optional*, returned when `add_pooling_layer=True` is passed):
            Average pooling of the last layer hidden-state.
        hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `tf.Tensor` (one for the output of the embeddings + one for the output of each stage) of shape
            `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `tf.Tensor` (one for each stage) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        reshaped_hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `tf.Tensor` (one for the output of the embeddings + one for the output of each stage) of shape
            `(batch_size, hidden_size, height, width)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs reshaped to
            include the spatial dimensions.
    """

    # 定义类属性，包括最后一个隐藏状态、pooler 输出、隐藏状态、注意力、重塑后的隐藏状态
    last_hidden_state: tf.Tensor = None
    pooler_output: tf.Tensor | None = None
    hidden_states: Tuple[tf.Tensor] | None = None
    attentions: Tuple[tf.Tensor] | None = None
    reshaped_hidden_states: Tuple[tf.Tensor] | None = None


@dataclass
class TFSwinMaskedImageModelingOutput(ModelOutput):
    """
    Swin masked image model outputs.
    # 定义函数的参数列表
    Args:
        loss (`tf.Tensor` of shape `(1,)`, *optional*, returned when `bool_masked_pos` is provided):
            Masked image modeling (MLM) loss. 代表有损失函数
        reconstruction (`tf.Tensor` of shape `(batch_size, num_channels, height, width)`):
            Reconstructed pixel values. 代表重构的像素值
        hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `tf.Tensor` (one for the output of the embeddings + one for the output of each stage) of shape
            `(batch_size, sequence_length, hidden_size)`. 代表模型隐藏状态
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `tf.Tensor` (one for each stage) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads. 代表注意力权重
        reshaped_hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `tf.Tensor` (one for the output of the embeddings + one for the output of each stage) of shape
            `(batch_size, hidden_size, height, width)`. 代表重塑后的隐藏状态
            Hidden-states of the model at the output of each layer plus the initial embedding outputs reshaped to
            include the spatial dimensions.
    
    # 声明变量并赋初始值
    loss: tf.Tensor | None = None 代表有损失的张量
    reconstruction: tf.Tensor = None 代表重构的张量
    hidden_states: Tuple[tf.Tensor] | None = None 代表隐藏状态
    attentions: Tuple[tf.Tensor] | None = None 代表注意力
    reshaped_hidden_states: Tuple[tf.Tensor] | None = None 代表重塑后的隐藏状态
    
    # 定义属性方法logits
    @property
    def logits(self): 代表属性方法
        warnings.warn(
            "logits attribute is deprecated and will be removed in version 5 of Transformers."
            " Please use the reconstruction attribute to retrieve the final output instead.",
            FutureWarning,
        )
        return self.reconstruction 代表返回重构属性值
# 定义了一个继承自ModelOutput的TFSwinImageClassifierOutput类
# 这个类用于表示Swin模型在图像分类任务中的输出结果
@dataclass
class TFSwinImageClassifierOutput(ModelOutput):
    """
    Swin outputs for image classification.

    Args:
        loss (`tf.Tensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            分类（或回归，如果config.num_labels==1）损失值。
        logits (`tf.Tensor` of shape `(batch_size, config.num_labels)`):
            分类（或回归，如果config.num_labels==1）的得分（在SoftMax之前）。
        hidden_states (`tuple(tf.Tensor)`, *可选*, 当`output_hidden_states=True`或`config.output_hidden_states=True`时返回):
            `(batch_size, sequence_length, hidden_size)`形状的`tf.Tensor`类型的元组，

            每一层模型的隐藏状态，以及初始嵌入输出。
        attentions (`tuple(tf.Tensor)`, *可选*, 当`output_attentions=True`或`config.output_attentions=True`时返回):
            `(batch_size, num_heads, sequence_length, sequence_length)`形状的`tf.Tensor`类型的元组。

            注意力权重，在自注意力头部中用于计算加权平均值。
        reshaped_hidden_states (`tuple(tf.Tensor)`, *可选*, 当`output_hidden_states=True`或`config.output_hidden_states=True`时返回):
            `(batch_size, hidden_size, height, width)`形状的`tf.Tensor`类型的元组。

            每一层模型的隐藏状态以及初始嵌入输出，经过调整形状以包含空间维度。
    """

    loss: tf.Tensor | None = None
    logits: tf.Tensor = None
    hidden_states: Tuple[tf.Tensor] | None = None
    attentions: Tuple[tf.Tensor] | None = None
    reshaped_hidden_states: Tuple[tf.Tensor] | None = None


# 将输入特征拆分成窗口的函数
def window_partition(input_feature: tf.Tensor, window_size: int) -> tf.Tensor:
    """
    Partitions the given input into windows.
    """

    # 获取输入特征的批量尺寸、高度、宽度以及通道数目
    batch_size, height, width, num_channels = shape_list(input_feature)

    # 将输入特征进行重新整形，将其划分为窗口
    # 划分后的形状为(batch_size, height // window_size, window_size, width // window_size, window_size, num_channels)
    input_feature = tf.reshape(
        input_feature,
        (batch_size, height // window_size, window_size, width // window_size, window_size, num_channels),
    )

    # 对划分后的窗口进行转置，以便更好地处理特征图
    # 转置后的形状为(batch_size, height // window_size, width // window_size, window_size, window_size, num_channels)
    windows = tf.transpose(input_feature, (0, 1, 3, 2, 4, 5))

    # 将转置后的窗口展平，以便后续处理
    # 展平后的形状为(-1, window_size, window_size, num_channels)
    windows = tf.reshape(windows, (-1, window_size, window_size, num_channels))

    # 返回划分后的窗口
    return windows


# 将窗口合并成更高分辨率的特征的函数
def window_reverse(windows: tf.Tensor, window_size: int, height: int, width: int) -> tf.Tensor:
    """
    Merges windows to produce higher resolution features.
    """

    # 获取输入窗口的形状
    x = tf.shape(windows)[0]

    # 计算目标特征图的长度
    y = tf.cast(height * width / (window_size * window_size), tf.int32)

    # 计算批量尺寸
    batch_size = tf.math.floordiv(x, y)

    # ...部分缺失，无法完整解释其作用
    # 将输入的窗口数据重新整形为指定形状
    windows = tf.reshape(
        windows, (batch_size, height // window_size, width // window_size, window_size, window_size, -1)
    )
    # 转置窗口数据，调整维度排列顺序
    windows = tf.transpose(windows, (0, 1, 3, 2, 4, 5))
    # 再次重新整形窗口数据为指定形状
    windows = tf.reshape(windows, (batch_size, height, width, -1))
    # 返回调整后的窗口数据
    return windows
def drop_path(
    input: tf.Tensor, drop_prob: float = 0.0, training: bool = False, scale_by_keep: bool = True
) -> tf.Tensor:
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    """
    # 如果 dropout 概率为 0 或者不处于训练状态，则直接返回输入张量
    if drop_prob == 0.0 or not training:
        return input
    # 计算保留概率
    keep_prob = 1 - drop_prob
    # 获取输入张量的形状信息
    input_shape = shape_list(input)
    # 获取输入张量的维度数
    ndim = len(input_shape)
    # 构建随机张量的形状，保证适用于不同维度的张量，而不仅仅是二维卷积网络
    shape = [input_shape[0]] + [1] * (ndim - 1)
    # 生成服从均匀分布的随机张量
    random_tensor = tf.random.uniform(shape)
    # 将随机张量中小于等于保留概率的元素置为 1，大于保留概率的元素置为 0
    random_tensor = tf.where(random_tensor <= keep_prob, 1.0, 0.0)
    # 如果保留概率大于 0 且设置了按保留概率缩放，则对随机张量进行缩放
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor /= keep_prob
    # 返回输入张量与随机张量的乘积
    return input * random_tensor


class TFSwinEmbeddings(tf.keras.layers.Layer):
    """
    Construct the patch and position embeddings. Optionally, also the mask token.
    """

    def __init__(self, config: SwinConfig, use_mask_token: bool = False, **kwargs) -> None:
        super().__init__(**kwargs)
        # 初始化 SwinTransformer 模型的补丁嵌入层对象
        self.patch_embeddings = TFSwinPatchEmbeddings(config, name="patch_embeddings")
        # 获取补丁数量
        self.num_patches = self.patch_embeddings.num_patches
        # 获取补丁网格大小
        self.patch_grid = self.patch_embeddings.grid_size
        # 获取嵌入维度
        self.embed_dim = config.embed_dim
        # 是否使用掩码令牌
        self.use_mask_token = use_mask_token
        # 是否使用绝对位置嵌入
        self.use_absolute_embeddings = config.use_absolute_embeddings

        # 初始化层归一化层对象
        self.norm = tf.keras.layers.LayerNormalization(name="norm", epsilon=1e-5)
        # 初始化丢弃层对象
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout_prob, name="dropout")
        self.config = config

    def build(self, input_shape: tf.TensorShape) -> None:
        # 如果使用掩码令牌，则添加掩码令牌权重
        if self.use_mask_token:
            self.mask_token = self.add_weight(shape=(1, 1, self.embed_dim), initializer="zeros", name="mask_token")
        else:
            self.mask_token = None

        # 如果使用绝对位置嵌入，则添加位置嵌入权重
        if self.use_absolute_embeddings:
            self.position_embeddings = self.add_weight(
                (1, self.num_patches + 1, self.embed_dim), initializer="zeros", name="positional_embeddings"
            )
        else:
            self.position_embeddings = None

        if self.built:
            return
        self.built = True
        # 构建补丁嵌入层
        if getattr(self, "patch_embeddings", None) is not None:
            with tf.name_scope(self.patch_embeddings.name):
                self.patch_embeddings.build(None)
        # 构建层归一化层
        if getattr(self, "norm", None) is not None:
            with tf.name_scope(self.norm.name):
                self.norm.build([None, None, self.config.embed_dim])
        # 构建丢弃层
        if getattr(self, "dropout", None) is not None:
            with tf.name_scope(self.dropout.name):
                self.dropout.build(None)

    def call(
        self, pixel_values: tf.Tensor, bool_masked_pos: bool = None, training: bool = False
    ):
    # 根据像素值计算嵌入向量和输出维度
    embeddings, output_dimensions = self.patch_embeddings(pixel_values, training=training)
    # 对嵌入向量进行归一化处理
    embeddings = self.norm(embeddings, training=training)
    # 获取嵌入向量的形状信息
    batch_size, seq_len, _ = shape_list(embeddings)

    # 如果存在掩码位置信息
    if bool_masked_pos is not None:
        # 创建掩码标记矩阵，大小与嵌入向量相同，将掩码位置替换为掩码标记
        mask_tokens = tf.repeat(self.mask_token, batch_size, 0)
        mask_tokens = tf.repeat(mask_tokens, seq_len, 1)
        # 将掩码位置信息扩展为与嵌入向量相同的维度
        mask = tf.expand_dims(bool_masked_pos, -1)
        # 将掩码位置信息转换为与掩码标记相同的数据类型
        mask = tf.cast(mask, mask_tokens.dtype)

        # 将嵌入向量中的掩码位置替换为掩码标记
        embeddings = embeddings * (1.0 - mask) + mask_tokens * mask

    # 如果存在位置嵌入向量，则将其加到嵌入向量中
    if self.position_embeddings is not None:
        embeddings = embeddings + self.position_embeddings

    # 对嵌入向量进行 dropout 处理
    embeddings = self.dropout(embeddings, training=training)

    # 返回处理后的嵌入向量和输出维度
    return embeddings, output_dimensions
class TFSwinPatchEmbeddings(tf.keras.layers.Layer):
    """
    Image to Patch Embedding.
    """

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        # 获取配置中的图像大小和补丁大小、通道数和嵌入维度
        image_size, patch_size = config.image_size, config.patch_size
        num_channels, hidden_size = config.num_channels, config.embed_dim
        # 将图像大小和补丁大小转换为迭代对象
        image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        # 计算图像可以划分的补丁数量
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        # 保存参数值
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches = num_patches
        self.grid_size = (image_size[0] // patch_size[0], image_size[1] // patch_size[1])

        # 创建卷积层，用于投影图像补丁
        self.projection = tf.keras.layers.Conv2D(
            filters=hidden_size,
            kernel_size=self.patch_size,
            strides=self.patch_size,
            padding="valid",
            name="projection",
        )

    # 如果需要，对输入进行填充，使其大小可以被self.patch_size整除
    def maybe_pad(self, pixel_values: tf.Tensor, height: int, width: int) -> tf.Tensor:
        if width % self.patch_size[1] != 0:
            pad_values = ((0, 0), (0, 0), (0, 0), (0, self.patch_size[1] - width % self.patch_size[1]))
            pixel_values = tf.pad(pixel_values, pad_values)
        if height % self.patch_size[0] != 0:
            pad_values = ((0, 0), (0, 0), (0, self.patch_size[0] - height % self.patch_size[0]), (0, 0))
            pixel_values = tf.pad(pixel_values, pad_values)
        return pixel_values

    # 对输入进行处理，返回嵌入和输出维度
    def call(self, pixel_values: tf.Tensor, training: bool = False) -> Tuple[tf.Tensor, Tuple[int, int]]:
        _, num_channels, height, width = shape_list(pixel_values)
        # 如果使用eager execution，并且通道数不匹配则报错
        if tf.executing_eagerly() and num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
            )
        # 对输入进行填充，使其大小可以被self.patch_size整除
        pixel_values = self.maybe_pad(pixel_values, height, width)

        # 调整张量维度顺序，使其变为B,H,W,C
        pixel_values = tf.transpose(pixel_values, (0, 2, 3, 1))

        # 将输入进行投影处理
        embeddings = self.projection(pixel_values, training=training)

        # 调整张量维度顺序，使其变为B,C,H,W
        embeddings = tf.transpose(embeddings, (0, 3, 1, 2))

        # 获取嵌入张量的维度信息
        batch_size, channels, height, width = shape_list(embeddings)
        output_dimensions = (height, width)

        # 重新构造张量，返回嵌入和输出维度
        embeddings = tf.reshape(embeddings, (batch_size, channels, -1))
        embeddings = tf.transpose(embeddings, (0, 2, 1))
        return embeddings, output_dimensions
    # 建立模型，如果已经建立过了，则直接返回
    def build(self, input_shape=None):
        if self.built:
            return
        # 标记模型已经建立
        self.built = True
        # 如果存在投影层属性，则建立投影层
        if getattr(self, "projection", None) is not None:
            # 在 TensorFlow 中使用命名空间创建投影层
            with tf.name_scope(self.projection.name):
                # 建立投影层，输入形状为 [None, None, None, self.num_channels]
                self.projection.build([None, None, None, self.num_channels])
# 定义 Patch Merging Layer 类，继承自 tf.keras.layers.Layer
class TFSwinPatchMerging(tf.keras.layers.Layer):
    """
    Patch Merging Layer.

    Args:
        input_resolution (`Tuple[int]`):
            Resolution of input feature.
        dim (`int`):
            Number of input channels.
        norm_layer (`tf.keras.layer.Layer`, *optional*, defaults to `tf.keras.layers.LayerNormalization`):
            Normalization layer class.
    """

    def __init__(
        self, input_resolution: Tuple[int, int], dim: int, norm_layer: Optional[Callable] = None, **kwargs
    ) -> None:
        # 调用父类的初始化方法
        super().__init__(**kwargs)
        # 记录输入特征的分辨率
        self.input_resolution = input_resolution
        # 记录输入特征的通道数
        self.dim = dim
        # 定义一个全连接层用于降维
        self.reduction = tf.keras.layers.Dense(2 * dim, use_bias=False, name="reduction")
        # 如果未指定 norm_layer，则使用 tf.keras.layers.LayerNormalization 作为默认归一化层
        if norm_layer is None:
            self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="norm")
        # 否则使用指定的归一化层
        else:
            self.norm = norm_layer(name="norm")

    # 如果输入特征的高度或宽度是奇数，需要对其进行填充
    def maybe_pad(self, input_feature: tf.Tensor, height: int, width: int) -> tf.Tensor:
        should_pad = (height % 2 == 1) or (width % 2 == 1)
        if should_pad:
            # 进行填充
            pad_values = ((0, 0), (0, height % 2), (0, width % 2), (0, 0))
            input_feature = tf.pad(input_feature, pad_values)

        return input_feature

    # 定义前向传播方法
    def call(self, input_feature: tf.Tensor, input_dimensions: Tuple[int, int], training: bool = False) -> tf.Tensor:
        # 获取输入特征的高度、宽度和通道数
        height, width = input_dimensions
        batch_size, _, num_channels = shape_list(input_feature)

        # 将输入特征reshape为4维
        input_feature = tf.reshape(input_feature, (batch_size, height, width, num_channels))
        # 对输入特征进行填充
        input_feature = self.maybe_pad(input_feature, height, width)
        # 对输入特征进行4等分
        input_feature_0 = input_feature[:, 0::2, 0::2, :]
        input_feature_1 = input_feature[:, 1::2, 0::2, :]
        input_feature_2 = input_feature[:, 0::2, 1::2, :]
        input_feature_3 = input_feature[:, 1::2, 1::2, :]
        # 将4等分的特征拼接起来
        input_feature = tf.concat([input_feature_0, input_feature_1, input_feature_2, input_feature_3], -1)
        # 将4维特征reshape为3维
        input_feature = tf.reshape(
            input_feature, (batch_size, -1, 4 * num_channels)
        )

        # 对特征进行归一化和降维
        input_feature = self.norm(input_feature, training=training)
        input_feature = self.reduction(input_feature, training=training)

        return input_feature
    # 构建模型，如果已经构建过则直接返回
    def build(self, input_shape=None):
        if self.built:
            return
        # 设置模型的构建标志为 True
        self.built = True
        # 如果有定义降维操作，则构建该层，输入形状为 [None, None, 4 * self.dim]
        if getattr(self, "reduction", None) is not None:
            # 使用 reduction 的名字作为作用域名称
            with tf.name_scope(self.reduction.name):
                # 构建 reduction 层，输入形状为 [None, None, 4 * self.dim]
                self.reduction.build([None, None, 4 * self.dim])
        # 如果有定义标准化操作，则构建该层，输入形状为 [None, None, 4 * self.dim]
        if getattr(self, "norm", None) is not None:
            # 使用 norm 的名字作为作用域名称
            with tf.name_scope(self.norm.name):
                # 构建 norm 层，输入形状为 [None, None, 4 * self.dim]
                self.norm.build([None, None, 4 * self.dim])
class TFSwinDropPath(tf.keras.layers.Layer):
    """定义一个自定义的 Keras 层，用于在残差块的主路径中按样本丢弃路径(随机深度)。"""

    def __init__(self, drop_prob: float = None, scale_by_keep: bool = True, **kwargs) -> None:
        super(TFSwinDropPath, self).__init__(**kwargs)
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def call(self, input: tf.Tensor, training: bool = False) -> tf.Tensor:
        return drop_path(input, self.drop_prob, training, self.scale_by_keep)


class TFSwinSelfAttention(tf.keras.layers.Layer):
    """定义一个自定义的 Keras 层，用于实现自注意力机制。"""

    def __init__(self, config: SwinConfig, dim: int, num_heads: int, **kwargs) -> None:
        super().__init__(**kwargs)
        if dim % num_heads != 0:
            raise ValueError(
                f"The hidden size ({dim}) is not a multiple of the number of attention heads ({num_heads})"
            )

        self.num_attention_heads = num_heads
        self.attention_head_size = int(dim / num_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        window_size = config.window_size
        self.window_size = (
            window_size if isinstance(window_size, collections.abc.Iterable) else (window_size, window_size)
        )

        self.query = tf.keras.layers.Dense(
            self.all_head_size,
            kernel_initializer=get_initializer(config.initializer_range),
            use_bias=config.qkv_bias,
            name="query",
        )
        self.key = tf.keras.layers.Dense(
            self.all_head_size,
            kernel_initializer=get_initializer(config.initializer_range),
            use_bias=config.qkv_bias,
            name="key",
        )
        self.value = tf.keras.layers.Dense(
            self.all_head_size,
            kernel_initializer=get_initializer(config.initializer_range),
            use_bias=config.qkv_bias,
            name="value",
        )

        self.dropout = tf.keras.layers.Dropout(config.attention_probs_dropout_prob)
```  
    # 构建相对位置偏置表的权重，用于存储相对位置偏置信息
    def build(self, input_shape: tf.TensorShape) -> None:
        # 添加相对位置偏置表的权重，初始化为零值
        self.relative_position_bias_table = self.add_weight(
            shape=(((2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1)), self.num_attention_heads),
            initializer="zeros",
            name="relative_position_bias_table",
        )
        # 添加相对位置索引的权重，用于存储相对位置索引信息，不可训练，数据类型为整数
        self.relative_position_index = self.add_weight(
            shape=(self.window_size[0] ** 2, self.window_size[1] ** 2),
            trainable=False,
            dtype=tf.int32,
            name="relative_position_index",
        )

        # 获取每个标记在窗口内的配对相对位置索引
        coords_h = tf.range(self.window_size[0])
        coords_w = tf.range(self.window_size[1])
        coords = tf.stack(tf.meshgrid(coords_h, coords_w, indexing="ij"))
        coords_flatten = tf.reshape(coords, (shape_list(coords)[0], -1))
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = tf.transpose(relative_coords, (1, 2, 0))

        stack_0, stack_1 = tf.unstack(relative_coords, axis=2)
        stack_0 += self.window_size[0] - 1
        stack_0 *= 2 * self.window_size[1] - 1
        stack_1 += self.window_size[1] - 1
        relative_coords = tf.stack([stack_0, stack_1], axis=2)

        # 将相对位置信息求和后，转换为整数类型后赋值给相对位置索引的权重
        self.relative_position_index.assign(tf.cast(tf.reduce_sum(relative_coords, axis=-1), tf.int32))

        # 如果已经构建完毕则直接返回
        if self.built:
            return
        # 设置构建完毕标志为True
        self.built = True
        # 如果存在查询(Query)权重，则构建查询权重
        if getattr(self, "query", None) is not None:
            with tf.name_scope(self.query.name):
                self.query.build([None, None, self.all_head_size])
        # 如果存在键(Key)权重，则构建键权重
        if getattr(self, "key", None) is not None:
            with tf.name_scope(self.key.name):
                self.key.build([None, None, self.all_head_size])
        # 如果存在数值(Value)权重，则构建数值权重
        if getattr(self, "value", None) is not None:
            with tf.name_scope(self.value.name):
                self.value.build([None, None, self.all_head_size])

    # 将输入张量重排为注意力得分矩阵所需的形状
    def transpose_for_scores(self, x: tf.Tensor) -> tf.Tensor:
        new_x_shape = shape_list(x)[:-1] + [self.num_attention_heads, self.attention_head_size]
        x = tf.reshape(x, new_x_shape)
        return tf.transpose(x, (0, 2, 1, 3))

    # 实现自注意力机制的调用函数
    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: tf.Tensor | None = None,
        head_mask: tf.Tensor | None = None,
        output_attentions: bool = False,
        training: bool = False,
    # 使用给定隐藏状态的形状确定批次大小、维度和分数矩阵维度
    batch_size, dim, _ = shape_list(hidden_states)
    # 对隐藏状态应用查询层
    mixed_query_layer = self.query(hidden_states)
    
    # 对隐藏状态应用键层并调整维度
    key_layer = self.transpose_for_scores(self.key(hidden_states))
    # 对隐藏状态应用值层并调整维度
    value_layer = self.transpose_for_scores(self.value(hidden_states))
    # 对查询层调整维度
    query_layer = self.transpose_for_scores(mixed_query_layer)
    
    # 计算"查询"和"键"之间的点积，得到原始的注意力分数
    attention_scores = tf.matmul(query_layer, tf.transpose(key_layer, (0, 1, 3, 2)))
    
    # 将得到的注意力分数除以scale值
    attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    # 根据相对位置索引获取相对位置偏置
    relative_position_bias = tf.gather(
        self.relative_position_bias_table, tf.reshape(self.relative_position_index, (-1,))
    )
    # 调整相对位置偏置的维度
    relative_position_bias = tf.reshape(
        relative_position_bias,
        (self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1),
    )
    
    # 对相对位置偏置进行转置
    relative_position_bias = tf.transpose(relative_position_bias, (2, 0, 1))
    # 将相对位置偏置添加到注意力分数中
    attention_scores = attention_scores + tf.expand_dims(relative_position_bias, 0)
    
    # 如果有注意力掩码，则应用注意力掩码
    if attention_mask is not None:
        # 调整注意力分数和注意力掩码的维度
        mask_shape = shape_list(attention_mask)[0]
        attention_scores = tf.reshape(
            attention_scores, (batch_size // mask_shape, mask_shape, self.num_attention_heads, dim, dim)
        )
        attention_mask = tf.expand_dims(attention_mask, 1)
        attention_mask = tf.expand_dims(attention_mask, 0)
        attention_scores = attention_scores + attention_mask
        attention_scores = tf.reshape(attention_scores, (-1, self.num_attention_heads, dim, dim))
    
    # 将注意力分数归一化为概率
    attention_probs = tf.nn.softmax(attention_scores, axis=-1)
    
    # 使用dropout对注意力概率进行随机遮挡
    attention_probs = self.dropout(attention_probs, training=training)
    
    # 如果有注意力头掩码，则应用头掩码
    if head_mask is not None:
        attention_probs = attention_probs * head_mask
    
    # 计算上下文层
    context_layer = tf.matmul(attention_probs, value_layer)
    # 转置上下文层
    context_layer = tf.transpose(context_layer, (0, 2, 1, 3))
    # 调整上下文层的维度
    new_context_layer_shape = shape_list(context_layer)[:-2] + [
        self.all_head_size,
    ]
    context_layer = tf.reshape(context_layer, new_context_layer_shape)
    
    # 创建输出并根据需要添加注意力概率信息
    outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
    
    return outputs
class TFSwinSelfOutput(tf.keras.layers.Layer):
    # 定义 TFSwinSelfOutput 类，继承自 tf.keras.layers.Layer
    def __init__(self, config: SwinConfig, dim: int, **kwargs) -> None:
        # 初始化函数，接收 SwinConfig 对象、维度 dim 和其他关键字参数
        super().__init__(**kwargs)
        # 调用父类初始化方法
        self.dense = tf.keras.layers.Dense(dim, name="dense")
        # 创建一个 Dense 层，维度为 dim，命名为 "dense"
        self.dropout = tf.keras.layers.Dropout(config.attention_probs_dropout_prob, name="dropout")
        # 创建一个 Dropout 层，使用 config 中的参数作为 dropout 概率，命名为 "dropout"
        self.dim = dim
        # 将维度 dim 存储在实例变量中

    def call(self, hidden_states: tf.Tensor, input_tensor: tf.Tensor, training: bool = False) -> tf.Tensor:
        # 定义 call 方法，接收隐藏状态 hidden_states、输入张量 input_tensor 和训练标志 training
        hidden_states = self.dense(hidden_states)
        # 使用 Dense 层处理隐藏状态
        hidden_states = self.dropout(hidden_states, training=training)
        # 使用 Dropout 层处理处理后的隐藏状态
        return hidden_states
        # 返回处理后的隐藏状态

    def build(self, input_shape=None):
        # 定义 build 方法，接收输入形状 input_shape，如果已经构建则直接返回
        if self.built:
            return
        # 如果已经构建，则直接返回
        self.built = True
        # 标记为已构建
        if getattr(self, "dense", None) is not None:
            # 如果存在 Dense 层
            with tf.name_scope(self.dense.name):
                # 使用 Dense 层的名称创建命名空间
                self.dense.build([None, None, self.dim])
                # 构建 Dense 层

        if getattr(self, "dropout", None) is not None:
            # 如果存在 Dropout 层
            with tf.name_scope(self.dropout.name):
                # 使用 Dropout 层的名称创建命名空间
                self.dropout.build(None)
                # 构建 Dropout 层


class TFSwinAttention(tf.keras.layers.Layer):
    # 定义 TFSwinAttention 类，继承自 tf.keras.layers.Layer
    def __init__(self, config: SwinConfig, dim: int, num_heads: int, **kwargs) -> None:
        # 初始化函数，接收 SwinConfig 对象、维度 dim、头数 num_heads 和其他关键字参数
        super().__init__(**kwargs)
        # 调用父类初始化方法
        self.self = TFSwinSelfAttention(config, dim, num_heads, name="self")
        # 创建一个 TFSwinSelfAttention 实例，使用传入的参数和名称 "self"
        self.self_output = TFSwinSelfOutput(config, dim, name="output")
        # 创建一个 TFSwinSelfOutput 实例，使用传入的参数和名称 "output"
        self.pruned_heads = set()
        # 创建一个空集合，用于存储修剪后的头部信息

    def prune_heads(self, heads):
        # 定义 prune_heads 方法，接收要修剪的头部信息 heads
        """
        Prunes heads of the model. See base class PreTrainedModel heads: dict of {layer_num: list of heads to prune in
        this layer}
        """
        raise NotImplementedError
        # 抛出未实现错误，需要在子类中实现该方法

    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: tf.Tensor | None = None,
        head_mask: tf.Tensor | None = None,
        output_attentions: bool = False,
        training: bool = False,
    ) -> tf.Tensor:
        # 定义 call 方法，接收隐藏状态 hidden_states、注意力掩码 attention_mask（可选）、头部掩码 head_mask（可选）、是否输出注意力 output_attentions、训练标志 training
        self_outputs = self.self(hidden_states, attention_mask, head_mask, output_attentions, training=training)
        # 使用 TFSwinSelfAttention 处理隐藏状态和掩码，得到自注意力输出
        attention_output = self.self_output(self_outputs[0], hidden_states, training=training)
        # 使用 TFSwinSelfOutput 处理自注意力输出和原始隐藏状态，得到最终注意力输出
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        # 构建输出元组，包含注意力输出和其他可能的输出
        return outputs
        # 返回输出元组

    def build(self, input_shape=None):
        # 定义 build 方法，接收输入形状 input_shape，如果已经构建则直接返回
        if self.built:
            return
        # 如果已经构建，则直接返回
        self.built = True
        # 标记为已构建
        if getattr(self, "self", None) is not None:
            # 如果存在自注意力层
            with tf.name_scope(self.self.name):
                # 使用自注意力层的名称创建命名空间
                self.self.build(None)
                # 构建自注意力层

        if getattr(self, "self_output", None) is not None:
            # 如果存在自注意力输出层
            with tf.name_scope(self.self_output.name):
                # 使用自注意力输出层的名称创建命名空间
                self.self_output.build(None)
                # 构建自注意力输出层


class TFSwinIntermediate(tf.keras.layers.Layer):
    # 定义 TFSwinIntermediate 类，继承自 tf.keras.layers.Layer
    # 初始化方法，接受SwinConfig对象和整数dim作为参数
    def __init__(self, config: SwinConfig, dim: int, **kwargs) -> None:
        # 调用父类的初始化方法
        super().__init__(**kwargs)
        # 创建一个全连接层，输出维度为config.mlp_ratio * dim，层名为"dense"
        self.dense = tf.keras.layers.Dense(int(config.mlp_ratio * dim), name="dense")
        # 判断config.hidden_act是否为字符串类型，是则使用ACT2FN中对应的激活函数，否则直接使用config.hidden_act
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act
        # 保存输入维度
        self.dim = dim

    # 前向传播方法，接受tf.Tensor类型的hidden_states参数，返回tf.Tensor类型
    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        # 通过全连接层处理hidden_states
        hidden_states = self.dense(hidden_states)
        # 通过激活函数处理处理后的hidden_states
        hidden_states = self.intermediate_act_fn(hidden_states)
        # 返回处理后的hidden_states
        return hidden_states

    # 构建方法，接受input_shape参数，无返回值
    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回
        if self.built:
            return
        # 标记为已构建
        self.built = True
        # 如果存在全连接层，根据self.dim构建全连接层
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.dim])
# 定义一个继承自 tf.keras.layers.Layer 的类 TFSwinOutput
class TFSwinOutput(tf.keras.layers.Layer):
    # 初始化函数，接收 SwinConfig 对象和维度 dim 作为输入
    def __init__(self, config: SwinConfig, dim: int, **kwargs) -> None:
        # 调用父类的初始化函数
        super().__init__(**kwargs)
        # 创建一个全连接层，输入维度为 dim，输出维度也为 dim
        self.dense = tf.keras.layers.Dense(dim, name="dense")
        # 创建一个 Dropout 层，dropout 概率为 config.hidden_dropout_prob
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout_prob, "dropout")
        # 保存 config 和 dim 属性
        self.config = config
        self.dim = dim

    # 定义前向传播函数
    def call(self, hidden_states: tf.Tensor, training: bool = False) -> tf.Tensor:
        # 将输入通过全连接层
        hidden_states = self.dense(hidden_states)
        # 将结果通过 Dropout 层
        hidden_states = self.dropout(hidden_states, training=training)
        # 返回结果
        return hidden_states

    # 定义 build 函数，用于初始化权重
    def build(self, input_shape=None):
        # 如果已经 built，则直接返回
        if self.built:
            return
        self.built = True
        # 如果 dense 层已经存在，则使用 tf.name_scope 给它命名
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, int(self.config.mlp_ratio * self.dim)])

# 定义一个继承自 tf.keras.layers.Layer 的类 TFSwinLayer
class TFSwinLayer(tf.keras.layers.Layer):
    # 初始化函数，接收 SwinConfig、维度 dim、输入分辨率 input_resolution、头数 num_heads 和 shift_size 作为输入
    def __init__(
        self, config, dim, input_resolution: Tuple[int, int], num_heads: int, shift_size: int = 0, **kwargs
    ) -> None:
        # 调用父类的初始化函数
        super().__init__(**kwargs)
        # 保存 chunk_size_feed_forward 属性
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        # 计算输入分辨率的最小值
        min_res = tf.reduce_min(input_resolution)
        # 如果最小值小于等于 config.window_size，则 window_size 就是最小值；否则就是 config.window_size
        self.window_size = min_res if min_res <= config.window_size else config.window_size
        # 如果最小值小于等于 window_size，则 shift_size 为 0；否则就是输入的 shift_size
        self.shift_size = 0 if min_res <= self.window_size else shift_size
        # 保存输入分辨率
        self.input_resolution = input_resolution

        # 创建一个 LayerNormalization 层，用于层归一化
        self.layernorm_before = tf.keras.layers.LayerNormalization(
            epsilon=config.layer_norm_eps, name="layernorm_before"
        )
        # 创建一个 TFSwinAttention 层，用于注意力机制
        self.attention = TFSwinAttention(config, dim, num_heads, name="attention")
        # 如果 config.drop_path_rate 大于 0.0，则创建一个 TFSwinDropPath 层；否则创建一个恒等变换层
        self.drop_path = (
            TFSwinDropPath(config.drop_path_rate, name="drop_path")
            if config.drop_path_rate > 0.0
            else tf.keras.layers.Activation("linear", name="drop_path")
        )
        # 创建一个 LayerNormalization 层，用于层归一化
        self.layernorm_after = tf.keras.layers.LayerNormalization(
            epsilon=config.layer_norm_eps, name="layernorm_after"
        )
        # 创建一个 TFSwinIntermediate 层，用于中间变换
        self.intermediate = TFSwinIntermediate(config, dim, name="intermediate")
        # 创建一个 TFSwinOutput 层，用于输出
        self.swin_output = TFSwinOutput(config, dim, name="output")
        # 保存 dim 属性
        self.dim = dim
    # 生成注意力掩码，用于Self-Attention操作
    def get_attn_mask(self, height: int, width: int, window_size: int, shift_size: int) -> tf.Tensor | None:
        # 创建一个全零张量作为初始的注意力掩码
        img_mask = tf.zeros((height, width))
        # 定义高度上的切片范围
        height_slices = ((0, -window_size), (-window_size, -shift_size), (-shift_size, -1))
        # 定义宽度上的切片范围
        width_slices = ((0, -window_size), (-window_size, -shift_size), (-shift_size, -1))

        # 计算 SW-MSA 的注意力掩码
        if shift_size > 0:
            count = 0
            for height_slice in height_slices:
                for width_slice in width_slices:
                    # 计算高度和宽度上的索引
                    height_inds = tf.range(height_slice[0] % height, height_slice[1] % height + 1)
                    width_inds = tf.range(width_slice[0] % width, width_slice[1] % width + 1)
                    indices = tf.reshape(tf.stack(tf.meshgrid(height_inds, width_inds), axis=-1), (-1, 2))
                    if len(indices) >= 1:
                        # 根据索引更新对应位置的值
                        updates = tf.ones((len(indices),), dtype=img_mask.dtype) * count
                        img_mask = tf.tensor_scatter_nd_update(img_mask, indices, updates)
                    count += 1

        # 在最后添加两个维度
        img_mask = tf.expand_dims(img_mask, -1)
        img_mask = tf.expand_dims(img_mask, 0)

        # 分割成窗口并重塑形状
        mask_windows = window_partition(img_mask, window_size)
        mask_windows = tf.reshape(mask_windows, (-1, window_size * window_size))
        # 生成注意力掩码
        attn_mask = tf.expand_dims(mask_windows, 1) - tf.expand_dims(mask_windows, 2)
        attn_mask = tf.where(attn_mask != 0, float(-100.0), attn_mask)
        attn_mask = tf.where(attn_mask == 0, float(0.0), attn_mask)
        
        # 返回生成的注意力掩码
        return attn_mask

    # 对隐藏状态进行填充
    def maybe_pad(
        self, hidden_states: tf.Tensor, window_size: int, height: int, width: int
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        # 计算需要填充的右侧和底部大小
        pad_right = (window_size - width % window_size) % window_size
        pad_bottom = (window_size - height % window_size) % window_size
        # 构建填充值
        pad_values = [[0, 0], [0, pad_bottom], [0, pad_right], [0, 0]]
        # 在隐藏状态张量上进行填充
        hidden_states = tf.pad(hidden_states, pad_values)
        # 重塑填充值的形状
        pad_values = tf.reshape(pad_values, (-1,))
        # 返回填充后的隐藏状态和填充值
        return hidden_states, pad_values

    # 实现TransformerLayer的调用
    def call(
        self,
        hidden_states: tf.Tensor,
        input_dimensions: Tuple[int, int],
        head_mask: tf.Tensor | None = None,
        output_attentions: bool = False,
        training: bool = False,
    ) -> tf.Tensor:
        # 如果窗口大小大于输入分辨率，则不分割窗口
        min_res = tf.reduce_min(input_dimensions)
        shift_size = 0 if min_res <= self.window_size else self.shift_size
        window_size = min_res if min_res <= self.window_size else self.window_size

        height, width = input_dimensions
        batch_size, _, channels = shape_list(hidden_states)
        shortcut = hidden_states

        # 在应用 layernorm 前对隐藏状态进行处理并reshape
        hidden_states = self.layernorm_before(hidden_states, training=training)
        hidden_states = tf.reshape(hidden_states, (batch_size, height, width, channels))
        
        # 对隐藏状态进行填充至窗口大小的倍数
        hidden_states, pad_values = self.maybe_pad(hidden_states, window_size, height, width)

        _, height_pad, width_pad, _ = shape_list(hidden_states)
        
        # 如果 shift_size 大于 0，则进行循环移位
        if shift_size > 0:
            shifted_hidden_states = tf.roll(hidden_states, shift=(-shift_size, -shift_size), axis=(1, 2))
        else:
            shifted_hidden_states = hidden_states

        # 分割窗口
        hidden_states_windows = window_partition(shifted_hidden_states, window_size)
        hidden_states_windows = tf.reshape(hidden_states_windows, (-1, window_size * window_size, channels))

        # 获取 attention mask
        attn_mask = self.get_attn_mask(
            height=height_pad, width=width_pad, window_size=window_size, shift_size=shift_size
        )

        # 进行注意力计算
        attention_outputs = self.attention(
            hidden_states_windows, attn_mask, head_mask, output_attentions=output_attentions, training=training
        )

        attention_output = attention_outputs[0]

        attention_windows = tf.reshape(attention_output, (-1, window_size, window_size, channels))
        shifted_windows = window_reverse(attention_windows, window_size, height_pad, width_pad)

        # 如果 shift_size 大于 0，则进行反向循环移位
        if shift_size > 0:
            attention_windows = tf.roll(shifted_windows, shift=(shift_size, shift_size), axis=(1, 2))
        else:
            attention_windows = shifted_windows

        was_padded = pad_values[3] > 0 or pad_values[5] > 0
        if was_padded:
            attention_windows = attention_windows[:, :height, :width, :]

        attention_windows = tf.reshape(attention_windows, (batch_size, height * width, channels))

        hidden_states = shortcut + self.drop_path(attention_windows, training=training)

        # 对输出进行 layernorm 处理
        layer_output = self.layernorm_after(hidden_states, training=training)
        layer_output = self.intermediate(layer_output)
        layer_output = hidden_states + self.swin_output(layer_output, training=training)

        # 如果需要输出注意力，则将注意力信息一起返回
        layer_outputs = (layer_output, attention_outputs[1]) if output_attentions else (layer_output,)
        return layer_outputs
    # 如果模型已经构建，则直接返回，不进行重复构建
    def build(self, input_shape=None):
        if self.built:
            return
        # 将模型标记为已构建状态
        self.built = True
        # 如果存在预先定义的 layernorm_before 属性
        if getattr(self, "layernorm_before", None) is not None:
            # 在 TensorFlow 的命名空间下构建 layernorm_before 层
            with tf.name_scope(self.layernorm_before.name):
                # 使用 layernorm_before 层的输入形状构建该层
                self.layernorm_before.build([None, None, self.dim])
        # 如果存在预先定义的 attention 属性
        if getattr(self, "attention", None) is not None:
            # 在 TensorFlow 的命名空间下构建 attention 层
            with tf.name_scope(self.attention.name):
                # 使用 attention 层的输入形状构建该层
                self.attention.build(None)
        # 如果存在预先定义的 drop_path 属性
        if getattr(self, "drop_path", None) is not None:
            # 在 TensorFlow 的命名空间下构建 drop_path 层
            with tf.name_scope(self.drop_path.name):
                # 使用 drop_path 层的输入形状构建该层
                self.drop_path.build(None)
        # 如果存在预先定义的 layernorm_after 属性
        if getattr(self, "layernorm_after", None) is not None:
            # 在 TensorFlow 的命名空间下构建 layernorm_after 层
            with tf.name_scope(self.layernorm_after.name):
                # 使用 layernorm_after 层的输入形状构建该层
                self.layernorm_after.build([None, None, self.dim])
        # 如果存在预先定义的 intermediate 属性
        if getattr(self, "intermediate", None) is not None:
            # 在 TensorFlow 的命名空间下构建 intermediate 层
            with tf.name_scope(self.intermediate.name):
                # 使用 intermediate 层的输入形状构建该层
                self.intermediate.build(None)
        # 如果存在预先定义的 swin_output 属性
        if getattr(self, "swin_output", None) is not None:
            # 在 TensorFlow 的命名空间下构建 swin_output 层
            with tf.name_scope(self.swin_output.name):
                # 使用 swin_output 层的输入形状构建该层
                self.swin_output.build(None)
# 定义一个名为 TFSwinStage 的 TensorFlow 自定义层，用于实现Swin Transformer中的一个阶段
class TFSwinStage(tf.keras.layers.Layer):
    def __init__(
        self,
        config: SwinConfig,  # Swin Transformer 的配置对象
        dim: int,  # 该阶段的维度
        input_resolution: Tuple[int, int],  # 输入图像的分辨率
        depth: int,  # 该阶段的层数
        num_heads: int,  # 注意力头的数量
        drop_path: List[float],  # 丢弃路径的概率列表
        downsample: Optional[Callable],  # 下采样函数，可选
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)  # 调用父类的初始化方法
        self.config = config  # 存储 Swin Transformer 的配置对象
        self.dim = dim  # 存储该阶段的维度
        # 创建由多个 TFSwinLayer 组成的 blocks 列表，每个 TFSwinLayer 表示一个阶段中的一个层
        self.blocks = [
            TFSwinLayer(
                config=config,
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                shift_size=0 if (i % 2 == 0) else config.window_size // 2,  # 计算位移大小
                name=f"blocks.{i}",  # 给每个 TFSwinLayer 命名
            )
            for i in range(depth)
        ]

        # 补丁合并层
        if downsample is not None:  # 如果传入了下采样函数
            # 使用下采样函数创建 downsample 层
            self.downsample = downsample(
                input_resolution,
                dim=dim,
                norm_layer=partial(tf.keras.layers.LayerNormalization, epsilon=1e-5),  # 规范化层
                name="downsample",  # 层的名称
            )
        else:
            self.downsample = None  # 否则设置为 None

        self.pointing = False  # 初始化指向属性为 False

    # 定义该层的前向传播方法
    def call(
        self,
        hidden_states: tf.Tensor,  # 输入的隐藏状态张量
        input_dimensions: Tuple[int, int],  # 输入图像的维度
        head_mask: tf.Tensor | None = None,  # 注意力头的掩码张量，默认为 None
        output_attentions: Optional[bool] = False,  # 是否输出注意力，默认为 False
        training: bool = False,  # 是否处于训练模式，默认为 False
    ) -> Tuple[tf.Tensor, ...]:  # 返回值为一个元组，包含输出张量及可能的注意力张量
        height, width = input_dimensions  # 获取输入图像的高度和宽度
        for i, layer_module in enumerate(self.blocks):  # 遍历每个 TFSwinLayer 层
            layer_head_mask = head_mask[i] if head_mask is not None else None  # 获取当前层的注意力头掩码

            # 调用当前 TFSwinLayer 层的前向传播方法
            layer_outputs = layer_module(
                hidden_states, input_dimensions, layer_head_mask, output_attentions, training=training
            )

            hidden_states = layer_outputs[0]  # 更新隐藏状态张量

        if self.downsample is not None:  # 如果存在下采样层
            height_downsampled, width_downsampled = (height + 1) // 2, (width + 1) // 2  # 计算下采样后的高度和宽度
            output_dimensions = (height, width, height_downsampled, width_downsampled)  # 输出的维度信息
            hidden_states = self.downsample(layer_outputs[0], input_dimensions, training=training)  # 对隐藏状态进行下采样
        else:
            output_dimensions = (height, width, height, width)  # 否则输出维度与输入维度相同

        stage_outputs = (hidden_states, output_dimensions)  # 阶段输出为隐藏状态和输出维度的元组

        if output_attentions:  # 如果需要输出注意力
            stage_outputs += layer_outputs[1:]  # 将注意力张量添加到阶段输出中
        return stage_outputs  # 返回阶段输出的元组

    # 构建层的方法
    def build(self, input_shape=None):
        if self.built:  # 如果已经构建过，则直接返回
            return
        self.built = True  # 将已构建标志设置为 True
        if getattr(self, "downsample", None) is not None:  # 如果存在下采样层
            with tf.name_scope(self.downsample.name):  # 使用下采样层的名称作为命名空间
                self.downsample.build(None)  # 构建下采样层
        if getattr(self, "blocks", None) is not None:  # 如果存在 blocks
            for layer in self.blocks:  # 遍历每个 TFSwinLayer 层
                with tf.name_scope(layer.name):  # 使用当前层的名称作为命名空间
                    layer.build(None)  # 构建当前层
    # 初始化 Swin Transformer 模型
    def __init__(self, config: SwinConfig, grid_size: Tuple[int, int], **kwargs):
        # 调用父类的 __init__ 方法
        super().__init__(**kwargs)
        # 获取 Swin Transformer 模型的层数
        self.num_layers = len(config.depths)
        # 保存 Swin Transformer 配置
        self.config = config
        # 计算每个层的 drop path rate
        dpr = list((tf.linspace(0, 1, sum(config.depths)) * config.drop_path_rate).numpy())
        # 创建 Swin Transformer 的各个层
        self.layers = [
            TFSwinStage(
                # 传递 Swin Transformer 配置
                config=config,
                # 计算每个层的嵌入维度
                dim=int(config.embed_dim * 2**i_layer),
                # 计算每个层的输入分辨率
                input_resolution=(grid_size[0] // (2**i_layer), grid_size[1] // (2**i_layer)),
                # 获取每个层的深度
                depth=config.depths[i_layer],
                # 获取每个层的头数
                num_heads=config.num_heads[i_layer],
                # 获取每个层的 drop path rate
                drop_path=dpr[sum(config.depths[:i_layer]) : sum(config.depths[: i_layer + 1])],
                # 如果不是最后一层，则添加下采样层
                downsample=TFSwinPatchMerging if (i_layer < self.num_layers - 1) else None,
                # 设置层名称
                name=f"layers.{i_layer}",
            )
            for i_layer in range(self.num_layers)
        ]
        # 关闭梯度检查点
        self.gradient_checkpointing = False
    
    # 定义模型的前向传播过程
    def call(
        self,
        hidden_states: tf.Tensor,
        input_dimensions: Tuple[int, int],
        head_mask: tf.Tensor | None = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        training: bool = False,
    ):
        ) -> Union[Tuple[tf.Tensor, ...], TFSwinEncoderOutput]:
        # 定义返回值的类型注解
        all_input_dimensions = ()
        all_hidden_states = () if output_hidden_states else None
        all_reshaped_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        if output_hidden_states:
            batch_size, _, hidden_size = shape_list(hidden_states)
            # rearrange b (h w) c -> b c h w
            # 重塑隐藏状态张量的形状，将 (batch_size, sequence_length, hidden_size) -> (batch_size, hidden_size, sequence_length)
            reshaped_hidden_state = tf.reshape(hidden_states, (batch_size, *input_dimensions, hidden_size))
            reshaped_hidden_state = tf.transpose(reshaped_hidden_state, (0, 3, 1, 2))
            all_hidden_states += (hidden_states,)
            all_reshaped_hidden_states += (reshaped_hidden_state,)

        for i, layer_module in enumerate(self.layers):
            layer_head_mask = head_mask[i] if head_mask is not None else None

            layer_outputs = layer_module(
                hidden_states, input_dimensions, layer_head_mask, output_attentions, training=training
            )

            hidden_states = layer_outputs[0]
            output_dimensions = layer_outputs[1]

            input_dimensions = (output_dimensions[-2], output_dimensions[-1])
            all_input_dimensions += (input_dimensions,)

            if output_hidden_states:
                batch_size, _, hidden_size = shape_list(hidden_states)
                # rearrange b (h w) c -> b c h w
                # 重塑隐藏状态张量的形状，将 (batch_size, sequence_length, hidden_size) -> (batch_size, hidden_size, sequence_length)
                reshaped_hidden_state = tf.reshape(hidden_states, (batch_size, *input_dimensions, hidden_size))
                reshaped_hidden_state = tf.transpose(reshaped_hidden_state, (0, 3, 1, 2))
                all_hidden_states += (hidden_states,)
                all_reshaped_hidden_states += (reshaped_hidden_state,)

            if output_attentions:
                all_self_attentions += layer_outputs[2:]

        if not return_dict:
            # 如果不返回字典，则将结果中为None的值去除并返回
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)

        return TFSwinEncoderOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            reshaped_hidden_states=all_reshaped_hidden_states,
        )

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "layers", None) is not None:
            for layer in self.layers:
                with tf.name_scope(layer.name):
                    layer.build(None)
# 定义 TFSwinPreTrainedModel 类，继承自 TFPreTrainedModel
class TFSwinPreTrainedModel(TFPreTrainedModel):
    """
    处理权重初始化和预训练模型下载和加载的抽象类
    """
    config_class = SwinConfig  # 使用 SwinConfig 类进行配置
    base_model_prefix = "swin"  # 基础模型前缀为 "swin"
    main_input_name = "pixel_values"  # 主输入参数名为 "pixel_values"

# SWIN_START_DOCSTRING 和 SWIN_INPUTS_DOCSTRING 是长字符串常量，用于描述模型和参数
SWIN_START_DOCSTRING = r"""
    This model is a Tensorflow
    [tf.keras.layers.Layer](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer) sub-class. Use it as a
    regular Tensorflow Module and refer to the Tensorflow documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`SwinConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

SWIN_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`tf.Tensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See [`ViTImageProcessor.__call__`]
            for details.
        head_mask (`tf.Tensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
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
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""

# 标准化数据格式函数，接受一个字符串并返回一个字符串
def normalize_data_format(value: str) -> str:
    """
    根据输入的规范化数据格式参数，返回规范化后的数据格式字符串
    """
    if value is None:
        value = tf.keras.backend.image_data_format()  # 如果参数为空，则使用 TensorFlow 的数据格式
    data_format = value.lower()  # 将参数转换为小写
    if data_format not in {"channels_first", "channels_last"}:  # 如果参数不是 "channels_first" �� "channels_last"，则抛出异常
        raise ValueError(
            'The `data_format` argument must be one of "channels_first", "channels_last". Received: ' + str(value)
        )
    return data_format  # 返回规范化后的数据格式字符串

# 定义 AdaptiveAveragePooling1D 类
class AdaptiveAveragePooling1D(tf.keras.layers.Layer):
    """
    Args:
    """
    # 实现具有自适应内核大小的平均 1D 池化。
    # 参数:
    #   output_size: 一个整数或单个整数的元组/列表，指定池化特征的数量。输出通道的新尺寸。
    #   data_format: 一个字符串，`channels_last`（默认）或`channels_first`之一。输入中维度的排序。
    #     `channels_last`对应于形状为`(batch, steps, channels)`的输入，而`channels_first`对应于形状为`(batch, channels, steps)`的输入。
    #   
    # 输入形状:
    #   - 如果`data_format='channels_last'`：形状为`(batch, steps, channels)`的3D张量。
    #   - 如果`data_format='channels_first'`：形状为`(batch, channels, steps)`的3D张量。
    # 输出形状:
    #   - 如果`data_format='channels_last'`：形状为`(batch_size, pooled_steps, channels)`的3D张量。
    #   - 如果`data_format='channels_first'`：形状为`(batch_size, channels, pooled_steps)`的3D张量。
    #
    # 从[tensorflow-addon's adaptive pooling.py](https://github.com/tensorflow/addons/blob/8cec33fcaaf1cf90aec7bdd55a0fcdbb251ce5c2/tensorflow_addons/layers/adaptive_pooling.py#L90-L120)中修改。
    """
    
    # 初始化函数，设置属性值
    def __init__(
        self,
        output_size: Union[int, Iterable[int]],
        reduce_function: Callable = tf.reduce_mean,
        data_format: Optional[str] = None,
        **kwargs,
    ) -> None:
        self.data_format = normalize_data_format(data_format)  # 标准化数据格式
        self.reduce_function = reduce_function  # 减少功能
        self.output_size = (output_size,) if isinstance(output_size, int) else tuple(output_size)  # 输出尺寸为整数或元组转换为元组
        super().__init__(**kwargs)  # 调用父类的初始化函数

    # 调用函数，根据输入数据执行平均池化操作
    def call(self, inputs: tf.Tensor, *args) -> None:
        bins = self.output_size[0]
        if self.data_format == "channels_last":
            splits = tf.split(inputs, bins, axis=1)  # 在通道轴上拆分输入
            splits = tf.stack(splits, axis=1)  # 在轴1上堆叠拆分后的张量
            out_vect = self.reduce_function(splits, axis=2)  # 对拆分后的张量进行降维操作
        else:
            splits = tf.split(inputs, bins, axis=2)  # 在通道轴上拆分输入
            splits = tf.stack(splits, axis=2)  # 在轴2上堆叠拆分后的张量
            out_vect = self.reduce_function(splits, axis=3)  # 对拆分后的张量进行降维操作
        return out_vect

    # 计算输出形状
    def compute_output_shape(self, input_shape: Iterable[int]) -> tf.TensorShape:
        input_shape = tf.TensorShape(input_shape).as_list()  # 将输入形状转换为列表
        if self.data_format == "channels_last":
            shape = tf.TensorShape([input_shape[0], self.output_size[0], input_shape[2]])  # 计算channels_last格式的输出形状
        else:
            shape = tf.TensorShape([input_shape[0], input_shape[1], self.output_size[0]])  # 计算channels_first格式的输出形状
        return shape

    # 获取配置信息
    def get_config(self) -> Dict[str, Any]:
        config = {
            "output_size": self.output_size,
            "data_format": self.data_format,
        }
        base_config = super().get_config()  # 获取父类的配置信息
        return {**base_config, **config}  # 返回合并后的配置信息
# 声明一个基于 keras 可序列化的类 TFSwinMainLayer
@keras_serializable
class TFSwinMainLayer(tf.keras.layers.Layer):
    # 定义 config_class 属性为 SwinConfig 类
    config_class = SwinConfig

    # 初始化方法
    def __init__(
        self, config: SwinConfig, add_pooling_layer: bool = True, use_mask_token: bool = False, **kwargs
    ) -> None:
        # 调用父类初始化方法
        super().__init__(**kwargs)
        # 初始化 config 属性为传入的参数 config
        self.config = config
        # 计算 num_layers 为 config.depths 的长度
        self.num_layers = len(config.depths)
        # 计算 num_features 为 config.embed_dim 乘以 2 的 (num_layers - 1) 次方后取整
        self.num_features = int(config.embed_dim * 2 ** (self.num_layers - 1))

        # 初始化 embeddings 属性为 TFSwinEmbeddings 类的实例
        self.embeddings = TFSwinEmbeddings(config, use_mask_token=use_mask_token, name="embeddings")
        # 初始化 encoder 属性为 TFSwinEncoder 类的实例
        self.encoder = TFSwinEncoder(config, self.embeddings.patch_grid, name="encoder")

        # 初始化 layernorm 属性为 LayerNormalization 类的实例
        self.layernorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="layernorm")
        # 如果 add_pooling_layer 为 True，初始化 pooler 属性为 AdaptiveAveragePooling1D 类的实例；否则为 None
        self.pooler = AdaptiveAveragePooling1D(output_size=(1,)) if add_pooling_layer else None

    # 获取输入 embeddings 的方法
    def get_input_embeddings(self) -> TFSwinPatchEmbeddings:
        return self.embeddings.patch_embeddings

    # 精简模型中的 heads 方法
    def _prune_heads(self, heads_to_prune: Dict[int, List]):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        # 遍历 heads_to_prune 中的层号和需要精简的 heads
        for layer, heads in heads_to_prune.items():
            # 调用对应 encoder 层中 attention 的 prune_heads 方法进行精简
            self.encoder.layer[layer].attention.prune_heads(heads)

    # 获取头 mask 方法
    def get_head_mask(self, head_mask: Optional[Any]) -> List:
        # 如果 head_mask 不为 None，抛出 NotImplementedError
        if head_mask is not None:
            raise NotImplementedError
        # 返回长度为 config.depths 的 None 列表
        return [None] * len(self.config.depths)

    # 调用方法
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
        ) -> Union[TFSwinModelOutput, Tuple[tf.Tensor, ...]]:
            # 设置输出参数是否包括注意力矩阵
            output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
            # 设置输出参数是否包括隐藏状态
            output_hidden_states = (
                output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
            )
            # 设置返回字典
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict

            if pixel_values is None:
                # 如果像素值为None，抛出数值错误
                raise ValueError("You have to specify pixel_values")

            # 准备头部掩码
            # 1.0表示保留该头
            # 注意力概率的形状为bsz x n_heads x N x N
            # 输入头掩码的形状为[num_heads] 或 [num_hidden_layers x num_heads]
            # 头掩码被转换为形状为[num_hidden_layers x batch x num_heads x seq_length x seq_length]
            head_mask = self.get_head_mask(head_mask)
            # 获取嵌入输出以及输入维度
            embedding_output, input_dimensions = self.embeddings(
                pixel_values, bool_masked_pos=bool_masked_pos, training=training
            )

            # 编码器输出
            encoder_outputs = self.encoder(
                embedding_output,
                input_dimensions,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                training=training,
            )

            # 序列输出
            sequence_output = encoder_outputs[0]
            # 序列输出进行层归一化
            sequence_output = self.layernorm(sequence_output, training=training)

            pooled_output = None
            if self.pooler is not None:
                batch_size, _, num_features = shape_list(sequence_output)
                # 池化输出
                pooled_output = self.pooler(sequence_output)
                pooled_output = tf.reshape(pooled_output, (batch_size, num_features))

            if not return_dict:
                output = (sequence_output, pooled_output) + encoder_outputs[1:]
                return output

            return TFSwinModelOutput(
                last_hidden_state=sequence_output,
                pooler_output=pooled_output,
                hidden_states=encoder_outputs.hidden_states,
                attentions=encoder_outputs.attentions,
                reshaped_hidden_states=encoder_outputs.reshaped_hidden_states,
            )

        def build(self, input_shape=None):
            if self.built:
                return
            self.built = True
            if getattr(self, "embeddings", None) is not None:
                with tf.name_scope(self.embeddings.name):
                    self.embeddings.build(None)
            if getattr(self, "encoder", None) is not None:
                with tf.name_scope(self.encoder.name):
                    self.encoder.build(None)
            if getattr(self, "layernorm", None) is not None:
                with tf.name_scope(self.layernorm.name):
                    self.layernorm.build([None, None, self.num_features])
# 使用装饰器添加开始文档字符串
@add_start_docstrings(
    "The bare Swin Model transformer outputting raw hidden-states without any specific head on top.",
    SWIN_START_DOCSTRING,
)
class TFSwinModel(TFSwinPreTrainedModel):
    # 初始化函数
    def __init__(
        self, config: SwinConfig, add_pooling_layer: bool = True, use_mask_token: bool = False, **kwargs
    ) -> None:
        # 调用父类的初始化函数
        super().__init__(config, **kwargs)
        self.config = config
        self.swin = TFSwinMainLayer(config, name="swin")

    # 调用函数
    @add_start_docstrings_to_model_forward(SWIN_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFSwinModelOutput,
        config_class=_CONFIG_FOR_DOC,
        modality="vision",
        expected_output=_EXPECTED_OUTPUT_SHAPE,
    )
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
    ) -> Union[TFSwinModelOutput, Tuple[tf.Tensor, ...]]:
        r"""
        bool_masked_pos (`tf.Tensor` of shape `(batch_size, num_patches)`, *optional*):
            Boolean masked positions. Indicates which patches are masked (1) and which aren't (0).
        """
        # 设置输出注意力
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 设置输出隐藏状态
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 设置返回字典
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 如果像素值为空，引发数值错误
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")
        # 使用 Swin 模块处理像素值
        swin_outputs = self.swin(
            pixel_values=pixel_values,
            bool_masked_pos=bool_masked_pos,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )
        # 返回处理后的结果
        return swin_outputs

    # 构建函数
    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 如果有 swin 属性，构建 swin 层
        if getattr(self, "swin", None) is not None:
            with tf.name_scope(self.swin.name):
                self.swin.build(None)


class TFSwinPixelShuffle(tf.keras.layers.Layer):
    """TF layer implementation of torch.nn.PixelShuffle"""

    # 初始化函数
    def __init__(self, upscale_factor: int, **kwargs) -> None:
        # 调用父类的初始化函数
        super().__init__(**kwargs)
        # 如果放大因子不是整数或小于2，引发数值错误
        if not isinstance(upscale_factor, int) or upscale_factor < 2:
            raise ValueError(f"upscale_factor must be an integer value >= 2 got {upscale_factor}")
        # 设置放大因子
        self.upscale_factor = upscale_factor
    # 定义一个方法，参数为一个张量 x，返回值也是一个张量
    def call(self, x: tf.Tensor) -> tf.Tensor:
        # 将输入张量赋值给隐藏状态
        hidden_states = x
        # 获取隐藏状态的形状信息
        batch_size, _, _, num_input_channels = shape_list(hidden_states)
        # 计算块的大小（尺寸）的平方
        block_size_squared = self.upscale_factor**2
        # 计算输出深度
        output_depth = int(num_input_channels / block_size_squared)
        # 创建一个排列，用于后续的通道重组
        permutation = tf.constant(
            [[i + j * block_size_squared for i in range(block_size_squared) for j in range(output_depth)]]
        )
        # 根据排列对隐藏状态进行通道重组
        hidden_states = tf.gather(params=hidden_states, indices=tf.tile(permutation, [batch_size, 1]), batch_dims=-1)
        # 使用深度到空间操作对隐藏状态进行重组，改变数据格式为 NHWC
        hidden_states = tf.nn.depth_to_space(hidden_states, block_size=self.upscale_factor, data_format="NHWC")
        # 返回重组后的隐藏状态
        return hidden_states
class TFSwinDecoder(tf.keras.layers.Layer):
    # 定义一个自定义的层，用于Swin解码器
    def __init__(self, config: SwinConfig, **kwargs):
        # 初始化函数，接受SwinConfig实例并添加额外的参数
        super().__init__(**kwargs)
        # 调用父类的初始化函数
        self.conv2d = tf.keras.layers.Conv2D(
            filters=config.encoder_stride**2 * config.num_channels, kernel_size=1, strides=1, name="0"
        )
        # 创建一个卷积层对象
        self.pixel_shuffle = TFSwinPixelShuffle(config.encoder_stride, name="1")
        # 创建一个像素重组对象
        self.config = config
        # 保存配置信息

    def call(self, x: tf.Tensor) -> tf.Tensor:
        # 定义调用函数，传入一个张量并返回一个张量
        hidden_states = x
        # 初始化隐藏层，存储输入张量
        # B,C,H,W -> B,H,W,C
        hidden_states = tf.transpose(hidden_states, (0, 2, 3, 1))
        # 调整维度顺序
        hidden_states = self.conv2d(hidden_states)
        # 通过卷积层处理隐藏层
        hidden_states = self.pixel_shuffle(hidden_states)
        # 通过像素重组处理隐藏层
        # B,H,W,C -> B,C,H,W
        hidden_states = tf.transpose(hidden_states, (0, 3, 1, 2))
        # 调整维度顺序
        return hidden_states
        # 返回处理后的隐藏层数据

    def build(self, input_shape=None):
        # 定义构建函数，处理输入形状
        if self.built:
            return
        # 如果已经构建过，直接返回
        self.built = True
        # 标记为已构建
        if getattr(self, "conv2d", None) is not None:
            with tf.name_scope(self.conv2d.name):
                self.conv2d.build([None, None, None, self.config.hidden_size])
        # 构建卷积层
        if getattr(self, "pixel_shuffle", None) is not None:
            with tf.name_scope(self.pixel_shuffle.name):
                self.pixel_shuffle.build(None)
        # 构建像素重组对象


@add_start_docstrings(
    "Swin Model with a decoder on top for masked image modeling, as proposed in"
    " [SimMIM](https://arxiv.org/abs/2111.09886).",
    SWIN_START_DOCSTRING,
)
class TFSwinForMaskedImageModeling(TFSwinPreTrainedModel):
    # 定义用于蒙版图像建模的Swin模型类
    def __init__(self, config: SwinConfig):
        # 初始化函数，传入SwinConfig实例
        super().__init__(config)
        # 调用父类初始化函数

        self.swin = TFSwinMainLayer(config, add_pooling_layer=False, use_mask_token=True, name="swin")
        # 初始化Swin主层
        self.decoder = TFSwinDecoder(config, name="decoder")
        # 初始化Swin解码器

    @add_start_docstrings_to_model_forward(SWIN_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFSwinMaskedImageModelingOutput, config_class=_CONFIG_FOR_DOC)
    @unpack_inputs
    def call(
        # 定义call函数
        self,
        pixel_values: tf.Tensor | None = None,
        bool_masked_pos: tf.Tensor | None = None,
        head_mask: tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
    def build(self, input_shape=None):
        # 定义构建函数，处理输入形状
        if self.built:
            return
        # 如果已经构建过，直接返回
        self.built = True
        # 标记为已构建
        if getattr(self, "swin", None) is not None:
            with tf.name_scope(self.swin.name):
                self.swin.build(None)
        # 构建Swin主层
        if getattr(self, "decoder", None) is not None:
            with tf.name_scope(self.decoder.name):
                self.decoder.build(None)
        # 构建解码器
    # 初始化方法，接受一个 SwinConfig 类型的参数
    def __init__(self, config: SwinConfig):
        # 调用父类的初始化方法
        super().__init__(config)

        # 设置类的属性 num_labels 为传入参数 config 的 num_labels 属性
        self.num_labels = config.num_labels
        # 创建一个 TFSwinMainLayer 类型的对象 swin，并命名为"swin"
        self.swin = TFSwinMainLayer(config, name="swin")

        # 分类器头部
        # 如果 config.num_labels 大于 0，则创建一个具有 config.num_labels 个单元的 Dense 层，命名为"classifier"；否则创建一个线性激活层
        self.classifier = (
            tf.keras.layers.Dense(config.num_labels, name="classifier")
            if config.num_labels > 0
            else tf.keras.layers.Activation("linear", name="classifier")
        )

    # 调用方法，对输入的图片数据进行预测或推理
    @add_start_docstrings_to_model_forward(SWIN_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_IMAGE_CLASS_CHECKPOINT,
        output_type=TFSwinImageClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_IMAGE_CLASS_EXPECTED_OUTPUT,
    )
    @unpack_inputs
    def call(
        self,
        pixel_values: tf.Tensor | None = None,
        head_mask: tf.Tensor | None = None,
        labels: tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
    ) -> Union[Tuple[tf.Tensor, ...], TFSwinImageClassifierOutput]:
        # 设置标签返回方式
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用 swin 的方法传入参数，得到输出
        outputs = self.swin(
            pixel_values,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        # 获取池化输出
        pooled_output = outputs[1]

        # 对池化输出使用分类器进行推理
        logits = self.classifier(pooled_output, training=training)

        # 如果有标签，则计算损失
        loss = None if labels is None else self.hf_compute_loss(labels, logits)

        # 如果 return_dict 为 False，则返回 output；否则返回 TFSwinImageClassifierOutput 对象
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TFSwinImageClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            reshaped_hidden_states=outputs.reshaped_hidden_states,
        )

    # 构建模型
    def build(self, input_shape=None):
        # 如果已经构建过模型，则直接返回
        if self.built:
            return
        self.built = True
        # 如果存在属性 swin，则构建 swin 模型
        if getattr(self, "swin", None) is not None:
            with tf.name_scope(self.swin.name):
                self.swin.build(None)
        # 如果存在属性 classifier，并且具有 name 属性，则构建 classifier 模型
        if getattr(self, "classifier", None) is not None:
            if hasattr(self.classifier, "name"):
                with tf.name_scope(self.classifier.name):
                    self.classifier.build([None, None, self.swin.num_features])
```