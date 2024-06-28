# `.\models\swin\modeling_tf_swin.py`

```py
# coding=utf-8
# 版权 2022 年 Microsoft Research 和 HuggingFace Inc. 团队保留所有权利。
#
# 根据 Apache 许可证 2.0 版本（“许可证”）许可；
# 除非符合许可证，否则不得使用此文件。
# 您可以在以下网址获取许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，软件根据“原样”分发，
# 没有任何明示或暗示的保证或条件。
# 有关特定语言的权限，请参阅许可证。

""" TF 2.0 Swin Transformer 模型。"""

from __future__ import annotations

import collections.abc  # 导入用于检查抽象基类的标准库模块
import math  # 导入数学函数库
import warnings  # 导入警告处理模块
from dataclasses import dataclass  # 导入用于数据类的装饰器
from functools import partial  # 导入用于创建偏函数的函数
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union  # 导入类型提示相关模块

import tensorflow as tf  # 导入 TensorFlow 库

from ...activations_tf import ACT2FN  # 导入 TensorFlow 激活函数映射
from ...modeling_tf_utils import (  # 导入 TensorFlow 模型相关工具函数
    TFPreTrainedModel,
    TFSequenceClassificationLoss,
    get_initializer,
    keras,
    keras_serializable,
    unpack_inputs,
)
from ...tf_utils import shape_list  # 导入 TensorFlow 工具函数，用于获取张量形状
from ...utils import (  # 导入通用工具函数
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from .configuration_swin import SwinConfig  # 导入 Swin 模型的配置类

logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器

# 用于文档的常量和字符串
_CONFIG_FOR_DOC = "SwinConfig"  # Swin 配置类的文档字符串
_CHECKPOINT_FOR_DOC = "microsoft/swin-tiny-patch4-window7-224"  # 预训练模型检查点的文档字符串
_EXPECTED_OUTPUT_SHAPE = [1, 49, 768]  # 预期输出形状的文档字符串

# 用于图像分类的常量和字符串
_IMAGE_CLASS_CHECKPOINT = "microsoft/swin-tiny-patch4-window7-224"  # 图像分类模型检查点的文档字符串
_IMAGE_CLASS_EXPECTED_OUTPUT = "tabby, tabby cat"  # 图像分类预期输出的文档字符串

# Swin 模型的预训练模型存档列表
TF_SWIN_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "microsoft/swin-tiny-patch4-window7-224",
    # 查看所有 Swin 模型，请访问 https://huggingface.co/models?filter=swin
]

# drop_path, TFSwinPatchEmbeddings, TFSwinPatchMerging 和 TFSwinDropPath 是 TensorFlow
# 中对 timm 库中 PyTorch 功能的实现。

@dataclass
class TFSwinEncoderOutput(ModelOutput):
    """
    Swin 编码器的输出，可能包括隐藏状态和注意力。
    """
    # 定义函数参数及其类型注解，用于接收模型的输出
    Args:
        last_hidden_state (`tf.Tensor` of shape `(batch_size, sequence_length, hidden_size)`):
            模型最后一层的隐藏状态序列的张量。
        hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            可选参数，当 `output_hidden_states=True` 或 `config.output_hidden_states=True` 时返回，包含模型每一层的隐藏状态的元组。
            每个张量的形状为 `(batch_size, sequence_length, hidden_size)`。
            包括初始嵌入输出后每个层的模型隐藏状态。
        attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            可选参数，当 `output_attentions=True` 或 `config.output_attentions=True` 时返回，包含模型每个阶段的注意力权重的元组。
            每个张量的形状为 `(batch_size, num_heads, sequence_length, sequence_length)`。
            在注意力 softmax 后的注意力权重，用于计算自注意力头部的加权平均值。
        reshaped_hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            可选参数，当 `output_hidden_states=True` 或 `config.output_hidden_states=True` 时返回，包含模型每一层的隐藏状态的元组。
            每个张量的形状为 `(batch_size, hidden_size, height, width)`。
            包括初始嵌入输出后每个层的模型隐藏状态，重塑以包括空间维度。
# 定义一个基于数据类的类 TFSwinModelOutput，继承自 ModelOutput
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

    # 定义成员变量并初始化，用来存储模型输出的不同部分
    last_hidden_state: tf.Tensor = None
    pooler_output: tf.Tensor | None = None
    hidden_states: Tuple[tf.Tensor, ...] | None = None
    attentions: Tuple[tf.Tensor, ...] | None = None
    reshaped_hidden_states: Tuple[tf.Tensor, ...] | None = None
    Args:
        loss (`tf.Tensor` of shape `(1,)`, *optional*, returned when `bool_masked_pos` is provided):
            Masked image modeling (MLM) loss.
        reconstruction (`tf.Tensor` of shape `(batch_size, num_channels, height, width)`):
            Reconstructed pixel values.
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

    # 初始化属性：损失、重构像素值、隐藏状态、注意力权重和重塑后的隐藏状态，默认为None
    loss: tf.Tensor | None = None
    reconstruction: tf.Tensor = None
    hidden_states: Tuple[tf.Tensor, ...] | None = None
    attentions: Tuple[tf.Tensor, ...] | None = None
    reshaped_hidden_states: Tuple[tf.Tensor, ...] | None = None

    @property
    def logits(self):
        # 发出警告，提醒用户logits属性即将在Transformers的第5个版本中移除，建议使用reconstruction属性获取最终输出
        warnings.warn(
            "logits attribute is deprecated and will be removed in version 5 of Transformers."
            " Please use the reconstruction attribute to retrieve the final output instead.",
            FutureWarning,
        )
        # 返回重构属性作为输出
        return self.reconstruction
@dataclass
class TFSwinImageClassifierOutput(ModelOutput):
    """
    Swin outputs for image classification.

    Args:
        loss (`tf.Tensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Classification (or regression if config.num_labels==1) loss.
        logits (`tf.Tensor` of shape `(batch_size, config.num_labels)`):
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
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

    loss: tf.Tensor | None = None  # 损失值，如果提供了 `labels` 参数，则返回；用于分类（如果 `config.num_labels==1` 则为回归）的损失。
    logits: tf.Tensor = None  # 分类（或回归，如果 `config.num_labels==1`）得分，未经 SoftMax 处理，形状为 `(batch_size, config.num_labels)`。
    hidden_states: Tuple[tf.Tensor, ...] | None = None  # 模型在每一层输出的隐藏状态和初始嵌入输出的元组，形状为 `(batch_size, sequence_length, hidden_size)`。
    attentions: Tuple[tf.Tensor, ...] | None = None  # 注意力权重，经过注意力 SoftMax 后的结果，用于计算自注意力头部中的加权平均值，形状为 `(batch_size, num_heads, sequence_length, sequence_length)` 的元组。
    reshaped_hidden_states: Tuple[tf.Tensor, ...] | None = None  # 模型在每一层输出的隐藏状态和初始嵌入输出的重塑版本，包括空间维度，形状为 `(batch_size, hidden_size, height, width)` 的元组。


def window_partition(input_feature: tf.Tensor, window_size: int) -> tf.Tensor:
    """
    Partitions the given input into windows.
    """
    batch_size, height, width, num_channels = shape_list(input_feature)  # 获取输入特征的形状信息
    input_feature = tf.reshape(
        input_feature,
        (batch_size, height // window_size, window_size, width // window_size, window_size, num_channels),  # 将输入特征重塑为窗口的形状
    )
    windows = tf.transpose(input_feature, (0, 1, 3, 2, 4, 5))  # 调整窗口的顺序
    windows = tf.reshape(windows, (-1, window_size, window_size, num_channels))  # 将调整顺序后的窗口展平
    return windows


def window_reverse(windows: tf.Tensor, window_size: int, height: int, width: int) -> tf.Tensor:
    """
    Merges windows to produce higher resolution features.
    """
    x = tf.shape(windows)[0]  # 获取窗口张量的第一维大小
    y = tf.cast(height * width / (window_size * window_size), tf.int32)  # 计算合并后特征的大小
    batch_size = tf.math.floordiv(x, y)  # 计算批次大小
    # 将输入的窗口数据重新形状为指定的多维张量，以便进行后续处理
    windows = tf.reshape(
        windows, (batch_size, height // window_size, width // window_size, window_size, window_size, -1)
    )
    # 转置张量的维度顺序，以便后续处理更方便
    windows = tf.transpose(windows, (0, 1, 3, 2, 4, 5))
    # 将张量重新形状为指定的多维张量，以便进行后续处理
    windows = tf.reshape(windows, (batch_size, height, width, -1))
    # 返回处理后的窗口数据张量
    return windows
def drop_path(
    input: tf.Tensor, drop_prob: float = 0.0, training: bool = False, scale_by_keep: bool = True
) -> tf.Tensor:
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    """
    # 如果 drop_prob 为 0 或者不处于训练模式，则直接返回输入
    if drop_prob == 0.0 or not training:
        return input
    # 计算保留的概率
    keep_prob = 1 - drop_prob
    # 获取输入张量的形状信息
    input_shape = shape_list(input)
    # 获取张量的维度数
    ndim = len(input_shape)
    # 构建一个形状与输入张量相同的随机张量，用于决定每个元素是否保留
    shape = [input_shape[0]] + [1] * (ndim - 1)  # 适用于不同维度的张量，不仅限于2D卷积网络
    random_tensor = tf.random.uniform(shape)
    # 将随机张量中小于等于保留概率的元素设置为1.0，其余设置为0.0
    random_tensor = tf.where(random_tensor <= keep_prob, 1.0, 0.0)
    # 如果保留概率大于0且需要按保留概率进行缩放，则对随机张量进行缩放处理
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor /= keep_prob
    # 返回经过随机路径丢弃后的输入张量
    return input * random_tensor


class TFSwinEmbeddings(keras.layers.Layer):
    """
    Construct the patch and position embeddings. Optionally, also the mask token.
    """

    def __init__(self, config: SwinConfig, use_mask_token: bool = False, **kwargs) -> None:
        super().__init__(**kwargs)
        # 初始化补丁和位置嵌入
        self.patch_embeddings = TFSwinPatchEmbeddings(config, name="patch_embeddings")
        # 获取补丁数量和网格大小
        self.num_patches = self.patch_embeddings.num_patches
        self.patch_grid = self.patch_embeddings.grid_size
        self.embed_dim = config.embed_dim
        self.use_mask_token = use_mask_token
        self.use_absolute_embeddings = config.use_absolute_embeddings

        # 层归一化
        self.norm = keras.layers.LayerNormalization(name="norm", epsilon=1e-5)
        # dropout
        self.dropout = keras.layers.Dropout(config.hidden_dropout_prob, name="dropout")
        self.config = config

    def build(self, input_shape: tf.TensorShape) -> None:
        # 如果需要使用掩码令牌，则添加掩码令牌的权重
        if self.use_mask_token:
            self.mask_token = self.add_weight(shape=(1, 1, self.embed_dim), initializer="zeros", name="mask_token")
        else:
            self.mask_token = None

        # 如果使用绝对位置嵌入，则添加位置嵌入的权重
        if self.use_absolute_embeddings:
            self.position_embeddings = self.add_weight(
                (1, self.num_patches + 1, self.embed_dim), initializer="zeros", name="positional_embeddings"
            )
        else:
            self.position_embeddings = None

        # 如果已经构建，则直接返回
        if self.built:
            return
        self.built = True
        # 构建补丁嵌入层、层归一化层和dropout层
        if getattr(self, "patch_embeddings", None) is not None:
            with tf.name_scope(self.patch_embeddings.name):
                self.patch_embeddings.build(None)
        if getattr(self, "norm", None) is not None:
            with tf.name_scope(self.norm.name):
                self.norm.build([None, None, self.config.embed_dim])
        if getattr(self, "dropout", None) is not None:
            with tf.name_scope(self.dropout.name):
                self.dropout.build(None)

    def call(
        self, pixel_values: tf.Tensor, bool_masked_pos: bool = None, training: bool = False
    ) -> tf.Tensor:
        # 留待实现，用于调用该层处理输入张量
        pass
    ) -> Tuple[tf.Tensor, Tuple[int, int]]:
        # 计算输入图像的嵌入向量和输出维度
        embeddings, output_dimensions = self.patch_embeddings(pixel_values, training=training)
        
        # 对嵌入向量进行归一化处理
        embeddings = self.norm(embeddings, training=training)
        
        # 获取嵌入向量的形状信息
        batch_size, seq_len, _ = shape_list(embeddings)

        # 如果存在需要屏蔽的位置信息
        if bool_masked_pos is not None:
            # 创建与嵌入向量相同形状的屏蔽标记
            mask_tokens = tf.repeat(self.mask_token, batch_size, 0)
            mask_tokens = tf.repeat(mask_tokens, seq_len, 1)
            # 将屏蔽位置的嵌入向量替换为屏蔽标记
            mask = tf.expand_dims(bool_masked_pos, -1)
            mask = tf.cast(mask, mask_tokens.dtype)

            embeddings = embeddings * (1.0 - mask) + mask_tokens * mask

        # 如果存在位置嵌入向量，则将其加到嵌入向量上
        if self.position_embeddings is not None:
            embeddings = embeddings + self.position_embeddings

        # 对嵌入向量进行dropout处理
        embeddings = self.dropout(embeddings, training=training)

        # 返回处理后的嵌入向量和输出维度
        return embeddings, output_dimensions
class TFSwinPatchEmbeddings(keras.layers.Layer):
    """
    Image to Patch Embedding.
    """

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        # 从配置中获取图像大小和patch大小
        image_size, patch_size = config.image_size, config.patch_size
        # 获取通道数和嵌入维度
        num_channels, hidden_size = config.num_channels, config.embed_dim
        # 如果图像大小和patch大小不是可迭代对象，转换为元组
        image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        # 计算patch的数量
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        
        # 设置类属性
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches = num_patches
        self.grid_size = (image_size[0] // patch_size[0], image_size[1] // patch_size[1])

        # 定义投影层，使用Conv2D将patch映射到隐藏维度空间
        self.projection = keras.layers.Conv2D(
            filters=hidden_size,
            kernel_size=self.patch_size,
            strides=self.patch_size,
            padding="valid",
            name="projection",
        )

    def maybe_pad(self, pixel_values: tf.Tensor, height: int, width: int) -> tf.Tensor:
        # 如果宽度不是patch宽度的整数倍，进行填充
        if width % self.patch_size[1] != 0:
            pad_values = ((0, 0), (0, 0), (0, 0), (0, self.patch_size[1] - width % self.patch_size[1]))
            pixel_values = tf.pad(pixel_values, pad_values)
        # 如果高度不是patch高度的整数倍，进行填充
        if height % self.patch_size[0] != 0:
            pad_values = ((0, 0), (0, 0), (0, self.patch_size[0] - height % self.patch_size[0]), (0, 0))
            pixel_values = tf.pad(pixel_values, pad_values)
        return pixel_values

    def call(self, pixel_values: tf.Tensor, training: bool = False) -> Tuple[tf.Tensor, Tuple[int, int]]:
        # 获取输入张量的形状信息
        _, num_channels, height, width = shape_list(pixel_values)
        # 在动态执行环境下，检查通道数是否与配置中设置的一致
        if tf.executing_eagerly() and num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
            )
        # 如果需要，对输入进行填充，使其可以被self.patch_size整除
        pixel_values = self.maybe_pad(pixel_values, height, width)

        # 调整输入张量的维度顺序 B,C,H,W -> B,H,W,C
        pixel_values = tf.transpose(pixel_values, (0, 2, 3, 1))

        # 使用投影层将patch映射到隐藏维度空间
        embeddings = self.projection(pixel_values, training=training)

        # 调整输出张量的维度顺序 B,H,W,C -> B,C,H,W
        embeddings = tf.transpose(embeddings, (0, 3, 1, 2))

        # 获取输出张量的形状信息
        batch_size, channels, height, width = shape_list(embeddings)
        output_dimensions = (height, width)

        # 将输出张量reshape为 B,N,C 的形式，其中N为patch的数量
        embeddings = tf.reshape(embeddings, (batch_size, channels, -1))
        embeddings = tf.transpose(embeddings, (0, 2, 1))
        return embeddings, output_dimensions
    # 定义一个方法用于构建模型，如果已经构建过则直接返回
    def build(self, input_shape=None):
        if self.built:
            return
        # 标记模型已经构建
        self.built = True
        # 检查是否存在投影层，并在 TensorFlow 的命名空间下构建投影层
        if getattr(self, "projection", None) is not None:
            with tf.name_scope(self.projection.name):
                # 使用投影层的建模方法来构建投影层，传入特定维度的列表
                self.projection.build([None, None, None, self.num_channels])
class TFSwinPatchMerging(keras.layers.Layer):
    """
    Patch Merging Layer.

    Args:
        input_resolution (`Tuple[int]`):
            Resolution of input feature.
        dim (`int`):
            Number of input channels.
        norm_layer (`keras.layer.Layer`, *optional*, defaults to `keras.layers.LayerNormalization`):
            Normalization layer class.
    """

    def __init__(
        self, input_resolution: Tuple[int, int], dim: int, norm_layer: Optional[Callable] = None, **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.input_resolution = input_resolution  # 设置输入特征的分辨率
        self.dim = dim  # 设置输入通道数
        self.reduction = keras.layers.Dense(2 * dim, use_bias=False, name="reduction")  # 创建一个稠密层用于特征降维
        if norm_layer is None:
            # 如果未提供自定义的归一化层，则使用默认的层归一化层，设置标准化的epsilon值与PyTorch相同
            self.norm = keras.layers.LayerNormalization(epsilon=1e-5, name="norm")
        else:
            self.norm = norm_layer(name="norm")  # 使用提供的自定义归一化层

    def maybe_pad(self, input_feature: tf.Tensor, height: int, width: int) -> tf.Tensor:
        should_pad = (height % 2 == 1) or (width % 2 == 1)
        if should_pad:
            pad_values = ((0, 0), (0, height % 2), (0, width % 2), (0, 0))  # 计算需要填充的值
            input_feature = tf.pad(input_feature, pad_values)  # 对输入特征进行填充

        return input_feature

    def call(self, input_feature: tf.Tensor, input_dimensions: Tuple[int, int], training: bool = False) -> tf.Tensor:
        height, width = input_dimensions
        batch_size, _, num_channels = shape_list(input_feature)  # 获取输入特征的形状信息

        input_feature = tf.reshape(input_feature, (batch_size, height, width, num_channels))  # 将输入特征重塑为四维张量
        input_feature = self.maybe_pad(input_feature, height, width)  # 可能对输入特征进行填充，使其尺寸可以被宽度和高度整除
        input_feature_0 = input_feature[:, 0::2, 0::2, :]  # 提取输入特征的每隔一个像素点的子集
        input_feature_1 = input_feature[:, 1::2, 0::2, :]  # 提取输入特征的每隔一个像素点的子集
        input_feature_2 = input_feature[:, 0::2, 1::2, :]  # 提取输入特征的每隔一个像素点的子集
        input_feature_3 = input_feature[:, 1::2, 1::2, :]  # 提取输入特征的每隔一个像素点的子集
        input_feature = tf.concat([input_feature_0, input_feature_1, input_feature_2, input_feature_3], -1)  # 合并这四个子集
        input_feature = tf.reshape(
            input_feature, (batch_size, -1, 4 * num_channels)
        )  # 将合并后的特征重塑为三维张量，以便进一步处理

        input_feature = self.norm(input_feature, training=training)  # 对特征进行归一化
        input_feature = self.reduction(input_feature, training=training)  # 对特征进行降维

        return input_feature
    # 定义 build 方法，用于构建模型，如果已经构建过，则直接返回
    def build(self, input_shape=None):
        # 检查是否已经构建过，如果是则返回，避免重复构建
        if self.built:
            return
        # 将标志设置为已构建
        self.built = True
        
        # 如果有指定的 reduction 属性，则在名为 reduction 的命名空间下构建
        if getattr(self, "reduction", None) is not None:
            with tf.name_scope(self.reduction.name):
                # 使用 4 * self.dim 的输入形状来构建 reduction 属性
                self.reduction.build([None, None, 4 * self.dim])
        
        # 如果有指定的 norm 属性，则在名为 norm 的命名空间下构建
        if getattr(self, "norm", None) is not None:
            with tf.name_scope(self.norm.name):
                # 使用 4 * self.dim 的输入形状来构建 norm 属性
                self.norm.build([None, None, 4 * self.dim])
class TFSwinDropPath(keras.layers.Layer):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: float = None, scale_by_keep: bool = True, **kwargs) -> None:
        super(TFSwinDropPath, self).__init__(**kwargs)
        self.drop_prob = drop_prob  # 初始化丢弃概率
        self.scale_by_keep = scale_by_keep  # 是否按保留比例缩放

    def call(self, input: tf.Tensor, training: bool = False) -> tf.Tensor:
        # 调用 drop_path 函数来应用丢弃路径操作
        return drop_path(input, self.drop_prob, training, self.scale_by_keep)


class TFSwinSelfAttention(keras.layers.Layer):
    def __init__(self, config: SwinConfig, dim: int, num_heads: int, **kwargs) -> None:
        super().__init__(**kwargs)
        if dim % num_heads != 0:
            raise ValueError(
                f"The hidden size ({dim}) is not a multiple of the number of attention heads ({num_heads})"
            )

        self.num_attention_heads = num_heads  # 设置注意力头数
        self.attention_head_size = int(dim / num_heads)  # 计算每个注意力头的大小
        self.all_head_size = self.num_attention_heads * self.attention_head_size  # 总的 QKV 大小
        window_size = config.window_size
        self.window_size = (
            window_size if isinstance(window_size, collections.abc.Iterable) else (window_size, window_size)
        )  # 窗口大小

        self.query = keras.layers.Dense(
            self.all_head_size,
            kernel_initializer=get_initializer(config.initializer_range),
            use_bias=config.qkv_bias,
            name="query",
        )  # 查询向量的全连接层

        self.key = keras.layers.Dense(
            self.all_head_size,
            kernel_initializer=get_initializer(config.initializer_range),
            use_bias=config.qkv_bias,
            name="key",
        )  # 键向量的全连接层

        self.value = keras.layers.Dense(
            self.all_head_size,
            kernel_initializer=get_initializer(config.initializer_range),
            use_bias=config.qkv_bias,
            name="value",
        )  # 值向量的全连接层

        self.dropout = keras.layers.Dropout(config.attention_probs_dropout_prob)  # 注意力概率的 dropout 层
    def build(self, input_shape: tf.TensorShape) -> None:
        # 创建一个用于存储相对位置偏置表的权重变量
        self.relative_position_bias_table = self.add_weight(
            shape=(((2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1)), self.num_attention_heads),
            initializer="zeros",
            name="relative_position_bias_table",
        )
        # 创建一个用于存储相对位置索引的权重变量，这些索引是窗口内每个标记的相对位置
        self.relative_position_index = self.add_weight(
            shape=(self.window_size[0] ** 2, self.window_size[1] ** 2),
            trainable=False,
            dtype=tf.int32,
            name="relative_position_index",
        )

        # 获取窗口内每个标记的成对相对位置索引
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

        # 计算相对位置索引的总和并分配给相对位置索引变量
        self.relative_position_index.assign(tf.cast(tf.reduce_sum(relative_coords, axis=-1), tf.int32))

        # 如果已经构建过，则直接返回
        if self.built:
            return
        # 标记模型已经构建
        self.built = True
        # 如果存在查询、键、值变量，则构建它们的结构
        if getattr(self, "query", None) is not None:
            with tf.name_scope(self.query.name):
                self.query.build([None, None, self.all_head_size])
        if getattr(self, "key", None) is not None:
            with tf.name_scope(self.key.name):
                self.key.build([None, None, self.all_head_size])
        if getattr(self, "value", None) is not None:
            with tf.name_scope(self.value.name):
                self.value.build([None, None, self.all_head_size])

    def transpose_for_scores(self, x: tf.Tensor) -> tf.Tensor:
        # 调整张量的形状以便计算注意力分数
        new_x_shape = shape_list(x)[:-1] + [self.num_attention_heads, self.attention_head_size]
        x = tf.reshape(x, new_x_shape)
        return tf.transpose(x, (0, 2, 1, 3))

    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: tf.Tensor | None = None,
        head_mask: tf.Tensor | None = None,
        output_attentions: bool = False,
        training: bool = False,
    ) -> Tuple[tf.Tensor, ...]:
        # 获取隐藏状态的形状信息：批大小、维度等
        batch_size, dim, _ = shape_list(hidden_states)
        # 对隐藏状态进行查询操作，生成混合的查询层
        mixed_query_layer = self.query(hidden_states)

        # 使用self.key对隐藏状态进行键的转换，并调整形状以适应注意力得分计算
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        # 使用self.value对隐藏状态进行值的转换，并调整形状以适应注意力得分计算
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        # 对混合的查询层进行形状调整，以适应注意力得分计算
        query_layer = self.transpose_for_scores(mixed_query_layer)

        # 计算查询层与键层之间的点积，得到原始的注意力得分
        attention_scores = tf.matmul(query_layer, tf.transpose(key_layer, (0, 1, 3, 2)))

        # 对注意力得分进行缩放，以减少数值大小对 softmax 函数计算的影响
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        # 根据相对位置索引从相对位置偏置表中获取相对位置偏置
        relative_position_bias = tf.gather(
            self.relative_position_bias_table, tf.reshape(self.relative_position_index, (-1,))
        )
        # 调整相对位置偏置的形状以匹配注意力得分的形状
        relative_position_bias = tf.reshape(
            relative_position_bias,
            (self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1),
        )
        # 转置相对位置偏置的维度顺序，以便与注意力得分相加
        relative_position_bias = tf.transpose(relative_position_bias, (2, 0, 1))
        attention_scores = attention_scores + tf.expand_dims(relative_position_bias, 0)

        # 如果存在注意力掩码，则应用它
        if attention_mask is not None:
            # 获取注意力掩码的形状信息
            mask_shape = shape_list(attention_mask)[0]
            # 调整注意力得分的形状以匹配掩码的形状
            attention_scores = tf.reshape(
                attention_scores, (batch_size // mask_shape, mask_shape, self.num_attention_heads, dim, dim)
            )
            # 扩展注意力掩码的维度以匹配注意力得分
            attention_mask = tf.expand_dims(attention_mask, 1)
            attention_mask = tf.expand_dims(attention_mask, 0)
            # 将注意力掩码加到注意力得分上
            attention_scores = attention_scores + attention_mask
            # 重新调整注意力得分的形状
            attention_scores = tf.reshape(attention_scores, (-1, self.num_attention_heads, dim, dim))

        # 对注意力得分进行 softmax 归一化，得到注意力概率
        attention_probs = tf.nn.softmax(attention_scores, axis=-1)

        # 使用 dropout 进行注意力概率的随机失活，仅在训练时生效
        attention_probs = self.dropout(attention_probs, training=training)

        # 如果指定了头部掩码，则应用头部掩码
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        # 计算上下文层，将注意力概率乘以值层
        context_layer = tf.matmul(attention_probs, value_layer)
        # 调整上下文层的维度顺序，以适应输出格式
        context_layer = tf.transpose(context_layer, (0, 2, 1, 3))
        # 调整上下文层的形状以匹配所有头部的输出大小
        new_context_layer_shape = shape_list(context_layer)[:-2] + [
            self.all_head_size,
        ]
        context_layer = tf.reshape(context_layer, new_context_layer_shape)

        # 输出结果，包括上下文层和可能的注意力概率
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs
# 定义一个名为 TFSwinSelfOutput 的自定义层，继承自 Keras 的 Layer 类
class TFSwinSelfOutput(keras.layers.Layer):
    # 初始化方法，接受 SwinConfig 对象、整数 dim 和额外的关键字参数
    def __init__(self, config: SwinConfig, dim: int, **kwargs) -> None:
        super().__init__(**kwargs)
        # 创建一个 Dense 层，用于线性变换，输出维度为 dim
        self.dense = keras.layers.Dense(dim, name="dense")
        # 创建一个 Dropout 层，使用配置中的 dropout 概率
        self.dropout = keras.layers.Dropout(config.attention_probs_dropout_prob, name="dropout")
        self.dim = dim

    # 前向传播方法，接受 hidden_states（输入张量）、input_tensor（输入张量）、training（布尔值，指示是否处于训练模式）
    def call(self, hidden_states: tf.Tensor, input_tensor: tf.Tensor, training: bool = False) -> tf.Tensor:
        # 将输入通过 Dense 层进行线性变换
        hidden_states = self.dense(hidden_states)
        # 对线性变换后的结果进行 Dropout 操作
        hidden_states = self.dropout(hidden_states, training=training)
        return hidden_states

    # 构建方法，用于构建层的内部结构
    def build(self, input_shape=None):
        # 如果层已经构建，则直接返回
        if self.built:
            return
        self.built = True
        # 如果存在 Dense 层，则构建该层
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.dim])
        # 如果存在 Dropout 层，则构建该层
        if getattr(self, "dropout", None) is not None:
            with tf.name_scope(self.dropout.name):
                self.dropout.build(None)


# 定义一个名为 TFSwinAttention 的自定义层，继承自 Keras 的 Layer 类
class TFSwinAttention(keras.layers.Layer):
    # 初始化方法，接受 SwinConfig 对象、整数 dim、整数 num_heads 和额外的关键字参数
    def __init__(self, config: SwinConfig, dim: int, num_heads: int, **kwargs) -> None:
        super().__init__(**kwargs)
        # 创建一个 TFSwinSelfAttention 层，用于处理注意力机制
        self.self = TFSwinSelfAttention(config, dim, num_heads, name="self")
        # 创建一个 TFSwinSelfOutput 层，用于处理自注意力输出
        self.self_output = TFSwinSelfOutput(config, dim, name="output")
        # 初始化一个空集合，用于存储要剪枝的注意力头
        self.pruned_heads = set()

    # 剪枝注意力头的方法，抛出未实现异常
    def prune_heads(self, heads):
        """
        Prunes heads of the model. See base class PreTrainedModel heads: dict of {layer_num: list of heads to prune in
        this layer}
        """
        raise NotImplementedError

    # 前向传播方法，接受 hidden_states（输入张量）、attention_mask（注意力掩码张量）、head_mask（头部掩码张量）、
    # output_attentions（布尔值，指示是否输出注意力矩阵）、training（布尔值，指示是否处于训练模式）
    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: tf.Tensor | None = None,
        head_mask: tf.Tensor | None = None,
        output_attentions: bool = False,
        training: bool = False,
    ) -> tf.Tensor:
        # 使用 self 层处理输入的 hidden_states，得到自注意力输出 self_outputs
        self_outputs = self.self(hidden_states, attention_mask, head_mask, output_attentions, training=training)
        # 使用 self_output 层处理 self_outputs 和原始 hidden_states，得到注意力输出 attention_output
        attention_output = self.self_output(self_outputs[0], hidden_states, training=training)
        # 构建输出元组 outputs，包括注意力输出和可能的注意力矩阵（如果有的话）
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs

    # 构建方法，用于构建层的内部结构
    def build(self, input_shape=None):
        # 如果层已经构建，则直接返回
        if self.built:
            return
        self.built = True
        # 如果存在 self 层，则构建该层
        if getattr(self, "self", None) is not None:
            with tf.name_scope(self.self.name):
                self.self.build(None)
        # 如果存在 self_output 层，则构建该层
        if getattr(self, "self_output", None) is not None:
            with tf.name_scope(self.self_output.name):
                self.self_output.build(None)


# 定义一个名为 TFSwinIntermediate 的自定义层，继承自 Keras 的 Layer 类
class TFSwinIntermediate(keras.layers.Layer):
    # 初始化方法，用于创建一个新的实例
    def __init__(self, config: SwinConfig, dim: int, **kwargs) -> None:
        # 调用父类（tf.keras.layers.Layer）的初始化方法
        super().__init__(**kwargs)
        # 创建一个全连接层，输出维度为 config.mlp_ratio * dim，命名为 "dense"
        self.dense = keras.layers.Dense(int(config.mlp_ratio * dim), name="dense")
        
        # 根据配置文件中的 hidden_act 参数确定中间激活函数
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act
        
        # 将维度信息保存在实例变量 dim 中
        self.dim = dim

    # 调用方法，定义了该层的正向传播逻辑
    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        # 通过全连接层处理输入的 hidden_states，得到输出 hidden_states
        hidden_states = self.dense(hidden_states)
        # 使用中间激活函数处理输出 hidden_states
        hidden_states = self.intermediate_act_fn(hidden_states)
        # 返回处理后的 hidden_states
        return hidden_states

    # 构建方法，用于构建层的变量（如果尚未构建）
    def build(self, input_shape=None):
        # 如果已经构建过，直接返回
        if self.built:
            return
        # 设置标志位，表明已经构建过
        self.built = True
        
        # 如果存在全连接层 dense，则根据输入形状构建该层
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                # 调用全连接层的 build 方法，指定输入形状 [None, None, self.dim]
                self.dense.build([None, None, self.dim])
# 定义一个名为 TFSwinOutput 的自定义层，继承自 keras 的 Layer 类
class TFSwinOutput(keras.layers.Layer):
    
    # 初始化方法，接受 SwinConfig 对象、维度 dim 和其他关键字参数
    def __init__(self, config: SwinConfig, dim: int, **kwargs) -> None:
        super().__init__(**kwargs)
        # 创建一个全连接层 dense，输出维度为 dim，命名为 "dense"
        self.dense = keras.layers.Dense(dim, name="dense")
        # 创建一个 Dropout 层，使用 SwinConfig 中的隐藏层 Dropout 概率作为参数，命名为 "dropout"
        self.dropout = keras.layers.Dropout(config.hidden_dropout_prob, name="dropout")
        # 将传入的 SwinConfig 对象保存到 self.config 中
        self.config = config
        # 将传入的维度 dim 保存到 self.dim 中

        self.dim = dim

    # 定义 call 方法，接收隐藏状态 hidden_states 和训练标志 training
    def call(self, hidden_states: tf.Tensor, training: bool = False) -> tf.Tensor:
        # 将隐藏状态输入到全连接层 dense 中，得到输出 hidden_states
        hidden_states = self.dense(hidden_states)
        # 对输出 hidden_states 应用 Dropout 操作，使用 training 参数控制是否训练模式
        hidden_states = self.dropout(hidden_states, training=training)
        # 返回经过全连接层和 Dropout 后的 hidden_states

        return hidden_states

    # 定义 build 方法，用于构建层的参数
    def build(self, input_shape=None):
        # 如果已经构建过，直接返回
        if self.built:
            return
        # 标记为已构建
        self.built = True
        # 检查是否存在 self.dense 属性
        if getattr(self, "dense", None) is not None:
            # 在命名空间 self.dense.name 下，构建全连接层，输入形状为 [None, None, int(self.config.mlp_ratio * self.dim)]
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, int(self.config.mlp_ratio * self.dim)])


# 定义一个名为 TFSwinLayer 的自定义层，继承自 keras 的 Layer 类
class TFSwinLayer(keras.layers.Layer):
    
    # 初始化方法，接受 config 对象、维度 dim、输入分辨率 input_resolution、注意力头数 num_heads 和其他关键字参数
    def __init__(
        self, config, dim, input_resolution: Tuple[int, int], num_heads: int, shift_size: int = 0, **kwargs
    ) -> None:
        super().__init__(**kwargs)
        # 设置前馈传输块的大小为 config 中的 chunk_size_feed_forward
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        # 计算输入分辨率的最小值
        min_res = tf.reduce_min(input_resolution)
        # 窗口大小为最小分辨率和 config 中的 window_size 的较小值
        self.window_size = min_res if min_res <= config.window_size else config.window_size
        # 如果最小分辨率小于等于窗口大小，则 shift_size 设为 0；否则使用传入的 shift_size
        self.shift_size = 0 if min_res <= self.window_size else shift_size
        # 保存输入分辨率到 self.input_resolution 中
        self.input_resolution = input_resolution

        # 创建 LayerNormalization 层，epsilon 使用 config 中的 layer_norm_eps，命名为 "layernorm_before"
        self.layernorm_before = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="layernorm_before")
        # 创建注意力机制层 TFSwinAttention，使用传入的 config、dim 和 num_heads，命名为 "attention"
        self.attention = TFSwinAttention(config, dim, num_heads, name="attention")
        # 如果 config 中的 drop_path_rate 大于 0.0，则创建 TFSwinDropPath 层，命名为 "drop_path"，否则使用线性激活层
        self.drop_path = (
            TFSwinDropPath(config.drop_path_rate, name="drop_path")
            if config.drop_path_rate > 0.0
            else keras.layers.Activation("linear", name="drop_path")
        )
        # 创建 LayerNormalization 层，epsilon 使用 config 中的 layer_norm_eps，命名为 "layernorm_after"
        self.layernorm_after = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="layernorm_after")
        # 创建 Swin 模型的中间层 TFSwinIntermediate，使用 config 和 dim，命名为 "intermediate"
        self.intermediate = TFSwinIntermediate(config, dim, name="intermediate")
        # 创建 Swin 模型的输出层 TFSwinOutput，使用 config 和 dim，命名为 "output"
        self.swin_output = TFSwinOutput(config, dim, name="output")
        # 保存维度 dim 到 self.dim 中
        self.dim = dim
    def get_attn_mask(self, height: int, width: int, window_size: int, shift_size: int) -> tf.Tensor | None:
        # 创建一个全零的图像掩码，形状为(height, width)
        img_mask = tf.zeros((height, width))
        # 定义高度和宽度的切片范围，用于创建注意力掩码
        height_slices = ((0, -window_size), (-window_size, -shift_size), (-shift_size, -1))
        width_slices = ((0, -window_size), (-window_size, -shift_size), (-shift_size, -1))

        # 计算 SW-MSA 的注意力掩码
        if shift_size > 0:
            count = 0
            for height_slice in height_slices:
                for width_slice in width_slices:
                    # 计算当前切片内的索引
                    height_inds = tf.range(height_slice[0] % height, height_slice[1] % height + 1)
                    width_inds = tf.range(width_slice[0] % width, width_slice[1] % width + 1)
                    indices = tf.reshape(tf.stack(tf.meshgrid(height_inds, width_inds), axis=-1), (-1, 2))
                    if len(indices) >= 1:
                        # 将更新值为 count 的掩码应用到图像掩码的对应位置
                        updates = tf.ones((len(indices),), dtype=img_mask.dtype) * count
                        img_mask = tf.tensor_scatter_nd_update(img_mask, indices, updates)
                    count += 1

        # 将图像掩码扩展维度以适应后续计算要求
        img_mask = tf.expand_dims(img_mask, -1)
        img_mask = tf.expand_dims(img_mask, 0)

        # 对图像掩码进行窗口划分，用于后续的注意力计算
        mask_windows = window_partition(img_mask, window_size)
        mask_windows = tf.reshape(mask_windows, (-1, window_size * window_size))
        # 构建注意力掩码，对角线上的元素为 -100.0，其余为 0.0
        attn_mask = tf.expand_dims(mask_windows, 1) - tf.expand_dims(mask_windows, 2)
        attn_mask = tf.where(attn_mask != 0, float(-100.0), attn_mask)
        attn_mask = tf.where(attn_mask == 0, float(0.0), attn_mask)
        return attn_mask

    def maybe_pad(
        self, hidden_states: tf.Tensor, window_size: int, height: int, width: int
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        # 计算需要在图像状态中填充的右边和底部的像素数
        pad_right = (window_size - width % window_size) % window_size
        pad_bottom = (window_size - height % window_size) % window_size
        # 定义填充的数值，填充右边和底部，保持其他维度不变
        pad_values = [[0, 0], [0, pad_bottom], [0, pad_right], [0, 0]]
        # 在隐藏状态张量上应用填充
        hidden_states = tf.pad(hidden_states, pad_values)
        # 将填充值转换为一维张量返回
        pad_values = tf.reshape(pad_values, (-1,))
        return hidden_states, pad_values

    def call(
        self,
        hidden_states: tf.Tensor,
        input_dimensions: Tuple[int, int],
        head_mask: tf.Tensor | None = None,
        output_attentions: bool = False,
        training: bool = False,
    ):
        # 神经网络层的调用函数，处理输入的隐藏状态和其他参数
    ) -> tf.Tensor:
        # 如果窗口大小大于输入分辨率，则不分割窗口
        min_res = tf.reduce_min(input_dimensions)  # 计算输入维度的最小值
        shift_size = 0 if min_res <= self.window_size else self.shift_size  # 如果最小分辨率小于等于窗口大小，则不进行移动；否则使用预设的移动大小
        window_size = min_res if min_res <= self.window_size else self.window_size  # 窗口大小取决于最小分辨率和设定的窗口大小

        height, width = input_dimensions  # 解包输入维度
        batch_size, _, channels = shape_list(hidden_states)  # 获取隐藏状态的批处理大小、高度、宽度和通道数
        shortcut = hidden_states  # 备份隐藏状态

        hidden_states = self.layernorm_before(hidden_states, training=training)  # 应用层归一化到隐藏状态之前
        hidden_states = tf.reshape(hidden_states, (batch_size, height, width, channels))  # 重新调整隐藏状态的形状为(batch_size, height, width, channels)
        hidden_states, pad_values = self.maybe_pad(hidden_states, window_size, height, width)  # 可能对隐藏状态进行填充，使其成为窗口大小的倍数

        _, height_pad, width_pad, _ = shape_list(hidden_states)  # 获取调整后隐藏状态的形状
        # 循环移位
        if shift_size > 0:
            shifted_hidden_states = tf.roll(hidden_states, shift=(-shift_size, -shift_size), axis=(1, 2))  # 在轴(1, 2)上执行负移位
        else:
            shifted_hidden_states = hidden_states  # 否则不进行移位

        # 分割窗口
        hidden_states_windows = window_partition(shifted_hidden_states, window_size)  # 将移位后的隐藏状态分割成窗口
        hidden_states_windows = tf.reshape(hidden_states_windows, (-1, window_size * window_size, channels))  # 重新调整窗口的形状为(-1, window_size * window_size, channels)
        attn_mask = self.get_attn_mask(
            height=height_pad, width=width_pad, window_size=window_size, shift_size=shift_size
        )  # 获取注意力掩码

        attention_outputs = self.attention(
            hidden_states_windows, attn_mask, head_mask, output_attentions=output_attentions, training=training
        )  # 应用自注意力机制

        attention_output = attention_outputs[0]  # 提取注意力输出的第一个元素

        attention_windows = tf.reshape(attention_output, (-1, window_size, window_size, channels))  # 重新调整注意力输出的形状为(-1, window_size, window_size, channels)
        shifted_windows = window_reverse(attention_windows, window_size, height_pad, width_pad)  # 反转窗口

        # 反向循环移位
        if shift_size > 0:
            attention_windows = tf.roll(shifted_windows, shift=(shift_size, shift_size), axis=(1, 2))  # 在轴(1, 2)上执行正移位
        else:
            attention_windows = shifted_windows  # 否则不进行移位

        was_padded = pad_values[3] > 0 or pad_values[5] > 0  # 检查是否对隐藏状态进行了填充
        if was_padded:
            attention_windows = attention_windows[:, :height, :width, :]  # 如果进行了填充，则截取有效部分

        attention_windows = tf.reshape(attention_windows, (batch_size, height * width, channels))  # 重新调整注意力窗口的形状为(batch_size, height * width, channels)

        hidden_states = shortcut + self.drop_path(attention_windows, training=training)  # 添加残差连接和DropPath

        layer_output = self.layernorm_after(hidden_states, training=training)  # 应用层归一化到隐藏状态之后
        layer_output = self.intermediate(layer_output)  # 应用中间层变换
        layer_output = hidden_states + self.swin_output(layer_output, training=training)  # 添加Swin Transformer的输出

        layer_outputs = (layer_output, attention_outputs[1]) if output_attentions else (layer_output,)  # 构造输出元组

        return layer_outputs  # 返回层输出
    # 构建模型的方法，用于设置层的输入形状并构建层的参数
    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回，避免重复构建
        if self.built:
            return
        # 将构建标志设置为已构建
        self.built = True
        
        # 如果存在layernorm_before属性，则构建layernorm_before层
        if getattr(self, "layernorm_before", None) is not None:
            # 使用layernorm_before层的名字作为命名空间
            with tf.name_scope(self.layernorm_before.name):
                # 构建layernorm_before层，设置输入形状为[None, None, self.dim]
                self.layernorm_before.build([None, None, self.dim])
        
        # 如果存在attention属性，则构建attention层
        if getattr(self, "attention", None) is not None:
            # 使用attention层的名字作为命名空间
            with tf.name_scope(self.attention.name):
                # 构建attention层，输入形状为None（表示不确定的形状）
                self.attention.build(None)
        
        # 如果存在drop_path属性，则构建drop_path层
        if getattr(self, "drop_path", None) is not None:
            # 使用drop_path层的名字作为命名空间
            with tf.name_scope(self.drop_path.name):
                # 构建drop_path层，输入形状为None
                self.drop_path.build(None)
        
        # 如果存在layernorm_after属性，则构建layernorm_after层
        if getattr(self, "layernorm_after", None) is not None:
            # 使用layernorm_after层的名字作为命名空间
            with tf.name_scope(self.layernorm_after.name):
                # 构建layernorm_after层，设置输入形状为[None, None, self.dim]
                self.layernorm_after.build([None, None, self.dim])
        
        # 如果存在intermediate属性，则构建intermediate层
        if getattr(self, "intermediate", None) is not None:
            # 使用intermediate层的名字作为命名空间
            with tf.name_scope(self.intermediate.name):
                # 构建intermediate层，输入形状为None
                self.intermediate.build(None)
        
        # 如果存在swin_output属性，则构建swin_output层
        if getattr(self, "swin_output", None) is not None:
            # 使用swin_output层的名字作为命名空间
            with tf.name_scope(self.swin_output.name):
                # 构建swin_output层，输入形状为None
                self.swin_output.build(None)
class TFSwinStage(keras.layers.Layer):
    # 定义一个名为 TFSwinStage 的自定义 Keras 层
    def __init__(
        self,
        config: SwinConfig,
        dim: int,
        input_resolution: Tuple[int, int],
        depth: int,
        num_heads: int,
        drop_path: List[float],
        downsample: Optional[Callable],
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        # 初始化函数，接受多个参数，其中包括 Swin 模型的配置、维度、输入分辨率、深度、头数、路径丢弃率等
        self.config = config
        self.dim = dim
        # 创建一个由 TFSwinLayer 实例组成的列表，每个实例代表一个层
        self.blocks = [
            TFSwinLayer(
                config=config,
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                shift_size=0 if (i % 2 == 0) else config.window_size // 2,
                name=f"blocks.{i}",
            )
            for i in range(depth)
        ]

        # 如果存在下采样函数，创建下采样层
        if downsample is not None:
            self.downsample = downsample(
                input_resolution,
                dim=dim,
                norm_layer=partial(keras.layers.LayerNormalization, epsilon=1e-5),
                name="downsample",
            )
        else:
            self.downsample = None

        # 初始化指向（pointing）为 False
        self.pointing = False

    # 定义调用函数，处理输入并返回输出
    def call(
        self,
        hidden_states: tf.Tensor,
        input_dimensions: Tuple[int, int],
        head_mask: tf.Tensor | None = None,
        output_attentions: Optional[bool] = False,
        training: bool = False,
    ) -> Tuple[tf.Tensor, ...]:
        height, width = input_dimensions
        # 遍历所有层，逐层处理隐藏状态
        for i, layer_module in enumerate(self.blocks):
            layer_head_mask = head_mask[i] if head_mask is not None else None

            # 调用每个层的处理函数，获取层的输出
            layer_outputs = layer_module(
                hidden_states, input_dimensions, layer_head_mask, output_attentions, training=training
            )

            # 更新隐藏状态为当前层的输出
            hidden_states = layer_outputs[0]

        # 如果存在下采样层，对隐藏状态进行下采样操作
        if self.downsample is not None:
            height_downsampled, width_downsampled = (height + 1) // 2, (width + 1) // 2
            output_dimensions = (height, width, height_downsampled, width_downsampled)
            hidden_states = self.downsample(layer_outputs[0], input_dimensions, training=training)
        else:
            output_dimensions = (height, width, height, width)

        # 组装阶段的输出，包括隐藏状态和输出尺寸
        stage_outputs = (hidden_states, output_dimensions)

        # 如果需要输出注意力权重，则将它们添加到阶段的输出中
        if output_attentions:
            stage_outputs += layer_outputs[1:]
        return stage_outputs

    # 定义构建函数，在第一次调用时构建层
    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 如果存在下采样层，构建该层
        if getattr(self, "downsample", None) is not None:
            with tf.name_scope(self.downsample.name):
                self.downsample.build(None)
        # 对每个层调用构建函数，构建所有的子层
        if getattr(self, "blocks", None) is not None:
            for layer in self.blocks:
                with tf.name_scope(layer.name):
                    layer.build(None)


class TFSwinEncoder(keras.layers.Layer):
    # 定义一个名为 TFSwinEncoder 的自定义 Keras 层
    # 初始化函数，接受一个SwinTransformer的配置对象和一个网格大小的元组作为参数
    def __init__(self, config: SwinConfig, grid_size: Tuple[int, int], **kwargs):
        # 调用父类的初始化函数
        super().__init__(**kwargs)
        # 计算SwinTransformer模型的层数
        self.num_layers = len(config.depths)
        # 保存传入的配置对象
        self.config = config
        # 计算每一层的DropPath率，并转换为列表
        dpr = list((tf.linspace(0, 1, sum(config.depths)) * config.drop_path_rate).numpy())
        
        # 创建SwinTransformer的各个层
        self.layers = [
            TFSwinStage(
                config=config,
                # 计算当前层的维度
                dim=int(config.embed_dim * 2**i_layer),
                # 计算当前层的输入分辨率
                input_resolution=(grid_size[0] // (2**i_layer), grid_size[1] // (2**i_layer)),
                # 设置当前层的深度
                depth=config.depths[i_layer],
                # 设置当前层的头数
                num_heads=config.num_heads[i_layer],
                # 为当前层设置DropPath率
                drop_path=dpr[sum(config.depths[:i_layer]) : sum(config.depths[: i_layer + 1])],
                # 如果当前层不是最后一层，设置下采样方法；否则为None
                downsample=TFSwinPatchMerging if (i_layer < self.num_layers - 1) else None,
                # 设置当前层的名称
                name=f"layers.{i_layer}",
            )
            # 对每一层进行迭代
            for i_layer in range(self.num_layers)
        ]
        
        # 默认关闭梯度检查点
        self.gradient_checkpointing = False

    # 模型调用函数，接受隐藏状态张量、输入维度元组等多个参数
    def call(
        self,
        hidden_states: tf.Tensor,
        input_dimensions: Tuple[int, int],
        head_mask: tf.Tensor | None = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        training: bool = False,
    ) -> Union[Tuple[tf.Tensor, ...], TFSwinEncoderOutput]:
        # 定义函数签名及返回类型，输入为隐藏状态及其他参数，输出为元组或TFSwinEncoderOutput类型
        all_input_dimensions = ()
        # 初始化空元组，用于存储所有输入维度信息
        all_hidden_states = () if output_hidden_states else None
        # 如果需要输出隐藏状态，则初始化空元组，否则置为None
        all_reshaped_hidden_states = () if output_hidden_states else None
        # 如果需要输出隐藏状态，则初始化空元组，否则置为None
        all_self_attentions = () if output_attentions else None
        # 如果需要输出注意力权重，则初始化空元组，否则置为None

        if output_hidden_states:
            batch_size, _, hidden_size = shape_list(hidden_states)
            # 获取隐藏状态的批量大小、高、宽、通道数信息
            # 重排形状为 b (h w) c -> b c h w
            reshaped_hidden_state = tf.reshape(hidden_states, (batch_size, *input_dimensions, hidden_size))
            reshaped_hidden_state = tf.transpose(reshaped_hidden_state, (0, 3, 1, 2))
            # 将形状调整为 b c h w，并进行转置以匹配预期的维度顺序
            all_hidden_states += (hidden_states,)
            all_reshaped_hidden_states += (reshaped_hidden_state,)
            # 将隐藏状态及其重排后的形状信息添加到对应的元组中

        for i, layer_module in enumerate(self.layers):
            # 遍历self.layers中的每一层模块
            layer_head_mask = head_mask[i] if head_mask is not None else None
            # 获取当前层的注意力头遮罩，如果未提供则置为None

            layer_outputs = layer_module(
                hidden_states, input_dimensions, layer_head_mask, output_attentions, training=training
            )
            # 调用当前层模块的前向传播方法，计算层的输出结果

            hidden_states = layer_outputs[0]
            # 更新隐藏状态为当前层输出的第一个元素（通常是最终的隐藏状态）
            output_dimensions = layer_outputs[1]
            # 获取当前层输出的维度信息

            input_dimensions = (output_dimensions[-2], output_dimensions[-1])
            # 更新输入维度为当前层输出的高和宽信息
            all_input_dimensions += (input_dimensions,)
            # 将更新后的输入维度信息添加到all_input_dimensions中

            if output_hidden_states:
                batch_size, _, hidden_size = shape_list(hidden_states)
                # 获取隐藏状态的批量大小、高、宽、通道数信息
                # 重排形状为 b (h w) c -> b c h w
                reshaped_hidden_state = tf.reshape(hidden_states, (batch_size, *input_dimensions, hidden_size))
                reshaped_hidden_state = tf.transpose(reshaped_hidden_state, (0, 3, 1, 2))
                # 将形状调整为 b c h w，并进行转置以匹配预期的维度顺序
                all_hidden_states += (hidden_states,)
                all_reshaped_hidden_states += (reshaped_hidden_state,)
                # 将隐藏状态及其重排后的形状信息添加到对应的元组中

            if output_attentions:
                all_self_attentions += layer_outputs[2:]
                # 如果需要输出注意力权重，则将当前层输出中的注意力权重信息添加到all_self_attentions中

        if not return_dict:
            # 如果不需要返回字典格式的输出结果
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
            # 返回所有非空的结果组成的元组

        return TFSwinEncoderOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            reshaped_hidden_states=all_reshaped_hidden_states,
        )
        # 返回以TFSwinEncoderOutput格式封装的输出结果

    def build(self, input_shape=None):
        # 定义build方法，用于构建模型层次结构
        if self.built:
            # 如果模型已构建完成，则直接返回
            return
        self.built = True
        # 将模型标记为已构建
        if getattr(self, "layers", None) is not None:
            # 如果存在模型层列表
            for layer in self.layers:
                # 遍历每一层
                with tf.name_scope(layer.name):
                    # 使用层的名称创建命名空间
                    layer.build(None)
                    # 调用层的build方法构建层次结构
class TFSwinPreTrainedModel(TFPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 使用 SwinConfig 类作为模型的配置类
    config_class = SwinConfig
    # 基础模型的前缀名为 "swin"
    base_model_prefix = "swin"
    # 主输入名称为 "pixel_values"
    main_input_name = "pixel_values"


SWIN_START_DOCSTRING = r"""
    This model is a Tensorflow
    [keras.layers.Layer](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer) sub-class. Use it as a
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


def normalize_data_format(value: str) -> str:
    """
    From tensorflow addons
    https://github.com/tensorflow/addons/blob/8cec33fcaaf1cf90aec7bdd55a0fcdbb251ce5c2/tensorflow_addons/utils/keras_utils.py#L71
    """
    # 如果值为 None，则使用 keras 后端的图像数据格式作为值
    if value is None:
        value = keras.backend.image_data_format()
    # 将值转换为小写
    data_format = value.lower()
    # 如果数据格式不是 "channels_first" 或 "channels_last"，则引发 ValueError 异常
    if data_format not in {"channels_first", "channels_last"}:
        raise ValueError(
            'The `data_format` argument must be one of "channels_first", "channels_last". Received: ' + str(value)
        )
    # 返回标准化后的数据格式
    return data_format


class AdaptiveAveragePooling1D(keras.layers.Layer):
    """
    Args:
"""
    """
    Average 1D Pooling with adaptive kernel size.
    output_size: An integer or tuple/list of a single integer, specifying pooled_features.
    The new size of output channels.
    data_format: A string,
    one of `channels_last` (default) or `channels_first`. The ordering of the dimensions in the inputs.
    `channels_last` corresponds to inputs with shape `(batch, steps, channels)` while `channels_first` corresponds
    to inputs with shape `(batch, channels, steps)`.
    
    Input shape:
    - If `data_format='channels_last'`: 3D tensor with shape `(batch, steps, channels)`.
    - If `data_format='channels_first'`: 3D tensor with shape `(batch, channels, steps)`.
    
    Output shape:
    - If `data_format='channels_last'`: 3D tensor with shape `(batch_size, pooled_steps, channels)`.
    - If `data_format='channels_first'`: 3D tensor with shape `(batch_size, channels, pooled_steps)`.
    
    Adapted from [tensorflow-addon's adaptive pooling.py](
        https://github.com/tensorflow/addons/blob/8cec33fcaaf1cf90aec7bdd55a0fcdbb251ce5c2/tensorflow_addons/layers/adaptive_pooling.py#L90-L120
    )
    """
    
    # 定义一个平均池化层，支持自适应核大小
    class AveragePooling1D(tf.keras.layers.Layer):
    
        def __init__(
            self,
            output_size: Union[int, Iterable[int]],  # 池化后的输出尺寸，可以是整数或整数组成的可迭代对象
            reduce_function: Callable = tf.reduce_mean,  # 池化使用的函数，默认为平均值池化
            data_format: Optional[str] = None,  # 数据格式，默认为 None
            **kwargs,  # 其他参数
        ) -> None:
            self.data_format = normalize_data_format(data_format)  # 标准化数据格式
            self.reduce_function = reduce_function  # 池化函数
            self.output_size = (output_size,) if isinstance(output_size, int) else tuple(output_size)  # 输出尺寸的元组形式
            super().__init__(**kwargs)  # 调用父类初始化方法
    
        def call(self, inputs: tf.Tensor, *args) -> None:
            bins = self.output_size[0]  # 获取输出尺寸中的第一个值作为 bins
            if self.data_format == "channels_last":
                splits = tf.split(inputs, bins, axis=1)  # 在通道维度上分割输入张量
                splits = tf.stack(splits, axis=1)  # 在第二个维度上堆叠分割后的张量
                out_vect = self.reduce_function(splits, axis=2)  # 沿着第三个维度对堆叠后的张量进行池化
            else:
                splits = tf.split(inputs, bins, axis=2)  # 在时间步维度上分割输入张量
                splits = tf.stack(splits, axis=2)  # 在第三个维度上堆叠分割后的张量
                out_vect = self.reduce_function(splits, axis=3)  # 沿着第四个维度对堆叠后的张量进行池化
            return out_vect  # 返回池化后的张量
    
        def compute_output_shape(self, input_shape: Iterable[int]) -> tf.TensorShape:
            input_shape = tf.TensorShape(input_shape).as_list()  # 将输入形状转换为列表形式
            if self.data_format == "channels_last":
                shape = tf.TensorShape([input_shape[0], self.output_size[0], input_shape[2]])  # 计算输出形状，通道在最后
            else:
                shape = tf.TensorShape([input_shape[0], input_shape[1], self.output_size[0]])  # 计算输出形状，通道在中间
            return shape  # 返回输出形状的张量形状对象
    
        def get_config(self) -> Dict[str, Any]:
            config = {
                "output_size": self.output_size,  # 输出尺寸配置
                "data_format": self.data_format,  # 数据格式配置
            }
            base_config = super().get_config()  # 调用父类配置方法
            return {**base_config, **config}  # 返回合并后的配置字典
    # 定义一个 Keras 自定义层 TFSwinMainLayer，并添加了 keras_serializable 装饰器，使其能够序列化
    @keras_serializable
    class TFSwinMainLayer(keras.layers.Layer):
        # 设置配置类为 SwinConfig
        config_class = SwinConfig

        # 初始化函数，接受 SwinConfig 类型的 config 参数，以及其他可选参数
        def __init__(
            self, config: SwinConfig, add_pooling_layer: bool = True, use_mask_token: bool = False, **kwargs
        ) -> None:
            # 调用父类的初始化方法
            super().__init__(**kwargs)
            # 将传入的配置参数 config 赋值给对象的 config 属性
            self.config = config
            # 计算层数，即配置的深度列表的长度
            self.num_layers = len(config.depths)
            # 计算特征数，为配置中的嵌入维度乘以 2 的 (层数 - 1) 次方
            self.num_features = int(config.embed_dim * 2 ** (self.num_layers - 1))

            # 创建 TFSwinEmbeddings 对象，并赋值给 embeddings 属性
            self.embeddings = TFSwinEmbeddings(config, use_mask_token=use_mask_token, name="embeddings")
            # 创建 TFSwinEncoder 对象，并传入 patch_grid 参数和名称 "encoder"，赋值给 encoder 属性
            self.encoder = TFSwinEncoder(config, self.embeddings.patch_grid, name="encoder")

            # 创建 LayerNormalization 层，epsilon 参数为配置中的层归一化 epsilon 值，名称为 "layernorm"，赋值给 layernorm 属性
            self.layernorm = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="layernorm")
            
            # 如果 add_pooling_layer 为 True，则创建 AdaptiveAveragePooling1D 层，输出大小为 (1,)，赋值给 pooler 属性；否则 pooler 属性为 None
            self.pooler = AdaptiveAveragePooling1D(output_size=(1,)) if add_pooling_layer else None

        # 获取输入嵌入的方法，返回 embeddings 对象的 patch_embeddings 属性
        def get_input_embeddings(self) -> TFSwinPatchEmbeddings:
            return self.embeddings.patch_embeddings

        # 模型头部修剪方法，接受 heads_to_prune 参数，用于剪枝模型中的注意力头
        def _prune_heads(self, heads_to_prune: Dict[int, List]):
            """
            Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
            class PreTrainedModel
            """
            # 遍历 heads_to_prune 字典中的每一层和对应要剪枝的注意力头列表
            for layer, heads in heads_to_prune.items():
                # 在编码器（self.encoder）的指定层（layer）的注意力部分（attention）进行头部剪枝操作
                self.encoder.layer[layer].attention.prune_heads(heads)

        # 获取头部掩码的方法，接受 head_mask 参数，如果非空则抛出未实现错误，否则返回与深度列表长度相同的 None 列表
        def get_head_mask(self, head_mask: Optional[Any]) -> List:
            if head_mask is not None:
                raise NotImplementedError
            return [None] * len(self.config.depths)

        # 调用方法，接受多个参数并进行处理，包括像素值、掩码位置、头部掩码等
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
        # 如果未指定，则根据配置确定是否输出注意力权重
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 如果未指定，则根据配置确定是否输出隐藏状态
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 如果未指定，则根据配置确定是否返回字典格式的输出
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 如果像素值为空，则抛出数值错误异常
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # 准备头部掩码（如果需要）
        # head_mask 中的 1.0 表示保留对应的注意力头部
        # attention_probs 的形状为 bsz x n_heads x N x N
        # 输入的 head_mask 形状为 [num_heads] 或者 [num_hidden_layers x num_heads]
        # head_mask 被转换为形状 [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask)
        
        # 将像素值传入嵌入层，并获取嵌入层的输出和输入维度
        embedding_output, input_dimensions = self.embeddings(
            pixel_values, bool_masked_pos=bool_masked_pos, training=training
        )

        # 将嵌入层的输出传入编码器，并返回编码器的输出
        encoder_outputs = self.encoder(
            embedding_output,
            input_dimensions,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        # 获取编码器的序列输出，并进行 layer normalization
        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm(sequence_output, training=training)

        # 初始化池化输出为 None
        pooled_output = None
        # 如果池化器不为空，则对序列输出进行池化
        if self.pooler is not None:
            batch_size, _, num_features = shape_list(sequence_output)
            pooled_output = self.pooler(sequence_output)
            pooled_output = tf.reshape(pooled_output, (batch_size, num_features))

        # 如果不需要返回字典，则返回输出元组
        if not return_dict:
            output = (sequence_output, pooled_output) + encoder_outputs[1:]
            return output

        # 如果需要返回字典格式的输出，则构建 TFSwinModelOutput 对象
        return TFSwinModelOutput(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            reshaped_hidden_states=encoder_outputs.reshaped_hidden_states,
        )

    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回
        if self.built:
            return
        # 标记已经构建
        self.built = True
        # 如果存在嵌入层，则构建嵌入层
        if getattr(self, "embeddings", None) is not None:
            with tf.name_scope(self.embeddings.name):
                self.embeddings.build(None)
        # 如果存在编码器，则构建编码器
        if getattr(self, "encoder", None) is not None:
            with tf.name_scope(self.encoder.name):
                self.encoder.build(None)
        # 如果存在层归一化，则构建层归一化
        if getattr(self, "layernorm", None) is not None:
            with tf.name_scope(self.layernorm.name):
                self.layernorm.build([None, None, self.num_features])
# 使用装饰器为类添加文档字符串，描述其作为裸的 Swin 模型变换器，输出未经任何特定头部处理的原始隐藏状态
@add_start_docstrings(
    "The bare Swin Model transformer outputting raw hidden-states without any specific head on top.",
    SWIN_START_DOCSTRING,
)
# 定义 TFSwinModel 类，继承自 TFSwinPreTrainedModel
class TFSwinModel(TFSwinPreTrainedModel):
    
    # 初始化方法
    def __init__(
        self, config: SwinConfig, add_pooling_layer: bool = True, use_mask_token: bool = False, **kwargs
    ) -> None:
        # 调用父类的初始化方法
        super().__init__(config, **kwargs)
        # 保存配置信息到实例变量
        self.config = config
        # 创建 TFSwinMainLayer 的实例 swin，并命名为 "swin"
        self.swin = TFSwinMainLayer(config, name="swin")

    # 为 call 方法添加文档字符串，描述其作为模型前向传播的入口点，使用 SWIN_INPUTS_DOCSTRING 作为输入文档字符串
    @add_start_docstrings_to_model_forward(SWIN_INPUTS_DOCSTRING)
    # 使用装饰器添加代码示例文档字符串，展示模型的使用示例
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFSwinModelOutput,
        config_class=_CONFIG_FOR_DOC,
        modality="vision",
        expected_output=_EXPECTED_OUTPUT_SHAPE,
    )
    # 使用装饰器解包输入，确保正确处理输入参数
    @unpack_inputs
    # 定义 call 方法，接收多个参数并返回 TFSwinModelOutput 或 tf.Tensor 元组
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
        # 根据需要确定是否输出注意力权重，默认使用配置中的设置
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 根据需要确定是否输出隐藏状态，默认使用配置中的设置
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 根据需要确定是否返回字典形式的输出，默认使用配置中的设置
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 如果未提供像素值，则引发值错误
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # 调用 self.swin 的前向传播方法，传递所有参数，并获取模型输出
        swin_outputs = self.swin(
            pixel_values=pixel_values,
            bool_masked_pos=bool_masked_pos,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        # 返回模型输出
        return swin_outputs

    # 实现 build 方法，用于构建模型层次结构
    def build(self, input_shape=None):
        # 如果已经构建过，直接返回
        if self.built:
            return
        # 标记为已构建
        self.built = True
        # 如果 self.swin 已存在，则在命名空间下构建 self.swin
        if getattr(self, "swin", None) is not None:
            with tf.name_scope(self.swin.name):
                self.swin.build(None)


# 定义 TFSwinPixelShuffle 类，继承自 keras.layers.Layer，实现了 torch.nn.PixelShuffle 的 TensorFlow 版本的层
class TFSwinPixelShuffle(keras.layers.Layer):
    """TF layer implementation of torch.nn.PixelShuffle"""

    # 初始化方法
    def __init__(self, upscale_factor: int, **kwargs) -> None:
        # 调用父类的初始化方法
        super().__init__(**kwargs)
        # 如果 upscale_factor 不是整数或小于 2，则引发值错误
        if not isinstance(upscale_factor, int) or upscale_factor < 2:
            raise ValueError(f"upscale_factor must be an integer value >= 2 got {upscale_factor}")
        # 保存 upscale_factor 到实例变量
        self.upscale_factor = upscale_factor
    # 定义一个方法，接受一个张量 x 作为输入，返回一个张量作为输出
    def call(self, x: tf.Tensor) -> tf.Tensor:
        # 将输入张量赋值给 hidden_states
        hidden_states = x
        # 调用 shape_list 函数获取 hidden_states 的形状信息，并解包得到 batch_size, _, _, num_input_channels
        batch_size, _, _, num_input_channels = shape_list(hidden_states)
        # 计算块大小的平方
        block_size_squared = self.upscale_factor**2
        # 计算输出深度，即 num_input_channels 除以块大小的平方后取整
        output_depth = int(num_input_channels / block_size_squared)
        # 创建一个常量张量 permutation，用于存储一个通道排列顺序的索引
        permutation = tf.constant(
            # 使用列表推导式生成的二维数组，每个元素是一个索引，按照不同通道和块的顺序排列
            [[i + j * block_size_squared for i in range(block_size_squared) for j in range(output_depth)]]
        )
        # 使用 tf.gather 函数根据 permutation 中的索引重新组织 hidden_states 的通道
        hidden_states = tf.gather(params=hidden_states, indices=tf.tile(permutation, [batch_size, 1]), batch_dims=-1)
        # 使用 tf.nn.depth_to_space 函数进行深度到空间的转换，根据 upscale_factor 参数进行块的重新排列
        hidden_states = tf.nn.depth_to_space(hidden_states, block_size=self.upscale_factor, data_format="NHWC")
        # 返回处理后的 hidden_states 作为结果
        return hidden_states
# 自定义的 TensorFlow 2.x 模型层，用于实现 TFSwin 模型的解码器部分
class TFSwinDecoder(keras.layers.Layer):
    def __init__(self, config: SwinConfig, **kwargs):
        super().__init__(**kwargs)
        # 定义一个 1x1 卷积层，用于特征变换
        self.conv2d = keras.layers.Conv2D(
            filters=config.encoder_stride**2 * config.num_channels, kernel_size=1, strides=1, name="0"
        )
        # 像素重排层，用于反向像素重排
        self.pixel_shuffle = TFSwinPixelShuffle(config.encoder_stride, name="1")
        # 保存 Swin 模型的配置信息
        self.config = config

    def call(self, x: tf.Tensor) -> tf.Tensor:
        # 将输入张量从 B,C,H,W 转置为 B,H,W,C
        hidden_states = x
        hidden_states = tf.transpose(hidden_states, (0, 2, 3, 1))
        # 经过 1x1 卷积层变换
        hidden_states = self.conv2d(hidden_states)
        # 经过像素重排层
        hidden_states = self.pixel_shuffle(hidden_states)
        # 将输出张量从 B,H,W,C 转置为 B,C,H,W
        hidden_states = tf.transpose(hidden_states, (0, 3, 1, 2))
        return hidden_states

    def build(self, input_shape=None):
        # 如果已经构建过，直接返回
        if self.built:
            return
        self.built = True
        # 构建卷积层
        if getattr(self, "conv2d", None) is not None:
            with tf.name_scope(self.conv2d.name):
                self.conv2d.build([None, None, None, self.config.hidden_size])
        # 构建像素重排层
        if getattr(self, "pixel_shuffle", None) is not None:
            with tf.name_scope(self.pixel_shuffle.name):
                self.pixel_shuffle.build(None)


# 基于 Swin 模型的一个变体，用于处理带掩码的图像建模，参考 SimMIM 论文提出的方法
@add_start_docstrings(
    "Swin Model with a decoder on top for masked image modeling, as proposed in"
    " [SimMIM](https://arxiv.org/abs/2111.09886).",
    SWIN_START_DOCSTRING,
)
class TFSwinForMaskedImageModeling(TFSwinPreTrainedModel):
    def __init__(self, config: SwinConfig):
        super().__init__(config)
        # Swin 主层，不包含池化层，使用掩码标记
        self.swin = TFSwinMainLayer(config, add_pooling_layer=False, use_mask_token=True, name="swin")
        # Swin 解码器层
        self.decoder = TFSwinDecoder(config, name="decoder")

    @add_start_docstrings_to_model_forward(SWIN_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFSwinMaskedImageModelingOutput, config_class=_CONFIG_FOR_DOC)
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
    ):
        # 略
        pass

    def build(self, input_shape=None):
        # 如果已经构建过，直接返回
        if self.built:
            return
        self.built = True
        # 构建 Swin 主层
        if getattr(self, "swin", None) is not None:
            with tf.name_scope(self.swin.name):
                self.swin.build(None)
        # 构建 Swin 解码器层
        if getattr(self, "decoder", None) is not None:
            with tf.name_scope(self.decoder.name):
                self.decoder.build(None)


# Swin 模型的图像分类变体，顶部附加了一个分类头部的线性层（在 [CLS] 标记的最终隐藏状态之上），例如用于 ImageNet
@add_start_docstrings(
    """
    Swin Model transformer with an image classification head on top (a linear layer on top of the final hidden state of
    the [CLS] token) e.g. for ImageNet.
    """,
    SWIN_START_DOCSTRING,
)
class TFSwinForImageClassification(TFSwinPreTrainedModel, TFSequenceClassificationLoss):
    # 略
    pass
    # 初始化函数，接受一个 SwinConfig 类型的配置对象作为参数
    def __init__(self, config: SwinConfig):
        # 调用父类的初始化方法
        super().__init__(config)

        # 设置类的属性，表示分类数目
        self.num_labels = config.num_labels
        # 创建一个 TFSwinMainLayer 类的实例，命名为 "swin"
        self.swin = TFSwinMainLayer(config, name="swin")

        # 分类器头部
        # 如果配置的标签数目大于 0，则创建一个全连接层作为分类器
        # 否则创建一个线性激活层作为分类器
        self.classifier = (
            keras.layers.Dense(config.num_labels, name="classifier")
            if config.num_labels > 0
            else keras.layers.Activation("linear", name="classifier")
        )

    # 根据装饰器提供的文档字符串，定义了模型前向传播的方法
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
        """
        labels (`tf.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        # 确定是否返回字典类型的输出
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用 Swin 模型的前向传播方法
        outputs = self.swin(
            pixel_values,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        # 获取池化后的输出
        pooled_output = outputs[1]

        # 将池化输出传递给分类器进行预测
        logits = self.classifier(pooled_output, training=training)

        # 如果有提供标签，则计算损失
        loss = None if labels is None else self.hf_compute_loss(labels, logits)

        # 如果不要求返回字典类型的输出，则按需返回输出的元组
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 否则返回 TFSwinImageClassifierOutput 类型的对象
        return TFSwinImageClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            reshaped_hidden_states=outputs.reshaped_hidden_states,
        )

    # 构建模型，设置模型的输入形状
    def build(self, input_shape=None):
        # 如果模型已经构建过，则直接返回
        if self.built:
            return
        # 标记模型已经构建
        self.built = True
        # 如果存在 Swin 层，则在其命名空间下构建 Swin 层
        if getattr(self, "swin", None) is not None:
            with tf.name_scope(self.swin.name):
                self.swin.build(None)
        # 如果存在分类器，则在其命名空间下构建分类器，并传入 Swin 特征数目作为输入形状的一部分
        if getattr(self, "classifier", None) is not None:
            if hasattr(self.classifier, "name"):
                with tf.name_scope(self.classifier.name):
                    self.classifier.build([None, None, self.swin.num_features])
```