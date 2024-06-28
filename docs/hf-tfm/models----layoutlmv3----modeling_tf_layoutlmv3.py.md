# `.\models\layoutlmv3\modeling_tf_layoutlmv3.py`

```py
# 导入必要的模块和类
from __future__ import annotations

import collections  # 导入collections模块，用于处理集合类型数据
import math  # 导入math模块，提供数学函数
from typing import List, Optional, Tuple, Union  # 导入类型注解相关的模块

import tensorflow as tf  # 导入TensorFlow库

from ...activations_tf import get_tf_activation  # 导入自定义的TensorFlow激活函数获取函数
from ...modeling_tf_outputs import (  # 导入TensorFlow模型输出相关类
    TFBaseModelOutput,
    TFQuestionAnsweringModelOutput,
    TFSequenceClassifierOutput,
    TFTokenClassifierOutput,
)
from ...modeling_tf_utils import (  # 导入TensorFlow模型工具函数
    TFPreTrainedModel,
    TFQuestionAnsweringLoss,
    TFSequenceClassificationLoss,
    TFTokenClassificationLoss,
    get_initializer,
    keras,
    keras_serializable,
    unpack_inputs,
)
from ...tf_utils import check_embeddings_within_bounds  # 导入检查嵌入是否在范围内的函数
from ...utils import (  # 导入通用工具函数
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from .configuration_layoutlmv3 import LayoutLMv3Config  # 导入LayoutLMv3的配置类


_CONFIG_FOR_DOC = "LayoutLMv3Config"  # 文档中使用的配置字符串

_DUMMY_INPUT_IDS = [  # 虚拟输入 ID 列表，用于测试
    [7, 6, 1],
    [1, 2, 0],
]

_DUMMY_BBOX = [  # 虚拟边界框列表，用于测试
    [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
    [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]],
]

TF_LAYOUTLMV3_PRETRAINED_MODEL_ARCHIVE_LIST = [  # 预训练模型存档列表
    "microsoft/layoutlmv3-base",
    "microsoft/layoutlmv3-large",
    # 查看所有LayoutLMv3模型：https://huggingface.co/models?filter=layoutlmv3
]

LARGE_NEGATIVE = -1e8  # 定义一个大负数常量


class TFLayoutLMv3PatchEmbeddings(keras.layers.Layer):
    """LayoutLMv3 图像（patch）嵌入层。"""

    def __init__(self, config: LayoutLMv3Config, **kwargs):
        super().__init__(**kwargs)  # 调用父类的初始化方法
        # 根据配置初始化图像补丁嵌入层
        patch_sizes = (
            config.patch_size
            if isinstance(config.patch_size, collections.abc.Iterable)
            else (config.patch_size, config.patch_size)
        )
        self.proj = keras.layers.Conv2D(  # 创建二维卷积层，用于投影图像补丁到隐藏向量空间
            filters=config.hidden_size,
            kernel_size=patch_sizes,
            strides=patch_sizes,
            padding="valid",
            data_format="channels_last",
            use_bias=True,
            kernel_initializer=get_initializer(config.initializer_range),
            name="proj",
        )
        self.hidden_size = config.hidden_size  # 记录隐藏大小
        self.num_patches = (config.input_size**2) // (patch_sizes[0] * patch_sizes[1])  # 计算补丁数量
        self.config = config  # 记录配置信息
    def call(self, pixel_values: tf.Tensor) -> tf.Tensor:
        # 当在 CPU 上运行时，`keras.layers.Conv2D` 不支持 `NCHW` 格式。
        # 因此，将输入格式从 `NCHW` 转换为 `NHWC`。
        pixel_values = tf.transpose(pixel_values, perm=[0, 2, 3, 1])

        # 使用 self.proj 对象进行投影操作，生成嵌入向量
        embeddings = self.proj(pixel_values)
        
        # 将嵌入向量重新整形为 (-1, self.num_patches, self.hidden_size) 的形状
        embeddings = tf.reshape(embeddings, (-1, self.num_patches, self.hidden_size))
        
        # 返回嵌入向量
        return embeddings

    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回
        if self.built:
            return
        
        # 标记当前模块为已构建
        self.built = True
        
        # 如果 self.proj 属性已存在，则构建它
        if getattr(self, "proj", None) is not None:
            with tf.name_scope(self.proj.name):
                # 使用 self.proj 的 build 方法构建投影层，输入形状为 [None, None, None, self.config.num_channels]
                self.proj.build([None, None, None, self.config.num_channels])
class TFLayoutLMv3TextEmbeddings(keras.layers.Layer):
    """
    LayoutLMv3 text embeddings. Same as `RobertaEmbeddings` but with added spatial (layout) embeddings.
    """

    def __init__(self, config: LayoutLMv3Config, **kwargs):
        super().__init__(**kwargs)
        # 初始化词嵌入层，用于将词汇 ID 映射为隐藏状态大小的向量
        self.word_embeddings = keras.layers.Embedding(
            config.vocab_size,
            config.hidden_size,
            embeddings_initializer=get_initializer(config.initializer_range),
            name="word_embeddings",
        )
        # 初始化 token 类型嵌入层，用于区分不同类型的 tokens（如 segment A/B）
        self.token_type_embeddings = keras.layers.Embedding(
            config.type_vocab_size,
            config.hidden_size,
            embeddings_initializer=get_initializer(config.initializer_range),
            name="token_type_embeddings",
        )
        # LayerNormalization 层，用于标准化输入数据
        self.LayerNorm = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
        # Dropout 层，用于随机失活，防止过拟合
        self.dropout = keras.layers.Dropout(config.hidden_dropout_prob)
        # 填充 token 的索引，用于在序列中标记填充位置
        self.padding_token_index = config.pad_token_id
        # 初始化位置嵌入层，用于编码位置信息
        self.position_embeddings = keras.layers.Embedding(
            config.max_position_embeddings,
            config.hidden_size,
            embeddings_initializer=get_initializer(config.initializer_range),
            name="position_embeddings",
        )
        # X 轴位置嵌入层，用于编码水平位置信息
        self.x_position_embeddings = keras.layers.Embedding(
            config.max_2d_position_embeddings,
            config.coordinate_size,
            embeddings_initializer=get_initializer(config.initializer_range),
            name="x_position_embeddings",
        )
        # Y 轴位置嵌入层，用于编码垂直位置信息
        self.y_position_embeddings = keras.layers.Embedding(
            config.max_2d_position_embeddings,
            config.coordinate_size,
            embeddings_initializer=get_initializer(config.initializer_range),
            name="y_position_embeddings",
        )
        # 高度位置嵌入层，用于编码形状高度信息
        self.h_position_embeddings = keras.layers.Embedding(
            config.max_2d_position_embeddings,
            config.shape_size,
            embeddings_initializer=get_initializer(config.initializer_range),
            name="h_position_embeddings",
        )
        # 宽度位置嵌入层，用于编码形状宽度信息
        self.w_position_embeddings = keras.layers.Embedding(
            config.max_2d_position_embeddings,
            config.shape_size,
            embeddings_initializer=get_initializer(config.initializer_range),
            name="w_position_embeddings",
        )
        # 最大二维位置数量，用于限制二维位置编码的范围
        self.max_2d_positions = config.max_2d_position_embeddings
        # 保存配置对象，包含模型的各种配置参数
        self.config = config
    # 计算空间位置嵌入的函数，接受一个边界框张量作为输入
    def calculate_spatial_position_embeddings(self, bbox: tf.Tensor) -> tf.Tensor:
        try:
            # 提取边界框的左边界位置索引
            left_position_ids = bbox[:, :, 0]
            # 提取边界框的上边界位置索引
            upper_position_ids = bbox[:, :, 1]
            # 提取边界框的右边界位置索引
            right_position_ids = bbox[:, :, 2]
            # 提取边界框的下边界位置索引
            lower_position_ids = bbox[:, :, 3]
        except IndexError as exception:
            # 如果边界框的形状不是 (batch_size, seq_length, 4)，抛出异常
            raise IndexError("Bounding box is not of shape (batch_size, seq_length, 4).") from exception

        try:
            # 使用 x_position_embeddings 方法为左边界位置创建嵌入
            left_position_embeddings = self.x_position_embeddings(left_position_ids)
            # 使用 y_position_embeddings 方法为上边界位置创建嵌入
            upper_position_embeddings = self.y_position_embeddings(upper_position_ids)
            # 使用 x_position_embeddings 方法为右边界位置创建嵌入
            right_position_embeddings = self.x_position_embeddings(right_position_ids)
            # 使用 y_position_embeddings 方法为下边界位置创建嵌入
            lower_position_embeddings = self.y_position_embeddings(lower_position_ids)
        except IndexError as exception:
            # 如果 bbox 坐标值不在 0 到 max_2d_positions 范围内，抛出异常
            raise IndexError(
                f"The `bbox` coordinate values should be within 0-{self.max_2d_positions} range."
            ) from exception

        # 计算高度嵌入，使用 h_position_embeddings 方法，并裁剪高度在 0 到 max_position_id 范围内
        max_position_id = self.max_2d_positions - 1
        h_position_embeddings = self.h_position_embeddings(
            tf.clip_by_value(bbox[:, :, 3] - bbox[:, :, 1], 0, max_position_id)
        )
        # 计算宽度嵌入，使用 w_position_embeddings 方法，并裁剪宽度在 0 到 max_position_id 范围内
        w_position_embeddings = self.w_position_embeddings(
            tf.clip_by_value(bbox[:, :, 2] - bbox[:, :, 0], 0, max_position_id)
        )

        # LayoutLMv1 将空间嵌入求和，但 LayoutLMv3 将它们连接起来
        # 将所有嵌入连接起来形成最终的空间位置嵌入
        spatial_position_embeddings = tf.concat(
            [
                left_position_embeddings,
                upper_position_embeddings,
                right_position_embeddings,
                lower_position_embeddings,
                h_position_embeddings,
                w_position_embeddings,
            ],
            axis=-1,
        )
        return spatial_position_embeddings

    # 从输入嵌入创建位置 id 的函数，接受一个输入嵌入张量作为输入
    def create_position_ids_from_inputs_embeds(self, inputs_embds: tf.Tensor) -> tf.Tensor:
        """
        We are provided embeddings directly. We cannot infer which are padded, so just generate sequential position
        ids.
        """
        # 获取输入张量的形状
        input_shape = tf.shape(inputs_embds)
        # 获取序列长度
        sequence_length = input_shape[1]
        # 计算起始位置索引，即填充令牌索引加 1
        start_index = self.padding_token_index + 1
        # 计算结束位置索引，即填充令牌索引加上序列长度再加 1
        end_index = self.padding_token_index + sequence_length + 1
        # 生成从 start_index 到 end_index 的连续整数作为位置 ids
        position_ids = tf.range(start_index, end_index, dtype=tf.int32)
        # 获取批量大小
        batch_size = input_shape[0]
        # 将位置 ids 重塑为形状为 (1, sequence_length) 的张量
        position_ids = tf.reshape(position_ids, (1, sequence_length))
        # 使用 tf.tile 在第一维度（批量大小维度）上复制位置 ids，使其形状与 inputs_embds 相同
        position_ids = tf.tile(position_ids, (batch_size, 1))
        return position_ids
    # 根据输入的输入 ID 创建位置 ID，并返回位置 ID 的张量
    def create_position_ids_from_input_ids(self, input_ids: tf.Tensor) -> tf.Tensor:
        """
        Replace non-padding symbols with their position numbers. Position numbers begin at padding_token_index + 1.
        """
        # 创建一个掩码，标记非填充符号的位置为1，填充符号位置为0
        mask = tf.cast(tf.not_equal(input_ids, self.padding_token_index), input_ids.dtype)
        # 计算累积和，乘以掩码，用于生成非填充符号的位置 ID
        position_ids = tf.cumsum(mask, axis=1) * mask
        # 将生成的位置 ID 调整为正确的位置，加上填充符号索引
        position_ids = position_ids + self.padding_token_index
        return position_ids

    # 根据输入的输入 ID 和嵌入张量创建位置 ID 的张量
    def create_position_ids(self, input_ids: tf.Tensor, inputs_embeds: tf.Tensor) -> tf.Tensor:
        if input_ids is None:
            return self.create_position_ids_from_inputs_embeds(inputs_embeds)
        else:
            return self.create_position_ids_from_input_ids(input_ids)

    # 模型的调用方法，根据传入的参数生成嵌入表示并返回
    def call(
        self,
        input_ids: tf.Tensor | None = None,
        bbox: tf.Tensor = None,
        token_type_ids: tf.Tensor | None = None,
        position_ids: tf.Tensor | None = None,
        inputs_embeds: tf.Tensor | None = None,
        training: bool = False,
    ) -> tf.Tensor:
        # 如果未提供位置 ID，则根据输入 ID 和输入嵌入生成位置 ID
        if position_ids is None:
            position_ids = self.create_position_ids(input_ids, inputs_embeds)

        # 根据输入 ID 或输入嵌入张量的形状确定输入形状
        if input_ids is not None:
            input_shape = tf.shape(input_ids)
        else:
            input_shape = tf.shape(inputs_embeds)[:-1]

        # 如果未提供 token_type_ids，则使用零张量填充，与位置 ID 张量的数据类型相同
        if token_type_ids is None:
            token_type_ids = tf.zeros(input_shape, dtype=position_ids.dtype)

        # 如果未提供输入嵌入张量，则根据输入 ID 检查并生成输入嵌入张量
        if inputs_embeds is None:
            check_embeddings_within_bounds(input_ids, self.word_embeddings.input_dim)
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        # 计算总的嵌入表示，包括输入嵌入、token_type 嵌入和位置嵌入
        embeddings = inputs_embeds + token_type_embeddings
        position_embeddings = self.position_embeddings(position_ids)
        embeddings += position_embeddings

        # 计算空间位置嵌入
        spatial_position_embeddings = self.calculate_spatial_position_embeddings(bbox)

        # 添加空间位置嵌入到总的嵌入表示中
        embeddings += spatial_position_embeddings

        # 应用 LayerNormalization
        embeddings = self.LayerNorm(embeddings)
        # 应用 Dropout，如果处于训练状态
        embeddings = self.dropout(embeddings, training=training)
        return embeddings
    # 如果已经构建过网络结构，则直接返回，不重复构建
    if self.built:
        return
    
    # 将构建标志设置为 True，表示网络结构已经构建
    self.built = True
    
    # 如果存在 word_embeddings 属性，则构建 word_embeddings 层
    if getattr(self, "word_embeddings", None) is not None:
        # 使用 word_embeddings 层的名称作为命名空间，构建该层
        with tf.name_scope(self.word_embeddings.name):
            self.word_embeddings.build(None)
    
    # 如果存在 token_type_embeddings 属性，则构建 token_type_embeddings 层
    if getattr(self, "token_type_embeddings", None) is not None:
        # 使用 token_type_embeddings 层的名称作为命名空间，构建该层
        with tf.name_scope(self.token_type_embeddings.name):
            self.token_type_embeddings.build(None)
    
    # 如果存在 LayerNorm 属性，则构建 LayerNorm 层
    if getattr(self, "LayerNorm", None) is not None:
        # 使用 LayerNorm 层的名称作为命名空间，构建该层
        self.LayerNorm.build([None, None, self.config.hidden_size])
    
    # 如果存在 position_embeddings 属性，则构建 position_embeddings 层
    if getattr(self, "position_embeddings", None) is not None:
        # 使用 position_embeddings 层的名称作为命名空间，构建该层
        with tf.name_scope(self.position_embeddings.name):
            self.position_embeddings.build(None)
    
    # 如果存在 x_position_embeddings 属性，则构建 x_position_embeddings 层
    if getattr(self, "x_position_embeddings", None) is not None:
        # 使用 x_position_embeddings 层的名称作为命名空间，构建该层
        with tf.name_scope(self.x_position_embeddings.name):
            self.x_position_embeddings.build(None)
    
    # 如果存在 y_position_embeddings 属性，则构建 y_position_embeddings 层
    if getattr(self, "y_position_embeddings", None) is not None:
        # 使用 y_position_embeddings 层的名称作为命名空间，构建该层
        with tf.name_scope(self.y_position_embeddings.name):
            self.y_position_embeddings.build(None)
    
    # 如果存在 h_position_embeddings 属性，则构建 h_position_embeddings 层
    if getattr(self, "h_position_embeddings", None) is not None:
        # 使用 h_position_embeddings 层的名称作为命名空间，构建该层
        with tf.name_scope(self.h_position_embeddings.name):
            self.h_position_embeddings.build(None)
    
    # 如果存在 w_position_embeddings 属性，则构建 w_position_embeddings 层
    if getattr(self, "w_position_embeddings", None) is not None:
        # 使用 w_position_embeddings 层的名称作为命名空间，构建该层
        with tf.name_scope(self.w_position_embeddings.name):
            self.w_position_embeddings.build(None)
class TFLayoutLMv3SelfAttention(keras.layers.Layer):
    def __init__(self, config: LayoutLMv3Config, **kwargs):
        super().__init__(**kwargs)
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.attention_score_normaliser = math.sqrt(self.attention_head_size)

        # 创建用于查询、键和值的全连接层，每个都初始化为给定的范围
        self.query = keras.layers.Dense(
            self.all_head_size,
            kernel_initializer=get_initializer(config.initializer_range),
            name="query",
        )
        self.key = keras.layers.Dense(
            self.all_head_size,
            kernel_initializer=get_initializer(config.initializer_range),
            name="key",
        )
        self.value = keras.layers.Dense(
            self.all_head_size,
            kernel_initializer=get_initializer(config.initializer_range),
            name="value",
        )

        # Dropout 层，用于注意力概率的随机失活
        self.dropout = keras.layers.Dropout(config.attention_probs_dropout_prob)
        self.has_relative_attention_bias = config.has_relative_attention_bias
        self.has_spatial_attention_bias = config.has_spatial_attention_bias
        self.config = config

    def transpose_for_scores(self, x: tf.Tensor):
        # 重塑张量形状，以便适应多头注意力的计算
        shape = tf.shape(x)
        new_shape = (
            shape[0],  # batch_size
            shape[1],  # seq_length
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = tf.reshape(x, new_shape)
        return tf.transpose(x, perm=[0, 2, 1, 3])  # 返回转置后的张量，用于多头注意力计算

    def cogview_attention(self, attention_scores: tf.Tensor, alpha: Union[float, int] = 32):
        """
        https://arxiv.org/abs/2105.13290 Section 2.4 Stabilization of training: Precision Bottleneck Relaxation
        (PB-Relax). A replacement of the original keras.layers.Softmax(axis=-1)(attention_scores). Seems the new
        attention_probs will result in a slower speed and a little bias. Can use
        tf.debugging.assert_near(standard_attention_probs, cogview_attention_probs, atol=1e-08) for comparison. The
        smaller atol (e.g., 1e-08), the better.
        """
        # 缩放注意力分数，根据给定的 alpha 参数
        scaled_attention_scores = attention_scores / alpha
        # 计算缩放后的注意力分数的最大值
        max_value = tf.expand_dims(tf.reduce_max(scaled_attention_scores, axis=-1), axis=-1)
        # 应用 PB-Relax 方法调整注意力分数，然后使用 softmax 计算新的注意力概率
        new_attention_scores = (scaled_attention_scores - max_value) * alpha
        return tf.math.softmax(new_attention_scores, axis=-1)
    def call(
        self,
        hidden_states: tf.Tensor,  # 输入张量，表示模型的隐藏状态
        attention_mask: tf.Tensor | None,  # 注意力掩码张量，用于屏蔽无效的注意力位置
        head_mask: tf.Tensor | None,  # 头部掩码张量，用于屏蔽特定注意力头部
        output_attentions: bool,  # 布尔值，表示是否输出注意力概率
        rel_pos: tf.Tensor | None = None,  # 相对位置张量，用于相对位置注意力
        rel_2d_pos: tf.Tensor | None = None,  # 二维相对位置张量，用于空间位置注意力
        training: bool = False,  # 布尔值，表示是否在训练模式下
    ) -> Union[Tuple[tf.Tensor], Tuple[tf.Tensor, tf.Tensor]]:
        # 计算 Query、Key 和 Value 张量的转置并处理成注意力得分
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(self.query(hidden_states))

        # 对 Query 和 Key 进行点积，得到原始的注意力分数
        normalised_query_layer = query_layer / self.attention_score_normaliser
        transposed_key_layer = tf.transpose(
            key_layer, perm=[0, 1, 3, 2]
        )  # batch_size, num_heads, attention_head_size, seq_length
        attention_scores = tf.matmul(normalised_query_layer, transposed_key_layer)

        # 添加相对注意力偏置（如果有的话）到注意力分数中
        if self.has_relative_attention_bias and self.has_spatial_attention_bias:
            attention_scores += (rel_pos + rel_2d_pos) / self.attention_score_normaliser
        elif self.has_relative_attention_bias:
            attention_scores += rel_pos / self.attention_score_normaliser

        # 如果存在注意力掩码，将其应用到注意力分数中
        if attention_mask is not None:
            # 应用预先计算的注意力掩码（在 TFLayoutLMv3Model 的 call() 函数中计算）
            attention_scores += attention_mask

        # 将注意力分数归一化为注意力概率
        # 使用 CogView 论文中的技巧来稳定训练
        attention_probs = self.cogview_attention(attention_scores)

        # 根据训练模式进行 dropout 处理
        attention_probs = self.dropout(attention_probs, training=training)

        # 如果存在头部掩码，将其应用到注意力概率中
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        # 计算上下文层张量，通过注意力概率和 Value 层张量的乘积得到
        context_layer = tf.matmul(attention_probs, value_layer)
        context_layer = tf.transpose(
            context_layer, perm=[0, 2, 1, 3]
        )  # batch_size, seq_length, num_heads, attention_head_size
        shape = tf.shape(context_layer)
        context_layer = tf.reshape(
            context_layer, (shape[0], shape[1], self.all_head_size)
        )  # batch_size, seq_length, num_heads * attention_head_size

        # 根据是否需要输出注意力概率来选择输出
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs
    # 构建方法，用于构建自定义层的输入形状
    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回
        if self.built:
            return
        # 标记为已构建
        self.built = True
        # 如果存在查询张量，则构建查询张量的形状
        if getattr(self, "query", None) is not None:
            # 使用查询张量的名称作为作用域，构建其形状
            with tf.name_scope(self.query.name):
                self.query.build([None, None, self.config.hidden_size])
        # 如果存在键张量，则构建键张量的形状
        if getattr(self, "key", None) is not None:
            # 使用键张量的名称作为作用域，构建其形状
            with tf.name_scope(self.key.name):
                self.key.build([None, None, self.config.hidden_size])
        # 如果存在值张量，则构建值张量的形状
        if getattr(self, "value", None) is not None:
            # 使用值张量的名称作为作用域，构建其形状
            with tf.name_scope(self.value.name):
                self.value.build([None, None, self.config.hidden_size])
# Copied from models.roberta.modeling_tf_roberta.TFRobertaSelfOutput
class TFLayoutLMv3SelfOutput(keras.layers.Layer):
    def __init__(self, config: LayoutLMv3Config, **kwargs):
        super().__init__(**kwargs)

        # 定义一个全连接层，用于对隐藏状态进行线性变换
        self.dense = keras.layers.Dense(
            units=config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )
        # 定义一个 LayerNormalization 层，用于归一化隐藏状态
        self.LayerNorm = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
        # 定义一个 Dropout 层，用于在训练时随机屏蔽神经元，防止过拟合
        self.dropout = keras.layers.Dropout(rate=config.hidden_dropout_prob)
        self.config = config

    def call(self, hidden_states: tf.Tensor, input_tensor: tf.Tensor, training: bool = False) -> tf.Tensor:
        # 将隐藏状态传入全连接层进行线性变换
        hidden_states = self.dense(inputs=hidden_states)
        # 在训练时应用 Dropout，随机屏蔽部分神经元
        hidden_states = self.dropout(inputs=hidden_states, training=training)
        # 使用 LayerNormalization 层归一化隐藏状态并加上输入张量（残差连接）
        hidden_states = self.LayerNorm(inputs=hidden_states + input_tensor)

        return hidden_states

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 如果 dense 层已定义，构建其权重
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.hidden_size])
        # 如果 LayerNorm 层已定义，构建其权重
        if getattr(self, "LayerNorm", None) is not None:
            with tf.name_scope(self.LayerNorm.name):
                self.LayerNorm.build([None, None, self.config.hidden_size])


class TFLayoutLMv3Attention(keras.layers.Layer):
    def __init__(self, config: LayoutLMv3Config, **kwargs):
        super().__init__(**kwargs)
        # 创建自注意力层和自注意力输出层对象
        self.self_attention = TFLayoutLMv3SelfAttention(config, name="self")
        self.self_output = TFLayoutLMv3SelfOutput(config, name="output")

    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: tf.Tensor | None,
        head_mask: tf.Tensor | None,
        output_attentions: bool,
        rel_pos: tf.Tensor | None = None,
        rel_2d_pos: tf.Tensor | None = None,
        training: bool = False,
    ) -> Union[Tuple[tf.Tensor], Tuple[tf.Tensor, tf.Tensor]]:
        # 调用自注意力层进行计算
        self_outputs = self.self_attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions,
            rel_pos,
            rel_2d_pos,
            training=training,
        )
        # 将自注意力层的输出传递给自注意力输出层进行处理
        attention_output = self.self_output(self_outputs[0], hidden_states, training=training)
        # 返回处理后的输出，包括注意力输出和可能的额外输出（如注意力权重）
        outputs = (attention_output,) + self_outputs[1:]  # 如果有额外输出，则将其添加到结果中
        return outputs

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 如果 self_attention 层已定义，构建其权重
        if getattr(self, "self_attention", None) is not None:
            with tf.name_scope(self.self_attention.name):
                self.self_attention.build(None)
        # 如果 self_output 层已定义，构建其权重
        if getattr(self, "self_output", None) is not None:
            with tf.name_scope(self.self_output.name):
                self.self_output.build(None)
# Copied from models.roberta.modeling_tf_bert.TFRobertaIntermediate
class TFLayoutLMv3Intermediate(keras.layers.Layer):
    def __init__(self, config: LayoutLMv3Config, **kwargs):
        super().__init__(**kwargs)

        # 创建一个全连接层，用于处理中间层的输出
        self.dense = keras.layers.Dense(
            units=config.intermediate_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )

        # 根据配置文件中的激活函数类型，获取对应的 TensorFlow 激活函数
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = get_tf_activation(config.hidden_act)
        else:
            self.intermediate_act_fn = config.hidden_act
        self.config = config

    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        # 通过全连接层处理输入的隐藏状态
        hidden_states = self.dense(inputs=hidden_states)
        # 应用中间层的激活函数
        hidden_states = self.intermediate_act_fn(hidden_states)

        return hidden_states

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 如果存在全连接层，根据输入的形状构建全连接层
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.hidden_size])


# Copied from models.roberta.modeling_tf_bert.TFRobertaOutput
class TFLayoutLMv3Output(keras.layers.Layer):
    def __init__(self, config: LayoutLMv3Config, **kwargs):
        super().__init__(**kwargs)

        # 创建一个全连接层，用于处理输出层的输出
        self.dense = keras.layers.Dense(
            units=config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )
        # 创建 LayerNormalization 层，用于规范化输出
        self.LayerNorm = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
        # 创建 Dropout 层，用于在训练时随机丢弃部分神经元，防止过拟合
        self.dropout = keras.layers.Dropout(rate=config.hidden_dropout_prob)
        self.config = config

    def call(self, hidden_states: tf.Tensor, input_tensor: tf.Tensor, training: bool = False) -> tf.Tensor:
        # 通过全连接层处理输入的隐藏状态
        hidden_states = self.dense(inputs=hidden_states)
        # 在训练时，通过 Dropout 层随机丢弃部分神经元
        hidden_states = self.dropout(inputs=hidden_states, training=training)
        # 对输出进行 LayerNormalization，并添加残差连接
        hidden_states = self.LayerNorm(inputs=hidden_states + input_tensor)

        return hidden_states

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 如果存在全连接层，根据输入的形状构建全连接层
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.intermediate_size])
        # 如果存在 LayerNormalization 层，根据输入的形状构建 LayerNormalization 层
        if getattr(self, "LayerNorm", None) is not None:
            with tf.name_scope(self.LayerNorm.name):
                self.LayerNorm.build([None, None, self.config.hidden_size])


class TFLayoutLMv3Layer(keras.layers.Layer):
    def __init__(self, config: LayoutLMv3Config, **kwargs):
        super().__init__(**kwargs)
        # 初始化注意力层对象
        self.attention = TFLayoutLMv3Attention(config, name="attention")
        # 初始化中间层对象
        self.intermediate = TFLayoutLMv3Intermediate(config, name="intermediate")
        # 初始化输出层对象
        self.bert_output = TFLayoutLMv3Output(config, name="output")
    # 定义一个方法 `call`，用于执行 Transformer 层的前向传播
    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: tf.Tensor | None,
        head_mask: tf.Tensor | None,
        output_attentions: bool,
        rel_pos: tf.Tensor | None = None,
        rel_2d_pos: tf.Tensor | None = None,
        training: bool = False,
    ) -> Union[Tuple[tf.Tensor], Tuple[tf.Tensor, tf.Tensor]]:
        # 调用注意力层进行自注意力计算
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            rel_pos=rel_pos,
            rel_2d_pos=rel_2d_pos,
            training=training,
        )
        # 获取自注意力计算的输出
        attention_output = self_attention_outputs[0]
        # 如果需要输出注意力权重，则将注意力权重也加入到输出中
        outputs = self_attention_outputs[1:]  # 如果输出注意力权重，则添加自注意力权重
        # 将自注意力输出传入中间层进行处理
        intermediate_output = self.intermediate(attention_output)
        # 将中间层的输出和自注意力输出传入 BERT 输出层进行最终处理
        layer_output = self.bert_output(intermediate_output, attention_output, training=training)
        # 将本层的输出和可能的注意力权重输出合并成最终输出
        outputs = (layer_output,) + outputs
        # 返回最终输出
        return outputs

    # 定义一个方法 `build`，用于构建 Transformer 层
    def build(self, input_shape=None):
        # 如果已经构建过，直接返回
        if self.built:
            return
        # 标记为已构建
        self.built = True
        # 如果存在注意力层，则构建注意力层
        if getattr(self, "attention", None) is not None:
            with tf.name_scope(self.attention.name):
                self.attention.build(None)
        # 如果存在中间层，则构建中间层
        if getattr(self, "intermediate", None) is not None:
            with tf.name_scope(self.intermediate.name):
                self.intermediate.build(None)
        # 如果存在 BERT 输出层，则构建 BERT 输出层
        if getattr(self, "bert_output", None) is not None:
            with tf.name_scope(self.bert_output.name):
                self.bert_output.build(None)
# 定义 TFLayoutLMv3Encoder 类，继承自 keras.layers.Layer
class TFLayoutLMv3Encoder(keras.layers.Layer):
    # 初始化方法，接受 LayoutLMv3Config 类型的 config 参数和其他关键字参数
    def __init__(self, config: LayoutLMv3Config, **kwargs):
        # 调用父类的初始化方法
        super().__init__(**kwargs)
        # 将传入的 config 参数赋值给对象的 config 属性
        self.config = config
        # 创建一个列表，包含 config.num_hidden_layers 个 TFLayoutLMv3Layer 对象，每个对象命名为 "layer.{i}"
        self.layer = [TFLayoutLMv3Layer(config, name=f"layer.{i}") for i in range(config.num_hidden_layers)]

        # 检查是否具有相对注意力偏置
        self.has_relative_attention_bias = config.has_relative_attention_bias
        # 检查是否具有空间注意力偏置
        self.has_spatial_attention_bias = config.has_spatial_attention_bias

        # 如果具有相对注意力偏置，进行以下设置
        if self.has_relative_attention_bias:
            # 将 config.rel_pos_bins 赋值给对象的 rel_pos_bins 属性
            self.rel_pos_bins = config.rel_pos_bins
            # 将 config.max_rel_pos 赋值给对象的 max_rel_pos 属性
            self.max_rel_pos = config.max_rel_pos
            # 创建一个 Dense 层用于相对位置偏置，单元数为 config.num_attention_heads，
            # 内核初始化方式为 get_initializer(config.initializer_range)，不使用偏置，命名为 "rel_pos_bias"
            self.rel_pos_bias = keras.layers.Dense(
                units=config.num_attention_heads,
                kernel_initializer=get_initializer(config.initializer_range),
                use_bias=False,
                name="rel_pos_bias",
            )

        # 如果具有空间注意力偏置，进行以下设置
        if self.has_spatial_attention_bias:
            # 将 config.max_rel_2d_pos 赋值给对象的 max_rel_2d_pos 属性
            self.max_rel_2d_pos = config.max_rel_2d_pos
            # 将 config.rel_2d_pos_bins 赋值给对象的 rel_2d_pos_bins 属性
            self.rel_2d_pos_bins = config.rel_2d_pos_bins
            # 创建一个 Dense 层用于 X 方向的相对位置偏置，单元数为 config.num_attention_heads，
            # 内核初始化方式为 get_initializer(config.initializer_range)，不使用偏置，命名为 "rel_pos_x_bias"
            self.rel_pos_x_bias = keras.layers.Dense(
                units=config.num_attention_heads,
                kernel_initializer=get_initializer(config.initializer_range),
                use_bias=False,
                name="rel_pos_x_bias",
            )
            # 创建一个 Dense 层用于 Y 方向的相对位置偏置，单元数为 config.num_attention_heads，
            # 内核初始化方式为 get_initializer(config.initializer_range)，不使用偏置，命名为 "rel_pos_y_bias"
            self.rel_pos_y_bias = keras.layers.Dense(
                units=config.num_attention_heads,
                kernel_initializer=get_initializer(config.initializer_range),
                use_bias=False,
                name="rel_pos_y_bias",
            )
    # 根据相对位置计算桶索引，用于位置编码
    def relative_position_bucket(self, relative_positions: tf.Tensor, num_buckets: int, max_distance: int):
        # 负的相对位置被分配到区间 [0, num_buckets / 2]
        # 我们通过将绝对值的相对位置分配到区间 [0, num_buckets / 2] 来处理这一点
        # 然后在最后将正的相对位置偏移 num_buckets / 2
        num_buckets = num_buckets // 2
        buckets = tf.abs(relative_positions)

        # 一半的桶用于精确增量的位置
        max_exact_buckets = num_buckets // 2
        is_small = buckets < max_exact_buckets

        # 另一半的桶用于位置在最大距离 max_distance 内的对数增大的区间
        buckets_log_ratio = tf.math.log(tf.cast(buckets, tf.float32) / max_exact_buckets)
        distance_log_ratio = math.log(max_distance / max_exact_buckets)
        buckets_big_offset = (
            buckets_log_ratio / distance_log_ratio * (num_buckets - max_exact_buckets)
        )  # 缩放在 [0, num_buckets - max_exact_buckets] 的范围内
        buckets_big = max_exact_buckets + buckets_big_offset  # 范围是 [max_exact_buckets, num_buckets]
        buckets_big = tf.cast(buckets_big, buckets.dtype)
        buckets_big = tf.minimum(buckets_big, num_buckets - 1)

        return (tf.cast(relative_positions > 0, buckets.dtype) * num_buckets) + tf.where(
            is_small, buckets, buckets_big
        )

    # 计算位置编码的一维版本
    def _cal_1d_pos_emb(self, position_ids: tf.Tensor):
        return self._cal_pos_emb(self.rel_pos_bias, position_ids, self.rel_pos_bins, self.max_rel_pos)

    # 计算位置编码的二维版本
    def _cal_2d_pos_emb(self, bbox: tf.Tensor):
        position_coord_x = bbox[:, :, 0]  # 左边界
        position_coord_y = bbox[:, :, 3]  # 底边界
        rel_pos_x = self._cal_pos_emb(
            self.rel_pos_x_bias,
            position_coord_x,
            self.rel_2d_pos_bins,
            self.max_rel_2d_pos,
        )
        rel_pos_y = self._cal_pos_emb(
            self.rel_pos_y_bias,
            position_coord_y,
            self.rel_2d_pos_bins,
            self.max_rel_2d_pos,
        )
        rel_2d_pos = rel_pos_x + rel_pos_y
        return rel_2d_pos
    def call(
        self,
        hidden_states: tf.Tensor,
        bbox: tf.Tensor | None = None,
        attention_mask: tf.Tensor | None = None,
        head_mask: tf.Tensor | None = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        position_ids: tf.Tensor | None = None,
        training: bool = False,
    ) -> Union[
        TFBaseModelOutput,
        Tuple[tf.Tensor],
        Tuple[tf.Tensor, tf.Tensor],
        Tuple[tf.Tensor, tf.Tensor, tf.Tensor],
    ]:
        # 如果需要输出隐藏状态，则初始化空元组
        all_hidden_states = () if output_hidden_states else None
        # 如果需要输出注意力权重，则初始化空元组
        all_self_attentions = () if output_attentions else None

        # 如果模型支持相对位置注意力偏置，则计算一维位置嵌入
        rel_pos = self._cal_1d_pos_emb(position_ids) if self.has_relative_attention_bias else None
        # 如果模型支持空间注意力偏置，则计算二维位置嵌入
        rel_2d_pos = self._cal_2d_pos_emb(bbox) if self.has_spatial_attention_bias else None

        # 遍历每个层模块
        for i, layer_module in enumerate(self.layer):
            # 如果需要输出隐藏状态，则保存当前隐藏状态
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 获取当前层的注意力头遮罩
            layer_head_mask = head_mask[i] if head_mask is not None else None

            # 调用层模块进行前向传播
            layer_outputs = layer_module(
                hidden_states,
                attention_mask,
                layer_head_mask,
                output_attentions,
                rel_pos=rel_pos,
                rel_2d_pos=rel_2d_pos,
                training=training,
            )

            # 更新隐藏状态为当前层模块的输出
            hidden_states = layer_outputs[0]
            # 如果需要输出注意力权重，则保存当前层的自注意力权重
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        # 如果需要输出隐藏状态，则保存最终的隐藏状态
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 如果设置返回字典，则返回 TFBaseModelOutput 对象
        if return_dict:
            return TFBaseModelOutput(
                last_hidden_state=hidden_states,
                hidden_states=all_hidden_states,
                attentions=all_self_attentions,
            )
        # 否则，根据是否有有效值返回元组
        else:
            return tuple(
                value for value in [hidden_states, all_hidden_states, all_self_attentions] if value is not None
            )

    def build(self, input_shape=None):
        # 如果模型已经构建，则直接返回
        if self.built:
            return
        # 标记模型已构建
        self.built = True

        # 如果存在相对位置偏置，则构建相对位置偏置
        if getattr(self, "rel_pos_bias", None) is not None:
            with tf.name_scope(self.rel_pos_bias.name):
                self.rel_pos_bias.build([None, None, self.rel_pos_bins])
        
        # 如果存在 X 方向的相对位置偏置，则构建 X 方向相对位置偏置
        if getattr(self, "rel_pos_x_bias", None) is not None:
            with tf.name_scope(self.rel_pos_x_bias.name):
                self.rel_pos_x_bias.build([None, None, self.rel_2d_pos_bins])
        
        # 如果存在 Y 方向的相对位置偏置，则构建 Y 方向相对位置偏置
        if getattr(self, "rel_pos_y_bias", None) is not None:
            with tf.name_scope(self.rel_pos_y_bias.name):
                self.rel_pos_y_bias.build([None, None, self.rel_2d_pos_bins])
        
        # 遍历每个层并构建它们
        if getattr(self, "layer", None) is not None:
            for layer in self.layer:
                with tf.name_scope(layer.name):
                    layer.build(None)
# 使用 keras_serializable 装饰器将该类声明为可序列化的 Keras 层
@keras_serializable
class TFLayoutLMv3MainLayer(keras.layers.Layer):
    # 指定配置类为 LayoutLMv3Config
    config_class = LayoutLMv3Config

    # 初始化方法，接收 LayoutLMv3Config 对象和其他关键字参数
    def __init__(self, config: LayoutLMv3Config, **kwargs):
        super().__init__(**kwargs)

        # 将传入的配置对象保存在 self.config 中
        self.config = config

        # 如果配置要求包含文本嵌入，则创建 TFLayoutLMv3TextEmbeddings 对象并命名为 "embeddings"
        if config.text_embed:
            self.embeddings = TFLayoutLMv3TextEmbeddings(config, name="embeddings")

        # 如果配置要求包含视觉嵌入
        if config.visual_embed:
            # 创建 TFLayoutLMv3PatchEmbeddings 对象并命名为 "patch_embed"
            self.patch_embed = TFLayoutLMv3PatchEmbeddings(config, name="patch_embed")
            # 创建 LayerNormalization 层并设置参数
            self.LayerNorm = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
            # 创建 Dropout 层并设置丢弃率
            self.dropout = keras.layers.Dropout(config.hidden_dropout_prob, name="dropout")

            # 如果配置中有相对注意力偏置或空间注意力偏置，则初始化视觉边界框
            if config.has_relative_attention_bias or config.has_spatial_attention_bias:
                image_size = config.input_size // config.patch_size
                self.init_visual_bbox(image_size=(image_size, image_size))

            # 创建 LayerNormalization 层并设置参数
            self.norm = keras.layers.LayerNormalization(epsilon=1e-6, name="norm")

        # 创建 TFLayoutLMv3Encoder 对象并命名为 "encoder"
        self.encoder = TFLayoutLMv3Encoder(config, name="encoder")

    # 构建方法，在此处根据输入形状构建网络层
    def build(self, input_shape=None):
        # 如果配置中包含视觉嵌入
        if self.config.visual_embed:
            image_size = self.config.input_size // self.config.patch_size
            # 创建用于分类的 token，初始化为全零
            self.cls_token = self.add_weight(
                shape=(1, 1, self.config.hidden_size),
                initializer="zeros",
                trainable=True,
                dtype=tf.float32,
                name="cls_token",
            )
            # 创建位置嵌入矩阵，初始化为全零
            self.pos_embed = self.add_weight(
                shape=(1, image_size * image_size + 1, self.config.hidden_size),
                initializer="zeros",
                trainable=True,
                dtype=tf.float32,
                name="pos_embed",
            )

        # 如果已经构建过，则直接返回
        if self.built:
            return
        self.built = True
        # 如果存在 self.encoder 属性，则在命名空间下构建 encoder
        if getattr(self, "encoder", None) is not None:
            with tf.name_scope(self.encoder.name):
                self.encoder.build(None)
        # 如果存在 self.embeddings 属性，则在命名空间下构建 embeddings
        if getattr(self, "embeddings", None) is not None:
            with tf.name_scope(self.embeddings.name):
                self.embeddings.build(None)
        # 如果存在 self.patch_embed 属性，则在命名空间下构建 patch_embed
        if getattr(self, "patch_embed", None) is not None:
            with tf.name_scope(self.patch_embed.name):
                self.patch_embed.build(None)
        # 如果存在 self.LayerNorm 属性，则在命名空间下构建 LayerNorm
        if getattr(self, "LayerNorm", None) is not None:
            with tf.name_scope(self.LayerNorm.name):
                self.LayerNorm.build([None, None, self.config.hidden_size])
        # 如果存在 self.dropout 属性，则在命名空间下构建 dropout
        if getattr(self, "dropout", None) is not None:
            with tf.name_scope(self.dropout.name):
                self.dropout.build(None)
        # 如果存在 self.norm 属性，则在命名空间下构建 norm
        if getattr(self, "norm", None) is not None:
            with tf.name_scope(self.norm.name):
                self.norm.build([None, None, self.config.hidden_size])

    # 获取输入嵌入层的方法，返回 embeddings 的 word_embeddings 属性
    def get_input_embeddings(self) -> keras.layers.Layer:
        return self.embeddings.word_embeddings

    # 设置输入嵌入层的方法，将 value 赋值给 embeddings 的 word_embeddings 属性
    def set_input_embeddings(self, value: tf.Variable):
        self.embeddings.word_embeddings.weight = value
    # 从 transformers.models.bert.modeling_tf_bert.TFBertMainLayer._prune_heads 复制而来的方法，用于剪枝模型中的注意力头
    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        raise NotImplementedError

    # 初始化视觉边界框，设置图像的大小和最大长度
    def init_visual_bbox(self, image_size: Tuple[int, int], max_len: int = 1000):
        # 不应该将 max_len 硬编码为 1000，但是参考实现这样做了，为了与预训练权重兼容，我们保留了这个值。
        # 更正确的做法应该是传递 max_len=config.max_2d_position_embeddings - 1。
        height, width = image_size

        # 计算水平边界框的 x 坐标
        visual_bbox_x = tf.range(0, max_len * (width + 1), max_len) // width
        visual_bbox_x = tf.expand_dims(visual_bbox_x, axis=0)
        visual_bbox_x = tf.tile(visual_bbox_x, [width, 1])  # (width, width + 1)

        # 计算垂直边界框的 y 坐标
        visual_bbox_y = tf.range(0, max_len * (height + 1), max_len) // height
        visual_bbox_y = tf.expand_dims(visual_bbox_y, axis=1)
        visual_bbox_y = tf.tile(visual_bbox_y, [1, height])  # (height + 1, height)

        # 组合 x 和 y 坐标，形成边界框的四个角的坐标
        visual_bbox = tf.stack(
            [visual_bbox_x[:, :-1], visual_bbox_y[:-1], visual_bbox_x[:, 1:], visual_bbox_y[1:]],
            axis=-1,
        )
        visual_bbox = tf.reshape(visual_bbox, [-1, 4])

        # 添加一个表示 [CLS] 标记的边界框
        cls_token_box = tf.constant([[1, 1, max_len - 1, max_len - 1]], dtype=tf.int32)
        self.visual_bbox = tf.concat([cls_token_box, visual_bbox], axis=0)

    # 计算视觉边界框的形状并复制到指定批次大小
    def calculate_visual_bbox(self, batch_size: int, dtype: tf.DType):
        visual_bbox = tf.expand_dims(self.visual_bbox, axis=0)
        visual_bbox = tf.tile(visual_bbox, [batch_size, 1, 1])
        visual_bbox = tf.cast(visual_bbox, dtype=dtype)
        return visual_bbox

    # 嵌入图像像素值，返回嵌入后的张量
    def embed_image(self, pixel_values: tf.Tensor) -> tf.Tensor:
        # 使用补丁嵌入器将像素值转换为嵌入表示
        embeddings = self.patch_embed(pixel_values)

        # 添加 [CLS] 标记
        batch_size = tf.shape(embeddings)[0]
        cls_tokens = tf.tile(self.cls_token, [batch_size, 1, 1])
        embeddings = tf.concat([cls_tokens, embeddings], axis=1)

        # 添加位置嵌入
        if getattr(self, "pos_embed", None) is not None:
            embeddings += self.pos_embed

        # 归一化嵌入张量
        embeddings = self.norm(embeddings)
        return embeddings
    # Adapted from transformers.modelling_utils.ModuleUtilsMixin.get_extended_attention_mask
    # 根据注意力掩码的维度数量进行扩展，使其适用于多头注意力机制

    n_dims = len(attention_mask.shape)
    # 获取注意力掩码张量的维度数量

    if n_dims == 3:
        # 如果维度为3，表示提供了自定义的自注意力掩码 [batch_size, from_seq_length, to_seq_length]
        # 扩展维度使其适用于所有注意力头
        extended_attention_mask = tf.expand_dims(attention_mask, axis=1)
    elif n_dims == 2:
        # 如果维度为2，表示提供了填充掩码 [batch_size, seq_length]
        # 扩展维度使其适用于 [batch_size, num_heads, seq_length, seq_length]
        extended_attention_mask = tf.expand_dims(attention_mask, axis=1)  # (batch_size, 1, seq_length)
        extended_attention_mask = tf.expand_dims(extended_attention_mask, axis=1)  # (batch_size, 1, 1, seq_length)
    else:
        # 抛出异常，注意力掩码的形状不正确
        raise ValueError(f"Wrong shape for attention_mask (shape {attention_mask.shape}).")

    # 由于注意力掩码中 1.0 表示要关注的位置，0.0 表示掩码位置，
    # 这个操作将创建一个张量，对于要关注的位置为 0.0，对于掩码位置为 -10000.0
    # 在 softmax 前将其加到原始分数中，等效于完全移除这些位置的影响
    extended_attention_mask = tf.cast(extended_attention_mask, self.compute_dtype)
    extended_attention_mask = (1.0 - extended_attention_mask) * LARGE_NEGATIVE

    return extended_attention_mask
    def get_head_mask(self, head_mask: tf.Tensor | None) -> Union[tf.Tensor, List[tf.Tensor | None]]:
        if head_mask is None:
            # 如果头部掩码为 None，则返回一个包含 None 的列表，长度为模型隐藏层的数量
            return [None] * self.config.num_hidden_layers

        # 获取头部掩码的张量维度数
        n_dims = tf.rank(head_mask)
        if n_dims == 1:
            # 获取每个头部的掩码张量
            head_mask = tf.expand_dims(head_mask, axis=0)  # 1, num_heads
            head_mask = tf.expand_dims(head_mask, axis=0)  # 1, 1, num_heads
            head_mask = tf.expand_dims(head_mask, axis=-1)  # 1, 1, num_heads, 1
            head_mask = tf.expand_dims(head_mask, axis=-1)  # 1, 1, num_heads, 1, 1
            # 复制头部掩码以适应每个隐藏层
            head_mask = tf.tile(
                head_mask, [self.config.num_hidden_layers, 1, 1, 1, 1]
            )  # seq_length, 1, num_heads, 1, 1
        elif n_dims == 2:
            # 获取每个层和头部的掩码张量
            head_mask = tf.expand_dims(head_mask, axis=1)  # seq_length, 1, num_heads
            head_mask = tf.expand_dims(head_mask, axis=-1)  # seq_length, 1, num_heads, 1
            head_mask = tf.expand_dims(head_mask, axis=-1)  # seq_length, 1, num_heads, 1, 1
        elif n_dims != 5:
            # 如果掩码维度不是5，则抛出异常
            raise ValueError(f"Wrong shape for head_mask (shape {head_mask.shape}).")
        # 确保头部掩码的维度为5
        assert tf.rank(head_mask) == 5, f"Got head_mask rank of {tf.rank(head_mask)}, but require 5."
        # 将头部掩码转换为计算数据类型
        head_mask = tf.cast(head_mask, self.compute_dtype)
        return head_mask

    @unpack_inputs
    def call(
        self,
        input_ids: tf.Tensor | None = None,
        bbox: tf.Tensor | None = None,
        attention_mask: tf.Tensor | None = None,
        token_type_ids: tf.Tensor | None = None,
        position_ids: tf.Tensor | None = None,
        head_mask: tf.Tensor | None = None,
        inputs_embeds: tf.Tensor | None = None,
        pixel_values: tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
    ) -> Union[
        TFBaseModelOutput,
        Tuple[tf.Tensor],
        Tuple[tf.Tensor, tf.Tensor],
        Tuple[tf.Tensor, tf.Tensor, tf.Tensor],
        Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor],
        Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor],
        Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor],
    ]:
        # 此函数定义了模型的调用方式，处理多个输入和配置参数，返回一个包含不同输出类型的联合类型
class TFLayoutLMv3PreTrainedModel(TFPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 指定配置类为 LayoutLMv3Config
    config_class = LayoutLMv3Config
    # 基础模型前缀为 "layoutlmv3"
    base_model_prefix = "layoutlmv3"

    @property
    def input_signature(self):
        # 获取父类 TFPreTrainedModel 的输入签名
        sig = super().input_signature
        # 添加一个新的输入 "bbox"，表示边界框，格式为 (None, None, 4) 的 int32 张量
        sig["bbox"] = tf.TensorSpec((None, None, 4), tf.int32, name="bbox")
        return sig


LAYOUTLMV3_START_DOCSTRING = r"""
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

    - a single Tensor with `input_ids` only and nothing else: `model(input_ids)`
    - a list of varying length with one or several input Tensors IN THE ORDER given in the docstring:
    `model([input_ids, attention_mask])` or `model([input_ids, attention_mask, token_type_ids])`
    - a dictionary with one or several input Tensors associated to the input names given in the docstring:
    `model({"input_ids": input_ids, "token_type_ids": token_type_ids})`

    Note that when creating models and layers with
    [subclassing](https://keras.io/guides/making_new_layers_and_models_via_subclassing/) then you don't need to worry
    about any of this, as you can just pass inputs like you would to any other Python function!

    </Tip>

    Parameters:
        config ([`LayoutLMv3Config`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~TFPreTrainedModel.from_pretrained`] method to load the model weights.
"""

LAYOUTLMV3_INPUTS_DOCSTRING = r"""
"""


@add_start_docstrings(
    # 添加开始文档字符串注释，后续继续
    "The bare LayoutLMv3 Model transformer outputting raw hidden-states without any specific head on top.",
    LAYOUTLMV3_START_DOCSTRING,
)
class TFLayoutLMv3Model(TFLayoutLMv3PreTrainedModel):
    # names with a '.' represents the authorized unexpected/missing layers when a TF model is loaded from a PT model
    _keys_to_ignore_on_load_unexpected = [r"position_ids"]

    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.layoutlmv3 = TFLayoutLMv3MainLayer(config, name="layoutlmv3")

    @unpack_inputs
    @add_start_docstrings_to_model_forward(LAYOUTLMV3_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFBaseModelOutput, config_class=_CONFIG_FOR_DOC)
    def call(
        self,
        input_ids: tf.Tensor | None = None,
        bbox: tf.Tensor | None = None,
        attention_mask: tf.Tensor | None = None,
        token_type_ids: tf.Tensor | None = None,
        position_ids: tf.Tensor | None = None,
        head_mask: tf.Tensor | None = None,
        inputs_embeds: tf.Tensor | None = None,
        pixel_values: tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
    ) -> Union[
        TFBaseModelOutput,
        Tuple[tf.Tensor],
        Tuple[tf.Tensor, tf.Tensor],
        Tuple[tf.Tensor, tf.Tensor, tf.Tensor],
    ]:
        r"""
        Forward pass for the TFLayoutLMv3Model.
        
        Args:
            input_ids (tf.Tensor, optional): The input token IDs.
            bbox (tf.Tensor, optional): The bounding boxes of tokens.
            attention_mask (tf.Tensor, optional): The attention mask.
            token_type_ids (tf.Tensor, optional): The token type IDs.
            position_ids (tf.Tensor, optional): The position IDs.
            head_mask (tf.Tensor, optional): The mask for attention heads.
            inputs_embeds (tf.Tensor, optional): The embedded inputs.
            pixel_values (tf.Tensor, optional): The pixel values of images.
            output_attentions (bool, optional): Whether to output attentions.
            output_hidden_states (bool, optional): Whether to output hidden states.
            return_dict (bool, optional): Whether to return a dictionary.
            training (bool, optional): Whether in training mode.

        Returns:
            Union[TFBaseModelOutput, Tuple[tf.Tensor], Tuple[tf.Tensor, tf.Tensor], Tuple[tf.Tensor, tf.Tensor, tf.Tensor]]:
            The model outputs.

        Examples:
            Example usage of TFLayoutLMv3Model for token classification.

            ```
            >>> from transformers import AutoProcessor, TFAutoModel
            >>> from datasets import load_dataset

            >>> processor = AutoProcessor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
            >>> model = TFAutoModel.from_pretrained("microsoft/layoutlmv3-base")

            >>> dataset = load_dataset("nielsr/funsd-layoutlmv3", split="train")
            >>> example = dataset[0]
            >>> image = example["image"]
            >>> words = example["tokens"]
            >>> boxes = example["bboxes"]

            >>> encoding = processor(image, words, boxes=boxes, return_tensors="tf")

            >>> outputs = model(**encoding)
            >>> last_hidden_states = outputs.last_hidden_state
            ```
        """

        # Pass input arguments to the main layer TFLayoutLMv3MainLayer
        outputs = self.layoutlmv3(
            input_ids=input_ids,
            bbox=bbox,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        return outputs

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "layoutlmv3", None) is not None:
            with tf.name_scope(self.layoutlmv3.name):
                self.layoutlmv3.build(None)


class TFLayoutLMv3ClassificationHead(keras.layers.Layer):
    """
    Placeholder for the classification head of the TFLayoutLMv3Model.
    """
    Head for sentence-level classification tasks. Reference: RobertaClassificationHead
    """

    # 初始化函数，用于创建一个分类器头部对象
    def __init__(self, config: LayoutLMv3Config, **kwargs):
        super().__init__(**kwargs)
        
        # 创建一个全连接层，输出维度为config.hidden_size，激活函数为tanh
        self.dense = keras.layers.Dense(
            config.hidden_size,
            activation="tanh",
            kernel_initializer=get_initializer(config.initializer_range),
            name="dense",
        )
        
        # 设置分类器的dropout层，根据config中的设定选择classifier_dropout或者hidden_dropout_prob
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = keras.layers.Dropout(
            classifier_dropout,
            name="dropout",
        )
        
        # 创建一个全连接层，输出维度为config.num_labels，用于最终的输出投影
        self.out_proj = keras.layers.Dense(
            config.num_labels,
            kernel_initializer=get_initializer(config.initializer_range),
            name="out_proj",
        )
        
        # 保存配置信息
        self.config = config

    # 调用函数，用于执行前向传播
    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        # 对输入数据进行dropout处理
        outputs = self.dropout(inputs, training=training)
        
        # 经过全连接层dense处理
        outputs = self.dense(outputs)
        
        # 再次对处理后的结果进行dropout处理
        outputs = self.dropout(outputs, training=training)
        
        # 最终通过全连接层out_proj输出结果
        outputs = self.out_proj(outputs)
        return outputs

    # 构建函数，用于构建模型的层次结构
    def build(self, input_shape=None):
        if self.built:
            return
        
        # 标记模型已经构建
        self.built = True
        
        # 如果dense层存在，则构建dense层
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.hidden_size])
        
        # 如果dropout层存在，则构建dropout层
        if getattr(self, "dropout", None) is not None:
            with tf.name_scope(self.dropout.name):
                self.dropout.build(None)
        
        # 如果out_proj层存在，则构建out_proj层
        if getattr(self, "out_proj", None) is not None:
            with tf.name_scope(self.out_proj.name):
                self.out_proj.build([None, None, self.config.hidden_size])
"""
LayoutLMv3 Model with a sequence classification head on top (a linear layer on top of the final hidden state of the
[CLS] token) e.g. for document image classification tasks such as the
[RVL-CDIP](https://www.cs.cmu.edu/~aharley/rvl-cdip/) dataset.
"""
# 继承自 TFLayoutLMv3PreTrainedModel 和 TFSequenceClassificationLoss 的 TFLayoutLMv3ForSequenceClassification 类，
# 用于文档图像分类任务，通过在最终隐藏状态的[CLS]标记之上添加线性层来进行序列分类。
@add_start_docstrings(
    """
    LayoutLMv3 Model with a sequence classification head on top (a linear layer on top of the final hidden state of the
    [CLS] token) e.g. for document image classification tasks such as the
    [RVL-CDIP](https://www.cs.cmu.edu/~aharley/rvl-cdip/) dataset.
    """,
    LAYOUTLMV3_START_DOCSTRING,
)
class TFLayoutLMv3ForSequenceClassification(TFLayoutLMv3PreTrainedModel, TFSequenceClassificationLoss):
    # 在从 PT 模型加载 TF 模型时，忽略的授权外层或缺失层的名称列表，包含不带位置标识符的项
    _keys_to_ignore_on_load_unexpected = [r"position_ids"]

    def __init__(self, config: LayoutLMv3Config, **kwargs):
        super().__init__(config, **kwargs)
        self.config = config
        # 创建 LayoutLMv3 主层，并命名为 "layoutlmv3"
        self.layoutlmv3 = TFLayoutLMv3MainLayer(config, name="layoutlmv3")
        # 创建 LayoutLMv3 分类头，并命名为 "classifier"
        self.classifier = TFLayoutLMv3ClassificationHead(config, name="classifier")

    @unpack_inputs
    @add_start_docstrings_to_model_forward(LAYOUTLMV3_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFSequenceClassifierOutput, config_class=_CONFIG_FOR_DOC)
    # 调用函数，接收多种输入参数，返回 TFSequenceClassifierOutput 或其它类型的元组
    def call(
        self,
        input_ids: tf.Tensor | None = None,
        attention_mask: tf.Tensor | None = None,
        token_type_ids: tf.Tensor | None = None,
        position_ids: tf.Tensor | None = None,
        head_mask: tf.Tensor | None = None,
        inputs_embeds: tf.Tensor | None = None,
        labels: tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        bbox: tf.Tensor | None = None,
        pixel_values: tf.Tensor | None = None,
        training: Optional[bool] = False,
    ) -> Union[
        TFSequenceClassifierOutput,
        Tuple[tf.Tensor],
        Tuple[tf.Tensor, tf.Tensor],
        Tuple[tf.Tensor, tf.Tensor, tf.Tensor],
        Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor],
        # 多种返回类型的联合
    ]:
        """
        Returns:

        Examples:

        ```
        >>> from transformers import AutoProcessor, TFAutoModelForSequenceClassification
        >>> from datasets import load_dataset
        >>> import tensorflow as tf

        >>> processor = AutoProcessor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
        >>> model = TFAutoModelForSequenceClassification.from_pretrained("microsoft/layoutlmv3-base")

        >>> dataset = load_dataset("nielsr/funsd-layoutlmv3", split="train")
        >>> example = dataset[0]
        >>> image = example["image"]
        >>> words = example["tokens"]
        >>> boxes = example["bboxes"]

        >>> encoding = processor(image, words, boxes=boxes, return_tensors="tf")
        >>> sequence_label = tf.convert_to_tensor([1])

        >>> outputs = model(**encoding, labels=sequence_label)
        >>> loss = outputs.loss
        >>> logits = outputs.logits
        ```"""

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 如果 return_dict 参数为 None，则使用模型配置中的默认设置

        outputs = self.layoutlmv3(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            bbox=bbox,
            pixel_values=pixel_values,
            training=training,
        )
        # 使用 LayoutLMv3 模型处理输入数据，输出模型的各种结果

        sequence_output = outputs[0][:, 0, :]
        # 提取模型输出的序列输出的第一个位置的特征向量

        logits = self.classifier(sequence_output, training=training)
        # 使用分类器对序列输出进行分类预测

        loss = None if labels is None else self.hf_compute_loss(labels, logits)
        # 如果没有标签，则损失值为 None；否则计算模型预测与标签之间的损失

        if not return_dict:
            # 如果不要求返回字典格式的输出
            output = (logits,) + outputs[1:]
            # 构建输出元组，包含 logits 和模型其他输出
            return ((loss,) + output) if loss is not None else output
            # 如果有损失则将损失加入输出，否则只输出 logits 和其他结果

        return TFSequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        # 返回 TFSequenceClassifierOutput 格式的输出，包括损失、logits、隐藏状态和注意力权重

    def build(self, input_shape=None):
        if self.built:
            return
        # 如果模型已经建立则直接返回

        self.built = True
        # 设置模型已建立标志为 True

        if getattr(self, "layoutlmv3", None) is not None:
            # 如果模型有 layoutlmv3 属性
            with tf.name_scope(self.layoutlmv3.name):
                self.layoutlmv3.build(None)
                # 在 TensorFlow 的命名空间下构建 layoutlmv3 模型

        if getattr(self, "classifier", None) is not None:
            # 如果模型有 classifier 属性
            with tf.name_scope(self.classifier.name):
                self.classifier.build(None)
                # 在 TensorFlow 的命名空间下构建 classifier 分类器模型
"""
LayoutLMv3 Model with a token classification head on top (a linear layer on top of the final hidden states) e.g.
for sequence labeling (information extraction) tasks such as [FUNSD](https://guillaumejaume.github.io/FUNSD/),
[SROIE](https://rrc.cvc.uab.es/?ch=13), [CORD](https://github.com/clovaai/cord) and
[Kleister-NDA](https://github.com/applicaai/kleister-nda).

This class inherits from TFLayoutLMv3PreTrainedModel and TFTokenClassificationLoss. It provides a token
classification model specifically tailored for layout-aware tasks.

Attributes:
    _keys_to_ignore_on_load_unexpected (list): Names of layers to ignore when loading a TF model from a PT model.

Args:
    config (LayoutLMv3Config): Configuration class instance defining the model architecture and hyperparameters.
"""
@add_start_docstrings(
    """
    LayoutLMv3 Model with a token classification head on top (a linear layer on top of the final hidden states) e.g.
    for sequence labeling (information extraction) tasks such as [FUNSD](https://guillaumejaume.github.io/FUNSD/),
    [SROIE](https://rrc.cvc.uab.es/?ch=13), [CORD](https://github.com/clovaai/cord) and
    [Kleister-NDA](https://github.com/applicaai/kleister-nda).
    """,
    LAYOUTLMV3_START_DOCSTRING,
)
class TFLayoutLMv3ForTokenClassification(TFLayoutLMv3PreTrainedModel, TFTokenClassificationLoss):
    # names with a '.' represents the authorized unexpected/missing layers when a TF model is loaded from a PT model
    _keys_to_ignore_on_load_unexpected = [r"position_ids"]

    def __init__(self, config: LayoutLMv3Config, **kwargs):
        super().__init__(config, **kwargs)
        self.num_labels = config.num_labels

        # Initialize the main layers of the LayoutLMv3 model
        self.layoutlmv3 = TFLayoutLMv3MainLayer(config, name="layoutlmv3")
        self.dropout = keras.layers.Dropout(config.hidden_dropout_prob, name="dropout")

        # Initialize the classifier layer based on the number of labels in the configuration
        if config.num_labels < 10:
            self.classifier = keras.layers.Dense(
                config.num_labels,
                kernel_initializer=get_initializer(config.initializer_range),
                name="classifier",
            )
        else:
            self.classifier = TFLayoutLMv3ClassificationHead(config, name="classifier")

        self.config = config

    @unpack_inputs
    @add_start_docstrings_to_model_forward(LAYOUTLMV3_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFTokenClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def call(
        self,
        input_ids: tf.Tensor | None = None,
        bbox: tf.Tensor | None = None,
        attention_mask: tf.Tensor | None = None,
        token_type_ids: tf.Tensor | None = None,
        position_ids: tf.Tensor | None = None,
        head_mask: tf.Tensor | None = None,
        inputs_embeds: tf.Tensor | None = None,
        labels: tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: tf.Tensor | None = None,
        training: Optional[bool] = False,
    ) -> Union[
        TFTokenClassifierOutput,
        Tuple[tf.Tensor],
        Tuple[tf.Tensor, tf.Tensor],
        Tuple[tf.Tensor, tf.Tensor, tf.Tensor],
        Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor],
        # More return types depending on the inputs and configuration
    ]:
        """
        Performs the forward pass of the model for token classification.

        Args (depending on the input types):
            input_ids (tf.Tensor, optional): Tensor of input token IDs.
            bbox (tf.Tensor, optional): Tensor of bounding boxes for each token.
            attention_mask (tf.Tensor, optional): Mask indicating which tokens should be attended to.
            token_type_ids (tf.Tensor, optional): Type IDs to distinguish different sequences in the input.
            position_ids (tf.Tensor, optional): Positional IDs to indicate the position of tokens.
            head_mask (tf.Tensor, optional): Mask to hide certain heads in the self-attention layers.
            inputs_embeds (tf.Tensor, optional): Embedded inputs if the input tokens are already embedded.
            labels (tf.Tensor, optional): Labels for the token classification task.
            output_attentions (bool, optional): Whether to output attentions.
            output_hidden_states (bool, optional): Whether to output hidden states.
            return_dict (bool, optional): Whether to return a dictionary instead of a tuple of outputs.
            pixel_values (tf.Tensor, optional): Pixel values for image tokens if images are part of inputs.
            training (bool, optional): Whether the model is in training mode.

        Returns:
            TFTokenClassifierOutput or Tuple of Tensors: Output depending on the configuration and inputs.

        Raises:
            ValueError: If the configuration is invalid or incompatible with the model.
        """
        # 如果 `return_dict` 未指定，则使用模型配置中的默认设置来确定是否返回字典格式的输出
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用 LayoutLMv3 模型进行前向传播
        outputs = self.layoutlmv3(
            input_ids,
            bbox=bbox,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            pixel_values=pixel_values,
            training=training,
        )

        # 如果提供了 `input_ids`，则获取其形状；否则获取 `inputs_embeds` 的形状，去掉最后一维
        if input_ids is not None:
            input_shape = tf.shape(input_ids)
        else:
            input_shape = tf.shape(inputs_embeds)[:-1]

        # 获取序列的长度
        seq_length = input_shape[1]

        # 从模型输出中提取文本部分的表示
        sequence_output = outputs[0][:, :seq_length]

        # 在训练过程中对序列输出进行 dropout 操作
        sequence_output = self.dropout(sequence_output, training=training)

        # 将处理后的序列输出传入分类器以获得 logits
        logits = self.classifier(sequence_output)

        # 如果没有提供标签，则不计算损失
        loss = None if labels is None else self.hf_compute_loss(labels, logits)

        # 如果不要求返回字典格式的输出，则按需返回 logits 和其他输出信息
        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        # 返回 TFTokenClassifierOutput 类型的对象，包含损失、logits、隐藏状态和注意力权重
        return TFTokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    # 如果模型已经构建完成，则直接返回，避免重复构建
    if self.built:
        return
    # 将模型标记为已构建状态
    self.built = True
    
    # 如果存在 layoutlmv3 属性，则构建 layoutlmv3 模型部分
    if getattr(self, "layoutlmv3", None) is not None:
        # 使用 layoutlmv3 的名称作为命名空间
        with tf.name_scope(self.layoutlmv3.name):
            # 构建 layoutlmv3 模型
            self.layoutlmv3.build(None)
    
    # 如果存在 dropout 属性，则构建 dropout 模型部分
    if getattr(self, "dropout", None) is not None:
        # 使用 dropout 的名称作为命名空间
        with tf.name_scope(self.dropout.name):
            # 构建 dropout 模型
            self.dropout.build(None)
    
    # 如果存在 classifier 属性，则构建 classifier 模型部分
    if getattr(self, "classifier", None) is not None:
        # 使用 classifier 的名称作为命名空间
        with tf.name_scope(self.classifier.name):
            # 构建 classifier 模型，输入形状为 [None, None, self.config.hidden_size]
            self.classifier.build([None, None, self.config.hidden_size])
"""
LayoutLMv3 Model with a span classification head on top for extractive question-answering tasks such as
[DocVQA](https://rrc.cvc.uab.es/?ch=17) (a linear layer on top of the text part of the hidden-states output to
compute `span start logits` and `span end logits`).
"""
@add_start_docstrings(
    """
    LayoutLMv3 Model with a span classification head on top for extractive question-answering tasks such as
    [DocVQA](https://rrc.cvc.uab.es/?ch=17) (a linear layer on top of the text part of the hidden-states output to
    compute `span start logits` and `span end logits`).
    """,
    LAYOUTLMV3_START_DOCSTRING,
)
class TFLayoutLMv3ForQuestionAnswering(TFLayoutLMv3PreTrainedModel, TFQuestionAnsweringLoss):
    # names with a '.' represents the authorized unexpected/missing layers when a TF model is loaded from a PT model
    _keys_to_ignore_on_load_unexpected = [r"position_ids"]

    def __init__(self, config: LayoutLMv3Config, **kwargs):
        super().__init__(config, **kwargs)

        self.num_labels = config.num_labels

        # Initialize the main LayoutLMv3 layer with the provided configuration
        self.layoutlmv3 = TFLayoutLMv3MainLayer(config, name="layoutlmv3")
        
        # Initialize the question answering classification head for LayoutLMv3
        self.qa_outputs = TFLayoutLMv3ClassificationHead(config, name="qa_outputs")

    @unpack_inputs
    @add_start_docstrings_to_model_forward(LAYOUTLMV3_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFQuestionAnsweringModelOutput, config_class=_CONFIG_FOR_DOC)
    def call(
        self,
        input_ids: tf.Tensor | None = None,
        attention_mask: tf.Tensor | None = None,
        token_type_ids: tf.Tensor | None = None,
        position_ids: tf.Tensor | None = None,
        head_mask: tf.Tensor | None = None,
        inputs_embeds: tf.Tensor | None = None,
        start_positions: tf.Tensor | None = None,
        end_positions: tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        bbox: tf.Tensor | None = None,
        pixel_values: tf.Tensor | None = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
    ) -> Union[
        TFQuestionAnsweringModelOutput,
        Tuple[tf.Tensor],
        Tuple[tf.Tensor, tf.Tensor],
        Tuple[tf.Tensor, tf.Tensor, tf.Tensor],
        Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor],
    ]:
        """
        Forward pass of the TFLayoutLMv3ForQuestionAnswering model.
        
        Args:
            input_ids: Tensor of input token IDs.
            attention_mask: Tensor of attention mask.
            token_type_ids: Tensor of token type IDs.
            position_ids: Tensor of position IDs.
            head_mask: Tensor of head masks.
            inputs_embeds: Tensor of input embeddings.
            start_positions: Tensor of start positions for QA.
            end_positions: Tensor of end positions for QA.
            output_attentions: Whether to output attentions.
            output_hidden_states: Whether to output hidden states.
            bbox: Tensor of bounding boxes.
            pixel_values: Tensor of pixel values.
            return_dict: Whether to return a dictionary of outputs.
            training: Whether the model is in training mode.
        
        Returns:
            TFQuestionAnsweringModelOutput or tuple of output tensors.
        """

    def build(self, input_shape=None):
        """
        Builds the TFLayoutLMv3ForQuestionAnswering model.
        
        Args:
            input_shape: Shape of the input tensor.
        """
        if self.built:
            return
        
        self.built = True
        
        # Build the LayoutLMv3 main layer if it exists
        if getattr(self, "layoutlmv3", None) is not None:
            with tf.name_scope(self.layoutlmv3.name):
                self.layoutlmv3.build(None)
        
        # Build the QA classification head if it exists
        if getattr(self, "qa_outputs", None) is not None:
            with tf.name_scope(self.qa_outputs.name):
                self.qa_outputs.build(None)
```