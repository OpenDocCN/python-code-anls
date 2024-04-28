# `.\transformers\models\layoutlmv3\modeling_tf_layoutlmv3.py`

```
# 设置编码格式为 UTF-8
# 版权声明
# 根据 Apache 2.0 许可协议，可以在符合许可内容的前提下使用该文件
# 获取许可协议的副本，访问 http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则按 "原样" 分发软件
# 没有任何形式的保证或条件，无论是明示的还是暗示的
# 请查看许可协议以获取具体语言的权限和限制

"""TF 2.0 LayoutLMv3 model."""

# 导入必要的包
# 从 __future__ 模块中导入 annotations
import collections
import math
from typing import List, Optional, Tuple, Union

import tensorflow as tf

# 导入必要的模块和类
# 从 ...activations_tf 模块中导入 get_tf_activation
# 从 ...modeling_tf_outputs 模块中导入 TFBaseModelOutput, TFQuestionAnsweringModelOutput, TFSequenceClassifierOutput, TFTokenClassifierOutput
# 从 ...modeling_tf_utils 模块中导入 TFPreTrainedModel, TFQuestionAnsweringLoss, TFSequenceClassificationLoss, TFTokenClassificationLoss, get_initializer, keras_serializable, unpack_inputs
# 从 ...tf_utils 模块中导入 check_embeddings_within_bounds
# 从 ...utils 模块中导入 add_start_docstrings, add_start_docstrings_to_model_forward, replace_return_docstrings
# 从 .configuration_layoutlmv3 模块中导入 LayoutLMv3Config

# 将 "LayoutLMv3Config" 存储到 _CONFIG_FOR_DOC 变量中
_CONFIG_FOR_DOC = "LayoutLMv3Config"

# 设置 DUMMY_INPUT_IDS 变量
# 用于后续示例
_DUMMY_INPUT_IDS = [
    [7, 6, 1],
    [1, 2, 0],
]

# 设置 DUMMY_BBOX 变量
# 用于后续示例
_DUMMY_BBOX = [
    [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
    [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]],
]

# 声明 LayoutLMv3 预训练模型的存档列表
TF_LAYOUTLMV3_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "microsoft/layoutlmv3-base",
    "microsoft/layoutlmv3-large",
    # See all LayoutLMv3 models at https://huggingface.co/models?filter=layoutlmv3
]

# 定义 LARGE_NEGATIVE 常量
LARGE_NEGATIVE = -1e8

# 定义 TFLayoutLMv3PatchEmbeddings 类
class TFLayoutLMv3PatchEmbeddings(tf.keras.layers.Layer):
    """LayoutLMv3 image (patch) embeddings."""

    def __init__(self, config: LayoutLMv3Config, **kwargs):
        # 执行父类初始化方法
        super().__init__(**kwargs)
        
        # 根据配置初始化图像 (patch) 嵌入对象
        patch_sizes = (
            config.patch_size
            if isinstance(config.patch_size, collections.abc.Iterable)
            else (config.patch_size, config.patch_size)
        )
        
        # 使用 Conv2D 层进行投影
        self.proj = tf.keras.layers.Conv2D(
            filters=config.hidden_size,
            kernel_size=patch_sizes,
            strides=patch_sizes,
            padding="valid",
            data_format="channels_last",
            use_bias=True,
            kernel_initializer=get_initializer(config.initializer_range),
            name="proj",
        )
        self.hidden_size = config.hidden_size
        # 计算图像中的补丁数
        self.num_patches = (config.input_size**2) // (patch_sizes[0] * patch_sizes[1])
        self.config = config
    # 定义一个方法call，接受输入像素值tf.Tensor，返回一个tf.Tensor
    def call(self, pixel_values: tf.Tensor) -> tf.Tensor:
        # 当在 CPU 上运行时，tf.keras.layers.Conv2D 不支持 NCHW 格式，所以需要将输入格式从 NCHW 转换为 NHWC
        pixel_values = tf.transpose(pixel_values, perm=[0, 2, 3, 1])

        # 使用self.proj对像素值进行投影
        embeddings = self.proj(pixel_values)
        # 重新调整embeddings的形状
        embeddings = tf.reshape(embeddings, (-1, self.num_patches, self.hidden_size))
        return embeddings

    # 定义一个build方法，接受输入形状input_shape，默认为None
    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回
        if self.built:
            return
        self.built = True
        # 如果self.proj存在
        if getattr(self, "proj", None) is not None:
            # 在self.proj的名字范围内构建投影层
            with tf.name_scope(self.proj.name):
                self.proj.build([None, None, None, self.config.num_channels])
class TFLayoutLMv3TextEmbeddings(tf.keras.layers.Layer):
    """
    LayoutLMv3 text embeddings. Same as `RobertaEmbeddings` but with added spatial (layout) embeddings.
    """

    def __init__(self, config: LayoutLMv3Config, **kwargs):
        super().__init__(**kwargs)  # 调用父类的构造函数初始化对象
        # 定义词嵌入层，将词的索引映射为对应的词嵌入向量
        self.word_embeddings = tf.keras.layers.Embedding(
            config.vocab_size,
            config.hidden_size,
            embeddings_initializer=get_initializer(config.initializer_range),  # 使用指定初始化器初始化嵌入矩阵
            name="word_embeddings",  # 层的名称
        )
        # 定义标记类型嵌入层，将标记类型的索引映射为对应的嵌入向量
        self.token_type_embeddings = tf.keras.layers.Embedding(
            config.type_vocab_size,
            config.hidden_size,
            embeddings_initializer=get_initializer(config.initializer_range),  # 使用指定初始化器初始化嵌入矩阵
            name="token_type_embeddings",  # 层的名称
        )
        # LayerNorm 层，用于层标准化
        self.LayerNorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
        # Dropout 层，用于随机失活，防止过拟合
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout_prob)
        # 填充标记的索引
        self.padding_token_index = config.pad_token_id
        # 位置嵌入层，将位置索引映射为对应的位置嵌入向量
        self.position_embeddings = tf.keras.layers.Embedding(
            config.max_position_embeddings,
            config.hidden_size,
            embeddings_initializer=get_initializer(config.initializer_range),  # 使用指定初始化器初始化嵌入矩阵
            name="position_embeddings",  # 层的名称
        )
        # X 方向位置嵌入层，将 X 方向位置索引映射为对应的 X 方向位置嵌入向量
        self.x_position_embeddings = tf.keras.layers.Embedding(
            config.max_2d_position_embeddings,
            config.coordinate_size,
            embeddings_initializer=get_initializer(config.initializer_range),  # 使用指定初始化器初始化嵌入矩阵
            name="x_position_embeddings",  # 层的名称
        )
        # Y 方向位置嵌入层，将 Y 方向位置索引映射为对应的 Y 方向位置嵌入向量
        self.y_position_embeddings = tf.keras.layers.Embedding(
            config.max_2d_position_embeddings,
            config.coordinate_size,
            embeddings_initializer=get_initializer(config.initializer_range),  # 使用指定初始化器初始化嵌入矩阵
            name="y_position_embeddings",  # 层的名称
        )
        # 高度位置嵌入层，将高度索引映射为对应的高度位置嵌入向量
        self.h_position_embeddings = tf.keras.layers.Embedding(
            config.max_2d_position_embeddings,
            config.shape_size,
            embeddings_initializer=get_initializer(config.initializer_range),  # 使用指定初始化器初始化嵌入矩阵
            name="h_position_embeddings",  # 层的名称
        )
        # 宽度位置嵌入层，将宽度索引映射为对应的宽度位置嵌入向量
        self.w_position_embeddings = tf.keras.layers.Embedding(
            config.max_2d_position_embeddings,
            config.shape_size,
            embeddings_initializer=get_initializer(config.initializer_range),  # 使用指定初始化器初始化嵌入矩阵
            name="w_position_embeddings",  # 层的名称
        )
        # 最大二维位置
        self.max_2d_positions = config.max_2d_position_embeddings
        # 配置
        self.config = config
    # 计算空间位置嵌入，接收边界框参数bbox，返回空间位置嵌入
    def calculate_spatial_position_embeddings(self, bbox: tf.Tensor) -> tf.Tensor:
        try:
            # 从bbox中提取左边界位置索引
            left_position_ids = bbox[:, :, 0]
            # 从bbox中提取上边界位置索引
            upper_position_ids = bbox[:, :, 1]
            # 从bbox中提取右边界位置索引
            right_position_ids = bbox[:, :, 2]
            # 从bbox中提取下边界位置索引
            lower_position_ids = bbox[:, :, 3]
        except IndexError as exception:
            # 抛出异常，说明bbox的形状不是(batch_size, seq_length, 4)
            raise IndexError("Bounding box is not of shape (batch_size, seq_length, 4).") from exception

        try:
            # 根据左边界位置索引获取x方向位置嵌入
            left_position_embeddings = self.x_position_embeddings(left_position_ids)
            # 根据上边界位置索引获取y方向位置嵌入
            upper_position_embeddings = self.y_position_embeddings(upper_position_ids)
            # 根据右边界位置索引获取x方向位置嵌入
            right_position_embeddings = self.x_position_embeddings(right_position_ids)
            # 根据下边界位置索引获取y方向位置嵌入
            lower_position_embeddings = self.y_position_embeddings(lower_position_ids)
        except IndexError as exception:
            # 抛出异常，说明bbox的坐标值应在0-{self.max_2d_positions}范围内
            raise IndexError(
                f"The `bbox` coordinate values should be within 0-{self.max_2d_positions} range."
            ) from exception

        max_position_id = self.max_2d_positions - 1
        # 计算高度方向位置嵌入，限制在0-max_position_id范围内
        h_position_embeddings = self.h_position_embeddings(
            tf.clip_by_value(bbox[:, :, 3] - bbox[:, :, 1], 0, max_position_id)
        )
        # 计算宽度方向位置嵌入，限制在0-max_position_id范围内
        w_position_embeddings = self.w_position_embeddings(
            tf.clip_by_value(bbox[:, :, 2] - bbox[:, :, 0], 0, max_position_id)
        )

        # LayoutLMv1将空间嵌入相加，LayoutLMv3将其连接起来
        # 按照最后一个维度连接所有位置嵌入
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

    # 根据输入的嵌入向量生成位置ID
    def create_position_ids_from_inputs_embeds(self, inputs_embds: tf.Tensor) -> tf.Tensor:
        """
        We are provided embeddings directly. We cannot infer which are padded, so just generate sequential position
        ids.
        """
        # 获取输入嵌入向量的形状
        input_shape = tf.shape(inputs_embds)
        # 获取序列长度
        sequence_length = input_shape[1]
        # 计算起始位置索引
        start_index = self.padding_token_index + 1
        # 计算结束位置索引
        end_index = self.padding_token_index + sequence_length + 1
        # 生成顺序的位置ID
        position_ids = tf.range(start_index, end_index, dtype=tf.int32)
        # 获取批次大小
        batch_size = input_shape[0]
        # 将位置ID形状调整为(1, sequence_length)
        position_ids = tf.reshape(position_ids, (1, sequence_length))
        # 将位置ID扩展为(batch_size, sequence_length)
        position_ids = tf.tile(position_ids, (batch_size, 1))
        return position_ids
    # 从输入的 input_ids 张量中创建位置编码张量，用非填充符号替换填充符号，位置编码从 padding_token_index + 1 开始
    def create_position_ids_from_input_ids(self, input_ids: tf.Tensor) -> tf.Tensor:
        """
        Replace non-padding symbols with their position numbers. Position numbers begin at padding_token_index + 1.
        """
        # 创建一个布尔类型的掩码张量，标记哪些位置为非填充符号
        mask = tf.cast(tf.not_equal(input_ids, self.padding_token_index), input_ids.dtype)
        # 计算位置编码张量，位置编码从 padding_token_index + 1 开始
        position_ids = tf.cumsum(mask, axis=1) * mask
        position_ids = position_ids + self.padding_token_index
        return position_ids

    # 根据输入的 input_ids 和 inputs_embeds 张量创建位置编码张量
    def create_position_ids(self, input_ids: tf.Tensor, inputs_embeds: tf.Tensor) -> tf.Tensor:
        # 如果 input_ids 为 None，根据 inputs_embeds 创建位置编码张量
        if input_ids is None:
            return self.create_position_ids_from_inputs_embeds(inputs_embeds)
        else:
            return self.create_position_ids_from_input_ids(input_ids)

    # 模型的调用函数，接收多个输入张量，返回嵌入张量
    def call(
        self,
        input_ids: tf.Tensor | None = None,
        bbox: tf.Tensor = None,
        token_type_ids: tf.Tensor | None = None,
        position_ids: tf.Tensor | None = None,
        inputs_embeds: tf.Tensor | None = None,
        training: bool = False,
    ) -> tf.Tensor:
        # 如果位置编码张量为 None，根据 input_ids 和 inputs_embeds 创建位置编码张量
        if position_ids is None:
            position_ids = self.create_position_ids(input_ids, inputs_embeds)

        # 根据 input_ids 或 inputs_embeds 计算输入张量的形状
        if input_ids is not None:
            input_shape = tf.shape(input_ids)
        else:
            input_shape = tf.shape(inputs_embeds)[:-1]

        # 如果 token_type_ids 为 None，创建一个与输入形状一致、元素为零的张量作为 token_type_ids
        if token_type_ids is None:
            token_type_ids = tf.zeros(input_shape, dtype=position_ids.dtype)

        # 如果 inputs_embeds 为 None，检查 input_ids 是否在词嵌入范围内，然后使用词嵌入层获取嵌入张量
        if inputs_embeds is None:
            check_embeddings_within_bounds(input_ids, self.word_embeddings.input_dim)
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        # 将词嵌入张量和 token 类型嵌入张量相加
        embeddings = inputs_embeds + token_type_embeddings
        position_embeddings = self.position_embeddings(position_ids)
        # 将位置编码张量加到总嵌入张量中
        embeddings += position_embeddings

        # 计算空间位置编码张量
        spatial_position_embeddings = self.calculate_spatial_position_embeddings(bbox)

        # 将空间位置编码张量加到总嵌入张量中
        embeddings += spatial_position_embeddings

        # 对总嵌入张量进行 LayerNormalization
        embeddings = self.LayerNorm(embeddings)
        # 对嵌入张量进行 dropout 处理
        embeddings = self.dropout(embeddings, training=training)
        return embeddings
    # 构建神经网络模型的方法
    def build(self, input_shape=None):
        # 如果模型已经构建，则直接返回
        if self.built:
            return
        # 设置模型已构建标志为真
        self.built = True
        # 如果存在词嵌入层，则构建词嵌入层
        if getattr(self, "word_embeddings", None) is not None:
            # 在命名空间中构建词嵌入层
            with tf.name_scope(self.word_embeddings.name):
                self.word_embeddings.build(None)
        # 如果存在标记类型嵌入层，则构建标记类型嵌入层
        if getattr(self, "token_type_embeddings", None) is not None:
            with tf.name_scope(self.token_type_embeddings.name):
                self.token_type_embeddings.build(None)
        # 如果存在 LayerNorm 层，则构建 LayerNorm 层
        if getattr(self, "LayerNorm", None) is not None:
            with tf.name_scope(self.LayerNorm.name):
                self.LayerNorm.build([None, None, self.config.hidden_size])
        # 如果存在位置嵌入层，则构建位置嵌入层
        if getattr(self, "position_embeddings", None) is not None:
            with tf.name_scope(self.position_embeddings.name):
                self.position_embeddings.build(None)
        # 如果存在 x 位置嵌入层，则构建 x 位置嵌入层
        if getattr(self, "x_position_embeddings", None) is not None:
            with tf.name_scope(self.x_position_embeddings.name):
                self.x_position_embeddings.build(None)
        # 如果存在 y 位置嵌入层，则构建 y 位置嵌入层
        if getattr(self, "y_position_embeddings", None) is not None:
            with tf.name_scope(self.y_position_embeddings.name):
                self.y_position_embeddings.build(None)
        # 如果存在 h 位置嵌入层，则构建 h 位置嵌入层
        if getattr(self, "h_position_embeddings", None) is not None:
            with tf.name_scope(self.h_position_embeddings.name):
                self.h_position_embeddings.build(None)
        # 如果存在 w 位置嵌入层，则构建 w 位置嵌入层
        if getattr(self, "w_position_embeddings", None) is not None:
            with tf.name_scope(self.w_position_embeddings.name):
                self.w_position_embeddings.build(None)
class TFLayoutLMv3SelfAttention(tf.keras.layers.Layer):
    def __init__(self, config: LayoutLMv3Config, **kwargs):
        super().__init__(**kwargs)
        # 检查隐藏层大小是否是注意力头数的倍数，如果不是则引发错误
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        # 初始化注意力头数和每个注意力头的大小
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        # 用于归一化注意力分数的缩放因子
        self.attention_score_normaliser = math.sqrt(self.attention_head_size)

        # 初始化查询、键和值的全连接层
        self.query = tf.keras.layers.Dense(
            self.all_head_size,
            kernel_initializer=get_initializer(config.initializer_range),
            name="query",
        )
        self.key = tf.keras.layers.Dense(
            self.all_head_size,
            kernel_initializer=get_initializer(config.initializer_range),
            name="key",
        )
        self.value = tf.keras.layers.Dense(
            self.all_head_size,
            kernel_initializer=get_initializer(config.initializer_range),
            name="value",
        )

        # 初始化 Dropout 层
        self.dropout = tf.keras.layers.Dropout(config.attention_probs_dropout_prob)
        # 是否具有相对位置注意力偏置
        self.has_relative_attention_bias = config.has_relative_attention_bias
        # 是否具有空间位置注意力偏置
        self.has_spatial_attention_bias = config.has_spatial_attention_bias
        self.config = config

    def transpose_for_scores(self, x: tf.Tensor):
        # 将张量形状转换为(batch_size, seq_length, num_heads, head_size)
        shape = tf.shape(x)
        new_shape = (
            shape[0],  # batch_size
            shape[1],  # seq_length
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = tf.reshape(x, new_shape)
        return tf.transpose(x, perm=[0, 2, 1, 3])  # 输出形状为(batch_size, num_heads, seq_length, attention_head_size)

    def cogview_attention(self, attention_scores: tf.Tensor, alpha: Union[float, int] = 32):
        """
        https://arxiv.org/abs/2105.13290 Section 2.4 Stabilization of training: Precision Bottleneck Relaxation
        (PB-Relax). A replacement of the original tf.keras.layers.Softmax(axis=-1)(attention_scores). Seems the new
        attention_probs will result in a slower speed and a little bias. Can use
        tf.debugging.assert_near(standard_attention_probs, cogview_attention_probs, atol=1e-08) for comparison. The
        smaller atol (e.g., 1e-08), the better.
        """
        # 缩放注意力分数并进行轴向 softmax 操作
        scaled_attention_scores = attention_scores / alpha
        max_value = tf.expand_dims(tf.reduce_max(scaled_attention_scores, axis=-1), axis=-1)
        new_attention_scores = (scaled_attention_scores - max_value) * alpha
        return tf.math.softmax(new_attention_scores, axis=-1)
        # 定义一个方法，用于计算自注意力机制
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
            # 计算key、value和query
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            query_layer = self.transpose_for_scores(self.query(hidden_states))

            # 计算"query"和"key"之间的点积，得到原始注意力分数
            normalised_query_layer = query_layer / self.attention_score_normaliser
            transposed_key_layer = tf.transpose(
                key_layer, perm=[0, 1, 3, 2]
            )  # batch_size, num_heads, attention_head_size, seq_length
            attention_scores = tf.matmul(normalised_query_layer, transposed_key_layer)

            # 如果存在相对注意力偏置和空间注意力偏置，则将其添加到注意力分数中
            if self.has_relative_attention_bias and self.has_spatial_attention_bias:
                attention_scores += (rel_pos + rel_2d_pos) / self.attention_score_normaliser
            # 如果只存在相对注意力偏置，则将其添加到注意力分数中
            elif self.has_relative_attention_bias:
                attention_scores += rel_pos / self.attention_score_normaliser

            # 如果存在注意力掩码，则将其应用到注意力分数中
            if attention_mask is not None:
                attention_scores += attention_mask

            # 将注意力分数归一化为概率
            # 使用CogView论文中的技巧来稳定训练
            attention_probs = self.cogview_attention(attention_scores)

            # 使用dropout来防止过拟合
            attention_probs = self.dropout(attention_probs, training=training)

            # 如果需要，对注意力概率进行遮罩处理
            if head_mask is not None:
                attention_probs = attention_probs * head_mask

            # 计算上下文向量
            context_layer = tf.matmul(attention_probs, value_layer)
            context_layer = tf.transpose(
                context_layer, perm=[0, 2, 1, 3]
            )  # batch_size, seq_length, num_heads, attention_head_size
            shape = tf.shape(context_layer)
            context_layer = tf.reshape(
                context_layer, (shape[0], shape[1], self.all_head_size)
            )  # batch_size, seq_length, num_heads * attention_head_size

            # 根据是否需要输出注意力，返回上下文向量或包括注意力概率的元组
            outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

            return outputs
    # 构建函数，用于构建自定义层的模型，接受输入形状作为参数
    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回，不再重复构建
        if self.built:
            return
        # 将构建状态标记为已构建
        self.built = True
        # 如果存在查询（query）属性，则构建查询张量
        if getattr(self, "query", None) is not None:
            # 在命名空间下构建查询张量，命名空间为查询张量的名称
            with tf.name_scope(self.query.name):
                # 构建查询张量，形状为 [None, None, self.config.hidden_size]
                self.query.build([None, None, self.config.hidden_size])
        # 如果存在键（key）属性，则构建键张量
        if getattr(self, "key", None) is not None:
            # 在命名空间下构建键张量，命名空间为键张量的名称
            with tf.name_scope(self.key.name):
                # 构建键张量，形状为 [None, None, self.config.hidden_size]
                self.key.build([None, None, self.config.hidden_size])
        # 如果存在值（value）属性，则构建值张量
        if getattr(self, "value", None) is not None:
            # 在命名空间下构建值张量，命名空间为值张量的名称
            with tf.name_scope(self.value.name):
                # 构建值张量，形状为 [None, None, self.config.hidden_size]
                self.value.build([None, None, self.config.hidden_size])
# 从 models.roberta.modeling_tf_roberta.TFRobertaSelfOutput 复制得到 TFLayoutLMv3SelfOutput 类
class TFLayoutLMv3SelfOutput(tf.keras.layers.Layer):
    def __init__(self, config: LayoutLMv3Config, **kwargs):
        super().__init__(**kwargs)

        # 创建一个全连接层，用于调整隐藏状态的维度
        self.dense = tf.keras.layers.Dense(
            units=config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )
        # 创建 LayerNormalization 层，用于归一化隐藏状态
        self.LayerNorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
        # 创建 Dropout 层，用于随机失活隐藏状态中的部分神经元
        self.dropout = tf.keras.layers.Dropout(rate=config.hidden_dropout_prob)
        self.config = config

    # 定义 call 方法，用于对输入进行处理得到输出
    def call(self, hidden_states: tf.Tensor, input_tensor: tf.Tensor, training: bool = False) -> tf.Tensor:
        # 使用全连接层调整隐藏状态维度
        hidden_states = self.dense(inputs=hidden_states)
        # 使用 Dropout 层随机失活隐藏状态中的部分神经元
        hidden_states = self.dropout(inputs=hidden_states, training=training)
        # 使用 LayerNormalization 层进行隐藏状态的归一化处理
        hidden_states = self.LayerNorm(inputs=hidden_states + input_tensor)

        # 返回处理后的隐藏状态
        return hidden_states

    # 定义 build 方法，用于构建层的参数
    def build(self, input_shape=None):
        # 如果已经构建，则直接返回
        if self.built:
            return
        self.built = True
        # 如果存在全连接层，则构建全连接层的参数
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.hidden_size])
        # 如果存在 LayerNormalization 层，则构建 LayerNormalization 层的参数
        if getattr(self, "LayerNorm", None) is not None:
            with tf.name_scope(self.LayerNorm.name):
                self.LayerNorm.build([None, None, self.config.hidden_size])


# 定义 TFLayoutLMv3Attention 类
class TFLayoutLMv3Attention(tf.keras.layers.Layer):
    def __init__(self, config: LayoutLMv3Config, **kwargs):
        super().__init__(**kwargs)
        # 创建 TFLayoutLMv3SelfAttention 层，用于处理自注意力机制
        self.self_attention = TFLayoutLMv3SelfAttention(config, name="self")
        # 创建 TFLayoutLMv3SelfOutput 层，用于处理自注意力输出
        self.self_output = TFLayoutLMv3SelfOutput(config, name="output")

    # 定义 call 方法，用于对输入进行处理得到输出
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
        # 调用 self_attention 层进行自注意力处理
        self_outputs = self.self_attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions,
            rel_pos,
            rel_2d_pos,
            training=training,
        )
        # 调用 self_output 层处理 self_attention 的输出
        attention_output = self.self_output(self_outputs[0], hidden_states, training=training)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs

    # 定义 build 方法，用于构建层的参数
    def build(self, input_shape=None):
        # 如果已经构建，则直接返回
        if self.built:
            return
        self.built = True
        # 如果存在 self_attention 层，则构建其参数
        if getattr(self, "self_attention", None) is not None:
            with tf.name_scope(self.self_attention.name):
                self.self_attention.build(None)
        # 如果存在 self_output 层，则构建其参数
        if getattr(self, "self_output", None) is not None:
            with tf.name_scope(self.self_output.name):
                self.self_output.build(None)
# 从模型.roberta.modeling_tf_bert.TFRobertaIntermediate中复制过来的类，用于中间层计算
class TFLayoutLMv3Intermediate(tf.keras.layers.Layer):
    def __init__(self, config: LayoutLMv3Config, **kwargs):
        super().__init__(**kwargs)

        # 创建一个全连接层，用于中间层计算，设置输出单元数和初始化方式
        self.dense = tf.keras.layers.Dense(
            units=config.intermediate_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )

        # 如果hidden_act是字符串，则使用get_tf_activation函数获取激活函数，否则直接使用提供的激活函数
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = get_tf_activation(config.hidden_act)
        else:
            self.intermediate_act_fn = config.hidden_act
        self.config = config

    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        # 经过全连接层计算
        hidden_states = self.dense(inputs=hidden_states)
        # 经过激活函数处理
        hidden_states = self.intermediate_act_fn(hidden_states)

        return hidden_states

    def build(self, input_shape=None):
        # 如果已经建立了该层，则直接返回
        if self.built:
            return
        self.built = True
        # 如果存在全连接层，则构建全连接层
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.hidden_size])


# 从模型.roberta.modeling_tf_bert.TFRobertaOutput中复制过来的类，用于输出层计算
class TFLayoutLMv3Output(tf.keras.layers.Layer):
    def __init__(self, config: LayoutLMv3Config, **kwargs):
        super().__init__(**kwargs)

        # 创建一个全连接层，用于输出层计算，设置输出单元数和初始化方式
        self.dense = tf.keras.layers.Dense(
            units=config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )
        # 创建一个LayerNorm层，用于归一化处理
        self.LayerNorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
        # 创建一个Dropout层，用于随机失活
        self.dropout = tf.keras.layers.Dropout(rate=config.hidden_dropout_prob)
        self.config = config

    def call(self, hidden_states: tf.Tensor, input_tensor: tf.Tensor, training: bool = False) -> tf.Tensor:
        # 经过全连接层计算
        hidden_states = self.dense(inputs=hidden_states)
        # 经过Dropout层处理
        hidden_states = self.dropout(inputs=hidden_states, training=training)
        # 经过LayerNorm层归一化处理并与输入张量相加
        hidden_states = self.LayerNorm(inputs=hidden_states + input_tensor)

        return hidden_states

    def build(self, input_shape=None):
        # 如果已经建立了该层，则直接返回
        if self.built:
            return
        self.built = True
        # 如果存在全连接层，则构建全连接层
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.intermediate_size])
        # 如果存在LayerNorm层，则构建LayerNorm层
        if getattr(self, "LayerNorm", None) is not None:
            with tf.name_scope(self.LayerNorm.name):
                self.LayerNorm.build([None, None, self.config.hidden_size])


# TFLayoutLMv3Layer类，包含了Attention层、Intermediate层和Output层
class TFLayoutLMv3Layer(tf.keras.layers.Layer):
    def __init__(self, config: LayoutLMv3Config, **kwargs):
        super().__init__(**kwargs)
        # 创建Attention层
        self.attention = TFLayoutLMv3Attention(config, name="attention")
        # 创建Intermediate层
        self.intermediate = TFLayoutLMv3Intermediate(config, name="intermediate")
        # 创建Output层
        self.bert_output = TFLayoutLMv3Output(config, name="output")
    # 定义一个方法，用于执行自注意力机制
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
        # 执行自注意力机制，获取自注意力输出
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            rel_pos=rel_pos,
            rel_2d_pos=rel_2d_pos,
            training=training,
        )
        # 获取自注意力输出的第一个元素，即注意力输出
        attention_output = self_attention_outputs[0]
        # 如果输出注意力权重，则将自注意力输出添加到输出中
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights
        # 将注意力输出传入中间层
        intermediate_output = self.intermediate(attention_output)
        # 将中间层输出和注意力输出传入BERT输出层
        layer_output = self.bert_output(intermediate_output, attention_output, training=training)
        # 更新输出，将本层的输出添加到输出元组中
        outputs = (layer_output,) + outputs
        # 返回输出
        return outputs

    # 构建网络层
    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回
        if self.built:
            return
        # 标记为已构建
        self.built = True
        # 构建注意力层
        if getattr(self, "attention", None) is not None:
            with tf.name_scope(self.attention.name):
                self.attention.build(None)
        # 构建中间层
        if getattr(self, "intermediate", None) is not None:
            with tf.name_scope(self.intermediate.name):
                self.intermediate.build(None)
        # 构建BERT输出层
        if getattr(self, "bert_output", None) is not None:
            with tf.name_scope(self.bert_output.name):
                self.bert_output.build(None)
# 定义 TFLayoutLMv3Encoder 类，用于 LayoutLMv3 模型的编码器部分
class TFLayoutLMv3Encoder(tf.keras.layers.Layer):
    def __init__(self, config: LayoutLMv3Config, **kwargs):
        # 调用父类的初始化方法
        super().__init__(**kwargs)
        # 将配置参数保存到对象属性中
        self.config = config
        # 创建多个 TFLayoutLMv3Layer 层，构成编码器的多层结构
        self.layer = [TFLayoutLMv3Layer(config, name=f"layer.{i}") for i in range(config.num_hidden_layers)]

        # 检查是否存在相对注意力偏置
        self.has_relative_attention_bias = config.has_relative_attention_bias
        # 检查是否存在空间注意力偏置
        self.has_spatial_attention_bias = config.has_spatial_attention_bias

        # 如果存在相对注意力偏置
        if self.has_relative_attention_bias:
            # 获取相对位置编码的参数
            self.rel_pos_bins = config.rel_pos_bins
            self.max_rel_pos = config.max_rel_pos
            # 创建用于相对位置偏置的 Dense 层
            self.rel_pos_bias = tf.keras.layers.Dense(
                units=config.num_attention_heads,
                kernel_initializer=get_initializer(config.initializer_range),
                use_bias=False,
                name="rel_pos_bias",
            )

        # 如果存在空间注意力偏置
        if self.has_spatial_attention_bias:
            # 获取二维相对位置编码的参数
            self.max_rel_2d_pos = config.max_rel_2d_pos
            self.rel_2d_pos_bins = config.rel_2d_pos_bins
            # 创建用于 X 方向相对位置偏置的 Dense 层
            self.rel_pos_x_bias = tf.keras.layers.Dense(
                units=config.num_attention_heads,
                kernel_initializer=get_initializer(config.initializer_range),
                use_bias=False,
                name="rel_pos_x_bias",
            )
            # 创建用于 Y 方向相对位置偏置的 Dense 层
            self.rel_pos_y_bias = tf.keras.layers.Dense(
                units=config.num_attention_heads,
                kernel_initializer=get_initializer(config.initializer_range),
                use_bias=False,
                name="rel_pos_y_bias",
            )
    #计算相对位置桶
    def relative_position_bucket(self, relative_positions: tf.Tensor, num_buckets: int, max_distance: int):
        # 将负相对位置分配给间隔[0， num_buckets / 2]
        # 我们通过将绝对相对位置分配给区间[0， num_buckets / 2]
        # 并在最后将正相对位置偏移num_buckets / 2来处理这个问题
        num_buckets = num_buckets // 2
        buckets = tf.abs(relative_positions)

        # 一半的桶用于确切的位置增量
        max_exact_buckets = num_buckets // 2
        is_small = buckets < max_exact_buckets

        # 另一半的桶用于对数增大的位置的较大容器，最大距离为max_distance
        buckets_log_ratio = tf.math.log(tf.cast(buckets, tf.float32) / max_exact_buckets)
        distance_log_ratio = math.log(max_distance / max_exact_buckets)
        buckets_big_offset = (
            buckets_log_ratio / distance_log_ratio * (num_buckets - max_exact_buckets)
        )  # 比例为[0, num_buckets - max_exact_buckets]
        buckets_big = max_exact_buckets + buckets_big_offset  # 比例为[max_exact_buckets, num_buckets]
        buckets_big = tf.cast(buckets_big, buckets.dtype)
        buckets_big = tf.minimum(buckets_big, num_buckets - 1)

        return (tf.cast(relative_positions > 0, buckets.dtype) * num_buckets) + tf.where(
            is_small, buckets, buckets_big
        )

    #计算位置嵌入
    def _cal_pos_emb(
        self,
        dense_layer: tf.keras.layers.Dense,
        position_ids: tf.Tensor,
        num_buckets: int,
        max_distance: int,
    ):
        rel_pos_matrix = tf.expand_dims(position_ids, axis=-2) - tf.expand_dims(position_ids, axis=-1)
        rel_pos = self.relative_position_bucket(rel_pos_matrix, num_buckets, max_distance)
        rel_pos_one_hot = tf.one_hot(rel_pos, depth=num_buckets, dtype=self.compute_dtype)
        embedding = dense_layer(rel_pos_one_hot)
        # batch_size, seq_length, seq_length, num_heads --> batch_size, num_heads, seq_length, seq_length
        embedding = tf.transpose(embedding, [0, 3, 1, 2])
        embedding = tf.cast(embedding, dtype=self.compute_dtype)
        return embedding

    #计算1维位置嵌入
    def _cal_1d_pos_emb(self, position_ids: tf.Tensor):
        return self._cal_pos_emb(self.rel_pos_bias, position_ids, self.rel_pos_bins, self.max_rel_pos)

    #计算2维位置嵌入
    def _cal_2d_pos_emb(self, bbox: tf.Tensor):
        position_coord_x = bbox[:, :, 0]  # 左边
        position_coord_y = bbox[:, :, 3]  # 底部
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
    # 定义一个带有多个参数和返回类型的方法
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
        # 如果output_hidden_states为真，则设置all_hidden_states为一个空元组，否则为None
        all_hidden_states = () if output_hidden_states else None
        # 如果output_attentions为真，则设置all_self_attentions为一个空元组，否则为None
        all_self_attentions = () if output_attentions else None

        # 如果模型支持相对位置注意力偏置，则计算1D位置嵌入
        rel_pos = self._cal_1d_pos_emb(position_ids) if self.has_relative_attention_bias else None
        # 如果模型支持空间注意力偏置，则计算2D位置嵌入
        rel_2d_pos = self._cal_2d_pos_emb(bbox) if self.has_spatial_attention_bias else None

        # 遍历模型的每个层
        for i, layer_module in enumerate(self.layer):
            # 如果output_hidden_states为真，则将当前hidden_states添加到all_hidden_states中
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 获取当前层的头部掩码
            layer_head_mask = head_mask[i] if head_mask is not None else None

            # 调用当前层模块的方法，传入相关参数
            layer_outputs = layer_module(
                hidden_states,
                attention_mask,
                layer_head_mask,
                output_attentions,
                rel_pos=rel_pos,
                rel_2d_pos=rel_2d_pos,
                training=training,
            )

            # 更新hidden_states为当前层输出的第一个元素
            hidden_states = layer_outputs[0]
            # 如果output_attentions为真，则将当前层输出的第二个元素添加到all_self_attentions中
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        # 如果output_hidden_states为真，则将最终的hidden_states添加到all_hidden_states中
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 如果return_dict为真，则返回TFBaseModelOutput对象，包含最终的hidden_states、all_hidden_states和all_self_attentions
        if return_dict:
            return TFBaseModelOutput(
                last_hidden_state=hidden_states,
                hidden_states=all_hidden_states,
                attentions=all_self_attentions,
            )
        # 如果return_dict为假，则返回包含非空元素的元组，包括hidden_states、all_hidden_states和all_self_attentions
        else:
            return tuple(
                value for value in [hidden_states, all_hidden_states, all_self_attentions] if value is not None
            )

    # 构建方法，用于构建模型
    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回，避免重复构建
        if self.built:
            return
        # 设置标志为已构建
        self.built = True
        # 如果rel_pos_bias属性存在，则构建1D相对位置偏置
        if getattr(self, "rel_pos_bias", None) is not None:
            with tf.name_scope(self.rel_pos_bias.name):
                self.rel_pos_bias.build([None, None, self.rel_pos_bins])
        # 如果rel_pos_x_bias属性存在，则构建2D X轴相对位置偏置
        if getattr(self, "rel_pos_x_bias", None) is not None:
            with tf.name_scope(self.rel_pos_x_bias.name):
                self.rel_pos_x_bias.build([None, None, self.rel_2d_pos_bins])
        # 如果rel_pos_y_bias属性存在，则构建2D Y轴相对位置偏置
        if getattr(self, "rel_pos_y_bias", None) is not None:
            with tf.name_scope(self.rel_pos_y_bias.name):
                self.rel_pos_y_bias.build([None, None, self.rel_2d_pos_bins])
        # 如果layer属性存在，则遍历每个层并构建
        if getattr(self, "layer", None) is not None:
            for layer in self.layer:
                with tf.name_scope(layer.name):
                    layer.build(None)
# 定义一个 TensorFlow 的 Layer 类，用于实现 LayoutLMv3 模型的主层
@keras_serializable
class TFLayoutLMv3MainLayer(tf.keras.layers.Layer):
    # 指定配置类为 LayoutLMv3Config
    config_class = LayoutLMv3Config

    # 初始化方法
    def __init__(self, config: LayoutLMv3Config, **kwargs):
        super().__init__(**kwargs)

        # 保存传入的配置
        self.config = config

        # 如果配置中包含文本嵌入，则初始化文本嵌入层
        if config.text_embed:
            self.embeddings = TFLayoutLMv3TextEmbeddings(config, name="embeddings")

        # 如果配置中包含视觉嵌入，则初始化视觉嵌入、LayerNormalization 和 Dropout 层
        if config.visual_embed:
            self.patch_embed = TFLayoutLMv3PatchEmbeddings(config, name="patch_embed")
            self.LayerNorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
            self.dropout = tf.keras.layers.Dropout(config.hidden_dropout_prob, name="dropout")

            # 如果配置中包含相对注意力偏置或空间注意力偏置，则初始化视觉边界框
            if config.has_relative_attention_bias or config.has_spatial_attention_bias:
                image_size = config.input_size // config.patch_size
                self.init_visual_bbox(image_size=(image_size, image_size))

            # 初始化额外的 LayerNormalization 层
            self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="norm")

        # 初始化编码器
        self.encoder = TFLayoutLMv3Encoder(config, name="encoder")

    # 构建方法
    def build(self, input_shape=None):
        # 如果配置中包含视觉嵌入，则初始化特殊的词元和位置嵌入
        if self.config.visual_embed:
            image_size = self.config.input_size // self.config.patch_size
            self.cls_token = self.add_weight(
                shape=(1, 1, self.config.hidden_size),
                initializer="zeros",
                trainable=True,
                dtype=tf.float32,
                name="cls_token",
            )
            self.pos_embed = self.add_weight(
                shape=(1, image_size * image_size + 1, self.config.hidden_size),
                initializer="zeros",
                trainable=True,
                dtype=tf.float32,
                name="pos_embed",
            )

        # 如果已经构建完毕，则直接返回
        if self.built:
            return
        self.built = True
        
        # 构建编码器
        if getattr(self, "encoder", None) is not None:
            with tf.name_scope(self.encoder.name):
                self.encoder.build(None)
        
        # 构建文本嵌入层
        if getattr(self, "embeddings", None) is not None:
            with tf.name_scope(self.embeddings.name):
                self.embeddings.build(None)
        
        # 构建视觉嵌入层
        if getattr(self, "patch_embed", None) is not None:
            with tf.name_scope(self.patch_embed.name):
                self.patch_embed.build(None)
        
        # 构建 LayerNormalization 层
        if getattr(self, "LayerNorm", None) is not None:
            with tf.name_scope(self.LayerNorm.name):
                self.LayerNorm.build([None, None, self.config.hidden_size])
        
        # 构建 Dropout 层
        if getattr(self, "dropout", None) is not None:
            with tf.name_scope(self.dropout.name):
                self.dropout.build(None)
        
        # 构建额外的 LayerNormalization 层
        if getattr(self, "norm", None) is not None:
            with tf.name_scope(self.norm.name):
                self.norm.build([None, None, self.config.hidden_size])

    # 获取输入嵌入层的方法
    def get_input_embeddings(self) -> tf.keras.layers.Layer:
        return self.embeddings.word_embeddings
    # 将给定的 value 赋值给模型的词嵌入权重
    def set_input_embeddings(self, value: tf.Variable):
        self.embeddings.word_embeddings.weight = value

    # 从 transformers 库中复制的函数，用于剪枝模型中的注意力头
    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        raise NotImplementedError

    # 初始化视觉区域边界框，主要是计算视觉区域的坐标
    def init_visual_bbox(self, image_size: Tuple[int, int], max_len: int = 1000):
        # 强制使用 1000，而不是通过配置文件传递，这是为了与预训练权重保持一致，更正确的方法应该是传递 max_len=config.max_2d_position_embeddings - 1。
        height, width = image_size

        visual_bbox_x = tf.range(0, max_len * (width + 1), max_len) // width
        visual_bbox_x = tf.expand_dims(visual_bbox_x, axis=0)
        visual_bbox_x = tf.tile(visual_bbox_x, [width, 1])  # (width, width + 1)

        visual_bbox_y = tf.range(0, max_len * (height + 1), max_len) // height
        visual_bbox_y = tf.expand_dims(visual_bbox_y, axis=1)
        visual_bbox_y = tf.tile(visual_bbox_y, [1, height])  # (height + 1, height)

        visual_bbox = tf.stack(
            [visual_bbox_x[:, :-1], visual_bbox_y[:-1], visual_bbox_x[:, 1:], visual_bbox_y[1:]],
            axis=-1,
        )
        visual_bbox = tf.reshape(visual_bbox, [-1, 4])

        cls_token_box = tf.constant([[1, 1, max_len - 1, max_len - 1]], dtype=tf.int32)
        self.visual_bbox = tf.concat([cls_token_box, visual_bbox], axis=0)

    # 计算视觉区域边界框，返回一个形状为 [batch_size, num_boxes, 4] 的张量
    def calculate_visual_bbox(self, batch_size: int, dtype: tf.DType):
        visual_bbox = tf.expand_dims(self.visual_bbox, axis=0)
        visual_bbox = tf.tile(visual_bbox, [batch_size, 1, 1])
        visual_bbox = tf.cast(visual_bbox, dtype=dtype)
        return visual_bbox

    # 对图片进行嵌入，包括通过 patch_embed 对图片进行编码，添加 [CLS] 标记，添加位置嵌入，最后进行归一化处理
    def embed_image(self, pixel_values: tf.Tensor) -> tf.Tensor:
        embeddings = self.patch_embed(pixel_values)

        # add [CLS] token
        batch_size = tf.shape(embeddings)[0]
        cls_tokens = tf.tile(self.cls_token, [batch_size, 1, 1])
        embeddings = tf.concat([cls_tokens, embeddings], axis=1)

        # add position embeddings
        if getattr(self, "pos_embed", None) is not None:
            embeddings += self.pos_embed

        embeddings = self.norm(embeddings)
        return embeddings
    def get_extended_attention_mask(self, attention_mask: tf.Tensor) -> tf.Tensor:
        # 从 transformers.modelling_utils.ModuleUtilsMixin.get_extended_attention_mask 调整而来

        # 获取注意力掩码张量的维度
        n_dims = len(attention_mask.shape)

        # 如果维度为3，则提供了自定义的自注意力掩码 [batch_size, from_seq_length, to_seq_length]，
        # 需要将其广播到所有头部
        if n_dims == 3:
            extended_attention_mask = tf.expand_dims(attention_mask, axis=1)
        elif n_dims == 2:
            # 如果提供了维度为 [batch_size, seq_length] 的填充掩码
            # 将掩码广播到 [batch_size, num_heads, seq_length, seq_length]
            extended_attention_mask = tf.expand_dims(attention_mask, axis=1)  # (batch_size, 1, seq_length)
            extended_attention_mask = tf.expand_dims(extended_attention_mask, axis=1)  # (batch_size, 1, 1, seq_length)
        else:
            raise ValueError(f"Wrong shape for attention_mask (shape {attention_mask.shape}).")

        # 由于 attention_mask 是对我们想要关注的位置为1.0，对被屏蔽的位置代表为0.0
        # 该操作将创建一个张量，对于我们想要关注的位置为0.0，对被屏蔽的位置为-10000.0
        # 这样在 softmax 之前将其加到原始分数中，等效于将它们全部移除
        extended_attention_mask = tf.cast(extended_attention_mask, self.compute_dtype)
            extended_attention_mask = (1.0 - extended_attention_mask) * LARGE_NEGATIVE

        return extended_attention_mask
    def get_head_mask(self, head_mask: tf.Tensor | None) -> Union[tf.Tensor, List[tf.Tensor | None]]:
        # 如果头部掩码为空，则返回与隐藏层数量相同数量的 None
        if head_mask is None:
            return [None] * self.config.num_hidden_layers

        # 获取头部掩码的维度
        n_dims = tf.rank(head_mask)
        # 如果头部掩码的维度为 1
        if n_dims == 1:
            # 获取每个头部的掩码张量
            head_mask = tf.expand_dims(head_mask, axis=0)  # 1, num_heads
            head_mask = tf.expand_dims(head_mask, axis=0)  # 1, 1, num_heads
            head_mask = tf.expand_dims(head_mask, axis=-1)  # 1, 1, num_heads, 1
            head_mask = tf.expand_dims(head_mask, axis=-1)  # 1, 1, num_heads, 1, 1
            # 复制头部掩码以匹配隐藏层数量
            head_mask = tf.tile(
                head_mask, [self.config.num_hidden_layers, 1, 1, 1, 1]
            )  # seq_length, 1, num_heads, 1, 1
        # 如果头部掩码的维度为 2
        elif n_dims == 2:
            # 获取每个层和头部的掩码张量
            head_mask = tf.expand_dims(head_mask, axis=1)  # seq_length, 1, num_heads
            head_mask = tf.expand_dims(head_mask, axis=-1)  # seq_length, 1, num_heads, 1
            head_mask = tf.expand_dims(head_mask, axis=-1)  # seq_length, 1, num_heads, 1, 1
        # 如果头部掩码的维度不为 5，则抛出异常
        elif n_dims != 5:
            raise ValueError(f"Wrong shape for head_mask (shape {head_mask.shape}).")
        # 断言头部掩码的维度为 5
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
# TFLayoutLMv3PreTrainedModel 类继承自 TFPreTrainedModel 类
# 用于处理权重初始化、下载和加载预训练模型的简单接口
class TFLayoutLMv3PreTrainedModel(TFPreTrainedModel):

    # config_class 属性指定了配置文件为 LayoutLMv3Config
    config_class = LayoutLMv3Config
    # base_model_prefix 属性为 "layoutlmv3"
    base_model_prefix = "layoutlmv3"

    # input_signature 属性为输入签名，定义了输入的张量规范
    @property
    def input_signature(self):
        # 获取父类的输入签名
        sig = super().input_signature
        # 添加 bbox 参数到签名中
        sig["bbox"] = tf.TensorSpec((None, None, 4), tf.int32, name="bbox")
        return sig

# LAYOUTLMV3_START_DOCSTRING 为模型的起始文档字符串，包括对模型的描述和使用的提示
LAYOUTLMV3_START_DOCSTRING = r"""

# LAYOUTLMV3_INPUTS_DOCSTRING 为模型的输入文档字符串，用于描述输入的格式和使用方法
LAYOUTLMV3_INPUTS_DOCSTRING = r"""

# add_start_docstrings() 为装饰器函数，用于向函数添加起始文档字符串
@add_start_docstrings(
    # 定义字符串描述 LayoutLMv3 模型，该模型输出原始隐藏状态，没有特定的输出头
    # 开始 LayoutLMv3 文档字符串
)
# TFLayoutLMv3Model类，继承自TFLayoutLMv3PreTrainedModel类
class TFLayoutLMv3Model(TFLayoutLMv3PreTrainedModel):
    # names with a '.' represents the authorized unexpected/missing layers when a TF model is loaded from a PT model
    # 定义在从PT模型加载TF模型时授权的意外/缺失层
    _keys_to_ignore_on_load_unexpected = [r"position_ids"]

    # 初始化方法
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        # 实例化TFLayoutLMv3MainLayer类，传入config并命名为"layoutlmv3"
        self.layoutlmv3 = TFLayoutLMv3MainLayer(config, name="layoutlmv3")

    # call方法，用于前向传播
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
        Returns:

        Examples:

        ```python
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
        ```"""

        # 调用layoutlmv3模型进行前向传播
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

        # 返回模型的输出
        return outputs

    # 构建方法
    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 如果layoutlmv3存在，则构建
        if getattr(self, "layoutlmv3", None) is not None:
            with tf.name_scope(self.layoutlmv3.name):
                self.layoutlmv3.build(None)


class TFLayoutLMv3ClassificationHead(tf.keras.layers.Layer):
    """
    # 句子级分类任务的头部结构。参考：RobertaClassificationHead
    class SentenceClassificationHead(tf.keras.layers.Layer):
        
        def __init__(self, config: LayoutLMv3Config, **kwargs):
            super().__init__(**kwargs)
            
            # 创建一个全连接层，输出维度为 hidden_size，激活函数为双曲正切函数，权重初始化方式为配置中的 initializer_range
            self.dense = tf.keras.layers.Dense(
                config.hidden_size,
                activation="tanh",
                kernel_initializer=get_initializer(config.initializer_range),
                name="dense",
            )
            
            # 分类器的 dropout 比率，默认为 hidden_dropout_prob；如果未设置，则使用 hidden_dropout_prob
            classifier_dropout = (
                config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
            )
            
            # 创建 dropout 层，丢弃率为 classifier_dropout
            self.dropout = tf.keras.layers.Dropout(
                classifier_dropout,
                name="dropout",
            )
            
            # 创建一个全连接层，输出维度为 num_labels，权重初始化方式为配置中的 initializer_range
            self.out_proj = tf.keras.layers.Dense(
                config.num_labels,
                kernel_initializer=get_initializer(config.initializer_range),
                name="out_proj",
            )
            
            # 保存配置信息
            self.config = config
        
        # 前向计算过程
        def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
            
            # 使用 dropout 层对输入进行随机失活（dropout）处理
            outputs = self.dropout(inputs, training=training)
            
            # 使用全连接层进行线性变换
            outputs = self.dense(outputs)
            
            # 再次使用 dropout 层对输出进行随机失活（dropout）处理
            outputs = self.dropout(outputs, training=training)
            
            # 使用全连接层进行线性变换
            outputs = self.out_proj(outputs)
            
            return outputs
        
        # 构建网络层
        def build(self, input_shape=None):
            
            # 如果已经构建，则直接返回
            if self.built:
                return
            
            self.built = True
            
            # 如果存在 dense 层对象，则构建 dense 层的权重
            if getattr(self, "dense", None) is not None:
                with tf.name_scope(self.dense.name):
                    self.dense.build([None, None, self.config.hidden_size])
            
            # 如果存在 dropout 层对象，则构建 dropout 层的权重
            if getattr(self, "dropout", None) is not None:
                with tf.name_scope(self.dropout.name):
                    self.dropout.build(None)
            
            # 如果存在 out_proj 层对象，则构建 out_proj 层的权重
            if getattr(self, "out_proj", None) is not None:
                with tf.name_scope(self.out_proj.name):
                    self.out_proj.build([None, None, self.config.hidden_size])
@add_start_docstrings(
    """
    LayoutLMv3 Model with a sequence classification head on top (a linear layer on top of the final hidden state of the
    [CLS] token) e.g. for document image classification tasks such as the
    [RVL-CDIP](https://www.cs.cmu.edu/~aharley/rvl-cdip/) dataset.
    """,
    LAYOUTLMV3_START_DOCSTRING,
)
class TFLayoutLMv3ForSequenceClassification(TFLayoutLMv3PreTrainedModel, TFSequenceClassificationLoss):
    # names with a '.' represents the authorized unexpected/missing layers when a TF model is loaded from a PT model
    _keys_to_ignore_on_load_unexpected = [r"position_ids"]

    def __init__(self, config: LayoutLMv3Config, **kwargs):
        super().__init__(config, **kwargs)
        self.config = config
        self.layoutlmv3 = TFLayoutLMv3MainLayer(config, name="layoutlmv3")  # 初始化LayoutLMv3主层
        self.classifier = TFLayoutLMv3ClassificationHead(config, name="classifier")  # 初始化LayoutLMv3分类头部

    @unpack_inputs
    @add_start_docstrings_to_model_forward(LAYOUTLMV3_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFSequenceClassifierOutput, config_class=_CONFIG_FOR_DOC)
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
    ]:
        """
        Returns:

        Examples:

        ```python
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

        # 检查是否使用了返回字典
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用 LayoutLMv3 模型，传入各种参数
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
        # 从输出中取出序列的最后一个隐藏状态
        sequence_output = outputs[0][:, 0, :]
        # 通过分类器得到 logits
        logits = self.classifier(sequence_output, training=training)

        # 如果没有标签则损失为 None，否则计算损失
        loss = None if labels is None else self.hf_compute_loss(labels, logits)

        # 如果不要求返回字典，则返回 logits 和其他输出
        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        # 否则返回 TFSequenceClassifierOutput 对象
        return TFSequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def build(self, input_shape=None):
        # 如果已经构建过则直接返回
        if self.built:
            return
        self.built = True
        # 如果 LayoutLMv3 模型存在则构建该模型
        if getattr(self, "layoutlmv3", None) is not None:
            with tf.name_scope(self.layoutlmv3.name):
                self.layoutlmv3.build(None)
        # 如果分类器存在则构建该分类器
        if getattr(self, "classifier", None) is not None:
            with tf.name_scope(self.classifier.name):
                self.classifier.build(None)
```  
# 将 LayoutLMv3 模型与一个标记分类头部连接在一起，用于序列标记任务，如信息提取任务
# 例如 FUNSD、SROIE、CORD 和 Kleister-NDA
@add_start_docstrings(
    """
    LayoutLMv3 Model with a token classification head on top (a linear layer on top of the final hidden states) e.g.
    for sequence labeling (information extraction) tasks such as [FUNSD](https://guillaumejaume.github.io/FUNSD/),
    [SROIE](https://rrc.cvc.uab.es/?ch=13), [CORD](https://github.com/clovaai/cord) and
    [Kleister-NDA](https://github.com/applicaai/kleister-nda).
    """,
    LAYOUTLMV3_START_DOCSTRING,
)

# 定义 TFLayoutLMv3ForTokenClassification 类，继承自 TFLayoutLMv3PreTrainedModel 和 TFTokenClassificationLoss
class TFLayoutLMv3ForTokenClassification(TFLayoutLMv3PreTrainedModel, TFTokenClassificationLoss):
    # names with a '.' represents the authorized unexpected/missing layers when a TF model is loaded from a PT model
    # 在从 PT 模型加载 TF 模型时，'.' 名称表示授权的意外/缺失层
    _keys_to_ignore_on_load_unexpected = [r"position_ids"]

    # 初始化方法
    def __init__(self, config: LayoutLMv3Config, **kwargs):
        # 调用父类的初始化方法
        super().__init__(config, **kwargs)
        # 设置标签数量
        self.num_labels = config.num_labels

        # 创建 LayoutLMv3 主层对象
        self.layoutlmv3 = TFLayoutLMv3MainLayer(config, name="layoutlmv3")
        # 创建 dropout 层
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout_prob, name="dropout")
        
        # 根据标签数量不同，选择合适的分类头
        if config.num_labels < 10:
            self.classifier = tf.keras.layers.Dense(
                config.num_labels,
                kernel_initializer=get_initializer(config.initializer_range),
                name="classifier",
            )
        else:
            self.classifier = TFLayoutLMv3ClassificationHead(config, name="classifier")
        self.config = config

    # 定义模型的调用方法
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
        # 定义函数的参数，labels为标记，用于计算标记分类损失。索引应在 `[0, ..., config.num_labels - 1]` 范围内。
        labels (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
        Returns:

        Examples:

        ```python
        >>> from transformers import AutoProcessor, TFAutoModelForTokenClassification
        >>> from datasets import load_dataset

        >>> processor = AutoProcessor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
        >>> model = TFAutoModelForTokenClassification.from_pretrained("microsoft/layoutlmv3-base", num_labels=7)

        >>> dataset = load_dataset("nielsr/funsd-layoutlmv3", split="train")
        >>> example = dataset[0]
        >>> image = example["image"]
        >>> words = example["tokens"]
        >>> boxes = example["bboxes"]
        >>> word_labels = example["ner_tags"]

        >>> encoding = processor(image, words, boxes=boxes, word_labels=word_labels, return_tensors="tf")

        >>> outputs = model(**encoding)
        >>> loss = outputs.loss
        >>> logits = outputs.logits
        ```"""

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用 layoutlmv3 模型的前向计算
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
        if input_ids is not None:
            input_shape = tf.shape(input_ids)
        else:
            input_shape = tf.shape(inputs_embeds)[:-1]

        seq_length = input_shape[1]
        # 仅获取输出表示的文本部分
        sequence_output = outputs[0][:, :seq_length]
        sequence_output = self.dropout(sequence_output, training=training)
        # 使用分类器进行标记分类
        logits = self.classifier(sequence_output)

        # 如果不存在标签，损失为None，否则计算损失
        loss = None if labels is None else self.hf_compute_loss(labels, logits)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        # 返回 TokenClassifierOutput 类型的对象
        return TFTokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    # 定义一个方法来构建模型
    def build(self, input_shape=None):
        # 如果模型已经构建则直接返回
        if self.built:
            return
        # 标记模型已经构建
        self.built = True
        # 如果存在 layoutlmv3 属性，则构建 layoutlmv3 对象
        if getattr(self, "layoutlmv3", None) is not None:
            with tf.name_scope(self.layoutlmv3.name):
                self.layoutlmv3.build(None)
        # 如果存在 dropout 属性，则构建 dropout 对象
        if getattr(self, "dropout", None) is not None:
            with tf.name_scope(self.dropout.name):
                self.dropout.build(None)
        # 如果存在 classifier 属性，则构建 classifier 对象
        if getattr(self, "classifier", None) is not None:
            with tf.name_scope(self.classifier.name):
                self.classifier.build([None, None, self.config.hidden_size])
# 为提取式问答任务添加一个带有跨度分类头部的LayoutLMv3模型，例如DocVQA (在隐藏状态输出的文本部分上面的线性层来计算`span start logits`和`span end logits`)
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

        self.layoutlmv3 = TFLayoutLMv3MainLayer(config, name="layoutlmv3")
        self.qa_outputs = TFLayoutLMv3ClassificationHead(config, name="qa_outputs")

    # 根据输入及参数的不同情况提供模型前向计算的说明
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
    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "layoutlmv3", None) is not None:
            with tf.name_scope(self.layoutlmv3.name):
                self.layoutlmv3.build(None)
        if getattr(self, "qa_outputs", None) is not None:
            with tf.name_scope(self.qa_outputs.name):
                self.qa_outputs.build(None)
```