# `.\models\pegasus\modeling_tf_pegasus.py`

```
# 定义函数 shift_tokens_right，将输入的 token 序列向右移动一位，用于生成模型的 decoder 输入
def shift_tokens_right(input_ids: tf.Tensor, pad_token_id: int, decoder_start_token_id: int):
    # 将 pad_token_id 和 decoder_start_token_id 转换为与 input_ids 相同的数据类型
    pad_token_id = tf.cast(pad_token_id, input_ids.dtype)
    decoder_start_token_id = tf.cast(decoder_start_token_id, input_ids.dtype)
    
    # 创建与 input_ids 相同大小的起始 token 序列
    start_tokens = tf.fill(
        (shape_list(input_ids)[0], 1),  # 使用 shape_list 获取 batch 大小，填充为列向量
        tf.convert_to_tensor(decoder_start_token_id, input_ids.dtype)  # 转换 decoder_start_token_id 的数据类型
    )
    
    # 将 start_tokens 和 input_ids 向右移动一位拼接起来，构成 shifted_input_ids
    shifted_input_ids = tf.concat([start_tokens, input_ids[:, :-1]], -1)
    
    # 将 labels 中可能的 -100 值替换为 pad_token_id
    shifted_input_ids = tf.where(
        shifted_input_ids == -100,  # 找到所有值为 -100 的位置
        tf.fill(shape_list(shifted_input_ids), tf.convert_to_tensor(pad_token_id, input_ids.dtype)),  # 替换为 pad_token_id
        shifted_input_ids,  # 否则保持原值不变
    )
    
    # 断言 shifted_input_ids 中的值大于等于 0，确保 labels 中的值为正值或 -100
    assert_gte0 = tf.debugging.assert_greater_equal(shifted_input_ids, tf.constant(0, dtype=input_ids.dtype))
    
    # 确保断言操作被调用，通过包装结果在 identity 操作中
    with tf.control_dependencies([assert_gte0]):
        shifted_input_ids = tf.identity(shifted_input_ids)
    
    # 返回处理后的 shifted_input_ids
    return shifted_input_ids
# 创建一个用于双向自注意力的因果掩码
def _make_causal_mask(input_ids_shape: tf.TensorShape, past_key_values_length: int = 0):
    """
    Make causal mask used for bi-directional self-attention.
    创建用于双向自注意力的因果掩码。
    """
    bsz = input_ids_shape[0]  # 获取批量大小
    tgt_len = input_ids_shape[1]  # 获取目标长度
    mask = tf.ones((tgt_len, tgt_len)) * LARGE_NEGATIVE  # 创建全为大负数的掩码矩阵
    mask_cond = tf.range(shape_list(mask)[-1])  # 生成一个形状与掩码最后一维相同的范围

    # 将掩码中的上三角区域置零，形成因果掩码
    mask = tf.where(mask_cond < tf.reshape(mask_cond + 1, (shape_list(mask)[-1], 1)), 0.0, mask)

    if past_key_values_length > 0:
        # 如果过去键值的长度大于零，则在掩码左侧添加零矩阵
        mask = tf.concat([tf.zeros((tgt_len, past_key_values_length)), mask], axis=-1)

    return tf.tile(mask[None, None, :, :], (bsz, 1, 1, 1))


# 从transformers.models.bart.modeling_tf_bart._expand_mask复制过来
def _expand_mask(mask: tf.Tensor, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    将注意力掩码从 `[bsz, seq_len]` 扩展到 `[bsz, 1, tgt_seq_len, src_seq_len]`。
    """
    src_len = shape_list(mask)[1]  # 获取掩码的源长度
    tgt_len = tgt_len if tgt_len is not None else src_len  # 如果目标长度不为None，则使用目标长度，否则使用源长度
    one_cst = tf.constant(1.0)  # 创建常数值为1.0的张量
    mask = tf.cast(mask, dtype=one_cst.dtype)  # 将掩码转换为与one_cst相同的数据类型
    expanded_mask = tf.tile(mask[:, None, None, :], (1, 1, tgt_len, 1))  # 在掩码的第二、三维度上复制掩码，以扩展维度

    return (one_cst - expanded_mask) * LARGE_NEGATIVE  # 返回取反后乘以大负数的扩展掩码


# 从transformers.models.marian.modeling_tf_marian.TFMarianSinusoidalPositionalEmbedding复制过来，将Marian改为Pegasus
class TFPegasusSinusoidalPositionalEmbedding(keras.layers.Layer):
    """This module produces sinusoidal positional embeddings of any length.
    该模块生成任意长度的正弦位置嵌入。
    """

    def __init__(self, num_positions: int, embedding_dim: int, **kwargs):
        super().__init__(**kwargs)

        if embedding_dim % 2 != 0:
            raise NotImplementedError(f"odd embedding_dim {embedding_dim} not supported")

        self.embedding_dim = embedding_dim  # 嵌入维度
        self.num_positions = num_positions  # 位置数量

    def build(self, input_shape: tf.TensorShape):
        """
        Build shared token embedding layer Shared weights logic adapted from
        https://github.com/tensorflow/models/blob/a009f4fb9d2fc4949e32192a944688925ef78659/official/transformer/v2/embedding_layer.py#L24
        构建共享的标记嵌入层，权重初始化逻辑参考自上述链接的实现。
        """

        weight = self._init_weight(self.num_positions, self.embedding_dim)  # 初始化权重

        self.weight = self.add_weight(
            name="embeddings",
            shape=[self.num_positions, self.embedding_dim],
        )
        weight = tf.cast(weight, dtype=self.weight.dtype)  # 将权重转换为与self.weight相同的数据类型

        self.weight.assign(weight)  # 分配权重

        super().build(input_shape)  # 调用父类的build方法
    def _init_weight(n_pos: int, dim: int):
        """
        Identical to the XLM create_sinusoidal_embeddings except features are not interleaved. The cos features are in
        the 2nd half of the vector. [dim // 2:]
        """
        # 创建一个二维数组，每行代表一个位置编码向量，计算公式与 Transformer 中的位置编码相同
        position_enc = np.array(
            [[pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)] for pos in range(n_pos)]
        )
        # 创建一个与 position_enc 相同形状的全零数组
        table = np.zeros_like(position_enc)
        # 将 position_enc 中的 sin 值复制到 table 的前半部分
        table[:, 0 : dim // 2] = np.sin(position_enc[:, 0::2])
        # 将 position_enc 中的 cos 值复制到 table 的后半部分
        table[:, dim // 2 :] = np.cos(position_enc[:, 1::2])
        # 将 table 转换为 TensorFlow 的 tensor 对象
        table = tf.convert_to_tensor(table)
        # 停止梯度在 table 上的传播
        tf.stop_gradient(table)
        return table

    def call(
        self, input_shape: tf.TensorShape, past_key_values_length: int = 0, position_ids: tf.Tensor | None = None
    ):
        """Input is expected to be of size [bsz x seqlen]."""
        # 如果未提供位置编码，根据输入的形状创建位置编码的索引
        if position_ids is None:
            seq_len = input_shape[1]
            position_ids = tf.range(past_key_values_length, seq_len + past_key_values_length, delta=1, name="range")
        # 根据位置编码索引从预先初始化的权重表中获取位置编码向量
        return tf.gather(self.weight, position_ids)
# 从 transformers.models.bart.modeling_tf_bart.TFBartAttention 复制而来，将 Bart 改为 Pegasus
class TFPegasusAttention(keras.layers.Layer):
    """Multi-headed attention from "Attention Is All You Need"""

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
        self.embed_dim = embed_dim  # 设置注意力机制的嵌入维度

        self.num_heads = num_heads  # 头数，即注意力头的数量
        self.dropout = keras.layers.Dropout(dropout)  # Dropout 层，用于随机失活
        self.head_dim = embed_dim // num_heads  # 每个注意力头的维度
        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"  # 抛出异常，如果 embed_dim 不能被 num_heads 整除
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5  # 缩放因子，用于注意力计算时的数值稳定性
        self.is_decoder = is_decoder  # 是否为解码器

        self.k_proj = keras.layers.Dense(embed_dim, use_bias=bias, name="k_proj")  # K 矩阵的投影层
        self.q_proj = keras.layers.Dense(embed_dim, use_bias=bias, name="q_proj")  # Q 矩阵的投影层
        self.v_proj = keras.layers.Dense(embed_dim, use_bias=bias, name="v_proj")  # V 矩阵的投影层
        self.out_proj = keras.layers.Dense(embed_dim, use_bias=bias, name="out_proj")  # 输出投影层

    def _shape(self, tensor: tf.Tensor, seq_len: int, bsz: int):
        return tf.transpose(tf.reshape(tensor, (bsz, seq_len, self.num_heads, self.head_dim)), (0, 2, 1, 3))
        # 重新塑造张量形状以匹配多头注意力机制的需求，返回转置后的张量

    def call(
        self,
        hidden_states: tf.Tensor,
        key_value_states: tf.Tensor | None = None,
        past_key_value: Tuple[Tuple[tf.Tensor]] | None = None,
        attention_mask: tf.Tensor | None = None,
        layer_head_mask: tf.Tensor | None = None,
        training: Optional[bool] = False,
    ):
        # 执行注意力层的前向传播
        ...

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "k_proj", None) is not None:
            with tf.name_scope(self.k_proj.name):
                self.k_proj.build([None, None, self.embed_dim])  # 构建 K 矩阵的投影层
        if getattr(self, "q_proj", None) is not None:
            with tf.name_scope(self.q_proj.name):
                self.q_proj.build([None, None, self.embed_dim])  # 构建 Q 矩阵的投影层
        if getattr(self, "v_proj", None) is not None:
            with tf.name_scope(self.v_proj.name):
                self.v_proj.build([None, None, self.embed_dim])  # 构建 V 矩阵的投影层
        if getattr(self, "out_proj", None) is not None:
            with tf.name_scope(self.out_proj.name):
                self.out_proj.build([None, None, self.embed_dim])  # 构建输出投影层
    # 初始化函数，用于创建一个新的 PegasusEncoderLayer 对象
    def __init__(self, config: PegasusConfig, **kwargs):
        # 调用父类的初始化方法
        super().__init__(**kwargs)
        # 设置嵌入维度为配置中的模型维度
        self.embed_dim = config.d_model
        # 创建自注意力层对象，指定注意力头数和注意力机制的丢弃率
        self.self_attn = TFPegasusAttention(
            self.embed_dim, config.encoder_attention_heads, dropout=config.attention_dropout, name="self_attn"
        )
        # 创建自注意力层后的层归一化对象
        self.self_attn_layer_norm = keras.layers.LayerNormalization(epsilon=1e-5, name="self_attn_layer_norm")
        # 创建丢弃层对象，使用指定的丢弃率
        self.dropout = keras.layers.Dropout(config.dropout)
        # 获取激活函数对象，根据配置中的激活函数类型
        self.activation_fn = get_tf_activation(config.activation_function)
        # 创建激活层的丢弃层对象，使用指定的丢弃率
        self.activation_dropout = keras.layers.Dropout(config.activation_dropout)
        # 创建全连接层1，指定输出维度为配置中的编码器前馈网络维度
        self.fc1 = keras.layers.Dense(config.encoder_ffn_dim, name="fc1")
        # 创建全连接层2，输出维度为之前设置的嵌入维度
        self.fc2 = keras.layers.Dense(self.embed_dim, name="fc2")
        # 创建最终层的层归一化对象
        self.final_layer_norm = keras.layers.LayerNormalization(epsilon=1e-5, name="final_layer_norm")
        # 保存配置对象，以便在需要时进行访问
        self.config = config

    # 前向传播函数，执行编码器层的前向计算
    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: tf.Tensor,
        layer_head_mask: tf.Tensor,
        training: Optional[bool] = False,
    ):
        """
        Args:
            hidden_states (`tf.Tensor`): 输入层的张量，形状为 *(batch, seq_len, embed_dim)*
            attention_mask (`tf.Tensor`): 注意力掩码张量，形状为 *(batch, 1, tgt_len, src_len)*，
                其中填充元素由极大的负值指示。
            layer_head_mask (`tf.Tensor`): 给定层中注意力头的掩码张量，形状为 *(encoder_attention_heads,)*
            training (`bool`, optional): 指示是否处于训练模式的布尔值，默认为 False。
        """
        # 将输入状态保存为残差连接的一部分
        residual = hidden_states
        # 执行自注意力层的层归一化
        hidden_states = self.self_attn_layer_norm(hidden_states)
        # 执行自注意力计算，并获取注意力权重
        hidden_states, self_attn_weights, _ = self.self_attn(
            hidden_states=hidden_states, attention_mask=attention_mask, layer_head_mask=layer_head_mask
        )
        
        # 断言确保自注意力未修改查询的形状
        tf.debugging.assert_equal(
            shape_list(hidden_states),
            shape_list(residual),
            message=f"Self attn modified the shape of query {shape_list(residual)} to {shape_list(hidden_states)}",
        )

        # 使用丢弃层进行输出的丢弃处理
        hidden_states = self.dropout(hidden_states, training=training)
        # 执行残差连接
        hidden_states = residual + hidden_states

        # 将输入状态保存为残差连接的一部分
        residual = hidden_states
        # 执行最终层的层归一化
        hidden_states = self.final_layer_norm(hidden_states)
        # 应用激活函数并执行第一个全连接层的计算
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        # 使用激活层的丢弃层进行输出的丢弃处理
        hidden_states = self.activation_dropout(hidden_states, training=training)
        # 执行第二个全连接层的计算
        hidden_states = self.fc2(hidden_states)
        # 使用丢弃层进行输出的丢弃处理
        hidden_states = self.dropout(hidden_states, training=training)
        # 执行残差连接
        hidden_states = residual + hidden_states

        # 返回编码器层的输出状态和自注意力权重
        return hidden_states, self_attn_weights
    # 构建模型结构，如果已经构建过，则直接返回
    def build(self, input_shape=None):
        if self.built:
            return
        
        # 标记模型为已构建状态
        self.built = True
        
        # 如果存在 self_attn 属性，则构建 self attention 层
        if getattr(self, "self_attn", None) is not None:
            with tf.name_scope(self.self_attn.name):
                self.self_attn.build(None)
        
        # 如果存在 self_attn_layer_norm 属性，则构建 layer normalization 层
        if getattr(self, "self_attn_layer_norm", None) is not None:
            with tf.name_scope(self.self_attn_layer_norm.name):
                self.self_attn_layer_norm.build([None, None, self.embed_dim])
        
        # 如果存在 fc1 属性，则构建第一个全连接层
        if getattr(self, "fc1", None) is not None:
            with tf.name_scope(self.fc1.name):
                self.fc1.build([None, None, self.embed_dim])
        
        # 如果存在 fc2 属性，则构建第二个全连接层
        if getattr(self, "fc2", None) is not None:
            with tf.name_scope(self.fc2.name):
                self.fc2.build([None, None, self.config.encoder_ffn_dim])
        
        # 如果存在 final_layer_norm 属性，则构建最终的 layer normalization 层
        if getattr(self, "final_layer_norm", None) is not None:
            with tf.name_scope(self.final_layer_norm.name):
                self.final_layer_norm.build([None, None, self.embed_dim])
# 从transformers.models.mbart.modeling_tf_mbart.TFMBartDecoderLayer复制到TFPegasusDecoderLayer，用MBart->Pegasus进行替换
class TFPegasusDecoderLayer(keras.layers.Layer):
    # 初始化方法，接受PegasusConfig类型的config对象和其他关键字参数
    def __init__(self, config: PegasusConfig, **kwargs):
        # 调用父类的初始化方法
        super().__init__(**kwargs)
        # 设定嵌入维度为config.d_model
        self.embed_dim = config.d_model
        # self注意力层，使用TFPegasusAttention，设定嵌入维度、注意力头数、dropout率，用于解码器自注意力
        self.self_attn = TFPegasusAttention(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
            name="self_attn",
            is_decoder=True,
        )
        # dropout层，使用config.dropout作为dropout率
        self.dropout = keras.layers.Dropout(config.dropout)
        # 激活函数，根据配置获取相应的TensorFlow激活函数
        self.activation_fn = get_tf_activation(config.activation_function)
        # 激活函数的dropout层，使用config.activation_dropout作为dropout率
        self.activation_dropout = keras.layers.Dropout(config.activation_dropout)

        # self注意力层归一化，使用LayerNormalization，epsilon设定为1e-5
        self.self_attn_layer_norm = keras.layers.LayerNormalization(epsilon=1e-5, name="self_attn_layer_norm")
        # encoder注意力层，使用TFPegasusAttention，设定嵌入维度、注意力头数、dropout率，用于编码器-解码器注意力
        self.encoder_attn = TFPegasusAttention(
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            name="encoder_attn",
            is_decoder=True,
        )
        # encoder注意力层归一化，使用LayerNormalization，epsilon设定为1e-5
        self.encoder_attn_layer_norm = keras.layers.LayerNormalization(epsilon=1e-5, name="encoder_attn_layer_norm")
        # 全连接层1，使用Dense层，输出维度为config.decoder_ffn_dim
        self.fc1 = keras.layers.Dense(config.decoder_ffn_dim, name="fc1")
        # 全连接层2，使用Dense层，输出维度为self.embed_dim
        self.fc2 = keras.layers.Dense(self.embed_dim, name="fc2")
        # 最终归一化层，使用LayerNormalization，epsilon设定为1e-5
        self.final_layer_norm = keras.layers.LayerNormalization(epsilon=1e-5, name="final_layer_norm")
        # 存储配置对象
        self.config = config

    # call方法，实现层的调用逻辑，接受多个输入张量和可选的训练标志
    def call(
        self,
        hidden_states: tf.Tensor,  # 隐藏状态张量，输入形状为(batch_size, seq_len, embed_dim)
        attention_mask: tf.Tensor | None = None,  # 注意力掩码张量，用于屏蔽无效位置
        encoder_hidden_states: tf.Tensor | None = None,  # 编码器隐藏状态张量，形状为(batch_size, enc_seq_len, embed_dim)
        encoder_attention_mask: tf.Tensor | None = None,  # 编码器注意力掩码张量，用于编码器-解码器注意力
        layer_head_mask: tf.Tensor | None = None,  # 层级头掩码张量，用于多头注意力机制
        cross_attn_layer_head_mask: tf.Tensor | None = None,  # 交叉注意力层级头掩码张量，用于编码器-解码器注意力的多头机制
        past_key_value: Tuple[tf.Tensor] | None = None,  # 过去的键值元组，用于实现增量解码
        training: Optional[bool] = False,  # 训练标志，控制是否启用训练模式
    # 根据输入形状构建模型，如果已经构建过则直接返回
    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 如果存在 self_attn 属性，则构建 self attention 层
        if getattr(self, "self_attn", None) is not None:
            with tf.name_scope(self.self_attn.name):
                self.self_attn.build(None)
        # 如果存在 self_attn_layer_norm 属性，则构建 self attention 层的 Layer Normalization
        if getattr(self, "self_attn_layer_norm", None) is not None:
            with tf.name_scope(self.self_attn_layer_norm.name):
                self.self_attn_layer_norm.build([None, None, self.embed_dim])
        # 如果存在 encoder_attn 属性，则构建 encoder attention 层
        if getattr(self, "encoder_attn", None) is not None:
            with tf.name_scope(self.encoder_attn.name):
                self.encoder_attn.build(None)
        # 如果存在 encoder_attn_layer_norm 属性，则构建 encoder attention 层的 Layer Normalization
        if getattr(self, "encoder_attn_layer_norm", None) is not None:
            with tf.name_scope(self.encoder_attn_layer_norm.name):
                self.encoder_attn_layer_norm.build([None, None, self.embed_dim])
        # 如果存在 fc1 属性，则构建第一个全连接层
        if getattr(self, "fc1", None) is not None:
            with tf.name_scope(self.fc1.name):
                self.fc1.build([None, None, self.embed_dim])
        # 如果存在 fc2 属性，则构建第二个全连接层
        if getattr(self, "fc2", None) is not None:
            with tf.name_scope(self.fc2.name):
                self.fc2.build([None, None, self.config.decoder_ffn_dim])
        # 如果存在 final_layer_norm 属性，则构建最终的 Layer Normalization 层
        if getattr(self, "final_layer_norm", None) is not None:
            with tf.name_scope(self.final_layer_norm.name):
                self.final_layer_norm.build([None, None, self.embed_dim])
class TFPegasusPreTrainedModel(TFPreTrainedModel):
    # 设置模型配置类为 PegasusConfig
    config_class = PegasusConfig
    # 模型参数名前缀为 "model"
    base_model_prefix = "model"
    ...     "PG&E stated it scheduled the blackouts in response to forecasts for high winds "
    ...     "amid dry conditions. The aim is to reduce the risk of wildfires. Nearly 800 thousand customers were "
    ...     "scheduled to be affected by the shutoffs which were expected to last through at least midday tomorrow."
    ... )


# 定义一段新闻文章内容，描述了 PG&E 因高风险天气和干燥条件而安排的停电计划，目的是减少火灾风险，影响约 80 万客户，预计持续到明天中午。
ARTICLE_TO_SUMMARIZE = (
    "PG&E stated it scheduled the blackouts in response to forecasts for high winds "
    "amid dry conditions. The aim is to reduce the risk of wildfires. Nearly 800 thousand customers were "
    "scheduled to be affected by the shutoffs which were expected to last through at least midday tomorrow."
)

# 使用分词器对文章进行预处理，设置最大长度为 1024，并返回 TensorFlow 格式的张量
inputs = tokenizer(ARTICLE_TO_SUMMARIZE, max_length=1024, return_tensors="tf")

# 生成摘要
summary_ids = model.generate(input_ids)  # 使用模型生成摘要的输入 ID
print(tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False))
"""

PEGASUS_INPUTS_DOCSTRING = r"""
"""

# 使用 keras_serializable 装饰器将类标记为可序列化
@keras_serializable
class TFPegasusEncoder(keras.layers.Layer):
    # 使用 PegasusConfig 类作为配置类
    config_class = PegasusConfig

    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    [`TFPegasusEncoderLayer`].

    Args:
        config: PegasusConfig
    """

    def __init__(self, config: PegasusConfig, embed_tokens: Optional[keras.layers.Embedding] = None, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.dropout = keras.layers.Dropout(config.dropout)  # 使用指定的 dropout 率创建 Dropout 层
        self.layerdrop = config.encoder_layerdrop  # 从配置中获取层 dropout 率
        self.padding_idx = config.pad_token_id  # 获取配置中的填充索引
        self.max_source_positions = config.max_position_embeddings  # 获取配置中的最大位置嵌入
        self.embed_scale = tf.math.sqrt(float(config.d_model)) if config.scale_embedding else 1.0  # 计算嵌入比例因子

        self.embed_tokens = embed_tokens  # 设置嵌入 tokens
        self.embed_positions = TFPegasusSinusoidalPositionalEmbedding(
            config.max_position_embeddings,
            config.d_model,
            name="embed_positions",
        )  # 使用 sinusoidal 位置嵌入创建位置嵌入层

        self.layers = [TFPegasusEncoderLayer(config, name=f"layers.{i}") for i in range(config.encoder_layers)]  # 创建多层编码器层
        self.layer_norm = keras.layers.LayerNormalization(epsilon=1e-5, name="layer_norm")  # 创建层归一化层

    def get_embed_tokens(self):
        return self.embed_tokens  # 返回嵌入 tokens

    def set_embed_tokens(self, embed_tokens):
        self.embed_tokens = embed_tokens  # 设置嵌入 tokens

    # 使用 unpack_inputs 装饰器来展开输入参数
    @unpack_inputs
    def call(
        self,
        input_ids: tf.Tensor | None = None,
        inputs_embeds: tf.Tensor | None = None,
        attention_mask: tf.Tensor | None = None,
        head_mask: tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: Optional[bool] = False,
    ):
        # 函数实现在 Transformer 编码器层的调用过程中使用

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "embed_positions", None) is not None:
            with tf.name_scope(self.embed_positions.name):
                self.embed_positions.build(None)  # 构建位置嵌入层
        if getattr(self, "layer_norm", None) is not None:
            with tf.name_scope(self.layer_norm.name):
                self.layer_norm.build([None, None, self.config.d_model])  # 构建层归一化层
        if getattr(self, "layers", None) is not None:
            for layer in self.layers:
                with tf.name_scope(layer.name):
                    layer.build(None)  # 构建每一层编码器层


@keras_serializable
class TFPegasusDecoder(keras.layers.Layer):
    config_class = PegasusConfig

    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a [`TFPegasusDecoderLayer`]

    Args:
        config: PegasusConfig
        embed_tokens: output embedding
    """
    # 初始化方法，接收配置参数 config，嵌入词标记 embed_tokens 和其他关键字参数
    def __init__(self, config: PegasusConfig, embed_tokens: Optional[keras.layers.Embedding] = None, **kwargs):
        # 调用父类的初始化方法
        super().__init__(**kwargs)
        # 将配置参数 config 保存到实例变量中
        self.config = config
        # 设置填充索引为配置中的 pad_token_id
        self.padding_idx = config.pad_token_id
        # 将嵌入词标记 embed_tokens 保存到实例变量中
        self.embed_tokens = embed_tokens
        # 设置层丢弃率为配置中的 decoder_layerdrop
        self.layerdrop = config.decoder_layerdrop
        # 使用 TF 的 PegasusSinusoidalPositionalEmbedding 创建位置嵌入
        self.embed_positions = TFPegasusSinusoidalPositionalEmbedding(
            config.max_position_embeddings,
            config.d_model,
            name="embed_positions",
        )
        # 如果配置中指定了缩放嵌入，则使用 sqrt(d_model) 缩放因子；否则为 1.0
        self.embed_scale = tf.math.sqrt(float(config.d_model)) if config.scale_embedding else 1.0
        # 创建多层解码器层列表，每层使用配置中的参数和命名
        self.layers = [TFPegasusDecoderLayer(config, name=f"layers.{i}") for i in range(config.decoder_layers)]
        # 创建层归一化层，设置 epsilon 为 1e-5
        self.layer_norm = keras.layers.LayerNormalization(epsilon=1e-5, name="layer_norm")

        # 创建 Dropout 层，设置丢弃率为配置中的 dropout
        self.dropout = keras.layers.Dropout(config.dropout)

    # 获取嵌入词标记 embed_tokens
    def get_embed_tokens(self):
        return self.embed_tokens

    # 设置嵌入词标记 embed_tokens
    def set_embed_tokens(self, embed_tokens):
        self.embed_tokens = embed_tokens

    # 使用装饰器 unpack_inputs 对输入参数进行解包处理
    def call(
        self,
        input_ids: tf.Tensor | None = None,
        inputs_embeds: tf.Tensor | None = None,
        attention_mask: tf.Tensor | None = None,
        position_ids: tf.Tensor | None = None,
        encoder_hidden_states: tf.Tensor | None = None,
        encoder_attention_mask: tf.Tensor | None = None,
        head_mask: tf.Tensor | None = None,
        cross_attn_head_mask: tf.Tensor | None = None,
        past_key_values: Tuple[Tuple[tf.Tensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: Optional[bool] = False,
    ):
        # 此方法定义了模型的前向传播逻辑，输入和输出的详细说明通常在文档中而不是注释中

    # 构建模型，根据输入形状建立相应的层和嵌入
    def build(self, input_shape=None):
        # 如果模型已经构建过，则直接返回
        if self.built:
            return
        # 标记模型已经构建
        self.built = True
        # 如果存在 embed_positions 属性，则建立位置嵌入
        if getattr(self, "embed_positions", None) is not None:
            with tf.name_scope(self.embed_positions.name):
                self.embed_positions.build(None)
        # 如果存在 layer_norm 属性，则建立层归一化层
        if getattr(self, "layer_norm", None) is not None:
            with tf.name_scope(self.layer_norm.name):
                # 建立层归一化层，输入形状为 [None, None, self.config.d_model]
                self.layer_norm.build([None, None, self.config.d_model])
        # 如果存在 layers 属性，则逐层建立解码器层
        if getattr(self, "layers", None) is not None:
            for layer in self.layers:
                with tf.name_scope(layer.name):
                    layer.build(None)
# 使用装饰器标记这个类是可以被 Keras 序列化的
@keras_serializable
class TFPegasusMainLayer(keras.layers.Layer):
    # 指定配置类
    config_class = PegasusConfig

    # 初始化方法，接受 PegasusConfig 对象作为参数，并调用父类的初始化方法
    def __init__(self, config: PegasusConfig, **kwargs):
        super().__init__(**kwargs)

        # 将传入的配置对象赋值给实例变量 self.config
        self.config = config

        # 创建一个共享的 Embedding 层，用于模型的输入
        self.shared = keras.layers.Embedding(
            input_dim=config.vocab_size,
            output_dim=config.d_model,
            embeddings_initializer=keras.initializers.TruncatedNormal(stddev=self.config.init_std),
            name="model.shared",
        )

        # 设置一个额外的属性，指定加载/存储权重时预期的命名范围
        self.shared.load_weight_prefix = "model.shared"

        # 创建 Pegasus 编码器和解码器对象
        self.encoder = TFPegasusEncoder(config, self.shared, name="encoder")
        self.decoder = TFPegasusDecoder(config, self.shared, name="decoder")

    # 获取输入 Embedding 层的方法
    def get_input_embeddings(self):
        return self.shared

    # 设置输入 Embedding 层的方法，同时更新编码器和解码器的 embed_tokens 属性
    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.embed_tokens = self.shared
        self.decoder.embed_tokens = self.shared

    # 使用装饰器标记的调用方法，接受多个输入参数和可选的控制参数
    @unpack_inputs
    def call(
        self,
        input_ids: tf.Tensor | None = None,
        attention_mask: tf.Tensor | None = None,
        decoder_input_ids: tf.Tensor | None = None,
        decoder_attention_mask: tf.Tensor | None = None,
        decoder_position_ids: tf.Tensor | None = None,
        head_mask: tf.Tensor | None = None,
        decoder_head_mask: tf.Tensor | None = None,
        cross_attn_head_mask: tf.Tensor | None = None,
        encoder_outputs: Optional[Union[Tuple, TFBaseModelOutput]] = None,
        past_key_values: Tuple[Tuple[tf.Tensor]] = None,
        inputs_embeds: tf.Tensor | None = None,
        decoder_inputs_embeds: tf.Tensor | None = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: Optional[bool] = False,
        **kwargs,
    ):
        # 在这个方法中执行模型的前向传播，处理输入和控制参数，返回相应的输出
        pass
        ):
        # 如果没有传入解码器的输入ID和嵌入向量，则不使用缓存
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            use_cache = False

        # 如果输出隐藏状态为None，则使用模型配置中的默认设置
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        # 如果没有传入编码器输出，则调用编码器进行前向传播
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                training=training,
            )
        # 如果用户传入了一个元组形式的encoder_outputs，在return_dict=True时，将其封装为TFBaseModelOutput
        elif return_dict and not isinstance(encoder_outputs, TFBaseModelOutput):
            encoder_outputs = TFBaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )
        # 如果用户传入了TFBaseModelOutput形式的encoder_outputs，在return_dict=False时，将其封装为元组形式
        elif not return_dict and not isinstance(encoder_outputs, tuple):
            encoder_outputs = encoder_outputs.to_tuple()

        # 调用解码器进行前向传播
        decoder_outputs = self.decoder(
            decoder_input_ids,
            attention_mask=decoder_attention_mask,
            position_ids=decoder_position_ids,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        # 如果return_dict为False，则返回解码器和编码器的输出作为元组形式
        if not return_dict:
            return decoder_outputs + encoder_outputs

        # 如果return_dict为True，则封装解码器和编码器的输出为TFSeq2SeqModelOutput
        return TFSeq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )
    def build(self, input_shape=None):
        # 如果已经构建过模型，则直接返回，避免重复构建
        if self.built:
            return
        # 设置标志位，表示模型已经构建
        self.built = True
        
        # 为了确保共享/绑定的权重位于模型基本命名空间中
        # 在 tf.name_scope 末尾添加 "/"（不是开头！）将其放置在根命名空间而不是当前命名空间
        with tf.name_scope(self.shared.load_weight_prefix + "/" + self.shared.name + "/"):
            # 构建共享的权重
            self.shared.build(None)
        
        # 如果存在编码器，则在其命名空间下构建
        if getattr(self, "encoder", None) is not None:
            with tf.name_scope(self.encoder.name):
                self.encoder.build(None)
        
        # 如果存在解码器，则在其命名空间下构建
        if getattr(self, "decoder", None) is not None:
            with tf.name_scope(self.decoder.name):
                self.decoder.build(None)
# 使用装饰器添加模型文档字符串，描述该类是一个不带特定头部的原始 PEGASUS 模型
@add_start_docstrings(
    "The bare PEGASUS Model outputting raw hidden-states without any specific head on top.",
    PEGASUS_START_DOCSTRING,
)
# 定义 TFPegasusModel 类，继承自 TFPegasusPreTrainedModel 类
class TFPegasusModel(TFPegasusPreTrainedModel):
    
    # 初始化方法，接受 PegasusConfig 类型的配置对象及其他输入参数
    def __init__(self, config: PegasusConfig, *inputs, **kwargs):
        # 调用父类的初始化方法，传入配置及其他输入参数
        super().__init__(config, *inputs, **kwargs)
        
        # 创建 TFPegasusMainLayer 对象作为模型的主要层，使用给定的配置对象及名称
        self.model = TFPegasusMainLayer(config, name="model")

    # 返回模型的编码器部分
    def get_encoder(self):
        return self.model.encoder

    # 返回模型的解码器部分
    def get_decoder(self):
        return self.model.decoder

    # 使用装饰器添加模型正向传播的文档字符串，描述输入参数及其作用
    @unpack_inputs
    @add_start_docstrings_to_model_forward(PEGASUS_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFSeq2SeqModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    # 定义 call 方法，接受多个输入参数，并返回 TFSeq2SeqModelOutput 类型的输出
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        decoder_input_ids: np.ndarray | tf.Tensor | None = None,
        decoder_attention_mask: np.ndarray | tf.Tensor | None = None,
        decoder_position_ids: np.ndarray | tf.Tensor | None = None,
        head_mask: np.ndarray | tf.Tensor | None = None,
        decoder_head_mask: np.ndarray | tf.Tensor | None = None,
        cross_attn_head_mask: np.ndarray | tf.Tensor | None = None,
        encoder_outputs: Optional[Union[Tuple, TFBaseModelOutput]] = None,
        past_key_values: Optional[Tuple[Tuple[Union[np.ndarray, tf.Tensor]]]] = None,
        inputs_embeds: np.ndarray | tf.Tensor | None = None,
        decoder_inputs_embeds: np.ndarray | tf.Tensor | None = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
        **kwargs,
    ) -> Union[TFSeq2SeqModelOutput, Tuple[tf.Tensor]]:
        # 调用模型的主要层的 __call__ 方法，传递所有输入参数
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            decoder_position_ids=decoder_position_ids,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            encoder_outputs=encoder_outputs,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        # 返回模型的输出
        return outputs

    # 从 transformers.models.bart.modeling_tf_bart.TFBartModel.serving_output 复制的方法
    # 定义一个方法用于处理模型输出，接受一个输出对象作为参数
    def serving_output(self, output):
        # 如果配置中使用缓存，则提取输出对象中的过去键值对（past_key_values）的第二个元素，否则为 None
        pkv = tf.tuple(output.past_key_values)[1] if self.config.use_cache else None
        # 如果配置中输出隐藏状态（output_hidden_states），则将输出对象中的解码器隐藏状态转换为 TensorFlow 张量，否则为 None
        dec_hs = tf.convert_to_tensor(output.decoder_hidden_states) if self.config.output_hidden_states else None
        # 如果配置中输出注意力权重（output_attentions），则将输出对象中的解码器注意力权重转换为 TensorFlow 张量，否则为 None
        dec_attns = tf.convert_to_tensor(output.decoder_attentions) if self.config.output_attentions else None
        # 如果配置中输出注意力权重（output_attentions），则将输出对象中的交叉注意力权重转换为 TensorFlow 张量，否则为 None
        cross_attns = tf.convert_to_tensor(output.cross_attentions) if self.config.output_attentions else None
        # 如果配置中输出隐藏状态（output_hidden_states），则将输出对象中的编码器隐藏状态转换为 TensorFlow 张量，否则为 None
        enc_hs = tf.convert_to_tensor(output.encoder_hidden_states) if self.config.output_hidden_states else None
        # 如果配置中输出注意力权重（output_attentions），则将输出对象中的编码器注意力权重转换为 TensorFlow 张量，否则为 None
        enc_attns = tf.convert_to_tensor(output.encoder_attentions) if self.config.output_attentions else None

        # 返回一个 TFSeq2SeqModelOutput 对象，包含了模型输出的相关信息
        return TFSeq2SeqModelOutput(
            last_hidden_state=output.last_hidden_state,
            past_key_values=pkv,
            decoder_hidden_states=dec_hs,
            decoder_attentions=dec_attns,
            cross_attentions=cross_attns,
            encoder_last_hidden_state=output.encoder_last_hidden_state,
            encoder_hidden_states=enc_hs,
            encoder_attentions=enc_attns,
        )

    # 定义一个方法用于构建模型
    def build(self, input_shape=None):
        # 如果已经构建过模型，则直接返回
        if self.built:
            return
        # 标记为已经构建
        self.built = True
        # 如果存在模型属性
        if getattr(self, "model", None) is not None:
            # 使用模型的名称作为命名空间，在该命名空间下构建模型，传入 None 作为输入形状
            with tf.name_scope(self.model.name):
                self.model.build(None)
# Copied from transformers.models.bart.modeling_tf_bart.BiasLayer
class BiasLayer(keras.layers.Layer):
    """
    Bias as a layer. It is used for serialization purposes: `keras.Model.save_weights` stores on a per-layer basis,
    so all weights have to be registered in a layer.
    """

    def __init__(self, shape, initializer, trainable, name, **kwargs):
        super().__init__(name=name, **kwargs)
        # 注：在序列化时，这个变量的名称不会被作用域化，即它不会以"outer_layer/inner_layer/.../name:0"的格式出现。
        # 而是直接是"name:0"。详情请参考：
        # https://github.com/huggingface/transformers/pull/18833#issuecomment-1233090214
        # 添加一个可训练的偏置权重，用于模型层的操作
        self.bias = self.add_weight(name=name, shape=shape, initializer=initializer, trainable=trainable)

    def call(self, x):
        # 将偏置加到输入张量上，并返回结果
        return x + self.bias


@add_start_docstrings(
    "The PEGASUS Model with a language modeling head. Can be used for summarization.",
    PEGASUS_START_DOCSTRING,
)
class TFPegasusForConditionalGeneration(TFPegasusPreTrainedModel, TFCausalLanguageModelingLoss):
    _keys_to_ignore_on_load_unexpected = [
        r"model.encoder.embed_tokens.weight",
        r"model.decoder.embed_tokens.weight",
    ]

    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        # 创建 PEGASUS 主模型层，并命名为"model"
        self.model = TFPegasusMainLayer(config, name="model")
        # 根据配置创建一个偏置层用于最终的对数概率偏置，这个层在 pytorch 中被注册为一个缓冲区，为保持一致性设置为不可训练
        self.bias_layer = BiasLayer(
            name="final_logits_bias", shape=[1, config.vocab_size], initializer="zeros", trainable=False
        )

    def get_decoder(self):
        # 获取模型的解码器
        return self.model.decoder

    def get_encoder(self):
        # 获取模型的编码器
        return self.model.encoder

    def get_output_embeddings(self):
        # 获取输出的嵌入层
        return self.get_input_embeddings()

    def set_output_embeddings(self, value):
        # 设置输出的嵌入层
        self.set_input_embeddings(value)

    def get_bias(self):
        # 返回模型的偏置信息
        return {"final_logits_bias": self.bias_layer.bias}

    def set_bias(self, value):
        # 替换现有的包含偏置的层以进行正确的（反）序列化
        vocab_size = value["final_logits_bias"].shape[-1]
        self.bias_layer = BiasLayer(
            name="final_logits_bias", shape=[1, vocab_size], initializer="zeros", trainable=False
        )
        # 分配给偏置层新的偏置值
        self.bias_layer.bias.assign(value["final_logits_bias"])

    @unpack_inputs
    @add_start_docstrings_to_model_forward(PEGASUS_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFSeq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    @add_end_docstrings(PEGASUS_GENERATION_EXAMPLE)
    # 定义一个方法 `call`，用于模型的前向传播和推理过程
    def call(
        # 输入序列的 token IDs，可以是 TensorFlow 的输入类型或者 None
        input_ids: TFModelInputType | None = None,
        # 注意力掩码，可以是 NumPy 数组、TensorFlow 张量或者 None
        attention_mask: np.ndarray | tf.Tensor | None = None,
        # 解码器的输入 token IDs，可以是 NumPy 数组、TensorFlow 张量或者 None
        decoder_input_ids: np.ndarray | tf.Tensor | None = None,
        # 解码器的注意力掩码，可以是 NumPy 数组、TensorFlow 张量或者 None
        decoder_attention_mask: np.ndarray | tf.Tensor | None = None,
        # 解码器的位置 IDs，可以是 NumPy 数组、TensorFlow 张量或者 None
        decoder_position_ids: np.ndarray | tf.Tensor | None = None,
        # 头掩码用于屏蔽不同注意力头的特定位置，可以是 NumPy 数组、TensorFlow 张量或者 None
        head_mask: np.ndarray | tf.Tensor | None = None,
        # 解码器头部掩码，可以是 NumPy 数组、TensorFlow 张量或者 None
        decoder_head_mask: np.ndarray | tf.Tensor | None = None,
        # 跨注意力头部掩码，可以是 NumPy 数组、TensorFlow 张量或者 None
        cross_attn_head_mask: np.ndarray | tf.Tensor | None = None,
        # 编码器的输出，类型为 TFBaseModelOutput 或者 None
        encoder_outputs: Optional[TFBaseModelOutput] = None,
        # 用于存储过去的键值对的元组，元素为 NumPy 数组或 TensorFlow 张量的元组的元组
        past_key_values: Optional[Tuple[Tuple[Union[np.ndarray, tf.Tensor]]]] = None,
        # 输入的嵌入向量，可以是 NumPy 数组、TensorFlow 张量或者 None
        inputs_embeds: np.ndarray | tf.Tensor | None = None,
        # 解码器的输入嵌入向量，可以是 NumPy 数组、TensorFlow 张量或者 None
        decoder_inputs_embeds: np.ndarray | tf.Tensor | None = None,
        # 是否使用缓存，布尔值或者 None
        use_cache: Optional[bool] = None,
        # 是否输出注意力权重，布尔值或者 None
        output_attentions: Optional[bool] = None,
        # 是否输出隐藏状态，布尔值或者 None
        output_hidden_states: Optional[bool] = None,
        # 是否返回字典格式的输出，布尔值或者 None
        return_dict: Optional[bool] = None,
        # 标签数据，可以是 NumPy 数组、TensorFlow 张量或者 None
        labels: np.ndarray | tf.Tensor | None = None,
        # 是否处于训练模式，布尔值，默认为 False
        training: bool = False,
    ) -> Union[TFSeq2SeqLMOutput, Tuple[tf.Tensor]]:
        """
        labels (`tf.tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:
            Either TFSeq2SeqLMOutput or a tuple containing tf.Tensor outputs.

        """

        if labels is not None:
            # Convert padding tokens in labels to -100 to ignore them during loss computation
            labels = tf.where(
                labels == self.config.pad_token_id,
                tf.cast(tf.fill(shape_list(labels), -100), labels.dtype),
                labels,
            )
            use_cache = False
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                # Shift labels to the right for decoder input
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        # Forward pass through the model
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            decoder_position_ids=decoder_position_ids,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )
        
        # Calculate language modeling logits
        lm_logits = tf.matmul(outputs[0], self.model.shared.weights, transpose_b=True)
        lm_logits = self.bias_layer(lm_logits)
        # Compute masked language modeling loss if labels are provided
        masked_lm_loss = None if labels is None else self.hf_compute_loss(labels, lm_logits)

        if not return_dict:
            # Prepare output tuple without returning a dictionary
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output
        # Return TFSeq2SeqLMOutput object if return_dict is True
        return TFSeq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,  # index 1 of d outputs
            decoder_hidden_states=outputs.decoder_hidden_states,  # index 2 of d outputs
            decoder_attentions=outputs.decoder_attentions,  # index 3 of d outputs
            cross_attentions=outputs.cross_attentions,  # index 4 of d outputs
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,  # index 0 of encoder outputs
            encoder_hidden_states=outputs.encoder_hidden_states,  # 1 of e out
            encoder_attentions=outputs.encoder_attentions,  # 2 of e out
        )

    # Copied from transformers.models.bart.modeling_tf_bart.TFBartForConditionalGeneration.serving_output
    # 定义一个方法用于处理模型的输出，生成适合序列到序列模型的输出对象
    def serving_output(self, output):
        # 如果配置中启用了缓存，则从输出中提取过去键值对的第二个元素
        pkv = tf.tuple(output.past_key_values)[1] if self.config.use_cache else None
        # 如果配置中启用了输出隐藏状态，则将输出的解码器隐藏状态转换为张量
        dec_hs = tf.convert_to_tensor(output.decoder_hidden_states) if self.config.output_hidden_states else None
        # 如果配置中启用了输出注意力权重，则将输出的解码器注意力权重转换为张量
        dec_attns = tf.convert_to_tensor(output.decoder_attentions) if self.config.output_attentions else None
        # 如果配置中启用了输出注意力权重，则将输出的交叉注意力权重转换为张量
        cross_attns = tf.convert_to_tensor(output.cross_attentions) if self.config.output_attentions else None
        # 如果配置中启用了输出隐藏状态，则将输出的编码器隐藏状态转换为张量
        enc_hs = tf.convert_to_tensor(output.encoder_hidden_states) if self.config.output_hidden_states else None
        # 如果配置中启用了输出注意力权重，则将输出的编码器注意力权重转换为张量
        enc_attns = tf.convert_to_tensor(output.encoder_attentions) if self.config.output_attentions else None

        # 返回一个 TFSeq2SeqLMOutput 对象，包含输出的各种信息
        return TFSeq2SeqLMOutput(
            logits=output.logits,
            past_key_values=pkv,
            decoder_hidden_states=dec_hs,
            decoder_attentions=dec_attns,
            cross_attentions=cross_attns,
            encoder_last_hidden_state=output.encoder_last_hidden_state,
            encoder_hidden_states=enc_hs,
            encoder_attentions=enc_attns,
        )

    # 从 transformers 库中 TF 版本的 BART 模型类中复制的方法，用于为生成准备输入
    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past_key_values=None,
        attention_mask=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs,
    ):
        # 如果使用了过去的键值对，截断 decoder_input_ids
        if past_key_values is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        # 如果存在 decoder_attention_mask，则在最后一个位置累加以获取 decoder_position_ids
        if decoder_attention_mask is not None:  # xla
            decoder_position_ids = tf.math.cumsum(decoder_attention_mask, axis=-1, exclusive=True)[:, -1:]
        # 如果没有使用 XLA 且存在过去的键值对，则从 past_key_values 中获取 decoder_position_ids
        elif past_key_values is not None:  # no xla + past_key_values
            decoder_position_ids = past_key_values[0][0].shape[2]
        # 否则，生成一个范围为 decoder_input_ids.shape[1] 的 decoder_position_ids
        else:  # no xla + no past_key_values
            decoder_position_ids = tf.range(decoder_input_ids.shape[1])

        # 返回一个包含各种生成所需输入的字典
        return {
            "input_ids": None,  # encoder_outputs 已定义，不需要 input_ids
            "encoder_outputs": encoder_outputs,
            "past_key_values": past_key_values,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "decoder_attention_mask": decoder_attention_mask,
            "decoder_position_ids": decoder_position_ids,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,  # 修改此处以避免缓存（可能用于调试）
        }

    # 定义一个方法用于根据标签从配置中获取的开始和填充令牌 ID 调整 decoder_input_ids
    def prepare_decoder_input_ids_from_labels(self, labels: tf.Tensor):
        return shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)
    # 定义模型建立函数，指定输入形状（可选）
    def build(self, input_shape=None):
        # 如果模型已经建立过，则直接返回
        if self.built:
            return
        # 将标志位设置为已建立
        self.built = True
        
        # 如果存在模型属性，则按模型名称创建命名空间，并建立模型
        if getattr(self, "model", None) is not None:
            with tf.name_scope(self.model.name):
                self.model.build(None)
        
        # 如果存在偏置层属性，则按偏置层名称创建命名空间，并建立偏置层
        if getattr(self, "bias_layer", None) is not None:
            with tf.name_scope(self.bias_layer.name):
                self.bias_layer.build(None)
```