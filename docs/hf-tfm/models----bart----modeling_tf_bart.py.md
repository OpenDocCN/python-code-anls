# `.\models\bart\modeling_tf_bart.py`

```py
    # 创建一个 mask tensor，用于标记输入的自回归性质
    """

    # "Verify that `labels` has only positive values and -100"
    assert_gte0 = tf.debugging.assert_greater_equal(shifted_input_ids, tf.constant(0, dtype=input_ids.dtype))

    # Make sure the assertion op is called by wrapping the result in an identity no-op
    with tf.control_dependencies([assert_gte0]):
        # 确保断言操作被调用，并返回与输入相同的 shifted_input_ids 张量
        shifted_input_ids = tf.identity(shifted_input_ids)

    return shifted_input_ids
    # 创建用于双向自注意力的因果掩码。
    bsz = input_ids_shape[0]  # 获取批次大小
    tgt_len = input_ids_shape[1]  # 获取目标长度
    mask = tf.ones((tgt_len, tgt_len)) * LARGE_NEGATIVE  # 创建一个初始掩码矩阵，用负无穷大填充
    
    mask_cond = tf.range(shape_list(mask)[-1])  # 创建一个与掩码矩阵最后一个维度大小相等的序列
    
    # 将掩码矩阵的下三角部分置零，实现因果性，确保每个位置只能依赖于它之前的位置
    mask = tf.where(mask_cond < tf.reshape(mask_cond + 1, (shape_list(mask)[-1], 1)), 0.0, mask)
    
    if past_key_values_length > 0:
        # 如果存在过去的键值对长度，则在掩码矩阵左侧填充零，以匹配过去键值对的长度
        mask = tf.concat([tf.zeros((tgt_len, past_key_values_length)), mask], axis=-1)
    
    # 使用 tf.tile 扩展掩码矩阵的维度以匹配输入的批次大小，并返回结果
    return tf.tile(mask[None, None, :, :], (bsz, 1, 1, 1))
    def _expand_mask(mask: tf.Tensor, tgt_len: Optional[int] = None):
        """
        Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
        """
        # 获取注意力掩码的序列长度
        src_len = shape_list(mask)[1]
        # 如果未指定目标长度，则使用源长度作为目标长度
        tgt_len = tgt_len if tgt_len is not None else src_len
        # 创建常数张量，值为1.0
        one_cst = tf.constant(1.0)
        # 将注意力掩码转换为与 one_cst 相同数据类型的张量
        mask = tf.cast(mask, dtype=one_cst.dtype)
        # 在第二维和第三维上对注意力掩码进行复制扩展
        expanded_mask = tf.tile(mask[:, None, None, :], (1, 1, tgt_len, 1))

        # 返回扩展后的掩码，并乘以一个大负数，表示未关注的区域
        return (one_cst - expanded_mask) * LARGE_NEGATIVE


class TFBartLearnedPositionalEmbedding(keras.layers.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, **kwargs):
        # 如果 padding_idx 被指定，Bart 模型会偏移嵌入的 id 值并相应调整 num_embeddings
        # 这是一个针对 Bart 模型的特殊处理，其他模型不需要
        self.offset = 2
        super().__init__(num_embeddings + self.offset, embedding_dim, **kwargs)

    def call(
        self,
        input_shape: Optional[tf.TensorShape] = None,
        past_key_values_length: int = 0,
        position_ids: tf.Tensor | None = None,
    ):
        """Input is expected to be of size [bsz x seqlen]."""
        if position_ids is None:
            # 如果未提供位置 id，则根据输入形状中的序列长度创建位置 id
            seq_len = input_shape[1]
            position_ids = tf.range(seq_len, delta=1, name="range")
            position_ids += past_key_values_length

        # 确定位置 id 的数据类型，并将其与偏移量相加后传递给父类的调用方法
        offset_dtype = position_ids.dtype if isinstance(position_ids, tf.Tensor) else tf.int32
        return super().call(position_ids + tf.constant(self.offset, dtype=offset_dtype))


class TFBartAttention(keras.layers.Layer):
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
        # 初始化注意力层的参数
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = keras.layers.Dropout(dropout)
        self.head_dim = embed_dim // num_heads
        # 确保 embed_dim 能够被 num_heads 整除
        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        # 缩放因子，用于缩放注意力分数
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder

        # 初始化线性变换层
        self.k_proj = keras.layers.Dense(embed_dim, use_bias=bias, name="k_proj")
        self.q_proj = keras.layers.Dense(embed_dim, use_bias=bias, name="q_proj")
        self.v_proj = keras.layers.Dense(embed_dim, use_bias=bias, name="v_proj")
        self.out_proj = keras.layers.Dense(embed_dim, use_bias=bias, name="out_proj")

    def _shape(self, tensor: tf.Tensor, seq_len: int, bsz: int):
        # 重塑张量形状，以便进行多头注意力计算
        return tf.transpose(tf.reshape(tensor, (bsz, seq_len, self.num_heads, self.head_dim)), (0, 2, 1, 3))
    # 定义一个方法，用于调用自定义层对象
    def call(
        self,
        hidden_states: tf.Tensor,
        key_value_states: tf.Tensor | None = None,
        past_key_value: Tuple[Tuple[tf.Tensor]] | None = None,
        attention_mask: tf.Tensor | None = None,
        layer_head_mask: tf.Tensor | None = None,
        training: Optional[bool] = False,
    ):
        # 如果已经构建过，则直接返回，不再重复构建
        if self.built:
            return
        # 将构建标志设置为 True，表示已经构建
        self.built = True
        # 如果存在 self.k_proj 属性，则构建 k_proj 层
        if getattr(self, "k_proj", None) is not None:
            # 在名为 self.k_proj 的命名作用域下，构建 k_proj 层
            with tf.name_scope(self.k_proj.name):
                self.k_proj.build([None, None, self.embed_dim])
        # 如果存在 self.q_proj 属性，则构建 q_proj 层
        if getattr(self, "q_proj", None) is not None:
            # 在名为 self.q_proj 的命名作用域下，构建 q_proj 层
            with tf.name_scope(self.q_proj.name):
                self.q_proj.build([None, None, self.embed_dim])
        # 如果存在 self.v_proj 属性，则构建 v_proj 层
        if getattr(self, "v_proj", None) is not None:
            # 在名为 self.v_proj 的命名作用域下，构建 v_proj 层
            with tf.name_scope(self.v_proj.name):
                self.v_proj.build([None, None, self.embed_dim])
        # 如果存在 self.out_proj 属性，则构建 out_proj 层
        if getattr(self, "out_proj", None) is not None:
            # 在名为 self.out_proj 的命名作用域下，构建 out_proj 层
            with tf.name_scope(self.out_proj.name):
                self.out_proj.build([None, None, self.embed_dim])
class TFBartEncoderLayer(keras.layers.Layer):
    # TFBartEncoderLayer 类定义，继承自 keras.layers.Layer
    def __init__(self, config: BartConfig, **kwargs):
        # 初始化函数，接受一个 BartConfig 类型的配置对象和其他关键字参数
        super().__init__(**kwargs)
        # 调用父类初始化方法

        # 设置嵌入维度为配置对象中的 d_model 属性
        self.embed_dim = config.d_model

        # 创建自注意力层，使用 TFBartAttention 类，配置包括嵌入维度、注意力头数、注意力层的名字为 "self_attn"
        self.self_attn = TFBartAttention(
            self.embed_dim, config.encoder_attention_heads, dropout=config.attention_dropout, name="self_attn"
        )

        # 创建自注意力层的 LayerNormalization 层，设置 epsilon 为 1e-5，名字为 "self_attn_layer_norm"
        self.self_attn_layer_norm = keras.layers.LayerNormalization(epsilon=1e-5, name="self_attn_layer_norm")

        # 创建 dropout 层，使用配置对象中的 dropout 率
        self.dropout = keras.layers.Dropout(config.dropout)

        # 获取激活函数，根据配置对象中的激活函数名获取对应的 TensorFlow 激活函数
        self.activation_fn = get_tf_activation(config.activation_function)

        # 创建激活 dropout 层，使用配置对象中的激活 dropout 率
        self.activation_dropout = keras.layers.Dropout(config.activation_dropout)

        # 创建第一个全连接层，输出维度为配置对象中的 encoder_ffn_dim，名字为 "fc1"
        self.fc1 = keras.layers.Dense(config.encoder_ffn_dim, name="fc1")

        # 创建第二个全连接层，输出维度为嵌入维度，名字为 "fc2"
        self.fc2 = keras.layers.Dense(self.embed_dim, name="fc2")

        # 创建最终的 LayerNormalization 层，设置 epsilon 为 1e-5，名字为 "final_layer_norm"
        self.final_layer_norm = keras.layers.LayerNormalization(epsilon=1e-5, name="final_layer_norm")

        # 保存配置对象
        self.config = config

    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: np.ndarray | tf.Tensor | None,
        layer_head_mask: tf.Tensor | None,
        training: Optional[bool] = False,
    ) -> tf.Tensor:
        """
        Args:
            hidden_states (`tf.Tensor`): 输入到该层的张量，形状为 `(batch, seq_len, embed_dim)`
            attention_mask (`tf.Tensor`): 注意力掩码张量，形状为 `(batch, 1, tgt_len, src_len)`，用大负值表示填充元素
            layer_head_mask (`tf.Tensor`): 给定层中注意力头的掩码张量，形状为 `(encoder_attention_heads,)`
            training (`Optional[bool]`): 是否处于训练模式，默认为 False
        """
        # 保存输入的原始状态作为残差连接的一部分
        residual = hidden_states

        # 调用自注意力层进行操作，返回处理后的 hidden_states、注意力权重和额外信息
        hidden_states, self_attn_weights, _ = self.self_attn(
            hidden_states=hidden_states, attention_mask=attention_mask, layer_head_mask=layer_head_mask
        )

        # 断言保证自注意力层没有修改查询的形状
        tf.debugging.assert_equal(
            shape_list(hidden_states),
            shape_list(residual),
            message=f"Self attn modified the shape of query {shape_list(residual)} to {shape_list(hidden_states)}",
        )

        # 应用 dropout 操作
        hidden_states = self.dropout(hidden_states, training=training)

        # 添加残差连接
        hidden_states = residual + hidden_states

        # 应用 LayerNormalization 操作
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # 保存当前状态作为新的残差
        residual = hidden_states

        # 应用激活函数和第一个全连接层
        hidden_states = self.activation_fn(self.fc1(hidden_states))

        # 应用激活 dropout 操作
        hidden_states = self.activation_dropout(hidden_states, training=training)

        # 应用第二个全连接层
        hidden_states = self.fc2(hidden_states)

        # 应用 dropout 操作
        hidden_states = self.dropout(hidden_states, training=training)

        # 添加残差连接
        hidden_states = residual + hidden_states

        # 应用最终的 LayerNormalization 操作
        hidden_states = self.final_layer_norm(hidden_states)

        # 返回处理后的 hidden_states 和自注意力权重
        return hidden_states, self_attn_weights
    # 构建神经网络层，如果已经构建过则直接返回
    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 如果存在 self_attn 属性，则构建 self_attn 层
        if getattr(self, "self_attn", None) is not None:
            with tf.name_scope(self.self_attn.name):
                self.self_attn.build(None)
        # 如果存在 self_attn_layer_norm 属性，则构建 self_attn_layer_norm 层
        if getattr(self, "self_attn_layer_norm", None) is not None:
            with tf.name_scope(self.self_attn_layer_norm.name):
                self.self_attn_layer_norm.build([None, None, self.embed_dim])
        # 如果存在 fc1 属性，则构建 fc1 层
        if getattr(self, "fc1", None) is not None:
            with tf.name_scope(self.fc1.name):
                self.fc1.build([None, None, self.embed_dim])
        # 如果存在 fc2 属性，则构建 fc2 层
        if getattr(self, "fc2", None) is not None:
            with tf.name_scope(self.fc2.name):
                self.fc2.build([None, None, self.config.encoder_ffn_dim])
        # 如果存在 final_layer_norm 属性，则构建 final_layer_norm 层
        if getattr(self, "final_layer_norm", None) is not None:
            with tf.name_scope(self.final_layer_norm.name):
                self.final_layer_norm.build([None, None, self.embed_dim])
# 定义 TFBartDecoderLayer 类，继承自 keras.layers.Layer，用于实现 BART 解码器的一个层
class TFBartDecoderLayer(keras.layers.Layer):
    # 初始化方法，接受配置参数 config 和其他关键字参数
    def __init__(self, config: BartConfig, **kwargs):
        # 调用父类的初始化方法
        super().__init__(**kwargs)
        # 设置层的嵌入维度为配置中的模型维度
        self.embed_dim = config.d_model
        # 创建自注意力机制层 self_attn，使用 TFBartAttention 类，配置包括嵌入维度、注意力头数、注意力 dropout 等
        self.self_attn = TFBartAttention(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
            name="self_attn",
            is_decoder=True,
        )
        # 创建 Dropout 层，用于 self_attn 层的输出
        self.dropout = keras.layers.Dropout(config.dropout)
        # 获取激活函数，并将其赋值给 activation_fn
        self.activation_fn = get_tf_activation(config.activation_function)
        # 创建用于激活函数 dropout 的 Dropout 层
        self.activation_dropout = keras.layers.Dropout(config.activation_dropout)

        # 创建 LayerNormalization 层，用于自注意力机制的输出
        self.self_attn_layer_norm = keras.layers.LayerNormalization(epsilon=1e-5, name="self_attn_layer_norm")
        
        # 创建编码器注意力机制层 encoder_attn，配置与 self_attn 类似，用于处理编码器的输出
        self.encoder_attn = TFBartAttention(
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            name="encoder_attn",
            is_decoder=True,
        )
        # 创建 LayerNormalization 层，用于编码器注意力机制的输出
        self.encoder_attn_layer_norm = keras.layers.LayerNormalization(epsilon=1e-5, name="encoder_attn_layer_norm")
        
        # 创建全连接层 fc1，用于进行维度变换和非线性变换
        self.fc1 = keras.layers.Dense(config.decoder_ffn_dim, name="fc1")
        # 创建全连接层 fc2，输出维度与嵌入维度相同
        self.fc2 = keras.layers.Dense(self.embed_dim, name="fc2")
        
        # 创建 LayerNormalization 层，用于最终输出的规范化
        self.final_layer_norm = keras.layers.LayerNormalization(epsilon=1e-5, name="final_layer_norm")
        
        # 保存配置对象
        self.config = config

    # 定义 call 方法，实现层的前向传播逻辑
    def call(
        self,
        hidden_states: tf.Tensor,  # 输入的隐藏状态张量
        attention_mask: np.ndarray | tf.Tensor | None = None,  # 注意力遮罩，可选参数
        encoder_hidden_states: np.ndarray | tf.Tensor | None = None,  # 编码器的隐藏状态张量，可选参数
        encoder_attention_mask: np.ndarray | tf.Tensor | None = None,  # 编码器的注意力遮罩，可选参数
        layer_head_mask: tf.Tensor | None = None,  # 层级头部掩码，可选参数
        cross_attn_layer_head_mask: tf.Tensor | None = None,  # 交叉注意力层级头部掩码，可选参数
        past_key_value: Optional[Tuple[Tuple[Union[np.ndarray, tf.Tensor]]]] = None,  # 过去的键值对，可选参数
        training: Optional[bool] = False,  # 是否处于训练模式，可选参数，默认为 False

        # 方法主体部分暂时省略，根据具体逻辑进行完整注释
        pass
    # 定义模型的构建方法，如果已经构建过则直接返回
    def build(self, input_shape=None):
        if self.built:
            return
        # 标记模型已构建
        self.built = True
        
        # 如果存在 self_attn 属性，则构建 self attention 层
        if getattr(self, "self_attn", None) is not None:
            with tf.name_scope(self.self_attn.name):
                self.self_attn.build(None)
        
        # 如果存在 self_attn_layer_norm 属性，则构建 self attention 层的 layer normalization 层
        if getattr(self, "self_attn_layer_norm", None) is not None:
            with tf.name_scope(self.self_attn_layer_norm.name):
                self.self_attn_layer_norm.build([None, None, self.embed_dim])
        
        # 如果存在 encoder_attn 属性，则构建 encoder-decoder attention 层
        if getattr(self, "encoder_attn", None) is not None:
            with tf.name_scope(self.encoder_attn.name):
                self.encoder_attn.build(None)
        
        # 如果存在 encoder_attn_layer_norm 属性，则构建 encoder-decoder attention 层的 layer normalization 层
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
        
        # 如果存在 final_layer_norm 属性，则构建最终的 layer normalization 层
        if getattr(self, "final_layer_norm", None) is not None:
            with tf.name_scope(self.final_layer_norm.name):
                self.final_layer_norm.build([None, None, self.embed_dim])
class TFBartClassificationHead(keras.layers.Layer):
    """Head for sentence-level classification tasks."""

    def __init__(self, inner_dim: int, num_classes: int, pooler_dropout: float, name: str, **kwargs):
        super().__init__(name=name, **kwargs)
        # 定义一个全连接层，输出维度为 inner_dim
        self.dense = keras.layers.Dense(inner_dim, name="dense")
        # 定义一个 dropout 层，用于在训练过程中随机失活部分神经元
        self.dropout = keras.layers.Dropout(pooler_dropout)
        # 定义一个全连接层，输出维度为 num_classes，用于分类任务的输出
        self.out_proj = keras.layers.Dense(num_classes, name="out_proj")
        # 记录输入维度和内部维度，这些参数在构建模型时会用到
        self.input_dim = inner_dim
        self.inner_dim = inner_dim

    def call(self, inputs):
        # 对输入进行 dropout 处理
        hidden_states = self.dropout(inputs)
        # 经过全连接层 dense 处理
        hidden_states = self.dense(hidden_states)
        # 使用 tanh 激活函数处理隐藏状态
        hidden_states = keras.activations.tanh(hidden_states)
        # 再次进行 dropout 处理
        hidden_states = self.dropout(hidden_states)
        # 最后经过全连接层 out_proj 输出最终的分类结果
        hidden_states = self.out_proj(hidden_states)
        return hidden_states

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 构建模型，如果已经构建则直接返回
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                # 使用 input_dim 构建 dense 层
                self.dense.build([None, None, self.input_dim])
        if getattr(self, "out_proj", None) is not None:
            with tf.name_scope(self.out_proj.name):
                # 使用 inner_dim 构建 out_proj 层
                self.out_proj.build([None, None, self.inner_dim])


class TFBartPretrainedModel(TFPreTrainedModel):
    config_class = BartConfig
    base_model_prefix = "model"

    @property
    def dummy_inputs(self):
        dummy_inputs = super().dummy_inputs
        # 修改虚拟输入，使得 input_ids 和 decoder_input_ids 均扩展为原来的两倍长度
        dummy_inputs["input_ids"] = dummy_inputs["input_ids"] * 2
        if "decoder_input_ids" in dummy_inputs:
            dummy_inputs["decoder_input_ids"] = dummy_inputs["decoder_input_ids"] * 2
        return dummy_inputs

    def tf_to_pt_weight_rename(self, tf_weight):
        # 将 TF 的权重名称转换为 PyTorch 风格的权重名称
        if tf_weight == "model.shared.weight":
            return tf_weight, "model.decoder.embed_tokens.weight"
        else:
            return (tf_weight,)


BART_START_DOCSTRING = r"""
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
"""
    # 使用 `BartConfig` 类型的配置参数 `config`，该类包含了模型的所有参数设定
    config ([`BartConfig`]): Model configuration class with all the parameters of the model.
        # 通过配置文件初始化模型不会加载模型的权重，只加载配置信息
        Initializing with a config file does not load the weights associated with the model, only the
        configuration.
        # 使用 [`~TFPreTrainedModel.from_pretrained`] 方法可以加载模型的权重
        Check out the [`~TFPreTrainedModel.from_pretrained`] method to load the model weights.
"""
"""


BART_GENERATION_EXAMPLE = r"""
    Summarization example:

    ```
    >>> from transformers import AutoTokenizer, TFBartForConditionalGeneration

    >>> model = TFBartForConditionalGeneration.from_pretrained("facebook/bart-large")
    >>> tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")

    >>> ARTICLE_TO_SUMMARIZE = "My friends are cool but they eat too many carbs."
    >>> inputs = tokenizer([ARTICLE_TO_SUMMARIZE], max_length=1024, return_tensors="tf")

    >>> # Generate Summary
    >>> summary_ids = model.generate(inputs["input_ids"], num_beams=4, max_length=5)
    >>> print(tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False))
    ```

    Mask filling example:

    ```
    >>> from transformers import AutoTokenizer, TFBartForConditionalGeneration

    >>> tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")
    >>> TXT = "My friends are <mask> but they eat too many carbs."

    >>> model = TFBartForConditionalGeneration.from_pretrained("facebook/bart-large")
    >>> input_ids = tokenizer([TXT], return_tensors="tf")["input_ids"]
    >>> logits = model(input_ids).logits
    >>> probs = tf.nn.softmax(logits[0])
    >>> # probs[5] is associated with the mask token
    ```
"""


BART_INPUTS_DOCSTRING = r"""
"""


@keras_serializable
class TFBartEncoder(keras.layers.Layer):
    config_class = BartConfig
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    [`TFBartEncoderLayer`].

    Args:
        config: BartConfig
    """

    def __init__(self, config: BartConfig, embed_tokens: Optional[keras.layers.Embedding] = None, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.dropout = keras.layers.Dropout(config.dropout)
        self.layerdrop = config.encoder_layerdrop
        self.padding_idx = config.pad_token_id
        self.max_source_positions = config.max_position_embeddings
        self.embed_scale = tf.math.sqrt(float(config.d_model)) if config.scale_embedding else 1.0

        self.embed_tokens = embed_tokens
        self.embed_positions = TFBartLearnedPositionalEmbedding(
            config.max_position_embeddings,
            config.d_model,
            name="embed_positions",
        )
        self.layers = [TFBartEncoderLayer(config, name=f"layers.{i}") for i in range(config.encoder_layers)]
        self.layernorm_embedding = keras.layers.LayerNormalization(epsilon=1e-5, name="layernorm_embedding")
        self.embed_dim = config.d_model

    @unpack_inputs
    # 这是一个装饰器，用于解包输入参数，使其可以作为函数的参数使用
    # 定义类方法 `call`，用于模型的前向传播
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        inputs_embeds: np.ndarray | tf.Tensor | None = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        head_mask: np.ndarray | tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: Optional[bool] = False,
    ):
        # 如果模型已经构建完成，则直接返回，避免重复构建
        if self.built:
            return
        # 设置模型已经构建的标志
        self.built = True

        # 如果存在 `embed_positions` 属性，则构建它
        if getattr(self, "embed_positions", None) is not None:
            with tf.name_scope(self.embed_positions.name):
                self.embed_positions.build(None)

        # 如果存在 `layernorm_embedding` 属性，则构建它
        if getattr(self, "layernorm_embedding", None) is not None:
            with tf.name_scope(self.layernorm_embedding.name):
                self.layernorm_embedding.build([None, None, self.embed_dim])

        # 如果存在 `layers` 属性，则逐层构建每一层
        if getattr(self, "layers", None) is not None:
            for layer in self.layers:
                with tf.name_scope(layer.name):
                    layer.build(None)
@keras_serializable
class TFBartDecoder(keras.layers.Layer):
    # 定义一个可序列化的 Keras 层，用于实现 BART 解码器
    config_class = BartConfig
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a [`TFBartDecoderLayer`]

    Args:
        config: BartConfig
        embed_tokens: output embedding
    """

    def __init__(self, config: BartConfig, embed_tokens: Optional[keras.layers.Embedding] = None, **kwargs):
        super().__init__(**kwargs)
        # 初始化函数，设置配置、填充索引、嵌入 tokens 和其他参数
        self.config = config
        self.padding_idx = config.pad_token_id
        self.embed_tokens = embed_tokens
        self.layerdrop = config.decoder_layerdrop
        self.embed_positions = TFBartLearnedPositionalEmbedding(
            config.max_position_embeddings,
            config.d_model,
            name="embed_positions",
        )
        self.embed_scale = tf.math.sqrt(float(config.d_model)) if config.scale_embedding else 1.0
        self.layers = [TFBartDecoderLayer(config, name=f"layers.{i}") for i in range(config.decoder_layers)]
        self.layernorm_embedding = keras.layers.LayerNormalization(epsilon=1e-5, name="layernorm_embedding")

        self.dropout = keras.layers.Dropout(config.dropout)

    @unpack_inputs
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        inputs_embeds: np.ndarray | tf.Tensor | None = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        position_ids: np.ndarray | tf.Tensor | None = None,
        encoder_hidden_states: np.ndarray | tf.Tensor | None = None,
        encoder_attention_mask: np.ndarray | tf.Tensor | None = None,
        head_mask: np.ndarray | tf.Tensor | None = None,
        cross_attn_head_mask: np.ndarray | tf.Tensor | None = None,
        past_key_values: Optional[Tuple[Tuple[Union[np.ndarray, tf.Tensor]]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: Optional[bool] = False,
    ):
        # 解码器的调用方法，接受多种输入和参数
        ...

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 如果层已经构建，则直接返回

        if getattr(self, "embed_positions", None) is not None:
            with tf.name_scope(self.embed_positions.name):
                self.embed_positions.build(None)
        # 如果存在嵌入位置信息，构建嵌入位置层

        if getattr(self, "layernorm_embedding", None) is not None:
            with tf.name_scope(self.layernorm_embedding.name):
                self.layernorm_embedding.build([None, None, self.config.d_model])
        # 如果存在层归一化层，构建层归一化层

        if getattr(self, "layers", None) is not None:
            for layer in self.layers:
                with tf.name_scope(layer.name):
                    layer.build(None)
        # 构建解码器的每一层

@keras_serializable
class TFBartMainLayer(keras.layers.Layer):
    # 定义一个可序列化的 Keras 主层，用于实现 BART 主要层
    config_class = BartConfig
    def __init__(self, config: BartConfig, load_weight_prefix=None, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        # 创建一个共享的嵌入层，用于输入的词汇表大小和模型维度
        self.shared = keras.layers.Embedding(
            input_dim=config.vocab_size,
            output_dim=config.d_model,
            # 使用 TruncatedNormal 初始化器初始化嵌入层权重
            embeddings_initializer=keras.initializers.TruncatedNormal(stddev=self.config.init_std),
            name="model.shared",
        )
        # 设置加载/存储权重时的预期名称空间
        self.shared.load_weight_prefix = "model.shared" if load_weight_prefix is None else load_weight_prefix

        # 创建编码器和解码器对象
        self.encoder = TFBartEncoder(config, self.shared, name="encoder")
        self.decoder = TFBartDecoder(config, self.shared, name="decoder")

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        # 更新共享的嵌入层权重
        self.shared = new_embeddings
        # 更新编码器和解码器的嵌入层权重
        self.encoder.embed_tokens = self.shared
        self.decoder.embed_tokens = self.shared

    @unpack_inputs
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
        training: Optional[bool] = False,
        **kwargs,
    ):
        # 省略模型调用的详细注释，因为这些参数涉及模型的输入和输出处理

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 设置共享/共享权重的名称空间预期在模型基本名称空间中
        # 在 tf.name_scope 的末尾添加 "/"（而不是开头！）将其放置在根名称空间而不是当前名称空间中。
        with tf.name_scope(self.shared.load_weight_prefix + "/" + self.shared.name + "/"):
            self.shared.build(None)
        # 如果存在编码器对象，则在其名称空间下构建
        if getattr(self, "encoder", None) is not None:
            with tf.name_scope(self.encoder.name):
                self.encoder.build(None)
        # 如果存在解码器对象，则在其名称空间下构建
        if getattr(self, "decoder", None) is not None:
            with tf.name_scope(self.decoder.name):
                self.decoder.build(None)
# 添加 BART 模型的文档字符串，描述该类用于输出没有特定头部的原始隐藏状态
@add_start_docstrings(
    "The bare BART Model outputting raw hidden-states without any specific head on top.",
    BART_START_DOCSTRING,
)
# 定义 TFBartModel 类，继承自 TFBartPretrainedModel
class TFBartModel(TFBartPretrainedModel):
    # 表示需要加载权重前缀
    _requires_load_weight_prefix = True

    # 初始化方法，接受 BartConfig 类型的配置对象和其他输入参数
    def __init__(self, config: BartConfig, load_weight_prefix=None, *inputs, **kwargs):
        # 调用父类的初始化方法
        super().__init__(config, *inputs, **kwargs)
        
        # 创建 TFBartMainLayer 实例作为 self.model，用于处理 BART 的主体部分
        self.model = TFBartMainLayer(config, load_weight_prefix=load_weight_prefix, name="model")

    # 返回 encoder 部分
    def get_encoder(self):
        return self.model.encoder

    # 返回 decoder 部分
    def get_decoder(self):
        return self.model.decoder

    # 调用方法，用于执行 BART 模型的前向传播
    @add_start_docstrings_to_model_forward(BART_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFSeq2SeqModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    @unpack_inputs
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
        training: Optional[bool] = False,
        **kwargs,
    ) -> Union[TFBaseModelOutput, Tuple[tf.Tensor]]:
        # 调用 self.model 的前向传播，传递所有参数
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
    # 定义一个方法用于处理模型的输出
    def serving_output(self, output):
        # 如果配置中使用缓存，则提取输出中的过去键值对中的第一个元素
        pkv = tf.tuple(output.past_key_values)[1] if self.config.use_cache else None
        # 如果配置中输出隐藏状态，则将输出的解码器隐藏状态转换为张量
        dec_hs = tf.convert_to_tensor(output.decoder_hidden_states) if self.config.output_hidden_states else None
        # 如果配置中输出注意力权重，则将输出的解码器注意力权重转换为张量
        dec_attns = tf.convert_to_tensor(output.decoder_attentions) if self.config.output_attentions else None
        # 如果配置中输出交叉注意力权重，则将输出的交叉注意力权重转换为张量
        cross_attns = tf.convert_to_tensor(output.cross_attentions) if self.config.output_attentions else None
        # 如果配置中输出隐藏状态，则将输出的编码器隐藏状态转换为张量
        enc_hs = tf.convert_to_tensor(output.encoder_hidden_states) if self.config.output_hidden_states else None
        # 如果配置中输出注意力权重，则将输出的编码器注意力权重转换为张量
        enc_attns = tf.convert_to_tensor(output.encoder_attentions) if self.config.output_attentions else None

        # 返回一个 TFSeq2SeqModelOutput 对象，包括最后隐藏状态、过去键值对、解码器隐藏状态、解码器注意力权重、
        # 交叉注意力权重、编码器最后隐藏状态、编码器隐藏状态和编码器注意力权重
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
        # 如果模型已经构建完成，则直接返回
        if self.built:
            return
        # 将模型标记为已构建
        self.built = True
        # 如果对象中存在模型属性，则在模型的命名空间下构建模型
        if getattr(self, "model", None) is not None:
            with tf.name_scope(self.model.name):
                self.model.build(None)
class BiasLayer(keras.layers.Layer):
    """
    Bias as a layer. It is used for serialization purposes: `keras.Model.save_weights` stores on a per-layer basis,
    so all weights have to be registered in a layer.
    """

    def __init__(self, shape, initializer, trainable, name, **kwargs):
        super().__init__(name=name, **kwargs)
        # 添加一个权重变量作为偏置，用于神经网络层的偏置项
        self.bias = self.add_weight(name=name, shape=shape, initializer=initializer, trainable=trainable)

    def call(self, x):
        # 在前向传播中，将输入张量和偏置相加并返回
        return x + self.bias


@add_start_docstrings(
    "The BART Model with a language modeling head. Can be used for summarization.",
    BART_START_DOCSTRING,
)
class TFBartForConditionalGeneration(TFBartPretrainedModel, TFCausalLanguageModelingLoss):
    _keys_to_ignore_on_load_missing = [r"final_logits_bias"]
    _requires_load_weight_prefix = True

    def __init__(self, config, load_weight_prefix=None, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        # 创建 BART 模型的主体部分，包括编码器和解码器
        self.model = TFBartMainLayer(config, load_weight_prefix=load_weight_prefix, name="model")
        self.use_cache = config.use_cache
        # 创建一个偏置层用于处理最终输出的偏置 logits
        # 在 PyTorch 中，final_logits_bias 被注册为一个缓冲区，为了保持一致性，这里设置为不可训练
        self.bias_layer = BiasLayer(
            name="final_logits_bias", shape=[1, config.vocab_size], initializer="zeros", trainable=False
        )

    def get_decoder(self):
        # 返回 BART 模型的解码器
        return self.model.decoder

    def get_encoder(self):
        # 返回 BART 模型的编码器
        return self.model.encoder

    def get_output_embeddings(self):
        # 返回输入嵌入层，用于获取输出的词汇表嵌入
        return self.get_input_embeddings()

    def set_output_embeddings(self, value):
        # 设置输入嵌入层，用于设置输出的词汇表嵌入
        self.set_input_embeddings(value)

    def get_bias(self):
        # 返回偏置层的偏置值，用于模型保存和加载
        return {"final_logits_bias": self.bias_layer.bias}

    def set_bias(self, value):
        # 替换现有的包含偏置的层，确保正确的序列化和反序列化
        vocab_size = value["final_logits_bias"].shape[-1]
        self.bias_layer = BiasLayer(
            name="final_logits_bias", shape=[1, vocab_size], initializer="zeros", trainable=False
        )
        self.bias_layer.bias.assign(value["final_logits_bias"])

    @add_start_docstrings_to_model_forward(BART_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFSeq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    @add_end_docstrings(BART_GENERATION_EXAMPLE)
    @unpack_inputs
    # 定义一个方法，用于调用模型。以下是方法的参数列表，每个参数都有特定的类型注解和默认值。

    # 输入序列的标识符，可以是 TFModelInputType 类型或者 None
    input_ids: TFModelInputType | None = None,

    # 注意力掩码，可以是 numpy 数组、TensorFlow 张量或者 None
    attention_mask: np.ndarray | tf.Tensor | None = None,

    # 解码器输入序列的标识符，可以是 numpy 数组、TensorFlow 张量或者 None
    decoder_input_ids: np.ndarray | tf.Tensor | None = None,

    # 解码器的注意力掩码，可以是 numpy 数组、TensorFlow 张量或者 None
    decoder_attention_mask: np.ndarray | tf.Tensor | None = None,

    # 解码器的位置标识符，可以是 numpy 数组、TensorFlow 张量或者 None
    decoder_position_ids: np.ndarray | tf.Tensor | None = None,

    # 头部掩码，用于控制每个注意力头部的屏蔽情况，可以是 numpy 数组、TensorFlow 张量或者 None
    head_mask: np.ndarray | tf.Tensor | None = None,

    # 解码器头部掩码，用于控制解码器每个注意力头部的屏蔽情况，可以是 numpy 数组、TensorFlow 张量或者 None
    decoder_head_mask: np.ndarray | tf.Tensor | None = None,

    # 交叉注意力头部掩码，用于控制编码器-解码器注意力每个头部的屏蔽情况，可以是 numpy 数组、TensorFlow 张量或者 None
    cross_attn_head_mask: np.ndarray | tf.Tensor | None = None,

    # 编码器输出，类型为 TFBaseModelOutput 或者 None
    encoder_outputs: Optional[TFBaseModelOutput] = None,

    # 缓存的键值对，可以是包含 numpy 数组或 TensorFlow 张量的元组的元组，或者 None
    past_key_values: Optional[Tuple[Tuple[Union[np.ndarray, tf.Tensor]]]] = None,

    # 输入的嵌入向量，可以是 numpy 数组、TensorFlow 张量或者 None
    inputs_embeds: np.ndarray | tf.Tensor | None = None,

    # 解码器的输入嵌入向量，可以是 numpy 数组、TensorFlow 张量或者 None
    decoder_inputs_embeds: np.ndarray | tf.Tensor | None = None,

    # 是否使用缓存，可以是布尔值或者 None
    use_cache: Optional[bool] = None,

    # 是否输出注意力权重，可以是布尔值或者 None
    output_attentions: Optional[bool] = None,

    # 是否输出隐藏状态，可以是布尔值或者 None
    output_hidden_states: Optional[bool] = None,

    # 是否返回一个字典格式的结果，可以是布尔值或者 None
    return_dict: Optional[bool] = None,

    # 标签，类型为 TensorFlow 张量或者 None
    labels: tf.Tensor | None = None,

    # 是否在训练模式，可以是布尔值，默认为 False
    training: Optional[bool] = False,
        ) -> Union[TFSeq2SeqLMOutput, Tuple[tf.Tensor]]:
        r"""
        labels (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:
            Either a `TFSeq2SeqLMOutput` object or a tuple containing a `tf.Tensor` depending on the `return_dict` parameter.

        """

        if labels is not None:
            # Replace tokens equal to pad_token_id with -100 for loss computation
            labels = tf.where(
                labels == self.config.pad_token_id,
                tf.cast(tf.fill(shape_list(labels), -100), labels.dtype),
                labels,
            )
            use_cache = False
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                # Shift labels to the right to obtain decoder_input_ids
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
        
        # Calculate logits and apply bias layer
        lm_logits = tf.matmul(outputs[0], self.model.shared.weights, transpose_b=True)
        lm_logits = self.bias_layer(lm_logits)
        
        # Compute masked language modeling loss if labels are provided
        masked_lm_loss = None if labels is None else self.hf_compute_loss(labels, lm_logits)

        # Prepare output based on return_dict flag
        if not return_dict:
            # Return tuple of lm_logits and other outputs
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output
        else:
            # Return TFSeq2SeqLMOutput object with specified attributes
            return TFSeq2SeqLMOutput(
                loss=masked_lm_loss,
                logits=lm_logits,
                past_key_values=outputs.past_key_values,  # index 1 of d outputs
                decoder_hidden_states=outputs.decoder_hidden_states,  # index 2 of d outputs
                decoder_attentions=outputs.decoder_attentions,  # index 3 of d outputs
                cross_attentions=outputs.cross_attentions,  # index 4 of d outputs
                encoder_last_hidden_state=outputs.encoder_last_hidden_state,  # index 0 of encoder outputs
                encoder_hidden_states=outputs.encoder_hidden_states,  # index 1 of encoder outputs
                encoder_attentions=outputs.encoder_attentions,  # index 2 of encoder outputs
            )
    # 定义一个方法，用于生成服务端输出对象，基于给定的模型输出参数
    def serving_output(self, output):
        # 如果配置中指定使用缓存，则从输出中提取过去的键值对，否则设置为 None
        pkv = tf.tuple(output.past_key_values)[1] if self.config.use_cache else None
        # 如果配置中设置输出隐藏状态，则将输出的解码器隐藏状态转换为张量，否则设置为 None
        dec_hs = tf.convert_to_tensor(output.decoder_hidden_states) if self.config.output_hidden_states else None
        # 如果配置中设置输出注意力权重，则将输出的解码器注意力权重转换为张量，否则设置为 None
        dec_attns = tf.convert_to_tensor(output.decoder_attentions) if self.config.output_attentions else None
        # 如果配置中设置输出交叉注意力权重，则将输出的交叉注意力权重转换为张量，否则设置为 None
        cross_attns = tf.convert_to_tensor(output.cross_attentions) if self.config.output_attentions else None
        # 如果配置中设置输出隐藏状态，则将输出的编码器隐藏状态转换为张量，否则设置为 None
        enc_hs = tf.convert_to_tensor(output.encoder_hidden_states) if self.config.output_hidden_states else None
        # 如果配置中设置输出注意力权重，则将输出的编码器注意力权重转换为张量，否则设置为 None
        enc_attns = tf.convert_to_tensor(output.encoder_attentions) if self.config.output_attentions else None

        # 返回一个 TFSeq2SeqLMOutput 对象，包含经处理后的输出参数
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

    # 定义一个方法，为生成过程准备输入参数
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
        # 如果 past_key_values 不为 None，则将 decoder_input_ids 截取最后一个位置的输入
        if past_key_values is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        # 如果 decoder_attention_mask 不为 None，用于 XLA 编译
        if decoder_attention_mask is not None:  # xla
            # 计算 decoder_attention_mask 在最后一个维度上的累积和，然后取最后一个位置的值
            decoder_position_ids = tf.math.cumsum(decoder_attention_mask, axis=-1, exclusive=True)[:, -1:]
        # 否则，如果同时没有使用 XLA 和 past_key_values
        elif past_key_values is not None:  # no xla + past_key_values
            # 获取 past_key_values 中第一个元素的第一个维度的长度作为 decoder_position_ids
            decoder_position_ids = past_key_values[0][0].shape[2]
        else:
            # 否则，使用 decoder_input_ids 的长度范围作为 decoder_position_ids
            decoder_position_ids = tf.range(decoder_input_ids.shape[1])

        # 返回一个字典，包含生成过程中所需的输入参数
        return {
            "input_ids": None,  # 如果定义了 encoder_outputs，则不需要 input_ids
            "encoder_outputs": encoder_outputs,
            "past_key_values": past_key_values,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "decoder_attention_mask": decoder_attention_mask,
            "decoder_position_ids": decoder_position_ids,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,  # 修改此项以避免缓存（可能用于调试目的）
        }

    # 定义一个方法，从标签生成器中准备解码器输入的标识符
    def prepare_decoder_input_ids_from_labels(self, labels: tf.Tensor):
        # 使用 shift_tokens_right 函数将标签右移，用 pad_token_id 填充，并加入 decoder_start_token_id
        return shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)
    # 定义一个方法 `build`，用于构建模型的结构
    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回，避免重复构建
        if self.built:
            return
        # 将构建状态标记为已构建
        self.built = True
        
        # 如果模型属性存在，则为模型命名空间添加一个名为模型名称的作用域，并构建模型
        if getattr(self, "model", None) is not None:
            with tf.name_scope(self.model.name):
                self.model.build(None)
        
        # 如果偏置层属性存在，则为偏置层命名空间添加一个名为偏置层名称的作用域，并构建偏置层
        if getattr(self, "bias_layer", None) is not None:
            with tf.name_scope(self.bias_layer.name):
                self.bias_layer.build(None)
@add_start_docstrings(
    """
    Bart model with a sequence classification/head on top (a linear layer on top of the pooled output) e.g. for GLUE
    tasks.
    """,
    BART_START_DOCSTRING,
)
class TFBartForSequenceClassification(TFBartPretrainedModel, TFSequenceClassificationLoss):
    def __init__(self, config: BartConfig, load_weight_prefix=None, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        # 初始化 BART 主模型层，加载预训练权重（如果提供），命名为 "model"
        self.model = TFBartMainLayer(config, load_weight_prefix=load_weight_prefix, name="model")
        # 初始化 BART 分类头部，用于分类任务，包括一个线性层在汇聚输出之上，设置丢弃率为 config.classifier_dropout，命名为 "classification_head"
        self.classification_head = TFBartClassificationHead(
            config.d_model, config.num_labels, config.classifier_dropout, name="classification_head"
        )

    @add_start_docstrings_to_model_forward(BART_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFSeq2SeqSequenceClassifierOutput, config_class=_CONFIG_FOR_DOC)
    @unpack_inputs
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
        encoder_outputs: Optional[TFBaseModelOutput] = None,
        past_key_values: Optional[Tuple[Tuple[Union[np.ndarray, tf.Tensor]]]] = None,
        inputs_embeds: np.ndarray | tf.Tensor | None = None,
        decoder_inputs_embeds: np.ndarray | tf.Tensor | None = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: tf.Tensor | None = None,
        training: Optional[bool] = False,
        # 输入参数详细描述见 BART_INPUTS_DOCSTRING
    # 定义一个方法用于处理模型的输出，将输出转换为 TensorFlow 张量
    def serving_output(self, output):
        # 将输出中的 logits 转换为 TensorFlow 张量
        logits = tf.convert_to_tensor(output.logits)
        # 如果配置要求使用缓存，则从输出中提取过去的键值
        pkv = tf.tuple(output.past_key_values)[1] if self.config.use_cache else None
        # 如果配置要求输出隐藏状态，则将输出中的解码器隐藏状态转换为 TensorFlow 张量
        dec_hs = tf.convert_to_tensor(output.decoder_hidden_states) if self.config.output_hidden_states else None
        # 如果配置要求输出注意力分布，则将输出中的解码器注意力分布转换为 TensorFlow 张量
        dec_attns = tf.convert_to_tensor(output.decoder_attentions) if self.config.output_attentions else None
        # 如果配置要求输出注意力分布，则将输出中的交叉注意力分布转换为 TensorFlow 张量
        cross_attns = tf.convert_to_tensor(output.cross_attentions) if self.config.output_attentions else None
        # 如果配置要求输出隐藏状态，则将输出中的编码器隐藏状态转换为 TensorFlow 张量
        enc_hs = tf.convert_to_tensor(output.encoder_hidden_states) if self.config.output_hidden_states else None
        # 如果配置要求输出注意力分布，则将输出中的编码器注意力分布转换为 TensorFlow 张量
        enc_attns = tf.convert_to_tensor(output.encoder_attentions) if self.config.output_attentions else None

        # 返回转换后的输出对象 TFSeq2SeqSequenceClassifierOutput
        return TFSeq2SeqSequenceClassifierOutput(
            logits=logits,
            past_key_values=pkv,
            decoder_hidden_states=dec_hs,
            decoder_attentions=dec_attns,
            cross_attentions=cross_attns,
            encoder_last_hidden_state=output.encoder_last_hidden_state,
            encoder_hidden_states=enc_hs,
            encoder_attentions=enc_attns,
        )

    # 构建方法，用于构建模型
    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回
        if self.built:
            return
        # 设置标志位，表示模型已构建
        self.built = True
        # 如果对象中存在模型属性，则在命名作用域内构建模型
        if getattr(self, "model", None) is not None:
            with tf.name_scope(self.model.name):
                self.model.build(None)
        # 如果对象中存在分类头属性，则在命名作用域内构建分类头
        if getattr(self, "classification_head", None) is not None:
            with tf.name_scope(self.classification_head.name):
                self.classification_head.build(None)
```