# `.\models\ctrl\modeling_tf_ctrl.py`

```py
# 设定编码格式为 UTF-8
# 版权声明
# 导入必要的包和模块
# 控制可以表示序列的配置信息
# 获取初始化器
# 定义一个函数，根据序列位置、维度和模型大小计算角度
def angle_defn(pos, i, d_model_size):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / d_model_size)
    return pos * angle_rates

# 定义一个函数，生成位置编码
def positional_encoding(position, d_model_size):
    # 创建用于位置编码的正弦模式
    angle_rads = angle_defn(np.arange(position)[:, np.newaxis], np.arange(d_model_size)[np.newaxis, :], d_model_size)

    sines = np.sin(angle_rads[:, 0::2])
    cosines = np.cos(angle_rads[:, 1::2])
    pos_encoding = tf.convert_to_tensor(np.concatenate([sines, cosines], axis=-1))

    return pos_encoding

# 定义一个函数，执行缩放点积注意力机制
def scaled_dot_product_attention(q, k, v, mask, attention_mask=None, head_mask=None):
    # 计算注意力
    matmul_qk = tf.matmul(q, k, transpose_b=True)

    dk = tf.cast(shape_list(k)[-1], dtype=matmul_qk.dtype)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    if mask is not None:
        scaled_attention_logits += tf.cast(mask * -1e4, dtype=scaled_attention_logits.dtype)

    if attention_mask is not None:
        # 应用注意力遮罩
        attention_mask = tf.cast(attention_mask, dtype=scaled_attention_logits.dtype)
        scaled_attention_logits = scaled_attention_logits + attention_mask

    attention_weights = stable_softmax(scaled_attention_logits, axis=-1)

    # 如果需要，对注意力头进行遮罩
    # 如果head_mask不为空
    if head_mask is not None:
        # 将注意力权重与head_mask相乘，实现掩码操作
        attention_weights = attention_weights * head_mask

    # 通过矩阵乘法计算输出结果
    output = tf.matmul(attention_weights, v)

    # 返回输出结果和注意力权重
    return output, attention_weights
# 定义了一个名为TFMultiHeadAttention的类，继承自tf.keras.layers.Layer
class TFMultiHeadAttention(tf.keras.layers.Layer):
    # 初始化方法，接受参数d_model_size表示模型大小，num_heads表示注意力头数，output_attentions表示是否输出注意力矩阵，**kwargs表示其它参数
    def __init__(self, d_model_size, num_heads, output_attentions=False, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.d_model_size = d_model_size
        self.output_attentions = output_attentions

        # 计算每个注意力头的深度
        self.depth = int(d_model_size / self.num_heads)

        # 创建Wq、Wk、Wv三个Dense层
        self.Wq = tf.keras.layers.Dense(d_model_size, name="Wq")
        self.Wk = tf.keras.layers.Dense(d_model_size, name="Wk")
        self.Wv = tf.keras.layers.Dense(d_model_size, name="Wv")

        # 创建一个Dense层
        self.dense = tf.keras.layers.Dense(d_model_size, name="dense")

    # 将输入x分割成多个头
    def split_into_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    # 调用方法，接受参数v、k、q、mask、layer_past、attention_mask、head_mask、use_cache、output_attentions、training
    def call(self, v, k, q, mask, layer_past, attention_mask, head_mask, use_cache, output_attentions, training=False):
        # 获取批量大小
        batch_size = shape_list(q)[0]

        # 使用Wq、Wk、Wv对q、k、v进行变换
        q = self.Wq(q)
        k = self.Wk(k)
        v = self.Wv(v)

        # 将q、k、v分割成多个头
        q = self.split_into_heads(q, batch_size)
        k = self.split_into_heads(k, batch_size)
        v = self.split_into_heads(v, batch_size)

        # 如果layer_past不为None，则在k和v的第一个维度上进行连接
        if layer_past is not None:
            past_key, past_value = tf.unstack(layer_past, axis=0)
            k = tf.concat((past_key, k), axis=-2)
            v = tf.concat((past_value, v), axis=-2)

        # 如果use_cache为True，则将k和v进行堆叠
        if use_cache:
            present = tf.stack((k, v), axis=0)
        else:
            present = (None,)

        # 调用scaled_dot_product_attention方法，进行注意力计算
        output = scaled_dot_product_attention(q, k, v, mask, attention_mask, head_mask)
        scaled_attention = tf.transpose(output[0], perm=[0, 2, 1, 3])
        attn = output[1]
        original_size_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model_size))
        output = self.dense(original_size_attention)
        outputs = (output, present)

        # 如果output_attentions为True，则将attn添加到outputs中
        if output_attentions:
            outputs = outputs + (attn,)

        return outputs

    # 构建方法
    def build(self, input_shape=None):
        # 如果已经构建完成，则直接返回
        if self.built:
            return
        self.built = True
        # 如果Wq存在则构建Wq
        if getattr(self, "Wq", None) is not None:
            with tf.name_scope(self.Wq.name):
                self.Wq.build([None, None, self.d_model_size])
        # 如果Wk存在则构建Wk
        if getattr(self, "Wk", None) is not None:
            with tf.name_scope(self.Wk.name):
                self.Wk.build([None, None, self.d_model_size])
        # 如果Wv存在则构建Wv
        if getattr(self, "Wv", None) is not None:
            with tf.name_scope(self.Wv.name):
                self.Wv.build([None, None, self.d_model_size])
        # 如果dense存在则构建dense
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.d_model_size])

class TFPointWiseFeedForwardLayer(tf.keras.layers.Layer):
    # 省略部分代码
    # 定义初始化方法，用于设置模型的参数
    def __init__(self, d_model_size, dff, **kwargs):
        super().__init__(**kwargs)
        
        # 创建一个全连接层，神经元数量为 dff，激活函数为 ReLU，命名为 dense_0
        self.dense_0 = tf.keras.layers.Dense(dff, activation="relu", name="0")
        
        # 创建一个全连接层，神经元数量为 d_model_size，没有激活函数，命名为 dense_2
        self.dense_2 = tf.keras.layers.Dense(d_model_size, name="2")
        
        # 保存参数 d_model_size 和 dff
        self.d_model_size = d_model_size
        self.dff = dff

    # 定义模型的前向传播过程
    def call(self, inputs, trainable=False):
        # 将输入数据传递给 dense_0 层，得到输出 dense_0_output
        dense_0_output = self.dense_0(inputs)
        
        # 将 dense_0_output 传递给 dense_2 层，得到最终输出 dense_2_output
        dense_2_output = self.dense_2(dense_0_output)

        # 返回 dense_2_output 作为模型的输出结果
        return dense_2_output
    
    # 定义模型的构建过程，在这里生成模型的权重
    def build(self, input_shape=None):
        # 如果模型已经构建过，则直接返回
        if self.built:
            return
        self.built = True
        
        # 如果 self.dense_0 已经定义，就使用 name_scope 为其设置名字并构建权重
        if getattr(self, "dense_0", None) is not None:
            with tf.name_scope(self.dense_0.name):
                self.dense_0.build([None, None, self.d_model_size])
        
        # 如果 self.dense_2 已经定义，就使用 name_scope 为其设置名字并构建权重
        if getattr(self, "dense_2", None) is not None:
            with tf.name_scope(self.dense_2.name):
                self.dense_2.build([None, None, self.dff])
# 定义一个 TFEncoderLayer 类，继承自 tf.keras.layers.Layer
class TFEncoderLayer(tf.keras.layers.Layer):
    # 初始化函数，接收一系列参数
    def __init__(
        self, d_model_size, num_heads, dff, rate=0.1, layer_norm_epsilon=1e-6, output_attentions=False, **kwargs
    ):
        super().__init__(**kwargs)

        # 是否输出注意力矩阵
        self.output_attentions = output_attentions

        # 创建一个 TFMultiHeadAttention 对象，用于进行多头注意力计算
        self.multi_head_attention = TFMultiHeadAttention(
            d_model_size, num_heads, output_attentions=self.output_attentions, name="multi_head_attention"
        )

        # 创建一个 TFPointWiseFeedForwardLayer 对象，用于前馈神经网络计算
        self.ffn = TFPointWiseFeedForwardLayer(d_model_size, dff, name="ffn")

        # 创建一个 LayerNormalization 对象，用于层归一化
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=layer_norm_epsilon, name="layernorm1")
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=layer_norm_epsilon, name="layernorm2")

        # 创建一个 Dropout 对象，用于随机失活
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

        # 记录 d_model_size
        self.d_model_size = d_model_size

    # 前向传播函数
    def call(self, x, mask, layer_past, attention_mask, head_mask, use_cache, output_attentions, training=False):
        # 对输入进行层归一化
        normed = self.layernorm1(x)

        # 使用多头注意力计算
        attn_outputs = self.multi_head_attention(
            normed,
            normed,
            normed,
            mask,
            layer_past,
            attention_mask,
            head_mask,
            use_cache,
            output_attentions,
            training=training,
        )

        # 获取注意力计算的输出结果
        attn_output = attn_outputs[0]

        # 对注意力计算的输出结果进行随机失活
        attn_output = self.dropout1(attn_output, training=training)

        # 对输入和注意力计算的输出进行相加
        out1 = x + attn_output

        # 对相加的结果进行层归一化
        out2 = self.layernorm2(out1)

        # 使用前馈神经网络计算
        ffn_output = self.ffn(out2)

        # 对前馈神经网络计算的结果进行随机失活
        ffn_output = self.dropout2(ffn_output, training=training)

        # 对前馈神经网络计算的结果和之前层归一化的结果进行相加
        out2 = out1 + ffn_output

        # 将输出结果和注意力计算的额外结果合并成一个元组
        outputs = (out2,) + attn_outputs[1:]

        # 返回输出结果
        return outputs

    # ��建网络层
    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True

        # 构建多头注意力计算网络层
        if getattr(self, "multi_head_attention", None) is not None:
            with tf.name_scope(self.multi_head_attention.name):
                self.multi_head_attention.build(None)

        # 构建前馈神经网络计算网络层
        if getattr(self, "ffn", None) is not None:
            with tf.name_scope(self.ffn.name):
                self.ffn.build(None)

        # 构建层归一化网络层
        if getattr(self, "layernorm1", None) is not None:
            with tf.name_scope(self.layernorm1.name):
                self.layernorm1.build([None, None, self.d_model_size])
        if getattr(self, "layernorm2", None) is not None:
            with tf.name_scope(self.layernorm2.name):
                self.layernorm2.build([None, None, self.d_model_size])


# 定义一个 TFCTRLMainLayer 类，继承自 tf.keras.layers.Layer
@keras_serializable
class TFCTRLMainLayer(tf.keras.layers.Layer):
    # 配置类
    config_class = CTRLConfig
    # 初始化函数，接受配置和可选参数
    def __init__(self, config, **kwargs):
        # 调用父类初始化函数
        super().__init__(**kwargs)

        # 将配置信息存储到实例中
        self.config = config
        self.output_hidden_states = config.output_hidden_states
        self.output_attentions = config.output_attentions
        self.use_cache = config.use_cache
        self.return_dict = config.use_return_dict

        # 存储模型中的一些参数
        self.d_model_size = config.n_embd
        self.num_layers = config.n_layer

        # 创建位置编码矩阵
        self.pos_encoding = positional_encoding(config.n_positions, self.d_model_size)

        # 创建词嵌入层
        self.w = tf.keras.layers.Embedding(
            input_dim=config.vocab_size,
            output_dim=config.n_embd,
            embeddings_initializer=get_initializer(config.initializer_range),
            name="w",
        )

        # 创建Dropout层
        self.dropout = tf.keras.layers.Dropout(config.embd_pdrop)

        # 创建Transformer Encoder层
        self.h = [
            TFEncoderLayer(
                config.n_embd,
                config.n_head,
                config.dff,
                config.resid_pdrop,
                config.layer_norm_epsilon,
                self.output_attentions,
                name=f"h_._{i}",
            )
            for i in range(config.n_layer)
        ]

        # 创建LayerNormalization层
        self.layernorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_epsilon, name="layernorm")

    # 返回词嵌入层
    def get_input_embeddings(self):
        return self.w

    # 设置新的词嵌入层
    def set_input_embeddings(self, new_embeddings):
        self.w = new_embeddings

    # 剪枝模型中的注意力头
    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        """
        raise NotImplementedError

    # 调用模型，接受多个输入参数
    @unpack_inputs
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        past_key_values: Optional[Tuple[Tuple[Union[np.ndarray, tf.Tensor]]] = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        token_type_ids: np.ndarray | tf.Tensor | None = None,
        position_ids: np.ndarray | tf.Tensor | None = None,
        head_mask: np.ndarray | tf.Tensor | None = None,
        inputs_embeds: np.ndarray | tf.Tensor | None = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: Optional[bool] = False,
    ) -> TFBaseModelOutput:
        raise NotImplementedError

    # 构建模型
    def build(self, input_shape=None):
        # 如果已经构建过模型，则直接返回
        if self.built:
            return
        self.built = True
        # 构建词嵌入层
        if getattr(self, "w", None) is not None:
            with tf.name_scope(self.w.name):
                self.w.build(None)
        # 构建LayerNormalization层
        if getattr(self, "layernorm", None) is not None:
            with tf.name_scope(self.layernorm.name):
                self.layernorm.build([None, None, self.config.o_embd])
        # 构建Transformer Encoder层
        if getattr(self, "h", None) is not None:
            for layer in self.h:
                with tf.name_scope(layer.name):
                    layer.build(None)
# `TFCTRLPreTrainedModel`类是`TFPreTrainedModel`类的子类，用于处理权重初始化、预训练模型的下载和加载等操作
class TFCTRLPreTrainedModel(TFPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # `CTRLConfig`类是该模型配置的基类
    config_class = CTRLConfig
    # 模型的前缀
    base_model_prefix = "transformer"


# `CTRL_START_DOCSTRING`是一个多行注释字符串，用于说明`TFCTRLModel`类的文档字符串
CTRL_START_DOCSTRING = r"""

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
        config ([`CTRLConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

# `CTRL_INPUTS_DOCSTRING`是一个空字符串，用于说明`TFCTRLModel`类的输入参数
CTRL_INPUTS_DOCSTRING = r"""
"""


@add_start_docstrings(
    "The bare CTRL Model transformer outputting raw hidden-states without any specific head on top.",
    CTRL_START_DOCSTRING,
)
class TFCTRLModel(TFCTRLPreTrainedModel):
    # 该类继承于`TFCTRLPreTrainedModel`
    # 初始化方法，接受配置和输入参数
    def __init__(self, config, *inputs, **kwargs):
        # 调用父类的初始化方法
        super().__init__(config, *inputs, **kwargs)
        # 创建一个名为"transformer"的TFCTRLMainLayer层实例
        self.transformer = TFCTRLMainLayer(config, name="transformer")

    # 装饰器：将输入参数解包并传递给被装饰的函数
    @unpack_inputs
    # 为模型前向传播添加文档字符串
    @add_start_docstrings_to_model_forward(CTRL_INPUTS_DOCSTRING)
    # 为代码示例添加文档字符串
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFBaseModelOutputWithPast,
        config_class=_CONFIG_FOR_DOC,
    )
    # 定义call方法
    def call(
        self,
        # 输入的token id序列
        input_ids: TFModelInputType | None = None,
        # 用于存储历史键值的元组
        past_key_values: Optional[Tuple[Tuple[Union[np.ndarray, tf.Tensor]]]] = None,
        # 注意力遮罩
        attention_mask: np.ndarray | tf.Tensor | None = None,
        # token类型id
        token_type_ids: np.ndarray | tf.Tensor | None = None,
        # 位置id
        position_ids: np.ndarray | tf.Tensor | None = None,
        # 头部遮罩
        head_mask: np.ndarray | tf.Tensor | None = None,
        # 输入嵌入
        inputs_embeds: np.ndarray | tf.Tensor | None = None,
        # 是否使用缓存
        use_cache: Optional[bool] = None,
        # 是否输出注意力
        output_attentions: Optional[bool] = None,
        # 是否输出隐藏状态
        output_hidden_states: Optional[bool] = None,
        # 是否返回字典
        return_dict: Optional[bool] = None,
        # 是否处于训练状态
        training: Optional[bool] = False,
    ) -> Union[Tuple, TFBaseModelOutputWithPast]:
        # 调用transformer层的前向传播方法
        outputs = self.transformer(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )
        # 返回前向传播结果
        return outputs

    # 构建模型
    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回
        if self.built:
            return
        # 设置模型为已构建状态
        self.built = True
        # 如果存在transformer层
        if getattr(self, "transformer", None) is not None:
            # 在transformer层的命名空间下构建模型
            with tf.name_scope(self.transformer.name):
                self.transformer.build(None)
class TFCTRLBiasLayer(tf.keras.layers.Layer):
    """
    Bias as a layer. It is used for serialization purposes: `tf.keras.Model.save_weights` stores on a per-layer basis,
    so all weights have to be registered in a layer.
    """

    def __init__(self, shape, initializer, trainable, name, **kwargs):
        # 初始化函数，接收形状、初始化器、是否可训练、名称等参数
        super().__init__(name=name, **kwargs)
        self.shape = shape
        self.initializer = initializer
        self.trainable = trainable

    def build(self, input_shape):
        # 根据输入形状构建该层的权重
        self.bias = self.add_weight(
            name="bias", shape=self.shape, initializer=self.initializer, trainable=self.trainable
        )
        # 构建函数继承父类的 build 方法
        super().build(input_shape)

    def call(self, x):
        # 在调用时返回输入加上偏置项
        return x + self.bias


@add_start_docstrings(
    """
    The CTRL Model transformer with a language modeling head on top (linear layer with weights tied to the input
    embeddings).
    """,
    CTRL_START_DOCSTRING,
)
class TFCTRLLMHeadModel(TFCTRLPreTrainedModel, TFCausalLanguageModelingLoss):
    def __init__(self, config, *inputs, **kwargs):
        # 初始化函数，接收配置项、输入等参数
        super().__init__(config, *inputs, **kwargs)
        # 初始化transformer和bias_layer层
        self.transformer = TFCTRLMainLayer(config, name="transformer")
        self.bias_layer = TFCTRLBiasLayer(
            name="lm_head", shape=[1, config.vocab_size], initializer="zeros", trainable=True
        )

    def get_output_embeddings():
        return self.get_input_embeddings()

    def set_output_embeddings(value):
        self.set_input_embeddings(value)

    def get_bias():
        return {"lm_head.bias": self.bias_layer.bias}

    def set_bias(value):
        # 替换包含偏置项的现有层，以进行正确的（反）序列化
        vocab_size = value["lm_head.bias"].shape[-1]
        self.bias_layer = TFCTRLBiasLayer(
            name="final_logits_bias", shape=[1, vocab_size], initializer="zeros", trainable=True
        )
        self.bias_layer.build(None)
        self.bias_layer.bias.assign(value["lm_head.bias"])

    # 从 transformers.models.gpt2.modeling_tf_gpt2.TFGPT2LMHeadModel.prepare_inputs_for_generation 复制过来
    # 为生成准备输入
    def prepare_inputs_for_generation(self, inputs, past_key_values=None, use_cache=None, **kwargs):
        # 获取token_type_ids，如果不存在则为None
        token_type_ids = kwargs.get("token_type_ids", None)
        # 如果past在kwargs中定义，则仅使用最后一个token作为输入
        if past_key_values:
            inputs = tf.expand_dims(inputs[:, -1], -1)
            # 如果token_type_ids不为空，则对其进行处理
            if token_type_ids is not None:
                token_type_ids = tf.expand_dims(token_type_ids[:, -1], -1)

        # 获取position_ids，如果不存在则为None
        position_ids = kwargs.get("position_ids", None)
        # 获取attention_mask，如果不存在则为None
        attention_mask = kwargs.get("attention_mask", None)

        # 如果attention_mask不为空且position_ids为空，则创建position_ids
        if attention_mask is not None and position_ids is None:
            position_ids = tf.math.cumsum(attention_mask, axis=-1, exclusive=True)
            # 如果past_key_values存在，则对position_ids进行处理
            if past_key_values:
                position_ids = tf.expand_dims(position_ids[:, -1], -1)

        # 返回准备好的输入
        return {
            "input_ids": inputs,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
            "token_type_ids": token_type_ids,
        }

    # 调用函数
    @unpack_inputs
    @add_start_docstrings_to_model_forward(CTRL_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFCausalLMOutputWithPast,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(
        # 定义输入参数及其类型和默认值
        self,
        input_ids: TFModelInputType | None = None,
        past_key_values: Optional[Tuple[Tuple[Union[np.ndarray, tf.Tensor]]]] = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        token_type_ids: np.ndarray | tf.Tensor | None = None,
        position_ids: np.ndarray | tf.Tensor | None = None,
        head_mask: np.ndarray | tf.Tensor | None = None,
        inputs_embeds: np.ndarray | tf.Tensor | None = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: np.ndarray | tf.Tensor | None = None,
        training: Optional[bool] = False,
    ) -> Union[Tuple, TFCausalLMOutputWithPast]:
        r"""
        labels (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the cross entropy classification loss. Indices should be in `[0, ...,
            config.vocab_size - 1]`.
        """
        # 使用 Transformer 处理输入序列，返回 Transformer 的输出
        transformer_outputs = self.transformer(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )
        # 从 Transformer 输出中获取隐藏状态
        hidden_states = transformer_outputs[0]
        # 计算 logits
        logits = tf.matmul(hidden_states, self.transformer.w.weights, transpose_b=True)
        # 添加偏置
        logits = self.bias_layer(logits)

        loss = None
        if labels is not None:
            # 将标签向左移动一位，并且移除最后一个 logits 标记
            shifted_logits = logits[:, :-1]
            labels = labels[:, 1:]
            # 计算损失
            loss = self.hf_compute_loss(labels, shifted_logits)

        if not return_dict:
            # 如果不返回字典，则返回 logits 和其他 Transformer 输出
            output = (logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        # 返回 TFCausalLMOutputWithPast 类型的对象，包含损失、logits 和其他 Transformer 输出
        return TFCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "transformer", None) is not None:
            with tf.name_scope(self.transformer.name):
                # 构建 Transformer
                self.transformer.build(None)
        if getattr(self, "bias_layer", None) is not None:
            with tf.name_scope(self.bias_layer.name):
                # 构建偏置层
                self.bias_layer.build(None)
# 在模型定义上添加文档字符串，说明这是一个在 CTRL 模型中添加一个序列分类头部的变压器模型（线性层）
# TFCTRLForSequenceClassification 使用最后一个标记进行分类，类似于其他因果模型 (例如 GPT-1, GPT-2)。
# 由于它对最后一个标记进行分类，因此需要知道最后一个标记的位置。如果在配置中定义了 'pad_token_id'，则它会找到每行中不是填充标记的最后一个标记。
# 如果没有定义 'pad_token_id'，则它简单地取每行中的最后一个数值。当传入 'inputs_embeds' 而不是 'input_ids' 时，由于无法猜测填充标记，它采取相同的方法（取每行的最后一个数值）。
class TFCTRLForSequenceClassification(TFCTRLPreTrainedModel, TFSequenceClassificationLoss):
    # 初始化函数
    def __init__(self, config, *inputs, **kwargs):
        # 调用父类的初始化函数
        super().__init__(config, *inputs, **kwargs)
        # 设置标签数量
        self.num_labels = config.num_labels
        # 添加分类器线性层
        self.classifier = tf.keras.layers.Dense(
            config.num_labels,
            # 初始化权重的方式
            kernel_initializer=get_initializer(config.initializer_range),
            name="classifier",
            use_bias=False,
        )
        # 添加变压器层
        self.transformer = TFCTRLMainLayer(config, name="transformer")
        # 设置配置
        self.config = config

    # 获取输出嵌入层
    def get_output_embeddings(self):
        # 从 transformers v4.32 开始移除。修复该模型的 `test_model_common_attributes` 测试。
        logger.warning(
            "Sequence classification models do not have output embeddings. `.get_output_embeddings` will be removed "
            "in transformers v4.32."
        )
        return self.transformer.w

    # 在模型前向传播中添加文档字符串，说明此时采用的输入格式
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        past_key_values: Optional[Tuple[Tuple[Union[np.ndarray, tf.Tensor]]]] = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        token_type_ids: np.ndarray | tf.Tensor | None = None,
        position_ids: np.ndarray | tf.Tensor | None = None,
        head_mask: np.ndarray | tf.Tensor | None = None,
        inputs_embeds: np.ndarray | tf.Tensor | None = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: np.ndarray | tf.Tensor | None = None,
        training: Optional[bool] = False,

这里是多行代码，每行都需要添加合适的注释。
    ) -> Union[Tuple, TFSequenceClassifierOutput]:
        r"""
        labels (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional`):
            Labels for computing the cross entropy classification loss. Indices should be in `[0, ...,
            config.vocab_size - 1]`.
        """

        # 使用transformer处理输入，获得transformer的输出
        transformer_outputs = self.transformer(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        # 从transformer的输出中获得隐藏状态
        hidden_states = transformer_outputs[0]
        # 使用分类器对隐藏状态进行分类，获得logits
        logits = self.classifier(hidden_states)
        in_logits = None
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                # 计算输入序列中padding token之前的长度作为sequence_lengths
                sequence_lengths = (
                    tf.argmax(tf.cast(tf.math.equal(input_ids, self.config.pad_token_id), input_ids.dtype), axis=-1)
                    - 1
                )
                sequence_lengths = tf.where(sequence_lengths >= 0, sequence_lengths, input_ids.shape[-1] - 1)
                in_logits = tf.gather(logits, sequence_lengths, batch_dims=1, axis=1)
            else:
                sequence_lengths = -1
                logger.warning(
                    f"{self.__class__.__name__} will not detect padding tokens in `inputs_embeds`. Results may be "
                    "unexpected if using padding tokens in conjunction with `inputs_embeds.`"
                )
        loss = None

        if labels is not None:
            if input_ids is not None:
                batch_size, sequence_length = shape_list(input_ids)[:2]
            else:
                batch_size, sequence_length = shape_list(inputs_embeds)[:2]
            if self.config.pad_token_id is None and batch_size != 1:
                raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")

            # 如果sequence_lengths不是张量，则选择logits的相应部分
            if not tf.is_tensor(sequence_lengths):
                in_logits = logits[0:batch_size, sequence_lengths]

            # 计算loss
            loss = self.hf_compute_loss(tf.reshape(labels, [-1, 1]), tf.reshape(in_logits, [-1, self.num_labels]))

        # 获取最终logits
        pooled_logits = in_logits if in_logits is not None else logits

        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        # 返回TFSequenceClassifierOutput对象
        return TFSequenceClassifierOutput(
            loss=loss,
            logits=pooled_logits,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )
    # 如果模型已经构建好了，就直接返回，不进行重复构建
    def build(self, input_shape=None):
        if self.built:
            return
        # 设置模型已经构建的标志为 True
        self.built = True
        # 如果模型中包含分类器，就构建分类器
        if getattr(self, "classifier", None) is not None:
            # 在命名范围内构建分类器
            with tf.name_scope(self.classifier.name):
                # 构建分类器，传入输入形状为 [None, None, self.config.n_embd]
                self.classifier.build([None, None, self.config.n_embd])
        # 如果模型中包含变换器，就构建变换器
        if getattr(self, "transformer", None) is not None:
            # 在命名范围内构建变换器
            with tf.name_scope(self.transformer.name):
                # 构建变换器，传入输入形状为 None
                self.transformer.build(None)
```