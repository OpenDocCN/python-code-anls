# `.\models\openai\modeling_tf_openai.py`

```py
# 定义 TFAttention 类，继承自 keras.layers.Layer，用于实现注意力机制
class TFAttention(keras.layers.Layer):
    # 初始化函数，设置注意力相关参数
    def __init__(self, nx, config, scale=False, **kwargs):
        super().__init__(**kwargs)

        # 在 Attention 中，n_state 表示隐藏状态的维度，通常等于嵌入维度 nx
        n_state = nx  # in Attention: n_state=768 (nx=n_embd)
        # 确保隐藏维度能够被注意力头数整除，保证多头注意力操作的有效性
        assert (
            n_state % config.n_head == 0
        ), f"Hidden dimension {n_state} not dividable by number of heads {config.n_head}"
        
        # 设置注意力头数和分割大小
        self.n_head = config.n_head
        self.split_size = n_state
        self.scale = scale  # 是否对注意力分数进行缩放
        self.output_attentions = config.output_attentions  # 是否输出注意力权重

        # c_attn 是注意力机制中的卷积层，用于计算注意力分布的相关参数
        self.c_attn = TFConv1D(n_state * 3, nx, initializer_range=config.initializer_range, name="c_attn")
        
        # c_proj 是注意力机制中的卷积层，用于计算最终输出的相关参数
        self.c_proj = TFConv1D(n_state, nx, initializer_range=config.initializer_range, name="c_proj")
        
        # attn_dropout 是注意力机制中的丢弃层，用于对注意力权重进行随机丢弃
        self.attn_dropout = keras.layers.Dropout(config.attn_pdrop)
        
        # resid_dropout 是注意力机制中的丢弃层，用于对残差连接进行随机丢弃
        self.resid_dropout = keras.layers.Dropout(config.resid_pdrop)
        
        # 记录隐藏状态维度和剪枝的注意力头部集合
        self.n_state = n_state
        self.pruned_heads = set()
    def prune_heads(self, heads):
        pass



    @staticmethod
    def causal_attention_mask(nd, ns):
        """
        1's in the lower triangle, counting from the lower right corner. Same as tf.matrix_band_part(tf.ones([nd, ns]),
        -1, ns-nd), but doesn't produce garbage on TPUs.
        """
        # Generate a causal attention mask matrix for self-attention mechanisms
        i = tf.range(nd)[:, None]
        j = tf.range(ns)
        m = i >= j - ns + nd
        return m



    def _attn(self, q, k, v, attention_mask, head_mask, output_attentions, training=False):
        # q, k, v have shape [batch, heads, sequence, features]
        # Compute the attention scores matrix
        w = tf.matmul(q, k, transpose_b=True)
        if self.scale:
            dk = tf.cast(shape_list(k)[-1], dtype=w.dtype)  # scale attention_scores
            w = w / tf.math.sqrt(dk)

        # Apply a causal attention mask to prevent attending to future positions
        _, _, nd, ns = shape_list(w)
        b = tf.cast(self.causal_attention_mask(nd, ns), dtype=w.dtype)
        b = tf.reshape(b, [1, 1, nd, ns])
        w = w * b - 1e4 * (1 - b)

        if attention_mask is not None:
            # Apply the provided attention mask
            attention_mask = tf.cast(attention_mask, dtype=w.dtype)
            w = w + attention_mask

        # Apply stable softmax activation function to compute attention weights
        w = stable_softmax(w, axis=-1)
        w = self.attn_dropout(w, training=training)

        # Mask heads if specified
        if head_mask is not None:
            w = w * head_mask

        outputs = [tf.matmul(w, v)]
        if output_attentions:
            outputs.append(w)
        return outputs



    def merge_heads(self, x):
        # Transpose and reshape tensor to merge the heads of multi-headed attention
        x = tf.transpose(x, [0, 2, 1, 3])
        x_shape = shape_list(x)
        new_x_shape = x_shape[:-2] + [x_shape[-2] * x_shape[-1]]
        return tf.reshape(x, new_x_shape)



    def split_heads(self, x):
        # Split tensor into multiple heads for multi-headed attention
        x_shape = shape_list(x)
        new_x_shape = x_shape[:-1] + [self.n_head, x_shape[-1] // self.n_head]
        x = tf.reshape(x, new_x_shape)
        return tf.transpose(x, (0, 2, 1, 3))  # (batch, head, seq_length, head_features)



    def call(self, x, attention_mask, head_mask, output_attentions, training=False):
        # Perform the main operation of the transformer block, including self-attention and feed-forward layers

        # Apply self-attention mechanism
        x = self.c_attn(x)
        query, key, value = tf.split(x, 3, axis=2)
        query = self.split_heads(query)
        key = self.split_heads(key)
        value = self.split_heads(value)

        # Compute attention outputs
        attn_outputs = self._attn(query, key, value, attention_mask, head_mask, output_attentions, training=training)
        a = attn_outputs[0]

        # Merge heads back together
        a = self.merge_heads(a)

        # Apply feed-forward layer and residual dropout
        a = self.c_proj(a)
        a = self.resid_dropout(a, training=training)

        # Prepare and return outputs
        outputs = [a] + attn_outputs[1:]
        return outputs  # a, (attentions)
    # 定义一个方法用于构建模型结构，可以接受输入形状参数
    def build(self, input_shape=None):
        # 如果模型已经构建过，则直接返回，不进行重复构建
        if self.built:
            return
        # 设置标志位表示模型已经构建
        self.built = True
        
        # 检查是否存在名为"c_attn"的属性，并且该属性不为None
        if getattr(self, "c_attn", None) is not None:
            # 使用 TensorFlow 的命名空间来命名"c_attn"的作用域
            with tf.name_scope(self.c_attn.name):
                # 根据给定的形状构建"c_attn"模块，形状为[None, None, self.n_state * 3]
                self.c_attn.build([None, None, self.n_state * 3])
        
        # 检查是否存在名为"c_proj"的属性，并且该属性不为None
        if getattr(self, "c_proj", None) is not None:
            # 使用 TensorFlow 的命名空间来命名"c_proj"的作用域
            with tf.name_scope(self.c_proj.name):
                # 根据给定的形状构建"c_proj"模块，形状为[None, None, self.n_state]
                self.c_proj.build([None, None, self.n_state])
# 定义一个自定义的 Keras 层 TFMLP，继承自 keras.layers.Layer 类
class TFMLP(keras.layers.Layer):
    # 初始化方法，接受神经元数目和配置参数 config
    def __init__(self, n_state, config, **kwargs):
        super().__init__(**kwargs)
        # 获取嵌入维度
        nx = config.n_embd
        # 创建一个 1 维卷积层 c_fc，用于全连接操作，使用配置中的初始化范围
        self.c_fc = TFConv1D(n_state, nx, initializer_range=config.initializer_range, name="c_fc")
        # 创建另一个 1 维卷积层 c_proj，用于投影操作，同样使用配置中的初始化范围
        self.c_proj = TFConv1D(nx, n_state, initializer_range=config.initializer_range, name="c_proj")
        # 获取 GELU 激活函数
        self.act = get_tf_activation("gelu")
        # 创建一个 dropout 层，使用给定的 resid_pdrop 参数
        self.dropout = keras.layers.Dropout(config.resid_pdrop)
        # 保存嵌入维度和神经元数目到实例变量
        self.nx = nx
        self.n_state = n_state

    # 定义调用方法，传入输入 x 和训练标志
    def call(self, x, training=False):
        # 使用 GELU 激活函数对输入进行卷积操作，并赋值给 h
        h = self.act(self.c_fc(x))
        # 对 h 进行投影操作，并赋值给 h2
        h2 = self.c_proj(h)
        # 使用 dropout 层对 h2 进行 dropout 操作，根据训练标志
        h2 = self.dropout(h2, training=training)
        # 返回处理后的 h2
        return h2

    # 定义构建方法，用于构建层次结构
    def build(self, input_shape=None):
        # 如果已经构建过，直接返回
        if self.built:
            return
        self.built = True
        # 如果存在 c_fc 层，使用 name_scope 构建 c_fc 层
        if getattr(self, "c_fc", None) is not None:
            with tf.name_scope(self.c_fc.name):
                self.c_fc.build([None, None, self.n_state])
        # 如果存在 c_proj 层，使用 name_scope 构建 c_proj 层
        if getattr(self, "c_proj", None) is not None:
            with tf.name_scope(self.c_proj.name):
                self.c_proj.build([None, None, self.nx])


# 定义一个自定义的 Keras 层 TFBlock，继承自 keras.layers.Layer 类
class TFBlock(keras.layers.Layer):
    # 初始化方法，接受配置参数 config 和一个可选的缩放参数 scale
    def __init__(self, config, scale=False, **kwargs):
        super().__init__(**kwargs)
        # 获取嵌入维度
        nx = config.n_embd
        # 创建注意力层 attn，使用配置参数和可能的缩放参数
        self.attn = TFAttention(nx, config, scale, name="attn")
        # 创建 LayerNormalization 层 ln_1，用于规范化
        self.ln_1 = keras.layers.LayerNormalization(epsilon=config.layer_norm_epsilon, name="ln_1")
        # 创建 MLP 层 mlp，四倍于嵌入维度，使用配置参数
        self.mlp = TFMLP(4 * nx, config, name="mlp")
        # 创建另一个 LayerNormalization 层 ln_2，用于规范化
        self.ln_2 = keras.layers.LayerNormalization(epsilon=config.layer_norm_epsilon, name="ln_2")
        # 保存嵌入维度到实例变量
        self.nx = nx

    # 定义调用方法，接受输入 x、注意力掩码、头部掩码、输出注意力和训练标志
    def call(self, x, attention_mask, head_mask, output_attentions, training=False):
        # 使用注意力层 attn 处理输入 x，并获取注意力输出
        output_attn = self.attn(x, attention_mask, head_mask, output_attentions, training=training)
        a = output_attn[0]  # output_attn: a, (attentions)

        # 对输入 x 和注意力输出进行加法操作，并进行 LayerNormalization
        n = self.ln_1(x + a)
        # 使用 MLP 层处理上一步的结果 n
        m = self.mlp(n, training=training)
        # 对 n 和 MLP 输出 m 进行加法操作，并进行 LayerNormalization
        h = self.ln_2(n + m)

        # 将最终结果组成列表输出，包含 h 和可能的注意力输出
        outputs = [h] + output_attn[1:]
        return outputs  # x, (attentions)

    # 定义构建方法，用于构建层次结构
    def build(self, input_shape=None):
        # 如果已经构建过，直接返回
        if self.built:
            return
        self.built = True
        # 如果存在 attn 层，使用 name_scope 构建 attn 层
        if getattr(self, "attn", None) is not None:
            with tf.name_scope(self.attn.name):
                self.attn.build(None)
        # 如果存在 ln_1 层，使用 name_scope 构建 ln_1 层
        if getattr(self, "ln_1", None) is not None:
            with tf.name_scope(self.ln_1.name):
                self.ln_1.build([None, None, self.nx])
        # 如果存在 mlp 层，使用 name_scope 构建 mlp 层
        if getattr(self, "mlp", None) is not None:
            with tf.name_scope(self.mlp.name):
                self.mlp.build(None)
        # 如果存在 ln_2 层，使用 name_scope 构建 ln_2 层
        if getattr(self, "ln_2", None) is not None:
            with tf.name_scope(self.ln_2.name):
                self.ln_2.build([None, None, self.nx])


# 定义一个 Keras 可序列化层 TFOpenAIGPTMainLayer，继承自 keras.layers.Layer 类
@keras_serializable
class TFOpenAIGPTMainLayer(keras.layers.Layer):
    # 设置配置类为 OpenAIGPTConfig
    config_class = OpenAIGPTConfig
    # 初始化函数，接受一个配置对象和额外的输入参数，并调用父类的初始化方法
    def __init__(self, config, *inputs, **kwargs):
        # 调用父类的初始化方法，传递额外的输入参数和关键字参数
        super().__init__(*inputs, **kwargs)

        # 将配置对象保存到实例中
        self.config = config
        # 是否输出隐藏层状态的标志位
        self.output_hidden_states = config.output_hidden_states
        # 是否输出注意力权重的标志位
        self.output_attentions = config.output_attentions
        # 是否使用返回字典作为输出的标志位
        self.return_dict = config.use_return_dict
        # 模型的隐藏层数量
        self.num_hidden_layers = config.n_layer
        # 嵌入向量的维度
        self.n_embd = config.n_embd
        # 位置编码的最大位置数
        self.n_positions = config.n_positions
        # 初始化范围
        self.initializer_range = config.initializer_range

        # 共享的嵌入层，用于输入的词汇表大小、嵌入维度和初始化范围
        self.tokens_embed = TFSharedEmbeddings(
            config.vocab_size, config.n_embd, initializer_range=config.initializer_range, name="tokens_embed"
        )
        # Dropout 层，使用配置中的嵌入丢弃率
        self.drop = keras.layers.Dropout(config.embd_pdrop)
        # 创建多个 Transformer Block 层，使用配置中的隐藏层数量和初始化标志
        self.h = [TFBlock(config, scale=True, name=f"h_._{i}") for i in range(config.n_layer)]

    # 构建模型，定义了位置编码的嵌入矩阵
    def build(self, input_shape=None):
        # 在 "positions_embed" 命名空间下，创建位置编码的嵌入矩阵
        with tf.name_scope("positions_embed"):
            self.positions_embed = self.add_weight(
                name="embeddings",
                shape=[self.n_positions, self.n_embd],
                initializer=get_initializer(self.initializer_range),
            )

        # 如果已经构建过，则直接返回
        if self.built:
            return
        self.built = True
        
        # 如果 tokens_embed 属性存在，则构建该属性
        if getattr(self, "tokens_embed", None) is not None:
            with tf.name_scope(self.tokens_embed.name):
                self.tokens_embed.build(None)
        
        # 如果 h 属性存在，则对每个 Transformer Block 层进行构建
        if getattr(self, "h", None) is not None:
            for layer in self.h:
                with tf.name_scope(layer.name):
                    layer.build(None)

    # 获取输入嵌入层对象
    def get_input_embeddings(self):
        return self.tokens_embed

    # 设置输入嵌入层的权重和词汇表大小
    def set_input_embeddings(self, value):
        self.tokens_embed.weight = value
        self.tokens_embed.vocab_size = shape_list(value)[0]

    # 剪枝模型中的注意力头部
    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        """
        raise NotImplementedError

    # 调用模型，实现模型的前向计算
    @unpack_inputs
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        token_type_ids: np.ndarray | tf.Tensor | None = None,
        position_ids: np.ndarray | tf.Tensor | None = None,
        head_mask: np.ndarray | tf.Tensor | None = None,
        inputs_embeds: np.ndarray | tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: Optional[bool] = False,
        OPENAI_GPT_START_DOCSTRING = r"""

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
    """



class TFOpenAIGPTPreTrainedModel(TFPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = OpenAIGPTConfig
    base_model_prefix = "transformer"



@dataclass
class TFOpenAIGPTDoubleHeadsModelOutput(ModelOutput):
    """
    Base class for outputs of models predicting if two sentences are consecutive or not.

    Args:
        logits (`tf.Tensor` of shape `(batch_size, num_choices, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        mc_logits (`tf.Tensor` of shape `(batch_size, num_choices)`):
            Prediction scores of the multiple choice classification head (scores for each choice before SoftMax).
        hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `tf.Tensor` (one for the output of the embeddings + one for the output of each layer) of shape
            `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    logits: tf.Tensor = None
    mc_logits: tf.Tensor = None
    hidden_states: Tuple[tf.Tensor] | None = None
    attentions: Tuple[tf.Tensor] | None = None



    """
    Defines a constant string providing an introductory documentation string for the OpenAI GPT model implementation.

    This docstring outlines the inheritance structure, general usage, and compatibility with TensorFlow 2.0,
    emphasizing the support for multiple input formats. It also offers a tip regarding the input format preference
    in TensorFlow's `transformers` library, ensuring seamless integration with Keras methods like `model.fit()`.
    """
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
        config ([`OpenAIGPTConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""
定义了一个文档字符串，用于描述 OpenAI GPT 相关的输入参数说明。
"""

@add_start_docstrings(
    "The bare OpenAI GPT transformer model outputting raw hidden-states without any specific head on top.",
    OPENAI_GPT_START_DOCSTRING,
)
"""
使用装饰器添加了文档字符串，描述了一个裸的 OpenAI GPT 变压器模型，输出原始的隐藏状态，没有特定的输出头。
"""

class TFOpenAIGPTModel(TFOpenAIGPTPreTrainedModel):
    """
    定义了 TFOpenAIGPTModel 类，继承自 TFOpenAIGPTPreTrainedModel。
    """

    def __init__(self, config, *inputs, **kwargs):
        """
        初始化方法，接受配置和其他参数，并调用父类初始化方法。
        """
        super().__init__(config, *inputs, **kwargs)
        self.transformer = TFOpenAIGPTMainLayer(config, name="transformer")
        """
        创建了 TFOpenAIGPTMainLayer 对象，作为 transformer 属性。
        """

    @unpack_inputs
    @add_start_docstrings_to_model_forward(OPENAI_GPT_INPUTS_DOCSTRING)
    """
    使用装饰器添加了文档字符串，描述了模型的前向传播函数，扩展了输入参数的文档说明。
    """

    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFBaseModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    """
    使用装饰器添加了代码示例的文档字符串，指定了用于文档化的检查点、输出类型和配置类。
    """

    def call(
        self,
        input_ids: TFModelInputType | None = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        token_type_ids: np.ndarray | tf.Tensor | None = None,
        position_ids: np.ndarray | tf.Tensor | None = None,
        head_mask: np.ndarray | tf.Tensor | None = None,
        inputs_embeds: np.ndarray | tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: Optional[bool] = False,
    ) -> Union[Tuple, TFBaseModelOutput]:
        """
        模型的调用方法，接受多个输入参数，并返回模型输出。
        """
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )
        return outputs

    def build(self, input_shape=None):
        """
        构建方法，用于构建模型的层次结构。
        """
        if self.built:
            return
        self.built = True
        if getattr(self, "transformer", None) is not None:
            with tf.name_scope(self.transformer.name):
                self.transformer.build(None)
                """
                在 transformer 属性上建立命名作用域，并调用其 build 方法。
                """

@add_start_docstrings(
    """
    OpenAI GPT Model transformer with a language modeling head on top (linear layer with weights tied to the input
    embeddings).
    """,
    OPENAI_GPT_START_DOCSTRING,
)
"""
使用装饰器添加了文档字符串，描述了带有语言建模头部的 OpenAI GPT 模型变压器。
"""

class TFOpenAIGPTLMHeadModel(TFOpenAIGPTPreTrainedModel, TFCausalLanguageModelingLoss):
    """
    定义了 TFOpenAIGPTLMHeadModel 类，继承自 TFOpenAIGPTPreTrainedModel 和 TFCausalLanguageModelingLoss。
    """

    def __init__(self, config, *inputs, **kwargs):
        """
        初始化方法，接受配置和其他参数，并调用父类初始化方法。
        """
        super().__init__(config, *inputs, **kwargs)
        self.transformer = TFOpenAIGPTMainLayer(config, name="transformer")
        """
        创建了 TFOpenAIGPTMainLayer 对象，作为 transformer 属性。
        OpenAIGPT 模型不支持过去的缓存特性。
        """
        self.supports_xla_generation = False

    def get_output_embeddings(self):
        """
        获取输出嵌入的方法，返回输入嵌入。
        """
        return self.get_input_embeddings()

    def set_output_embeddings(self, value):
        """
        设置输出嵌入的方法，设置输入嵌入的值。
        """
        self.set_input_embeddings(value)

    @unpack_inputs
    @add_start_docstrings_to_model_forward(OPENAI_GPT_INPUTS_DOCSTRING)
    """
    使用装饰器添加了文档字符串，扩展了模型的前向传播函数的输入参数文档说明。
    """
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFCausalLMOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    # 使用装饰器添加代码示例的文档字符串，指定文档检查点、输出类型和配置类

    def call(
        self,
        input_ids: TFModelInputType | None = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        token_type_ids: np.ndarray | tf.Tensor | None = None,
        position_ids: np.ndarray | tf.Tensor | None = None,
        head_mask: np.ndarray | tf.Tensor | None = None,
        inputs_embeds: np.ndarray | tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: np.ndarray | tf.Tensor | None = None,
        training: Optional[bool] = False,
    ) -> Union[Tuple, TFCausalLMOutput]:
        r"""
        labels (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the cross entropy classification loss. Indices should be in `[0, ...,
            config.vocab_size - 1]`.
        """
        # 定义模型的调用方法，接受多个输入参数并返回输出结果

        # 调用 transformer 模型进行前向传播
        transformer_outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )
        # 提取 transformer 的隐藏状态作为输出的第一个元素
        hidden_states = transformer_outputs[0]

        # 对隐藏状态进行线性变换，生成预测的 logits
        logits = self.transformer.tokens_embed(hidden_states, mode="linear")

        # 初始化损失为 None
        loss = None
        # 如果 labels 不为空，则计算损失
        if labels is not None:
            # 将 logits 向左移动一位并截断最后一个 logit token
            shifted_logits = logits[:, :-1]
            labels = labels[:, 1:]
            # 使用预测的 logits 和实际的 labels 计算损失
            loss = self.hf_compute_loss(labels, shifted_logits)

        # 如果 return_dict 为 False，则返回元组形式的输出
        if not return_dict:
            output = (logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        # 如果 return_dict 为 True，则返回 TFCausalLMOutput 对象作为输出
        return TFCausalLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

    # 准备生成的输入数据格式
    def prepare_inputs_for_generation(self, inputs, **kwargs):
        return {"input_ids": inputs}

    # 构建模型
    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 如果模型已经构建，则直接返回
        if getattr(self, "transformer", None) is not None:
            # 在 transformer 的命名空间下构建模型
            with tf.name_scope(self.transformer.name):
                self.transformer.build(None)
@add_start_docstrings(
    """
    OpenAI GPT Model transformer with a language modeling and a multiple-choice classification head on top e.g. for
    RocStories/SWAG tasks. The two heads are two linear layers. The language modeling head has its weights tied to the
    input embeddings, the classification head takes as input the input of a specified classification token index in the
    input sequence).
    """,
    OPENAI_GPT_START_DOCSTRING,
)
class TFOpenAIGPTDoubleHeadsModel(TFOpenAIGPTPreTrainedModel):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        config.num_labels = 1
        self.transformer = TFOpenAIGPTMainLayer(config, name="transformer")
        self.multiple_choice_head = TFSequenceSummary(
            config, initializer_range=config.initializer_range, name="multiple_choice_head"
        )

    @unpack_inputs
    @add_start_docstrings_to_model_forward(OPENAI_GPT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFOpenAIGPTDoubleHeadsModelOutput, config_class=_CONFIG_FOR_DOC)
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        token_type_ids: np.ndarray | tf.Tensor | None = None,
        position_ids: np.ndarray | tf.Tensor | None = None,
        head_mask: np.ndarray | tf.Tensor | None = None,
        inputs_embeds: np.ndarray | tf.Tensor | None = None,
        mc_token_ids: np.ndarray | tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: Optional[bool] = False,
    ):
        """
        Perform the forward pass of the OpenAI GPT model with two heads.

        Args:
            input_ids: Optional[input_ids: tf.TensorSpec((None, None, None), tf.int32, name="input_ids")],
                The input tensor of shape [batch_size, sequence_length].
            attention_mask: Optional[tf.TensorSpec((None, None, None), tf.int32, name="attention_mask")],
                The attention mask tensor of shape [batch_size, sequence_length].
            token_type_ids: Optional[tf.TensorSpec((None, None), tf.int32, name="token_type_ids")],
                The token type ids tensor of shape [batch_size, sequence_length].
            position_ids: Optional[tf.TensorSpec((None, None), tf.int32, name="position_ids")],
                The position ids tensor of shape [batch_size, sequence_length].
            head_mask: Optional[tf.TensorSpec((None, None), tf.float32, name="head_mask")],
                The head mask tensor of shape [num_heads, sequence_length].
            inputs_embeds: Optional[tf.TensorSpec((None, None, None), tf.float32, name="inputs_embeds")],
                The input embeddings tensor of shape [batch_size, sequence_length, hidden_size].
            mc_token_ids: Optional[tf.TensorSpec((None, None), tf.int32, name="mc_token_ids")],
                The multiple choice token ids tensor of shape [batch_size, num_choices].
            output_attentions: Optional[bool],
                Whether to return attentions weights.
            output_hidden_states: Optional[bool],
                Whether to return hidden states.
            return_dict: Optional[bool],
                Whether to return a dictionary instead of a tuple.
            training: Optional[bool],
                Whether in training mode or not.

        Returns:
            TFOpenAIGPTDoubleHeadsModelOutput or tf.Tensor,
            The model output as a named tuple or a tensor.
        """
        pass

    @property
    def input_signature(self):
        """
        Return the input signature for the TensorFlow model.
        """
        return {
            "input_ids": tf.TensorSpec((None, None, None), tf.int32, name="input_ids"),
            "attention_mask": tf.TensorSpec((None, None, None), tf.int32, name="attention_mask"),
            "mc_token_ids": tf.TensorSpec((None, None), tf.int32, name="token_type_ids"),
        }

    def build(self, input_shape=None):
        """
        Build the OpenAI GPT model.

        Args:
            input_shape: Optional, The shape of the input tensor.
        """
        if self.built:
            return
        self.built = True
        if getattr(self, "transformer", None) is not None:
            with tf.name_scope(self.transformer.name):
                self.transformer.build(None)
        if getattr(self, "multiple_choice_head", None) is not None:
            with tf.name_scope(self.multiple_choice_head.name):
                self.multiple_choice_head.build(None)


@add_start_docstrings(
    """
    The OpenAI GPT Model transformer with a sequence classification head on top (linear layer).

    [`TFOpenAIGPTForSequenceClassification`] uses the last token in order to do the classification, as other causal
    models (e.g. GPT-2) do.

    Since it does classification on the last token, it requires to know the position of the last token. If a
    """,
    OPENAI_GPT_START_DOCSTRING,
)
    `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If
    no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
    padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
    each row of the batch).


OPENAI_GPT_START_DOCSTRING,



    OPENAI_GPT_START_DOCSTRING,
    )
    # 定义一个继承自 TFOpenAIGPTPreTrainedModel 和 TFSequenceClassificationLoss 的类
    class TFOpenAIGPTForSequenceClassification(TFOpenAIGPTPreTrainedModel, TFSequenceClassificationLoss):
        # 初始化函数，接受配置参数 config 和额外的输入 *inputs 和 **kwargs
        def __init__(self, config, *inputs, **kwargs):
            # 调用父类的初始化方法
            super().__init__(config, *inputs, **kwargs)
            # 设置类别数目为配置中的 num_labels
            self.num_labels = config.num_labels
            # 创建一个全连接层 Dense 对象用于生成输出分数
            self.score = keras.layers.Dense(
                config.num_labels,
                kernel_initializer=get_initializer(config.initializer_range),
                name="score",
                use_bias=False,
            )
            # 创建一个 OpenAIGPT 主层对象用于处理输入数据
            self.transformer = TFOpenAIGPTMainLayer(config, name="transformer")
            # 保存配置对象到类的属性中
            self.config = config

        @unpack_inputs
        @add_start_docstrings_to_model_forward(OPENAI_GPT_INPUTS_DOCSTRING)
        @add_code_sample_docstrings(
            checkpoint=_CHECKPOINT_FOR_DOC,
            output_type=TFSequenceClassifierOutput,
            config_class=_CONFIG_FOR_DOC,
        )
        # 定义模型的前向传播函数，接受多种输入参数并返回模型输出
        def call(
            self,
            input_ids: TFModelInputType | None = None,
            attention_mask: np.ndarray | tf.Tensor | None = None,
            token_type_ids: np.ndarray | tf.Tensor | None = None,
            position_ids: np.ndarray | tf.Tensor | None = None,
            head_mask: np.ndarray | tf.Tensor | None = None,
            inputs_embeds: np.ndarray | tf.Tensor | None = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            labels: np.ndarray | tf.Tensor | None = None,
            training: Optional[bool] = False,
        ) -> Union[Tuple, TFSequenceClassifierOutput]:
        r"""
        labels (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the cross entropy classification loss. Indices should be in `[0, ...,
            config.vocab_size - 1]`.
        """
        # 调用transformer模型，获取transformer的输出结果
        transformer_outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        # 从transformer输出中获取隐藏状态
        hidden_states = transformer_outputs[0]
        
        # 将隐藏状态输入score层，得到logits
        logits = self.score(hidden_states)
        
        # 初始化in_logits变量
        in_logits = None
        
        # 如果没有定义pad_token_id，则将sequence_lengths设为-1
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            # 如果输入中有input_ids
            if input_ids is not None:
                # 计算每个样本的序列长度
                sequence_lengths = (
                    tf.argmax(tf.cast(tf.math.equal(input_ids, self.config.pad_token_id), input_ids.dtype), axis=-1)
                    - 1
                )
                # 如果长度小于0，则设为默认序列长度-1
                sequence_lengths = tf.where(sequence_lengths >= 0, sequence_lengths, input_ids.shape[-1] - 1)
                # 从logits中根据序列长度取出对应位置的值
                in_logits = tf.gather(logits, sequence_lengths, batch_dims=1, axis=1)
            else:
                sequence_lengths = -1
                # 如果没有input_ids，发出警告
                logger.warning(
                    f"{self.__class__.__name__} will not detect padding tokens in `inputs_embeds`. Results may be "
                    "unexpected if using padding tokens in conjunction with `inputs_embeds.`"
                )
        
        # 初始化loss为None
        loss = None
        
        # 如果提供了labels
        if labels is not None:
            # 根据input_ids或inputs_embeds获取batch_size和sequence_length
            if input_ids is not None:
                batch_size, sequence_length = shape_list(input_ids)[:2]
            else:
                batch_size, sequence_length = shape_list(inputs_embeds)[:2]
            # 检查是否定义了pad_token_id或者batch_size为1，否则报错
            assert (
                self.config.pad_token_id is not None or batch_size == 1
            ), "Cannot handle batch sizes > 1 if no padding token is defined."

            # 如果sequence_lengths不是tensor，则根据batch_size和sequence_lengths取出对应的logits值
            if not tf.is_tensor(sequence_lengths):
                in_logits = logits[0:batch_size, sequence_lengths]

            # 计算损失函数
            loss = self.hf_compute_loss(tf.reshape(labels, [-1, 1]), tf.reshape(in_logits, [-1, self.num_labels]))

        # 如果in_logits不为None，则使用in_logits作为pooled_logits，否则使用logits
        pooled_logits = in_logits if in_logits is not None else logits

        # 如果return_dict为False，则返回输出元组
        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        # 如果return_dict为True，则返回TFSequenceClassifierOutput对象
        return TFSequenceClassifierOutput(
            loss=loss,
            logits=pooled_logits,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )
    # 定义一个方法 `build`，用于构建模型的结构
    def build(self, input_shape=None):
        # 如果已经构建过，直接返回，避免重复构建
        if self.built:
            return
        # 将 built 属性设置为 True，表示模型已构建
        self.built = True
        # 检查是否存在 `score` 属性，并且属性值不为 None
        if getattr(self, "score", None) is not None:
            # 在命名空间 `self.score.name` 下构建 `score` 属性
            with tf.name_scope(self.score.name):
                # 调用 `build` 方法构建 `score`，输入形状为 [None, None, self.config.n_embd]
                self.score.build([None, None, self.config.n_embd])
        # 检查是否存在 `transformer` 属性，并且属性值不为 None
        if getattr(self, "transformer", None) is not None:
            # 在命名空间 `self.transformer.name` 下构建 `transformer` 属性
            with tf.name_scope(self.transformer.name):
                # 调用 `build` 方法构建 `transformer`，输入形状为 None
                self.transformer.build(None)
```