# `.\models\xlm\modeling_tf_xlm.py`

```
# 计算位置编码并将其写入输出张量中
def create_sinusoidal_embeddings(n_pos, dim, out):
    # 生成位置编码矩阵，其中每个位置的编码包括正弦和余弦部分
    position_enc = np.array([[pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)] for pos in range(n_pos)])
    # 将正弦编码写入输出张量的偶数索引位置
    out[:, 0::2] = tf.constant(np.sin(position_enc[:, 0::2]))
    # 将余弦编码写入输出张量的奇数索引位置
    out[:, 1::2] = tf.constant(np.cos(position_enc[:, 1::2]))

# 生成隐藏状态掩码和可选的注意力掩码
def get_masks(slen, lengths, causal, padding_mask=None):
    # 获取批次大小
    bs = shape_list(lengths)[0]
    # 如果存在填充掩码，则使用填充掩码作为掩码
    if padding_mask is not None:
        mask = padding_mask
    else:
        # 如果不是 causal 模式，则创建长度等于 slen 的序列 alen
        alen = tf.range(slen, dtype=lengths.dtype)
        # 创建一个掩码 mask，标记 alen 中小于每个长度值的位置为 True，其余为 False
        mask = alen < tf.expand_dims(lengths, axis=1)

    # 如果是 causal 模式，则创建一个上三角形式的注意力掩码 attn_mask
    # 否则，attn_mask 与 mask 相同
    if causal:
        attn_mask = tf.less_equal(
            tf.tile(tf.reshape(alen, (1, 1, slen)), (bs, slen, 1)), tf.reshape(alen, (1, slen, 1))
        )
    else:
        attn_mask = mask

    # 对掩码 mask 进行形状检查，确保其形状为 [bs, slen]
    tf.debugging.assert_equal(shape_list(mask), [bs, slen])
    # 如果是 causal 模式，则对 attn_mask 进行形状检查，确保其形状为 [bs, slen, slen]
    if causal:
        tf.debugging.assert_equal(shape_list(attn_mask), [bs, slen, slen])

    # 返回 mask 和 attn_mask 作为结果
    return mask, attn_mask
class TFXLMMultiHeadAttention(keras.layers.Layer):
    # 类变量，用于生成唯一的层标识符
    NEW_ID = itertools.count()

    def __init__(self, n_heads, dim, config, **kwargs):
        super().__init__(**kwargs)
        # 分配当前层的唯一标识符
        self.layer_id = next(TFXLMMultiHeadAttention.NEW_ID)
        self.dim = dim  # 设置注意力机制的维度
        self.n_heads = n_heads  # 设置注意力头的数量
        self.output_attentions = config.output_attentions  # 是否输出注意力权重
        assert self.dim % self.n_heads == 0  # 断言：确保维度可以被注意力头数量整除

        # 定义用于查询、键、值的线性层，并初始化
        self.q_lin = keras.layers.Dense(dim, kernel_initializer=get_initializer(config.init_std), name="q_lin")
        self.k_lin = keras.layers.Dense(dim, kernel_initializer=get_initializer(config.init_std), name="k_lin")
        self.v_lin = keras.layers.Dense(dim, kernel_initializer=get_initializer(config.init_std), name="v_lin")
        self.out_lin = keras.layers.Dense(dim, kernel_initializer=get_initializer(config.init_std), name="out_lin")
        self.dropout = keras.layers.Dropout(config.attention_dropout)  # 定义注意力机制的dropout层
        self.pruned_heads = set()  # 初始化一个空集合，用于记录被修剪的注意力头
        self.dim = dim  # 更新维度信息

    def prune_heads(self, heads):
        raise NotImplementedError  # 剪枝注意力头的方法，目前未实现

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 构建查询、键、值以及输出线性层的神经网络结构
        if getattr(self, "q_lin", None) is not None:
            with tf.name_scope(self.q_lin.name):
                self.q_lin.build([None, None, self.dim])
        if getattr(self, "k_lin", None) is not None:
            with tf.name_scope(self.k_lin.name):
                self.k_lin.build([None, None, self.dim])
        if getattr(self, "v_lin", None) is not None:
            with tf.name_scope(self.v_lin.name):
                self.v_lin.build([None, None, self.dim])
        if getattr(self, "out_lin", None) is not None:
            with tf.name_scope(self.out_lin.name):
                self.out_lin.build([None, None, self.dim])


class TFXLMTransformerFFN(keras.layers.Layer):
    def __init__(self, in_dim, dim_hidden, out_dim, config, **kwargs):
        super().__init__(**kwargs)

        # 定义前馈神经网络的两个线性层，并初始化
        self.lin1 = keras.layers.Dense(dim_hidden, kernel_initializer=get_initializer(config.init_std), name="lin1")
        self.lin2 = keras.layers.Dense(out_dim, kernel_initializer=get_initializer(config.init_std), name="lin2")
        # 根据配置选择激活函数（GELU或ReLU）
        self.act = get_tf_activation("gelu") if config.gelu_activation else get_tf_activation("relu")
        self.dropout = keras.layers.Dropout(config.dropout)  # 定义前馈神经网络的dropout层
        self.in_dim = in_dim  # 输入维度
        self.dim_hidden = dim_hidden  # 隐藏层维度

    def call(self, input, training=False):
        # 前向传播函数
        x = self.lin1(input)  # 第一层线性变换
        x = self.act(x)  # 应用激活函数
        x = self.lin2(x)  # 第二层线性变换
        x = self.dropout(x, training=training)  # 应用dropout层

        return x  # 返回前向传播的结果
    # 如果模型已经建立，直接返回，不进行重复建立
    if self.built:
        return
    # 设置模型状态为已建立
    self.built = True

    # 如果存在名为lin1的属性，并且不为None，则进行下面的操作
    if getattr(self, "lin1", None) is not None:
        # 使用tf.name_scope为self.lin1的操作定义命名空间
        with tf.name_scope(self.lin1.name):
            # 使用self.in_dim作为输入维度，构建self.lin1层
            self.lin1.build([None, None, self.in_dim])

    # 如果存在名为lin2的属性，并且不为None，则进行下面的操作
    if getattr(self, "lin2", None) is not None:
        # 使用tf.name_scope为self.lin2的操作定义命名空间
        with tf.name_scope(self.lin2.name):
            # 使用self.dim_hidden作为隐藏层维度，构建self.lin2层
            self.lin2.build([None, None, self.dim_hidden])
# 将类 TFXLMMainLayer 声明为可序列化的 Keras 层
@keras_serializable
class TFXLMMainLayer(keras.layers.Layer):
    # 使用 XLMConfig 类作为配置类
    config_class = XLMConfig

    # 构建层的方法，初始化位置嵌入等权重
    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回
        if self.built:
            return
        # 标记此层为已构建
        self.built = True
        # 在命名作用域 "position_embeddings" 下添加位置嵌入权重
        with tf.name_scope("position_embeddings"):
            self.position_embeddings = self.add_weight(
                name="embeddings",
                shape=[self.max_position_embeddings, self.dim],
                initializer=get_initializer(self.embed_init_std),
            )

        # 如果有多于一个语言并且使用语言嵌入，则添加语言嵌入权重
        if self.n_langs > 1 and self.use_lang_emb:
            with tf.name_scope("lang_embeddings"):
                self.lang_embeddings = self.add_weight(
                    name="embeddings",
                    shape=[self.n_langs, self.dim],
                    initializer=get_initializer(self.embed_init_std),
                )
        
        # 如果存在 embeddings 属性，则对其进行构建
        if getattr(self, "embeddings", None) is not None:
            with tf.name_scope(self.embeddings.name):
                self.embeddings.build(None)
        
        # 如果存在 layer_norm_emb 属性，则对其进行构建
        if getattr(self, "layer_norm_emb", None) is not None:
            with tf.name_scope(self.layer_norm_emb.name):
                self.layer_norm_emb.build([None, None, self.dim])
        
        # 对每个自注意力层进行构建
        for layer in self.attentions:
            with tf.name_scope(layer.name):
                layer.build(None)
        
        # 对每个 LayerNorm 层进行构建
        for layer in self.layer_norm1:
            with tf.name_scope(layer.name):
                layer.build([None, None, self.dim])
        
        # 对每个前馈神经网络层进行构建
        for layer in self.ffns:
            with tf.name_scope(layer.name):
                layer.build(None)
        
        # 对每个 LayerNorm 层进行构建
        for layer in self.layer_norm2:
            with tf.name_scope(layer.name):
                layer.build([None, None, self.dim])

    # 获取输入嵌入的方法
    def get_input_embeddings(self):
        return self.embeddings

    # 设置输入嵌入的方法
    def set_input_embeddings(self, value):
        self.embeddings.weight = value
        self.embeddings.vocab_size = shape_list(value)[0]

    # 剪枝模型中的注意力头部的方法
    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        raise NotImplementedError

    # 解包输入参数并调用模型的方法
    @unpack_inputs
    def call(
        self,
        input_ids=None,
        attention_mask=None,
        langs=None,
        token_type_ids=None,
        position_ids=None,
        lengths=None,
        cache=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        training=False,



        ):
        # 这里会实际执行模型的计算过程，在此不做具体注释，具体操作会依赖模型的实现细节
        pass

# TFXLMPreTrainedModel 类，继承自 TFPreTrainedModel，用于处理模型权重初始化以及预训练模型下载和加载的抽象类
class TFXLMPreTrainedModel(TFPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 使用 XLMConfig 类作为配置类
    config_class = XLMConfig
    # 基础模型名称前缀为 "transformer"
    base_model_prefix = "transformer"

    # 属性装饰器，用于返回输入嵌入
    @property
    def dummy_inputs(self):
        # 定义一个包含了一些假输入数据的函数
        # 创建包含输入数据的张量列表，数据类型为整型
        inputs_list = tf.constant([[7, 6, 0, 0, 1], [1, 2, 3, 0, 0], [0, 0, 0, 4, 5]], dtype=tf.int32)
        # 创建包含注意力掩码数据的张量列表，数据类型为整型
        attns_list = tf.constant([[1, 1, 0, 0, 1], [1, 1, 1, 0, 0], [1, 0, 0, 1, 1]], dtype=tf.int32)
        # 如果需要使用语言嵌入并且有多个语言，则返回包含输入、注意力掩码和语言信息的字典
        if self.config.use_lang_emb and self.config.n_langs > 1:
            return {
                "input_ids": inputs_list,
                "attention_mask": attns_list,
                "langs": tf.constant([[1, 1, 0, 0, 1], [1, 1, 1, 0, 0], [1, 0, 0, 1, 1]], dtype=tf.int32),
            }
        # 如果不需要使用语言嵌入或者只有一种语言，则返回包含输入和注意力掩码信息的字典
        else:
            return {"input_ids": inputs_list, "attention_mask": attns_list}
# 当 XLMWithLMHead 计算损失类似于其他语言模型时移除
@dataclass
class TFXLMWithLMHeadModelOutput(ModelOutput):
    """
    [`TFXLMWithLMHeadModel`] 输出的基类。

    Args:
        logits (`tf.Tensor`，形状为 `(batch_size, sequence_length, config.vocab_size)`):
            语言建模头部的预测分数（SoftMax 之前的每个词汇标记的分数）。
        hidden_states (`tuple(tf.Tensor)`，*可选*，当传入 `output_hidden_states=True` 或 `config.output_hidden_states=True` 时返回):
            形状为 `(batch_size, sequence_length, hidden_size)` 的 `tf.Tensor` 元组。
            
            模型在每层输出的隐藏状态以及初始嵌入输出。
        attentions (`tuple(tf.Tensor)`，*可选*，当传入 `output_attentions=True` 或 `config.output_attentions=True` 时返回):
            形状为 `(batch_size, num_heads, sequence_length, sequence_length)` 的 `tf.Tensor` 元组。
            
            注意力 SoftMax 之后的注意力权重，用于计算自注意力头中的加权平均值。
    """

    logits: tf.Tensor = None
    hidden_states: Tuple[tf.Tensor, ...] | None = None
    attentions: Tuple[tf.Tensor, ...] | None = None


XLM_START_DOCSTRING = r"""

    该模型继承自 [`TFPreTrainedModel`]。请查阅超类文档以获取库实现的所有模型通用方法（如下载或保存模型、调整输入嵌入大小、修剪头等）。

    该模型也是 [keras.Model](https://www.tensorflow.org/api_docs/python/tf/keras/Model) 的子类。可以将其用作常规的 TF 2.0 Keras 模型，并参考 TF 2.0 文档以获取与一般使用和行为相关的所有内容。

    <Tip>

    `transformers` 中的 TensorFlow 模型和层接受两种输入格式：

    - 将所有输入作为关键字参数传递（类似于 PyTorch 模型），或者
    - 将所有输入作为列表、元组或字典传递给第一个位置参数。

    支持第二种格式的原因是 Keras 方法更喜欢在将输入传递给模型和层时使用此格式。由于此支持，当使用诸如 `model.fit()` 等方法时，您只需将输入和标签以 `model.fit()` 支持的任何格式传递即可！但是，如果您希望在 Keras 方法之外（例如在使用 Keras `Functional` API 创建自己的层或模型时）使用第二种格式，那么您可以使用以下三种可能性将所有输入张量收集到第一个位置参数中：

    - 仅具有 `input_ids` 的单个张量且没有其他内容：`model(input_ids)`
    - 长度不同的列表，其中按照文档字符串中给定的顺序包含一个或多个输入张量：
    `model([input_ids, attention_mask])` or `model([input_ids, attention_mask, token_type_ids])`
    - 根据不同的输入情况，可以接受一个或多个输入张量的字典。
    `model({"input_ids": input_ids, "token_type_ids": token_type_ids})`

    Note that when creating models and layers with
    [subclassing](https://keras.io/guides/making_new_layers_and_models_via_subclassing/) then you don't need to worry
    about any of this, as you can just pass inputs like you would to any other Python function!
    在使用子类化创建模型和层时，可以像传递给任何其他Python函数一样传递输入，因此无需担心这些细节。

    Parameters:
        config ([`XLMConfig`]): Model configuration class with all the parameters of the model.
            使用包含模型所有参数的配置类（例如`XLMConfig`）进行模型的配置。
            使用配置文件进行初始化不会加载与模型关联的权重，仅加载配置。
            可以查看 [`~PreTrainedModel.from_pretrained`] 方法来加载模型的权重。
"""

XLM_INPUTS_DOCSTRING = r"""
"""


@add_start_docstrings(
    "The bare XLM Model transformer outputting raw hidden-states without any specific head on top.",
    XLM_START_DOCSTRING,
)
class TFXLMModel(TFXLMPreTrainedModel):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        # 初始化 transformer 层，使用配置对象创建 TFXLMMainLayer，并命名为 "transformer"
        self.transformer = TFXLMMainLayer(config, name="transformer")

    @unpack_inputs
    @add_start_docstrings_to_model_forward(XLM_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFBaseModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        attention_mask: tf.Tensor | None = None,
        langs: tf.Tensor | None = None,
        token_type_ids: tf.Tensor | None = None,
        position_ids: tf.Tensor | None = None,
        lengths: tf.Tensor | None = None,
        cache: Dict[str, tf.Tensor] | None = None,
        head_mask: tf.Tensor | None = None,
        inputs_embeds: tf.Tensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        training: bool = False,
    ) -> TFBaseModelOutput | Tuple[tf.Tensor]:
        # 调用 transformer 层的前向传播函数，传入各种输入参数
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            langs=langs,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            lengths=lengths,
            cache=cache,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
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
        if getattr(self, "transformer", None) is not None:
            with tf.name_scope(self.transformer.name):
                # 构建 transformer 层
                self.transformer.build(None)


class TFXLMPredLayer(keras.layers.Layer):
    """
    Prediction layer (cross_entropy or adaptive_softmax).
    """

    def __init__(self, config, input_embeddings, **kwargs):
        super().__init__(**kwargs)

        # 初始化预测层参数
        self.asm = config.asm
        self.n_words = config.n_words
        self.pad_index = config.pad_index

        if config.asm is False:
            # 如果不使用自适应 softmax，直接使用输入的嵌入层
            self.input_embeddings = input_embeddings
        else:
            # 抛出未实现的错误，因为自适应 softmax 模块未实现
            raise NotImplementedError
            # self.proj = nn.AdaptiveLogSoftmaxWithLoss(
            #     in_features=dim,
            #     n_classes=config.n_words,
            #     cutoffs=config.asm_cutoffs,
            #     div_value=config.asm_div_value,
            #     head_bias=True,  # default is False
            # )
    # 在神经网络层的构建方法中，初始化一个偏置项，每个标记对应一个输出偏置。
    self.bias = self.add_weight(shape=(self.n_words,), initializer="zeros", trainable=True, name="bias")

    # 调用父类的构建方法
    super().build(input_shape)

# 获取输出的嵌入向量
def get_output_embeddings(self):
    return self.input_embeddings

# 设置输出的嵌入向量
def set_output_embeddings(self, value):
    # 更新输入嵌入的权重值
    self.input_embeddings.weight = value
    # 更新输入嵌入的词汇量大小
    self.input_embeddings.vocab_size = shape_list(value)[0]

# 获取偏置项
def get_bias(self):
    return {"bias": self.bias}

# 设置偏置项
def set_bias(self, value):
    # 更新偏置项的值
    self.bias = value["bias"]
    # 更新词汇量的大小
    self.vocab_size = shape_list(value["bias"])[0]

# 神经网络的调用方法，接受隐藏状态作为输入
def call(self, hidden_states):
    # 使用输入嵌入对隐藏状态进行线性变换
    hidden_states = self.input_embeddings(hidden_states, mode="linear")
    # 添加偏置项到隐藏状态中
    hidden_states = hidden_states + self.bias

    # 返回处理后的隐藏状态
    return hidden_states
"""
The XLM Model transformer with a language modeling head on top (linear layer with weights tied to the input
embeddings).
"""
# 引入装饰器和必要的模块
@add_start_docstrings(
    """
    The XLM Model transformer with a language modeling head on top (linear layer with weights tied to the input
    embeddings).
    """,
    XLM_START_DOCSTRING,  # 引用 XLM 的起始文档字符串
)
# TFXLMWithLMHeadModel 类定义，继承自 TFXLMPreTrainedModel
class TFXLMWithLMHeadModel(TFXLMPreTrainedModel):
    
    # 初始化方法
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)  # 调用父类的初始化方法
        self.transformer = TFXLMMainLayer(config, name="transformer")  # 创建 XLM 主层对象
        self.pred_layer = TFXLMPredLayer(config, self.transformer.embeddings, name="pred_layer_._proj")
        # 创建 XLM 预测层对象，并连接到嵌入层
        
        # XLM 不支持过去的缓存特性
        self.supports_xla_generation = False

    # 获取语言模型头部
    def get_lm_head(self):
        return self.pred_layer

    # 获取前缀偏置名称（已弃用）
    def get_prefix_bias_name(self):
        warnings.warn("The method get_prefix_bias_name is deprecated. Please use `get_bias` instead.", FutureWarning)
        return self.name + "/" + self.pred_layer.name

    # 为生成准备输入
    def prepare_inputs_for_generation(self, inputs, **kwargs):
        mask_token_id = self.config.mask_token_id  # 获取掩码标记 ID
        lang_id = self.config.lang_id  # 获取语言 ID

        effective_batch_size = inputs.shape[0]  # 计算有效批次大小
        mask_token = tf.fill((effective_batch_size, 1), 1) * mask_token_id  # 创建掩码令牌
        inputs = tf.concat([inputs, mask_token], axis=1)  # 将掩码令牌连接到输入末尾

        if lang_id is not None:
            langs = tf.ones_like(inputs) * lang_id  # 如果存在语言 ID，创建相应的语言张量
        else:
            langs = None
        
        # 返回输入字典，包含输入 ID 和语言信息
        return {"input_ids": inputs, "langs": langs}

    # 调用方法，处理各种输入和选项
    @unpack_inputs
    @add_start_docstrings_to_model_forward(XLM_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,  # 示例使用的检查点
        output_type=TFXLMWithLMHeadModelOutput,  # 输出类型
        config_class=_CONFIG_FOR_DOC,  # 配置类
    )
    def call(
        self,
        input_ids: TFModelInputType | None = None,  # 输入 ID
        attention_mask: np.ndarray | tf.Tensor | None = None,  # 注意力掩码
        langs: np.ndarray | tf.Tensor | None = None,  # 语言标识
        token_type_ids: np.ndarray | tf.Tensor | None = None,  # 令牌类型 ID
        position_ids: np.ndarray | tf.Tensor | None = None,  # 位置 ID
        lengths: np.ndarray | tf.Tensor | None = None,  # 序列长度
        cache: Optional[Dict[str, tf.Tensor]] = None,  # 缓存
        head_mask: np.ndarray | tf.Tensor | None = None,  # 头部掩码
        inputs_embeds: np.ndarray | tf.Tensor | None = None,  # 嵌入输入
        output_attentions: Optional[bool] = None,  # 输出注意力
        output_hidden_states: Optional[bool] = None,  # 输出隐藏状态
        return_dict: Optional[bool] = None,  # 返回字典
        training: bool = False,  # 是否训练模式
        ):
    # 定义函数的返回类型为 TFXLMWithLMHeadModelOutput 或者 (TFXLMWithLMHeadModelOutput, tf.Tensor) 的元组
    -> Union[TFXLMWithLMHeadModelOutput, Tuple[tf.Tensor]]:
        # 调用 self.transformer 对象的 __call__ 方法，传入各种输入参数
        transformer_outputs = self.transformer(
            input_ids=input_ids,                  # 输入的 token IDs
            attention_mask=attention_mask,        # 注意力遮罩
            langs=langs,                          # 语言 ID（如果模型支持多语言）
            token_type_ids=token_type_ids,        # token 类型 IDs（如果模型支持）
            position_ids=position_ids,            # 位置 IDs
            lengths=lengths,                      # 序列长度
            cache=cache,                          # 缓存（如果有）
            head_mask=head_mask,                  # 头部遮罩（如果有）
            inputs_embeds=inputs_embeds,          # 嵌入的输入（如果有）
            output_attentions=output_attentions,  # 是否输出注意力权重
            output_hidden_states=output_hidden_states,  # 是否输出隐藏状态
            return_dict=return_dict,              # 是否以字典形式返回输出
            training=training,                    # 是否处于训练模式
        )

        # 从 transformer_outputs 中取出第一个元素，通常是模型的输出
        output = transformer_outputs[0]
        # 将模型输出通过 self.pred_layer 进行预测
        outputs = self.pred_layer(output)

        # 如果 return_dict 为 False，则返回一个元组，包含 outputs 和 transformer_outputs 的其余部分
        if not return_dict:
            return (outputs,) + transformer_outputs[1:]

        # 如果 return_dict 为 True，则构造一个 TFXLMWithLMHeadModelOutput 对象，并返回
        return TFXLMWithLMHeadModelOutput(
            logits=outputs,                              # 预测的逻辑回归输出
            hidden_states=transformer_outputs.hidden_states,  # 隐藏状态列表
            attentions=transformer_outputs.attentions     # 注意力权重列表
        )

    # 构建模型，在此处初始化所有子层的权重
    def build(self, input_shape=None):
        # 如果模型已经构建好，直接返回
        if self.built:
            return
        # 标记模型为已构建状态
        self.built = True

        # 如果 self.transformer 存在，则在名为 self.transformer.name 的作用域内构建它
        if getattr(self, "transformer", None) is not None:
            with tf.name_scope(self.transformer.name):
                self.transformer.build(None)

        # 如果 self.pred_layer 存在，则在名为 self.pred_layer.name 的作用域内构建它
        if getattr(self, "pred_layer", None) is not None:
            with tf.name_scope(self.pred_layer.name):
                self.pred_layer.build(None)
"""
XLM Model with a sequence classification/regression head on top (a linear layer on top of the pooled output) e.g.
for GLUE tasks.
"""
# 继承自 TFXLMPreTrainedModel 和 TFSequenceClassificationLoss 的 XLM 模型，用于序列分类或回归任务
class TFXLMForSequenceClassification(TFXLMPreTrainedModel, TFSequenceClassificationLoss):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        # 初始化时设置类别数量
        self.num_labels = config.num_labels

        # 创建 XLM 主层对象，命名为 "transformer"
        self.transformer = TFXLMMainLayer(config, name="transformer")
        
        # 创建序列摘要对象，用于生成序列摘要特征
        self.sequence_summary = TFSequenceSummary(config, initializer_range=config.init_std, name="sequence_summary")

    @unpack_inputs
    # 将输入解包并添加关于模型前向传播的文档字符串，展示输入的批次大小和序列长度
    @add_start_docstrings_to_model_forward(XLM_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    # 添加关于模型前向传播的代码示例的文档字符串，展示相关的检查点、输出类型和配置类
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFSequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        langs: np.ndarray | tf.Tensor | None = None,
        token_type_ids: np.ndarray | tf.Tensor | None = None,
        position_ids: np.ndarray | tf.Tensor | None = None,
        lengths: np.ndarray | tf.Tensor | None = None,
        cache: Optional[Dict[str, tf.Tensor]] = None,
        head_mask: np.ndarray | tf.Tensor | None = None,
        inputs_embeds: np.ndarray | tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: np.ndarray | tf.Tensor | None = None,
        training: bool = False,
        # ...
    ) -> Union[TFSequenceClassifierOutput, Tuple[tf.Tensor]]:
        r"""
        labels (`tf.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        # 调用transformer模型进行前向传播，传入各种参数
        transformer_outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            langs=langs,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            lengths=lengths,
            cache=cache,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )
        # 获取transformer的输出作为序列总结的输入
        output = transformer_outputs[0]

        # 通过序列总结模型得到最终的logits
        logits = self.sequence_summary(output)

        # 如果提供了labels，则计算损失函数（交叉熵或均方损失）
        loss = None if labels is None else self.hf_compute_loss(labels, logits)

        # 如果不需要返回字典，则按照tuple的形式返回输出
        if not return_dict:
            output = (logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        # 如果需要返回字典，则构造TFSequenceClassifierOutput对象返回
        return TFSequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

    def build(self, input_shape=None):
        # 如果已经构建过，直接返回
        if self.built:
            return
        self.built = True
        # 如果transformer模型存在，则在命名空间下构建transformer模型
        if getattr(self, "transformer", None) is not None:
            with tf.name_scope(self.transformer.name):
                self.transformer.build(None)
        # 如果序列总结模型存在，则在命名空间下构建序列总结模型
        if getattr(self, "sequence_summary", None) is not None:
            with tf.name_scope(self.sequence_summary.name):
                self.sequence_summary.build(None)
"""
XLM Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
softmax) e.g. for RocStories/SWAG tasks.
"""
# 定义了一个 XLM 模型，该模型在其顶部添加了一个多选分类头部，包括一个线性层和一个 softmax 层，用于例如 RocStories/SWAG 任务。

class TFXLMForMultipleChoice(TFXLMPreTrainedModel, TFMultipleChoiceLoss):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        # 初始化 XLM 主层，命名为 "transformer"
        self.transformer = TFXLMMainLayer(config, name="transformer")
        
        # 初始化序列摘要层，用于生成序列摘要，命名为 "sequence_summary"
        self.sequence_summary = TFSequenceSummary(config, initializer_range=config.init_std, name="sequence_summary")
        
        # 初始化 logits 投影层，一个全连接层用于多选分类任务，输出维度为 1
        self.logits_proj = keras.layers.Dense(
            1, kernel_initializer=get_initializer(config.initializer_range), name="logits_proj"
        )
        
        # 保存配置对象
        self.config = config

    @property
    def dummy_inputs(self):
        """
        Dummy inputs to build the network.

        Returns:
            tf.Tensor with dummy inputs
        """
        # 如果配置要求使用语言嵌入且语言数大于1，则返回包含 "input_ids" 和 "langs" 的虚拟输入
        if self.config.use_lang_emb and self.config.n_langs > 1:
            return {
                "input_ids": tf.constant(MULTIPLE_CHOICE_DUMMY_INPUTS, dtype=tf.int32),
                "langs": tf.constant(MULTIPLE_CHOICE_DUMMY_INPUTS, dtype=tf.int32),
            }
        else:
            # 否则，只返回包含 "input_ids" 的虚拟输入
            return {
                "input_ids": tf.constant(MULTIPLE_CHOICE_DUMMY_INPUTS, dtype=tf.int32),
            }

    @unpack_inputs
    @add_start_docstrings_to_model_forward(XLM_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFMultipleChoiceModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        langs: np.ndarray | tf.Tensor | None = None,
        token_type_ids: np.ndarray | tf.Tensor | None = None,
        position_ids: np.ndarray | tf.Tensor | None = None,
        lengths: np.ndarray | tf.Tensor | None = None,
        cache: Optional[Dict[str, tf.Tensor]] = None,
        head_mask: np.ndarray | tf.Tensor | None = None,
        inputs_embeds: np.ndarray | tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: np.ndarray | tf.Tensor | None = None,
        training: bool = False,
        # 神经网络前向传播方法，接收并处理各种输入和配置参数，生成预测或特征输出
    # 定义方法签名，指定输入参数和返回类型
    ) -> Union[TFMultipleChoiceModelOutput, Tuple[tf.Tensor]]:
        # 如果输入参数 input_ids 不为 None，则获取其第二维和第三维的大小
        if input_ids is not None:
            num_choices = shape_list(input_ids)[1]  # 获取选项数量
            seq_length = shape_list(input_ids)[2]   # 获取序列长度
        else:
            # 否则，获取 inputs_embeds 的第二维和第三维的大小
            num_choices = shape_list(inputs_embeds)[1]  # 获取选项数量
            seq_length = shape_list(inputs_embeds)[2]   # 获取序列长度

        # 将 input_ids 重新 reshape 成 (-1, seq_length)，如果 input_ids 不为 None
        flat_input_ids = tf.reshape(input_ids, (-1, seq_length)) if input_ids is not None else None
        # 将 attention_mask 重新 reshape 成 (-1, seq_length)，如果 attention_mask 不为 None
        flat_attention_mask = tf.reshape(attention_mask, (-1, seq_length)) if attention_mask is not None else None
        # 将 token_type_ids 重新 reshape 成 (-1, seq_length)，如果 token_type_ids 不为 None
        flat_token_type_ids = tf.reshape(token_type_ids, (-1, seq_length)) if token_type_ids is not None else None
        # 将 position_ids 重新 reshape 成 (-1, seq_length)，如果 position_ids 不为 None
        flat_position_ids = tf.reshape(position_ids, (-1, seq_length)) if position_ids is not None else None
        # 将 langs 重新 reshape 成 (-1, seq_length)，如果 langs 不为 None
        flat_langs = tf.reshape(langs, (-1, seq_length)) if langs is not None else None
        # 将 inputs_embeds 重新 reshape 成 (-1, seq_length, shape_list(inputs_embeds)[3])，如果 inputs_embeds 不为 None
        flat_inputs_embeds = (
            tf.reshape(inputs_embeds, (-1, seq_length, shape_list(inputs_embeds)[3]))
            if inputs_embeds is not None
            else None
        )

        # 如果 lengths 不为 None，则发出警告并将其设为 None，因为 XLM 多选模型不能使用 lengths 参数
        if lengths is not None:
            logger.warning(
                "The `lengths` parameter cannot be used with the XLM multiple choice models. Please use the "
                "attention mask instead.",
            )
            lengths = None

        # 调用 self.transformer 方法进行转换器的前向计算
        transformer_outputs = self.transformer(
            flat_input_ids,
            flat_attention_mask,
            flat_langs,
            flat_token_type_ids,
            flat_position_ids,
            lengths,
            cache,
            head_mask,
            flat_inputs_embeds,
            output_attentions,
            output_hidden_states,
            return_dict=return_dict,
            training=training,
        )
        # 获取 transformer 输出的第一个元素作为输出
        output = transformer_outputs[0]
        # 对输出进行序列总结，得到 logits
        logits = self.sequence_summary(output)
        # 对 logits 进行投影处理
        logits = self.logits_proj(logits)
        # 将 logits 重新 reshape 成 (-1, num_choices)，以符合多选题的格式
        reshaped_logits = tf.reshape(logits, (-1, num_choices))

        # 如果 labels 为 None，则损失为 None；否则，调用 self.hf_compute_loss 方法计算损失
        loss = None if labels is None else self.hf_compute_loss(labels, reshaped_logits)

        # 如果不需要返回字典形式的输出，则重新组织 output，并在其前面加上损失值（如果有）
        if not return_dict:
            output = (reshaped_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        # 返回 TFMultipleChoiceModelOutput 对象，包括损失、logits、隐藏状态和注意力权重
        return TFMultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )
    # 构建函数，用于构建神经网络层次结构
    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回，避免重复构建
        if self.built:
            return
        # 将标记位设置为已构建
        self.built = True
        
        # 如果存在变换器(transformer)对象，则构建其内部结构
        if getattr(self, "transformer", None) is not None:
            # 在 TensorFlow 中使用名称作用域，指定变换器的名称范围
            with tf.name_scope(self.transformer.name):
                self.transformer.build(None)
        
        # 如果存在序列摘要(sequence_summary)对象，则构建其内部结构
        if getattr(self, "sequence_summary", None) is not None:
            # 在 TensorFlow 中使用名称作用域，指定序列摘要的名称范围
            with tf.name_scope(self.sequence_summary.name):
                self.sequence_summary.build(None)
        
        # 如果存在分类投影(logits_proj)对象，则构建其内部结构
        if getattr(self, "logits_proj", None) is not None:
            # 在 TensorFlow 中使用名称作用域，指定分类投影的名称范围
            with tf.name_scope(self.logits_proj.name):
                # 构建分类投影层，输入形状为 [None, None, self.config.num_labels]
                self.logits_proj.build([None, None, self.config.num_labels])
@add_start_docstrings(
    """
    XLM Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
    Named-Entity-Recognition (NER) tasks.
    """,
    XLM_START_DOCSTRING,
)
class TFXLMForTokenClassification(TFXLMPreTrainedModel, TFTokenClassificationLoss):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.num_labels = config.num_labels  # 从配置中获取标签的数量

        self.transformer = TFXLMMainLayer(config, name="transformer")  # 初始化 XLM 主层模型
        self.dropout = keras.layers.Dropout(config.dropout)  # 设置 dropout 层，用于正则化
        self.classifier = keras.layers.Dense(
            config.num_labels, kernel_initializer=get_initializer(config.init_std), name="classifier"
        )  # 设置分类器，用于输出标签的预测结果
        self.config = config  # 存储配置信息

    @unpack_inputs
    @add_start_docstrings_to_model_forward(XLM_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFTokenClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(
        self,
        input_ids: TFModelInputType | None = None,  # 输入的 token IDs
        attention_mask: np.ndarray | tf.Tensor | None = None,  # 注意力掩码，控制哪些位置需要注意
        langs: np.ndarray | tf.Tensor | None = None,  # 语言 ID 或者语言掩码
        token_type_ids: np.ndarray | tf.Tensor | None = None,  # token 类型 IDs，用于区分 segment
        position_ids: np.ndarray | tf.Tensor | None = None,  # 位置 IDs，用于标记 token 在句子中的位置
        lengths: np.ndarray | tf.Tensor | None = None,  # 句子长度
        cache: Optional[Dict[str, tf.Tensor]] = None,  # 缓存用于加速推断
        head_mask: np.ndarray | tf.Tensor | None = None,  # 头部掩码，用于控制哪些头部需要注意
        inputs_embeds: np.ndarray | tf.Tensor | None = None,  # 输入的嵌入表示
        output_attentions: Optional[bool] = None,  # 是否输出注意力权重
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态
        return_dict: Optional[bool] = None,  # 是否返回字典格式的输出
        labels: np.ndarray | tf.Tensor | None = None,  # 真实的标签
        training: bool = False,  # 是否处于训练模式
    ) -> Union[TFTokenClassifierOutput, Tuple[tf.Tensor]]:
        r"""
        labels (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        # 调用 Transformer 模型来处理输入数据，获取转换后的输出
        transformer_outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            langs=langs,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            lengths=lengths,
            cache=cache,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )
        # 从 Transformer 输出中获取序列输出
        sequence_output = transformer_outputs[0]

        # 对序列输出应用 dropout，用于防止过拟合
        sequence_output = self.dropout(sequence_output, training=training)
        # 将 dropout 后的输出送入分类器，得到分类 logits
        logits = self.classifier(sequence_output)

        # 如果提供了 labels，则计算损失；否则损失设为 None
        loss = None if labels is None else self.hf_compute_loss(labels, logits)

        # 如果不要求返回字典形式的输出，则按需返回不同的输出格式
        if not return_dict:
            output = (logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        # 返回 TFTokenClassifierOutput 对象，包含损失、logits、隐藏状态和注意力权重
        return TFTokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

    def build(self, input_shape=None):
        # 如果模型已经构建过，直接返回
        if self.built:
            return
        # 标记模型已构建
        self.built = True
        # 如果 Transformer 存在，则构建 Transformer 模型
        if getattr(self, "transformer", None) is not None:
            with tf.name_scope(self.transformer.name):
                self.transformer.build(None)
        # 如果分类器存在，则构建分类器模型
        if getattr(self, "classifier", None) is not None:
            with tf.name_scope(self.classifier.name):
                # 构建分类器，设置输入形状为 [None, None, self.config.hidden_size]
                self.classifier.build([None, None, self.config.hidden_size])
@add_start_docstrings(
    """
    XLM Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear layer
    on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    XLM_START_DOCSTRING,
)
class TFXLMForQuestionAnsweringSimple(TFXLMPreTrainedModel, TFQuestionAnsweringLoss):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        # 初始化一个 XLM 主模型层，命名为 "transformer"
        self.transformer = TFXLMMainLayer(config, name="transformer")
        # 初始化一个全连接层用于问题-答案抽取任务的输出，参数由配置文件中的标签数量决定
        self.qa_outputs = keras.layers.Dense(
            config.num_labels, kernel_initializer=get_initializer(config.init_std), name="qa_outputs"
        )
        # 将配置信息保存到对象属性中
        self.config = config

    @unpack_inputs
    @add_start_docstrings_to_model_forward(XLM_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFQuestionAnsweringModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        langs: np.ndarray | tf.Tensor | None = None,
        token_type_ids: np.ndarray | tf.Tensor | None = None,
        position_ids: np.ndarray | tf.Tensor | None = None,
        lengths: np.ndarray | tf.Tensor | None = None,
        cache: Optional[Dict[str, tf.Tensor]] = None,
        head_mask: np.ndarray | tf.Tensor | None = None,
        inputs_embeds: np.ndarray | tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        start_positions: np.ndarray | tf.Tensor | None = None,
        end_positions: np.ndarray | tf.Tensor | None = None,
        training: bool = False,
    ) -> Union[TFQuestionAnsweringModelOutput, Tuple[tf.Tensor]]:
        r"""
        start_positions (`tf.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`tf.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        """
        transformer_outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            langs=langs,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            lengths=lengths,
            cache=cache,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )
        # 从transformer模型获得输出序列
        sequence_output = transformer_outputs[0]

        # 将序列输出通过问答输出层获得logits
        logits = self.qa_outputs(sequence_output)
        
        # 将logits按照最后一个维度分割为start_logits和end_logits
        start_logits, end_logits = tf.split(logits, 2, axis=-1)
        
        # 去除start_logits和end_logits中的多余维度
        start_logits = tf.squeeze(start_logits, axis=-1)
        end_logits = tf.squeeze(end_logits, axis=-1)

        # 计算损失
        loss = None
        if start_positions is not None and end_positions is not None:
            labels = {"start_position": start_positions}
            labels["end_position"] = end_positions
            loss = self.hf_compute_loss(labels, (start_logits, end_logits))

        # 如果不返回字典格式的结果，组合输出
        if not return_dict:
            output = (start_logits, end_logits) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        # 返回TFQuestionAnsweringModelOutput格式的结果
        return TFQuestionAnsweringModelOutput(
            loss=loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

    def build(self, input_shape=None):
        # 如果模型已经建立，直接返回
        if self.built:
            return
        # 标记模型为已建立状态
        self.built = True
        
        # 如果transformer模型已定义，则建立transformer模型
        if getattr(self, "transformer", None) is not None:
            with tf.name_scope(self.transformer.name):
                self.transformer.build(None)
        
        # 如果qa_outputs已定义，则建立qa_outputs
        if getattr(self, "qa_outputs", None) is not None:
            with tf.name_scope(self.qa_outputs.name):
                self.qa_outputs.build([None, None, self.config.hidden_size])
```