# `.\models\whisper\modeling_flax_whisper.py`

```
# 定义一个文档字符串，用于描述 WHISPER 模型的基本信息和继承关系
WHISPER_START_DOCSTRING = r"""
    This model inherits from [`FlaxPreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its models (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.) This model is also a Flax Linen
    # flax.nn.Module 的子类，可作为常规的 Flax 模块使用，参考 Flax 文档以了解一般用法和行为。
    # 最终，此模型支持 JAX 的内置特性，例如：
    # - Just-In-Time (JIT) 编译
    # - 自动微分
    # - 向量化
    # - 并行化
    
    # 参数：
    # config ([`WhisperConfig`]): 模型配置类，包含模型的所有参数。
    # 初始化时使用配置文件不会加载模型的权重，只会加载配置。请查看 [`~FlaxPreTrainedModel.from_pretrained`] 方法以加载模型权重。
    # dtype (`jax.numpy.dtype`, *optional*, 默认为 `jax.numpy.float32`):
    # 计算的数据类型。可以是 `jax.numpy.float32`，`jax.numpy.float16`（在 GPU 上），以及 `jax.numpy.bfloat16`（在 TPU 上）。
    # 这可以用于在 GPU 或 TPU 上启用混合精度训练或半精度推断。如果指定了dtype，则所有计算将使用给定的dtype执行。
    # **注意，这仅指定计算的数据类型，不会影响模型参数的数据类型。**
    # 如果要更改模型参数的数据类型，请参阅 [`~FlaxPreTrainedModel.to_fp16`] 和 [`~FlaxPreTrainedModel.to_bf16`]。
"""

WHISPER_INPUTS_DOCSTRING = r"""
"""

WHISPER_ENCODE_INPUTS_DOCSTRING = r"""
    Args:
        input_features (`numpy.ndarray` of shape `(batch_size, feature_size, sequence_length)`):
            Float values mel features extracted from the raw speech waveform. Raw speech waveform can be obtained by
            loading a `.flac` or `.wav` audio file into an array of type `List[float]` or a `numpy.ndarray`, *e.g.* via
            the soundfile library (`pip install soundfile`). To prepare the array into `input_features`, the
            [`WhisperFeatureExtractor`] should be used for extracting the mel features, padding and conversion into a
            tensor of type `numpy.ndarray`. See [`~WhisperFeatureExtractor.__call__`].
        attention_mask (`numpy.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
            Whisper does not support masking of the `input_features`, this argument is preserved for compatibility, but
            is not used. By default the silence in the input log mel spectrogram are ignored.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""

WHISPER_DECODE_INPUTS_DOCSTRING = r"""
"""

注释：
    Args:
        decoder_input_ids (`numpy.ndarray` of shape `(batch_size, target_sequence_length)`):
            # 解码器输入序列的标记索引，对应词汇表中的位置。索引可通过 `WhisperTokenizer` 获取。参见 `PreTrainedTokenizer.encode` 和 `PreTrainedTokenizer.__call__` 获取更多细节。
            # [decoder input IDs 是什么？](../glossary#decoder-input-ids)
            Indices of decoder input sequence tokens in the vocabulary. Indices can be obtained using
            [`WhisperTokenizer`]. See [`PreTrainedTokenizer.encode`] and [`PreTrainedTokenizer.__call__`] for details.
        encoder_outputs (`tuple(tuple(numpy.ndarray)`):
            # 编码器的输出元组，包含 (`last_hidden_state`, *可选*: `hidden_states`, *可选*: `attentions`)。
            # `last_hidden_state` 的形状为 `(batch_size, sequence_length, hidden_size)`，*可选* 是编码器最后一层的隐藏状态序列。用于解码器的交叉注意力。
            Tuple consists of (`last_hidden_state`, *optional*: `hidden_states`, *optional*: `attentions`)
            `last_hidden_state` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) is a sequence of
            hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.
        encoder_attention_mask (`numpy.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
            # 编码器注意力掩码。Whisper 不支持 `input_features` 的屏蔽，该参数为了兼容性而保留，但不被使用。
            # 默认情况下，会忽略输入 log mel 频谱图中的静默部分。
            Whisper does not support masking of the `input_features`, this argument is preserved for compatibility,
            but it is not used. By default the silence in the input log mel spectrogram are ignored.
        decoder_attention_mask (`numpy.ndarray` of shape `(batch_size, target_sequence_length)`, *optional*):
            # 解码器注意力掩码。默认行为：生成一个张量，忽略 `decoder_input_ids` 中的填充标记。
            # 默认还使用因果掩码。如需更改填充行为，应根据需求进行修改。参见论文中的图 1 获取更多关于默认策略的信息。
            Default behavior: generate a tensor that ignores pad tokens in `decoder_input_ids`. Causal mask will also
            be used by default. If you want to change padding behavior, you should modify to your needs. See diagram 1
            in [the paper](https://arxiv.org/abs/1910.13461) for more information on the default strategy.
        decoder_position_ids (`numpy.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
            # 解码器输入序列中每个标记的位置索引，在位置嵌入中选择的范围为 `[0, config.max_position_embeddings - 1]`。
            Indices of positions of each decoder input sequence tokens in the position embeddings. Selected in the
            range `[0, config.max_position_embeddings - 1]`.
        past_key_values (`Dict[str, numpy.ndarray]`, *optional*, returned by `init_cache` or when passing previous `past_key_values`):
            # 预计算的隐藏状态字典，包含用于快速自回归解码的注意力块中的键和值的隐藏状态。预计算的键和值的隐藏状态形状为 *[batch_size, max_length]*。
            Dictionary of pre-computed hidden-states (key and values in the attention blocks) that can be used for fast
            auto-regressive decoding. Pre-computed key and value hidden-states are of shape *[batch_size, max_length]*.
        output_attentions (`bool`, *optional*):
            # 是否返回所有注意力层的注意力张量。有关更多详细信息，请参见返回张量中的 `attentions`。
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            # 是否返回所有层的隐藏状态。有关更多详细信息，请参见返回张量中的 `hidden_states`。
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            # 是否返回一个 `~utils.ModelOutput` 而不是简单的元组。
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""
定义一个自定义的Flax模块，用于实现注意力机制。

config: WhisperConfig  # 用于存储配置信息的属性
embed_dim: int  # 嵌入维度
num_heads: int  # 注意力头的数量
dropout: float = 0.0  # 可选的丢弃率，默认为0.0
causal: bool = False  # 是否使用因果注意力，默认为False
bias: bool = True  # 是否使用偏置项，默认为True
dtype: jnp.dtype = jnp.float32  # 数据类型，默认为32位浮点型

def setup(self) -> None:
    self.head_dim = self.embed_dim // self.num_heads  # 计算每个注意力头的维度
    if self.head_dim * self.num_heads != self.embed_dim:
        raise ValueError(
            f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
            f" and `num_heads`: {self.num_heads})."
        )

    # 创建部分应用了配置的全连接层
    dense = partial(
        nn.Dense,
        self.embed_dim,
        dtype=self.dtype,
        kernel_init=jax.nn.initializers.normal(self.config.init_std),
    )

    self.q_proj = dense(use_bias=self.bias)  # Query投影层
    self.k_proj = dense(use_bias=False)  # Key投影层
    self.v_proj = dense(use_bias=self.bias)  # Value投影层
    self.out_proj = dense(use_bias=self.bias)  # 输出投影层

    if self.causal:
        # 如果启用了因果注意力，创建因果掩码
        self.causal_mask = make_causal_mask(
            jnp.ones((1, self.config.max_target_positions), dtype="bool"), dtype="bool"
        )

def __call__(
    self,
    hidden_states: jnp.ndarray,
    key_value_states: Optional[jnp.ndarray] = None,
    attention_mask: Optional[jnp.ndarray] = None,
    init_cache: bool = False,
    deterministic: bool = True,
):
    """
    实现模块的调用方法，执行注意力机制。
    Args:
        hidden_states: 输入的隐藏状态张量
        key_value_states: 可选的键值状态张量，用于自注意力机制
        attention_mask: 可选的注意力掩码张量，控制哪些位置参与注意力计算
        init_cache: 是否初始化缓存，通常用于Transformer模型
        deterministic: 是否使用确定性计算，影响是否使用随机性如dropout
    """
    ...

def _split_heads(self, hidden_state) -> jnp.ndarray:
    """
    将隐藏状态张量分割成多个注意力头。
    Args:
        hidden_state: 输入的隐藏状态张量
    Returns:
        jnp.ndarray: 分割后的张量
    """
    return hidden_state.reshape(hidden_state.shape[:2] + (self.num_heads, self.head_dim))

def _merge_heads(self, hidden_state) -> jnp.ndarray:
    """
    合并多个注意力头成一个隐藏状态张量。
    Args:
        hidden_state: 多头注意力张量
    Returns:
        jnp.ndarray: 合并后的张量
    """
    return hidden_state.reshape(hidden_state.shape[:2] + (self.embed_dim,))

@nn.compact
"""
    def _concatenate_to_cache(self, key, value, query, attention_mask) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        # 检测是否通过缺少现有缓存数据来初始化。
        is_initialized = self.has_variable("cache", "cached_key")
        # 获取或创建缓存的键，并用零值初始化
        cached_key = self.variable("cache", "cached_key", jnp.zeros, key.shape, key.dtype)
        # 获取或创建缓存的值，并用零值初始化
        cached_value = self.variable("cache", "cached_value", jnp.zeros, value.shape, value.dtype)
        # 获取或创建缓存索引，并用整数值0初始化
        cache_index = self.variable("cache", "cache_index", lambda: jnp.array(0, dtype=jnp.int32))

        if is_initialized:
            # 获取批次维度和注意力头数等维度信息
            *batch_dims, max_length, num_heads, depth_per_head = cached_key.value.shape
            # 使用新的1维空间切片更新键值缓存
            cur_index = cache_index.value
            # 计算动态更新切片的索引
            indices = (0,) * len(batch_dims) + (cur_index, 0, 0)
            # 动态更新缓存键值
            key = lax.dynamic_update_slice(cached_key.value, key, indices)
            value = lax.dynamic_update_slice(cached_value.value, value, indices)
            cached_key.value = key
            cached_value.value = value
            # 更新缓存索引，增加已更新的缓存向量数
            num_updated_cache_vectors = query.shape[1]
            cache_index.value = cache_index.value + num_updated_cache_vectors
            # 为缓存的解码器自注意力生成因果掩码：
            # 我们的单个查询位置只应注意已生成并缓存的键位置，而不是剩余的零元素。
            pad_mask = jnp.broadcast_to(
                jnp.arange(max_length) < cur_index + num_updated_cache_vectors,
                tuple(batch_dims) + (1, num_updated_cache_vectors, max_length),
            )
            # 合并生成的掩码和输入的注意力掩码
            attention_mask = combine_masks(pad_mask, attention_mask)

        # 返回更新后的键、值和注意力掩码
        return key, value, attention_mask
# 从transformers.models.mbart.modeling_flax_mbart.FlaxMBartEncoderLayer复制并修改为FlaxWhisperEncoderLayer
class FlaxWhisperEncoderLayer(nn.Module):
    # WhisperConfig类型的配置参数
    config: WhisperConfig
    # 计算时使用的数据类型，默认为32位浮点数
    dtype: jnp.dtype = jnp.float32

    # 模块的设置方法，用于初始化各个子模块
    def setup(self) -> None:
        # 编码器层的维度等于模型配置中的d_model
        self.embed_dim = self.config.d_model
        # 创建WhisperAttention自注意力机制对象
        self.self_attn = FlaxWhisperAttention(
            config=self.config,
            embed_dim=self.embed_dim,
            num_heads=self.config.encoder_attention_heads,
            dropout=self.config.attention_dropout,
            dtype=self.dtype,
        )
        # 对自注意力输出进行LayerNorm归一化处理
        self.self_attn_layer_norm = nn.LayerNorm(dtype=self.dtype, epsilon=1e-05)
        # 定义dropout层，用于随机屏蔽输入元素以防止过拟合
        self.dropout_layer = nn.Dropout(rate=self.config.dropout)
        # 激活函数，根据配置选择激活函数类型
        self.activation_fn = ACT2FN[self.config.activation_function]
        # 激活函数后的dropout层
        self.activation_dropout_layer = nn.Dropout(rate=self.config.activation_dropout)
        # 第一个全连接层，使用正态分布初始化权重
        self.fc1 = nn.Dense(
            self.config.encoder_ffn_dim,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.init_std),
        )
        # 第二个全连接层，输出维度为embed_dim，使用正态分布初始化权重
        self.fc2 = nn.Dense(
            self.embed_dim, dtype=self.dtype, kernel_init=jax.nn.initializers.normal(self.config.init_std)
        )
        # 最终输出的LayerNorm层
        self.final_layer_norm = nn.LayerNorm(dtype=self.dtype, epsilon=1e-05)

    # 对象调用方法，执行编码器层的前向传播
    def __call__(
        self,
        hidden_states: jnp.ndarray,          # 输入的隐藏状态张量
        attention_mask: jnp.ndarray,        # 注意力遮罩，用于屏蔽无效位置
        output_attentions: bool = True,      # 是否输出注意力权重
        deterministic: bool = True,          # 是否使用确定性计算
    ) -> Tuple[jnp.ndarray]:                # 返回类型为包含一个张量元组的元组
        # 保留残差连接
        residual = hidden_states
        # 对输入进行LayerNorm归一化
        hidden_states = self.self_attn_layer_norm(hidden_states)
        # 执行自注意力计算，并获取注意力权重
        hidden_states, attn_weights = self.self_attn(hidden_states=hidden_states, attention_mask=attention_mask)
        # 应用dropout层
        hidden_states = self.dropout_layer(hidden_states, deterministic=deterministic)
        # 残差连接
        hidden_states = residual + hidden_states

        # 保留残差连接
        residual = hidden_states
        # 对输出进行LayerNorm归一化
        hidden_states = self.final_layer_norm(hidden_states)
        # 应用激活函数和第一个全连接层
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        # 应用激活函数后的dropout层
        hidden_states = self.activation_dropout_layer(hidden_states, deterministic=deterministic)
        # 应用第二个全连接层
        hidden_states = self.fc2(hidden_states)
        # 应用最终的dropout层
        hidden_states = self.dropout_layer(hidden_states, deterministic=deterministic)
        # 残差连接
        hidden_states = residual + hidden_states

        # 输出结果为隐藏状态张量的元组
        outputs = (hidden_states,)

        # 如果需要输出注意力权重，则将注意力权重加入输出元组中
        if output_attentions:
            outputs += (attn_weights,)

        # 返回输出结果元组
        return outputs


# 编码器层集合类，包含多个编码器层对象
class FlaxWhisperEncoderLayerCollection(nn.Module):
    # WhisperConfig类型的配置参数
    config: WhisperConfig
    # 计算时使用的数据类型，默认为32位浮点数
    dtype: jnp.dtype = jnp.float32  # 计算的数据类型
    # 梯度检查点标志，默认为False
    gradient_checkpointing: bool = False
    # 初始化模型的设置
    def setup(self):
        # 如果启用了梯度检查点，则使用可重定向的编码器层，否则使用常规编码器层
        if self.gradient_checkpointing:
            FlaxWhisperEncoderCheckpointLayer = remat(FlaxWhisperEncoderLayer, static_argnums=(2, 3))
            # 创建编码器层列表，每层使用可重定向的编码器层实例化
            self.layers = [
                FlaxWhisperEncoderCheckpointLayer(self.config, name=str(i), dtype=self.dtype)
                for i in range(self.config.encoder_layers)
            ]
        else:
            # 创建编码器层列表，每层使用常规编码器层实例化
            self.layers = [
                FlaxWhisperEncoderLayer(self.config, name=str(i), dtype=self.dtype)
                for i in range(self.config.encoder_layers)
            ]
        # 设置编码器层的 LayerDrop 概率
        self.layerdrop = self.config.encoder_layerdrop

    # 模型的调用函数，接收隐藏状态、注意力掩码等输入参数
    def __call__(
        self,
        hidden_states,
        attention_mask,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        # 如果需要输出注意力矩阵，则初始化一个空元组以存储所有注意力矩阵
        all_attentions = () if output_attentions else None
        # 如果需要输出隐藏状态，则初始化一个空元组以存储所有隐藏状态
        all_hidden_states = () if output_hidden_states else None

        # 遍历每个编码器层
        for encoder_layer in self.layers:
            # 如果需要输出隐藏状态，则将当前隐藏状态加入到所有隐藏状态元组中
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            # 添加 LayerDrop 机制，根据概率跳过当前编码器层
            dropout_probability = random.uniform(0, 1)
            if not deterministic and (dropout_probability < self.layerdrop):
                # 如果跳过当前层，则设置输出为 None
                layer_outputs = (None, None)
            else:
                # 否则，调用当前编码器层的前向传播
                layer_outputs = encoder_layer(
                    hidden_states,
                    attention_mask,
                    output_attentions,
                    deterministic,
                )
            # 更新隐藏状态为当前层的输出隐藏状态
            hidden_states = layer_outputs[0]
            # 如果需要输出注意力矩阵，则将当前层的注意力矩阵加入到所有注意力矩阵元组中
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # 如果需要输出隐藏状态，则将最终的隐藏状态加入到所有隐藏状态元组中
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        # 组装模型的输出，包括最终的隐藏状态、所有隐藏状态和所有注意力矩阵
        outputs = (hidden_states, all_hidden_states, all_attentions)

        # 如果不需要返回字典，则返回元组形式的输出，去除值为 None 的部分
        if not return_dict:
            return tuple(v for v in outputs if v is not None)

        # 否则，返回 FlaxBaseModelOutput 类型的字典形式输出
        return FlaxBaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_attentions
        )
# 从transformers.models.mbart.modeling_flax_mbart.FlaxMBartDecoderLayer复制而来，改名为FlaxWhisperDecoderLayer
class FlaxWhisperDecoderLayer(nn.Module):
    # 类属性：配置信息为WhisperConfig类型
    config: WhisperConfig
    # 类属性：数据类型为jnp.float32
    dtype: jnp.dtype = jnp.float32

    # 初始化方法，设置各个层和模块
    def setup(self) -> None:
        # 设定embedding维度为配置中的模型维度
        self.embed_dim = self.config.d_model
        # 创建自注意力层对象FlaxWhisperAttention
        self.self_attn = FlaxWhisperAttention(
            config=self.config,
            embed_dim=self.embed_dim,
            num_heads=self.config.decoder_attention_heads,
            dropout=self.config.attention_dropout,
            causal=True,
            dtype=self.dtype,
        )
        # Dropout层，用于self-attention之后
        self.dropout_layer = nn.Dropout(rate=self.config.dropout)
        # 激活函数选择，根据配置选择相应的激活函数
        self.activation_fn = ACT2FN[self.config.activation_function]
        # 激活函数之后的dropout层
        self.activation_dropout_layer = nn.Dropout(rate=self.config.activation_dropout)

        # Layer normalization层，用于self-attention输出
        self.self_attn_layer_norm = nn.LayerNorm(dtype=self.dtype, epsilon=1e-05)
        
        # 创建编码器-解码器注意力层对象FlaxWhisperAttention
        self.encoder_attn = FlaxWhisperAttention(
            config=self.config,
            embed_dim=self.embed_dim,
            num_heads=self.config.decoder_attention_heads,
            dropout=self.config.attention_dropout,
            dtype=self.dtype,
        )
        # 编码器-解码器注意力层后的Layer normalization层
        self.encoder_attn_layer_norm = nn.LayerNorm(dtype=self.dtype, epsilon=1e-05)
        
        # 第一个全连接层，即前馈神经网络的第一层
        self.fc1 = nn.Dense(
            self.config.decoder_ffn_dim,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.init_std),
        )
        # 第二个全连接层，即前馈神经网络的第二层，输出维度为embedding维度
        self.fc2 = nn.Dense(
            self.embed_dim, dtype=self.dtype, kernel_init=jax.nn.initializers.normal(self.config.init_std)
        )
        # 最终输出的Layer normalization层
        self.final_layer_norm = nn.LayerNorm(dtype=self.dtype, epsilon=1e-05)

    # 对象调用方法，执行解码层的前向计算
    def __call__(
        self,
        hidden_states: jnp.ndarray,  # 输入的隐藏状态
        attention_mask: jnp.ndarray,  # 自注意力和编码器-解码器注意力的掩码
        encoder_hidden_states: Optional[jnp.ndarray] = None,  # 编码器的隐藏状态，可选
        encoder_attention_mask: Optional[jnp.ndarray] = None,  # 编码器的注意力掩码，可选
        init_cache: bool = False,  # 是否初始化缓存，布尔类型
        output_attentions: bool = True,  # 是否输出注意力权重，布尔类型
        deterministic: bool = True,  # 是否确定性计算，布尔类型
        ):
    ) -> Tuple[jnp.ndarray]:
        # 保存输入的隐藏状态作为残差连接的基础
        residual = hidden_states
        # 对输入的隐藏状态进行 Layer normalization 处理
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # Self Attention
        # 调用自注意力机制，处理输入的隐藏状态，生成新的隐藏状态和注意力权重
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states, attention_mask=attention_mask, init_cache=init_cache
        )
        # 对生成的新的隐藏状态进行 Dropout 处理
        hidden_states = self.dropout_layer(hidden_states, deterministic=deterministic)
        # 使用残差连接，将处理后的隐藏状态与原始输入相加
        hidden_states = residual + hidden_states

        # Cross-Attention Block
        cross_attn_weights = None
        if encoder_hidden_states is not None:
            # 保存当前隐藏状态作为残差连接的基础
            residual = hidden_states
            # 对当前隐藏状态进行 Layer normalization 处理
            hidden_states = self.encoder_attn_layer_norm(hidden_states)
            # 调用编码器-解码器注意力机制，处理当前隐藏状态和编码器的隐藏状态，生成新的隐藏状态和注意力权重
            hidden_states, cross_attn_weights = self.encoder_attn(
                hidden_states=hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
            )
            # 对生成的新的隐藏状态进行 Dropout 处理
            hidden_states = self.dropout_layer(hidden_states, deterministic=deterministic)
            # 使用残差连接，将处理后的隐藏状态与原始输入相加
            hidden_states = residual + hidden_states

        # Fully Connected
        # 保存当前隐藏状态作为残差连接的基础
        residual = hidden_states
        # 对当前隐藏状态进行 Layer normalization 处理
        hidden_states = self.final_layer_norm(hidden_states)
        # 使用激活函数对处理后的隐藏状态进行非线性变换
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        # 对生成的新的隐藏状态进行 Dropout 处理
        hidden_states = self.activation_dropout_layer(hidden_states, deterministic=deterministic)
        # 使用全连接层进行线性变换
        hidden_states = self.fc2(hidden_states)
        # 对生成的新的隐藏状态进行 Dropout 处理
        hidden_states = self.dropout_layer(hidden_states, deterministic=deterministic)
        # 使用残差连接，将处理后的隐藏状态与原始输入相加
        hidden_states = residual + hidden_states

        # 构造输出元组
        outputs = (hidden_states,)

        # 如果需要输出注意力权重，则将自注意力和交叉注意力的权重添加到输出中
        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)

        # 返回最终的输出
        return outputs
# 定义一个名为 FlaxWhisperDecoderLayerCollection 的类，继承自 nn.Module
class FlaxWhisperDecoderLayerCollection(nn.Module):
    # 类变量 config，类型为 WhisperConfig，用于存储配置信息
    config: WhisperConfig
    # 类变量 dtype，默认为 jnp.float32，表示计算中使用的数据类型
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation
    # 类变量 gradient_checkpointing，默认为 False，用于控制是否使用梯度检查点
    gradient_checkpointing: bool = False

    # 定义类方法 setup，用于初始化类的实例
    def setup(self):
        # 如果启用梯度检查点
        if self.gradient_checkpointing:
            # 动态创建 FlaxWhisperDecoderCheckpointLayer 类的实例，设置静态参数编号
            FlaxWhisperDecoderCheckpointLayer = remat(FlaxWhisperDecoderLayer, static_argnums=(4, 5, 6))
            # 创建多个 FlaxWhisperDecoderCheckpointLayer 实例，存储在 self.layers 中
            self.layers = [
                FlaxWhisperDecoderCheckpointLayer(self.config, name=str(i), dtype=self.dtype)
                for i in range(self.config.decoder_layers)
            ]
        else:
            # 如果未启用梯度检查点，创建多个 FlaxWhisperDecoderLayer 实例，存储在 self.layers 中
            self.layers = [
                FlaxWhisperDecoderLayer(self.config, name=str(i), dtype=self.dtype)
                for i in range(self.config.decoder_layers)
            ]
        # 设置类变量 layerdrop 为 config 中的 decoder_layerdrop 参数
        self.layerdrop = self.config.decoder_layerdrop

    # 定义 __call__ 方法，使实例对象可以像函数一样调用
    def __call__(
        self,
        hidden_states,
        attention_mask,
        encoder_hidden_states: Optional[jnp.ndarray] = None,
        encoder_attention_mask: Optional[jnp.ndarray] = None,
        deterministic: bool = True,
        init_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        # 此处省略了方法的余下部分，不在此注释范围内
    ):
        # 如果输出隐藏状态，则初始化一个空元组；否则设为 None
        all_hidden_states = () if output_hidden_states else None
        # 如果输出注意力权重，则初始化一个空元组；否则设为 None
        all_self_attns = () if output_attentions else None
        # 如果输出交叉注意力权重，并且编码器隐藏状态不为空，则初始化一个空元组；否则设为 None
        all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None

        # 遍历解码器每一层
        for decoder_layer in self.layers:
            # 如果需要输出隐藏状态，则将当前隐藏状态加入 all_hidden_states 元组
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
                # 添加层丢弃 (LayerDrop)，详见 https://arxiv.org/abs/1909.11556
            # 生成一个0到1之间的随机数作为丢弃概率
            dropout_probability = random.uniform(0, 1)
            # 如果非确定性模式且随机数小于层丢弃率，则将层输出置为None
            if not deterministic and (dropout_probability < self.layerdrop):
                layer_outputs = (None, None, None)
            else:
                # 否则，调用当前解码器层进行计算
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    init_cache,
                    output_attentions,
                    deterministic,
                )

            # 更新隐藏状态为当前层输出的第一个元素
            hidden_states = layer_outputs[0]
            # 如果需要输出注意力权重，则将当前层的自注意力权重加入 all_self_attns 元组
            if output_attentions:
                all_self_attns += (layer_outputs[1],)

                # 如果编码器隐藏状态不为空，则将当前层的交叉注意力权重加入 all_cross_attentions 元组
                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)

        # 如果需要输出隐藏状态，则将最后一个解码器层的隐藏状态加入 all_hidden_states 元组
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        # 将所有输出结果放入 outputs 列表中
        outputs = [hidden_states, all_hidden_states, all_self_attns, all_cross_attentions]

        # 如果不需要返回字典形式的输出，则返回 outputs 中非空的元素组成的元组
        if not return_dict:
            return tuple(v for v in outputs if v is not None)

        # 否则，返回带有过去和交叉注意力的 FlaxBaseModelOutputWithPastAndCrossAttentions 对象
        return FlaxBaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
        )
# 定义一个名为 FlaxWhisperEncoder 的神经网络模块，继承自 nn.Module
class FlaxWhisperEncoder(nn.Module):
    # 定义类变量 config，类型为 WhisperConfig，用于存储模型配置信息
    config: WhisperConfig
    # 定义变量 dtype，默认为 jnp.float32，指定模型数据类型为 32 位浮点数
    dtype: jnp.dtype = jnp.float32
    # 定义变量 gradient_checkpointing，默认为 False，用于指示是否启用梯度检查点
    gradient_checkpointing: bool = False

    # 定义初始化方法，没有参数返回值
    def setup(self) -> None:
        # 创建第一个卷积层 conv1
        self.conv1 = nn.Conv(
            self.config.d_model,  # 输入通道数为 d_model
            kernel_size=(3,),     # 卷积核大小为 3
            padding=1,            # 使用 1 像素的填充
            # 使用正态分布初始化卷积核，标准差为 config.init_std
            kernel_init=jax.nn.initializers.normal(self.config.init_std),
            dtype=self.dtype,     # 指定数据类型为 dtype
        )
        # 创建第二个卷积层 conv2
        self.conv2 = nn.Conv(
            self.config.d_model,  # 输入通道数为 d_model
            kernel_size=(3,),     # 卷积核大小为 3
            strides=2,            # 步长为 2
            padding=1,            # 使用 1 像素的填充
            # 使用正态分布初始化卷积核，标准差为 config.init_std
            kernel_init=jax.nn.initializers.normal(self.config.init_std),
            dtype=self.dtype,     # 指定数据类型为 dtype
        )

        # 创建一个 Dropout 层，丢弃率为 config.dropout
        self.dropout_layer = nn.Dropout(rate=self.config.dropout)

        # 创建一个 FlaxWhisperEncoderLayerCollection 对象 layers，用于存储编码器的层集合
        self.layers = FlaxWhisperEncoderLayerCollection(
            self.config,  # 传入编码器配置信息
            dtype=self.dtype,  # 指定数据类型为 dtype
            gradient_checkpointing=self.gradient_checkpointing,  # 传入梯度检查点标志
        )

        # 创建一个位置嵌入层 embed_positions
        self.embed_positions = nn.Embed(
            self.config.max_source_positions,  # 最大源位置数
            self.config.d_model,               # 嵌入向量维度为 d_model
            dtype=self.dtype,                 # 指定数据类型为 dtype
            embedding_init=sinusoidal_embedding_init,  # 使用正弦嵌入初始化
        )

        # 创建一个 LayerNorm 层 layer_norm，用于归一化层的输出
        self.layer_norm = nn.LayerNorm(dtype=self.dtype, epsilon=1e-05)

    # 定义调用方法，接收多个输入参数并返回处理结果
    def __call__(
        self,
        input_features: jnp.ndarray,  # 输入特征，类型为 jnp.ndarray
        output_attentions: bool = False,  # 是否输出注意力权重，默认为 False
        output_hidden_states: bool = False,  # 是否输出隐藏状态，默认为 False
        return_dict: bool = True,  # 是否以字典形式返回，默认为 True
        deterministic: bool = True,  # 是否确定性运行，默认为 True

        # 方法内容在此继续
    # 指定函数的返回类型为包含单个元组的元组，元组的唯一元素为 jnp.ndarray 类型的对象
    ) -> Tuple[jnp.ndarray]:
        # 如果输入特征的形状的第二维不等于 (self.config.num_mel_bins, self.config.max_source_positions * 2)
        if input_features.shape[1:] != (self.config.num_mel_bins, self.config.max_source_positions * 2):
            # 抛出值错误，提示详细信息
            raise ValueError(
                "input_features.shape[1:], must be equal to (self.config.num_mel_bins,"
                f" self.config.max_source_positions * 2) (got {input_features.shape[1:]}, but should be"
                f" ({self.config.num_mel_bins}, {self.config.max_source_positions * 2}))"
            )

        # 调整输入特征的维度顺序，将第二维移动到第三维
        input_features = input_features.transpose(0, 2, 1)
        # 使用 GELU 激活函数对卷积层 conv1 处理后的隐藏状态进行非线性变换
        hidden_states = jax.nn.gelu(self.conv1(input_features), approximate=False)
        # 使用 GELU 激活函数对卷积层 conv2 处理后的隐藏状态进行非线性变换
        hidden_states = jax.nn.gelu(self.conv2(hidden_states), approximate=False)

        # 生成 sinusoidal embeddings，用于位置编码，采用自然数序列 0 到 self.config.max_source_positions
        embed_positions = self.embed_positions(jnp.arange(self.config.max_source_positions))
        # 停止位置编码的梯度传播，使其在后续训练中保持不变
        embed_positions = jax.lax.stop_gradient(embed_positions)
        # 将位置编码添加到隐藏状态中
        hidden_states = hidden_states + embed_positions

        # 使用 dropout 层对隐藏状态进行随机失活，若 deterministic 为 True，则保持确定性
        hidden_states = self.dropout_layer(hidden_states, deterministic=deterministic)

        # 将隐藏状态传递给模型的各层进行处理，包括注意力掩码和其他输出参数
        outputs = self.layers(
            hidden_states,
            attention_mask=None,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取模型输出的最后隐藏状态
        last_hidden_states = outputs[0]
        # 对最后隐藏状态进行 layer normalization
        last_hidden_states = self.layer_norm(last_hidden_states)

        # 如果需要输出隐藏状态，更新隐藏状态的最后一个元素为经过 layernorm 处理后的最后隐藏状态
        hidden_states = None
        if output_hidden_states:
            hidden_states = outputs[1]
            hidden_states = hidden_states[:-1] + (last_hidden_states,)

        # 如果不返回字典形式的结果，则构造输出元组并返回
        if not return_dict:
            outputs = (last_hidden_states, hidden_states) + (outputs[2:] if output_hidden_states else outputs[1:])
            return tuple(v for v in outputs if v is not None)

        # 返回 FlaxBaseModelOutput 对象，包含最后的隐藏状态、隐藏状态和注意力输出
        return FlaxBaseModelOutput(
            last_hidden_state=last_hidden_states,
            hidden_states=hidden_states,
            attentions=outputs.attentions,
        )
# 定义 FlaxWhisperDecoder 类，继承自 nn.Module，用于解码器模型
class FlaxWhisperDecoder(nn.Module):
    # 定义类属性 config，类型为 WhisperConfig，dtype 默认为 jnp.float32，梯度检查点默认为 False
    config: WhisperConfig
    dtype: jnp.dtype = jnp.float32
    gradient_checkpointing: bool = False

    # 初始化方法，无返回值
    def setup(self) -> None:
        # 创建词嵌入层，vocab_size 和 d_model 从 config 中获取，dtype 为 self.dtype
        self.embed_tokens = nn.Embed(self.config.vocab_size, self.config.d_model, dtype=self.dtype)
        # 创建位置嵌入层，max_target_positions 和 d_model 从 config 中获取，dtype 为 self.dtype
        self.embed_positions = nn.Embed(self.config.max_target_positions, self.config.d_model, dtype=self.dtype)

        # 创建解码器层集合，传入 config、dtype 和 gradient_checkpointing 参数
        self.layers = FlaxWhisperDecoderLayerCollection(
            self.config, dtype=self.dtype, gradient_checkpointing=self.gradient_checkpointing
        )

        # 创建 Dropout 层，dropout 率从 config 中获取
        self.dropout_layer = nn.Dropout(rate=self.config.dropout)

        # 创建 LayerNorm 层，使用 self.dtype 和 epsilon=1e-5
        self.layer_norm = nn.LayerNorm(dtype=self.dtype, epsilon=1e-5)

    # 定义 __call__ 方法，接受多个输入和返回一个元组的输出
    def __call__(
        self,
        input_ids: jnp.ndarray,
        attention_mask: jnp.ndarray,
        position_ids: jnp.ndarray,
        encoder_hidden_states: Optional[jnp.ndarray] = None,
        init_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        deterministic: bool = True,
    ) -> Tuple[jnp.ndarray]:
        # 获取输入和位置嵌入
        input_embeds = self.embed_tokens(input_ids)
        position_embeds = self.embed_positions(position_ids)

        # 计算隐藏状态，将输入嵌入和位置嵌入相加
        hidden_states = input_embeds + position_embeds
        # 对隐藏状态应用 Dropout 层
        hidden_states = self.dropout_layer(hidden_states, deterministic=deterministic)

        # 调用解码器层集合 layers 进行解码器的前向传播
        outputs = self.layers(
            hidden_states,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            deterministic=deterministic,
            init_cache=init_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取最后的隐藏状态，并应用 LayerNorm 层
        last_hidden_states = outputs[0]
        last_hidden_states = self.layer_norm(last_hidden_states)

        # 更新输出的 hidden_states 变量，如果需要输出隐藏状态
        hidden_states = None
        if output_hidden_states:
            hidden_states = outputs[1]
            hidden_states = hidden_states[:-1] + (last_hidden_states,)

        # 如果 return_dict 为 False，则返回一个元组，包含最后的隐藏状态和隐藏状态列表（如果有）
        if not return_dict:
            outputs = (last_hidden_states, hidden_states) + (outputs[2:] if output_hidden_states else outputs[1:])
            return tuple(v for v in outputs if v is not None)

        # 如果 return_dict 为 True，则返回 FlaxBaseModelOutputWithPastAndCrossAttentions 对象
        return FlaxBaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=last_hidden_states,
            hidden_states=hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )


# 定义 FlaxWhisperModule 类，继承自 nn.Module
class FlaxWhisperModule(nn.Module):
    # 定义类属性 config，类型为 WhisperConfig，dtype 默认为 jnp.float32，梯度检查点默认为 False
    config: WhisperConfig
    dtype: jnp.dtype = jnp.float32
    gradient_checkpointing: bool = False
    # 设置模型的初始化，初始化编码器和解码器
    def setup(self) -> None:
        self.encoder = FlaxWhisperEncoder(
            self.config, dtype=self.dtype, gradient_checkpointing=self.gradient_checkpointing
        )
        self.decoder = FlaxWhisperDecoder(
            self.config, dtype=self.dtype, gradient_checkpointing=self.gradient_checkpointing
        )

    # 调用模型时执行的方法，将输入特征和解码器的输入传递给编码器和解码器，生成序列到序列的输出
    def __call__(
        self,
        input_features: jnp.ndarray,
        decoder_input_ids: jnp.ndarray,
        decoder_attention_mask: jnp.ndarray,
        decoder_position_ids: jnp.ndarray,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        deterministic: bool = True,
    ):
        # 调用编码器进行编码器输出的计算
        encoder_outputs = self.encoder(
            input_features,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            deterministic=deterministic,
        )

        # 调用解码器进行解码器输出的计算，传入编码器输出作为输入
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            position_ids=decoder_position_ids,
            encoder_hidden_states=encoder_outputs[0],
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            deterministic=deterministic,
        )

        # 如果不返回字典形式的输出，则将解码器和编码器的输出合并返回
        if not return_dict:
            return decoder_outputs + encoder_outputs

        # 返回字典形式的序列到序列模型输出
        return FlaxSeq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

    # 获取编码器模块的方法
    def _get_encoder_module(self):
        return self.encoder

    # 获取解码器模块的方法
    def _get_decoder_module(self):
        return self.decoder
# 定义一个继承自FlaxPreTrainedModel的类，用于预训练的Whisper模型
class FlaxWhisperPreTrainedModel(FlaxPreTrainedModel):
    # 配置类为WhisperConfig
    config_class = WhisperConfig
    # 基础模型的前缀名为"model"
    base_model_prefix: str = "model"
    # 主要输入的名称为"input_features"
    main_input_name = "input_features"
    # 模块类初始化为None
    module_class: nn.Module = None

    # 初始化函数
    def __init__(
        self,
        config: WhisperConfig,
        input_shape: Tuple[int] = None,
        seed: int = 0,
        dtype: jnp.dtype = jnp.float32,
        _do_init: bool = True,
        gradient_checkpointing: bool = False,
        **kwargs,
    ):
        # 使用给定的config、dtype和gradient_checkpointing参数来初始化模块
        module = self.module_class(config=config, dtype=dtype, gradient_checkpointing=gradient_checkpointing, **kwargs)
        # 如果未提供input_shape，则默认为(1, num_mel_bins, 2 * max_source_positions)
        if input_shape is None:
            input_shape = (1, config.num_mel_bins, 2 * config.max_source_positions)
        # 调用父类的初始化方法，传递config、module、input_shape等参数
        super().__init__(config, module, input_shape=input_shape, seed=seed, dtype=dtype, _do_init=_do_init)

    # 启用梯度检查点的函数
    def enable_gradient_checkpointing(self):
        # 设置模块的_gradient_checkpointing属性为True
        self._module = self.module_class(
            config=self.config,
            dtype=self.dtype,
            gradient_checkpointing=True,
        )

    # 初始化权重的函数
    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict = None) -> FrozenDict:
        # 初始化输入张量input_features，全零张量，并设置最后一个位置为eos_token_id
        input_features = jnp.zeros(input_shape, dtype="f4")
        input_features = input_features.at[(..., -1)].set(self.config.eos_token_id)

        # 初始化decoder_input_ids为全零张量，decoder_attention_mask为全1张量
        decoder_input_ids = jnp.zeros((input_shape[0], 1), dtype="i4")
        decoder_attention_mask = jnp.ones_like(decoder_input_ids)

        # 获取decoder_input_ids的批次大小和序列长度，初始化decoder_position_ids
        batch_size, sequence_length = decoder_input_ids.shape
        decoder_position_ids = jnp.broadcast_to(jnp.arange(sequence_length)[None, :], (batch_size, sequence_length))

        # 分割随机数生成器rng，获取params_rng和dropout_rng
        params_rng, dropout_rng = jax.random.split(rng)
        rngs = {"params": params_rng, "dropout": dropout_rng}

        # 使用module的init方法初始化随机参数
        random_params = self.module.init(
            rngs,
            input_features=input_features,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            decoder_position_ids=decoder_position_ids,
        )["params"]

        # 如果提供了params，则将缺失的键补全并返回冻结后的参数
        if params is not None:
            random_params = flatten_dict(unfreeze(random_params))
            params = flatten_dict(unfreeze(params))
            for missing_key in self._missing_keys:
                params[missing_key] = random_params[missing_key]
            self._missing_keys = set()
            return freeze(unflatten_dict(params))
        else:
            # 否则直接返回随机初始化的参数
            return random_params

    # 从transformers.models.bart.modeling_flax_bart.FlaxBartPreTrainedModel.init_cache复制过来，将Bart替换为Whisper
    # 这部分代码实现了缓存的初始化，但具体细节在此不详述
    # 初始化缓存方法，用于快速自回归解码
    def init_cache(self, batch_size, max_length, encoder_outputs):
        r"""
        Args:
            batch_size (`int`):
                用于快速自回归解码的批大小。定义了初始化缓存时的批大小。
            max_length (`int`):
                自回归解码的最大可能长度。定义了初始化缓存时的序列长度。
            encoder_outputs (`Union[FlaxBaseModelOutput, tuple(tuple(jnp.ndarray)]`):
                `encoder_outputs` 包含 (`last_hidden_state`, *可选*: `hidden_states`, *可选*: `attentions`)。
                `last_hidden_state` 的形状为 `(batch_size, sequence_length, hidden_size)`，*可选* 是编码器最后一层的隐藏状态的序列。
                在解码器的交叉注意力中使用。

        """
        # 初始化解码器输入的标识符，默认为全1矩阵，形状为(batch_size, max_length)
        decoder_input_ids = jnp.ones((batch_size, max_length), dtype="i4")
        # 初始化解码器注意力遮罩，与decoder_input_ids形状相同的全1矩阵
        decoder_attention_mask = jnp.ones_like(decoder_input_ids)
        # 初始化解码器位置标识符，广播初始化为解码器输入标识符的长度
        decoder_position_ids = jnp.broadcast_to(
            jnp.arange(jnp.atleast_2d(decoder_input_ids).shape[-1]), decoder_input_ids.shape
        )

        def _decoder_forward(module, decoder_input_ids, decoder_attention_mask, decoder_position_ids, **kwargs):
            # 获取解码器模块
            decoder_module = module._get_decoder_module()
            # 调用解码器模块进行前向传播
            return decoder_module(
                decoder_input_ids,
                decoder_attention_mask,
                decoder_position_ids,
                **kwargs,
            )

        # 初始化模型参数，用于获取缓存
        init_variables = self.module.init(
            jax.random.PRNGKey(0),
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            decoder_position_ids=decoder_position_ids,
            encoder_hidden_states=encoder_outputs[0],  # 使用编码器的最后隐藏状态
            init_cache=True,
            method=_decoder_forward,  # 只需调用解码器以初始化缓存
        )
        # 返回解冻后的缓存变量
        return unfreeze(init_variables["cache"])

    @add_start_docstrings(WHISPER_ENCODE_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=FlaxBaseModelOutput, config_class=WhisperConfig)
    # 编码方法，用于将输入特征编码为输出
    def encode(
        self,
        input_features: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        train: bool = False,
        params: dict = None,
        dropout_rng: PRNGKey = None,
        **kwargs,
    ):
        r"""
        Returns:

        Example:

        ```python
        >>> from transformers import WhisperProcessor, FlaxWhisperForConditionalGeneration
        >>> from datasets import load_dataset

        >>> processor = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")
        >>> model = FlaxWhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en", from_pt=True)
        >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        >>> inputs = processor(ds[0]["audio"]["array"], return_tensors="np")
        >>> input_features = inputs.input_features
        >>> encoder_outputs = model.encode(input_features=input_features)
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        # Handle any PRNG if needed
        rngs = {}
        if dropout_rng is not None:
            rngs["dropout"] = dropout_rng

        def _encoder_forward(module, input_features, **kwargs):
            encode_module = module._get_encoder_module()
            return encode_module(input_features, **kwargs)

        return self.module.apply(
            {"params": params or self.params},
            input_features=jnp.array(input_features, dtype="f4"),
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            deterministic=not train,
            rngs=rngs,
            method=_encoder_forward,
        )

    @add_start_docstrings(WHISPER_DECODE_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=FlaxBaseModelOutputWithPastAndCrossAttentions, config_class=WhisperConfig)
    def decode(
        self,
        decoder_input_ids,
        encoder_outputs,
        encoder_attention_mask: Optional[jnp.ndarray] = None,
        decoder_attention_mask: Optional[jnp.ndarray] = None,
        decoder_position_ids: Optional[jnp.ndarray] = None,
        past_key_values: dict = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        train: bool = False,
        params: dict = None,
        dropout_rng: PRNGKey = None,
    ):
        """
        Decoder function for the Whisper model. Transforms decoder input into model predictions.

        Args:
            decoder_input_ids: Input IDs for the decoder.
            encoder_outputs: Outputs from the encoder model.
            encoder_attention_mask: Mask for encoder attention.
            decoder_attention_mask: Mask for decoder attention.
            decoder_position_ids: Position IDs for the decoder.
            past_key_values: Cached key-value states for fast decoding.
            output_attentions: Whether to output attention weights.
            output_hidden_states: Whether to output hidden states.
            return_dict: Whether to return a dictionary of outputs.
            train: Whether in training mode.
            params: Model parameters to use.
            dropout_rng: Random number generator for dropout.

        Returns:
            Output with past and cross attentions as specified by FlaxBaseModelOutputWithPastAndCrossAttentions.

        Example:

        ```python
        >>> model = FlaxWhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en", from_pt=True)
        >>> encoder_outputs = model.encode(input_features=input_features)
        >>> decoder_inputs = {"input_ids": decoder_input_ids}
        >>> outputs = model.decode(**decoder_inputs, encoder_outputs=encoder_outputs)
        ```
        """
    def __call__(
        self,
        input_features: jnp.ndarray,
        decoder_input_ids: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray] = None,
        decoder_attention_mask: Optional[jnp.ndarray] = None,
        position_ids: Optional[jnp.ndarray] = None,
        decoder_position_ids: Optional[jnp.ndarray] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        train: bool = False,
        params: dict = None,
        dropout_rng: PRNGKey = None,
    ):
        # 设置输出注意力权重的选择，如果未指定则使用默认配置
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 设置输出隐藏状态的选择，如果未指定则使用默认配置
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 设置返回字典的选择，如果未指定则使用默认配置
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        # 准备解码器输入位置信息
        if decoder_position_ids is None:
            # 如果解码器位置信息未提供，根据解码器注意力掩码生成位置信息
            if decoder_attention_mask is not None:
                decoder_position_ids = (decoder_attention_mask.cumsum(-1) * decoder_attention_mask) - 1
            else:
                # 否则，根据解码器输入的形状生成默认位置信息
                batch_size, sequence_length = decoder_input_ids.shape
                decoder_position_ids = jnp.broadcast_to(
                    jnp.arange(sequence_length)[None, :], (batch_size, sequence_length)
                )
        
        # 如果解码器注意力掩码未提供，使用全1的掩码
        if decoder_attention_mask is None:
            decoder_attention_mask = jnp.ones_like(decoder_input_ids)

        # 如果需要处理任何的随机数生成器
        rngs = {"dropout": dropout_rng} if dropout_rng is not None else {}

        # 调用模块的应用方法，传递参数和输入数据
        return self.module.apply(
            {"params": params or self.params},
            input_features=jnp.array(input_features, dtype="f4"),
            decoder_input_ids=jnp.array(decoder_input_ids, dtype="i4"),
            decoder_attention_mask=jnp.array(decoder_attention_mask, dtype="i4"),
            decoder_position_ids=jnp.array(decoder_position_ids, dtype="i4"),
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            deterministic=not train,
            rngs=rngs,
        )
# 使用装饰器添加文档字符串到类 FlaxWhisperModel 上，描述其作用是生成不带特定头部的原始隐藏状态的 Whisper 模型转换器。
@add_start_docstrings(
    "The bare Whisper Model transformer outputting raw hidden-states without any specific head on top.",
    WHISPER_START_DOCSTRING,
)
# 定义 FlaxWhisperModel 类，继承自 FlaxWhisperPreTrainedModel
class FlaxWhisperModel(FlaxWhisperPreTrainedModel):
    # 配置项，指定为 WhisperConfig 类型
    config: WhisperConfig
    # 计算中使用的数据类型，默认为 jnp.float32
    dtype: jnp.dtype = jnp.float32
    # 模块类别为 FlaxWhisperModule
    module_class = FlaxWhisperModule

# 调用函数 append_call_sample_docstring，添加示例代码的文档字符串到 FlaxWhisperModel 类上
append_call_sample_docstring(FlaxWhisperModel, _CHECKPOINT_FOR_DOC, FlaxSeq2SeqModelOutput, _CONFIG_FOR_DOC)

# 定义 FlaxWhisperForConditionalGenerationModule 类，继承自 nn.Module
class FlaxWhisperForConditionalGenerationModule(nn.Module):
    # 配置项，指定为 WhisperConfig 类型
    config: WhisperConfig
    # 计算中使用的数据类型，默认为 jnp.float32
    dtype: jnp.dtype = jnp.float32
    # 是否使用梯度检查点，默认为 False
    gradient_checkpointing: bool = False

    # 初始化函数，设置模型和语言模型头部
    def setup(self) -> None:
        # 创建 FlaxWhisperModule 实例作为模型
        self.model = FlaxWhisperModule(
            config=self.config, dtype=self.dtype, gradient_checkpointing=self.gradient_checkpointing
        )
        # 创建 nn.Dense 实例作为语言模型头部
        self.lm_head = nn.Dense(
            self.config.vocab_size,
            use_bias=False,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.init_std),
        )

    # 获取编码器模块
    def _get_encoder_module(self):
        return self.model.encoder

    # 获取解码器模块
    def _get_decoder_module(self):
        return self.model.decoder

    # 定义 __call__ 方法，实现类的可调用功能
    def __call__(
        self,
        input_features,
        decoder_input_ids,
        decoder_attention_mask: jnp.ndarray = None,
        decoder_position_ids: jnp.ndarray = None,
        position_ids: jnp.ndarray = None,
        attention_mask: jnp.ndarray = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        deterministic: bool = True,
    ):
        # 调用模型的 __call__ 方法，传入参数，并接收输出
        outputs = self.model(
            input_features=input_features,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            decoder_position_ids=decoder_position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            deterministic=deterministic,
        )

        # 获取模型的隐藏状态输出
        hidden_states = outputs[0]

        # 如果配置中要求共享词嵌入
        if self.config.tie_word_embeddings:
            # 获取共享的嵌入层参数
            shared_embedding = self.model.decoder.embed_tokens.variables["params"]["embedding"]
            # 应用语言模型头部到隐藏状态上，使用共享的嵌入参数
            lm_logits = self.lm_head.apply({"params": {"kernel": shared_embedding.T}}, hidden_states)
        else:
            # 应用语言模型头部到隐藏状态上
            lm_logits = self.lm_head(hidden_states)

        # 如果不要求返回字典格式
        if not return_dict:
            # 组装输出元组
            output = (lm_logits,) + outputs[1:]
            return output

        # 返回 FlaxSeq2SeqLMOutput 类型的输出对象，包括 logits 和可能的其他属性
        return FlaxSeq2SeqLMOutput(
            logits=lm_logits,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )
@add_start_docstrings("The Whisper Model with a language modeling head.", WHISPER_START_DOCSTRING)
# 使用装饰器为类添加文档字符串，描述它是一个带有语言建模头的 Whisper 模型。

class FlaxWhisperForConditionalGeneration(FlaxWhisperPreTrainedModel):
    # 设置模块类为 FlaxWhisperForConditionalGenerationModule
    module_class = FlaxWhisperForConditionalGenerationModule
    # 设定数据类型为 jnp.float32
    dtype: jnp.dtype = jnp.float32

    @add_start_docstrings(WHISPER_DECODE_INPUTS_DOCSTRING)
    # 使用装饰器添加解码方法的输入文档字符串，描述解码方法的输入参数含义。

    @replace_return_docstrings(output_type=FlaxCausalLMOutputWithCrossAttentions, config_class=WhisperConfig)
    # 使用装饰器替换返回值的文档字符串，指定输出类型和配置类为 FlaxCausalLMOutputWithCrossAttentions 和 WhisperConfig。

    def decode(
        self,
        decoder_input_ids,
        encoder_outputs,
        encoder_attention_mask: Optional[jnp.ndarray] = None,
        decoder_attention_mask: Optional[jnp.ndarray] = None,
        decoder_position_ids: Optional[jnp.ndarray] = None,
        past_key_values: dict = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        train: bool = False,
        params: dict = None,
        dropout_rng: PRNGKey = None,
    ):
        # 解析方法，负责解码任务
        # decoder_input_ids: 解码器输入的 token IDs
        # encoder_outputs: 编码器的输出
        # encoder_attention_mask: 编码器的注意力掩码，可选
        # decoder_attention_mask: 解码器的注意力掩码，可选
        # decoder_position_ids: 解码器位置 IDs，可选
        # past_key_values: 用于存储历史键值的字典，可选
        # output_attentions: 是否输出注意力权重，可选
        # output_hidden_states: 是否输出隐藏状态，可选
        # return_dict: 是否以字典形式返回结果，可选
        # train: 是否处于训练模式，默认为 False
        # params: 模型参数，字典类型，可选
        # dropout_rng: 随机数生成器用于 dropout 操作，可选

    def generate(
        self,
        input_features,
        generation_config=None,
        logits_processor=None,
        return_timestamps=None,
        task=None,
        language=None,
        is_multilingual=None,
        **kwargs,
    ):
        # 生成方法，用于生成文本或其他任务的输出
        # input_features: 输入特征，通常是 token IDs 或其他输入形式
        # generation_config: 生成配置，控制生成行为，可选
        # logits_processor: logits 处理器，用于调整输出 logits，可选
        # return_timestamps: 是否返回时间戳，可选
        # task: 任务类型，例如生成文本的特定任务，可选
        # language: 生成文本的语言，可选
        # is_multilingual: 是否多语言生成，可选
        # **kwargs: 其他可能的关键字参数
    ):
        # 如果 generation_config 参数为 None，则使用类中的默认生成配置
        if generation_config is None:
            generation_config = self.generation_config

        # 如果 return_timestamps 参数不为 None，则设置生成配置中的 return_timestamps 属性
        if return_timestamps is not None:
            generation_config.return_timestamps = return_timestamps

        # 如果 task 参数不为 None，则设置生成配置中的 task 属性
        if task is not None:
            generation_config.task = task

        # 如果 is_multilingual 参数不为 None，则设置生成配置中的 is_multilingual 属性
        if is_multilingual is not None:
            generation_config.is_multilingual = is_multilingual

        # 如果 language 参数不为 None，则设置生成配置中的 language 属性
        if language is not None:
            generation_config.language = language

        # 如果 kwargs 参数不为 None 并且包含 "decoder_input_ids" 键，则获取其长度作为 decoder_input_length
        # 否则，将 decoder_input_length 设置为 1
        if kwargs is not None and "decoder_input_ids" in kwargs:
            decoder_input_length = len(kwargs["decoder_input_ids"])
        else:
            decoder_input_length = 1

        # 初始化强制解码器输入列表
        forced_decoder_ids = []

        # 如果生成配置中具有 "is_multilingual" 属性且为 True，则处理多语言设置
        if hasattr(generation_config, "is_multilingual") and generation_config.is_multilingual:
            # 如果生成配置中具有 "language" 属性，则根据语言映射添加到强制解码器输入列表
            if hasattr(generation_config, "language"):
                forced_decoder_ids.append((1, generation_config.lang_to_id[generation_config.language]))
            else:
                # 否则，添加一个空语言 ID
                forced_decoder_ids.append((1, None))

            # 如果生成配置中具有 "task" 属性，则根据任务映射添加到强制解码器输入列表
            if hasattr(generation_config, "task"):
                forced_decoder_ids.append((2, generation_config.task_to_id[generation_config.task]))
            else:
                # 否则，默认添加一个 "transcribe" 任务 ID
                forced_decoder_ids.append((2, generation_config.task_to_id["transcribe"]))

        # 如果生成配置中具有 "return_timestamps" 属性且为 True，或者 return_timestamps 参数为 True，则配置 logits_processor
        if (
            hasattr(generation_config, "return_timestamps") and generation_config.return_timestamps
        ) or return_timestamps:
            logits_processor = [
                FlaxWhisperTimeStampLogitsProcessor(generation_config, self.config, decoder_input_length)
            ]
        else:
            # 否则，如果存在强制解码器输入且最后一个元素不等于 no_timestamps_token_id，则添加一个默认的时间戳标记
            if forced_decoder_ids and forced_decoder_ids[-1][0] != generation_config.no_timestamps_token_id:
                idx = forced_decoder_ids[-1][0] + 1 if forced_decoder_ids else 1
                forced_decoder_ids.append((idx, generation_config.no_timestamps_token_id))

        # 如果强制解码器输入列表长度大于 0，则将其设置到生成配置中的 forced_decoder_ids 属性
        if len(forced_decoder_ids) > 0:
            generation_config.forced_decoder_ids = forced_decoder_ids

        # 调用父类的 generate 方法，生成文本序列
        return super().generate(
            input_features,
            generation_config,
            logits_processor=logits_processor,
            **kwargs,
        )

    # 准备生成输入的方法，根据给定参数设置解码器输入
    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        max_length,
        attention_mask: Optional[jax.Array] = None,
        decoder_attention_mask: Optional[jax.Array] = None,
        encoder_outputs=None,
        **kwargs,
    ):
        # initializing the cache
        # 解码器输入的批量大小和序列长度
        batch_size, seq_length = decoder_input_ids.shape

        # 使用 self.init_cache 方法初始化过去的键值对
        past_key_values = self.init_cache(batch_size, max_length, encoder_outputs)

        # 注意：通常需要将注意力掩码中超出 input_ids.shape[-1] 和小于 cache_length 的位置设置为 0，
        # 但由于解码器使用因果掩码，这些位置已经被掩码处理。
        # 因此，我们可以在此处创建一个静态的注意力掩码，对编译更有效。
        extended_attention_mask = jnp.ones((batch_size, max_length), dtype="i4")

        # 如果存在解码器的注意力掩码，则计算位置 ids
        if decoder_attention_mask is not None:
            position_ids = decoder_attention_mask.cumsum(-1) - 1
            # 使用 lax.dynamic_update_slice 将 decoder_attention_mask 更新到 extended_attention_mask 中
            extended_attention_mask = lax.dynamic_update_slice(extended_attention_mask, decoder_attention_mask, (0, 0))
        else:
            # 否则，使用广播方式创建位置 ids
            position_ids = jnp.broadcast_to(jnp.arange(seq_length, dtype="i4")[None, :], (batch_size, seq_length))

        # 返回更新后的字典，包括过去的键值对、编码器输出、编码器注意力掩码、解码器注意力掩码和位置 ids
        return {
            "past_key_values": past_key_values,
            "encoder_outputs": encoder_outputs,
            "encoder_attention_mask": attention_mask,
            "decoder_attention_mask": extended_attention_mask,
            "decoder_position_ids": position_ids,
        }

    def update_inputs_for_generation(self, model_outputs, model_kwargs):
        # 将模型输出中的过去的键值对更新到模型参数中
        model_kwargs["past_key_values"] = model_outputs.past_key_values
        # 更新解码器位置 ids 以便生成下一个词
        model_kwargs["decoder_position_ids"] = model_kwargs["decoder_position_ids"][:, -1:] + 1
        return model_kwargs
# 覆盖并修改 FlaxWhisperForConditionalGeneration 类的文档字符串，添加了条件生成的例子和返回说明
FLAX_WHISPER_CONDITIONAL_GENERATION_DOCSTRING = r"""
    Returns:

    Transcription example:

    ```python
    >>> from transformers import WhisperProcessor, FlaxWhisperForConditionalGeneration
    >>> from datasets import load_dataset

    >>> processor = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")
    >>> model = FlaxWhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en", from_pt=True)
    >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    >>> inputs = processor(ds[0]["audio"]["array"], return_tensors="np")
    >>> input_features = inputs.input_features
    >>> generated_ids = model.generate(input_ids=input_features)
    >>> transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    >>> transcription
    ' Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel.'
    ```
"""

# 调用函数覆盖并修改 FlaxWhisperForConditionalGeneration 类的文档字符串
overwrite_call_docstring(
    FlaxWhisperForConditionalGeneration, WHISPER_INPUTS_DOCSTRING + FLAX_WHISPER_CONDITIONAL_GENERATION_DOCSTRING
)

# 追加并替换 FlaxWhisperForConditionalGeneration 类的返回文档字符串
append_replace_return_docstrings(
    FlaxWhisperForConditionalGeneration, output_type=FlaxSeq2SeqLMOutput, config_class=_CONFIG_FOR_DOC
)


class FlaxWhisperForAudioClassificationModule(nn.Module):
    # 定义 FlaxWhisperForAudioClassificationModule 类
    config: WhisperConfig
    dtype: jnp.dtype = jnp.float32
    gradient_checkpointing: bool = False

    def setup(self) -> None:
        # 设置函数，初始化模型组件
        self.encoder = FlaxWhisperEncoder(
            config=self.config, dtype=self.dtype, gradient_checkpointing=self.gradient_checkpointing
        )
        # 设置编码器组件
        self.config.is_encoder_decoder = False
        # 确定非编码器解码器模式
        num_layers = self.config.num_hidden_layers + 1
        # 计算层数
        if self.config.use_weighted_layer_sum:
            # 如果使用加权层求和
            self.layer_weights = jnp.repeat(1 / num_layers, num_layers)
            # 设置层权重
        self.projector = nn.Dense(self.config.classifier_proj_size, dtype=self.dtype)
        # 定义分类投影器
        self.classifier = nn.Dense(self.config.num_labels, dtype=self.dtype)
        # 定义分类器

    def __call__(
        self,
        input_features,
        encoder_outputs=None,
        output_attentions=None,
        output_hidden_states: bool = True,
        return_dict: bool = True,
        # 重载调用操作，接受输入特征等参数，返回字典格式结果
        ):
        # 如果未指定输出注意力的设置，则使用配置中的默认值
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 如果未指定输出隐藏状态的设置，则使用配置中的默认值
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 如果未指定返回字典的设置，则使用配置中的默认值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 如果编码器输出为空，则调用编码器进行编码
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_features,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        # 如果配置中使用加权层求和
        if self.config.use_weighted_layer_sum:
            # 将编码器输出堆叠起来形成张量
            hidden_states = jnp.stack(encoder_outputs, axis=1)
            # 对层权重进行 softmax 归一化
            norm_weights = jax.nn.softmax(self.layer_weights, axis=-1)
            # 加权求和后的隐藏状态
            hidden_states = jnp.sum(hidden_states * jnp.reshape(norm_weights, [-1, 1, 1]), axis=1)
        else:
            # 否则直接使用编码器的第一个输出作为隐藏状态
            hidden_states = encoder_outputs[0]

        # 将隐藏状态投影到新的空间
        hidden_states = self.projector(hidden_states)
        # 对隐藏状态进行平均池化，生成池化输出
        pooled_output = jnp.mean(hidden_states, axis=1)

        # 将池化输出送入分类器得到预测 logits
        logits = self.classifier(pooled_output)

        # 如果不需要返回字典形式的输出
        if not return_dict:
            # 返回 logits 和编码器的其他输出状态
            return (logits,) + encoder_outputs[1:]

        # 否则以 FlaxSequenceClassifierOutput 的形式返回结果
        return FlaxSequenceClassifierOutput(
            logits=logits,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
@add_start_docstrings("The Whisper Model with an audio classification head on top.", WHISPER_START_DOCSTRING)
class FlaxWhisperForAudioClassification(FlaxWhisperPreTrainedModel):
    module_class = FlaxWhisperForAudioClassificationModule  # 设置模型的模块类为FlaxWhisperForAudioClassificationModule
    dtype: jnp.dtype = jnp.float32  # 设置数据类型为32位浮点数

    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict = None) -> FrozenDict:
        # 初始化输入张量
        input_features = jnp.zeros(input_shape, dtype="f4")  # 创建全零的输入特征张量，数据类型为32位浮点数
        input_features = input_features.at[(..., -1)].set(self.config.eos_token_id)  # 将输入张量的最后一个位置设置为配置中的eos_token_id

        params_rng, dropout_rng = jax.random.split(rng)  # 使用随机数生成器rng分割得到参数随机数生成器和dropout随机数生成器
        rngs = {"params": params_rng, "dropout": dropout_rng}  # 创建随机数生成器字典

        random_params = self.module.init(  # 使用模块的初始化方法初始化随机参数
            rngs,
            input_features=input_features,
        )["params"]

        if params is not None:
            random_params = flatten_dict(unfreeze(random_params))  # 展开和解冻随机参数
            params = flatten_dict(unfreeze(params))  # 展开和解冻给定参数
            for missing_key in self._missing_keys:
                params[missing_key] = random_params[missing_key]  # 将缺失的键添加到参数字典中
            self._missing_keys = set()  # 清空缺失键集合
            return freeze(unflatten_dict(params))  # 冻结并返回重构的参数字典
        else:
            return random_params  # 返回随机初始化的参数

    @add_start_docstrings_to_model_forward(WHISPER_INPUTS_DOCSTRING)
    def __call__(
        self,
        input_features: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        train: bool = False,
        params: dict = None,
        dropout_rng: PRNGKey = None,
        **kwargs,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions  # 根据参数设置是否输出注意力权重
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states  # 根据参数设置是否输出隐藏状态
        )
        return_dict = return_dict if return_dict is not None else self.config.return_dict  # 根据参数设置是否返回字典形式的结果

        # 如果需要处理任何随机数生成器
        rngs = {}
        if dropout_rng is not None:
            rngs["dropout"] = dropout_rng  # 将dropout随机数生成器添加到随机数生成器字典中

        return self.module.apply(  # 使用模块的应用方法进行前向传播
            {"params": params or self.params},  # 使用给定参数或默认参数
            input_features=jnp.array(input_features, dtype="f4"),  # 将输入特征转换为32位浮点数的JAX数组
            output_attentions=output_attentions,  # 输出注意力权重
            output_hidden_states=output_hidden_states,  # 输出隐藏状态
            return_dict=return_dict,  # 返回字典形式的结果
            rngs=rngs,  # 随机数生成器字典
        )
    # 加载数据集 "google/fleurs" 的验证集，使用流式数据加载方式
    ds = load_dataset("google/fleurs", "all", split="validation", streaming=True)
    
    # 从数据集中获取下一个样本
    sample = next(iter(ds))
    
    # 使用特征提取器从音频样本中提取特征，并返回NumPy数组格式的张量
    inputs = feature_extractor(
        sample["audio"]["array"],  # 音频数据数组
        sampling_rate=sample["audio"]["sampling_rate"],  # 音频采样率
        return_tensors="np"  # 返回NumPy数组格式的张量
    )
    
    # 获取输入特征
    input_features = inputs.input_features
    
    # 使用模型对输入特征进行推理，获取预测的 logits（未归一化的预测分数）
    logits = model(input_features).logits
    
    # 根据 logits 计算预测的类别编号
    predicted_class_ids = jnp.argmax(logits).item()
    
    # 根据模型配置中的 id2label 映射，获取预测类别的标签名
    predicted_label = model.config.id2label[predicted_class_ids]
    
    # 返回预测的标签名
    predicted_label
"""
调用函数 `overwrite_call_docstring`，用于覆盖指定类的文档字符串，结合给定的文档字符串常量。
"""
overwrite_call_docstring(
    FlaxWhisperForAudioClassification, WHISPER_INPUTS_DOCSTRING + FLAX_WHISPER_AUDIO_CLASSIFICATION_DOCSTRING
)

"""
调用函数 `append_replace_return_docstrings`，用于在指定类的文档字符串末尾追加并替换返回值的描述信息。
"""
append_replace_return_docstrings(
    FlaxWhisperForAudioClassification, output_type=FlaxSequenceClassifierOutput, config_class=_CONFIG_FOR_DOC
)
```