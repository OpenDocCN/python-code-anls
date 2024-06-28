# `.\models\roberta\modeling_flax_roberta.py`

```py
# 声明一个长字符串作为模型文档字符串的一部分，用于生成 ROBERTA_START_DOCSTRING
ROBERTA_START_DOCSTRING = r"""

    This model inherits from [`FlaxPreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading, saving and converting weights from PyTorch models)

    This model is also a
    [flax.linen.Module](https://flax.readthedocs.io/en/latest/api_reference/flax.linen/module.html) subclass. Use it as
    a regular Flax linen Module and refer to the Flax documentation for all matter related to general usage and
    behavior.

    Finally, this model supports inherent JAX features such as:

    - [Just-In-Time (JIT) compilation](https://jax.readthedocs.io/en/latest/jax.html#just-in-time-compilation-jit)
    - [Automatic Differentiation](https://jax.readthedocs.io/en/latest/jax.html#automatic-differentiation)
    - [Vectorization](https://jax.readthedocs.io/en/latest/jax.html#vectorization-vmap)
    - [Parallelization](https://jax.readthedocs.io/en/latest/jax.html#parallelization-pmap)

    Parameters:
        config ([`RobertaConfig`]): Model configuration class with all the parameters of the
            model. Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~FlaxPreTrainedModel.from_pretrained`] method to load the model weights.
# 定义了一个长字符串，用于文档化 RobertaEmbeddings 类的输入参数及其说明
ROBERTA_INPUTS_DOCSTRING = r"""
Args:
    input_ids (`numpy.ndarray` of shape `({0})`):
        Indices of input sequence tokens in the vocabulary.
        
        Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
        [`PreTrainedTokenizer.__call__`] for details.
        
        [What are input IDs?](../glossary#input-ids)
    attention_mask (`numpy.ndarray` of shape `({0})`, *optional*):
        Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
        
        - 1 for tokens that are **not masked**,
        - 0 for tokens that are **masked**.
        
        [What are attention masks?](../glossary#attention-mask)
    token_type_ids (`numpy.ndarray` of shape `({0})`, *optional*):
        Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0,
        1]`:
        
        - 0 corresponds to a *sentence A* token,
        - 1 corresponds to a *sentence B* token.
        
        [What are token type IDs?](../glossary#token-type-ids)
    position_ids (`numpy.ndarray` of shape `({0})`, *optional*):
        Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
        config.max_position_embeddings - 1]`.
    head_mask (`numpy.ndarray` of shape `({0})`, `optional):
        Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:
        
        - 1 indicates the head is **not masked**,
        - 0 indicates the head is **masked**.
        
    return_dict (`bool`, *optional*):
        Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""

# 定义了一个类 FlaxRobertaEmbeddings，继承自 nn.Module
class FlaxRobertaEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    # 类属性 config，指定为 RobertaConfig 类型
    config: RobertaConfig
    # 类属性 dtype，指定为 jnp.float32 类型，用于计算的数据类型
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation
    def setup(self):
        # 初始化词嵌入层，用于将词汇 ID 映射到隐藏大小的向量空间
        self.word_embeddings = nn.Embed(
            self.config.vocab_size,
            self.config.hidden_size,
            embedding_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
            dtype=self.dtype,
        )
        # 初始化位置嵌入层，用于将位置 ID 映射到隐藏大小的向量空间
        self.position_embeddings = nn.Embed(
            self.config.max_position_embeddings,
            self.config.hidden_size,
            embedding_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
            dtype=self.dtype,
        )
        # 初始化类型嵌入层，用于将类型 ID 映射到隐藏大小的向量空间
        self.token_type_embeddings = nn.Embed(
            self.config.type_vocab_size,
            self.config.hidden_size,
            embedding_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
            dtype=self.dtype,
        )
        # 初始化层归一化模块，使用给定的 epsilon 参数
        self.LayerNorm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)
        # 初始化 dropout 模块，使用给定的 dropout 率
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout_prob)

    def __call__(self, input_ids, token_type_ids, position_ids, attention_mask, deterministic: bool = True):
        # 将输入的词汇 ID 转换为词嵌入向量
        inputs_embeds = self.word_embeddings(input_ids.astype("i4"))
        # 将位置 ID 转换为位置嵌入向量
        position_embeds = self.position_embeddings(position_ids.astype("i4"))
        # 将类型 ID 转换为类型嵌入向量
        token_type_embeddings = self.token_type_embeddings(token_type_ids.astype("i4"))

        # 合并所有嵌入向量
        hidden_states = inputs_embeds + token_type_embeddings + position_embeds

        # 应用层归一化
        hidden_states = self.LayerNorm(hidden_states)
        # 应用 dropout
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        # 返回最终的隐藏状态表示
        return hidden_states
# 从 transformers.models.bert.modeling_flax_bert.FlaxBertSelfAttention 复制过来的类，修改了 Bert -> Roberta
class FlaxRobertaSelfAttention(nn.Module):
    # 类的构造函数中声明了配置参数 config，以及两个额外的类属性：causal 表示是否使用因果（causal）注意力，dtype 表示计算过程中使用的数据类型，默认为 jnp.float32
    config: RobertaConfig
    causal: bool = False
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        # 计算每个注意力头的维度
        self.head_dim = self.config.hidden_size // self.config.num_attention_heads
        # 检查 hidden_size 是否能被 num_attention_heads 整除，如果不能则抛出 ValueError 异常
        if self.config.hidden_size % self.config.num_attention_heads != 0:
            raise ValueError(
                "`config.hidden_size`: {self.config.hidden_size} has to be a multiple of `config.num_attention_heads` "
                "                   : {self.config.num_attention_heads}"
            )

        # 创建 query、key 和 value 网络层，分别初始化为指定的 hidden_size，并使用 normal 分布初始化权重
        self.query = nn.Dense(
            self.config.hidden_size,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
        )
        self.key = nn.Dense(
            self.config.hidden_size,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
        )
        self.value = nn.Dense(
            self.config.hidden_size,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
        )

        # 如果设置了 causal=True，则创建一个因果掩码，用于在自注意力机制中排除未来信息
        if self.causal:
            self.causal_mask = make_causal_mask(
                jnp.ones((1, self.config.max_position_embeddings), dtype="bool"), dtype="bool"
            )

    # 将隐藏状态分割成多个注意力头，返回的形状为 (batch_size, seq_length, num_attention_heads, head_dim)
    def _split_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.config.num_attention_heads, self.head_dim))

    # 合并多个注意力头到原始隐藏状态，返回的形状为 (batch_size, seq_length, hidden_size)
    def _merge_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.config.hidden_size,))

    @nn.compact
    # 从 transformers.models.bart.modeling_flax_bart.FlaxBartAttention._concatenate_to_cache 复制过来的方法
    def _concatenate_to_cache(self, key, value, query, attention_mask):
        """
        This function takes projected key, value states from a single input token and concatenates the states to cached
        states from previous steps. This function is slightly adapted from the official Flax repository:
        https://github.com/google/flax/blob/491ce18759622506588784b4fca0e4bf05f8c8cd/flax/linen/attention.py#L252
        """
        # 检测是否已经初始化缓存数据
        is_initialized = self.has_variable("cache", "cached_key")
        # 获取或创建缓存的键和值，并初始化为零矩阵，与输入的形状和数据类型相匹配
        cached_key = self.variable("cache", "cached_key", jnp.zeros, key.shape, key.dtype)
        cached_value = self.variable("cache", "cached_value", jnp.zeros, value.shape, value.dtype)
        # 获取或创建缓存索引，初始化为整数0
        cache_index = self.variable("cache", "cache_index", lambda: jnp.array(0, dtype=jnp.int32))

        if is_initialized:
            # 获取批次维度、最大长度、头数和每头深度等维度信息
            *batch_dims, max_length, num_heads, depth_per_head = cached_key.value.shape
            # 使用新的一维空间片段更新键和值的缓存
            cur_index = cache_index.value
            indices = (0,) * len(batch_dims) + (cur_index, 0, 0)
            key = lax.dynamic_update_slice(cached_key.value, key, indices)
            value = lax.dynamic_update_slice(cached_value.value, value, indices)
            # 更新缓存的键和值
            cached_key.value = key
            cached_value.value = value
            # 更新缓存索引，增加已更新的缓存向量数量
            num_updated_cache_vectors = query.shape[1]
            cache_index.value = cache_index.value + num_updated_cache_vectors
            # 为缓存的解码器自注意力生成因果掩码：单个查询位置应只关注已生成和缓存的键位置，而不是剩余的零元素。
            pad_mask = jnp.broadcast_to(
                jnp.arange(max_length) < cur_index + num_updated_cache_vectors,
                tuple(batch_dims) + (1, num_updated_cache_vectors, max_length),
            )
            # 合并因果掩码和输入的注意力掩码
            attention_mask = combine_masks(pad_mask, attention_mask)
        
        # 返回更新后的键、值和注意力掩码
        return key, value, attention_mask
# 定义一个用于Roberta模型自注意力层输出的类
class FlaxRobertaSelfOutput(nn.Module):
    config: RobertaConfig  # Roberta模型的配置信息
    dtype: jnp.dtype = jnp.float32  # 计算时使用的数据类型

    def setup(self):
        # 定义一个全连接层，将隐藏状态映射到隐藏大小，使用正态分布初始化权重
        self.dense = nn.Dense(
            self.config.hidden_size,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            dtype=self.dtype,
        )
        # LayerNorm层，用于归一化隐藏状态
        self.LayerNorm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)
        # Dropout层，用于随机失活以防止过拟合
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout_prob)

    def __call__(self, hidden_states, input_tensor, deterministic: bool = True):
        # 全连接层处理隐藏状态
        hidden_states = self.dense(hidden_states)
        # 使用Dropout层进行随机失活
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        # LayerNorm归一化并与输入张量相加
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


# 定义一个用于Roberta模型注意力机制的类
class FlaxRobertaAttention(nn.Module):
    config: RobertaConfig  # Roberta模型的配置信息
    causal: bool = False  # 是否是因果（causal）注意力
    dtype: jnp.dtype = jnp.float32  # 计算时使用的数据类型

    def setup(self):
        # 定义自注意力层
        self.self = FlaxRobertaSelfAttention(self.config, causal=self.causal, dtype=self.dtype)
        # 定义自注意力层输出层
        self.output = FlaxRobertaSelfOutput(self.config, dtype=self.dtype)

    def __call__(
        self,
        hidden_states,
        attention_mask,
        layer_head_mask,
        key_value_states=None,
        init_cache=False,
        deterministic=True,
        output_attentions: bool = False,
    ):
        # 调用自注意力层进行处理
        attn_outputs = self.self(
            hidden_states,
            attention_mask,
            layer_head_mask=layer_head_mask,
            key_value_states=key_value_states,
            init_cache=init_cache,
            deterministic=deterministic,
            output_attentions=output_attentions,
        )
        # 取得自注意力层的输出
        attn_output = attn_outputs[0]
        # 使用自注意力层输出层处理自注意力层的输出和隐藏状态
        hidden_states = self.output(attn_output, hidden_states, deterministic=deterministic)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_outputs[1],)  # 如果需要输出注意力权重，则加入到输出中

        return outputs


# 定义一个用于Roberta模型中间层的类
class FlaxRobertaIntermediate(nn.Module):
    config: RobertaConfig  # Roberta模型的配置信息
    dtype: jnp.dtype = jnp.float32  # 计算时使用的数据类型

    def setup(self):
        # 定义一个全连接层，将隐藏状态映射到中间大小，使用正态分布初始化权重
        self.dense = nn.Dense(
            self.config.intermediate_size,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            dtype=self.dtype,
        )
        # 激活函数，根据配置选择激活函数类型
        self.activation = ACT2FN[self.config.hidden_act]
    # 定义类的方法 __call__，用于对输入的 hidden_states 进行处理并返回结果
    def __call__(self, hidden_states):
        # 将输入的 hidden_states 通过全连接层 dense 进行线性变换
        hidden_states = self.dense(hidden_states)
        # 将线性变换后的结果通过激活函数 activation 进行非线性变换
        hidden_states = self.activation(hidden_states)
        # 返回经过线性变换和激活函数处理后的 hidden_states 结果
        return hidden_states
# 从 transformers.models.bert.modeling_flax_bert.FlaxBertOutput 复制并将 Bert 替换为 Roberta
class FlaxRobertaOutput(nn.Module):
    config: RobertaConfig  # Roberta 模型的配置对象
    dtype: jnp.dtype = jnp.float32  # 计算过程中使用的数据类型

    def setup(self):
        # 定义全连接层，将隐藏状态映射到指定大小的输出空间
        self.dense = nn.Dense(
            self.config.hidden_size,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),  # 使用正态分布初始化权重
            dtype=self.dtype,
        )
        # 定义 Dropout 层，用于随机屏蔽神经元，防止过拟合
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout_prob)
        # 定义 LayerNorm 层，用于归一化隐藏状态
        self.LayerNorm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)

    def __call__(self, hidden_states, attention_output, deterministic: bool = True):
        # 全连接层处理隐藏状态
        hidden_states = self.dense(hidden_states)
        # 应用 Dropout 层
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        # 应用 LayerNorm 层，并将注意力输出加到处理后的隐藏状态上
        hidden_states = self.LayerNorm(hidden_states + attention_output)
        return hidden_states


# 从 transformers.models.bert.modeling_flax_bert.FlaxBertLayer 复制并将 Bert 替换为 Roberta
class FlaxRobertaLayer(nn.Module):
    config: RobertaConfig  # Roberta 模型的配置对象
    dtype: jnp.dtype = jnp.float32  # 计算过程中使用的数据类型

    def setup(self):
        # 定义 Roberta 自注意力层
        self.attention = FlaxRobertaAttention(self.config, causal=self.config.is_decoder, dtype=self.dtype)
        # 定义 Roberta 中间层
        self.intermediate = FlaxRobertaIntermediate(self.config, dtype=self.dtype)
        # 定义 Roberta 输出层
        self.output = FlaxRobertaOutput(self.config, dtype=self.dtype)
        # 如果配置中包含跨注意力机制，则定义 Roberta 交叉注意力层
        if self.config.add_cross_attention:
            self.crossattention = FlaxRobertaAttention(self.config, causal=False, dtype=self.dtype)

    def __call__(
        self,
        hidden_states,
        attention_mask,
        layer_head_mask,
        encoder_hidden_states: Optional[jnp.ndarray] = None,
        encoder_attention_mask: Optional[jnp.ndarray] = None,
        init_cache: bool = False,
        deterministic: bool = True,
        output_attentions: bool = False,
        # Self Attention
        # 使用 self.attention 方法进行自注意力计算，传入隐藏状态、注意力掩码等参数
        attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            layer_head_mask=layer_head_mask,
            init_cache=init_cache,
            deterministic=deterministic,
            output_attentions=output_attentions,
        )
        # 获取注意力计算的输出
        attention_output = attention_outputs[0]

        # Cross-Attention Block
        # 如果存在编码器的隐藏状态，执行交叉注意力计算
        if encoder_hidden_states is not None:
            # 使用 self.crossattention 方法进行交叉注意力计算，传入自注意力输出、编码器注意力掩码、编码器隐藏状态等参数
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask=encoder_attention_mask,
                layer_head_mask=layer_head_mask,
                key_value_states=encoder_hidden_states,
                deterministic=deterministic,
                output_attentions=output_attentions,
            )
            # 获取交叉注意力计算的输出作为最终的注意力输出
            attention_output = cross_attention_outputs[0]

        # 使用 self.intermediate 方法进行隐藏状态的中间层处理
        hidden_states = self.intermediate(attention_output)
        # 使用 self.output 方法生成最终的输出隐藏状态
        hidden_states = self.output(hidden_states, attention_output, deterministic=deterministic)

        # 输出为隐藏状态的元组
        outputs = (hidden_states,)

        # 如果需要输出注意力权重信息
        if output_attentions:
            # 将自注意力的注意力权重添加到输出元组中
            outputs += (attention_outputs[1],)
            # 如果存在编码器的隐藏状态，将交叉注意力的注意力权重也添加到输出元组中
            if encoder_hidden_states is not None:
                outputs += (cross_attention_outputs[1],)
        # 返回最终的输出元组
        return outputs
# 从 transformers.models.bert.modeling_flax_bert.FlaxBertLayerCollection 复制代码，并将 Bert 替换为 Roberta
class FlaxRobertaLayerCollection(nn.Module):
    config: RobertaConfig  # 使用 RobertaConfig 类型的配置
    dtype: jnp.dtype = jnp.float32  # 计算过程中使用的数据类型
    gradient_checkpointing: bool = False  # 是否使用梯度检查点技术

    def setup(self):
        if self.gradient_checkpointing:
            # 如果启用梯度检查点技术，则使用 remat 函数重新定义 FlaxRobertaLayer 类
            FlaxRobertaCheckpointLayer = remat(FlaxRobertaLayer, static_argnums=(5, 6, 7))
            # 创建包含梯度检查点层的列表，每层以字符串形式命名
            self.layers = [
                FlaxRobertaCheckpointLayer(self.config, name=str(i), dtype=self.dtype)
                for i in range(self.config.num_hidden_layers)
            ]
        else:
            # 否则，创建普通的 FlaxRobertaLayer 层列表，每层以字符串形式命名
            self.layers = [
                FlaxRobertaLayer(self.config, name=str(i), dtype=self.dtype)
                for i in range(self.config.num_hidden_layers)
            ]

    def __call__(
        self,
        hidden_states,
        attention_mask,
        head_mask,
        encoder_hidden_states: Optional[jnp.ndarray] = None,
        encoder_attention_mask: Optional[jnp.ndarray] = None,
        init_cache: bool = False,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        ):
            # 初始化空元组或 None，根据 output_attentions 的值确定是否返回注意力信息
            all_attentions = () if output_attentions else None
            # 初始化空元组或 None，根据 output_hidden_states 的值确定是否返回隐藏状态信息
            all_hidden_states = () if output_hidden_states else None
            # 初始化空元组或 None，根据 output_attentions 和 encoder_hidden_states 的值确定是否返回交叉注意力信息
            all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None

            # 检查 head_mask 是否正确指定了每层的屏蔽信息
            if head_mask is not None:
                if head_mask.shape[0] != (len(self.layers)):
                    # 抛出异常，提示 head_mask 应该对应于 self.layers 的层数
                    raise ValueError(
                        f"The head_mask should be specified for {len(self.layers)} layers, but it is for "
                        f"{head_mask.shape[0]}."
                    )

            # 遍历所有层，进行前向传播计算
            for i, layer in enumerate(self.layers):
                # 如果输出隐藏状态信息，则将当前隐藏状态加入到 all_hidden_states 中
                if output_hidden_states:
                    all_hidden_states += (hidden_states,)

                # 调用当前层的前向传播函数
                layer_outputs = layer(
                    hidden_states,
                    attention_mask,
                    head_mask[i] if head_mask is not None else None,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    init_cache,
                    deterministic,
                    output_attentions,
                )

                # 更新当前隐藏状态为当前层的输出的第一个元素（通常是隐藏状态）
                hidden_states = layer_outputs[0]

                # 如果输出注意力信息，则将当前层的注意力加入到 all_attentions 中
                if output_attentions:
                    all_attentions += (layer_outputs[1],)

                    # 如果同时存在 encoder_hidden_states，则将当前层的交叉注意力加入到 all_cross_attentions 中
                    if encoder_hidden_states is not None:
                        all_cross_attentions += (layer_outputs[2],)

            # 如果输出隐藏状态信息，则将最终的隐藏状态加入到 all_hidden_states 中
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            # 整理最终的输出结果
            outputs = (hidden_states, all_hidden_states, all_attentions, all_cross_attentions)

            # 如果 return_dict 为 False，则返回 outputs 中非 None 的部分作为元组
            if not return_dict:
                return tuple(v for v in outputs if v is not None)

            # 如果 return_dict 为 True，则将输出整理成 FlaxBaseModelOutputWithPastAndCrossAttentions 类的对象返回
            return FlaxBaseModelOutputWithPastAndCrossAttentions(
                last_hidden_state=hidden_states,
                hidden_states=all_hidden_states,
                attentions=all_attentions,
                cross_attentions=all_cross_attentions,
            )
# 从 transformers.models.bert.modeling_flax_bert.FlaxBertEncoder 复制并替换为 Roberta
class FlaxRobertaEncoder(nn.Module):
    config: RobertaConfig  # 使用 RobertaConfig 类型的配置信息
    dtype: jnp.dtype = jnp.float32  # 计算时的数据类型为 jnp.float32
    gradient_checkpointing: bool = False  # 是否使用梯度检查点

    def setup(self):
        self.layer = FlaxRobertaLayerCollection(  # 初始化 FlaxRobertaLayerCollection 实例
            self.config,  # 使用给定的配置信息
            dtype=self.dtype,  # 使用指定的数据类型
            gradient_checkpointing=self.gradient_checkpointing,  # 梯度检查点设置
        )

    def __call__(  # 定义对象调用时的行为
        self,
        hidden_states,  # 输入的隐藏状态张量
        attention_mask,  # 注意力掩码张量
        head_mask,  # 头部掩码张量
        encoder_hidden_states: Optional[jnp.ndarray] = None,  # 编码器隐藏状态（可选）
        encoder_attention_mask: Optional[jnp.ndarray] = None,  # 编码器注意力掩码（可选）
        init_cache: bool = False,  # 是否初始化缓存
        deterministic: bool = True,  # 是否确定性计算
        output_attentions: bool = False,  # 是否输出注意力权重
        output_hidden_states: bool = False,  # 是否输出隐藏状态
        return_dict: bool = True,  # 是否以字典形式返回结果
    ):
        return self.layer(  # 调用 FlaxRobertaLayerCollection 的前向传播
            hidden_states,
            attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            init_cache=init_cache,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


# 从 transformers.models.bert.modeling_flax_bert.FlaxBertPooler 复制并替换为 Roberta
class FlaxRobertaPooler(nn.Module):
    config: RobertaConfig  # 使用 RobertaConfig 类型的配置信息
    dtype: jnp.dtype = jnp.float32  # 计算时的数据类型为 jnp.float32

    def setup(self):
        self.dense = nn.Dense(  # 初始化密集连接层
            self.config.hidden_size,  # 输出大小为配置中的隐藏大小
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),  # 使用正态分布初始化权重
            dtype=self.dtype,  # 使用指定的数据类型
        )

    def __call__(self, hidden_states):
        cls_hidden_state = hidden_states[:, 0]  # 取出每个样本的第一个位置的隐藏状态
        cls_hidden_state = self.dense(cls_hidden_state)  # 将其通过密集连接层
        return nn.tanh(cls_hidden_state)  # 返回经过 tanh 激活函数后的结果


class FlaxRobertaLMHead(nn.Module):
    config: RobertaConfig  # 使用 RobertaConfig 类型的配置信息
    dtype: jnp.dtype = jnp.float32  # 计算时的数据类型为 jnp.float32
    bias_init: Callable[..., np.ndarray] = jax.nn.initializers.zeros  # 偏置初始化函数为零初始化

    def setup(self):
        self.dense = nn.Dense(  # 初始化第一个密集连接层
            self.config.hidden_size,  # 输出大小为配置中的隐藏大小
            dtype=self.dtype,  # 使用指定的数据类型
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),  # 使用正态分布初始化权重
        )
        self.layer_norm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)  # 初始化 LayerNorm 层
        self.decoder = nn.Dense(  # 初始化解码器密集连接层
            self.config.vocab_size,  # 输出大小为词汇表大小
            dtype=self.dtype,  # 使用指定的数据类型
            use_bias=False,  # 不使用偏置
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),  # 使用正态分布初始化权重
        )
        self.bias = self.param("bias", self.bias_init, (self.config.vocab_size,))  # 初始化偏置参数
    # 定义类的调用方法，接受隐藏状态和可选的共享嵌入作为输入
    def __call__(self, hidden_states, shared_embedding=None):
        # 使用全连接层对隐藏状态进行线性变换
        hidden_states = self.dense(hidden_states)
        # 应用 GELU 激活函数到线性变换后的隐藏状态
        hidden_states = ACT2FN["gelu"](hidden_states)
        # 对激活后的隐藏状态进行层归一化
        hidden_states = self.layer_norm(hidden_states)

        # 如果提供了共享的嵌入向量，使用解码器对隐藏状态进行处理
        if shared_embedding is not None:
            # 使用解码器对隐藏状态应用共享的嵌入向量的核参数
            hidden_states = self.decoder.apply({"params": {"kernel": shared_embedding.T}}, hidden_states)
        else:
            # 否则，直接使用解码器处理隐藏状态
            hidden_states = self.decoder(hidden_states)

        # 将偏置项转换为与模型指定的数据类型相匹配的 JAX 数组
        bias = jnp.asarray(self.bias, self.dtype)
        # 将偏置项加到隐藏状态上
        hidden_states += bias
        # 返回处理后的最终隐藏状态
        return hidden_states
class FlaxRobertaClassificationHead(nn.Module):
    config: RobertaConfig  # 定义一个属性 config，类型为 RobertaConfig，用于存储配置信息
    dtype: jnp.dtype = jnp.float32  # 定义一个属性 dtype，默认为 jnp.float32

    def setup(self):
        self.dense = nn.Dense(
            self.config.hidden_size,  # 使用 config 中的 hidden_size 初始化一个全连接层
            dtype=self.dtype,  # 指定数据类型为 dtype
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),  # 使用正态分布初始化权重
        )
        classifier_dropout = (
            self.config.classifier_dropout  # 获取 config 中的 classifier_dropout 属性
            if self.config.classifier_dropout is not None  # 如果不为 None，则使用该值
            else self.config.hidden_dropout_prob  # 否则使用 config 中的 hidden_dropout_prob 属性的值
        )
        self.dropout = nn.Dropout(rate=classifier_dropout)  # 使用 classifier_dropout 率初始化一个 Dropout 层
        self.out_proj = nn.Dense(
            self.config.num_labels,  # 使用 config 中的 num_labels 初始化一个全连接层
            dtype=self.dtype,  # 指定数据类型为 dtype
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),  # 使用正态分布初始化权重
        )

    def __call__(self, hidden_states, deterministic=True):
        hidden_states = hidden_states[:, 0, :]  # 仅保留每个样本的第一个 token 的隐藏状态
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)  # 应用 Dropout
        hidden_states = self.dense(hidden_states)  # 全连接层处理隐藏状态
        hidden_states = nn.tanh(hidden_states)  # 使用双曲正切作为激活函数
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)  # 再次应用 Dropout
        hidden_states = self.out_proj(hidden_states)  # 使用最后一个全连接层进行最终的分类预测
        return hidden_states


class FlaxRobertaPreTrainedModel(FlaxPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = RobertaConfig  # 类属性，指定配置类为 RobertaConfig
    base_model_prefix = "roberta"  # 类属性，指定基础模型前缀为 "roberta"

    module_class: nn.Module = None  # 类属性，用于存储模块类，默认为 None

    def __init__(
        self,
        config: RobertaConfig,
        input_shape: Tuple = (1, 1),
        seed: int = 0,
        dtype: jnp.dtype = jnp.float32,
        _do_init: bool = True,
        gradient_checkpointing: bool = False,
        **kwargs,
    ):
        module = self.module_class(config=config, dtype=dtype, gradient_checkpointing=gradient_checkpointing, **kwargs)
        super().__init__(config, module, input_shape=input_shape, seed=seed, dtype=dtype, _do_init=_do_init)

    # Copied from transformers.models.bert.modeling_flax_bert.FlaxBertPreTrainedModel.enable_gradient_checkpointing
    def enable_gradient_checkpointing(self):
        self._module = self.module_class(
            config=self.config,  # 使用当前实例的 config 属性初始化模块
            dtype=self.dtype,  # 使用当前实例的 dtype 属性指定数据类型
            gradient_checkpointing=True,  # 启用梯度检查点
        )
    # 初始化权重方法，用于模型参数初始化
    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict = None) -> FrozenDict:
        # 初始化输入张量
        input_ids = jnp.zeros(input_shape, dtype="i4")
        # token_type_ids初始化为与input_ids相同形状的全1张量
        token_type_ids = jnp.ones_like(input_ids)
        # 根据input_ids创建position_ids，并用config中的pad_token_id进行填充
        position_ids = create_position_ids_from_input_ids(input_ids, self.config.pad_token_id)
        # attention_mask初始化为与input_ids相同形状的全1张量
        attention_mask = jnp.ones_like(input_ids)
        # head_mask初始化为形状为(config.num_hidden_layers, config.num_attention_heads)的全1张量
        head_mask = jnp.ones((self.config.num_hidden_layers, self.config.num_attention_heads))

        # 划分随机数生成器rng，分为params和dropout两个部分
        params_rng, dropout_rng = jax.random.split(rng)
        rngs = {"params": params_rng, "dropout": dropout_rng}

        # 如果config中包含cross-attention，初始化encoder_hidden_states和encoder_attention_mask
        if self.config.add_cross_attention:
            encoder_hidden_states = jnp.zeros(input_shape + (self.config.hidden_size,))
            encoder_attention_mask = attention_mask
            # 使用module的init方法初始化模块，传入必要参数，不返回字典
            module_init_outputs = self.module.init(
                rngs,
                input_ids,
                attention_mask,
                token_type_ids,
                position_ids,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                return_dict=False,
            )
        else:
            # 使用module的init方法初始化模块，传入必要参数，不返回字典
            module_init_outputs = self.module.init(
                rngs, input_ids, attention_mask, token_type_ids, position_ids, head_mask, return_dict=False
            )

        # 从初始化输出中获取随机参数
        random_params = module_init_outputs["params"]

        # 如果传入了params，则将随机参数与params进行融合
        if params is not None:
            random_params = flatten_dict(unfreeze(random_params))
            params = flatten_dict(unfreeze(params))
            for missing_key in self._missing_keys:
                # 将随机参数中缺失的键添加到params中
                params[missing_key] = random_params[missing_key]
            self._missing_keys = set()
            # 返回融合后的冻结params
            return freeze(unflatten_dict(params))
        else:
            # 否则直接返回随机参数
            return random_params

    # 从transformers库中复制的初始化缓存方法
    def init_cache(self, batch_size, max_length):
        r"""
        Args:
            batch_size (`int`):
                fast auto-regressive decoding使用的batch_size，定义了初始化缓存的批大小。
            max_length (`int`):
                auto-regressive decoding的最大可能长度，定义了初始化缓存的序列长度。
        """
        # 初始化用于检索缓存的输入变量
        input_ids = jnp.ones((batch_size, max_length), dtype="i4")
        # attention_mask初始化为与input_ids相同形状的全1张量
        attention_mask = jnp.ones_like(input_ids, dtype="i4")
        # position_ids根据input_ids广播而来，形状与input_ids相同
        position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_ids.shape)

        # 使用module的init方法初始化模块，传入必要参数，返回不包含字典的初始化变量
        init_variables = self.module.init(
            jax.random.PRNGKey(0), input_ids, attention_mask, position_ids, return_dict=False, init_cache=True
        )
        # 返回解冻的初始化变量中的cache
        return unfreeze(init_variables["cache"])

    # 将开始字符串的文档字符串添加到模型前向传播方法
    @add_start_docstrings_to_model_forward(ROBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    # 定义类的 __call__ 方法，使对象可以像函数一样被调用
    def __call__(
        # 输入的 token IDs，用于模型的输入
        self,
        # 注意力掩码，指示模型在哪些位置需要注意
        attention_mask=None,
        # token 类型 IDs，用于区分不同句子的 token
        token_type_ids=None,
        # 位置 IDs，指示每个 token 在句子中的位置
        position_ids=None,
        # 头部掩码，控制多头注意力机制中哪些头部生效
        head_mask=None,
        # 编码器隐藏状态，用于模型的编码器
        encoder_hidden_states=None,
        # 编码器注意力掩码，指示编码器中需要注意的位置
        encoder_attention_mask=None,
        # 参数字典，包含其他参数的字典形式输入
        params: dict = None,
        # 随机数生成器的密钥，用于随机性操作
        dropout_rng: jax.random.PRNGKey = None,
        # 是否在训练阶段，影响是否应用 dropout 等训练相关操作
        train: bool = False,
        # 是否输出注意力矩阵
        output_attentions: Optional[bool] = None,
        # 是否输出隐藏状态
        output_hidden_states: Optional[bool] = None,
        # 是否返回字典形式的输出结果
        return_dict: Optional[bool] = None,
        # 过去的键值对，用于处理带有过去状态的模型
        past_key_values: dict = None,
# 从transformers.models.bert.modeling_flax_bert.FlaxBertModule复制代码，并将Bert替换为Roberta
class FlaxRobertaModule(nn.Module):
    config: RobertaConfig
    dtype: jnp.dtype = jnp.float32  # 计算中的数据类型
    add_pooling_layer: bool = True  # 是否添加池化层
    gradient_checkpointing: bool = False  # 是否使用梯度检查点

    def setup(self):
        self.embeddings = FlaxRobertaEmbeddings(self.config, dtype=self.dtype)  # 初始化Roberta模型的嵌入层
        self.encoder = FlaxRobertaEncoder(
            self.config,
            dtype=self.dtype,
            gradient_checkpointing=self.gradient_checkpointing,
        )  # 初始化Roberta模型的编码器
        self.pooler = FlaxRobertaPooler(self.config, dtype=self.dtype)  # 初始化Roberta模型的池化层

    def __call__(
        self,
        input_ids,
        attention_mask,
        token_type_ids: Optional[jnp.ndarray] = None,
        position_ids: Optional[jnp.ndarray] = None,
        head_mask: Optional[jnp.ndarray] = None,
        encoder_hidden_states: Optional[jnp.ndarray] = None,
        encoder_attention_mask: Optional[jnp.ndarray] = None,
        init_cache: bool = False,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        # 当token_type_ids未传递时，确保其正确初始化为零数组
        if token_type_ids is None:
            token_type_ids = jnp.zeros_like(input_ids)

        # 当position_ids未传递时，确保其正确初始化为广播到适当形状的数组
        if position_ids is None:
            position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_ids.shape)

        # 使用嵌入层处理输入，生成隐藏状态
        hidden_states = self.embeddings(
            input_ids, token_type_ids, position_ids, attention_mask, deterministic=deterministic
        )
        
        # 使用编码器处理隐藏状态，返回模型输出
        outputs = self.encoder(
            hidden_states,
            attention_mask,
            head_mask=head_mask,
            deterministic=deterministic,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            init_cache=init_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        hidden_states = outputs[0]  # 获取编码器输出的隐藏状态

        # 如果不返回字典格式的输出
        if not return_dict:
            # 如果pooled为None，则不返回它
            if pooled is None:
                return (hidden_states,) + outputs[1:]
            return (hidden_states, pooled) + outputs[1:]

        # 返回带池化和交叉注意力的基础模型输出
        return FlaxBaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=hidden_states,
            pooler_output=pooled,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )


@add_start_docstrings(
    "The bare RoBERTa Model transformer outputting raw hidden-states without any specific head on top.",
    ROBERTA_START_DOCSTRING,
)
class FlaxRobertaModel(FlaxRobertaPreTrainedModel):
    module_class = FlaxRobertaModule

# 调用函数向模型类添加示例文档字符串
append_call_sample_docstring(FlaxRobertaModel, _CHECKPOINT_FOR_DOC, FlaxBaseModelOutputWithPooling, _CONFIG_FOR_DOC)


class FlaxRobertaForMaskedLMModule(nn.Module):
    config: RobertaConfig
    dtype: jnp.dtype = jnp.float32
    gradient_checkpointing: bool = False

    def setup(self):
        # 初始化 RoBERTa 模型
        self.roberta = FlaxRobertaModule(
            config=self.config,
            add_pooling_layer=False,
            dtype=self.dtype,
            gradient_checkpointing=self.gradient_checkpointing,
        )
        # 初始化 RoBERTa 语言模型头部
        self.lm_head = FlaxRobertaLMHead(config=self.config, dtype=self.dtype)

    def __call__(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        position_ids,
        head_mask,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        # 调用 RoBERTa 模型，获取模型输出
        outputs = self.roberta(
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            head_mask,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        # 获取共享的词嵌入（如果配置允许）
        if self.config.tie_word_embeddings:
            shared_embedding = self.roberta.variables["params"]["embeddings"]["word_embeddings"]["embedding"]
        else:
            shared_embedding = None

        # 计算预测得分
        logits = self.lm_head(hidden_states, shared_embedding=shared_embedding)

        # 根据 return_dict 决定返回的格式
        if not return_dict:
            return (logits,) + outputs[1:]

        # 返回 RoBERTa 对象的输出作为 MaskedLM 的输出
        return FlaxMaskedLMOutput(
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings("""RoBERTa Model with a `language modeling` head on top.""", ROBERTA_START_DOCSTRING)
class FlaxRobertaForMaskedLM(FlaxRobertaPreTrainedModel):
    module_class = FlaxRobertaForMaskedLMModule

# 调用函数向模型类添加示例文档字符串
append_call_sample_docstring(
    FlaxRobertaForMaskedLM,
    _CHECKPOINT_FOR_DOC,
    FlaxBaseModelOutputWithPooling,
    _CONFIG_FOR_DOC,
    mask="<mask>",
)


class FlaxRobertaForSequenceClassificationModule(nn.Module):
    config: RobertaConfig
    dtype: jnp.dtype = jnp.float32
    gradient_checkpointing: bool = False

    def setup(self):
        # 初始化 RoBERTa 模型
        self.roberta = FlaxRobertaModule(
            config=self.config,
            dtype=self.dtype,
            add_pooling_layer=False,
            gradient_checkpointing=self.gradient_checkpointing,
        )
        # 初始化 RoBERTa 序列分类头部
        self.classifier = FlaxRobertaClassificationHead(config=self.config, dtype=self.dtype)
    # 定义一个方法，使得对象可以像函数一样被调用，接受多个输入参数和多个关键字参数
    def __call__(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        position_ids,
        head_mask,
        deterministic: bool = True,  # 控制模型行为的布尔参数，默认为True
        output_attentions: bool = False,  # 是否输出注意力权重的布尔参数，默认为False
        output_hidden_states: bool = False,  # 是否输出隐藏状态的布尔参数，默认为False
        return_dict: bool = True,  # 是否返回结果字典的布尔参数，默认为True
    ):
        # 使用预训练模型进行前向传播
        outputs = self.roberta(
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            head_mask,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 提取模型输出的序列输出（通常是最后一层隐藏状态）
        sequence_output = outputs[0]
        # 使用分类器对序列输出进行分类，得到预测的逻辑回归结果
        logits = self.classifier(sequence_output, deterministic=deterministic)

        # 如果不要求返回结果字典，则返回一个元组，包含逻辑回归结果和其他输出
        if not return_dict:
            return (logits,) + outputs[1:]

        # 否则，返回一个FlaxSequenceClassifierOutput对象，包含逻辑回归结果、隐藏状态和注意力权重
        return FlaxSequenceClassifierOutput(
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
@add_start_docstrings(
    """
    Roberta Model transformer with a sequence classification/regression head on top (a linear layer on top of the
    pooled output) e.g. for GLUE tasks.
    """,
    ROBERTA_START_DOCSTRING,
)



# 使用装饰器添加文档字符串，描述这是一个在Roberta模型基础上构建的序列分类/回归模型，顶部有一个线性层作为池化输出的一部分，例如用于GLUE任务。
class FlaxRobertaForSequenceClassification(FlaxRobertaPreTrainedModel):
    module_class = FlaxRobertaForSequenceClassificationModule



append_call_sample_docstring(
    FlaxRobertaForSequenceClassification,
    _CHECKPOINT_FOR_DOC,
    FlaxSequenceClassifierOutput,
    _CONFIG_FOR_DOC,
)



# 从transformers.models.bert.modeling_flax_bert.FlaxBertForMultipleChoiceModule复制过来，将Bert改为Roberta，self.bert改为self.roberta
class FlaxRobertaForMultipleChoiceModule(nn.Module):
    config: RobertaConfig
    dtype: jnp.dtype = jnp.float32
    gradient_checkpointing: bool = False

    def setup(self):
        # 初始化Roberta模块，包括配置、数据类型和梯度检查点设置
        self.roberta = FlaxRobertaModule(
            config=self.config,
            dtype=self.dtype,
            gradient_checkpointing=self.gradient_checkpointing,
        )
        # 使用配置中的隐藏层dropout率初始化dropout层
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout_prob)
        # 初始化分类器层，输出为1，数据类型与配置中的隐藏层一致
        self.classifier = nn.Dense(1, dtype=self.dtype)

    def __call__(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        position_ids,
        head_mask,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        # 获取选择项的数量
        num_choices = input_ids.shape[1]
        # 重新整形输入以便于模型处理
        input_ids = input_ids.reshape(-1, input_ids.shape[-1]) if input_ids is not None else None
        attention_mask = attention_mask.reshape(-1, attention_mask.shape[-1]) if attention_mask is not None else None
        token_type_ids = token_type_ids.reshape(-1, token_type_ids.shape[-1]) if token_type_ids is not None else None
        position_ids = position_ids.reshape(-1, position_ids.shape[-1]) if position_ids is not None else None

        # 调用Roberta模型
        outputs = self.roberta(
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            head_mask,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取池化后的输出
        pooled_output = outputs[1]
        # 对池化输出应用dropout，具体行为根据deterministic参数确定
        pooled_output = self.dropout(pooled_output, deterministic=deterministic)
        # 将dropout后的输出通过分类器层，得到logits
        logits = self.classifier(pooled_output)

        # 将logits重新整形为（batch_size, num_choices）
        reshaped_logits = logits.reshape(-1, num_choices)

        if not return_dict:
            # 如果不需要返回字典，则返回元组形式的输出
            return (reshaped_logits,) + outputs[2:]

        # 返回多选题模型的输出，包括logits、隐藏状态和注意力
        return FlaxMultipleChoiceModelOutput(
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )



@add_start_docstrings(
    """
    Roberta Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
    softmax) e.g. for RocStories/SWAG tasks.
    """,



# 这部分代码片段未完成，应继续添加代码以完成类的定义和功能。
    # 定义 ROBERTA_START_DOCSTRING 常量，通常用于标识文档字符串的起始位置
    ROBERTA_START_DOCSTRING,
# 在 FlaxRobertaForMultipleChoice 类中设置模块类为 FlaxRobertaForMultipleChoiceModule
class FlaxRobertaForMultipleChoice(FlaxRobertaPreTrainedModel):
    module_class = FlaxRobertaForMultipleChoiceModule

# 覆盖 FlaxRobertaForMultipleChoice 类的调用文档字符串，使用 ROBERTA_INPUTS_DOCSTRING 格式化字符串
overwrite_call_docstring(
    FlaxRobertaForMultipleChoice, ROBERTA_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length")
)

# 向 FlaxRobertaForMultipleChoice 类的示例调用文档字符串附加示例代码和相关说明
append_call_sample_docstring(
    FlaxRobertaForMultipleChoice,
    _CHECKPOINT_FOR_DOC,
    FlaxMultipleChoiceModelOutput,
    _CONFIG_FOR_DOC,
)


# 从 transformers.models.bert.modeling_flax_bert.FlaxBertForTokenClassificationModule 复制到 FlaxRobertaForTokenClassificationModule，将 Bert 替换为 Roberta，self.bert 替换为 self.roberta
class FlaxRobertaForTokenClassificationModule(nn.Module):
    config: RobertaConfig
    dtype: jnp.dtype = jnp.float32
    gradient_checkpointing: bool = False

    def setup(self):
        # 使用 FlaxRobertaModule 创建 self.roberta，配置为不添加池化层，是否进行梯度检查点由 self.gradient_checkpointing 控制
        self.roberta = FlaxRobertaModule(
            config=self.config,
            dtype=self.dtype,
            add_pooling_layer=False,
            gradient_checkpointing=self.gradient_checkpointing,
        )
        # 根据配置设置分类器的 dropout 率
        classifier_dropout = (
            self.config.classifier_dropout
            if self.config.classifier_dropout is not None
            else self.config.hidden_dropout_prob
        )
        # 创建 dropout 层
        self.dropout = nn.Dropout(rate=classifier_dropout)
        # 创建分类器，输出维度为 self.config.num_labels
        self.classifier = nn.Dense(self.config.num_labels, dtype=self.dtype)

    def __call__(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        position_ids,
        head_mask,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        # 调用 self.roberta 进行模型前向传播
        outputs = self.roberta(
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            head_mask,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取模型的隐藏状态
        hidden_states = outputs[0]
        # 对隐藏状态应用 dropout
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        # 对应用 dropout 后的隐藏状态应用分类器，得到 logits
        logits = self.classifier(hidden_states)

        # 如果不返回字典，则返回 logits 以及 outputs 中的其余部分
        if not return_dict:
            return (logits,) + outputs[1:]

        # 返回 FlaxTokenClassifierOutput 对象，包含 logits、隐藏状态和注意力权重
        return FlaxTokenClassifierOutput(
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


# 为 FlaxRobertaForTokenClassification 类添加起始文档字符串，描述其为在隐藏状态之上具有标记分类头的 Roberta 模型，例如用于命名实体识别 (NER) 任务
@add_start_docstrings(
    """
    Roberta Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
    Named-Entity-Recognition (NER) tasks.
    """,
    ROBERTA_START_DOCSTRING,
)
class FlaxRobertaForTokenClassification(FlaxRobertaPreTrainedModel):
    module_class = FlaxRobertaForTokenClassificationModule

# 向 FlaxRobertaForTokenClassification 类的示例调用文档字符串附加示例代码和相关说明
append_call_sample_docstring(
    FlaxRobertaForTokenClassification,
    _CHECKPOINT_FOR_DOC,
    FlaxTokenClassifierOutput,
    _CONFIG_FOR_DOC,
)
# 从transformers.models.bert.modeling_flax_bert.FlaxBertForQuestionAnsweringModule复制代码到此处，并将Bert->Roberta，self.bert->self.roberta
class FlaxRobertaForQuestionAnsweringModule(nn.Module):
    config: RobertaConfig
    dtype: jnp.dtype = jnp.float32
    gradient_checkpointing: bool = False

    def setup(self):
        # 初始化Roberta模型作为self.roberta，不包含池化层，支持梯度检查点
        self.roberta = FlaxRobertaModule(
            config=self.config,
            dtype=self.dtype,
            add_pooling_layer=False,
            gradient_checkpointing=self.gradient_checkpointing,
        )
        # 初始化用于QA任务的输出层self.qa_outputs，包含num_labels个输出单元
        self.qa_outputs = nn.Dense(self.config.num_labels, dtype=self.dtype)

    def __call__(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        position_ids,
        head_mask,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        # 调用Roberta模型self.roberta进行前向传播
        outputs = self.roberta(
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            head_mask,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 从输出中获取隐藏状态
        hidden_states = outputs[0]

        # 将隐藏状态传入QA输出层self.qa_outputs，得到起始和结束位置的logits
        logits = self.qa_outputs(hidden_states)
        start_logits, end_logits = jnp.split(logits, self.config.num_labels, axis=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        # 如果不返回字典，直接返回logits和可能存在的额外输出
        if not return_dict:
            return (start_logits, end_logits) + outputs[1:]

        # 返回QA模型的输出，包括起始和结束logits、隐藏状态和注意力权重
        return FlaxQuestionAnsweringModelOutput(
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings(
    """
    为抽取式问答任务（如SQuAD）设计的Roberta模型，顶部有一个用于span分类的线性层，
    用于计算`span start logits`和`span end logits`的隐藏状态输出。
    """,
    ROBERTA_START_DOCSTRING,
)
class FlaxRobertaForQuestionAnswering(FlaxRobertaPreTrainedModel):
    # 使用FlaxRobertaForQuestionAnsweringModule作为模型类
    module_class = FlaxRobertaForQuestionAnsweringModule


# 添加调用示例的文档字符串
append_call_sample_docstring(
    FlaxRobertaForQuestionAnswering,
    _CHECKPOINT_FOR_DOC,
    FlaxQuestionAnsweringModelOutput,
    _CONFIG_FOR_DOC,
)


class FlaxRobertaForCausalLMModule(nn.Module):
    config: RobertaConfig
    dtype: jnp.dtype = jnp.float32
    gradient_checkpointing: bool = False

    def setup(self):
        # 初始化Roberta模型作为self.roberta，不包含池化层，支持梯度检查点
        self.roberta = FlaxRobertaModule(
            config=self.config,
            add_pooling_layer=False,
            dtype=self.dtype,
            gradient_checkpointing=self.gradient_checkpointing,
        )
        # 初始化用于Causal LM任务的LM头部self.lm_head
        self.lm_head = FlaxRobertaLMHead(config=self.config, dtype=self.dtype)
    # 定义一个特殊方法 __call__，用于实现对象的可调用行为
    def __call__(
        self,
        input_ids,
        attention_mask,
        position_ids,
        token_type_ids: Optional[jnp.ndarray] = None,
        head_mask: Optional[jnp.ndarray] = None,
        encoder_hidden_states: Optional[jnp.ndarray] = None,
        encoder_attention_mask: Optional[jnp.ndarray] = None,
        init_cache: bool = False,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        # 调用模型的主体部分
        outputs = self.roberta(
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            init_cache=init_cache,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 从模型输出中获取隐藏状态
        hidden_states = outputs[0]

        # 如果配置允许词嵌入共享
        if self.config.tie_word_embeddings:
            # 获取共享的词嵌入层
            shared_embedding = self.roberta.variables["params"]["embeddings"]["word_embeddings"]["embedding"]
        else:
            shared_embedding = None

        # 计算预测分数（logits）
        logits = self.lm_head(hidden_states, shared_embedding=shared_embedding)

        # 如果不要求返回字典形式的结果，则返回元组形式的输出
        if not return_dict:
            return (logits,) + outputs[1:]

        # 返回带有交叉注意力的因果语言模型输出对象
        return FlaxCausalLMOutputWithCrossAttentions(
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )
"""
Roberta Model with a language modeling head on top (a linear layer on top of the hidden-states output) e.g for
autoregressive tasks.
"""

# 将 FlaxRobertaForCausalLM 类定义为一个特定的 RoBERTa 模型，用于自回归任务，包括语言建模等
@add_start_docstrings(
    """
    Roberta Model with a language modeling head on top (a linear layer on top of the hidden-states output) e.g for
    autoregressive tasks.
    """,
    ROBERTA_START_DOCSTRING,
)
class FlaxRobertaForCausalLM(FlaxRobertaPreTrainedModel):
    # 设置模型的主体类为 FlaxRobertaForCausalLMModule
    module_class = FlaxRobertaForCausalLMModule

    # 准备用于生成的输入数据，包括初始化缓存和注意力遮罩
    def prepare_inputs_for_generation(self, input_ids, max_length, attention_mask: Optional[jax.Array] = None):
        # 获取输入的批量大小和序列长度
        batch_size, seq_length = input_ids.shape

        # 初始化缓存 past_key_values
        past_key_values = self.init_cache(batch_size, max_length)

        # 注意：通常情况下，需要在 attention_mask 中为超出 input_ids.shape[-1] 和小于 cache_length 的位置放置 0
        # 但由于解码器使用因果遮罩，这些位置已经被遮蔽了。
        # 因此，我们可以在这里创建一个静态的 attention_mask，这对编译效率更高。
        extended_attention_mask = jnp.ones((batch_size, max_length), dtype="i4")
        if attention_mask is not None:
            # 计算位置 IDs
            position_ids = attention_mask.cumsum(axis=-1) - 1
            # 将 attention_mask 动态更新到 extended_attention_mask 中
            extended_attention_mask = lax.dynamic_update_slice(extended_attention_mask, attention_mask, (0, 0))
        else:
            # 如果 attention_mask 为 None，则创建一个广播后的位置 IDs
            position_ids = jnp.broadcast_to(jnp.arange(seq_length, dtype="i4")[None, :], (batch_size, seq_length))

        return {
            "past_key_values": past_key_values,
            "attention_mask": extended_attention_mask,
            "position_ids": position_ids,
        }

    # 更新生成过程中的输入参数，包括 past_key_values 和 position_ids
    def update_inputs_for_generation(self, model_outputs, model_kwargs):
        model_kwargs["past_key_values"] = model_outputs.past_key_values
        model_kwargs["position_ids"] = model_kwargs["position_ids"][:, -1:] + 1
        return model_kwargs


# 附加调用示例文档字符串，指定模型类、检查点信息、输出类型及配置信息
append_call_sample_docstring(
    FlaxRobertaForCausalLM,
    _CHECKPOINT_FOR_DOC,
    FlaxCausalLMOutputWithCrossAttentions,
    _CONFIG_FOR_DOC,
)
```