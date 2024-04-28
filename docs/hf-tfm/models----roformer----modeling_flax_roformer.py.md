# `.\transformers\models\roformer\modeling_flax_roformer.py`

```
# coding=utf-8
    # 定义模型配置类 RoFormerConfig 的参数
    Parameters:
        # 模型配置类对象，包含模型所有参数
        config ([`RoFormerConfig`]): Model configuration class with all the parameters of the
            model. Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~FlaxPreTrainedModel.from_pretrained`] method to load the model weights.
        # 计算使用的数据类型，可以是 jax.numpy.float32、jax.numpy.float16 (GPU) 或 jax.numpy.bfloat16 (TPU)
        dtype (`jax.numpy.dtype`, *optional*, defaults to `jax.numpy.float32`):
            The data type of the computation. Can be one of `jax.numpy.float32`, `jax.numpy.float16` (on GPUs) and
            `jax.numpy.bfloat16` (on TPUs).
    
            This can be used to enable mixed-precision training or half-precision inference on GPUs or TPUs. If
            specified all the computation will be performed with the given `dtype`.
    
            **Note that this only specifies the dtype of the computation and does not influence the dtype of model
            parameters.**
    
            If you wish to change the dtype of the model parameters, see [`~FlaxPreTrainedModel.to_fp16`] and
            [`~FlaxPreTrainedModel.to_bf16`].
# 这是 ROFORMER 模型的输入文档字符串，描述了模型的输入参数
ROFORMER_INPUTS_DOCSTRING = r"""
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

# 这个函数用于创建正弦编码的位置编码
def create_sinusoidal_positions(n_pos, dim):
    # 计算位置编码
    position_enc = np.array([[pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)] for pos in range(n_pos)])
    # 将位置编码分为sin和cos部分
    sentinel = dim // 2 + dim % 2
    out = np.zeros_like(position_enc)
    out[:, 0:sentinel] = np.sin(position_enc[:, 0::2])
    out[:, sentinel:] = np.cos(position_enc[:, 1::2])
    # 将位置编码转换为jnp.array格式
    return jnp.array(out)

# FlaxRoFormerEmbeddings 类用于构建词汇和标记类型的嵌入
class FlaxRoFormerEmbeddings(nn.Module):
    """Construct the embeddings from word and token_type embeddings."""

    config: RoFormerConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation
    # 初始化模型参数
    def setup(self):
        # 创建词嵌入层，参数为词汇表大小和隐藏层大小，使用正态分布初始化
        self.word_embeddings = nn.Embed(
            self.config.vocab_size,
            self.config.hidden_size,
            embedding_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
        )
        # 创建标记类型嵌入层，参数为标记类型数和隐藏层大小，使用正态分布初始化
        self.token_type_embeddings = nn.Embed(
            self.config.type_vocab_size,
            self.config.hidden_size,
            embedding_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
        )
        # 创建层归一化层，参数为 epsilon 和数据类型
        self.LayerNorm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)
        # 创建 dropout 层，参数为 dropout 率
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout_prob)

    # 模型调用方法
    def __call__(self, input_ids, token_type_ids, attention_mask, deterministic: bool = True):
        # 将输入 id 转换为整数类型，并通过词嵌入层得到词嵌入
        inputs_embeds = self.word_embeddings(input_ids.astype("i4"))
        # 将标记类型 id 转换为整数类型，并通过标记类型嵌入层得到标记类型嵌入
        token_type_embeddings = self.token_type_embeddings(token_type_ids.astype("i4"))

        # 将两种嵌入进行相加得到隐藏状态
        hidden_states = inputs_embeds + token_type_embeddings

        # 对隐藏状态进行层归一化处理
        hidden_states = self.LayerNorm(hidden_states)
        # 对隐藏状态进行 dropout 处理
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        # 返回处理后的隐藏状态
        return hidden_states
# 定义一个 FlaxRoFormerSelfAttention 的PyTorch模块类
class FlaxRoFormerSelfAttention(nn.Module):
    # 初始化该类时传入的 RoFormerConfig 配置
    config: RoFormerConfig
    # 计算使用的数据类型，默认为 float32
    dtype: jnp.dtype = jnp.float32  

    # 在模块初始化时执行的 setup 方法
    def setup(self) -> None:
        # 检查隐藏层大小是否能被注意力头数整除，如果不能则报错
        if self.config.hidden_size % self.config.num_attention_heads != 0:
            raise ValueError(
                "`config.hidden_size`: {self.config.hidden_size} has to be a multiple of `config.num_attention_heads` "
                "                   : {self.config.num_attention_heads}"
            )
        
        # 使用 nn.Dense 定义查询、键和值的全连接层
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

        # 设置是否使用旋转值
        self.rotary_value = self.config.rotary_value

    # 定义前向传播方法
    def __call__(
        self,
        hidden_states,
        attention_mask,
        sinusoidal_pos,
        layer_head_mask,
        deterministic=True,
        output_attentions: bool = False,
    ):
        ):
            # 定义每个注意力头的维度
            head_dim = self.config.hidden_size // self.config.num_attention_heads

            # 使用 query 网络对隐藏状态进行变换，并reshape成合适的形状
            query_states = self.query(hidden_states).reshape(
                hidden_states.shape[:2] + (self.config.num_attention_heads, head_dim)
            )
            # 使用 value 网络对隐藏状态进行变换，并reshape成合适的形状
            value_states = self.value(hidden_states).reshape(
                hidden_states.shape[:2] + (self.config.num_attention_heads, head_dim)
            )
            # 使用 key 网络对隐藏状态进行变换，并reshape成合适的形状
            key_states = self.key(hidden_states).reshape(
                hidden_states.shape[:2] + (self.config.num_attention_heads, head_dim)
            )

            # 如果有正弦位置编码，则将其应用到注意力查询、键和值中
            if sinusoidal_pos is not None:
                if self.rotary_value:
                    # 如果使用旋转值，则将正弦位置编码应用到查询、键和值中
                    query_states, key_states, value_states = self.apply_rotary_position_embeddings(
                        sinusoidal_pos, query_states, key_states, value_states
                    )
                else:
                    # 否则只将正弦位置编码应用到查询和键中
                    query_states, key_states = self.apply_rotary_position_embeddings(
                        sinusoidal_pos, query_states, key_states
                    )

            # 将布尔类型的注意力掩码转换为注意力偏置
            if attention_mask is not None:
                # 将注意力掩码扩展为注意力偏置
                attention_mask = jnp.expand_dims(attention_mask, axis=(-3, -2))
                attention_bias = lax.select(
                    attention_mask > 0,
                    jnp.full(attention_mask.shape, 0.0).astype(self.dtype),
                    jnp.full(attention_mask.shape, jnp.finfo(self.dtype).min).astype(self.dtype),
                )
            else:
                attention_bias = None

            dropout_rng = None
            # 如果不是确定性的，并且注意力概率的丢弃率大于0，则创建一个丢弃率
            if not deterministic and self.config.attention_probs_dropout_prob > 0.0:
                dropout_rng = self.make_rng("dropout")

            # 计算点积注意力权重
            attn_weights = dot_product_attention_weights(
                query_states,
                key_states,
                bias=attention_bias,
                dropout_rng=dropout_rng,
                dropout_rate=self.config.attention_probs_dropout_prob,
                broadcast_dropout=True,
                deterministic=deterministic,
                dtype=self.dtype,
                precision=None,
            )

            # 如果有层级头掩码，则将其应用到注意力权重上
            if layer_head_mask is not None:
                attn_weights = jnp.einsum("...hqk,h->...hqk", attn_weights, layer_head_mask)

            # 计算注意力输出
            attn_output = jnp.einsum("...hqk,...khd->...qhd", attn_weights, value_states)
            attn_output = attn_output.reshape(attn_output.shape[:2] + (-1,))

            # 返回输出结果，如果需要输出注意力权重，则包括在内
            outputs = (attn_output, attn_weights) if output_attentions else (attn_output,)
            return outputs

        @staticmethod
    # 将输入的 sinusoidal_pos 按照最后一个维度拆分成两部分，分别表示 sin 和 cos
    sin, cos = sinusoidal_pos.split(2, axis=-1)
    # 构造 sin 和 cos 的位置编码，将其沿着最后一个维度堆叠成两个相同的张量
    sin_pos = jnp.stack([sin, sin], axis=-1).reshape(sinusoidal_pos.shape)
    cos_pos = jnp.stack([cos, cos], axis=-1).reshape(sinusoidal_pos.shape)

    # 定义函数，用于对给定的 layer 进行旋转操作
    def rotate_layer(layer, sin_pos, cos_pos):
        # 将给定的 layer 沿着最后一个维度拆分成两部分，并交换位置，实现旋转
        rotate_half_layer = jnp.stack([-layer[..., 1::2], layer[..., ::2]], axis=-1).reshape(layer.shape)
        # 使用正弦位置编码对原始 layer 进行旋转的余弦部分计算
        rotary_matrix_cos = jnp.einsum("bslh,...sh->bslh", layer, cos_pos)
        # 使用旋转后的一半 layer 和正弦位置编码进行旋转的正弦部分计算
        rotary_matrix_sin = jnp.einsum("bslh,...sh->bslh", rotate_half_layer, sin_pos)
        # 将旋转后的余弦部分和正弦部分相加，得到最终旋转后的结果
        return rotary_matrix_cos + rotary_matrix_sin

    # 对 query_layer 和 key_layer 应用旋转操作
    query_layer = rotate_layer(query_layer, sin_pos, cos_pos)
    key_layer = rotate_layer(key_layer, sin_pos, cos_pos)
    # 如果给定了 value_layer，则对其也应用旋转操作，并返回旋转后的三个张量；否则只返回旋转后的 query_layer 和 key_layer
    if value_layer is not None:
        value_layer = rotate_layer(value_layer, sin_pos, cos_pos)
        return query_layer, key_layer, value_layer
    return query_layer, key_layer
# 这是 FlaxRoFormerSelfOutput 类，负责处理从 FlaxRoFormerSelfAttention 输出的隐藏状态
class FlaxRoFormerSelfOutput(nn.Module):
    # 保存 RoFormerConfig 对象
    config: RoFormerConfig
    # 指定计算使用的数据类型为 float32
    dtype: jnp.dtype = jnp.float32  

    # 初始化模块内部的层
    def setup(self):
        # 创建一个全连接层，将输入映射到 hidden_size 大小的输出
        self.dense = nn.Dense(
            self.config.hidden_size,
            # 使用正态分布初始化权重
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            dtype=self.dtype,
        )
        # 创建一个层归一化层
        self.LayerNorm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)
        # 创建一个dropout层
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout_prob)

    # 前向传播函数
    def __call__(self, hidden_states, input_tensor, deterministic: bool = True):
        # 通过全连接层处理隐藏状态
        hidden_states = self.dense(hidden_states)
        # 应用dropout
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        # 将dropout后的结果与输入相加，并通过层归一化
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


# FlaxRoFormerAttention 类负责处理注意力机制
class FlaxRoFormerAttention(nn.Module):
    # 保存 RoFormerConfig 对象
    config: RoFormerConfig
    # 指定计算使用的数据类型为 float32
    dtype: jnp.dtype = jnp.float32

    # 初始化模块内部的层
    def setup(self):
        # 创建一个 FlaxRoFormerSelfAttention 实例
        self.self = FlaxRoFormerSelfAttention(self.config, dtype=self.dtype)
        # 创建一个 FlaxRoFormerSelfOutput 实例
        self.output = FlaxRoFormerSelfOutput(self.config, dtype=self.dtype)

    # 前向传播函数
    def __call__(
        self,
        hidden_states,
        attention_mask,
        sinusoidal_pos,
        layer_head_mask,
        deterministic=True,
        output_attentions: bool = False,
    ):
        # 通过 FlaxRoFormerSelfAttention 处理隐藏状态
        attn_outputs = self.self(
            hidden_states,
            attention_mask,
            sinusoidal_pos,
            layer_head_mask=layer_head_mask,
            deterministic=deterministic,
            output_attentions=output_attentions,
        )
        # 获取注意力输出
        attn_output = attn_outputs[0]
        # 通过 FlaxRoFormerSelfOutput 处理注意力输出
        hidden_states = self.output(attn_output, hidden_states, deterministic=deterministic)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_outputs[1],)

        return outputs


# FlaxRoFormerIntermediate 类负责处理中间层的计算
class FlaxRoFormerIntermediate(nn.Module):
    # 保存 RoFormerConfig 对象
    config: RoFormerConfig
    # 指定计算使用的数据类型为 float32
    dtype: jnp.dtype = jnp.float32  

    # 初始化模块内部的层
    def setup(self):
        # 创建一个全连接层，将输入映射到 intermediate_size 大小的输出
        self.dense = nn.Dense(
            self.config.intermediate_size,
            # 使用正态分布初始化权重
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            dtype=self.dtype,
        )
        # 根据配置选择激活函数
        self.activation = ACT2FN[self.config.hidden_act]

    # 前向传播函数
    def __call__(self, hidden_states):
        # 通过全连接层处理隐藏状态
        hidden_states = self.dense(hidden_states)
        # 应用激活函数
        hidden_states = self.activation(hidden_states)
        return hidden_states
# 定义一个表示 RoFormer 输出的类，继承自 nn.Module
class FlaxRoFormerOutput(nn.Module):
    config: RoFormerConfig  # RoFormerConfig 对象，存储了 RoFormer 的配置参数
    dtype: jnp.dtype = jnp.float32  # 计算过程中的数据类型，默认为 float32

    # setup 方法，设置模块的属性
    def setup(self):
        # 创建一个全连接层，将输入维度转换为 self.config.hidden_size
        self.dense = nn.Dense(
            self.config.hidden_size,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            dtype=self.dtype,
        )
        # 创建一个 dropout 层，用于随机丢弃部分神经元
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout_prob)
        # 创建一个 layer norm 层，用于标准化隐藏层输出
        self.LayerNorm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)

    # __call__ 方法，定义模块的前向传播逻辑
    def __call__(self, hidden_states, attention_output, deterministic: bool = True):
        # 使用全连接层处理隐藏状态数据
        hidden_states = self.dense(hidden_states)
        # 对输出结果进行 dropout 操作
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        # 将隐藏层输出与注意力输出相加，并进行 layer norm 处理
        hidden_states = self.LayerNorm(hidden_states + attention_output)
        # 返回处理后的隐藏层输出
        return hidden_states



# 定义一个表示 RoFormer 层的类，继承自 nn.Module
class FlaxRoFormerLayer(nn.Module):
    config: RoFormerConfig  # RoFormerConfig 对象，存储了 RoFormer 的配置参数
    dtype: jnp.dtype = jnp.float32  # 计算过程中的数据类型，默认为 float32

    # setup 方法，设置模块的属性
    def setup(self):
        # 创建一个 RoFormerAttention 层，用于计算自注意力
        self.attention = FlaxRoFormerAttention(self.config, dtype=self.dtype)
        # 创建一个 FlaxRoFormerIntermediate 层，用于处理隐藏层输出
        self.intermediate = FlaxRoFormerIntermediate(self.config, dtype=self.dtype)
        # 创建一个 FlaxRoFormerOutput 层，用于处理隐藏层输出
        self.output = FlaxRoFormerOutput(self.config, dtype=self.dtype)

    # __call__ 方法，定义模块的前向传播逻辑
    def __call__(
        self,
        hidden_states,
        attention_mask,
        sinusoidal_pos,
        layer_head_mask,
        deterministic: bool = True,
        output_attentions: bool = False,
    ):
        # 使用自注意力层处理隐藏状态数据
        attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            sinusiodal_pos,
            layer_head_mask=layer_head_mask,
            deterministic=deterministic,
            output_attentions=output_attentions,
        )
        # 获取自注意力层的输出
        attention_output = attention_outputs[0]

        # 使用 FlaxRoFormerIntermediate 层处理自注意力层的输出
        hidden_states = self.intermediate(attention_output)
        # 使用 FlaxRoFormerOutput 层处理 FlaxRoFormerIntermediate 层的输出，并与自注意力层的输出相加
        hidden_states = self.output(hidden_states, attention_output, deterministic=deterministic)

        # 将隐藏层输出放入元组并返回
        outputs = (hidden_states,)

        # 如果需要输出注意力权重，则将注意力权重放入元组
        if output_attentions:
            outputs += (attention_outputs[1],)
        return outputs



# 定义一个表示 RoFormer 层集合的类，继承自 nn.Module
class FlaxRoFormerLayerCollection(nn.Module):
    config: RoFormerConfig  # RoFormerConfig 对象，存储了 RoFormer 的配置参数
    dtype: jnp.dtype = jnp.float32  # 计算过程中的数据类型，默认为 float32

    # setup 方法，设置模块的属性
    def setup(self):
        # 创建一组 FlaxRoFormerLayer 层，个数为 self.config.num_hidden_layers
        self.layers = [
            FlaxRoFormerLayer(self.config, name=str(i), dtype=self.dtype) for i in range(self.config.num_hidden_layers)
        ]

    # __call__ 方法，定义模块的前向传播逻辑
    def __call__(
        self,
        hidden_states,
        attention_mask,
        sinusoidal_pos,
        head_mask,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        # 遍历每个 RoFormerLayer 层，并将输出传递给下一层
        for layer in self.layers:
            hidden_states = layer.__call__(
                hidden_states,
                attention_mask,
                sinusiodal_pos,
                layer_head_mask=head_mask,
                deterministic=deterministic,
                output_attentions=output_attentions,
            )

        # 将隐藏层输出放入元组并返回
        outputs = (hidden_states,)

        # 如果需要输出注意力权重，则将注意力权重放入元组
        if output_attentions:
            outputs += (attention_outputs[1],)

        return outputs
        # 如果不输出注意力权重，则将所有注意力权重置为空元组
        all_attentions = () if output_attentions else None
        # 如果不输出隐藏状态，则将所有隐藏状态置为空元组
        all_hidden_states = () if output_hidden_states else None

        # 如果指定了头部遮罩，检查其是否指定了正确数量的层
        if head_mask is not None:
            # 头部遮罩的第一维应该等于层的数量
            if head_mask.shape[0] != (len(self.layers)):
                raise ValueError(
                    f"The head_mask should be specified for {len(self.layers)} layers, but it is for                  "
                    f"       {head_mask.shape[0]}."
                )

        # 遍历每一层
        for i, layer in enumerate(self.layers):
            # 如果输出隐藏状态，则将当前隐藏状态添加到隐藏状态元组中
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            # 调用当前层的前向传播函数
            layer_outputs = layer(
                hidden_states,
                attention_mask,
                sinusoidal_pos,
                # 如果指定了头部遮罩，则传递给当前层
                layer_head_mask=head_mask[i] if head_mask is not None else None,
                deterministic=deterministic,
                output_attentions=output_attentions,
            )

            # 更新隐藏状态为当前层的输出
            hidden_states = layer_outputs[0]

            # 如果输出注意力权重，则将当前层的注意力权重添加到注意力权重元组中
            if output_attentions:
                all_attentions += (layer_outputs[1],)

        # 如果输出隐藏状态，则将最终隐藏状态添加到隐藏状态元组中
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        # 将最终的隐藏状态作为输出
        outputs = (hidden_states,)

        # 如果不返回字典，则返回元组中不为空的项
        if not return_dict:
            return tuple(v for v in outputs if v is not None)

        # 返回包含模型输出的字典
        return FlaxBaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_attentions
        )
# RoFormer 编码器模块
class FlaxRoFormerEncoder(nn.Module):
    # 配置信息
    config: RoFormerConfig
    # 计算的数据类型
    dtype: jnp.dtype = jnp.float32

    # 模块初始化
    def setup(self):
        # 创建正弦位置编码
        self.embed_positions = create_sinusoidal_positions(
            self.config.max_position_embeddings, self.config.hidden_size // self.config.num_attention_heads
        )
        # 创建 RoFormer 层集合
        self.layer = FlaxRoFormerLayerCollection(self.config, dtype=self.dtype)

    # 前向计算
    def __call__(
        self,
        hidden_states,
        attention_mask,
        head_mask,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        # 获取位置编码
        sinusoidal_pos = self.embed_positions[: hidden_states.shape[1], :]

        # 传入 RoFormer 层集合进行计算
        return self.layer(
            hidden_states,
            attention_mask,
            sinusoidal_pos,
            head_mask,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

# RoFormer 预测头的变换模块
class FlaxRoFormerPredictionHeadTransform(nn.Module):
    # 配置信息
    config: RoFormerConfig
    # 计算的数据类型
    dtype: jnp.dtype = jnp.float32

    # 模块初始化
    def setup(self):
        # 全连接层
        self.dense = nn.Dense(self.config.hidden_size, dtype=self.dtype)
        # 激活函数
        self.activation = ACT2FN[self.config.hidden_act]
        # 层归一化
        self.LayerNorm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)

    # 前向计算
    def __call__(self, hidden_states):
        # 全连接变换
        hidden_states = self.dense(hidden_states)
        # 激活函数
        hidden_states = self.activation(hidden_states)
        # 层归一化
        return self.LayerNorm(hidden_states)

# RoFormer 语言模型预测头
class FlaxRoFormerLMPredictionHead(nn.Module):
    # 配置信息
    config: RoFormerConfig
    # 计算的数据类型
    dtype: jnp.dtype = jnp.float32
    # 偏置初始化函数
    bias_init: Callable[..., np.ndarray] = jax.nn.initializers.zeros

    # 模块初始化
    def setup(self):
        # 创建预测头变换模块
        self.transform = FlaxRoFormerPredictionHeadTransform(self.config, dtype=self.dtype)
        # 创建解码全连接层
        self.decoder = nn.Dense(self.config.vocab_size, dtype=self.dtype, use_bias=False)
        # 创建预测偏置
        self.bias = self.param("bias", self.bias_init, (self.config.vocab_size,))

    # 前向计算
    def __call__(self, hidden_states, shared_embedding=None):
        # 预测头变换
        hidden_states = self.transform(hidden_states)

        # 如果有共享的词嵌入权重, 使用它来计算预测
        if shared_embedding is not None:
            hidden_states = self.decoder.apply({"params": {"kernel": shared_embedding.T}}, hidden_states)
        # 否则使用自己的解码层计算预测
        else:
            hidden_states = self.decoder(hidden_states)

        # 加上偏置得到最终预测
        bias = jnp.asarray(self.bias, self.dtype)
        hidden_states += bias
        return hidden_states

# RoFormer 只有 MLM 头的模型
class FlaxRoFormerOnlyMLMHead(nn.Module):
    # 配置信息
    config: RoFormerConfig
    # 计算的数据类型
    dtype: jnp.dtype = jnp.float32
    # 在模型的初始化方法中，创建一个FlaxRoFormerLMPredictionHead对象，并将其存储在self.predictions中
    def setup(self):
        self.predictions = FlaxRoFormerLMPredictionHead(self.config, dtype=self.dtype)

    # 在调用模型对象时，将输入的hidden_states传递给self.predictions，并返回预测结果
    def __call__(self, hidden_states, shared_embedding=None):
        hidden_states = self.predictions(hidden_states, shared_embedding=shared_embedding)
        return hidden_states
# 定义一个用于 RoFormer 分类任务头部的模块
class FlaxRoFormerClassificationHead(nn.Module):
    # 用于存储 RoFormerConfig 的配置信息
    config: RoFormerConfig
    # 默认数据类型为 jnp.float32
    dtype: jnp.dtype = jnp.float32

    # 设置模块
    def setup(self):
        # 定义一个全连接层，输出维度为配置中的 hidden_size，使用配置中的初始化器范围进行初始化
        self.dense = nn.Dense(
            self.config.hidden_size,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
        )
        # 定义一个丢弃层，丢弃率为配置中的 hidden_dropout_prob
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout_prob)
        # 定义一个全连接层，输出维度为配置中的 num_labels，使用配置中的初始化器范围进行初始化
        self.out_proj = nn.Dense(
            self.config.num_labels,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
        )
        # 根据配置中的 hidden_act 选择激活函数
        self.activation = ACT2FN[self.config.hidden_act]

    # 定义模块的调用方式
    def __call__(self, hidden_states, deterministic=True):
        # 取第一个 token 的隐藏状态，相当于取 [CLS] token 的隐藏状态
        hidden_states = hidden_states[:, 0, :]
        # 对隐藏状态进行丢弃操作
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        # 通过全连接层进行线性变换
        hidden_states = self.dense(hidden_states)
        # 使用激活函数激活隐藏状态
        hidden_states = self.activation(hidden_states)
        # 再次对隐藏状态进行丢弃操作
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        # 通过输出层得到最终的分类结果
        hidden_states = self.out_proj(hidden_states)
        return hidden_states


# 定义一个用于 RoFormer 预训练模型的抽象类，用于处理权重初始化和下载预训练模型等操作
class FlaxRoFormerPreTrainedModel(FlaxPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 存储 RoFormer 配置类
    config_class = RoFormerConfig
    # 模型的基本名称前缀
    base_model_prefix = "roformer"
    # 模块类属性，默认为空
    module_class: nn.Module = None

    # 初始化方法
    def __init__(
        self,
        config: RoFormerConfig,
        input_shape: Tuple = (1, 1),
        seed: int = 0,
        dtype: jnp.dtype = jnp.float32,
        _do_init: bool = True,
        **kwargs,
    ):
        # 创建模块实例
        module = self.module_class(config=config, dtype=dtype, **kwargs)
        # 调用父类初始化方法
        super().__init__(config, module, input_shape=input_shape, seed=seed, dtype=dtype, _do_init=_do_init)

    # 初始化权重方法
    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict = None) -> FrozenDict:
        # 初始化输入张量
        input_ids = jnp.zeros(input_shape, dtype="i4")
        token_type_ids = jnp.zeros_like(input_ids)
        attention_mask = jnp.ones_like(input_ids)
        head_mask = jnp.ones((self.config.num_hidden_layers, self.config.num_attention_heads))

        # 划分随机数种子
        params_rng, dropout_rng = jax.random.split(rng)
        rngs = {"params": params_rng, "dropout": dropout_rng}

        # 使用模块的初始化方法初始化参数
        random_params = self.module.init(
            rngs, input_ids, attention_mask, token_type_ids, head_mask, return_dict=False
        )["params"]

        # 如果给定了已有的参数，则使用给定参数替换随机初始化的参数
        if params is not None:
            random_params = flatten_dict(unfreeze(random_params))
            params = flatten_dict(unfreeze(params))
            for missing_key in self._missing_keys:
                params[missing_key] = random_params[missing_key]
            self._missing_keys = set()
            return freeze(unflatten_dict(params))
        else:
            return random_params
    # 将函数的输入参数和调用时的参数格式化成文档字符串，并添加到模型前
    @add_start_docstrings_to_model_forward(ROFORMER_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    # 定义模型调用函数
    def __call__(
        self,
        input_ids,  # 输入的token IDs
        attention_mask=None,  # 注意力掩码，默认为None
        token_type_ids=None,  # token类型IDs，默认为None
        head_mask=None,  # 头掩码，默认为None
        params: dict = None,  # 参数字典，默认为None
        dropout_rng: jax.random.PRNGKey = None,  # 随机数生成器，默认为None
        train: bool = False,  # 训练标志，默认为False
        output_attentions: Optional[bool] = None,  # 输出注意力权重，默认为None
        output_hidden_states: Optional[bool] = None,  # 输出隐藏状态，默认为None
        return_dict: Optional[bool] = None,  # 返回字典，默认为None
    ):
        # 如果未指定，则根据配置确定是否输出注意力权重
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 如果未指定，则根据配置确定是否输出隐藏状态
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 如果未指定，则根据配置确定是否返回字典形式的输出
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        # 如果未传入token类型IDs，则初始化为与输入token IDs相同形状的全0数组
        if token_type_ids is None:
            token_type_ids = jnp.zeros_like(input_ids)

        # 如果未传入注意力掩码，则初始化为与输入token IDs相同形状的全1数组
        if attention_mask is None:
            attention_mask = jnp.ones_like(input_ids)

        # 如果未传入头掩码，则初始化为形状为(num_hidden_layers, num_attention_heads)的全1数组
        if head_mask is None:
            head_mask = jnp.ones((self.config.num_hidden_layers, self.config.num_attention_heads))

        # 如果需要处理任何PRNG（伪随机数生成器）
        rngs = {}
        if dropout_rng is not None:
            rngs["dropout"] = dropout_rng

        # 调用模型的apply方法进行前向传播
        return self.module.apply(
            {"params": params or self.params},  # 参数为传入的params字典或模型自带的参数字典
            jnp.array(input_ids, dtype="i4"),  # 将输入token IDs转换为32位整型数组
            jnp.array(attention_mask, dtype="i4"),  # 将注意力掩码转换为32位整型数组
            jnp.array(token_type_ids, dtype="i4"),  # 将token类型IDs转换为32位整型数组
            jnp.array(head_mask, dtype="i4"),  # 将头掩码转换为32位整型数组
            not train,  # 如果不处于训练模式，则传入True
            output_attentions,  # 是否输出注意力权重
            output_hidden_states,  # 是否输出隐藏状态
            return_dict,  # 是否返回字典形式的输出
            rngs=rngs,  # 传入PRNGs
        )
# 定义一个 FlaxRoFormerModule 类，继承自 nn.Module
class FlaxRoFormerModule(nn.Module):
    # 配置属性，指定 RoFormer 的配置
    config: RoFormerConfig
    # 计算时的数据类型，默认为 jnp.float32
    dtype: jnp.dtype = jnp.float32  

    # 模块的初始化方法
    def setup(self):
        # 初始化嵌入层对象，使用 FlaxRoFormerEmbeddings 类
        self.embeddings = FlaxRoFormerEmbeddings(self.config, dtype=self.dtype)
        # 初始化编码器对象，使用 FlaxRoFormerEncoder 类
        self.encoder = FlaxRoFormerEncoder(self.config, dtype=self.dtype)

    # 模块的调用方法
    def __call__(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        head_mask,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        # 通过嵌入层获取隐藏状态
        hidden_states = self.embeddings(input_ids, token_type_ids, attention_mask, deterministic=deterministic)
        # 通过编码器获取输出
        outputs = self.encoder(
            hidden_states,
            attention_mask,
            head_mask=head_mask,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # 提取隐藏状态
        hidden_states = outputs[0]

        # 如果不返回字典形式的结果，则返回元组形式
        if not return_dict:
            return (hidden_states,) + outputs[1:]

        # 返回 FlaxBaseModelOutput 对象，其中包括最后的隐藏状态、隐藏状态列表和注意力矩阵列表
        return FlaxBaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


# 添加文档字符串说明的装饰器，用于 RoFormer 模型输出原始隐藏状态
@add_start_docstrings(
    "The bare RoFormer Model transformer outputting raw hidden-states without any specific head on top.",
    ROFORMER_START_DOCSTRING,
)
# 定义 FlaxRoFormerModel 类，继承自 FlaxRoFormerPreTrainedModel 类
class FlaxRoFormerModel(FlaxRoFormerPreTrainedModel):
    # 模块类属性指定为 FlaxRoFormerModule 类
    module_class = FlaxRoFormerModule


# 添加调用样本文档字符串的方法
append_call_sample_docstring(FlaxRoFormerModel, _CHECKPOINT_FOR_DOC, FlaxBaseModelOutput, _CONFIG_FOR_DOC)


# 定义 FlaxRoFormerForMaskedLMModule 类，继承自 nn.Module
class FlaxRoFormerForMaskedLMModule(nn.Module):
    # 配置属性，指定 RoFormer 的配置
    config: RoFormerConfig
    # 计算时的数据类型，默认为 jnp.float32
    dtype: jnp.dtype = jnp.float32

    # 模块的初始化方法
    def setup(self):
        # 初始化 RoFormer 模型对象，使用 FlaxRoFormerModule 类
        self.roformer = FlaxRoFormerModule(config=self.config, dtype=self.dtype)
        # 初始化仅包含 MLM 头部的对象，使用 FlaxRoFormerOnlyMLMHead 类
        self.cls = FlaxRoFormerOnlyMLMHead(config=self.config, dtype=self.dtype)

    # 模块的调用方法
    def __call__(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        head_mask,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        ):
        # 定义模型的输出
        outputs = self.roformer(
            input_ids,
            attention_mask,
            token_type_ids,
            head_mask,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        # 如果绑定了词嵌入，则使用共享的嵌入变量
        if self.config.tie_word_embeddings:
            shared_embedding = self.roformer.variables["params"]["embeddings"]["word_embeddings"]["embedding"]
        else:
            shared_embedding = None

        # 计算预测分数
        logits = self.cls(hidden_states, shared_embedding=shared_embedding)

        # 如果不返回字典，则返回预测分数和其它输出
        if not return_dict:
            return (logits,) + outputs[1:]

        # 返回预测分数、隐藏状态和注意力信息组成的 FLAX MaskedLM 输出
        return FlaxMaskedLMOutput(
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 使用 add_start_docstrings 函数添加文档字符串到 FlaxRoFormerForMaskedLM 类
@add_start_docstrings("""RoFormer Model with a `language modeling` head on top.""", ROFORMER_START_DOCSTRING)
# 设置 FlaxRoFormerForMaskedLM 类的模块类为 FlaxRoFormerForMaskedLMModule
class FlaxRoFormerForMaskedLM(FlaxRoFormerPreTrainedModel):
    module_class = FlaxRoFormerForMaskedLMModule

# 使用 append_call_sample_docstring 函数为 FlaxRoFormerForMaskedLM 类添加调用示例的文档字符串
append_call_sample_docstring(
    FlaxRoFormerForMaskedLM,
    _CHECKPOINT_FOR_DOC,
    FlaxMaskedLMOutput,
    _CONFIG_FOR_DOC,
    mask="<mask>",
)


# 定义 FlaxRoFormerForSequenceClassificationModule 类
class FlaxRoFormerForSequenceClassificationModule(nn.Module):
    # 设置配置和数据类型
    config: RoFormerConfig
    dtype: jnp.dtype = jnp.float32

    # 设置初始化方法
    def setup(self):
        # 创建 RoFormer 模块对象
        self.roformer = FlaxRoFormerModule(config=self.config, dtype=self.dtype)
        # 创建分类器对象
        self.classifier = FlaxRoFormerClassificationHead(config=self.config, dtype=self.dtype)

    # 定义调用方法
    def __call__(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        head_mask,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        # 使用 RoFormer 模块对象处理输入，得到输出
        outputs = self.roformer(
            input_ids,
            attention_mask,
            token_type_ids,
            head_mask,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取序列输出
        sequence_output = outputs[0]
        # 使用分类器对序列输出进行分类
        logits = self.classifier(sequence_output, deterministic=deterministic)

        # 如果不返回字典，则返回 logits 和其他输出
        if not return_dict:
            return (logits,) + outputs[1:]

        # 返回 FlaxSequenceClassifierOutput 对象
        return FlaxSequenceClassifierOutput(
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


# 使用 add_start_docstrings 函数添加文档字符串到 FlaxRoFormerForSequenceClassification 类
@add_start_docstrings(
    """
    RoFormer Model transformer with a sequence classification/regression head on top (a linear layer on top of the
    pooled output) e.g. for GLUE tasks.
    """,
    ROFORMER_START_DOCSTRING,
)
# 设置 FlaxRoFormerForSequenceClassification 类的模块类为 FlaxRoFormerForSequenceClassificationModule
class FlaxRoFormerForSequenceClassification(FlaxRoFormerPreTrainedModel):
    module_class = FlaxRoFormerForSequenceClassificationModule

# 使用 append_call_sample_docstring 函数为 FlaxRoFormerForSequenceClassification 类添加调用示例的文档字符串
append_call_sample_docstring(
    FlaxRoFormerForSequenceClassification,
    _CHECKPOINT_FOR_DOC,
    FlaxSequenceClassifierOutput,
    _CONFIG_FOR_DOC,
)


# 定义 FlaxRoFormerForMultipleChoiceModule 类
class FlaxRoFormerForMultipleChoiceModule(nn.Module):
    # 设置配置和数据类型
    config: RoFormerConfig
    dtype: jnp.dtype = jnp.float32

    # 设置初始化方法
    def setup(self):
        # 创建 RoFormer 模块对象
        self.roformer = FlaxRoFormerModule(config=self.config, dtype=self.dtype)
        # 创建丢弃层对象
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout_prob)
        # 创建分类器对象
        self.classifier = nn.Dense(1, dtype=self.dtype)

    # 定义调用方法
    def __call__(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        head_mask,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        
        ):
        # 获取输入的选项数量
        num_choices = input_ids.shape[1]
        # 重新组织输入的 IDs 张量的形状
        input_ids = input_ids.reshape(-1, input_ids.shape[-1])
        # 重新组织注意力掩码张量的形状
        attention_mask = attention_mask.reshape(-1, attention_mask.shape[-1])
        # 重新组织标记类型 ID 张量的形状
        token_type_ids = token_type_ids.reshape(-1, token_type_ids.shape[-1])

        # 模型前向传播
        outputs = self.roformer(
            input_ids,
            attention_mask,
            token_type_ids,
            head_mask,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 相当于 PyTorch 实现中的 sequence_summary 调用
        hidden_states = outputs[0]
        # 提取池化输出
        pooled_output = hidden_states[:, -1]
        # 进行 dropout 操作
        pooled_output = self.dropout(pooled_output, deterministic=deterministic)

        # 使用分类器进行分类预测
        logits = self.classifier(pooled_output)

        # 重新组织 logits 张量的形状
        reshaped_logits = logits.reshape(-1, num_choices)

        # 如果不返回字典，则返回相应的输出
        if not return_dict:
            return (reshaped_logits,) + outputs[2:]

        # 返回 FlaxMultipleChoiceModelOutput 对象
        return FlaxMultipleChoiceModelOutput(
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
@add_start_docstrings(
    """
    RoFormer Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
    softmax) e.g. for RocStories/SWAG tasks.
    """,
    ROFORMER_START_DOCSTRING,
)
# 定义一个 RoFormer 模型，顶部有一个多项选择分类头部（线性层叠在池化输出之上并带有 softmax），用于 RocStories/SWAG 等任务。
class FlaxRoFormerForMultipleChoice(FlaxRoFormerPreTrainedModel):
    module_class = FlaxRoFormerForMultipleChoiceModule


# 重写 __call__ 方法的文档字符串
overwrite_call_docstring(
    FlaxRoFormerForMultipleChoice, ROFORMER_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length")
)
# 添加调用示例的文档字符串
append_call_sample_docstring(
    FlaxRoFormerForMultipleChoice,
    _CHECKPOINT_FOR_DOC,
    FlaxMultipleChoiceModelOutput,
    _CONFIG_FOR_DOC,
)


class FlaxRoFormerForTokenClassificationModule(nn.Module):
    config: RoFormerConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        # 设置 RoFormer 模型
        self.roformer = FlaxRoFormerModule(config=self.config, dtype=self.dtype)
        # 设置 dropout 层
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout_prob)
        # 设置分类器
        self.classifier = nn.Dense(self.config.num_labels, dtype=self.dtype)

    def __call__(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        head_mask,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        # 调用 RoFormer 模型
        outputs = self.roformer(
            input_ids,
            attention_mask,
            token_type_ids,
            head_mask,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取隐藏状态并应用 dropout
        hidden_states = outputs[0]
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        # 通过分类器生成 logits
        logits = self.classifier(hidden_states)

        if not return_dict:
            return (logits,) + outputs[1:]

        # 返回 Token 分类器输出
        return FlaxTokenClassifierOutput(
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings(
    """
    RoFormer Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
    Named-Entity-Recognition (NER) tasks.
    """,
    ROFORMER_START_DOCSTRING,
)
# 定义一个 RoFormer 模型，顶部有一个标记分类头部（线性层叠在隐藏状态输出之上），用于命名实体识别（NER）等任务。
class FlaxRoFormerForTokenClassification(FlaxRoFormerPreTrainedModel):
    module_class = FlaxRoFormerForTokenClassificationModule


# 添加调用示例的文档字符串
append_call_sample_docstring(
    FlaxRoFormerForTokenClassification,
    _CHECKPOINT_FOR_DOC,
    FlaxTokenClassifierOutput,
    _CONFIG_FOR_DOC,
)


class FlaxRoFormerForQuestionAnsweringModule(nn.Module):
    config: RoFormerConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        # 设置 RoFormer 模型
        self.roformer = FlaxRoFormerModule(config=self.config, dtype=self.dtype)
        # 设置问答输出
        self.qa_outputs = nn.Dense(self.config.num_labels, dtype=self.dtype)
```  
    def __call__(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        head_mask,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        # 定义一个函数，接受多个参数，其中包括输入的文本id，注意力掩码，标记类型id，头掩码，确定性标志，输出注意力标志，输出隐藏状态标志，返回字典标志

        # 使用 RoFormer 模型处理输入数据
        outputs = self.roformer(
            input_ids,
            attention_mask,
            token_type_ids,
            head_mask,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取模型输出的隐藏状态
        hidden_states = outputs[0]

        # 使用 QA 输出层获取逻辑回归值
        logits = self.qa_outputs(hidden_states)
        # 分割逻辑回归值，得到起始位置和结束位置的预测
        start_logits, end_logits = logits.split(self.config.num_labels, axis=-1)
        # 压缩起始位置和结束位置的维度
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        # 如果不返回字典，则返回元组和额外的输出
        if not return_dict:
            return (start_logits, end_logits) + outputs[1:]

        # 返回FlaxQuestionAnsweringModelOutput对象，包含起始位置的预测，结束位置的预测，隐藏状态和注意力
        return FlaxQuestionAnsweringModelOutput(
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 添加关于 RoFormer 模型的文档字符串，用于提取式问答任务，如 SQuAD 数据集
# 包括在隐藏状态输出之上的线性层，用于计算“起始位置对数”和“结束位置对数”
class FlaxRoFormerForQuestionAnswering(FlaxRoFormerPreTrainedModel):
    # 模块类为 FlaxRoFormerForQuestionAnsweringModule
    module_class = FlaxRoFormerForQuestionAnsweringModule

# 添加样本调用文档字符串
append_call_sample_docstring(
    FlaxRoFormerForQuestionAnswering,
    _CHECKPOINT_FOR_DOC,
    FlaxQuestionAnsweringModelOutput,
    _CONFIG_FOR_DOC,
)
```