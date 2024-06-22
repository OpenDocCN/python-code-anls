# `.\transformers\models\longt5\modeling_flax_longt5.py`

```py
# 定义了编码器-解码器模型的配置类，继承自LongT5Config
class LongT5Config:

    # 构造函数，接收一个参数
    def __init__(
        self,
        vocab_size: int = 32128,
        d_model: int = 1024,
        d_ff: int = 4096,
        num_layers: int = 24,
        num_heads: int = 16,
        relative_attention_num_buckets: int = 32,
        dropout_rate: float = 0.0,
        layer_norm_epsilon: float = 1e-6,
        pad_token_id: int = 0,
        eos_token_id: int = 1,
        max_position_embeddings: int = 4096,
        axial_pos_embds: bool = False,
        axial_pos_shape: Tuple[int, int] = (64, 64),
        axial_pos_embds_dim: Tuple[int, int] = (64, 192),
        use_cache: bool = True,
        return_dict: bool = True,
        seed: Optional[int] = None,
        bos_token_id: int = 0,
        **kwargs
    ):
        # 初始化编码器-解码器模型的参数
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.relative_attention_num_buckets = relative_attention_num_buckets
        self.dropout_rate = dropout_rate
        self.layer_norm_epsilon = layer_norm_epsilon
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.max_position_embeddings = max_position_embeddings
        self.axial_pos_embds = axial_pos_embds
        self.axial_pos_shape = axial_pos_shape
        self.axial_pos_embds_dim = axial_pos_embds_dim
        self.use_cache = use_cache
        self.return_dict = return_dict
        self.seed = seed
        self.bos_token_id = bos_token_id
        # 其他参数
        for key, value in kwargs.items():
            setattr(self, key, value)
    """
    # 如果张量的指定轴长度不是 block_len 的倍数，则使用选定的 pad_value 进行填充
    # 对张量进行填充，使其长度成为 block_len 的倍数
    if x.shape[axis] % block_len != 0:
        x = _pad_to_multiple(x, block_len, axis, pad_value=0)
    # 计算得到分块的数量
    num_blocks = x.shape[axis] // block_len
    # 计算输出形状
    output_shape = x.shape[:axis] + (num_blocks, block_len) + x.shape[(axis + 1) :]
    # 返回重新调整形状后的张量
    return x.reshape(output_shape)
# 将三个连续的块拼接成本地注意力所需的输入格式
def _concatenate_3_blocks(x: jnp.ndarray, block_axis: int, sequence_axis: int, pad_value: int = 0) -> jnp.ndarray:
    """Concatenate three consecutive blocks for each input block for local attentiont.
    For more information, see: https://arxiv.org/pdf/2112.07916.pdf.
    """
    num_blocks = x.shape[block_axis]

    pad = [(0, 0)] * x.ndim
    pad[block_axis] = (1, 1)
    # 在指定轴上填充0值，使得x的形状从[batch_size, num_blocks, block_len]变为[batch_size, num_blocks + 2, block_len]
    x = jnp.pad(x, pad_width=pad, mode="constant", constant_values=pad_value)

    blocks_list: List[np.array] = []
    for i in range(3):
        # 通过索引获取指定范围内的数据块
        indices = [slice(0, None)] * x.ndim
        indices[block_axis] = slice(i, i + num_blocks)
        indices = tuple(indices)
        blocks_list.append(x[indices])
    return jnp.concatenate(blocks_list, axis=sequence_axis)  # [batch_size, num_blocks, 3 * block_len, ...]


# 生成用于本地注意力的相对位置编码
def _make_3block_relative_position_ids(block_len: int) -> jnp.ndarray:
    """Makes 3-blocked relative position ids for local attention."""
    position_ids = jnp.arange(3 * block_len, dtype=jnp.int32)
    center_position_ids = position_ids[block_len:-block_len]
    relative_position_ids = position_ids[None, :] - center_position_ids[:, None]  # [block_len, 3 * block_len]
    return relative_position_ids


# 将本地注意力掩码限制在指定范围内
def _mask_local_attention_mask(local_attention_mask: np.ndarray, block_len: int) -> jnp.ndarray:
    """Mask local attention mask to enforce that tokens are not allowed to attend tokens farther than ``local_radius."""
    relative_position_ids = _make_3block_relative_position_ids(block_len)
    locality_mask = jnp.abs(relative_position_ids) < block_len
    locality_mask = locality_mask[None, None, :, :]
    return jnp.logical_and(local_attention_mask, locality_mask)


# 准备应用于本地注意力的注意力掩码
def _get_local_attention_mask(attention_mask: np.ndarray, block_len: int) -> jnp.ndarray:
    """Prepare attention mask to be applied for a local attention."""
    # [batch_size, num_blocks, block_len] -> 拆分成多个块
    _blocked_attention_mask = _split_into_blocks(attention_mask, block_len, axis=1)
    # [batch_size, num_block, 3 * block_len] -> 拼接成3个块
    _3blocked_attention_mask = _concatenate_3_blocks(_blocked_attention_mask, block_axis=1, sequence_axis=2)

    _blocked_attention_mask = _blocked_attention_mask[..., None]
    _3blocked_attention_mask = _3blocked_attention_mask[..., None, :]
    # 在适当位置放置掩码，以实现本地注意力
    local_attention_mask = jnp.logical_and(_blocked_attention_mask, _3blocked_attention_mask)
    local_attention_mask = _mask_local_attention_mask(local_attention_mask, block_len)
    # 扩展维度，返回本地注意力掩码
    return local_attention_mask[:, None, ...]


# 生成全局的固定块ID
def _make_global_fixed_block_ids(attention_mask: np.ndarray, global_block_size: int) -> Tuple[jnp.ndarray, np.ndarray]:
    """Obtain the "fixed block" global id corresponding to each input token.

    This implementation is a simlified version of the original Flaxformr implementation adopted from:
    https://github.com/google/flaxformer/blob/main/flaxformer/architectures/longt5/long_attention.py.

    In our scenario, as we use this strategy only for a decoder, orphan tokens, i.e. those tokens which do not make for
    the whole fixed block, are assigned to the preceding block.

    Padding tokens from the original sequence are represented by -1.
    """
    # 获取注意力掩码张量的批量大小和序列长度
    batch_size, seq_len = attention_mask.shape[:2]

    # 处理孤立的标记（orphan tokens）的函数
    def handle_orphan_tokens(block_ids: np.ndarray) -> jnp.ndarray:
        # 计算每个块的结尾
        block_ends = (jnp.arange(seq_len) % global_block_size) == global_block_size - 1
        # 找到真正的块结尾，并且这些块是完整的
        true_block_ends = jnp.logical_and(block_ends, block_ids >= 0)
        # 计算完整块的数量
        full_blocks = true_block_ends.sum(-1)[..., None]
        # 将孤立标记的块索引调整到前一个完整块内
        block_ids = jnp.minimum(block_ids, full_blocks - 1)
        return block_ids

    # 创建固定块掩码，每个位置的值为 1 / 全局块大小
    fixed_block_mask = jnp.ones_like(attention_mask) / global_block_size
    # 累计固定块掩码，得到每个位置所在的块的编号
    fixed_block_mask = jnp.cumsum(fixed_block_mask, axis=1) - fixed_block_mask
    # 创建掩码张量，其中非零位置为 1.0，零位置为 -1000.0
    mask = jnp.where(attention_mask != 0.0, 1.0, -1000.0)
    # 计算全局块编号
    global_block_ids = jnp.maximum(
        jnp.floor(mask + fixed_block_mask - 1.0), jnp.array(-1.0, dtype=attention_mask.dtype)
    )
    # 将填充标记设为 -1
    global_block_ids = (global_block_ids * attention_mask) + (attention_mask - 1)
    # 处理孤立标记
    global_block_ids = handle_orphan_tokens(global_block_ids)
    # 计算全局块的数量
    num_globals = seq_len // global_block_size

    # 创建全局块段编号张量
    if num_globals > 0:
        # 若存在全局块，则最大块编号为每个样本的最大块编号
        _sequence_block_ids_max = jnp.repeat(global_block_ids.max(axis=-1)[:, None], repeats=num_globals, axis=1)
    else:
        # 若不存在全局块，则设为空数组
        _sequence_block_ids_max = jnp.zeros((batch_size, 0), dtype=global_block_ids.dtype)
    # 创建全局段编号张量
    global_segment_ids = jnp.cumsum(jnp.ones((batch_size, num_globals)), axis=-1) - 1
    # 将全局段编号张量中小于等于最大块编号的位置设为 1，否则为 0
    global_segment_ids = jnp.where(global_segment_ids <= _sequence_block_ids_max, 1, 0)
    # 返回全局块编号张量和全局段编号张量
    return global_block_ids, global_segment_ids
```  
# 创建用于局部到全局注意力的相对位置张量
def _make_side_relative_position_ids(attention_mask: np.ndarray, global_block_size: int) -> np.ndarray:
    # 创建全局固定块的块ID和全局段ID
    block_ids, global_segment_ids = _make_global_fixed_block_ids(attention_mask, global_block_size)
    # 获取全局序列长度
    global_seq_len = global_segment_ids.shape[-1]
    # 创建全局位置张量
    global_positions = jnp.arange(global_seq_len)
    # 计算侧面相对位置
    side_relative_position = global_positions - block_ids[..., None]
    # 返回侧面相对位置张量
    return side_relative_position

# 计算通过对个体块求和来计算单个块聚合的全局聚合
def _create_global_aggregates(hidden_states: np.ndarray, block_ids: np.ndarray, global_seq_len: int) -> np.ndarray:
    # (batch..., seq_len, global_seq_len)
    # 创建 one-hot 形式的块ID张量
    one_hot_block_ids = jax.nn.one_hot(block_ids, global_seq_len)
    # 使用矩阵乘法计算块聚合结果
    return jnp.einsum("...nd,...ng->...gd", hidden_states, one_hot_block_ids)


# 根据 LongT5 风格构建的层归一化模块，无偏置和去均值操作
class FlaxLongT5LayerNorm(nn.Module):
    hidden_size: int
    dtype: jnp.dtype = jnp.float32
    eps: float = 1e-6
    weight_init: Callable[..., np.ndarray] = jax.nn.initializers.ones

    def setup(self):
        # 初始化权重
        self.weight = self.param("weight", self.weight_init, (self.hidden_size,))

    def __call__(self, hidden_states):
        """
        构建一个 LongT5 风格的层归一化模块；无偏置和去均值操作。
        """
        # 归一化输入张量为 float32 类型
        variance = jnp.power(hidden_states.astype("f4"), 2).mean(axis=-1, keepdims=True)
        hidden_states = hidden_states / jnp.sqrt(variance + self.eps)

        return self.weight * hidden_states


# 根据 LongT5 风格构建的全连接激活层
class FlaxLongT5DenseActDense(nn.Module):
    config: LongT5Config
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        # 初始化权重标准差
        wi_init_std = self.config.initializer_factor * (self.config.d_model**-0.5)
        wo_init_std = self.config.initializer_factor * (self.config.d_ff**-0.5)

        # 创建输入层和输出层
        self.wi = nn.Dense(
            self.config.d_ff,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(wi_init_std),
            dtype=self.dtype,
        )
        self.wo = nn.Dense(
            self.config.d_model,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(wo_init_std),
            dtype=self.dtype,
        )
        # 创建 Dropout 层
        self.dropout = nn.Dropout(self.config.dropout_rate)
        # 创建激活函数
        self.act = ACT2FN[self.config.dense_act_fn]

    def __call__(self, hidden_states, deterministic=True):
        # 输入层计算
        hidden_states = self.wi(hidden_states)
        # 激活函数
        hidden_states = self.act(hidden_states)
        # Dropout 操作
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        # 输出层计算
        hidden_states = self.wo(hidden_states)
        # 返回结果
        return hidden_states


# 根据 LongT5 风格构建的全连接激活门控层
class FlaxLongT5DenseGatedActDense(nn.Module):
# 定义一个自定义的 FlaxLongT5DenseGatedActDense 类，继承自 nn.Module
class FlaxLongT5DenseGatedActDense(nn.Module):
    # 初始化类的属性 config 为 LongT5Config 类型
    config: LongT5Config
    # 设置数据类型为 jnp.float32，默认值为 jnp.float32
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    # 定义 setup 方法
    def setup(self):
        # 根据配置参数设置 wi_init_std 和 wo_init_std
        wi_init_std = self.config.initializer_factor * (self.config.d_model**-0.5)
        wo_init_std = self.config.initializer_factor * (self.config.d_ff**-0.5)

        # 创建 nn.Dense 层 wi_0，设置相关参数
        self.wi_0 = nn.Dense(
            self.config.d_ff,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(wi_init_std),
            dtype=self.dtype,
        )
        # 创建 nn.Dense 层 wi_1，设置相关参数
        self.wi_1 = nn.Dense(
            self.config.d_ff,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(wi_init_std),
            dtype=self.dtype,
        )
        # 创建 nn.Dense 层 wo，设置相关参数
        self.wo = nn.Dense(
            self.config.d_model,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(wo_init_std),
            dtype=self.dtype,
        )
        # 创建 nn.Dropout 层，设置 dropout rate
        self.dropout = nn.Dropout(self.config.dropout_rate)
        # 根据配置参数获取激活函数
        self.act = ACT2FN[self.config.dense_act_fn]

    # 定义 __call__ 方法，对输入进行前向传播
    def __call__(self, hidden_states, deterministic):
        # 应用激活函数至 wi_0(hidden_states)，得到 hidden_gelu
        hidden_gelu = self.act(self.wi_0(hidden_states))
        # 计算 hidden_linear = wi_1(hidden_states)
        hidden_linear = self.wi_1(hidden_states)
        # 计算 hidden_states = hidden_gelu * hidden_linear
        hidden_states = hidden_gelu * hidden_linear
        # 对 hidden_states 进行 dropout 操作
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        # 计算输出结果 hidden_states = self.wo(hidden_states)
        hidden_states = self.wo(hidden_states)
        # 返回计算结果
        return hidden_states


# 定义一个自定义的 FlaxLongT5LayerFF 类，继承自 nn.Module
class FlaxLongT5LayerFF(nn.Module):
    # 初始化类的属性 config 为 LongT5Config 类型
    config: LongT5Config
    # 设置数据类型为 jnp.float32，默认值为 jnp.float32
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    # 定义 setup 方法
    def setup(self):
        # 根据配置参数设置 DenseReluDense 层，根据是否存在 gated activation 决定使用��种 Dense 类
        if self.config.is_gated_act:
            self.DenseReluDense = FlaxLongT5DenseGatedActDense(self.config, dtype=self.dtype)
        else:
            self.DenseReluDense = FlaxLongT5DenseActDense(self.config, dtype=self.dtype)

        # 创建 FlaxLongT5LayerNorm 层，设置相关参数
        self.layer_norm = FlaxLongT5LayerNorm(
            self.config.d_model, eps=self.config.layer_norm_epsilon, dtype=self.dtype
        )
        # 创建 nn.Dropout 层，设置 dropout rate
        self.dropout = nn.Dropout(self.config.dropout_rate)

    # 定义 __call__ 方法，对输入进行前向传播
    def __call__(self, hidden_states, deterministic=True):
        # 应用 LayerNorm 到 hidden_states 上
        forwarded_states = self.layer_norm(hidden_states)
        # 传播 hidden_states 通过 DenseReluDense 层
        forwarded_states = self.DenseReluDense(forwarded_states, deterministic=deterministic)
        # 计算 hidden_states = hidden_states + self.dropout(forwarded_states)
        hidden_states = hidden_states + self.dropout(forwarded_states, deterministic=deterministic)
        # 返回计算结果
        return hidden_states


# 定义一个自定义的 FlaxLongT5Attention 类，继承自 nn.Module
class FlaxLongT5Attention(nn.Module):
    # 初始化类的属性 config 为 LongT5Config 类型
    config: LongT5Config
    # 是否存在 relative attention bias，默认为 False
    has_relative_attention_bias: bool = False
    # 是否为因果 attention，默认为 False
    causal: bool = False
    # 设置数据类型为 jnp.float32，默认为 jnp.float32
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation
    # 定义一个 setup 方法，用于设置对象的属性
    def setup(self):
        # 从配置中获取相对注意力的桶数量
        self.relative_attention_num_buckets = self.config.relative_attention_num_buckets
        # 从配置中获取相对注意力的最大距离
        self.relative_attention_max_distance = self.config.relative_attention_max_distance
        # 从配置中获取模型的维度
        self.d_model = self.config.d_model
        # 从配置中获取键值投影的维度
        self.key_value_proj_dim = self.config.d_kv
        # 从配置中获取多头注意力的头数
        self.n_heads = self.config.num_heads
        # 从配置中获取丢弃率
        self.dropout = self.config.dropout_rate
        # 计算内在维度为多头注意力的头数乘以键值投影维度
        self.inner_dim = self.n_heads * self.key_value_proj_dim

        # 根据初始化因子和维度计算查询向量的标准差
        q_init_std = self.config.initializer_factor * ((self.inner_dim * self.key_value_proj_dim) ** -0.5)
        # 根据初始化因子和维度计算键值向量的标准差
        kv_init_std = self.config.initializer_factor * (self.inner_dim**-0.5)
        # 根据初始化因子和维度计算输出向量的标准差
        o_init_std = self.config.initializer_factor * (self.inner_dim**-0.5)

        # 使用标准差创建查询向量的密集层，无偏差
        self.q = nn.Dense(
            self.inner_dim,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(q_init_std),
            dtype=self.dtype,
        )
        # 使用标准差创建键向量的密集层，无偏差
        self.k = nn.Dense(
            self.inner_dim,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(kv_init_std),
            dtype=self.dtype,
        )
        # 使用标准差创建值向量的密集层，无偏差
        self.v = nn.Dense(
            self.inner_dim,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(kv_init_std),
            dtype=self.dtype,
        )
        # 使用标准差创建输出向量的密集层，无偏差
        self.o = nn.Dense(
            self.d_model,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(o_init_std),
            dtype=self.dtype,
        )

        # 如果存在相对注意力偏置
        if self.has_relative_attention_bias:
            # 创建相对注意力偏置的嵌入层
            self.relative_attention_bias = nn.Embed(
                self.relative_attention_num_buckets,
                self.n_heads,
                embedding_init=jax.nn.initializers.normal(kv_init_std),
                dtype=self.dtype,
            )

    # 定义一个静态方法，代码在此处省略
    @staticmethod
    # 将相对位置映射到相对注意力的桶号
    def _relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        """
        从 Mesh TensorFlow 改编来的：
        https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593

        将相对位置转换为用于相对注意力的桶号。相对位置定义为 memory_position - query_position，即从参与位置到被参与位置的令牌距离。
        如果 bidirectional=False，则正的相对位置是无效的。我们对小的绝对相对位置使用较小的桶，并对较大的绝对相对位置使用较大的桶。所有相对位置 >=max_distance 映射到相同的桶。
        所有相对位置 <=-max_distance 映射到相同的桶。这应该允许模型更优雅地推广到比其训练过的更长序列
        """
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0) * num_buckets
            relative_position = jnp.abs(relative_position)
        else:
            relative_position = -jnp.clip(relative_position, a_max=0)
        # 现在 relative_position 在范围 [0, inf) 内

        # 一半的桶用于精确的位置增量
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        # 另一半桶用于位置增至 max_distance 时对数上更大的位置
        relative_position_if_large = max_exact + (
            jnp.log(relative_position / max_exact) / jnp.log(max_distance / max_exact) * (num_buckets - max_exact)
        )
        relative_position_if_large = jnp.clip(relative_position_if_large, a_max=num_buckets - 1)

        relative_buckets += jnp.where(is_small, relative_position, relative_position_if_large)

        return relative_buckets.astype("i4")

    def compute_bias(self, query_length, key_length):
        """计算分桶相对位置偏置"""
        context_position = jnp.arange(query_length, dtype="i4")[:, None]
        memory_position = jnp.arange(key_length, dtype="i4")[None, :]

        relative_position = memory_position - context_position
        relative_position_bucket = self._relative_position_bucket(
            relative_position,
            bidirectional=(not self.causal),
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance,
        )

        values = self.relative_attention_bias(relative_position_bucket)
        values = values.transpose((2, 0, 1))[None, :, :, :]
        return values

    def _split_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.n_heads, self.key_value_proj_dim))
    # 将隐藏状态重新调整为指定形状的函数
    def _merge_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.inner_dim,))

    # 在缓存状态中连接并保存传入的键、值和查询的函数
    @nn.compact
    def _concatenate_to_cache(self, key, value, query, attention_mask):
        """
        此函数将来自单个输入标记的投影键、值状态与先前步骤中的缓存状态连接起来并保存。
        此函数略微改编自官方的Flax存储库：
        https://github.com/google/flax/blob/491ce18759622506588784b4fca0e4bf05f8c8cd/flax/linen/attention.py#L252
        """
        # 通过缺少现有缓存数据来检测是否正在初始化
        is_initialized = self.has_variable("cache", "cached_key")
        # 检测并获取缓存的键
        cached_key = self.variable("cache", "cached_key", jnp.zeros, key.shape, key.dtype)
        # 检测并获取缓存的值
        cached_value = self.variable("cache", "cached_value", jnp.zeros, value.shape, value.dtype)
        # 检测并获取缓存的索引
        cache_index = self.variable("cache", "cache_index", lambda: jnp.array(0, dtype=jnp.int32))

        if is_initialized:
            # 获取批处理维度、最大长度、头数和每个头的深度
            *batch_dims, max_length, num_heads, depth_per_head = cached_key.value.shape
            # 使用新的一维空间切片更新键、值缓存
            cur_index = cache_index.value
            indices = (0,) * len(batch_dims) + (cur_index, 0, 0)
            key = jax.lax.dynamic_update_slice(cached_key.value, key, indices)
            value = jax.lax.dynamic_update_slice(cached_value.value, value, indices)
            cached_key.value = key
            cached_value.value = value
            num_updated_cache_vectors = query.shape[1]
            cache_index.value = cache_index.value + num_updated_cache_vectors
            # 用于缓存的因果注意力掩码：我们的单个查询位置应该只与已生成和缓存的键位置相对应，而不是剩余的零元素
            pad_mask = jnp.broadcast_to(
                jnp.arange(max_length) < cur_index + num_updated_cache_vectors,
                tuple(batch_dims) + (1, num_updated_cache_vectors, max_length),
            )
            # 合并掩码
            attention_mask = combine_masks(pad_mask, attention_mask)
        返回键、值和注意力掩码
        return key, value, attention_mask

    # 创建位置偏移量的函数
    def _create_position_bias(
        self, key_states, query_states, attention_mask, init_cache, seq_length, causal_attention_mask_shift
        ):
        # 判断缓存是否已填充，且存在缓存的关键键和不需要初始化缓存
        cache_is_filled = self.causal and self.has_variable("cache", "cached_key") and (not init_cache)
        # 计算键值的长度
        key_length = key_states.shape[1]
        # 如果缓存已填充，则查询长度等于键值的长度，否则等于查询状态的长度
        query_length = key_length if cache_is_filled else query_states.shape[1]

        # 如果具有相对注意偏置
        if self.has_relative_attention_bias:
            # 计算位置偏置
            position_bias = self.compute_bias(query_length, key_length)
        # 如果存在注意力掩码
        elif attention_mask is not None:
            # 创建全零位置偏置与注意力掩码形状相同
            position_bias = jnp.zeros_like(attention_mask)
        else:
            # 创建全零位置偏置，形状为 (1, self.n_heads, query_length, key_length)，数据类型为 self.dtype
            position_bias = jnp.zeros((1, self.n_heads, query_length, key_length), dtype=self.dtype)

        # 如果缓存已填充，只需要最后一个查询位置的偏置
        if cache_is_filled:
            # 计算最大解码器长度
            max_decoder_length = self.variables["cache"]["cached_key"].shape[1]
            # 从位置偏置中动态切片出需要的部分，形状为 (1, self.n_heads, seq_length, max_decoder_length)
            position_bias = jax.lax.dynamic_slice(
                position_bias,
                (0, 0, causal_attention_mask_shift, 0),
                (1, self.n_heads, seq_length, max_decoder_length),
            )
        # 返回位置偏置
        return position_bias

    # 定义 __call__ 方法
    def __call__(
        self,
        hidden_states,
        attention_mask=None,
        key_value_states=None,
        position_bias=None,
        use_cache=False,
        output_attentions=False,
        deterministic=True,
        init_cache=False,
class FlaxLongT5LocalAttention(nn.Module):
    config: LongT5Config  # 声明一个名为config的类型为LongT5Config的属性
    has_relative_attention_bias: bool = False  # 声明一个名为has_relative_attention_bias的布尔类型属性，默认值为False
    dtype: jnp.dtype = jnp.float32  # 声明一个名为dtype的属性，类型为jnp.dtype，默认值为jnp.float32（计算的数据类型）

    def setup(self):  # 声明一个名为setup的方法，用于初始化模型参数
        self.relative_attention_num_buckets = self.config.relative_attention_num_buckets  # 设置相对注意力的桶数
        self.relative_attention_max_distance = self.config.relative_attention_max_distance  # 设置相对注意力的最大距离
        self.d_model = self.config.d_model  # 设置模型维度
        self.key_value_proj_dim = self.config.d_kv  # 设置键值映射维度
        self.n_heads = self.config.num_heads  # 设置头数
        self.local_radius = self.config.local_radius  # 设置局部半径
        self.block_len = self.local_radius + 1  # 设置局部块长度
        self.dropout = self.config.dropout_rate  # 设置dropout比率
        self.inner_dim = self.n_heads * self.key_value_proj_dim  # 计算内部维度

        # 根据配置初始化标准差
        q_init_std = self.config.initializer_factor * ((self.inner_dim * self.key_value_proj_dim) ** -0.5)
        kv_init_std = self.config.initializer_factor * (self.inner_dim**-0.5)
        o_init_std = self.config.initializer_factor * (self.inner_dim**-0.5)

        # 创建可学习的全连接层，并初始化权重
        self.q = nn.Dense(
            self.inner_dim,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(q_init_std),
            dtype=self.dtype,
        )
        self.k = nn.Dense(
            self.inner_dim,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(kv_init_std),
            dtype=self.dtype,
        )
        self.v = nn.Dense(
            self.inner_dim,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(kv_init_std),
            dtype=self.dtype,
        )
        self.o = nn.Dense(
            self.d_model,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(o_init_std),
            dtype=self.dtype,
        )

        # 如果存在相对注意力偏置，创建相对注意力偏置的Embed层并初始化权重
        if self.has_relative_attention_bias:
            self.relative_attention_bias = nn.Embed(
                self.relative_attention_num_buckets,
                self.n_heads,
                embedding_init=jax.nn.initializers.normal(kv_init_std),
            )

    @staticmethod
    # 从transformers.models.t5.modeling_flax_t5.FlaxT5Attention._relative_position_bucket中复制代码
    # 定义一个私有方法，用来将相对位置转换成相对注意力的桶编号
    def _relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        """
        修改自 Mesh Tensorflow:
        https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593

        将相对位置转换为相对注意力的桶编号。相对位置定义为 memory_position - query_position，即从参与位置到被关注位置的距离（令牌数）。如果 bidirectional=False，则正的相对位置无效。我们对小的绝对 relative_position 使用较小的桶，对大的绝对 relative_position 则使用较大的桶。所有大于等于max_distance 的相对位置将映射到同一个桶。所有小于等于-max_distance 的 相对位置也将映射到同一个桶。这应该允许在模型训练过的更长序列上更优雅地推广。
        """
        relative_buckets = 0
        # 如果是双向的，则将桶的数量减半，并且根据相对位置是正数还是负数确定所在的位置
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0) * num_buckets
            relative_position = jnp.abs(relative_position)
        else:
            # 如果不是双向的，即不支持正的相对位置，将相对位置取负数，并clip到大于等于0的范围内
            relative_position = -jnp.clip(relative_position, a_max=0)
        # 现在相对位置都位于[0, inf)的范围内

        # 桶的一半用于精确增量的位置
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        # 另一半的桶用于位置对数增长的大 bins，直到 max_distance
        relative_position_if_large = max_exact + (
            jnp.log(relative_position / max_exact) / jnp.log(max_distance / max_exact) * (num_buckets - max_exact)
        )
        # 将相对位置限制在 [0, num_buckets-1] 的范围内
        relative_position_if_large = jnp.clip(relative_position_if_large, a_max=num_buckets - 1)

        # 根据相对位置的大小判断将其放入哪个桶中
        relative_buckets += jnp.where(is_small, relative_position, relative_position_if_large)

        # 返回相对位置桶编号的整数表示，即将数据类型转换为"i4"
        return relative_buckets.astype("i4")

    # 计算分段的相对位置偏置
    def compute_bias(self, block_length: int):
        """Compute binned relative position bias"""
        # 生成一个长度为 3 * block_length 的整数数组，表示记忆位置
        memory_position = jnp.arange(3 * block_length, dtype="i4")
        # 计算上下文位置，即除去前 block_length 和后 block_length 的位置
        context_position = memory_position[block_length:-block_length]

        # 计算相对位置，即记忆位置与上下文位置的差，生成一个矩阵
        relative_position = memory_position[None, :] - context_position[:, None]
        # 将相对位置转换成对应的相对位置桶编号
        relative_position_bucket = self._relative_position_bucket(
            relative_position,
            bidirectional=True,
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance,
        )

        # 用相对位置桶编号调用相对注意力偏置函数，得到偏置值
        values = self.relative_attention_bias(relative_position_bucket)
        # 将维度顺序调整为 (2, 0, 1)
        values = values.transpose((2, 0, 1))[None, None, :, :, :]
        # 返回偏置值
        return values

    # 将隐藏状态拆分成多头
    def _split_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.n_heads, self.key_value_proj_dim))
    # 将隐藏状态重塑成指定形状
    def _merge_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[0], -1, self.inner_dim)

    # 创建用于位置偏置的数组
    def _create_position_bias(self, block_len: int, attention_mask: Optional[np.ndarray]) -> np.ndarray:
        # 如果存在相对注意力偏差，则计算偏差
        if self.has_relative_attention_bias:
            position_bias = self.compute_bias(block_len)
        # 如果存在注意力遮罩，则使用全零数组
        elif attention_mask is not None:
            position_bias = jnp.zeros_like(attention_mask)
        else:
            # 否则，使用指定形状和数据类型创建全零数组
            position_bias = jnp.zeros((1, 1, self.n_heads, block_len, 3 * block_len), dtype=self.dtype)

        return position_bias

    # 定义 __call__ 方法，执行特定操作
    def __call__(
        self,
        hidden_states,
        attention_mask=None,
        key_value_states=None,
        position_bias=None,
        output_attentions=False,
        deterministic=True,
# 定义一个名为FlaxLongT5TransientGlobalAttention的类，继承自nn.Module
class FlaxLongT5TransientGlobalAttention(nn.Module):
    # 类属性
    config: LongT5Config  # 引用LongT5Config类的实例
    has_relative_attention_bias: bool = False  # 是否具有相对注意力偏置，初始值为False
    dtype: jnp.dtype = jnp.float32  # 计算的数据类型，初始值为jnp.float32

    # 定义setup方法
    def setup(self):
        # 设置各种属性的值
        self.relative_attention_num_buckets = self.config.relative_attention_num_buckets
        self.relative_attention_max_distance = self.config.relative_attention_max_distance
        self.d_model = self.config.d_model
        self.key_value_proj_dim = self.config.d_kv
        self.n_heads = self.config.num_heads
        self.local_radius = self.config.local_radius
        self.block_len = self.local_radius + 1
        self.global_block_size = self.config.global_block_size
        self.dropout = self.config.dropout_rate
        self.inner_dim = self.n_heads * self.key_value_proj_dim

        # 计算初始化标准差
        q_init_std = self.config.initializer_factor * ((self.inner_dim * self.key_value_proj_dim) ** -0.5)
        kv_init_std = self.config.initializer_factor * (self.inner_dim**-0.5)
        o_init_std = self.config.initializer_factor * (self.inner_dim**-0.5)

        # 定义全连接层
        self.q = nn.Dense(
            self.inner_dim,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(q_init_std),
            dtype=self.dtype,
        )
        self.k = nn.Dense(
            self.inner_dim,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(kv_init_std),
            dtype=self.dtype,
        )
        self.v = nn.Dense(
            self.inner_dim,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(kv_init_std),
            dtype=self.dtype,
        )
        self.o = nn.Dense(
            self.d_model,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(o_init_std),
            dtype=self.dtype,
        )

        # 如果具有相对注意力偏置，则定义相对注意力偏置的Embed层
        if self.has_relative_attention_bias:
            self.relative_attention_bias = nn.Embed(
                self.relative_attention_num_buckets,
                self.n_heads,
                embedding_init=jax.nn.initializers.normal(kv_init_std),
            )

        # 如果具有相对注意力偏置
        # 则定义全局相对注意力偏置的Embed层
        if self.has_relative_attention_bias:
            self.global_relative_attention_bias = nn.Embed(
                self.relative_attention_num_buckets,
                self.n_heads,
                embedding_init=jax.nn.initializers.normal(kv_init_std),
            )
        # 定义全局注意力的输入层归一化
        self.global_input_layer_norm = FlaxLongT5LayerNorm(
            self.config.d_model, eps=self.config.layer_norm_epsilon, dtype=self.dtype
        )

    # 静态方法
    @staticmethod
    # 从transformers.models.t5.modeling_flax_t5.FlaxT5Attention._relative_position_bucket中复制代码
    #  将相对位置转换为相对注意力的桶编号
    def _relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        """
        从 Mesh Tensorflow 中调整而来：
        https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593
    
        将相对位置转换为相对注意力的桶编号。相对位置定义为 memory_position - query_position，即从关注位置到被关注位置的 token 距离。
        如果 bidirectional=False，则正的相对位置是无效的。对于较小的绝对相对位置，我们使用较小的桶，而对于较大的绝对相对位置，我们使用较大的桶。
        所有相对位置 >=max_distance 映射到同一个桶。所有相对位置 <=-max_distance 映射到同一个桶。
        这应该允许更加平稳地推广到比模型训练时更长的序列
        """
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0) * num_buckets
            relative_position = jnp.abs(relative_position)
        else:
            relative_position = -jnp.clip(relative_position, a_max=0)
        # 现在 relative_position 在范围 [0, 无穷)
    
        # 一半的桶用于确切的位置增量
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact
    
        # 另一半的桶用于相对于 max_distance 的对数增加的位置
        relative_position_if_large = max_exact + (
            jnp.log(relative_position / max_exact) / jnp.log(max_distance / max_exact) * (num_buckets - max_exact)
        )
        relative_position_if_large = jnp.clip(relative_position_if_large, a_max=num_buckets - 1)
    
        relative_buckets += jnp.where(is_small, relative_position, relative_position_if_large)
    
        return relative_buckets.astype("i4")
    
    # 计算分箱相对位置偏差
    def compute_bias(self, block_length: int):
        """计算分箱相对位置偏差"""
        memory_position = jnp.arange(3 * block_length, dtype="i4")
        context_position = memory_position[block_length:-block_length]
    
        relative_position = memory_position[None, :] - context_position[:, None]
        relative_position_bucket = self._relative_position_bucket(
            relative_position,
            bidirectional=True,
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance,
        )
    
        values = self.relative_attention_bias(relative_position_bucket)
        values = values.transpose((2, 0, 1))[None, None, :, :, :]
        return values
    # 计算侧边注意力偏置，用于考虑全局信息
    def compute_side_bias(self, attention_mask: np.ndarray, global_segment_ids: np.ndarray) -> np.ndarray:
        # 创建侧边注意力的掩码，指示每个位置是否关注全局信息
        # 形状为(batch_size, 1, 1, seq_len, global_seq_len)
        side_attention_mask = jnp.equal(attention_mask[..., None], global_segment_ids[:, None, :])[:, None, ...]
        # 根据侧边注意力掩码生成注意力偏置，关注全局信息的位置偏置为0，否则为负无穷大
        attention_side_bias = jax.lax.select(
            side_attention_mask > 0,
            jnp.full(side_attention_mask.shape, 0.0).astype(self.dtype),
            jnp.full(side_attention_mask.shape, -1e10).astype(self.dtype),
        )
        # 计算相对位置编码，用于全局相对位置注意力
        side_relative_position = _make_side_relative_position_ids(attention_mask, self.global_block_size)
        # 将相对位置编码分桶，以便用于注意力偏置计算
        side_relative_position_bucket = self._relative_position_bucket(
            side_relative_position,
            bidirectional=True,
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance,
        )
        # 根据相对位置编码计算全局相对位置注意力偏置
        side_bias = self.global_relative_attention_bias(side_relative_position_bucket)

        # 将注意力偏置张量进行转置以匹配注意力计算的维度顺序
        # 形状变为(batch_size, 1, num_heads, seq_len, global_seq_len)
        side_bias = jnp.transpose(side_bias, (0, 3, 1, 2))
        # 将侧边注意力偏置与全局相对位置注意力偏置相加，得到最终的注意力偏置
        # 形状为(batch_size, num_heads, seq_len, global_seq_len)
        attention_side_bias = attention_side_bias + side_bias
        return attention_side_bias

    # 将隐藏状态张量按照注意力头数和维度进行分割
    def _split_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.n_heads, self.key_value_proj_dim))

    # 将分割后的注意力头重新合并为隐藏状态张量
    def _merge_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[0], -1, self.inner_dim)

    # 创建位置偏置张量，用于注意力计算
    def _create_position_bias(self, block_len: int, attention_mask: Optional[np.ndarray]) -> np.ndarray:
        # 如果存在相对位置编码，则调用compute_bias方法计算位置偏置
        if self.has_relative_attention_bias:
            position_bias = self.compute_bias(block_len)
        # 如果存在注意力掩码，则初始化位置偏置为与其形状相同的全零张量
        elif attention_mask is not None:
            position_bias = jnp.zeros_like(attention_mask)
        # 否则，初始化位置偏置为全零张量，形状为(1, 1, n_heads, block_len, 3 * block_len)
        else:
            position_bias = jnp.zeros((1, 1, self.n_heads, block_len, 3 * block_len), dtype=self.dtype)

        return position_bias

    # 模型的调用方法，用于执行注意力计算等操作
    def __call__(
        self,
        hidden_states,
        attention_mask=None,
        key_value_states=None,
        position_bias=None,
        output_attentions=False,
        deterministic=True,
# 定义 FlaxLongT5LayerLocalSelfAttention 类，用于编码器中的局部自注意力机制
class FlaxLongT5LayerLocalSelfAttention(nn.Module):
    """Local self attention used in encoder"""
    
    # 配置信息：LongT5Config 类型
    config: LongT5Config
    # 是否有相对注意力偏置
    has_relative_attention_bias: bool = False
    # 计算中的数据类型
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation
    
    def setup(self):
        # 创建 FlaxLongT5LocalAttention 对象，参数传入配置信息等
        self.LocalSelfAttention = FlaxLongT5LocalAttention(
            self.config, has_relative_attention_bias=self.has_relative_attention_bias, dtype=self.dtype
        )
        # 创建 FlaxLongT5LayerNorm 对象，用于层归一化
        self.layer_norm = FlaxLongT5LayerNorm(
            self.config.d_model, eps=self.config.layer_norm_epsilon, dtype=self.dtype
        )
        # 创建 nn.Dropout 对象，用于 dropout
        self.dropout = nn.Dropout(self.config.dropout_rate)

    def __call__(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        output_attentions=False,
        deterministic=True,
        **kwargs: Any,  # to accept init_cache kwargs
    ):
        # 对隐藏状态进行层归一化
        normed_hidden_states = self.layer_norm(hidden_states)
        # 使用局部自注意力机制进行注意力计算
        attention_output = self.LocalSelfAttention(
            normed_hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            output_attentions=output_attentions,
            deterministic=deterministic,
        )
        # 更新隐藏状态，加上 dropout 之后的注意力输出
        hidden_states = hidden_states + self.dropout(attention_output[0], deterministic=deterministic)
        # 输出包括更新后的隐藏状态和注意力信息（如果输出）
        outputs = (hidden_states,) + attention_output[1:]  # add attentions if we output them
        return outputs


# 定义 FlaxLongT5LayerTransientGlobalSelfAttention 类，用于编码器中的瞬态全局自注意力机制
class FlaxLongT5LayerTransientGlobalSelfAttention(nn.Module):
    """Transient-Global self attention used in encoder"""
    
    # 配置信息：LongT5Config 类型
    config: LongT5Config
    # 是否有相对注意力偏置
    has_relative_attention_bias: bool = False
    # 计算中的数据类型
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation
    
    def setup(self):
        # 创建 FlaxLongT5TransientGlobalAttention 对象，参数传入配置信息等
        self.TransientGlobalSelfAttention = FlaxLongT5TransientGlobalAttention(
            self.config, has_relative_attention_bias=self.has_relative_attention_bias, dtype=self.dtype
        )
        # 创建 FlaxLongT5LayerNorm 对象，用于层归一化
        self.layer_norm = FlaxLongT5LayerNorm(
            self.config.d_model, eps=self.config.layer_norm_epsilon, dtype=self.dtype
        )
        # 创建 nn.Dropout 对象，用于 dropout
        self.dropout = nn.Dropout(self.config.dropout_rate)

    def __call__(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        output_attentions=False,
        deterministic=True,
        **kwargs: Any,  # to accept init_cache kwargs
    ):
        # 对隐藏状态进行层归一化
        normed_hidden_states = self.layer_norm(hidden_states)
        # 使用瞬态全局自注意力机制进行注意力计算
        attention_output = self.TransientGlobalSelfAttention(
            normed_hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            output_attentions=output_attentions,
            deterministic=deterministic,
        )
        # 更新隐藏状态，加上 dropout 之后的注意力输出
        hidden_states = hidden_states + self.dropout(attention_output[0], deterministic=deterministic)
        # 输出包括更新后的隐藏状态和注意力信息（如果输出）
        outputs = (hidden_states,) + attention_output[1:]  # add attentions if we output them
        return outputs


# 从transformers.models.t5.modeling_flax_t5中复制 FlaxT5LayerSelfAttention ，将T5替换为LongT5
# 定义一个FlaxLongT5LayerSelfAttention类，继承自nn.Module
class FlaxLongT5LayerSelfAttention(nn.Module):
    # 定义config属性，并指定类型为LongT5Config
    config: LongT5Config
    # 定义has_relative_attention_bias属性，并设置默认值为False
    has_relative_attention_bias: bool = False
    # 定义dtype属性，并指定类型为jnp.dtype，默认值为jnp.float32，表示计算中的数据类型

    # 初始化函数
    def setup(self):
        # 创建SelfAttention对象，调用FlaxLongT5Attention类
        self.SelfAttention = FlaxLongT5Attention(
            self.config,
            has_relative_attention_bias=self.has_relative_attention_bias,
            causal=self.config.causal,
            dtype=self.dtype,
        )
        # 创建layer_norm对象，调用FlaxLongT5LayerNorm类
        self.layer_norm = FlaxLongT5LayerNorm(
            self.config.d_model, eps=self.config.layer_norm_epsilon, dtype=self.dtype
        )
        # 创建dropout对象，调用nn.Dropout类
        self.dropout = nn.Dropout(self.config.dropout_rate)

    # 调用函数
    def __call__(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        output_attentions=False,
        deterministic=True,
        init_cache=False,
    ):
        # 对隐藏状态进行归一化
        normed_hidden_states = self.layer_norm(hidden_states)
        # 调用SelfAttention对象
        attention_output = self.SelfAttention(
            normed_hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            output_attentions=output_attentions,
            deterministic=deterministic,
            init_cache=init_cache,
        )
        # 更新隐藏状态
        hidden_states = hidden_states + self.dropout(attention_output[0], deterministic=deterministic)
        # 输出结果
        outputs = (hidden_states,) + attention_output[1:]  # add attentions if we output them
        return outputs


# 从transformers.models.t5.modeling_flax_t5中复制FlaxT5LayerCrossAttention类，并将T5->LongT5
class FlaxLongT5LayerCrossAttention(nn.Module):
    # 定义config属性，并指定类型为LongT5Config
    config: LongT5Config
    # 定义dtype属性，并指定类型为jnp.dtype，默认值为jnp.float32，表示计算中的数据类型

    # 初始化函数
    def setup(self):
        # 创建EncDecAttention对象，调用FlaxLongT5Attention类
        self.EncDecAttention = FlaxLongT5Attention(
            self.config, has_relative_attention_bias=False, causal=False, dtype=self.dtype
        )
        # 创建layer_norm对象，调用FlaxLongT5LayerNorm类
        self.layer_norm = FlaxLongT5LayerNorm(
            self.config.d_model, eps=self.config.layer_norm_epsilon, dtype=self.dtype
        )
        # 创建dropout对象，调用nn.Dropout类
        self.dropout = nn.Dropout(self.config.dropout_rate)

    # 调用函数
    def __call__(
        self,
        hidden_states,
        key_value_states,
        attention_mask=None,
        position_bias=None,
        output_attentions=False,
        deterministic=True,
    ):
        # 对隐藏状态进行归一化
        normed_hidden_states = self.layer_norm(hidden_states)
        # 调用EncDecAttention对象
        attention_output = self.EncDecAttention(
            normed_hidden_states,
            attention_mask=attention_mask,
            key_value_states=key_value_states,
            position_bias=position_bias,
            output_attentions=output_attentions,
        )
        # 更新隐藏状态
        hidden_states = hidden_states + self.dropout(attention_output[0], deterministic=deterministic)
        # 输出结果
        outputs = (hidden_states,) + attention_output[1:]  # add attentions if we output them
        return outputs


# 定义一个FlaxLongT5Block类，继承自nn.Module
class FlaxLongT5Block(nn.Module):
    # 定义config属性，并指定类型为LongT5Config
    config: LongT5Config
    # 定义has_relative_attention_bias属性，并设置默认值为False
    has_relative_attention_bias: bool = False
    # 定义dtype属性，并指定类型为jnp.dtype，默认值为jnp.float32，表示计算中的数据类型
    # 设置函数，用于初始化模型的一些参数
    def setup(self):
        # 设置causal属性为config中的causal值
        self.causal = self.config.causal
        # 如果causal为True，则使用FlaxLongT5LayerSelfAttention作为attention_layer
        if self.causal:
            attention_layer = FlaxLongT5LayerSelfAttention
        # 如果encoder_attention_type为"local"，则使用FlaxLongT5LayerLocalSelfAttention作为attention_layer
        elif self.config.encoder_attention_type == "local":
            attention_layer = FlaxLongT5LayerLocalSelfAttention
        # 如果encoder_attention_type为"transient-global"，则使用FlaxLongT5LayerTransientGlobalSelfAttention作为attention_layer
        elif self.config.encoder_attention_type == "transient-global":
            attention_layer = FlaxLongT5LayerTransientGlobalSelfAttention
        # 如果以上条件都不满足，则抛出数值错误
        else:
            raise ValueError(
                "For encoder attention mechanism, either `local` or `transient-global` attention type is expected, "
                f"but got {self.config.encoder_attention_type}."
            )
        # 设置self.layer为具体的attention_layer，并且根据是否causal来选择是否添加cross-attention和feed forward层
        self.layer = (
            attention_layer(
                self.config,
                has_relative_attention_bias=self.has_relative_attention_bias,
                name=str(0),
                dtype=self.dtype,
            ),
        )
        feed_forward_index = 1
        # 如果causal为True，则再添加一个FlaxLongT5LayerCrossAttention层
        if self.causal:
            self.layer += (FlaxLongT5LayerCrossAttention(self.config, name=str(1), dtype=self.dtype),)
            feed_forward_index += 1

        # 最后添加一个FlaxLongT5LayerFF层
        self.layer += (FlaxLongT5LayerFF(self.config, name=str(feed_forward_index), dtype=self.dtype),)

    # 从transformers.models.t5.modeling_flax_t5中的FlaxT5Block.__call__复制得到的函数，参数作用同原函数
    def __call__(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        encoder_decoder_position_bias=None,
        output_attentions=False,
        return_dict=True,
        deterministic=True,
        init_cache=False,
    ):
        # 使用 self-attention 层处理隐藏状态，包括注意力掩码、位置偏置等参数
        self_attention_outputs = self.layer[0](
            hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            output_attentions=output_attentions,
            deterministic=deterministic,
            init_cache=init_cache,
        )
        # 获取经过 self-attention 处理后的隐藏状态
        hidden_states = self_attention_outputs[0]
        # 保留 self-attention 输出和相对位置权重
        attention_outputs = self_attention_outputs[1:]  # Keep self-attention outputs and relative position weights

        # 如果是有交叉注意力的情况，并且编码器的隐藏状态不为空
        do_cross_attention = self.causal and encoder_hidden_states is not None
        if do_cross_attention:
            # 使用交叉注意力层处理隐藏状态，包括键值状态、编码器注意力掩码等参数
            cross_attention_outputs = self.layer[1](
                hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                position_bias=encoder_decoder_position_bias,
                output_attentions=output_attentions,
                deterministic=deterministic,
            )
            # 获取经过交叉注意力处理后的隐藏状态
            hidden_states = cross_attention_outputs[0]

            # 保留交叉注意力输出和相对位置权重
            attention_outputs = attention_outputs + cross_attention_outputs[1:]

        # 应用前馈网络层处理隐藏状态
        hidden_states = self.layer[-1](hidden_states, deterministic=deterministic)

        # 构建输出元组
        outputs = (hidden_states,)

        # 将注意力输出连接到输出元组中
        outputs = outputs + attention_outputs

        # 返回隐藏状态、当前键值状态、(self-attention 位置偏置)、(self-attention 权重)、
        # (cross-attention 位置偏置)、(cross-attention 权重)
        return outputs
# 从transformers.models.t5.modeling_flax_t5.FlaxT5LayerCollection复制过来，将T5改为LongT5
class FlaxLongT5LayerCollection(nn.Module):
    # 配置项: LongT5Config
    config: LongT5Config
    # 是否具有相对注意力偏置
    has_relative_attention_bias: bool
    # 计算时的数据类型
    dtype: jnp.dtype = jnp.float32  

    def setup(self):
        # 创建一个FlaxLongT5Block实例
        self.layer = FlaxLongT5Block(
            self.config, has_relative_attention_bias=self.has_relative_attention_bias, dtype=self.dtype
        )

    def __call__(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        encoder_decoder_position_bias=None,
        output_attentions=False,
        deterministic=True,
        init_cache=False,
    ):
        return self.layer(
            hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            encoder_decoder_position_bias=encoder_decoder_position_bias,
            output_attentions=output_attentions,
            deterministic=deterministic,
            init_cache=init_cache,
        )

# 从transformers.models.t5.modeling_flax_t5.FlaxT5BlockCollection复制过来，将T5改为LongT5
class FlaxLongT5BlockCollection(nn.Module):
    # 配置项: LongT5Config
    config: LongT5Config
    # 计算时的数据类型
    dtype: jnp.dtype = jnp.float32 
    # 是否开启梯度检查点
    gradient_checkpointing: bool = False

    def setup(self):
        # 是否是有因果关系
        self.causal = self.config.causal
        if self.gradient_checkpointing:
            # 使用remat生成FlaxLongT5CheckpointLayer
            FlaxLongT5CheckpointLayer = remat(FlaxLongT5LayerCollection, static_argnums=(6, 7, 8))
            self.blocks = [
                FlaxLongT5CheckpointLayer(
                    self.config,
                    has_relative_attention_bias=(i == 0),
                    dtype=self.dtype,
                    name=str(i),
                )
                for i in range(self.config.num_layers)
            ]
        else:
            self.blocks = [
                FlaxLongT5LayerCollection(
                    self.config,
                    has_relative_attention_bias=(i == 0),
                    dtype=self.dtype,
                    name=str(i),
                )
                for i in range(self.config.num_layers)
            ]

    def __call__(
        self,
        hidden_states=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        deterministic: bool = True,
        init_cache: bool = False,

```py      
        # 如果需要，准备头部掩码
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and self.causal) else None
        position_bias = None
        encoder_decoder_position_bias = None

        # 遍历每个模块
        for i, layer_module in enumerate(self.blocks):
            # 如果需要输出隐藏状态，则添加到所有隐藏状态中
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 调用层模块进行前向传播
            layer_outputs = layer_module(
                hidden_states,
                attention_mask,
                position_bias,
                encoder_hidden_states,
                encoder_attention_mask,
                encoder_decoder_position_bias,
                output_attentions,
                deterministic,
                init_cache,
            )

            # 更新隐藏状态为当前层模块的输出
            hidden_states = layer_outputs[0]

            # 共享位置偏差在各层之间 - 第一层存储它们
            # layer_outputs = hidden-states, key-value-states (self-attention position bias), (self-attention weights),
            # (cross-attention position bias), (cross-attention weights)
            position_bias = layer_outputs[1]

            # 如果是因果的，且有编码器隐藏状态，则更新编码器-解码器位置偏差
            if self.causal and encoder_hidden_states is not None:
                encoder_decoder_position_bias = layer_outputs[3 if output_attentions else 2]

            # 如果需要输出注意力权重，则添加到所有注意力权重中
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[2],)
                if self.causal:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[4],)

        # 返回最终的模型输出，包括过去和交叉注意力
        return FlaxBaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions,
        )
# Copied from transformers.models.t5.modeling_flax_t5.FlaxT5Stack with T5->LongT5
class FlaxLongT5Stack(nn.Module):
    # 定义了一个 nn.Module 的子类 FlaxLongT5Stack，继承了 nn.Module 的所有属性和方法，并在使用 T5 模型过程中将其替换为 LongT5。
    config: LongT5Config  # config 是一个 LongT5Config 类型的变量
    embed_tokens: nn.Embed  # embed_tokens 是 nn.Embed 类型的变量
    dtype: jnp.dtype = jnp.float32  # 设置 dtype 的默认值为 jnp.float32，用于计算过程中的数据类型
    gradient_checkpointing: bool = False  # 设置 gradient_checkpointing 的默认值为 False，用于指定是否使用梯度检查点技术

    def setup(self):
        # 初始化方法
        self.causal = self.config.causal  # 从 config 中获取 causal 属性的值

        self.block = FlaxLongT5BlockCollection(
            # 创建一个 FlaxLongT5BlockCollection 类型的对象，并传入 config、dtype 和 gradient_checkpointing 等参数
            self.config, dtype=self.dtype, gradient_checkpointing=self.gradient_checkpointing
        )
        self.final_layer_norm = FlaxLongT5LayerNorm(
            # 创建一个 FlaxLongT5LayerNorm 类型的对象，并传入 config.d_model、eps 和 dtype 等参数
            self.config.d_model, eps=self.config.layer_norm_epsilon, dtype=self.dtype
        )
        self.dropout = nn.Dropout(self.config.dropout_rate)  # 创建一个 Dropout 类型的对象，并传入 config.dropout_rate 参数

    def __call__(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        deterministic: bool = True,
        init_cache: bool = False,
    ):
        # 对象的可调用方法，用于进行前向传播计算
        hidden_states = self.embed_tokens(input_ids)  # 将 input_ids 作为参数传入 embed_tokens 方法进行嵌入表示计算
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)  # 对计算结果进行 dropout 操作

        outputs = self.block(
            hidden_states,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            deterministic=deterministic,
            init_cache=init_cache,
        )
        # 调用 block 对象进行计算，传入隐藏状态、注意力掩码、编码器隐藏状态、编码器注意力掩码和其他相关参数

        hidden_states = outputs[0]  # 从 outputs 中获取第一个元素作为隐藏状态

        hidden_states = self.final_layer_norm(hidden_states)  # 对隐藏状态进行最后一层的 LayerNorm 处理
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)  # 对处理结果进行 dropout 操作

        # Add last layer
        all_hidden_states = None  # 初始化一个 all_hidden_states 变量，用于存储所有的隐藏状态

        if output_hidden_states:
            all_hidden_states = outputs.hidden_states  # 如果输出所有的隐藏状态，则将 outputs 中的 hidden_states 保存到 all_hidden_states 中
            all_hidden_states = all_hidden_states + (hidden_states,)  # 将最后一层的隐藏状态添加到 all_hidden_states 中

        if not return_dict:
            if output_hidden_states:
                return (
                    hidden_states,
                    all_hidden_states,
                ) + outputs[2:]  # 如果不返回字典类型的结果，且输出所有的隐藏状态，则返回 hidden_states, all_hidden_states 和 outputs[2:] 的结果
            return (hidden_states,) + outputs[1:]  # 如果只输出 hidden_states，则返回 hidden_states 和 outputs[1:] 的结果

        return FlaxBaseModelOutputWithPastAndCrossAttentions(
            # 封装结果为 FlaxBaseModelOutputWithPastAndCrossAttentions 类型的对象并返回
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )


LONGT5_ENCODE_INPUTS_DOCSTRING = r"""
"""  # 设置 LONGT5_ENCODE_INPUTS_DOCSTRING 为一个空字符串

# 注释：
    Args:
        input_ids (`jnp.ndarray` of shape `(batch_size, sequence_length)`):
            # 输入序列标记在词汇表中的索引。LongT5 是一个具有相对位置嵌入的模型，因此您应该能够在右侧和左侧都填充输入。
            # 可以使用 `AutoTokenizer` 获取索引。参见 `PreTrainedTokenizer.encode` 和 `PreTrainedTokenizer.__call__` 了解详细信息。
            # 想要了解如何准备 `input_ids` 进行预训练，请查看 [LONGT5 Training](./longt5#training)。
        attention_mask (`jnp.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
            # 避免对填充标记索引执行注意力的掩码。掩码值选在 `[0, 1]`：

            # - 对于 **未掩码** 的标记，值为 1，
            # - 对于 **掩码** 的标记，值为 0。

            # [什么是注意力掩码？](../glossary#attention-mask)
        output_attentions (`bool`, *optional*):
            # 是否返回所有注意力层的注意力张量。有关更多详细信息，请查看返回的张量中的 `attentions`。
        output_hidden_states (`bool`, *optional*):
            # 是否返回所有层的隐藏状态。有关更多详细信息，请查看返回的张量中的 `hidden_states`。
        return_dict (`bool`, *optional*):
            # 是否返回 [`~utils.ModelOutput`] 而不是普通元组。
"""
FlaxLongT5PreTrainedModel 是 `FlaxPreTrainedModel` 的子类，用来处理权重初始化、预训练模型的下载和加载等。
"""

# 导入所需依赖

# 导入 `FlaxPreTrainedModel` 类
from transformers import FlaxPreTrainedModel
# 导入 `LongT5Config` 类
from transformers.models.longformer.modeling_flax_longformer import LongT5Config
# 导入 `nn.Module` 类
from flax import nn


LONGT5_DECODE_INPUTS_DOCSTRING = r"""
    Args:
        decoder_input_ids (`jnp.ndarray` of shape `(batch_size, target_sequence_length)`):
            Indices of decoder input sequence tokens in the vocabulary.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are decoder input IDs?](../glossary#decoder-input-ids)

            For training, `decoder_input_ids` should be provided.
        encoder_outputs (`tuple(tuple(jnp.ndarray)`):
            Tuple consists of (`last_hidden_state`, *optional*: `hidden_states`, *optional*: `attentions`)
            `last_hidden_state` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) is a sequence of
            hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.
        encoder_attention_mask (`jnp.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        decoder_attention_mask (`jnp.ndarray` of shape `(batch_size, target_sequence_length)`, *optional*):
            Default behavior: generate a tensor that ignores pad tokens in `decoder_input_ids`. Causal mask will also
            be used by default.

            If you want to change padding behavior, you should modify to your needs. See diagram 1 in [the
            paper](https://arxiv.org/abs/1910.13461) for more information on the default strategy.
        past_key_values (`Dict[str, np.ndarray]`, *optional*, returned by `init_cache` or when passing previous `past_key_values`):
            Dictionary of pre-computed hidden-states (key and values in the attention blocks) that can be used for fast
            auto-regressive decoding. Pre-computed key and value hidden-states are of shape *[batch_size, max_length]*.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""

LONGT5_INPUTS_DOCSTRING = r"""
"""


class FlaxLongT5PreTrainedModel(FlaxPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 类属性：指定配置类为 `LongT5Config`
    config_class = LongT5Config
    # 类属性：指定模型前缀为 "transformer"
    base_model_prefix = "transformer"
    # 类属性：指定模块类为 `nn.Module`
    module_class: nn.Module = None
    # 初始化 LongT5Model 类
    def __init__(
        self,
        config: LongT5Config, # 模型配置
        input_shape: Tuple[int] = (1, 1), # 输入形状
        seed: int = 0, # 随机种子
        dtype: jnp.dtype = jnp.float32, # 数据类型
        _do_init: bool = True, # 是否执行初始化
        **kwargs,
    ):
        # 根据配置创建 module 对象
        module = self.module_class(config=config, dtype=dtype, **kwargs)
        # 调用父类的初始化方法
        super().__init__(config, module, input_shape=input_shape, seed=seed, dtype=dtype, _do_init=_do_init)
    
    # 启用梯度检查点
    def enable_gradient_checkpointing(self):
        # 根据配置创建新的 module 对象，并设置 gradient_checkpointing 为 True
        self._module = self.module_class(
            config=self.config,
            dtype=self.dtype,
            gradient_checkpointing=True,
        )
    
    # 初始化模型参数
    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict = None) -> FrozenDict:
        # 创建输入 Tensor
        input_ids = jnp.zeros(input_shape, dtype="i4")
        attention_mask = jnp.ones_like(input_ids)
        decoder_input_ids = jnp.ones_like(input_ids)
        decoder_attention_mask = jnp.ones_like(input_ids)
    
        # 分离 RNG 用于参数和 dropout
        params_rng, dropout_rng = jax.random.split(rng)
        rngs = {"params": params_rng, "dropout": dropout_rng}
    
        # 使用输入 Tensor 初始化模型参数
        random_params = self.module.init(
            rngs,
            input_ids,
            attention_mask,
            decoder_input_ids,
            decoder_attention_mask,
        )["params"]
    
        # 如果提供了现有参数，则合并缺失的参数
        if params is not None:
            random_params = flatten_dict(unfreeze(random_params))
            params = flatten_dict(unfreeze(params))
            for missing_key in self._missing_keys:
                params[missing_key] = random_params[missing_key]
            self._missing_keys = set()
            return freeze(unflatten_dict(params))
        else:
            return random_params
    
    # 执行模型前向传播
    @add_start_docstrings_to_model_forward(LONGT5_INPUTS_DOCSTRING)
    def __call__(
        self,
        input_ids: jnp.ndarray, # 输入 ID 序列
        attention_mask: Optional[jnp.ndarray] = None, # 输入注意力掩码
        decoder_input_ids: jnp.ndarray = None, # 解码器输入 ID 序列
        decoder_attention_mask: Optional[jnp.ndarray] = None, # 解码器注意力掩码
        output_attentions: Optional[bool] = None, # 是否输出注意力权重
        output_hidden_states: Optional[bool] = None, # 是否输出隐藏状态
        return_dict: Optional[bool] = None, # 是否返回字典
        train: bool = False, # 是否为训练模式
        params: dict = None, # 自定义模型参数
        dropout_rng: PRNGKey = None, # 自定义 dropout RNG
    ):
        # 检查是否需要输出注意力权重信息，若未指定则使用配置中的默认值
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 检查是否需要输出隐藏状态信息，若未指定则使用配置中的默认值
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 检查是否需要返回字典形式的输出，若未指定则使用配置中的默认值
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        # 如果未提供解码器的输入ID，则抛出错误
        if decoder_input_ids is None:
            raise ValueError(
                "Make sure to provide both `input_ids` and `decoder_input_ids`. `decoder_input_ids` is not passed"
                " here."
            )

        # 准备编码器输入的注意力掩码，若未提供则使用全 1 的掩码
        if attention_mask is None:
            attention_mask = jnp.ones_like(input_ids)

        # 准备解码器输入的注意力掩码，若未提供则使用全 1 的掩码
        if decoder_attention_mask is None:
            decoder_attention_mask = jnp.ones_like(decoder_input_ids)

        # 处理需要的伪随机数生成器（PRNG）
        rngs = {"dropout": dropout_rng} if dropout_rng is not None else {}

        # 调用模块的apply方法执行Transformer的前向传播
        return self.module.apply(
            {"params": params or self.params},  # 参数字典
            input_ids=jnp.array(input_ids, dtype="i4"),  # 编码器输入的ID
            attention_mask=jnp.array(attention_mask, dtype="i4"),  # 编码器的注意力掩码
            decoder_input_ids=jnp.array(decoder_input_ids, dtype="i4"),  # 解码器输入的ID
            decoder_attention_mask=jnp.array(decoder_attention_mask, dtype="i4"),  # 解码器的注意力掩码
            output_attentions=output_attentions,  # 是否输出注意力权重信息
            output_hidden_states=output_hidden_states,  # 是否输出隐藏状态信息
            return_dict=return_dict,  # 是否以字典形式返回输出
            deterministic=not train,  # 是否使用确定性前向传播（非训练模式）
            rngs=rngs,  # 伪随机数生成器
        )
    # 初始化缓存函数，用于快速自回归解码
    def init_cache(self, batch_size, max_length, encoder_outputs):
        r"""
        Args:
            batch_size (`int`):
                用于快速自回归解码的 batch_size。定义初始化缓存的批处理大小。
            max_length (`int`):
                自回归解码的最大可能长度。定义初始化缓存的序列长度。
            encoder_outputs (`Union[FlaxBaseModelOutput, tuple(tuple(jnp.ndarray)]`):
                `encoder_outputs` 包括 (`last_hidden_state`, *可选*: `hidden_states`, *可选*: `attentions`)。
                `last_hidden_state` 的形状为 `(batch_size, sequence_length, hidden_size)`，*可选*）是编码器最后一层的隐藏状态序列。
                用于解码器的交叉注意力。

        """
        # 初始化用于检索缓存的输入变量
        decoder_input_ids = jnp.ones((batch_size, max_length), dtype="i4")
        decoder_attention_mask = jnp.ones_like(decoder_input_ids)

        # 定义内部函数，用于调用解码器以初始化缓存
        def _decoder_forward(module, decoder_input_ids, decoder_attention_mask, **kwargs):
            decoder_module = module._get_decoder_module()
            return decoder_module(
                decoder_input_ids,
                decoder_attention_mask,
                **kwargs,
            )

        # 使用模型的初始化方法初始化变量，包括解码器输入、注意力掩码、编码器隐藏状态等，并调用解码器初始化缓存
        init_variables = self.module.init(
            jax.random.PRNGKey(0),
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            init_cache=True,
            method=_decoder_forward,  # 只需调用解码器来初始化缓存
        )
        # 返回解冻的缓存变量
        return unfreeze(init_variables["cache"])

    # 编码函数，对输入进行编码
    @add_start_docstrings(LONGT5_ENCODE_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=FlaxBaseModelOutput, config_class=LongT5Config)
    def encode(
        self,
        input_ids: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        train: bool = False,
        params: dict = None,
        dropout_rng: PRNGKey = None,
    # This block of code defines a function or method that returns some outputs
        ):
            r"""
            # This string defines the returns and provides an example usage with a code snippet
            Returns:
    
            Example:
    
            ```python
            >>> from transformers import AutoTokenizer, FlaxLongT5ForConditionalGeneration
            # Importing AutoTokenizer to tokenize text, and FlaxLongT5ForConditionalGeneration to generate text conditionally
    
            >>> tokenizer = AutoTokenizer.from_pretrained("t5-base")
            # Create a tokenizer using a pre-trained model "t5-base"
    
            >>> model = FlaxLongT5ForConditionalGeneration.from_pretrained("google/long-t5-local-base")
            # Initialize the model using a specific pre-trained model variant "google/long-t5-local-base"
    
            >>> text = "My friends are cool but they eat too many carbs."
            # Define a text to be tokenized and passed through the model
    
            >>> inputs = tokenizer(text, return_tensors="np")
            # Tokenize the text and convert it to a format compatible with Flax models
    
            >>> encoder_outputs = model.encode(**inputs)
            # Encode the tokenized text with the model to obtain encoder outputs
            ```py"""
            # Check if output_attentions is set, otherwise use the default from the model's config
            output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
            # Check if output_hidden_states is set, otherwise use the default from the model's config
            output_hidden_states = (
                output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
            )
            # Check if return_dict is set, otherwise use the default from the model's config
            return_dict = return_dict if return_dict is not None else self.config.return_dict
    
            # If attention_mask is not provided, create a mask with all ones of the same shape as input_ids
            if attention_mask is None:
                attention_mask = jnp.ones_like(input_ids)
    
            # Create a dictionary for random number generation, initializing with "dropout" key if dropout_rng is provided
            rngs = {}
            if dropout_rng is not None:
                rngs["dropout"] = dropout_rng
    
            # Define a nested function to forward-encode input data using a specific module
            def _encoder_forward(module, input_ids, attention_mask, **kwargs):
                # Retrieve the encoder module from the provided module
                encode_module = module._get_encoder_module()
                # Apply the encoder module to the input data with given attention mask and additional arguments
                return encode_module(input_ids, attention_mask, **kwargs)
    
            # Apply the main module with specified parameters and inputs
            return self.module.apply(
                {"params": params or self.params},  # Use provided params or default to self.params
                input_ids=jnp.array(input_ids, dtype="i4"),  # Convert input_ids to a JAX array with int4 type
                attention_mask=jnp.array(attention_mask, dtype="i4"),  # Convert attention_mask similarly
                output_attentions=output_attentions,  # Specify whether to return attention outputs
                output_hidden_states=output_hidden_states,  # Specify whether to return hidden states
                return_dict=return_dict,  # Specify whether to return a dictionary of outputs
                deterministic=not train,  # If not in training mode, ensure deterministic behavior
                rngs=rngs,  # Use the created random number generators
                method=_encoder_forward,  # Define the method for applying encoding
            )
    
        # Decorate the next function with a custom docstring and set specific return types
        @add_start_docstrings(LONGT5_DECODE_INPUTS_DOCSTRING)
        @replace_return_docstrings(output_type=FlaxBaseModelOutputWithPastAndCrossAttentions, config_class=LongT5Config)
        # Define the decode function for a Transformer-based model with various parameters
        def decode(
            self,
            decoder_input_ids,  # The IDs for the decoder inputs
            encoder_outputs,  # The outputs from the encoder part of the model
            encoder_attention_mask: Optional[jnp.ndarray] = None,  # Optional attention mask for the encoder
            decoder_attention_mask: Optional[jnp.ndarray] = None,  # Optional attention mask for the decoder
            past_key_values: dict = None,  # Dictionary of past key values for caching
            output_attentions: Optional[bool] = None,  # Optional flag for returning attention outputs
            output_hidden_states: Optional[bool] = None,  # Optional flag for returning hidden states
            return_dict: Optional[bool] = None,  # Optional flag for returning a dictionary of outputs
            train: bool = False,  # Boolean flag indicating if the model is in training mode
            params: dict = None,  # Optional dictionary of parameters to use for this call
            dropout_rng: PRNGKey = None,  # Optional random number generator key for dropout operations
LONGT5_START_DOCSTRING = r"""
    The LongT5 model was proposed in [LongT5: Efficient Text-To-Text Transformer for Long
    Sequences](https://arxiv.org/abs/2112.07916) by Mandy Guo, Joshua Ainslie, David Uthus, Santiago Ontanon, Jianmo
    Ni, Yun-Hsuan Sung and Yinfei Yang. It's an encoder-decoder transformer pre-trained in a text-to-text denoising
    generative setting. LongT5 model is an extension of T5 model, and it enables using one of the two different
    efficient attention mechanisms - (1) Local attention, or (2) Transient-Global attention.

    This model inherits from [`FlaxPreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a Flax Linen
    [flax.nn.Module](https://flax.readthedocs.io/en/latest/_autosummary/flax.nn.module.html) subclass. Use it as a
    regular Flax Module and refer to the Flax documentation for all matter related to general usage and behavior.

    Finally, this model supports inherent JAX features such as:

    - [Just-In-Time (JIT) compilation](https://jax.readthedocs.io/en/latest/jax.html#just-in-time-compilation-jit)
    - [Automatic Differentiation](https://jax.readthedocs.io/en/latest/jax.html#automatic-differentiation)
    - [Vectorization](https://jax.readthedocs.io/en/latest/jax.html#vectorization-vmap)
    - [Parallelization](https://jax.readthedocs.io/en/latest/jax.html#parallelization-pmap)

    Parameters:
        config ([`LongT5Config`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~FlaxPreTrainedModel.from_pretrained`] method to load the model weights.
        dtype (`jax.numpy.dtype`, *optional*, defaults to `jax.numpy.float32`):
            The data type of the computation. Can be one of `jax.numpy.float32`, `jax.numpy.float16` (on GPUs) and
            `jax.numpy.bfloat16` (on TPUs).

            This can be used to enable mixed-precision training or half-precision inference on GPUs or TPUs. If
            specified all the computation will be performed with the given `dtype`.

            **Note that this only specifies the dtype of the computation and does not influence the dtype of model
            parameters.**

            If you wish to change the dtype of the model parameters, see [`~FlaxPreTrainedModel.to_fp16`] and
            [`~FlaxPreTrainedModel.to_bf16`].
"""


@add_start_docstrings(
    "The bare LONGT5 Model transformer outputting raw hidden-stateswithout any specific head on top.",
    LONGT5_START_DOCSTRING,
)
# Copied from transformers.models.t5.modeling_flax_t5.FlaxT5Module with T5->LongT5
class FlaxLongT5Module(nn.Module):
    # 构造函数，初始化模型配置和数据类型
    config: LongT5Config
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation
    # 设置默认值为False的梯度检查点
    gradient_checkpointing: bool = False

    # 返回编码器模块
    def _get_encoder_module(self):
        return self.encoder

    # 返回解码器模块
    def _get_decoder_module(self):
        return self.decoder

    # 设置模型，包括共享的嵌入层、编码器和解码器
    def setup(self):
        # 创建共享的嵌入层
        self.shared = nn.Embed(
            self.config.vocab_size,
            self.config.d_model,
            embedding_init=jax.nn.initializers.normal(self.config.initializer_factor * 1.0),
            dtype=self.dtype,
        )

        # 复制编码器配置，设置 causal 为 False，创建编码器
        encoder_config = copy.deepcopy(self.config)
        encoder_config.causal = False
        self.encoder = FlaxLongT5Stack(
            encoder_config,
            embed_tokens=self.shared,
            dtype=self.dtype,
            gradient_checkpointing=self.gradient_checkpointing,
        )

        # 复制解码器配置，设置 causal 为 True 和层数为 self.config.num_decoder_layers，创建解码器
        decoder_config = copy.deepcopy(self.config)
        decoder_config.causal = True
        decoder_config.num_layers = self.config.num_decoder_layers
        self.decoder = FlaxLongT5Stack(
            decoder_config,
            embed_tokens=self.shared,
            dtype=self.dtype,
            gradient_checkpointing=self.gradient_checkpointing,
        )

    # 模型的调用方法，包括编码和解码
    def __call__(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        encoder_outputs=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        deterministic: bool = True,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 如果需要编码（训练或第一次预测），则进行编码
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            deterministic=deterministic,
        )

        # 解码
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            deterministic=deterministic,
        )

        # 如果���需要返回字典，则返回解码器和编码器的输出
        if not return_dict:
            return decoder_outputs + encoder_outputs

        # 返回自定义的输出，包括解码器和编码器的输出信息
        return FlaxSeq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )
# 从transformers.models.t5.modeling_flax_t5.FlaxT5Model 复制代码，并将T5->LongT5
class FlaxLongT5Model(FlaxLongT5PreTrainedModel):
    module_class = FlaxLongT5Module

# 添加示例代码的文档字符串
append_call_sample_docstring(FlaxLongT5Model, _CHECKPOINT_FOR_DOC, FlaxSeq2SeqModelOutput, _CONFIG_FOR_DOC)

# FlaxLongT5Model的文档字符串
FLAX_LONGT5_MODEL_DOCSTRING = """
    Returns:

    Example:

    ```python
    >>> from transformers import AutoTokenizer, FlaxLongT5Model

    >>> tokenizer = AutoTokenizer.from_pretrained("t5-base")
    >>> model = FlaxLongT5Model.from_pretrained("google/long-t5-local-base")

    >>> input_ids = tokenizer(
    ...     "Studies have been shown that owning a dog is good for you", return_tensors="np"
    ... ).input_ids
    >>> decoder_input_ids = tokenizer("Studies show that", return_tensors="np").input_ids

    >>> # forward pass
    >>> outputs = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)
    >>> last_hidden_states = outputs.last_hidden_state
    ```py
"""

# 更新FlaxLongT5Model的文档字符串
overwrite_call_docstring(FlaxLongT5Model, LONGT5_INPUTS_DOCSTRING + FLAX_LONGT5_MODEL_DOCSTRING)
append_replace_return_docstrings(FlaxLongT5Model, output_type=FlaxSeq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)

# LONGT5 Model与具有`language modeling`头部的FlaxLongT5ForConditionalGenerationModule
# 从transformers.models.t5.modeling_flax_t5.FlaxT5ForConditionalGenerationModule复制代码，并将T5->LongT5
class FlaxLongT5ForConditionalGenerationModule(nn.Module):
    config: LongT5Config
    dtype: jnp.dtype = jnp.float32  # 计算的数据类型
    gradient_checkpointing: bool = False

    def _get_encoder_module(self):
        return self.encoder

    def _get_decoder_module(self):
        return self.decoder

    def setup(self):
        self.model_dim = self.config.d_model

        # 共享的嵌入层
        self.shared = nn.Embed(
            self.config.vocab_size,
            self.config.d_model,
            embedding_init=jax.nn.initializers.normal(self.config.initializer_factor),
            dtype=self.dtype,
        )

        # 编码器模块
        encoder_config = copy.deepcopy(self.config)
        encoder_config.causal = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = FlaxLongT5Stack(
            encoder_config, self.shared, dtype=self.dtype, gradient_checkpointing=self.gradient_checkpointing
        )

        # 解码器模块
        decoder_config = copy.deepcopy(self.config)
        decoder_config.causal = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = self.config.num_decoder_layers
        self.decoder = FlaxLongT5Stack(
            decoder_config, self.shared, dtype=self.dtype, gradient_checkpointing=self.gradient_checkpointing
        )

        # 语言模型头部
        self.lm_head = nn.Dense(
            self.config.vocab_size,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_factor),
            dtype=self.dtype,
        )
    # 定义一个方法，接受多个参数作为输入
    def __call__(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        encoder_outputs=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        deterministic: bool = True,
    ):
        # 如果 return_dict 为 None，则使用配置文件中的 use_return_dict 值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用编码器，传入参数并返回编码器输出
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            deterministic=deterministic,
        )

        # 从编码器输出中获取隐藏状态
        hidden_states = encoder_outputs[0]

        # 调用解码器，传入参数并返回解码器输出
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            deterministic=deterministic,
        )

        # 从解码器输出中获取序列输出
        sequence_output = decoder_outputs[0]

        # 如果配置要求共享词嵌入，则对序列输出进行重新缩放
        if self.config.tie_word_embeddings:
            # 重新缩放输出以便在词汇表上进行投影
            # 详见 https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim**-0.5)

        # 如果配置要求共享词嵌入，则使用共享的嵌入层进行计算 lm_logits
        if self.config.tie_word_embeddings:
            shared_embedding = self.shared.variables["params"]["embedding"]
            lm_logits = self.lm_head.apply({"params": {"kernel": shared_embedding.T}}, sequence_output)
        else:
            # 否则直接使用 lm_head 计算 lm_logits
            lm_logits = self.lm_head(sequence_output)

        # 如果 return_dict 为 False，则返回 lm_logits 以及其他输出
        if not return_dict:
            return (lm_logits,) + decoder_outputs[1:] + encoder_outputs
        
        # 否则返回 FlaxSeq2SeqLMOutput 对象
        return FlaxSeq2SeqLMOutput(
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )
# FlaxLongT5ForConditionalGeneration类是FlaxLongT5PreTrainedModel的子类
class FlaxLongT5ForConditionalGeneration(FlaxLongT5PreTrainedModel):
    # 模型类的类属性
    module_class = FlaxLongT5ForConditionalGenerationModule

    # decode方法，用于解码生成文本
    # 输入参数包括decoder_input_ids, encoder_outputs, 等等
    # 返回参数为FlaxCausalLMOutputWithCrossAttentions类型，根据LongT5Config类的配置
    @add_start_docstrings(LONGT5_DECODE_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=FlaxCausalLMOutputWithCrossAttentions, config_class=LongT5Config)
    def decode(
        self,
        decoder_input_ids,
        encoder_outputs,
        encoder_attention_mask: Optional[jnp.ndarray] = None,
        decoder_attention_mask: Optional[jnp.ndarray] = None,
        past_key_values: dict = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        train: bool = False,
        params: dict = None,
        dropout_rng: PRNGKey = None,
    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        max_length,
        attention_mask: Optional[jax.Array] = None,
        decoder_attention_mask: Optional[jax.Array] = None,
        encoder_outputs=None,
        **kwargs,
    ):
        # 初始化缓存
        batch_size, seq_length = decoder_input_ids.shape
        past_key_values = self.init_cache(batch_size, max_length, encoder_outputs)
        # 注意通常需要将 attention_mask 中 x > input_ids.shape[-1] 和 x < cache_length 的位置设置为0
        # 但由于解码器使用的是因果掩码，这些位置已经被掩盖了
        # 因此我们可以在这里创建一个静态的 attention_mask，这样更高效
        extended_attention_mask = jnp.ones((batch_size, max_length), dtype="i4")
        if decoder_attention_mask is not None:
            extended_attention_mask = jax.lax.dynamic_update_slice(
                extended_attention_mask, decoder_attention_mask, (0, 0)
            )
        # 返回输入参数的字典
        return {
            "past_key_values": past_key_values,
            "encoder_outputs": encoder_outputs,
            "encoder_attention_mask": attention_mask,
            "decoder_attention_mask": extended_attention_mask,
        }

    # 更新生成过程中的输入参数
    def update_inputs_for_generation(self, model_outputs, model_kwargs):
        model_kwargs["past_key_values"] = model_outputs.past_key_values
        return model_kwargs

# 额外的FlaxLongT5ForConditionalGeneration文档字符串
FLAX_LONGT5_CONDITIONAL_GENERATION_DOCSTRING = """
    Returns:

    Example:

    ```python
    >>> from transformers import AutoTokenizer, FlaxLongT5ForConditionalGeneration

    >>> tokenizer = AutoTokenizer.from_pretrained("t5-base")
    >>> model = FlaxLongT5ForConditionalGeneration.from_pretrained("google/long-t5-local-base")

    >>> ARTICLE_TO_SUMMARIZE = "summarize: My friends are cool but they eat too many carbs."
    >>> inputs = tokenizer([ARTICLE_TO_SUMMARIZE], return_tensors="np")

    >>> # Generate Summary
    >>> summary_ids = model.generate(inputs["input_ids"]).sequences
    >>> print(tokenizer.decode(summary_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False))
    ```py
"""

# 调用overwrite_call_docstring函数 with 添加的文档字符串作为参数
overwrite_call_docstring(
    # 导入FlaxLongT5ForConditionalGeneration模块，并拼接输入和条件生成的文档字符串
    FlaxLongT5ForConditionalGeneration, LONGT5_INPUTS_DOCSTRING + FLAX_LONGT5_CONDITIONAL_GENERATION_DOCSTRING
# 调用 append_replace_return_docstrings 函数，将 FlaxLongT5ForConditionalGeneration 类的文档字符串追加或替换为指定格式的字符串
(
append_replace_return_docstrings(
    FlaxLongT5ForConditionalGeneration, output_type=FlaxSeq2SeqLMOutput, config_class=_CONFIG_FOR_DOC
)
# 函数调用的参数：
# - FlaxLongT5ForConditionalGeneration: 要操作的类对象，为 FlaxLongT5ForConditionalGeneration
# - output_type: 输出类型，为 FlaxSeq2SeqLMOutput 类型
# - config_class: 配置类，为 _CONFIG_FOR_DOC 对象
)
```