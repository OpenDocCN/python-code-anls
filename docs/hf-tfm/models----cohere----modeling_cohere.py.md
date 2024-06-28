# `.\models\cohere\modeling_cohere.py`

```
# 定义 CohereLayerNorm 类，用于实现 Cohere 模型中的 LayerNorm 层
class CohereLayerNorm(nn.Module):
    # 初始化函数
    def __init__(self, hidden_size, eps=1e-5, bias=False):
        super().__init__()
        # 定义可学习参数 weight，形状为 hidden_size，初始化为全1张量
        self.weight = nn.Parameter(torch.ones(hidden_size))
        # 如果 bias=True，则定义可学习参数 bias，形状同样为 hidden_size，初始化为全0张量；否则为 None
        self.bias = nn.Parameter(torch.zeros(hidden_size)) if bias else None
        # 设定 LayerNorm 中的 epsilon 值
        self.variance_epsilon = eps
    # 定义前向传播方法，接收隐藏状态作为输入参数
    def forward(self, hidden_states):
        # 获取输入张量的数据类型
        input_dtype = hidden_states.dtype
        # 将隐藏状态张量转换为 float32 类型
        hidden_states = hidden_states.to(torch.float32)
        # 计算隐藏状态在最后一个维度上的均值
        mean = hidden_states.mean(-1, keepdim=True)
        # 计算隐藏状态在最后一个维度上的方差
        variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
        # 根据均值和方差对隐藏状态进行归一化
        hidden_states = (hidden_states - mean) * torch.rsqrt(variance + self.variance_epsilon)
        # 使用权重对归一化后的隐藏状态进行加权
        hidden_states = self.weight.to(torch.float32) * hidden_states
        # 如果存在偏置项，将偏置项加到隐藏状态上
        if self.bias is not None:
            hidden_states = hidden_states + self.bias.to(torch.float32)
        # 将处理后的隐藏状态张量转回初始输入数据类型
        return hidden_states.to(input_dtype)
# 将 CohereLayerNorm 添加到 ALL_LAYERNORM_LAYERS 列表中
ALL_LAYERNORM_LAYERS.append(CohereLayerNorm)

class CohereRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        super().__init__()
        self.scaling_factor = scaling_factor
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        # 计算旋转位置嵌入的频率逆向
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))
        # 将 inv_freq 注册为不可训练的缓冲区
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids):
        # x: [bs, num_attention_heads, seq_len, head_size]
        # 扩展 inv_freq 到与 position_ids 相同的形状
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()

        # 强制将频率计算结果转为 float32，因为 bfloat16 在长上下文中会丢失精度
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            # 计算旋转角度的余弦和正弦值
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.repeat_interleave(freqs, 2, dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x):
    # 将输入张量 x 拆分和旋转
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    rot_x = torch.stack([-x2, x1], dim=-1).flatten(-2)
    return rot_x


# 从 transformers.models.llama.modeling_llama.apply_rotary_pos_emb 复制过来的函数
def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    # 将 cos 张量在指定维度上扩展维度
    cos = cos.unsqueeze(unsqueeze_dim)
    # 将 sin 张量在指定维度上扩展维度
    sin = sin.unsqueeze(unsqueeze_dim)
    # 计算查询向量的旋转位置嵌入，使用 cos 和 sin 对应元素加权
    q_embed = (q * cos) + (rotate_half(q) * sin)
    # 计算键向量的旋转位置嵌入，使用 cos 和 sin 对应元素加权
    k_embed = (k * cos) + (rotate_half(k) * sin)
    # 返回旋转后的查询向量和键向量作为元组
    return q_embed, k_embed
# 从 transformers.models.llama.modeling_llama.LlamaMLP Llama->Cohere 复制过来的类，用于定义 CohereMLP 模型
class CohereMLP(nn.Module):
    # 初始化函数，接受一个配置对象 config 作为参数
    def __init__(self, config):
        super().__init__()
        # 将配置对象保存到实例中
        self.config = config
        # 设置隐藏层大小为配置中的 hidden_size
        self.hidden_size = config.hidden_size
        # 设置中间层大小为配置中的 intermediate_size
        self.intermediate_size = config.intermediate_size
        # 创建一个线性层，用于门控投影，输入维度为 hidden_size，输出维度为 intermediate_size，没有偏置项
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        # 创建一个线性层，用于上投影，输入维度为 hidden_size，输出维度为 intermediate_size，没有偏置项
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        # 创建一个线性层，用于下投影，输入维度为 intermediate_size，输出维度为 hidden_size，没有偏置项
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        # 根据配置中的激活函数类型选择对应的激活函数，保存到实例中
        self.act_fn = ACT2FN[config.hidden_act]

    # 前向传播函数，接受输入张量 x
    def forward(self, x):
        # 计算门控投影后的结果，应用激活函数，再与上投影结果相乘，然后应用下投影
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        # 返回下投影的结果
        return down_proj


# 从 transformers.models.llama.modeling_llama.repeat_kv 复制过来的函数
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    这是等效于 torch.repeat_interleave(x, dim=1, repeats=n_rep) 的函数。
    将隐藏状态从 (batch, num_key_value_heads, seqlen, head_dim) 扩展为 (batch, num_attention_heads, seqlen, head_dim)
    """
    # 获取隐藏状态张量的形状信息
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    # 如果 n_rep 等于 1，直接返回原始隐藏状态张量
    if n_rep == 1:
        return hidden_states
    # 扩展隐藏状态张量的维度，重复 n_rep 次
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    # 重新整形张量，以得到期望的形状 (batch, num_attention_heads * n_rep, seqlen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


# 从 transformers.models.llama.modeling_llama.LlamaAttention Llama->Cohere 复制过来的类，用于定义 CohereAttention 模型
class CohereAttention(nn.Module):
    """来自 'Attention Is All You Need' 论文的多头注意力机制"""

    # 类的文档字符串，描述其为来自论文的多头注意力机制
    # 初始化函数，用于创建一个新的实例
    def __init__(self, config: CohereConfig, layer_idx: Optional[int] = None):
        # 调用父类的初始化函数
        super().__init__()
        # 将传入的配置信息和层索引保存到对象中
        self.config = config
        self.layer_idx = layer_idx
        # 如果未传入层索引，则记录警告信息，建议在使用缓存时传入层索引以避免错误
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        # 从配置中获取注意力机制的丢弃率、隐藏层大小、注意力头数等信息
        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        # 检查隐藏层大小是否可以被注意力头数整除，否则抛出数值错误
        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        # 初始化线性投影层，用于将隐藏状态映射到注意力头和维度上
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=config.attention_bias)
        
        # 初始化旋转嵌入，这是一个辅助函数，用于增强模型在序列位置上的表示能力
        self._init_rope()

    # 辅助函数，用于初始化旋转嵌入
    def _init_rope(self):
        self.rotary_emb = CohereRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

    # 前向传播函数，接受输入的隐藏状态并计算输出
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        # 获取输入张量的尺寸信息
        bsz, q_len, _ = hidden_states.size()

        # 使用线性投影层对隐藏状态进行变换，得到查询、键、值的状态
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # 将查询、键、值的状态按照指定维度重新组织，并进行维度转置
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # 获取已存储的过去键值对信息，若存在则更新当前键值对状态
        past_key_value = getattr(self, "past_key_value", past_key_value)
        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # 对于已存储的过去键值对，根据 RoPE 模型的特定要求进行更新
            # 需要提供 sin 和 cos 参数以及缓存位置信息
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # 将键值对状态分组复制以便多头注意力计算
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # 计算注意力权重
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        # 如果存在注意力掩码，则将其应用于注意力权重
        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # 将注意力权重归一化并进行 dropout 处理
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        # 检查注意力输出的尺寸是否符合预期
        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        # 调整注意力输出张量的维度顺序，并保证连续性
        attn_output = attn_output.transpose(1, 2).contiguous()

        # 将注意力输出张量重新调整为隐藏状态的形状
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        # 使用输出投影层处理最终的注意力输出
        attn_output = self.o_proj(attn_output)

        # 如果不需要输出注意力权重，则将其置为 None
        if not output_attentions:
            attn_weights = None

        # 返回注意力输出张量、注意力权重张量以及更新后的过去键值对信息（如果有）
        return attn_output, attn_weights, past_key_value
# 从 `transform
    ):
        """
        Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
        first unpad the input, then computes the attention scores and pad the final attention scores.

        Args:
            query_states (`torch.Tensor`):
                Input query states to be passed to Flash Attention API
            key_states (`torch.Tensor`):
                Input key states to be passed to Flash Attention API
            value_states (`torch.Tensor`):
                Input value states to be passed to Flash Attention API
            attention_mask (`torch.Tensor`):
                The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
                position of padding tokens and 1 for the position of non-padding tokens.
            dropout (`float`):
                Attention dropout
            softmax_scale (`float`, *optional*):
                The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
        """
        # Determine if causal masking is required based on the configuration
        if not self._flash_attn_uses_top_left_mask:
            causal = self.is_causal
        else:
            # Temporary workaround for an issue with Flash Attention on RoCm
            # Remove this check once the issue is resolved in future versions
            causal = self.is_causal and query_length != 1

        # Check if there are any padding tokens in the input sequence
        if attention_mask is not None:
            batch_size = query_states.shape[0]
            # Unpad the input based on the attention mask
            query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(
                query_states, key_states, value_states, attention_mask, query_length
            )

            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

            # Compute attention scores for variable-length sequences
            attn_output_unpad = flash_attn_varlen_func(
                query_states,
                key_states,
                value_states,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_in_batch_q,
                max_seqlen_k=max_seqlen_in_batch_k,
                dropout_p=dropout,
                softmax_scale=softmax_scale,
                causal=causal,
            )

            # Pad the computed attention scores to match the original input length
            attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
        else:
            # Compute attention scores without considering padding
            attn_output = flash_attn_func(
                query_states, key_states, value_states, dropout, softmax_scale=softmax_scale, causal=causal
            )

        return attn_output
    # 从注意力掩码中获取未填充的数据的索引、当前序列长度和批次中最大序列长度
    indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)
    # 获取输入张量 key_layer 的形状信息
    batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape

    # 根据获取的索引重新排列 key_layer 张量的第一个轴，以对应未填充的数据
    key_layer = index_first_axis(
        key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
    )
    # 根据获取的索引重新排列 value_layer 张量的第一个轴，以对应未填充的数据
    value_layer = index_first_axis(
        value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
    )

    # 根据查询长度调整查询层的数据
    if query_length == kv_seq_len:
        # 若查询长度等于 key_value 序列长度，则根据获取的索引重新排列查询层的数据
        query_layer = index_first_axis(
            query_layer.reshape(batch_size * kv_seq_len, self.num_heads, head_dim), indices_k
        )
        # 更新当前查询序列长度和批次中最大查询序列长度
        cu_seqlens_q = cu_seqlens_k
        max_seqlen_in_batch_q = max_seqlen_in_batch_k
        indices_q = indices_k
    elif query_length == 1:
        # 若查询长度为1，则对应的当前查询序列长度为1，且索引为批次索引
        max_seqlen_in_batch_q = 1
        cu_seqlens_q = torch.arange(
            batch_size + 1, dtype=torch.int32, device=query_layer.device
        )  # 这里有一个内存拷贝，这样做效率很低。
        indices_q = cu_seqlens_q[:-1]
        query_layer = query_layer.squeeze(1)
    else:
        # 如果查询长度不等于 key_value 序列长度且不等于1，则假设存在左填充情况，根据注意力掩码和查询层调整输入数据
        attention_mask = attention_mask[:, -query_length:]
        query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, attention_mask)

    # 返回调整后的查询层、键层、值层、查询索引、当前序列长度元组和最大序列长度元组
    return (
        query_layer,
        key_layer,
        value_layer,
        indices_q,
        (cu_seqlens_q, cu_seqlens_k),
        (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
    )
# 定义一个名为 CohereSdpaAttention 的类，它继承自 CohereAttention 类
# 该类用于实现 Cohere 模型的自注意力机制，使用 torch.nn.functional.scaled_dot_product_attention
class CohereSdpaAttention(CohereAttention):
    """
    Cohere attention module using torch.nn.functional.scaled_dot_product_attention. This module inherits from
    `CohereAttention` as the weights of the module stays untouched. The only changes are on the forward pass to adapt to
    SDPA API.
    """

    # 重写 forward 方法，适应 SDPA API
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
    ):
        # forward 方法的具体实现在子类中完成，本处不直接提供具体实现
        pass

# 定义一个字典 COHERE_ATTENTION_CLASSES，包含不同的 CohereAttention 类型作为值，以字符串为键
COHERE_ATTENTION_CLASSES = {
    "eager": CohereAttention,
    "flash_attention_2": CohereFlashAttention2,
    "sdpa": CohereSdpaAttention,  # 将 CohereSdpaAttention 类作为 sdpa 类型的实现
}


class CohereDecoderLayer(nn.Module):
    # CohereDecoderLayer 类，继承自 nn.Module
    def __init__(self, config: CohereConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        # 初始化 self_attn 属性，根据配置选择不同的注意力机制实现类
        self.self_attn = COHERE_ATTENTION_CLASSES[config._attn_implementation](config=config, layer_idx=layer_idx)

        # 初始化 mlp 属性为 CohereMLP 类的实例
        self.mlp = CohereMLP(config)
        # 初始化 input_layernorm 属性为 CohereLayerNorm 类的实例，用于层归一化
        self.input_layernorm = CohereLayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    # 定义 forward 方法，实现模型的前向传播
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        # forward 方法的具体实现在子类中完成，本处不直接提供具体实现
        pass
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """
        if "padding_mask" in kwargs:
            # 发出警告信息，提示 `padding_mask` 参数已经不推荐使用
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

        # 将输入状态作为残差连接的起始点
        residual = hidden_states

        # Layer normalization，对输入状态进行归一化处理
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        # 调用自注意力机制进行处理，获取注意力加权的输出、注意力权重及可能的过去键值状态
        hidden_states_attention, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        # Fully Connected Layer
        # 通过多层感知机（MLP）进行全连接层的处理
        hidden_states_mlp = self.mlp(hidden_states)

        # 将残差、注意力加权输出和MLP输出相加得到最终的隐藏状态表示
        hidden_states = residual + hidden_states_attention + hidden_states_mlp

        # 构建输出元组
        outputs = (hidden_states,)

        # 如果需要返回注意力权重，则添加到输出元组中
        if output_attentions:
            outputs += (self_attn_weights,)

        # 如果需要返回缓存状态，则添加到输出元组中
        if use_cache:
            outputs += (present_key_value,)

        # 返回最终的输出元组
        return outputs
"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`CohereConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

@add_start_docstrings(
    "The bare Cohere Model outputting raw hidden-states without any specific head on top.",
    COHERE_START_DOCSTRING,
)
# Copied from transformers.models.llama.modeling_llama.LlamaPreTrainedModel with Llama->Cohere
class CoherePreTrainedModel(PreTrainedModel):
    config_class = CohereConfig  # 设置模型配置类为CohereConfig
    base_model_prefix = "model"  # 模型基本前缀为"model"
    supports_gradient_checkpointing = True  # 支持梯度检查点
    _no_split_modules = ["CohereDecoderLayer"]  # 不分割的模块列表，包括"CohereDecoderLayer"
    _skip_keys_device_placement = ["past_key_values"]  # 跳过设备放置的键列表，包括"past_key_values"
    _supports_flash_attn_2 = True  # 支持flash attention 2
    _supports_sdpa = True  # 支持sdpa
    _supports_cache_class = True  # 支持缓存类操作

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)  # 初始化线性层权重为正态分布
            if module.bias is not None:
                module.bias.data.zero_()  # 如果有偏置，初始化为零
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)  # 初始化嵌入层权重为正态分布
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()  # 如果有填充索引，对应位置初始化为零

    def _setup_cache(self, cache_cls, max_batch_size, max_cache_len: Optional[int] = None):
        if self.config._attn_implementation == "flash_attention_2" and cache_cls == StaticCache:
            raise ValueError(
                "`static` cache implementation is not compatible with `attn_implementation==flash_attention_2` "
                "make sure to use `sdpa` in the mean time, and open an issue at https://github.com/huggingface/transformers"
            )

        for layer in self.model.layers:
            device = layer.input_layernorm.weight.device
            if hasattr(self.config, "_pre_quantization_dtype"):
                dtype = self.config._pre_quantization_dtype
            else:
                dtype = layer.self_attn.o_proj.weight.dtype
            layer.self_attn.past_key_value = cache_cls(
                self.config, max_batch_size, max_cache_len, device=device, dtype=dtype
            )

    def _reset_cache(self):
        for layer in self.model.layers:
            layer.self_attn.past_key_value = None
# CohereModel 类定义，继承自 CoherePreTrainedModel
class CohereModel(CoherePreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`CohereDecoderLayer`]

    Args:
        config: CohereConfig
    """

    # 初始化函数，接受一个 config 对象作为参数
    def __init__(self, config: CohereConfig):
        # 调用父类的初始化函数
        super().__init__(config)
        # 将配置中的 pad_token_id 赋给当前实例的 padding_idx 属性
        self.padding_idx = config.pad_token_id
        # 将配置中的 vocab_size 赋给当前实例的 vocab_size 属性
        self.vocab_size = config.vocab_size

        # 创建一个词嵌入层，参数为 vocab_size（词汇表大小）、hidden_size（隐藏层大小）、padding_idx（填充标记的索引）
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        
        # 使用列表推导式创建包含多个 CohereDecoderLayer 对象的层列表，数量为 config.num_hidden_layers
        self.layers = nn.ModuleList(
            [CohereDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        
        # 创建一个 CohereLayerNorm 层，参数为 hidden_size（隐藏层大小）、eps（层归一化的 epsilon 值）
        self.norm = CohereLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # 设置梯度检查点为 False
        self.gradient_checkpointing = False

        # 创建一个因果遮罩（causal mask），用于区分因果和填充遮罩的创建。注意：这不利于 TorchScript、ONNX 和大型 max_position_embeddings 的序列化导出。
        # causal_mask 是一个二维的布尔类型张量，形状为 (config.max_position_embeddings, config.max_position_embeddings)
        causal_mask = torch.full(
            (config.max_position_embeddings, config.max_position_embeddings), fill_value=True, dtype=torch.bool
        )
        # 将上三角矩阵部分设置为 False，保留下三角矩阵和对角线为 True
        self.register_buffer("causal_mask", torch.triu(causal_mask, diagonal=1), persistent=False)
        
        # 调用初始化后的处理函数
        self.post_init()

    # 返回词嵌入层 embed_tokens
    def get_input_embeddings(self):
        return self.embed_tokens

    # 设置词嵌入层 embed_tokens 的值
    def set_input_embeddings(self, value):
        self.embed_tokens = value

    # forward 函数重写，对模型进行前向传播
    @add_start_docstrings_to_model_forward(COHERE_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ):
        # TODO: As of torch==2.2.0, the `attention_mask` passed to the model in `generate` is 2D and of dynamic length even when the static
        # KV cache is used. This is an issue for torch.compile which then recaptures cudagraphs at each decode steps due to the dynamic shapes.
        # (`recording cudagraph tree for symint key 13`, etc.), which is VERY slow. A workaround is `@torch.compiler.disable`, but this prevents using
        # `fullgraph=True`. See more context in https://github.com/huggingface/transformers/pull/29114
# 从 transformers.models.llama.modeling_llama.LlamaForCausalLM 复制而来，将 Llama 替换为 Cohere
class CohereForCausalLM(CoherePreTrainedModel):
    # 定义与 lm_head 权重相关的键列表，用于权重共享
    _tied_weights_keys = ["lm_head.weight"]

    # 初始化方法，接受一个配置对象 config
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)
        # 创建 CohereModel 的实例，并保存到 self.model 中
        self.model = CohereModel(config)
        # 从 config 中获取词汇表大小，并保存到 self.vocab_size 中
        self.vocab_size = config.vocab_size
        # 创建一个线性层，用于 LM 头部，将隐藏大小转换为词汇表大小，无偏置
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # 从 config 中获取 logit_scale，并保存到 self.logit_scale 中
        self.logit_scale = config.logit_scale
        # 从 config 中获取 tie_word_embeddings，并保存到 self.tie_word_embeddings 中
        self.tie_word_embeddings = config.tie_word_embeddings
        # 调用后处理初始化方法
        self.post_init()

    # 返回模型的输入嵌入层对象
    def get_input_embeddings(self):
        return self.model.embed_tokens

    # 设置模型的输入嵌入层对象为指定的值
    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    # 返回模型的输出嵌入层对象
    def get_output_embeddings(self):
        return self.lm_head

    # 设置模型的输出嵌入层对象为指定的新嵌入层
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    # 设置解码器为指定的 decoder 对象
    def set_decoder(self, decoder):
        self.model = decoder

    # 获取当前使用的解码器对象
    def get_decoder(self):
        return self.model

    # 前向传播方法，接收多个输入参数，详见装饰器中的 COHERE_INPUTS_DOCSTRING 描述
    # 返回类型为 CausalLMOutputWithPast，详见配置类 _CONFIG_FOR_DOC
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ):
        # 前向传播逻辑在后续实现中定义，具体实现细节参考具体模型文档

    # 为生成准备输入数据的静态方法，处理生成需要的输入参数
    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, cache_position=None, **kwargs
    ):
        # 输入数据准备逻辑在后续实现中定义，具体实现细节参考具体模型文档

    # 静态方法，重新排序缓存中的过去键值，以适应 beam search 中的索引变化
    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past
```