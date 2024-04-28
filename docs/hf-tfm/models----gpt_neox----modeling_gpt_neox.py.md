# `.\models\gpt_neox\modeling_gpt_neox.py`

```py
# 设置文件编码为 UTF-8
# 版权声明
# 许可证信息
# 引入 typing 模块
# 引入 torch 模块
# 引入 nn、BCEWithLogitsLoss、CrossEntropyLoss、MSELoss 模块
# 引入 functional 模块
# 引入 activations 模块下的 ACT2FN
# 引入 file_utils 模块下的函数和类
# 引入 modeling_outputs 模块下的类
# 引入 PreTrainedModel 类
# 引入 utils 模块下的函数和类
# 引入 configuration_gpt_neox 模块
# 检查是否存在 flash_attn_2
# 如果 flash_attn_2 存在，则引入 flash_attn 相关函数和类
# 引入 logger
# 检查点路径示例
# 真实检查点路径
# 配置类名
# GPT NeoX 预训练模型列表
# 函数：从注意力掩码中获取非填充数据
def _get_unpad_data(attention_mask):
    # 计算序列长度
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    # 扁平化注意力掩码，获取非零索引
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    # 批次中的最大序列长度
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    # 在批次中各序列长度的累积和
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )
# GPT NeoX 预训练模型父类
class GPTNeoXPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    # 配置类
    config_class = GPTNeoXConfig
    # 基础模型前缀
    base_model_prefix = "gpt_neox"
    # 是否支持梯度检查点
    supports_gradient_checkpointing = True
    # 需要跳过的模块
    _no_split_modules = ["GPTNeoXLayer"]
    # 需要跳过的键（用于设备定位）
    _skip_keys_device_placement = "past_key_values"
    # 是否支持 flash_attn_2
    _supports_flash_attn_2 = True
    # 初始化模型参数的权重
    def _init_weights(self, module):
        """Initialize the weights"""
        # 如果模块是线性层
        if isinstance(module, nn.Linear):
            # 初始化权重数据为正态分布
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果存在偏置，则初始化为零
            if module.bias is not None:
                module.bias.data.zero_()
        # 如果模块是嵌入层
        elif isinstance(module, nn.Embedding):
            # 初始化权重数据为正态分布
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果存在填充索引，则将填充索引对应的权重初始化为零
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        # 如果模块是层归一化层
        elif isinstance(module, nn.LayerNorm):
            # 将偏置初始化为零
            module.bias.data.zero_()
            # 将权重初始化为 1.0
            module.weight.data.fill_(1.0)
class GPTNeoXAttention(nn.Module):
    # 定义一个 GPTNeoXAttention 类，继承自 nn.Module
    def __init__(self, config):
        # 初始化方法，接收参数 config
        super().__init__()
        # 调用父类的初始化方法
        self.config = config
        # 将参数 config 赋值给实例变量 config
        self.num_attention_heads = config.num_attention_heads
        # 从 config 中获取注意力头的数量赋值给实例变量 num_attention_heads
        self.hidden_size = config.hidden_size
        # 从 config 中获取隐藏大小赋值给实例变量 hidden_size
        if self.hidden_size % self.num_attention_heads != 0:
            # 如果隐藏大小不能被注意力头数量整除
            raise ValueError(
                "The hidden size is not divisble by the number of attention heads! Make sure to update them"
                # 抛出错误提示
            )
        self.head_size = self.hidden_size // self.num_attention_heads
        # 计算每个注意力头的大小赋值给实例变量 head_size
        self.rotary_ndims = int(self.head_size * config.rotary_pct)
        # 计算旋转维度的大小并转换为整数赋值给实例变量 rotary_ndims
        self._init_bias(config.max_position_embeddings)
        # 调用初始化偏置的方法传入max_position_embeddings参数

        self.register_buffer("masked_bias", torch.tensor(-1e9), persistent=False)
        # 注册缓冲区，设置掩码偏置为常量-1e9
        self._init_rope()
        # 初始化绳索（Rotary Positional Embedding）

        self.norm_factor = self.head_size**-0.5
        # 计算规范化因子，即头的大小的倒数赋值给实例变量 norm_factor
        self.query_key_value = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=config.attention_bias)
        # 创建一个线性层，用于计算查询、键、值的线性变换
        self.dense = nn.Linear(config.hidden_size, config.hidden_size, bias=config.attention_bias)
        # 创建一个线性层，用于最终的输出变换
        self.attention_dropout = nn.Dropout(config.attention_dropout)
        # 创建一个 Dropout 层，并传入注意力丢弃率参数
        self.is_causal = True
        # 设置为因果关系（causal）模式

    def _init_bias(self, max_positions, device=None):
        # 初始化偏置的方法，接收最大位置参数和设备参数
        self.register_buffer(
            "bias",
            torch.tril(torch.ones((max_positions, max_positions), dtype=torch.bool)).view(
                1, 1, max_positions, max_positions
            ),
            persistent=False,
        )
        # 注册缓冲区，生成下三角矩阵，用于计算掩码
        if device is not None:
            self.bias = self.bias.to(device)
            # 如果有设备参数，则移动偏置到指定设备

    def _init_rope(self):
        # 初始化绳索（Rotary Positional Embedding）的方法
        if self.config.rope_scaling is None:
            # 如果没有指定绳索缩放
            self.rotary_emb = GPTNeoXRotaryEmbedding(
                self.rotary_ndims, self.config.max_position_embeddings, base=self.config.rotary_emb_base
            )
            # 创建 GPTNeoXRotaryEmbedding 类实例
        else:
            scaling_type = self.config.rope_scaling["type"]
            # 获取绳索缩放类型
            scaling_factor = self.config.rope_scaling["factor"]
            # 获取绳索缩放因子
            if scaling_type == "linear":
                # 如果绳索缩放类型是线性
                self.rotary_emb = GPTNeoXLinearScalingRotaryEmbedding(
                    self.rotary_ndims,
                    self.config.max_position_embeddings,
                    base=self.config.rotary_emb_base,
                    scaling_factor=scaling_factor,
                )
                # 创建 GPTNeoXLinearScalingRotaryEmbedding 类实例
            elif scaling_type == "dynamic":
                # 如果绳索缩放类型是动态
                self.rotary_emb = GPTNeoXDynamicNTKScalingRotaryEmbedding(
                    self.rotary_ndims,
                    self.config.max_position_embeddings,
                    base=self.config.rotary_emb_base,
                    scaling_factor=scaling_factor,
                )
                # 创建 GPTNeoXDynamicNTKScalingRotaryEmbedding 类实例
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")
                # 抛出错误，未知的绳索缩放类型
    # 前向传播函数，接收隐藏状态、注意力掩码、位置编码、头部掩码、过去层状态、是否使用缓存、是否输出注意力权重、填充掩码等参数
    def forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: torch.FloatTensor,
        position_ids: torch.LongTensor,
        head_mask: Optional[torch.FloatTensor] = None,
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        padding_mask: Optional[torch.Tensor] = None,
    ):
        # 检查是否有过去层状态
        has_layer_past = layer_past is not None

        # 计算 QKV
        qkv = self.query_key_value(hidden_states)

        # 重塑 QKV 张量形状
        new_qkv_shape = qkv.size()[:-1] + (self.num_attention_heads, 3 * self.head_size)
        qkv = qkv.view(*new_qkv_shape)

        # 分离 Q、K、V
        query = qkv[..., : self.head_size].permute(0, 2, 1, 3)
        key = qkv[..., self.head_size : 2 * self.head_size].permute(0, 2, 1, 3)
        value = qkv[..., 2 * self.head_size :].permute(0, 2, 1, 3)

        # 计算旋转嵌入
        query_rot = query[..., : self.rotary_ndims]
        query_pass = query[..., self.rotary_ndims :]
        key_rot = key[..., : self.rotary_ndims]
        key_pass = key[..., self.rotary_ndims :]

        # 计算旋转嵌入的位置偏移
        seq_len = key.shape[-2]
        if has_layer_past:
            seq_len += layer_past[0].shape[-2]
        cos, sin = self.rotary_emb(value, seq_len=seq_len)
        query, key = apply_rotary_pos_emb(query_rot, key_rot, cos, sin, position_ids)
        query = torch.cat((query, query_pass), dim=-1)
        key = torch.cat((key, key_pass), dim=-1)

        # 缓存 QKV 值
        if has_layer_past:
            past_key = layer_past[0]
            past_value = layer_past[1]
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)
        present = (key, value) if use_cache else None

        # 计算注意力
        attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)

        # 重塑输出
        attn_output = self._merge_heads(attn_output, self.num_attention_heads, self.head_size)
        attn_output = self.dense(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs

    @classmethod
    def _split_heads(cls, tensor, num_attention_heads, attn_head_size):
        """
        Splits hidden dim into attn_head_size and num_attention_heads
        """
        # tensor: [bs, seq_len, hidden_size]
        # 将隐藏维度分割成注意力头大小和注意力头数量
        new_shape = tensor.size()[:-1] + (num_attention_heads, attn_head_size)
        # -> [bs, seq_len, num_attention_heads, attn_head_size]
        tensor = tensor.view(new_shape)
        # -> [bs, num_attention_heads, seq_len, attn_head_size]
        tensor = tensor.permute(0, 2, 1, 3)
        return tensor

    @classmethod
    def _merge_heads(cls, tensor, num_attention_heads, attn_head_size):
        """
        Merges attn_head_size dim and num_attn_heads dim into hidden dim
        """
        # tensor [bs, num_attention_heads, seq_len, attn_head_size]
        # 合并注意力头大小维度和注意力头数量维度到隐藏维度
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        # -> [bs, seq_len, num_attention_heads, attn_head_size]
        tensor = tensor.view(tensor.size(0), tensor.size(1), num_attention_heads * attn_head_size)
        # -> [bs, seq_len, hidden_size]
        return tensor
    # 定义注意力计算函数，接受查询、键、值以及注意力掩码等参数
    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        # 获取查询、键、值的维度信息
        # q, k, v: [bs, num_attention_heads, seq_len, attn_head_size]
        batch_size, num_attention_heads, query_length, attn_head_size = query.size()
        key_length = key.size(-2)

        # 根据键的长度动态生成因果掩码
        # 如果需要，根据键的长度动态增加因果掩码
        if key_length > self.bias.shape[-1]:
            self._init_bias(key_length, device=key.device)
        causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]

        # 重塑查询和键的形状以便进行矩阵乘法
        query = query.view(batch_size * num_attention_heads, query_length, attn_head_size)
        key = key.view(batch_size * num_attention_heads, key_length, attn_head_size)
        # 初始化注意力分数矩阵
        attn_scores = torch.zeros(
            batch_size * num_attention_heads,
            query_length,
            key_length,
            dtype=query.dtype,
            device=key.device,
        )
        # 计算注意力分数
        attn_scores = torch.baddbmm(
            attn_scores,
            query,
            key.transpose(1, 2),
            beta=1.0,
            alpha=self.norm_factor,
        )
        attn_scores = attn_scores.view(batch_size, num_attention_heads, query_length, key_length)

        # 初始化掩码值
        mask_value = torch.finfo(attn_scores.dtype).min
        # 需要将掩码值转换为张量，以避免出现错误
        mask_value = torch.tensor(mask_value, dtype=attn_scores.dtype).to(attn_scores.device)
        # 根据因果掩码对注意力分数进行掩码处理
        attn_scores = torch.where(causal_mask, attn_scores, mask_value)

        # 如果存在注意力掩码，则应用该掩码
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask

        # 对注意力分数进行 softmax 操作
        attn_weights = nn.functional.softmax(attn_scores, dim=-1)
        attn_weights = attn_weights.to(value.dtype)

        # 如果存在头部掩码，则对注意力权重进行掩码处理
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        # 对注意力权重进行 dropout 处理
        attn_weights = self.attention_dropout(attn_weights)

        # 计算注意力输出
        attn_output = torch.matmul(attn_weights, value)
        return attn_output, attn_weights
class GPTNeoXFlashAttention2(GPTNeoXAttention):
    """
    GPTNeoX flash attention module. This module inherits from `GPTNeoXAttention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: Should be removed once Flash Attention for RoCm is bumped to 2.1.
        # flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignement, that was made default for flash_attn>=2.1. This attribute is used to handle this difference. Reference: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.
        # Beware that with flash_attn<2.1, using q_seqlen != k_seqlen (except for the case q_seqlen == 1) produces a wrong mask (top-left).
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: torch.FloatTensor,
        position_ids: torch.LongTensor,
        head_mask: Optional[torch.FloatTensor] = None,
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    # Copied from transformers.models.llama.modeling_llama.LlamaFlashAttention2._flash_attention_forward
    def _flash_attention_forward(
        self, query_states, key_states, value_states, attention_mask, query_length, dropout=0.0, softmax_scale=None
    ):
        """
        调用 Flash Attention 的前向方法 - 如果输入的隐藏状态包含至少一个填充标记，则首先对输入进行去填充，然后计算注意力分数并填充最终的注意力分数。

        Args:
            query_states (`torch.Tensor`):
                要传递给 Flash Attention API 的输入查询状态
            key_states (`torch.Tensor`):
                要传递给 Flash Attention API 的输入键状态
            value_states (`torch.Tensor`):
                要传递给 Flash Attention API 的输入值状态
            attention_mask (`torch.Tensor`):
                填充掩码 - 对应于大小为 `(batch_size, seq_len)` 的张量，其中 0 表示填充标记的位置，1 表示非填充标记的位置。
            dropout (`int`, *optional*):
                注意力丢弃率
            softmax_scale (`float`, *optional*):
                在应用 softmax 之前的 QK^T 的缩放。默认为 1 / sqrt(head_dim)
        """
        if not self._flash_attn_uses_top_left_mask:
            causal = self.is_causal
        else:
            # TODO: 一旦 Flash Attention for RoCm 升级到 2.1，删除 `query_length != 1` 检查。有关详细信息，请参阅 LlamaFlashAttention2 __init__ 中的注释。
            causal = self.is_causal and query_length != 1

        # 序列中至少包含一个填充标记
        if attention_mask is not None:
            batch_size = query_states.shape[0]
            query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(
                query_states, key_states, value_states, attention_mask, query_length
            )

            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

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

            attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
        else:
            attn_output = flash_attn_func(
                query_states, key_states, value_states, dropout, softmax_scale=softmax_scale, causal=causal
            )

        return attn_output

    # 从 transformers.models.llama.modeling_llama.LlamaFlashAttention2._upad_input 复制，将 num_heads->num_attention_heads
    # 用于处理输入数据，根据注意力掩码获取未填充数据的索引、当前序列长度和批次中的最大序列长度
    def _upad_input(self, query_layer, key_layer, value_layer, attention_mask, query_length):
        # 获取未填充数据的索引、当前序列长度和批次中的最大序列长度
        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)
        # 获取批次大小、键值序列长度、键值头数和头维度
        batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape

        # 重新组织键层数据，根据未填充数据的索引
        key_layer = index_first_axis(
            key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        # 重新组织值层数据，根据未填充数据的索引
        value_layer = index_first_axis(
            value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        
        # 如果查询长度等于键值序列长度
        if query_length == kv_seq_len:
            # 重新组织查询层数据，根据未填充数据的索引
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, self.num_attention_heads, head_dim), indices_k
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        # 如果查询长度为1
        elif query_length == 1:
            max_seqlen_in_batch_q = 1
            # 生成一个序列长度为批次大小的序列
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            )  # There is a memcpy here, that is very bad.
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            # 假设左填充，根据查询长度截取注意力掩码
            attention_mask = attention_mask[:, -query_length:]
            # 处理查询层数据，获取未填充数据的索引、当前序列长度和批次中的最大序列长度
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, attention_mask)

        # 返回处理后的查询层、键层、值层、查询层的索引、当前序列长度元组、最大序列长度元组
        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )
def attention_mask_func(attention_scores, ltor_mask):
    # 使用左到右掩码填充注意力分数张量
    attention_scores.masked_fill_(~ltor_mask, torch.finfo(attention_scores.dtype).min)
    return attention_scores


class GPTNeoXRotaryEmbedding(nn.Module):
    # 从transformers.models.llama.modeling_llama.LlamaRotaryEmbedding.__init__中复制而来
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # 为了使`torch.jit.trace`正常工作，在此处构建
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.outer(t, self.inv_freq)
        # 与论文不同，但使用不同的排列顺序以获得相同的计算结果
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len],
            self.sin_cached[:seq_len],
        )


class GPTNeoXLinearScalingRotaryEmbedding(GPTNeoXRotaryEmbedding):
    """GPTNeoXRotaryEmbedding扩展了线性缩放。感谢Reddit用户/u/kaiokendev"""

    # 从transformers.models.llama.modeling_llama.LlamaLinearScalingRotaryEmbedding.__init__中复制而来
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        t = t / self.scaling_factor

        freqs = torch.outer(t, self.inv_freq)
        # 与论文不同，但使用不同的排列顺序以获得相同的计算结果
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)


class GPTNeoXDynamicNTKScalingRotaryEmbedding(GPTNeoXRotaryEmbedding):
    """GPTNeoXRotaryEmbedding extended with Dynamic NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla"""

    # 从transformers.models.llama.modeling_llama.LlamaDynamicNTKScalingRotaryEmbedding.__init__中复制而来
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        # 初始化GPTNeoXRotaryEmbedding对象，设置动态NTK缩放因子
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        # 设置cos和sin缓存，用于后续计算
        self.max_seq_len_cached = seq_len

        if seq_len > self.max_position_embeddings:
            # 根据序列长度和最大位置嵌入长度计算基础值
            base = self.base * (
                (self.scaling_factor * seq_len / self.max_position_embeddings) - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            # 计算频率的倒数
            inv_freq = 1.0 / (base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
            # 注册缓冲区，存储频率的倒数
            self.register_buffer("inv_freq", inv_freq, persistent=False)

        # 生成序列长度范围的张量
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        # 计算频率矩阵
        freqs = torch.outer(t, self.inv_freq)
        # 不同于论文，但使用不同的排列顺序以获得相同的计算结果
        emb = torch.cat((freqs, freqs), dim=-1)
        # 注册缓冲区，存储cos和sin值
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    # 将输入的隐藏维度的一半进行旋转
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# Copied from transformers.models.llama.modeling_llama.apply_rotary_pos_emb
def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    # 从旋转位置嵌入到查询和键张量
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class GPTNeoXMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense_h_to_4h = nn.Linear(config.hidden_size, config.intermediate_size)
        self.dense_4h_to_h = nn.Linear(config.intermediate_size, config.hidden_size)
        self.act = ACT2FN[config.hidden_act]

    def forward(self, hidden_states):
        hidden_states = self.dense_h_to_4h(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.dense_4h_to_h(hidden_states)
        return hidden_states


GPT_NEOX_ATTENTION_CLASSES = {
    "eager": GPTNeoXAttention,
    "flash_attention_2": GPTNeoXFlashAttention2,
}


class GPTNeoXLayer(nn.Module):
    # 初始化函数，接受配置参数
    def __init__(self, config):
        # 调用父类的初始化函数
        super().__init__()
        # 根据配置参数设置是否使用并行残差连接
        self.use_parallel_residual = config.use_parallel_residual
        # 初始化输入层归一化
        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 初始化注意力后归一化
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 初始化注意力后的dropout
        self.post_attention_dropout = nn.Dropout(config.hidden_dropout)
        # 初始化MLP后的dropout
        self.post_mlp_dropout = nn.Dropout(config.hidden_dropout)
        # 初始化注意力层
        self.attention = GPT_NEOX_ATTENTION_CLASSES[config._attn_implementation](config)
        # 初始化MLP层
        self.mlp = GPTNeoXMLP(config)

    # 前向传播函数
    def forward(
        self,
        hidden_states: Optional[torch.FloatTensor],
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
    ):
        # 使用注意力层处理输入hidden_states
        attention_layer_outputs = self.attention(
            self.input_layernorm(hidden_states),
            attention_mask=attention_mask,
            position_ids=position_ids,
            layer_past=layer_past,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        # 获取注意力层的输出
        attn_output = attention_layer_outputs[0]  # output_attn: attn_output, present, (attn_weights)
        # 对注意力输出进行dropout
        attn_output = self.post_attention_dropout(attn_output)
        # 获取注意力层的其他输出
        outputs = attention_layer_outputs[1:]

        if self.use_parallel_residual:
            # 如果使用并行残差连接
            # 伪代码:
            # x = x + attn(ln1(x)) + mlp(ln2(x))
            # 使用MLP处理注意力后归一化的hidden_states
            mlp_output = self.mlp(self.post_attention_layernorm(hidden_states))
            # 对MLP输出进行dropout
            mlp_output = self.post_mlp_dropout(mlp_output)
            # 计算最终的hidden_states
            hidden_states = mlp_output + attn_output + hidden_states
        else:
            # 如果不使用并行残差连接
            # 伪代码:
            # x = x + attn(ln1(x))
            # x = x + mlp(ln2(x))
            # 将注意力输出与hidden_states相加
            attn_output = attn_output + hidden_states
            # 使用MLP处理注意力输出后归一化的hidden_states
            mlp_output = self.mlp(self.post_attention_layernorm(attn_output))
            # 对MLP输出进行dropout
            mlp_output = self.post_mlp_dropout(mlp_output)
            # 计算最终的hidden_states
            hidden_states = mlp_output + attn_output

        if use_cache:
            # 如果使用缓存，将hidden_states添加到outputs中
            outputs = (hidden_states,) + outputs  # hidden_states, present, (attn_weights)
        else:
            # 如果不使用缓存，将hidden_states添加到outputs中，但不包含present
            outputs = (hidden_states,) + outputs[1:]  # hidden_states, (attn_weights)

        # 返回outputs
        return outputs
# GPT_NEOX_START_DOCSTRING 是一个包含模型文档字符串的原始字符串，描述了模型的基本信息和参数
GPT_NEOX_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`~GPTNeoXConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

# GPT_NEOX_INPUTS_DOCSTRING 是一个包含模型输入文档字符串的原始字符串，描述了模型输入参数的详细信息
GPT_NEOX_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `({0})`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.FloatTensor` of shape `({0})`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        position_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        inputs_embeds (`torch.FloatTensor` of shape `({0}, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert *input_ids* indices into associated vectors than the
            model's internal embedding lookup matrix.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~file_utils.ModelOutput`] instead of a plain tuple.
"""

# @add_start_docstrings 是一个装饰器，用于添加文档字符串到函数或类中
@add_start_docstrings(
    # 描述 GPTNeoX 模型的基本特性，输出原始隐藏状态而不带任何特定的头部
    # GPT_NEOX_START_DOCSTRING 是一个文档字符串的起始标记
# 定义 GPTNeoXModel 类，继承自 GPTNeoXPreTrainedModel 类
class GPTNeoXModel(GPTNeoXPreTrainedModel):
    # 初始化方法，接受配置参数 config
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)
        # 将配置参数保存到 self.config 中
        self.config = config

        # 创建词嵌入层，输入大小为词汇表大小，输出大小为隐藏层大小
        self.embed_in = nn.Embedding(config.vocab_size, config.hidden_size)
        # 创建词嵌入层的 dropout 层
        self.emb_dropout = nn.Dropout(config.hidden_dropout)
        # 创建多层 GPTNeoXLayer，根据配置中的隐藏层数量
        self.layers = nn.ModuleList([GPTNeoXLayer(config) for _ in range(config.num_hidden_layers)])
        # 创建最终的 LayerNorm 层
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 根据配置中的 _attn_implementation 判断是否使用 flash_attention_2
        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"

        # 初始化梯度检查点为 False
        self.gradient_checkpointing = False

        # 初始化权重并应用最终处理
        self.post_init()

    # 获取输入词嵌入层
    def get_input_embeddings(self):
        return self.embed_in

    # 设置输入词嵌入层
    def set_input_embeddings(self, value):
        self.embed_in = value

    # 前向传播方法，接受多个输入参数
    @add_start_docstrings_to_model_forward(GPT_NEOX_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        real_checkpoint=_REAL_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPast,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,



# 定义 GPTNeoXForCausalLM 类，继承自 GPTNeoXPreTrainedModel 类
@add_start_docstrings(
    """GPTNeoX Model with a `language modeling` head on top for CLM fine-tuning.""", GPT_NEOX_START_DOCSTRING
)
class GPTNeoXForCausalLM(GPTNeoXPreTrainedModel):
    # 定义 tied_weights_keys 列表
    _tied_weights_keys = ["embed_out.weight"]

    # 初始化方法，接受配置参数 config
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)

        # 创建 GPTNeoXModel 实例
        self.gpt_neox = GPTNeoXModel(config)
        # 创建线性层，将隐藏层映射到词汇表大小，不使用偏置
        self.embed_out = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # 初始化权重并应用最终处理
        self.post_init()

    # 获取输出词嵌入层
    def get_output_embeddings(self):
        return self.embed_out

    # 设置输出词嵌入层
    def set_output_embeddings(self, new_embeddings):
        self.embed_out = new_embeddings

    # 前向传播方法，接受多个输入参数
    @add_start_docstrings_to_model_forward(GPT_NEOX_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    # 定义一个方法用于模型的前向传播
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,  # 输入的 token ID
        attention_mask: Optional[torch.FloatTensor] = None,  # 注意力掩码
        position_ids: Optional[torch.LongTensor] = None,  # 位置 ID
        inputs_embeds: Optional[torch.FloatTensor] = None,  # 输入的嵌入向量
        head_mask: Optional[torch.FloatTensor] = None,  # 头部掩码
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,  # 过去的键值对
        labels: Optional[torch.LongTensor] = None,  # 标签
        use_cache: Optional[bool] = None,  # 是否使用缓存
        output_attentions: Optional[bool] = None,  # 是否输出注意力
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态
        return_dict: Optional[bool] = None,  # 是否返回字典
    # 准备生成的输入
    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        input_shape = input_ids.shape  # 获取输入的形状
        # 如果使用过去的键值对，则截断 decoder_input_ids
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]

            # 一些生成方法已经只传递最后一个输入 ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # 默认保留最后一个 ID
                remove_prefix_length = input_ids.shape[1] - 1

            input_ids = input_ids[:, remove_prefix_length:]

        position_ids = kwargs.get("position_ids", None)  # 获取位置 ID
        if attention_mask is not None and position_ids is None:
            # 为批量生成动态创建位置 ID
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # 如果没有注意力掩码，则创建一个全为 1 的掩码
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_shape)

        # 如果传递了 `inputs_embeds`，则只在第一个生成步骤中使用它们
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}
        model_inputs.update(
            {
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
        )

        return model_inputs

    # 重新排序缓存
    def _reorder_cache(self, past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past[:2])
                + layer_past[2:],
            )
        return reordered_past
# 定义一个带有序列分类头的 GPTNeoX 模型变换器（线性层）
# GPTNeoXForSequenceClassification 使用最后一个标记进行分类，与其他因果模型（例如 GPT-1）一样。
# 由于它在最后一个标记上进行分类，需要知道最后一个标记的位置。如果在配置中定义了 pad_token_id，则在每一行中找到不是填充标记的最后一个标记。
# 如果未定义 pad_token_id，则简单地取每一行中的最后一个值。当传递 inputs_embeds 而不是 input_ids 时，无法猜测填充标记，因此执行相同操作（取每一行中的最后一个值）。
class GPTNeoXForSequenceClassification(GPTNeoXPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.gpt_neox = GPTNeoXModel(config)
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)

        # 初始化权重并应用最终处理
        self.post_init()

    # 前向传播函数
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
# GPTNeoXForTokenClassification 类
class GPTNeoXForTokenClassification(GPTNeoXPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.gpt_neox = GPTNeoXModel(config)
        self.dropout = nn.Dropout(config.classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # 初始化权重并应用最终处理
        self.post_init()

    # 前向传播函数
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    # 定义一个方法用于前向传播，接收多个输入参数并返回预测结果
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,  # 输入的 token ID
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,  # 用于存储过去的 key 和 value
        attention_mask: Optional[torch.FloatTensor] = None,  # 注意力掩码
        token_type_ids: Optional[torch.LongTensor] = None,  # token 类型 ID
        position_ids: Optional[torch.LongTensor] = None,  # 位置 ID
        head_mask: Optional[torch.FloatTensor] = None,  # 头部掩码
        inputs_embeds: Optional[torch.FloatTensor] = None,  # 输入的嵌入向量
        labels: Optional[torch.LongTensor] = None,  # 标签
        use_cache: Optional[bool] = None,  # 是否使用缓存
        output_attentions: Optional[bool] = None,  # 是否输出注意力
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态
        return_dict: Optional[bool] = None,  # 是否返回字典形式的结果
    ) -> Union[Tuple, TokenClassifierOutput]:  # 返回值的类型注释

        # 如果没有指定返回字典，则使用配置中的默认值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用 GPT-NeoX 模型进行前向传播
        outputs = self.gpt_neox(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取模型输出的隐藏状态
        hidden_states = outputs[0]
        # 对隐藏状态进行 dropout 处理
        hidden_states = self.dropout(hidden_states)
        # 将隐藏状态传入分类器得到 logits
        logits = self.classifier(hidden_states)

        # 初始化损失值
        loss = None
        # 如果存在标签，则计算损失
        if labels is not None:
            labels = labels.to(logits.device)
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        # 如果不返回字典形式的结果
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 返回 TokenClassifierOutput 类型的结果
        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 定义一个 GPT-NeoX 模型，带有一个用于提取式问答任务（如 SQuAD）的跨度分类头部（在隐藏状态输出之上的线性层，用于计算“跨度起始对数”和“跨度结束对数”）
@add_start_docstrings(
    """
    The GPT-NeoX Model transformer with a span classification head on top for extractive question-answering tasks like
    SQuAD (a linear layer on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    GPT_NEOX_START_DOCSTRING,
)
class GPTNeoXForQuestionAnswering(GPTNeoXPreTrainedModel):
    # 初始化函数
    def __init__(self, config):
        # 调用父类的初始化函数
        super().__init__(config)
        # 获取标签数量
        self.num_labels = config.num_labels
        # 创建 GPT-NeoX 模型
        self.gpt_neox = GPTNeoXModel(config)
        # 创建一个线性层，用于输出答案的起始和结束位置
        self.qa_outputs = nn.Linear(config.hidden_size, 2)

        # 初始化权重并应用最终处理
        self.post_init()

    # 前向传播函数
    @add_start_docstrings_to_model_forward(GPT_NEOX_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=QuestionAnsweringModelOutput,
        config_class=_CONFIG_FOR_DOC,
        real_checkpoint=_REAL_CHECKPOINT_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        start_positions: Optional[torch.LongTensor] = None,
        end_positions: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, QuestionAnsweringModelOutput]:
        r"""
        start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        """
        # 设置返回字典，如果未指定则使用配置中的返回字典设置
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 使用 GPT-NeoX 模型进行前向传播
        outputs = self.gpt_neox(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取序列输出
        sequence_output = outputs[0]

        # 通过 QA 输出层获取 logits
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # 如果在多 GPU 上，添加一个维度
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1).to(start_logits.device)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1).to(end_logits.device)
            # 有时开始/结束位置超出模型输入范围，忽略这些项
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            # 定义交叉熵损失函数
            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        # 返回问题回答模型输出
        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
```