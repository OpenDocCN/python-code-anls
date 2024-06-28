# `.\models\distilbert\modeling_distilbert.py`

```py
# Copied from transformers.models.llama.modeling_llama._get_unpad_data
# 从transformers.models.llama.modeling_llama._get_unpad_data复制而来，用于获取不需要填充的数据
def _get_unpad_data(attention_mask):
    # 计算每个样本序列的长度，attention_mask是一个表示注意力掩码的张量
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    # 找到所有需要注意力的位置的索引，返回的indices是一个一维张量
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    # 在批次中计算序列长度的最大值
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    
    # 对序列长度进行累积和，并在左侧填充一个元素为0的张量
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    
    # 返回计算得到的结果：索引、累积序列长度、批次中的最大序列长度
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )
# 创建正弦位置嵌入向量的函数，根据是否启用了 DeepSpeed 的 zero3 功能进行处理
def create_sinusoidal_embeddings(n_pos: int, dim: int, out: torch.Tensor):
    # 检查是否启用了 DeepSpeed 的 zero3 功能
    if is_deepspeed_zero3_enabled():
        # 导入 DeepSpeed 库
        import deepspeed
        # 使用 DeepSpeed 的 GatheredParameters 函数，确保仅在 modifier_rank=0 的进程上执行
        with deepspeed.zero.GatheredParameters(out, modifier_rank=0):
            # 如果当前进程的 rank 为 0，则执行创建正弦位置嵌入向量的函数
            if torch.distributed.get_rank() == 0:
                _create_sinusoidal_embeddings(n_pos=n_pos, dim=dim, out=out)
    else:
        # 如果未启用 DeepSpeed 的 zero3 功能，则直接调用创建正弦位置嵌入向量的函数
        _create_sinusoidal_embeddings(n_pos=n_pos, dim=dim, out=out)


# 实际创建正弦位置嵌入向量的函数，计算正弦和余弦值并将其赋给输出张量
def _create_sinusoidal_embeddings(n_pos: int, dim: int, out: torch.Tensor):
    # 根据位置和维度创建正弦位置编码矩阵
    position_enc = np.array([[pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)] for pos in range(n_pos)])
    # 设置输出张量不需要梯度
    out.requires_grad = False
    # 将计算得到的正弦值赋给输出张量的偶数索引位置
    out[:, 0::2] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
    # 将计算得到的余弦值赋给输出张量的奇数索引位置
    out[:, 1::2] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
    # 分离输出张量，使其不再跟踪梯度
    out.detach_()


# 表示嵌入层的类，用于组合词嵌入和位置嵌入
class Embeddings(nn.Module):
    # 初始化函数，配置词嵌入和位置嵌入
    def __init__(self, config: PretrainedConfig):
        super().__init__()
        # 初始化词嵌入层，包括词汇大小、嵌入维度和填充标记索引
        self.word_embeddings = nn.Embedding(config.vocab_size, config.dim, padding_idx=config.pad_token_id)
        # 初始化位置嵌入层，包括最大位置嵌入大小和嵌入维度
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.dim)
        # 如果配置要求使用正弦位置嵌入，调用创建正弦位置嵌入的函数
        if config.sinusoidal_pos_embds:
            create_sinusoidal_embeddings(
                n_pos=config.max_position_embeddings, dim=config.dim, out=self.position_embeddings.weight
            )
        
        # 初始化 LayerNorm 层，用于标准化输入张量
        self.LayerNorm = nn.LayerNorm(config.dim, eps=1e-12)
        # 初始化 Dropout 层，用于随机失活输入张量的元素
        self.dropout = nn.Dropout(config.dropout)
        # 注册位置标识符张量，用于标识输入张量中每个位置的索引
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )
    def forward(self, input_ids: torch.Tensor, input_embeds: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Parameters:
            input_ids (torch.Tensor):
                torch.tensor(bs, max_seq_length) The token ids to embed.
            input_embeds (*optional*, torch.Tensor):
                The pre-computed word embeddings. Can only be passed if the input ids are `None`.

        Returns:
            torch.tensor(bs, max_seq_length, dim) The embedded tokens (plus position embeddings, no token_type embeddings)
        """
        # 如果传入了 input_ids，则使用 self.word_embeddings 对其进行 embedding
        if input_ids is not None:
            input_embeds = self.word_embeddings(input_ids)  # (bs, max_seq_length, dim)

        seq_length = input_embeds.size(1)  # 获取序列长度

        # 如果模型实例包含名为 "position_ids" 的属性，则使用其注册的位置信息
        if hasattr(self, "position_ids"):
            position_ids = self.position_ids[:, :seq_length]  # (1, max_seq_length)
        else:
            # 否则根据序列长度生成位置信息（从0到seq_length-1）
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)  # (max_seq_length)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)  # (bs, max_seq_length)

        position_embeddings = self.position_embeddings(position_ids)  # 根据位置信息获取位置嵌入 (bs, max_seq_length, dim)

        embeddings = input_embeds + position_embeddings  # 将词嵌入和位置嵌入相加得到最终嵌入 (bs, max_seq_length, dim)
        embeddings = self.LayerNorm(embeddings)  # Layer normalization (bs, max_seq_length, dim)
        embeddings = self.dropout(embeddings)  # Dropout 正则化 (bs, max_seq_length, dim)

        return embeddings  # 返回嵌入后的结果
    # 定义一个多头自注意力机制的神经网络模块
    class MultiHeadSelfAttention(nn.Module):
        # 初始化函数，接受一个预训练配置对象作为参数
        def __init__(self, config: PretrainedConfig):
            super().__init__()
            self.config = config

            # 从配置对象中获取多头注意力的数量和维度
            self.n_heads = config.n_heads
            self.dim = config.dim
            # 使用指定的注意力丢弃率创建一个dropout层
            self.dropout = nn.Dropout(p=config.attention_dropout)
            # 默认非因果关系
            self.is_causal = False

            # 确保多头数能够均匀地分割维度
            if self.dim % self.n_heads != 0:
                # 如果无法均匀分割则引发值错误
                raise ValueError(f"self.n_heads: {self.n_heads} must divide self.dim: {self.dim} evenly")

            # 分别定义线性变换层：query、key、value和输出层
            self.q_lin = nn.Linear(in_features=config.dim, out_features=config.dim)
            self.k_lin = nn.Linear(in_features=config.dim, out_features=config.dim)
            self.v_lin = nn.Linear(in_features=config.dim, out_features=config.dim)
            self.out_lin = nn.Linear(in_features=config.dim, out_features=config.dim)

            # 初始化一个空集合，用于记录被修剪的注意力头
            self.pruned_heads: Set[int] = set()
            # 计算每个注意力头的大小
            self.attention_head_size = self.dim // self.n_heads

        # 定义修剪注意力头的方法
        def prune_heads(self, heads: List[int]):
            if len(heads) == 0:
                return
            # 调用外部函数找到可修剪的注意力头和索引
            heads, index = find_pruneable_heads_and_indices(
                heads, self.n_heads, self.attention_head_size, self.pruned_heads
            )
            # 修剪线性层
            self.q_lin = prune_linear_layer(self.q_lin, index)
            self.k_lin = prune_linear_layer(self.k_lin, index)
            self.v_lin = prune_linear_layer(self.v_lin, index)
            self.out_lin = prune_linear_layer(self.out_lin, index, dim=1)
            # 更新超参数：减少的注意力头数和调整后的维度
            self.n_heads = self.n_heads - len(heads)
            self.dim = self.attention_head_size * self.n_heads
            # 将修剪的头添加到已修剪的头的集合中
            self.pruned_heads = self.pruned_heads.union(heads)

        # 前向传播方法，实现自注意力机制
        def forward(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            mask: torch.Tensor,
            head_mask: Optional[torch.Tensor] = None,
            output_attentions: bool = False,
        """
        Parameters:
            query: torch.tensor(bs, seq_length, dim)
            key: torch.tensor(bs, seq_length, dim)
            value: torch.tensor(bs, seq_length, dim)
            mask: torch.tensor(bs, seq_length)

        Returns:
            weights: torch.tensor(bs, n_heads, seq_length, seq_length) Attention weights
            context: torch.tensor(bs, seq_length, dim) Contextualized layer.
                    Optional: only if `output_attentions=True`
        """
        # 获取输入张量的尺寸信息
        bs, q_length, dim = query.size()
        k_length = key.size(1)

        # 计算每个头部的维度
        dim_per_head = self.dim // self.n_heads

        # 创建用于遮罩的形状
        mask_reshp = (bs, 1, 1, k_length)

        def shape(x: torch.Tensor) -> torch.Tensor:
            """将输入张量重塑以便多头注意力"""
            return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)

        def unshape(x: torch.Tensor) -> torch.Tensor:
            """将多头注意力结果合并"""
            return x.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * dim_per_head)

        # 对查询、键和值进行线性变换并重塑以多头注意力的形式
        q = shape(self.q_lin(query))  # (bs, n_heads, q_length, dim_per_head)
        k = shape(self.k_lin(key))    # (bs, n_heads, k_length, dim_per_head)
        v = shape(self.v_lin(value))  # (bs, n_heads, k_length, dim_per_head)

        # 缩放查询向量以增强稳定性
        q = q / math.sqrt(dim_per_head)  # (bs, n_heads, q_length, dim_per_head)

        # 计算注意力分数
        scores = torch.matmul(q, k.transpose(2, 3))  # (bs, n_heads, q_length, k_length)

        # 创建注意力遮罩
        mask = (mask == 0).view(mask_reshp).expand_as(scores)  # (bs, n_heads, q_length, k_length)

        # 对于遮罩中的位置，用极小值填充注意力分数
        scores = scores.masked_fill(mask, torch.tensor(torch.finfo(scores.dtype).min))  # (bs, n_heads, q_length, k_length)

        # 计算注意力权重
        weights = nn.functional.softmax(scores, dim=-1)  # (bs, n_heads, q_length, k_length)
        weights = self.dropout(weights)  # (bs, n_heads, q_length, k_length)

        # 如果有头部遮罩，将其应用到权重上
        if head_mask is not None:
            weights = weights * head_mask

        # 计算上下文向量
        context = torch.matmul(weights, v)  # (bs, n_heads, q_length, dim_per_head)

        # 将多头注意力结果重新合并为单一张量
        context = unshape(context)  # (bs, q_length, dim)

        # 将上下文向量通过输出线性层得到最终的上下文表示
        context = self.out_lin(context)  # (bs, q_length, dim)

        # 如果需要输出注意力权重，则返回上下文和注意力权重；否则，只返回上下文
        if output_attentions:
            return (context, weights)
        else:
            return (context,)
# DistilBertFlashAttention2 类继承自 MultiHeadSelfAttention，用于实现 DistilBERT 的闪存注意力模块。
# 这个模块保留了 MultiHeadSelfAttention 的权重。唯一需要改变的是前向传播的实现，需要正确调用闪存注意力的公共 API，并处理输入中可能存在的填充标记。

# 从 transformers.models.llama.modeling_llama.LlamaFlashAttention2.__init__ 复制而来
def __init__(self, *args, **kwargs):
    # 调用父类 MultiHeadSelfAttention 的构造函数
    super().__init__(*args, **kwargs)

    # TODO: 一旦 Flash Attention for RoCm 升级到 2.1 版本后应该移除这段注释。
    # flash_attn<2.1 生成左上角对齐的因果掩码，而这里需要的是右下角对齐，flash_attn>=2.1 已经将右下角对齐作为默认行为。
    # 这个属性用于处理这种差异。参考：https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0。
    # 需要注意的是，对于 flash_attn<2.1，除了 q_seqlen == 1 的情况外，使用 q_seqlen != k_seqlen 会产生错误的掩码（左上角）。
    self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

def forward(
    self,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor,
    head_mask: Optional[torch.Tensor] = None,
    output_attentions: bool = False,
        """
        Parameters:
            query: torch.tensor(bs, seq_length, dim)  # 输入查询张量，形状为(batch_size, seq_length, dim)
            key: torch.tensor(bs, seq_length, dim)    # 键张量，形状为(batch_size, seq_length, dim)
            value: torch.tensor(bs, seq_length, dim)  # 值张量，形状为(batch_size, seq_length, dim)
            mask: torch.tensor(bs, seq_length)         # 掩码张量，形状为(batch_size, seq_length)

        Returns:
            weights: torch.tensor(bs, n_heads, seq_length, seq_length)  # 注意力权重张量，形状为(batch_size, n_heads, seq_length, seq_length)
            context: torch.tensor(bs, seq_length, dim)  # 上下文化层，可选：仅在`output_attentions=True`时返回
        """
        # 获取批大小、查询长度和维度
        batch_size, q_length, dim = query.size()

        # 计算每个头部的维度
        dim_per_head = self.dim // self.n_heads

        def reshape(x: torch.Tensor) -> torch.Tensor:
            """将张量重新形状为(batch_size, seq_length, n_heads, dim_per_head)"""
            return x.view(batch_size, -1, self.n_heads, dim_per_head)

        # Flash Attention 要求输入的形状为 batch_size x seq_length x head_dim x hidden_dim
        # 对查询、键和值进行线性变换并按头部进行重塑
        query_states = reshape(self.q_lin(query))  # 查询状态
        key_states = reshape(self.k_lin(key))      # 键状态
        value_states = reshape(self.v_lin(value))  # 值状态

        # 注意力丢弃率，在训练时使用配置中的值，在评估时为0
        attn_dropout = self.config.attention_dropout if self.training else 0.0

        # 在 PEFT 中，通常会将层归一化转换为 float32，以提高训练稳定性
        # 因此输入的隐藏状态可能被静默转换为 float32。为确保一切按预期工作，需要将其转换回正确的数据类型
        if query_states.dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_lin.weight.dtype

            # 记录警告日志，指出可能的 float32 转换，并将查询、键、值状态转换回目标数据类型
            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        # 执行 Flash Attention 的前向传播
        attn_weights = self._flash_attention_forward(
            query_states, key_states, value_states, mask, q_length, dropout=attn_dropout
        )

        # 将注意力权重重塑为(batch_size, seq_length, n_heads * dim_per_head)
        attn_weights_reshaped = attn_weights.reshape(batch_size, q_length, self.n_heads * dim_per_head)

        # 对重塑后的注意力权重进行线性变换
        attn_output = self.out_lin(attn_weights_reshaped)

        # 如果需要输出注意力权重，则返回注意力输出和注意力权重；否则，仅返回注意力输出
        if output_attentions:
            return (attn_output, attn_weights)
        else:
            return (attn_output,)
    # Copied from transformers.models.llama.modeling_llama.LlamaFlashAttention2._flash_attention_forward with causal=True->causal=False
    def _flash_attention_forward(
        self, query_states, key_states, value_states, attention_mask, query_length, dropout=0.0, softmax_scale=None
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
        # Determine the causal mode for Flash Attention
        if not self._flash_attn_uses_top_left_mask:
            causal = self.is_causal  # Use the model's default causal setting
        else:
            # Special case for ROCm where `query_length != 1` affects causal mode
            causal = self.is_causal and query_length != 1

        # Check if there are any padding tokens in the input sequence
        if attention_mask is not None:
            batch_size = query_states.shape[0]
            # Unpad the input based on attention_mask
            query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(
                query_states, key_states, value_states, attention_mask, query_length
            )

            # Extract sequence lengths
            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

            # Perform Flash Attention with variable-length sequences
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

            # Pad the attention output based on the unpadded indices
            attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
        else:
            # Perform regular Flash Attention without considering padding
            attn_output = flash_attn_func(
                query_states, key_states, value_states, dropout, softmax_scale=softmax_scale, causal=causal
            )

        return attn_output
    # 从 transformers.models.llama.modeling_llama.LlamaFlashAttention2._upad_input 复制代码，将 num_heads 重命名为 n_heads
    def _upad_input(self, query_layer, key_layer, value_layer, attention_mask, query_length):
        # 使用 _get_unpad_data 函数获取解包数据的索引、当前序列长度和批次中的最大序列长度
        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)
        
        # 获取 key_layer 的形状信息：批次大小、键值序列长度、键值头数、头维度
        batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape
        
        # 根据 indices_k 对 key_layer 进行重新索引，重塑成 (batch_size * kv_seq_len, num_key_value_heads, head_dim) 的形状
        key_layer = index_first_axis(
            key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        
        # 根据 indices_k 对 value_layer 进行重新索引，重塑成 (batch_size * kv_seq_len, num_key_value_heads, head_dim) 的形状
        value_layer = index_first_axis(
            value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        
        # 根据 query_length 的不同情况进行处理
        if query_length == kv_seq_len:
            # 如果 query_length 等于 kv_seq_len，则根据 indices_k 对 query_layer 进行重新索引
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, self.n_heads, head_dim), indices_k
            )
            cu_seqlens_q = cu_seqlens_k  # 设置当前查询序列长度为 cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k  # 设置批次中的最大查询序列长度为 max_seqlen_in_batch_k
            indices_q = indices_k  # 设置查询的索引为 indices_k
        elif query_length == 1:
            # 如果 query_length 等于 1，则进行单个查询的处理
            max_seqlen_in_batch_q = 1  # 设置批次中的最大查询序列长度为 1
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            )  # 在 query_layer 设备上创建一个整数张量，表示当前查询序列长度
            indices_q = cu_seqlens_q[:-1]  # 设置查询的索引为 cu_seqlens_q 的前 n-1 项
            query_layer = query_layer.squeeze(1)  # 压缩 query_layer 的第一个维度
        else:
            # 否则，根据左填充的假设，截取 attention_mask 的后 query_length 列，然后调用 unpad_input 处理 query_layer 和 attention_mask
            attention_mask = attention_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, attention_mask)
        
        # 返回处理后的查询层、键层、值层、查询索引、当前序列长度元组、最大序列长度元组
        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )
class FFN(nn.Module):
    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.dropout = nn.Dropout(p=config.dropout)  # 使用给定的 dropout 概率创建 Dropout 层
        self.chunk_size_feed_forward = config.chunk_size_feed_forward  # 设置前向传播的块大小
        self.seq_len_dim = 1  # 序列长度的维度设为1
        self.lin1 = nn.Linear(in_features=config.dim, out_features=config.hidden_dim)  # 创建线性层 lin1
        self.lin2 = nn.Linear(in_features=config.hidden_dim, out_features=config.dim)  # 创建线性层 lin2
        self.activation = get_activation(config.activation)  # 根据配置获取激活函数

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return apply_chunking_to_forward(self.ff_chunk, self.chunk_size_feed_forward, self.seq_len_dim, input)

    def ff_chunk(self, input: torch.Tensor) -> torch.Tensor:
        x = self.lin1(input)  # 输入经过 lin1 线性层
        x = self.activation(x)  # 应用激活函数
        x = self.lin2(x)  # 经过 lin2 线性层
        x = self.dropout(x)  # 应用 dropout
        return x


DISTILBERT_ATTENTION_CLASSES = {
    "eager": MultiHeadSelfAttention,
    "flash_attention_2": DistilBertFlashAttention2,
}


class TransformerBlock(nn.Module):
    def __init__(self, config: PretrainedConfig):
        super().__init__()

        # Ensure number of heads evenly divides dimension
        if config.dim % config.n_heads != 0:
            raise ValueError(f"config.n_heads {config.n_heads} must divide config.dim {config.dim} evenly")

        self.attention = DISTILBERT_ATTENTION_CLASSES[config._attn_implementation](config)  # 根据配置选择注意力机制类别
        self.sa_layer_norm = nn.LayerNorm(normalized_shape=config.dim, eps=1e-12)  # 创建自注意力层的 LayerNorm

        self.ffn = FFN(config)  # 创建 FeedForward 网络
        self.output_layer_norm = nn.LayerNorm(normalized_shape=config.dim, eps=1e-12)  # 创建输出层的 LayerNorm

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        """
        Parameters:
            x: torch.tensor(bs, seq_length, dim)
            attn_mask: torch.tensor(bs, seq_length)

        Returns:
            sa_weights: torch.tensor(bs, n_heads, seq_length, seq_length) 注意力权重
            ffn_output: torch.tensor(bs, seq_length, dim) Transformer 块的输出
        """
        # Self-Attention 自注意力机制
        sa_output = self.attention(
            query=x,
            key=x,
            value=x,
            mask=attn_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
        )
        if output_attentions:
            sa_output, sa_weights = sa_output  # (bs, seq_length, dim), (bs, n_heads, seq_length, seq_length)
        else:  # 处理输出注意力权重或隐藏状态的情况，返回元组的情况
            if type(sa_output) != tuple:
                raise TypeError(f"sa_output must be a tuple but it is {type(sa_output)} type")

            sa_output = sa_output[0]  # 仅获取自注意力输出

        sa_output = self.sa_layer_norm(sa_output + x)  # Self-Attention 后的 Layer Normalization

        # Feed Forward Network 前馈神经网络
        ffn_output = self.ffn(sa_output)  # (bs, seq_length, dim)
        ffn_output: torch.Tensor = self.output_layer_norm(ffn_output + sa_output)  # 前馈网络后的 Layer Normalization

        output = (ffn_output,)  # 输出结果元组

        if output_attentions:
            output = (sa_weights,) + output  # 如果需要输出注意力权重，将注意力权重加入输出元组

        return output
class Transformer(nn.Module):
    # Transformer 类，继承自 nn.Module
    def __init__(self, config: PretrainedConfig):
        # 初始化函数，接受一个预训练配置对象 config
        super().__init__()
        # 调用父类的初始化函数
        self.n_layers = config.n_layers
        # 设置 Transformer 的层数为 config 中指定的层数
        self.layer = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        # 创建 nn.ModuleList，其中包含 config.n_layers 个 TransformerBlock 模块
        self.gradient_checkpointing = False
        # 设置梯度检查点为 False，默认不启用梯度检查点功能

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: Optional[bool] = None,
    ) -> Union[BaseModelOutput, Tuple[torch.Tensor, ...]]:  # docstyle-ignore
        """
        Parameters:
            x: torch.tensor(bs, seq_length, dim) Input sequence embedded.
                输入的嵌入序列张量，形状为 (bs, seq_length, dim)
            attn_mask: torch.tensor(bs, seq_length) Attention mask on the sequence.
                序列的注意力掩码张量，形状为 (bs, seq_length)

        Returns:
            hidden_state: torch.tensor(bs, seq_length, dim) Sequence of hidden states in the last (top)
                layer all_hidden_states: Tuple[torch.tensor(bs, seq_length, dim)]
                最后（顶部）层的隐藏状态序列，形状为 (bs, seq_length, dim)
                一个包含每层隐藏状态的元组，形状为 (n_layers, bs, seq_length, dim)
                可选项：仅在 output_hidden_states=True 时返回
            all_attentions: Tuple[torch.tensor(bs, n_heads, seq_length, seq_length)]
                每层的注意力权重张量的元组，形状为 (n_layers, bs, n_heads, seq_length, seq_length)
                可选项：仅在 output_attentions=True 时返回
        """
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        hidden_state = x
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_state,)

            if self.gradient_checkpointing and self.training:
                # 如果启用了梯度检查点且处于训练模式，则使用梯度检查点函数处理
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_state,
                    attn_mask,
                    head_mask[i],
                    output_attentions,
                )
            else:
                # 否则，直接调用层模块处理输入
                layer_outputs = layer_module(
                    hidden_state,
                    attn_mask,
                    head_mask[i],
                    output_attentions,
                )

            hidden_state = layer_outputs[-1]

            if output_attentions:
                if len(layer_outputs) != 2:
                    raise ValueError(f"The length of the layer_outputs should be 2, but it is {len(layer_outputs)}")

                attentions = layer_outputs[0]
                all_attentions = all_attentions + (attentions,)
            else:
                if len(layer_outputs) != 1:
                    raise ValueError(f"The length of the layer_outputs should be 1, but it is {len(layer_outputs)}")

        # 添加最后一层的隐藏状态
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_state,)

        # 如果 return_dict 为 False，则返回所有非 None 的值的元组
        if not return_dict:
            return tuple(v for v in [hidden_state, all_hidden_states, all_attentions] if v is not None)
        # 否则，返回 BaseModelOutput 对象，包含最后的隐藏状态、所有隐藏状态和所有注意力权重
        return BaseModelOutput(
            last_hidden_state=hidden_state, hidden_states=all_hidden_states, attentions=all_attentions
        )
# INTERFACE FOR ENCODER AND TASK SPECIFIC MODEL #
class DistilBertPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = DistilBertConfig  # 使用DistilBertConfig配置类来配置模型
    load_tf_weights = None  # 用于加载TensorFlow权重的标志，暂时未定义
    base_model_prefix = "distilbert"  # 基础模型的名称前缀
    supports_gradient_checkpointing = True  # 支持梯度检查点的标志
    _supports_flash_attn_2 = True  # 特殊支持的特性标志，用于Flash Attention机制

    def _init_weights(self, module: nn.Module):
        """Initialize the weights."""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            # 使用正态分布初始化线性层的权重，标准差为config.initializer_range
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()  # 如果有偏置，初始化为零
        elif isinstance(module, nn.Embedding):
            # 使用正态分布初始化嵌入层的权重，标准差为config.initializer_range
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                # 如果指定了padding_idx，将对应的嵌入向量初始化为零
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            # 初始化LayerNorm层的偏置为零，权重为1.0
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


DISTILBERT_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`DistilBertConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

DISTILBERT_INPUTS_DOCSTRING = r"""
    Describes the inputs to the DistilBERT model and how to prepare them.
"""
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
        
        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
            
            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        
        inputs_embeds (`torch.FloatTensor` of shape `({0}, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""
Define a DistilBERT model for encoding text using transformer architecture.

@add_start_docstrings(
    "The bare DistilBERT encoder/transformer outputting raw hidden-states without any specific head on top.",
    DISTILBERT_START_DOCSTRING,
)
class DistilBertModel(DistilBertPreTrainedModel):
    def __init__(self, config: PretrainedConfig):
        super().__init__(config)

        self.embeddings = Embeddings(config)  # Embeddings
        self.transformer = Transformer(config)  # Encoder
        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"

        # Initialize weights and apply final processing
        self.post_init()

    def get_position_embeddings(self) -> nn.Embedding:
        """
        Returns the position embeddings
        """
        return self.embeddings.position_embeddings
"""
    def resize_position_embeddings(self, new_num_position_embeddings: int):
        """
        Resizes position embeddings of the model if `new_num_position_embeddings != config.max_position_embeddings`.

        Arguments:
            new_num_position_embeddings (`int`):
                The number of new position embedding matrix. If position embeddings are learned, increasing the size
                will add newly initialized vectors at the end, whereas reducing the size will remove vectors from the
                end. If position embeddings are not learned (*e.g.* sinusoidal position embeddings), increasing the
                size will add correct vectors at the end following the position encoding algorithm, whereas reducing
                the size will remove vectors from the end.
        """
        # 计算新旧位置嵌入矩阵长度之差
        num_position_embeds_diff = new_num_position_embeddings - self.config.max_position_embeddings

        # 如果长度没有变化，则无需调整
        if num_position_embeds_diff == 0:
            return

        # 记录信息：设置 `config.max_position_embeddings` 的新值
        logger.info(f"Setting `config.max_position_embeddings={new_num_position_embeddings}`...")
        self.config.max_position_embeddings = new_num_position_embeddings

        # 备份旧的位置嵌入权重
        old_position_embeddings_weight = self.embeddings.position_embeddings.weight.clone()

        # 根据新的 `max_position_embeddings` 大小重新创建位置嵌入层
        self.embeddings.position_embeddings = nn.Embedding(self.config.max_position_embeddings, self.config.dim)

        # 如果使用正弦位置嵌入，根据新的大小重新创建正弦位置嵌入
        if self.config.sinusoidal_pos_embds:
            create_sinusoidal_embeddings(
                n_pos=self.config.max_position_embeddings, dim=self.config.dim, out=self.position_embeddings.weight
            )
        else:
            with torch.no_grad():
                # 根据位置嵌入大小的变化，重新设置位置嵌入权重
                if num_position_embeds_diff > 0:
                    self.embeddings.position_embeddings.weight[:-num_position_embeds_diff] = nn.Parameter(
                        old_position_embeddings_weight
                    )
                else:
                    self.embeddings.position_embeddings.weight = nn.Parameter(
                        old_position_embeddings_weight[:num_position_embeds_diff]
                    )

        # 将更新后的位置嵌入层移动到正确的设备上
        self.embeddings.position_embeddings.to(self.device)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, new_embeddings: nn.Embedding):
        # 设置输入词嵌入层的新权重
        self.embeddings.word_embeddings = new_embeddings

    def _prune_heads(self, heads_to_prune: Dict[int, List[List[int]]]):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        # 遍历需要修剪的层和头部，进行修剪操作
        for layer, heads in heads_to_prune.items():
            self.transformer.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(DISTILBERT_INPUTS_DOCSTRING.format("batch_size, num_choices"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,  # 添加代码示例文档字符串，指定检查点为给定的文档检查点
        output_type=BaseModelOutput,  # 指定输出类型为BaseModelOutput类
        config_class=_CONFIG_FOR_DOC,  # 指定配置类为给定的配置类
    )
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,  # 输入的token IDs张量，可选
        attention_mask: Optional[torch.Tensor] = None,  # 注意力掩码张量，可选
        head_mask: Optional[torch.Tensor] = None,  # 头部掩码张量，可选
        inputs_embeds: Optional[torch.Tensor] = None,  # 嵌入输入张量，可选
        output_attentions: Optional[bool] = None,  # 是否输出注意力权重，可选
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态，可选
        return_dict: Optional[bool] = None,  # 是否返回字典格式的输出，可选
    ) -> Union[BaseModelOutput, Tuple[torch.Tensor, ...]]:  # 返回值可以是BaseModelOutput或者张量元组

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 如果未指定output_attentions，则使用self.config中的默认值

        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 如果未指定output_hidden_states，则使用self.config中的默认值

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 如果未指定return_dict，则使用self.config中的默认值

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")
        # 检查输入参数的有效性，确保只能同时指定input_ids或inputs_embeds，并获取输入的形状

        device = input_ids.device if input_ids is not None else inputs_embeds.device
        # 获取输入所在的设备

        # Prepare head mask if needed
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
        # 如果需要，准备头部掩码

        embeddings = self.embeddings(input_ids, inputs_embeds)  # (bs, seq_length, dim)
        # 生成输入的嵌入表示

        if self._use_flash_attention_2:
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        else:
            if attention_mask is None:
                attention_mask = torch.ones(input_shape, device=device)  # (bs, seq_length)
        # 根据self._use_flash_attention_2的条件设置注意力掩码，如果未提供则使用全1的默认掩码

        return self.transformer(
            x=embeddings,
            attn_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # 调用transformer模块进行前向传播，传入嵌入表示、注意力掩码、头部掩码等参数，并返回结果
# 使用装饰器添加文档字符串，描述此类是在DistilBert模型基础上增加了遮盖语言建模头部的模型
@add_start_docstrings(
    """DistilBert Model with a `masked language modeling` head on top.""",
    DISTILBERT_START_DOCSTRING,
)
# 定义DistilBertForMaskedLM类，继承自DistilBertPreTrainedModel
class DistilBertForMaskedLM(DistilBertPreTrainedModel):
    # 定义_tied_weights_keys属性，指定需要绑定权重的键名
    _tied_weights_keys = ["vocab_projector.weight"]

    # 初始化函数，接收一个PretrainedConfig类型的config对象作为参数
    def __init__(self, config: PretrainedConfig):
        # 调用父类的初始化函数
        super().__init__(config)

        # 根据配置中指定的激活函数名称，获取对应的激活函数
        self.activation = get_activation(config.activation)

        # 创建DistilBertModel模型对象，并赋值给self.distilbert
        self.distilbert = DistilBertModel(config)
        
        # 创建一个线性层，用于词汇转换，输入和输出维度均为config.dim
        self.vocab_transform = nn.Linear(config.dim, config.dim)
        
        # 创建一个LayerNorm层，用于词汇层的归一化，输入维度为config.dim
        self.vocab_layer_norm = nn.LayerNorm(config.dim, eps=1e-12)
        
        # 创建一个线性层，用于将模型的输出映射到词汇表大小的向量
        self.vocab_projector = nn.Linear(config.dim, config.vocab_size)

        # 执行初始化权重操作和最终处理
        self.post_init()

        # 定义模型的损失函数为交叉熵损失函数
        self.mlm_loss_fct = nn.CrossEntropyLoss()

    # 获取位置嵌入的方法，返回DistilBert模型中的位置嵌入
    def get_position_embeddings(self) -> nn.Embedding:
        """
        Returns the position embeddings
        """
        return self.distilbert.get_position_embeddings()

    # 调整位置嵌入的方法，根据新的位置嵌入数量调整模型的位置嵌入矩阵
    def resize_position_embeddings(self, new_num_position_embeddings: int):
        """
        Resizes position embeddings of the model if `new_num_position_embeddings != config.max_position_embeddings`.

        Arguments:
            new_num_position_embeddings (`int`):
                The number of new position embedding matrix. If position embeddings are learned, increasing the size
                will add newly initialized vectors at the end, whereas reducing the size will remove vectors from the
                end. If position embeddings are not learned (*e.g.* sinusoidal position embeddings), increasing the
                size will add correct vectors at the end following the position encoding algorithm, whereas reducing
                the size will remove vectors from the end.
        """
        self.distilbert.resize_position_embeddings(new_num_position_embeddings)

    # 获取输出嵌入的方法，返回词汇投影层对象
    def get_output_embeddings(self) -> nn.Module:
        return self.vocab_projector

    # 设置输出嵌入的方法，用新的嵌入层对象替换词汇投影层
    def set_output_embeddings(self, new_embeddings: nn.Module):
        self.vocab_projector = new_embeddings

    # 使用装饰器添加文档字符串到模型前向传播方法，描述输入参数和输出类型
    @add_start_docstrings_to_model_forward(DISTILBERT_INPUTS_DOCSTRING.format("batch_size, num_choices"))
    # 使用代码示例装饰器添加文档字符串，提供模型前向传播的示例和其他相关信息
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=MaskedLMOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    # 模型的前向传播方法，接收多个输入参数，返回一个输出对象或字典
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        ) -> Union[MaskedLMOutput, Tuple[torch.Tensor, ...]]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        """
        # 根据函数声明，定义了输入参数和返回类型，包括可选的标签用于计算MLM损失
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用DistilBERT模型，获取输出结果
        dlbrt_output = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # 提取DistilBERT模型的隐藏状态
        hidden_states = dlbrt_output[0]  # (bs, seq_length, dim)
        # 将隐藏状态转换为预测的对数概率
        prediction_logits = self.vocab_transform(hidden_states)  # (bs, seq_length, dim)
        # 应用激活函数到预测的对数概率
        prediction_logits = self.activation(prediction_logits)  # (bs, seq_length, dim)
        # 对预测的对数概率进行层归一化
        prediction_logits = self.vocab_layer_norm(prediction_logits)  # (bs, seq_length, dim)
        # 使用投影层将预测的对数概率映射到词汇表大小的空间
        prediction_logits = self.vocab_projector(prediction_logits)  # (bs, seq_length, vocab_size)

        mlm_loss = None
        # 如果提供了标签，计算MLM损失
        if labels is not None:
            mlm_loss = self.mlm_loss_fct(prediction_logits.view(-1, prediction_logits.size(-1)), labels.view(-1))

        # 如果不要求返回字典格式的输出，构建输出元组
        if not return_dict:
            output = (prediction_logits,) + dlbrt_output[1:]
            return ((mlm_loss,) + output) if mlm_loss is not None else output

        # 返回MaskedLMOutput对象，包括损失、预测的对数概率、隐藏状态和注意力权重
        return MaskedLMOutput(
            loss=mlm_loss,
            logits=prediction_logits,
            hidden_states=dlbrt_output.hidden_states,
            attentions=dlbrt_output.attentions,
        )
"""
DistilBert模型转换器，顶部带有序列分类/回归头（即顶部的线性层，用于池化输出），例如用于GLUE任务。
"""
@add_start_docstrings(
    """
    DistilBert Model transformer with a sequence classification/regression head on top (a linear layer on top of the
    pooled output) e.g. for GLUE tasks.
    """,
    DISTILBERT_START_DOCSTRING,
)
class DistilBertForSequenceClassification(DistilBertPreTrainedModel):
    def __init__(self, config: PretrainedConfig):
        """
        初始化方法，配置DistilBert序列分类/回归模型。

        Arguments:
            config (:class:`~transformers.PretrainedConfig`):
                包含模型配置信息的预训练配置对象。
        """
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        # DistilBert模型实例化
        self.distilbert = DistilBertModel(config)
        # 预分类器，线性层
        self.pre_classifier = nn.Linear(config.dim, config.dim)
        # 分类器，线性层
        self.classifier = nn.Linear(config.dim, config.num_labels)
        # Dropout层
        self.dropout = nn.Dropout(config.seq_classif_dropout)

        # 初始化权重并应用最终处理
        self.post_init()

    def get_position_embeddings(self) -> nn.Embedding:
        """
        返回位置嵌入
        """
        return self.distilbert.get_position_embeddings()

    def resize_position_embeddings(self, new_num_position_embeddings: int):
        """
        如果`new_num_position_embeddings != config.max_position_embeddings`，调整模型的位置嵌入。

        Arguments:
            new_num_position_embeddings (`int`):
                新的位置嵌入矩阵数量。如果位置嵌入是学习的，则增加大小将在末尾添加新初始化的向量，
                而减小大小将从末尾删除向量。如果位置嵌入不是学习的（例如正弦位置嵌入），
                增加大小将按照位置编码算法在末尾添加正确的向量，而减小大小将从末尾删除向量。
        """
        self.distilbert.resize_position_embeddings(new_num_position_embeddings)

    @add_start_docstrings_to_model_forward(DISTILBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=SequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        ):
        """
        前向传播方法，执行DistilBert序列分类/回归模型的计算。

        Arguments:
            input_ids (:obj:`torch.Tensor`, optional):
                输入序列的token IDs张量。如果不提供`inputs_embeds`，则必须提供此项。
            attention_mask (:obj:`torch.Tensor`, optional):
                注意力遮罩张量，指示哪些tokens应被忽略。默认为`None`。
            head_mask (:obj:`torch.Tensor`, optional):
                多头注意力层的遮罩张量，用于控制每个注意力头的输出。默认为`None`。
            inputs_embeds (:obj:`torch.Tensor`, optional):
                直接传入模型的嵌入张量，而不是输入IDs。如果提供了此项，则`input_ids`应为`None`。
            labels (:obj:`torch.LongTensor`, optional):
                标签张量，用于模型训练的目标值。默认为`None`。
            output_attentions (:obj:`bool`, optional):
                是否返回所有注意力权重。默认为`None`。
            output_hidden_states (:obj:`bool`, optional):
                是否返回所有隐藏状态。默认为`None`。
            return_dict (:obj:`bool`, optional):
                是否返回字典类型的输出。默认为`None`。

        Returns:
            :class:`~transformers.modeling_outputs.SequenceClassifierOutput`:
                包含模型输出的命名元组。
        """
        # 实现前向传播逻辑，计算输出结果
        # （具体实现详见模型具体代码，此处略）
    ) -> Union[SequenceClassifierOutput, Tuple[torch.Tensor, ...]]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        # 根据是否需要返回字典来确定返回值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 将输入传递给DistilBERT模型，获取输出
        distilbert_output = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # 提取DistilBERT输出的隐藏状态
        hidden_state = distilbert_output[0]  # (bs, seq_len, dim)
        # 提取池化后的输出，取每个序列的第一个标记
        pooled_output = hidden_state[:, 0]  # (bs, dim)
        # 应用预分类器（一个线性层）到池化输出
        pooled_output = self.pre_classifier(pooled_output)  # (bs, dim)
        # 应用ReLU激活函数到预分类器输出
        pooled_output = nn.ReLU()(pooled_output)  # (bs, dim)
        # 应用dropout操作到ReLU输出
        pooled_output = self.dropout(pooled_output)  # (bs, dim)
        # 将池化后的输出传递给分类器，得到logits
        logits = self.classifier(pooled_output)  # (bs, num_labels)

        # 初始化损失值为None
        loss = None
        # 如果有标签输入
        if labels is not None:
            # 如果问题类型尚未确定
            if self.config.problem_type is None:
                # 根据标签数量确定问题类型为回归或分类
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            # 根据问题类型计算损失值
            if self.config.problem_type == "regression":
                # 使用均方误差损失函数
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                # 使用交叉熵损失函数
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                # 使用带logits的二元交叉熵损失函数
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        # 如果不需要返回字典，则只返回logits和可能的其他输出
        if not return_dict:
            output = (logits,) + distilbert_output[1:]
            return ((loss,) + output) if loss is not None else output

        # 如果需要返回字典，则返回包含损失、logits和其他输出的SequenceClassifierOutput对象
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=distilbert_output.hidden_states,
            attentions=distilbert_output.attentions,
        )
@add_start_docstrings(
    """
    DistilBert Model with a span classification head on top for extractive question-answering tasks like SQuAD (a
    linear layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    DISTILBERT_START_DOCSTRING,
)



class DistilBertForQuestionAnswering(DistilBertPreTrainedModel):
    def __init__(self, config: PretrainedConfig):
        super().__init__(config)

        # 初始化 DistilBERT 模型
        self.distilbert = DistilBertModel(config)
        # 线性层用于输出 span 开始和结束的逻辑回归
        self.qa_outputs = nn.Linear(config.dim, config.num_labels)
        # 检查标签数是否为2，否则引发错误
        if config.num_labels != 2:
            raise ValueError(f"config.num_labels should be 2, but it is {config.num_labels}")

        # Dropout 层
        self.dropout = nn.Dropout(config.qa_dropout)

        # 初始化权重并进行最终处理
        self.post_init()

    def get_position_embeddings(self) -> nn.Embedding:
        """
        Returns the position embeddings
        """
        return self.distilbert.get_position_embeddings()

    def resize_position_embeddings(self, new_num_position_embeddings: int):
        """
        Resizes position embeddings of the model if `new_num_position_embeddings != config.max_position_embeddings`.

        Arguments:
            new_num_position_embeddings (`int`):
                The number of new position embedding matrix. If position embeddings are learned, increasing the size
                will add newly initialized vectors at the end, whereas reducing the size will remove vectors from the
                end. If position embeddings are not learned (*e.g.* sinusoidal position embeddings), increasing the
                size will add correct vectors at the end following the position encoding algorithm, whereas reducing
                the size will remove vectors from the end.
        """
        self.distilbert.resize_position_embeddings(new_num_position_embeddings)

    @add_start_docstrings_to_model_forward(DISTILBERT_INPUTS_DOCSTRING.format("batch_size, num_choices"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=QuestionAnsweringModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        start_positions: Optional[torch.Tensor] = None,
        end_positions: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
):



@add_start_docstrings(
    """
    DistilBert Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g.
    for Named-Entity-Recognition (NER) tasks.
    """,
    DISTILBERT_START_DOCSTRING,
)
class DistilBertForTokenClassification(DistilBertPreTrainedModel):
    def __init__(self, config: PretrainedConfig):
        super().__init__(config)
        self.num_labels = config.num_labels  # 从配置中获取标签数量

        self.distilbert = DistilBertModel(config)  # 初始化 DistilBERT 模型
        self.dropout = nn.Dropout(config.dropout)  # 根据配置添加 dropout 层
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)  # 添加线性分类器

        # 初始化权重并应用最终处理
        self.post_init()

    def get_position_embeddings(self) -> nn.Embedding:
        """
        Returns the position embeddings
        """
        return self.distilbert.get_position_embeddings()  # 返回 DistilBERT 模型的位置嵌入

    def resize_position_embeddings(self, new_num_position_embeddings: int):
        """
        Resizes position embeddings of the model if `new_num_position_embeddings != config.max_position_embeddings`.

        Arguments:
            new_num_position_embeddings (`int`):
                The number of new position embedding matrix. If position embeddings are learned, increasing the size
                will add newly initialized vectors at the end, whereas reducing the size will remove vectors from the
                end. If position embeddings are not learned (*e.g.* sinusoidal position embeddings), increasing the
                size will add correct vectors at the end following the position encoding algorithm, whereas reducing
                the size will remove vectors from the end.
        """
        self.distilbert.resize_position_embeddings(new_num_position_embeddings)  # 调整 DistilBERT 模型的位置嵌入

    @add_start_docstrings_to_model_forward(DISTILBERT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TokenClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        """
        Defines how inputs are processed through the model layers.

        Arguments:
            input_ids (`torch.Tensor`, optional):
                Indices of input sequence tokens in the vocabulary.
            attention_mask (`torch.Tensor`, optional):
                Mask to avoid performing attention on padding token indices.
            head_mask (`torch.Tensor`, optional):
                Mask to nullify selected heads of the self-attention modules.
            inputs_embeds (`torch.Tensor`, optional):
                Embedded representation of input tokens.
            labels (`torch.LongTensor`, optional):
                Labels for computing the token classification loss.
            output_attentions (`bool`, optional):
                Whether to return attentions tensors.
            output_hidden_states (`bool`, optional):
                Whether to return hidden states.
            return_dict (`bool`, optional):
                Whether to return a dictionary instead of a tuple of outputs.

        Returns:
            Output of the model, usually a tuple with various elements depending on the configuration.
        """
    ) -> Union[TokenClassifierOutput, Tuple[torch.Tensor, ...]]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        # 根据函数定义，返回值可以是 TokenClassifierOutput 对象或者元组形式的 Tensor
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 使用 DistilBERT 模型进行前向传播，获取输出结果
        outputs = self.distilbert(
            input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 从输出结果中提取序列输出
        sequence_output = outputs[0]

        # 应用 dropout 操作
        sequence_output = self.dropout(sequence_output)
        
        # 使用分类器对序列输出进行分类，得到分类 logits
        logits = self.classifier(sequence_output)

        # 初始化损失为 None
        loss = None

        # 如果有标签输入，则计算交叉熵损失
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        # 如果不需要返回字典，则按元组形式返回输出结果
        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        # 如果需要返回字典形式的输出，构造 TokenClassifierOutput 对象返回
        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
"""
DistilBert Model with a multiple choice classification head on top (a linear layer on top of the pooled output and
a softmax) e.g. for RocStories/SWAG tasks.
"""
# 导入必要的库和模块
import torch
import torch.nn as nn
from .configuration_distilbert import DistilBertConfig
from .modeling_distilbert import DistilBertModel, DistilBertPreTrainedModel
from .file_utils import add_start_docstrings, add_start_docstrings_to_model_forward, replace_return_docstrings
from typing import Optional
from transformers.file_utils import ModelOutput, PretrainedConfig

# 定义 DistilBertForMultipleChoice 类，继承自 DistilBertPreTrainedModel
@add_start_docstrings(
    """
    DistilBert Model with a multiple choice classification head on top (a linear layer on top of the pooled output and
    a softmax) e.g. for RocStories/SWAG tasks.
    """,
    DISTILBERT_START_DOCSTRING,
)
class DistilBertForMultipleChoice(DistilBertPreTrainedModel):
    
    def __init__(self, config: PretrainedConfig):
        super().__init__(config)
        
        # 初始化 DistilBert 模型
        self.distilbert = DistilBertModel(config)
        
        # 多选分类任务的预分类器
        self.pre_classifier = nn.Linear(config.dim, config.dim)
        
        # 用于二分类的线性层
        self.classifier = nn.Linear(config.dim, 1)
        
        # Dropout 层，用于防止过拟合
        self.dropout = nn.Dropout(config.seq_classif_dropout)

        # 初始化权重并应用最终处理
        self.post_init()

    def get_position_embeddings(self) -> nn.Embedding:
        """
        Returns the position embeddings
        """
        # 调用 DistilBertModel 的方法获取位置嵌入
        return self.distilbert.get_position_embeddings()

    def resize_position_embeddings(self, new_num_position_embeddings: int):
        """
        Resizes position embeddings of the model if `new_num_position_embeddings != config.max_position_embeddings`.

        Arguments:
            new_num_position_embeddings (`int`)
                The number of new position embeddings. If position embeddings are learned, increasing the size will add
                newly initialized vectors at the end, whereas reducing the size will remove vectors from the end. If
                position embeddings are not learned (*e.g.* sinusoidal position embeddings), increasing the size will
                add correct vectors at the end following the position encoding algorithm, whereas reducing the size
                will remove vectors from the end.
        """
        # 调整 DistilBertModel 的位置嵌入
        self.distilbert.resize_position_embeddings(new_num_position_embeddings)

    @add_start_docstrings_to_model_forward(
        DISTILBERT_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length")
    )
    @replace_return_docstrings(output_type=MultipleChoiceModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> ModelOutput:
        """
        Forward pass for DistilBertForMultipleChoice.
        
        Args:
            input_ids (Optional[torch.Tensor], optional):
                Indices of input sequence tokens in the vocabulary.
            attention_mask (Optional[torch.Tensor], optional):
                Mask to avoid performing attention on padding token indices.
            head_mask (Optional[torch.Tensor], optional):
                Mask to nullify selected heads of the self-attention modules.
            inputs_embeds (Optional[torch.Tensor], optional):
                Optionally, instead of passing `input_ids`, you can directly pass an embedded representation.
            labels (Optional[torch.LongTensor], optional):
                Labels for computing the multiple choice classification loss.
            output_attentions (Optional[bool], optional):
                Whether to return attentions weights.
            output_hidden_states (Optional[bool], optional):
                Whether to return hidden states.
            return_dict (Optional[bool], optional):
                Whether to return a dictionary instead of a tuple.
            **kwargs:
                Additional keyword arguments for the DistilBertModel forward method.
        
        Returns:
            ModelOutput: A namedtuple with the model outputs: last_hidden_state, (optional) hidden_states, (optional) attentions.
        """
        # 调用 DistilBertModel 的 forward 方法进行前向传播
        return self.distilbert(
            input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )
```