# `.\transformers\models\reformer\modeling_reformer.py`

```py
# 定义一个稳定的 argsort 函数，以确保 torch.argsort 函数的稳定性
def _stable_argsort(vector, dim):
    # 创建一个范围在 [0, vector.shape[dim]) 的张量，用于缩放
    scale_offset = torch.arange(vector.shape[dim], device=vector.device).view(1, 1, -1)
    # 扩展张量的维度以匹配输入向量的形状
    scale_offset = scale_offset.expand(vector.shape)
    # 将输入的向量 vector 在指定维度 dim 上进行缩放操作
    # 缩放操作包括:
    # 1. 将 vector 在维度 dim 上的大小乘以 vector.shape[dim]
    # 2. 将缩放后的向量与 scale_offset % vector.shape[dim] 相加
    # 3. 对缩放后的向量进行排序，返回排序后的索引
        scaled_vector = vector.shape[dim] * vector + (scale_offset % vector.shape[dim])
        return torch.argsort(scaled_vector, dim=dim)
# 根据配置获取最小公倍数块长度
def _get_least_common_mult_chunk_len(config):
    # 获取注意力层类型
    attn_types = config.attn_layers
    # 转换为集合
    attn_types_set = set(attn_types)
    # 如果只有一个注意力类型且为 "lsh"，返回配置中的 LSH 注意力块长度
    if len(attn_types_set) == 1 and attn_types[0] == "lsh":
        return config.lsh_attn_chunk_length
    # 如果只有一个注意力类型且为 "local"，返回配置中的 Local 注意力块长度
    elif len(attn_types_set) == 1 and attn_types[0] == "local":
        return config.local_attn_chunk_length
    # 如果有两个不同的注意力类型且分别为 "lsh" 和 "local"，返回它们的最小公倍数
    elif len(attn_types_set) == 2 and attn_types_set == {"lsh", "local"}:
        return np.lcm(config.lsh_attn_chunk_length, config.local_attn_chunk_length)
    else:
        # 抛出未实现错误
        raise NotImplementedError(
            f"Only attn layer types 'lsh' and 'local' exist, but `config.attn_layers`: {config.attn_layers}. Select "
            "attn layer types from ['lsh', 'local'] only."
        )

# 根据配置获取最小块长度
def _get_min_chunk_len(config):
    # 获取注意力层类型
    attn_types = config.attn_layers
    # 转换为集合
    attn_types_set = set(attn_types)
    # 如果只有一个注意力类型且为 "lsh"，返回配置中的 LSH 注意力块长度
    if len(attn_types_set) == 1 and attn_types[0] == "lsh":
        return config.lsh_attn_chunk_length
    # 如果只有一个注意力类型且为 "local"，返回配置中的 Local 注意力块长度
    elif len(attn_types_set) == 1 and attn_types[0] == "local":
        return config.local_attn_chunk_length
    # 如果有两个不同的注意力类型且分别为 "lsh" 和 "local"，返回它们的最小值
    elif len(attn_types_set) == 2 and attn_types_set == {"lsh", "local"}:
        return min(config.lsh_attn_chunk_length, config.local_attn_chunk_length)
    else:
        # 抛出未实现错误
        raise NotImplementedError(
            f"Only attn layer types 'lsh' and 'local' exist, but `config.attn_layers`: {config.attn_layers}. Select "
            "attn layer types from ['lsh', 'local'] only."
        )

# AxialPositionEmbeddings 类，用于构建轴向位置嵌入，适用于非常长的输入序列以节省内存和时间
class AxialPositionEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 获取轴向位置形状和维度
        self.axial_pos_shape = config.axial_pos_shape
        self.axial_pos_embds_dim = config.axial_pos_embds_dim
        self.dropout = config.hidden_dropout_prob

        # 获取最小公倍数块长度
        self.least_common_mult_chunk_length = _get_least_common_mult_chunk_len(config)
        # 创建参数列表
        self.weights = nn.ParameterList()

        # 如果轴向位置嵌入维度之和不等于隐藏大小，抛出值错误
        if sum(self.axial_pos_embds_dim) != config.hidden_size:
            raise ValueError(
                f"Make sure that config.axial_pos_embds factors: {self.axial_pos_embds_dim} sum to "
                f"config.hidden_size: {config.hidden_size}"
            )

        # 创建权重
        for axis, axial_pos_embd_dim in enumerate(self.axial_pos_embds_dim):
            # 创建扩展形状
            ax_shape = [1] * len(self.axial_pos_shape)
            ax_shape[axis] = self.axial_pos_shape[axis]
            ax_shape = tuple(ax_shape) + (axial_pos_embd_dim,)

            # 创建张量并初始化
            self.weights.append(nn.Parameter(torch.ones(ax_shape, dtype=torch.float32)))

# PositionEmbeddings 类，用于构建传统的位置嵌入，形状为 `[max_pos_embeddings, hidden_size]`。
    # 初始化方法，用于类的实例化时进行初始化操作
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()
        # 设置类属性dropout为给定配置中的隐藏层dropout概率
        self.dropout = config.hidden_dropout_prob
        # 创建一个嵌入层，用于将位置ID映射到隐藏大小的向量空间
        self.embedding = nn.Embedding(config.max_position_embeddings, config.hidden_size)
    
    # 前向传播方法，用于定义模型的前向计算逻辑
    def forward(self, position_ids):
        # 根据位置ID获取位置嵌入
        position_embeddings = self.embedding(position_ids)
        # 对位置嵌入进行dropout操作，以减少过拟合风险
        position_embeddings = nn.functional.dropout(position_embeddings, p=self.dropout, training=self.training)
        # 返回处理后的位置嵌入
        return position_embeddings
class ReformerEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        self.max_position_embeddings = config.max_position_embeddings  # 保存最大位置编码长度
        self.dropout = config.hidden_dropout_prob  # 保存隐藏层dropout概率

        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)  # 创建词嵌入层
        self.position_embeddings = (
            AxialPositionEmbeddings(config) if config.axial_pos_embds else PositionEmbeddings(config)
        )  # 创建位置编码层，根据配置选择使用AxialPositionEmbeddings或PositionEmbeddings

    def forward(self, input_ids=None, position_ids=None, inputs_embeds=None, start_idx_pos_encodings=0):
        if input_ids is not None:
            input_shape = input_ids.size()  # 获取输入id的形状
            device = input_ids.device  # 获取输入id的设备
        else:
            input_shape = inputs_embeds.size()[:-1]  # 获取嵌入输入的形状
            device = inputs_embeds.device  # 获取嵌入输入的设备

        seq_length = input_shape[1]  # 获取序列长度
        if position_ids is None:
            position_ids = torch.arange(
                start_idx_pos_encodings, start_idx_pos_encodings + seq_length, dtype=torch.long, device=device
            )  # 生成位置编码ids
            position_ids = position_ids.unsqueeze(0).expand(input_shape)  # 扩展位置编码ids形状为和输入相同

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)  # 通过输入id获取词嵌入

        if position_ids.shape[-1] > self.max_position_embeddings:  # 如果位置编码长度超过最大位置编码长度
            raise ValueError(
                f"Sequence Length: {position_ids.shape[-1]} has to be less or equal than "
                f"config.max_position_embeddings {self.max_position_embeddings}."
            )  # 抛出数值错误异常

        # dropout
        embeddings = nn.functional.dropout(inputs_embeds, p=self.dropout, training=self.training)  # 对输入嵌入应用dropout

        # add positional embeddings
        position_embeddings = self.position_embeddings(position_ids)  # 获取位置编码嵌入
        embeddings = embeddings + position_embeddings  # 将位置编码嵌入加到输入嵌入上
        return embeddings  # 返回得到的嵌入向量


class EfficientAttentionMixin:
    """
    A few utilities for nn.Modules in Reformer, to be used as a mixin.
    """

    def _look_adjacent(self, vectors, num_chunks_before, num_chunks_after):
        """
        Used to implement attention between consecutive chunks.

        Args:
            vectors: array of shape [batch_size, num_attention_heads, n_chunks, chunk_len, ...]
            num_chunks_before: chunks before current chunk to include in attention
            num_chunks_after: chunks after current chunk to include in attention

        Returns:
            tensor of shape [num_chunks, N * chunk_length, ...], where N = (1 + num_chunks_before + num_chunks_after).
        """
        if num_chunks_before == 0 and num_chunks_after == 0:  # 如果前后chunk数量都为0
            return vectors  # 直接返回输入的向量

        slices = []
        for i in range(-num_chunks_before, num_chunks_after + 1):  # 遍历前后chunk数量范围
            if i == 0:  # 如果是当前chunk
                slices.append(vectors)  # 直接添加当前chunk的向量
            else:
                slices.append(torch.cat([vectors[:, :, i:, ...], vectors[:, :, :i, ...]], dim=2))  # 拼接前后chunk的向量
        return torch.cat(slices, dim=3)  # 返回拼接后的向量
    def _split_hidden_size_dim(self, x, num_attn_heads, attn_head_size):
        """
        splits hidden_size dim into attn_head_size and num_attn_heads
        将 hidden_size 维度划分成 attn_head_size 和 num_attn_heads
        """
        # 计算新的张量形状
        new_x_shape = x.size()[:-1] + (num_attn_heads, attn_head_size)
        # 重新调整张量形状
        x = x.view(*new_x_shape)
        return x.transpose(2, 1)

    def _merge_hidden_size_dims(self, x, num_attn_heads, attn_head_size):
        """
        merges attn_head_size dim and num_attn_heads dim into hidden_size
        将 attn_head_size 维度和 num_attn_heads 维度合并成 hidden_size
        """
        # 对张量进行转置操作
        x = x.permute(0, 2, 1, 3)
        return torch.reshape(x, (x.size()[0], -1, num_attn_heads * attn_head_size))

    def _split_seq_length_dim_to(self, vectors, dim_factor_1, dim_factor_2, num_attn_heads, attn_head_size=None):
        """
        splits sequence length dim of vectors into `dim_factor_1` and `dim_factor_2` dims
        将向量的序列长度维度划分为 `dim_factor_1` 和 `dim_factor_2` 维度
        """
        # 获取批处理大小
        batch_size = vectors.shape[0]
        split_dim_shape = (batch_size, num_attn_heads, dim_factor_1, dim_factor_2)

        # 根据向量维度情况进行不同的处理
        if len(vectors.shape) == 4:
            return torch.reshape(vectors, split_dim_shape + (attn_head_size,))
        elif len(vectors.shape) == 3:
            return torch.reshape(vectors, split_dim_shape)
        else:
            raise ValueError(f"Input vector rank should be one of [3, 4], but is: {len(vectors.shape)}")
```  
class LSHSelfAttention(nn.Module, EfficientAttentionMixin):
    # 定义了一个自注意力机制的类，继承自 nn.Module 和 EfficientAttentionMixin

    def __init__(self, config):
        # 初始化函数，传入配置参数
        super().__init__()
        # 调用父类的初始化函数
        self.config = config
        # 将传入的配置参数保存在实例中

        # 以下是初始化一系列配置参数
        self.chunk_length = config.lsh_attn_chunk_length
        self.num_hashes = config.num_hashes
        self.num_buckets = config.num_buckets
        self.num_chunks_before = config.lsh_num_chunks_before
        self.num_chunks_after = config.lsh_num_chunks_after
        self.hash_seed = config.hash_seed
        self.is_decoder = config.is_decoder
        self.max_position_embeddings = config.max_position_embeddings

        self.dropout = config.lsh_attention_probs_dropout_prob

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.attention_head_size
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.hidden_size = config.hidden_size

        # 创建查询（query）和键值对（key-value）的线性变换层
        self.query_key = nn.Linear(self.hidden_size, self.all_head_size, bias=False)
        self.value = nn.Linear(self.hidden_size, self.all_head_size, bias=False)

        # 保存不同数据类型下的 mask 值，使用 register_buffer 创建持久化的缓冲区
        self.register_buffer("self_mask_value_float16", torch.tensor(-1e3), persistent=False)
        self.register_buffer("self_mask_value_float32", torch.tensor(-1e5), persistent=False)
        self.register_buffer("mask_value_float16", torch.tensor(-1e4), persistent=False)
        self.register_buffer("mask_value_float32", torch.tensor(-1e9), persistent=False)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        num_hashes=None,
        buckets=None,
        past_buckets_states=None,
        use_cache=False,
        output_attentions=False,
        **kwargs,
    ):
        # 定义前向传播函数，包含多个输入参数和关键字参数

    def _query_per_attn_head(self, hidden_states):
        # 定义查询向量的函数，针对每个注意力头分别计算
        per_head_query_key = self.query_key.weight.reshape(
            self.num_attention_heads, self.attention_head_size, self.hidden_size
        ).transpose(-2, -1)
        # 根据 linear 层的权重重塑并转置得到每个注意力头的查询向量
        query_key_vectors = torch.einsum("balh,ahr->balr", hidden_states, per_head_query_key)
        # 使用 einsum 计算查询向量
        return query_key_vectors

    def _value_per_attn_head(self, hidden_states):
        # 定义值向量的函数，针对每个注意力头分别计算
        per_head_value = self.value.weight.reshape(
            self.num_attention_heads, self.attention_head_size, self.hidden_size
        ).transpose(-2, -1)
        # 根据 linear 层的权重重塑并转置得到每个注意力头的值向量
        value_vectors = torch.einsum("balh,ahr->balr", hidden_states, per_head_value)
        # 使用 einsum 计算值向量
        return value_vectors
    def _get_sorted_bucket_idx_and_undo_sorted_bucket_idx(self, sequence_length, buckets, num_hashes):
        # 不需要梯度
        with torch.no_grad():
            # 基于哈希值对 buckets 进行排序
            sorted_bucket_idx = _stable_argsort(buckets, dim=-1)

            # 创建简单的索引来散开，以便进行撤销排序
            indices = (
                torch.arange(sorted_bucket_idx.shape[-1], device=buckets.device)
                .view(1, 1, -1)
                .expand(sorted_bucket_idx.shape)
            )

            # 获取撤销排序
            undo_sorted_bucket_idx = sorted_bucket_idx.new(*sorted_bucket_idx.size())
            undo_sorted_bucket_idx.scatter_(-1, sorted_bucket_idx, indices)

        return sorted_bucket_idx, undo_sorted_bucket_idx

    def _set_num_buckets(self, sequence_length):
        # 推荐将 `num_buckets` 设置为 2 * sequence_length // chunk_length
        num_buckets_pow_2 = (2 * (sequence_length // self.chunk_length)).bit_length() - 1
        # 确保桶的数量是 2 的幂
        num_buckets = 2**num_buckets_pow_2

        # 如果 `num_buckets` 变得太大，则对其进行因式分解
        num_buckets_limit = 2 * max(
            int((self.max_position_embeddings // self.chunk_length) ** (0.5)),
            self.chunk_length,
        )
        if num_buckets > num_buckets_limit:
            num_buckets = [2 ** (num_buckets_pow_2 // 2), 2 ** (num_buckets_pow_2 - num_buckets_pow_2 // 2)]

        logger.warning(f"config.num_buckets 未设置。将 config.num_buckets 设置为 {num_buckets}...")

        # 在配置中设置 num_buckets 以便正确保存
        self.config.num_buckets = num_buckets
        self.num_buckets = num_buckets

    def _attend(
        self,
        query_vectors,
        key_vectors,
        value_vectors,
        sorted_bucket_idx_per_hash,
        attention_mask,
        head_mask,
        do_standard_self_attention,
        do_cached_attention,
    # 计算注意力蒙版
    def _compute_attn_mask(
        self, query_indices, key_indices, attention_mask, query_key_dot_shape, do_standard_self_attention
    ):
        # 如果注意力mask不为None，为LSH准备注意力mask
        if attention_mask is not None:
            # 如果使用分块注意力，注意力mask需要对应LSH的顺序
            attention_mask = attention_mask.to(torch.bool)[:, None, :]
            if not do_standard_self_attention:
                # 将attention_mask扩展到与key_value_bucket_idx形状匹配
                attention_mask = attention_mask[:, None, :]
                attention_mask = attention_mask.expand(query_indices.shape[:-1] + (-1,))
                # 从LSH排序的key_indices中提取注意力mask
                attention_mask = torch.gather(attention_mask, -1, key_indices)

            attention_mask = attention_mask.unsqueeze(-2).expand(query_key_dot_shape)

        # 因果mask
        if self.is_decoder is True:
            causal_mask = torch.ge(query_indices.unsqueeze(-1), key_indices.unsqueeze(-2)).to(query_indices.device)

            # 如果attention_mask不为None，添加注意力mask
            if attention_mask is not None:
                attention_mask = causal_mask * attention_mask
            else:
                attention_mask = causal_mask

        return attention_mask

    def _get_relevant_hid_states_and_buckets(
        self, query_vectors, attention_mask, num_hashes, hidden_states, past_states, past_buckets
    def _expand_to_indices_in_relevant_chunk(self, indices, sequence_length):
        # 获取分块开始位置及其大小的相关索引
        start_indices_chunk = ((indices[:, -1] // self.chunk_length) - self.num_chunks_before) * self.chunk_length
        total_chunk_size = self.chunk_length * (1 + self.num_chunks_before + self.num_chunks_after)

        # 扩展起始索引并通过arange添加正确的块偏移量
        expanded_start_indices = start_indices_chunk.unsqueeze(-1).expand(indices.shape[0], total_chunk_size)
        chunk_sequence_indices = expanded_start_indices + torch.arange(
            total_chunk_size, device=indices.device, dtype=torch.long
        ).unsqueeze(0).expand(indices.shape[0], total_chunk_size)

        # 确保通过% seq len保持循环逻辑
        chunk_sequence_indices = chunk_sequence_indices.flatten() % sequence_length

        # 扩展索引并设置正确的索引
        indices = indices.unsqueeze(1).expand((indices.shape[0], total_chunk_size, -1)).flatten(0, 1).clone()
        indices[:, -1] = chunk_sequence_indices

        return indices

    def _len_and_dim_norm(self, vectors, sqrt_num):
        """
        长度和注意力头大小的维度标准化
        """
        vectors = self._len_norm(vectors)
        vectors = vectors / sqrt_num
        return vectors

    def _len_norm(self, x, epsilon=1e-6):
        """
        长度标准化
        """
        variance = torch.mean(x**2, -1, keepdim=True)
        norm_x = x * torch.rsqrt(variance + epsilon)
        return norm_x
    # 通过扩展维度来收集向量和索引
    def _gather_by_expansion(self, vectors, idxs, num_hashes):
        # 扩展 idxs 的维度，使其与 vectors 的维度匹配
        # 第一维度保持不变，第二维度也保持不变
        # 第三维度从原来的单个值变为 num_hashes 个值
        # 第四维度保持不变，为 self.attention_head_size
        expanded_idxs = idxs.unsqueeze(-1).expand(-1, -1, -1, self.attention_head_size)
        # 复制 vectors ，使其第三维度从原来的单个值变为 num_hashes 个值
        vectors = vectors.repeat(1, 1, num_hashes, 1)
        # 根据扩展后的 idxs 从 vectors 中收集对应的值，返回结果
        return torch.gather(vectors, 2, expanded_idxs)
class ReverseSort(Function):
    """
    After chunked attention is applied which sorted clusters, original ordering has to be restored. Since customized
    backward function is used for Reformer, the gradients of the output vectors have to be explicitly sorted here.
    """

    @staticmethod
    def forward(ctx, out_vectors, logits, sorted_bucket_idx, undo_sorted_bucket_idx):
        # save sorted_bucket_idx for backprop
        with torch.no_grad():
            ctx.sorted_bucket_idx = sorted_bucket_idx

            # undo sort to have correct order for next layer
            expanded_undo_sort_indices = undo_sorted_bucket_idx.unsqueeze(-1).expand(out_vectors.shape)
            out_vectors = torch.gather(out_vectors, 2, expanded_undo_sort_indices)
            logits = torch.gather(logits, 2, undo_sorted_bucket_idx)
        return out_vectors, logits

    @staticmethod
    def backward(ctx, grad_out_vectors, grad_logits):
        # get parameters saved in ctx
        sorted_bucket_idx = ctx.sorted_bucket_idx

        expanded_sort_indices = sorted_bucket_idx.unsqueeze(-1).expand(grad_out_vectors.shape)
        # reverse sort of forward
        grad_out_vectors = torch.gather(grad_out_vectors, 2, expanded_sort_indices)
        grad_logits = torch.gather(grad_logits, 2, sorted_bucket_idx)

        # return grad and `None` fillers for last 2 forward args
        return grad_out_vectors, grad_logits, None, None


class LocalSelfAttention(nn.Module, EfficientAttentionMixin):
    def __init__(self, config):
        super().__init__()

        self.num_attention_heads = config.num_attention_heads
        self.chunk_length = config.local_attn_chunk_length
        self.num_chunks_before = config.local_num_chunks_before
        self.num_chunks_after = config.local_num_chunks_after
        self.is_decoder = config.is_decoder
        self.pad_token_id = config.pad_token_id

        self.attention_head_size = config.attention_head_size
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.hidden_size = config.hidden_size

        # projection matrices
        self.query = nn.Linear(self.hidden_size, self.all_head_size, bias=False)
        self.key = nn.Linear(self.hidden_size, self.all_head_size, bias=False)
        self.value = nn.Linear(self.hidden_size, self.all_head_size, bias=False)

        self.dropout = config.local_attention_probs_dropout_prob

        # save mask value here
        self.register_buffer("mask_value_float16", torch.tensor(-1e4), persistent=False)
        self.register_buffer("mask_value_float32", torch.tensor(-1e9), persistent=False)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        past_buckets_states=None,
        use_cache=False,
        output_attentions=False,
        **kwargs,
    ):
        # method for performing forward pass of the LocalSelfAttention module

    def _compute_attn_mask(
        self, query_indices, key_indices, attention_mask, query_key_dots_shape, do_standard_self_attention
    ):
        # internal method for computing attention mask based on query and key indices
    # 从输入的查询、键以及注意力掩码构建注意力掩码
    def build_attention_mask(
        self,
        query_indices,
        key_indices,
        attention_mask=None,
        do_standard_self_attention=False
    ):
        # 如果传入了注意力掩码，则将其转换为布尔类型并在第二个维度上添加一个新维度
        if attention_mask is not None:
            attention_mask = attention_mask.to(torch.bool)[:, None, :]
    
            # 如果不是标准的自注意力，则将注意力掩码按块分割，并在前后添加掩码
            if not do_standard_self_attention:
                attention_mask = self._split_seq_length_dim_to(attention_mask, -1, self.chunk_length, 1)
                attention_mask = self._look_adjacent(attention_mask, self.num_chunks_before, self.num_chunks_after)
            # 在最后一个维度上添加一个新维度，并扩展到 query_key_dots_shape 的大小
            attention_mask = attention_mask.unsqueeze(-2).expand(query_key_dots_shape)
    
        # 如果是解码器模型，则创建因果掩码
        if self.is_decoder is True:
            causal_mask = torch.ge(query_indices.unsqueeze(-1), key_indices.unsqueeze(-2)).to(query_indices.device)
    
            # 如果有注意力掩码，则将因果掩码与注意力掩码相乘
            if attention_mask is not None:
                attention_mask = causal_mask * attention_mask
            else:
                attention_mask = causal_mask
    
        return attention_mask
    
    # 根据前一个隐藏状态和块长度、块数获取相关的隐藏状态
    @staticmethod
    def _retrieve_relevant_hidden_states(previous_hidden_states, chunk_length, num_chunks_before):
        # 计算起始位置，使用上一个隐藏状态的长度减去前面的块数乘以块长度
        start_position = ((previous_hidden_states.shape[1] // chunk_length) - num_chunks_before) * chunk_length
        # 返回从起始位置开始的隐藏状态
        return previous_hidden_states[:, start_position:]
class ReformerSelfOutput(nn.Module):
    # 定义一个 ReformerSelfOutput 类，继承自 nn.Module
    def __init__(self, config):
        # 初始化方法
        super().__init__()
        # 调用父类的初始化方法
        all_head_size = config.num_attention_heads * config.attention_head_size
        # 计算总的头部大小
        self.dropout = config.hidden_dropout_prob
        # 设置 dropout 的概率

        self.dense = nn.Linear(all_head_size, config.hidden_size, bias=False)
        # 初始化一个全连接层

    def forward(self, hidden_states):
        # 前向传播方法
        hidden_states = self.dense(hidden_states)
        # 全连接层对隐藏状态进行变换
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        # 使用 dropout 进行正则化处理
        return hidden_states
        # 返回处理后的隐藏状态


class ReformerAttention(nn.Module):
    # 定义一个 ReformerAttention 类，继承自 nn.Module
    def __init__(self, config, layer_id=0):
        # 初始化方法
        super().__init__()
        # 调用父类的初始化方法
        self.layer_id = layer_id
        # 设置层级 ID
        self.attn_layers = config.attn_layers
        # 获取 self attention 层的配置参数

        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 初始化 LayerNorm 层

        if len(set(self.attn_layers)) == 1 and self.attn_layers[0] == "lsh":
            # 如果只有一种注意力层类型为 LSH
            self.self_attention = LSHSelfAttention(config)
            # 初始化 LSHSelfAttention
        elif len(set(self.attn_layers)) == 1 and self.attn_layers[0] == "local":
            # 如果只有一种注意力层类型为 local
            self.self_attention = LocalSelfAttention(config)
            # 初始化 LocalSelfAttention
        elif len(set(self.attn_layers)) == 2 and set(self.attn_layers) == {"lsh", "local"}:
            # 如果既有 LSH 又有 local 类型的注意力层
            if self.attn_layers[self.layer_id] == "lsh":
                # 根据层级选择正确的注意力层
                self.self_attention = LSHSelfAttention(config)
            else:
                self.self_attention = LocalSelfAttention(config)
        else:
            raise NotImplementedError(
                f"Only attn layer types 'lsh' and 'local' exist, but got `config.attn_layers`: {self.attn_layers}. "
                "Select attn layer types from ['lsh', 'local'] only."
            )
        # 抛出 NotImplementedError 异常

        self.output = ReformerSelfOutput(config)
        # 初始化 ReformerSelfOutput

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        num_hashes=None,
        past_buckets_states=None,
        use_cache=False,
        orig_sequence_length=None,
        output_attentions=False,
        buckets=None,
        hidden_states = self.layer_norm(hidden_states)

        # 确保在反向传播时缓存的隐藏状态设置为 None
        if past_buckets_states is not None:
            past_buckets_states_layer = past_buckets_states[self.layer_id]
        else:
            past_buckets_states_layer = None

        # 如果 LSMSelfAttention 的桶不为 None，则使用缓存的桶进行反向传播
        self_attention_outputs = self.self_attention(
            hidden_states=hidden_states,
            head_mask=head_mask,
            attention_mask=attention_mask,
            num_hashes=num_hashes,
            past_buckets_states=past_buckets_states_layer,
            use_cache=use_cache,
            output_attentions=output_attentions,
            buckets=buckets,
        )

        # 如果 self_attention_outputs 有 "buckets" 属性，则添加桶
        if hasattr(self_attention_outputs, "buckets"):
            buckets = self_attention_outputs.buckets
        else:
            buckets = None

        # 缓存隐藏状态以备将来使用
        if use_cache:
            if past_buckets_states[self.layer_id][0] is None:
                # 填充的输入不应被缓存
                past_buckets = (
                    buckets[:, :, :, :orig_sequence_length]
                    if (buckets is not None and orig_sequence_length > 1)
                    else buckets
                )
            else:
                past_buckets = torch.cat([past_buckets_states[self.layer_id][0], buckets], dim=-1)

            if past_buckets_states[self.layer_id][1] is None:
                # 填充的输入不应被缓存
                past_states = hidden_states[:, :orig_sequence_length]
            else:
                past_states = torch.cat([past_buckets_states[self.layer_id][1], hidden_states], dim=1)

            past_buckets_states[self.layer_id] = (past_buckets, past_states)
        # 计算注意力前向传播的输出
        attention_output = self.output(self_attention_outputs.hidden_states)

        return AttentionOutput(
            hidden_states=attention_output,
            attention_probs=self_attention_outputs.attention_probs,
            buckets=buckets,
        )
class ReformerFeedForwardDense(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dropout = config.hidden_dropout_prob  # 从配置中获取隐藏层dropout概率

        # 如果配置中的隐藏激活函数是字符串，则获取对应的函数，否则直接使用配置中的函数
        if isinstance(config.hidden_act, str):  
            self.act_fn = ACT2FN[config.hidden_act]  # 获取隐藏激活函数对应的函数
        else:
            self.act_fn = config.hidden_act  # 直接使用配置中的隐藏激活函数

        self.dense = nn.Linear(config.hidden_size, config.feed_forward_size)  # 创建线性变换层

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)  # 将隐藏状态传递给线性变换层
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)  # 使用dropout函数进行丢弃
        hidden_states = self.act_fn(hidden_states)  # 使用隐藏激活函数处理隐藏状态
        return hidden_states  # 返回处理后的隐藏状态


class ReformerFeedForwardOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dropout = config.hidden_dropout_prob  # 从配置中获取隐藏层dropout概率

        self.dense = nn.Linear(config.feed_forward_size, config.hidden_size)  # 创建线性变换层

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)  # 将隐藏状态传递给线性变换层
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)  # 使用dropout函数进行丢弃
        return hidden_states  # 返回处理后的隐藏状态


class ChunkReformerFeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward  # 获取前馈层的块大小
        self.seq_len_dim = 1  # 序列长度维度

        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)  # 创建层归一化层
        self.dense = ReformerFeedForwardDense(config)  # 创建前馈层的线性变换层
        self.output = ReformerFeedForwardOutput(config)  # 创建前馈层的输出层

    def forward(self, attention_output):
        return apply_chunking_to_forward(
            self.forward_chunk,
            self.chunk_size_feed_forward,
            self.seq_len_dim,
            attention_output,
        )

    def forward_chunk(self, hidden_states):
        hidden_states = self.layer_norm(hidden_states)  # 使用层归一化处理隐藏状态
        hidden_states = self.dense(hidden_states)  # 将隐藏状态传递给前馈层的线性变换层
        return self.output(hidden_states)  # 返回前馈层输出的隐藏状���


class ReformerLayer(nn.Module):
    def __init__(self, config, layer_id=0):
        super().__init__()
        self.attention = ReformerAttention(config, layer_id)  # 创建ReformerAttention实例
        # dropout requires to have the same
        # seed for forward and backward pass
        self.attention_seed = None
        self.feed_forward_seed = None

        self.feed_forward = ChunkReformerFeedForward(config)  # 创建ChunkReformerFeedForward实例
    def _init_attention_seed(self):
        """
        This function sets a new seed for the attention layer to make dropout deterministic for both forward calls: 1
        normal forward call and 1 forward call in backward to recalculate activations.
        """

        # 随机化种子
        # 如果有可用的 CUDA 生成器，则使用 CUDA 生成器
        if hasattr(torch.cuda, "default_generators") and len(torch.cuda.default_generators) > 0:
            # GPU
            device_idx = torch.cuda.current_device()
            self.attention_seed = torch.cuda.default_generators[device_idx].seed()
        else:
            # CPU
            self.attention_seed = int(torch.seed() % sys.maxsize)

        # 设置随机数种子
        torch.manual_seed(self.attention_seed)

    def _init_feed_forward_seed(self):
        """
        This function sets a new seed for the feed forward layer to make dropout deterministic for both forward calls:
        1 normal forward call and 1 forward call in backward to recalculate activations.
        """
        # 随机化种子
        # 如果有可用的 CUDA 生成器，则使用 CUDA 生成器
        if hasattr(torch.cuda, "default_generators") and len(torch.cuda.default_generators) > 0:
            # GPU
            device_idx = torch.cuda.current_device()
            self.feed_forward_seed = torch.cuda.default_generators[device_idx].seed()
        else:
            # CPU
            self.feed_forward_seed = int(torch.seed() % sys.maxsize)

        # 设置随机数种子
        torch.manual_seed(self.feed_forward_seed)

    def forward(
        self,
        prev_attn_output,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        num_hashes=None,
        past_buckets_states=None,
        use_cache=False,
        orig_sequence_length=None,
        output_attentions=False,
        # 在没有梯度下降的情况下执行操作
        with torch.no_grad():
            # 每次前向传递我们采样一个不同的种子(dropout种子)
            # 并保存在向后传递的前向函数中以保持正确的dropout效果
            if self.training:
                self._init_attention_seed()

            # 对输入进行自注意力计算
            attn_outputs = self.attention(
                hidden_states=hidden_states,
                head_mask=head_mask,
                attention_mask=attention_mask,
                num_hashes=num_hashes,
                past_buckets_states=past_buckets_states,
                use_cache=use_cache,
                orig_sequence_length=orig_sequence_length,
                output_attentions=output_attentions,
            )
            # 得到自注意力输出
            attn_output = attn_outputs.hidden_states

            # 实施RevNet（参见 https://towardsdatascience.com/illustrating-the-reformer-393575ac6ba0 中的图6）
            # Y_1 = X_1 + f(X_2)
            attn_output = prev_attn_output + attn_output

            # 释放内存
            del prev_attn_output

            # 每次前向传递我们采样一个不同的种子(dropout种子)
            # 并保存在向后传递的前向函数中以保持正确的dropout效果
            if self.training:
                self._init_feed_forward_seed()
            # Y_2 = X_2 + g(Y_1)
            hidden_states = hidden_states + self.feed_forward(attn_output)

        # 返回ReformerOutput
        return ReformerOutput(
            attn_output=attn_output,
            hidden_states=hidden_states,
            attention_probs=attn_outputs.attention_probs,
            buckets=attn_outputs.buckets,
        )

    def backward_pass(
        self,
        next_attn_output,
        hidden_states,
        grad_attn_output,
        grad_hidden_states,
        attention_mask=None,
        head_mask=None,
        buckets=None,
    # 实现可逆 ResNets 的反向传播。
    # 详情可参考这篇博文: https://towardsdatascience.com/illustrating-the-reformer-393575ac6ba0
    # 这段代码主要参考自 https://github.com/lucidrains/reformer-pytorch/blob/master/reformer_pytorch/reversible.py
    
    # 确保模型处于训练模式
    assert self.training, (
        "If you want to train `ReformerModel` and its variations, make sure to use `model.train()` to put the"
        " model into training mode."
    )
    
    # 启用梯度计算
    with torch.enable_grad():
        # 设置 next_attn_output 需要计算梯度
        next_attn_output.requires_grad = True
    
        # 设置随机种子以确保 dropout 正确
        torch.manual_seed(self.feed_forward_seed)
        # 计算 g(Y_1)
        res_hidden_states = self.feed_forward(next_attn_output)
        # 对 res_hidden_states 进行反向传播计算梯度, 并保留计算图
        res_hidden_states.backward(grad_hidden_states, retain_graph=True)
    
    # 关闭梯度计算
    with torch.no_grad():
        # X_2 = Y_2 - g(Y_1)
        hidden_states = hidden_states - res_hidden_states
        # 释放 res_hidden_states 占用的内存
        del res_hidden_states
    
        # 将 next_attn_output 的梯度累加到 grad_attn_output 中
        grad_attn_output = grad_attn_output + next_attn_output.grad
        # 清除 next_attn_output 的梯度
        next_attn_output.grad = None
    
    # 启用梯度计算
    with torch.enable_grad():
        # 设置 hidden_states 需要计算梯度
        hidden_states.requires_grad = True
    
        # 设置随机种子以确保 dropout 正确
        torch.manual_seed(self.attention_seed)
        # 计算 f(X_2)
        # 如果 buckets 不为 None, 则使用缓存的 buckets 进行反向传播
        output = self.attention(
            hidden_states=hidden_states,
            head_mask=head_mask,
            attention_mask=attention_mask,
            buckets=buckets,
        ).hidden_states
        # 对 output 进行反向传播计算梯度, 并保留计算图
        output.backward(grad_attn_output, retain_graph=True)
    
    # 关闭梯度计算
    with torch.no_grad():
        # X_1 = Y_1 - f(X_2)
        attn_output = next_attn_output - output
        # 释放 output 和 next_attn_output 占用的内存
        del output, next_attn_output
    
        # 将 hidden_states 的梯度累加到 grad_hidden_states 中
        grad_hidden_states = grad_hidden_states + hidden_states.grad
        # 清除 hidden_states 的梯度
        hidden_states.grad = None
        # 将 hidden_states 转换为不需要梯度的 tensor
        hidden_states = hidden_states.detach()
    
    # 返回反向传播的输出
    return ReformerBackwardOutput(
        attn_output=attn_output,
        hidden_states=hidden_states,
        grad_attn_output=grad_attn_output,
        grad_hidden_states=grad_hidden_states,
    )
class _ReversibleFunction(Function):
    """
    To prevent PyTorch from performing the usual backpropagation, a customized backward function is implemented here.
    This way it is made sure that no memory expensive activations are saved during the forward pass. This function is
    heavily inspired by https://github.com/lucidrains/reformer-pytorch/blob/master/reformer_pytorch/reversible.py
    """

    @staticmethod
    def forward(
        ctx,
        hidden_states,
        layers,
        attention_mask,
        head_mask,
        num_hashes,
        all_hidden_states,
        all_attentions,
        past_buckets_states,
        use_cache,
        orig_sequence_length,
        output_hidden_states,
        output_attentions,
    ):
        all_buckets = ()

        # split duplicated tensor
        hidden_states, attn_output = torch.chunk(hidden_states, 2, dim=-1)

        for layer_id, (layer, layer_head_mask) in enumerate(zip(layers, head_mask)):
            if output_hidden_states is True:
                all_hidden_states.append(hidden_states)

            # call the current layer with the provided arguments
            layer_outputs = layer(
                prev_attn_output=attn_output,
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                head_mask=layer_head_mask,
                num_hashes=num_hashes,
                past_buckets_states=past_buckets_states,
                use_cache=use_cache,
                orig_sequence_length=orig_sequence_length,
                output_attentions=output_attentions,
            )

            # update attn_output and hidden_states with the outputs from the current layer
            attn_output = layer_outputs.attn_output
            hidden_states = layer_outputs.hidden_states
            # add the buckets from the current layer to the all_buckets tuple
            all_buckets = all_buckets + (layer_outputs.buckets,)

            # if output_attentions is True, then append the attention_probs from the current layer to all_attentions
            if output_attentions:
                all_attentions.append(layer_outputs.attention_probs)

        # Add last layer to all_hidden_states if output_hidden_states is True
        if output_hidden_states is True:
            all_hidden_states.append(hidden_states)

        # attach necessary parameters to ctx for backward
        ctx.save_for_backward(attn_output.detach(), hidden_states.detach())
        ctx.layers = layers
        ctx.all_buckets = all_buckets
        ctx.head_mask = head_mask
        ctx.attention_mask = attention_mask

        # Concatenate the attention output and hidden_states and return the result
        return torch.cat([attn_output, hidden_states], dim=-1)

    @staticmethod
    # 该函数实现了反向传播的计算过程
    def backward(ctx, grad_hidden_states):
        # 将 grad_hidden_states 张量拆分为 grad_attn_output 和 grad_hidden_states 两部分
        grad_attn_output, grad_hidden_states = torch.chunk(grad_hidden_states, 2, dim=-1)
    
        # 从 ctx 中取出正向传播时保存的中间结果
        attn_output, hidden_states = ctx.saved_tensors
    
        # 创建一个 namedtuple 对象, 用于存储反向传播所需的中间结果
        output = ReformerBackwardOutput(
            attn_output=attn_output,
            hidden_states=hidden_states,
            grad_attn_output=grad_attn_output,
            grad_hidden_states=grad_hidden_states,
        )
    
        # 释放一些不再需要的内存
        del grad_attn_output, grad_hidden_states, attn_output, hidden_states
    
        # 从 ctx 中取出其他所需的参数
        layers = ctx.layers
        all_buckets = ctx.all_buckets
        head_mask = ctx.head_mask
        attention_mask = ctx.attention_mask
    
        # 反向遍历各个层, 进行反向传播
        for idx, layer in enumerate(layers[::-1]):
            # 从栈中弹出最后一个 buckets
            buckets = all_buckets[-1]
            all_buckets = all_buckets[:-1]
    
            # 调用当前层的 backward_pass 方法, 完成反向传播计算
            output = layer.backward_pass(
                next_attn_output=output.attn_output,
                hidden_states=output.hidden_states,
                grad_attn_output=output.grad_attn_output,
                grad_hidden_states=output.grad_hidden_states,
                head_mask=head_mask[len(layers) - idx - 1],
                attention_mask=attention_mask,
                buckets=buckets,
            )
    
        # 检查 all_buckets 是否为空, 确保反向传播完成
        assert all_buckets == (), "buckets have to be empty after backpropagation"
    
        # 将 grad_attn_output 和 grad_hidden_states 拼接成最终的梯度
        grad_hidden_states = torch.cat([output.grad_attn_output, output.grad_hidden_states], dim=-1)
    
        # 返回梯度, 其他参数返回 None
        return grad_hidden_states, None, None, None, None, None, None, None, None, None, None, None
# 定义了一个 ReformerEncoder 类，继承自 nn.Module 类
class ReformerEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 设置 dropout 概率为 config.hidden_dropout_prob
        self.dropout = config.hidden_dropout_prob

        # 创建一个 nn.ModuleList，其中包含 config.num_hidden_layers 个 ReformerLayer 实例
        self.layers = nn.ModuleList([ReformerLayer(config, i) for i in range(config.num_hidden_layers)])
        # Reformer 使用 Rev Nets，因此最后一层的输出被连接起来，并且 Layer Norm 应用于 2 * hidden_size
        self.layer_norm = nn.LayerNorm(2 * config.hidden_size, eps=config.layer_norm_eps)

    # 前向传播函数，接受多个输入参数，并返回 ReformerEncoderOutput 对象
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        num_hashes=None,
        past_buckets_states=None,
        use_cache=False,
        orig_sequence_length=None,
        output_hidden_states=False,
        output_attentions=False,
    ):
        # 如果希望填充 hidden_states 和 attention 列表，则初始化它们
        all_hidden_states = []
        all_attentions = []

        # 如果 past_buckets_states 为 None，则初始化缓存的隐藏状态
        if past_buckets_states is None:
            past_buckets_states = [((None), (None)) for i in range(len(self.layers))]

        # 在最后一个维度上拼接相同的张量，用于可逆 ResNet
        hidden_states = torch.cat([hidden_states, hidden_states], dim=-1)
        # 使用自定义的 _ReversibleFunction 类的 apply 方法进行前向传播
        hidden_states = _ReversibleFunction.apply(
            hidden_states,
            self.layers,
            attention_mask,
            head_mask,
            num_hashes,
            all_hidden_states,
            all_attentions,
            past_buckets_states,
            use_cache,
            orig_sequence_length,
            output_hidden_states,
            output_attentions,
        )

        # 对连接后的隐藏状态应用 Layer Norm
        hidden_states = self.layer_norm(hidden_states)

        # 应用 dropout
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        # 返回 ReformerEncoderOutput 对象，包含隐藏状态、所有隐藏状态、所有注意力、过去的桶状态
        return ReformerEncoderOutput(
            hidden_states=hidden_states,
            all_hidden_states=all_hidden_states,
            all_attentions=all_attentions,
            past_buckets_states=past_buckets_states,
        )


# 定义了一个 ReformerOnlyLMHead 类，继承自 nn.Module 类
class ReformerOnlyLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Reformer 使用 Rev Nets，因此最后一层的输出被连接起来，并且 Layer Norm 应用于 2 * hidden_size
        self.seq_len_dim = 1
        self.chunk_size_lm_head = config.chunk_size_lm_head
        # 创建一个线性层，输入尺寸为 2 * hidden_size，输出尺寸为 vocab_size，无偏置
        self.decoder = nn.Linear(2 * config.hidden_size, config.vocab_size, bias=False)
        # 创建一个偏置参数，并将其赋值给 decoder 的偏置
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.decoder.bias = self.bias

    # 前向传播函数，接受隐藏状态作为输入，并返回输出结果
    def forward(self, hidden_states):
        # 将前向传播应用于隐藏状态的块，以处理长序列
        return apply_chunking_to_forward(self.forward_chunk, self.chunk_size_lm_head, self.seq_len_dim, hidden_states)

    # 处理隐藏状态的块，接受隐藏状态作为输入，并返回输出结果
    def forward_chunk(self, hidden_states):
        # 将隐藏状态传递给线性层
        hidden_states = self.decoder(hidden_states)
        return hidden_states
    # 将权重绑定在一起，以防它们在 TPU 上或在调整偏置时断开连接
    self.bias = self.decoder.bias
class ReformerPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 指定默认的配置类为ReformerConfig
    config_class = ReformerConfig
    # 模型基础前缀为"reformer"
    base_model_prefix = "reformer"

    @property
    def dummy_inputs(self):
        # 创建一个包含虚拟输入的字典
        input_ids = torch.tensor(DUMMY_INPUTS)
        input_mask = torch.tensor(DUMMY_MASK)
        dummy_inputs = {
            "input_ids": input_ids,
            "attention_mask": input_mask,
        }
        return dummy_inputs

    def _init_weights(self, module):
        """Initialize the weights"""
        # 如果模块是AxialPositionEmbeddings类型
        if isinstance(module, AxialPositionEmbeddings):
            # 初始化AxialPositionEmbeddings中的权重
            for weight in module.weights:
                nn.init.normal_(weight, std=self.config.axial_norm_std)
        # 如果模块是nn.Embedding类型
        elif isinstance(module, nn.Embedding):
            # 初始化嵌入层的权重
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果有填充索引，则将填充索引处的权重设为0
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        # 如果模块是nn.Linear类型
        elif isinstance(module, nn.Linear):
            # 稍微与TF版本有所不同，TF版本使用截断正态分布进行初始化
            # 参考 https://github.com/pytorch/pytorch/pull/5617
            # 初始化线性层的权重
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果有偏置，则将偏置设为0
            if module.bias is not None:
                module.bias.data.zero_()
        # 如果模块是nn.LayerNorm类型
        elif isinstance(module, nn.LayerNorm):
            # 将LayerNorm的偏置设为0
            module.bias.data.zero_()
            # 将LayerNorm的权重设为1
            module.weight.data.fill_(1.0)


@dataclass
class ReformerModelOutput(ModelOutput):
    """
    Output type of [`ReformerModel`].
    """
    # 定义输出的各个字段的类型和含义
    Args:
        # last_hidden_state 表示最后一层的隐藏状态，shape 为 (batch_size, num_predict, hidden_size)
        # num_predict 对应 target_mapping.shape[1]，如果 target_mapping 为 None，则 num_predict 对应 sequence_length
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, num_predict, hidden_size)`):
            Sequence of hidden-states at the last layer of the model.
        # past_buckets_states 是过去的桶和隐藏状态的列表，当 use_cache=True 或 config.use_cache=True 时返回
        # 其中第一个元素表示前一个 buckets，shape 为 (batch_size, num_heads, num_hashes, sequence_length)
        # 第二个元素表示前一个隐藏状态，shape 为 (batch_size, sequence_length, hidden_size)
        past_buckets_states (`List[Tuple(torch.LongTensor, torch.FloatTensor)]`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            List of `Tuple(torch.LongTensor, torch.FloatTensor` of length `config.n_layers`, with the first element
            being the previous *buckets* of shape `(batch_size, num_heads, num_hashes, sequence_length)`) and the
            second being the previous *hidden_states* of shape `(batch_size, sequence_length, hidden_size)`).
    
            Contains precomputed buckets and hidden-states that can be used (see `past_buckets_states` input) to speed
            up sequential decoding.
        # hidden_states 是模型各层的隐藏状态，当 output_hidden_states=True 或 config.output_hidden_states=True 时返回
        # 每个元素的 shape 为 (batch_size, sequence_length, hidden_size)
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings and one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.
    
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        # attentions 是各层的注意力权重，当 output_attentions=True 或 config.output_attentions=True 时返回
        # 每个元素的 shape 为 (batch_size, num_heads, sequence_length, sequence_length)
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.
    
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    
    # 定义输出字段的类型
    last_hidden_state: torch.FloatTensor
    past_buckets_states: Optional[List[Tuple[torch.LongTensor, torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
# 使用 dataclass 装饰器定义了一个类，用于存储 ReformerModelWithLMHead 模型的输出
@dataclass
class ReformerModelWithLMHeadOutput(ModelOutput):
    """
    Output type of [`ReformerModelWithLMHead`].

    Args:
        loss (`torch.FloatTensor` of shape *(1,)*, *optional*, returned when `labels` is provided)
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, num_predict, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).

            `num_predict` corresponds to `target_mapping.shape[1]`. If `target_mapping` is `None`, then `num_predict`
            corresponds to `sequence_length`.
        past_buckets_states (`List[Tuple(torch.LongTensor, torch.FloatTensor)]`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            List of `Tuple(torch.LongTensor, torch.FloatTensor` of length `config.n_layers`, with the first element
            being the previous *buckets* of shape `(batch_size, num_heads, num_hashes, sequence_length)`) and the
            second being the previous *hidden_states* of shape `(batch_size, sequence_length, hidden_size)`).

            Contains precomputed buckets and hidden-states that can be used (see `past_buckets_states` input) to speed
            up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings and one for the output of each layer)
            of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    # 定义 loss 属性，表示语言建模损失（下一个标记的预测）
    loss: Optional[torch.FloatTensor] = None
    # 定义 logits 属性，表示语言建模头的预测得分（SoftMax之前的每个词汇标记的分数）
    logits: torch.FloatTensor = None
    # 定义 past_buckets_states 属性，包含预先计算的 buckets 和隐藏状态列表，用于加速顺序解码
    past_buckets_states: Optional[List[Tuple[torch.LongTensor, torch.FloatTensor]]] = None
    # 定义 hidden_states 属性，是模型每一层输出的隐藏状态的元组
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # 定义 attentions 属性，是每一层注意力权重的元组，用于计算自注意力头的加权平均值
    attentions: Optional[Tuple[torch.FloatTensor]] = None


# 定义 REFORMER_START_DOCSTRING 常量，包含 Reformer 模型的描述信息
REFORMER_START_DOCSTRING = r"""
    Reformer was proposed in [Reformer: The Efficient Transformer](https://arxiv.org/abs/2001.04451) by Nikita Kitaev,
    Łukasz Kaiser, Anselm Levskaya.

    This model inherits from[`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)
    # 这个模型也是 PyTorch 的 torch.nn.Module 的子类
    # 可以像常规的 PyTorch 模块一样使用，并参考 PyTorch 文档以了解与一般使用和行为有关的所有事项
    
    # 参数：
    # config: ReformerConfig 类型，模型的所有参数的配置类。
    # 使用配置文件初始化不会加载与模型关联的权重，仅加载配置。查看 PreTrainedModel.from_pretrained 方法以加载模型权重。
"""

REFORMER_INPUTS_DOCSTRING = r"""
"""


@add_start_docstrings(
    "The bare Reformer Model transformer outputting raw hidden-stateswithout any specific head on top.",
    REFORMER_START_DOCSTRING,
)
class ReformerModel(ReformerPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        assert (
            self.config.num_hidden_layers > 0
        ), "`config.attn_layers` is empty. Select at least one attn layer form ['lsh', 'local']"

        self.embeddings = ReformerEmbeddings(config)  # 初始化 ReformerEmbeddings 层
        self.encoder = ReformerEncoder(config)  # 初始化 ReformerEncoder 层

        # 初始化权重并应用最终处理
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings  # 返回输入 embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value  # 设置输入 embeddings

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)  # 剪枝模型的注意力头部

    @add_start_docstrings_to_model_forward(REFORMER_INPUTS_DOCSTRING)  # 添加文档字符串到模型前向传播
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,  # 添加代码样本文档字符串
        output_type=ReformerModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        num_hashes: Optional[int] = None,
        past_buckets_states: Optional[List[Tuple[torch.Tensor]]] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    def _pad_to_mult_of_chunk_length(
        self,
        input_ids,
        inputs_embeds=None,
        attention_mask=None,
        position_ids=None,
        input_shape=None,
        padding_length=None,
        padded_seq_length=None,
        device=None,
    # 该函数用于处理输入数据的填充，以确保输入长度是 config.chunk_length 的倍数
    def auto_pad_inputs(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        inputs_embeds=None,
        device=None,
    ):
        # 如果输入长度不是 config.chunk_length 的倍数，则进行警告
        if padding_length > 0:
            logger.warning_once(
                f"Input ids are automatically padded from {input_shape[-1]} to {input_shape[-1] + padding_length} to be a "
                f"multiple of `config.chunk_length`: {padded_seq_length}"
            )
    
        # 创建填充的输入 ID，使用 config.pad_token_id 进行填充
        padded_input_ids = torch.full(
            (input_shape[0], padding_length),
            self.config.pad_token_id,
            device=device,
            dtype=torch.long,
        )
    
        # 如果有 attention_mask，则进行填充
        if attention_mask is not None:
            pad_attention_mask = torch.zeros(input_shape[0], padding_length, device=device, dtype=attention_mask.dtype)
            attention_mask = torch.cat([attention_mask, pad_attention_mask], dim=-1)
        # 如果没有 attention_mask，则创建一个新的 attention_mask
        else:
            attention_mask = torch.cat(
                [
                    torch.ones(input_shape, device=device, dtype=torch.bool),
                    torch.zeros((input_shape[0], padding_length), device=device, dtype=torch.bool),
                ],
                dim=-1,
            )
    
        # 如果有 input_ids，则进行填充
        if input_ids is not None:
            input_ids = torch.cat([input_ids, padded_input_ids], dim=-1)
            input_shape = input_ids.size()
    
            # 如果有 position_ids，则进行填充
            if position_ids is not None:
                padded_position_ids = torch.arange(input_shape[-1], padded_seq_length, dtype=torch.long, device=device)
                padded_position_ids = position_ids.unsqueeze(0).expand(input_shape[0], padding_length)
                position_ids = torch.cat([position_ids, padded_position_ids], dim=-1)
    
        # 如果有 inputs_embeds，则进行填充
        if inputs_embeds is not None:
            padded_inputs_embeds = self.embeddings(padded_input_ids, position_ids)
            inputs_embeds = torch.cat([inputs_embeds, padded_inputs_embeds], dim=-2)
            input_shape = inputs_embeds.size()
    
        return input_ids, inputs_embeds, attention_mask, position_ids, input_shape
# 该类是基于 Reformer 模型的语言建模头部的实现
@add_start_docstrings("""Reformer Model with a `language modeling` head on top.""", REFORMER_START_DOCSTRING)
class ReformerModelWithLMHead(ReformerPreTrainedModel):
    # 需要绑定权重的关键词
    _tied_weights_keys = ["lm_head.decoder.weight", "lm_head.decoder.bias"]

    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)
        # 检查配置是否为解码器模式
        assert config.is_decoder, "If you want to use `ReformerModelWithLMHead` make sure that `is_decoder=True`."
        # 如果使用了局部注意力机制，检查 local_num_chunks_after 是否为 0
        assert "local" not in self.config.attn_layers or config.local_num_chunks_after == 0, (
            "If causal mask is enabled, make sure that `config.local_num_chunks_after` is set to 0 and not"
            f" {config.local_num_chunks_after}."
        )
        # 如果使用了 LSH 注意力机制，检查 lsh_num_chunks_after 是否为 1
        assert "lsh" not in self.config.attn_layers or config.lsh_num_chunks_after == 0, (
            "If causal mask is enabled, make sure that `config.lsh_num_chunks_after` is set to 1 and not"
            f" {config.lsh_num_chunks_after}."
        )

        # 初始化 Reformer 模型
        self.reformer = ReformerModel(config)
        # 初始化语言建模头部
        self.lm_head = ReformerOnlyLMHead(config)

        # 初始化权重并进行后处理
        self.post_init()

    # 获取输出嵌入层
    def get_output_embeddings(self):
        return self.lm_head.decoder

    # 设置输出嵌入层
    def set_output_embeddings(self, new_embeddings):
        self.lm_head.decoder = new_embeddings

    # 模型前向传播方法
    @add_start_docstrings_to_model_forward(REFORMER_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=CausalLMOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        num_hashes: Optional[int] = None,
        past_buckets_states: Optional[List[Tuple[torch.Tensor]]] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,
    ):
    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, use_cache=None, num_hashes=None, **kwargs
    ):
        # 如果 past_key_values 不为 None，则只保留输入的最后一个 token
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
    
        # 组装输入参数字典
        inputs_dict = {
            "input_ids": input_ids,
            "past_buckets_states": past_key_values,
            "use_cache": use_cache,
            "num_hashes": num_hashes,
        }
    
        # 返回输入参数字典
        return inputs_dict
    # 重新排序缓存中的过去键值对
    def _reorder_cache(self, past_key_values, beam_idx):
        # 存储重新排序后的过去桶状态和隐藏状态
        reord_past_buckets_states = []
        # 遍历每一层的过去键值对
        for layer_past in past_key_values:
            # 如果存在桶状态
            if layer_past[0] is not None:
                # 根据给定的索引重新排序桶状态
                reord_buckets = layer_past[0].index_select(0, beam_idx.to(layer_past[0].device))
            else:
                # 如果不存在桶状态，则置为 None
                reord_buckets = None

            # 根据给定的索引重新排序隐藏状态
            reord_hidden_states = layer_past[1].index_select(0, beam_idx.to(layer_past[1].device))
            # 将重新排序后的桶状态和隐藏状态添加到列表中
            reord_past_buckets_states.append((reord_buckets, reord_hidden_states))
        # 返回重新排序后的过去桶状态和隐藏状态
        return reord_past_buckets_states
# 添加模型开始时的文档字符串，用于描述 ReformerForMaskedLM 的功能
@add_start_docstrings("""Reformer Model with a `language modeling` head on top.""", REFORMER_START_DOCSTRING)
class ReformerForMaskedLM(ReformerPreTrainedModel):
    # 定义权重共享的关键字
    _tied_weights_keys = ["lm_head.decoder.weight", "lm_head.decoder.bias"]

    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)
        # 确保配置选项 is_decoder 为 False，因为这是一个双向的自注意力模型
        assert not config.is_decoder, (
            "If you want to use `ReformerForMaskedLM` make sure `config.is_decoder=False` for bi-directional"
            " self-attention."
        )
        # 创建 Reformer 模型对象
        self.reformer = ReformerModel(config)
        # 创建 Reformer 语言建模头部对象
        self.lm_head = ReformerOnlyLMHead(config)

        # 初始化权重并执行最终处理
        self.post_init()

    # 获取输出词嵌入层
    def get_output_embeddings(self):
        return self.lm_head.decoder

    # 设置输出词嵌入层
    def set_output_embeddings(self, new_embeddings):
        self.lm_head.decoder = new_embeddings

    # 添加模型前向传播方法的文档字符串
    @add_start_docstrings_to_model_forward(REFORMER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=MaskedLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        num_hashes: Optional[int] = None,
        labels: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
@add_start_docstrings(
    """
    Reformer Model transformer with a sequence classification/regression head on top (a linear layer on top of the
    pooled output) e.g. for GLUE tasks.
    """,
    REFORMER_START_DOCSTRING,
)
class ReformerForSequenceClassification(ReformerPreTrainedModel):
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)
        # 设置分类任务的标签数量
        self.num_labels = config.num_labels
        # 存储配置
        self.config = config

        # 创建 Reformer 模型对象
        self.reformer = ReformerModel(config)
        # 创建 Reformer 分类头部对象
        self.classifier = ReformerClassificationHead(config)
        # 如果配置选项 is_decoder 为 True，则显示警告信息
        if config.is_decoder is True:
            logger.warning("You might want to disable causal masking for sequence classification")

        # 初始化权重并执行最终处理
        self.post_init()

    # 添加模型前向传播方法的文档字符串
    @add_start_docstrings_to_model_forward(REFORMER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=SequenceClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        num_hashes: Optional[int] = None,
        labels: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
class ReformerClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(2 * config.hidden_size, config.hidden_size)
        # 定义分类器的 dropout，如果未指定则使用隐藏层 dropout
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, hidden_states, **kwargs):
        # 取隐藏状态的第一个位置，对应于<s> token（相当于[CLS]）
        hidden_states = hidden_states[:, 0, :]
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = torch.tanh(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.out_proj(hidden_states)
        return hidden_states


@add_start_docstrings(
    """
    Reformer Model with a span classification head on top for extractive question-answering tasks like SQuAD / TriviaQA
    ( a linear layer on top of hidden-states output to compute `span start logits` and `span end logits`.
    """,
    REFORMER_START_DOCSTRING,
)
class ReformerForQuestionAnswering(ReformerPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.reformer = ReformerModel(config)
        # 因为我们使用可逆的残差层，所以是 2 * config.hidden_size
        self.qa_outputs = nn.Linear(2 * config.hidden_size, config.num_labels)

        # 初始化权重并进行最终处理
        self.post_init()

    @add_start_docstrings_to_model_forward(REFORMER_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=QuestionAnsweringModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        num_hashes: Optional[int] = None,
        start_positions: Optional[torch.Tensor] = None,
        end_positions: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
```