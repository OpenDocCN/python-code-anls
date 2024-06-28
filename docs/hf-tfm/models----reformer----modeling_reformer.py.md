# `.\models\reformer\modeling_reformer.py`

```
# 定义一个函数 _stable_argsort，用于稳定地对输入的向量进行排序操作
def _stable_argsort(vector, dim):
    # 此函数对向量进行缩放以确保 torch.argsort 的稳定性
    # torch.argsort 在默认情况下不是稳定的排序算法
    # 创建一个偏移量张量，其值从 0 到向量的长度，用于稳定化排序
    scale_offset = torch.arange(vector.shape[dim], device=vector.device).view(1, 1, -1)
    scale_offset = scale_offset.expand(vector.shape)
    # 根据给定的维度（dim），对输入向量（vector）进行缩放和排序
    scaled_vector = vector.shape[dim] * vector + (scale_offset % vector.shape[dim])
    # 使用PyTorch中的argsort函数对缩放后的向量进行排序，按照指定的维度（dim）排序
    return torch.argsort(scaled_vector, dim=dim)
def _get_least_common_mult_chunk_len(config):
    attn_types = config.attn_layers  # 获取配置中的注意力类型列表
    attn_types_set = set(attn_types)  # 将注意力类型转换为集合，去除重复项
    if len(attn_types_set) == 1 and attn_types[0] == "lsh":  # 如果只有一种注意力类型且为'lsh'
        return config.lsh_attn_chunk_length  # 返回配置中的LSH注意力块长度
    elif len(attn_types_set) == 1 and attn_types[0] == "local":  # 如果只有一种注意力类型且为'local'
        return config.local_attn_chunk_length  # 返回配置中的本地注意力块长度
    elif len(attn_types_set) == 2 and attn_types_set == {"lsh", "local"}:  # 如果有两种注意力类型且分别为'lsh'和'local'
        return np.lcm(config.lsh_attn_chunk_length, config.local_attn_chunk_length)  # 返回LSH和本地注意力块长度的最小公倍数
    else:
        raise NotImplementedError(
            f"Only attn layer types 'lsh' and 'local' exist, but `config.attn_layers`: {config.attn_layers}. Select "
            "attn layer types from ['lsh', 'local'] only."
        )  # 抛出未实现的错误，提示只能选择 'lsh' 和 'local' 两种类型的注意力层


def _get_min_chunk_len(config):
    attn_types = config.attn_layers  # 获取配置中的注意力类型列表
    attn_types_set = set(attn_types)  # 将注意力类型转换为集合，去除重复项
    if len(attn_types_set) == 1 and attn_types[0] == "lsh":  # 如果只有一种注意力类型且为'lsh'
        return config.lsh_attn_chunk_length  # 返回配置中的LSH注意力块长度
    elif len(attn_types_set) == 1 and attn_types[0] == "local":  # 如果只有一种注意力类型且为'local'
        return config.local_attn_chunk_length  # 返回配置中的本地注意力块长度
    elif len(attn_types_set) == 2 and attn_types_set == {"lsh", "local"}:  # 如果有两种注意力类型且分别为'lsh'和'local'
        return min(config.lsh_attn_chunk_length, config.local_attn_chunk_length)  # 返回LSH和本地注意力块长度的最小值
    else:
        raise NotImplementedError(
            f"Only attn layer types 'lsh' and 'local' exist, but `config.attn_layers`: {config.attn_layers}. Select "
            "attn layer types from ['lsh', 'local'] only."
        )  # 抛出未实现的错误，提示只能选择 'lsh' 和 'local' 两种类型的注意力层


class AxialPositionEmbeddings(nn.Module):
    """
    Constructs axial position embeddings. Useful for very long input sequences to save memory and time.
    """

    def __init__(self, config):
        super().__init__()
        self.axial_pos_shape = config.axial_pos_shape  # 设置轴向位置嵌入的形状
        self.axial_pos_embds_dim = config.axial_pos_embds_dim  # 设置轴向位置嵌入的维度
        self.dropout = config.hidden_dropout_prob  # 设置隐藏层的dropout比例

        self.least_common_mult_chunk_length = _get_least_common_mult_chunk_len(config)  # 计算最小公倍数块长度
        self.weights = nn.ParameterList()  # 初始化参数列表

        if sum(self.axial_pos_embds_dim) != config.hidden_size:  # 如果轴向位置嵌入的维度之和不等于隐藏层大小
            raise ValueError(
                f"Make sure that config.axial_pos_embds factors: {self.axial_pos_embds_dim} sum to "
                f"config.hidden_size: {config.hidden_size}"
            )  # 抛出值错误，提示轴向位置嵌入的维度之和应等于隐藏层大小

        # create weights
        for axis, axial_pos_embd_dim in enumerate(self.axial_pos_embds_dim):
            # create expanded shapes
            ax_shape = [1] * len(self.axial_pos_shape)  # 创建轴向形状的扩展列表
            ax_shape[axis] = self.axial_pos_shape[axis]  # 设置当前轴的形状
            ax_shape = tuple(ax_shape) + (axial_pos_embd_dim,)  # 转换为元组并添加嵌入维度

            # create tensor and init
            self.weights.append(nn.Parameter(torch.ones(ax_shape, dtype=torch.float32)))  # 创建参数张量并初始化


class PositionEmbeddings(nn.Module):
    """Constructs conventional position embeddings of shape `[max_pos_embeddings, hidden_size]`."""
    # 初始化方法，用于初始化对象
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()
        # 将配置中的隐藏层dropout概率赋值给对象的dropout属性
        self.dropout = config.hidden_dropout_prob
        # 创建一个Embedding层，用于位置ID到隐藏层大小的嵌入映射
        self.embedding = nn.Embedding(config.max_position_embeddings, config.hidden_size)

    # 前向传播方法，定义了数据如何在模型中前向传播
    def forward(self, position_ids):
        # 将位置ID转换为位置嵌入向量
        position_embeddings = self.embedding(position_ids)
        # 对位置嵌入向量进行dropout操作，根据self.training确定是否训练模式
        position_embeddings = nn.functional.dropout(position_embeddings, p=self.dropout, training=self.training)
        # 返回处理后的位置嵌入向量作为模型的输出
        return position_embeddings
class ReformerEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        self.max_position_embeddings = config.max_position_embeddings  # 初始化最大位置嵌入数
        self.dropout = config.hidden_dropout_prob  # 初始化隐藏层dropout概率

        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)  # 创建词嵌入层
        self.position_embeddings = (
            AxialPositionEmbeddings(config) if config.axial_pos_embds else PositionEmbeddings(config)
        )  # 根据配置选择轴向或普通位置嵌入层

    def forward(self, input_ids=None, position_ids=None, inputs_embeds=None, start_idx_pos_encodings=0):
        if input_ids is not None:
            input_shape = input_ids.size()  # 获取输入ids的形状
            device = input_ids.device  # 获取输入ids所在设备
        else:
            input_shape = inputs_embeds.size()[:-1]  # 获取嵌入输入的形状（去掉最后一维）
            device = inputs_embeds.device  # 获取嵌入输入所在设备

        seq_length = input_shape[1]  # 获取序列长度
        if position_ids is None:
            position_ids = torch.arange(
                start_idx_pos_encodings, start_idx_pos_encodings + seq_length, dtype=torch.long, device=device
            )  # 创建位置ids，如果未提供的话
            position_ids = position_ids.unsqueeze(0).expand(input_shape)  # 扩展位置ids到输入形状

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)  # 使用词嵌入层获取嵌入输入

        if position_ids.shape[-1] > self.max_position_embeddings:
            raise ValueError(
                f"Sequence Length: {position_ids.shape[-1]} has to be less or equal than "
                f"config.max_position_embeddings {self.max_position_embeddings}."
            )  # 检查位置ids的长度是否超过最大位置嵌入数，如果超过则抛出异常

        # dropout
        embeddings = nn.functional.dropout(inputs_embeds, p=self.dropout, training=self.training)  # 应用dropout

        # add positional embeddings
        position_embeddings = self.position_embeddings(position_ids)  # 添加位置嵌入
        embeddings = embeddings + position_embeddings  # 将位置嵌入加到词嵌入上
        return embeddings


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
        if num_chunks_before == 0 and num_chunks_after == 0:
            return vectors  # 如果没有前后的chunk，直接返回向量

        slices = []
        for i in range(-num_chunks_before, num_chunks_after + 1):
            if i == 0:
                slices.append(vectors)  # 中心chunk直接添加
            else:
                slices.append(torch.cat([vectors[:, :, i:, ...], vectors[:, :, :i, ...]], dim=2))  # 添加前后的chunk
        return torch.cat(slices, dim=3)  # 合并所有chunk并返回
    # 将输入张量 x 的最后一维划分为 num_attn_heads 和 attn_head_size 维度，并重新构造张量形状
    def _split_hidden_size_dim(self, x, num_attn_heads, attn_head_size):
        new_x_shape = x.size()[:-1] + (num_attn_heads, attn_head_size)
        x = x.view(*new_x_shape)
        return x.transpose(2, 1)

    # 将输入张量 x 的第三和第四维度互换，然后将其余维度展平为 hidden_size 维度
    def _merge_hidden_size_dims(self, x, num_attn_heads, attn_head_size):
        x = x.permute(0, 2, 1, 3)
        return torch.reshape(x, (x.size()[0], -1, num_attn_heads * attn_head_size))

    # 将输入张量 vectors 的序列长度维度划分为 dim_factor_1 和 dim_factor_2 维度
    # 如果 vectors 是四维张量，则还需添加 attn_head_size 维度
    def _split_seq_length_dim_to(self, vectors, dim_factor_1, dim_factor_2, num_attn_heads, attn_head_size=None):
        batch_size = vectors.shape[0]
        split_dim_shape = (batch_size, num_attn_heads, dim_factor_1, dim_factor_2)

        if len(vectors.shape) == 4:
            return torch.reshape(vectors, split_dim_shape + (attn_head_size,))
        elif len(vectors.shape) == 3:
            return torch.reshape(vectors, split_dim_shape)
        else:
            raise ValueError(f"Input vector rank should be one of [3, 4], but is: {len(vectors.shape)}")
# 定义一个名为 LSHSelfAttention 的类，继承自 nn.Module 和 EfficientAttentionMixin
class LSHSelfAttention(nn.Module, EfficientAttentionMixin):
    # 初始化方法，接收一个 config 参数
    def __init__(self, config):
        super().__init__()
        # 将传入的配置保存到实例变量 self.config 中
        self.config = config

        # 从配置中获取并设置各种参数
        self.chunk_length = config.lsh_attn_chunk_length  # LSH 注意力的块长度
        self.num_hashes = config.num_hashes  # 哈希函数的数量
        self.num_buckets = config.num_buckets  # 桶的数量
        self.num_chunks_before = config.lsh_num_chunks_before  # 注意力前的块数量
        self.num_chunks_after = config.lsh_num_chunks_after  # 注意力后的块数量
        self.hash_seed = config.hash_seed  # 哈希种子
        self.is_decoder = config.is_decoder  # 是否为解码器
        self.max_position_embeddings = config.max_position_embeddings  # 最大位置编码

        self.dropout = config.lsh_attention_probs_dropout_prob  # 注意力概率 dropout

        self.num_attention_heads = config.num_attention_heads  # 注意力头的数量
        self.attention_head_size = config.attention_head_size  # 每个注意力头的大小
        self.all_head_size = self.num_attention_heads * self.attention_head_size  # 所有注意力头的总大小
        self.hidden_size = config.hidden_size  # 隐藏层大小

        # 定义查询和键的投影矩阵，无偏置
        self.query_key = nn.Linear(self.hidden_size, self.all_head_size, bias=False)
        self.value = nn.Linear(self.hidden_size, self.all_head_size, bias=False)

        # 注册缓冲区，保存不同精度的掩码值
        self.register_buffer("self_mask_value_float16", torch.tensor(-1e3), persistent=False)
        self.register_buffer("self_mask_value_float32", torch.tensor(-1e5), persistent=False)
        self.register_buffer("mask_value_float16", torch.tensor(-1e4), persistent=False)
        self.register_buffer("mask_value_float32", torch.tensor(-1e9), persistent=False)

    # 前向传播方法
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
        # 基于每个注意力头的查询矩阵
        def _query_per_attn_head(self, hidden_states):
            # 重塑和转置查询矩阵，以便用于每个注意力头
            per_head_query_key = self.query_key.weight.reshape(
                self.num_attention_heads, self.attention_head_size, self.hidden_size
            ).transpose(-2, -1)
            # 使用 einsum 计算查询向量
            query_key_vectors = torch.einsum("balh,ahr->balr", hidden_states, per_head_query_key)
            return query_key_vectors

        # 基于每个注意力头的值矩阵
        def _value_per_attn_head(self, hidden_states):
            # 重塑和转置值矩阵，以便用于每个注意力头
            per_head_value = self.value.weight.reshape(
                self.num_attention_heads, self.attention_head_size, self.hidden_size
            ).transpose(-2, -1)
            # 使用 einsum 计算值向量
            value_vectors = torch.einsum("balh,ahr->balr", hidden_states, per_head_value)
            return value_vectors
    def _get_sorted_bucket_idx_and_undo_sorted_bucket_idx(self, sequence_length, buckets, num_hashes):
        # 不需要计算梯度
        with torch.no_grad():
            # 基于哈希进行排序
            sorted_bucket_idx = _stable_argsort(buckets, dim=-1)

            # 创建简单的索引用于散开操作，以便进行反排序
            indices = (
                torch.arange(sorted_bucket_idx.shape[-1], device=buckets.device)
                .view(1, 1, -1)
                .expand(sorted_bucket_idx.shape)
            )

            # 获取反排序的索引
            undo_sorted_bucket_idx = sorted_bucket_idx.new(*sorted_bucket_idx.size())
            undo_sorted_bucket_idx.scatter_(-1, sorted_bucket_idx, indices)

        return sorted_bucket_idx, undo_sorted_bucket_idx

    def _set_num_buckets(self, sequence_length):
        # 根据论文推荐，`num_buckets` 应该设置为 2 * sequence_length // chunk_length
        num_buckets_pow_2 = (2 * (sequence_length // self.chunk_length)).bit_length() - 1
        # 确保 buckets 是2的幂
        num_buckets = 2**num_buckets_pow_2

        # 如果 `num_buckets` 太大，则进行因式分解
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
    ):
        # 这是一个方法用于注意力机制的实现，处理给定的向量和掩码等

    def _compute_attn_mask(
        self, query_indices, key_indices, attention_mask, query_key_dot_shape, do_standard_self_attention
    ):
        # 这是一个方法用于计算注意力掩码，根据给定的索引、掩码和其他参数进行操作
        # attention mask for LSH
        if attention_mask is not None:
            # 如果存在注意力掩码，则将其转换为布尔型，并扩展维度以匹配LSH的顺序
            attention_mask = attention_mask.to(torch.bool)[:, None, :]
            if not do_standard_self_attention:
                # 如果不是标准的自注意力机制，则需要将注意力掩码扩展以适应key_value_bucket_idx的形状
                attention_mask = attention_mask[:, None, :]
                attention_mask = attention_mask.expand(query_indices.shape[:-1] + (-1,))
                # 从LSH排序后的key_indices中提取注意力掩码
                attention_mask = torch.gather(attention_mask, -1, key_indices)

            # 将注意力掩码扩展以适应query_key_dot_shape的形状
            attention_mask = attention_mask.unsqueeze(-2).expand(query_key_dot_shape)

        # Causal mask
        if self.is_decoder is True:
            # 如果是解码器，创建因果掩码，使得查询的索引大于等于键的索引的位置为True
            causal_mask = torch.ge(query_indices.unsqueeze(-1), key_indices.unsqueeze(-2)).to(query_indices.device)

            # 如果注意力掩码不为None，则将因果掩码与注意力掩码相乘
            if attention_mask is not None:
                attention_mask = causal_mask * attention_mask
            else:
                attention_mask = causal_mask

        # 返回最终的注意力掩码
        return attention_mask

    def _get_relevant_hid_states_and_buckets(
        self, query_vectors, attention_mask, num_hashes, hidden_states, past_states, past_buckets
    ):
        # 获取相关隐藏状态和存储桶
        # 这个函数用于从查询向量中获取相关的隐藏状态和存储桶

    def _expand_to_indices_in_relevant_chunk(self, indices, sequence_length):
        # 获取相关块中的索引并扩展
        # 根据给定的索引确定块的起始位置和大小，并通过arange添加正确的块偏移量

        # 计算块的起始索引并扩展
        start_indices_chunk = ((indices[:, -1] // self.chunk_length) - self.num_chunks_before) * self.chunk_length
        total_chunk_size = self.chunk_length * (1 + self.num_chunks_before + self.num_chunks_after)

        expanded_start_indices = start_indices_chunk.unsqueeze(-1).expand(indices.shape[0], total_chunk_size)
        
        # 创建块序列索引，确保通过取模运算满足循环逻辑
        chunk_sequence_indices = expanded_start_indices + torch.arange(
            total_chunk_size, device=indices.device, dtype=torch.long
        ).unsqueeze(0).expand(indices.shape[0], total_chunk_size)

        chunk_sequence_indices = chunk_sequence_indices.flatten() % sequence_length

        # 扩展索引并设置正确的索引
        indices = indices.unsqueeze(1).expand((indices.shape[0], total_chunk_size, -1)).flatten(0, 1).clone()
        indices[:, -1] = chunk_sequence_indices

        return indices

    def _len_and_dim_norm(self, vectors, sqrt_num):
        """
        length and attention head size dim normalization
        """
        # 对向量进行长度和注意力头尺寸维度归一化处理

        # 首先进行长度归一化
        vectors = self._len_norm(vectors)
        vectors = vectors / sqrt_num
        return vectors

    def _len_norm(self, x, epsilon=1e-6):
        """
        length normalization
        """
        # 长度归一化处理

        # 计算方差
        variance = torch.mean(x**2, -1, keepdim=True)
        # 根据方差进行归一化处理
        norm_x = x * torch.rsqrt(variance + epsilon)
        return norm_x
    # 定义一个私有方法 `_gather_by_expansion`，用于扩展 `vectors` 和 `idxs` 的维度，并根据所有哈希值进行聚合
    def _gather_by_expansion(self, vectors, idxs, num_hashes):
        # 将 `idxs` 在最后一个维度上增加一个维度，并在所有维度上进行扩展，以便与 `vectors` 的维度匹配
        expanded_idxs = idxs.unsqueeze(-1).expand(-1, -1, -1, self.attention_head_size)
        # 将 `vectors` 在第三个维度上重复 `num_hashes` 次，以便与 `expanded_idxs` 的维度匹配
        vectors = vectors.repeat(1, 1, num_hashes, 1)
        # 使用 `torch.gather` 函数根据 `expanded_idxs` 在第三个维度上聚合 `vectors` 的数据
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
        """
        Performs the forward pass for local self-attention mechanism.

        Args:
            hidden_states (torch.Tensor): Input embeddings or hidden states.
            attention_mask (torch.Tensor, optional): Mask indicating which elements should be attended to.
            head_mask (torch.Tensor, optional): Mask indicating heads to be masked out.
            past_buckets_states (torch.Tensor, optional): States from previous attention buckets.
            use_cache (bool, optional): Whether to use caching mechanism.
            output_attentions (bool, optional): Whether to output attention scores.

        Returns:
            torch.Tensor: Output embeddings or hidden states.
        """

    def _compute_attn_mask(
        self, query_indices, key_indices, attention_mask, query_key_dots_shape, do_standard_self_attention
    ):
        """
        Computes the attention mask based on query and key indices.

        Args:
            query_indices (torch.Tensor): Indices for queries.
            key_indices (torch.Tensor): Indices for keys.
            attention_mask (torch.Tensor): Attention mask.
            query_key_dots_shape (Tuple): Shape of the query-key dot product.
            do_standard_self_attention (bool): Whether to perform standard self-attention.

        Returns:
            torch.Tensor: Computed attention mask.
        """
        # chunk attention mask and look before and after
        # 如果存在注意力掩码，则将其转换为布尔型并添加维度以适应后续操作
        if attention_mask is not None:
            attention_mask = attention_mask.to(torch.bool)[:, None, :]

            # 如果不使用标准的自注意力机制，则分割注意力掩码并在最后一个维度上添加分块前后的注意力
            if not do_standard_self_attention:
                attention_mask = self._split_seq_length_dim_to(attention_mask, -1, self.chunk_length, 1)
                attention_mask = self._look_adjacent(attention_mask, self.num_chunks_before, self.num_chunks_after)

            # 创建注意力掩码
            attention_mask = attention_mask.unsqueeze(-2).expand(query_key_dots_shape)

        # Causal mask
        # 如果是解码器，创建因果注意力掩码
        if self.is_decoder is True:
            causal_mask = torch.ge(query_indices.unsqueeze(-1), key_indices.unsqueeze(-2)).to(query_indices.device)

            # 如果注意力掩码不为空，则将因果掩码与注意力掩码相乘
            if attention_mask is not None:
                attention_mask = causal_mask * attention_mask
            else:
                attention_mask = causal_mask

        # 返回最终的注意力掩码
        return attention_mask


    @staticmethod
    def _retrieve_relevant_hidden_states(previous_hidden_states, chunk_length, num_chunks_before):
        # 计算需要检索的相关隐藏状态的起始位置
        start_position = ((previous_hidden_states.shape[1] // chunk_length) - num_chunks_before) * chunk_length
        # 返回从起始位置开始的相关隐藏状态
        return previous_hidden_states[:, start_position:]
class ReformerSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        all_head_size = config.num_attention_heads * config.attention_head_size
        self.dropout = config.hidden_dropout_prob  # 设置dropout比率

        self.dense = nn.Linear(all_head_size, config.hidden_size, bias=False)  # 创建线性层，将注意力头的输出映射到隐藏层大小

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)  # 前向传播中，将隐藏状态输入到线性层中
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)  # 使用dropout进行正则化
        return hidden_states  # 返回处理后的隐藏状态


class ReformerAttention(nn.Module):
    def __init__(self, config, layer_id=0):
        super().__init__()
        self.layer_id = layer_id  # 层的编号
        self.attn_layers = config.attn_layers  # 注意力层列表

        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)  # 应用Layer normalization

        # 根据配置选择合适的自注意力机制
        if len(set(self.attn_layers)) == 1 and self.attn_layers[0] == "lsh":
            self.self_attention = LSHSelfAttention(config)  # 使用LSH自注意力
        elif len(set(self.attn_layers)) == 1 and self.attn_layers[0] == "local":
            self.self_attention = LocalSelfAttention(config)  # 使用局部自注意力
        elif len(set(self.attn_layers)) == 2 and set(self.attn_layers) == {"lsh", "local"}:
            # 如果同时支持LSH和局部注意力，则根据层的编号选择正确的注意力机制
            if self.attn_layers[self.layer_id] == "lsh":
                self.self_attention = LSHSelfAttention(config)
            else:
                self.self_attention = LocalSelfAttention(config)
        else:
            # 抛出未实现错误，说明配置不支持的注意力类型
            raise NotImplementedError(
                f"Only attn layer types 'lsh' and 'local' exist, but got `config.attn_layers`: {self.attn_layers}. "
                "Select attn layer types from ['lsh', 'local'] only."
            )
        
        self.output = ReformerSelfOutput(config)  # 创建自注意力层输出对象

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
        ):
            # 对隐藏状态进行层归一化处理
            hidden_states = self.layer_norm(hidden_states)

            # 确保缓存的隐藏状态在反向传播时设置为None
            if past_buckets_states is not None:
                past_buckets_states_layer = past_buckets_states[self.layer_id]
            else:
                past_buckets_states_layer = None

            # 如果需要，使用缓存的桶进行反向传播，用于LSHSelfAttention
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

            # 如果self_attention_outputs具有"buckets"属性，则将其分配给buckets变量
            if hasattr(self_attention_outputs, "buckets"):
                buckets = self_attention_outputs.buckets
            else:
                buckets = None

            # 如果需要，将隐藏状态缓存以供将来使用
            if use_cache:
                if past_buckets_states[self.layer_id][0] is None:
                    # 填充的输入不应该被缓存
                    past_buckets = (
                        buckets[:, :, :, :orig_sequence_length]
                        if (buckets is not None and orig_sequence_length > 1)
                        else buckets
                    )
                else:
                    past_buckets = torch.cat([past_buckets_states[self.layer_id][0], buckets], dim=-1)

                if past_buckets_states[self.layer_id][1] is None:
                    # 填充的输入不应该被缓存
                    past_states = hidden_states[:, :orig_sequence_length]
                else:
                    past_states = torch.cat([past_buckets_states[self.layer_id][1], hidden_states], dim=1)

                past_buckets_states[self.layer_id] = (past_buckets, past_states)

            # 计算注意力前馈输出
            attention_output = self.output(self_attention_outputs.hidden_states)

            # 返回AttentionOutput对象，包含注意力机制的输出
            return AttentionOutput(
                hidden_states=attention_output,
                attention_probs=self_attention_outputs.attention_probs,
                buckets=buckets,
            )
class ReformerFeedForwardDense(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dropout = config.hidden_dropout_prob  # 从配置中获取隐藏层dropout概率

        if isinstance(config.hidden_act, str):
            self.act_fn = ACT2FN[config.hidden_act]  # 如果隐藏层激活函数是字符串，从预定义映射中获取对应的函数
        else:
            self.act_fn = config.hidden_act  # 否则直接使用配置中的激活函数

        self.dense = nn.Linear(config.hidden_size, config.feed_forward_size)  # 创建线性层

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)  # 输入隐藏状态经过线性层
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)  # 使用dropout对隐藏状态进行处理
        hidden_states = self.act_fn(hidden_states)  # 使用激活函数对处理后的隐藏状态进行非线性变换
        return hidden_states  # 返回处理后的隐藏状态


class ReformerFeedForwardOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dropout = config.hidden_dropout_prob  # 从配置中获取隐藏层dropout概率

        self.dense = nn.Linear(config.feed_forward_size, config.hidden_size)  # 创建线性层

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)  # 输入隐藏状态经过线性层
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)  # 使用dropout对隐藏状态进行处理
        return hidden_states  # 返回处理后的隐藏状态


class ChunkReformerFeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward  # 从配置中获取前馈层的分块大小
        self.seq_len_dim = 1  # 序列长度的维度设定为1

        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)  # 创建Layer Normalization层
        self.dense = ReformerFeedForwardDense(config)  # 创建前馈层的Dense层
        self.output = ReformerFeedForwardOutput(config)  # 创建前馈层的输出层

    def forward(self, attention_output):
        return apply_chunking_to_forward(
            self.forward_chunk,
            self.chunk_size_feed_forward,
            self.seq_len_dim,
            attention_output,
        )

    def forward_chunk(self, hidden_states):
        hidden_states = self.layer_norm(hidden_states)  # 对隐藏状态进行Layer Normalization
        hidden_states = self.dense(hidden_states)  # 输入隐藏状态经过前馈层的Dense层
        return self.output(hidden_states)  # 返回前馈层的输出


class ReformerLayer(nn.Module):
    def __init__(self, config, layer_id=0):
        super().__init__()
        self.attention = ReformerAttention(config, layer_id)  # 创建ReformerAttention层，用于注意力机制
        # dropout requires to have the same
        # seed for forward and backward pass
        self.attention_seed = None
        self.feed_forward_seed = None

        self.feed_forward = ChunkReformerFeedForward(config)  # 创建分块前馈层
    def _init_attention_seed(self):
        """
        This function sets a new seed for the attention layer to make dropout deterministic for both forward calls: 1
        normal forward call and 1 forward call in backward to recalculate activations.
        """

        # randomize seeds
        # 指定一个新的种子给注意力层，以便在前向调用（普通的前向调用和反向调用中的前向调用）中使dropout具有确定性。

        # use cuda generator if available
        # 如果存在 CUDA 生成器，则使用它
        if hasattr(torch.cuda, "default_generators") and len(torch.cuda.default_generators) > 0:
            # GPU
            device_idx = torch.cuda.current_device()
            self.attention_seed = torch.cuda.default_generators[device_idx].seed()
        else:
            # CPU
            self.attention_seed = int(torch.seed() % sys.maxsize)

        # 设置 PyTorch 的随机种子
        torch.manual_seed(self.attention_seed)

    def _init_feed_forward_seed(self):
        """
        This function sets a new seed for the feed forward layer to make dropout deterministic for both forward calls:
        1 normal forward call and 1 forward call in backward to recalculate activations.
        """
        # randomize seeds
        # 指定一个新的种子给前馈层，以便在前向调用（普通的前向调用和反向调用中的前向调用）中使dropout具有确定性。

        # use cuda generator if available
        # 如果存在 CUDA 生成器，则使用它
        if hasattr(torch.cuda, "default_generators") and len(torch.cuda.default_generators) > 0:
            # GPU
            device_idx = torch.cuda.current_device()
            self.feed_forward_seed = torch.cuda.default_generators[device_idx].seed()
        else:
            # CPU
            self.feed_forward_seed = int(torch.seed() % sys.maxsize)

        # 设置 PyTorch 的随机种子
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
    ):
        # 在没有梯度的情况下执行代码块
        with torch.no_grad():
            # 每次前向传播时采样不同的种子
            # 用于dropout，并保存在反向传播的前向函数中
            # 以确保正确的dropout效果
            if self.training:
                self._init_attention_seed()

            # 执行注意力计算
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
            # 获取注意力输出的隐藏状态
            attn_output = attn_outputs.hidden_states

            # 实现RevNet（参见https://towardsdatascience.com/illustrating-the-reformer-393575ac6ba0中的图6）
            # Y_1 = X_1 + f(X_2)
            attn_output = prev_attn_output + attn_output

            # 释放内存
            del prev_attn_output

            # 每次前向传播时采样不同的种子
            # 用于dropout，并保存种子以便在反向传播中使用
            # 以确保正确的dropout效果
            if self.training:
                self._init_feed_forward_seed()

            # Y_2 = X_2 + g(Y_1)
            hidden_states = hidden_states + self.feed_forward(attn_output)

        # 返回ReformerOutput对象，包含注意力输出、隐藏状态、注意力概率和buckets
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
        # 实现可逆 ResNets 的反向传播过程。
        # 关于这个工作原理的良好博客文章可以在以下链接找到：
        # 实现 RevNet（参见 https://towardsdatascience.com/illustrating-the-reformer-393575ac6ba0 中的图 6）
        # 这段代码受 https://github.com/lucidrains/reformer-pytorch/blob/master/reformer_pytorch/reversible.py 启发

        # 断言处于训练模式，确保在训练 `ReformerModel` 及其变体时使用 `model.train()` 来将模型置于训练模式
        assert self.training, (
            "If you want to train `ReformerModel` and its variations, make sure to use `model.train()` to put the"
            " model into training mode."
        )

        with torch.enable_grad():
            # 设置下一个注意力输出的梯度需求为True
            next_attn_output.requires_grad = True

            # 设置种子以确保正确的dropout
            torch.manual_seed(self.feed_forward_seed)
            # g(Y_1)
            # 使用 feed_forward 方法计算下一个注意力输出的隐藏状态
            res_hidden_states = self.feed_forward(next_attn_output)
            # 反向传播 g(Y_1) 的梯度到 grad_hidden_states，保留计算图以备后续使用
            res_hidden_states.backward(grad_hidden_states, retain_graph=True)

        with torch.no_grad():
            # X_2 = Y_2 - g(Y_1)
            # 更新隐藏状态，减去 g(Y_1) 的结果
            hidden_states = hidden_states - res_hidden_states
            # 删除 res_hidden_states 变量以释放内存
            del res_hidden_states

            # 累加 next_attn_output 的梯度到 grad_attn_output
            grad_attn_output = grad_attn_output + next_attn_output.grad
            # 清空 next_attn_output 的梯度，以便下一轮使用
            next_attn_output.grad = None

        with torch.enable_grad():
            # 设置隐藏状态的梯度需求为True
            hidden_states.requires_grad = True

            # 设置种子以确保正确的dropout
            torch.manual_seed(self.attention_seed)
            # f(X_2)
            # 使用 attention 方法计算隐藏状态的输出
            # 如果 buckets 不为 None，则使用缓存的 buckets 进行反向传播
            output = self.attention(
                hidden_states=hidden_states,
                head_mask=head_mask,
                attention_mask=attention_mask,
                buckets=buckets,
            ).hidden_states
            # 反向传播 f(X_2) 的梯度到 grad_attn_output，保留计算图以备后续使用
            output.backward(grad_attn_output, retain_graph=True)

        with torch.no_grad():
            # X_1 = Y_1 - f(X_2)
            # 更新注意力输出，减去 f(X_2) 的结果
            attn_output = next_attn_output - output
            # 删除 output 和 next_attn_output 变量以释放内存
            del output, next_attn_output

            # 累加 hidden_states 的梯度到 grad_hidden_states
            grad_hidden_states = grad_hidden_states + hidden_states.grad
            # 清空 hidden_states 的梯度，以便下一轮使用
            hidden_states.grad = None
            # 分离 hidden_states 的计算图，使其不再跟踪梯度
            hidden_states = hidden_states.detach()

        # 返回 ReformerBackwardOutput 对象，其中包括更新后的 attn_output、hidden_states、grad_attn_output 和 grad_hidden_states
        return ReformerBackwardOutput(
            attn_output=attn_output,
            hidden_states=hidden_states,
            grad_attn_output=grad_attn_output,
            grad_hidden_states=grad_hidden_states,
        )
    """
    针对可逆函数的自定义反向传播函数，以防止 PyTorch 执行通常的反向传播。
    通过这种方式确保在前向传播期间不保存内存消耗昂贵的激活值。
    本函数的实现受到 https://github.com/lucidrains/reformer-pytorch/blob/master/reformer_pytorch/reversible.py 的启发。
    """

    @staticmethod
    # 定义静态方法 forward，用于执行前向传播
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
        # 初始化空的所有桶
        all_buckets = ()

        # 将 hidden_states 张量按照最后一个维度分为两部分
        hidden_states, attn_output = torch.chunk(hidden_states, 2, dim=-1)

        # 遍历层和对应的头部掩码
        for layer_id, (layer, layer_head_mask) in enumerate(zip(layers, head_mask)):
            # 如果需要输出隐藏状态，则将当前隐藏状态添加到 all_hidden_states 列表中
            if output_hidden_states is True:
                all_hidden_states.append(hidden_states)

            # 调用层的前向传播函数，获取层的输出
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

            # 更新 attn_output 和 hidden_states
            attn_output = layer_outputs.attn_output
            hidden_states = layer_outputs.hidden_states
            # 将当前层的桶添加到 all_buckets 中
            all_buckets = all_buckets + (layer_outputs.buckets,)

            # 如果需要输出注意力权重，则将当前层的注意力权重添加到 all_attentions 列表中
            if output_attentions:
                all_attentions.append(layer_outputs.attention_probs)

        # 如果需要输出隐藏状态，则将最后一个隐藏状态添加到 all_hidden_states 列表中
        if output_hidden_states is True:
            all_hidden_states.append(hidden_states)

        # 将 attn_output 和 hidden_states 的梯度信息保存到 ctx 中，以备反向传播使用
        ctx.save_for_backward(attn_output.detach(), hidden_states.detach())
        ctx.layers = layers
        ctx.all_buckets = all_buckets
        ctx.head_mask = head_mask
        ctx.attention_mask = attention_mask

        # 将 attn_output 和 hidden_states 拼接在一起作为输出
        return torch.cat([attn_output, hidden_states], dim=-1)

    @staticmethod
    def backward(ctx, grad_hidden_states):
        # 将 grad_hidden_states 按最后一个维度分成两部分
        grad_attn_output, grad_hidden_states = torch.chunk(grad_hidden_states, 2, dim=-1)

        # 从上下文 ctx 中获取保存的张量参数
        attn_output, hidden_states = ctx.saved_tensors

        # 创建包含各种参数的元组 output
        output = ReformerBackwardOutput(
            attn_output=attn_output,
            hidden_states=hidden_states,
            grad_attn_output=grad_attn_output,
            grad_hidden_states=grad_hidden_states,
        )

        # 释放内存，删除不再需要的变量
        del grad_attn_output, grad_hidden_states, attn_output, hidden_states

        # 从上下文中获取反向传播所需的各个参数
        layers = ctx.layers
        all_buckets = ctx.all_buckets
        head_mask = ctx.head_mask
        attention_mask = ctx.attention_mask

        # 对每一层进行反向传播
        for idx, layer in enumerate(layers[::-1]):
            # 弹出最后一个 buckets 并从堆栈中移除
            buckets = all_buckets[-1]
            all_buckets = all_buckets[:-1]

            # 执行反向传播
            output = layer.backward_pass(
                next_attn_output=output.attn_output,
                hidden_states=output.hidden_states,
                grad_attn_output=output.grad_attn_output,
                grad_hidden_states=output.grad_hidden_states,
                head_mask=head_mask[len(layers) - idx - 1],
                attention_mask=attention_mask,
                buckets=buckets,
            )

        # 断言所有 buckets 必须为空元组，用于确认反向传播后所有 buckets 已清空
        assert all_buckets == (), "buckets have to be empty after backpropagation"

        # 将 grad_attn_output 和 grad_hidden_states 沿最后一个维度拼接
        grad_hidden_states = torch.cat([output.grad_attn_output, output.grad_hidden_states], dim=-1)

        # 返回与 forward() 中参数个数相匹配的梯度，其他返回 None
        return grad_hidden_states, None, None, None, None, None, None, None, None, None, None, None
class ReformerEncoder(nn.Module):
    # 定义 Reformer 编码器模型，继承自 nn.Module
    def __init__(self, config):
        super().__init__()
        # 初始化函数，接受配置参数 config

        # 设置 dropout 概率
        self.dropout = config.hidden_dropout_prob

        # 创建多层 ReformerLayer 组成的层列表
        self.layers = nn.ModuleList([ReformerLayer(config, i) for i in range(config.num_hidden_layers)])
        # Reformer 使用 Rev Nets，因此最后一层的输出会被连接起来，
        # 并且对 2 * hidden_size 进行 Layer Norm 处理
        self.layer_norm = nn.LayerNorm(2 * config.hidden_size, eps=config.layer_norm_eps)

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
        # 声明存储所有隐藏状态和注意力权重的列表
        all_hidden_states = []
        all_attentions = []

        # 如果需要的话，初始化缓存的历史桶状态
        if past_buckets_states is None:
            past_buckets_states = [((None), (None)) for i in range(len(self.layers))]

        # 将隐藏状态进行拼接，用于可逆 ResNet
        hidden_states = torch.cat([hidden_states, hidden_states], dim=-1)
        # 调用自定义的可逆函数进行前向传播
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

        # 对拼接后的隐藏状态应用 Layer Norm
        hidden_states = self.layer_norm(hidden_states)

        # 对隐藏状态应用 dropout
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        # 返回 ReformerEncoderOutput 对象，包含隐藏状态、所有隐藏状态列表、所有注意力权重列表和历史桶状态
        return ReformerEncoderOutput(
            hidden_states=hidden_states,
            all_hidden_states=all_hidden_states,
            all_attentions=all_attentions,
            past_buckets_states=past_buckets_states,
        )


class ReformerOnlyLMHead(nn.Module):
    # 定义仅包含语言模型头部的 Reformer 模型，继承自 nn.Module
    def __init__(self, config):
        super().__init__()

        # Reformer 使用 Rev Nets，因此最后一层的输出会被连接起来，
        # 并且对 2 * hidden_size 进行 Layer Norm 处理
        self.seq_len_dim = 1
        self.chunk_size_lm_head = config.chunk_size_lm_head
        # 定义线性层作为解码器，输出大小为 vocab_size
        self.decoder = nn.Linear(2 * config.hidden_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        # 应用分块处理来执行前向传播
        return apply_chunking_to_forward(self.forward_chunk, self.chunk_size_lm_head, self.seq_len_dim, hidden_states)

    def forward_chunk(self, hidden_states):
        # 使用解码器层进行前向传播
        hidden_states = self.decoder(hidden_states)
        return hidden_states
    def _tie_weights(self):
        # 如果两个权重被断开连接（在TPU上或者当偏置被重新调整大小时），用于将它们绑定在一起。
        self.bias = self.decoder.bias
# 定义一个抽象类，用于处理权重初始化以及下载和加载预训练模型的简单接口
class ReformerPreTrainedModel(PreTrainedModel):
    # 指定配置类
    config_class = ReformerConfig
    # 模型名称前缀
    base_model_prefix = "reformer"

    # 返回一个包含虚拟输入数据的字典
    @property
    def dummy_inputs(self):
        input_ids = torch.tensor(DUMMY_INPUTS)  # 创建包含虚拟输入 IDs 的张量
        input_mask = torch.tensor(DUMMY_MASK)   # 创建包含虚拟输入掩码的张量
        dummy_inputs = {
            "input_ids": input_ids,             # 将输入 IDs 放入字典
            "attention_mask": input_mask,       # 将注意力掩码放入字典
        }
        return dummy_inputs                    # 返回虚拟输入字典

    # 初始化模型的权重
    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, AxialPositionEmbeddings):
            # 如果模块是 AxialPositionEmbeddings 类型，则初始化其权重
            for weight in module.weights:
                nn.init.normal_(weight, std=self.config.axial_norm_std)
        elif isinstance(module, nn.Embedding):
            # 如果模块是 nn.Embedding 类型，则初始化其权重和可能的填充索引
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.Linear):
            # 如果模块是 nn.Linear 类型，则初始化其权重和偏置
            # 与 TF 版本稍有不同，这里使用正态分布来初始化权重
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            # 如果模块是 nn.LayerNorm 类型，则初始化其权重和偏置
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


@dataclass
class ReformerModelOutput(ModelOutput):
    """
    Output type of [`ReformerModel`].
    """
    # 定义函数参数及其类型注释
    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, num_predict, hidden_size)`):
            模型最后一层的隐藏状态序列。
    
            `num_predict` 对应于 `target_mapping.shape[1]`。如果 `target_mapping` 是 `None`，那么 `num_predict` 对应于 `sequence_length`。
        past_buckets_states (`List[Tuple(torch.LongTensor, torch.FloatTensor)]`, *optional*, 在 `use_cache=True` 或 `config.use_cache=True` 时返回):
            包含预先计算的桶和隐藏状态的列表，用于加速顺序解码。
    
            每个元素是一个元组，第一个元素是形状为 `(batch_size, num_heads, num_hashes, sequence_length)` 的先前*桶*，
            第二个元素是形状为 `(batch_size, sequence_length, hidden_size)` 的先前*隐藏状态*。
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, 在 `output_hidden_states=True` 或 `config.output_hidden_states=True` 时返回):
            元组中包含了每层的输出 (`embeddings` 输出的一个和每层输出的一个) 的 `torch.FloatTensor`，形状为 `(batch_size, sequence_length, hidden_size)`。
    
            模型每一层的隐藏状态，加上初始嵌入的输出。
        attentions (`tuple(torch.FloatTensor)`, *optional*, 在 `output_attentions=True` 或 `config.output_attentions=True` 时返回):
            元组中包含了每层的注意力权重 `torch.FloatTensor` (每层一个)，形状为 `(batch_size, num_heads, sequence_length, sequence_length)`。
    
            经过注意力 softmax 后的注意力权重，用于计算自注意力头中的加权平均值。
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

    # 定义 ReformerModelWithLMHeadOutput 类，作为 Reformer 模型输出的数据结构
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_buckets_states: Optional[List[Tuple[torch.LongTensor, torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


REFORMER_START_DOCSTRING = r"""
    Reformer was proposed in [Reformer: The Efficient Transformer](https://arxiv.org/abs/2001.04451) by Nikita Kitaev,
    Łukasz Kaiser, Anselm Levskaya.

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)
"""
# 定义 REFORMER_START_DOCSTRING 字符串，描述 Reformer 模型的起始文档字符串
    # 这是一个 PyTorch 的模型，继承自 [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) 类。
    # 可以像普通的 PyTorch 模块一样使用，关于一般使用和行为的所有问题，请参考 PyTorch 文档。
    
    # 参数：
    #     config ([`ReformerConfig`]): 这是一个模型配置类，包含了模型的所有参数。
    #         使用配置文件初始化模型不会加载与模型关联的权重，只会加载配置信息。
    #         若要加载模型权重，请查看 [`~PreTrainedModel.from_pretrained`] 方法。
"""
REFORMER_INPUTS_DOCSTRING = r"""
"""

# 使用装饰器添加文档字符串，描述模型输出原始隐藏状态的Reformer模型，没有特定的输出头
@add_start_docstrings(
    "The bare Reformer Model transformer outputting raw hidden-states without any specific head on top.",
    REFORMER_START_DOCSTRING,
)
# 定义ReformerModel类，继承自ReformerPreTrainedModel
class ReformerModel(ReformerPreTrainedModel):
    def __init__(self, config):
        # 调用父类构造函数
        super().__init__(config)
        # 将配置保存在self.config中
        self.config = config
        # 断言确保num_hidden_layers大于0，否则抛出异常
        assert (
            self.config.num_hidden_layers > 0
        ), "`config.attn_layers` is empty. Select at least one attn layer form ['lsh', 'local']"

        # 初始化词嵌入和编码器
        self.embeddings = ReformerEmbeddings(config)
        self.encoder = ReformerEncoder(config)

        # 初始化权重并进行最终处理
        self.post_init()

    # 返回输入嵌入
    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    # 设置输入嵌入
    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    # 剪枝模型的注意力头
    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    # 使用装饰器添加文档字符串，描述模型前向传播函数的输入
    @add_start_docstrings_to_model_forward(REFORMER_INPUTS_DOCSTRING)
    # 添加代码示例的文档字符串
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=ReformerModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    # 定义前向传播函数
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
    ):
        # 剪切到多个块长度的填充函数
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
        ):
            # 如果输入的序列长度不是 `config.chunk_length` 的倍数，则发出警告
            logger.warning_once(
                f"Input ids are automatically padded from {input_shape[-1]} to {input_shape[-1] + padding_length} to be a "
                f"multiple of `config.chunk_length`: {padded_seq_length}"
            )

        # 使用指定的填充长度和 pad_token_id 创建填充后的输入 ids 张量
        padded_input_ids = torch.full(
            (input_shape[0], padding_length),
            self.config.pad_token_id,
            device=device,
            dtype=torch.long,
        )

        # 扩展 `attention_mask`
        if attention_mask is not None:
            # 创建一个与输入形状相同的全零张量，并将其与原 attention_mask 进行拼接
            pad_attention_mask = torch.zeros(input_shape[0], padding_length, device=device, dtype=attention_mask.dtype)
            attention_mask = torch.cat([attention_mask, pad_attention_mask], dim=-1)
        else:
            # 如果原 attention_mask 不存在，则创建一个全真值的张量和一个全零张量，然后进行拼接
            attention_mask = torch.cat(
                [
                    torch.ones(input_shape, device=device, dtype=torch.bool),
                    torch.zeros((input_shape[0], padding_length), device=device, dtype=torch.bool),
                ],
                dim=-1,
            )

        # 如果存在输入 ids，则将填充后的输入 ids 进行拼接，并更新输入形状
        if input_ids is not None:
            input_ids = torch.cat([input_ids, padded_input_ids], dim=-1)
            input_shape = input_ids.size()

            # 如果存在位置 ids，则对位置 ids 进行填充
            if position_ids is not None:
                # 创建一个从原始长度到填充长度的序列，然后拼接到原位置 ids 后面
                padded_position_ids = torch.arange(input_shape[-1], padded_seq_length, dtype=torch.long, device=device)
                padded_position_ids = position_ids.unsqueeze(0).expand(input_shape[0], padding_length)
                position_ids = torch.cat([position_ids, padded_position_ids], dim=-1)

        # 如果存在输入嵌入张量，则对其进行填充，并更新输入形状
        if inputs_embeds is not None:
            # 使用填充后的输入 ids 和位置 ids 重新计算输入嵌入张量，并进行拼接
            padded_inputs_embeds = self.embeddings(padded_input_ids, position_ids)
            inputs_embeds = torch.cat([inputs_embeds, padded_inputs_embeds], dim=-2)
            input_shape = inputs_embeds.size()

        # 返回填充后的 input_ids、inputs_embeds、attention_mask、position_ids 和更新后的 input_shape
        return input_ids, inputs_embeds, attention_mask, position_ids, input_shape
# 使用装饰器为 ReformerModelWithLMHead 类添加文档字符串，描述其作为带有语言建模头部的 Reformer 模型
@add_start_docstrings("""Reformer Model with a `language modeling` head on top.""", REFORMER_START_DOCSTRING)
class ReformerModelWithLMHead(ReformerPreTrainedModel):
    # 定义需要共享权重的键列表，这些键对应于语言建模头部的权重和偏置
    _tied_weights_keys = ["lm_head.decoder.weight", "lm_head.decoder.bias"]

    def __init__(self, config):
        # 调用父类的构造函数，传入配置参数
        super().__init__(config)
        # 断言配置中必须设置 is_decoder=True，否则抛出异常
        assert config.is_decoder, "If you want to use `ReformerModelWithLMHead` make sure that `is_decoder=True`."
        # 断言配置中不应该使用 "local" 注意力层或者 local_num_chunks_after 应为 0，否则抛出异常
        assert "local" not in self.config.attn_layers or config.local_num_chunks_after == 0, (
            "If causal mask is enabled, make sure that `config.local_num_chunks_after` is set to 0 and not"
            f" {config.local_num_chunks_after}."
        )
        # 断言配置中不应该使用 "lsh" 注意力层或者 lsh_num_chunks_after 应为 0，否则抛出异常
        assert "lsh" not in self.config.attn_layers or config.lsh_num_chunks_after == 0, (
            "If causal mask is enabled, make sure that `config.lsh_num_chunks_after` is set to 1 and not"
            f" {config.lsh_num_chunks_after}."
        )

        # 实例化 ReformerModel，传入配置参数
        self.reformer = ReformerModel(config)
        # 实例化 ReformerOnlyLMHead，传入配置参数
        self.lm_head = ReformerOnlyLMHead(config)

        # 初始化权重并应用最终处理
        self.post_init()

    # 获取语言建模头部的输出嵌入层
    def get_output_embeddings(self):
        return self.lm_head.decoder

    # 设置语言建模头部的输出嵌入层为新的嵌入层
    def set_output_embeddings(self, new_embeddings):
        self.lm_head.decoder = new_embeddings

    # 重写 forward 方法，并使用装饰器添加文档字符串和代码示例文档字符串
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
    ) -> Union[Tuple, CausalLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
                Labels for computing the sequence classification/regression loss. Indices should be in `[-100, 0, ...,
                config.vocab_size - 1]`. All labels set to `-100` are ignored (masked), the loss is only computed for
                labels in `[0, ..., config.vocab_size]`
        """
        # 如果 return_dict 已经被定义，则使用其当前值；否则使用 self.config.use_return_dict 的值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用 Reformer 模型，传递多个参数以获取输出
        reformer_outputs = self.reformer(
            input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            num_hashes=num_hashes,
            past_buckets_states=past_buckets_states,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            return_dict=return_dict,
        )

        # 从 Reformer 输出中获取序列输出
        sequence_output = reformer_outputs[0]
        # 将序列输出传递给语言模型头部，获取 logits
        logits = self.lm_head(sequence_output)

        # 初始化损失为 None
        loss = None
        # 如果 labels 不为空，则计算损失
        if labels is not None:
            # 将 logits 左移一位，以便对应标签右移一位，即 tokens < n 预测 n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # 使用交叉熵损失函数计算损失
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

        # 如果 return_dict 为 False，则返回一个元组，包含 logits 和其他输出
        if not return_dict:
            output = (logits,) + reformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        # 如果 return_dict 为 True，则返回一个包含损失和各种输出的 ReformerModelWithLMHeadOutput 对象
        return ReformerModelWithLMHeadOutput(
            loss=loss,
            logits=logits,
            past_buckets_states=reformer_outputs.past_buckets_states,
            hidden_states=reformer_outputs.hidden_states,
            attentions=reformer_outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, use_cache=None, num_hashes=None, **kwargs
    ):
        # 如果 past_key_values 不为空，则只保留输入的最后一个 token
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]

        # 准备用于生成的输入字典
        inputs_dict = {
            "input_ids": input_ids,
            "past_buckets_states": past_key_values,
            "use_cache": use_cache,
            "num_hashes": num_hashes,
        }

        # 返回输入字典
        return inputs_dict
    # 重新排序缓存中的过去键值对状态
    def _reorder_cache(self, past_key_values, beam_idx):
        # 存储重新排序后的过去桶状态和隐藏状态的列表
        reord_past_buckets_states = []
        # 遍历每个层级的过去键值对
        for layer_past in past_key_values:
            # 如果当前层的桶状态不为None，则根据beam_idx重新排序
            if layer_past[0] is not None:
                reord_buckets = layer_past[0].index_select(0, beam_idx.to(layer_past[0].device))
            else:
                reord_buckets = None

            # 根据beam_idx重新排序当前层的隐藏状态
            reord_hidden_states = layer_past[1].index_select(0, beam_idx.to(layer_past[1].device))
            # 将重新排序后的桶状态和隐藏状态组成元组，并添加到列表中
            reord_past_buckets_states.append((reord_buckets, reord_hidden_states))
        
        # 返回所有层级的重新排序后的过去桶状态和隐藏状态的列表
        return reord_past_buckets_states
@add_start_docstrings("""Reformer Model with a `language modeling` head on top.""", REFORMER_START_DOCSTRING)
class ReformerForMaskedLM(ReformerPreTrainedModel):
    _tied_weights_keys = ["lm_head.decoder.weight", "lm_head.decoder.bias"]

    def __init__(self, config):
        super().__init__(config)
        # 确保不是解码器模式，因为此模型用于双向自注意力
        assert not config.is_decoder, (
            "If you want to use `ReformerForMaskedLM` make sure `config.is_decoder=False` for bi-directional"
            " self-attention."
        )
        # 初始化Reformer模型和ReformerLMHead
        self.reformer = ReformerModel(config)
        self.lm_head = ReformerOnlyLMHead(config)

        # 初始化权重并应用最终处理
        self.post_init()

    def get_output_embeddings(self):
        # 返回语言建模头部的解码器权重
        return self.lm_head.decoder

    def set_output_embeddings(self, new_embeddings):
        # 设置新的输出嵌入到语言建模头部的解码器
        self.lm_head.decoder = new_embeddings

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
        super().__init__(config)
        # 保存标签数和配置
        self.num_labels = config.num_labels
        self.config = config

        # 初始化Reformer模型和分类器
        self.reformer = ReformerModel(config)
        self.classifier = ReformerClassificationHead(config)
        if config.is_decoder is True:
            # 如果配置为解码器，警告可能需要禁用因果遮蔽以进行序列分类
            logger.warning("You might want to disable causal masking for sequence classification")

        # 初始化权重并应用最终处理
        self.post_init()

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
        # 初始化分类器头部，输入维度为2倍的隐藏状态大小，输出维度为隐藏状态大小
        self.dense = nn.Linear(2 * config.hidden_size, config.hidden_size)
        
        # 根据配置设置分类器的dropout，如果未提供分类器dropout，则使用隐藏层的dropout概率
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        
        # 最终的线性投影层，将隐藏状态映射到标签数量大小的输出
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, hidden_states, **kwargs):
        # 只取第一个特殊符号 <s> 的隐藏状态，通常对应于 [CLS] 标记
        hidden_states = hidden_states[:, 0, :]  # take <s> token (equiv. to [CLS])
        
        # 应用dropout以减少过拟合风险
        hidden_states = self.dropout(hidden_states)
        
        # 使用全连接层进行特征变换
        hidden_states = self.dense(hidden_states)
        
        # 应用tanh激活函数，增强非线性建模能力
        hidden_states = torch.tanh(hidden_states)
        
        # 再次应用dropout
        hidden_states = self.dropout(hidden_states)
        
        # 最终通过线性投影层得到分类任务的输出
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
        
        # 记录标签数量
        self.num_labels = config.num_labels
        
        # 初始化Reformer模型，这里的隐藏状态是2倍的隐藏大小，因为使用了可逆残差层
        self.reformer = ReformerModel(config)
        
        # 初始化问题回答任务的输出层，输入维度是2倍的隐藏状态大小，输出维度是标签数量
        self.qa_outputs = nn.Linear(2 * config.hidden_size, config.num_labels)

        # 初始化权重并应用最终处理
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