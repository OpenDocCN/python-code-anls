# `.\models\canine\modeling_canine.py`

```
# 定义一个数据类，用于存储 CANINE 模型的输出，包含了额外的池化信息
@dataclass
class CanineModelOutputWithPooling(ModelOutput):
    """
    Output type of [`CanineModel`]. Based on [`~modeling_outputs.BaseModelOutputWithPooling`], but with slightly
    different `hidden_states` and `attentions`, as these also include the hidden states and attentions of the shallow
    Transformer encoders.
    """
    # 继承自 `ModelOutput` 类，包含了基本的模型输出信息
    # 定义函数参数和返回类型注释
    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            模型最后一层的隐藏状态序列，是深度Transformer编码器的输出。
        pooler_output (`torch.FloatTensor` of shape `(batch_size, hidden_size)`):
            序列中第一个标记（分类标记）在深度Transformer编码器最后一层的隐藏状态，经过线性层和Tanh激活函数进一步处理。
            线性层的权重在预训练期间从下一个句子预测（分类）目标中训练得到。
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            元组类型，包含`torch.FloatTensor`类型的张量，每个编码器的输入和每个编码器每一层的输出。
            第一个张量的形状为 `(batch_size, sequence_length, hidden_size)`，第二个张量的形状为
            `(batch_size, sequence_length // config.downsampling_rate, hidden_size)`。
            浅层编码器的隐藏状态长度为 `sequence_length`，深层编码器的隐藏状态长度为 `sequence_length // config.downsampling_rate`。
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            元组类型，包含`torch.FloatTensor`类型的张量，每个编码器的注意力权重。
            第一个张量的形状为 `(batch_size, num_heads, sequence_length, sequence_length)`，
            第二个张量的形状为 `(batch_size, num_heads, sequence_length // config.downsampling_rate, sequence_length // config.downsampling_rate)`。
            在注意力softmax之后的注意力权重，用于计算自注意力头中的加权平均值。
    """
    
    # 初始化函数参数的默认值
    last_hidden_state: torch.FloatTensor = None
    pooler_output: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
def load_tf_weights_in_canine(model, config, tf_checkpoint_path):
    """Load tf checkpoints in a pytorch model."""
    try:
        import re  # 导入正则表达式模块，用于处理字符串匹配
        import numpy as np  # 导入NumPy库，用于数值计算
        import tensorflow as tf  # 导入TensorFlow库，用于加载TensorFlow模型权重
    except ImportError:
        logger.error(
            "Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions."
        )
        raise

    tf_path = os.path.abspath(tf_checkpoint_path)  # 获取TensorFlow模型检查点文件的绝对路径
    logger.info(f"Converting TensorFlow checkpoint from {tf_path}")  # 记录日志信息，显示正在转换的TensorFlow检查点路径

    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)  # 获取TensorFlow模型中所有变量的名称和形状信息
    names = []
    arrays = []
    for name, shape in init_vars:
        logger.info(f"Loading TF weight {name} with shape {shape}")  # 记录日志信息，显示正在加载的TensorFlow权重名称和形状
        array = tf.train.load_variable(tf_path, name)  # 加载TensorFlow模型中指定变量的权重数据
        names.append(name)
        arrays.append(array)

    return model


class CanineEmbeddings(nn.Module):
    """Construct the character, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()

        self.config = config

        # character embeddings
        shard_embedding_size = config.hidden_size // config.num_hash_functions
        for i in range(config.num_hash_functions):
            name = f"HashBucketCodepointEmbedder_{i}"
            setattr(self, name, nn.Embedding(config.num_hash_buckets, shard_embedding_size))
            # 设置每个哈希桶代码点嵌入层，使用nn.Embedding创建嵌入矩阵

        self.char_position_embeddings = nn.Embedding(config.num_hash_buckets, config.hidden_size)
        # 设置字符位置嵌入层，使用nn.Embedding创建嵌入矩阵，嵌入维度为config.hidden_size
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        # 设置令牌类型嵌入层，使用nn.Embedding创建嵌入矩阵，嵌入维度为config.hidden_size

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 设置LayerNorm层，使用nn.LayerNorm进行层归一化，归一化维度为config.hidden_size，设置epsilon为config.layer_norm_eps
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # 设置Dropout层，使用nn.Dropout进行Dropout操作，设置丢弃概率为config.hidden_dropout_prob

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )
        # 注册position_ids作为缓冲区，存储长度为config.max_position_embeddings的位置ID张量，不持久化保存

        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        # 设置位置嵌入类型，默认为"absolute"，如果config中有指定position_embedding_type则使用指定值
    def _hash_bucket_tensors(self, input_ids, num_hashes: int, num_buckets: int):
        """
        Converts ids to hash bucket ids via multiple hashing.

        Args:
            input_ids: The codepoints or other IDs to be hashed.
            num_hashes: The number of hash functions to use.
            num_buckets: The number of hash buckets (i.e. embeddings in each table).

        Returns:
            A list of tensors, each of which is the hash bucket IDs from one hash function.
        """
        # 检查 `num_hashes` 是否超过了预定义的素数列表长度，抛出异常
        if num_hashes > len(_PRIMES):
            raise ValueError(f"`num_hashes` must be <= {len(_PRIMES)}")

        # 选择前 `num_hashes` 个素数作为哈希函数的参数
        primes = _PRIMES[:num_hashes]

        result_tensors = []
        # 对每一个素数进行哈希计算
        for prime in primes:
            # 根据哈希函数计算输入 ID 的哈希桶 ID
            hashed = ((input_ids + 1) * prime) % num_buckets
            result_tensors.append(hashed)
        return result_tensors

    def _embed_hash_buckets(self, input_ids, embedding_size: int, num_hashes: int, num_buckets: int):
        """Converts IDs (e.g. codepoints) into embeddings via multiple hashing."""
        # 检查 `embedding_size` 是否可以被 `num_hashes` 整除，否则抛出异常
        if embedding_size % num_hashes != 0:
            raise ValueError(f"Expected `embedding_size` ({embedding_size}) % `num_hashes` ({num_hashes}) == 0")

        # 使用 `_hash_bucket_tensors` 方法将输入 ID 转换为哈希桶 ID 的张量列表
        hash_bucket_tensors = self._hash_bucket_tensors(input_ids, num_hashes=num_hashes, num_buckets=num_buckets)
        embedding_shards = []
        # 对每一个哈希桶 ID 张量进行嵌入映射
        for i, hash_bucket_ids in enumerate(hash_bucket_tensors):
            name = f"HashBucketCodepointEmbedder_{i}"
            # 调用模型的子模块进行哈希桶 ID 的嵌入映射
            shard_embeddings = getattr(self, name)(hash_bucket_ids)
            embedding_shards.append(shard_embeddings)

        # 将所有嵌入映射拼接成一个张量
        return torch.cat(embedding_shards, dim=-1)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        # 获取输入序列的长度
        seq_length = input_shape[1]

        # 如果未提供位置 ID，则使用预定义的位置 ID
        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        # 如果未提供 token 类型 ID，则默认为全零张量
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        # 如果未提供输入嵌入张量，则通过 `_embed_hash_buckets` 方法生成
        if inputs_embeds is None:
            inputs_embeds = self._embed_hash_buckets(
                input_ids, self.config.hidden_size, self.config.num_hash_functions, self.config.num_hash_buckets
            )

        # 获取 token 类型的嵌入映射
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        # 将输入嵌入张量与 token 类型嵌入映射相加
        embeddings = inputs_embeds + token_type_embeddings

        # 如果位置嵌入类型为 "absolute"，则加上字符位置嵌入映射
        if self.position_embedding_type == "absolute":
            position_embeddings = self.char_position_embeddings(position_ids)
            embeddings += position_embeddings

        # 执行 LayerNorm 操作
        embeddings = self.LayerNorm(embeddings)
        # 执行 dropout 操作
        embeddings = self.dropout(embeddings)
        return embeddings
class CharactersToMolecules(nn.Module):
    """Convert character sequence to initial molecule sequence (i.e. downsample) using strided convolutions."""

    def __init__(self, config):
        super().__init__()

        # Define 1D convolutional layer for downsampling
        self.conv = nn.Conv1d(
            in_channels=config.hidden_size,
            out_channels=config.hidden_size,
            kernel_size=config.downsampling_rate,
            stride=config.downsampling_rate,
        )
        
        # Activation function based on the configuration
        self.activation = ACT2FN[config.hidden_act]

        # Layer normalization to normalize outputs across the hidden_size dimension
        # `self.LayerNorm` is kept as is to maintain compatibility with TensorFlow checkpoints
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, char_encoding: torch.Tensor) -> torch.Tensor:
        # Extract the [CLS] token encoding: [batch, 1, hidden_size]
        cls_encoding = char_encoding[:, 0:1, :]

        # Transpose `char_encoding` to [batch, hidden_size, char_seq]
        char_encoding = torch.transpose(char_encoding, 1, 2)

        # Apply convolution for downsampling, then transpose back
        downsampled = self.conv(char_encoding)
        downsampled = torch.transpose(downsampled, 1, 2)

        # Apply activation function to the downsampled sequence
        downsampled = self.activation(downsampled)

        # Remove the last molecule to reserve space for [CLS], maintaining alignment on TPUs
        downsampled_truncated = downsampled[:, 0:-1, :]

        # Concatenate [CLS] encoding with downsampled sequence
        result = torch.cat([cls_encoding, downsampled_truncated], dim=1)

        # Apply LayerNorm to the concatenated sequence
        result = self.LayerNorm(result)

        return result


class ConvProjection(nn.Module):
    """
    Project representations from hidden_size*2 back to hidden_size across a window of w = config.upsampling_kernel_size
    characters.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Define 1D convolutional layer for upsampling
        self.conv = nn.Conv1d(
            in_channels=config.hidden_size * 2,
            out_channels=config.hidden_size,
            kernel_size=config.upsampling_kernel_size,
            stride=1,
        )

        # Activation function based on the configuration
        self.activation = ACT2FN[config.hidden_act]

        # Layer normalization to normalize outputs across the hidden_size dimension
        # `self.LayerNorm` is kept as is to maintain compatibility with TensorFlow checkpoints
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # Dropout layer for regularization
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self,
        inputs: torch.Tensor,
        final_seq_char_positions: Optional[torch.Tensor] = None,
        # inputs has shape [batch, mol_seq, molecule_hidden_size+char_hidden_final]
        # we transpose it to be [batch, molecule_hidden_size+char_hidden_final, mol_seq]
        inputs = torch.transpose(inputs, 1, 2)

        # PyTorch < 1.9 does not support padding="same" (which is used in the original implementation),
        # so we pad the tensor manually before passing it to the conv layer
        # based on https://github.com/google-research/big_transfer/blob/49afe42338b62af9fbe18f0258197a33ee578a6b/bit_tf2/models.py#L36-L38
        # Calculate total padding needed to achieve 'same' padding
        pad_total = self.config.upsampling_kernel_size - 1
        pad_beg = pad_total // 2  # Calculate padding to be added at the beginning
        pad_end = pad_total - pad_beg  # Calculate padding to be added at the end

        # Create a 1-dimensional constant padding layer for convolution
        pad = nn.ConstantPad1d((pad_beg, pad_end), 0)
        # Apply padding to inputs tensor before passing it through convolutional layer
        padded_inputs = pad(inputs)

        # Perform convolution operation on the padded inputs
        # `result`: shape (batch_size, char_seq_len, hidden_size)
        result = self.conv(padded_inputs)

        # Transpose result tensor to revert to original shape [batch, mol_seq, hidden_size]
        result = torch.transpose(result, 1, 2)

        # Apply activation function (e.g., ReLU) to the convolved result
        result = self.activation(result)

        # Apply layer normalization to stabilize training
        result = self.LayerNorm(result)

        # Apply dropout for regularization
        result = self.dropout(result)

        # Store the processed character sequence as the final output
        final_char_seq = result

        if final_seq_char_positions is not None:
            # Limit transformer query seq and attention mask to these character
            # positions to greatly reduce the compute cost. Typically, this is just
            # done for the MLM training task.
            # TODO add support for MLM
            raise NotImplementedError("CanineForMaskedLM is currently not supported")
        else:
            # If no specific character positions are provided, use the entire processed sequence
            query_seq = final_char_seq

        # Return the final processed query sequence
        return query_seq
# 定义一个名为 CanineSelfOutput 的神经网络模块，用于处理自注意力机制的输出
class CanineSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 线性变换层，将隐藏状态的维度转换为 config.hidden_size
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # LayerNorm 层，用于规范化隐藏状态，以减少内部协变量偏移
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # Dropout 层，用于随机置零隐藏状态的部分单元，以减少过拟合风险
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self, hidden_states: Tuple[torch.FloatTensor], input_tensor: torch.FloatTensor
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        # 线性变换操作，将隐藏状态转换为 config.hidden_size 维度
        hidden_states = self.dense(hidden_states)
        # 对转换后的隐藏状态进行随机置零处理，以减少过拟合
        hidden_states = self.dropout(hidden_states)
        # 对处理后的隐藏状态进行 LayerNorm 规范化，加上输入的 tensor，形成残差连接
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        # 返回规范化后的隐藏状态
        return hidden_states
    def __init__(
        self,
        config,
        local=False,
        always_attend_to_first_position: bool = False,
        first_position_attends_to_all: bool = False,
        attend_from_chunk_width: int = 128,
        attend_from_chunk_stride: int = 128,
        attend_to_chunk_width: int = 128,
        attend_to_chunk_stride: int = 128,
    ):
        super().__init__()
        # 初始化自注意力机制和自注意力输出层
        self.self = CanineSelfAttention(config)
        self.output = CanineSelfOutput(config)
        # 初始化一个空的剪枝头集合
        self.pruned_heads = set()

        # 检查是否开启局部注意力
        self.local = local
        # 检查块大小和跨步是否合理，防止序列位置被跳过
        if attend_from_chunk_width < attend_from_chunk_stride:
            raise ValueError(
                "`attend_from_chunk_width` < `attend_from_chunk_stride` would cause sequence positions to get skipped."
            )
        if attend_to_chunk_width < attend_to_chunk_stride:
            raise ValueError(
                "`attend_to_chunk_width` < `attend_to_chunk_stride` would cause sequence positions to get skipped."
            )
        # 设置额外的局部注意力参数
        self.always_attend_to_first_position = always_attend_to_first_position
        self.first_position_attends_to_all = first_position_attends_to_all
        self.attend_from_chunk_width = attend_from_chunk_width
        self.attend_from_chunk_stride = attend_from_chunk_stride
        self.attend_to_chunk_width = attend_to_chunk_width
        self.attend_to_chunk_stride = attend_to_chunk_stride
    # 对 self 对象中的注意力头进行修剪操作
    def prune_heads(self, heads):
        # 如果 heads 列表为空，则直接返回，不执行修剪操作
        if len(heads) == 0:
            return
        # 调用 find_pruneable_heads_and_indices 函数查找可修剪的注意力头及其索引
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # 对线性层进行修剪
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # 更新超参数并存储修剪后的头信息
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    # 定义前向传播函数
    def forward(
        self,
        hidden_states: Tuple[torch.FloatTensor],
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
# 定义一个名为 CanineIntermediate 的神经网络模块类
class CanineIntermediate(nn.Module):
    # 初始化方法，接受一个 config 参数
    def __init__(self, config):
        super().__init__()
        # 创建一个线性层，输入大小为 config.hidden_size，输出大小为 config.intermediate_size
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        # 根据 config 中的 hidden_act 字段选择激活函数，存储在 self.intermediate_act_fn 中
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    # 前向传播方法，接受 hidden_states 参数作为输入张量，返回处理后的张量
    def forward(self, hidden_states: torch.FloatTensor) -> torch.FloatTensor:
        # 将输入张量通过 self.dense 线性层处理
        hidden_states = self.dense(hidden_states)
        # 将处理后的张量通过选定的激活函数 self.intermediate_act_fn 进行激活
        hidden_states = self.intermediate_act_fn(hidden_states)
        # 返回处理后的张量作为输出
        return hidden_states


# 定义一个名为 CanineOutput 的神经网络模块类
class CanineOutput(nn.Module):
    # 初始化方法，接受一个 config 参数
    def __init__(self, config):
        super().__init__()
        # 创建一个线性层，输入大小为 config.intermediate_size，输出大小为 config.hidden_size
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        # 创建一个 LayerNorm 层，对输入大小为 config.hidden_size 的张量进行归一化
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 创建一个 Dropout 层，使用 config.hidden_dropout_prob 作为丢弃概率
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    # 前向传播方法，接受 hidden_states 和 input_tensor 两个参数作为输入，返回处理后的张量
    def forward(self, hidden_states: Tuple[torch.FloatTensor], input_tensor: torch.FloatTensor) -> torch.FloatTensor:
        # 将输入张量通过 self.dense 线性层处理
        hidden_states = self.dense(hidden_states)
        # 对处理后的张量应用 Dropout 操作
        hidden_states = self.dropout(hidden_states)
        # 将 Dropout 后的张量与输入张量 input_tensor 相加，并通过 LayerNorm 层处理
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        # 返回处理后的张量作为输出
        return hidden_states


# 定义一个名为 CanineLayer 的神经网络模块类
class CanineLayer(nn.Module):
    # 初始化方法，接受多个参数，包括 config 和各种注意力机制的相关参数
    def __init__(
        self,
        config,
        local,
        always_attend_to_first_position,
        first_position_attends_to_all,
        attend_from_chunk_width,
        attend_from_chunk_stride,
        attend_to_chunk_width,
        attend_to_chunk_stride,
    ):
        super().__init__()
        # 设定块大小 feed forward 的大小为 config.chunk_size_feed_forward
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        # 序列长度维度为 1
        self.seq_len_dim = 1
        # 创建 CanineAttention 层，使用给定的参数进行初始化
        self.attention = CanineAttention(
            config,
            local,
            always_attend_to_first_position,
            first_position_attends_to_all,
            attend_from_chunk_width,
            attend_from_chunk_stride,
            attend_to_chunk_width,
            attend_to_chunk_stride,
        )
        # 创建 CanineIntermediate 层，使用 config 进行初始化
        self.intermediate = CanineIntermediate(config)
        # 创建 CanineOutput 层，使用 config 进行初始化
        self.output = CanineOutput(config)

    # 前向传播方法，接受 hidden_states、attention_mask、head_mask、output_attentions 四个参数，
    # 返回处理后的张量和可能的注意力权重张量
    def forward(
        self,
        hidden_states: Tuple[torch.FloatTensor],
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor, Optional[torch.FloatTensor]]:
        # 使用 self.attention 对 hidden_states 进行自注意力机制处理
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
        )
        # 获取注意力机制处理后的输出
        attention_output = self_attention_outputs[0]

        # 如果输出注意力权重，则添加自注意力权重到输出中
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        # 将 attention_output 通过 apply_chunking_to_forward 方法进行分块处理
        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        # 将分块处理后的输出添加到 outputs 中
        outputs = (layer_output,) + outputs

        # 返回处理后的输出
        return outputs
    # 定义神经网络的前向传播方法，处理注意力输出作为输入
    def feed_forward_chunk(self, attention_output):
        # 将注意力输出作为输入，调用中间层的方法处理
        intermediate_output = self.intermediate(attention_output)
        # 使用中间层的输出和注意力输出调用输出层的方法，计算最终层的输出
        layer_output = self.output(intermediate_output, attention_output)
        # 返回最终的层输出作为这一块的前向传播结果
        return layer_output
class CanineEncoder(nn.Module):
    # CanineEncoder 类，用于实现特定的编码器模型
    def __init__(
        self,
        config,
        local=False,
        always_attend_to_first_position=False,
        first_position_attends_to_all=False,
        attend_from_chunk_width=128,
        attend_from_chunk_stride=128,
        attend_to_chunk_width=128,
        attend_to_chunk_stride=128,
    ):
        super().__init__()
        self.config = config
        # 创建一个由 CanineLayer 组成的层列表，根据 config 中的隐藏层数量进行初始化
        self.layer = nn.ModuleList(
            [
                CanineLayer(
                    config,
                    local,
                    always_attend_to_first_position,
                    first_position_attends_to_all,
                    attend_from_chunk_width,
                    attend_from_chunk_stride,
                    attend_to_chunk_width,
                    attend_to_chunk_stride,
                )
                for _ in range(config.num_hidden_layers)
            ]
        )
        self.gradient_checkpointing = False  # 梯度检查点标志，默认为 False

    def forward(
        self,
        hidden_states: Tuple[torch.FloatTensor],
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple, BaseModelOutput]:
        # 初始化空元组，用于存储所有隐藏状态和自注意力分数（根据需要）
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        # 遍历所有的层，并执行前向传播
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 获取当前层的头部掩码
            layer_head_mask = head_mask[i] if head_mask is not None else None

            if self.gradient_checkpointing and self.training:
                # 如果启用梯度检查点且处于训练模式，使用梯度检查点函数来计算当前层的输出
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    output_attentions,
                )
            else:
                # 否则，直接调用当前层的__call__方法来计算当前层的输出
                layer_outputs = layer_module(hidden_states, attention_mask, layer_head_mask, output_attentions)

            # 更新隐藏状态为当前层的输出的第一个元素（通常是隐藏状态）
            hidden_states = layer_outputs[0]
            if output_attentions:
                # 如果需要输出自注意力分数，将当前层的自注意力分数添加到 all_self_attentions 中
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            # 如果需要输出所有隐藏状态，将最终的隐藏状态添加到 all_hidden_states 中
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            # 如果不需要返回字典形式的输出，返回一个元组，其中包含非空的结果项
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        # 否则，返回一个 BaseModelOutput 对象，包含最终的隐藏状态、所有隐藏状态和自注意力分数
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class CaninePooler(nn.Module):
    # CaninePooler 类，用于实现特定的池化器模型
    def __init__(self, config):
        super().__init__()
        # 全连接层，将输入的大小转换为隐藏大小
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # Tanh 激活函数
        self.activation = nn.Tanh()
    # 定义类方法 `forward`，接受 `hidden_states` 参数作为输入，并返回 `torch.FloatTensor` 类型的张量
    def forward(self, hidden_states: Tuple[torch.FloatTensor]) -> torch.FloatTensor:
        # 通过取第一个令牌的隐藏状态来"汇聚"模型的输出
        first_token_tensor = hidden_states[:, 0]
        # 将第一个令牌的隐藏状态输入全连接层 `self.dense` 进行线性变换
        pooled_output = self.dense(first_token_tensor)
        # 对线性变换的结果应用激活函数 `self.activation`
        pooled_output = self.activation(pooled_output)
        # 返回汇聚后的输出张量 `pooled_output`
        return pooled_output
class CaninePredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 初始化一个全连接层，输入和输出维度都是 config.hidden_size
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        
        # 根据配置选择激活函数，如果配置中指定了激活函数名称，则使用对应的函数；否则直接使用配置中的激活函数对象
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        
        # 初始化 LayerNorm 层，归一化大小为 config.hidden_size，epsilon 值为 config.layer_norm_eps
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: Tuple[torch.FloatTensor]) -> torch.FloatTensor:
        # 将输入 hidden_states 通过全连接层 dense
        hidden_states = self.dense(hidden_states)
        
        # 使用预先选择的激活函数进行变换
        hidden_states = self.transform_act_fn(hidden_states)
        
        # 对变换后的 hidden_states 进行 LayerNorm 归一化处理
        hidden_states = self.LayerNorm(hidden_states)
        
        # 返回处理后的 hidden_states
        return hidden_states


class CanineLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        # 初始化预测头变换层，使用 CaninePredictionHeadTransform 类处理输入的 config
        self.transform = CaninePredictionHeadTransform(config)

        # 初始化解码层，使用全连接层实现，输入维度为 config.hidden_size，输出维度为 config.vocab_size，无偏置
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # 初始化偏置参数，维度为 config.vocab_size，作为解码层的偏置
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # 将解码层的偏置参数设置为初始化的偏置参数
        self.decoder.bias = self.bias

    def forward(self, hidden_states: Tuple[torch.FloatTensor]) -> torch.FloatTensor:
        # 使用预测头变换层处理输入 hidden_states
        hidden_states = self.transform(hidden_states)
        
        # 使用解码层对处理后的 hidden_states 进行预测
        hidden_states = self.decoder(hidden_states)
        
        # 返回预测得分
        return hidden_states


class CanineOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        # 初始化 MLM 头部，使用 CanineLMPredictionHead 类处理输入的 config
        self.predictions = CanineLMPredictionHead(config)

    def forward(
        self,
        sequence_output: Tuple[torch.Tensor],
    ) -> Tuple[torch.Tensor]:
        # 将序列输出作为输入，通过预测头进行预测
        prediction_scores = self.predictions(sequence_output)
        
        # 返回预测分数
        return prediction_scores


class CaninePreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 指定模型对应的配置类为 CanineConfig
    config_class = CanineConfig
    
    # 指定加载 TensorFlow 权重的函数为 load_tf_weights_in_canine
    load_tf_weights = load_tf_weights_in_canine
    
    # 设置基础模型的名称前缀为 "canine"
    base_model_prefix = "canine"
    
    # 支持梯度检查点
    supports_gradient_checkpointing = True
    def _init_weights(self, module):
        """Initialize the weights"""
        # 如果 module 是 nn.Linear 或 nn.Conv1d 类型
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            # 使用正态分布随机初始化权重，均值为 0.0，标准差为 self.config.initializer_range
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果 module 存在偏置项，则将偏置项初始化为 0
            if module.bias is not None:
                module.bias.data.zero_()
        # 如果 module 是 nn.Embedding 类型
        elif isinstance(module, nn.Embedding):
            # 使用正态分布随机初始化权重，均值为 0.0，标准差为 self.config.initializer_range
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果 module 设置了 padding_idx，将对应位置的权重初始化为 0
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        # 如果 module 是 nn.LayerNorm 类型
        elif isinstance(module, nn.LayerNorm):
            # 将 LayerNorm 层的偏置项初始化为 0
            module.bias.data.zero_()
            # 将 LayerNorm 层的权重初始化为 1
            module.weight.data.fill_(1.0)
CANINE_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.
    
    Parameters:
        config ([`CanineConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

CANINE_INPUTS_DOCSTRING = r"""
    This string is intended to provide documentation about the expected inputs for the CANINE model. However, this section
    currently lacks specific content and requires further completion to describe the inputs comprehensively.
"""
    Args:
        input_ids (`torch.LongTensor` of shape `({0})`):
            # 输入序列中的标记索引，在词汇表中的位置
            # 可以使用 AutoTokenizer 获取这些索引。参见 PreTrainedTokenizer.encode 和 PreTrainedTokenizer.__call__ 进行详细说明。
            # 什么是输入 ID？请参见 ../glossary#input-ids

        attention_mask (`torch.FloatTensor` of shape `({0})`, *optional*):
            # 遮罩，用于在填充标记索引上避免执行注意力操作
            # 遮罩的值选择在 [0, 1] 范围内：
            # - 1 表示 **未被遮罩** 的标记
            # - 0 表示 **被遮罩** 的标记
            # 什么是注意力遮罩？请参见 ../glossary#attention-mask

        token_type_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            # 段标记索引，用于指示输入的第一部分和第二部分
            # 索引在 [0, 1] 范围内选择：
            # - 0 对应 *句子 A* 的标记
            # - 1 对应 *句子 B* 的标记
            # 什么是标记类型 ID？请参见 ../glossary#token-type-ids

        position_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            # 每个输入序列标记在位置嵌入中的位置索引
            # 选择范围在 [0, config.max_position_embeddings - 1] 内
            # 什么是位置 ID？请参见 ../glossary#position-ids

        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            # 用于置空自注意力模块的选定头部的遮罩
            # 遮罩的值选择在 [0, 1] 范围内：
            # - 1 表示 **未被遮罩** 的头部
            # - 0 表示 **被遮罩** 的头部

        inputs_embeds (`torch.FloatTensor` of shape `({0}, hidden_size)`, *optional*):
            # 可选项，可以直接传递嵌入表示而不是传递 input_ids
            # 如果您想要更多控制如何将 input_ids 索引转换为相关联的向量，而不是使用模型内部的嵌入查找矩阵，则这很有用。

        output_attentions (`bool`, *optional*):
            # 是否返回所有注意力层的注意力张量。更多细节请参见返回张量中的 `attentions`。

        output_hidden_states (`bool`, *optional*):
            # 是否返回所有层的隐藏状态。更多细节请参见返回张量中的 `hidden_states`。

        return_dict (`bool`, *optional*):
            # 是否返回 `~utils.ModelOutput` 而不是普通元组。
"""
@add_start_docstrings(
    "The bare CANINE Model transformer outputting raw hidden-states without any specific head on top.",
    CANINE_START_DOCSTRING,
)
class CanineModel(CaninePreTrainedModel):
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config
        shallow_config = copy.deepcopy(config)
        shallow_config.num_hidden_layers = 1

        self.char_embeddings = CanineEmbeddings(config)
        # 初始化字符嵌入层
        self.initial_char_encoder = CanineEncoder(
            shallow_config,
            local=True,
            always_attend_to_first_position=False,
            first_position_attends_to_all=False,
            attend_from_chunk_width=config.local_transformer_stride,
            attend_from_chunk_stride=config.local_transformer_stride,
            attend_to_chunk_width=config.local_transformer_stride,
            attend_to_chunk_stride=config.local_transformer_stride,
        )
        # 初始化字符到分子的转换层
        self.chars_to_molecules = CharactersToMolecules(config)
        # 初始化深层 transformer 编码器
        self.encoder = CanineEncoder(config)
        # 初始化投影层
        self.projection = ConvProjection(config)
        # 初始化最终字符编码的浅层 transformer 编码器
        self.final_char_encoder = CanineEncoder(shallow_config)

        self.pooler = CaninePooler(config) if add_pooling_layer else None

        # 初始化权重并应用最终处理
        self.post_init()

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def _create_3d_attention_mask_from_input_mask(self, from_tensor, to_mask):
        """
        Create 3D attention mask from a 2D tensor mask.

        Args:
            from_tensor: 2D or 3D Tensor of shape [batch_size, from_seq_length, ...].
            to_mask: int32 Tensor of shape [batch_size, to_seq_length].

        Returns:
            float Tensor of shape [batch_size, from_seq_length, to_seq_length].
        """
        # 获取输入张量的批量大小和序列长度
        batch_size, from_seq_length = from_tensor.shape[0], from_tensor.shape[1]

        # 获取目标掩码的序列长度
        to_seq_length = to_mask.shape[1]

        # 将目标掩码重塑为正确形状，并转换为浮点型张量
        to_mask = torch.reshape(to_mask, (batch_size, 1, to_seq_length)).float()

        # 创建一个全为1的张量，用于掩盖
        broadcast_ones = torch.ones(size=(batch_size, from_seq_length, 1), dtype=torch.float32, device=to_mask.device)

        # 使用广播操作创建掩码
        mask = broadcast_ones * to_mask

        return mask
    def _downsample_attention_mask(self, char_attention_mask: torch.Tensor, downsampling_rate: int):
        """Downsample 2D character attention mask to 2D molecule attention mask using MaxPool1d layer."""

        # 将二维字符注意力掩码转换为三维，添加一个通道维度
        batch_size, char_seq_len = char_attention_mask.shape
        poolable_char_mask = torch.reshape(char_attention_mask, (batch_size, 1, char_seq_len))

        # 使用 MaxPool1d 进行下采样，得到形状为 (batch_size, 1, mol_seq_len) 的池化分子注意力掩码
        pooled_molecule_mask = torch.nn.MaxPool1d(kernel_size=downsampling_rate, stride=downsampling_rate)(
            poolable_char_mask.float()
        )

        # 最后，压缩维度，得到形状为 (batch_size, mol_seq_len) 的张量
        molecule_attention_mask = torch.squeeze(pooled_molecule_mask, dim=-1)

        return molecule_attention_mask

    def _repeat_molecules(self, molecules: torch.Tensor, char_seq_length: torch.Tensor) -> torch.Tensor:
        """Repeats molecules to make them the same length as the char sequence."""

        rate = self.config.downsampling_rate

        # 从 `molecules` 中去除额外的 `<cls>` 标记，形状为 [batch_size, almost_char_seq_len, molecule_hidden_size]
        molecules_without_extra_cls = molecules[:, 1:, :]
        # 使用 repeat_interleave 函数按指定的倍数 `rate` 在指定维度 `-2` 重复张量
        repeated = torch.repeat_interleave(molecules_without_extra_cls, repeats=rate, dim=-2)

        # 现在，我们已经为任何 `char_seq_length` 是 `downsampling_rate` 的倍数的情况重复了足够的元素。
        # 现在我们处理最后的 n 个元素（其中 n < `downsampling_rate`），即 floor 除法的余数部分。
        # 我们通过额外多次重复最后一个分子来处理这部分。
        last_molecule = molecules[:, -1:, :]
        remainder_length = torch.fmod(torch.tensor(char_seq_length), torch.tensor(rate)).item()
        remainder_repeated = torch.repeat_interleave(
            last_molecule,
            repeats=remainder_length + rate,  # 加1个分子以弥补截断。
            dim=-2,
        )

        # 将重复后的结果拼接起来，形状为 [batch_size, char_seq_len, molecule_hidden_size]
        return torch.cat([repeated, remainder_repeated], dim=-2)

    @add_start_docstrings_to_model_forward(CANINE_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=CanineModelOutputWithPooling,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
@add_start_docstrings(
    """
    CANINE Model transformer with a sequence classification/regression head on top (a linear layer on top of the pooled
    output) e.g. for GLUE tasks.
    """,
    CANINE_START_DOCSTRING,
)
class CanineForSequenceClassification(CaninePreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels  # 初始化模型时设置标签数量

        self.canine = CanineModel(config)  # 使用配置初始化CANINE模型
        self.dropout = nn.Dropout(config.hidden_dropout_prob)  # 根据配置设置dropout层
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)  # 线性层，用于分类任务

        # 初始化权重并进行最终处理
        self.post_init()

    @add_start_docstrings_to_model_forward(CANINE_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=SequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        """
        前向传播函数，接收多个输入参数，执行模型推断。

        Args:
            input_ids (Optional[torch.LongTensor]): 输入的token IDs序列.
            attention_mask (Optional[torch.FloatTensor]): 注意力掩码，指示哪些位置是填充的.
            token_type_ids (Optional[torch.LongTensor]): token类型IDs，如用于BERT模型的segment IDs.
            position_ids (Optional[torch.LongTensor]): 位置IDs，用于指定每个token的绝对位置.
            head_mask (Optional[torch.FloatTensor]): 多头注意力层的掩码.
            inputs_embeds (Optional[torch.FloatTensor]): 直接的嵌入表示输入.
            labels (Optional[torch.LongTensor]): 模型的标签.
            output_attentions (Optional[bool]): 是否输出注意力权重.
            output_hidden_states (Optional[bool]): 是否输出所有隐藏状态.
            return_dict (Optional[bool]): 是否返回输出字典.

        Returns:
            SequenceClassifierOutput: 序列分类器的输出，包括预测和额外的元数据.
        """
        # 通过CANINE模型获取输出的隐藏状态和池化的输出
        outputs = self.canine(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 对CANINE的输出进行dropout处理
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)

        # 使用线性分类器进行分类预测
        logits = self.classifier(pooled_output)

        # 构建返回的序列分类器输出对象
        return SequenceClassifierOutput(
            loss=None if labels is None else F.cross_entropy(logits, labels),
            logits=logits,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=outputs.attentions if output_attentions else None,
        )
    ) -> Union[Tuple, SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        # 如果 return_dict 不为 None，则使用其值；否则使用 self.config.use_return_dict 的值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用 Canine 模型进行推理，获取模型的输出
        outputs = self.canine(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 从模型输出中获取 pooled_output（通常是 BERT 模型中的 [CLS] token 的输出）
        pooled_output = outputs[1]

        # 对 pooled_output 应用 dropout
        pooled_output = self.dropout(pooled_output)

        # 将经过 dropout 后的 pooled_output 输入到分类器（通常是一个线性层）
        logits = self.classifier(pooled_output)

        # 初始化 loss 为 None
        loss = None

        # 如果 labels 不为 None，则计算损失
        if labels is not None:
            # 确定问题类型（回归、单标签分类、多标签分类）
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            # 根据问题类型选择相应的损失函数并计算损失
            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    # 如果只有一个标签，使用 squeeze() 去除维度为 1 的维度
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                # 调整 logits 和 labels 的形状以匹配损失函数的要求
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        # 如果 return_dict 为 False，则组合输出成 tuple
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 如果 return_dict 为 True，则返回 SequenceClassifierOutput 类型的对象
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
@add_start_docstrings(
    """
    CANINE Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
    softmax) e.g. for RocStories/SWAG tasks.
    """,
    CANINE_START_DOCSTRING,
)
class CanineForMultipleChoice(CaninePreTrainedModel):
    """
    CANINE模型，顶部带有多选分类头部（在汇总输出之上的线性层和softmax），例如用于RocStories/SWAG任务。
    继承自CaninePreTrainedModel类。
    """

    def __init__(self, config):
        """
        初始化方法，设置模型结构。

        Args:
            config (CanineConfig): 模型配置对象，包含模型的各种参数设置。
        """
        super().__init__(config)

        # 加载预训练的CANINE模型
        self.canine = CanineModel(config)
        # Dropout层，用于防止过拟合
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # 多选分类线性层，将CANINE模型输出映射到分类标签
        self.classifier = nn.Linear(config.hidden_size, 1)

        # 初始化权重并应用最终处理
        self.post_init()

    @add_start_docstrings_to_model_forward(CANINE_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=MultipleChoiceModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        """
        前向传播方法，定义模型的数据流。

        Args:
            input_ids (Optional[torch.LongTensor]): 输入的token IDs张量。
            attention_mask (Optional[torch.FloatTensor]): 注意力掩码张量，用于指定哪些位置是填充的。
            token_type_ids (Optional[torch.LongTensor]): 分段类型IDs张量，用于区分不同句子的位置。
            position_ids (Optional[torch.LongTensor]): 位置IDs张量，用于指定输入token的绝对位置。
            head_mask (Optional[torch.FloatTensor]): 多头注意力机制的掩码张量，用于指定哪些头部是无效的。
            inputs_embeds (Optional[torch.FloatTensor]): 嵌入向量的输入张量。
            labels (Optional[torch.LongTensor]): 标签张量，用于多选分类任务的真实标签。
            output_attentions (Optional[bool]): 是否输出注意力权重。
            output_hidden_states (Optional[bool]): 是否输出隐藏状态。
            return_dict (Optional[bool]): 是否返回字典格式的输出。

        Returns:
            MultipleChoiceModelOutput: 包含模型输出的对象，包括分类预测和其他可选的输出（如注意力权重、隐藏状态）。
        """
        # 省略了具体的前向传播逻辑，由于这里没有具体代码实现，无法添加进一步的注释。
        pass
        ) -> Union[Tuple, MultipleChoiceModelOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the multiple choice classification loss. Indices should be in `[0, ...,
            num_choices-1]` where `num_choices` is the size of the second dimension of the input tensors. (See
            `input_ids` above)
        """
        # 如果 return_dict 为 None，则使用模型配置中的 use_return_dict 参数来决定返回类型
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 计算输入的 num_choices，如果 input_ids 不为 None，则为 input_ids 的第二维大小，否则为 inputs_embeds 的第二维大小
        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

        # 将 input_ids 展平为二维张量，每行表示一个选择项的输入 ids；如果 input_ids 为 None，则 input_ids 也为 None
        input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        # 将 attention_mask 展平为二维张量，每行表示一个选择项的 attention mask；如果 attention_mask 为 None，则为 None
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        # 将 token_type_ids 展平为二维张量，每行表示一个选择项的 token type ids；如果 token_type_ids 为 None，则为 None
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        # 将 position_ids 展平为二维张量，每行表示一个选择项的 position ids；如果 position_ids 为 None，则为 None
        position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        # 将 inputs_embeds 展平为三维张量，每行表示一个选择项的嵌入；如果 inputs_embeds 为 None，则为 None
        inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )

        # 使用模型的前向传播方法 canine 进行推断
        outputs = self.canine(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 从模型输出中获取汇聚的输出（通常是 pooler 输出）
        pooled_output = outputs[1]

        # 对汇聚的输出应用 dropout
        pooled_output = self.dropout(pooled_output)
        # 将 dropout 后的输出送入分类器得到 logits
        logits = self.classifier(pooled_output)
        # 将 logits 重新调整为二维张量，每行表示一个选择项的分类结果
        reshaped_logits = logits.view(-1, num_choices)

        # 初始化损失值为 None
        loss = None
        # 如果 labels 不为 None，则计算交叉熵损失
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)

        # 如果不需要返回一个字典形式的结果
        if not return_dict:
            # 组装输出元组，包含重新调整的 logits 和可能的隐藏状态
            output = (reshaped_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 返回多选模型的输出，包括损失、调整后的 logits、隐藏状态和注意力分布
        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 使用 @add_start_docstrings 装饰器为类添加文档字符串，描述了该模型在 token 分类任务上的应用
@add_start_docstrings(
    """
    CANINE Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
    Named-Entity-Recognition (NER) tasks.
    """,
    CANINE_START_DOCSTRING,  # 引用预定义的 CANINE_START_DOCSTRING 常量
)
# 定义 CanineForTokenClassification 类，继承自 CaninePreTrainedModel
class CanineForTokenClassification(CaninePreTrainedModel):
    
    # 初始化方法，接收一个配置对象 config
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)
        # 从配置对象中获取标签数目并保存
        self.num_labels = config.num_labels
        
        # 创建 CANINE 模型实例
        self.canine = CanineModel(config)
        # 添加 dropout 层，使用配置中的隐藏层 dropout 概率
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # 创建线性分类器，输入大小为隐藏层大小，输出大小为标签数目
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        
        # 初始化权重并进行最终处理
        self.post_init()

    # 使用 @add_start_docstrings_to_model_forward 装饰器为 forward 方法添加文档字符串，描述输入参数
    # 使用 @replace_return_docstrings 装饰器替换返回值的文档字符串，输出类型为 TokenClassifierOutput，配置类为 _CONFIG_FOR_DOC
    @add_start_docstrings_to_model_forward(CANINE_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=TokenClassifierOutput, config_class=_CONFIG_FOR_DOC)
    # 定义 forward 方法，处理输入并返回模型输出
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,

        # forward 方法接收多个输入参数，均为可选类型的 torch 张量
        
        # input_ids: 输入 token 的 IDs，长整型张量，可选
        # attention_mask: 注意力遮罩，浮点型张量，可选
        # token_type_ids: token 类型 IDs，长整型张量，可选
        # position_ids: 位置 IDs，长整型张量，可选
        # head_mask: 头部遮罩，浮点型张量，可选
        # inputs_embeds: 嵌入式输入，浮点型张量，可选
        # labels: 标签，长整型张量，可选
        # output_attentions: 是否输出注意力，布尔类型，可选
        # output_hidden_states: 是否输出隐藏状态，布尔类型，可选
        # return_dict: 是否返回字典形式的输出，布尔类型，可选
        
        # 方法内部进行模型的前向传播计算，返回模型输出
        ) -> Union[Tuple, TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.

        Returns:
            Depending on `return_dict`:
            - If `return_dict=True`, returns a `TokenClassifierOutput` containing `loss`, `logits`, `hidden_states`, and `attentions`.
            - If `return_dict=False`, returns a tuple with `logits` followed by additional outputs.

        Example:

        ```python
        >>> from transformers import AutoTokenizer, CanineForTokenClassification
        >>> import torch

        >>> tokenizer = AutoTokenizer.from_pretrained("google/canine-s")
        >>> model = CanineForTokenClassification.from_pretrained("google/canine-s")

        >>> inputs = tokenizer(
        ...     "HuggingFace is a company based in Paris and New York", add_special_tokens=False, return_tensors="pt"
        ... )

        >>> with torch.no_grad():
        ...     logits = model(**inputs).logits

        >>> predicted_token_class_ids = logits.argmax(-1)

        >>> # Note that tokens are classified rather then input words which means that
        >>> # there might be more predicted token classes than words.
        >>> # Multiple token classes might account for the same word
        >>> predicted_tokens_classes = [model.config.id2label[t.item()] for t in predicted_token_class_ids[0]]
        >>> predicted_tokens_classes  # doctest: +SKIP
        ```

        ```python
        >>> labels = predicted_token_class_ids
        >>> loss = model(**inputs, labels=labels).loss
        >>> round(loss.item(), 2)  # doctest: +SKIP
        ```"""
        # Determine if the output should be in dictionary format based on the `return_dict` argument
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Perform token classification using the Canine model
        outputs = self.canine(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # Extract the sequence output from the model's outputs
        sequence_output = outputs[0]

        # Apply dropout to the sequence output
        sequence_output = self.dropout(sequence_output)

        # Generate logits using the classifier layer
        logits = self.classifier(sequence_output)

        # Initialize loss as None
        loss = None

        # Compute the loss if labels are provided
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        # Prepare the output based on the `return_dict` setting
        if not return_dict:
            # If `return_dict=False`, return a tuple with logits and additional outputs
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # If `return_dict=True`, return a `TokenClassifierOutput` object
        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
@add_start_docstrings(
    """
    CANINE Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
    layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    CANINE_START_DOCSTRING,
)
class CanineForQuestionAnswering(CaninePreTrainedModel):
    """
    CANINE模型，顶部带有用于提取式问答任务（如SQuAD）的跨度分类头部（在隐藏状态输出之上的线性层，用于计算`span start logits`和`span end logits`）。
    继承自CaninePreTrainedModel。
    """

    def __init__(self, config):
        """
        初始化方法，设置模型参数和各层。
        
        Args:
            config (CanineConfig): 模型的配置对象，包含模型的各种参数。
        """
        super().__init__(config)
        self.num_labels = config.num_labels

        # 使用给定的配置初始化CANINE模型
        self.canine = CanineModel(config)
        # 初始化一个线性层，用于输出问题-答案对的标签数
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        # 执行额外的初始化步骤，包括权重初始化和最终处理
        self.post_init()

    @add_start_docstrings_to_model_forward(CANINE_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint="Splend1dchan/canine-c-squad",
        output_type=QuestionAnsweringModelOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output="'nice puppet'",
        expected_loss=8.81,
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
    ):
        """
        前向传播方法，执行模型的前向计算。

        Args:
            input_ids (Optional[torch.LongTensor]): 输入token的ids。
            attention_mask (Optional[torch.FloatTensor]): 注意力掩码，指示哪些tokens需要注意，哪些不需要。
            token_type_ids (Optional[torch.LongTensor]): token类型ids，如segment ids。
            position_ids (Optional[torch.LongTensor]): token位置ids。
            head_mask (Optional[torch.FloatTensor]): 头部掩码，用于指定哪些层的注意力是有效的。
            inputs_embeds (Optional[torch.FloatTensor]): 嵌入的输入。
            start_positions (Optional[torch.LongTensor]): 答案起始位置的ids。
            end_positions (Optional[torch.LongTensor]): 答案结束位置的ids。
            output_attentions (Optional[bool]): 是否返回注意力权重。
            output_hidden_states (Optional[bool]): 是否返回隐藏状态。
            return_dict (Optional[bool]): 是否返回字典格式的输出。

        Returns:
            QuestionAnsweringModelOutput: 包含模型预测结果的输出对象。
        """
        # 实现前向传播逻辑的具体计算，包括如何处理输入和输出
        pass
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
        # 初始化返回字典，如果 return_dict 为 None，则使用模型配置中的设定
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用模型的前向传播方法 `canine`，传入各种输入参数
        outputs = self.canine(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 从模型输出中获取序列输出（通常是 BERT 的最后一层隐藏状态）
        sequence_output = outputs[0]

        # 将序列输出传入问答模型输出层，得到起始位置和结束位置的 logits
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # 如果 start_positions 和 end_positions 都不为 None，则计算损失
            # 如果在多GPU情况下，添加一个维度以匹配 logits 的维度
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # 忽略超出模型输入长度的位置
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            # 使用交叉熵损失函数来计算起始位置和结束位置的损失
            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            # 如果 return_dict 为 False，则返回一个包含损失和 logits 的 tuple
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        # 如果 return_dict 为 True，则返回一个 QuestionAnsweringModelOutput 对象
        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
```