# `.\models\switch_transformers\modeling_switch_transformers.py`

```py
def load_balancing_loss_func(router_probs: torch.Tensor, expert_indices: torch.Tensor) -> float:
    r"""
    计算负载平衡损失函数。

    负载平衡损失函数用于Switch Transformers中，旨在通过调整专家分配概率来实现负载平衡，以优化模型性能。

    Args:
        router_probs (torch.Tensor):
            形状为 [batch_size, sequence_length, num_experts] 的输入概率张量，表示每个位置选择每个专家的概率。
        expert_indices (torch.Tensor):
            形状为 [batch_size, sequence_length] 的整数张量，表示每个位置选择的专家索引。

    Returns:
        float:
            标量，表示计算得到的负载平衡损失值。
    """
    num_groups, tokens_per_group, _ = router_probs.shape
    # 计算对数概率的和，对应于每个位置的专家选择概率
    log_z = torch.logsumexp(router_probs, dim=-1)
    # 计算负载平衡损失，以提高模型的稳定性
    balancing_loss = log_z**2
    # 返回平均负载平衡损失
    return torch.sum(balancing_loss) / (num_groups * tokens_per_group)
    # 计算辅助负载平衡损失，类似于Switch Transformer中的实现，使用PyTorch实现。
    
    # 查看Switch Transformer论文（https://arxiv.org/abs/2101.03961）以获取更多细节。
    # 此函数实现论文中第4到第6方程中的损失函数，旨在惩罚专家之间路由过于不平衡的情况。
    
    # 参数:
    #   router_probs (`torch.Tensor`):
    #       每个令牌分配给每个专家的概率。形状为 [batch_size, seqeunce_length, num_experts]。
    #   expert_indices (`torch.Tensor`):
    #       形状为 [batch_size, seqeunce_length] 的索引张量，用于标识每个令牌选择的专家。
    
    # 返回:
    #   辅助损失值。
    num_experts = router_probs.shape[-1]
    
    # 将专家索引转换为int64类型，否则独热编码将失败
    if expert_indices.dtype != torch.int64:
        expert_indices = expert_indices.to(torch.int64)
    
    # 如果专家索引张量的维度为2，则添加一个维度以匹配独热编码的要求
    if len(expert_indices.shape) == 2:
        expert_indices = expert_indices.unsqueeze(2)
    
    # 创建独热编码，标识每个令牌是否分配给特定专家
    expert_mask = torch.nn.functional.one_hot(expert_indices, num_experts)
    
    # 对于每个令牌，确定其是否被路由到某个专家
    expert_mask = torch.max(expert_mask, axis=-2).values
    
    # 将独热编码张量转换为float32类型，否则计算平均值时会失败
    expert_mask = expert_mask.to(torch.float32)
    
    # 计算每个组和专家的令牌比例，用于平均计算
    tokens_per_group_and_expert = torch.mean(expert_mask, axis=-2)
    
    # 计算每个组和专家的路由概率，用于平均计算
    router_prob_per_group_and_expert = torch.mean(router_probs, axis=-2)
    
    # 返回平均令牌比例和路由概率乘积的平均值，乘以专家数量的平方
    return torch.mean(tokens_per_group_and_expert * router_prob_per_group_and_expert) * (num_experts**2)
class SwitchTransformersTop1Router(nn.Module):
    """
    Router using tokens choose top-1 experts assignment.

    This router uses the same mechanism as in Switch Transformer (https://arxiv.org/abs/2101.03961) and V-MoE
    (https://arxiv.org/abs/2106.05974): tokens choose their top experts. Items are sorted by router_probs and then
    routed to their choice of expert until the expert's expert_capacity is reached. **There is no guarantee that each
    token is processed by an expert**, or that each expert receives at least one token.
    """

    def __init__(self, config: SwitchTransformersConfig):
        super().__init__()
        self.num_experts = config.num_experts  # 设置专家数量
        self.expert_capacity = config.expert_capacity  # 每个专家的容量
        self.classifier = nn.Linear(config.hidden_size, self.num_experts, bias=config.router_bias)
        self.jitter_noise = config.router_jitter_noise  # 噪声抖动大小
        self.ignore_padding_tokens = config.router_ignore_padding_tokens  # 是否忽略填充标记
        self.dtype = getattr(torch, config.router_dtype)  # 指定的张量数据类型

    def _compute_router_probabilities(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Computes router probabilities from input hidden states.

        Args:
            hidden_states (`torch.Tensor`):
                (batch_size, sequence_length, hidden_dim) from which router probabilities are computed.
        Returns:
            router_probabilities (`torch.Tensor`):
                Tensor of shape (batch_size, sequence_length, num_experts) corresponding to the probabilities for each
                token and expert. Used for routing tokens to experts.
            router_logits (`torch.Tensor`):
                Logits tensor of shape (batch_size, sequence_length, num_experts) corresponding to raw router logits.
                This is used later for computing router z-loss.
        """
        # 使用float32以确保稳定性。参见https://arxiv.org/abs/2101.03961中关于“选择性精度”的讨论。
        # 还存储先前的dtype以便将输出转换回先前的dtype
        self.input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(self.dtype)  # 将输入张量转换为指定的数据类型

        if self.training and self.jitter_noise > 0:
            # 如果处于训练模式且设置了抖动噪声，则通过乘以均匀分布的随机数添加噪声
            hidden_states *= torch.empty_like(hidden_states).uniform_(1.0 - self.jitter_noise, 1.0 + self.jitter_noise)

        # Shape: [num_groups, tokens_per_group, num_experts]
        self._cast_classifier()  # 调用内部函数_cast_classifier()
        router_logits = self.classifier(hidden_states)  # 使用分类器计算路由器的逻辑输出

        # 应用softmax并将数据类型转回原始的`dtype`
        router_probabilities = nn.functional.softmax(router_logits, dim=-1, dtype=self.dtype).to(self.input_dtype)
        return router_probabilities, router_logits  # 返回路由器概率和logits
    def _cast_classifier(self):
        r"""
        `bitsandbytes` `Linear8bitLt` layers does not support manual casting Therefore we need to check if they are an
        instance of the `Linear8bitLt` class by checking special attributes.
        """
        # 检查 self.classifier 是否为 Linear8bitLt 类的实例，如果不是，则转换其数据类型
        if not (hasattr(self.classifier, "SCB") or hasattr(self.classifier, "CB")):
            self.classifier = self.classifier.to(self.dtype)

    def forward(self, hidden_states: torch.Tensor) -> Tuple:
        r"""
        Generic forward function for every Router class. Each Router expects to have the same input hidden states
        (`hidden_states`) corresponding to the hidden states for each token, the `expert_capacity` corresponding to the
        number of tokens the Router will send to each expert, some Routers can send up to few tokens to each expert.

        Each Router works as the following: it expects the hidden states for each token, gets the `router_probs` and
        `router_logits` from the `router_weights`. This will assign for each token, the raw probability to be assigned
        to an expert. Then each Router class will have to define its own `_compute_routing_instructions`.

        Args:
            hidden_states (`torch.Tensor`) :
                [num_groups, tokens_per_group, hidden_dim] inputs to send to experts.
        Returns:
            Tuple[`torch.Tensor`, `torch.Tensor`, `torch.Tensor`] Tuple containing the expert index, the router probs
            and the router logits. The router probabilities and logits are required to compute the loss.
        """
        # 计算路由概率和路由 logits
        router_probs, router_logits = self._compute_router_probabilities(hidden_states)

        # 根据概率选择每个 token 分配的 expert 索引
        expert_index = torch.argmax(router_probs, dim=-1)
        expert_index = torch.nn.functional.one_hot(expert_index, num_classes=self.num_experts)

        # 通过累加判断是否超出 expert 的容量限制，并进行掩码处理
        token_priority = torch.cumsum(expert_index, dim=-2)
        expert_capacity_mask = token_priority <= self.expert_capacity
        expert_index = expert_index * expert_capacity_mask

        # 计算最大的路由概率，用于计算损失
        router_probs = torch.max(router_probs, dim=-1).values.unsqueeze(-1)
        
        return expert_index, router_probs, router_logits
# 复制并修改自transformers.models.t5.modeling_t5.py中的T5LayerNorm类，更名为SwitchTransformersLayerNorm
class SwitchTransformersLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        初始化层规范化模块，LayerNorm按照SwitchTransformers风格处理，不使用偏差和平均值减去操作。
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))  # 初始化缩放参数
        self.variance_epsilon = eps  # 防止除以零的情况，设置一个很小的正数

    def forward(self, hidden_states):
        """
        执行前向传播，计算并应用在hidden_states上的层规范化。
        无需减去平均值，使用根均方计算变差。
        """
        # 将隐藏状态转化为半精度浮点数，进行计算后再转换回来进行层规范化
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        # 如果权重使用半精度浮点数或bfloat16，将隐藏状态转换到相同类型
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        return self.weight * hidden_states

# 将SwitchTransformersLayerNorm添加到ALL_LAYERNORM_LAYERS列表中以供使用

# 复制并修改自transformers.models.t5.modeling_t5.py中的T5DenseActDense类，更名并修改为SwitchTransformersDenseActDense
class SwitchTransformersDenseActDense(nn.Module):
    def __init__(self, config: SwitchTransformersConfig):
        """
        初始化稠密激活密集层模块，包含输入线性变换、激活函数应用、模版速率下采样线性变换。
        """
        super().__init__()
        self.wi = nn.Linear(config.d_model, config.d_ff, bias=False)  # 第一层线性变换
        self.wo = nn.Linear(config.d_ff, config.d_model, bias=False)  # 输出层线性变换
        self.dropout = nn.Dropout(config.dropout_rate)  # 带dropout的下采样层
        self.act = ACT2FN[config.dense_act_fn]  # 激活函数

    def forward(self, hidden_states):
        """
        执行前向传播，通过线性变换、激活函数、dropout和线性变换逐层更新hidden_states。
        注意转换类型要和权重一致，对于tensor的转换也会基于数据类型的规则进行处理。
        """
        hidden_states = self.wi(hidden_states)  # 应用线性变换
        hidden_states = self.act(hidden_states)  # 激活函数处理
        hidden_states = self.dropout(hidden_states)  # 降低过拟合的dropout步骤
        # 确保权重和计算结果数据类型一致，在某些情况下需要转换类型
        if (
            isinstance(self.wo.weight, torch.Tensor)
            and hidden_states.dtype != self.wo.weight.dtype
            and self.wo.weight.dtype != torch.int8
        ):
            hidden_states = hidden_states.to(self.wo.weight.dtype)
        hidden_states = self.wo(hidden_states)  # 应用最终的线性变换

        return hidden_states

# 定义SparseMLP类，实现Switch Transformers的稀疏多层感知机（MLP）模块
class SwitchTransformersSparseMLP(nn.Module):
    """
    实现了Switch Transformers稀疏多层感知机（MLP）模块的特异性参数和结构，包括路由模块和专家模块。
    """

    def __init__(self, config: SwitchTransformersConfig, expert_class: nn.Module = SwitchTransformersDenseActDense):
        """
        根据配置初始化SparseMLP类，包括所需的路由器和专家模块。
        """
        super().__init__()
        # 根据不同策略获取路由层
        self.router = SwitchTransformersTop1Router(config)
        # 初始化并配置专家模块列表
        self.experts = nn.ModuleDict()
        for idx in range(config.num_experts):
            expert_name = f"expert_{idx}"
            self.experts[expert_name] = expert_class(config)  # 注册指定配置的专家模块
    def forward(self, hidden_states):
        r"""
        Hold on, this will be slightly tricky to understand In the correct order, a MoE layer does the following:

        1- Gets the `router_mask` from the router. The shape of the mask is `(batch_size, sequence_length, num_expert)`
        and corresponds to the argmax of the `router_probs`. The probabilities are needed in the computation of the
        hidden states : they are broadcasted to the hidden states values (can be interpreted as a scaling factor).

        2- Dispatch the tokens to its associated experts. We do a classic for loop over the experts and assign for each
        expert the corresponding hidden states.

        """
        # Step 1: Get the router_mask from the router as well as the probabilities
        # 从路由器获取路由器掩码、概率和对数概率
        router_mask, router_probs, router_logits = self.router(hidden_states)
        
        # 根据路由器掩码的最大值索引确定每个 token 分配给哪个专家
        expert_index = torch.argmax(router_mask, dim=-1)

        # 备注: 由于引入的路由器可能并不总是将所有 token 映射到一个路由器上，因此有些隐藏状态可能在层与层之间保持不变。
        # 因此在更新之前需要克隆隐藏状态。

        # 克隆隐藏状态，准备更新仅选择的部分
        next_states = hidden_states.clone()
        
        # 遍历专家列表，为每个专家分配对应的隐藏状态
        for idx, expert in enumerate(self.experts.values()):
            token_indices = router_mask[:, :, idx].bool()  # 获取当前专家对应的 token 索引
            next_states[token_indices] = expert(hidden_states[token_indices]).to(next_states.dtype)

        # 更新隐藏状态，乘以路由器概率作为缩放因子
        hidden_states = router_probs * next_states
        
        # 返回更新后的隐藏状态以及路由器的输出信息（对数概率和专家索引）
        return hidden_states, (router_logits, expert_index)
# 定义一个Switch Transformers的注意力模块，继承自PyTorch的nn.Module类
class SwitchTransformersAttention(nn.Module):
    """
    Switch Transformers Attention module, based on PyTorch's nn.Module.

    This module is responsible for handling the attention mechanism within Switch Transformers.
    """

    def __init__(self, config: SwitchTransformersConfig):
        super().__init__()

        # 初始化时根据给定的配置参数创建注意力层
        self.self = SwitchTransformersSelfAttention(config)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states, attention_mask=None, head_mask=None, output_attentions=False):
        """
        Forward pass for the Switch Transformers Attention module.

        Args:
            hidden_states (`torch.Tensor`):
                The input hidden states for the attention calculation.
            attention_mask (`torch.Tensor`, optional):
                Mask to avoid performing attention on padding tokens.
            head_mask (`torch.Tensor`, optional):
                Mask to nullify specific heads of the attention tensors.
            output_attentions (`bool`, optional):
                Whether to return attentions tensors.

        Returns:
            `torch.Tensor` or (`torch.Tensor`, `torch.Tensor`):
                Depending on `output_attentions`, either returns the contextualized representation
                or a tuple with the contextualized representation and the attention tensors.
        """

        # 使用Switch Transformers自注意力层计算注意力
        self_output = self.self(hidden_states, attention_mask, head_mask, output_attentions)

        # 如果需要输出注意力张量，则将其返回；否则只返回上下文表示
        if output_attentions:
            attention_outputs = self_output[1]  # attention_outputs is a tuple (self_output[1])
            outputs = self_output[0]  # self_output[0] is the contextualized representation
            return outputs, attention_outputs
        else:
            return self_output  # Return just the contextualized representation
    # 初始化函数，接受一个配置对象和一个布尔值参数，用于设置是否具有相对注意力偏置
    def __init__(self, config: SwitchTransformersConfig, has_relative_attention_bias=False):
        # 调用父类的初始化函数
        super().__init__()
        
        # 设置当前层是否为解码器
        self.is_decoder = config.is_decoder
        # 设置是否具有相对注意力偏置
        self.has_relative_attention_bias = has_relative_attention_bias
        # 设置相对注意力的桶数量
        self.relative_attention_num_buckets = config.relative_attention_num_buckets
        # 设置相对注意力的最大距离
        self.relative_attention_max_distance = config.relative_attention_max_distance
        # 设置模型的维度
        self.d_model = config.d_model
        # 设置键值投影的维度
        self.key_value_proj_dim = config.d_kv
        # 设置注意力头的数量
        self.n_heads = config.num_heads
        # 设置dropout率
        self.dropout = config.dropout_rate
        # 计算内部维度，即注意力头乘以键值投影的维度
        self.inner_dim = self.n_heads * self.key_value_proj_dim

        # 初始化线性层，用于查询、键、值、输出
        self.q = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.k = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.v = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.o = nn.Linear(self.inner_dim, self.d_model, bias=False)

        # 如果需要相对注意力偏置，初始化相对注意力偏置的嵌入层
        if self.has_relative_attention_bias:
            self.relative_attention_bias = nn.Embedding(self.relative_attention_num_buckets, self.n_heads)
        
        # 初始化已剪枝的注意力头集合为空集
        self.pruned_heads = set()
        # 禁用梯度检查点
        self.gradient_checkpointing = False

    # 剪枝注意力头的方法
    def prune_heads(self, heads):
        # 如果没有需要剪枝的头，则直接返回
        if len(heads) == 0:
            return
        
        # 找到需要剪枝的头以及它们的索引
        heads, index = find_pruneable_heads_and_indices(
            heads, self.n_heads, self.key_value_proj_dim, self.pruned_heads
        )
        
        # 对线性层进行剪枝
        self.q = prune_linear_layer(self.q, index)
        self.k = prune_linear_layer(self.k, index)
        self.v = prune_linear_layer(self.v, index)
        self.o = prune_linear_layer(self.o, index, dim=1)
        
        # 更新超参数
        self.n_heads = self.n_heads - len(heads)
        self.inner_dim = self.key_value_proj_dim * self.n_heads
        # 将剪枝的头添加到已剪枝的头集合中
        self.pruned_heads = self.pruned_heads.union(heads)
    def _relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        """
        Adapted from Mesh Tensorflow:
        https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593

        Translate relative position to a bucket number for relative attention. The relative position is defined as
        memory_position - query_position, i.e. the distance in tokens from the attending position to the attended-to
        position. If bidirectional=False, then positive relative positions are invalid. We use smaller buckets for
        small absolute relative_position and larger buckets for larger absolute relative_positions. All relative
        positions >=max_distance map to the same bucket. All relative positions <=-max_distance map to the same bucket.
        This should allow for more graceful generalization to longer sequences than the model has been trained on

        Args:
            relative_position: an int32 Tensor - the relative position between memory and query positions
            bidirectional: a boolean - whether the attention is bidirectional
            num_buckets: an integer - number of buckets to categorize relative positions
            max_distance: an integer - maximum distance considered for bucketing

        Returns:
            a Tensor with the same shape as relative_position, containing int32 values in the range [0, num_buckets)
        """
        # Initialize the relative bucket index
        relative_buckets = 0
        
        # Adjust num_buckets if bidirectional attention is disabled
        if bidirectional:
            num_buckets //= 2
            # Calculate relative_buckets based on whether relative_position is positive
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            # Ensure relative_position is non-positive if bidirectional=False
            relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
        
        # relative_position is now in the range [0, inf)

        # Determine if the relative_position is small (less than max_exact)
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        # Calculate relative_position_if_large for positions larger than max_exact
        relative_position_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)
        
        # Clamp relative_position_if_large to num_buckets - 1
        relative_position_if_large = torch.min(
            relative_position_if_large, torch.full_like(relative_position_if_large, num_buckets - 1)
        )

        # Determine final relative_buckets based on whether relative_position is small
        relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)
        
        return relative_buckets
    def compute_bias(self, query_length, key_length, device=None):
        """Compute binned relative position bias"""
        # 如果未指定设备，则使用相对注意力偏置权重的设备
        if device is None:
            device = self.relative_attention_bias.weight.device
        
        # 创建表示上下文位置的张量，范围是 [0, query_length)
        context_position = torch.arange(query_length, dtype=torch.long, device=device)[:, None]
        
        # 创建表示记忆位置的张量，范围是 [0, key_length)
        memory_position = torch.arange(key_length, dtype=torch.long, device=device)[None, :]
        
        # 计算相对位置，形状为 (query_length, key_length)
        relative_position = memory_position - context_position
        
        # 对相对位置进行分桶化处理，返回形状为 (query_length, key_length) 的张量
        relative_position_bucket = self._relative_position_bucket(
            relative_position,
            bidirectional=(not self.is_decoder),
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance,
        )
        
        # 使用相对注意力偏置计算相对位置的值，形状为 (query_length, key_length, num_heads)
        values = self.relative_attention_bias(relative_position_bucket)
        
        # 调整维度顺序，形状变为 (1, num_heads, query_length, key_length)
        values = values.permute([2, 0, 1]).unsqueeze(0)
        
        # 返回计算得到的相对位置值
        return values

    def forward(
        self,
        hidden_states,
        mask=None,
        key_value_states=None,
        position_bias=None,
        past_key_value=None,
        layer_head_mask=None,
        query_length=None,
        use_cache=False,
        output_attentions=False,
# Copied from transformers.models.t5.modeling_t5.T5LayerSelfAttention with T5->SwitchTransformers
class SwitchTransformersLayerSelfAttention(nn.Module):
    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__()
        # 初始化自注意力层，使用SwitchTransformersAttention替代T5中的SelfAttention
        self.SelfAttention = SwitchTransformersAttention(
            config, has_relative_attention_bias=has_relative_attention_bias
        )
        # 初始化层归一化模块，使用SwitchTransformersLayerNorm替代T5中的LayerNorm
        self.layer_norm = SwitchTransformersLayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        # 初始化dropout层，使用config中的dropout_rate
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        output_attentions=False,
    ):
        # 对隐藏状态进行层归一化处理
        normed_hidden_states = self.layer_norm(hidden_states)
        # 使用SwitchTransformersAttention进行自注意力计算
        attention_output = self.SelfAttention(
            normed_hidden_states,
            mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        # 将原始的隐藏状态与注意力输出加权和，然后应用dropout
        hidden_states = hidden_states + self.dropout(attention_output[0])
        # 构建输出元组，包含加权和后的隐藏状态及可能的注意力结果
        outputs = (hidden_states,) + attention_output[1:]  # add attentions if we output them
        return outputs


# Copied from transformers.models.t5.modeling_t5.T5LayerCrossAttention with T5->SwitchTransformers
class SwitchTransformersLayerCrossAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 初始化跨注意力层，使用SwitchTransformersAttention替代T5中的EncDecAttention
        self.EncDecAttention = SwitchTransformersAttention(config, has_relative_attention_bias=False)
        # 初始化层归一化模块，使用SwitchTransformersLayerNorm替代T5中的LayerNorm
        self.layer_norm = SwitchTransformersLayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        # 初始化dropout层，使用config中的dropout_rate
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        hidden_states,
        key_value_states,
        attention_mask=None,
        position_bias=None,
        layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        query_length=None,
        output_attentions=False,
    ):
        # 对隐藏状态进行层归一化处理
        normed_hidden_states = self.layer_norm(hidden_states)
        # 使用SwitchTransformersAttention进行跨注意力计算
        attention_output = self.EncDecAttention(
            normed_hidden_states,
            mask=attention_mask,
            key_value_states=key_value_states,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            query_length=query_length,
            output_attentions=output_attentions,
        )
        # 将原始的隐藏状态与注意力输出加权和，然后应用dropout
        layer_output = hidden_states + self.dropout(attention_output[0])
        # 构建输出元组，包含加权和后的隐藏状态及可能的注意力结果
        outputs = (layer_output,) + attention_output[1:]  # add attentions if we output them
        return outputs


class SwitchTransformersBlock(nn.Module):
    # 该类尚未完成，需要继续实现Switch Transformers中的Block结构
    pass
    # 初始化方法，用于创建一个Switch Transformers层的模型
    def __init__(self, config, has_relative_attention_bias=False, is_sparse=False):
        # 调用父类的初始化方法
        super().__init__()
        # 根据配置确定是否为解码器
        self.is_decoder = config.is_decoder
        # 设置是否为稀疏模式
        self.is_sparse = is_sparse
        # 创建一个空的模块列表，用于存储各层的模块
        self.layer = nn.ModuleList()
        # 将自注意力层添加到模块列表中
        self.layer.append(
            SwitchTransformersLayerSelfAttention(config, has_relative_attention_bias=has_relative_attention_bias)
        )
        # 如果是解码器，则添加交叉注意力层
        if self.is_decoder:
            self.layer.append(SwitchTransformersLayerCrossAttention(config))

        # 添加前馈神经网络层到模块列表中
        self.layer.append(SwitchTransformersLayerFF(config, is_sparse=self.is_sparse))

    # 前向传播方法，用于模型推断
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        encoder_decoder_position_bias=None,
        layer_head_mask=None,
        cross_attn_layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        output_attentions=False,
        output_router_logits=True,
        return_dict=True,
# 定义 SwitchTransformersPreTrainedModel 类，继承自 PreTrainedModel
class SwitchTransformersPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 设置配置类为 SwitchTransformersConfig
    config_class = SwitchTransformersConfig
    # 设置基础模型前缀为 "switch_transformers"
    base_model_prefix = "switch_transformers"
    # 支持梯度检查点
    supports_gradient_checkpointing = True
    # 不需要分割的模块
    _no_split_modules = ["SwitchTransformersBlock"]

    # 定义属性 dummy_inputs，返回一个包含输入和注意力掩码的字典
    @property
    def dummy_inputs(self):
        input_ids = torch.tensor(DUMMY_INPUTS)
        input_mask = torch.tensor(DUMMY_MASK)
        dummy_inputs = {
            "decoder_input_ids": input_ids,
            "input_ids": input_ids,
            "decoder_attention_mask": input_mask,
        }
        return dummy_inputs

    # 定义内部函数 _shift_right，用于将输入 ids 向右移动
    def _shift_right(self, input_ids):
        decoder_start_token_id = self.config.decoder_start_token_id
        pad_token_id = self.config.pad_token_id

        # 如果 decoder_start_token_id 未定义，抛出数值错误
        if decoder_start_token_id is None:
            raise ValueError(
                "self.model.config.decoder_start_token_id has to be defined. In SwitchTransformers it is usually set"
                " to the pad_token_id. See SwitchTransformers docs for more information"
            )

        # 将输入向右移动
        if is_torch_fx_proxy(input_ids):
            # 对于代理对象，不支持原生的项目赋值
            shifted_input_ids = torch.full(input_ids.shape[:-1] + (1,), decoder_start_token_id)
            shifted_input_ids = torch.cat([shifted_input_ids, input_ids[..., :-1]], dim=-1)
        else:
            shifted_input_ids = input_ids.new_zeros(input_ids.shape)
            shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
            shifted_input_ids[..., 0] = decoder_start_token_id

        # 如果 pad_token_id 未定义，抛出数值错误
        if pad_token_id is None:
            raise ValueError("self.model.config.pad_token_id has to be defined.")
        # 用 pad_token_id 替换标签中可能存在的 -100 值
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

        return shifted_input_ids

# 定义 SwitchTransformersStack 类，继承自 SwitchTransformersPreTrainedModel
class SwitchTransformersStack(SwitchTransformersPreTrainedModel):
    def __init__(self, config, embed_tokens=None):
        super().__init__(config)  # 调用父类的构造函数，初始化模型基础配置

        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)  # 初始化词嵌入层

        if embed_tokens is not None:
            self.embed_tokens.weight = embed_tokens.weight  # 如果传入了预训练的词嵌入，使用传入的词嵌入权重

        self.is_decoder = config.is_decoder  # 标记模型是否为解码器

        sparse_step = config.decoder_sparse_step if self.is_decoder else config.encoder_sparse_step  # 设置稀疏步长
        config.num_layers = config.num_decoder_layers if self.is_decoder else config.num_layers  # 设置层数
        self.block = nn.ModuleList()  # 创建模块列表，用于存储多层块

        # 循环创建多层块
        for i in range(config.num_layers):
            is_sparse = (i % sparse_step == 1 or sparse_step == 1) if sparse_step > 0 else False  # 判断当前层是否为稀疏层

            self.block.append(
                SwitchTransformersBlock(config, has_relative_attention_bias=bool(i == 0), is_sparse=is_sparse)
            )  # 将创建的块添加到模块列表中

        self.final_layer_norm = SwitchTransformersLayerNorm(config.d_model, eps=config.layer_norm_epsilon)  # 初始化最终的层归一化层
        self.dropout = nn.Dropout(config.dropout_rate)  # 初始化丢弃层，用于防止过拟合

        # 初始化权重并应用最终处理
        self.post_init()  # 执行额外的初始化步骤

        self.device_map = None  # 设备映射设为None
        self.gradient_checkpointing = False  # 梯度检查点设为False

    def get_input_embeddings(self):
        return self.embed_tokens  # 返回输入的词嵌入层

    def set_input_embeddings(self, new_embeddings):
        self.embed_tokens = new_embeddings  # 设置新的输入词嵌入层

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        inputs_embeds=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        output_router_logits=True,
        return_dict=None,
# SWITCH_TRANSFORMERS_START_DOCSTRING 是一个长字符串，包含了关于 SWITCH_TRANSFORMERS 模型的详细介绍和相关文献引用。
# 该模型由 William Fedus、Barret Zoph 和 Noam Shazeer 提出，是一种类似于 T5 的编码-解码模型，具有稀疏的前馈结构，采用专家混合 (MoE) 架构。
# 继承自 PreTrainedModel 类，可以查看超类文档以了解该库为所有模型实现的通用方法，如下载或保存模型、调整输入嵌入大小、修剪头等。
# 也是 PyTorch 的 torch.nn.Module 子类，可以像普通 PyTorch 模块一样使用，并参考 PyTorch 文档了解一般使用和行为。
# 参数:
#     config ([SwitchTransformersConfig]): 模型配置类，包含模型的所有参数。使用配置文件初始化不会加载与模型关联的权重，只加载配置。查看 ~PreTrainedModel.from_pretrained 方法以加载模型权重。
SWITCH_TRANSFORMERS_START_DOCSTRING = r"""
    The SWITCH_TRANSFORMERS model was proposed in [Switch Transformers: Scaling to Trillion Parameter Models with
    Simple and Efficient Sparsity](https://arxiv.org/abs/2101.03961) by [William
    Fedus](https://arxiv.org/search/cs?searchtype=author&query=Fedus%2C+W), [Barret
    Zoph](https://arxiv.org/search/cs?searchtype=author&query=Zoph%2C+B), and [Noam
    Shazeer](https://arxiv.org/search/cs?searchtype=author&query=Shazeer%2C+N). It's an encoder-decoder T5-like model
    with sparse Feed Forward that stands for Mixture of Experts (MoE) architecture.

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`SwitchTransformersConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

# SWITCH_TRANSFORMERS_INPUTS_DOCSTRING 是一个空字符串，可能用于描述 SWITCH_TRANSFORMERS 模型的输入。
SWITCH_TRANSFORMERS_INPUTS_DOCSTRING = r"""
"""

# SWITCH_TRANSFORMERS_ENCODER_INPUTS_DOCSTRING 是一个空字符串，可能用于描述 SWITCH_TRANSFORMERS 编码器的输入。
SWITCH_TRANSFORMERS_ENCODER_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            # 输入序列标记在词汇表中的索引。SWITCH_TRANSFORMERS 是一个带有相对位置嵌入的模型，因此可以在左右两侧填充输入。

            # 可以使用 `AutoTokenizer` 获取索引。详见 `PreTrainedTokenizer.encode` 和 `PreTrainedTokenizer.__call__`。

            # 如需了解如何为预训练准备 `input_ids`，请查看 [SWITCH_TRANSFORMERS Training](./switch_transformers#training)。

        attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            # 避免对填充的标记索引执行注意力操作的掩码。掩码值在 `[0, 1]` 之间：

            # - 1 表示**不掩盖**的标记，
            # - 0 表示**掩盖**的标记。

            # [什么是注意力掩码?](../glossary#attention-mask)

        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            # 用于使自注意力模块中选择的头部失效的掩码。掩码值在 `[0, 1]` 之间：

            # - 1 表示头部**不被掩盖**，
            # - 0 表示头部**被掩盖**。

        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            # 可选项，直接传递嵌入表示而不是传递 `input_ids`。如果您希望更好地控制如何将 `input_ids` 索引转换为相关联的向量，而不是使用模型的内部嵌入查找矩阵，则此选项很有用。

        output_attentions (`bool`, *optional*):
            # 是否返回所有注意力层的注意力张量。有关详细信息，请查看返回张量下的 `attentions`。

        output_hidden_states (`bool`, *optional*):
            # 是否返回所有层的隐藏状态。有关详细信息，请查看返回张量下的 `hidden_states`。

        output_router_logits (`bool`, *optional*):
            # 是否返回所有路由器的 logits。这对计算路由器损失很有用，在推断期间不应返回。

        return_dict (`bool`, *optional*):
            # 是否返回 `~utils.ModelOutput` 而不是普通元组。
"""

# 未来警告消息：head_mask 参数已分为两个输入参数 - head_mask 和 decoder_head_mask
__HEAD_MASK_WARNING_MSG = """
The input argument `head_mask` was split into two arguments `head_mask` and `decoder_head_mask`. Currently,
`decoder_head_mask` is set to copy `head_mask`, but this feature is deprecated and will be removed in future versions.
If you do not want to use any `decoder_head_mask` now, please set `decoder_head_mask = torch.ones(num_layers,
num_heads)`.
"""


@add_start_docstrings(
    "The bare SWITCH_TRANSFORMERS Model transformer outputting raw hidden-states without any specific head on top.",
    SWITCH_TRANSFORMERS_START_DOCSTRING,
)
class SwitchTransformersModel(SwitchTransformersPreTrainedModel):
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight"]

    def __init__(self, config: SwitchTransformersConfig):
        super().__init__(config)
        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        # 创建编码器和解码器的配置副本，并设置相关参数
        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        # 初始化编码器
        self.encoder = SwitchTransformersStack(encoder_config, self.shared)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        # 初始化解码器
        self.decoder = SwitchTransformersStack(decoder_config, self.shared)

        # 初始化权重并应用最终处理
        self.post_init()

        # 模型并行化
        self.device_map = None

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        # 设置新的输入嵌入层
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    def _tie_weights(self):
        # 如果配置要求词嵌入层权重共享，则共享编码器和解码器的嵌入层权重
        if self.config.tie_word_embeddings:
            self._tie_or_clone_weights(self.encoder.embed_tokens, self.shared)
            self._tie_or_clone_weights(self.decoder.embed_tokens, self.shared)

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        # 剪枝模型的注意力头
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(SWITCH_TRANSFORMERS_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqMoEModelOutput, config_class=_CONFIG_FOR_DOC)
    # 定义模型的前向传播方法，接收多个输入参数并返回模型输出
    def forward(
        self,
        # 输入序列的 token IDs，类型为长整型张量，可选
        input_ids: Optional[torch.LongTensor] = None,
        # 输入序列的注意力掩码，类型为浮点型张量，可选
        attention_mask: Optional[torch.FloatTensor] = None,
        # 解码器输入序列的 token IDs，类型为长整型张量，可选
        decoder_input_ids: Optional[torch.LongTensor] = None,
        # 解码器输入序列的注意力掩码，类型为布尔型张量，可选
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        # 头部掩码，类型为浮点型张量，用于控制哪些头部不参与计算，可选
        head_mask: Optional[torch.FloatTensor] = None,
        # 解码器头部掩码，类型为浮点型张量，可选
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        # 跨注意力头部掩码，类型为张量，用于跨注意力层的头部控制，可选
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        # 编码器输出，类型为元组的元组，可选
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        # 过去的键值对，类型为元组的元组，用于缓存，可选
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        # 输入的嵌入向量，类型为张量，可选
        inputs_embeds: Optional[torch.Tensor] = None,
        # 解码器输入的嵌入向量，类型为张量，可选
        decoder_inputs_embeds: Optional[torch.Tensor] = None,
        # 是否使用缓存，类型为布尔值，可选
        use_cache: Optional[bool] = None,
        # 是否输出注意力权重，类型为布尔值，可选
        output_attentions: Optional[bool] = None,
        # 是否输出隐藏状态，类型为布尔值，可选
        output_hidden_states: Optional[bool] = None,
        # 是否输出路由器 logits，类型为布尔值，可选
        output_router_logits: Optional[bool] = None,
        # 是否以字典形式返回结果，类型为布尔值，可选
        return_dict: Optional[bool] = None,

        # 函数参数列表结束，以下为函数体的实现
# 添加注释至类的开头，描述该类的主要功能为带有语言建模头的 SWITCH_TRANSFORMERS 模型
@add_start_docstrings(
    """SWITCH_TRANSFORMERS Model with a `language modeling` head on top.""", SWITCH_TRANSFORMERS_START_DOCSTRING
)
class SwitchTransformersForConditionalGeneration(SwitchTransformersPreTrainedModel):
    # 定义共享权重的键列表，这些权重将被绑定
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight", "lm_head.weight"]

    def __init__(self, config: SwitchTransformersConfig):
        # 调用父类的初始化方法
        super().__init__(config)
        # 设置模型维度为配置中的 d_model 参数
        self.model_dim = config.d_model

        # 创建共享的嵌入层，用于输入编码器和解码器的词汇表
        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        # 复制编码器配置，设定为非解码器模式，并关闭缓存和编码解码器模式
        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        # 初始化编码器堆栈，传入编码器配置和共享的嵌入层
        self.encoder = SwitchTransformersStack(encoder_config, self.shared)

        # 复制解码器配置，设定为解码器模式，并关闭编码解码器模式
        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        # 初始化解码器堆栈，传入解码器配置和共享的嵌入层
        self.decoder = SwitchTransformersStack(decoder_config, self.shared)

        # 创建语言模型头部，线性层，输入维度为 d_model，输出维度为词汇表大小，无偏置
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # 设置路由器的 Z 损失系数和辅助损失系数
        self.router_z_loss_coef = config.router_z_loss_coef
        self.router_aux_loss_coef = config.router_aux_loss_coef

        # 初始化权重并应用最终处理
        self.post_init()

        # 模型并行化相关，设备映射设为 None
        self.device_map = None

    # 获取输入嵌入层对象
    def get_input_embeddings(self):
        return self.shared

    # 设置新的输入嵌入层对象，并将其应用于编码器和解码器
    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    # 绑定编码器和解码器的权重，如果配置要求绑定词嵌入权重
    def _tie_weights(self):
        if self.config.tie_word_embeddings:
            self._tie_or_clone_weights(self.encoder.embed_tokens, self.shared)
            self._tie_or_clone_weights(self.decoder.embed_tokens, self.shared)

    # 设置新的输出嵌入层对象
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    # 获取输出嵌入层对象
    def get_output_embeddings(self):
        return self.lm_head

    # 获取编码器对象
    def get_encoder(self):
        return self.encoder

    # 获取解码器对象
    def get_decoder(self):
        return self.decoder

    # 添加模型前向传播的文档注释，并替换返回文档注释为 Seq2SeqMoEOutput 类型，配置类为 _CONFIG_FOR_DOC
    @add_start_docstrings_to_model_forward(SWITCH_TRANSFORMERS_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqMoEOutput, config_class=_CONFIG_FOR_DOC)
    # 定义一个方法用于模型的前向传播，接收多个可选参数来处理输入和解码器相关的信息
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = True,
        return_dict: Optional[bool] = None,
    ):
        # 返回模型前向传播需要的所有参数，这些参数包括输入的编码和解码信息
        # 这个方法用于模型的前向计算，处理输入数据并生成输出结果
        ...

    # 定义一个内部方法，用于解析路由器输出，提取路由器的逻辑和专家索引
    def _unpack_router_logits(self, router_outputs):
        total_router_logits = []
        total_expert_indexes = []
        for router_output in router_outputs:
            if len(router_output[0].shape) > 1:
                router_logits, expert_indexes = router_output
                total_router_logits.append(router_logits)
                total_expert_indexes.append(expert_indexes)
        # 将所有路由器的逻辑和专家索引拼接在一起，并按指定的维度连接
        return torch.cat(total_router_logits, dim=1), torch.cat(total_expert_indexes, dim=1)

    # 定义一个方法，用于为生成准备输入数据
    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs,
    ):
        # 如果使用了过去的键值（past_key_values），则需要裁剪decoder_input_ids
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]

            # 一些生成方法已经仅传递了最后一个输入ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # 默认使用旧的行为：仅保留最后一个ID
                remove_prefix_length = input_ids.shape[1] - 1

            # 裁剪输入的ID，保留从remove_prefix_length到末尾的部分
            input_ids = input_ids[:, remove_prefix_length:]

        # 返回一个包含生成所需输入的字典
        return {
            "decoder_input_ids": input_ids,
            "past_key_values": past_key_values,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
        }

    # 定义一个方法，从标签（labels）中准备解码器的输入ID
    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        # 调用内部方法 _shift_right，将标签（labels）右移一位，作为解码器的输入
        return self._shift_right(labels)
    # 重新排序缓存中的过去键值对，根据beam_idx参数进行重排序
    def _reorder_cache(self, past_key_values, beam_idx):
        # 如果decoder的过去状态未包含在输出中
        # 禁用快速解码，并且无需重新排序
        if past_key_values is None:
            logger.warning("You might want to consider setting `use_cache=True` to speed up decoding")
            return past_key_values

        # 初始化重新排序后的decoder过去状态的元组
        reordered_decoder_past = ()
        # 遍历每一层的过去状态
        for layer_past_states in past_key_values:
            # 初始化重新排序后的当前层过去状态的元组
            reordered_layer_past_states = ()
            # 遍历当前层的每一个过去状态
            for layer_past_state in layer_past_states:
                # 根据beam_idx参数选择正确的批次索引，以获取正确的过去状态
                reordered_layer_past_states = reordered_layer_past_states + (
                    layer_past_state.index_select(0, beam_idx.to(layer_past_state.device)),
                )

            # 检查重新排序后的当前层过去状态的形状是否与原始的过去状态一致
            if reordered_layer_past_states[0].shape != layer_past_states[0].shape:
                raise ValueError(
                    "expected reordered_layer_past_states to have the same shape than layer_past_states, "
                    f"but got {reordered_layer_past_states[0].shape} and {layer_past_states[0].shape}"
                )
            # 检查重新排序后的当前层过去状态的长度是否与原始的过去状态一致
            if len(reordered_layer_past_states) != len(layer_past_states):
                raise ValueError(
                    "expected layer_past_states to have the same length as reordered_layer_past_states, "
                    f"but got {len(layer_past_states)} and {len(reordered_layer_past_states)}"
                )

            # 将当前层重新排序后的过去状态添加到总的重新排序过的decoder过去状态中
            reordered_decoder_past = reordered_decoder_past + (reordered_layer_past_states,)

        # 返回重新排序后的decoder过去状态
        return reordered_decoder_past
# 使用装饰器为类添加文档字符串，描述此类是一个 SWITCH_TRANSFORMERS 模型的编码器，输出编码器的原始隐藏状态而不包含特定的顶部头信息
@add_start_docstrings(
    "The bare SWITCH_TRANSFORMERS Model transformer outputting encoder's raw hidden-states without any specific head"
    " on top.",
    SWITCH_TRANSFORMERS_START_DOCSTRING,
)
class SwitchTransformersEncoderModel(SwitchTransformersPreTrainedModel):
    # 定义权重绑定的键列表，这些权重将与 encoder.embed_tokens.weight 共享
    _tied_weights_keys = ["encoder.embed_tokens.weight"]

    def __init__(self, config: SwitchTransformersConfig):
        # 调用父类的初始化方法，传入配置参数
        super().__init__(config)
        # 创建一个共享的嵌入层，将词汇表大小和模型配置中的 d_model 作为参数
        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        # 复制配置并调整为不使用缓存、非编码-解码模式的编码器配置
        encoder_config = copy.deepcopy(config)
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        # 创建 SwitchTransformersStack 对象作为编码器，并传入共享的嵌入层
        self.encoder = SwitchTransformersStack(encoder_config, self.shared)

        # 初始化权重并应用最终处理
        self.post_init()

        # 模型并行化设置，设备映射初始化为 None
        self.device_map = None

    # 返回共享的嵌入层对象
    def get_input_embeddings(self):
        return self.shared

    # 设置新的输入嵌入层，并更新编码器中的共享嵌入层
    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)

    # 如果配置要求，将编码器的嵌入层权重与共享的嵌入层权重进行绑定
    def _tie_weights(self):
        if self.config.tie_word_embeddings:
            self._tie_or_clone_weights(self.encoder.embed_tokens, self.shared)

    # 返回编码器对象
    def get_encoder(self):
        return self.encoder

    # 对模型的头部进行修剪，heads_to_prune 是一个字典，格式为 {层号: 需要在此层中修剪的头部列表}
    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            # 在指定层中的自注意力模块中修剪头部
            self.encoder.block[layer].layer[0].SelfAttention.prune_heads(heads)

    # 使用装饰器为前向传播方法添加文档字符串，描述其输入参数和返回类型
    @add_start_docstrings_to_model_forward(SWITCH_TRANSFORMERS_ENCODER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=MoEModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = True,
        return_dict: Optional[bool] = None,
        # 省略了函数体，因为代码截断在此处，应继续注释直到代码结束
        ) -> Union[Tuple[torch.FloatTensor], MoEModelOutput]:
        r"""
        Returns the encoder outputs based on the given inputs and optional configurations.

        Example:

        ```
        >>> from transformers import AutoTokenizer, SwitchTransformersEncoderModel

        >>> tokenizer = AutoTokenizer.from_pretrained("google/switch-base-8")
        >>> model = SwitchTransformersEncoderModel.from_pretrained("google/switch-base-8")
        >>> input_ids = tokenizer(
        ...     "Studies have been shown that owning a dog is good for you", return_tensors="pt"
        ... ).input_ids  # Batch size 1
        >>> outputs = model(input_ids=input_ids)
        >>> last_hidden_states = outputs.last_hidden_state
        ```
        """
        # Determine whether to use the return_dict based on the provided argument or default configuration
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Pass input arguments to the encoder module for processing
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_router_logits=output_router_logits,
            return_dict=return_dict,
        )

        # Return the outputs generated by the encoder module
        return encoder_outputs
```