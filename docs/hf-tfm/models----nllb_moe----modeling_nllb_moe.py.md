# `.\models\nllb_moe\modeling_nllb_moe.py`

```py
# 定义一个函数，根据输入的 `input_ids` 张量，将其中的 token 向右移动一位
def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    将输入的 `input_ids` 张量中的 token 向右移动一位。
    """
    # 创建一个与 `input_ids` 形状相同的全零张量
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    # 将 `input_ids` 张量中的数据复制到新张量中，每个序列向右移动一个位置
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    # 将每个序列的第一个 token 替换为 `decoder_start_token_id`
    shifted_input_ids[:, 0] = decoder_start_token_id

    # 如果 `pad_token_id` 为 None，则抛出异常
    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")
    # 将标签中可能存在的 -100 值替换为 `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    # 返回向右移动后的 `input_ids` 张量
    return shifted_input_ids


# 定义一个函数，根据输入的 `input_ids` 和 `padding_idx` 创建位置编码
def create_position_ids_from_input_ids(input_ids, padding_idx, past_key_values_length=0):
    """
    Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding symbols
    将非填充符号替换为它们的位置编号。位置编号从 `padding_idx+1` 开始。填充符号不变。
    """
    # 返回一个与 `input_ids` 形状相同的张量，其中非填充符号被替换为它们的位置编号
    pass  # 函数体未完，后续代码未提供，请补充完整。
    # 这一系列的类型转换和类型转换是精心平衡的，旨在同时与ONNX导出和XLA兼容。
    mask = input_ids.ne(padding_idx).int()
    # 创建一个与输入张量维度相同的掩码张量，其中非填充位置为1，填充位置为0
    incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask
    # 计算递增的索引，用于非填充位置的元素，累加非填充位置的数量，并加上过去键值长度，再乘以掩码
    return incremental_indices.long() + padding_idx
    # 将增量索引转换为长整型并加上填充索引，以得到最终的位置索引张量
# 定义一个函数，计算辅助的负载平衡损失，如在Switch Transformer中实现的PyTorch版本。
# 详细信息请参考Switch Transformer论文（https://arxiv.org/abs/2101.03961）。
# 该函数实现了论文中方程（4）到（6）中提出的损失函数，旨在惩罚路由专家之间不平衡的情况。
def load_balancing_loss_func(router_probs: torch.Tensor, expert_indices: torch.Tensor) -> float:
    if router_probs is None:
        return 0

    # 获取专家数量
    num_experts = router_probs.shape[-1]

    # 将专家索引转换为int64类型，否则独热编码将失败
    if expert_indices.dtype != torch.int64:
        expert_indices = expert_indices.to(torch.int64)

    # 如果专家索引的维度是2，则扩展一个维度以适应独热编码的需求
    if len(expert_indices.shape) == 2:
        expert_indices = expert_indices.unsqueeze(2)

    # 创建一个独热编码的专家掩码
    expert_mask = torch.nn.functional.one_hot(expert_indices, num_experts)

    # 对于每个token，确定它是否路由到了某个专家
    expert_mask = torch.max(expert_mask, axis=-2).values

    # 将专家掩码转换为float32类型，否则计算均值会失败
    expert_mask = expert_mask.to(torch.float32)

    # 计算每个专家组和专家的token平均分布
    tokens_per_group_and_expert = torch.mean(expert_mask, axis=-2)

    # 计算每个专家组和专家的路由概率平均值
    router_prob_per_group_and_expert = torch.mean(router_probs, axis=-2)

    # 计算辅助损失，乘以(num_experts的平方)，以增强惩罚效果
    return torch.mean(tokens_per_group_and_expert * router_prob_per_group_and_expert) * (num_experts**2)


# 从transformers.models.m2m_100.modeling_m2m_100.M2M100SinusoidalPositionalEmbedding中复制的类
class NllbMoeSinusoidalPositionalEmbedding(nn.Module):
    """This module produces sinusoidal positional embeddings of any length."""

    def __init__(self, num_positions: int, embedding_dim: int, padding_idx: Optional[int] = None):
        super().__init__()
        self.offset = 2
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.make_weights(num_positions + self.offset, embedding_dim, padding_idx)

    # 创建权重矩阵的函数
    def make_weights(self, num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None):
        # 获取嵌入权重矩阵
        emb_weights = self.get_embedding(num_embeddings, embedding_dim, padding_idx)
        if hasattr(self, "weights"):
            # 在前向传播中，将权重转换为正确的dtype和设备上的参数
            emb_weights = emb_weights.to(dtype=self.weights.dtype, device=self.weights.device)

        # 注册权重缓冲区，这里不是持久性的缓冲区
        self.register_buffer("weights", emb_weights, persistent=False)

    @staticmethod
    def get_embedding(num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None):
        """
        Build sinusoidal embeddings.

        This matches the implementation in tensor2tensor, but differs slightly from the description in Section 3.5 of
        "Attention Is All You Need".
        """
        # 计算嵌入维度的一半
        half_dim = embedding_dim // 2
        # 计算对数值，用于生成正弦位置编码
        emb = math.log(10000) / (half_dim - 1)
        # 计算正弦位置编码
        emb = torch.exp(torch.arange(half_dim, dtype=torch.int64).float() * -emb)
        # 计算位置编码矩阵
        emb = torch.arange(num_embeddings, dtype=torch.int64).float().unsqueeze(1) * emb.unsqueeze(0)
        # 将正弦和余弦编码连接起来，构成最终的位置编码矩阵
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
        # 如果嵌入维度是奇数，进行零填充
        if embedding_dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        # 如果提供了填充索引，将对应位置的编码置为零
        if padding_idx is not None:
            emb[padding_idx, :] = 0

        return emb.to(torch.get_default_dtype())

    @torch.no_grad()
    def forward(
        self, input_ids: torch.Tensor = None, inputs_embeds: torch.Tensor = None, past_key_values_length: int = 0
    ):
        if input_ids is not None:
            bsz, seq_len = input_ids.size()
            # 根据输入的标记 id 创建位置 id，保持任何填充标记的填充状态
            position_ids = create_position_ids_from_input_ids(input_ids, self.padding_idx, past_key_values_length).to(
                input_ids.device
            )
        else:
            bsz, seq_len = inputs_embeds.size()[:-1]
            # 根据输入的嵌入向量创建位置 id
            position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds, past_key_values_length)

        # 如果需要扩展嵌入向量，确保权重矩阵足够大
        max_pos = self.padding_idx + 1 + seq_len + past_key_values_length
        if max_pos > self.weights.size(0):
            self.make_weights(max_pos + self.offset, self.embedding_dim, self.padding_idx)

        return self.weights.index_select(0, position_ids.view(-1)).view(bsz, seq_len, self.weights.shape[-1]).detach()

    def create_position_ids_from_inputs_embeds(self, inputs_embeds, past_key_values_length):
        """
        We are provided embeddings directly. We cannot infer which are padded so just generate sequential position ids.

        Args:
            inputs_embeds: torch.Tensor

        Returns: torch.Tensor
        """
        # 获取输入嵌入的形状信息
        input_shape = inputs_embeds.size()[:-1]
        sequence_length = input_shape[1]

        # 创建顺序位置 id，假设所有输入都是有效的，无法推断填充状态
        position_ids = torch.arange(
            self.padding_idx + 1, sequence_length + self.padding_idx + 1, dtype=torch.long, device=inputs_embeds.device
        )
        # 扩展位置 id 到与输入形状相匹配
        return position_ids.unsqueeze(0).expand(input_shape).contiguous() + past_key_values_length
    class NllbMoeTop2Router(nn.Module):
        """
        Router using tokens choose top-2 experts assignment.

        This router uses the same mechanism as in NLLB-MoE from the fairseq repository. Items are sorted by router_probs
        and then routed to their choice of expert until the expert's expert_capacity is reached. **There is no guarantee
        that each token is processed by an expert**, or that each expert receives at least one token.

        The router combining weights are also returned to make sure that the states that are not updated will be masked.

        """

        def __init__(self, config: NllbMoeConfig):
            super().__init__()
            self.num_experts = config.num_experts
            self.expert_capacity = config.expert_capacity
            self.classifier = nn.Linear(config.hidden_size, self.num_experts, bias=config.router_bias)
            self.router_ignore_padding_tokens = config.router_ignore_padding_tokens
            self.dtype = getattr(torch, config.router_dtype)

            self.second_expert_policy = config.second_expert_policy
            self.normalize_router_prob_before_dropping = config.normalize_router_prob_before_dropping
            self.batch_prioritized_routing = config.batch_prioritized_routing
            self.moe_eval_capacity_token_fraction = config.moe_eval_capacity_token_fraction

        def _cast_classifier(self):
            r"""
            `bitsandbytes` `Linear8bitLt` layers does not support manual casting Therefore we need to check if they are an
            instance of the `Linear8bitLt` class by checking special attributes.
            """
            # 检查是否存在特定属性以判断是否需要类型转换
            if not (hasattr(self.classifier, "SCB") or hasattr(self.classifier, "CB")):
                self.classifier = self.classifier.to(self.dtype)

        def normalize_router_probabilities(self, router_probs, top_1_mask, top_2_mask):
            # 计算每个样本中top-1和top-2的概率
            top_1_max_probs = (router_probs * top_1_mask).sum(dim=1)
            top_2_max_probs = (router_probs * top_2_mask).sum(dim=1)
            # 规范化概率，避免除零错误
            denom_s = torch.clamp(top_1_max_probs + top_2_max_probs, min=torch.finfo(router_probs.dtype).eps)
            top_1_max_probs = top_1_max_probs / denom_s
            top_2_max_probs = top_2_max_probs / denom_s
            return top_1_max_probs, top_2_max_probs

        def route_tokens(
            self,
            router_logits: torch.Tensor,
            input_dtype: torch.dtype = torch.float32,
            padding_mask: Optional[torch.LongTensor] = None,
    # 定义前向传播方法，用于处理隐藏状态和可选的填充掩码，返回三个张量元组
    def forward(self, hidden_states: torch.Tensor, padding_mask: Optional[torch.LongTensor] = None) -> Tuple:
        r"""
        The hidden states are reshaped to simplify the computation of the router probabilities (combining weights for
        each experts.)
        隐藏状态被重新整形，以简化路由概率的计算（结合每个专家的权重）。

        Args:
            hidden_states (`torch.Tensor`):
                (batch_size, sequence_length, hidden_dim) from which router probabilities are computed.
                用于计算路由概率的隐藏状态张量，形状为(batch_size, sequence_length, hidden_dim)。

        Returns:
            top_1_mask (`torch.Tensor` of shape (batch_size, sequence_length)):
                Index tensor of shape [batch_size, sequence_length] corresponding to the expert selected for each token
                using the top1 probabilities of the router.
                形状为(batch_size, sequence_length)的索引张量，每个令牌对应的专家选择索引，使用路由器的top1概率。

            router_probabilities (`torch.Tensor` of shape (batch_size, sequence_length, nump_experts)):
                Tensor of shape (batch_size, sequence_length, num_experts) corresponding to the probabilities for each
                token and expert. Used for routing tokens to experts.
                形状为(batch_size, sequence_length, num_experts)的张量，每个令牌和专家的概率值，用于将令牌路由到专家。

            router_logits (`torch.Tensor` of shape (batch_size, sequence_length))):
                Logits tensor of shape (batch_size, sequence_length, num_experts) corresponding to raw router logits.
                This is used later for computing router z-loss.
                形状为(batch_size, sequence_length, num_experts)的原始路由器logits张量，用于后续计算路由器的z-loss。
        """
        # 将输入的dtype赋值给实例变量input_dtype
        self.input_dtype = hidden_states.dtype
        # 获取输入hidden_states的维度信息
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        # 将hidden_states重新整形为(batch_size * sequence_length, hidden_dim)
        hidden_states = hidden_states.reshape((batch_size * sequence_length), hidden_dim)
        # 将hidden_states转换为self.dtype类型
        hidden_states = hidden_states.to(self.dtype)
        # 调用内部方法_cast_classifier，执行类型转换
        self._cast_classifier()
        # 将转换后的hidden_states输入分类器，得到router_logits
        router_logits = self.classifier(hidden_states)
        # 调用route_tokens方法，使用router_logits、self.input_dtype和padding_mask计算top_1_mask和router_probs
        top_1_mask, router_probs = self.route_tokens(router_logits, self.input_dtype, padding_mask)
        # 返回top_1_mask和router_probs作为前向传播的输出
        return top_1_mask, router_probs
# 定义一个名为 NllbMoeDenseActDense 的神经网络模块类，继承自 nn.Module
class NllbMoeDenseActDense(nn.Module):
    
    # 初始化方法，接收一个 NllbMoeConfig 类型的配置参数和一个整数 ffn_dim
    def __init__(self, config: NllbMoeConfig, ffn_dim: int):
        super().__init__()
        
        # 创建一个线性层 fc1，输入维度为 config.d_model，输出维度为 ffn_dim
        self.fc1 = nn.Linear(config.d_model, ffn_dim)
        
        # 创建一个线性层 fc2，输入维度为 ffn_dim，输出维度为 config.d_model
        self.fc2 = nn.Linear(ffn_dim, config.d_model)
        
        # 创建一个以 config.activation_dropout 为概率的 Dropout 层
        self.dropout = nn.Dropout(config.activation_dropout)
        
        # 根据配置中的激活函数名称从预定义的 ACT2FN 字典中选择激活函数，并赋值给 self.act
        self.act = ACT2FN[config.activation_function]

    # 前向传播方法，接收输入的 hidden_states
    def forward(self, hidden_states):
        
        # 将输入 hidden_states 通过线性层 fc1
        hidden_states = self.fc1(hidden_states)
        
        # 将线性层的输出通过激活函数 self.act
        hidden_states = self.act(hidden_states)
        
        # 对激活后的结果应用 Dropout
        hidden_states = self.dropout(hidden_states)
        
        # 如果 fc2 的权重是 Tensor 类型，并且 hidden_states 的数据类型不等于 fc2 的权重的数据类型
        # 并且 fc2 的权重数据类型不是 torch.int8 和 torch.uint8 类型
        if (
            isinstance(self.fc2.weight, torch.Tensor)
            and hidden_states.dtype != self.fc2.weight.dtype
            and (self.fc2.weight.dtype != torch.int8 and self.fc2.weight.dtype != torch.uint8)
        ):
            # 将 hidden_states 转换成与 fc2 权重相同的数据类型
            hidden_states = hidden_states.to(self.fc2.weight.dtype)
        
        # 将转换后的 hidden_states 通过线性层 fc2
        hidden_states = self.fc2(hidden_states)
        
        # 返回线性层 fc2 的输出结果
        return hidden_states


# 定义一个名为 NllbMoeSparseMLP 的神经网络模块类，继承自 nn.Module
class NllbMoeSparseMLP(nn.Module):
    
    # 初始化方法，接收一个 NllbMoeConfig 类型的配置参数，一个整数 ffn_dim 和一个专家类 expert_class
    def __init__(self, config: NllbMoeConfig, ffn_dim: int, expert_class: nn.Module = NllbMoeDenseActDense):
        super().__init__()
        
        # 创建一个 NllbMoeTop2Router 类的实例，存储在 self.router 中
        self.router = NllbMoeTop2Router(config)
        
        # 设置 self.moe_token_dropout 为 config 中的 moe_token_dropout 参数
        self.moe_token_dropout = config.moe_token_dropout
        
        # 创建一个以 moe_token_dropout 为概率的 Dropout 层，存储在 self.token_dropout 中
        self.token_dropout = nn.Dropout(self.moe_token_dropout)
        
        # 设置 self.num_experts 为 config 中的 num_experts 参数
        self.num_experts = config.num_experts
        
        # 创建一个 ModuleDict 存储专家模型，key 是 "expert_0", "expert_1", ...，值是 expert_class 的实例
        self.experts = nn.ModuleDict()
        for idx in range(self.num_experts):
            self.experts[f"expert_{idx}"] = expert_class(config, ffn_dim)
    def forward(self, hidden_states: torch.Tensor, padding_mask: Optional[torch.Tensor] = False):
        r"""
        The goal of this forward pass is to have the same number of operation as the equivalent `NllbMoeDenseActDense`
        (mlp) layer. This means that all of the hidden states should be processed at most twice ( since we are using a
        top_2 gating mecanism). This means that we keep the complexity to O(batch_size x sequence_length x hidden_dim)
        instead of O(num_experts x batch_size x sequence_length x hidden_dim).

        1- Get the `router_probs` from the `router`. The shape of the `router_mask` is `(batch_size X sequence_length,
        num_expert)` and corresponds to the boolean version of the `router_probs`. The inputs are masked using the
        `router_mask`.
        """
        batch_size, sequence_length, hidden_dim = hidden_states.shape

        # 2- Dispatch the hidden_states to its associated experts. The router probabilities are used to weight the
        # contribution of each experts when updating the masked hidden states.
        
        # Obtain the top 1 mask and router probabilities from the router module
        top_1_mask, router_probs = self.router(hidden_states, padding_mask)

        # Convert router_probs to boolean router_mask
        router_mask = router_probs.bool()

        # Reshape hidden_states for efficient masking
        hidden_states = hidden_states.reshape((batch_size * sequence_length), hidden_dim)

        # Mask the hidden_states using router_mask
        masked_hidden_states = torch.einsum("bm,be->ebm", hidden_states, router_mask)

        # Iterate over each expert and update the masked hidden states accordingly
        for idx, expert in enumerate(self.experts.values()):
            token_indices = router_mask[:, idx]
            combining_weights = router_probs[token_indices, idx]
            expert_output = expert(masked_hidden_states[idx, token_indices])

            # Apply MoE token dropout if configured
            if self.moe_token_dropout > 0:
                if self.training:
                    expert_output = self.token_dropout(expert_output)
                else:
                    expert_output *= 1 - self.moe_token_dropout

            masked_hidden_states[idx, token_indices] = torch.einsum("b,be->be", combining_weights, expert_output)

        # Aggregate the masked hidden states to get updated hidden_states
        hidden_states = masked_hidden_states.sum(dim=0).reshape(batch_size, sequence_length, hidden_dim)

        # Determine the index of the top 1 expert based on top_1_mask
        top_1_expert_index = torch.argmax(top_1_mask, dim=-1)

        # Return updated hidden_states and necessary values for loss computation
        return hidden_states, (router_probs, top_1_expert_index)
# 从transformers.models.bart.modeling_bart.BartAttention复制的NllbMoeAttention类，用于NllbMoe模型的注意力机制实现
class NllbMoeAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
        is_causal: bool = False,
        config: Optional[NllbMoeConfig] = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim  # 设定注意力机制的输入维度
        self.num_heads = num_heads  # 设定注意力头的数量
        self.dropout = dropout  # 设定dropout概率
        self.head_dim = embed_dim // num_heads  # 计算每个头的维度
        self.config = config  # 存储模型配置信息

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5  # 缩放因子，用于注意力分数的缩放
        self.is_decoder = is_decoder  # 是否为解码器的标志
        self.is_causal = is_causal  # 是否使用因果注意力

        # 线性层，用于投影键（k_proj）、值（v_proj）、查询（q_proj）、输出（out_proj）到指定维度
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    # 将张量重塑为适合多头注意力计算的形状
    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    # 前向传播函数，实现注意力机制的计算
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,



# NllbMoeEncoderLayer类，用于NllbMoe模型中的编码器层实现
class NllbMoeEncoderLayer(nn.Module):
    def __init__(self, config: NllbMoeConfig, is_sparse: bool = False):
        super().__init__()
        self.embed_dim = config.d_model  # 获取模型配置中的输入维度
        self.is_sparse = is_sparse  # 是否为稀疏模型的标志
        # 自注意力层，使用NllbMoeAttention实例化，处理自注意力机制
        self.self_attn = NllbMoeAttention(
            embed_dim=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            dropout=config.attention_dropout,
        )
        self.attn_dropout = nn.Dropout(config.dropout)  # 注意力机制的dropout层
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)  # 自注意力层的归一化层
        # 如果不是稀疏模型，则使用NllbMoeDenseActDense处理前馈神经网络
        if not self.is_sparse:
            self.ffn = NllbMoeDenseActDense(config, ffn_dim=config.encoder_ffn_dim)
        else:  # 否则使用NllbMoeSparseMLP处理前馈神经网络
            self.ffn = NllbMoeSparseMLP(config, ffn_dim=config.encoder_ffn_dim)
        self.ff_layer_norm = nn.LayerNorm(config.d_model)  # 前馈网络的归一化层
        self.ff_dropout = nn.Dropout(config.activation_dropout)  # 前馈网络的dropout层

    # 编码器层的前向传播函数
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_head_mask: torch.Tensor,
        output_attentions: bool = False,
        output_router_logits: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.FloatTensor`):
                input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`):
                attention mask of size `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very
                large negative values.
            layer_head_mask (`torch.FloatTensor`): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        # 保存残差连接的输入hidden_states
        residual = hidden_states
        # 对输入hidden_states进行 layer normalization
        hidden_states = self.self_attn_layer_norm(hidden_states)
        # 使用自注意力机制计算新的hidden_states，同时获取注意力权重attn_weights
        hidden_states, attn_weights, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        # 对自注意力机制的输出进行dropout处理
        hidden_states = self.attn_dropout(hidden_states)
        # 将残差连接的输入与经过注意力机制处理后的hidden_states相加
        hidden_states = residual + hidden_states

        # 再次保存残差连接的输入hidden_states
        residual = hidden_states

        # 对更新后的hidden_states进行 layer normalization
        hidden_states = self.ff_layer_norm(hidden_states)
        
        # 如果模型是稀疏的，使用稀疏的前馈网络FFN计算新的hidden_states和router_states
        if self.is_sparse:
            hidden_states, router_states = self.ffn(hidden_states, attention_mask)
        else:
            # 否则，直接使用前馈网络FFN计算新的hidden_states，同时将router_states设为None
            # 用于追踪哪些层的梯度为None
            hidden_states, router_states = self.ffn(hidden_states), None

        # 对前馈网络的输出进行dropout处理
        hidden_states = self.ff_dropout(hidden_states)

        # 将残差连接的输入与经过前馈网络处理后的hidden_states相加
        hidden_states = residual + hidden_states

        # 如果hidden_states的数据类型为torch.float16并且包含无穷大或NaN值，进行截断处理
        if hidden_states.dtype == torch.float16 and (
            torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any()
        ):
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        # 将处理后的hidden_states作为输出
        outputs = (hidden_states,)

        # 如果需要输出注意力权重，则将attn_weights加入输出元组
        if output_attentions:
            outputs += (attn_weights,)

        # 如果需要输出router_states，则将router_states加入输出元组
        if output_router_logits:
            outputs += (router_states,)

        # 返回最终的输出元组
        return outputs
class NllbMoeDecoderLayer(nn.Module):
    def __init__(self, config: NllbMoeConfig, is_sparse: bool = False):
        super().__init__()
        self.embed_dim = config.d_model  # 从配置中获取编码维度
        self.is_sparse = is_sparse  # 标记是否稀疏模式
        # 初始化自注意力层，使用NllbMoeAttention模块
        self.self_attn = NllbMoeAttention(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
        )
        self.dropout = config.dropout  # 配置的dropout值
        self.activation_fn = ACT2FN[config.activation_function]  # 激活函数
        self.attn_dropout = nn.Dropout(config.dropout)  # 注意力机制的dropout

        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)  # 自注意力层后的LayerNorm
        # 初始化跨注意力层，使用NllbMoeAttention模块
        self.cross_attention = NllbMoeAttention(
            self.embed_dim, config.decoder_attention_heads, config.attention_dropout, is_decoder=True
        )
        self.cross_attention_layer_norm = nn.LayerNorm(self.embed_dim)  # 跨注意力层后的LayerNorm
        if not self.is_sparse:
            # 如果不是稀疏模式，使用全连接层的FFN
            self.ffn = NllbMoeDenseActDense(config, ffn_dim=config.decoder_ffn_dim)
        else:
            # 如果是稀疏模式，使用稀疏MLP
            self.ffn = NllbMoeSparseMLP(config, ffn_dim=config.decoder_ffn_dim)
        self.ff_layer_norm = nn.LayerNorm(config.d_model)  # FFN层后的LayerNorm
        self.ff_dropout = nn.Dropout(config.activation_dropout)  # FFN层的dropout

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        cross_attn_layer_head_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        output_router_logits: Optional[bool] = False,
        use_cache: Optional[bool] = True,
    ):
        # 省略了具体的前向传播逻辑，这里应该包含该层的前向传播逻辑
        pass


class NllbMoePreTrainedModel(PreTrainedModel):
    config_class = NllbMoeConfig  # 使用的配置类
    base_model_prefix = "model"  # 基础模型前缀
    supports_gradient_checkpointing = True  # 支持梯度检查点
    _no_split_modules = ["NllbMoeEncoderLayer", "NllbMoeDecoderLayer"]  # 不进行分割的模块列表

    def _init_weights(self, module):
        """Initialize the weights"""
        std = self.config.init_std  # 初始化标准差
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)  # 线性层权重初始化为正态分布
            if module.bias is not None:
                module.bias.data.zero_()  # 如果有偏置，则初始化为零
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)  # Embedding层权重初始化为正态分布
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()  # 如果有padding_idx，则对应位置初始化为零
    # 使用此类作为普通的 PyTorch 模块，并参考 PyTorch 文档以获取有关一般使用和行为的所有信息。

    Parameters:
        config ([`NllbMoeConfig`]):
            这是模型配置类，包含模型的所有参数。使用配置文件初始化类时，并不会加载模型的权重，只会加载配置信息。
            可以查看 [`~PreTrainedModel.from_pretrained`] 方法来加载模型权重。
"""

NLLB_MOE_GENERATION_EXAMPLE = r"""
    Translation example:

    ```
    >>> from transformers import AutoTokenizer, NllbMoeForConditionalGeneration

    >>> model = NllbMoeForConditionalGeneration.from_pretrained("facebook/nllb-moe-54b")
    >>> tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-moe-54b")

    >>> text_to_translate = "Life is like a box of chocolates"
    >>> model_inputs = tokenizer(text_to_translate, return_tensors="pt")

    >>> # translate to French
    >>> gen_tokens = model.generate(**model_inputs, forced_bos_token_id=tokenizer.get_lang_id("eng_Latn"))
    >>> print(tokenizer.batch_decode(gen_tokens, skip_special_tokens=True))
    ```
"""

NLLB_MOE_INPUTS_DOCSTRING = r"""
"""


class NllbMoeEncoder(NllbMoePreTrainedModel):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    [`NllbMoeEncoderLayer`].

    Args:
        config:
            NllbMoeConfig
        embed_tokens (nn.Embedding):
            output embedding
    """

    def __init__(self, config: NllbMoeConfig, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__(config)

        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop

        embed_dim = config.d_model
        self.padding_idx = config.pad_token_id
        self.max_source_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0

        self.embed_tokens = nn.Embedding(config.vocab_size, embed_dim, self.padding_idx)

        if embed_tokens is not None:
            self.embed_tokens.weight = embed_tokens.weight

        self.embed_positions = NllbMoeSinusoidalPositionalEmbedding(
            config.max_position_embeddings,
            embed_dim,
            self.padding_idx,
        )
        sparse_step = config.encoder_sparse_step
        self.layers = nn.ModuleList()
        for i in range(config.encoder_layers):
            is_sparse = (i + 1) % sparse_step == 0 if sparse_step > 0 else False
            self.layers.append(NllbMoeEncoderLayer(config, is_sparse))

        self.layer_norm = nn.LayerNorm(config.d_model)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        """
        Perform forward pass of the NllbMoeEncoder.

        Args:
            input_ids (Optional[torch.Tensor]): Input token IDs.
            attention_mask (Optional[torch.Tensor]): Attention mask.
            head_mask (Optional[torch.Tensor]): Head mask for attention computation.
            inputs_embeds (Optional[torch.Tensor]): Embedded input tokens.
            output_attentions (Optional[bool]): Whether to output attentions.
            output_hidden_states (Optional[bool]): Whether to output hidden states.
            output_router_logits (Optional[bool]): Whether to output router logits.
            return_dict (Optional[bool]): Whether to return a dictionary.

        Returns:
            Depending on `return_dict`, either a tuple or a dictionary with model outputs.
        """
        # TODO: Implement the forward pass logic here
        pass


class NllbMoeDecoder(NllbMoePreTrainedModel):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a [`NllbMoeDecoderLayer`].
    """
    # The implementation of the NllbMoeDecoder class would go here, but it's not provided in the snippet.
    Args:
        config:
            NllbMoeConfig
            模型配置对象，包含模型的各种配置参数
        embed_tokens (nn.Embedding):
            output embedding
            输出嵌入层，用于将输入的标记转换为嵌入表示
    """

    def __init__(self, config: NllbMoeConfig, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__(config)
        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0

        # 初始化输入标记的嵌入层，将词汇表大小、嵌入维度和填充标记应用于嵌入层
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model, self.padding_idx)

        if embed_tokens is not None:
            # 如果提供了额外的嵌入层，使用其权重来初始化当前的嵌入层
            self.embed_tokens.weight = embed_tokens.weight

        # 初始化位置嵌入层，使用正弦函数生成位置编码
        self.embed_positions = NllbMoeSinusoidalPositionalEmbedding(
            config.max_position_embeddings,
            config.d_model,
            self.padding_idx,
        )

        sparse_step = config.decoder_sparse_step
        self.layers = nn.ModuleList()
        for i in range(config.decoder_layers):
            # 判断当前层是否是稀疏注意力层
            is_sparse = (i + 1) % sparse_step == 0 if sparse_step > 0 else False
            # 向层列表中添加一个解码器层对象
            self.layers.append(NllbMoeDecoderLayer(config, is_sparse))

        # 初始化层归一化层，对模型输出进行归一化处理
        self.layer_norm = nn.LayerNorm(config.d_model)

        # 初始化梯度检查点，默认为 False
        self.gradient_checkpointing = False
        # 初始化权重并应用最终处理
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        return_dict: Optional[bool] = None,
@add_start_docstrings(
    "The bare NllbMoe Model outputting raw hidden-states without any specific head on top.",
    NLLB_MOE_START_DOCSTRING,
)
# 定义 NllbMoeModel 类，用于输出不带特定头部的原始隐藏状态
class NllbMoeModel(NllbMoePreTrainedModel):
    # 定义共享权重的键列表，这些权重会被绑定
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight"]

    def __init__(self, config: NllbMoeConfig):
        super().__init__(config)

        # 初始化填充索引和词汇表大小
        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        # 创建共享的嵌入层，用于输入词汇表到指定的 d_model 维度
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        # 初始化编码器和解码器
        self.encoder = NllbMoeEncoder(config, self.shared)
        self.decoder = NllbMoeDecoder(config, self.shared)

        # 初始化权重并应用最终处理
        self.post_init()

    # 返回共享的嵌入层
    def get_input_embeddings(self):
        return self.shared

    # 设置输入的嵌入层
    def set_input_embeddings(self, value):
        self.shared = value
        self.encoder.embed_tokens = self.shared
        self.decoder.embed_tokens = self.shared

    # 绑定权重，如果配置要求绑定词嵌入
    def _tie_weights(self):
        if self.config.tie_word_embeddings:
            self._tie_or_clone_weights(self.encoder.embed_tokens, self.shared)
            self._tie_or_clone_weights(self.decoder.embed_tokens, self.shared)

    # 获取编码器
    def get_encoder(self):
        return self.encoder

    # 获取解码器
    def get_decoder(self):
        return self.decoder

    @add_start_docstrings_to_model_forward(NLLB_MOE_INPUTS_DOCSTRING)
    @add_start_docstrings_to_model_forward(NLLB_MOE_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqMoEModelOutput, config_class=_CONFIG_FOR_DOC)
    # 定义前向方法，用于模型的正向传播，支持多种输入参数和返回类型
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        return_dict: Optional[bool] = None,



@add_start_docstrings(
    "The NllbMoe Model with a language modeling head. Can be used for summarization.", NLLB_MOE_START_DOCSTRING
)
# 定义 NllbMoeForConditionalGeneration 类，具有语言建模头部，可用于摘要生成
class NllbMoeForConditionalGeneration(NllbMoePreTrainedModel):
    # 基础模型前缀
    base_model_prefix = "model"
    # 定义共享权重的键列表，这些权重会被绑定
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight", "lm_head.weight"]
    # 使用给定的配置初始化模型
    def __init__(self, config: NllbMoeConfig):
        # 调用父类的初始化方法
        super().__init__(config)
        # 使用给定配置创建 NllbMoeModel 模型实例
        self.model = NllbMoeModel(config)
        # 创建一个线性层，用于生成输出的词汇表大小的结果，没有偏置
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # 设置路由器 z 损失系数和辅助损失系数
        self.router_z_loss_coef = config.router_z_loss_coef
        self.router_aux_loss_coef = config.router_aux_loss_coef
        # 执行初始化权重和应用最终处理
        self.post_init()

    # 获取编码器部分
    def get_encoder(self):
        return self.model.get_encoder()

    # 获取解码器部分
    def get_decoder(self):
        return self.model.get_decoder()

    # 获取输出嵌入层
    def get_output_embeddings(self):
        return self.lm_head

    # 设置输出嵌入层
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    # 重写的前向方法，用于模型前向推断
    @add_start_docstrings_to_model_forward(NLLB_MOE_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqMoEOutput, config_class=_CONFIG_FOR_DOC)
    @add_end_docstrings(NLLB_MOE_GENERATION_EXAMPLE)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        # 此方法是用来执行模型的前向推断

    # 解压路由器输出的方法，从路由器输出中提取总的路由器对数和专家索引
    def _unpack_router_logits(self, router_outputs):
        # 初始化总路由器对数和总专家索引列表
        total_router_logits = []
        total_expert_indexes = []
        # 遍历路由器输出列表
        for router_output in router_outputs:
            # 如果路由器输出不为空
            if router_output is not None:
                # 分别取出路由器对数和专家索引
                router_logits, expert_indexes = router_output
                # 将路由器对数和专家索引添加到总列表中
                total_router_logits.append(router_logits)
                total_expert_indexes.append(expert_indexes)

        # 如果总路由器对数列表不为空，则沿着维度1拼接成一个张量，否则为 None
        total_router_logits = torch.cat(total_router_logits, dim=1) if len(total_router_logits) > 0 else None
        # 如果总专家索引列表不为空，则沿着维度1堆叠成一个张量，否则为 None
        total_expert_indexes = torch.stack(total_expert_indexes, dim=1) if len(total_expert_indexes) > 0 else None
        # 返回总路由器对数和总专家索引
        return total_router_logits, total_expert_indexes

    # 从 transformers.models.switch_transformers.SwitchTransformersForConditionalGeneration.prepare_inputs_for_generation 复制而来
    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past_key_values=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs,
    ):
        # 如果使用了过去的键值对，则需要调整decoder_input_ids的长度
        if past_key_values is not None:
            # 获取过去键值对的长度
            past_length = past_key_values[0][0].shape[2]

            # 一些生成方法已经仅传递最后一个输入ID
            if decoder_input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # 默认情况下保留仅最后一个ID的旧行为
                remove_prefix_length = decoder_input_ids.shape[1] - 1

            # 裁剪decoder_input_ids以去除过去的前缀
            decoder_input_ids = decoder_input_ids[:, remove_prefix_length:]

        # 返回一个包含各种生成过程输入的字典
        return {
            "input_ids": None,  # encoder_outputs已定义。不需要input_ids
            "encoder_outputs": encoder_outputs,
            "past_key_values": past_key_values,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,  # 更改此处以避免缓存（可能是为了调试）
        }

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        # 重新排序过去的键值对，以匹配新的beam索引
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                # 对每个层的过去状态进行索引选择，以匹配新的beam索引并添加到重排序过去中
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past
```