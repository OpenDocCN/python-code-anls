# `.\models\gptsan_japanese\modeling_gptsan_japanese.py`

```
# 将注释后的代码包含在一个代码块中
# coding=utf-8

# 引入所需模块和类
# 来自 typing 模块的类型提示类
# torch 库中的各种类和函数
# 引入自定义的模块和函数
# 打印日志信息的函数
# 定义了一个字典对象
# 此字典包含了预训练模型的权重的 id 和 url

# 定义了一个函数，用于计算 router z 损失
# 此函数用于计算 router logits 的其他损失

# 定义了一个函数，用于计算平衡负载损失
# 此函数用于计算 router 概率和专家索引的其他损失
    # 定义一个函数，实现论文中方程（4）-（6）中呈现的功能。其目的是惩罚路由专家之间不平衡的情况。
    # 参数：
    #   router_probs（`torch.Tensor`）：每个令牌分配给每个专家的概率。形状：[batch_size, seqeunce_length, num_experts]。
    #   expert_indices（`torch.Tensor`）：形状为[batch_size, seqeunce_length]的索引张量，用于标识给定令牌的选定专家。
    # 返回值：
    #   辅助损失。

    # 获取专家的数量
    num_experts = router_probs.shape[-1]

    # 将专家索引转换为int64类型，否则独热编码将失败
    if expert_indices.dtype != torch.int64:
        expert_indices = expert_indices.to(torch.int64)

    # 如果专家索引的维度为2，则在最后添加一个维度
    if len(expert_indices.shape) == 2:
        expert_indices = expert_indices.unsqueeze(2)

    # 使用独热编码获取专家掩码
    expert_mask = torch.nn.functional.one_hot(expert_indices, num_experts)

    # 对于给定的令牌，确定它是否被路由到给定的专家
    expert_mask = torch.max(expert_mask, axis=-2).values

    # 将专家掩码转换为float32类型，否则平均值计算将失败
    expert_mask = expert_mask.to(torch.float32)

    # 计算每组和专家的令牌数量的平均值
    tokens_per_group_and_expert = torch.mean(expert_mask, axis=-2)

    # 计算每组和专家的路由概率的平均值
    router_prob_per_group_and_expert = torch.mean(router_probs, axis=-2)

    # 返回每组和专家的令牌数量和路由概率的平均值乘积乘以专家数量的平方的平均值
    return torch.mean(tokens_per_group_and_expert * router_prob_per_group_and_expert) * (num_experts**2)
class GPTSanJapaneseDenseActDense(nn.Module):
    """
    FFN Layer for Switch Transformer and Extra layers

    GPTSAN can mix Switch Transformer layers and normal Transformer layers This class is used as Expert in Switch
    Transformer layers and as FFN in regular Transformer layers. RELU is used in the Switch Transformer layer, and
    Swish is used in the normal Transformer layer, so there is a choice of which is used in the argument.

    """

    def __init__(self, config: GPTSanJapaneseConfig, ext_layer=False):
        # 初始化函数，定义了一个FFN（Feed-Forward Neural Network）层，用于Switch Transformer和Extra layers
        super().__init__()
        d_inter = config.d_ext if ext_layer else config.d_ff
        self.wi = nn.Linear(config.d_model, d_inter, bias=ext_layer)  # 使用全连接层，从输入维度到d_inter的输出维度
        self.wo = nn.Linear(d_inter, config.d_model, bias=ext_layer)  # 使用全连接层，从d_inter的输入维度到输出维度
        self.dropout = nn.Identity() if ext_layer else nn.Dropout(config.dropout_rate)  # 根据是否为ext_layer选择是否使用dropout
        self.act = ACT2FN["swish" if ext_layer else "relu"]  # 根据是否为ext_layer选择使用swish或relu作为激活函数

    def forward(self, hidden_states):
        r"""
        Args:
            hidden_states (`torch.Tensor`) :
                [num_groups, tokens_per_group, hidden_dim] inputs to send to experts.
        Returns:
            torch.Tensor[num_groups, tokens_per_group, hidden_dim]

        """
        hidden_states = self.wi(hidden_states)  # 输入经过全连接层wi
        hidden_states = self.act(hidden_states)  # 经过激活函数
        hidden_states = self.dropout(hidden_states)  # 使用dropout
        hidden_states = self.wo(hidden_states)  # 经过全连接层wo
        return hidden_states  # 返回处理后的结果


# Copied from transformers.models.switch_transformers.modeling_switch_transformers.SwitchTransformersTop1Router with SwitchTransformers->GPTSanJapanese
class GPTSanJapaneseTop1Router(nn.Module):
    """
    Router using tokens choose top-1 experts assignment.

    This router uses the same mechanism as in Switch Transformer (https://arxiv.org/abs/2101.03961) and V-MoE
    (https://arxiv.org/abs/2106.05974): tokens choose their top experts. Items are sorted by router_probs and then
    routed to their choice of expert until the expert's expert_capacity is reached. **There is no guarantee that each
    token is processed by an expert**, or that each expert receives at least one token.

    """

    def __init__(self, config: GPTSanJapaneseConfig):
        # 初始化函数，定义了一个使用tokens选择top-1 experts分配的Router
        super().__init__()
        self.num_experts = config.num_experts  # 专家的数量
        self.expert_capacity = config.expert_capacity  # 专家的容量
        self.classifier = nn.Linear(config.hidden_size, self.num_experts, bias=config.router_bias)  # 使用线性分类器，将hidden_size映射到num_experts
        self.jitter_noise = config.router_jitter_noise  # 路由器的抖动噪声
        self.ignore_padding_tokens = config.router_ignore_padding_tokens  # 是否忽略填充的tokens
        self.dtype = getattr(torch, config.router_dtype)  # 路由器的数据类型
    # 计算从输入隐藏状态到路由器概率的方法
    def _compute_router_probabilities(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # 根据输入的隐藏状态计算路由器概率

        # 用来确保稳定性的 float32 类型。参见 https://arxiv.org/abs/2101.03961 中关于 "selective precision" 的讨论。
        # 同时也存储之前的数据类型，以便在计算结束后将输出转换回之前的数据类型
        self.input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(self.dtype)

        if self.training and self.jitter_noise > 0:
            # 通过将输入的 token 乘以均匀分布来增加一些噪音
            hidden_states *= torch.empty_like(hidden_states).uniform_(1.0 - self.jitter_noise, 1.0 + self.jitter_noise)

        # Shape: [num_groups, tokens_per_group, num_experts]
        # 调用 _cast_classifier 方法
        self._cast_classifier()
        # 调用分类器得到路由器 logit
        router_logits = self.classifier(hidden_states)

        # 应用 Softmax，并转换回原始的 `dtype`
        # 使用 nn.functional.softmax 计算路由概率，并指定在最后一个维度进行 softmax 操作，然后转换为之前的输入数据类型
        router_probabilities = nn.functional.softmax(router_logits, dim=-1, dtype=self.dtype).to(self.input_dtype)
        # 返回路由概率和路由 logit
        return router_probabilities, router_logits

    # 转换分类器
    def _cast_classifier(self):
        # `bitsandbytes` `Linear8bitLt` 层不支持手动转换，因此需要检查它们是否是 `Linear8bitLt` 类的实例，可以通过检查特殊属性来进行判断
        if not (hasattr(self.classifier, "SCB") or hasattr(self.classifier, "CB")):
            # 如果不是 `Linear8bitLt` 类的实例，则将分类器转换为指定的数据类型
            self.classifier = self.classifier.to(self.dtype)
    # 定义一个 forward 方法，用于每个 Router 类的通用前向传播函数
    # Router 预期具有相同的输入隐藏状态（`hidden_states`），对应于每个令牌的隐藏状态，`expert_capacity` 对应于 Router 将发送到每个专家的令牌数，某些 Router 可以向每个专家发送多个令牌。
    def forward(self, hidden_states: torch.Tensor) -> Tuple:
        r"""
        通用的 Router 类的前向传播函数。每个 Router 期望具有相同的输入隐藏状态（`hidden_states`），对应于每个令牌的隐藏状态，`expert_capacity` 对应于 Router 将发送到每个专家的令牌数，某些 Router 可以向每个专家发送多个令牌。
    
        每个 Router 的工作方式如下：它期望每个令牌的隐藏状态，从`router_weights` 获取 `router_probs` 和 `router_logits`。这将为每个令牌分配原始概率以分配给专家。然后每个 Router 类将不得不定义自己的`_compute_routing_instructions`。
    
        Args:
            hidden_states (`torch.Tensor`) :
                [num_groups, tokens_per_group, hidden_dim] 要发送到专家的输入。
        Returns:
            Tuple[`torch.Tensor`, `torch.Tensor`, `torch.Tensor`] 包含专家索引、路由器概率和路由器 logits 的元组。路由器概率和 logits 是计算损失所需的。
        """
        # 从隐藏状态计算路由器概率和路由器 logits
        router_probs, router_logits = self._compute_router_probabilities(hidden_states)
    
        # 根据路由器概率找到最大值的索引，得到专家索引
        expert_index = torch.argmax(router_probs, dim=-1)
        expert_index = torch.nn.functional.one_hot(expert_index, num_classes=self.num_experts)
    
        # 令牌超出专家容量的掩码。沿每个序列求和
        token_priority = torch.cumsum(expert_index, dim=-2)
        # 如果路由到专家的令牌会溢出，则进行掩码处理
        expert_capacity_mask = token_priority <= self.expert_capacity
        expert_index = expert_index * expert_capacity_mask
    
        # 获取路由器概率的最大值并增加维度
        router_probs = torch.max(router_probs, dim=-1).values.unsqueeze(-1)
        # 返回专家索引、路由器概率和路由器 logits
        return expert_index, router_probs, router_logits
# 从 transformers.models.switch_transformers.modeling_switch_transformers.SwitchTransformersSparseMLP 复制到 GPTSanJapaneseSparseMLP
class GPTSanJapaneseSparseMLP(nn.Module):
    r"""
    Switch Transformers稀疏MLP模块的实现。
    """

    def __init__(self, config: GPTSanJapaneseConfig, expert_class: nn.Module = GPTSanJapaneseDenseActDense):
        # 调用父类构造函数
        super().__init__()
        # 步骤1：根据类获取正确的路由器
        self.router = GPTSanJapaneseTop1Router(config)

        # 步骤2：获取专家
        self.experts = nn.ModuleDict()
        for idx in range(config.num_experts):
            # 为每个专家创建实例并存储在模块字典中
            self.experts[f"expert_{idx}"] = expert_class(config)

    def forward(self, hidden_states):
        r"""
        Hold on, this will be slightly tricky to understand In the correct order, a MoE layer does the following:

        1- Gets the `router_mask` from the router. The shape of the mask is `(batch_size, sequence_length, num_expert)`
        and corresponds to the argmax of the `router_probs`. The probabilities are needed in the computation of the
        hidden states : they are broadcasted to the hidden states values (can be interpreted as a scaling factor).

        2- Dispatch the tokens to its associated experts. We do a classic for loop over the experts and assign for each
        expert the corresponding hidden states.

        """
        # 步骤1：从路由器获取路由器掩码，以及概率
        router_mask, router_probs, router_logits = self.router(hidden_states)
        # 获取专家索引
        expert_index = torch.argmax(router_mask, dim=-1)

        # introducd的路由器可能不总是将所有token映射到路由器，这意味着某些隐藏状态可能会从一个层保持不变到另一个层，因此在更新仅选定的状态之前，需要克隆隐藏状态。

        next_states = hidden_states.clone()
        # 遍历专家并为每个专家分配对应的隐藏状态
        for idx, expert in enumerate(self.experts.values()):
            token_indices = router_mask[:, :, idx].bool()
            next_states[token_indices] = expert(hidden_states[token_indices]).to(next_states.dtype)

        hidden_states = router_probs * next_states
        return hidden_states, (router_logits, expert_index)


class GPTSanJapaneseLayerSparseFF(nn.Module):
    r"""
    Switch Transformers前馈层模块。这是对Mixture of Experts模块的包装。

    Parameters:
        config : ([`GPTSanJapaneseConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
    """
    # 初始化方法，接受一个 GPTSanJapaneseConfig 类型的配置参数
    def __init__(self, config: GPTSanJapaneseConfig):
        # 调用父类的初始化方法
        super().__init__()
        # 初始化一个 GPTSanJapaneseSparseMLP 实例，用于处理稀疏连接
        self.mlp = GPTSanJapaneseSparseMLP(config)
        # 初始化一个线性层，用于软绕过连接
        self.soft_bypass_mlp = nn.Linear(config.d_model, config.d_model, bias=False)
        # 初始化一个 LayerNorm 层，用于归一化隐藏状态
        self.norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_epsilon)

    # 前向传播方法
    def forward(self, hidden_states, output_router_logits):
        r"""
        Args:
            hidden_states (`torch.Tensor`) :
                [num_groups, tokens_per_group, hidden_dim] inputs to send to experts.
            output_router_logits (`bool`) :
                output experts router output.
        Returns:
            torch.Tensor[num_groups, tokens_per_group, hidden_dim]

        """
        # 使用 MLP 处理隐藏状态，并返回处理后的状态和路由器的输出
        forwarded_states, router_tuple = self.mlp(hidden_states)
        # 将隐藏状态经过一个 tanh 激活函数并加上软绕过连接的输出
        forwarded_states += torch.tanh(self.soft_bypass_mlp(hidden_states))
        # 将原始隐藏状态和处理后的状态经过 LayerNorm 层并相加，得到最终的输出
        output = hidden_states + self.norm(forwarded_states)

        # 如果需要输出路由器的输出且路由器元组不为空，则返回输出和路由器元组
        if output_router_logits and router_tuple is not None:
            return output, router_tuple
        else:
            return output
class GPTSanJapaneseLayerDenseFF(nn.Module):
    r"""
    Extra Transformers Feed Forward layer module.

    Parameters:
        config : ([`GPTSanJapaneseConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
    """

    # 初始化函数，接收一个 GPTSanJapaneseConfig 类型的参数
    def __init__(self, config: GPTSanJapaneseConfig):
        # 调用父类的初始化函数
        super().__init__()
        # 判断是否为稀疏层，如果不是则为密集层，创建一个 GPTSanJapaneseDenseActDense 对象
        self.mlp = GPTSanJapaneseDenseActDense(config, ext_layer=True)
        # 创建 nn.LayerNorm 层，处理 Transformer 模型输出的 hidden states
        self.norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_epsilon)

    # 前向传播函数，将 hidden_states 传入 MLP 层并返回输出
    def forward(self, hidden_states):
        forwarded_states = self.mlp(hidden_states)
        # 将输出结果加上 LayerNorm 处理后返回
        output = hidden_states + self.norm(forwarded_states)
        return output


# 从 transformers.models.bart.modeling_bart.BartAttention 复制到 GPTSanJapaneseAttention
class GPTSanJapaneseAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    # 初始化函数，接收多个参数用于构建多头自注意力层
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
        is_causal: bool = False,
        config: Optional[GPTSanJapaneseConfig] = None,
    ):
        # 调用父类的初始化函数
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.config = config

        # 检查embed_dim是否能被num_heads整除
        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        
        # 计算缩放因子
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder
        self.is_causal = is_causal

        # 创建线性映射层用于 Q，K，V 和输出
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    # 格式化输入张量的形状，使其适用于多头自注意力层
    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
``` 
    # 前向传播函数，用于Transformer模型的前向计算
    def forward(
        self,
        # 隐藏状态张量，即输入的特征表示
        hidden_states: torch.Tensor,
        # 键值状态张量，可选参数，用于自注意力机制的注意力计算
        key_value_states: Optional[torch.Tensor] = None,
        # 过去的键值状态元组，可选参数，用于获得当前步长的自注意力计算结果
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        # 注意力遮罩张量，可选参数，用于指示哪些位置需要被忽略
        attention_mask: Optional[torch.Tensor] = None,
        # 层级头部遮罩张量，可选参数，用于指定哪些头部被屏蔽
        layer_head_mask: Optional[torch.Tensor] = None,
        # 是否输出注意力权重，可选参数，默认为False
        output_attentions: bool = False,
# 定义 GPTSanJapaneseLayerSelfAttention 类，表示自注意力和归一化单元
class GPTSanJapaneseLayerSelfAttention(nn.Module):
    """
    Self Attention and Normalization Unit
    """

    # 初始化方法
    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__()
        # 创建自注意力层对象
        self.self_attn = GPTSanJapaneseAttention(
            embed_dim=config.d_model,
            num_heads=config.num_heads,
            is_decoder=True,
            bias=has_relative_attention_bias,
        )
        # 创建层归一化对象
        self.norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_epsilon)

    # 前向传播方法
    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
class GPTSanJapaneseBlock(nn.Module):
    """
    Self Attention and FFN Unit
    """

    # 初始化方法
    def __init__(self, config, ext_layer=False):
        super().__init__()
        # 创建自注意力层对象
        self.self_attn = GPTSanJapaneseLayerSelfAttention(config)
        # 如果 ext_layer 参数为 False，则创建稠密前馈神经网络层对象，否则创建稀疏前馈神经网络层对象
        self.feed_forward = GPTSanJapaneseLayerDenseFF(config) if ext_layer else GPTSanJapaneseLayerSparseFF(config)

    # 前向传播方法
    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        output_router_tuple: Optional[bool] = False,
class GPTSanJapanesePreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 定义属性 config_class
    config_class = GPTSanJapaneseConfig
    # 定义基本模型前缀
    base_model_prefix = "gptsan_japanese"
    # 是否支持梯度检查点
    supports_gradient_checkpointing = False
    # 不分割的模块列表
    _no_split_modules = ["GPTSanJapaneseBlock"]
    # 跳过设备放置的键
    _skip_keys_device_placement = "past_key_values"

    # 定义属性 dummy_inputs
    @property
    def dummy_inputs(self):
        input_ids = torch.tensor(DUMMY_INPUTS)
        input_mask = torch.tensor(DUMMY_MASK)
        dummy_inputs = {
            "input_ids": input_ids,
            "attention_mask": input_mask,
        }
        return dummy_inputs

    # 从 transformers.models.t5.modeling_t5.T5PreTrainedModel._shift_right 复制而来的方法
    # 定义向右移位函数，用于修改输入的 token 序列，首位是 decoder 的起始 token
    def _shift_right(self, input_ids):
        # 获取解码器的起始 token id 和填充 token id
        decoder_start_token_id = self.config.decoder_start_token_id
        pad_token_id = self.config.pad_token_id

        # 如果解码器的起始 token id 未定义，则抛出数值错误异常
        if decoder_start_token_id is None:
            raise ValueError(
                "self.model.config.decoder_start_token_id has to be defined. In T5 it is usually set to the pad_token_id. "
                "See T5 docs for more information."
            )

        # 将输入向右移动
        if is_torch_fx_proxy(input_ids):
            # 对于代理对象，不支持原生的 item 赋值操作
            shifted_input_ids = torch.full(input_ids.shape[:-1] + (1,), decoder_start_token_id)
            shifted_input_ids = torch.cat([shifted_input_ids, input_ids[..., :-1]], dim=-1)
        else:
            shifted_input_ids = input_ids.new_zeros(input_ids.shape)
            shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
            shifted_input_ids[..., 0] = decoder_start_token_id

        # 如果填充 token id 未定义，则抛出数值错误异常
        if pad_token_id is None:
            raise ValueError("self.model.config.pad_token_id has to be defined.")
        # 将标签中可能存在的 -100 值替换为填充 token id
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

        return shifted_input_ids
# 设置 GPTSAN_JAPANESE_START_DOCSTRING 变量，包含 GPTSAN-japanese 模型的描述信息和参数说明
GPTSAN_JAPANESE_START_DOCSTRING = r"""

    The [GPTSAN-japanese](https://github.com/tanreinama/GPTSAN) model was proposed in General-purpose Swich transformer
    based Japanese language model

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`GPTSanJapaneseConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

# 设置 GPTSAN_JAPANESE_INPUTS_DOCSTRING 变量为空字符串
GPTSAN_JAPANESE_INPUTS_DOCSTRING = r"""
"""

# 装饰 GPTSanJapaneseModel 类，添加文档字符串和描述信息
@add_start_docstrings(
    "The bare GPTSAN-japanese Model transformer outputting raw hidden-states without any specific head on top.",
    GPTSAN_JAPANESE_START_DOCSTRING,
)
# 定义 GPTSanJapaneseModel 类
class GPTSanJapaneseModel(GPTSanJapanesePreTrainedModel):
    # 初始化方法
    def __init__(self, config: GPTSanJapaneseConfig):
        super().__init__(config)
        # 初始化位置嵌入层
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.d_model)
        self.config = copy.deepcopy(config)
        # 初始化词嵌入层
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)
        # 初始化最后的投影层
        self.last_project = nn.Linear(config.d_model, config.d_model, bias=True)
        # 设置激活函数
        self.act = ACT2FN["swish"]

        self.blocks = torch.nn.ModuleList([])
        # 循环添加 GPTSanJapaneseBlock 模块到 blocks
        for _ in range(config.num_switch_layers):
            self.blocks.append(GPTSanJapaneseBlock(config))
        for _ in range(config.num_ext_layers):
            self.blocks.append(GPTSanJapaneseBlock(config, ext_layer=True))

        # 如果存在额外的扩展层，则初始化额外的位置嵌入层
        if config.num_ext_layers > 0:
            self.extra_position_embeddings = nn.Embedding(config.max_position_embeddings, config.d_model)

        # 如果存在 d_spout，则初始化 spout 层
        if config.d_spout:
            spouts = []
            for _ in range(8):
                spouts.append(nn.Linear(config.d_spout, config.d_spout, bias=False))
                spouts.append(nn.Tanh())
            spouts.append(nn.Linear(config.d_spout, config.num_layers * 2 * config.d_model, bias=False))
            self.spout = nn.Sequential(*spouts)

        # 执行后初始化方法
        self.post_init()

    # 获取输入的嵌入层
    def get_input_embeddings(self):
        return self.embed_tokens

    # 设置输入的嵌入层
    def set_input_embeddings(self, new_embeddings):
        self.embed_tokens = new_embeddings

    # 添加输入的文档字符串到模型前向方法
    @add_start_docstrings_to_model_forward(GPTSAN_JAPANESE_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,  # 输入的标记化的文本序列的张量表示
        attention_mask: Optional[torch.FloatTensor] = None,  # 控制每个标记是否被注意力机制考虑的张量表示
        token_type_ids: Optional[torch.FloatTensor] = None,  # 标记的类型的张量表示
        spout: Optional[torch.FloatTensor] = None,  # 生成的口号的张量表示
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,  # 循环神经网络的隐藏状态的张量表示
        head_mask: Optional[torch.FloatTensor] = None,  # 控制每个注意力头是否被屏蔽的张量表示
        use_cache: Optional[bool] = False,  # 是否使用缓存的布尔值
        inputs_embeds: Optional[torch.FloatTensor] = None,  # 输入的嵌入矩阵的张量表示
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,  # 解码器的输入嵌入矩阵的张量表示
        output_attentions: Optional[bool] = None,  # 是否输出注意力权重的布尔值
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态的布尔值
        return_dict: Optional[bool] = None,  # 是否返回字典格式的布尔值
        output_router_logits: Optional[bool] = None,  # 是否输出路由器的逻辑张量的布尔值
        num_precontext: Optional[torch.LongTensor] = None,  # 先前上下文的数量的张量表示
# 使用装饰器添加文档字符串，描述 GPTSan-japanese 模型及其语言建模头
@add_start_docstrings(
    "The bare GPTSan-japanese Model with a language modeling head.",
    GPTSAN_JAPANESE_START_DOCSTRING,
)
# 定义 GPTSanJapaneseForConditionalGeneration 类，继承自 GPTSanJapanesePreTrainedModel 类
class GPTSanJapaneseForConditionalGeneration(GPTSanJapanesePreTrainedModel):
    # 定义 tied_weights_keys 属性
    _tied_weights_keys = ["lm_head.weight"]

    # 初始化方法，接收一个 GPTSanJapaneseConfig 类型的 config 参数
    def __init__(self, config: GPTSanJapaneseConfig):
        # 调用父类的初始化方法
        super().__init__(config)
        # 创建 GPTSanJapaneseModel 对象
        self.model = GPTSanJapaneseModel(config)
        # 注册一个名为 final_logits_bias 的缓冲层，用零填充，形状为 [1, config.vocab_size]
        self.register_buffer("final_logits_bias", torch.zeros([1, config.vocab_size]))
        # 创建一个线性层 lm_head，输入维度为 config.d_model，输出维度为 config.vocab_size，不带偏置
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        # 如果 self.config.torchscript 为 False，将 lm_head 的权重设置为 model.embed_tokens 的权重
        if not self.config.torchscript:
            self.lm_head.weight = self.model.embed_tokens.weight

    # 使用装饰器添加文档字符串到模型的前向传播方法
    @add_start_docstrings_to_model_forward(GPTSAN_JAPANESE_INPUTS_DOCSTRING)
    # 定义前向传播方法，接收多个输入参数
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.FloatTensor] = None,
        spout: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None,
    # 定义一个方法，为生成准备输入
    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.FloatTensor,
        token_type_ids: Optional[torch.FloatTensor] = None,
        spout: Optional[Union[List, torch.FloatTensor]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        **kwargs,
    ):
        # 如果 spout 是列表，则将其转换为 torch.Tensor 类型并转为浮点类型
        if isinstance(spout, list):
            spout = torch.tensor(spout).float()
            # 如果 input_ids 不为空，将 spout 移动到与 input_ids 相同的设备上
            if input_ids is not None:
                spout = spout.to(input_ids.device)
        # 如果 past_key_values 不为空，则返回一些特定的输入
        if past_key_values is not None:
            return {
                "input_ids": input_ids[:, -1:] if input_ids is not None else None,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids[:, -1:] if token_type_ids is not None else None,
                "spout": spout,
                "past_key_values": past_key_values,
            }
        # 否则返回一般的输入
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "spout": spout,
            "past_key_values": None,
        }

    # 从 transformers.models.switch_transformers.modeling_switch_transformers.SwitchTransformersForConditionalGeneration.prepare_decoder_input_ids_from_labels 复制过来的方法，修改为 GPTSanJapaneseForConditionalGeneration
    # 从标签中准备解码器的输入ID，通过向右移动
    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return self._shift_right(labels)

    # 从transformers.models.mbart.modeling_mbart.MBartForConditionalGeneration.resize_token_embeddings复制而来，用于调整token的嵌入
    def resize_token_embeddings(self, new_num_tokens: int, pad_to_multiple_of: Optional[int] = None) -> nn.Embedding:
        new_embeddings = super().resize_token_embeddings(new_num_tokens, pad_to_multiple_of)
        self._resize_final_logits_bias(new_embeddings.weight.shape[0])
        return new_embeddings

    # 从transformers.models.mbart.modeling_mbart.MBartForConditionalGeneration._resize_final_logits_bias复制而来，用于调整最终logits的偏差
    def _resize_final_logits_bias(self, new_num_tokens: int) -> None:
        old_num_tokens = self.final_logits_bias.shape[-1]
        if new_num_tokens <= old_num_tokens:
            new_bias = self.final_logits_bias[:, :new_num_tokens]
        else:
            extra_bias = torch.zeros((1, new_num_tokens - old_num_tokens), device=self.final_logits_bias.device)
            new_bias = torch.cat([self.final_logits_bias, extra_bias], dim=1)
        self.register_buffer("final_logits_bias", new_bias)

    # 获取输入嵌入
    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    # 设置输入嵌入
    def set_input_embeddings(self, new_embeddings):
        self.model.set_input_embeddings(new_embeddings)

    # 从transformers.models.switch_transformers.modeling_switch_transformers.SwitchTransformersForConditionalGeneration.set_output_embeddings复制而来，用于设置输出嵌入
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    # 从transformers.models.switch_transformers.modeling_switch_transformers.SwitchTransformersForConditionalGeneration.get_output_embeddings复制而来，用于获取输出嵌入
    def get_output_embeddings(self):
        return self.lm_head

    # 从transformers.models.switch_transformers.modeling_switch_transformers.SwitchTransformersForConditionalGeneration._unpack_router_logits复制而来，用于解包路由器logits
    def _unpack_router_logits(self, router_outputs):
        total_router_logits = []
        total_expert_indexes = []
        for router_output in router_outputs:
            if len(router_output[0].shape) > 1:
                router_logits, expert_indexes = router_output
                total_router_logits.append(router_logits)
                total_expert_indexes.append(expert_indexes)
        return torch.cat(total_router_logits, dim=1), torch.cat(total_expert_indexes, dim=1)
```