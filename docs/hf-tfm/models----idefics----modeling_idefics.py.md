# `.\models\idefics\modeling_idefics.py`

```
# 设置文件编码为 UTF-8
# 版权声明，基于 EleutherAI 的 GPT-NeoX 库和 GPT-NeoX 中的 GPT-NeoX 和 OPT 实现
# 根据 Apache 许可证 2.0 版本授权
# 导入所需库和模块
# 定义 Idefics 模型的配置类
# 导入相关的模型输出类和预训练配置类
# 定义 Idefics 模型的输出类，包含过去的键/值信息以加速顺序解码
    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.

            If `past_key_values` is used only the last hidden-state of the sequences of shape `(batch_size, 1,
            hidden_size)` is output.
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and optionally if
            `config.is_encoder_decoder=True` 2 additional tensors of shape `(batch_size, num_heads,
            encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and optionally if
            `config.is_encoder_decoder=True` in the cross-attention blocks) that can be used (see `past_key_values`
            input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        image_hidden_states (`tuple(torch.FloatTensor)`, *optional*):
            Tuple of `torch.FloatTensor` (one for the output of the image embeddings, `(batch_size, num_images,
            sequence_length, hidden_size)`.

            image_hidden_states of the model produced by the vision encoder, and optionally by the perceiver
    """

    # 定义变量并初始化为 None，用于存储模型的不同输出
    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    image_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
@dataclass
class IdeficsCausalLMOutputWithPast(ModelOutput):
    """
    Base class for Idefics causal language model (or autoregressive) outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

            Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        image_hidden_states (`tuple(torch.FloatTensor)`, *optional*):
            Tuple of `torch.FloatTensor` (one for the output of the image embeddings, `(batch_size, num_images,
            sequence_length, hidden_size)`.

            image_hidden_states of the model produced by the vision encoder, and optionally by the perceiver
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    image_hidden_states: Optional[Tuple[torch.FloatTensor]] = None


def expand_inputs_for_generation(
    input_ids,
    expand_size=1,
    is_encoder_decoder=False,
    attention_mask=None,
    encoder_outputs=None,
    **model_kwargs,
):
    # 创建一个扩展后的返回索引，用于扩展输入
    expanded_return_idx = (
        torch.arange(input_ids.shape[0]).view(-1, 1).repeat(1, expand_size).view(-1).to(input_ids.device)
    )
    # 根据索引重新排列输入的标识符张量
    input_ids = input_ids.index_select(0, expanded_return_idx)
    
    # 获取或设置模型参数中的像素值
    model_kwargs["pixel_values"] = model_kwargs.get("pixel_values", None)
    
    # 获取或设置模型参数中的图像编码器嵌入
    model_kwargs["image_encoder_embeddings"] = model_kwargs.get("image_encoder_embeddings", None)
    
    # 获取或设置模型参数中的感知器嵌入
    model_kwargs["perceiver_embeddings"] = model_kwargs.get("perceiver_embeddings", None)
    
    # 获取或设置模型参数中的图像注意力掩码
    model_kwargs["image_attention_mask"] = model_kwargs.get("image_attention_mask", None)

    # 如果模型参数中包含"token_type_ids"
    if "token_type_ids" in model_kwargs:
        # 获取"token_type_ids"并重新排列
        token_type_ids = model_kwargs["token_type_ids"]
        model_kwargs["token_type_ids"] = token_type_ids.index_select(0, expanded_return_idx)

    # 如果存在注意力掩码
    if attention_mask is not None:
        # 重新排列注意力掩码
        model_kwargs["attention_mask"] = attention_mask.index_select(0, expanded_return_idx)

    # 如果模型参数中的图像注意力掩码不为空
    if model_kwargs["image_attention_mask"] is not None:
        # 重新排列图像注意力掩码
        model_kwargs["image_attention_mask"] = model_kwargs["image_attention_mask"].index_select(
            0, expanded_return_idx
        )

    # 如果模型参数中的像素值不为空
    if model_kwargs["pixel_values"] is not None:
        # 重新排列像素值
        model_kwargs["pixel_values"] = model_kwargs["pixel_values"].index_select(0, expanded_return_idx)

    # 如果模型参数中的图像编码器嵌入不为空
    elif model_kwargs["image_encoder_embeddings"] is not None:
        # 重新排列图像编码器嵌入
        model_kwargs["image_encoder_embeddings"] = model_kwargs["image_encoder_embeddings"].index_select(
            0, expanded_return_idx
        )

    # 如果模型参数中的感知器嵌入不为空
    elif model_kwargs["perceiver_embeddings"] is not None:
        # 重新排列感知器嵌入
        model_kwargs["perceiver_embeddings"] = model_kwargs["perceiver_embeddings"].index_select(
            0, expanded_return_idx
        )

    # 返回重新排列后的输入标识符和模型参数
    return input_ids, model_kwargs
# 更新用于生成的模型参数
def update_model_kwargs_for_generation(outputs, model_kwargs):
    # 如果输出中包含“past_key_values”键，则将模型参数中的“past_key_values”设置为输出中的值，否则设置为None
    if "past_key_values" in outputs:
        model_kwargs["past_key_values"] = outputs.past_key_values
    else:
        model_kwargs["past_key_values"] = None

    # 更新token_type_ids为最后一个值
    if "token_type_ids" in model_kwargs:
        token_type_ids = model_kwargs["token_type_ids"]
        model_kwargs["token_type_ids"] = torch.cat([token_type_ids, token_type_ids[:, -1].unsqueeze(-1)], dim=-1)

    # 更新注意力掩码
    if "attention_mask" in model_kwargs:
        attention_mask = model_kwargs["attention_mask"]
        model_kwargs["attention_mask"] = torch.cat(
            [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
        )
    if "image_attention_mask" in model_kwargs:
        image_attention_mask = model_kwargs["image_attention_mask"]
        last_mask = image_attention_mask[:, -1, :].unsqueeze(1)
        model_kwargs["image_attention_mask"] = last_mask

    # 获取预先计算的图像隐藏状态
    model_kwargs["image_hidden_states"] = outputs.image_hidden_states

    return model_kwargs


def prepare_inputs_for_generation(input_ids, past_key_values=None, **kwargs):
    token_type_ids = kwargs.get("token_type_ids", None)
    # 如果kwargs中定义了past，则仅使用input_ids的最后一个标记
    if past_key_values:
        input_ids = input_ids[:, -1].unsqueeze(-1)
        if token_type_ids is not None:
            token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

    attention_mask = kwargs.get("attention_mask", None)
    position_ids = kwargs.get("position_ids", None)

    if attention_mask is not None and position_ids is None:
        # 为批量生成动态创建position_ids
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        if past_key_values:
            position_ids = position_ids[:, -1].unsqueeze(-1)

    pixel_values = kwargs.get("pixel_values", None)
    image_encoder_embeddings = kwargs.get("image_encoder_embeddings", None)
    perceiver_embeddings = kwargs.get("perceiver_embeddings", None)
    image_attention_mask = kwargs.get("image_attention_mask", None)
    interpolate_pos_encoding = kwargs.get("interpolate_pos_encoding", False)

    return {
        "input_ids": input_ids,
        "past_key_values": past_key_values,
        "use_cache": kwargs.get("use_cache"),
        "position_ids": position_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
        "pixel_values": pixel_values,
        "image_encoder_embeddings": image_encoder_embeddings,
        "perceiver_embeddings": perceiver_embeddings,
        "image_attention_mask": image_attention_mask,
        "interpolate_pos_encoding": interpolate_pos_encoding,
    }


def freeze_model(model, module_exceptions=[]):
    # 定义一个映射，将字符串映射到对应的 PyTorch 模块类
    mapping = {
        "LayerNorm": nn.LayerNorm,
        "Linear": nn.Linear,
        "Embedding": nn.Embedding,
    }
    # 将模块异常列表中的字符串映射为对应的 PyTorch 模块类
    module_exceptions_mapped = [mapping[m] for m in module_exceptions]
    # 遍历模型的所有模块
    for module in model.modules():
        # 如果模块异常列表不为空，并且模块是模块异常列表中映射后的类之一
        if module_exceptions and any(isinstance(module, t) for t in module_exceptions_mapped):
            # 显式地将模块的 requires_grad 属性设置为 True，以避免任何错误
            module.requires_grad_(True)
        else:
            # 否则将模块的 requires_grad 属性设置为 False
            module.requires_grad_(False)
    # 返回修改后的模型
    return model
class IdeficsDecoupledEmbedding(nn.Embedding):
    # 从 https://pytorch.org/docs/stable/_modules/torch/nn/modules/sparse.html#Embedding 派生而来的类
    """
    实现参数的解耦，允许冻结（或不冻结）嵌入的子集。实际上，常规的 `weight` 可以训练或冻结（即 `partially_freeze=True`），
    如果 `num_additional_embeddings` > 0，则会创建 `num_additional_embeddings` 个额外的参数，这些参数始终被训练。如果
    `num_additional_embeddings=0`，则模块将默认回到 `nn.Embedding` 的常规行为。
    """

    def __init__(
        self,
        num_embeddings,
        num_additional_embeddings,
        embedding_dim,
        partially_freeze: Optional[bool] = False,
        device=None,
        dtype=None,
        padding_idx=None,
        **kwargs,
    ) -> None:
        """
        Args:
            num_embeddings (`int`):
                嵌入字典的大小
            num_additional_embeddings (`int`):
                额外嵌入的数量。仅在 `partially_freeze=True` 时有用。
            embedding_dim (`int`):
                每个嵌入向量的大小
            partially_freeze: (`bool`, *可选*, 默认为 `False`):
                如果为 `True`，则常规的 `weight` 将被冻结。`additional_weight` 永远不会被冻结。
            padding_idx (`int`, *可选*):
                填充索引（必须小于 num_embeddings）

        注意：还有很多其他参数用于初始化标准的 `nn.Embedding`，如 `padding_idx`、`max_norm` 或 `norm_type`。我们不支持这些参数。
        """
        if padding_idx is not None and padding_idx > num_embeddings:
            raise ValueError(f"padding_idx must be within num_embeddings. Got {padding_idx} and {num_embeddings}")
        调用父类的初始化方法
        super().__init__(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            device=device,
            dtype=dtype,
            padding_idx=padding_idx,
            **kwargs,
        )
        self.num_embeddings = num_embeddings
        self.padding_idx = padding_idx
        self.num_additional_embeddings = num_additional_embeddings
        self.partially_freeze = partially_freeze

        如果部分冻结为真
        if partially_freeze:
            冻结权重
            self.weight.requires_grad_(False)

        如果有额外的嵌入数量大于0
        if self.num_additional_embeddings > 0:
            创建额外的嵌入
            self.additional_embedding = nn.Embedding(
                num_embeddings=self.num_additional_embeddings,
                embedding_dim=embedding_dim,
                device=device,
                dtype=dtype,
            )
    # 定义一个前向传播函数，接受输入的标识符
    def forward(self, input_ids):
        """
        we have 2 embeddings, with different indices - one pretrained self.weight and another
        self.additional_embedding.weight that is being trained.

        in order to make a lookup of the input ids, we:
        1. find out the indices of the entries belonging to the 2nd embedding
        2. extract those values while subtracting the size of the first embedding (num_embeddings), since the 2nd
           embedding starts from 0 and not num_embeddings
        3. perform the 2nd embedding lookup
        4. now we handle the 1st embedding, we overwrite indices belonging to the 2nd embedding with a padding index
        5. perform the 1st embedding lookup
        6. now we overwrite the values in the 1st embedding lookup with the values of the 2nd embedding lookup

        note: for the 1st embedding lookup we could have looked up only the low indices and not do the padding, but
        then we have to create a new tensor and populate it with 2 tensors that are spread out across various indices -
        i.e. not a simple concat - I haven't benchmarked the complex case if it's any faster, given that seqlens are
        usually relatively short it's probably not faster or if faster not by much - but might be a good idea to
        measure.

        """
        # 如果没有额外的嵌入层，则直接返回使用 self.weight 进行嵌入查找的结果
        if self.num_additional_embeddings == 0:
            return F.embedding(input_ids, self.weight)

        # 克隆输入，以防后续修改原始输入
        input_ids = input_ids.clone()
        # 找到属于第二个嵌入层的索引
        additional_vocab_indices = torch.where(input_ids >= self.num_embeddings)
        # 提取这些值，同时减去第一个嵌入层的大小（num_embeddings），因为第二个嵌入层从0开始而不是从num_embeddings开始
        input_ids_additional_vocab = input_ids[additional_vocab_indices]
        # 进行第二个嵌入层的查找
        additional_embeddings = self.additional_embedding(input_ids_additional_vocab - self.num_embeddings)

        # 对于成功的查找，将输入标识符替换为0，这些结果将被丢弃
        input_ids[additional_vocab_indices] = 0
        # 进行第一个嵌入层的查找
        full_vector = F.embedding(input_ids, self.weight)

        # 用高索引的记录覆盖
        full_vector[additional_vocab_indices] = additional_embeddings

        return full_vector

    # 返回额外的表示信息
    def extra_repr(self) -> str:
        return "num_embeddings={}, num_additional_embeddings={}, embedding_dim={}, partially_freeze={}".format(
            self.num_embeddings,
            self.num_additional_embeddings,
            self.embedding_dim,
            self.partially_freeze,
        )
class IdeficsDecoupledLinear(nn.Linear):
    # 从 https://pytorch.org/docs/stable/_modules/torch/nn/modules/linear.html#Linear 派生而来
    """
    实现参数的解耦，允许冻结（或不冻结）参数的子集。实际上，常规的 `weight` 可以被训练或冻结（即 `partially_freeze=True`），
    如果 `out_additional_features` > 0，则会创建 `out_additional_features * in_features` 个额外的参数，这些参数始终被训练。
    如果 `out_additional_features=0`，则模块会默认回到 `nn.Linear` 的常规行为。
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        out_additional_features: int = 0,
        bias: bool = True,
        partially_freeze: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        """
        out_additional_features: int。额外可训练维度的数量。仅在 `partially_freeze=True` 时才有意义。
        partially_freeze: bool。如果为 True，则常规的 `weight` 将被冻结，额外参数（如果有）将可训练。
        如果为 False，则默认为 nn.Linear 的常规行为。
        """
        super().__init__(in_features, out_features, bias, device, dtype)
        self.out_additional_features = out_additional_features
        self.partially_freeze = partially_freeze

        self.in_features = in_features
        self.out_features = out_features

        if partially_freeze:
            self.weight.requires_grad_(False)
            if bias:
                self.bias.requires_grad_(False)

        if out_additional_features > 0:
            self.additional_fc = nn.Linear(
                in_features=in_features,
                out_features=out_additional_features,
                bias=bias,
                device=device,
                dtype=dtype,
            )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = F.linear(input, self.weight, self.bias)

        if self.out_additional_features > 0:
            additional_features = self.additional_fc(input)
            output = torch.cat((output, additional_features), -1)

        return output

    def extra_repr(self) -> str:
        """重写 `nn.Linear.extra_repr` 以包含新参数。"""
        return "in_features={}, out_features={}, out_additional_features={}, bias={}, partially_freeze={}".format(
            self.in_features,
            self.out_features,
            self.out_additional_features,
            self.bias is not None,
            self.partially_freeze,
        )


# 这是从 LlamaRMSNorm 改编而来
class IdeficsRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        IdeficsRMSNorm 等同于 T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
    # 前向传播函数，接收隐藏状态作为输入
    def forward(self, hidden_states):
        # 计算隐藏状态的方差，转换为 float32 类型，并在最后一个维度上求均值
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        # 对隐藏状态进行归一化处理，乘以方差的倒数再加上一个很小的数
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        # 如果权重的数据类型是 float16 或 bfloat16，则将隐藏状态转换为相同的数据类型
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        # 返回加权后的隐藏状态
        return self.weight * hidden_states
# 将 IdeficsRMSNorm 添加到 ALL_LAYERNORM_LAYERS 列表中
ALL_LAYERNORM_LAYERS.append(IdeficsRMSNorm)

# 定义一个名为 IdeficsEmbedding 的类，继承自 torch.nn.Module
class IdeficsEmbedding(torch.nn.Module):
    # 初始化方法，接受维度 dim、最大位置嵌入数 max_position_embeddings、基数 base 和设备 device
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        # 初始化类属性
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        # 注册缓冲区 inv_freq，用于存储频率倒数
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # 构建余弦和正弦缓存，以便使 `torch.jit.trace` 正常工作
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    # 设置余弦和正弦缓存的方法
    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # 不同于论文，但使用不同的排列顺序以获得相同的计算结果
        emb = torch.cat((freqs, freqs), dim=-1)
        # 注册余弦缓存和正弦缓存
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    # 前向传播方法
    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )

# 旋转输入的一半隐藏维度的函数
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

# 从 transformers.models.llama.modeling_llama.apply_rotary_pos_emb 复制的函数
def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.
    Args:
        q (`torch.Tensor`): 查询张量。
        k (`torch.Tensor`): 键张量。
        cos (`torch.Tensor`): 旋转嵌入的余弦部分。
        sin (`torch.Tensor`): 旋转嵌入的正弦部分。
        position_ids (`torch.Tensor`):
            与查询和键张量对应的标记位置索引。例如，这可以用于在使用 KV 缓存时传递偏移的位置 id。
        unsqueeze_dim (`int`, *可选*, 默认为 1):
            'unsqueeze_dim' 参数指定要对 cos[position_ids] 和 sin[position_ids] 进行展开的维度，以便它们可以正确广播到 q 和 k 的维度。
            例如，注意 cos[position_ids] 和 sin[position_ids] 的形状为 [batch_size, seq_len, head_dim]。
            然后，如果 q 和 k 的形状为 [batch_size, heads, seq_len, head_dim]，则设置 unsqueeze_dim=1 使 cos[position_ids] 和 sin[position_ids] 可以广播到 q 和 k 的形状。
            类似地，如果 q 和 k 的形状为 [batch_size, seq_len, heads, head_dim]，则设置 unsqueeze_dim=2。
    Returns:
        由使用旋转位置嵌入旋转的查询和键张量组成的 `tuple(torch.Tensor)`。
    """
    # 在指定维度上对 cos[position_ids] 进行展开
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    # 在指定维度上对 sin[position_ids] 进行展开
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    # 计算旋转后的查询嵌入
    q_embed = (q * cos) + (rotate_half(q) * sin)
    # 计算旋转后的键嵌入
    k_embed = (k * cos) + (rotate_half(k) * sin)
    # 返回旋转后的查询和键张量
    return q_embed, k_embed
# 这是从LlamaMLP中改编的类
class IdeficsMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
    ):
        # 初始化函数，定义MLP模型的结构
        super().__init__()
        # 创建一个线性层，用于门控投影
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        # 创建一个线性层，用于下投影
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        # 创建一个线性层，用于上投影
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        # 获取隐藏激活函数
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, x):
        # 前向传播函数，计算MLP模型的输出
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


# 这是从LlamaAttention中改编的类
class IdeficsAttention(nn.Module):
    """来自'Attention Is All You Need'论文的多头注意力机制"""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout: float = 0.0,
        is_cross_attention: bool = False,
        config: PretrainedConfig = None,
        qk_layer_norms: bool = False,
        ):
        # 调用父类的初始化方法
        super().__init__()
        # 设置隐藏层大小
        self.hidden_size = hidden_size
        # 设置头数
        self.num_heads = num_heads
        # 计算每个头的维度
        self.head_dim = hidden_size // num_heads
        # 设置dropout率
        self.dropout = dropout
        # 设置是否是因果关系
        self.is_causal = True

        # 检查隐藏层大小是否可以被头数整除
        if (self.head_dim * num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {num_heads})."
            )

        # 设置是否是跨注意力
        self.is_cross_attention = is_cross_attention

        # 检查是否有scaled_dot_product_attention函数
        if not hasattr(nn.functional, "scaled_dot_product_attention"):
            raise ValueError("this model requires pytorch 2.0 or higher")

        # 如果是跨注意力
        if self.is_cross_attention:
            # 计算键值对输入维度
            kv_input_dim = (
                self.hidden_size if not hasattr(config.vision_config, "embed_dim") else config.vision_config.embed_dim
            )
            # 初始化查询投影层
            self.q_proj = nn.Linear(
                self.hidden_size,
                num_heads * self.head_dim,
                bias=False,
            )
            # 初始化键投影层
            self.k_proj = nn.Linear(kv_input_dim, num_heads * self.head_dim, bias=False)
            # 初始化值投影层
            self.v_proj = nn.Linear(
                kv_input_dim,
                num_heads * self.head_dim,
                bias=False,
            )
        else:
            # 初始化查询投影层
            self.q_proj = nn.Linear(
                self.hidden_size,
                num_heads * self.head_dim,
                bias=False,
            )
            # 初始化键投影层
            self.k_proj = nn.Linear(
                self.hidden_size,
                num_heads * self.head_dim,
                bias=False,
            )
            # 初始化值投影层
            self.v_proj = nn.Linear(
                self.hidden_size,
                num_heads * self.head_dim,
                bias=False,
            )
        # 初始化输出投影层
        self.o_proj = nn.Linear(
            num_heads * self.head_dim,
            hidden_size,
            bias=False,
        )
        # 初始化���转嵌入
        self.rotary_emb = IdeficsEmbedding(self.head_dim)

        # 设置查询键层归一化
        self.qk_layer_norms = qk_layer_norms
        if self.qk_layer_norms:
            self.q_layer_norm = IdeficsRMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.k_layer_norm = IdeficsRMSNorm(self.head_dim, eps=config.rms_norm_eps)

    # 定义形状函数
    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    # 前向传播函数
    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
# 定义一个名为 IdeficsDecoderLayer 的类，继承自 nn.Module 类
class IdeficsDecoderLayer(nn.Module):
    # 初始化方法，接受一个 IdeficsConfig 类型的参数 config
    def __init__(self, config: IdeficsConfig):
        # 调用父类的初始化方法
        super().__init__()
        # 从 config 中获取 hidden_size，并赋值给 self.hidden_size
        self.hidden_size = config.hidden_size
        # 创建一个 IdeficsAttention 对象，并赋值给 self.self_attn
        self.self_attn = IdeficsAttention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.dropout,
            config=config,
        )
        # 创建一个 IdeficsMLP 对象，并赋值给 self.mlp
        self.mlp = IdeficsMLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )
        # 创建一个 IdeficsRMSNorm 对象，并赋值给 self.input_layernorm
        self.input_layernorm = IdeficsRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # 创建一个 IdeficsRMSNorm 对象，并赋值给 self.post_attention_layernorm
        self.post_attention_layernorm = IdeficsRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # 从 config 中获取 dropout，并赋值给 self.dropout
        self.dropout = config.dropout

    # 前向传播方法
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """

        residual = hidden_states

        # 对输入进行 layer normalization
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        # 进行自注意力机制计算
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        # 对输出进行 dropout
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        # 加上残差连接
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        # 对输出进行 layer normalization
        hidden_states = self.post_attention_layernorm(hidden_states)
        # 进行全连接层计算
        hidden_states = self.mlp(hidden_states)
        # 对输出进行 dropout
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        # 加上残差连接
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs
class IdeficsGatedCrossAttentionLayer(nn.Module):
    # 定义一个自定义的 PyTorch 模块，用于执行交叉注意力操作
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_hidden_states: Optional[torch.Tensor] = None,
        image_attention_mask: Optional[torch.Tensor] = None,
        cross_attention_gate: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
    LLAMA_START_DOCSTRING = r"""
        This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
        library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
        etc.)

        This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
        Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
        and behavior.

        Parameters:
            config ([`IdeficsConfig`]):
                Model configuration class with all the parameters of the model. Initializing with a config file does not
                load the weights associated with the model, only the configuration. Check out the
                [`~PreTrainedModel.from_pretrained`] method to load the model weights.
    """

    @add_start_docstrings(
        "The bare LLaMA Model outputting raw hidden-states without any specific head on top.",
        LLAMA_START_DOCSTRING,
    )
    # 定义一个继承自 PreTrainedModel 的模型类，用于输出原始隐藏状态而不添加特定的头部
    class IdeficsPreTrainedModel(PreTrainedModel):
        config_class = IdeficsConfig
        base_model_prefix = "model"
        supports_gradient_checkpointing = True
        _no_split_modules = ["IdeficsDecoderLayer", "IdeficsGatedCrossAttentionLayer"]
        _supports_sdpa = True

        def _init_weights(self, module):
            # 重要提示：这个 Idefics 的移植版本不适用于从头开始训练，只适用于推理和微调
            # 因此正确的初始化权重代码已被移除，应该使用 m4 代码库进行从头训练，其中包含正确的代码。
            std = self.config.initializer_range
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=std)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.Embedding):
                module.weight.data.normal_(mean=0.0, std=std)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()

        # Adapted from transformers.modeling_utils.PreTrainedModel._check_and_enable_sdpa
        @classmethod
    # 检查并启用 SDPA（Sparse-Dense Parallel Attention）实现，返回预训练配置对象
    def _check_and_enable_sdpa(cls, config, hard_check_only: bool = False) -> PretrainedConfig:
        # 我们移除对 `is_torch_sdpa_available()` 和 `cls._supports_sdpa` 的检查，因为 Falcon 支持从 torch==2.0.0 开始的 SDPA（无需 2.1 版本）
        # 检查是否启用了 BetterTransformer，如果是则直接返回配置对象
        _is_bettertransformer = getattr(cls, "use_bettertransformer", False)
        if _is_bettertransformer:
            return config

        # 如果不仅进行硬性检查，则将注意力实现设置为 "sdpa"
        if not hard_check_only:
            config._attn_implementation = "sdpa"
        # 返回配置对象
        return config
# LLAMA_INPUTS_DOCSTRING 是一个空的文档字符串，用于描述 LLAMA 模型的输入
LLAMA_INPUTS_DOCSTRING = r"""
"""


# 创建一个 LLAMA 模型，输出原始隐藏状态，没有特定的头部
# 继承自 IdeficsPreTrainedModel 类
@add_start_docstrings(
    "The bare LLaMA Model outputting raw hidden-states without any specific head on top.",
    LLAMA_START_DOCSTRING,
)
class IdeficsModel(IdeficsPreTrainedModel):
    """
    Transformer decoder consisting of `config.num_hidden_layers` layers. Each layer is a [`IdeficsDecoderLayer`]

    Args:
        config: IdeficsConfig
    """

    # 初始化方法，接受一个 IdeficsConfig 对象作为参数
    def __init__(self, config: IdeficsConfig):
        super().__init__(config)
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        # 创建一个 IdeficsDecoupledEmbedding 对象，用于处理嵌入层
        self.embed_tokens = IdeficsDecoupledEmbedding(
            num_embeddings=config.vocab_size,
            num_additional_embeddings=config.additional_vocab_size,
            embedding_dim=config.hidden_size,
            partially_freeze=config.freeze_text_layers,
            padding_idx=self.padding_idx,
        )

        self.image_size = config.vision_config.image_size
        self.vision_config = config.vision_config
        # 创建一个 IdeficsVisionTransformer 对象，用于处理视觉信息
        self.vision_model = IdeficsVisionTransformer(config.vision_config)

        # 如果配置中使用了 Resampler，则创建一个 IdeficsPerceiverResampler 对象
        if config.use_resampler:
            perceiver_config = config.perceiver_config
            self.perceiver_resampler = IdeficsPerceiverResampler(
                config,
                config.vision_config.embed_dim,
                perceiver_config.resampler_depth,
                perceiver_config.resampler_n_heads,
                perceiver_config.resampler_head_dim,
                perceiver_config.resampler_n_latents,
            )

        # 创建一个包含多个 IdeficsDecoderLayer 对象的模块列表
        self.layers = nn.ModuleList([IdeficsDecoderLayer(config) for _ in range(config.num_hidden_layers)])

        self.cross_layer_interval = config.cross_layer_interval
        num_cross_layers = config.num_hidden_layers // self.cross_layer_interval
        # 创建一个包含多个 IdeficsGatedCrossAttentionLayer 对象的模块列表
        self.gated_cross_attn_layers = nn.ModuleList(
            [IdeficsGatedCrossAttentionLayer(config) for _ in range(num_cross_layers)]
        )
        self.gradient_checkpointing = False

        # 创建一个 IdeficsRMSNorm 对象，用于处理归一化
        self.norm = IdeficsRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # 初始化权重并应用最终处理
        self.post_init()

        # 冻结相关参数
        self.freeze_relevant_params(config)

    # 冻结相关参数的方法
    def freeze_relevant_params(self, config=None):
        if config is None:
            config = self.config

        if config.freeze_text_layers:
            self.freeze_text_layers(config.freeze_text_module_exceptions)

        if config.freeze_vision_layers:
            freeze_model(self.vision_model, module_exceptions=config.freeze_vision_module_exceptions)

    # 冻结文本层参数的方法
    def freeze_text_layers(self, module_exceptions=[]):
        for module in [self.layers, self.norm]:
            freeze_model(module, module_exceptions=module_exceptions)

    # 冻结视觉层参数的方法
    def freeze_vision_layers(self, module_exceptions=[]):
        freeze_model(self.vision_model, module_exceptions=module_exceptions)

    # 获取输入嵌入的方法
    def get_input_embeddings(self):
        return self.embed_tokens
    # 设置输入嵌入层的数值
    def set_input_embeddings(self, value):
        self.embed_tokens = value

    # 将模型的输入参数传递给 forward 方法，并添加文档字符串
    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,  # 输入的 token ID
        attention_mask: Optional[torch.Tensor] = None,  # 注意力掩码
        position_ids: Optional[torch.LongTensor] = None,  # 位置 ID
        past_key_values: Optional[List[torch.FloatTensor]] = None,  # 过去的键值对
        inputs_embeds: Optional[torch.FloatTensor] = None,  # 输入的嵌入向量
        pixel_values: Optional[torch.FloatTensor] = None,  # 像素值
        image_encoder_embeddings: Optional[torch.FloatTensor] = None,  # 图像编码器的嵌入向量
        perceiver_embeddings: Optional[torch.FloatTensor] = None,  # 感知器的嵌入向量
        image_attention_mask: Optional[torch.Tensor] = None,  # 图像的注意力掩码
        use_cache: Optional[bool] = None,  # 是否使用缓存
        output_attentions: Optional[bool] = None,  # 是否输出注意力权重
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态
        interpolate_pos_encoding: Optional[bool] = False,  # 是否插值位置编码
        return_dict: Optional[bool] = None,  # 是否返回字典形式的结果
class IdeficsForVisionText2Text(IdeficsPreTrainedModel):
    # 定义一个新的类，继承自IdeficsPreTrainedModel类
    _keys_to_ignore_on_load_missing = [r"lm_head.weight"]
    # 定义一个私有变量_keys_to_ignore_on_load_missing，用于在加载时忽略指定的键
    _tied_weights_keys = ["model.embed_tokens.weight", "lm_head.weight"]
    # 定义一个私有变量_tied_weights_keys，用于指定需要绑定权重的键

    def __init__(self, config, vision_model=None):
        # 初始化函数，接受config和vision_model两个参数
        super().__init__(config)
        # 调用父类的初始化函数
        self.model = IdeficsModel(config)
        # 创建一个IdeficsModel对象并赋值给self.model

        self.lm_head = IdeficsDecoupledLinear(
            in_features=config.hidden_size,
            out_features=config.vocab_size,
            out_additional_features=config.additional_vocab_size,
            bias=False,
            partially_freeze=config.freeze_lm_head,
        )
        # 创建一个IdeficsDecoupledLinear对象并赋值给self.lm_head，设置相关参数

        # Initialize weights and apply final processing
        # 初始化权重并应用最终处理
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens
        # 返回self.model的embed_tokens属性

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value
        # 设置self.model的embed_tokens属性为给定的value

    def get_output_embeddings(self):
        return self.lm_head
        # 返回self.lm_head属性

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings
        # 设置self.lm_head属性为给定的new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder
        # 设置self.model属性为给定的decoder

    def get_decoder(self):
        return self.model
        # 返回self.model属性

    def tie_weights(self):
        """
        Overwrite `transformers.modeling_utils.PreTrainedModel.tie_weights` to handle the case of
        IdeficsDecoupledLinear and IdeficsDecoupledEmbedding.
        """
        # 重写`transformers.modeling_utils.PreTrainedModel.tie_weights`以处理IdeficsDecoupledLinear和IdeficsDecoupledEmbedding的情况
        output_embeddings = self.get_output_embeddings()
        # 获取输出嵌入层
        input_embeddings = self.get_input_embeddings()
        # 获取输入嵌入层

        if getattr(self.config, "tie_word_embeddings", True):
            # 如果配置中存在"tie_word_embeddings"属性且为True
            output_embeddings.weight = input_embeddings.weight
            # 将输出嵌入层的权重与输入嵌入层的权重绑定
            if input_embeddings.num_additional_embeddings > 0:
                assert output_embeddings.out_additional_features == input_embeddings.num_additional_embeddings
                output_embeddings.additional_fc.weight = input_embeddings.additional_embedding.weight
                # 如果输入嵌入层有额外的嵌入，将输出嵌入层的额外嵌入与输入嵌入层的额外嵌入绑定

        if hasattr(output_embeddings, "out_features") and hasattr(input_embeddings, "num_embeddings"):
            # 如果输出嵌入层有"out_features"属性且输入嵌入层有"num_embeddings"属性
            output_embeddings.out_features = input_embeddings.num_embeddings
            # 设置输出嵌入层的输出特征数为输入嵌入层的嵌入数
            if hasattr(output_embeddings, "out_additional_features") and hasattr(
                input_embeddings, "num_additional_embeddings"
            ):
                output_embeddings.out_additional_features = input_embeddings.num_additional_embeddings
                # 如果输出嵌入层有"out_additional_features"属性且输入嵌入层有"num_additional_embeddings"属性，设置输出嵌入层的额外特征数为输入嵌入层的额外嵌入数

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=IdeficsCausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    # 添加模型前向传播的文档字符串和替换返回文档字符串的装饰器
    # 定义一个方法用于前向传播，接收多个参数，包括输入的张量、注意力掩码、位置编码等
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        image_encoder_embeddings: Optional[torch.FloatTensor] = None,
        perceiver_embeddings: Optional[torch.FloatTensor] = None,
        image_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: Optional[bool] = False,
        return_dict: Optional[bool] = None,
    
    # 准备生成的输入，接收输入的张量和过去的状态
    def prepare_inputs_for_generation(self, input_ids, past=None, **kwargs):
        # 从kwargs中获取图像隐藏状态
        image_hidden_states = kwargs.pop("image_hidden_states", None)
        # 如果图像隐藏状态不为空
        if image_hidden_states is not None:
            # 如果配置使用重采样器
            if self.config.use_resampler:
                # 将图像隐藏状态放入感知器嵌入中
                kwargs["perceiver_embeddings"] = image_hidden_states
            else:
                # 否则将图像隐藏状态放入图像编码器嵌入中
                kwargs["image_encoder_embeddings"] = image_hidden_states
            # 将像素值设为None
            kwargs["pixel_values"] = None
        # 准备生成的输入
        inputs = prepare_inputs_for_generation(input_ids, past=past, **kwargs)
        # 不需要的参数
        unwanted_kwargs = ["token_type_ids"]
        # 遍历不需要的参数
        for kwarg in unwanted_kwargs:
            # 从输入中删除这些参数
            inputs.pop(kwarg, None)
        # 返回输入
        return inputs

    # 扩展生成的输入
    @staticmethod
    def _expand_inputs_for_generation(
        *args,
        **model_kwargs,
    ):
        return expand_inputs_for_generation(*args, **model_kwargs)

    # 更新生成时的模型参数
    @staticmethod
    def _update_model_kwargs_for_generation(outputs, model_kwargs, is_encoder_decoder):
        return update_model_kwargs_for_generation(outputs, model_kwargs)

    # 重新排序缓存
    @staticmethod
    def _reorder_cache(past, beam_idx):
        # 重新排序过去的状态
        reordered_past = ()
        for layer_past in past:
            reordered_past += (tuple(past_state.index_select(0, beam_idx) for past_state in layer_past),)
        return reordered_past
```