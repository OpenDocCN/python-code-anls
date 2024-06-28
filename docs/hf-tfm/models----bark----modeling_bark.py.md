# `.\models\bark\modeling_bark.py`

```
# 定义了一个 PyTorch 模型中的自注意力机制类 BarkSelfAttention
class BarkSelfAttention(nn.Module):
    # 从 GPTNeoSelfAttention 和 Bark 代码适配而来的自注意力机制类 BarkSelfAttention
    # BarkSelfAttention 可以有两种注意力类型，即全局注意力和因果注意力
    def __init__(self, config, is_causal=False):
        super().__init__()

        # regularization
        self.dropout = config.dropout  # 设置对象的 dropout 率
        self.attn_dropout = nn.Dropout(config.dropout)  # 创建一个 Dropout 层，用于注意力计算中的 dropout
        self.resid_dropout = nn.Dropout(config.dropout)  # 创建一个 Dropout 层，用于残差连接中的 dropout

        self.embed_dim = config.hidden_size  # 设置嵌入维度
        self.num_heads = config.num_heads  # 设置注意力头的数量
        self.head_dim = self.embed_dim // self.num_heads  # 计算每个注意力头的维度

        if config.hidden_size % config.num_heads != 0:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )

        # key, query, value projections for all heads, but in a batch
        self.att_proj = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=config.bias)
        # 使用线性变换定义注意力机制的 key、query、value 投影，以及多头机制
        # 输出投影
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=config.bias)

        self.is_causal = is_causal  # 是否使用因果关系（causal）的标志
        if is_causal:
            block_size = config.block_size
            # 创建一个下三角形式的因果关系矩阵，并注册为模型的缓冲区
            bias = torch.tril(torch.ones((block_size, block_size), dtype=bool)).view(1, 1, block_size, block_size)
            self.register_buffer("bias", bias)

    # Copied from transformers.models.gpt_neo.modeling_gpt_neo.GPTNeoSelfAttention._split_heads
    def _split_heads(self, tensor, num_heads, attn_head_size):
        """
        Splits hidden_size dim into attn_head_size and num_heads
        """
        new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
        # 重新塑造张量的形状，将隐藏层维度分割为注意力头大小和注意力头数量
        tensor = tensor.view(new_shape)
        return tensor.permute(0, 2, 1, 3)  # 返回重组后的张量，变换维度顺序为 (batch, head, seq_length, head_features)

    def _merge_heads(self, tensor, num_heads, attn_head_size):
        """
        Merges attn_head_size dim and num_attn_heads dim into hidden_size
        """

        # re-assemble all head outputs side by side
        # (batch, num_heads, seq_len, attn_head_size) -> (batch, seq_len, num_heads*attn_head_size)
        tensor = tensor.transpose(1, 2).contiguous()
        # 将所有注意力头的输出重新组合到一起，变换维度为 (batch, seq_len, num_heads*attn_head_size)
        tensor = tensor.view(tensor.size()[:-2] + (num_heads * attn_head_size,))

        return tensor
    # 定义注意力机制函数，接受查询（query）、键（key）、值（value）、注意力掩码（attention_mask）和头部掩码（head_mask）
    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        # 计算注意力权重，使用query与key的转置矩阵相乘，乘以1除以query维度与key维度的平方根
        attn_weights = torch.matmul(query, key.transpose(-1, -2)) * (1.0 / math.sqrt(self.head_dim))

        # 如果是因果注意力机制，需要处理因果性，即当前位置只能依赖于之前的位置
        if self.is_causal:
            query_length, key_length = query.size(-2), key.size(-2)

            # 填充注意力权重的左上部分（上三角部分）为负无穷大（inf）
            attn_weights = attn_weights.masked_fill(
                self.bias[:, :, key_length - query_length : key_length, :key_length] == 0,
                torch.finfo(attn_weights.dtype).min,
            )

        # 如果有注意力掩码，将其应用于注意力权重
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # 对注意力权重进行 softmax 操作，使得所有注意力权重的总和为1
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_weights = attn_weights.to(value.dtype)

        # 对注意力权重应用注意力dropout，以减少过拟合
        attn_weights = self.attn_dropout(attn_weights)

        # 如果指定了头部掩码，将其应用于注意力权重
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        # 计算注意力输出，将注意力权重与值进行加权求和
        attn_output = torch.matmul(attn_weights, value)

        return attn_output, attn_weights

    # 定义前向传播函数，接受隐藏状态（hidden_states）、注意力掩码（attention_mask）、过去键值（past_key_values）、头部掩码（head_mask）、缓存使用标志（use_cache）、输出注意力权重标志（output_attentions）
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        past_key_values=None,
        head_mask=None,
        use_cache=False,
        output_attentions=False,
    ):
        # 通过线性投影层将隐藏状态映射为查询（query）、键（key）、值（value）
        query, key, value = self.att_proj(hidden_states).split(self.embed_dim, dim=2)

        # 将查询（query）、键（key）、值（value）按头部数和头部维度进行分割
        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        # 如果存在过去的键值，将当前的键值与过去的键值连接起来
        if past_key_values is not None:
            past_key = past_key_values[0]
            past_value = past_key_values[1]
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        # 如果需要使用缓存，设置当前的键值对为当前状态的键值对
        if use_cache is True:
            present = (key, value)
        else:
            present = None

        # 调用注意力函数进行注意力计算，得到注意力输出和注意力权重
        attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)

        # 合并多头注意力的输出结果
        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)

        # 通过输出投影层映射为最终的注意力输出
        attn_output = self.out_proj(attn_output)

        # 应用残差连接的dropout，以防止过拟合
        attn_output = self.resid_dropout(attn_output)

        # 将注意力输出和可能的缓存输出作为模型的输出
        outputs = (attn_output, present)

        # 如果需要输出注意力权重，将注意力权重也添加到模型输出中
        if output_attentions:
            outputs += (attn_weights,)

        return outputs
    class BarkSelfFlashAttention2(BarkSelfAttention):
        """
        Bark flash attention module. This module inherits from `BarkSelfAttention` as the weights of the module stays
        untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
        flash attention and deal with padding tokens in case the input contains any of them.
        """

        # Copied from transformers.models.llama.modeling_llama.LlamaFlashAttention2.__init__
        def __init__(self, *args, **kwargs):
            # 调用父类的初始化函数
            super().__init__(*args, **kwargs)

            # TODO: Should be removed once Flash Attention for RoCm is bumped to 2.1.
            # flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignement, that was made default for flash_attn>=2.1. This attribute is used to handle this difference. Reference: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.
            # Beware that with flash_attn<2.1, using q_seqlen != k_seqlen (except for the case q_seqlen == 1) produces a wrong mask (top-left).
            # 设置一个标志来表示是否使用顶部左对齐的掩码
            self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

        def _split_heads(self, tensor, num_heads, attn_head_size):
            """
            Splits hidden_size dim into attn_head_size and num_heads
            """
            # 重新调整张量的形状，将隐藏尺寸分成头数和注意力头大小
            new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
            tensor = tensor.view(new_shape)
            # Flash attention 要求输入具有以下形状
            # batch_size x seq_length x head_dim x hidden_dim - (batch, seq_length, head, head_features)
            return tensor

        def _merge_heads(self, tensor, num_heads, attn_head_size):
            """
            Merges attn_head_size dim and num_attn_heads dim into hidden_size
            """
            # 重新组合所有头部的输出并排放在一起
            # (batch, seq_len, num_heads, attn_head_size) -> (batch, seq_len, num_heads*attn_head_size)
            tensor = tensor.view(tensor.size()[:-2] + (num_heads * attn_head_size,))
            return tensor

        def forward(
            self,
            hidden_states,
            attention_mask=None,
            past_key_values=None,
            head_mask=None,
            use_cache=False,
            output_attentions=False,
        ):
            # 获取隐藏状态张量的批量大小、查询长度和最后一个维度的大小
            batch_size, query_len, _ = hidden_states.size()

            # 使用注意力投影层分别计算查询、键、值，并按照 embed_dim 维度进行切分
            query, key, value = self.att_proj(hidden_states).split(self.embed_dim, dim=2)

            # 将切分后的查询、键、值张量按照 num_heads 和 head_dim 进行重新组合
            query = self._split_heads(query, self.num_heads, self.head_dim)
            key = self._split_heads(key, self.num_heads, self.head_dim)
            value = self._split_heads(value, self.num_heads, self.head_dim)

            if past_key_values is not None:
                # 如果过去的键值对不为 None，则进行维度转置操作
                # (batch, head, seq_length, head_features) -> (batch, seq_length, head, head_features)
                past_key = past_key_values[0].transpose(1, 2)
                past_value = past_key_values[1].transpose(1, 2)
                # 在 seq_length 维度上合并当前和过去的键值对
                key = torch.cat((past_key, key), dim=1)
                value = torch.cat((past_value, value), dim=1)

            if use_cache is True:
                # 如果使用缓存，则将键和值张量的 head 维度与 seq_length 维度交换位置
                # (batch, head, seq_length, head_features)
                present = (key.transpose(1, 2), value.transpose(1, 2))
            else:
                present = None

            # 执行闪电注意力机制的前向传播，得到注意力输出
            attn_output = self._flash_attention_forward(query, key, value, attention_mask, query_len, dropout=self.dropout)

            # 将多头注意力的输出张量按 head 维度进行合并
            attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
            # 使用输出投影层处理合并后的注意力输出
            attn_output = self.out_proj(attn_output)
            # 对输出应用残差连接的 dropout
            attn_output = self.resid_dropout(attn_output)

            # 组装最终的输出元组
            outputs = (attn_output, present)
            if output_attentions:
                # 如果需要输出注意力权重，则设置 attn_weights 为 None 并加入到 outputs 中
                attn_weights = None
                outputs += (attn_weights,)

            return outputs

        # 从 transformers.models.llama.modeling_llama.LlamaFlashAttention2._flash_attention_forward 复制而来
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
        # 如果未使用 Flash Attention 中的 top-left mask，则确定是否是因果关系（causal）
        if not self._flash_attn_uses_top_left_mask:
            causal = self.is_causal
        else:
            # TODO: 一旦 Flash Attention for RoCm 升级到 2.1 版本，可以移除 `query_length != 1` 的检查。详细信息请参考 LlamaFlashAttention2 __init__ 中的注释。
            # 否则，需要同时满足因果关系（causal）和 query_length 不等于 1
            causal = self.is_causal and query_length != 1

        # 如果存在至少一个填充标记的情况
        if attention_mask is not None:
            # 获取批次大小
            batch_size = query_states.shape[0]
            # 调用 _upad_input 方法，用于处理输入数据的填充问题
            query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(
                query_states, key_states, value_states, attention_mask, query_length
            )

            # 获取当前序列长度
            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            # 获取批次中最大序列长度
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

            # 调用 flash_attn_varlen_func 方法计算注意力输出（未经填充）
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

            # 调用 pad_input 方法对注意力输出进行填充处理
            attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
        else:
            # 如果不存在填充标记，则直接调用 flash_attn_func 方法计算注意力输出
            attn_output = flash_attn_func(
                query_states, key_states, value_states, dropout, softmax_scale=softmax_scale, causal=causal
            )

        # 返回注意力输出
        return attn_output

    # 从 transformers.models.llama.modeling_llama.LlamaFlashAttention2._upad_input 复制而来
    # 定义一个私有方法，用于处理注意力机制的输入数据，根据输入的参数进行数据处理和重组
    def _upad_input(self, query_layer, key_layer, value_layer, attention_mask, query_length):
        # 调用 _get_unpad_data 函数获取未填充数据的索引、当前序列长度及批次内最大序列长度信息
        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)
        
        # 获取 key_layer 的形状信息，包括批次大小、键-值序列长度、键值头数和头维度
        batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape

        # 根据 indices_k 重新索引 key_layer，重塑形状以便与未填充数据匹配
        key_layer = index_first_axis(
            key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )

        # 根据 indices_k 重新索引 value_layer，重塑形状以便与未填充数据匹配
        value_layer = index_first_axis(
            value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )

        # 根据 query_length 的不同情况处理 query_layer
        if query_length == kv_seq_len:
            # 如果 query_length 等于键-值序列长度，则根据 indices_k 重新索引 query_layer
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, self.num_heads, head_dim), indices_k
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            # 如果 query_length 等于 1，则使用简化的处理方式
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            )  # 这里有一个内存拷贝，效率较差。
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            # 否则，假设存在左填充，根据 query_length 和 attention_mask 处理 query_layer
            attention_mask = attention_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, attention_mask)

        # 返回处理后的 query_layer、key_layer、value_layer、查询索引 indices_q、
        # 以及 cu_seqlens 和 max_seqlen_in_batch 的元组
        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )
BARK_ATTENTION_CLASSES = {
    "eager": BarkSelfAttention,  # 定义一个字典，将字符串映射到对应的自定义自注意力类
    "flash_attention_2": BarkSelfFlashAttention2,  # 同上，另一个字符串映射到自定义闪存注意力类
}


class BarkLayerNorm(nn.Module):
    """LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False."""

    def __init__(self, hidden_size, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))  # 初始化可学习的权重参数
        self.bias = nn.Parameter(torch.zeros(hidden_size)) if bias else None  # 根据bias参数初始化可学习的偏置参数，如果bias=False，则为None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, eps=1e-5)  # 使用PyTorch的layer_norm函数进行Layer Normalization


class BarkMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.in_proj = nn.Linear(config.hidden_size, 4 * config.hidden_size, bias=config.bias)  # 输入投影层，线性变换，可选择是否包含偏置
        self.out_proj = nn.Linear(4 * config.hidden_size, config.hidden_size, bias=config.bias)  # 输出投影层，线性变换，可选择是否包含偏置
        self.dropout = nn.Dropout(config.dropout)  # Dropout层，根据配置概率丢弃输入
        self.gelu = nn.GELU()  # GELU激活函数

    def forward(self, hidden_states):
        hidden_states = self.in_proj(hidden_states)  # 输入投影层
        hidden_states = self.gelu(hidden_states)  # GELU激活函数
        hidden_states = self.out_proj(hidden_states)  # 输出投影层
        hidden_states = self.dropout(hidden_states)  # Dropout层
        return hidden_states  # 返回处理后的隐藏状态


class BarkBlock(nn.Module):
    def __init__(self, config, is_causal=False):
        super().__init__()

        if is_causal:
            # 如果是因果的，使用自定义的LayerNorm，以便支持可选的偏置
            # 这个自定义的LayerNorm用于与Bark中留有可选偏置的自回归模型（对应于“Text”和“Coarse”模块）保持一致
            self.layernorm_1 = BarkLayerNorm(config.hidden_size, bias=config.bias)
            self.layernorm_2 = BarkLayerNorm(config.hidden_size, bias=config.bias)
        else:
            self.layernorm_1 = nn.LayerNorm(config.hidden_size)  # 否则使用PyTorch标准的LayerNorm
            self.layernorm_2 = nn.LayerNorm(config.hidden_size)

        self.attn = BARK_ATTENTION_CLASSES[config._attn_implementation](config, is_causal=is_causal)  # 根据配置选择对应的注意力机制类

        self.mlp = BarkMLP(config)  # 创建MLP模块

    def forward(
        self,
        hidden_states,
        past_key_values=None,
        attention_mask=None,
        head_mask=None,
        use_cache=False,
        output_attentions=False,
        ):
        # 此处省略了forward函数的余下部分，因为要保持代码完整性，不做更改
        ):
            # 对隐藏状态进行 layer normalization
            intermediary_hidden_states = self.layernorm_1(hidden_states)

            # 使用 self.attn 进行注意力计算
            attn_outputs = self.attn(
                intermediary_hidden_states,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                head_mask=head_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )

            # 获取注意力计算的输出
            attn_output = attn_outputs[0]  # output_attn: output, present_key_values, (attn_weights)
            # 剩余的输出
            outputs = attn_outputs[1:]

            # 更新中间隐藏状态
            intermediary_hidden_states = hidden_states + attn_output
            # 经过第二个层归一化和多层感知机处理
            intermediary_hidden_states = intermediary_hidden_states + self.mlp(
                self.layernorm_2(intermediary_hidden_states)
            )

            # 如果使用缓存，将更新后的中间隐藏状态添加到输出中
            if use_cache:
                outputs = (intermediary_hidden_states,) + outputs
            else:
                # 否则，仅将更新后的中间隐藏状态与原输出的后续部分合并
                outputs = (intermediary_hidden_states,) + outputs[1:]

            # 返回更新后的输出，包括隐藏状态和可能的缓存
            return outputs  # hidden_states, ((present), attentions)
"""
An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
models.
"""

class BarkPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # Configuration class specific to BarkPreTrainedModel
    config_class = BarkConfig

    # Gradient checkpointing support is disabled
    supports_gradient_checkpointing = False

    # Specific attribute for flash attention 2 support
    _supports_flash_attn_2 = True

    def _init_weights(self, module):
        """Initialize the weights of the module based on its type."""
        if isinstance(module, (nn.Linear,)):
            # Initialize linear layers' weights with normal distribution
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            # Initialize embedding weights with normal distribution
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                # Zero out weights corresponding to padding_idx if specified
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            # Initialize LayerNorm bias to zero and weight to 1
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

    @property
    def device(self) -> torch.device:
        """
        `torch.device`: The device on which the module is (assuming that all the module parameters are on the same
        device).
        """
        # Check if the module has a _hf_hook to determine if it has been offloaded
        if not hasattr(self, "_hf_hook"):
            return get_parameter_device(self)
        
        # Traverse through all modules to find the execution device based on _hf_hook
        for module in self.modules():
            if (
                hasattr(module, "_hf_hook")
                and hasattr(module._hf_hook, "execution_device")
                and module._hf_hook.execution_device is not None
            ):
                return torch.device(module._hf_hook.execution_device)

        # Return the device of the parameters if no _hf_hook is found
        return get_parameter_device(self)
"""
This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

Parameters:
    config ([`{config}`]):
        Model configuration class with all the parameters of the model. Initializing with a config file does not
        load the weights associated with the model, only the configuration. Check out the
        [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""
BARK_MODEL_START_DOCSTRING = BARK_MODEL_START_DOCSTRING.strip()

"""
This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
"""
BARK_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
"""
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`BarkConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""
BARK_FINE_INPUTS_DOCSTRING = r"""
    Args:
        codebook_idx (`int`):
            Index of the codebook that will be predicted.
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length, number_of_codebooks)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it. Initially, indices of the first two codebooks are obtained from the `coarse` sub-model. The rest is
            predicted recursively by attending the previously predicted channels. The model predicts on windows of
            length 1024.
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.max_position_embeddings - 1]`.

            [What are position IDs?](../glossary#position-ids)
        head_mask (`torch.Tensor` of shape `(encoder_layers, encoder_attention_heads)`, *optional*):
            Mask to nullify selected heads of the attention modules in the encoder. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*): NOT IMPLEMENTED YET.
        input_embeds (`torch.FloatTensor` of shape `(batch_size, input_sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. If
            `past_key_values` is used, optionally only the last `input_embeds` have to be input (see
            `past_key_values`). This is useful if you want more control over how to convert `input_ids` indices into
            associated vectors than the model's internal embedding lookup matrix.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""

BARK_CAUSAL_MODEL_INPUTS_DOCSTRING = r"""
"""

# GPT2-like autoregressive model
class BarkCausalModel(BarkPreTrainedModel):
    # 使用BarkSubModelConfig类来配置模型
    config_class = BarkSubModelConfig
    # 初始化方法，接受一个配置参数 config
    def __init__(self, config):
        # 调用父类的初始化方法，传入配置参数 config
        super().__init__(config)
        # 将配置参数 config 保存在实例变量中
        self.config = config

        # 初始化输入词嵌入层，根据输入词汇大小和隐藏层大小创建 Embedding 层
        self.input_embeds_layer = nn.Embedding(config.input_vocab_size, config.hidden_size)
        # 初始化位置嵌入层，根据块大小和隐藏层大小创建 Embedding 层
        self.position_embeds_layer = nn.Embedding(config.block_size, config.hidden_size)

        # 初始化 Dropout 层，使用配置中的 dropout 比率
        self.drop = nn.Dropout(config.dropout)

        # 使用列表推导式创建多层 BarkBlock 模块组成的模块列表，每层使用相同的配置和是因果的标志
        self.layers = nn.ModuleList([BarkBlock(config, is_causal=True) for _ in range(config.num_layers)])

        # 根据配置中的 _attn_implementation 判断是否使用 Flash Attention 2
        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"

        # 初始化最终的 LayerNorm 层，使用隐藏层大小和偏置配置进行初始化
        self.layernorm_final = BarkLayerNorm(config.hidden_size, bias=config.bias)

        # 初始化语言模型的线性层，将隐藏层映射到输出词汇大小的空间，没有偏置
        self.lm_head = nn.Linear(config.hidden_size, config.output_vocab_size, bias=False)

        # 禁用渐变检查点，用于后续的优化和训练过程
        self.gradient_checkpointing = False

        # 执行后期初始化，可能包括权重初始化和其他配置
        self.post_init()

    # 返回输入词嵌入层
    def get_input_embeddings(self):
        return self.input_embeds_layer

    # 设置新的输入词嵌入层
    def set_input_embeddings(self, new_embeddings):
        self.input_embeds_layer = new_embeddings
    # 准备用于生成的输入，根据传入的参数进行处理
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        # 从 kwargs 中获取输入的嵌入向量（如果有）
        input_embeds = kwargs.get("input_embeds", None)

        # 从 kwargs 中获取注意力遮罩（如果有）
        attention_mask = kwargs.get("attention_mask", None)
        # 从 kwargs 中获取位置编码（如果有）
        position_ids = kwargs.get("position_ids", None)

        # 如果 past_key_values 不为 None，则执行以下操作
        if past_key_values is not None:
            # 忽略已经被 past_key_values 覆盖的 token
            seq_len = input_ids.shape[1]  # 获取输入序列的长度
            past_length = past_key_values[0][0].shape[2]  # 获取过去键值的长度

            # 如果输入序列的长度大于过去键值的长度，则移除前缀长度为 past_length
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # 否则，默认行为：仅保留最后一个 token
                remove_prefix_length = input_ids.shape[1] - 1

            input_ids = input_ids[:, remove_prefix_length:]  # 更新 input_ids 为移除前缀后的序列

            # input_embeds 已经被使用过，不再需要
            input_embeds = None
        else:
            # 如果 past_key_values 为 None，则执行以下操作
            if input_embeds is not None and kwargs.get("use_cache"):
                seq_len = input_embeds.shape[1]  # 获取嵌入向量的序列长度
            else:
                seq_len = input_ids.shape[1]  # 获取输入序列的长度

        # 确保 attention_mask 和 position_ids 的形状与奇怪的 Bark hack 对序列长度的减少对齐
        if attention_mask is not None:
            attention_mask = attention_mask[:, :seq_len]  # 调整 attention_mask 的长度为 seq_len
        if position_ids is not None:
            position_ids = position_ids[:, :seq_len]  # 调整 position_ids 的长度为 seq_len

        # 如果 attention_mask 存在且 position_ids 不存在，则执行以下操作
        if attention_mask is not None and position_ids is None:
            # 在批次生成时动态创建 position_ids
            position_ids = attention_mask.long().cumsum(-1) - 1  # 根据 attention_mask 创建 position_ids
            position_ids.masked_fill_(attention_mask == 0, 1)  # 将 position_ids 中 attention_mask 为 0 的位置填充为 1
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]  # 如果 past_key_values 存在，截取最后 input_ids 长度的 position_ids
        else:
            position_ids = None  # 否则置为 None

        # 如果 input_embeds 存在且 use_cache 为 True，则返回以下结果字典
        if input_embeds is not None and kwargs.get("use_cache"):
            return {
                "input_ids": None,
                "input_embeds": input_embeds,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "position_ids": position_ids,
                "attention_mask": attention_mask,
            }
        # 否则，返回以下结果字典
        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "position_ids": position_ids,
            "attention_mask": attention_mask,
        }

    # 将 Bark_causal_model 的输入文档字符串添加到模型前向方法中
    @add_start_docstrings_to_model_forward(BARK_CAUSAL_MODEL_INPUTS_DOCSTRING)
    # 定义一个类方法 `forward`，用于模型的前向传播。
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,  # 输入的token IDs张量，可选
        past_key_values: Optional[Tuple[torch.FloatTensor]] = None,  # 用于存储过去的键值，可选
        attention_mask: Optional[torch.Tensor] = None,  # 注意力掩码张量，可选
        position_ids: Optional[torch.Tensor] = None,  # 位置ID张量，可选
        head_mask: Optional[torch.Tensor] = None,  # 头部掩码张量，可选
        labels: Optional[torch.LongTensor] = None,  # 标签张量，可选
        input_embeds: Optional[torch.Tensor] = None,  # 输入的嵌入张量，可选
        use_cache: Optional[bool] = None,  # 是否使用缓存，可选
        output_attentions: Optional[bool] = None,  # 是否输出注意力权重，可选
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态，可选
        return_dict: Optional[bool] = None,  # 是否返回字典格式的结果，可选
    ):
        # 静态方法 `_reorder_cache`，用于在beam搜索或采样时重新排序 `past_key_values` 缓存，
        # 以确保在每个生成步骤中与正确的beam_idx匹配。
        @staticmethod
        def _reorder_cache(
            past_key_values: Tuple[Tuple[torch.Tensor]],  # 包含过去键值的元组，必须为张量
            beam_idx: torch.Tensor  # beam索引张量，指定重新排序顺序
        ) -> Tuple[Tuple[torch.Tensor]]:
            """
            This function is used to re-order the `past_key_values` cache if [`~PreTrainedModel.beam_search`] or
            [`~PreTrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
            beam_idx at every generation step.
            """
            # 对于每个层的过去状态，使用 `beam_idx` 在设备上选择正确的过去状态，返回重新排序后的元组
            return tuple(
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past)
                for layer_past in past_key_values
            )
# 定义 BarkSemanticModel 类，继承自 BarkCausalModel 类
@add_start_docstrings(
    """Bark semantic (or text) model. It shares the same architecture as the coarse model.
    It is a GPT-2 like autoregressive model with a language modeling head on top.""",
    BARK_MODEL_START_DOCSTRING.format(config="BarkSemanticConfig"),
)
class BarkSemanticModel(BarkCausalModel):
    # 指定模型的前缀名称为 'semantic'
    base_model_prefix = "semantic"
    # 指定配置类为 BarkSemanticConfig
    config_class = BarkSemanticConfig

    # 定义生成方法
    def generate(
        self,
        input_ids: torch.Tensor,
        semantic_generation_config: BarkSemanticGenerationConfig = None,
        history_prompt: Optional[Dict[str, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,



# 定义 BarkCoarseModel 类，继承自 BarkCausalModel 类
@add_start_docstrings(
    """Bark coarse acoustics model.
    It shares the same architecture as the semantic (or text) model. It is a GPT-2 like autoregressive model with a
    language modeling head on top.""",
    BARK_MODEL_START_DOCSTRING.format(config="BarkCoarseConfig"),
)
class BarkCoarseModel(BarkCausalModel):
    # 指定模型的前缀名称为 'coarse_acoustics'
    base_model_prefix = "coarse_acoustics"
    # 指定配置类为 BarkCoarseConfig
    config_class = BarkCoarseConfig

    # 定义预处理历史数据的方法
    def preprocess_histories(
        self,
        max_coarse_history: int,
        semantic_to_coarse_ratio: int,
        batch_size: int,
        semantic_generation_config: int,
        codebook_size: int,
        history_prompt: Optional[Dict[str, torch.Tensor]] = None,
    # 定义生成方法
    def generate(
        self,
        semantic_output: torch.Tensor,
        semantic_generation_config: BarkSemanticGenerationConfig = None,
        coarse_generation_config: BarkCoarseGenerationConfig = None,
        codebook_size: int = 1024,
        history_prompt: Optional[Dict[str, torch.Tensor]] = None,
        return_output_lengths: Optional[bool] = None,
        **kwargs,



# 定义 BarkFineModel 类，继承自 BarkPreTrainedModel 类
@add_start_docstrings(
    """Bark fine acoustics model. It is a non-causal GPT-like model with `config.n_codes_total` embedding layers and
    language modeling heads, one for each codebook.""",
    BARK_MODEL_START_DOCSTRING.format(config="BarkFineConfig"),
)
class BarkFineModel(BarkPreTrainedModel):
    # 指定模型的前缀名称为 'fine_acoustics'
    base_model_prefix = "fine_acoustics"
    # 指定配置类为 BarkFineConfig
    config_class = BarkFineConfig
    # 主输入名称为 'codebook_idx'
    main_input_name = "codebook_idx"
    def __init__(self, config):
        # non-causal gpt-like model with one embedding layer and one lm_head for each codebook of Encodec
        # 使用给定的配置初始化模型
        super().__init__(config)
        self.config = config

        # initialize a modified non causal GPT-like model
        # note that for there is one embedding layer and one lm_head for each codebook of Encodec
        # 初始化修改后的非因果关系的类似GPT的模型
        # 每个Encodec编码书中都有一个嵌入层和一个lm_head
        self.input_embeds_layers = nn.ModuleList(
            [nn.Embedding(config.input_vocab_size, config.hidden_size) for _ in range(config.n_codes_total)]
        )
        # 初始化输入嵌入层列表，每个编码书一个嵌入层

        self.position_embeds_layer = nn.Embedding(config.block_size, config.hidden_size)
        # 初始化位置嵌入层，用于位置编码

        self.drop = nn.Dropout(config.dropout)
        # 初始化Dropout层，用于随机丢弃输入的一部分特征

        self.layers = nn.ModuleList([BarkBlock(config, is_causal=False) for _ in range(config.num_layers)])
        # 初始化模型的层列表，每层使用BarkBlock，is_causal参数为False表示非因果关系

        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"
        # 根据配置决定是否使用Flash Attention版本2

        self.layernorm_final = nn.LayerNorm(config.hidden_size)
        # 初始化最终的Layer Norm层，对模型的隐藏状态进行归一化

        self.lm_heads = nn.ModuleList(
            [
                nn.Linear(config.hidden_size, config.output_vocab_size, bias=False)
                for _ in range(config.n_codes_given, config.n_codes_total)
            ]
        )
        # 初始化语言模型头列表，每个编码书一个lm_head

        self.gradient_checkpointing = False
        # 梯度检查点默认关闭

        self.n_codes_total = config.n_codes_total
        # 记录总编码书的数量

        # Initialize weights and apply final processing
        # 初始化权重并应用最终处理
        self.post_init()

    def get_input_embeddings(self):
        # one embedding layers for each codebook
        # 返回每个编码书的输入嵌入层列表
        return self.input_embeds_layers

    def set_input_embeddings(self, new_embeddings):
        # one embedding layers for each codebook
        # 设置每个编码书的输入嵌入层列表
        self.input_embeds_layers = new_embeddings

    def get_output_embeddings(self):
        # one lm_head for each codebook
        # 返回每个编码书的语言模型头列表
        return self.lm_heads

    def set_output_embeddings(self, new_output_embeddings):
        # one lm_head for each codebook
        # 设置每个编码书的语言模型头列表
        self.lm_heads = new_output_embeddings

    def _resize_token_embeddings(self, new_num_tokens, pad_to_multiple_of=None):
        old_embeddings_list = self.get_input_embeddings()
        # 获取当前的输入嵌入层列表

        new_embeddings_list = nn.ModuleList(
            [
                self._get_resized_embeddings(old_embeddings, new_num_tokens, pad_to_multiple_of)
                for old_embeddings in old_embeddings_list
            ]
        )
        # 生成新的调整大小后的输入嵌入层列表

        self.set_input_embeddings(new_embeddings_list)
        # 设置模型的新输入嵌入层列表

        new_num_tokens = new_embeddings_list[0].weight.shape[0]
        # 更新新的嵌入层中的标记数量

        # if word embeddings are not tied, make sure that lm head is resized as well
        # 如果单词嵌入未绑定，则确保也调整lm头的大小
        if self.get_output_embeddings() is not None and not self.config.tie_word_embeddings:
            old_lm_head_list = self.get_output_embeddings()
            # 获取当前的语言模型头列表

            new_lm_head_list = nn.ModuleList(
                [self._get_resized_lm_head(old_lm_head, new_num_tokens) for old_lm_head in old_lm_head_list]
            )
            # 生成新的调整大小后的语言模型头列表

            self.set_output_embeddings(new_lm_head_list)
            # 设置模型的新语言模型头列表

        return self.get_input_embeddings()
    def resize_token_embeddings(
        self, new_num_tokens: Optional[int] = None, pad_to_multiple_of: Optional[int] = None
    ) -> nn.Embedding:
        """
        Resizes input token embeddings matrix of the model if `new_num_tokens != config.vocab_size`.

        Takes care of tying weights embeddings afterwards if the model class has a `tie_weights()` method.

        Arguments:
            new_num_tokens (`int`, *optional*):
                The number of new tokens in the embedding matrix. Increasing the size will add newly initialized
                vectors at the end. Reducing the size will remove vectors from the end. If not provided or `None`, just
                returns a pointer to the input tokens `torch.nn.Embedding` module of the model without doing anything.
            pad_to_multiple_of (`int`, *optional*):
                If set will pad the embedding matrix to a multiple of the provided value.

                This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability
                `>= 7.5` (Volta), or on TPUs which benefit from having sequence lengths be a multiple of 128. For more
                details about this, or help on choosing the correct value for resizing, refer to this guide:
                https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html#requirements-tc

        Return:
            `torch.nn.Embedding`: Pointer to the input tokens Embeddings Module of the model.
        """
        # 调用内部方法调整模型的词嵌入大小，并获取返回的词嵌入模块
        model_embeds = self._resize_token_embeddings(new_num_tokens, pad_to_multiple_of)
        # 如果未指定新的词汇量大小和填充到的倍数，直接返回原始的词嵌入模块
        if new_num_tokens is None and pad_to_multiple_of is None:
            return model_embeds

        # 更新基础模型和当前模型配置的词汇量大小
        self.config.output_vocab_size = model_embeds[0].weight.shape[0]
        self.config.vocab_size = model_embeds[0].weight.shape[0]
        # 更新当前对象的输出词汇量大小和词汇量大小
        self.output_vocab_size = model_embeds[0].weight.shape[0]
        self.vocab_size = model_embeds[0].weight.shape[0]

        # 如果需要，重新绑定权重
        self.tie_weights()

        # 返回调整后的词嵌入模块
        return model_embeds
    # 将输入嵌入列表和输出嵌入列表之间的权重进行绑定或克隆。

    if getattr(self.config, "tie_word_embeddings", True):
        # 如果配置中设置了torchscript标志，则无法处理参数共享，因此我们克隆权重而不是绑定。
        self._tied_weights_keys = []  # 初始化存储绑定权重的键列表
        output_embeddings = self.get_output_embeddings()  # 获取输出嵌入层对象
        input_embeddings = self.get_input_embeddings()    # 获取输入嵌入层对象

        for i in range(self.config.n_codes_total - self.config.n_codes_given):
            # 将输出嵌入层i的权重绑定到输入嵌入层i+1的权重上
            self._tie_or_clone_weights(output_embeddings[i], input_embeddings[i + 1])
            # 记录已绑定权重的键名，格式为"lm_heads.{i}.weight"
            self._tied_weights_keys.append(f"lm_heads.{i}.weight")

    # 递归地对模型的所有子模块调用 _tie_weights 方法，如果模块有该方法的话
    for module in self.modules():
        if hasattr(module, "_tie_weights"):
            module._tie_weights()



    # 生成模型的前向传播方法，接受一系列输入和可选的配置参数作为输入。

    @add_start_docstrings_to_model_forward(BARK_FINE_INPUTS_DOCSTRING)
    def forward(
        self,
        codebook_idx: int,  # 用于预测的代码本身的附加索引
        input_ids: Optional[torch.Tensor] = None,         # 输入的token id张量
        attention_mask: Optional[torch.Tensor] = None,    # 注意力掩码张量
        position_ids: Optional[torch.Tensor] = None,      # 位置id张量
        head_mask: Optional[torch.Tensor] = None,         # 头部掩码张量
        labels: Optional[torch.LongTensor] = None,        # 标签张量（用于监督学习）
        input_embeds: Optional[torch.Tensor] = None,      # 输入嵌入张量（可以替代input_ids）
        output_attentions: Optional[bool] = None,         # 是否输出注意力权重
        output_hidden_states: Optional[bool] = None,      # 是否输出隐藏状态
        return_dict: Optional[bool] = None,               # 是否以字典形式返回结果
    ):



    # 生成模型的生成方法，接受粗粒度输出、语义生成配置、粗粒度生成配置、细粒度生成配置、代码本尺寸等参数以及历史提示的可选字典输入。

    def generate(
        self,
        coarse_output: torch.Tensor,                            # 粗粒度输出的张量
        semantic_generation_config: BarkSemanticGenerationConfig = None,  # 语义生成配置对象
        coarse_generation_config: BarkCoarseGenerationConfig = None,      # 粗粒度生成配置对象
        fine_generation_config: BarkFineGenerationConfig = None,          # 细粒度生成配置对象
        codebook_size: int = 1024,                           # 代码本尺寸，默认为1024
        history_prompt: Optional[Dict[str, torch.Tensor]] = None,  # 历史提示的可选字典输入
        **kwargs,                                            # 其他关键字参数
    ):
"""
The full Bark model, a text-to-speech model composed of 4 sub-models:
- `BarkSemanticModel` (also referred to as the 'text' model): a causal auto-regressive transformer model that
  takes as input tokenized text, and predicts semantic text tokens that capture the meaning of the text.
- `BarkCoarseModel` (also referred to as the 'coarse acoustics' model), also a causal autoregressive transformer,
  that takes as input the results of the last model. It aims at regressing the first two audio codebooks necessary
  to `encodec`.
- `BarkFineModel` (the 'fine acoustics' model), this time a non-causal autoencoder transformer, which iteratively
  predicts the last codebooks based on the sum of the previous codebooks embeddings.
- After having predicted all the codebook channels from the `EncodecModel`, Bark uses it to decode the output audio
  array.

It should be noted that each of the first three modules can support conditional speaker embeddings to condition the
output sound according to a specific predefined voice.
"""
@add_start_docstrings(
    """
    The full Bark model, a text-to-speech model composed of 4 sub-models:
    - [`BarkSemanticModel`] (also referred to as the 'text' model): a causal auto-regressive transformer model that
      takes
    as input tokenized text, and predicts semantic text tokens that capture the meaning of the text.
    - [`BarkCoarseModel`] (also refered to as the 'coarse acoustics' model), also a causal autoregressive transformer,
    that takes into input the results of the last model. It aims at regressing the first two audio codebooks necessary
    to `encodec`.
    - [`BarkFineModel`] (the 'fine acoustics' model), this time a non-causal autoencoder transformer, which iteratively
    predicts the last codebooks based on the sum of the previous codebooks embeddings.
    - having predicted all the codebook channels from the [`EncodecModel`], Bark uses it to decode the output audio
      array.

    It should be noted that each of the first three modules can support conditional speaker embeddings to condition the
    output sound according to specific predefined voice.
    """,
    BARK_START_DOCSTRING,
)
class BarkModel(BarkPreTrainedModel):
    config_class = BarkConfig

    def __init__(self, config):
        super().__init__(config)

        # Initialize the BarkSemanticModel with the provided semantic configuration
        self.semantic = BarkSemanticModel(config.semantic_config)

        # Initialize the BarkCoarseModel with the provided coarse acoustics configuration
        self.coarse_acoustics = BarkCoarseModel(config.coarse_acoustics_config)

        # Initialize the BarkFineModel with the provided fine acoustics configuration
        self.fine_acoustics = BarkFineModel(config.fine_acoustics_config)

        # Initialize the codec_model using AutoModel and the provided codec configuration
        self.codec_model = AutoModel.from_config(config.codec_config)

        # Store the provided configuration for later reference
        self.config = config

    @property
    def device(self) -> torch.device:
        """
        `torch.device`: The device on which the module is (assuming that all the module parameters are on the same
        device).
        """
        # Determine the device on which the BarkModel resides
        # Check if semantic model has the _hf_hook attribute indicating it has been offloaded
        if not hasattr(self.semantic, "_hf_hook"):
            return get_parameter_device(self)  # Return the device of the current module
        # If semantic model has _hf_hook, find the execution device from the hook
        for module in self.semantic.modules():
            if (
                hasattr(module, "_hf_hook")
                and hasattr(module._hf_hook, "execution_device")
                and module._hf_hook.execution_device is not None
            ):
                return torch.device(module._hf_hook.execution_device)  # Return the execution device found
    def enable_cpu_offload(self, gpu_id: Optional[int] = 0):
        r"""
        Offloads all sub-models to CPU using accelerate, reducing memory usage with a low impact on performance. This
        method moves one whole sub-model at a time to the GPU when it is used, and the sub-model remains in GPU until
        the next sub-model runs.

        Args:
            gpu_id (`int`, *optional*, defaults to 0):
                GPU id on which the sub-models will be loaded and offloaded.
        """
        if is_accelerate_available():
            from accelerate import cpu_offload_with_hook  # Importing the function for offloading to CPU
        else:
            raise ImportError("`enable_model_cpu_offload` requires `accelerate`.")

        device = torch.device(f"cuda:{gpu_id}")  # Define the CUDA device based on given GPU id

        if self.device.type != "cpu":
            self.to("cpu")  # Move the entire model to CPU
            torch.cuda.empty_cache()  # Clear GPU cache to free up memory

        # Offload the input embedding layer to CPU and receive a hook for later layers
        self.semantic.input_embeds_layer, _ = cpu_offload_with_hook(self.semantic.input_embeds_layer, device)

        hook = None
        # Offload each sub-model to CPU sequentially and chain hooks between them
        for cpu_offloaded_model in [
            self.semantic,
            self.coarse_acoustics,
            self.fine_acoustics,
        ]:
            _, hook = cpu_offload_with_hook(cpu_offloaded_model, device, prev_module_hook=hook)

        self.fine_acoustics_hook = hook  # Store the final hook after offloading fine_acoustics

        _, hook = cpu_offload_with_hook(self.codec_model, device, prev_module_hook=hook)
        self.codec_model_hook = hook  # Store the hook after offloading the codec_model

        # We'll offload the last model manually.
        self.codec_model_hook = hook

    def codec_decode(self, fine_output, output_lengths=None):
        """Turn quantized audio codes into audio array using encodec."""

        fine_output = fine_output.transpose(0, 1)  # Transpose the fine_output tensor
        emb = self.codec_model.quantizer.decode(fine_output)  # Decode the quantized audio codes

        if output_lengths is not None:
            # Decode audio samples, handling variable lengths with padding
            out = [sample[:, :l].unsqueeze(0) for (sample, l) in zip(emb, output_lengths)]
            audio_arr = [self.codec_model.decoder(sample).squeeze() for sample in out]
        else:
            out = self.codec_model.decoder(emb)  # Decode audio samples without length restrictions
            audio_arr = out.squeeze(1)  # Squeeze the codebook dimension

        return audio_arr

    @torch.no_grad()
    def generate(
        self,
        input_ids: Optional[torch.Tensor] = None,
        history_prompt: Optional[Dict[str, torch.Tensor]] = None,
        return_output_lengths: Optional[bool] = None,
        **kwargs,
    ):
        """Method for generating outputs based on input_ids and optional prompts."""
        # Method for generating outputs based on input_ids and optional prompts
        pass

    @classmethod
    def _check_and_enable_flash_attn_2(
        cls,
        config,
        torch_dtype: Optional[torch.dtype] = None,
        device_map: Optional[Union[str, Dict[str, int]]] = None,
        hard_check_only: bool = False,
        check_device_map: bool = False,
    ):
        """Check and potentially enable flash attention mechanism."""
        # Check and potentially enable flash attention mechanism
        pass
        """
        `_check_and_enable_flash_attn_2`原本不扩展闪存注意力使能到模型的子配置。我们重写原始方法以确保Bark子模型在需要时使用Flash Attention。

        如果你不了解Flash Attention，请查看官方的Flash Attention存储库：
        https://github.com/Dao-AILab/flash-attention

        若要直接通过`BetterTransformer` API使用Flash Attention 1.0，请查看文档的特定部分以了解更多信息：
        https://huggingface.co/docs/transformers/main/en/perf_infer_gpu_one#decoder-models

        该方法检查当前设置是否与Flash Attention兼容，因为它要求模型处于半精度并且不在CPU上运行。

        如果所有检查都通过且`hard_check_only`为False，则该方法将把配置属性`_attn_implementation`设置为"flash_attention_2"，以便模型可以初始化正确的注意力模块。
        """
        # 调用父类方法以检查和启用Flash Attention 2
        config = super()._check_and_enable_flash_attn_2(
            config, torch_dtype, device_map, hard_check_only=hard_check_only, check_device_map=check_device_map
        )

        # 设置语义配置、粗略声学配置和精细声学配置的注意力实现属性与主配置保持一致
        config.semantic_config._attn_implementation = config._attn_implementation
        config.coarse_acoustics_config._attn_implementation = config._attn_implementation
        config.fine_acoustics_config._attn_implementation = config._attn_implementation

        # 返回更新后的配置对象
        return config
```