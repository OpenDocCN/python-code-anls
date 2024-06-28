# `.\models\musicgen\modeling_musicgen.py`

```
# 设置编码格式为UTF-8，确保脚本中的中文等字符能正确处理
# 版权声明和许可条款，告知使用者如何合法使用代码
# 导入所需模块和类
# 从dataclasses模块导入dataclass装饰器，用于定义数据类
# 从typing模块导入类型检查相关的工具
# 导入PyTorch库
# 从torch.nn模块导入神经网络相关的类和函数
# 从...activations模块导入ACT2FN，用于激活函数映射
# 从...generation.configuration_utils模块导入GenerationConfig，生成配置类
# 从...generation.logits_process模块导入分类器无指导的logits处理器和logits处理器列表
# 从...generation.stopping_criteria模块导入停止标准列表
# 从...modeling_attn_mask_utils模块导入准备4D注意力掩码的工具函数
# 从...modeling_outputs模块导入各种模型输出类
# 从...modeling_utils模块导入预训练模型基类PreTrainedModel
# 从...utils模块导入各种实用函数和工具类
# 如果是类型检查阶段，则从...generation.streamers模块导入BaseStreamer类
# 获取日志记录器，用于在运行时记录消息和警告
    # 定义函数的参数和类型注释
    Args:
        encoder_outputs (`Tuple[torch.FloatTensor]` of length 1, with tensor shape `(batch_size, sequence_length, hidden_size)`):
            文本编码器模型最后一层的隐藏状态序列。
        attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            编码器注意力掩码，用于避免对填充的标记索引执行注意力操作。掩码值为 `[0, 1]`：1 表示**未被掩码**的标记，0 表示**被掩码**的标记。
        guidance_scale (`float`, *optional*):
            分类器自由引导的指导比例，用于设置条件对数（从提示预测的）与无条件对数（没有提示预测的）之间的平衡。
    """
    
    # 初始化函数的参数，设置默认值为 None
    encoder_outputs: Tuple[torch.FloatTensor] = None
    attention_mask: torch.LongTensor = None
    guidance_scale: float = None
# Copied from transformers.models.encoder_decoder.modeling_encoder_decoder.shift_tokens_right
def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    # 创建一个与输入形状相同的全零张量，用于存储右移后的输入ids
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    # 将原始输入ids的除了第一个token外的所有token复制到右移后的张量中
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    if decoder_start_token_id is None:
        raise ValueError("Make sure to set the decoder_start_token_id attribute of the model's configuration.")
    # 将decoder的起始token id放到右移后的张量的第一个位置
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("Make sure to set the pad_token_id attribute of the model's configuration.")
    # 将右移后的张量中可能存在的-100值替换为pad_token_id
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids


class MusicgenSinusoidalPositionalEmbedding(nn.Module):
    """This module produces sinusoidal positional embeddings of any length."""

    def __init__(self, num_positions: int, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        # 调用make_weights方法初始化权重
        self.make_weights(num_positions, embedding_dim)

    def make_weights(self, num_embeddings: int, embedding_dim: int):
        # 调用get_embedding方法生成sinusoidal位置编码的权重
        emb_weights = self.get_embedding(num_embeddings, embedding_dim)
        if hasattr(self, "weights"):
            # 在前向传播时将权重调整为参数的正确dtype和device
            emb_weights = emb_weights.to(dtype=self.weights.dtype, device=self.weights.device)

        self.weights = nn.Parameter(emb_weights)
        self.weights.requires_grad = False
        self.weights.detach_()

    @staticmethod
    def get_embedding(num_embeddings: int, embedding_dim: int):
        """
        Build sinusoidal embeddings. This matches the implementation in tensor2tensor, but differs slightly from the
        description in Section 3.5 of "Attention Is All You Need".
        """
        # 计算sinusoidal位置编码的半维度
        half_dim = embedding_dim // 2
        # 计算emb参数
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.int64).float() * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.int64).float().unsqueeze(1) * emb.unsqueeze(0)
        # 构建sinusoidal位置编码张量，按照cos和sin的方式组合
        emb = torch.cat([torch.cos(emb), torch.sin(emb)], dim=1).view(num_embeddings, -1)
        if embedding_dim % 2 == 1:
            # 如果embedding_dim为奇数，进行零填充
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        return emb.to(torch.get_default_dtype())

    @torch.no_grad()
    # 定义一个前向传播方法，接收输入的 token ids 和过去键值长度作为参数
    def forward(self, input_ids: torch.Tensor, past_key_values_length: int = 0):
        # 获取输入 tensor 的批大小（batch size），代码簿数量（codebooks），以及序列长度（seq_len）
        bsz, codebooks, seq_len = input_ids.size()
        
        # 从输入的 token ids 创建位置 ids
        # 使用 torch.arange 生成长度为 seq_len 的序列，并加上 past_key_values_length 以处理位置偏移
        position_ids = (torch.arange(seq_len) + past_key_values_length).to(input_ids.device)
        
        # 如果序列长度大于当前权重张量的大小，则扩展权重张量
        if seq_len > self.weights.size(0):
            self.make_weights(seq_len + self.offset, self.embedding_dim)
        
        # 根据位置 ids 从权重张量中选择对应的权重，并分离（detach）出来
        return self.weights.index_select(0, position_ids.view(-1)).detach()
# 从transformers.models.bart.modeling_bart.BartAttention复制并修改为MusicgenAttention
class MusicgenAttention(nn.Module):
    """来自论文'Attention Is All You Need'的多头注意力机制"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
        is_causal: bool = False,
        config: Optional[MusicgenConfig] = None,
    ):
        super().__init__()
        # 初始化模型参数
        self.embed_dim = embed_dim  # 嵌入维度
        self.num_heads = num_heads  # 注意力头数
        self.dropout = dropout  # dropout概率
        self.head_dim = embed_dim // num_heads  # 每个注意力头的维度
        self.config = config  # 配置对象

        # 检查embed_dim必须能被num_heads整除
        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim必须能被num_heads整除 (得到 `embed_dim`: {self.embed_dim}"
                f" 和 `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5  # 缩放因子
        self.is_decoder = is_decoder  # 是否为解码器注意力
        self.is_causal = is_causal  # 是否为因果注意力

        # 线性变换层定义
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)  # K矩阵投影
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)  # V矩阵投影
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)  # Q矩阵投影
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)  # 输出投影层

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        # 重新整形注意力张量，调整维度顺序以便多头注意力计算
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ):
        # 前向传播函数，实现注意力计算
        # hidden_states: 输入的隐藏状态张量
        # key_value_states: 键值状态张量（可选）
        # past_key_value: 过去的键值状态元组（可选）
        # attention_mask: 注意力掩码（可选）
        # layer_head_mask: 层头掩码（可选）
        # output_attentions: 是否输出注意力权重（布尔值）

        # 进行自注意力计算
        # 1. 计算查询、键、值的投影
        query = self.q_proj(hidden_states)
        key = self.k_proj(key_value_states if key_value_states is not None else hidden_states)
        value = self.v_proj(key_value_states if key_value_states is not None else hidden_states)

        # 2. 重塑张量以便并行计算多头注意力
        query = self._shape(query, query.size(1), query.size(0))
        key = self._shape(key, key.size(1), key.size(0))
        value = self._shape(value, value.size(1), value.size(0))

        # 3. 计算注意力分数及归一化
        attn_weights = torch.matmul(query, key.transpose(-1, -2))
        attn_weights *= self.scaling
        if attention_mask is not None:
            attn_weights += attention_mask

        attn_probs = nn.functional.softmax(attn_weights, dim=-1)
        attn_probs = nn.functional.dropout(attn_probs, p=self.dropout, training=self.training)

        # 4. 使用注意力权重计算加权和
        attn_output = torch.matmul(attn_probs, value)

        # 5. 将多头注意力结果重塑回原始形状
        attn_output = attn_output.transpose(1, 2).contiguous().view(attn_output.size(0), attn_output.size(2), -1)

        # 6. 执行最终的线性变换并返回结果
        attn_output = self.out_proj(attn_output)
        return attn_output
    # Copied from transformers.models.mbart.modeling_mbart.MBartDecoderLayer.forward
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
        use_cache: Optional[bool] = True,
class MusicgenPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 使用 MusicgenDecoderConfig 作为配置类
    config_class = MusicgenDecoderConfig
    # 模型权重前缀
    base_model_prefix = "model"
    # 支持梯度检查点
    supports_gradient_checkpointing = True
    # 不需要拆分的模块列表
    _no_split_modules = ["MusicgenDecoderLayer", "MusicgenAttention"]

    def _init_weights(self, module):
        # 从配置中获取初始化因子
        std = self.config.initializer_factor
        # 如果是线性层或者卷积层
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            # 初始化权重为正态分布
            module.weight.data.normal_(mean=0.0, std=std)
            # 如果有偏置项，初始化为零
            if module.bias is not None:
                module.bias.data.zero_()
        # 如果是嵌入层
        elif isinstance(module, nn.Embedding):
            # 初始化权重为正态分布
            module.weight.data.normal_(mean=0.0, std=std)
            # 如果有填充索引，将填充索引位置的权重初始化为零
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


MUSICGEN_START_DOCSTRING = r"""

    The Musicgen model was proposed in [Simple and Controllable Music Generation](https://arxiv.org/abs/2306.05284) by
    Jade Copet, Felix Kreuk, Itai Gat, Tal Remez, David Kant, Gabriel Synnaeve, Yossi Adi, Alexandre Défossez. It is an
    encoder decoder transformer trained on the task of conditional music generation

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`MusicgenConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

MUSICGEN_INPUTS_DOCSTRING = r"""
"""

MUSICGEN_DECODER_INPUTS_DOCSTRING = r"""
"""


class MusicgenDecoder(MusicgenPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`MusicgenDecoderLayer`]
    """
    # 初始化函数，接受一个MusicgenDecoderConfig对象作为配置参数
    def __init__(self, config: MusicgenDecoderConfig):
        # 调用父类的初始化函数
        super().__init__(config)
        # 设置dropout率
        self.dropout = config.dropout
        # 设置层级dropout率
        self.layerdrop = config.layerdrop
        # 设置最大目标位置数
        self.max_target_positions = config.max_position_embeddings
        # 设置模型的隐藏层大小
        self.d_model = config.hidden_size
        # 设置代码本数目
        self.num_codebooks = config.num_codebooks
        # 设置嵌入缩放比例，如果配置中开启了嵌入缩放则为隐藏层大小的平方根，否则为1.0
        self.embed_scale = math.sqrt(config.hidden_size) if config.scale_embedding else 1.0

        # 设置嵌入维度为词汇表大小加1（用于特殊符号），创建一个嵌入模块列表
        embed_dim = config.vocab_size + 1
        self.embed_tokens = nn.ModuleList(
            [nn.Embedding(embed_dim, config.hidden_size) for _ in range(config.num_codebooks)]
        )

        # 设置位置嵌入对象，使用正弦函数生成的位置嵌入
        self.embed_positions = MusicgenSinusoidalPositionalEmbedding(
            config.max_position_embeddings,
            config.hidden_size,
        )

        # 创建解码器层列表，包含config.num_hidden_layers个MusicgenDecoderLayer对象
        self.layers = nn.ModuleList([MusicgenDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        
        # 创建层级归一化层，使用隐藏层大小进行归一化
        self.layer_norm = nn.LayerNorm(config.hidden_size)

        # 初始化梯度检查点标志为False
        self.gradient_checkpointing = False
        
        # 执行额外的初始化步骤，包括权重初始化和最终处理
        self.post_init()

    # 获取输入嵌入模块列表
    def get_input_embeddings(self):
        return self.embed_tokens

    # 设置输入嵌入模块列表的值
    def set_input_embeddings(self, value):
        self.embed_tokens = value

    # 前向传播函数，接受多个输入参数并返回相应的输出
    @add_start_docstrings_to_model_forward(MUSICGEN_DECODER_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
# 在 MusicgenModel 类之上添加文档字符串，描述这是一个 Musicgen 解码器模型，输出原始隐藏状态，没有特定的顶层头部。
# MUSICGEN_START_DOCSTRING 是一个预定义的文档字符串常量，用于提供更详细的模型描述信息。
@add_start_docstrings(
    "The bare Musicgen decoder model outputting raw hidden-states without any specific head on top.",
    MUSICGEN_START_DOCSTRING,
)
# MusicgenModel 类，继承自 MusicgenPreTrainedModel 类。
class MusicgenModel(MusicgenPreTrainedModel):
    def __init__(self, config: MusicgenDecoderConfig):
        # 调用父类的初始化方法，传入配置对象 config。
        super().__init__(config)
        # 创建一个 MusicgenDecoder 对象并赋值给 self.decoder。
        self.decoder = MusicgenDecoder(config)
        # 初始化权重并应用最终处理（这里可能包括一些额外的初始化操作或配置参数）。
        self.post_init()

    # 获取输入嵌入的方法，返回解码器对象中的嵌入 tokens。
    def get_input_embeddings(self):
        return self.decoder.embed_tokens

    # 设置输入嵌入的方法，将输入的嵌入值赋给解码器对象的 embed_tokens 属性。
    def set_input_embeddings(self, value):
        self.decoder.embed_tokens = value

    # 获取解码器对象的方法，返回 self.decoder。
    def get_decoder(self):
        return self.decoder

    # 在 forward 方法上添加模型前向传播的文档字符串，这里使用了 MUSICGEN_DECODER_INPUTS_DOCSTRING。
    @add_start_docstrings_to_model_forward(MUSICGEN_DECODER_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
        # 如果没有显式指定，使用默认的输出注意力机制
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 如果没有显式指定，使用默认的输出隐藏状态
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 如果没有显式指定，使用默认的缓存策略
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        # 如果没有显式指定，使用默认的返回字典设置
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        # 调用解码器模型，返回解码器的输出结果
        decoder_outputs = self.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_attention_mask=encoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            head_mask=head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 如果没有设置返回字典，则直接返回解码器输出
        if not return_dict:
            return decoder_outputs

        # 如果设置了返回字典，则构建包含过去键值和交叉注意力的输出对象
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            hidden_states=decoder_outputs.hidden_states,
            attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
        )
# 使用装饰器为类添加文档字符串，描述此类为带有语言建模头的 MusicGen 解码器模型
@add_start_docstrings(
    "The MusicGen decoder model with a language modelling head on top.",
    MUSICGEN_START_DOCSTRING,
)
class MusicgenForCausalLM(MusicgenPreTrainedModel):
    def __init__(self, config: MusicgenDecoderConfig):
        # 调用父类构造函数初始化配置
        super().__init__(config)

        # 创建基础的 MusicgenModel 模型
        self.model = MusicgenModel(config)

        # 设置编码本数量和语言模型头列表
        self.num_codebooks = config.num_codebooks
        self.lm_heads = nn.ModuleList(
            [nn.Linear(config.hidden_size, config.vocab_size, bias=False) for _ in range(config.num_codebooks)]
        )

        # 初始化权重并应用最终处理
        self.post_init()

    def get_input_embeddings(self):
        # 返回解码器的嵌入层
        return self.model.decoder.embed_tokens

    def set_input_embeddings(self, value):
        # 设置解码器的嵌入层
        self.model.decoder.embed_tokens = value

    def get_output_embeddings(self):
        # 返回语言模型头列表
        return self.lm_heads

    def set_output_embeddings(self, new_embeddings):
        # 设置新的语言模型头列表
        self.lm_heads = new_embeddings

    def set_decoder(self, decoder):
        # 设置解码器
        self.model.decoder = decoder

    def get_decoder(self):
        # 获取解码器
        return self.model.decoder

    @add_start_docstrings_to_model_forward(MUSICGEN_DECODER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        # 函数签名，定义了该解码器模型的前向传播方法，支持多种输入参数和可选的输出控制标志
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
                Returns:
        """

        # 根据 return_dict 参数决定是否返回字典形式的结果
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 将模型的输入传递给模型，并获取模型的输出
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            head_mask=head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 从模型输出中获取隐藏状态（hidden_states）
        hidden_states = outputs[0]

        # 使用 lm_heads 对隐藏状态进行预测，得到语言模型的 logits
        lm_logits = torch.stack([head(hidden_states) for head in self.lm_heads], dim=1)

        # 初始化损失值为 None
        loss = None
        # 如果存在 labels，则抛出未实现错误，因为 Musicgen 的训练尚未实现
        if labels is not None:
            raise NotImplementedError("Training is not implemented for Musicgen.")

        # 重新组织 lm_logits 的形状以适应后续处理
        # (bsz, num_codebooks, seq_len, vocab_size) -> (bsz * num_codebooks, seq_len, vocab_size)
        lm_logits = lm_logits.reshape(-1, *lm_logits.shape[2:])

        # 如果不要求返回字典形式的输出，则将 lm_logits 与其他输出组合成 tuple 返回
        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        # 如果要求返回字典形式的输出，则构造 CausalLMOutputWithCrossAttentions 对象返回
        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        use_cache=True,
        delay_pattern_mask=None,
        guidance_scale=None,
        **kwargs,
    ):
        # 如果延迟模式掩码为 None，则构建一个延迟模式掩码
        if delay_pattern_mask is None:
            input_ids, delay_pattern_mask = self.build_delay_pattern_mask(
                input_ids,
                pad_token_id=self.generation_config.pad_token_id,
                max_length=self.generation_config.max_length,
            )

        # 应用延迟模式掩码到输入的 token IDs
        input_ids = self.apply_delay_pattern_mask(input_ids, delay_pattern_mask)

        # 如果有指导比例且大于 1，则为无分类器指导复制解码器参数到批次维度（在采样前将其拆分）
        if guidance_scale is not None and guidance_scale > 1:
            input_ids = input_ids.repeat((2, 1))
            # 如果存在注意力掩码，则将其在批次维度上重复
            if attention_mask is not None:
                attention_mask = attention_mask.repeat((2, 1))

        # 如果过去的键值不为 None，则仅保留最后一个 token ID
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]

        # 返回生成方法的参数字典
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "encoder_hidden_states": encoder_hidden_states,
            "encoder_attention_mask": encoder_attention_mask,
            "head_mask": head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
        }

    @staticmethod
    def apply_delay_pattern_mask(input_ids, decoder_pad_token_mask):
        """Apply a delay pattern mask to the decoder input ids, only preserving predictions where
        the mask is set to -1, and otherwise setting to the value detailed in the mask."""
        # 获取输入 token IDs 的序列长度
        seq_len = input_ids.shape[-1]
        # 将解码器的 pad token 掩码裁剪到与序列长度相匹配的维度
        decoder_pad_token_mask = decoder_pad_token_mask[..., :seq_len]
        # 根据解码器 pad token 掩码，保留掩码值为 -1 的预测，其余设置为掩码中的详细值
        input_ids = torch.where(decoder_pad_token_mask == -1, input_ids, decoder_pad_token_mask)
        return input_ids

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        synced_gpus: Optional[bool] = None,
        streamer: Optional["BaseStreamer"] = None,
        **kwargs,
@add_start_docstrings(
    "The composite MusicGen model with a text encoder, audio encoder and Musicgen decoder, "
    "for music generation tasks with one or both of text and audio prompts.",
    MUSICGEN_START_DOCSTRING,
)
class MusicgenForConditionalGeneration(PreTrainedModel):
    # 指定配置类为MusicgenConfig
    config_class = MusicgenConfig
    # 指定基础模型前缀为"encoder_decoder"
    base_model_prefix = "encoder_decoder"
    # 主要输入名称为"input_ids"
    main_input_name = "input_ids"
    # 支持梯度检查点
    supports_gradient_checkpointing = True

    def __init__(
        self,
        config: Optional[MusicgenConfig] = None,
        text_encoder: Optional[PreTrainedModel] = None,
        audio_encoder: Optional[PreTrainedModel] = None,
        decoder: Optional[MusicgenForCausalLM] = None,
    ):
        # 构造函数，初始化函数，接受MusicgenConfig配置，文本编码器、音频编码器和解码器作为参数

    def tie_weights(self):
        # 绑定权重函数，用于可能需要绑定文本编码器和解码器的情况
        if self.config.tie_encoder_decoder:
            # 如果配置要求绑定文本编码器和解码器，则执行以下操作
            decoder_base_model_prefix = self.decoder.base_model_prefix
            self._tie_encoder_decoder_weights(
                self.text_encoder, self.decoder._modules[decoder_base_model_prefix], self.decoder.base_model_prefix
            )

    def get_audio_encoder(self):
        # 返回音频编码器
        return self.audio_encoder

    def get_text_encoder(self):
        # 返回文本编码器
        return self.text_encoder

    def get_encoder(self):
        # 获取文本编码器以计算生成时的编码器隐藏状态
        return self.get_text_encoder()

    def get_decoder(self):
        # 返回解码器
        return self.decoder

    def get_input_embeddings(self):
        # 返回文本编码器的输入嵌入
        return self.text_encoder.get_input_embeddings()

    def get_output_embeddings(self):
        # 返回解码器的输出嵌入
        return self.decoder.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        # 设置解码器的输出嵌入
        return self.decoder.set_output_embeddings(new_embeddings)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        r"""
        Example:

        ```python
        >>> from transformers import MusicgenForConditionalGeneration

        >>> model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")
        ```"""

        # 目前不支持快速初始化复合模型
        if kwargs.get("_fast_init", False):
            logger.warning(
                "Fast initialization is currently not supported for MusicgenForConditionalGeneration. "
                "Falling back to slow initialization..."
            )
        kwargs["_fast_init"] = False

        return super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)

    @classmethod
    def from_sub_models_pretrained(
        cls,
        text_encoder_pretrained_model_name_or_path: str = None,
        audio_encoder_pretrained_model_name_or_path: str = None,
        decoder_pretrained_model_name_or_path: str = None,
        *model_args,
        **kwargs,
    ):
        # 从预训练子模型加载复合模型的类方法

    @add_start_docstrings_to_model_forward(MUSICGEN_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    # 定义一个方法用于模型的前向传播
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,  # 输入的 token IDs，类型为可选的长整型张量
        attention_mask: Optional[torch.BoolTensor] = None,  # 注意力遮罩，类型为可选的布尔张量
        input_values: Optional[torch.FloatTensor] = None,  # 输入的值，类型为可选的浮点张量
        padding_mask: Optional[torch.BoolTensor] = None,  # 填充遮罩，类型为可选的布尔张量
        decoder_input_ids: Optional[torch.LongTensor] = None,  # 解码器输入的 token IDs，类型为可选的长整型张量
        decoder_attention_mask: Optional[torch.BoolTensor] = None,  # 解码器注意力遮罩，类型为可选的布尔张量
        encoder_outputs: Optional[Tuple[torch.FloatTensor]] = None,  # 编码器输出，类型为可选的浮点张量元组
        past_key_values: Tuple[Tuple[torch.FloatTensor]] = None,  # 过去的键值，类型为元组的元组，包含浮点张量
        inputs_embeds: Optional[torch.FloatTensor] = None,  # 输入的嵌入，类型为可选的浮点张量
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,  # 解码器输入的嵌入，类型为可选的浮点张量
        labels: Optional[torch.LongTensor] = None,  # 标签，类型为可选的长整型张量
        use_cache: Optional[bool] = None,  # 是否使用缓存，类型为可选的布尔值
        output_attentions: Optional[bool] = None,  # 是否输出注意力权重，类型为可选的布尔值
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态，类型为可选的布尔值
        return_dict: Optional[bool] = None,  # 是否返回字典，类型为可选的布尔值
        **kwargs,  # 其他关键字参数，包括所有未列出的参数
    ):
        pass  # 这里是方法的占位符，未实现具体的功能逻辑

    # 定义一个方法用于为生成准备输入
    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,  # 解码器输入的 token IDs，必填参数
        past_key_values=None,  # 过去的键值，类型为可选的默认空值
        attention_mask=None,  # 注意力遮罩，类型为可选的默认空值
        head_mask=None,  # 头部遮罩，类型为可选的默认空值
        decoder_attention_mask=None,  # 解码器注意力遮罩，类型为可选的默认空值
        decoder_head_mask=None,  # 解码器头部遮罩，类型为可选的默认空值
        cross_attn_head_mask=None,  # 交叉注意力头部遮罩，类型为可选的默认空值
        use_cache=None,  # 是否使用缓存，类型为可选的默认空值
        encoder_outputs=None,  # 编码器输出，类型为可选的默认空值
        decoder_delay_pattern_mask=None,  # 解码器延迟模式遮罩，类型为可选的默认空值
        guidance_scale=None,  # 引导比例，类型为可选的默认空值
        **kwargs,  # 其他关键字参数，包括所有未列出的参数
    ):
        pass  # 这里是方法的占位符，未实现具体的功能逻辑
    ):
        # 如果没有提供解码器延迟模式掩码，则从解码器构建一个
        if decoder_delay_pattern_mask is None:
            decoder_input_ids, decoder_delay_pattern_mask = self.decoder.build_delay_pattern_mask(
                decoder_input_ids,
                self.generation_config.pad_token_id,
                max_length=self.generation_config.max_length,
            )

        # 应用延迟模式掩码到解码器输入IDs
        decoder_input_ids = self.decoder.apply_delay_pattern_mask(decoder_input_ids, decoder_delay_pattern_mask)

        # 如果给定了guidance_scale并且大于1，则进行以下操作
        if guidance_scale is not None and guidance_scale > 1:
            # 对于无分类器引导，需要在批次维度上复制解码器参数（在采样之前将其拆分）
            decoder_input_ids = decoder_input_ids.repeat((2, 1))
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.repeat((2, 1))

        # 如果给定了过去的键值，则执行以下操作
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]

            # 一些生成方法已经仅传递最后一个输入ID
            if decoder_input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # 默认使用旧的行为：仅保留最后一个ID
                remove_prefix_length = decoder_input_ids.shape[1] - 1

            # 从解码器输入IDs中去除前缀长度
            decoder_input_ids = decoder_input_ids[:, remove_prefix_length:]

        # 返回准备好的字典，包含用于生成的各种输入和掩码
        return {
            "input_ids": None,  # encoder_outputs已定义，不需要input_ids
            "encoder_outputs": encoder_outputs,
            "past_key_values": past_key_values,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "decoder_attention_mask": decoder_attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
        }

    def _prepare_decoder_input_ids_for_generation(
        self,
        batch_size: int,
        model_input_name: str,
        model_kwargs: Dict[str, torch.Tensor],
        decoder_start_token_id: int = None,
        bos_token_id: int = None,
        device: torch.device = None,
    ) -> Tuple[torch.LongTensor, Dict[str, torch.Tensor]]:
        """Prepares `decoder_input_ids` for generation with encoder-decoder models"""

        # 1. Check whether the user has defined `decoder_input_ids` manually. To facilitate in terms of input naming,
        # we also allow the user to pass it under `input_ids`, if the encoder does not use it as the main input.
        if model_kwargs is not None and "decoder_input_ids" in model_kwargs:
            # Retrieve `decoder_input_ids` from `model_kwargs` and remove it from the dictionary
            decoder_input_ids = model_kwargs.pop("decoder_input_ids")
        elif "input_ids" in model_kwargs and model_input_name != "input_ids":
            # If `input_ids` is found in `model_kwargs` and it's not the main input name, assign it to `decoder_input_ids`
            decoder_input_ids = model_kwargs.pop("input_ids")
        else:
            # If neither `decoder_input_ids` nor `input_ids` are provided, initialize `decoder_input_ids` as None
            decoder_input_ids = None

        # 2. Encoder-decoder models expect the `decoder_input_ids` to start with a special token. Let's ensure that.
        # Get the special token ID to start `decoder_input_ids` sequence
        decoder_start_token_id = self._get_decoder_start_token_id(decoder_start_token_id, bos_token_id)
        if device is None:
            device = self.device
        # Create a tensor to initialize `decoder_input_ids` starting with `decoder_start_token_id`
        decoder_input_ids_start = (
            torch.ones((batch_size * self.decoder.num_codebooks, 1), dtype=torch.long, device=device)
            * decoder_start_token_id
        )

        # If no `decoder_input_ids` provided by the user, use `decoder_input_ids_start`
        if decoder_input_ids is None:
            decoder_input_ids = decoder_input_ids_start

        # If user-provided `decoder_input_ids` does not start with `decoder_start_token_id`, prepend it
        elif (decoder_input_ids[..., 0] != decoder_start_token_id).all().item():
            decoder_input_ids = torch.cat([decoder_input_ids_start, decoder_input_ids], dim=-1)
            # Adjust `decoder_attention_mask` if provided along with `decoder_input_ids`
            if "decoder_attention_mask" in model_kwargs:
                decoder_attention_mask = model_kwargs["decoder_attention_mask"]
                decoder_attention_mask = torch.cat(
                    (torch.ones_like(decoder_attention_mask)[:, :1], decoder_attention_mask),
                    dim=-1,
                )
                model_kwargs["decoder_attention_mask"] = decoder_attention_mask

        return decoder_input_ids, model_kwargs
    ) -> Dict[str, Any]:
        # 1. 获取文本编码器
        encoder = self.get_text_encoder()
        
        # 2. 准备编码器参数和编码器关键字参数，从模型关键字参数中获取
        irrelevant_prefix = ["decoder_", "cross_attn", "use_cache"]
        encoder_kwargs = {
            argument: value
            for argument, value in model_kwargs.items()
            if not any(argument.startswith(p) for p in irrelevant_prefix)
        }
        
        # 检查编码器的参数签名
        encoder_signature = set(inspect.signature(encoder.forward).parameters)
        encoder_accepts_wildcard = "kwargs" in encoder_signature or "model_kwargs" in encoder_signature
        
        # 如果编码器不接受通配符参数，则过滤掉不在签名内的参数
        if not encoder_accepts_wildcard:
            encoder_kwargs = {
                argument: value for argument, value in encoder_kwargs.items() if argument in encoder_signature
            }

        # 3. 确保编码器返回 `ModelOutput`
        model_input_name = model_input_name if model_input_name is not None else self.text_encoder.main_input_name
        encoder_kwargs["return_dict"] = True
        encoder_kwargs[model_input_name] = inputs_tensor
        
        # 调用编码器的 forward 方法获取最后隐藏状态
        last_hidden_state = encoder(**encoder_kwargs).last_hidden_state

        # 如果有指导比例并且大于1，则添加一个“null”输入到编码器隐藏状态中
        if guidance_scale is not None and guidance_scale > 1:
            last_hidden_state = torch.concatenate([last_hidden_state, torch.zeros_like(last_hidden_state)], dim=0)
            if "attention_mask" in model_kwargs:
                model_kwargs["attention_mask"] = torch.concatenate(
                    [model_kwargs["attention_mask"], torch.zeros_like(model_kwargs["attention_mask"])], dim=0
                )

        # 将编码器的输出设置为模型基本输出的最后隐藏状态
        model_kwargs["encoder_outputs"] = BaseModelOutput(last_hidden_state=last_hidden_state)

        return model_kwargs

    def _prepare_audio_encoder_kwargs_for_generation(
        self, input_values, model_kwargs, model_input_name: Optional[str] = None
    ):
        raise NotImplementedError("This method is not implemented yet.")

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        # 根据标签准备解码器的输入 ID，将标签向右移动一位
        return shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)

    def resize_token_embeddings(self, *args, **kwargs):
        # 抛出未实现错误，不能直接通过 EncoderDecoderModel 调整嵌入层大小
        raise NotImplementedError(
            "Resizing the embedding layers via the EncoderDecoderModel directly is not supported. Please use the"
            " respective methods of the wrapped objects (model.encoder.resize_token_embeddings(...) or"
            " model.decoder.resize_token_embeddings(...))"
        )

    def _maybe_initialize_input_ids_for_generation(
        self,
        inputs: Optional[torch.Tensor] = None,
        bos_token_id: Optional[int] = None,
        model_kwargs: Optional[Dict[str, torch.Tensor]] = None,
    ):
        # 这个方法可能用于为生成初始化输入 ID，具体功能暂时不明确，需要进一步分析上下文来理解其作用。
        pass
    ) -> torch.LongTensor:
        """Initializes input ids for generation, if necessary."""
        # 如果已经提供了输入，则直接返回这些输入
        if inputs is not None:
            return inputs

        # 检查是否在 `model_kwargs` 中存在 `encoder_outputs`
        encoder_outputs = model_kwargs.get("encoder_outputs")
        if encoder_outputs is not None:
            # 创建一个具有 `-100` 值的虚拟 input_ids，用作健全性检查，确保它们不会用于编码
            shape = encoder_outputs[0].size()[:-1]
            return torch.ones(shape, dtype=torch.long, device=self.device) * -100

        # 如果未提供 `input_ids` 但未定义 `bos_token_id`，则抛出错误
        if bos_token_id is None:
            raise ValueError("`bos_token_id` has to be defined when no `input_ids` are provided.")

        # 如果 `model_kwargs` 中存在某些张量，则可以从中推断出批量大小。这在软提示或基于解码器的多模态实现中特别有用。
        batch_size = 1
        for value in model_kwargs.values():
            if isinstance(value, torch.Tensor):
                batch_size = value.shape[0]
                break
        # 创建一个形状为 (batch_size, 1) 的张量，填充值为 bos_token_id，并使用设备 self.device
        return torch.ones((batch_size, 1), dtype=torch.long, device=self.device) * bos_token_id
    # 定义一个方法，用于获取无条件生成的输入，以便在没有特征提取器或分词器的情况下使用模型。
    def get_unconditional_inputs(self, num_samples=1):
        """
        Helper function to get null inputs for unconditional generation, enabling the model to be used without the
        feature extractor or tokenizer.
    
        Args:
            num_samples (int, *optional*):
                Number of audio samples to unconditionally generate.
            max_new_tokens (int, *optional*):
                Number of tokens to generate for each sample. More tokens means longer audio samples, at the expense of
                longer inference (since more audio tokens need to be generated per sample).
    
        Example:
        ```python
        >>> from transformers import MusicgenForConditionalGeneration
    
        >>> model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")
    
        >>> # get the unconditional (or 'null') inputs for the model
        >>> unconditional_inputs = model.get_unconditional_inputs(num_samples=1)
        >>> audio_samples = model.generate(**unconditional_inputs, max_new_tokens=256)
        ```"""
        # 创建一个全零张量，用于存储模型的隐藏状态输出，形状为 (num_samples, 1, hidden_size)
        last_hidden_state = torch.zeros(
            (num_samples, 1, self.config.text_encoder.hidden_size), device=self.device, dtype=self.dtype
        )
    
        # 创建一个全零张量作为注意力掩码，形状为 (num_samples, 1)，用于指示哪些位置需要注意力
        attention_mask = torch.zeros((num_samples, 1), device=self.device, dtype=torch.long)
    
        # 返回一个包含无条件生成所需输入的 MusicgenUnconditionalInput 对象
        return MusicgenUnconditionalInput(
            encoder_outputs=(last_hidden_state,),  # 编码器输出，包含隐藏状态
            attention_mask=attention_mask,          # 注意力掩码，全零表示不区分注意力
            guidance_scale=1.0,                     # 指导尺度，通常设置为1.0
        )
```