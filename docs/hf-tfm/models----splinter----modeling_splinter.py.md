# `.\transformers\models\splinter\modeling_splinter.py`

```
# 设置 Python 文件编码格式为 UTF-8
# 版权声明
# 实体类，表示 Splinter 模型的配置信息
# 对于导入的模块和类进行解释
# 导入数学模块
# 导入 dataclass 模块
# 导入 List、Optional、Tuple、Union 模块
# 导入 torch 模块
# 导入 nn（神经网络）模块
# 导入 CrossEntropyLoss 损失函数
# 导入相关模块和类
# 导入辅助工具方法
# 导入配置类
# 获取日志记录器
# 文档中的检查点示例
# 文档中的配置示例
# 预训练模型的存档列表
# SplinterEmbeddings 类，用于构建来自单词、位置和令牌类型嵌入的嵌入
class SplinterEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    # 初始化函数
    def __init__(self, config):
        # 调用父类的初始化函数
        super().__init__()
        # 初始化一个单词嵌入层，用于将单词映射到隐藏层大小
        # 设定padding_idx参数，表示填充标记（如[CLS]、[SEP]等）的索引位置
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        # 初始化一个位置嵌入层，用于将位置信息映射到隐藏层大小
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        # 初始化一个令牌嵌入层，用于将令牌类型映射到隐藏层大小
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # LayerNorm 层，用于对隐藏层进行归一化处理
        # eps 参数表示 LayerNorm 的 epsilon 值
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # Dropout 层，用于在训练过程中对隐藏层的部分神经元进行随机置零
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # position_ids 张量表示位置的索引信息
        # torch.arange 函数生成一维张量用于表示位置的索引
        # persistent=False 表示该 buffer 不会保存在 checkpoints 中
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )
        # 获取配置中的位置嵌入类型，默认为"absolute"
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
    # 定义一个前向传播函数，用于处理输入并生成嵌入表示
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,  # 输入的词语在词汇表中的索引
        token_type_ids: Optional[torch.LongTensor] = None,  # 标识输入中的句子类型
        position_ids: Optional[torch.LongTensor] = None,  # 标识输入中的词语位置
        inputs_embeds: Optional[torch.FloatTensor] = None,  # 输入的嵌入表示
        past_key_values_length: Optional[int] = 0,  # 过去的键值对的长度（用于注意力机制）
    ) -> Tuple:
        # 如果存在输入的词语索引，则获取输入的形状
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            # 否则，获取输入嵌入的形状（不包括最后一个维度，通常是批处理大小）
            input_shape = inputs_embeds.size()[:-1]

        # 获取序列长度
        seq_length = input_shape[1]

        # 如果位置标识不存在，则从位置嵌入中获取相应的位置标识
        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]

        # 如果标记类型标识不存在，则创建全零的标记类型标识
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        # 如果输入嵌入不存在，则通过词语索引获取输入嵌入
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        # 获取标记类型的嵌入表示
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        # 将词嵌入和标记类型嵌入相加以获得最终嵌入表示
        embeddings = inputs_embeds + token_type_embeddings
        
        # 如果位置嵌入的类型是“absolute”，则添加绝对位置嵌入
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        
        # 应用层归一化
        embeddings = self.LayerNorm(embeddings)
        # 应用 Dropout 层
        embeddings = self.dropout(embeddings)
        # 返回嵌入表示
        return embeddings
# 从 transformers.models.bert.modeling_bert.BertSelfAttention 复制代码，并将类名由BertSelfAttention改为SplinterSelfAttention
class SplinterSelfAttention(nn.Module):
    # 初始化函数，接收config和position_embedding_type两个参数
    def __init__(self, config, position_embedding_type=None):
        # 调用父类的初始化函数
        super().__init__()
        # 检查config.hidden_size是否能被config.num_attention_heads整除，若不能则抛出数值错误
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        # 初始化self.num_attention_heads, self.attention_head_size和self.all_head_size
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # 初始化self.query, self.key, self.value为线性层对象
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        # 初始化self.dropout为丢弃层对象
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        # 设置self.position_embedding_type为给定值或获取config中的值，若没有则设为"absolute"
        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )
        # 若self.position_embedding_type为"relative_key"或"relative_key_query"，则初始化self.max_position_embeddings和self.distance_embedding
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

        # 初始化self.is_decoder为config中的is_decoder值

    # 定义转置操作，将x变形成scores形式输出
    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    # 前向传播函数，接收多个输入参数，返回tensor类型对象
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
# 从 transformers.models.bert.modeling_bert.BertSelfOutput 复制代码，并将类名由BertSelfOutput改为SplinterSelfOutput
class SplinterSelfOutput(nn.Module):
    # 初始化函数，接收config参数
    def __init__(self, config):
        # 调用父类的初始化函数
        super().__init__()
        # 初始化self.dense为线性层对象
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 初始化self.LayerNorm为LayerNorm对象
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 初始化self.dropout为丢弃层对象

    # 前向传播函数，接收hidden_states和input_tensor两个输入参数，返回tensor类型对象
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 将hidden_states输入dense线性层
        hidden_states = self.dense(hidden_states)
        # 使用dropout对hidden_states进行丢弃操作
        hidden_states = self.dropout(hidden_states)
        # 将input_tensor添加到LayerNorm层对应hidden_states上，并返回结果
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states
# 从 transformers.models.bert.modeling_bert.BertAttention 复制代码并将 Bert->Splinter
class SplinterAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        # 初始化 SplinterSelfAttention 层
        self.self = SplinterSelfAttention(config, position_embedding_type=position_embedding_type)
        # 初始化 SplinterSelfOutput 层
        self.output = SplinterSelfOutput(config)
        # 用于记录已经剪枝的注意力头的集合
        self.pruned_heads = set()

    # 剪枝注意力头
    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        # 查找可剪枝的注意力头
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # 剪枝线性层
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # 更新超参数并记录剪枝的注意力头
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    # 前向传播
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        # 使用 SplinterSelfAttention 层进行自注意力计算
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        # 使用 SplinterSelfOutput 层处理自注意力输出
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # 如果有需要，添加注意力权重输出
        return outputs


# 从 transformers.models.bert.modeling_bert.BertIntermediate 复制代码并将 Bert->Splinter
class SplinterIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 初始化线性层
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        # 初始化激活函数
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    # 前向传播
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 线性变换
        hidden_states = self.dense(hidden_states)
        # 激活函数
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


# 从 transformers.models.bert.modeling_bert.BertOutput 复制代码并将 Bert->Splinter
class SplinterOutput(nn.Module):
```   
    # 初始化函数，接受一个配置参数
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()
        # 创建一个全连接层，输入维度为config.intermediate_size，输出维度为config.hidden_size
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        # 创建一个LayerNorm层，对隐藏状态进行标准化处理，设置epsilon为config.layer_norm_eps
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 创建一个Dropout层，设置隐藏状态的丢弃概率为config.hidden_dropout_prob
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
    
    # 前向传播函数，接受两个Tensor类型的参数，并返回一个Tensor类型的结果
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 将隐藏状态通过全连接层dense
        hidden_states = self.dense(hidden_states)
        # 对隐藏状态应用Dropout
        hidden_states = self.dropout(hidden_states)
        # 对隐藏状态应用LayerNorm并加上输入的Tensor
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        # 返回处理后的隐藏状态
        return hidden_states
# 从 transformers.models.bert.modeling_bert.BertLayer 复制代码，并将 Bert 改为 Splinter
class SplinterLayer(nn.Module):
    def __init__(self, config):
        # 初始化 SplinterLayer 类
        super().__init__()
        # 设置前馈网络的块大小
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        # 序列长度的维度，通常为 1
        self.seq_len_dim = 1
        # 初始化 SplinterAttention 类
        self.attention = SplinterAttention(config)
        # 判断是否为解码器
        self.is_decoder = config.is_decoder
        # 判断是否添加交叉注意力
        self.add_cross_attention = config.add_cross_attention
        # 如果添加交叉注意力
        if self.add_cross_attention:
            # 如果不是解码器，则抛出错误
            if not self.is_decoder:
                raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
            # 使用绝对位置编码类型初始化 SplinterAttention 类
            self.crossattention = SplinterAttention(config, position_embedding_type="absolute")
        # 初始化 SplinterIntermediate 类
        self.intermediate = SplinterIntermediate(config)
        # 初始化 SplinterOutput 类
        self.output = SplinterOutput(config)

    # 定义前向传播函数
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
        ) -> Tuple[torch.Tensor]:
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        # 如果前向传播携带了过往的key和value信息，则将其存储到self_attn_past_key_value变量中，否则为None
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        # 调用self.attention函数进行自注意力计算
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        # 获取自注意力计算的输出
        attention_output = self_attention_outputs[0]

        # 如果当前模块为decoder，则最后的输出为self-attention的缓存信息
        if self.is_decoder:
            # 从self_attention_outputs中提取除了第一个和最后两个元素之外的所有元素
            outputs = self_attention_outputs[1:-1]
            # 获取缓存的key和value信息
            present_key_value = self_attention_outputs[-1]
        else:
            outputs = self_attention_outputs[1:]  # 如果需要输出注意力权重，则加入自注意力的输出

        cross_attn_present_key_value = None
        # 如果是decoder并且encoder_hidden_states不为None
        if self.is_decoder and encoder_hidden_states is not None:
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers"
                    " by setting `config.add_cross_attention=True`"
                )

            # 提取crossattention中过往key和value信息的最后两个元素
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            # 使用crossattention模块进行交叉注意力计算
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                cross_attn_past_key_value,
                output_attentions,
            )
            # 获取交叉注意力的输出
            attention_output = cross_attention_outputs[0]
            # 将交叉注意力的输出加入到outputs中
            outputs = outputs + cross_attention_outputs[1:-1]  # 如果需要输出注意力权重，则加入交叉注意力的输出

            # 把cross-attn缓存信息加入进present_key_value中
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        # 使用分块的方式处理前馈神经网络计算
        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs

        # 如果是decoder模块，则将注意力的key和value信息作为最后的输出结果之一
        if self.is_decoder:
            outputs = outputs + (present_key_value,)

        return outputs

    def feed_forward_chunk(self, attention_output):
        # 进行前馈神经网络的计算
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output
# 从transformers.models.bert.modeling_bert.BertEncoder复制而来，将Bert改为Splinter
class SplinterEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # 创建一个由SplinterLayer组成的列表，列表长度为config.num_hidden_layers
        self.layer = nn.ModuleList([SplinterLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False  # 初始化梯度检查点为False

    # 前向传播函数
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    # 定义函数，返回类型为联合类型，可能为元组或 BaseModelOutputWithPastAndCrossAttentions 类型
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPastAndCrossAttentions]:
        # 如果输出隐藏状态，则初始化空元组；否则设为 None
        all_hidden_states = () if output_hidden_states else None
        # 如果输出注意力权重，则初始化空元组；否则设为 None
        all_self_attentions = () if output_attentions else None
        # 如果输出注意力权重且模型配置添加交叉注意力，则初始化空元组；否则设为 None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        # 如果启用渐变检查点并且处于训练模式
        if self.gradient_checkpointing and self.training:
            # 如果使用缓存，则警告并将 use_cache 设置为 False
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # 如果使用缓存，则初始化下一个解码器缓存为空元组；否则设为 None
        next_decoder_cache = () if use_cache else None
        # 遍历每个解码器层
        for i, layer_module in enumerate(self.layer):
            # 如果输出隐藏状态，则添加当前隐藏状态到 all_hidden_states 元组中
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 如果指定了头部掩码，则获取当前层的头部掩码；否则设为 None
            layer_head_mask = head_mask[i] if head_mask is not None else None
            # 如果提供了过去的键值，则获取当前层的过去键值；否则设为 None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            # 如果启用渐变检查点并且处于训练模式
            if self.gradient_checkpointing and self.training:
                # 使用渐变检查点函数进行前向传播
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )
            else:
                # 否则直接调用解码器层进行前向传播
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )

            # 获取当前层的隐藏状态
            hidden_states = layer_outputs[0]
            # 如果使用缓存，则将当前层的缓存添加到下一个解码器缓存中
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            # 如果输出注意力权重，则将当前层的注意力权重添加到 all_self_attentions 元组中
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                # 如果模型配置中添加了交叉注意力，则将当前层的交叉注意力权重添加到 all_cross_attentions 元组中
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        # 如果输出隐藏状态，则添加最后一个隐藏状态到 all_hidden_states 元组中
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 如果不返回字典格式的结果
        if not return_dict:
            # 返回不为 None 的元组元素
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        # 返回字典格式的结果
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )
class SplinterPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 指定配置类
    config_class = SplinterConfig
    # 指定基础模型前缀
    base_model_prefix = "splinter"
    # 指定是否支持梯度检查点
    supports_gradient_checkpointing = True

    # 从transformers.models.bert.modeling_bert.BertPreTrainedModel._init_weights复制而来的函数
    # 初始化模型权重
    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # 对线性层的权重进行初始化
            # 与TF版本略有不同，在这里使用正态分布进行初始化
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                # 如果有偏置，初始化为零
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            # 对嵌入层的权重进行初始化
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                # 如果有padding_idx，将对应位置的权重初始化为零
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            # 对层归一化层的权重进行初始化
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


SPLINTER_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`SplinterConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

SPLINTER_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `({0})`):
            # 定义输入序列中词汇的索引张量
            # 通过 AutoTokenizer 可以获得索引。参见 PreTrainedTokenizer.encode 和 PreTrainedTokenizer.__call__ 获取详情
            # 输入 IDs 是什么？参见 glossary#input-ids
        attention_mask (`torch.FloatTensor` of shape `{0}`, *optional*):
            # 遮盖填充标记索引的注意力掩码。掩码值在 [0, 1] 之间：

            # - 1 表示未被遮掩的标记，
            # - 0 表示被遮掩的标记。

            # 注意力掩码是什么？参见 glossary#attention-mask
        token_type_ids (`torch.LongTensor` of shape `{0}`, *optional*):
            # 指示输入的第一部分和第二部分的段标记索引。索引在 [0, 1] 中选择：

            # - 0 对应于 *sentence A* 的标记，
            # - 1 对应于 *sentence B* 的标记。

            # 什么是令牌类型 IDs？参见 glossary#token-type-ids
        position_ids (`torch.LongTensor` of shape `{0}`, *optional*):
            # 输入序列令牌在位置嵌入中的位置索引。在范围 [0, config.max_position_embeddings - 1] 中选择。

            # 什么是位置 IDs？参见 glossary#position-ids
        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            # 用于将自注意力模块中的选定头部置零的掩码。掩码值在 [0, 1] 中选择：

            # - 1 表示头部未被遮掩，
            # - 0 表示头部被遮掩。

        inputs_embeds (`torch.FloatTensor` of shape `({0}, hidden_size)`, *optional*):
            # 可选地，您可以直接传递嵌入表示而不是传递 input_ids。如果您想要更多控制如何将 input_ids 索引转换为相关向量，而不是使用模型的内部嵌入查找矩阵，这将很有用。
        output_attentions (`bool`, *optional*):
            # 是否返回所有注意力层的注意力张量。
            # 有关更多详细信息，请参见返回张量中的 attentions。
        output_hidden_states (`bool`, *optional*):
            # 是否返回所有层的隐藏状态。
            # 有关更多详细信息，请参见返回张量中的 hidden_states。
        return_dict (`bool`, *optional*):
            # 是否返回一个 utils.ModelOutput 而不是普通元组。
# 导入必要的库
import torch
import torch.nn as nn

# 添加起始文档注释
@add_start_docstrings(
    "The bare Splinter Model transformer outputting raw hidden-states without any specific head on top.",
    SPLINTER_START_DOCSTRING,
)

# 定义SplinterModel类
class SplinterModel(SplinterPreTrainedModel):
    """
    The model is an encoder (with only self-attention) following the architecture described in [Attention is all you
    need](https://arxiv.org/abs/1706.03762) by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones,
    Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.
    """

    # 初始化方法
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        # 实例化SplinterEmbeddings和SplinterEncoder
        self.embeddings = SplinterEmbeddings(config)
        self.encoder = SplinterEncoder(config)

        # 初始化权重并应用最终处理
        self.post_init()

    # 获取输入嵌入
    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    # 设置输入嵌入
    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    # 剪枝模型的头
    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    # 添加模型前向方法的文档注释和代码示例文档注释
    @add_start_docstrings_to_model_forward(SPLINTER_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPastAndCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    
    # 定义前向传播方法
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
class SplinterFullyConnectedLayer(nn.Module):
    # 初始化方法
    def __init__(self, input_dim, output_dim, hidden_act="gelu"):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        # 线性层和激活函数
        self.dense = nn.Linear(self.input_dim, self.output_dim)
        self.act_fn = ACT2FN[hidden_act]
        self.LayerNorm = nn.LayerNorm(self.output_dim)

    # 定义前向传播方法
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(inputs)
        hidden_states = self.act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


# 定义QuestionAwareSpanSelectionHead类
class QuestionAwareSpanSelectionHead(nn.Module):
    """
    实现了问答相关的跨度选择（QASS）头部，参考 Splinter 论文描述。
    
    """
    
    # 初始化方法，接受一个配置参数对象
    def __init__(self, config):
        # 调用父类初始化方法
        super().__init__()
    
        # 创建查询开始转换层对象，输入输出维度为隐藏层大小
        self.query_start_transform = SplinterFullyConnectedLayer(config.hidden_size, config.hidden_size)
        # 创建查询结束转换层对象，输入输出维度为隐藏层大小
        self.query_end_transform = SplinterFullyConnectedLayer(config.hidden_size, config.hidden_size)
        # 创建开始转换层对象，输入输出维度为隐藏层大小
        self.start_transform = SplinterFullyConnectedLayer(config.hidden_size, config.hidden_size)
        # 创建结束转换层对象，输入输出维度为隐藏层大小
        self.end_transform = SplinterFullyConnectedLayer(config.hidden_size, config.hidden_size)
    
        # 创建开始分类器线性层对象，输入输出维度为隐藏层大小，不带偏置项
        self.start_classifier = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        # 创建结束分类器线性层对象，输入输出维度为隐藏层大小，不带偏置项
        self.end_classifier = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
    
    # 前向传播方法，接受输入张量和位置张量
    def forward(self, inputs, positions):
        # 获取输入张量的维度信息
        _, _, dim = inputs.size()
        # 扩展位置张量的最后一个维度，重复 dim 次，得到 [batch_size, num_positions, dim] 大小的张量
        index = positions.unsqueeze(-1).repeat(1, 1, dim)
        # 根据位置张量从输入张量中聚合得到对应位置的表示，得到 [batch_size, num_positions, dim] 大小的张量
        gathered_reps = torch.gather(inputs, dim=1, index=index)
    
        # 对聚合表示进行查询开始转换操作，得到 [batch_size, num_positions, dim] 大小的张量
        query_start_reps = self.query_start_transform(gathered_reps)
        # 对聚合表示进行查询结束转换操作，得到 [batch_size, num_positions, dim] 大小的张量
        query_end_reps = self.query_end_transform(gathered_reps)
        # 对输入张量进行开始转换操作，得到 [batch_size, seq_length, dim] 大小的张量
        start_reps = self.start_transform(inputs)
        # 对输入张量进行结束转换操作，得到 [batch_size, seq_length, dim] 大小的张量
        end_reps = self.end_transform(inputs)
    
        # 使用开始分类器对查询开始表示进行分类，得到 [batch_size, num_positions, dim] 大小的张量
        hidden_states = self.start_classifier(query_start_reps)
        # 将开始表示张量的维度进行转置，得到 [batch_size, dim, seq_length] 大小的张量
        start_reps = start_reps.permute(0, 2, 1)
        # 计算开始位置的 logits，得到 [batch_size, num_positions, seq_length] 大小的张量
        start_logits = torch.matmul(hidden_states, start_reps)
    
        # 使用结束分类器对查询结束表示进行分类，得到 [batch_size, num_positions, dim] 大小的张量
        hidden_states = self.end_classifier(query_end_reps)
        # 将结束表示张量的维度进行转置，得到 [batch_size, dim, seq_length] 大小的张量
        end_reps = end_reps.permute(0, 2, 1)
        # 计算结束位置的 logits，得到 [batch_size, num_positions, seq_length] 大小的张量
        end_logits = torch.matmul(hidden_states, end_reps)
    
        # 返回开始位置 logits 和结束位置 logits
        return start_logits, end_logits
# 添加起始文档字符串，描述了 Splinter 模型和用于提取性问题回答任务的跨度分类头部的作用
class SplinterForQuestionAnswering(SplinterPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        # 初始化 Splinter 模型
        self.splinter = SplinterModel(config)
        # 初始化问题感知的跨度选择头部
        self.splinter_qass = QuestionAwareSpanSelectionHead(config)
        # 获取问题符号的 token id
        self.question_token_id = config.question_token_id

        # 初始化权重并进行最终处理
        self.post_init()

    # 前向传播函数
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        start_positions: Optional[torch.LongTensor] = None,
        end_positions: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        question_positions: Optional[torch.LongTensor] = None,
        
#定义 Splinter 的预训练输出类，作为一个跨度选择模型的输出
@dataclass
class SplinterForPreTrainingOutput(ModelOutput):
    """
    Class for outputs of Splinter as a span selection model.
    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when start and end positions are provided):
            Total span extraction loss is the sum of a Cross-Entropy for the start and end positions.
            总的跨度抽取损失是起始位置和结束位置的交叉熵之和。
        start_logits (`torch.FloatTensor` of shape `(batch_size, num_questions, sequence_length)`):
            Span-start scores (before SoftMax).
            跨度起始得分（SoftMax之前）。
        end_logits (`torch.FloatTensor` of shape `(batch_size, num_questions, sequence_length)`):
            Span-end scores (before SoftMax).
            跨度结束得分（SoftMax之前）。
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
            每一层模型的隐藏状态以及可选的初始嵌入输出。
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
            注意力软化后的注意力权重，用于计算自注意力头中的加权平均值。
    """

    loss: Optional[torch.FloatTensor] = None  # 总跨度抽取损失
    start_logits: torch.FloatTensor = None  # 跨度起始得分
    end_logits: torch.FloatTensor = None  # 跨度结束得分
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None  # 隐藏状态
    attentions: Optional[Tuple[torch.FloatTensor]] = None  # 注意力权重
# 为递归跨度选择任务创建Splinter模型，与QA任务的区别在于没有问题，而是有多个问题标记替换了重复跨度的出现
@add_start_docstrings(
    """
    Splinter Model for the recurring span selection task as done during the pretraining. The difference to the QA task
    is that we do not have a question, but multiple question tokens that replace the occurrences of recurring spans
    instead.
    """,
    SPLINTER_START_DOCSTRING,
)
class SplinterForPreTraining(SplinterPreTrainedModel):
    # 初始化函数
    def __init__(self, config):
        # 调用父类的初始化函数
        super().__init__(config)
        # 实例化Splinter模型
        self.splinter = SplinterModel(config)
        # 实例化QuestionAwareSpanSelectionHead模型
        self.splinter_qass = QuestionAwareSpanSelectionHead(config)
        # 设置问题标记ID
        self.question_token_id = config.question_token_id
        
        # 初始化权重并应用最终处理
        self.post_init()

    # 前向传播函数
    @add_start_docstrings_to_model_forward(
        SPLINTER_INPUTS_DOCSTRING.format("batch_size, num_questions, sequence_length")
    )
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        start_positions: Optional[torch.LongTensor] = None,
        end_positions: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        question_positions: Optional[torch.LongTensor] = None,
    # 准备问题位置的内部函数
    def _prepare_question_positions(self, input_ids: torch.Tensor) -> torch.Tensor:
        # 在输入ID中查找问题标记的位置
        rows, flat_positions = torch.where(input_ids == self.config.question_token_id)
        # 统计每行的问题标记个数
        num_questions = torch.bincount(rows)
        # 创建位置张量，并用填充标记填充
        positions = torch.full(
            (input_ids.size(0), num_questions.max()),
            self.config.pad_token_id,
            dtype=torch.long,
            device=input_ids.device,
        )
        # 构建问题位置张量
        cols = torch.cat([torch.arange(n) for n in num_questions])
        positions[rows, cols] = flat_positions
        return positions
```