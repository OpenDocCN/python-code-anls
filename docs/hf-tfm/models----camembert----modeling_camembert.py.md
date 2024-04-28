# `.\transformers\models\camembert\modeling_camembert.py`

```py
# 设置编码格式为 UTF-8
# 声明版权信息
# 导入所需模块
# 定义额外的模块导入形式：List, Optional, Tuple, Union
# 导入 PyTorch
# 导入神经网络模块
# 从本地或远程导入自定义的激活函数
# 导入模型输出相关的类
# 导入 PyTorch 模型的实用函数
# 导入日志记录器
# 设置文档注释中用于示例的检查点名称
# 设置文档注释中用于示例的配置文件名称
# 定义 CamemBERT 模型的预训练模型存档列表
# 开始文档字符串，提供关于 CamemBERT 模型的基本介绍和使用说明
class CamembertEmbeddings(nn.Module):
    """
    Camembert 的嵌入层，将输入的 tokens 转换为词嵌入表示。

    参数：
        config (CamembertConfig): 包含模型配置信息的对象。
    """
class CamembertEmbeddings(nn.Module):
    """
    Same as BertEmbeddings with a tiny tweak for positional embeddings indexing.
    """

    # 从transformers.models.bert.modeling_bert.BertEmbeddings.__init__复制而来的初始化函数
    def __init__(self, config):
        # 调用父类初始化函数
        super().__init__()
        # 词嵌入层，根据词汇表大小和隐藏层大小创建Embedding层
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        # 位置嵌入层，根据最大位置嵌入长度和隐藏层大小创建Embedding层
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        # 标记类型嵌入层，根据类型词汇表大小和隐藏层大小创建Embedding层
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm未使用蛇形命名，以保持与TensorFlow模型变量名称一致，以便加载任何TensorFlow检查点文件
        # LayerNorm层，对隐藏层进行归一化处理
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # Dropout层，用于随机失活以防止过拟合
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # 位置嵌入类型，默认为"absolute"
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        # 注册缓冲区，存储位置ID，位置ID在序列化时连续存储，并在导出时可见
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )
        # 注册缓冲区，存储标记类型ID，标记类型ID初始化为全0的张量
        self.register_buffer(
            "token_type_ids", torch.zeros(self.position_ids.size(), dtype=torch.long), persistent=False
        )

        # End copy
        # 填充ID，用于指定填充标记的索引
        self.padding_idx = config.pad_token_id
        # 位置嵌入层，根据最大位置嵌入长度和隐藏层大小创建Embedding层，指定填充索引
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size, padding_idx=self.padding_idx
        )

    # 前向传播函数
    def forward(
        self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0
        ):
            # 如果未提供位置编码（position_ids），则根据输入的标记 ID 创建位置编码
            if position_ids is None:
                # 如果提供了输入标记 ID，则根据输入标记 ID 创建位置编码。任何填充的标记仍然保持填充状态。
                position_ids = create_position_ids_from_input_ids(input_ids, self.padding_idx, past_key_values_length)
            else:
                # 否则，根据提供的输入嵌入（inputs_embeds）创建位置编码
                position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds)

        # 如果提供了输入标记 ID，则获取其形状，否则获取输入嵌入的形状
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        # 获取序列长度
        seq_length = input_shape[1]

        # 将 token_type_ids 设置为构造函数中注册的缓冲区，在那里全为零。这通常发生在自动生成时，
        # 注册的缓冲区可帮助用户在不传递 token_type_ids 的情况下跟踪模型，解决问题 #5664
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        # 如果未提供输入嵌入，则使用 word_embeddings 获取输入嵌入
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        # 使用 token_type_embeddings 获取 token 类型嵌入
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        # 将输入嵌入和 token 类型嵌入相加得到整合嵌入
        embeddings = inputs_embeds + token_type_embeddings
        # 如果位置嵌入类型是 "absolute"，则添加位置嵌入到整合嵌入中
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        # 对整合嵌入进行 LayerNormalization
        embeddings = self.LayerNorm(embeddings)
        # 对 LayerNormalization 后的结果进行 dropout 处理
        embeddings = self.dropout(embeddings)
        # 返回处理后的嵌入结果
        return embeddings

    # 从输入嵌入中创建位置编码
    def create_position_ids_from_inputs_embeds(self, inputs_embeds):
        """
        We are provided embeddings directly. We cannot infer which are padded so just generate sequential position ids.

        Args:
            inputs_embeds: torch.Tensor

        Returns: torch.Tensor
        """
        # 获取输入嵌入的形状
        input_shape = inputs_embeds.size()[:-1]
        # 获取序列长度
        sequence_length = input_shape[1]

        # 生成连续的位置编码，因为我们直接提供了嵌入，无法推断哪些是填充的
        position_ids = torch.arange(
            self.padding_idx + 1, sequence_length + self.padding_idx + 1, dtype=torch.long, device=inputs_embeds.device
        )
        # 将位置编码扩展为与输入形状相同的形状
        return position_ids.unsqueeze(0).expand(input_shape)
# 从transformers.models.roberta.modeling_roberta.RobertaSelfAttention复制代码，并将Roberta->Camembert
class CamembertSelfAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        # 检查隐藏大小是否是注意力头数的倍数，如果不是则引发错误
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # 初始化查询、键、值的线性层
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        # 初始化dropout层
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )
        # 如果位置嵌入类型是相对键或相对键查询，则初始化距离嵌入
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

        self.is_decoder = config.is_decoder

    # 将输入张量转换为分数张量
    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    # 前向传播函数
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
# 从transformers.models.roberta.modeling_roberta.RobertaSelfOutput复制代码，并将Roberta->Camembert
class CamembertSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 初始化全连接层、LayerNorm和dropout层
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    # 前向传播函数
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states
# 定义了一个自注意力模块，用于Camembert模型
class CamembertAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        # 初始化自注意力层
        self.self = CamembertSelfAttention(config, position_embedding_type=position_embedding_type)
        # 初始化输出层
        self.output = CamembertSelfOutput(config)
        # 初始化已剪枝的注意力头集合
        self.pruned_heads = set()

    # 剪枝注意力头
    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        # 寻找可剪枝的注意力头及其索引
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # 剪枝线性层
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # 更新超参数并存储已剪枝的注意力头
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
        # 自注意力层的前向传播
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        # 输出层的前向传播
        attention_output = self.output(self_outputs[0], hidden_states)
        # 如果需要输出注意力矩阵，则将其添加到输出中
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


# 定义了一个中间层模块，用于Camembert模型
class CamembertIntermediate(nn.Module):
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
        # 经过线性层
        hidden_states = self.dense(hidden_states)
        # 经过激活函数
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


# 定义了一个输出层模块，用于Camembert模型
class CamembertOutput(nn.Module):
    # 初始化函数，接受一个配置参数
    def __init__(self, config):
        # 调用父类的初始化函数
        super().__init__()
        # 创建一个全连接层，输入大小为config.intermediate_size，输出大小为config.hidden_size
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        # 创建一个LayerNorm层，输入大小为config.hidden_size，eps为config.layer_norm_eps
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 创建一个Dropout层，概率为config.hidden_dropout_prob
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
    
    # 前向传播函数，接受两个张量参数，返回一个张量
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 将hidden_states传入全连接层，得到输出
        hidden_states = self.dense(hidden_states)
        # 对输出进行Dropout操作
        hidden_states = self.dropout(hidden_states)
        # 将Dropout后的输出与input_tensor相加，然后传入LayerNorm层
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        # 返回LayerNorm层的输出
        return hidden_states
# 从transformers.models.roberta.modeling_roberta.RobertaLayer复制而来，修改Roberta为Camembert
class CamembertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 设置前向传播的分块大小
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        # 序列长度的维度
        self.seq_len_dim = 1
        # CamembertAttention层
        self.attention = CamembertAttention(config)
        # 是否作为解码器
        self.is_decoder = config.is_decoder
        # 是否添加交叉注意力
        self.add_cross_attention = config.add_cross_attention
        # 如果添加了交叉注意力
        if self.add_cross_attention:
            # 如果不是解码器，则抛出错误
            if not self.is_decoder:
                raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
            # 设置交叉注意力层
            self.crossattention = CamembertAttention(config, position_embedding_type="absolute")
        # 中间层
        self.intermediate = CamembertIntermediate(config)
        # 输出层
        self.output = CamembertOutput(config)

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
        # 如果过去的键/值已经存在，则使用前两个位置的元素，否则设置为 None
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        # 使用 self-attention 模块进行注意力计算
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        # 获取 self-attention 输出
        attention_output = self_attention_outputs[0]

        # 如果是解码器，则最后一个输出是 self-attn 缓存的元组
        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            # 如果不是解码器，输出中包括 self-attention 的结果
            outputs = self_attention_outputs[1:]  # 如果我们输出注意力权重，则添加自注意力
        

        cross_attn_present_key_value = None
        if self.is_decoder and encoder_hidden_states is not None:
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers"
                    " by setting `config.add_cross_attention=True`"
                )

            # cross_attn 缓存的键/值元组位于 past_key_value 元组的第三、第四个位置
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            # 使用 cross-attention 模块进行注意力计算
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                cross_attn_past_key_value,
                output_attentions,
            )
            # 获取 cross-attention 输出
            attention_output = cross_attention_outputs[0]
            # 如果输出注意力权重，则添加交叉注意力结果
            outputs = outputs + cross_attention_outputs[1:-1]  # 如果我们输出注意力权重，则添加交叉注意力
            # 将交叉注意力缓存添加到 present_key_value 元组的第三、第四个位置
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        # 将 attention_output 传递给 feed_forward_chunk，应用前馈网络
        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        # 将前馈网络的输出添加到 outputs 中
        outputs = (layer_output,) + outputs

        # 如果是解码器，将注意力键/值作为最后一个输出返回
        if self.is_decoder:
            outputs = outputs + (present_key_value,)

        return outputs

    # 前馈网络的一部分，处理注意力输出
    def feed_forward_chunk(self, attention_output):
        # 使用中间层进行前馈计算
        intermediate_output = self.intermediate(attention_output)
        # 使用输出层进行前馈计算
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output
# 从transformers.models.roberta.modeling_roberta.RobertaEncoder复制而来，将Roberta替换为Camembert
class CamembertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([CamembertLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

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
        # 定义函数的返回类型为包含torch.Tensor的元组，或者BaseModelOutputWithPastAndCrossAttentions对象
        ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPastAndCrossAttentions]:
        # 如果output_hidden_states为True，则初始化一个空元组以保存所有隐藏状态，否则设为None
        all_hidden_states = () if output_hidden_states else None
        # 如果output_attentions为True，则初始化一个空元组以保存所有自注意力权重，否则设为None
        all_self_attentions = () if output_attentions else None
        # 如果output_attentions为True并且config中添加了交叉注意力，初始化一个空元组以保存所有交叉注意力权重，否则设为None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        # 如果启用了梯度检查点且处于训练模式
        if self.gradient_checkpointing and self.training:
            # 如果use_cache为True，警告use_cache与梯度检查点不兼容，将use_cache设为False
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # 如果use_cache为True，则初始化一个空元组以保存下一个解码器缓存，否则设为None
        next_decoder_cache = () if use_cache else None
        # 遍历每个解码器层
        for i, layer_module in enumerate(self.layer):
            # 如果output_hidden_states为True，则将当前隐藏状态添加到all_hidden_states中
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 获取当前层的头部遮罩，如果head_mask不为None，则取出对应位置的遮罩，否则设为None
            layer_head_mask = head_mask[i] if head_mask is not None else None
            # 获取当前层的过去键值对，如果past_key_values不为None，则取出对应位置的键值对，否则设为None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            # 如果启用了梯度检查点且处于训练模式
            if self.gradient_checkpointing and self.training:
                # 使用梯度检查点函数进行前向传播
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
                # 否则直接调用当前层的前向传播函数
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )

            # 更新当前隐藏状态为当前层的输出的第一个元素
            hidden_states = layer_outputs[0]
            # 如果use_cache为True，则将当前层的输出的最后一个元素加入到next_decoder_cache中
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            # 如果output_attentions为True，则将当前层的自注意力权重加入到all_self_attentions中
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                # 如果config中添加了交叉注意力，则将当前层的交叉注意力权重加入到all_cross_attentions中
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        # 如果output_hidden_states为True，则将当前隐藏状态添加到all_hidden_states中
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 如果return_dict为False，则返回包含非空元素的元组
        if not return_dict:
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
        # 否则，返回BaseModelOutputWithPastAndCrossAttentions对象，包含最后隐藏状态、下一个解码器缓存、所有隐藏状态、所有自注意力权重和所有交叉注意力权重
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )
# 从transformers.models.bert.modeling_bert.BertPooler中复制过来的类，用于Camembert模型的池化操作
class CamembertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 通过简单地取第一个标记对应的隐藏状态来"池化"模型
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


# 用于处理权重初始化和下载/加载预训练模型的抽象类
class CamembertPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = CamembertConfig
    base_model_prefix = "roberta"
    supports_gradient_checkpointing = True

    # 从transformers.models.bert.modeling_bert.BertPreTrainedModel._init_weights中复制过来的方法，用于初始化权重
    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # 与TF版本稍有不同，TF版本使用截断正态分布进行初始化
            # 参考 https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


# 输入文档字符串
CAMEMBERT_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `({0})`):
            # 输入序列标记在词汇表中的索引。
            # 可以使用 [`AutoTokenizer`] 获取索引。参见 [`PreTrainedTokenizer.encode`] 和 [`PreTrainedTokenizer.__call__`] 了解详情。
            # [什么是输入 ID？](../glossary#input-ids)
        attention_mask (`torch.FloatTensor` of shape `({0})`, *optional*):
            # 避免在填充标记索引上执行注意力的掩码。掩码值选择在 `[0, 1]` 之间：
            # - 1 表示**未被掩码**的标记，
            # - 0 表示**被掩码**的标记。
            # [什么是注意力掩码？](../glossary#attention-mask)
        token_type_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            # 指示输入的第一部分和第二部分的段标记索引。索引选择在 `[0, 1]` 之间：
            # - 0 对应于*句子 A* 标记，
            # - 1 对应于*句子 B* 标记。
            # [什么是标记类型 ID？](../glossary#token-type-ids)
        position_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            # 每个输入序列标记在位置嵌入中的位置索引。选择范围在 `[0, config.max_position_embeddings - 1]` 之间。
            # [什么是位置 ID？](../glossary#position-ids)
        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            # 用于将自注意力模块的选定头部置零的掩码。掩码值选择在 `[0, 1]` 之间：
            # - 1 表示头部**未被掩码**，
            # - 0 表示头部**被掩码**。
        inputs_embeds (`torch.FloatTensor` of shape `({0}, hidden_size)`, *optional*):
            # 可选地，可以直接传递嵌入表示而不是传递 `input_ids`。如果您想要更多控制如何将 `input_ids` 索引转换为关联向量，这将很有用，而不是使用模型的内部嵌入查找矩阵。
        output_attentions (`bool`, *optional*):
            # 是否返回所有注意力层的注意力张量。有关更多详细信息，请参见返回的张量下的 `attentions`。
        output_hidden_states (`bool`, *optional*):
            # 是否返回所有层的隐藏状态。有关更多详细信息，请参见返回的张量下的 `hidden_states`。
        return_dict (`bool`, *optional*):
            # 是否返回一个 [`~utils.ModelOutput`] 而不是一个普通的元组。
# 从 transformers.models.roberta.modeling_roberta 中复制了 CamembertClassificationHead 类，用于句子级别分类任务
class CamembertClassificationHead(nn.Module):
    """用于句子级别分类任务的头部。"""

    def __init__(self, config):
        super().__init__()
        # 稠密连接层，输入维度为 config.hidden_size，输出维度为 config.hidden_size
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 分类器的 dropout，如果配置中没有指定，则使用隐藏层 dropout
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        # 输出投影层，输入维度为 config.hidden_size，输出维度为 config.num_labels
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        # 取 features 的第一个 token，相当于取 [CLS] token
        x = features[:, 0, :]
        # 对 x 进行 dropout
        x = self.dropout(x)
        # 经过稠密连接层
        x = self.dense(x)
        # 使用 tanh 激活函数
        x = torch.tanh(x)
        # 再次进行 dropout
        x = self.dropout(x)
        # 经过输出投影层
        x = self.out_proj(x)
        return x


# 从 transformers.models.roberta.modeling_roberta 中复制了 CamembertLMHead 类，用于遮蔽语言建模任务
class CamembertLMHead(nn.Module):
    """用于遮蔽语言建模任务的头部。"""

    def __init__(self, config):
        super().__init__()
        # 稠密连接层，输入维度为 config.hidden_size，输出维度为 config.hidden_size
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # Layer normalization
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 解码器，输入维度为 config.hidden_size，输出维度为 config.vocab_size
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.decoder.bias = self.bias

    def forward(self, features, **kwargs):
        # 经过稠密连接层
        x = self.dense(features)
        # 经过 GELU 激活函数
        x = gelu(x)
        # Layer normalization
        x = self.layer_norm(x)

        # 使用解码器投影回词汇表大小，并添加偏置
        x = self.decoder(x)

        return x

    def _tie_weights(self):
        # 如果连接断开（在 TPU 上或偏置被调整大小时），将两个权重相连接
        # 为了加速兼容性和不破坏向后兼容性
        if self.decoder.bias.device.type == "meta":
            self.decoder.bias = self.bias
        else:
            self.bias = self.decoder.bias


# 从 transformers.models.roberta.modeling_roberta 中复制了 CamembertModel 类，该模型输出原始隐藏状态，没有额外的头部
@add_start_docstrings(
    "The bare CamemBERT Model transformer outputting raw hidden-states without any specific head on top.",
    CAMEMBERT_START_DOCSTRING,
)
class CamembertModel(CamembertPreTrainedModel):
    """输出原始隐藏状态的基本 CamemBERT 模型，没有额外的头部。"""

    def __init__(self, config):
        super().__init__()
        # 在这里定义模型结构，根据需要添加层次
        pass
    `add_cross_attention` set to `True`; an `encoder_hidden_states` is then expected as an input to the forward pass.
    设置`add_cross_attention`为`True`；然后期望在前向传播中输入`encoder_hidden_states`。

    .. _*Attention is all you need*: https://arxiv.org/abs/1706.03762
    参考自《Attention is all you need》：https://arxiv.org/abs/1706.03762

    """

    _no_split_modules = []

    # Copied from transformers.models.bert.modeling_bert.BertModel.__init__ with Bert->Camembert
    # 从transformers.models.bert.modeling_bert.BertModel.__init__复制而来，将Bert->Camembert
    def __init__(self, config, add_pooling_layer=True):
        # 调用父类的构造函数
        super().__init__(config)
        # 保存配置信息
        self.config = config

        # 初始化嵌入层
        self.embeddings = CamembertEmbeddings(config)
        # 初始化编码器层
        self.encoder = CamembertEncoder(config)

        # 如果需要添加池化层，则初始化池化层，否则置为None
        self.pooler = CamembertPooler(config) if add_pooling_layer else None

        # 初始化权重并应用最终处理
        self.post_init()

    def get_input_embeddings(self):
        # 返回输入嵌入层的词嵌入
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        # 设置输入嵌入层的词嵌入
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        # 对模型的头进行修剪
        for layer, heads in heads_to_prune.items():
            # 在这一层中修剪指定的头
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(CAMEMBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPoolingAndCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    # Copied from transformers.models.bert.modeling_bert.BertModel.forward
    # 从transformers.models.bert.modeling_bert.BertModel.forward复制而来
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
# 使用装饰器为类添加文档字符串，描述了 CamemBERT 模型及其特点
# 在此处使用了 CamembertForMaskedLM 类的文档字符串和 CAMEMBERT_START_DOCSTRING 常量
@add_start_docstrings(
    """CamemBERT Model with a `language modeling` head on top.""",
    CAMEMBERT_START_DOCSTRING,
)
# 从 transformers.models.roberta.modeling_roberta.RobertaForMaskedLM 复制代码，并将其中的 Roberta->Camembert, ROBERTA->CAMEMBERT
class CamembertForMaskedLM(CamembertPreTrainedModel):
    # 定义了绑定权重的键列表
    _tied_weights_keys = ["lm_head.decoder.weight", "lm_head.decoder.bias"]

    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)

        # 如果配置是一个解码器，发出警告信息
        if config.is_decoder:
            logger.warning(
                "If you want to use `CamembertForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        # 实例化一个 CamembertModel 对象，不添加池化层
        self.roberta = CamembertModel(config, add_pooling_layer=False)
        # 实例化一个 CamembertLMHead 对象
        self.lm_head = CamembertLMHead(config)

        # 初始化权重并应用最终处理
        self.post_init()

    # 获取输出嵌入
    def get_output_embeddings(self):
        return self.lm_head.decoder

    # 设置输出嵌入
    def set_output_embeddings(self, new_embeddings):
        self.lm_head.decoder = new_embeddings

    # 前向传播方法
    @add_start_docstrings_to_model_forward(CAMEMBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=MaskedLMOutput,
        config_class=_CONFIG_FOR_DOC,
        mask="<mask>",
        expected_output="' Paris'",
        expected_loss=0.1,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], MaskedLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        kwargs (`Dict[str, any]`, optional, defaults to *{}*):
            Used to hide legacy arguments that have been deprecated.
        """
        # 设置是否返回字典形式的输出，如果未指定则使用配置中的默认值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 将输入传递给 RoBERTa 模型进行处理
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # 获取 RoBERTa 模型的输出序列
        sequence_output = outputs[0]
        # 通过 lm_head 层对输出序列进行预测
        prediction_scores = self.lm_head(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            # 将标签移动到正确的设备以启用模型并行处理
            labels = labels.to(prediction_scores.device)
            # 定义交叉熵损失函数
            loss_fct = CrossEntropyLoss()
            # 计算掩码语言建模损失
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            # 如果不返回字典形式的输出，则返回元组形式的结果
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        # 返回 MaskedLMOutput 对象，包含损失、logits、隐藏状态和注意力权重
        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 导入依赖的模块和函数，并添加文档字符串，描述了 CamemBERT 模型的功能和用途
@add_start_docstrings(
    """
    CamemBERT Model transformer with a sequence classification/regression head on top (a linear layer on top of the
    pooled output) e.g. for GLUE tasks.
    """,
    CAMEMBERT_START_DOCSTRING,
)
# 定义 CamembertForSequenceClassification 类，继承自 CamembertPreTrainedModel 类
# 这个类用于在 Camembert 模型的基础上添加一个顶层的线性层，用于序列分类或回归任务，比如 GLUE 任务
class CamembertForSequenceClassification(CamembertPreTrainedModel):
    # 初始化方法，接受一个 config 参数
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)
        # 设置实例变量 num_labels，表示标签的数量
        self.num_labels = config.num_labels
        # 将传入的配置参数保存到实例变量 config 中
        self.config = config

        # 创建一个 CamembertModel 对象，用于处理输入序列
        self.roberta = CamembertModel(config, add_pooling_layer=False)
        # 创建一个 CamembertClassificationHead 对象，用于序列分类任务
        self.classifier = CamembertClassificationHead(config)

        # 初始化模型权重并应用最终处理
        self.post_init()

    # forward 方法，接受多个输入参数，并返回模型的输出结果
    @add_start_docstrings_to_model_forward(CAMEMBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint="cardiffnlp/twitter-roberta-base-emotion",
        output_type=SequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output="'optimism'",
        expected_loss=0.08,
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
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        # 如果 return_dict 为 None，则使用配置中的 use_return_dict
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用 RoBERTa 模型进行前向传播
        outputs = self.roberta(
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
        # 获取 RoBERTa 模型的输出序列
        sequence_output = outputs[0]
        # 将序列输出传入分类器，得到分类 logits
        logits = self.classifier(sequence_output)

        loss = None
        # 如果存在标签，则计算损失
        if labels is not None:
            # 将标签移动到正确的设备上以启用模型并行处理
            labels = labels.to(logits.device)
            # 确定问题类型
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            # 根据问题类型选择损失函数
            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        # 如果 return_dict 为 False，则返回非字典形式的输出
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 返回字典形式的输出
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 使用多项选择分类头部的 CamemBERT 模型（在汇总输出之上有一个线性层和 softmax），例如用于 RocStories/SWAG 任务
# 从 transformers.models.roberta.modeling_roberta.RobertaForMultipleChoice 复制并替换 Roberta->Camembert, ROBERTA->CAMEMBERT
class CamembertForMultipleChoice(CamembertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        # 初始化 Camembert 模型
        self.roberta = CamembertModel(config)
        # 添加 dropout 层
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # 添加分类器，输出维度为1
        self.classifier = nn.Linear(config.hidden_size, 1)

        # 初始化权重并应用最终处理
        self.post_init()

    # 将注释添加到 model_forward 函数
    @add_start_docstrings_to_model_forward(
        CAMEMBERT_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length")
    )
    # 添加代码示例的注释
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=MultipleChoiceModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
):
```  
    ) -> Union[Tuple[torch.Tensor], MultipleChoiceModelOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the multiple choice classification loss. Indices should be in `[0, ...,
            num_choices-1]` where `num_choices` is the size of the second dimension of the input tensors. (See
            `input_ids` above)
        """
        # 确定是否返回字典格式的结果，如果未指定，则使用配置中的默认设置
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 计算选择数量，即输入张量的第二个维度的大小
        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

        # 如果存在输入的id张量，则将其展平，否则保持为None
        flat_input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        # 如果存在位置id张量，则将其展平，否则保持为None
        flat_position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        # 如果存在token类型id张量，则将其展平，否则保持为None
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        # 如果存在注意力掩码张量，则将其展平，否则保持为None
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        # 如果存在输入嵌入张量，则将其展平，否则保持为None
        flat_inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )

        # 将平铺后的张量传递给RoBERTa模型进行处理
        outputs = self.roberta(
            flat_input_ids,
            position_ids=flat_position_ids,
            token_type_ids=flat_token_type_ids,
            attention_mask=flat_attention_mask,
            head_mask=head_mask,
            inputs_embeds=flat_inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # 获取汇聚输出
        pooled_output = outputs[1]

        # 对汇聚输出应用丢弃层
        pooled_output = self.dropout(pooled_output)
        # 将丢弃后的输出传递给分类器，以获取分类结果的对数概率
        logits = self.classifier(pooled_output)
        # 重新整形对数概率张量以匹配多项选择的形状
        reshaped_logits = logits.view(-1, num_choices)

        # 计算损失，如果提供了标签
        loss = None
        if labels is not None:
            # 将标签移动到正确的设备以启用模型并行处理
            labels = labels.to(reshaped_logits.device)
            # 使用交叉熵损失函数计算损失
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)

        # 如果不返回字典格式的结果，则按元组格式返回输出
        if not return_dict:
            output = (reshaped_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 返回字典格式的结果
        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 为CamembertForTokenClassification类添加文档字符串，描述其在顶部的token分类头部分（隐藏状态输出的顶部的线性层），例如用于命名实体识别（NER）任务
@add_start_docstrings(
    """
    CamemBERT Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g.
    for Named-Entity-Recognition (NER) tasks.
    """,
    CAMEMBERT_START_DOCSTRING,
)
# 从transformers.models.roberta.modeling_roberta.RobertaForTokenClassification复制而来，将Roberta->Camembert, ROBERTA->CAMEMBERT
class CamembertForTokenClassification(CamembertPreTrainedModel):
    # 初始化方法
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)
        # 设置标签数量
        self.num_labels = config.num_labels

        # 使用CamembertModel创建一个Camembert模型，不添加池化层
        self.roberta = CamembertModel(config, add_pooling_layer=False)
        # 设置分类器的dropout
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        # 创建一个dropout层
        self.dropout = nn.Dropout(classifier_dropout)
        # 创建一个线性层，将隐藏状态映射到标签数量
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # 初始化权重并应用最终处理
        self.post_init()

    # 为forward方法添加文档字符串，描述其输入
    @add_start_docstrings_to_model_forward(CAMEMBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    # 为forward方法添加代码示例文档字符串
    @add_code_sample_docstrings(
        checkpoint="Jean-Baptiste/roberta-large-ner-english",
        output_type=TokenClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output="['O', 'ORG', 'ORG', 'O', 'O', 'O', 'O', 'O', 'LOC', 'O', 'LOC', 'LOC']",
        expected_loss=0.01,
    )
    # 前向传播方法
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
    ) -> Union[Tuple[torch.Tensor], TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        # 确保返回字典不为空，如果为空则使用模型配置中的设定
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 将输入传递给 RoBERTa 模型，获取模型输出
        outputs = self.roberta(
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

        # 从模型输出中获取序列输出
        sequence_output = outputs[0]

        # 应用 dropout 到序列输出
        sequence_output = self.dropout(sequence_output)
        # 将序列输出传递给分类器，得到预测 logits
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            # 将标签移到正确的设备上，以启用模型并行计算
            labels = labels.to(logits.device)
            # 定义交叉熵损失函数
            loss_fct = CrossEntropyLoss()
            # 计算损失
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            # 如果不返回字典，则返回元组
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 返回 TokenClassifierOutput 对象，包含损失、logits、隐藏状态和注意力权重
        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 导入函数装饰器，用于给模型添加文档字符串
@add_start_docstrings(
    """
    CamemBERT Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
    layers on top of the hidden-states output to compute `span start logits` and `span end logits`
    """,
    CAMEMBERT_START_DOCSTRING,
)
# 从transformers.models.roberta.modeling_roberta中复制RobertForQuestionAnswering类，并修改为CamembertForQuestionAnswering，修改Robert->Camembert，ROBERTA->CAMEMBERT
class CamembertForQuestionAnswering(CamembertPreTrainedModel):
    def __init__(self, config):
        # 调用父类构造函数初始化模型
        super().__init__(config)
        # 设置模型输出标签数
        self.num_labels = config.num_labels

        # 初始化Camembert模型，不添加池化层
        self.roberta = CamembertModel(config, add_pooling_layer=False)
        # 初始化线性层，用于预测答案起始和结束位置的logits
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        # 初始化模型权重并进行最终处理
        self.post_init()

    # 添加文档字符串到模型的前向传播函数
    @add_start_docstrings_to_model_forward(CAMEMBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    # 添加代码示例的文档字符串
    @add_code_sample_docstrings(
        checkpoint="deepset/roberta-base-squad2",
        output_type=QuestionAnsweringModelOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output="' puppet'",
        expected_loss=0.86,
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
    ) -> Union[Tuple[torch.Tensor], QuestionAnsweringModelOutput]:
        r"""
        start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional`):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional`):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        """
        # 设置返回字典，如果未指定则使用配置中的设置
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 使用 RoBERTa 模型处理输入数据
        outputs = self.roberta(
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

        # 获取序列输出
        sequence_output = outputs[0]

        # 通过 QA 输出层获取 logits
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # 如果在多 GPU 上，添加一个维度
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # 有时开始/结束位置超出模型输入范围，忽略这些项
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            # 定义交叉熵损失函数
            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        # 返回 QA 模型输出
        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 添加模型文档字符串，说明这是一个带有语言建模头部的 CamemBERT 模型，用于条件语言模型微调
@add_start_docstrings(
    """CamemBERT Model with a `language modeling` head on top for CLM fine-tuning.""", CAMEMBERT_START_DOCSTRING
)
# 从 transformers.models.roberta.modeling_roberta.RobertaForCausalLM 复制并修改而来，将所有的 "Roberta" 替换为 "Camembert"，"ROBERTA" 替换为 "CAMEMBERT"，"roberta-base" 替换为 "camembert-base"
class CamembertForCausalLM(CamembertPreTrainedModel):
    _tied_weights_keys = ["lm_head.decoder.weight", "lm_head.decoder.bias"]

    def __init__(self, config):
        super().__init__(config)

        # 如果要将 `CamembertLMHeadModel` 作为独立模型使用，请添加 `is_decoder=True`
        if not config.is_decoder:
            logger.warning("If you want to use `CamembertLMHeadModel` as a standalone, add `is_decoder=True.`")

        # 初始化 CamemBERT 模型和语言建模头部
        self.roberta = CamembertModel(config, add_pooling_layer=False)
        self.lm_head = CamembertLMHead(config)

        # 初始化权重并应用最终处理
        self.post_init()

    # 获取输出嵌入
    def get_output_embeddings(self):
        return self.lm_head.decoder

    # 设置输出嵌入
    def set_output_embeddings(self, new_embeddings):
        self.lm_head.decoder = new_embeddings

    # 前向传播函数，使用文档字符串 `add_start_docstrings_to_model_forward` 和 `replace_return_docstrings` 进行说明
    @add_start_docstrings_to_model_forward(CAMEMBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=CausalLMOutputWithCrossAttentions, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        past_key_values: Tuple[Tuple[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    # 为生成准备输入数据，包括输入的 ID、过去的键值对、注意力掩码等
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, **model_kwargs):
        # 获取输入数据的形状
        input_shape = input_ids.shape
        # 如果没有提供注意力掩码，则创建一个全为1的注意力掩码
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_shape)

        # 如果存在过去的键值对，则根据过去的键值对裁剪输入的 ID
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]

            # 一些生成方法可能只传递最后一个输入 ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # 默认保留最后一个 ID
                remove_prefix_length = input_ids.shape[1] - 1

            input_ids = input_ids[:, remove_prefix_length:]

        # 返回包含输入 ID、注意力掩码和过去键值对的字典
        return {"input_ids": input_ids, "attention_mask": attention_mask, "past_key_values": past_key_values}

    # 重新排序缓存中的过去键值对
    def _reorder_cache(self, past_key_values, beam_idx):
        reordered_past = ()
        # 遍历每一层的过去键值对
        for layer_past in past_key_values:
            # 对每个过去状态按照 beam_idx 重新排序
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        # 返回重新排序后的过去键值对
        return reordered_past
# 从输入的 input_ids 中创建位置编码，用于标记非填充符号的位置。位置编码从 padding_idx+1 开始。填充符号将被忽略。
# 此函数修改自 fairseq 的 `utils.make_positions`。

def create_position_ids_from_input_ids(input_ids, padding_idx, past_key_values_length=0):
    # 创建一个掩码，用于标记非填充符号
    mask = input_ids.ne(padding_idx).int()
    # 对掩码进行累积求和，并转换数据类型，使其与 mask 相同。然后加上过去的键值长度 past_key_values_length 并乘以 mask。
    incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask
    # 将结果转换为 long 类型，并加上填充索引 padding_idx，得到最终的位置编码
    return incremental_indices.long() + padding_idx
```