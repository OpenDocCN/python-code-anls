# `.\transformers\models\xlm\modeling_xlm.py`

```py
# 设置编码格式为 UTF-8
# 版权声明，告知使用者可以在遵守许可证的情况下使用该文件
# 导入需要的库
# 设置对文本进行分类的 loss 函数
# 设置文档生成的检查点
# 设置配置用于文档生成
# 设置预训练模型的文件列表
# 创建正弦嵌入
# 使用给定的参数创建隐藏状态的 mask，以及可选的注意力 mask
    # 检查mask张量的形状是否为(bs, slen)，如果不是则触发断言错误
    assert mask.size() == (bs, slen)
    # 如果causal为False，则检查attn_mask张量的形状是否为(bs, slen, slen)，如果不是则触发断言错误
    assert causal is False or attn_mask.size() == (bs, slen, slen)

    # 返回mask和attn_mask张量
    return mask, attn_mask
# 定义一个多头注意力机制的类
class MultiHeadAttention(nn.Module):
    # 定义一个迭代器，生成每个实例唯一的ID
    NEW_ID = itertools.count()

    def __init__(self, n_heads, dim, config):
        super().__init__()
        # 为当前实例生成一个唯一的ID
        self.layer_id = next(MultiHeadAttention.NEW_ID)
        self.dim = dim
        self.n_heads = n_heads
        self.dropout = config.attention_dropout
        # 确保dim可以被n_heads整除
        assert self.dim % self.n_heads == 0

        # 定义4个线性层，用于输入向量的线性变换
        self.q_lin = nn.Linear(dim, dim)
        self.k_lin = nn.Linear(dim, dim)
        self.v_lin = nn.Linear(dim, dim)
        self.out_lin = nn.Linear(dim, dim)
        # 设置一个用于记录被删除头的集合
        self.pruned_heads = set()

    # 根据给定的头数删除一些头，并更新模型的参数
    def prune_heads(self, heads):
        # 计算每个注意力头的大小
        attention_head_size = self.dim // self.n_heads
        # 如果没有指定需要删除的头，则直接返回
        if len(heads) == 0:
            return
        # 查找可删除的头的索引
        heads, index = find_pruneable_heads_and_indices(heads, self.n_heads, attention_head_size, self.pruned_heads)
        # 对线性层进行剪枝
        self.q_lin = prune_linear_layer(self.q_lin, index)
        self.k_lin = prune_linear_layer(self.k_lin, index)
        self.v_lin = prune_linear_layer(self.v_lin, index)
        self.out_lin = prune_linear_layer(self.out_lin, index, dim=1)
        # 更新超参数
        self.n_heads = self.n_heads - len(heads)
        self.dim = attention_head_size * self.n_heads
        self.pruned_heads = self.pruned_heads.union(heads)
    def forward(self, input, mask, kv=None, cache=None, head_mask=None, output_attentions=False):
        """
        Self-attention (if kv is None) or attention over source sentence (provided by kv).
        """
        # 定义模型前向传播函数，包含输入，mask，kv，cache，head_mask等参数
        # 当kv为None时表示自注意力机制，否则表示对源句子的注意力
        # Input is (bs, qlen, dim)
        # Mask is (bs, klen) (non-causal) or (bs, klen, klen)
        bs, qlen, dim = input.size()
        # 获取输入的batch size，query长度和维度
        if kv is None:
            klen = qlen if cache is None else cache["slen"] + qlen
        else:
            klen = kv.size(1)
        # 如果kv为None，klen为输入的query长度，否则为kv的长度
        n_heads = self.n_heads
        dim_per_head = self.dim // n_heads
        # 计算头数和每个头的维度
        mask_reshape = (bs, 1, qlen, klen) if mask.dim() == 3 else (bs, 1, 1, klen)
        # 根据mask的维度不同进行reshape，变为(bs, 1, qlen, klen)或(bs, 1, 1, klen)

        def shape(x):
            """projection"""
            # 定义函数shape，用于将输入进行投影操作
            return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)

        def unshape(x):
            """compute context"""
            # 定义函数unshape，用于计算上下文
            return x.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * dim_per_head)

        q = shape(self.q_lin(input))  # (bs, n_heads, qlen, dim_per_head)
        # 对输入进行投影并reshape为(bs, n_heads, qlen, dim_per_head)
        if kv is None:
            k = shape(self.k_lin(input))  # (bs, n_heads, qlen, dim_per_head)
            v = shape(self.v_lin(input))  # (bs, n_heads, qlen, dim_per_head)
        elif cache is None or self.layer_id not in cache:
            k = v = kv
            k = shape(self.k_lin(k))  # (bs, n_heads, qlen, dim_per_head)
            v = shape(self.v_lin(v))  # (bs, n_heads, qlen, dim_per_head)

        if cache is not None:
            if self.layer_id in cache:
                if kv is None:
                    k_, v_ = cache[self.layer_id]
                    k = torch.cat([k_, k], dim=2)  # (bs, n_heads, klen, dim_per_head)
                    v = torch.cat([v_, v], dim=2)  # (bs, n_heads, klen, dim_per_head)
                else:
                    k, v = cache[self.layer_id]
            cache[self.layer_id] = (k, v)

        q = q / math.sqrt(dim_per_head)  # (bs, n_heads, qlen, dim_per_head)
        # 对query进行缩放
        scores = torch.matmul(q, k.transpose(2, 3))  # (bs, n_heads, qlen, klen)
        # 计算点积得分
        mask = (mask == 0).view(mask_reshape).expand_as(scores)  # (bs, n_heads, qlen, klen)
        # 根据mask的情况进行遮罩
        scores.masked_fill_(mask, torch.finfo(scores.dtype).min)  # (bs, n_heads, qlen, klen)
        # 将遮罩后的分数置为最小值

        weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)  # (bs, n_heads, qlen, klen)
        # 计算注意力权重
        weights = nn.functional.dropout(weights, p=self.dropout, training=self.training)  # (bs, n_heads, qlen, klen)
        # dropout操作

        # Mask heads if we want to
        if head_mask is not None:
            weights = weights * head_mask
        # 根据head_mask对注意力权重进行遮罩

        context = torch.matmul(weights, v)  # (bs, n_heads, qlen, dim_per_head)
        # 计算上下文表示
        context = unshape(context)  # (bs, qlen, dim)
        # 还原上下文表示的维度

        outputs = (self.out_lin(context),)
        # 输出结果
        if output_attentions:
            outputs = outputs + (weights,)
        # 如果需要输出注意力权重，将其添加到输出结果中
        return outputs
        # 返回输出结果
# 定义一个名为TransformerFFN的类，继承自nn.Module
class TransformerFFN(nn.Module):
    # 初始化方法，接受输入维度、隐藏层维度、输出维度和配置参数
    def __init__(self, in_dim, dim_hidden, out_dim, config):
        # 调用父类初始化方法
        super().__init__()
        # 保存dropout参数
        self.dropout = config.dropout
        # 创建输入层到隐藏层的线性变换
        self.lin1 = nn.Linear(in_dim, dim_hidden)
        # 创建隐藏层到输出层的线性变换
        self.lin2 = nn.Linear(dim_hidden, out_dim)
        # 选择使用GELU激活函数或ReLU激活函数
        self.act = gelu if config.gelu_activation else nn.functional.relu
        # 保存前馈传播的块大小
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        # 保存序列长度维度
        self.seq_len_dim = 1

    # 前向传播方法，接受输入并返回结果
    def forward(self, input):
        # 调用apply_chunking_to_forward方法对输入进行分块处理
        return apply_chunking_to_forward(self.ff_chunk, self.chunk_size_feed_forward, self.seq_len_dim, input)

    # 定义前馈传播的块方法，接受输入并返回结果
    def ff_chunk(self, input):
        # 对输入进行线性变换
        x = self.lin1(input)
        # 对结果应用激活函数
        x = self.act(x)
        # 对结果进行线性变换
        x = self.lin2(x)
        # 对结果进行dropout处理
        x = nn.functional.dropout(x, p=self.dropout, training=self.training)
        # 返回处理后的结果
        return x


# 定义XLMPreTrainedModel类，继承自PreTrainedModel
class XLMPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    # 定义配置类为XLMConfig
    config_class = XLMConfig
    # 不加载TensorFlow权重
    load_tf_weights = None
    # 设置基础模型前缀为"transformer"
    base_model_prefix = "transformer"

    # 初始化方法，接受任意输入参数
    def __init__(self, *inputs, **kwargs):
        # 调用父类初始化方法
        super().__init__(*inputs, **kwargs)

    # 定义dummy_inputs属性
    @property
    def dummy_inputs(self):
        # 创建模拟输入数据
        inputs_list = torch.tensor([[7, 6, 0, 0, 1], [1, 2, 3, 0, 0], [0, 0, 0, 4, 5]])
        attns_list = torch.tensor([[1, 1, 0, 0, 1], [1, 1, 1, 0, 0], [1, 0, 0, 1, 1]])
        # 如果使用语言嵌入并且语言数大于1
        if self.config.use_lang_emb and self.config.n_langs > 1:
            langs_list = torch.tensor([[1, 1, 0, 0, 1], [1, 1, 1, 0, 0], [1, 0, 0, 1, 1]])
        else:
            langs_list = None
        # 返回模拟输入数据字典
        return {"input_ids": inputs_list, "attention_mask": attns_list, "langs": langs_list}

    # 初始化权重方法，接受模块
    def _init_weights(self, module):
        # 如果模块是嵌入层
        if isinstance(module, nn.Embedding):
            # 如果配置不为空且包含嵌入初始化标准差
            if self.config is not None and self.config.embed_init_std is not None:
                # 使用正态分布初始化嵌入层权重
                nn.init.normal_(module.weight, mean=0, std=self.config.embed_init_std)
            # 如果模块包含填充索引
            if module.padding_idx is not None:
                # 将填充索引位置的权重设为0
                module.weight.data[module.padding_idx].zero_()
        # 如果模块是线性层
        if isinstance(module, nn.Linear):
            # 如果配置不为空且包含初始化标准差
            if self.config is not None and self.config.init_std is not None:
                # 使用正态分布初始化权重
                nn.init.normal_(module.weight, mean=0, std=self.config.init_std)
                # 如果模块包含偏置
                if module.bias is not None:
                    # 将偏置初始化为0
                    nn.init.constant_(module.bias, 0.0)
        # 如果模块是LayerNorm层
        if isinstance(module, nn.LayerNorm):
            # 将偏置初始化为0
            module.bias.data.zero_()
            # 将权重初始化为1
            module.weight.data.fill_(1.0)


# 定义XLMForQuestionAnsweringOutput类，继承自ModelOutput
@dataclass
class XLMForQuestionAnsweringOutput(ModelOutput):
    """
    Base class for outputs of question answering models using a `SquadHead`.
    """
    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned if both `start_positions` and `end_positions` are provided):
            分类损失，作为起始标记、结束标记（如果提供）分类损失的总和。
        start_top_log_probs (`torch.FloatTensor` of shape `(batch_size, config.start_n_top)`, *optional*, returned if `start_positions` or `end_positions` is not provided):
            前config.start_n_top个起始标记可能性的对数概率（beam-search）。
        start_top_index (`torch.LongTensor` of shape `(batch_size, config.start_n_top)`, *optional*, returned if `start_positions` or `end_positions` is not provided):
            前config.start_n_top个起始标记可能性的索引（beam-search）。
        end_top_log_probs (`torch.FloatTensor` of shape `(batch_size, config.start_n_top * config.end_n_top)`, *optional*, returned if `start_positions` or `end_positions` is not provided):
            前 `config.start_n_top * config.end_n_top` 个结束标记可能性的对数概率（beam-search）。
        end_top_index (`torch.LongTensor` of shape `(batch_size, config.start_n_top * config.end_n_top)`, *optional*, returned if `start_positions` or `end_positions` is not provided):
            前 `config.start_n_top * config.end_n_top` 个结束标记可能性的索引（beam-search）。
        cls_logits (`torch.FloatTensor` of shape `(batch_size,)`, *optional*, returned if `start_positions` or `end_positions` is not provided):
            答案“不可能”的标签对应的对数概率。
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            模型在每一层输出的隐藏状态的元组（一个用于嵌入的输出，一个用于每一层的输出），形状为 `(batch_size, sequence_length, hidden_size)`。

            每一层输出的隐藏状态以及初始嵌入的输出。
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            注意力权重的元组（每一层一个），形状为 `(batch_size, num_heads, sequence_length, sequence_length)`。

            注意力softmax之后的注意力权重，用于计算自注意力头中的加权平均值。
    """

    loss: Optional[torch.FloatTensor] = None
    start_top_log_probs: Optional[torch.FloatTensor] = None
    start_top_index: Optional[torch.LongTensor] = None
    end_top_log_probs: Optional[torch.FloatTensor] = None
    end_top_index: Optional[torch.LongTensor] = None
    cls_logits: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
XLM_START_DOCSTRING = r"""

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`XLMConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

XLM_INPUTS_DOCSTRING = r"""
"""


@add_start_docstrings(
    "The bare XLM Model transformer outputting raw hidden-states without any specific head on top.",
    XLM_START_DOCSTRING,
)
class XLMModel(XLMPreTrainedModel):
    # 获取输入嵌入层
    def get_input_embeddings(self):
        return self.embeddings

    # 设置输入嵌入层
    def set_input_embeddings(self, new_embeddings):
        self.embeddings = new_embeddings

    # 剪枝模型的注意力头
    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.attentions[layer].prune_heads(heads)

    @add_start_docstrings_to_model_forward(XLM_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        langs: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        lengths: Optional[torch.Tensor] = None,
        cache: Optional[Dict[str, torch.Tensor]] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,

class XLMPredLayer(nn.Module):
    """
    Prediction layer (cross_entropy or adaptive_softmax).
    """
    def __init__(self, config):
        super().__init__()
        # 初始化神经网络模型，从配置中获取相关参数
        self.asm = config.asm
        self.n_words = config.n_words
        self.pad_index = config.pad_index
        dim = config.emb_dim

        # 根据配置参数选择是否使用自适应softmax（Adaptive LogSoftmax）进行计算
        if config.asm is False:
            # 如果不使用自适应softmax，创建线性变换层（Linear），将输入维度映射到词汇表大小的维度
            self.proj = nn.Linear(dim, config.n_words, bias=True)
        else:
            # 如果使用自适应softmax，创建自适应softmax计算损失
            self.proj = nn.AdaptiveLogSoftmaxWithLoss(
                in_features=dim,
                n_classes=config.n_words,
                cutoffs=config.asm_cutoffs,
                div_value=config.asm_div_value,
                head_bias=True,  # 默认为False
            )

    def forward(self, x, y=None):
        """Compute the loss, and optionally the scores."""
        outputs = ()
        # 根据配置参数选择计算逻辑
        if self.asm is False:
            # 如果不使用自适应softmax，计算线性变换后的分数
            scores = self.proj(x)
            outputs = (scores,) + outputs
            # 如果目标（y）不为空，则计算交叉熵损失
            if y is not None:
                loss = nn.functional.cross_entropy(scores.view(-1, self.n_words), y.view(-1), reduction="mean")
                outputs = (loss,) + outputs
        else:
            # 如果使用自适应softmax，计算对数概率
            scores = self.proj.log_prob(x)
            outputs = (scores,) + outputs
            # 如果目标（y）不为空，则计算自适应softmax损失
            if y is not None:
                _, loss = self.proj(x, y)
                outputs = (loss,) + outputs

        return outputs
# 使用 XLM 模型架构的带有语言建模头的模型
@add_start_docstrings(
    """
    The XLM Model transformer with a language modeling head on top (linear layer with weights tied to the input
    embeddings).
    """,
    XLM_START_DOCSTRING,
)
class XLMWithLMHeadModel(XLMPreTrainedModel):
    # 绑定权重的键
    _tied_weights_keys = ["pred_layer.proj.weight"]

    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)
        # 创建 XLM 模型对象
        self.transformer = XLMModel(config)
        # 创建语言建模头对象
        self.pred_layer = XLMPredLayer(config)

        # 初始化权重并应用最终处理
        self.post_init()

    # 获取语言建模头的输出embedding
    def get_output_embeddings(self):
        return self.pred_layer.proj

    # 设置语言建模头的输出embedding
    def set_output_embeddings(self, new_embeddings):
        self.pred_layer.proj = new_embeddings

    # 为生成准备输入
    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        mask_token_id = self.config.mask_token_id
        lang_id = self.config.lang_id

        effective_batch_size = input_ids.shape[0]
        # 创建遮罩token
        mask_token = torch.full((effective_batch_size, 1), mask_token_id, dtype=torch.long, device=input_ids.device)
        # 将输入id拼接上遮罩token
        input_ids = torch.cat([input_ids, mask_token], dim=1)
        # 如果配置中有lang_id，创建lang_ids tensor
        if lang_id is not None:
            langs = torch.full_like(input_ids, lang_id)
        else:
            langs = None
        # 返回输入所需的字典
        return {"input_ids": input_ids, "langs": langs}

    # 模型前向计算
    @add_start_docstrings_to_model_forward(XLM_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=MaskedLMOutput,
        config_class=_CONFIG_FOR_DOC,
        mask="<special1>",
    )
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        langs: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        lengths: Optional[torch.Tensor] = None,
        cache: Optional[Dict[str, torch.Tensor]] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, MaskedLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        # 确定是否返回字典形式的输出
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 通过transformer进行模型的前向传播
        transformer_outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            langs=langs,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            lengths=lengths,
            cache=cache,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 从transformer输出中获取模型的输出
        output = transformer_outputs[0]

        # 使用预测层进行预测，返回损失值和logits或者仅返回logits，取决于是否提供了标签
        outputs = self.pred_layer(output, labels)  # (loss, logits) or (logits,) depending on if labels are provided.

        # 如果不以字典形式返回结果，则返回输出和transformer的其他输出（隐藏状态和注意力）
        if not return_dict:
            return outputs + transformer_outputs[1:]

        # 以MaskedLMOutput字典形式返回结果，包括损失、logits、隐藏状态和注意力
        return MaskedLMOutput(
            loss=outputs[0] if labels is not None else None,
            logits=outputs[0] if labels is None else outputs[1],
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )
# 基于 XLM 模型的序列分类/回归头部（在池化输出之上的线性层）的模型定义，用于 GLUE 任务等
class XLMForSequenceClassification(XLMPreTrainedModel):
    def __init__(self, config):
        # 调用父类构造函数
        super().__init__(config)
        # 记录标签数量和配置信息
        self.num_labels = config.num_labels
        self.config = config

        # 创建 XLM 模型和序列汇总层
        self.transformer = XLMModel(config)
        self.sequence_summary = SequenceSummary(config)

        # 初始化权重并应用最终处理
        self.post_init()

    # 前向传播函数的注释（输入参数的说明和示例）
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        langs: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        lengths: Optional[torch.Tensor] = None,
        cache: Optional[Dict[str, torch.Tensor]] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        返回值为元组或SequenceClassifierOutput类型
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            langs=langs,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            lengths=lengths,
            cache=cache,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        output = transformer_outputs[0]
        logits = self.sequence_summary(output)

        loss = None
        if labels is not None:
            如果标签不为None
            if self.config.problem_type is None:
                如果问题类型为None
                if self.num_labels == 1:
                    如果标签数为1
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    如果标签数大于1且标签类型为长整型或整型
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                如果问题类型为回归
                loss_fct = MSELoss()
                定义均方误差损失函数
                if self.num_labels == 1:
                    如果标签数为1
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                    使用损失函数计算损失
                else:
                    loss = loss_fct(logits, labels)
                    使用损失函数计算损失
            elif self.config.problem_type == "single_label_classification":
                如果问题类型为单标签分类
                loss_fct = CrossEntropyLoss()
                定义交叉熵损失函数
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                使用损失函数计算损失
            elif self.config.problem_type == "multi_label_classification":
                如果问题类型为多标签分类
                loss_fct = BCEWithLogitsLoss()
                定义二元交叉熵损失函数
                loss = loss_fct(logits, labels)
                使用损失函数计算损失

        if not return_dict:
            如果不返回字典
            output = (logits,) + transformer_outputs[1:]
            输出结果为logits和transformer_outputs的子集
            return ((loss,) + output) if loss is not None else output
            如果损失不为None，则返回(loss,) + output，否则返回output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )
# 添加 Span 分类头部的 XLM 模型，用于类似 SQuAD 这样的抽取式问答任务（在隐藏状态的输出上添加线性层计算`span start logits`和 `span end logits`）。
class XLMForQuestionAnsweringSimple(XLMPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.transformer = XLMModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        # 初始化权重并应用最终处理
        self.post_init()

    # 前向传播函数
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        langs: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        lengths: Optional[torch.Tensor] = None,
        cache: Optional[Dict[str, torch.Tensor]] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        start_positions: Optional[torch.Tensor] = None,
        end_positions: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,



# 添加带有 Beam-Search Span 分类头部的 XLM 模型，用于类似 SQuAD 这样的抽取式问答任务（在隐藏状态的输出上添加线性层计算`span start logits`和 `span end logits`）。
class XLMForQuestionAnswering(XLMPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.transformer = XLMModel(config)
        self.qa_outputs = SQuADHead(config)

        # 初始化权重并应用最终处理
        self.post_init()

    # 替换返回文档字符串
    # 此方法用于模型的前向传播
    def forward(
        # 输入序列的 token ID，类型为可选的 torch 张量
        input_ids: Optional[torch.Tensor] = None,
        # 注意力掩码，类型为可选的 torch 张量
        attention_mask: Optional[torch.Tensor] = None,
        # 输入序列的语言 ID，类型为可选的 torch 张量
        langs: Optional[torch.Tensor] = None,
        # token 类型 ID，类型为可选的 torch 张量
        token_type_ids: Optional[torch.Tensor] = None,
        # 位置 ID，类型为可选的 torch 张量
        position_ids: Optional[torch.Tensor] = None,
        # 输入序列的长度，类型为可选的 torch 张量
        lengths: Optional[torch.Tensor] = None,
        # 缓存，类型为可选的字典，包含字符串到 torch 张量的映射
        cache: Optional[Dict[str, torch.Tensor]] = None,
        # 头部屏蔽，类型为可选的 torch 张量
        head_mask: Optional[torch.Tensor] = None,
        # 输入嵌入，类型为可选的 torch 张量
        inputs_embeds: Optional[torch.Tensor] = None,
        # 起始位置，类型为可选的 torch 张量
        start_positions: Optional[torch.Tensor] = None,
        # 结束位置，类型为可选的 torch 张量
        end_positions: Optional[torch.Tensor] = None,
        # 是否不可能的标记，类型为可选的 torch 张量
        is_impossible: Optional[torch.Tensor] = None,
        # cls 索引，类型为可选的 torch 张量
        cls_index: Optional[torch.Tensor] = None,
        # p 掩码，类型为可选的 torch 张量
        p_mask: Optional[torch.Tensor] = None,
        # 是否输出注意力矩阵，类型为可选的布尔值
        output_attentions: Optional[bool] = None,
        # 是否输出隐藏状态，类型为可选的布尔值
        output_hidden_states: Optional[bool] = None,
        # 是否返回字典结果，类型为可选的布尔值
        return_dict: Optional[bool] = None,
# 使用装饰器为 XLMForTokenClassification 类添加文档字符串，描述其作用：在 XLM 模型的基础上添加一个标记分类头部，用于命名实体识别 (NER) 等任务
# 包含 XLM 模型的开始文档字符串
class XLMForTokenClassification(XLMPreTrainedModel):
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)
        # 获取标签数量
        self.num_labels = config.num_labels

        # 初始化 XLM 模型
        self.transformer = XLMModel(config)
        # 添加 dropout 层
        self.dropout = nn.Dropout(config.dropout)
        # 添加线性分类器
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # 初始化权重并应用最终处理
        self.post_init()

    # 将文档字符串添加到模型前向传播方法
    # 描述输入参数的作用和形状
    # 包含样例代码的文档字符串
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        langs: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        lengths: Optional[torch.Tensor] = None,
        cache: Optional[Dict[str, torch.Tensor]] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        ):
```  
    ) -> Union[Tuple, TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        # 如果没有提供 return_dict 参数，则根据配置决定是否使用返回字典
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用 transformer 方法进行模型转换
        outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            langs=langs,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            lengths=lengths,
            cache=cache,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取模型输出的序列表示
        sequence_output = outputs[0]

        # 对序列输出进行 dropout 操作
        sequence_output = self.dropout(sequence_output)
        # 使用分类器对序列输出进行分类
        logits = self.classifier(sequence_output)

        # 初始化损失值
        loss = None
        # 如果提供了标签，则计算损失值
        if labels is not None:
            # 使用交叉熵损失函数计算损失
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        # 如果没有要求返回字典，则组装输出结果
        if not return_dict:
            output = (logits,) + outputs[1:]
            # 如果存在损失值，则将损失值添加到输出中
            return ((loss,) + output) if loss is not None else output

        # 如果要求返回字典，则创建 TokenClassifierOutput 对象作为输出
        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 引入多项选择模型类，并在顶部添加了一个多项选择分类头（一个线性层叠加在池化输出之上，再加一个 softmax），用于例如 RocStories/SWAG 任务。
class XLMForMultipleChoice(XLMPreTrainedModel):
    def __init__(self, config, *inputs, **kwargs):
        # 调用父类构造函数初始化模型配置
        super().__init__(config, *inputs, **kwargs)

        # 实例化 XLMModel 类，并传入配置
        self.transformer = XLMModel(config)
        # 实例化 SequenceSummary 类，并传入配置
        self.sequence_summary = SequenceSummary(config)
        # 添加一个线性层，将输出尺寸变为 1
        self.logits_proj = nn.Linear(config.num_labels, 1)

        # 初始化权重并进行最终处理
        self.post_init()

    # 对模型的前向传播进行定义，接受一系列输入参数并返回相应输出
    @add_start_docstrings_to_model_forward(XLM_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=MultipleChoiceModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,  # 输入序列的 token IDs
        attention_mask: Optional[torch.Tensor] = None,  # 用于掩盖某些 token 的注意力掩码
        langs: Optional[torch.Tensor] = None,  # 语言标识符
        token_type_ids: Optional[torch.Tensor] = None,  # token 类型 IDs
        position_ids: Optional[torch.Tensor] = None,  # 位置 IDs
        lengths: Optional[torch.Tensor] = None,  # 输入序列的长度
        cache: Optional[Dict[str, torch.Tensor]] = None,  # 缓存字典
        head_mask: Optional[torch.Tensor] = None,  # 头部掩码
        inputs_embeds: Optional[torch.Tensor] = None,  # 输入嵌入
        labels: Optional[torch.Tensor] = None,  # 标签
        output_attentions: Optional[bool] = None,  # 是否返回注意力权重
        output_hidden_states: Optional[bool] = None,  # 是否返回隐藏状态
        return_dict: Optional[bool] = None,  # 是否以字典形式返回输出
        ) -> Union[Tuple, MultipleChoiceModelOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the multiple choice classification loss. Indices should be in `[0, ...,
            num_choices-1]` where `num_choices` is the size of the second dimension of the input tensors. (See
            `input_ids` above)
        """
        # 确保返回的字典类型是否为指定类型，如果没有指定则使用配置文件中的默认值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 计算选择数量（即选项的数量）
        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

        # 重塑输入的ids，如果存在则重塑，否则设为None
        input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        langs = langs.view(-1, langs.size(-1)) if langs is not None else None
        inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )

        # 如果长度参数不为空，则发出警告并将长度参数置为None
        if lengths is not None:
            logger.warning(
                "The `lengths` parameter cannot be used with the XLM multiple choice models. Please use the "
                "attention mask instead."
            )
            lengths = None

        # 运行transformer模块并获取输出
        transformer_outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            langs=langs,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            lengths=lengths,
            cache=cache,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # 从transformer输出中提取logits
        output = transformer_outputs[0]
        logits = self.sequence_summary(output)
        logits = self.logits_proj(logits)
        reshaped_logits = logits.view(-1, num_choices)

        # 如果存在labels，则计算loss，否则设为None
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)

        # 如果return_dict为False，则返回loss和transformer的输出，否则返回带有loss、logits、hidden_states、attentions的MultipleChoiceModelOutput类型
        if not return_dict:
            output = (reshaped_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )
```