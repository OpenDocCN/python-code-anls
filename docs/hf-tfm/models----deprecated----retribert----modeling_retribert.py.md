# `.\models\deprecated\retribert\modeling_retribert.py`

```
# 该代码定义了一个名为 RetriBertPreTrainedModel 的抽象基类，用于处理 RetriBERT 模型的权重初始化和加载预训练模型的接口
class RetriBertPreTrainedModel(PreTrainedModel):
    # 设置配置类为 RetriBertConfig
    config_class = RetriBertConfig
    # 设置加载 TensorFlow 权重的方法为 None
    load_tf_weights = None
    # 设置基础模型前缀为 "retribert"
    base_model_prefix = "retribert"

    # 定义初始化权重的方法
    def _init_weights(self, module):
        # 如果 module 是 nn.Linear，用正态分布初始化权重，并将偏置设为 0
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        # 如果 module 是 nn.Embedding，用正态分布初始化权重，并将填充索引对应的权重设为 0
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        # 如果 module 是 nn.LayerNorm，将偏置设为 0，权重设为 1
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


这个基类提供了对 RetriBERT 模型权重初始化和加载预训练模型的通用方法。它继承自 `PreTrainedModel`，并定义了一些属性和方法，如配置类、TensorFlow 权重加载方法以及基础模型前缀。此外,还实现了一个私有方法 `_init_weights`，用于初始化不同类型的模块(如 `nn.Linear`、`nn.Embedding` 和 `nn.LayerNorm`)的权重。
    # 参数说明：
    # config ([`RetriBertConfig`]): 包含模型所有参数的模型配置类。
    # 用配置文件初始化不会加载与模型关联的权重，只会加载配置。查看 [`~PreTrainedModel.from_pretrained`] 方法以加载模型权重。
# 导入必要的库
from transformers import BertModel
from transformers.file_utils import add_start_docstrings
from transformers.modeling_utils import RetriBertConfig, RetriBertPreTrainedModel

# 定义基于 Bert 的模型用于查询或文档嵌入以进行文档检索
@add_start_docstrings(
    """Bert Based model to embed queries or document for document retrieval.""",
    RETRIBERT_START_DOCSTRING,
)
class RetriBertModel(RetriBertPreTrainedModel):
    # 初始化函数
    def __init__(self, config: RetriBertConfig) -> None:
        super().__init__(config)
        # 设置投影维度
        self.projection_dim = config.projection_dim

        # 定义查询 Bert 模型
        self.bert_query = BertModel(config)
        # 根据配置是否共享编码器，定义文档 Bert 模型
        self.bert_doc = None if config.share_encoders else BertModel(config)
        # 定义丢弃层
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # 为查询投影
        self.project_query = nn.Linear(config.hidden_size, config.projection_dim, bias=False)
        # 为文档投影
        self.project_doc = nn.Linear(config.hidden_size, config.projection_dim, bias=False)

        # 定义交叉熵损失
        self.ce_loss = nn.CrossEntropyLoss(reduction="mean")

        # 初始化权重并应用最终处理
        self.post_init()

    # 进行检查点检测的句子嵌入
    def embed_sentences_checkpointed(
        self,
        input_ids,
        attention_mask,
        sent_encoder,
        checkpoint_batch_size=-1,
    ):
        # 重现具有检查点的 BERT 前向传递
        if checkpoint_batch_size < 0 or input_ids.shape[0] < checkpoint_batch_size:
            return sent_encoder(input_ids, attention_mask=attention_mask)[1]
        else:
            # 准备隐式变量
            device = input_ids.device
            input_shape = input_ids.size()
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
            head_mask = [None] * sent_encoder.config.num_hidden_layers
            extended_attention_mask: torch.Tensor = sent_encoder.get_extended_attention_mask(
                attention_mask, input_shape
            )

            # 定义用于检查点检测的函数
            def partial_encode(*inputs):
                encoder_outputs = sent_encoder.encoder(
                    inputs[0],
                    attention_mask=inputs[1],
                    head_mask=head_mask,
                )
                sequence_output = encoder_outputs[0]
                pooled_output = sent_encoder.pooler(sequence_output)
                return pooled_output

            # 在所有内容上同时运行嵌入层
            embedding_output = sent_encoder.embeddings(
                input_ids=input_ids, position_ids=None, token_type_ids=token_type_ids, inputs_embeds=None
            )
            # 在一个小批次上一次运行编码和池化
            pooled_output_list = []
            for b in range(math.ceil(input_ids.shape[0] / checkpoint_batch_size)):
                b_embedding_output = embedding_output[b * checkpoint_batch_size : (b + 1) * checkpoint_batch_size]
                b_attention_mask = extended_attention_mask[b * checkpoint_batch_size : (b + 1) * checkpoint_batch_size]
                pooled_output = checkpoint.checkpoint(partial_encode, b_embedding_output, b_attention_mask)
                pooled_output_list.append(pooled_output)
            return torch.cat(pooled_output_list, dim=0)
    # 在嵌入问题中使用BERT模型并生成问题的表示形式
    def embed_questions(
        self,
        input_ids,  # 输入的问题文本的ID序列
        attention_mask=None,  # 用于掩码的注意力掩码
        checkpoint_batch_size=-1,  # 检查点批大小，默认为-1
    ):
        q_reps = self.embed_sentences_checkpointed(
            input_ids,  # 输入的问题文本的ID序列
            attention_mask,  # 用于掩码的注意力掩码
            self.bert_query,  # 问题的BERT模型
            checkpoint_batch_size,  # 检查点批大小
        )
        返回问题的表示形式
        return self.project_query(q_reps)
    
    # 在嵌入答案中使用BERT模型并生成答案的表示形式
    def embed_answers(
        self,
        input_ids,  # 输入的答案文本的ID序列
        attention_mask=None,  # 用于掩码的注意力掩码
        checkpoint_batch_size=-1,  # 检查点批大小，默认为-1
    ):
        a_reps = self.embed_sentences_checkpointed(
            input_ids,  # 输入的答案文本的ID序列
            attention_mask,  # 用于掩码的注意力掩码
            self.bert_query if self.bert_doc is None else self.bert_doc,  # 使用问题的BERT模型或文档的BERT模型（如果存在）
            checkpoint_batch_size,  # 检查点批大小
        )
        返回答案的表示形式
        return self.project_doc(a_reps)
    
    # 在前向传播中执行以下操作
    def forward(
        self,
        input_ids_query: torch.LongTensor,  # 输入的问题文本的ID序列
        attention_mask_query: Optional[torch.FloatTensor],  # 用于掩码的问题的注意力掩码
        input_ids_doc: torch.LongTensor,  # 输入的答案文本的ID序列
        attention_mask_doc: Optional[torch.FloatTensor],  # 用于掩码的答案的注意力掩码
        checkpoint_batch_size: int = -1,  # 检查点批大小，默认为-1
    # 返回类型提示，说明该函数返回一个 torch.FloatTensor 类型的值
    ) -> torch.FloatTensor:
        # 函数的文档字符串，描述了函数的参数和返回值
        r"""
        Args:
            input_ids_query (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                # 查询的输入序列标记的索引，在批处理中的查询的词汇表中。
                # 这些索引可以使用 AutoTokenizer 来获取。参见 PreTrainedTokenizer.encode 和 PreTrainedTokenizer.__call__ 的细节。
                # 什么是输入 ID？的解释
            attention_mask_query (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
                # 遮蔽掩码，避免在填充标记索引上进行注意力操作的遮罩。遮罩值选在[0, 1]范围内:
                # - 1 代表**未遮罩**的标记，
                # - 0 代表**被遮罩**的标记。
                # 什么是注意力遮罩？的解释
            input_ids_doc (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                # 在批处理中文档的词汇表中的输入序列标记的索引。
            attention_mask_doc (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
                # 避免在文档填充标记索引上进行注意力操作的遮罩。
            checkpoint_batch_size (`int`, *optional*, defaults to `-1`):
                # 如果大于 0，则使用梯度检查点，在 GPU 上一次只计算 `checkpoint_batch_size` 个示例的序列表示。
                # 所有查询表示仍然与批处理中的所有文档表示进行比较。
        
        Return:
            `torch.FloatTensor``: 尝试匹配每个查询与其对应文档，以及每个文档与其对应查询的双向交叉熵损失值
        """
        # 获取输入序列标记的设备信息
        device = input_ids_query.device
        # 使用输入的查询信息对问题进行嵌入表示
        q_reps = self.embed_questions(input_ids_query, attention_mask_query, checkpoint_batch_size)
        # 使用输入的文档信息对答案进行嵌入表示
        a_reps = self.embed_answers(input_ids_doc, attention_mask_doc, checkpoint_batch_size)
        # 比较两个表示之间的得分
        compare_scores = torch.mm(q_reps, a_reps.t())
        # 计算查询到答案的交叉熵损失
        loss_qa = self.ce_loss(compare_scores, torch.arange(compare_scores.shape[1]).to(device))
        # 计算答案到查询的交叉熵损失
        loss_aq = self.ce_loss(compare_scores.t(), torch.arange(compare_scores.shape[0]).to(device))
        # 计算平均损失
        loss = (loss_qa + loss_aq) / 2
        # 返回损失值
        return loss
```