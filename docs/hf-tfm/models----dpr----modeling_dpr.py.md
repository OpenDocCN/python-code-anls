# `.\models\dpr\modeling_dpr.py`

```py
# 设置代码文件的编码格式为 UTF-8
# 版权声明，指出代码的版权信息和使用许可
# 依照 Apache License 2.0 许可证使用本代码
# 获取许可证的详细信息请访问指定网址
# 根据适用法律或书面同意，本软件是基于"原样"提供的，没有任何形式的担保或条件
# 请查阅许可证获取更多信息
""" PyTorch DPR model for Open Domain Question Answering. """

# 导入必要的模块和类
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
from torch import Tensor, nn

# 导入模型输出相关的基类
from ...modeling_outputs import BaseModelOutputWithPooling
# 导入预训练模型相关的基类
from ...modeling_utils import PreTrainedModel
# 导入日志相关的模块
from ...utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
# 导入 BERT 模型
from ..bert.modeling_bert import BertModel
# 导入 DPR 配置
from .configuration_dpr import DPRConfig

# 获取日志记录器
logger = logging.get_logger(__name__)

# 用于文档的配置信息
_CONFIG_FOR_DOC = "DPRConfig"
_CHECKPOINT_FOR_DOC = "facebook/dpr-ctx_encoder-single-nq-base"

# 预训练模型的存档列表
DPR_CONTEXT_ENCODER_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/dpr-ctx_encoder-single-nq-base",
    "facebook/dpr-ctx_encoder-multiset-base",
]
DPR_QUESTION_ENCODER_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/dpr-question_encoder-single-nq-base",
    "facebook/dpr-question_encoder-multiset-base",
]
DPR_READER_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/dpr-reader-single-nq-base",
    "facebook/dpr-reader-multiset-base",
]


##########
# Outputs
##########

# DPRContextEncoderOutput 类，用于保存 DPRContextEncoder 的输出结果
@dataclass
class DPRContextEncoderOutput(ModelOutput):
    """
    Class for outputs of [`DPRQuestionEncoder`].
    """
    # `pooler_output` 是一个 `torch.FloatTensor` 张量，形状为 `(batch_size, embeddings_size)`，
    # 表示 DPR 编码器输出的池化器输出，对应于上下文表示。这是通过线性层进一步处理的序列第一个标记（分类标记）的最后一层隐藏状态。
    # 此输出用于将上下文嵌入以便与问题嵌入一起进行最近邻查询。

    pooler_output: torch.FloatTensor

    # `hidden_states` 是一个可选的元组，类型为 `tuple(torch.FloatTensor)`。
    # 当 `output_hidden_states=True` 或 `config.output_hidden_states=True` 时返回。
    # 它包含两个张量，第一个是嵌入的输出，第二个是每个层的输出。
    # 形状为 `(batch_size, sequence_length, hidden_size)`。

    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None

    # `attentions` 是一个可选的元组，类型为 `tuple(torch.FloatTensor)`。
    # 当 `output_attentions=True` 或 `config.output_attentions=True` 时返回。
    # 它包含每个层的注意力张量，形状为 `(batch_size, num_heads, sequence_length, sequence_length)`。
    # 这些是经过注意力 softmax 后的注意力权重，用于计算自注意力头中的加权平均值。

    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
# 使用 @dataclass 装饰器定义数据类 DPRQuestionEncoderOutput，继承自 ModelOutput
@dataclass
class DPRQuestionEncoderOutput(ModelOutput):
    """
    Class for outputs of [`DPRQuestionEncoder`].

    Args:
        pooler_output (`torch.FloatTensor` of shape `(batch_size, embeddings_size)`):
            The DPR encoder outputs the *pooler_output* that corresponds to the question representation. Last layer
            hidden-state of the first token of the sequence (classification token) further processed by a Linear layer.
            This output is to be used to embed questions for nearest neighbors queries with context embeddings.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """
    # 定义 pooler_output 字段，类型为 torch.FloatTensor，用于存储 DPR 编码器输出的池化表示
    pooler_output: torch.FloatTensor
    # 可选字段 hidden_states，存储 DPR 模型各层的隐藏状态，类型为 Tuple[torch.FloatTensor, ...]
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    # 可选字段 attentions，存储注意力权重，类型为 Tuple[torch.FloatTensor, ...]
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


# 使用 @dataclass 装饰器定义数据类 DPRReaderOutput，继承自 ModelOutput
@dataclass
class DPRReaderOutput(ModelOutput):
    """
    Class for outputs of [`DPRQuestionEncoder`].
    """
    # 定义函数参数及其类型注解：起始位置的预测概率张量，形状为 (段落数, 序列长度)
    start_logits: torch.FloatTensor
    # 可选参数：结束位置的预测概率张量，形状同上，默认为 None
    end_logits: torch.FloatTensor = None
    # 可选参数：DPRReader QA 分类器的输出，表示每个段落回答问题的相关性分数，形状为 (段落数, )
    relevance_logits: torch.FloatTensor = None
    # 可选参数：隐藏状态元组，包含模型每层的隐藏状态张量，形状为 (批大小, 序列长度, 隐藏大小)
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    # 可选参数：注意力权重元组，包含每层注意力权重张量，形状为 (批大小, 注意力头数, 序列长度, 序列长度)
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
class DPRPreTrainedModel(PreTrainedModel):
    def _init_weights(self, module):
        """Initialize the weights"""
        # 如果模块是线性层，则使用正态分布初始化权重，标准差为配置文件中的初始化范围
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果存在偏置项，则将其初始化为零
            if module.bias is not None:
                module.bias.data.zero_()
        # 如果模块是嵌入层，则同样使用正态分布初始化权重，标准差为配置文件中的初始化范围
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果指定了填充索引，则将对应位置的权重初始化为零
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        # 如果模块是层归一化层，则初始化偏置为零，权重为全1
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class DPREncoder(DPRPreTrainedModel):
    base_model_prefix = "bert_model"

    def __init__(self, config: DPRConfig):
        super().__init__(config)
        # 初始化BERT模型，不包含池化层
        self.bert_model = BertModel(config, add_pooling_layer=False)
        # 检查隐藏层大小是否合理
        if self.bert_model.config.hidden_size <= 0:
            raise ValueError("Encoder hidden_size can't be zero")
        # 设定投影维度
        self.projection_dim = config.projection_dim
        # 如果投影维度大于零，则增加一个线性层进行投影
        if self.projection_dim > 0:
            self.encode_proj = nn.Linear(self.bert_model.config.hidden_size, config.projection_dim)
        # 执行初始化权重和最终处理
        self.post_init()

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = False,
    ) -> Union[BaseModelOutputWithPooling, Tuple[Tensor, ...]]:
        # 调用BERT模型的前向传播
        outputs = self.bert_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # 获取序列输出和汇聚输出
        sequence_output = outputs[0]
        pooled_output = sequence_output[:, 0, :]

        # 如果设定了投影维度，则对汇聚输出进行投影
        if self.projection_dim > 0:
            pooled_output = self.encode_proj(pooled_output)

        # 如果不要求返回字典，则返回序列输出、汇聚输出以及其他隐藏状态
        if not return_dict:
            return (sequence_output, pooled_output) + outputs[2:]

        # 如果要求返回字典，则返回一个包含汇聚输出、最后隐藏状态、隐藏状态和注意力的模型输出对象
        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    @property
    def embeddings_size(self) -> int:
        # 如果设定了投影维度，则返回投影层的输出维度
        if self.projection_dim > 0:
            return self.encode_proj.out_features
        # 否则返回BERT模型的隐藏层大小
        return self.bert_model.config.hidden_size


class DPRSpanPredictor(DPRPreTrainedModel):
    base_model_prefix = "encoder"
    # 初始化方法，接受一个DPRConfig对象作为参数
    def __init__(self, config: DPRConfig):
        # 调用父类构造方法，初始化模型
        super().__init__(config)
        # 创建一个DPREncoder对象并赋值给self.encoder
        self.encoder = DPREncoder(config)
        # 创建一个线性层，用于生成问题答案的logits，输入大小为self.encoder.embeddings_size，输出大小为2
        self.qa_outputs = nn.Linear(self.encoder.embeddings_size, 2)
        # 创建一个线性层，用于生成问题相关性的logits，输入大小为self.encoder.embeddings_size，输出大小为1
        self.qa_classifier = nn.Linear(self.encoder.embeddings_size, 1)
        # 调用自定义的post_init方法，用于初始化权重和应用最终处理
        self.post_init()

    # 前向传播方法
    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        inputs_embeds: Optional[Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = False,
    ) -> Union[DPRReaderOutput, Tuple[Tensor, ...]]:
        # 计算输入的数量和序列长度
        n_passages, sequence_length = input_ids.size() if input_ids is not None else inputs_embeds.size()[:2]
        
        # 将输入传递给编码器进行处理
        outputs = self.encoder(
            input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # 获取编码器输出的序列张量
        sequence_output = outputs[0]

        # 计算问题答案的logits
        logits = self.qa_outputs(sequence_output)
        # 将logits沿着最后一个维度分割成start_logits和end_logits
        start_logits, end_logits = logits.split(1, dim=-1)
        # 去除不必要的维度并保证内存连续性
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()
        # 计算问题相关性的logits，仅使用序列输出的第一个向量
        relevance_logits = self.qa_classifier(sequence_output[:, 0, :])

        # 调整张量的大小以匹配预期的输出形状
        start_logits = start_logits.view(n_passages, sequence_length)
        end_logits = end_logits.view(n_passages, sequence_length)
        relevance_logits = relevance_logits.view(n_passages)

        # 如果return_dict为False，则返回一个元组
        if not return_dict:
            return (start_logits, end_logits, relevance_logits) + outputs[2:]

        # 如果return_dict为True，则返回一个DPRReaderOutput对象
        return DPRReaderOutput(
            start_logits=start_logits,
            end_logits=end_logits,
            relevance_logits=relevance_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
"""
    Contains the docstring for input specifications to DPR encoders.
"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            # 输入序列标记的索引，表示在词汇表中的位置。为了与预训练匹配，DPR输入序列应按以下方式格式化，包括[CLS]和[SEP]标记：

            (a) 对于序列对（例如标题+文本对）：

            ```
            tokens:         [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
            token_type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
            ```

            (b) 对于单个序列（例如问题）：

            ```
            tokens:         [CLS] the dog is hairy . [SEP]
            token_type_ids:   0   0   0   0  0     0   0
            ```

            DPR是一个具有绝对位置嵌入的模型，因此通常建议在右侧而不是左侧填充输入。

            可以使用[`AutoTokenizer`]获取索引。详见[`PreTrainedTokenizer.encode`]和[`PreTrainedTokenizer.__call__`]的详细说明。

            [什么是输入ID？](../glossary#input-ids)
        attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            # 避免在填充标记索引上执行注意力的掩码。掩码值选择在 `[0, 1]` 范围内：

            - 1 表示**未掩码**的标记，
            - 0 表示**已掩码**的标记。

            [什么是注意力掩码？](../glossary#attention-mask)
        token_type_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            # 段标记索引，指示输入的第一部分和第二部分。索引选择在 `[0, 1]` 范围内：

            - 0 对应于*句子A*标记，
            - 1 对应于*句子B*标记。

            [什么是标记类型ID？](../glossary#token-type-ids)
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            # 可选，可以直接传递嵌入表示而不是传递`input_ids`。如果需要比模型内部嵌入查找矩阵更多控制权，这将很有用。

        output_attentions (`bool`, *optional*):
            # 是否返回所有注意力层的注意力张量。查看返回的张量中的 `attentions` 以获取更多详细信息。

        output_hidden_states (`bool`, *optional*):
            # 是否返回所有层的隐藏状态。查看返回的张量中的 `hidden_states` 以获取更多详细信息。

        return_dict (`bool`, *optional*):
            # 是否返回 [`~utils.ModelOutput`] 而不是普通元组。
"""
DPR_READER_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`Tuple[torch.LongTensor]` of shapes `(n_passages, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. It has to be a sequence triplet with 1) the question
            and 2) the passages titles and 3) the passages texts To match pretraining, DPR `input_ids` sequence should
            be formatted with [CLS] and [SEP] with the format:

                `[CLS] <question token ids> [SEP] <titles ids> [SEP] <texts ids>`

            DPR is a model with absolute position embeddings so it's usually advised to pad the inputs on the right
            rather than the left.

            Indices can be obtained using [`DPRReaderTokenizer`]. See this class documentation for more details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.FloatTensor` of shape `(n_passages, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        inputs_embeds (`torch.FloatTensor` of shape `(n_passages, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare DPRContextEncoder transformer outputting pooler outputs as context representations.",
    DPR_START_DOCSTRING,
)
class DPRContextEncoder(DPRPretrainedContextEncoder):
    """
    DPRContextEncoder extends DPRPretrainedContextEncoder to encode context using DPR models.

    Args:
        config (DPRConfig): Configuration object specifying the model configuration.

    Attributes:
        config (DPRConfig): The configuration object used to initialize the model.
        ctx_encoder (DPREncoder): Encoder instance responsible for encoding contexts.

    Methods:
        post_init(): Initializes weights and applies final processing after initialization.
    """

    def __init__(self, config: DPRConfig):
        """
        Initializes a DPRContextEncoder instance.

        Args:
            config (DPRConfig): Configuration object specifying the model configuration.
        """
        super().__init__(config)
        self.config = config
        self.ctx_encoder = DPREncoder(config)
        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(DPR_ENCODERS_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=DPRContextEncoderOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        """
        Performs forward pass of the DPRContextEncoder model.

        Args:
            input_ids (Tuple[torch.LongTensor] of shapes `(n_passages, sequence_length)`, optional):
                Indices of input sequence tokens in the vocabulary.
            attention_mask (torch.FloatTensor of shape `(n_passages, sequence_length)`, optional):
                Mask to avoid performing attention on padding token indices.
            inputs_embeds (torch.FloatTensor of shape `(n_passages, sequence_length, hidden_size)`, optional):
                Optionally, directly pass an embedded representation instead of `input_ids`.
            output_attentions (bool, optional):
                Whether or not to return the attentions tensors of all attention layers.
            output_hidden_states (bool, optional):
                Whether or not to return the hidden states of all layers.
            return_dict (bool, optional):
                Whether or not to return a `ModelOutput` instead of a plain tuple.

        Returns:
            DPRContextEncoderOutput or tuple:
                The output of the DPRContextEncoder model.

        Raises:
            ValueError: If `input_ids` and `inputs_embeds` are both provided.
        """
        # Implementation of forward pass is handled by the superclass DPRPretrainedContextEncoder
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[DPRContextEncoderOutput, Tuple[Tensor, ...]]:
        r"""
        此方法定义了模型的前向传播逻辑，接受多个输入参数并返回相应的输出。

        Return:
            返回一个包含池化输出、隐藏状态、注意力权重的对象，具体取决于return_dict参数设置。

        Examples:

        ```
        >>> from transformers import DPRContextEncoder, DPRContextEncoderTokenizer

        >>> tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
        >>> model = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
        >>> input_ids = tokenizer("Hello, is my dog cute ?", return_tensors="pt")["input_ids"]
        >>> embeddings = model(input_ids).pooler_output
        ```"""

        # 确定是否输出注意力权重，默认使用配置中的设置
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 确定是否输出隐藏状态，默认使用配置中的设置
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 确定是否使用返回字典形式的输出，默认使用配置中的设置
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 检查输入参数的一致性
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        # 确定设备位置，根据输入参数选择
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # 如果未提供注意力掩码，则根据输入数据是否为填充令牌来生成
        if attention_mask is None:
            attention_mask = (
                torch.ones(input_shape, device=device)
                if input_ids is None
                else (input_ids != self.config.pad_token_id)
            )
        
        # 如果未提供token_type_ids，则默认为全零向量
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # 调用上下文编码器模型的前向传播
        outputs = self.ctx_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 如果不要求返回字典形式的输出，则返回元组形式的结果
        if not return_dict:
            return outputs[1:]
        # 否则返回自定义输出对象，包含池化输出、隐藏状态和注意力权重
        return DPRContextEncoderOutput(
            pooler_output=outputs.pooler_output, hidden_states=outputs.hidden_states, attentions=outputs.attentions
        )
# 为 DPRQuestionEncoder 类添加文档字符串，描述其作为问题表示的池化输出的基本功能
@add_start_docstrings(
    "The bare DPRQuestionEncoder transformer outputting pooler outputs as question representations.",
    DPR_START_DOCSTRING,
)
class DPRQuestionEncoder(DPRPretrainedQuestionEncoder):
    def __init__(self, config: DPRConfig):
        # 调用父类的初始化方法，传入配置对象
        super().__init__(config)
        # 存储配置对象到实例中
        self.config = config
        # 创建一个 DPREncoder 对象作为问题编码器
        self.question_encoder = DPREncoder(config)
        # 初始化权重并进行最终处理
        self.post_init()

    @add_start_docstrings_to_model_forward(DPR_ENCODERS_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=DPRQuestionEncoderOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[DPRQuestionEncoderOutput, Tuple[Tensor, ...]]:
        r"""
        定义函数的返回类型，可以是 DPRQuestionEncoderOutput 类型或者 Tensor 的元组类型

        Examples:

        ```
        >>> from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer

        >>> tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
        >>> model = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
        >>> input_ids = tokenizer("Hello, is my dog cute ?", return_tensors="pt")["input_ids"]
        >>> embeddings = model(input_ids).pooler_output
        ```
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 如果未指定 output_attentions，则使用 self.config 中的默认值

        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 如果未指定 output_hidden_states，则使用 self.config 中的默认值

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 如果未指定 return_dict，则使用 self.config 中的 use_return_dict 的值

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")
        # 根据输入的 input_ids 或 inputs_embeds 确定 input_shape，如果都未指定则报错

        device = input_ids.device if input_ids is not None else inputs_embeds.device
        # 根据 input_ids 或 inputs_embeds 确定设备类型

        if attention_mask is None:
            attention_mask = (
                torch.ones(input_shape, device=device)
                if input_ids is None
                else (input_ids != self.config.pad_token_id)
            )
        # 如果未指定 attention_mask，则根据 input_ids 是否为 None 来确定是否为 pad_token_id

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
        # 如果未指定 token_type_ids，则创建全零张量，形状与 input_shape 相同，数据类型为 long

        outputs = self.question_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # 调用 question_encoder 方法进行编码，根据参数进行不同的处理

        if not return_dict:
            return outputs[1:]
        # 如果 return_dict 为 False，则返回 outputs 的第二个元素之后的内容

        return DPRQuestionEncoderOutput(
            pooler_output=outputs.pooler_output, hidden_states=outputs.hidden_states, attentions=outputs.attentions
        )
        # 否则，返回一个包含 pooler_output、hidden_states 和 attentions 的 DPRQuestionEncoderOutput 对象
# 使用装饰器为类添加文档字符串，描述了该类的基本功能和用途
@add_start_docstrings(
    "The bare DPRReader transformer outputting span predictions.",
    DPR_START_DOCSTRING,
)
# 定义 DPRReader 类，继承自 DPRPretrainedReader 类
class DPRReader(DPRPretrainedReader):
    
    # 初始化方法，接受一个 DPRConfig 类型的参数 config
    def __init__(self, config: DPRConfig):
        # 调用父类的初始化方法，传入 config 参数
        super().__init__(config)
        # 将传入的 config 参数保存为类的属性
        self.config = config
        # 初始化一个 DPRSpanPredictor 实例作为类的属性 span_predictor
        self.span_predictor = DPRSpanPredictor(config)
        # 调用类的 post_init 方法，用于初始化权重和应用最终处理
        self.post_init()

    # 前向传播方法，接受多个输入参数并返回输出结果
    @add_start_docstrings_to_model_forward(DPR_READER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=DPRReaderOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        # 方法接受的输入参数详细文档，包括输入类型和说明
        # 输出返回值类型的文档替换为 DPRReaderOutput 类型，使用 _CONFIG_FOR_DOC 配置类
        ) -> Union[DPRReaderOutput, Tuple[Tensor, ...]]:
        r"""
        返回预测结果或包含多个张量的元组。

        Examples:

        ```
        >>> from transformers import DPRReader, DPRReaderTokenizer

        >>> tokenizer = DPRReaderTokenizer.from_pretrained("facebook/dpr-reader-single-nq-base")
        >>> model = DPRReader.from_pretrained("facebook/dpr-reader-single-nq-base")
        >>> encoded_inputs = tokenizer(
        ...     questions=["What is love ?"],
        ...     titles=["Haddaway"],
        ...     texts=["'What Is Love' is a song recorded by the artist Haddaway"],
        ...     return_tensors="pt",
        ... )
        >>> outputs = model(**encoded_inputs)
        >>> start_logits = outputs.start_logits
        >>> end_logits = outputs.end_logits
        >>> relevance_logits = outputs.relevance_logits
        ```
        """
        # 根据输入的输出参数设置是否返回注意力权重
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 根据输入的输出参数设置是否返回隐藏状态
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 根据输入的输出参数设置是否返回字典形式的输出
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 如果同时指定了 input_ids 和 inputs_embeds，则抛出错误
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        # 如果只指定了 input_ids，则检查是否需要警告没有指定 attention_mask
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
        # 如果只指定了 inputs_embeds，则获取其形状
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        # 如果既没有指定 input_ids 也没有指定 inputs_embeds，则抛出错误
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        # 确定使用的设备，根据是否有 input_ids 来决定
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # 如果没有提供 attention_mask，则创建一个全为1的张量作为默认的 attention_mask
        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)

        # 调用 span_predictor 方法进行预测并返回结果
        return self.span_predictor(
            input_ids,
            attention_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
```