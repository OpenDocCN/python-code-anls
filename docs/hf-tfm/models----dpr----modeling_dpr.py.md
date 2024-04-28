# `.\models\dpr\modeling_dpr.py`

```py
# 设置文件编码格式为utf-8
# 版权声明及许可信息
# 导入所需模块和库
# 定义了Open Domain Question Answering的PyTorch DPR模型
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
from torch import Tensor, nn
from ...modeling_outputs import BaseModelOutputWithPooling
from ...modeling_utils import PreTrainedModel
from ...utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from ..bert.modeling_bert import BertModel
from .configuration_dpr import DPRConfig
# 获取logger对象
logger = logging.get_logger(__name__)

# 定义用于文档的配置和检查点名称
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

# 定义DPRContextEncoderOutput类，表示DPRContextEncoder的输出结果
@dataclass
class DPRContextEncoderOutput(ModelOutput):
    """
    Class for outputs of [`DPRQuestionEncoder`].
    # 函数的参数，表示DPR编码器输出的“pooler_output”，对应于上下文表示。是通过线性层进一步处理的序列中第一个标记（分类标记）的最后一层隐藏状态。
    # 该输出将用于嵌入上下文，用于使用问题嵌入进行最近邻查询。
    pooler_output: torch.FloatTensor
    # 函数的参数，表示模型在每一层的输出的隐藏状态，可以选择返回该参数，当传递参数“output_hidden_states=True”或者“config.output_hidden_states=True”时返回
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # 函数的参数，表示模型在每一层的输出的注意力权重，可以选择返回该参数，当传递参数“output_attentions=True”或者“config.output_attentions=True”时返回
    attentions: Optional[Tuple[torch.FloatTensor]] = None
# 使用 @dataclass 装饰器来定义数据类 DPRQuestionEncoderOutput，用于存储 DPRQuestionEncoder 的输出结果
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

    # 代表池化输出，对应于问题表示，DPR 编码器输出用于嵌入问题以进行最接近邻居查询
    pooler_output: torch.FloatTensor
    # 隐藏状态，用于存储每个层的输出的元组
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # 注意力权重，用于存储每个层的注意力权重输出的元组
    attentions: Optional[Tuple[torch.FloatTensor]] = None


# 使用 @dataclass 装饰器来定义数据类 DPRReaderOutput，用于存储 DPRReader 的输出结果
@dataclass
class DPRReaderOutput(ModelOutput):
    """
    Class for outputs of [`DPRQuestionEncoder`].
    # 参数：start_logits（torch.FloatTensor，形状为`(n_passages, sequence_length)`）：
    #     每个段落开始索引的 logits。
    start_logits: torch.FloatTensor
    # 参数：end_logits（torch.FloatTensor，形状为`(n_passages, sequence_length)`）：
    #     每个段落结束索引的 logits。
    end_logits: torch.FloatTensor = None
    # 参数：relevance_logits（torch.FloatTensor，形状为`(n_passages, )`）：
    #     DPRReader 的 QA 分类器的输出，对应于每个段落回答问题的得分，与所有其他段落相比较。
    relevance_logits: torch.FloatTensor = None
    # 参数：hidden_states（tuple(torch.FloatTensor)，*可选*，当传递`output_hidden_states=True`时返回或者当`config.output_hidden_states=True`时返回）：
    #     一个元组的`torch.FloatTensor`（一个用于嵌入的输出 + 一个用于每一层的输出），
    #     形状为`(batch_size, sequence_length, hidden_size)`。
    #
    #     模型在每一层的隐藏状态以及初始嵌入输出。
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # 参数：attentions（tuple(torch.FloatTensor)，*可选*，当传递`output_attentions=True`时返回或者当`config.output_attentions=True`时返回）：
    #     一个元组的`torch.FloatTensor`（每一层一个），形状为`(batch_size, num_heads, sequence_length, sequence_length)`。
    #
    #     注意力 softmax 后的注意力权重，用于计算自注意力头中的加权平均值。
    attentions: Optional[Tuple[torch.FloatTensor]] = None
# 定义一个继承自PreTrainedModel的DPRPreTrainedModel类
class DPRPreTrainedModel(PreTrainedModel):
    # 初始化模型权重的方法
    def _init_weights(self, module):
        """Initialize the weights"""
        # 如果是线性层，则初始化权重和偏置
        if isinstance(module, nn.Linear):
            # 与 TF 版本略有不同，使用正态分布初始化
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        # 如果是嵌入层，则初始化权重
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果有填充索引，请将对应位置的权重初始化为0
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        # 如果是LayerNorm层，则初始化偏置为0，权重为1
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

# 定义一个继承自DPRPreTrainedModel的DPRSpanPredictor类
class DPRSpanPredictor(DPRPreTrainedModel):
    base_model_prefix = "encoder"  # 指定基础模型前缀为"encoder"
    def __init__(self, config: DPRConfig):
        super().__init__(config)
        # 创建一个 DPREncoder 对象，用来编码输入数据
        self.encoder = DPREncoder(config)
        # 定义一个线性层，用来生成问题回答的开始和结束的 logits
        self.qa_outputs = nn.Linear(self.encoder.embeddings_size, 2)
        # 定义一个线性层，用来生成问题和段落之间的相关性 logits
        self.qa_classifier = nn.Linear(self.encoder.embeddings_size, 1)
        # 初始化权重并且应用最终处理
        self.post_init()

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        inputs_embeds: Optional[Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = False,
    ) -> Union[DPRReaderOutput, Tuple[Tensor, ...]]:
        # 定义符号：N - 问题的批次数量，M - 每个问题的段落数量，L - 序列的长度
        # 获取段落的数量和序列的长度
        n_passages, sequence_length = input_ids.size() if input_ids is not None else inputs_embeds.size()[:2]
        # 将输入的参数传递给编码器
        outputs = self.encoder(
            input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # 获取编码器的输出序列
        sequence_output = outputs[0]

        # 计算开始和结束的 logits
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        # 移除 logits 的最后一个维度，并且使得其连续存储
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()
        # 计算问题和段落之间的相关性(logits)
        relevance_logits = self.qa_classifier(sequence_output[:, 0, :])

        # 重新调整 logits 的大小
        start_logits = start_logits.view(n_passages, sequence_length)
        end_logits = end_logits.view(n_passages, sequence_length)
        relevance_logits = relevance_logits.view(n_passages)

        if not return_dict:
            # 若返回的不是字典，则返回 logits 和其他输出
            return (start_logits, end_logits, relevance_logits) + outputs[2:]

        return DPRReaderOutput(
            # 若返回的是字典，则返回 logits 和其他输出
            start_logits=start_logits,
            end_logits=end_logits,
            relevance_logits=relevance_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
##################
# PreTrainedModel
##################


class DPRPretrainedContextEncoder(DPRPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 指定配置类
    config_class = DPRConfig
    # 不加载 TensorFlow 权重
    load_tf_weights = None
    # 基础模型前缀为 "ctx_encoder"
    base_model_prefix = "ctx_encoder"


class DPRPretrainedQuestionEncoder(DPRPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 指定配置类
    config_class = DPRConfig
    # 不加载 TensorFlow 权重
    load_tf_weights = None
    # 基础模型前缀为 "question_encoder"
    base_model_prefix = "question_encoder"


class DPRPretrainedReader(DPRPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 指定配置类
    config_class = DPRConfig
    # 不加载 TensorFlow 权重
    load_tf_weights = None
    # 基础模型前缀为 "span_predictor"
    base_model_prefix = "span_predictor"


###############
# Actual Models
###############


DPR_START_DOCSTRING = r"""

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`DPRConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

DPR_ENCODERS_INPUTS_DOCSTRING = r"""



注释：
    # 定义函数参数和参数类型
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            # input_ids是输入序列标记在词汇表中的索引。为了匹配预训练模型，DPR输入序列应该以以下方式格式化，包括[CLS]和[SEP]标记：
            # 对于序列对（例如标题+文本对）：
            # tokens:         [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
            # token_type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
            # 对于单个序列（例如问题）：
            # tokens:         [CLS] the dog is hairy . [SEP]
            # token_type_ids:   0   0   0   0  0     0   0
            # DPR是一个带有绝对位置嵌入的模型，因此通常建议在右侧而不是左侧进行填充输入。
            # 可以使用AutoTokenizer获取索引。有关详情，请参阅PreTrainedTokenizer.encode和PreTrainedTokenizer.__call__

        attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            # 避免对填充标记索引执行注意力的遮罩。掩码值选择在[0, 1]范围内：
            # - 1表示**未遮蔽**的标记，
            # - 0表示**已遮蔽**的标记。

        token_type_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            # 段标记索引，指示输入的第一部分和第二部分。索引在[0,1]中选择：
            # - 0对应于*句子A*标记，
            # - 1对应于*句子B*标记。

        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            # 可选地，可以直接传递嵌入表示，而不是传递input_ids。如果要更好地控制如何将input_ids索引转换为相关联向量，则这很有用，而不是使用模型的内部嵌入查找矩阵。

        output_attentions (`bool`, *optional*):
            # 是否返回所有注意力层的注意力张量。有关更多详细信息，请参见返回张量下的attentions。

        output_hidden_states (`bool`, *optional*):
            # 是否返回所有层的隐藏状态。有关更多详细信息，请参见返回张量下的hidden_states。

        return_dict (`bool`, *optional*):
            # 是否返回[`~utils.ModelOutput`]而不是普通元组。
# 定义了文档字符串，用于描述函数的参数和返回值
DPR_READER_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`Tuple[torch.LongTensor]` of shapes `(n_passages, sequence_length)`):
            表示输入序列标记在词汇表中的索引。必须是一个包含三个元素的元组，分别是1）问题 2）段落标题和3）段落文本。为了匹配预训练，DPR `input_ids` 序列应该格式化为[CLS]和[SEP]，格式如下：

                `[CLS] <question token ids> [SEP] <titles ids> [SEP] <texts ids>`

            DPR 是一个带有绝对位置嵌入的模型，所以通常建议在右侧而不是左侧对输入进行填充。

            可以使用 [`DPRReaderTokenizer`] 来获得这些索引。更多详细信息请参阅该类的文档。

            [什么是输入 ID？](../glossary#input-ids)
        attention_mask (`torch.FloatTensor` of shape `(n_passages, sequence_length)`, *optional*):
            用于避免在填充标记索引上执行注意力的遮罩。掩码值选在 `[0, 1]`：

            - 1表示**未被遮罩**的标记，
            - 0表示**被遮罩**的标记。

            [什么是注意力遮罩？](../glossary#attention-mask)
        inputs_embeds (`torch.FloatTensor` of shape `(n_passages, sequence_length, hidden_size)`, *optional*):
            可选参数，可以直接传递嵌入的表示。如果您想更多地控制如何将 `input_ids` 索引转换为关联向量，而不是模型的内部嵌入查找矩阵，这是有用的。
        output_attentions (`bool`, *optional*):
            是否返回所有注意力层的注意力张量。有关更多详细信息，请参阅返回张量中的 `attentions`。
        output_hidden_states (`bool`, *optional*):
            是否返回所有层的隐藏状态。有关更多详细信息，请参阅返回张量中的 `hidden_states`。
        return_dict (`bool`, *optional*):
            是否返回 [`~utils.ModelOutput`] 而不是普通元组。
"""

# 定义了一个类，继承自DPRPretrainedContextEncoder，并添加了文档字符串
@add_start_docstrings(
    "The bare DPRContextEncoder transformer outputting pooler outputs as context representations.",
    DPR_START_DOCSTRING,
)
class DPRContextEncoder(DPRPretrainedContextEncoder):

    # 定义了初始化方法，初始化了配置和上下文编码器
    def __init__(self, config: DPRConfig):
        super().__init__(config)
        self.config = config
        self.ctx_encoder = DPREncoder(config)
        # 初始化权重并应用最终处理
        self.post_init()

    # 添加了模型前向方法的文档字符串
    @add_start_docstrings_to_model_forward(DPR_ENCODERS_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=DPRContextEncoderOutput, config_class=_CONFIG_FOR_DOC)
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
        前向传播函数，用于模型推理

        Return:
            返回模型输出

        Examples:

        ```python
        >>> from transformers import DPRContextEncoder, DPRContextEncoderTokenizer

        >>> tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
        >>> model = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
        >>> input_ids = tokenizer("Hello, is my dog cute ?", return_tensors="pt")["input_ids"]
        >>> embeddings = model(input_ids).pooler_output
        ```py"""

        # 根据参数设置是否输出注意力权重，默认使用模型配置中的设置
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 根据参数设置是否输出隐藏状态，默认使用模型配置中的设置
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 根据参数设置是否返回字典格式的输出，默认使用模型配置中的设置
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 检查输入参数的有效性
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        # 获取输入数据的设备信息
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # 如果未提供注意力掩码，则创建一个默认的掩码
        if attention_mask is None:
            attention_mask = (
                torch.ones(input_shape, device=device)
                if input_ids is None
                else (input_ids != self.config.pad_token_id)
            )
        # 如果未提供标记类型 ID，则创建一个全零的标记类型 ID
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # 调用上下文编码器进行前向传播
        outputs = self.ctx_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 根据返回格式决定输出
        if not return_dict:
            return outputs[1:]  # 返回隐藏状态和注意力权重
        return DPRContextEncoderOutput(
            pooler_output=outputs.pooler_output, hidden_states=outputs.hidden_states, attentions=outputs.attentions
        )
# 添加起始文档字符串，描述了DPRQuestionEncoder的作用，输出池化后的输出作为问题表示
# 包含了DPR_START_DOCSTRING提供的初始文档字符串
@add_start_docstrings(
    "The bare DPRQuestionEncoder transformer outputting pooler outputs as question representations.",
    DPR_START_DOCSTRING,
)
# 定义DPRQuestionEncoder类，继承自DPRPretrainedQuestionEncoder类
class DPRQuestionEncoder(DPRPretrainedQuestionEncoder):
    # 初始化函数，接受一个DPRConfig类型的参数
    def __init__(self, config: DPRConfig):
        # 调用父类的初始化函数
        super().__init__(config)
        # 保存传入的配置参数
        self.config = config
        # 创建问题编码器，使用给定的配置
        self.question_encoder = DPREncoder(config)
        # 初始化权重并应用最终处理
        self.post_init()

    # 前向传播函数
    @add_start_docstrings_to_model_forward(DPR_ENCODERS_INPUTS_DOCSTRING)
    # 替换返回文档字符串，指定输出类型为DPRQuestionEncoderOutput，配置类为_CONFIG_FOR_DOC
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
        Return:

        Examples:

        ```python
        >>> from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer

        >>> tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
        >>> model = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
        >>> input_ids = tokenizer("Hello, is my dog cute ?", return_tensors="pt")["input_ids"]
        >>> embeddings = model(input_ids).pooler_output
        ```py
        """
        # 如果输出注意力权重参数不为None，则使用参数值，否则使用配置中的默认值
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 如果输出隐藏状态参数不为None，则使用参数值，否则使用配置中的默认值
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 如果返回字典参数不为None，则使用参数值，否则使用配置中的默认值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 如果既指定了input_ids又指定了inputs_embeds，则引发错误
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        # 如果指定了input_ids
        elif input_ids is not None:
            # 如果input_ids含有padding并且没有提供attention_mask，则发出警告
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            # 获取input_ids的形状
            input_shape = input_ids.size()
        # 如果指定了inputs_embeds
        elif inputs_embeds is not None:
            # 获取inputs_embeds的形状，不包括最后一个维度（即batch维度）
            input_shape = inputs_embeds.size()[:-1]
        else:
            # 如果既未指定input_ids又未指定inputs_embeds，则引发错误
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        # 获取设备信息，如果指定了input_ids，则使用其设备信息，否则使用inputs_embeds的设备信息
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # 如果没有提供attention_mask，则创建一个全1的张量，形状与input_shape相同，如果没有提供input_ids，则使用pad_token_id进行判断
        if attention_mask is None:
            attention_mask = (
                torch.ones(input_shape, device=device)
                if input_ids is None
                else (input_ids != self.config.pad_token_id)
            )
        # 如果没有提供token_type_ids，则创建一个全0的长整型张量，形状与input_shape相同
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # 调用question_encoder进行编码
        outputs = self.question_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 如果return_dict为False，则返回编码结果的pooler_output和hidden_states
        if not return_dict:
            return outputs[1:]
        # 如果return_dict为True，则返回一个DPRQuestionEncoderOutput对象，包含pooler_output、hidden_states和attentions
        return DPRQuestionEncoderOutput(
            pooler_output=outputs.pooler_output, hidden_states=outputs.hidden_states, attentions=outputs.attentions
        )
# 在DPRReader类上添加文档字符串，描述了该类的作用
@add_start_docstrings(
    "The bare DPRReader transformer outputting span predictions.",
    DPR_START_DOCSTRING,
)
class DPRReader(DPRPretrainedReader):
    # 初始化方法，接受一个config参数
    def __init__(self, config: DPRConfig):
        # 调用父类(DPRPretrainedReader)的初始化方法
        super().__init__(config)
        # 保存config参数
        self.config = config
        # 创建DPRSpanPredictor对象，保存到self.span_predictor属性中
        self.span_predictor = DPRSpanPredictor(config)
        # 调用post_init方法进行权重初始化和最终处理
        # 初始化权重和应用最终处理
        self.post_init()

    # 前向传播方法，接受多个输入参数，并且有多个可选参数
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
    ) -> Union[DPRReaderOutput, Tuple[Tensor, ...]]:
        r"""
        Return:
        
        返回值为DPRReaderOutput或者包含张量的元组

        Examples:

        示例代码，展示如何使用DPRReader模型进行问答

        ```python
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
        ```py

        """
        # 设置是否输出注意力的参数
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 设置是否输出隐藏状态的参数
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 设置是否返回字典的参数
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 如果同时提供了input_ids和inputs_embeds，则抛出错误
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        # 如果提供了input_ids，则检查是否需要警告，并获取input_shape
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
        # 如果提供了inputs_embeds，则获取input_shape
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        # 获取设备的信息
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # 如果没有提供attention_mask，则创建一个全1的attention_mask
        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)

        # 调用span_predictor方法进行跨度预测
        return self.span_predictor(
            input_ids,
            attention_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
```