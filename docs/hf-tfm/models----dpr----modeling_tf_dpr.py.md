# `.\models\dpr\modeling_tf_dpr.py`

```py
# coding=utf-8
# 定义文件编码格式和版权信息

# 导入必要的模块和库
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Union

import tensorflow as tf

# 导入相关的模型输出类和工具函数
from ...modeling_tf_outputs import TFBaseModelOutputWithPooling
from ...modeling_tf_utils import TFModelInputType, TFPreTrainedModel, get_initializer, keras, shape_list, unpack_inputs
from ...utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from ..bert.modeling_tf_bert import TFBertMainLayer
from .configuration_dpr import DPRConfig

# 获取日志记录器对象
logger = logging.get_logger(__name__)

# 预定义一些用于文档生成的配置常量
_CONFIG_FOR_DOC = "DPRConfig"

# 预训练模型的存档列表
TF_DPR_CONTEXT_ENCODER_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/dpr-ctx_encoder-single-nq-base",
    "facebook/dpr-ctx_encoder-multiset-base",
]
TF_DPR_QUESTION_ENCODER_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/dpr-question_encoder-single-nq-base",
    "facebook/dpr-question_encoder-multiset-base",
]
TF_DPR_READER_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/dpr-reader-single-nq-base",
    "facebook/dpr-reader-multiset-base",
]

##########
# Outputs
##########

# 数据类，用于封装`TFDPRContextEncoder`的输出
@dataclass
class TFDPRContextEncoderOutput(ModelOutput):
    r"""
    Class for outputs of [`TFDPRContextEncoder`].
    """
    """
    Args:
        pooler_output (`tf.Tensor` of shape `(batch_size, embeddings_size)`):
            DPR编码器的输出，对应于上下文表示。是序列中第一个标记（分类标记）的最后一层隐藏状态，
            进一步由线性层处理。此输出用于嵌入上下文，以便使用问题嵌入进行最近邻查询。

        hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            `tf.Tensor`元组（当传递`output_hidden_states=True`或`config.output_hidden_states=True`时返回），
            包含形状为`(batch_size, sequence_length, hidden_size)`的张量。

            模型在每个层的输出隐藏状态，以及初始嵌入输出。

        attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            `tf.Tensor`元组（当传递`output_attentions=True`或`config.output_attentions=True`时返回），
            包含形状为`(batch_size, num_heads, sequence_length, sequence_length)`的张量。

            注意力 softmax 后的注意力权重，用于计算自注意力头中的加权平均值。

    """

    pooler_output: tf.Tensor = None
    hidden_states: Tuple[tf.Tensor, ...] | None = None
    attentions: Tuple[tf.Tensor, ...] | None = None
# 使用 `dataclass` 装饰器定义一个数据类，用于存储 `TFDPRQuestionEncoder` 的输出结果
@dataclass
class TFDPRQuestionEncoderOutput(ModelOutput):
    """
    Class for outputs of [`TFDPRQuestionEncoder`].

    Args:
        pooler_output (`tf.Tensor` of shape `(batch_size, embeddings_size)`):
            The DPR encoder outputs the *pooler_output* that corresponds to the question representation. Last layer
            hidden-state of the first token of the sequence (classification token) further processed by a Linear layer.
            This output is to be used to embed questions for nearest neighbors queries with context embeddings.
        hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `tf.Tensor` (one for the output of the embeddings + one for the output of each layer) of shape
            `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    # 定义属性 `pooler_output`，类型为 `tf.Tensor`，用于存储 DPR 编码器对问题的池化输出
    pooler_output: tf.Tensor = None
    # 定义属性 `hidden_states`，类型为元组 `Tuple[tf.Tensor, ...]` 或 `None`，存储模型每一层的隐藏状态
    hidden_states: Tuple[tf.Tensor, ...] | None = None
    # 定义属性 `attentions`，类型为元组 `Tuple[tf.Tensor, ...]` 或 `None`，存储模型每一层的注意力权重
    attentions: Tuple[tf.Tensor, ...] | None = None


# 使用 `dataclass` 装饰器定义一个数据类，用于存储 `TFDPRReaderEncoder` 的输出结果
@dataclass
class TFDPRReaderOutput(ModelOutput):
    """
    Class for outputs of [`TFDPRReaderEncoder`].
    """

    # 这里省略了具体的属性定义，根据文档需求，应包含与 `TFDPRQuestionEncoderOutput` 类似的属性定义
    # `start_logits` 是一个 Tensor，形状为 `(n_passages, sequence_length)`，包含每个段落中起始索引的预测值
    start_logits: tf.Tensor = None
    # `end_logits` 是一个 Tensor，形状为 `(n_passages, sequence_length)`，包含每个段落中结束索引的预测值
    end_logits: tf.Tensor = None
    # `relevance_logits` 是一个 Tensor，形状为 `(n_passages, )`，包含每个段落对于问题的相关性预测分数
    relevance_logits: tf.Tensor = None
    # `hidden_states` 是一个可选的元组，包含多个 Tensor，形状为 `(batch_size, sequence_length, hidden_size)`，表示模型在每一层输出的隐藏状态以及初始嵌入输出
    hidden_states: Tuple[tf.Tensor, ...] | None = None
    # `attentions` 是一个可选的元组，包含多个 Tensor，形状为 `(batch_size, num_heads, sequence_length, sequence_length)`，表示每层的注意力权重
    attentions: Tuple[tf.Tensor, ...] | None = None
# 定义 TF-DPR 编码器层
class TFDPREncoderLayer(keras.layers.Layer):
    # 基础模型前缀设定为 "bert_model"
    base_model_prefix = "bert_model"

    def __init__(self, config: DPRConfig, **kwargs):
        super().__init__(**kwargs)

        # 解决与 TFBertMainLayer 的名称冲突，使用 TFBertMainLayer 而不是 TFBertModel
        self.bert_model = TFBertMainLayer(config, add_pooling_layer=False, name="bert_model")
        self.config = config

        # 检查隐藏层大小是否为非正数，如果是则抛出 ValueError
        if self.config.hidden_size <= 0:
            raise ValueError("Encoder hidden_size can't be zero")
        
        # 设置投影维度为配置中的投影维度
        self.projection_dim = config.projection_dim
        
        # 如果投影维度大于 0，则创建投影层 Dense
        if self.projection_dim > 0:
            self.encode_proj = keras.layers.Dense(
                config.projection_dim, kernel_initializer=get_initializer(config.initializer_range), name="encode_proj"
            )

    # 解包输入参数装饰器，定义 call 方法
    @unpack_inputs
    def call(
        self,
        input_ids: tf.Tensor = None,
        attention_mask: tf.Tensor | None = None,
        token_type_ids: tf.Tensor | None = None,
        inputs_embeds: tf.Tensor | None = None,
        output_attentions: bool = None,
        output_hidden_states: bool = None,
        return_dict: bool = None,
        training: bool = False,
    ) -> Union[TFBaseModelOutputWithPooling, Tuple[tf.Tensor, ...]]:
        # 调用 bert_model 的 call 方法，传入参数并获取输出
        outputs = self.bert_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        # 提取序列输出和池化输出
        sequence_output = outputs[0]
        pooled_output = sequence_output[:, 0, :]  # 取序列输出的第一个 token 的输出作为池化输出
        
        # 如果有投影维度，则应用投影层到池化输出上
        if self.projection_dim > 0:
            pooled_output = self.encode_proj(pooled_output)

        # 如果不返回字典，则返回序列输出、池化输出和其他输出
        if not return_dict:
            return (sequence_output, pooled_output) + outputs[1:]

        # 返回 TFBaseModelOutputWithPooling 对象，包含最后的隐藏状态、池化输出、隐藏状态和注意力权重
        return TFBaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    # 嵌入大小的属性方法，返回投影维度或者 bert_model 配置中的隐藏大小
    @property
    def embeddings_size(self) -> int:
        if self.projection_dim > 0:
            return self.projection_dim
        return self.bert_model.config.hidden_size

    # 构建方法，构建 bert_model 和 encode_proj 层
    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        
        # 如果存在 bert_model，则在 bert_model 名称空间下构建
        if getattr(self, "bert_model", None) is not None:
            with tf.name_scope(self.bert_model.name):
                self.bert_model.build(None)
        
        # 如果存在 encode_proj，则在 encode_proj 名称空间下构建
        if getattr(self, "encode_proj", None) is not None:
            with tf.name_scope(self.encode_proj.name):
                self.encode_proj.build(None)


# 定义 TF-DPR 跨度预测器层
class TFDPRSpanPredictorLayer(keras.layers.Layer):
    # 基础模型前缀设定为 "encoder"
    base_model_prefix = "encoder"
    # 初始化函数，用于创建一个新的DPRReader对象
    def __init__(self, config: DPRConfig, **kwargs):
        # 调用父类的初始化函数，传递任何额外的关键字参数
        super().__init__(**kwargs)
        # 存储传入的配置对象
        self.config = config
        # 创建一个TFDPREncoderLayer对象作为编码器，并命名为"encoder"
        self.encoder = TFDPREncoderLayer(config, name="encoder")

        # 创建一个全连接层用于生成答案的起始和结束logits
        self.qa_outputs = keras.layers.Dense(
            2, kernel_initializer=get_initializer(config.initializer_range), name="qa_outputs"
        )
        # 创建一个全连接层用于生成问题与文本段落相关性的logits
        self.qa_classifier = keras.layers.Dense(
            1, kernel_initializer=get_initializer(config.initializer_range), name="qa_classifier"
        )

    # 使用装饰器unpack_inputs，对输入进行解包
    def call(
        self,
        input_ids: tf.Tensor = None,
        attention_mask: tf.Tensor | None = None,
        inputs_embeds: tf.Tensor | None = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = False,
        training: bool = False,
    ) -> Union[TFDPRReaderOutput, Tuple[tf.Tensor, ...]]:
        # 获取输入张量input_ids的形状，n_passages表示问题批次中的段落数，sequence_length表示序列的长度
        n_passages, sequence_length = shape_list(input_ids) if input_ids is not None else shape_list(inputs_embeds)[:2]
        
        # 将输入传递给编码器进行处理
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )
        # 从编码器的输出中获取序列输出
        sequence_output = outputs[0]

        # 计算起始和结束logits
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = tf.split(logits, 2, axis=-1)
        start_logits = tf.squeeze(start_logits, axis=-1)
        end_logits = tf.squeeze(end_logits, axis=-1)
        
        # 计算问题与文本段落相关性的logits
        relevance_logits = self.qa_classifier(sequence_output[:, 0, :])

        # 调整logits的形状
        start_logits = tf.reshape(start_logits, [n_passages, sequence_length])
        end_logits = tf.reshape(end_logits, [n_passages, sequence_length])
        relevance_logits = tf.reshape(relevance_logits, [n_passages])

        # 如果return_dict为False，则返回元组形式的输出
        if not return_dict:
            return (start_logits, end_logits, relevance_logits) + outputs[2:]

        # 如果return_dict为True，则返回TFDPRReaderOutput对象的形式
        return TFDPRReaderOutput(
            start_logits=start_logits,
            end_logits=end_logits,
            relevance_logits=relevance_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    # 构建方法用于构建模型的层和变量，如果已经构建过，则直接返回
    def build(self, input_shape=None):
        if self.built:
            return
        # 设置标志表示模型已构建
        self.built = True
        # 如果存在编码器(encoder)，则构建编码器的层和变量
        if getattr(self, "encoder", None) is not None:
            # 在命名作用域下构建编码器
            with tf.name_scope(self.encoder.name):
                self.encoder.build(None)
        # 如果存在问答输出(qa_outputs)，则构建其层和变量
        if getattr(self, "qa_outputs", None) is not None:
            # 在命名作用域下构建问答输出
            with tf.name_scope(self.qa_outputs.name):
                self.qa_outputs.build([None, None, self.encoder.embeddings_size])
        # 如果存在问答分类器(qa_classifier)，则构建其层和变量
        if getattr(self, "qa_classifier", None) is not None:
            # 在命名作用域下构建问答分类器
            with tf.name_scope(self.qa_classifier.name):
                self.qa_classifier.build([None, None, self.encoder.embeddings_size])
class TFDPRSpanPredictor(TFPreTrainedModel):
    base_model_prefix = "encoder"

    def __init__(self, config: DPRConfig, **kwargs):
        super().__init__(config, **kwargs)
        # 初始化编码器层，使用给定的配置参数
        self.encoder = TFDPRSpanPredictorLayer(config)

    @unpack_inputs
    def call(
        self,
        input_ids: tf.Tensor = None,
        attention_mask: tf.Tensor | None = None,
        token_type_ids: tf.Tensor | None = None,
        inputs_embeds: tf.Tensor | None = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = False,
        training: bool = False,
    ) -> Union[TFDPRReaderOutput, Tuple[tf.Tensor, ...]]:
        # 调用编码器层的call方法，传递参数并获取输出
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )
        return outputs


class TFDPREncoder(TFPreTrainedModel):
    base_model_prefix = "encoder"

    def __init__(self, config: DPRConfig, **kwargs):
        super().__init__(config, **kwargs)
        # 初始化编码器层，使用给定的配置参数
        self.encoder = TFDPREncoderLayer(config)

    @unpack_inputs
    def call(
        self,
        input_ids: tf.Tensor = None,
        attention_mask: tf.Tensor | None = None,
        token_type_ids: tf.Tensor | None = None,
        inputs_embeds: tf.Tensor | None = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = False,
        training: bool = False,
    ) -> Union[TFDPRReaderOutput, Tuple[tf.Tensor, ...]]:
        # 调用编码器层的call方法，传递参数并获取输出
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )
        return outputs


##################
# PreTrainedModel
##################


class TFDPRPretrainedContextEncoder(TFPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = DPRConfig
    base_model_prefix = "ctx_encoder"


class TFDPRPretrainedQuestionEncoder(TFPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = DPRConfig
    base_model_prefix = "question_encoder"


class TFDPRPretrainedReader(TFPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = DPRConfig
    base_model_prefix = "reader"


###############
# Actual Models
###############


TF_DPR_START_DOCSTRING = r"""
    # 此模型继承自 `TFPreTrainedModel`。请查看超类文档，了解库实现的通用方法，如下载或保存模型、调整输入嵌入大小、修剪头等。
    
    # 此模型还是一个 Tensorflow 的 `keras.Model` 子类。您可以将其用作常规的 TF 2.0 Keras 模型，并参考 TF 2.0 文档了解所有与一般用法和行为相关的事项。
    
    # <Tip>
    
    # 在 `transformers` 中，TensorFlow 模型和层接受两种输入格式：
    
    # - 将所有输入作为关键字参数（类似于 PyTorch 模型），
    # - 将所有输入作为列表、元组或字典传递给第一个位置参数。
    
    # 支持第二种格式的原因是，Keras 方法在将输入传递给模型和层时更倾向于此格式。因此，在使用 `model.fit()` 等方法时，只需将输入和标签以 `model.fit()` 支持的任何格式传递即可！但是，如果您想在 Keras `Functional` API 中创建自己的层或模型时使用第二种格式，比如在创建自己的层或模型时，可以使用以下三种可能性将所有输入张量收集到第一个位置参数中：
    
    # - 只有 `input_ids` 的单个张量：`model(input_ids)`
    # - 长度不同的列表，按照文档字符串中给定的顺序包含一个或多个输入张量：`model([input_ids, attention_mask])` 或 `model([input_ids, attention_mask, token_type_ids])`
    # - 一个字典，其中包含一个或多个输入张量，与文档字符串中给定的输入名称关联：`model({"input_ids": input_ids, "token_type_ids": token_type_ids})`
    
    # 注意，如果使用 [subclassing](https://keras.io/guides/making_new_layers_and_models_via_subclassing/) 创建模型和层，则无需担心这些问题，因为可以像对任何其他 Python 函数一样传递输入！
    
    # </Tip>
    
    # 参数:
    #     config ([`DPRConfig`]): 包含模型所有参数的模型配置类。
    #         使用配置文件初始化不会加载与模型关联的权重，仅加载配置。查看 [`~TFPreTrainedModel.from_pretrained`] 方法以加载模型权重。
"""

TF_DPR_ENCODERS_INPUTS_DOCSTRING = r"""
"""

TF_DPR_READER_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`Numpy array` or `tf.Tensor` of shapes `(n_passages, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. It has to be a sequence triplet with 1) the question
            and 2) the passages titles and 3) the passages texts To match pretraining, DPR `input_ids` sequence should
            be formatted with [CLS] and [SEP] with the format:

                `[CLS] <question token ids> [SEP] <titles ids> [SEP] <texts ids>`

            DPR is a model with absolute position embeddings so it's usually advised to pad the inputs on the right
            rather than the left.

            Indices can be obtained using [`DPRReaderTokenizer`]. See this class documentation for more details.
        attention_mask (`Numpy array` or `tf.Tensor` of shape `(n_passages, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        inputs_embeds (`Numpy array` or `tf.Tensor` of shape `(n_passages, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail. This argument can be used only in eager mode, in graph mode the value in the config will be
            used instead.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple. This argument can be used in
            eager mode, in graph mode the value will always be set to True.
        training (`bool`, *optional*, defaults to `False`):
            Whether or not to use the model in training mode (some modules like dropout modules have different
            behaviors between training and evaluation).
"""


@add_start_docstrings(
    "The bare DPRContextEncoder transformer outputting pooler outputs as context representations.",
    TF_DPR_START_DOCSTRING,
)
class TFDPRContextEncoder(TFDPRPretrainedContextEncoder):
    def __init__(self, config: DPRConfig, *args, **kwargs):
        # 调用父类构造函数，传递配置和其他可选参数
        super().__init__(config, *args, **kwargs)
        # 创建一个上下文编码器层对象，使用给定的配置，命名为"ctx_encoder"
        self.ctx_encoder = TFDPREncoderLayer(config, name="ctx_encoder")
    # 尝试获取上下文编码器（Context Encoder）中 BERT 模型的输入嵌入
    try:
        return self.ctx_encoder.bert_model.get_input_embeddings()
    except AttributeError:
        # 如果属性错误，则调用 build 方法重新构建模型
        self.build()
        # 返回重新构建后的 BERT 模型的输入嵌入
        return self.ctx_encoder.bert_model.get_input_embeddings()

    # 调用方法的装饰器：解压输入
    @unpack_inputs
    # 调用方法的装饰器：将模型前向方法的文档字符串添加到模型中
    @add_start_docstrings_to_model_forward(TF_DPR_ENCODERS_INPUTS_DOCSTRING)
    # 调用方法的装饰器：替换返回值的文档字符串为指定的类型和配置类
    @replace_return_docstrings(output_type=TFDPRContextEncoderOutput, config_class=_CONFIG_FOR_DOC)
    # 定义模型的前向传播方法
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        attention_mask: tf.Tensor | None = None,
        token_type_ids: tf.Tensor | None = None,
        inputs_embeds: tf.Tensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        training: bool = False,
    ) -> TFDPRContextEncoderOutput | Tuple[tf.Tensor, ...]:
        r"""
        返回模型的输出：

        Examples:

        ```
        >>> from transformers import TFDPRContextEncoder, DPRContextEncoderTokenizer

        >>> tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
        >>> model = TFDPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base", from_pt=True)
        >>> input_ids = tokenizer("Hello, is my dog cute ?", return_tensors="tf")["input_ids"]
        >>> embeddings = model(input_ids).pooler_output
        ```
        """
        # 如果同时指定了 input_ids 和 inputs_embeds，则引发 ValueError
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        # 如果指定了 input_ids，则获取其形状
        elif input_ids is not None:
            input_shape = shape_list(input_ids)
        # 如果指定了 inputs_embeds，则获取其形状，去掉最后一个维度
        elif inputs_embeds is not None:
            input_shape = shape_list(inputs_embeds)[:-1]
        else:
            # 如果既未指定 input_ids 也未指定 inputs_embeds，则引发 ValueError
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        # 如果 attention_mask 为 None，则根据 input_ids 是否为 None，选择性地创建 attention_mask
        if attention_mask is None:
            attention_mask = (
                tf.ones(input_shape, dtype=tf.dtypes.int32)
                if input_ids is None
                else (input_ids != self.config.pad_token_id)
            )
        # 如果 token_type_ids 为 None，则创建与 input_shape 相同形状的全零 tensor
        if token_type_ids is None:
            token_type_ids = tf.zeros(input_shape, dtype=tf.dtypes.int32)

        # 调用上下文编码器（Context Encoder）的前向传播方法
        outputs = self.ctx_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        # 如果 return_dict 为 False，则返回 outputs 的所有元素，除去第一个元素
        if not return_dict:
            return outputs[1:]

        # 如果 return_dict 为 True，则构建 TFDPRContextEncoderOutput 对象，并返回
        return TFDPRContextEncoderOutput(
            pooler_output=outputs.pooler_output, hidden_states=outputs.hidden_states, attentions=outputs.attentions
        )
    # 定义一个方法 `build`，用于构建模型的层次结构
    def build(self, input_shape=None):
        # 如果模型已经构建完成，直接返回，不再重复构建
        if self.built:
            return
        # 将模型标记为已构建状态
        self.built = True
        # 如果存在上下文编码器 `ctx_encoder`
        if getattr(self, "ctx_encoder", None) is not None:
            # 在 TensorFlow 的命名作用域下，构建上下文编码器
            with tf.name_scope(self.ctx_encoder.name):
                # 调用上下文编码器的 `build` 方法，并传入 `None` 作为输入形状
                self.ctx_encoder.build(None)
# 使用装饰器添加文档字符串，描述了该类的基本信息以及继承的文档字符串内容
@add_start_docstrings(
    "The bare DPRQuestionEncoder transformer outputting pooler outputs as question representations.",
    TF_DPR_START_DOCSTRING,
)
# 定义一个 TF 版本的 DPRQuestionEncoder 类，继承自 TFDPRPretrainedQuestionEncoder 类
class TFDPRQuestionEncoder(TFDPRPretrainedQuestionEncoder):
    
    # 初始化方法，接受一个 DPRConfig 类型的配置对象以及其他参数和关键字参数
    def __init__(self, config: DPRConfig, *args, **kwargs):
        # 调用父类的初始化方法
        super().__init__(config, *args, **kwargs)
        # 创建一个 TFDPREncoderLayer 类的实例作为 question_encoder 属性
        self.question_encoder = TFDPREncoderLayer(config, name="question_encoder")

    # 获取输入嵌入的方法
    def get_input_embeddings(self):
        try:
            # 尝试获取 question_encoder 属性中 bert_model 的输入嵌入
            return self.question_encoder.bert_model.get_input_embeddings()
        except AttributeError:
            # 如果属性错误（即不存在），则调用 build 方法重新构建
            self.build()
            return self.question_encoder.bert_model.get_input_embeddings()

    # 使用装饰器为 call 方法添加文档字符串，描述输入参数和返回输出
    @unpack_inputs
    @add_start_docstrings_to_model_forward(TF_DPR_ENCODERS_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFDPRQuestionEncoderOutput, config_class=_CONFIG_FOR_DOC)
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        attention_mask: tf.Tensor | None = None,
        token_type_ids: tf.Tensor | None = None,
        inputs_embeds: tf.Tensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        training: bool = False,
        # 函数签名中省略部分，以保持完整性
    ) -> TFDPRQuestionEncoderOutput | Tuple[tf.Tensor, ...]:
        r"""
        指定该方法的返回类型为 TFDPRQuestionEncoderOutput 或包含 tf.Tensor 的元组

        Examples:

        ```
        >>> from transformers import TFDPRQuestionEncoder, DPRQuestionEncoderTokenizer

        >>> tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
        >>> model = TFDPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base", from_pt=True)
        >>> input_ids = tokenizer("Hello, is my dog cute ?", return_tensors="tf")["input_ids"]
        >>> embeddings = model(input_ids).pooler_output
        ```
        """
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = shape_list(input_ids)
        elif inputs_embeds is not None:
            input_shape = shape_list(inputs_embeds)[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if attention_mask is None:
            # 如果未提供 attention_mask，则根据输入的 shape 自动生成全为 1 的 attention_mask
            attention_mask = (
                tf.ones(input_shape, dtype=tf.dtypes.int32)
                if input_ids is None
                else (input_ids != self.config.pad_token_id)
            )
        if token_type_ids is None:
            # 如果未提供 token_type_ids，则生成一个全为 0 的 token_type_ids，与输入的 shape 相同
            token_type_ids = tf.zeros(input_shape, dtype=tf.dtypes.int32)

        outputs = self.question_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        if not return_dict:
            # 如果 return_dict 为 False，则返回输出中除第一个元素外的所有元素
            return outputs[1:]
        # 如果 return_dict 为 True，则以 TFDPRQuestionEncoderOutput 形式返回指定的输出
        return TFDPRQuestionEncoderOutput(
            pooler_output=outputs.pooler_output, hidden_states=outputs.hidden_states, attentions=outputs.attentions
        )

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "question_encoder", None) is not None:
            with tf.name_scope(self.question_encoder.name):
                # 构建 question_encoder 模型的网络结构
                self.question_encoder.build(None)
@add_start_docstrings(
    "The bare DPRReader transformer outputting span predictions.",
    TF_DPR_START_DOCSTRING,
)
class TFDPRReader(TFDPRPretrainedReader):
    def __init__(self, config: DPRConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        # 初始化span预测器，使用给定的配置
        self.span_predictor = TFDPRSpanPredictorLayer(config, name="span_predictor")

    def get_input_embeddings(self):
        try:
            # 尝试获取输入嵌入层
            return self.span_predictor.encoder.bert_model.get_input_embeddings()
        except AttributeError:
            # 如果属性错误，重新构建模型并返回输入嵌入层
            self.build()
            return self.span_predictor.encoder.bert_model.get_input_embeddings()

    @unpack_inputs
    @add_start_docstrings_to_model_forward(TF_DPR_READER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFDPRReaderOutput, config_class=_CONFIG_FOR_DOC)
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        attention_mask: tf.Tensor | None = None,
        inputs_embeds: tf.Tensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        training: bool = False,
    ) -> TFDPRReaderOutput | Tuple[tf.Tensor, ...]:
        r"""
        模型前向传播函数，接受多种输入参数，返回预测结果。

        Return:
            TFDPRReaderOutput或者一个元组包含tf.Tensor

        Examples:

        ```
        >>> from transformers import TFDPRReader, DPRReaderTokenizer

        >>> tokenizer = DPRReaderTokenizer.from_pretrained("facebook/dpr-reader-single-nq-base")
        >>> model = TFDPRReader.from_pretrained("facebook/dpr-reader-single-nq-base", from_pt=True)
        >>> encoded_inputs = tokenizer(
        ...     questions=["What is love ?"],
        ...     titles=["Haddaway"],
        ...     texts=["'What Is Love' is a song recorded by the artist Haddaway"],
        ...     return_tensors="tf",
        ... )
        >>> outputs = model(encoded_inputs)
        >>> start_logits = outputs.start_logits
        >>> end_logits = outputs.end_logits
        >>> relevance_logits = outputs.relevance_logits
        ```
        """
        if input_ids is not None and inputs_embeds is not None:
            # 如果同时指定了input_ids和inputs_embeds，则引发值错误异常
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            # 获取input_ids的形状列表
            input_shape = shape_list(input_ids)
        elif inputs_embeds is not None:
            # 获取inputs_embeds的形状列表，去掉最后一个维度
            input_shape = shape_list(inputs_embeds)[:-1]
        else:
            # 如果既没有指定input_ids也没有指定inputs_embeds，则引发值错误异常
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if attention_mask is None:
            # 如果attention_mask为None，则创建全1的张量作为attention_mask
            attention_mask = tf.ones(input_shape, dtype=tf.dtypes.int32)

        # 调用span_predictor的前向传播方法，传递所有参数并返回结果
        return self.span_predictor(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )
    # 定义模型的构建方法，参数 input_shape 可选，默认为 None
    def build(self, input_shape=None):
        # 如果模型已经构建过，则直接返回，避免重复构建
        if self.built:
            return
        # 标记模型为已构建状态
        self.built = True
        
        # 检查模型是否具有 span_predictor 属性，并且该属性不为 None
        if getattr(self, "span_predictor", None) is not None:
            # 在 TensorFlow 中使用命名空间来组织模型的组件，这里将使用 span_predictor 的名字空间
            with tf.name_scope(self.span_predictor.name):
                # 构建 span_predictor 组件，input_shape 参数这里传入 None
                self.span_predictor.build(None)
```