# `.\models\dpr\modeling_tf_dpr.py`

```py
# 引入必要的库
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Union
import tensorflow as tf
# 引入模型输出相关的类
from ...modeling_tf_outputs import TFBaseModelOutputWithPooling
# 引入模型输入类型相关的类和函数
from ...modeling_tf_utils import TFModelInputType, TFPreTrainedModel, get_initializer, shape_list, unpack_inputs
# 引入工具函数和类
from ...utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
# 引入BERT模型的主要层
from ..bert.modeling_tf_bert import TFBertMainLayer
# 引入DPR配置
from .configuration_dpr import DPRConfig

# 获取日志记录器
logger = logging.get_logger(__name__)

# 用于文档的配置信息
_CONFIG_FOR_DOC = "DPRConfig"

# 预训练的上下文编码器模型存档列表
TF_DPR_CONTEXT_ENCODER_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/dpr-ctx_encoder-single-nq-base",
    "facebook/dpr-ctx_encoder-multiset-base",
]
# 预训练的问题编码器模型存档列表
TF_DPR_QUESTION_ENCODER_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/dpr-question_encoder-single-nq-base",
    "facebook/dpr-question_encoder-multiset-base",
]
# 预训练的阅读器模型存档列表
TF_DPR_READER_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/dpr-reader-single-nq-base",
    "facebook/dpr-reader-multiset-base",
]


##########
# Outputs
##########


@dataclass
class TFDPRContextEncoderOutput(ModelOutput):
    r"""
    Class for outputs of [`TFDPRContextEncoder`].
    Args:
        pooler_output (`tf.Tensor` of shape `(batch_size, embeddings_size)`):
            DPR编码器输出的*pooler_output*，对应于上下文表示。通过线性层进一步处理的序列的第一个标记（分类标记）的最后一层隐藏状态。
            该输出将用于将上下文嵌入到与问题嵌入进行最近邻查询的上下文中。
        hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            形状为`(batch_size, sequence_length, hidden_size)`的`tf.Tensor`元组（当传递`output_hidden_states=True`或`config.output_hidden_states=True`时返回）。

            每个层的模型在每个层的输出加上初始嵌入输出之后的隐藏状态。
        attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            形状为`(batch_size, num_heads, sequence_length, sequence_length)`的`tf.Tensor`元组（当传递`output_attentions=True`或`config.output_attentions=True`时返回）。

            在注意力softmax之后的注意力权重，用于在自注意力头中计算加权平均值。
    """

    pooler_output: tf.Tensor = None  # 用于存储DPR编码器输出的上下文表示
    hidden_states: Tuple[tf.Tensor] | None = None  # 用于存储模型每一层的隐藏状态
    attentions: Tuple[tf.Tensor] | None = None  # 用于存储注意力权重，用于计算自注意力头中的加权平均值
```  
from
    Args:
        start_logits (`tf.Tensor` of shape `(n_passages, sequence_length)`):
            每个段落中起始索引的逻辑值。
        end_logits (`tf.Tensor` of shape `(n_passages, sequence_length)`):
            每个段落中结束索引的逻辑值。
        relevance_logits (`tf.Tensor` of shape `(n_passages, )`):
            对应于DPRReader的QA分类器的输出，表示每个段落回答问题时的分数，与所有其他段落进行比较。
        hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            包含以下形状tf.Tensor元组（在传递output_hidden_states=True或config.output_hidden_states=True时返回）：
            （嵌入输出的一个 + 每个层的输出）的形状为(batch_size, sequence_length, hidden_size)的张量。

            模型的每一层输出的隐藏状态加上初始嵌入输出。
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            包含以下形状的torch.FloatTensor元组（在传递output_attentions=True或config.output_attentions=True时返回）：
            （每个层的一个）的形状为(batch_size, num_heads, sequence_length, sequence_length)的张量。

            注意力softmax后的注意力权重，用于计算自注意力头中的加权平均值。
    """

    start_logits: tf.Tensor = None
    end_logits: tf.Tensor = None
    relevance_logits: tf.Tensor = None
    hidden_states: Tuple[tf.Tensor] | None = None
    attentions: Tuple[tf.Tensor] | None = None
# 定义一个 TF-DPR 编码器层
class TFDPREncoderLayer(tf.keras.layers.Layer):
    # 基础模型的前缀
    base_model_prefix = "bert_model"

    def __init__(self, config: DPRConfig, **kwargs):
        # 调用父类的初始化方法
        super().__init__(**kwargs)

        # 解决与 TFBertMainLayer 的名称冲突，使用 TFBertMainLayer 而不是 TFBertModel
        self.bert_model = TFBertMainLayer(config, add_pooling_layer=False, name="bert_model")
        # 存储配置信息
        self.config = config

        # 检查隐藏层大小是否为非零值
        if self.config.hidden_size <= 0:
            raise ValueError("Encoder hidden_size can't be zero")
        # 如果投影维度大于零，则创建投影层
        self.projection_dim = config.projection_dim
        if self.projection_dim > 0:
            self.encode_proj = tf.keras.layers.Dense(
                config.projection_dim, kernel_initializer=get_initializer(config.initializer_range), name="encode_proj"
            )

    # 定义调用该层时执行的操作
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
        # 调用 BERT 模型
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

        # 获取序列输出和池化输出
        sequence_output = outputs[0]
        pooled_output = sequence_output[:, 0, :]
        # 如果有投影维度，则进行投影
        if self.projection_dim > 0:
            pooled_output = self.encode_proj(pooled_output)

        # 如果不返回字典，则返回输出元组
        if not return_dict:
            return (sequence_output, pooled_output) + outputs[1:]

        # 返回带有池化输出的 TFBaseModelOutputWithPooling 对象
        return TFBaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    # 返回嵌入大小
    @property
    def embeddings_size(self) -> int:
        if self.projection_dim > 0:
            return self.projection_dim
        return self.bert_model.config.hidden_size

    # 构建层
    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 如果存在 bert_model 属性，则构建它
        if getattr(self, "bert_model", None) is not None:
            with tf.name_scope(self.bert_model.name):
                self.bert_model.build(None)
        # 如果存在 encode_proj 属性，则构建它
        if getattr(self, "encode_proj", None) is not None:
            with tf.name_scope(self.encode_proj.name):
                self.encode_proj.build(None)


# 定义一个 TF-DPR Span 预测器层
class TFDPRSpanPredictorLayer(tf.keras.layers.Layer):
    # 基础模型的前缀
    base_model_prefix = "encoder"
    # 初始化方法，使用给定的配置参数初始化类实例
    def __init__(self, config: DPRConfig, **kwargs):
        # 调用父类初始化方法
        super().__init__(**kwargs)
        # 将配置参数保存到实例属性中
        self.config = config
        # 创建 TFDPREncoderLayer 对象，命名为"encoder"
        self.encoder = TFDPREncoderLayer(config, name="encoder")

        # 创建用于问答输出的全连接层，输出维度为2，使用给定的初始化方法初始化参数
        self.qa_outputs = tf.keras.layers.Dense(
            2, kernel_initializer=get_initializer(config.initializer_range), name="qa_outputs"
        )
        # 创建用于问答分类的全连接层，输出维度为1，使用给定的初始化方法初始化参数
        self.qa_classifier = tf.keras.layers.Dense(
            1, kernel_initializer=get_initializer(config.initializer_range), name="qa_classifier"
        )

    # 装饰器，用于解包输入，输入参数为调用方法时的输入
    @unpack_inputs
    # 定义 call 方法，接受一系列参数并返回 Union[TFDPRReaderOutput, Tuple[tf.Tensor, ...]] 类型对象
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
        # notations: N - number of questions in a batch, M - number of passages per questions, L - sequence length
        # 确定输入数据的维度
        n_passages, sequence_length = shape_list(input_ids) if input_ids is not None else shape_list(inputs_embeds)[:2]
        # 将输入数据传递给编码器进行处理
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )
        # 从输出中提取序列输出
        sequence_output = outputs[0]

        # 计算logits
        logits = self.qa_outputs(sequence_output)
        # 将logits拆分为start_logits和end_logits
        start_logits, end_logits = tf.split(logits, 2, axis=-1)
        # 去除多余的维度
        start_logits = tf.squeeze(start_logits, axis=-1)
        end_logits = tf.squeeze(end_logits, axis=-1)
        # 通过qa_classifier计算relevance_logits
        relevance_logits = self.qa_classifier(sequence_output[:, 0, :])

        # 重新调整tensor的形状
        start_logits = tf.reshape(start_logits, [n_passages, sequence_length])
        end_logits = tf.reshape(end_logits, [n_passages, sequence_length])
        relevance_logits = tf.reshape(relevance_logits, [n_passages])

        # 如果不要求返回字典形式的数据，则返回一系列tuple并附带额外的输出
        if not return_dict:
            return (start_logits, end_logits, relevance_logits) + outputs[2:]

        # 返回TFDPRReaderOutput对象，包括start_logits、end_logits、relevance_logits、hidden_states和attentions
        return TFDPRReaderOutput(
            start_logits=start_logits,
            end_logits=end_logits,
            relevance_logits=relevance_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    # 构建模型，如果已经构建过则直接返回
    def build(self, input_shape=None):
        if self.built:
            return
        # 设置标志表示模型已经构建
        self.built = True
        # 如果存在编码器，构建编码器
        if getattr(self, "encoder", None) is not None:
            # 使用编码器的名称创建命名空间
            with tf.name_scope(self.encoder.name):
                # 构建编码器，输入形状为None表示未指定输入形状
                self.encoder.build(None)
        # 如果存在qa_outputs，构建qa_outputs
        if getattr(self, "qa_outputs", None) is not None:
            # 使用qa_outputs的名称创建命名空间
            with tf.name_scope(self.qa_outputs.name):
                # 构建qa_outputs，输入形状为[None, None, self.encoder.embeddings_size]
                self.qa_outputs.build([None, None, self.encoder.embeddings_size])
        # 如果存在qa_classifier，构建qa_classifier
        if getattr(self, "qa_classifier", None) is not None:
            # 使用qa_classifier的名称创建命名空间
            with tf.name_scope(self.qa_classifier.name):
                # 构建qa_classifier，输入形状为[None, None, self.encoder.embeddings_size]
                self.qa_classifier.build([None, None, self.encoder.embeddings_size])
# 定义 TFDPRSpanPredictor 类，该类继承自 TFPreTrainedModel 类
class TFDPRSpanPredictor(TFPreTrainedModel):
    # 基础模型的前缀是 "encoder"
    base_model_prefix = "encoder"

    # 初始化函数，接受配置和额外参数
    def __init__(self, config: DPRConfig, **kwargs):
        # 调用父类的初始化函数
        super().__init__(config, **kwargs)
        # 创建 TFDPRSpanPredictorLayer 的实例，并将其分配给 encoder 属性
        self.encoder = TFDPRSpanPredictorLayer(config)

    # 定义 call 方法，接收输入数据并返回输出
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
        # 调用 encoder 的 call 方法并返回输出
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        # 返回 encoder 的输出
        return outputs

# 定义 TFDPREncoder 类，该类继承自 TFPreTrainedModel 类
class TFDPREncoder(TFPreTrainedModel):
    # 基础模型的前缀是 "encoder"
    base_model_prefix = "encoder"

    # 初始化函数，接受配置和额外参数
    def __init__(self, config: DPRConfig, **kwargs):
        # 调用父类的初始化函数
        super().__init__(config, **kwargs)

        # 创建 TFDPREncoderLayer 的实例，并将其分配给 encoder 属性
        self.encoder = TFDPREncoderLayer(config)

    # 定义 call 方法，接收输入数据并返回输出
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
        # 调用 encoder 的 call 方法并返回输出
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )
        # 返回 encoder 的输出
        return outputs

# 定义 TFDPRPretrainedContextEncoder 类，该类继承自 TFPreTrainedModel 类
class TFDPRPretrainedContextEncoder(TFPreTrainedModel):
    """
    一个抽象类，用于处理权重初始化以及下载和加载预训练模型的简单接口。
    """
    # 配置类为 DPRConfig
    config_class = DPRConfig
    # 基础模型的前缀是 "ctx_encoder"
    base_model_prefix = "ctx_encoder"

# 定义 TFDPRPretrainedQuestionEncoder 类，该类继承自 TFPreTrainedModel 类
class TFDPRPretrainedQuestionEncoder(TFPreTrainedModel):
    """
    一个抽象类，用于处理权重初始化以及下载和加载预训练模型的简单接口。
    """
    # 配置类为 DPRConfig
    config_class = DPRConfig
    # 基础模型的前缀是 "question_encoder"
    base_model_prefix = "question_encoder"

# 定义 TFDPRPretrainedReader 类，该类继承自 TFPreTrainedModel 类
class TFDPRPretrainedReader(TFPreTrainedModel):
    """
    一个抽象类，用于处理权重初始化以及下载和加载预训练模型的简单接口。
    """
    # 配置类为 DPRConfig
    config_class = DPRConfig
    # 基础模型的前缀是 "reader"
    base_model_prefix = "reader"

# 定义 TF_DPR_START_DOCSTRING 为模型的文档字符串
TF_DPR_START_DOCSTRING = r"""
    # 这个模型继承自[`TFPreTrainedModel`]。查看超类文档以了解库为所有模型实现的通用方法（例如下载或保存、调整输入嵌入、修剪头等）。

    # 这个模型也是一个Tensorflow [tf.keras.Model](https://www.tensorflow.org/api_docs/python/tf/keras/Model)子类。将其用作常规的TF 2.0 Keras模型，并参考TF 2.0文档，以了解所有与一般用法和行为相关的问题。

    # <提示>
    
    # `transformers`中的TensorFlow模型和层接受两种格式的输入：
    # 1. 将所有输入作为关键字参数（类似于PyTorch模型）
    # 2. 将所有输入作为列表、元组或字典的第一个位置参数。
    
    # 第二种格式受支持的原因是，当将输入传递给模型和层时，Keras方法更喜欢这种格式。由于有了这种支持，当使用`model.fit()`等方法时，应该会"正常工作" - 只需以`model.fit()`支持的任何格式传递输入和标签！
    # 但是，如果您想在Keras方法之外使用第二种格式，比如在使用Keras `Functional` API创建自己的层或模型时，则可以使用三种可能性来在第一个位置参数中收集所有输入张量：
    # - 仅包含`input_ids`和没有其他内容的单个张量：`model(input_ids)`
    # - 包含一个或多个输入张量的长度可变的列表，顺序与docstring中给出的相同：`model([input_ids, attention_mask])`或`model([input_ids, attention_mask, token_type_ids])`
    # - 包含与docstring中给出的输入名称相关联的一个或多个输入张量的字典：`model({"input_ids": input_ids, "token_type_ids": token_type_ids})`

    # 请注意，在使用[子类化](https://keras.io/guides/making_new_layers_and_models_via_subclassing/)创建模型和层时，您无需担心其中的任何内容，因为您可以像传递给任何其他Python函数一样传递输入！

    # </提示>

    # 参数：
    #     config ([`DPRConfig`]): 具有模型所有参数的模型配置类。
    #         使用配置文件进行初始化不会加载与模型关联的权重，只加载配置。查看[`~TFPreTrainedModel.from_pretrained`]方法以加载模型权重。
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
        super().__init__(config, *args, **kwargs)
        self.ctx_encoder = TFDPREncoderLayer(config, name="ctx_encoder")
    # 获取输入的嵌入层对象
    def get_input_embeddings(self):
        # 尝试返回上下文编码器的BERT模型的输入嵌入层对象
        try:
            return self.ctx_encoder.bert_model.get_input_embeddings()
        # 如果发生属性错误，则构建模型并返回其BERT模型的输入嵌入层对象
        except AttributeError:
            self.build()
            return self.ctx_encoder.bert_model.get_input_embeddings()

    # 调用函数，接收一系列输入参数，并返回相应的输出
    @unpack_inputs
    @add_start_docstrings_to_model_forward(TF_DPR_ENCODERS_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFDPRContextEncoderOutput, config_class=_CONFIG_FOR_DOC)
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
        Return:

        Examples:

        ```py
        >>> from transformers import TFDPRContextEncoder, DPRContextEncoderTokenizer

        >>> tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
        >>> model = TFDPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base", from_pt=True)
        >>> input_ids = tokenizer("Hello, is my dog cute ?", return_tensors="tf")["input_ids"]
        >>> embeddings = model(input_ids).pooler_output
        ```
        """
        # 如果同时指定了input_ids和inputs_embeds，则引发值错误
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        # 如果只指定了input_ids，则计算其形状
        elif input_ids is not None:
            input_shape = shape_list(input_ids)
        # 如果只指定了inputs_embeds，则计算其形状
        elif inputs_embeds is not None:
            input_shape = shape_list(inputs_embeds)[:-1]
        # 如果既没有指定input_ids，也没有指定inputs_embeds，则引发值错误
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        # 如果attention_mask为None，则根据input_ids是否为None分配相应的值
        if attention_mask is None:
            attention_mask = (
                tf.ones(input_shape, dtype=tf.dtypes.int32)
                if input_ids is None
                else (input_ids != self.config.pad_token_id)
            )
        # 如果token_type_ids为None，则分配全为零的值
        if token_type_ids is None:
            token_type_ids = tf.zeros(input_shape, dtype=tf.dtypes.int32)

        # 使用输入参数调用上下文编码器函数
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

        # 如果return_dict为False，则返回除第一个元素外的所有元素
        if not return_dict:
            return outputs[1:]

        # 如果return_dict为True，则返回包含池化输出、隐藏状态和注意力分布的TFDPRContextEncoderOutput对象
        return TFDPRContextEncoderOutput(
            pooler_output=outputs.pooler_output, hidden_states=outputs.hidden_states, attentions=outputs.attentions
        )
    # 构建神经网络模型
    def build(self, input_shape=None):
        # 如果模型已经构建完成，则直接返回
        if self.built:
            return
        # 设置标记为已构建
        self.built = True
        # 检查是否存在上下文编码器，如果存在，则构建该编码器
        if getattr(self, "ctx_encoder", None) is not None:
            # 使用上下文编码器的名称作为命名空间
            with tf.name_scope(self.ctx_encoder.name):
                # 构建上下文编码器，传入参数为None
                self.ctx_encoder.build(None)
# 使用装饰器添加文档字符串，说明这是一个裸的DPRQuestionEncoder转换器，输出pooler输出作为问题表示
@add_start_docstrings(
    "The bare DPRQuestionEncoder transformer outputting pooler outputs as question representations.",
    TF_DPR_START_DOCSTRING,
)
# 定义TFDPRQuestionEncoder类，继承自TFDPRPretrainedQuestionEncoder类
class TFDPRQuestionEncoder(TFDPRPretrainedQuestionEncoder):
    # 定义初始化方法，接受配置参数和其他可选参数
    def __init__(self, config: DPRConfig, *args, **kwargs):
        # 调用父类的初始化方法
        super().__init__(config, *args, **kwargs)
        # 初始化question_encoder属性，使用TFDPREncoderLayer类的实例化对象，传入配置参数和名称
        self.question_encoder = TFDPREncoderLayer(config, name="question_encoder")

    # 定义获取输入嵌入的方法
    def get_input_embeddings(self):
        # 尝试返回question_encoder的bert_model的输入嵌入
        try:
            return self.question_encoder.bert_model.get_input_embeddings()
        # 如果出现AttributeError异常
        except AttributeError:
            # 构建question_encoder
            self.build()
            return self.question_encoder.bert_model.get_input_embeddings()

    # 使用装饰器添加文档字符串和替换返回值的文档字符串，说明细节模型的正向传播输入和输出
    @unpack_inputs
    @add_start_docstrings_to_model_forward(TF_DPR_ENCODERS_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFDPRQuestionEncoderOutput, config_class=_CONFIG_FOR_DOC)
    # 定义call方法，接受多个输入参数
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
    ) -> TFDPRQuestionEncoderOutput | Tuple[tf.Tensor, ...]:
        r"""
        Return:
        指定函数的返回类型注解

        Examples:
        示例用法
        
        ```py
        >>> from transformers import TFDPRQuestionEncoder, DPRQuestionEncoderTokenizer

        >>> tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
        >>> model = TFDPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base", from_pt=True)
        >>> input_ids = tokenizer("Hello, is my dog cute ?", return_tensors="tf")["input_ids"]
        >>> embeddings = model(input_ids).pooler_output
        ```
        """
        if input_ids is not None and inputs_embeds is not None:
            # 如果同时指定 input_ids 和 inputs_embeds，则抛出数值错误
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = shape_list(input_ids)
        elif inputs_embeds is not None:
            input_shape = shape_list(inputs_embeds)[:-1]
        else:
            # 如果没有指定 input_ids 或 inputs_embeds，则抛出数值错误
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if attention_mask is None:
            attention_mask = (
                tf.ones(input_shape, dtype=tf.dtypes.int32)
                if input_ids is None
                else (input_ids != self.config.pad_token_id)
            )
        if token_type_ids is None:
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
            return outputs[1:]
        return TFDPRQuestionEncoderOutput(
            pooler_output=outputs.pooler_output, hidden_states=outputs.hidden_states, attentions=outputs.attentions
        )

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "question_encoder", None) is not None:
            with tf.name_scope(self.question_encoder.name):
                self.question_encoder.build(None)
# 导入必要的库
@add_start_docstrings(
    "The bare DPRReader transformer outputting span predictions.",  # 添加起始文档字符串
    TF_DPR_START_DOCSTRING,  # 添加DPR模型的文档字符串
)
class TFDPRReader(TFDPRPretrainedReader):  # 定义TFDPRReader类，继承自TFDPRPretrainedReader基类
    def __init__(self, config: DPRConfig, *args, **kwargs):  # 初始化方法，接收DPRConfig类型的config参数和任意数量的位置参数和关键字参数
        super().__init__(config, *args, **kwargs)  # 使用super()调用父类的初始化方法
        self.span_predictor = TFDPRSpanPredictorLayer(config, name="span_predictor")  # 初始化span_predictor属性为TFDPRSpanPredictorLayer对象

    def get_input_embeddings(self):  # 定义获取输入嵌入向量的方法
        try:  # 异常处理
            return self.span_predictor.encoder.bert_model.get_input_embeddings()  # 返回span_predictor.encoder.bert_model的输入嵌入向量
        except AttributeError:  # 处理AttributeError异常
            self.build()  # 调用build方法
            return self.span_predictor.encoder.bert_model.get_input_embeddings()  # 返回span_predictor.encoder.bert_model的输入嵌入向量

    @unpack_inputs  # 使用unpack_inputs装饰器
    @add_start_docstrings_to_model_forward(TF_DPR_READER_INPUTS_DOCSTRING)  # 添加model_forward方法的起始文档字符串
    @replace_return_docstrings(output_type=TFDPRReaderOutput, config_class=_CONFIG_FOR_DOC)  # 替换返回值的文档字符串
    def call(  # 定义call方法
        self,
        input_ids: TFModelInputType | None = None,  # 输入的token IDs或None
        attention_mask: tf.Tensor | None = None,  # 注意力掩码张量或None
        inputs_embeds: tf.Tensor | None = None,  # 嵌入向量或None
        output_attentions: bool | None = None,  # 输出注意力权重的标志或None
        output_hidden_states: bool | None = None,  # 输出隐藏状态的标志或None
        return_dict: bool | None = None,  # 返回结果字典的标志或None
        training: bool = False,  # 是否处于训练模式
    ) -> TFDPRReaderOutput | Tuple[tf.Tensor, ...]:  # 方法的返回值类型
        r"""
        Return:  # 返回值的描述

        Examples:  # 示例

        ```py
        >>> from transformers import TFDPRReader, DPRReaderTokenizer  # 导入所需库

        >>> tokenizer = DPRReaderTokenizer.from_pretrained("facebook/dpr-reader-single-nq-base")  # 使用预训练的DPRReaderTokenizer
        >>> model = TFDPRReader.from_pretrained("facebook/dpr-reader-single-nq-base", from_pt=True)  # 从预训练模型创建TFDPRReader对象
        >>> encoded_inputs = tokenizer(  # 使用tokenizer对输入进行编码
        ...     questions=["What is love ?"],  # 问题
        ...     titles=["Haddaway"],  # 标题
        ...     texts=["'What Is Love' is a song recorded by the artist Haddaway"],  # 文本
        ...     return_tensors="tf",  # 返回TensorFlow张量
        ... )
        >>> outputs = model(encoded_inputs)  # 对编码后的输入进行模型预测
        >>> start_logits = outputs.start_logits  # 获取开始位置的logits
        >>> end_logits = outputs.end_logits  # 获取结束位置的logits
        >>> relevance_logits = outputs.relevance_logits  # 获取相关性的logits
        ```
        """
        if input_ids is not None and inputs_embeds is not None:  # 如果同时指定了input_ids和inputs_embeds
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")  # 抛出数值错误
        elif input_ids is not None:  # 否则，如果指定了input_ids
            input_shape = shape_list(input_ids)  # 获取input_ids的形状
        elif inputs_embeds is not None:  # 否则，如果指定了inputs_embeds
            input_shape = shape_list(inputs_embeds)[:-1]  # 获取inputs_embeds的形状（去掉最后一个维度）
        else:  # 否则（既没有input_ids也没有inputs_embeds）
            raise ValueError("You have to specify either input_ids or inputs_embeds")  # 抛出数值错误

        if attention_mask is None:  # 如果attention_mask为None
            attention_mask = tf.ones(input_shape, dtype=tf.dtypes.int32)  # 使用全1张量作为attention_mask

        return self.span_predictor(  # 调用span_predictor
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )
    # 构建模型，如果已经构建过，则直接返回
    def build(self, input_shape=None):
        if self.built:
            return
        # 设置标志表示模型已经构建
        self.built = True
        # 检查是否存在 span_predictor 属性，若存在则构建它
        if getattr(self, "span_predictor", None) is not None:
            # 使用 TensorFlow 的命名作用域为 span_predictor 构建模型
            with tf.name_scope(self.span_predictor.name):
                # 调用 span_predictor 对象的 build 方法进行构建
                self.span_predictor.build(None)
```