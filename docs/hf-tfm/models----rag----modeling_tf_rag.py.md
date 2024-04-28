# `.\transformers\models\rag\modeling_tf_rag.py`

```py
# 设置文件编码为 utf-8
# 版权声明

"""TFRAG model implementation."""  # 定义 TFRAG 模型实现

# 导入必要的库和模块
from __future__ import annotations
import copy
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import numpy as np
import tensorflow as tf
# 导入 HuggingFace 库中的一些模块和类
from ...configuration_utils import PretrainedConfig
from ...generation import TFLogitsProcessorList
from ...modeling_tf_utils import (
    TFCausalLanguageModelingLoss,
    TFModelInputType,
    TFPreTrainedModel,
    shape_list,
    unpack_inputs,
)
from ...utils import ModelOutput, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_rag import RagConfig
from .retrieval_rag import RagRetriever

# 获取日志记录器
logger = logging.get_logger(__name__)

# 文档中的配置
_CONFIG_FOR_DOC = "RagConfig"

# TFRetrievAugLMMarginOutput 类，用于表示带检索增强的边际模型输出
@dataclass
class TFRetrievAugLMMarginOutput(ModelOutput):
    # 损失
    loss: tf.Tensor | None = None
    logits: tf.Tensor = None
    past_key_values: List[tf.Tensor] | None = None
    doc_scores: tf.Tensor | None = None
    retrieved_doc_embeds: tf.Tensor | None = None
    retrieved_doc_ids: tf.Tensor | None = None
    context_input_ids: tf.Tensor | None = None
    context_attention_mask: tf.Tensor | None = None
    question_encoder_last_hidden_state: tf.Tensor | None = None
    question_enc_hidden_states: Tuple[tf.Tensor] | None = None
    question_enc_attentions: Tuple[tf.Tensor] | None = None
    generator_enc_last_hidden_state: tf.Tensor | None = None
    generator_enc_hidden_states: Tuple[tf.Tensor] | None = None
    generator_enc_attentions: Tuple[tf.Tensor] | None = None
    generator_dec_hidden_states: Tuple[tf.Tensor] | None = None
    generator_dec_attentions: Tuple[tf.Tensor] | None = None

# TFRetrievAugLMOutput 类，用于表示带检索增强的语言模型输出
@dataclass
class TFRetrievAugLMOutput(ModelOutput):
    # logits
    logits: tf.Tensor = None
    past_key_values: List[tf.Tensor] | None = None
    doc_scores: tf.Tensor | None = None
    retrieved_doc_embeds: tf.Tensor | None = None
    retrieved_doc_ids: tf.Tensor | None = None
    context_input_ids: tf.Tensor | None = None
    context_attention_mask: tf.Tensor | None = None
    question_encoder_last_hidden_state: tf.Tensor | None = None
    question_enc_hidden_states: Tuple[tf.Tensor] | None = None
    question_enc_attentions: Tuple[tf.Tensor] | None = None
    generator_enc_last_hidden_state: tf.Tensor | None = None
    # 定义了四个变量，分别是生成器编码器隐藏状态、生成器编码器注意力权重、生成器解码器隐藏状态、生成器解码器注意力权重
    # 变量类型为元组，其中包含了 TensorFlow 张量和可选的 None 值
    generator_enc_hidden_states: Tuple[tf.Tensor] | None = None
    generator_enc_attentions: Tuple[tf.Tensor] | None = None
    generator_dec_hidden_states: Tuple[tf.Tensor] | None = None
    generator_dec_attentions: Tuple[tf.Tensor] | None = None
# 该类是基于 TFPreTrainedModel 的 RAG 预训练模型
class TFRagPreTrainedModel(TFPreTrainedModel):
    # 文档描述 RAG 模型的基本情况
    r"""
    RAG models were released with the paper [Retrieval-Augmented Generation for Knowledge-Intensive NLP
    Tasks](https://arxiv.org/abs/2005.11401) by Patrick Lewis, Ethan Perez, Aleksandra Piktus et al.

    RAG is a retriever augmented model and encapsulate three components: a question encoder, a dataset retriever and a
    generator, the encoder and generator are trainable while the retriever is just an indexed dataset.
    """

    # RAG 模型使用的配置类
    config_class = RagConfig
    # RAG 模型的基本前缀
    base_model_prefix = "rag"
    # 在模型加载时忽略的键
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    # 从预训练的问题编码器和生成器中加载 RAG 模型
    @classmethod
    def from_pretrained_question_encoder_generator(
        cls,
        question_encoder_pretrained_model_name_or_path: str = None,
        generator_pretrained_model_name_or_path: str = None,
        retriever: RagRetriever = None,
        *model_args,
        **kwargs,

# RAG 模型的开始文档字符串
RAG_START_DOCSTRING = r"""
    # RAG 模型是一个序列到序列的模型,包含两个核心组件:问题编码器和生成器
    RAG is a sequence-to-sequence model which encapsulates two core components: a question encoder and a generator.
    # 在前向传递过程中,我们使用问题编码器对输入进行编码,并将其传递给检索器以提取相关的上下文文档
    During a forward pass, we encode the input with the question encoder and pass it to the retriever to extract
    relevant context documents. The documents are then prepended to the input. Such contextualized inputs is passed to
    the generator.

    # 问题编码器可以是任何自动编码模型,最好是 TFDPRQuestionEncoder,生成器可以是任何序列到序列模型,最好是 TFBartForConditionalGeneration
    The question encoder can be any *autoencoding* model, preferably [`TFDPRQuestionEncoder`], and the generator can be
    any *seq2seq* model, preferably [`TFBartForConditionalGeneration`].

    # 该模型可以使用 RagRetriever 进行端到端生成,也可以与检索器的输出在多个步骤中组合使用
    The model can be initialized with a [`RagRetriever`] for end-to-end generation or used in combination with the
    outputs of a retriever in multiple steps---see examples for more details. The model is compatible any
    *autoencoding* model as the `question_encoder` and any *seq2seq* model with language model head as the `generator`.
    It has been tested with [`TFDPRQuestionEncoder`] as the `question_encoder` and [`TFBartForConditionalGeneration`]
    as the `generator`.

    # 该模型继承自 TFPreTrainedModel,支持通用的预训练模型操作
    This model inherits from [`TFPreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    # 该模型也是一个 TensorFlow 2.0 的 Keras 模型子类
    This model is also a Tensorflow [tf.keras.Model](https://www.tensorflow.org/api_docs/python/tf/keras/Model)
    subclass. Use it as a regular TF 2.0 Keras Model and refer to the TF 2.0 documentation for all matter related to
    general usage and behavior.

    # 该模型当前处于开发状态,只支持 eager 模式,可能无法导出为 SavedModel 格式
    The model is in a developing state as it is now fully supports in eager-mode only, and may not be exported in
    SavedModel format.
"""
    Args:
        config ([`RagConfig`]):
            # 模型配置类，包含模型的所有参数。初始化时使用配置文件不会加载模型关联的权重，只会加载配置。查看
            # [`~TFPreTrainedModel.from_pretrained`] 方法来加载模型权重。
            模型配置类，包含所有模型参数。初始化时使用配置文件不会加载模型权重，仅加载配置。可以使用
            [`~TFPreTrainedModel.from_pretrained`] 方法加载模型权重。
        question_encoder ([`TFPreTrainedModel`]):
            # 与检索器封装的 faiss 索引兼容的编码器模型。
            与 RAG 结构中的检索器封装的 faiss 索引兼容的编码器模型。
        generator ([`TFPreTrainedModel`]):
            # 在 RAG 结构中用作生成器的 seq2seq 模型。
            在 RAG 结构中用作生成器的 seq2seq 模型。
        retriever ([`RagRetriever`]):
            # 封装了 faiss 索引的检索器类，用于获取当前输入的上下文文档。
            封装了 faiss 索引的检索器类，用于获取当前输入的上下文文档。
"""
RAG_FORWARD_INPUTS_DOCSTRING = r"""

@add_start_docstrings_to_model_forward(RAG_START_DOCSTRING)
类 TFRagModel，继承自 TFRagPreTrainedModel

    load_weight_prefix = "tf_rag_model_1"
    
    def __init__(
        self,
        config: Optional[PretrainedConfig] = None,
        question_encoder: Optional[TFPreTrainedModel] = None,
        generator: Optional[TFPreTrainedModel] = None,
        retriever: Optional[RagRetriever] = None,
        load_weight_prefix: Optional[str] = None,
        **kwargs,
    ):
        assert config is not None or (
            question_encoder is not None and generator is not None
        ), "Either a configuration or an question_encoder and a generator has to be provided."

        if config is None:
            # 根据 question_encoder 和 generator 的配置创建 RagConfig 对象
            config = RagConfig.from_question_encoder_generator_configs(
                question_encoder.config, generator.config, **kwargs
            )
        else:
            assert isinstance(config, self.config_class), f"config: {config} has to be of type {self.config_class}"
        # 调用父类的构造函数
        super().__init__(config, **kwargs)

        if question_encoder is None:
            from ..auto.modeling_tf_auto import TFAutoModel

            # 创建 TFAutoModel 实例作为 question_encoder
            question_encoder = TFAutoModel.from_config(config.question_encoder, name="question_encoder")

        if generator is None:
            from ..auto.modeling_tf_auto import TFAutoModelForSeq2SeqLM

            # 如果未提供 load_weight_prefix，使用默认前缀
            load_weight_prefix = load_weight_prefix if load_weight_prefix is not None else self.load_weight_prefix
            # 创建 TFAutoModelForSeq2SeqLM 实例作为 generator
            generator = TFAutoModelForSeq2SeqLM.from_config(
                config.generator, name="generator", load_weight_prefix=load_weight_prefix + "/generator"
            )

        # 设置实例变量 retriever
        self.retriever = retriever
        if self.retriever is not None:
            assert isinstance(
                retriever, RagRetriever
            ), f"`self.retriever` is of type {type(self.retriever)}, but should be of type `RagRetriever`"
            self.retriever = retriever

        # 设置实例变量 question_encoder 和 generator
        self.question_encoder = question_encoder
        self.generator = generator

    # 设置 retriever 实例变量
    def set_retriever(self, retriever: RagRetriever):
        self.retriever = retriever

    @unpack_inputs
    @add_start_docstrings_to_model_forward(RAG_FORWARD_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFRetrievAugLMOutput, config_class=_CONFIG_FOR_DOC)
    # 定义调用函数，用于模型的前向传播
    def call(
        self,
        # 输入文本的标识符，类型可以是 TFModelInputType 或 None
        input_ids: TFModelInputType | None = None,
        # 注意力遮罩，类型可以是 np.ndarray、tf.Tensor 或 None
        attention_mask: np.ndarray | tf.Tensor | None = None,
        # 编码器输出，类型可以是 np.ndarray、tf.Tensor 或 None
        encoder_outputs: np.ndarray | tf.Tensor | None = None,
        # 解码器输入的标识符，类型可以是 np.ndarray、tf.Tensor 或 None
        decoder_input_ids: np.ndarray | tf.Tensor | None = None,
        # 解码器注意力遮罩，类型可以是 np.ndarray、tf.Tensor 或 None
        decoder_attention_mask: np.ndarray | tf.Tensor | None = None,
        # 过去的键值对，类型可以是元组的元组，其中元素可以是 np.ndarray 或 tf.Tensor，或者为 None
        past_key_values: Tuple[Tuple[Union[np.ndarray, tf.Tensor]]] | None = None,
        # 文档分数，类型可以是 np.ndarray、tf.Tensor 或 None
        doc_scores: np.ndarray | tf.Tensor | None = None,
        # 上下文输入的标识符，类型可以是 np.ndarray、tf.Tensor 或 None
        context_input_ids: np.ndarray | tf.Tensor | None = None,
        # 上下文注意力遮罩，类型可以是 np.ndarray、tf.Tensor 或 None
        context_attention_mask: np.ndarray | tf.Tensor | None = None,
        # 是否使用缓存，类型可以是布尔值或 None
        use_cache: bool | None = None,
        # 是否输出注意力权重，类型可以是布尔值或 None
        output_attentions: bool | None = None,
        # 是否输出隐藏状态，类型可以是布尔值或 None
        output_hidden_states: bool | None = None,
        # 是否输出检索到的信息，类型可以是布尔值或 None
        output_retrieved: bool | None = None,
        # 文档数量，类型可以是整数或 None
        n_docs: int | None = None,
        # 是否返回字典，类型可以是布尔值或 None
        return_dict: bool | None = None,
        # 是否处于训练模式，默认为 False
        training: bool = False,
        # 其他关键字参数
        **kwargs,
    # 构建模型的方法，用于初始化模型的参数
    def build(self, input_shape=None):
        # 如果模型已经构建过，则直接返回
        if self.built:
            return
        # 设置模型为已构建状态
        self.built = True
        # 在生成器的命名空间下构建生成器
        with tf.name_scope(self.generator.name):
            self.generator.build(None)
        # 在问题编码器的命名空间下构建问题编码器
        with tf.name_scope(self.question_encoder.name):
            self.question_encoder.build(None)
# 定义 TF RAG-token 模型，执行 RAG-token 特定的前向传播边际化
@add_start_docstrings_to_model_forward(
    """
    A TF RAG-token model implementation. It performs RAG-token specific marginalization in the forward pass.
    """,
    RAG_START_DOCSTRING,
)
class TFRagTokenForGeneration(TFRagPreTrainedModel, TFCausalLanguageModelingLoss):
    # 加载权重的前缀
    load_weight_prefix = "tf_rag_token_for_generation_1/rag"

    def __init__(
        self,
        config: Optional[PretrainedConfig] = None,
        question_encoder: Optional[TFPreTrainedModel] = None,
        generator: Optional[TFPreTrainedModel] = None,
        retriever: Optional[RagRetriever] = None,
        **kwargs,
    ):
        # 断言需要提供一个配置或一个编码器和一个生成器
        assert config is not None or (
            question_encoder is not None and generator is not None
        ), "Either a configuration or an encoder and a generator has to be provided."

        # 如果未提供配置，则基于问题编码器和生成器配置创建配置
        if config is None:
            config = RagConfig.from_question_encoder_generator_configs(
                question_encoder.config, generator.config, **kwargs
            )

        super().__init__(config)

        # 实例化模型
        self.rag = TFRagModel(
            config=config,
            question_encoder=question_encoder,
            generator=generator,
            retriever=retriever,
            load_weight_prefix=self.load_weight_prefix,
            name="rag",
        )

    # 设置检索器
    def set_retriever(self, retriever: RagRetriever):
        self.rag.retriever = retriever

    # 准备用于生成的输入
    # 来源：https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_tf_bart.py
    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past_key_values=None,
        attention_mask=None,
        use_cache=None,
        encoder_outputs=None,
        doc_scores=None,
        n_docs=None,
        **kwargs,
    ):
        if past_key_values is not None:
            # 如果存在过去值，则仅使用最后一个解码器输入 ID
            decoder_input_ids = decoder_input_ids[:, -1:]

        return {
            "input_ids": None,
            "encoder_outputs": encoder_outputs,
            "doc_scores": doc_scores,
            "context_attention_mask": attention_mask,
            "decoder_input_ids": decoder_input_ids,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
            "do_marginalize": True,
            "n_docs": n_docs,
        }

    @property
    def retriever(self):
        # 获取检索器
        return self.rag.retriever

    @property
    def generator(self):
        # 获取生成器
        return self.rag.generator

    @property
    def question_encoder(self):
        # 获取问题编码器
        return self.rag.question_encoder

    @staticmethod
    # 定义一个内部函数，用于从嵌套的张量中收集由beam_indices索引的beam切片到新的beam数组中。如果嵌套张量与beam索引存在形状不匹配，则意味着这是缓存。在这种情况下，需要隔离并处理ndocs额外的维度。
    def _gather_beams(nested, beam_indices, batch_axis=0):
        # 定义一个收集函数，根据不同情况对张量进行重构
        def gather_fn(tensor):
            # 判断是否是RAG缓存
            is_rag_cache = tensor.shape[0] != beam_indices.shape[0]
            if is_rag_cache:
                n_docs = tensor.shape[0] // beam_indices.shape[0]
                batch_size = beam_indices.shape[0]
                # 重新调整形状为(batch size, num beams, n_docs, ...)，符合RAG所期望的缓存格式
                tensor = tf.reshape(tensor, (batch_size, -1, n_docs, *tensor.shape[2:]))
    
            # 根据beam_indices在指定轴上收集张量
            gathered_tensor = tf.gather(params=tensor, indices=beam_indices, axis=1, batch_dims=1)
    
            if is_rag_cache:
                # 将张量重新调整为beam搜索所期望的形状
                gathered_tensor = tf.reshape(gathered_tensor, (batch_size * n_docs, -1, *gathered_tensor.shape[3:]))
    
            return gathered_tensor
    
        # 对nested中的每个元素应用gather_fn函数
        return tf.nest.map_structure(gather_fn, nested)
    
    # 执行marginalize方法，用于RAG-token边际化
    # seq_logits代表序列的logits
    # doc_scores代表文档的得分
    # n_docs是指定的文档数量，默认为self.config.n_docs
    def marginalize(self, seq_logits, doc_scores, n_docs=None):
        n_docs = n_docs if n_docs is not None else self.config.n_docs
    
        # 对序列logits进行log_softmax操作
        seq_logprobs = tf.nn.log_softmax(seq_logits, axis=-1)
        seq_logprobs = tf.reshape(seq_logprobs, [seq_logits.shape[0] // n_docs, n_docs, -1, seq_logits.shape[-1]])
        # 对文档得分进行log_softmax操作
        doc_logprobs = tf.nn.log_softmax(doc_scores, axis=1)
        doc_logprobs = tf.expand_dims(doc_logprobs, axis=-1)
        doc_logprobs = tf.expand_dims(doc_logprobs, axis=-1)  # 两次扩展
        # 计算log概率总和
        log_prob_sum = seq_logprobs + doc_logprobs
        # 在第1轴上减小logsumexp操作
        return tf.reduce_logsumexp(log_prob_sum, axis=1)
    
    # 对模型前向输入添加描述信息，替换返回文档字符串的类型以及配置类
    @unpack_inputs
    @add_start_docstrings_to_model_forward(RAG_FORWARD_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFRetrievAugLMMarginOutput, config_class=_CONFIG_FOR_DOC)
    # 定义模型调用方法，用于生成输出
    def call(
        self,
        input_ids: TFModelInputType | None = None,  # 输入的 token IDs，数据类型可以是 TensorFlow 或 Numpy
        attention_mask: np.ndarray | tf.Tensor | None = None,  # 注意力遮罩，数据类型可以是 TensorFlow 或 Numpy
        decoder_input_ids: np.ndarray | tf.Tensor | None = None,  # 解码器输入的 token IDs，数据类型可以是 TensorFlow 或 Numpy
        decoder_attention_mask: np.ndarray | tf.Tensor | None = None,  # 解码器的注意力遮罩，数据类型可以是 TensorFlow 或 Numpy
        encoder_outputs: np.ndarray | tf.Tensor | None = None,  # 编码器的输出，数据类型可以是 TensorFlow 或 Numpy
        past_key_values: Tuple[Tuple[Union[np.ndarray, tf.Tensor]]] | None = None,  # 解码器的过去 key-value pairs
        doc_scores: np.ndarray | tf.Tensor | None = None,  # 文档分数，数据类型可以是 TensorFlow 或 Numpy
        context_input_ids: np.ndarray | tf.Tensor | None = None,  # 上下文输入的 token IDs，数据类型可以是 TensorFlow 或 Numpy
        context_attention_mask: np.ndarray | tf.Tensor | None = None,  # 上下文的注意力遮罩，数据类型可以是 TensorFlow 或 Numpy
        use_cache: bool | None = None,  # 是否使用缓存
        output_attentions: bool | None = None,  # 是否输出注意力权重
        output_hidden_states: bool | None = None,  # 是否输出隐藏状态
        output_retrieved: bool | None = None,  # 是否输出检索到的文档
        n_docs: int | None = None,  # 文档数量
        do_marginalize: bool | None = None,  # 是否计算边缘概率
        labels: np.ndarray | tf.Tensor | None = None,  # 标签，数据类型可以是 TensorFlow 或 Numpy
        reduce_loss: bool | None = None,  # 是否减少损失
        return_dict: bool | None = None,  # 是否返回字典形式的结果
        training: bool = False,  # 是否处于训练模式
        **kwargs,  # 其他参数，用于生成
    ):
    
    # 生成方法，用于生成文本
    def generate(
        self,
        input_ids: TFModelInputType | None = None,  # 输入的 token IDs，数据类型可以是 TensorFlow 或 Numpy
        attention_mask: tf.Tensor | None = None,  # 注意力遮罩，数据类型为 TensorFlow
        context_input_ids=None,  # 上下文输入的 token IDs
        context_attention_mask=None,  # 上下文的注意力遮罩
        doc_scores=None,  # 文档分数
        n_docs=None,  # 文档数量
        generation_config=None,  # 生成配置
        logits_processor=TFLogitsProcessorList(),  # logits 处理器
        **kwargs,  # 其他参数
    ):
    
    # 获取输入嵌入
    def get_input_embeddings(self):
        return self.rag.generator.get_input_embeddings()

    # 获取输出嵌入
    def get_output_embeddings(self):
        return self.rag.generator.get_output_embeddings()

    # 从 tf_t5 和 tf_bart 的 _shift_right 方法适配
   
    # 计算自定义的损失函数，忽略填充标记
    def hf_compute_loss(self, labels, y_pred, smooth_epsilon=0.0, from_logits=True, reduce_loss=False):
        """CrossEntropyLoss that ignores pad tokens"""
        # 使用 SparseCategoricalCrossentropy 创建损失函数对象
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True,
            reduction=tf.keras.losses.Reduction.SUM,
        )

        # 如果不是从 logits 开始，将预测值转换为 logits
        if from_logits is False:  # convert to logits
            eps = 1e-9
            y_pred = tf.clip_by_value(y_pred, clip_value_min=eps, clip_value_max=1 - eps)
            y_pred = tf.math.log(y_pred)

        # 使用预测值作为 logits
        logits = y_pred
        # 将标签展平
        melted_labels = tf.reshape(labels, (-1,))
        # 获取非填充标记的索引
        active_loss = tf.not_equal(melted_labels, self.config.generator.pad_token_id)

        # 根据非填充标记的索引，获取对应的 logits 和标签
        reduced_logits = tf.boolean_mask(tf.reshape(logits, (-1, logits.shape[2])), active_loss)
        labels = tf.boolean_mask(melted_labels, active_loss)
        # 计算原始的负对数似然损失
        nll_loss = loss_fn(labels, reduced_logits)

        # 计算平滑损失
        smooth_loss = -tf.reduce_sum(reduced_logits, axis=-1)
        smooth_loss = tf.reduce_sum(smooth_loss)  # sum and squeeze like torch
        eps_i = smooth_epsilon / reduced_logits.shape[-1]

        # 计算总损失，包括负对数似然损失和平滑损失
        loss = (1.0 - smooth_epsilon) * nll_loss + eps_i * smooth_loss

        return loss

    # 构建模型
    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回
        if self.built:
            return
        self.built = True
        # 如果存在 RAG 模型，则构建该模型
        if getattr(self, "rag", None) is not None:
            with tf.name_scope(self.rag.name):
                self.rag.build(None)
# 导入所需模块或函数
@add_start_docstrings_to_model_forward(
    """
    A TF RAG-sequence model implementation. It performs RAG-sequence specific marginalization in the forward pass.
    """,
    RAG_START_DOCSTRING,
)
# 定义 TF RAG-sequence 生成模型类，继承自 TFRagPreTrainedModel 和 TFCausalLanguageModelingLoss
class TFRagSequenceForGeneration(TFRagPreTrainedModel, TFCausalLanguageModelingLoss):
    # 加载权重的前缀
    load_weight_prefix = "tf_rag_sequence_for_generation_1/rag"

    # 初始化方法
    def __init__(
        self,
        config: Optional[PretrainedConfig] = None,
        question_encoder: Optional[TFPreTrainedModel] = None,
        generator: Optional[TFPreTrainedModel] = None,
        retriever: Optional[RagRetriever] = None,
        **kwargs,
    ):
        # 断言确保提供了配置或编码器和生成器中的一个
        assert config is not None or (
            question_encoder is not None and generator is not None
        ), "Either a configuration or an encoder and a generator has to be provided."

        # 如果没有提供配置，则根据问题编码器和生成器的配置创建一个
        if config is None:
            config = RagConfig.from_question_encoder_generator_configs(
                question_encoder.config, generator.config, **kwargs
            )

        # 调用父类的初始化方法
        super().__init__(config)

        # 实例化模型
        self.rag = TFRagModel(
            config=config,
            question_encoder=question_encoder,
            generator=generator,
            retriever=retriever,
            load_weight_prefix=self.load_weight_prefix,
            name="rag",
        )

    # 设置检索器的方法
    def set_retriever(self, retriever: RagRetriever):
        self.rag.retriever = retriever

    # 返回检索器属性
    @property
    def retriever(self):
        return self.rag.retriever

    # 返回生成器属性
    @property
    def generator(self):
        return self.rag.generator

    # 返回问题编码器属性
    @property
    def question_encoder(self):
        return self.rag.question_encoder

    # 前向传播方法，调用了 TFRagModel 的 call 方法
    @unpack_inputs
    @add_start_docstrings_to_model_forward(RAG_FORWARD_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFRetrievAugLMMarginOutput, config_class=_CONFIG_FOR_DOC)
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        decoder_input_ids: np.ndarray | tf.Tensor | None = None,
        decoder_attention_mask: np.ndarray | tf.Tensor | None = None,
        encoder_outputs: np.ndarray | tf.Tensor | None = None,
        past_key_values: Optional[Tuple[Tuple[Union[np.ndarray, tf.Tensor]]]] = None,
        doc_scores: np.ndarray | tf.Tensor | None = None,
        context_input_ids: np.ndarray | tf.Tensor | None = None,
        context_attention_mask: np.ndarray | tf.Tensor | None = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_retrieved: Optional[bool] = None,
        n_docs: Optional[int] = None,
        exclude_bos_score: Optional[bool] = None,
        labels: np.ndarray | tf.Tensor | None = None,
        reduce_loss: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
        **kwargs,  # needs kwargs for generation
    ):
    # 计算给定序列的负对数似然损失
    def get_nll(
        self, seq_logits, doc_scores, target, reduce_loss=False, epsilon=0.0, exclude_bos_score=False, n_docs=None
    ):
        # 输入参数:
        # - seq_logits: 序列的预测对数概率
        # - doc_scores: 与序列相关文档的相关性得分
        # - target: 目标序列
        # - reduce_loss: 是否对损失进行平均
        # - epsilon: 添加到损失上的平滑因子
        # - exclude_bos_score: 是否排除开头的分数
        # - n_docs: 相关文档的数量
    
        # 生成输出序列
    def generate(
        self,
        input_ids: TFModelInputType | None = None,
        attention_mask: tf.Tensor | None = None,
        context_input_ids=None,
        context_attention_mask=None,
        doc_scores=None,
        do_deduplication=None,  # defaults to True
        num_return_sequences=None,  # defaults to 1
        num_beams=None,  # defaults to 1
        n_docs=None,
        **model_kwargs,
    ):
        # 输入参数:
        # - input_ids: 输入序列的ID
        # - attention_mask: 输入序列的注意力掩码
        # - context_input_ids: 相关文档的ID
        # - context_attention_mask: 相关文档的注意力掩码
        # - doc_scores: 相关文档的相关性得分
        # - do_deduplication: 是否执行重复输出的去重
        # - num_return_sequences: 返回的序列数量
        # - num_beams: 光束搜索的宽度
        # - n_docs: 相关文档的数量
        # - **model_kwargs: 任何其他模型特定的关键字参数
    
    # 将一批张量进行拼接和填充
    @staticmethod
    def _cat_and_pad(tensors, pad_token_id):
        # 输入参数:
        # - tensors: 需要拼接和填充的张量列表
        # - pad_token_id: 填充时使用的占位符ID
    
        # 初始化一个形状为(所有候选项, 最大候选长度)的填充张量
        new_shape = sum([t.shape[0] for t in tensors]), max([t.shape[1] for t in tensors])
        output = tf.fill(new_shape, pad_token_id)
    
        # 将张量逐个赋值到填充的张量中
        output = tf.Variable(output)
        ind = 0
        for t in tensors:
            output[ind : ind + t.shape[0], : t.shape[1]].assign(t)
            ind += t.shape[0]
    
        # 将填充的张量转回为Tensor
        output = tf.convert_to_tensor(output)
        return tf.cast(output, tensors[0][0][0].dtype)
    
    # 构建模型
    def build(self, input_shape=None):
        # 如果模型已经构建过, 则直接返回
        if self.built:
            return
        self.built = True
    
        # 如果模型有RAG属性, 则在RAG的命名域内构建RAG模型
        if getattr(self, "rag", None) is not None:
            with tf.name_scope(self.rag.name):
                self.rag.build(None)
```