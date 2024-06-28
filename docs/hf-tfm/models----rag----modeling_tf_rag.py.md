# `.\models\rag\modeling_tf_rag.py`

```py
# coding=utf-8
# 版权所有 2020 年，RAG 作者和 HuggingFace 公司团队。
#
# 根据 Apache 许可证 2.0 版本（“许可证”）许可；
# 您不得使用此文件，除非符合许可证的规定。
# 您可以在以下网址获取许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则本软件根据“原样”分发，
# 不提供任何明示或暗示的担保或条件。
# 有关详细信息，请参阅许可证。

"""TFRAG 模型实现。"""


from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf

# 导入各种模块和函数
from ...configuration_utils import PretrainedConfig
from ...generation import TFLogitsProcessorList
from ...modeling_tf_utils import (
    TFCausalLanguageModelingLoss,
    TFModelInputType,
    TFPreTrainedModel,
    keras,
    shape_list,
    unpack_inputs,
)
from ...utils import ModelOutput, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_rag import RagConfig
from .retrieval_rag import RagRetriever

# 获取日志记录器
logger = logging.get_logger(__name__)

# 用于文档的配置名称
_CONFIG_FOR_DOC = "RagConfig"

# 定义 TFRetrievAugLMMarginOutput 类，继承自 ModelOutput
@dataclass
class TFRetrievAugLMMarginOutput(ModelOutput):
    """
    用于检索增强的边缘化模型输出的基类。

    """

    loss: tf.Tensor | None = None  # 损失张量，可选
    logits: tf.Tensor = None  # logits 张量
    past_key_values: List[tf.Tensor] | None = None  # 过去键值列表，可选
    doc_scores: tf.Tensor | None = None  # 文档分数张量，可选
    retrieved_doc_embeds: tf.Tensor | None = None  # 检索的文档嵌入张量，可选
    retrieved_doc_ids: tf.Tensor | None = None  # 检索的文档 ID 张量，可选
    context_input_ids: tf.Tensor | None = None  # 上下文输入 ID 张量，可选
    context_attention_mask: tf.Tensor | None = None  # 上下文注意力掩码张量，可选
    question_encoder_last_hidden_state: tf.Tensor | None = None  # 问题编码器最后隐藏状态张量，可选
    question_enc_hidden_states: Tuple[tf.Tensor, ...] | None = None  # 问题编码器隐藏状态元组，可选
    question_enc_attentions: Tuple[tf.Tensor, ...] | None = None  # 问题编码器注意力元组，可选
    generator_enc_last_hidden_state: tf.Tensor | None = None  # 生成器编码器最后隐藏状态张量，可选
    generator_enc_hidden_states: Tuple[tf.Tensor, ...] | None = None  # 生成器编码器隐藏状态元组，可选
    generator_enc_attentions: Tuple[tf.Tensor, ...] | None = None  # 生成器编码器注意力元组，可选
    generator_dec_hidden_states: Tuple[tf.Tensor, ...] | None = None  # 生成器解码器隐藏状态元组，可选
    generator_dec_attentions: Tuple[tf.Tensor, ...] | None = None  # 生成器解码器注意力元组，可选

# 定义 TFRetrievAugLMOutput 类，继承自 ModelOutput
@dataclass
class TFRetrievAugLMOutput(ModelOutput):
    """
    """

    logits: tf.Tensor = None  # logits 张量
    past_key_values: List[tf.Tensor] | None = None  # 过去键值列表，可选
    doc_scores: tf.Tensor | None = None  # 文档分数张量，可选
    retrieved_doc_embeds: tf.Tensor | None = None  # 检索的文档嵌入张量，可选
    retrieved_doc_ids: tf.Tensor | None = None  # 检索的文档 ID 张量，可选
    context_input_ids: tf.Tensor | None = None  # 上下文输入 ID 张量，可选
    context_attention_mask: tf.Tensor | None = None  # 上下文注意力掩码张量，可选
    question_encoder_last_hidden_state: tf.Tensor | None = None  # 问题编码器最后隐藏状态张量，可选
    question_enc_hidden_states: Tuple[tf.Tensor, ...] | None = None  # 问题编码器隐藏状态元组，可选
    question_enc_attentions: Tuple[tf.Tensor, ...] | None = None  # 问题编码器注意力元组，可选
    # 定义变量 generator_enc_last_hidden_state，用于存储生成器编码器的最后隐藏状态，初始值为 None
    generator_enc_last_hidden_state: tf.Tensor | None = None
    # 定义变量 generator_enc_hidden_states，用于存储生成器编码器的所有隐藏状态的元组，初始值为 None
    generator_enc_hidden_states: Tuple[tf.Tensor, ...] | None = None
    # 定义变量 generator_enc_attentions，用于存储生成器编码器的所有注意力权重的元组，初始值为 None
    generator_enc_attentions: Tuple[tf.Tensor, ...] | None = None
    # 定义变量 generator_dec_hidden_states，用于存储生成器解码器的所有隐藏状态的元组，初始值为 None
    generator_dec_hidden_states: Tuple[tf.Tensor, ...] | None = None
    # 定义变量 generator_dec_attentions，用于存储生成器解码器的所有注意力权重的元组，初始值为 None
    generator_dec_attentions: Tuple[tf.Tensor, ...] | None = None
# 定义一个名为 TFRagPreTrainedModel 的类，继承自 TFPreTrainedModel 类
class TFRagPreTrainedModel(TFPreTrainedModel):
    # 类的文档字符串，描述了 RAG 模型的功能和组成部分
    r"""
    RAG models were released with the paper [Retrieval-Augmented Generation for Knowledge-Intensive NLP
    Tasks](https://arxiv.org/abs/2005.11401) by Patrick Lewis, Ethan Perez, Aleksandra Piktus et al.

    RAG is a retriever augmented model and encapsulate three components: a question encoder, a dataset retriever and a
    generator, the encoder and generator are trainable while the retriever is just an indexed dataset.
    """

    # 类属性，指定配置类为 RagConfig
    config_class = RagConfig
    # 类属性，基础模型前缀为 "rag"
    base_model_prefix = "rag"
    # 在加载时要忽略的键的列表
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    @classmethod
    # 类方法，用于从预训练的问题编码器和生成器创建实例
    def from_pretrained_question_encoder_generator(
        cls,
        question_encoder_pretrained_model_name_or_path: str = None,
        generator_pretrained_model_name_or_path: str = None,
        retriever: RagRetriever = None,
        *model_args,
        **kwargs,
RAG_START_DOCSTRING = r"""
    Args:
        config ([`RagConfig`]):
            # 模型配置类，包含模型的所有参数。使用配置文件初始化时不会加载模型的权重，只加载配置信息。
            # 若要加载模型权重，请参考 [`~TFPreTrainedModel.from_pretrained`] 方法。
        question_encoder ([`TFPreTrainedModel`]):
            # 编码器模型，与由 `retriever` 封装的 faiss 索引兼容。
        generator ([`TFPreTrainedModel`]):
            # 在 RAG 架构中用作生成器的 seq2seq 模型。
        retriever ([`RagRetriever`]):
            # 检索器类，封装了一个 faiss 索引，用于获取当前输入的上下文文档。
"""
"""


RAG_FORWARD_INPUTS_DOCSTRING = r"""
"""


@add_start_docstrings_to_model_forward(RAG_START_DOCSTRING)
class TFRagModel(TFRagPreTrainedModel):
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
            # 从问题编码器和生成器的配置中创建一个 RagConfig 对象
            config = RagConfig.from_question_encoder_generator_configs(
                question_encoder.config, generator.config, **kwargs
            )
        else:
            assert isinstance(config, self.config_class), f"config: {config} has to be of type {self.config_class}"
        super().__init__(config, **kwargs)

        if question_encoder is None:
            # 如果没有提供问题编码器，则使用自动加载的 TFAutoModel 创建一个
            from ..auto.modeling_tf_auto import TFAutoModel

            question_encoder = TFAutoModel.from_config(config.question_encoder, name="question_encoder")

        if generator is None:
            # 如果没有提供生成器，则使用自动加载的 TFAutoModelForSeq2SeqLM 创建一个
            from ..auto.modeling_tf_auto import TFAutoModelForSeq2SeqLM

            load_weight_prefix = load_weight_prefix if load_weight_prefix is not None else self.load_weight_prefix
            generator = TFAutoModelForSeq2SeqLM.from_config(
                config.generator, name="generator", load_weight_prefix=load_weight_prefix + "/generator"
            )

        self.retriever = retriever
        if self.retriever is not None:
            # 如果提供了检索器，确保它是 RagRetriever 类型的对象
            assert isinstance(
                retriever, RagRetriever
            ), f"`self.retriever` is of type {type(self.retriever)}, but should be of type `RagRetriever`"
            self.retriever = retriever

        self.question_encoder = question_encoder
        self.generator = generator

    def set_retriever(self, retriever: RagRetriever):
        # 设置检索器
        self.retriever = retriever

    @unpack_inputs
    @add_start_docstrings_to_model_forward(RAG_FORWARD_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFRetrievAugLMOutput, config_class=_CONFIG_FOR_DOC)
    # 定义类方法 `call`，用于模型调用和推理
    def call(
        self,
        input_ids: TFModelInputType | None = None,  # 输入序列的标识符（TensorFlow 模型输入类型或 None）
        attention_mask: np.ndarray | tf.Tensor | None = None,  # 注意力掩码，用于指定哪些位置的输入要被关注
        encoder_outputs: np.ndarray | tf.Tensor | None = None,  # 编码器的输出，可能是 numpy 数组或 TensorFlow 张量
        decoder_input_ids: np.ndarray | tf.Tensor | None = None,  # 解码器的输入标识符序列，可能是 numpy 数组或 TensorFlow 张量
        decoder_attention_mask: np.ndarray | tf.Tensor | None = None,  # 解码器的注意力掩码
        past_key_values: Tuple[Tuple[Union[np.ndarray, tf.Tensor]]] | None = None,  # 过去的键值对，用于 Transformer 模型的存储和重用
        doc_scores: np.ndarray | tf.Tensor | None = None,  # 文档评分，可能是 numpy 数组或 TensorFlow 张量
        context_input_ids: np.ndarray | tf.Tensor | None = None,  # 上下文输入的标识符序列
        context_attention_mask: np.ndarray | tf.Tensor | None = None,  # 上下文输入的注意力掩码
        use_cache: bool | None = None,  # 是否使用缓存来加速解码过程
        output_attentions: bool | None = None,  # 是否输出注意力权重
        output_hidden_states: bool | None = None,  # 是否输出隐藏状态
        output_retrieved: bool | None = None,  # 是否输出检索到的信息（如检索式推理）
        n_docs: int | None = None,  # 文档数量
        return_dict: bool | None = None,  # 是否返回字典格式的输出
        training: bool = False,  # 是否在训练模式下
        **kwargs,  # 其他关键字参数，用于接收任何未指定的额外参数
    ):
        # 如果模型已经构建好，则直接返回
        if self.built:
            return
        # 标记模型已经构建
        self.built = True
        # 使用 TensorFlow 的名称作用域，构建生成器部分的模型
        with tf.name_scope(self.generator.name):
            self.generator.build(None)
        # 使用 TensorFlow 的名称作用域，构建问题编码器部分的模型
        with tf.name_scope(self.question_encoder.name):
            self.question_encoder.build(None)
@add_start_docstrings_to_model_forward(
    """
    A TF RAG-token model implementation. It performs RAG-token specific marginalization in the forward pass.
    """,
    RAG_START_DOCSTRING,
)
class TFRagTokenForGeneration(TFRagPreTrainedModel, TFCausalLanguageModelingLoss):
    load_weight_prefix = "tf_rag_token_for_generation_1/rag"

    def __init__(
        self,
        config: Optional[PretrainedConfig] = None,
        question_encoder: Optional[TFPreTrainedModel] = None,
        generator: Optional[TFPreTrainedModel] = None,
        retriever: Optional[RagRetriever] = None,
        **kwargs,
    ):
        assert config is not None or (
            question_encoder is not None and generator is not None
        ), "Either a configuration or an encoder and a generator has to be provided."

        if config is None:
            # 如果未提供配置，根据提供的问题编码器和生成器配置生成一个新的RagConfig对象
            config = RagConfig.from_question_encoder_generator_configs(
                question_encoder.config, generator.config, **kwargs
            )

        super().__init__(config)

        # 实例化RAG模型
        self.rag = TFRagModel(
            config=config,
            question_encoder=question_encoder,
            generator=generator,
            retriever=retriever,
            load_weight_prefix=self.load_weight_prefix,
            name="rag",
        )

    def set_retriever(self, retriever: RagRetriever):
        # 设置RAG模型的检索器
        self.rag.retriever = retriever

    # 从 https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_tf_bart.py 改编而来
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
            # 如果定义了过去的键值，只使用最后一个decoder_input_ids
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
        # 返回RAG模型的检索器
        return self.rag.retriever

    @property
    def generator(self):
        # 返回RAG模型的生成器
        return self.rag.generator

    @property
    def question_encoder(self):
        # 返回RAG模型的问题编码器
        return self.rag.question_encoder

    @staticmethod
    def _gather_beams(nested, beam_indices, batch_axis=0):
        """
        RAG-specific `_gather_beams`: gathers the beam slices indexed by beam_indices into new beam array. If the
        nested tensor has a shape mismatch with the beam indices, then it means it is the cache. In that case, isolates
        and takes care of the extra dimension for ndocs.
        """

        def gather_fn(tensor):
            # 判断是否为 RAG 的缓存数据
            is_rag_cache = tensor.shape[0] != beam_indices.shape[0]
            if is_rag_cache:
                # 如果是缓存数据，则计算每个文档的数量和批次大小
                n_docs = tensor.shape[0] // beam_indices.shape[0]
                batch_size = beam_indices.shape[0]
                # 重塑张量为 (批次大小, num beams, n_docs, ...) 的格式，这是 RAG 期望的缓存格式
                tensor = tf.reshape(tensor, (batch_size, -1, n_docs, *tensor.shape[2:]))

            # 使用给定的索引从张量中收集数据
            gathered_tensor = tf.gather(params=tensor, indices=beam_indices, axis=1, batch_dims=1)

            if is_rag_cache:
                # 如果是缓存数据，则重新塑造成 beam search 期望的形状
                gathered_tensor = tf.reshape(gathered_tensor, (batch_size * n_docs, -1, *gathered_tensor.shape[3:]))

            return gathered_tensor

        # 对嵌套结构应用 gather_fn 函数，用于收集索引的数据
        return tf.nest.map_structure(gather_fn, nested)

    def marginalize(self, seq_logits, doc_scores, n_docs=None):
        n_docs = n_docs if n_docs is not None else self.config.n_docs

        # RAG-token marginalization
        # 对序列 logits 应用 log_softmax，在指定轴上进行归一化
        seq_logprobs = tf.nn.log_softmax(seq_logits, axis=-1)
        # 重新塑造成 [batch_size // n_docs, n_docs, -1, seq_logits.shape[-1]] 的形状
        seq_logprobs = tf.reshape(seq_logprobs, [seq_logits.shape[0] // n_docs, n_docs, -1, seq_logits.shape[-1]])
        # 对文档分数应用 log_softmax，在第 1 轴上进行归一化
        doc_logprobs = tf.nn.log_softmax(doc_scores, axis=1)
        # 在最后添加两个维度
        doc_logprobs = tf.expand_dims(doc_logprobs, axis=-1)
        doc_logprobs = tf.expand_dims(doc_logprobs, axis=-1)  # 两次
        # 计算序列和文档 log-probabilities 的总和
        log_prob_sum = seq_logprobs + doc_logprobs
        # 在第 1 轴上计算 logsumexp
        return tf.reduce_logsumexp(log_prob_sum, axis=1)

    @unpack_inputs
    @add_start_docstrings_to_model_forward(RAG_FORWARD_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFRetrievAugLMMarginOutput, config_class=_CONFIG_FOR_DOC)
    # 定义一个方法 `call`，用于模型调用。接受多个输入参数，包括输入的编码 `input_ids`，
    # 注意力掩码 `attention_mask`，解码器的输入编码 `decoder_input_ids` 和注意力掩码 `decoder_attention_mask`，
    # 编码器的输出 `encoder_outputs`，过去的键值对 `past_key_values`，文档分数 `doc_scores`，
    # 上下文输入编码 `context_input_ids` 和上下文注意力掩码 `context_attention_mask` 等等。
    # 其他参数包括是否使用缓存 `use_cache`，是否输出注意力 `output_attentions` 和隐藏状态 `output_hidden_states`，
    # 是否输出检索结果 `output_retrieved`，文档数量 `n_docs`，是否边际化 `do_marginalize`，
    # 标签 `labels`，是否减少损失 `reduce_loss`，是否返回字典 `return_dict`，
    # 是否处于训练模式 `training` 等等。
    # 方法允许传递任意其他关键字参数 `kwargs`。
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        decoder_input_ids: np.ndarray | tf.Tensor | None = None,
        decoder_attention_mask: np.ndarray | tf.Tensor | None = None,
        encoder_outputs: np.ndarray | tf.Tensor | None = None,
        past_key_values: Tuple[Tuple[Union[np.ndarray, tf.Tensor]]] | None = None,
        doc_scores: np.ndarray | tf.Tensor | None = None,
        context_input_ids: np.ndarray | tf.Tensor | None = None,
        context_attention_mask: np.ndarray | tf.Tensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        output_retrieved: bool | None = None,
        n_docs: int | None = None,
        do_marginalize: bool | None = None,
        labels: np.ndarray | tf.Tensor | None = None,
        reduce_loss: bool | None = None,
        return_dict: bool | None = None,
        training: bool = False,
        **kwargs,  # needs kwargs for generation
    ):
        pass  # 方法主体未提供

    # 定义一个生成方法 `generate`，用于生成文本。接受多个输入参数，包括输入的编码 `input_ids`，
    # 注意力掩码 `attention_mask`，上下文输入编码 `context_input_ids` 和上下文注意力掩码 `context_attention_mask`，
    # 文档分数 `doc_scores`，文档数量 `n_docs`，生成配置 `generation_config`，
    # 对 logits 进行处理的处理器 `logits_processor`，以及其他关键字参数 `kwargs`。
    def generate(
        self,
        input_ids: TFModelInputType | None = None,
        attention_mask: tf.Tensor | None = None,
        context_input_ids=None,
        context_attention_mask=None,
        doc_scores=None,
        n_docs=None,
        generation_config=None,
        logits_processor=TFLogitsProcessorList(),
        **kwargs,
    ):
        pass  # 方法主体未提供

    # 返回 RAG 模型中生成器的输入嵌入
    def get_input_embeddings(self):
        return self.rag.generator.get_input_embeddings()

    # 返回 RAG 模型中生成器的输出嵌入
    def get_output_embeddings(self):
        return self.rag.generator.get_output_embeddings()

    # 从 tf_t5 和 tf_bart 的 _shift_right 方法进行适配
    # 该方法可能实现了类似于将序列向右移动一个位置的功能
    # 但具体实现细节不在此处提供
    # 适配自 tf_t5 和 tf_bart 的 _shift_right 方法
    # 将输入的 token ids 向右移动一位，并用 start_token_id 进行填充
    def shift_tokens_right(self, input_ids, start_token_id=None):
        """Shift input ids one token to the right, and pad with start_token_id"""

        if start_token_id is None:
            start_token_id = self.generator.config.decoder_start_token_id
            assert start_token_id is not None, (
                "self.generator.config.decoder_start_token_id has to be defined. In Rag we commonly use Bart as"
                " generator, see Bart docs for more information"
            )

        pad_token_id = self.generator.config.pad_token_id
        assert pad_token_id is not None, "self.model.config.pad_token_id has to be defined."

        # 创建一个形状为 (batch_size, 1) 的张量，用 start_token_id 填充
        start_tokens = tf.fill((shape_list(input_ids)[0], 1), tf.cast(start_token_id, input_ids.dtype))
        # 将 start_tokens 与 input_ids 的前 n-1 列拼接起来，实现向右移动一位的效果
        shifted_input_ids = tf.concat([start_tokens, input_ids[:, :-1]], -1)

        # 将 labels 中可能存在的 -100 值替换为 pad_token_id
        shifted_input_ids = tf.where(
            shifted_input_ids == -100,
            tf.fill(shape_list(shifted_input_ids), tf.cast(pad_token_id, input_ids.dtype)),
            shifted_input_ids,
        )

        # 使用断言确保 `labels` 中只有正值和 -100
        assert_gte0 = tf.debugging.assert_greater_equal(shifted_input_ids, tf.cast(0, shifted_input_ids.dtype))

        # 确保断言操作被调用，通过在结果中包裹一个 identity 操作
        with tf.control_dependencies([assert_gte0]):
            shifted_input_ids = tf.identity(shifted_input_ids)

        return shifted_input_ids

    # nll 代表 'negative log likelihood'
    def get_nll(self, seq_logits, doc_scores, target, reduce_loss=False, epsilon=0.0, n_docs=None):
        n_docs = n_docs if n_docs is not None else self.config.n_docs
        # 将 tokens 向左移动（来自原始的 PyTorch 版本）

        # 将 target 的每一行向左移动一个 token，并用 self.config.generator.pad_token_id 进行填充
        target = tf.concat(
            [target[:, 1:], tf.fill([target.shape[0], 1], tf.cast(self.config.generator.pad_token_id, target.dtype))],
            axis=1,
        )
        # 对 seq_logits 和 doc_scores 进行边缘化，得到 rag_logprobs
        rag_logprobs = self.marginalize(seq_logits, doc_scores, n_docs)
        # 计算损失，匹配 logits 版本，reduce_loss 参数决定是否减少损失
        loss = self.hf_compute_loss(target, rag_logprobs, from_logits=True, reduce_loss=reduce_loss)

        return loss

    # 采用 modeling_tf_bart，并添加 smooth_loss 以匹配 PyTorch 版本
    def hf_compute_loss(self, labels, y_pred, smooth_epsilon=0.0, from_logits=True, reduce_loss=False):
        """计算损失函数，忽略填充标记的交叉熵损失"""
        # Matt: 该损失函数目前无法与XLA兼容，但它执行了一些非常奇怪的操作，
        #       我不太确定如何转换它。
        #       这里执行了一些非常奇怪的操作，我不太确定如何转换它。

        # 定义损失函数为稀疏分类交叉熵损失，用于处理输出为 logits 的情况
        loss_fn = keras.losses.SparseCategoricalCrossentropy(
            from_logits=True,  # 输出是否为 logits
            reduction=keras.losses.Reduction.SUM,  # 损失函数如何进行汇总
        )

        if from_logits is False:  # 如果输出不是 logits，则将其转换为 logits
            eps = 1e-9
            y_pred = tf.clip_by_value(y_pred, clip_value_min=eps, clip_value_max=1 - eps)
            y_pred = tf.math.log(y_pred)

        logits = y_pred
        melted_labels = tf.reshape(labels, (-1,))
        # 找出非填充标记的位置
        active_loss = tf.not_equal(melted_labels, self.config.generator.pad_token_id)

        # 根据非填充标记的位置，筛选出有效的 logits 和 labels
        reduced_logits = tf.boolean_mask(tf.reshape(logits, (-1, logits.shape[2])), active_loss)
        labels = tf.boolean_mask(melted_labels, active_loss)

        # 计算交叉熵损失
        nll_loss = loss_fn(labels, reduced_logits)

        # 计算平滑损失
        smooth_loss = -tf.reduce_sum(reduced_logits, axis=-1)
        smooth_loss = tf.reduce_sum(smooth_loss)  # 类似于 torch 的 sum 和 squeeze 操作
        eps_i = smooth_epsilon / reduced_logits.shape[-1]

        # 计算最终损失函数，结合交叉熵损失和平滑损失
        loss = (1.0 - smooth_epsilon) * nll_loss + eps_i * smooth_loss

        return loss

    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回
        if self.built:
            return
        self.built = True
        # 如果存在 rag 属性，则在 rag 的命名空间下构建模型
        if getattr(self, "rag", None) is not None:
            with tf.name_scope(self.rag.name):
                self.rag.build(None)
# 使用装饰器为模型的 call 方法添加文档字符串，描述其功能为执行RAG-sequence模型的前向传播过程
@add_start_docstrings_to_model_forward(
    """
    A TF RAG-sequence model implementation. It performs RAG-sequence specific marginalization in the forward pass.
    """,
    RAG_START_DOCSTRING,
)
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
        # 断言确保提供了配置或者问题编码器与生成器
        assert config is not None or (
            question_encoder is not None and generator is not None
        ), "Either a configuration or an encoder and a generator has to be provided."

        # 如果未提供配置，则从问题编码器和生成器配置中创建 RagConfig 对象
        if config is None:
            config = RagConfig.from_question_encoder_generator_configs(
                question_encoder.config, generator.config, **kwargs
            )

        # 调用父类初始化方法
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

    # 检索器属性的 getter 方法
    @property
    def retriever(self):
        return self.rag.retriever

    # 生成器属性的 getter 方法
    @property
    def generator(self):
        return self.rag.generator

    # 问题编码器属性的 getter 方法
    @property
    def question_encoder(self):
        return self.rag.question_encoder

    # 装饰器为 call 方法添加文档字符串，描述其输入输出及功能
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
        # 实现模型的前向传播
        pass  # The actual implementation details would follow here, but are not provided in the snippet
    # 定义一个方法 get_nll，接受一些参数：seq_logits 是序列的逻辑回归输出，doc_scores 是文档得分，target 是目标值
    # reduce_loss 控制是否减少损失，默认为 False；epsilon 是一个小数，排除 BOS 得分，默认为 False
    # n_docs 是文档数量，默认为 None
    def get_nll(
        self, seq_logits, doc_scores, target, reduce_loss=False, epsilon=0.0, exclude_bos_score=False, n_docs=None
    ):
        pass  # 这里只是定义方法的结构，没有具体实现

    # 定义一个生成方法 generate，接受多个输入参数：input_ids 是模型输入的 token IDs
    # attention_mask 是注意力掩码，context_input_ids 和 context_attention_mask 是上下文相关的输入
    # doc_scores 是文档得分，do_deduplication 控制是否去重，默认为 True；num_return_sequences 控制返回的序列数量，默认为 1
    # num_beams 控制束搜索的数量，默认为 1，n_docs 是文档数量，默认为 None
    def generate(
        self,
        input_ids: TFModelInputType | None = None,
        attention_mask: tf.Tensor | None = None,
        context_input_ids=None,
        context_attention_mask=None,
        doc_scores=None,
        do_deduplication=None,  # 默认为 True
        num_return_sequences=None,  # 默认为 1
        num_beams=None,  # 默认为 1
        n_docs=None,
        **model_kwargs,
    ):
        pass  # 这里只是定义方法的结构，没有具体实现

    # 静态方法 _cat_and_pad 用于生成方法 generate 中的输入张量列表的拼接和填充
    @staticmethod
    def _cat_and_pad(tensors, pad_token_id):
        # used by generate(): tensors is a (batched) list of (candidates, len); len is varied across batch
        # 方法 generate 的辅助方法，tensors 是一个 (批量化的) 列表，每个元素是 (候选项，长度)；长度在批次中可能不同

        # 初始化一个填充后的张量，形状为 (所有候选项总数，最大候选项长度)
        new_shape = sum([t.shape[0] for t in tensors]), max([t.shape[1] for t in tensors])
        output = tf.fill(new_shape, pad_token_id)

        # 使用 tf.Variable 创建可变张量，因为普通张量不支持切片赋值
        output = tf.Variable(output)

        # 逐个赋值每个输入张量的内容到 output 中相应位置
        ind = 0
        for t in tensors:
            output[ind : ind + t.shape[0], : t.shape[1]].assign(t)
            ind += t.shape[0]

        # 转换回普通张量并返回，确保类型与第一个张量的元素类型一致
        output = tf.convert_to_tensor(output)
        return tf.cast(output, tensors[0][0].dtype)

    # 方法 build 用于构建对象，初始化对象的属性和状态
    def build(self, input_shape=None):
        if self.built:
            return  # 如果已经构建过，直接返回

        self.built = True  # 标记对象已经构建

        # 如果对象具有属性 rag，则在 rag 的命名空间下构建对象
        if getattr(self, "rag", None) is not None:
            with tf.name_scope(self.rag.name):
                self.rag.build(None)
```