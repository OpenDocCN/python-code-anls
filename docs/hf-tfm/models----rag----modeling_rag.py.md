# `.\transformers\models\rag\modeling_rag.py`

```
# 设定文件编码为 UTF-8
# 版权声明
# 版权所有 2020 年，The RAG Authors 和 The HuggingFace Inc. 团队。
# 根据 Apache 许可证 2.0 版（“许可证”）授权；
# 除非符合许可证的规定，否则您不得使用此文件。
# 您可以在以下网址获取许可证的副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面约定，否则按“原样”分发的软件
# 没有任何形式的保证或条件，无论是明示条件还是隐含条件。
# 参见许可证以获取特定语言管理权限和限制。
"""RAG 模型实现。"""

# 导入所需的库和模块
import copy
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Union
import torch
from torch import nn
from ...configuration_utils import PretrainedConfig
from ...generation import BeamSearchScorer, GenerationConfig, LogitsProcessorList, StoppingCriteriaList
from ...modeling_outputs import ModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_rag import RagConfig
from .retrieval_rag import RagRetriever

# 获取日志记录器
logger = logging.get_logger(__name__)

# 文档配置
_CONFIG_FOR_DOC = "RagConfig"

# 定义检索增强语言模型边际输出类
@dataclass
class RetrievAugLMMarginOutput(ModelOutput):
    """
    检索增强边际模型输出的基类。

    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    doc_scores: torch.FloatTensor = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    retrieved_doc_embeds: Optional[torch.FloatTensor] = None
    retrieved_doc_ids: Optional[torch.LongTensor] = None
    context_input_ids: Optional[torch.LongTensor] = None
    context_attention_mask: Optional[torch.LongTensor] = None
    question_encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    question_enc_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    question_enc_attentions: Optional[Tuple[torch.FloatTensor]] = None
    generator_enc_last_hidden_state: Optional[torch.FloatTensor] = None
    generator_enc_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    generator_enc_attentions: Optional[Tuple[torch.FloatTensor]] = None
    generator_dec_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    generator_dec_attentions: Optional[Tuple[torch.FloatTensor]] = None
    generator_cross_attentions: Optional[Tuple[torch.FloatTensor]] = None

# 定义检索增强语言模型输出类
@dataclass
class RetrievAugLMOutput(ModelOutput):
    """
    """

    logits: torch.FloatTensor = None
    doc_scores: torch.FloatTensor = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    retrieved_doc_embeds: Optional[torch.FloatTensor] = None
    retrieved_doc_ids: Optional[torch.LongTensor] = None
    context_input_ids: Optional[torch.LongTensor] = None
    context_attention_mask: Optional[torch.LongTensor] = None
    # 问题编码器最后隐藏状态（可选）
    question_encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    # 问题编码器隐藏状态（可选）
    question_enc_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # 问题编码器注意力权重（可选）
    question_enc_attentions: Optional[Tuple[torch.FloatTensor]] = None
    # 生成器编码器最后隐藏状态（可选）
    generator_enc_last_hidden_state: Optional[torch.FloatTensor] = None
    # 生成器编码器隐藏状态（可选）
    generator_enc_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # 生成器编码器注意力权重（可选）
    generator_enc_attentions: Optional[Tuple[torch.FloatTensor]] = None
    # 生成器解码器隐藏状态（可选）
    generator_dec_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # 生成器解码器注意力权重（可选）
    generator_dec_attentions: Optional[Tuple[torch.FloatTensor]] = None
    # 生成器交叉注意力权重（可选）
    generator_cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
# 通过继承 PreTrainedModel 类定义 RAG 模型，RAG模型是带有检索功能的生成模型
class RagPreTrainedModel(PreTrainedModel):
    r"""
    RAG models were released with the paper [Retrieval-Augmented Generation for Knowledge-Intensive NLP
    Tasks](https://arxiv.org/abs/2005.11401) by Patrick Lewis, Ethan Perez, Aleksandra Piktus et al.

    RAG is a retriever augmented model and encapsulate three components: a question encoder, a dataset retriever and a
    generator, the encoder and generator are trainable while the retriever is just an indexed dataset.

    """
    
    # 指定 RAG 的配置类
    config_class = RagConfig
    
    # 指定 RAG 的基础模型名称前缀
    base_model_prefix = "rag"

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        # 目前不支持复合模型进行快速初始化
        # for composite models
        kwargs["_fast_init"] = False
        return super().from_pretrained(*args, **kwargs)

    @classmethod
    def from_pretrained_question_encoder_generator(
        cls,
        question_encoder_pretrained_model_name_or_path: str = None,
        generator_pretrained_model_name_or_path: str = None,
        retriever: RagRetriever = None,
        **kwargs,
RAG_START_DOCSTRING = r"""

    RAG is a seq2seq model which encapsulates two core components: a question encoder and a generator. During a forward
    pass, we encode the input with the question encoder and pass it to the retriever to extract relevant context
    documents. The documents are then prepended to the input. Such contextualized inputs is passed to the generator.

    The question encoder can be any *autoencoding* model, preferably [`DPRQuestionEncoder`], and the generator can be
    any *seq2seq* model, preferably [`BartForConditionalGeneration`].

    The model can be initialized with a [`RagRetriever`] for end-to-end generation or used in combination with the
    outputs of a retriever in multiple steps---see examples for more details. The model is compatible any
    *autoencoding* model as the `question_encoder` and any *seq2seq* model with language model head as the `generator`.
    It has been tested with [`DPRQuestionEncoder`] as the `question_encoder` and [`BartForConditionalGeneration`] or
    [`T5ForConditionalGeneration`] as the `generator`.

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.
    Args:
        config ([`RagConfig`]):
            # 模型配置类，包含模型的所有参数。使用配置文件初始化时不会加载模型的权重，只会加载配置。可以查看 [`~PreTrainedModel.from_pretrained`] 方法来加载模型的权重。
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
        question_encoder ([`PreTrainedModel`]):
            # 一个与 RAG 检索器中的 faiss 索引兼容的编码器模型。
            An encoder model compatible with the faiss index encapsulated by the `retriever`.
        generator ([`PreTrainedModel`]):
            # 在 RAG 架构中用作生成器的 seq2seq 模型。
            A seq2seq model used as the generator in the RAG architecture.
        retriever ([`RagRetriever`]):
            # 一个检索器类，封装了一个 faiss 索引，用于获取当前输入的上下文文档。
            A retriever class encapsulating a faiss index queried to obtain context documents for current inputs.
"""
定义了 RAG 前向输入的文档字符串
"""
RAG_FORWARD_INPUTS_DOCSTRING = r"""
"""

"""
基于 RAG_START_DOCSTRING 添加文档字符串到模型的前向方法
"""
@add_start_docstrings_to_model_forward(RAG_START_DOCSTRING)
class RagModel(RagPreTrainedModel):
    def __init__(
        self,
        config: Optional[PretrainedConfig] = None,
        question_encoder: Optional[PreTrainedModel] = None,
        generator: Optional[PreTrainedModel] = None,
        retriever: Optional[RagRetriever] = None,  # 或许可以使用 `set_retriever(...)` 方法
        **kwargs,
    ):
        assert config is not None or (
            question_encoder is not None and generator is not None
        ), "需要提供配置或者问题编码器和生成器。"

        if config is None:
            config = RagConfig.from_question_encoder_generator_configs(
                question_encoder.config, generator.config, **kwargs
            )
        else:
            assert isinstance(config, self.config_class), f"config: {config} 必须是类型为 {self.config_class}"
        super().__init__(config)
        if question_encoder is None:
            from ..auto.modeling_auto import AutoModel

            question_encoder = AutoModel.from_config(config.question_encoder)

        if generator is None:
            from ..auto.modeling_auto import AutoModelForSeq2SeqLM

            generator = AutoModelForSeq2SeqLM.from_config(config.generator)

        self.retriever = retriever
        if self.retriever is not None:
            assert isinstance(
                retriever, RagRetriever
            ), f"`self.retriever` 的类型为 {type(self.retriever)}, 但应为类型 `RagRetriever`"
            self.retriever = retriever

        self.question_encoder = question_encoder
        self.generator = generator

        self.ctx_encoder = None
        self.context_encoder_training = False

    @add_start_docstrings_to_model_forward(RAG_FORWARD_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=RetrievAugLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        doc_scores: Optional[torch.FloatTensor] = None,
        context_input_ids: Optional[torch.LongTensor] = None,
        context_attention_mask: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_retrieved: Optional[bool] = None,
        n_docs: Optional[int] = None,
@add_start_docstrings_to_model_forward(
    """
    实现了 RAG-sequence 模型。它在前向传递中执行 RAG-sequence 特定的边际化处理。
    """,
    # 定义开始文档字符串的标记
    RAG_START_DOCSTRING,
# 定义了 RagSequenceForGeneration 类，继承自 RagPreTrainedModel 类
class RagSequenceForGeneration(RagPreTrainedModel):
    # 初始化函数，接受一系列参数
    def __init__(
        self,
        config: Optional[PretrainedConfig] = None,
        question_encoder: Optional[PreTrainedModel] = None,
        generator: Optional[PreTrainedModel] = None,
        retriever: Optional[RagRetriever] = None,
        **kwargs,
    ):
        # 断言条件，要求必须提供配置文件或者问题编码器和生成器
        assert config is not None or (
            question_encoder is not None and generator is not None
        ), "Either a configuration or an encoder and a generator has to be provided."
        
        # 如果未提供配置文件，则根据问题编码器和生成器的配置创建配置文件
        if config is None:
            config = RagConfig.from_question_encoder_generator_configs(
                question_encoder.config, generator.config, **kwargs
            )
        # 调用父类的初始化函数，传入配置文件
        super().__init__(config)

        # 实例化模型
        self.rag = RagModel(config=config, question_encoder=question_encoder, generator=generator, retriever=retriever)

    # 设置检索器
    def set_retriever(self, retriever: RagRetriever):
        self.rag.retriever = retriever

    # 设置用于训练的上下文编码器
    def set_context_encoder_for_training(self, ctx_encoder: PreTrainedModel):
        self.rag.context_encoder_training = True
        self.rag.ctx_encoder = ctx_encoder

    # 重写 forward 方法，添加了文档字符串和替换返回文档字符串的装饰器
    @add_start_docstrings_to_model_forward(RAG_FORWARD_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=RetrievAugLMMarginOutput, config_class=_CONFIG_FOR_DOC)
    # 前向传播方法，接受多个输入参数
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        context_input_ids: Optional[torch.LongTensor] = None,
        context_attention_mask: Optional[torch.LongTensor] = None,
        doc_scores: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_retrieved: Optional[bool] = None,
        exclude_bos_score: Optional[bool] = None,
        reduce_loss: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None,
        n_docs: Optional[int] = None,
        **kwargs,  # 需要 kwargs 用于生成
    # 获取检索器的属性
    @property
    def retriever(self):
        return self.rag.retriever

    # 获取生成器的属性
    @property
    def generator(self):
        return self.rag.generator

    # 获取问题编码器的属性
    @property
    def question_encoder(self):
        return self.rag.question_encoder

    # 禁用梯度计算的装饰器
    @torch.no_grad()
    # 生成方法，生成模型的输出序列
    def generate(
        # 输入序列的索引，类型为 LongTensor，可选，默认为 None
        input_ids: Optional[torch.LongTensor] = None,
        # 注意力掩码，类型为 LongTensor，可选，默认为 None
        attention_mask: Optional[torch.LongTensor] = None,
        # 上下文输入序列的索引，类型为 LongTensor，可选，默认为 None
        context_input_ids: Optional[torch.LongTensor] = None,
        # 上下文输入序列的注意力掩码，类型为 LongTensor，可选，默认为 None
        context_attention_mask: Optional[torch.LongTensor] = None,
        # 文档分数，类型为 FloatTensor，可选，默认为 None
        doc_scores: Optional[torch.FloatTensor] = None,
        # 是否执行去重，类型为 bool，可选，默认为 True
        do_deduplication: Optional[bool] = None,  # defaults to True
        # 返回序列的数量，类型为 int，可选，默认为 1
        num_return_sequences: Optional[int] = None,  # defaults to 1
        # Beam 搜索中的束宽度，类型为 int，可选，默认为 1
        num_beams: Optional[int] = None,  # defaults to 1
        # 文档数量，类型为 int，可选
        n_docs: Optional[int] = None,
        # 其他模型参数
        **model_kwargs,
    # 获取负对数似然损失
    def get_nll(
        # 序列的对数概率，类型为张量，表示预测的序列的对数概率
        self, seq_logits,
        # 文档分数，类型为张量，表示文档的分数
        doc_scores,
        # 目标序列，类型为张量，表示真实的目标序列
        target,
        # 是否减少损失，默认为 False
        reduce_loss=False,
        # epsilon 值，用于数值稳定性，默认为 0.0
        epsilon=0.0,
        # 是否排除 BOS 分数，默认为 False
        exclude_bos_score=False,
        # 文档数量，可选
        n_docs=None
        ):
        # shift tokens left
        # 将目标张量向左移动，拼接生成新的张量
        target = torch.cat(
            [target[:, 1:], target.new(target.shape[0], 1).fill_(self.config.generator.pad_token_id)], 1
        )

        # 如果未指定 n_docs，则设置为模型配置中的 n_docs
        n_docs = n_docs if n_docs is not None else self.config.n_docs

        # 对于 T5 模型，bos_token_id 可能为空
        bos_token_id = self.config.bos_token_id or self.config.generator.bos_token_id
        use_bos = bos_token_id is not None and target[:, 0].eq(bos_token_id).all()

        def _mask_pads(ll, smooth_obj):
            # 创建用于指示填充标记位置的掩码
            pad_mask = target.eq(self.config.generator.pad_token_id)
            if pad_mask.any():
                # 将损失和平滑对象的填充位置置为零
                ll.masked_fill_(pad_mask, 0.0)
                smooth_obj.masked_fill_(pad_mask, 0.0)
            return ll.squeeze(-1), smooth_obj.squeeze(-1)

        # 对序列的对数概率进行归一化并重塑形状
        seq_logprobs = nn.functional.log_softmax(seq_logits, dim=-1).view(
            seq_logits.shape[0] // n_docs, n_docs, -1, seq_logits.size(-1)
        )  # batch_size x n_docs x tgt_len x #vocab_size
        doc_logprobs = nn.functional.log_softmax(doc_scores, dim=1).unsqueeze(-1).unsqueeze(-1)

        # RAG-sequence 边际概率
        first_token_scores = seq_logprobs[:, :, :1, :]
        second_token_scores = seq_logprobs[:, :, 1:2, :]
        remainder = seq_logprobs[:, :, 2:, :]
        rag_logprobs = torch.cat([first_token_scores, second_token_scores + doc_logprobs, remainder], dim=2)

        # 计算损失
        target = target.unsqueeze(1).unsqueeze(-1).repeat(1, n_docs, 1, 1)
        assert target.dim() == rag_logprobs.dim()

        ll = rag_logprobs.gather(dim=-1, index=target)
        smooth_obj = rag_logprobs.sum(dim=-1, keepdim=True)  # 所有（归一化）对数的总和

        ll, smooth_obj = _mask_pads(ll, smooth_obj)

        # 对标记求和，排除起始标记来计算得分
        ll = ll[:, :, 1:].sum(2) if exclude_bos_score and use_bos else ll.sum(2)
        smooth_obj = smooth_obj.sum(2)
        ll = ll.logsumexp(1)  # 对文档求 logsumexp
        smooth_obj = smooth_obj.logsumexp(1)

        nll_loss = -ll
        smooth_loss = -smooth_obj

        if reduce_loss:
            nll_loss = nll_loss.sum()
            smooth_loss = smooth_loss.sum()

        eps_i = epsilon / rag_logprobs.size(-1)
        loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
        return loss

    @staticmethod
    def _cat_and_pad(tensors, pad_token_id):
        # 拼接并填充张量
        output = (
            tensors[0].new(sum([t.shape[0] for t in tensors]), max([t.shape[1] for t in tensors])).fill_(pad_token_id)
        )
        ind = 0
        for t in tensors:
            output[ind : ind + t.shape[0], : t.shape[1]] = t
            ind += t.shape[0]
        return output
# 添加模型文档字符串到模型前向函数，并对RAG-token特定的边际化进行前向传递
@add_start_docstrings_to_model_forward(
    """
    A RAG-token model implementation. It performs RAG-token specific marginalization in the forward pass.
    """,
    RAG_START_DOCSTRING,
)
class RagTokenForGeneration(RagPreTrainedModel):
    def __init__(
        self,
        config: Optional[PretrainedConfig] = None,
        question_encoder: Optional[PreTrainedModel] = None,
        generator: Optional[PreTrainedModel] = None,
        retriever: Optional[RagRetriever] = None,
        **kwargs,
    ):
        # 如果未提供配置，必须提供问题编码器和生成器
        assert config is not None or (
            question_encoder is not None and generator is not None
        ), "Either a configuration or an encoder and a generator has to be provided."
        
        if config is None:
            # 根据问题编码器和生成器的配置创建RAG配置
            config = RagConfig.from_question_encoder_generator_configs(
                question_encoder.config, generator.config, **kwargs
            )

        super().__init__(config)

        # 实例化模型
        self.rag = RagModel(config=config, question_encoder=question_encoder, generator=generator, retriever=retriever)

    # 设置检索器
    def set_retriever(self, retriever: RagRetriever):
        self.rag.retriever = retriever

    # 用于训练设置上下文编码器
    def set_context_encoder_for_training(self, ctx_encoder: PreTrainedModel):
        self.rag.context_encoder_training = True
        self.rag.ctx_encoder = ctx_encoder

    # 准备用于生成的输入
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
        # 如果定义了过去的值，只使用最后的解码器输入id
        if past_key_values is not None:
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

    # 获取检索器
    @property
    def retriever(self):
        return self.rag.retriever

    # 获取生成器
    @property
    def generator(self):
        return self.rag.generator

    # 获取问题编码器
    @property
    def question_encoder(self):
        return self.rag.question_encoder

    @staticmethod
    # 重排序生成所需的缓存。受 BART 启发，但我们需要关注文档的额外维度
    def _reorder_cache(past_key_values, beam_idx):
        # 重新排序堆叠的隐藏状态
        def _reorder_stacked(hidden_states, new_order):
            n_docs = hidden_states.shape[0] // new_order.shape[0]
            hidden_states = hidden_states.view(-1, n_docs, *hidden_states.shape[1:])
            hidden_states = hidden_states.index_select(0, new_order)
            result = hidden_states.view(-1, *hidden_states.shape[2:])
            return result

        reordered_past = ()
        for layer_past in past_key_values:
            # 从解码器层的批次维度中获取正确的批次索引，用于跨越和自注意力
            reordered_past += (
                tuple(_reorder_stacked(past_state, beam_idx.to(past_state.device)) for past_state in layer_past),
            )

        return reordered_past

    # RAG-token 边际化
    def marginalize(self, seq_logits, doc_scores, n_docs=None):
        n_docs = n_docs if n_docs is not None else self.config.n_docs

        # 对序列 logits 进行 log_softmax 并重新形状
        seq_logprobs = nn.functional.log_softmax(seq_logits, dim=-1).view(
            seq_logits.shape[0] // n_docs, n_docs, -1, seq_logits.size(-1)
        )
        doc_logprobs = torch.log_softmax(doc_scores, dim=1)
        log_prob_sum = seq_logprobs + doc_logprobs.unsqueeze(-1).unsqueeze(-1)
        return torch.logsumexp(log_prob_sum, dim=1)

    # RAG 模型的前向传播
    @add_start_docstrings_to_model_forward(RAG_FORWARD_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=RetrievAugLMMarginOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        context_input_ids: Optional[torch.LongTensor] = None,
        context_attention_mask: Optional[torch.LongTensor] = None,
        doc_scores: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_retrieved: Optional[bool] = None,
        do_marginalize: Optional[bool] = None,
        reduce_loss: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None,
        n_docs: Optional[int] = None,
        **kwargs,  # 生成需要 kwargs
    @torch.no_grad()
    # 为生成文本准备输入参数
    def generate(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        context_input_ids: Optional[torch.LongTensor] = None,
        context_attention_mask: Optional[torch.LongTensor] = None,
        doc_scores: Optional[torch.FloatTensor] = None,
        n_docs: Optional[int] = None,
        generation_config: Optional[GenerationConfig] = None,
        prefix_allowed_tokens_fn: Callable[[int, torch.Tensor], List[int]] = None,
        logits_processor: Optional[LogitsProcessorList] = LogitsProcessorList(),
        stopping_criteria: Optional[StoppingCriteriaList] = StoppingCriteriaList(),
        **kwargs,
    
    # 获取输入嵌入层
    def get_input_embeddings(self):
        return self.rag.generator.get_input_embeddings()

    # 获取输出嵌入层
    def get_output_embeddings(self):
        return self.rag.generator.get_output_embeddings()

    # 设置输出嵌入层
    def set_output_embeddings(self, new_embeddings):
        return self.rag.generator.set_output_embeddings(new_embeddings)

    # 将输入序列向右移动一个位置，并用指定的起始标记填充
    def shift_tokens_right(self, input_ids, start_token_id=None):
        """Shift input ids one token to the right, and pad with start_token_id"""
        if start_token_id is None:
            start_token_id = self.config.decoder_start_token_id
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
        shifted_input_ids[:, 0] = start_token_id
        return shifted_input_ids

    # 计算负对数似然损失函数
    def get_nll(self, seq_logits, doc_scores, target, reduce_loss=False, epsilon=0.0, n_docs=None):
        n_docs = n_docs if n_docs is not None else self.config.n_docs
        # 将目标序列向左移动，以准备计算损失
        target = torch.cat(
            [target[:, 1:], target.new(target.shape[0], 1).fill_(self.config.generator.pad_token_id)], 1
        )

        # 处理填充标记
        def _mask_pads(ll, smooth_obj):
            pad_mask = target.eq(self.config.generator.pad_token_id)
            if pad_mask.any():
                ll.masked_fill_(pad_mask, 0.0)
                smooth_obj.masked_fill_(pad_mask, 0.0)
            return ll.squeeze(-1), smooth_obj.squeeze(-1)

        # 对生成的文本进行边际化，考虑多个文档的情况
        rag_logprobs = self.marginalize(seq_logits, doc_scores, n_docs)

        target = target.unsqueeze(-1)
        assert target.dim() == rag_logprobs.dim()

        # 从边际化的概率中收集目标标记的对数概率
        ll = rag_logprobs.gather(dim=-1, index=target)
        smooth_obj = rag_logprobs.sum(dim=-1, keepdim=True)  # 所有（标准化）对数值的总和
        ll, smooth_obj = _mask_pads(ll, smooth_obj)
        ll = ll.sum(1)  # 各个标记之间求和
        smooth_obj = smooth_obj.sum(1)

        nll_loss = -ll
        smooth_loss = -smooth_obj

        if reduce_loss:
            nll_loss = nll_loss.sum()
            smooth_loss = smooth_loss.sum()

        eps_i = epsilon / rag_logprobs.size(-1)
        loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
        return loss
```