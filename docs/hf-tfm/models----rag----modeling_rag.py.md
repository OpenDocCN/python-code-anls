# `.\models\rag\modeling_rag.py`

```
# 导入必要的库和模块
import copy  # 导入深拷贝模块
from dataclasses import dataclass  # 导入dataclass装饰器
from typing import Callable, List, Optional, Tuple, Union  # 导入类型提示

import torch  # 导入PyTorch库
from torch import nn  # 导入神经网络模块

# 导入配置相关的工具和类
from ...configuration_utils import PretrainedConfig  # 导入预训练配置类
from ...generation import BeamSearchScorer, GenerationConfig, LogitsProcessorList, StoppingCriteriaList  # 导入生成相关类和模块
from ...modeling_outputs import ModelOutput  # 导入模型输出基类
from ...modeling_utils import PreTrainedModel  # 导入预训练模型类
from ...utils import add_start_docstrings_to_model_forward, logging, replace_return_docstrings  # 导入工具类和函数

# 导入RAG模型相关配置和检索器
from .configuration_rag import RagConfig  # 导入RAG配置类
from .retrieval_rag import RagRetriever  # 导入RAG检索器类

# 获取日志记录器
logger = logging.get_logger(__name__)

# 用于文档的配置名称
_CONFIG_FOR_DOC = "RagConfig"

@dataclass
class RetrievAugLMMarginOutput(ModelOutput):
    """
    检索增强的边际化模型输出的基类。

    """
    loss: Optional[torch.FloatTensor] = None  # 损失值，可选的浮点张量
    logits: torch.FloatTensor = None  # 对数张量，浮点张量
    doc_scores: torch.FloatTensor = None  # 文档分数，浮点张量
    past_key_values: Optional[List[torch.FloatTensor]] = None  # 过去的键值，可选的浮点张量列表
    retrieved_doc_embeds: Optional[torch.FloatTensor] = None  # 检索到的文档嵌入，可选的浮点张量
    retrieved_doc_ids: Optional[torch.LongTensor] = None  # 检索到的文档ID，可选的长整型张量
    context_input_ids: Optional[torch.LongTensor] = None  # 上下文输入ID，可选的长整型张量
    context_attention_mask: Optional[torch.LongTensor] = None  # 上下文注意力掩码，可选的长整型张量
    question_encoder_last_hidden_state: Optional[torch.FloatTensor] = None  # 问题编码器最后隐藏状态，可选的浮点张量
    question_enc_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None  # 问题编码器隐藏状态元组，可选的浮点张量元组
    question_enc_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None  # 问题编码器注意力元组，可选的浮点张量元组
    generator_enc_last_hidden_state: Optional[torch.FloatTensor] = None  # 生成器编码器最后隐藏状态，可选的浮点张量
    generator_enc_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None  # 生成器编码器隐藏状态元组，可选的浮点张量元组
    generator_enc_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None  # 生成器编码器注意力元组，可选的浮点张量元组
    generator_dec_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None  # 生成器解码器隐藏状态元组，可选的浮点张量元组
    generator_dec_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None  # 生成器解码器注意力元组，可选的浮点张量元组
    generator_cross_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None  # 生成器交叉注意力元组，可选的浮点张量元组

@dataclass
class RetrievAugLMOutput(ModelOutput):
    """
    检索增强的语言模型输出基类。

    """
    logits: torch.FloatTensor = None  # 对数张量，浮点张量
    doc_scores: torch.FloatTensor = None  # 文档分数，浮点张量
    past_key_values: Optional[List[torch.FloatTensor]] = None  # 过去的键值，可选的浮点张量列表
    retrieved_doc_embeds: Optional[torch.FloatTensor] = None  # 检索到的文档嵌入，可选的浮点张量
    retrieved_doc_ids: Optional[torch.LongTensor] = None  # 检索到的文档ID，可选的长整型张量
    context_input_ids: Optional[torch.LongTensor] = None  # 上下文输入ID，可选的长整型张量
    context_attention_mask: Optional[torch.LongTensor] = None  # 上下文注意力掩码，可选的长整型张量
    # 定义问题编码器的最后隐藏状态，初始值为None，类型为可选的浮点张量
    question_encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    
    # 定义问题编码器的隐藏状态列表，初始值为None，类型为可选的包含多个浮点张量的元组
    question_enc_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    
    # 定义问题编码器的注意力列表，初始值为None，类型为可选的包含多个浮点张量的元组
    question_enc_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    
    # 定义生成器编码器的最后隐藏状态，初始值为None，类型为可选的浮点张量
    generator_enc_last_hidden_state: Optional[torch.FloatTensor] = None
    
    # 定义生成器编码器的隐藏状态列表，初始值为None，类型为可选的包含多个浮点张量的元组
    generator_enc_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    
    # 定义生成器编码器的注意力列表，初始值为None，类型为可选的包含多个浮点张量的元组
    generator_enc_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    
    # 定义生成器解码器的隐藏状态列表，初始值为None，类型为可选的包含多个浮点张量的元组
    generator_dec_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    
    # 定义生成器解码器的注意力列表，初始值为None，类型为可选的包含多个浮点张量的元组
    generator_dec_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    
    # 定义生成器交叉注意力列表，初始值为None，类型为可选的包含多个浮点张量的元组
    generator_cross_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
# 定义一个自定义的 RAG 预训练模型类，继承自 PreTrainedModel
class RagPreTrainedModel(PreTrainedModel):
    r"""
    RAG models were released with the paper [Retrieval-Augmented Generation for Knowledge-Intensive NLP
    Tasks](https://arxiv.org/abs/2005.11401) by Patrick Lewis, Ethan Perez, Aleksandra Piktus et al.

    RAG is a retriever augmented model and encapsulate three components: a question encoder, a dataset retriever and a
    generator, the encoder and generator are trainable while the retriever is just an indexed dataset.

    """

    # 指定配置类为 RagConfig
    config_class = RagConfig
    # 指定基础模型的前缀为 "rag"
    base_model_prefix = "rag"

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        # 目前不支持快速初始化
        # 对于复合模型
        kwargs["_fast_init"] = False
        # 调用父类的 from_pretrained 方法
        return super().from_pretrained(*args, **kwargs)

    @classmethod
    def from_pretrained_question_encoder_generator(
        cls,
        question_encoder_pretrained_model_name_or_path: str = None,
        generator_pretrained_model_name_or_path: str = None,
        retriever: RagRetriever = None,
        **kwargs,
    ):
        # 以下是 RAG 模型的文档字符串定义，描述了模型的结构和使用方法
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
        """
    Args:
        config ([`RagConfig`]):
            模型配置类，包含模型的所有参数。通过配置文件初始化不会加载与模型相关的权重，只加载配置信息。
            若要加载模型权重，请查看 [`~PreTrainedModel.from_pretrained`] 方法。
        question_encoder ([`PreTrainedModel`]):
            编码器模型，与由 `retriever` 封装的 faiss 索引兼容。
        generator ([`PreTrainedModel`]):
            在 RAG 结构中用作生成器的 seq2seq 模型。
        retriever ([`RagRetriever`]):
            检索器类，封装了一个 faiss 索引，用于查询获取当前输入的上下文文档。
"""
"""


RAG_FORWARD_INPUTS_DOCSTRING = r"""
"""


@add_start_docstrings_to_model_forward(RAG_START_DOCSTRING)
class RagModel(RagPreTrainedModel):
    def __init__(
        self,
        config: Optional[PretrainedConfig] = None,
        question_encoder: Optional[PreTrainedModel] = None,
        generator: Optional[PreTrainedModel] = None,
        retriever: Optional[RagRetriever] = None,  # or maybe just use a `set_retriever(...)` method
        **kwargs,
    ):
        assert config is not None or (
            question_encoder is not None and generator is not None
        ), "Either a configuration or an question_encoder and a generator has to be provided."

        if config is None:
            # Constructing a RagConfig object from provided question_encoder and generator configurations
            config = RagConfig.from_question_encoder_generator_configs(
                question_encoder.config, generator.config, **kwargs
            )
        else:
            assert isinstance(config, self.config_class), f"config: {config} has to be of type {self.config_class}"
        super().__init__(config)
        
        if question_encoder is None:
            # If question_encoder is not provided, instantiate a default model using AutoModel
            from ..auto.modeling_auto import AutoModel
            question_encoder = AutoModel.from_config(config.question_encoder)

        if generator is None:
            # If generator is not provided, instantiate a default Seq2SeqLM model using AutoModelForSeq2SeqLM
            from ..auto.modeling_auto import AutoModelForSeq2SeqLM
            generator = AutoModelForSeq2SeqLM.from_config(config.generator)

        self.retriever = retriever
        if self.retriever is not None:
            # Ensure retriever is of type RagRetriever
            assert isinstance(
                retriever, RagRetriever
            ), f"`self.retriever` is of type {type(self.retriever)}, but should be of type `RagRetriever`"
            self.retriever = retriever

        self.question_encoder = question_encoder
        self.generator = generator

        # Initialize context encoder attributes
        self.ctx_encoder = None
        self.context_encoder_training = False

    @add_start_docstrings_to_model_forward(
        """
        A RAG-sequence model implementation. It performs RAG-sequence specific marginalization in the forward pass.
        """
    )
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
    ):
        """
        Perform a forward pass through the RAG model.

        This method implements specific marginalization for RAG-sequence models.

        Args:
            input_ids (Optional[torch.LongTensor]): Input tensor of token indices.
            attention_mask (Optional[torch.Tensor]): Mask tensor indicating which elements in the input should be attended to.
            encoder_outputs (Optional[Tuple[Tuple[torch.FloatTensor]]]): Outputs of the encoder.
            decoder_input_ids (Optional[torch.LongTensor]): Input tensor for decoder.
            decoder_attention_mask (Optional[torch.BoolTensor]): Mask tensor for decoder attention.
            past_key_values (Optional[Tuple[Tuple[torch.FloatTensor]]]): Cached key-values for faster decoding.
            doc_scores (Optional[torch.FloatTensor]): Scores indicating relevance of retrieved documents.
            context_input_ids (Optional[torch.LongTensor]): Tensor of token indices for context.
            context_attention_mask (Optional[torch.LongTensor]): Mask tensor for context attention.
            use_cache (Optional[bool]): Whether to use cached values.
            output_attentions (Optional[bool]): Whether to output attention weights.
            output_hidden_states (Optional[bool]): Whether to output hidden states.
            output_retrieved (Optional[bool]): Whether to output retrieved documents.
            n_docs (Optional[int]): Number of documents to retrieve.

        Returns:
            RetrievAugLMOutput: Object containing the model outputs.
        """
        pass
    RAG_START_DOCSTRING,
# 定义一个继承自 RagPreTrainedModel 的类，用于生成RAG（Retrieval-Augmented Generation）模型的序列
class RagSequenceForGeneration(RagPreTrainedModel):
    def __init__(
        self,
        config: Optional[PretrainedConfig] = None,
        question_encoder: Optional[PreTrainedModel] = None,
        generator: Optional[PreTrainedModel] = None,
        retriever: Optional[RagRetriever] = None,
        **kwargs,
    ):
        # 断言语句，要求提供配置信息或者问题编码器和生成器的组合之一
        assert config is not None or (
            question_encoder is not None and generator is not None
        ), "Either a configuration or an encoder and a generator has to be provided."

        # 如果未提供配置信息，则根据提供的问题编码器和生成器配置生成一个 RagConfig 对象
        if config is None:
            config = RagConfig.from_question_encoder_generator_configs(
                question_encoder.config, generator.config, **kwargs
            )
        # 调用父类的初始化方法，传入配置信息
        super().__init__(config)

        # 实例化 RAG 模型，传入配置信息、问题编码器、生成器和检索器
        self.rag = RagModel(config=config, question_encoder=question_encoder, generator=generator, retriever=retriever)

    # 设置模型的检索器
    def set_retriever(self, retriever: RagRetriever):
        self.rag.retriever = retriever

    # 设置用于训练的上下文编码器
    def set_context_encoder_for_training(self, ctx_encoder: PreTrainedModel):
        self.rag.context_encoder_training = True
        self.rag.ctx_encoder = ctx_encoder

    # 前向传播方法，接收多个输入参数，详细的参数说明由装饰器 @add_start_docstrings_to_model_forward 和 @replace_return_docstrings 提供
    @add_start_docstrings_to_model_forward(RAG_FORWARD_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=RetrievAugLMMarginOutput, config_class=_CONFIG_FOR_DOC)
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
        **kwargs,  # 需要传递给生成过程的额外参数
    ):
        pass  # 实际前向传播逻辑在 RagModel 类中定义

    # 返回模型的检索器属性
    @property
    def retriever(self):
        return self.rag.retriever

    # 返回模型的生成器属性
    @property
    def generator(self):
        return self.rag.generator

    # 返回模型的问题编码器属性
    @property
    def question_encoder(self):
        return self.rag.question_encoder

    # 使用 torch.no_grad 装饰器，表示该方法不需要计算梯度信息
    @torch.no_grad()
    # 定义一个生成方法，用于生成文本序列。
    def generate(
        self,
        input_ids: Optional[torch.LongTensor] = None,  # 输入序列的索引张量，可以为空
        attention_mask: Optional[torch.LongTensor] = None,  # 注意力掩码张量，可以为空
        context_input_ids: Optional[torch.LongTensor] = None,  # 上下文输入序列的索引张量，可以为空
        context_attention_mask: Optional[torch.LongTensor] = None,  # 上下文输入的注意力掩码张量，可以为空
        doc_scores: Optional[torch.FloatTensor] = None,  # 文档评分张量，可以为空
        do_deduplication: Optional[bool] = None,  # 是否去重，默认为True
        num_return_sequences: Optional[int] = None,  # 返回序列的数量，默认为1
        num_beams: Optional[int] = None,  # Beam搜索中的Beam大小，默认为1
        n_docs: Optional[int] = None,  # 文档数量，可以为空
        **model_kwargs,  # 其他模型相关参数，接收任意关键字参数
    ):
        # 定义一个计算负对数似然（Negative Log-Likelihood，NLL）的方法
    def get_nll(
        self,
        seq_logits,  # 序列的logits，用于计算NLL
        doc_scores,  # 文档评分，用于加权序列NLL
        target,  # 目标序列，用于计算NLL
        reduce_loss=False,  # 是否减少损失，默认为False
        epsilon=0.0,  # 平滑项，用于数值稳定性，默认为0.0
        exclude_bos_score=False,  # 是否排除起始标记得分，默认为False
        n_docs=None  # 文档数量，可以为空
        # shift tokens left
        target = torch.cat(
            [target[:, 1:], target.new(target.shape[0], 1).fill_(self.config.generator.pad_token_id)], 1
        )

        # Determine the number of documents to consider, defaulting to self.config.n_docs if not specified
        n_docs = n_docs if n_docs is not None else self.config.n_docs

        # Determine the beginning of sequence token ID (`bos_token_id`) based on model configuration
        bos_token_id = self.config.bos_token_id or self.config.generator.bos_token_id
        use_bos = bos_token_id is not None and target[:, 0].eq(bos_token_id).all()

        def _mask_pads(ll, smooth_obj):
            # Create a mask for padding tokens in the target sequence
            pad_mask = target.eq(self.config.generator.pad_token_id)
            if pad_mask.any():
                # Apply the mask to log-likelihood and smoothing objective
                ll.masked_fill_(pad_mask, 0.0)
                smooth_obj.masked_fill_(pad_mask, 0.0)
            return ll.squeeze(-1), smooth_obj.squeeze(-1)

        # Compute log softmax over sequence logits and reshape for RAG sequence marginalization
        seq_logprobs = nn.functional.log_softmax(seq_logits, dim=-1).view(
            seq_logits.shape[0] // n_docs, n_docs, -1, seq_logits.size(-1)
        )  # batch_size x n_docs x tgt_len x #vocab_size
        doc_logprobs = nn.functional.log_softmax(doc_scores, dim=1).unsqueeze(-1).unsqueeze(-1)

        # RAG-sequence marginalization
        first_token_scores = seq_logprobs[:, :, :1, :]
        second_token_scores = seq_logprobs[:, :, 1:2, :]
        remainder = seq_logprobs[:, :, 2:, :]
        rag_logprobs = torch.cat([first_token_scores, second_token_scores + doc_logprobs, remainder], dim=2)

        # Ensure target tensor matches dimensions of rag_logprobs for indexing
        target = target.unsqueeze(1).unsqueeze(-1).repeat(1, n_docs, 1, 1)
        assert target.dim() == rag_logprobs.dim()

        # Gather log probabilities corresponding to target indices and apply padding mask
        ll = rag_logprobs.gather(dim=-1, index=target)
        smooth_obj = rag_logprobs.sum(dim=-1, keepdim=True)  # total sum of all (normalized) logits

        # Apply padding mask to log-likelihood and smoothing objective
        ll, smooth_obj = _mask_pads(ll, smooth_obj)

        # Sum over tokens to compute loss, optionally excluding beginning of sequence token
        ll = ll[:, :, 1:].sum(2) if exclude_bos_score and use_bos else ll.sum(2)
        smooth_obj = smooth_obj.sum(2)
        ll = ll.logsumexp(1)  # logsumexp over docs
        smooth_obj = smooth_obj.logsumexp(1)

        # Calculate negative log-likelihood (nll) loss and smoothed loss
        nll_loss = -ll
        smooth_loss = -smooth_obj

        # Optionally reduce loss across batches
        if reduce_loss:
            nll_loss = nll_loss.sum()
            smooth_loss = smooth_loss.sum()

        # Compute final loss using nll_loss, smooth_loss, and epsilon for smoothing
        eps_i = epsilon / rag_logprobs.size(-1)
        loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
        return loss

    @staticmethod
    def _cat_and_pad(tensors, pad_token_id):
        # Concatenate tensors into a padded tensor with specified pad_token_id
        output = (
            tensors[0].new(sum([t.shape[0] for t in tensors]), max([t.shape[1] for t in tensors])).fill_(pad_token_id)
        )
        ind = 0
        for t in tensors:
            output[ind : ind + t.shape[0], : t.shape[1]] = t
            ind += t.shape[0]
        return output
"""
一个实现了RAG-token模型的类。在前向传播中执行了RAG-token特定的边缘化操作。
"""
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
        # 断言：确保提供了配置或者问题编码器和生成器的组合
        assert config is not None or (
            question_encoder is not None and generator is not None
        ), "Either a configuration or an encoder and a generator has to be provided."

        # 如果没有提供配置，则根据问题编码器和生成器的配置创建RAG配置对象
        if config is None:
            config = RagConfig.from_question_encoder_generator_configs(
                question_encoder.config, generator.config, **kwargs
            )

        # 调用父类初始化方法
        super().__init__(config)

        # 实例化RAG模型
        self.rag = RagModel(config=config, question_encoder=question_encoder, generator=generator, retriever=retriever)

    # 设置检索器
    def set_retriever(self, retriever: RagRetriever):
        self.rag.retriever = retriever

    # 设置用于训练的上下文编码器
    def set_context_encoder_for_training(self, ctx_encoder: PreTrainedModel):
        self.rag.context_encoder_training = True
        self.rag.ctx_encoder = ctx_encoder

    # 准备生成的输入
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
        # 如果已经定义了过去的键值对，则只使用最后一个decoder_input_ids
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

    # 检索器的属性
    @property
    def retriever(self):
        return self.rag.retriever

    # 生成器的属性
    @property
    def generator(self):
        return self.rag.generator

    # 问题编码器的属性
    @property
    def question_encoder(self):
        return self.rag.question_encoder

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        """Reorders cache for generation. BART-inspired but we need to take care of the extra dimension for docs"""

        def _reorder_stacked(hidden_states, new_order):
            # 计算每个文档的数量
            n_docs = hidden_states.shape[0] // new_order.shape[0]
            # 将隐藏状态重塑为 [batch_size, n_docs, ...] 的形状
            hidden_states = hidden_states.view(-1, n_docs, *hidden_states.shape[1:])
            # 根据新的顺序索引选择隐藏状态
            hidden_states = hidden_states.index_select(0, new_order)
            # 恢复原来的形状
            result = hidden_states.view(-1, *hidden_states.shape[2:])
            return result

        # 初始化重新排序后的缓存
        reordered_past = ()
        # 遍历每一层的缓存
        for layer_past in past_key_values:
            # 对每个缓存状态重新排序，并添加到结果中
            reordered_past += (
                tuple(_reorder_stacked(past_state, beam_idx.to(past_state.device)) for past_state in layer_past),
            )

        return reordered_past

    def marginalize(self, seq_logits, doc_scores, n_docs=None):
        # 如果未提供 n_docs，则使用默认值 self.config.n_docs
        n_docs = n_docs if n_docs is not None else self.config.n_docs

        # 对序列的 logits 进行 log_softmax，并重塑为 [batch_size / n_docs, n_docs, ..., num_labels]
        seq_logprobs = nn.functional.log_softmax(seq_logits, dim=-1).view(
            seq_logits.shape[0] // n_docs, n_docs, -1, seq_logits.size(-1)
        )
        # 对文档分数进行 log_softmax
        doc_logprobs = torch.log_softmax(doc_scores, dim=1)
        # 计算序列 log_probs 和文档 log_probs 的和，并进行 logsumexp 运算
        log_prob_sum = seq_logprobs + doc_logprobs.unsqueeze(-1).unsqueeze(-1)
        return torch.logsumexp(log_prob_sum, dim=1)

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
        **kwargs,  # 需要用于生成的其他参数
    ):
        # 在 forward 方法中使用 torch.no_grad()，确保不计算梯度
        @torch.no_grad()
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
    ):
        """
        Generate function for the model to generate text outputs based on given inputs.
        """
        # Implementation details are encapsulated in the class and not commented here.

    def get_input_embeddings(self):
        """
        Retrieve input embeddings from the RAG generator.
        """
        return self.rag.generator.get_input_embeddings()

    def get_output_embeddings(self):
        """
        Retrieve output embeddings from the RAG generator.
        """
        return self.rag.generator.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        """
        Set new output embeddings for the RAG generator.
        """
        return self.rag.generator.set_output_embeddings(new_embeddings)

    def shift_tokens_right(self, input_ids, start_token_id=None):
        """
        Shift input ids one token to the right, and pad with start_token_id.
        """
        if start_token_id is None:
            start_token_id = self.config.decoder_start_token_id
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
        shifted_input_ids[:, 0] = start_token_id
        return shifted_input_ids

    def get_nll(self, seq_logits, doc_scores, target, reduce_loss=False, epsilon=0.0, n_docs=None):
        """
        Calculate negative log likelihood loss for sequence logits and document scores.
        """
        n_docs = n_docs if n_docs is not None else self.config.n_docs

        # Shift tokens left and handle padding
        target = torch.cat(
            [target[:, 1:], target.new(target.shape[0], 1).fill_(self.config.generator.pad_token_id)], 1
        )

        def _mask_pads(ll, smooth_obj):
            """
            Mask padding tokens in loss calculations.
            """
            pad_mask = target.eq(self.config.generator.pad_token_id)
            if pad_mask.any():
                ll.masked_fill_(pad_mask, 0.0)
                smooth_obj.masked_fill_(pad_mask, 0.0)
            return ll.squeeze(-1), smooth_obj.squeeze(-1)

        # Marginalize logits and calculate log probabilities
        rag_logprobs = self.marginalize(seq_logits, doc_scores, n_docs)

        target = target.unsqueeze(-1)
        assert target.dim() == rag_logprobs.dim()

        # Gather log probabilities based on target indices
        ll = rag_logprobs.gather(dim=-1, index=target)
        smooth_obj = rag_logprobs.sum(dim=-1, keepdim=True)  # total sum of all (normalised) logits
        ll, smooth_obj = _mask_pads(ll, smooth_obj)
        ll = ll.sum(1)  # sum over tokens
        smooth_obj = smooth_obj.sum(1)

        # Compute final negative log likelihood loss and smooth loss
        nll_loss = -ll
        smooth_loss = -smooth_obj

        if reduce_loss:
            nll_loss = nll_loss.sum()
            smooth_loss = smooth_loss.sum()

        eps_i = epsilon / rag_logprobs.size(-1)
        loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
        return loss
```