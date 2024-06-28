# `.\generation\utils.py`

```
# coding=utf-8
# 版权声明和许可信息，指定了本文件使用的Apache License, Version 2.0许可
# 此处为代码导入所需的标准库、第三方库及自定义模块

import copy  # 导入copy模块，用于对象的浅复制和深复制操作
import inspect  # 导入inspect模块，用于获取对象信息
import warnings  # 导入warnings模块，用于警告处理
from dataclasses import dataclass  # 从dataclasses模块导入dataclass装饰器，用于简化数据类的定义
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union  # 导入类型提示相关的类和函数

import torch  # 导入PyTorch库
import torch.distributed as dist  # 导入PyTorch分布式训练相关模块
from torch import nn  # 从torch模块中导入nn模块，用于神经网络构建

from ..cache_utils import Cache, DynamicCache, StaticCache  # 导入缓存相关的自定义模块
from ..integrations.deepspeed import is_deepspeed_zero3_enabled  # 导入深度学习加速相关模块
from ..modeling_outputs import CausalLMOutputWithPast, Seq2SeqLMOutput  # 导入模型输出相关类
from ..models.auto import (  # 导入自动模型加载相关映射
    MODEL_FOR_CAUSAL_IMAGE_MODELING_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING,
    MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING,
    MODEL_FOR_VISION_2_SEQ_MAPPING,
)
from ..utils import ModelOutput, is_accelerate_available, is_torchdynamo_compiling, logging  # 导入工具类和函数
from .beam_constraints import DisjunctiveConstraint, PhrasalConstraint  # 导入束搜索相关约束类
from .beam_search import BeamScorer, BeamSearchScorer, ConstrainedBeamSearchScorer  # 导入束搜索相关评分器类
from .candidate_generator import (  # 导入候选生成器相关函数和类
    AssistedCandidateGenerator,
    CandidateGenerator,
    PromptLookupCandidateGenerator,
    _crop_past_key_values,
    _prepare_attention_mask,
    _prepare_token_type_ids,
)
from .configuration_utils import GenerationConfig, GenerationMode  # 导入生成配置和模式相关类
from .logits_process import (  # 导入logits处理相关类
    EncoderNoRepeatNGramLogitsProcessor,
    EncoderRepetitionPenaltyLogitsProcessor,
    EpsilonLogitsWarper,
    EtaLogitsWarper,
    ExponentialDecayLengthPenalty,
    ForcedBOSTokenLogitsProcessor,
    ForcedEOSTokenLogitsProcessor,
    ForceTokensLogitsProcessor,
    HammingDiversityLogitsProcessor,
    InfNanRemoveLogitsProcessor,
    LogitNormalization,
    LogitsProcessorList,
    MinLengthLogitsProcessor,
    MinNewTokensLengthLogitsProcessor,
    NoBadWordsLogitsProcessor,
    NoRepeatNGramLogitsProcessor,
    PrefixConstrainedLogitsProcessor,
    RepetitionPenaltyLogitsProcessor,
    SequenceBiasLogitsProcessor,
    SuppressTokensAtBeginLogitsProcessor,
    SuppressTokensLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
    TypicalLogitsWarper,
    UnbatchedClassifierFreeGuidanceLogitsProcessor,
)
from .stopping_criteria import (  # 导入停止条件相关类
    MaxLengthCriteria,
    MaxTimeCriteria,
    StoppingCriteria,
    StoppingCriteriaList,
    validate_stopping_criteria,
)

if TYPE_CHECKING:
    # 从相对路径导入模块中的PreTrainedModel类，用于模型预训练
    # 从相对路径导入streamers模块中的BaseStreamer类，用作基础流处理器
    from ..modeling_utils import PreTrainedModel
    from .streamers import BaseStreamer
# 获取名为__name__的模块的日志记录器对象
logger = logging.get_logger(__name__)

# 如果加速可用，导入加速相关的钩子函数和模块扩展函数
if is_accelerate_available():
    from accelerate.hooks import AlignDevicesHook, add_hook_to_module

# 静态缓存类型映射，将字符串"static"映射到StaticCache类
NEED_SETUP_CACHE_CLASSES_MAPPING = {
    "static": StaticCache,
}

# 数据类，用于生成仅解码器输出的模型结果，继承自ModelOutput类
@dataclass
class GenerateDecoderOnlyOutput(ModelOutput):
    """
    Outputs of decoder-only generation models, when using non-beam methods.
    """
    Args:
        sequences (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            The generated sequences. The second dimension (sequence_length) is either equal to `max_length` or shorter
            if all batches finished early due to the `eos_token_id`.
        scores (`tuple(torch.FloatTensor)` *optional*, returned when `output_scores=True` is passed or when `config.output_scores=True`):
            Processed prediction scores of the language modeling head (scores for each vocabulary token before SoftMax)
            at each generation step. Tuple of `torch.FloatTensor` with up to `max_new_tokens` elements (one element for
            each generated token), with each tensor of shape `(batch_size, config.vocab_size)`.
        logits (`tuple(torch.FloatTensor)` *optional*, returned when `output_logits=True` is passed or when `config.output_logits=True`):
            Unprocessed prediction scores of the language modeling head (scores for each vocabulary token before SoftMax)
            at each generation step. Tuple of `torch.FloatTensor` with up to `max_new_tokens` elements (one element for
            each generated token), with each tensor of shape `(batch_size, config.vocab_size)`.
        attentions (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_attentions=True` is passed or `config.output_attentions=True`):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            `torch.FloatTensor` of shape `(batch_size, num_heads, generated_length, sequence_length)`.
        hidden_states (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            `torch.FloatTensor` of shape `(batch_size, generated_length, hidden_size)`.
        past_key_values (`tuple(tuple(torch.FloatTensor)))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            NOTE: some models have a different `past_key_values` format, confirm with the model's documentation.
            Usually a Tuple (one element for each layer of the decoder) of tuples (two elements, key tensor and value
            tensor). The first Tuple is of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and optionally if
            `config.is_encoder_decoder=True` 2 additional tensors of shape `(batch_size, num_heads,
            encoder_sequence_length, embed_size_per_head)`.
    # 声明一个可选的变量 hidden_states，其类型是一个元组，包含一个元组，该元组中包含一个 torch.FloatTensor 类型的值
    hidden_states: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    
    # 声明一个可选的变量 past_key_values，其类型是一个元组，包含一个元组，该元组中包含一个元组，该元组中包含一个 torch.FloatTensor 类型的值
    past_key_values: Optional[Tuple[Tuple[Tuple[torch.FloatTensor]]]] = None
# 用于生成编码器-解码器模型的输出，非使用 Beam 方法时的情况
@dataclass
class GenerateEncoderDecoderOutput(ModelOutput):
    """
    编码器-解码器生成模型的输出，当不使用 Beam 方法时。

    """

    sequences: torch.LongTensor = None  # 生成的序列（token ID）
    scores: Optional[Tuple[torch.FloatTensor]] = None  # 每个生成序列的分数
    logits: Optional[Tuple[torch.FloatTensor]] = None  # 每个生成序列的 logits
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None  # 编码器注意力权重
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None  # 编码器隐藏状态
    decoder_attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None  # 解码器注意力权重
    cross_attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None  # 交叉注意力权重
    decoder_hidden_states: Optional[Tuple[Tuple[torch.FloatTensor]]] = None  # 解码器隐藏状态
    past_key_values: Optional[Tuple[Tuple[Tuple[torch.FloatTensor]]]] = None  # 额外的过去键值（针对 Transformer 模型）

# 用于生成仅解码器模型的输出，使用 Beam 方法时的情况
@dataclass
class GenerateBeamDecoderOnlyOutput(ModelOutput):
    """
    解码器生成模型的输出，仅在使用 Beam 方法时。

    """

    sequences: torch.LongTensor = None  # 生成的序列（token ID）
    sequences_scores: Optional[torch.FloatTensor] = None  # 生成序列的分数
    scores: Optional[Tuple[torch.FloatTensor]] = None  # 每个生成序列的分数
    logits: Optional[Tuple[torch.FloatTensor]] = None  # 每个生成序列的 logits
    beam_indices: Optional[torch.LongTensor] = None  # Beam 搜索时使用的索引
    attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None  # 注意力权重
    hidden_states: Optional[Tuple[Tuple[torch.FloatTensor]]] = None  # 隐藏状态
    past_key_values: Optional[Tuple[Tuple[Tuple[torch.FloatTensor]]]] = None  # 额外的过去键值（针对 Transformer 模型）

# 用于生成编码器-解码器模型的输出，使用 Beam 方法时的情况
@dataclass
class GenerateBeamEncoderDecoderOutput(ModelOutput):
    """
    编码器-解码器生成模型的输出，使用 Beam 方法时。

    """

    sequences: torch.LongTensor = None  # 生成的序列（token ID）
    sequences_scores: Optional[torch.FloatTensor] = None  # 生成序列的分数
    scores: Optional[Tuple[torch.FloatTensor]] = None  # 每个生成序列的分数
    logits: Optional[Tuple[torch.FloatTensor]] = None  # 每个生成序列的 logits
    beam_indices: Optional[torch.LongTensor] = None  # Beam 搜索时使用的索引
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None  # 编码器注意力权重
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None  # 编码器隐藏状态
    decoder_attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None  # 解码器注意力权重
    cross_attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None  # 交叉注意力权重
    decoder_hidden_states: Optional[Tuple[Tuple[torch.FloatTensor]]] = None  # 解码器隐藏状态
    past_key_values: Optional[Tuple[Tuple[Tuple[torch.FloatTensor]]]] = None  # 额外的过去键值（针对 Transformer 模型）

# 以下是为了向后兼容而保留的等效类
GreedySearchDecoderOnlyOutput = GenerateDecoderOnlyOutput  # 贪婪搜索解码器模型的输出
ContrastiveSearchDecoderOnlyOutput = GenerateDecoderOnlyOutput  # 对比搜索解码器模型的输出
SampleDecoderOnlyOutput = GenerateDecoderOnlyOutput  # 示例解码器模型的输出

ContrastiveSearchEncoderDecoderOutput = GenerateEncoderDecoderOutput  # 对比搜索编码器-解码器模型的输出
GreedySearchEncoderDecoderOutput = GenerateEncoderDecoderOutput  # 贪婪搜索编码器-解码器模型的输出
SampleEncoderDecoderOutput = GenerateEncoderDecoderOutput  # 示例编码器-解码器模型的输出

BeamSearchDecoderOnlyOutput = GenerateBeamDecoderOnlyOutput  # Beam 搜索解码器模型的输出
BeamSampleDecoderOnlyOutput = GenerateBeamDecoderOnlyOutput  # Beam 示例解码器模型的输出

BeamSearchEncoderDecoderOutput = GenerateBeamEncoderDecoderOutput  # Beam 搜索编码器-解码器模型的输出
BeamSampleEncoderDecoderOutput = GenerateBeamEncoderDecoderOutput  # Beam 示例编码器-解码器模型的输出

GreedySearchOutput = Union[GreedySearchEncoderDecoderOutput, GreedySearchDecoderOnlyOutput]  # 贪婪搜索的输出类型
# Typing shortcuts for specific types of model outputs
SampleOutput = Union[SampleEncoderDecoderOutput, SampleDecoderOnlyOutput]
BeamSearchOutput = Union[BeamSearchEncoderDecoderOutput, BeamSearchDecoderOnlyOutput]
BeamSampleOutput = Union[BeamSampleEncoderDecoderOutput, BeamSampleDecoderOnlyOutput]
ContrastiveSearchOutput = Union[ContrastiveSearchEncoderDecoderOutput, ContrastiveSearchDecoderOnlyOutput]

# Typing shortcut for non-beam text generation output
GenerateNonBeamOutput = Union[GenerateDecoderOnlyOutput, GenerateEncoderDecoderOutput]
# Typing shortcut for beam search text generation output
GenerateBeamOutput = Union[GenerateBeamDecoderOnlyOutput, GenerateBeamEncoderDecoderOutput]
# Typing shortcut for any text generation output
GenerateOutput = Union[GenerateNonBeamOutput, GenerateBeamOutput]


class GenerationMixin:
    """
    A class containing all functions for auto-regressive text generation, to be used as a mixin in [`PreTrainedModel`].

    The class exposes [`~generation.GenerationMixin.generate`], which can be used for:
        - *greedy decoding* by calling [`~generation.GenerationMixin._greedy_search`] if `num_beams=1` and
          `do_sample=False`
        - *contrastive search* by calling [`~generation.GenerationMixin._contrastive_search`] if `penalty_alpha>0` and
          `top_k>1`
        - *multinomial sampling* by calling [`~generation.GenerationMixin._sample`] if `num_beams=1` and
          `do_sample=True`
        - *beam-search decoding* by calling [`~generation.GenerationMixin._beam_search`] if `num_beams>1` and
          `do_sample=False`
        - *beam-search multinomial sampling* by calling [`~generation.GenerationMixin._beam_sample`] if `num_beams>1`
          and `do_sample=True`
        - *diverse beam-search decoding* by calling [`~generation.GenerationMixin._group_beam_search`], if `num_beams>1`
          and `num_beam_groups>1`
        - *constrained beam-search decoding* by calling [`~generation.GenerationMixin._constrained_beam_search`], if
          `constraints!=None` or `force_words_ids!=None`
        - *assisted decoding* by calling [`~generation.GenerationMixin._assisted_decoding`], if
            `assistant_model` or `prompt_lookup_num_tokens` is passed to `.generate()`

    You do not need to call any of the above methods directly. Pass custom parameter values to 'generate' instead. To
    learn more about decoding strategies refer to the [text generation strategies guide](../generation_strategies).
    """

    def prepare_inputs_for_generation(self, *args, **kwargs):
        # Raise an error if this method is not implemented in the subclass
        raise NotImplementedError(
            "A model class needs to define a `prepare_inputs_for_generation` method in order to use `.generate()`."
        )

    def _prepare_model_inputs(
        self,
        inputs: Optional[torch.Tensor] = None,
        bos_token_id: Optional[int] = None,
        model_kwargs: Optional[Dict[str, torch.Tensor]] = None,
    ):
        # Internal method for preparing model inputs for text generation
        ...

    def _maybe_initialize_input_ids_for_generation(
        self,
        inputs: Optional[torch.Tensor] = None,
        bos_token_id: Optional[int] = None,
        model_kwargs: Optional[Dict[str, torch.Tensor]] = None,
    ):
        # Internal method to initialize input IDs for text generation if necessary
        ...
    ) -> torch.LongTensor:
        """Initializes input ids for generation, if necessary."""
        # 如果已经提供了输入，则直接返回输入
        if inputs is not None:
            return inputs

        # 获取模型关键字参数中的 encoder_outputs
        encoder_outputs = model_kwargs.get("encoder_outputs")
        # 如果模型是编码-解码模型且 encoder_outputs 不为空
        if self.config.is_encoder_decoder and encoder_outputs is not None:
            # 创建一个与 encoder_outputs 最后一层隐藏状态相同形状的输入 id 张量，填充值为 -100
            shape = encoder_outputs.last_hidden_state.size()[:-1]
            return torch.ones(shape, dtype=torch.long, device=self.device) * -100

        # 如果未提供 input_ids 且未定义 bos_token_id，则引发错误
        if bos_token_id is None:
            raise ValueError("`bos_token_id` has to be defined when no `input_ids` are provided.")

        # 如果 model_kwargs 中有某些张量，则可以从中推断出批量大小
        batch_size = 1
        for value in model_kwargs.values():
            if isinstance(value, torch.Tensor):
                batch_size = value.shape[0]
                break

        # 如果 model_kwargs 中包含 "inputs_embeds" 键
        if "inputs_embeds" in model_kwargs:
            # 返回一个形状为 (batch_size, 0) 的全 1 张量，dtype 为 torch.long
            return torch.ones((batch_size, 0), dtype=torch.long, device=self.device)
        # 否则返回一个形状为 (batch_size, 1) 的全 bos_token_id 值的张量，dtype 为 torch.long
        return torch.ones((batch_size, 1), dtype=torch.long, device=self.device) * bos_token_id

    def _prepare_attention_mask_for_generation(
        self,
        inputs: torch.Tensor,
        pad_token_id: Optional[int],
        eos_token_id: Optional[Union[int, List[int]]],
    ) -> torch.LongTensor:
        # 检查输入是否为 input_ids 且已被填充，只有这种情况下才定义 attention_mask
        is_input_ids = len(inputs.shape) == 2 and inputs.dtype in [torch.int, torch.long]
        is_pad_token_in_inputs = (pad_token_id is not None) and (pad_token_id in inputs)
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        is_pad_token_not_equal_to_eos_token_id = (eos_token_id is None) or (pad_token_id not in eos_token_id)

        # 如果输入是 input_ids 且已填充，并且填充标记不等于 eos_token_id，则返回 attention_mask
        if is_input_ids and is_pad_token_in_inputs and is_pad_token_not_equal_to_eos_token_id:
            return inputs.ne(pad_token_id).long()
        else:
            # 否则返回一个形状与 inputs 的前两维相同的全 1 张量，dtype 为 torch.long
            return torch.ones(inputs.shape[:2], dtype=torch.long, device=inputs.device)

    def _prepare_encoder_decoder_kwargs_for_generation(
        self, inputs_tensor: torch.Tensor, model_kwargs, model_input_name: Optional[str] = None
        # 1. 获取编码器
        encoder = self.get_encoder()

        # 2. 兼容加速大模型推断：确保编码器在与输入相同的设备上输出结果
        if hasattr(self, "hf_device_map"):
            # 如果编码器有 `_hf_hook` 属性，设置其 `io_same_device` 为 True
            if hasattr(encoder, "_hf_hook"):
                encoder._hf_hook.io_same_device = True
            # 否则，向编码器添加一个 AlignDevicesHook，设置 `io_same_device` 为 True
            else:
                add_hook_to_module(encoder, AlignDevicesHook(io_same_device=True))

        # 3. 从模型参数中准备编码器的参数和关键字参数
        irrelevant_prefix = ["decoder_", "cross_attn", "use_cache"]
        # 从 `model_kwargs` 中选择与编码器相关的参数和值
        encoder_kwargs = {
            argument: value
            for argument, value in model_kwargs.items()
            if not any(argument.startswith(p) for p in irrelevant_prefix)
        }
        # 检查编码器的输入签名，确定是否支持 `kwargs` 或 `model_kwargs`
        encoder_signature = set(inspect.signature(encoder.forward).parameters)
        encoder_accepts_wildcard = "kwargs" in encoder_signature or "model_kwargs" in encoder_signature
        if not encoder_accepts_wildcard:
            # 如果编码器不支持通配符参数，仅选择编码器签名中存在的参数和值
            encoder_kwargs = {
                argument: value for argument, value in encoder_kwargs.items() if argument in encoder_signature
            }

        # 4. 确保编码器返回 `ModelOutput`
        model_input_name = model_input_name if model_input_name is not None else self.main_input_name
        encoder_kwargs["return_dict"] = True
        encoder_kwargs[model_input_name] = inputs_tensor

        # 调用编码器并将结果保存在 `model_kwargs` 的 `encoder_outputs` 键中
        model_kwargs["encoder_outputs"]: ModelOutput = encoder(**encoder_kwargs)

        return model_kwargs

    # 准备用于生成的解码器输入 ID
    def _prepare_decoder_input_ids_for_generation(
        self,
        batch_size: int,
        model_input_name: str,
        model_kwargs: Dict[str, torch.Tensor],
        decoder_start_token_id: Union[int, List[int]] = None,
        bos_token_id: int = None,
        device: torch.device = None,
    ) -> Dict[str, torch.Tensor]:
        ...

    # 获取解码器起始标记 ID
    def _get_decoder_start_token_id(
        self, decoder_start_token_id: Union[int, List[int]] = None, bos_token_id: int = None
    ) -> int:
        ...

    # 扩展用于生成的输入
    @staticmethod
    def _expand_inputs_for_generation(
        expand_size: int = 1,
        is_encoder_decoder: bool = False,
        input_ids: Optional[torch.LongTensor] = None,
        **model_kwargs,
    ) -> Dict[str, torch.Tensor]:
        ...
    ) -> Tuple[torch.LongTensor, Dict[str, Any]]:
        """Expands tensors from [batch_size, ...] to [batch_size * expand_size, ...]"""
        # 定义函数签名，指定返回类型为元组，包含一个长整型张量和一个任意类型字典

        def _expand_dict_for_generation(dict_to_expand):
            # 为生成过程扩展字典中的张量
            for key in dict_to_expand:
                if dict_to_expand[key] is not None and isinstance(dict_to_expand[key], torch.Tensor):
                    dict_to_expand[key] = dict_to_expand[key].repeat_interleave(expand_size, dim=0)
            return dict_to_expand

        if input_ids is not None:
            input_ids = input_ids.repeat_interleave(expand_size, dim=0)
            # 如果输入 ID 不为空，则按照指定的扩展大小在指定维度上重复扩展

        model_kwargs = _expand_dict_for_generation(model_kwargs)
        # 扩展模型参数字典中的张量

        if is_encoder_decoder:
            if model_kwargs.get("encoder_outputs") is None:
                raise ValueError("If `is_encoder_decoder` is True, make sure that `encoder_outputs` is defined.")
            model_kwargs["encoder_outputs"] = _expand_dict_for_generation(model_kwargs["encoder_outputs"])
            # 如果是编码器-解码器模型，确保编码器输出在模型参数中被定义，并进行扩展

        return input_ids, model_kwargs
        # 返回扩展后的输入 ID 和模型参数字典

    def _extract_past_from_model_output(self, outputs: ModelOutput, standardize_cache_format: bool = False):
        past_key_values = None
        if "past_key_values" in outputs:
            past_key_values = outputs.past_key_values
        elif "mems" in outputs:
            past_key_values = outputs.mems
        elif "past_buckets_states" in outputs:
            past_key_values = outputs.past_buckets_states
        # 从模型输出中提取过去的键-值对

        # Bloom fix: standardizes the cache format when requested
        if standardize_cache_format and hasattr(self, "_convert_to_standard_cache"):
            batch_size = outputs.logits.shape[0]
            past_key_values = self._convert_to_standard_cache(past_key_values, batch_size=batch_size)
            # 在请求时，如果需要，标准化缓存格式

        return past_key_values
        # 返回提取的过去键-值对

    def _update_model_kwargs_for_generation(
        self,
        outputs: ModelOutput,
        model_kwargs: Dict[str, Any],
        is_encoder_decoder: bool = False,
        standardize_cache_format: bool = False,
        # 更新用于生成的模型参数字典
    ) -> Dict[str, Any]:
        # 更新 model_kwargs 中的 past_key_values，从模型输出中提取过去的键值
        model_kwargs["past_key_values"] = self._extract_past_from_model_output(
            outputs, standardize_cache_format=standardize_cache_format
        )
        # 如果 outputs 有 state 属性，则更新 model_kwargs 中的 state
        if getattr(outputs, "state", None) is not None:
            model_kwargs["state"] = outputs.state

        # 更新 token_type_ids，使用最后一个值进行扩展
        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = torch.cat([token_type_ids, token_type_ids[:, -1].unsqueeze(-1)], dim=-1)

        # 如果不是 encoder-decoder 架构
        if not is_encoder_decoder:
            # 更新 attention_mask
            if "attention_mask" in model_kwargs:
                attention_mask = model_kwargs["attention_mask"]
                model_kwargs["attention_mask"] = torch.cat(
                    [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
                )
        else:
            # 更新 decoder_attention_mask
            if "decoder_attention_mask" in model_kwargs:
                decoder_attention_mask = model_kwargs["decoder_attention_mask"]
                model_kwargs["decoder_attention_mask"] = torch.cat(
                    [decoder_attention_mask, decoder_attention_mask.new_ones((decoder_attention_mask.shape[0], 1))],
                    dim=-1,
                )

        # 如果 model_kwargs 中存在 cache_position 并且不为 None，则更新 cache_position
        if "cache_position" in model_kwargs and model_kwargs["cache_position"] is not None:
            model_kwargs["cache_position"] = model_kwargs["cache_position"][-1:] + 1

        # 返回更新后的 model_kwargs
        return model_kwargs

    # 抛出未实现错误，提示在当前类的模块中实现 _reorder_cache 函数以启用 beam search
    def _reorder_cache(self, past_key_values, beam_idx):
        raise NotImplementedError(
            f"Make sure that a `_reorder_cache` function is correctly implemented in {self.__class__.__module__} to"
            f" enable beam search for {self.__class__}"
        )

    # 返回用于辅助生成的候选生成器
    def _get_candidate_generator(
        self,
        generation_config: GenerationConfig,
        input_ids: torch.LongTensor,
        inputs_tensor: torch.Tensor,
        assistant_model: "PreTrainedModel",
        logits_processor: LogitsProcessorList,
        model_kwargs: Dict,
    ) -> CandidateGenerator:
        """
        Returns the candidate generator to be used in `assisted_generation`
        """
        # 如果指定了 prompt_lookup_num_tokens，则使用 PromptLookupCandidateGenerator
        if generation_config.prompt_lookup_num_tokens is not None:
            candidate_generator = PromptLookupCandidateGenerator(
                num_output_tokens=generation_config.prompt_lookup_num_tokens,
                max_matching_ngram_size=generation_config.max_matching_ngram_size,
            )
        else:
            # 否则使用 AssistedCandidateGenerator
            candidate_generator = AssistedCandidateGenerator(
                input_ids=input_ids,
                assistant_model=assistant_model,
                generation_config=generation_config,
                logits_processor=logits_processor,
                model_kwargs=model_kwargs,
                inputs_tensor=inputs_tensor,
            )
        return candidate_generator
    def _get_logits_warper(
        self,
        generation_config: GenerationConfig,
    ) -> LogitsProcessorList:
        """
        This class returns a [`LogitsProcessorList`] list object that contains all relevant [`LogitsWarper`] instances
        used for multinomial sampling.
        """

        # instantiate warpers list
        warpers = LogitsProcessorList()

        # In beam methods, we need to keep at least one non-eos token to explore continuations that might have a
        # better score (i.e. keep len(list(generation_config.eos_token_id)) + 1)
        if generation_config.num_beams > 1:
            if isinstance(generation_config.eos_token_id, list):
                min_tokens_to_keep = len(generation_config.eos_token_id) + 1
            else:
                min_tokens_to_keep = 2
        else:
            min_tokens_to_keep = 1

        # the following idea is largely copied from this PR: https://github.com/huggingface/transformers/pull/5420/files
        # all samplers can be found in `generation_utils_samplers.py`
        
        # Apply temperature warping if temperature is defined and not equal to 1.0
        if generation_config.temperature is not None and generation_config.temperature != 1.0:
            warpers.append(TemperatureLogitsWarper(generation_config.temperature))
        
        # Apply top-k warping if top-k is defined and not equal to 0
        if generation_config.top_k is not None and generation_config.top_k != 0:
            warpers.append(TopKLogitsWarper(top_k=generation_config.top_k, min_tokens_to_keep=min_tokens_to_keep))
        
        # Apply top-p warping if top-p is defined and less than 1.0
        if generation_config.top_p is not None and generation_config.top_p < 1.0:
            warpers.append(TopPLogitsWarper(top_p=generation_config.top_p, min_tokens_to_keep=min_tokens_to_keep))
        
        # Apply typical-p warping if typical-p is defined and less than 1.0
        if generation_config.typical_p is not None and generation_config.typical_p < 1.0:
            warpers.append(
                TypicalLogitsWarper(mass=generation_config.typical_p, min_tokens_to_keep=min_tokens_to_keep)
            )
        
        # Apply epsilon cutoff warping if epsilon cutoff is defined and within (0, 1)
        if generation_config.epsilon_cutoff is not None and 0.0 < generation_config.epsilon_cutoff < 1.0:
            warpers.append(
                EpsilonLogitsWarper(epsilon=generation_config.epsilon_cutoff, min_tokens_to_keep=min_tokens_to_keep)
            )
        
        # Apply eta cutoff warping if eta cutoff is defined and within (0, 1)
        if generation_config.eta_cutoff is not None and 0.0 < generation_config.eta_cutoff < 1.0:
            warpers.append(
                EtaLogitsWarper(epsilon=generation_config.eta_cutoff, min_tokens_to_keep=min_tokens_to_keep)
            )
        
        # `LogitNormalization` should always be the last logit processor, when present
        # Apply logit normalization if renormalize_logits flag is True
        if generation_config.renormalize_logits is True:
            warpers.append(LogitNormalization())
        
        # Return the list of warpers containing all relevant LogitsWarper instances
        return warpers
    # 获取 logits 处理器函数，根据给定的配置和参数
    def _get_logits_processor(
        self,
        generation_config: GenerationConfig,  # 生成配置对象
        input_ids_seq_length: int,  # 输入的序列长度
        encoder_input_ids: torch.LongTensor,  # 编码器输入的张量
        prefix_allowed_tokens_fn: Callable[[int, torch.Tensor], List[int]],  # 可以使用的前缀令牌函数
        logits_processor: Optional[LogitsProcessorList],  # logits 处理器的可选列表
        model_kwargs: Optional[Dict[str, Any]] = None,  # 模型参数的可选字典，默认为空
        negative_prompt_ids: Optional[torch.Tensor] = None,  # 负面提示的可选张量，默认为空
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,  # 负面提示的注意力掩码，可选，默认为空
    ):
        # 定义 stopping_criteria 对象并初始化为空列表
        criteria = StoppingCriteriaList()
        
        # 如果生成配置中指定了最大长度
        if generation_config.max_length is not None:
            # 从模型配置中获取最大位置嵌入数
            max_position_embeddings = getattr(self.config, "max_position_embeddings", None)
            # 向 criteria 中添加最大长度的停止条件
            criteria.append(
                MaxLengthCriteria(
                    max_length=generation_config.max_length,
                    max_position_embeddings=max_position_embeddings,
                )
            )
        
        # 如果生成配置中指定了最大时间
        if generation_config.max_time is not None:
            # 向 criteria 中添加最大时间的停止条件
            criteria.append(MaxTimeCriteria(max_time=generation_config.max_time))
        
        # 将自定义的 stopping_criteria 合并到 criteria 中
        criteria = self._merge_criteria_processor_list(criteria, stopping_criteria)
        
        # 返回最终的 criteria 列表
        return criteria

    # 合并默认列表和自定义列表的 logits 处理器或停止条件
    def _merge_criteria_processor_list(
        self,
        default_list: Union[LogitsProcessorList, StoppingCriteriaList],  # 默认的处理器或停止条件列表
        custom_list: Union[LogitsProcessorList, StoppingCriteriaList],  # 自定义的处理器或停止条件列表
    ) -> Union[LogitsProcessorList, StoppingCriteriaList]:  # 返回合并后的处理器或停止条件列表
        # 如果自定义列表为空，直接返回默认列表
        if len(custom_list) == 0:
            return default_list
        
        # 遍历默认列表
        for default in default_list:
            # 遍历自定义列表
            for custom in custom_list:
                # 如果自定义的对象类型和默认的对象类型相同
                if type(custom) is type(default):
                    # 确定对象类型是停止条件还是 logits 处理器
                    object_type = "stopping criteria" if isinstance(custom, StoppingCriteria) else "logits processor"
                    # 抛出值错误，提示不允许自定义与默认相同类型的处理器或条件
                    raise ValueError(
                        f"A custom {object_type} of type {type(custom)} with values {custom} has been passed to"
                        f" `.generate()`, but it has already been created with the values {default}. {default} has been"
                        " created by passing the corresponding arguments to generate or by the model's config default"
                        f" values. If you just want to change the default values of {object_type} consider passing"
                        f" them as arguments to `.generate()` instead of using a custom {object_type}."
                    )
        
        # 将自定义列表的内容扩展到默认列表中
        default_list.extend(custom_list)
        
        # 返回合并后的默认列表
        return default_list

    # 计算转移分数的函数
    def compute_transition_scores(
        self,
        sequences: torch.Tensor,  # 序列张量
        scores: Tuple[torch.Tensor],  # 分数元组
        beam_indices: Optional[torch.Tensor] = None,  # 光束索引的可选张量，默认为空
        normalize_logits: bool = False,  # 是否对 logits 进行归一化，默认为 False
    def _validate_model_class(self):
        """
        Confirms that the model class is compatible with generation. If not, raises an exception that points to the
        right class to use.
        """
        # 检查当前模型是否能够生成文本
        if not self.can_generate():
            # 可生成的模型映射列表
            generate_compatible_mappings = [
                MODEL_FOR_CAUSAL_LM_MAPPING,
                MODEL_FOR_CAUSAL_IMAGE_MODELING_MAPPING,
                MODEL_FOR_VISION_2_SEQ_MAPPING,
                MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING,
                MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING,
            ]
            generate_compatible_classes = set()
            # 遍历可生成的模型映射列表，获取支持的模型类名集合
            for model_mapping in generate_compatible_mappings:
                supported_models = model_mapping.get(type(self.config), default=None)
                if supported_models is not None:
                    generate_compatible_classes.add(supported_models.__name__)
            # 出现异常的错误信息
            exception_message = (
                f"The current model class ({self.__class__.__name__}) is not compatible with `.generate()`, as "
                "it doesn't have a language model head."
            )
            # 如果存在兼容的模型类名集合，则添加到异常信息中
            if generate_compatible_classes:
                exception_message += f" Please use one of the following classes instead: {generate_compatible_classes}"
            # 抛出类型错误异常，包含详细的异常信息
            raise TypeError(exception_message)
    # 执行与生成长度相关的验证，包括警告和错误处理

    # 1. 针对参数化不良的最大长度警告
    if has_default_max_length and generation_config.max_new_tokens is None and generation_config.max_length == 20:
        # 如果使用了默认的 `max_length`（=20）来控制生成长度，会发出警告
        warnings.warn(
            f"Using the model-agnostic default `max_length` (={generation_config.max_length}) to control the "
            "generation length. We recommend setting `max_new_tokens` to control the maximum length of the "
            "generation.",
            UserWarning,
        )
    
    # 如果输入的ids长度超过了指定的最大长度，会引发异常
    if input_ids_length >= generation_config.max_length:
        input_ids_string = "decoder_input_ids" if self.config.is_encoder_decoder else "input_ids"
        raise ValueError(
            f"Input length of {input_ids_string} is {input_ids_length}, but `max_length` is set to"
            f" {generation_config.max_length}. This can lead to unexpected behavior. You should consider"
            " increasing `max_length` or, better yet, setting `max_new_tokens`."
        )

    # 2. 由于不可行的参数组合，发出最小长度警告
    min_length_error_suffix = (
        " Generation will stop at the defined maximum length. You should decrease the minimum length and/or "
        "increase the maximum length."
    )
    if has_default_max_length:
        min_length_error_suffix += (
            f" Note that `max_length` is set to {generation_config.max_length}, its default value."
        )
    
    # 如果设定了最小长度，并且该长度大于最大长度，则发出警告
    if generation_config.min_length is not None and generation_config.min_length > generation_config.max_length:
        warnings.warn(
            f"Unfeasible length constraints: `min_length` ({generation_config.min_length}) is larger than"
            f" the maximum possible length ({generation_config.max_length})." + min_length_error_suffix,
            UserWarning,
        )
    
    # 如果设置了最小新token数量，并且计算后的最小长度超过了最大长度，则发出警告
    if generation_config.min_new_tokens is not None:
        min_length = generation_config.min_new_tokens + input_ids_length
        if min_length > generation_config.max_length:
            warnings.warn(
                f"Unfeasible length constraints: `min_new_tokens` ({generation_config.min_new_tokens}), when "
                f"added to the prompt length ({input_ids_length}), is larger than"
                f" the maximum possible length ({generation_config.max_length})." + min_length_error_suffix,
                UserWarning,
            )
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
        synced_gpus: Optional[bool] = None,
        assistant_model: Optional["PreTrainedModel"] = None,
        streamer: Optional["BaseStreamer"] = None,
        negative_prompt_ids: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """
        Generates sequences based on the provided inputs and configuration.

        Args:
            inputs (Optional[torch.Tensor]): Input tensor for generation.
            generation_config (Optional[GenerationConfig]): Configuration for generation.
            logits_processor (Optional[LogitsProcessorList]): Processors for logits during generation.
            stopping_criteria (Optional[StoppingCriteriaList]): Criteria for stopping generation.
            prefix_allowed_tokens_fn (Optional[Callable[[int, torch.Tensor], List[int]]]): Function to allow tokens during generation.
            synced_gpus (Optional[bool]): Whether to synchronize generation across GPUs.
            assistant_model (Optional["PreTrainedModel"]): Model used for generation assistance.
            streamer (Optional["BaseStreamer"]): Streamer for generation.
            negative_prompt_ids (Optional[torch.Tensor]): IDs for negative prompts.
            negative_prompt_attention_mask (Optional[torch.Tensor]): Attention mask for negative prompts.
            **kwargs: Additional keyword arguments.

        Returns:
            dict: Dictionary containing generated sequences and other relevant outputs.
        """
        ...

    def _has_unfinished_sequences(self, this_peer_finished: bool, synced_gpus: bool, device: torch.device) -> bool:
        """
        Returns whether there are still unfinished sequences on the specified device.

        Args:
            this_peer_finished (bool): Flag indicating if the current peer has finished generation.
            synced_gpus (bool): Whether generation is synchronized across GPUs.
            device (torch.device): Device on which generation is performed.

        Returns:
            bool: True if there are unfinished sequences, False otherwise.
        """
        if synced_gpus:
            # Under synced_gpus, ensure all GPUs complete their sequence generation.
            this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(device)
            # Send 0.0 if this peer finished, 1.0 otherwise
            dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
            # Check if all peers finished (sum should be 0.0 if all finished)
            if this_peer_finished_flag.item() == 0.0:
                return False
        elif this_peer_finished:
            return False
        return True

    def contrastive_search(self, *args, **kwargs):
        """
        Deprecated method for performing contrastive search. Use `generate` or a custom generation loop instead.

        Args:
            *args: Positional arguments passed to `_contrastive_search`.
            **kwargs: Keyword arguments passed to `_contrastive_search`.

        Returns:
            Any: Result from `_contrastive_search`.
        """
        logger.warning_once(
            "Calling `contrastive_search` directly is deprecated and will be removed in v4.41. Use `generate` or a "
            "custom generation loop instead.",
        )
        return self._contrastive_search(*args, **kwargs)

    @torch.no_grad()
    def _contrastive_search(
        self,
        input_ids: torch.LongTensor,
        top_k: Optional[int] = 1,
        penalty_alpha: Optional[float] = 0,
        logits_processor: Optional[LogitsProcessorList] = None,
        logits_warper: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        output_logits: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        synced_gpus: bool = False,
        streamer: Optional["BaseStreamer"] = None,
        sequential: Optional[bool] = None,
        **model_kwargs,
    ):
        """
        Performs contrastive search to generate sequences based on the input_ids and additional arguments.

        Args:
            input_ids (torch.LongTensor): Input tensor containing token IDs.
            top_k (Optional[int]): Number of top-k results to consider.
            penalty_alpha (Optional[float]): Penalty factor for contrastive search.
            logits_processor (Optional[LogitsProcessorList]): Processors for logits during contrastive search.
            logits_warper (Optional[LogitsProcessorList]): Processors for logits warping during contrastive search.
            stopping_criteria (Optional[StoppingCriteriaList]): Criteria for stopping contrastive search.
            pad_token_id (Optional[int]): Token ID for padding.
            eos_token_id (Optional[Union[int, List[int]]]): Token ID(s) for end-of-sequence.
            output_attentions (Optional[bool]): Whether to output attention weights.
            output_hidden_states (Optional[bool]): Whether to output hidden states.
            output_scores (Optional[bool]): Whether to output scores.
            output_logits (Optional[bool]): Whether to output logits.
            return_dict_in_generate (Optional[bool]): Whether to return results in a dictionary format.
            synced_gpus (bool): Whether generation is synchronized across GPUs.
            streamer (Optional["BaseStreamer"]): Streamer for contrastive search.
            sequential (Optional[bool]): Whether to generate sequentially.
            **model_kwargs: Additional keyword arguments.

        Returns:
            Any: Result of contrastive search, typically sequences or generated outputs.
        """
        ...
    # 发出警告日志，提醒直接调用该方法已经被废弃，将在 v4.41 版本中移除，建议使用 `generate` 方法或自定义生成循环代替。
    def greedy_search(self, *args, **kwargs):
        logger.warning_once(
            "Calling `greedy_search` directly is deprecated and will be removed in v4.41. Use `generate` or a "
            "custom generation loop instead.",
        )
        # 调用 `_greedy_search` 方法，并将所有传入的位置参数和关键字参数传递给它
        return self._greedy_search(*args, **kwargs)

    # 发出警告日志，提醒直接调用该方法已经被废弃，将在 v4.41 版本中移除，建议使用 `generate` 方法或自定义生成循环代替。
    def _greedy_search(
        self,
        input_ids: torch.LongTensor,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        output_logits: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        synced_gpus: bool = False,
        streamer: Optional["BaseStreamer"] = None,
        **model_kwargs,
    ):
        # 方法实现略去，用于执行贪婪搜索算法或相关任务
        pass

    # 发出警告日志，提醒直接调用该方法已经被废弃，将在 v4.41 版本中移除，建议使用 `generate` 方法或自定义生成循环代替。
    def sample(self, *args, **kwargs):
        logger.warning_once(
            "Calling `sample` directly is deprecated and will be removed in v4.41. Use `generate` or a "
            "custom generation loop instead.",
        )
        # 调用 `_sample` 方法，并将所有传入的位置参数和关键字参数传递给它
        return self._sample(*args, **kwargs)

    # 发出警告日志，提醒直接调用该方法已经被废弃，将在 v4.41 版本中移除，建议使用 `generate` 方法或自定义生成循环代替。
    def _sample(
        self,
        input_ids: torch.LongTensor,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        logits_warper: Optional[LogitsProcessorList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        output_logits: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        synced_gpus: bool = False,
        streamer: Optional["BaseStreamer"] = None,
        **model_kwargs,
    ):
        # 方法实现略去，用于执行采样或相关生成任务
        pass
    def _temporary_reorder_cache(self, past_key_values, beam_idx):
        """
        Temporary function to handle the different types of cache reordering processes while we roll out `Cache`.

        TODO: standardize cache formats and make all models compatible with `Cache`. It would remove the need
        for this function, with `Cache.reorder_cache` being the sole remaining code path
        """
        # 获取当前类名的小写形式
        model_class = self.__class__.__name__.lower()
        
        # 异常情况1：处理使用传统缓存格式的模型的代码路径
        if isinstance(past_key_values, (tuple, list)):
            past_key_values = self._reorder_cache(past_key_values, beam_idx)
        
        # 异常情况2：处理具有不同缓存格式的模型。这些模型目前仅限于 `DynamicCache`，直到它们的缓存格式标准化为止。
        elif "bloom" in model_class or "gptbigcode" in model_class:
            if not isinstance(past_key_values, DynamicCache):
                raise ValueError(
                    f"Using an unsupported cache format with {model_class}. Currently, it only supports the "
                    "legacy tuple format or `DynamicCache`"
                )
            past_key_values = self._reorder_cache(past_key_values, beam_idx)
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)
        
        # 标准代码路径：使用 `Cache.reorder_cache`
        else:
            past_key_values.reorder_cache(beam_idx)
        
        return past_key_values

    def beam_search(self, *args, **kwargs):
        logger.warning_once(
            "Calling `beam_search` directly is deprecated and will be removed in v4.41. Use `generate` or a "
            "custom generation loop instead.",
        )
        return self._beam_search(*args, **kwargs)

    def _beam_search(
        self,
        input_ids: torch.LongTensor,
        beam_scorer: BeamScorer,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        output_logits: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        synced_gpus: bool = False,
        sequential: Optional[bool] = None,
        **model_kwargs,
    ):
        """
        Perform beam search to generate sequences based on input_ids and beam_scorer.
        """
        logger.warning_once(
            "Calling `beam_search` directly is deprecated and will be removed in v4.41. Use `generate` or a "
            "custom generation loop instead.",
        )
        return self._beam_search(*args, **kwargs)

    def beam_sample(self, *args, **kwargs):
        logger.warning_once(
            "Calling `beam_sample` directly is deprecated and will be removed in v4.41. Use `generate` or a "
            "custom generation loop instead.",
        )
        return self._beam_sample(*args, **kwargs)
    # 定义一个私有方法 `_beam_sample`，用于执行束搜索采样
    def _beam_sample(
        self,
        input_ids: torch.LongTensor,
        beam_scorer: BeamScorer,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        logits_warper: Optional[LogitsProcessorList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        output_logits: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        synced_gpus: bool = False,
        **model_kwargs,
    ):
        # 具体功能的注释可以在方法内部详细描述
        pass

    # 警告用户 `group_beam_search` 方法即将在 v4.41 版本中移除，建议使用 `generate` 方法或自定义生成循环
    def group_beam_search(self, *args, **kwargs):
        logger.warning_once(
            "Calling `group_beam_search` directly is deprecated and will be removed in v4.41. Use `generate` or a "
            "custom generation loop instead.",
        )
        # 调用 `_group_beam_search` 方法来执行实际的束搜索操作
        return self._group_beam_search(*args, **kwargs)

    # 定义一个私有方法 `_group_beam_search`，用于执行束搜索
    def _group_beam_search(
        self,
        input_ids: torch.LongTensor,
        beam_scorer: BeamScorer,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        output_logits: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        synced_gpus: bool = False,
        **model_kwargs,
    ):
        # 具体功能的注释可以在方法内部详细描述
        pass

    # 警告用户 `constrained_beam_search` 方法即将在 v4.41 版本中移除，建议使用 `generate` 方法或自定义生成循环
    def constrained_beam_search(self, *args, **kwargs):
        logger.warning_once(
            "Calling `constrained_beam_search` directly is deprecated and will be removed in v4.41. Use `generate` or a "
            "custom generation loop instead.",
        )
        # 调用 `_constrained_beam_search` 方法来执行实际的约束束搜索操作
        return self._constrained_beam_search(*args, **kwargs)

    # 定义一个私有方法 `_constrained_beam_search`，用于执行约束束搜索
    def _constrained_beam_search(
        self,
        input_ids: torch.LongTensor,
        constrained_beam_scorer: ConstrainedBeamSearchScorer,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        output_logits: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        synced_gpus: Optional[bool] = None,
        **model_kwargs,
    ):
        # 具体功能的注释可以在方法内部详细描述
        pass
    # 发出警告日志，提醒直接调用 `_assisted_decoding` 方法已不推荐，在 v4.41 版本中将被移除。建议使用 `generate` 方法或自定义生成循环。
    logger.warning_once(
        "Calling `_assisted_decoding` directly is deprecated and will be removed in v4.41. Use `generate` or a "
        "custom generation loop instead.",
    )
    # 调用 `_assisted_decoding` 方法，将所有传入的位置参数和关键字参数传递给它，并返回其结果。
    return self._assisted_decoding(*args, **kwargs)
def _speculative_sampling(
    candidate_input_ids,
    candidate_logits,
    candidate_length,
    new_logits,
    last_assistant_token_is_eos,
    max_matches,
):
    """
    Applies sampling as in the speculative decoding paper (https://arxiv.org/pdf/2211.17192.pdf, algorithm 1). Returns
    the selected tokens, as well as the number of candidate matches.

    NOTE: Unless otherwise stated, the variable names match those in the paper.
    """
    # Selects the last `candidate_length` tokens from `candidate_input_ids`
    new_candidate_input_ids = candidate_input_ids[:, -candidate_length:]

    # Converts logits to probabilities and extracts assistant (q_i) and model (p_i) probabilities for selected tokens
    q = candidate_logits.softmax(dim=-1)
    q_i = q[:, torch.arange(candidate_length), new_candidate_input_ids].squeeze(0, 1)
    p = new_logits.softmax(dim=-1)
    p_i = p[:, torch.arange(candidate_length), new_candidate_input_ids].squeeze(0, 1)
    probability_ratio = p_i / q_i

    # Determines which tokens to accept based on probability ratios
    r_i = torch.rand_like(probability_ratio)
    is_accepted = r_i <= probability_ratio

    # Computes the number of accepted tokens (`n_matches` in algorithm 1)
    n_matches = ((~is_accepted).cumsum(dim=-1) < 1).sum()

    # Ensures the generated sequence does not exceed `max_matches` or end with an EOS token
    if last_assistant_token_is_eos and n_matches == candidate_length:
        # Adjusts `n_matches` if the sequence ends with an EOS token
        n_matches -= 1
        valid_tokens = new_candidate_input_ids[:, : n_matches + 1]
    else:
        n_matches = min(n_matches, max_matches)

        # Selects the next token considering rejection and adjusts probabilities if needed
        gamma = min(candidate_logits.shape[1], max_matches)
        p_n_plus_1 = p[:, n_matches, :]
        if n_matches < gamma:
            q_n_plus_1 = q[:, n_matches, :]
            p_prime = torch.clamp((p_n_plus_1 - q_n_plus_1), min=0)
            p_prime.div_(p_prime.sum())
        else:
            p_prime = p_n_plus_1
        t = torch.multinomial(p_prime, num_samples=1).squeeze(1)[None, :]

        # Constructs the final sequence of valid tokens
        if n_matches > 0:
            valid_tokens = torch.cat((new_candidate_input_ids[:, :n_matches], t), dim=-1)
        else:
            valid_tokens = t

    return valid_tokens, n_matches
    # 给定多个生成的标记的解码器注意力或隐藏状态，将其拆分成一个元组，其中每个成员对应于单个生成的标记。
    """
    # 兼容性调整：在我们的生成函数中，第一次迭代包含了关于提示的注意力/隐藏状态。
    if len(outputs) == 0:
        # 初始化一个空的元组
        new_tuple = ()
        # 遍历新输出的每一层
        for layer in new_outputs:
            # 如果是解码器的注意力，使用当前长度和最后一维的大小；否则使用整个层的大小
            last_dim_size = cur_len if is_decoder_attention else layer.shape[-1]
            # 将当前层的片段添加到新元组中
            new_tuple += (layer[..., :cur_len, :last_dim_size],)
        # 将新元组作为一个元素添加到输出元组中
        outputs += (new_tuple,)
        # 更新当前长度变量，因为第一次迭代包含了提示 + 1个生成的标记
        cur_len += 1
        # 更新添加的长度变量
        added_len -= cur_len
    
    # 对于每个额外添加的长度
    for i in range(added_len):
        # 初始化一个空的元组
        new_tuple = ()
        # 遍历新输出的每一层
        for layer in new_outputs:
            # 如果是解码器的注意力，使用当前长度加上i和最后一维的大小；否则使用整个层的大小
            last_dim_size = cur_len + i if is_decoder_attention else layer.shape[-1]
            # 将当前层的片段添加到新元组中
            new_tuple += (layer[..., i : i + 1, :last_dim_size],)
        # 将新元组作为一个元素添加到输出元组中
        outputs += (new_tuple,)
    # 返回输出元组
    return outputs
# 根据上下文隐藏状态的每个向量的L2范数归一化，使其长度为1，以便计算余弦相似度
norm_context_hidden = context_hidden / context_hidden.norm(dim=2, keepdim=True)

# 根据下一个隐藏状态的每个向量的L2范数归一化，使其长度为1，以便计算余弦相似度
norm_next_hidden = next_hidden / next_hidden.norm(dim=2, keepdim=True)

# 计算上下文隐藏状态与下一个隐藏状态之间的余弦相似度矩阵，将维度调整为[B*K, S]
cosine_matrix = torch.matmul(norm_context_hidden, norm_next_hidden.transpose(1, 2)).squeeze(-1)

# 在余弦相似度矩阵的最后一个维度上取最大值，得到每个样本的最大相似度，形状为[B*K]
degeneration_penalty, _ = torch.max(cosine_matrix, dim=-1)

# 将下一个顶部K个候选项的概率视图调整为一维数组，形状为[B*K]
next_top_k_probs = next_top_k_probs.view(-1)

# 计算对比分数，根据论文中的对比框架计算每个候选项的分数
contrastive_score = (1.0 - alpha) * next_top_k_probs - alpha * degeneration_penalty

# 将对比分数按照beam_width分割，形状调整为[B, K]的张量
contrastive_score = torch.stack(torch.split(contrastive_score, beam_width))

# 在每行中选择最高分数对应的索引，形状为[B]
_, selected_idx = contrastive_score.max(dim=-1)

# 返回每个批次中最佳候选项的索引
return selected_idx



# 处理数据分割的函数，根据数据类型分别处理不同情况的数据分割
def _split(data, full_batch_size: int, split_size: int = None):
    if data is None:
        # 如果数据为None，则返回与分割大小对应的None列表
        return [None] * (full_batch_size // split_size)
    if isinstance(data, torch.Tensor):
        # 如果数据为Tensor，则按照分割大小分割Tensor，返回Tensor列表
        return [data[i : i + split_size] for i in range(0, full_batch_size, split_size)]
    elif isinstance(data, tuple):
        # 如果数据为元组，根据元组中元素的类型进行不同的分割处理
        if isinstance(data[0], tuple):
            # 如果元组中的元素也是元组，则按照分割大小分割每个元组中的Tensor，返回元组列表的元组列表
            return [
                tuple(tuple(tensor[i : i + split_size] for tensor in inner_tuple) for inner_tuple in data)
                for i in range(0, full_batch_size, split_size)
            ]
        else:
            # 如果元组中的元素不是元组，则按照分割大小分割每个Tensor，返回元组列表
            return [
                tuple(sub_tensor[i : i + split_size] for sub_tensor in data)
                for i in range(0, full_batch_size, split_size)
            ]
    else:
        # 如果数据类型不符合预期，则引发值错误异常
        raise ValueError(f"Unexpected attribute type: {type(data)}")



# 将模型输入（可能是ModelOutput或Dict类型）按照指定的分割大小拆分成相同类型的对象列表
def _split_model_inputs(
    model_input: Union[ModelOutput, Dict], split_size: int, full_batch_size: int
) -> List[Union[ModelOutput, Dict]]:
    """
    Split a ModelOutput object (or its subclasses) or Dict into a list of same-class objects based on a specified split
    size. The input object is dict when it was prepared for forward pass and ModelOutput when it was returned from
    previous forward pass.
    """
    # 如果 model_input 为 None，则返回一个 Nones 列表
    # 在 Whisper 中，encoder_outputs 为 None 时会发生这种情况
    if model_input is None:
        return [model_input] * (full_batch_size // split_size)
    # 从对象中推断出类
    model_output_cls = type(model_input)
    if (full_batch_size % split_size) != 0:
        # 如果 full_batch_size 不能被 split_size 整除，则引发 ValueError
        raise ValueError("`full_batch_size` must be divisible by `split_size`")

    if split_size > full_batch_size:
        # 如果 split_size 大于 full_batch_size，则引发 ValueError
        raise ValueError("`split_size` must be smaller or equal to `full_batch_size`")

    # 用于拆分张量或张量的元组的辅助函数

    # 查找所有数据类字段（例如，last_hidden_state，pooler_output 等），并对它们进行拆分
    keys = (
        model_input.__dataclass_fields__.keys() if hasattr(model_input, "__dataclass_fields__") else model_input.keys()
    )
    # 仅保留在 model_input 中的键
    keys = [k for k in keys if k in model_input]
    # 在这里，我们可以有四种类型的值：张量、张量的元组和布尔值，以及 encoder_outputs，后者是一个 ModelOutput 对象。
    # 布尔值不应该被拆分，而应该为每个拆分复制
    bool_keys = [k for k in keys if isinstance(model_input[k], bool) or k == "cache_position"]
    keys_to_ignore = ["cache_position", "encoder_outputs"]
    non_bool_keys = [k for k in keys if not isinstance(model_input[k], bool) and k not in keys_to_ignore]

    # 拆分张量和张量的元组
    data_split_list = [
        {k: _split(model_input[k], full_batch_size, split_size)[i] for k in non_bool_keys}
        for i in range(full_batch_size // split_size)
    ]
    # 布尔值是相同的，每个拆分中都会复制
    bool_data = {k: model_input[k] for k in bool_keys}
    # encoder_outputs 是一个 ModelOutput 对象，应该单独拆分
    if "encoder_outputs" in model_input:
        encoder_outputs_split = _split_model_inputs(model_input["encoder_outputs"], split_size, full_batch_size)
        data_split_list = [
            {**data_split, "encoder_outputs": encoder_outputs_split[i]} for i, data_split in enumerate(data_split_list)
        ]

    # 将列表中的每个字典转换为推断类的对象
    split_model_inputs: List[Union[ModelOutput, Dict]] = [
        model_output_cls(**data_split, **bool_data) for data_split in data_split_list
    ]

    return split_model_inputs
# 将给定的 ModelOutput 对象列表沿着 batch_size 维度堆叠起来。该函数推断出列表中的具体 ModelOutput 子类。
def stack_model_outputs(model_outputs: List[ModelOutput]) -> ModelOutput:
    """
    Stack a list of ModelOutput objects (or its subclasses) along the batch_size dimension. The function infers the
    specific ModelOutput subclass from the list provided.
    """
    # 如果输入的列表为空，则抛出数值错误
    if not model_outputs:
        raise ValueError("Input list is empty.")

    # 推断出列表中第一个对象的类
    model_output_cls = type(model_outputs[0])

    # 确保所有对象都是同一类型
    if not all(isinstance(obj, model_output_cls) for obj in model_outputs):
        raise ValueError("All elements in the list should be of the same type.")

    # 辅助函数，用于连接张量或张量元组
    def _concat(data):
        """
        Reverse of `_split` function above.
        """
        # 如果数据中任意元素为 None，则返回 None
        if any(data is None for data in data):
            return None
        # 如果第一个元素是 torch.Tensor
        if isinstance(data[0], torch.Tensor):
            # 沿着 dim=0 连接所有张量
            return torch.cat(data, dim=0)
        # 如果第一个元素是元组
        elif isinstance(data[0], tuple):
            # 如果元组的元素也是元组（例如我们之前的示例中的 past_key_values）
            if isinstance(data[0][0], tuple):
                # 对每个元组的每个元素，沿着 dim=0 连接所有张量
                return tuple(
                    tuple(torch.cat([attr[i][j] for attr in data], dim=0) for j in range(len(data[0][0])))
                    for i in range(len(data[0]))
                )
            else:
                # 否则，对元组中的每个元素，沿着 dim=0 连接所有张量
                return tuple(torch.cat([attr[i] for attr in data], dim=0) for i in range(len(data[0])))
        # 如果第一个元素是整数或浮点数，返回一个张量
        elif isinstance(data[0], (int, float)):
            return torch.tensor(data)
        else:
            # 抛出数值错误，显示意外的属性类型
            raise ValueError(f"Unexpected attribute type: {type(data[0])}")

    # 使用字典推导式，从所有对象中收集属性并连接它们
    concatenated_data = {
        # 对于每个属性 k，在所有模型输出对象中，获取属性 k 的值并连接它们
        k: _concat([getattr(model_output, k) for model_output in model_outputs])
        for k in model_output_cls.__dataclass_fields__.keys()
    }

    # 返回一个新的推断类对象，其中包含连接后的属性
    return model_output_cls(**concatenated_data)
```