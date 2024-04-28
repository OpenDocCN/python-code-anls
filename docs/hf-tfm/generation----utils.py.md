# `.\transformers\generation\utils.py`

```py
# 设置文件编码为 UTF-8
# 版权声明，包括作者信息和许可证信息
# 导入所需的库和模块
import copy  # 导入 copy 模块，用于深拷贝对象
import inspect  # 导入 inspect 模块，用于获取对象信息
import warnings  # 导入 warnings 模块，用于警告处理
from dataclasses import dataclass  # 导入 dataclass 装饰器，用于创建数据类
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union  # 导入类型提示相关的模块

import torch  # 导入 PyTorch 库
import torch.distributed as dist  # 导入 PyTorch 分布式库
from torch import nn  # 导入 PyTorch 神经网络模块

from ..cache_utils import Cache, DynamicCache  # 导入缓存相关的模块
from ..integrations.deepspeed import is_deepspeed_zero3_enabled  # 导入 DeepSpeed 零3模块的集成函数
from ..modeling_outputs import CausalLMOutputWithPast, Seq2SeqLMOutput  # 导入模型输出相关的模块
from ..models.auto import (  # 导入自动模型相关的模块
    MODEL_FOR_CAUSAL_IMAGE_MODELING_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING,
    MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING,
    MODEL_FOR_VISION_2_SEQ_MAPPING,
)
from ..utils import ExplicitEnum, ModelOutput, is_accelerate_available, logging  # 导入工具函数和日志模块
from .beam_constraints import DisjunctiveConstraint, PhrasalConstraint  # 导入束搜索相关的约束模块
from .beam_search import BeamScorer, BeamSearchScorer, ConstrainedBeamSearchScorer  # 导入束搜索相关的模块
from .candidate_generator import (  # 导入候选生成器相关的模块
    AssistedCandidateGenerator,
    CandidateGenerator,
    PromptLookupCandidateGenerator,
    _crop_past_key_values,
    _prepare_attention_mask,
    _prepare_token_type_ids,
)
from .configuration_utils import GenerationConfig  # 导入生成配置相关的模块
from .logits_process import (  # 导入logits处理相关的模块
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
from .stopping_criteria import (  # 导入停止标准相关的模块
    MaxLengthCriteria,
    MaxTimeCriteria,
    StoppingCriteria,
    StoppingCriteriaList,
    validate_stopping_criteria,
)

# 如果是类型检查，则导入预训练模型模块
if TYPE_CHECKING:
    from ..modeling_utils import PreTrainedModel
    # 从当前目录下的streamers模块中导入BaseStreamer类
    from .streamers import BaseStreamer
# 导入日志模块中的 getLogger 函数，用于获取一个指定名称的 logger 对象
logger = logging.get_logger(__name__)

# 检查是否支持加速
if is_accelerate_available():
    # 如果支持加速，从 accelerate.hooks 模块导入 AlignDevicesHook 和 add_hook_to_module 函数
    from accelerate.hooks import AlignDevicesHook, add_hook_to_module


# 定义一个数据类 GenerateDecoderOnlyOutput，用于表示仅解码器生成模型的输出
@dataclass
class GenerateDecoderOnlyOutput(ModelOutput):
    """
    Outputs of decoder-only generation models, when using non-beam methods.

    Args:
        sequences (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            The generated sequences. The second dimension (sequence_length) is either equal to `max_length` or shorter
            if all batches finished early due to the `eos_token_id`.
        scores (`tuple(torch.FloatTensor)` *optional*, returned when `output_scores=True` is passed or when `config.output_scores=True`):
            Processed prediction scores of the language modeling head (scores for each vocabulary token before SoftMax)
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

    """
    # 生成的序列，形状为 `(batch_size, sequence_length)` 的 `torch.LongTensor`
    sequences: torch.LongTensor = None
    # 语言建模头部的处理过的预测分数（SoftMax 前每个词汇标记的分数）在每个生成步骤上
    # Tuple，包含最多 `max_new_tokens` 个元素（每个生成的词汇标记一个元素），每个张量形状为 `(batch_size, config.vocab_size)`
    scores: Optional[Tuple[torch.FloatTensor]] = None
    # 生成器注意力机制的注意力权重
    # Tuple，每个生成的词汇标记一个元素，每个元素是一个元组（解码器每层一个元素），其中包含形状为 `(batch_size, num_heads, generated_length, sequence_length)` 的 `torch.FloatTensor`
    attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    # 解码器隐藏状态
    # Tuple，每个生成的词汇标记一个元素，每个元素是一个元组（解码器每层一个元素），其中包含形状为 `(batch_size, generated_length, hidden_size)` 的 `torch.FloatTensor`
    hidden_states: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    # 过去的键值对
    # Tuple，每个解码器层一个元素，其中每个元素是一个元组（两个张量，键张量和值张量）
    # 第一个元组长度为 `config.n_layers`，每个元组有 2 个形状为 `(batch_size, num_heads, sequence_length, embed_size_per_head)` 的张量
    # 如果 `config.is_encoder_decoder=True`，还有两个形状为 `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)` 的张量
    past_key_values: Optional[Tuple[Tuple[Tuple[torch.FloatTensor]]]] = None


# 定义一个数据类 GenerateEncoderDecoderOutput，用于表示编码器-解码器生成模型的输出
@dataclass
class GenerateEncoderDecoderOutput(ModelOutput):
    """
    Outputs of encoder-decider generation models, when using non-beam methods.

    """
    # 定义一个名为sequences的torch.LongTensor类型变量，初始值为None
    sequences: torch.LongTensor = None
    # 定义一个名为scores的可选类型元组，元组中包含一个torch.FloatTensor类型变量
    scores: Optional[Tuple[torch.FloatTensor]] = None
    # 定义一个名为encoder_attentions的可选类型元组，元组中包含一个torch.FloatTensor类型变量
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    # 定义一个名为encoder_hidden_states的可选类型元组，元组中包含一个torch.FloatTensor类型变量
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # 定义一个名为decoder_attentions的可选类型元组，元组中包含一个元组，元组中包含一个torch.FloatTensor类型变量
    decoder_attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    # 定义一个名为cross_attentions的可选类型元组，元组中包含一个元组，元组中包含一个torch.FloatTensor类型变量
    cross_attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    # 定义一个名为decoder_hidden_states的可选类型元组，元组中包含一个元组，元组中包含一个torch.FloatTensor类型变量
    decoder_hidden_states: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    # 定义一个名为past_key_values的可选类型元组，元组中包含一个元组，元组中包含一个元组，元组中包含一个torch.FloatTensor类型变量
    past_key_values: Optional[Tuple[Tuple[Tuple[torch.FloatTensor]]]] = None
# 导入 dataclass 模块，用于创建数据类
from dataclasses import dataclass

# 定义 GenerateBeamDecoderOnlyOutput 类，继承自 ModelOutput 类
@dataclass
class GenerateBeamDecoderOnlyOutput(ModelOutput):
    """
    Outputs of decoder-only generation models, when using beam methods.
    """
        Args:
            sequences (`torch.LongTensor` of shape `(batch_size*num_return_sequences, sequence_length)`):
                生成的序列。第二维（sequence_length）要么等于 `max_length`，要么比它短，如果所有批次由于 `eos_token_id` 提前结束。
            sequences_scores (`torch.FloatTensor` of shape `(batch_size*num_return_sequences)`, *optional*, returned when `output_scores=True` is passed or when `config.output_scores=True`):
                生成的 `sequences` 的最终 beam 得分。
            scores (`tuple(torch.FloatTensor)` *optional*, returned when `output_scores=True` is passed or when `config.output_scores=True`):
                每个生成步骤每个词汇标记的 beam 过渡得分。Beam 过渡得分由给定上一 beam 中先前生成的标记的 log softmax 条件下的标记的 log 概率组成。
                Tuple 的每个元素为 `torch.FloatTensor`，最多有 `max_new_tokens` 个元素（每个生成标记一个元素），每个张量的形状为 `(batch_size*num_beams*num_return_sequences, config.vocab_size)`。
            beam_indices (`torch.LongTensor`, *optional*, returned when `output_scores=True` is passed or when `config.output_scores=True`):
                每个生成步骤生成的标记 id 的 beam 索引。形状为 `(batch_size*num_return_sequences, sequence_length)` 的 `torch.LongTensor`。
            attentions (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_attentions=True` is passed or `config.output_attentions=True`):
                元组（每个生成标记一个元素），其中每个元素为元组（解码器的每个层一个元素）的元组（注意力头的数量，生成长度，序列长度）的 `torch.FloatTensor`。
                其形状为 `(batch_size*num_beams, num_heads, generated_length, sequence_length)`。
            hidden_states (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
                元组（每个生成标记一个元素），其中每个元素为元组（解码器的每个层一个元素）的 `torch.FloatTensor`。
                其形状为 `(batch_size*num_beams*num_return_sequences, generated_length, hidden_size)`。
            past_key_values (`tuple(tuple(torch.FloatTensor)))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
                注意：一些模型具有不同的 `past_key_values` 格式，请参阅模型文档确认。
                通常为元组（解码器的每一层一个元素），其中每个元素为元组（键张量和值张量，两个元素）。第一个元组的长度为 `config.n_layers`，
                每个元组包含 2 个形状为 `(batch_size, num_heads, sequence_length, embed_size_per_head)` 的张量，
                如果 `config.is_encoder_decoder=True`，则还有 2 个额外的形状为 `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)` 的张量。
```  
    # 定义一个 torch.LongTensor 类型的变量 sequences，初始值为 None
    sequences: torch.LongTensor = None
    
    # 定义一个 torch.FloatTensor 类型的可选变量 sequences_scores，初始值为 None
    sequences_scores: Optional[torch.FloatTensor] = None
    
    # 定义一个包含 torch.FloatTensor 类型元组的可选变量 scores，初始值为 None
    scores: Optional[Tuple[torch.FloatTensor]] = None
    
    # 定义一个 torch.LongTensor 类型的可选变量 beam_indices，初始值为 None
    beam_indices: Optional[torch.LongTensor] = None
    
    # 定义一个包含 torch.FloatTensor 类型元组的可选变量 attentions，初始值为 None
    attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    
    # 定义一个包含 torch.FloatTensor 类型元组的可选变量 hidden_states，初始值为 None
    hidden_states: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    
    # 定义一个包含 torch.FloatTensor 类型元组的三重嵌套可选变量 past_key_values，初始值为 None
    past_key_values: Optional[Tuple[Tuple[Tuple[torch.FloatTensor]]]] = None
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
from transformers.modeling_outputs import ModelOutput
from enum import Enum


@dataclass
class GenerateBeamEncoderDecoderOutput(ModelOutput):
    """
    Outputs of encoder-decoder generation models, when using beam methods.

    """

    sequences: torch.LongTensor = None  # 生成的序列
    sequences_scores: Optional[torch.FloatTensor] = None  # 生成序列的分数
    scores: Optional[Tuple[torch.FloatTensor]] = None  # 分数
    beam_indices: Optional[torch.LongTensor] = None  # beam 索引
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None  # 编码器注意力
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None  # 编码器隐藏状态
    decoder_attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None  # 解码器注意力
    cross_attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None  # 交叉注意力
    decoder_hidden_states: Optional[Tuple[Tuple[torch.FloatTensor]]] = None  # 解码器隐藏状态
    past_key_values: Optional[Tuple[Tuple[Tuple[torch.FloatTensor]]]] = None  # 过去的键值


# Equivalent classes (kept for retrocompatibility purposes)
GreedySearchDecoderOnlyOutput = GenerateDecoderOnlyOutput  # 为了向后兼容而保留的等效类
ContrastiveSearchDecoderOnlyOutput = GenerateDecoderOnlyOutput  # 为了向后兼容而保留的等效类
SampleDecoderOnlyOutput = GenerateDecoderOnlyOutput  # 为了向后兼容而保留的等效类

ContrastiveSearchEncoderDecoderOutput = GenerateEncoderDecoderOutput  # 为了向后兼容而保留的等效类
GreedySearchEncoderDecoderOutput = GenerateEncoderDecoderOutput  # 为了向后兼容而保留的等效类
SampleEncoderDecoderOutput = GenerateEncoderDecoderOutput  # 为了向后兼容而保留的等效类

BeamSearchDecoderOnlyOutput = GenerateBeamDecoderOnlyOutput  # 为了向后兼容而保留的等效类
BeamSampleDecoderOnlyOutput = GenerateBeamDecoderOnlyOutput  # 为了向后兼容而保留的等效类

BeamSearchEncoderDecoderOutput = GenerateBeamEncoderDecoderOutput  # 为了向后兼容而保留的等效类
BeamSampleEncoderDecoderOutput = GenerateBeamEncoderDecoderOutput  # 为了向后兼容而保留的等效类


# Typing shortcuts
GenerateNonBeamOutput = Union[GenerateDecoderOnlyOutput, GenerateEncoderDecoderOutput]  # 生成非 beam 输出的类型快捷方式
GenerateBeamOutput = Union[GenerateBeamDecoderOnlyOutput, GenerateBeamEncoderDecoderOutput]  # 生成 beam 输出的类型快捷方式
GenerateOutput = Union[GenerateNonBeamOutput, GenerateBeamOutput]  # 生成输出的类型快捷方式


class GenerationMode(ExplicitEnum):
    """
    Possible generation modes, downstream of the [`~generation.GenerationMixin.generate`] method.
    """

    # Non-beam methods
    CONTRASTIVE_SEARCH = "contrastive_search"  # 对比搜索
    GREEDY_SEARCH = "greedy_search"  # 贪婪搜索
    SAMPLE = "sample"  # 随机采样
    ASSISTED_GENERATION = "assisted_generation"  # 辅助生成
    # Beam methods
    BEAM_SEARCH = "beam_search"  # Beam 搜索
    BEAM_SAMPLE = "beam_sample"  # Beam 随机采样
    CONSTRAINED_BEAM_SEARCH = "constrained_beam_search"  # 限制 Beam 搜索
    GROUP_BEAM_SEARCH = "group_beam_search"  # 分组 Beam 搜索


class GenerationMixin:
    """
    A class containing all functions for auto-regressive text generation, to be used as a mixin in [`PreTrainedModel`].

    """
    # 该类公开了一些方法，可以用于不同的生成策略：
    # - 如果 `num_beams=1` 且 `do_sample=False`，可以通过调用 `~generation.GenerationMixin.greedy_search` 进行贪婪解码
    # - 如果 `penalty_alpha>0` 且 `top_k>1`，可以通过调用 `~generation.GenerationMixin.contrastive_search` 进行对比搜索
    # - 如果 `num_beams=1` 且 `do_sample=True`，可以通过调用 `~generation.GenerationMixin.sample` 进行多项式采样
    # - 如果 `num_beams>1` 且 `do_sample=False`，可以通过调用 `~generation.GenerationMixin.beam_search` 进行束搜索解码
    # - 如果 `num_beams>1` 且 `do_sample=True`，可以通过调用 `~generation.GenerationMixin.beam_sample` 进行束搜索多项式采样
    # - 如果 `num_beams>1` 且 `num_beam_groups>1`，可以通过调用 `~generation.GenerationMixin.group_beam_search` 进行多束搜索解码
    # - 如果 `constraints!=None` 或 `force_words_ids!=None`，可以通过调用 `~generation.GenerationMixin.constrained_beam_search` 进行约束束搜索解码

    # 不需要直接调用上述方法中的任何一个。而是将自定义参数值传递给 'generate' 方法。要了解更多关于解码策略的信息，请参考 [文本生成策略指南](../generation_strategies)。

    # 抛出未实现错误，提示模型类需要定义一个 `prepare_inputs_for_generation` 方法才能使用 `.generate()`。
    def prepare_inputs_for_generation(self, *args, **kwargs):
        raise NotImplementedError(
            "A model class needs to define a `prepare_inputs_for_generation` method in order to use `.generate()`."
        )

    # 准备模型输入的方法，接受输入参数和模型参数
    def _prepare_model_inputs(
        self,
        inputs: Optional[torch.Tensor] = None,
        bos_token_id: Optional[int] = None,
        model_kwargs: Optional[Dict[str, torch.Tensor]] = None,
    
    # 可能初始化生成输入的方法，接受输入参数和模型参数
    def _maybe_initialize_input_ids_for_generation(
        self,
        inputs: Optional[torch.Tensor] = None,
        bos_token_id: Optional[int] = None,
        model_kwargs: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.LongTensor:
        """初始化用于生成的输入 id（如果需要）。"""
        # 如果提供了输入，则直接返回输入
        if inputs is not None:
            return inputs

        # 获取编码器输出
        encoder_outputs = model_kwargs.get("encoder_outputs")
        # 如果模型是编码器-解码器模型且存在编码器输出，则创建一个包含-100值的虚拟输入 id，以确保它们不会被用于编码
        if self.config.is_encoder_decoder and encoder_outputs is not None:
            shape = encoder_outputs.last_hidden_state.size()[:-1]
            return torch.ones(shape, dtype=torch.long, device=self.device) * -100

        # 如果未提供输入 id，且未提供 `bos_token_id`，则抛出错误
        if bos_token_id is None:
            raise ValueError("`bos_token_id` has to be defined when no `input_ids` are provided.")

        # 如果 `model_kwargs` 中包含张量，则从中推断出批次大小，这对于使用软提示或在仅解码器语言模型上构建的多模态实现很有帮助
        batch_size = 1
        for value in model_kwargs.values():
            if isinstance(value, torch.Tensor):
                batch_size = value.shape[0]
                break
        # 返回具有指定 bos_token_id 的张量
        return torch.ones((batch_size, 1), dtype=torch.long, device=self.device) * bos_token_id

    def _prepare_attention_mask_for_generation(
        self,
        inputs: torch.Tensor,
        pad_token_id: Optional[int],
        eos_token_id: Optional[Union[int, List[int]]],
    ) -> torch.LongTensor:
        # 检查输入是否为 input_ids，并且是否存在填充 -> 只有在这种情况下才定义 attention_mask
        is_input_ids = len(inputs.shape) == 2 and inputs.dtype in [torch.int, torch.long]
        is_pad_token_in_inputs = (pad_token_id is not None) and (pad_token_id in inputs)
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        is_pad_token_not_equal_to_eos_token_id = (eos_token_id is None) or (pad_token_id not in eos_token_id)

        # 如果输入是 input_ids 并且存在填充且填充标记不等于 eos_token_id，则返回 attention_mask
        if is_input_ids and is_pad_token_in_inputs and is_pad_token_not_equal_to_eos_token_id:
            return inputs.ne(pad_token_id).long()
        else:
            # 否则，返回全 1 的张量
            return torch.ones(inputs.shape[:2], dtype=torch.long, device=inputs.device)

    def _prepare_encoder_decoder_kwargs_for_generation(
        self, inputs_tensor: torch.Tensor, model_kwargs, model_input_name: Optional[str] = None
        # 定义函数的输入和输出类型注解
        ) -> Dict[str, Any]:
        # 1. 获取编码器
        encoder = self.get_encoder()
        
        # 为了与加速大型模型推理兼容：我们需要编码器在相同设备上输出与输入相同的内容
        if hasattr(self, "hf_device_map"):
            if hasattr(encoder, "_hf_hook"):
                encoder._hf_hook.io_same_device = True
            else:
                add_hook_to_module(encoder, AlignDevicesHook(io_same_device=True))

        # 2. 从模型参数准备编码器参数和编码器关键字参数
        irrelevant_prefix = ["decoder_", "cross_attn", "use_cache"]
        encoder_kwargs = {
            argument: value
            for argument, value in model_kwargs.items()
            if not any(argument.startswith(p) for p in irrelevant_prefix)
        }
        encoder_signature = set(inspect.signature(encoder.forward).parameters)
        encoder_accepts_wildcard = "kwargs" in encoder_signature or "model_kwargs" in encoder_signature
        if not encoder_accepts_wildcard:
            encoder_kwargs = {
                argument: value for argument, value in encoder_kwargs.items() if argument in encoder_signature
            }

        # 3. 确保编码器返回 `ModelOutput`
        model_input_name = model_input_name if model_input_name is not None else self.main_input_name
        encoder_kwargs["return_dict"] = True
        encoder_kwargs[model_input_name] = inputs_tensor
        model_kwargs["encoder_outputs"]: ModelOutput = encoder(**encoder_kwargs)

        # 返回模型参数
        return model_kwargs

    def _prepare_decoder_input_ids_for_generation(
        self,
        batch_size: int,
        model_input_name: str,
        model_kwargs: Dict[str, torch.Tensor],
        decoder_start_token_id: int = None,
        bos_token_id: int = None,
        device: torch.device = None,
    ) -> Tuple[torch.LongTensor, Dict[str, torch.Tensor]]:
        """为编码器-解码器模型准备`decoder_input_ids`"""
        # 1. 检查用户是否手动定义了`decoder_input_ids`。为了方便输入命名，如果编码器不将其用作主要输入，则还允许用户在`input_ids`下传递它。
        if model_kwargs is not None and "decoder_input_ids" in model_kwargs:
            decoder_input_ids = model_kwargs.pop("decoder_input_ids")
        elif "input_ids" in model_kwargs and model_input_name != "input_ids":
            decoder_input_ids = model_kwargs.pop("input_ids")
        else:
            decoder_input_ids = None

        # 2. 编码器-解码器模型期望`decoder_input_ids`以特殊标记开始。让我们确保这一点。
        decoder_start_token_id = self._get_decoder_start_token_id(decoder_start_token_id, bos_token_id)
        if device is None:
            device = self.device
        decoder_input_ids_start = torch.ones((batch_size, 1), dtype=torch.long, device=device) * decoder_start_token_id

        # 没有用户输入 -> 使用decoder_start_token_id作为decoder_input_ids
        if decoder_input_ids is None:
            decoder_input_ids = decoder_input_ids_start
        # 异常情况：Donut检查点具有特定于任务的解码器起始点，并且不需要BOS标记
        elif self.config.model_type == "vision-encoder-decoder" and "donut" in self.name_or_path.lower():
            pass
        elif self.config.model_type in ["whisper"]:
            pass
        # 用户输入但不以decoder_start_token_id开头 -> 在前面添加decoder_start_token_id（并调整decoder_attention_mask（如果提供））
        elif (decoder_input_ids[:, 0] != decoder_start_token_id).all().item():
            decoder_input_ids = torch.cat([decoder_input_ids_start, decoder_input_ids], dim=-1)
            if "decoder_attention_mask" in model_kwargs:
                decoder_attention_mask = model_kwargs["decoder_attention_mask"]
                decoder_attention_mask = torch.cat(
                    (torch.ones_like(decoder_attention_mask)[:, :1], decoder_attention_mask),
                    dim=-1,
                )
                model_kwargs["decoder_attention_mask"] = decoder_attention_mask

        return decoder_input_ids, model_kwargs
    def _get_decoder_start_token_id(self, decoder_start_token_id: int = None, bos_token_id: int = None) -> int:
        # 如果给定了decoder_start_token_id，则使用给定值，否则使用generation_config中的值
        decoder_start_token_id = (
            decoder_start_token_id
            if decoder_start_token_id is not None
            else self.generation_config.decoder_start_token_id
        )
        # 如果给定了bos_token_id，则使用给定值，否则使用generation_config中的值
        bos_token_id = bos_token_id if bos_token_id is not None else self.generation_config.bos_token_id

        # 如果decoder_start_token_id已定义，则返回其值
        if decoder_start_token_id is not None:
            return decoder_start_token_id
        # 如果bos_token_id已定义，则返回其值
        elif bos_token_id is not None:
            return bos_token_id
        # 否则引发值错误
        raise ValueError(
            "`decoder_start_token_id` or `bos_token_id` has to be defined for encoder-decoder generation."
        )

    @staticmethod
    def _expand_inputs_for_generation(
        expand_size: int = 1,
        is_encoder_decoder: bool = False,
        input_ids: Optional[torch.LongTensor] = None,
        **model_kwargs,
    ) -> Tuple[torch.LongTensor, Dict[str, Any]]:
        """Expands tensors from [batch_size, ...] to [batch_size * expand_size, ...]"""

        # 定义用于扩展字典的内部函数
        def _expand_dict_for_generation(dict_to_expand):
            for key in dict_to_expand:
                if dict_to_expand[key] is not None and isinstance(dict_to_expand[key], torch.Tensor):
                    # 使用repeat_interleave函数在指定维度上重复张量
                    dict_to_expand[key] = dict_to_expand[key].repeat_interleave(expand_size, dim=0)
            return dict_to_expand

        # 如果给定了input_ids，则在指定维度上重复张量
        if input_ids is not None:
            input_ids = input_ids.repeat_interleave(expand_size, dim=0)

        # 扩展model_kwargs中的张量
        model_kwargs = _expand_dict_for_generation(model_kwargs)

        # 如果是encoder-decoder模型，则扩展encoder_outputs中的张量
        if is_encoder_decoder:
            if model_kwargs.get("encoder_outputs") is None:
                raise ValueError("If `is_encoder_decoder` is True, make sure that `encoder_outputs` is defined.")
            model_kwargs["encoder_outputs"] = _expand_dict_for_generation(model_kwargs["encoder_outputs"])

        # 返回扩展后的input_ids和model_kwargs
        return input_ids, model_kwargs

    def _extract_past_from_model_output(self, outputs: ModelOutput, standardize_cache_format: bool = False):
        # 从模型输出中提取过去的键值对
        past_key_values = None
        if "past_key_values" in outputs:
            past_key_values = outputs.past_key_values
        elif "mems" in outputs:
            past_key_values = outputs.mems
        elif "past_buckets_states" in outputs:
            past_key_values = outputs.past_buckets_states

        # 当standardize_cache_format为True时，修复缓存格式
        if standardize_cache_format and hasattr(self, "_convert_to_standard_cache"):
            # 获取批量大小
            batch_size = outputs.logits.shape[0]
            # 将past_key_values转换为标准缓存格式
            past_key_values = self._convert_to_standard_cache(past_key_values, batch_size=batch_size)
        return past_key_values

    def _update_model_kwargs_for_generation(
        self,
        outputs: ModelOutput,
        model_kwargs: Dict[str, Any],
        is_encoder_decoder: bool = False,
        standardize_cache_format: bool = False,
    ) -> Dict[str, Any]:
        # 更新模型关键值的过去值
        model_kwargs["past_key_values"] = self._extract_past_from_model_output(
            outputs, standardize_cache_format=standardize_cache_format
        )
        # 如果输出有状态信息，则更新模型关键字典中的状态信息
        if getattr(outputs, "state", None) is not None:
            model_kwargs["state"] = outputs.state

        # 使用最后一个值更新 token_type_ids
        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = torch.cat([token_type_ids, token_type_ids[:, -1].unsqueeze(-1)], dim=-1)

        # 如果不是编码-解码模型
        if not is_encoder_decoder:
            # 更新注意力遮罩
            if "attention_mask" in model_kwargs:
                attention_mask = model_kwargs["attention_mask"]
                model_kwargs["attention_mask"] = torch.cat(
                    [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
                )
        else:
            # 更新解码器注意力遮罩
            if "decoder_attention_mask" in model_kwargs:
                decoder_attention_mask = model_kwargs["decoder_attention_mask"]
                model_kwargs["decoder_attention_mask"] = torch.cat(
                    [decoder_attention_mask, decoder_attention_mask.new_ones((decoder_attention_mask.shape[0], 1))],
                    dim=-1,
                )

        return model_kwargs

    # 重新排序缓存
    def _reorder_cache(self, past_key_values, beam_idx):
        raise NotImplementedError(
            f"Make sure that a `_reorder_cache` function is correctly implemented in {self.__class__.__module__} to"
            f" enable beam search for {self.__class__}"
        )

    # 获取候选生成器
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
        返回要在 `assisted_generation` 中使用的候选生成器
        """
        # 如果有提供提示查找的令牌数量，则使用 PromptLookupCandidateGenerator
        if generation_config.prompt_lookup_num_tokens is not None:
            candidate_generator = PromptLookupCandidateGenerator(
                num_output_tokens=generation_config.prompt_lookup_num_tokens,
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

    # 获取 logits 处理器
    def _get_logits_warper(
        self,
        generation_config: GenerationConfig,
    ) -> LogitsProcessorList:
        """
        This class returns a [`LogitsProcessorList`] list object that contains all relevant [`LogitsWarper`] instances
        used for multinomial sampling.
        """

        # instantiate warpers list
        # 实例化一个 warpers 列表对象
        warpers = LogitsProcessorList()

        # In beam methods, we need to keep at least one non-eos token to explore continuations that might have a
        # better score (i.e. keep len(list(generation_config.eos_token_id)) + 1)
        # 在 beam 方法中，我们需要至少保留一个非 eos 标记以探索可能得分更高的延续（即保留 len(list(generation_config.eos_token_id)) + 1）
        if generation_config.num_beams > 1:
            if isinstance(generation_config.eos_token_id, list):
                min_tokens_to_keep = len(generation_config.eos_token_id) + 1
            else:
                min_tokens_to_keep = 2
        else:
            min_tokens_to_keep = 1

        # the following idea is largely copied from this PR: https://github.com/huggingface/transformers/pull/5420/files
        # all samplers can be found in `generation_utils_samplers.py`
        # 以下想法在很大程度上是从此 PR 复制的：https://github.com/huggingface/transformers/pull/5420/files
        # 所有的采样器都可以在 `generation_utils_samplers.py` 中找到
        if generation_config.temperature is not None and generation_config.temperature != 1.0:
            warpers.append(TemperatureLogitsWarper(generation_config.temperature))
        if generation_config.top_k is not None and generation_config.top_k != 0:
            warpers.append(TopKLogitsWarper(top_k=generation_config.top_k, min_tokens_to_keep=min_tokens_to_keep))
        if generation_config.top_p is not None and generation_config.top_p < 1.0:
            warpers.append(TopPLogitsWarper(top_p=generation_config.top_p, min_tokens_to_keep=min_tokens_to_keep))
        if generation_config.typical_p is not None and generation_config.typical_p < 1.0:
            warpers.append(
                TypicalLogitsWarper(mass=generation_config.typical_p, min_tokens_to_keep=min_tokens_to_keep)
            )
        if generation_config.epsilon_cutoff is not None and 0.0 < generation_config.epsilon_cutoff < 1.0:
            warpers.append(
                EpsilonLogitsWarper(epsilon=generation_config.epsilon_cutoff, min_tokens_to_keep=min_tokens_to_keep)
            )
        if generation_config.eta_cutoff is not None and 0.0 < generation_config.eta_cutoff < 1.0:
            warpers.append(
                EtaLogitsWarper(epsilon=generation_config.eta_cutoff, min_tokens_to_keep=min_tokens_to_keep)
            )
        # `LogitNormalization` should always be the last logit processor, when present
        # `LogitNormalization` 应该始终是最后一个 logit 处理器，如果存在的话
        if generation_config.renormalize_logits is True:
            warpers.append(LogitNormalization())
        return warpers

    def _get_generation_mode(
        self, generation_config: GenerationConfig, assistant_model: Optional["PreTrainedModel"]
    # 返回由 GenerationConfig 实例触发的生成模式
    def get_generation_mode(self, generation_config: GenerationConfig, assistant_model: Optional[Any] = None) -> GenerationMode:
        # 如果生成配置中包含约束或强制词汇 ID，则使用 CONSTRAINED_BEAM_SEARCH 生成模式
        if generation_config.constraints is not None or generation_config.force_words_ids is not None:
            generation_mode = GenerationMode.CONSTRAINED_BEAM_SEARCH
        # 如果 num_beams 为 1
        elif generation_config.num_beams == 1:
            # 如果不进行采样
            if generation_config.do_sample is False:
                # 如果设置了 top_k 大于 1，penalty_alpha 大于 0，则使用 CONTRASTIVE_SEARCH 生成模式
                if (
                    generation_config.top_k is not None
                    and generation_config.top_k > 1
                    and generation_config.penalty_alpha is not None
                    and generation_config.penalty_alpha > 0
                ):
                    generation_mode = GenerationMode.CONTRASTIVE_SEARCH
                # 否则使用 GREEDY_SEARCH 生成模式
                else:
                    generation_mode = GenerationMode.GREEDY_SEARCH
            # 如果进行采样，则使用 SAMPLE 生成模式
            else:
                generation_mode = GenerationMode.SAMPLE
        else:
            # 如果 num_beam_groups 大于 1，则使用 GROUP_BEAM_SEARCH 生成模式
            if generation_config.num_beam_groups > 1:
                generation_mode = GenerationMode.GROUP_BEAM_SEARCH
            # 如果进行采样，则使用 BEAM_SAMPLE 生成模式
            elif generation_config.do_sample is True:
                generation_mode = GenerationMode.BEAM_SAMPLE
            # 否则使用 BEAM_SEARCH 生成模式
            else:
                generation_mode = GenerationMode.BEAM_SEARCH

        # 辅助生成可能会扩展一些生成模式
        if assistant_model is not None or generation_config.prompt_lookup_num_tokens is not None:
            # 如果当前生成模式为 GREEDY_SEARCH 或 SAMPLE，则使用 ASSISTED_GENERATION 生成模式
            if generation_mode in ("greedy_search", "sample"):
                generation_mode = GenerationMode.ASSISTED_GENERATION
            else:
                # 否则抛出异常
                raise ValueError(
                    "You've set `assistant_model`, which triggers assisted generate. Currently, assisted generate "
                    "is only supported with Greedy Search and Sample."
                )
        return generation_mode

    # 获取 logits 处理器
    def _get_logits_processor(
        self,
        generation_config: GenerationConfig,
        input_ids_seq_length: int,
        encoder_input_ids: torch.LongTensor,
        prefix_allowed_tokens_fn: Callable[[int, torch.Tensor], List[int]],
        logits_processor: Optional[LogitsProcessorList],
        model_kwargs: Optional[Dict[str, Any]] = None,
        negative_prompt_ids: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
    # 获取停止标准
    def _get_stopping_criteria(
        self, generation_config: GenerationConfig, stopping_criteria: Optional[StoppingCriteriaList]
    # 返回一个包含停止条件的列表
    def __call__(self) -> StoppingCriteriaList:
        # 创建一个空的停止条件列表
        criteria = StoppingCriteriaList()
        # 如果生成配置中指定了最大长度
        if generation_config.max_length is not None:
            # 获取最大位置嵌入长度
            max_position_embeddings = getattr(self.config, "max_position_embeddings", None)
            # 添加最大长度停止条件到列表中
            criteria.append(
                MaxLengthCriteria(
                    max_length=generation_config.max_length,
                    max_position_embeddings=max_position_embeddings,
                )
            )
        # 如果生成配置中指定了最大时间
        if generation_config.max_time is not None:
            # 添加最大时间停止条件到列表中
            criteria.append(MaxTimeCriteria(max_time=generation_config.max_time))
        # 合并默认列表和自定义列表中的停止条件
        criteria = self._merge_criteria_processor_list(criteria, stopping_criteria)
        # 返回停止条件列表
        return criteria

    # 合并默认列表和自定义列表中的处理器列表
    def _merge_criteria_processor_list(
        self,
        default_list: Union[LogitsProcessorList, StoppingCriteriaList],
        custom_list: Union[LogitsProcessorList, StoppingCriteriaList],
    ) -> Union[LogitsProcessorList, StoppingCriteriaList]:
        # 如果自定义列表为空，则返回默认列表
        if len(custom_list) == 0:
            return default_list
        # 遍历默认列表
        for default in default_list:
            # 遍历自定义列表
            for custom in custom_list:
                # 如果自定义处理器与默认处理器类型相同
                if type(custom) is type(default):
                    # 确定对象类型
                    object_type = "stopping criteria" if isinstance(custom, StoppingCriteria) else "logits processor"
                    # 抛出值错误，提示用户不要重复传递自定义处理器
                    raise ValueError(
                        f"A custom {object_type} of type {type(custom)} with values {custom} has been passed to"
                        f" `.generate()`, but it has already been created with the values {default}. {default} has been"
                        " created by passing the corresponding arguments to generate or by the model's config default"
                        f" values. If you just want to change the default values of {object_type} consider passing"
                        f" them as arguments to `.generate()` instead of using a custom {object_type}."
                    )
        # 将自定义列表中的处理器添加到默认列表中
        default_list.extend(custom_list)
        # 返回合并后的列表
        return default_list

    # 计算转移分数
    def compute_transition_scores(
        self,
        sequences: torch.Tensor,
        scores: Tuple[torch.Tensor],
        beam_indices: Optional[torch.Tensor] = None,
        normalize_logits: bool = False,
    # 验证模型类是否与生成兼容，如果不兼容，则引发异常，并指出正确的类使用方式
    def _validate_model_class(self):
        """
        Confirms that the model class is compatible with generation. If not, raises an exception that points to the
        right class to use.
        """
        # 如果模型无法生成
        if not self.can_generate():
            # 定义与生成兼容的映射列表
            generate_compatible_mappings = [
                MODEL_FOR_CAUSAL_LM_MAPPING,
                MODEL_FOR_CAUSAL_IMAGE_MODELING_MAPPING,
                MODEL_FOR_VISION_2_SEQ_MAPPING,
                MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING,
                MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING,
            ]
            # 创建用于存储生成兼容类的集合
            generate_compatible_classes = set()
            # 遍历生成兼容映射列表
            for model_mapping in generate_compatible_mappings:
                # 获取当前模型配置的支持模型，如果没有则使用默认值None
                supported_models = model_mapping.get(type(self.config), default=None)
                # 如果支持模型不为None，则将其类名添加到生成兼容类集合中
                if supported_models is not None:
                    generate_compatible_classes.add(supported_models.__name__)
            # 构建异常消息
            exception_message = (
                f"The current model class ({self.__class__.__name__}) is not compatible with `.generate()`, as "
                "it doesn't have a language model head."
            )
            # 如果存在生成兼容类，则在异常消息中加入建议使用的类名
            if generate_compatible_classes:
                exception_message += f" Please use one of the following classes instead: {generate_compatible_classes}"
            # 抛出类型错误异常，包含异常消息
            raise TypeError(exception_message)
    # 执行与生成长度相关的验证

    # 1. 与参数配置不佳相关的最大长度警告
    if has_default_max_length and generation_config.max_new_tokens is None and generation_config.max_length == 20:
        # 20 是生成配置的默认 max_length
        warnings.warn(
            f"Using the model-agnostic default `max_length` (={generation_config.max_length}) to control the "
            "generation length. We recommend setting `max_new_tokens` to control the maximum length of the "
            "generation.",
            UserWarning,
        )
    if input_ids_length >= generation_config.max_length:
        input_ids_string = "decoder_input_ids" if self.config.is_encoder_decoder else "input_ids"
        warnings.warn(
            f"Input length of {input_ids_string} is {input_ids_length}, but `max_length` is set to"
            f" {generation_config.max_length}. This can lead to unexpected behavior. You should consider"
            " increasing `max_new_tokens`.",
            UserWarning,
        )

    # 2. 由于不可行的参数组合而导致的最小长度警告
    min_length_error_suffix = (
        " Generation will stop at the defined maximum length. You should decrease the minimum length and/or "
        "increase the maximum length."
    )
    if has_default_max_length:
        min_length_error_suffix += (
            f" Note that `max_length` is set to {generation_config.max_length}, its default value."
        )
    if generation_config.min_length is not None and generation_config.min_length > generation_config.max_length:
        warnings.warn(
            f"Unfeasible length constraints: `min_length` ({generation_config.min_length}) is larger than"
            f" the maximum possible length ({generation_config.max_length})." + min_length_error_suffix,
            UserWarning,
        )
    if generation_config.min_new_tokens is not None:
        min_length = generation_config.min_new_tokens + input_ids_length
        if min_length > generation_config.max_length:
            warnings.warn(
                f"Unfeasible length constraints: `min_new_tokens` ({generation_config.min_new_tokens}), when "
                f"added to the prompt length ({input_ids_length}), is larger than"
                f" the maximum possible length ({generation_config.max_length})." + min_length_error_suffix,
                UserWarning,
            )

@torch.no_grad()
    # 定义一个生成方法，用于生成文本
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,  # 输入的张量，默认为None
        generation_config: Optional[GenerationConfig] = None,  # 生成配置，默认为None
        logits_processor: Optional[LogitsProcessorList] = None,  # logits处理器列表，默认为None
        stopping_criteria: Optional[StoppingCriteriaList] = None,  # 停止条件列表，默认为None
        prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,  # 前缀允许的标记函数，默认为None
        synced_gpus: Optional[bool] = None,  # 同步的GPU，默认为None
        assistant_model: Optional["PreTrainedModel"] = None,  # 助理模型，默认为None
        streamer: Optional["BaseStreamer"] = None,  # 流处理器，默认为None
        negative_prompt_ids: Optional[torch.Tensor] = None,  # 负面提示的ID，默认为None
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,  # 负面提示的注意力掩码，默认为None
        **kwargs,  # 其他关键字参数
    @torch.no_grad()  # 禁用梯度计算
    # 对比搜索方法，用于搜索与输入ID最相似的文本
    def contrastive_search(
        self,
        input_ids: torch.LongTensor,  # 输入的长整型张量
        top_k: Optional[int] = 1,  # 前k个结果，默认为1
        penalty_alpha: Optional[float] = 0,  # 惩罚系数，默认为0
        logits_processor: Optional[LogitsProcessorList] = None,  # logits处理器列表，默认为None
        logits_warper: Optional[LogitsProcessorList] = None,  # logits调整器列表，默认为None
        stopping_criteria: Optional[StoppingCriteriaList] = None,  # 停止条件列表，默认为None
        pad_token_id: Optional[int] = None,  # 填充标记ID，默认为None
        eos_token_id: Optional[Union[int, List[int]]] = None,  # 结束标记ID，默认为None
        output_attentions: Optional[bool] = None,  # 输出注意力，默认为None
        output_hidden_states: Optional[bool] = None,  # 输出隐藏状态，默认为None
        output_scores: Optional[bool] = None,  # 输出分数，默认为None
        return_dict_in_generate: Optional[bool] = None,  # 在生成中返回字典，默认为None
        synced_gpus: bool = False,  # 同步的GPU，默认为False
        streamer: Optional["BaseStreamer"] = None,  # 流处理器，默认为None
        sequential: Optional[bool] = None,  # 顺序的，默认为None
        **model_kwargs,  # 模型关键字参数
    # 贪婪搜索方法，用于贪婪地生成文本
    def greedy_search(
        self,
        input_ids: torch.LongTensor,  # 输入的长整型张量
        logits_processor: Optional[LogitsProcessorList] = None,  # logits处理器列表，默认为None
        stopping_criteria: Optional[StoppingCriteriaList] = None,  # 停止条件列表，默认为None
        max_length: Optional[int] = None,  # 最大长度，默认为None
        pad_token_id: Optional[int] = None,  # 填充标记ID，默认为None
        eos_token_id: Optional[Union[int, List[int]]] = None,  # 结束标记ID，默认为None
        output_attentions: Optional[bool] = None,  # 输出注意力，默认为None
        output_hidden_states: Optional[bool] = None,  # 输出隐藏状态，默认为None
        output_scores: Optional[bool] = None,  # 输出分数，默认为None
        return_dict_in_generate: Optional[bool] = None,  # 在生成中返回字典，默认为None
        synced_gpus: bool = False,  # 同步的GPU，默认为False
        streamer: Optional["BaseStreamer"] = None,  # 流处理器，默认为None
        **model_kwargs,  # 模型关键字参数
    # 随机采样方法，用于随机生成文本
    def sample(
        self,
        input_ids: torch.LongTensor,  # 输入的长整型张量
        logits_processor: Optional[LogitsProcessorList] = None,  # logits处理器列表，默认为None
        stopping_criteria: Optional[StoppingCriteriaList] = None,  # 停止条件列表，默认为None
        logits_warper: Optional[LogitsProcessorList] = None,  # logits调整器列表，默认为None
        max_length: Optional[int] = None,  # 最大长度，默认为None
        pad_token_id: Optional[int] = None,  # 填充标记ID，默认为None
        eos_token_id: Optional[Union[int, List[int]]] = None,  # 结束标记ID，默认为None
        output_attentions: Optional[bool] = None,  # 输出注意力，默认为None
        output_hidden_states: Optional[bool] = None,  # 输出隐藏状态，默认为None
        output_scores: Optional[bool] = None,  # 输出分数，默认为None
        return_dict_in_generate: Optional[bool] = None,  # 在生成中返回字典，默认为None
        synced_gpus: bool = False,  # 同步的GPU，默认为False
        streamer: Optional["BaseStreamer"] = None,  # 流处理器，默认为None
        **model_kwargs,  # 模型关键字参数
    def _temporary_reorder_cache(self, past_key_values, beam_idx):
        """
        Temporary function to handle the different types of cache reordering processes while we roll out `Cache`.

        TODO: standardize cache formats and make all models compatible with `Cache`. It would remove the need
        for this function, with `Cache.reorder_cache` being the sole remaining code path
        """
        # 获取模型类名，用于判断模型是否属于特定类型
        model_class = self.__class__.__name__.lower()
        # 异常情况1：处理使用传统缓存格式的模型
        if isinstance(past_key_values, (tuple, list)):
            # 重新排序缓存
            past_key_values = self._reorder_cache(past_key_values, beam_idx)
        # 异常情况2：处理缓存格式不同的模型
        elif "bloom" in model_class or "gptbigcode" in model_class:
            if not isinstance(past_key_values, DynamicCache):
                # 抛出值错误，提示不支持的缓存格式
                raise ValueError(
                    f"Using an unsupported cache format with {model_class}. Currently, it only supports the "
                    "legacy tuple format or `DynamicCache`"
                )
            # 重新排序缓存
            past_key_values = self._reorder_cache(past_key_values, beam_idx)
            # 将传统缓存格式转换为动态缓存
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)
        # 标准情况：使用`Cache.reorder_cache`
        else:
            # 调用缓存对象的重新排序方法
            past_key_values.reorder_cache(beam_idx)
        # 返回重新排序后的缓存
        return past_key_values

    def beam_search(
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
        return_dict_in_generate: Optional[bool] = None,
        synced_gpus: bool = False,
        **model_kwargs,
    ):
        # 略


    def beam_sample(
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
        return_dict_in_generate: Optional[bool] = None,
        synced_gpus: bool = False,
        **model_kwargs,
    ):
        # 略
    # 使用束搜索算法对输入进行分组搜索
    def group_beam_search(
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
        return_dict_in_generate: Optional[bool] = None,
        synced_gpus: bool = False,
        **model_kwargs,
    # 使用受限束搜索算法对输入进行搜索
    def constrained_beam_search(
        self,
        input_ids: torch.LongTensor,
        constrained_beam_scorer: ConstrainedBeamSearchScorer,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        synced_gpus: Optional[bool] = None,
        **model_kwargs,
    # 使用辅助解码算法对输入进行解码
    def assisted_decoding(
        self,
        input_ids: torch.LongTensor,
        assistant_model: Optional["PreTrainedModel"] = None,
        candidate_generator: Optional["CandidateGenerator"] = None,
        do_sample: bool = False,
        logits_processor: Optional[LogitsProcessorList] = None,
        logits_warper: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        synced_gpus: bool = False,
        streamer: Optional["BaseStreamer"] = None,
        **model_kwargs,
def _speculative_sampling(
    candidate_input_ids,
    candidate_logits,
    candidate_length,
    new_logits,
    last_assistant_token_is_eos,
    max_matches,
):
    """
    应用类似于投机解码论文（https://arxiv.org/pdf/2211.17192.pdf，算法1）中的采样方法。返回所选标记以及候选匹配数。

    注意：除非另有说明，变量名与论文中相同。
    """
    # 从 logits 中获取概率。q_i 和 p_i 分别表示助手选择的标记的助手和模型概率。
    q = candidate_logits.softmax(dim=-1)
    q_i = q[:, torch.arange(candidate_length), candidate_input_ids[:, -candidate_length:]].squeeze(0, 1)
    p = new_logits.softmax(dim=-1)
    p_i = p[:, torch.arange(candidate_length), candidate_input_ids[:, -candidate_length:]].squeeze(0, 1)
    probability_ratio = p_i / q_i

    # 当 probability_ratio > 1 时（即 q_i(x) < p_i(x)，“候选标记的助手概率小于相同标记的模型概率”），保留标记。
    # 否则以 p = 1 - probability_ratio 拒绝（以 p = probability_ratio 保留）。保留所有标记直到第一个拒绝。
    r_i = torch.rand_like(probability_ratio)
    is_accepted = r_i <= probability_ratio
    n_matches = ((~is_accepted).cumsum(dim=-1) < 1).sum()  # 这是算法1中的 `n`

    # 确保我们不会生成超过 max_len 或 EOS 标记（不在算法1中，但为了正确的行为而需要）
    if last_assistant_token_is_eos and n_matches == candidate_length:
        # 假设输出长度为 `n_matches + 1`。由于我们由于在 EOS 上接受不会再生成另一个目标模型的标记，我们固定了 `n_matches`
        n_matches -= 1
        valid_tokens = candidate_input_ids[:, -candidate_length:-candidate_length + n_matches + 1]
    else:
        n_matches = min(n_matches, max_matches)

        # 下一个标记选择：如果有拒绝，调整主模型的分布后再采样。
        gamma = min(candidate_logits.shape[1], max_matches)
        p_n_plus_1 = p[:, n_matches, :]
        if n_matches < gamma:
            q_n_plus_1 = q[:, n_matches, :]
            p_prime = torch.clamp((p_n_plus_1 - q_n_plus_1), min=0)
            p_prime.div_(p_prime.sum())
        else:
            p_prime = p_n_plus_1
        t = torch.multinomial(p_prime, num_samples=1).squeeze(1)[None, :]

        # 所选标记包括匹配（如果有）加上下一个采样标记
        if n_matches > 0:
            valid_tokens = torch.cat((candidate_input_ids[:, -candidate_length:-candidate_length + n_matches], t), dim=-1)
        else:
            valid_tokens = t

    return valid_tokens, n_matches


def _split_model_outputs(outputs, new_outputs, cur_len, added_len, is_decoder_attention=False):
    """
    Given the (decoder/cross attentions)/(decoder hidden states) for multiple generated tokens, splits it into a tuple
    where each member corresponds to a single generated token.
    """
    # Retrocompatibility: in our generation functions, the first iteration includes the attention/hidden states for the
    # prompt.
    # 如果输出为空，则将第一个生成的 token 的注意力/隐藏状态添加到输出中
    if len(outputs) == 0:
        new_tuple = ()
        for layer in new_outputs:
            last_dim_size = cur_len if is_decoder_attention else layer.shape[-1]
            new_tuple += (layer[..., :cur_len, :last_dim_size],)
        outputs += (new_tuple,)
        # 第一个迭代包含提示 + 1 个生成的 token，因此相应地更新长度变量
        cur_len += 1
        added_len -= cur_len

    # 遍历每个新增的 token
    for i in range(added_len):
        new_tuple = ()
        for layer in new_outputs:
            last_dim_size = cur_len + i if is_decoder_attention else layer.shape[-1]
            new_tuple += (layer[..., i : i + 1, :last_dim_size],)
        outputs += (new_tuple,)
    # 返回拆分后的输出
    return outputs
# 使用 top-k 和/或 nucleus（top-p）过滤来过滤 logits 分布
def top_k_top_p_filtering(
    logits: torch.FloatTensor,
    top_k: int = 0,
    top_p: float = 1.0,
    filter_value: float = -float("Inf"),
    min_tokens_to_keep: int = 1,
) -> torch.FloatTensor:
    """
    根据 top-k 和/或 nucleus（top-p）过滤来过滤 logits 分布

    Args:
        logits: logits 分布的形状（批量大小，词汇大小）
        top_k (`int`, *optional*, defaults to 0):
            如果 > 0，则仅保留具有最高概率的前 k 个令牌（top-k 过滤）
        top_p (`float`, *optional*, defaults to 1.0):
            如果 < 1.0，则仅保留累积概率 >= top_p 的前几个令牌（nucleus 过滤）。Nucleus 过滤在 Holtzman 等人的论文中有描述（http://arxiv.org/abs/1904.09751）
        min_tokens_to_keep (`int`, *optional*, defaults to 1):
            输出中每个批量示例保留的最小令牌数。

    From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    # 发出警告，表明 `top_k_top_p_filtering` 在 v4.39 中将被删除。请使用 `TopKLogitsWarper` 和 `TopPLogitsWarper` 替代。
    warnings.warn(
        "`top_k_top_p_filtering` is scheduled for deletion in v4.39. Use `TopKLogitsWarper` and `TopPLogitsWarper` "
        "instead.",
        DeprecationWarning,
    )

    # 如果 top_k 大于 0，则使用 TopKLogitsWarper 进行过滤
    if top_k > 0:
        logits = TopKLogitsWarper(top_k=top_k, filter_value=filter_value, min_tokens_to_keep=min_tokens_to_keep)(
            None, logits
        )

    # 如果 0 <= top_p <= 1.0，则使用 TopPLogitsWarper 进行过滤
    if 0 <= top_p <= 1.0:
        logits = TopPLogitsWarper(top_p=top_p, filter_value=filter_value, min_tokens_to_keep=min_tokens_to_keep)(
            None, logits
        )

    # 返回过滤后的 logits
    return logits


# 根据先前标记的余弦相似性对 top_k 候选项进行重新排名，如论文 "A Contrastive Framework for Neural Text Generation" 中所述。返回每个批量示例中最佳候选项的索引。
def _ranking_fast(
    context_hidden: torch.FloatTensor,
    next_hidden: torch.FloatTensor,
    next_top_k_probs: torch.FloatTensor,
    alpha: float,
    beam_width: int,
) -> torch.FloatTensor:
    """
    根据一个退化惩罚（与先前令牌的余弦相似性）对 top_k 候选项进行重新排名，如论文 "A Contrastive Framework for Neural Text Generation" 所述。返回每个批量示例中最佳候选项的索引。
    """
    # 对上下文隐藏状态和下一个隐藏状态进行归一化
    norm_context_hidden = context_hidden / context_hidden.norm(dim=2, keepdim=True)
    norm_next_hidden = next_hidden / next_hidden.norm(dim=2, keepdim=True)
    # 计算余弦相似性矩阵
    cosine_matrix = torch.matmul(norm_context_hidden, norm_next_hidden.transpose(1, 2)).squeeze(-1)  # [B*K, S]
    # 获取每个候选项的最大余弦相似性
    degeneration_penalty, _ = torch.max(cosine_matrix, dim=-1)  # [B*K]
    next_top_k_probs = next_top_k_probs.view(-1)  # [B*K]
    # 计算对比分数
    contrastive_score = (1.0 - alpha) * next_top_k_probs - alpha * degeneration_penalty
    contrastive_score = torch.stack(torch.split(contrastive_score, beam_width))  # [B, K]
    # 获取每个批量示例中最佳候选项的索引
    _, selected_idx = contrastive_score.max(dim=-1)  # [B]
    return selected_idx
```