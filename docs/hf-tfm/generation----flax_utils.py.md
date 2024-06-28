# `.\generation\flax_utils.py`

```py
# 导入必要的库和模块
import copy                    # 导入 copy 模块，用于复制对象
import inspect                 # 导入 inspect 模块，用于检查对象
import warnings                # 导入 warnings 模块，用于警告处理
from functools import partial  # 导入 partial 函数，用于创建部分应用的函数
from typing import Any, Dict, Optional, Union  # 导入类型提示相关模块

import flax                   # 导入 flax 框架
import jax                    # 导入 jax 框架
import jax.numpy as jnp       # 导入 jax 的 numpy 接口作为 jnp
import numpy as np            # 导入 numpy 库
from jax import lax           # 导入 jax 的 lax 模块

from ..models.auto import (   # 导入自定义模块中的多个自动模型映射
    FLAX_MODEL_FOR_CAUSAL_LM_MAPPING,
    FLAX_MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING,
    FLAX_MODEL_FOR_VISION_2_SEQ_MAPPING,
)
from ..utils import ModelOutput, logging  # 导入自定义模块中的 ModelOutput 和 logging
from .configuration_utils import GenerationConfig  # 导入本地模块中的 GenerationConfig 类
from .flax_logits_process import (  # 导入本地模块中的多个 logits 处理器类
    FlaxForcedBOSTokenLogitsProcessor,
    FlaxForcedEOSTokenLogitsProcessor,
    FlaxForceTokensLogitsProcessor,
    FlaxLogitsProcessorList,
    FlaxMinLengthLogitsProcessor,
    FlaxSuppressTokensAtBeginLogitsProcessor,
    FlaxSuppressTokensLogitsProcessor,
    FlaxTemperatureLogitsWarper,
    FlaxTopKLogitsWarper,
    FlaxTopPLogitsWarper,
)

logger = logging.get_logger(__name__)  # 获取当前模块的 logger 实例


@flax.struct.dataclass
class FlaxGreedySearchOutput(ModelOutput):
    """
    Flax Base class for outputs of decoder-only generation models using greedy search.


    Args:
        sequences (`jnp.ndarray` of shape `(batch_size, max_length)`):
            The generated sequences.
    """

    sequences: jnp.ndarray = None  # 类属性，存储生成的序列数据


@flax.struct.dataclass
class FlaxSampleOutput(ModelOutput):
    """
    Flax Base class for outputs of decoder-only generation models using sampling.


    Args:
        sequences (`jnp.ndarray` of shape `(batch_size, max_length)`):
            The generated sequences.
    """

    sequences: jnp.ndarray = None  # 类属性，存储生成的序列数据


@flax.struct.dataclass
class FlaxBeamSearchOutput(ModelOutput):
    """
    Flax Base class for outputs of decoder-only generation models using beam search.


    Args:
        sequences (`jnp.ndarray` of shape `(batch_size, max_length)`):
            The generated sequences.
        scores (`jnp.ndarray` of shape `(batch_size,)`):
            The scores (log probabilities) of the generated sequences.
    """

    sequences: jnp.ndarray = None  # 类属性，存储生成的序列数据
    scores: jnp.ndarray = None     # 类属性，存储生成序列的分数（对数概率）


@flax.struct.dataclass
class GreedyState:
    """
    Dataclass to store state during greedy decoding.

    Args:
        cur_len (`jnp.ndarray`): Current lengths of sequences.
        sequences (`jnp.ndarray`): Generated sequences.
        running_token (`jnp.ndarray`): Running tokens for decoding.
        is_sent_finished (`jnp.ndarray`): Boolean array indicating finished sentences.
        model_kwargs (Dict[str, jnp.ndarray]): Additional model arguments.
    """

    cur_len: jnp.ndarray            # 当前序列长度
    sequences: jnp.ndarray          # 生成的序列
    running_token: jnp.ndarray      # 解码中的当前 token
    is_sent_finished: jnp.ndarray   # 表示句子是否结束的布尔数组
    model_kwargs: Dict[str, jnp.ndarray]  # 存储额外模型参数的字典


@flax.struct.dataclass
class SampleState:
    """
    Dataclass to store state during sampling.

    Args:
        cur_len (`jnp.ndarray`): Current lengths of sequences.
    """

    cur_len: jnp.ndarray  # 当前序列长度
    # 定义变量 sequences，类型为 jnp.ndarray，用于存储序列数据
    sequences: jnp.ndarray
    # 定义变量 running_token，类型为 jnp.ndarray，用于存储运行中的标记数据
    running_token: jnp.ndarray
    # 定义变量 is_sent_finished，类型为 jnp.ndarray，用于存储句子完成状态的数据
    is_sent_finished: jnp.ndarray
    # 定义变量 prng_key，类型为 jnp.ndarray，用于存储伪随机数生成器密钥的数据
    prng_key: jnp.ndarray
    # 定义变量 model_kwargs，类型为 Dict[str, jnp.ndarray]，用于存储模型参数的字典，其中键为字符串，值为 jnp.ndarray 类型
    model_kwargs: Dict[str, jnp.ndarray]
@flax.struct.dataclass
class BeamSearchState:
    cur_len: jnp.ndarray  # 当前长度，作为一个 NumPy 数组
    running_sequences: jnp.ndarray  # 正在运行的序列，作为一个 NumPy 数组
    running_scores: jnp.ndarray  # 运行中的分数，作为一个 NumPy 数组
    sequences: jnp.ndarray  # 序列，作为一个 NumPy 数组
    scores: jnp.ndarray  # 分数，作为一个 NumPy 数组
    is_sent_finished: jnp.ndarray  # 标志句子是否完成的数组，作为一个 NumPy 数组
    model_kwargs: Dict[str, jnp.ndarray]  # 模型参数，字典形式，键为字符串，值为 NumPy 数组


class FlaxGenerationMixin:
    """
    包含自回归文本生成的所有函数的类，作为[`FlaxPreTrainedModel`]的混合类使用。

    该类公开[`~generation.FlaxGenerationMixin.generate`]方法，可用于：
            - 当`num_beams=1`且`do_sample=False`时通过调用[`~generation.FlaxGenerationMixin._greedy_search`]进行贪婪解码
            - 当`num_beams=1`且`do_sample=True`时通过调用[`~generation.FlaxGenerationMixin._sample`]进行多项式采样
            - 当`num_beams>1`且`do_sample=False`时通过调用[`~generation.FlaxGenerationMixin._beam_search`]进行束搜索解码

    无需直接调用上述任何方法。只需将自定义参数值传递给'generate'方法即可。有关解码策略的更多信息，请参阅[文本生成策略指南](../generation_strategies)。
    """

    def prepare_inputs_for_generation(self, *args, **kwargs):
        raise NotImplementedError(
            "A model class needs to define a `prepare_inputs_for_generation` method in order to use `generate`."
        )

    @staticmethod
    def _run_loop_in_debug(cond_fn, body_fn, init_state):
        """
        以非跟踪模式运行生成过程。仅用于调试目的。
        """
        state = init_state  # 初始化状态
        while cond_fn(state):  # 当条件函数为真时循环执行
            state = body_fn(state)  # 执行主体函数
        return state  # 返回最终状态

    def _prepare_encoder_decoder_kwargs_for_generation(self, input_ids, params, model_kwargs):
        encoder_kwargs = {
            argument: value
            for argument, value in model_kwargs.items()
            if not (argument.startswith("decoder_") or argument.startswith("cross_attn"))
        }
        model_kwargs["encoder_outputs"] = self.encode(input_ids, params=params, return_dict=True, **encoder_kwargs)
        return model_kwargs

    def _prepare_decoder_input_ids_for_generation(
        self,
        batch_size: int,
        decoder_start_token_id: int = None,
        bos_token_id: int = None,
        model_kwargs: Optional[Dict[str, jnp.ndarray]] = None,
    ) -> jnp.ndarray:
        if model_kwargs is not None and "decoder_input_ids" in model_kwargs:
            # 如果模型参数中存在'decoder_input_ids'，则使用它，否则从模型参数中移除
            decoder_input_ids = model_kwargs.pop("decoder_input_ids")
            if decoder_input_ids is not None:
                return decoder_input_ids  # 返回decoder_input_ids
        decoder_start_token_id = self._get_decoder_start_token_id(decoder_start_token_id, bos_token_id)
        return jnp.array(decoder_start_token_id, dtype="i4").reshape(1, -1).repeat(batch_size, axis=0)
    def _get_decoder_start_token_id(self, decoder_start_token_id: int = None, bos_token_id: int = None) -> int:
        # 检索用于编码器-解码器模型的decoder_start_token_id
        # 如果需要，可以回退到bos_token_id
        decoder_start_token_id = (
            decoder_start_token_id
            if decoder_start_token_id is not None
            else self.generation_config.decoder_start_token_id
        )
        bos_token_id = bos_token_id if bos_token_id is not None else self.generation_config.bos_token_id
        # 如果decoder_start_token_id已经定义，则返回它
        if decoder_start_token_id is not None:
            return decoder_start_token_id
        # 否则，检查配置是否具有decoder_start_token_id，并且不为None
        elif (
            hasattr(self.config, "decoder")
            and hasattr(self.config.decoder, "decoder_start_token_id")
            and self.config.decoder.decoder_start_token_id is not None
        ):
            return self.config.decoder.decoder_start_token_id
        # 如果以上条件不满足，检查是否定义了bos_token_id，并且不为None
        elif bos_token_id is not None:
            return bos_token_id
        # 最后如果bos_token_id也未定义，则引发ValueError
        elif (
            hasattr(self.config, "decoder")
            and hasattr(self.config.decoder, "bos_token_id")
            and self.config.decoder.bos_token_id is not None
        ):
            return self.config.decoder.bos_token_id
        raise ValueError(
            "`decoder_start_token_id` or `bos_token_id` has to be defined for encoder-decoder generation."
        )

    @staticmethod
    def _expand_to_num_beams(tensor, num_beams):
        # 将tensor扩展为num_beams数量的beam搜索结果
        return jnp.broadcast_to(tensor[:, None], (tensor.shape[0], num_beams) + tensor.shape[1:])

    def _adapt_logits_for_beam_search(self, logits):
        """
        This function can be overwritten in the specific modeling_flax_<model-name>.py classes to allow for custom beam
        search behavior. Note that the only model that overwrites this method is [`~transformes.FlaxMarianMTModel`].
        """
        # 默认情况下，直接返回logits，这个方法可以在具体的modeling_flax_<model-name>.py类中被覆盖，以允许自定义beam搜索行为。
        return logits
    def _validate_model_class(self):
        """
        Confirms that the model class is compatible with generation. If not, raises an exception that points to the
        right class to use.
        """
        # 检查当前模型是否支持生成操作
        if not self.can_generate():
            # 定义支持生成操作的模型映射列表
            generate_compatible_mappings = [
                FLAX_MODEL_FOR_CAUSAL_LM_MAPPING,
                FLAX_MODEL_FOR_VISION_2_SEQ_MAPPING,
                FLAX_MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING,
            ]
            # 收集所有兼容的模型类名
            generate_compatible_classes = set()
            for model_mapping in generate_compatible_mappings:
                # 获取当前模型配置对应的支持模型
                supported_models = model_mapping.get(type(self.config), default=None)
                if supported_models is not None:
                    generate_compatible_classes.add(supported_models.__name__)
            # 构建异常消息
            exception_message = (
                f"The current model class ({self.__class__.__name__}) is not compatible with `.generate()`, as "
                "it doesn't have a language model head."
            )
            # 如果存在兼容的模型类，则添加建议使用的类名到异常消息中
            if generate_compatible_classes:
                exception_message += f" Please use one of the following classes instead: {generate_compatible_classes}"
            # 抛出类型错误异常，指示模型类不兼容生成操作
            raise TypeError(exception_message)

    def _validate_model_kwargs(self, model_kwargs: Dict[str, Any]):
        """Validates model kwargs for generation. Generate argument typos will also be caught here."""
        # 初始化未使用的模型参数列表
        unused_model_args = []
        # 获取用于生成输入的参数名称集合
        model_args = set(inspect.signature(self.prepare_inputs_for_generation).parameters)
        # 如果 `kwargs` 或 `model_kwargs` 在模型参数中，扩展模型参数集合
        if "kwargs" in model_args or "model_kwargs" in model_args:
            model_args |= set(inspect.signature(self.__call__).parameters)
        # 检查传入的 `model_kwargs` 是否有未使用的参数
        for key, value in model_kwargs.items():
            if value is not None and key not in model_args:
                unused_model_args.append(key)

        # 如果存在未使用的模型参数，抛出值错误异常
        if unused_model_args:
            raise ValueError(
                f"The following `model_kwargs` are not used by the model: {unused_model_args} (note: typos in the"
                " generate arguments will also show up in this list)"
            )
    
    def generate(
        self,
        input_ids: jnp.ndarray,
        generation_config: Optional[GenerationConfig] = None,
        prng_key: Optional[jnp.ndarray] = None,
        trace: bool = True,
        params: Optional[Dict[str, jnp.ndarray]] = None,
        logits_processor: Optional[FlaxLogitsProcessorList] = None,
        **kwargs,
    def _get_logits_warper(self, generation_config: GenerationConfig) -> FlaxLogitsProcessorList:
        """
        返回一个 [`FlaxLogitsProcessorList`] 列表对象，其中包含所有用于多项式采样的相关 [`FlaxLogitsWarper`] 实例。
        """
        # 创建一个空的 FlaxLogitsProcessorList 对象，用于存储 Logits 处理器
        warpers = FlaxLogitsProcessorList()

        # 如果设置了温度且不等于 1.0，则添加温度调整器
        if generation_config.temperature is not None and generation_config.temperature != 1.0:
            warpers.append(FlaxTemperatureLogitsWarper(generation_config.temperature))
        # 如果设置了 top_k 且不等于 0，则添加 top_k 调整器
        if generation_config.top_k is not None and generation_config.top_k != 0:
            warpers.append(FlaxTopKLogitsWarper(top_k=generation_config.top_k, min_tokens_to_keep=1))
        # 如果设置了 top_p 且小于 1.0，则添加 top_p 调整器
        if generation_config.top_p is not None and generation_config.top_p < 1.0:
            warpers.append(FlaxTopPLogitsWarper(top_p=generation_config.top_p, min_tokens_to_keep=1))

        # 返回配置好的 warpers 列表对象
        return warpers

    def _get_logits_processor(
        self,
        generation_config: GenerationConfig,
        input_ids_seq_length: int,
        logits_processor: Optional[FlaxLogitsProcessorList],
    ) -> FlaxLogitsProcessorList:
        """
        This method returns a [`FlaxLogitsProcessorList`] object containing all relevant
        [`FlaxLogitsProcessor`] instances used to modify the scores of the language model head.
        """
        processors = FlaxLogitsProcessorList()

        # Check if minimum length and end-of-sequence token ID are specified and valid
        if (
            generation_config.min_length is not None
            and generation_config.eos_token_id is not None
            and generation_config.min_length > -1
        ):
            # Append a processor to enforce minimum length and end token ID constraints
            processors.append(
                FlaxMinLengthLogitsProcessor(generation_config.min_length, generation_config.eos_token_id)
            )
        
        # Check if forced beginning-of-sequence token ID is specified
        if generation_config.forced_bos_token_id is not None:
            # Append a processor to force the beginning-of-sequence token ID
            processors.append(FlaxForcedBOSTokenLogitsProcessor(generation_config.forced_bos_token_id))
        
        # Check if forced end-of-sequence token ID is specified
        if generation_config.forced_eos_token_id is not None:
            # Append a processor to force the end-of-sequence token ID
            processors.append(
                FlaxForcedEOSTokenLogitsProcessor(generation_config.max_length, generation_config.forced_eos_token_id)
            )
        
        # Check if tokens to suppress are specified
        if generation_config.suppress_tokens is not None:
            # Append a processor to suppress specific tokens
            processors.append(FlaxSuppressTokensLogitsProcessor(generation_config.suppress_tokens))
        
        # Check if tokens to suppress at the beginning are specified
        if generation_config.begin_suppress_tokens is not None:
            begin_index = input_ids_seq_length
            
            # Adjust beginning index based on conditions
            begin_index = (
                begin_index
                if (input_ids_seq_length > 1 or generation_config.forced_bos_token_id is None)
                else begin_index + 1
            )
            
            # Adjust beginning index further based on forced decoder IDs
            if generation_config.forced_decoder_ids is not None and len(generation_config.forced_decoder_ids) > 0:
                begin_index += generation_config.forced_decoder_ids[-1][0]
            
            # Append a processor to suppress tokens at the beginning
            processors.append(
                FlaxSuppressTokensAtBeginLogitsProcessor(generation_config.begin_suppress_tokens, begin_index)
            )
        
        # Check if forced decoder IDs are specified
        if generation_config.forced_decoder_ids is not None:
            # Calculate adjusted IDs for forced tokens
            forced_decoder_ids = [
                [input_ids_seq_length + i[0] - 1, i[1]] for i in generation_config.forced_decoder_ids
            ]
            
            # Append a processor to force tokens based on adjusted IDs
            processors.append(FlaxForceTokensLogitsProcessor(forced_decoder_ids))
        
        # Merge the default processors list with any custom processors provided
        processors = self._merge_criteria_processor_list(processors, logits_processor)

        return processors

    def _merge_criteria_processor_list(
        self,
        default_list: FlaxLogitsProcessorList,
        custom_list: FlaxLogitsProcessorList,
        ) -> FlaxLogitsProcessorList:
        """
        This method merges a default list of logits processors with a custom list of logits processors.
        It returns a combined [`FlaxLogitsProcessorList`] object.
        """
    ) -> FlaxLogitsProcessorList:
        # 如果自定义列表为空，则直接返回默认列表
        if len(custom_list) == 0:
            return default_list
        # 遍历默认列表中的每个元素
        for default in default_list:
            # 遍历自定义列表中的每个元素
            for custom in custom_list:
                # 如果自定义元素的类型与默认元素相同
                if type(custom) is type(default):
                    # 确定对象类型为"logits processor"
                    object_type = "logits processor"
                    # 抛出值错误，说明已经创建了相同类型的自定义对象
                    raise ValueError(
                        f"A custom {object_type} of type {type(custom)} with values {custom} has been passed to"
                        f" `generate`, but it has already been created with the values {default}. {default} has been"
                        " created by passing the corresponding arguments to generate or by the model's config default"
                        f" values. If you just want to change the default values of {object_type} consider passing"
                        f" them as arguments to `generate` instead of using a custom {object_type}."
                    )
        # 将自定义列表中的元素追加到默认列表中
        default_list.extend(custom_list)
        # 返回合并后的默认列表
        return default_list

    def _greedy_search(
        self,
        input_ids: None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        logits_processor: Optional[FlaxLogitsProcessorList] = None,
        trace: bool = True,
        params: Optional[Dict[str, jnp.ndarray]] = None,
        model_kwargs: Optional[Dict[str, jnp.ndarray]] = None,
    def _sample(
        self,
        input_ids: None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        prng_key: Optional[jnp.ndarray] = None,
        logits_processor: Optional[FlaxLogitsProcessorList] = None,
        logits_warper: Optional[FlaxLogitsProcessorList] = None,
        trace: bool = True,
        params: Optional[Dict[str, jnp.ndarray]] = None,
        model_kwargs: Optional[Dict[str, jnp.ndarray]] = None,
    def _beam_search(
        self,
        input_ids: None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        length_penalty: Optional[float] = None,
        early_stopping: Optional[Union[bool, str]] = None,
        logits_processor: Optional[FlaxLogitsProcessorList] = None,
        trace: bool = True,
        params: Optional[Dict[str, jnp.ndarray]] = None,
        num_return_sequences: Optional[int] = None,
        model_kwargs: Optional[Dict[str, jnp.ndarray]] = None,
```