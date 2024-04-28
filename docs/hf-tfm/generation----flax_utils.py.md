# `.\transformers\generation\flax_utils.py`

```
# 设置文件编码为 UTF-8
# 版权声明，版权归 Google AI Flax 团队和 HuggingFace Inc. 团队所有
# 版权声明，版权归 NVIDIA 公司所有
# 使用 Apache 许可证 2.0 版本
# 除非符合许可证，否则不得使用此文件
# 可以在以下网址获取许可证副本
# http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则按"原样"分发软件
# 不附带任何担保或条件，无论是明示的还是暗示的
# 请参阅许可证获取特定语言的权限

# 导入所需库
import copy  # 导入 copy 库，用于对象的深拷贝
import inspect  # 导入 inspect 库，用于获取对象信息
import warnings  # 导入 warnings 库，用于警告处理
from functools import partial  # 导入 functools 库的 partial 函数，用于创建可调用对象
from typing import Any, Dict, Optional, Union  # 导入 typing 库，用于类型提示

import flax  # 导入 flax 库，用于基于 JAX 的灵活深度学习
import jax  # 导入 jax 库，用于自动求导和并行计算
import jax.numpy as jnp  # 导入 jax 库中的 numpy 模块，用于数组操作
import numpy as np  # 导入 numpy 库，用于数值计算
from jax import lax  # 导入 jax 库中的 lax 模块，用于定义计算原语

# 导入 transformers 库中的相关模块
from ..models.auto import (
    FLAX_MODEL_FOR_CAUSAL_LM_MAPPING,  # 导入自动模型映射字典
    FLAX_MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING,  # 导入序列到序列自动模型映射字典
    FLAX_MODEL_FOR_VISION_2_SEQ_MAPPING,  # 导入视觉到序列自动模型映射字典
)
from ..utils import ModelOutput, logging  # 导入工具函数和日志记录模块
from .configuration_utils import GenerationConfig  # 导入生成配置类
from .flax_logits_process import (  # 导入 flax_logits_process 模块中的相关类
    FlaxForcedBOSTokenLogitsProcessor,  # 强制开始标记的逻辑处理器类
    FlaxForcedEOSTokenLogitsProcessor,  # 强制结束标记的逻辑处理器类
    FlaxForceTokensLogitsProcessor,  # 强制标记的逻辑处理器类
    FlaxLogitsProcessorList,  # 逻辑处理器列表类
    FlaxMinLengthLogitsProcessor,  # 最小长度的逻辑处理器类
    FlaxSuppressTokensAtBeginLogitsProcessor,  # 在开头抑制标记的逻辑处理器类
    FlaxSuppressTokensLogitsProcessor,  # 抑制标记的逻辑处理器类
    FlaxTemperatureLogitsWarper,  # 温度调节逻辑处理器类
    FlaxTopKLogitsWarper,  # 基于 Top-K 的逻辑处理器类
    FlaxTopPLogitsWarper,  # 基于 Top-P 的逻辑处理器类
)


# 获取日志记录器
logger = logging.get_logger(__name__)


# 定义基于贪婪搜索的解码器输出类
@flax.struct.dataclass
class FlaxGreedySearchOutput(ModelOutput):
    """
    Flax Base class for outputs of decoder-only generation models using greedy search.


    Args:
        sequences (`jnp.ndarray` of shape `(batch_size, max_length)`):
            The generated sequences.
    """

    sequences: jnp.ndarray = None


# 定义基于采样的解码器输出类
@flax.struct.dataclass
class FlaxSampleOutput(ModelOutput):
    """
    Flax Base class for outputs of decoder-only generation models using sampling.


    Args:
        sequences (`jnp.ndarray` of shape `(batch_size, max_length)`):
            The generated sequences.
    """

    sequences: jnp.ndarray = None


# 定义基于 Beam Search 的解码器输出类
@flax.struct.dataclass
class FlaxBeamSearchOutput(ModelOutput):
    """
    Flax Base class for outputs of decoder-only generation models using greedy search.


    Args:
        sequences (`jnp.ndarray` of shape `(batch_size, max_length)`):
            The generated sequences.
        scores (`jnp.ndarray` of shape `(batch_size,)`):
            The scores (log probabilities) of the generated sequences.
    """

    sequences: jnp.ndarray = None
    scores: jnp.ndarray = None


# 定义贪婪搜索状态类
@flax.struct.dataclass
class GreedyState:
    cur_len: jnp.ndarray  # 当前长度
    sequences: jnp.ndarray  # 生成的序列
    running_token: jnp.ndarray  # 运行标记
    is_sent_finished: jnp.ndarray  # 句子是否已完成
    model_kwargs: Dict[str, jnp.ndarray]  # 模型关键字参数


# 定义采样状态类
@flax.struct.dataclass
class SampleState:
    cur_len: jnp.ndarray  # 当前长度
    # 定义变量 sequences，类型为 jnp.ndarray，用于存储序列数据
    sequences: jnp.ndarray
    # 定义变量 running_token，类型为 jnp.ndarray，用于存储运行中的令牌数据
    running_token: jnp.ndarray
    # 定义变量 is_sent_finished，类型为 jnp.ndarray，用于标记句子是否已完成的数据
    is_sent_finished: jnp.ndarray
    # 定义变量 prng_key，类型为 jnp.ndarray，用于存储伪随机数生成器密钥数据
    prng_key: jnp.ndarray
    # 定义变量 model_kwargs，类型为 Dict[str, jnp.ndarray]，用于存储模型参数的字典，键为字符串，值为 jnp.ndarray 类型的数组
    model_kwargs: Dict[str, jnp.ndarray]
# 定义一个数据类 BeamSearchState，用于存储 Beam Search 过程中的状态信息
@flax.struct.dataclass
class BeamSearchState:
    cur_len: jnp.ndarray  # 当前长度
    running_sequences: jnp.ndarray  # 正在运行的序列
    running_scores: jnp.ndarray  # 正在运行的分数
    sequences: jnp.ndarray  # 序列
    scores: jnp.ndarray  # 分数
    is_sent_finished: jnp.ndarray  # 是否句子已完成
    model_kwargs: Dict[str, jnp.ndarray]  # 模型参数

# 定义一个混合类 FlaxGenerationMixin，包含自回归文本生成的所有函数，用作 [`FlaxPreTrainedModel`] 中的混合类
class FlaxGenerationMixin:
    """
    A class containing all functions for auto-regressive text generation, to be used as a mixin in
    [`FlaxPreTrainedModel`].

    The class exposes [`~generation.FlaxGenerationMixin.generate`], which can be used for:
            - *greedy decoding* by calling [`~generation.FlaxGenerationMixin._greedy_search`] if `num_beams=1` and
              `do_sample=False`
            - *multinomial sampling* by calling [`~generation.FlaxGenerationMixin._sample`] if `num_beams=1` and
              `do_sample=True`
            - *beam-search decoding* by calling [`~generation.FlaxGenerationMixin._beam_search`] if `num_beams>1` and
              `do_sample=False`

    You do not need to call any of the above methods directly. Pass custom parameter values to 'generate' instead. To
    learn more about decoding strategies refer to the [text generation strategies guide](../generation_strategies).
    """

    # 抛出 NotImplementedError 异常，提示子类需要定义一个 `prepare_inputs_for_generation` 方法以使用 `generate`
    def prepare_inputs_for_generation(self, *args, **kwargs):
        raise NotImplementedError(
            "A model class needs to define a `prepare_inputs_for_generation` method in order to use `generate`."
        )

    # 静态方法，用于在调试模式下运行生成过程
    @staticmethod
    def _run_loop_in_debug(cond_fn, body_fn, init_state):
        """
        Run generation in untraced mode. This should only be used for debugging purposes.
        """
        state = init_state
        while cond_fn(state):
            state = body_fn(state)
        return state

    # 准备用于生成的编码器-解码器参数
    def _prepare_encoder_decoder_kwargs_for_generation(self, input_ids, params, model_kwargs):
        encoder_kwargs = {
            argument: value
            for argument, value in model_kwargs.items()
            if not (argument.startswith("decoder_") or argument.startswith("cross_attn"))
        }
        model_kwargs["encoder_outputs"] = self.encode(input_ids, params=params, return_dict=True, **encoder_kwargs)
        return model_kwargs

    # 准备用于生成的解码器输入 ID
    def _prepare_decoder_input_ids_for_generation(
        self,
        batch_size: int,
        decoder_start_token_id: int = None,
        bos_token_id: int = None,
        model_kwargs: Optional[Dict[str, jnp.ndarray]] = None,
    ) -> jnp.ndarray:
        if model_kwargs is not None and "decoder_input_ids" in model_kwargs:
            # 如果 model_kwargs 中包含 "decoder_input_ids"，则返回该值
            decoder_input_ids = model_kwargs.pop("decoder_input_ids")
            if decoder_input_ids is not None:
                return decoder_input_ids
        # 获取解码器起始标记 ID，并返回 jnp 数组
        decoder_start_token_id = self._get_decoder_start_token_id(decoder_start_token_id, bos_token_id)
        return jnp.array(decoder_start_token_id, dtype="i4").reshape(1, -1).repeat(batch_size, axis=0)
    # 获取解码器起始标记的ID，用于编码器-解码器模型
    def _get_decoder_start_token_id(self, decoder_start_token_id: int = None, bos_token_id: int = None) -> int:
        # 如果提供了decoder_start_token_id，则使用该值，否则使用生成配置中的decoder_start_token_id
        decoder_start_token_id = (
            decoder_start_token_id
            if decoder_start_token_id is not None
            else self.generation_config.decoder_start_token_id
        )
        # 如果提供了bos_token_id，则使用该值，否则使用生成配置中的bos_token_id
        bos_token_id = bos_token_id if bos_token_id is not None else self.generation_config.bos_token_id
        # 如果decoder_start_token_id已定义，则返回其值
        if decoder_start_token_id is not None:
            return decoder_start_token_id
        # 如果配置中的decoder对象具有decoder_start_token_id属性且已定义，则返回其值
        elif (
            hasattr(self.config, "decoder")
            and hasattr(self.config.decoder, "decoder_start_token_id")
            and self.config.decoder.decoder_start_token_id is not None
        ):
            return self.config.decoder.decoder_start_token_id
        # 如果提供了bos_token_id，则返回其值
        elif bos_token_id is not None:
            return bos_token_id
        # 如果配置中的decoder对象具有bos_token_id属性且已定义，则返回其值
        elif (
            hasattr(self.config, "decoder")
            and hasattr(self.config.decoder, "bos_token_id")
            and self.config.decoder.bos_token_id is not None
        ):
            return self.config.decoder.bos_token_id
        # 如果以上情况均未满足，则引发值错误
        raise ValueError(
            "`decoder_start_token_id` or `bos_token_id` has to be defined for encoder-decoder generation."
        )

    # 将张量扩展到指定数量的beam上
    @staticmethod
    def _expand_to_num_beams(tensor, num_beams):
        return jnp.broadcast_to(tensor[:, None], (tensor.shape[0], num_beams) + tensor.shape[1:])

    # 适应logits以进行beam搜索的函数
    def _adapt_logits_for_beam_search(self, logits):
        """
        This function can be overwritten in the specific modeling_flax_<model-name>.py classes to allow for custom beam
        search behavior. Note that the only model that overwrites this method is [`~transformes.FlaxMarianMTModel`].
        """
        return logits
    def _validate_model_class(self):
        """
        Confirms that the model class is compatible with generation. If not, raises an exception that points to the
        right class to use.
        """
        # 检查当前模型类是否兼容生成。如果不兼容，则抛出一个异常，指向正确的类以使用。
        if not self.can_generate():
            # 生成兼容映射列表
            generate_compatible_mappings = [
                FLAX_MODEL_FOR_CAUSAL_LM_MAPPING,
                FLAX_MODEL_FOR_VISION_2_SEQ_MAPPING,
                FLAX_MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING,
            ]
            generate_compatible_classes = set()
            # 遍历兼容映射列表，获取支持当前配置的模型类
            for model_mapping in generate_compatible_mappings:
                supported_models = model_mapping.get(type(self.config), default=None)
                if supported_models is not None:
                    generate_compatible_classes.add(supported_models.__name__)
            # 构建异常信息
            exception_message = (
                f"The current model class ({self.__class__.__name__}) is not compatible with `.generate()`, as "
                "it doesn't have a language model head."
            )
            # 如果存在兼容的模型类，则在异常信息中添加相应提示
            if generate_compatible_classes:
                exception_message += f" Please use one of the following classes instead: {generate_compatible_classes}"
            # 抛出异常
            raise TypeError(exception_message)

    def _validate_model_kwargs(self, model_kwargs: Dict[str, Any]):
        """Validates model kwargs for generation. Generate argument typos will also be caught here."""
        # 验证用于生成的模型 kwargs。此处还将捕获生成参数的拼写错误。
        unused_model_args = []
        model_args = set(inspect.signature(self.prepare_inputs_for_generation).parameters)
        # `kwargs`/`model_kwargs` 通常用于处理可选的前向传递输入，如 `attention_mask`。如果 `prepare_inputs_for_generation`
        # 不接受它们，则可以进行更严格的检查。
        if "kwargs" in model_args or "model_kwargs" in model_args:
            model_args |= set(inspect.signature(self.__call__).parameters)
        # 遍历模型 kwargs，检查未使用的参数
        for key, value in model_kwargs.items():
            if value is not None and key not in model_args:
                unused_model_args.append(key)

        # 如果存在未使用的模型参数，则抛出 ValueError
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
    # 返回一个包含所有与多项式采样相关的 FlaxLogitsWarper 实例的 FlaxLogitsProcessorList 列表对象
    def _get_logits_warper(self, generation_config: GenerationConfig) -> FlaxLogitsProcessorList:
        """
        This class returns a [`FlaxLogitsProcessorList`] list object that contains all relevant [`FlaxLogitsWarper`]
        instances used for multinomial sampling.
        """
        # 创建一个空的 FlaxLogitsProcessorList 对象
        warpers = FlaxLogitsProcessorList()

        # 如果设置了温度且温度不为 1.0
        if generation_config.temperature is not None and generation_config.temperature != 1.0:
            # 添加温度处理器到 warpers 中
            warpers.append(FlaxTemperatureLogitsWarper(generation_config.temperature))
        # 如果设置了 top_k 且 top_k 不为 0
        if generation_config.top_k is not None and generation_config.top_k != 0:
            # 添加 top_k 处理器到 warpers 中
            warpers.append(FlaxTopKLogitsWarper(top_k=generation_config.top_k, min_tokens_to_keep=1))
        # 如果设置了 top_p 且 top_p 小于 1.0
        if generation_config.top_p is not None and generation_config.top_p < 1.0:
            # 添加 top_p 处理器到 warpers 中
            warpers.append(FlaxTopPLogitsWarper(top_p=generation_config.top_p, min_tokens_to_keep=1))

        # 返回处理器列表对象
        return warpers

    # 获取 logits 处理器
    def _get_logits_processor(
        self,
        generation_config: GenerationConfig,
        input_ids_seq_length: int,
        logits_processor: Optional[FlaxLogitsProcessorList],
    ) -> FlaxLogitsProcessorList:
        """
        This method constructs and returns a FlaxLogitsProcessorList object containing various FlaxLogitsProcessor instances,
        which are used to modify the scores of the language model head during generation.
        """
        # Initialize an empty list to hold FlaxLogitsProcessor instances
        processors = FlaxLogitsProcessorList()

        # Check if minimum length, end of sequence token ID, and minimum length constraint are specified
        if (
            generation_config.min_length is not None
            and generation_config.eos_token_id is not None
            and generation_config.min_length > -1
        ):
            # Append FlaxMinLengthLogitsProcessor instance to the list
            processors.append(
                FlaxMinLengthLogitsProcessor(generation_config.min_length, generation_config.eos_token_id)
            )
        # Check if forced beginning of sequence token ID is specified
        if generation_config.forced_bos_token_id is not None:
            # Append FlaxForcedBOSTokenLogitsProcessor instance to the list
            processors.append(FlaxForcedBOSTokenLogitsProcessor(generation_config.forced_bos_token_id))
        # Check if forced end of sequence token ID is specified
        if generation_config.forced_eos_token_id is not None:
            # Append FlaxForcedEOSTokenLogitsProcessor instance to the list
            processors.append(
                FlaxForcedEOSTokenLogitsProcessor(generation_config.max_length, generation_config.forced_eos_token_id)
            )
        # Check if tokens to suppress during generation are specified
        if generation_config.suppress_tokens is not None:
            # Append FlaxSuppressTokensLogitsProcessor instance to the list
            processors.append(FlaxSuppressTokensLogitsProcessor(generation_config.suppress_tokens))
        # Check if tokens to suppress at the beginning of generation are specified
        if generation_config.begin_suppress_tokens is not None:
            # Calculate the index from which to begin suppressing tokens
            begin_index = input_ids_seq_length
            begin_index = (
                begin_index
                if (input_ids_seq_length > 1 or generation_config.forced_bos_token_id is None)
                else begin_index + 1
            )
            if generation_config.forced_decoder_ids is not None and len(generation_config.forced_decoder_ids) > 0:
                # Generation starts after the last token that is forced
                begin_index += generation_config.forced_decoder_ids[-1][0]
            # Append FlaxSuppressTokensAtBeginLogitsProcessor instance to the list
            processors.append(
                FlaxSuppressTokensAtBeginLogitsProcessor(generation_config.begin_suppress_tokens, begin_index)
            )
        # Check if specific tokens are forced during generation
        if generation_config.forced_decoder_ids is not None:
            # Calculate the indices and corresponding token IDs for forced tokens
            forced_decoder_ids = [
                [input_ids_seq_length + i[0] - 1, i[1]] for i in generation_config.forced_decoder_ids
            ]
            # Append FlaxForceTokensLogitsProcessor instance to the list
            processors.append(FlaxForceTokensLogitsProcessor(forced_decoder_ids))
        # Merge default and custom processor lists
        processors = self._merge_criteria_processor_list(processors, logits_processor)

        # Return the final list of processors
        return processors

    def _merge_criteria_processor_list(
        self,
        default_list: FlaxLogitsProcessorList,
        custom_list: FlaxLogitsProcessorList,
    # 定义一个方法，用于将自定义的logits处理器列表与默认的logits处理器列表合并
    def _merge_custom_lists(
        self,
        custom_list: FlaxLogitsProcessorList,
        default_list: FlaxLogitsProcessorList,
    ) -> FlaxLogitsProcessorList:
        # 如果自定义列表为空，则直接返回默认列表
        if len(custom_list) == 0:
            return default_list
        # 遍历默认列表中的处理器
        for default in default_list:
            # 遍历自定义列表中的处理器
            for custom in custom_list:
                # 如果自定义处理器与默认处理器的类型相同
                if type(custom) is type(default):
                    object_type = "logits processor"
                    # 抛出值错误，提醒用户自定义处理器与默认处理器重复
                    raise ValueError(
                        f"A custom {object_type} of type {type(custom)} with values {custom} has been passed to"
                        f" `generate`, but it has already been created with the values {default}. {default} has been"
                        " created by passing the corresponding arguments to generate or by the model's config default"
                        f" values. If you just want to change the default values of {object_type} consider passing"
                        f" them as arguments to `generate` instead of using a custom {object_type}."
                    )
        # 将自定义列表中的处理器添加到默认列表中
        default_list.extend(custom_list)
        # 返回合并后的列表
        return default_list

    # 贪婪搜索方法
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
    # 随机采样方法
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
    # 束搜索方法
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
```