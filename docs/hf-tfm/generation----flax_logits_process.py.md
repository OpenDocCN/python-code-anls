# `.\generation\flax_logits_process.py`

```py
# coding=utf-8
# 导入inspect模块，用于检查和获取源代码信息
import inspect

# 导入JAX库
import jax
import jax.lax as lax
import jax.numpy as jnp

# 从上级目录中导入工具函数
from ..utils import add_start_docstrings
# 从日志记录工具中导入日志记录器
from ..utils.logging import get_logger

# 获取当前模块的日志记录器
logger = get_logger(__name__)

# 定义文档字符串常量，描述了logits处理器的输入和返回值
LOGITS_PROCESSOR_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`jnp.ndarray` of shape `(batch_size, sequence_length)`):
            输入序列标记的索引，形状为(batch_size, sequence_length)。

            可以使用[`PreTrainedTokenizer`]来获取索引。参见[`PreTrainedTokenizer.encode`]和
            [`PreTrainedTokenizer.__call__`]获取详情。

            [什么是输入ID？](../glossary#input-ids)
        scores (`jnp.ndarray` of shape `(batch_size, config.vocab_size)`):
            语言模型头的预测分数。当不使用beam搜索时，这些可以是每个词汇的logits；当使用beam搜索时，可以是
            每个词汇token的log softmax。
        kwargs (`Dict[str, Any]`, *optional*):
            特定于logits处理器的额外kwargs参数。

    Return:
        `jnp.ndarray` of shape `(batch_size, config.vocab_size)`: 处理后的预测分数。

"""

# 定义FlaxLogitsProcessor类，抽象基类，用于在生成过程中应用所有logits处理器
class FlaxLogitsProcessor:
    """用于生成过程中可以应用的所有logits处理器的抽象基类。"""

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: jnp.ndarray, scores: jnp.ndarray) -> jnp.ndarray:
        """处理logits的Flax方法。"""
        # 抛出未实现错误，提示该类为抽象类，只能通过继承该类的子类调用
        raise NotImplementedError(
            f"{self.__class__}是一个抽象类。只有继承了这个类的类才能被调用。"
        )

# 定义FlaxLogitsWarper类，抽象基类，用于在使用多项式采样的生成过程中应用所有logit变形器
class FlaxLogitsWarper:
    """用于使用多项式采样生成过程中可以应用的所有logit变形器的抽象基类。"""

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: jnp.ndarray, scores: jnp.ndarray) -> jnp.ndarray:
        """变形logits的Flax方法。"""
        # 抛出未实现错误，提示该类为抽象类，只能通过继承该类的子类调用
        raise NotImplementedError(
            f"{self.__class__}是一个抽象类。只有继承了这个类的类才能被调用。"
        )

# 定义FlaxLogitsProcessorList类，继承自list，用于创建一个[`FlaxLogitsProcessor`]或[`FlaxLogitsWarper`]列表，
# 并能够对输入的`scores`张量应用每一个处理器或变形器
class FlaxLogitsProcessorList(list):
    """
    此类可用于创建[`FlaxLogitsProcessor`]或[`FlaxLogitsWarper`]的列表，以随后处理`scores`输入张量。
    此类继承自列表，并添加了一个特定的*__call__*方法来应用每个[`FlaxLogitsProcessor`]或[`FlaxLogitsWarper`]到输入上。
    """
    """
    对象方法，根据给定的输入和参数处理逻辑
    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: jnp.ndarray, scores: jnp.ndarray, cur_len: int, **kwargs) -> jnp.ndarray:
        # 遍历每个处理器对象
        for processor in self:
            # 获取处理器的调用方法参数签名
            function_args = inspect.signature(processor.__call__).parameters
            # 如果参数个数大于3
            if len(function_args) > 3:
                # 检查是否所有所需的参数都在kwargs中
                if not all(arg in kwargs for arg in list(function_args.keys())[2:]):
                    # 如果有缺失参数，抛出数值错误异常
                    raise ValueError(
                        f"Make sure that all the required parameters: {list(function_args.keys())} for "
                        f"{processor.__class__} are passed to the logits processor."
                    )
                # 调用处理器的方法，传入输入数据、得分、当前长度和其他参数
                scores = processor(input_ids, scores, cur_len, **kwargs)
            else:
                # 如果参数个数不大于3，直接调用处理器的方法，传入输入数据、得分和当前长度
                scores = processor(input_ids, scores, cur_len)
        # 返回处理后的得分
        return scores
    ```
class FlaxTemperatureLogitsWarper(FlaxLogitsWarper):
    r"""
    [`FlaxLogitsWarper`] for temperature (exponential scaling output probability distribution).

    Args:
        temperature (`float`):
            The value used to module the logits distribution.
    """

    def __init__(self, temperature: float):
        # 检查温度参数是否为正浮点数，如果不是则抛出异常
        if not isinstance(temperature, float) or not (temperature > 0):
            raise ValueError(f"`temperature` has to be a strictly positive float, but is {temperature}")

        self.temperature = temperature

    def __call__(self, input_ids: jnp.ndarray, scores: jnp.ndarray, cur_len: int) -> jnp.ndarray:
        # 将得分按温度值缩放，用于温度调节输出概率分布
        scores = scores / self.temperature
        return scores


class FlaxTopPLogitsWarper(FlaxLogitsWarper):
    """
    [`FlaxLogitsWarper`] that performs top-p, i.e. restricting to top tokens summing to prob_cut_off <= prob_cut_off.

    Args:
        top_p (`float`):
            If set to < 1, only the smallest set of most probable tokens with probabilities that add up to `top_p` or
            higher are kept for generation.
        filter_value (`float`, *optional*, defaults to -inf):
            All filtered values will be set to this float value.
        min_tokens_to_keep (`int`, *optional*, defaults to 1):
            Minimum number of tokens that cannot be filtered.
    """

    def __init__(self, top_p: float, filter_value: float = -float("Inf"), min_tokens_to_keep: int = 1):
        # 检查 top_p 是否为介于 0 和 1 之间的浮点数，否则抛出异常
        if not isinstance(top_p, float) or (top_p < 0 or top_p > 1.0):
            raise ValueError(f"`top_p` has to be a float > 0 and < 1, but is {top_p}")
        # 检查 min_tokens_to_keep 是否为正整数，否则抛出异常
        if not isinstance(min_tokens_to_keep, int) or (min_tokens_to_keep < 1):
            raise ValueError(f"`min_tokens_to_keep` has to be a positive integer, but is {min_tokens_to_keep}")

        self.top_p = top_p
        self.filter_value = filter_value
        self.min_tokens_to_keep = min_tokens_to_keep

    def __call__(self, input_ids: jnp.ndarray, scores: jnp.ndarray, cur_len: int) -> jnp.ndarray:
        # 获取前 k 个最高得分和其对应的索引
        topk_scores, topk_indices = lax.top_k(scores, scores.shape[-1])

        # 创建一个与 scores 形状相同的数组，填充为 filter_value
        mask_scores = jnp.full_like(scores, self.filter_value)
        # 计算 softmax 后的累积概率
        cumulative_probs = jax.nn.softmax(topk_scores, axis=-1).cumsum(axis=-1)
        # 创建用于掩码的布尔数组，仅保留累积概率小于 top_p 的部分
        score_mask = cumulative_probs < self.top_p

        # 将累积概率大于 top_p 的位置移到 score_mask 中
        score_mask = jnp.roll(score_mask, 1)
        score_mask |= score_mask.at[:, 0].set(True)

        # 至少保留 min_tokens_to_keep 个 token
        score_mask = score_mask.at[:, : self.min_tokens_to_keep].set(True)

        # 根据 score_mask 选择相应的得分值或者 filter_value
        topk_next_scores = jnp.where(score_mask, topk_scores, mask_scores)
        # 按照 topk_indices 排序，获取排序后的最终得分
        next_scores = jax.lax.sort_key_val(topk_indices, topk_next_scores)[-1]

        return next_scores


class FlaxTopKLogitsWarper(FlaxLogitsWarper):
    r"""
    [`FlaxLogitsWarper`] that performs top-k, i.e. restricting to the k highest probability elements.
    Args:
        top_k (`int`):
            The number of highest probability vocabulary tokens to keep for top-k-filtering.
        filter_value (`float`, *optional*, defaults to -inf):
            All filtered values will be set to this float value.
        min_tokens_to_keep (`int`, *optional*, defaults to 1):
            Minimum number of tokens that cannot be filtered.
    """
    定义一个类，用于执行Top-K筛选操作，保留概率最高的词汇标记。

    def __init__(self, top_k: int, filter_value: float = -float("Inf"), min_tokens_to_keep: int = 1):
        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError(f"`top_k` has to be a strictly positive integer, but is {top_k}")

        # 初始化方法，设置Top-K值，并确保不小于最小保留标记数
        self.top_k = max(top_k, min_tokens_to_keep)
        self.filter_value = filter_value

    def __call__(self, input_ids: jnp.ndarray, scores: jnp.ndarray, cur_len: int) -> jnp.ndarray:
        # 调用实例时，执行Top-K筛选操作

        # 获取输入的批次大小和词汇表大小
        batch_size, vocab_size = scores.shape

        # 初始化一个数组，用来存储被过滤后的分数值，默认为filter_value
        next_scores_flat = jnp.full(batch_size * vocab_size, self.filter_value)

        # 确定实际的Top-K值，避免超过分数数组的长度
        topk = min(self.top_k, scores.shape[-1])

        # 使用JAX库中的top_k函数找到每个批次中前Top-K个分数及其对应的索引
        topk_scores, topk_indices = lax.top_k(scores, topk)

        # 计算扁平化后的索引偏移，以便在一维数组中正确设置Top-K分数
        shift = jnp.broadcast_to((jnp.arange(batch_size) * vocab_size)[:, None], (batch_size, topk)).flatten()
        topk_scores_flat = topk_scores.flatten()
        topk_indices_flat = topk_indices.flatten() + shift

        # 在next_scores_flat数组中设置Top-K分数值
        next_scores_flat = next_scores_flat.at[topk_indices_flat].set(topk_scores_flat)

        # 将扁平化后的数组重新形状为(batch_size, vocab_size)，得到最终的Top-K分数数组
        next_scores = next_scores_flat.reshape(batch_size, vocab_size)
        return next_scores
class FlaxForcedBOSTokenLogitsProcessor(FlaxLogitsProcessor):
    r"""
    [`FlaxLogitsProcessor`] that enforces the specified token as the first generated token.

    Args:
        bos_token_id (`int`):
            The id of the token to force as the first generated token.
    """

    def __init__(self, bos_token_id: int):
        self.bos_token_id = bos_token_id  # 初始化函数，保存要强制作为第一个生成token的token id

    def __call__(self, input_ids: jnp.ndarray, scores: jnp.ndarray, cur_len: int) -> jnp.ndarray:
        new_scores = jnp.full(scores.shape, -float("inf"))  # 创建一个形状与scores相同的全负无穷数组

        apply_penalty = 1 - jnp.bool_(cur_len - 1)  # 根据当前生成长度是否为0，决定是否应用惩罚

        scores = jnp.where(apply_penalty, new_scores.at[:, self.bos_token_id].set(0), scores)
        # 根据apply_penalty条件，将scores中对应bos_token_id列的值设置为0，其它位置不变

        return scores


class FlaxForcedEOSTokenLogitsProcessor(FlaxLogitsProcessor):
    r"""
    [`FlaxLogitsProcessor`] that enforces the specified token as the last generated token when `max_length` is reached.

    Args:
        max_length (`int`):
            The maximum length of the sequence to be generated.
        eos_token_id (`int`):
            The id of the token to force as the last generated token when `max_length` is reached.
    """

    def __init__(self, max_length: int, eos_token_id: int):
        self.max_length = max_length  # 初始化函数，保存最大生成长度
        self.eos_token_id = eos_token_id  # 初始化函数，保存要强制作为末尾生成token的token id

    def __call__(self, input_ids: jnp.ndarray, scores: jnp.ndarray, cur_len: int) -> jnp.ndarray:
        new_scores = jnp.full(scores.shape, -float("inf"))  # 创建一个形状与scores相同的全负无穷数组

        apply_penalty = 1 - jnp.bool_(cur_len - self.max_length + 1)
        # 根据当前生成长度是否为max_length，决定是否应用惩罚

        scores = jnp.where(apply_penalty, new_scores.at[:, self.eos_token_id].set(0), scores)
        # 根据apply_penalty条件，将scores中对应eos_token_id列的值设置为0，其它位置不变

        return scores


class FlaxMinLengthLogitsProcessor(FlaxLogitsProcessor):
    r"""
    [`FlaxLogitsProcessor`] enforcing a min-length by setting EOS probability to 0.

    Args:
        min_length (`int`):
            The minimum length below which the score of `eos_token_id` is set to `-float("Inf")`.
        eos_token_id (`int`):
            The id of the *end-of-sequence* token.
    """

    def __init__(self, min_length: int, eos_token_id: int):
        if not isinstance(min_length, int) or min_length < 0:
            raise ValueError(f"`min_length` has to be a positive integer, but is {min_length}")

        if not isinstance(eos_token_id, int) or eos_token_id < 0:
            raise ValueError(f"`eos_token_id` has to be a positive integer, but is {eos_token_id}")

        self.min_length = min_length  # 初始化函数，保存最小生成长度
        self.eos_token_id = eos_token_id  # 初始化函数，保存要设置其概率为负无穷的token id

    def __call__(self, input_ids: jnp.ndarray, scores: jnp.ndarray, cur_len: int) -> jnp.ndarray:
        # create boolean flag to decide if min length penalty should be applied
        apply_penalty = 1 - jnp.clip(cur_len - self.min_length, 0, 1)
        # 根据当前生成长度是否小于min_length，决定是否应用惩罚

        scores = jnp.where(apply_penalty, scores.at[:, self.eos_token_id].set(-float("inf")), scores)
        # 根据apply_penalty条件，将scores中对应eos_token_id列的值设置为负无穷，其它位置不变

        return scores


class FlaxSuppressTokensAtBeginLogitsProcessor(FlaxLogitsProcessor):
    r"""
    # 定义一个处理类 `FlaxLogitsProcessor`，用于在 `generate` 函数开始生成时抑制一组指定的 token。
    # 这应该确保在生成的开头，由 `begin_suppress_tokens` 定义的 token 不会被抽样到。

    Args:
        begin_suppress_tokens (`List[int]`):
            不抽样的 token 列表。
        begin_index (`int`):
            开始抑制 token 的索引位置。
    """

    class FlaxLogitsProcessor:
        def __init__(self, begin_suppress_tokens, begin_index):
            # 将输入的 begin_suppress_tokens 转换为列表
            self.begin_suppress_tokens = list(begin_suppress_tokens)
            # 设置开始抑制 token 的索引位置
            self.begin_index = begin_index

        def __call__(self, input_ids, scores, cur_len: int):
            # 根据当前生成长度 `cur_len` 和开始抑制的索引 `begin_index` 计算是否应用惩罚
            apply_penalty = 1 - jnp.bool_(cur_len - self.begin_index)

            # 根据应用的惩罚，将指定的 `begin_suppress_tokens` 的分数设置为负无穷大
            scores = jnp.where(apply_penalty, scores.at[:, self.begin_suppress_tokens].set(-float("inf")), scores)

            # 返回处理后的分数
            return scores
class FlaxSuppressTokensLogitsProcessor(FlaxLogitsProcessor):
    r"""
    [`FlaxLogitsProcessor`] suppressing a list of tokens at each decoding step. The processor will set their log probs
    to be `-inf` so they are not sampled.

    Args:
        suppress_tokens (`list`):
            Tokens to not sample.
    """

    def __init__(self, suppress_tokens: list):
        # 初始化方法，接收一个要抑制的token列表
        self.suppress_tokens = list(suppress_tokens)

    def __call__(self, input_ids: jnp.ndarray, scores: jnp.ndarray, cur_len: int) -> jnp.ndarray:
        # 在scores张量的指定位置设置为负无穷，以便在采样时不被选中
        scores = scores.at[..., self.suppress_tokens].set(-float("inf"))

        return scores


class FlaxForceTokensLogitsProcessor(FlaxLogitsProcessor):
    r"""
    [`FlaxLogitsProcessor`] that takes a list of pairs of integers which indicates a mapping from generation indices to
    token indices that will be forced before sampling. The processor will set their log probs to 0 and all other tokens
    to `-inf` so that they are sampled at their corresponding index.

    Args:
        force_token_map (`list`):
            Map giving token ids and indices where they will be forced to be sampled.
    """

    def __init__(self, force_token_map):
        # 将force_token_map转换为字典格式，并初始化一个强制token的数组以提高XLA的兼容性
        force_token_map = dict(force_token_map)
        force_token_array = jnp.ones((max(force_token_map.keys()) + 1), dtype=jnp.int32) * -1
        for index, token in force_token_map.items():
            if token is not None:
                force_token_array = force_token_array.at[index].set(token)
        self.force_token_array = jnp.int32(force_token_array)

    def __call__(self, input_ids: jnp.ndarray, scores: jnp.ndarray, cur_len: int) -> jnp.ndarray:
        def _force_token(generation_idx):
            # 根据generation_idx确定要强制采样的token，并更新scores张量
            batch_size = scores.shape[0]
            current_token = self.force_token_array[generation_idx]

            new_scores = jnp.ones_like(scores, dtype=scores.dtype) * -float("inf")
            updates = jnp.zeros((batch_size, 1), dtype=scores.dtype)
            new_scores = lax.dynamic_update_slice(new_scores, updates, (0, current_token))
            return new_scores

        # 使用lax.cond根据cur_len的值来决定是否进行token强制操作
        scores = lax.cond(
            cur_len >= self.force_token_array.shape[0],
            # 如果当前长度大于等于force_token_array的长度，则不进行强制操作
            lambda: scores,
            # 否则，根据force_token_array[cur_len]的值来判断是否强制采样特定token
            lambda: lax.cond(
                self.force_token_array[cur_len] >= 0,
                # 只有有效（非负）的token才会被强制采样
                lambda: _force_token(cur_len),
                # 否则不进行强制操作
                lambda: scores,
            ),
        )
        return scores
class FlaxWhisperTimeStampLogitsProcessor(FlaxLogitsProcessor):
    r"""
    Whisper specific Processor. This processor can be used to force a list of tokens. The processor will set their log
    probs to `inf` so that they are sampled at their corresponding index.

    Args:
        generate_config (`GenerateConfig`):
            The generate config used to generate the output. The following parameters are required:
                eos_token_id (`int`, *optional*, defaults to 50257):
                    The id of the *end-of-sequence* token.
                no_timestamps_token_id (`int`, *optional*, defaults to 50363):
                    The id of the `"<|notimestamps|>"` token.
                max_initial_timestamp_index (`int`, *optional*, defaults to 1):
                    Used to set the maximum value of the initial timestamp. This is used to prevent the model from
                    predicting timestamps that are too far in the future.
    """

    def __init__(self, generate_config, model_config, decoder_input_length):
        # 初始化方法，设置对象的初始属性
        self.eos_token_id = generate_config.eos_token_id
        self.no_timestamps_token_id = generate_config.no_timestamps_token_id
        # 设置时间戳开始的位置
        self.timestamp_begin = generate_config.no_timestamps_token_id + 1

        # 设置开始索引，考虑解码器输入长度
        self.begin_index = decoder_input_length + 1

        # 如果是多语言模型，为语言标记和任务标记预留空间
        if generate_config.is_multilingual:
            self.begin_index += 2
        
        # 如果生成配置有最大初始时间戳索引属性，使用该值；否则使用模型词汇表大小
        if hasattr(generate_config, "max_initial_timestamp_index"):
            self.max_initial_timestamp_index = generate_config.max_initial_timestamp_index
        else:
            self.max_initial_timestamp_index = model_config.vocab_size
        
        # 如果最大初始时间戳索引为 None，则设为模型词汇表大小
        if self.max_initial_timestamp_index is None:
            self.max_initial_timestamp_index = model_config.vocab_size
    def __call__(self, input_ids, scores, cur_len):
        # 将包含 self.no_timestamps_token_id 的列设为负无穷，这由 without_timestamps 处理
        scores = scores.at[:, self.no_timestamps_token_id].set(-float("inf"))

        def handle_pairs(input_ids_k, scores_k):
            # 判断前一个 token 是否为时间戳，如果是，则设置为 True，否则为 False
            last_was_timestamp = jnp.where((cur_len - self.begin_index) >= 1, True, False)
            last_was_timestamp = jnp.where(
                input_ids_k[cur_len - 1] >= self.timestamp_begin,
                True and last_was_timestamp,
                False,
            )

            # 判断倒数第二个 token 是否为时间戳，如果是，则设置为 True，否则为 False
            penultimate_was_timestamp = jnp.where((cur_len - self.begin_index) < 2, True, False)
            penultimate_was_timestamp = jnp.where(
                input_ids_k[cur_len - 2] >= self.timestamp_begin,
                True,
                penultimate_was_timestamp,
            )

            return jnp.where(
                last_was_timestamp,
                jnp.where(
                    penultimate_was_timestamp > 0,
                    scores_k.at[self.timestamp_begin :].set(-float("inf")),  # 如果倒数第二个是时间戳，则将时间戳之后的分数设为负无穷
                    scores_k.at[: self.eos_token_id].set(-float("inf")),  # 否则将句子结束符之前的分数设为负无穷
                ),
                scores_k,  # 如果前一个不是时间戳，则保持分数不变
            )

        # 对每对 (input_ids, scores) 应用 handle_pairs 函数
        scores = jax.vmap(handle_pairs)(input_ids, scores)

        # 判断是否应用最大初始时间戳策略
        apply_max_initial_timestamp = jnp.where(cur_len == self.begin_index, True, False)
        apply_max_initial_timestamp = jnp.where(
            self.max_initial_timestamp_index is not None,
            True and apply_max_initial_timestamp,
            False,
        )

        # 计算最大允许的时间戳
        last_allowed = self.timestamp_begin + self.max_initial_timestamp_index

        # 如果应用最大初始时间戳策略，则将分数矩阵中大于最大允许时间戳之后的分数设为负无穷
        scores = jnp.where(
            apply_max_initial_timestamp,
            scores.at[:, last_allowed + 1 :].set(-float("inf")),
            scores,
        )

        # 如果时间戳的概率总和超过其它 token 的概率总和，则将时间戳之前的分数设为负无穷
        logprobs = jax.nn.log_softmax(scores, axis=-1)

        def handle_cumulative_probs(logprobs_k, scores_k):
            timestamp_logprob = jax.nn.logsumexp(logprobs_k[self.timestamp_begin :], axis=-1)
            max_text_token_logprob = jnp.max(logprobs_k[: self.timestamp_begin])
            return jnp.where(
                timestamp_logprob > max_text_token_logprob,
                scores_k.at[: self.timestamp_begin].set(-float("inf")),  # 如果时间戳的概率总和高于其它 token，则将时间戳之前的分数设为负无穷
                scores_k,  # 否则保持分数不变
            )

        # 对每个 (logprobs, scores) 应用 handle_cumulative_probs 函数
        scores = jax.vmap(handle_cumulative_probs)(logprobs, scores)

        # 返回处理后的分数矩阵
        return scores
```