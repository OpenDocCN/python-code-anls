# `.\transformers\generation\flax_logits_process.py`

```
# 设置文件编码为 UTF-8
# 版权声明
# 根据 Apache 许可证 2.0 版本授权
# 除非符合许可证要求，否则不得使用此文件
# 您可以在以下网址获取许可证副本
#     http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据"原样"分发软件
# 没有任何形式的担保或条件，无论是明示的还是暗示的
# 请查看许可证以获取有关权限和限制的详细信息

# 导入模块
import inspect
import jax
import jax.lax as lax
import jax.numpy as jnp
# 从上级目录中导入工具函数
from ..utils import add_start_docstrings
from ..utils.logging import get_logger

# 获取日志记录器
logger = get_logger(__name__)

# 定义文档字符串，描述输入参数和返回值
LOGITS_PROCESSOR_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`jnp.ndarray` of shape `(batch_size, sequence_length)`):
            输入序列标记在词汇表中的索引。

            可以使用 [`PreTrainedTokenizer`] 获取索引。有关详细信息，请参阅 [`PreTrainedTokenizer.encode`] 和
            [`PreTrainedTokenizer.__call__`]。

            [什么是输入 ID？](../glossary#input-ids)
        scores (`jnp.ndarray` of shape `(batch_size, config.vocab_size)`):
            语言建模头的预测分数。当不使用波束搜索时，这些可以是每个词汇表的对数，当使用波束搜索时，可以是每个词汇表标记的对数 softmax
        kwargs (`Dict[str, Any]`, *optional*):
            额外的与 logits 处理器相关的关键字参数。

    Return:
        `jnp.ndarray` of shape `(batch_size, config.vocab_size)`: 处理后的预测分数。

"""

# 定义 FlaxLogitsProcessor 类，用于处理生成期间应用的所有 logits 处理器的抽象基类
class FlaxLogitsProcessor:
    """Abstract base class for all logit processors that can be applied during generation."""

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: jnp.ndarray, scores: jnp.ndarray) -> jnp.ndarray:
        """Flax method for processing logits."""
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can be called."
        )

# 定义 FlaxLogitsWarper 类，用于在使用多项式采样生成期间应用的所有 logits warper 的抽象基类
class FlaxLogitsWarper:
    """Abstract base class for all logit warpers that can be applied during generation with multinomial sampling."""

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: jnp.ndarray, scores: jnp.ndarray) -> jnp.ndarray:
        """Flax method for warping logits."""
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can be called."
        )

# 定义 FlaxLogitsProcessorList 类，用于创建 [`FlaxLogitsProcessor`] 或 [`FlaxLogitsWarper`] 的列表，以便随后处理 `scores` 输入张量
# 该类继承自列表，并添加了一个特定的 *__call__* 方法来应用每个 [`FlaxLogitsProcessor`] 或 [`FlaxLogitsWarper`] 到输入
class FlaxLogitsProcessorList(list):
    """
    This class can be used to create a list of [`FlaxLogitsProcessor`] or [`FlaxLogitsWarper`] to subsequently process
    a `scores` input tensor. This class inherits from list and adds a specific *__call__* method to apply each
    [`FlaxLogitsProcessor`] or [`FlaxLogitsWarper`] to the inputs.
    """
    使用输入的参数调用 LogitsProcessor 实例

    Args:
        input_ids: 输入的 token IDs，类型为 jnp.ndarray
        scores: 模型输出的分数，类型为 jnp.ndarray
        cur_len: 当前序列的长度，类型为 int
        **kwargs: 其它可能的参数

    Returns:
        jnp.ndarray: 处理后的分数

    Raises:
        ValueError: 如果缺少必要的参数则抛出异常
    """
    def __call__(self, input_ids: jnp.ndarray, scores: jnp.ndarray, cur_len: int, **kwargs) -> jnp.ndarray:
        # 遍历所有的 LogitsProcessor 实例
        for processor in self:
            # 获取每个处理器的调用方法的参数
            function_args = inspect.signature(processor.__call__).parameters
            # 如果参数个数大于3
            if len(function_args) > 3:
                # 检查是否所有必要参数都被传递了
                if not all(arg in kwargs for arg in list(function_args.keys())[2:]):
                    raise ValueError(
                        f"Make sure that all the required parameters: {list(function_args.keys())} for "
                        f"{processor.__class__} are passed to the logits processor."
                    )
                # 调用处理器并传递所有参数
                scores = processor(input_ids, scores, cur_len, **kwargs)
            else:
                # 调用处理器并传递部分参数
                scores = processor(input_ids, scores, cur_len)
        # 返回处理后的分数
        return scores
class FlaxTemperatureLogitsWarper(FlaxLogitsWarper):
    r"""
    [`FlaxLogitsWarper`] for temperature (exponential scaling output probability distribution).

    Args:
        temperature (`float`):
            The value used to module the logits distribution.
    """

    def __init__(self, temperature: float):
        # 检查温度值是否为正浮点数，若不是则引发 ValueError 异常
        if not isinstance(temperature, float) or not (temperature > 0):
            raise ValueError(f"`temperature` has to be a strictly positive float, but is {temperature}")

        # 设置温度属性
        self.temperature = temperature

    def __call__(self, input_ids: jnp.ndarray, scores: jnp.ndarray, cur_len: int) -> jnp.ndarray:
        # 通过温度对分数进行缩放
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
        # 检查 top_p 是否为 0 到 1 之间的浮点数，若不是则引发 ValueError 异常
        if not isinstance(top_p, float) or (top_p < 0 or top_p > 1.0):
            raise ValueError(f"`top_p` has to be a float > 0 and < 1, but is {top_p}")
        # 检查 min_tokens_to_keep 是否为正整数，若不是则引发 ValueError 异常
        if not isinstance(min_tokens_to_keep, int) or (min_tokens_to_keep < 1):
            raise ValueError(f"`min_tokens_to_keep` has to be a positive integer, but is {min_tokens_to_keep}")

        # 设置 top-p、过滤值和最小保留的令牌数属性
        self.top_p = top_p
        self.filter_value = filter_value
        self.min_tokens_to_keep = min_tokens_to_keep

    def __call__(self, input_ids: jnp.ndarray, scores: jnp.ndarray, cur_len: int) -> jnp.ndarray:
        # 获得分数最高的 k 个令牌及其索引
        topk_scores, topk_indices = lax.top_k(scores, scores.shape[-1])

        # 创建一个与 scores 形状相同的掩码数组，初始化为 filter_value
        mask_scores = jnp.full_like(scores, self.filter_value)
        # 计算 softmax 后的累积概率
        cumulative_probs = jax.nn.softmax(topk_scores, axis=-1).cumsum(axis=-1)
        # 根据 top-p 进行掩码筛选
        score_mask = cumulative_probs < self.top_p

        # 包括累积概率高于 top_p 的令牌
        score_mask = jnp.roll(score_mask, 1)
        score_mask |= score_mask.at[:, 0].set(True)

        # 最小保留的令牌数
        score_mask = score_mask.at[:, : self.min_tokens_to_keep].set(True)

        # 根据掩码确定下一个分数
        topk_next_scores = jnp.where(score_mask, topk_scores, mask_scores)
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
    
    # 初始化方法，用于设置top-k值、过滤值和最小保留token数
    def __init__(self, top_k: int, filter_value: float = -float("Inf"), min_tokens_to_keep: int = 1):
        # 检查top_k是否为正整数
        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError(f"`top_k` has to be a strictly positive integer, but is {top_k}")
        
        # 设置top_k值，确保不低于最小保留token数
        self.top_k = max(top_k, min_tokens_to_keep)
        # 设置过滤值
        self.filter_value = filter_value

    # 调用方法，用于对输入的ids和scores进行处理
    def __call__(self, input_ids: jnp.ndarray, scores: jnp.ndarray, cur_len: int) -> jnp.ndarray:
        # 获取batch大小和词汇表大小
        batch_size, vocab_size = scores.shape
        # 初始化一个数组，用于存储下一个scores的值，全部设为过滤值
        next_scores_flat = jnp.full(batch_size * vocab_size, self.filter_value)

        # 确保topk值不超过scores的大小
        topk = min(self.top_k, scores.shape[-1])  # Safety check
        # 获取topk的分数和索引
        topk_scores, topk_indices = lax.top_k(scores, topk)
        # 计算偏移，用于展平后的索引
        shift = jnp.broadcast_to((jnp.arange(batch_size) * vocab_size)[:, None], (batch_size, topk)).flatten()
        # 展平topk的分数和索引
        topk_scores_flat = topk_scores.flatten()
        topk_indices_flat = topk_indices.flatten() + shift

        # 将topk的分数赋值给next_scores_flat
        next_scores_flat = next_scores_flat.at[topk_indices_flat].set(topk_scores_flat)
        # 重新将展平后的数组转换成原始形状的数组
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
        # 初始化函数，设置初始属性值
        self.bos_token_id = bos_token_id

    def __call__(self, input_ids: jnp.ndarray, scores: jnp.ndarray, cur_len: int) -> jnp.ndarray:
        # 创建与输入分数数组相同形状的数组，所有值初始化为负无穷
        new_scores = jnp.full(scores.shape, -float("inf"))

        # 根据当前长度是否为1，决定是否应用惩罚
        apply_penalty = 1 - jnp.bool_(cur_len - 1)

        # 将新分数数组中第一维度、第bos_token_id列的值设为0，其他不变
        scores = jnp.where(apply_penalty, new_scores.at[:, self.bos_token_id].set(0), scores)

        # 返回处理后的分数数组
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
        # 初始化函数，设置初始属性值
        self.max_length = max_length
        self.eos_token_id = eos_token_id

    def __call__(self, input_ids: jnp.ndarray, scores: jnp.ndarray, cur_len: int) -> jnp.ndarray:
        # 创建与输入分数数组相同形状的数组，所有值初始化为负无穷
        new_scores = jnp.full(scores.shape, -float("inf"))

        # 根据当前长度是否接近最大长度，决定是否应用惩罚
        apply_penalty = 1 - jnp.bool_(cur_len - self.max_length + 1)

        # 将新分数数组中第一维度、第eos_token_id列的值设为0，其他不变
        scores = jnp.where(apply_penalty, new_scores.at[:, self.eos_token_id].set(0), scores)

        # 返回处理后的分数数组
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
        # 检查并设置最小长度和EOS标记的属性
        if not isinstance(min_length, int) or min_length < 0:
            raise ValueError(f"`min_length` has to be a positive integer, but is {min_length}")

        if not isinstance(eos_token_id, int) or eos_token_id < 0:
            raise ValueError(f"`eos_token_id` has to be a positive integer, but is {eos_token_id}")

        self.min_length = min_length
        self.eos_token_id = eos_token_id

    def __call__(self, input_ids: jnp.ndarray, scores: jnp.ndarray, cur_len: int) -> jnp.ndarray:
        # 创建布尔标志以决定是否应用最小长度惩罚
        apply_penalty = 1 - jnp.clip(cur_len - self.min_length, 0, 1)

        # 将分数数组中EOS标记的值设为负无穷，其他不变
        scores = jnp.where(apply_penalty, scores.at[:, self.eos_token_id].set(-float("inf")), scores)

        # 返回处理后的分数数组
        return scores


class FlaxSuppressTokensAtBeginLogitsProcessor(FlaxLogitsProcessor):
    r"""
    # FlaxLogitsProcessor 类用于在生成过程中，从生成开始就抑制一组标记，使用 begin_index 标记确定。
    # 这应该确保在生成的开始处不会采样到由 begin_suppress_tokens 定义的标记。
    
    class FlaxLogitsProcessor:
        # 初始化方法，接受需要抑制的标记列表 begin_suppress_tokens 和抑制开始的索引 begin_index
        def __init__(self, begin_suppress_tokens, begin_index):
            # 将传入的 begin_suppress_tokens 转换为列表形式
            self.begin_suppress_tokens = list(begin_suppress_tokens)
            # 设置抑制开始的索引
            self.begin_index = begin_index
    
        # 对象被调用时的方法，用于处理输入的标识符、分数以及当前长度
        def __call__(self, input_ids, scores, cur_len: int):
            # 计算是否应用抑制，cur_len - self.begin_index > 0 则为 1，否则为 0
            apply_penalty = 1 - jnp.bool_(cur_len - self.begin_index)
    
            # 将需要抑制的标记对应的分数设置为负无穷大，以实现抑制的效果
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
        # 初始化函数，接收要抑制的标记列表
        self.suppress_tokens = list(suppress_tokens)

    def __call__(self, input_ids: jnp.ndarray, scores: jnp.ndarray, cur_len: int) -> jnp.ndarray:
        # 在得分中将要抑制的标记的概率设置为负无穷
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
        # 初始化函数，将强制标记映射转换为数组以便在 XLA 中使用
        force_token_map = dict(force_token_map)
        force_token_array = jnp.ones((max(force_token_map.keys()) + 1), dtype=jnp.int32) * -1
        for index, token in force_token_map.items():
            if token is not None:
                force_token_array = force_token_array.at[index].set(token)
        self.force_token_array = jnp.int32(force_token_array)

    def __call__(self, input_ids: jnp.ndarray, scores: jnp.ndarray, cur_len: int) -> jnp.ndarray:
        def _force_token(generation_idx):
            # 强制特定标记的函数
            batch_size = scores.shape[0]
            current_token = self.force_token_array[generation_idx]

            new_scores = jnp.ones_like(scores, dtype=scores.dtype) * -float("inf")
            updates = jnp.zeros((batch_size, 1), dtype=scores.dtype)
            new_scores = lax.dynamic_update_slice(new_scores, updates, (0, current_token))
            return new_scores

        scores = lax.cond(
            cur_len >= self.force_token_array.shape[0],
            # 如果当前长度大于等于 force_token_array 的长度，则处理器不执行任何操作
            lambda: scores,
            # 否则，可能会强制某个标记
            lambda: lax.cond(
                self.force_token_array[cur_len] >= 0,
                # 仅强制有效（正数）标记
                lambda: _force_token(cur_len),
                # 否则，处理器不执行任何操作
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
        # 初始化对象属性
        self.eos_token_id = generate_config.eos_token_id
        self.no_timestamps_token_id = generate_config.no_timestamps_token_id
        self.timestamp_begin = generate_config.no_timestamps_token_id + 1

        self.begin_index = decoder_input_length + 1

        # 如果是多语言模型，需要额外的空间用于语言标记和任务标记
        if generate_config.is_multilingual:
            self.begin_index += 2
        # 设置最大初始时间戳索引，用于防止模型预测过远的时间戳
        if hasattr(generate_config, "max_initial_timestamp_index"):
            self.max_initial_timestamp_index = generate_config.max_initial_timestamp_index
        else:
            self.max_initial_timestamp_index = model_config.vocab_size
        # 如果最大初始时间戳索引为None，则设置为词汇表大小
        if self.max_initial_timestamp_index is None:
            self.max_initial_timestamp_index = model_config.vocab_size
    def __call__(self, input_ids, scores, cur_len):
        # suppress <|notimestamps|> which is handled by without_timestamps
        # 将 <|notimestamps|> 的分数设为负无穷，由 without_timestamps 处理
        scores = scores.at[:, self.no_timestamps_token_id].set(-float("inf"))

        def handle_pairs(input_ids_k, scores_k):
            # 判断上一个 token 是否为时间戳
            last_was_timestamp = jnp.where((cur_len - self.begin_index) >= 1, True, False)
            last_was_timestamp = jnp.where(
                input_ids_k[cur_len - 1] >= self.timestamp_begin,
                True and last_was_timestamp,
                False,
            )

            # 判断倒数第二个 token 是否为时间戳
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
                    scores_k.at[self.timestamp_begin :].set(-float("inf")),
                    scores_k.at[: self.eos_token_id].set(-float("inf")),
                ),
                scores_k,
            )

        # 处理每个 token 对应的分数
        scores = jax.vmap(handle_pairs)(input_ids, scores)

        apply_max_initial_timestamp = jnp.where(cur_len == self.begin_index, True, False)
        apply_max_initial_timestamp = jnp.where(
            self.max_initial_timestamp_index is not None,
            True and apply_max_initial_timestamp,
            False,
        )

        last_allowed = self.timestamp_begin + self.max_initial_timestamp_index

        scores = jnp.where(
            apply_max_initial_timestamp,
            scores.at[:, last_allowed + 1 :].set(-float("inf")),
            scores,
        )

        # if sum of probability over timestamps is above any other token, sample timestamp
        # 如果时间戳的概率之和高于其他 token，则采样时间戳
        logprobs = jax.nn.log_softmax(scores, axis=-1)

        def handle_cumulative_probs(logprobs_k, scores_k):
            # 计算时间戳的累积概率和最大文本 token 的概率
            timestamp_logprob = jax.nn.logsumexp(logprobs_k[self.timestamp_begin :], axis=-1)
            max_text_token_logprob = jnp.max(logprobs_k[: self.timestamp_begin])
            return jnp.where(
                timestamp_logprob > max_text_token_logprob,
                scores_k.at[: self.timestamp_begin].set(-float("inf")),
                scores_k,
            )

        # 处理每个 token 对应的累积概率
        scores = jax.vmap(handle_cumulative_probs)(logprobs, scores)

        return scores
```