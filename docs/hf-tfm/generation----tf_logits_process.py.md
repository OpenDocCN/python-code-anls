# `.\transformers\generation\tf_logits_process.py`

```py
# 设置文件编码为 UTF-8
# 版权声明
# 根据 Apache 许可证 2.0 版本授权
# 除非符合许可证要求或书面同意，否则不得使用此文件
# 您可以在以下网址获取许可证副本
# http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则按“原样”分发软件
# 没有任何形式的担保或条件，无论是明示的还是暗示的
# 请查看许可证以获取特定语言的权限和限制

# 导入模块
import inspect
from typing import List, Tuple

import numpy as np
import tensorflow as tf

# 导入自定义模块
from ..tf_utils import stable_softmax
from ..utils import add_start_docstrings
from ..utils.logging import get_logger

# 获取日志记录器
logger = get_logger(__name__)

# 定义 TF_LOGITS_PROCESSOR_INPUTS_DOCSTRING 字符串
TF_LOGITS_PROCESSOR_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`tf.Tensor` of shape `(batch_size, sequence_length)`):
            输入序列标记在词汇表中的索引。

            可以使用 [`PreTrainedTokenizer`] 获取索引。有关详细信息，请参阅 [`PreTrainedTokenizer.encode`] 和
            [`PreTrainedTokenizer.__call__`]。

            [什么是输入 ID？](../glossary#input-ids)
        scores (`tf.Tensor` of shape `(batch_size, config.vocab_size)`):
            语言建模头的预测分数。当不使用波束搜索时，这些可以是每个词汇表的对数，当使用波束搜索时，可以是每个词汇表标记的对数 softmax。
        cur_len (`int`):
            有效输入序列标记的当前长度。在 TF 实现中，input_ids 的序列长度是生成器可以生成的最大长度，我们需要知道哪些标记是有效的。
        kwargs (`Dict[str, Any]`, *optional*):
            额外的 logits 处理器特定参数。

    Return:
        `tf.Tensor` of shape `(batch_size, config.vocab_size)`: 处理后的预测分数。
"""


class TFLogitsProcessor:
    """应用于生成过程中的所有 logits 处理器的抽象基类。"""

    @add_start_docstrings(TF_LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: tf.Tensor, scores: tf.Tensor, cur_len: int) -> tf.Tensor:
        """处理 logits 的 TF 方法。"""
        raise NotImplementedError(
            f"{self.__class__} 是一个抽象类。只有继承此类的类才能被调用。"
        )


class TFLogitsWarper:
    """应用于使用多项式采样生成过程中的所有 logits 包装器的抽象基类。"""

    @add_start_docstrings(TF_LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: tf.Tensor, scores: tf.Tensor, cur_len: int) -> tf.Tensor:
        """包装 logits 的 TF 方法。"""
        raise NotImplementedError(
            f"{self.__class__} 是一个抽象类。只有继承此类的类才能被调用。"
        )
# 定义一个继承自list的类TFLogitsProcessorList，用于创建[`TFLogitsProcessor`]列表以后处理`scores`输入张量
# 该类添加了一个特定的`__call__`方法，用于将每个[`TFLogitsProcessor`]应用于输入
class TFLogitsProcessorList(list):
    """
    This class can be used to create a list of [`TFLogitsProcessor`] to subsequently process a `scores` input tensor.
    This class inherits from list and adds a specific *__call__* method to apply each [`TFLogitsProcessor`] to the
    inputs.
    """

    # 使用装饰器`add_start_docstrings`添加文档字符串，描述TF_LOGITS_PROCESSOR_INPUTS_DOCSTRING
    @add_start_docstrings(TF_LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    # 定义`__call__`方法，接受输入参数`input_ids: tf.Tensor`, `scores: tf.Tensor`, `cur_len: int`和`**kwargs`，返回`tf.Tensor`
    def __call__(self, input_ids: tf.Tensor, scores: tf.Tensor, cur_len: int, **kwargs) -> tf.Tensor:
        # 遍历每个processor
        for processor in self:
            # 获取processor的`__call__`方法的参数
            function_args = inspect.signature(processor.__call__).parameters
            # 如果参数个数大于3
            if len(function_args) > 3:
                # 检查是否所有必需参数都传递给了logits processor
                if not all(arg in kwargs for arg in list(function_args.keys())[2:]):
                    raise ValueError(
                        f"Make sure that all the required parameters: {list(function_args.keys())} for "
                        f"{processor.__class__} are passed to the logits processor."
                    )
                # 调用processor的`__call__`方法
                scores = processor(input_ids, scores, cur_len, **kwargs)
            else:
                # 调用processor的`__call__`方法
                scores = processor(input_ids, scores, cur_len)
        # 返回处理后的scores
        return scores


# 定义一个继承自TFLogitsWarper的类TFTemperatureLogitsWarper，用于温度（指数缩放输出概率分布）
class TFTemperatureLogitsWarper(TFLogitsWarper):
    r"""
    [`TFLogitsWarper`] for temperature (exponential scaling output probability distribution).

    Args:
        temperature (`float`):
            The value used to module the logits distribution.
    """

    # 初始化方法，接受参数`temperature: float`
    def __init__(self, temperature: float):
        # 如果temperature不是float类型或不大于0，则抛出ValueError
        if not isinstance(temperature, float) or not (temperature > 0):
            raise ValueError(f"`temperature` has to be a strictly positive float, but is {temperature}")

        # 设��temperature属性
        self.temperature = temperature

    # 定义`__call__`方法，接受输入参数`input_ids: tf.Tensor`, `scores: tf.Tensor`, `cur_len: int`，返回`tf.Tensor`
    def __call__(self, input_ids: tf.Tensor, scores: tf.Tensor, cur_len: int) -> tf.Tensor:
        # 对scores进行温度缩放
        scores = scores / self.temperature
        # 返回处理后的scores
        return scores


# 定义一个继承自TFLogitsWarper的类TFTopKLogitsWarper，用于执行top-k，即限制保留前k个最高概率元素
class TFTopKLogitsWarper(TFLogitsWarper):
    r"""
    [`TFLogitsWarper`] that performs top-k, i.e. restricting to the k highest probability elements.

    Args:
        top_k (`int`):
            The number of highest probability vocabulary tokens to keep for top-k-filtering.
        filter_value (`float`, *optional*, defaults to -inf):
            All filtered values will be set to this float value.
        min_tokens_to_keep (`int`, *optional*, defaults to 1):
            Minimum number of tokens that cannot be filtered.
    """

    # 初始化方法，接受参数`top_k: int`, `filter_value: float = -float("Inf")`, `min_tokens_to_keep: int = 1`
    def __init__(self, top_k: int, filter_value: float = -float("Inf"), min_tokens_to_keep: int = 1):
        # 如果top_k不是正整数，则抛出ValueError
        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError(f"`top_k` has to be a strictly positive integer, but is {top_k}")

        # 设置top_k属性为max(top_k, min_tokens_to_keep)，设置filter_value属性
        self.top_k = max(top_k, min_tokens_to_keep)
        self.filter_value = filter_value
    # 定义一个类的调用方法，接受输入的token id张量、分数张量和当前长度，返回更新后的分数张量
    def __call__(self, input_ids: tf.Tensor, scores: tf.Tensor, cur_len: int) -> tf.Tensor:
        # 确保top_k不超过分数张量的最后一个维度大小
        top_k = min(self.top_k, scores.shape[-1])  # Safety check
        # 创建一个布尔掩码，包含所有概率小于top-k中最后一个token的token
        indices_to_remove = scores < tf.math.top_k(scores, k=top_k)[0][..., -1:]
        # 根据掩码，将小于top-k中最后一个token的token的分数替换为filter_value
        next_scores = tf.where(indices_to_remove, self.filter_value, scores)
        # 返回更新后的分数张量
        return next_scores
class TFTopPLogitsWarper(TFLogitsWarper):
    """
    [`TFLogitsWarper`] that performs top-p, i.e. restricting to top tokens summing to <= prob_cut_off.

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
        # 检查 top_p 是否为浮点数且在 0 到 1 之间
        if not isinstance(top_p, float) or (top_p < 0 or top_p > 1.0):
            raise ValueError(f"`top_p` has to be a float > 0 and < 1, but is {top_p}")
        # 检查 min_tokens_to_keep 是否为正整数
        if not isinstance(min_tokens_to_keep, int) or (min_tokens_to_keep < 1):
            raise ValueError(f"`min_tokens_to_keep` has to be a positive integer, but is {min_tokens_to_keep}")

        # 初始化对象的属性
        self.top_p = top_p
        self.filter_value = filter_value
        self.min_tokens_to_keep = min_tokens_to_keep

    def __call__(self, input_ids: tf.Tensor, scores: tf.Tensor, cur_len: int) -> tf.Tensor:
        # 获取最高的 k 个分数和对应的索引
        topk_scores, topk_indices = tf.math.top_k(scores, scores.shape[-1])

        # 创建一个与 scores 相同形状的张量，并填充为 filter_value
        mask_scores = tf.fill(scores.shape, self.filter_value)
        # 计算稳定的 softmax 函数，并计算累积概率
        cumulative_probs = tf.math.cumsum(stable_softmax(topk_scores, axis=-1), axis=-1)
        score_mask = cumulative_probs < self.top_p

        # 包括高于 top_p 的令牌（第一个 false = 左移并在左侧插入一个 True）
        score_mask = tf.concat((tf.ones([score_mask.shape[0], 1], dtype=tf.bool), score_mask[:, :-1]), axis=-1)

        # 确保保留最小令牌数
        score_mask = tf.concat(
            (
                tf.ones([score_mask.shape[0], self.min_tokens_to_keep], dtype=tf.bool),
                score_mask[:, self.min_tokens_to_keep :],
            ),
            axis=-1,
        )

        # 屏蔽不符合条件的值
        topk_next_scores = tf.where(score_mask, topk_scores, mask_scores)

        # 撤消 topk 排序：将每行原始索引的 2D 矩阵转换为包含原始分数坐标的形状为 (batch_size, vocab_size, 2) 的 3D 张量，
        # 从中我们可以散布（即 `scatter_indices[row, col, :]` 是包含 `[row, topk_indices[row, col]]` 的张量）
        scatter_rows = tf.tile(tf.expand_dims(tf.range(topk_indices.shape[0]), axis=-1), [1, topk_indices.shape[-1])
        scatter_indices = tf.stack((scatter_rows, topk_indices), axis=-1)
        next_scores = tf.scatter_nd(scatter_indices, topk_next_scores, shape=topk_next_scores.shape)

        return next_scores


class TFMinLengthLogitsProcessor(TFLogitsProcessor):
    r"""
    # TFLogitsProcessor类，用于根据最小长度设置EOS概率为0。
    class TFLogitsProcessor:
        # 初始化方法，接受最小长度和EOS标记的ID作为参数
        def __init__(self, min_length: int, eos_token_id: int):
            # 检查最小长度参数是否为正整数
            if not isinstance(min_length, int) or min_length < 0:
                raise ValueError(f"`min_length` has to be a positive integer, but is {min_length}")
    
            # 检查EOS标记ID参数是否为正整数
            if not isinstance(eos_token_id, int) or eos_token_id < 0:
                raise ValueError(f"`eos_token_id` has to be a positive integer, but is {eos_token_id}")
    
            # 将参数赋值给对象属性
            self.min_length = min_length
            self.eos_token_id = eos_token_id
    
        # 应用EOS标记屏蔽的内部方法，接受分数张量作为输入，返回处理后的分数张量
        def _apply_eos_token_mask(self, scores: tf.Tensor) -> tf.Tensor:
            # 创建一个布尔掩码，指示哪些位置是EOS标记的ID
            eos_token_id_mask = tf.range(scores.shape[-1]) == self.eos_token_id
            # 在EOS标记的位置将分数设为负无穷
            scores = tf.where(eos_token_id_mask, float("-inf"), scores)
            return scores
    
        # 调用方法，根据当前长度是否小于最小长度来决定是否应用EOS标记屏蔽
        def __call__(self, input_ids: tf.Tensor, scores: tf.Tensor, cur_len: int) -> tf.Tensor:
            # 如果当前长度小于最小长度，应用EOS标记屏蔽，否则保持原样
            scores = tf.cond(
                tf.less(cur_len, self.min_length),
                lambda: self._apply_eos_token_mask(scores),
                lambda: tf.identity(scores),
            )
            return scores
# 定义一个继承自 TFLogitsProcessor 的类 TFRepetitionPenaltyLogitsProcessor，
# 用于对重复序列施加指数惩罚

class TFRepetitionPenaltyLogitsProcessor(TFLogitsProcessor):
    """
    [`TFLogitsProcessor`] enforcing an exponential penalty on repeated sequences.

    Args:
        repetition_penalty (`float`):
            The parameter for repetition penalty. 1.0 means no penalty. See [this
            paper](https://arxiv.org/pdf/1909.05858.pdf) for more details.
    """

    # 定义初始化方法，接受 penalty 参数
    def __init__(self, penalty: float):
        # 检查 penalty 是否为正浮点数，若不是则抛出 ValueError 异常
        if not isinstance(penalty, float) or not (penalty > 0):
            raise ValueError(f"`penalty` has to be a strictly positive float, but is {penalty}")

        self.penalty = penalty

    # 创建评分惩罚方法，接受 input_ids 和 logits 作为输入，返回 token_penalties
    def _create_score_penalties(self, input_ids: tf.Tensor, logits: tf.Tensor) -> tf.Tensor:
        # 我们希望在 `input_ids` 的位置上填入惩罚值。由于 XLA 无法处理在运行时前不知道形状的情况，
        # 因此无法使用 `tf.unique`。因此，当给定行多次具有相同标记时，可能会有冗余的更新。

        # 收集要应用的惩罚
        logit_penalties = tf.gather(logits, input_ids, axis=1, batch_dims=1)
        logit_penalties = tf.where(logit_penalties > 0, 1 / self.penalty, logit_penalties)
        logit_penalties = tf.where(logit_penalties < 0, self.penalty, logit_penalties)

        # 散射惩罚
        token_penalties = tf.ones(logits.shape)
        batch_size = input_ids.shape[0]
        seq_len = tf.shape(input_ids)[1]  # 序列长度具有动态大小，因此是动态形状
        indexable_prev_input_ids = tf.concat(
            (
                tf.expand_dims(tf.repeat(tf.range(batch_size), seq_len), axis=-1),
                tf.expand_dims(tf.reshape(input_ids, [-1]), axis=-1),
            ),
            axis=1,
        )
        token_penalties = tf.tensor_scatter_nd_update(
            token_penalties, indices=indexable_prev_input_ids, updates=tf.reshape(logit_penalties, [-1])
        )
        return token_penalties

    # 调用方法，接受 input_ids、scores 和 cur_len 作为输入，返回处理后的 scores
    def __call__(self, input_ids: tf.Tensor, scores: tf.Tensor, cur_len: int) -> tf.Tensor:
        # 创建得分惩罚
        score_penalties = self._create_score_penalties(input_ids[:, :cur_len], scores)

        # 将 scores 与 score_penalties 相乘
        scores = tf.math.multiply(scores, score_penalties)

        return scores


# 定义一个继承自 TFLogitsProcessor 的类 TFNoBadWordsLogitsProcessor，
# 用于确保指定的序列永远不会被采样
class TFNoBadWordsLogitsProcessor(TFLogitsProcessor):
    """
    [`TFLogitsProcessor`] that enforces that specified sequences will never be sampled.
    """
    Args:
        bad_words_ids (`List[List[int]]`):
            不允许生成的标记 id 的列表的列表。为了获取不应出现在生成文本中的单词的标记，请确保在初始化分词器时设置 `add_prefix_space=True`，并使用 `tokenizer(bad_words, add_special_tokens=False).input_ids`。`add_prefix_space` 参数仅支持某些慢速分词器，因为快速分词器的前缀行为来自 `pre tokenizers`。在此处阅读更多信息：https://huggingface.co/docs/tokenizers/api/pre-tokenizers。
        eos_token_id (`int`):
            *end-of-sequence* 标记的 id。
    """

    def __init__(self, bad_words_ids: List[List[int]], eos_token_id: int):
        如果 `bad_words_ids` 不是列表类型或长度为 0，则引发 ValueError 异常
        if not isinstance(bad_words_ids, List) or len(bad_words_ids) == 0:
            raise ValueError(f"`bad_words_ids` has to be a non-empty list, but is {bad_words_ids}.")
        如果 `bad_words_ids` 中有任何元素不是列表类型，则引发 ValueError 异常
        if any(not isinstance(bad_word_ids, list) for bad_word_ids in bad_words_ids):
            raise ValueError(f"`bad_words_ids` has to be a list of lists, but is {bad_words_ids}.")
        如果 `bad_words_ids` 中有任何元素不是正整数或小于 0，则引发 ValueError 异常
        if any(
            any((not isinstance(token_id, (int, np.integer)) or token_id < 0) for token_id in bad_word_ids)
            for bad_word_ids in bad_words_ids
        ):
            raise ValueError(
                f"Each list in `bad_words_ids` has to be a list of positive integers, but is {bad_words_ids}."
            )

        # 存储有关不良单词的信息，使用三个张量：
        # 1. 一个带有禁止序列的矩形张量（用 `-1` 填充），用于完整数据比较
        self.bad_word_seqs_ids = tf.ragged.constant(bad_words_ids).to_tensor(default_value=-1)
        # 2. 一个包含每个禁止序列未填充长度的张量，用于快速长度比较
        bad_word_seqs_len = [len(bad_words) for bad_words in bad_words_ids]
        如果任何一个单词长度为 0，则引发 ValueError 异常
        if any(word_len == 0 for word_len in bad_word_seqs_len):
            raise ValueError(f"Banned words token sequences {bad_words_ids} cannot have an empty list")
        self.bad_word_seqs_len = tf.convert_to_tensor(bad_word_seqs_len, dtype=tf.int32)
        # 3. 一个包含每个序列的最后一个标记的张量，用于轻松访问可能被禁止的标记
        self.seq_forbidden_tokens = tf.convert_to_tensor([bad_words[-1] for bad_words in bad_words_ids])
    # 计算行中被禁止的不良标记
    def _calc_row_banned_bad_tokens(self, row_input_ids: tf.Tensor) -> tf.Tensor:
        # 定义一个内部函数，用于检查标记是否匹配不良标记序列
        def _tokens_match(bad_word_seq_number):
            # 定义一个内部函数，用于处理只有一个标记的情况
            def _len_one():
                # 如果不良序列只有一个标记，则始终将其屏蔽
                return tf.cond(
                    tf.math.equal(self.bad_word_seqs_len[bad_word_seq_number], 1),
                    lambda: tf.ones((), dtype=tf.bool),
                    _len_greater_than_cur_len,
                )

            # 定义一个内部函数，处理不良序列长度大于当前长度的情况
            def _len_greater_than_cur_len():
                # 否则，如果不良序列比当前长度长，则永远不会匹配
                return tf.cond(
                    tf.math.greater(self.bad_word_seqs_len[bad_word_seq_number], tf.shape(row_input_ids)[0]),
                    lambda: tf.zeros((), dtype=tf.bool),
                    _match_found,
                )

            # 定义一个内部函数，处理找到匹配的情况
            def _match_found():
                # 最后，运行实际的比较。只有在前面的比较未产生结果时才能调用它（否则会出现索引异常）
                compare_len = self.bad_word_seqs_len[bad_word_seq_number] - 1
                return tf.cond(
                    tf.math.reduce_all(
                        tf.math.equal(
                            row_input_ids[-compare_len:], self.bad_word_seqs_ids[bad_word_seq_number, :compare_len]
                        )
                    ),
                    lambda: tf.ones((), dtype=tf.bool),
                    lambda: tf.zeros((), dtype=tf.bool),
                )

            match = _len_one()  # 调用_len_one函数
            return match

        # 使用map_fn对当前行与所有不良序列进行比较，得到一个匹配的掩码
        match_mask = tf.map_fn(_tokens_match, tf.range(self.bad_word_seqs_ids.shape[0]), fn_output_signature=tf.bool)
        # 根据匹配掩码获取行中被禁止的标记
        row_banned_tokens = self.seq_forbidden_tokens[match_mask]
        return row_banned_tokens
    # 定义一个类方法，接受输入的 input_ids（tf.Tensor）、scores（tf.Tensor）和当前长度 cur_len（int），返回更新后的 scores（tf.Tensor）
    def __call__(self, input_ids: tf.Tensor, scores: tf.Tensor, cur_len: int) -> tf.Tensor:
        # 我们希望在得分级别上屏蔽一些禁止的标记。由于禁止的标记取决于先前的 `input_ids`，它们可能对每一行有不同的长度，甚至对某些行为空。
        # 为了保持简单和与 XLA 兼容，我们以每行的方式进行操作。
        # TODO（Joao）：这个函数可能会在 `cur_len` 增加时触发 XLA 重追踪。如果它成为频繁的瓶颈，请修复它。（将 `cur_len` 设为张量？）
        def _get_row_updated_score(row_inputs: Tuple[tf.Tensor]) -> tf.Tensor:
            # 从输入元组中获取行的 input_ids 和 scores
            row_input_ids, row_score = row_inputs
            # 计算当前行的被禁止的坏标记
            banned_tokens = self._calc_row_banned_bad_tokens(row_input_ids[:cur_len])
            # 创建一个被禁止标记的掩码
            banned_tokens_mask = tf.scatter_nd(
                indices=tf.expand_dims(banned_tokens, axis=-1),
                updates=tf.ones_like(banned_tokens, dtype=tf.bool),
                shape=row_score.shape,
            )
            # 根据被禁止标记的掩码，将对应位置的得分设为负无穷
            row_score = tf.where(banned_tokens_mask, -float("inf"), row_score)
            return row_score

        # 使用 tf.map_fn 对每行调用 _get_row_updated_score 函数，更新得分
        scores = tf.map_fn(_get_row_updated_score, (input_ids, scores), fn_output_signature=tf.float32)
        # 返回更新后的 scores
        return scores
# 定义一个继承自 TFLogitsProcessor 的类 TFNoRepeatNGramLogitsProcessor，用于处理 logits，以确保不出现重复的 n-grams。
# 详情可参考 Fairseq 中对应实现的注释，链接：https://github.com/pytorch/fairseq/blob/a07cb6f40480928c9e0548b737aadd36ee66ac76/fairseq/sequence_generator.py#L345
class TFNoRepeatNGramLogitsProcessor(TFLogitsProcessor):
    def __init__(self, ngram_size: int):
        # 检查 ngram_size 是否为正整数，否则抛出 ValueError 异常
        if not isinstance(ngram_size, int) or ngram_size <= 0:
            raise ValueError(f"`ngram_size` has to be a strictly positive integer, but is {ngram_size}")
        # 初始化 ngram_size 属性
        self.ngram_size = ngram_size

    def calc_banned_ngram_tokens(self, input_ids, num_hypos, cur_len):
        # 从 fairseq 中复制用于在 beam_search 中处理 no_repeat_ngram 的方法
        if cur_len + 1 < self.ngram_size:
            # 如果当前长度还未达到 ngram_size，则返回空列表作为禁用的 token
            return [[] for _ in range(num_hypos)]
        # 初始化一个列表，用于存储已生成的 ngram
        generated_ngrams = [{} for _ in range(num_hypos)]
        # 获取前一个 token 序列
        prev_input_ids = input_ids[:, :cur_len]
        # 遍历每个候选的索引
        for idx in range(num_hypos):
            # 将前面已生成的 token 序列转换为列表
            gen_tokens = prev_input_ids[idx].numpy().tolist()
            # 初始化一个字典，用于存储已生成的 ngram
            generated_ngram = generated_ngrams[idx]
            # 遍历生成当前 token 前的 ngram
            for ngram in zip(*[gen_tokens[i:] for i in range(self.ngram_size)]):
                # 获取前 n-1 个 token 组成的 tuple
                prev_ngram_tuple = tuple(ngram[:-1])
                # 将当前 n-gram 添加到已生成的 n-gram 中
                generated_ngram[prev_ngram_tuple] = generated_ngram.get(prev_ngram_tuple, []) + [ngram[-1]]

        def _get_generated_ngrams(hypo_idx):
            # 在解码下一个 token 之前，防止已出现的 ngram 再次出现
            start_idx = cur_len + 1 - self.ngram_size
            ngram_idx = tuple(prev_input_ids[hypo_idx, start_idx:cur_len].numpy().tolist())
            return generated_ngrams[hypo_idx].get(ngram_idx, [])

        # 获取禁用的 token 列表
        banned_tokens = [_get_generated_ngrams(hypo_idx) for hypo_idx in range(num_hypos)]

        return banned_tokens

    def __call__(self, input_ids: tf.Tensor, scores: tf.Tensor, cur_len: int) -> tf.Tensor:
        # TODO (joao): enable XLA on this logits processor. See discussion and attempts in
        # https://github.com/huggingface/transformers/pull/16974
        # 如果不是 eager 执行模式，则抛出 NotImplementedError 异常
        if not tf.executing_eagerly():
            raise NotImplementedError("TFNoRepeatNGramLogitsProcessor is only implemented for eager execution.")

        # 获取 batch_size 和 vocab_size
        batch_size, vocab_size = scores.shape
        # 计算禁用的 n-gram token
        banned_tokens = self.calc_banned_ngram_tokens(input_ids, batch_size, cur_len)

        # 创建禁用的 token 的布尔掩码
        banned_tokens_indices_mask = []
        for banned_tokens_slice in banned_tokens:
            banned_tokens_indices_mask.append(
                [True if token in banned_tokens_slice else False for token in range(vocab_size)]
            )

        # 将布尔掩码应用于 scores，将禁用的 token 对应的 logits 设置为负无穷
        scores = tf.where(tf.convert_to_tensor(banned_tokens_indices_mask, dtype=tf.bool), -float("inf"), scores)

        return scores


class TFForcedBOSTokenLogitsProcessor(TFLogitsProcessor):
    r"""
    # 强制指定的 token 作为生成的第一个 token 的 TFLogitsProcessor
    # 参数：
    #   bos_token_id (`int`):
    #       强制作为第一个生成 token 的 token id。
    #
    class TFLogitsProcessor:
        # 初始化方法
        def __init__(self, bos_token_id: int):
            # 如果 bos_token_id 小于 0，则引发 ValueError 异常
            if bos_token_id < 0:
                raise ValueError(f"The forced bos token id must be a non-negative integer, got {bos_token_id}")
            # 将参数中的 bos_token_id 赋值给对象的属性
            self.bos_token_id = bos_token_id
        
        # 调用对象时执行的方法
        def __call__(self, input_ids: tf.Tensor, scores: tf.Tensor, cur_len: int) -> tf.Tensor:
            # 如果当前长度为 1
            if cur_len == 1:
                # 获取 batch 大小和 token 数量
                batch_size, num_tokens = scores.shape
                # 将 bos_token_id 列的分数设置为 0
                scores = tf.zeros((batch_size, 1))
                # 在其他位置将分数设置为负无穷
                if self.bos_token_id > 0:
                    scores = tf.concat((tf.broadcast_to(-float("inf"), (batch_size, self.bos_token_id)), scores), axis=-1)
                if self.bos_token_id < (num_tokens - 1):
                    scores = tf.concat(
                        (scores, tf.broadcast_to(-float("inf"), (batch_size, (num_tokens - 1) - self.bos_token_id))),
                        axis=-1,
                    )
            # 返回更新后的分数
            return scores
class TFForcedEOSTokenLogitsProcessor(TFLogitsProcessor):
    r"""
    [`TFLogitsProcessor`] that enforces the specified token as the last generated token when `max_length` is reached.

    Args:
        max_length (`int`):
            The maximum length of the sequence to be generated.
        eos_token_id (`int`):
            The id of the token to force as the last generated token when `max_length` is reached.
    """

    def __init__(self, max_length: int, eos_token_id: int):
        # 初始化函数，设置最大长度和强制结束标记的id
        self.max_length = max_length
        # 如果强制结束标记的id小于0，抛出异常
        if eos_token_id < 0:
            raise ValueError(f"The forced eos token id must be a non-negative integer, got {eos_token_id}")
        self.eos_token_id = eos_token_id

    def __call__(self, input_ids: tf.Tensor, scores: tf.Tensor, cur_len: int) -> tf.Tensor:
        # 当当前长度等于最大长度减1时
        if cur_len == self.max_length - 1:
            batch_size, num_tokens = scores.shape
            # 将结束标记的列的分数设置为0
            scores = tf.zeros((batch_size, 1))
            # 在其他位置将分数设置为负无穷
            if self.eos_token_id > 0:
                scores = tf.concat((tf.broadcast_to(-float("inf"), (batch_size, self.eos_token_id)), scores), axis=-1)
            if self.eos_token_id < (num_tokens - 1):
                scores = tf.concat(
                    (scores, tf.broadcast_to(-float("inf"), (batch_size, (num_tokens - 1) - self.eos_token_id))),
                    axis=-1,
                )
        return scores


class TFSuppressTokensAtBeginLogitsProcessor(TFLogitsProcessor):
    r"""
    [`TFSuppressTokensAtBeginLogitsProcessor`] suppresses a list of tokens as soon as the `generate` function starts
    generating using `begin_index` tokens. This should ensure that the tokens defined by `begin_suppress_tokens` at not
    sampled at the begining of the generation.
    """

    def __init__(self, begin_suppress_tokens, begin_index):
        # 初始化函数，设置开始抑制的标记列表和开始索引
        self.begin_suppress_tokens = list(begin_suppress_tokens)
        self.begin_index = begin_index

    def __call__(self, input_ids: tf.Tensor, scores: tf.Tensor, cur_len: int) -> tf.Tensor:
        # 使用tf.cond条件语句，当当前长度等于开始索引时
        scores = tf.cond(
            tf.equal(cur_len, self.begin_index),
            # 更新分数，将开始抑制的标记位置的分数设置为负无穷
            lambda: tf.tensor_scatter_nd_update(
                scores,
                indices=[[i, token] for i in range(scores.shape[0]) for token in self.begin_suppress_tokens],
                updates=[-float("inf") for _ in range(scores.shape[0] * len(self.begin_suppress_tokens))],
            ),
            lambda: scores,
        )
        return scores


class TFSuppressTokensLogitsProcessor(TFLogitsProcessor):
    r"""This processor can be used to suppress a list of tokens. The processor will set their log probs to `-inf` so that they
    are not sampled."""

    def __init__(self, suppress_tokens):
        # 初始化函数，设置要抑制的标记列表
        self.suppress_tokens = list(suppress_tokens)
    # 定义类的调用方法，接受输入的input_ids张量、分数张量scores、当前长度cur_len，并返回更新后的分数张量
    def __call__(self, input_ids: tf.Tensor, scores: tf.Tensor, cur_len: int) -> tf.Tensor:
        # 使用tf.tensor_scatter_nd_update函数对scores张量进行更新
        scores = tf.tensor_scatter_nd_update(
            # 要更新的张量为scores
            scores,
            # 更新的索引为二维列表，包含每个位置i和每个位置对应的要被抑制的token
            indices=[[i, token] for i in range(scores.shape[0]) for token in self.suppress_tokens],
            # 更新的值为负无穷大，对应于抑制的token位置
            updates=[-float("inf") for _ in range(scores.shape[0] * len(self.suppress_tokens))],
        )
        # 返回更新后的scores张量
        return scores
class TFForceTokensLogitsProcessor(TFLogitsProcessor):
    r"""This processor takes a list of pairs of integers which indicates a mapping from generation indices to token
    indices that will be forced before sampling. The processor will set their log probs to `0` and all other tokens to
    `-inf` so that they are sampled at their corresponding index."""

    def __init__(self, force_token_map: List[List[int]]):
        # 将强制token映射转换为字典形式，格式为 {index: token}
        force_token_map = dict(force_token_map)
        # 将含有强制token的字典转换为数组形式，其中数组索引对应着要强制的token的索引，以便于 XLA 兼容性。
        # 没有强制token的索引将会有负值。
        force_token_array = np.ones((max(force_token_map.keys()) + 1), dtype=np.int32) * -1
        for index, token in force_token_map.items():
            if token is not None:
                force_token_array[index] = token
        # 将数组转换为 TensorFlow 张量
        self.force_token_array = tf.convert_to_tensor(force_token_array, dtype=tf.int32)

    def __call__(self, input_ids: tf.Tensor, scores: tf.Tensor, cur_len: int) -> tf.Tensor:
        def _force_token(generation_idx):
            # 获取批量大小
            batch_size = scores.shape[0]
            # 获取当前要强制的token
            current_token = self.force_token_array[generation_idx]

            # 创建与 scores 相同形状的新得分张量，全部初始化为负无穷
            new_scores = tf.ones_like(scores, dtype=scores.dtype) * -float("inf")
            # 创建索引张量，用于更新 scores 中需要强制的token的位置
            indices = tf.stack((tf.range(batch_size), tf.tile([current_token], [batch_size])), axis=1)
            # 创建更新张量，将需要强制的token位置的得分更新为0
            updates = tf.zeros((batch_size,), dtype=scores.dtype)
            # 使用 tf.tensor_scatter_nd_update 函数更新得分张量
            new_scores = tf.tensor_scatter_nd_update(new_scores, indices, updates)
            return new_scores

        # 使用 tf.cond 条件语句根据当前长度决定是否需要强制 token
        scores = tf.cond(
            tf.greater_equal(cur_len, tf.shape(self.force_token_array)[0]),
            # 如果当前长度大于等于强制token数组的长度，则不做任何操作
            lambda: tf.identity(scores),
            # 否则，可能需要强制某个特定的token
            lambda: tf.cond(
                tf.greater_equal(self.force_token_array[cur_len], 0),
                # 只有有效（正数）的token才会被强制
                lambda: _force_token(cur_len),
                # 否则，不做任何操作
                lambda: scores,
            ),
        )
        return scores
```