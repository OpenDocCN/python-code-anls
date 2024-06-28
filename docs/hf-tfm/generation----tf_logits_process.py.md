# `.\generation\tf_logits_process.py`

```py
# 导入模块inspect用于检查对象，并从typing导入List和Tuple
import inspect
from typing import List, Tuple

# 导入NumPy和TensorFlow库
import numpy as np
import tensorflow as tf

# 从上级目录的tf_utils模块导入stable_softmax函数
from ..tf_utils import stable_softmax
# 从上级目录的utils模块导入add_start_docstrings函数
from ..utils import add_start_docstrings
# 从utils.logging模块导入get_logger函数
from ..utils.logging import get_logger

# 使用get_logger函数获取当前模块的日志记录器
logger = get_logger(__name__)

# TF_LOGITS_PROCESSOR_INPUTS_DOCSTRING是一个原始字符串，描述了TFLogitsProcessor和TFLogitsWarper类中__call__方法的参数和返回值
TF_LOGITS_PROCESSOR_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`tf.Tensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using [`PreTrainedTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        scores (`tf.Tensor` of shape `(batch_size, config.vocab_size)`):
            Prediction scores of a language modeling head. These can be logits for each vocabulary when not using beam
            search or log softmax for each vocabulary token when using beam search.
        cur_len (`int`):
            The current length of valid input sequence tokens. In the TF implementation, the input_ids' sequence length
            is the maximum length generate can produce, and we need to know which of its tokens are valid.
        kwargs (`Dict[str, Any]`, *optional*):
            Additional logits processor specific kwargs.

    Return:
        `tf.Tensor` of shape `(batch_size, config.vocab_size)`: The processed prediction scores.
"""

# TFLogitsProcessor类定义了一个抽象基类，用于在生成过程中应用的所有logit处理器
class TFLogitsProcessor:
    """Abstract base class for all logit processors that can be applied during generation."""

    # 使用add_start_docstrings装饰器，添加了TF_LOGITS_PROCESSOR_INPUTS_DOCSTRING作为文档字符串
    @add_start_docstrings(TF_LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: tf.Tensor, scores: tf.Tensor, cur_len: int) -> tf.Tensor:
        """TF method for processing logits."""
        # 抛出未实现错误，提示该类是抽象类，只能由继承它的类调用
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can be called."
        )

# TFLogitsWarper类定义了一个抽象基类，用于在生成过程中使用多项式抽样时应用的所有logit包装器
class TFLogitsWarper:
    """Abstract base class for all logit warpers that can be applied during generation with multinomial sampling."""

    # 使用add_start_docstrings装饰器，添加了TF_LOGITS_PROCESSOR_INPUTS_DOCSTRING作为文档字符串
    @add_start_docstrings(TF_LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: tf.Tensor, scores: tf.Tensor, cur_len: int) -> tf.Tensor:
        """TF method for warping logits."""
        # 抛出未实现错误，提示该类是抽象类，只能由继承它的类调用
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can be called."
        )
# 定义一个继承自列表的类 `TFLogitsProcessorList`，用于存储一组 `TFLogitsProcessor` 对象，以便后续处理输入张量 `scores`。
# 该类添加了特定的 `__call__` 方法，用于对每个 `TFLogitsProcessor` 对象应用处理。
class TFLogitsProcessorList(list):
    
    # 使用装饰器 `add_start_docstrings` 应用输入参数的文档字符串 `TF_LOGITS_PROCESSOR_INPUTS_DOCSTRING`
    @add_start_docstrings(TF_LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: tf.Tensor, scores: tf.Tensor, cur_len: int, **kwargs) -> tf.Tensor:
        # 遍历列表中的每个处理器 `processor`
        for processor in self:
            # 检索处理器 `processor` 的调用方法的参数列表
            function_args = inspect.signature(processor.__call__).parameters
            # 如果参数个数超过 3
            if len(function_args) > 3:
                # 检查是否传递了所有必需的参数到 `processor` 的调用方法
                if not all(arg in kwargs for arg in list(function_args.keys())[2:]):
                    raise ValueError(
                        f"Make sure that all the required parameters: {list(function_args.keys())} for "
                        f"{processor.__class__} are passed to the logits processor."
                    )
                # 调用 `processor` 的方法，并更新 `scores`
                scores = processor(input_ids, scores, cur_len, **kwargs)
            else:
                # 否则，调用 `processor` 的方法，并更新 `scores`
                scores = processor(input_ids, scores, cur_len)
        # 返回处理后的 `scores`
        return scores


# 定义一个继承自 `TFLogitsWarper` 的类 `TFTemperatureLogitsWarper`
# 用于温度调节（指数缩放输出概率分布）的 `TFLogitsWarper`
class TFTemperatureLogitsWarper(TFLogitsWarper):
    
    # 初始化方法，接受一个 `temperature` 参数作为温度值
    def __init__(self, temperature: float):
        # 如果 `temperature` 不是 `float` 类型或者不是严格正数，则抛出 `ValueError`
        if not isinstance(temperature, float) or not (temperature > 0):
            raise ValueError(f"`temperature` has to be a strictly positive float, but is {temperature}")
        
        # 将 `temperature` 赋值给实例变量 `self.temperature`
        self.temperature = temperature
    
    # 调用方法，接受 `input_ids`、`scores`、`cur_len` 参数，返回处理后的 `scores`
    def __call__(self, input_ids: tf.Tensor, scores: tf.Tensor, cur_len: int) -> tf.Tensor:
        # 将 `scores` 按 `self.temperature` 进行缩放处理
        scores = scores / self.temperature
        # 返回处理后的 `scores`
        return scores


# 定义一个继承自 `TFLogitsWarper` 的类 `TFTopKLogitsWarper`
# 用于进行 top-k 操作的 `TFLogitsWarper`，即保留概率最高的 `top_k` 个元素
class TFTopKLogitsWarper(TFLogitsWarper):
    
    # 初始化方法，接受 `top_k`、`filter_value`（可选，默认为 `-inf`）、`min_tokens_to_keep`（可选，默认为 `1`）三个参数
    def __init__(self, top_k: int, filter_value: float = -float("Inf"), min_tokens_to_keep: int = 1):
        # 如果 `top_k` 不是正整数，则抛出 `ValueError`
        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError(f"`top_k` has to be a strictly positive integer, but is {top_k}")
        
        # 将 `top_k` 和 `min_tokens_to_keep` 中的最大值赋值给实例变量 `self.top_k`
        self.top_k = max(top_k, min_tokens_to_keep)
        # 将 `filter_value` 赋值给实例变量 `self.filter_value`
        self.filter_value = filter_value
    def __call__(self, input_ids: tf.Tensor, scores: tf.Tensor, cur_len: int) -> tf.Tensor:
        # 计算实际需要考虑的top_k值，确保不超过scores张量的最后一个维度的大小
        top_k = min(self.top_k, scores.shape[-1])  # Safety check
        
        # 创建一个布尔遮罩，标记所有概率小于top-k中最后一个概率的token
        indices_to_remove = scores < tf.math.top_k(scores, k=top_k)[0][..., -1:]
        
        # 根据遮罩，将需要移除的token对应的分数替换为过滤值self.filter_value
        next_scores = tf.where(indices_to_remove, self.filter_value, scores)
        
        # 返回更新后的分数张量
        return next_scores
    # `TFLogitsWarper`的子类，执行top-p截断，即限制保留加起来小于等于prob_cut_off的前几个最有可能的token。

    Args:
        top_p (`float`):
            如果设置为小于1的值，则只保留概率相加达到`top_p`或更高的最有可能的token用于生成。
        filter_value (`float`, *optional*, 默认为负无穷):
            所有被过滤的值将被设置为这个浮点数值。
        min_tokens_to_keep (`int`, *optional*, 默认为1):
            不能被过滤掉的最小token数目。
    """

    def __init__(self, top_p: float, filter_value: float = -float("Inf"), min_tokens_to_keep: int = 1):
        # 检查top_p是否为浮点数且在0到1之间
        if not isinstance(top_p, float) or (top_p < 0 or top_p > 1.0):
            raise ValueError(f"`top_p`必须是一个大于0且小于1的浮点数，当前值为{top_p}")
        # 检查min_tokens_to_keep是否为正整数
        if not isinstance(min_tokens_to_keep, int) or (min_tokens_to_keep < 1):
            raise ValueError(f"`min_tokens_to_keep`必须是一个正整数，当前值为{min_tokens_to_keep}")

        # 初始化实例变量
        self.top_p = top_p
        self.filter_value = filter_value
        self.min_tokens_to_keep = min_tokens_to_keep

    def __call__(self, input_ids: tf.Tensor, scores: tf.Tensor, cur_len: int) -> tf.Tensor:
        # 获取前k个最高分数和对应的索引
        topk_scores, topk_indices = tf.math.top_k(scores, scores.shape[-1])

        # 创建与scores相同形状的填充值为filter_value的张量
        mask_scores = tf.fill(scores.shape, self.filter_value)
        # 计算topk_scores的稳定softmax，并累积概率
        cumulative_probs = tf.math.cumsum(stable_softmax(topk_scores, axis=-1), axis=-1)
        # 创建一个布尔掩码，标记累积概率小于top_p的位置
        score_mask = cumulative_probs < self.top_p

        # 将第一个false替换为true，确保包含大于top_p的token
        score_mask = tf.concat((tf.ones([score_mask.shape[0], 1], dtype=tf.bool), score_mask[:, :-1]), axis=-1)

        # 确保保留至少min_tokens_to_keep个token
        score_mask = tf.concat(
            (
                tf.ones([score_mask.shape[0], self.min_tokens_to_keep], dtype=tf.bool),
                score_mask[:, self.min_tokens_to_keep:],
            ),
            axis=-1,
        )

        # 根据掩码将不符合条件的值设为filter_value
        topk_next_scores = tf.where(score_mask, topk_scores, mask_scores)

        # 恢复topk排序的顺序：将原始索引位置重新分散到张量中
        scatter_rows = tf.tile(tf.expand_dims(tf.range(topk_indices.shape[0]), axis=-1), [1, topk_indices.shape[-1]])
        scatter_indices = tf.stack((scatter_rows, topk_indices), axis=-1)
        next_scores = tf.scatter_nd(scatter_indices, topk_next_scores, shape=topk_next_scores.shape)

        return next_scores
    # 定义一个 TFLogitsProcessor 类，用于处理 logits（预测得分），实现通过设置 EOS 概率为 0 来强制最小长度。

    Args:
        min_length (`int`):
            最小长度，低于此长度时，`eos_token_id` 的得分被设置为 `-float("Inf")`。
        eos_token_id (`int`):
            *end-of-sequence*（EOS）标记的 id。
    """

    def __init__(self, min_length: int, eos_token_id: int):
        # 检查并设置 `min_length` 参数，必须为正整数
        if not isinstance(min_length, int) or min_length < 0:
            raise ValueError(f"`min_length` 必须是正整数，但其值为 {min_length}")

        # 检查并设置 `eos_token_id` 参数，必须为正整数
        if not isinstance(eos_token_id, int) or eos_token_id < 0:
            raise ValueError(f"`eos_token_id` 必须是正整数，但其值为 {eos_token_id}")

        # 初始化对象的属性
        self.min_length = min_length
        self.eos_token_id = eos_token_id

    def _apply_eos_token_mask(self, scores: tf.Tensor) -> tf.Tensor:
        # 创建一个掩码，标记出 scores 中等于 eos_token_id 的位置
        eos_token_id_mask = tf.range(scores.shape[-1]) == self.eos_token_id
        # 使用 tf.where 函数将 eos_token_id 的位置对应的 scores 设置为 -inf
        scores = tf.where(eos_token_id_mask, float("-inf"), scores)
        return scores

    def __call__(self, input_ids: tf.Tensor, scores: tf.Tensor, cur_len: int) -> tf.Tensor:
        # 如果当前长度 cur_len 小于 min_length，则应用 eos token 掩码
        scores = tf.cond(
            tf.less(cur_len, self.min_length),
            lambda: self._apply_eos_token_mask(scores),
            lambda: tf.identity(scores),
        )
        return scores
class TFRepetitionPenaltyLogitsProcessor(TFLogitsProcessor):
    r"""
    [`TFLogitsProcessor`] enforcing an exponential penalty on repeated sequences.

    Args:
        repetition_penalty (`float`):
            The parameter for repetition penalty. 1.0 means no penalty. See [this
            paper](https://arxiv.org/pdf/1909.05858.pdf) for more details.
    """

    def __init__(self, penalty: float):
        # 检查 penalty 参数是否为正浮点数，若不是则抛出 ValueError 异常
        if not isinstance(penalty, float) or not (penalty > 0):
            raise ValueError(f"`penalty` has to be a strictly positive float, but is {penalty}")

        self.penalty = penalty

    def _create_score_penalties(self, input_ids: tf.Tensor, logits: tf.Tensor) -> tf.Tensor:
        # 我们希望在 `input_ids` 的位置上填充惩罚值。由于 XLA 不能处理运行时未知的形状，
        # 不能使用 `tf.unique`。因此，当给定行中的同一标记出现多次时，可能会有冗余更新。

        # 收集要应用的惩罚值
        logit_penalties = tf.gather(logits, input_ids, axis=1, batch_dims=1)
        logit_penalties = tf.where(logit_penalties > 0, 1 / self.penalty, logit_penalties)
        logit_penalties = tf.where(logit_penalties < 0, self.penalty, logit_penalties)

        # 分散惩罚值
        token_penalties = tf.ones(logits.shape)
        batch_size = input_ids.shape[0]
        seq_len = tf.shape(input_ids)[1]  # 序列长度具有动态大小，因此使用动态形状
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

    def __call__(self, input_ids: tf.Tensor, scores: tf.Tensor, cur_len: int) -> tf.Tensor:
        # 创建分数惩罚
        score_penalties = self._create_score_penalties(input_ids[:, :cur_len], scores)

        # 将分数乘以相应的惩罚值
        scores = tf.math.multiply(scores, score_penalties)

        return scores


class TFNoBadWordsLogitsProcessor(TFLogitsProcessor):
    """
    [`TFLogitsProcessor`] that enforces that specified sequences will never be sampled.
    """
    Args:
        bad_words_ids (`List[List[int]]`):
            不允许生成的令牌 ID 列表的列表。为了获取不应出现在生成文本中的词汇的令牌，请确保在初始化分词器时设置 `add_prefix_space=True`，并使用 `tokenizer(bad_words, add_special_tokens=False).input_ids` 来获取这些词汇的令牌 ID 列表。对于某些较慢的分词器，`add_prefix_space` 参数是支持的，因为快速分词器的前缀行为来自于 `pre tokenizers`。详细信息请参阅 [这里](https://huggingface.co/docs/tokenizers/api/pre-tokenizers)。
        eos_token_id (`int`):
            *end-of-sequence*（EOS）令牌的 ID。
    """

    def __init__(self, bad_words_ids: List[List[int]], eos_token_id: int):
        # 检查 `bad_words_ids` 是否为列表且非空
        if not isinstance(bad_words_ids, List) or len(bad_words_ids) == 0:
            raise ValueError(f"`bad_words_ids` 必须是非空列表，当前为 {bad_words_ids}。")
        # 检查 `bad_words_ids` 中的每个元素是否为列表
        if any(not isinstance(bad_word_ids, list) for bad_word_ids in bad_words_ids):
            raise ValueError(f"`bad_words_ids` 必须是列表的列表，当前为 {bad_words_ids}。")
        # 检查 `bad_words_ids` 中的每个元素是否为正整数列表
        if any(
            any((not isinstance(token_id, (int, np.integer)) or token_id < 0) for token_id in bad_word_ids)
            for bad_word_ids in bad_words_ids
        ):
            raise ValueError(
                f"`bad_words_ids` 中的每个列表必须是正整数列表，当前为 {bad_words_ids}。"
            )

        # 存储关于不允许的词汇的信息，使用三个张量：
        # 1. 一个矩形张量，包含禁止序列（用 `-1` 填充），用于完整数据比较
        self.bad_word_seqs_ids = tf.ragged.constant(bad_words_ids).to_tensor(default_value=-1)
        # 2. 一个张量，包含每个禁止序列的未填充长度，用于快速长度比较
        bad_word_seqs_len = [len(bad_words) for bad_words in bad_words_ids]
        # 检查禁止词汇序列的长度是否为零
        if any(word_len == 0 for word_len in bad_word_seqs_len):
            raise ValueError(f"禁止词汇序列 {bad_words_ids} 不能包含空列表")
        self.bad_word_seqs_len = tf.convert_to_tensor(bad_word_seqs_len, dtype=tf.int32)
        # 3. 一个张量，包含每个序列的最后一个令牌，便于访问可能被禁止的令牌
        self.seq_forbidden_tokens = tf.convert_to_tensor([bad_words[-1] for bad_words in bad_words_ids])
    def _calc_row_banned_bad_tokens(self, row_input_ids: tf.Tensor) -> tf.Tensor:
        def _tokens_match(bad_word_seq_number):
            def _len_one():
                # 如果坏序列只有一个标记，则始终屏蔽它
                return tf.cond(
                    tf.math.equal(self.bad_word_seqs_len[bad_word_seq_number], 1),
                    lambda: tf.ones((), dtype=tf.bool),
                    _len_greater_than_cur_len,
                )

            def _len_greater_than_cur_len():
                # 否则，如果坏序列比当前长度长，它们永远不会匹配
                return tf.cond(
                    tf.math.greater(self.bad_word_seqs_len[bad_word_seq_number], tf.shape(row_input_ids)[0]),
                    lambda: tf.zeros((), dtype=tf.bool),
                    _match_found,
                )

            def _match_found():
                # 最后，执行实际的比较。只有在之前的比较没有结果时才能调用（否则会导致索引异常）
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

            match = _len_one()
            return match

        # 将当前行与所有坏词序列进行比较，获取匹配的掩码
        match_mask = tf.map_fn(_tokens_match, tf.range(self.bad_word_seqs_ids.shape[0]), fn_output_signature=tf.bool)
        row_banned_tokens = self.seq_forbidden_tokens[match_mask]
        return row_banned_tokens
        # 定义一个调用函数，接受输入的 `input_ids`（Tensor 类型）、分数 `scores`（Tensor 类型）、当前长度 `cur_len`（整数类型），返回更新后的分数 `scores`（Tensor 类型）
        # 我们希望在分数级别上屏蔽一些被禁止的令牌。由于被禁止的令牌取决于前一个 `input_ids`，它们可能对每一行具有不同的长度，甚至对某些行来说可能为空。
        # 为了保持简单并与 XLA 兼容，我们以逐行的方式进行操作。
        # TODO（Joao）：这个函数可能会因为 `cur_len` 的增加而触发 XLA 重追踪。如果这成为频繁的瓶颈，请修复它。（将 `cur_len` 设为一个张量？）
        def _get_row_updated_score(row_inputs: Tuple[tf.Tensor]) -> tf.Tensor:
            # 获取当前行的输入 `row_input_ids` 和分数 `row_score`
            row_input_ids, row_score = row_inputs
            # 计算当前行被禁止的坏令牌列表，基于 `row_input_ids` 的前 `cur_len` 部分
            banned_tokens = self._calc_row_banned_bad_tokens(row_input_ids[:cur_len])
            # 创建一个布尔类型的张量，表示被禁止的令牌的位置，其形状与 `row_score` 相同
            banned_tokens_mask = tf.scatter_nd(
                indices=tf.expand_dims(banned_tokens, axis=-1),
                updates=tf.ones_like(banned_tokens, dtype=tf.bool),
                shape=row_score.shape,
            )
            # 使用 `-inf` 替换被禁止令牌的位置上的分数，保持其它位置不变
            row_score = tf.where(banned_tokens_mask, -float("inf"), row_score)
            return row_score
        
        # 对每一行调用 `_get_row_updated_score` 函数，更新分数 `scores`，并返回更新后的 `scores`
        scores = tf.map_fn(_get_row_updated_score, (input_ids, scores), fn_output_signature=tf.float32)
        return scores
class TFNoRepeatNGramLogitsProcessor(TFLogitsProcessor):
    r"""
    [`TFLogitsProcessor`] that enforces no repetition of n-grams. See
    [Fairseq](https://github.com/pytorch/fairseq/blob/a07cb6f40480928c9e0548b737aadd36ee66ac76/fairseq/sequence_generator.py#L345).

    Args:
        ngram_size (`int`):
            All ngrams of size `ngram_size` can only occur once.
    """

    def __init__(self, ngram_size: int):
        # 初始化方法，验证并设置 ngram_size 参数
        if not isinstance(ngram_size, int) or ngram_size <= 0:
            raise ValueError(f"`ngram_size` has to be a strictly positive integer, but is {ngram_size}")
        self.ngram_size = ngram_size

    def calc_banned_ngram_tokens(self, input_ids, num_hypos, cur_len):
        # 计算禁止的 ngram tokens，用于防止 ngram 重复
        # 从 fairseq 中复制用于在 beam search 中实现 no_repeat_ngram
        if cur_len + 1 < self.ngram_size:
            # 如果当前长度加 1 小于 ngram_size，返回空列表表示没有禁止的 token
            return [[] for _ in range(num_hypos)]
        generated_ngrams = [{} for _ in range(num_hypos)]
        prev_input_ids = input_ids[:, :cur_len]
        for idx in range(num_hypos):
            gen_tokens = prev_input_ids[idx].numpy().tolist()
            generated_ngram = generated_ngrams[idx]
            for ngram in zip(*[gen_tokens[i:] for i in range(self.ngram_size)]):
                prev_ngram_tuple = tuple(ngram[:-1])
                generated_ngram[prev_ngram_tuple] = generated_ngram.get(prev_ngram_tuple, []) + [ngram[-1]]

        def _get_generated_ngrams(hypo_idx):
            # 在解码下一个 token 前，防止解码已经出现的 ngrams
            start_idx = cur_len + 1 - self.ngram_size
            ngram_idx = tuple(prev_input_ids[hypo_idx, start_idx:cur_len].numpy().tolist())
            return generated_ngrams[hypo_idx].get(ngram_idx, [])

        # 返回禁止的 tokens 列表
        banned_tokens = [_get_generated_ngrams(hypo_idx) for hypo_idx in range(num_hypos)]

        return banned_tokens

    def __call__(self, input_ids: tf.Tensor, scores: tf.Tensor, cur_len: int) -> tf.Tensor:
        # 调用对象时的处理方法，用于处理 logits
        # TODO (joao): enable XLA on this logits processor. See discussion and attempts in
        # https://github.com/huggingface/transformers/pull/16974
        if not tf.executing_eagerly():
            raise NotImplementedError("TFNoRepeatNGramLogitsProcessor is only implemented for eager execution.")

        batch_size, vocab_size = scores.shape
        # 计算禁止的 ngram tokens
        banned_tokens = self.calc_banned_ngram_tokens(input_ids, batch_size, cur_len)

        # 创建禁止 tokens 的布尔掩码
        banned_tokens_indices_mask = []
        for banned_tokens_slice in banned_tokens:
            banned_tokens_indices_mask.append(
                [True if token in banned_tokens_slice else False for token in range(vocab_size)]
            )

        # 将禁止的 tokens 对应位置的 logits 设置为负无穷
        scores = tf.where(tf.convert_to_tensor(banned_tokens_indices_mask, dtype=tf.bool), -float("inf"), scores)

        return scores


class TFForcedBOSTokenLogitsProcessor(TFLogitsProcessor):
    r"""
    # 初始化函数，接受强制作为第一个生成标记的标记 ID
    def __init__(self, bos_token_id: int):
        # 如果 bos_token_id 小于 0，则引发值错误异常
        if bos_token_id < 0:
            raise ValueError(f"The forced bos token id must be a non-negative integer, got {bos_token_id}")
        # 将传入的 bos_token_id 分配给实例变量
        self.bos_token_id = bos_token_id

    # 调用函数，处理输入的 token IDs 和对应的分数，根据当前生成的长度 cur_len 进行调整
    def __call__(self, input_ids: tf.Tensor, scores: tf.Tensor, cur_len: int) -> tf.Tensor:
        # 如果当前生成的长度为 1
        if cur_len == 1:
            # 获取批处理大小和标记数
            batch_size, num_tokens = scores.shape
            # 将 bos_token_id 列的分数设为 0
            scores = tf.zeros((batch_size, 1))
            # 如果 bos_token_id 大于 0，将除了第 bos_token_id 列外的分数设置为负无穷
            if self.bos_token_id > 0:
                scores = tf.concat((tf.broadcast_to(-float("inf"), (batch_size, self.bos_token_id)), scores), axis=-1)
            # 如果 bos_token_id 小于 (num_tokens - 1)，将除了第 bos_token_id 列外的分数设置为负无穷
            if self.bos_token_id < (num_tokens - 1):
                scores = tf.concat(
                    (scores, tf.broadcast_to(-float("inf"), (batch_size, (num_tokens - 1) - self.bos_token_id))),
                    axis=-1,
                )
        # 返回调整后的分数张量
        return scores
# 定义一个继承自 `TFLogitsProcessor` 的类，用于在达到 `max_length` 时强制指定的 token 成为生成序列的最后一个 token。
class TFForcedEOSTokenLogitsProcessor(TFLogitsProcessor):
    r"""
    [`TFLogitsProcessor`] that enforces the specified token as the last generated token when `max_length` is reached.

    Args:
        max_length (`int`):
            The maximum length of the sequence to be generated.
        eos_token_id (`int`):
            The id of the token to force as the last generated token when `max_length` is reached.
    """

    # 初始化方法，设置 `max_length` 和 `eos_token_id`
    def __init__(self, max_length: int, eos_token_id: int):
        self.max_length = max_length
        # 如果 `eos_token_id` 小于 0，则抛出错误
        if eos_token_id < 0:
            raise ValueError(f"The forced eos token id must be a non-negative integer, got {eos_token_id}")
        self.eos_token_id = eos_token_id

    # 调用方法，根据当前生成的长度 `cur_len` 对 `scores` 进行处理
    def __call__(self, input_ids: tf.Tensor, scores: tf.Tensor, cur_len: int) -> tf.Tensor:
        # 当当前长度 `cur_len` 等于 `max_length - 1` 时
        if cur_len == self.max_length - 1:
            batch_size, num_tokens = scores.shape
            # 将 `scores` 在 `eos_token_id` 列上的值设为 0
            scores = tf.zeros((batch_size, 1))
            # 在除了 `eos_token_id` 外的其他位置上的值设为负无穷
            if self.eos_token_id > 0:
                scores = tf.concat((tf.broadcast_to(-float("inf"), (batch_size, self.eos_token_id)), scores), axis=-1)
            if self.eos_token_id < (num_tokens - 1):
                scores = tf.concat(
                    (scores, tf.broadcast_to(-float("inf"), (batch_size, (num_tokens - 1) - self.eos_token_id))),
                    axis=-1,
                )
        return scores


# 定义一个继承自 `TFLogitsProcessor` 的类，用于在生成序列开始时抑制一组 token 的生成。
class TFSuppressTokensAtBeginLogitsProcessor(TFLogitsProcessor):
    r"""
    [`TFSuppressTokensAtBeginLogitsProcessor`] suppresses a list of tokens as soon as the `generate` function starts
    generating using `begin_index` tokens. This should ensure that the tokens defined by `begin_suppress_tokens` at not
    sampled at the begining of the generation.
    """

    # 初始化方法，设置 `begin_suppress_tokens` 和 `begin_index`
    def __init__(self, begin_suppress_tokens, begin_index):
        self.begin_suppress_tokens = list(begin_suppress_tokens)
        self.begin_index = begin_index

    # 调用方法，根据当前生成的长度 `cur_len` 对 `scores` 进行处理
    def __call__(self, input_ids: tf.Tensor, scores: tf.Tensor, cur_len: int) -> tf.Tensor:
        # 当当前长度 `cur_len` 等于 `begin_index` 时
        scores = tf.cond(
            tf.equal(cur_len, self.begin_index),
            # 使用 `tf.tensor_scatter_nd_update` 将 `scores` 中指定位置的值更新为负无穷
            lambda: tf.tensor_scatter_nd_update(
                scores,
                indices=[[i, token] for i in range(scores.shape[0]) for token in self.begin_suppress_tokens],
                updates=[-float("inf") for _ in range(scores.shape[0] * len(self.begin_suppress_tokens))],
            ),
            lambda: scores,  # 如果条件不满足，返回原始的 `scores`
        )
        return scores


# 定义一个继承自 `TFLogitsProcessor` 的类，用于抑制一组 token 的生成。
class TFSuppressTokensLogitsProcessor(TFLogitsProcessor):
    r"""This processor can be used to suppress a list of tokens. The processor will set their log probs to `-inf` so that they
    are not sampled."""

    # 初始化方法，设置 `suppress_tokens`
    def __init__(self, suppress_tokens):
        self.suppress_tokens = list(suppress_tokens)
    # 定义一个方法 __call__，该方法接受三个参数：input_ids 是 tf.Tensor 类型，scores 是 tf.Tensor 类型，cur_len 是整数类型
    def __call__(self, input_ids: tf.Tensor, scores: tf.Tensor, cur_len: int) -> tf.Tensor:
        # 使用 tf.tensor_scatter_nd_update 函数更新 scores 张量
        scores = tf.tensor_scatter_nd_update(
            # 更新的目标张量是 scores
            scores,
            # 更新操作的索引是一个列表推导式，生成所有 (i, token) 对的索引
            indices=[[i, token] for i in range(scores.shape[0]) for token in self.suppress_tokens],
            # 更新操作的值是一个列表推导式，生成所有需要更新的 -inf 值
            updates=[-float("inf") for _ in range(scores.shape[0] * len(self.suppress_tokens))],
        )
        # 返回更新后的 scores 张量
        return scores
class TFForceTokensLogitsProcessor(TFLogitsProcessor):
    r"""This processor takes a list of pairs of integers which indicates a mapping from generation indices to token
    indices that will be forced before sampling. The processor will set their log probs to `0` and all other tokens to
    `-inf` so that they are sampled at their corresponding index."""

    def __init__(self, force_token_map: List[List[int]]):
        # 将输入的强制 token 映射列表转换为字典形式，格式为 {index: token}
        force_token_map = dict(force_token_map)
        
        # 创建一个数组 force_token_array，其长度为 force_token_map 中最大的索引加一，
        # 初始化所有元素为 -1，用于表示未被强制的 token
        force_token_array = np.ones((max(force_token_map.keys()) + 1), dtype=np.int32) * -1
        
        # 遍历 force_token_map，将指定索引位置的 token 值存入 force_token_array
        for index, token in force_token_map.items():
            if token is not None:
                force_token_array[index] = token
        
        # 将 force_token_array 转换为 TensorFlow 张量，并存储在实例变量 self.force_token_array 中
        self.force_token_array = tf.convert_to_tensor(force_token_array, dtype=tf.int32)

    def __call__(self, input_ids: tf.Tensor, scores: tf.Tensor, cur_len: int) -> tf.Tensor:
        # 定义内部函数 _force_token，用于处理强制 token 的逻辑
        def _force_token(generation_idx):
            batch_size = scores.shape[0]
            current_token = self.force_token_array[generation_idx]

            # 创建一个新的得分张量 new_scores，初始化为 -inf
            new_scores = tf.ones_like(scores, dtype=scores.dtype) * -float("inf")
            
            # 创建索引张量 indices，用于更新 new_scores 中的特定位置为 0
            indices = tf.stack((tf.range(batch_size), tf.tile([current_token], [batch_size])), axis=1)
            updates = tf.zeros((batch_size,), dtype=scores.dtype)
            new_scores = tf.tensor_scatter_nd_update(new_scores, indices, updates)
            
            return new_scores
        
        # 根据当前序列长度 cur_len 和 force_token_array 的长度，决定是否对 scores 进行处理
        scores = tf.cond(
            tf.greater_equal(cur_len, tf.shape(self.force_token_array)[0]),
            # 如果当前长度大于等于 force_token_array 的长度，不进行处理，直接返回 scores
            lambda: tf.identity(scores),
            # 否则，根据 force_token_array 中对应位置的值决定是否强制 token
            lambda: tf.cond(
                tf.greater_equal(self.force_token_array[cur_len], 0),
                # 如果 force_token_array[cur_len] 大于等于 0，调用 _force_token 强制 token
                lambda: _force_token(cur_len),
                # 否则，不进行处理，直接返回 scores
                lambda: scores,
            ),
        )
        
        return scores
```