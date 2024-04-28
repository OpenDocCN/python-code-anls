# `.\transformers\generation\beam_search.py`

```py
# 设置文件编码为 UTF-8
# 版权声明，版权归 The HuggingFace Inc. 团队所有
# 根据 Apache 许可证 2.0 版本授权
# 除非符合许可证规定，否则不得使用此文件
# 可以在以下网址获取许可证副本
# http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则按"原样"分发软件
# 没有任何形式的担保或条件，无论是明示的还是暗示的
# 请查看许可证以获取特定语言的权限和限制

# 导入必要的库
from abc import ABC, abstractmethod
from collections import UserDict
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch

# 导入自定义的模块
from ..utils import add_start_docstrings
from .beam_constraints import Constraint, ConstraintListState

# 定义一个文档字符串，用于描述处理输入的函数
PROCESS_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size * num_beams, sequence_length)`):
            输入序列标记在词汇表中的索引。

            可以使用任何继承自 [`PreTrainedTokenizer`] 的类来获取索引。参见
            [`PreTrainedTokenizer.encode`] 和 [`PreTrainedTokenizer.__call__`] 获取详细信息。

            [什么是输入 ID？](../glossary#input-ids)
        next_scores (`torch.FloatTensor` of shape `(batch_size, 2 * num_beams)`):
            当前顶部 `2 * num_beams` 个未完成的束假设的分数。
        next_tokens (`torch.LongTensor` of shape `(batch_size, 2 * num_beams)`):
            与顶部 `2 * num_beams` 个未完成的束假设对应的 `input_ids`。
        next_indices (`torch.LongTensor` of shape `(batch_size, 2 * num_beams)`):
            指示 `next_tokens` 对应的束假设的束索引。
        pad_token_id (`int`, *optional*):
            *填充* 标记的 ID。
        eos_token_id (`Union[int, List[int]]`, *optional*):
            *序列结束* 标记的 ID。可选，使用列表设置多个 *序列结束* 标记。
        beam_indices (`torch.LongTensor`, *optional*):
            指示每个标记对应的束假设的束索引。
        group_index (`int`, *optional*):
            束组的索引。与 [`~PreTrainedModel.group_beam_search`] 一起使用。

    Return:
        `UserDict`: 由上述字段组成的字典:

            - **next_beam_scores** (`torch.FloatTensor` of shape `(batch_size * num_beams)`) -- 所有未完成束的更新分数。
            - **next_beam_tokens** (`torch.FloatTensor` of shape `(batch_size * num_beams)`) -- 要添加到未完成束假设的下一个标记。
            - **next_beam_indices** (`torch.FloatTensor` of shape `(batch_size * num_beams)`) -- 指示下一个标记应添加到哪个束中。
"""
# 定义了一个长字符串，用于描述输入参数和返回值的含义
FINALIZE_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size * num_beams, sequence_length)`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using any class inheriting from [`PreTrainedTokenizer`]. See
            [`PreTrainedTokenizer.encode`] and [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        final_beam_scores (`torch.FloatTensor` of shape `(batch_size * num_beams)`):
            The final scores of all non-finished beams.
        final_beam_tokens (`torch.FloatTensor` of shape `(batch_size * num_beams)`):
            The last tokens to be added to the non-finished beam_hypotheses.
        final_beam_indices (`torch.FloatTensor` of shape `(batch_size * num_beams)`):
            The beam indices indicating to which beam the `final_beam_tokens` shall be added.
        pad_token_id (`int`, *optional*):
            The id of the *padding* token.
        eos_token_id (`Union[int, List[int]]`, *optional*):
            The id of the *end-of-sequence* token. Optionally, use a list to set multiple *end-of-sequence* tokens.

    Return:
        `torch.LongTensor` of shape `(batch_size * num_return_sequences, sequence_length)`: The generated sequences.
        The second dimension (sequence_length) is either equal to `max_length` or shorter if all batches finished early
        due to the `eos_token_id`.

"""


class BeamScorer(ABC):
    """
    Abstract base class for all beam scorers that are used for [`~PreTrainedModel.beam_search`] and
    [`~PreTrainedModel.beam_sample`].
    """

    @abstractmethod
    @add_start_docstrings(PROCESS_INPUTS_DOCSTRING)
    # 定义了一个抽象方法，用于处理beam搜索中的输入
    def process(
        self,
        input_ids: torch.LongTensor,
        next_scores: torch.FloatTensor,
        next_tokens: torch.LongTensor,
        next_indices: torch.LongTensor,
        **kwargs,
    ) -> Tuple[torch.Tensor]:
        raise NotImplementedError("This is an abstract method.")

    @abstractmethod
    @add_start_docstrings(FINALIZE_INPUTS_DOCSTRING)
    # 定义了一个抽象方法，用于最终处理beam搜索的结果
    def finalize(
        self,
        input_ids: torch.LongTensor,
        next_scores: torch.FloatTensor,
        next_tokens: torch.LongTensor,
        next_indices: torch.LongTensor,
        max_length: int,
        **kwargs,
    ) -> torch.LongTensor:
        raise NotImplementedError("This is an abstract method.")


class BeamSearchScorer(BeamScorer):
    r"""
    [`BeamScorer`] implementing standard beam search decoding.

    Adapted in part from [Facebook's XLM beam search
    code](https://github.com/facebookresearch/XLM/blob/9e6f6814d17be4fe5b15f2e6c43eb2b2d76daeb4/src/model/transformer.py#L529).

    Reference for the diverse beam search algorithm and implementation [Ashwin Kalyan's DBS
    implementation](https://github.com/ashwinkalyan/dbs/blob/master/dbs/beam_utils.lua)
    Args:
        batch_size (`int`):
            并行运行标准束搜索解码的 `input_ids` 批量大小。
        num_beams (`int`):
            束搜索的束数。
        device (`torch.device`):
            定义此 `BeamSearchScorer` 实例将分配的设备类型（例如 `"cpu"` 或 `"cuda"`）。
        length_penalty (`float`, *optional*, 默认为 1.0):
            用于束搜索生成的长度的指数惩罚。它作为指数应用于序列长度，然后用于除以序列的得分。由于得分是序列的对数似然（即负数），`length_penalty` > 0.0 促进更长的序列，而 `length_penalty` < 0.0 鼓励更短的序列。
        do_early_stopping (`bool` or `str`, *optional*, 默认为 `False`):
            控制束搜索等基于束的方法的停止条件。它接受以下值：`True`，当有 `num_beams` 个完整候选时，生成立即停止；`False`，应用启发式方法，当很难找到更好的候选时停止生成；`"never"`，当不能有更好的候选时束搜索过程才停止（经典束搜索算法）。
        num_beam_hyps_to_keep (`int`, *optional*, 默认为 1):
            在调用 [`~transformers.BeamSearchScorer.finalize`] 时返回的束假设数。
        num_beam_groups (`int`, *optional*, 默认为 1):
            将 `num_beams` 分成多少组，以确保不同组束之间的多样性。有关更多详细信息，请参阅[此论文](https://arxiv.org/pdf/1610.02424.pdf)。
        max_length (`int`, *optional*):
            要生成的序列的最大长度。
    """

    def __init__(
        self,
        batch_size: int,
        num_beams: int,
        device: torch.device,
        length_penalty: Optional[float] = 1.0,
        do_early_stopping: Optional[Union[bool, str]] = False,
        num_beam_hyps_to_keep: Optional[int] = 1,
        num_beam_groups: Optional[int] = 1,
        max_length: Optional[int] = None,
        # 初始化 BeamSearchScorer 对象，设置束搜索参数
        self.num_beams = num_beams
        self.device = device
        self.length_penalty = length_penalty
        self.do_early_stopping = do_early_stopping
        self.num_beam_hyps_to_keep = num_beam_hyps_to_keep
        self.num_beam_groups = num_beam_groups
        self.group_size = self.num_beams // self.num_beam_groups

        self._is_init = False
        # 初始化 `_beam_hyps` 列表，用于存储各个组的 BeamHypotheses 对象
        # `_beam_hyps[i*self.num_beam_groups+j]` 表示第 i 个 mini-batch 中第 j 个组的 BeamHypotheses 对象
        # 如果不使用 group_beam_search，则列表包含 `batch_size` 个 BeamHypotheses 对象
        self._beam_hyps = [
            BeamHypotheses(
                num_beams=self.group_size,
                length_penalty=self.length_penalty,
                early_stopping=self.do_early_stopping,
                max_length=max_length,
            )
            for _ in range(batch_size * self.num_beam_groups)
        ]
        # 初始化 `_done` 张量，表示各个组的 BeamHypotheses 是否生成完成
        self._done = torch.tensor(
            [False for _ in range(batch_size * self.num_beam_groups)], dtype=torch.bool, device=self.device
        )

        # 检查 num_beams 和 num_beam_groups 参数是否合法
        if not isinstance(num_beams, int) or num_beams <= 1:
            raise ValueError(
                f"`num_beams` has to be an integer strictly greater than 1, but is {num_beams}. For `num_beams` == 1,"
                " one should make use of `greedy_search` instead."
            )

        if not isinstance(num_beam_groups, int) or (num_beam_groups > num_beams) or (num_beams % num_beam_groups != 0):
            raise ValueError(
                "`num_beam_groups` has to be an integer smaller or equal than `num_beams` and `num_beams` has to be"
                f" divisible by `num_beam_groups`, but is {num_beam_groups} with `num_beams` being {num_beams}."
            )

    @property
    # 定义 is_done 属性，表示所有 BeamHypotheses 是否生成完成
    def is_done(self) -> bool:
        return self._done.all()

    # 定义 process 方法，用于处理下一个 token 的得分和索引
    def process(
        self,
        input_ids: torch.LongTensor,
        next_scores: torch.FloatTensor,
        next_tokens: torch.LongTensor,
        next_indices: torch.LongTensor,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        beam_indices: Optional[torch.LongTensor] = None,
        group_index: Optional[int] = 0,
        decoder_prompt_len: Optional[int] = 0,
    # 定义 finalize 方法，用于处理最终生成的序列及相关参数
    def finalize(
        self,
        input_ids: torch.LongTensor,
        final_beam_scores: torch.FloatTensor,
        final_beam_tokens: torch.LongTensor,
        final_beam_indices: torch.LongTensor,
        max_length: int,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        beam_indices: Optional[torch.LongTensor] = None,
        decoder_prompt_len: Optional[int] = 0,
class ConstrainedBeamSearchScorer(BeamScorer):
    r"""
    [`BeamScorer`] implementing constrained beam search decoding.


    Args:
        batch_size (`int`):
            Batch Size of `input_ids` for which standard beam search decoding is run in parallel.
        num_beams (`int`):
            Number of beams for beam search.
        constraints (`List[Constraint]`):
            A list of positive constraints represented as `Constraint` objects that must be fulfilled in the generation
            output. For more information, the documentation of [`Constraint`] should be read.
        device (`torch.device`):
            Defines the device type (*e.g.*, `"cpu"` or `"cuda"`) on which this instance of `BeamSearchScorer` will be
            allocated.
        length_penalty (`float`, *optional*, defaults to 1.0):
            Exponential penalty to the length that is used with beam-based generation. It is applied as an exponent to
            the sequence length, which in turn is used to divide the score of the sequence. Since the score is the log
            likelihood of the sequence (i.e. negative), `length_penalty` > 0.0 promotes longer sequences, while
            `length_penalty` < 0.0 encourages shorter sequences.
        do_early_stopping (`bool` or `str`, *optional*, defaults to `False`):
            Controls the stopping condition for beam-based methods, like beam-search. It accepts the following values:
            `True`, where the generation stops as soon as there are `num_beams` complete candidates; `False`, where an
            heuristic is applied and the generation stops when is it very unlikely to find better candidates;
            `"never"`, where the beam search procedure only stops when there cannot be better candidates (canonical
            beam search algorithm).
        num_beam_hyps_to_keep (`int`, *optional*, defaults to 1):
            The number of beam hypotheses that shall be returned upon calling
            [`~transformers.BeamSearchScorer.finalize`].
        num_beam_groups (`int`, *optional*, defaults to 1):
            Number of groups to divide `num_beams` into in order to ensure diversity among different groups of beams.
            See [this paper](https://arxiv.org/pdf/1610.02424.pdf) for more details.
        max_length (`int`, *optional*):
            The maximum length of the sequence to be generated.
    """

    def __init__(
        self,
        batch_size: int,
        num_beams: int,
        constraints: List[Constraint],
        device: torch.device,
        length_penalty: Optional[float] = 1.0,
        do_early_stopping: Optional[Union[bool, str]] = False,
        num_beam_hyps_to_keep: Optional[int] = 1,
        num_beam_groups: Optional[int] = 1,
        max_length: Optional[int] = None,
        # 初始化 BeamSearchScorer 实例的属性
        self.num_beams = num_beams
        self.device = device
        self.length_penalty = length_penalty
        self.do_early_stopping = do_early_stopping
        self.num_beam_hyps_to_keep = num_beam_hyps_to_keep
        self.num_beam_groups = num_beam_groups
        self.group_size = self.num_beams // self.num_beam_groups
        self.constraints = constraints

        # 初始化 BeamSearchScorer 实例的其他属性
        self._is_init = False
        # 创建一个包含 BeamHypotheses 实例的列表，用于存储每个 batch 中的 beam hypotheses
        self._beam_hyps = [
            BeamHypotheses(
                num_beams=self.num_beams,
                length_penalty=self.length_penalty,
                early_stopping=self.do_early_stopping,
                max_length=max_length,
            )
            for _ in range(batch_size)
        ]
        # 创建一个 Tensor，用于标记每个 batch 中的句子是否已经生成完成
        self._done = torch.tensor([False for _ in range(batch_size)], dtype=torch.bool, device=self.device)

        # 检查 num_beams 和 num_beam_groups 是否满足条件，若不满足则抛出 ValueError
        if not isinstance(num_beams, int) or num_beams <= 1:
            raise ValueError(
                f"`num_beams` has to be an integer strictly greater than 1, but is {num_beams}. For `num_beams` == 1,"
                " one should make use of `greedy_search` instead."
            )

        if not isinstance(num_beam_groups, int) or (num_beam_groups > num_beams) or (num_beams % num_beam_groups != 0):
            raise ValueError(
                "`num_beam_groups` has to be an integer smaller or equal than `num_beams` and `num_beams` has to be"
                f" divisible by `num_beam_groups`, but is {num_beam_groups} with `num_beams` being {num_beams}."
            )

    @property
    def is_done(self) -> bool:
        # 返回所有 batch 是否都已经生成完成的标志
        return self._done.all()

    def make_constraint_states(self, n):
        # 创建包含 n 个 ConstraintListState 实例的列表，用于表示约束的状态
        return [ConstraintListState([constraint.copy() for constraint in self.constraints]) for _ in range(n)]

    def check_completes_constraints(self, sequence):
        # 根据输入序列检查是否满足约束
        new_state = self.make_constraint_states(1)[0]
        new_state.reset(sequence)
        return new_state.completed

    def process(
        self,
        input_ids: torch.LongTensor,
        next_scores: torch.FloatTensor,
        next_tokens: torch.LongTensor,
        next_indices: torch.LongTensor,
        scores_for_all_vocab: torch.FloatTensor,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        beam_indices: Optional[torch.LongTensor] = None,
        decoder_prompt_len: Optional[int] = 0,
    ):
        # Beam search 的主要处理方法，用于生成下一个 token 的序列
        ...

    def step_sentence_constraint(
        self,
        batch_idx: int,
        input_ids: torch.LongTensor,
        vocab_scores: torch.FloatTensor,
        sent_beam_scores: torch.FloatTensor,
        sent_beam_tokens: torch.LongTensor,
        sent_beam_indices: torch.LongTensor,
        push_progress: bool = False,
    ):
        # 实现对句子级别约束的处理
        ...
    # 定义一个方法用于最终处理生成的序列，对应的输入 ID
    def finalize(
        self,
        # 模型生成的输入 ID
        input_ids: torch.LongTensor,
        # 最终的束搜索分数
        final_beam_scores: torch.FloatTensor,
        # 最终的束搜索 token 序列
        final_beam_tokens: torch.LongTensor,
        # 最终的束搜索 token 对应的索引
        final_beam_indices: torch.LongTensor,
        # 序列的最大长度
        max_length: int,
        # 填充 token 的 ID，可选参数，默认为 None
        pad_token_id: Optional[int] = None,
        # 结束 token 的 ID，可选参数，默认为 None
        eos_token_id: Optional[Union[int, List[int]]] = None,
        # 束搜索的索引，可选参数，默认为 None
        beam_indices: Optional[torch.LongTensor] = None,
        # 解码器的提示长度，可选参数，默认为 0
        decoder_prompt_len: Optional[int] = 0,
class BeamHypotheses:
    def __init__(self, num_beams: int, length_penalty: float, early_stopping: bool, max_length: Optional[int] = None):
        """
        Initialize n-best list of hypotheses.
        """
        # 初始化 BeamHypotheses 类，设置初始参数
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping
        self.max_length = max_length
        self.num_beams = num_beams
        self.beams = []  # 初始化空的假设列表
        self.worst_score = 1e9  # 初始化最差分数为1e9

        # 如果 early_stopping 不是布尔类型且 max_length 为 None，则抛出异常
        if not isinstance(self.early_stopping, bool) and self.max_length is None:
            raise ValueError(
                "When `do_early_stopping` is set to a string, `max_length` must be defined. Ensure it is passed to the"
                " BeamScorer class instance at initialization time."
            )

    def __len__(self):
        """
        Number of hypotheses in the list.
        """
        return len(self.beams)  # 返回假设列表中的假设数量

    def add(
        self,
        hyp: torch.LongTensor,
        sum_logprobs: float,
        beam_indices: Optional[torch.LongTensor] = None,
        generated_len: Optional[int] = None,
    ):
        """
        Add a new hypothesis to the list.
        """
        # 计算新假设的分数
        if generated_len is not None:
            score = sum_logprobs / (generated_len**self.length_penalty)
        # 这个 'else' 情况是为了向后兼容
        else:
            score = sum_logprobs / (hyp.shape[-1] ** self.length_penalty)

        # 如果假设列表中的假设数量小于 num_beams 或者分数大于最差分数
        if len(self) < self.num_beams or score > self.worst_score:
            # 添加新的假设到列表中
            self.beams.append((score, hyp, beam_indices))
            # 如果假设数量超过 num_beams
            if len(self) > self.num_beams:
                # 按分数排序假设列表，删除分数最低的假设
                sorted_next_scores = sorted([(s, idx) for idx, (s, _, _) in enumerate(self.beams)])
                del self.beams[sorted_next_scores[0][1]]
                # 更新最差分数为第二低的分数
                self.worst_score = sorted_next_scores[1][0]
            else:
                # 更新最差分数为当前分数和最差分数中的最小值
                self.worst_score = min(score, self.worst_score)
    def is_done(self, best_sum_logprobs: float, cur_len: int, decoder_prompt_len: Optional[int] = 0) -> bool:
        """
        If there are enough hypotheses and that none of the hypotheses being generated can become better than the worst
        one in the heap, then we are done with this sentence.
        """

        # 如果当前堆中的假设数量小于束搜索的数量，则返回 False，表示尚未完成生成
        if len(self) < self.num_beams:
            return False

        # 如果早停策略为 True，则立即停止生成，无论何时生成了足够数量的假设
        if self.early_stopping is True:
            return True
        # 如果早停策略为 False，则根据当前长度计算可能的最高分数，尽管在 `length_penalty` 为正值时不完全准确。
        # 详情请参阅下面的讨论。
        # https://github.com/huggingface/transformers/pull/20901#issuecomment-1369845565
        elif self.early_stopping is False:
            highest_attainable_score = best_sum_logprobs / (cur_len - decoder_prompt_len) ** self.length_penalty
            # 检查当前最差得分是否高于或等于可能的最高分数，若是则返回 True，表示已经完成生成
            ret = self.worst_score >= highest_attainable_score
            return ret
        # 如果早停策略为 "never"，则根据长度惩罚的信号计算可能的最高分数
        else:
            # 当长度惩罚大于 0.0 时，从 `max_length` 而不是 `cur_len` 获取最大的分母
            # 最小化 `highest_attainable_score` 的绝对值，因此 `highest_attainable_score` 是负值，
            # 因此通过这种方式获得其最大值
            if self.length_penalty > 0.0:
                # 若 `max_length` 小于等于解码器提示的长度，则抛出 ValueError 异常
                if self.max_length <= decoder_prompt_len:
                    raise ValueError("max_length is not larger than decoder prompt length")
                highest_attainable_score = (
                    best_sum_logprobs / (self.max_length - decoder_prompt_len) ** self.length_penalty
                )
            # 当长度惩罚小于等于 0.0 时，根据当前长度计算可能的最高分数
            else:
                highest_attainable_score = best_sum_logprobs / (cur_len - decoder_prompt_len) ** self.length_penalty
            # 检查当前最差得分是否高于或等于可能的最高分数，若是则返回 True，表示已经完成生成
            ret = self.worst_score >= highest_attainable_score
            return ret
```