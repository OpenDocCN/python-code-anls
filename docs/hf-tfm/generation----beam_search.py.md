# `.\generation\beam_search.py`

```
# 导入必要的模块和库
from abc import ABC, abstractmethod  # 导入抽象基类和抽象方法装饰器
from collections import UserDict  # 导入用户自定义字典类
from typing import Dict, List, Optional, Tuple, Union  # 导入类型提示

import numpy as np  # 导入 NumPy 库
import torch  # 导入 PyTorch 库

from ..utils import add_start_docstrings  # 从上级目录的 utils 模块导入 add_start_docstrings 函数
from .beam_constraints import Constraint, ConstraintListState  # 从当前目录的 beam_constraints 模块导入 Constraint 和 ConstraintListState 类

# 定义常量，该常量包含一个多行的文档字符串，用于描述函数 process_inputs 的参数和返回值
PROCESS_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size * num_beams, sequence_length)`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using any class inheriting from [`PreTrainedTokenizer`]. See
            [`PreTrainedTokenizer.encode`] and [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        next_scores (`torch.FloatTensor` of shape `(batch_size, 2 * num_beams)`):
            Current scores of the top `2 * num_beams` non-finished beam hypotheses.
        next_tokens (`torch.LongTensor` of shape `(batch_size, 2 * num_beams)`):
            `input_ids` of the tokens corresponding to the top `2 * num_beams` non-finished beam hypotheses.
        next_indices (`torch.LongTensor` of shape `(batch_size, 2 * num_beams)`):
            Beam indices indicating to which beam hypothesis the `next_tokens` correspond.
        pad_token_id (`int`, *optional*):
            The id of the *padding* token.
        eos_token_id (`Union[int, List[int]]`, *optional*):
            The id of the *end-of-sequence* token. Optionally, use a list to set multiple *end-of-sequence* tokens.
        beam_indices (`torch.LongTensor`, *optional*):
            Beam indices indicating to which beam hypothesis each token correspond.
        group_index (`int`, *optional*):
            The index of the group of beams. Used with [`~PreTrainedModel.group_beam_search`].

    Return:
        `UserDict`: A dictionary composed of the fields as defined above:

            - **next_beam_scores** (`torch.FloatTensor` of shape `(batch_size * num_beams)`) -- Updated scores of all
              non-finished beams.
            - **next_beam_tokens** (`torch.FloatTensor` of shape `(batch_size * num_beams)`) -- Next tokens to be added
              to the non-finished beam_hypotheses.
            - **next_beam_indices** (`torch.FloatTensor` of shape `(batch_size * num_beams)`) -- Beam indices
              indicating to which beam the next tokens shall be added.
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
    @add_start_docstrings(PROCESS_INPUTS_DOCSTRING)  # 添加输入处理方法的文档字符串
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
    @add_start_docstrings(FINALIZE_INPUTS_DOCSTRING)  # 添加最终处理方法的文档字符串
    def finalize(
        self,
        input_ids: torch.LongTensor,
        next_scores: torch.FloatTensor,
        next_tokens: torch.LongTensor,
        next_indices: torch.LongTensor,
        max_length: int,
        **kwargs,
    ) ->
        Args:
            batch_size (`int`):
                并行运行标准束搜索解码的 `input_ids` 的批大小。
            num_beams (`int`):
                梁搜索的束大小。
            device (`torch.device`):
                分配此 `BeamSearchScorer` 实例的设备类型（例如 `"cpu"` 或 `"cuda"`）。
            length_penalty (`float`, *optional*, defaults to 1.0):
                用于基于束搜索的生成的指数长度惩罚。应用为序列长度的指数，然后用于将序列的分数除以此值。由于分数是序列的对数似然（即负数），`length_penalty` > 0.0 会促进更长的序列，而 `length_penalty` < 0.0 会鼓励更短的序列。
            do_early_stopping (`bool` or `str`, *optional*, defaults to `False`):
                控制束搜索等方法（如束搜索）的停止条件。接受以下值：
                `True`，生成器一旦有 `num_beams` 个完整候选项即停止；
                `False`，应用启发式方法，生成器停止时不太可能找到更好的候选项；
                `"never"`，束搜索过程仅在不能有更好的候选项时停止（典型的束搜索算法）。
            num_beam_hyps_to_keep (`int`, *optional*, defaults to 1):
                在调用 [`~transformers.BeamSearchScorer.finalize`] 后返回的束假设数量。
            num_beam_groups (`int`, *optional*, defaults to 1):
                为了确保不同束组之间的多样性，将 `num_beams` 分成的组数。详细信息请参阅[此论文](https://arxiv.org/pdf/1610.02424.pdf)。
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
        ):
        self.num_beams = num_beams
        self.device = device
        self.length_penalty = length_penalty
        self.do_early_stopping = do_early_stopping
        self.num_beam_hyps_to_keep = num_beam_hyps_to_keep
        self.num_beam_groups = num_beam_groups
        self.group_size = self.num_beams // self.num_beam_groups

        self._is_init = False
        # self._beam_hyps[i*self.num_beam_groups+j] is the beam_hyps of the j-th group in the i-th mini-batch.
        # If group_beam_search is not used, the list consists of `batch_size` beam_hyps.
        self._beam_hyps = [
            BeamHypotheses(
                num_beams=self.group_size,  # 创建 BeamHypotheses 对象，设置每个组的 beam 数量
                length_penalty=self.length_penalty,  # 设置长度惩罚因子
                early_stopping=self.do_early_stopping,  # 设置是否提前停止
                max_length=max_length,  # 设置最大生成长度
            )
            for _ in range(batch_size * self.num_beam_groups)  # 根据 mini-batch 大小和组数创建多个 BeamHypotheses 对象
        ]
        # self._done[i*self.num_beam_groups+j] indicates whether the generation of the beam_hyps of the j-th group
        # in the i-th mini-batch is complete.
        self._done = torch.tensor(
            [False for _ in range(batch_size * self.num_beam_groups)], dtype=torch.bool, device=self.device  # 创建表示生成是否完成的张量
        )

        if not isinstance(num_beams, int) or num_beams <= 1:  # 检查 num_beams 是否为大于1的整数
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
        return self._done.all()  # 返回是否所有生成操作均完成的布尔值

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
    ):  # 定义一个处理生成过程的方法，接受多个参数

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
    ):  # 定义一个完成生成过程的方法，接受多个参数
    # 定义一个新的类 `ConstrainedBeamSearchScorer`，继承自 `BeamScorer` 类
    r"""
    [`BeamScorer`] implementing constrained beam search decoding.
    实现受限束搜索解码的 [`BeamScorer`]。
    

    Args:
        batch_size (`int`):
            Batch Size of `input_ids` for which standard beam search decoding is run in parallel.
            输入 `input_ids` 的批处理大小，用于并行运行标准的束搜索解码。
        num_beams (`int`):
            Number of beams for beam search.
            束搜索的束数。
        constraints (`List[Constraint]`):
            A list of positive constraints represented as `Constraint` objects that must be fulfilled in the generation
            output. For more information, the documentation of [`Constraint`] should be read.
            表示为 `Constraint` 对象的正约束列表，必须在生成的输出中满足。有关更多信息，请阅读 [`Constraint`] 的文档。
        device (`torch.device`):
            Defines the device type (*e.g.*, `"cpu"` or `"cuda"`) on which this instance of `BeamSearchScorer` will be
            allocated.
            定义此 `BeamSearchScorer` 实例将分配到的设备类型（例如 `"cpu"` 或 `"cuda"`）。
        length_penalty (`float`, *optional*, defaults to 1.0):
            Exponential penalty to the length that is used with beam-based generation. It is applied as an exponent to
            the sequence length, which in turn is used to divide the score of the sequence. Since the score is the log
            likelihood of the sequence (i.e. negative), `length_penalty` > 0.0 promotes longer sequences, while
            `length_penalty` < 0.0 encourages shorter sequences.
            用于基于束的生成的长度的指数惩罚。它作为序列长度的指数应用，进而用于分割序列的分数。由于分数是序列的对数似然（即负数），`length_penalty` > 0.0 促进更长的序列，而 `length_penalty` < 0.0 鼓励更短的序列。
        do_early_stopping (`bool` or `str`, *optional*, defaults to `False`):
            Controls the stopping condition for beam-based methods, like beam-search. It accepts the following values:
            `True`, where the generation stops as soon as there are `num_beams` complete candidates; `False`, where an
            heuristic is applied and the generation stops when is it very unlikely to find better candidates;
            `"never"`, where the beam search procedure only stops when there cannot be better candidates (canonical
            beam search algorithm).
            控制基于束的方法（如束搜索）的停止条件。它接受以下值：`True`，生成在有 `num_beams` 个完整候选时停止；`False`，应用启发式并在很不可能找到更好的候选时停止生成；`"never"`，束搜索过程仅在不能有更好的候选时停止（经典的束搜索算法）。
        num_beam_hyps_to_keep (`int`, *optional*, defaults to 1):
            The number of beam hypotheses that shall be returned upon calling
            [`~transformers.BeamSearchScorer.finalize`].
            在调用 [`~transformers.BeamSearchScorer.finalize`] 时将返回的束假设数。
        num_beam_groups (`int`, *optional*, defaults to 1):
            Number of groups to divide `num_beams` into in order to ensure diversity among different groups of beams.
            See [this paper](https://arxiv.org/pdf/1610.02424.pdf) for more details.
            为了确保不同组的束之间的多样性，将 `num_beams` 分成的组数。有关更多详细信息，请参见 [此文献](https://arxiv.org/pdf/1610.02424.pdf)。
        max_length (`int`, *optional*):
            The maximum length of the sequence to be generated.
            要生成的序列的最大长度。
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
        ):
        # 初始化 BeamSearch 类的实例
        self.num_beams = num_beams
        self.device = device
        self.length_penalty = length_penalty
        self.do_early_stopping = do_early_stopping
        self.num_beam_hyps_to_keep = num_beam_hyps_to_keep
        self.num_beam_groups = num_beam_groups
        self.group_size = self.num_beams // self.num_beam_groups
        self.constraints = constraints

        self._is_init = False
        # 初始化 `_beam_hyps` 属性，存储 BeamHypotheses 的列表
        self._beam_hyps = [
            BeamHypotheses(
                num_beams=self.num_beams,
                length_penalty=self.length_penalty,
                early_stopping=self.do_early_stopping,
                max_length=max_length,
            )
            for _ in range(batch_size)
        ]
        # 初始化 `_done` 属性为 torch tensor，表示是否完成的状态
        self._done = torch.tensor([False for _ in range(batch_size)], dtype=torch.bool, device=self.device)

        # 检查 `num_beams` 是否是正整数且大于 1，否则抛出异常
        if not isinstance(num_beams, int) or num_beams <= 1:
            raise ValueError(
                f"`num_beams` has to be an integer strictly greater than 1, but is {num_beams}. For `num_beams` == 1,"
                " one should make use of `greedy_search` instead."
            )

        # 检查 `num_beam_groups` 是否是正整数且满足条件，否则抛出异常
        if not isinstance(num_beam_groups, int) or (num_beam_groups > num_beams) or (num_beams % num_beam_groups != 0):
            raise ValueError(
                "`num_beam_groups` has to be an integer smaller or equal than `num_beams` and `num_beams` has to be"
                f" divisible by `num_beam_groups`, but is {num_beam_groups} with `num_beams` being {num_beams}."
            )

    @property
    def is_done(self) -> bool:
        # 返回 `_done` 属性是否全部为 True
        return self._done.all()

    def make_constraint_states(self, n):
        # 根据约束条件创建状态列表的实例，返回列表
        return [ConstraintListState([constraint.copy() for constraint in self.constraints]) for _ in range(n)]

    def check_completes_constraints(self, sequence):
        # 创建约束状态的实例，并重置为给定的序列，返回是否完成的布尔值
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
        # 处理 beam search 的每个步骤，计算下一个可能的 token
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
        # 执行句子级别的约束步骤，更新相关的输入状态
        ...
    # 定义一个方法 finalize，用于处理束搜索的结果并生成最终的输出序列
    def finalize(
        self,
        # 输入的 token IDs，是一个 LongTensor
        input_ids: torch.LongTensor,
        # 最终束搜索得分，是一个 FloatTensor
        final_beam_scores: torch.FloatTensor,
        # 最终的束搜索 token 序列，是一个 LongTensor
        final_beam_tokens: torch.LongTensor,
        # 最终的束搜索索引，是一个 LongTensor，指示每个最终结果的束索引
        final_beam_indices: torch.LongTensor,
        # 最大生成长度，一个整数值
        max_length: int,
        # 填充 token 的 ID，可选参数，默认为 None
        pad_token_id: Optional[int] = None,
        # 结束 token 的 ID，可以是一个整数或整数列表，可选参数，默认为 None
        eos_token_id: Optional[Union[int, List[int]]] = None,
        # 生成结果时每个 token 序列对应的束索引，可选的 LongTensor，默认为 None
        beam_indices: Optional[torch.LongTensor] = None,
        # 解码器提示长度，可选的整数，默认为 0
        decoder_prompt_len: Optional[int] = 0,
# 定义一个类 BeamHypotheses，用于存储 Beam Search 算法生成的假设列表
class BeamHypotheses:
    # 初始化方法，设置各种参数和初始值
    def __init__(self, num_beams: int, length_penalty: float, early_stopping: bool, max_length: Optional[int] = None):
        """
        Initialize n-best list of hypotheses.

        Args:
            num_beams (int): Beam size, i.e., number of beams to keep.
            length_penalty (float): Length penalty to be applied to scores.
            early_stopping (bool): Whether to stop generation early based on conditions.
            max_length (Optional[int]): Optional maximum length for generated hypotheses.
        """
        self.length_penalty = length_penalty  # 设置长度惩罚参数
        self.early_stopping = early_stopping  # 是否启用提前停止
        self.max_length = max_length  # 最大生成长度限制
        self.num_beams = num_beams  # Beam 的数量
        self.beams = []  # 用于存储假设的列表
        self.worst_score = 1e9  # 初始设置一个极大值作为最差分数的初始值

        # 检查 early_stopping 参数类型，如果不是布尔值且 max_length 未定义，则引发错误
        if not isinstance(self.early_stopping, bool) and self.max_length is None:
            raise ValueError(
                "When `do_early_stopping` is set to a string, `max_length` must be defined. Ensure it is passed to the"
                " BeamScorer class instance at initialization time."
            )

    # 返回当前假设列表中假设的数量
    def __len__(self):
        """
        Number of hypotheses in the list.
        """
        return len(self.beams)

    # 向假设列表中添加新的假设
    def add(
        self,
        hyp: torch.LongTensor,
        sum_logprobs: float,
        beam_indices: Optional[torch.LongTensor] = None,
        generated_len: Optional[int] = None,
    ):
        """
        Add a new hypothesis to the list.

        Args:
            hyp (torch.LongTensor): Tensor representing the hypothesis.
            sum_logprobs (float): Sum of log probabilities associated with the hypothesis.
            beam_indices (Optional[torch.LongTensor]): Optional tensor of beam indices.
            generated_len (Optional[int]): Optional length of the generated sequence.
        """
        # 根据生成的序列长度或者假设的最后一个维度计算得分
        if generated_len is not None:
            score = sum_logprobs / (generated_len**self.length_penalty)
        else:
            score = sum_logprobs / (hyp.shape[-1] ** self.length_penalty)

        # 如果假设列表中假设数量小于 Beam 数量或者当前分数大于最差分数，则添加新假设
        if len(self) < self.num_beams or score > self.worst_score:
            self.beams.append((score, hyp, beam_indices))
            # 如果假设列表超过了 Beam 数量，则删除分数最低的假设
            if len(self) > self.num_beams:
                sorted_next_scores = sorted([(s, idx) for idx, (s, _, _) in enumerate(self.beams)])
                del self.beams[sorted_next_scores[0][1]]
                self.worst_score = sorted_next_scores[1][0]
            else:
                self.worst_score = min(score, self.worst_score)
    def is_done(self, best_sum_logprobs: float, cur_len: int, decoder_prompt_len: Optional[int] = 0) -> bool:
        """
        If there are enough hypotheses and that none of the hypotheses being generated can become better than the worst
        one in the heap, then we are done with this sentence.
        """

        # 如果当前堆中的假设数量小于要求的最大堆大小（num_beams），则返回 False
        if len(self) < self.num_beams:
            return False

        # 如果设定了 early_stopping 为 True，则立即停止，即使未满足其他条件
        if self.early_stopping is True:
            return True
        
        # 如果 early_stopping 设为 False，则根据当前长度计算最高可达分数，并检查是否达到最低分数标准
        elif self.early_stopping is False:
            highest_attainable_score = best_sum_logprobs / (cur_len - decoder_prompt_len) ** self.length_penalty
            ret = self.worst_score >= highest_attainable_score
            return ret
        
        # 如果 early_stopping 设为 "never"，则根据 length_penalty 的值计算最高可达分数
        else:
            # 当 length_penalty 大于 0.0 时，从 max_length 而不是 cur_len 计算最高可达分数
            if self.length_penalty > 0.0:
                if self.max_length <= decoder_prompt_len:
                    raise ValueError("max_length is not larger than decoder prompt length")
                highest_attainable_score = (
                    best_sum_logprobs / (self.max_length - decoder_prompt_len) ** self.length_penalty
                )
            # 当 length_penalty 小于等于 0.0 时，从 cur_len 计算最高可达分数
            else:
                highest_attainable_score = best_sum_logprobs / (cur_len - decoder_prompt_len) ** self.length_penalty
            
            ret = self.worst_score >= highest_attainable_score
            return ret
```