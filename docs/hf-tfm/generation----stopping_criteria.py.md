# `.\transformers\generation\stopping_criteria.py`

```
# 导入时间模块
import time
# 导入警告模块
import warnings
# 从 abc 模块导入 ABC 抽象基类
from abc import ABC
# 从 copy 模块导入 deepcopy 函数
from copy import deepcopy
# 从 typing 模块导入 Optional 类型提示
from typing import Optional
# 导入 torch 库
import torch

# 从上级目录中的 utils 模块中导入 add_start_docstrings 和 logging 函数
from ..utils import add_start_docstrings, logging

# 获取日志记录器对象
logger = logging.get_logger(__name__)

# 定义停止条件输入文档字符串
STOPPING_CRITERIA_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        scores (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`):
            Prediction scores of a language modeling head. These can be scores for each vocabulary token before SoftMax
            or scores for each vocabulary token after SoftMax. If this stopping criteria depends on the `scores` input,
            make sure you pass `return_dict_in_generate=True, output_scores=True` to `generate`.
        kwargs (`Dict[str, Any]`, *optional*):
            Additional stopping criteria specific kwargs.

    Return:
        `bool`. `False` indicates we should continue, `True` indicates we should stop.

"""


# 定义停止条件抽象基类
class StoppingCriteria(ABC):
    """Abstract base class for all stopping criteria that can be applied during generation.

    If your stopping criteria depends on the `scores` input, make sure you pass `return_dict_in_generate=True,
    output_scores=True` to `generate`.
    """

    # 添加文档字符串
    @add_start_docstrings(STOPPING_CRITERIA_INPUTS_DOCSTRING)
    # 定义抽象方法
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # 抛出未实现错误
        raise NotImplementedError("StoppingCriteria needs to be subclassed")


# 定义最大长度停止条件类
class MaxLengthCriteria(StoppingCriteria):
    """
    This class can be used to stop generation whenever the full generated number of tokens exceeds `max_length`. Keep
    in mind for decoder-only type of transformers, this will include the initial prompted tokens.

    Args:
        max_length (`int`):
            The maximum length that the output sequence can have in number of tokens.
        max_position_embeddings (`int`, *optional*):
            The maximum model length, as defined by the model's `config.max_position_embeddings` attribute.
    """

    # 定义初始化方法
    def __init__(self, max_length: int, max_position_embeddings: Optional[int] = None):
        # 最大长度
        self.max_length = max_length
        # 最大位置嵌入长度
        self.max_position_embeddings = max_position_embeddings

    # 添加文档字符串
    @add_start_docstrings(STOPPING_CRITERIA_INPUTS_DOCSTRING)
    # 定义一个方法，用于检查生成的文本是否达到了最大长度限制
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # 获取当前生成文本的长度
        cur_len = input_ids.shape[-1]
        # 检查当前生成文本长度是否已达到最大长度限制
        is_done = cur_len >= self.max_length
        # 如果设置了最大位置嵌入数，并且当前长度未达到最大长度限制且超过了最大位置嵌入数，则发出警告
        if self.max_position_embeddings is not None and not is_done and cur_len >= self.max_position_embeddings:
            logger.warning_once(
                "This is a friendly reminder - the current text generation call will exceed the model's predefined "
                f"maximum length ({self.max_position_embeddings}). Depending on the model, you may observe "
                "exceptions, performance degradation, or nothing at all."
            )
        # 返回是否已完成生成
        return is_done
class MaxNewTokensCriteria(StoppingCriteria):
    """
    This class can be used to stop generation whenever the generated number of tokens exceeds `max_new_tokens`. Keep in
    mind for decoder-only type of transformers, this will **not** include the initial prompted tokens. This is very
    close to `MaxLengthCriteria` but ignores the number of initial tokens.

    Args:
        start_length (`int`):
            The number of initial tokens.
        max_new_tokens (`int`):
            The maximum number of tokens to generate.
    """

    def __init__(self, start_length: int, max_new_tokens: int):
        warnings.warn(
            "The class `MaxNewTokensCriteria` is deprecated. "
            f"Please use `MaxLengthCriteria(max_length={start_length + max_new_tokens})` "
            "with `max_length = start_length + max_new_tokens` instead.",
            FutureWarning,
        )
        # 初始化函数，设置起始长度和最大新生成的标记数
        self.start_length = start_length
        self.max_new_tokens = max_new_tokens
        self.max_length = start_length + max_new_tokens

    @add_start_docstrings(STOPPING_CRITERIA_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # 判断输入的标记数是否超过最大长度
        return input_ids.shape[-1] >= self.max_length


class MaxTimeCriteria(StoppingCriteria):
    """
    This class can be used to stop generation whenever the full generation exceeds some amount of time. By default, the
    time will start being counted when you initialize this function. You can override this by passing an
    `initial_time`.

    Args:
        max_time (`float`):
            The maximum allowed time in seconds for the generation.
        initial_time (`float`, *optional*, defaults to `time.time()`):
            The start of the generation allowed time.
    """

    def __init__(self, max_time: float, initial_timestamp: Optional[float] = None):
        # 初始化函数，设置最大允许的时间和初始时间戳
        self.max_time = max_time
        self.initial_timestamp = time.time() if initial_timestamp is None else initial_timestamp

    @add_start_docstrings(STOPPING_CRITERIA_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # 判断当前时间与初始时间戳之间的差是否超过最大允许时间
        return time.time() - self.initial_timestamp > self.max_time


class StoppingCriteriaList(list):
    @add_start_docstrings(STOPPING_CRITERIA_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # 判断是否有任何一个停止标准满足条件
        return any(criteria(input_ids, scores) for criteria in self)

    @property
    def max_length(self) -> Optional[int]:
        # 获取停止标准列表中的最大长度
        for stopping_criterium in self:
            if isinstance(stopping_criterium, MaxLengthCriteria):
                return stopping_criterium.max_length
            elif isinstance(stopping_criterium, MaxNewTokensCriteria):
                return stopping_criterium.max_length
        return None


def validate_stopping_criteria(stopping_criteria: StoppingCriteriaList, max_length: int) -> StoppingCriteriaList:
    # 获取停止条件的最大长度
    stopping_max_length = stopping_criteria.max_length
    # 深拷贝停止条件对象
    new_stopping_criteria = deepcopy(stopping_criteria)
    # 如果停止条件的最大长度不为空且与给定的最大长度不相等，则发出警告
    if stopping_max_length is not None and stopping_max_length != max_length:
        warnings.warn("You set different `max_length` for stopping criteria and `max_length` parameter", UserWarning)
    # 如果停止条件的最大长度为空，则添加一个新的最大长度条件到停止条件中
    elif stopping_max_length is None:
        new_stopping_criteria.append(MaxLengthCriteria(max_length=max_length))
    # 返回更新后的停止条件
    return new_stopping_criteria
```