# `.\generation\stopping_criteria.py`

```
# 导入时间模块，用于处理时间相关功能
import time
# 导入警告模块，用于发出警告信息
import warnings
# 导入抽象基类模块，用于定义抽象类
from abc import ABC
# 导入深拷贝函数，用于创建对象的深层副本
from copy import deepcopy
# 导入类型提示模块，用于指定参数和返回值的类型
from typing import Optional

# 导入PyTorch库
import torch

# 从本地utils模块中导入指定函数和类
from ..utils import add_start_docstrings, logging

# 从logging模块中获取当前模块的日志记录器
logger = logging.get_logger(__name__)

# 停止条件的文档字符串，使用原始字符串表示（r"..."），包含参数和返回值的描述
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
        `torch.BoolTensor`. (`torch.BoolTensor` of shape `(batch_size, 1)`), where `True` indicates we stop generation
            for a particular row, `True` indicates we should continue.

"""


class StoppingCriteria(ABC):
    """Abstract base class for all stopping criteria that can be applied during generation.

    If your stopping criteria depends on the `scores` input, make sure you pass `return_dict_in_generate=True,
    output_scores=True` to `generate`.
    """

    @add_start_docstrings(STOPPING_CRITERIA_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> torch.BoolTensor:
        # 抽象方法，子类需实现该方法来定义停止生成的具体逻辑
        raise NotImplementedError("StoppingCriteria needs to be subclassed")


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

    def __init__(self, max_length: int, max_position_embeddings: Optional[int] = None):
        # 初始化最大长度和最大位置嵌入
        self.max_length = max_length
        self.max_position_embeddings = max_position_embeddings

    @add_start_docstrings(STOPPING_CRITERIA_INPUTS_DOCSTRING)
    # 定义一个调用函数，用于生成文本序列的逻辑
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> torch.BoolTensor:
        # 获取当前输入序列的长度
        cur_len = input_ids.shape[-1]
        # 检查当前序列长度是否已经达到或超过最大生成长度
        is_done = cur_len >= self.max_length
        # 如果模型限制了最大位置嵌入数量且当前长度未达到生成上限，并且当前长度已经超过最大位置嵌入数量，则发出警告
        if self.max_position_embeddings is not None and not is_done and cur_len >= self.max_position_embeddings:
            logger.warning_once(
                "This is a friendly reminder - the current text generation call will exceed the model's predefined "
                f"maximum length ({self.max_position_embeddings}). Depending on the model, you may observe "
                "exceptions, performance degradation, or nothing at all."
            )
        # 返回一个布尔张量，表示每个输入序列是否已完成生成
        return torch.full((input_ids.shape[0],), is_done, device=input_ids.device, dtype=torch.bool)
# 继承自 `StoppingCriteria` 类的子类 `MaxNewTokensCriteria`，用于在生成的标记数超过 `max_new_tokens` 时停止生成。
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

    # 初始化方法，发出警告信息表明该类已被弃用，建议使用 `MaxLengthCriteria` 替代
    def __init__(self, start_length: int, max_new_tokens: int):
        warnings.warn(
            "The class `MaxNewTokensCriteria` is deprecated. "
            f"Please use `MaxLengthCriteria(max_length={start_length + max_new_tokens})` "
            "with `max_length = start_length + max_new_tokens` instead.",
            FutureWarning,
        )
        # 初始化属性，记录初始标记数和允许生成的最大标记数
        self.start_length = start_length
        self.max_new_tokens = max_new_tokens
        self.max_length = start_length + max_new_tokens

    # 调用对象时的方法，检查是否达到生成的最大标记数
    @add_start_docstrings(STOPPING_CRITERIA_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> torch.BoolTensor:
        # 判断输入标记的长度是否大于等于设定的最大长度
        is_done = input_ids.shape[-1] >= self.max_length
        return torch.full((input_ids.shape[0],), is_done, device=input_ids.device, dtype=torch.bool)


# 继承自 `StoppingCriteria` 类的子类 `MaxTimeCriteria`，用于在生成时间超过 `max_time` 秒时停止生成。
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

    # 初始化方法，记录最大允许生成时间和开始计时的时间戳（默认为当前时间）
    def __init__(self, max_time: float, initial_timestamp: Optional[float] = None):
        self.max_time = max_time
        self.initial_timestamp = time.time() if initial_timestamp is None else initial_timestamp

    # 调用对象时的方法，检查是否超过了允许的生成时间
    @add_start_docstrings(STOPPING_CRITERIA_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> torch.BoolTensor:
        # 计算当前时间与初始时间戳之间的差值，判断是否超过了最大允许时间
        is_done = time.time() - self.initial_timestamp > self.max_time
        return torch.full((input_ids.shape[0],), is_done, device=input_ids.device, dtype=torch.bool)


# 继承自列表的子类 `StoppingCriteriaList`，用于存储多个停止生成的条件，并在任何一个条件满足时停止生成。
class StoppingCriteriaList(list):
    @add_start_docstrings(STOPPING_CRITERIA_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> torch.BoolTensor:
        # 初始化一个全为 False 的 torch.BoolTensor，表示生成未完成
        is_done = torch.full((input_ids.shape[0],), False, device=input_ids.device)
        # 遍历存储的所有停止条件，如果任何一个条件返回 True，则更新 is_done 为 True
        for criteria in self:
            is_done = is_done | criteria(input_ids, scores, **kwargs)
        return is_done
    # 定义一个方法 `max_length`，返回类型是可选的整数（可能为None）
    def max_length(self) -> Optional[int]:
        # 遍历当前对象实例中的每一个停止条件
        for stopping_criterium in self:
            # 如果当前停止条件是 `MaxLengthCriteria` 类型的实例
            if isinstance(stopping_criterium, MaxLengthCriteria):
                # 返回 `MaxLengthCriteria` 实例中定义的最大长度
                return stopping_criterium.max_length
            # 如果当前停止条件是 `MaxNewTokensCriteria` 类型的实例
            elif isinstance(stopping_criterium, MaxNewTokensCriteria):
                # 返回 `MaxNewTokensCriteria` 实例中定义的最大长度
                return stopping_criterium.max_length
        # 如果没有找到符合条件的停止条件，返回 None
        return None
# 定义一个函数，用于验证停止条件列表是否符合规范，并返回更新后的停止条件列表对象
def validate_stopping_criteria(stopping_criteria: StoppingCriteriaList, max_length: int) -> StoppingCriteriaList:
    # 获取停止条件列表中的最大长度
    stopping_max_length = stopping_criteria.max_length
    # 深度复制原始的停止条件列表对象，以免修改原始数据
    new_stopping_criteria = deepcopy(stopping_criteria)
    
    # 如果停止条件列表中的最大长度存在，并且与传入的 max_length 参数不相等
    if stopping_max_length is not None and stopping_max_length != max_length:
        # 发出警告，指出设置的停止条件最大长度与传入参数的最大长度不一致
        warnings.warn("You set different `max_length` for stopping criteria and `max_length` parameter", UserWarning)
    # 如果停止条件列表中的最大长度不存在
    elif stopping_max_length is None:
        # 向新的停止条件列表中添加一个新的最大长度停止条件对象
        new_stopping_criteria.append(MaxLengthCriteria(max_length=max_length))
    
    # 返回更新后的停止条件列表对象
    return new_stopping_criteria
```