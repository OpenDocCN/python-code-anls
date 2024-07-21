# `.\pytorch\torch\distributed\checkpoint\api.py`

```py
# mypy: allow-untyped-defs
# 导入 traceback 模块并重命名为 tb
import traceback as tb
# 导入 Any、Dict、Tuple 类型提示
from typing import Any, Dict, Tuple

# 定义一个元组类型 WRAPPED_EXCEPTION，用于包装异常和堆栈信息
WRAPPED_EXCEPTION = Tuple[BaseException, tb.StackSummary]

# 将 CheckpointException 添加到 __all__ 列表中，表示它是模块的公共接口之一
__all__ = ["CheckpointException"]

# 定义一个函数 _wrap_exception，用于将 BaseException 包装为包含堆栈信息的元组
def _wrap_exception(exc: BaseException) -> WRAPPED_EXCEPTION:
    return (exc, tb.extract_tb(exc.__traceback__))

# 定义一个函数 _is_wrapped_exception，用于检查对象是否为包含正确类型和长度的元组
def _is_wrapped_exception(obj: Any) -> bool:
    if not isinstance(obj, tuple):
        return False
    if len(obj) != 2:
        return False
    return isinstance(obj[0], BaseException) and isinstance(obj[1], tb.StackSummary)

# 定义 CheckpointException 类，继承自 BaseException
class CheckpointException(BaseException):
    """Exception raised if failure was detected as part of a checkpoint load or save."""

    # 初始化方法，接受一个错误信息字符串 msg 和一个包含失败信息的字典 failures
    def __init__(self, msg: str, failures: Dict[int, WRAPPED_EXCEPTION]):
        # 调用父类 BaseException 的初始化方法
        super().__init__(msg, failures)
        # 保存传入的 failures 字典到实例属性 _failures 中
        self._failures = failures

    # failures 属性的 getter 方法，返回保存在实例中的失败信息字典
    @property
    def failures(self) -> Dict[int, WRAPPED_EXCEPTION]:
        """Return a dictionary mapping node ranks to their associated exceptions in case of failure."""
        return self._failures

    # __str__ 方法，返回异常的字符串表示形式
    def __str__(self):
        # 初始化返回字符串 str，包含失败信息字典的键（节点排名）
        str = f"CheckpointException ranks:{self._failures.keys()}\n"
        # 遍历 failures 字典，处理每个节点排名及其关联的异常信息
        for rank, exc_pair in self._failures.items():
            exc, trace = exc_pair
            # 添加每个排名的异常信息和堆栈回溯（如果有）
            str += f"Traceback (most recent call last): (RANK {rank})\n"
            if trace is not None:
                str += "".join(tb.format_list(trace))
            str += "".join(tb.format_exception_only(type(exc), value=exc))
        return str
```