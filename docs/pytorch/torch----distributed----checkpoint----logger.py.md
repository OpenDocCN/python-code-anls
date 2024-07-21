# `.\pytorch\torch\distributed\checkpoint\logger.py`

```
# mypy: allow-untyped-defs
# 导入 functools 模块，用于高阶函数的支持
import functools
# 导入 time 模块，用于时间相关操作
import time
# 导入类型提示相关模块
from typing import Any, Callable, Dict, List, TypeVar
from typing_extensions import ParamSpec

# 导入分布式日志记录模块
import torch.distributed.c10d_logger as c10d_logger
# 从 torch.distributed.checkpoint.logging_handlers 导入 DCP_LOGGER_NAME 常量
from torch.distributed.checkpoint.logging_handlers import DCP_LOGGER_NAME

# 空列表，用于存放导出的所有名称
__all__: List[str] = []

# 全局变量 _dcp_logger，用于存放获取或创建的分布式日志记录器
global _dcp_logger
_dcp_logger = c10d_logger._get_or_create_logger(DCP_LOGGER_NAME)

# 定义类型变量 _T 和 ParamSpec 参数规范 _P

_T = TypeVar("_T")
_P = ParamSpec("_P")

# 函数 _msg_dict_from_dcp_method_args，从 dcp 方法参数中提取日志数据字典
def _msg_dict_from_dcp_method_args(*args, **kwargs) -> Dict[str, Any]:
    """
    Extracts log data from dcp method args
    """
    msg_dict = {}

    # 从 kwargs 中获取 storage_writer, storage_reader 和 planner
    storage_writer = kwargs.get("storage_writer", None)
    storage_reader = kwargs.get("storage_reader", None)
    planner = kwargs.get("planner", None)

    # 从 kwargs 中获取 checkpoint_id
    checkpoint_id = kwargs.get("checkpoint_id", None)
    # 如果 checkpoint_id 不存在并且 serializer 存在，则尝试从 serializer 中获取 checkpoint_id
    if not checkpoint_id and (serializer := storage_writer or storage_reader):
        checkpoint_id = getattr(serializer, "checkpoint_id", None)

    # 将 checkpoint_id 转换为字符串并存入 msg_dict
    msg_dict["checkpoint_id"] = (
        str(checkpoint_id) if checkpoint_id is not None else checkpoint_id
    )

    # 如果 storage_writer 存在，将其类名存入 msg_dict
    if storage_writer:
        msg_dict["storage_writer"] = storage_writer.__class__.__name__

    # 如果 storage_reader 存在，将其类名存入 msg_dict
    if storage_reader:
        msg_dict["storage_reader"] = storage_reader.__class__.__name__

    # 如果 planner 存在，将其类名存入 msg_dict
    if planner:
        msg_dict["planner"] = planner.__class__.__name__

    return msg_dict

# 函数 _get_msg_dict，获取函数名及其参数，返回完整的日志数据字典
def _get_msg_dict(func_name, *args, **kwargs) -> Dict[str, Any]:
    msg_dict = _msg_dict_from_dcp_method_args(*args, **kwargs)
    msg_dict.update(c10d_logger._get_msg_dict(func_name, **msg_dict))

    return msg_dict

# 装饰器函数 _dcp_method_logger，用于日志记录装饰器，记录包装事件的开始、结束和异常
def _dcp_method_logger(
    log_exceptions: bool = False, **wrapper_kwargs: Any
) -> Callable[[Callable[_P, _T]], Callable[_P, _T]]:  # pyre-ignore
    """This method decorator logs the start, end, and exception of wrapped events."""

    def decorator(func: Callable[_P, _T]):
        @functools.wraps(func)
        def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _T:
            # 获取函数调用的详细日志数据字典
            msg_dict = _get_msg_dict(
                func.__name__, *args, **{**wrapper_kwargs, **kwargs}
            )

            # 记录开始事件
            msg_dict["event"] = "start"
            t0 = time.time_ns()  # 记录开始时间戳
            msg_dict["time"] = t0
            _dcp_logger.debug(msg_dict)  # 使用分布式日志记录器记录调试信息

            # 异常处理
            try:
                result = func(*args, **kwargs)  # 执行被装饰的函数
            except Exception as error:
                # 如果允许记录异常，则记录异常事件
                if log_exceptions:
                    msg_dict["event"] = "exception"
                    msg_dict["error"] = f"{error}"
                    msg_dict["time"] = time.time_ns()
                    _dcp_logger.error(msg_dict)  # 使用分布式日志记录器记录错误信息
                raise

            # 记录结束事件
            msg_dict["event"] = "end"
            t1 = time.time_ns()  # 记录结束时间戳
            msg_dict["time"] = time.time_ns()
            msg_dict["times_spent"] = t1 - t0  # 计算事件持续时间
            _dcp_logger.debug(msg_dict)  # 使用分布式日志记录器记录调试信息

            return result

        return wrapper

    return decorator
    # 返回装饰器函数对象
    return decorator
```