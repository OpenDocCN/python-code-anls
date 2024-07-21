# `.\pytorch\torch\utils\benchmark\utils\_stubs.py`

```
from typing import Any, Callable, Dict, Protocol, runtime_checkable


# 定义 TimerClass 协议，用于描述 timeit.Timer API 的一部分
class TimerClass(Protocol):
    """This is the portion of the `timeit.Timer` API used by benchmark utils."""
    def __init__(
        self,
        stmt: str,
        setup: str,
        timer: Callable[[], float],
        globals: Dict[str, Any],
        **kwargs: Any,
    ) -> None:
        ...

    # 定义 timeit 方法，用于执行计时操作并返回时间
    def timeit(self, number: int) -> float:
        ...


# 定义 TimeitModuleType 协议，表示从 `timeit_template.cpp` 生成的模块
@runtime_checkable
class TimeitModuleType(Protocol):
    """Modules generated from `timeit_template.cpp`."""
    # 定义 timeit 方法，用于执行计时操作并返回时间
    def timeit(self, number: int) -> float:
        ...


# 定义 CallgrindModuleType 协议，模拟 `torch._C` 中的 valgrind 端点
class CallgrindModuleType(Protocol):
    """Replicates the valgrind endpoints in `torch._C`.

    These bindings are used to collect Callgrind profiles on earlier versions
    of PyTorch and will eventually be removed.
    """
    __file__: str  # 模块文件路径
    __name__: str  # 模块名称

    # 检查当前平台是否支持 valgrind
    def _valgrind_supported_platform(self) -> bool:
        ...

    # 开关 valgrind 收集器
    def _valgrind_toggle(self) -> None:
        ...
```