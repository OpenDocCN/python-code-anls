# `D:\src\scipysrc\numpy\numpy\_pytesttester.pyi`

```py
from collections.abc import Iterable
from typing import Literal as L

# 定义一个列表，用于指定模块的公开接口
__all__: list[str]

# 定义一个名为 PytestTester 的类
class PytestTester:
    # 类属性，存储模块名
    module_name: str

    # 类的初始化方法，接受一个模块名参数
    def __init__(self, module_name: str) -> None: ...

    # 类的调用方法，用于执行测试
    def __call__(
        self,
        label: L["fast", "full"] = ...,
        verbose: int = ...,
        extra_argv: None | Iterable[str] = ...,
        doctests: L[False] = ...,
        coverage: bool = ...,
        durations: int = ...,
        tests: None | Iterable[str] = ...,
    ) -> bool: ...
```