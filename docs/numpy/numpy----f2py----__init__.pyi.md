# `.\numpy\numpy\f2py\__init__.pyi`

```py
# 导入标准库模块
import os
# 导入 subprocess 模块，用于执行外部命令
import subprocess
# 导入 Iterable 类型用于类型提示
from collections.abc import Iterable
# 导入 Literal 类型别名用于类型提示
from typing import Literal as L, Any, overload, TypedDict

# 导入 numpy._pytesttester 模块中的 PytestTester 类
from numpy._pytesttester import PytestTester

# 定义 _F2PyDictBase 类型字典，包含 csrc 和 h 两个键
class _F2PyDictBase(TypedDict):
    csrc: list[str]
    h: list[str]

# 定义 _F2PyDict 类型字典，继承自 _F2PyDictBase，可选包含 fsrc 和 ltx 两个键
class _F2PyDict(_F2PyDictBase, total=False):
    fsrc: list[str]
    ltx: list[str]

# __all__ 列表，用于定义模块导出的所有公共符号
__all__: list[str]

# test 变量，类型为 PytestTester 类
test: PytestTester

# run_main 函数定义，接受 comline_list 参数（可迭代对象），返回字典，键为 str 类型，值为 _F2PyDict 类型
def run_main(comline_list: Iterable[str]) -> dict[str, _F2PyDict]: ...

# compile 函数的第一个重载定义，接受多个参数并返回 int 类型结果
@overload
def compile(  # type: ignore[misc]
    source: str | bytes,
    modulename: str = ...,
    extra_args: str | list[str] = ...,
    verbose: bool = ...,
    source_fn: None | str | bytes | os.PathLike[Any] = ...,
    extension: L[".f", ".f90"] = ...,
    full_output: L[False] = ...,
) -> int: ...

# compile 函数的第二个重载定义，接受多个参数并返回 subprocess.CompletedProcess[bytes] 类型结果
@overload
def compile(
    source: str | bytes,
    modulename: str = ...,
    extra_args: str | list[str] = ...,
    verbose: bool = ...,
    source_fn: None | str | bytes | os.PathLike[Any] = ...,
    extension: L[".f", ".f90"] = ...,
    full_output: L[True] = ...,
) -> subprocess.CompletedProcess[bytes]: ...

# get_include 函数定义，返回 str 类型结果，用于获取包含文件的路径
def get_include() -> str: ...
```