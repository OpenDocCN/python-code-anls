# `D:\src\scipysrc\matplotlib\lib\matplotlib\testing\__init__.pyi`

```
# 导入所需模块和类型定义
from collections.abc import Callable
import subprocess
from typing import Any, IO, Literal, overload

# 定义三个占位函数，没有具体实现，返回类型为 None
def set_font_settings_for_testing() -> None: ...
def set_reproducibility_for_testing() -> None: ...
def setup() -> None: ...

# 定义三个重载函数，用于执行子进程命令并返回结果
@overload
def subprocess_run_for_testing(
    command: list[str],
    env: dict[str, str] | None = ...,
    timeout: float | None = ...,
    stdout: int | IO[Any] | None = ...,
    stderr: int | IO[Any] | None = ...,
    check: bool = ...,
    *,
    text: Literal[True],
    capture_output: bool = ...,
) -> subprocess.CompletedProcess[str]: ...

@overload
def subprocess_run_for_testing(
    command: list[str],
    env: dict[str, str] | None = ...,
    timeout: float | None = ...,
    stdout: int | IO[Any] | None = ...,
    stderr: int | IO[Any] | None = ...,
    check: bool = ...,
    text: Literal[False] = ...,
    capture_output: bool = ...,
) -> subprocess.CompletedProcess[bytes]: ...

@overload
def subprocess_run_for_testing(
    command: list[str],
    env: dict[str, str] | None = ...,
    timeout: float | None = ...,
    stdout: int | IO[Any] | None = ...,
    stderr: int | IO[Any] | None = ...,
    check: bool = ...,
    text: bool = ...,
    capture_output: bool = ...,
) -> subprocess.CompletedProcess[bytes] | subprocess.CompletedProcess[str]: ...

# 辅助函数，用于调用给定函数并在子进程中执行，返回一个 subprocess.CompletedProcess[str] 对象
def subprocess_run_helper(
    func: Callable[[], None],
    *args: Any,
    timeout: float,
    extra_env: dict[str, str] | None = ...,
) -> subprocess.CompletedProcess[str]: ...

# 检查是否存在指定的 TeX 程序
def _check_for_pgf(texsystem: str) -> bool: ...

# 检查是否安装了指定的 TeX 包
def _has_tex_package(package: str) -> bool: ...

# 在子进程中运行 IPython 并指定请求的后端或 GUI 框架
def ipython_in_subprocess(
    requested_backend_or_gui_framework: str,
    all_expected_backends: dict[tuple[int, int], str],
) -> None: ...
```