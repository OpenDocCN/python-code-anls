# `D:\src\scipysrc\matplotlib\lib\matplotlib\__init__.pyi`

```py
# 定义一个列表，包含了模块中需要被公开的符号名称
__all__ = [
    "__bibtex__",
    "__version__",
    "__version_info__",
    "set_loglevel",
    "ExecutableNotFoundError",
    "get_configdir",
    "get_cachedir",
    "get_data_path",
    "matplotlib_fname",
    "MatplotlibDeprecationWarning",
    "RcParams",
    "rc_params",
    "rc_params_from_file",
    "rcParamsDefault",
    "rcParams",
    "rcParamsOrig",
    "defaultParams",
    "rc",
    "rcdefaults",
    "rc_file_defaults",
    "rc_file",
    "rc_context",
    "use",
    "get_backend",
    "interactive",
    "is_interactive",
    "colormaps",
    "color_sequences",
]

# 导入标准库模块
import os
# 导入路径操作相关的路径类
from pathlib import Path

# 导入集合抽象类中的可调用对象和生成器
from collections.abc import Callable, Generator
# 导入上下文管理模块
import contextlib
# 从打包版本模块中导入版本类
from packaging.version import Version

# 导入 Matplotlib 库中的警告类
from matplotlib._api import MatplotlibDeprecationWarning
# 导入类型提示模块中的任意类型和命名元组
from typing import Any, NamedTuple

# 定义模块级别的变量，并注明其类型
__bibtex__: str
__version__: str
__version_info__: _VersionInfo

# 定义一个设置日志级别的函数，参数为字符串类型，无返回值
def set_loglevel(level: str) -> None: ...

# 定义一个命名元组，表示执行信息，包含可执行文件名、原始版本字符串和版本对象
class _ExecInfo(NamedTuple):
    executable: str
    raw_version: str
    version: Version

# 定义一个自定义异常类，继承自 FileNotFoundError
class ExecutableNotFoundError(FileNotFoundError): ...

# 定义一个内部函数，返回执行信息的命名元组，参数为名称字符串，返回类型为 _ExecInfo
def _get_executable_info(name: str) -> _ExecInfo: ...

# 定义获取配置目录路径的函数，返回字符串类型
def get_configdir() -> str: ...

# 定义获取缓存目录路径的函数，返回字符串类型
def get_cachedir() -> str: ...

# 定义获取数据路径的函数，返回字符串类型
def get_data_path() -> str: ...

# 定义获取 Matplotlib 配置文件路径的函数，返回字符串类型
def matplotlib_fname() -> str: ...

# 定义一个继承自字典的类 RcParams，键为字符串类型，值为任意类型
class RcParams(dict[str, Any]):
    validate: dict[str, Callable]
    def __init__(self, *args, **kwargs) -> None: ...
    def _set(self, key: str, val: Any) -> None: ...
    def _get(self, key: str) -> Any: ...
    def __setitem__(self, key: str, val: Any) -> None: ...
    def __getitem__(self, key: str) -> Any: ...
    def __iter__(self) -> Generator[str, None, None]: ...
    def __len__(self) -> int: ...
    def find_all(self, pattern: str) -> RcParams: ...
    def copy(self) -> RcParams: ...

# 定义一个返回 RcParams 对象的函数，可选参数为错误标志，返回类型为 RcParams
def rc_params(fail_on_error: bool = ...) -> RcParams: ...

# 定义一个从文件中读取 RcParams 对象的函数，参数为文件名或路径，错误标志和使用默认模板标志，返回类型为 RcParams
def rc_params_from_file(
    fname: str | Path | os.PathLike,
    fail_on_error: bool = ...,
    use_default_template: bool = ...,
) -> RcParams: ...

# 定义默认的 RcParams 对象
rcParamsDefault: RcParams

# 定义当前 RcParams 对象
rcParams: RcParams

# 定义原始 RcParams 对象
rcParamsOrig: RcParams

# 定义默认参数字典
defaultParams: dict[str, Any]

# 定义一个配置参数的函数，参数为组名和关键字参数，无返回值
def rc(group: str, **kwargs) -> None: ...

# 定义一个恢复默认配置的函数，无参数和返回值
def rcdefaults() -> None: ...

# 定义一个恢复默认配置文件的函数，无参数和返回值
def rc_file_defaults() -> None: ...

# 定义一个从文件中加载配置的函数，参数为文件名或路径和是否使用默认模板，无返回值
def rc_file(
    fname: str | Path | os.PathLike, *, use_default_template: bool = ...
) -> None: ...

# 定义一个上下文管理器，用于配置上下文，参数为配置字典或 None 和文件名或路径或 None，返回生成器类型
@contextlib.contextmanager
def rc_context(
    rc: dict[str, Any] | None = ..., fname: str | Path | os.PathLike | None = ...
) -> Generator[None, None, None]: ...

# 定义一个设置后端的函数，参数为后端名称和是否强制，无返回值
def use(backend: str, *, force: bool = ...) -> None: ...

# 定义一个获取当前后端名称的函数，返回字符串类型
def get_backend() -> str: ...

# 定义一个设置交互模式的函数，参数为布尔类型，无返回值
def interactive(b: bool) -> None: ...

# 定义一个判断是否为交互模式的函数，返回布尔类型
def is_interactive() -> bool: ...

# 定义一个数据预处理的函数，参数为可调用对象或 None，替换名称列表或 None 和标签命名器字符串或 None，返回类型为可调用对象
def _preprocess_data(
    func: Callable | None = ...,
    *,
    replace_names: list[str] | None = ...,
    label_namer: str | None = ...
) -> Callable: ...

# 从 Matplotlib 的颜色映射模块中导入颜色映射
from matplotlib.cm import _colormaps as colormaps
# 从 Matplotlib 的颜色模块中导入颜色序列
from matplotlib.colors import _color_sequences as color_sequences
```