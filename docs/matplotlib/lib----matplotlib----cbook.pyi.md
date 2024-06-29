# `D:\src\scipysrc\matplotlib\lib\matplotlib\cbook.pyi`

```
# 导入 collections.abc 模块，包含了抽象基类，用于定义集合和可迭代对象的抽象接口
import collections.abc
# 从 collections.abc 模块中导入 Callable、Collection、Generator、Iterable、Iterator 抽象基类
from collections.abc import Callable, Collection, Generator, Iterable, Iterator
# 导入 contextlib 模块，提供了用于管理上下文的工具
import contextlib
# 导入 os 模块，提供了与操作系统交互的功能
import os
# 从 pathlib 模块中导入 Path 类，用于处理文件路径
from pathlib import Path

# 从 matplotlib.artist 模块中导入 Artist 类
from matplotlib.artist import Artist

# 导入 numpy 库
import numpy as np
# 从 numpy.typing 模块中导入 ArrayLike 类型
from numpy.typing import ArrayLike

# 导入 typing 模块中的各种类型提示相关内容
from typing import (
    Any,  # 任意类型
    Generic,  # 泛型
    IO,  # 输入输出流
    Literal,  # 字面值类型
    TypeVar,  # 类型变量
    overload,  # 函数重载装饰器
)

_T = TypeVar("_T")  # 定义类型变量 _T

# 定义一个函数 _get_running_interactive_framework，返回类型为 str 或 None
def _get_running_interactive_framework() -> str | None: ...

# 定义一个类 CallbackRegistry
class CallbackRegistry:
    exception_handler: Callable[[Exception], Any]  # 异常处理器，接收一个异常对象并返回任意类型
    callbacks: dict[Any, dict[int, Any]]  # 回调函数字典，键为任意类型，值为整数到任意类型的字典

    # 初始化方法
    def __init__(
        self,
        exception_handler: Callable[[Exception], Any] | None = ...,  # 异常处理器，可选参数，默认为 None
        *,
        signals: Iterable[Any] | None = ...,  # 信号，可迭代对象，可选参数，默认为 None
    ) -> None: ...

    # 连接方法，连接信号和回调函数
    def connect(self, signal: Any, func: Callable) -> int: ...

    # 断开连接方法，根据连接 ID 断开连接
    def disconnect(self, cid: int) -> None: ...

    # 处理方法，处理信号和对应的参数
    def process(self, s: Any, *args, **kwargs) -> None: ...

    # 被阻塞方法，返回一个用于管理阻塞状态的上下文管理器
    def blocked(
        self, *, signal: Any | None = ...  # 信号，可选参数，默认为 None
    ) -> contextlib.AbstractContextManager[None]: ...

# 定义一个 silent_list 类，继承自 list，元素类型为 _T
class silent_list(list[_T]):
    type: str | None  # 类型属性，字符串或 None
    # 初始化方法
    def __init__(self, type: str | None, seq: Iterable[_T] | None = ...) -> None: ...

# 定义一个 strip_math 函数，接收一个字符串 s，返回一个字符串
def strip_math(s: str) -> str: ...

# 定义一个 is_writable_file_like 函数，接收一个任意类型的参数 obj，返回一个布尔值
def is_writable_file_like(obj: Any) -> bool: ...

# 定义一个 file_requires_unicode 函数，接收一个任意类型的参数 x，返回一个布尔值
def file_requires_unicode(x: Any) -> bool: ...

# 函数重载，根据不同的参数类型返回不同的类型
@overload
def to_filehandle(
    fname: str | os.PathLike | IO,  # 文件名，路径对象或者 IO 对象
    flag: str = ...,  # 标志，字符串，默认为 ...
    return_opened: Literal[False] = ...,  # 是否返回打开的文件对象，字面值类型为 False
    encoding: str | None = ...,  # 编码，字符串或 None
) -> IO: ...

@overload
def to_filehandle(
    fname: str | os.PathLike | IO,  # 文件名，路径对象或者 IO 对象
    flag: str,  # 标志，字符串
    return_opened: Literal[True],  # 是否返回打开的文件对象，字面值类型为 True
    encoding: str | None = ...,  # 编码，字符串或 None
) -> tuple[IO, bool]: ...

@overload
def to_filehandle(
    fname: str | os.PathLike | IO,  # 文件名，路径对象或者 IO 对象
    *,  # 使用关键字参数
    return_opened: Literal[True],  # 是否返回打开的文件对象，字面值类型为 True
    encoding: str | None = ...,  # 编码，字符串或 None
) -> tuple[IO, bool]: ...

# 定义一个 open_file_cm 函数，接收路径或文件对象，模式和编码参数，返回一个用于管理文件打开状态的上下文管理器
def open_file_cm(
    path_or_file: str | os.PathLike | IO,  # 路径、文件对象或者 IO 对象
    mode: str = ...,  # 模式，字符串，默认为 ...
    encoding: str | None = ...,  # 编码，字符串或 None
) -> contextlib.AbstractContextManager[IO]: ...

# 定义一个 is_scalar_or_string 函数，接收一个任意类型的参数 val，返回一个布尔值
def is_scalar_or_string(val: Any) -> bool: ...

# 函数重载，根据参数返回不同类型的数据
@overload
def get_sample_data(
    fname: str | os.PathLike,  # 文件名，路径对象
    asfileobj: Literal[True] = ...,  # 是否返回文件对象，字面值类型为 True
    *,  # 使用关键字参数
    np_load: Literal[True]  # 是否使用 NumPy 加载数据，字面值类型为 True
) -> np.ndarray: ...

@overload
def get_sample_data(
    fname: str | os.PathLike,  # 文件名，路径对象
    asfileobj: Literal[True] = ...,  # 是否返回文件对象，字面值类型为 True
    *,  # 使用关键字参数
    np_load: Literal[False]  # 是否使用 NumPy 加载数据，字面值类型为 False
) -> IO: ...

@overload
def get_sample_data(
    fname: str | os.PathLike,  # 文件名，路径对象
    asfileobj: Literal[False],  # 是否返回文件对象，字面值类型为 False
    *,  # 使用关键字参数
    np_load: bool  # 是否使用 NumPy 加载数据，布尔类型
) -> str: ...

# 定义一个 _get_data_path 函数，接收多个路径参数，返回一个 Path 对象
def _get_data_path(*args: Path | str) -> Path: ...

# 定义一个 flatten 函数，接收一个可迭代对象 seq 和一个判断元素是否标量的函数 scalarp，返回一个生成器
def flatten(
    seq: Iterable[Any],  # 可迭代对象
    scalarp: Callable[[Any], bool] = ...  # 判断是否标量的函数，可选参数，默认为 ...
) -> Generator[Any, None, None]: ...

# 定义一个泛型类 Stack，接收泛型类型 _T
class Stack(Generic[_T]):
    # 初始化方法，接收一个默认值参数 default，类型为 _T 或 None
    def __init__(self, default: _T | None = ...) -> None: ...

    # __call__ 方法，使对象可调用，返回类型为 _T
    def __call__(self) -> _T: ...

    # __len__ 方法，返回栈的长度，类型为整数
    def __len__(self) -> int: ...

    # __getitem__ 方法，根据索引返回栈中的元素，类型为 _T
    def __getitem__(self, ind: int) -> _T: ...

    # forward 方法，返回栈顶元素，类型为 _T
    def forward(self) -> _T: ...

    # back 方法，返回栈底元素，类型为 _T
    def back(self) -> _T: ...

    # push 方法，将元素压入栈，返回压入的元素，类型为 _T
    def push(self, o: _T) -> _T: ...

    # home 方法
    # 定义一个方法 clear，返回类型为 None，不包含具体实现（占位符）
    def clear(self) -> None: ...

    # 定义一个方法 bubble，接受一个参数 o（类型为 _T），返回类型为 _T，不包含具体实现（占位符）
    def bubble(self, o: _T) -> _T: ...

    # 定义一个方法 remove，接受一个参数 o（类型为 _T），返回类型为 None，不包含具体实现（占位符）
    def remove(self, o: _T) -> None: ...
# 安全地处理具有无效掩码的数组，返回一个 NumPy 数组
def safe_masked_invalid(x: ArrayLike, copy: bool = ...) -> np.ndarray: ...

# 打印对象的循环依赖关系，输出到指定流，可以选择是否显示进度条
def print_cycles(
    objects: Iterable[Any], outstream: IO = ..., show_progress: bool = ...
) -> None: ...


class Grouper(Generic[_T]):
    # 初始化 Grouper 类，接受一个可迭代对象作为初始值
    def __init__(self, init: Iterable[_T] = ...) -> None: ...
    
    # 检查是否包含指定元素
    def __contains__(self, item: _T) -> bool: ...
    
    # 清空当前 Grouper 对象中的所有元素
    def clean(self) -> None: ...
    
    # 将多个元素加入到 Grouper 对象中，以第一个元素为主
    def join(self, a: _T, *args: _T) -> None: ...
    
    # 检查两个元素是否已经在同一个分组中
    def joined(self, a: _T, b: _T) -> bool: ...
    
    # 移除指定元素及其相关的所有元素
    def remove(self, a: _T) -> None: ...
    
    # 返回一个迭代器，用于迭代 Grouper 对象中的所有列表
    def __iter__(self) -> Iterator[list[_T]]: ...
    
    # 获取与指定元素同组的所有元素列表
    def get_siblings(self, a: _T) -> list[_T]: ...


class GrouperView(Generic[_T]):
    # 初始化 GrouperView 类，接受一个 Grouper 对象作为参数
    def __init__(self, grouper: Grouper[_T]) -> None: ...
    
    # 检查是否包含指定元素
    def __contains__(self, item: _T) -> bool: ...
    
    # 返回一个迭代器，用于迭代 Grouper 对象中的所有列表
    def __iter__(self) -> Iterator[list[_T]]: ...
    
    # 检查两个元素是否已经在同一个分组中
    def joined(self, a: _T, b: _T) -> bool: ...
    
    # 获取与指定元素同组的所有元素列表
    def get_siblings(self, a: _T) -> list[_T]: ...


# 简单线性插值函数，给定数组和步数，返回插值后的数组
def simple_linear_interpolation(a: ArrayLike, steps: int) -> np.ndarray: ...

# 删除具有掩码的点
def delete_masked_points(*args): ...

# 根据掩码进行广播操作，并返回广播后的数组列表
def _broadcast_with_masks(*args: ArrayLike, compress: bool = ...) -> list[ArrayLike]: ...

# 计算箱线图的统计数据
def boxplot_stats(
    X: ArrayLike,
    whis: float | tuple[float, float] = ...,
    bootstrap: int | None = ...,
    labels: ArrayLike | None = ...,
    autorange: bool = ...,
) -> list[dict[str, Any]]: ...

# 文件名映射字典
ls_mapper: dict[str, str]

# 反向文件名映射字典
ls_mapper_r: dict[str, str]

# 计算掩码的连续区域
def contiguous_regions(mask: ArrayLike) -> list[np.ndarray]: ...

# 判断字符串是否为数学表达式
def is_math_text(s: str) -> bool: ...

# 计算小提琴图的统计数据
def violin_stats(
    X: ArrayLike, method: Callable, points: int = ..., quantiles: ArrayLike | None = ...
) -> list[dict[str, Any]]: ...

# 将点转换为前步骤
def pts_to_prestep(x: ArrayLike, *args: ArrayLike) -> np.ndarray: ...

# 将点转换为后步骤
def pts_to_poststep(x: ArrayLike, *args: ArrayLike) -> np.ndarray: ...

# 将点转换为中间步骤
def pts_to_midstep(x: np.ndarray, *args: np.ndarray) -> np.ndarray: ...

# 查找与给定参数匹配的步骤函数
STEP_LOOKUP_MAP: dict[str, Callable]

# 计算浮点数或数组的索引位置
def index_of(y: float | ArrayLike) -> tuple[np.ndarray, np.ndarray]: ...

# 获取集合的第一个元素
def safe_first_element(obj: Collection[_T]) -> _T: ...

# 标准化序列数据
def sanitize_sequence(data): ...

# 标准化关键字参数
def normalize_kwargs(
    kw: dict[str, Any],
    alias_mapping: dict[str, list[str]] | type[Artist] | Artist | None = ...,
) -> dict[str, Any]: ...

# 锁定路径，返回一个上下文管理器
def _lock_path(path: str | os.PathLike) -> contextlib.AbstractContextManager[None]: ...

# 比较对象和字符串是否相等（忽略大小写）
def _str_equal(obj: Any, s: str) -> bool: ...

# 比较对象和字符串是否相等（忽略大小写）
def _str_lower_equal(obj: Any, s: str) -> bool: ...

# 计算数组的周长
def _array_perimeter(arr: np.ndarray) -> np.ndarray: ...

# 对数组进行展开操作
def _unfold(arr: np.ndarray, axis: int, size: int, step: int) -> np.ndarray: ...

# 计算数组的补丁周长
def _array_patch_perimeters(x: np.ndarray, rstride: int, cstride: int) -> np.ndarray: ...

# 设置对象的属性，返回一个上下文管理器
def _setattr_cm(obj: Any, **kwargs) -> contextlib.AbstractContextManager[None]: ...


class _OrderedSet(collections.abc.MutableSet):
    # 初始化 _OrderedSet 类
    def __init__(self) -> None: ...
    
    # 检查集合中是否包含指定键值
    def __contains__(self, key) -> bool: ...
    
    # 返回集合的迭代器
    def __iter__(self): ...
    
    # 返回集合的长度
    def __len__(self) -> int: ...
    
    # 向集合中添加元素
    def add(self, key) -> None: ...
    
    # 从集合中移除元素
    def discard(self, key) -> None: ...


# 设置新的 GUI 应用程序
def _setup_new_guiapp() -> None: ...
# 格式化一个浮点数为指定精度的字符串表示
def _format_approx(number: float, precision: int) -> str:
    ...

# 计算浮点数的有效数字位数
def _g_sig_digits(value: float, delta: float) -> int:
    ...

# 将唯一键或键符号转换为 matplotlib 键的字符串表示
def _unikey_or_keysym_to_mplkey(unikey: str, keysym: str) -> str:
    ...

# 检查给定对象是否为 Torch 张量
def _is_torch_array(x: Any) -> bool:
    ...

# 检查给定对象是否为 JAX 数组
def _is_jax_array(x: Any) -> bool:
    ...

# 将任意对象解包为 NumPy 对象
def _unpack_to_numpy(x: Any) -> Any:
    ...

# 自动根据给定格式字符串格式化值
def _auto_format_str(fmt: str, value: Any) -> str:
    ...
```