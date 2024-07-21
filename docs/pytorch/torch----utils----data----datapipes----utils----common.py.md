# `.\pytorch\torch\utils\data\datapipes\utils\common.py`

```
# mypy: allow-untyped-defs
# 导入必要的模块和函数
import fnmatch  # 用于文件名匹配
import functools  # 用于创建偏函数
import inspect  # 用于函数签名检查
import os  # 提供与操作系统交互的功能
import warnings  # 用于警告处理
from io import IOBase  # 提供用于处理文件流的基本工具
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union  # 提供类型提示支持

from torch.utils._import_utils import dill_available  # 导入特定的 Torch 工具函数


__all__ = [
    "validate_input_col",
    "StreamWrapper",
    "get_file_binaries_from_pathnames",
    "get_file_pathnames_from_root",
    "match_masks",
    "validate_pathname_binary_tuple",
]

# BC for torchdata
# 检查是否存在 dill 库可用
DILL_AVAILABLE = dill_available()


def validate_input_col(fn: Callable, input_col: Optional[Union[int, tuple, list]]):
    """
    Check that function used in a callable datapipe works with the input column.

    This simply ensures that the number of positional arguments matches the size
    of the input column. The function must not contain any non-default
    keyword-only arguments.

    Examples:
        >>> # xdoctest: +SKIP("Failing on some CI machines")
        >>> def f(a, b, *, c=1):
        >>>     return a + b + c
        >>> def f_def(a, b=1, *, c=1):
        >>>     return a + b + c
        >>> assert validate_input_col(f, [1, 2])
        >>> assert validate_input_col(f_def, 1)
        >>> assert validate_input_col(f_def, [1, 2])

    Notes:
        If the function contains variable positional (`inspect.VAR_POSITIONAL`) arguments,
        for example, f(a, *args), the validator will accept any size of input column
        greater than or equal to the number of positional arguments.
        (in this case, 1).

    Args:
        fn: The function to check.
        input_col: The input column to check.

    Raises:
        ValueError: If the function is not compatible with the input column.
    """
    try:
        sig = inspect.signature(fn)  # 获取函数的签名信息
    except (
        ValueError
    ):  # Signature cannot be inspected, likely it is a built-in fn or written in C
        return  # 如果无法获取函数签名，直接返回

    if isinstance(input_col, (list, tuple)):
        input_col_size = len(input_col)  # 获取输入列的长度
    else:
        input_col_size = 1  # 如果输入列不是列表或元组，则长度为1

    pos = []
    var_positional = False
    non_default_kw_only = []

    # 遍历函数签名中的参数
    for p in sig.parameters.values():
        if p.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            pos.append(p)  # 将位置参数添加到列表中
        elif p.kind is inspect.Parameter.VAR_POSITIONAL:
            var_positional = True  # 函数有可变位置参数
        elif p.kind is inspect.Parameter.KEYWORD_ONLY:
            if p.default is p.empty:
                non_default_kw_only.append(p)  # 函数有无默认值的关键字参数

    # 获取函数的名称
    if isinstance(fn, functools.partial):
        fn_name = getattr(fn.func, "__name__", repr(fn.func))
    else:
        fn_name = getattr(fn, "__name__", repr(fn))

    # 如果函数有非默认值的关键字参数，抛出 ValueError 异常
    if len(non_default_kw_only) > 0:
        raise ValueError(
            f"The function {fn_name} takes {len(non_default_kw_only)} "
            f"non-default keyword-only parameters, which is not allowed."
        )
    # 检查函数签名的参数数量是否小于所需的输入列数
    if len(sig.parameters) < input_col_size:
        # 如果没有可变位置参数，并且参数数量不足，则引发数值错误异常
        if not var_positional:
            raise ValueError(
                f"The function {fn_name} takes {len(sig.parameters)} "
                f"parameters, but {input_col_size} are required."
            )
    else:
        # 如果位置参数数量大于输入列数
        if len(pos) > input_col_size:
            # 如果任何超出输入列数的位置参数没有默认值，则引发数值错误异常
            if any(p.default is p.empty for p in pos[input_col_size:]):
                raise ValueError(
                    f"The function {fn_name} takes {len(pos)} "
                    f"positional parameters, but {input_col_size} are required."
                )
        elif len(pos) < input_col_size:
            # 如果位置参数数量少于输入列数，并且没有可变位置参数，则引发数值错误异常
            if not var_positional:
                raise ValueError(
                    f"The function {fn_name} takes {len(pos)} "
                    f"positional parameters, but {input_col_size} are required."
                )
# Functions or Methods
def _is_local_fn(fn):
    # 检查函数是否具有 __code__ 属性，表示它是函数或方法
    if hasattr(fn, "__code__"):
        # 返回函数的代码标志位中是否包含 CO_NESTED 标志位
        return fn.__code__.co_flags & inspect.CO_NESTED
    # Callable Objects
    else:
        # 如果对象具有 __qualname__ 属性，检查是否包含 "<locals>" 表示是局部函数
        if hasattr(fn, "__qualname__"):
            return "<locals>" in fn.__qualname__
        # 检查对象类型是否具有 __qualname__ 属性，如果有，检查是否包含 "<locals>" 表示是局部函数
        fn_type = type(fn)
        if hasattr(fn_type, "__qualname__"):
            return "<locals>" in fn_type.__qualname__
    # 默认返回 False，表示不是局部函数
    return False


def _check_unpickable_fn(fn: Callable):
    """
    Check function is pickable or not.

    If it is a lambda or local function, a UserWarning will be raised. If it's not a callable function, a TypeError will be raised.
    """
    # 检查传入的对象是否可调用
    if not callable(fn):
        raise TypeError(f"A callable function is expected, but {type(fn)} is provided.")

    # 如果传入的是 functools.partial 对象，则获取其对应的原始函数
    if isinstance(fn, functools.partial):
        fn = fn.func

    # 如果函数是局部函数且不支持 dill 序列化，则发出警告
    if _is_local_fn(fn) and not dill_available():
        warnings.warn(
            "Local function is not supported by pickle, please use "
            "regular python function or functools.partial instead."
        )
        return

    # 如果函数是 lambda 函数且不支持 dill 序列化，则发出警告
    if hasattr(fn, "__name__") and fn.__name__ == "<lambda>" and not dill_available():
        warnings.warn(
            "Lambda function is not supported by pickle, please use "
            "regular python function or functools.partial instead."
        )
        return


def match_masks(name: str, masks: Union[str, List[str]]) -> bool:
    # 空的 masks 匹配任何输入的 name
    if not masks:
        return True

    # 如果 masks 是字符串，则使用 fnmatch 模块判断 name 是否匹配该字符串
    if isinstance(masks, str):
        return fnmatch.fnmatch(name, masks)

    # 如果 masks 是列表，则逐个匹配 name 是否满足列表中的任意一个 mask
    for mask in masks:
        if fnmatch.fnmatch(name, mask):
            return True
    return False


def get_file_pathnames_from_root(
    root: str,
    masks: Union[str, List[str]],
    recursive: bool = False,
    abspath: bool = False,
    non_deterministic: bool = False,
) -> Iterable[str]:
    # 如果 root 是文件，则生成其路径名
    def onerror(err: OSError):
        warnings.warn(err.filename + " : " + err.strerror)
        raise err

    if os.path.isfile(root):
        path = root
        # 如果 abspath 为 True，则返回绝对路径名
        if abspath:
            path = os.path.abspath(path)
        fname = os.path.basename(path)
        # 如果文件名 fname 匹配 masks 中的任何一个 mask，则生成路径名 path
        if match_masks(fname, masks):
            yield path
    else:
        # 使用 os.walk 遍历指定根目录下的所有文件和文件夹
        for path, dirs, files in os.walk(root, onerror=onerror):
            if abspath:
                # 如果需要，将当前路径转换为绝对路径
                path = os.path.abspath(path)
            if not non_deterministic:
                # 如果需要按字母顺序排序文件列表
                files.sort()
            # 遍历当前目录下的所有文件
            for f in files:
                # 检查文件名是否匹配指定的文件名模式
                if match_masks(f, masks):
                    # 使用 os.path.join 将路径和文件名拼接，生成文件的完整路径
                    yield os.path.join(path, f)
            # 如果不需要递归遍历子目录，则退出循环
            if not recursive:
                break
            if not non_deterministic:
                # 如果需要按字母顺序排序目录列表（这里是对 dirs 的原地排序）
                # 注意：这里是对 os.walk 内部列表的原地修改
                dirs.sort()
# 从给定路径名列表中获取文件的二进制数据和流对象的生成器函数
def get_file_binaries_from_pathnames(
    pathnames: Iterable, mode: str, encoding: Optional[str] = None
):
    # 如果 pathnames 不是可迭代对象，则转换为包含单个元素的列表
    if not isinstance(pathnames, Iterable):
        pathnames = [
            pathnames,
        ]

    # 根据 mode 参数设置文件打开模式，如果 mode 是 'b' 或 't'，则修改为 'rb' 或 'rt'
    if mode in ("b", "t"):
        mode = "r" + mode

    # 遍历路径名列表，每次生成一个元组，包含路径名和通过 StreamWrapper 封装的文件流对象
    for pathname in pathnames:
        # 如果 pathname 不是字符串类型，则引发 TypeError 异常
        if not isinstance(pathname, str):
            raise TypeError(
                f"Expected string type for pathname, but got {type(pathname)}"
            )
        yield pathname, StreamWrapper(open(pathname, mode, encoding=encoding))


# 验证路径名和二进制流元组的类型和结构是否符合预期
def validate_pathname_binary_tuple(data: Tuple[str, IOBase]):
    # 确保 data 是元组类型，否则引发 TypeError 异常
    if not isinstance(data, tuple):
        raise TypeError(
            f"pathname binary data should be tuple type, but it is type {type(data)}"
        )
    # 确保元组长度为 2，否则引发 TypeError 异常
    if len(data) != 2:
        raise TypeError(
            f"pathname binary stream tuple length should be 2, but got {len(data)}"
        )
    # 确保第一个元素是字符串类型的路径名，否则引发 TypeError 异常
    if not isinstance(data[0], str):
        raise TypeError(
            f"pathname within the tuple should have string type pathname, but it is type {type(data[0])}"
        )
    # 确保第二个元素是 IOBase 或 StreamWrapper 类型的二进制流对象，否则引发 TypeError 异常
    if not isinstance(data[1], IOBase) and not isinstance(data[1], StreamWrapper):
        raise TypeError(
            f"binary stream within the tuple should have IOBase or"
            f"its subclasses as type, but it is type {type(data[1])}"
        )


# 用于存储被弃用函数名及其对应的 DataPipe 类型和关键字参数的字典
_iter_deprecated_functional_names: Dict[str, Dict] = {}
_map_deprecated_functional_names: Dict[str, Dict] = {}


# 打印弃用警告消息，指示用户应停止使用特定的旧函数或参数
def _deprecation_warning(
    old_class_name: str,
    *,
    deprecation_version: str,
    removal_version: str,
    old_functional_name: str = "",
    old_argument_name: str = "",
    new_class_name: str = "",
    new_functional_name: str = "",
    new_argument_name: str = "",
    deprecate_functional_name_only: bool = False,
) -> None:
    # 如果指定了新函数名但未指定旧函数名，则引发 ValueError 异常
    if new_functional_name and not old_functional_name:
        raise ValueError(
            "Old functional API needs to be specified for the deprecation warning."
        )
    # 如果指定了新参数名但未指定旧参数名，则引发 ValueError 异常
    if new_argument_name and not old_argument_name:
        raise ValueError(
            "Old argument name needs to be specified for the deprecation warning."
        )

    # 如果同时指定了旧函数名和旧参数名，则引发 ValueError 异常
    if old_functional_name and old_argument_name:
        raise ValueError(
            "Deprecating warning for functional API and argument should be separated."
        )

    # 构造弃用警告消息
    msg = f"`{old_class_name}()`"
    if deprecate_functional_name_only and old_functional_name:
        msg = f"{msg}'s functional API `.{old_functional_name}()` is"
    elif old_functional_name:
        msg = f"{msg} and its functional API `.{old_functional_name}()` are"
    elif old_argument_name:
        msg = f"The argument `{old_argument_name}` of {msg} is"
    else:
        msg = f"{msg} is"
    # 构建警告消息，说明某个函数或类已被弃用，并在未来版本中将被移除
    msg = (
        f"{msg} deprecated since {deprecation_version} and will be removed in {removal_version}."
        f"\nSee https://github.com/pytorch/data/issues/163 for details."
    )

    # 如果有新的类名或者新的函数名，添加相应的建议信息到警告消息中
    if new_class_name or new_functional_name:
        msg = f"{msg}\nPlease use"
        if new_class_name:
            msg = f"{msg} `{new_class_name}()`"
        if new_class_name and new_functional_name:
            msg = f"{msg} or"
        if new_functional_name:
            msg = f"{msg} `.{new_functional_name}()`"
        msg = f"{msg} instead."

    # 如果有新的参数名，添加相应的建议信息到警告消息中
    if new_argument_name:
        msg = f"{msg}\nPlease use `{old_class_name}({new_argument_name}=)` instead."

    # 发出警告，标记为未来可能移除的警告
    warnings.warn(msg, FutureWarning)
# StreamWrapper 类，用于封装由 DataPipe 操作（如 FileOpener）生成的文件处理程序。
class StreamWrapper:
    """
    StreamWrapper is introduced to wrap file handler generated by DataPipe operation like `FileOpener`.
    
    StreamWrapper would guarantee the wrapped file handler is closed when it's out of scope.
    """

    # 类变量：用于跟踪会话中的流对象及其计数
    session_streams: Dict[Any, int] = {}
    # 类变量：用于调试目的，开启后将跟踪未关闭的流对象
    debug_unclosed_streams: bool = False

    # 初始化方法，接收文件对象、父级流对象和名称作为参数
    def __init__(self, file_obj, parent_stream=None, name=None):
        # 将文件对象存储到实例变量中
        self.file_obj = file_obj
        # 子流计数器，记录当前对象有多少子流
        self.child_counter = 0
        # 父级流对象
        self.parent_stream = parent_stream
        # 当关闭最后一个子流时是否关闭自身的标志
        self.close_on_last_child = False
        # 流对象的名称
        self.name = name
        # 流对象是否已关闭的标志
        self.closed = False

        # 如果存在父级流对象，确保其为 StreamWrapper 类型，否则抛出 RuntimeError 异常
        if parent_stream is not None:
            if not isinstance(parent_stream, StreamWrapper):
                raise RuntimeError(
                    f"Parent stream should be StreamWrapper, {type(parent_stream)} was given"
                )
            # 增加父级流对象的子流计数器
            parent_stream.child_counter += 1
            self.parent_stream = parent_stream
        
        # 如果开启了 debug_unclosed_streams 标志，将当前流对象添加到会话流字典中
        if StreamWrapper.debug_unclosed_streams:
            StreamWrapper.session_streams[self] = 1

    # 类方法：深度遍历结构并尝试关闭所有发现的 StreamWrapper 对象
    @classmethod
    def close_streams(cls, v, depth=0):
        """Traverse structure and attempts to close all found StreamWrappers on best effort basis."""
        if depth > 10:
            return
        if isinstance(v, StreamWrapper):
            v.close()
        else:
            # 只遍历简单结构（字典、列表、元组）
            if isinstance(v, dict):
                for vv in v.values():
                    cls.close_streams(vv, depth=depth + 1)
            elif isinstance(v, (list, tuple)):
                for vv in v:
                    cls.close_streams(vv, depth=depth + 1)

    # 获取属性的特殊方法，委托给文件对象的对应属性
    def __getattr__(self, name):
        file_obj = self.__dict__["file_obj"]
        return getattr(file_obj, name)

    # 关闭流对象的方法
    def close(self, *args, **kwargs):
        if self.closed:
            return
        # 如果开启了 debug_unclosed_streams 标志，从会话流字典中删除当前流对象
        if StreamWrapper.debug_unclosed_streams:
            del StreamWrapper.session_streams[self]
        # 如果存在父级流对象，减少其子流计数器，并在必要时关闭父级流对象
        if hasattr(self, "parent_stream") and self.parent_stream is not None:
            self.parent_stream.child_counter -= 1
            if (
                not self.parent_stream.child_counter
                and self.parent_stream.close_on_last_child
            ):
                self.parent_stream.close()
        # 尝试关闭文件对象
        try:
            self.file_obj.close(*args, **kwargs)
        except AttributeError:
            pass
        # 设置流对象为已关闭状态
        self.closed = True

    # 自动关闭流对象的方法，当所有子流都关闭或没有子流时自动关闭当前流对象
    def autoclose(self):
        """Automatically close stream when all child streams are closed or if there are none."""
        self.close_on_last_child = True
        if self.child_counter == 0:
            self.close()

    # 返回对象的属性列表
    def __dir__(self):
        attrs = list(self.__dict__.keys()) + list(StreamWrapper.__dict__.keys())
        attrs += dir(self.file_obj)
        return list(set(attrs))

    # 迭代器接口方法，委托给文件对象
    def __iter__(self):
        yield from self.file_obj

    # 迭代器接口方法，委托给文件对象
    def __next__(self):
        return next(self.file_obj)
    # 定义对象的字符串表示形式，用于返回对象的描述信息
    def __repr__(self):
        # 如果对象的名称为 None，则返回格式化的字符串，显示文件对象信息
        if self.name is None:
            return f"StreamWrapper<{self.file_obj!r}>"
        else:
            # 否则返回格式化的字符串，显示对象名称和文件对象信息
            return f"StreamWrapper<{self.name},{self.file_obj!r}>"

    # 定义对象的序列化状态获取方法
    def __getstate__(self):
        # 返回对象的文件对象作为其状态信息
        return self.file_obj

    # 定义对象的序列化状态设置方法
    def __setstate__(self, obj):
        # 将传入的对象作为对象的新状态
        self.file_obj = obj
```