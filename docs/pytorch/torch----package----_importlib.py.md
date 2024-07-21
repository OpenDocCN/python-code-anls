# `.\pytorch\torch\package\_importlib.py`

```py
# mypy: allow-untyped-defs
import _warnings  # 导入_warnings模块，用于处理警告
import os.path  # 导入os.path模块，用于处理路径相关操作

# note: implementations
# copied from cpython's import code
# 注释：实现部分代码来自cpython的导入代码

# _zip_searchorder定义了在Zip存档中搜索模块的顺序：
# 首先搜索包的__init__，然后是非包的.py文件，最后是.py文件。
# 如果在优化模式下运行，.pyc条目会被initzipimport()替换。此外，
# '/' 在这里被path_sep替换。
_zip_searchorder = (
    ("/__init__.py", True),  # 搜索包的__init__.py文件
    (".py", False),  # 搜索非包的.py文件
)


# Replace any occurrences of '\r\n?' in the input string with '\n'.
# This converts DOS and Mac line endings to Unix line endings.
# 将输入字符串中的'\r\n?'替换为'\n'。
# 这将DOS和Mac的换行符转换为Unix的换行符。
def _normalize_line_endings(source):
    source = source.replace(b"\r\n", b"\n")  # 替换\r\n为\n
    source = source.replace(b"\r", b"\n")    # 替换\r为\n
    return source


def _resolve_name(name, package, level):
    """Resolve a relative module name to an absolute one."""
    bits = package.rsplit(".", level - 1)  # 使用包名和级别拆分路径
    if len(bits) < level:  # 如果拆分后的部分少于级别
        raise ValueError("attempted relative import beyond top-level package")  # 抛出值错误异常
    base = bits[0]  # 获取基本路径
    return f"{base}.{name}" if name else base  # 返回完整的模块名或基本路径


def _sanity_check(name, package, level):
    """Verify arguments are "sane"."""
    if not isinstance(name, str):  # 如果模块名不是字符串类型
        raise TypeError(f"module name must be str, not {type(name)}")  # 抛出类型错误异常
    if level < 0:  # 如果级别小于0
        raise ValueError("level must be >= 0")  # 抛出值错误异常
    if level > 0:  # 如果级别大于0
        if not isinstance(package, str):  # 如果包不是字符串类型
            raise TypeError("__package__ not set to a string")  # 抛出类型错误异常
        elif not package:  # 如果包为空
            raise ImportError("attempted relative import with no known parent package")  # 抛出导入错误异常
    if not name and level == 0:  # 如果模块名为空且级别为0
        raise ValueError("Empty module name")  # 抛出值错误异常


def _calc___package__(globals):
    """Calculate what __package__ should be.

    __package__ is not guaranteed to be defined or could be set to None
    to represent that its proper value is unknown.

    """
    package = globals.get("__package__")  # 获取全局变量中的__package__
    spec = globals.get("__spec__")  # 获取全局变量中的__spec__
    if package is not None:  # 如果__package__不为None
        if spec is not None and package != spec.parent:  # 如果__spec__不为None且__package__不等于__spec__.parent
            _warnings.warn(  # 发出警告
                f"__package__ != __spec__.parent ({package!r} != {spec.parent!r})",  # 警告消息
                ImportWarning,
                stacklevel=3,
            )
        return package  # 返回__package__
    elif spec is not None:  # 如果__spec__不为None
        return spec.parent  # 返回__spec__.parent
    else:  # 如果都为None
        _warnings.warn(  # 发出警告
            "can't resolve package from __spec__ or __package__, "
            "falling back on __name__ and __path__",
            ImportWarning,
            stacklevel=3,
        )
        package = globals["__name__"]  # 获取全局变量中的__name__
        if "__path__" not in globals:  # 如果__path__不在全局变量中
            package = package.rpartition(".")[0]  # 获取最后一个点之前的部分作为包名
    return package  # 返回计算出的包名


def _normalize_path(path):
    """Normalize a path by ensuring it is a string.

    If the resulting string contains path separators, an exception is raised.
    """
    parent, file_name = os.path.split(path)  # 将路径拆分为父目录和文件名
    if parent:  # 如果有父目录
        raise ValueError(f"{path!r} must be only a file name")  # 抛出值错误异常
    else:
        return file_name  # 返回文件名
```