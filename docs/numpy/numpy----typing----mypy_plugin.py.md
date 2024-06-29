# `.\numpy\numpy\typing\mypy_plugin.py`

```
"""A mypy_ plugin for managing a number of platform-specific annotations.
Its functionality can be split into three distinct parts:

* Assigning the (platform-dependent) precisions of certain `~numpy.number`
  subclasses, including the likes of `~numpy.int_`, `~numpy.intp` and
  `~numpy.longlong`. See the documentation on
  :ref:`scalar types <arrays.scalars.built-in>` for a comprehensive overview
  of the affected classes. Without the plugin the precision of all relevant
  classes will be inferred as `~typing.Any`.
* Removing all extended-precision `~numpy.number` subclasses that are
  unavailable for the platform in question. Most notably this includes the
  likes of `~numpy.float128` and `~numpy.complex256`. Without the plugin *all*
  extended-precision types will, as far as mypy is concerned, be available
  to all platforms.
* Assigning the (platform-dependent) precision of `~numpy.ctypeslib.c_intp`.
  Without the plugin the type will default to `ctypes.c_int64`.

  .. versionadded:: 1.22

Examples
--------
To enable the plugin, one must add it to their mypy `configuration file`_:

.. code-block:: ini

    [mypy]
    plugins = numpy.typing.mypy_plugin

.. _mypy: https://mypy-lang.org/
.. _configuration file: https://mypy.readthedocs.io/en/stable/config_file.html

"""

from __future__ import annotations  # 允许使用自身类型作为返回类型或者参数类型提示

from collections.abc import Iterable  # 导入抽象基类 Iterable
from typing import Final, TYPE_CHECKING, Callable  # 导入 Final 类型注解，TYPE_CHECKING 常量和 Callable 类型注解

import numpy as np  # 导入 NumPy 库

try:
    import mypy.types  # 尝试导入 mypy.types 模块
    from mypy.types import Type  # 从 mypy.types 导入 Type 类型注解
    from mypy.plugin import Plugin, AnalyzeTypeContext  # 从 mypy.plugin 导入 Plugin 和 AnalyzeTypeContext 类
    from mypy.nodes import MypyFile, ImportFrom, Statement  # 从 mypy.nodes 导入 MypyFile, ImportFrom 和 Statement 类
    from mypy.build import PRI_MED  # 导入 PRI_MED 常量

    _HookFunc = Callable[[AnalyzeTypeContext], Type]  # 定义 _HookFunc 类型别名
    MYPY_EX: None | ModuleNotFoundError = None  # 初始化 MYPY_EX 变量为 None 或 ModuleNotFoundError 类型
except ModuleNotFoundError as ex:
    MYPY_EX = ex  # 捕获 ModuleNotFoundError 异常并赋值给 MYPY_EX 变量

__all__: list[str] = []  # 初始化 __all__ 变量为空列表，用于控制模块的公开接口


def _get_precision_dict() -> dict[str, str]:
    # 定义一个列表 names，包含多个元组，每个元组包含名称和对应的 NumPy 类型
    names = [
        ("_NBitByte", np.byte),
        ("_NBitShort", np.short),
        ("_NBitIntC", np.intc),
        ("_NBitIntP", np.intp),
        ("_NBitInt", np.int_),
        ("_NBitLong", np.long),
        ("_NBitLongLong", np.longlong),

        ("_NBitHalf", np.half),
        ("_NBitSingle", np.single),
        ("_NBitDouble", np.double),
        ("_NBitLongDouble", np.longdouble),
    ]
    ret = {}  # 初始化一个空字典 ret
    for name, typ in names:  # 遍历 names 列表
        n: int = 8 * typ().dtype.itemsize  # 计算每种类型的位数
        ret[f'numpy._typing._nbit.{name}'] = f"numpy._{n}Bit"  # 构建精度字典，键为名称，值为精度字符串
    return ret  # 返回精度字典


def _get_extended_precision_list() -> list[str]:
    # 定义一个列表 extended_names，包含多个字符串，表示扩展精度的类型名称
    extended_names = [
        "uint128",
        "uint256",
        "int128",
        "int256",
        "float80",
        "float96",
        "float128",
        "float256",
        "complex160",
        "complex192",
        "complex256",
        "complex512",
    ]
    return [i for i in extended_names if hasattr(np, i)]  # 返回列表中存在于 NumPy 中的类型名称列表


def _get_c_intp_name() -> str:
    # 从 np.core._internal._getintp_ctype 改编而来
    char = np.dtype('n').char  # 获取字符 'n' 的数据类型对象
    if char == 'i':  # 如果字符为 'i'
        return "c_int"  # 返回字符串 "c_int"
    elif char == 'l':
        # 如果字符为 'l'，返回字符串 "c_long"
        return "c_long"
    elif char == 'q':
        # 如果字符为 'q'，返回字符串 "c_longlong"
        return "c_longlong"
    else:
        # 对于其他任何字符，返回字符串 "c_long"
        return "c_long"
#: A dictionary mapping type-aliases in `numpy._typing._nbit` to
#: concrete `numpy.typing.NBitBase` subclasses.
_PRECISION_DICT: Final = _get_precision_dict()

#: A list with the names of all extended precision `np.number` subclasses.
_EXTENDED_PRECISION_LIST: Final = _get_extended_precision_list()

#: The name of the ctypes equivalent of `np.intp`
_C_INTP: Final = _get_c_intp_name()

# 定义一个函数 `_hook`，用于替换类型别名为具体的 `NBitBase` 子类。
def _hook(ctx: AnalyzeTypeContext) -> Type:
    """Replace a type-alias with a concrete ``NBitBase`` subclass."""
    typ, _, api = ctx
    name = typ.name.split(".")[-1]
    name_new = _PRECISION_DICT[f"numpy._typing._nbit.{name}"]
    return api.named_type(name_new)

# 当类型检查开启或者 MYPY_EX 为 None 时，定义函数 `_index`。
def _index(iterable: Iterable[Statement], id: str) -> int:
    """Identify the first ``ImportFrom`` instance the specified `id`."""
    for i, value in enumerate(iterable):
        if getattr(value, "id", None) == id:
            return i
    raise ValueError("Failed to identify a `ImportFrom` instance "
                     f"with the following id: {id!r}")

# 当类型检查开启或者 MYPY_EX 为 None 时，定义函数 `_override_imports`。
def _override_imports(
    file: MypyFile,
    module: str,
    imports: list[tuple[str, None | str]],
) -> None:
    """Override the first `module`-based import with new `imports`."""
    # 构造一个新的 `from module import y` 语句
    import_obj = ImportFrom(module, 0, names=imports)
    import_obj.is_top_level = True

    # 替换第一个基于 `module` 的导入语句为 `import_obj`
    for lst in [file.defs, file.imports]:  # type: list[Statement]
        i = _index(lst, module)
        lst[i] = import_obj
    class _NumpyPlugin(Plugin):
        """定义一个名为 _NumpyPlugin 的插件类，继承自 Plugin 类。"""

        def get_type_analyze_hook(self, fullname: str) -> None | _HookFunc:
            """获取类型分析钩子函数。

            如果 fullname 在 _PRECISION_DICT 中，返回 _hook 函数；否则返回 None。
            """
            if fullname in _PRECISION_DICT:
                return _hook
            return None

        def get_additional_deps(
            self, file: MypyFile
        ) -> list[tuple[int, str, int]]:
            """获取额外的依赖项列表。

            根据 file 的 fullname 添加适当的依赖项：
            - 如果 fullname 是 "numpy"，则导入 numpy._typing._extended_precision 下的 _EXTENDED_PRECISION_LIST。
            - 如果 fullname 是 "numpy.ctypeslib"，则导入 ctypes 下的 _C_INTP。

            返回一个包含元组的列表，元组包含优先级 PRI_MED、file 的 fullname、和 -1。
            """
            ret = [(PRI_MED, file.fullname, -1)]

            if file.fullname == "numpy":
                _override_imports(
                    file, "numpy._typing._extended_precision",
                    imports=[(v, v) for v in _EXTENDED_PRECISION_LIST],
                )
            elif file.fullname == "numpy.ctypeslib":
                _override_imports(
                    file, "ctypes",
                    imports=[(_C_INTP, "_c_intp")],
                )
            return ret

    def plugin(version: str) -> type[_NumpyPlugin]:
        """插件的入口点函数，返回 _NumpyPlugin 类型。"""
        return _NumpyPlugin
else:
    # 定义一个名为 plugin 的函数，接受一个参数 version，返回一个 _NumpyPlugin 类型
    def plugin(version: str) -> type[_NumpyPlugin]:
        """An entry-point for mypy."""
        # 抛出 MYPY_EX 异常
        raise MYPY_EX
```