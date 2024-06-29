# `D:\src\scipysrc\pandas\pandas\core\computation\scope.py`

```
"""
Module for scope operations
"""

from __future__ import annotations

from collections import ChainMap  # 导入 ChainMap 类，用于管理多个映射对象
import datetime  # 导入 datetime 模块，处理日期和时间
import inspect  # 导入 inspect 模块，提供了解析 Python 对象的工具
from io import StringIO  # 导入 StringIO 类，用于在内存中操作文本
import itertools  # 导入 itertools 模块，提供了用于迭代的工具函数
import pprint  # 导入 pprint 模块，用于漂亮打印数据结构
import struct  # 导入 struct 模块，用于处理字节数据和 C 结构体的转换
import sys  # 导入 sys 模块，提供了与 Python 解释器交互的函数和变量
from typing import TypeVar  # 导入 TypeVar 类，用于声明泛型类型

import numpy as np  # 导入 NumPy 库，用于数值计算

from pandas._libs.tslibs import Timestamp  # 导入 Timestamp 类，处理时间序列数据
from pandas.errors import UndefinedVariableError  # 导入 UndefinedVariableError 类，处理未定义变量的异常


_KT = TypeVar("_KT")  # 声明泛型类型变量 _KT
_VT = TypeVar("_VT")  # 声明泛型类型变量 _VT


# https://docs.python.org/3/library/collections.html#chainmap-examples-and-recipes
class DeepChainMap(ChainMap[_KT, _VT]):
    """
    Variant of ChainMap that allows direct updates to inner scopes.

    Only works when all passed mapping are mutable.
    """

    def __setitem__(self, key: _KT, value: _VT) -> None:
        for mapping in self.maps:
            if key in mapping:
                mapping[key] = value
                return
        self.maps[0][key] = value

    def __delitem__(self, key: _KT) -> None:
        """
        Raises
        ------
        KeyError
            If `key` doesn't exist.
        """
        for mapping in self.maps:
            if key in mapping:
                del mapping[key]
                return
        raise KeyError(key)


def ensure_scope(
    level: int, global_dict=None, local_dict=None, resolvers=(), target=None
) -> Scope:
    """Ensure that we are grabbing the correct scope."""
    return Scope(
        level + 1,
        global_dict=global_dict,
        local_dict=local_dict,
        resolvers=resolvers,
        target=target,
    )


def _replacer(x) -> str:
    """
    Replace a number with its hexadecimal representation. Used to tag
    temporary variables with their calling scope's id.
    """
    # get the hex repr of the binary char and remove 0x and pad by pad_size
    # zeros
    try:
        hexin = ord(x)
    except TypeError:
        # bytes literals masquerade as ints when iterating in py3
        hexin = x

    return hex(hexin)


def _raw_hex_id(obj) -> str:
    """Return the padded hexadecimal id of ``obj``."""
    # interpret as a pointer since that's what really what id returns
    packed = struct.pack("@P", id(obj))
    return "".join([_replacer(x) for x in packed])


DEFAULT_GLOBALS = {
    "Timestamp": Timestamp,  # 将 Timestamp 类对象映射到 "Timestamp" 键
    "datetime": datetime.datetime,  # 将 datetime 模块中的 datetime 类对象映射到 "datetime" 键
    "True": True,  # 将布尔值 True 映射到 "True" 键
    "False": False,  # 将布尔值 False 映射到 "False" 键
    "list": list,  # 将 list 类型映射到 "list" 键
    "tuple": tuple,  # 将 tuple 类型映射到 "tuple" 键
    "inf": np.inf,  # 将 NumPy 中的正无穷大映射到 "inf" 键
    "Inf": np.inf,  # 将 NumPy 中的正无穷大映射到 "Inf" 键
}


def _get_pretty_string(obj) -> str:
    """
    Return a prettier version of obj.

    Parameters
    ----------
    obj : object
        Object to pretty print

    Returns
    -------
    str
        Pretty print object repr
    """
    sio = StringIO()  # 创建一个 StringIO 对象 sio，用于内存中的文本操作
    pprint.pprint(obj, stream=sio)  # 使用 pprint 模块漂亮地打印 obj，并输出到 sio
    return sio.getvalue()  # 返回 sio 对象中的文本内容


class Scope:
    """
    Object to hold scope, with a few bells to deal with some custom syntax
    and contexts added by pandas.

    Parameters
    ----------
    level : int
        Scope level identifier.
    global_dict : dict or None, optional, default None
        Global variables dictionary.
    local_dict : dict or Scope or None, optional, default None
        Local variables dictionary or another Scope object.
    """

    def __init__(self, level: int, global_dict=None, local_dict=None):
        self.level = level
        self.global_dict = global_dict if global_dict is not None else {}
        self.local_dict = local_dict if local_dict is not None else {}
    resolvers : list-like or None, optional, default None
    target : object

    Attributes
    ----------
    level : int
        # 表示当前对象的嵌套层级
    scope : DeepChainMap
        # 存储作用域链映射关系的深度链映射对象
    target : object
        # 表示当前对象的目标对象
    temps : dict
        # 存储临时变量的字典
    """

    __slots__ = ["level", "scope", "target", "resolvers", "temps"]
    level: int
    scope: DeepChainMap
    resolvers: DeepChainMap
    temps: dict

    def __init__(
        self, level: int, global_dict=None, local_dict=None, resolvers=(), target=None
    ) -> None:
        self.level = level + 1
        # 增加当前层级数，用于对象嵌套

        # 创建全局作用域的深度链映射对象，并浅拷贝默认全局字典
        self.scope = DeepChainMap(DEFAULT_GLOBALS.copy())
        self.target = target

        if isinstance(local_dict, Scope):
            # 如果local_dict是Scope类型，更新作用域并处理目标对象
            self.scope.update(local_dict.scope)
            if local_dict.target is not None:
                self.target = local_dict.target
            self._update(local_dict.level)

        frame = sys._getframe(self.level)

        try:
            # 创建全局作用域的深度链映射对象，浅拷贝frame的全局变量字典或者传入的全局字典
            scope_global = self.scope.new_child(
                (global_dict if global_dict is not None else frame.f_globals).copy()
            )
            self.scope = DeepChainMap(scope_global)
            if not isinstance(local_dict, Scope):
                # 如果local_dict不是Scope类型，创建局部作用域的深度链映射对象，浅拷贝frame的局部变量字典或者传入的局部字典
                scope_local = self.scope.new_child(
                    (local_dict if local_dict is not None else frame.f_locals).copy()
                )
                self.scope = DeepChainMap(scope_local)
        finally:
            del frame

        # 将外部解析器的深度链映射对象组合成一个新的深度链映射对象
        if isinstance(local_dict, Scope):
            resolvers += tuple(local_dict.resolvers.maps)
        self.resolvers = DeepChainMap(*resolvers)
        self.temps = {}

    def __repr__(self) -> str:
        # 获取作用域和解析器的键的漂亮字符串表示
        scope_keys = _get_pretty_string(list(self.scope.keys()))
        res_keys = _get_pretty_string(list(self.resolvers.keys()))
        return f"{type(self).__name__}(scope={scope_keys}, resolvers={res_keys})"

    @property
    def has_resolvers(self) -> bool:
        """
        Return whether we have any extra scope.

        For example, DataFrames pass Their columns as resolvers during calls to
        ``DataFrame.eval()`` and ``DataFrame.query()``.

        Returns
        -------
        hr : bool
            # 返回是否有额外作用域的布尔值
        """
        return bool(len(self.resolvers))
    def resolve(self, key: str, is_local: bool):
        """
        Resolve a variable name in a possibly local context.

        Parameters
        ----------
        key : str
            A variable name
        is_local : bool
            Flag indicating whether the variable is local or not (prefixed with
            the '@' symbol)

        Returns
        -------
        value : object
            The value of a particular variable
        """
        try:
            # 如果变量是本地变量，则直接从当前作用域中获取
            if is_local:
                return self.scope[key]

            # 如果不是本地变量，则在解析器中查找
            if self.has_resolvers:
                return self.resolvers[key]

            # 如果既没有本地变量也没有解析器，则从当前作用域中获取
            assert not is_local and not self.has_resolvers
            return self.scope[key]
        except KeyError:
            try:
                # 如果以上都失败了，则尝试从临时变量中获取
                # 这些变量通常在解析索引表达式时创建，例如 df[df > 0]
                return self.temps[key]
            except KeyError as err:
                # 如果变量未定义，则抛出未定义变量错误
                raise UndefinedVariableError(key, is_local) from err

    def swapkey(self, old_key: str, new_key: str, new_value=None) -> None:
        """
        Replace a variable name, with a potentially new value.

        Parameters
        ----------
        old_key : str
            Current variable name to replace
        new_key : str
            New variable name to replace `old_key` with
        new_value : object
            Value to be replaced along with the possible renaming
        """
        if self.has_resolvers:
            # 如果存在解析器，则合并解析器和当前作用域的映射
            maps = self.resolvers.maps + self.scope.maps
        else:
            # 否则只使用当前作用域的映射
            maps = self.scope.maps

        # 将临时变量映射添加到映射列表中
        maps.append(self.temps)

        # 在所有映射中查找并替换旧变量名为新变量名及其可能的新值
        for mapping in maps:
            if old_key in mapping:
                mapping[new_key] = new_value
                return

    def _get_vars(self, stack, scopes: list[str]) -> None:
        """
        Get specifically scoped variables from a list of stack frames.

        Parameters
        ----------
        stack : list
            A list of stack frames as returned by ``inspect.stack()``
        scopes : sequence of strings
            A sequence containing valid stack frame attribute names that
            evaluate to a dictionary. For example, ('locals', 'globals')
        """
        variables = itertools.product(scopes, stack)
        for scope, (frame, _, _, _, _, _) in variables:
            try:
                # 获取指定作用域的变量字典，并更新当前作用域
                d = getattr(frame, f"f_{scope}")
                self.scope = DeepChainMap(self.scope.new_child(d))
            finally:
                # 不会移除它，但会降低它的引用计数
                # 在 Python 3 中可能不是必要的，因为循环后 frame 不再是作用域
                del frame
    def _update(self, level: int) -> None:
        """
        Update the current scope by going back `level` levels.

        Parameters
        ----------
        level : int
            Number of levels to go back in the call stack.
        """
        sl = level + 1  # Calculate the stack depth to capture

        # Retrieve stack frames up to sl levels to capture variable scopes
        stack = inspect.stack()

        try:
            # Fetch variables from the stack frames up to sl levels deep
            self._get_vars(stack[:sl], scopes=["locals"])
        finally:
            # Clean up the stack to prevent memory leaks
            del stack[:], stack

    def add_tmp(self, value) -> str:
        """
        Add a temporary variable to the scope.

        Parameters
        ----------
        value : object
            An arbitrary object to be assigned to a temporary variable.

        Returns
        -------
        str
            The name of the temporary variable created.
        """
        # Generate a unique name for the temporary variable based on its type and context
        name = f"{type(value).__name__}_{self.ntemps}_{_raw_hex_id(self)}"

        # Ensure the variable name is not already in use in the current scope
        assert name not in self.temps

        # Assign the temporary variable to the current scope
        self.temps[name] = value

        # Assert that the variable is now in the scope
        assert name in self.temps

        # Increment the count of temporary variables
        # (only when the variable is successfully added to the scope)
        return name

    @property
    def ntemps(self) -> int:
        """The number of temporary variables in this scope"""
        # Return the count of temporary variables currently in the scope
        return len(self.temps)

    @property
    def full_scope(self) -> DeepChainMap:
        """
        Return the full scope for use with passing to engines transparently
        as a mapping.

        Returns
        -------
        vars : DeepChainMap
            All variables in this scope.
        """
        # Construct a DeepChainMap combining temporary variables, resolvers, and other scopes
        maps = [self.temps] + self.resolvers.maps + self.scope.maps
        return DeepChainMap(*maps)
```