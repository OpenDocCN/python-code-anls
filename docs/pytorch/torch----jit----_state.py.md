# `.\pytorch\torch\jit\_state.py`

```
# mypy: allow-untyped-defs
"""JIT-related state.

This module stores various pieces of Python-global state relating to the JIT.

This is not intended to be imported directly; please use the exposed
functionalities in `torch.jit`.
"""
import os
import weakref
from typing import Any, Dict, Type

import torch


class EnabledProxy:
    """Stores whether the JIT is enabled or not.

    This is just a wrapper for a bool, so that we get reference semantics
    """

    def __init__(self):
        # 初始化时根据环境变量设置 self.enabled 的值
        self.enabled = self.parse_env(
            "PYTORCH_JIT", True, "> Using PyTorch JIT", "> PyTorch JIT DISABLED"
        )

    def parse_env(self, name, default, true_message, false_message):
        # 解析环境变量，根据其值设置 bool 类型的返回值
        value = os.environ.get(name)
        if value is None:
            return default
        if value.lower() in {"1", "true", "yes"}:
            return True
        elif value.lower() in {"0", "false", "no"}:
            return False
        if value == "1v":
            print(true_message)
            return True
        elif value == "0v":
            print(false_message)
            return False
        # 未知的设置值，抛出异常
        raise ValueError(f"Unknown setting of {name}. Try using 0 or 1.")

    def __bool__(self):
        # 重载 bool() 函数，返回 self.enabled 的布尔值
        return self.enabled


_enabled = EnabledProxy()


def disable():
    # 将 _enabled 的 enabled 属性设为 False，禁用 JIT
    _enabled.enabled = False


def enable():
    # 将 _enabled 的 enabled 属性设为 True，启用 JIT
    _enabled.enabled = True


# The Python CompilationUnit. All functions and modules defined in Python will
# live in here. It's defined in Python because doing in cpp creates static
# destruction order issues.
# Python 的 CompilationUnit。所有在 Python 中定义的函数和模块都将驻留在这里。
# 由于在 cpp 中进行定义会导致静态销毁顺序问题，因此选择在 Python 中定义。
_python_cu = torch._C.CompilationUnit()


# python class => ScriptClass mapping
# Python 类到 ScriptClass 的映射
_script_classes: Dict[Type[Any], Type[Any]] = {}
_name_to_pyclass: Dict[str, Type[Any]] = {}


def _add_script_class(python_class, script_class):
    # 将 Python 类和对应的 ScriptClass 添加到 _script_classes 和 _name_to_pyclass 字典中
    _script_classes[python_class] = script_class
    _name_to_pyclass[script_class.qualified_name()] = python_class


def _get_script_class(python_class):
    # 获取给定 Python 类对应的 ScriptClass
    override = getattr(python_class, "_jit_override_qualname", None)
    if override is not None:
        python_class = _get_python_class(override)
    return _script_classes.get(python_class, None)


def _get_python_class(qualified_name):
    # 根据限定名获取 Python 类
    return _name_to_pyclass.get(qualified_name, None)


def _clear_class_state():
    # 清空 _script_classes 和 _name_to_pyclass 字典，用于状态清理
    _script_classes.clear()
    _name_to_pyclass.clear()


# Caching: we currently cache compilation of free functions and overloaded functions.
# To cache free functions we hold a weak ref to the function object and
# map to the compiled fn's qualified name.
# To cache overloaded functions we hold a weak ref to the function obj and
# map to all of its overloaded compiled fns.
# In the future we could consider caching more types of objects so that
# aliasing is preserved across separate compilations of the same object.

# 缓存：目前我们缓存自由函数和重载函数的编译。
# 对于自由函数，我们持有函数对象的弱引用，并映射到编译后函数的限定名。
# 对于重载函数，我们持有函数对象的弱引用，并映射到其所有重载编译后的函数。
# 未来我们可以考虑缓存更多类型的对象，以便在同一对象的不同编译中保留别名。

_jit_caching_layer: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
_jit_function_overload_caching: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def _try_get_jit_cached_overloads(key):
    # 从 _jit_function_overload_caching 字典中获取给定 key 对应的值，如果不存在则返回 None
    qual_names = _jit_function_overload_caching.get(key, None)
    
    # 如果 qual_names 不为空（即存在值），则返回一个列表，列表包含了通过 _python_cu 对象查找每个 qual_name 对应的函数
    if qual_names:
        return [_python_cu.find_function(qual_name) for qual_name in qual_names]
    # 如果 qual_names 为空，则返回 None
    else:
        return None
# 将编译后的函数列表存入 JIT 函数重载缓存中的指定键值
def _set_jit_overload_cache(key, compiled_fns):
    _jit_function_overload_caching[key] = [fn.qualified_name for fn in compiled_fns]

# 尝试从 JIT 函数缓存中获取已缓存的函数
def _try_get_jit_cached_function(key):
    # 如果键对象有禁用 JIT 函数缓存的属性，则返回 None
    if getattr(key, "__disable_jit_function_caching__", False) is True:
        return None
    # 获取键对应的函数的限定名
    qual_name = _jit_caching_layer.get(key, None)
    if qual_name:
        # 根据限定名查找并返回函数对象
        return _python_cu.find_function(qual_name)
    else:
        return None

# 将 JIT 函数缓存中的键值对设置为给定的值
def _set_jit_function_cache(key, value):
    # 断言值是 torch.jit.ScriptFunction 类型的对象
    assert isinstance(value, torch.jit.ScriptFunction)
    # 将键值对存入 JIT 函数缓存层，并使用函数的限定名作为值
    _jit_caching_layer[key] = value.qualified_name
```