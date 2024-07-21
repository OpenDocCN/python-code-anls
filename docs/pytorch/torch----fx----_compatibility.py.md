# `.\pytorch\torch\fx\_compatibility.py`

```
# mypy: allow-untyped-defs
# 引入必要的模块和类型
from typing import Any, Dict
import textwrap

# 创建全局字典，用于存储与兼容性相关的对象
_BACK_COMPAT_OBJECTS: Dict[Any, None] = {}
_MARKED_WITH_COMPATIBILITY: Dict[Any, None] = {}

# 定义一个装饰器函数 `compatibility`
def compatibility(is_backward_compatible: bool):
    # 如果标记为向后兼容
    if is_backward_compatible:

        # 定义内部装饰器函数 `mark_back_compat`
        def mark_back_compat(fn):
            # 获取函数的文档字符串（如果有的话），去除缩进
            docstring = textwrap.dedent(getattr(fn, '__doc__', None) or '')
            # 在文档字符串末尾添加向后兼容的说明
            docstring += """
.. note::
    Backwards-compatibility for this API is guaranteed.
"""
            # 更新函数的文档字符串
            fn.__doc__ = docstring
            # 将函数添加到全局字典 `_BACK_COMPAT_OBJECTS` 中
            _BACK_COMPAT_OBJECTS.setdefault(fn)
            # 将函数添加到全局字典 `_MARKED_WITH_COMPATIBILITY` 中
            _MARKED_WITH_COMPATIBILITY.setdefault(fn)
            # 返回更新后的函数
            return fn

        # 返回装饰器函数 `mark_back_compat`
        return mark_back_compat
    else:
        # 如果不标记为向后兼容

        # 定义内部装饰器函数 `mark_not_back_compat`
        def mark_not_back_compat(fn):
            # 获取函数的文档字符串（如果有的话），去除缩进
            docstring = textwrap.dedent(getattr(fn, '__doc__', None) or '')
            # 在文档字符串末尾添加不向后兼容的警告
            docstring += """
.. warning::
    This API is experimental and is *NOT* backward-compatible.
"""
            # 更新函数的文档字符串
            fn.__doc__ = docstring
            # 将函数添加到全局字典 `_MARKED_WITH_COMPATIBILITY` 中
            _MARKED_WITH_COMPATIBILITY.setdefault(fn)
            # 返回更新后的函数
            return fn

        # 返回装饰器函数 `mark_not_back_compat`
        return mark_not_back_compat
```