# `D:\src\scipysrc\scipy\scipy\_lib\_uarray\__init__.py`

```
"""
.. note:
    如果你正在寻找 NumPy 特定方法的覆盖，请参阅 :obj:`unumpy` 的文档。该页面解释了如何编写后端和多方法。

``uarray`` 建立在后端协议和可覆盖的多方法之上。
定义多方法是必要的，以便后端可以覆盖它们。
请查看 :obj:`generate_multimethod` 的文档，了解如何编写多方法。

让我们从最简单的开始：

``__ua_domain__`` 定义后端的 *域*。域由模块及其子模块组成的点分隔字符串。
例如，如果子模块 ``module2.submodule`` 扩展了 ``module1``
（即它公开了在 ``module1`` 中标记为类型的可分派对象），
则域字符串应为 ``"module1.module2.submodule"``。

为了演示目的，我们将创建一个对象并直接设置其属性。但请注意，您也可以使用模块或自定义类型作为后端。

>>> class Backend: pass
>>> be = Backend()
>>> be.__ua_domain__ = "ua_examples"

此时可能会有必要偏离到 :obj:`generate_multimethod` 的文档中，
了解如何生成一个可被 :obj:`uarray` 覆盖的多方法。无需多言，
编写后端和创建多方法通常是相互独立的活动，了解其中一个并不一定需要了解另一个，
尽管这肯定是有帮助的。我们期望核心 API 的设计者/规范者编写多方法，并由实现者进行覆盖。
但是，通常情况下，相似的人会同时编写两者。

接下来是一个示例多方法：

>>> import uarray as ua
>>> from uarray import Dispatchable
>>> def override_me(a, b):
...   return Dispatchable(a, int),
>>> def override_replacer(args, kwargs, dispatchables):
...     return (dispatchables[0], args[1]), {}
>>> overridden_me = ua.generate_multimethod(
...     override_me, override_replacer, "ua_examples"
... )

接下来是关于覆盖多方法的部分。这需要 ``__ua_function__`` 协议和 ``__ua_convert__`` 协议。
``__ua_function__`` 协议具有签名 ``(method, args, kwargs)``，其中 ``method`` 是传递的多方法，
``args``/``kwargs`` 指定参数，``dispatchables`` 是传入的转换后的可分派对象列表。

>>> def __ua_function__(method, args, kwargs):
...     return method.__name__, args, kwargs
>>> be.__ua_function__ = __ua_function__

另一个感兴趣的协议是 ``__ua_convert__`` 协议。它具有签名 ``(dispatchables, coerce)``。
当 ``coerce`` 为 ``False`` 时，理想情况下格式之间的转换应该是 ``O(1)`` 操作，
但这意味着不应涉及内存复制，只有现有数据的视图。

>>> def __ua_convert__(dispatchables, coerce):
...     for d in dispatchables:
...         if d.type is int:
...             if coerce and d.coercible:
# 从 _backend 模块中导入所有内容
from ._backend import *
# 设置当前模块的版本号
__version__ = '0.8.8.dev0+aa94c5a4.scipy'
```