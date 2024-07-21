# `.\pytorch\torch\_library\simple_registry.py`

```py
# mypy: allow-untyped-defs
# 导入自定义模块.fake_impl中的FakeImplHolder类
from .fake_impl import FakeImplHolder

# 定义__all__变量，指定在使用from ... import *时导出的符号
__all__ = ["SimpleLibraryRegistry", "SimpleOperatorEntry", "singleton"]

# 定义SimpleLibraryRegistry类，用于管理"simple" torch.library APIs的注册
class SimpleLibraryRegistry:
    """Registry for the "simple" torch.library APIs

    The "simple" torch.library APIs are a higher-level API on top of the
    raw PyTorch DispatchKey registration APIs that includes:
    - fake impl

    Registrations for these APIs do not go into the PyTorch dispatcher's
    table because they may not directly involve a DispatchKey. For example,
    the fake impl is a Python function that gets invoked by FakeTensor.
    Instead, we manage them here.

    SimpleLibraryRegistry is a mapping from a fully qualified operator name
    (including the overload) to SimpleOperatorEntry.
    """

    def __init__(self):
        # 初始化_simple库的注册数据存储字典
        self._data = {}

    def find(self, qualname: str) -> "SimpleOperatorEntry":
        # 如果qualname不在_data字典中，创建一个新的SimpleOperatorEntry对象并加入_data中
        if qualname not in self._data:
            self._data[qualname] = SimpleOperatorEntry(qualname)
        # 返回qualname对应的SimpleOperatorEntry对象
        return self._data[qualname]

# 创建全局的singleton对象，类型为SimpleLibraryRegistry类
singleton: SimpleLibraryRegistry = SimpleLibraryRegistry()

# 定义SimpleOperatorEntry类，与一个操作符重载一对一对应
class SimpleOperatorEntry:
    """This is 1:1 to an operator overload.

    The fields of SimpleOperatorEntry are Holders where kernels can be
    registered to.
    """

    def __init__(self, qualname: str):
        # 初始化SimpleOperatorEntry对象，设置qualname和fake_impl
        self.qualname: str = qualname
        self.fake_impl: FakeImplHolder = FakeImplHolder(qualname)

    # 兼容性原因添加的属性，将abstract_impl属性返回fake_impl对象
    @property
    def abstract_impl(self):
        return self.fake_impl
```