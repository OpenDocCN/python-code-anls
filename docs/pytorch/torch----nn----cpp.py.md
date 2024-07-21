# `.\pytorch\torch\nn\cpp.py`

```py
# mypy: allow-untyped-defs
"""Functionality for Python <-> C++ frontend inter-op."""

from torch import nn


class OrderedDictWrapper:
    """A wrapper around a C++ OrderedDict.

    It dynamically evaluates the OrderedDict getter on a bound C++ module, such
    that new changes on the C++ side are picked up. Otherwise accessing e.g.
    ``cpp_module._parameters`` just once would get a frozen copy of the parameters
    at the time of access. ``torch.nn.Module`` accesses ``_parameters`` et al. via ``self.__dict__``
    so using properties does not work.
    """

    def __init__(self, cpp_module, attr):
        self.cpp_module = cpp_module  # 绑定的 C++ 模块实例
        self.attr = attr  # C++ 模块中的属性名称

    @property
    def cpp_dict(self):
        return getattr(self.cpp_module, self.attr)  # 返回绑定模块中指定属性的值

    # Magic methods cannot be assigned dynamically and bypass ``getattr``, so we
    # must manually override them.

    def items(self):
        return self.cpp_dict.items()  # 返回绑定模块中属性字典的键值对

    def keys(self):
        return self.cpp_dict.keys()  # 返回绑定模块中属性字典的键

    def values(self):
        return self.cpp_dict.values()  # 返回绑定模块中属性字典的值

    def __iter__(self):
        return self.cpp_dict.__iter__()  # 实现迭代器协议，返回绑定模块中属性字典的迭代器

    def __len__(self):
        return self.cpp_dict.__len__()  # 返回绑定模块中属性字典的长度

    def __contains__(self, key):
        return self.cpp_dict.__contains__(key)  # 检查给定键是否在绑定模块的属性字典中

    def __getitem__(self, key):
        return self.cpp_dict.__getitem__(key)  # 获取绑定模块中属性字典指定键的值


class ModuleWrapper(nn.Module):
    """A subclass of ``torch.nn.Module`` that wraps a C++ frontend module and delegates all access."""

    def __init__(self, cpp_module):
        # Assign before the super class constructor so ``self.training`` can be
        # assigned to in the super class constructor.
        self.cpp_module = cpp_module  # 绑定的 C++ 前端模块实例
        super().__init__()
        self._parameters = OrderedDictWrapper(cpp_module, "_parameters")  # type: ignore[assignment]
        self._buffers: OrderedDictWrapper = OrderedDictWrapper(cpp_module, "_buffers")  # type: ignore[assignment]
        self._modules: OrderedDictWrapper = OrderedDictWrapper(cpp_module, "_modules")  # type: ignore[assignment]
        for attr in dir(cpp_module):
            # Skip magic methods and the three attributes above.
            if not attr.startswith("_"):  # 跳过魔法方法和上述三个属性
                setattr(self, attr, getattr(self.cpp_module, attr))  # 动态设置实例属性为绑定模块中相应属性的值

    def _apply(self, fn, recurse=True):
        for param in self.parameters():
            # Tensors stored in modules are graph leaves, and we don't
            # want to create copy nodes, so we have to unpack the data.
            param.data = fn(param.data)
            if param._grad is not None:
                param._grad.data = fn(param._grad.data)

        for buf in self.buffers():
            buf.data = fn(buf.data)

        return self

    # nn.Module defines training as a boolean
    @property  # type: ignore[override]
    def training(self):
        return self.cpp_module.training  # 获取绑定模块的训练模式属性值

    @training.setter
    def training(self, mode):
        self.cpp_module.train(mode)  # 设置绑定模块的训练模式
    # 定义对象的字符串表示方法，返回底层 C++ 模块的字符串表示
    def __repr__(self):
        # 调用底层 C++ 模块的 __repr__() 方法并返回其结果
        return self.cpp_module.__repr__()
```