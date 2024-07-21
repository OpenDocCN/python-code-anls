# `.\pytorch\torch\nn\modules\container.py`

```py
# mypy: allow-untyped-defs
# 引入运算符模块，用于操作符相关功能
import operator
# 引入 collections.abc 模块的别名 container_abcs 和 OrderedDict 类
from collections import abc as container_abcs, OrderedDict
# 引入 itertools 模块的 chain 和 islice 函数
from itertools import chain, islice
# 引入 typing 模块中的各种类型声明
from typing import (
    Any,
    Dict,
    Iterable,
    Iterator,
    Mapping,
    Optional,
    overload,
    Tuple,
    TypeVar,
    Union,
)
# 从 typing_extensions 模块引入 deprecated 和 Self
from typing_extensions import deprecated, Self

# 引入 PyTorch 库
import torch
# 从 torch._jit_internal 模块引入 _copy_to_script_wrapper
from torch._jit_internal import _copy_to_script_wrapper
# 从 torch.nn.parameter 模块引入 Parameter 类
from torch.nn.parameter import Parameter

# 从当前包中引入 module 模块的 Module 类
from .module import Module


# 定义 __all__ 列表，指定模块对外公开的接口
__all__ = [
    "Container",
    "Sequential",
    "ModuleList",
    "ModuleDict",
    "ParameterList",
    "ParameterDict",
]

# 定义类型变量 T，限定为 Module 类型
T = TypeVar("T", bound=Module)


# 从 torch.nn.modules.module 模块复制的函数，用于为 ModuleList 提供自定义 __repr__
def _addindent(s_, numSpaces):
    s = s_.split("\n")
    # 对单行内容不进行处理
    if len(s) == 1:
        return s_
    # 移除第一行，并对后续每一行添加指定数量的空格
    first = s.pop(0)
    s = [(numSpaces * " ") + line for line in s]
    # 将处理后的行重新组合成字符串
    s = "\n".join(s)
    # 将第一行添加回去，并返回完整的字符串
    s = first + "\n" + s
    return s


# 被弃用的 Container 类，继承自 Module 类
@deprecated(
    "`nn.Container` is deprecated. "
    "All of it's functionality is now implemented in `nn.Module`. Subclass that instead.",
    category=FutureWarning,
)
class Container(Module):
    # 初始化方法，接收任意关键字参数
    def __init__(self, **kwargs: Any) -> None:
        # 调用父类 Module 的初始化方法
        super().__init__()
        # 遍历关键字参数，逐个添加为模块
        for key, value in kwargs.items():
            self.add_module(key, value)


# Sequential 类，继承自 Module 类
class Sequential(Module):
    # 顺序容器，按顺序添加模块
    r"""A sequential container.

    Modules will be added to it in the order they are passed in the
    constructor. Alternatively, an ``OrderedDict`` of modules can be
    passed in. The ``forward()`` method of ``Sequential`` accepts any
    input and forwards it to the first module it contains. It then
    "chains" outputs to inputs sequentially for each subsequent module,
    finally returning the output of the last module.

    The value a ``Sequential`` provides over manually calling a sequence
    of modules is that it allows treating the whole container as a
    single module, such that performing a transformation on the
    ``Sequential`` applies to each of the modules it stores (which are
    each a registered submodule of the ``Sequential``).

    What's the difference between a ``Sequential`` and a
    :class:`torch.nn.ModuleList`? A ``ModuleList`` is exactly what it
    sounds like--a list for storing ``Module`` s! On the other hand,
    the layers in a ``Sequential`` are connected in a cascading way.
    """

    # 以下省略了部分类的定义，因为超出了示例代码块的长度限制
    # 创建一个顺序模型 `Sequential`，用于按顺序组合各层模块
    model = nn.Sequential(
              nn.Conv2d(1,20,5),   # 添加一个二维卷积层，输入通道数为1，输出通道数为20，卷积核大小为5
              nn.ReLU(),           # 添加ReLU激活函数层
              nn.Conv2d(20,64,5),  # 添加第二个二维卷积层，输入通道数为20，输出通道数为64，卷积核大小为5
              nn.ReLU()            # 添加第二个ReLU激活函数层
            )

    # 使用带有 `OrderedDict` 的 `Sequential`，与上述代码功能上相同
    model = nn.Sequential(OrderedDict([
              ('conv1', nn.Conv2d(1,20,5)),  # 使用指定名称添加第一个二维卷积层
              ('relu1', nn.ReLU()),          # 使用指定名称添加第一个ReLU激活函数层
              ('conv2', nn.Conv2d(20,64,5)), # 使用指定名称添加第二个二维卷积层
              ('relu2', nn.ReLU())           # 使用指定名称添加第二个ReLU激活函数层
            ]))
    """

    _modules: Dict[str, Module]  # type: ignore[assignment]

    @overload
    def __init__(self, *args: Module) -> None:
        ...

    @overload
    def __init__(self, arg: "OrderedDict[str, Module]") -> None:
        ...

    def __init__(self, *args):
        # 调用父类的初始化方法
        super().__init__()
        # 如果传入的参数长度为1且为OrderedDict类型，则按顺序添加模块
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            # 否则，按顺序添加每个模块
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)

    def _get_item_by_idx(self, iterator, idx) -> T:  # type: ignore[misc, type-var]
        """获取迭代器中索引为idx的元素。"""
        size = len(self)
        idx = operator.index(idx)
        if not -size <= idx < size:
            raise IndexError(f"index {idx} is out of range")
        idx %= size
        return next(islice(iterator, idx, None))

    @_copy_to_script_wrapper
    def __getitem__(self, idx: Union[slice, int]) -> Union["Sequential", T]:
        """获取指定索引或切片范围内的子序列。"""
        if isinstance(idx, slice):
            return self.__class__(OrderedDict(list(self._modules.items())[idx]))
        else:
            return self._get_item_by_idx(self._modules.values(), idx)

    def __setitem__(self, idx: int, module: Module) -> None:
        """设置指定索引处的模块。"""
        key: str = self._get_item_by_idx(self._modules.keys(), idx)
        return setattr(self, key, module)

    def __delitem__(self, idx: Union[slice, int]) -> None:
        """删除指定索引或切片范围内的模块，并保持编号顺序。"""
        if isinstance(idx, slice):
            for key in list(self._modules.keys())[idx]:
                delattr(self, key)
        else:
            key = self._get_item_by_idx(self._modules.keys(), idx)
            delattr(self, key)
        # 重新编号以保持顺序
        str_indices = [str(i) for i in range(len(self._modules))]
        self._modules = OrderedDict(list(zip(str_indices, self._modules.values())))

    @_copy_to_script_wrapper
    def __len__(self) -> int:
        """返回模块列表的长度。"""
        return len(self._modules)
    # 定义一个特殊方法 __add__，用于实现对象的加法操作，返回一个新的 Sequential 对象
    def __add__(self, other) -> "Sequential":
        # 检查 other 是否是 Sequential 类型的对象
        if isinstance(other, Sequential):
            # 创建一个新的 Sequential 对象 ret
            ret = Sequential()
            # 将当前对象 self 的所有层逐一添加到 ret 中
            for layer in self:
                ret.append(layer)
            # 将对象 other 的所有层逐一添加到 ret 中
            for layer in other:
                ret.append(layer)
            return ret
        else:
            # 如果 other 不是 Sequential 类型，则抛出 ValueError 异常
            raise ValueError(
                "add operator supports only objects "
                f"of Sequential class, but {str(type(other))} is given."
            )

    # 定义 pop 方法，用于从 Sequential 对象中移除指定索引或切片对应的模块，并返回该模块
    def pop(self, key: Union[int, slice]) -> Module:
        # 获取指定索引或切片对应的模块 v
        v = self[key]
        # 从 Sequential 对象中删除指定索引或切片对应的模块
        del self[key]
        # 返回被删除的模块 v
        return v

    # 定义特殊方法 __iadd__，实现就地加法操作，将另一个 Sequential 对象的所有模块添加到当前对象中
    def __iadd__(self, other) -> Self:
        # 检查 other 是否是 Sequential 类型的对象
        if isinstance(other, Sequential):
            # 记录当前对象中模块的数量，作为添加新模块时的偏移量
            offset = len(self)
            # 遍历 other 对象中的每个模块，并将其添加到当前对象中
            for i, module in enumerate(other):
                self.add_module(str(i + offset), module)
            return self
        else:
            # 如果 other 不是 Sequential 类型，则抛出 ValueError 异常
            raise ValueError(
                "add operator supports only objects "
                f"of Sequential class, but {str(type(other))} is given."
            )

    # 定义特殊方法 __mul__，实现对象的乘法操作，复制当前对象中的所有模块若干次，并返回新的 Sequential 对象
    def __mul__(self, other: int) -> "Sequential":
        # 检查 other 是否为整数类型
        if not isinstance(other, int):
            # 如果 other 不是整数类型，则抛出 TypeError 异常
            raise TypeError(
                f"unsupported operand type(s) for *: {type(self)} and {type(other)}"
            )
        # 检查 other 是否为正整数
        elif other <= 0:
            # 如果 other 不是正整数，则抛出 ValueError 异常
            raise ValueError(
                f"Non-positive multiplication factor {other} for {type(self)}"
            )
        else:
            # 创建一个新的 Sequential 对象 combined
            combined = Sequential()
            # 设置偏移量为 0
            offset = 0
            # 将当前对象中的每个模块复制 other 次，添加到 combined 对象中
            for _ in range(other):
                for module in self:
                    combined.add_module(str(offset), module)
                    offset += 1
            return combined

    # 定义特殊方法 __rmul__，实现对象的右乘法操作，即乘法操作的反向版本
    def __rmul__(self, other: int) -> "Sequential":
        # 直接调用 __mul__ 方法实现右乘法操作
        return self.__mul__(other)

    # 定义特殊方法 __imul__，实现对象的就地乘法操作，复制当前对象中的所有模块若干次，并将复制后的模块直接添加到当前对象中
    def __imul__(self, other: int) -> Self:
        # 检查 other 是否为整数类型
        if not isinstance(other, int):
            # 如果 other 不是整数类型，则抛出 TypeError 异常
            raise TypeError(
                f"unsupported operand type(s) for *: {type(self)} and {type(other)}"
            )
        # 检查 other 是否为正整数
        elif other <= 0:
            # 如果 other 不是正整数，则抛出 ValueError 异常
            raise ValueError(
                f"Non-positive multiplication factor {other} for {type(self)}"
            )
        else:
            # 记录当前对象中模块的数量
            len_original = len(self)
            # 设置偏移量为当前对象中模块的数量
            offset = len(self)
            # 将当前对象中的每个模块复制 other-1 次，并将复制后的模块添加到当前对象中
            for _ in range(other - 1):
                for i in range(len_original):
                    self.add_module(str(i + offset), self._modules[str(i)])
                offset += len_original
            return self

    # 使用装饰器 @_copy_to_script_wrapper 装饰，定义特殊方法 __dir__，返回对象的属性列表，并过滤掉所有由数字组成的属性名
    @_copy_to_script_wrapper
    def __dir__(self):
        keys = super().__dir__()
        keys = [key for key in keys if not key.isdigit()]
        return keys

    # 使用装饰器 @_copy_to_script_wrapper 装饰，定义特殊方法 __iter__，返回当前对象中模块的迭代器
    def __iter__(self) -> Iterator[Module]:
        return iter(self._modules.values())

    # 注释：由于该方法的输入类型可能动态变化，因此无法进行类型检查，函数注释说明了这一点
    # NB: We can't really type check this function as the type of input
    # may change dynamically (as is tested in
    # TestScript.test_sequential_intermediary_types).  Cannot annotate
    # with Any as TorchScript expects a more precise type
    # 对于给定的输入数据，依次通过每个模块进行前向传播计算，并更新输入数据
    def forward(self, input):
        for module in self:
            input = module(input)
        return input

    # 将指定的模块添加到当前 Sequential 对象的末尾
    def append(self, module: Module) -> "Sequential":
        # 使用当前 Sequential 对象的长度作为新模块的索引，将模块添加到内部模块字典中
        self.add_module(str(len(self)), module)
        return self

    # 在指定索引处插入给定的模块到当前 Sequential 对象中
    def insert(self, index: int, module: Module) -> "Sequential":
        # 检查模块类型是否为 Module 类的实例
        if not isinstance(module, Module):
            raise AssertionError(f"module should be of type: {Module}")
        
        n = len(self._modules)
        # 检查索引是否在合法范围内
        if not (-n <= index <= n):
            raise IndexError(f"Index out of range: {index}")
        
        # 处理负索引，使其转换为正索引
        if index < 0:
            index += n
        
        # 将当前索引及之后的模块依次向后移动一个位置，为新模块腾出空间并插入
        for i in range(n, index, -1):
            self._modules[str(i)] = self._modules[str(i - 1)]
        self._modules[str(index)] = module
        return self

    # 将给定的 Sequential 对象中的每个层逐一添加到当前 Sequential 对象的末尾
    def extend(self, sequential) -> "Sequential":
        for layer in sequential:
            self.append(layer)
        return self
class ModuleList(Module):
    r"""Holds submodules in a list.

    :class:`~torch.nn.ModuleList` can be indexed like a regular Python list, but
    modules it contains are properly registered, and will be visible by all
    :class:`~torch.nn.Module` methods.

    Args:
        modules (iterable, optional): an iterable of modules to add

    Example::

        class MyModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.linears = nn.ModuleList([nn.Linear(10, 10) for i in range(10)])

            def forward(self, x):
                # ModuleList can act as an iterable, or be indexed using ints
                for i, l in enumerate(self.linears):
                    x = self.linears[i // 2](x) + l(x)
                return x
    """

    _modules: Dict[str, Module]  # type: ignore[assignment]

    def __init__(self, modules: Optional[Iterable[Module]] = None) -> None:
        super().__init__()
        if modules is not None:
            self += modules
        # 初始化 ModuleList 类，继承自 Module 类

    def _get_abs_string_index(self, idx):
        """Get the absolute index for the list of modules."""
        idx = operator.index(idx)  # 将 idx 转换为整数索引
        if not (-len(self) <= idx < len(self)):
            raise IndexError(f"index {idx} is out of range")  # 抛出索引超出范围的异常
        if idx < 0:
            idx += len(self)  # 处理负索引，转换为正索引
        return str(idx)  # 返回索引的字符串表示形式

    @_copy_to_script_wrapper
    def __getitem__(self, idx: Union[int, slice]) -> Union[Module, "ModuleList"]:
        if isinstance(idx, slice):
            return self.__class__(list(self._modules.values())[idx])  # 返回切片后的新 ModuleList 对象
        else:
            return self._modules[self._get_abs_string_index(idx)]  # 返回指定索引的模块对象

    def __setitem__(self, idx: int, module: Module) -> None:
        idx = self._get_abs_string_index(idx)  # 获取绝对索引
        return setattr(self, str(idx), module)  # 设置指定索引位置的模块对象

    def __delitem__(self, idx: Union[int, slice]) -> None:
        if isinstance(idx, slice):
            for k in range(len(self._modules))[idx]:  # 遍历并删除切片范围内的模块对象
                delattr(self, str(k))
        else:
            delattr(self, self._get_abs_string_index(idx))  # 删除指定索引位置的模块对象
        # 重新构建 self._modules，以保持编号的连续性
        str_indices = [str(i) for i in range(len(self._modules))]
        self._modules = OrderedDict(list(zip(str_indices, self._modules.values())))

    @_copy_to_script_wrapper
    def __len__(self) -> int:
        return len(self._modules)  # 返回模块列表的长度

    @_copy_to_script_wrapper
    def __iter__(self) -> Iterator[Module]:
        return iter(self._modules.values())  # 返回模块列表的迭代器，按值迭代

    def __iadd__(self, modules: Iterable[Module]) -> Self:
        return self.extend(modules)  # 添加模块列表，返回扩展后的自身引用

    def __add__(self, other: Iterable[Module]) -> "ModuleList":
        combined = ModuleList()
        for i, module in enumerate(chain(self, other)):  # 将两个 ModuleList 合并成一个新的 ModuleList
            combined.add_module(str(i), module)
        return combined  # 返回合并后的 ModuleList 对象
    # 返回自定义的 ModuleList 的字符串表示形式，压缩重复模块的表示
    def __repr__(self):
        # 获取列表中每个模块的字符串表示形式
        list_of_reprs = [repr(item) for item in self]
        # 如果列表为空，返回模块列表名称
        if len(list_of_reprs) == 0:
            return self._get_name() + "()"

        # 初始化重复块列表和起始结束索引
        start_end_indices = [[0, 0]]
        repeated_blocks = [list_of_reprs[0]]
        # 遍历列表中的模块表示形式，找到重复块并记录起始和结束索引
        for i, r in enumerate(list_of_reprs[1:], 1):
            if r == repeated_blocks[-1]:
                start_end_indices[-1][1] += 1
                continue

            start_end_indices.append([i, i])
            repeated_blocks.append(r)

        # 构建主字符串
        lines = []
        main_str = self._get_name() + "("
        # 遍历重复块和其起始结束索引，构建每个块的局部表示形式
        for (start_id, end_id), b in zip(start_end_indices, repeated_blocks):
            local_repr = f"({start_id}): {b}"  # 默认表示形式

            # 如果起始和结束索引不相同，表示有重复的块
            if start_id != end_id:
                n = end_id - start_id + 1
                local_repr = f"({start_id}-{end_id}): {n} x {b}"

            # 添加缩进后的局部表示形式到行列表中
            local_repr = _addindent(local_repr, 2)
            lines.append(local_repr)

        # 将所有行连接到主字符串中，并添加结尾的括号
        main_str += "\n  " + "\n  ".join(lines) + "\n"
        main_str += ")"
        return main_str

    # 装饰器：将函数标记为脚本复制的包装器
    @_copy_to_script_wrapper
    def __dir__(self):
        # 调用父类的 __dir__ 方法获取所有属性名称
        keys = super().__dir__()
        # 过滤掉所有数字字符串作为属性名称的键
        keys = [key for key in keys if not key.isdigit()]
        return keys

    # 在给定索引之前插入指定模块到列表中
    def insert(self, index: int, module: Module) -> None:
        # 将索引之后的模块向后移动一个位置
        for i in range(len(self._modules), index, -1):
            self._modules[str(i)] = self._modules[str(i - 1)]
        # 在索引位置插入新的模块
        self._modules[str(index)] = module

    # 将指定模块追加到列表的末尾
    def append(self, module: Module) -> "ModuleList":
        # 使用模块列表的长度作为新模块的索引，并添加到列表中
        self.add_module(str(len(self)), module)
        return self

    # 弹出指定键的模块并返回
    def pop(self, key: Union[int, slice]) -> Module:
        v = self[key]
        del self[key]
        return v

    # 将 Python 可迭代对象中的模块追加到列表的末尾
    def extend(self, modules: Iterable[Module]) -> Self:
        # 检查 modules 是否是可迭代对象，否则抛出类型错误
        if not isinstance(modules, container_abcs.Iterable):
            raise TypeError(
                "ModuleList.extend should be called with an "
                "iterable, but got " + type(modules).__name__
            )
        # 计算偏移量，以确定从哪里开始添加模块
        offset = len(self)
        # 遍历可迭代对象中的模块，并添加到模块列表中
        for i, module in enumerate(modules):
            self.add_module(str(offset + i), module)
        return self

    # 完全删除 forward 方法，使用 Module 的 _forward_unimplemented 作为后备
class ModuleDict(Module):
    r"""Holds submodules in a dictionary.

    :class:`~torch.nn.ModuleDict` can be indexed like a regular Python dictionary,
    but modules it contains are properly registered, and will be visible by all
    :class:`~torch.nn.Module` methods.

    :class:`~torch.nn.ModuleDict` is an **ordered** dictionary that respects

    * the order of insertion, and

    * in :meth:`~torch.nn.ModuleDict.update`, the order of the merged
      ``OrderedDict``, ``dict`` (started from Python 3.6) or another
      :class:`~torch.nn.ModuleDict` (the argument to
      :meth:`~torch.nn.ModuleDict.update`).

    Note that :meth:`~torch.nn.ModuleDict.update` with other unordered mapping
    types (e.g., Python's plain ``dict`` before Python version 3.6) does not
    preserve the order of the merged mapping.

    Args:
        modules (iterable, optional): a mapping (dictionary) of (string: module)
            or an iterable of key-value pairs of type (string, module)

    Example::

        class MyModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.choices = nn.ModuleDict({
                        'conv': nn.Conv2d(10, 10, 3),
                        'pool': nn.MaxPool2d(3)
                })
                self.activations = nn.ModuleDict([
                        ['lrelu', nn.LeakyReLU()],
                        ['prelu', nn.PReLU()]
                ])

            def forward(self, x, choice, act):
                x = self.choices[choice](x)
                x = self.activations[act](x)
                return x
    """

    _modules: Dict[str, Module]  # type: ignore[assignment]
    # _modules 字段用于存储模块字典，其中键为字符串，值为模块对象

    def __init__(self, modules: Optional[Mapping[str, Module]] = None) -> None:
        # 初始化方法，继承自父类 Module
        super().__init__()
        # 如果传入了 modules 参数，则调用 update 方法将其添加到 ModuleDict 中
        if modules is not None:
            self.update(modules)

    @_copy_to_script_wrapper
    def __getitem__(self, key: str) -> Module:
        # 根据键获取对应的模块对象
        return self._modules[key]

    def __setitem__(self, key: str, module: Module) -> None:
        # 添加或更新指定键的模块对象
        self.add_module(key, module)

    def __delitem__(self, key: str) -> None:
        # 删除指定键的模块对象
        del self._modules[key]

    @_copy_to_script_wrapper
    def __len__(self) -> int:
        # 返回 ModuleDict 中存储的模块数量
        return len(self._modules)

    @_copy_to_script_wrapper
    def __iter__(self) -> Iterator[str]:
        # 返回 ModuleDict 中键的迭代器
        return iter(self._modules)

    @_copy_to_script_wrapper
    def __contains__(self, key: str) -> bool:
        # 检查指定键是否在 ModuleDict 中
        return key in self._modules

    def clear(self) -> None:
        """Remove all items from the ModuleDict."""
        # 清空 ModuleDict 中所有的模块对象
        self._modules.clear()

    def pop(self, key: str) -> Module:
        r"""Remove key from the ModuleDict and return its module.

        Args:
            key (str): key to pop from the ModuleDict
        """
        # 弹出并返回指定键对应的模块对象
        v = self[key]
        del self[key]
        return v

    @_copy_to_script_wrapper
    def keys(self) -> Iterable[str]:
        r"""Return an iterable of the ModuleDict keys."""
        # 返回 ModuleDict 中所有键的可迭代对象
        return self._modules.keys()

    @_copy_to_script_wrapper
    def values(self) -> Iterable[Module]:
        r"""Return an iterable of the ModuleDict values."""
        # 返回 ModuleDict 中所有值（即模块对象）的可迭代对象
        return self._modules.values()
    # 返回一个可迭代的 ModuleDict 键值对的 Iterable
    def items(self) -> Iterable[Tuple[str, Module]]:
        r"""Return an iterable of the ModuleDict key/value pairs."""
        return self._modules.items()

    # 将 values 方法装饰为脚本版本的拷贝
    @_copy_to_script_wrapper
    def values(self) -> Iterable[Module]:
        r"""Return an iterable of the ModuleDict values."""
        return self._modules.values()

    # 使用给定的映射更新 ModuleDict 的键值对，覆盖现有键
    def update(self, modules: Mapping[str, Module]) -> None:
        r"""Update the :class:`~torch.nn.ModuleDict` with key-value pairs from a mapping, overwriting existing keys.

        .. note::
            If :attr:`modules` is an ``OrderedDict``, a :class:`~torch.nn.ModuleDict`, or
            an iterable of key-value pairs, the order of new elements in it is preserved.

        Args:
            modules (iterable): a mapping (dictionary) from string to :class:`~torch.nn.Module`,
                or an iterable of key-value pairs of type (string, :class:`~torch.nn.Module`)
        """
        # 如果 modules 不是 Iterable，则抛出 TypeError 异常
        if not isinstance(modules, container_abcs.Iterable):
            raise TypeError(
                "ModuleDict.update should be called with an "
                "iterable of key/value pairs, but got " + type(modules).__name__
            )

        # 如果 modules 是 OrderedDict, ModuleDict 或 Mapping 类型之一，则保持新元素的顺序
        if isinstance(modules, (OrderedDict, ModuleDict, container_abcs.Mapping)):
            # 遍历 modules 的键值对，并将其添加到 self 中
            for key, module in modules.items():
                self[key] = module
        else:
            # 否则，假设 modules 是一个由键值对组成的可迭代对象
            # 遍历 modules 中的每个项目，确保每个项目是一个长度为 2 的 Iterable
            for j, m in enumerate(modules):
                if not isinstance(m, container_abcs.Iterable):
                    raise TypeError(
                        "ModuleDict update sequence element "
                        "#" + str(j) + " should be Iterable; is" + type(m).__name__
                    )
                if not len(m) == 2:
                    raise ValueError(
                        "ModuleDict update sequence element "
                        "#" + str(j) + " has length " + str(len(m)) + "; 2 is required"
                    )
                # 将 modules 的每个项目作为键值对添加到 self 中
                self[m[0]] = m[1]  # type: ignore[assignment]

    # 移除 forward 方法以便完全依赖于 Module 的 _forward_unimplemented 方法
# 定义一个继承自 Module 的类 ParameterList，用于管理参数列表
class ParameterList(Module):
    r"""Holds parameters in a list.

    :class:`~torch.nn.ParameterList` can be used like a regular Python
    list, but Tensors that are :class:`~torch.nn.Parameter` are properly registered,
    and will be visible by all :class:`~torch.nn.Module` methods.

    Note that the constructor, assigning an element of the list, the
    :meth:`~torch.nn.ParameterList.append` method and the :meth:`~torch.nn.ParameterList.extend`
    method will convert any :class:`~torch.Tensor` into :class:`~torch.nn.Parameter`.

    Args:
        parameters (iterable, optional): an iterable of elements to add to the list.

    Example::

        class MyModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.params = nn.ParameterList([nn.Parameter(torch.randn(10, 10)) for i in range(10)])

            def forward(self, x):
                # ParameterList can act as an iterable, or be indexed using ints
                for i, p in enumerate(self.params):
                    x = self.params[i // 2].mm(x) + p.mm(x)
                return x
    """

    # 初始化方法，设置初始大小为 0，并根据传入的 values 添加元素
    def __init__(self, values: Optional[Iterable[Any]] = None) -> None:
        super().__init__()
        self._size = 0
        if values is not None:
            self += values

    # 辅助方法，获取模块列表的绝对索引字符串表示
    def _get_abs_string_index(self, idx):
        """Get the absolute index for the list of modules."""
        idx = operator.index(idx)
        if not (-len(self) <= idx < len(self)):
            raise IndexError(f"index {idx} is out of range")
        if idx < 0:
            idx += len(self)
        return str(idx)

    # 索引获取方法的重载，支持通过整数或切片索引获取元素
    @overload
    def __getitem__(self, idx: int) -> Any:
        ...

    @overload
    def __getitem__(self: T, idx: slice) -> T:
        ...

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            start, stop, step = idx.indices(len(self))
            out = self.__class__()
            for i in range(start, stop, step):
                out.append(self[i])
            return out
        else:
            idx = self._get_abs_string_index(idx)
            return getattr(self, str(idx))

    # 索引设置方法，根据索引将参数转换为 Parameter 对象并添加到列表中
    def __setitem__(self, idx: int, param: Any) -> None:
        # Note that all other function that add an entry to the list part of
        # the ParameterList end up here. So this is the only place where we need
        # to wrap things into Parameter if needed.
        # Objects added via setattr() are not in the list part and thus won't
        # call into this function.
        idx = self._get_abs_string_index(idx)
        if isinstance(param, torch.Tensor) and not isinstance(param, Parameter):
            param = Parameter(param)
        return setattr(self, str(idx), param)

    # 返回列表的长度
    def __len__(self) -> int:
        return self._size

    # 返回一个迭代器，用于迭代访问列表中的元素
    def __iter__(self) -> Iterator[Any]:
        return iter(self[i] for i in range(len(self)))

    # 实现就地加法操作，将参数列表扩展
    def __iadd__(self, parameters: Iterable[Any]) -> 'ParameterList':
        return self.extend(parameters)
    def __dir__(self):
        # 调用父类的 __dir__ 方法获取所有属性名
        keys = super().__dir__()
        # 过滤掉属性名为数字的键
        keys = [key for key in keys if not key.isdigit()]
        return keys

    def append(self, value: Any) -> "ParameterList":
        """Append a given value at the end of the list.

        Args:
            value (Any): 要追加的数值
        """
        # 计算新元素的索引并增加列表大小
        new_idx = len(self)
        self._size += 1
        # 将值添加到指定索引处
        self[new_idx] = value
        return self

    def extend(self, values: Iterable[Any]) -> Self:
        """Append values from a Python iterable to the end of the list.

        Args:
            values (iterable): 要追加的值的可迭代对象
        """
        # Tensor 是可迭代的，但这里不希望解包它
        if not isinstance(values, container_abcs.Iterable) or isinstance(
            values, torch.Tensor
        ):
            raise TypeError(
                "ParameterList.extend 应该传入一个可迭代对象，但传入了 " + type(values).__name__
            )
        # 遍历并逐个追加值到列表中
        for value in values:
            self.append(value)
        return self

    def extra_repr(self) -> str:
        # 生成描述对象的额外信息的字符串表示
        child_lines = []
        for k, p in enumerate(self):
            if isinstance(p, torch.Tensor):
                # 如果元素是 Tensor，则生成相应的描述信息
                size_str = "x".join(str(size) for size in p.size())
                if p.device.type in ["cuda", torch._C._get_privateuse1_backend_name()]:
                    device_str = f" ({p.device})"
                else:
                    device_str = ""
                parastr = "{} containing: [{} of size {}{}]".format(
                    "Parameter" if isinstance(p, Parameter) else "Tensor",
                    p.dtype,
                    size_str,
                    device_str,
                )
                child_lines.append("  (" + str(k) + "): " + parastr)
            else:
                # 如果元素不是 Tensor，则生成相应的描述信息
                child_lines.append(
                    "  (" + str(k) + "): Object of type: " + type(p).__name__
                )

        tmpstr = "\n".join(child_lines)
        return tmpstr

    def __call__(self, *args, **kwargs):
        # 不允许调用 ParameterList 实例
        raise RuntimeError("ParameterList 不应该被调用。")
class ParameterDict(Module):
    r"""Holds parameters in a dictionary.

    ParameterDict can be indexed like a regular Python dictionary, but Parameters it
    contains are properly registered, and will be visible by all Module methods.
    Other objects are treated as would be done by a regular Python dictionary

    :class:`~torch.nn.ParameterDict` is an **ordered** dictionary.
    :meth:`~torch.nn.ParameterDict.update` with other unordered mapping
    types (e.g., Python's plain ``dict``) does not preserve the order of the
    merged mapping. On the other hand, ``OrderedDict`` or another :class:`~torch.nn.ParameterDict`
    will preserve their ordering.

    Note that the constructor, assigning an element of the dictionary and the
    :meth:`~torch.nn.ParameterDict.update` method will convert any :class:`~torch.Tensor` into
    :class:`~torch.nn.Parameter`.

    Args:
        values (iterable, optional): a mapping (dictionary) of
            (string : Any) or an iterable of key-value pairs
            of type (string, Any)

    Example::

        class MyModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.params = nn.ParameterDict({
                        'left': nn.Parameter(torch.randn(5, 10)),
                        'right': nn.Parameter(torch.randn(5, 10))
                })

            def forward(self, x, choice):
                x = self.params[choice].mm(x)
                return x
    """

    def __init__(self, parameters: Any = None) -> None:
        # 调用父类构造函数
        super().__init__()
        # 初始化一个空的字典，用于存储参数的键
        self._keys: Dict[str, None] = {}
        # 如果传入了参数，则调用 update 方法添加参数
        if parameters is not None:
            self.update(parameters)

    def _key_to_attr(self, key: str) -> str:
        # 如果键不是字符串类型，则抛出类型错误
        if not isinstance(key, str):
            raise TypeError(
                "Index given to ParameterDict cannot be used as a key as it is "
                f"not a string (type is '{type(key).__name__}'). Open an issue on "
                "github if you need non-string keys."
            )
        else:
            # 返回键本身作为属性名
            return key

    def __getitem__(self, key: str) -> Any:
        # 将键转换为属性名并返回相应属性的值
        attr = self._key_to_attr(key)
        return getattr(self, attr)

    def __setitem__(self, key: str, value: Any) -> None:
        # 将键添加到键字典中
        self._keys[key] = None
        # 将键转换为属性名
        attr = self._key_to_attr(key)
        # 如果值是 torch.Tensor 类型但不是 Parameter 类型，则转换为 Parameter 类型
        if isinstance(value, torch.Tensor) and not isinstance(value, Parameter):
            value = Parameter(value)
        # 使用 setattr 设置属性值
        setattr(self, attr, value)
    def __delitem__(self, key: str) -> None:
        # 删除 self._keys 中的键值对，键为 key
        del self._keys[key]
        # 根据 key 获取相应的属性名，并删除当前对象中的该属性
        attr = self._key_to_attr(key)
        delattr(self, attr)

    def __len__(self) -> int:
        # 返回 self._keys 中键值对的数量
        return len(self._keys)

    def __iter__(self) -> Iterator[str]:
        # 返回一个迭代器，迭代 self._keys 中的所有键
        return iter(self._keys)

    def __reversed__(self) -> Iterator[str]:
        # 返回一个逆序迭代器，逆序迭代 self._keys 中的所有键
        return reversed(list(self._keys))

    def copy(self) -> "ParameterDict":
        """Return a copy of this :class:`~torch.nn.ParameterDict` instance."""
        # 返回当前 ParameterDict 实例的一个副本，使用 OrderedDict 来保持顺序
        # 因为 ParameterDict 在构造函数中对普通 dict 和 OrderedDict 表现不同
        return ParameterDict(OrderedDict((k, self[k]) for k in self._keys))

    def __contains__(self, key: str) -> bool:
        # 判断 key 是否在 self._keys 中
        return key in self._keys

    def setdefault(self, key: str, default: Optional[Any] = None) -> Any:
        """Set the default for a key in the Parameterdict.

        If key is in the ParameterDict, return its value.
        If not, insert `key` with a parameter `default` and return `default`.
        `default` defaults to `None`.

        Args:
            key (str): key to set default for
            default (Any): the parameter set to the key
        """
        # 如果 key 不在 self 中，则将其设置为 default
        if key not in self:
            self[key] = default
        # 返回 key 对应的值
        return self[key]

    def clear(self) -> None:
        """Remove all items from the ParameterDict."""
        # 使用循环删除 self._keys 中的所有键值对
        for k in self._keys.copy():
            del self[k]

    def pop(self, key: str) -> Any:
        r"""Remove key from the ParameterDict and return its parameter.

        Args:
            key (str): key to pop from the ParameterDict
        """
        # 获取 key 对应的值
        v = self[key]
        # 删除 key 及其对应的值
        del self[key]
        # 返回 key 对应的值
        return v

    def popitem(self) -> Tuple[str, Any]:
        """Remove and return the last inserted `(key, parameter)` pair from the ParameterDict."""
        # 弹出并返回最后插入的 (key, parameter) 对
        k, _ = self._keys.popitem()
        # 将弹出的 key 重新插入 self._keys 中，值设为 None，以便访问/删除
        self._keys[k] = None
        # 获取弹出的 key 对应的值
        val = self[k]
        # 删除 key 及其对应的值
        del self[k]
        # 返回 key 和其对应的值
        return k, val

    def get(self, key: str, default: Optional[Any] = None) -> Any:
        r"""Return the parameter associated with key if present. Otherwise return default if provided, None if not.

        Args:
            key (str): key to get from the ParameterDict
            default (Parameter, optional): value to return if key not present
        """
        # 如果 key 在 self 中，返回其对应的值；否则返回 default（默认为 None）
        return self[key] if key in self else default

    def fromkeys(
        self, keys: Iterable[str], default: Optional[Any] = None
    ) -> "ParameterDict":
        r"""Return a new ParameterDict with the keys provided.

        Args:
            keys (iterable, string): keys to make the new ParameterDict from
            default (Parameter, optional): value to set for all keys
        """
        # 使用 keys 中的键创建一个新的 ParameterDict 实例，所有键的值为 default
        return ParameterDict((k, default) for k in keys)

    def keys(self) -> Iterable[str]:
        r"""Return an iterable of the ParameterDict keys."""
        # 返回 self._keys 中所有键的可迭代对象
        return self._keys.keys()
    # 返回一个迭代器，迭代返回 ParameterDict 的键值对元组
    def items(self) -> Iterable[Tuple[str, Any]]:
        r"""Return an iterable of the ParameterDict key/value pairs."""
        return ((k, self[k]) for k in self._keys)

    # 返回一个迭代器，迭代返回 ParameterDict 的值
    def values(self) -> Iterable[Any]:
        r"""Return an iterable of the ParameterDict values."""
        return (self[k] for k in self._keys)

    # 更新 ParameterDict 中的键值对，从 parameters 中获取，覆盖已有的键
    def update(self, parameters: Union[Mapping[str, Any], "ParameterDict"]) -> None:
        r"""Update the :class:`~torch.nn.ParameterDict` with key-value pairs from ``parameters``, overwriting existing keys.

        .. note::
            If :attr:`parameters` is an ``OrderedDict``, a :class:`~torch.nn.ParameterDict`, or
            an iterable of key-value pairs, the order of new elements in it is preserved.

        Args:
            parameters (iterable): a mapping (dictionary) from string to
                :class:`~torch.nn.Parameter`, or an iterable of
                key-value pairs of type (string, :class:`~torch.nn.Parameter`)
        """
        # 如果 parameters 不是可迭代对象，则抛出类型错误
        if not isinstance(parameters, container_abcs.Iterable):
            raise TypeError(
                "ParametersDict.update should be called with an "
                "iterable of key/value pairs, but got " + type(parameters).__name__
            )

        # 如果 parameters 是 OrderedDict 或 ParameterDict 类型，则保持顺序地更新键值对
        if isinstance(parameters, (OrderedDict, ParameterDict)):
            for key, parameter in parameters.items():
                self[key] = parameter
        # 如果 parameters 是普通的映射类型，则按键排序后更新键值对
        elif isinstance(parameters, container_abcs.Mapping):
            for key, parameter in sorted(parameters.items()):
                self[key] = parameter
        else:
            # 如果 parameters 是其他类型的可迭代对象，则逐个检查其元素
            for j, p in enumerate(parameters):
                # 检查元素是否为可迭代对象
                if not isinstance(p, container_abcs.Iterable):
                    raise TypeError(
                        "ParameterDict update sequence element "
                        "#" + str(j) + " should be Iterable; is" + type(p).__name__
                    )
                # 检查元素长度是否为2，否则引发值错误异常
                if not len(p) == 2:
                    raise ValueError(
                        "ParameterDict update sequence element "
                        "#" + str(j) + " has length " + str(len(p)) + "; 2 is required"
                    )
                # 更新 ParameterDict 中的键值对，使用可迭代对象的第一个和第二个元素作为键和值
                self[p[0]] = p[1]  # type: ignore[assignment]
    # 返回一个描述对象内容的字符串
    def extra_repr(self) -> str:
        # 初始化一个空列表，用于存储子对象的描述信息
        child_lines = []
        # 遍历 ParameterDict 对象的每个键值对
        for k, p in self.items():
            # 如果值是 torch.Tensor 类型
            if isinstance(p, torch.Tensor):
                # 生成 tensor 大小的字符串表示
                size_str = "x".join(str(size) for size in p.size())
                # 判断 tensor 的设备类型，生成设备信息字符串
                if p.device.type in ["cuda", torch._C._get_privateuse1_backend_name()]:
                    device_str = f" ({p.device})"
                else:
                    device_str = ""
                # 根据值的类型生成相应的描述字符串
                parastr = "{} containing: [{} of size {}{}]".format(
                    "Parameter" if isinstance(p, Parameter) else "Tensor",
                    torch.typename(p),
                    size_str,
                    device_str,
                )
                # 将生成的描述字符串添加到子对象描述列表中
                child_lines.append("  (" + str(k) + "): " + parastr)
            else:
                # 如果值不是 tensor 类型，则生成通用对象类型的描述字符串
                child_lines.append(
                    "  (" + str(k) + "): Object of type: " + type(p).__name__
                )
        # 将所有子对象的描述信息用换行符连接成一个完整的字符串
        tmpstr = "\n".join(child_lines)
        # 返回生成的描述信息字符串
        return tmpstr

    # 禁止对 ParameterDict 对象进行调用，抛出运行时错误
    def __call__(self, input):
        raise RuntimeError("ParameterDict should not be called.")

    # 实现两个 ParameterDict 对象的合并操作，返回合并后的新对象
    def __or__(self, other: "ParameterDict") -> "ParameterDict":
        # 复制当前对象
        copy = self.copy()
        # 将另一个对象的所有元素更新到复制的对象中
        copy.update(other)
        # 返回合并后的新对象
        return copy

    # 实现两个 ParameterDict 对象的合并操作（反向），返回合并后的新对象
    def __ror__(self, other: "ParameterDict") -> "ParameterDict":
        # 复制另一个对象
        copy = other.copy()
        # 将当前对象的所有元素更新到复制的对象中
        copy.update(self)
        # 返回合并后的新对象
        return copy

    # 实现就地更新（in-place update），将另一个 ParameterDict 对象的所有元素合并到当前对象中
    def __ior__(self, other: "ParameterDict") -> Self:
        # 将另一个对象的所有元素更新到当前对象中
        self.update(other)
        # 返回更新后的当前对象自身
        return self
```