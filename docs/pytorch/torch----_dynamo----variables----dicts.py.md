# `.\pytorch\torch\_dynamo\variables\dicts.py`

```
# 忽略 mypy 的错误信息，这通常用于在类型检查时排除特定的错误或警告
# 导入必要的模块和类
import collections  # 导入集合模块，用于操作集合数据类型
import dataclasses  # 导入 dataclasses 模块，用于创建和操作数据类
import functools    # 导入 functools 模块，用于高阶函数（Higher-order functions）操作
import inspect      # 导入 inspect 模块，用于检查对象
import sys          # 导入 sys 模块，提供对 Python 解释器的访问

from typing import Dict, List, Optional  # 从 typing 模块导入类型提示，用于类型注解

from torch._subclasses.fake_tensor import is_fake  # 导入 torch._subclasses.fake_tensor 模块中的 is_fake 函数

from .. import polyfill, variables  # 从相对包中导入 polyfill 和 variables 模块

from ..bytecode_transformation import create_call_function, create_instruction  # 从相对包中导入字节码转换相关的函数
from ..eval_frame import skip_code  # 从相对包中导入 skip_code 函数
from ..exc import unimplemented  # 从相对包中导入 unimplemented 异常
from ..guards import GuardBuilder, install_guard  # 从相对包中导入 GuardBuilder 和 install_guard
from ..source import AttrSource, GetItemSource  # 从相对包中导入 AttrSource 和 GetItemSource
from ..utils import dict_keys, dict_values, istype, specialize_symnode  # 从相对包中导入一些实用函数
from .base import MutableLocal, VariableTracker  # 从当前包的 base 模块导入 MutableLocal 和 VariableTracker
from .constant import ConstantVariable  # 从当前包的 constant 模块导入 ConstantVariable

# [Adding a new supported class within the keys of ConstDictVarialble]
# - Add its tracker type to is_hashable
# - (perhaps) Define how it is compared in _HashableTracker._eq_impl

# 定义函数 is_hashable，判断对象是否可哈希
def is_hashable(x):
    if isinstance(x, variables.TensorVariable):
        # 如果 x 是 TensorVariable 类型
        # 张量对象是可哈希的，如果它有一个示例值（即一个假张量）
        # 大多数 VT（VariableTracker）应该都有一个。
        # 在某些时候，我们可能可以断言它们都有一个
        return x.as_proxy().node.meta.get("example_value") is not None
    elif isinstance(x, variables.TupleVariable):
        # 如果 x 是 TupleVariable 类型
        # 则所有元素都是可哈希的，只要每个元素都是可哈希的
        return all(is_hashable(e) for e in x.items)
    else:
        # 对于其他类型的变量 x
        # 判断它是否属于以下可哈希的变量类型之一
        return isinstance(
            x,
            (
                variables.BuiltinVariable,
                variables.SymNodeVariable,
                variables.ConstantVariable,
                variables.EnumVariable,
                variables.user_defined.UserDefinedClassVariable,
                variables.UserFunctionVariable,
                variables.SkipFunctionVariable,
                variables.misc.NumpyVariable,
                variables.NNModuleVariable,
                variables.UnspecializedNNModuleVariable,
                variables.MethodWrapperVariable,
                variables.TorchInGraphFunctionVariable,
                variables.TypingVariable,
                variables.FunctoolsPartialVariable,
            ),
        )


# 定义 ConstDictVariable 类，继承自 VariableTracker 类
class ConstDictVariable(VariableTracker):
    # 定义 _nonvar_fields 集合，包含 "user_cls" 和 VariableTracker 类的 _nonvar_fields 中的所有元素
    _nonvar_fields = {
        "user_cls",
        *VariableTracker._nonvar_fields,
    }
    class _HashableTracker:
        """
        Auxiliary opaque internal class that wraps a VariableTracker and makes it hashable
        This should not be seen or touched by anything outside of ConstDictVariable and its children
        Note that it's also fine to put VTs into dictionaries and sets, but doing so does not take into account aliasing
        """

        def __init__(self, vt):
            # We specialize SymNodes
            # 使用 specialize_symnode 函数处理 vt，使其特化为 SymNode
            vt = specialize_symnode(vt)
            # TODO Temorarily remove to figure out what keys are we breaking on
            # and add proper support for them
            # 暂时移除以便确定我们正在破坏哪些键，并为其添加正确的支持
            if not is_hashable(vt):
                unimplemented(f"Dict key of type {type(vt)}. Key: {vt}")
            self.vt = vt

        @property
        def underlying_value(self):
            # 获取底层值
            if isinstance(self.vt, variables.TensorVariable):
                x = self.vt.as_proxy().node.meta["example_value"]
            elif isinstance(self.vt, variables.TupleVariable):
                Hashable = ConstDictVariable._HashableTracker
                x = tuple(Hashable(e).underlying_value for e in self.vt.items)
            elif isinstance(self.vt, variables.NNModuleVariable):
                return self.vt.module
            elif isinstance(self.vt, variables.UnspecializedNNModuleVariable):
                return self.vt.value
            elif isinstance(self.vt, variables.UserFunctionVariable):
                return self.vt.get_function()
            else:
                x = self.vt.as_python_constant()
            return x

        def __hash__(self):
            # 返回对象的哈希值
            return hash(self.underlying_value)

        @staticmethod
        def _eq_impl(a, b):
            # TODO: Put this in utils and share it between variables/builtin.py and here
            # 将此函数放入 utils 中，在 variables/builtin.py 和此处共享
            if type(a) != type(b):
                return False
            elif isinstance(a, tuple):
                Hashable = ConstDictVariable._HashableTracker
                return len(a) == len(b) and all(
                    Hashable._eq_impl(u, v) for u, v in zip(a, b)
                )
            elif is_fake(a):
                return a is b
            else:
                return a == b

        def __eq__(self, other: "ConstDictVariable._HashableTracker") -> bool:
            Hashable = ConstDictVariable._HashableTracker
            assert isinstance(other, Hashable) or ConstantVariable.is_literal(
                other
            ), type(other)
            if isinstance(other, Hashable):
                return Hashable._eq_impl(self.underlying_value, other.underlying_value)

            # constant
            return Hashable._eq_impl(self.underlying_value, other)

    def __init__(
        self, items: Dict[VariableTracker, VariableTracker], user_cls=dict, **kwargs
    ):
        # 构造函数，初始化 ConstDictVariable 实例
        ):
        # 调用父类的初始化方法，传入所有关键字参数
        super().__init__(**kwargs)

        # 获取常量字典变量的可哈希化追踪器类
        Hashable = ConstDictVariable._HashableTracker

        # 确保所有项都是变量追踪器或可哈希化追踪器的实例，并且值也是变量追踪器的实例
        assert all(
            isinstance(x, (VariableTracker, Hashable))
            and isinstance(v, VariableTracker)
            for x, v in items.items()
        )

        # 定义一个函数，用于将键转换为可哈希化类型
        def make_hashable(key):
            return key if isinstance(key, Hashable) else Hashable(key)

        # 使用字典推导式，将所有项转换为可哈希化类型的键，并保留原始值
        self.items = {make_hashable(x): v for x, v in items.items()}
        self.user_cls = user_cls

    # 返回一个字典，其中所有键和值均转换为代理对象
    def as_proxy(self):
        return {k.vt.as_proxy(): v.as_proxy() for k, v in self.items.items()}

    # 返回一个字符串，包含所有键值对的调试表示形式
    def debug_repr(self):
        return (
            "{"
            + ", ".join(
                f"{k.vt.debug_repr()}: {v.debug_repr()}" for k, v in self.items.items()
            )
            + "}"
        )

    # 返回一个字典，其中所有键和值均转换为 Python 中的常量表示形式
    def as_python_constant(self):
        return {
            k.vt.as_python_constant(): v.as_python_constant()
            for k, v in self.items.items()
        }

    # 返回一个字典，其中所有键转换为 Python 中的常量表示形式，而值保持不变
    def keys_as_python_constant(self):
        return {k.vt.as_python_constant(): v for k, v in self.items.items()}

    # 返回用户定义的类
    def python_type(self):
        return self.user_cls

    # 检查变量追踪器对象是否存在于字典中，并且其对应值不是已删除的变量
    def __contains__(self, vt):
        assert isinstance(vt, VariableTracker)
        Hashable = ConstDictVariable._HashableTracker
        return (
            is_hashable(vt)
            and Hashable(vt) in self.items
            and not isinstance(self.items[Hashable(vt)], variables.DeletedVariable)
        )

    # 返回字典中非删除变量的数量
    def len(self):
        return len(
            [
                x
                for x in self.items.values()
                if not isinstance(x, variables.DeletedVariable)
            ]
        )

    # 重构方法，用于生成代码
    def reconstruct(self, codegen):
        # 如果用户定义的类是 collections.OrderedDict，则加载相关模块
        if self.user_cls is collections.OrderedDict:
            codegen.add_push_null(
                lambda: codegen.extend_output(
                    [
                        codegen.create_load_python_module(collections),
                        codegen.create_load_attr("OrderedDict"),
                    ]
                )
            )
        
        # 生成字典的键和值的代码指令
        for key, value in self.items.items():
            codegen(key.vt)
            codegen(value)
        
        # 如果用户定义的类是 collections.OrderedDict，则生成 BUILD_MAP 指令，并调用相应函数
        if self.user_cls is collections.OrderedDict:
            codegen.extend_output(
                [
                    create_instruction("BUILD_MAP", arg=len(self.items)),
                    *create_call_function(1, False),
                ]
            )
        
        # 如果用户定义的类是 dict，则只生成 BUILD_MAP 指令
        else:
            codegen.append_output(create_instruction("BUILD_MAP", arg=len(self.items)))
    # 获取指定参数对应的项（常量版本）
    def getitem_const(self, arg: VariableTracker):
        # 使用参数生成可哈希的键
        key = ConstDictVariable._HashableTracker(arg)
        # 如果键不存在于字典中，则抛出未实现的错误，显示相关参数值
        if key not in self.items:
            unimplemented(f"dict KeyError: {arg.value}")
        # 返回对应键的项
        return self.items[key]

    # 获取指定参数对应的项（可能为空，常量版本）
    def maybe_getitem_const(self, arg: VariableTracker):
        # 使用参数生成可哈希的键
        key = ConstDictVariable._HashableTracker(arg)
        # 如果键不存在于字典中，返回 None
        if key not in self.items:
            return None
        # 返回对应键的项
        return self.items[key]

    # 调用方法
    def call_method(
        self,
        tx,
        name,
        args: "List[VariableTracker]",
        kwargs: "Dict[str, VariableTracker]",
    ):
        # 这里应该有方法的具体实现，但未提供

    # 解包变量序列
    def unpack_var_sequence(self, tx):
        # 返回字典中所有键对应的变量追踪对象组成的列表
        return [x.vt for x in self.items.keys()]
class DefaultDictVariable(ConstDictVariable):
    # 初始化方法，继承父类 ConstDictVariable 的初始化方法，并添加默认工厂参数
    def __init__(self, items, user_cls, default_factory=None, **kwargs):
        super().__init__(items, user_cls, **kwargs)
        # 断言 user_cls 必须是 collections.defaultdict
        assert user_cls is collections.defaultdict
        # 设置默认工厂属性
        self.default_factory = default_factory

    # 判断是否为 Python 常量
    def is_python_constant(self):
        # 如果默认工厂不是 list、tuple 或 dict，且 items 为空，则返回 False
        if self.default_factory not in [list, tuple, dict] and not self.items:
            return False
        # 否则调用父类方法判断是否为 Python 常量
        return super().is_python_constant()

    # 返回调试字符串表示形式
    def debug_repr(self):
        return (
            # 返回以 default_factory.debug_repr() 和父类的调试表示形式构成的字符串
            f"defaultdict({self.default_factory.debug_repr()}, {super().debug_repr()})"
        )

    # 判断参数是否为支持的参数类型
    @staticmethod
    def is_supported_arg(arg):
        if isinstance(arg, variables.BuiltinVariable):
            # 如果参数是 BuiltinVariable 类型，则判断其函数是否为 list、tuple 或 dict
            return arg.fn in [list, tuple, dict]
        else:
            # 否则判断参数是否为 BaseUserFunctionVariable 类型
            return isinstance(arg, variables.functions.BaseUserFunctionVariable)

    # 调用方法
    def call_method(
        self,
        tx,
        name,
        args: "List[VariableTracker]",
        kwargs: "Dict[str, VariableTracker]",
    ) -> "VariableTracker":
        if name == "__getitem__":
            # 如果调用的方法名为 '__getitem__'，则断言参数 args 的长度为 1
            assert len(args) == 1

            if args[0] in self:
                # 如果参数 args[0] 在当前对象中，则返回其对应的常量值
                return self.getitem_const(args[0])
            else:
                if self.default_factory is None:
                    # 如果没有默认工厂，则抛出 KeyError 异常
                    raise KeyError(f"{args[0]}")
                else:
                    # 否则调用默认工厂的 call_function 方法创建默认值
                    default_var = self.default_factory.call_function(tx, [], {})
                    # 调用父类的 call_method 方法设置默认值
                    super().call_method(
                        tx, "__setitem__", (args[0], default_var), kwargs
                    )
                    return default_var
        else:
            # 如果调用的方法名不是 '__getitem__'，则调用父类的 call_method 方法处理
            return super().call_method(tx, name, args, kwargs)


class SetVariable(ConstDictVariable):
    """We model a sets as dictonary with None values"""

    # 初始化方法，将 items 转换为字典形式，使用 None 作为默认值
    def __init__(
        self,
        items: List[VariableTracker],
        **kwargs,
    ):
        items = dict.fromkeys(items, SetVariable._default_value())
        super().__init__(items, **kwargs)

    # 返回调试字符串表示形式
    def debug_repr(self):
        if not self.items:
            return "set()"
        else:
            # 返回集合的调试表示形式
            return "{" + ",".join(k.vt.debug_repr() for k in self.items.keys()) + "}"

    # 返回集合的项
    @property
    def set_items(self):
        return set(self.items.keys())

    # 返回默认值，填充字典的键
    @staticmethod
    def _default_value():
        return ConstantVariable.create(None)

    # 返回代理对象
    def as_proxy(self):
        return {k.vt.as_proxy() for k in self.set_items}

    # 返回 Python 类型
    def python_type(self):
        return set

    # 返回作为 Python 常量的集合
    def as_python_constant(self):
        return {k.vt.as_python_constant() for k in self.set_items}

    # 重构方法
    def reconstruct(self, codegen):
        # 生成循环代码，遍历集合的每个项
        codegen.foreach([x.vt for x in self.set_items])
        # 添加构建集合的指令
        codegen.append_output(create_instruction("BUILD_SET", arg=len(self.set_items)))
    # 定义一个实例方法，用于调用特定对象的方法
    def call_method(
        self,
        tx,  # 事务对象，用于处理事务相关操作
        name,  # 方法名，表示要调用的方法名称
        args: List[VariableTracker],  # 参数列表，包含要传递给方法的位置参数
        kwargs: Dict[str, VariableTracker],  # 关键字参数字典，包含要传递给方法的关键字参数
    ) -> "VariableTracker":  # 返回类型为 VariableTracker 对象

        from . import ListVariable, TupleVariable  # 导入必要的模块

        # 将调用转发给字典模型
        if name == "add":  # 如果调用的方法是 "add"
            assert not kwargs  # 确保没有关键字参数
            assert len(args) == 1  # 确保只有一个位置参数
            name = "__setitem__"  # 将方法名修改为 "__setitem__"
            args = (args[0], SetVariable._default_value())  # 修改参数为第一个参数和默认值的元组
        elif name == "pop":  # 如果调用的方法是 "pop"
            assert not kwargs  # 确保没有关键字参数
            assert not args  # 确保没有位置参数
            # 随机选择一个项，并通过 Dict.pop 方法弹出它
            result = self.set_items.pop().vt  # 从集合中弹出一个项，并获取其值
            super().call_method(tx, name, (result,), kwargs)  # 调用父类的 call_method 方法
            return result  # 返回弹出的项
        elif name == "isdisjoint":  # 如果调用的方法是 "isdisjoint"
            assert not kwargs  # 确保没有关键字参数
            assert len(args) == 1  # 确保只有一个位置参数
            return variables.UserFunctionVariable(  # 返回一个用户自定义函数变量的调用结果
                polyfill.set_isdisjoint  # 调用 polyfill 模块中的 set_isdisjoint 函数
            ).call_function(tx, [self, args[0]], {})  # 调用该函数并传递事务对象和参数
        elif (  # 如果条件如下：
            name == "update"  # 方法名是 "update"
            and len(args) == 1  # 只有一个位置参数
            and isinstance(  # 并且该参数是以下类型之一：
                args[0],
                (
                    SetVariable,  # SetVariable 类型
                    ListVariable,  # ListVariable 类型
                    TupleVariable,  # TupleVariable 类型
                ),
            )
            and self.mutable_local  # 并且 self.mutable_local 为真
        ):
            if isinstance(args[0], (ListVariable, TupleVariable)):  # 如果参数是 ListVariable 或 TupleVariable 类型
                arg = SetVariable(args[0].unpack_var_sequence(tx))  # 将参数解包成 SetVariable 对象
            else:
                arg = args[0]  # 否则直接使用参数
            return super().call_method(tx, "update", (arg,), kwargs)  # 调用父类的 update 方法
        return super().call_method(tx, name, args, kwargs)  # 默认情况下调用父类的 call_method 方法

    # 定义一个方法，用于处理获取常量的操作，抛出异常
    def getitem_const(self, arg: VariableTracker):
        raise RuntimeError("Illegal to getitem on a set")  # 抛出运行时异常，不允许在集合上执行获取操作
class DictView(VariableTracker):
    """
    Models _PyDictViewObject
    
    This is an "abstract" class. Subclasses will override kv and the items method
    """

    kv: Optional[str] = None  # Optional key type ('keys' or 'values')

    def __init__(self, dv_dict: ConstDictVariable, **kwargs):
        super().__init__(**kwargs)
        assert self.kv in ("keys", "values")  # Assert kv is either 'keys' or 'values'
        assert isinstance(dv_dict, ConstDictVariable)
        self.dv_dict = dv_dict  # Initialize dv_dict as a ConstDictVariable instance

    @property
    def view_items(self):
        return getattr(self.dv_dict.items, self.kv)()  # Dynamically fetch items as keys() or values()

    @property
    def view_items_vt(self):
        # Returns an iterable of the unpacked items
        raise NotImplementedError  # To be implemented by subclasses

    def unpack_var_sequence(self, tx):
        def unwrap(x):
            return x.vt if self.kv == "keys" else x  # Unwrap as 'vt' if kv is 'keys', else return x

        return [unwrap(x) for x in self.view_items]  # Unwrap all items in view_items

    def reconstruct(self, codegen):
        codegen(self.dv_dict)  # Generate code for dv_dict
        codegen.load_method(self.kv)  # Load method specified by kv ('keys' or 'values')
        codegen.call_method(0)  # Call the loaded method

    def call_method(
        self,
        tx,
        name,
        args: List["VariableTracker"],
        kwargs: Dict[str, "VariableTracker"],
    ) -> "VariableTracker":
        if name == "__len__":
            return self.dv_dict.call_method(tx, name, args, kwargs)  # Call __len__ method of dv_dict
        return super().call_method(tx, name, args, kwargs)  # Call superclass's call_method


class DictKeys(DictView):
    kv = "keys"  # Set kv to 'keys'

    @property
    def set_items(self):
        return set(self.view_items)  # Return set of view_items

    @property
    def view_items_vt(self):
        # Returns an iterable of the unpacked items
        return [x.vt for x in self.view_items]  # Unwrap items in view_items to 'vt'

    def python_type(self):
        return dict_keys  # Return type dict_keys

    def call_method(
        self,
        tx,
        name,
        args: List["VariableTracker"],
        kwargs: Dict[str, "VariableTracker"],
    ) -> "VariableTracker":
        if name == "__contains__":
            return self.dv_dict.call_method(tx, name, args, kwargs)  # Call __contains__ method of dv_dict
        return super().call_method(tx, name, args, kwargs)  # Call superclass's call_method


class DictValues(DictView):
    # DictValues is an iterable but cannot be compared.
    kv = "values"  # Set kv to 'values'

    @property
    def view_items_vt(self):
        return list(self.view_items)  # Return list of view_items

    def python_type(self):
        return dict_values  # Return type dict_values


def _is_matching_transformers_cls(cls) -> bool:
    mod = sys.modules.get("transformers.file_utils")  # Get module 'transformers.file_utils'
    if mod is None:
        mod = sys.modules.get("transformers.utils.generic")  # Get module 'transformers.utils.generic' if first is None
    return mod is not None and issubclass(cls, mod.ModelOutput)  # Check if cls is a subclass of mod.ModelOutput


def _is_matching_diffusers_cls(cls) -> bool:
    mod = sys.modules.get("diffusers.utils")  # Get module 'diffusers.utils'
    return mod is not None and issubclass(cls, mod.BaseOutput)  # Check if cls is a subclass of mod.BaseOutput


def _call_hasattr_customobj(self, tx, name: str) -> "VariableTracker":
    """Shared method between DataClassVariable and CustomizedDictVariable where items are attrs"""
    # 检查事务输出是否对自身属性进行了修改
    if tx.output.side_effects.is_attribute_mutation(self):
        try:
            # 尝试从事务输出中加载属性，并返回是否已删除的常量变量
            result = tx.output.side_effects.load_attr(self, name, deleted_ok=True)
            return variables.ConstantVariable.create(
                not isinstance(result, variables.DeletedVariable)
            )
        except KeyError:
            pass
    # 检查属性名是否存在于对象的项中或者用户类中是否存在该属性
    if name in self.items or hasattr(self.user_cls, name):
        return ConstantVariable(True)
    elif istype(self.mutable_local, MutableLocal) and self.source is None:
        # 如果可变局部对象存在且没有来源，则返回False常量变量
        return ConstantVariable(False)
    elif self.source:
        # 如果存在来源，则尝试获取来源对象的示例值，并添加一个属性存在的保护
        try:
            example = tx.output.root_tx.get_example_value(self.source)
            install_guard(
                AttrSource(self.source, name).make_guard(GuardBuilder.HASATTR)
            )
            return ConstantVariable(hasattr(example, name))
        except KeyError:
            pass
    # 报告未实现的功能，显示相应的类名、属性名、可变局部对象和来源
    unimplemented(
        f"hasattr({self.__class__.__name__}, {name}) {self.mutable_local} {self.source}"
    )
# 定义一个名为 DataClassVariable 的类，继承自 ConstDictVariable
class DataClassVariable(ConstDictVariable):
    """
    This class doesn't appear to be used anywhere.
    It used to be used to deal with transformers.file_utils.ModelOutput
    from huggingface.

    Keeping since we wish to support dataclasses in general in the future
    """

    # 该类目前未被任何地方使用。曾经用于处理来自 huggingface 的 transformers.file_utils.ModelOutput。
    # 保留该类是因为我们希望在将来支持数据类。
    pass


class CustomizedDictVariable(ConstDictVariable):
    # 静态方法，用于检查是否与 Huggingface 的类匹配
    @staticmethod
    def is_matching_cls_hf(cls):
        return _is_matching_transformers_cls(cls) or _is_matching_diffusers_cls(cls)

    # 静态方法，用于检查是否与给定类匹配
    @staticmethod
    def is_matching_cls(cls):
        # 如果是 collections.OrderedDict 的子类，并且未实现 __post_init__ 方法
        if (
            issubclass(cls, collections.OrderedDict)
            and cls is not collections.OrderedDict
            and cls.__init__ is collections.OrderedDict.__init__
            and not hasattr(cls, "__post_init__")
        ):
            return True
        # 对于 Huggingface 的用例的 hack：
        #   假设 ModelOutput 的子类使用数据类注释
        #   假设 self.create 是 AA 到 ModelOutput.__post_init__
        return CustomizedDictVariable.is_matching_cls_hf(cls)

    @classmethod
    def is_matching_object(cls, obj):
        # 检查给定对象是否与当前类匹配
        return cls.is_matching_cls(type(obj))

    # 被 user_defined.py 调用，
    # 当 is_matching_cls(cls) 返回 True 时
    @classmethod
    # 创建一个类方法 `create`，用于根据给定的参数创建对象
    def create(cls, user_cls, args, kwargs, options):
        # 避免在从 forward 函数返回 ModelOutput 时进行跟踪
        for attr_name in ("__init__", "__post_init__", "__setattr__", "__setitem__"):
            # 检查用户定义的类是否具有指定的特定属性名
            if hasattr(user_cls, attr_name):
                fn = getattr(user_cls, attr_name)
                # 断言该属性名对应的对象是可调用的函数
                assert callable(fn), f"expect callable attr {attr_name}"
                # 如果函数对象具有 `__code__` 属性，则调用 skip_code 函数
                if hasattr(fn, "__code__"):
                    skip_code(fn.__code__)

        # 如果用户定义的类是 dataclass
        if dataclasses.is_dataclass(user_cls):
            # 绑定用户类的初始化方法的签名和参数
            bound = inspect.signature(user_cls).bind(*args, **kwargs)
            bound.apply_defaults()

            # 定义一个用于处理变量的函数 make_var
            def make_var(x):
                if isinstance(x, VariableTracker):
                    return x
                elif ConstantVariable.is_literal(x):
                    return ConstantVariable.create(x)
                else:
                    unimplemented(
                        "expect VariableTracker or ConstantVariable.is_literal"
                    )

            bound_args = {}
            # 如果类是匹配 cls.is_matching_cls_hf 函数返回 True 的类
            if cls.is_matching_cls_hf(user_cls):
                # 跳过值为 None 的常量变量
                for k, v in bound.arguments.items():
                    if isinstance(v, ConstantVariable) and v.value is None or v is None:
                        continue
                    bound_args[k] = v
            else:
                bound_args = bound.arguments

            # 根据 bound_args 构建 items 字典
            items = {
                ConstantVariable.create(k): make_var(v) for k, v in bound_args.items()
            }
        elif not args:
            # 在一般情况下（非 dataclass 的情况下），使用 kwargs 构建 items 字典
            items = {ConstantVariable.create(k): v for k, v in kwargs.items()}
        elif len(args) == 1 and isinstance(args[0], ConstDictVariable) and not kwargs:
            # 使用 ConstDictVariable 对象的 items 属性构建 items 字典
            items = args[0].items
        else:
            # 当使用 args 或 kwargs 初始化自定义字典时，抛出未实现异常
            unimplemented("custom dict init with args/kwargs unimplemented")

        # 使用 items 和 user_cls 构建一个新的对象并返回
        return cls(items, user_cls, **options)

    # 在 builder.py 中调用的类方法 `wrap`
    @classmethod
    def wrap(cls, builder, obj):
        # 获取 obj 对象的类类型
        user_cls = type(obj)

        # 如果不是匹配 cls.is_matching_cls_hf 函数返回 True 的类，抛出未实现异常
        if not cls.is_matching_cls_hf(user_cls):
            unimplemented("custom non-hf dict subclass wrap unimplemented")

        # 使用 builder 构建 items 字典，并保持 OrderedDict 的顺序
        items = builder.__class__(tx=builder.tx, source=builder.source)(
            collections.OrderedDict(obj)
        ).items

        # 获取 dataclass 中定义的字段名列表
        keys = [f.name for f in dataclasses.fields(user_cls)]
        for key in keys:
            # 如果 obj 对象具有 key 字段
            if hasattr(obj, key):
                val = getattr(obj, key)
                # 使用 builder 构建 val 对象
                var = builder.__class__(
                    tx=builder.tx, source=AttrSource(builder.source, key)
                )(val)
                # 如果 val 不为 None，则创建常量变量 key 并将 var 添加到 items 字典中
                if val is not None:
                    key = ConstantVariable.create(key)
                    items[key] = var
        # 使用 items 和 user_cls 构建一个新的对象并返回
        return cls(items, user_cls)
    def __init__(self, items, user_cls, **options):
        super().__init__(items, user_cls, **options)
        assert self.is_matching_cls(user_cls)


        # 调用父类的初始化方法，传递参数 items, user_cls 和 options
        super().__init__(items, user_cls, **options)
        # 断言当前实例的 user_cls 是否与预期匹配
        assert self.is_matching_cls(user_cls)



    def as_proxy(self):
        raise NotImplementedError


        # 抛出未实现错误，表示该方法需要在子类中被实现
        raise NotImplementedError



    # 'RETURN_VALUE triggered compile'
    # called from torch/_dynamo/codegen.py
    def reconstruct(self, codegen):
        is_hf_model_output = self.is_matching_cls_hf(self.user_cls)

        def gen_fn1():
            # If the user class is a ModelOutput, then wrap the instance creation in
            # torch._dynamo.disable(). Even though we mark the __post_init__ as skip
            # in `create` function, this is not enough. TorchDynamo can still get
            # triggered on the child functions of __post_init__. This upsets export.
            # Since, we know that ModelOutput __post_init__ is not worth optimizing,
            # we just wrap the instance creation in torch._dynamo.disable(),
            # regardless whether its export or not.
            if is_hf_model_output:
                # load torch._dynamo.disable
                def gen_fn2():
                    codegen.append_output(codegen.create_load_global("torch", add=True))
                    codegen.append_output(codegen.create_load_attr("_dynamo"))
                    codegen.append_output(codegen.create_load_attr("disable"))

                codegen.add_push_null(gen_fn2)

            codegen.extend_output([codegen._create_load_const(self.user_cls)])

            if is_hf_model_output:
                # Wrap user_cls with disable
                codegen.extend_output(create_call_function(1, False))

        codegen.add_push_null(gen_fn1)

        # All the keys are just wrapped strings
        d = self.keys_as_python_constant()
        codegen.foreach(d.values())
        keys = tuple(d.keys())
        codegen.extend_output(codegen.create_call_function_kw(len(keys), keys, False))


        # 根据传入的 codegen 生成重建代码
        is_hf_model_output = self.is_matching_cls_hf(self.user_cls)

        def gen_fn1():
            # 如果 user_cls 是 ModelOutput，则将实例创建包装在 torch._dynamo.disable() 中
            if is_hf_model_output:
                # 加载 torch._dynamo.disable
                def gen_fn2():
                    codegen.append_output(codegen.create_load_global("torch", add=True))
                    codegen.append_output(codegen.create_load_attr("_dynamo"))
                    codegen.append_output(codegen.create_load_attr("disable"))

                codegen.add_push_null(gen_fn2)

            # 将 self.user_cls 加载为常量
            codegen.extend_output([codegen._create_load_const(self.user_cls)])

            if is_hf_model_output:
                # 使用 disable 包装 user_cls
                codegen.extend_output(create_call_function(1, False))

        # 将 gen_fn1 添加到生成的代码中
        codegen.add_push_null(gen_fn1)

        # 获取所有键作为 Python 常量
        d = self.keys_as_python_constant()
        # 对 d.values() 执行 foreach 操作
        codegen.foreach(d.values())
        keys = tuple(d.keys())
        # 创建带关键字参数的函数调用
        codegen.extend_output(codegen.create_call_function_kw(len(keys), keys, False))



    def call_method(
        self,
        tx,
        name,
        args: "List[VariableTracker]",
        kwargs: "Dict[str, VariableTracker]",


        # 定义一个方法 call_method，接受参数 tx, name, args 和 kwargs
    ) -> "VariableTracker":
        # 获取用户定义类 self.user_cls 中名为 name 的属性或方法对象 fn
        fn = getattr(self.user_cls, name)
        # 如果存在源信息 self.source，则创建一个 AttrSource 对象用于跟踪属性来源
        source = None if self.source is None else AttrSource(self.source, name)

        # 如果 fn 是 dict 或 collections.OrderedDict，并且没有被重写，则调用父类的方法处理
        if hasattr(fn, "__objclass__") and fn.__objclass__ in (
            dict,
            collections.OrderedDict,
        ):
            # 对于没有被重写的 Python 字典方法
            return super().call_method(tx, name, args, kwargs)
        # 如果 name 是用户重写的方法名之一，则内联调用用户函数返回
        elif name in (
            "__getitem__",
            "to_tuple",
            "__setitem__",
            "__setattr__",
            "__post_init__",
        ):
            return tx.inline_user_function_return(
                variables.UserFunctionVariable(fn, source=source),
                [self] + list(args),
                kwargs,
            )
        # 如果 fn 是 collections.OrderedDict 的方法，则调用父类方法处理
        elif fn is getattr(collections.OrderedDict, name, None):
            return super().call_method(tx, name, args, kwargs)

        # 报告未实现的错误，表示自定义字典方法调用的处理未实现
        unimplemented(f"custom dict: call_method unimplemented name={name}")

    def var_getattr(self, tx, name: str) -> "VariableTracker":
        # 创建名为 name 的常量变量
        name_vt = ConstantVariable.create(name)
        # 如果 self 中存在名为 name 的属性或方法，则调用 self 的 call_method 方法获取其值
        if name_vt in self:
            return self.call_method(tx, "__getitem__", [name_vt], {})
        # 如果 self.user_cls 是数据类，则获取其字段的默认值，并检查 name 是否在默认值中
        if dataclasses.is_dataclass(self.user_cls):
            defaults = {f.name: f.default for f in dataclasses.fields(self.user_cls)}
            if name in defaults:
                # 断言默认值为字面值常量，并创建对应的常量变量返回
                assert variables.ConstantVariable.is_literal(defaults[name])
                return variables.ConstantVariable.create(defaults[name])
        # 否则调用父类的 var_getattr 方法处理
        return super().var_getattr(tx, name)

    # 将 _call_hasattr_customobj 方法指定为 call_hasattr 方法的实现
    call_hasattr = _call_hasattr_customobj
@functools.lru_cache(None)
# 使用 functools 模块中的 lru_cache 装饰器，将函数结果缓存，None 表示缓存大小无限制
def _install_PretrainedConfig_patch():
    import transformers

    # 在这里我们需要对 transformers 进行 monkeypatch（猴子补丁）操作。
    # TODO(voz): Upstream to transformers lib
    # 将自定义的 __eq__ 方法赋值给 transformers 库中 PretrainedConfig 类的 __eq__ 方法
    def _dynamo_overriden_transformers_eq(self, other):
        if not hasattr(other, "__dict__"):
            return False
        return self.__dict__ == other.__dict__

    transformers.configuration_utils.PretrainedConfig.__eq__ = (
        _dynamo_overriden_transformers_eq
    )


class HFPretrainedConfigVariable(VariableTracker):
    """
    Hack for HuggingFace PretrainedConfig
    """

    @staticmethod
    def is_matching_cls(cls):
        mod = sys.modules.get("transformers.configuration_utils")
        # 检查给定的类是否是 transformers.configuration_utils.PretrainedConfig 的子类
        is_match = mod is not None and issubclass(cls, mod.PretrainedConfig)

        # 当第一次在 dynamo 中看到匹配时，懒惰地安装 monkeypatch
        if is_match:
            _install_PretrainedConfig_patch()
        return is_match

    @classmethod
    def is_matching_object(cls, obj):
        # 检查给定对象是否属于 transformers.configuration_utils.PretrainedConfig 的子类
        return cls.is_matching_cls(type(obj))

    def __init__(self, obj, **kwargs):
        super().__init__(**kwargs)
        self.obj = obj
        assert self.is_matching_cls(type(obj))

    def var_getattr(self, tx, name: str) -> "VariableTracker":
        from . import ConstantVariable

        # 返回指定名称在对象上的属性的常量变量
        return ConstantVariable.create(getattr(self.obj, name))

    def call_hasattr(self, tx, name: str) -> "VariableTracker":
        # 返回对象是否具有指定名称的属性的常量变量
        return variables.ConstantVariable.create(hasattr(self.obj, name))


class PythonSysModulesVariable(VariableTracker):
    """Special case for sys.modules.

    Without this we will guard on the exact set of modules imported in the
    lifetime of the python program.
    """

    def python_type(self):
        # 返回该变量的 Python 类型为字典
        return dict

    def reconstruct(self, codegen):
        # 重建该变量的代码生成器
        codegen.add_push_null(
            lambda: codegen.extend_output(
                [
                    codegen.create_load_python_module(sys),
                    codegen.create_load_attr("modules"),
                ]
            )
        )

    def call_method(
        self, tx, name, args: List[VariableTracker], kwargs: Dict[str, VariableTracker]
    ):
        # 根据方法名调用 sys.modules 的方法
        if name == "__getitem__":
            return self.call_getitem(tx, *args, **kwargs)
        elif name == "get":
            return self.call_get(tx, *args, **kwargs)
        elif name == "__contains__":
            return self.call_contains(tx, *args, **kwargs)
        unimplemented(f"sys.modules.{name}(*{args}, **{kwargs})")

    def _contains_helper(self, tx, key: VariableTracker):
        # 辅助方法：检查指定键是否存在于 sys.modules 中
        k = key.as_python_constant()
        has_key = k in sys.modules
        install_guard(
            self.make_guard(
                functools.partial(GuardBuilder.DICT_CONTAINS, key=k, invert=not has_key)
            )
        )
        return k, has_key

    def call_contains(self, tx, key: VariableTracker):
        # 调用 sys.modules.__contains__ 方法检查指定键是否存在于 sys.modules 中
        k, has_key = self._contains_helper(tx, key)
        return ConstantVariable.create(value=has_key)
    # 定义一个方法 `call_get`，用于从事务 `tx` 中获取变量 `key` 对应的值
    # `default` 参数可选，表示默认返回值
    def call_get(
        self, tx, key: VariableTracker, default: Optional[VariableTracker] = None
    ):
        # 导入变量构建器 `VariableBuilder`，用于构建变量对象
        from .builder import VariableBuilder

        # 使用 `_contains_helper` 方法检查 `key` 是否存在于事务 `tx` 中，并获取对应的 `k` 和 `has_key` 值
        k, has_key = self._contains_helper(tx, key)

        # 如果 `key` 存在于事务中，则返回其对应的变量构建器
        if has_key:
            return VariableBuilder(
                tx,
                GetItemSource(self.source, k),  # 获取 `self.source` 中 `k` 对应的项目来源
            )(sys.modules[k])  # 调用变量构建器，传入 `sys.modules[k]` 作为参数

        # 如果 `default` 参数不为 `None`，则返回 `default` 对象
        if default is not None:
            return default

        # 否则，返回一个空值的常量变量对象
        return ConstantVariable.create(value=None)

    # 定义一个方法 `call_getitem`，用于从事务 `tx` 中获取变量 `key` 对应的值
    def call_getitem(self, tx, key: VariableTracker):
        # 导入变量构建器 `VariableBuilder`，用于构建变量对象
        from .builder import VariableBuilder

        # 使用 `_contains_helper` 方法检查 `key` 是否存在于事务 `tx` 中，并获取对应的 `k` 和 `has_key` 值
        k, has_key = self._contains_helper(tx, key)

        # 返回变量构建器，传入 `GetItemSource(self.source, k)` 作为参数
        return VariableBuilder(
            tx,
            GetItemSource(self.source, k),  # 获取 `self.source` 中 `k` 对应的项目来源
        )(sys.modules[k])  # 调用变量构建器，传入 `sys.modules[k]` 作为参数
```