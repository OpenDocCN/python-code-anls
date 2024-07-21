# `.\pytorch\torch\utils\data\datapipes\_typing.py`

```
# mypy: allow-untyped-defs
# Taking reference from official Python typing
# https://github.com/python/cpython/blob/master/Lib/typing.py

# 导入必要的模块
import collections
import functools
import numbers
import sys

# Please check [Note: TypeMeta and TypeAlias]
# In case of metaclass conflict due to ABCMeta or _ProtocolMeta
# For Python 3.9, only Protocol in typing uses metaclass
# 导入 ABCMeta 类型定义
from abc import ABCMeta
# 导入 typing 模块中的相关类型，忽略属性定义
from typing import (
    _eval_type,
    _GenericAlias,  # TODO: Use TypeAlias when Python 3.6 is deprecated
    _tp_cache,
    _type_check,
    _type_repr,
    Any,
    Dict,
    ForwardRef,
    Generic,
    get_type_hints,
    Iterator,
    List,
    Set,
    Tuple,
    TypeVar,
    Union,
)

# 导入 Torch 相关模块
from torch.utils.data.datapipes._hook_iterator import _SnapshotState, hook_iterator


# 定义 GenericMeta 类，忽略 no-redef 类型检查
class GenericMeta(ABCMeta):  # type: ignore[no-redef]
    pass


# 定义 Integer 类，继承自 numbers.Integral
class Integer(numbers.Integral):
    pass


# 定义 Boolean 类，继承自 numbers.Integral
class Boolean(numbers.Integral):
    pass


# Python 'type' object is not subscriptable
# Tuple[int, List, dict] -> valid
# tuple[int, list, dict] -> invalid
# Map Python 'type' to abstract base class
# 定义字典 TYPE2ABC，将 Python 类型映射到对应的抽象基类
TYPE2ABC = {
    bool: Boolean,
    int: Integer,
    float: numbers.Real,
    complex: numbers.Complex,
    dict: Dict,
    list: List,
    set: Set,
    tuple: Tuple,
    None: type(None),
}


# 定义函数 issubtype，检查左侧类型是否是右侧类型的子类型
def issubtype(left, right, recursive=True):
    r"""
    Check if the left-side type is a subtype of the right-side type.

    If any of type is a composite type like `Union` and `TypeVar` with
    bounds, it would be expanded into a list of types and check all
    of left-side types are subtypes of either one from right-side types.
    """
    # 获取左侧类型的抽象基类，如果不存在映射则保持原样
    left = TYPE2ABC.get(left, left)
    # 获取右侧类型的抽象基类，如果不存在映射则保持原样
    right = TYPE2ABC.get(right, right)

    # 如果右侧类型是 Any 或者左右相等，则返回 True
    if right is Any or left == right:
        return True

    # 如果右侧类型是泛型别名，且具有 Generic 的属性，则返回 True
    if isinstance(right, _GenericAlias):
        if getattr(right, "__origin__", None) is Generic:
            return True

    # 如果右侧类型是 NoneType，则返回 False
    if right == type(None):
        return False

    # 解析右侧类型的约束条件
    constraints = _decompose_type(right)

    # 如果约束条件为空或包含 Any，则返回 True
    if len(constraints) == 0 or Any in constraints:
        return True

    # 如果左侧类型是 Any，则返回 False
    if left is Any:
        return False

    # 解析左侧类型的变体
    variants = _decompose_type(left)

    # 如果变体列表为空，则返回 False
    if len(variants) == 0:
        return False

    # 检查所有左侧变体是否都是右侧约束条件的子类型
    return all(
        _issubtype_with_constraints(variant, constraints, recursive)
        for variant in variants
    )


# 函数 _decompose_type，解析类型 t 并返回类型列表
def _decompose_type(t, to_list=True):
    if isinstance(t, TypeVar):
        # 如果 t 是 TypeVar 类型，则获取其绑定的约束条件
        if t.__bound__ is not None:
            ts = [t.__bound__]
        else:
            # 对于 T_co，__constraints__ 是空的情况
            ts = list(t.__constraints__)
    elif hasattr(t, "__origin__") and t.__origin__ == Union:
        # 如果 t 有 __origin__ 属性并且是 Union 类型，则获取其参数列表
        ts = t.__args__
    else:
        # 否则将 t 转换为单元素列表
        if not to_list:
            return None
        ts = [t]
    # 忽略 _t 不在 TYPE2ABC 中的情况，将 ts 中的类型映射为其抽象基类
    ts = [TYPE2ABC.get(_t, _t) for _t in ts]  # type: ignore[misc]
    return ts
def _issubtype_with_constraints(variant, constraints, recursive=True):
    r"""
    Check if the variant is a subtype of either one from constraints.

    For composite types like `Union` and `TypeVar` with bounds, they
    would be expanded for testing.
    """
    # 如果 variant 在 constraints 中，则直接返回 True
    if variant in constraints:
        return True

    # 处理复合类型，如 Union 和 TypeVar，需要展开进行检查

    # 处理 variant 是 TypeVar 或 Union 的情况
    vs = _decompose_type(variant, to_list=False)
    if vs is not None:
        # 对于 variant 中的每个子类型，递归调用 _issubtype_with_constraints
        return all(_issubtype_with_constraints(v, constraints, recursive) for v in vs)

    # 处理 variant 不是 TypeVar 或 Union 的情况
    if hasattr(variant, "__origin__") and variant.__origin__ is not None:
        v_origin = variant.__origin__
        v_args = getattr(variant, "__args__", None)
    else:
        v_origin = variant
        v_args = None

    # 处理 constraints 中的每个约束
    for constraint in constraints:
        cs = _decompose_type(constraint, to_list=False)

        # 处理 constraint 是 TypeVar 或 Union 的情况
        if cs is not None:
            # 如果 variant 是 cs 的子类型，则返回 True
            if _issubtype_with_constraints(variant, cs, recursive):
                return True
        # 处理 constraint 不是 TypeVar 或 Union 的情况
        else:
            if hasattr(constraint, "__origin__") and constraint.__origin__ is not None:
                c_origin = constraint.__origin__
                # 处理 v_origin 和 c_origin 相同的情况
                if v_origin == c_origin:
                    if not recursive:
                        return True
                    c_args = getattr(constraint, "__args__", None)
                    # 处理参数匹配的情况
                    if c_args is None or len(c_args) == 0:
                        return True
                    if (
                        v_args is not None
                        and len(v_args) == len(c_args)
                        and all(
                            issubtype(v_arg, c_arg)
                            for v_arg, c_arg in zip(v_args, c_args)
                        )
                    ):
                        return True
            else:
                # 处理简单类型直接匹配的情况，如 Tuple[int] -> Tuple
                if v_origin == constraint:
                    return True

    # 如果所有检查都不符合，则返回 False
    return False
    # 检查数据类型是否符合指定的类型（data_type），如果不符合则返回 False
    if not issubtype(type(data), data_type, recursive=False):
        return False

    # 获取数据类型（data_type）的 __args__ 属性，该属性在 Python 3.9 中对于未指定泛型的类型不定义
    dt_args = getattr(data_type, "__args__", None)
    
    # 如果数据是一个元组
    if isinstance(data, tuple):
        # 如果 dt_args 未定义或长度为 0，表示元组可以是任何类型，返回 True
        if dt_args is None or len(dt_args) == 0:
            return True
        # 检查元组的长度是否与 dt_args 的长度相同，如果不同则返回 False
        if len(dt_args) != len(data):
            return False
        # 逐个检查元组中每个元素是否符合对应的类型
        return all(issubinstance(d, t) for d, t in zip(data, dt_args))
    
    # 如果数据是列表或集合
    elif isinstance(data, (list, set)):
        # 如果 dt_args 未定义或长度为 0，表示列表或集合可以是任何类型，返回 True
        if dt_args is None or len(dt_args) == 0:
            return True
        # 获取列表或集合中元素的类型
        t = dt_args[0]
        # 逐个检查列表或集合中每个元素是否符合指定的类型 t
        return all(issubinstance(d, t) for d in data)
    
    # 如果数据是字典
    elif isinstance(data, dict):
        # 如果 dt_args 未定义或长度为 0，表示字典可以是任何键值对类型，返回 True
        if dt_args is None or len(dt_args) == 0:
            return True
        # 获取字典键和值的类型
        kt, vt = dt_args
        # 逐个检查字典中每个键值对的键和值是否符合指定的类型 kt 和 vt
        return all(
            issubinstance(k, kt) and issubinstance(v, vt) for k, v in data.items()
        )

    # 如果数据类型不是元组、列表、集合或字典，则默认返回 True
    return True
# [Note: TypeMeta and TypeAlias]
# In order to keep compatibility for Python 3.6, use Meta for the typing.
# TODO: When PyTorch drops the support for Python 3.6, it can be converted
# into the Alias system and using `__class_getitem__` for DataPipe. The
# typing system will gain benefit of performance and resolving metaclass
# conflicts as elaborated in https://www.python.org/dev/peps/pep-0560/

class _DataPipeType:
    r"""Save type annotation in `param`."""

    def __init__(self, param):
        self.param = param  # Store the type annotation parameter

    def __repr__(self):
        return _type_repr(self.param)  # Return string representation of the type

    def __eq__(self, other):
        if isinstance(other, _DataPipeType):
            return self.param == other.param
        return NotImplemented

    def __hash__(self):
        return hash(self.param)  # Return hash value based on the type parameter

    def issubtype(self, other):
        if isinstance(other.param, _GenericAlias):
            if getattr(other.param, "__origin__", None) is Generic:
                return True
        if isinstance(other, _DataPipeType):
            return issubtype(self.param, other.param)
        if isinstance(other, type):
            return issubtype(self.param, other)
        raise TypeError(f"Expected '_DataPipeType' or 'type', but found {type(other)}")

    def issubtype_of_instance(self, other):
        return issubinstance(other, self.param)  # Check if `other` is a subtype of `self.param`

# Default type for DataPipe without annotation
T_co = TypeVar("T_co", covariant=True)
_DEFAULT_TYPE = _DataPipeType(Generic[T_co])


class _DataPipeMeta(GenericMeta):
    r"""
    Metaclass for `DataPipe`.

    Add `type` attribute and `__init_subclass__` based on the type, and validate the return hint of `__iter__`.

    Note that there is subclass `_IterDataPipeMeta` specifically for `IterDataPipe`.
    """

    type: _DataPipeType

    def __new__(cls, name, bases, namespace, **kwargs):
        # Create a new metaclass instance for `DataPipe` classes
        return super().__new__(cls, name, bases, namespace, **kwargs)  # type: ignore[call-overload]

        # TODO: the statements below are not reachable by design as there is a bug and typing is low priority for now.
        cls.__origin__ = None
        if "type" in namespace:
            return super().__new__(cls, name, bases, namespace, **kwargs)  # type: ignore[call-overload]

        namespace["__type_class__"] = False
        # For plain derived class without annotation, set default attributes
        for base in bases:
            if isinstance(base, _DataPipeMeta):
                return super().__new__(cls, name, bases, namespace, **kwargs)  # type: ignore[call-overload]

        namespace.update(
            {"type": _DEFAULT_TYPE, "__init_subclass__": _dp_init_subclass}
        )
        return super().__new__(cls, name, bases, namespace, **kwargs)  # type: ignore[call-overload]

    def __init__(self, name, bases, namespace, **kwargs):
        super().__init__(name, bases, namespace, **kwargs)  # type: ignore[call-overload]

    # TODO: Fix isinstance bug
    @_tp_cache
    # 定义 _getitem_ 方法，处理实例的参数获取逻辑
    def _getitem_(self, params):
        # 如果参数为空，抛出类型错误异常
        if params is None:
            raise TypeError(f"{self.__name__}[t]: t can not be None")
        # 如果参数是字符串，将其转换为 ForwardRef 对象
        if isinstance(params, str):
            params = ForwardRef(params)
        # 如果参数不是元组，将其转换为单元素元组
        if not isinstance(params, tuple):
            params = (params,)

        # 构建错误消息，要求参数必须是类型
        msg = f"{self.__name__}[t]: t must be a type"
        # 对参数中的每个元素进行类型检查，确保都是类型对象
        params = tuple(_type_check(p, msg) for p in params)

        # 如果 self.type.param 是 _GenericAlias 类型
        if isinstance(self.type.param, _GenericAlias):
            # 获取 self.type.param 的原始类型
            orig = getattr(self.type.param, "__origin__", None)
            # 如果原始类型是类型且不是 Generic
            if isinstance(orig, type) and orig is not Generic:
                # 获取参数对应的类型
                p = self.type.param[params]  # type: ignore[index]
                # 创建一个新的 DataPipeType 对象
                t = _DataPipeType(p)
                # 计算类名长度
                l = len(str(self.type)) + 2
                # 构建新的类名
                name = self.__name__[:-l]
                name = name + "[" + str(t) + "]"
                # 组合基类
                bases = (self,) + self.__bases__
                # 返回一个新的类对象
                return self.__class__(
                    name,
                    bases,
                    {
                        "__init_subclass__": _dp_init_subclass,
                        "type": t,
                        "__type_class__": True,
                    },
                )

        # 如果参数数量大于1，抛出类型错误异常
        if len(params) > 1:
            raise TypeError(
                f"Too many parameters for {self} actual {len(params)}, expected 1"
            )

        # 创建一个新的 DataPipeType 对象
        t = _DataPipeType(params[0])

        # 如果 t 不是 self.type 的子类型，抛出类型错误异常
        if not t.issubtype(self.type):
            raise TypeError(
                f"Can not subclass a DataPipe[{t}] from DataPipe[{self.type}]"
            )

        # 如果类型相等，直接返回 self
        if self.type == t:
            return self

        # 构建新的类名和基类
        name = self.__name__ + "[" + str(t) + "]"
        bases = (self,) + self.__bases__

        # 返回一个新的类对象
        return self.__class__(
            name,
            bases,
            {"__init_subclass__": _dp_init_subclass, "__type_class__": True, "type": t},
        )

    # TODO: Fix isinstance bug
    # 定义 _eq_ 方法，处理实例的相等比较逻辑
    def _eq_(self, other):
        # 如果 other 不是 _DataPipeMeta 类型，返回 NotImplemented
        if not isinstance(other, _DataPipeMeta):
            return NotImplemented
        # 如果 self.__origin__ 或 other.__origin__ 为 None，返回 self 是否等于 other
        if self.__origin__ is None or other.__origin__ is None:  # type: ignore[has-type]
            return self is other
        # 返回 self.__origin__ 和 other.__origin__ 是否相等，以及 self.type 是否等于 other.type
        return (
            self.__origin__ == other.__origin__  # type: ignore[has-type]
            and self.type == other.type
        )

    # TODO: Fix isinstance bug
    # 定义 _hash_ 方法，返回实例的哈希值
    def _hash_(self):
        return hash((self.__name__, self.type))
class _IterDataPipeMeta(_DataPipeMeta):
    r"""
    `IterDataPipe` 的元类，继承自 `_DataPipeMeta`。

    添加了一些特定于 `IterDataPipe` 的行为函数。
    """

    def __new__(cls, name, bases, namespace, **kwargs):
        # 如果命名空间中有 "reset" 方法
        if "reset" in namespace:
            reset_func = namespace["reset"]

            @functools.wraps(reset_func)
            def conditional_reset(*args, **kwargs):
                r"""
                当 `_SnapshotState` 是 `Iterating` 或 `NotStarted` 时才执行 DataPipe 的 `reset()` 方法。

                这允许最近恢复的 DataPipe 在初始 `__iter__` 调用期间保留其恢复状态。
                """
                datapipe = args[0]
                # 只有在 `_snapshot_state` 是 `Iterating` 或 `NotStarted` 时才重置状态
                if datapipe._snapshot_state in (
                    _SnapshotState.Iterating,
                    _SnapshotState.NotStarted,
                ):
                    # 重置 `_number_of_samples_yielded`，因为 DataPipe 的 `source_datapipe` 可能已经开始迭代。
                    datapipe._number_of_samples_yielded = 0
                    datapipe._fast_forward_iterator = None
                    reset_func(*args, **kwargs)
                datapipe._snapshot_state = _SnapshotState.Iterating

            # 将修改后的 reset 方法设置回命名空间中
            namespace["reset"] = conditional_reset

        # 如果命名空间中有 "__iter__" 方法，则调用 hook_iterator 函数
        if "__iter__" in namespace:
            hook_iterator(namespace)

        # 调用父类的 __new__ 方法创建类
        return super().__new__(cls, name, bases, namespace, **kwargs)  # type: ignore[call-overload]


def _dp_init_subclass(sub_cls, *args, **kwargs):
    # 为 datapipe 实例添加类型强化函数
    sub_cls.reinforce_type = reinforce_type

    # TODO:
    # - 添加用于编译时类型检查的全局开关

    # 忽略内部类型类
    if getattr(sub_cls, "__type_class__", False):
        return

    # 检查字符串类型是否有效
    if isinstance(sub_cls.type.param, ForwardRef):
        base_globals = sys.modules[sub_cls.__module__].__dict__
        try:
            # 尝试评估类型参数
            param = _eval_type(sub_cls.type.param, base_globals, locals())
            sub_cls.type.param = param
        except TypeError as e:
            raise TypeError(
                f"{sub_cls.type.param.__forward_arg__} is not supported by Python typing"
            ) from e
    # 检查子类（sub_cls）是否定义了 '__iter__' 方法
    if "__iter__" in sub_cls.__dict__:
        # 获取 '__iter__' 方法对象
        iter_fn = sub_cls.__dict__["__iter__"]
        # 获取 '__iter__' 方法的类型提示信息
        hints = get_type_hints(iter_fn)
        # 检查是否存在返回值类型提示
        if "return" in hints:
            # 获取返回类型提示
            return_hint = hints["return"]
            # 对于 Python 3.6 的普通返回类型提示
            if return_hint == Iterator:
                return
            # 检查返回类型提示是否为 Iterator 或其子类
            if not (
                hasattr(return_hint, "__origin__")
                and (
                    return_hint.__origin__ == Iterator
                    or return_hint.__origin__ == collections.abc.Iterator
                )
            ):
                # 如果不是期望的返回类型，引发类型错误
                raise TypeError(
                    "Expected 'Iterator' as the return annotation for `__iter__` of {}"
                    ", but found {}".format(
                        sub_cls.__name__, _type_repr(hints["return"])
                    )
                )
            # 获取迭代器返回值类型的泛型参数
            data_type = return_hint.__args__[0]
            # 检查迭代器返回值类型是否是子类的参数化类型的子类型
            if not issubtype(data_type, sub_cls.type.param):
                raise TypeError(
                    f"Expected return type of '__iter__' as a subtype of {sub_cls.type},"
                    f" but found {_type_repr(data_type)} for {sub_cls.__name__}"
                )
# 对于 DataPipe 实例进行类型强化

# 如果 expected_type 是一个元组，则将其转换为 Tuple 类型
if isinstance(expected_type, tuple):
    expected_type = Tuple[expected_type]

# 检查 expected_type 是否符合类型要求，确保其是一个类型而不是具体的实例
_type_check(expected_type, msg="'expected_type' must be a type")

# 检查 expected_type 是否是 self.type.param 的子类型
if not issubtype(expected_type, self.type.param):
    # 如果不是子类型，则抛出 TypeError 异常
    raise TypeError(
        f"Expected 'expected_type' as subtype of {self.type}, but found {_type_repr(expected_type)}"
    )

# 将 self.type 设置为经过强化后的类型
self.type = _DataPipeType(expected_type)

# 返回当前对象自身，以支持链式调用
return self
```