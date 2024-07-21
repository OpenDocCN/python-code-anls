# `.\pytorch\torch\utils\_pytree.py`

```py
"""
Contains utility functions for working with nested python data structures.

A *pytree* is Python nested data structure. It is a tree in the sense that
nodes are Python collections (e.g., list, tuple, dict) and the leaves are
Python values. Furthermore, a pytree should not contain reference cycles.

pytrees are useful for working with nested collections of Tensors. For example,
one can use `tree_map` to map a function over all Tensors inside some nested
collection of Tensors and `tree_leaves` to get a flat list of all Tensors
inside some nested collection. pytrees are helpful for implementing nested
collection support for PyTorch APIs.

This pytree implementation is not very performant due to Python overhead
To improve the performance we can move parts of the implementation to C++.
"""

import dataclasses
import functools
import importlib
import json
import sys
import threading
import types
import warnings
from collections import defaultdict, deque, namedtuple, OrderedDict
from typing import (
    Any,
    Callable,
    cast,
    DefaultDict,
    Deque,
    Dict,
    FrozenSet,
    Generic,
    Hashable,
    Iterable,
    List,
    Mapping,
    NamedTuple,
    Optional,
    OrderedDict as GenericOrderedDict,
    overload,
    Protocol,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
)
from typing_extensions import deprecated


__all__ = [
    "PyTree",
    "Context",
    "FlattenFunc",
    "UnflattenFunc",
    "DumpableContext",
    "ToDumpableContextFn",
    "FromDumpableContextFn",
    "TreeSpec",
    "LeafSpec",
    "keystr",
    "key_get",
    "register_pytree_node",
    "tree_flatten",
    "tree_flatten_with_path",
    "tree_unflatten",
    "tree_iter",
    "tree_leaves",
    "tree_leaves_with_path",
    "tree_structure",
    "tree_map",
    "tree_map_with_path",
    "tree_map_",
    "tree_map_only",
    "tree_map_only_",
    "tree_all",
    "tree_any",
    "tree_all_only",
    "tree_any_only",
    "treespec_dumps",
    "treespec_loads",
    "treespec_pprint",
]


T = TypeVar("T")
S = TypeVar("S")
U = TypeVar("U")
R = TypeVar("R")


DEFAULT_TREESPEC_SERIALIZATION_PROTOCOL = 1
NO_SERIALIZED_TYPE_NAME_FOUND = "NO_SERIALIZED_TYPE_NAME_FOUND"


class KeyEntry(Protocol):
    def __hash__(self) -> int:
        ...

    def __eq__(self, other: object) -> bool:
        ...

    def __str__(self) -> str:
        ...

    def get(self, parent: Any) -> Any:
        ...


Context = Any  # 上下文变量可以是任何类型
PyTree = Any  # pytree 是任何类型的 Python 嵌套数据结构
FlattenFunc = Callable[[PyTree], Tuple[List[Any], Context]]  # 扁平化函数类型定义
UnflattenFunc = Callable[[Iterable[Any], Context], PyTree]  # 反扁平化函数类型定义
DumpableContext = Any  # 可以被 JSON 序列化的上下文变量
ToDumpableContextFn = Callable[[Context], DumpableContext]  # 转换为可 JSON 序列化的上下文函数类型定义
FromDumpableContextFn = Callable[[DumpableContext], Context]  # 从 JSON 反序列化为上下文函数类型定义
ToStrFunc = Callable[["TreeSpec", List[str]], str]  # 转换为字符串函数类型定义
MaybeFromStrFunc = Callable[[str], Optional[Tuple[Any, Context, str]]]  # 从字符串转换为可能的元组函数类型定义
KeyPath = Tuple[KeyEntry, ...]  # 键路径的类型定义，由 KeyEntry 组成的元组
FlattenWithKeysFunc = Callable[[PyTree], Tuple[List[Tuple[KeyEntry, Any]], Any]]  # 带键扁平化函数类型定义
# A NodeDef holds two callables:
# - flatten_fn should take the collection and return a flat list of values.
#   It can also return some context that is used in reconstructing the
#   collection.
# - unflatten_fn should take a flat list of values and some context
#   (returned by flatten_fn). It returns the collection by reconstructing
#   it from the list and the context.
# - flatten_with_keys_fn, which is a callable that takes a
#   pytree and returns a list of (keypath, value) pairs and a context.
class NodeDef(NamedTuple):
    type: Type[Any]                # 定义节点类型
    flatten_fn: FlattenFunc        # 用于将集合扁平化的函数
    unflatten_fn: UnflattenFunc    # 用于将扁平化的列表和上下文还原为集合的函数
    flatten_with_keys_fn: Optional[FlattenWithKeysFunc]  # 可选的用于键路径扁平化的函数


_NODE_REGISTRY_LOCK = threading.Lock()   # 创建线程锁对象
SUPPORTED_NODES: Dict[Type[Any], NodeDef] = {}   # 支持的节点类型及其定义


# _SerializeNodeDef holds the following:
# - typ: the type of the node (e.g., "Dict", "List", etc)
# - serialized_type_name: the fully qualified name of the type, e.g. "collections.OrderedDict"
# - to_dumpable_context takes a TreeSpec, and returns a serialized string format of the
#   context, and the version number
# - from_dumpable_context takes in a string representation of the context, and the
#   version, and returns the deserialized context
class _SerializeNodeDef(NamedTuple):
    typ: Type[Any]                    # 节点类型
    serialized_type_name: str         # 类型的完全限定名，例如 "collections.OrderedDict"
    to_dumpable_context: Optional[ToDumpableContextFn]     # 将 TreeSpec 转换为序列化字符串格式的函数
    from_dumpable_context: Optional[FromDumpableContextFn] # 从序列化字符串表示的上下文和版本中返回反序列化的上下文


SUPPORTED_SERIALIZED_TYPES: Dict[Type[Any], _SerializeNodeDef] = {}   # 支持的序列化类型及其定义
SERIALIZED_TYPE_TO_PYTHON_TYPE: Dict[str, Type[Any]] = {}             # 序列化类型到 Python 类型的映射字典


def register_pytree_node(
    cls: Type[Any],
    flatten_fn: FlattenFunc,
    unflatten_fn: UnflattenFunc,
    *,
    serialized_type_name: Optional[str] = None,
    to_dumpable_context: Optional[ToDumpableContextFn] = None,
    from_dumpable_context: Optional[FromDumpableContextFn] = None,
    flatten_with_keys_fn: Optional[FlattenWithKeysFunc] = None,
) -> None:
    """Register a container-like type as pytree node.
    """
    # 使用线程安全的上下文管理器，确保注册过程中不会被中断或同时访问
    with _NODE_REGISTRY_LOCK:
        # 如果要注册的类型已经在支持的节点列表中，抛出数值错误异常
        if cls in SUPPORTED_NODES:
            raise ValueError(f"{cls} is already registered as pytree node.")

    # 调用私有函数来注册 pytree 节点，将类、展平函数、展开函数等信息注册到全局注册表中
    _private_register_pytree_node(
        cls,
        flatten_fn,
        unflatten_fn,
        serialized_type_name=serialized_type_name,
        to_dumpable_context=to_dumpable_context,
        from_dumpable_context=from_dumpable_context,
        flatten_with_keys_fn=flatten_with_keys_fn,
    )

    # 尝试导入 C++ 扩展模块，用于更高效的 pytree 节点注册
    try:
        from . import _cxx_pytree as cxx
    except ImportError:
        pass
    else:
        # 如果导入成功，则使用 C++ 扩展模块注册相同的 pytree 节点信息
        cxx._private_register_pytree_node(
            cls,
            flatten_fn,
            unflatten_fn,
            serialized_type_name=serialized_type_name,
            to_dumpable_context=to_dumpable_context,
            from_dumpable_context=from_dumpable_context,
        )
# 注册一个命名元组作为有效的 pytree 节点。默认情况下，命名元组是有效的 pytree 节点，但它们不能被序列化。
# 该 API 提供了 `serialized_type_name` 参数，允许这些命名元组被序列化。
#
# Args:
#     cls: 要注册的数据类类型
#     serialized_type_name: 数据类的序列化名称。如果要序列化包含此命名元组的 pytree TreeSpec，则需要此参数。
def _register_namedtuple(
    cls: Type[Any],
    *,
    serialized_type_name: str,
) -> None:
    # 使用私有函数 `_private_register_pytree_node` 注册 pytree 节点
    _private_register_pytree_node(
        cls,
        _namedtuple_flatten,  # 用于将命名元组展平的函数
        _namedtuple_unflatten,  # 用于从展平的数据恢复命名元组的函数
        serialized_type_name=serialized_type_name,  # 序列化时使用的类型名称
        to_dumpable_context=_namedtuple_serialize,  # 将命名元组序列化为可转储上下文的函数
        from_dumpable_context=_namedtuple_deserialize,  # 从可转储上下文中反序列化命名元组的函数
        flatten_with_keys_fn=_namedtuple_flatten_with_keys,  # 用于带键展平命名元组的函数
    )


@deprecated(
    "`torch.utils._pytree._register_pytree_node` is deprecated. "
    "Please use `torch.utils._pytree.register_pytree_node` instead.",
    category=FutureWarning,
)
# 注册一个类似容器的类型作为 Python pytree 的节点。
def _register_pytree_node(
    cls: Type[Any],
    flatten_fn: FlattenFunc,
    unflatten_fn: UnflattenFunc,
    to_str_fn: Optional[ToStrFunc] = None,  # 已弃用
    maybe_from_str_fn: Optional[MaybeFromStrFunc] = None,  # 已弃用
    *,
    serialized_type_name: Optional[str] = None,  # 可选的序列化名称
    to_dumpable_context: Optional[ToDumpableContextFn] = None,  # 将对象转换为可转储上下文的函数
    from_dumpable_context: Optional[FromDumpableContextFn] = None,  # 从可转储上下文中恢复对象的函数
    flatten_with_keys_fn: Optional[FlattenWithKeysFunc] = None,  # 带键展平对象的函数
) -> None:
    """Register a container-like type as pytree node for the Python pytree only.
    
    Args:
        cls: The dataclass type to register.
        flatten_fn: Function to flatten the object.
        unflatten_fn: Function to unflatten the object.
        to_str_fn: Deprecated function for converting to string.
        maybe_from_str_fn: Deprecated function for maybe converting from string.
        serialized_type_name: Optional serialized name for the dataclass.
        to_dumpable_context: Function to convert object to dumpable context.
        from_dumpable_context: Function to recover object from dumpable context.
        flatten_with_keys_fn: Function to flatten object with keys.
    """
    Args:
        cls: 要注册的类型
        flatten_fn: 一个可调用对象，接受一个 pytree 并返回该 pytree 的扁平表示及额外的上下文，用于表示扁平化的 pytree。
        unflatten_fn: 一个可调用对象，接受 pytree 的扁平化版本、额外的上下文，并返回未扁平化的 pytree。
        serialized_type_name: 用于指定序列化树规范时使用的完全限定名称的关键字参数。
        to_dumpable_context: 可选关键字参数，自定义指定如何将 pytree 的上下文转换为自定义的可 JSON 序列化表示形式。
                            这用于当前正在使用的 torch.export 中的 JSON 序列化。
        from_dumpable_context: 可选关键字参数，自定义指定如何将上下文的自定义 JSON 可序列化表示形式转换回原始上下文。
                               这用于当前正在使用的 torch.export 中的 JSON 反序列化。
        flatten_with_keys_fn: 可选关键字参数，用于指定在扁平化和树映射时如何访问每个 pytree 叶子的键路径。
                              类似于 `flatten_fn`，但是它应该返回一个 List[(keypath, leaf)]，而不是 List[leaf]。
    """
    if to_str_fn is not None or maybe_from_str_fn is not None:
        # 如果存在 `to_str_fn` 或 `maybe_from_str_fn`，发出警告，因为它们已经弃用。
        # 请改用 `to_dumpable_context` 和 `from_dumpable_context`。
        warnings.warn(
            "`to_str_fn` and `maybe_from_str_fn` is deprecated. "
            "Please use `to_dumpable_context` and `from_dumpable_context` instead.",
            FutureWarning,
            stacklevel=2,
        )

    # 调用内部函数 `_private_register_pytree_node` 来注册 pytree 节点
    _private_register_pytree_node(
        cls,
        flatten_fn,
        unflatten_fn,
        serialized_type_name=serialized_type_name,
        to_dumpable_context=to_dumpable_context,
        from_dumpable_context=from_dumpable_context,
        flatten_with_keys_fn=flatten_with_keys_fn,
    )
# 定义一个私有函数，用于注册一个 pytree 节点类型，仅供 Python pytree 使用。最终用户应使用 `register_pytree_node` 函数。
def _private_register_pytree_node(
    cls: Type[Any],  # 接受一个类作为参数，表示要注册的节点类型
    flatten_fn: FlattenFunc,  # 用于扁平化的函数，将节点展平为数据结构
    unflatten_fn: UnflattenFunc,  # 用于反扁平化的函数，将数据结构还原为节点
    *,
    serialized_type_name: Optional[str] = None,  # 序列化类型的名称，可选参数，默认为 None
    to_dumpable_context: Optional[ToDumpableContextFn] = None,  # 将节点转换为可转储上下文的函数，可选参数，默认为 None
    from_dumpable_context: Optional[FromDumpableContextFn] = None,  # 从可转储上下文转换回节点的函数，可选参数，默认为 None
    flatten_with_keys_fn: Optional[FlattenWithKeysFunc] = None,  # 与键一起扁平化的函数，可选参数，默认为 None
) -> None:  # 函数无返回值
    """This is an internal function that is used to register a pytree node type
    for the Python pytree only. End-users should use :func:`register_pytree_node`
    instead.
    """
    with _NODE_REGISTRY_LOCK:  # 使用节点注册锁，确保注册过程线程安全
        if cls in SUPPORTED_NODES:  # 如果要注册的类已经在支持的节点中
            # TODO: 在 OSS/内部稳定后，将此警告更改为错误
            warnings.warn(
                f"{cls} is already registered as pytree node. "
                "Overwriting the previous registration.",
            )

        # 创建节点定义对象
        node_def = NodeDef(cls, flatten_fn, unflatten_fn, flatten_with_keys_fn)
        SUPPORTED_NODES[cls] = node_def  # 将节点定义对象存入支持的节点字典中

        # 检查 to_dumpable_context 和 from_dumpable_context 参数是否同时为 None 或已注册
        if (to_dumpable_context is None) ^ (from_dumpable_context is None):
            raise ValueError(
                f"Both to_dumpable_context and from_dumpable_context for {cls} must "
                "be None or registered."
            )

        # 如果 serialized_type_name 参数为 None，则设置为默认值
        if serialized_type_name is None:
            serialized_type_name = NO_SERIALIZED_TYPE_NAME_FOUND

        # 创建序列化节点定义对象
        serialize_node_def = _SerializeNodeDef(
            cls,
            serialized_type_name,
            to_dumpable_context,
            from_dumpable_context,
        )
        SUPPORTED_SERIALIZED_TYPES[cls] = serialize_node_def  # 将序列化节点定义存入支持的序列化类型字典中
        SERIALIZED_TYPE_TO_PYTHON_TYPE[serialized_type_name] = cls  # 将序列化类型名称映射到 Python 类型


@dataclasses.dataclass(frozen=True)
class SequenceKey(Generic[T]):
    idx: int

    def __str__(self) -> str:
        return f"[{self.idx!r}]"

    def get(self, sequence: Sequence[T]) -> T:
        return sequence[self.idx]


K = TypeVar("K", bound=Hashable)


@dataclasses.dataclass(frozen=True)
class MappingKey(Generic[K, T]):
    key: K

    def __str__(self) -> str:
        return f"[{self.key!r}]"

    def get(self, mapping: Mapping[K, T]) -> T:
        return mapping[self.key]


@dataclasses.dataclass(frozen=True)
class GetAttrKey:
    name: str

    def __str__(self) -> str:
        return f".{self.name}"

    def get(self, obj: Any) -> Any:
        return getattr(obj, self.name)


# 定义一个函数，用于扁平化元组
def _tuple_flatten(d: Tuple[Any, ...]) -> Tuple[List[Any], Context]:
    return list(d), None


# 定义一个函数，用于带键扁平化元组
def _tuple_flatten_with_keys(
    d: Tuple[Any, ...]
) -> Tuple[List[Tuple[KeyEntry, Any]], Context]:
    values, context = _tuple_flatten(d)
    return [(SequenceKey(i), v) for i, v in enumerate(values)], context


# 定义一个函数，用于反扁平化元组
def _tuple_unflatten(values: Iterable[Any], context: Context) -> Tuple[Any, ...]:
    return tuple(values)


# 定义一个函数，用于扁平化列表
def _list_flatten(d: List[Any]) -> Tuple[List[Any], Context]:
    return d, None


# 定义一个函数，用于带键扁平化列表
def _list_flatten_with_keys(d: List[Any]) -> Tuple[List[Tuple[KeyEntry, Any]], Context]:
    values, context = _list_flatten(d)
    # 使用列表推导式构建一个由元组组成的列表，元组的第一个元素是通过 SequenceKey 函数处理后的索引 i，第二个元素是 values 列表中的对应元素 v。
    return [(SequenceKey(i), v) for i, v in enumerate(values)], context
# 将输入的可迭代对象转换为列表并返回
def _list_unflatten(values: Iterable[Any], context: Context) -> List[Any]:
    return list(values)


# 将字典的值转换为列表，并返回包含该列表和键的元组
def _dict_flatten(d: Dict[Any, Any]) -> Tuple[List[Any], Context]:
    return list(d.values()), list(d.keys())


# 将字典的值转换为列表，并返回包含键和值的元组的列表
def _dict_flatten_with_keys(
    d: Dict[Any, Any]
) -> Tuple[List[Tuple[KeyEntry, Any]], Context]:
    values, context = _dict_flatten(d)
    return [(MappingKey(k), v) for k, v in zip(context, values)], context


# 根据给定的键和值序列生成字典并返回
def _dict_unflatten(values: Iterable[Any], context: Context) -> Dict[Any, Any]:
    return dict(zip(context, values))


# 将命名元组转换为包含其字段值的列表，并返回其类型作为上下文
def _namedtuple_flatten(d: NamedTuple) -> Tuple[List[Any], Context]:
    return list(d), type(d)


# 将命名元组转换为包含字段名和字段值的元组的列表，并返回其类型作为上下文
def _namedtuple_flatten_with_keys(
    d: NamedTuple,
) -> Tuple[List[Tuple[KeyEntry, Any]], Context]:
    values, context = _namedtuple_flatten(d)
    return (
        [(GetAttrKey(field), v) for field, v in zip(context._fields, values)],
        context,
    )


# 根据给定的值和上下文类型生成命名元组并返回
def _namedtuple_unflatten(values: Iterable[Any], context: Context) -> NamedTuple:
    return cast(NamedTuple, context(*values))


# 根据上下文返回可序列化的命名元组类型名称
def _namedtuple_serialize(context: Context) -> DumpableContext:
    if context not in SUPPORTED_SERIALIZED_TYPES:
        raise NotImplementedError(
            f"Can't serialize TreeSpec of namedtuple class {context} because we "
            "didn't register a serializated_type_name. Please register using "
            "`_register_namedtuple`."
        )

    serialize_node_def = SUPPORTED_SERIALIZED_TYPES[context]
    serialized_type_name = serialize_node_def.serialized_type_name

    if serialized_type_name == NO_SERIALIZED_TYPE_NAME_FOUND:
        raise NotImplementedError(
            f"Can't serialize TreeSpec of namedtuple class {context} because we "
            "couldn't find a serializated_type_name. Please register using "
            "`_register_namedtuple`."
        )
    return serialized_type_name


# 根据可序列化的命名元组类型名称返回对应的命名元组类型
def _namedtuple_deserialize(dumpable_context: DumpableContext) -> Context:
    if dumpable_context not in SERIALIZED_TYPE_TO_PYTHON_TYPE:
        raise NotImplementedError(
            f"Can't deserialize TreeSpec of namedtuple class {dumpable_context} "
            "because we couldn't find a serializated name."
        )

    typ = SERIALIZED_TYPE_TO_PYTHON_TYPE[dumpable_context]
    return typ


# 将有序字典的值转换为列表，并返回包含该列表和键的元组
def _ordereddict_flatten(d: GenericOrderedDict[Any, Any]) -> Tuple[List[Any], Context]:
    return list(d.values()), list(d.keys())


# 将有序字典的值转换为列表，并返回包含键和值的元组的列表
def _ordereddict_flatten_with_keys(
    d: GenericOrderedDict[Any, Any]
) -> Tuple[List[Tuple[KeyEntry, Any]], Context]:
    values, context = _ordereddict_flatten(d)
    return [(MappingKey(k), v) for k, v in zip(context, values)], context


# 根据给定的值和上下文类型生成有序字典并返回
def _ordereddict_unflatten(
    values: Iterable[Any],
    context: Context,
) -> GenericOrderedDict[Any, Any]:
    return OrderedDict((key, value) for key, value in zip(context, values))


# 为了简化，将有序字典的扁平化函数赋值给简称以供使用
_odict_flatten = _ordereddict_flatten
# 为了简化，将有序字典的展开函数赋值给简称以供使用
_odict_unflatten = _ordereddict_unflatten
# 定义一个函数 _defaultdict_flatten，接受一个 DefaultDict 类型的参数 d，返回一个元组，包含值列表和上下文
def _defaultdict_flatten(d: DefaultDict[Any, Any]) -> Tuple[List[Any], Context]:
    # 调用 _dict_flatten 函数，将 DefaultDict 类型的 d 扁平化为值列表和字典上下文
    values, dict_context = _dict_flatten(d)
    # 返回值列表和包含默认工厂函数与字典上下文的元组
    return values, [d.default_factory, dict_context]


# 定义一个函数 _defaultdict_flatten_with_keys，接受一个 DefaultDict 类型的参数 d，返回一个元组，包含键值对元组列表和上下文
def _defaultdict_flatten_with_keys(
    d: DefaultDict[Any, Any]
) -> Tuple[List[Tuple[KeyEntry, Any]], Context]:
    # 调用 _defaultdict_flatten 函数，将 DefaultDict 类型的 d 扁平化为值列表和上下文
    values, context = _defaultdict_flatten(d)
    # 从上下文中获取字典上下文
    _, dict_context = context
    # 将字典上下文中的键值对转换为由 MappingKey 封装的键和值的元组列表
    return [(MappingKey(k), v) for k, v in zip(dict_context, values)], context


# 定义一个函数 _defaultdict_unflatten，接受一个值列表和上下文作为参数，返回一个 DefaultDict 类型的对象
def _defaultdict_unflatten(
    values: Iterable[Any],
    context: Context,
) -> DefaultDict[Any, Any]:
    # 从上下文中获取默认工厂函数和字典上下文
    default_factory, dict_context = context
    # 使用默认工厂函数和字典上下文将值列表反扁平化为 DefaultDict 类型的对象
    return defaultdict(default_factory, _dict_unflatten(values, dict_context))


# 定义一个函数 _defaultdict_serialize，接受一个上下文作为参数，返回一个序列化后的上下文对象
def _defaultdict_serialize(context: Context) -> DumpableContext:
    # 从上下文中获取默认工厂函数和字典上下文
    default_factory, dict_context = context
    # 创建一个 JSON 格式的序列化后的 DefaultDict 上下文对象
    json_defaultdict = {
        "default_factory_module": default_factory.__module__,
        "default_factory_name": default_factory.__qualname__,
        "dict_context": dict_context,
    }
    # 返回 JSON 格式的序列化后的 DefaultDict 上下文对象
    return json_defaultdict


# 定义一个函数 _defaultdict_deserialize，接受一个可序列化的 DefaultDict 上下文对象作为参数，返回一个上下文对象
def _defaultdict_deserialize(dumpable_context: DumpableContext) -> Context:
    # 断言 dumpable_context 是一个字典对象，并且包含特定的键
    assert isinstance(dumpable_context, dict)
    assert set(dumpable_context) == {
        "default_factory_module",
        "default_factory_name",
        "dict_context",
    }

    # 从 dumpable_context 中获取默认工厂函数的模块和名称
    default_factory_module = dumpable_context["default_factory_module"]
    default_factory_name = dumpable_context["default_factory_name"]
    assert isinstance(default_factory_module, str)
    assert isinstance(default_factory_name, str)
    # 动态导入模块
    module = importlib.import_module(default_factory_module)
    # 从模块中获取默认工厂函数对象
    default_factory = getattr(module, default_factory_name)

    # 从 dumpable_context 中获取字典上下文
    dict_context = dumpable_context["dict_context"]
    # 返回包含默认工厂函数和字典上下文的列表对象
    return [default_factory, dict_context]


# 定义一个函数 _deque_flatten，接受一个 Deque 类型的参数 d，返回一个元组，包含值列表和上下文
def _deque_flatten(d: Deque[Any]) -> Tuple[List[Any], Context]:
    # 将 Deque 对象转换为列表，并返回列表和 Deque 的最大长度作为上下文
    return list(d), d.maxlen


# 定义一个函数 _deque_flatten_with_keys，接受一个 Deque 类型的参数 d，返回一个元组，包含键值对元组列表和上下文
def _deque_flatten_with_keys(
    d: Deque[Any],
) -> Tuple[List[Tuple[KeyEntry, Any]], Context]:
    # 调用 _deque_flatten 函数，将 Deque 类型的 d 扁平化为值列表和上下文
    values, context = _deque_flatten(d)
    # 将值列表转换为由 SequenceKey 封装的索引和值的元组列表，并返回
    return [(SequenceKey(i), v) for i, v in enumerate(values)], context


# 定义一个函数 _deque_unflatten，接受一个值列表和上下文作为参数，返回一个 Deque 类型的对象
def _deque_unflatten(values: Iterable[Any], context: Context) -> Deque[Any]:
    # 使用值列表和上下文中的最大长度参数创建 Deque 对象，并返回
    return deque(values, maxlen=context)


# 调用 _private_register_pytree_node 函数注册 tuple 类型的序列化和反序列化方法
_private_register_pytree_node(
    tuple,
    _tuple_flatten,
    _tuple_unflatten,
    serialized_type_name="builtins.tuple",
    flatten_with_keys_fn=_tuple_flatten_with_keys,
)

# 调用 _private_register_pytree_node 函数注册 list 类型的序列化和反序列化方法
_private_register_pytree_node(
    list,
    _list_flatten,
    _list_unflatten,
    serialized_type_name="builtins.list",
    flatten_with_keys_fn=_list_flatten_with_keys,
)

# 调用 _private_register_pytree_node 函数注册 dict 类型的序列化和反序列化方法
_private_register_pytree_node(
    dict,
    _dict_flatten,
    _dict_unflatten,
    serialized_type_name="builtins.dict",
    flatten_with_keys_fn=_dict_flatten_with_keys,
)

# 调用 _private_register_pytree_node 函数注册 namedtuple 类型的序列化和反序列化方法
_private_register_pytree_node(
    namedtuple,  # type: ignore[arg-type]
    _namedtuple_flatten,
    _namedtuple_unflatten,
    serialized_type_name="collections.namedtuple",
    to_dumpable_context=_namedtuple_serialize,
    from_dumpable_context=_namedtuple_deserialize,
)
    flatten_with_keys_fn=_namedtuple_flatten_with_keys,


# 使用指定的函数将命名元组展平并保留键
# 使用私有函数注册 OrderedDict 类型的 pytree 节点，提供序列化和反序列化函数及其他相关信息
_private_register_pytree_node(
    OrderedDict,
    _ordereddict_flatten,
    _ordereddict_unflatten,
    serialized_type_name="collections.OrderedDict",
    flatten_with_keys_fn=_ordereddict_flatten_with_keys,
)

# 使用私有函数注册 defaultdict 类型的 pytree 节点，提供序列化和反序列化函数及其他相关信息
_private_register_pytree_node(
    defaultdict,
    _defaultdict_flatten,
    _defaultdict_unflatten,
    serialized_type_name="collections.defaultdict",
    to_dumpable_context=_defaultdict_serialize,
    from_dumpable_context=_defaultdict_deserialize,
    flatten_with_keys_fn=_defaultdict_flatten_with_keys,
)

# 使用私有函数注册 deque 类型的 pytree 节点，提供序列化和反序列化函数及其他相关信息
_private_register_pytree_node(
    deque,
    _deque_flatten,
    _deque_unflatten,
    serialized_type_name="collections.deque",
    flatten_with_keys_fn=_deque_flatten_with_keys,
)

# 定义一个不可变的集合，包含标准字典类型（dict、OrderedDict、defaultdict）
STANDARD_DICT_TYPES: FrozenSet[type] = frozenset(
    {dict, OrderedDict, defaultdict},
)

# 定义一个不可变的集合，包含内置类型（tuple、list、dict、namedtuple、OrderedDict、defaultdict、deque）
BUILTIN_TYPES: FrozenSet[type] = frozenset(
    {tuple, list, dict, namedtuple, OrderedDict, defaultdict, deque},  # type: ignore[arg-type]
)

# 检查对象是否为 namedtuple 的实例
def _is_namedtuple_instance(tree: Any) -> bool:
    typ = type(tree)
    bases = typ.__bases__
    if len(bases) != 1 or bases[0] != tuple:
        return False
    fields = getattr(typ, "_fields", None)
    if not isinstance(fields, tuple):
        return False
    return all(type(entry) == str for entry in fields)

# 获取树结构中节点的类型
def _get_node_type(tree: Any) -> Any:
    if _is_namedtuple_instance(tree):
        return namedtuple
    return type(tree)

# 判断节点是否为叶子节点（非节点的情况）
def _is_leaf(tree: PyTree, is_leaf: Optional[Callable[[PyTree], bool]] = None) -> bool:
    return (is_leaf is not None and is_leaf(tree)) or _get_node_type(
        tree
    ) not in SUPPORTED_NODES

# 表示 pytree 结构的 TreeSpec 类，包含根节点类型、上下文、每个子节点的规范、叶子节点数量等信息
@dataclasses.dataclass(init=True, frozen=True, eq=True, repr=False)
class TreeSpec:
    type: Any
    context: Context
    children_specs: List["TreeSpec"]

    num_nodes: int = dataclasses.field(init=False)
    num_leaves: int = dataclasses.field(init=False)
    num_children: int = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        num_nodes = sum((spec.num_nodes for spec in self.children_specs), start=1)
        num_leaves = sum(spec.num_leaves for spec in self.children_specs)
        num_children = len(self.children_specs)
        object.__setattr__(self, "num_nodes", num_nodes)
        object.__setattr__(self, "num_leaves", num_leaves)
        object.__setattr__(self, "num_children", num_children)
    # 返回表示对象的字符串表示形式，带有指定的缩进级别
    def __repr__(self, indent: int = 0) -> str:
        # 构建对象的前缀字符串，包括对象类型名称、上下文和起始列表标记
        repr_prefix: str = f"TreeSpec({self.type.__name__}, {self.context}, ["
        
        # 初始化子树规范字符串
        children_specs_str: str = ""
        
        # 如果存在子节点
        if self.num_children > 0:
            # 增加缩进级别
            indent += 2
            
            # 递归调用第一个子树的 __repr__ 方法，添加到子树字符串中
            children_specs_str += self.children_specs[0].__repr__(indent)
            
            # 如果存在多个子节点，添加逗号分隔的其余子节点字符串
            children_specs_str += "," if self.num_children > 1 else ""
            children_specs_str += ",".join(
                [
                    "\n" + " " * indent + child.__repr__(indent)
                    for child in self.children_specs[1:]
                ]
            )
        
        # 构建对象的后缀字符串，包括子树规范字符串和结束列表标记
        repr_suffix: str = f"{children_specs_str}])"
        
        # 返回对象的字符串表示形式
        return repr_prefix + repr_suffix

    # 检查对象是否为叶子节点，即节点数和叶子节点数均为1时返回 True
    def is_leaf(self) -> bool:
        return self.num_nodes == 1 and self.num_leaves == 1

    # 将当前节点及其所有子节点展平，直到指定的树对象
    def flatten_up_to(self, tree: PyTree) -> List[PyTree]:
        # 存储展平后的子树列表
        subtrees: List[PyTree] = []
        
        # 辅助方法，递归地展平当前节点及其子节点到指定的树对象
        self._flatten_up_to_helper(tree, subtrees)
        
        # 返回展平后的子树列表
        return subtrees

    # 根据给定的叶子节点列表重新构建一个 PyTree 对象
    def unflatten(self, leaves: Iterable[Any]) -> PyTree:
        # 如果 leaves 不是列表或元组，则转换为列表
        if not isinstance(leaves, (list, tuple)):
            leaves = list(leaves)
        
        # 如果 leaves 的长度与当前对象引用的叶子节点数不匹配，则引发 ValueError 异常
        if len(leaves) != self.num_leaves:
            raise ValueError(
                f"treespec.unflatten(leaves): `leaves` has length {len(leaves)} "
                f"but the spec refers to a pytree that holds {self.num_leaves} "
                f"items ({self}).",
            )
        
        # 如果当前节点为叶子节点，则直接返回第一个叶子节点作为结果
        if self.is_leaf():
            return leaves[0]
        
        # 获取当前节点类型对应的 unflatten 函数
        unflatten_fn = SUPPORTED_NODES[self.type].unflatten_fn
        
        # 递归地对每个子节点进行 unflatten 操作，并组装成子树列表
        start = 0
        end = 0
        child_pytrees = []
        for child_spec in self.children_specs:
            end += child_spec.num_leaves
            child_pytrees.append(child_spec.unflatten(leaves[start:end]))
            start = end
        
        # 调用对应的 unflatten 函数重新构建当前节点及其子节点的 PyTree 对象
        return unflatten_fn(child_pytrees, self.context)
class LeafSpec(TreeSpec):
    # LeafSpec 类继承自 TreeSpec 类，用于表示树中的叶子节点规范

    def __init__(self) -> None:
        # LeafSpec 的初始化方法
        super().__init__(None, None, [])
        # 调用父类 TreeSpec 的初始化方法，并传入特定参数

    def __post_init__(self) -> None:
        # 在对象初始化之后执行的方法
        object.__setattr__(self, "num_nodes", 1)
        object.__setattr__(self, "num_leaves", 1)
        object.__setattr__(self, "num_children", 0)
        # 使用 setattr 方法为对象动态添加属性及其对应的值

    def __repr__(self, indent: int = 0) -> str:
        # 返回对象的字符串表示形式，带有缩进参数
        return "*"
        # 返回一个星号作为 LeafSpec 对象的字符串表示


# 所有叶子节点都是相同的，因此使用单个对象表示以节省对象构造时间
_LEAF_SPEC = LeafSpec()


def _tree_flatten_helper(
    tree: PyTree,
    leaves: List[Any],
    is_leaf: Optional[Callable[[PyTree], bool]] = None,
) -> TreeSpec:
    # 辅助函数，用于将树结构扁平化为值列表和对应的 TreeSpec

    if _is_leaf(tree, is_leaf=is_leaf):
        # 如果当前节点是叶子节点
        leaves.append(tree)
        # 将该叶子节点添加到 leaves 列表中
        return _LEAF_SPEC
        # 返回预定义的 _LEAF_SPEC 表示这是一个叶子节点

    node_type = _get_node_type(tree)
    # 获取当前节点的类型
    flatten_fn = SUPPORTED_NODES[node_type].flatten_fn
    # 根据节点类型获取相应的 flatten 函数
    child_pytrees, context = flatten_fn(tree)
    # 使用 flatten 函数处理当前节点，获取子树列表和上下文信息

    # 递归地扁平化子节点
    children_specs = [
        _tree_flatten_helper(child, leaves, is_leaf=is_leaf) for child in child_pytrees
    ]
    # 对子节点逐一调用 _tree_flatten_helper 进行扁平化，并得到每个子节点的 TreeSpec

    return TreeSpec(node_type, context, children_specs)
    # 返回当前节点的 TreeSpec 对象


def tree_flatten(
    tree: PyTree,
    is_leaf: Optional[Callable[[PyTree], bool]] = None,
) -> Tuple[List[Any], TreeSpec]:
    """Flattens a pytree into a list of values and a TreeSpec that can be used
    to reconstruct the pytree.
    """
    # 将 pytree 扁平化为值列表和 TreeSpec 对象，用于重建 pytree

    leaves: List[Any] = []
    # 初始化一个空列表 leaves，用于存储叶子节点的值
    spec = _tree_flatten_helper(tree, leaves, is_leaf=is_leaf)
    # 调用辅助函数 _tree_flatten_helper 进行实际的扁平化操作
    return leaves, spec
    # 返回叶子节点列表和最终的 TreeSpec 对象


def tree_unflatten(leaves: Iterable[Any], treespec: TreeSpec) -> PyTree:
    """Given a list of values and a TreeSpec, builds a pytree.
    This is the inverse operation of `tree_flatten`.
    """
    # 根据值列表和 TreeSpec 对象构建 pytree
    if not isinstance(treespec, TreeSpec):
        # 如果 treespec 不是 TreeSpec 类型的对象
        raise TypeError(
            f"tree_unflatten(leaves, treespec): Expected `treespec` to be "
            f"instance of TreeSpec but got item of type {type(treespec)}.",
        )
    return treespec.unflatten(leaves)
    # 调用 TreeSpec 对象的 unflatten 方法，根据 leaves 和 treespec 构建 pytree


def tree_iter(
    tree: PyTree,
    is_leaf: Optional[Callable[[PyTree], bool]] = None,
) -> Iterable[Any]:
    """Get an iterator over the leaves of a pytree."""
    # 获取 pytree 中叶子节点的迭代器
    if _is_leaf(tree, is_leaf=is_leaf):
        # 如果当前节点是叶子节点
        yield tree
        # 使用生成器返回当前叶子节点
    else:
        node_type = _get_node_type(tree)
        # 获取当前节点的类型
        flatten_fn = SUPPORTED_NODES[node_type].flatten_fn
        # 根据节点类型获取相应的 flatten 函数
        child_pytrees, _ = flatten_fn(tree)

        # 递归地获取子节点的叶子节点迭代器
        for child in child_pytrees:
            yield from tree_iter(child, is_leaf=is_leaf)


def tree_leaves(
    tree: PyTree,
    is_leaf: Optional[Callable[[PyTree], bool]] = None,
) -> List[Any]:
    """Get a list of leaves of a pytree."""
    # 获取 pytree 中所有叶子节点的列表
    return list(tree_iter(tree, is_leaf=is_leaf))


def tree_structure(
    tree: PyTree,
    is_leaf: Optional[Callable[[PyTree], bool]] = None,
) -> TreeSpec:
    """Get the TreeSpec for a pytree."""
    # 获取 pytree 的 TreeSpec 对象
    return tree_flatten(tree, is_leaf=is_leaf)[1]


def tree_map(
    func: Callable[..., Any],
    tree: PyTree,
    *rests: PyTree,
    is_leaf: Optional[Callable[[PyTree], bool]] = None,
) -> PyTree:
    # 对 pytree 中的每个节点应用指定的函数
    """Map a multi-input function over pytree args to produce a new pytree.
    
    See also :func:`tree_map_`.
    
    >>> tree_map(lambda x: x + 1, {'x': 7, 'y': (42, 64)})
    {'x': 8, 'y': (43, 65)}
    >>> tree_map(lambda x: x is None, {'x': 7, 'y': (42, 64), 'z': None})
    {'x': False, 'y': (False, False), 'z': True}
    
    If multiple inputs are given, the structure of the tree is taken from the first input;
    subsequent inputs need only have ``tree`` as a prefix:
    
    >>> tree_map(lambda x, y: [x] + y, [5, 6], [[7, 9], [1, 2]])
    [[5, 7, 9], [6, 1, 2]]
    
    Args:
        func (callable): A function that takes ``1 + len(rests)`` arguments, to be applied at the
            corresponding leaves of the pytrees.
        tree (pytree): A pytree to be mapped over, with each leaf providing the first positional
            argument to function ``func``.
        rests (tuple of pytree): A tuple of pytrees, each of which has the same structure as
            ``tree`` or has ``tree`` as a prefix.
        is_leaf (callable, optional): An extra leaf predicate function that will be called at each
            flattening step. The function should have a single argument with signature
            ``is_leaf(node) -> bool``. If it returns :data:`True`, the whole subtree being treated
            as a leaf. Otherwise, the default pytree registry will be used to determine a node is a
            leaf or not. If the function is not specified, the default pytree registry will be used.
    
    Returns:
        A new pytree with the same structure as ``tree`` but with the value at each leaf given by
        ``func(x, *xs)`` where ``x`` is the value at the corresponding leaf in ``tree`` and ``xs``
        is the tuple of values at corresponding nodes in ``rests``.
    """
    leaves, treespec = tree_flatten(tree, is_leaf=is_leaf)
    # Flatten the input pytree into a list of its leaves and a treespec describing its structure
    flat_args = [leaves] + [treespec.flatten_up_to(r) for r in rests]
    # Construct a list of arguments to be passed to func, including the leaves and flattened rests
    return treespec.unflatten(map(func, *flat_args))
    # Apply func to the flattened arguments and reconstruct the pytree structure using treespec
# 定义一个函数 tree_map_，用于在树形数据结构上进行映射操作，并返回原始树
def tree_map_(
    func: Callable[..., Any],
    tree: PyTree,
    *rests: PyTree,
    is_leaf: Optional[Callable[[PyTree], bool]] = None,
) -> PyTree:
    """Like :func:`tree_map`, but do an inplace call on each leaf and return the original tree.

    See also :func:`tree_map`.

    Args:
        func (callable): A function that takes ``1 + len(rests)`` arguments, to be applied at the
            corresponding leaves of the pytrees.
        tree (pytree): A pytree to be mapped over, with each leaf providing the first positional
            argument to function ``func``.
        rests (tuple of pytree): A tuple of pytrees, each of which has the same structure as
            ``tree`` or has ``tree`` as a prefix.
        is_leaf (callable, optional): An extra leaf predicate function that will be called at each
            flattening step. The function should have a single argument with signature
            ``is_leaf(node) -> bool``. If it returns :data:`True`, the whole subtree being treated
            as a leaf. Otherwise, the default pytree registry will be used to determine a node is a
            leaf or not. If the function is not specified, the default pytree registry will be used.

    Returns:
        The original ``tree`` with the value at each leaf is given by the side-effect of function
        ``func(x, *xs)`` (not the return value) where ``x`` is the value at the corresponding leaf
        in ``tree`` and ``xs`` is the tuple of values at values at corresponding nodes in ``rests``.
    """
    # 将树结构展平，获取所有叶子节点和树结构描述符
    leaves, treespec = tree_flatten(tree, is_leaf=is_leaf)
    # 构建函数所需的参数列表，包括所有叶子节点和 rests 中各个树结构的展平结果
    flat_args = [leaves] + [treespec.flatten_up_to(r) for r in rests]
    # 对参数列表应用函数 func，但仅处理迭代过程中的副作用，不处理返回值
    tuple(map(func, *flat_args))  # consume and exhaust the iterable
    # 返回原始的树结构
    return tree


# 定义类型 Type2, Type3, TypeAny, Fn2, Fn3, Fn, FnAny 和 MapOnlyFn 用于类型标注
Type2 = Tuple[Type[T], Type[S]]
Type3 = Tuple[Type[T], Type[S], Type[U]]
if sys.version_info >= (3, 10):
    TypeAny = Union[Type[Any], Tuple[Type[Any], ...], types.UnionType]
else:
    TypeAny = Union[Type[Any], Tuple[Type[Any], ...]]

Fn2 = Callable[[Union[T, S]], R]
Fn3 = Callable[[Union[T, S, U]], R]
Fn = Callable[[T], R]
FnAny = Callable[[Any], R]

MapOnlyFn = Callable[[T], Callable[[Any], Any]]


# 使用 @overload 装饰器定义多个重载版本的 map_only 函数，以支持不同的参数类型
@overload
def map_only(__type_or_types_or_pred: Type2[T, S]) -> MapOnlyFn[Fn2[T, S, Any]]:
    ...


@overload
def map_only(__type_or_types_or_pred: Type3[T, S, U]) -> MapOnlyFn[Fn3[T, S, U, Any]]:
    ...


@overload
def map_only(__type_or_types_or_pred: Type[T]) -> MapOnlyFn[Fn[T, Any]]:
    ...


@overload
def map_only(__type_or_types_or_pred: TypeAny) -> MapOnlyFn[FnAny[Any]]:
    ...


@overload
def map_only(__type_or_types_or_pred: Callable[[Any], bool]) -> MapOnlyFn[FnAny[Any]]:
    ...


# 最终的实现函数，根据参数类型 __type_or_types_or_pred 返回特定的映射函数
def map_only(
    __type_or_types_or_pred: Union[TypeAny, Callable[[Any], bool]]
) -> MapOnlyFn[FnAny[Any]]:
    """
    Suppose you are writing a tree_map over tensors, leaving everything
    """
    # 省略函数体，具体功能根据参数类型的不同而变化
    pass
    """
    Define a decorator function that conditionally applies another function based on type or predicate.

    If __type_or_types_or_pred is a type or tuple of types, or a UnionType (available in Python 3.10+), 
    create a predicate function 'pred' that checks if an object x is an instance of __type_or_types_or_pred.

    If __type_or_types_or_pred is callable, assign it directly to 'pred'.

    If neither of the above conditions is met, raise a TypeError.

    Define a wrapper function 'wrapper' that takes a function 'func' as input. It applies 'func' to its 
    argument 'x' only if 'pred(x)' returns True. Otherwise, it returns 'x' unchanged.

    Return the 'wrapper' function as the decorator function.
    """
    if isinstance(__type_or_types_or_pred, (type, tuple)) or (
        sys.version_info >= (3, 10)
        and isinstance(__type_or_types_or_pred, types.UnionType)
    ):
        # Define a predicate function that checks if an object is an instance of the specified type(s)
        def pred(x: Any) -> bool:
            return isinstance(x, __type_or_types_or_pred)  # type: ignore[arg-type]

    elif callable(__type_or_types_or_pred):
        # If __type_or_types_or_pred is callable, use it directly as the predicate function
        pred = __type_or_types_or_pred  # type: ignore[assignment]
    else:
        # Raise an error if __type_or_types_or_pred does not match any expected types or callable
        raise TypeError("Argument must be a type, a tuple of types, or a callable.")

    # Define a decorator function that wraps another function 'func'
    def wrapper(func: Callable[[T], Any]) -> Callable[[Any], Any]:
        @functools.wraps(func)
        def wrapped(x: T) -> Any:
            # Apply 'func' to 'x' only if 'pred(x)' is True; otherwise, return 'x' unchanged
            if pred(x):
                return func(x)
            return x

        return wrapped

    # Return the decorator function 'wrapper'
    return wrapper
@overload
def tree_map_only(
    __type_or_types_or_pred: Type[T],
    func: Fn[T, Any],
    tree: PyTree,
    is_leaf: Optional[Callable[[PyTree], bool]] = None,
) -> PyTree:
    ...


@overload
def tree_map_only(
    __type_or_types_or_pred: Type2[T, S],
    func: Fn2[T, S, Any],
    tree: PyTree,
    is_leaf: Optional[Callable[[PyTree], bool]] = None,
) -> PyTree:
    ...


@overload
def tree_map_only(
    __type_or_types_or_pred: Type3[T, S, U],
    func: Fn3[T, S, U, Any],
    tree: PyTree,
    is_leaf: Optional[Callable[[PyTree], bool]] = None,
) -> PyTree:
    ...


@overload
def tree_map_only(
    __type_or_types_or_pred: Callable[[Any], bool],
    func: FnAny[Any],
    tree: PyTree,
    is_leaf: Optional[Callable[[PyTree], bool]] = None,
) -> PyTree:
    ...


def tree_map_only(
    __type_or_types_or_pred: Union[TypeAny, Callable[[Any], bool]],
    func: FnAny[Any],
    tree: PyTree,
    is_leaf: Optional[Callable[[PyTree], bool]] = None,
) -> PyTree:
    # 调用 map_only 函数，生成一个新的函数，接受类型或判断函数作为参数
    mapped_func = map_only(__type_or_types_or_pred)(func)
    # 调用 tree_map 函数，使用上一步得到的函数处理树结构，并可选地指定叶子节点的判断函数
    return tree_map(mapped_func, tree, is_leaf=is_leaf)


@overload
def tree_map_only_(
    __type_or_types_or_pred: Type[T],
    func: Fn[T, Any],
    tree: PyTree,
    is_leaf: Optional[Callable[[PyTree], bool]] = None,
) -> PyTree:
    ...


@overload
def tree_map_only_(
    __type_or_types_or_pred: Type2[T, S],
    func: Fn2[T, S, Any],
    tree: PyTree,
    is_leaf: Optional[Callable[[PyTree], bool]] = None,
) -> PyTree:
    ...


@overload
def tree_map_only_(
    __type_or_types_or_pred: Type3[T, S, U],
    func: Fn3[T, S, U, Any],
    tree: PyTree,
    is_leaf: Optional[Callable[[PyTree], bool]] = None,
) -> PyTree:
    ...


@overload
def tree_map_only_(
    __type_or_types_or_pred: Callable[[Any], bool],
    func: FnAny[Any],
    tree: PyTree,
    is_leaf: Optional[Callable[[PyTree], bool]] = None,
) -> PyTree:
    ...


def tree_map_only_(
    __type_or_types_or_pred: Union[TypeAny, Callable[[Any], bool]],
    func: FnAny[Any],
    tree: PyTree,
    is_leaf: Optional[Callable[[PyTree], bool]] = None,
) -> PyTree:
    # 调用 map_only 函数，生成一个新的函数，接受类型或判断函数作为参数
    mapped_func = map_only(__type_or_types_or_pred)(func)
    # 调用 tree_map_ 函数，使用上一步得到的函数处理树结构，并可选地指定叶子节点的判断函数
    return tree_map_(mapped_func, tree, is_leaf=is_leaf)


def tree_all(
    pred: Callable[[Any], bool],
    tree: PyTree,
    is_leaf: Optional[Callable[[PyTree], bool]] = None,
) -> bool:
    # 通过 tree_iter 函数展平树结构，并应用预测函数于其中所有元素，返回是否全部为真
    flat_args = tree_iter(tree, is_leaf=is_leaf)
    return all(map(pred, flat_args))


def tree_any(
    pred: Callable[[Any], bool],
    tree: PyTree,
    is_leaf: Optional[Callable[[PyTree], bool]] = None,
) -> bool:
    # 通过 tree_iter 函数展平树结构，并应用预测函数于其中所有元素，返回是否至少一个为真
    flat_args = tree_iter(tree, is_leaf=is_leaf)
    return any(map(pred, flat_args))


@overload
def tree_all_only(
    __type_or_types: Type[T],
    pred: Fn[T, bool],
    tree: PyTree,
    is_leaf: Optional[Callable[[PyTree], bool]] = None,
) -> bool:
    ...


@overload
def tree_all_only(
    __type_or_types: Type2[T, S],
    pred: Fn2[T, S, bool],
    tree: PyTree,
    is_leaf: Optional[Callable[[PyTree], bool]] = None,
) -> bool:
    ...


@overload
def tree_all_only(
    __type_or_types: Type3[T, S, U],
    pred: Fn3[T, S, U, bool],
    tree: PyTree,
    is_leaf: Optional[Callable[[PyTree], bool]] = None,
) -> bool:
    ...


def tree_all_only(
    __type_or_types: Union[TypeAny],
    pred: FnAny[bool],
    tree: PyTree,
    is_leaf: Optional[Callable[[PyTree], bool]] = None,
) -> bool:
    # 调用 tree_all 函数，使用类型和预测函数处理树结构，并可选地指定叶子节点的判断函数
    return tree_all(lambda x: isinstance(x, __type_or_types) and pred(x), tree, is_leaf=is_leaf)
    # 定义一个名为 pred 的参数，类型为 Fn3[T, S, U, bool]
    pred: Fn3[T, S, U, bool],
    # 定义一个名为 tree 的参数，类型为 PyTree
    tree: PyTree,
    # 定义一个名为 is_leaf 的可选参数，类型为可调用对象，用于判断 PyTree 是否为叶子节点，初始值为 None
    is_leaf: Optional[Callable[[PyTree], bool]] = None,
# 定义一个类型提示的函数签名，表明函数返回一个布尔值
) -> bool:
    ...


# 定义一个函数，用于检查树形结构中所有符合条件的节点是否都满足某种条件
# 参数说明：
# - __type_or_types: 可以是单个类型或类型的列表，用于指定要检查的节点类型
# - pred: 用于检查节点的条件函数
# - tree: 表示输入的树形结构
# - is_leaf: 可选参数，用于检查是否为叶子节点的函数
# 返回值：布尔值，表示是否所有符合条件的节点都满足 pred 函数的条件
def tree_all_only(
    __type_or_types: TypeAny,
    pred: FnAny[bool],
    tree: PyTree,
    is_leaf: Optional[Callable[[PyTree], bool]] = None,
) -> bool:
    # 将树形结构展开为一个迭代器
    flat_args = tree_iter(tree, is_leaf=is_leaf)
    # 检查所有符合条件的节点是否都满足 pred 函数的条件
    return all(pred(x) for x in flat_args if isinstance(x, __type_or_types))


# 定义函数重载，用于检查树形结构中是否存在符合条件的节点
# 参数和返回值与上一个函数类似，区别在于检查条件为任意类型
@overload
def tree_any_only(
    __type_or_types: Type[T],
    pred: Fn[T, bool],
    tree: PyTree,
    is_leaf: Optional[Callable[[PyTree], bool]] = None,
) -> bool:
    ...


# 函数重载，用于检查树形结构中是否存在符合条件的节点
# 参数和返回值与上一个函数类似，区别在于检查条件为两种类型的组合
@overload
def tree_any_only(
    __type_or_types: Type2[T, S],
    pred: Fn2[T, S, bool],
    tree: PyTree,
    is_leaf: Optional[Callable[[PyTree], bool]] = None,
) -> bool:
    ...


# 函数重载，用于检查树形结构中是否存在符合条件的节点
# 参数和返回值与上一个函数类似，区别在于检查条件为三种类型的组合
@overload
def tree_any_only(
    __type_or_types: Type3[T, S, U],
    pred: Fn3[T, S, U, bool],
    tree: PyTree,
    is_leaf: Optional[Callable[[PyTree], bool]] = None,
) -> bool:
    ...


# 定义一个函数，用于检查树形结构中是否存在符合条件的节点
# 参数说明与 tree_all_only 相同
# 返回值：布尔值，表示是否存在符合条件的节点满足 pred 函数的条件
def tree_any_only(
    __type_or_types: TypeAny,
    pred: FnAny[bool],
    tree: PyTree,
    is_leaf: Optional[Callable[[PyTree], bool]] = None,
) -> bool:
    # 将树形结构展开为一个迭代器
    flat_args = tree_iter(tree, is_leaf=is_leaf)
    # 检查是否存在符合条件的节点满足 pred 函数的条件
    return any(pred(x) for x in flat_args if isinstance(x, __type_or_types))


# 广播一个树形结构到指定的树结构规范，并返回扁平化的值列表
# 如果无法广播，则返回 None。
#
# 参数说明：
# - tree: 输入的树形结构
# - treespec: 树结构的规范(TreeSpec 类型)，描述了树的结构和内容
# - is_leaf: 可选参数，用于检查是否为叶子节点的函数
# 返回值：可选的列表，包含根据规范扁平化后的值，或者 None（表示无法匹配规范）
def _broadcast_to_and_flatten(
    tree: PyTree,
    treespec: TreeSpec,
    is_leaf: Optional[Callable[[PyTree], bool]] = None,
) -> Optional[List[Any]]:
    # 断言 treespec 是 TreeSpec 类型的实例
    assert isinstance(treespec, TreeSpec)

    # 如果 tree 是叶子节点，则返回重复 tree 的列表，长度为 treespec.num_leaves
    if _is_leaf(tree, is_leaf=is_leaf):
        return [tree] * treespec.num_leaves
    # 如果 treespec 本身是叶子节点，则返回 None，表示无法匹配规范
    if treespec.is_leaf():
        return None
    # 获取节点的类型
    node_type = _get_node_type(tree)
    # 如果节点类型与规范的类型不匹配，则返回 None
    if node_type != treespec.type:
        return None

    # 获取节点的扁平化函数
    flatten_fn = SUPPORTED_NODES[node_type].flatten_fn
    # 使用扁平化函数获取子树结构和上下文
    child_pytrees, ctx = flatten_fn(tree)

    # 检查节点的子节点数量和上下文是否与规范匹配
    if len(child_pytrees) != treespec.num_children or ctx != treespec.context:
        return None

    # 递归扁平化子节点
    result: List[Any] = []
    for child, child_spec in zip(child_pytrees, treespec.children_specs):
        flat = _broadcast_to_and_flatten(child, child_spec, is_leaf=is_leaf)
        if flat is not None:
            result += flat
        else:
            return None

    return result


# 数据类，表示树结构的规范
# _TreeSpecSchema 是用于序列化 TreeSpec 的模式
# 包含以下字段：
# - type: 类型的字符串名称，对于 LeafSpec 的情况为 null
# - context: 任何可以 JSON 序列化的格式
# - children_spec: 子节点序列化规范的列表
@dataclasses.dataclass
class _TreeSpecSchema:
    type: Optional[str]
    # 定义变量 context，类型为 DumpableContext，用于存储上下文信息
    context: DumpableContext
    # 定义变量 children_spec，类型为 "_TreeSpecSchema" 类型的列表，用于存储子树规范的信息
    children_spec: List["_TreeSpecSchema"]
# 定义了一个名为 _ProtocolFn 的命名元组，包含两个属性：treespec_to_json 和 json_to_treespec，分别是将 TreeSpec 转换为 DumpableContext 和将 DumpableContext 转换为 TreeSpec 的可调用函数类型。
class _ProtocolFn(NamedTuple):
    treespec_to_json: Callable[[TreeSpec], DumpableContext]
    json_to_treespec: Callable[[DumpableContext], TreeSpec]

# _SUPPORTED_PROTOCOLS 是一个字典，用于存储支持的协议编号与对应的 _ProtocolFn 对象之间的映射关系。
_SUPPORTED_PROTOCOLS: Dict[int, _ProtocolFn] = {}

# _treespec_to_json 函数将给定的 TreeSpec 对象转换为 _TreeSpecSchema 对象。
def _treespec_to_json(treespec: TreeSpec) -> _TreeSpecSchema:
    # 如果 treespec 是叶子节点，则返回一个特定格式的 _TreeSpecSchema 对象
    if treespec.is_leaf():
        return _TreeSpecSchema(None, None, [])

    # 如果 treespec 的类型不在 SUPPORTED_SERIALIZED_TYPES 中，抛出未实现错误
    if treespec.type not in SUPPORTED_SERIALIZED_TYPES:
        raise NotImplementedError(
            f"Serializing {treespec.type} in pytree is not registered.",
        )

    # 获取序列化定义对象
    serialize_node_def = SUPPORTED_SERIALIZED_TYPES[treespec.type]

    # 获取序列化类型名
    serialized_type_name = serialize_node_def.serialized_type_name

    # 如果序列化类型名为 NO_SERIALIZED_TYPE_NAME_FOUND，则抛出未实现错误
    if serialized_type_name == NO_SERIALIZED_TYPE_NAME_FOUND:
        raise NotImplementedError(
            f"No registered serialization name for {treespec.type} found. "
            "Please update your _register_pytree_node call with a `serialized_type_name` kwarg."
        )

    # 如果序列化定义中没有提供 to_dumpable_context 函数，则尝试将 treespec 的 context 序列化为 JSON 字符串
    if serialize_node_def.to_dumpable_context is None:
        try:
            serialized_context = json.dumps(treespec.context)
        except TypeError as e:
            raise TypeError(
                "Unable to serialize context. "
                "Please make the context json dump-able, or register a "
                "custom serializer using _register_pytree_node."
            ) from e
    else:
        # 否则，使用序列化定义中提供的 to_dumpable_context 函数进行序列化
        serialized_context = serialize_node_def.to_dumpable_context(treespec.context)

    # 递归处理所有子节点，将每个子节点转换为 _TreeSpecSchema 对象，构建 child_schemas 列表
    child_schemas = [_treespec_to_json(child) for child in treespec.children_specs]

    # 返回一个 _TreeSpecSchema 对象，表示当前 treespec 的序列化结果
    return _TreeSpecSchema(serialized_type_name, serialized_context, child_schemas)


# _json_to_treespec 函数将给定的 DumpableContext 对象转换为 TreeSpec 对象。
def _json_to_treespec(json_schema: DumpableContext) -> TreeSpec:
    # 如果 json_schema 表示一个叶子节点的特定格式，则返回预定义的 _LEAF_SPEC 对象
    if (
        json_schema["type"] is None
        and json_schema["context"] is None
        and len(json_schema["children_spec"]) == 0
    ):
        return _LEAF_SPEC

    # 如果 json_schema 的类型不在 SERIALIZED_TYPE_TO_PYTHON_TYPE 中，抛出未实现错误
    if json_schema["type"] not in SERIALIZED_TYPE_TO_PYTHON_TYPE:
        raise NotImplementedError(
            f'Deserializing {json_schema["type"]} in pytree is not registered.',
        )

    # 获取类型映射
    typ = SERIALIZED_TYPE_TO_PYTHON_TYPE[json_schema["type"]]

    # 获取序列化定义对象
    serialize_node_def = SUPPORTED_SERIALIZED_TYPES[typ]

    # 如果序列化定义中没有提供 from_dumpable_context 函数，则尝试将 json_schema 的 context 反序列化为 Python 对象
    if serialize_node_def.from_dumpable_context is None:
        try:
            context = json.loads(json_schema["context"])
        except TypeError as ex:
            raise TypeError(
                "Unable to deserialize context. "
                "Please make the context json load-able, or register a "
                "custom serializer using _register_pytree_node.",
            ) from ex
    else:
        # 否则，使用序列化定义中提供的 from_dumpable_context 函数进行反序列化
        context = serialize_node_def.from_dumpable_context(json_schema["context"])

    # 递归处理所有子节点，将每个子节点的 DumpableContext 转换为 TreeSpec 对象，构建 children_specs 列表
    children_specs = []
    for child_string in json_schema["children_spec"]:
        children_specs.append(_json_to_treespec(child_string))

    # 返回一个 TreeSpec 对象，表示当前 json_schema 的反序列化结果
    return TreeSpec(typ, context, children_specs)


# 将 _treespec_to_json 和 _json_to_treespec 函数作为 _ProtocolFn 对象的属性，添加到 _SUPPORTED_PROTOCOLS 字典中，键为 1
_SUPPORTED_PROTOCOLS[1] = _ProtocolFn(_treespec_to_json, _json_to_treespec)
    # 检查 treespec 是否为 TreeSpec 的实例，如果不是则抛出 TypeError 异常
    if not isinstance(treespec, TreeSpec):
        raise TypeError(
            # 格式化字符串，指示错误的类型和详细信息
            f"treespec_dumps(treespec, protocol): Expected `treespec` to be instance of "
            f"TreeSpec but got item of type {type(treespec)}.",
        )

    # 如果 protocol 为 None，则使用默认的序列化协议
    if protocol is None:
        protocol = DEFAULT_TREESPEC_SERIALIZATION_PROTOCOL

    # 检查 protocol 是否在支持的协议列表中
    if protocol in _SUPPORTED_PROTOCOLS:
        # 使用对应协议的方法将 treespec 转换为 JSON 格式
        json_spec = _SUPPORTED_PROTOCOLS[protocol].treespec_to_json(treespec)
    else:
        # 如果 protocol 不在支持的协议列表中，则抛出 ValueError 异常
        raise ValueError(
            # 格式化字符串，指示未知的协议和可用的协议列表
            f"Unknown protocol {protocol}. "
            f"Available protocols: {list(_SUPPORTED_PROTOCOLS.keys())}",
        )

    # 将 protocol 和转换后的 JSON 数据（作为字典）转换为 JSON 字符串
    str_spec = json.dumps((protocol, dataclasses.asdict(json_spec)))
    # 返回最终的 JSON 字符串表示
    return str_spec
def treespec_loads(serialized: str) -> TreeSpec:
    # 解析序列化的字符串，获取协议和 JSON 模式
    protocol, json_schema = json.loads(serialized)

    # 如果协议在支持的协议列表中
    if protocol in _SUPPORTED_PROTOCOLS:
        # 调用相应协议的方法将 JSON 模式转换为 TreeSpec 对象
        return _SUPPORTED_PROTOCOLS[protocol].json_to_treespec(json_schema)
    # 如果协议不在支持的协议列表中，抛出值错误异常
    raise ValueError(
        f"Unknown protocol {protocol}. "
        f"Available protocols: {list(_SUPPORTED_PROTOCOLS.keys())}",
    )


class _DummyLeaf:
    def __repr__(self) -> str:
        # 返回表示该对象的字符串
        return "*"


def treespec_pprint(treespec: TreeSpec) -> str:
    # 创建一个虚拟树，用于打印展示
    dummy_tree = tree_unflatten(
        [_DummyLeaf() for _ in range(treespec.num_leaves)],
        treespec,
    )
    # 返回虚拟树的字符串表示形式
    return repr(dummy_tree)


# TODO(angelayi): remove this function after OSS/internal stabilize
@deprecated(
    "`pytree_to_str` is deprecated. Please use `treespec_dumps` instead.",
    category=FutureWarning,
)
def pytree_to_str(treespec: TreeSpec) -> str:
    # 将 TreeSpec 对象转换为字符串表示形式
    return treespec_dumps(treespec)


# TODO(angelayi): remove this function after OSS/internal stabilize
@deprecated(
    "`str_to_pytree` is deprecated. Please use `treespec_loads` instead.",
    category=FutureWarning,
)
def str_to_pytree(json: str) -> TreeSpec:
    # 将 JSON 字符串转换为 TreeSpec 对象
    return treespec_loads(json)


def arg_tree_leaves(*args: PyTree, **kwargs: PyTree) -> List[Any]:
    """Get a flat list of arguments to this function

    A slightly faster version of tree_leaves((args, kwargs))
    """
    # 获取函数的所有参数的扁平化列表
    leaves: List[Any] = []
    for a in args:
        leaves.extend(tree_iter(a))
    for a in kwargs.values():
        leaves.extend(tree_iter(a))
    return leaves


def tree_flatten_with_path(
    tree: PyTree,
    is_leaf: Optional[Callable[[PyTree], bool]] = None,
) -> Tuple[List[Tuple[KeyPath, Any]], TreeSpec]:
    """Flattens a pytree like :func:`tree_flatten`, but also returns each leaf's key path.

    Args:
        tree: a pytree to flatten. If it contains a custom type, that type must be
            registered with an appropriate `tree_flatten_with_path_fn` when registered
            with :func:`register_pytree_node`.
        is_leaf: An extra leaf predicate function that will be called at each
            flattening step. The function should have a single argument with signature
            ``is_leaf(node) -> bool``. If it returns :data:`True`, the whole subtree being treated
            as a leaf. Otherwise, the default pytree registry will be used to determine a node is a
            leaf or not. If the function is not specified, the default pytree registry will be used.
    Returns:
        A tuple where the first element is a list of (key path, leaf) pairs, and the
        second element is a :class:`TreeSpec` representing the structure of the flattened
        tree.
    """
    # 对 pytree 进行扁平化处理，并返回每个叶子节点的键路径
    _, treespec = tree_flatten(tree, is_leaf)
    return list(_generate_key_paths((), tree, is_leaf)), treespec


def tree_leaves_with_path(
    tree: PyTree,
    is_leaf: Optional[Callable[[PyTree], bool]] = None,
) -> List[Tuple[KeyPath, Any]]:
    """Gets the leaves of a pytree like ``tree_leaves`` and returns each leaf's key path.

    Args:
        tree: a pytree to extract leaves and key paths from.
        is_leaf: An extra leaf predicate function that will be called at each
            flattening step. The function should have a single argument with signature
            ``is_leaf(node) -> bool``. If it returns :data:`True`, the whole subtree being treated
            as a leaf. Otherwise, the default pytree registry will be used to determine a node is a
            leaf or not. If the function is not specified, the default pytree registry will be used.
    Returns:
        A list of tuples where each tuple consists of a key path and its corresponding leaf value.
    """
    # 获取 pytree 的所有叶子节点及其键路径
    # 根据给定的 pytree 结构生成所有键路径和叶子节点的对应关系列表
    Args:
        tree: pytree 数据结构。如果包含自定义类型，必须在使用 :func:`register_pytree_node` 注册时提供适当的 `tree_flatten_with_path_fn`。
        is_leaf: 一个额外的叶子节点判定函数，在每个展平步骤都会被调用。
            该函数应该具有一个参数，签名为 ``is_leaf(node) -> bool``。如果返回 :data:`True`，则整个子树将被视为叶子节点。
            否则，默认的 pytree 注册表将用于确定节点是否为叶子节点。
            如果未指定此函数，则将使用默认的 pytree 注册表。
    Returns:
        返回一个键路径和叶子节点组成的列表。
    """
    return list(_generate_key_paths((), tree, is_leaf))
# 生成包含键路径的迭代器，用于遍历树结构中的每个节点及其路径
def _generate_key_paths(
    key_path: KeyPath,
    tree: PyTree,
    is_leaf: Optional[Callable[[PyTree], bool]] = None,
) -> Iterable[Tuple[KeyPath, Any]]:
    # 如果定义了 is_leaf 函数并且当前节点是叶子节点，则返回当前节点的路径和值
    if is_leaf and is_leaf(tree):
        yield key_path, tree
        return

    # 获取当前节点的类型
    node_type = _get_node_type(tree)
    # 根据节点类型获取处理函数
    handler = SUPPORTED_NODES.get(node_type)
    # 如果没有找到处理函数，说明当前节点是叶子节点，返回当前节点的路径和值
    if not handler:
        yield key_path, tree
        return

    # 获取节点的 flatten_with_keys_fn 函数
    flatten_with_keys = handler.flatten_with_keys_fn
    # 如果存在 flatten_with_keys_fn 函数，则使用该函数将节点展开为键-子节点对，并递归生成键路径
    if flatten_with_keys:
        key_children, _ = flatten_with_keys(tree)
        for k, c in key_children:
            yield from _generate_key_paths((*key_path, k), c, is_leaf)
    else:
        # 如果节点注册了但未添加 flatten_with_keys_fn 函数，则抛出 ValueError 异常
        raise ValueError(
            f"Did not find a flatten_with_keys_fn for type: {node_type}. "
            "Please pass a flatten_with_keys_fn argument to register_pytree_node."
        )


# 对树结构中的每个叶子节点应用给定函数，并返回处理后的新树
def tree_map_with_path(
    func: Callable[..., Any],
    tree: PyTree,
    *rests: PyTree,
    is_leaf: Optional[Callable[[PyTree], bool]] = None,
) -> PyTree:
    """类似于 tree_map，但提供的函数接受额外的键路径参数。

    Args:
        func: 接受 ``2 + len(rests)`` 个参数的函数，应用于树结构的对应叶子节点。
              第一个位置参数是问题叶子节点的键路径。第二个位置参数是叶子节点的值。
        tree: 要映射的树结构，其中每个叶子节点作为函数 ``func`` 的第一个位置参数。
        rests: 一组树结构，每个树结构具有与 ``tree`` 相同的结构或以 ``tree`` 为前缀。
        is_leaf: 一个额外的叶子节点断言函数，在每个展开步骤中调用。函数应该接受一个参数，签名为
                 ``is_leaf(node) -> bool``。如果返回 :data:`True`，则整个子树被视为叶子节点。
                 否则，将使用默认的 pytree 注册表来确定节点是否是叶子节点。如果未指定该函数，
                 将使用默认的 pytree 注册表。

    Returns:
        与 ``tree`` 具有相同结构的新树，但每个叶子节点的值由 ``func(keypath, x, *xs)`` 给出，
        其中 ``keypath`` 是 ``tree`` 中相应叶子节点的键路径，``x`` 是该叶子节点的值，
        ``xs`` 是 ``rests`` 中对应节点的值的元组。
    """
    # 获取树结构的键路径和规范化信息
    keypath_leaves, treespec = tree_flatten_with_path(tree, is_leaf)
    keypath_leaves = list(zip(*keypath_leaves))
    # 将所有键路径和规范化信息合并到一个列表中
    all_keypath_leaves = keypath_leaves + [treespec.flatten_up_to(r) for r in rests]
    # 使用 func 函数对合并后的所有键路径和规范化信息执行解压缩操作，并返回结果
    return treespec.unflatten(func(*xs) for xs in zip(*all_keypath_leaves))


# 给定一个键路径，返回其漂亮打印的表示形式
def keystr(kp: KeyPath) -> str:
    """给定一个键路径，返回其漂亮打印的表示形式。"""
    return "".join([str(k) for k in kp])
# 给定一个对象和一个键路径，返回键路径指定位置的值
def key_get(obj: Any, kp: KeyPath) -> Any:
    # 遍历键路径中的每一个键
    for k in kp:
        # 使用当前键从对象中获取值，并更新对象为该值（可能是子对象）
        obj = k.get(obj)
    # 返回最终键路径指定位置的值
    return obj
```