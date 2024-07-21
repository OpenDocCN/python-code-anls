# `.\pytorch\torch\utils\_cxx_pytree.py`

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
"""

import functools
import sys
import types
from typing import (
    Any,
    Callable,
    Iterable,
    List,
    Optional,
    overload,
    Tuple,
    Type,
    TypeVar,
    Union,
)
from typing_extensions import deprecated

import torch

if torch._running_with_deploy():  # type: ignore[no-untyped-call]
    raise ImportError("C++ pytree utilities do not work with torch::deploy.")

import optree
from optree import PyTreeSpec  # direct import for type annotations

from torch.utils._pytree import KeyEntry


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


Context = Any
PyTree = Any
TreeSpec = PyTreeSpec
FlattenFunc = Callable[[PyTree], Tuple[List[Any], Context]]
UnflattenFunc = Callable[[Iterable[Any], Context], PyTree]
OpTreeUnflattenFunc = Callable[[Context, Iterable[Any]], PyTree]
DumpableContext = Any  # Any json dumpable text
ToDumpableContextFn = Callable[[Context], DumpableContext]
FromDumpableContextFn = Callable[[DumpableContext], Context]
KeyPath = Tuple[KeyEntry, ...]
FlattenWithKeysFunc = Callable[[PyTree], Tuple[List[Tuple[KeyEntry, Any]], Any]]


def _reverse_args(func: UnflattenFunc) -> OpTreeUnflattenFunc:
    # 返回一个新的函数，该函数调用时参数顺序与原函数相反
    @functools.wraps(func)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        return func(*reversed(args), **kwargs)

    return wrapped


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
    """
    Registers a Python class as a pytree node with custom functions for flattening and unflattening.

    Args:
    - cls: The Python class to register as a pytree node.
    - flatten_fn: Function to flatten an instance of cls into a list of data and a context.
    - unflatten_fn: Function to unflatten a list of data and a context into an instance of cls.
    - serialized_type_name: Optional name for the serialized type.
    - to_dumpable_context: Optional function to convert context into a JSON dumpable format.
    - from_dumpable_context: Optional function to convert JSON dumpable context back into context.
    - flatten_with_keys_fn: Optional function to flatten with keys.

    Returns:
    - None

    Notes:
    - This function registers the provided class as a pytree node with custom serialization functions.
    """
    pass  # 空函数体，仅用于声明函数结束
    """
    Register a container-like type as pytree node.
    
    Args:
        cls (type): A Python type to treat as an internal pytree node.
        flatten_fn (callable): A function to be used during flattening, taking an instance of
            ``cls`` and returning a pair, with (1) an iterable for the children to be flattened
            recursively, and (2) some hashable auxiliary data to be stored in the treespec and to be
            passed to the ``unflatten_fn``.
        unflatten_fn (callable): A function taking two arguments: the auxiliary data that was
            returned by ``flatten_fn`` and stored in the treespec, and the unflattened children.
            The function should return an instance of ``cls``.
        serialized_type_name (str, optional): A keyword argument used to specify the fully
            qualified name used when serializing the tree spec.
        to_dumpable_context (callable, optional): An optional keyword argument to custom specify how
            to convert the context of the pytree to a custom json dumpable representation. This is
            used for json serialization, which is being used in :mod:`torch.export` right now.
        from_dumpable_context (callable, optional): An optional keyword argument to custom specify
            how to convert the custom json dumpable representation of the context back to the
            original context. This is used for json deserialization, which is being used in
            :mod:`torch.export` right now.
    
    Example::
    
        >>> # xdoctest: +SKIP
        >>> # Registry a Python type with lambda functions
        >>> register_pytree_node(
        ...     set,
        ...     lambda s: (sorted(s), None, None),
        ...     lambda children, _: set(children),
        ... )
    """
    # 如果 flatten_with_keys_fn 不为 None，抛出 NotImplementedError
    if flatten_with_keys_fn is not None:
        raise NotImplementedError("KeyPaths are not yet supported in cxx_pytree.")
    
    # 调用 _private_register_pytree_node 函数注册 pytree 节点类型
    _private_register_pytree_node(
        cls,
        flatten_fn,
        unflatten_fn,
        serialized_type_name=serialized_type_name,
        to_dumpable_context=to_dumpable_context,
        from_dumpable_context=from_dumpable_context,
    )
    
    # 导入 _pytree 模块中的私有函数 _private_register_pytree_node
    from . import _pytree as python
    
    # 调用 _private_register_pytree_node 函数注册 pytree 节点类型，使用 python 模块中的函数
    python._private_register_pytree_node(
        cls,
        flatten_fn,
        unflatten_fn,
        serialized_type_name=serialized_type_name,
        to_dumpable_context=to_dumpable_context,
        from_dumpable_context=from_dumpable_context,
    )
@deprecated(
    "`torch.utils._cxx_pytree._register_pytree_node` is deprecated. "
    "Please use `torch.utils._cxx_pytree.register_pytree_node` instead.",
    category=FutureWarning,
)
# 标记函数为已弃用，并提供替代方法的建议信息
def _register_pytree_node(
    cls: Type[Any],
    flatten_fn: FlattenFunc,
    unflatten_fn: UnflattenFunc,
    *,
    serialized_type_name: Optional[str] = None,
    to_dumpable_context: Optional[ToDumpableContextFn] = None,
    from_dumpable_context: Optional[FromDumpableContextFn] = None,
) -> None:
    """Register a container-like type as pytree node for the C++ pytree only.

    The ``namespace`` argument is used to avoid collisions that occur when different libraries
    register the same Python type with different behaviors. It is recommended to add a unique prefix
    to the namespace to avoid conflicts with other libraries. Namespaces can also be used to specify
    the same class in different namespaces for different use cases.

    .. warning::
        For safety reasons, a ``namespace`` must be specified while registering a custom type. It is
        used to isolate the behavior of flattening and unflattening a pytree node type. This is to
        prevent accidental collisions between different libraries that may register the same type.

    Args:
        cls (type): A Python type to treat as an internal pytree node.
        flatten_fn (callable): A function to be used during flattening, taking an instance of
            ``cls`` and returning a pair, with (1) an iterable for the children to be flattened
            recursively, and (2) some hashable auxiliary data to be stored in the treespec and to be
            passed to the ``unflatten_fn``.
        unflatten_fn (callable): A function taking two arguments: the auxiliary data that was
            returned by ``flatten_fn`` and stored in the treespec, and the unflattened children.
            The function should return an instance of ``cls``.
        serialized_type_name (str, optional): A keyword argument used to specify the fully
            qualified name used when serializing the tree spec.
        to_dumpable_context (callable, optional): An optional keyword argument to custom specify how
            to convert the context of the pytree to a custom json dumpable representation. This is
            used for json serialization, which is being used in :mod:`torch.export` right now.
        from_dumpable_context (callable, optional): An optional keyword argument to custom specify
            how to convert the custom json dumpable representation of the context back to the
            original context. This is used for json deserialization, which is being used in
            :mod:`torch.export` right now.
    """

    # 调用内部函数来实际执行注册操作
    _private_register_pytree_node(
        cls,
        flatten_fn,
        unflatten_fn,
        serialized_type_name=serialized_type_name,
        to_dumpable_context=to_dumpable_context,
        from_dumpable_context=from_dumpable_context,
    )


def _private_register_pytree_node(
    cls: Type[Any],  # 声明一个参数 cls，类型为 Type[Any]，表示一个类对象
    flatten_fn: FlattenFunc,  # 声明一个参数 flatten_fn，类型为 FlattenFunc，表示一个用于扁平化的函数
    unflatten_fn: UnflattenFunc,  # 声明一个参数 unflatten_fn，类型为 UnflattenFunc，表示一个用于反扁平化的函数
    *,  # 表示接下来的参数都是关键字参数
    serialized_type_name: Optional[str] = None,  # 声明一个关键字参数 serialized_type_name，类型为可选的字符串，默认为 None，用于指定序列化类型的名称
    to_dumpable_context: Optional[ToDumpableContextFn] = None,  # 声明一个关键字参数 to_dumpable_context，类型为可选的 ToDumpableContextFn 函数，用于将对象转换为可转储的上下文
    from_dumpable_context: Optional[FromDumpableContextFn] = None,  # 声明一个关键字参数 from_dumpable_context，类型为可选的 FromDumpableContextFn 函数，用于从可转储的上下文中恢复对象
def register_pytree_node(
    cls: Type,
) -> None:
    """
    This is an internal function that is used to register a pytree node type
    for the C++ pytree only. End-users should use :func:`register_pytree_node`
    instead.
    """
    # TODO(XuehaiPan): remove this condition when we make Python pytree out-of-box support
    # PyStructSequence types
    # 检查是否不是结构序列类，如果不是，则注册 pytree 节点
    if not optree.is_structseq_class(cls):
        optree.register_pytree_node(
            cls,
            flatten_fn,
            _reverse_args(unflatten_fn),
            namespace="torch",
        )


def tree_flatten(
    tree: PyTree,
    is_leaf: Optional[Callable[[PyTree], bool]] = None,
) -> Tuple[List[Any], TreeSpec]:
    """
    Flatten a pytree.

    See also :func:`tree_unflatten`.

    The flattening order (i.e., the order of elements in the output list) is deterministic,
    corresponding to a left-to-right depth-first tree traversal.
    
    >>> tree = {'b': (2, [3, 4]), 'a': 1, 'c': None, 'd': 5}
    >>> tree_flatten(tree)
    ([1, 2, 3, 4, None, 5], PyTreeSpec({'a': *, 'b': (*, [*, *]), 'c': *, 'd': *}, NoneIsLeaf))
    >>> tree_flatten(1)
    ([1], PyTreeSpec(*, NoneIsLeaf))
    >>> tree_flatten(None)
    ([None], PyTreeSpec(*, NoneIsLeaf))

    For unordered dictionaries, :class:`dict` and :class:`collections.defaultdict`, the order is
    dependent on the **sorted** keys in the dictionary. Please use :class:`collections.OrderedDict`
    if you want to keep the keys in the insertion order.
    
    >>> from collections import OrderedDict
    >>> tree = OrderedDict([('b', (2, [3, 4])), ('a', 1), ('c', None), ('d', 5)])
    >>> tree_flatten(tree)
    ([2, 3, 4, 1, None, 5], PyTreeSpec(OrderedDict([('b', (*, [*, *])), ('a', *), ('c', *), ('d', *)]), NoneIsLeaf))

    Args:
        tree (pytree): A pytree to flatten.
        is_leaf (callable, optional): An extra leaf predicate function that will be called at each
            flattening step. The function should have a single argument with signature
            ``is_leaf(node) -> bool``. If it returns :data:`True`, the whole subtree being treated
            as a leaf. Otherwise, the default pytree registry will be used to determine a node is a
            leaf or not. If the function is not specified, the default pytree registry will be used.

    Returns:
        A pair ``(leaves, treespec)`` where the first element is a list of leaf values and the
        second element is a treespec representing the structure of the pytree.
    """
    return optree.tree_flatten(  # type: ignore[return-value]
        tree,
        is_leaf=is_leaf,
        none_is_leaf=True,
        namespace="torch",
    )


def tree_unflatten(leaves: Iterable[Any], treespec: TreeSpec) -> PyTree:
    """
    Reconstruct a pytree from the treespec and the leaves.

    The inverse of :func:`tree_flatten`.

    >>> tree = {'b': (2, [3, 4]), 'a': 1, 'c': None, 'd': 5}
    >>> leaves, treespec = tree_flatten(tree)
    >>> tree == tree_unflatten(leaves, treespec)
    True
    """
    # 从 leaves 和 treespec 重新构建 pytree
    return optree.tree_unflatten(leaves, treespec)
    # 检查 treespec 是否为 TreeSpec 类型，如果不是则抛出 TypeError 异常
    if not isinstance(treespec, TreeSpec):
        raise TypeError(
            f"tree_unflatten(values, spec): Expected `spec` to be instance of "
            f"TreeSpec but got item of type {type(treespec)}."
        )
    # 使用 optree 库中的 tree_unflatten 函数，根据给定的 treespec 和 leaves 重构 pytree 结构
    return optree.tree_unflatten(treespec, leaves)  # type: ignore[arg-type]
# 定义函数tree_iter，用于迭代pytree的叶子节点
def tree_iter(
    tree: PyTree,
    is_leaf: Optional[Callable[[PyTree], bool]] = None,
) -> Iterable[Any]:
    """Get an iterator over the leaves of a pytree.

    See also :func:`tree_flatten`.

    >>> tree = {'b': (2, [3, 4]), 'a': 1, 'c': None, 'd': 5}
    >>> list(tree_iter(tree))
    [1, 2, 3, 4, None, 5]
    >>> list(tree_iter(1))
    [1]
    >>> list(tree_iter(None))
    [None]

    Args:
        tree (pytree): A pytree to flatten.
        is_leaf (callable, optional): An extra leaf predicate function that will be called at each
            flattening step. The function should have a single argument with signature
            ``is_leaf(node) -> bool``. If it returns :data:`True`, the whole subtree being treated
            as a leaf. Otherwise, the default pytree registry will be used to determine a node is a
            leaf or not. If the function is not specified, the default pytree registry will be used.

    Returns:
        An iterator over the leaf values.
    """
    # 调用optree模块的tree_iter函数，返回迭代器，设置默认的none_is_leaf=True，命名空间为"torch"
    return optree.tree_iter(
        tree,
        is_leaf=is_leaf,
        none_is_leaf=True,
        namespace="torch",
    )


# 定义函数tree_leaves，用于获取pytree的叶子节点列表
def tree_leaves(
    tree: PyTree,
    is_leaf: Optional[Callable[[PyTree], bool]] = None,
) -> List[Any]:
    """Get the leaves of a pytree.

    See also :func:`tree_flatten`.

    >>> tree = {'b': (2, [3, 4]), 'a': 1, 'c': None, 'd': 5}
    >>> tree_leaves(tree)
    [1, 2, 3, 4, None, 5]
    >>> tree_leaves(1)
    [1]
    >>> tree_leaves(None)
    [None]

    Args:
        tree (pytree): A pytree to flatten.
        is_leaf (callable, optional): An extra leaf predicate function that will be called at each
            flattening step. The function should have a single argument with signature
            ``is_leaf(node) -> bool``. If it returns :data:`True`, the whole subtree being treated
            as a leaf. Otherwise, the default pytree registry will be used to determine a node is a
            leaf or not. If the function is not specified, the default pytree registry will be used.

    Returns:
        A list of leaf values.
    """
    # 调用optree模块的tree_leaves函数，返回叶子节点值的列表，设置默认的none_is_leaf=True，命名空间为"torch"
    return optree.tree_leaves(
        tree,
        is_leaf=is_leaf,
        none_is_leaf=True,
        namespace="torch",
    )


# 定义函数tree_structure，用于获取pytree的结构描述
def tree_structure(
    tree: PyTree,
    is_leaf: Optional[Callable[[PyTree], bool]] = None,
) -> TreeSpec:
    """Get the treespec for a pytree.

    See also :func:`tree_flatten`.

    >>> tree = {'b': (2, [3, 4]), 'a': 1, 'c': None, 'd': 5}
    >>> tree_structure(tree)
    PyTreeSpec({'a': *, 'b': (*, [*, *]), 'c': *, 'd': *}, NoneIsLeaf)
    >>> tree_structure(1)
    PyTreeSpec(*, NoneIsLeaf)
    >>> tree_structure(None)
    PyTreeSpec(*, NoneIsLeaf)

    Args:
        tree (pytree): A pytree to flatten.
        is_leaf (callable, optional): An extra leaf predicate function that will be called at each
            flattening step. The function should have a single argument with signature
            ``is_leaf(node) -> bool``. If it returns :data:`True`, the whole subtree being treated
            as a leaf. Otherwise, the default pytree registry will be used to determine a node is a
            leaf or not. If the function is not specified, the default pytree registry will be used.

    Returns:
        TreeSpec: The treespec describing the structure of the pytree.
    """
    # 调用optree模块的tree_structure函数，返回描述pytree结构的TreeSpec对象，设置默认的none_is_leaf=True，命名空间为"torch"
    return optree.tree_structure(
        tree,
        is_leaf=is_leaf,
        none_is_leaf=True,
        namespace="torch",
    )
    # 接受一个 pytree 参数并将其展平后返回其结构的 treespec 对象
    def flatten_tree(tree, is_leaf=None):
        # 返回 optree 模块中的 tree_structure 函数的结果，用于生成 pytree 的结构描述对象
        return optree.tree_structure(  # type: ignore[return-value]
            tree,
            # 如果提供了 is_leaf 参数，则使用提供的函数来判断是否为叶子节点
            is_leaf=is_leaf,
            # 设定 none_is_leaf 参数为 True，表示 None 节点也被视为叶子节点
            none_is_leaf=True,
            # namespace 参数设定为 "torch"，影响生成的结构对象的命名空间
            namespace="torch",
        )
# 使用给定的函数 func 对输入的 pytree 及其后续参数进行映射，返回一个新的 pytree
def tree_map(
    func: Callable[..., Any],
    tree: PyTree,
    *rests: PyTree,
    is_leaf: Optional[Callable[[PyTree], bool]] = None,
) -> PyTree:
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
    # 调用 optree 模块中的 tree_map 函数，传入参数 func, tree, *rests, is_leaf=is_leaf, none_is_leaf=True, namespace="torch"
    return optree.tree_map(
        func,
        tree,
        *rests,
        is_leaf=is_leaf,
        none_is_leaf=True,
        namespace="torch",
    )


def tree_map_(
    func: Callable[..., Any],
    tree: PyTree,
    *rests: PyTree,
    is_leaf: Optional[Callable[[PyTree], bool]] = None,
) -> PyTree:
    """Like :func:`tree_map`, but do an inplace call on each leaf and return the original tree.

    See also :func:`tree_map`.
    """
    # 与 tree_map 类似，但在每个叶子节点上进行就地调用，并返回原始的 tree 结构
    return optree.tree_map_(
        func,
        tree,
        *rests,
        is_leaf=is_leaf,
    )
    # 使用 `optree.tree_map_` 函数对给定的 pytree 进行映射操作
    # 将函数 `func` 应用于 pytree 的每个叶子节点及其对应的 rests 参数节点
    # 这个函数的作用是在每个叶子节点上执行 `func(x, *xs)`，并将结果作为副作用应用到原始 `tree` 上
    
    return optree.tree_map_(
        func,  # 要应用的函数，接受 `1 + len(rests)` 个参数
        tree,  # 要映射的 pytree，每个叶子节点提供 `func` 函数的第一个位置参数
        *rests,  # 包含多个 pytree 的元组，每个 pytree 结构与 `tree` 相同或为其前缀
        is_leaf=is_leaf,  # 可选参数，用于判断是否为叶子节点的函数，返回 True 表示该子树作为叶子处理
        none_is_leaf=True,  # 可选参数，指示 None 值是否应视为叶子节点
        namespace="torch",  # 可选参数，命名空间，用于查找和注册 pytree 类型的默认行为
    )
# 定义一个用于类型推断的类型别名，表示包含两种类型的元组
Type2 = Tuple[Type[T], Type[S]]
# 定义一个用于类型推断的类型别名，表示包含三种类型的元组
Type3 = Tuple[Type[T], Type[S], Type[U]]

# 根据 Python 版本选择合适的 TypeAny 类型别名定义
if sys.version_info >= (3, 10):
    TypeAny = Union[Type[Any], Tuple[Type[Any], ...], types.UnionType]
else:
    TypeAny = Union[Type[Any], Tuple[Type[Any], ...]]

# 定义一个接受两种类型参数并返回一个结果的函数类型别名
Fn2 = Callable[[Union[T, S]], R]
# 定义一个接受三种类型参数并返回一个结果的函数类型别名
Fn3 = Callable[[Union[T, S, U]], R]
# 定义一个接受一个类型参数并返回一个结果的函数类型别名
Fn = Callable[[T], R]
# 定义一个接受任意类型参数并返回一个结果的函数类型别名
FnAny = Callable[[Any], R]

# 定义一个映射函数类型别名，接受一个参数 T 并返回一个接受任意参数的函数
MapOnlyFn = Callable[[T], Callable[[Any], Any]]


# 以下是用于特定类型推断的重载函数声明，用于帮助 lambda 函数的类型推断
@overload
def map_only(__type_or_types_or_pred: Type2[T, S]) -> MapOnlyFn[Fn2[T, S, Any]]:
    ...


@overload
def map_only(__type_or_types_or_pred: Type3[T, S, U]) -> MapOnlyFn[Fn3[T, S, U, Any]]:
    ...


@overload
def map_only(__type_or_types_or_pred: Type[T]) -> MapOnlyFn[Fn[T, Any]]:
    ...


# 用于处理其它复杂类型或者可调用对象的重载函数声明
@overload
def map_only(__type_or_types_or_pred: TypeAny) -> MapOnlyFn[FnAny[Any]]:
    ...


@overload
def map_only(__type_or_types_or_pred: Callable[[Any], bool]) -> MapOnlyFn[FnAny[Any]]:
    ...


# 定义 map_only 函数，接受一个参数 __type_or_types_or_pred，并返回一个函数
def map_only(
    __type_or_types_or_pred: Union[TypeAny, Callable[[Any], bool]]
) -> MapOnlyFn[FnAny[Any]]:
    """
    Suppose you are writing a tree_map over tensors, leaving everything
    else unchanged.  Ordinarily you would have to write:

        def go(t):
            if isinstance(t, Tensor):
                return ...
            else:
                return t

    With this function, you only need to write:

        @map_only(Tensor)
        def go(t):
            return ...

    You can also directly use 'tree_map_only'
    """
    # 根据参数类型判断是否为类型或类型元组
    if isinstance(__type_or_types_or_pred, (type, tuple)) or (
        sys.version_info >= (3, 10)
        and isinstance(__type_or_types_or_pred, types.UnionType)
    ):
        # 如果是类型或类型元组，则定义一个类型检查函数 pred
        def pred(x: Any) -> bool:
            return isinstance(x, __type_or_types_or_pred)  # type: ignore[arg-type]

    elif callable(__type_or_types_or_pred):
        # 如果是可调用对象，则直接将其赋值给 pred
        pred = __type_or_types_or_pred  # type: ignore[assignment]
    else:
        # 如果参数不是类型、类型元组或可调用对象，则抛出类型错误
        raise TypeError("Argument must be a type, a tuple of types, or a callable.")

    # 定义一个装饰器函数 wrapper，接受一个函数 func 并返回一个函数
    def wrapper(func: Callable[[T], Any]) -> Callable[[Any], Any]:
        @functools.wraps(func)
        def wrapped(x: T) -> Any:
            # 如果 x 符合预期的类型条件，则调用 func 函数处理
            if pred(x):
                return func(x)
            # 否则直接返回 x
            return x

        return wrapped

    return wrapper


# 以下是用于处理树结构映射的重载函数声明，用于不同类型的参数
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


# 用于处理其它复杂类型或者可调用对象的重载函数声明
@overload
def tree_map_only(

    __type_or_types_or_pred: TypeAny,
    func: FnAny[Any],
    tree: PyTree,
    is_leaf: Optional[Callable[[PyTree], bool]] = None,
) -> PyTree:
    ...


注释：
这段代码定义了用于树结构映射的重载函数 `tree_map_only`，接受不同类型的参数，并返回映射后的树结构。
    __type_or_types_or_pred: Callable[[Any], bool],
    # __type_or_types_or_pred 是一个函数类型注解，接受一个参数，返回布尔值
    func: FnAny[Any],
    # func 是一个泛型函数，接受任意类型的参数，返回任意类型的结果
    tree: PyTree,
    # tree 是一个变量，类型为 PyTree
    is_leaf: Optional[Callable[[PyTree], bool]] = None,
    # is_leaf 是一个可选的函数类型注解，接受一个 PyTree 参数，返回布尔值，初始值为 None
def tree_map_only(
    __type_or_types_or_pred: Union[TypeAny, Callable[[Any], bool]],
    func: FnAny[Any],
    tree: PyTree,
    is_leaf: Optional[Callable[[PyTree], bool]] = None,
) -> PyTree:
    return tree_map(map_only(__type_or_types_or_pred)(func), tree, is_leaf=is_leaf)



# 将指定类型或预测函数应用于树形结构的每个节点，并应用给定函数，返回新的树形结构
def tree_map_only(
    __type_or_types_or_pred: Union[TypeAny, Callable[[Any], bool]],
    func: FnAny[Any],
    tree: PyTree,
    is_leaf: Optional[Callable[[PyTree], bool]] = None,
) -> PyTree:
    # 使用 map_only 函数根据类型或预测函数对 func 应用到树形结构中的每个节点
    return tree_map(map_only(__type_or_types_or_pred)(func), tree, is_leaf=is_leaf)


@overload
def tree_map_only_(
    __type_or_types_or_pred: Type[T],
    func: Fn[T, Any],
    tree: PyTree,
    is_leaf: Optional[Callable[[PyTree], bool]] = None,
) -> PyTree:
    ...



@overload
# 重载：接受特定类型参数，并应用函数于树的每个节点
def tree_map_only_(
    __type_or_types_or_pred: Type[T],
    func: Fn[T, Any],
    tree: PyTree,
    is_leaf: Optional[Callable[[PyTree], bool]] = None,
) -> PyTree:
    ...



@overload
# 重载：接受两个类型参数，并应用函数于树的每个节点
def tree_map_only_(
    __type_or_types_or_pred: Type2[T, S],
    func: Fn2[T, S, Any],
    tree: PyTree,
    is_leaf: Optional[Callable[[PyTree], bool]] = None,
) -> PyTree:
    ...



@overload
# 重载：接受三个类型参数，并应用函数于树的每个节点
def tree_map_only_(
    __type_or_types_or_pred: Type3[T, S, U],
    func: Fn3[T, S, U, Any],
    tree: PyTree,
    is_leaf: Optional[Callable[[PyTree], bool]] = None,
) -> PyTree:
    ...



@overload
# 重载：接受预测函数参数，并应用函数于树的每个节点
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
    # 使用 map_only_ 函数根据类型或预测函数对 func 应用到树形结构中的每个节点
    return tree_map_(map_only(__type_or_types_or_pred)(func), tree, is_leaf=is_leaf)



def tree_all(
    pred: Callable[[Any], bool],
    tree: PyTree,
    is_leaf: Optional[Callable[[PyTree], bool]] = None,
) -> bool:
    # 展开树结构中的所有参数，应用预测函数，并且所有的结果都为真时返回真
    flat_args = tree_iter(tree, is_leaf=is_leaf)
    return all(map(pred, flat_args))



def tree_any(
    pred: Callable[[Any], bool],
    tree: PyTree,
    is_leaf: Optional[Callable[[PyTree], bool]] = None,
) -> bool:
    # 展开树结构中的所有参数，应用预测函数，并且任意一个结果为真时返回真
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
# 重载：接受特定类型参数，并应用预测函数于树的每个节点，且所有结果为真时返回真
def tree_all_only(
    __type_or_types: Type[T],
    pred: Fn[T, bool],
    tree: PyTree,
    is_leaf: Optional[Callable[[PyTree], bool]] = None,
) -> bool:
    ...



@overload
# 重载：接受两个类型参数，并应用预测函数于树的每个节点，且所有结果为真时返回真
def tree_all_only(
    __type_or_types: Type2[T, S],
    pred: Fn2[T, S, bool],
    tree: PyTree,
    is_leaf: Optional[Callable[[PyTree], bool]] = None,
) -> bool:
    ...



def tree_all_only(
    __type_or_types: TypeAny,
    pred: FnAny[bool],
    tree: PyTree,
    is_leaf: Optional[Callable[[PyTree], bool]] = None,
) -> bool:
    # 展开树结构中的所有参数，应用预测函数，并且对于符合指定类型的参数，所有的结果为真时返回真
    flat_args = tree_iter(tree, is_leaf=is_leaf)
    return all(pred(x) for x in flat_args if isinstance(x, __type_or_types))



@overload
def tree_any_only(
    __type_or_types: Type[T],
    pred: Fn[T, bool],
    tree: PyTree,
    is_leaf: Optional[Callable[[PyTree], bool]] = None,
) -> bool:
    ...



@overload
# 重载：接受特定类型参数，并应用预测函数于树的每个节点，且任意结果为真时返回真
def tree_any_only(
    __type_or_types: Type[T],
    pred: Fn[T, bool],
    tree: PyTree,
    is_leaf: Optional[Callable[[PyTree], bool]] = None,
) -> bool:
    ...



@overload
# 重载：接受两个类型参数，并应用预测函数于树的每个节点，且任意结果为真时返回真
def tree_any_only(
    __type_or_types: Type2[T, S],
    pred: Fn2[T, S, bool],
    tree: PyTree,
    is_leaf: Optional[Callable[[PyTree], bool]] = None,
) -> bool:
    ...



def tree_any_only(
    __type_or_types: TypeAny,
    pred: FnAny[bool],
    tree: PyTree,
    is_leaf: Optional[Callable[[PyTree], bool]] = None,
) -> bool:
    # 展开树结构中的所有参数，应用预测函数，并且对于符合指定类型的参数，任意结果为真时返回真
    flat_args = tree_iter(tree, is_leaf=is_leaf)
    return any(pred(x) for x in flat_args if isinstance(x, __type_or_types))
# 定义一个函数签名，表示返回一个布尔值
) -> bool:
    ...


# @overload 装饰器，指定函数重载的特定情况
@overload
# 定义函数 tree_any_only，接受三种类型参数 T, S, U，以及一个谓词函数 pred
def tree_any_only(
    __type_or_types: Type3[T, S, U],
    pred: Fn3[T, S, U, bool],
    tree: PyTree,
    is_leaf: Optional[Callable[[PyTree], bool]] = None,
) -> bool:
    ...


# 定义函数 tree_any_only，接受任意类型参数 __type_or_types 和一个谓词函数 pred
def tree_any_only(
    __type_or_types: TypeAny,
    pred: FnAny[bool],
    tree: PyTree,
    is_leaf: Optional[Callable[[PyTree], bool]] = None,
) -> bool:
    # 对树进行迭代，获取扁平化后的参数列表
    flat_args = tree_iter(tree, is_leaf=is_leaf)
    # 判断是否存在任意一个满足谓词函数的参数 x，并且 x 是指定类型 __type_or_types 的实例
    return any(pred(x) for x in flat_args if isinstance(x, __type_or_types))


# 定义函数 broadcast_prefix，返回一个列表，包含按照 full_tree 结构广播后的 prefix_tree 叶节点
def broadcast_prefix(
    prefix_tree: PyTree,
    full_tree: PyTree,
    is_leaf: Optional[Callable[[PyTree], bool]] = None,
) -> List[Any]:
    """Return a list of broadcasted leaves in ``prefix_tree`` to match the number of leaves in ``full_tree``.

    If a ``prefix_tree`` is a prefix of a ``full_tree``, this means the ``full_tree`` can be
    constructed by replacing the leaves of ``prefix_tree`` with appropriate **subtrees**.

    This function returns a list of leaves with the same size as ``full_tree``. The leaves are
    replicated from ``prefix_tree``. The number of replicas is determined by the corresponding
    subtree in ``full_tree``.

    >>> broadcast_prefix(1, [1, 2, 3])
    [1, 1, 1]
    >>> broadcast_prefix([1, 2, 3], [1, 2, 3])
    [1, 2, 3]
    >>> broadcast_prefix([1, 2, 3], [1, 2, 3, 4])
    Traceback (most recent call last):
        ...
    ValueError: list arity mismatch; expected: 3, got: 4; list: [1, 2, 3, 4].
    >>> broadcast_prefix([1, 2, 3], [1, 2, (3, 4)])
    [1, 2, 3, 3]
    >>> broadcast_prefix([1, 2, 3], [1, 2, {'a': 3, 'b': 4, 'c': (None, 5)}])
    [1, 2, 3, 3, 3, 3]

    Args:
        prefix_tree (pytree): A pytree with the same structure as a prefix of ``full_tree``.
        full_tree (pytree): A pytree with the same structure as a suffix of ``prefix_tree``.
        is_leaf (callable, optional): An extra leaf predicate function that will be called at each
            flattening step. The function should have a single argument with signature
            ``is_leaf(node) -> bool``. If it returns :data:`True`, the whole subtree being treated
            as a leaf. Otherwise, the default pytree registry will be used to determine a node is a
            leaf or not. If the function is not specified, the default pytree registry will be used.

    Returns:
        A list of leaves in ``prefix_tree`` broadcasted to match the number of leaves in ``full_tree``.
    """
    return optree.broadcast_prefix(
        prefix_tree,
        full_tree,
        is_leaf=is_leaf,
        none_is_leaf=True,
        namespace="torch",
    )
# 将输入的树结构 `tree` 广播到 `inputs` 的树结构，并使用 _broadcast_to_and_flatten 进行检查。
def _broadcast_to_and_flatten(
    tree: PyTree,
    treespec: TreeSpec,
    is_leaf: Optional[Callable[[PyTree], bool]] = None,
) -> Optional[List[Any]]:
    # 断言 `treespec` 是 TreeSpec 类型的实例
    assert isinstance(treespec, TreeSpec)
    # 创建一个完整的树结构 `full_tree`，其形状与 `treespec` 中的叶子数相同
    full_tree = tree_unflatten([0] * treespec.num_leaves, treespec)
    try:
        # 尝试将 `tree` 广播到 `full_tree` 的前缀，同时展平，并可选地检查是否是叶子节点
        return broadcast_prefix(tree, full_tree, is_leaf=is_leaf)
    except ValueError:
        # 如果广播失败，则返回 None
        return None


def treespec_dumps(treespec: TreeSpec, protocol: Optional[int] = None) -> str:
    """将 treespec 序列化为 JSON 字符串。"""
    if not isinstance(treespec, TreeSpec):
        # 如果 `treespec` 不是 TreeSpec 类型的实例，则抛出类型错误
        raise TypeError(
            f"treespec_dumps(spec): Expected `spec` to be instance of "
            f"TreeSpec but got item of type {type(treespec)}."
        )
    from ._pytree import (
        tree_structure as _tree_structure,
        treespec_dumps as _treespec_dumps,
    )

    # 创建原始树结构 `orig_treespec`，通过将 `treespec` 展开成树
    orig_treespec = _tree_structure(tree_unflatten([0] * treespec.num_leaves, treespec))
    # 将 `orig_treespec` 转换成 JSON 字符串，使用指定的协议（可选）
    return _treespec_dumps(orig_treespec, protocol=protocol)


def treespec_loads(serialized: str) -> TreeSpec:
    """从 JSON 字符串反序列化 treespec。"""
    from ._pytree import (
        tree_unflatten as _tree_unflatten,
        treespec_loads as _treespec_loads,
    )

    # 反序列化 JSON 字符串 `serialized`，得到原始的 `orig_treespec`
    orig_treespec = _treespec_loads(serialized)
    # 使用 `orig_treespec` 创建一个虚拟树 `dummy_tree`
    dummy_tree = _tree_unflatten([0] * orig_treespec.num_leaves, orig_treespec)
    # 从 `dummy_tree` 推导出树结构 `treespec`
    treespec = tree_structure(dummy_tree)
    return treespec


class _DummyLeaf:
    def __repr__(self) -> str:
        return "*"  # 返回表示虚拟叶子的字符串 "*"


def treespec_pprint(treespec: TreeSpec) -> str:
    # 使用 `treespec` 创建一个虚拟树 `dummy_tree`
    dummy_tree = tree_unflatten(
        [_DummyLeaf() for _ in range(treespec.num_leaves)],
        treespec,
    )
    # 返回虚拟树的字符串表示形式
    return repr(dummy_tree)


class LeafSpecMeta(type(TreeSpec)):  # type: ignore[misc]
    def __instancecheck__(self, instance: object) -> bool:
        # 检查 `instance` 是否是 TreeSpec 的实例且是叶子节点
        return isinstance(instance, TreeSpec) and instance.is_leaf()


class LeafSpec(TreeSpec, metaclass=LeafSpecMeta):
    def __new__(cls) -> "LeafSpec":
        # 创建 LeafSpec 实例，调用 optree.treespec_leaf 并指定 none_is_leaf=True，忽略返回类型检查
        return optree.treespec_leaf(none_is_leaf=True)  # type: ignore[return-value]


def tree_flatten_with_path(
    tree: PyTree,
    is_leaf: Optional[Callable[[PyTree], bool]] = None,
) -> Tuple[List[Tuple[KeyPath, Any]], TreeSpec]:
    """展平一个类似于 tree_flatten 的 PyTree，同时返回每个叶子节点的键路径。"""
    """
    Args:
        tree: 要展平的 pytree。如果它包含自定义类型，则在注册时必须使用适当的 `tree_flatten_with_path_fn`
              来注册该类型，使用 :func:`register_pytree_node`。
        is_leaf: 一个额外的叶子节点判断函数，在每个展平步骤中调用。该函数应该具有一个参数，签名为
                 ``is_leaf(node) -> bool``。如果返回 :data:`True`，则整个子树将被视为叶子节点。
                 否则，将使用默认的 pytree 注册表来确定节点是否是叶子节点。如果未指定该函数，则将使用
                 默认的 pytree 注册表。
    Returns:
        返回一个元组，第一个元素是 (键路径, 叶子节点) 对的列表，第二个元素是表示展平树结构的 :class:`TreeSpec`。
    """
    raise NotImplementedError("KeyPaths are not yet supported in cxx_pytree.")
# 获取一个 pytree 的叶子节点，并返回每个叶子节点的键路径
def tree_leaves_with_path(
    tree: PyTree,
    is_leaf: Optional[Callable[[PyTree], bool]] = None,
) -> List[Tuple[KeyPath, Any]]:
    """Gets the leaves of a pytree like ``tree_leaves`` and returns each leaf's key path.

    Args:
        tree: a pytree. If it contains a custom type, that type must be
            registered with an appropriate `tree_flatten_with_path_fn` when registered
            with :func:`register_pytree_node`.
        is_leaf: An extra leaf predicate function that will be called at each
            flattening step. The function should have a single argument with signature
            ``is_leaf(node) -> bool``. If it returns :data:`True`, the whole subtree being treated
            as a leaf. Otherwise, the default pytree registry will be used to determine a node is a
            leaf or not. If the function is not specified, the default pytree registry will be used.
    Returns:
        A list of (key path, leaf) pairs.
    """
    raise NotImplementedError("KeyPaths are not yet supported in cxx_pytree.")


# 类似于 tree_map，但提供的可调用函数接受额外的键路径参数
def tree_map_with_path(
    func: Callable[..., Any],
    tree: PyTree,
    *rests: PyTree,
    is_leaf: Optional[Callable[[PyTree], bool]] = None,
) -> PyTree:
    """Like :func:`tree_map`, but the provided callable takes an additional key path argument.

    Args:
        func: A function that takes ``2 + len(rests)`` arguments, to be applied at the
            corresponding leaves of the pytrees. The first positional argument
            to ``func`` is the key path of the leaf in question. The second
            positional argument is the value of the leaf.
        tree: A pytree to be mapped over, with each leaf providing the first positional
            argument to function ``func``.
        rests: A tuple of pytrees, each of which has the same structure as
            ``tree`` or has ``tree`` as a prefix.
        is_leaf: An extra leaf predicate function that will be called at each
            flattening step. The function should have a single argument with signature
            ``is_leaf(node) -> bool``. If it returns :data:`True`, the whole subtree being treated
            as a leaf. Otherwise, the default pytree registry will be used to determine a node is a
            leaf or not. If the function is not specified, the default pytree registry will be used.

    Returns
        A new pytree with the same structure as ``tree`` but with the value at each leaf given by
        ``func(keypath, x, *xs)`` where ``keypath`` is the key path at the
        corresponding leaf in ``tree``, ``x`` is the value at that leaf, and
        ``xs`` is the tuple of values at corresponding nodes in ``rests``.
    """
    raise NotImplementedError("KeyPaths are not yet supported in cxx_pytree.")


# 给定一个键路径，返回一个漂亮打印的表示形式
def keystr(kp: KeyPath) -> str:
    """Given a key path, return a pretty-printed representation."""
    raise NotImplementedError("KeyPaths are not yet supported in cxx_pytree.")


# 从对象中获取指定键路径的值
def key_get(obj: Any, kp: KeyPath) -> Any:
    """
    给定一个对象和一个键路径，返回键路径处的值。
    """
    # 抛出未实现错误，表示在 cxx_pytree 中暂不支持键路径操作
    raise NotImplementedError("KeyPaths are not yet supported in cxx_pytree.")
```