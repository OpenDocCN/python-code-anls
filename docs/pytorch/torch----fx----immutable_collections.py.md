# `.\pytorch\torch\fx\immutable_collections.py`

```
# mypy: allow-untyped-defs
# 导入必要的类型定义
from typing import Any, Dict, Iterable, List, Tuple

# 导入需要的函数和类
from torch.utils._pytree import (
    _dict_flatten,
    _dict_flatten_with_keys,
    _dict_unflatten,
    _list_flatten,
    _list_flatten_with_keys,
    _list_unflatten,
    Context,
    register_pytree_node,
)

# 导入兼容性函数
from ._compatibility import compatibility

# 将这些类和函数添加到模块的公共接口中
__all__ = ["immutable_list", "immutable_dict"]

# 关于不可变容器对象的变异说明
_help_mutation = """\
如果您试图修改 torch.fx.Node 对象的 kwargs 或 args，
请创建其副本，并将副本赋给该节点：
    new_args = ... # 复制和修改 args
    node.args = new_args
"""

# 抛出未实现错误，阻止对象的任何变异操作
def _no_mutation(self, *args, **kwargs):
    raise NotImplementedError(
        f"'{type(self).__name__}' object does not support mutation. {_help_mutation}",
    )

# 创建不可变容器的工厂函数
def _create_immutable_container(base, mutable_functions):
    container = type("immutable_" + base.__name__, (base,), {})
    for attr in mutable_functions:
        setattr(container, attr, _no_mutation)
    return container

# 创建不可变列表类型
immutable_list = _create_immutable_container(
    list,
    (
        "__delitem__",
        "__iadd__",
        "__imul__",
        "__setitem__",
        "append",
        "clear",
        "extend",
        "insert",
        "pop",
        "remove",
        "reverse",
        "sort",
    ),
)
# 重定义序列化和哈希函数，以便在不可变列表上使用
immutable_list.__reduce__ = lambda self: (immutable_list, (tuple(iter(self)),))
immutable_list.__hash__ = lambda self: hash(tuple(self))

# 将兼容性函数应用于不可变列表
compatibility(is_backward_compatible=True)(immutable_list)

# 创建不可变字典类型
immutable_dict = _create_immutable_container(
    dict,
    (
        "__delitem__",
        "__ior__",
        "__setitem__",
        "clear",
        "pop",
        "popitem",
        "setdefault",
        "update",
    ),
)
# 重定义序列化和哈希函数，以便在不可变字典上使用
immutable_dict.__reduce__ = lambda self: (immutable_dict, (iter(self.items()),))
immutable_dict.__hash__ = lambda self: hash(tuple(self.items()))

# 将兼容性函数应用于不可变字典
compatibility(is_backward_compatible=True)(immutable_dict)

# 注册不可变集合用于 PyTree 操作
# 定义字典的扁平化和还原函数
def _immutable_dict_flatten(d: Dict[Any, Any]) -> Tuple[List[Any], Context]:
    return _dict_flatten(d)

def _immutable_dict_unflatten(
    values: Iterable[Any],
    context: Context,
) -> Dict[Any, Any]:
    return immutable_dict(_dict_unflatten(values, context))

# 定义列表的扁平化和还原函数
def _immutable_list_flatten(d: List[Any]) -> Tuple[List[Any], Context]:
    return _list_flatten(d)

def _immutable_list_unflatten(
    values: Iterable[Any],
    context: Context,
) -> List[Any]:
    return immutable_list(_list_unflatten(values, context))

# 注册不可变字典类型的 PyTree 节点
register_pytree_node(
    immutable_dict,
    _immutable_dict_flatten,
    _immutable_dict_unflatten,
    serialized_type_name="torch.fx.immutable_collections.immutable_dict",
    flatten_with_keys_fn=_dict_flatten_with_keys,
)

# 注册不可变列表类型的 PyTree 节点
register_pytree_node(
    immutable_list,
    _immutable_list_flatten,
    _immutable_list_unflatten,
    serialized_type_name="torch.fx.immutable_collections.immutable_list",
    flatten_with_keys_fn=_list_flatten_with_keys,
)
```