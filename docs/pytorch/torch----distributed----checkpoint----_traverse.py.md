# `.\pytorch\torch\distributed\checkpoint\_traverse.py`

```
# 版权声明和导入模块声明
from typing import (
    Callable,
    cast,
    Collection,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Tuple,
    TypeVar,
    Union,
)

# 导入 torch 库
import torch
# 导入分布式张量相关模块
from torch.distributed._shard.sharded_tensor.api import ShardedTensor
from torch.distributed._tensor import DTensor
from torch.distributed.checkpoint.metadata import STATE_DICT_TYPE

# 定义路径项的类型
PATH_ITEM = Union[str, int]
# 定义对象路径的类型
OBJ_PATH = Tuple[PATH_ITEM, ...]
# 定义类型变量 T
T = TypeVar("T")

# 定义状态字典项的通用类型
STATE_DICT_ITEM = object
# 定义容器类型为可变映射，键为路径项，值为状态字典项
CONTAINER_TYPE = MutableMapping[PATH_ITEM, STATE_DICT_ITEM]

# 定义模块中公开的函数和类
__all__ = ["traverse_state_dict", "set_element", "get_element", "print_tensor"]


# 定义函数，判断状态字典项是否为张量
def _keep_visiting_tensors(value: STATE_DICT_ITEM) -> bool:
    return isinstance(value, torch.Tensor)


# TODO: update docstring for traverse.py
# 遍历状态字典并调用访问者函数
def traverse_state_dict(
    state_dict: STATE_DICT_TYPE,
    visitor: Callable[[OBJ_PATH, STATE_DICT_ITEM], None],
    keep_traversing: Callable[[STATE_DICT_ITEM], bool] = _keep_visiting_tensors,
) -> None:
    """
    递归地对状态字典中的每个值调用访问者函数“visitor”。
    映射、列表和元组将被展平，其他值类型被视为终端值，并将调用“visitor”。
    映射被视为非终端节点并将被展平。
    列表和元组不会被展平，除非包含其他映射容器或张量。
    """

    # 判断值是否为终端节点
    def _is_terminal(value: STATE_DICT_ITEM) -> bool:
        values: Collection[STATE_DICT_ITEM]
        if isinstance(value, Mapping):
            return False
        elif isinstance(value, list):
            values = value
        else:
            return True

        for entry in values:
            if isinstance(entry, (Mapping, list)) and not _is_terminal(entry):
                return False
            if keep_traversing is not None and keep_traversing(entry):
                return False
        return True

    # 递归遍历对象
    def _traverse_obj(path: OBJ_PATH, value: STATE_DICT_ITEM) -> None:
        if isinstance(value, Mapping):
            for k, v in value.items():
                _traverse_obj(path + (str(k),), v)
        elif _is_terminal(value):
            visitor(path, value)
        elif isinstance(value, (list, tuple)):
            for i, v in enumerate(value):
                _traverse_obj(path + (i,), v)

    # 遍历状态字典中的每个键值对，并递归调用_traverse_obj函数
    for key, value in state_dict.items():
        _traverse_obj((str(key),), value)


# 设置状态字典中指定路径的元素值
def set_element(
    root_dict: STATE_DICT_TYPE, path: OBJ_PATH, value: STATE_DICT_ITEM
) -> None:
    """Set ``value`` in ``root_dict`` along the ``path`` object path."""
    cur_container = cast(CONTAINER_TYPE, root_dict)

    # 定义函数，用于扩展列表并设置元素值
    def extend_list(lst: List[STATE_DICT_ITEM], idx: int) -> None:
        while len(lst) <= idx:
            lst.append(None)
    # 遍历路径中的每个索引，从第二个元素开始
    for i in range(1, len(path)):
        # 获取前一个键名
        prev_key = path[i - 1]
        # 获取当前键名
        key = path[i]
        # 根据键名类型选择默认值，转换为 STATE_DICT_ITEM 类型的字典或列表
        def_val = cast(STATE_DICT_ITEM, {} if type(key) == str else [])

        # 如果当前容器是映射类型（字典）
        if isinstance(cur_container, Mapping):
            # 设置默认值并更新当前容器
            cur_container = cast(CONTAINER_TYPE, cur_container.setdefault(prev_key, def_val))
        else:
            # 如果当前容器不是映射类型（可能是列表），扩展列表
            extend_list(cur_container, prev_key)
            # 如果前一个键在当前容器中为 None，则设置为默认值
            if cur_container[prev_key] is None:
                cur_container[prev_key] = def_val
            # 更新当前容器为前一个键所指向的对象

            cur_container = cur_container[prev_key]

    # 获取路径中的最后一个键
    key = path[-1]
    # 如果最后一个键是整数类型
    if type(key) == int:
        # 将当前容器强制转换为 STATE_DICT_ITEM 类型的列表，并扩展列表
        extend_list(cast(List[STATE_DICT_ITEM], cur_container), key)

    # 将值赋给当前容器中的指定键
    cur_container[key] = value
# 从给定的根字典中获取指定路径处的元素值，如果路径不存在则返回默认值
def get_element(
    root_dict: STATE_DICT_TYPE,  # 根字典，包含要获取值的数据结构
    path: OBJ_PATH,  # 要获取值的路径，可以是字符串或整数列表
    default_value: Optional[T] = None,  # 默认返回值，如果路径不存在时返回
) -> Optional[T]:
    """Retrieve the value at ``path``from ``root_dict``, returning ``default_value`` if not found."""
    cur_value = cast(CONTAINER_TYPE, root_dict)  # 当前处理的值，从根字典开始
    for part in path:  # 遍历路径中的各个部分
        if type(part) is int:  # 如果路径部分是整数
            if not isinstance(cur_value, list) or len(cur_value) < part:  # 如果当前值不是列表或者列表长度小于指定的整数
                return default_value  # 返回默认值
        elif not isinstance(cur_value, Mapping) or part not in cur_value:  # 如果当前值不是映射类型或者路径部分不在映射中
            return default_value  # 返回默认值

        cur_value = cast(CONTAINER_TYPE, cur_value[part])  # 更新当前值为路径部分对应的值
    return cast(Optional[T], cur_value)  # 返回找到的值或默认值


def _print_nested(
    value: STATE_DICT_ITEM,  # 要打印的值，可能是各种类型的数据
    prefix: str = "",  # 打印前缀，用于标识该值在数据结构中的位置
    print_fun: Callable[[str], None] = print,  # 打印函数，默认为内置的 print 函数
) -> None:
    if type(value) is ShardedTensor:  # 如果值的类型是 ShardedTensor
        print_fun(f"{prefix} ShardedTensor size: {value.size()}")  # 打印 ShardedTensor 的大小信息
        for shard in value.local_shards():  # 遍历 ShardedTensor 的本地分片
            _print_nested(
                shard.tensor,  # 递归打印分片的张量数据
                f"{shard.metadata.shard_offsets} ",  # 更新打印前缀，包含分片的偏移信息
                print_fun=print_fun,  # 传递打印函数
            )
    elif type(value) is (DTensor):  # 如果值的类型是 DTensor
        print_fun(f"{prefix} DistributedTensor size: {value.size()}")  # 打印 DistributedTensor 的大小信息
        # TODO: add local offset for _local_tensor in print_nested.
        _print_nested(
            value._local_tensor,  # 递归打印分布式张量的本地张量数据
            print_fun=print_fun,  # 传递打印函数
        )
    elif isinstance(value, torch.Tensor):  # 如果值是 torch.Tensor 类型
        print_fun(f"{prefix} Tensor size: {value.size()}")  # 打印 Tensor 的大小信息
    else:
        print_fun(f"{prefix} Type: {type(value)}")  # 打印值的类型信息


def print_tensor(
    path: OBJ_PATH,  # 打印值的路径
    value: STATE_DICT_ITEM,  # 要打印的值，可能是复杂数据结构
    print_fun: Callable[[str], None] = print,  # 打印函数，默认为内置的 print 函数
) -> None:
    """
    Use this callback with traverse_state_dict to print its content.

    By default the content is printed using the builtin ``print`` but this can
    be change by passing a different ``print_fun` callable.
    """
    _print_nested(value, prefix=str(path), print_fun=print_fun)  # 调用 _print_nested 函数打印值和路径信息
```