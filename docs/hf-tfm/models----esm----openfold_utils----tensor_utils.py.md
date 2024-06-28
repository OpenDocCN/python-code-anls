# `.\models\esm\openfold_utils\tensor_utils.py`

```
# 导入 functools 模块中的 partial 函数
from functools import partial
# 导入 typing 模块中的各种类型提示
from typing import Any, Callable, Dict, List, Type, TypeVar, Union, overload

# 导入 PyTorch 库
import torch
# 导入 torch.nn 模块
import torch.nn as nn
# 导入 torch.types 模块
import torch.types

# 定义一个函数 add，接受两个 torch.Tensor 类型的参数 m1 和 m2，以及一个布尔类型的 inplace 参数，返回一个 torch.Tensor
def add(m1: torch.Tensor, m2: torch.Tensor, inplace: bool) -> torch.Tensor:
    # 如果 inplace 参数为 False，则进行非就地操作
    # 第一个操作不能是就地操作，但在推理期间执行就地加法会更好。因此...
    if not inplace:
        m1 = m1 + m2  # 非就地加法
    else:
        m1 += m2  # 就地加法

    return m1  # 返回结果 m1

# 定义一个函数 permute_final_dims，接受一个 torch.Tensor 类型的 tensor 参数和一个 List[int] 类型的 inds 参数，返回一个 torch.Tensor
def permute_final_dims(tensor: torch.Tensor, inds: List[int]) -> torch.Tensor:
    zero_index = -1 * len(inds)
    first_inds = list(range(len(tensor.shape[:zero_index])))
    return tensor.permute(first_inds + [zero_index + i for i in inds])

# 定义一个函数 flatten_final_dims，接受一个 torch.Tensor 类型的 t 参数和一个整数类型的 no_dims 参数，返回一个 torch.Tensor
def flatten_final_dims(t: torch.Tensor, no_dims: int) -> torch.Tensor:
    return t.reshape(t.shape[:-no_dims] + (-1,))

# 定义一个函数 masked_mean，接受一个 torch.Tensor 类型的 mask 参数，一个 torch.Tensor 类型的 value 参数，一个整数类型的 dim 参数，以及一个浮点数类型的 eps 参数，默认值为 1e-4，返回一个 torch.Tensor
def masked_mean(mask: torch.Tensor, value: torch.Tensor, dim: int, eps: float = 1e-4) -> torch.Tensor:
    mask = mask.expand(*value.shape)
    return torch.sum(mask * value, dim=dim) / (eps + torch.sum(mask, dim=dim))

# 定义一个函数 pts_to_distogram，接受一个 torch.Tensor 类型的 pts 参数，以及三个可选参数 min_bin、max_bin 和 no_bins，都是 torch.types.Number 类型，默认值分别为 2.3125、21.6875 和 64，返回一个 torch.Tensor
def pts_to_distogram(
    pts: torch.Tensor, min_bin: torch.types.Number = 2.3125, max_bin: torch.types.Number = 21.6875, no_bins: int = 64
) -> torch.Tensor:
    boundaries = torch.linspace(min_bin, max_bin, no_bins - 1, device=pts.device)
    dists = torch.sqrt(torch.sum((pts.unsqueeze(-2) - pts.unsqueeze(-3)) ** 2, dim=-1))
    return torch.bucketize(dists, boundaries)

# 定义一个函数 dict_multimap，接受一个 Callable[[list], Any] 类型的 fn 参数和一个 List[dict] 类型的 dicts 参数，返回一个 dict
def dict_multimap(fn: Callable[[list], Any], dicts: List[dict]) -> dict:
    first = dicts[0]
    new_dict = {}
    for k, v in first.items():
        all_v = [d[k] for d in dicts]
        if isinstance(v, dict):
            new_dict[k] = dict_multimap(fn, all_v)
        else:
            new_dict[k] = fn(all_v)

    return new_dict

# 定义一个函数 one_hot，接受一个 torch.Tensor 类型的 x 参数和一个 torch.Tensor 类型的 v_bins 参数，返回一个 torch.Tensor
def one_hot(x: torch.Tensor, v_bins: torch.Tensor) -> torch.Tensor:
    reshaped_bins = v_bins.view(((1,) * len(x.shape)) + (len(v_bins),))
    diffs = x[..., None] - reshaped_bins
    am = torch.argmin(torch.abs(diffs), dim=-1)
    return nn.functional.one_hot(am, num_classes=len(v_bins)).float()

# 定义一个函数 batched_gather，接受一个 torch.Tensor 类型的 data 参数，一个 torch.Tensor 类型的 inds 参数，以及两个可选参数 dim 和 no_batch_dims，都是整数类型，默认值分别为 0 和 0，返回一个 torch.Tensor
def batched_gather(data: torch.Tensor, inds: torch.Tensor, dim: int = 0, no_batch_dims: int = 0) -> torch.Tensor:
    ranges: List[Union[slice, torch.Tensor]] = []
    # 遍历数据的形状的前几个维度（不包括批量维度）
    for i, s in enumerate(data.shape[:no_batch_dims]):
        # 创建一个包含从0到s-1的整数的张量
        r = torch.arange(s)
        # 根据当前维度i的索引，以及数据索引的形状，重新视图化张量r
        r = r.view(*(*((1,) * i), -1, *((1,) * (len(inds.shape) - i - 1))))
        # 将r添加到ranges列表中
        ranges.append(r)

    # 创建一个包含slice或者张量的列表，用于处理剩余的维度
    remaining_dims: List[Union[slice, torch.Tensor]] = [slice(None) for _ in range(len(data.shape) - no_batch_dims)]
    # 将inds插入到对应的维度位置中，处理维度偏移问题
    remaining_dims[dim - no_batch_dims if dim >= 0 else dim] = inds
    # 将剩余的维度信息添加到ranges列表中
    ranges.extend(remaining_dims)

    # 返回根据ranges索引得到的数据
    # Matt 注意：修改此处以避免在最近的Numpy版本中使用列表作为数组索引的行为变化
    return data[tuple(ranges)]
# 使用 TypeVar 创建一个泛型变量 T，用于表示函数的参数类型
T = TypeVar("T")

# 定义 dict_map 函数，用于对字典及其嵌套结构中的各种类型进行映射操作
def dict_map(
    fn: Callable[[T], Any], dic: Dict[Any, Union[dict, list, tuple, T]], leaf_type: Type[T]
) -> Dict[Any, Union[dict, list, tuple, Any]]:
    # 创建一个新的空字典，用于存储映射后的结果
    new_dict: Dict[Any, Union[dict, list, tuple, Any]] = {}
    # 遍历输入的字典 dic 的键值对
    for k, v in dic.items():
        # 如果值 v 是字典类型，则递归调用 dict_map 对其进行映射
        if isinstance(v, dict):
            new_dict[k] = dict_map(fn, v, leaf_type)
        # 否则，调用 tree_map 函数对 v 进行映射（tree_map 函数在后面定义）
        else:
            new_dict[k] = tree_map(fn, v, leaf_type)

    return new_dict


# tree_map 函数的重载定义：处理输入 tree 为单个元素的情况
@overload
def tree_map(fn: Callable[[T], Any], tree: T, leaf_type: Type[T]) -> Any:
    ...


# tree_map 函数的重载定义：处理输入 tree 为字典的情况
@overload
def tree_map(fn: Callable[[T], Any], tree: dict, leaf_type: Type[T]) -> dict:
    ...


# tree_map 函数的重载定义：处理输入 tree 为列表的情况
@overload
def tree_map(fn: Callable[[T], Any], tree: list, leaf_type: Type[T]) -> list:
    ...


# tree_map 函数的重载定义：处理输入 tree 为元组的情况
@overload
def tree_map(fn: Callable[[T], Any], tree: tuple, leaf_type: Type[T]) -> tuple:
    ...


# 定义 tree_map 函数，用于对树状数据结构 tree 进行映射操作
def tree_map(fn, tree, leaf_type):
    # 如果 tree 是字典类型，则调用 dict_map 对其进行映射
    if isinstance(tree, dict):
        return dict_map(fn, tree, leaf_type)
    # 如果 tree 是列表类型，则递归调用 tree_map 对其内部每个元素进行映射
    elif isinstance(tree, list):
        return [tree_map(fn, x, leaf_type) for x in tree]
    # 如果 tree 是元组类型，则递归调用 tree_map 对其内部每个元素进行映射，并返回元组
    elif isinstance(tree, tuple):
        return tuple(tree_map(fn, x, leaf_type) for x in tree)
    # 如果 tree 是 leaf_type 类型，则直接调用 fn 对其进行映射
    elif isinstance(tree, leaf_type):
        return fn(tree)
    # 如果 tree 不属于以上任何类型，则抛出 ValueError 异常
    else:
        print(type(tree))
        raise ValueError("Not supported")


# 使用 partial 函数创建 tensor_tree_map 函数，固定 leaf_type 参数为 torch.Tensor 类型
tensor_tree_map = partial(tree_map, leaf_type=torch.Tensor)
```