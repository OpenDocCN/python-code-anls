# `.\models\esm\openfold_utils\tensor_utils.py`

```py
# 引入 functools 中的 partial 函数，用于部分应用函数
# 引入 Any, Callable, Dict, List, Type, TypeVar, Union, overload 类型提示
# 引入 torch 和 torch 中的 nn、types 模块
from functools import partial
from typing import Any, Callable, Dict, List, Type, TypeVar, Union, overload

import torch
import torch.nn as nn
import torch.types

# 定义函数 add，接受两个 torch.Tensor 类型的参数 m1 和 m2，以及一个布尔类型的参数 inplace，返回一个 torch.Tensor 类型的结果
def add(m1: torch.Tensor, m2: torch.Tensor, inplace: bool) -> torch.Tensor:
    # 如果不是原地操作，则 m1 和 m2 相加后赋值给 m1
    if not inplace:
        m1 = m1 + m2
    # 如果是原地操作，则直接在 m1 上进行加法操作
    else:
        m1 += m2

    return m1

# 定义函数 permute_final_dims，接受一个 torch.Tensor 类型的参数 tensor 和一个整数列表 inds，返回一个 torch.Tensor 类型的结果
def permute_final_dims(tensor: torch.Tensor, inds: List[int]) -> torch.Tensor:
    # 计算零维度索引
    zero_index = -1 * len(inds)
    first_inds = list(range(len(tensor.shape[:zero_index]))
    # 对 tensor 进行维度重排
    return tensor.permute(first_inds + [zero_index + i for i in inds])

# 定义函数 flatten_final_dims，接受一个 torch.Tensor 类型的参数 t 和一个整数值 no_dims，返回一个 torch.Tensor 类型的结果
def flatten_final_dims(t: torch.Tensor, no_dims: int) -> torch.Tensor:
    # 对 tensor 进行形状重塑
    return t.reshape(t.shape[:-no_dims] + (-1,))

# 定义函数 masked_mean，接受一个 torch.Tensor 类型的参数 mask、value 和一个整数值 dim，以及一个浮点数 eps，默认值为 1e-4，返回一个 torch.Tensor 类型的结果
def masked_mean(mask: torch.Tensor, value: torch.Tensor, dim: int, eps: float = 1e-4) -> torch.Tensor:
    mask = mask.expand(*value.shape)
    # 计算加权平均值
    return torch.sum(mask * value, dim=dim) / (eps + torch.sum(mask, dim=dim))

# 定义函数 pts_to_distogram，接受一个 torch.Tensor 类型的参数 pts 和三个 torch.types.Number 类型的参数 min_bin、max_bin、no_bins（默认值为 64），返回一个 torch.Tensor 类型的结果
def pts_to_distogram(pts: torch.Tensor, min_bin: torch.types.Number = 2.3125, max_bin: torch.types.Number = 21.6875, no_bins: int = 64) -> torch.Tensor:
    # 在指定范围内生成分界点
    boundaries = torch.linspace(min_bin, max_bin, no_bins - 1, device=pts.device)
    # 计算距离并进行分桶
    dists = torch.sqrt(torch.sum((pts.unsqueeze(-2) - pts.unsqueeze(-3)) ** 2, dim=-1))
    return torch.bucketize(dists, boundaries)

# 定义函数 dict_multimap，接受一个 Callable 类型的参数 fn 和一个字典列表 dicts，返回一个字典类��的结果
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

# 定义函数 one_hot，接受一个 torch.Tensor 类型的参数 x 和 v_bins，返回一个 torch.Tensor 类型的结果
def one_hot(x: torch.Tensor, v_bins: torch.Tensor) -> torch.Tensor:
    reshaped_bins = v_bins.view(((1,) * len(x.shape)) + (len(v_bins),))
    diffs = x[..., None] - reshaped_bins
    am = torch.argmin(torch.abs(diffs), dim=-1)
    return nn.functional.one_hot(am, num_classes=len(v_bins)).float()

# 定义函数 batched_gather，接受一个 torch.Tensor 类型的参数 data 和 inds，还有两个整数参数 dim 和 no_batch_dims（默认值为 0），返回一个 torch.Tensor 类型的结果
def batched_gather(data: torch.Tensor, inds: torch.Tensor, dim: int = 0, no_batch_dims: int = 0) -> torch.Tensor:
    ranges: List[Union[slice, torch.Tensor]] = []
    # 遍历数据形状的前 no_batch_dims 维度的索引和大小
    for i, s in enumerate(data.shape[:no_batch_dims]):
        # 创建从 0 到 s-1 的索引张量
        r = torch.arange(s)
        # 重新调整张量形状，使得在第 i 维度的大小变为-1
        r = r.view(*(*((1,) * i), -1, *((1,) * (len(inds.shape) - i - 1))))
        # 将重新调整形状后的张量添加到 ranges 列表中
        ranges.append(r)

    # 初始化剩余维度的列表为全切片
    remaining_dims: List[Union[slice, torch.Tensor]] = [slice(None) for _ in range(len(data.shape) - no_batch_dims)]
    # 将 inds 替换到正确的维度位置
    remaining_dims[dim - no_batch_dims if dim >= 0 else dim] = inds
    # 将剩余维度列表添加到 ranges 列表中
    ranges.extend(remaining_dims)
    # Matt 注意：编辑这部分以解决在最近的 Numpy 版本中使用列表作为数组索引会发生变化的行为
    # 使用 ranges 列表作为索引来获取数据，并返回结果
    return data[tuple(ranges)]
# 导入必要的类型
T = TypeVar("T")

# 定义一个字典映射函数，输入一个函数、一个包含字典的字典、和叶子的类型，输出一个经过函数映射的新字典
def dict_map(
    fn: Callable[[T], Any], dic: Dict[Any, Union[dict, list, tuple, T]], leaf_type: Type[T]
) -> Dict[Any, Union[dict, list, tuple, Any]]:
    # 创建一个空字典
    new_dict: Dict[Any, Union[dict, list, tuple, Any]] = {}
    # 遍历输入字典的键值对
    for k, v in dic.items():
        # 如果值是字典类型，则递归调用字典映射函数，进行深度优先的映射操作
        if isinstance(v, dict):
            new_dict[k] = dict_map(fn, v, leaf_type)
        # 如果值不是字典类型，则调用 tree_map 函数进行映射
        else:
            new_dict[k] = tree_map(fn, v, leaf_type)

    return new_dict

# 重载，定义函数 tree_map 的不同形式重载
@overload
def tree_map(fn: Callable[[T], Any], tree: T, leaf_type: Type[T]) -> Any:
    ...


@overload
def tree_map(fn: Callable[[T], Any], tree: dict, leaf_type: Type[T]) -> dict:
    ...


@overload
def tree_map(fn: Callable[[T], Any], tree: list, leaf_type: Type[T]) -> list:
    ...


@overload
def tree_map(fn: Callable[[T], Any], tree: tuple, leaf_type: Type[T]) -> tuple:
    ...


# 定义一个树映射函数，接收一个函数、一个树、和叶子的类型作为参数
def tree_map(fn, tree, leaf_type):
    # 如果输入的树是字典类型，则调用 dict_map 函数进行映射
    if isinstance(tree, dict):
        return dict_map(fn, tree, leaf_type)
    # 如果输入的树是列表类型，则对每个列表元素调用 tree_map 函数进行映射
    elif isinstance(tree, list):
        return [tree_map(fn, x, leaf_type) for x in tree]
    # 如果输入的树是元组类型，则对每个元组元素调用 tree_map 函数进行映射
    elif isinstance(tree, tuple):
        return tuple(tree_map(fn, x, leaf_type) for x in tree)
    # 如果输入的树是指定的叶子类型，则对其调用映射函数
    elif isinstance(tree, leaf_type):
        return fn(tree)
    else:  # 如果输入的树是其他类型，则抛出异常
        print(type(tree))
        raise ValueError("Not supported")

# 定义一个 tensor_tree_map 函数，使用 functools.partial 对 tree_map 进行部分应用，固定叶子类型为 torch.Tensor
tensor_tree_map = partial(tree_map, leaf_type=torch.Tensor)
```