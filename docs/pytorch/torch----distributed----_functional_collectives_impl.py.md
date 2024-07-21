# `.\pytorch\torch\distributed\_functional_collectives_impl.py`

```py
# mypy: allow-untyped-defs
# 引入必要的类型声明
from typing import List, Optional

# 引入 PyTorch 库
import torch
# 引入 PyTorch 分布式通信模块
import torch.distributed.distributed_c10d as c10d

"""
This file contains the op impls for the legacy (c10d_functional) functional collectives.
These impls simply call into the native (_c10d_functional) functional collectives.
"""

# 实现广播操作
def _broadcast(input, src, tag, ranks, group_size):
    # 通过 ranks 和 tag 解析分组名称
    group_name = c10d._resolve_group_name_by_ranks_and_tag(ranks, tag)
    # 调用底层的广播操作
    return torch.ops._c10d_functional.broadcast(
        input,
        src,
        group_name,
    )

# 实现全局归约操作
def _all_reduce(input, reduce_op, tag, ranks, group_size):
    # 通过 ranks 和 tag 解析分组名称
    group_name = c10d._resolve_group_name_by_ranks_and_tag(ranks, tag)
    # 调用底层的全局归约操作
    return torch.ops._c10d_functional.all_reduce(
        input,
        reduce_op,
        group_name,
    )

# 实现全局归约合并操作
def _all_reduce_coalesced(inputs, reduce_op, tag, ranks, group_size):
    # 通过 ranks 和 tag 解析分组名称
    group_name = c10d._resolve_group_name_by_ranks_and_tag(ranks, tag)
    # 调用底层的全局归约合并操作
    return torch.ops._c10d_functional.all_reduce_coalesced(
        inputs,
        reduce_op,
        group_name,
    )

# 实现全局收集进张量操作
def _all_gather_into_tensor(input, tag, ranks, group_size):
    # 通过 ranks 和 tag 解析分组名称
    group_name = c10d._resolve_group_name_by_ranks_and_tag(ranks, tag)
    # 调用底层的全局收集进张量操作
    return torch.ops._c10d_functional.all_gather_into_tensor(
        input,
        group_size,
        group_name,
    )

# 实现全局收集进张量合并操作
def _all_gather_into_tensor_coalesced(input, tag, ranks, group_size):
    # 通过 ranks 和 tag 解析分组名称
    group_name = c10d._resolve_group_name_by_ranks_and_tag(ranks, tag)
    # 调用底层的全局收集进张量合并操作
    return torch.ops._c10d_functional.all_gather_into_tensor_coalesced(
        input,
        group_size,
        group_name,
    )

# 实现张量按分组归约散开操作
def _reduce_scatter_tensor(
    input: torch.Tensor,
    reduce_op: str,
    tag: str,
    ranks: List[int],
    group_size: int,
):
    # 通过 ranks 和 tag 解析分组名称
    group_name = c10d._resolve_group_name_by_ranks_and_tag(ranks, tag)
    # 调用底层的张量按分组归约散开操作
    return torch.ops._c10d_functional.reduce_scatter_tensor(
        input,
        reduce_op,
        group_size,
        group_name,
    )

# 实现张量按分组归约散开合并操作
def _reduce_scatter_tensor_coalesced(
    inputs: List[torch.Tensor],
    reduce_op: str,
    tag: str,
    ranks: List[int],
    group_size: int,
):
    # 通过 ranks 和 tag 解析分组名称
    group_name = c10d._resolve_group_name_by_ranks_and_tag(ranks, tag)
    # 调用底层的张量按分组归约散开合并操作
    return torch.ops._c10d_functional.reduce_scatter_tensor_coalesced(
        inputs,
        reduce_op,
        group_size,
        group_name,
    )

# 实现单一全局全对全操作
def _all_to_all_single(
    input: torch.Tensor,
    output_split_sizes: Optional[List[int]],
    input_split_sizes: Optional[List[int]],
    tag: str,
    ranks: List[int],
    group_size: int,
):
    # 如果未提供输出和输入分割大小，则默认相等分割
    if output_split_sizes is None or input_split_sizes is None:
        assert output_split_sizes is None and input_split_sizes is None, (
            "output_split_sizes and input_split_sizes must either be "
            "specified together or both set to None"
        )
        output_split_sizes = [input.shape[0] // group_size] * group_size
        input_split_sizes = output_split_sizes

    # 通过 ranks 和 tag 解析分组名称
    group_name = c10d._resolve_group_name_by_ranks_and_tag(ranks, tag)
    # 调用 Torch 库中的 C10D 模块的函数 all_to_all_single，用于执行分布式通信操作
    return torch.ops._c10d_functional.all_to_all_single(
        # 输入数据，即要发送或接收的张量
        input,
        # 输出数据的分割大小列表，指定每个分组成员接收的数据大小
        output_split_sizes,
        # 输入数据的分割大小列表，指定每个分组成员发送的数据大小
        input_split_sizes,
        # 分组的名称，指定参与通信的组成员
        group_name,
    )
# 定义一个函数 `_wait_tensor`，接收一个参数 `tensor`，类型为 `torch.Tensor`，返回类型也为 `torch.Tensor`
def _wait_tensor(tensor: torch.Tensor) -> torch.Tensor:
    # 调用 Torch C++ 扩展中的 `_c10d_functional.wait_tensor` 函数来等待给定的张量 `tensor` 完成
    return torch.ops._c10d_functional.wait_tensor(tensor)
```