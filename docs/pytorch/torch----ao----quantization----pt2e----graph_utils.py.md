# `.\pytorch\torch\ao\quantization\pt2e\graph_utils.py`

```
# 设置静态类型检查允许未标注的函数定义
# Allow untyped definitions for mypy
# ------------------------------------------------------------
mypy: allow-untyped-defs

# 导入必要的模块和类
# Import necessary modules and classes
# ------------------------------------------------------------
import itertools
from typing import Any, List, OrderedDict, Set, Optional, Callable
import operator
from torch.fx import Node

import torch

# 导入用于源匹配的实用程序函数
# Import utility functions for source matching
# ------------------------------------------------------------
from torch.fx.passes.utils.source_matcher_utils import (
    check_subgraphs_connected,
    get_source_partitions,
    SourcePartition,
)

# 暴露给外部的模块列表
# Expose the following symbols to the outside world
# ------------------------------------------------------------
__all__ = [
    "find_sequential_partitions",
    "get_equivalent_types",
    "update_equivalent_types_dict",
]

# 等价类型的集合列表
# List of sets of equivalent types
# ------------------------------------------------------------
_EQUIVALENT_TYPES: List[Set] = [
    {torch.nn.Conv1d, torch.nn.functional.conv1d},
    {torch.nn.Conv2d, torch.nn.functional.conv2d},
    {torch.nn.AdaptiveAvgPool2d, torch.nn.functional.adaptive_avg_pool2d},
    {torch.nn.ReLU, torch.nn.functional.relu, torch.nn.functional.relu_},
    {torch.nn.BatchNorm2d, torch.nn.functional.batch_norm},
    {torch.nn.Hardtanh, torch.nn.functional.hardtanh, torch.nn.functional.hardtanh_},
    {torch.add, operator.add, operator.iadd, "add", "add_"},
    {torch.mul, operator.mul, operator.imul, "mul", "mul_"},
]

# 创建等价类型的字典
# Create a dictionary of equivalent types
# ------------------------------------------------------------
def _create_equivalent_types_dict():
    _DICT = {}
    for values in _EQUIVALENT_TYPES:
        for v in values:
            _DICT[v] = list(values)
    return _DICT

# 初始化等价类型的字典
# Initialize the dictionary of equivalent types
# ------------------------------------------------------------
_EQUIVALENT_TYPES_DICT = _create_equivalent_types_dict()

# 获取等价类型列表的函数
# Function to retrieve the list of equivalent types
# ------------------------------------------------------------
def get_equivalent_types() -> List[Set]:
    return _EQUIVALENT_TYPES

# 更新等价类型字典的函数，允许用户自定义
# Function to update the equivalent types dictionary, allowing customization by the user
# ------------------------------------------------------------
def update_equivalent_types_dict(customized_equivalent_types=None):
    """Help function for user who wants to customize the _EQUIVALENT_TYPES and _EQUIVALENT_TYPES_DICT.
    When customized_equivalent_types passes in,
    re-generate _EQUIVALENT_TYPES and _EQUIVALENT_TYPES_DICT.
    """
    if customized_equivalent_types is None:
        raise ValueError("customized_equivalent_types should not be None")
    global _EQUIVALENT_TYPES
    global _EQUIVALENT_TYPES_DICT
    _EQUIVALENT_TYPES = customized_equivalent_types
    _EQUIVALENT_TYPES_DICT = _create_equivalent_types_dict()

# 检查分区是否依次连接的函数
# Function to check if partitions are sequentially connected
# ------------------------------------------------------------
def _partitions_sequential(partitions: List[SourcePartition]):
    prev_partition = None
    for partition in partitions:
        if prev_partition is not None and not check_subgraphs_connected(
            prev_partition, partition
        ):
            return False
        prev_partition = partition
    return True

# 获取匹配类型的函数
# Function to retrieve matching types
# ------------------------------------------------------------
def _get_matching_types(partition_type):
    matching_types = [partition_type]
    if partition_type in _EQUIVALENT_TYPES_DICT:
        matching_types.extend(_EQUIVALENT_TYPES_DICT[partition_type])
    return matching_types

# 检查有效类型序列的函数
# Function to check for valid type sequences
# ------------------------------------------------------------
def _valid_type_sequence(partition_types: List[Any]):
    partition_types_set = set()  # type: ignore[var-annotated]
    for partition_type in partition_types:
        matching_types = _get_matching_types(partition_type)
        matching_types_set = set(matching_types)
        if len(partition_types_set & matching_types_set) > 0:
            return False
        partition_types_set |= matching_types_set
    return True

# 查找依次分区的函数，用于图模块
# Function to find sequentially connected partitions within a graph module
# ------------------------------------------------------------
def find_sequential_partitions(
    gm: torch.fx.GraphModule,
    partition_types: List[Any],
    include_functional_equivalent=True,
    # 定义一个可选参数filter_fn，类型为Callable，接受一个Node类型参数并返回布尔值，初始值为None
    # 检查分区类型序列是否有效，如果无效则引发 ValueError 异常
    if not _valid_type_sequence(partition_types):
        raise ValueError(
            f"Invalid partition types: {partition_types}. Each type in the sequence must be unique"
        )

    # 使用有序字典存储按类型分组的源分区列表
    typed_partitions: OrderedDict[Any, List[SourcePartition]] = OrderedDict()

    # 遍历每种分区类型
    for partition_type in partition_types:
        # 获取与当前类型匹配的所有子类型
        types_to_match = _get_matching_types(partition_type)
        # 获取符合指定类型条件的源分区
        partitions = get_source_partitions(gm.graph, types_to_match, filter_fn)
        # 将获取的分区展开并存入按类型分类的有序字典中
        typed_partitions[partition_type] = list(itertools.chain.from_iterable(partitions.values()))

    # 将有序字典中的值转换为列表
    typed_partitions_list = list(typed_partitions.values())
    
    # 生成所有可能的分区组合
    fusion_candidates = itertools.product(*typed_partitions_list)
    
    # 存储符合顺序条件的融合分区列表
    fused_partitions = []
    # 遍历所有分区组合
    for candidate in fusion_candidates:
        # 如果分区组合满足顺序条件，则加入到融合分区列表中
        if _partitions_sequential(candidate):  # type: ignore[arg-type]
            fused_partitions.append(candidate)
    
    # 返回最终的融合分区列表
    return fused_partitions
```