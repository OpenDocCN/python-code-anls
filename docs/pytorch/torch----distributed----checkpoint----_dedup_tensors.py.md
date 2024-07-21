# `.\pytorch\torch\distributed\checkpoint\_dedup_tensors.py`

```py
# Copyright (c) Meta Platforms, Inc. and affiliates
# 导入必要的模块和类
import dataclasses
import logging
from typing import Dict, List, TYPE_CHECKING

# 导入 SavePlan 类用于类型提示
from torch.distributed.checkpoint.planner import SavePlan

# 如果是类型检查环境，导入 MetadataIndex 类
if TYPE_CHECKING:
    from torch.distributed.checkpoint.metadata import MetadataIndex

# 将 dedup_tensors 函数添加到 __all__ 列表中，表示它是公开可用的函数
__all__ = ["dedup_tensors"]

# 初始化日志记录器并返回
def init_logger() -> logging.Logger:
    logger = logging.getLogger(__name__)
    level = logging.INFO
    logger.setLevel(level)
    console = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s %(filename)s:%(lineno)s %(levelname)s p:%(processName)s t:%(threadName)s: %(message)s"
    )
    console.setFormatter(formatter)
    console.setLevel(level)
    logger.addHandler(console)
    logger.propagate = False
    return logger

# 初始化全局日志记录器
logger = init_logger()

# TODO 添加 dedup_tensors 函数的文档字符串
# 从给定的 SavePlan 列表中去重重复的写入项，返回去重后的 SavePlan 列表
def dedup_tensors(all_plans: List[SavePlan]) -> List[SavePlan]:
    # 将输入的 all_plans 转换为列表形式
    all_plans = list(all_plans)

    # 创建一个字典，用于存储索引到计划索引列表的映射关系
    key_to_plan: Dict[MetadataIndex, List[int]] = {}
    for plan_idx, plan in enumerate(all_plans):
        for write_item in plan.items:
            key_to_plan.setdefault(write_item.index, []).append(plan_idx)

    # 从 key_to_plan 字典中筛选出存在重复计划索引的项
    replicated_items = {k: v for k, v in key_to_plan.items() if len(v) > 1}

    # 创建一个字典，用于存储计划索引到重复键列表的映射关系
    plan_to_keys: Dict[int, List[MetadataIndex]] = {}
    for key, plans in replicated_items.items():
        for plan_idx in plans[1:]:
            plan_to_keys.setdefault(plan_idx, []).append(key)

    # 如果 plan_to_keys 中有任何项，则记录日志，指出要删除的重复键
    if len(plan_to_keys) > 0:
        logger.info("Duplicate keys to remove: %s", plan_to_keys)

    # 遍历 plan_to_keys 字典，从 all_plans 中移除重复的键对应的写入项
    for plan_idx, keys in plan_to_keys.items():
        key_set = set(keys)
        # 生成新的写入项列表，不包含需要移除的键
        new_items = [
            write_item
            for write_item in all_plans[plan_idx].items
            if write_item.index not in key_set
        ]
        # 使用 dataclasses.replace 替换原有计划中的写入项，并更新 all_plans
        all_plans[plan_idx] = dataclasses.replace(all_plans[plan_idx], items=new_items)

    # 返回更新后的 all_plans 列表，已经移除了重复的写入项
    return all_plans
```