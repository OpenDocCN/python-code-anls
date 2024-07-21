# `.\pytorch\torch\distributed\checkpoint\_dedup_save_plans.py`

```py
# 著作权声明，由 Meta Platforms, Inc. 及其关联公司所有
import dataclasses  # 导入 dataclasses 模块，用于创建和操作数据类
from collections import defaultdict  # 导入 defaultdict，用于创建默认值为集合的字典
from typing import Dict, List, Set, TYPE_CHECKING  # 导入类型提示相关的类和集合

from torch.distributed.checkpoint.planner import SavePlan, WriteItem  # 导入 SavePlan 和 WriteItem 类


if TYPE_CHECKING:
    from torch.distributed.checkpoint.metadata import MetadataIndex  # 条件导入，仅在类型检查时导入 MetadataIndex 类

__all__ = ["dedup_save_plans"]  # 模块中公开的函数和类名列表


def dedup_save_plans(
    all_plans: List[SavePlan],  # 输入参数：SavePlan 对象的列表
    save_to_lowest_rank: bool = False,  # 是否保存到最低等级的标志，默认为 False
) -> List[SavePlan]:
    """
    从多个 SavePlan 中删除重复条目。对于每个重复出现在多个 SavePlan 中的条目，
    只保留计划存储空间最小的 SavePlan 中的条目。
    """

    write_item_to_plan_indices: Dict[MetadataIndex, Set[int]] = defaultdict(set)  # 创建一个默认值为集合的字典，用于存储 WriteItem 在哪些计划中出现过
    write_item_idx_to_write_item: Dict[MetadataIndex, WriteItem] = {}  # 创建一个字典，用于将 WriteItem 的索引映射到 WriteItem 对象

    # 遍历所有的 SavePlan 对象列表
    for plan_idx, plan in enumerate(all_plans):
        # 遍历每个 SavePlan 中的 WriteItem 对象列表
        for write_item in plan.items:
            # 将每个 WriteItem 对象映射到其所在的计划中
            write_item_to_plan_indices[write_item.index].add(plan_idx)
            # 将 WriteItem 的索引映射到 WriteItem 对象本身
            write_item_idx_to_write_item[write_item.index] = write_item

    # 根据选择条件，将条目放入计划中存储空间最小的计划，并从其他计划中移除它
    to_remove: List[Set] = [set() for _ in range(len(all_plans))]  # 创建一个列表，每个元素为一个集合，用于存储需要移除的 WriteItem 索引
    plan_to_size = [0] * len(all_plans)  # 创建一个列表，用于存储每个计划的存储空间大小

    # 遍历 WriteItem 到计划索引的映射
    for write_item_idx, plan_indices in write_item_to_plan_indices.items():
        # 根据 save_to_lowest_rank 的值选择最小的计划索引
        if save_to_lowest_rank:
            select_plan_idx = min(plan_indices)
        else:
            select_plan_idx = min(
                plan_indices, key=lambda plan_idx: plan_to_size[plan_idx]
            )

        write_item = write_item_idx_to_write_item[write_item_idx]
        
        # 忽略非张量类型的存储大小，因为无法确定它们代表的存储量
        plan_to_size[select_plan_idx] += write_item.tensor_storage_size() or 1

        # 从其他计划索引中移除选择的计划索引
        plan_indices.remove(select_plan_idx)
        for plan_idx in plan_indices:
            to_remove[plan_idx].add(write_item_idx)

    # 根据需要移除的 WriteItem 索引，更新每个 SavePlan 中的 WriteItem 列表
    for plan_idx, remove_set in enumerate(to_remove):
        new_items = [
            write_item
            for write_item in all_plans[plan_idx].items
            if write_item.index not in remove_set
        ]
        all_plans[plan_idx] = dataclasses.replace(all_plans[plan_idx], items=new_items)

    # 返回更新后的 SavePlan 对象列表
    return all_plans
```