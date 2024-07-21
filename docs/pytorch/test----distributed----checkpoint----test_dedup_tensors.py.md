# `.\pytorch\test\distributed\checkpoint\test_dedup_tensors.py`

```py
# Owner(s): ["oncall: distributed"]

import dataclasses  # 导入用于创建数据类的模块

import torch  # 导入PyTorch深度学习框架
from torch.distributed.checkpoint._dedup_tensors import dedup_tensors  # 导入用于去重张量的函数
from torch.distributed.checkpoint.planner import SavePlan, WriteItemType  # 导入保存计划和写入项类型
from torch.distributed.checkpoint.planner_helpers import _create_write_item_for_tensor  # 导入用于创建张量写入项的辅助函数
from torch.testing._internal.common_utils import run_tests, TestCase  # 导入用于测试的实用函数和测试用例


# TODO: add comments for create_plan
def create_plan(second_fqn) -> SavePlan:
    # 第一个写入项用于重复的分片（覆盖整个张量）
    write_item_1 = _create_write_item_for_tensor("tensor_0", torch.rand(4))
    write_item_1 = dataclasses.replace(write_item_1, type=WriteItemType.SHARD)

    # 第二个写入项具有不同的键
    write_item_2 = _create_write_item_for_tensor(second_fqn, torch.rand(10))

    return SavePlan([write_item_1, write_item_2])


# TODO: add comments for TestDedupTensor
class TestDedupTensor(TestCase):
    def test_dedup_shards(self):
        rank0 = create_plan("r0")  # 创建排名0的保存计划
        rank1 = create_plan("r1")  # 创建排名1的保存计划

        dedup_plans = dedup_tensors([rank0, rank1])  # 对保存计划进行张量去重

        self.assertEqual(2, len(dedup_plans[0].items))  # 检查排名0的去重计划项数量
        self.assertEqual(1, len(dedup_plans[1].items))  # 检查排名1的去重计划项数量

        self.assertIn("tensor_0", (item.index.fqn for item in dedup_plans[0].items))  # 检查排名0的去重计划项中是否包含"tensor_0"
        self.assertIn("r0", (item.index.fqn for item in dedup_plans[0].items))  # 检查排名0的去重计划项中是否包含"r0"

        self.assertIn("r1", (item.index.fqn for item in dedup_plans[1].items))  # 检查排名1的去重计划项中是否包含"r1"


if __name__ == "__main__":
    run_tests()  # 运行测试
```