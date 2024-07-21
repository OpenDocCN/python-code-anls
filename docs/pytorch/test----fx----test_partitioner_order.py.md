# `.\pytorch\test\fx\test_partitioner_order.py`

```py
# Owner(s): ["module: fx"]

# 引入单元测试框架
import unittest

# 引入类型提示
from typing import Mapping

# 引入 PyTorch 库
import torch
# 引入 Torch FX 的分区器
from torch.fx.passes.infra.partitioner import CapabilityBasedPartitioner
# 引入 Torch FX 的运算符支持
from torch.fx.passes.operator_support import OperatorSupport
# 引入 Torch 测试工具中的测试用例基类
from torch.testing._internal.common_utils import TestCase

# 定义一个模拟的设备运算符支持类，继承自 OperatorSupport
class DummyDevOperatorSupport(OperatorSupport):
    # 判断节点是否被支持的方法
    def is_node_supported(
        self, submodules: Mapping[str, torch.nn.Module], node: torch.fx.Node
    ) -> bool:
        return True

# 定义一个虚拟的分区器类，继承自 CapabilityBasedPartitioner
class DummyPartitioner(CapabilityBasedPartitioner):
    # 初始化方法，接受一个图模块作为参数
    def __init__(self, graph_module: torch.fx.GraphModule):
        # 调用父类的初始化方法
        super().__init__(
            graph_module,
            DummyDevOperatorSupport(),  # 使用自定义的运算符支持类
            allows_single_node_partition=True,  # 允许单节点分区
        )

# 定义一个简单的 Torch 模块类，实现了一个简单的加法操作
class AddModule(torch.nn.Module):
    # 前向传播方法
    def forward(self, x):
        y = torch.add(x, x)
        z = torch.add(y, x)
        return z

# 定义一个测试类，继承自 Torch 测试工具中的测试用例基类
class TestPartitionerOrder(TestCase):
    # 测试方法，用于验证分区器的节点顺序
    def test_partitioner_order(self):
        m = AddModule()  # 创建 AddModule 实例
        traced_m = torch.fx.symbolic_trace(m)  # 对模块进行符号跟踪
        partions = DummyPartitioner(traced_m).propose_partitions()  # 提议分区
        partion_nodes = [list(partition.nodes) for partition in partions]  # 获取分区的节点列表
        node_order = [n.name for n in partion_nodes[0]]  # 获取第一个分区的节点名称列表

        # 执行十次循环，重复测试分区器的节点顺序是否稳定
        for _ in range(10):
            traced_m = torch.fx.symbolic_trace(m)  # 再次对模块进行符号跟踪
            new_partion = DummyPartitioner(traced_m).propose_partitions()  # 提议新的分区
            new_partion_nodes = [list(partition.nodes) for partition in new_partion]  # 获取新分区的节点列表
            new_node_order = [n.name for n in new_partion_nodes[0]]  # 获取新分区的第一个分区的节点名称列表
            self.assertTrue(node_order == new_node_order)  # 断言：节点顺序是否与原始分区相同

# 程序入口，执行单元测试
if __name__ == "__main__":
    unittest.main()
```