# `.\pytorch\test\test_fx_passes.py`

```py
# 导入必要的库
from dataclasses import dataclass
import operator
import logging
import sys

import torch
from torch.fx._symbolic_trace import symbolic_trace

# 导入分区器、操作符支持、融合工具和子图匹配器
from torch.fx.passes.infra.partitioner import CapabilityBasedPartitioner
from torch.fx.passes.operator_support import OperatorSupport
from torch.fx.passes.utils.fuser_utils import fuse_by_partitions
from torch.fx.passes.utils.matcher_utils import SubgraphMatcher

# 导入测试相关的库
from torch.testing._internal.common_utils import run_tests, parametrize, instantiate_parametrized_tests
from torch.testing._internal.jit_utils import JitTestCase

# 配置日志记录器
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# 定义一个测试模块
class TestModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 4)
        self.linear2 = torch.nn.Linear(4, 4)
        self.param = torch.nn.Parameter(torch.rand(4, 4))

    def forward(self, a, b, c):
        add = a + b

        linear_1 = self.linear(add)

        add_1 = add + c
        add_2 = add_1 + self.param
        add_3 = add_1 + linear_1
        add_4 = add_2 + add_3

        linear_2 = self.linear2(add_4)

        add_5 = linear_2 + add_4
        add_6 = add_5 + a
        relu = add_6.relu()

        return add_4, add_6, relu

# 定义另一个测试模块
class TestDeepModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 4)

    def forward(self, a, b, c):
        o = a + b
        o = o + 1.0

        # 测试以避免在通过中使用深度优先搜索。由于 Python 有最大递归深度。
        for _ in range(sys.getrecursionlimit() + 1):
            o = o - c

        return o

# 定义一些静态方法，用于测试不同的前向传播函数
class TestPartitionFunctions:
    @staticmethod
    def forward1(a, b, c):
        add = a + b
        add_1 = add + b
        add_2 = add_1 + c
        relu_1 = add_2.relu()
        add_3 = add_1 + add_2
        add_4 = add_1 + relu_1 + add_3
        relu_2 = add_4.relu()
        add_5 = relu_2 + add_4
        add_6 = add_5 + add_4
        return add_4, add_6

    @staticmethod
    def forward2(a, b, _):
        add = a + b
        add_1 = add + b
        relu_1 = add_1.relu()  # 被此处阻塞
        add_3 = add_1 + relu_1
        add_4 = add_1 + add_3
        return add_4, add_1

    @staticmethod
    def forward3(a, b, c):
        add = a + b
        add_1 = a + c
        add_2 = b + c
        return add, add_1, add_2

    @staticmethod
    def forward4(a, b, c):
        add = a + b
        add_1 = a + c
        add_2 = b + c
        return torch.where(add > 0, add_1, add_2)

    @staticmethod
    def forward5(a, b, c):
        # add 应该被融合到右分支，因为左分支不受支持
        add = a + 1
        # 左分支
        relu = add.relu()
        # 右分支
        add_1 = add + 2
        return relu, add_1

    @staticmethod
    @staticmethod
    def forward6(a, b, c):
        # Define 'add' as 'a + 1'
        add = a + 1
        # Apply Rectified Linear Unit (ReLU) function to 'add' for left branch
        relu = add.relu()
        # Apply Rectified Linear Unit (ReLU) function to 'add' for right branch
        relu_1 = add.relu()
        # Return ReLU outputs from both branches
        return relu, relu_1

    @staticmethod
    def forward7(a, b, c):
        # Define 'add' as 'a + 1'
        add = a + 1
        # Left branch: 'add_1' is 'add + 2'
        add_1 = add + 2
        # Right branch: 'add_2' is 'add + 1'
        add_2 = add + 1
        # Right branch: 'add_3' is 'add_2 + 1'
        add_3 = add_2 + 1
        # Return results from both branches
        return add_3, add_1

    @staticmethod
    def forward8(a, b, c):
        # Define 'add' as 'a + 1'
        add = a + 1
        # Left branch: 'add_1' is 'add + 2'
        add_1 = add + 2
        # Right branch: 'add_2' is 'add + 1'
        add_2 = add + 1
        # Merge results from both branches: 'add_3' is 'add_2 + add_1'
        add_3 = add_2 + add_1
        # Return merged result
        return add_3

    @staticmethod
    def forward9(a, b, c):
        # Define 'add' as 'a + 1'
        add = a + 1
        # Branch 1: 'add_1' is 'add + 1'
        add_1 = add + 1
        # Branch 2: 'add_2' is 'add + 1'
        add_2 = add + 1
        # Branch 3: 'add_3' is 'add + 1'
        add_3 = add + 1
        # Stack results of all branches in a tensor 'out'
        out = torch.stack([add_1, add_2, add_3])
        # Return stacked tensor
        return out

    @staticmethod
    def forward10(a, b, c):
        # Define 'add' as 'a + 1'
        add = a + 1
        # Branch 1: 'add_1' is 'add + 1'
        add_1 = add + 1
        # Branch 2: 'add_2' is 'add + 1'
        add_2 = add + 1
        # Branch 3: 'add_3' is 'add + add_2'
        add_3 = add + add_2
        # Stack results of all branches in a tensor 'out'
        out = torch.stack([add_1, add_2, add_3])
        # Return stacked tensor
        return out

    @staticmethod
    def forward11(a, b, c):
        # Define 'add' as 'a + 1'
        add = a + 1
        # Branch 1: Apply ReLU function to 'add' and assign to 'add_1'
        add_1 = add.relu()
        # Branch 2: 'add_2' is 'add + add_1'
        add_2 = add + add_1
        # Branch 3: Apply ReLU function to 'add' and assign to 'add_3'
        add_3 = add.relu()
        # Stack results of all branches in a tensor 'out'
        out = torch.stack([add_1, add_2, add_3])
        # Return stacked tensor
        return out

    @staticmethod
    def forward12(a, b, c):
        # Compute 'b0' as 'a + 1.0'
        b0 = a + 1.0
        # Compute 'c0' as 'a + 1.5'
        c0 = a + 1.5
        # Apply ReLU to 'b0' and assign to 'x0'
        x0 = b0.relu()
        # Apply ReLU to 'c0' and assign to 'x1'
        x1 = c0.relu()
        # Compute 'b1' as 'b0 + x1'
        b1 = b0 + x1
        # Compute 'c2' as 'x0 + c0'
        c2 = x0 + c0
        # Return computed values 'b1' and 'c2'
        return b1, c2

    @staticmethod
    def forward13(a, b, c):
        # Split tensor 'a' into four parts: 'a0', 'a1', 'a2', 'a3'
        a0, a1, a2, a3 = a.split(1, 0)
        # Compute 'b1' as 'a0 + b'
        b1 = a0 + b
        # Compute 'c1' as 'a1 + c'
        c1 = a1 + c
        # Return sum of 'b1' and 'c1'
        return b1 + c1

    @staticmethod
    def forward14(a, b, c):
        # Compute mean and standard deviation of tensor 'a' and assign to 'a0', 'a1'
        a0, a1 = torch.ops.aten.std_mean(a)
        # Compute 'out' as 'a0 + 1.0'
        out = a0 + 1.0
        # Return 'out'
        return out

    @staticmethod
    def forward15(a, b, c):
        # Reshape tensor 'a' to a 2x2 matrix and assign to 'a0'
        a0 = torch.ops.aten.view(a, [2, 2])
        # Permute dimensions of 'a0' and assign to 'a1'
        a1 = torch.ops.aten.permute(a0, [1, 0])
        # Compute 'a2' as 'a1 + 1.0'
        a2 = a1 + 1.0
        # Permute dimensions of 'a2' and assign to 'a3'
        a3 = torch.ops.aten.permute(a2, [1, 0])
        # Compute 'a4' as 'a3 + 1.0'
        a4 = a3 + 1.0
        # Permute dimensions of 'a4' and return the result
        a5 = torch.ops.aten.permute(a4, [1, 0])
        return a5
    # 定义一个静态方法 `forward16`，接受三个参数 `a`, `b`, `c`
    def forward16(a, b, c):
        # 将 `a` 减去 1.0 赋给 `a0`
        a0 = a - 1.0
        # 使用 `torch.ops.aten.view` 将 `a0` 视图转换为一个 2x2 的张量 `a1`
        a1 = torch.ops.aten.view(a0, [2, 2])
        # 使用 `torch.ops.aten.permute` 对 `a1` 进行维度置换，得到 `a2`
        a2 = torch.ops.aten.permute(a1, [1, 0])
        # 将 `a2` 加上 1.0 赋给 `a3`
        a3 = a2 + 1.0
        # 再次使用 `torch.ops.aten.permute` 对 `a3` 进行维度置换，得到 `a4`
        a4 = torch.ops.aten.permute(a3, [1, 0])
        # 将 `a4` 加上 1.0 赋给 `a5`
        a5 = a4 + 1.0
        # 再次使用 `torch.ops.aten.permute` 对 `a5` 进行维度置换，得到 `a6`
        a6 = torch.ops.aten.permute(a5, [1, 0])
        # 再次使用 `torch.ops.aten.permute` 对 `a6` 进行维度置换，得到 `a7`
        a7 = torch.ops.aten.permute(a6, [1, 0])
        # 返回 `a7` 减去 1.0 的结果
        return a7 - 1.0

    # 定义一个静态方法 `forward17`，接受六个参数 `a`, `b`, `c`, `d`, `e`, `f`
    @staticmethod
    def forward17(a, b, c, d, e, f):
        # 将 `a` 加上 `b` 赋给 `a0`
        a0 = a + b
        # 将 `c` 加上 `d` 赋给 `a1`
        a1 = c + d
        # 将 `e` 加上 `f` 赋给 `a2`
        a2 = e + f
        # 返回 `a0`, `a1`, `a2` 作为结果
        return a0, a1, a2

    # 定义一个静态方法 `forward18`，接受三个参数 `a`, `b`, `c`
    @staticmethod
    def forward18(a, b, c):
        # 使用 `torch.ops.aten.var_mean` 函数计算 `a` 的方差和均值，分别赋给 `a0`, `a1`
        a0, a1 = torch.ops.aten.var_mean(a)
        # 返回 `a0` 作为结果
        return a0
# 定义一个模拟的 OperatorSupport 类，仅支持 operator.add 操作
class MockOperatorSupport(OperatorSupport):
    # 判断节点是否受支持的方法
    def is_node_supported(self, submodules, node: torch.fx.Node) -> bool:
        # 如果节点的操作是调用函数，且目标函数是 operator.add、operator.getitem、torch.ops.aten.view、torch.ops.aten.permute、torch.ops.aten.std_mean 中的一个，则返回 True
        return (node.op == "call_function" and
                node.target in {operator.add, operator.getitem,
                                torch.ops.aten.view,
                                torch.ops.aten.permute,
                                torch.ops.aten.std_mean})

# 使用参数化测试实例化的类
@instantiate_parametrized_tests
class TestFXGraphPasses(JitTestCase):

    # 参数化测试的参数，每组参数包含一个测试函数、预期分区结果以及是否进行非计算传递的标志
    @parametrize("fn, expected_partition, bookend_non_compute_pass", [
        # 测试函数 forward1，期望分区结果为多个子列表，不包含非计算传递
        (TestPartitionFunctions.forward1, [["add_7", "add_6"], ["add_5", "add_4", "add_3"], ["add_2", "add_1", "add"]], False),
        # 测试函数 forward2，期望分区结果为多个子列表，不包含非计算传递
        (TestPartitionFunctions.forward2, [["add_3", "add_2"], ["add_1", "add"]], False),

        # 以下为其他测试函数的参数化设置，每组参数都包含测试函数、预期分区结果和是否进行非计算传递的标志
        (TestPartitionFunctions.forward3, [["add_2", "add_1", "add"]], False),
        (TestPartitionFunctions.forward4, [["add_2", "add_1", "add"]], False),
        (TestPartitionFunctions.forward5, [["add_1", "add"]], False),
        (TestPartitionFunctions.forward6, [["add"]], False),
        (TestPartitionFunctions.forward7, [["add_3", "add_2", "add", "add_1"]], False),
        (TestPartitionFunctions.forward8, [["add_3", "add_2", "add", "add_1"]], False),
        (TestPartitionFunctions.forward9, [['add_3', 'add_2', 'add_1', 'add']], False),
        (TestPartitionFunctions.forward10, [['add_3', 'add_2', 'add', 'add_1']], False),
        (TestPartitionFunctions.forward11, [['add_1'], ['add']], False),
        (TestPartitionFunctions.forward12, [["add_2", "add_3", "add_4"], ["add", "add_1"]], False),
        (TestPartitionFunctions.forward13, [["add_2", "add_1", "add"]], False),
        (TestPartitionFunctions.forward14, [["add", "std_mean", "getitem", "getitem_1"]], False),
        (TestPartitionFunctions.forward15, [["permute_1", "add_1", "add"]], True),
        (TestPartitionFunctions.forward15, [['add_1', 'add', 'permute_1', 'view', 'permute_2', 'permute_3', 'permute']], False),
        (TestPartitionFunctions.forward16, [["permute_1", "add_1", "add"]], True),
        (TestPartitionFunctions.forward16, [['add_1', 'add', 'permute_1', 'view', 'permute_2', 'permute_3', 'permute']], False),
        (TestPartitionFunctions.forward18, [], False),  # forward18 测试函数的预期分区结果为空列表，不包含非计算传递
    ])
    # 定义一个测试方法，用于测试分区器的功能
    def test_partitioner(self, fn, expected_partition, bookend_non_compute_pass):
        # 对输入的函数进行符号化追踪，生成追踪对象
        traced = symbolic_trace(fn)

        # 如果需要在分区处理前后进行非计算操作的书签操作
        non_compute_ops = []
        if bookend_non_compute_pass:
            non_compute_ops = ["torch.ops.aten.view", "torch.ops.aten.permute"]

        # 创建模拟的运算符支持对象
        supported_ops = MockOperatorSupport()

        # 使用能力基础分区器创建分区器对象
        partitioner = CapabilityBasedPartitioner(traced,
                                                 supported_ops,
                                                 allows_single_node_partition=True,
                                                 non_compute_ops=non_compute_ops)

        # 提议分区方案
        partitions = partitioner.propose_partitions()

        # 如果需要，在分区中去除非计算操作的书签操作
        if bookend_non_compute_pass:
            partitioner.remove_bookend_non_compute_ops(partitions)

        # 提取每个分区中节点的名称，形成列表嵌套结构
        partitions_name = [[node.name for node in partition.nodes] for partition in partitions]

        # 断言分区名称列表的长度与预期分区的长度相同
        assert len(partitions_name) == len(expected_partition)

        # 逐个比较每个分区的节点名称集合与预期分区的节点名称集合是否相同
        for i in range(len(partitions_name)):
            assert set(partitions_name[i]) == set(expected_partition[i])

        # 融合分区得到融合图
        fused_graph = partitioner.fuse_partitions(partitions)

        # 生成随机的张量输入 a, b, c
        a, b, c = torch.rand(4), torch.rand(4), torch.rand(4)

        # 计算预期的输出结果
        expected = fn(a, b, c)

        # 使用融合图计算结果
        result = fused_graph(a, b, c)

        # 断言预期输出与计算结果的接近程度
        torch.testing.assert_close(expected, result)

    # 使用参数化装饰器，定义另一个测试方法，用于测试分区器生成独立输出的功能
    @parametrize("fn, expected_partition", [
        (TestPartitionFunctions.forward17, [['add', 'add_1', 'add_2']]),
    ])
    def test_partitioner_independent_output(self, fn, expected_partition):
        # 对输入的函数进行符号化追踪，生成追踪对象
        traced = symbolic_trace(fn)

        # 创建模拟的运算符支持对象
        supported_ops = MockOperatorSupport()

        # 使用能力基础分区器创建分区器对象
        partitioner = CapabilityBasedPartitioner(traced,
                                                 supported_ops,
                                                 allows_single_node_partition=True)

        # 提议分区方案
        partitions = partitioner.propose_partitions()

        # 提取每个分区中节点的名称，形成列表嵌套结构
        partitions_name = [[node.name for node in partition.nodes] for partition in partitions]

        # 断言分区名称列表的长度与预期分区的长度相同
        assert len(partitions_name) == len(expected_partition)

        # 逐个比较每个分区的节点名称集合与预期分区的节点名称集合是否相同
        for i in range(len(partitions_name)):
            assert set(partitions_name[i]) == set(expected_partition[i])

        # 融合分区得到融合图
        fused_graph = partitioner.fuse_partitions(partitions)

        # 生成随机的张量输入 a, b, c, d, e, f
        a, b, c, d, e, f = torch.rand(4), torch.rand(4), torch.rand(4), torch.rand(4), torch.rand(4), torch.rand(4)

        # 计算预期的输出结果
        expected = fn(a, b, c, d, e, f)

        # 使用融合图计算结果
        result = fused_graph(a, b, c, d, e, f)

        # 断言预期输出与计算结果的接近程度
        torch.testing.assert_close(expected, result)
    @parametrize("partition", [
        [['add', 'add_1'], ['add_5', 'add_6']],  # 测试参数化：多个分区定义，每个分区由节点名称列表组成
        [['add', 'add_1', 'add_2']],  # 垂直融合
        [['add_2', 'add_3']],         # 水平融合
        [['add_3', 'add_4']],         # 水平融合
        [['add_6', 'add_5']],         # 节点顺序任意
        [['add_4', 'add_1', 'add_3', 'add_2']],  # 节点顺序任意
        [['add_5', 'add_6'], ['add_1', 'add_2', 'add_3', 'add_4']],  # 分区顺序任意
        [['add_5', 'linear2']],       # 包含 call_function 和 call_module 节点
        [['add_6', 'relu']],          # 包含 call_function 和 call_module 节点
        [['param', 'add_2']],         # 包含 get_attr 和 call_module 节点
        [['param', 'add_1', 'linear']],  # 包含 get_attr、call_function 和 call_module 节点
        [["add", "linear", "add_1", "param", "add_2", "add_3", "add_4", "linear2", "add_5", "add_6", "relu"]],  # 完整的图
    ])
    def test_fuser_util(self, partition):
        # 创建测试模块实例
        m = TestModule()
        # 对模块进行符号跟踪
        gm = symbolic_trace(m)

        # 创建节点名称到节点对象的映射
        nodes_by_name = {node.name : node for node in gm.graph.nodes}

        # 初始化分区列表
        partitions = []
        # 根据每个分区的节点名称列表，获取节点对象并加入分区列表
        for node_names in partition:
            partitions.append([nodes_by_name[name] for name in node_names])

        # 使用分区信息对图进行融合
        fused_graph = fuse_by_partitions(gm, partitions)

        # 创建测试数据
        a, b, c = torch.rand(4), torch.rand(4), torch.rand(4)

        # 计算预期输出
        expected = m(a, b, c)
        # 在融合后的图上进行计算
        result = fused_graph(a, b, c)

        # 断言预期输出与计算结果的近似性
        torch.testing.assert_close(expected, result)

    @parametrize("partition", [
        [['add', 'add_1'], ['add_1', 'add_5', 'add_6']],  # add_1 存在于多个分区中
        [['add', 'add_1', 'add_3']],    # 无效的分区：循环依赖
        [['add_4', 'add_5']],    # 无效的分区：循环依赖
        [['relu', 'add_5']],    # 无效的分区：循环依赖
    ])
    def test_fuser_util_xfail(self, partition):
        # 创建测试模块实例
        m = TestModule()
        # 对模块进行符号跟踪
        gm = symbolic_trace(m)

        # 创建节点名称到节点对象的映射
        nodes_by_name = {node.name : node for node in gm.graph.nodes}

        # 初始化分区列表
        partitions = []
        # 根据每个分区的节点名称列表，获取节点对象并加入分区列表
        for node_names in partition:
            partitions.append([nodes_by_name[name] for name in node_names])

        # 使用分区信息尝试进行图的融合，期望抛出异常
        with self.assertRaises(Exception):
            fuse_by_partitions(gm, partitions)

    def test_fuser_pass_deep_model(self):
        # 创建深层测试模块实例
        m = TestDeepModule()
        # 对模块进行符号跟踪
        traced = symbolic_trace(m)

        # 创建模拟的运算符支持实例
        supported_ops = MockOperatorSupport()
        # 基于能力的分区器，允许单节点分区
        partitioner = CapabilityBasedPartitioner(traced,
                                                 supported_ops,
                                                 allows_single_node_partition=True)
        # 提议生成分区
        partitions = partitioner.propose_partitions()
@dataclass
class TestCase:
    match_output: bool  # 是否匹配输出
    match_placeholder: bool  # 是否匹配占位符
    num_matches: int  # 匹配数量
    remove_overlapping_matches: bool = True  # 是否移除重叠匹配，默认为True，表示移除

class SingleNodePattern:
    @staticmethod
    def forward(x):
        val = torch.neg(x)  # 计算输入张量的负值
        return torch.add(val, val)  # 返回两倍的负值

    @staticmethod
    def pattern(a):
        return torch.neg(a)  # 返回输入张量的负值

    test_cases = [
        # match_output, match_placeholder, num_matches
        TestCase(False, False, 1),  # 第一个测试用例：不匹配输出，不匹配占位符，有1个匹配
        TestCase(True, False, 0),   # 第二个测试用例：匹配输出，不匹配占位符，没有匹配
        TestCase(False, True, 1),   # 第三个测试用例：不匹配输出，匹配占位符，有1个匹配
        TestCase(True, True, 0)     # 第四个测试用例：匹配输出，匹配占位符，没有匹配
    ]

class SimplePattern:
    @staticmethod
    def forward(x, w1, w2):
        m1 = torch.cat([w1, w2]).sum()  # 将 w1 和 w2 拼接后求和
        m2 = torch.cat([w2, w1]).sum()  # 将 w2 和 w1 拼接后求和
        m3 = torch.cat([m1, m2]).sum()  # 将 m1 和 m2 拼接后求和
        return x + torch.max(m1) + torch.max(m2) + m3  # 返回 x 加上 m1、m2 的最大值及其和

    @staticmethod
    def pattern(a, b):
        return torch.cat([a, b]).sum()  # 返回 a 和 b 拼接后求和

    test_cases = [
        # match_output, match_placeholder, num_matches
        TestCase(False, False, 3),  # 第一个测试用例：不匹配输出，不匹配占位符，有3个匹配
        TestCase(True, False, 0),   # 第二个测试用例：匹配输出，不匹配占位符，没有匹配
        TestCase(False, True, 2),   # 第三个测试用例：不匹配输出，匹配占位符，有2个匹配
        TestCase(True, True, 0)     # 第四个测试用例：匹配输出，匹配占位符，没有匹配
    ]

class SimpleFullGraphMatching:
    @staticmethod
    def forward(x):
        a = torch.neg(x)  # 计算输入张量的负值
        return torch.add(a, a)  # 返回两倍的负值

    @staticmethod
    def pattern(x):
        a = torch.neg(x)  # 计算输入张量的负值
        return torch.add(a, a)  # 返回两倍的负值

    test_cases = [
        # match_output, match_placeholder, num_matches
        TestCase(False, False, 1),  # 第一个测试用例：不匹配输出，不匹配占位符，有1个匹配
        TestCase(True, False, 1),   # 第二个测试用例：匹配输出，不匹配占位符，有1个匹配
        TestCase(False, True, 1),   # 第三个测试用例：不匹配输出，匹配占位符，有1个匹配
        TestCase(True, True, 1)     # 第四个测试用例：匹配输出，匹配占位符，有1个匹配
    ]

class DiamondShapePatternTestCase:
    @staticmethod
    def forward(x):
        a = torch.neg(x)  # 计算输入张量的负值

        a = a.relu()  # 对 a 应用 ReLU 激活函数
        left = a.sigmoid()  # 对 a 应用 sigmoid 激活函数
        right = a.relu()  # 对 a 应用 ReLU 激活函数
        out = left + right  # 计算 left 和 right 的和

        return out  # 返回结果 out

    @staticmethod
    def pattern(a):
        a = a.relu()  # 对 a 应用 ReLU 激活函数
        left = a.sigmoid()  # 对 a 应用 sigmoid 激活函数
        right = a.relu()  # 对 a 应用 ReLU 激活函数
        out = left + right  # 计算 left 和 right 的和
        return out  # 返回结果 out

    test_cases = [
        # match_output, match_placeholder, num_matches
        TestCase(False, False, 1),  # 第一个测试用例：不匹配输出，不匹配占位符，有1个匹配
        TestCase(True, False, 1),   # 第二个测试用例：匹配输出，不匹配占位符，有1个匹配
        TestCase(False, True, 0),   # 第三个测试用例：不匹配输出，匹配占位符，没有匹配
        TestCase(True, True, 0)     # 第四个测试用例：匹配输出，匹配占位符，没有匹配
    ]

class NonFullyContainedMatches:
    @staticmethod
    def forward(x, w1, w2, b1, b2):
        # fully contained matched subgraph
        m1 = torch.cat([w1, w2])  # 将 w1 和 w2 拼接
        m2 = torch.cat([x, b2])   # 将 x 和 b2 拼接
        t0 = torch.addmm(b1, m1, m2.t())  # 使用 b1、m1 和 m2 的转置进行矩阵乘法加法运算
        t0_sum = torch.sum(t0)   # 对 t0 求和

        # leaking matched subgraph, m3 is leaked
        m3 = torch.cat([w1, w2])  # 将 w1 和 w2 拼接
        m4 = torch.cat([x, b2])   # 将 x 和 b2 拼接
        t1 = torch.addmm(b1, m3, m4.t())  # 使用 b1、m3 和 m4 的转置进行矩阵乘法加法运算
        m3_sum = torch.sum(m3)   # 对 m3 求和

        return t0_sum, m3_sum  # 返回 t0 的和及 m3 的和

    @staticmethod
    def pattern(x, w1, w2, b1, b2):
        m1 = torch.cat([w1, w2])  # 将 w1 和 w2 拼接
        m2 = torch.cat([x, b2])   # 将 x 和 b2 拼接
        return torch.addmm(b1, m1, m2.t())  # 返回使用 b1、m1 和 m2 的转置进行矩阵乘法加法运算
    # 定义测试用例列表，每个元素是一个 TestCase 对象，用于测试不同情况下的匹配输出、匹配占位符和匹配数量
    test_cases = [
        # 创建一个 TestCase 对象，期望的输出为 False，不期望匹配占位符，期望匹配数量为 1
        TestCase(False, False, 1),

        # 创建一个 TestCase 对象，期望的输出为 True，不期望匹配占位符，期望匹配数量为 0
        TestCase(True, False, 0),

        # 创建一个 TestCase 对象，期望的输出为 False，期望匹配占位符，期望匹配数量为 1
        TestCase(False, True, 1),     # 注意：泄露的占位符使用不会泄露
    ]
class ChainRepeatedPattern:
    @staticmethod
    def forward(x):
        x = torch.sigmoid(x)  # 对输入张量进行 sigmoid 激活函数操作
        x = torch.sigmoid(x)  # 再次对结果进行 sigmoid 操作
        x = torch.sigmoid(x)  # 再次对结果进行 sigmoid 操作
        return torch.sigmoid(x)  # 返回最终结果的 sigmoid 操作结果

    @staticmethod
    def pattern(x):
        return torch.sigmoid(torch.sigmoid(x))  # 对输入张量进行两次 sigmoid 操作

    test_cases = [
        # match_output, match_placeholder, num_matches
        TestCase(False, False, 3, remove_overlapping_matches=False),  # 不匹配输出、占位符，期望三次匹配
        TestCase(False, False, 2, remove_overlapping_matches=True),   # 不匹配输出、占位符，期望两次匹配，允许重叠
        TestCase(True, False, 1),    # 匹配输出，不匹配占位符，期望一次匹配
        TestCase(False, True, 1),    # 不匹配输出，匹配占位符，期望一次匹配
        TestCase(True, True, 0)      # 匹配输出、占位符，期望零次匹配
    ]

class QuantizationModel:
    @staticmethod
    def forward(x):
        x += 3  # 张量加 3
        x = x.dequantize()  # 反量化处理
        x = torch.sigmoid(x)  # sigmoid 操作
        x = x.to(torch.float16)  # 转换为 float16 类型
        return x

    @staticmethod
    def pattern(x):
        x = x.dequantize()  # 反量化处理
        x = torch.sigmoid(x)  # sigmoid 操作
        x = x.to(torch.float16)  # 转换为 float16 类型
        return x

    test_cases = [
        # match_output, match_placeholder, num_matches
        TestCase(False, False, 1),  # 不匹配输出、占位符，期望一次匹配
        TestCase(True, False, 1),   # 匹配输出，不匹配占位符，期望一次匹配
        TestCase(False, True, 0),   # 不匹配输出，匹配占位符，期望零次匹配
        TestCase(True, True, 0)     # 匹配输出、占位符，期望零次匹配
    ]

class MultipleOutputsWithDependency:
    @staticmethod
    def forward(x):
        y = x.relu()      # 计算 x 的 ReLU 激活值
        z = y.sigmoid()   # 对 y 进行 sigmoid 操作
        return z, y       # 返回 z 和 y

    @staticmethod
    def pattern(a):
        b = a.relu()      # 计算 a 的 ReLU 激活值
        c = b.sigmoid()   # 对 b 进行 sigmoid 操作
        return b, c       # 返回 b 和 c，存在数据依赖关系

    test_cases = [
        # match_output, match_placeholder, num_matches
        TestCase(False, False, 1),  # 不匹配输出、占位符，期望一次匹配
        TestCase(True, False, 0),   # 匹配输出，不匹配占位符，期望零次匹配
        TestCase(False, True, 1),   # 不匹配输出，匹配占位符，期望一次匹配
        TestCase(True, True, 0)     # 匹配输出、占位符，期望零次匹配
    ]

class MultipleOutputsWithoutDependency:
    @staticmethod
    def forward(x):
        x = x + 1  # 张量加 1

        # target subgraph to match
        x = x.relu()      # 计算 x 的 ReLU 激活值
        z = x.sum()       # 计算 x 的和
        y = x.sigmoid()   # 对 x 进行 sigmoid 操作

        out = y.sigmoid() + z.sum()  # 计算输出值
        return out

    @staticmethod
    def pattern(a):
        a = a.relu()      # 计算 a 的 ReLU 激活值
        b = a.sigmoid()   # 对 a 进行 sigmoid 操作
        c = a.sum()       # 计算 a 的和
        return b, c       # 返回 b 和 c，不存在数据依赖关系

    test_cases = [
        # match_output, match_placeholder, num_matches
        TestCase(False, False, 1),  # 不匹配输出、占位符，期望一次匹配
        TestCase(True, False, 0),   # 匹配输出，不匹配占位符，期望零次匹配
        TestCase(False, True, 0),   # 不匹配输出，匹配占位符，期望零次匹配
        TestCase(True, True, 0)     # 匹配输出、占位符，期望零次匹配
    ]

class MultipleOutputsMultipleOverlappingMatches:
    @staticmethod
    def forward(x):
        x = x + 1  # 张量加 1

        # target subgraph to match
        x = x.relu()      # 计算 x 的 ReLU 激活值
        z = x.sum()       # 计算 x 的和
        z1 = x.sum()      # 再次计算 x 的和
        y = x.sigmoid()   # 对 x 进行 sigmoid 操作
        y1 = x.sigmoid()  # 再次对 x 进行 sigmoid 操作

        return z + z1 + y + y1  # 返回所有计算结果的总和

    @staticmethod
    def pattern(a):
        a = a.relu()      # 计算 a 的 ReLU 激活值
        b = a.sigmoid()   # 对 a 进行 sigmoid 操作
        c = a.sum()       # 计算 a 的和
        return a, b, c    # 返回 a、b、c，允许重叠匹配

    test_cases = [
        # match_output, match_placeholder, num_matches
        TestCase(False, False, 4, remove_overlapping_matches=False),  # 不匹配输出、占位符，期望四次匹配，不移除重叠
        TestCase(False, False, 1, remove_overlapping_matches=True),   # 不匹配输出、占位符，期望一次匹配，允许移除重叠
    ]

class MultipleOutputsMultipleNonOverlappingMatches:
    @staticmethod
    def forward(x):
        # 省略部分代码...
        x = x + 1

        # target subgraph to match
        x = x.relu()      # 计算 x 的 ReLU 激活值
        z = x.sum()       # 计算 x 的和
        z1 = x.sum()      # 再次计算 x 的和
        y = x.sigmoid()   # 对 x 进行 sigmoid 操作
        y1 = x.sigmoid()  # 再次对 x 进行 sigmoid 操作

        return z + z1 + y + y1  # 返回所有计算结果的总和

    @staticmethod
    def pattern(a):
        a = a.relu()      # 计算 a 的 ReLU 激活值
        b = a.sigmoid()   # 对 a 进行 sigmoid 操作
        c = a.sum()       # 计算 a 的和
        return b, c       # 返回 b 和 c，不允许重叠匹配

    test_cases = [
        # match_output, match_placeholder, num_matches
        TestCase(False, False, 1),  # 不匹配输出、占位符，期望一次匹配
        TestCase(True, False, 0),   # 匹配输出，不匹配占位符，期望零次匹配
        TestCase(False, True, 0),   # 不匹配输出，匹配占位符，期望零次匹配
        TestCase(True, True, 0)     # 匹配输出、占位符，期望零次匹配
    ]
    # 定义一个前向传播函数，输入参数为 x
    def forward(x):
        # x 值加 1
        x = x + 1

        # 对 x 应用 ReLU 激活函数
        x = x.relu()
        # 计算 x 的元素和
        z = x.sum()
        # 对 x 应用 Sigmoid 激活函数
        y = x.sigmoid()

        # 再次对 x 应用 ReLU 激活函数
        x = x.relu()
        # 再次计算 x 的元素和
        z1 = x.sum()
        # 再次对 x 应用 Sigmoid 激活函数
        y1 = x.sigmoid()

        # 返回 z、z1、y 和 y1 的总和作为前向传播的结果
        return z + z1 + y + y1

    @staticmethod
    def pattern(a):
        # 对输入 a 应用 ReLU 激活函数
        a = a.relu()
        # 对 a 应用 Sigmoid 激活函数
        b = a.sigmoid()
        # 计算 a 的元素和
        c = a.sum()
        # 返回经过激活函数处理后的结果 b 和元素和 c
        return b, c

    # 定义测试用例列表
    test_cases = [
        # 第一个测试用例：不匹配输出和占位符，期望匹配数为 1
        TestCase(False, False, 1),
    ]
class MultipleOutputsIdenticalAnchor:
    @staticmethod
    def forward(x):
        x = x + 1
        # 目标子图以匹配
        x = x.relu()  # 对输入进行ReLU激活操作
        y = x.sigmoid()  # 对ReLU结果进行sigmoid激活操作
        y1 = x.sigmoid()  # 再次对ReLU结果进行sigmoid激活操作
        return y, y1  # 返回两个输出 y 和 y1

    @staticmethod
    def pattern(a):
        a = a.relu()  # 对输入进行ReLU激活操作
        b = a.sigmoid()  # 对ReLU结果进行sigmoid激活操作
        b1 = a.sigmoid()  # 再次对ReLU结果进行sigmoid激活操作
        return b, b1  # 返回两个输出 b 和 b1

    test_cases = [
        # 匹配输出, 匹配占位符, 匹配数量
        TestCase(True, False, 1),  # 预期匹配一个输出（True），不匹配占位符（False），预期匹配数量为1
        TestCase(False, True, 0),  # 不预期匹配输出（False），预期匹配占位符（True），预期匹配数量为0
    ]


class MultipleOutputsHorizontalPattern:
    @staticmethod
    def forward(x):
        x = x + 1
        # 目标子图以匹配
        y1 = x.relu()  # 对输入进行ReLU激活操作
        y2 = x.sigmoid()  # 对输入进行sigmoid激活操作
        return y1, y2  # 返回两个输出 y1 和 y2

    @staticmethod
    def pattern(a):
        b1 = a.relu()  # 对输入进行ReLU激活操作
        b2 = a.sigmoid()  # 对输入进行sigmoid激活操作
        return b1, b2  # 返回两个输出 b1 和 b2

    test_cases = [
        # 匹配输出, 匹配占位符, 匹配数量
        TestCase(False, False, 1),  # 不预期匹配输出（False），不匹配占位符（False），预期匹配数量为1
        TestCase(True, False, 1),   # 预期匹配一个输出（True），不匹配占位符（False），预期匹配数量为1
        TestCase(False, True, 0),   # 不预期匹配输出（False），预期匹配占位符（True），预期匹配数量为0
        TestCase(True, True, 0)     # 预期匹配一个输出（True），预期匹配占位符（True），预期匹配数量为0
    ]


class MultiOutputWithWithInvalidMatches:
    @staticmethod
    def forward(x):
        res0 = torch.nn.functional.linear(x, torch.rand(3, 3))  # 计算线性变换
        res1 = torch.sigmoid(res0)  # 对线性变换结果进行sigmoid激活
        res2 = res0 * res1  # 逐元素乘法
        res3 = torch.sum(res2, dim=1)  # 按维度求和
        return res3  # 返回最终结果

    @staticmethod
    def pattern(a, b, c):
        lin_res = torch.nn.functional.linear(a, b)  # 计算线性变换
        mul_res = lin_res * c  # 逐元素乘法
        return lin_res, mul_res  # 返回两个输出，线性变换结果和逐元素乘法结果

    test_cases = [
        # 匹配输出, 匹配占位符, 匹配数量
        TestCase(False, False, 0),  # 不预期匹配输出（False），不匹配占位符（False），预期匹配数量为0
        TestCase(True, False, 0),   # 预期匹配一个输出（True），不匹配占位符（False），预期匹配数量为0
        TestCase(False, True, 0),   # 不预期匹配输出（False），预期匹配占位符（True），预期匹配数量为0
    ]


class QuantizationFp8Pattern:
    @classmethod
    def setup(cls):
        cls.quantization = torch.library.Library("fp8_quantization", "DEF")  # 创建fp8量化库的实例
        cls.quantization.define("quantize_per_tensor_affine_fp8(Tensor self, int dtype, float scale) -> Tensor")  # 定义量化操作
        cls.quantization.define("dequantize_per_tensor_affine_fp8(Tensor self, int dtype, float scale) -> Tensor")  # 定义反量化操作

    @classmethod
    def tearDown(cls):
        del cls.quantization  # 删除量化库实例
    # 定义类方法 `forward`，接受两个参数 arg0_1 和 arg1_1
    def forward(self, arg0_1, arg1_1):
        # 引用 Torch 自定义操作库中的量化函数
        qt = torch.ops.fp8_quantization
        # 从对象属性中获取量化比例 _scale_0
        _scale_0 = self._scale_0
        # 使用量化函数对 arg0_1 进行量化操作，使用 _scale_0 作为比例
        quantize_per_tensor_affine_fp8 = qt.quantize_per_tensor_affine_fp8(arg0_1, 0, _scale_0)
        # 使用量化函数对量化后的数据进行反量化操作，使用 _scale_0 作为比例
        dequantize_per_tensor_affine_fp8 = qt.dequantize_per_tensor_affine_fp8(quantize_per_tensor_affine_fp8, 0, _scale_0)
        # 再次使用相同的比例 _scale_0 对 arg1_1 进行量化操作
        quantize_per_tensor_affine_fp8_1 = qt.quantize_per_tensor_affine_fp8(arg1_1, 0, _scale_0)
        # 对量化后的数据进行反量化操作，使用 _scale_0 作为比例
        dequantize_per_tensor_affine_fp8_1 = qt.dequantize_per_tensor_affine_fp8(quantize_per_tensor_affine_fp8_1, 0, _scale_0)
        # 使用 Torch 的 add 操作将两个反量化后的数据相加
        add = torch.ops.aten.add.Tensor(dequantize_per_tensor_affine_fp8, dequantize_per_tensor_affine_fp8_1)
        # 再次使用 _scale_0 对相加后的结果进行量化操作
        quantize_per_tensor_affine_fp8_2 = qt.quantize_per_tensor_affine_fp8(add, 0, _scale_0)
        # 对量化后的结果进行反量化操作，使用 _scale_0 作为比例
        dequantize_per_tensor_affine_fp8_2 = qt.dequantize_per_tensor_affine_fp8(quantize_per_tensor_affine_fp8_2, 0, _scale_0)
        # 返回最终的反量化结果
        return dequantize_per_tensor_affine_fp8_2

    @staticmethod
    # 定义静态方法 `pattern`，接受多个参数用于模式匹配处理
    def pattern(a, a_dtype, a_scale, b, b_dtype, b_scale, out_scale):
        # 引用 Torch 自定义操作库中的量化函数
        qt = torch.ops.fp8_quantization
        # 使用量化函数对 a 进行反量化操作，使用 a_dtype 和 a_scale
        a = qt.dequantize_per_tensor_affine_fp8(a, a_dtype, a_scale)
        # 使用量化函数对 b 进行反量化操作，使用 b_dtype 和 b_scale
        b = qt.dequantize_per_tensor_affine_fp8(b, b_dtype, b_scale)
        # 使用 Torch 的 add 操作将反量化后的 a 和 b 相加
        output = torch.ops.aten.add.Tensor(a, b)

        # 此行看似为无效语句，因为没有赋值给任何变量或使用结果
        qt.dequantize_per_tensor_affine_fp8

        # 使用量化函数对相加后的结果 output 进行量化操作，使用 a_dtype 和 out_scale
        output = qt.quantize_per_tensor_affine_fp8(output, a_dtype, out_scale)
        # 返回量化后的输出结果
        return output

    # 定义测试用例列表，用于验证模式匹配
    test_cases = [
        # 创建一个 TestCase 对象，指示不匹配输出、不匹配占位符、1 个匹配
        TestCase(False, False, 1),
    ]
class NoAnchorFound:
    # 表示当在目标图中找不到匹配锚点时的测试案例
    # `anchor` 是模式匹配的起点，通常是返回节点的边界

    @staticmethod
    def forward(x):
        # 对输入 x 执行加一操作
        x = x + 1
        return x

    @staticmethod
    def pattern(a):
        # 对输入 a 执行 ReLU 激活函数操作
        b1 = a.relu()
        return b1

    # 定义多个测试案例
    test_cases = [
        # 匹配输出, 匹配占位符, 匹配数目
        TestCase(False, False, 0),
        TestCase(True, False, 0),
        TestCase(False, True, 0),
        TestCase(True, True, 0)
    ]

@instantiate_parametrized_tests
class TestFXMatcherUtils(JitTestCase):

    @parametrize("test_model", [
        SingleNodePattern,
        SimplePattern,
        SimpleFullGraphMatching,
        DiamondShapePatternTestCase,
        NonFullyContainedMatches,
        ChainRepeatedPattern,
        QuantizationModel,
        MultipleOutputsWithDependency,
        MultipleOutputsWithoutDependency,
        MultipleOutputsMultipleOverlappingMatches,
        MultipleOutputsMultipleNonOverlappingMatches,
        MultipleOutputsIdenticalAnchor,
        MultipleOutputsHorizontalPattern,
        MultiOutputWithWithInvalidMatches,
        QuantizationFp8Pattern,
        NoAnchorFound,
    ])
    def test_subgraph_matcher(self, test_model):

        setup = getattr(test_model, "setup", None)
        if callable(setup):
            setup()

        traced = symbolic_trace(test_model.forward)
        pattern_traced = symbolic_trace(test_model.pattern)

        for test_case in test_model.test_cases:
            # 创建子图匹配器对象
            matcher = SubgraphMatcher(pattern_traced.graph,
                                      match_output=test_case.match_output,
                                      match_placeholder=test_case.match_placeholder,
                                      remove_overlapping_matches=test_case.remove_overlapping_matches)
            # 执行匹配
            matches = matcher.match(traced.graph)

            # 断言匹配数目符合预期
            assert len(matches) == test_case.num_matches

            # 验证每个匹配的节点
            for match in matches:
                for node in pattern_traced.graph.nodes:
                    if not test_case.match_placeholder and node.op == "placeholder":
                        continue
                    if not test_case.match_output and node.op == "output":
                        continue
                    # 断言节点在匹配映射中
                    assert node in match.nodes_map

        tearDown = getattr(test_model, "tearDown", None)
        if callable(tearDown):
            tearDown()


if __name__ == "__main__":
    run_tests()
```