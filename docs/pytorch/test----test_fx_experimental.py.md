# `.\pytorch\test\test_fx_experimental.py`

```
# Owner(s): ["module: fx"]

# 导入必要的模块和库
import functools  # 提供了高阶函数：部分函数应用、函数包装等
import math  # 提供了数学函数和常量
import numbers  # 提供了数值抽象基类和相关实用函数
import operator  # 提供了一组函数作为 Python 的标准运算符的替代
import pickle  # 实现了 Python 对象的序列化和反序列化
import sys  # 提供了与 Python 解释器相关的变量和函数
import sympy  # 提供了符号数学的功能
import tempfile  # 提供了生成临时文件和目录的功能
import unittest  # 提供了单元测试框架
from types import BuiltinFunctionType  # 引入内置函数类型的引用
from typing import Callable, Dict, List, NamedTuple, Optional, Tuple, Union  # 引入类型提示

import torch  # PyTorch 深度学习库
import torch.fx.experimental.meta_tracer  # 导入实验性的 FX 元追踪模块
import torch.fx.experimental.optimization as optimization  # 导入实验性的 FX 优化模块
from torch.fx._symbolic_trace import symbolic_trace  # 导入符号追踪函数
from torch.fx.experimental import merge_matmul  # 导入实验性的矩阵乘法合并模块
from torch.fx.experimental.accelerator_partitioner import Partitioner  # 导入加速器分区器
from torch.fx.experimental.normalize import NormalizeArgs, NormalizeOperators  # 导入规范化功能
from torch.fx.experimental.partitioner_utils import (  # 导入分区器实用工具
    Device,
    get_latency_of_partitioned_graph,
    get_partition_to_latency_mapping,
    NodeLatency,
    PartitionerConfig,
    PartitionMode,
)
from torch.fx.experimental.rewriter import RewritingTracer  # 导入重写追踪器
from torch.fx.experimental.schema_type_annotation import AnnotateTypesWithSchema  # 导入模式类型注释器
from torch.fx.graph_module import GraphModule  # 导入图模块
from torch.fx.node import Node  # 导入节点类
from torch.fx.operator_schemas import (  # 导入操作符模式
    _torchscript_type_to_python_type,
    create_type_hint,
    normalize_function,
    normalize_module,
    type_matches,
)
from torch.fx.passes import graph_manipulation  # 导入图操作模块
from torch.fx.passes.param_fetch import lift_lowering_attrs_to_nodes  # 导入参数提取函数
from torch.fx.passes.shape_prop import ShapeProp  # 导入形状属性推断模块
from torch.fx.passes.split_module import split_module  # 导入模块拆分函数
from torch.fx.passes.annotate_getitem_nodes import annotate_getitem_nodes  # 导入注释getitem节点函数
from torch.testing._internal.common_device_type import (  # 导入设备类型测试函数
    instantiate_device_type_tests,
    onlyCPU,
    ops,
)
from torch.testing._internal.common_methods_invocations import (  # 导入方法调用测试函数
    op_db,
)
from torch.testing._internal.common_nn import (  # 导入神经网络模块测试函数
    module_tests,
    new_module_tests,
)
from torch.testing._internal.common_utils import (  # 导入常用实用工具函数
    TEST_Z3,
    run_tests,
    TestCase,
)
from torch.testing._internal.jit_utils import JitTestCase  # 导入 JIT 测试用例
import torch.utils._pytree as pytree  # 导入 PyTree 模块

try:
    import torchvision.models  # 尝试导入 torchvision 模型
    from torchvision.models import resnet18  # 导入 resnet18 模型

    HAS_TORCHVISION = True  # 标记导入 torchvision 成功
except ImportError:
    HAS_TORCHVISION = False  # 标记未能导入 torchvision
skipIfNoTorchVision = unittest.skipIf(  # 根据是否有 torchvision 决定是否跳过测试
    not HAS_TORCHVISION, "no torchvision"
)
skipIfNoMkldnn = unittest.skipIf(  # 根据是否有 MKLDNN 决定是否跳过测试
    not (torch.backends.mkldnn.enabled and torch.backends.mkldnn.is_available()),
    "no MKLDNN",
)


def symbolic_trace_with_rewrite(root: Union[torch.nn.Module, Callable]) -> GraphModule:
    """
    对给定的根模块或函数进行符号追踪，并进行重写，返回一个图模块对象。

    Args:
    - root: 要追踪的根模块或函数

    Returns:
    - GraphModule: 符号追踪并重写后的图模块对象
    """
    return GraphModule(
        root if isinstance(root, torch.nn.Module) else torch.nn.Module(),
        RewritingTracer().trace(root),
    )


class TestFXExperimental(JitTestCase):
    # FX 实验性功能的测试用例基类
    def test_find_single_partition(self):
        # 定义一个简单的测试模块，实现向量加法
        class TestModule(torch.nn.Module):
            def forward(self, a, b):
                return a + b
        
        # 创建 TestModule 实例
        m = TestModule()
        # 对模块进行符号化追踪
        traced = symbolic_trace(m)
        # 创建随机张量 a 和 b
        a = torch.rand(1)
        b = torch.rand(1)
        # 获取追踪图中所有节点的大小信息
        graph_manipulation.get_size_of_all_nodes(traced, [a, b])
        # 创建分区器实例
        partitioner = Partitioner()
        # 定义设备列表
        devices = [
            Device("dev_0", 125, 0),
            Device("dev_1", 150, 1),
            Device("dev_2", 125, 2),
        ]
        # 使用设备列表创建分区器配置
        partitioner_config = PartitionerConfig(devices)
        # 对追踪后的模块进行图分区
        ret = partitioner.partition_graph(traced, m, partitioner_config)
        # 获取分区后的模块及其子模块
        module_with_submodules = ret.module_with_submodules
        # 获取分区后的有向无环图
        dag = ret.dag
        # 断言追踪后的模块与分区后模块的执行结果相同
        self.assertEqual(traced(a, b), module_with_submodules(a, b))
        # 断言 DAG 的第一个节点的逻辑设备 ID 应为 [1]
        assert dag.nodes[0].logical_device_ids == [1]

    def test_lack_of_devices(self):
        # 定义一个简单的测试模块，实现向量加法
        class TestModule(torch.nn.Module):
            def forward(self, a, b):
                return a + b
        
        # 创建 TestModule 实例
        m = TestModule()
        # 对模块进行符号化追踪
        traced = symbolic_trace(m)
        # 创建随机张量 a 和 b
        a = torch.rand(4)
        b = torch.rand(4)
        # 获取追踪图中所有节点的大小信息
        graph_manipulation.get_size_of_all_nodes(traced, [a, b])
        # 创建分区器实例
        partitioner = Partitioner()
        # 定义设备列表，只有两个设备
        devices = [Device("dev_0", 4, 0), Device("dev_1", 4, 1)]
        # 使用设备列表创建分区器配置，并指定分区模式为基于大小
        partitioner_config = PartitionerConfig(devices, PartitionMode.size_based)
        # 初始化捕获运行时错误标志
        catch_runtime_error = False
        # 尝试对追踪后的模块进行图分区，预期会捕获 RuntimeError 异常
        try:
            ret = partitioner.partition_graph(traced, m, partitioner_config)
        except RuntimeError:
            catch_runtime_error = True
        # 断言成功捕获到 RuntimeError 异常
        assert catch_runtime_error

    def test_large_node_error(self):
        # 定义一个包含线性层的测试模块，实现线性操作和加法
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)

            def forward(self, a):
                linear = self.linear(a)
                add = linear + a
                return add
        
        # 创建 TestModule 实例
        m = TestModule()
        # 对模块进行符号化追踪
        traced = symbolic_trace(m)
        # 创建随机张量 a
        a = torch.rand(4)
        # 获取追踪图中所有节点的大小信息
        graph_manipulation.get_size_of_all_nodes(traced, [a])
        # 创建分区器实例
        partitioner = Partitioner()
        # 定义设备列表，包含五个相同的设备
        devices = [
            Device("dev_0", 40, 0),
            Device("dev_1", 40, 0),
            Device("dev_2", 40, 0),
            Device("dev_3", 40, 0),
            Device("dev_4", 40, 0),
        ]
        # 使用设备列表创建分区器配置，并指定分区模式为基于大小
        partitioner_config = PartitionerConfig(devices, PartitionMode.size_based)
        # 初始化捕获运行时错误标志
        catch_runtime_error = False
        # 尝试对追踪后的模块进行图分区，预期会捕获 RuntimeError 异常
        try:
            ret = partitioner.partition_graph(traced, m, partitioner_config)
        except RuntimeError:
            catch_runtime_error = True
        # 断言成功捕获到 RuntimeError 异常
        assert catch_runtime_error
    def test_partition_node_manipulation(self):
        # 定义一个测试用的神经网络模块类
        class TestModule(torch.nn.Module):
            def forward(self, a, b):
                # 计算输入张量 a 和 b 的加法
                add_1 = a + b
                # 将 add_1 和一个随机张量进行加法操作
                add_2 = add_1 + torch.rand(4)
                # 再次将 add_2 和另一个随机张量进行加法操作
                add_3 = add_2 + torch.rand(4)
                return add_3

        # 创建 TestModule 实例
        m = TestModule()
        # 对模块进行符号化跟踪
        traced = symbolic_trace(m)
        # 创建两个随机张量 a 和 b
        a, b = torch.rand(4), torch.rand(4)
        # 获取跟踪模块的所有节点的大小信息
        graph_manipulation.get_size_of_all_nodes(traced, [a, b])
        # 创建一个分区器实例
        partitioner = Partitioner()
        # 定义设备列表，包含一个设备 "dev_0"，内存大小为 1000，逻辑设备号为 0
        devices = [Device("dev_0", 1000, 0)]
        # 使用设备列表创建分区器配置
        partitioner_config = PartitionerConfig(devices)
        # 对跟踪模块进行图分区，返回分区器的结果
        ret = partitioner.partition_graph(traced, m, partitioner_config)
        # 获取第一个分区
        partition = partitioner.partitions[0]
        # 断言分区使用的内存字节数为 112
        assert partition.used_mem_bytes == 112
        # 选择要移除的 add_2 节点
        selected_node = None
        # 在分区的节点列表中查找名为 "add_2" 的节点
        for node in partition.nodes:
            if node.name == "add_2":
                selected_node = node
        # 从分区中移除选定的节点
        partition.remove_node(selected_node)
        # 再次断言分区使用的内存字节数为 80
        assert partition.used_mem_bytes == 80

    def test_size_based_partition(self):
        # 定义一个测试用的神经网络模块类
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 添加一个线性层和一个随机张量到模块中
                self.linear = torch.nn.Linear(4, 4)
                self.c = torch.rand(4)

            def forward(self, a, b):
                # 计算输入张量 a 和 b 的加法
                add_1 = a + b
                # 使用线性层处理 add_1
                linear = self.linear(add_1)
                # 将线性层的输出和模块中的随机张量进行加法操作
                add_2 = linear + self.c
                return add_2

        # 创建 TestModule 实例
        m = TestModule()
        # 对模块进行符号化跟踪
        traced = symbolic_trace(m)
        # 创建两个随机张量 a 和 b
        a = torch.rand(4)
        b = torch.rand(4)
        # 获取跟踪模块的所有节点的大小信息
        graph_manipulation.get_size_of_all_nodes(traced, [a, b])
        # 创建一个分区器实例
        partitioner = Partitioner()
        # 定义设备列表，包含三个设备，每个设备的内存大小为 125，逻辑设备号分别为 0, 1, 2
        devices = [
            Device("dev_0", 125, 0),
            Device("dev_1", 125, 1),
            Device("dev_2", 125, 2),
        ]
        # 使用设备列表创建分区器配置，指定分区模式为基于大小的分区
        partitioner_config = PartitionerConfig(devices, PartitionMode.size_based)
        # 对跟踪模块进行图分区，返回分区器的结果
        ret = partitioner.partition_graph(traced, m, partitioner_config)
        # 获取包含子模块的分区结果
        module_with_submodules = ret.module_with_submodules
        # 获取分区后的有向无环图 (DAG)
        dag = ret.dag
        # 断言符号化跟踪模块在输入 a 和 b 上的输出与分区后模块在相同输入上的输出一致
        self.assertEqual(traced(a, b), module_with_submodules(a, b))
        # 遍历 DAG 的节点列表，断言每个节点的逻辑设备号为其在节点列表中的索引值
        for i, node in enumerate(dag.nodes):
            assert node.logical_device_ids == [i]
    def test_partition_device_mapping(self):
        # 定义一个测试类，继承自 torch.nn.Module
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 定义一个线性层，输入和输出维度均为 4
                self.linear = torch.nn.Linear(4, 4)

            # 前向传播方法，接收输入 a
            def forward(self, a):
                # 生成一个随机张量 b，维度为 4
                b = torch.rand(4)
                # 将输入 a 和随机张量 b 相加
                add_1 = a + b
                # 对 add_1 应用定义的线性层
                linear_1 = self.linear(add_1)
                # 再次生成一个随机张量，与输入 a 相加
                add_2 = torch.rand(4) + a
                # 将 add_2 与 linear_1 相加
                add_3 = add_2 + linear_1
                # 返回 add_3 作为输出结果
                return add_3

        # 创建 TestModule 类的实例
        m = TestModule()
        # 对模型进行符号跟踪，生成符号化的模型
        traced = symbolic_trace(m)
        # 生成一个维度为 4 的随机张量 a
        a = torch.rand(4)
        # 获取符号化图中所有节点的大小
        graph_manipulation.get_size_of_all_nodes(traced, [a])
        # 创建一个分区器实例
        partitioner = Partitioner()
        # 定义两个设备，每个设备有不同的设备 ID、内存大小和逻辑设备索引
        devices = [Device("dev_0", 120, 0), Device("dev_1", 160, 1)]
        # 使用设备和基于大小的分区模式创建分区器配置
        partitioner_config = PartitionerConfig(devices, PartitionMode.size_based)
        # 对符号化图进行分区
        ret = partitioner.partition_graph(traced, m, partitioner_config)
        # 获取分区后的模块和子模块
        module_with_submodules = ret.module_with_submodules
        # 获取分区后的有向无环图（DAG）
        dag = ret.dag
        # 断言符号化图在输入 a 上的输出与分区后模块在输入 a 上的输出相等
        self.assertEqual(traced(a), module_with_submodules(a))
        # 遍历 DAG 中的节点，检查逻辑设备 ID 是否正确分配
        for i, node in enumerate(dag.nodes):
            if i == 1:
                # 如果节点索引为 1，断言其逻辑设备 ID 为 [1]
                assert node.logical_device_ids == [1]
            else:
                # 其它情况下，断言其逻辑设备 ID 为 [0]
                assert node.logical_device_ids == [0]
    def test_sparse_nn_partition(self):
        # 定义一个名为 test_sparse_nn_partition 的测试方法
        class MyRecommendationModule(torch.nn.Module):
            # 定义一个推荐模块类 MyRecommendationModule，继承自 torch.nn.Module
            def create_mlp(self, num_of_layers: int, input_size: int, output_size: int):
                # 创建多层感知机（MLP）模型的方法
                layers = torch.nn.ModuleList()
                for _ in range(num_of_layers):
                    # 循环创建指定数量的线性层和激活函数 ReLU，并添加到 layers 中
                    ll = torch.nn.Linear(input_size, output_size)
                    layers.append(ll)
                    layers.append(torch.nn.ReLU())
                return layers

            def __init__(self):
                # 初始化方法
                super().__init__()
                # 创建底层 MLP 模型，包含 4 层，每层输入输出大小均为 4
                layers = self.create_mlp(4, 4, 4)
                self.bottom_layers = torch.nn.Sequential(*layers)
                # 创建顶层 MLP 模型，包含 3 层，每层输入输出大小均为 24
                layers = self.create_mlp(3, 24, 24)
                self.top_layers = torch.nn.Sequential(*layers)
                # 创建嵌入层列表
                self.embedding_layers = torch.nn.ModuleList()
                # 添加第一个嵌入层，词汇表大小为 500000，输出维度为 4，使用稀疏模式 "sum"
                el = torch.nn.EmbeddingBag(500000, 4, mode="sum", sparse=True)
                self.embedding_layers.append(el)
                # 循环添加三个嵌入层，词汇表大小为 1000000，输出维度为 4，使用稀疏模式 "sum"
                for i in range(3):
                    el = torch.nn.EmbeddingBag(1000000, 4, mode="sum", sparse=True)
                    self.embedding_layers.append(el)
                # 添加最后一个嵌入层，词汇表大小为 500000，输出维度为 4，使用稀疏模式 "sum"
                el = torch.nn.EmbeddingBag(500000, 4, mode="sum", sparse=True)
                self.embedding_layers.append(el)

            def forward(self, a, b, offset):
                # 前向传播方法
                # 应用底层 MLP 模型到输入 a
                x = self.bottom_layers(a)
                y = []
                c = []
                for i in range(len(self.embedding_layers)):
                    # 循环遍历所有嵌入层，生成随机张量并与 b 相加，添加到 c 中
                    temp = torch.randint(10, (8,))
                    c.append(temp + b)
                for i in range(len(self.embedding_layers)):
                    # 根据嵌入层的索引选择不同的操作
                    if i % 2 == 0:
                        y.append(self.embedding_layers[i](c[i], offset))
                    else:
                        y.append(
                            self.embedding_layers[i](torch.randint(10, (8,)), offset)
                        )
                # 将底层 MLP 输出 x 和所有嵌入层输出 y 连接在一起，按列连接
                z = torch.cat([x] + y, dim=1)
                # 应用顶层 MLP 模型到连接后的 z
                p = self.top_layers(z)
                return p

        m = MyRecommendationModule()
        a = torch.rand(2, 4)
        b = torch.randint(10, (8,))
        offset = torch.randint(1, (2,))
        traced = symbolic_trace(m)
        # 获取模型所有节点的大小
        graph_manipulation.get_size_of_all_nodes(traced, [a, b, offset])
        devices = [
            Device("dev_0", 33000000, 0),
            Device("dev_1", 33000000, 1),
            Device("dev_2", 33000000, 2),
        ]
        # 配置分区器，使用稀疏神经网络分区模式
        partitioner_config = PartitionerConfig(devices, PartitionMode.sparse_nn)
        partitioner = Partitioner()
        # 对模型进行图分区
        ret = partitioner.partition_graph(traced, m, partitioner_config)
        module_with_submodules = ret.module_with_submodules
        dag = ret.dag
        # 断言前向传播的结果与分区后的模型的前向传播结果相等
        self.assertEqual(traced(a, b, offset), module_with_submodules(a, b, offset))
        # 断言模型的图节点数量为 24
        assert len(module_with_submodules.graph.nodes) == 24
    # 定义一个测试方法，用于测试分区的延迟
    def test_partition_latency(self):
        # 定义一个测试用的神经网络模块
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)

            def forward(self, a):
                # 执行前向传播
                add_1 = a + torch.rand(4)  # 加上随机向量
                add_2 = add_1 + torch.rand(4)  # 再次加上随机向量
                linear_1 = self.linear(add_1)  # 线性层计算
                add_3 = add_2 + linear_1  # 加上线性层输出
                add_4 = add_2 + add_3  # 再次加上前一步结果
                return add_4

        # 定义一个函数，生成节点的延迟信息
        def get_node_to_latency_mapping(fx_module: GraphModule):
            """Given a fx module, generate node latency for each node
            based on the size of each node
            """
            node_to_latency_mapping: Dict[Node, NodeLatency] = {}
            for node in fx_module.graph.nodes:
                if node.op not in {"output", "placeholder", "get_attr"}:
                    if node.size_bytes.total_size == node.size_bytes.output_size:
                        node_to_latency_mapping[node] = NodeLatency(
                            node.size_bytes.total_size, 2.0 * node.size_bytes.total_size
                        )
                    else:
                        node_to_latency_mapping[node] = NodeLatency(
                            node.size_bytes.total_size, node.size_bytes.output_size
                        )
            return node_to_latency_mapping

        # 创建一个测试模块实例
        m = TestModule()
        # 对模块进行符号化跟踪
        traced = symbolic_trace(m)
        a = torch.rand(4)
        # 获取所有节点的大小信息
        graph_manipulation.get_size_of_all_nodes(traced, [a])
        # 获取节点到延迟映射
        node_to_latency_mapping = get_node_to_latency_mapping(traced)
        # 定义设备列表
        devices = [Device("dev_0", 200, 0), Device("dev_1", 200, 1)]
        # 创建分区器实例
        partitioner = Partitioner()
        # 定义分区器配置
        partitioner_config = PartitionerConfig(devices)
        # 对图进行分区
        ret = partitioner.partition_graph(traced, m, partitioner_config)
        # 获取带有子模块的模块
        module_with_submodules = ret.module_with_submodules
        # 断言跟踪模块输出与带子模块的模块输出相等
        self.assertEqual(traced(a), module_with_submodules(a))
        # 获取分区信息
        partitions = partitioner.partitions
        # 获取分区到延迟映射
        partition_to_latency_mapping = get_partition_to_latency_mapping(
            partitions, node_to_latency_mapping
        )
        # 遍历分区到延迟映射，检查每个分区的延迟
        for p in partition_to_latency_mapping:
            if p.partition_id == 0:
                assert partition_to_latency_mapping[p] == (128.0, 80.0, 160.0)
            else:
                assert partition_to_latency_mapping[p] == (16.0, 32.0, 32.0)
        # 定义传输速率
        transfer_rate_bytes_per_sec = 2
        # 获取分区图的关键路径延迟
        critical_path_latency_sec = get_latency_of_partitioned_graph(
            partitions, partition_to_latency_mapping, transfer_rate_bytes_per_sec
        )
        # 断言关键路径延迟等于预期值
        assert critical_path_latency_sec == 208.0
    # 定义一个测试方法，用于测试成本感知分区功能
    def test_cost_aware_partition(self):
        # 定义一个简单的神经网络模块，包含一个线性层
        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)

            def forward(self, a):
                # 在输入张量a上执行一系列操作
                add_1 = a + torch.rand(4)  # 加上随机噪声
                add_2 = add_1 + torch.rand(4)  # 再次加上随机噪声
                linear_1 = self.linear(add_1)  # 使用线性层处理add_1
                add_3 = add_2 + torch.rand(4)  # 再次加上随机噪声
                add_4 = add_2 + linear_1  # 将add_2和线性层的结果相加
                add_5 = add_3 + add_4  # 最终加总得到输出
                return add_5

        # 定义一个函数，从图模块中获取节点到延迟映射的字典
        def get_node_to_latency_mapping(fx_module: GraphModule):
            node_to_latency_mapping: Dict[Node, NodeLatency] = {}
            for node in fx_module.graph.nodes:
                # 排除输出节点、占位符和获取属性操作节点
                if node.op not in {"output", "placeholder", "get_attr"}:
                    if node.size_bytes.total_size == node.size_bytes.output_size:
                        # 若节点的输入大小等于输出大小，则延迟设置为1
                        node_to_latency_mapping[node] = NodeLatency(
                            node.size_bytes.total_size, 1
                        )
                    else:
                        # 否则将输入大小和输出大小作为延迟的设置
                        node_to_latency_mapping[node] = NodeLatency(
                            node.size_bytes.total_size, node.size_bytes.output_size
                        )
            return node_to_latency_mapping

        # 创建一个MyModule实例
        m = MyModule()
        # 对模块进行符号跟踪
        traced = symbolic_trace(m)
        # 创建一个大小为4的随机张量a
        a = torch.rand(4)
        # 获取跟踪后图模块的所有节点的大小信息
        graph_manipulation.get_size_of_all_nodes(traced, [a])
        # 创建四个设备对象的列表
        devices = [
            Device("dev_0", 125, 0),
            Device("dev_1", 125, 1),
            Device("dev_2", 125, 2),
            Device("dev_3", 125, 3),
        ]
        # 获取节点到延迟映射的字典
        node_to_latency_mapping = get_node_to_latency_mapping(traced)
        # 创建分区器的配置对象，使用成本感知分区模式
        partitioner_config = PartitionerConfig(
            devices,
            mode=PartitionMode.cost_aware,
            transfer_rate_bytes_per_sec=2,
            node_to_latency_mapping=node_to_latency_mapping,
        )
        # 创建分区器对象
        partitioner = Partitioner()
        # 执行图分区，并返回结果
        ret = partitioner.partition_graph(traced, m, partitioner_config)
        # 获取带有子模块的模块对象
        module_with_submodules = ret.module_with_submodules
        # 获取分区后的DAG图
        dag = ret.dag
        # 断言符号跟踪后的模块对随机输入a的输出与带子模块的模块对a的输出相等
        self.assertEqual(traced(a), module_with_submodules(a))
        # 获取分区器的所有分区
        partitions = partitioner.partitions
        # 获取分区到延迟映射的字典
        partition_to_latency_mapping = get_partition_to_latency_mapping(
            partitions, node_to_latency_mapping
        )
        # 获取分区化图的关键路径延迟（秒）
        critical_path_latency_sec = get_latency_of_partitioned_graph(
            partitions,
            partition_to_latency_mapping,
            partitioner_config.transfer_rate_bytes_per_sec,
        )
        # 断言关键路径的延迟为160.0秒
        assert critical_path_latency_sec == 160.0
    def test_aot_based_partition(self):
        # 定义一个继承自 torch.nn.Module 的测试模块类
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 初始化两个大小为4的随机张量作为模块的属性
                self.b = torch.rand(4)
                self.c = torch.rand(4)

            def forward(self, a):
                # 执行模块的前向传播，计算两次加法操作
                add_1 = a + self.b
                add_2 = self.c + add_1
                return add_2

        # 创建 TestModule 的实例
        m = TestModule()
        # 对模块进行符号化跟踪
        traced = symbolic_trace(m)
        # 创建一个大小为4的随机张量 a
        a = torch.rand(4)
        # 创建空的字典 node_to_partition_id 和 partition_to_logical_devices
        node_to_partition_id = {}
        partition_to_logical_devices = {}
        # 初始化计数器 count
        count = 0
        # 调用图操作函数，计算 traced 模块及其输入 a 的所有节点的大小
        graph_manipulation.get_size_of_all_nodes(traced, [a])
        # 遍历 traced 模块的所有节点
        for node in traced.graph.nodes:
            # 如果节点的操作不是占位符、获取属性或输出
            if node.op not in {"placeholder", "get_attr", "output"}:
                # 将节点映射到分区 ID，并将分区 ID 映射到逻辑设备列表
                node_to_partition_id[node] = count
                partition_to_logical_devices[count] = [0]
                count += 1
        # 创建一个包含单个逻辑设备的设备列表
        devices = [Device("dev_0", 200, 0)]
        # 创建分区器配置对象，使用 AOT（Ahead-Of-Time）模式进行分区
        partitioner_config = PartitionerConfig(
            devices=devices,
            mode=PartitionMode.aot_based,
            node_to_partition_mapping=node_to_partition_id,
            partition_to_logical_device_mapping=partition_to_logical_devices,
        )
        # 创建分区器实例
        partitioner = Partitioner()
        # 执行图分区操作，得到分区后的模块及子模块和分区 DAG
        ret = partitioner.partition_graph(traced, m, partitioner_config)
        # 获取分区后的模块及子模块
        module_with_submodules = ret.module_with_submodules
        # 获取分区后的 DAG（有向无环图）
        dag = ret.dag
        # 断言分区后的模块与原始 traced 模块在输入 a 上的输出一致
        self.assertEqual(module_with_submodules(a), traced(a))
        # 遍历 DAG 的所有节点
        for node in dag.nodes:
            # 断言每个节点的大小为48字节
            assert node.size_bytes == 48
            # 断言每个节点的逻辑设备 ID 为 [0]
            assert node.logical_device_ids == [0]

    def test_replace_target_nodes_with(self):
        # 定义一个简单的继承自 torch.nn.Module 的测试模块类
        class testModule(torch.nn.Module):
            def forward(self, a, b):
                # 执行模块的前向传播，返回输入张量 a 和 b 的加法结果
                return a + b

        # 创建 testModule 的实例
        m = testModule()
        # 对模块进行符号化跟踪
        traced = symbolic_trace(m)
        # 创建两个大小为1的随机输入张量 input1 和 input2
        input1 = torch.randn(1)
        input2 = torch.randn(1)
        # 断言 input1 和 input2 的加法结果与 traced 模块在这些输入下的输出一致
        assert (input1 + input2) == traced(input1, input2)
        # 使用图操作函数替换 traced 模块中的目标节点
        graph_manipulation.replace_target_nodes_with(
            fx_module=traced,
            old_op="call_function",
            old_target=operator.add,
            new_op="call_function",
            new_target=operator.mul,
        )
        # 断言 input1 和 input2 的乘法结果与替换节点后 traced 模块在这些输入下的输出一致
        assert (input1 * input2) == traced(input1, input2)
    # 定义一个测试方法，用于测试主机饱和功能
    def test_saturate_host(self):
        # 定义一个测试用的模块，包含一个线性层
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)

            def forward(self, a):
                # 执行前向传播，包括张量的加法操作
                add_1 = a + torch.rand(4)
                add_2 = add_1 + torch.rand(4)
                linear_1 = self.linear(add_1)
                add_3 = add_2 + linear_1
                add_4 = add_2 + add_3
                return add_4

        # 创建测试模块实例
        m = TestModule()
        # 对模块进行符号化跟踪
        traced = symbolic_trace(m)
        # 创建一个随机张量作为输入
        a = torch.rand(4)
        # 获取跟踪图中所有节点的大小信息
        graph_manipulation.get_size_of_all_nodes(traced, [a])
        # 定义多个设备
        devices = [
            Device("dev_0", 200, 0),
            Device("dev_1", 200, 1),
            Device("dev_2", 100, 2),
            Device("dev_3", 100, 3),
            Device("dev_4", 200, 4),
            Device("dev_5", 100, 5),
        ]
        # 创建一个分区器实例
        partitioner = Partitioner()
        # 定义分区器配置，包括开启主机饱和选项
        partitioner_config = PartitionerConfig(devices, saturate_host=True)
        # 对跟踪图进行分区
        ret = partitioner.partition_graph(traced, m, partitioner_config)
        # 获取带有子模块的分区模块
        module_with_submodules = ret.module_with_submodules
        # 断言跟踪模块的前向输出与分区后模块的前向输出相等
        self.assertEqual(traced(a), module_with_submodules(a))

        # 获取分区信息
        partitions = partitioner.partitions
        # 断言分区的数量为2
        self.assertEqual(len(partitions), 2)
        # 当开启主机饱和时，分区1会被复制到dev_4，分区2会被复制到dev_2
        self.assertEqual(partitions[0].logical_device_ids, [0, 4])
        self.assertEqual(partitions[1].logical_device_ids, [1, 2])

    # 如果没有TorchVision，跳过此测试
    @skipIfNoTorchVision
    def test_conv_bn_fusion(self):
        # 创建并加载一个ResNet-18模型，并设置为评估模式
        rn18 = resnet18().eval()
        # 对模型进行符号化跟踪
        traced = symbolic_trace(rn18)
        # 对跟踪后的模型进行优化，包括融合操作
        fused = optimization.fuse(traced)

        # 断言融合后的模型中不包含任何BatchNorm2d层
        self.assertTrue(
            all(not isinstance(m, torch.nn.BatchNorm2d) for m in fused.modules())
        )

        # 定义输入张量的形状
        N, C, H, W = 20, 3, 224, 224
        inp = torch.randn(N, C, H, W)

        # 断言融合后的模型与原始ResNet-18模型在给定输入下的输出相等
        self.assertEqual(fused(inp), rn18(inp))

    # 测试不处于运行状态的Conv-BN融合
    def test_conv_bn_fusion_not_running_state(self):
        # 定义一个包含卷积层和不跟踪运行状态的BatchNorm2d层的模块
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(32, 64, 3, stride=2)
                self.bn = torch.nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)

            def forward(self, x):
                x = self.conv(x)
                x = self.bn(x)
                return x

        # 创建模块实例并设置为评估模式
        model = M().eval()

        # 对模型进行符号化跟踪
        traced = symbolic_trace(model)
        # 对跟踪后的模型进行优化，包括融合操作
        fused = optimization.fuse(traced)
        # 创建一个随机输入张量
        inp = torch.randn([1, 32, 50, 50])

        # 断言融合后的模型中至少包含一个BatchNorm2d层
        self.assertTrue(
            any(isinstance(m, torch.nn.BatchNorm2d) for m in fused.modules())
        )
        # 断言融合后的模型与原始模型在给定输入下的输出相等
        self.assertEqual(fused(inp), model(inp))
    # 定义一个测试函数，用于测试混合数据类型的卷积和批量归一化融合情况
    def test_conv_bn_fusion_mixed_dtype(self):
        # 定义一个简单的神经网络模型类
        class M(torch.nn.Module):
            # 初始化函数，定义模型的结构
            def __init__(self):
                super().__init__()
                # 添加一个卷积层，输入通道为3，输出通道为16，使用3x3的卷积核，无偏置项，数据类型为torch.bfloat16
                self.conv = torch.nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False, dtype=torch.bfloat16)
                # 添加一个批量归一化层，输入通道为16，设置eps=0.001，momentum=0.1，affine=True，track_running_stats=True
                self.bn = torch.nn.BatchNorm2d(16, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

            # 前向传播函数，接收输入x，经过卷积和批量归一化后返回结果
            def forward(self, x):
                x = self.conv(x)  # 卷积操作
                x = self.bn(x)    # 批量归一化操作
                return x

        # 创建并评估模型
        model = M().eval()

        # 对模型进行符号跟踪
        traced = symbolic_trace(model)

        # 对跟踪后的模型进行优化融合
        fused = optimization.fuse(traced)

        # 创建一个输入张量，形状为[1, 3, 64, 64]，数据类型为torch.bfloat16
        inp = torch.randn(1, 3, 64, 64, dtype=torch.bfloat16)

        # 断言：确保融合后的模型中不含有torch.nn.BatchNorm2d模块
        self.assertTrue(
            all(not isinstance(m, torch.nn.BatchNorm2d) for m in fused.modules())
        )

        # 断言：确保优化后的模型的输出与原始模型的输出相等
        self.assertEqual(fused(inp), model(inp))

    # 定义一个测试函数，用于测试不带消息的assert语句
    def test_call_to_assert_no_msg(self):
        # 定义一个简单的神经网络模型类
        class M(torch.nn.Module):
            # 前向传播函数，接收输入a和b，执行assert a == b操作，并返回a + b的结果
            def forward(self, a, b):
                assert a == b
                return a + b

        # 创建模型实例
        m = M()

        # 对模型进行符号跟踪并使用重写
        traced = symbolic_trace_with_rewrite(m)

        # 确保跟踪后的图形式良好的（well-formed）
        traced.graph.lint()

        # 检查图形中是否存在一个call_function节点，其target为torch._assert
        self.assertTrue(
            any(
                node.op == "call_function" and node.target == torch._assert
                for node in traced.graph.nodes
            )
        )

        # 确保当断言应该抛出时，确实抛出异常；当不应该抛出时，不抛出异常
        traced(3, 3)
        with self.assertRaisesRegex(AssertionError, ""):
            traced(3, 5)

        # 确认输出结果正确
        self.assertEqual(traced(3, 3), m(3, 3))
    def test_meta_tracer(self):
        # 定义一个测试类 MetaTracerTestModule，继承自 torch.nn.Module
        class MetaTracerTestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 初始化一个嵌入层，num_embeddings=42，embedding_dim=16
                self.emb = torch.nn.Embedding(num_embeddings=42, embedding_dim=16)
                # 初始化一个 LayerNorm 层，输入维度为 16
                self.layernorm = torch.nn.LayerNorm(16)

            # 前向传播函数
            def forward(self, x):
                # 对输入 x 进行嵌入操作
                emb = self.emb(x)
                # 将嵌入结果与从0到 emb.shape[-1]-1 的序列相加，设备为 emb 所在设备
                emb = emb + torch.arange(emb.shape[-1], dtype=torch.float, device=emb.device)
                # 对加和结果进行 LayerNorm 处理
                lol = self.layernorm(emb)
                # 如果 lol 的第一维小于30，返回 ReLU 激活后的结果，否则返回 sigmoid 激活后的结果
                return torch.relu(lol) if lol.shape[0] < 30 else torch.sigmoid(lol)

        # 创建 MetaTracerTestModule 实例
        mttm = MetaTracerTestModule()
        
        # 针对不同的批次大小进行测试
        for BS in [15, 35]:
            # 创建输入张量 x，形状为 (BS,)
            x = torch.zeros(BS, dtype=torch.long).random_(42)
            # 构建 meta_args 字典，将 x 转移到 'meta' 设备上
            meta_args = {'x' : x.to(device='meta')}
            # 对 mttm 进行符号跟踪，使用 meta_args 作为元数据参数
            gm = torch.fx.experimental.meta_tracer.symbolic_trace(mttm, meta_args=meta_args)
            # 断言 gm(x) 与 mttm(x) 的结果近似相等
            torch.testing.assert_close(gm(x), mttm(x))

            # 测试序列化和反序列化
            with tempfile.TemporaryDirectory() as tmp_dir:
                # 将 gm 对象序列化到临时目录下的 meta_module.pkl 文件中
                with open(f'{tmp_dir}/meta_module.pkl', 'wb') as f:
                    pickle.dump(gm, f)

                # 从 meta_module.pkl 文件中反序列化出加载对象
                with open(f'{tmp_dir}/meta_module.pkl', 'rb') as f:
                    loaded = pickle.load(f)

                # 断言 loaded(x) 与 mttm(x) 的结果近似相等
                torch.testing.assert_close(loaded(x), mttm(x))


    def test_call_to_assert_with_msg(self):
        # 定义一个简单的 Module 类 M
        class M(torch.nn.Module):
            # 前向传播函数，进行断言 a == b
            def forward(self, a, b):
                assert a == b, "test message"
                return a + b

        # 创建 M 类的实例 m
        m = M()
        # 对 m 进行符号跟踪和重写
        traced = symbolic_trace_with_rewrite(m)

        # 确保图形是良好形成的
        traced.graph.lint()

        # 检查图形中是否有一个 call_function 节点，其目标为 torch._assert
        self.assertTrue(
            any(
                node.op == "call_function" and node.target == torch._assert
                for node in traced.graph.nodes
            )
        )

        # 确保断言在需要时抛出异常，在不需要时不抛出异常
        traced(3, 3)
        with self.assertRaisesRegex(AssertionError, "test message"):
            traced(3, 5)

        # 确认输出的正确性
        self.assertEqual(traced(3, 3), m(3, 3))
    def test_call_to_assert_with_empty_msg(self):
        # 定义一个内嵌的 torch.nn.Module 类 M，用于测试
        class M(torch.nn.Module):
            # 定义 forward 方法，实现模型的前向传播
            def forward(self, a, b):
                # 断言 a 等于 b，如果不等则抛出 AssertionError，空消息字符串表示错误消息为空
                assert a == b, ""
                # 返回 a + b 的结果
                return a + b

        # 创建 M 类的实例
        m = M()
        # 对模型进行符号化跟踪并重写
        traced = symbolic_trace_with_rewrite(m)

        # 确保生成的图形结构良好
        traced.graph.lint()

        # 检查图形中是否存在 target 为 torch._assert 的 call_function 节点
        self.assertTrue(
            any(
                node.op == "call_function" and node.target == torch._assert
                for node in traced.graph.nodes
            )
        )

        # 确保断言在应该抛出异常的情况下抛出异常，在不应该抛出异常的情况下不抛出异常
        traced(3, 3)
        with self.assertRaisesRegex(AssertionError, ""):
            traced(3, 5)

        # 确认输出结果的正确性
        self.assertEqual(traced(3, 3), m(3, 3))

    def test_call_to_assert_with_multiline_message(self):
        # 内嵌的 torch.nn.Module 类 M，用于测试多行消息的情况
        class M(torch.nn.Module):
            # 定义 forward 方法，实现模型的前向传播
            def forward(self, a, b):
                # 多行错误消息
                error_msg = """
        """
        确保函数的断言逻辑正确性

        确认 a 等于 b，否则引发 AssertionError 并返回 a + b
        """
        assert a == b, error_msg
        return a + b

m = M()
traced = symbolic_trace_with_rewrite(m)

# 确保图形表示形式的正确性
traced.graph.lint()

# 检查中间表示(IR)，确保存在一个 target 为 torch._assert 的 call_function 节点
self.assertTrue(
    any(
        node.op == "call_function" and node.target == torch._assert
        for node in traced.graph.nodes
    )
)

# 确保断言在预期时引发异常，在非预期时不引发异常
error_msg = """
An error message with
terrible spacing
"""
traced(3, 3)
with self.assertRaisesRegex(AssertionError, error_msg):
    traced(3, 5)

# 确认输出结果的正确性
self.assertEqual(traced(3, 3), m(3, 3))
    def test_subgraph_creation(self):
        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.param = torch.nn.Parameter(torch.rand(3, 4))  # 初始化一个参数张量
                self.linear = torch.nn.Linear(4, 5)  # 初始化一个线性层

            def forward(self, x, y):
                z = self.linear(x + self.param).clamp(min=0.0, max=1.0)  # 计算线性层输出并进行值范围限制
                w = self.linear(y).clamp(min=0.0, max=1.0)  # 计算另一个线性层输出并进行值范围限制
                return z + w  # 返回两个输出的和

        # 对模型进行符号化跟踪
        my_module = MyModule()
        my_module_traced = symbolic_trace(my_module)

        # 随机模块分区
        partition_counter = 0
        NPARTITIONS = 3

        # 添加一些随机的元信息以确保它被保留
        for node in my_module_traced.graph.nodes:
            if node.op != "output":
                node.meta["test_meta_info"] = True

        def mod_partition(node: Node):
            nonlocal partition_counter
            partition = partition_counter % NPARTITIONS
            partition_counter = (partition_counter + 1) % NPARTITIONS
            return partition

        # 将模块分割成具有子模块的模块
        module_with_submodules = split_module(
            my_module_traced, my_module, mod_partition
        )

        # 检查所有节点上是否仍然存在 test_meta_info
        submodules = dict(module_with_submodules.named_modules())
        for node in module_with_submodules.graph.nodes:
            if node.op == "call_module":
                submod = submodules[node.target]
                self.assertTrue(isinstance(submod, torch.fx.GraphModule))
                for submod_node in submod.graph.nodes:
                    if submod_node.op != "output":
                        stored_op = submod_node.meta.get("test_meta_info")
                        self.assertTrue(stored_op is not None and stored_op)

        x = torch.rand(3, 4)
        y = torch.rand(3, 4)

        orig_out = my_module_traced(x, y)
        submodules_out = module_with_submodules(x, y)

        self.assertEqual(orig_out, submodules_out)

    def test_split_module_dead_code(self):
        class ModWithDeadCode(torch.nn.Module):
            def forward(self, x):
                output = x * 2  # 我们需要这部分代码
                dead_line = x + 2  # 这部分代码是死代码
                return output

        mod = ModWithDeadCode()
        traced = torch.fx.symbolic_trace(mod)

        # 分割成前部分（0）、目标部分（1）和后部分（2）
        saw_mul = False

        def split_callback(n):
            nonlocal saw_mul
            if n.target == operator.mul:
                saw_mul = True
                return 1

            if not saw_mul:
                return 0
            if saw_mul:
                return 2

        split = split_module(traced, mod, split_callback)

        x = torch.randn((5,))
        torch.testing.assert_close(
            split(x), traced(x)
        )
    # 定义一个测试方法，用于测试模块参数扩展的情况
    def test_split_module_kwargs_expansion(self):
        # 定义一个继承自 torch.nn.Module 的类，实现 forward 方法以支持参数扩展
        class ModuleWithKwargsExpansion(torch.nn.Module):
            def forward(self, x, **kwargs):
                return x + kwargs['foo']

        # 创建 ModuleWithKwargsExpansion 类的实例
        mod = ModuleWithKwargsExpansion()
        # 对模块进行符号化追踪
        traced = torch.fx.symbolic_trace(mod)

        # 初始化一个标志，用于检测是否已经遇到了 getitem 操作
        seen_getitem = False

        # 定义一个回调函数，根据节点的目标操作来分割模块
        def split_callback(n):
            nonlocal seen_getitem
            # 如果当前节点的目标操作是 getitem，则将 seen_getitem 设置为 True
            if n.target == operator.getitem:
                seen_getitem = True
            # 返回分割的索引，这里根据 seen_getitem 的布尔值来确定
            return int(seen_getitem)

        # 调用 split_module 函数进行模块分割
        split = split_module(traced, mod, split_callback)

        # 创建输入张量 x 和 foo，然后使用 torch.testing.assert_close 来验证 split 的输出与 traced 的输出是否一致
        x = torch.randn(5, 3)
        foo = torch.randn(5, 3)
        torch.testing.assert_close(split(x, foo=foo), traced(x, foo=foo))

    # 如果有 TorchVision，执行一个测试子图将 ResNet 简单地分割成一个分区的情况
    @skipIfNoTorchVision
    def test_subgraph_trivial_resnet(self):
        # 烟雾测试，验证将 ResNet 简单分割为 1 个分区是否正常工作
        # 在此之前可能存在子模块名称别名的问题
        m = resnet18()
        # 对模型进行符号化追踪
        traced = symbolic_trace(m)
        # 创建输入张量 a
        a = torch.rand(64, 3, 7, 7)
        # 调用 split_module 函数进行模块分割
        module_with_submodules = split_module(traced, m, lambda node: 0)
        # 对分割后的模块进行调用
        module_with_submodules(a)

    # 测试带有默认参数的模块分割情况
    def test_split_module_default_arg(self):
        # 定义一个用于追踪的模型类 ModelToTrace
        class ModelToTrace(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = torch.nn.Linear(512, 512)

            def forward(self, x, targets=None):
                x = self.lin(x)

                if targets is not None:
                    x = x + targets

                return x

        # 创建 ModelToTrace 的实例
        mtt = ModelToTrace()
        # 对模型进行符号化追踪，使用 concrete_args 参数将 targets 设置为 None
        traced = torch.fx.symbolic_trace(mtt, concrete_args={'targets': None})

        # 调用 split_module 函数进行模块分割
        split = split_module(traced, mtt, lambda node: 0)

        # 创建输入张量 x
        x = torch.randn(50, 512)
        # 使用 torch.testing.assert_close 验证 split 的输出与 traced 的输出是否一致
        torch.testing.assert_close(split(x), traced(x))
    def test_normalize_binary_operators(self):
        # 要测试的操作集合，包括加法、乘法、减法等
        ops_to_test = {
            torch.add,
            torch.mul,
            torch.sub,
            torch.div,
            torch.floor_divide,
            torch.remainder,
            torch.eq,
            torch.ne,
            torch.lt,
            torch.le,
            torch.gt,
            torch.ge,
        }

        # 测试张量与张量之间的调用
        for op in ops_to_test:

            # 定义一个继承自torch.nn.Module的包装器类
            class WrapperMod(torch.nn.Module):
                # 定义前向传播函数，调用给定的操作op
                def forward(self, x, y):
                    return op(x, y)

            # 对WrapperMod进行符号跟踪
            traced = symbolic_trace(WrapperMod())
            # 对跟踪后的模型应用操作符规范化
            normalized = NormalizeOperators(traced).transform()
            x, y = torch.randn(3, 4), torch.randn(3, 4)
            # 断言跟踪前后的输出结果近似相等
            torch.testing.assert_close(traced(x, y), normalized(x, y))
            # 断言规范化后的图中不包含ops_to_test中的目标操作
            self.assertFalse(
                any(n.target in ops_to_test for n in normalized.graph.nodes)
            )

        # 测试张量与标量之间的调用
        for op in ops_to_test:

            # 定义一个继承自torch.nn.Module的包装器类
            class WrapperMod(torch.nn.Module):
                # 定义前向传播函数，调用给定的操作op
                def forward(self, x):
                    return op(x, 42)

            # 对WrapperMod进行符号跟踪
            traced = symbolic_trace(WrapperMod())
            # 对跟踪后的模型应用操作符规范化
            normalized = NormalizeOperators(traced).transform()
            x = torch.randn(3, 4)
            # 断言跟踪前后的输出结果近似相等
            torch.testing.assert_close(traced(x), normalized(x))
            # 断言规范化后的图中不包含ops_to_test中的目标操作
            self.assertFalse(
                any(n.target in ops_to_test for n in normalized.graph.nodes)
            )

    @skipIfNoTorchVision
    def test_normalize_args(self):
        # 加载ResNet18模型
        m = resnet18()

        # 定义一个继承自torch.fx.Tracer的功能追踪器类
        class FunctionalTracer(torch.fx.Tracer):
            # 判断是否为叶子模块的函数
            def is_leaf_module(
                self, m: torch.nn.Module, module_qualified_name: str
            ) -> bool:
                # `leaves`包含一组标准的`nn.Modules`，它们目前不能进行符号跟踪
                leaves = {torch.nn.BatchNorm2d}
                return type(m) in leaves

        # 对ResNet18模型进行符号跟踪
        traced = torch.fx.GraphModule(m, FunctionalTracer().trace(m))

        input = torch.randn(5, 3, 224, 224)
        ref_outs = traced(input)

        # 对跟踪后的模型应用形状传播
        ShapeProp(traced).propagate(input)
        # 对跟踪后的模型应用参数规范化
        traced = NormalizeArgs(traced).transform()

        # 获取所有模块的字典
        modules = dict(traced.named_modules())

        # 遍历跟踪后的图中的每个节点
        for node in traced.graph.nodes:
            if node.op == "call_function" and node.target != operator.add:
                # 如果节点的操作是"call_function"且目标不是加法操作，则断言其参数列表为空
                self.assertEqual(len(node.args), 0)
            elif node.op == "call_module":
                # 如果节点的操作是"call_module"
                submod_class = modules[node.target].__class__
                nn_class = getattr(torch.nn, submod_class.__name__)
                if submod_class == nn_class:
                    # 如果子模块类与torch.nn中对应的类相同，则断言其参数列表为空
                    self.assertEqual(len(node.args), 0)
        
        # 再次对模型进行输入测试，并断言输出结果与参考输出相等
        traced(input)
        self.assertEqual(traced(input), ref_outs)
    def test_normalize_modules_exhaustive(self):
        """
        Exhaustively test `Node.normalized_arguments` on all standard
        torch.nn Module classes
        """
        # 循环遍历 module_tests 和 new_module_tests 中的测试参数
        for test_params in module_tests + new_module_tests:
            # 如果 test_params 中没有 "constructor" 键，则从 torch.nn 中获取相应的类
            if "constructor" not in test_params:
                constructor = getattr(torch.nn, test_params["module_name"])
            else:
                constructor = test_params["constructor"]

            # 如果 test_params 中没有 "constructor_args" 键，则设定为空元组
            if "constructor_args" not in test_params:
                args = ()
            else:
                args = test_params["constructor_args"]

            # 使用 constructor 和 args 创建一个模块实例 mod
            mod = constructor(*args)

            # 如果 mod 的类名不在 torch.nn 的目录中，跳过当前模块的测试
            if mod.__class__.__name__ not in dir(torch.nn):
                continue

            # 如果 test_params 中没有 "input_fn" 键，则使用 torch.randn 生成输入数据
            if "input_fn" not in test_params:
                inputs = torch.randn(test_params["input_size"])
            else:
                inputs = test_params["input_fn"]()

            # 如果 inputs 不是 tuple 或 list，则转换为单元素元组
            if not isinstance(inputs, (tuple, list)):
                inputs = (inputs,)

            # 生成一个类来包装这个标准的 `nn.Module` 实例
            test_classname = f"Test{mod.__class__.__name__}"
            test_mod_code = f"""
class {test_classname}(torch.nn.Module):
    # 定义一个继承自 torch.nn.Module 的类 {test_classname}
    def __init__(self, mod):
        # 初始化方法，接受一个参数 mod
        super().__init__()
        # 调用父类的初始化方法
        self.mod = mod
        # 将传入的 mod 参数赋值给实例属性 self.mod

    def forward(self, {params}):
        # 定义 forward 方法，处理模型的前向传播逻辑，接受一个参数 {params}
        return self.mod({params})
        # 调用 self.mod 进行前向传播，并返回结果

    """
    gbls = {"torch": torch}
    exec(test_mod_code, gbls)

    test_instance = gbls[test_classname](mod)
    # 在全局变量 gbls 中执行 test_mod_code，创建一个 {test_classname} 类的实例 test_instance
    traced = symbolic_trace(test_instance)

    # Use `Node.normalized_arguments` to get a new set of arguments
    # to feed to the Module. Then, rewrite the node to only take
    # in those arguments as kwargs
    modules = dict(traced.named_modules())
    for node in traced.graph.nodes:
        # 遍历计算图中的每个节点
        if node.op == "call_module":
            # 如果节点的操作是调用模块
            submod_class = modules[node.target].__class__
            # 获取目标节点对应的模块类
            nn_class = getattr(torch.nn, submod_class.__name__)
            # 获取 torch.nn 中与目标子模块类同名的类
            if submod_class == nn_class:
                # 如果目标子模块类与 torch.nn 中同名类相同
                normalized_args = node.normalized_arguments(traced)
                # 使用 Node.normalized_arguments 方法获取标准化的参数
                normalized_args2 = normalize_module(
                    traced, node.target, node.args, node.kwargs
                )
                # 调用 normalize_module 函数对模块进行标准化处理
                assert normalized_args == normalized_args2
                # 断言标准化后的参数与处理后的参数相等
                assert normalized_args
                # 断言标准化后的参数存在
                node.args = normalized_args.args
                # 更新节点的位置参数
                node.kwargs = normalized_args.kwargs
                # 更新节点的关键字参数

    traced.recompile()
    # 重新编译跟踪后的模型

    # These Modules have an RNG in their forward, so testing
    # correctness by comparing outputs is not correct. Skip that
    # check for these
    stochastic_modules = {"FractionalMaxPool2d", "FractionalMaxPool3d", "RReLU"}

    if mod.__class__.__name__ not in stochastic_modules:
        # 如果模型不属于随机模块
        self.assertEqual(traced(*inputs), mod(*inputs))
        # 使用 self.assertEqual 检查跟踪后的模型输出与原始模型输出是否相等

    traced = NormalizeArgs(symbolic_trace(test_instance)).transform()
    # 对跟踪后的模型进行参数规范化处理

    modules = dict(traced.named_modules())
    for node in traced.graph.nodes:
        # 再次遍历计算图中的每个节点
        if node.op == "call_module":
            # 如果节点的操作是调用模块
            submod_class = modules[node.target].__class__
            # 获取目标节点对应的模块类
            nn_class = getattr(torch.nn, submod_class.__name__)
            # 获取 torch.nn 中与目标子模块类同名的类
            if submod_class == nn_class:
                # 如果目标子模块类与 torch.nn 中同名类相同
                self.assertEqual(len(node.args), 0)
                # 使用 self.assertEqual 检查节点的位置参数个数是否为 0
    # 定义一个测试方法，用于测试参数规范化并保留元数据
    def test_normalize_args_preserve_meta(self):
        # 定义一个简单的自定义神经网络模块
        class MyModule(torch.nn.Module):
            # 模块的前向传播方法，将输入张量 a 加上常数 3 并返回
            def forward(self, a):
                return torch.add(a, 3)

        # 创建 MyModule 的实例 m
        m = MyModule()
        # 对模块 m 进行符号化跟踪，生成一个符号化跟踪对象 traced
        traced = symbolic_trace(m)

        # 遍历符号化跟踪对象 traced 的图中的节点
        for node in traced.graph.nodes:
            # 如果节点是调用函数且目标函数是 torch.add
            if node.op == "call_function" and node.target == torch.add:
                # 在节点的元数据中添加键值对 "my_key": 7
                node.meta["my_key"] = 7
                break
        else:
            # 如果未找到目标节点，则测试失败，输出错误信息
            self.fail("Didn't find call_function torch.add")

        # 创建一个输入张量，形状为 (2, 3)
        input = torch.randn(2, 3)
        # 使用 ShapeProp 类对 traced 对象进行形状传播
        ShapeProp(traced).propagate(input)
        # 对 traced 对象进行参数规范化处理
        traced = NormalizeArgs(traced).transform()

        # 再次遍历处理后的 traced 对象的图中的节点
        for node in traced.graph.nodes:
            # 如果节点是调用函数且目标函数是 torch.add
            if node.op == "call_function" and node.target == torch.add:
                # 断言节点的元数据中存在键 "my_key"
                self.assertTrue("my_key" in node.meta)
                # 断言节点的元数据中 "my_key" 的值为 7
                self.assertEqual(node.meta["my_key"], 7)
                break
        else:
            # 如果未找到目标节点，则测试失败，输出错误信息
            self.fail("Didn't find call_function torch.add")

    # 定义一个测试方法，用于测试参数规范化并保留类型信息
    def test_normalize_args_perserve_type(self):
        # 定义一个简单的自定义神经网络模块，其前向传播方法接受一个列表，包含两个 torch.Tensor 元素
        class MyModule(torch.nn.Module):
            def forward(self, a: List[torch.Tensor]):
                return torch.add(a[0], a[1])

        # 创建 MyModule 的实例 m
        m = MyModule()
        # 对模块 m 进行符号化跟踪，生成一个符号化跟踪对象 traced
        traced = symbolic_trace(m)
        # 对 traced 对象进行参数规范化处理
        traced = NormalizeArgs(traced).transform()

        # 遍历处理后的 traced 对象的图中的节点
        for node in traced.graph.nodes:
            # 如果节点是占位符
            if node.op == "placeholder":
                # 断言节点的类型为 List[torch.Tensor]
                self.assertEqual(node.type, List[torch.Tensor])

    # 如果没有 TorchVision，跳过这个测试
    @skipIfNoTorchVision
    # 定义测试函数，验证带有模式的返回值
    def test_annotate_returns_with_schema(self):
        # 创建一个 ResNet-18 模型实例
        m = resnet18()

        # 对模型进行符号化追踪
        traced_modules = symbolic_trace(m)
        
        # 使用 AnnotateTypesWithSchema 对象转换追踪到的模型，并获得带有模式的追踪模型
        traced_modules_annotated = AnnotateTypesWithSchema(traced_modules).transform()
        
        # 遍历带有模式的追踪模型的图中的节点
        for node in traced_modules_annotated.graph.nodes:
            if node.type is None:
                # 检查节点类型和目标，确保它们在预期的集合中
                check = (node.op, node.target)
                self.assertIn(
                    check,
                    {
                        ("placeholder", "x"),
                        ("call_module", "maxpool"),
                        ("call_function", operator.add),
                        ("call_function", torch.flatten),
                        ("output", "output"),
                    }
                )

        # 对带有模式的追踪模型进行 TorchScript 编译的烟雾测试，因为现在我们生成了类型注解

        torch.jit.script(traced_modules_annotated)

        # 定义 FunctionalTracer 类，继承自 torch.fx.Tracer
        class FunctionalTracer(torch.fx.Tracer):
            # 判断模块是否为叶子模块的方法
            def is_leaf_module(
                self, m: torch.nn.Module, module_qualified_name: str
            ) -> bool:
                # `leaves` 包含了一些标准的 `nn.Module`，它们不能被符号化追踪
                leaves = {torch.nn.BatchNorm2d}
                return type(m) in leaves

        # 使用 FunctionalTracer 对象追踪模型 m，并创建带有图模型的 GraphModule 实例
        traced_functionals = torch.fx.GraphModule(m, FunctionalTracer().trace(m))

        # 使用 AnnotateTypesWithSchema 对象转换追踪到的函数模型，并获得带有模式的函数模型
        traced_functionals_annotated = AnnotateTypesWithSchema(
            traced_functionals
        ).transform()

        # 遍历带有模式的函数模型的图中的节点
        for node in traced_functionals_annotated.graph.nodes:
            if node.type is None:
                # 检查节点类型和目标，确保它们在预期的集合中
                check = (node.op, node.target)
                excluded_nodes = {
                    ("placeholder", "x"),
                    # 根据布尔分发返回类型可能有所不同 :(
                    ("call_function", torch.nn.functional.max_pool2d),
                    ("output", "output"),
                }
                # AnnotateTypesWithSchema 不能处理绑定的 C++ 函数
                if not isinstance(node.target, BuiltinFunctionType):
                    self.assertIn(check, excluded_nodes)

        # 对带有模式的函数模型进行 TorchScript 编译的烟雾测试，因为现在我们生成了类型注解

        torch.jit.script(traced_functionals_annotated)
    def test_annotate_getitem_node(self):
        # 定义一个自定义类型 CustomType
        class CustomType:
            pass

        # 定义一个带命名字段的命名元组 CustomNamedTuple
        class CustomNamedTuple(NamedTuple):
            x: int
            y: float

        # 定义一个继承自 torch.nn.Module 的自定义模块 MyModule
        class MyModule(torch.nn.Module):
            # 定义模块的前向传播函数
            def forward(self, inp: Tuple[CustomType, torch.Tensor], inp2: List[CustomType], inp3: CustomNamedTuple):
                # 从输入元组中获取第一个元素 inp[0]
                inp_0 = inp[0]
                # 从输入元组中获取第二个元素 inp[1]
                inp_1 = inp[1]
                # 从输入列表中获取第一个元素 inp2[0]
                inp2_0 = inp2[0]
                # 从输入命名元组中获取字段 x
                inp3_x = inp3.x
                # 从输入命名元组中获取字段 y
                inp3_y = inp3.y
                # 返回各个输入的求和结果
                return inp_0 + inp_1 + inp2_0 + inp3_x + inp3_y

        # 创建 MyModule 类的实例 my_module
        my_module = MyModule()
        # 对 my_module 进行符号化跟踪
        my_module_traced = torch.fx.symbolic_trace(my_module)

        # 默认情况下，fx 转换会丢失 getitem 节点的类型注释
        for node in my_module_traced.graph.nodes:
            if node.target == operator.getitem:
                assert node.type is None

        # 添加注释到 getitem 节点
        annotate_getitem_nodes(my_module_traced.graph)

        # 验证每个 getitem 节点是否已经被正确注释
        for node in my_module_traced.graph.nodes:
            if node.target == operator.getitem:
                self.assertIsNotNone(node.type, f"Node {node} should be annotated but is not.")

    def test_subgraph_uniquename(self):
        # 定义一个继承自 torch.nn.Module 的自定义模块 MyModule
        class MyModule(torch.nn.Module):
            # 定义模块的初始化函数
            def __init__(self):
                super().__init__()
                # 添加一个线性层 self.linear
                self.linear = torch.nn.Linear(4, 4)

            # 定义模块的前向传播函数
            def forward(self, a, b, c, d):
                # 计算 a 和 b 的和
                add_1 = a + b
                # 计算 add_1 和 c 的和
                add_2 = add_1 + c
                # 对 add_1 应用线性层 self.linear
                linear_1 = self.linear(add_1)
                # 计算 add_2 和 d 的和
                add_3 = add_2 + d
                # 计算 add_2 和 linear_1 的和
                add_4 = add_2 + linear_1
                # 计算 add_3 和 add_4 的和
                add_5 = add_3 + add_4
                # 返回 add_5 的结果
                return add_5

        # 创建四个张量 a, b, c, d，每个张量的元素均为 1
        a, b, c, d = torch.ones(4), torch.ones(4), torch.ones(4), torch.ones(4)
        # 创建 MyModule 类的实例 mm
        mm = MyModule()
        # 对 mm 进行符号化跟踪
        traced = symbolic_trace(mm)

        # 定义一个回调函数 split_cb，用于确定节点分组策略
        def split_cb(node: torch.fx.Node):
            if node.name == "a" or node.name == "b" or node.name == "add":
                return 0
            else:
                return 1

        # 使用 split_cb 函数对 traced 模块进行分组，得到 module_with_submodule
        module_with_submodule = split_module(traced, mm, split_cb)
        # 验证 module_with_submodule 的前向传播结果与 traced 模块的前向传播结果一致
        self.assertEqual(module_with_submodule(a, b, c, d), traced(a, b, c, d))
    def test_split_qualname_mapping(self):
        # 定义隐藏单元的维度
        d_hid = 4

        # 定义一个示例模块 ExampleCode，继承自 torch.nn.Module
        class ExampleCode(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 定义两个随机初始化的参数矩阵作为模型的参数
                self.mm_param = torch.nn.Parameter(torch.randn(d_hid, d_hid))
                self.mm_param2 = torch.nn.Parameter(torch.randn(d_hid, d_hid))
                # 定义一个线性层
                self.lin = torch.nn.Linear(d_hid, d_hid)

            def forward(self, x):
                # 模型的前向传播逻辑
                x = torch.mm(x, self.mm_param)  # 矩阵乘法
                x = torch.relu(x)  # ReLU 激活函数
                x = torch.mm(x, self.mm_param)  # 矩阵乘法
                x = self.lin(x)  # 线性层
                x = torch.relu(x)  # ReLU 激活函数
                x = torch.mm(x, self.mm_param2)  # 矩阵乘法
                x = self.lin(x)  # 线性层
                return x

        # 创建 ExampleCode 类的实例
        my_module = ExampleCode()
        # 对模型进行符号化跟踪
        my_module_traced = symbolic_trace(my_module)

        # 初始化分割索引
        part_idx = 0

        # 定义分割回调函数，用于标识模块中的线性层并增加分割索引
        def split_callback(n : torch.fx.Node):
            nonlocal part_idx
            # 当节点操作为调用模块且目标为 'lin' 时，增加分割索引
            if (n.op, n.target) == ('call_module', 'lin'):
                part_idx += 1
            return part_idx

        # 在模块中进行子模块分割
        qualname_map : Dict[str, str] = {}
        module_with_submodules = split_module(
            my_module_traced, my_module, split_callback, qualname_map
        )
        # 预期的全限定名称映射
        expected_qualname_map = {
            'submod_1.lin': 'lin', 'submod_2.lin': 'lin'
        }
        # 断言全限定名称映射与预期相同
        self.assertEqual(qualname_map, expected_qualname_map)

    def test_traceable_function_with_nonstandard_name(self):
        # 定义一个函数 foo，对输入应用 ReLU 激活函数并返回结果
        def foo(x):
            return torch.relu(x)

        # 对函数进行符号化跟踪并进行重写
        traced = symbolic_trace_with_rewrite(foo)
    def test_to_folder(self):
        # 定义一个测试类 Test，继承自 torch.nn.Module
        class Test(torch.nn.Module):
            # 初始化方法
            def __init__(self):
                super().__init__()
                # 定义一个可学习参数 W，形状为 (2,)
                self.W = torch.nn.Parameter(torch.randn(2))
                # 定义一个序列模块，包含 BatchNorm1d 层
                self.seq = torch.nn.Sequential(torch.nn.BatchNorm1d(2, 2))
                # 定义一个线性层，输入维度为 2，输出维度为 2
                self.linear = torch.nn.Linear(2, 2)
                # 定义一个非参数化的属性 attr，形状为 (2,)
                self.attr = torch.randn(2)
                # 注册一个缓冲属性 attr2，形状为 (2,)
                self.register_buffer("attr2", torch.randn(2))
                # 注册一个缓冲属性 attr3，形状为 (2,)，数据类型为 torch.int32
                self.register_buffer("attr3", torch.ones(2, dtype=torch.int32))

            # 前向传播方法
            def forward(self, x):
                # 计算前向传播，应用序列模块和线性层到输入 x 上
                return self.linear(self.seq(self.W + self.attr + self.attr2 + self.attr3 + x))

        # 对 Test 类进行符号跟踪
        mod = symbolic_trace(Test())
        # 模块名称设为 "Foo"
        module_name = "Foo"
        # 导入临时文件和路径类 Path
        import tempfile
        from pathlib import Path

        # 使用临时目录进行上下文管理
        with tempfile.TemporaryDirectory() as tmp_dir:
            # 将临时目录路径转换为 Path 对象
            tmp_dir = Path(tmp_dir)
            # 将符号跟踪得到的模块保存到临时目录下，名称为 module_name
            mod.to_folder(tmp_dir, module_name)

            # 导入 importlib.util 用于加载模块
            import importlib.util

            # 根据临时目录下的 "__init__.py" 文件创建模块的规范
            spec = importlib.util.spec_from_file_location(
                module_name, tmp_dir / "__init__.py"
            )
            # 根据规范加载模块
            module = importlib.util.module_from_spec(spec)
            # 将加载的模块添加到 sys.modules 中
            sys.modules[module_name] = module
            # 执行加载的模块
            spec.loader.exec_module(module)

            # 创建一个形状为 (2, 2) 的随机张量 t
            t = torch.randn(2, 2)
            # 断言模块中的 Foo 类在输入 t 时与符号跟踪的模块 mod 在 t 上的输出相同
            self.assertEqual(module.Foo()(t), mod(t))

    def test_fetch(self):
        # 定义一个字典，包含用于降低操作的属性列表
        attrs_for_lowering: Dict[str, List[str]] = {
            "torch.nn.modules.conv.Conv2d": [
                "weight",
                "bias",
                "kernel_size",
                "stride",
                "padding",
                "dilation",
                "groups",
                "padding_mode",
            ],
            "torch.nn.modules.batchnorm.BatchNorm2d": [
                "weight",
                "bias",
                "running_mean",
                "running_var",
                "eps",
            ],
        }

        # 定义一个测试模块 TestModule，继承自 torch.nn.Module
        class TestModule(torch.nn.Module):
            # 初始化方法
            def __init__(self):
                super().__init__()
                # 定义一个 3 输入、3 输出、2 核大小的卷积层
                self.conv = torch.nn.Conv2d(3, 3, 2)
                # 定义一个 3 通道的 BatchNorm2d 层
                self.bn = torch.nn.BatchNorm2d(3)

            # 前向传播方法
            def forward(self, a):
                # 应用卷积层到输入 a 上
                a = self.conv(a)
                # a 自加，再应用 BatchNorm2d 层到结果上
                a += a
                return self.bn(a)

        # 创建 TestModule 类的实例 mod
        mod = TestModule()
        # 对 mod 进行符号跟踪
        traced = symbolic_trace(mod)
        # 将下降属性提升到节点
        lift_lowering_attrs_to_nodes(traced)

        # 遍历符号跟踪后的图中的每个节点
        for node in traced.graph.nodes:
            # 如果节点的操作是 "call_module"
            if node.op == "call_module":
                # 断言节点的 attrs_for_lowering 属性存在
                assert hasattr(node, "attrs_for_lowering")
                # 获取节点 attrs_for_lowering 字典中的类名对应的参数列表
                para_list = attrs_for_lowering[node.attrs_for_lowering["name"]]

                # 断言节点 attrs_for_lowering 字典中的键值对数量比 para_list 多一个（包含类名）
                assert len(para_list) + 1 == len(node.attrs_for_lowering)
                # 遍历 para_list 中的每个参数名
                for p_name in para_list:
                    # 断言参数名存在于节点 attrs_for_lowering 字典中
                    assert p_name in node.attrs_for_lowering
    # 定义一个单元测试方法，用于测试类型匹配函数的行为
    def test_type_matches(self):
        # 预期匹配的类型对，每对包含两个类型对象
        should_be_equal = [
            (int, int),  # 整数与整数，应该相等
            (numbers.Number, int),  # 数字类型与整数，应该相等
            (numbers.Number, float),  # 数字类型与浮点数，应该相等
            (int, type(torch.float)),  # 整数与 Torch 浮点数类型，应该相等
            (Union[int, float], int),  # 整数或浮点数与整数，应该相等
            (Union[int, float], float),  # 整数或浮点数与浮点数，应该相等
            (List[int], int),  # 整数列表与整数，应该相等
            (List[int], create_type_hint([int, int])),  # 整数列表与整数列表类型提示，应该相等
            (List[int], create_type_hint((int, int))),  # 整数列表与整数元组类型提示，应该相等
            (List[torch.Tensor], create_type_hint([torch.Tensor, torch.Tensor])),  # Torch 张量列表与 Torch 张量列表类型提示，应该相等
            (
                List[torch.Tensor],
                create_type_hint([torch.nn.Parameter, torch.nn.Parameter]),  # Torch 张量列表与 Torch 参数列表类型提示，应该相等
            ),
            (torch.Tensor, torch.nn.Parameter),  # Torch 张量与 Torch 参数，应该相等
            (List[torch.Tensor], create_type_hint([torch.nn.Parameter, torch.Tensor])),  # Torch 张量列表与 Torch 参数与张量列表类型提示，应该相等
            (List[torch.Tensor], create_type_hint([torch.Tensor, torch.nn.Parameter])),  # Torch 张量列表与 Torch 张量与参数类型提示，应该相等
            (List[torch.Tensor], create_type_hint((torch.Tensor, torch.Tensor))),  # Torch 张量列表与 Torch 张量元组类型提示，应该相等
            (
                List[torch.Tensor],
                create_type_hint((torch.nn.Parameter, torch.nn.Parameter)),  # Torch 张量列表与 Torch 参数元组类型提示，应该相等
            ),
            (torch.Tensor, torch.nn.Parameter),  # Torch 张量与 Torch 参数，应该相等
            (List[torch.Tensor], create_type_hint((torch.nn.Parameter, torch.Tensor))),  # Torch 张量列表与 Torch 参数与张量元组类型提示，应该相等
            (List[torch.Tensor], create_type_hint((torch.Tensor, torch.nn.Parameter))),  # Torch 张量列表与 Torch 张量与参数元组类型提示，应该相等
            (Optional[List[torch.Tensor]], List[torch.Tensor]),  # 可选的 Torch 张量列表与 Torch 张量列表，应该相等
            (Optional[List[int]], List[int]),  # 可选的整数列表与整数列表，应该相等
        ]
        # 对于每一对类型，验证类型匹配函数返回 True
        for sig_type, arg_type in should_be_equal:
            self.assertTrue(type_matches(sig_type, arg_type))

        # 预期不匹配的类型对，每对包含两个类型对象
        should_fail = [
            (int, float),  # 整数与浮点数，应该不相等
            (Union[int, float], str),  # 整数或浮点数与字符串，应该不相等
            (List[torch.Tensor], List[int]),  # Torch 张量列表与整数列表，应该不相等
        ]

        # 对于每一对类型，验证类型匹配函数返回 False
        for sig_type, arg_type in should_fail:
            self.assertFalse(type_matches(sig_type, arg_type))

    # 如果没有 MKLDNN，跳过这个测试
    @skipIfNoMkldnn
    def test_optimize_for_inference_cpu(self):
        # 导入 PyTorch 的神经网络模块
        import torch.nn as nn

        # 定义一个自定义的神经网络模块 Foo
        class Foo(nn.Module):
            def __init__(self):
                super().__init__()
                layers = []
                layers2 = []
                # 创建 10 层卷积神经网络模型和对应的 Batch Normalization 层和 ReLU 激活函数层
                for _ in range(10):
                    layers.append(nn.Conv2d(3, 3, 1))
                    layers.append(nn.BatchNorm2d(3))
                    layers.append(nn.ReLU())

                    layers2.append(nn.Conv2d(3, 3, 1))
                    layers2.append(nn.BatchNorm2d(3))
                    layers2.append(nn.ReLU())
                # 将定义好的网络层序列化为 nn.Sequential 模型
                self.model = nn.Sequential(*layers)
                self.model2 = nn.Sequential(*layers2)

            def forward(self, x):
                # 前向传播函数，返回两个模型的输出之和
                return self.model(x) + self.model2(x)

        # 定义输入张量的维度
        N, C, H, W, = (
            1,
            3,
            224,
            224,
        )
        # 生成指定维度的随机输入张量
        inp = torch.randn(N, C, H, W)
        # 在不计算梯度的上下文中
        with torch.no_grad():
            # 创建并评估 Foo 模型
            model = Foo().eval()
            # 对模型进行推理优化
            optimized_model = optimization.optimize_for_inference(model)
            # 断言优化前后的模型在给定输入下输出近似相等
            torch.testing.assert_close(model(inp), optimized_model(inp))

            # 使用自定义配置进行推理优化
            optimized_model2 = optimization.optimize_for_inference(
                model, pass_config={"remove_dropout": False}
            )
            # 断言优化前后的模型在给定输入下输出近似相等
            torch.testing.assert_close(model(inp), optimized_model2(inp))

    @skipIfNoTorchVision
    @skipIfNoMkldnn
    def test_optimize_for_inference_cpu_torchvision(self):
        # 导入 TorchVision 模块
        models = [
            torchvision.models.resnet18,
            torchvision.models.resnet50,
            torchvision.models.densenet121,
            torchvision.models.shufflenet_v2_x1_0,
            torchvision.models.vgg16,
            torchvision.models.mobilenet_v2,
            torchvision.models.mnasnet1_0,
            torchvision.models.resnext50_32x4d,
        ]
        # 在不计算梯度的上下文中
        with torch.no_grad():
            # 遍历 TorchVision 中的模型列表
            for model_type in models:
                # 创建指定类型的模型实例
                model = model_type()
                # 定义输入张量的维度
                C, H, W, = (
                    3,
                    224,
                    224,
                )
                # 生成指定维度的随机输入张量
                inp = torch.randn(3, C, H, W)
                # 将输入张量传递给模型，执行前向传播
                model(inp)
                # 设置模型为评估模式
                model.eval()
                # 重新生成指定维度的随机输入张量
                inp = torch.randn(1, C, H, W)
                # 生成 MKL 自动调优的启发式配置
                heuristic = optimization.gen_mkl_autotuner(inp, iters=0, warmup=0)
                # 对模型进行推理优化
                optimized_model = optimization.optimize_for_inference(model)

                # 计算模型在原始输入下的输出
                orig_out = model(inp)
                # 计算优化后模型在相同输入下的输出
                new_out = optimized_model(inp)
                # 断言优化前后的模型在给定输入下输出近似相等
                torch.testing.assert_close(orig_out, new_out)
# 定义一个名为 TestNormalizeOperators 的测试类，继承自 JitTestCase
class TestNormalizeOperators(JitTestCase):
    
    # 装饰器，标记只在 CPU 上运行的测试用例
    @onlyCPU
    # 装饰器，标记允许使用的操作和数据类型为 torch.float
    @ops(op_db, allowed_dtypes=(torch.float,))
    
# 定义一个名为 TestModule 的类，继承自 torch.nn.Module
class TestModule(torch.nn.Module):
    
    # 定义前向传播方法
    def forward(self, {', '.join(param_names)}):
        # 返回执行指定操作（op.name）的 torch 张量
        return torch.{op.name}({', '.join(fx_args)})
            """
            
            # 创建全局变量 g，包含 torch 和 math.inf
            g = {"torch": torch, "inf": math.inf}
            # 在全局环境中执行给定的代码块（code），将结果存储在 g 中
            exec(code, g)
            # 从 g 中获取名为 TestModule 的对象
            TestModule = g["TestModule"]

            # 创建 TestModule 的实例 m
            m = TestModule()
            # 对 m 进行符号跟踪，得到 traced 对象
            traced = torch.fx.symbolic_trace(m)
            # 使用 traced 对象执行前向传播，并将结果存储在 ref_out 中
            ref_out = traced(*param_values)

            # 遍历 traced 图中的每个节点
            for node in traced.graph.nodes:
                # 如果节点的操作为 "call_function"
                if node.op == "call_function":
                    # 根据给定的参数类型（arg_types, kwarg_types）规范化节点的参数
                    normalized_args = node.normalized_arguments(
                        traced, arg_types, kwarg_types
                    )
                    # 断言规范化后的参数不为空
                    assert normalized_args
                    # 更新节点的位置参数为规范化后的参数位置参数
                    node.args = normalized_args.args
                    # 更新节点的关键字参数为规范化后的关键字参数
                    node.kwargs = normalized_args.kwargs
            
            # 重新编译 traced 对象
            traced.recompile()

            # 使用规范化后的参数执行 traced 对象的前向传播，并将结果存储在 test_out 中
            test_out = traced(*param_values)
            # 断言 test_out 等于 ref_out
            self.assertEqual(test_out, ref_out)

    # 定义测试规范化量化嵌入包函数的方法
    def test_normalize_quantized_eb(self):
        # 目标操作为 torch.ops.quantized.embedding_bag_byte_rowwise_offsets
        target = torch.ops.quantized.embedding_bag_byte_rowwise_offsets
        # 定义参数 args
        args = (
            torch.empty((2, 3), dtype=torch.uint8),
            torch.empty((2,), dtype=torch.int64),
            torch.empty((2,), dtype=torch.int64),
        )
        # 调用 normalize_function 函数，规范化 target 和 args，并仅使用关键字参数
        norm_args_and_kwargs = normalize_function(
            target, args, normalize_to_only_use_kwargs=True
        )
        # 断言 norm_args_and_kwargs 不为空
        self.assertTrue(norm_args_and_kwargs is not None)
        # 断言规范化后的关键字参数包含特定的键集合
        self.assertEqual(
            set(norm_args_and_kwargs.kwargs.keys()),
            {
                "weight",
                "indices",
                "offsets",
                "scale_grad_by_freq",
                "mode",
                "pruned_weights",
                "per_sample_weights",
                "compressed_indices_mapping",
                "include_last_offset",
            },
        )
        # 断言规范化后的位置参数为空元组
        self.assertEqual(norm_args_and_kwargs.args, tuple())

    # 定义测试规范化参数重载操作的方法
    def test_normalize_args_op_overload(self):
        # 遍历目标操作列表
        for target in [torch.ops.aten.resize_as_.default, torch.ops.aten.resize_as_]:
            # 创建随机张量 inp1 和 inp2
            inp1 = torch.rand([1])
            inp2 = torch.rand([4])
            # 调用 normalize_function 函数，规范化 target 和参数（inp1），使用给定的关键字参数
            args, kwargs = normalize_function(target, (inp1,), {"the_template": inp2}, normalize_to_only_use_kwargs=True)
            # 断言 kwargs 的 "input" 键是 inp1
            self.assertIs(kwargs["input"], inp1)
            # 断言 kwargs 的 "the_template" 键是 inp2


# 如果 TEST_Z3 为真，则导入 z3 模块及其他相关模块
if TEST_Z3:
    import z3
    import torch._dynamo.config
    from torch.fx.experimental.validator import SympyToZ3, TranslationValidator, ValidationException, z3str
    from torch.utils._sympy.functions import FloorDiv, Mod

# 实例化设备类型测试，将 TestNormalizeOperators 类的测试方法添加到全局作用域
instantiate_device_type_tests(TestNormalizeOperators, globals())

# 如果当前脚本作为主程序执行，则运行测试
if __name__ == "__main__":
    run_tests()
```