# `.\pytorch\test\fx\test_source_matcher_utils.py`

```py
# Owner(s): ["module: fx"]

# 导入必要的模块
import os
import sys
import unittest

import torch

# 获取当前测试文件的父目录并添加到系统路径中
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)

# 导入需要测试的模块和函数
from torch._dynamo.eval_frame import is_dynamo_supported
from torch.fx.passes.tools_common import legalize_graph
from torch.fx.passes.utils.source_matcher_utils import (
    check_subgraphs_connected,
    get_source_partitions,
)
from torch.testing._internal.jit_utils import JitTestCase


class TestSourceMatcher(JitTestCase):
    @unittest.skipIf(not is_dynamo_supported(), "Dynamo not supported")
    def test_module_partitioner_linear_relu_linear(self):
        # 定义一个简单的神经网络模型类
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 定义模型的层结构
                self.linear1 = torch.nn.Linear(3, 3)
                self.relu = torch.nn.ReLU()
                self.linear2 = torch.nn.Linear(3, 5)

            def forward(self, x):
                # 定义前向传播过程
                x = self.linear1(x)
                x = self.linear1(x)  # 这里可能是拼写错误，应该是 self.linear2(x)
                x = self.relu(x)
                x = self.linear2(x)
                return x

        inputs = (torch.randn(3, 3),)  # 定义输入数据
        # 使用 Torch FX 功能导出模型的图表示，并获取运行时图和静态图
        gm, _ = torch._dynamo.export(M(), aten_graph=True)(*inputs)
        gm.graph.eliminate_dead_code()  # 消除图中的死代码

        # 获取模型图中线性层和ReLU层的源分区
        module_partitions = get_source_partitions(
            gm.graph, [torch.nn.Linear, torch.nn.ReLU]
        )

        # 断言模型分区的数量和每个分区中层的数量
        self.assertEqual(len(module_partitions), 2)
        self.assertEqual(len(module_partitions[torch.nn.Linear]), 3)
        self.assertEqual(len(module_partitions[torch.nn.ReLU]), 1)

        # 检查线性层和ReLU层之间的子图连接性
        self.assertFalse(
            check_subgraphs_connected(
                module_partitions[torch.nn.Linear][0],
                module_partitions[torch.nn.ReLU][0],
            )
        )
        self.assertTrue(
            check_subgraphs_connected(
                module_partitions[torch.nn.Linear][1],
                module_partitions[torch.nn.ReLU][0],
            )
        )
        self.assertFalse(
            check_subgraphs_connected(
                module_partitions[torch.nn.Linear][2],
                module_partitions[torch.nn.ReLU][0],
            )
        )

    @unittest.skipIf(not is_dynamo_supported(), "Dynamo not supported")
    # 定义一个测试函数，用于测试模块分区的卷积-ReLU-最大池化模型
    def test_module_partitioner_conv_relu_maxpool(self):
        # 定义一个继承自torch.nn.Module的内部类M，用于构建模型
        class M(torch.nn.Module):
            def __init__(self, constant_tensor: torch.Tensor) -> None:
                super().__init__()
                self.constant_tensor = constant_tensor  # 初始化常量张量
                # 定义三个卷积层，分别包含输入通道数、输出通道数、卷积核大小和填充数
                self.conv1 = torch.nn.Conv2d(
                    in_channels=3, out_channels=16, kernel_size=3, padding=1
                )
                self.conv2 = torch.nn.Conv2d(
                    in_channels=16, out_channels=16, kernel_size=3, padding=1
                )
                self.conv3 = torch.nn.Conv2d(
                    in_channels=16, out_channels=16, kernel_size=3, padding=1
                )
                self.relu = torch.nn.ReLU()  # 定义ReLU激活函数层
                self.maxpool = torch.nn.MaxPool2d(kernel_size=3)  # 定义最大池化层

            # 前向传播函数，定义模型的计算流程
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                a = self.conv1(x)  # 第一次卷积操作
                b = self.conv2(a)  # 第二次卷积操作
                c = a + self.constant_tensor  # 加上常量张量
                z = self.conv3(b + c)  # 第三次卷积操作，并加上之前的结果
                return self.maxpool(self.relu(z))  # 使用ReLU后再进行最大池化

        inputs = (torch.randn(1, 3, 256, 256),)  # 定义输入数据
        # 使用torch._dynamo.export导出模型的图结构
        gm, _ = torch._dynamo.export(M(torch.ones(1, 16, 256, 256)), aten_graph=True)(
            *inputs
        )
        gm.graph.eliminate_dead_code()  # 消除图中的死代码

        # 调用get_source_partitions函数，获取模块分区的信息，包括卷积、ReLU和最大池化层
        module_partitions = get_source_partitions(
            gm.graph, [torch.nn.Conv2d, torch.nn.ReLU, torch.nn.MaxPool2d]
        )

        # 断言模块分区的数量为3
        self.assertEqual(len(module_partitions), 3)
        # 断言卷积层的分区数量为3
        self.assertEqual(len(module_partitions[torch.nn.Conv2d]), 3)
        # 断言ReLU层的分区数量为1
        self.assertEqual(len(module_partitions[torch.nn.ReLU]), 1)
        # 断言最大池化层的分区数量为1
        self.assertEqual(len(module_partitions[torch.nn.MaxPool2d]), 1)

        # 检查第一个卷积分区和ReLU分区之间是否连接
        self.assertFalse(
            check_subgraphs_connected(
                module_partitions[torch.nn.Conv2d][0],
                module_partitions[torch.nn.ReLU][0],
            )
        )
        # 检查第二个卷积分区和ReLU分区之间是否连接
        self.assertFalse(
            check_subgraphs_connected(
                module_partitions[torch.nn.Conv2d][1],
                module_partitions[torch.nn.ReLU][0],
            )
        )
        # 检查第三个卷积分区和ReLU分区之间是否连接
        self.assertTrue(
            check_subgraphs_connected(
                module_partitions[torch.nn.Conv2d][2],
                module_partitions[torch.nn.ReLU][0],
            )
        )
        # 检查最大池化分区和ReLU分区之间是否连接
        self.assertFalse(
            check_subgraphs_connected(
                module_partitions[torch.nn.MaxPool2d][0],
                module_partitions[torch.nn.ReLU][0],
            )
        )
        # 检查ReLU分区和最大池化分区之间是否连接
        self.assertTrue(
            check_subgraphs_connected(
                module_partitions[torch.nn.ReLU][0],
                module_partitions[torch.nn.MaxPool2d][0],
            )
        )

    # 如果不支持Dynamo，则跳过测试
    @unittest.skipIf(not is_dynamo_supported(), "Dynamo not supported")
    @unittest.skipIf(not is_dynamo_supported(), "Dynamo not supported")
    # 如果当前环境不支持 Dynamo，则跳过测试
    def test_module_partitioner_functional_conv_relu_conv(self):
        # 定义一个功能性的二维卷积类
        class FunctionalConv2d(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.stride = (1, 1)
                self.padding = (0, 0)
                self.dilation = (1, 1)
                self.groups = 1

            def forward(self, x, weight, bias):
                # 使用功能性接口执行二维卷积操作
                return torch.nn.functional.conv2d(
                    x,
                    weight,
                    bias,
                    self.stride,
                    self.padding,
                    self.dilation,
                    self.groups,
                )

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 创建两个 FunctionalConv2d 类的实例作为模块的成员
                self.conv1 = FunctionalConv2d()
                self.conv2 = FunctionalConv2d()

            def forward(self, x, weight, bias):
                # 执行模型的前向传播：卷积 -> ReLU -> 卷积
                x = self.conv1(x, weight, bias)
                x = torch.nn.functional.relu(x)
                x = self.conv2(x, weight, bias)
                return x

        # 生成随机输入数据和权重
        inputs = (torch.randn(1, 3, 5, 5), torch.rand(3, 3, 3, 3), torch.rand(3))
        # 导出模型并获取计算图
        gm, _ = torch._dynamo.export(M(), aten_graph=True)(*inputs)
        # 删除无用的计算节点
        gm.graph.eliminate_dead_code()

        # 获取模块分区，限定使用 torch.nn.functional.conv2d 函数
        module_partitions = get_source_partitions(
            gm.graph, [torch.nn.functional.conv2d]
        )

        # 断言模块分区的数量和 conv2d 函数的调用次数
        self.assertEqual(len(module_partitions), 1)
        self.assertEqual(len(module_partitions[torch.nn.functional.conv2d]), 2)

    @unittest.skipIf(not is_dynamo_supported(), "Dynamo not supported")
    # 如果当前环境不支持 Dynamo，则跳过测试
    def test_module_partitioner_functional_linear_relu_linear(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, weight, bias):
                # 执行模型的前向传播：线性 -> 线性 -> ReLU -> 线性 -> 线性 -> ReLU
                x = torch.nn.functional.linear(x, weight, bias)
                x = torch.nn.functional.linear(x, weight, bias)
                x = torch.nn.functional.relu(x)
                x = torch.nn.functional.linear(x, weight, bias)
                x = torch.nn.functional.linear(x, weight, bias)
                x = torch.nn.functional.relu(x)
                return x

        # 生成随机输入数据和权重
        inputs = (torch.randn(1, 5), torch.rand((5, 5)), torch.zeros(5))
        # 导出模型并获取计算图
        gm, _ = torch._dynamo.export(M(), aten_graph=True)(*inputs)
        # 删除无用的计算节点
        gm.graph.eliminate_dead_code()

        # 获取模块分区，限定使用 torch.nn.functional.linear 和 torch.nn.functional.relu 函数
        module_partitions = get_source_partitions(
            gm.graph, [torch.nn.functional.linear, torch.nn.functional.relu]
        )

        # 断言模块分区的数量和 linear、relu 函数的调用次数
        self.assertEqual(len(module_partitions), 2)
        self.assertEqual(len(module_partitions[torch.nn.functional.linear]), 4)
        self.assertEqual(len(module_partitions[torch.nn.functional.relu]), 2)
    def test_legalize_slice(self):
        # 定义一个名为 test_legalize_slice 的测试方法
        class M(torch.nn.Module):
            # 定义一个内部类 M，继承自 torch.nn.Module
            def forward(self, x, y):
                # 定义 forward 方法，接受输入参数 x 和 y
                b = x.item()
                # 从张量 x 中获取其数值并赋给变量 b
                torch._check_is_size(b)
                # 调用 torch 内部函数检查 b 是否合法作为尺寸
                torch._check(b + 1 < y.size(0))
                # 调用 torch 内部函数检查 b + 1 是否小于 y 的第一个维度大小
                return y[: b + 1]
                # 返回 y 的切片，切片范围为从头到 b+1

        ep = torch.export.export(M(), (torch.tensor(4), torch.randn(10)))
        # 调用 torch.export.export 导出内部类 M 的模型，使用给定的输入
        fake_inputs = [
            node.meta["val"] for node in ep.graph.nodes if node.op == "placeholder"
        ]
        # 创建 fake_inputs 列表，包含 ep 图中所有操作为 "placeholder" 的节点的 meta["val"] 值
        gm = ep.module()
        # 从 ep 中获取模块并赋值给 gm
        with fake_inputs[0].fake_mode:
            # 进入 fake_inputs 列表的第一个元素的 fake_mode 上下文
            torch.fx.Interpreter(gm).run(*fake_inputs)
            # 使用 gm 创建 torch.fx.Interpreter 对象，并运行它，传入 fake_inputs
        legalized_gm = legalize_graph(gm)
        # 调用 legalize_graph 函数对 gm 进行合法化处理，并将结果赋给 legalized_gm
        with fake_inputs[0].fake_mode:
            # 进入 fake_inputs 列表的第一个元素的 fake_mode 上下文
            torch.fx.Interpreter(legalized_gm).run(*fake_inputs)
            # 使用 legalized_gm 创建 torch.fx.Interpreter 对象，并运行它，传入 fake_inputs
```