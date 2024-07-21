# `.\pytorch\test\quantization\pt2e\test_graph_utils.py`

```py
# Owner(s): ["oncall: quantization"]
# 导入必要的库和模块
import copy  # 导入深拷贝函数用于复制对象
import unittest  # 导入单元测试模块

import torch  # 导入PyTorch库
import torch._dynamo as torchdynamo  # 导入私有模块 torch._dynamo

from torch.ao.quantization.pt2e.graph_utils import (  # 导入图操作相关的函数和类
    find_sequential_partitions,
    get_equivalent_types,
    update_equivalent_types_dict,
)
from torch.testing._internal.common_utils import IS_WINDOWS, TestCase  # 导入测试工具和标志


class TestGraphUtils(TestCase):
    @unittest.skipIf(IS_WINDOWS, "torch.compile is not supported on Windows")
    def test_conv_bn_conv_relu(self):
        # 定义一个包含多层神经网络的测试模块
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(3, 3, 3)  # 第一层卷积层
                self.bn1 = torch.nn.BatchNorm2d(3)  # 第一个批归一化层
                self.conv2 = torch.nn.Conv2d(3, 3, 3)  # 第二层卷积层
                self.relu2 = torch.nn.ReLU()  # 第二个ReLU激活层

            def forward(self, x):
                bn_out = self.bn1(self.conv1(x))  # 执行卷积和批归一化操作
                relu_out = torch.nn.functional.relu(bn_out)  # 执行ReLU激活
                return self.relu2(self.conv2(relu_out))  # 执行第二次卷积和ReLU操作

        m = M().eval()  # 创建并评估模型实例
        example_inputs = (torch.randn(1, 3, 5, 5),)  # 创建输入示例数据

        # 使用torchdynamo导出模型并捕获程序
        m, guards = torchdynamo.export(
            m,
            *copy.deepcopy(example_inputs),
            aten_graph=True,
        )

        # 查找并返回顺序分区中的融合分区
        fused_partitions = find_sequential_partitions(
            m, [torch.nn.Conv2d, torch.nn.BatchNorm2d]
        )
        self.assertEqual(len(fused_partitions), 1)  # 断言只有一个融合分区

        # 再次查找并返回包含ReLU的顺序分区中的融合分区
        fused_partitions = find_sequential_partitions(
            m, [torch.nn.Conv2d, torch.nn.BatchNorm2d, torch.nn.ReLU]
        )
        self.assertEqual(len(fused_partitions), 1)  # 断言只有一个融合分区

        # 定义一个函数用于引发 ValueError 异常
        def x():
            find_sequential_partitions(
                m,
                [
                    torch.nn.Conv2d,
                    torch.nn.BatchNorm2d,
                    torch.nn.ReLU,
                    torch.nn.functional.conv2d,
                ],
            )

        self.assertRaises(ValueError, x)  # 断言调用 x 函数会引发 ValueError 异常

    @unittest.skipIf(IS_WINDOWS, "torch.compile is not supported on Windows")
    def test_conv_bn_relu(self):
        # 定义一个名为 test_conv_bn_relu 的测试方法
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 初始化模型的各个层：BatchNorm2d、Conv2d 和 ReLU
                self.bn1 = torch.nn.BatchNorm2d(3)  # 创建一个二维批量归一化层，输入通道数为 3
                self.conv2 = torch.nn.Conv2d(3, 3, 3)  # 创建一个二维卷积层，输入通道数、输出通道数和卷积核大小均为 3
                self.relu2 = torch.nn.ReLU()  # 创建一个 ReLU 激活函数层

            def forward(self, x):
                bn_out = self.bn1(x)  # 对输入 x 进行批量归一化
                return self.relu2(self.conv2(bn_out))  # 先进行卷积，然后使用 ReLU 激活函数

        m = M().eval()  # 创建一个模型实例并设为评估模式
        example_inputs = (torch.randn(1, 3, 5, 5),)  # 创建一个示例输入

        # 对模型进行程序捕获，导出 ATen 图
        m, guards = torchdynamo.export(
            m,
            *copy.deepcopy(example_inputs),
            aten_graph=True,
        )
        # 查找连续分区中的 BatchNorm2d 和 Conv2d 的组合，并断言没有这样的分区
        fused_partitions = find_sequential_partitions(
            m, [torch.nn.Conv2d, torch.nn.BatchNorm2d]
        )
        self.assertEqual(len(fused_partitions), 0)
        # 再次查找连续分区中的 BatchNorm2d 和 Conv2d 的组合，并断言有一个这样的分区
        fused_partitions = find_sequential_partitions(
            m, [torch.nn.BatchNorm2d, torch.nn.Conv2d]
        )
        self.assertEqual(len(fused_partitions), 1)
        # 查找连续分区中的 BatchNorm2d 和 ReLU 的组合，并断言没有这样的分区
        fused_partitions = find_sequential_partitions(
            m, [torch.nn.BatchNorm2d, torch.nn.ReLU]
        )
        self.assertEqual(len(fused_partitions), 0)

    @unittest.skipIf(IS_WINDOWS, "torch.compile is not supported on Windows")
    def test_customized_equivalet_types_dict(self):
        # 定义一个名为 test_customized_equivalet_types_dict 的测试方法，如果在 Windows 上则跳过
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 3, 3)  # 创建一个二维卷积层，输入通道数、输出通道数和卷积核大小均为 3

            def forward(self, x):
                return torch.nn.functional.relu6(self.conv(x))  # 对输入 x 进行卷积并使用 ReLU6 激活函数

        m = M().eval()  # 创建一个模型实例并设为评估模式
        example_inputs = (torch.randn(1, 3, 5, 5),)  # 创建一个示例输入

        # 对模型进行程序捕获，导出 ATen 图
        m, guards = torchdynamo.export(
            m,
            *copy.deepcopy(example_inputs),
            aten_graph=True,
        )
        # 获取自定义等效类型字典
        customized_equivalent_types = get_equivalent_types()
        # 将自定义的等效类型添加到字典中
        customized_equivalent_types.append({torch.nn.ReLU6, torch.nn.functional.relu6})
        update_equivalent_types_dict(customized_equivalent_types)
        # 查找连续分区中的 Conv2d 和 ReLU6 的组合，并断言有一个这样的分区
        fused_partitions = find_sequential_partitions(
            m,
            [torch.nn.Conv2d, torch.nn.ReLU6],
        )
        self.assertEqual(len(fused_partitions), 1)
```