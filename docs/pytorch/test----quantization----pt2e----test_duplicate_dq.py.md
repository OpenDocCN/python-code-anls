# `.\pytorch\test\quantization\pt2e\test_duplicate_dq.py`

```py
# 代码导入了所需的库和模块，包括标准库和第三方库
# 这些库包括 unittest 用于单元测试，torch 用于深度学习框架操作
# 还有来自 torch.ao.quantization 模块的量化相关功能
# 另外还有来自 torch.testing._internal.common_quantization 和 torch.testing._internal.common_utils 的一些测试和实用工具
import copy
import unittest
from typing import Any, Dict

import torch
from torch._export import capture_pre_autograd_graph

# 从 torch.ao.quantization.observer 模块中导入相关观察器类
from torch.ao.quantization.observer import (
    HistogramObserver,
    MinMaxObserver,
    PlaceholderObserver,
)
# 从 torch.ao.quantization.quantize_pt2e 模块中导入转换和准备函数
from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_pt2e
# 从 torch.ao.quantization.quantizer 模块中导入量化相关类和函数
from torch.ao.quantization.quantizer import (
    QuantizationAnnotation,
    QuantizationSpec,
    Quantizer,
    SharedQuantizationSpec,
)
# 从 torch.ao.quantization.quantizer.xnnpack_quantizer 模块中导入对称量化配置函数
from torch.ao.quantization.quantizer.xnnpack_quantizer import (
    get_symmetric_quantization_config,
)
# 从 torch.ao.quantization.quantizer.xnnpack_quantizer_utils 模块中导入量化配置类和常量
from torch.ao.quantization.quantizer.xnnpack_quantizer_utils import (
    OP_TO_ANNOTATOR,
    QuantizationConfig,
)

# 从 torch.testing._internal.common_quantization 模块中导入量化测试用例类
from torch.testing._internal.common_quantization import QuantizationTestCase
# 从 torch.testing._internal.common_utils 模块中导入 IS_WINDOWS 常量
from torch.testing._internal.common_utils import IS_WINDOWS


# 定义一个帮助测试的模块类 TestHelperModules
class TestHelperModules:
    # 包含一个具有不同操作的卷积模块的类 Conv2dWithObsSharingOps
    class Conv2dWithObsSharingOps(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 3, 3)
            self.hardtanh = torch.nn.Hardtanh()
            self.adaptive_avg_pool2d = torch.nn.AdaptiveAvgPool2d((1, 1))
            self.linear = torch.nn.Linear(3, 3)

        def forward(self, x):
            x = self.conv(x)
            x = self.adaptive_avg_pool2d(x)
            x = self.hardtanh(x)
            x = x.view(-1, 3)
            x = self.linear(x)
            return x

    # 包含一个具有共享量化参数的卷积模块的类 Conv2dWithSharedDQ
    class Conv2dWithSharedDQ(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = torch.nn.Conv2d(3, 3, 3)
            self.conv2 = torch.nn.Conv2d(3, 3, 1)
            self.linear = torch.nn.Linear(3, 3)

        def forward(self, x):
            x = self.conv1(x)
            z = x.view(-1, 3)
            w = self.linear(z)

            y = self.conv2(x)
            add_output = x + y

            extra_output = x * 2
            return w, add_output, extra_output

    # 包含不同量化配置的模块的类 ModuleForDifferentQconfig
    class ModuleForDifferentQconfig(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = torch.nn.Conv2d(3, 3, 3)
            self.conv2 = torch.nn.Conv2d(3, 3, 1)
            self.adaptive_avg_pool2d = torch.nn.AdaptiveAvgPool2d((1, 1))

        def forward(self, x):
            x = self.conv1(x)
            w = self.adaptive_avg_pool2d(x)

            y = self.conv2(x)
            add_output = x + y

            extra_output = x + 2
            return w, add_output, extra_output


# 定义一个列表，包含了 torch.ops.quantized_decomposed 模块中的一些反量化操作
_DEQUANTIZE_OPS = [
    torch.ops.quantized_decomposed.dequantize_per_tensor.default,
    torch.ops.quantized_decomposed.dequantize_per_tensor.tensor,
    torch.ops.quantized_decomposed.dequantize_per_channel.default,
]

# 标记 unittest 测试类 TestDuplicateDQPass，继承自 QuantizationTestCase
# 在 Windows 系统上跳过这个测试，因为目前不支持 torch.compile
@unittest.skipIf(IS_WINDOWS, "Windows not yet supported for torch.compile")
class TestDuplicateDQPass(QuantizationTestCase):
    # 定义一个测试函数 _test_duplicate_dq，用于测试重复的量化操作
    def _test_duplicate_dq(
        self,
        model,
        example_inputs,
        quantizer,
        ):
        # 将模型设置为评估模式
        m_eager = model.eval()

        # 创建模型的深层副本，以便捕获自动求导图
        m = copy.deepcopy(m_eager)
        m = capture_pre_autograd_graph(
            m,
            example_inputs,
        )

        # 准备模型进行量化转换
        m = prepare_pt2e(m, quantizer)
        # 在量化前进行模型校准
        m(*example_inputs)
        # 将模型转换为量化模型
        m = convert_pt2e(m)

        # 使用示例输入执行量化后的模型，并遍历计算图中的节点
        pt2_quant_output = m(*example_inputs)
        for n in m.graph.nodes:
            # 获取节点的量化标注信息
            annotation = n.meta.get("quantization_annotation", None)
            if annotation is not None:
                # 检查节点的参数，确保没有重复的反量化操作
                for arg in n.args:
                    if isinstance(arg, torch.fx.Node) and arg.target in _DEQUANTIZE_OPS:
                        self.assertEqual(len(arg.users.keys()), 1)

    def test_no_need_for_duplicate_dq(self):
        """
        Model under test
        conv2d -> avgpool -> hardtanh -> linear
        Check quantization tags on conv2d, avgpool and linear are correctly set
        """

        class BackendAQuantizer(Quantizer):
            # 实现对图模块的注释方法
            def annotate(self, gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
                backend_string = "BackendA"
                # 获取对称量化配置
                quantization_config = get_symmetric_quantization_config(
                    is_per_channel=True
                )
                # 在图模块上标注线性操作的量化配置
                OP_TO_ANNOTATOR["linear"](gm, quantization_config)
                # 在图模块上标注卷积操作的量化配置
                OP_TO_ANNOTATOR["conv"](gm, quantization_config)
                # 在图模块上标注自适应平均池化操作的量化配置
                OP_TO_ANNOTATOR["adaptive_avg_pool2d"](gm, quantization_config)

            # 实现模型验证方法
            def validate(self, model: torch.fx.GraphModule) -> None:
                pass

        # 定义示例输入
        example_inputs = (torch.randn(1, 3, 5, 7),)
        # 执行测试，验证无需重复反量化操作
        self._test_duplicate_dq(
            TestHelperModules.Conv2dWithObsSharingOps(),
            example_inputs,
            BackendAQuantizer(),
        )
    def test_simple_duplicate_dq(self):
        """
        Model under test
        conv2d -> conv2d -> add
             |          |
              --------->

        -----> view_copy --> linear
             |
              -----> mul
        There should be three dq nodes because output for the
        first conv2d is fed to next conv2d, add, and view_copy + linear.
        All three are quantized.
        Thus DQ node is not duplicated for those three uses
        """

        class BackendAQuantizer(Quantizer):
            # 用于在图模块上注释量化信息，使用 BackendA 配置对称量化
            def annotate(self, gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
                backend_string = "BackendA"
                quantization_config = get_symmetric_quantization_config(
                    is_per_channel=True
                )
                # 对线性层进行量化注释
                OP_TO_ANNOTATOR["linear"](gm, quantization_config)
                # 对卷积操作进行量化注释
                OP_TO_ANNOTATOR["conv"](gm, quantization_config)
                # 对加法操作进行量化注释
                OP_TO_ANNOTATOR["add"](gm, quantization_config)

            # 模型验证函数，此处为空实现
            def validate(self, model: torch.fx.GraphModule) -> None:
                pass

        example_inputs = (torch.randn(1, 3, 5, 7),)
        # 调用 _test_duplicate_dq 函数，测试量化是否正确重用
        self._test_duplicate_dq(
            TestHelperModules.Conv2dWithSharedDQ(),
            example_inputs,
            BackendAQuantizer(),
        )

    def test_no_add_quant_duplicate_dq(self):
        """
        Model under test
        conv2d -> conv2d -> add
             |          |
              --------->

        -----> view_copy --> linear
             |
              -----> mul
        There should be three dq nodes because output for the
        first conv2d is fed to next conv2d, and view_copy + linear.
        Both are quantized.
        However the skip connection to add and mul are not quantized.
        Thus DQ node is not duplicated for those two uses
        """

        class BackendAQuantizer(Quantizer):
            # 用于在图模块上注释量化信息，使用 BackendA 配置对称量化
            def annotate(self, gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
                backend_string = "BackendA"
                quantization_config = get_symmetric_quantization_config(
                    is_per_channel=True
                )
                # 对线性层进行量化注释
                OP_TO_ANNOTATOR["linear"](gm, quantization_config)
                # 对卷积操作进行量化注释
                OP_TO_ANNOTATOR["conv"](gm, quantization_config)

            # 模型验证函数，此处为空实现
            def validate(self, model: torch.fx.GraphModule) -> None:
                pass

        example_inputs = (torch.randn(1, 3, 5, 7),)
        # 调用 _test_duplicate_dq 函数，测试量化是否正确重用
        self._test_duplicate_dq(
            TestHelperModules.Conv2dWithSharedDQ(),
            example_inputs,
            BackendAQuantizer(),
        )
```