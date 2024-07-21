# `.\pytorch\test\quantization\pt2e\test_xnnpack_quantizer.py`

```py
# 导入必要的模块和函数
import copy  # 导入 copy 模块，用于对象的浅复制和深复制
import operator  # 导入 operator 模块，包含了一些 Python 的运算符和函数

import torch  # 导入 PyTorch 库
import torch._dynamo as torchdynamo  # 导入私有模块 torch._dynamo
from torch._export import capture_pre_autograd_graph  # 从 torch._export 模块导入 capture_pre_autograd_graph 函数
from torch.ao.ns.fx.utils import compute_sqnr  # 从 torch.ao.ns.fx.utils 模块导入 compute_sqnr 函数
from torch.ao.quantization import (  # 导入量化相关的模块和函数
    default_dynamic_fake_quant,  # 默认的动态伪量化函数
    default_dynamic_qconfig,  # 默认的动态量化配置
    observer,  # 观察器函数
    QConfig,  # 量化配置类
    QConfigMapping,  # 量化配置映射类
)
from torch.ao.quantization.backend_config import get_qnnpack_backend_config  # 从 backend_config 模块导入获取 qnnpack 后端配置函数
from torch.ao.quantization.qconfig import (  # 导入量化配置相关的模块和函数
    default_per_channel_symmetric_qnnpack_qconfig,  # 默认的通道对称量化 qnnpack 配置
    default_symmetric_qnnpack_qconfig,  # 默认的对称量化 qnnpack 配置
    per_channel_weight_observer_range_neg_127_to_127,  # 通道权重观察器范围 -127 到 127
    weight_observer_range_neg_127_to_127,  # 权重观察器范围 -127 到 127
)
from torch.ao.quantization.quantize_fx import (  # 从 quantize_fx 模块导入量化 FX 功能
    _convert_to_reference_decomposed_fx,  # 转换为参考分解 FX 函数
    convert_to_reference_fx,  # 转换为参考 FX 函数
    prepare_fx,  # 准备 FX 函数
)
from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_pt2e  # 从 quantize_pt2e 模块导入转换 PT2E 和准备 PT2E 函数
from torch.ao.quantization.quantizer.xnnpack_quantizer import (  # 从 xnnpack_quantizer 模块导入 XNNPACK 量化器相关功能
    get_symmetric_quantization_config,  # 获取对称量化配置函数
    XNNPACKQuantizer,  # XNNPACK 量化器类
)

from torch.testing._internal.common_quantization import (  # 导入量化测试相关的模块和函数
    NodeSpec as ns,  # 重命名 NodeSpec 到 ns
    PT2EQuantizationTestCase,  # PT2E 量化测试用例类
    skip_if_no_torchvision,  # 如果没有 torchvision 则跳过测试
    skipIfNoQNNPACK,  # 如果没有 QNNPACK 则跳过测试
    TestHelperModules,  # 测试辅助模块
)
from torch.testing._internal.common_quantized import override_quantized_engine  # 从 common_quantized 模块导入覆盖量化引擎函数

@skipIfNoQNNPACK  # 如果没有 QNNPACK，则跳过测试
class TestXNNPACKQuantizer(PT2EQuantizationTestCase):
    def test_conv1d(self):
        # 创建 XNNPACK 量化器对象
        quantizer = XNNPACKQuantizer()
        # 获取对称量化配置，采用通道对称量化
        quantization_config = get_symmetric_quantization_config(is_per_channel=True)
        # 设置全局量化配置到量化器中
        quantizer.set_global(quantization_config)
        # 创建示例输入
        example_inputs = (torch.randn(1, 3, 5),)
        # 定义节点发生情况字典
        node_occurrence = {
            # 输入和输出使用 quantize_per_tensor，权重使用 quantize_per_channel
            torch.ops.quantized_decomposed.quantize_per_tensor.default: 2,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default: 2,
            torch.ops.quantized_decomposed.quantize_per_channel.default: 0,
            torch.ops.quantized_decomposed.dequantize_per_channel.default: 1,
        }
        # 定义节点列表
        node_list = [
            torch.ops.quantized_decomposed.dequantize_per_tensor.default,
            torch.ops.aten.conv1d.default,
            torch.ops.quantized_decomposed.quantize_per_tensor.default,
        ]
        # 调用测试量化器函数，验证量化器的功能
        self._test_quantizer(
            TestHelperModules.ConvWithBNRelu(dim=1, relu=False, bn=False),  # 使用 ConvWithBNRelu 测试模块
            example_inputs,  # 示例输入
            quantizer,  # 量化器对象
            node_occurrence,  # 节点发生情况字典
            node_list,  # 节点列表
        )
    # 定义一个名为 test_conv2d 的测试方法，用于测试 Conv2d 相关功能
    def test_conv2d(self):
        # 创建 XNNPACKQuantizer 实例，用于量化操作
        quantizer = XNNPACKQuantizer()
        # 获取对称量化配置，每个通道使用相同的量化参数
        quantization_config = get_symmetric_quantization_config(is_per_channel=True)
        # 将全局量化配置应用到量化器中
        quantizer.set_global(quantization_config)
        # 准备示例输入数据，这里是一个 1x3x5x5 的张量
        example_inputs = (torch.randn(1, 3, 5, 5),)
        # 定义节点发生的次数字典，描述了各操作在计算图中出现的次数
        node_occurrence = {
            # 输入和输出使用 quantize_per_tensor，权重使用 quantize_per_channel
            torch.ops.quantized_decomposed.quantize_per_tensor.default: 2,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default: 2,
            # 权重的 quantize_per_channel 在常量传播中使用
            torch.ops.quantized_decomposed.quantize_per_channel.default: 0,
            torch.ops.quantized_decomposed.dequantize_per_channel.default: 1,
        }
        # 定义节点列表，表示测试中需要验证的操作节点顺序
        node_list = [
            torch.ops.quantized_decomposed.dequantize_per_tensor.default,
            torch.ops.aten.conv2d.default,
            torch.ops.quantized_decomposed.quantize_per_tensor.default,
        ]
        # 调用 _test_quantizer 方法进行量化器的测试
        self._test_quantizer(
            TestHelperModules.ConvWithBNRelu(relu=False, bn=False),  # 使用 ConvWithBNRelu 模型进行测试
            example_inputs,  # 示例输入数据
            quantizer,  # 使用的量化器
            node_occurrence,  # 节点发生次数字典
            node_list,  # 需要验证的节点列表
        )

    # 定义一个名为 test_conv1d_with_conv2d 的测试方法，测试 Conv1d 和 Conv2d 结合使用时的量化功能
    def test_conv1d_with_conv2d(self):
        # 创建 XNNPACKQuantizer 实例，用于量化操作
        quantizer = XNNPACKQuantizer()
        # 获取对称量化配置，每个通道使用相同的量化参数
        quantization_config = get_symmetric_quantization_config(is_per_channel=True)
        # 将全局量化配置应用到量化器中
        quantizer.set_global(quantization_config)
        # 定义节点发生的次数字典，描述了各操作在计算图中出现的次数
        node_occurrence = {
            # 输入和输出使用 quantize_per_tensor，权重使用 quantize_per_channel
            torch.ops.quantized_decomposed.quantize_per_tensor.default: 4,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default: 4,
            torch.ops.quantized_decomposed.quantize_per_channel.default: 0,
            torch.ops.quantized_decomposed.dequantize_per_channel.default: 2,
        }
        # 定义节点列表，表示测试中需要验证的操作节点顺序
        node_list = [
            torch.ops.quantized_decomposed.dequantize_per_tensor.default,
            torch.ops.aten.conv2d.default,
            torch.ops.quantized_decomposed.quantize_per_tensor.default,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default,
            torch.ops.aten.conv1d.default,
            torch.ops.quantized_decomposed.quantize_per_tensor.default,
        ]
        # 创建 Conv2dThenConv1d 模型实例，用于测试 Conv2d 和 Conv1d 结合的情况
        m = TestHelperModules.Conv2dThenConv1d()
        # 调用 _test_quantizer 方法进行量化器的测试
        self._test_quantizer(
            m,  # 使用 Conv2dThenConv1d 模型进行测试
            m.example_inputs(),  # 模型的示例输入数据
            quantizer,  # 使用的量化器
            node_occurrence,  # 节点发生次数字典
            node_list,  # 需要验证的节点列表
        )
    # 定义一个测试方法，用于测试量化器在不同输入维度下的行为
    def test_linear(self):
        # 创建一个 XNNPACKQuantizer 实例
        quantizer = XNNPACKQuantizer()
        
        # 获取对称量化配置，每通道独立量化
        quantization_config = get_symmetric_quantization_config(is_per_channel=True)
        
        # 将全局量化配置应用到量化器上
        quantizer.set_global(quantization_config)
        
        # 创建并评估一个包含两个线性层的测试模块
        m_eager = TestHelperModules.TwoLinearModule().eval()

        # 以下是不同维度输入的示例数据
        example_inputs_2d = (torch.randn(9, 8),)
        example_inputs_3d = (torch.randn(9, 10, 8),)
        example_inputs_4d = (torch.randn(9, 10, 11, 8),)
        
        # 定义节点出现的次数字典，指定不同操作对应的节点出现次数
        node_occurrence = {
            # 输入和输出使用 quantize_per_tensor，权重使用 quantize_per_channel
            torch.ops.quantized_decomposed.quantize_per_tensor.default: 3,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default: 3,
            # 权重使用 quantize_per_channel 并进行常量传播
            torch.ops.quantized_decomposed.quantize_per_channel.default: 0,
            torch.ops.quantized_decomposed.dequantize_per_channel.default: 2,
        }
        
        # 默认的每通道对称 QNNPACK 量化配置
        qconfig = default_per_channel_symmetric_qnnpack_qconfig
        
        # 创建 QConfigMapping 实例并将全局量化配置应用到映射中
        qconfig_mapping = QConfigMapping().set_global(qconfig)
        
        # 对每个示例输入进行量化器测试
        for example_inputs in [example_inputs_2d, example_inputs_3d, example_inputs_4d]:
            self._test_quantizer(
                m_eager,
                example_inputs,
                quantizer,
                node_occurrence,
                [],  # 空的额外参数列表
                True,  # 测试过程中启用损失计算
                qconfig_mapping,
            )
    # 定义一个测试方法，用于测试线性ReLU模型的量化
    def test_linear_relu(self):
        # 创建XNNPACK量化器对象
        quantizer = XNNPACKQuantizer()
        # 获取对称量化配置，通道独立量化
        quantization_config = get_symmetric_quantization_config(is_per_channel=True)
        # 将全局量化配置应用于量化器
        quantizer.set_global(quantization_config)
        # 创建并评估一个线性ReLU模型
        m_eager = TestHelperModules.LinearReluModel().eval()

        # 用于测试的2维输入示例
        example_inputs_2d = (torch.randn(1, 5),)
        # 用于测试的3维输入示例
        example_inputs_3d = (torch.randn(1, 2, 5),)
        # 用于测试的4维输入示例
        example_inputs_4d = (torch.randn(1, 2, 3, 5),)

        # 定义节点出现次数的期望字典
        node_occurrence = {
            # 输入和输出使用quantize_per_tensor，权重使用quantize_per_channel
            torch.ops.quantized_decomposed.quantize_per_tensor.default: 2,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default: 2,
            # 权重使用quantize_per_channel常量传播
            torch.ops.quantized_decomposed.quantize_per_channel.default: 0,
            torch.ops.quantized_decomposed.dequantize_per_channel.default: 1,
        }
        # 获取默认的通道独立对称量化配置
        qconfig = default_per_channel_symmetric_qnnpack_qconfig
        # 创建量化配置映射对象，并设置全局配置
        qconfig_mapping = QConfigMapping().set_global(qconfig)

        # 对每种输入示例进行量化器测试
        for example_inputs in [example_inputs_2d, example_inputs_3d, example_inputs_4d]:
            self._test_quantizer(
                m_eager,
                example_inputs,
                quantizer,
                node_occurrence,
                [],  # node_list为空列表
                False,  # executor_backend_config()不会融合linear-relu
                qconfig_mapping,
            )

    # 定义一个测试方法，用于测试卷积线性模型（不进行转置）
    def test_conv_linear_no_permute(self):
        # 创建XNNPACK量化器对象
        quantizer = XNNPACKQuantizer()
        # 获取对称量化配置，通道独立量化
        quantization_config = get_symmetric_quantization_config(is_per_channel=True)
        # 将全局量化配置应用于量化器
        quantizer.set_global(quantization_config)
        
        # 定义节点出现次数的期望字典
        node_occurrence = {
            # 输入和输出使用quantize_per_tensor，权重使用quantize_per_channel
            torch.ops.quantized_decomposed.quantize_per_tensor.default: 5,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default: 5,
            # 权重使用quantize_per_channel常量传播
            torch.ops.quantized_decomposed.quantize_per_channel.default: 0,
            torch.ops.quantized_decomposed.dequantize_per_channel.default: 3,
        }
        # 获取默认的通道独立对称量化配置
        qconfig = default_per_channel_symmetric_qnnpack_qconfig
        # 创建量化配置映射对象，并设置全局配置
        qconfig_mapping = QConfigMapping().set_global(qconfig)
        
        # 用于测试的2维输入示例
        example_inputs = (torch.randn(2, 3, 4, 4),)
        # 对卷积线性模型进行量化器测试
        self._test_quantizer(
            TestHelperModules.Conv2dWithTwoLinear(),
            example_inputs,
            quantizer,
            node_occurrence,
            [],
            True,  # executor_backend_config()会融合linear-relu
            qconfig_mapping,
        )
    # 定义一个测试方法 test_conv_linear，用于测试线性卷积操作的量化功能
    def test_conv_linear(self):
        # 创建 XNNPACKQuantizer 的实例
        quantizer = XNNPACKQuantizer()
        # 获取对称量化配置（按通道量化）
        quantization_config = get_symmetric_quantization_config(is_per_channel=True)
        # 设置全局量化配置
        quantizer.set_global(quantization_config)

        # 使用 2 维输入进行测试
        example_inputs = (torch.randn(2, 3, 4, 4),)
        # 定义节点出现次数的字典，记录量化操作的调用次数
        node_occurrence = {
            torch.ops.quantized_decomposed.quantize_per_tensor.default: 5,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default: 5,
            # 权重的 quantize_per_channel 被常量传播
            torch.ops.quantized_decomposed.quantize_per_channel.default: 0,
            torch.ops.quantized_decomposed.dequantize_per_channel.default: 3,
        }
        # 使用默认的按通道对称量化配置
        qconfig = default_per_channel_symmetric_qnnpack_qconfig
        # 创建量化配置映射对象，并设置全局量化配置
        qconfig_mapping = QConfigMapping().set_global(qconfig)
        # 调用 _test_quantizer 方法，对量化器进行测试
        self._test_quantizer(
            TestHelperModules.Conv2dWithTwoLinearPermute(),  # 测试用例为 Conv2dWithTwoLinearPermute 模块
            example_inputs,  # 测试输入数据
            quantizer,  # 量化器对象
            node_occurrence,  # 节点出现次数字典
            [],  # 额外参数为空列表
            True,  # 使用动态形状
            qconfig_mapping,  # 量化配置映射对象
        )

    # 定义一个测试方法 test_linear_with_dynamic_shape，用于测试具有动态形状的线性操作的量化功能
    def test_linear_with_dynamic_shape(self):
        # 创建 XNNPACKQuantizer 的实例
        quantizer = XNNPACKQuantizer()
        # 获取对称量化配置（按通道量化）
        quantization_config = get_symmetric_quantization_config(is_per_channel=True)
        # 设置全局量化配置
        quantizer.set_global(quantization_config)
        # 创建评估模式下的 TwoLinearModule 实例
        m_eager = TestHelperModules.TwoLinearModule().eval()

        # 使用 2 维输入进行测试
        example_inputs_3d = (torch.randn(9, 10, 8),)
        # 定义节点出现次数的字典，记录量化操作的调用次数
        node_occurrence = {
            # 输入和输出使用 quantize_per_tensor，权重使用 quantize_per_channel
            torch.ops.quantized_decomposed.quantize_per_tensor.default: 3,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default: 3,
            # 权重的 quantize_per_channel 被常量传播
            torch.ops.quantized_decomposed.quantize_per_channel.default: 0,
            torch.ops.quantized_decomposed.dequantize_per_channel.default: 2,
        }
        # 使用默认的按通道对称量化配置
        qconfig = default_per_channel_symmetric_qnnpack_qconfig
        # 创建量化配置映射对象，并设置全局量化配置
        qconfig_mapping = QConfigMapping().set_global(qconfig)
        # 调用 _test_quantizer 方法，对量化器进行测试
        self._test_quantizer(
            m_eager,  # 测试用例为 m_eager 模块
            example_inputs_3d,  # 测试输入数据
            quantizer,  # 量化器对象
            node_occurrence,  # 节点出现次数字典
            [],  # 额外参数为空列表
            True,  # 使用动态形状
            qconfig_mapping,  # 量化配置映射对象
            export_with_dynamic_shape=True,  # 使用动态形状导出
        )
    # 定义一个测试方法，用于测试观察共享操作的量化器功能
    def test_obs_sharing_ops(self):
        # 创建一个 XNNPACKQuantizer 的实例
        quantizer = XNNPACKQuantizer()
        # 获取对称量化配置，设置为每通道量化
        quantization_config = get_symmetric_quantization_config(is_per_channel=True)
        # 将量化配置应用到全局量化器
        quantizer.set_global(quantization_config)
        # 创建一个 Conv2dWithObsSharingOps 的实例，并设置为评估模式
        m = TestHelperModules.Conv2dWithObsSharingOps().eval()
        # 创建一个包含随机张量输入的示例输入元组
        example_inputs = (torch.randn(1, 3, 5, 5),)
        # 定义节点出现次数的字典，说明了各节点的量化方式及其出现次数
        node_occurrence = {
            # 输入和输出使用 quantize_per_tensor，权重使用 quantize_per_channel
            torch.ops.quantized_decomposed.quantize_per_tensor.default: 5,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default: 5,
            # 权重的 quantize_per_channel 被常量传播了
            torch.ops.quantized_decomposed.quantize_per_channel.default: 0,
            torch.ops.quantized_decomposed.dequantize_per_channel.default: 1,
        }
        # 定义节点列表，描述了测试中涉及的各个操作节点
        node_list = [
            torch.ops.quantized_decomposed.dequantize_per_tensor.default,
            torch.ops.aten.conv2d.default,
            torch.ops.quantized_decomposed.quantize_per_tensor.default,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default,
            torch.ops.aten.adaptive_avg_pool2d.default,
            torch.ops.quantized_decomposed.quantize_per_tensor.default,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default,
            torch.ops.aten.hardtanh.default,
            torch.ops.quantized_decomposed.quantize_per_tensor.default,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default,
            torch.ops.aten.mean.default,
            torch.ops.quantized_decomposed.quantize_per_tensor.default,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default,
        ]
        # 调用 _test_quantizer 方法，进行量化器的测试
        self._test_quantizer(m, example_inputs, quantizer, node_occurrence, node_list)
    # 定义一个测试函数，用于测试设置模块名称的功能
    def test_set_module_name(self):
        # 定义一个继承自torch.nn.Module的子类Sub
        class Sub(torch.nn.Module):
            # 子类构造函数，初始化一个线性层
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(5, 5)

            # 子类前向传播函数，对输入x进行线性变换
            def forward(self, x):
                return self.linear(x)

        # 定义一个继承自torch.nn.Module的主类M
        class M(torch.nn.Module):
            # 主类构造函数，初始化一个线性层和一个Sub子模块
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(5, 5)
                self.sub = Sub()

            # 主类前向传播函数，对输入x先进行线性变换，然后通过sub模块进行处理
            def forward(self, x):
                x = self.linear(x)
                x = self.sub(x)
                return x

        # 创建一个M类的实例并设置为评估模式
        m = M().eval()
        # 创建一个示例输入，为一个5维张量的元组
        example_inputs = (torch.randn(3, 5),)
        # 创建一个XNNPACKQuantizer的实例
        quantizer = XNNPACKQuantizer()
        # 获取对称量化配置，按通道量化
        quantization_config = get_symmetric_quantization_config(is_per_channel=True)
        # 设置量化器quantizer的模块名称为"sub"，并使用指定的量化配置
        quantizer.set_module_name("sub", quantization_config)
        # 定义节点出现次数的字典，指定特定操作的出现次数
        node_occurrence = {
            torch.ops.aten.linear.default: 2,
            # 第二个线性操作的输入和输出
            torch.ops.quantized_decomposed.quantize_per_tensor.default: 2,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default: 2,
        }
        # 定义节点列表，描述了前向传播中各操作的顺序
        node_list = [
            # 第一个线性操作未量化
            torch.ops.aten.linear.default,
            # 第二个线性操作量化
            torch.ops.quantized_decomposed.quantize_per_tensor.default,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default,
            torch.ops.aten.linear.default,
            torch.ops.quantized_decomposed.quantize_per_tensor.default,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default,
        ]
        # 调用测试方法_test_quantizer，用于测试量化器的功能
        self._test_quantizer(m, example_inputs, quantizer, node_occurrence, node_list)
    def test_set_module_name_with_underscores(self) -> None:
        """Test that if a module name has an underscore, we can still quantize it"""

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # This module name has underscores, which can be part of a mangled
                # name.
                self.foo_bar = torch.nn.Linear(2, 2)
                self.baz = torch.nn.Linear(2, 2)

            def forward(self, x):
                return self.baz(self.foo_bar(x))

        quantizer = XNNPACKQuantizer()
        # 设置模块名为 "foo_bar"，使用对称量化配置，设置为按通道量化
        quantizer.set_module_name(
            "foo_bar", get_symmetric_quantization_config(is_per_channel=True)
        )
        example_inputs = (torch.randn(2, 2),)
        m = M().eval()
        # 捕获前自动图，以便后续量化准备
        m = capture_pre_autograd_graph(m, example_inputs)
        m = prepare_pt2e(m, quantizer)
        # 使用线性计数而不是名称，因为名称可能会改变，但顺序应保持不变。
        count = 0
        for n in m.graph.nodes:
            if n.op == "call_function" and n.target == torch.ops.aten.linear.default:
                # 获取权重观察器，以查看按通道与按张量的情况。
                weight_observer_node = n.args[1]
                if count == 0:
                    # 对于 foo_bar，权重张量应该是按张量而非按通道量化的。
                    self.assertEqual(weight_observer_node.op, "call_module")
                    observer_instance = getattr(m, weight_observer_node.target)
                    self.assertEqual(
                        observer_instance.qscheme, torch.per_channel_symmetric
                    )
                else:
                    # 对于 baz，它不应该有任何观察器。
                    self.assertNotEqual(weight_observer_node.op, "call_module")
                count += 1
    def test_set_module_type(self):
        # 定义一个名为 test_set_module_type 的测试方法
        class Sub(torch.nn.Module):
            # 定义一个继承自 torch.nn.Module 的子类 Sub
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(5, 5)
                # 初始化一个线性层，输入维度和输出维度均为 5

            def forward(self, x):
                # 定义前向传播方法 forward，接收输入 x
                return self.linear(x)
                # 返回线性层对输入 x 的计算结果

        class M(torch.nn.Module):
            # 定义一个继承自 torch.nn.Module 的主模块类 M
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(5, 5)
                # 初始化一个线性层，输入维度和输出维度均为 5
                self.sub = Sub()
                # 初始化 Sub 类的实例作为 M 类的一个属性

            def forward(self, x):
                # 定义前向传播方法 forward，接收输入 x
                x = self.linear(x)
                # 对输入 x 应用主模块的线性层
                x = self.sub(x)
                # 对输入 x 应用子模块 Sub 的前向传播
                return x
                # 返回计算结果 x

        m = M().eval()
        # 创建 M 类的一个实例 m，并将其设为评估模式
        example_inputs = (torch.randn(3, 5),)
        # 创建一个示例输入，是一个形状为 (3, 5) 的张量元组
        quantizer = XNNPACKQuantizer()
        # 创建一个 XNNPACKQuantizer 的实例 quantizer
        quantization_config = get_symmetric_quantization_config(is_per_channel=True)
        # 获得一个对称量化配置，每通道量化设置为 True
        quantizer.set_module_type(Sub, quantization_config)
        # 使用 quantization_config 对子模块 Sub 进行量化设置
        node_occurrence = {
            torch.ops.aten.linear.default: 2,
            # torch.ops.aten.linear.default 的出现次数为 2
            torch.ops.quantized_decomposed.quantize_per_tensor.default: 2,
            # torch.ops.quantized_decomposed.quantize_per_tensor.default 的出现次数为 2
            torch.ops.quantized_decomposed.dequantize_per_tensor.default: 2,
            # torch.ops.quantized_decomposed.dequantize_per_tensor.default 的出现次数为 2
        }
        node_list = [
            # 定义一个包含多个操作节点的列表
            torch.ops.aten.linear.default,
            # 第一个线性操作节点未量化
            torch.ops.quantized_decomposed.quantize_per_tensor.default,
            # 第二个线性操作节点量化
            torch.ops.quantized_decomposed.dequantize_per_tensor.default,
            # 第三个操作节点反量化
            torch.ops.aten.linear.default,
            # 第四个线性操作节点未量化
            torch.ops.quantized_decomposed.quantize_per_tensor.default,
            # 第五个线性操作节点量化
            torch.ops.quantized_decomposed.dequantize_per_tensor.default,
            # 第六个操作节点反量化
        ]
        self._test_quantizer(m, example_inputs, quantizer, node_occurrence, node_list)
        # 调用测试方法 _test_quantizer 进行量化器的测试
    # 定义一个测试用例，测试量化器在设置模块类型时的行为
    def test_set_module_type_case_2(self):
        # 定义一个简单的神经网络模型类 M
        class M(torch.nn.Module):
            # 初始化函数，设置模型的各个层和操作
            def __init__(self):
                super().__init__()
                # 第一个卷积层，输入输出通道数为 3，卷积核大小为 3x3，步长为 1，填充为 1，包含偏置
                self.conv = torch.nn.Conv2d(
                    in_channels=3,
                    out_channels=3,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=True,
                )
                # 第二个卷积层，与第一个卷积层相同的设置
                self.conv2 = torch.nn.Conv2d(
                    in_channels=3,
                    out_channels=3,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=True,
                )
                # 第三个卷积层，与前两个卷积层相同的设置
                self.conv3 = torch.nn.Conv2d(
                    in_channels=3,
                    out_channels=3,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=True,
                )
                # ReLU 激活函数层
                self.relu = torch.nn.ReLU()
                # 自适应平均池化层，输出大小为 (1, 1)
                self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
                # 全连接层，输入大小为 3，输出大小为 16
                self.fc = torch.nn.Linear(3, 16)

            # 前向传播函数，定义模型的数据流向
            def forward(self, x):
                # 第一次卷积操作
                x1 = self.conv(x)
                # 第二次卷积后接 ReLU 激活函数，加上第三次卷积的结果
                x2 = self.relu(self.conv2(x1) + self.conv3(x1))
                # 平均池化操作
                x3 = self.avgpool(x2)
                # 将输出展平
                x4 = torch.flatten(x3, 1)
                # 全连接层处理展平后的数据
                x5 = self.fc(x4)
                # 返回最终输出
                return x5

        # 创建 M 类的实例，并设置为评估模式
        m = M().eval()
        # 创建一个示例输入，形状为 (1, 3, 16, 16)
        example_inputs = (torch.randn(1, 3, 16, 16),)
        # 创建一个 XNNPACKQuantizer 实例
        quantizer = XNNPACKQuantizer()
        # 获取对称量化配置，按通道量化
        quantization_config = get_symmetric_quantization_config(is_per_channel=True)
        # 设置量化器仅对 Linear 类型进行注释
        quantizer.set_module_type(torch.nn.Linear, quantization_config)
        # 定义节点出现次数的字典
        node_occurrence = {
            torch.ops.aten.conv2d.default: 3,
            torch.ops.aten.linear.default: 1,
            # 线性层的输入和输出
            torch.ops.quantized_decomposed.quantize_per_tensor.default: 2,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default: 2,
        }
        # 定义节点列表
        node_list = [
            # 仅量化线性层
            torch.ops.quantized_decomposed.quantize_per_tensor.default,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default,
            torch.ops.aten.linear.default,
            torch.ops.quantized_decomposed.quantize_per_tensor.default,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default,
        ]
        # 执行量化器测试函数
        self._test_quantizer(m, example_inputs, quantizer, node_occurrence, node_list)
    def test_propagate_annotation(self):
        # 创建 XNNPACKQuantizer 对象
        quantizer = XNNPACKQuantizer()
        # 获取对称量化配置，通道级别量化
        quantization_config = get_symmetric_quantization_config(is_per_channel=True)
        # 设置全局量化配置
        quantizer.set_global(quantization_config)
        # 创建测试模型，并设置为评估模式
        m = TestHelperModules.Conv2dPropAnnotaton().eval()
        # 创建示例输入数据
        example_inputs = (torch.randn(1, 3, 5, 5),)

        # 捕获自动求导图
        m = capture_pre_autograd_graph(
            m,
            example_inputs,
        )

        # 准备模型进行量化
        m = prepare_pt2e(m, quantizer)
        # 执行模型推理
        m(*example_inputs)
        # 初始化活动后处理对
        act_post_processes_pairs = []
        # 遍历图中的节点
        for n in m.graph.nodes:
            # 如果节点的目标是指定的操作
            if n.target in [
                torch.ops.aten.view.default,
                torch.ops.aten.hardtanh.default,
            ]:
                # 获取输入激活
                input_act = getattr(m, n.args[0].target)
                # 获取输出激活
                output_act = getattr(m, next(iter(n.users)).target)
                # 断言输入激活和输出激活是同一个对象
                self.assertIs(input_act, output_act)

        # 将模型转换为量化引擎格式
        m = convert_pt2e(m)
        # 定义节点出现次数字典
        node_occurrence = {
            # 输入和输出使用量化操作 quantize_per_tensor，出现次数为 5 次
            ns.call_function(
                torch.ops.quantized_decomposed.quantize_per_tensor.default
            ): 5,
            # 输入和输出使用反量化操作 dequantize_per_tensor，出现次数为 5 次
            ns.call_function(
                torch.ops.quantized_decomposed.dequantize_per_tensor.default
            ): 5,
            # 注意：权重使用的量化操作已经在编译时常量传播，出现次数为 0 次
            ns.call_function(
                torch.ops.quantized_decomposed.quantize_per_channel.default
            ): 0,
            # 权重使用的反量化操作 dequantize_per_channel，出现次数为 2 次
            ns.call_function(
                torch.ops.quantized_decomposed.dequantize_per_channel.default
            ): 2,
        }
        # 检查模型的节点出现次数是否符合预期
        self.checkGraphModuleNodes(m, expected_node_occurrence=node_occurrence)
    def test_dynamic_linear(self):
        # 创建 XNNPACKQuantizer 实例
        quantizer = XNNPACKQuantizer()
        # 获取对称量化配置，通道粒度为 True，动态量化为 True
        quantization_config = get_symmetric_quantization_config(
            is_per_channel=True, is_dynamic=True
        )
        # 设置全局量化配置到 quantizer
        quantizer.set_global(quantization_config)
        # 创建并评估 TwoLinearModule 实例
        m_eager = TestHelperModules.TwoLinearModule().eval()

        # 定义节点出现次数的字典
        node_occurrence = {
            # 输入和输出使用 quantize_per_tensor，权重使用 quantize_per_channel
            torch.ops.quantized_decomposed.quantize_per_tensor.tensor: 2,
            torch.ops.quantized_decomposed.dequantize_per_tensor.tensor: 2,
            # 注意: 权重的量化操作已经进行了常量传播
            torch.ops.quantized_decomposed.quantize_per_channel.default: 0,
            torch.ops.quantized_decomposed.dequantize_per_channel.default: 2,
        }
        # 创建激活函数的 PlaceholderObserver 实例
        act_affine_quant_obs = observer.PlaceholderObserver.with_args(
            dtype=torch.qint8,
            qscheme=torch.per_tensor_affine,
            quant_min=-128,
            quant_max=127,
            eps=2**-12,
            is_dynamic=True,
        )
        # 创建 QConfig 实例，指定激活函数和权重观察器
        qconfig = QConfig(
            activation=act_affine_quant_obs,
            weight=per_channel_weight_observer_range_neg_127_to_127,
        )
        # 创建 QConfigMapping 实例，并设置为全局配置
        qconfig_mapping = QConfigMapping().set_global(qconfig)
        
        # 在 2D 和 4D 输入上进行量化器测试
        example_inputs_2d = (torch.randn(9, 8),)
        example_inputs_4d = (torch.randn(9, 10, 11, 8),)
        for example_inputs in [example_inputs_2d, example_inputs_4d]:
            self._test_quantizer(
                m_eager,
                example_inputs,
                quantizer,
                node_occurrence,
                [],  # 空列表作为额外参数
                True,  # 布尔值参数
                qconfig_mapping,
            )
    # 定义一个测试方法，用于测试动态线性整数4位量化器
    def test_dynamic_linear_int4_weight(self):
        # 创建一个XNNPACK量化器实例
        quantizer = XNNPACKQuantizer()
        # 获取对称量化配置，用于权重按通道量化，动态量化，权重量化范围为0到15
        quantization_config = get_symmetric_quantization_config(
            is_per_channel=True,
            is_dynamic=True,
            weight_qmin=0,
            weight_qmax=15,
        )
        # 设置全局量化配置到量化器中
        quantizer.set_global(quantization_config)
        # 创建一个测试帮助模块的两层线性模块，并将其设为评估模式
        m_eager = TestHelperModules.TwoLinearModule().eval()

        # 定义节点出现次数的字典，指定了输入和输出使用per_tensor量化，权重使用per_channel量化
        node_occurrence = {
            torch.ops.quantized_decomposed.quantize_per_tensor.tensor: 2,
            torch.ops.quantized_decomposed.dequantize_per_tensor.tensor: 2,
            torch.ops.quantized_decomposed.quantize_per_channel.default: 0,  # 注意：权重的量化操作是常数传播的
            torch.ops.quantized_decomposed.dequantize_per_channel.default: 2,
        }
        # 创建用于激活的占位符观察器，采用torch.qint8整数类型，per_tensor_affine量化方案，
        # 量化范围为-128到127，epsilon为2的-12次方，采用动态量化
        act_affine_quant_obs = observer.PlaceholderObserver.with_args(
            dtype=torch.qint8,
            qscheme=torch.per_tensor_affine,
            quant_min=-128,
            quant_max=127,
            eps=2**-12,
            is_dynamic=True,
        )
        # 定义量化配置qconfig，指定了激活使用act_affine_quant_obs观察器，
        # 权重使用per_channel_weight_observer_range_neg_127_to_127观察器，量化范围为0到15
        qconfig = QConfig(
            activation=act_affine_quant_obs,
            weight=per_channel_weight_observer_range_neg_127_to_127.with_args(
                quant_min=0, quant_max=15
            ),
        )
        # 创建量化配置映射qconfig_mapping，并设置为全局配置
        qconfig_mapping = QConfigMapping().set_global(qconfig)
        
        # 测试2维输入和4维输入的情况
        example_inputs_2d = (torch.randn(9, 8),)
        example_inputs_4d = (torch.randn(9, 10, 11, 8),)
        for example_inputs in [example_inputs_2d, example_inputs_4d]:
            # 调用测试量化器方法_test_quantizer，传入模型m_eager，示例输入example_inputs，
            # 量化器quantizer，节点出现次数node_occurrence，空列表[]，True值，量化配置映射qconfig_mapping
            self._test_quantizer(
                m_eager,
                example_inputs,
                quantizer,
                node_occurrence,
                [],
                True,
                qconfig_mapping,
            )
    # 定义测试方法，用于测试动态量化的线性模型
    def test_qat_dynamic_linear(self):
        # 创建 XNNPACKQuantizer 实例，用于量化操作
        quantizer = XNNPACKQuantizer()
        # 获取对称量化配置，包括通道级量化、动态量化、量化训练等设置
        quantization_config = get_symmetric_quantization_config(
            is_per_channel=True,
            is_dynamic=True,
            is_qat=True,
        )
        # 将全局量化配置应用到量化器上
        quantizer.set_global(quantization_config)
        # 创建并加载一个测试用的两层线性模型（eager 模式），并设置为评估模式
        m_eager = TestHelperModules.TwoLinearModule().eval()

        # 定义节点出现次数的期望字典，指定了各操作在模型中出现的次数
        node_occurrence = {
            torch.ops.quantized_decomposed.choose_qparams.tensor: 2,
            # 输入和输出使用 quantize_per_tensor，权重使用 quantize_per_channel
            torch.ops.quantized_decomposed.quantize_per_tensor.tensor: 2,
            torch.ops.quantized_decomposed.dequantize_per_tensor.tensor: 2,
            # 注意：权重的量化操作是常量传播的
            torch.ops.quantized_decomposed.quantize_per_channel.default: 0,
            torch.ops.quantized_decomposed.dequantize_per_channel.default: 2,
        }
        # 定义激活函数的仿真量化观察器
        act_affine_quant_obs = default_dynamic_fake_quant
        # 定义量化配置对象，指定了激活函数和权重的量化观察器
        qconfig = QConfig(
            activation=act_affine_quant_obs,
            weight=per_channel_weight_observer_range_neg_127_to_127,
        )
        # 创建量化配置映射对象，并将全局量化配置应用于映射
        qconfig_mapping = QConfigMapping().set_global(qconfig)
        
        # 测试用例：使用 2D 和 4D 输入分别测试量化器的行为
        example_inputs_2d = (torch.randn(9, 8),)
        example_inputs_4d = (torch.randn(9, 10, 11, 8),)
        for example_inputs in [example_inputs_2d, example_inputs_4d]:
            # 执行量化器测试方法，验证量化模型的行为
            self._test_quantizer(
                m_eager,
                example_inputs,
                quantizer,
                node_occurrence,
                [],  # 暂无需传递附加参数
                True,  # 使用量化训练模式
                qconfig_mapping,
                is_qat=True,  # 明确指定为量化训练模式
            )
    # 定义一个测试方法，用于测试动态线性层和卷积的量化
    def test_dynamic_linear_with_conv(self):
        # 创建一个XNNPACKQuantizer实例，用于量化
        quantizer = XNNPACKQuantizer()
        # 获取对称量化配置，不按通道量化，动态量化
        quantization_config = get_symmetric_quantization_config(
            is_per_channel=False, is_dynamic=True
        )
        # 将全局量化配置应用到量化器中
        quantizer.set_global(quantization_config)
        # 创建一个评估模式下的ConvLinearWPermute实例
        m_eager = TestHelperModules.ConvLinearWPermute().eval()

        # 定义节点出现次数的字典，用于描述量化操作的使用情况
        node_occurrence = {
            # 输入和输出使用quantize_per_tensor，权重使用quantize_per_channel
            torch.ops.quantized_decomposed.quantize_per_tensor.tensor: 1,
            torch.ops.quantized_decomposed.dequantize_per_tensor.tensor: 1,
            # 注意：权重的量化操作是常数传播的
            torch.ops.quantized_decomposed.quantize_per_tensor.default: 0,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default: 1,
        }
        # 创建一个以占位符观察者为参数的激活量化观察器实例
        act_affine_quant_obs = observer.PlaceholderObserver.with_args(
            dtype=torch.qint8,
            qscheme=torch.per_tensor_affine,
            quant_min=-128,
            quant_max=127,
            eps=2**-12,
            is_dynamic=True,
        )
        # 创建一个量化配置对象QConfig
        qconfig = QConfig(
            activation=act_affine_quant_obs,
            weight=weight_observer_range_neg_127_to_127,
        )
        # 准备一个2D输入的示例
        example_inputs = (torch.randn(2, 3, 4, 4),)
        # 创建一个QConfigMapping对象并设置全局QConfig
        qconfig_mapping = QConfigMapping().set_global(qconfig)
        # 调用测试方法_test_quantizer进行量化器的测试
        self._test_quantizer(
            m_eager,
            example_inputs,
            quantizer,
            node_occurrence,
            [],
            True,
            qconfig_mapping,
        )
    def test_gru(self):
        """
        this is a test for annotating fp32 GRU so that it produces
        q -> dq -> fp32_gru -> q -> dq, this is currently enough for our use cases,
        but we may change the annotation to be more precise in the future
        """

        class RNNDynamicModel(torch.nn.Module):
            def __init__(self, mod_type):
                super().__init__()
                self.qconfig = default_dynamic_qconfig  # 设置量化配置为默认动态量化配置
                if mod_type == "GRU":
                    self.mod = torch.nn.GRU(2, 2).to(dtype=torch.float)  # 创建一个双向GRU模型，输入和隐藏状态的大小均为2
                if mod_type == "LSTM":
                    self.mod = torch.nn.LSTM(2, 2).to(dtype=torch.float)  # 创建一个双向LSTM模型，输入和隐藏状态的大小均为2

            def forward(self, input_tensor, hidden_tensor):
                input_tensor = 1 * input_tensor  # 复制输入张量
                hidden_tensor = 1 * hidden_tensor  # 复制隐藏张量
                output_tensor, hidden_out = self.mod(input_tensor, hidden_tensor)  # 在模型上执行前向传播
                return 1 * output_tensor, 1 * hidden_out  # 返回复制的输出张量和隐藏状态张量

        with override_quantized_engine("qnnpack"):  # 使用qnnpack作为量化引擎
            model_fx = RNNDynamicModel("GRU")  # 创建一个使用GRU模型的动态模型实例
            module_types = [torch.nn.GRU]  # 将GRU模型添加到模型类型列表中
            niter = 10  # 迭代次数设为10
            example_inputs = (
                # input_tensor
                torch.tensor([[100, -155], [-155, 100], [100, -155]], dtype=torch.float)
                .unsqueeze(0)
                .repeat(niter, 1, 1),  # 创建包含多次重复输入张量的示例输入
                # hidden_tensor
                # (D * num_layers, N, H_out)
                torch.tensor([[[100, -155]]], dtype=torch.float).repeat(1, 3, 1),  # 创建包含隐藏状态张量的示例输入
            )
            model_graph = copy.deepcopy(model_fx)  # 深度复制动态模型以创建静态图模型

            qconfig_mapping = QConfigMapping().set_object_type(
                operator.mul, default_symmetric_qnnpack_qconfig
            )  # 创建一个运算符乘法对应的量化配置映射
            model_fx = prepare_fx(
                model_fx,
                qconfig_mapping,
                example_inputs,
                backend_config=get_qnnpack_backend_config(),
            )  # 准备动态模型以进行量化
            model_fx(*example_inputs)  # 在示例输入上运行动态模型
            model_fx = _convert_to_reference_decomposed_fx(model_fx)  # 将动态模型转换为参考分解形式

            with torchdynamo.config.patch(allow_rnn=True):  # 使用torchdynamo配置允许RNN模型
                model_graph = capture_pre_autograd_graph(
                    model_graph,
                    example_inputs,
                )  # 捕获自动求导前的静态图模型

            quantizer = XNNPACKQuantizer()  # 创建一个XNNPACK量化器实例
            quantization_config = get_symmetric_quantization_config(
                is_per_channel=False, is_dynamic=False
            )  # 获取对称量化配置，不分通道，非动态
            quantizer.set_global(quantization_config)  # 设置全局量化配置
            model_graph = prepare_pt2e(model_graph, quantizer)  # 准备将静态图模型转换为pt2e
            model_graph(*example_inputs)  # 在示例输入上运行pt2e模型
            model_graph = convert_pt2e(model_graph)  # 将静态图模型转换为pt2e模型
            self.assertEqual(model_fx(*example_inputs), model_graph(*example_inputs))  # 断言动态模型和静态图模型的输出相等
    def test_linear_gru(self):
        """this test is to make sure GRU annotation does not interfere with linear annotation"""

        class RNNDynamicModel(torch.nn.Module):
            def __init__(self, mod_type):
                super().__init__()
                self.qconfig = default_dynamic_qconfig  # 设置模型的量化配置为默认的动态量化配置
                self.linear = torch.nn.Linear(2, 2)  # 创建一个线性层，输入和输出维度均为2
                if mod_type == "GRU":
                    self.mod = torch.nn.GRU(2, 2).to(dtype=torch.float)  # 如果模型类型为GRU，则创建一个GRU模型
                if mod_type == "LSTM":
                    self.mod = torch.nn.LSTM(2, 2).to(dtype=torch.float)  # 如果模型类型为LSTM，则创建一个LSTM模型

            def forward(self, input_tensor, hidden_tensor):
                input_tensor = self.linear(input_tensor)  # 将输入张量通过线性层处理
                input_tensor = 1 * input_tensor  # 乘以1，不改变张量
                hidden_tensor = 1 * hidden_tensor  # 乘以1，不改变张量
                output_tensor, hidden_out = self.mod(input_tensor, hidden_tensor)  # 将输入张量和隐藏张量传入RNN模型进行前向计算
                return 1 * output_tensor, 1 * hidden_out  # 返回乘以1后的输出张量和隐藏张量

        with override_quantized_engine("qnnpack"):
            model_fx = RNNDynamicModel("GRU")  # 创建一个动态RNN模型实例，模型类型为GRU
            module_types = [torch.nn.GRU]  # 指定模型类型列表包含GRU
            niter = 10  # 迭代次数设置为10
            example_inputs = (
                # input_tensor
                torch.tensor([[100, -155], [-155, 100], [100, -155]], dtype=torch.float)
                .unsqueeze(0)
                .repeat(niter, 1, 1),  # 创建一个示例输入张量，并重复niter次作为输入
                # hidden_tensor
                # (D * num_layers, N, H_out)
                torch.tensor([[[100, -155]]], dtype=torch.float).repeat(1, 3, 1),  # 创建一个示例隐藏张量
            )
            model_graph = copy.deepcopy(model_fx)  # 深拷贝动态模型实例为静态图模型

            qconfig_mapping = (
                QConfigMapping()
                .set_object_type(operator.mul, default_symmetric_qnnpack_qconfig)
                .set_object_type(torch.nn.Linear, default_symmetric_qnnpack_qconfig)
            )  # 创建量化配置映射，设置乘法运算符和线性层的量化配置为默认的对称量化配置

            model_fx = prepare_fx(
                model_fx,
                qconfig_mapping,
                example_inputs,
                backend_config=get_qnnpack_backend_config(),
            )  # 准备动态模型实例，应用量化配置映射和后端配置

            model_fx(*example_inputs)  # 对动态模型实例进行前向计算

            model_fx = _convert_to_reference_decomposed_fx(model_fx)  # 将动态模型实例转换为参考分解后的动态模型实例

            with torchdynamo.config.patch(allow_rnn=True):
                model_graph = capture_pre_autograd_graph(
                    model_graph,
                    example_inputs,
                )  # 使用torchdynamo捕获模型的预自动微分图

            quantizer = XNNPACKQuantizer()  # 创建XNNPACK量化器实例
            quantization_config = get_symmetric_quantization_config(
                is_per_channel=False, is_dynamic=False
            )  # 获取对称量化配置，通道数为False，动态量化为False

            quantizer.set_global(quantization_config)  # 设置量化器的全局量化配置

            model_graph = prepare_pt2e(model_graph, quantizer)  # 准备静态图模型，应用量化器

            model_graph(*example_inputs)  # 对静态图模型进行前向计算

            model_graph = convert_pt2e(model_graph)  # 将静态图模型转换为PT2E模型

            self.assertEqual(model_fx(*example_inputs), model_graph(*example_inputs))  # 断言动态模型实例和静态图模型的输出是否相等
    # 定义测试函数，测试加法和原位加法的量化效果
    def test_add_and_inplace_add(self):
        # 创建 XNNPACKQuantizer 实例
        quantizer = XNNPACKQuantizer()
        # 获取对称量化配置，每通道量化
        quantization_config = get_symmetric_quantization_config(is_per_channel=True)
        # 应用全局量化配置到量化器中
        quantizer.set_global(quantization_config)
        # 准备示例输入数据，包括两个形状为 (1, 3, 5, 5) 的张量
        example_inputs = (
            torch.randn(1, 3, 5, 5),
            torch.randn(1, 3, 5, 5),
        )
        # 指定节点发生次数的字典，描述第一个加法操作有两个输入和一个输出，第二个加法操作只有输出
        node_occurrence = {
            torch.ops.quantized_decomposed.quantize_per_tensor.default: 4,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default: 5,
        }
        # 定义节点列表，包括量化、反量化和加法操作
        node_list = [
            torch.ops.quantized_decomposed.dequantize_per_tensor.default,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default,
            torch.ops.aten.add.Tensor,
            torch.ops.quantized_decomposed.quantize_per_tensor.default,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default,
            # TODO torch.ops.aten.add.Tensor,
            torch.ops.quantized_decomposed.quantize_per_tensor.default,
        ]
        # 执行量化器测试，传入测试模块、示例输入、量化器、节点出现次数和节点列表
        self._test_quantizer(
            TestHelperModules.AddInplaceAdd(),
            example_inputs,
            quantizer,
            node_occurrence,
            node_list,
        )

    # 定义测试函数，测试乘法和原位乘法的量化效果
    def test_mul_and_inplace_mul(self):
        # 创建 XNNPACKQuantizer 实例
        quantizer = XNNPACKQuantizer()
        # 获取对称量化配置，每通道量化
        quantization_config = get_symmetric_quantization_config(is_per_channel=True)
        # 应用全局量化配置到量化器中
        quantizer.set_global(quantization_config)
        # 准备示例输入数据，包括两个形状为 (1, 3, 5, 5) 的张量
        example_inputs = (
            torch.randn(1, 3, 5, 5),
            torch.randn(1, 3, 5, 5),
        )
        # 指定节点发生次数的字典，描述第一个加法操作有两个输入和一个输出，第二个加法操作只有输出
        node_occurrence = {
            torch.ops.quantized_decomposed.quantize_per_tensor.default: 4,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default: 5,
        }
        # 定义节点列表，包括量化、反量化和乘法操作
        node_list = [
            torch.ops.quantized_decomposed.dequantize_per_tensor.default,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default,
            torch.ops.aten.mul.Tensor,
            torch.ops.quantized_decomposed.quantize_per_tensor.default,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default,
            # TODO torch.ops.aten.mul.Tensor,
            torch.ops.quantized_decomposed.quantize_per_tensor.default,
        ]
        # 执行量化器测试，传入测试模块、示例输入、量化器、节点出现次数和节点列表
        self._test_quantizer(
            TestHelperModules.MulInplaceMul(),
            example_inputs,
            quantizer,
            node_occurrence,
            node_list,
        )
    # 定义一个测试方法，用于测试加法和乘法的量化操作
    def test_add_mul_scalar(self):
        # 创建一个 XNNPACKQuantizer 实例
        quantizer = XNNPACKQuantizer()
        # 获取对称量化配置，通道独立
        quantization_config = get_symmetric_quantization_config(is_per_channel=True)
        # 设置全局量化配置
        quantizer.set_global(quantization_config)
        # 准备示例输入数据，一个形状为 (1, 3, 5, 5) 的张量
        example_inputs = (torch.randn(1, 3, 5, 5),)
        # 定义节点出现的频率字典，指定了每个节点的出现次数
        node_occurrence = {
            # 第一个加法有两个输入和一个输出，第二个加法有一个输出
            torch.ops.quantized_decomposed.quantize_per_tensor.default: 5,
            # TODO torch.ops.quantized_decomposed.dequantize_per_tensor.default: 9,
        }
        # 定义节点列表，描述了量化和非量化操作的顺序
        node_list = [
            torch.ops.quantized_decomposed.dequantize_per_tensor.default,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default,
            torch.ops.aten.add.Tensor,
            torch.ops.quantized_decomposed.quantize_per_tensor.default,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default,
            torch.ops.aten.mul.Tensor,
            torch.ops.quantized_decomposed.quantize_per_tensor.default,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default,
            # TODO torch.ops.aten.add.Tensor,
            torch.ops.quantized_decomposed.quantize_per_tensor.default,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default,
            # TODO torch.ops.aten.mul.Tensor,
            torch.ops.quantized_decomposed.quantize_per_tensor.default,
        ]
        # 使用自定义方法 _test_quantizer 进行量化器测试
        self._test_quantizer(
            TestHelperModules.AddMulScalar(),  # 使用 AddMulScalar 模块进行测试
            example_inputs,  # 示例输入数据
            quantizer,  # 使用的量化器实例
            node_occurrence,  # 节点出现频率的字典
            node_list,  # 节点列表，描述了操作的顺序
        )

    # 测试浮点数乘法的量化
    def test_mul_float32_max(self):
        # 定义一个简单的 PyTorch 模块 M
        class M(torch.nn.Module):
            def forward(self, x):
                return x * 3.4028235e38

        # 创建一个 XNNPACKQuantizer 实例
        quantizer = XNNPACKQuantizer()
        # 获取对称量化配置，通道独立
        quantization_config = get_symmetric_quantization_config(is_per_channel=True)
        # 设置全局量化配置
        quantizer.set_global(quantization_config)
        # 准备示例输入数据，一个形状为 (1, 3, 5, 5) 的张量
        example_inputs = (torch.randn(1, 3, 5, 5),)
        # 指定节点出现的频率，这里都是未量化的情况
        node_occurrence = {
            torch.ops.quantized_decomposed.quantize_per_tensor.default: 0,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default: 0,
        }
        # 节点列表中仅包含一个乘法操作
        node_list = [
            torch.ops.aten.mul.Tensor,
        ]
        # 使用自定义方法 _test_quantizer 进行量化器测试
        self._test_quantizer(
            M(),  # 使用定义的简单模块 M 进行测试
            example_inputs,  # 示例输入数据
            quantizer,  # 使用的量化器实例
            node_occurrence,  # 节点出现频率的字典
            node_list,  # 节点列表，描述了操作的顺序
        )
    # 定义一个测试方法，测试长整数加法和乘法的量化行为
    def test_add_mul_long(self):
        # 定义一个简单的神经网络模型
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 初始化一个包含数值 100 的张量
                self.t = torch.tensor([100])

            def forward(self, x):
                # 执行输入张量 x 和 self.t 的加法操作
                x = x + self.t
                # 执行结果张量 x 和 self.t 的乘法操作
                x = x * self.t
                return x

        # 实例化一个 XNNPACKQuantizer 对象
        quantizer = XNNPACKQuantizer()
        # 获取对称量化配置，设置为按通道量化
        quantization_config = get_symmetric_quantization_config(is_per_channel=True)
        # 将量化配置应用到全局
        quantizer.set_global(quantization_config)
        # 创建一个示例输入张量
        example_inputs = (torch.randn(1, 3, 5, 5),)
        # 创建一个节点发生次数字典，初始值为 0，表示未量化
        node_occurrence = {
            torch.ops.quantized_decomposed.quantize_per_tensor.default: 0,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default: 0,
        }
        # 创建一个节点列表，包含要测试的张量操作节点
        node_list = [
            torch.ops.aten.add.Tensor,
            torch.ops.aten.mul.Tensor,
        ]
        # 调用内部方法 _test_quantizer，测试量化器行为
        self._test_quantizer(
            M(),  # 使用定义的模型 M 进行测试
            example_inputs,  # 使用示例输入数据 example_inputs 进行测试
            quantizer,  # 使用设置好的量化器 quantizer 进行测试
            node_occurrence,  # 使用定义的节点发生次数字典 node_occurrence 进行测试
            node_list,  # 使用定义的节点列表 node_list 进行测试
        )
# 在 TestXNNPACKQuantizerModels 类中定义了一个测试用例类，继承自 PT2EQuantizationTestCase
class TestXNNPACKQuantizerModels(PT2EQuantizationTestCase):
    # 装饰器，如果没有安装 torchvision，则跳过测试
    @skip_if_no_torchvision
    # 装饰器，如果没有安装 QNNPACK，则跳过测试
    @skipIfNoQNNPACK
    # 测试 ResNet18 模型
    def test_resnet18(self):
        # 导入 torchvision 库
        import torchvision

        # 使用 qnnpack 引擎进行量化
        with override_quantized_engine("qnnpack"):
            # 创建一个示例输入
            example_inputs = (torch.randn(1, 3, 224, 224),)
            # 实例化 ResNet18 模型并设置为评估模式
            m = torchvision.models.resnet18().eval()
            # 深拷贝 ResNet18 模型
            m_copy = copy.deepcopy(m)
            # 对模型进行捕获前自动微分图
            m = capture_pre_autograd_graph(
                m,
                example_inputs,
            )

            # 实例化 XNNPACKQuantizer
            quantizer = XNNPACKQuantizer()
            # 获取对称量化配置
            quantization_config = get_symmetric_quantization_config(is_per_channel=True)
            # 设置全局量化配置
            quantizer.set_global(quantization_config)
            # 对模型进行准备
            m = prepare_pt2e(m, quantizer)
            # 检查我们是否正确插入了观察者以用于最大池化运算符（输入和输出共享观察者实例）
            self.assertEqual(
                id(m.activation_post_process_3), id(m.activation_post_process_2)
            )
            # 在准备后的结果上执行模型
            after_prepare_result = m(*example_inputs)
            # 将模型转换为参考 FX 图模式
            m = convert_pt2e(m)

            after_quant_result = m(*example_inputs)

            # 与现有的 FX 图模式量化参考流程进行比较
            qconfig = default_per_channel_symmetric_qnnpack_qconfig
            qconfig_mapping = QConfigMapping().set_global(qconfig)
            backend_config = get_qnnpack_backend_config()
            m_fx = prepare_fx(
                m_copy, qconfig_mapping, example_inputs, backend_config=backend_config
            )
            after_prepare_result_fx = m_fx(*example_inputs)
            m_fx = convert_to_reference_fx(m_fx, backend_config=backend_config)

            after_quant_result_fx = m_fx(*example_inputs)

            # 在准备后的结果完全匹配
            self.assertEqual(after_prepare_result, after_prepare_result_fx)
            self.assertEqual(
                compute_sqnr(after_prepare_result, after_prepare_result_fx),
                torch.tensor(float("inf")),
            )
            # 在转换后由于量化/反量化的不同实现会有轻微差异
            self.assertTrue(
                torch.max(after_quant_result - after_quant_result_fx) < 1e-1
            )
            self.assertTrue(
                compute_sqnr(after_quant_result, after_quant_result_fx) > 35
            )
```