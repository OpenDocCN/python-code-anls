# `.\pytorch\test\quantization\pt2e\test_representation.py`

```py
# Owner(s): ["oncall: quantization"]
# 导入必要的库和模块
import copy  # 导入copy模块，用于深拷贝对象
from typing import Any, Dict, Tuple  # 导入类型提示模块

import torch  # 导入PyTorch库
from torch._export import capture_pre_autograd_graph  # 导入捕获前向图函数
from torch._higher_order_ops.out_dtype import out_dtype  # noqa: F401
from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_pt2e  # 导入量化相关函数
from torch.ao.quantization.quantizer import Quantizer  # 导入量化器类
from torch.ao.quantization.quantizer.xnnpack_quantizer import (  # 导入XNNPACK量化器相关函数
    get_symmetric_quantization_config,
    XNNPACKQuantizer,
)
from torch.testing._internal.common_quantization import (  # 导入量化测试相关模块
    NodeSpec as ns,
    QuantizationTestCase,
    skipIfNoQNNPACK,
    TestHelperModules,
)

# 使用skipIfNoQNNPACK装饰器标记测试类，仅当QNNPACK可用时运行测试
@skipIfNoQNNPACK
class TestPT2ERepresentation(QuantizationTestCase):
    # 定义测试量化表示方法的方法
    def _test_representation(
        self,
        model: torch.nn.Module,
        example_inputs: Tuple[Any, ...],
        quantizer: Quantizer,
        ref_node_occurrence: Dict[ns, int],
        non_ref_node_occurrence: Dict[ns, int],
        fixed_output_tol: float = None,
        output_scale_idx: int = 2,
    ) -> torch.nn.Module:
        # 重置动态缓存
        torch._dynamo.reset()
        # 捕获模型的前向图并返回新模型
        model = capture_pre_autograd_graph(
            model,
            example_inputs,
        )
        # 深拷贝模型
        model_copy = copy.deepcopy(model)

        # 准备模型的PT2E表示形式
        model = prepare_pt2e(model, quantizer)
        # 进行校准
        model(*example_inputs)
        # 将模型转换为PT2E表示形式，并使用参考表示
        model = convert_pt2e(model, use_reference_representation=True)
        # 检查模型的图节点是否与预期一致
        self.checkGraphModuleNodes(model, expected_node_occurrence=ref_node_occurrence)
        # 确保模型可以运行
        pt2e_quant_output = model(*example_inputs)

        # TODO: 当torchdynamo修复后，可以启用数值检查
        model_copy = prepare_pt2e(model_copy, quantizer)
        # 进行校准
        model_copy(*example_inputs)
        # 将模型转换为PT2E表示形式，并使用非参考表示
        model_copy = convert_pt2e(model_copy, use_reference_representation=False)
        # 检查模型的图节点是否与预期的非参考节点一致
        self.checkGraphModuleNodes(
            model_copy, expected_node_occurrence=non_ref_node_occurrence
        )
        # 确保模型可以运行
        pt2e_quant_output_copy = model_copy(*example_inputs)

        output_tol = None
        # 如果提供了固定的输出容差值，则使用该值
        if fixed_output_tol is not None:
            output_tol = fixed_output_tol
        else:
            idx = 0
            # 遍历模型的图节点，查找量化整数表示的输出标度索引
            for n in model_copy.graph.nodes:
                if (
                    n.target
                    == torch.ops.quantized_decomposed.quantize_per_tensor.default
                ):
                    idx += 1
                    if idx == output_scale_idx:
                        output_tol = n.args[1]
            # 确保找到了输出容差值
            assert output_tol is not None

        # 确保量化整数表示的结果在最多一个单位误差内
        self.assertTrue(
            torch.max(torch.abs(pt2e_quant_output_copy - pt2e_quant_output))
            <= (2 * output_tol + 1e-5)
        )
    def test_static_linear(self):
        # 定义一个包含线性层的简单神经网络模型
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(5, 5)

            def forward(self, x):
                return self.linear(x)

        # 创建一个 XNNPACKQuantizer 的实例
        quantizer = XNNPACKQuantizer()
        # 获取对称量化配置，全局设置，不按通道量化
        operator_config = get_symmetric_quantization_config(is_per_channel=False)
        # 应用全局量化配置到量化器中
        quantizer.set_global(operator_config)
        # 创建一个示例输入，包含一个形状为 (2, 5) 的随机张量
        example_inputs = (torch.randn(2, 5),)

        # 调用 _test_representation 方法，验证模型的表示
        self._test_representation(
            M().eval(),  # 评估模式下的 M 模型实例
            example_inputs,  # 示例输入
            quantizer,  # 量化器实例
            ref_node_occurrence={},  # 引用节点出现次数空字典
            non_ref_node_occurrence={},  # 非引用节点出现次数空字典
        )

    def test_dynamic_linear(self):
        # 定义一个包含线性层的简单神经网络模型
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(5, 5)

            def forward(self, x):
                return self.linear(x)

        # 创建一个 XNNPACKQuantizer 的实例
        quantizer = XNNPACKQuantizer()
        # 获取对称量化配置，全局设置，按通道动态量化
        operator_config = get_symmetric_quantization_config(
            is_per_channel=False, is_dynamic=True
        )
        # 应用全局量化配置到量化器中
        quantizer.set_global(operator_config)
        # 创建一个示例输入，包含一个形状为 (2, 5) 的随机张量
        example_inputs = (torch.randn(2, 5),)

        # 调用 _test_representation 方法，验证模型的表示
        self._test_representation(
            M().eval(),  # 评估模式下的 M 模型实例
            example_inputs,  # 示例输入
            quantizer,  # 量化器实例
            ref_node_occurrence={},  # 引用节点出现次数空字典
            non_ref_node_occurrence={},  # 非引用节点出现次数空字典
            fixed_output_tol=1e-4,  # 固定输出容差设为 0.0001
        )

    def test_conv2d(self):
        # 定义一个包含卷积层的简单神经网络模型
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv2d = torch.nn.Conv2d(3, 3, 3)

            def forward(self, x):
                return self.conv2d(x)

        # 创建一个 XNNPACKQuantizer 的实例
        quantizer = XNNPACKQuantizer()
        # 获取对称量化配置，全局设置，不按通道量化
        operator_config = get_symmetric_quantization_config(is_per_channel=False)
        # 应用全局量化配置到量化器中
        quantizer.set_global(operator_config)
        # 创建一个示例输入，包含一个形状为 (1, 3, 3, 3) 的随机张量
        example_inputs = (torch.randn(1, 3, 3, 3),)

        # 调用 _test_representation 方法，验证模型的表示
        self._test_representation(
            M().eval(),  # 评估模式下的 M 模型实例
            example_inputs,  # 示例输入
            quantizer,  # 量化器实例
            ref_node_occurrence={},  # 引用节点出现次数空字典
            non_ref_node_occurrence={},  # 非引用节点出现次数空字典
        )

    def test_add(self):
        # 定义一个简单的加法运算模型
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                return x + y

        # 创建一个 XNNPACKQuantizer 的实例
        quantizer = XNNPACKQuantizer()
        # 获取对称量化配置，全局设置，按通道量化
        quantization_config = get_symmetric_quantization_config(is_per_channel=True)
        # 应用全局量化配置到量化器中
        quantizer.set_global(quantization_config)
        # 创建一个示例输入，包含两个形状为 (1, 3, 3, 3) 的随机张量
        example_inputs = (
            torch.randn(1, 3, 3, 3),
            torch.randn(1, 3, 3, 3),
        )

        # 调用 _test_representation 方法，验证模型的表示
        self._test_representation(
            M().eval(),  # 评估模式下的 M 模型实例
            example_inputs,  # 示例输入
            quantizer,  # 量化器实例
            ref_node_occurrence={},  # 引用节点出现次数空字典
            non_ref_node_occurrence={},  # 非引用节点出现次数空字典
        )
    # 定义一个测试类，用于测试添加ReLU激活函数的功能
    def test_add_relu(self):
        # 定义一个简单的神经网络模型类M，继承自torch.nn.Module
        class M(torch.nn.Module):
            # 构造函数，初始化模型
            def __init__(self):
                super().__init__()

            # 前向传播函数，接受输入x和y，返回添加ReLU后的输出
            def forward(self, x, y):
                # 计算输入的和
                out = x + y
                # 对和进行ReLU激活函数处理
                out = torch.nn.functional.relu(out)
                return out

        # 创建一个XNNPACKQuantizer的实例
        quantizer = XNNPACKQuantizer()
        # 获取对称量化配置，每通道独立的配置
        operator_config = get_symmetric_quantization_config(is_per_channel=True)
        # 设置全局量化配置
        quantizer.set_global(operator_config)

        # 创建示例输入数据
        example_inputs = (
            torch.randn(1, 3, 3, 3),  # 输入1
            torch.randn(1, 3, 3, 3),  # 输入2
        )
        # 参考节点出现次数的预期字典
        ref_node_occurrence = {
            ns.call_function(out_dtype): 2,  # 调用函数(out_dtype)出现2次
        }

        # 调用测试辅助函数_test_representation，验证模型的表示是否正确
        self._test_representation(
            M().eval(),  # 评估模型M的表示
            example_inputs,  # 示例输入数据
            quantizer,  # 量化器
            ref_node_occurrence=ref_node_occurrence,  # 参考节点出现次数
            non_ref_node_occurrence={},  # 非参考节点出现次数为空字典
        )

    # 定义测试最大池化操作的函数
    def test_maxpool2d(self):
        # 创建一个XNNPACKQuantizer的实例
        quantizer = XNNPACKQuantizer()
        # 获取对称量化配置，每通道独立的配置
        operator_config = get_symmetric_quantization_config(is_per_channel=True)
        # 设置全局量化配置
        quantizer.set_global(operator_config)
        
        # 创建一个ConvMaxPool2d的测试辅助模块实例，并评估
        m_eager = TestHelperModules.ConvMaxPool2d().eval()

        # 创建示例输入数据
        example_inputs = (torch.randn(1, 2, 2, 2),)

        # 调用测试辅助函数_test_representation，验证模型的表示是否正确
        self._test_representation(
            m_eager,  # ConvMaxPool2d模块实例
            example_inputs,  # 示例输入数据
            quantizer,  # 量化器
            ref_node_occurrence={},  # 参考节点出现次数为空字典
            non_ref_node_occurrence={},  # 非参考节点出现次数为空字典
        )
    def test_qdq_per_channel(self):
        """Test representation for quantize_per_channel and dequantize_per_channel op"""
        
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(5, 5)

            def forward(self, x):
                return self.linear(x)

        # 创建一个 XNNPACKQuantizer 的实例
        quantizer = XNNPACKQuantizer()
        
        # 配置操作符使用每通道量化
        operator_config = get_symmetric_quantization_config(is_per_channel=True)
        quantizer.set_global(operator_config)
        
        # 创建一个评估模式下的 M 实例
        m_eager = M().eval()

        # 定义多个输入例子，每个例子包含不同维度的随机张量
        inputs = [
            (torch.randn(1, 5),),
            (torch.randn(1, 3, 5),),
            (torch.randn(1, 3, 3, 5),),
            (torch.randn(1, 3, 3, 3, 5),),
        ]
        
        # 遍历每个输入例子
        for example_inputs in inputs:
            # 初始化参考节点出现次数字典和非参考节点出现次数字典
            ref_node_occurrence = {
                # quantize_per_channel 默认函数调用次数设为 0
                ns.call_function(
                    torch.ops.quantized_decomposed.quantize_per_channel.default
                ): 0,
                # dequantize_per_channel 默认函数调用次数设为 0
                ns.call_function(
                    torch.ops.quantized_decomposed.dequantize_per_channel.default
                ): 0,
            }
            
            non_ref_node_occurrence = {
                # quantize_per_channel 被折叠后的函数调用次数设为 0
                ns.call_function(
                    torch.ops.quantized_decomposed.quantize_per_channel.default
                ): 0,
                # dequantize_per_channel 默认函数调用次数设为 1
                ns.call_function(
                    torch.ops.quantized_decomposed.dequantize_per_channel.default
                ): 1,
            }

            # 执行测试函数 _test_representation，验证模型的表示
            self._test_representation(
                M().eval(),
                example_inputs,
                quantizer,
                ref_node_occurrence,
                non_ref_node_occurrence,
                output_scale_idx=2,
            )
    def test_qdq(self):
        """Test representation for quantize and dequantize op"""

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                # 简单地返回输入张量的和
                return x + y

        # 创建一个 XNNPACKQuantizer 实例
        quantizer = XNNPACKQuantizer()
        # 获取对称量化配置，每通道独立量化
        quantization_config = get_symmetric_quantization_config(is_per_channel=True)
        # 设置全局量化配置
        quantizer.set_global(quantization_config)
        # 创建一个模型实例并设置为评估模式
        m_eager = M().eval()

        # 创建示例输入张量
        example_inputs = (
            torch.randn(1, 3, 3, 3),
            torch.randn(1, 3, 3, 3),
        )

        # 参考节点出现次数字典，初始化为0
        ref_node_occurrence = {
            ns.call_function(torch.ops.quantized_decomposed.quantize_per_tensor): 0,
            ns.call_function(torch.ops.quantized_decomposed.dequantize_per_tensor): 0,
        }

        # 非参考节点出现次数字典，设置默认量化和反量化函数的初始次数为3
        non_ref_node_occurrence = {
            ns.call_function(
                torch.ops.quantized_decomposed.quantize_per_tensor.default
            ): 3,
            ns.call_function(
                torch.ops.quantized_decomposed.dequantize_per_tensor.default
            ): 3,
        }

        # 调用 _test_representation 方法进行测试模型表示
        self._test_representation(
            M().eval(),  # 使用评估模式的模型实例
            example_inputs,  # 示例输入
            quantizer,  # 量化器实例
            ref_node_occurrence,  # 参考节点出现次数字典
            non_ref_node_occurrence,  # 非参考节点出现次数字典
        )
```