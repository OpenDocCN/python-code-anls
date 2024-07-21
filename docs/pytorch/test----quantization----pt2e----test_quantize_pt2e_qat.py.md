# `.\pytorch\test\quantization\pt2e\test_quantize_pt2e_qat.py`

```py
# 导入必要的模块和类
# Owner(s): ["oncall: quantization"]
import copy  # 导入 copy 模块，用于对象的深拷贝操作
import operator  # 导入 operator 模块，提供了许多内置运算符的函数实现
import unittest  # 导入 unittest 模块，用于编写和运行单元测试
from typing import Any, Optional, Tuple, Type  # 导入类型提示相关的模块

import torch  # 导入 PyTorch 模块
from torch._export import capture_pre_autograd_graph  # 导入函数 capture_pre_autograd_graph
from torch.ao.quantization import (  # 导入量化相关模块和类
    default_fake_quant,
    FusedMovingAvgObsFakeQuantize,
    MovingAverageMinMaxObserver,
    MovingAveragePerChannelMinMaxObserver,
    QConfigMapping,
)
from torch.ao.quantization.backend_config import get_qnnpack_backend_config  # 获取 QNNPACK 后端配置
from torch.ao.quantization.qconfig import (  # 导入量化配置相关类和函数
    default_per_channel_symmetric_qnnpack_qat_qconfig,
    default_symmetric_qnnpack_qat_qconfig,
)
from torch.ao.quantization.quantize_fx import prepare_qat_fx  # 导入固定点量化 QAT 准备函数
from torch.ao.quantization.quantize_pt2e import (  # 导入 PyTorch 到 Essentia 转换相关函数
    _convert_to_reference_decomposed_fx,
    convert_pt2e,
    prepare_pt2e,
    prepare_qat_pt2e,
)
from torch.ao.quantization.quantizer import (  # 导入量化器相关类和函数
    DerivedQuantizationSpec,
    QuantizationAnnotation,
    QuantizationSpec,
    Quantizer,
)
from torch.ao.quantization.quantizer.xnnpack_quantizer import (  # 导入 XNNPACK 量化器相关类和函数
    get_symmetric_quantization_config,
    XNNPACKQuantizer,
)
from torch.testing._internal.common_cuda import TEST_CUDA  # 导入 CUDA 相关测试标记
from torch.testing._internal.common_quantization import (  # 导入量化相关测试辅助函数和类
    NodeSpec as ns,
    QuantizationTestCase,
    skip_if_no_torchvision,
    skipIfNoQNNPACK,
)
from torch.testing._internal.common_quantized import override_quantized_engine  # 导入覆盖量化引擎的函数


class PT2EQATTestCase(QuantizationTestCase):
    """
    Base QuantizationTestCase for PT2E QAT with some helper methods.
    """

    class _BaseConvBnModel(torch.nn.Module):
        def __init__(
            self,
            conv_class: Type[torch.nn.Module],
            bn_class: Type[torch.nn.Module],
            has_conv_bias: bool,
            has_bn: bool,
            has_relu: bool,
            **conv_kwargs,
        ):
            super().__init__()
            # 设置默认的卷积参数
            conv_kwargs.setdefault("in_channels", 3)
            conv_kwargs.setdefault("out_channels", 3)
            conv_kwargs.setdefault("kernel_size", 3)
            conv_kwargs.setdefault("bias", has_conv_bias)
            # 创建卷积层和可选的批归一化层、ReLU 激活层
            self.conv = conv_class(**conv_kwargs)
            self.bn = bn_class(conv_kwargs["out_channels"]) if has_bn else None
            self.relu = torch.nn.ReLU() if has_relu else None

        def forward(self, x):
            # 前向传播函数，依次应用卷积、批归一化、ReLU
            x = self.conv(x)
            if self.bn is not None:
                x = self.bn(x)
            if self.relu is not None:
                x = self.relu(x)
            return x

    def _get_conv_bn_model(
        self,
        has_conv_bias: bool = True,
        has_bn: bool = True,
        has_relu: bool = False,
        transpose: bool = False,
        **conv_kwargs,
    ):
        """
        返回一个包含conv[-bn][-relu]模式的简单测试模型实例。默认返回带有conv偏置的conv-bn模型。
        """
        return self._BaseConvBnModel(
            self.conv_transpose_class if transpose else self.conv_class,
            self.bn_class,
            has_conv_bias,
            has_bn,
            has_relu,
            **conv_kwargs,
        )

    def _verify_symmetric_xnnpack_qat_numerics(
        self,
        model: torch.nn.Module,
        example_inputs: Tuple[Any, ...],
    ):
        """
        对称性验证_xnnpack_qat数值的辅助函数，用于验证模型在给定示例输入上的数值对称性。
        """
        self._verify_symmetric_xnnpack_qat_numerics_helper(
            model,
            example_inputs,
            is_per_channel=True,
        )
        self._verify_symmetric_xnnpack_qat_numerics_helper(
            model,
            example_inputs,
            is_per_channel=False,
        )

    def _verify_symmetric_xnnpack_qat_numerics_helper(
        self,
        model: torch.nn.Module,
        example_inputs: Tuple[Any, ...],
        is_per_channel: bool,
        verify_convert: bool = True,
        ):
        """
        对称性验证_xnnpack_qat数值的辅助函数，用于验证模型在给定示例输入上的数值对称性。
        """
    ):
        """
        Helper method to verify that the QAT numerics for PT2E quantization match those of
        FX graph mode quantization for symmetric qnnpack.
        """
        # resetting dynamo cache
        torch._dynamo.reset()
        MANUAL_SEED = 100

        # PT2 export

        # 深拷贝原始模型，以便修改不影响原模型
        model_pt2e = copy.deepcopy(model)
        # 创建 XNNPACKQuantizer 实例
        quantizer = XNNPACKQuantizer()
        # 设置全局量化配置为对称量化配置，用于 QAT（Quantization Aware Training）
        quantizer.set_global(
            get_symmetric_quantization_config(
                is_per_channel=is_per_channel, is_qat=True
            )
        )
        # 捕获预自动微分图并返回修改后的模型
        model_pt2e = capture_pre_autograd_graph(
            model_pt2e,
            example_inputs,
        )
        # 准备 QAT PT2E 模型
        model_pt2e = prepare_qat_pt2e(model_pt2e, quantizer)
        torch.manual_seed(MANUAL_SEED)
        # 对 PT2E 模型进行推理
        after_prepare_result_pt2e = model_pt2e(*example_inputs)

        # FX export

        # 深拷贝原始模型，以便修改不影响原模型
        model_fx = copy.deepcopy(model)
        # 根据是否按通道量化选择默认量化配置
        if is_per_channel:
            default_qconfig = default_per_channel_symmetric_qnnpack_qat_qconfig
        else:
            default_qconfig = default_symmetric_qnnpack_qat_qconfig
        # 创建 QConfigMapping 实例并设置全局量化配置
        qconfig_mapping = QConfigMapping().set_global(default_qconfig)
        # 获取 QNNPACK 后端配置
        backend_config = get_qnnpack_backend_config()
        # 准备 QAT FX 模型
        model_fx = prepare_qat_fx(
            model_fx, qconfig_mapping, example_inputs, backend_config=backend_config
        )
        torch.manual_seed(MANUAL_SEED)
        # 对 FX 模型进行推理
        after_prepare_result_fx = model_fx(*example_inputs)

        # Verify that numerics match
        # 验证两种导出模型的数值结果是否一致
        self.assertEqual(after_prepare_result_pt2e, after_prepare_result_fx)

        if verify_convert:
            # 如果需要验证转换

            # 将导出的 PT2E 模型移至评估模式
            torch.ao.quantization.move_exported_model_to_eval(model_pt2e)
            # 转换 PT2E 模型
            model_pt2e = convert_pt2e(model_pt2e)
            # 对转换后的 PT2E 模型进行量化推理
            quant_result_pt2e = model_pt2e(*example_inputs)
            
            # 将 FX 模型设置为评估模式
            model_fx.eval()
            # 将 FX 模型转换为参考分解版本
            model_fx = _convert_to_reference_decomposed_fx(
                model_fx,
                backend_config=backend_config,
            )
            # 对转换后的 FX 模型进行量化推理
            quant_result_fx = model_fx(*example_inputs)
            
            # 验证量化后的 PT2E 和 FX 模型的推理结果是否一致
            self.assertEqual(quant_result_pt2e, quant_result_fx)

    def _verify_symmetric_xnnpack_qat_graph(
        self,
        m: torch.fx.GraphModule,
        example_inputs: Tuple[Any, ...],
        has_relu: bool,
        has_bias: bool = True,
        is_cuda: bool = False,
        expected_conv_literal_args: Optional[Tuple[Any, ...]] = None,
        # TODO: set this to true by default
        verify_convert: bool = False,
    ):
        # 使用 _verify_symmetric_xnnpack_qat_graph_helper 方法来验证对称的 XNNPACK QAT 图
        self._verify_symmetric_xnnpack_qat_graph_helper(
            # 使用传入的 torch.fx.GraphModule 对象 m 进行验证
            m,
            # 使用示例输入 example_inputs 来验证
            example_inputs,
            # 设置为按通道进行量化的验证
            is_per_channel=True,
            # 是否包含 ReLU 激活函数的验证条件
            has_relu=has_relu,
            # 是否包含偏置的验证条件
            has_bias=has_bias,
            # 是否在 CUDA 上进行验证
            is_cuda=is_cuda,
            # 预期的卷积文字参数的验证
            expected_conv_literal_args=expected_conv_literal_args,
            # 是否验证转换的条件
            verify_convert=verify_convert,
        )
        # 再次使用 _verify_symmetric_xnnpack_qat_graph_helper 方法进行验证，但这次设置为非按通道进行量化的验证
        self._verify_symmetric_xnnpack_qat_graph_helper(
            # 使用传入的 torch.fx.GraphModule 对象 m 进行验证
            m,
            # 使用示例输入 example_inputs 来验证
            example_inputs,
            # 设置为非按通道进行量化的验证
            is_per_channel=False,
            # 是否包含 ReLU 激活函数的验证条件
            has_relu=has_relu,
            # 是否包含偏置的验证条件
            has_bias=has_bias,
            # 是否在 CUDA 上进行验证
            is_cuda=is_cuda,
            # 预期的卷积文字参数的验证
            expected_conv_literal_args=expected_conv_literal_args,
            # 是否验证转换的条件
            verify_convert=verify_convert,
        )

    # 定义一个私有方法 _verify_symmetric_xnnpack_qat_graph_helper
    def _verify_symmetric_xnnpack_qat_graph_helper(
        self,
        # 传入 torch.fx.GraphModule 对象 m，用于验证图
        m: torch.fx.GraphModule,
        # 示例输入 example_inputs，用于执行图的验证
        example_inputs: Tuple[Any, ...],
        # 是否按通道进行量化的标志
        is_per_channel: bool,
        # 是否有 ReLU 激活函数的标志
        has_relu: bool,
        # 是否有偏置的标志，默认为 True
        has_bias: bool = True,
        # 是否在 CUDA 上运行的标志，默认为 False
        is_cuda: bool = False,
        # 期望的卷积文字参数的元组，用于验证
        expected_conv_literal_args: Optional[Tuple[Any, ...]] = None,
        # 是否验证转换的标志，默认为 False
        verify_convert: bool = False,
class TestQuantizePT2EQAT_ConvBn_Base(PT2EQATTestCase):
    """
    Base TestCase to be used for all conv-bn[-relu] fusion patterns.
    """

    # TODO: how can we avoid adding every new test to dynamo/expected_test_failures?
    # Otherwise it fails with the following error:
    #   torch._dynamo.exc.InternalTorchDynamoError:
    #   'QuantizationConfig' object has no attribute '__bool__'

    def setUp(self):
        # NB: Skip the test if this is a base class, this is to handle the test
        # discovery logic in buck which finds and runs all tests here including
        # the base class which we don't want to run
        if self.id() and "_Base" in self.id():
            self.skipTest("Skipping test running from base class")

    def test_qat_conv_no_bias(self):
        # 获取没有偏置项的卷积-BN模型，并验证对称的XNNPack量化训练数字特性
        m1 = self._get_conv_bn_model(has_conv_bias=False, has_bn=False, has_relu=True)
        # 获取没有偏置项且没有ReLU的卷积-BN模型，并验证对称的XNNPack量化训练数字特性
        m2 = self._get_conv_bn_model(has_conv_bias=False, has_bn=False, has_relu=False)
        self._verify_symmetric_xnnpack_qat_numerics(m1, self.example_inputs)
        self._verify_symmetric_xnnpack_qat_numerics(m2, self.example_inputs)

    def test_qat_conv_bn_fusion(self):
        # 获取卷积-BN融合模型，并验证其图结构在没有ReLU下的对称XNNPack量化训练特性
        m = self._get_conv_bn_model()
        self._verify_symmetric_xnnpack_qat_graph(m, self.example_inputs, has_relu=False)
        self._verify_symmetric_xnnpack_qat_numerics(m, self.example_inputs)

    @unittest.skipIf(not TEST_CUDA, "CUDA unavailable")
    def test_qat_conv_bn_fusion_cuda(self):
        # 获取卷积-BN融合模型，并验证其图结构在没有ReLU下的对称XNNPack量化训练特性（CUDA版本）
        m = self._get_conv_bn_model().cuda()
        example_inputs = (self.example_inputs[0].cuda(),)
        self._verify_symmetric_xnnpack_qat_graph(
            m,
            example_inputs,
            has_relu=False,
            is_cuda=True,
        )
        self._verify_symmetric_xnnpack_qat_numerics(m, example_inputs)

    def test_qat_conv_bn_fusion_literal_args(self):
        class M(torch.nn.Module):
            def __init__(self, conv_class, bn_class):
                super().__init__()
                # 初始化包含给定参数的卷积和BN层
                self.conv = conv_class(3, 3, 3, stride=2, padding=4)
                self.bn = bn_class(3)

            def forward(self, x):
                x = self.conv(x)
                x = self.bn(x)
                return x

        assert self.dim in [1, 2]
        if self.dim == 1:
            # stride, padding, dilation, transposed, output_padding, groups
            conv_args = ((2,), (4,), (1,), False, (0,), 1)
            example_inputs = (torch.randn(1, 3, 5),)
        else:
            # stride, padding, dilation, transposed, output_padding, groups
            conv_args = ((2, 2), (4, 4), (1, 1), False, (0, 0), 1)
            example_inputs = (torch.randn(1, 3, 5, 5),)

        m = M(self.conv_class, self.bn_class)

        self._verify_symmetric_xnnpack_qat_graph(
            m,
            example_inputs,
            has_relu=False,
            expected_conv_literal_args=conv_args,
        )
        self._verify_symmetric_xnnpack_qat_numerics(m, example_inputs)
    # 定义一个测试用例函数，用于测试没有卷积偏置的量化训练（QAT）中的卷积+批归一化+ReLU的融合情况
    def test_qat_conv_bn_fusion_no_conv_bias(self):
        # 定义一个内部类M2，表示混合的卷积+批归一化，其中包含有/无卷积偏置的情况
        class M2(torch.nn.Module):
            """
            Mixed conv + BN with and without conv bias.
            """

            def __init__(self, conv_class, bn_class):
                super().__init__()
                # 第一个卷积层，无卷积偏置
                self.conv1 = conv_class(3, 3, 3, bias=False)
                # 第一个批归一化层
                self.bn1 = bn_class(3)
                # 第二个卷积层，有卷积偏置
                self.conv2 = conv_class(3, 3, 3, bias=True)
                # 第二个批归一化层
                self.bn2 = bn_class(3)

            def forward(self, x):
                # 执行第一个卷积操作
                x = self.conv1(x)
                # 执行第一个批归一化操作
                x = self.bn1(x)
                # 执行第二个卷积操作
                x = self.conv2(x)
                # 执行第二个批归一化操作
                x = self.bn2(x)
                return x

        # 通过调用_get_conv_bn_model方法获取一个没有卷积偏置的模型m1
        m1 = self._get_conv_bn_model(has_conv_bias=False)
        # 创建一个M2类的实例m2，传入卷积类和批归一化类作为参数
        m2 = M2(self.conv_class, self.bn_class)

        # 断言维度dim在[1, 2]之中
        assert self.dim in [1, 2]
        # 如果维度是1，则使用随机生成的张量作为示例输入
        if self.dim == 1:
            example_inputs = (torch.randn(3, 3, 5),)
        # 如果维度是2，则使用随机生成的2D张量作为示例输入
        else:
            example_inputs = (torch.randn(3, 3, 5, 5),)

        # 验证在QAT图中的对称XNNPACK图形，确保没有ReLU和偏置
        self._verify_symmetric_xnnpack_qat_graph(
            m1,
            example_inputs,
            has_relu=False,
            has_bias=False,
        )
        # 验证在QAT数值上的对称XNNPACK数值，使用模型m1和示例输入
        self._verify_symmetric_xnnpack_qat_numerics(m1, example_inputs)
        # 验证在QAT数值上的对称XNNPACK数值，使用模型m2和示例输入
        self._verify_symmetric_xnnpack_qat_numerics(m2, example_inputs)

    # 定义一个测试用例函数，用于测试卷积+批归一化+ReLU的融合情况
    def test_qat_conv_bn_relu_fusion(self):
        # 通过调用_get_conv_bn_model方法获取一个具有ReLU的卷积+批归一化模型m
        m = self._get_conv_bn_model(has_relu=True)
        # 验证在QAT图中的对称XNNPACK图形，确保有ReLU
        self._verify_symmetric_xnnpack_qat_graph(m, self.example_inputs, has_relu=True)
        # 验证在QAT数值上的对称XNNPACK数值，使用模型m和示例输入
        self._verify_symmetric_xnnpack_qat_numerics(m, self.example_inputs)

    # 如果CUDA可用，则定义一个测试用例函数，用于测试在CUDA上的卷积+批归一化+ReLU的融合情况
    @unittest.skipIf(not TEST_CUDA, "CUDA unavailable")
    def test_qat_conv_bn_relu_fusion_cuda(self):
        # 通过调用_get_conv_bn_model方法获取一个具有ReLU的卷积+批归一化模型m，并将其移动到CUDA上
        m = self._get_conv_bn_model(has_relu=True).cuda()
        # 将示例输入的第一个张量也移动到CUDA上
        example_inputs = (self.example_inputs[0].cuda(),)
        # 验证在QAT图中的对称XNNPACK图形，确保有ReLU和CUDA加速
        self._verify_symmetric_xnnpack_qat_graph(
            m,
            example_inputs,
            has_relu=True,
            is_cuda=True,
        )
        # 验证在QAT数值上的对称XNNPACK数值，使用模型m和CUDA示例输入
        self._verify_symmetric_xnnpack_qat_numerics(m, example_inputs)

    # 定义一个测试用例函数，用于测试没有卷积偏置的卷积+批归一化+ReLU的融合情况
    def test_qat_conv_bn_relu_fusion_no_conv_bias(self):
        # 通过调用_get_conv_bn_model方法获取一个没有卷积偏置的、具有ReLU的卷积+批归一化模型m
        m = self._get_conv_bn_model(has_conv_bias=False, has_relu=True)
        # 验证在QAT图中的对称XNNPACK图形，确保有ReLU和没有偏置
        self._verify_symmetric_xnnpack_qat_graph(
            m,
            self.example_inputs,
            has_relu=True,
            has_bias=False,
        )
        # 验证在QAT数值上的对称XNNPACK数值，使用模型m和示例输入
        self._verify_symmetric_xnnpack_qat_numerics(m, self.example_inputs)
    def test_qat_inplace_add_relu(self):
        # 定义一个测试函数，用于测试 QAT（量化感知训练）中的 inplace 加法和 ReLU 激活操作
        class M(torch.nn.Module):
            def __init__(self, conv_class):
                super().__init__()
                # 初始化一个卷积层和一个 inplace 为 True 的 ReLU 激活函数
                self.conv = conv_class(1, 1, 1)
                self.relu = torch.nn.ReLU(inplace=True)

            def forward(self, x):
                # 将输入 x 复制给 x0
                x0 = x
                # 执行卷积操作
                x = self.conv(x)
                # 将卷积结果与 x0 原地相加
                x += x0
                # 对相加后的结果应用 ReLU 激活函数
                x = self.relu(x)
                return x

        # 断言 self.dim 的值为 1 或 2
        assert self.dim in [1, 2]
        if self.dim == 1:
            # 如果 self.dim 为 1，使用随机生成的 1 维输入数据作为示例输入
            example_inputs = (torch.randn(1, 1, 3),)
        else:
            # 如果 self.dim 不为 1，使用随机生成的 2 维输入数据作为示例输入
            example_inputs = (torch.randn(1, 1, 3, 3),)

        # 创建 M 类的实例 m，并传入卷积类 self.conv_class 作为参数
        m = M(self.conv_class)
        # 使用 _verify_symmetric_xnnpack_qat_numerics 方法验证对称的 xnnpack QAT 数值特性
        self._verify_symmetric_xnnpack_qat_numerics(m, example_inputs)

    def test_qat_update_shared_qspec(self):
        """
        Test the case where nodes used in SharedQuantizationSpec were replaced
        during QAT subgraph rewriting.
        """

        # 定义一个测试函数，用于测试 SharedQuantizationSpec 中节点在 QAT 子图重写期间的更新情况
        class M(torch.nn.Module):
            def __init__(self, conv_class, bn_class):
                super().__init__()
                # 初始化一个包含 3 个输入通道、3 个输出通道的卷积层和一个批归一化层
                self.conv = conv_class(3, 3, 3)
                self.bn = bn_class(3)
                # 初始化一个 Hardtanh 激活函数
                self.hardtanh = torch.nn.Hardtanh()

            def forward(self, x):
                # 执行卷积操作
                x = self.conv(x)
                # 执行批归一化操作
                x = self.bn(x)
                # 对批归一化后的结果应用 Hardtanh 激活函数
                x = self.hardtanh(x)
                return x

        # 创建 M 类的实例 m，并传入卷积类 self.conv_class 和批归一化类 self.bn_class 作为参数
        m = M(self.conv_class, self.bn_class)
        # 使用 _verify_symmetric_xnnpack_qat_numerics 方法验证对称的 xnnpack QAT 数值特性，使用 self.example_inputs 作为输入
        self._verify_symmetric_xnnpack_qat_numerics(m, self.example_inputs)
    # 定义测试方法：测试量化感知训练（QAT）中卷积层和批归一化（BN）的偏置（bias）推导的量化规格（qspec）
    def test_qat_conv_bn_bias_derived_qspec(self):
        # 获取包含卷积层和批归一化的模型
        m = self._get_conv_bn_model()
        # 获取示例输入
        example_inputs = self.example_inputs
        # 捕获前自动微分图
        m = capture_pre_autograd_graph(m, example_inputs)
        # 创建卷积层和BN推导偏置的量化器
        quantizer = ConvBnDerivedBiasQuantizer()
        # 准备QAT（量化感知训练）后端
        m = prepare_qat_pt2e(m, quantizer)
        # 执行模型并传入示例输入
        m(*example_inputs)
        # 将PyTorch模型转换为Triton端（pt2e）
        m = convert_pt2e(m)
        # 再次执行模型并传入示例输入
        m(*example_inputs)

        # 断言权重和偏置均已量化
        (conv_node, _, _) = _get_conv_bn_getitem_nodes(m)
        weight_dq = conv_node.args[1]
        bias_dq = conv_node.args[2]
        self.assertEqual(
            weight_dq.target,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default,
        )
        self.assertEqual(
            bias_dq.target,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default,
        )
        weight_getattr = weight_dq.args[0]
        bias_getattr = bias_dq.args[0]
        # 断言权重和偏置的操作类型为"get_attr"
        self.assertEqual(
            weight_getattr.op,
            "get_attr",
        )
        self.assertEqual(
            bias_getattr.op,
            "get_attr",
        )

        # 断言偏置的量化尺度等于权重尺度乘以输入尺度
        input_dq = conv_node.args[0]
        input_scale = input_dq.args[1]
        bias_scale = bias_dq.args[1]
        weight_scale = weight_dq.args[1]
        self.assertEqual(bias_scale, input_scale * weight_scale)

        # 断言偏置的量化和反量化操作的参数在子图重写后正确复制
        (bias_qmin, bias_qmax, bias_dtype) = bias_dq.args[3:]
        self.assertEqual(bias_qmin, -(2**31))
        self.assertEqual(bias_qmax, 2**31 - 1)
        self.assertEqual(bias_dtype, torch.int32)

    # 定义测试方法：测试QAT中卷积权重自定义数据类型的量化
    def test_qat_per_channel_weight_custom_dtype(self):
        # 获取包含卷积层和批归一化的模型
        m = self._get_conv_bn_model()
        # 获取示例输入
        example_inputs = self.example_inputs
        # 捕获前自动微分图
        m = capture_pre_autograd_graph(m, example_inputs)
        # 创建卷积层整数32位权重量化器
        quantizer = ConvBnInt32WeightQuantizer()
        # 准备QAT后端
        m = prepare_qat_pt2e(m, quantizer)
        # 执行模型并传入示例输入
        m(*example_inputs)
        # 将PyTorch模型转换为Triton端
        m = convert_pt2e(m)
        # 再次执行模型并传入示例输入
        m(*example_inputs)

        # 断言卷积层权重是按通道量化的
        (conv_node, _, _) = _get_conv_bn_getitem_nodes(m)
        weight_dq = conv_node.args[1]
        self.assertEqual(
            weight_dq.target,
            torch.ops.quantized_decomposed.dequantize_per_channel.default,
        )
        weight_getattr = weight_dq.args[0]
        # 断言权重的操作类型为"get_attr"
        self.assertEqual(
            weight_getattr.op,
            "get_attr",
        )

        # 断言权重反量化操作的参数在子图重写后正确复制
        (dq_axis, dq_qmin, dq_qmax, dq_dtype) = weight_dq.args[3:]
        self.assertEqual(dq_axis, 0)
        self.assertEqual(dq_qmin, 0)
        self.assertEqual(dq_qmax, 2**31 - 1)
        self.assertEqual(dq_dtype, torch.int32)
    # 定义一个测试方法，用于测试带有反卷积和批量归一化的量化感知训练（QAT）模型
    def _do_test_qat_conv_transpose_bn(self, has_relu: bool):
        # 使用不同的输入/输出通道大小来测试卷积权重在QAT模式下是否正确反转
        m = self._get_conv_bn_model(
            has_relu=has_relu,
            transpose=True,
            in_channels=3,
            out_channels=5,
            kernel_size=3,
        )
        # 验证对称的XNNPACK QAT图
        self._verify_symmetric_xnnpack_qat_graph(
            m,
            self.example_inputs,
            has_relu=has_relu,
            verify_convert=True,
        )

    # 测试不带ReLU的卷积反卷积和批量归一化的QAT模型
    def test_qat_conv_transpose_bn(self):
        self._do_test_qat_conv_transpose_bn(has_relu=False)

    # 测试带ReLU的卷积反卷积和批量归一化的QAT模型
    def test_qat_conv_transpose_bn_relu(self):
        self._do_test_qat_conv_transpose_bn(has_relu=True)

    # 测试带通道权重和偏置项的卷积批量归一化的QAT模型
    def test_qat_conv_bn_per_channel_weight_bias(self):
        # 获取一个卷积批量归一化模型
        m = self._get_conv_bn_model()
        example_inputs = self.example_inputs
        # 捕获前自动求导图
        m = capture_pre_autograd_graph(m, example_inputs)
        # 创建一个通道权重量化器
        quantizer = ConvBnDerivedBiasQuantizer(is_per_channel=True)
        # 准备QAT，阶段2
        m = prepare_qat_pt2e(m, quantizer)
        # 在示例输入上运行模型
        m(*example_inputs)
        # 将模型转换为量化感知训练（QAT）模式
        m = convert_pt2e(m)
        # 再次在示例输入上运行模型
        m(*example_inputs)

        # 期望的计算图：
        #      x -> q_tensor -> dq_tensor -> conv -> q_tensor -> dq_tensor -> output
        #  weight -> q_channel -> dq_channel /
        #    bias -> q_channel -> dq_channel /

        # 获取卷积批量归一化模型中的节点和操作
        (conv_node, _, _) = _get_conv_bn_getitem_nodes(m)
        conv_op = conv_node.target
        conv_weight_dq_op = (
            torch.ops.quantized_decomposed.dequantize_per_channel.default
        )
        # 定义节点出现次数的期望
        node_occurrence = {
            ns.call_function(
                torch.ops.quantized_decomposed.quantize_per_tensor.default
            ): 2,
            ns.call_function(
                torch.ops.quantized_decomposed.dequantize_per_tensor.default
            ): 2,
            ns.call_function(
                torch.ops.quantized_decomposed.dequantize_per_channel.default
            ): 2,
        }
        # 定义节点列表
        node_list = [
            ns.call_function(
                torch.ops.quantized_decomposed.quantize_per_tensor.default
            ),
            ns.call_function(
                torch.ops.quantized_decomposed.dequantize_per_tensor.default
            ),
            ns.call_function(conv_weight_dq_op),
            ns.call_function(conv_weight_dq_op),
            ns.call_function(conv_op),
            ns.call_function(
                torch.ops.quantized_decomposed.quantize_per_tensor.default
            ),
            ns.call_function(
                torch.ops.quantized_decomposed.dequantize_per_tensor.default
            ),
        ]
        # 检查图模块中的节点是否符合预期
        self.checkGraphModuleNodes(
            m,
            expected_node_list=node_list,
            expected_node_occurrence=node_occurrence,
        )
@skipIfNoQNNPACK
# 如果没有 QNNPACK，跳过测试用例
class TestQuantizePT2EQAT_ConvBn1d(TestQuantizePT2EQAT_ConvBn_Base):
    # 继承基类 TestQuantizePT2EQAT_ConvBn_Base，并指定维度为 1
    dim = 1
    # 示例输入为一个形状为 (1, 3, 5) 的随机张量
    example_inputs = (torch.randn(1, 3, 5),)
    # 使用的卷积层类为 torch.nn.Conv1d
    conv_class = torch.nn.Conv1d
    # 使用的转置卷积层类为 torch.nn.ConvTranspose1d
    conv_transpose_class = torch.nn.ConvTranspose1d
    # 使用的批归一化层类为 torch.nn.BatchNorm1d


@skipIfNoQNNPACK
# 如果没有 QNNPACK，跳过测试用例
class TestQuantizePT2EQAT_ConvBn2d(TestQuantizePT2EQAT_ConvBn_Base):
    # 继承基类 TestQuantizePT2EQAT_ConvBn_Base，并指定维度为 2
    dim = 2
    # 示例输入为一个形状为 (1, 3, 5, 5) 的随机张量
    example_inputs = (torch.randn(1, 3, 5, 5),)
    # 使用的卷积层类为 torch.nn.Conv2d
    conv_class = torch.nn.Conv2d
    # 使用的转置卷积层类为 torch.nn.ConvTranspose2d
    conv_transpose_class = torch.nn.ConvTranspose2d
    # 使用的批归一化层类为 torch.nn.BatchNorm2d


def _is_conv_node(n: torch.fx.Node):
    # 判断给定的节点是否为调用卷积函数的节点
    return n.op == "call_function" and n.target in [
        torch.ops.aten.conv1d.default,
        torch.ops.aten.conv2d.default,
        torch.ops.aten.conv_transpose1d,
        torch.ops.aten.conv_transpose1d.default,
        torch.ops.aten.conv_transpose2d,
        torch.ops.aten.conv_transpose2d.input,
    ]


def _get_conv_bn_getitem_nodes(model: torch.fx.GraphModule):
    """
    Return a 3-tuple of (conv, bn, getitem) nodes from the graph.
    返回图中卷积、批归一化和索引操作节点的三元组。
    """
    model.graph.eliminate_dead_code()
    model.recompile()
    conv_node = None
    bn_node = None
    getitem_node = None
    for n in model.graph.nodes:
        if _is_conv_node(n):
            conv_node = n
        if n.target == torch.ops.aten._native_batch_norm_legit.default:
            bn_node = n
        if n.target == operator.getitem:
            getitem_node = n
    assert conv_node is not None, "bad test setup"
    return (conv_node, bn_node, getitem_node)


class ConvBnInt32WeightQuantizer(Quantizer):
    """
    Dummy quantizer that annotates conv bn in such a way that the weights
    are quantized per channel to int32.
    伪量化器，将卷积和批归一化的权重按通道量化为 int32。
    """

    def annotate(self, model: torch.fx.GraphModule) -> torch.fx.GraphModule:
        conv_node, _, getitem_node = _get_conv_bn_getitem_nodes(model)
        # 定义激活量化规范，按照 uint8 类型量化，量化范围为 0 到 255，使用每张张量的仿射量化方案
        act_qspec = QuantizationSpec(
            dtype=torch.uint8,
            quant_min=0,
            quant_max=255,
            qscheme=torch.per_tensor_affine,
            observer_or_fake_quant_ctr=default_fake_quant,
        )
        # 定义权重量化规范，按照 int32 类型量化，量化范围为 0 到 2^31-1，使用每通道的仿射量化方案
        weight_qspec = QuantizationSpec(
            dtype=torch.int32,
            quant_min=0,
            quant_max=2**31 - 1,
            qscheme=torch.per_channel_affine,
            observer_or_fake_quant_ctr=FusedMovingAvgObsFakeQuantize.with_args(
                observer=MovingAveragePerChannelMinMaxObserver,
            ),
        )
        # 将卷积节点的量化注释设置为定义好的量化规范
        conv_node.meta["quantization_annotation"] = QuantizationAnnotation(
            input_qspec_map={
                conv_node.args[0]: act_qspec,
                conv_node.args[1]: weight_qspec,
            },
            _annotated=True,
        )
        # 将索引操作节点的量化注释设置为激活量化规范
        getitem_node.meta["quantization_annotation"] = QuantizationAnnotation(
            output_qspec=act_qspec,
            _annotated=True,
        )
        return model

    def validate(self, model: torch.fx.GraphModule):
        # 不执行验证操作，留空
        pass


class ConvBnDerivedBiasQuantizer(Quantizer):
    """
    """
    """
    Dummy quantizer that annotates conv bn in such a way that the bias qparams are
    derived from the conv input activation and weight qparams.
    """

    # 初始化函数，设置是否按通道量化
    def __init__(self, is_per_channel: bool = False):
        super().__init__()
        self.is_per_channel = is_per_channel

    # 根据输入的观察值或伪量化参数计算出偏置量化参数
    def _derive_bias_qparams_from_act_and_weight_qparams(self, obs_or_fqs):
        # 获取输入激活的量化参数，act_scale 是量化比例，_ 是量化零点
        act_scale, _ = obs_or_fqs[0].calculate_qparams()
        # 获取权重的量化参数，weight_scale 是量化比例，_ 是量化零点
        weight_scale, _ = obs_or_fqs[1].calculate_qparams()
        
        # 根据是否按通道量化设置偏置的量化比例和零点
        if self.is_per_channel:
            # 按通道量化时，偏置的量化比例是激活量化比例和权重量化比例的乘积
            bias_scale = act_scale * weight_scale
            # 偏置的量化零点初始化为与量化比例相同维度的零张量
            bias_zero_point = torch.zeros_like(bias_scale, dtype=torch.int32)
        else:
            # 非按通道量化时，偏置的量化比例是激活量化比例和权重量化比例的乘积
            bias_scale = torch.tensor([act_scale * weight_scale], dtype=torch.float32)
            # 偏置的量化零点初始化为零
            bias_zero_point = torch.tensor([0], dtype=torch.int32)
        
        return bias_scale, bias_zero_point

    # 为模型添加量化注释
    def annotate(self, model: torch.fx.GraphModule) -> torch.fx.GraphModule:
        # 根据是否按通道量化选择权重的量化方案和伪量化方法
        if self.is_per_channel:
            weight_qscheme = torch.per_channel_symmetric
            weight_fq = FusedMovingAvgObsFakeQuantize.with_args(
                observer=MovingAveragePerChannelMinMaxObserver,
            )
        else:
            weight_qscheme = torch.per_tensor_affine
            weight_fq = default_fake_quant
        
        # 获取卷积、批归一化和获取项的节点
        conv_node, _, getitem_node = _get_conv_bn_getitem_nodes(model)
        
        # 激活量化规范，设置量化类型、量化范围和量化方案
        act_qspec = QuantizationSpec(
            dtype=torch.uint8,
            quant_min=0,
            quant_max=255,
            qscheme=torch.per_tensor_affine,
            observer_or_fake_quant_ctr=default_fake_quant,
        )
        
        # 权重量化规范，设置量化类型、量化范围、量化方案和伪量化方法
        weight_qspec = QuantizationSpec(
            dtype=torch.uint8,
            quant_min=0,
            quant_max=255,
            qscheme=weight_qscheme,
            observer_or_fake_quant_ctr=weight_fq,
        )
        
        # 偏置量化规范，设置量化类型、量化范围、量化方案、从哪些节点派生、派生参数计算函数、数据类型和通道轴（按通道量化时）
        bias_qspec = DerivedQuantizationSpec(
            derived_from=[
                (conv_node.args[0], conv_node),
                (conv_node.args[1], conv_node),
            ],
            derive_qparams_fn=self._derive_bias_qparams_from_act_and_weight_qparams,
            dtype=torch.int32,
            quant_min=-(2**31),
            quant_max=2**31 - 1,
            qscheme=weight_qscheme,
            ch_axis=0 if self.is_per_channel else None,
        )
        
        # 为卷积节点和获取项节点添加量化注释
        conv_node.meta["quantization_annotation"] = QuantizationAnnotation(
            input_qspec_map={
                conv_node.args[0]: act_qspec,
                conv_node.args[1]: weight_qspec,
                conv_node.args[2]: bias_qspec,
            },
            _annotated=True,
        )
        getitem_node.meta["quantization_annotation"] = QuantizationAnnotation(
            output_qspec=act_qspec,
            _annotated=True,
        )
        
        return model

    # 验证函数，暂未实现具体功能
    def validate(self, model: torch.fx.GraphModule):
        pass
# 如果没有 QNNPACK 模块，则跳过测试类
@skipIfNoQNNPACK
class TestQuantizePT2EQATModels(PT2EQATTestCase):
    # 如果没有 torchvision 模块，则跳过测试方法
    @skip_if_no_torchvision
    # 如果没有 QNNPACK 模块，则跳过测试方法
    @skipIfNoQNNPACK
    def test_qat_resnet18(self):
        import torchvision

        # 使用 qnnpack 引擎来覆盖量化引擎
        with override_quantized_engine("qnnpack"):
            example_inputs = (torch.randn(1, 3, 224, 224),)
            # 创建 ResNet-18 模型
            m = torchvision.models.resnet18()
            # 验证 QAT 数值的对称性使用 xnnpack
            self._verify_symmetric_xnnpack_qat_numerics(m, example_inputs)

    # 如果没有 torchvision 模块，则跳过测试方法
    @skip_if_no_torchvision
    # 如果没有 QNNPACK 模块，则跳过测试方法
    @skipIfNoQNNPACK
    def test_qat_mobilenet_v2(self):
        import torchvision

        # 使用 qnnpack 引擎来覆盖量化引擎
        with override_quantized_engine("qnnpack"):
            example_inputs = (torch.randn(1, 3, 224, 224),)
            # 创建 MobileNet V2 模型
            m = torchvision.models.mobilenet_v2()
            # 验证 QAT 数值的对称性使用 xnnpack
            self._verify_symmetric_xnnpack_qat_numerics(m, example_inputs)


class TestQuantizeMixQATAndPTQ(QuantizationTestCase):
    # 定义一个包含两个线性层的简单模型
    class TwoLinear(torch.nn.Module):
        def __init__(self):
            super().__init__()
            # 第一个线性层，无偏置
            self.linear1 = torch.nn.Linear(16, 8, bias=False)
            # 第二个线性层，有偏置
            self.linear2 = torch.nn.Linear(8, 8)

        def forward(self, x):
            return self.linear2(self.linear1(x))

    # 定义一个包含量化和非量化模块的测试模型
    class QATPTQTestModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            # 卷积层，输入通道 3，输出通道 16，卷积核大小 3x3
            self.conv = torch.nn.Conv2d(3, 16, 3)
            # 包含 TwoLinear 类定义的线性层
            self.linears = TestQuantizeMixQATAndPTQ.TwoLinear()
            # 自定义线性层，输入输出都是 8 维
            self.my_linear = torch.nn.Linear(8, 8)

        def forward(self, x):
            # 卷积计算
            conv_out = self.conv(x)
            # 维度置换
            permute_out = torch.permute(conv_out, (0, 2, 3, 1))
            # 使用包含的线性层进行计算
            linear_out = self.linears(permute_out)
            # 使用自定义线性层进行计算，并应用 hardtanh 函数
            # 在这个测试中，hardtanh 函数不会通过 xnnpack 量化器进行量化
            # 因为它依赖于传播规则，需要修复这个问题
            return torch.nn.functional.hardtanh(my_linear_out)

    # 准备 QAT 线性层
    def _prepare_qat_linears(self, model):
        for name, child in model.named_children():
            if isinstance(child, (torch.nn.Linear, TestQuantizeMixQATAndPTQ.TwoLinear)):
                if isinstance(child, torch.nn.Linear):
                    in_channels = child.weight.size(1)
                else:
                    in_channels = child.linear1.weight.size(1)

                example_input = (torch.rand((1, in_channels)),)
                # 捕获子模块的预自动梯度图
                traced_child = capture_pre_autograd_graph(child, example_input)
                # 创建 XNNPACK 量化器
                quantizer = XNNPACKQuantizer()
                # 获取对称量化配置，每通道独立，QAT 模式
                quantization_config = get_symmetric_quantization_config(
                    is_per_channel=True, is_qat=True
                )
                # 设置全局量化配置
                quantizer.set_global(quantization_config)
                # 准备 QAT-PT2E
                traced_child_prepared = prepare_qat_pt2e(traced_child, quantizer)
                # 设置模型的子模块为准备好的 QAT-PT2E 模块
                setattr(model, name, traced_child_prepared)
            else:
                # 递归处理其他类型的子模块
                self._prepare_qat_linears(child)
    # 将模型中的子模块递归地转换为评估模式，如果是 `GraphModule` 则移动到评估模式
    def _convert_qat_linears(self, model):
        for name, child in model.named_children():
            if isinstance(child, torch.fx.GraphModule):
                torch.ao.quantization.move_exported_model_to_eval(child)
                # 将子模块转换为评估模式后，再进行从 PyTorch 到 ESIM 的转换
                converted_child = convert_pt2e(child)
                setattr(model, name, converted_child)
            else:
                self._convert_qat_linears(child)

    # 测试混合量化训练和量化训练的效果
    def test_mixing_qat_ptq(self):
        example_inputs = (torch.randn(2, 3, 4, 4),)
        # 创建混合量化训练和量化训练测试模块
        model = TestQuantizeMixQATAndPTQ.QATPTQTestModule()

        # 准备模型中的量化感知训练线性层
        self._prepare_qat_linears(model)

        # 使用例子输入进行模型评估，并记录评估后的结果
        after_prepare_result_pt2e = model(*example_inputs)

        # 将模型中的量化感知训练线性层转换为评估模式线性层
        self._convert_qat_linears(model)

        # 使用例子输入进行量化评估，并记录量化评估后的结果
        quant_result_pt2e = model(*example_inputs)

        # 捕获模型的预自动梯度图
        model_pt2e = capture_pre_autograd_graph(
            model,
            example_inputs,
        )

        # 创建 XNNPACK 量化器实例
        quantizer = XNNPACKQuantizer()
        # 设置量化器的模块类型为线性层，并清空之前的设置
        quantizer.set_module_type(torch.nn.Linear, None)

        # 获取对称量化配置
        quantization_config = get_symmetric_quantization_config()
        # 在量化器中全局设置量化配置
        quantizer.set_global(quantization_config)

        # 准备 ESIM 模型，使用量化器进行量化
        model_pt2e = prepare_pt2e(model_pt2e, quantizer)

        # 使用例子输入进行 ESIM 模型评估，并记录评估后的结果
        after_prepare_result_pt2e = model_pt2e(*example_inputs)

        # 将 ESIM 模型从 PyTorch 转换为 ESIM
        model_pt2e = convert_pt2e(model_pt2e)

        # 使用例子输入进行 ESIM 模型的量化评估，并记录量化评估后的结果
        quant_result_pt2e = model_pt2e(*example_inputs)

        # 导出模型为 TorchScript，以便部署和后续优化
        exported_model = torch.export.export(model_pt2e, example_inputs)

        # 期望每个节点出现的次数
        node_occurrence = {
            # conv2d: 1 for act, 1 for weight, 1 for output
            # 3 x linear: 1 for act, 1 for output
            ns.call_function(
                torch.ops.quantized_decomposed.quantize_per_tensor.default
            ): 8,
            ns.call_function(
                torch.ops.quantized_decomposed.dequantize_per_tensor.default
            ): 9,
            ns.call_function(
                torch.ops.quantized_decomposed.dequantize_per_channel.default
            ): 3,
            # 需要一个用于 hardtanh
        }

        # 检查导出模型的图模块节点，确保节点出现次数符合预期
        self.checkGraphModuleNodes(
            exported_model.graph_module, expected_node_occurrence=node_occurrence
        )
```