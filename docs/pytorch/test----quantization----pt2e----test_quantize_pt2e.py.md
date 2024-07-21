# `.\pytorch\test\quantization\pt2e\test_quantize_pt2e.py`

```py
# 引入所需模块和类
from typing import List, Tuple  # 引入类型提示 List 和 Tuple

import torch  # 引入 PyTorch 库
from torch import Tensor  # 引入 Tensor 类型
from torch._export import capture_pre_autograd_graph  # 引入捕获前自动求导图的函数
from torch.ao.quantization import observer, ObserverOrFakeQuantize, QConfigMapping  # 引入量化相关模块

from torch.ao.quantization.qconfig import (  # 引入量化配置
    default_per_channel_symmetric_qnnpack_qconfig,
    float_qparams_weight_only_qconfig,
    per_channel_weight_observer_range_neg_127_to_127,
    QConfig,
    weight_observer_range_neg_127_to_127,
)

from torch.ao.quantization.quantize_pt2e import (  # 引入量化的转换和准备模块
    convert_pt2e,
    prepare_pt2e,
    prepare_qat_pt2e,
)
from torch.ao.quantization.quantizer import (  # 引入量化器相关类
    DerivedQuantizationSpec,
    FixedQParamsQuantizationSpec,
    QuantizationAnnotation,
    QuantizationSpec,
    Quantizer,
    SharedQuantizationSpec,
)
from torch.ao.quantization.quantizer.composable_quantizer import (  # 引入可组合量化器
    ComposableQuantizer,
)
from torch.ao.quantization.quantizer.embedding_quantizer import (  # 引入嵌入量化器
    EmbeddingQuantizer,
)
from torch.ao.quantization.quantizer.xnnpack_quantizer import (  # 引入 XNNPACK 量化器
    get_symmetric_quantization_config,
    XNNPACKQuantizer,
)
from torch.ao.quantization.quantizer.xnnpack_quantizer_utils import (  # 引入 XNNPACK 量化配置工具
    OP_TO_ANNOTATOR,
    QuantizationConfig,
)
from torch.fx import Node  # 引入 Torch FX 模块的节点类

from torch.testing._internal.common_quantization import (  # 引入量化测试相关模块
    NodeSpec as ns,
    PT2EQuantizationTestCase,
    skipIfNoQNNPACK,
    TestHelperModules,
)
from torch.testing._internal.common_utils import (  # 引入通用测试工具函数和类
    instantiate_parametrized_tests,
    parametrize,
    TemporaryFileName,
    TEST_CUDA,
    TEST_WITH_ROCM,
)

@skipIfNoQNNPACK
class TestQuantizePT2E(PT2EQuantizationTestCase):
    # 测试固定量化参数和量化规格的 PTQ
    def test_fixed_qparams_qspec_ptq(self):
        self._test_fixed_qparams_qspec(is_qat=False)

    # TODO: 重构并将此方法移到 test_quantize_pt2_qat.py 中
    # 测试固定量化参数和量化规格的 QAT（量化感知训练）
    def test_fixed_qparams_qspec_qat(self):
        self._test_fixed_qparams_qspec(is_qat=True)

    @parametrize("dtype", (torch.int16, torch.float8_e5m2, torch.float8_e4m3fn))
    def test_quantization_dtype(self, dtype):
        # 定义一个名为 test_quantization_dtype 的测试方法，接受一个 dtype 参数

        class DtypeActQuantizer(Quantizer):
            # 定义一个名为 DtypeActQuantizer 的内部类，继承自 Quantizer 类

            def annotate(self, model: torch.fx.GraphModule) -> torch.fx.GraphModule:
                # 定义 annotate 方法，接受一个 torch.fx.GraphModule 类型的 model 参数，返回一个 torch.fx.GraphModule 类型的对象
                # 根据 dtype 类型选择合适的 info 函数，torch.iinfo 用于整数类型，torch.finfo 用于浮点数类型
                info_fun = torch.iinfo if dtype == torch.int16 else torch.finfo
                
                # 创建激活量化规范对象 activate_qspec
                activate_qspec = QuantizationSpec(
                    dtype=dtype,
                    quant_min=int(info_fun(dtype).min),
                    quant_max=int(info_fun(dtype).max),
                    qscheme=torch.per_tensor_affine,
                    is_dynamic=False,
                    observer_or_fake_quant_ctr=observer.default_observer,
                )
                
                # 创建 int8 类型的量化规范对象 int8_qspec
                int8_qspec = QuantizationSpec(
                    dtype=torch.int8,
                    quant_min=-128,
                    quant_max=127,
                    qscheme=torch.per_tensor_symmetric,
                    is_dynamic=False,
                    observer_or_fake_quant_ctr=observer.default_weight_observer,
                )
                
                # 创建量化配置对象 quantization_config，指定输入激活、权重、偏置和输出激活的量化规范
                quantization_config = QuantizationConfig(
                    input_activation=activate_qspec,
                    weight=int8_qspec,
                    bias=None,
                    output_activation=activate_qspec,
                )
                
                # 调用 OP_TO_ANNOTATOR 中的 "conv" 对应的函数，对模型进行量化配置
                OP_TO_ANNOTATOR["conv"](model, quantization_config)

            def validate(self, model: torch.fx.GraphModule) -> None:
                # 定义 validate 方法，接受一个 torch.fx.GraphModule 类型的 model 参数，无返回值，用于验证模型
                pass

        class M(torch.nn.Module):
            # 定义一个名为 M 的 torch.nn.Module 类

            def __init__(self):
                # 初始化方法，定义了一个卷积层 self.conv，输入通道 3，输出通道 3，卷积核大小 3x3
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 3, 3)

            def forward(self, x):
                # 前向传播方法，将输入 x 经过 self.conv 卷积层处理后返回
                return self.conv(x)

        quantizer = DtypeActQuantizer()
        # 创建 DtypeActQuantizer 类的实例 quantizer

        node_occurrence = {
            # 定义一个字典 node_occurrence，用于记录不同操作节点的出现次数
            torch.ops.quantized_decomposed.quantize_per_tensor.default: 2,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default: 3,
        }

        node_list = [
            # 定义一个列表 node_list，包含了一系列操作节点的默认操作符
            torch.ops.quantized_decomposed.dequantize_per_tensor.default,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default,
            torch.ops.aten.conv2d.default,
            torch.ops.quantized_decomposed.quantize_per_tensor.default,
        ]

        example_inputs = (torch.randn(1, 3, 3, 3),)
        # 定义一个示例输入 example_inputs，包含一个 shape 为 (1, 3, 3, 3) 的张量

        self._test_quantizer(
            M().eval(),
            example_inputs,
            quantizer,
            node_occurrence,
            node_list,
        )
        # 调用类内部的 _test_quantizer 方法，传入一个 M 类的实例，示例输入 example_inputs，quantizer 实例，node_occurrence 字典和 node_list 列表作为参数
    def test_input_edge_sanity_check(self):
        """测试输入边界的合理性检查"""

        class M(torch.nn.Module):
            def forward(self, x):
                return x + 6

        class BackendAQuantizer(Quantizer):
            def annotate(self, model: torch.fx.GraphModule) -> torch.fx.GraphModule:
                # 遍历模型图中的每个节点
                for node in model.graph.nodes:
                    # 检查节点是否为函数调用，并且目标函数为 torch.ops.aten.add.Tensor
                    if (
                        node.op == "call_function"
                        and node.target == torch.ops.aten.add.Tensor
                    ):
                        # 获取节点的第一个参数作为输入激活值1
                        input_act1 = node.args[0]
                        # 第二个参数是常数，因此不能用于注释
                        input_act2 = node.args[1]
                        # 定义量化规格
                        act_qspec = QuantizationSpec(
                            dtype=torch.uint8,
                            quant_min=0,
                            quant_max=255,
                            qscheme=torch.per_tensor_affine,
                            is_dynamic=False,
                            observer_or_fake_quant_ctr=observer.default_observer,
                        )
                        # 给节点添加量化注释
                        node.meta["quantization_annotation"] = QuantizationAnnotation(
                            input_qspec_map={
                                input_act1: act_qspec,
                                # 这里预期会出错
                                input_act2: act_qspec,
                            },
                            output_qspec=act_qspec,
                            _annotated=True,
                        )

            def validate(self, model: torch.fx.GraphModule) -> None:
                pass

        m = M().eval()
        example_inputs = torch.randn(1, 2, 3, 3)
        # 捕获前向自动求导图
        m = capture_pre_autograd_graph(m, example_inputs)
        # 断言捕获到异常
        with self.assertRaises(Exception):
            # 准备通过 BackendAQuantizer 进行量化后模型
            m = prepare_pt2e(m, BackendAQuantizer())

    def test_fold_quantize(self):
        """测试确保量化模型得到量化权重（quantize_per_tensor 操作被折叠）"""

        # 获取量化线性模型
        m = self._get_pt2e_quantized_linear()
        # 预期节点出现次数
        node_occurrence = {
            # 折叠权重节点的量化操作
            ns.call_function(
                torch.ops.quantized_decomposed.quantize_per_tensor.default
            ): 2,
            ns.call_function(
                torch.ops.quantized_decomposed.dequantize_per_tensor.default
            ): 3,
        }
        # 检查模型节点是否符合预期出现次数
        self.checkGraphModuleNodes(m, expected_node_occurrence=node_occurrence)
    def test_fold_quantize_per_channel(self):
        """Test to make sure the quantized model gets quantized weight (quantize_per_channel op is folded)"""
        # 获取一个使用通道量化的 PyTorch 2E 量化线性模型
        m = self._get_pt2e_quantized_linear(is_per_channel=True)
        # 定义节点出现次数的预期字典
        node_occurrence = {
            # 权重节点的量化操作被折叠
            ns.call_function(
                torch.ops.quantized_decomposed.quantize_per_tensor.default
            ): 2,
            # 通道反量化操作
            ns.call_function(
                torch.ops.quantized_decomposed.dequantize_per_channel.default
            ): 1,
            # 默认张量反量化操作
            ns.call_function(
                torch.ops.quantized_decomposed.dequantize_per_tensor.default
            ): 2,
        }
        # 检查模型的节点是否符合预期的出现次数
        self.checkGraphModuleNodes(m, expected_node_occurrence=node_occurrence)

    def test_dont_fold_other_constant(self):
        """Make sure the constant propagation does not apply to things unrelated to
        quantization
        """
        # 定义一个包含线性层和不应被折叠的参数的模型类
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(2, 2)
                self.dont_fold_me = torch.nn.Parameter(torch.randn(2, 2))

            def forward(self, x):
                # 不应被折叠的操作，转置操作
                t = self.dont_fold_me.t()
                return self.linear(x) + t

        # 创建 XNNPACK 量化器实例
        quantizer = XNNPACKQuantizer()
        # 获取对称量化配置
        operator_config = get_symmetric_quantization_config(is_per_channel=False)
        # 设置线性层为量化模块类型
        quantizer.set_module_type(torch.nn.Linear, operator_config)
        example_inputs = (torch.randn(2, 2),)
        # 创建并评估模型
        m = M().eval()
        m = self._quantize(m, quantizer, example_inputs)
        # 定义节点出现次数的预期字典
        node_occurrence = {
            # 权重节点的量化操作被折叠
            ns.call_function(
                torch.ops.quantized_decomposed.quantize_per_tensor.default
            ): 2,
            # 默认张量反量化操作
            ns.call_function(
                torch.ops.quantized_decomposed.dequantize_per_tensor.default
            ): 3,
            # 转置操作未被折叠
            ns.call_function(torch.ops.aten.t.default): 1,
        }
        # 检查模型的节点是否符合预期的出现次数
        self.checkGraphModuleNodes(m, expected_node_occurrence=node_occurrence)
    def test_fold_all_ops_before_quantize(self):
        """Test folding all ops that's before quantized operator:
        Before:
            get_attr(weight) -> transpose -> quantize -> dequantize
        After:
            get_attr(folded_weight) -> dequantize
        """

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.randn(2, 2)

            def forward(self, x):
                # 将权重矩阵转置
                t = self.weight.t()
                # 返回线性变换结果
                return torch.nn.functional.linear(x, t)

        # 创建量化器对象
        quantizer = XNNPACKQuantizer()
        # 获取对称量化配置
        operator_config = get_symmetric_quantization_config(is_per_channel=False)
        # 设置全局量化配置
        quantizer.set_global(operator_config)
        # 创建示例输入
        example_inputs = (torch.randn(2, 2),)
        # 实例化模型并设置为评估模式
        m = M().eval()
        # 对模型进行量化
        m = self._quantize(m, quantizer, example_inputs)
        # 定义节点出现次数的预期字典
        node_occurrence = {
            # 权重节点的量化操作已经被折叠
            ns.call_function(
                torch.ops.quantized_decomposed.quantize_per_tensor.default
            ): 2,
            # 权重节点的反量化操作出现了3次
            ns.call_function(
                torch.ops.quantized_decomposed.dequantize_per_tensor.default
            ): 3,
        }
        # 检查图模块节点是否符合预期
        self.checkGraphModuleNodes(m, expected_node_occurrence=node_occurrence)

    def test_constant_prop_preserve_metadata(self):
        """Test to make sure the get_attr node for const propagated weight Tensor gets the correct
        metadata (from original get_attr node from weight)
        """

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(2, 2)

            def forward(self, x):
                # 返回线性变换结果
                return self.linear(x)

        # 创建量化器对象
        quantizer = XNNPACKQuantizer()
        # 获取对称量化配置
        operator_config = get_symmetric_quantization_config()
        # 设置全局量化配置
        quantizer.set_global(operator_config)
        # 创建示例输入
        example_inputs = (torch.randn(2, 2),)
        # 实例化模型并设置为评估模式
        m = M().eval()
        # 捕获前自动求导图
        m = capture_pre_autograd_graph(
            m,
            example_inputs,
        )
        weight_meta = None
        # 遍历图中的节点
        for n in m.graph.nodes:
            # 如果节点操作是获取属性，并且用户指向 torch.ops.aten.linear.default
            if (
                n.op == "get_attr"
                and next(iter(n.users)).target == torch.ops.aten.linear.default
            ):
                # 获取权重节点的元数据
                weight_meta = n.meta
                break
        # 断言：期望找到权重节点的元数据
        assert weight_meta is not None, "Expect to find metadata for weight node"

        # 准备模型转换为 PyTorch 2 后端
        m = prepare_pt2e(m, quantizer)
        # 运行模型
        m(*example_inputs)
        # 将模型转换为 PyTorch 2 后端
        m = convert_pt2e(m)

        # 再次遍历图中的节点
        for n in m.graph.nodes:
            # 如果节点操作是获取属性，并且属性名称中包含 "frozen_param"
            if n.op == "get_attr" and "frozen_param" in n.target:
                # 断言：确保元数据中包含 "stack_trace"
                self.assertIn("stack_trace", n.meta)
                # 遍历元数据中的键
                for key in n.meta:
                    # 断言：确保节点的元数据与权重节点的元数据一致
                    self.assertEqual(n.meta[key], weight_meta[key])
    def test_save_load(self):
        """Test save/load a quantized model"""
        # 获取一个量化线性模型实例
        m = self._get_pt2e_quantized_linear()
        # 准备一个示例输入
        example_inputs = (torch.randn(2, 2),)
        # 获取参考结果
        ref_res = m(*example_inputs)

        with TemporaryFileName() as fname:
            # 序列化
            quantized_ep = torch.export.export(m, example_inputs)
            # 将序列化后的模型保存到文件
            torch.export.save(quantized_ep, fname)
            # 反序列化
            loaded_ep = torch.export.load(fname)
            # 获取加载后的量化模型
            loaded_quantized_model = loaded_ep.module()
            # 对加载的模型进行推理
            res = loaded_quantized_model(*example_inputs)
            # 断言加载后的结果与参考结果相等
            self.assertEqual(ref_res, res)

    def test_composable_quantizer_throw(self):
        class BadQuantizer(Quantizer):
            def annotate(self, gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
                # 遍历图中的节点，为每个节点设置量化注释为 None
                for n in gm.graph.nodes:
                    n.meta["quantization_annotation"] = None

            def validate(self, model: torch.fx.GraphModule) -> None:
                # 不执行验证操作
                pass

        # 初始化 XNNPACK 量化器
        quantizer = XNNPACKQuantizer()
        # 获取对称量化配置
        quantization_config = get_symmetric_quantization_config(is_per_channel=True)
        # 配置全局量化配置
        quantizer.set_global(quantization_config)
        # 初始化不良量化器
        bad_quantizer = BadQuantizer()
        # 初始化可组合量化器
        composable_quantizer = ComposableQuantizer([quantizer, bad_quantizer])
        # 准备一个测试用的卷积线性模块
        m_eager = TestHelperModules.ConvLinearWPermute().eval()
        # 准备示例输入
        example_inputs = (torch.randn(2, 3, 4, 4),)
        # 断言执行量化测试时会抛出 RuntimeError 异常
        self.assertRaises(
            RuntimeError,
            lambda: self._test_quantizer(
                m_eager, example_inputs, composable_quantizer, {}
            ),
        )

    def test_transform_for_annotation(self):
        class TestQuantizer(Quantizer):
            def transform_for_annotation(
                self, model: torch.fx.GraphModule
            ) -> torch.fx.GraphModule:
                # 遍历模型的图节点，将所有的加法操作替换为乘法操作
                for n in model.graph.nodes:
                    if n.target == torch.ops.aten.add.Tensor:
                        n.target = torch.ops.aten.mul.Tensor
                return model

            def annotate(self, model: torch.fx.GraphModule) -> torch.fx.GraphModule:
                # 返回未做任何修改的模型
                return model

            def validate(self, model: torch.fx.GraphModule) -> None:
                # 不执行验证操作
                pass

        # 定义一个简单的模块 M，进行加法操作
        class M(torch.nn.Module):
            def forward(self, x):
                return x + 3

        # 实例化模块 M
        m = M().eval()
        # 准备示例输入
        example_inputs = (torch.randn(1, 2, 3, 3),)
        # 捕获 Autograd 图
        m = capture_pre_autograd_graph(m, example_inputs)
        # 准备 PT2E 模型，应用量化器
        m = prepare_pt2e(m, quantizer)
        # 运行模型推理
        m(*example_inputs)
        # 期望的节点出现次数字典
        node_occurrence = {
            ns.call_function(torch.ops.aten.add.Tensor): 0,
            ns.call_function(torch.ops.aten.mul.Tensor): 1,
        }
        # 检查模型中的节点是否符合预期
        self.checkGraphModuleNodes(m, expected_node_occurrence=node_occurrence)
    def test_composable_quantizer_transform_for_annotation(self):
        # 定义一个测试方法，用于测试可组合的量化器在注释转换方面的功能

        class TestQuantizer1(Quantizer):
            # 定义量化器类 TestQuantizer1，继承自 Quantizer 类
            def transform_for_annotation(
                self, model: torch.fx.GraphModule
            ) -> torch.fx.GraphModule:
                # 用于生成模型注释的转换方法
                for n in model.graph.nodes:
                    # 遍历模型的图中的每个节点
                    if n.target == torch.ops.aten.add.Tensor:
                        # 如果节点的目标是 torch.ops.aten.add.Tensor
                        n.target = torch.ops.aten.mul.Tensor
                        # 将节点的目标修改为 torch.ops.aten.mul.Tensor
                return model
                # 返回修改后的模型

            def annotate(self, model: torch.fx.GraphModule) -> torch.fx.GraphModule:
                # 生成模型注释的方法
                return model
                # 返回原始模型

            def validate(self, model: torch.fx.GraphModule) -> None:
                # 用于验证模型的方法
                pass
                # 不执行任何操作

        class TestQuantizer2(Quantizer):
            # 定义量化器类 TestQuantizer2，继承自 Quantizer 类
            def transform_for_annotation(
                self, model: torch.fx.GraphModule
            ) -> torch.fx.GraphModule:
                # 用于生成模型注释的转换方法
                for n in model.graph.nodes:
                    # 遍历模型的图中的每个节点
                    if n.target == torch.ops.aten.sub.Tensor:
                        # 如果节点的目标是 torch.ops.aten.sub.Tensor
                        n.target = torch.ops.aten.div.Tensor
                        # 将节点的目标修改为 torch.ops.aten.div.Tensor
                return model
                # 返回修改后的模型

            def annotate(self, model: torch.fx.GraphModule) -> torch.fx.GraphModule:
                # 生成模型注释的方法
                return model
                # 返回原始模型

            def validate(self, model: torch.fx.GraphModule) -> None:
                # 用于验证模型的方法
                pass
                # 不执行任何操作

        class M(torch.nn.Module):
            # 定义一个简单的神经网络模型 M
            def forward(self, x, y, z):
                return x + y - z
                # 定义前向传播方法

        m = M().eval()
        # 创建 M 类的实例，并将其设置为评估模式
        quantizer = ComposableQuantizer([TestQuantizer1(), TestQuantizer2()])
        # 创建可组合的量化器实例，包含 TestQuantizer1 和 TestQuantizer2
        example_inputs = (
            torch.randn(1, 2, 3, 3),
            torch.randn(1, 2, 3, 3),
            torch.randn(1, 2, 3, 3),
        )
        # 创建示例输入张量
        m = capture_pre_autograd_graph(m, example_inputs)
        # 捕获 Autograd 图前的模型状态，并使用示例输入
        m = prepare_pt2e(m, quantizer)
        # 准备将 PyTorch 模型转换为量化模型，应用量化器
        m(*example_inputs)
        # 对模型应用输入张量进行前向传播
        node_occurrence = {
            ns.call_function(torch.ops.aten.add.Tensor): 0,
            ns.call_function(torch.ops.aten.sub.Tensor): 0,
            ns.call_function(torch.ops.aten.mul.Tensor): 1,
            ns.call_function(torch.ops.aten.div.Tensor): 1,
        }
        # 定义预期节点出现次数的字典
        self.checkGraphModuleNodes(m, expected_node_occurrence=node_occurrence)
        # 调用检查模型节点的方法，并传入预期的节点出现次数参数
    # 定义一个测试函数，用于测试嵌入量化器
    def test_embedding_quantizer(self):
        # 创建一个评估模式下的嵌入模块实例
        m_eager = TestHelperModules.EmbeddingModule().eval()
        # 定义一个包含索引的张量
        indices = torch.tensor(
            [
                9,
                6,
                5,
                7,
                8,
                8,
                9,
                2,
                8,
                6,
                6,
                9,
                1,
                6,
                8,
                8,
                3,
                2,
                3,
                6,
                3,
                6,
                5,
                7,
                0,
                8,
                4,
                6,
                5,
                8,
                2,
                3,
            ]
        )
        # 定义一个包含示例输入的元组
        example_inputs = (indices,)

        # 创建一个嵌入量化器实例
        quantizer = EmbeddingQuantizer()
        # 定义节点出现次数的字典
        node_occurrence = {
            # 注意：权重的量化操作被常量传播
            torch.ops.quantized_decomposed.quantize_per_channel.default: 0,
            torch.ops.quantized_decomposed.dequantize_per_channel.default: 1,
        }
        # 定义节点列表
        node_list = [
            torch.ops.quantized_decomposed.dequantize_per_channel.default,
            torch.ops.aten.embedding.default,
        ]
        # 与短期工作流进行比较
        # 由于量化和去量化操作导致的数值差异，不能与 FX 量化进行比较
        qconfig = default_per_channel_symmetric_qnnpack_qconfig
        # 创建 QConfigMapping 实例并设置全局配置
        qconfig_mapping = QConfigMapping().set_global(qconfig)
        # 设置对象类型映射，这里将 torch.nn.Embedding 映射到 float_qparams_weight_only_qconfig
        qconfig_mapping = qconfig_mapping.set_object_type(
            torch.nn.Embedding, float_qparams_weight_only_qconfig
        )
        # 调用测试量化器方法，传入相应参数进行测试
        self._test_quantizer(
            m_eager,
            example_inputs,
            quantizer,
            node_occurrence,
            node_list,
            True,
            qconfig_mapping,
        )
    # 定义一个测试方法，用于测试可组合量化器在线性卷积中的应用
    def test_composable_quantizer_linear_conv(self):
        # 创建动态量化器对象
        dynamic_quantizer = XNNPACKQuantizer()
        # 获取动态对称量化配置
        quantization_config_dynamic = get_symmetric_quantization_config(
            is_per_channel=False, is_dynamic=True
        )
        # 将动态量化配置应用到动态量化器上
        dynamic_quantizer.set_global(quantization_config_dynamic)
        
        # 创建静态量化器对象
        static_quantizer = XNNPACKQuantizer()
        # 获取静态对称量化配置
        quantization_config = get_symmetric_quantization_config(is_per_channel=True)
        # 将静态量化配置应用到静态量化器上
        static_quantizer.set_global(quantization_config)
        
        # 注意：这里必须先应用动态量化器
        # 因为静态量化器会使用静态的量化规格进行线性量化
        # 如果先应用静态量化器，则无法再应用动态量化器
        # 创建可组合量化器对象，按顺序包含动态量化器和静态量化器
        composable_quantizer = ComposableQuantizer(
            [dynamic_quantizer, static_quantizer]
        )
        
        # 创建测试用的 ConvLinearWPermute 模块并设为评估模式
        m_eager = TestHelperModules.ConvLinearWPermute().eval()

        # 定义节点出现次数字典，用于设置量化操作的期望次数
        node_occurrence = {
            torch.ops.quantized_decomposed.quantize_per_tensor.tensor: 1,
            torch.ops.quantized_decomposed.dequantize_per_tensor.tensor: 1,
            # 注意: 权重的量化操作已经被常量传播
            torch.ops.quantized_decomposed.quantize_per_tensor.default: 3,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default: 4,
            # 注意: 权重的量化操作已经被常量传播
            torch.ops.quantized_decomposed.quantize_per_channel.default: 0,
            torch.ops.quantized_decomposed.dequantize_per_channel.default: 1,
        }
        
        # 创建激活函数的仿真量化观察者对象
        act_affine_quant_obs = observer.PlaceholderObserver.with_args(
            dtype=torch.qint8,
            qscheme=torch.per_tensor_affine,
            quant_min=-128,
            quant_max=127,
            eps=2**-12,
            is_dynamic=True,
        )
        
        # 创建动态量化配置对象，指定激活函数的仿真量化观察者和权重的范围为[-127, 127]
        dynamic_qconfig = QConfig(
            activation=act_affine_quant_obs,
            weight=weight_observer_range_neg_127_to_127,
        )
        
        # 测试 2D 输入情况
        example_inputs = (torch.randn(2, 3, 4, 4),)
        
        # 使用默认的每通道对称 QNNPACK 量化配置
        qconfig = default_per_channel_symmetric_qnnpack_qconfig
        
        # 创建量化配置映射对象，并将默认配置设置为全局配置
        qconfig_mapping = QConfigMapping().set_global(qconfig)
        
        # 将 Linear 层的量化配置设置为动态量化配置
        qconfig_mapping.set_object_type(torch.nn.Linear, dynamic_qconfig)
        
        # 由于 fx 量化工作流似乎不会为此模型的 permute 节点传播观察者，因此关闭与 fx 的检查
        # 但出乎意料的是，对于 EmbeddingConvLinearModule 模型，它确实会传播观察者
        # TODO: 弄清楚传播的正确行为
        # 调用 _test_quantizer 方法，测试量化器的功能
        self._test_quantizer(
            m_eager,
            example_inputs,
            composable_quantizer,
            node_occurrence,
            [],
            False,
            qconfig_mapping,
        )
    # 在给定的图模块中查找并返回第一个目标与指定操作重载匹配的节点，如果找不到则抛出异常
    def _get_node(self, m: torch.fx.GraphModule, target: torch._ops.OpOverload):
        """
        Return the first node matching the specified target, throwing an exception
        if no such batch norm node is found.
        """
        # 遍历图中的节点
        for n in m.graph.nodes:
            # 检查节点的目标是否与给定的目标匹配
            if n.target == target:
                return n
        # 如果找不到匹配的节点，则抛出值错误异常
        raise ValueError("Did not find node with target ", target)

    # 测试在训练和评估模式之间切换 dropout 行为，使用 `move_exported_model_to_eval` 和 `move_exported_model_to_train` API
    def _test_move_exported_model_dropout(self, inplace: bool):
        """
        Test switching dropout behavior between train and eval modes using
        `move_exported_model_to_eval` and `move_exported_model_to_train` APIs.
        """

        # 定义一个简单的模型类 M
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 添加一个 dropout 层，根据 inplace 参数选择是否原地操作
                self.dropout = torch.nn.Dropout(0.5, inplace=inplace)

            def forward(self, x):
                # 模型的前向传播，应用 dropout 层
                return self.dropout(x)

        # 创建一个示例输入
        example_inputs = (torch.randn(1),)
        # 实例化模型并设置为训练模式
        m = M().train()
        # 捕获模型的预自动微分图
        m = capture_pre_autograd_graph(m, example_inputs)
        # 根据 inplace 参数选择对应的 dropout 操作目标
        if inplace:
            target = torch.ops.aten.dropout_.default
        else:
            target = torch.ops.aten.dropout.default

        # 断言 dropout 操作存在，并且处于训练模式
        dropout_node = self._get_node(m, target)
        self.assertTrue(dropout_node is not None)
        self.assertTrue(dropout_node.args[2])

        # 将模型切换到评估模式
        torch.ao.quantization.move_exported_model_to_eval(m)

        # 断言 dropout 操作现在处于评估模式
        dropout_node = self._get_node(m, target)
        self.assertTrue(dropout_node is not None)
        self.assertTrue(not dropout_node.args[2])

        # 将模型切换回训练模式
        torch.ao.quantization.move_exported_model_to_train(m)

        # 断言 dropout 操作现在又回到了训练模式
        dropout_node = self._get_node(m, target)
        self.assertTrue(dropout_node is not None)
        self.assertTrue(dropout_node.args[2])

    # 测试非原地操作下的 dropout 行为切换
    def test_move_exported_model_dropout(self):
        self._test_move_exported_model_dropout(inplace=False)

    # 测试原地操作下的 dropout 行为切换
    def test_move_exported_model_dropout_inplace(self):
        self._test_move_exported_model_dropout(inplace=True)

    # 获取当前环境中批量归一化操作的训练和评估模式对应的操作
    def _get_bn_train_eval_ops(self):
        if TEST_WITH_ROCM:
            return (
                torch.ops.aten.miopen_batch_norm.default,
                torch.ops.aten.miopen_batch_norm.default,
            )
        elif TEST_CUDA:
            return (
                torch.ops.aten.cudnn_batch_norm.default,
                torch.ops.aten.cudnn_batch_norm.default,
            )
        else:
            return (
                torch.ops.aten._native_batch_norm_legit.default,
                torch.ops.aten._native_batch_norm_legit_no_training.default,
            )
    def test_move_exported_model_bn(self):
        """
        Test switching batch_norm behavior between train and eval modes using
        `move_exported_model_to_eval` and `move_exported_model_to_train` APIs.
        """

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.bn = torch.nn.BatchNorm2d(3)  # 初始化一个包含3个通道的批标准化层

            def forward(self, x):
                return self.bn(x)  # 前向传播函数，应用批标准化层到输入张量x

        if TEST_CUDA:
            m = M().train().cuda()  # 如果CUDA可用，创建一个训练模式下的M模型并移至CUDA设备
            example_inputs = (torch.randn(1, 3, 3, 3).cuda(),)  # 生成一个随机输入张量，也移至CUDA设备
        else:
            m = M().train()  # 创建一个训练模式下的M模型
            example_inputs = (torch.randn(1, 3, 3, 3),)  # 生成一个随机输入张量

        bn_train_op, bn_eval_op = self._get_bn_train_eval_ops()  # 获取批标准化层的训练和评估操作
        m = capture_pre_autograd_graph(m, example_inputs)  # 捕获前自动求导图

        # Assert that batch norm op exists and is in train mode
        bn_node = self._get_node(m, bn_train_op)  # 获取模型中批标准化层的训练操作节点
        self.assertTrue(bn_node is not None)  # 断言批标准化层节点存在
        self.assertTrue(bn_node.args[5])  # 断言批标准化层处于训练模式

        # Move to eval
        torch.ao.quantization.move_exported_model_to_eval(m)  # 将模型转换为评估模式

        # Assert that batch norm op is now in eval mode
        bn_node = self._get_node(m, bn_eval_op)  # 获取模型中批标准化层的评估操作节点
        self.assertTrue(bn_node is not None)  # 断言批标准化层节点存在

        # Move to train
        torch.ao.quantization.move_exported_model_to_train(m)  # 将模型转换为训练模式

        # Assert that batch norm op is now in train mode again
        bn_node = self._get_node(m, bn_train_op)  # 再次获取模型中批标准化层的训练操作节点
        self.assertTrue(bn_node is not None)  # 断言批标准化层节点存在
        self.assertTrue(bn_node.args[5])  # 断言批标准化层处于训练模式

    def test_disallow_eval_train(self):
        m = TestHelperModules.ConvWithBNRelu(relu=True)  # 创建具有ReLU的带有批标准化的卷积模型
        example_inputs = (torch.rand(3, 3, 5, 5),)  # 生成一个随机输入示例

        # Before export: this is OK
        m.eval()  # 设置模型为评估模式
        m.train()  # 设置模型为训练模式

        # After export: this is not OK
        m = capture_pre_autograd_graph(m, example_inputs)  # 捕获前自动求导图
        with self.assertRaises(NotImplementedError):
            m.eval()  # 断言在导出后调用评估模式会引发未实现错误
        with self.assertRaises(NotImplementedError):
            m.train()  # 断言在导出后调用训练模式会引发未实现错误

        # After prepare: still not OK
        quantizer = XNNPACKQuantizer()  # 创建一个XNNPACK量化器
        m = prepare_qat_pt2e(m, quantizer)  # 准备模型进行量化感知训练
        with self.assertRaises(NotImplementedError):
            m.eval()  # 断言在准备后调用评估模式会引发未实现错误
        with self.assertRaises(NotImplementedError):
            m.train()  # 断言在准备后调用训练模式会引发未实现错误

        # After convert: still not OK
        m = convert_pt2e(m)  # 将PyTorch模型转换为Essential模型
        with self.assertRaises(NotImplementedError):
            m.eval()  # 断言在转换后调用评估模式会引发未实现错误
        with self.assertRaises(NotImplementedError):
            m.train()  # 断言在转换后调用训练模式会引发未实现错误
    def test_allow_exported_model_train_eval(self):
        # 定义一个简单的神经网络模型类
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.bn = torch.nn.BatchNorm2d(3)  # 初始化一个批归一化层
                self.dropout = torch.nn.Dropout(0.5)  # 初始化一个dropout层

            def forward(self, x):
                x = self.bn(x)  # 对输入进行批归一化处理
                x = self.dropout(x)  # 对输入进行dropout处理
                return x

        if TEST_CUDA:
            m = M().train().cuda()  # 如果支持CUDA，则将模型实例化为在训练模式下并移到GPU上
            example_inputs = (torch.randn(1, 3, 3, 3).cuda(),)
        else:
            m = M().train()  # 否则将模型实例化为在训练模式下
            example_inputs = (torch.randn(1, 3, 3, 3),)
        
        # 获取批归一化层在训练和评估模式下的操作
        bn_train_op, bn_eval_op = self._get_bn_train_eval_ops()
        
        # 对模型进行预捕捉前自动图形
        m = capture_pre_autograd_graph(m, example_inputs)

        def _assert_ops_are_correct(m: torch.fx.GraphModule, train: bool):
            targets = [n.target for n in m.graph.nodes]
            bn_op = bn_train_op if train else bn_eval_op
            bn_node = self._get_node(m, bn_op)  # 获取指定操作对应的节点
            self.assertTrue(bn_node is not None)
            if TEST_CUDA:
                self.assertEqual(bn_node.args[5], train)
            dropout_node = self._get_node(m, torch.ops.aten.dropout.default)  # 获取dropout操作对应的节点
            self.assertEqual(dropout_node.args[2], train)

        # 在包装之前：这是不允许的
        with self.assertRaises(NotImplementedError):
            m.eval()
        with self.assertRaises(NotImplementedError):
            m.train()

        # 包装之后：不会报错，并相应地交换操作
        torch.ao.quantization.allow_exported_model_train_eval(m)
        m.eval()
        _assert_ops_are_correct(m, train=False)
        m.train()
        _assert_ops_are_correct(m, train=True)

        # 在准备但未包装之后：这是不允许的
        quantizer = XNNPACKQuantizer()
        m = prepare_qat_pt2e(m, quantizer)
        with self.assertRaises(NotImplementedError):
            m.eval()
        with self.assertRaises(NotImplementedError):
            m.train()

        # 在准备并包装之后：不会报错，并相应地交换操作
        torch.ao.quantization.allow_exported_model_train_eval(m)
        m.eval()
        _assert_ops_are_correct(m, train=False)
        m.train()
        _assert_ops_are_correct(m, train=True)

        # 在转换但未包装之后：这是不允许的
        m = convert_pt2e(m, fold_quantize=True)
        with self.assertRaises(NotImplementedError):
            m.eval()
        with self.assertRaises(NotImplementedError):
            m.train()

        # 在转换并包装之后：不会报错，并相应地交换操作
        torch.ao.quantization.allow_exported_model_train_eval(m)
        m.eval()
        _assert_ops_are_correct(m, train=False)
        m.train()
        _assert_ops_are_correct(m, train=True)
    # 定义测试方法，用于验证模型是否已导出
    def test_model_is_exported(self):
        # 创建带有 Batch Normalization 和 ReLU 的 ConvWithBNRelu 测试模型实例
        m = TestHelperModules.ConvWithBNRelu(relu=True)
        # 创建示例输入数据
        example_inputs = (torch.rand(3, 3, 5, 5),)
        # 获取通过 capture_pre_autograd_graph 导出的图
        exported_gm = capture_pre_autograd_graph(m, example_inputs)
        # 对模型 m 进行符号化追踪，得到 fx_traced_gm
        fx_traced_gm = torch.fx.symbolic_trace(m, example_inputs)
        
        # 断言：使用 export_utils 检查 exported_gm 是否已导出
        self.assertTrue(
            torch.ao.quantization.pt2e.export_utils.model_is_exported(exported_gm)
        )
        # 断言：使用 export_utils 检查 fx_traced_gm 是否已导出
        self.assertFalse(
            torch.ao.quantization.pt2e.export_utils.model_is_exported(fx_traced_gm)
        )
        # 断言：使用 export_utils 检查模型 m 是否已导出
        self.assertFalse(torch.ao.quantization.pt2e.export_utils.model_is_exported(m))

    # 定义测试方法，验证量化 API 是否可以安全地多次调用
    def test_reentrant(self):
        """Test we can safely call quantization apis multiple times"""
        # 创建带有 ConvBnReLU2dAndLinearReLU 的测试模型实例 m
        m = TestHelperModules.ConvBnReLU2dAndLinearReLU()
        # 创建示例输入数据
        example_inputs = (torch.randn(3, 3, 10, 10),)

        # 创建 XNNPACKQuantizer 实例 quantizer，配置为按通道对称量化和量化感知训练
        quantizer = XNNPACKQuantizer().set_global(
            get_symmetric_quantization_config(is_per_channel=True, is_qat=True)
        )
        # 使用 capture_pre_autograd_graph 捕获 m.conv_bn_relu 的图
        m.conv_bn_relu = capture_pre_autograd_graph(m.conv_bn_relu, example_inputs)
        # 使用 prepare_qat_pt2e 准备 m.conv_bn_relu 以进行量化感知训练
        m.conv_bn_relu = prepare_qat_pt2e(m.conv_bn_relu, quantizer)
        # 执行 m(*example_inputs) 来计算输出
        m(*example_inputs)
        # 使用 convert_pt2e 将 m.conv_bn_relu 转换为量化引擎格式
        m.conv_bn_relu = convert_pt2e(m.conv_bn_relu)

        # 创建新的 XNNPACKQuantizer 实例 quantizer，配置为线性层按通道非对称量化
        quantizer = XNNPACKQuantizer().set_module_type(
            torch.nn.Linear, get_symmetric_quantization_config(is_per_channel=False)
        )
        # 使用 capture_pre_autograd_graph 捕获整个模型 m 的图
        m = capture_pre_autograd_graph(m, example_inputs)
        # 使用 prepare_pt2e 准备模型 m 以适应新的量化设置
        m = prepare_pt2e(m, quantizer)
        # 使用 convert_pt2e 将整个模型 m 转换为量化引擎格式
        m = convert_pt2e(m)

        # 定义期望出现次数和节点列表，用于验证图模块的节点
        node_occurrence = {
            ns.call_function(
                torch.ops.quantized_decomposed.quantize_per_tensor.default
            ): 4,
            ns.call_function(
                torch.ops.quantized_decomposed.dequantize_per_tensor.default
            ): 5,
            ns.call_function(
                torch.ops.quantized_decomposed.dequantize_per_channel.default
            ): 1,
        }
        node_list = [
            ns.call_function(
                torch.ops.quantized_decomposed.dequantize_per_tensor.default
            ),
            ns.call_function(torch.ops.aten.conv2d.default),
            ns.call_function(torch.ops.aten.relu.default),
            ns.call_function(
                torch.ops.quantized_decomposed.quantize_per_tensor.default
            ),
            ns.call_function(
                torch.ops.quantized_decomposed.dequantize_per_tensor.default
            ),
            ns.call_function(torch.ops.aten.linear.default),
            ns.call_function(
                torch.ops.quantized_decomposed.quantize_per_tensor.default
            ),
        ]
        
        # 调用 checkGraphModuleNodes 方法验证图模块的节点是否符合预期
        self.checkGraphModuleNodes(
            m, expected_node_occurrence=node_occurrence, expected_node_list=node_list
        )
    # 定义测试函数，用于测试分组卷积操作的逐通道量化
    def test_groupwise_per_channel_quant(self):
        # 创建测试用的 GroupwiseConv2d 模块实例
        m = TestHelperModules.GroupwiseConv2d()
        # 创建 XNNPACKQuantizer 实例，用于量化操作
        quantizer = XNNPACKQuantizer()
        # 获取逐通道对称量化配置
        operator_config = get_symmetric_quantization_config(is_per_channel=True)
        # 设置全局量化配置
        quantizer.set_global(operator_config)
        # 获取模块的示例输入
        example_inputs = m.example_inputs()
        # 对模块进行量化
        m = self._quantize(m, quantizer, example_inputs)
        # 确保模块可以正常运行
        m(*example_inputs)

    # 测试速度的函数
    def test_speed(self):
        import time

        # 定义动态量化函数
        def dynamic_quantize_pt2e(model, example_inputs):
            # 重置 PyTorch 动态量化状态
            torch._dynamo.reset()
            # 捕获前向传播图以进行自动微分
            model = capture_pre_autograd_graph(model, example_inputs)
            # 创建嵌入量化器实例
            embedding_quantizer = EmbeddingQuantizer()
            # 创建动态量化器实例
            dynamic_quantizer = XNNPACKQuantizer()
            # 获取动态逐通道对称量化配置
            operator_config_dynamic = get_symmetric_quantization_config(
                is_per_channel=True, is_dynamic=True
            )
            # 设置全局动态量化配置
            dynamic_quantizer.set_global(operator_config_dynamic)
            # 创建可组合量化器实例，包括嵌入量化器和动态量化器
            composed_quantizer = ComposableQuantizer(
                [embedding_quantizer, dynamic_quantizer]
            )
            # 记录当前时间
            prev = time.time()
            # 准备量化感知训练阶段
            model = prepare_qat_pt2e(model, composed_quantizer)
            # 记录当前时间
            cur = time.time()
            # 运行模型的前向传播，进行校准
            model(*example_inputs)
            # 记录当前时间
            prev = time.time()
            # 将模型转换为量化表示
            model = convert_pt2e(model)
            # 记录当前时间
            cur = time.time()
            # 返回量化后的模型
            return model

        # 定义一个简单的线性模型类
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(5, 5)

            def forward(self, x):
                return self.linear(x)

        # 创建并评估模型
        m = M().eval()
        # 创建示例输入
        example_inputs = (torch.randn(5, 5),)
        # 进行动态量化
        _ = dynamic_quantize_pt2e(m, example_inputs)
    def test_conv_transpose_bn_relu(self):
        # 定义一个名为BackendAQuantizer的内部类，继承自Quantizer类
        class BackendAQuantizer(Quantizer):
            # 为模型图注释量化信息的方法
            def annotate(self, model: torch.fx.GraphModule) -> torch.fx.GraphModule:
                # 定义8位整数量化规格
                int8_qspec = QuantizationSpec(
                    dtype=torch.int8,
                    quant_min=-128,
                    quant_max=127,
                    qscheme=torch.per_tensor_symmetric,
                    is_dynamic=False,
                    observer_or_fake_quant_ctr=observer.default_weight_observer,
                )
                # 定义量化配置
                quantization_config = QuantizationConfig(
                    input_activation=int8_qspec,
                    weight=int8_qspec,
                    bias=None,
                    output_activation=int8_qspec,
                )
                # conv_transpose + bn在PTQ中自动融合（不可配置），因此只需为conv_transpose + bn + relu模式注释conv_transpose + relu
                OP_TO_ANNOTATOR["conv_transpose_relu"](model, quantization_config)

            # 验证模型图的方法，暂时未实现
            def validate(self, model: torch.fx.GraphModule) -> None:
                pass

        # 示例输入数据
        example_inputs = (torch.randn(1, 3, 5, 5),)
        # 节点出现次数的字典
        node_occurrence = {
            # 第一个卷积输入的两次，第一个卷积输出的一次
            torch.ops.quantized_decomposed.quantize_per_tensor.default: 2,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default: 3,
        }
        # 节点列表
        node_list = [
            torch.ops.quantized_decomposed.dequantize_per_tensor.default,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default,
            torch.ops.aten.conv_transpose2d.input,
            torch.ops.aten.relu.default,
            torch.ops.quantized_decomposed.quantize_per_tensor.default,
        ]
        # 调用测试量化器方法
        self._test_quantizer(
            TestHelperModules.ConvTWithBNRelu(relu=True, bn=True),  # 使用带有ReLU和BN的测试辅助模块
            example_inputs,  # 示例输入数据
            BackendAQuantizer(),  # 使用BackendAQuantizer作为量化器
            node_occurrence,  # 节点出现次数的字典
            node_list,  # 节点列表
        )
    def test_multi_users_without_output_observer(self):
        """
        Test the case in which a node is used by multiple users,
        and had its output observer removed.
        """

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 定义一个包含 3 个输入通道和输出通道的 3x3 卷积层
                self.conv = torch.nn.Conv2d(3, 3, 3)

            def forward(self, x):
                # 在前向传播中应用定义的卷积层
                x = self.conv(x)
                # 返回两个张量：卷积输出和卷积输出加一
                return x, x + 1

        # 创建一个示例输入张量
        example_inputs = (torch.randn(1, 3, 5, 5),)
        # 实例化模型 M
        m = M()
        # 在模型上捕获预自动微分图
        m = capture_pre_autograd_graph(m, example_inputs)
        # 实例化 XNNPACKQuantizer，并设置全局的对称量化配置
        quantizer = XNNPACKQuantizer().set_global(
            get_symmetric_quantization_config(),
        )
        # 使用量化器准备模型以进行量化
        m = prepare_pt2e(m, quantizer)
        # 在示例输入上运行模型的前向传播
        m(*example_inputs)

        # 移除输出观察器
        observer_to_remove = None
        # 遍历模型图中的节点
        for n in m.graph.nodes:
            if n.op == "output":
                # 找到要移除的观察器节点
                observer_to_remove = n.args[0][0]
                assert observer_to_remove.op == "call_module"
                assert observer_to_remove.target.startswith("activation_post_process_")
                break
        # 确保找到要移除的观察器节点
        assert observer_to_remove is not None
        # 用观察器节点的参数替换所有使用它的地方
        observer_to_remove.replace_all_uses_with(observer_to_remove.args[0])
        # 从模型图中删除观察器节点
        m.graph.erase_node(observer_to_remove)
        # 重新编译模型
        m.recompile()

        # 执行模型转换应当成功
        m = convert_pt2e(m)
        # 在示例输入上再次运行模型的前向传播
        m(*example_inputs)
# 使用给定的参数化测试类 TestQuantizePT2E 实例化并执行测试
instantiate_parametrized_tests(TestQuantizePT2E)
```