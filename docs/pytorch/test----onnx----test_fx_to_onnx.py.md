# `.\pytorch\test\onnx\test_fx_to_onnx.py`

```py
# Owner(s): ["module: onnx"]
from __future__ import annotations  # 允许使用注解类型的未来特性

import logging  # 导入日志模块

import tempfile  # 导入临时文件模块

from typing import Mapping, Tuple  # 导入类型提示的 Mapping 和 Tuple

import onnx  # 导入 ONNX 模块
import onnx.inliner  # 导入 ONNX 模块中的 inliner 子模块
import pytorch_test_common  # 导入 PyTorch 测试通用模块
import transformers  # type: ignore[import] 导入 transformers 模块，忽略导入时的类型检查提示

import torch  # 导入 PyTorch 模块
from torch import nn  # 从 PyTorch 中导入神经网络模块
from torch._subclasses import fake_tensor  # 从 PyTorch 中导入 fake_tensor 子类
from torch.nn import functional as F  # 导入 PyTorch 中的 functional 模块并命名为 F
from torch.onnx import dynamo_export, ExportOptions  # 从 PyTorch 中导入 dynamo_export 和 ExportOptions
from torch.onnx._internal.diagnostics import infra  # noqa: TCH001 导入内部诊断模块 infra，忽略 TCH001 错误类型
from torch.onnx._internal.fx import diagnostics, registration  # 从内部 FX 模块中导入诊断和注册模块
from torch.testing._internal import common_utils  # 从内部测试模块中导入通用工具模块


def assert_has_diagnostics(
    diagnostic_context: diagnostics.DiagnosticContext,  # 定义断言函数 assert_has_diagnostics，接收诊断上下文和规则作为参数
    rule: infra.Rule,  # 规则参数
    level: infra.Level,  # 级别参数
    expected_node: str,  # 预期节点字符串
):
    rule_level_pairs = (rule.id, level.name.lower())  # 构建规则和级别的元组对
    sarif_log = diagnostic_context.sarif_log()  # 调用诊断上下文的 sarif_log 方法
    actual_results = []  # 初始化空列表，用于存储实际结果
    for run in sarif_log.runs:  # 遍历 sarif_log 的运行结果
        if run.results is None:  # 如果运行结果为空，则跳过
            continue
        for result in run.results:  # 遍历每个运行结果中的结果
            id_level_pair = (result.rule_id, result.level)  # 获取结果的规则 ID 和级别
            actual_results.append(id_level_pair)  # 将规则 ID 和级别添加到实际结果列表中
            if (
                rule_level_pairs == id_level_pair  # 如果规则 ID 和级别匹配
                and result.message.text  # 并且结果包含文本消息
                and result.message.markdown  # 并且结果包含 Markdown 格式的消息
                and expected_node in result.message.text  # 并且预期节点包含在文本消息中
            ):
                return  # 如果匹配，则返回

    raise AssertionError(  # 如果未找到匹配结果，则抛出断言错误
        f"Expected diagnostic results of rule id and level pair {rule_level_pairs} "
        f"not found with expected error node {expected_node} and "
        f"Actual diagnostic results: {actual_results}"
    )


@common_utils.instantiate_parametrized_tests  # 使用 common_utils 模块中的 instantiate_parametrized_tests 装饰器
class TestFxToOnnx(pytorch_test_common.ExportTestCase):  # 定义测试类 TestFxToOnnx，继承自 ExportTestCase 类
    def setUp(self):  # 设置测试环境
        super().setUp()  # 调用父类的 setUp 方法
        self.export_options = ExportOptions()  # 初始化 ExportOptions 实例作为导出选项

    def tearDown(self):  # 清理测试环境
        super().tearDown()  # 调用父类的 tearDown 方法

    def test_simple_function(self):  # 测试简单函数
        def func(x):  # 定义函数 func，接收参数 x
            y = x + 1  # 计算 x + 1，赋值给 y
            z = y.relu()  # 计算 y 的 ReLU，赋值给 z
            return (y, z)  # 返回 y 和 z 的元组

        _ = dynamo_export(  # 执行 dynamo_export 函数
            func, torch.randn(1, 1, 2), export_options=self.export_options  # 使用 func 和随机张量调用 dynamo_export，并传入导出选项
        )

    def test_empty(self):  # 测试空函数
        # 由于 `torch.empty` 返回具有未初始化数据的张量，因此无法在 `test_fx_to_onnx_with_onnxruntime.py` 中进行结果比较测试。
        def func(x):  # 定义函数 func，接收参数 x
            return torch.empty(x.size(), dtype=torch.int64)  # 返回一个形状与 x 相同、数据类型为 torch.int64 的空张量

        tensor_x = torch.randn(1, 1, 2)  # 创建一个随机张量 tensor_x
        _ = dynamo_export(func, tensor_x, export_options=self.export_options)  # 使用 func 和 tensor_x 调用 dynamo_export

    def test_args_used_for_export_is_not_converted_to_fake_tensors(self):  # 测试用于导出的参数不会转换为 fake_tensors
        def func(x, y):  # 定义函数 func，接收参数 x 和 y
            return x + y  # 返回 x + y 的结果

        tensor_x = torch.randn(1, 1, 2)  # 创建一个随机张量 tensor_x
        tensor_y = torch.randn(1, 1, 2)  # 创建一个随机张量 tensor_y
        _ = dynamo_export(func, tensor_x, tensor_y, export_options=self.export_options)  # 使用 func、tensor_x 和 tensor_y 调用 dynamo_export
        self.assertNotIsInstance(tensor_x, fake_tensor.FakeTensor)  # 断言 tensor_x 不是 fake_tensor.FakeTensor 的实例
        self.assertNotIsInstance(tensor_y, fake_tensor.FakeTensor)  # 断言 tensor_y 不是 fake_tensor.FakeTensor 的实例
    @common_utils.parametrize(
        "diagnostic_rule",
        [
            common_utils.subtest(
                diagnostics.rules.find_opschema_matched_symbolic_function,
                name="optional_inputs",
            ),
            common_utils.subtest(
                diagnostics.rules.op_level_debugging,
                name="get_attr_node_in_op_level_debug",
            ),
        ],
    )
    # 使用 parametrize 装饰器定义参数化测试，参数为 diagnostic_rule
    def test_mnist_exported_with_no_warnings(self, diagnostic_rule):
        # 定义一个 MNISTModel 类，继承自 nn.Module
        class MNISTModel(nn.Module):
            def __init__(self):
                super().__init__()
                # 定义模型的卷积层和全连接层
                self.conv1 = nn.Conv2d(1, 32, 3, 1, bias=False)
                self.conv2 = nn.Conv2d(32, 64, 3, 1, bias=False)
                self.fc1 = nn.Linear(9216, 128, bias=False)
                self.fc2 = nn.Linear(128, 10, bias=False)

            # 定义模型的前向传播方法
            def forward(self, tensor_x: torch.Tensor):
                tensor_x = self.conv1(tensor_x)
                tensor_x = F.sigmoid(tensor_x)
                tensor_x = self.conv2(tensor_x)
                tensor_x = F.sigmoid(tensor_x)
                tensor_x = F.max_pool2d(tensor_x, 2)
                tensor_x = torch.flatten(tensor_x, 1)
                tensor_x = self.fc1(tensor_x)
                tensor_x = F.sigmoid(tensor_x)
                tensor_x = self.fc2(tensor_x)
                output = F.log_softmax(tensor_x, dim=1)
                return output

        # 创建一个随机张量作为输入
        tensor_x = torch.rand((64, 1, 28, 28), dtype=torch.float32)
        # 调用 dynamo_export 函数导出模型到 ONNX 格式
        onnx_program = dynamo_export(
            MNISTModel(), tensor_x, export_options=ExportOptions(op_level_debug=True)
        )

        # 断言模型导出后的诊断信息中包含特定的诊断规则和节点
        assert_has_diagnostics(
            onnx_program.diagnostic_context,
            diagnostic_rule,
            diagnostics.levels.NONE,
            expected_node="aten.convolution.default",
        )

    # 定义一个测试函数，测试复杂数据类型在操作级调试中没有警告
    def test_no_warnings_on_complex_dtype_in_op_level_debug(self):
        # 定义一个复杂模型类，继承自 torch.nn.Module
        class ComplexModel(torch.nn.Module):
            # 定义模型的前向传播方法
            def forward(self, input):
                return torch.ops.aten.mul(input, input)

        # 创建实部和虚部张量
        real = torch.tensor([1, 2], dtype=torch.float32)
        imag = torch.tensor([3, 4], dtype=torch.float32)
        # 创建一个复数张量
        x = torch.complex(real, imag)

        # 调用 dynamo_export 函数导出复杂模型到 ONNX 格式
        onnx_program = dynamo_export(
            ComplexModel(), x, export_options=ExportOptions(op_level_debug=True)
        )

        # 断言模型导出后的诊断信息中包含特定的诊断规则和节点
        assert_has_diagnostics(
            onnx_program.diagnostic_context,
            diagnostics.rules.op_level_debugging,
            diagnostics.levels.NONE,
            expected_node="aten.mul.Tensor",
        )
    def test_trace_only_op_with_evaluator(self):
        # 创建输入张量
        model_input = torch.tensor([[1.0, 2.0, 3.0], [1.0, 1.0, 2.0]])

        # 定义一个继承自 torch.nn.Module 的模型类
        class ArgminArgmaxModel(torch.nn.Module):
            # 定义模型的前向传播方法
            def forward(self, input):
                return (
                    torch.argmin(input),  # 计算输入张量的最小值索引
                    torch.argmax(input),  # 计算输入张量的最大值索引
                    torch.argmin(input, keepdim=True),  # 在保持维度的情况下计算最小值索引
                    torch.argmax(input, keepdim=True),  # 在保持维度的情况下计算最大值索引
                    torch.argmin(input, dim=0, keepdim=True),  # 沿着指定维度计算最小值索引并保持维度
                    torch.argmax(input, dim=1, keepdim=True),  # 沿着指定维度计算最大值索引并保持维度
                )

        # 使用 dynamo_export 导出模型
        _ = dynamo_export(
            ArgminArgmaxModel(), model_input, export_options=self.export_options
        )

    def test_multiple_outputs_op_with_evaluator(self):
        # 定义一个继承自 torch.nn.Module 的模型类
        class TopKModel(torch.nn.Module):
            # 定义模型的前向传播方法
            def forward(self, x):
                values, _ = torch.topk(x, 3)  # 获取输入张量 x 中的前三个最大值
                return torch.sum(values)  # 返回前三个最大值的和

        # 创建一个张量 x，用于模型输入
        x = torch.arange(1.0, 6.0, requires_grad=True)

        # 使用 dynamo_export 导出模型
        _ = dynamo_export(TopKModel(), x, export_options=self.export_options)

    def test_unsupported_indices_fake_tensor_generated_with_op_level_debug(self):
        # 定义一个继承自 torch.nn.Module 的模型类
        class EmbedModelWithoutPaddingIdx(torch.nn.Module):
            # 定义模型的前向传播方法
            def forward(self, input, emb):
                return torch.nn.functional.embedding(input, emb)  # 使用嵌入矩阵 emb 对输入 input 进行嵌入

        # 创建输入张量 x
        model = EmbedModelWithoutPaddingIdx()
        x = torch.randint(4, (4, 3, 2))
        embedding_matrix = torch.rand(10, 3)

        # 使用 dynamo_export 导出模型，并设置 op_level_debug=True
        onnx_program = dynamo_export(
            model,
            x,
            embedding_matrix,
            export_options=ExportOptions(op_level_debug=True),
        )

        # 断言是否有特定诊断信息
        assert_has_diagnostics(
            onnx_program.diagnostic_context,
            diagnostics.rules.op_level_debugging,
            diagnostics.levels.WARNING,
            expected_node="aten.embedding.default",
        )

    def test_unsupported_function_schema_raises_diagnostic_warning_when_found_nearest_match(
        self,
    ):
        # 定义一个继承自 torch.nn.Module 的模型类
        class TraceModel(torch.nn.Module):
            # 定义模型的前向传播方法
            def forward(self, input):
                return input.new_zeros(())  # 创建一个形状为 () 的零张量

        # 创建输入张量 x
        x = torch.randn((2, 3), dtype=torch.float32)

        # 使用 dynamo_export 导出模型
        onnx_program = dynamo_export(TraceModel(), x)

        # 断言是否有特定诊断信息
        assert_has_diagnostics(
            onnx_program.diagnostic_context,
            diagnostics.rules.find_opschema_matched_symbolic_function,
            diagnostics.levels.WARNING,
            expected_node="aten.new_zeros.default",
        )

    def test_perfect_match_on_sequence_and_bool_attributes(
        self,
    ):
        # 定义一个内部的 PyTorch 模型类 TraceModel
        class TraceModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 定义一个卷积层，输入通道数为16，输出通道数为33，卷积核大小为(3, 5)，步长为(2, 1)，填充为(4, 2)，扩张为(3, 1)
                self.conv2 = torch.nn.Conv2d(
                    16, 33, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1)
                )

            def forward(self, input):
                # 模型的前向传播，返回卷积层的输出
                return self.conv2(input)

        # 创建一个大小为(20, 16, 50, 50)的随机张量
        x = torch.randn(20, 16, 50, 50)
        # 导出模型为 ONNX 格式的程序，并且不包含调试级别的信息
        onnx_program = dynamo_export(
            TraceModel(), x, export_options=ExportOptions(op_level_debug=False)
        )
        # 断言程序中有诊断上下文，并且没有与预期的节点匹配的诊断
        assert_has_diagnostics(
            onnx_program.diagnostic_context,
            diagnostics.rules.find_opschema_matched_symbolic_function,
            diagnostics.levels.NONE,
            expected_node="aten.convolution.default",
        )

    # 定义测试函数：当使用默认参数时，ATen 克隆操作不会引发内存格式不足的警告
    def test_aten_clone_does_not_raise_warning_of_lack_of_memory_format(self):
        # 定义一个自定义的 PyTorch 模块类 CustomModule
        class CustomModule(torch.nn.Module):
            def forward(self, input):
                # 返回一个使用 ATen 克隆操作的张量，保留内存格式
                return torch.ops.aten.clone(input, memory_format=torch.preserve_format)

        # 创建一个大小为3的张量
        x = torch.tensor(3)
        # 导出模型为 ONNX 格式的程序
        onnx_program = dynamo_export(CustomModule(), x)
        # 断言程序中有诊断上下文，并且没有与预期的节点匹配的诊断
        assert_has_diagnostics(
            onnx_program.diagnostic_context,
            diagnostics.rules.find_opschema_matched_symbolic_function,
            diagnostics.levels.NONE,
            expected_node="aten.clone.default",
        )
    def test_missing_complex_onnx_variant_raises_errors_in_dispatcher(self):
        registry = torch.onnx.OnnxRegistry()

        # NOTE: simulate unsupported nodes
        # 创建 OpName 对象，表示名为 "aten.mul" 的张量操作
        aten_mul_tensor = registration.OpName.from_name_parts(
            namespace="aten", op_name="mul", overload="Tensor"
        )

        # Only keep real aten.mul to test missing complex aten.mul
        # 将注册表中对应于 aten.mul 张量操作的函数列表中，排除复数操作的函数
        registry._registry[aten_mul_tensor] = [
            onnx_func
            for onnx_func in registry._registry[aten_mul_tensor]
            if not onnx_func.is_complex
        ]

        class TraceModel(torch.nn.Module):
            def forward(self, input):
                return torch.ops.aten.mul.Tensor(input, input)

        x = torch.tensor([1 + 2j, 3 + 4j], dtype=torch.complex64)

        with self.assertRaises(torch.onnx.OnnxExporterError) as e:
            # 导出 ONNX 模型时，预期会触发异常，因为缺少复数版本的 aten.mul
            torch.onnx.dynamo_export(
                TraceModel(),
                x,
                export_options=torch.onnx.ExportOptions(onnx_registry=registry),
            )

        try:
            # 尝试再次导出 ONNX 模型
            torch.onnx.dynamo_export(
                TraceModel(),
                x,
                export_options=torch.onnx.ExportOptions(onnx_registry=registry),
            )
        except torch.onnx.OnnxExporterError as e:
            # 断言异常包含特定诊断信息，确认问题出现在 "aten.mul.Tensor" 操作
            assert_has_diagnostics(
                e.onnx_program.diagnostic_context,
                diagnostics.rules.no_symbolic_function_for_call_function,
                diagnostics.levels.ERROR,
                expected_node="aten.mul.Tensor",
            )

    def test_symbolic_shape_of_values_inside_function_is_exported_as_graph_value_info(
        self,
        ):
            # 定义一个名为 SubModule 的内部类，继承自 torch.nn.Module
            class SubModule(torch.nn.Module):
                # 实现 SubModule 类的 forward 方法，接受输入 x、y 和偏置 bias
                def forward(self, x, y, bias):
                    # 执行矩阵乘法操作 x @ y，并加上偏置 bias
                    output = x @ y
                    return output + bias

            # 定义一个名为 Module 的内部类，继承自 torch.nn.Module
            class Module(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    # 创建 SubModule 的实例并存储在 self.submodule 中
                    self.submodule = SubModule()

                # 实现 Module 类的 forward 方法，接受输入 x、y 和偏置 bias
                def forward(self, x, y, bias):
                    # 调用 self.submodule 的 forward 方法，传入 x、y 和 bias，返回其结果
                    return self.submodule(x, y, bias)

            # 创建一个大小为 (2, 3) 的随机张量 x
            x = torch.randn(2, 3)
            # 创建一个大小为 (3, 4) 的随机张量 y
            y = torch.randn(3, 4)
            # 创建一个大小为 (4,) 的随机张量 bias
            bias = torch.randn(4)
            # 使用 torch.onnx.dynamo_export 方法将 Module 实例转换为 ONNX 程序
            onnx_program = torch.onnx.dynamo_export(
                Module(),
                x,
                y,
                bias,
                # 使用动态形状导出选项
                export_options=torch.onnx.ExportOptions(dynamic_shapes=True),
            )
            # 获取导出的模型协议
            model_proto = onnx_program.model_proto

            # 定义一个内部函数 _assert_node_outputs_has_value_info，用于验证节点输出中是否包含值信息
            def _assert_node_outputs_has_value_info(
                node: onnx.NodeProto,
                value_infos: Mapping[str, onnx.ValueInfoProto],
                local_functions: Mapping[Tuple[str, str], onnx.FunctionProto],
                exclude_names_in_value_info,
                function_id: str = "",
            ):
                for output in node.output:
                    # 构建输出名称，如果存在 function_id，则加上前缀
                    name = f"{function_id}/{output}" if function_id else output
                    # 如果名称不在排除列表中，则断言该名称存在于 value_infos 中
                    if name not in exclude_names_in_value_info:
                        self.assertIn(name, value_infos)
                # 如果节点的领域以 "pkg.onnxscript.torch_lib" 开头，则没有可用的形状信息
                if node.domain.startswith("pkg.onnxscript.torch_lib"):
                    return
                # 如果存在本地函数，则递归检查函数中的节点输出
                if (
                    function := local_functions.get((node.domain, node.op_type))
                ) is not None:
                    for node in function.node:
                        function_id = f"{function.domain}::{function.name}"
                        _assert_node_outputs_has_value_info(
                            node,
                            value_infos,
                            local_functions,
                            exclude_names_in_value_info,
                            function_id,
                        )

            # 创建一个字典 type_infos，存储 model_proto 中的 value_info 信息
            type_infos = {vi.name: vi for vi in model_proto.graph.value_info}
            # 创建一个字典 functions，存储 model_proto 中的函数信息
            functions = {(f.domain, f.name): f for f in model_proto.functions}
            # 构建一个排除列表 exclude_names_in_value_info，包含输入、输出和初始化器的名称
            exclude_names_in_value_info = (
                [input.name for input in model_proto.graph.input]
                + [output.name for output in model_proto.graph.output]
                + [init.name for init in model_proto.graph.initializer]
            )
            # 遍历模型图中的每个节点，并调用 _assert_node_outputs_has_value_info 进行验证
            for node in model_proto.graph.node:
                _assert_node_outputs_has_value_info(
                    node, type_infos, functions, exclude_names_in_value_info
                )
    def test_dynamo_export_retains_readable_parameter_and_buffer_names(self):
        # 定义测试函数，用于验证导出的 ONNX 模型是否保留了可读参数和缓冲区的名称
        class SubModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 初始化子模块的卷积层和全连接层，以及一个注册的缓冲区
                self.conv2 = nn.Conv2d(32, 64, 3, 1, bias=False)
                self.fc1 = nn.Linear(9216, 128, bias=False)
                self.register_buffer("buffer", torch.randn(1, 128))

            def forward(self, tensor_x: torch.Tensor):
                # 子模块的前向传播方法，依次执行卷积、激活函数、池化、展平、全连接及缓冲区加和等操作
                tensor_x = self.conv2(tensor_x)
                tensor_x = F.sigmoid(tensor_x)
                tensor_x = F.max_pool2d(tensor_x, 2)
                tensor_x = torch.flatten(tensor_x, 1)
                tensor_x = self.fc1(tensor_x)
                tensor_x = tensor_x + self.buffer
                tensor_x = F.sigmoid(tensor_x)
                return tensor_x

        class MNISTModel(nn.Module):
            def __init__(self):
                super().__init__()
                # 初始化 MNIST 模型的卷积层、子模块和全连接层
                self.conv1 = nn.Conv2d(1, 32, 3, 1, bias=False)
                self.submodule = SubModule()
                self.fc2 = nn.Linear(128, 10, bias=False)

            def forward(self, tensor_x: torch.Tensor):
                # MNIST 模型的前向传播方法，依次执行卷积、激活函数、子模块、全连接及 log_softmax 操作
                tensor_x = self.conv1(tensor_x)
                tensor_x = F.sigmoid(tensor_x)
                tensor_x = self.submodule(tensor_x)
                tensor_x = self.fc2(tensor_x)
                output = F.log_softmax(tensor_x, dim=1)
                return output

        tensor_x = torch.rand((64, 1, 28, 28), dtype=torch.float32)

        model = MNISTModel()
        # 调用 torch.onnx.dynamo_export 方法导出模型到 ONNX 格式
        onnx_program = torch.onnx.dynamo_export(model, tensor_x)
        model_proto = onnx_program.model_proto

        # NOTE: initializers could be optimized away by onnx optimizer
        # 检查导出的模型中的初始化器是否被 ONNX 优化器优化掉了
        onnx_initilizers = {init.name for init in model_proto.graph.initializer}
        # 获取模型的所有状态字典的键集合
        torch_weights = {*model.state_dict().keys()}
        # 使用断言确保 ONNX 初始化器中的所有项都在模型的状态字典中
        self.assertTrue(onnx_initilizers.issubset(torch_weights))

    @common_utils.parametrize(
        "checkpoint_type",
        [
            common_utils.subtest(
                "state_dict",
                name="state_dict",
            ),
            common_utils.subtest(
                "state_dict",
                name="checkpoint_file",
            ),
        ],
    )
    def test_fake_tensor_mode_simple_invalid_input(self):
        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(2, 2)

            def forward(self, x):
                out = self.linear(x)
                return out

        real_model = Model()
        real_x = torch.rand(5, 2, 2)
        with torch.onnx.enable_fake_mode() as fake_context:
            fake_model = Model()
            fake_x = torch.rand(5, 2, 2)

            # TODO: Split each scenario on its own test case

            # Scenario 1: Fake model and fake input WITHOUT ExportOptions(fake_context=...)
            # Raises an exception because fake_context is None
            with self.assertRaises(torch.onnx.OnnxExporterError):
                export_options = ExportOptions(fake_context=None)
                _ = torch.onnx.dynamo_export(
                    fake_model, fake_x, export_options=export_options
                )

            # Scenario 2: Fake model and real input WITHOUT fake_context
            # Raises an exception because fake_context is None
            with self.assertRaises(torch.onnx.OnnxExporterError):
                export_options = ExportOptions(fake_context=None)
                _ = torch.onnx.dynamo_export(
                    fake_model, real_x, export_options=export_options
                )

            # Scenario 3: Real model and real input WITH fake_context
            # Raises an exception using a real model with fake_context enabled
            with self.assertRaises(torch.onnx.OnnxExporterError):
                export_options = ExportOptions(fake_context=fake_context)
                _ = torch.onnx.dynamo_export(
                    real_model, real_x, export_options=export_options
                )

            # Scenario 4: Fake model and real input WITH fake_context
            # Raises an exception using a fake model with fake_context enabled
            with self.assertRaises(torch.onnx.OnnxExporterError):
                export_options = ExportOptions(fake_context=fake_context)
                _ = torch.onnx.dynamo_export(
                    fake_model, real_x, export_options=export_options
                )

    @pytorch_test_common.xfail(
        error_message="Dynamic control flow is not supported at the moment."
    )
    # 定义测试函数，用于测试 Huggingface 的 LLAMA 模型的假张量模式
    def test_fake_tensor_mode_huggingface_llama(self):
        # 创建 LLAMA 模型的配置对象，设置词汇大小、隐藏层大小、隐藏层数和注意力头数
        config = transformers.LlamaConfig(
            vocab_size=8096, hidden_size=256, num_hidden_layers=2, num_attention_heads=2
        )
        # 定义批量大小和序列长度
        batch, seq = 4, 256

        # 开启假张量模式的上下文
        with torch.onnx.enable_fake_mode() as fake_context:
            # 创建 LLAMA 模型，并设置为评估模式
            model = transformers.LlamaModel(config).eval()
            # 生成随机整数张量作为输入标识符
            input_ids = torch.randint(0, config.vocab_size, (batch, seq))
            # 创建全为 True 的注意力掩码张量
            attention_mask = torch.ones(batch, seq, dtype=torch.bool)
            # 创建位置标识符张量
            position_ids = torch.arange(0, seq, dtype=torch.long)
            position_ids = position_ids.unsqueeze(0).view(-1, seq)

            # 创建导出选项，包括假张量模式的上下文
            export_options = torch.onnx.ExportOptions(fake_context=fake_context)
            # 导出模型到 ONNX 程序
            onnx_program = torch.onnx.dynamo_export(
                model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                export_options=export_options,
            )
            # 检查导出的 ONNX 模型
            onnx.checker.check_model(onnx_program.model_proto)
            # 推断 ONNX 模型的形状信息
            onnx.shape_inference.infer_shapes(onnx_program.model_proto)

    # 标记为预期失败的测试函数，用于测试 Huggingface 的 TIIUAE FALCON 模型的假张量模式
    @pytorch_test_common.xfail(
        error_message="Dynamic control flow is not supported at the moment."
    )
    def test_fake_tensor_mode_huggingface_tiiuae_falcon(self):
        # 创建 FALCON 模型的配置对象
        config = transformers.FalconConfig()
        # 定义批量大小和序列长度
        batch, seq = 4, 256

        # 开启假张量模式的上下文
        with torch.onnx.enable_fake_mode() as fake_context:
            # 创建 FALCON 模型，并设置为评估模式
            model = transformers.FalconModel(config).eval()
            # 生成随机整数张量作为输入标识符
            input_ids = torch.randint(0, config.vocab_size, (batch, seq))
            # 创建全为 True 的注意力掩码张量

            attention_mask = torch.ones(batch, seq, dtype=torch.bool)

            # 创建导出选项，包括假张量模式的上下文
            export_options = torch.onnx.ExportOptions(fake_context=fake_context)
            # 导出模型到 ONNX 程序
            onnx_program = torch.onnx.dynamo_export(
                model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                export_options=export_options,
            )
            # 检查导出的 ONNX 模型
            onnx.checker.check_model(onnx_program.model_proto)
            # 推断 ONNX 模型的形状信息
            onnx.shape_inference.infer_shapes(onnx_program.model_proto)
    def test_exported_program_input_with_custom_fx_tracer(self):
        # 导入必要的库和模块
        from torch.onnx._internal import exporter
        from torch.onnx._internal.fx import dynamo_graph_extractor
        
        # 定义一个简单的模型类，实现 forward 方法
        class Model(torch.nn.Module):
            def forward(self, x):
                return x + 1
        
        # 创建一个随机输入张量
        x = torch.randn(1, 1, 2)
        
        # 导出模型的计算图
        exported_program = torch.export.export(Model(), args=(x,))
        
        # 设置导出选项
        export_options = torch.onnx.ExportOptions()
        export_options = exporter.ResolvedExportOptions(
            export_options, model=exported_program
        )
        
        # 设置自定义的 FX 追踪器
        export_options.fx_tracer = (
            dynamo_graph_extractor.DynamoExport()
        )  # Override fx_tracer to an unsupported tracer
        
        # 使用断言捕获预期的异常
        with self.assertRaises(torch.onnx.OnnxExporterError):
            # 进行动态导出
            onnx_program = torch.onnx.dynamo_export(
                exported_program,
                x,
                export_options=export_options,
            )
            # 断言导出过程中发生异常
            self.assertTrue(onnx_program._export_exception is not None)
            # 使用断言捕获预期的异常
            with self.assertRaises(torch.onnx.InvalidExportOptionsError):
                raise self._export_exception

    def test_exported_program_torch_distributions_normal_Normal(self):
        # 定义一个简单的模型类，使用 torch.distributions.normal.Normal
        class Model(torch.nn.Module):
            def __init__(self):
                self.normal = torch.distributions.normal.Normal(0, 1)
                super().__init__()

            def forward(self, x):
                return self.normal.sample(x.shape)
        
        # 创建一个随机输入张量
        x = torch.randn(2, 3)
        
        # 使用 torch.no_grad 上下文管理器
        with torch.no_grad():
            # 导出模型的计算图
            exported_program = torch.export.export(Model(), args=(x,))
            # 调用动态导出函数
            _ = torch.onnx.dynamo_export(
                exported_program,
                x,
            )

    def test_aten_div_no_opmath_type_promotion(self):
        # 定义一个简单的模型类，实现 forward 方法
        class Model(torch.nn.Module):
            def forward(self, input):
                return input / 2
        
        # 创建模型实例
        model = Model()
        
        # 创建一个随机输入张量，指定 requires_grad=True 和 dtype=torch.float16
        input = torch.randn(3, 5, requires_grad=True, dtype=torch.float16)
        
        # 使用动态导出函数获取模型的 protobuf 对象
        model_proto = torch.onnx.dynamo_export(model, input).model_proto
        
        # 调用 onnx.inliner.inline_local_functions 函数，内联局部函数
        model_proto = onnx.inliner.inline_local_functions(model_proto)
        
        # 获取模型计算图中的除法节点
        div_node = next(
            node for node in model_proto.graph.node if node.op_type == "Div"
        )
        
        # 断言 Div 节点的输入应该是模型的输入，中间不应该有 Cast 节点
        # 对模型的输入进行断言
        self.assertEqual(div_node.input[0], model_proto.graph.input[0].name)

    def test_exported_program_as_input_with_model_signature(self):
        # 定义一个简单的模型类，实现 forward 方法
        class Model(torch.nn.Module):
            def forward(self, x):
                return x + 1.0
        
        # 创建一个随机输入张量，指定 dtype=torch.float
        x = torch.randn(1, 1, 2, dtype=torch.float)
        
        # 导出模型的计算图
        exported_program = torch.export.export(Model(), args=(x,))
        
        # 调用动态导出函数
        onnx_program = torch.onnx.dynamo_export(
            exported_program,
            x,
        )
        
        # 断言导出的模型签名应为 ExportGraphSignature
        self.assertTrue(onnx_program.model_signature, torch.export.ExportGraphSignature)
    @common_utils.parametrize(
        "float8_type",
        [
            # 使用 common_utils.subtest 创建一个参数化测试，使用 torch.float8_e5m2 作为参数，命名为 "torch_float8_e5m2"
            common_utils.subtest(
                torch.float8_e5m2,
                name="torch_float8_e5m2",
            ),
            # 使用 common_utils.subtest 创建一个参数化测试，使用 torch.float8_e5m2fnuz 作为参数，命名为 "torch_float8_e5m2fnuz"
            common_utils.subtest(
                torch.float8_e5m2fnuz,
                name="torch_float8_e5m2fnuz",
            ),
            # 使用 common_utils.subtest 创建一个参数化测试，使用 torch.float8_e4m3fn 作为参数，命名为 "torch_float8_e4m3fn"
            common_utils.subtest(
                torch.float8_e4m3fn,
                name="torch_float8_e4m3fn",
            ),
            # 使用 common_utils.subtest 创建一个参数化测试，使用 torch.float8_e4m3fnuz 作为参数，命名为 "torch_float8_e4m3fnuz"
            common_utils.subtest(
                torch.float8_e4m3fnuz,
                name="torch_float8_e4m3fnuz",
            ),
        ],
    )
    # 定义一个测试方法，测试 float8 支持
    def test_float8_support(self, float8_type):
        # 定义一个名为 Float8Module 的内部类，继承自 torch.nn.Module
        class Float8Module(torch.nn.Module):
            # 实现 Module 的 forward 方法
            def forward(self, input: torch.Tensor):
                # 将输入张量转换为指定的 float8_type 类型
                input = input.to(float8_type)
                # 返回加上一个 float8_type 类型常量 1.0 的结果
                return input + torch.tensor(1.0, dtype=float8_type)

        # 在使用 Optimizer 过程中可能由于不支持的数据类型而引发形状推断错误的警告
        with self.assertWarnsOnceRegex(
            UserWarning, "ONNXScript optimizer failed. Skipping optimization."
        ):
            # 尝试导出 Float8Module 模型的 ONNX 表示
            _ = torch.onnx.dynamo_export(Float8Module(), torch.randn(1, 2, 3, 4))

    # 定义一个测试方法，测试在带有 logging 的 logger 下导出模型
    def test_export_with_logging_logger(self):
        # 获取当前模块的 logger 对象
        logger = logging.getLogger(__name__)

        # 定义一个名为 LoggingLoggerModule 的内部类，继承自 torch.nn.Module
        class LoggingLoggerModule(torch.nn.Module):
            # 实现 Module 的 forward 方法
            def forward(self, x):
                # 记录日志消息 "abc"
                logger.log("abc")
                return x + 1

        # 创建一个随机输入张量
        input = torch.randn(2, 3)
        # 创建 LoggingLoggerModule 实例
        model = LoggingLoggerModule()
        # 尝试导出 LoggingLoggerModule 模型的 ONNX 表示
        _ = torch.onnx.dynamo_export(model, input)

    # 定义一个测试方法，测试在带有 HF logging 的 logger 下导出模型
    def test_export_with_hf_logging_logger(self):
        # 获取 transformers 库中 logging 模块的 logger 对象
        logger = transformers.utils.logging.get_logger(__name__)

        # 定义一个名为 HFLoggingLoggerModule 的内部类，继承自 torch.nn.Module
        class HFLoggingLoggerModule(torch.nn.Module):
            # 实现 Module 的 forward 方法
            def forward(self, x):
                # 记录警告消息 "abc"，但只记录一次
                logger.warning_once("abc")
                return x + 1

        # 创建一个随机输入张量
        input = torch.randn(2, 3)
        # 创建 HFLoggingLoggerModule 实例
        model = HFLoggingLoggerModule()
        # 尝试导出 HFLoggingLoggerModule 模型的 ONNX 表示
        _ = torch.onnx.dynamo_export(model, input)
    def test_checkpoint_cast(self):
        # 设定模型标识符
        model_id = "openai/whisper-large-v3"
        # 创建特征提取器对象，设定特征大小为128
        feature_extractor = transformers.WhisperFeatureExtractor(feature_size=128)
        # 设定批处理大小为4
        batch = 4

        # 使用 torch.onnx.enable_fake_mode() 上下文管理器，开启伪造模式
        with torch.onnx.enable_fake_mode() as ctx:
            # 从预训练模型加载自动语音序列到序列模型，关闭低CPU内存使用和安全张量
            model = transformers.AutoModelForSpeechSeq2Seq.from_pretrained(
                model_id, low_cpu_mem_usage=False, use_safetensors=False
            )
            # 创建输入数据字典
            input = {
                "input_features": torch.randn(
                    (
                        batch,
                        feature_extractor.feature_size,
                        feature_extractor.nb_max_frames,
                    )
                ),
                # 设定解码器输入的初始标记
                "decoder_input_ids": torch.tensor([[1, 1]]) * 8001,
                # 设定不返回字典
                "return_dict": False,
            }

        # 设定导出选项，使用前面创建的伪造上下文
        export_options = torch.onnx.ExportOptions(fake_context=ctx)
        # 使用 torch.onnx.dynamo_export 导出模型到 ONNX 程序
        onnx_program = torch.onnx.dynamo_export(
            model, **input, export_options=export_options
        )
        # 使用临时文件保存导出的 ONNX 程序，并进行模型检查
        with tempfile.NamedTemporaryFile(suffix=".onnx") as tmp_onnx_file:
            onnx_program.save(tmp_onnx_file.name)
            onnx.checker.check_model(tmp_onnx_file.name, full_check=True)

    @common_utils.parametrize(
        "include_initializer",
        [
            common_utils.subtest(
                True,
                name="include_initializer",
            ),
            common_utils.subtest(
                False,
                name="dont_include_initializer",
            ),
        ],
    )
    @common_utils.parametrize(
        "use_fake_mode",
        [
            common_utils.subtest(
                True,
                name="use_fake_mode",
            ),
            common_utils.subtest(
                False,
                name="no_fake_mode",
            ),
        ],
    )
    @common_utils.parametrize(
        "use_exported_program",
        [
            common_utils.subtest(
                True,
                name="use_exported_program",
            ),
            common_utils.subtest(
                False,
                name="no_exported_program",
            ),
        ],
    )
    # 测试函数，参数化测试保存时包含或不包含初始设定值、使用或不使用伪造模式、使用或不使用导出的程序
    def test_save_with_without_initializer(
        self, include_initializer, use_fake_mode, use_exported_program
        ):
            # 定义一个名为 MNISTModel 的神经网络模型类，继承自 nn.Module
            class MNISTModel(nn.Module):
                # 初始化函数，定义神经网络的各个层和参数
                def __init__(self):
                    super().__init__()
                    self.conv1 = nn.Conv2d(1, 32, 3, 1, bias=False)  # 第一个卷积层，输入通道数为1，输出通道数为32，卷积核大小为3x3，步长为1，无偏置
                    self.conv2 = nn.Conv2d(32, 64, 3, 1, bias=False)  # 第二个卷积层，输入通道数为32，输出通道数为64，卷积核大小为3x3，步长为1，无偏置
                    self.fc1 = nn.Linear(9216, 128, bias=False)  # 第一个全连接层，输入特征数为9216，输出特征数为128，无偏置
                    self.fc2 = nn.Linear(128, 10, bias=False)  # 第二个全连接层，输入特征数为128，输出特征数为10（分类数），无偏置

                # 前向传播函数，定义神经网络的数据流向
                def forward(self, tensor_x: torch.Tensor):
                    tensor_x = self.conv1(tensor_x)  # 第一层卷积
                    tensor_x = F.sigmoid(tensor_x)  # 使用 sigmoid 激活函数
                    tensor_x = self.conv2(tensor_x)  # 第二层卷积
                    tensor_x = F.sigmoid(tensor_x)  # 使用 sigmoid 激活函数
                    tensor_x = F.max_pool2d(tensor_x, 2)  # 最大池化操作，池化核大小为2x2
                    tensor_x = torch.flatten(tensor_x, 1)  # 将多维张量展平成一维张量
                    tensor_x = self.fc1(tensor_x)  # 第一层全连接
                    tensor_x = F.sigmoid(tensor_x)  # 使用 sigmoid 激活函数
                    tensor_x = self.fc2(tensor_x)  # 第二层全连接
                    output = F.log_softmax(tensor_x, dim=1)  # 输出层使用 log_softmax 函数进行分类
                    return output

            # 获取 MNISTModel 的初始状态字典
            state_dict = MNISTModel().state_dict()
            # 如果使用 fake mode
            if use_fake_mode:
                # 启用 fake mode
                with torch.onnx.enable_fake_mode() as ctx:
                    model = MNISTModel()  # 创建 MNISTModel 实例
                    tensor_x = torch.rand((64, 1, 28, 28), dtype=torch.float32)  # 创建随机张量作为输入数据
                    # 如果使用导出的程序
                    if use_exported_program:
                        model = torch.export.export(model, args=(tensor_x,))
                    # 设置导出选项，使用 fake context
                    export_options = torch.onnx.ExportOptions(fake_context=ctx)
            else:
                model = MNISTModel()  # 创建 MNISTModel 实例
                tensor_x = torch.rand((64, 1, 28, 28), dtype=torch.float32)  # 创建随机张量作为输入数据
                # 如果使用导出的程序
                if use_exported_program:
                    model = torch.export.export(model, args=(tensor_x,))
                # 设置默认导出选项
                export_options = torch.onnx.ExportOptions()

            # 使用 torch.onnx.dynamo_export 导出 ONNX 程序
            onnx_program = torch.onnx.dynamo_export(
                model, tensor_x, export_options=export_options
            )
            # 使用临时文件保存导出的 ONNX 程序
            with tempfile.NamedTemporaryFile(suffix=".onnx") as tmp_onnx_file:
                # 将 ONNX 程序保存到临时文件中
                onnx_program.save(
                    tmp_onnx_file.name,
                    include_initializers=include_initializer,
                    model_state=state_dict if include_initializer else None,
                )
                # 加载保存的 ONNX 模型
                onnx_model = onnx.load(tmp_onnx_file.name)
                # 断言检查初始值是否包含在模型图中
                self.assertEqual(
                    (include_initializer and len(onnx_model.graph.initializer) > 0)
                    or (not include_initializer and len(onnx_model.graph.initializer) == 0),
                    True,
                )

    # 定义测试导出包含打印语句的模型
    def test_export_with_print(self):
        # 定义一个简单的打印模块类 PrintModule，继承自 torch.nn.Module
        class PrintModule(torch.nn.Module):
            # 前向传播函数，打印字符串 "abc"，并返回输入数据加一的结果
            def forward(self, x):
                print("abc")
                return x + 1

        input = torch.randn(2, 3)  # 创建一个2x3的随机张量作为输入数据
        model = PrintModule()  # 创建 PrintModule 的实例
        _ = torch.onnx.dynamo_export(model, input)  # 导出模型到 ONNX 格式
# 如果当前脚本被直接运行（而不是被导入到其他模块中执行），则执行以下代码块
if __name__ == "__main__":
    # 调用common_utils模块中的run_tests函数，用于执行测试代码或单元测试
    common_utils.run_tests()
```