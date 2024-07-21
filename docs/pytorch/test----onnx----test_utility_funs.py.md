# `.\pytorch\test\onnx\test_utility_funs.py`

```py
# Owner(s): ["module: onnx"]

# 导入必要的模块和库
import copy  # 导入复制操作相关的模块
import functools  # 导入函数工具相关的模块
import io  # 导入输入输出流相关的模块
import re  # 导入正则表达式相关的模块
import warnings  # 导入警告处理相关的模块
from typing import Callable  # 从 typing 模块中导入 Callable 类型注解

# 导入第三方库和自定义模块
import onnx  # 导入 ONNX 模块
import parameterized  # 导入参数化测试相关的模块
import pytorch_test_common  # 导入 PyTorch 测试公共模块
import torchvision  # 导入 torchvision 库
from autograd_helper import CustomFunction as CustomFunction2  # 导入自动求导辅助函数
from pytorch_test_common import (  # 从 PyTorch 测试公共模块导入以下函数
    skipIfNoCuda,
    skipIfUnsupportedMaxOpsetVersion,
    skipIfUnsupportedMinOpsetVersion,
)

import torch  # 导入 PyTorch 库
import torch.onnx  # 导入 PyTorch 的 ONNX 模块
import torch.utils.cpp_extension  # 导入 PyTorch 的 C++ 扩展工具模块
from torch.onnx import (  # 从 PyTorch 的 ONNX 模块中导入以下符号
    _constants,
    OperatorExportTypes,
    TrainingMode,
    utils,
)
from torch.onnx._globals import GLOBALS  # 导入 PyTorch ONNX 的全局变量
from torch.onnx.symbolic_helper import _unpack_list, parse_args  # 导入符号化助手函数
from torch.testing._internal import common_utils  # 导入 PyTorch 内部测试的公共工具模块
from torch.testing._internal.common_utils import skipIfNoLapack  # 导入测试工具模块的 Lapack 缺失跳过函数

def _remove_test_environment_prefix_from_scope_name(scope_name: str) -> str:
    """Remove test environment prefix added to module.

    Remove prefix to normalize scope names, since different test environments add
    prefixes with slight differences.

    Example:

        >>> _remove_test_environment_prefix_from_scope_name(
        >>>     "test_utility_funs.M"
        >>> )
        "M"
        >>> _remove_test_environment_prefix_from_scope_name(
        >>>     "test_utility_funs.test_abc.<locals>.M"
        >>> )
        "M"
        >>> _remove_test_environment_prefix_from_scope_name(
        >>>     "__main__.M"
        >>> )
        "M"
    """
    prefixes_to_remove = ["test_utility_funs", "__main__"]  # 待移除的前缀列表
    for prefix in prefixes_to_remove:
        scope_name = re.sub(f"{prefix}\\.(.*?<locals>\\.)?", "", scope_name)  # 使用正则表达式替换前缀
    return scope_name  # 返回处理后的模块名称


class _BaseTestCase(pytorch_test_common.ExportTestCase):
    """Base test case for PyTorch export testing."""

    def _model_to_graph(
        self,
        model,
        input,
        do_constant_folding=True,
        training=TrainingMode.EVAL,
        operator_export_type=OperatorExportTypes.ONNX,
        input_names=None,
        dynamic_axes=None,
    ):
        """Convert PyTorch model to ONNX graph.

        Args:
            model: PyTorch model to convert.
            input: Input data for the model.
            do_constant_folding: Whether to fold constants in the model graph.
            training: Training mode ('TRAINING' or 'EVAL').
            operator_export_type: Type of operator export for ONNX.
            input_names: Names of input nodes in the ONNX graph.
            dynamic_axes: Dynamic axes specification.

        Returns:
            ONNX graph, parameters dictionary, and output from Torch.
        """
        torch.onnx.utils._setup_trace_module_map(model, False)  # 设置追踪模块映射
        if training == torch.onnx.TrainingMode.TRAINING:  # 如果是训练模式
            model.train()  # 设置模型为训练模式
        elif training == torch.onnx.TrainingMode.EVAL:  # 如果是评估模式
            model.eval()  # 设置模型为评估模式
        utils._validate_dynamic_axes(dynamic_axes, model, None, None)  # 验证动态轴
        graph, params_dict, torch_out = utils._model_to_graph(
            model,
            input,
            do_constant_folding=do_constant_folding,
            _disable_torch_constant_prop=True,
            operator_export_type=operator_export_type,
            training=training,
            input_names=input_names,
            dynamic_axes=dynamic_axes,
        )  # 将模型转换为 ONNX 图
        return graph, params_dict, torch_out  # 返回 ONNX 图、参数字典和 Torch 输出


@common_utils.instantiate_parametrized_tests
class TestUnconvertibleOps(pytorch_test_common.ExportTestCase):
    """Unit tests for the `unconvertible_ops` function."""

    def setUp(self):
        """Set up for each test case."""
        class EinsumModule(torch.nn.Module):
            """Example PyTorch module using einsum."""

            def forward(self, x):
                return torch.einsum("ii", x)  # 使用 einsum 运算

        self.einsum_module = EinsumModule()  # 实例化 EinsumModule 类作为测试用例
    def test_it_returns_graph_and_unconvertible_ops_at_lower_opset_version(self):
        x = torch.randn(4, 4)

        # Einsum is supported since opset 12. It should be unconvertible at opset 9.
        # 使用 utils 模块中的 unconvertible_ops 函数检查 einsum_module 在 opset_version=9 下的不可转换操作
        graph, unconvertible_ops = utils.unconvertible_ops(
            self.einsum_module, (x,), opset_version=9
        )
        nodes = graph.nodes()
        # 断言各个节点的类型，确保输出的图形结构符合预期
        self.assertEqual(next(nodes).kind(), "prim::Constant")
        self.assertEqual(next(nodes).kind(), "prim::ListConstruct")
        self.assertEqual(next(nodes).kind(), "prim::Constant")
        self.assertEqual(next(nodes).kind(), "aten::einsum")
        # 断言 unconvertible_ops 中包含 "aten::einsum"
        self.assertEqual(unconvertible_ops, ["aten::einsum"])

    @common_utils.parametrize(
        "jit_function",
        [
            common_utils.subtest(
                functools.partial(torch.jit.trace, example_inputs=torch.randn(4, 4)),
                name="traced",
            ),
            common_utils.subtest(torch.jit.script, name="scripted"),
        ],
    )
    def test_it_returns_unconvertible_ops_at_lower_opset_version_for_jit_module(
        self, jit_function: Callable
    ):
        module = jit_function(self.einsum_module)
        x = torch.randn(4, 4)

        # Einsum is supported since opset 12. It should be unconvertible at opset 9.
        # 使用 utils 模块中的 unconvertible_ops 函数检查 jit 编译的模块在 opset_version=9 下的不可转换操作
        _, unconvertible_ops = utils.unconvertible_ops(module, (x,), opset_version=9)
        # 断言 unconvertible_ops 中包含 "aten::einsum"
        self.assertEqual(unconvertible_ops, ["aten::einsum"])

    @common_utils.parametrize(
        "jit_function",
        [
            common_utils.subtest(lambda x: x, name="nn_module"),
            common_utils.subtest(
                functools.partial(torch.jit.trace, example_inputs=torch.randn(4, 4)),
                name="traced",
            ),
            common_utils.subtest(torch.jit.script, name="scripted"),
        ],
    )
    def test_it_returns_empty_list_when_all_ops_convertible(
        self, jit_function: Callable
    ):
        module = jit_function(self.einsum_module)
        x = torch.randn(4, 4)

        # Einsum is supported since opset 12
        # 使用 utils 模块中的 unconvertible_ops 函数检查 jit 编译的模块在 opset_version=12 下的不可转换操作
        _, unconvertible_ops = utils.unconvertible_ops(module, (x,), opset_version=12)
        # 断言 unconvertible_ops 为空列表，即所有操作均可转换
        self.assertEqual(unconvertible_ops, [])

    def test_it_returns_empty_list_when_model_contains_supported_inplace_ops(self):
        class SkipConnectionModule(torch.nn.Module):
            def forward(self, x):
                out = x
                out += x
                out = torch.nn.functional.relu(out, inplace=True)
                return out

        module = SkipConnectionModule()
        x = torch.randn(4, 4)
        # 使用 utils 模块中的 unconvertible_ops 函数检查 SkipConnectionModule 类实例在 opset_version=13 下的不可转换操作
        _, unconvertible_ops = utils.unconvertible_ops(module, (x,), opset_version=13)
        # 断言 unconvertible_ops 为空列表，即所有操作均可转换
        self.assertEqual(unconvertible_ops, [])
# 使用 parameterized 库来生成参数化测试类，每个类的 opset_version 参数从 ONNX_BASE_OPSET 到 ONNX_TORCHSCRIPT_EXPORTER_MAX_OPSET + 1 变化
@parameterized.parameterized_class(
    [
        {"opset_version": opset}
        for opset in range(
            _constants.ONNX_BASE_OPSET,
            _constants.ONNX_TORCHSCRIPT_EXPORTER_MAX_OPSET + 1,
        )
    ],
    # 定义测试类的名称函数，结合 opset_version 参数
    class_name_func=lambda cls, num, params_dict: f"{cls.__name__}_opset_{params_dict['opset_version']}",
)
# 继承自 _BaseTestCase 的测试类 TestUtilityFuns
class TestUtilityFuns(_BaseTestCase):
    # 类变量 opset_version 初始化为 None
    opset_version = None

    # 测试函数，验证是否处于 ONNX 导出状态
    def test_is_in_onnx_export(self):
        # 将当前的测试实例赋值给 test_self
        test_self = self

        # 定义一个继承自 torch.nn.Module 的类 MyModule
        class MyModule(torch.nn.Module):
            # 定义 forward 方法
            def forward(self, x):
                # 断言当前是否处于 ONNX 导出状态
                test_self.assertTrue(torch.onnx.is_in_onnx_export())
                # 抛出 ValueError 异常
                raise ValueError
                return x + 1

        # 创建一个 3x4 的随机张量 x
        x = torch.randn(3, 4)
        # 创建一个字节流对象 f
        f = io.BytesIO()
        try:
            # 尝试导出 MyModule 到 ONNX 格式
            torch.onnx.export(MyModule(), x, f, opset_version=self.opset_version)
        except ValueError:
            # 如果捕获到 ValueError 异常，则断言不处于 ONNX 导出状态
            self.assertFalse(torch.onnx.is_in_onnx_export())

    # 测试函数，验证动态轴的有效性
    def test_validate_dynamic_axes_invalid_input_output_name(self):
        # 使用警告捕获器捕获所有警告
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            # 调用 utils 模块的 _validate_dynamic_axes 函数
            utils._validate_dynamic_axes(
                {"input1": {}, "output": {}, "invalid_name1": {}, "invalid_name2": {}},
                None,
                ["input1", "input2"],
                ["output"],
            )
            # 将所有警告消息转换为字符串列表
            messages = [str(warning.message) for warning in w]
        # 断言特定的警告消息是否在消息列表中
        self.assertIn(
            "Provided key invalid_name1 for dynamic axes is not a valid input/output name",
            messages,
        )
        self.assertIn(
            "Provided key invalid_name2 for dynamic axes is not a valid input/output name",
            messages,
        )
        # 断言消息列表的长度为 2
        self.assertEqual(len(messages), 2)

    # 根据条件跳过不支持的最小 opset 版本进行测试
    @skipIfUnsupportedMinOpsetVersion(11)
    def test_split_to_slice(self):
        # 定义一个继承自 torch.nn.Module 的类 SplitModule
        class SplitModule(torch.nn.Module):
            # 定义 forward 方法
            def forward(self, x, y, t):
                # 定义分割点
                splits = (x.size(1), y.size(1))
                # 使用给定的分割点和维度对 t 进行切分
                out, out2 = torch.split(t, splits, dim=1)
                return out, out2

        # 设置全局变量的导出 opset 版本为当前的 opset_version
        GLOBALS.export_onnx_opset_version = self.opset_version
        # 设置全局变量的操作符导出类型为 ONNX
        GLOBALS.operator_export_type = OperatorExportTypes.ONNX
        # 创建随机张量 x, y, t
        x = torch.randn(2, 3)
        y = torch.randn(2, 4)
        t = torch.randn(2, 7)
        # 将 SplitModule 转换为图形表示
        graph, _, _ = self._model_to_graph(
            SplitModule(),
            (x, y, t),
            input_names=["x", "y", "t"],
            dynamic_axes={"x": [0, 1], "y": [0, 1], "t": [0, 1]},
        )
        # 遍历图中的每个节点
        for node in graph.nodes():
            # 断言节点的类型不是 "onnx::SplitToSequence"
            self.assertNotEqual(node.kind(), "onnx::SplitToSequence")
    def test_constant_fold_transpose(self):
        class TransposeModule(torch.nn.Module):
            def forward(self, x):
                a = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
                b = torch.transpose(a, 1, 0)  # 对张量 a 进行转置操作，交换维度 1 和 0
                return b + x  # 返回转置后的张量 b 与输入张量 x 的和

        GLOBALS.export_onnx_opset_version = self.opset_version
        GLOBALS.operator_export_type = OperatorExportTypes.ONNX
        x = torch.ones(3, 2)
        graph, _, __ = self._model_to_graph(
            TransposeModule(), (x,), input_names=["x"], dynamic_axes={"x": [0, 1]}
        )

        for node in graph.nodes():
            self.assertNotEqual(node.kind(), "onnx::Transpose")  # 断言图中不存在 "onnx::Transpose" 节点
            self.assertNotEqual(node.kind(), "onnx::Cast")  # 断言图中不存在 "onnx::Cast" 节点
        self.assertEqual(len(list(graph.nodes())), 2)  # 断言图中节点的数量为 2

    @skipIfUnsupportedMaxOpsetVersion(17)
    def test_constant_fold_reduceL2(self):
        class ReduceModule(torch.nn.Module):
            def forward(self, x):
                a = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
                b = torch.norm(a, p=2, dim=-2, keepdim=False)  # 计算张量 a 在最后一个维度上的 L2 范数
                return b + x  # 返回范数 b 与输入张量 x 的和

        GLOBALS.export_onnx_opset_version = self.opset_version
        GLOBALS.operator_export_type = OperatorExportTypes.ONNX
        x = torch.ones(2, 3)
        graph, _, __ = self._model_to_graph(
            ReduceModule(), (x,), input_names=["x"], dynamic_axes={"x": [0, 1]}
        )

        for node in graph.nodes():
            self.assertNotEqual(node.kind(), "onnx::ReduceL2")  # 断言图中不存在 "onnx::ReduceL2" 节点

    @skipIfUnsupportedMaxOpsetVersion(17)
    def test_constant_fold_reduceL1(self):
        class NormModule(torch.nn.Module):
            def forward(self, x):
                a = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
                b = torch.norm(a, p=1, dim=-2)  # 计算张量 a 在最后一个维度上的 L1 范数
                return b + x  # 返回范数 b 与输入张量 x 的和

        GLOBALS.export_onnx_opset_version = self.opset_version
        GLOBALS.operator_export_type = OperatorExportTypes.ONNX
        x = torch.ones(2, 3)
        graph, _, __ = self._model_to_graph(
            NormModule(), (x,), input_names=["x"], dynamic_axes={"x": [0, 1]}
        )

        for node in graph.nodes():
            self.assertNotEqual(node.kind(), "onnx::ReduceL1")  # 断言图中不存在 "onnx::ReduceL1" 节点

    def test_constant_fold_slice(self):
        class NarrowModule(torch.nn.Module):
            def forward(self, x):
                a = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
                b = torch.narrow(a, 0, 0, 1)  # 在张量 a 的第 0 维上进行切片操作，从索引 0 开始，长度为 1
                return b + x  # 返回切片 b 与输入张量 x 的和

        GLOBALS.export_onnx_opset_version = self.opset_version
        GLOBALS.operator_export_type = OperatorExportTypes.ONNX
        x = torch.ones(1, 3)
        graph, _, __ = self._model_to_graph(
            NarrowModule(), (x,), input_names=["x"], dynamic_axes={"x": [0, 1]}
        )

        for node in graph.nodes():
            self.assertNotEqual(node.kind(), "onnx::Slice")  # 断言图中不存在 "onnx::Slice" 节点
            self.assertNotEqual(node.kind(), "onnx::Cast")  # 断言图中不存在 "onnx::Cast" 节点
        self.assertEqual(len(list(graph.nodes())), 2)  # 断言图中节点的数量为 2
    def test_constant_fold_slice_index_exceeds_dim(self):
        class SliceIndexExceedsDimModule(torch.nn.Module):
            def forward(self, x):
                # 创建一个2x3的张量
                a = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
                # 使用超出维度范围的索引，尝试对张量进行切片操作
                b = a[1:10]  # index exceeds dimension
                return b + x

        # 设置全局变量
        GLOBALS.export_onnx_opset_version = self.opset_version
        GLOBALS.operator_export_type = OperatorExportTypes.ONNX
        x = torch.ones(1, 3)
        # 将模型转换为计算图
        graph, _, __ = self._model_to_graph(
            SliceIndexExceedsDimModule(),
            (x,),
            input_names=["x"],
            dynamic_axes={"x": [0, 1]},
        )

        # 检查计算图中的节点类型，确保没有Slice或Cast操作
        for node in graph.nodes():
            self.assertNotEqual(node.kind(), "onnx::Slice")
            self.assertNotEqual(node.kind(), "onnx::Cast")
        # 断言计算图中的节点数为2
        self.assertEqual(len(list(graph.nodes())), 2)

    def test_constant_fold_slice_negative_index(self):
        class SliceNegativeIndexModule(torch.nn.Module):
            def forward(self, x):
                # 创建一个2x3的张量
                a = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
                # 使用负数索引，相对于张量末尾进行切片操作
                b = a[0:-1]
                # 使用torch.select进行索引操作，选取倒数第二列
                c = torch.select(a, dim=-1, index=-2)
                # 使用torch.select进行索引操作，选取第一行
                d = torch.select(a, dim=1, index=0)
                return b + x, c + d

        # 设置全局变量
        GLOBALS.export_onnx_opset_version = self.opset_version
        GLOBALS.operator_export_type = OperatorExportTypes.ONNX
        x = torch.ones(1, 3)
        # 将模型转换为计算图
        graph, _, __ = self._model_to_graph(
            SliceNegativeIndexModule(),
            (x,),
            input_names=["x"],
            dynamic_axes={"x": [0, 1]},
        )

        # 检查计算图中的节点类型，确保没有Slice或Cast操作
        for node in graph.nodes():
            self.assertNotEqual(node.kind(), "onnx::Slice")
            self.assertNotEqual(node.kind(), "onnx::Cast")

    def test_constant_fold_gather(self):
        class GatherModule(torch.nn.Module):
            def forward(self, x):
                # 创建一个2x3的张量
                a = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
                # 使用torch.select进行gather操作，选取倒数第二列
                b = torch.select(a, dim=1, index=-2)
                # 使用torch.index_select进行gather操作，选取第0和第1行
                c = torch.index_select(a, dim=-2, index=torch.tensor([0, 1]))
                return b + 1, c + x

        # 设置全局变量
        GLOBALS.export_onnx_opset_version = self.opset_version
        GLOBALS.operator_export_type = OperatorExportTypes.ONNX
        x = torch.ones(1, 3)
        model = GatherModule()
        model(x)
        # 将模型转换为计算图
        graph, _, __ = self._model_to_graph(
            GatherModule(), (x,), input_names=["x"], dynamic_axes={"x": [0, 1]}
        )

        # 检查计算图中的节点类型，确保没有Gather操作
        for node in graph.nodes():
            self.assertNotEqual(node.kind(), "onnx::Gather")
    def test_constant_fold_unsqueeze(self):
        # 定义一个测试函数，用于测试常量折叠与unsqueeze操作
        class UnsqueezeModule(torch.nn.Module):
            def forward(self, x):
                # 定义一个张量a
                a = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
                # 对张量a进行unsqueeze操作，将维度扩展
                b = torch.unsqueeze(a, -2)
                return b + x

        # 设置全局变量
        GLOBALS.export_onnx_opset_version = self.opset_version
        GLOBALS.operator_export_type = OperatorExportTypes.ONNX
        # 创建一个形状为(1, 2, 3)的全1张量x
        x = torch.ones(1, 2, 3)
        # 将UnsqueezeModule模型转换为计算图graph，并传入输入x
        graph, _, __ = self._model_to_graph(
            UnsqueezeModule(), (x,), input_names=["x"], dynamic_axes={"x": [0, 1, 2]}
        )

        # 遍历计算图的每个节点
        for node in graph.nodes():
            # 断言计算图中不存在"onnx::Unsqueeze"和"onnx::Cast"节点
            self.assertNotEqual(node.kind(), "onnx::Unsqueeze")
            self.assertNotEqual(node.kind(), "onnx::Cast")
        # 断言计算图节点数为2
        self.assertEqual(len(list(graph.nodes())), 2)

    def test_constant_fold_unsqueeze_multi_axies(self):
        # 定义一个测试函数，用于测试多维度情况下的unsqueeze操作
        class PReluModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.prelu = torch.nn.PReLU()

            def forward(self, x):
                # 定义一个形状为(2, 3, 4, 5, 8, 7)的随机张量a
                a = torch.randn(2, 3, 4, 5, 8, 7)
                return self.prelu(x) + a

        # 设置全局变量
        GLOBALS.export_onnx_opset_version = self.opset_version
        GLOBALS.operator_export_type = OperatorExportTypes.ONNX
        # 创建一个形状为(2, 3, 4, 5, 8, 7)的随机张量x
        x = torch.randn(2, 3, 4, 5, 8, 7)
        # 将PReluModel模型转换为计算图graph，并传入输入x
        graph, _, __ = self._model_to_graph(
            PReluModel(), x, input_names=["x"], dynamic_axes={"x": [0, 1, 2, 3, 4, 5]}
        )

        # 遍历计算图的每个节点
        for node in graph.nodes():
            # 断言计算图中不存在"onnx::Unsqueeze"和"onnx::Cast"节点
            self.assertNotEqual(node.kind(), "onnx::Unsqueeze")
            self.assertNotEqual(node.kind(), "onnx::Cast")
        # 断言计算图节点数为5
        self.assertEqual(len(list(graph.nodes())), 5)

    def test_constant_fold_squeeze_without_axes(self):
        # 定义一个测试函数，用于测试无轴压缩操作的常量折叠
        class SqueezeModule(torch.nn.Module):
            def forward(self, x):
                # 定义一个形状为(1, 2, 3)的张量a
                a = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])
                return torch.squeeze(a) + x + torch.squeeze(a)

        # 设置全局变量
        GLOBALS.export_onnx_opset_version = self.opset_version
        GLOBALS.operator_export_type = OperatorExportTypes.ONNX
        # 创建一个形状为(2, 3)的全1张量x
        x = torch.ones(2, 3)
        # 将SqueezeModule模型转换为计算图graph，并传入输入x
        graph, _, __ = self._model_to_graph(
            SqueezeModule(), (x,), input_names=["x"], dynamic_axes={"x": [0, 1]}
        )
        # 遍历计算图的每个节点
        for node in graph.nodes():
            # 断言计算图中不存在"onnx::Squeeze"和"onnx::Cast"节点
            self.assertNotEqual(node.kind(), "onnx::Squeeze")
            self.assertNotEqual(node.kind(), "onnx::Cast")
        # 断言计算图节点数为4
        self.assertEqual(len(list(graph.nodes())), 4)
    def test_constant_fold_squeeze_with_axes(self):
        # 定义一个名为 test_constant_fold_squeeze_with_axes 的测试方法
        class SqueezeAxesModule(torch.nn.Module):
            # 定义一个继承自 torch.nn.Module 的子类 SqueezeAxesModule
            def forward(self, x):
                # 实现前向传播方法，输入参数 x
                a = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])
                # 创建一个形状为 (1, 2, 3) 的张量 a
                return torch.squeeze(a, dim=-3) + x
                # 对张量 a 进行压缩操作，并与输入 x 相加得到输出

        GLOBALS.export_onnx_opset_version = self.opset_version
        # 设置全局变量 export_onnx_opset_version 为当前对象的 opset_version 属性值
        GLOBALS.operator_export_type = OperatorExportTypes.ONNX
        # 设置全局变量 operator_export_type 为 ONNX
        x = torch.ones(2, 3)
        # 创建一个形状为 (2, 3) 的张量 x，元素全为 1
        graph, _, __ = self._model_to_graph(
            SqueezeAxesModule(), (x,), input_names=["x"], dynamic_axes={"x": [0, 1]}
        )
        # 将 SqueezeAxesModule 实例化并转换为图形表示，传入参数 x，并指定输入名称和动态轴

        for node in graph.nodes():
            # 遍历图中的节点
            self.assertNotEqual(node.kind(), "onnx::Squeeze")
            # 断言节点的类型不是 "onnx::Squeeze"
            self.assertNotEqual(node.kind(), "onnx::Cast")
            # 断言节点的类型不是 "onnx::Cast"
        self.assertEqual(len(list(graph.nodes())), 2)
        # 断言图中节点的数量为 2

    def test_constant_fold_concat(self):
        # 定义一个名为 test_constant_fold_concat 的测试方法
        class ConcatModule(torch.nn.Module):
            # 定义一个继承自 torch.nn.Module 的子类 ConcatModule
            def forward(self, x):
                # 实现前向传播方法，输入参数 x
                # 以下是关于 ONNX 常量折叠的注释和解释
                a = torch.tensor([[1.0, 2.0, 3.0]]).to(torch.float)
                # 创建一个形状为 (1, 3) 的浮点型张量 a
                b = torch.tensor([[4.0, 5.0, 6.0]]).to(torch.float)
                # 创建一个形状为 (1, 3) 的浮点型张量 b
                c = torch.cat((a, b), 0)
                # 沿着第 0 维度拼接张量 a 和 b 得到张量 c
                d = b + c
                # 张量 d 等于张量 b 与张量 c 的元素级加法
                return x + d
                # 返回输入 x 与张量 d 的元素级加法结果

        GLOBALS.export_onnx_opset_version = self.opset_version
        # 设置全局变量 export_onnx_opset_version 为当前对象的 opset_version 属性值
        GLOBALS.operator_export_type = OperatorExportTypes.ONNX
        # 设置全局变量 operator_export_type 为 ONNX
        x = torch.ones(2, 3)
        # 创建一个形状为 (2, 3) 的张量 x，元素全为 1
        graph, _, __ = self._model_to_graph(
            ConcatModule(), (x,), input_names=["x"], dynamic_axes={"x": [0, 1]}
        )
        # 将 ConcatModule 实例化并转换为图形表示，传入参数 x，并指定输入名称和动态轴

        for node in graph.nodes():
            # 遍历图中的节点
            self.assertNotEqual(node.kind(), "onnx::Concat")
            # 断言节点的类型不是 "onnx::Concat"
            self.assertNotEqual(node.kind(), "onnx::Cast")
            # 断言节点的类型不是 "onnx::Cast"
        self.assertEqual(len(list(graph.nodes())), 2)
        # 断言图中节点的数量为 2
    def test_constant_fold_lstm(self):
        # 定义一个名为 GruNet 的类，继承自 torch.nn.Module
        class GruNet(torch.nn.Module):
            # 初始化方法，定义了一个名为 mygru 的 GRU 模型
            def __init__(self):
                super().__init__()
                self.mygru = torch.nn.GRU(7, 3, 1, bidirectional=False)

            # 前向传播方法，接受 input 和 initial_state 两个参数
            def forward(self, input, initial_state):
                return self.mygru(input, initial_state)

        # 设置全局变量，导出 ONNX 操作集版本号
        GLOBALS.export_onnx_opset_version = self.opset_version
        # 设置全局变量，导出操作类型为 ONNX
        GLOBALS.operator_export_type = OperatorExportTypes.ONNX
        # 创建一个形状为 (5, 3, 7) 的随机张量 input
        input = torch.randn(5, 3, 7)
        # 创建一个形状为 (1, 3, 3) 的随机张量 h0
        h0 = torch.randn(1, 3, 3)
        # 将 GruNet 实例化为 model，将 input 和 h0 作为参数传入 _model_to_graph 方法
        # 设置 input 和 h0 的动态轴
        graph, _, __ = self._model_to_graph(
            GruNet(),
            (input, h0),
            input_names=["input", "h0"],
            dynamic_axes={"input": [0, 1, 2], "h0": [0, 1, 2]},
        )

        # 遍历图中的每个节点
        for node in graph.nodes():
            # 断言当前节点的类型不是 "onnx::Slice"
            self.assertNotEqual(node.kind(), "onnx::Slice")
            # 断言当前节点的类型不是 "onnx::Concat"
            self.assertNotEqual(node.kind(), "onnx::Concat")
            # 断言当前节点的类型不是 "onnx::Unsqueeze"

        # 如果 opset 版本小于等于 12
        if self.opset_version <= 12:
            # 断言图中节点的数量为 3
            self.assertEqual(len(list(graph.nodes())), 3)
        else:
            # 否则，当 opset 版本大于 12 时，断言图中节点的数量为 4
            # 当 opset 版本 >= 13 时，Unsqueeze 操作的参数 "axes" 作为输入而不是属性
            self.assertEqual(len(list(graph.nodes())), 4)

    def test_constant_fold_transpose_matmul(self):
        # 定义一个名为 MatMulNet 的类，继承自 torch.nn.Module
        class MatMulNet(torch.nn.Module):
            # 初始化方法，定义了一个名为 B 的参数，形状为 (5, 3)
            def __init__(self):
                super().__init__()
                self.B = torch.nn.Parameter(torch.ones(5, 3))

            # 前向传播方法，接受 A 作为参数，执行矩阵乘法和转置操作
            def forward(self, A):
                return torch.matmul(A, torch.transpose(self.B, -1, -2))

        # 设置全局变量，导出 ONNX 操作集版本号
        GLOBALS.export_onnx_opset_version = self.opset_version
        # 设置全局变量，导出操作类型为 ONNX
        GLOBALS.operator_export_type = OperatorExportTypes.ONNX
        # 创建一个形状为 (2, 3) 的随机张量 A
        A = torch.randn(2, 3)
        # 将 MatMulNet 实例化为 model，将 A 作为参数传入 _model_to_graph 方法
        # 设置 A 的动态轴
        graph, _, __ = self._model_to_graph(
            MatMulNet(), (A,), input_names=["A"], dynamic_axes={"A": [0, 1]}
        )

        # 遍历图中的每个节点
        for node in graph.nodes():
            # 断言当前节点的类型不是 "onnx::Transpose"
            self.assertNotEqual(node.kind(), "onnx::Transpose")
        # 断言图中节点的数量为 1
        self.assertEqual(len(list(graph.nodes())), 1)

    def test_constant_fold_reshape(self):
        # 定义一个名为 ReshapeModule 的类，继承自 torch.nn.Module
        class ReshapeModule(torch.nn.Module):
            # 初始化方法，注册一个形状为 (5,) 的缓冲区 weight
            def __init__(
                self,
            ):
                super().__init__()
                self.register_buffer("weight", torch.ones(5))

            # 前向传播方法，将输入 x 与缓冲区 weight 进行形状重塑后相乘
            def forward(self, x):
                b = self.weight.reshape(1, -1, 1, 1)
                return x * b

        # 设置全局变量，导出 ONNX 操作集版本号
        GLOBALS.export_onnx_opset_version = self.opset_version
        # 设置全局变量，导出操作类型为 ONNX
        GLOBALS.operator_export_type = OperatorExportTypes.ONNX
        # 创建一个形状为 (4, 5) 的随机张量 x
        x = torch.randn(4, 5)
        # 将 ReshapeModule 实例化为 model，将 x 作为参数传入 _model_to_graph 方法
        # 设置 x 的动态轴
        graph, _, __ = self._model_to_graph(
            ReshapeModule(), (x,), input_names=["x"], dynamic_axes={"x": [0, 1]}
        )

        # 遍历图中的每个节点
        for node in graph.nodes():
            # 断言当前节点的类型不是 "onnx::Reshape"
            self.assertNotEqual(node.kind(), "onnx::Reshape")
        # 断言图中节点的数量为 1
        self.assertEqual(len(list(graph.nodes())), 1)
    def test_constant_fold_add(self):
        # 定义一个测试方法，用于验证常量折叠中的加法操作
        class Module(torch.nn.Module):
            def __init__(
                self,
            ):
                super().__init__()
                # 注册一个名为 "weight" 的缓冲区，其中元素均为1的张量
                self.register_buffer("weight", torch.ones(5))

            def forward(self, x):
                # 执行加法操作，将模块中的权重张量和给定张量 [1, 2, 3, 4, 5] 相加
                add = self.weight + torch.tensor([1, 2, 3, 4, 5])
                # 返回加法结果减去输入张量 x
                return add - x

        # 生成一个形状为 (2, 5) 的随机张量 x
        x = torch.randn(2, 5)
        # 设置全局变量，导出 ONNX 操作集版本号
        GLOBALS.export_onnx_opset_version = self.opset_version
        # 设置全局变量，指定导出类型为 ONNX
        GLOBALS.operator_export_type = OperatorExportTypes.ONNX
        # 将 Module 实例转换为计算图 graph，params_dict 包含参数信息
        graph, params_dict, __ = self._model_to_graph(
            Module(),
            (x,),
            do_constant_folding=True,
            operator_export_type=OperatorExportTypes.ONNX,
            input_names=["x"],
            dynamic_axes={"x": [0, 1]},
        )

        # 遍历图中的每个节点
        for node in graph.nodes():
            # 断言所有节点的类型不是 "onnx::Add"
            self.assertTrue(node.kind() != "onnx::Add")
        # 断言图中节点的数量为1
        self.assertEqual(len(list(graph.nodes())), 1)
        
        # 获取参数字典中的所有参数
        params = list(params_dict.values())
        # 断言参数数量为1
        self.assertEqual(len(params), 1)
        # 获取权重参数
        weight = params[0]
        # 断言权重张量与预期的张量相等
        self.assertEqual(weight, torch.tensor([2.0, 3.0, 4.0, 5.0, 6.0]))
    def test_constant_fold_sub(self):
        class Module(torch.nn.Module):
            def __init__(
                self,
            ):
                super().__init__()
                # 注册一个名为 "weight" 的缓冲区，其值为全1的张量，形状为 (5,)
                self.register_buffer("weight", torch.ones(5))

            def forward(self, x):
                # 计算 self.weight 减去 [1, 2, 3, 4, 5] 的张量
                sub = self.weight - torch.tensor([1, 2, 3, 4, 5])
                # 返回 sub 加上输入张量 x 的结果
                return sub + x

        x = torch.randn(2, 5)
        # 设置全局变量 export_onnx_opset_version 为当前对象的 opset_version 属性值
        GLOBALS.export_onnx_opset_version = self.opset_version
        # 设置全局变量 operator_export_type 为 ONNX 运算符导出类型
        GLOBALS.operator_export_type = OperatorExportTypes.ONNX
        # 转换 Module 实例为计算图，获取图、参数字典及额外信息
        graph, params_dict, __ = self._model_to_graph(
            Module(),
            (x,),
            do_constant_folding=True,  # 启用常量折叠
            operator_export_type=OperatorExportTypes.ONNX,
            input_names=["x"],  # 设置输入名称
            dynamic_axes={"x": [0, 1]},  # 设置动态轴
        )
        # 遍历计算图中的节点，确保没有 "onnx::Sub" 类型的节点
        for node in graph.nodes():
            self.assertNotEqual(node.kind(), "onnx::Sub")
        # 断言计算图节点的数量为 1
        self.assertEqual(len(list(graph.nodes())), 1)
        # 获取参数字典中的所有参数
        params = list(params_dict.values())
        # 断言参数列表长度为 1
        self.assertEqual(len(params), 1)
        # 获取参数列表中的第一个参数作为 weight
        weight = params[0]
        # 断言 weight 的值为 torch.tensor([0.0, -1.0, -2.0, -3.0, -4.0])
        self.assertEqual(weight, torch.tensor([0.0, -1.0, -2.0, -3.0, -4.0]))

    def test_constant_fold_sqrt(self):
        class Module(torch.nn.Module):
            def __init__(
                self,
            ):
                super().__init__()
                # 注册一个名为 "weight" 的缓冲区，其值为全1的张量，形状为 (5,)
                self.register_buffer("weight", torch.ones(5))

            def forward(self, x):
                # 计算 self.weight 的平方根
                sqrt = torch.sqrt(self.weight)
                # 返回 sqrt 除以输入张量 x 的结果
                return sqrt / x

        x = torch.randn(2, 5)
        GLOBALS.export_onnx_opset_version = self.opset_version
        GLOBALS.operator_export_type = OperatorExportTypes.ONNX
        # 转换 Module 实例为计算图，获取图、空参数字典及额外信息
        graph, _, __ = self._model_to_graph(
            Module(), (x,), input_names=["x"], dynamic_axes={"x": [0, 1]}
        )
        # 遍历计算图中的节点，确保没有 "onnx::Sqrt" 类型的节点
        for node in graph.nodes():
            self.assertNotEqual(node.kind(), "onnx::Sqrt")
        # 断言计算图节点的数量为 1
        self.assertEqual(len(list(graph.nodes())), 1)

    def test_constant_fold_shape(self):
        class ShapeModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 注册一个名为 "weight" 的缓冲区，其值为全1的张量，形状为 (5,)
                self.register_buffer("weight", torch.ones(5))

            def forward(self, x):
                # 获取 self.weight 的形状的第一个维度（即 5）
                shape = self.weight.shape[0]
                # 返回输入张量 x 加上 shape 的结果
                return x + shape

        x = torch.randn(2, 5)
        GLOBALS.export_onnx_opset_version = self.opset_version
        GLOBALS.operator_export_type = OperatorExportTypes.ONNX
        # 转换 ShapeModule 实例为计算图，获取图、空参数字典及额外信息
        graph, _, __ = self._model_to_graph(
            ShapeModule(), (x,), input_names=["x"], dynamic_axes={"x": [0, 1]}
        )
        # 遍历计算图中的节点，确保没有 "onnx::Shape" 类型的节点
        for node in graph.nodes():
            self.assertNotEqual(node.kind(), "onnx::Shape")
        # 断言计算图节点的数量为 2
        self.assertEqual(len(list(graph.nodes())), 2)
    # 定义一个测试函数，测试常量折叠和上采样比例被视为常量的情况
    def test_constant_fold_upsample_scale_fold_as_constant(self):
        # 上采样比例是一个常量，不是模型参数，
        # 因此在常量折叠后不应将其添加为初始化器。
        # 创建一个 torch 的 Upsample 模型实例，指定上采样比例为 2，模式为双线性插值，角落对齐为 True
        model = torch.nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        # 创建一个大小为 (1, 32, 224, 224) 的随机张量 x
        x = torch.randn(1, 32, 224, 224)
        # 创建一个字节流对象 f
        f = io.BytesIO()
        # 将模型导出为 ONNX 格式，存储到字节流 f 中
        torch.onnx.export(model, x, f)
        # 从字节流中加载 ONNX 模型
        onnx_model = onnx.load(io.BytesIO(f.getvalue()))
        # 断言 ONNX 模型的初始化器列表长度为 0
        self.assertEqual(len(onnx_model.graph.initializer), 0)

    # 定义一个测试函数，测试 verbose 参数
    def test_verbose(self):
        # 定义一个简单的 torch Module，用于计算输入的指数
        class MyModule(torch.nn.Module):
            def forward(self, input):
                return torch.exp(input)

        # 创建一个大小为 (3, 4) 的随机张量 x
        x = torch.randn(3, 4)

        # 定义一个函数，用于检查模型是否被剥离了文档字符串
        def is_model_stripped(f, verbose=None):
            # 如果 verbose 为 None，则导出模型到 ONNX 格式，不输出详细信息
            if verbose is None:
                torch.onnx.export(MyModule(), x, f, opset_version=self.opset_version)
            else:
                # 否则，导出模型到 ONNX 格式，输出详细信息
                torch.onnx.export(
                    MyModule(), x, f, verbose=verbose, opset_version=self.opset_version
                )
            # 从字节流中加载 ONNX 模型
            model = onnx.load(io.BytesIO(f.getvalue()))
            # 复制一份模型并剥离文档字符串
            model_strip = copy.copy(model)
            onnx.helper.strip_doc_string(model_strip)
            # 返回模型是否与剥离文档字符串后的模型相等的布尔值
            return model == model_strip

        # 测试 verbose=False （默认情况）
        self.assertTrue(is_model_stripped(io.BytesIO()))
        # 测试 verbose=True
        self.assertFalse(is_model_stripped(io.BytesIO(), True))

    # NB: DataParallel 可能无法正确处理时，请移除此测试
    # 定义一个测试函数，测试在 DataParallel 模式下导出时是否会出错
    def test_error_on_data_parallel(self):
        # 创建一个 DataParallel 模型，对输入进行反射填充
        model = torch.nn.DataParallel(torch.nn.ReflectionPad2d((1, 2, 3, 4)))
        # 创建一个大小为 (1, 2, 3, 4) 的随机张量 x
        x = torch.randn(1, 2, 3, 4)
        # 创建一个字节流对象 f
        f = io.BytesIO()
        # 使用 assertRaisesRegex 检查在导出时是否引发特定错误信息
        with self.assertRaisesRegex(
            ValueError,
            "torch.nn.DataParallel is not supported by ONNX "
            "exporter, please use 'attribute' module to "
            "unwrap model from torch.nn.DataParallel. Try ",
        ):
            # 尝试将 DataParallel 模型导出到 ONNX 格式
            torch.onnx.export(model, x, f, opset_version=self.opset_version)

    # 在不支持的最小 opset 版本为 11 时跳过此测试
    @skipIfUnsupportedMinOpsetVersion(11)
    # 定义一个测试方法，用于测试模型的序列维度处理
    def test_sequence_dim(self):
        # 定义一个继承自torch.nn.Module的模型类
        class Module(torch.nn.Module):
            # 模型的前向传播方法，接受两个参数 x 和 y，并将它们作为列表返回
            def forward(self, x, y):
                return [x, y]

        # 创建 Module 类的实例
        model = Module()
        # 使用 torch.jit.script 方法将模型转换为 Torch 脚本，以保持输出为序列类型
        script_model = torch.jit.script(model)
        # 生成一个形状为 (2, 3) 的随机张量 x
        x = torch.randn(2, 3)

        # Case 1: dynamic axis
        # 创建一个 BytesIO 对象 f
        f = io.BytesIO()
        # 生成一个形状为 (2, 3) 的随机张量 y
        y = torch.randn(2, 3)
        # 使用 torch.onnx.export 导出 ONNX 模型，指定动态轴 "y" 的维度为 [1]
        torch.onnx.export(
            script_model,
            (x, y),
            f,
            opset_version=self.opset_version,
            input_names=["x", "y"],
            dynamic_axes={"y": [1]},
        )
        # 加载导出的 ONNX 模型
        onnx_model = onnx.load(io.BytesIO(f.getvalue()))
        # 获取 ONNX 模型图的输出张量信息的第一个输出
        loop_output_value_info_proto = onnx_model.graph.output[0]
        # 创建一个参考的 Tensor 序列值信息 proto，维度为 [2, None]
        ref_value_info_proto = onnx.helper.make_tensor_sequence_value_info(
            loop_output_value_info_proto.name, 1, [2, None]
        )
        # 断言 ONNX 模型输出与参考值信息 proto 相等
        self.assertEqual(loop_output_value_info_proto, ref_value_info_proto)

        # Case 2: no dynamic axes.
        # 创建一个新的 BytesIO 对象 f
        f = io.BytesIO()
        # 重新生成一个形状为 (2, 3) 的随机张量 y
        y = torch.randn(2, 3)
        # 使用 torch.onnx.export 导出 ONNX 模型，未指定动态轴
        torch.onnx.export(script_model, (x, y), f, opset_version=self.opset_version)
        # 加载导出的 ONNX 模型
        onnx_model = onnx.load(io.BytesIO(f.getvalue()))
        # 获取 ONNX 模型图的输出张量信息的第一个输出
        loop_output_value_info_proto = onnx_model.graph.output[0]
        # 创建一个参考的 Tensor 序列值信息 proto，维度为 [2, 3]
        ref_value_info_proto = onnx.helper.make_tensor_sequence_value_info(
            loop_output_value_info_proto.name, 1, [2, 3]
        )
        # 断言 ONNX 模型输出与参考值信息 proto 相等
        self.assertEqual(loop_output_value_info_proto, ref_value_info_proto)

    # 定义一个测试方法，用于测试模型导出的模式
    def test_export_mode(self):
        # 定义一个继承自 torch.nn.Module 的模型类
        class MyModule(torch.nn.Module):
            # 模型的前向传播方法，接受一个参数 x，并返回 x + 1
            def forward(self, x):
                y = x + 1
                return y

        # 创建 MyModule 类的实例
        model = MyModule()
        # 生成一个形状为 (10, 3, 128, 128) 的随机张量 x
        x = torch.randn(10, 3, 128, 128)
        # 创建一个 BytesIO 对象 f
        f = io.BytesIO()

        # 设置模型为评估模式，并导出为训练模式的 ONNX 模型
        model.eval()
        old_state = model.training
        torch.onnx.export(
            model,
            (x,),
            f,
            opset_version=self.opset_version,
            training=torch.onnx.TrainingMode.TRAINING,
        )
        # 验证模型状态是否保持不变
        self.assertEqual(model.training, old_state)

        # 设置模型为训练模式，并导出为推理模式的 ONNX 模型
        model.train()
        old_state = model.training
        torch.onnx.export(
            model,
            (x,),
            f,
            opset_version=self.opset_version,
            training=torch.onnx.TrainingMode.EVAL,
        )
        # 验证模型状态是否保持不变
        self.assertEqual(model.training, old_state)
    def test_export_does_not_fail_on_frozen_scripted_module(self):
        # 定义内部的 PyTorch 模块 Inner，继承自 torch.nn.Module
        class Inner(torch.nn.Module):
            # 定义前向传播函数
            def forward(self, x):
                # 如果输入 x 大于 0，则返回 x
                if x > 0:
                    return x
                # 否则返回 x 的平方
                else:
                    return x * x

        # 定义外部的 PyTorch 模块 Outer，继承自 torch.nn.Module
        class Outer(torch.nn.Module):
            # 构造函数
            def __init__(self):
                super().__init__()
                # 内部包含一个通过 torch.jit.script 脚本化的 Inner 模块
                self.inner = torch.jit.script(Inner())

            # 前向传播函数
            def forward(self, x):
                # 调用内部的 Inner 模块进行前向传播
                return self.inner(x)

        # 创建一个大小为 1 的全零张量 x
        x = torch.zeros(1)
        # 创建 Outer 类的实例并将其设置为评估模式
        outer_module = Outer().eval()
        # 使用 torch.jit.trace_module 对 Outer 模块进行跟踪
        module = torch.jit.trace_module(outer_module, {"forward": (x)})
        # 使用 torch.jit.freeze 冻结模块，移除模块中的训练属性
        module = torch.jit.freeze(module)

        # 使用 torch.onnx.export 导出模块到 ONNX 格式
        torch.onnx.export(module, (x,), io.BytesIO(), opset_version=self.opset_version)

    @skipIfUnsupportedMinOpsetVersion(15)
    @skipIfUnsupportedMinOpsetVersion(15)
    def test_local_function_overloads(self):
        # 定义一个带有多个重载的 PyTorch 模块 NWithOverloads，继承自 torch.nn.Module
        class NWithOverloads(torch.nn.Module):
            # 定义前向传播函数，支持多个参数组合
            def forward(self, x, y=None, z=None):
                if y is None:
                    return x + 1
                elif z is None:
                    return x + y
                else:
                    return x + y, x + z

        # 定义一个 PyTorch 模块 M，继承自 torch.nn.Module
        class M(torch.nn.Module):
            # 构造函数，接受一个参数 num_layers
            def __init__(self, num_layers):
                super().__init__()
                # 创建一个 NWithOverloads 的实例
                self.n = NWithOverloads()

            # 前向传播函数，调用内部的 NWithOverloads 模块的不同重载
            def forward(self, x, y, z):
                return self.n(x), self.n(x, y), self.n(x, y, z)

        # 创建大小为 (2, 3) 的随机张量 x, y, z
        x = torch.randn(2, 3)
        y = torch.randn(2, 3)
        z = torch.randn(2, 3)

        # 创建一个字节流对象 f
        f = io.BytesIO()
        # 使用 torch.onnx.export 导出 M(3) 模块到 ONNX 格式
        torch.onnx.export(
            M(3),
            (x, y, z),
            f,
            opset_version=self.opset_version,
            export_modules_as_functions={NWithOverloads},
        )

        # 加载导出的 ONNX 模型并获取其中的函数列表
        onnx_model = onnx.load(io.BytesIO(f.getvalue()))
        funcs = onnx_model.functions
        # 断言函数列表的长度为 3
        self.assertEqual(len(funcs), 3)
        # 获取函数名称列表
        func_names = [f.name for f in funcs]
        # 断言函数列表包含特定的函数名称
        self.assertIn("NWithOverloads", func_names)
        self.assertIn("NWithOverloads.1", func_names)
        self.assertIn("NWithOverloads.2", func_names)

    # 在 ONNX 1.13.0 之后版本失败
    @skipIfUnsupportedMaxOpsetVersion(1)
    def test_local_function_infer_scopes(self):
        class M(torch.nn.Module):
            def forward(self, x):
                # Concatenation of scalars inserts unscoped tensors in IR graph.
                # 构造新的张量形状，将最后两维替换为 (1, 1, -1)
                new_tensor_shape = x.size()[:-1] + (1, 1, -1)
                # 对输入张量按新形状进行视图变换
                tensor = x.view(*new_tensor_shape)
                return tensor

        x = torch.randn(4, 5)
        f = io.BytesIO()
        # 导出模型到ONNX格式
        torch.onnx.export(
            M(),
            (x,),
            f,
            export_modules_as_functions=True,
            opset_version=self.opset_version,
            do_constant_folding=False,
        )

        # 加载导出的ONNX模型
        onnx_model = onnx.load(io.BytesIO(f.getvalue()))
        funcs = onnx_model.functions
        # 断言函数'M'在ONNX模型的函数列表中
        self.assertIn("M", [f.name for f in funcs])

    @skipIfUnsupportedMinOpsetVersion(15)
    def test_local_function_predefined_attributes(self):
        class M(torch.nn.Module):
            num_layers: int

            def __init__(self, num_layers):
                super().__init__()
                self.num_layers = num_layers
                # 初始化LayerNorm层的列表
                self.lns = torch.nn.ModuleList(
                    [torch.nn.LayerNorm(3, eps=1e-4) for _ in range(num_layers)]
                )

            def forward(self, x):
                # 在每一层LayerNorm上进行前向传播
                for ln in self.lns:
                    x = ln(x)
                return x

        x = torch.randn(2, 3)
        f = io.BytesIO()
        model = M(3)
        # 导出模型到ONNX格式
        torch.onnx.export(
            model,
            (x,),
            f,
            export_modules_as_functions=True,
            opset_version=self.opset_version,
        )

        # 加载导出的ONNX模型
        onnx_model = onnx.load(io.BytesIO(f.getvalue()))
        funcs = onnx_model.functions
        # 获取名称为'M'的函数
        m_funcs = [fn for fn in funcs if fn.name == "M"]
        # 断言'M'函数的属性为["num_layers"]
        self.assertEqual(m_funcs[0].attribute, ["num_layers"])
        # 获取名称为'LayerNorm'的函数
        ln_funcs = [fn for fn in funcs if fn.name == "LayerNorm"]
        # 断言'LayerNorm'函数的属性为["eps", "elementwise_affine"]
        self.assertEqual(ln_funcs[0].attribute, ["eps", "elementwise_affine"])

        from onnx import helper

        # 获取ONNX图中操作类型为'M'的节点
        m_node = [n for n in onnx_model.graph.node if n.op_type == "M"]
        # 断言第一个属性为"num_layers"，其值为模型的层数
        self.assertEqual(
            m_node[0].attribute[0],
            helper.make_attribute("num_layers", model.num_layers),
        )

        # 获取'M'函数中操作类型为'LayerNorm'的节点
        ln_nodes = [n for n in m_funcs[0].node if n.op_type == "LayerNorm"]
        # 期望的LayerNorm属性列表
        expected_ln_attrs = [
            helper.make_attribute(
                "elementwise_affine", model.lns[0].elementwise_affine
            ),
            helper.make_attribute("eps", model.lns[0].eps),
        ]
        # 遍历LayerNorm节点，断言其属性在期望的属性列表中
        for ln_node in ln_nodes:
            self.assertIn(ln_node.attribute[0], expected_ln_attrs)
            self.assertIn(ln_node.attribute[1], expected_ln_attrs)
    # 使用 @skipIfUnsupportedMinOpsetVersion 装饰器，跳过不支持最小操作集版本的测试用例
    @skipIfUnsupportedMinOpsetVersion(15)
    def test_local_function_subset_of_predefined_attributes(self):
        # 定义一个名为 M 的继承自 torch.nn.Module 的类
        class M(torch.nn.Module):
            # 类型提示：整数类型的成员变量 num_layers
            num_layers: int

            # 构造函数，接受 num_layers 参数
            def __init__(self, num_layers):
                super().__init__()
                # 创建一个 Embedding 层，从预训练的 FloatTensor 初始化
                self.embed_layer = torch.nn.Embedding.from_pretrained(
                    torch.FloatTensor([[1, 2.3, 3], [4, 5.1, 6.3]])
                )
                # 初始化 num_layers 属性
                self.num_layers = num_layers
                # 创建一个包含 num_layers 个 LayerNorm 层的 ModuleList
                self.lns = torch.nn.ModuleList(
                    [torch.nn.LayerNorm(3, eps=1e-4) for _ in range(num_layers)]
                )

            # 前向传播方法，接受输入 x
            def forward(self, x):
                # 使用 embed_layer 处理输入向量 [1]，得到 e
                e = self.embed_layer(torch.LongTensor([1]))
                # 遍历 self.lns 中的每个 LayerNorm 层 ln，并将 x 依次传递给它们
                for ln in self.lns:
                    x = ln(x)
                # 返回处理后的 x 和 e
                return x, e

        # 创建一个形状为 (2, 3) 的随机输入张量 x
        x = torch.randn(2, 3)
        # 创建一个字节流对象 f
        f = io.BytesIO()
        # 创建一个 M 类的实例 model，传入参数 num_layers=3
        model = M(3)
        # 使用 torch.onnx.export 导出模型 model 到字节流 f
        torch.onnx.export(
            model,
            (x,),  # 输入是 x
            f,     # 输出到 f
            export_modules_as_functions=True,  # 将模块作为函数导出
            opset_version=self.opset_version,   # 使用指定的操作集版本
            verbose=True,  # 允许打印详细信息，如“Skipping module attribute 'freeze'”
        )
    def test_node_scope(self):
        # 定义内部类 N，继承自 torch.nn.Module
        class N(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.relu = torch.nn.ReLU()

            # 定义前向传播函数，使用 ReLU 激活函数
            def forward(self, x):
                return self.relu(x)

        # 定义内部类 M，继承自 torch.nn.Module
        class M(torch.nn.Module):
            def __init__(self, num_layers):
                super().__init__()
                self.num_layers = num_layers
                # 初始化 num_layers 个 LayerNorm 层，每个层的 eps 值不同
                self.lns = torch.nn.ModuleList(
                    [torch.nn.LayerNorm(3, eps=float(i)) for i in range(num_layers)]
                )
                # 初始化两个 GELU 激活函数
                self.gelu1 = torch.nn.GELU()
                self.gelu2 = torch.nn.GELU()
                # 初始化 N 类实例，包含 ReLU 激活函数
                self.relu = N()

            # 定义前向传播函数，执行 gelu1、gelu2 操作，然后对 z 应用多个 LayerNorm 层，并返回结果
            def forward(self, x, y, z):
                res1 = self.gelu1(x)
                res2 = self.gelu2(y)
                for ln in self.lns:
                    z = ln(z)
                return res1 + res2, self.relu(z)

        # 生成输入张量 x, y, z
        x = torch.randn(2, 3)
        y = torch.randn(2, 3)
        z = torch.randn(2, 3)

        # 实例化 M 类对象 model，num_layers 设置为 3
        model = M(3)

        # 期望的作用域名集合，用于验证模型转换后的图结构
        expected_scope_names = {
            "M::/torch.nn.modules.activation.GELU::gelu1",
            "M::/torch.nn.modules.activation.GELU::gelu2",
            "M::/torch.nn.modules.normalization.LayerNorm::lns.0",
            "M::/torch.nn.modules.normalization.LayerNorm::lns.1",
            "M::/torch.nn.modules.normalization.LayerNorm::lns.2",
            "M::/N::relu/torch.nn.modules.activation.ReLU::relu",
            "M::",
        }

        # 将模型 model 转换为图形表示，获取图形、输入输出信息
        graph, _, _ = self._model_to_graph(
            model, (x, y, z), input_names=[], dynamic_axes={}
        )

        # 遍历图形中的每个节点，验证其作用域名是否在期望的集合中
        for node in graph.nodes():
            self.assertIn(
                _remove_test_environment_prefix_from_scope_name(node.scopeName()),
                expected_scope_names,
            )

        # 将 torch script 模型转换为图形表示，获取图形、输入输出信息
        graph, _, _ = self._model_to_graph(
            torch.jit.script(model), (x, y, z), input_names=[], dynamic_axes={}
        )

        # 遍历图形中的每个节点，验证其作用域名是否在期望的集合中
        for node in graph.nodes():
            self.assertIn(
                _remove_test_environment_prefix_from_scope_name(node.scopeName()),
                expected_scope_names,
            )
    def test_scope_of_constants_when_combined_by_cse_pass(self):
        # 定义测试函数，验证常量在 CSE（公共子表达式消除）过程中的作用域
        layer_num = 3  # 设置层数为3

        class M(torch.nn.Module):
            def __init__(self, constant):
                super().__init__()
                self.constant = constant  # 初始化模块的常量属性

            def forward(self, x):
                # 'self.constant' 设计为所有层都相同，因此是常见的子表达式
                return x + self.constant  # 返回输入 x 加上常量的结果

        class N(torch.nn.Module):
            def __init__(self, layers: int = layer_num):
                super().__init__()
                # 使用 ModuleList 创建包含多个 M 类型模块的层列表，每个层的常量为1.0
                self.layers = torch.nn.ModuleList(
                    [M(constant=torch.tensor(1.0)) for i in range(layers)]
                )

            def forward(self, x):
                # 遍历每一层，并对输入 x 进行前向传播
                for layer in self.layers:
                    x = layer(x)
                return x

        # 将 N 类模型实例化并转换为图形表示
        graph, _, _ = self._model_to_graph(
            N(), (torch.randn(2, 3)), input_names=[], dynamic_axes={}
        )

        # 期望的常量作用域名，用于验证不同层的常量作用域
        # 预期的根作用域名
        expected_root_scope_name = "N::"
        # 预期的层级作用域名
        expected_layer_scope_name = "M::layers"
        # 预期的常量作用域名列表，由于有3层，因此会有3个常量作用域名
        expected_constant_scope_name = [
            f"{expected_root_scope_name}/{expected_layer_scope_name}.{i}"
            for i in range(layer_num)
        ]

        # 存储实际的常量作用域名
        constant_scope_names = []
        for node in graph.nodes():
            if node.kind() == "onnx::Constant":
                # 获取节点的作用域名，并去除测试环境前缀
                constant_scope_names.append(
                    _remove_test_environment_prefix_from_scope_name(node.scopeName())
                )
        
        # 断言实际的常量作用域名与预期的常量作用域名列表相匹配
        self.assertEqual(constant_scope_names, expected_constant_scope_name)
    def test_scope_of_nodes_when_combined_by_cse_pass(self):
        # 定义层数
        layer_num = 3

        class M(torch.nn.Module):
            def __init__(self, constant, bias):
                super().__init__()
                # 初始化常量和偏置
                self.constant = constant
                self.bias = bias

            def forward(self, x):
                # 对所有层设计为`constant`和`x`相同，因此 `x + self.constant` 是共同子表达式。
                # `bias` 对每一层设计为不同，因此 `* self.bias` 不是共同子表达式。
                return (x + self.constant) * self.bias

        class N(torch.nn.Module):
            def __init__(self, layers: int = layer_num):
                super().__init__()

                # 使用 ModuleList 创建层列表
                self.layers = torch.nn.ModuleList(
                    [
                        M(constant=torch.tensor([1.0]), bias=torch.randn(1))
                        for i in range(layers)
                    ]
                )

            def forward(self, x):
                y = []
                # 遍历每一层并将结果存储在 y 中
                for layer in self.layers:
                    y.append(layer(x))
                return y[0], y[1], y[2]

        # 将 N 模型转换为图形表示
        graph, _, _ = self._model_to_graph(
            N(), (torch.randn(2, 3)), input_names=[], dynamic_axes={}
        )
        # 预期的根节点作用域名称
        expected_root_scope_name = "N::"
        # 预期的层级作用域名称
        expected_layer_scope_name = "M::layers"
        # 预期的加法操作作用域名称列表
        expected_add_scope_names = [
            f"{expected_root_scope_name}/{expected_layer_scope_name}.0"
        ]
        # 预期的乘法操作作用域名称列表
        expected_mul_scope_names = [
            f"{expected_root_scope_name}/{expected_layer_scope_name}.{i}"
            for i in range(layer_num)
        ]

        # 初始化空列表以存储实际的加法和乘法操作的作用域名称
        add_scope_names = []
        mul_scope_names = []
        # 遍历图中的每个节点
        for node in graph.nodes():
            if node.kind() == "onnx::Add":
                # 添加去除测试环境前缀后的加法操作的作用域名称
                add_scope_names.append(
                    _remove_test_environment_prefix_from_scope_name(node.scopeName())
                )
            elif node.kind() == "onnx::Mul":
                # 添加去除测试环境前缀后的乘法操作的作用域名称
                mul_scope_names.append(
                    _remove_test_environment_prefix_from_scope_name(node.scopeName())
                )
        # 断言实际的加法操作作用域名称与预期的一致
        self.assertEqual(add_scope_names, expected_add_scope_names)
        # 断言实际的乘法操作作用域名称与预期的一致
        self.assertEqual(mul_scope_names, expected_mul_scope_names)

    def test_aten_fallthrough(self):
        # 测试没有符号的 op 的 ATen 导出
        class Module(torch.nn.Module):
            def forward(self, x):
                return torch.erfc(x)

        # 创建输入张量 x
        x = torch.randn(2, 3, 4)
        # 设置全局变量 export_onnx_opset_version 为指定的 opset_version
        GLOBALS.export_onnx_opset_version = self.opset_version
        # 将 Module 模型转换为图形表示
        graph, _, __ = self._model_to_graph(
            Module(),
            (x,),
            operator_export_type=OperatorExportTypes.ONNX_FALLTHROUGH,
            input_names=["x"],
            dynamic_axes={"x": [0, 1, 2]},
        )
        # 获取图中的第一个节点，并断言其类型为 "aten::erfc"
        iter = graph.nodes()
        self.assertEqual(next(iter).kind(), "aten::erfc")
    def test_custom_op_fallthrough(self):
        # Test custom op

        # 定义包含自定义操作的 C++ 源码字符串
        op_source = """
        #include <torch/script.h>

        torch::Tensor custom_add(torch::Tensor self, torch::Tensor other) {
          return self + other;
        }

        // 注册自定义操作到 TorchScript
        static auto registry =
          torch::RegisterOperators("custom_namespace::custom_op", &custom_add);
        """

        # 使用 torch.utils.cpp_extension.load_inline 方法加载内联的 C++ 源码作为扩展模块
        torch.utils.cpp_extension.load_inline(
            name="custom_add",
            cpp_sources=op_source,
            is_python_module=False,
            verbose=True,
        )

        # 定义一个继承自 torch.nn.Module 的模型类 FooModel
        class FooModel(torch.nn.Module):
            def forward(self, input, other):
                # 调用自定义操作 custom_namespace::custom_op
                return torch.ops.custom_namespace.custom_op(input, other)

        # 生成随机张量 x 和 y
        x = torch.randn(2, 3, 4, requires_grad=False)
        y = torch.randn(2, 3, 4, requires_grad=False)
        # 创建 FooModel 的实例
        model = FooModel()
        # 将模型转换为计算图
        graph, _, __ = self._model_to_graph(
            model,
            (x, y),
            operator_export_type=torch.onnx.OperatorExportTypes.ONNX_FALLTHROUGH,
            input_names=["x", "y"],
            dynamic_axes={"x": [0, 1, 2], "y": [0, 1, 2]},
        )
        # 获取计算图的迭代器
        iter = graph.nodes()
        # 断言第一个节点的操作类型为 "custom_namespace::custom_op"
        self.assertEqual(next(iter).kind(), "custom_namespace::custom_op")

    # gelu is exported as onnx::Gelu for opset >= 20
    @skipIfUnsupportedMaxOpsetVersion(19)
    def test_custom_opsets_gelu(self):
        # 使用 self.addCleanup 注册清理函数，取消注册 "::gelu" 符号对应的自定义操作
        self.addCleanup(torch.onnx.unregister_custom_op_symbolic, "::gelu", 9)

        # 定义 gelu 函数作为符号函数实现
        def gelu(g, self, approximate):
            return g.op("com.microsoft::Gelu", self).setType(self.type())

        # 注册 "::gelu" 符号对应的自定义操作
        torch.onnx.register_custom_op_symbolic("::gelu", gelu, 9)
        # 创建使用 GELU 激活函数的模型
        model = torch.nn.GELU(approximate="none")
        # 生成随机张量 x
        x = torch.randn(3, 3)
        f = io.BytesIO()
        # 将模型导出为 ONNX 格式，并写入字节流 f
        torch.onnx.export(
            model,
            (x,),
            f,
            opset_version=self.opset_version,
            custom_opsets={"com.microsoft": 1},
        )

        # 加载导出的 ONNX 模型
        graph = onnx.load(io.BytesIO(f.getvalue()))
        # 断言第一个节点的操作类型为 "Gelu"
        self.assertEqual(graph.graph.node[0].op_type, "Gelu")
        # 断言第一个 opset_import 的版本为 self.opset_version
        self.assertEqual(graph.opset_import[0].version, self.opset_version)
        # 断言第二个 opset_import 的域为 "com.microsoft"，版本为 1
        self.assertEqual(graph.opset_import[1].domain, "com.microsoft")
        self.assertEqual(graph.opset_import[1].version, 1)
    def test_register_aten_custom_op_symbolic(self):
        # 在测试开始前注册自定义的 ATen 操作符的符号化处理函数，并在测试结束时取消注册
        self.addCleanup(torch.onnx.unregister_custom_op_symbolic, "aten::gelu", 9)

        def gelu(g, self, approximate):
            # 实现自定义的 ATen gelu 操作符的符号化处理函数，返回符号化的 ONNX 操作
            return g.op("com.microsoft::Gelu", self).setType(self.type())

        # 注册自定义的 ATen gelu 操作符的符号化处理函数，指定版本号为 9
        torch.onnx.register_custom_op_symbolic("aten::gelu", gelu, 9)
        # 创建一个 GELU 模型实例
        model = torch.nn.GELU(approximate="none")
        # 创建输入张量 x
        x = torch.randn(3, 3)
        # 创建一个字节流对象
        f = io.BytesIO()
        # 导出模型到 ONNX 格式，写入字节流
        torch.onnx.export(model, (x,), f, opset_version=self.opset_version)
        # 从导出的 ONNX 字节流中加载计算图
        graph = onnx.load(io.BytesIO(f.getvalue()))

        # 断言第一个节点的操作类型为 "Gelu"
        self.assertEqual(graph.graph.node[0].op_type, "Gelu")
        # 断言导入的第二个操作集的域为 "com.microsoft"
        self.assertEqual(graph.opset_import[1].domain, "com.microsoft")

    @skipIfNoLapack
    def test_custom_opsets_inverse(self):
        # 在测试开始前注册自定义的 linalg_inv 操作符的符号化处理函数，并在测试结束时取消注册
        self.addCleanup(torch.onnx.unregister_custom_op_symbolic, "::linalg_inv", 9)

        class CustomInverse(torch.nn.Module):
            def forward(self, x):
                # 自定义的模块前向传播函数，返回输入张量的逆矩阵加上原始输入
                return torch.inverse(x) + x

        def linalg_inv(g, self):
            # 实现自定义的 linalg_inv 操作符的符号化处理函数，返回符号化的 ONNX 操作
            return g.op("com.microsoft::Inverse", self).setType(self.type())

        # 注册自定义的 linalg_inv 操作符的符号化处理函数，指定版本号为 9
        torch.onnx.register_custom_op_symbolic("::linalg_inv", linalg_inv, 9)
        # 创建 CustomInverse 模型实例
        model = CustomInverse()
        # 创建输入张量 x
        x = torch.randn(2, 3, 3)
        # 创建一个字节流对象
        f = io.BytesIO()
        # 导出模型到 ONNX 格式，写入字节流，并指定自定义操作集
        torch.onnx.export(
            model,
            (x,),
            f,
            opset_version=self.opset_version,
            custom_opsets={"com.microsoft": 1},
        )

        # 从导出的 ONNX 字节流中加载计算图
        graph = onnx.load(io.BytesIO(f.getvalue()))
        # 断言第一个节点的操作类型为 "Inverse"
        self.assertEqual(graph.graph.node[0].op_type, "Inverse")
        # 断言导入的第一个操作集的版本与当前测试设置的 opset_version 相等
        self.assertEqual(graph.opset_import[0].version, self.opset_version)
        # 断言导入的第二个操作集的域为 "com.microsoft"，版本为 1
        self.assertEqual(graph.opset_import[1].domain, "com.microsoft")
        self.assertEqual(graph.opset_import[1].version, 1)

    def test_onnx_fallthrough(self):
        # 测试 ATen 操作的导出，使用 ATen 的符号化处理函数
        class Module(torch.nn.Module):
            def forward(self, x):
                # 模块的前向传播函数，返回输入张量的 digamma 函数值
                return torch.digamma(x)

        x = torch.randn(100, 128)
        # 将模型转换为计算图，使用 ONNX_FALLTHROUGH 导出类型，指定输入名称和动态轴
        graph, _, __ = self._model_to_graph(
            Module(),
            (x,),
            operator_export_type=OperatorExportTypes.ONNX_FALLTHROUGH,
            input_names=["x"],
            dynamic_axes={"x": [0, 1]},
        )
        # 断言第一个节点的操作类型为 "aten::digamma"
        self.assertEqual(next(iter).kind(), "aten::digamma")

    # prim::ListConstruct is exported as onnx::SequenceConstruct for opset >= 11
    @skipIfUnsupportedMaxOpsetVersion(10)
    def test_prim_fallthrough(self):
        # Test prim op
        # 定义一个继承自torch.jit.ScriptModule的PrimModule类，用于测试基本操作
        class PrimModule(torch.jit.ScriptModule):
            @torch.jit.script_method
            def forward(self, x):
                # 如果输入x是列表，则y等于x本身
                if isinstance(x, list):
                    y = x
                else:
                    # 否则将x放入列表中赋给y
                    y = [x]
                return y

        x = torch.tensor([2])
        model = PrimModule()
        model.eval()
        # 将模型转换为计算图，使用输入x，导出类型为ONNX_FALLTHROUGH
        graph, _, __ = self._model_to_graph(
            model,
            (x,),
            operator_export_type=OperatorExportTypes.ONNX_FALLTHROUGH,
            input_names=["x"],
            dynamic_axes={"x": [0]},
        )
        # 获取计算图的节点迭代器，断言第一个节点的类型为"prim::ListConstruct"
        iter = graph.nodes()
        self.assertEqual(next(iter).kind(), "prim::ListConstruct")

    def test_custom_layer_tuple(self):
        # 定义一个自定义的torch.autograd.Function类CustomFunction
        class CustomFunction(torch.autograd.Function):
            @staticmethod
            def symbolic(g, input):
                # 在符号图中创建一个自定义命名空间的操作"CustomNamespace::Custom"，输出两个结果
                return g.op("CustomNamespace::Custom", input, outputs=2)

            @staticmethod
            def forward(ctx, input):
                # 前向传播函数，对输入进行限制在非负数范围内并返回
                return input, input

        # 定义一个继承自torch.nn.Module的Custom类
        class Custom(torch.nn.Module):
            def forward(self, input):
                # 使用CustomFunction中的forward方法进行前向传播
                return CustomFunction.apply(input)

        model = Custom()
        batch = torch.FloatTensor(1, 3)

        # 将模型转换为计算图，使用输入batch，输入名称为"batch"，动态轴为[0, 1]
        graph, _, _ = self._model_to_graph(
            model, batch, input_names=["batch"], dynamic_axes={"batch": [0, 1]}
        )
        # 获取计算图的节点迭代器，断言第一个节点的类型为"CustomNamespace::Custom"
        iter = graph.nodes()
        self.assertEqual(next(iter).kind(), "CustomNamespace::Custom")

    def test_autograd_onnx_fallthrough(self):
        # 定义一个自定义的torch.autograd.Function类CustomFunction
        class CustomFunction(torch.autograd.Function):
            @staticmethod
            def forward(ctx, input):
                # 在反向传播时保存输入，并对小于0的部分进行梯度清零
                ctx.save_for_backward(input)
                return input.clamp(min=0)

            @staticmethod
            def backward(ctx, grad_output):
                # 反向传播函数，计算梯度
                (input,) = ctx.saved_tensors
                grad_input = grad_output.clone()
                grad_input[input < 0] = 0
                return grad_input

        # 定义一个继承自torch.nn.Module的Custom类
        class Custom(torch.nn.Module):
            def forward(self, input):
                # 使用CustomFunction中的forward方法进行前向传播
                return CustomFunction.apply(input)

        model = Custom()
        batch = torch.FloatTensor(1, 3)

        # 将模型转换为计算图，使用输入batch，导出类型为ONNX_FALLTHROUGH，输入名称为"batch"，动态轴为[0, 1]
        graph, _, _ = self._model_to_graph(
            model,
            batch,
            operator_export_type=OperatorExportTypes.ONNX_FALLTHROUGH,
            input_names=["batch"],
            dynamic_axes={"batch": [0, 1]},
        )
        # 获取计算图的节点迭代器，断言第一个节点的类型为"prim::PythonOp"
        iter = graph.nodes()
        self.assertEqual(next(iter).kind(), "prim::PythonOp")
    def test_autograd_module_name(self):
        # 定义自定义的 autograd Function
        class CustomFunction(torch.autograd.Function):
            @staticmethod
            def forward(ctx, input):
                # 保存输入张量到上下文，以便在反向传播时使用
                ctx.save_for_backward(input)
                # 返回输入张量的非负部分
                return input.clamp(min=0)

            @staticmethod
            def backward(ctx, grad_output):
                # 获取保存的输入张量
                (input,) = ctx.saved_tensors
                # 克隆梯度输出张量
                grad_input = grad_output.clone()
                # 将小于零的输入部分的梯度设为零
                grad_input[input < 0] = 0
                return grad_input

        # 定义自定义的神经网络模块
        class Custom(torch.nn.Module):
            def forward(self, input):
                # 应用自定义的 autograd Function，并返回结果
                return CustomFunction.apply(input) + CustomFunction2.apply(input)

        # 创建自定义模型实例
        model = Custom()
        # 创建输入数据张量
        batch = torch.FloatTensor(1, 3)

        # 将模型转换为图形表示，用于导出 ONNX 格式
        graph, _, _ = self._model_to_graph(
            model,
            batch,
            operator_export_type=OperatorExportTypes.ONNX_FALLTHROUGH,
            input_names=["batch"],
            dynamic_axes={"batch": [0, 1]},
        )
        # 获取图中的第一个和第二个节点
        iter = graph.nodes()
        autograd1 = next(iter)
        autograd2 = next(iter)
        # 断言第一个节点是 PythonOp 类型
        self.assertEqual(autograd1.kind(), "prim::PythonOp")
        # 断言第二个节点是 PythonOp 类型
        self.assertEqual(autograd2.kind(), "prim::PythonOp")
        # 断言第一个节点的模块不等于第二个节点的模块
        self.assertNotEqual(autograd1.s("module"), autograd2.s("module"))

    def test_unused_initializers(self):
        # 定义一个简单的神经网络模型
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 添加一个转置卷积层和一个线性层
                self.conv2 = torch.nn.ConvTranspose2d(
                    16, 33, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(1, 1)
                )
                self.k_proj = torch.nn.Linear(5, 5, bias=True)

            def forward(self, x):
                # 在模型前向传播中使用转置卷积层
                x = self.conv2(x)
                return x

        # 创建输入数据张量
        x = torch.randn(20, 16, 50, 100)
        # 设置全局变量用于导出 ONNX 操作集版本和导出类型
        GLOBALS.export_onnx_opset_version = self.opset_version
        GLOBALS.operator_export_type = OperatorExportTypes.ONNX
        # 将模型转换为图形表示，并获取参数字典
        _, params_dict, __ = self._model_to_graph(
            Model(),
            (x,),
            do_constant_folding=False,
            operator_export_type=OperatorExportTypes.ONNX,
            input_names=["x"],
            dynamic_axes={"x": [0, 1, 2, 3]},
        )
        # 断言参数字典的长度为2，即有两个初始化器未被使用
        self.assertEqual(len(params_dict), 2)
    def test_scripting_param(self):
        # 定义一个继承自torch.nn.Module的自定义模块MyModule
        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 添加一个2维卷积层到模块中，输入通道数为3，输出通道数为16，卷积核大小为1，步长为2，填充为3，启用偏置
                self.conv = torch.nn.Conv2d(
                    3, 16, kernel_size=1, stride=2, padding=3, bias=True
                )
                # 添加一个2维批量归一化层到模块中，输入通道数为16，启用仿射变换
                self.bn = torch.nn.BatchNorm2d(16, affine=True)

            # 前向传播函数，接受输入x，经过卷积和批量归一化后返回结果
            def forward(self, x):
                x = self.conv(x)
                bn = self.bn(x)
                return bn

        # 使用torch.jit.script将MyModule实例化为一个脚本化模块
        model = torch.jit.script(MyModule())
        # 生成一个形状为(10, 3, 128, 128)的张量x，其中10是批量大小，3是通道数，128是高度和宽度
        x = torch.randn(10, 3, 128, 128)
        # 设置全局变量GLOBALS中的导出ONNX操作集版本号为self.opset_version
        GLOBALS.export_onnx_opset_version = self.opset_version
        # 设置全局变量GLOBALS中的操作员导出类型为ONNX
        GLOBALS.operator_export_type = OperatorExportTypes.ONNX
        # 调用self._model_to_graph方法将模型转换为图形表示
        graph, _, __ = self._model_to_graph(
            model,
            (x,),
            do_constant_folding=True,
            operator_export_type=OperatorExportTypes.ONNX,
            training=torch.onnx.TrainingMode.TRAINING,
            input_names=["x"],
            dynamic_axes={"x": [0, 1, 2, 3]},
        )

        # 获取图形输入参数的调试名称列表
        graph_input_params = [param.debugName() for param in graph.inputs()]
        # 遍历模型中的参数，并断言它们在图形输入参数列表中
        for item in dict(model.named_parameters()):
            self.assertIn(
                item,
                graph_input_params,
                "Graph parameter names does not match model parameters.",
            )

    def test_fuse_conv_bn(self):
        # 定义一个包含卷积和批量归一化层的模块Fuse
        class Fuse(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 添加一个2维卷积层到模块中，输入通道数为3，输出通道数为2，卷积核大小为1，步长为2，填充为3，启用偏置
                self.conv = torch.nn.Conv2d(
                    3, 2, kernel_size=1, stride=2, padding=3, bias=True
                )
                # 添加一个2维批量归一化层到模块中，输入通道数为2
                self.bn = torch.nn.BatchNorm2d(2)

            # 前向传播函数，接受输入x，经过卷积和批量归一化后返回结果
            def forward(self, x):
                out = self.conv(x)
                return self.bn(out)

        # 生成一个形状为(2, 3, 2, 2)的张量x，其中2是批量大小，3是通道数，2是高度和宽度，要求梯度计算
        x = torch.randn(2, 3, 2, 2, requires_grad=True)
        # 调用self._model_to_graph方法将Fuse模块转换为图形表示
        graph, _, __ = self._model_to_graph(
            Fuse(),
            (x,),
            training=TrainingMode.EVAL,
            input_names=["x"],
            dynamic_axes={"x": [0, 1, 2, 3]},
        )
        # 断言图中不存在onnx::BatchNormalization节点
        for node in graph.nodes():
            self.assertNotEqual(node.kind(), "onnx::BatchNormalization")
        # 断言图中只有一个节点
        self.assertEqual(len(list(graph.nodes())), 1)

    def test_fuse_resnet18(self):
        # 使用torchvision中的resnet18模型，权重未指定
        model = torchvision.models.resnet18(weights=None)
        # 生成一个形状为(2, 3, 224, 224)的张量x，其中2是批量大小，3是通道数，224是高度和宽度，要求梯度计算
        x = torch.randn(2, 3, 224, 224, requires_grad=True)
        # 调用self._model_to_graph方法将ResNet18模型转换为图形表示
        graph, _, __ = self._model_to_graph(
            model,
            (x,),
            training=TrainingMode.EVAL,
            input_names=["x"],
            dynamic_axes={"x": [0, 1, 2, 3]},
        )

        # 遍历图中的节点，并断言不存在onnx::BatchNormalization节点
        for node in graph.nodes():
            self.assertNotEqual(node.kind(), "onnx::BatchNormalization")
    def test_onnx_function_substitution_pass(self):
        @torch.jit.script
        def f(x: torch.Tensor, y: torch.Tensor):
            z = x - y
            return x + z
        # 定义一个 TorchScript 函数 f，对两个张量进行减法和加法操作

        class MyModule(torch.nn.Module):
            def forward(self, x, y):
                return f(x, y)
        # 定义一个 PyTorch 模块 MyModule，其 forward 方法调用了函数 f

        input_1 = torch.tensor([11])
        input_2 = torch.tensor([12])
        GLOBALS.export_onnx_opset_version = self.opset_version
        GLOBALS.operator_export_type = OperatorExportTypes.ONNX
        # 设置全局变量，指定导出的 ONNX opset 版本和操作符导出类型

        graph, _, __ = self._model_to_graph(
            MyModule(),
            (input_1, input_2),
            do_constant_folding=True,
            operator_export_type=OperatorExportTypes.ONNX,
            input_names=["input_1", "input_2"],
            dynamic_axes={"input_1": [0], "input_2": [0]},
        )
        # 将 MyModule 模块转换为计算图 graph，使用给定的输入和其他参数

        # 检查图中是否移除了表示 TorchScript 函数 `f` 的 prim::Constant 节点，
        # 并且 prim::CallFunction 节点是否被内联替换为 onnx::Sub 和 onnx::Add 节点
        for node in graph.nodes():
            self.assertNotEqual(node.kind(), "prim::Constant")
        self.assertEqual(
            len(list(graph.nodes())), 2
        )  # 只剩下 onnx::Sub 和 onnx::Add 节点。

    def test_onnx_value_name(self):
        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.in_weight = torch.nn.Parameter(torch.Tensor(3, 3))
                self.in_bias = torch.nn.Parameter(torch.Tensor(3))
                # 初始化模块参数：in_weight 和 in_bias

            def forward(self, x):
                start = 0
                end = None
                weight = self.in_weight
                bias = self.in_bias
                weight = weight[start:end, :]
                if bias is not None:
                    bias = bias[start:end]
                return torch.nn.functional.linear(x, weight, bias)
                # 在 forward 方法中，使用模块的参数 weight 和 bias 进行线性变换

        model = MyModule()
        x = torch.randn(3, 3)
        f = io.BytesIO()
        # 创建模块实例 model，生成输入张量 x 和字节流 f

        model.eval()
        torch.onnx.export(
            model,
            (x,),
            f,
            opset_version=self.opset_version,
            keep_initializers_as_inputs=True,
        )
        # 导出模型为 ONNX 格式，输出到字节流 f，保留初始器作为输入

        graph = onnx.load(io.BytesIO(f.getvalue()))
        # 加载导出的 ONNX 图

        self.assertEqual(graph.graph.input[1].name, "in_weight")
        self.assertEqual(graph.graph.input[2].name, "in_bias")
        # 检查 ONNX 图的第二个和第三个输入是否分别命名为 "in_weight" 和 "in_bias"
    def test_onnx_node_naming(self):
        # 定义一个名为 MainModule 的子类，继承自 torch.nn.Module
        class MainModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 初始化四个线性层模块
                self._module_1 = torch.nn.Linear(10, 10)
                self._module_2 = torch.nn.Linear(10, 10)
                self._module_3 = torch.nn.Linear(10, 10)
                self._module_4 = torch.nn.Linear(10, 10)

            # 前向传播函数定义
            def forward(self, x):
                y = self._module_1(x)  # 第一层线性层的前向计算
                z = self._module_2(y)  # 第二层线性层的前向计算
                z = self._module_3(y * z)  # 第三层线性层的前向计算
                z = self._module_4(y * z)  # 第四层线性层的前向计算
                return z

        # 创建 MainModule 实例
        module = MainModule()
        # 参考节点名称列表
        ref_node_names = [
            "/_module_1/Gemm",
            "/_module_2/Gemm",
            "/_module_3/Gemm",
            "/_module_4/Gemm",
            "/Mul",
            "/Mul_1",
        ]
        # 创建一个 BytesIO 对象
        f = io.BytesIO()

        # 导出 ONNX 模型，输出节点名称为 "y"
        torch.onnx.export(module, torch.ones(1, 10), f, output_names=["y"])
        # 加载导出的 ONNX 模型
        onnx_model = onnx.load(io.BytesIO(f.getvalue()))
        # 遍历模型中的节点，验证节点名称是否在参考节点名称列表中
        for n in onnx_model.graph.node:
            self.assertIn(n.name, ref_node_names)

        # 使用 Torch JIT 脚本导出 ONNX 模型，输出节点名称为 "y"
        torch.onnx.export(
            torch.jit.script(module), torch.ones(1, 10), f, output_names=["y"]
        )
        # 再次加载导出的 ONNX 模型
        onnx_model = onnx.load(io.BytesIO(f.getvalue()))
        # 再次遍历模型中的节点，验证节点名称是否在参考节点名称列表中
        for n in onnx_model.graph.node:
            self.assertIn(n.name, ref_node_names)
    # 定义一个测试方法，用于检验在 TorchScript 下是否去除重复的初始化器
    def _test_deduplicate_initializers(self, torchscript=False):
        # 定义一个内部模块类 MyModule，继承自 torch.nn.Module
        class MyModule(torch.nn.Module):
            # 初始化方法
            def __init__(self):
                super().__init__()
                # 初始化两个线性层，每个都是输入和输出大小为 3
                self.layer1 = torch.nn.Linear(3, 3)
                self.layer2 = torch.nn.Linear(3, 3)

                # 重用第一个层作为第三个层
                self.layer3 = self.layer1

                # 将第二个层的权重与第一个层的权重相等
                self.layer2.weight = self.layer1.weight
                # 将第一个层的偏置与第二个层的偏置相等
                self.layer1.bias = self.layer2.bias

                # 创建一个具有相同数值的不同张量的参数
                self.param1 = torch.nn.Parameter(torch.tensor([1.0, 2.0, 3.0]))
                self.param2 = torch.nn.Parameter(torch.tensor([1.0, 2.0, 3.0]))

            # 前向传播方法
            def forward(self, x):
                # 返回 layer3(layer2(layer1(x))) + param1 + param2 的结果
                return (
                    self.layer3(self.layer2(self.layer1(x))) + self.param1 + self.param2
                )

        # 根据 torchscript 参数决定是否将 MyModule 脚本化
        model = torch.jit.script(MyModule()) if torchscript else MyModule()

        # 生成一个大小为 3x3 的随机张量 x
        x = torch.randn(3, 3)
        # 获取模型中命名参数的集合
        param_name_set = {k for k, _ in model.named_parameters()}

        # 测试训练模式下的导出
        model.train()
        # 创建一个字节流对象 f
        f = io.BytesIO()
        # 将模型导出为 ONNX 格式，使用训练模式，并指定 opset 版本
        torch.onnx.export(
            model,
            (x,),
            f,
            training=TrainingMode.TRAINING,
            opset_version=self.opset_version,
        )
        # 加载导出的 ONNX 模型并获取初始化器的名称集合
        graph = onnx.load(io.BytesIO(f.getvalue()))
        self.assertSetEqual({i.name for i in graph.graph.initializer}, param_name_set)

        # 再次测试训练模式下的导出，但保留训练模式
        model.train()
        f = io.BytesIO()
        torch.onnx.export(
            model,
            (x,),
            f,
            training=TrainingMode.PRESERVE,
            opset_version=self.opset_version,
        )
        graph = onnx.load(io.BytesIO(f.getvalue()))
        self.assertSetEqual({i.name for i in graph.graph.initializer}, param_name_set)

        # 测试评估模式下的导出
        model.eval()
        f = io.BytesIO()
        torch.onnx.export(model, (x,), f, opset_version=self.opset_version)
        graph = onnx.load(io.BytesIO(f.getvalue()))
        # 移除 param2 后检查初始化器的名称集合是否与 param_name_set 相等
        param_name_set.remove("param2")
        self.assertSetEqual({i.name for i in graph.graph.initializer}, param_name_set)

    # 测试不使用 TorchScript 的情况下去除重复初始化器
    def test_deduplicate_initializers(self):
        self._test_deduplicate_initializers(torchscript=False)

    # 测试使用 TorchScript 的情况下去除重复初始化器
    def test_deduplicate_initializers_torchscript(self):
        self._test_deduplicate_initializers(torchscript=True)

    @skipIfNoCuda
    def test_deduplicate_initializers_diff_devices(self):
        # 定义一个测试用例，用于验证在不同设备上初始化参数时的去重行为
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 在 CPU 上初始化参数 w_cpu
                self.w_cpu = torch.nn.Parameter(
                    torch.ones(3, device=torch.device("cpu"))
                )
                # 在 CUDA 上初始化参数 w_cuda
                self.w_cuda = torch.nn.Parameter(
                    torch.ones(3, device=torch.device("cuda"))
                )

            def forward(self, x, y):
                # 模型的前向传播，返回 x + w_cpu 和 y + w_cuda
                return x + self.w_cpu, y + self.w_cuda

        # 创建 CPU 和 CUDA 上的输入数据
        x = torch.randn(3, 3, device=torch.device("cpu"))
        y = torch.randn(3, 3, device=torch.device("cuda"))
        
        # 创建一个字节流对象
        f = io.BytesIO()
        
        # 导出模型到 ONNX 格式，并将结果写入字节流
        torch.onnx.export(Model(), (x, y), f, opset_version=self.opset_version)
        
        # 从导出的字节流中加载 ONNX 图形
        graph = onnx.load(io.BytesIO(f.getvalue()))
        
        # 断言模型初始化器的名称集合，验证是否只有 "w_cpu"
        self.assertSetEqual({i.name for i in graph.graph.initializer}, {"w_cpu"})

    def test_duplicated_output_node(self):
        # 定义一个测试用例，用于验证在模型输出节点重复的情况下导出 ONNX 格式的正确性
        class DuplicatedOutputNet(torch.nn.Module):
            def __init__(self, input_size, num_classes):
                super().__init__()
                self.fc1 = torch.nn.Linear(input_size, num_classes)

            def forward(self, input0, input1):
                # 模型的前向传播，输出节点 out1 和 out2 都使用相同的 self.fc1
                out1 = self.fc1(input0)
                out2 = self.fc1(input1)
                return out1, out1, out2, out1, out2

        N, D_in, H, D_out = 64, 784, 500, 10
        # 创建一个重复输出节点的 PyTorch 模型
        pt_model = DuplicatedOutputNet(D_in, D_out)

        # 创建一个字节流对象
        f = io.BytesIO()
        x = torch.randn(N, D_in)
        
        # 定义动态轴，用于 ONNX 导出
        dynamic_axes = {
            "input0": {0: "input0_dim0", 1: "input0_dim1"},
            "input1": {0: "input1_dim0", 1: "input1_dim1"},
            "output-0": {0: "output-0_dim0", 1: "output-0_dim1"},
            "output-1": {0: "output-1_dim0", 1: "output-1_dim1"},
            "output-2": {0: "output-2_dim0", 1: "output-2_dim1"},
            "output-3": {0: "output-3_dim0", 1: "output-3_dim1"},
            "output-4": {0: "output-4_dim0", 1: "output-4_dim1"},
        }

        # 导出模型到 ONNX 格式，并将结果写入字节流
        torch.onnx.export(
            pt_model,
            (x, x),
            f,
            input_names=["input0", "input1"],
            output_names=["output-0", "output-1", "output-2", "output-3", "output-4"],
            do_constant_folding=False,
            training=torch.onnx.TrainingMode.TRAINING,
            dynamic_axes=dynamic_axes,
            verbose=True,
            keep_initializers_as_inputs=True,
        )

        # 从导出的字节流中加载 ONNX 图形
        graph = onnx.load(io.BytesIO(f.getvalue()))
        
        # 验证输入名称和输出名称
        self.assertEqual(graph.graph.input[0].name, "input0")
        self.assertEqual(graph.graph.input[1].name, "input1")
        for i in range(5):
            self.assertEqual(graph.graph.output[i].name, f"output-{i}")
        
        # 验证 ONNX 图中节点的操作类型
        self.assertEqual(graph.graph.node[0].op_type, "Gemm")
        self.assertEqual(graph.graph.node[1].op_type, "Identity")
        self.assertEqual(graph.graph.node[2].op_type, "Identity")
        self.assertEqual(graph.graph.node[3].op_type, "Gemm")
        self.assertEqual(graph.graph.node[4].op_type, "Identity")
    # 定义一个测试方法，用于验证忽略共享权重去重复中的上采样比例
    def test_deduplicate_ignore_upsample_scale(self):
        # 上采样比例是一个常数，不是模型参数，因此在共享权重去重复过程中应该被忽略。
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 定义模型中的两个上采样层，每个层的放大比例为2
                self.upsample_1 = torch.nn.Upsample(scale_factor=2)
                self.upsample_2 = torch.nn.Upsample(scale_factor=2)

            def forward(self, x):
                # 模型的前向传播，对输入数据进行两次上采样操作
                return self.upsample_1(x), self.upsample_2(x)

        # 创建一个字节流对象
        f = io.BytesIO()
        # 生成一个随机张量作为模型的输入数据
        x = torch.randn(1, 32, 224, 224)
        # 将模型导出为ONNX格式并写入字节流中
        torch.onnx.export(Model(), x, f)
        # 从字节流中加载ONNX模型
        onnx_model = onnx.load(io.BytesIO(f.getvalue()))
        # 查找ONNX图中所有的resize操作节点（对应PyTorch中的upsample操作）
        resize_nodes = [n for n in onnx_model.graph.node if n.op_type == "Resize"]
        # 断言resize操作节点的数量为2（因为模型中有两个上采样层）
        self.assertEqual(len(resize_nodes), 2)
        # 遍历每个resize节点
        for resize_node in resize_nodes:
            # 查找与当前resize节点连接的scale节点（该节点用于指定上采样的比例）
            scale_node = [
                n for n in onnx_model.graph.node if n.output[0] == resize_node.input[2]
            ]
            # 断言每个resize节点都有且仅有一个与之连接的scale节点
            self.assertEqual(len(scale_node), 1)
            # 断言连接的scale节点的操作类型为"Constant"
            self.assertEqual(scale_node[0].op_type, "Constant")

    # 定义一个测试方法，用于验证错误的符号注册
    def test_bad_symbolic_registration(self):
        # 设置ONNX操作集的版本号
        _onnx_opset_version = 9

        # 自定义符号化函数"cat"，用于在ONNX图中执行拼接操作
        @parse_args("v")
        def cat(g, tensor_list, dim):
            tensors = _unpack_list(tensor_list)
            return g.op("Concat", *tensors, axis_i=dim)

        # 将自定义符号化函数"cat"注册到ONNX模型中
        torch.onnx.register_custom_op_symbolic("::cat", cat, _onnx_opset_version)

        # 定义一个简单的模型，用于测试拼接操作
        class CatModel(torch.nn.Module):
            def forward(self, x):
                return torch.cat((x, x, x), 0)

        # 创建一个CatModel实例
        model = CatModel()
        # 生成一个随机张量作为模型的输入数据
        x = torch.randn(2, 3)
        # 创建一个字节流对象
        f = io.BytesIO()
        # 断言在导出ONNX模型时会引发预期的AssertionError异常
        self.assertExpectedRaisesInline(
            AssertionError,
            lambda: torch.onnx.export(
                model, (x,), f, opset_version=_onnx_opset_version
            ),
            (
                "A mismatch between the number of arguments (2) and their descriptors (1) was found at symbolic function "
                "'cat'. If you believe this is not due to custom symbolic implementation within your code or an external "
                "library, please file an issue at https://github.com/pytorch/pytorch/issues/new?template=bug-report.yml to "
                "report this bug."
            ),
        )
        # 注销自定义符号化函数"cat"
        torch.onnx.unregister_custom_op_symbolic("::cat", _onnx_opset_version)
# 如果当前脚本作为主程序执行（而不是被导入），则执行以下代码块
if __name__ == "__main__":
    # 调用 common_utils 模块中的 run_tests 函数，用于执行测试
    common_utils.run_tests()
```