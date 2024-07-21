# `.\pytorch\test\onnx\dynamo\test_registry_dispatcher.py`

```py
# Owner(s): ["module: onnx"]
"""Unit tests for the internal registration wrapper module."""
# 引入未来版本兼容模块
from __future__ import annotations

# 导入必要的库和模块
import operator
from typing import TypeVar, Union

# 引入onnxscript，忽略导入类型错误
import onnxscript  # type: ignore[import]
# 从onnxscript中导入特定符号，忽略导入类型错误
from onnxscript import BFLOAT16, DOUBLE, FLOAT, FLOAT16  # type: ignore[import]
# 从onnxscript的torch_lib模块中导入ops，忽略导入类型错误
from onnxscript.function_libs.torch_lib import ops  # type: ignore[import]
# 从onnxscript的onnx_opset模块中导入opset15作为op，忽略导入类型错误
from onnxscript.onnx_opset import opset15 as op  # type: ignore[import]

# 导入torch和torch.fx模块
import torch
import torch.fx
# 从torch.onnx._internal.diagnostics中导入infra模块
from torch.onnx._internal.diagnostics import infra
# 从torch.onnx._internal.fx中导入analysis, diagnostics, onnxfunction_dispatcher, registration模块
from torch.onnx._internal.fx import (
    analysis,
    diagnostics,
    onnxfunction_dispatcher,
    registration,
)
# 从torch.testing._internal中导入common_utils模块
from torch.testing._internal import common_utils

# TODO: this can only be global. https://github.com/microsoft/onnxscript/issues/805
# 定义一个类型变量TCustomFloat，它是FLOAT16、FLOAT、DOUBLE、BFLOAT16中的一种
TCustomFloat = TypeVar("TCustomFloat", bound=Union[FLOAT16, FLOAT, DOUBLE, BFLOAT16])


# 定义测试类TestRegistration，继承自common_utils.TestCase
class TestRegistration(common_utils.TestCase):
    # 设置测试环境的初始化方法
    def setUp(self) -> None:
        # 创建一个torch.onnx.OnnxRegistry实例，并赋值给self.registry
        self.registry = torch.onnx.OnnxRegistry()
        # 创建一个onnxscript.values.Opset实例，自定义域名为"custom"，版本号为1，并赋值给self.custom_domain
        self.custom_domain = onnxscript.values.Opset(domain="custom", version=1)

    # 设置测试环境的清理方法
    def tearDown(self) -> None:
        # 创建一个registration.OpName实例，表示操作名称，命名空间为"test"，操作名称为"test_op"
        internal_name_instance = registration.OpName.from_name_parts(
            namespace="test", op_name="test_op"
        )
        # 从self.registry中移除internal_name_instance对应的条目
        self.registry._registry.pop(internal_name_instance, None)

    # 定义测试方法，验证注册自定义操作是否成功注册自定义函数
    def test_register_custom_op_registers_custom_function(self):
        # 断言self.registry中未注册操作"test"，"test_op"，"default"
        self.assertFalse(self.registry.is_registered_op("test", "test_op", "default"))

        # 定义一个装饰器，将custom_add函数注册到self.custom_domain的命名空间中
        @onnxscript.script(self.custom_domain)
        def custom_add(x, y):
            return op.Add(x, y)

        # 使用self.registry注册custom_add函数，操作为"test"，操作名称为"test_op"，默认版本为"default"
        self.registry.register_op(custom_add, "test", "test_op", "default")
        # 断言self.registry中已注册操作"test"，"test_op"，"default"
        self.assertTrue(self.registry.is_registered_op("test", "test_op", "default"))

        # 测试获取操作函数组
        function_group = self.registry.get_op_functions("test", "test_op", "default")
        # 断言function_group不为None
        self.assertIsNotNone(function_group)
        # 断言function_group中所有函数的onnx_function属性为{custom_add}
        self.assertEqual({func.onnx_function for func in function_group}, {custom_add})  # type: ignore[arg-type]
   `
    # 测试自定义的 ONNX 符号连接现有函数是否正常工作
    def test_custom_onnx_symbolic_joins_existing_function(self):
        # 断言在注册表中“test”命名空间下的“test_op”操作尚未注册
        self.assertFalse(self.registry.is_registered_op("test", "test_op"))

        # 使用装饰器创建一个原始的 ONNX 脚本函数 test_original
        @onnxscript.script(self.custom_domain)
        def test_original(x, y):
            return op.Add(x, y)

        # 创建操作名称实例 internal_name_instance，指定命名空间为“test”，操作名称为“test_op”，重载为"default"
        internal_name_instance = registration.OpName.from_name_parts(
            namespace="test", op_name="test_op", overload="default"
        )
        # 创建一个注册的 ONNX 函数 symbolic_fn，将 test_original 绑定到操作全名上
        symbolic_fn = registration.ONNXFunction(
            test_original, op_full_name=internal_name_instance.qualified_name()
        )
        # 将 symbolic_fn 注册到注册表中
        self.registry._register(internal_name_instance, symbolic_fn)
        # 断言在注册表中“test”命名空间下的“test_op”操作已经注册
        self.assertTrue(self.registry.is_registered_op("test", "test_op"))

        # 使用装饰器创建一个定制的 ONNX 脚本函数 test_custom
        @onnxscript.script(self.custom_domain)
        def test_custom(x, y):
            return op.Add(x, y)

        # 将 test_custom 函数注册到注册表中的“test”命名空间下的“test_op”操作
        self.registry.register_op(test_custom, "test", "test_op")

        # 获取注册表中“test”命名空间下“test_op”操作的函数组
        function_group = self.registry.get_op_functions("test", "test_op")
        # 断言函数组不为空
        assert function_group is not None
        # 断言函数组中的函数顺序与期望顺序一致（列表比较）
        self.assertEqual(
            [func.onnx_function for func in function_group],
            [test_original, test_custom],
        )
    # 定义一个测试函数，用于分析在缺失 ATen 操作时不支持的节点情况
    def test_unsupported_nodes_analysis_with_missing_aten_op(self):
        # 模拟不支持的节点，创建 ATen 操作的名称对象
        aten_mul_tensor = registration.OpName.from_name_parts(
            namespace="aten", op_name="mul", overload="Tensor"
        )
        aten_mul_default = registration.OpName.from_name_parts(
            namespace="aten", op_name="mul"
        )
        aten_add_tensor = registration.OpName.from_name_parts(
            namespace="aten", op_name="add", overload="Tensor"
        )
        aten_add_default = registration.OpName.from_name_parts(
            namespace="aten", op_name="add"
        )

        # 从注册表中移除上述创建的 ATen 操作名称对象
        self.registry._registry.pop(aten_mul_tensor)
        self.registry._registry.pop(aten_mul_default)
        self.registry._registry.pop(aten_add_tensor)
        self.registry._registry.pop(aten_add_default)

        # 创建诊断上下文对象，用于记录诊断信息
        diagnostic_context = diagnostics.DiagnosticContext(
            "torch.onnx.dynamo_export", torch.__version__
        )

        # 创建 OnnxFunctionDispatcher 对象，用于分派 ONNX 函数
        dispatcher = onnxfunction_dispatcher.OnnxFunctionDispatcher(
            self.registry, diagnostic_context
        )

        # 创建一个空的 Torch FX 图对象
        graph: torch.fx.Graph = torch.fx.Graph()

        # 创建表示输入节点 x 的 Torch FX 节点，并设置节点元数据
        x: torch.fx.Node = graph.create_node("placeholder", "x")
        x.meta["val"] = torch.tensor(3.0)

        # 创建表示乘法操作节点 b 的 Torch FX 节点
        b: torch.fx.Node = graph.create_node(
            "call_function", target=torch.ops.aten.mul.Tensor, args=(x, x)
        )

        # 创建表示加法操作节点 c 的 Torch FX 节点
        c: torch.fx.Node = graph.create_node(
            "call_function", target=torch.ops.aten.add.Tensor, args=(b, b)
        )

        # 将节点 c 设置为输出节点
        output: torch.fx.Node = graph.output(c)

        # 创建一个 Torch FX 模块对象
        module = torch.fx.GraphModule(torch.nn.Module(), graph)

        # 断言以下代码块会抛出 infra.RuntimeErrorWithDiagnostic 异常
        with self.assertRaises(infra.RuntimeErrorWithDiagnostic):
            # 进行不支持的 FX 节点分析，期望在错误级别（ERROR）触发分析
            analysis.UnsupportedFxNodesAnalysis(
                diagnostic_context, module, dispatcher
            ).analyze(infra.levels.ERROR)

        try:
            # 尝试进行不支持的 FX 节点分析
            analysis.UnsupportedFxNodesAnalysis(
                diagnostic_context, module, dispatcher
            ).analyze(infra.levels.ERROR)
        except infra.RuntimeErrorWithDiagnostic as e:
            # 断言异常消息包含不支持的 FX 节点信息
            self.assertIn(
                "Unsupported FX nodes: {'call_function': ['aten.mul.Tensor', 'aten.add.Tensor']}.",
                e.diagnostic.message,
            )
# 使用装饰器实例化参数化测试，将该类注册为测试类
@common_utils.instantiate_parametrized_tests
class TestDispatcher(common_utils.TestCase):
    # 在每个测试方法运行前执行的设置方法
    def setUp(self):
        # 创建一个新的 ONNX 注册表实例
        self.registry = torch.onnx.OnnxRegistry()
        # 创建诊断上下文对象，用于诊断信息
        self.diagnostic_context = diagnostics.DiagnosticContext(
            "torch.onnx.dynamo_export", torch.__version__
        )
        # 创建 ONNX 函数调度器对象，传入注册表和诊断上下文
        self.dispatcher = onnxfunction_dispatcher.OnnxFunctionDispatcher(
            self.registry, self.diagnostic_context
        )

    # 参数化测试方法，用来测试不同的输入组合
    @common_utils.parametrize(
        "node, expected_name",
        [
            # 子测试1：模拟一个 torch.fx.Node 对象，测试获取操作名称
            common_utils.subtest(
                (
                    torch.fx.Node(
                        graph=torch.fx.Graph(),
                        name="aten::add.Tensor",
                        op="call_function",
                        target=torch.ops.aten.add.Tensor,  # type: ignore[attr-defined]
                        args=(torch.tensor(3), torch.tensor(4)),
                        kwargs={},
                    ),
                    ("aten", "add", "Tensor"),
                ),
                name="get_Opoverload_name",  # 子测试名称
            ),
            # 子测试2：模拟另一个 torch.fx.Node 对象，测试获取操作名称
            common_utils.subtest(
                (
                    torch.fx.Node(
                        graph=torch.fx.Graph(),
                        name="aten::sym_size",
                        op="call_function",
                        target=torch.ops.aten.sym_size,
                        args=(),
                        kwargs={},
                    ),
                    ("aten", "sym_size", None),
                ),
                name="get_Opoverloadpacket_name",  # 子测试名称
            ),
            # 子测试3：模拟使用内置操作的 torch.fx.Node 对象，测试获取操作名称
            common_utils.subtest(
                (
                    torch.fx.Node(
                        graph=torch.fx.Graph(),
                        name="builtin_add",
                        op="call_function",
                        target=operator.add,
                        args=(1, 2),
                        kwargs={},
                    ),
                    ("_operator", "add", None),
                ),
                name="get_builtin_op_name",  # 子测试名称
            ),
        ],
    )
    # 测试方法：验证对于支持的 FX 节点，获取操作名称的正确性
    def test_get_aten_name_on_supported_fx_node(
        self, node: torch.fx.Node, expected_name: str
    ):
        # 从期望的名称部分创建 OpName 对象
        expected_name_class = registration.OpName.from_name_parts(*expected_name)
        # 断言：调度器根据节点和诊断上下文获取的操作名称应与期望的名称对象相等
        self.assertEqual(
            self.dispatcher._get_aten_name(node, self.diagnostic_context),
            expected_name_class,
        )
    @common_utils.parametrize(
        "node",
        [  # 参数化测试，使用不同的节点进行测试
            common_utils.subtest(
                torch.fx.Node(  # 创建一个 PyTorch FX 图节点对象
                    graph=torch.fx.Graph(),  # 使用空图创建节点
                    name="aten::add",  # 节点名称为 'aten::add'
                    op="call_function",  # 节点操作为调用函数
                    target=torch.ops.aten.add,  # 目标函数为 torch.ops.aten.add
                    args=(),  # 无位置参数
                    kwargs={},  # 无关键字参数
                ),
                name="unsupported_Opoverloadpacket_name",  # 子测试名称为 'unsupported_Opoverloadpacket_name'
            ),
            common_utils.subtest(
                torch.fx.Node(  # 创建另一个 PyTorch FX 图节点对象
                    graph=torch.fx.Graph(),  # 使用空图创建节点
                    name="builtin_add",  # 节点名称为 'builtin_add'
                    op="call_function",  # 节点操作为调用函数
                    target=operator.add,  # 目标函数为 Python 内置的 operator.add
                    args=("A", "B"),  # 位置参数为 'A' 和 'B'
                    kwargs={},  # 无关键字参数
                ),
                name="unsupported_input_dtypes_for_builtin_op",  # 子测试名称为 'unsupported_input_dtypes_for_builtin_op'
            ),
            common_utils.subtest(
                torch.fx.Node(  # 创建第三个 PyTorch FX 图节点对象
                    graph=torch.fx.Graph(),  # 使用空图创建节点
                    name="aten::made_up_node",  # 节点名称为 'aten::made_up_node'
                    op="call_function",  # 节点操作为调用函数
                    target=lambda: None,  # 目标函数为 lambda 函数返回 None
                    args=(),  # 无位置参数
                    kwargs={},  # 无关键字参数
                ),
                name="unsupported_target_function",  # 子测试名称为 'unsupported_target_function'
            ),
        ],
    )
    def test_get_aten_name_on_unsupported_fx_node(self, node: torch.fx.Node):
        with self.assertRaises(RuntimeError):  # 断言引发 RuntimeError 异常
            self.dispatcher._get_aten_name(node, self.diagnostic_context)  # 调用 _get_aten_name 方法获取节点的名称

    def test_get_function_overloads_gives_overload_fall_back_default(self):
        # 测试回退到默认操作名称
        node_overload = torch.fx.Node(  # 创建 PyTorch FX 图节点对象
            graph=torch.fx.Graph(),  # 使用空图创建节点
            name="aten::add.Tensor",  # 节点名称为 'aten::add.Tensor'
            op="call_function",  # 节点操作为调用函数
            target=torch.ops.aten.add.Tensor,  # 目标函数为 torch.ops.aten.add.Tensor
            args=(torch.tensor(3), torch.tensor(4)),  # 位置参数为 torch.tensor(3) 和 torch.tensor(4)
            kwargs={},  # 无关键字参数
        )
        node_overloadpacket = torch.fx.Node(  # 创建另一个 PyTorch FX 图节点对象
            graph=torch.fx.Graph(),  # 使用空图创建节点
            name="aten::add",  # 节点名称为 'aten::add'
            op="call_function",  # 节点操作为调用函数
            target=torch.ops.aten.add.Tensor,  # 目标函数为 torch.ops.aten.add.Tensor
            args=(),  # 无位置参数
            kwargs={},  # 无关键字参数
        )

        self.assertEqual(
            self.dispatcher.get_function_overloads(  # 断言获取的函数重载相等
                node_overload, self.diagnostic_context
            ),
            self.dispatcher.get_function_overloads(
                node_overloadpacket,
                self.diagnostic_context,
            ),
        )

        # 非注册的操作
        unsupported_op_node = torch.fx.Node(  # 创建 PyTorch FX 图节点对象
            graph=torch.fx.Graph(),  # 使用空图创建节点
            name="aten::made_up_node",  # 节点名称为 'aten::made_up_node'
            op="call_function",  # 节点操作为调用函数
            target=lambda: None,  # 目标函数为 lambda 函数返回 None
            args=(),  # 无位置参数
            kwargs={},  # 无关键字参数
        )
        with self.assertRaises(RuntimeError):  # 断言引发 RuntimeError 异常
            self.dispatcher.get_function_overloads(  # 调用 get_function_overloads 方法获取函数重载
                unsupported_op_node,
                self.diagnostic_context,
            )
    @common_utils.parametrize(
        "node",
        [
            common_utils.subtest(
                torch.fx.Node(
                    graph=torch.fx.Graph(),
                    name="aten::add.Tensor",
                    op="call_function",
                    target=torch.ops.aten.add.Tensor,  # type: ignore[attr-defined]
                    args=(torch.tensor(3.0), torch.tensor(4.0)),
                    kwargs={},
                ),
                name="nearest_match",
            ),
            common_utils.subtest(
                torch.fx.Node(
                    graph=torch.fx.Graph(),
                    name="aten::add.Tensor",
                    op="call_function",
                    target=torch.ops.aten.add.Tensor,  # type: ignore[attr-defined]
                    args=(torch.tensor(3.0), torch.tensor(4.0)),
                    kwargs={"alpha": 1},
                ),
                name="perfect_match_with_kwargs",
            ),
        ],
    )
    def test_find_the_perfect_or_nearest_match_onnxfunction_gives_custom_ops_precedence(
        self, node
    ):
        # 定义自定义域
        custom_domain = onnxscript.values.Opset(domain="custom", version=1)

        @onnxscript.script(custom_domain)
        def test_custom_op(
            x: TCustomFloat, y: TCustomFloat, alpha: int = 1
        ) -> TCustomFloat:
            return op.Add(x, y)

        @onnxscript.script(custom_domain)
        def test_default_op(
            x: TCustomFloat, y: TCustomFloat, alpha: int = 1
        ) -> TCustomFloat:
            return op.Add(x, y)

        # 定义操作的全名
        op_full_name = "test::test_op"

        # 创建自定义重载列表
        custom_overloads = [
            registration.ONNXFunction(
                test_custom_op, op_full_name=op_full_name, is_custom=True
            )
        ]
        # 创建功能重载列表，包括默认操作和自定义操作
        function_overloads = [
            registration.ONNXFunction(test_default_op, op_full_name=op_full_name)
        ] + custom_overloads

        # 使用调度程序找到最佳或最接近的 ONNX 函数
        symbolic_fn = self.dispatcher._find_the_perfect_or_nearest_match_onnxfunction(
            node,
            function_overloads,
            node.args,
            node.kwargs,
            self.diagnostic_context,
        )
        # 断言找到的符号函数与自定义操作函数相等
        self.assertEqual(symbolic_fn, test_custom_op)
    # 使用 common_utils.parametrize 装饰器来参数化测试用例，参数为"node"
    @common_utils.parametrize(
        "node",
        [
            # 定义第一个子测试用例，调用 common_utils.subtest 创建
            common_utils.subtest(
                # 创建一个 torch.fx.Node 对象，表示一个 PyTorch 的 FX 图节点
                torch.fx.Node(
                    graph=torch.fx.Graph(),  # 创建一个空的 FX 图
                    name="aten::add.Tensor",  # 节点的操作名为 "aten::add.Tensor"
                    op="call_function",  # 表示调用函数的操作类型
                    target=torch.ops.aten.add.Tensor,  # 目标函数为 torch.ops.aten.add.Tensor
                    args=(torch.tensor(3.0), torch.tensor(4.0)),  # 调用函数的位置参数
                    kwargs={"attr": None},  # 调用函数的关键字参数
                ),
                name="perfect_match_with_ignoring_none_attribute",  # 子测试用例的名称
            ),
            # 定义第二个子测试用例，调用 common_utils.subtest 创建
            common_utils.subtest(
                # 创建一个 torch.fx.Node 对象，表示一个 PyTorch 的 FX 图节点
                torch.fx.Node(
                    graph=torch.fx.Graph(),  # 创建一个空的 FX 图
                    name="aten::add.Tensor",  # 节点的操作名为 "aten::add.Tensor"
                    op="call_function",  # 表示调用函数的操作类型
                    target=torch.ops.aten.add.Tensor,  # 目标函数为 torch.ops.aten.add.Tensor
                    args=(torch.tensor(3.0), torch.tensor(4.0)),  # 调用函数的位置参数
                    kwargs={"unrelated": None},  # 调用函数的关键字参数
                ),
                name="perfect_match_with_ignoring_unrelated_none_attribute",  # 子测试用例的名称
            ),
        ],
    )
    # 定义测试方法 test_find_the_perfect_or_nearest_match_onnxfunction_ignores_attribute_with_none
    def test_find_the_perfect_or_nearest_match_onnxfunction_ignores_attribute_with_none(
        self, node
    ):
        # 创建一个自定义的域对象 onnxscript.values.Opset，表示自定义域和版本
        custom_domain = onnxscript.values.Opset(domain="custom", version=1)

        # 定义一个在自定义域 custom_domain 中的脚本函数 test_op_attribute
        @onnxscript.script(custom_domain)
        def test_op_attribute(
            x: TCustomFloat, y: TCustomFloat, attr: int
        ) -> TCustomFloat:
            return op.Add(x, y)  # 执行自定义操作 op.Add，并返回结果

        # 定义一个在自定义域 custom_domain 中的脚本函数 test_op
        @onnxscript.script(custom_domain)
        def test_op(x: TCustomFloat, y: TCustomFloat) -> TCustomFloat:
            return op.Add(x, y)  # 执行自定义操作 op.Add，并返回结果

        op_full_name = "test::test_op"  # 定义操作的完整名称

        # 创建一个包含两个 ONNXFunction 注册的列表 function_overloads
        function_overloads = [
            registration.ONNXFunction(test_op_attribute, op_full_name=op_full_name),
            registration.ONNXFunction(test_op, op_full_name=op_full_name),
        ]

        # 调用 self.dispatcher._find_the_perfect_or_nearest_match_onnxfunction 方法，
        # 寻找最佳或最接近匹配的 ONNXFunction
        symbolic_fn = self.dispatcher._find_the_perfect_or_nearest_match_onnxfunction(
            node,
            function_overloads,
            node.args,
            node.kwargs,
            self.diagnostic_context,
        )

        # 断言找到的符号函数 symbolic_fn 等于 test_op
        self.assertEqual(symbolic_fn, test_op)
    # 使用 common_utils.parametrize 装饰器定义参数化测试方法，用于测试不同的输入情况
    @common_utils.parametrize(
        "node",
        [
            # 定义第一个子测试，创建一个代表 Torch FX 节点的对象
            common_utils.subtest(
                torch.fx.Node(
                    graph=torch.fx.Graph(),
                    name="aten::add.Tensor",
                    op="call_function",
                    target=torch.ops.aten.add.Tensor,  # 操作目标为 torch.ops.aten.add.Tensor
                    args=(torch.tensor(3.0), torch.tensor(4.0)),  # 设置函数调用的参数
                    kwargs={},  # 不带关键字参数
                ),
                name="nearest_match",  # 子测试名称为 "nearest_match"
            ),
            # 定义第二个子测试，创建另一个代表 Torch FX 节点的对象，带有关键字参数
            common_utils.subtest(
                torch.fx.Node(
                    graph=torch.fx.Graph(),
                    name="aten::add.Tensor",
                    op="call_function",
                    target=torch.ops.aten.add.Tensor,  # 操作目标同上
                    args=(torch.tensor(3.0), torch.tensor(4.0)),  # 设置函数调用的参数
                    kwargs={"alpha": 1},  # 带有关键字参数 "alpha"
                ),
                name="perfect_match_with_kwargs",  # 子测试名称为 "perfect_match_with_kwargs"
            ),
        ],
    )
    # 定义测试方法，用于测试在 ONNX 函数中寻找完全匹配或最接近匹配的函数
    def test_find_the_perfect_or_nearest_match_onnxfunction_gives_tie_breaks_to_registered_order(
        self, node
    ):
        # 创建自定义域名对象
        custom_domain = onnxscript.values.Opset(domain="custom", version=1)

        # 定义三个不同的自定义 ONNX 脚本函数
        @onnxscript.script(custom_domain)
        def test_second_custom_op(
            x: TCustomFloat, y: TCustomFloat, alpha: int = 1
        ) -> TCustomFloat:
            return op.Add(x, y)

        @onnxscript.script(custom_domain)
        def test_third_custom_op(
            x: TCustomFloat, y: TCustomFloat, alpha: int = 1
        ) -> TCustomFloat:
            return op.Add(x, y)

        @onnxscript.script(custom_domain)
        def test_first_custom_op(
            x: TCustomFloat, y: TCustomFloat, alpha: int = 1
        ) -> TCustomFloat:
            return op.Add(x, y)

        op_full_name = "aten::add"  # 设置操作的完整名称为 "aten::add"

        # 创建包含三个注册的 ONNXFunction 对象的列表，每个对象表示一个自定义函数
        function_overloads = [
            registration.ONNXFunction(
                test_first_custom_op, op_full_name=op_full_name, is_custom=True
            ),
            registration.ONNXFunction(
                test_second_custom_op, op_full_name=op_full_name, is_custom=True
            ),
            registration.ONNXFunction(
                test_third_custom_op, op_full_name=op_full_name, is_custom=True
            ),
        ]

        # 调用方法 self.dispatcher._find_the_perfect_or_nearest_match_onnxfunction 来寻找符号函数
        # node 是当前测试的 Torch FX 节点，function_overloads 是函数重载的列表，node.args 和 node.kwargs 是节点的参数和关键字参数
        # self.diagnostic_context 是诊断上下文对象
        symbolic_fn = self.dispatcher._find_the_perfect_or_nearest_match_onnxfunction(
            node,
            function_overloads,
            node.args,
            node.kwargs,
            self.diagnostic_context,
        )
        # 使用断言方法检查找到的符号函数是否与 test_third_custom_op 相等
        self.assertEqual(symbolic_fn, test_third_custom_op)
@common_utils.instantiate_parametrized_tests
class TestOpSchemaWrapper(common_utils.TestCase):
    # 在测试类上应用参数化测试装饰器，实例化参数化测试
    def setUp(self):
        # 设置测试用例的前置条件：重载类型，可选的数据类型
        self.onnx_function_new_full = ops.core.aten_new_full
        self.onnx_function_new_full_dtype = ops.core.aten_new_full_dtype

    @common_utils.parametrize(
        "inputs, attributes, assertion",
        [
            common_utils.subtest(
                # 子测试1: 完全匹配，并带有关键字参数
                ([torch.randn(3, 4), torch.randn(3, 4)], {"alpha": 2.0}, True),
                name="perfect_match_with_kwargs",
            ),
            common_utils.subtest(
                # 子测试2: 非完全匹配，由于存在非张量类型的输入
                (["A", "B"], {}, False),
                name="non_perfect_match_due_to_non_tensor_inputs",
            ),
            common_utils.subtest(
                # 子测试3: 非完全匹配，由于输入数量过多
                ([torch.randn(3, 4), torch.randn(3, 4), torch.randn(3, 4)], {}, False),
                name="non_perfect_match_due_to_too_many_inputs",
            ),
            common_utils.subtest(
                # 子测试4: 非完全匹配，由于存在错误的关键字参数
                ([torch.randn(3, 4), torch.randn(3, 4)], {"wrong_kwargs": 2.0}, False),
                name="non_perfect_match_due_to_wrong_kwargs",
            ),
        ],
    )
    def test_perfect_match_inputs(self, inputs, attributes, assertion):
        # 测试函数：检查输入是否与ONNX函数的模式完全匹配
        dummy_diagnostic = diagnostics.Diagnostic(
            rule=diagnostics.rules.find_opschema_matched_symbolic_function,
            level=diagnostics.levels.WARNING,
        )
        # 创建ONNX函数的模式包装器，用于添加操作
        op_schema_wrapper_add = onnxfunction_dispatcher._OnnxSchemaChecker(
            ops.core.aten_add
        )
        # 断言：检查输入是否完全匹配预期的结果
        self.assertEqual(
            op_schema_wrapper_add.perfect_match_inputs(
                dummy_diagnostic, inputs, attributes
            ),
            assertion,
        )
    @common_utils.parametrize(
        "inputs, kwargs, op, score",
        [  # 参数化测试数据
            common_utils.subtest(  # 使用参数化测试的子测试
                ([torch.randn(3, 4), torch.randn(3, 4)], {}, ops.core.aten_mul, 2),  # 第一组测试数据：两个随机张量，空字典参数，aten_mul 操作，预期得分为 2
                name="match_2_inputs",  # 测试名称为 "match_2_inputs"
            ),
            common_utils.subtest(  # 使用参数化测试的子测试
                (
                    [
                        torch.randint(0, 2, size=(3, 4), dtype=torch.int).bool(),
                        torch.randint(0, 2, size=(3, 4), dtype=torch.int).bool(),
                    ],
                    {},  # 空字典参数
                    ops.core.aten_mul,  # aten_mul 操作
                    0,  # 预期得分为 0
                ),
                name="match_0_inputs",  # 测试名称为 "match_0_inputs"
            ),
            common_utils.subtest(  # 使用参数化测试的子测试
                ([torch.randn(3, 4), torch.randn(3, 4)], {}, ops.core.aten_mul_bool, 0),  # 第三组测试数据：两个随机张量，空字典参数，aten_mul_bool 操作，预期得分为 0
                name="match_0_inputs_bool",  # 测试名称为 "match_0_inputs_bool"
            ),
            common_utils.subtest(  # 使用参数化测试的子测试
                (
                    [
                        torch.randint(0, 2, size=(3, 4), dtype=torch.int).bool(),
                        torch.randint(0, 2, size=(3, 4), dtype=torch.int).bool(),
                    ],
                    {},  # 空字典参数
                    ops.core.aten_mul_bool,  # aten_mul_bool 操作
                    2,  # 预期得分为 2
                ),
                name="match_2_inputs_bool",  # 测试名称为 "match_2_inputs_bool"
            ),
        ],
    )
    def test_matching_score_system_on_overload_dtypes(self, inputs, kwargs, op, score):
        op_schema_wrapper = onnxfunction_dispatcher._OnnxSchemaChecker(op)  # 创建操作模式检查器对象
        op_schema_wrapper._record_matching_score(inputs, kwargs)  # 记录匹配得分
        self.assertEqual(op_schema_wrapper.match_score, score)  # 断言操作模式检查器的匹配得分与预期得分相等

    @common_utils.parametrize(
        "inputs, kwargs, op, score",
        [  # 参数化测试数据
            common_utils.subtest(  # 使用参数化测试的子测试
                ([torch.randn(3, 4), torch.tensor(3)], {}, ops.core.aten_new_full, 2),  # 第一组测试数据：一个随机张量和一个张量标量 3，空字典参数，aten_new_full 操作，预期得分为 2
                name="match_2_inputs",  # 测试名称为 "match_2_inputs"
            ),
            common_utils.subtest(  # 使用参数化测试的子测试
                (
                    [torch.randn(3, 4), torch.tensor(3)],  # 一个随机张量和一个张量标量 3
                    {"dtype": 2},  # 参数中包含 dtype 为 2，此时 dtype 应转换为整数
                    ops.core.aten_new_full_dtype,  # aten_new_full_dtype 操作
                    2,  # 预期得分为 2
                ),
                name="match_2_input_and_match_1_kwargs_optional",  # 测试名称为 "match_2_input_and_match_1_kwargs_optional"
            ),
        ],
    )
    def test_matching_score_system_on_optional_dtypes(self, inputs, kwargs, op, score):
        op_schema_wrapper = onnxfunction_dispatcher._OnnxSchemaChecker(op)  # 创建操作模式检查器对象
        op_schema_wrapper._record_matching_score(inputs, kwargs)  # 记录匹配得分
        self.assertEqual(op_schema_wrapper.match_score, score)  # 断言操作模式检查器的匹配得分与预期得分相等
    # 使用 common_utils.parametrize 装饰器为测试函数参数化，提供不同的输入和期望输出
    @common_utils.parametrize(
        "value, expected_onnx_str_dtype",
        [
            # 子测试 1: 整数类型的输入值，期望的输出数据类型集合包括 int64、int16、int32 的张量
            common_utils.subtest(
                (1, {"tensor(int64)", "tensor(int16)", "tensor(int32)"}),
                name="all_ints",
            ),
            # 子测试 2: 浮点数类型的输入值，期望的输出数据类型集合包括 float、double、float16 的张量
            common_utils.subtest(
                (1.0, {"tensor(float)", "tensor(double)", "tensor(float16)"}),
                name="all_floats",
            ),
            # 子测试 3: 布尔类型的输入值，期望的输出数据类型集合仅包括 bool 类型的张量
            common_utils.subtest(
                (torch.tensor([True]), {"tensor(bool)"}),
                name="bool",
            ),
            # 子测试 4: int64 类型的张量作为输入值，期望的输出数据类型集合仅包括 int64 类型的张量
            common_utils.subtest(
                (torch.tensor([1], dtype=torch.int64), {"tensor(int64)"}),
                name="int64",
            ),
            # 子测试 5: int32 类型的张量作为输入值，期望的输出数据类型集合仅包括 int32 类型的张量
            common_utils.subtest(
                (torch.tensor([1], dtype=torch.int32), {"tensor(int32)"}),
                name="int32",
            ),
            # 子测试 6: int16 类型的张量作为输入值，期望的输出数据类型集合仅包括 int16 类型的张量
            common_utils.subtest(
                (torch.tensor([1], dtype=torch.int16), {"tensor(int16)"}),
                name="int16",
            ),
            # 子测试 7: float 类型的张量作为输入值，期望的输出数据类型集合仅包括 float 类型的张量
            common_utils.subtest(
                (torch.tensor([1], dtype=torch.float), {"tensor(float)"}),
                name="float",
            ),
            # 子测试 8: float16 类型的张量作为输入值，期望的输出数据类型集合仅包括 float16 类型的张量
            common_utils.subtest(
                (torch.tensor([1], dtype=torch.float16), {"tensor(float16)"}),
                name="float16",
            ),
            # 子测试 9: double 类型的张量作为输入值，期望的输出数据类型集合仅包括 double 类型的张量
            common_utils.subtest(
                (torch.tensor([1], dtype=torch.double), {"tensor(double)"}),
                name="double",
            ),
            # 子测试 10: None 作为输入值，期望的输出数据类型集合为空集合
            common_utils.subtest((None, set()), name="None"),
            # 子测试 11: 空列表作为输入值，期望的输出数据类型集合为空集合
            common_utils.subtest(
                ([], set()), name="empaty_list"
            ),
        ],
    )
    # 测试函数，测试 _find_onnx_data_type 方法的返回值是否符合期望的输出数据类型集合
    def test_find_onnx_data_type(self, value, expected_onnx_str_dtype):
        self.assertEqual(
            onnxfunction_dispatcher._find_onnx_data_type(value), expected_onnx_str_dtype
        )
# 如果当前脚本被直接执行（而不是作为模块被导入），则执行以下代码块
if __name__ == "__main__":
    # 调用 common_utils 模块中的 run_tests 函数，用于执行测试
    common_utils.run_tests()
```