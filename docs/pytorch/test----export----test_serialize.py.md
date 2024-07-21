# `.\pytorch\test\export\test_serialize.py`

```
"""
PYTEST_DONT_REWRITE (prevents pytest from rewriting assertions, which interferes
with test_sym_bool)
"""


# Owner(s): ["oncall: export"]
import copy  # 导入 copy 模块，用于对象的浅复制和深复制操作
import io  # 导入 io 模块，用于处理文件流等 IO 操作
import tempfile  # 导入 tempfile 模块，用于创建临时文件和目录
import unittest  # 导入 unittest 模块，用于编写和运行单元测试
import zipfile  # 导入 zipfile 模块，用于 ZIP 文件的读写操作
from pathlib import Path  # 从 pathlib 模块导入 Path 类，用于处理文件路径

import torch  # 导入 PyTorch 深度学习库
import torch._dynamo as torchdynamo  # 导入 torch._dynamo 模块
import torch.export._trace  # 导入 torch.export._trace 模块
import torch.utils._pytree as pytree  # 导入 torch.utils._pytree 模块
from torch._export.db.case import ExportCase, normalize_inputs, SupportLevel  # 导入导出相关的类和函数
from torch._export.db.examples import all_examples  # 导入导出示例相关函数
from torch._export.serde.serialize import (  # 导入序列化相关函数和类
    canonicalize,
    deserialize,
    ExportedProgramDeserializer,
    ExportedProgramSerializer,
    serialize,
    SerializeError,
)
from torch._higher_order_ops.torchbind import enable_torchbind_tracing  # 导入 torchbind 相关操作
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode  # 导入 FakeTensor 相关类
from torch.export import Dim, export, load, save  # 导入导出相关函数和类
from torch.fx.experimental.symbolic_shapes import is_concrete_int, ValueRanges  # 导入符号形状相关函数和类
from torch.testing._internal.common_utils import (  # 导入测试相关的通用函数
    instantiate_parametrized_tests,
    IS_WINDOWS,
    parametrize,
    run_tests,
    TemporaryFileName,
    TestCase,
)
from torch.testing._internal.torchbind_impls import init_torchbind_implementations  # 导入 torchbind 实现相关函数


def get_filtered_export_db_tests():
    return [  # 返回筛选后的导出数据库测试用例
        (name, case)
        for name, case in all_examples().items()
        if case.support_level == SupportLevel.SUPPORTED
    ]


@unittest.skipIf(not torchdynamo.is_dynamo_supported(), "dynamo doesn't support")
class TestSerialize(TestCase):
    def test_export_with_custom_op_serialization(self):
        class TestModule(torch.nn.Module):
            def forward(self, x):
                return x + 1

        class FooCustomOp(torch.nn.Module):
            pass

        class FooCustomOpHandler(torch._export.serde.serialize.CustomOpHandler):
            def namespace(self):
                return "Foo"  # 自定义操作的命名空间为 "Foo"

            def op_name(self, op_type):
                if op_type == FooCustomOp:
                    return "FooCustomOp"  # 返回自定义操作的名称 "FooCustomOp"
                return None

            def op_type(self, op_name):
                if op_name == "FooCustomOp":
                    return FooCustomOp  # 返回与名称对应的自定义操作类 FooCustomOp
                return None

            def op_schema(self, op_type):
                if op_type == FooCustomOp:
                    return self.attached_schema  # 返回与自定义操作关联的模式（schema）
                return None

        inp = (torch.ones(10),)  # 创建输入张量
        ep = export(TestModule(), inp)  # 对测试模块进行导出，得到 ExportedProgram 对象 ep

        # Register the custom op handler.
        foo_custom_op = FooCustomOp()  # 创建自定义操作对象
        foo_custom_op_handler = FooCustomOpHandler()  # 创建自定义操作处理器对象
        torch._export.serde.serialize.register_custom_op_handler(
            foo_custom_op_handler, type(foo_custom_op)
        )  # 注册自定义操作处理器

        # Inject the custom operator.
        for node in ep.graph.nodes:
            if node.name == "add":
                foo_custom_op_handler.attached_schema = node.target._schema
                node.target = foo_custom_op  # 将图中名称为 "add" 的节点替换为自定义操作对象 foo_custom_op

        # Serialization.
        serialize(ep)  # 对 ExportedProgram 对象 ep 进行序列化
    def test_predispatch_export_with_autograd_op(self):
        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                # 在 forward 方法中，启用梯度计算
                with torch.enable_grad():
                    return x + x

        inp = (torch.ones(10),)
        with torch.no_grad():
            # 导入 _export 函数，用于导出模型
            from torch.export._trace import _export
            
            # 调用 _export 导出 Foo 模型的预分发版本
            ep = _export(Foo(), inp, pre_dispatch=True)

        buffer = io.BytesIO()
        # 将导出的模型保存到字节流中
        torch.export.save(ep, buffer)
        buffer.seek(0)
        # 从字节流中加载导出的模型
        loaded_ep = torch.export.load(buffer)

        # 获取预期的输出
        exp_out = ep.module()(*inp)
        # 获取加载后模型的实际输出
        actual_out = loaded_ep.module()(*inp)
        # 断言预期输出和加载后输出相等
        self.assertEqual(exp_out, actual_out)
        # 断言预期输出和加载后输出的梯度属性相等
        self.assertEqual(exp_out.requires_grad, actual_out.requires_grad)

    def test_export_example_inputs_preserved(self):
        class MyModule(torch.nn.Module):
            """A test module with that has multiple args and uses kwargs"""

            def __init__(self):
                super().__init__()
                # 创建一个参数 p，形状为 (2, 3)，并添加到模块中
                self.p = torch.nn.Parameter(torch.ones(2, 3))

            def forward(self, x, y, use_p=False):
                # 在 forward 方法中计算 x + y
                out = x + y
                if use_p:
                    # 如果 use_p 为 True，则将 self.p 加到 out 中
                    out += self.p
                return out

        # 创建一个测试模型 MyModule 的实例，并设为评估模式
        model = MyModule().eval()
        # 创建随机的输入数据
        random_inputs = (torch.rand([2, 3]), torch.rand([2, 3]))
        # 导出模型及其参数和输入，使用 use_p=True
        exp_program = torch.export.export(model, random_inputs, {"use_p": True})

        # 创建一个字节流缓冲区
        output_buffer = io.BytesIO()
        # 将导出的模型保存到字节流中
        # 测试示例输入在保存和加载模块时是否被保留
        torch.export.save(exp_program, output_buffer)
        # 从字节流中加载模型
        loaded_model = torch.export.load(output_buffer)
        # 从导出程序中提取保存前后的示例输入
        orig_args, orig_kwargs = exp_program.example_inputs
        loaded_args, loaded_kwargs = loaded_model.example_inputs
        # 运行原始模型和加载后的模型，并确认它们的输出是否一致
        orig_out = exp_program.module()(*orig_args, **orig_kwargs)
        loaded_out = loaded_model.module()(*loaded_args, **loaded_kwargs)
        # 断言原始输出和加载后输出相等
        self.assertEqual(orig_out, loaded_out)
    def test_metadata_parsing_with_layer_split(self):
        # Tests that modules with more complicated layer patterns can be serialized
        # and deserialized correctly.
        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # Define a sequential stack of SiLU activation layers
                self.layers = torch.nn.Sequential(
                    torch.nn.SiLU(),
                    torch.nn.SiLU(),
                    torch.nn.SiLU(),
                )

            def forward(self, x):
                # Splitting layers of a sequential stack introduces commas and parens
                # into metadata trace.
                # Extract the first layer and the rest of the layers
                out_start, out_rest = self.layers[0], self.layers[1:]
                # Apply the first layer to input x
                h = out_start(x)
                # Apply the rest of the layers sequentially to h
                h = out_rest(h)
                return h

        # Prepare input tensor
        inp = (torch.ones(10),)
        # Serialize and export the module with input
        ep = export(MyModule(), inp)
        # Create a BytesIO buffer to save the serialized module
        buffer = io.BytesIO()
        # Save the serialized module to the buffer
        save(ep, buffer)
        # Load the serialized module from the buffer
        loaded_ep = load(buffer)

        # Check that both original and loaded modules produce the same output
        exp_out = ep.module()(*inp)
        actual_out = loaded_ep.module()(*inp)
        self.assertEqual(exp_out, actual_out)

    def test_serialize_constant_outputs(self):
        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                # Along with tensor output, return Nonetype
                # and constant. Although these outputs aren't
                # very useful, they do show up in graphs.
                return x + 1, None, 1024

        # Check that module can be roundtripped, thereby confirming proper deserialization.
        inp = (torch.ones(10),)
        # Serialize and export the module with input
        ep = export(MyModule(), inp)
        # Create a BytesIO buffer to save the serialized module
        buffer = io.BytesIO()
        # Save the serialized module to the buffer
        save(ep, buffer)
        # Load the serialized module from the buffer
        loaded_ep = load(buffer)

        # Check that both original and loaded modules produce the same output
        exp_out = ep.module()(*inp)
        actual_out = loaded_ep.module()(*inp)
        self.assertEqual(exp_out, actual_out)
    def test_serialize_multiple_returns_from_node(self) -> None:
        # 定义一个测试函数，用于测试从节点返回多个结果的序列化情况

        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, w, b):
                # 模型的前向传播方法，执行 layer normalization 操作
                return torch.nn.functional.layer_norm(
                    x,
                    x.size()[1:],  # 取输入张量 x 的除第一维外的维度作为参数传递给 layer_norm
                    weight=w,
                    bias=b,
                    eps=1e-5,
                )

        # 导出 MyModule 模型，并运行分解
        exported_module = export(
            MyModule(),
            (
                torch.ones([512, 512], requires_grad=True),  # 输入张量 x
                torch.ones([512]),  # 权重张量 w
                torch.ones([512]),  # 偏置张量 b
            ),
        ).run_decompositions()

        # 序列化导出的模型
        serialized = ExportedProgramSerializer().serialize(exported_module)
        # 获取序列化后的模型的最后一个节点
        node = serialized.exported_program.graph_module.graph.nodes[-1]
        # 断言最后一个节点的目标操作是 torch.ops.aten.native_layer_norm.default
        self.assertEqual(node.target, "torch.ops.aten.native_layer_norm.default")
        # 断言节点输出的张量数量为 3
        self.assertEqual(len(node.outputs), 3)

        # 检查输出张量的名称是否唯一
        seen = set()
        for output in node.outputs:
            name = output.as_tensor.name
            self.assertNotIn(name, seen)
            seen.add(name)

    def test_serialize_list_returns(self) -> None:
        # 定义一个测试函数，用于测试序列化返回列表的情况

        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                # 模型的前向传播方法，执行张量 x 的切分操作
                return torch.split(x, 2)

        # 准备输入张量
        input = torch.arange(10.0).reshape(5, 2)
        # 导出 MyModule 模型，并运行分解
        exported_module = export(MyModule(), (input,)).run_decompositions()

        # 序列化导出的模型
        serialized = ExportedProgramSerializer().serialize(exported_module)
        # 获取序列化后的模型的最后一个节点
        node = serialized.exported_program.graph_module.graph.nodes[-1]
        # 断言最后一个节点的目标操作是 torch.ops.aten.split_with_sizes.default
        self.assertEqual(node.target, "torch.ops.aten.split_with_sizes.default")
        # 断言节点输出列表的长度为 1
        self.assertEqual(len(node.outputs), 1)
        # 断言节点输出列表中包含 3 个张量
        self.assertEqual(len(node.outputs[0].as_tensors), 3)

        # 检查输出张量的名称是否唯一
        seen = set()
        for output in node.outputs[0].as_tensors:
            name = output.name
            self.assertNotIn(name, seen)
            seen.add(name)
    def test_multi_return_some_unused(self) -> None:
        """
        Make sure the serialized output matches the op schema, even if some of
        the arguments are never used in the graph.
        """

        # 定义一个名为MyModule的子类，继承自torch.nn.Module
        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            # 定义forward方法，接受参数x并返回torch.ops.aten.var_mean.correction的第一个输出
            def forward(self, x):
                return torch.ops.aten.var_mean.correction(x, [1])[0]

        # 创建MyModule的实例，并使用torch.ones([512, 512], requires_grad=True)作为输入进行导出
        exported_module = export(
            MyModule(),
            (torch.ones([512, 512], requires_grad=True),),
        ).run_decompositions()

        # 对导出的模块进行序列化
        serialized = ExportedProgramSerializer().serialize(exported_module)
        
        # 获取序列化后的导出程序的最后一个节点
        node = serialized.exported_program.graph_module.graph.nodes[-1]
        
        # 断言最后一个节点的目标是"torch.ops.aten.var_mean.correction"
        self.assertEqual(node.target, "torch.ops.aten.var_mean.correction")
        
        # 断言最后一个节点的输出个数为2
        self.assertEqual(len(node.outputs), 2)

        # 检查输出名称是唯一的
        seen = set()
        for output in node.outputs:
            name = output.as_tensor.name
            self.assertNotIn(name, seen)
            seen.add(name)

    def test_rational_ranges(self) -> None:
        # 定义一个名为M的子类，继承自torch.nn.Module
        class M(torch.nn.Module):
            # 定义forward方法，接受参数x并返回x + x
            def forward(self, x):
                return x + x

        # 导出M的实例，使用torch.randn(4)作为输入，并指定动态形状
        ep = torch.export.export(
            M(), (torch.randn(4),), dynamic_shapes=({0: Dim("temp")},)
        )

        # 获取导出程序的范围约束键列表
        range_constraints = list(ep.range_constraints.keys())
        assert len(range_constraints) == 1
        symint = range_constraints[0]

        # 导入sympy模块
        import sympy

        # 设置上界和下界的有理数范围
        upper_range = sympy.Rational(10, 3)
        lower_range = sympy.Rational(10, 6)
        ep.range_constraints[symint] = ValueRanges(lower=lower_range, upper=upper_range)

        # 对导出的程序进行序列化
        serialized = ExportedProgramSerializer().serialize(ep)
        
        # 断言序列化后的导出程序的范围约束中"s0"的最小值为2
        self.assertEqual(serialized.exported_program.range_constraints["s0"].min_val, 2)
        
        # 断言序列化后的导出程序的范围约束中"s0"的最大值为3
        self.assertEqual(serialized.exported_program.range_constraints["s0"].max_val, 3)

    def test_kwargs_default(self) -> None:
        """
        Tests that the kwargs default values are serialized even if they are not
        specified
        """

        # 定义一个名为Foo的子类，继承自torch.nn.Module
        class Foo(torch.nn.Module):
            # 定义forward方法，接受参数x，并返回对x进行搜索后的结果
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                values = torch.randn(3, 2)
                return torch.searchsorted(x, values, side="right", right=True)

        # 创建Foo的实例f
        f = Foo()

        # 对torch.randn(3, 4)进行排序，获取排序后的结果x
        x, _ = torch.sort(torch.randn(3, 4))
        
        # 导出Foo的实例f，使用x作为输入，并运行分解
        exported_module = export(f, (x,)).run_decompositions()
        
        # 对导出的模块进行序列化
        serialized = ExportedProgramSerializer().serialize(exported_module)

        # 获取序列化后的导出程序的最后一个节点
        node = serialized.exported_program.graph_module.graph.nodes[-1]
        
        # 断言最后一个节点的目标是"torch.ops.aten.searchsorted.Tensor"
        self.assertEqual(node.target, "torch.ops.aten.searchsorted.Tensor")
        
        # 断言最后一个节点的输入个数为4
        self.assertEqual(len(node.inputs), 4)
        
        # 断言最后一个节点的第三个输入的名称为"right"
        self.assertEqual(node.inputs[2].name, "right")
        
        # 断言最后一个节点的第三个输入的布尔值为True
        self.assertEqual(node.inputs[2].arg.as_bool, True)
        
        # 断言最后一个节点的第四个输入的名称为"side"
        self.assertEqual(node.inputs[3].name, "side")
        
        # 断言最后一个节点的第四个输入的字符串值为"right"
        self.assertEqual(node.inputs[3].arg.as_string, "right")
    # 定义测试方法 test_canonicalize，用于测试规范化处理
    def test_canonicalize(self) -> None:
        # 定义一个内嵌的 Module 类，继承自 torch.nn.Module
        class Module(torch.nn.Module):
            # 定义前向传播方法，接收两个张量 x 和 y，返回一个张量
            def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                # 计算张量 y 和 x 的和，赋值给变量 a
                a = y + x
                # 计算张量 x 和 y 的和，赋值给变量 b
                b = x + y
                # 返回 a 和 b 的和作为最终结果
                return b + a

        # 使用 torch.export.export 方法导出 Module 的结果，传入两个随机张量
        ep = torch.export.export(Module(), (torch.randn(3, 2), torch.randn(3, 2)))
        # 使用 ExportedProgramSerializer 对象序列化导出结果 ep
        s = ExportedProgramSerializer().serialize(ep)
        # 调用 canonicalize 方法对导出的程序进行规范化处理，返回规范化后的结果
        c = canonicalize(s.exported_program)
        # 获取规范化后的结果中的图模块的图对象
        g = c.graph_module.graph
        # 断言第一个节点的第一个输入的张量名小于第二个节点的第一个输入的张量名
        self.assertLess(
            g.nodes[0].inputs[0].arg.as_tensor.name,
            g.nodes[1].inputs[0].arg.as_tensor.name,
        )

    # 定义测试方法 test_int_list，用于测试整数列表的处理
    def test_int_list(self) -> None:
        # 定义一个内嵌的 M 类，继承自 torch.nn.Module
        class M(torch.nn.Module):
            # 定义前向传播方法，只接收一个参数 x
            def forward(self, x):
                # 调用 torch.ops.aten.sum.dim_IntList 方法对张量 x 进行求和操作
                return torch.ops.aten.sum.dim_IntList(x, [])

        # 使用 torch.export.export 方法导出 M 的结果，传入一个随机张量
        ep = torch.export.export(M(), (torch.randn(3, 2),))
        # 使用 ExportedProgramSerializer 对象序列化导出结果 ep
        serialized = ExportedProgramSerializer().serialize(ep)
        # 遍历序列化后的导出程序的图模块的所有节点
        for node in serialized.exported_program.graph_module.graph.nodes:
            # 如果节点的目标操作包含字符串 "aten.sum.dim_IntList"
            if "aten.sum.dim_IntList" in node.target:
                # 断言节点的第二个输入参数的类型为 "as_ints"
                self.assertEqual(node.inputs[1].arg.type, "as_ints")
# 如果运行环境为 Windows，则跳过此测试，因为 Windows 不支持该测试
@unittest.skipIf(IS_WINDOWS, "Windows not supported for this test")
# 如果当前环境不支持 dynamo，则跳过此测试
@unittest.skipIf(not torchdynamo.is_dynamo_supported(), "dynamo doesn't support")
# 定义一个 TestCase 类，用于测试反序列化功能
class TestDeserialize(TestCase):

    # 在每个测试方法执行前执行的设置方法
    def setUp(self):
        super().setUp()
        # 初始化 torchbind 实现
        init_torchbind_implementations()

    # 定义一个方法，用于检查导出的图的反序列化结果
    def check_graph(
        self,
        fn,
        inputs,
        dynamic_shapes=None,
        _check_meta=True,
        use_pre_dispatch=True,
        strict=True,
    ) -> None:
        """Export a graph, serialize it, deserialize it, and compare the results."""

        # 内部方法，用于检查图的反序列化结果
        def _check_graph(pre_dispatch):
            # 如果使用预分发
            if pre_dispatch:
                # 利用 torch.export._trace._export 导出图
                ep = torch.export._trace._export(
                    fn,
                    copy.deepcopy(inputs),
                    {},
                    dynamic_shapes=dynamic_shapes,
                    pre_dispatch=True,
                    strict=strict,
                )
            else:
                # 否则，使用 torch.export.export 导出图
                ep = torch.export.export(
                    fn,
                    copy.deepcopy(inputs),
                    {},
                    dynamic_shapes=dynamic_shapes,
                    strict=strict,
                )
            
            # 消除死代码
            ep.graph.eliminate_dead_code()

            # 序列化图的产物
            serialized_artifact = serialize(ep, opset_version={"aten": 0})
            # 反序列化图
            deserialized_ep = deserialize(
                serialized_artifact, expected_opset_version={"aten": 0}
            )
            # 消除反序列化后图的死代码
            deserialized_ep.graph.eliminate_dead_code()

            # 使用原始输出和加载后输出进行比较
            orig_outputs = ep.module()(*copy.deepcopy(inputs))
            loaded_outputs = deserialized_ep.module()(*copy.deepcopy(inputs))

            # 扁平化原始输出和加载后输出
            flat_orig_outputs = pytree.tree_leaves(orig_outputs)
            flat_loaded_outputs = pytree.tree_leaves(loaded_outputs)

            # 逐个比较原始输出和加载后输出
            for orig, loaded in zip(flat_orig_outputs, flat_loaded_outputs):
                self.assertEqual(type(orig), type(loaded))
                # 如果是 torch.Tensor 对象
                if isinstance(orig, torch.Tensor):
                    # 如果原始张量是元数据
                    if orig.is_meta:
                        self.assertEqual(orig, loaded)
                    else:
                        # 否则，确保张量值相似
                        self.assertTrue(torch.allclose(orig, loaded))
                else:
                    # 对于非张量对象，直接比较值
                    self.assertEqual(orig, loaded)
            
            # 检查图节点的一致性
            self._check_graph_nodes(
                ep.graph_module, deserialized_ep.graph_module, _check_meta
            )

        # 如果使用预分发，则依次检查预分发和非预分发情况
        if use_pre_dispatch:
            _check_graph(pre_dispatch=True)
            _check_graph(pre_dispatch=False)
        else:
            # 否则，只检查非预分发情况
            _check_graph(pre_dispatch=False)
    # 定义一个测试方法，用于测试带有可选参数的元组输入情况
    def test_optional_tuple(self):
        # 使用 torch 库的 _scoped_library 方法创建一个名为 "mylib" 的库，并设置作用域为 "FRAGMENT"
        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            # 在 "mylib" 库中定义函数 "mylib::foo"，该函数接受参数 (Tensor a, Tensor b, Tensor? c)，返回 (Tensor, Tensor?)
            # 设置函数的标签为 torch.Tag.pt2_compliant_tag，并指定库为 lib
            torch.library.define(
                "mylib::foo",
                "(Tensor a, Tensor b, Tensor? c) -> (Tensor, Tensor?)",
                tags=torch.Tag.pt2_compliant_tag,
                lib=lib,
            )

            # 在 "mylib" 库中实现函数 "mylib::foo"，针对 CPU 设备，实现函数功能
            @torch.library.impl("mylib::foo", "cpu", lib=lib)
            # 同时在抽象层面实现函数 "mylib::foo"，使其可用于模型构建
            @torch.library.impl_abstract("mylib::foo")
            def foo_impl(a, b, c):
                # 初始化变量 res2 为 None
                res2 = None
                # 如果参数 c 不为 None，则计算 res2 的值为 c + a + b
                if c is not None:
                    res2 = c + a + b
                # 返回 a + b 和计算后的 res2
                return a + b, res2

            # 定义一个继承自 torch.nn.Module 的模型类 M
            class M(torch.nn.Module):
                # 实现模型类的 forward 方法，接受参数 a, b, c，并调用 torch.ops.mylib.foo 来处理输入
                def forward(self, a, b, c):
                    return torch.ops.mylib.foo(a, b, c)

            # 使用 self.check_graph 方法检查模型 M 的计算图，传入随机生成的张量作为输入
            self.check_graph(M(), (torch.randn(3), torch.randn(3), torch.randn(3)))
    # 定义一个测试方法，用于测试自动功能化
    def test_multi_return(self) -> None:
        """
        Test multiple return from a single node (ex. layer_norm has 2 outputs)
        """

        # 定义一个继承自 torch.nn.Module 的自定义模块 MyModule
        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            # 定义模块的前向传播方法
            def forward(self, x, w, b):
                # 调用 torch.nn.functional 中的 layer_norm 函数进行层归一化
                return torch.nn.functional.layer_norm(
                    x,
                    x.size()[1:],  # 使用 x 的除了第一个维度外的大小作为归一化维度
                    weight=w,      # 归一化时使用的权重
                    bias=b,        # 归一化时使用的偏置
                    eps=1e-5,      # 归一化过程中使用的 epsilon 值
                )

        # 创建输入数据元组，包含三个张量，用于测试前向传播
        inputs = (
            torch.ones([512, 512], requires_grad=True),  # 输入张量 x，形状为 [512, 512]
            torch.ones([512]),                          # 输入张量 w，形状为 [512]
            torch.ones([512]),                          # 输入张量 b，形状为 [512]
        )
        # 调用检查图方法，验证 MyModule 在给定输入上的图结构
        self.check_graph(MyModule(), inputs)
    # 定义一个测试方法，用于测试基本功能
    def test_basic(self) -> None:
        # 定义一个简单的神经网络模块
        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                # 对输入数据进行加法操作
                x = x + x
                # 对加法结果进行乘法操作
                x = x * x
                # 对乘法结果进行除法操作
                x = x / x
                # 返回操作后的数据及其克隆
                return x, x.clone()

        # 准备输入数据
        inputs = (torch.ones([512], requires_grad=True),)
        # 调用自定义函数检查模块的图结构
        self.check_graph(MyModule(), inputs)

    # 定义一个测试方法，用于测试动态形状
    def test_dynamic(self) -> None:
        # 定义一个处理动态形状输入的简单模型
        class DynamicShapeSimpleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, a, b, c) -> torch.Tensor:
                # 执行矩阵乘法并加上偏置，再除以2
                d = (torch.matmul(a, b) + c) / 2
                # 计算第一维度和第二维度的形状
                d_s0 = d.shape[0]
                d_s1 = d.shape[1]
                # 计算新的形状维度
                d_s3 = d_s0 * d_s1
                # 将数据展平为一维
                e = d.view(d_s3)
                # 返回连接两个展平结果的张量
                return torch.cat([e, e])

        # 准备输入数据
        inputs = (torch.randn(2, 4), torch.randn(4, 7), torch.randn(2, 7))
        # 定义动态形状的字典
        dim0_ac = torch.export.Dim("dim0_ac")
        dynamic_shapes = {"a": {0: dim0_ac}, "b": None, "c": {0: dim0_ac}}
        # 调用自定义函数检查模块的图结构
        self.check_graph(DynamicShapeSimpleModel(), inputs, dynamic_shapes)

    # TODO: 因"constraining non-Symbols NYI (Piecewise((1, Eq(u1, 1)), (0, True)), 1, 1)"而导致测试失败
    @unittest.expectedFailure
    def test_sym_bool(self):
        # 定义一个简单的神经网络模块
        class Module(torch.nn.Module):
            def forward(self, x, y):
                # 断言x的大小是否在y中
                assert x.size(0) in y
                # 返回x和y的和
                return x + y

        # 实例化模块对象
        f = Module()
        # 调用自定义函数检查模块的图结构，传入输入数据
        self.check_graph(f, (torch.ones(1), torch.ones(3)))

    # 定义一个测试方法，用于测试形状处理
    def test_shape(self):
        # 定义一个简单的神经网络模块
        class Foo(torch.nn.Module):
            def forward(self, x):
                # 获取输入张量的维度
                z, y = x.size()
                # 返回输入张量的维度和张量的第一个元素的和，以及张量的第一维度
                return z + y + x[0], z

        # 准备输入数据
        inputs = (torch.ones(2, 3),)
        # 定义动态形状的字典
        dim0_x, dim1_x = torch.export.dims("dim0_x", "dim1_x")
        dynamic_shapes = {"x": (dim0_x, dim1_x)}
        # 调用自定义函数检查模块的图结构
        self.check_graph(Foo(), inputs, dynamic_shapes)

    # 定义一个测试方法，用于测试包含子模块的神经网络模块
    def test_module(self):
        # 定义一个包含子模块的神经网络模块
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = torch.nn.Linear(3, 3)
                self.relu = torch.nn.ReLU()
                self.linear2 = torch.nn.Linear(3, 5)

            def forward(self, x):
                # 使用线性层进行前向传播
                x = self.linear1(x)
                x = self.linear1(x)
                x = torch.nn.functional.relu(x)
                x = self.linear2(x)
                # 返回输出结果
                return x

        # 准备输入数据
        inputs = (torch.randn(3, 3),)
        # 调用自定义函数检查模块的图结构
        self.check_graph(M(), inputs)

    # 定义一个测试方法，用于测试包含元数据的神经网络模块
    def test_module_meta(self):
        # 定义一个包含元数据的神经网络模块
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.p = torch.nn.Parameter(torch.ones(3, 3))

            def forward(self, x):
                # 返回参数张量加上输入张量的结果
                return self.p + x

        # 在meta设备上创建模块实例
        with torch.device("meta"):
            mod = M()

        # 准备输入数据
        inputs = (torch.randn(3, 3, device="meta"),)
        # 调用自定义函数检查模块的图结构
        self.check_graph(mod, inputs)
    def test_cond(self):
        # 导入 cond 函数从 functorch.experimental.control_flow 模块
        from functorch.experimental.control_flow import cond

        # 定义输入数据
        inputs = torch.ones(4, 3), torch.zeros(4, 3)

        # 定义一个继承自 torch.nn.Module 的类 M
        class M(torch.nn.Module):
            def forward(self, x, y):
                # 定义条件为 x[0][0] > 4 的条件执行分支 t 和 f
                def t(x, y):
                    return x + y

                def f(x, y):
                    return x - y

                # 调用 cond 函数根据条件 x[0][0] > 4 执行 t 或 f 分支
                return cond(x[0][0] > 4, t, f, [x, y])

        # 检查类 M 的计算图
        self.check_graph(M(), inputs)

    def test_map(self):
        # 导入 control_flow 模块从 functorch.experimental
        from functorch.experimental import control_flow

        # 定义函数 f，用于 map 操作
        def f(x, y):
            return x + y

        # 定义一个继承自 torch.nn.Module 的类 Module
        class Module(torch.nn.Module):
            def forward(self, xs, y):
                # 使用 control_flow 模块中的 map 函数将 f 映射到 xs 中的每个元素
                return control_flow.map(f, xs, y)

        # 创建 Module 类的实例 g
        g = Module()
        inputs = (torch.ones(3, 2, 2), torch.ones(2))
        # 检查类 Module 的计算图，忽略元数据检查
        self.check_graph(g, inputs, _check_meta=False)

    def test_tensor_tensor_list(self):
        # 使用 torch.library._scoped_library 定义库作用域 "_export"，"FRAGMENT"
        with torch.library._scoped_library("_export", "FRAGMENT") as lib:
            # 定义函数签名和标签
            lib.define(
                "_test_tensor_tensor_list_output(Tensor x, Tensor y) -> (Tensor, Tensor[])",
                tags=torch.Tag.pt2_compliant_tag,
            )

            # 定义函数 _test_tensor_tensor_list_output 的实现
            def _test_tensor_tensor_list_output(x, y):
                return y, [x]

            # 将 _test_tensor_tensor_list_output 函数实现注册到库中，CPU 平台
            lib.impl(
                "_test_tensor_tensor_list_output",
                _test_tensor_tensor_list_output,
                "CPU",
            )
            # 将 _test_tensor_tensor_list_output 函数实现注册到库中，Meta 平台
            lib.impl(
                "_test_tensor_tensor_list_output",
                _test_tensor_tensor_list_output,
                "Meta",
            )

            # 定义一个继承自 torch.nn.Module 的类 M
            class M(torch.nn.Module):
                def forward(self, x, y):
                    # 调用库中的函数 torch.ops._export._test_tensor_tensor_list_output.default
                    a, b = torch.ops._export._test_tensor_tensor_list_output.default(
                        x, y
                    )
                    # 返回 a 和 b[0] 的和
                    return a + b[0]

            # 检查类 M 的计算图，使用随机生成的输入数据
            self.check_graph(M(), (torch.rand(3, 2), torch.rand(3, 2)))

    def test_list_of_optional_tensors(self) -> None:
        # 定义一个继承自 torch.nn.Module 的类 MyModule
        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y, z):
                # 定义索引列表，包含 None 和 torch.tensor([1, 3, 5, 7])
                indices = [None, None, torch.tensor([1, 3, 5, 7])]
                # 使用 torch.ops.aten.index.Tensor 进行索引操作
                indexed = torch.ops.aten.index.Tensor(x + y, indices)
                # 返回 indexed 和 z 的和
                return indexed + z

        # 定义输入数据
        inputs = (torch.rand(8, 8, 8), torch.rand(8, 8, 8), torch.rand(8, 8, 4))
        # 检查类 MyModule 的计算图
        self.check_graph(MyModule(), inputs)

    def test_sym_ite(self):
        # 定义一个继承自 torch.nn.Module 的类 Foo
        class Foo(torch.nn.Module):
            def forward(self, x):
                # 判断 x 的第一个维度是否为 5
                b = x.shape[0] == 5
                # 使用 torch.sym_ite 执行条件分支
                ret = torch.sym_ite(b, x.shape[0], x.shape[1])
                # 返回结果 ret
                return ret

        # 定义动态形状字典
        dynamic_shapes = {"x": {0: Dim("dim0"), 1: Dim("dim1")}}
        # 检查类 Foo 的计算图，使用输入数据 torch.ones(4, 5)，并指定动态形状
        self.check_graph(Foo(), (torch.ones(4, 5),), dynamic_shapes=dynamic_shapes)
    def test_multiple_getitem(self):
        # 定义一个继承自 torch.nn.Module 的简单模块类 M
        class M(torch.nn.Module):
            # 实现模块的前向传播方法
            def forward(self, x):
                # 对输入张量 x 进行 topk 操作，获取前两个最大值及其索引
                a, b = torch.topk(x, 2)
                # 将 a 中的每个元素乘以 2
                a = a * 2
                # 返回修改后的张量 a 和 b
                return a, b

        # 使用 torch.export.export 函数将模块 M 导出为可序列化的表示
        ep = torch.export.export(M(), (torch.ones(3),))

        # 在序列化表示的图中插入另一个 getitem 节点
        for node in ep.graph.nodes:
            # 查找函数调用节点，目标函数为 torch.ops.aten.mul.Tensor
            if node.op == "call_function" and node.target == torch.ops.aten.mul.Tensor:
                # 获取第一个参数作为 getitem 节点
                getitem_0 = node.args[0]
                # 在 getitem_0 节点之前插入一个副本节点 getitem_copy
                with ep.graph.inserting_before(getitem_0):
                    getitem_copy = ep.graph.node_copy(getitem_0)
                    # 创建新的乘法节点 mul_node，将 getitem_copy 和 2 作为参数
                    mul_node = ep.graph.call_function(
                        torch.ops.aten.mul.Tensor, (getitem_copy, 2)
                    )
                    # 复制 getitem_copy 的元数据到 mul_node
                    mul_node.meta = copy.copy(getitem_copy.meta)
                    # 更新节点参数，替换原来的 getitem_0
                    node.args = (getitem_0, mul_node)

        # 反序列化表示的模块对象
        deserialized_ep = deserialize(serialize(ep))

        # 定义输入张量
        inp = (torch.randn(3),)
        # 在原模块和反序列化后的模块上分别执行前向传播
        orig_res = ep.module()(*inp)
        res = deserialized_ep.module()(*inp)
        # 断言两次前向传播的输出在数值上接近
        self.assertTrue(torch.allclose(orig_res[0], res[0]))
        self.assertTrue(torch.allclose(orig_res[1], res[1]))

        # 断言反序列化后的图中应该已经去重了 getitem 调用
        self.assertExpectedInline(
            deserialized_ep.graph_module.code.strip("\n"),
            """\
    # 在输入张量 x 上执行 torch.ops.aten.topk.default 操作，返回前两个最大值及其对应的索引
    topk_default = torch.ops.aten.topk.default(x, 2);  x = None
    # 获取 topk_default 结果的第一个元素
    getitem = topk_default[0]
    # 获取 topk_default 结果的第二个元素
    getitem_1 = topk_default[1];  topk_default = None
    # 将 getitem 与标量 2 相乘得到新的张量
    mul_tensor = torch.ops.aten.mul.Tensor(getitem, 2)
    # 将 getitem 与 mul_tensor 相乘得到新的张量 mul，并清除中间变量
    mul = torch.ops.aten.mul.Tensor(getitem, mul_tensor);  getitem = mul_tensor = None
    # 返回两个张量 mul 和 getitem_1 作为结果
    return (mul, getitem_1)
    # 定义一个名为 test_custom_obj 的测试方法
    def test_custom_obj(self):
        # 定义一个内部类 MyModule，继承自 torch.nn.Module
        class MyModule(torch.nn.Module):
            # 类的初始化方法
            def __init__(self):
                super().__init__()  # 调用父类的初始化方法
                # 创建一个名为 attr 的属性，其值为 torch.classes._TorchScriptTesting._Foo 的实例
                self.attr = torch.classes._TorchScriptTesting._Foo(10, 20)

            # 前向传播方法
            def forward(self, x):
                # 调用 torch.ops._TorchScriptTesting.takes_foo 方法，传入 self.attr 和 x 作为参数，得到结果赋给 a
                a = torch.ops._TorchScriptTesting.takes_foo(self.attr, x)
                # 再次调用 torch.ops._TorchScriptTesting.takes_foo 方法，传入 self.attr 和 a 作为参数，得到结果赋给 b
                b = torch.ops._TorchScriptTesting.takes_foo(self.attr, a)
                # 返回 x 加上 b 的结果
                return x + b

        # 创建 MyModule 类的实例 m
        m = MyModule()
        # 创建输入数据，这里是一个包含 torch.ones(2, 3) 的元组
        inputs = (torch.ones(2, 3),)
        # 调用 self.check_graph 方法，传入 m、inputs 和 strict=False 作为参数进行测试
        self.check_graph(m, inputs, strict=False)

    # 定义一个名为 test_custom_obj_list_out 的测试方法
    def test_custom_obj_list_out(self):
        # 定义一个内部类 MyModule，继承自 torch.nn.Module
        class MyModule(torch.nn.Module):
            # 类的初始化方法
            def __init__(self):
                super().__init__()  # 调用父类的初始化方法
                # 创建一个名为 attr 的属性，其值为 torch.classes._TorchScriptTesting._Foo 的实例
                self.attr = torch.classes._TorchScriptTesting._Foo(10, 20)

            # 前向传播方法
            def forward(self, x):
                # 调用 torch.ops._TorchScriptTesting.takes_foo_list_return 方法，传入 self.attr 和 x 作为参数，得到结果赋给 a
                a = torch.ops._TorchScriptTesting.takes_foo_list_return(self.attr, x)
                # 计算 y 为 a[0]、a[1] 和 a[2] 的和
                y = a[0] + a[1] + a[2]
                # 再次调用 torch.ops._TorchScriptTesting.takes_foo 方法，传入 self.attr 和 y 作为参数，得到结果赋给 b
                b = torch.ops._TorchScriptTesting.takes_foo(self.attr, y)
                # 返回 x 加上 b 的结果
                return x + b

        # 创建 MyModule 类的实例 m
        m = MyModule()
        # 创建输入数据，这里是一个包含 torch.ones(2, 3) 的元组
        inputs = (torch.ones(2, 3),)
        # 调用 self.check_graph 方法，传入 m、inputs 和 strict=False 作为参数进行测试
        self.check_graph(m, inputs, strict=False)

    # 定义一个名为 test_export_no_inputs 的测试方法
    def test_export_no_inputs(self):
        # 定义一个内部类 M，继承自 torch.nn.Module
        class M(torch.nn.Module):
            # 类的初始化方法
            def __init__(self):
                super().__init__()  # 调用父类的初始化方法
                # 创建一个名为 p 的属性，其值为 torch.ones(3, 3)
                self.p = torch.ones(3, 3)

            # 前向传播方法，没有参数
            def forward(self):
                # 返回属性 self.p 与自身相乘的结果
                return self.p * self.p

        # 调用 torch.export.export 方法，传入 M 的实例和空元组 ()，导出为一个对象 ep
        ep = torch.export.export(M(), ())
        # 设置 ep 的 _example_inputs 属性为 None
        ep._example_inputs = None
        # 将 ep 序列化后再反序列化，得到 roundtrip_ep 对象
        roundtrip_ep = deserialize(serialize(ep))
        # 断言判断 ep.module()() 和 roundtrip_ep.module()() 的结果是否全部近似相等
        self.assertTrue(torch.allclose(ep.module()(), roundtrip_ep.module()()))
# 使用给定的参数化测试类实例化参数化测试
instantiate_parametrized_tests(TestDeserialize)

# 如果 torchdynamo 不支持，则跳过当前测试类
@unittest.skipIf(not torchdynamo.is_dynamo_supported(), "dynamo doesn't support")
class TestSchemaVersioning(TestCase):
    def test_error(self):
        # 定义一个简单的 torch 模块用于测试
        class Module(torch.nn.Module):
            def forward(self, x):
                return x + x

        # 实例化 Module 类
        f = Module()
        # 导出模块 f，并传入示例输入
        ep = export(f, (torch.randn(1, 3),))

        # 序列化导出的程序
        serialized_program = ExportedProgramSerializer().serialize(ep)
        # 将导出的程序的 schema 版本主版本号设为 -1，触发预期的异常
        serialized_program.exported_program.schema_version.major = -1
        # 断言在反序列化时抛出预期的异常 SerializeError
        with self.assertRaisesRegex(
            SerializeError, r"Serialized schema version .* does not match our current"
        ):
            ExportedProgramDeserializer().deserialize(
                serialized_program.exported_program,
                serialized_program.state_dict,
                serialized_program.constants,
                serialized_program.example_inputs,
            )

# 期望 TestDeserialize.test_exportdb_supported_case_fn_with_kwargs 失败，因为尚未设置 kwargs 输入
unittest.expectedFailure(TestDeserialize.test_exportdb_supported_case_fn_with_kwargs)

# 期望 TestDeserialize.test_exportdb_supported_case_scalar_output 失败，因为在追踪时未能生成图形
unittest.expectedFailure(TestDeserialize.test_exportdb_supported_case_scalar_output)

# 如果 torchdynamo 不支持，则跳过当前测试类
@unittest.skipIf(not torchdynamo.is_dynamo_supported(), "dynamo doesn't support")
class TestSaveLoad(TestCase):
    def test_save_buffer(self):
        # 定义一个简单的 torch 模块用于测试
        class Module(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(2, 2)

            def forward(self, x):
                x = x + 1
                y = x.t()
                y = y.relu()
                y = self.linear(y)
                return y

        # 导出模块 Module，并传入示例输入 inp
        ep = export(Module(), inp=(torch.tensor([0.1, 0.1]),))

        # 创建一个字节流缓冲区
        buffer = io.BytesIO()
        # 将导出的模块 ep 保存到 buffer 中
        save(ep, buffer)
        # 重置缓冲区指针到起始位置
        buffer.seek(0)
        # 从缓冲区加载导出的程序
        loaded_ep = load(buffer)

        # 断言导出和加载后的模块在相同输入下输出相似的结果
        self.assertTrue(torch.allclose(ep.module()(*inp), loaded_ep.module()(*inp)))

    def test_save_file(self):
        # 定义一个简单的 torch 模块用于测试
        class Foo(torch.nn.Module):
            def forward(self, x):
                return x * x

        # 实例化模块 Foo
        f = Foo()

        # 示例输入
        inp = (torch.randn(2, 2),)
        # 导出模块 f，并传入示例输入 inp
        ep = export(f, inp)

        # 使用临时文件保存导出的模块
        with tempfile.NamedTemporaryFile() as f:
            # 将导出的模块保存到临时文件 f 中
            save(ep, f)
            # 重置文件指针到起始位置
            f.seek(0)
            # 从文件中加载导出的程序
            loaded_ep = load(f)

        # 断言导出和加载后的模块在相同输入下输出相似的结果
        self.assertTrue(torch.allclose(ep.module()(*inp), loaded_ep.module()(*inp)))

    def test_save_path(self):
        # 定义一个简单的 torch 模块用于测试
        class Foo(torch.nn.Module):
            def forward(self, x, y):
                return x + y

        # 实例化模块 Foo
        f = Foo()

        # 示例输入
        inp = (torch.tensor([6]), torch.tensor([7]))
        # 导出模块 f，并传入示例输入 inp
        ep = export(f, inp)

        # 使用临时文件名作为路径保存导出的模块
        with TemporaryFileName() as fname:
            path = Path(fname)
            # 将导出的模块保存到指定路径 path 中
            save(ep, path)
            # 从指定路径 path 加载导出的程序
            loaded_ep = load(path)

        # 断言导出和加载后的模块在相同输入下输出相似的结果
        self.assertTrue(torch.allclose(ep.module()(*inp), loaded_ep.module()(*inp)))
    def test_save_extra(self):
        # 准备输入数据，一个包含一个张量的元组
        inp = (torch.tensor([0.1, 0.1]),)

        # 定义一个简单的神经网络模块，计算输入的平方加上输入本身
        class Foo(torch.nn.Module):
            def forward(self, x):
                return x * x + x

        # 创建神经网络模块实例
        f = Foo()

        # 导出神经网络模块和输入数据，得到一个序列化的对象
        ep = export(f, inp)

        # 创建一个字节流缓冲区
        buffer = io.BytesIO()
        # 将导出的对象保存到字节流缓冲区，并添加额外文件 "extra.txt": "moo"
        save(ep, buffer, extra_files={"extra.txt": "moo"})
        buffer.seek(0)
        # 准备一个空的字典来存放额外文件内容
        extra_files = {"extra.txt": ""}
        # 从字节流缓冲区加载对象，并加载额外文件内容到 extra_files
        loaded_ep = load(buffer, extra_files=extra_files)

        # 断言：验证导出和加载后的神经网络模块输出结果相等
        self.assertTrue(torch.allclose(ep.module()(*inp), loaded_ep.module()(*inp)))
        # 断言：验证额外文件内容正确加载
        self.assertEqual(extra_files["extra.txt"], "moo")

    def test_version_error(self):
        # 定义一个简单的神经网络模块，计算输入的两倍
        class Foo(torch.nn.Module):
            def forward(self, x):
                return x + x

        # 创建神经网络模块实例
        f = Foo()

        # 导出神经网络模块和输入数据，得到一个序列化的对象
        ep = export(f, (torch.randn(1, 3),))

        # 使用临时文件来保存序列化的对象
        with tempfile.NamedTemporaryFile() as f:
            save(ep, f)
            f.seek(0)

            # 修改版本信息，向临时文件中的 ZIP 文件中写入新的版本信息
            with zipfile.ZipFile(f, "a") as zipf:
                zipf.writestr("version", "-1.1")

            # 断言：加载时应该抛出版本不匹配的异常
            with self.assertRaisesRegex(
                RuntimeError, r"Serialized version .* does not match our current"
            ):
                f.seek(0)
                load(f)

    def test_save_constants(self):
        # 定义一个包含常量和列表的神经网络模块
        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.a = torch.tensor(3)

            def forward(self, x):
                list_tensor = [torch.tensor(3), torch.tensor(4)]
                return x + self.a + list_tensor[0] + list_tensor[1]

        # 导出神经网络模块和输入数据，得到一个序列化的对象
        ep = export(Foo(), (torch.tensor(1),))
        # 创建一个字节流缓冲区
        buffer = io.BytesIO()
        # 将导出的对象保存到字节流缓冲区
        save(ep, buffer)
        buffer.seek(0)
        # 从字节流缓冲区加载对象
        loaded_ep = load(buffer)

        inp = (torch.tensor(1),)
        # 断言：验证导出和加载后的神经网络模块输出结果相等
        self.assertTrue(torch.allclose(ep.module()(*inp), loaded_ep.module()(*inp)))
@unittest.skipIf(not torchdynamo.is_dynamo_supported(), "dynamo doesn't support")
class TestSerializeCustomClass(TestCase):
    # 测试类：检查自定义类的序列化功能
    def setUp(self):
        # 设置测试环境
        super().setUp()
        # 初始化 TorchBind 实现
        init_torchbind_implementations()

    def test_custom_class(self):
        # 测试用例：测试自定义类的行为
        custom_obj = torch.classes._TorchScriptTesting._PickleTester([3, 4])

        class Foo(torch.nn.Module):
            def forward(self, x):
                return x + x

        f = Foo()

        inputs = (torch.zeros(4, 4),)
        ep = export(f, inputs)

        # 替换其中一个值为自定义类的实例
        for node in ep.graph.nodes:
            if node.op == "call_function" and node.target == torch.ops.aten.add.Tensor:
                with ep.graph.inserting_before(node):
                    # 创建自定义节点
                    custom_node = ep.graph.call_function(
                        torch.ops._TorchScriptTesting.take_an_instance.default,
                        (custom_obj,),
                    )
                    # 设置自定义节点的元数据
                    custom_node.meta["val"] = torch.ones(4, 4)
                    custom_node.meta["torch_fn"] = (
                        "take_an_instance",
                        "take_an_instance",
                    )
                    arg0, _ = node.args
                    node.args = (arg0, custom_node)

        serialized_vals = serialize(ep)

        # 将序列化的结果解码为字符串
        ep_str = serialized_vals.exported_program.decode("utf-8")
        assert "class_fqn" in ep_str
        assert custom_obj._type().qualified_name() in ep_str

        # 反序列化导出的程序
        deserialized_ep = deserialize(serialized_vals)

        # 检查反序列化后的节点
        for node in deserialized_ep.graph.nodes:
            if (
                node.op == "call_function"
                and node.target
                == torch.ops._TorchScriptTesting.take_an_instance.default
            ):
                # 获取节点的参数
                arg = node.args[0]
                self.assertTrue(isinstance(arg, torch._C.ScriptObject))
                self.assertEqual(arg._type(), custom_obj._type())
                self.assertEqual(arg.__getstate__(), custom_obj.__getstate__())
                self.assertEqual(arg.top(), 7)

    def test_custom_class_containing_fake_tensor(self):
        # 测试用例：测试包含伪张量的自定义类的行为
        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.custom_obj = torch.classes._TorchScriptTesting._ContainsTensor(
                    torch.rand(2, 3)
                )

            def forward(self, x):
                return x + self.custom_obj.get()

        with FakeTensorMode():
            f = Foo()

        inputs = (torch.zeros(2, 3),)
        with enable_torchbind_tracing():
            ep = export(f, inputs, strict=False)

        serialized_vals = serialize(ep)
        ep = deserialize(serialized_vals)
        self.assertTrue(isinstance(ep.constants["custom_obj"].get(), FakeTensor))


if __name__ == "__main__":
    run_tests()
```