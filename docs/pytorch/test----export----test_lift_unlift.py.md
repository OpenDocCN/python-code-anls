# `.\pytorch\test\export\test_lift_unlift.py`

```py
# Owner(s): ["oncall: export"]
import unittest  # 导入单元测试模块
from typing import Any, Dict, Optional, OrderedDict, Tuple  # 导入类型提示相关模块

import torch  # 导入PyTorch库
from torch._export.passes.lift_constants_pass import (  # 从torch._export.passes.lift_constants_pass导入相关函数和类
    ConstantAttrMap,
    lift_constants_pass,
)
from torch.export._unlift import _unlift_exported_program_lifted_states  # 导入_unlift_exported_program_lifted_states函数
from torch.export.exported_program import (  # 从torch.export.exported_program导入多个类和函数
    ExportGraphSignature,
    InputKind,
    InputSpec,
    OutputKind,
    OutputSpec,
    TensorArgument,
)

from torch.export.graph_signature import CustomObjArgument  # 导入CustomObjArgument类
from torch.testing._internal.common_utils import (  # 从torch.testing._internal.common_utils导入多个函数和常量
    find_library_location,
    IS_FBCODE,
    IS_MACOS,
    IS_SANDCASTLE,
    IS_WINDOWS,
    run_tests,
    TestCase,
)


class GraphBuilder:
    def __init__(self):
        self.graph = torch.fx.Graph()  # 创建一个空的torch.fx.Graph对象
        self.nodes = {}  # 初始化节点字典
        self.values = {}  # 初始化值字典
        self.nn_module_stack_key: Dict[str, int] = {}  # 初始化神经网络模块堆栈键的字典
        self.latest_id = 0  # 初始化最新ID为0
        self.input_to_kind: Dict[torch.fx.Node, InputKind] = {}  # 初始化输入到输入类型的映射字典

    def input(self, name: str, value: torch.Tensor, kind: InputKind):
        node = self.graph.placeholder(name)  # 在图中创建一个占位节点，并命名为name
        node.meta["val"] = value  # 在节点的元数据中存储值
        self.nodes[name] = node  # 将节点添加到节点字典中
        self.values[name] = value  # 将值添加到值字典中
        self.input_to_kind[node] = kind  # 将节点与输入类型的映射关系存储起来

    def add(self, x: str, y: str, out: str, module_fqn: str = ""):
        node = self.graph.create_node(  # 创建一个"call_function"类型的节点，执行torch.ops.aten.add.Tensor操作
            "call_function",
            torch.ops.aten.add.Tensor,
            (self.nodes[x], self.nodes[y]),  # 将输入节点连接为参数
            name=out,
        )
        self.values[out] = self.values[x] + self.values[y]  # 计算并存储输出值
        node.meta["val"] = self.values[out]  # 在节点的元数据中存储值
        node.meta["nn_module_stack"] = self.create_nn_module_stack(module_fqn)  # 创建并存储神经网络模块堆栈
        self.nodes[out] = node  # 将输出节点添加到节点字典中

    def call_function(self, target, args, out: str, module_fqn: str = ""):
        arg_nodes = tuple(self.nodes[arg] for arg in args)  # 获取参数节点
        arg_values = tuple(self.values[arg] for arg in args)  # 获取参数值
        node = self.graph.create_node(  # 创建一个"call_function"类型的节点，调用目标函数
            "call_function",
            target,
            arg_nodes,
            name=out,
        )
        self.values[out] = target(*arg_values)  # 计算并存储输出值
        node.meta["val"] = self.values[out]  # 在节点的元数据中存储值
        node.meta["nn_module_stack"] = self.create_nn_module_stack(module_fqn)  # 创建并存储神经网络模块堆栈
        self.nodes[out] = node  # 将输出节点添加到节点字典中

    def constant(
        self, name: str, value: Any, target: Optional[str] = None, module_fqn: str = ""
    ):
        if target is None:
            target = name
        node = self.graph.get_attr(target)  # 获取给定名称的属性节点
        node.meta["val"] = value  # 在节点的元数据中存储值
        node.meta["nn_module_stack"] = self.create_nn_module_stack(module_fqn)  # 创建并存储神经网络模块堆栈
        self.nodes[name] = node  # 将节点添加到节点字典中
        self.values[name] = value  # 将值添加到值字典中

    def output(self, out: str):
        self.graph.output(self.nodes[out])  # 将指定节点添加到图的输出节点列表中

    def create_nn_module_stack(
        self, module_fqn: str
    ):  # 定义一个用于创建神经网络模块堆栈的方法，接受模块全限定名作为参数
    ) -> OrderedDict[int, Tuple[str, type]]:
        # 当前模块名初始化为空字符串
        cur_name = ""
        # 用于存储神经网络模块堆栈的有序字典
        nn_module_stack = OrderedDict()
        # 遍历模块全限定名中的每个原子名称
        for atom in module_fqn.split("."):
            # 如果当前模块名为空，则直接赋值为当前原子名称
            if cur_name == "":
                cur_name = atom
            else:
                # 否则，将当前模块名更新为当前模块名加上当前原子名称
                cur_name = cur_name + "." + atom

            # 如果当前模块名不在神经网络模块堆栈键集合中
            if cur_name not in self.nn_module_stack_key:
                # 获取当前最新的 ID 计数器，并将最新 ID 计数器增加 1
                id_counter = self.latest_id
                self.latest_id += 1
                # 将当前模块名与 ID 计数器的对应关系存入神经网络模块堆栈键集合中
                self.nn_module_stack_key[cur_name] = id_counter
            else:
                # 否则，从神经网络模块堆栈键集合中获取当前模块名对应的 ID 计数器
                id_counter = self.nn_module_stack_key[cur_name]

            # 将当前 ID 计数器与其对应的模块名和 torch.nn.Module 类型元组存入神经网络模块堆栈中
            nn_module_stack[id_counter] = (cur_name, torch.nn.Module)
        # 返回神经网络模块堆栈
        return nn_module_stack

    def create_input_specs(self):
        # 初始化输入规范列表
        input_specs = []
        # 遍历图中的每个节点
        for node in self.graph.nodes:
            # 如果节点的操作为占位符
            if node.op == "placeholder":
                # 将节点的输入规范添加到输入规范列表中
                input_specs.append(
                    InputSpec(
                        kind=self.input_to_kind[node],
                        arg=TensorArgument(name=node.name),
                        target=None,
                        persistent=(
                            True
                            if self.input_to_kind[node] == InputKind.BUFFER
                            else None
                        ),
                    )
                )
        # 返回输入规范列表
        return input_specs

    # NOTE: does not handle non-user-outputs atm
    def gen_graph_signature(self) -> ExportGraphSignature:
        # 获取所有输出节点
        output = [n for n in self.graph.nodes if n.op == "output"]
        # 确保只有一个输出节点
        assert len(output) == 1
        output = output[0]
        # 确保输出节点只有一个参数
        assert len(output.args) == 1, "multiple outputs NYI"

        # 生成图签名，包括输入规范和输出规范
        return ExportGraphSignature(
            input_specs=self.create_input_specs(),
            output_specs=[
                OutputSpec(
                    kind=OutputKind.USER_OUTPUT,
                    arg=TensorArgument(name=n.name),
                    target=None,
                )
                for n in output.args
            ],
        )
class TestLift(TestCase):
    # 测试类的初始化方法，在每个测试方法之前执行
    def setUp(self):
        # 如果运行在 macOS 上，则跳过测试，抛出 SkipTest 异常
        if IS_MACOS:
            raise unittest.SkipTest("non-portable load_library call used in test")
        # 如果运行在 Sandcastle 或者 FBCODE 环境下，则加载自定义库
        elif IS_SANDCASTLE or IS_FBCODE:
            torch.ops.load_library(
                "//caffe2/test/cpp/jit:test_custom_class_registrations"
            )
        # 如果运行在 Windows 上，则根据特定库文件名加载库
        elif IS_WINDOWS:
            lib_file_path = find_library_location("torchbind_test.dll")
            torch.ops.load_library(str(lib_file_path))
        # 否则，根据特定库文件名加载库（假设是 Linux 环境）
        else:
            lib_file_path = find_library_location("libtorchbind_test.so")
            torch.ops.load_library(str(lib_file_path))

    # 测试方法：测试嵌套的图构建和常量提升
    def test_lift_nested(self):
        # 创建图构建器对象
        builder = GraphBuilder()
        # 添加用户输入节点 'x'，并设置输入类型为 USER_INPUT
        builder.input("x", torch.rand(2, 3), InputKind.USER_INPUT)
        # 添加用户输入节点 'y'，并设置输入类型为 USER_INPUT
        builder.input("y", torch.rand(2, 3), InputKind.USER_INPUT)
        # 添加用户输入节点 'z'，并设置输入类型为 USER_INPUT
        builder.input("z", torch.rand(2, 3), InputKind.USER_INPUT)

        # 在图中添加操作：将 'x' 和 'y' 相加，输出为 'foo'
        builder.add("x", "y", out="foo")
        # 在图中添加操作：将 'foo' 和 'z' 相加，输出为 'bar'，并指定模块全名为 'foo'
        builder.add("foo", "z", out="bar", module_fqn="foo")
        # 在图中添加常量节点 'const_tensor'，内容为随机生成的张量，模块全名为 'foo'
        builder.constant("const_tensor", torch.rand(2, 3), module_fqn="foo")
        # 在图中添加操作：将 'bar' 和 'const_tensor' 相加，输出为 'out'
        builder.add("bar", "const_tensor", "out")
        # 将 'out' 指定为图的输出节点
        builder.output("out")

        # 获取构建完成的图对象
        graph = builder.graph
        # 对图进行静态分析和检查
        graph.lint()

        # 获取常量 'const_tensor' 的引用
        const_tensor = builder.values["const_tensor"]
        # 构建根节点字典，包含常量 'const_tensor'
        root = {"const_tensor": builder.values["const_tensor"]}

        # 生成图的签名，描述图中的输入和输出
        graph_signature = builder.gen_graph_signature()
        # 创建图模块对象 'gm'，包含根节点和构建的图
        gm = torch.fx.GraphModule(root, graph)

        # 对常量进行提升处理，返回提升后的常量字典
        constants = lift_constants_pass(gm, graph_signature, {})
        # 对提升后的图进行静态分析和检查
        gm.graph.lint()

        # 断言提升的常量数量为 1
        self.assertEqual(len(constants), 1)

        # 断言常量表中应包含提升的常量 'foo.lifted_tensor_0'
        self.assertIn("foo.lifted_tensor_0", constants)
        # 断言提升的常量值与预期的常量 'const_tensor' 相同
        self.assertEqual(constants["foo.lifted_tensor_0"], const_tensor)

        # 断言图中不应存在操作为 'get_attr' 的节点
        getattr_nodes = [n for n in gm.graph.nodes if n.op == "get_attr"]
        self.assertEqual(len(getattr_nodes), 0)

        # 断言图中应存在操作为 'placeholder' 的节点，数量为 4
        placeholder_nodes = [n for n in gm.graph.nodes if n.op == "placeholder"]
        self.assertEqual(len(placeholder_nodes), 4)

        # 断言提升的常量节点应位于用户输入节点之前，但在参数/缓冲区之后
        lifted_constant_placeholder = placeholder_nodes[0]
        self.assertEqual(lifted_constant_placeholder.target, "lifted_tensor_0")

        # 断言图签名应已被修改，以反映新的占位符节点
        constant_input_spec = graph_signature.input_specs[0]
        self.assertEqual(constant_input_spec.kind, InputKind.CONSTANT_TENSOR)
        self.assertIsInstance(constant_input_spec.arg, TensorArgument)
        self.assertEqual(constant_input_spec.arg.name, lifted_constant_placeholder.name)
    def test_duplicate_constant_access(self):
        const = torch.rand(2, 3)  # 创建一个形状为 (2, 3) 的随机张量 const
        const_obj = torch.classes._TorchScriptTesting._Foo(10, 20)  # 使用 _Foo 类创建一个常量对象 const_obj

        builder = GraphBuilder()  # 创建一个图构建器实例 builder
        builder.input("x", torch.rand(2, 3), InputKind.USER_INPUT)  # 添加一个名为 "x" 的用户输入节点
        builder.constant("const_tensor", const, target="const_tensor")  # 添加一个名为 "const_tensor" 的常量节点，内容为 const 张量
        # 第二次加载相同的目标
        builder.constant("const_tensor2", const, target="const_tensor")  # 添加另一个名为 "const_tensor2" 的常量节点，内容也为 const 张量

        # 第二次加载相同的对象但使用不同的目标
        builder.constant("const_obj", const_obj)  # 添加一个名为 "const_obj" 的常量节点，内容为 const_obj 对象
        builder.constant("const_obj2", const_obj)  # 添加另一个名为 "const_obj2" 的常量节点，内容也为 const_obj 对象
        builder.call_function(
            torch.ops._TorchScriptTesting.takes_foo,
            ("const_obj", "x"),
            out="torchbind_out",
        )  # 调用一个函数，将 "const_obj" 和 "x" 作为参数传递，并指定输出为 "torchbind_out"
        builder.call_function(
            torch.ops._TorchScriptTesting.takes_foo,
            ("const_obj2", "x"),
            out="torchbind_out2",
        )  # 再次调用同一个函数，将 "const_obj2" 和 "x" 作为参数传递，并指定输出为 "torchbind_out2"
        builder.add("x", "const_tensor", out="foo")  # 添加一个加法操作，将 "x" 和 "const_tensor" 相加，结果保存为 "foo"
        builder.add("foo", "const_tensor2", out="tensor_out")  # 添加另一个加法操作，将 "foo" 和 "const_tensor2" 相加，结果保存为 "tensor_out"
        builder.add("torchbind_out", "torchbind_out2", out="obj_out")  # 添加一个加法操作，将 "torchbind_out" 和 "torchbind_out2" 相加，结果保存为 "obj_out"
        builder.add("tensor_out", "obj_out", out="out")  # 添加最后一个加法操作，将 "tensor_out" 和 "obj_out" 相加，结果保存为 "out"
        builder.output("out")  # 将 "out" 设置为输出节点
        graph = builder.graph  # 获取构建完成的图

        graph.lint()  # 对图进行 lint 检查

        input_specs = builder.create_input_specs()  # 创建输入规格
        output_specs = [
            OutputSpec(
                kind=OutputKind.USER_OUTPUT,
                arg=TensorArgument(name=builder.nodes["out"].name),
                target=None,
            )
        ]  # 创建输出规格，指定输出类型为用户输出
        graph_signature = ExportGraphSignature(input_specs, output_specs)  # 创建图的导出签名

        root = {"const_tensor": const, "const_obj": const_obj, "const_obj2": const_obj}  # 定义根节点，包含 const 和 const_obj 对象
        gm = torch.fx.GraphModule(root, graph)  # 创建图模块，传入根节点和构建的图

        constants = lift_constants_pass(gm, graph_signature, {})  # 对图模块进行常量提升操作
        gm.graph.lint()  # 对处理后的图进行 lint 检查

        self.assertEqual(len(constants), 2)  # 断言常量的数量为 2

        # 所有的 get_attr 节点应该被移除
        getattr_nodes = [n for n in gm.graph.nodes if n.op == "get_attr"]
        self.assertEqual(len(getattr_nodes), 0)  # 断言不存在 get_attr 节点

        # 应该只有两个额外的输入节点（加上现有的用户输入节点）
        placeholder_nodes = [n for n in gm.graph.nodes if n.op == "placeholder"]
        self.assertEqual(len(placeholder_nodes), 3)  # 断言存在三个占位符节点

        # 图签名应该被修改以反映占位符的存在
        self.assertEqual(len(graph_signature.input_specs), 3)  # 断言输入规格中有三个元素
        constant_input_spec = graph_signature.input_specs[0]
        self.assertEqual(constant_input_spec.kind, InputKind.CONSTANT_TENSOR)  # 断言第一个输入规格的类型为 CONSTANT_TENSOR
        self.assertIsInstance(constant_input_spec.arg, TensorArgument)  # 断言第一个输入规格的参数是 TensorArgument 类型
    # 定义一个测试方法，测试非持久性缓冲区的行为
    def test_unlift_nonpersistent_buffer(self):
        # 定义一个继承自torch.nn.Module的类Foo
        class Foo(torch.nn.Module):
            # 类的初始化方法
            def __init__(self):
                super().__init__()
                # 注册一个名为"non_persistent_buf"的非持久性缓冲区，初始值为torch.zeros(1)
                self.register_buffer(
                    "non_persistent_buf", torch.zeros(1), persistent=False
                )

            # 前向传播方法
            def forward(self, x):
                # 非持久性缓冲区增加1
                self.non_persistent_buf.add_(1)
                # 返回输入张量x的和，加上非持久性缓冲区的和
                return x.sum() + self.non_persistent_buf.sum()

        # 创建类Foo的实例foo
        foo = Foo()
        # 导出foo的模型表示，将其存储在exported变量中
        exported = torch.export.export(foo, (torch.ones(5, 5),), strict=False)
        # 使用_unlift_exported_program_lifted_states函数将导出的模型表示解析成有状态的图形模型(stateful_gm)
        stateful_gm = _unlift_exported_program_lifted_states(exported)

        # 检查解析后的stateful_gm是否包含原始的非持久性缓冲区
        self.assertTrue(hasattr(stateful_gm, "non_persistent_buf"))
        # 获取stateful_gm中名为"non_persistent_buf"的缓冲区对象
        non_persistent_buf = stateful_gm.get_buffer("non_persistent_buf")
        # 断言解析后的非持久性缓冲区对象与原始foo对象中的非持久性缓冲区对象相等
        self.assertEqual(non_persistent_buf, foo.get_buffer("non_persistent_buf"))
        # 断言"non_persistent_buf"存在于stateful_gm的非持久性缓冲区集合中
        self.assertIn("non_persistent_buf", stateful_gm._non_persistent_buffers_set)
        # 断言"non_persistent_buf"不在stateful_gm的状态字典(state_dict)中
        self.assertNotIn("non_persistent_buf", stateful_gm.state_dict())
class ConstantAttrMapTest(TestCase):
    def setUp(self):
        # 如果运行在 macOS 上，跳过测试，因为测试中使用了非可移植的 load_library 调用
        if IS_MACOS:
            raise unittest.SkipTest("non-portable load_library call used in test")
        # 如果运行在 Sandcastle 或者 FBCODE 环境下，加载自定义库
        elif IS_SANDCASTLE or IS_FBCODE:
            torch.ops.load_library(
                "//caffe2/test/cpp/jit:test_custom_class_registrations"
            )
        # 如果运行在 Windows 环境下，根据库文件路径加载 DLL 文件
        elif IS_WINDOWS:
            lib_file_path = find_library_location("torchbind_test.dll")
            torch.ops.load_library(str(lib_file_path))
        # 否则，假定运行在类 Unix 系统下，根据库文件路径加载 SO 文件
        else:
            lib_file_path = find_library_location("libtorchbind_test.so")
            torch.ops.load_library(str(lib_file_path))

    def test_dict_api(self):
        # 创建 ConstantAttrMap 对象用于测试
        constant_attr_map = ConstantAttrMap()
        # 创建一个 torch 脚本类的实例 const_obj
        const_obj = torch.classes._TorchScriptTesting._Foo(10, 20)
        # 创建一个全为 1 的 torch Tensor 实例 const_tensor
        const_tensor = torch.ones(2, 3)
        # 向 constant_attr_map 中添加 const_obj 和对应的属性路径 "foo.bar"
        constant_attr_map.add(const_obj, "foo.bar")
        # 向 constant_attr_map 中添加 const_tensor 和对应的属性路径 "foo.bar.baz"
        constant_attr_map.add(const_tensor, "foo.bar.baz")
        # 断言 constant_attr_map 的长度为 2
        self.assertEqual(len(constant_attr_map), 2)
        # 断言 constant_attr_map 中的键列表为 [const_obj, const_tensor]
        self.assertEqual(list(constant_attr_map), [const_obj, const_tensor])
        # 断言 constant_attr_map 的键列表为 [const_obj, const_tensor]
        self.assertEqual(list(constant_attr_map.keys()), [const_obj, const_tensor])
        # 断言 constant_attr_map 的值列表为 [["foo.bar"], ["foo.bar.baz"]]
        self.assertEqual(
            list(constant_attr_map.values()), [["foo.bar"], ["foo.bar.baz"]]
        )
        # 断言 constant_attr_map 中 const_obj 对应的值为 ["foo.bar"]
        self.assertEqual(constant_attr_map[const_obj], ["foo.bar"])
        # 断言 constant_attr_map 中 const_tensor 对应的值为 ["foo.bar.baz"]
        self.assertEqual(constant_attr_map[const_tensor], ["foo.bar.baz"])
        # 断言 const_obj 是否存在于 constant_attr_map 中
        self.assertTrue(const_obj in constant_attr_map)
        # 使用 assertRaises 检查添加不合法类型时是否会抛出 TypeError
        with self.assertRaises(TypeError):
            constant_attr_map.add(1, "foo.bar")

        # 删除 constant_attr_map 中的 const_obj
        del constant_attr_map[const_obj]
        # 再次断言 constant_attr_map 的长度为 1
        self.assertEqual(len(constant_attr_map), 1)


if __name__ == "__main__":
    run_tests()
```