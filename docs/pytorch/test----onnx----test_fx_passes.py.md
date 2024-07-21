# `.\pytorch\test\onnx\test_fx_passes.py`

```
# Owner(s): ["module: onnx"]
import pytorch_test_common  # 导入测试公共模块

import torch  # 导入PyTorch模块
import torch._dynamo  # 导入PyTorch内部动态图模块
import torch.fx  # 导入PyTorch的特征图模块

from torch.onnx._internal.fx.passes import _utils as pass_utils  # 导入ONNX内部特征图的通用工具模块
from torch.testing._internal import common_utils  # 导入PyTorch内部测试公共工具模块


class TestFxPasses(common_utils.TestCase):
    def test_set_node_name_correctly_renames_when_new_name_collides_recursively(self):
        def func(x, y, z):
            return x + y + z

        x = torch.randn(3)  # 创建一个形状为(3,)的随机张量x
        y = torch.randn(3)  # 创建一个形状为(3,)的随机张量y
        z = torch.randn(3)  # 创建一个形状为(3,)的随机张量z
        gm, _ = torch._dynamo.export(func)(x, y, z)  # 使用动态图导出函数func，并获取导出的图gm
        torch._dynamo.reset()  # 重置动态图状态

        # Purposely name the nodes in a way that will cause a recursive collision later.
        # See :func:`set_node_name` for name collision renaming logic.
        base_name = "tensor"  # 设定基础名称为"tensor"
        nodes = list(gm.graph.nodes)  # 获取图gm中的所有节点列表
        for i, node in enumerate(nodes[1:]):
            if i == 0:
                node.name = base_name  # 将第一个节点命名为base_name
            else:
                node.name = f"{base_name}.{i}"  # 将后续节点命名为"base_name.i"

        # Run `set_node_name` and verify that the names are correct.
        name_to_node = {node.name: node for node in gm.graph.nodes}  # 创建节点名称到节点对象的映射字典
        pass_utils.set_node_name(nodes[0], base_name, name_to_node)  # 调用set_node_name函数，设置第一个节点的名称
        assert nodes[0].name == base_name, f"Expected {base_name}, got {nodes[0].name}"  # 断言第一个节点的名称是否为base_name
        assert len({node.name for node in nodes}) == len(
            nodes
        ), f"Expected all names to be unique, got {nodes}"  # 断言所有节点的名称是否唯一

    def test_set_node_name_succeeds_when_no_name_collisions(self):
        def func(x, y, z):
            return x + y + z

        x = torch.randn(3)  # 创建一个形状为(3,)的随机张量x
        y = torch.randn(3)  # 创建一个形状为(3,)的随机张量y
        z = torch.randn(3)  # 创建一个形状为(3,)的随机张量z
        gm, _ = torch._dynamo.export(func)(x, y, z)  # 使用动态图导出函数func，并获取导出的图gm
        torch._dynamo.reset()  # 重置动态图状态

        # Run `set_node_name` and verify that the names are correct.
        new_name = "some_tensor"  # 设定新的节点名称为"some_tensor"
        nodes = list(gm.graph.nodes)  # 获取图gm中的所有节点列表
        name_to_node = {node.name: node for node in nodes}  # 创建节点名称到节点对象的映射字典
        pass_utils.set_node_name(nodes[1], new_name, name_to_node)  # 调用set_node_name函数，设置第二个节点的名称
        assert nodes[1].name == new_name, f"Expected {new_name}, got {nodes[0].name}"  # 断言第二个节点的名称是否为new_name
        assert len({node.name for node in nodes}) == len(
            nodes
        ), f"Expected all names to be unique, got {nodes}"  # 断言所有节点的名称是否唯一
    # 定义一个测试方法，用于验证当模型包含不支持的 FX 节点时是否会引发异常
    def test_onnx_dynamo_export_raises_when_model_contains_unsupported_fx_nodes(self):
        
        # 定义自定义操作 foo_op，用于处理 torch.Tensor 类型的输入并返回相应的结果
        @torch.library.custom_op(
            "mylibrary::foo_op", device_types="cpu", mutates_args=()
        )
        def foo_op(x: torch.Tensor) -> torch.Tensor:
            return x + 1
        
        # 定义自定义操作 bar_op，用于处理 torch.Tensor 类型的输入并返回相应的结果
        @torch.library.custom_op(
            "mylibrary::bar_op", device_types="cpu", mutates_args=()
        )
        def bar_op(x: torch.Tensor) -> torch.Tensor:
            return x + 2
        
        # 注册 foo_op 的虚拟实现，返回一个与输入张量 x 类型相同的空张量
        @foo_op.register_fake
        def _(x):
            return torch.empty_like(x)
        
        # 注册 bar_op 的虚拟实现，返回一个与输入张量 x 类型相同的空张量
        @bar_op.register_fake
        def _(x):
            return torch.empty_like(x)
        
        # 定义一个函数 func，接受三个输入参数 x, y, z，并返回它们经过 foo_op, bar_op 处理后的结果
        def func(x, y, z):
            return foo_op(x) + bar_op(y) + z
        
        # 生成三个随机张量 x, y, z，每个张量包含三个元素
        x = torch.randn(3)
        y = torch.randn(3)
        z = torch.randn(3)
        
        # 使用 assertRaises 上下文管理器，期望捕获 torch.onnx.OnnxExporterError 异常
        with self.assertRaises(torch.onnx.OnnxExporterError) as ctx:
            # 调用 torch.onnx.dynamo_export 导出函数 func，并传入参数 x, y, z
            torch.onnx.dynamo_export(func, x, y, z)
        
        # 获取异常上下文中的原因异常对象
        inner_exception = ctx.exception.__cause__
        
        # 使用正则表达式匹配异常信息，检查是否包含 "Unsupported FX nodes" 以及自定义操作的名称
        self.assertRegex(
            str(inner_exception),
            r"Unsupported FX nodes.*mylibrary\.foo_op.*mylibrary\.bar_op",
        )
        
        # 重置 torch._dynamo 模块，清除所有的动态注册操作
        torch._dynamo.reset()
@common_utils.instantiate_parametrized_tests
class TestModularizePass(common_utils.TestCase):
    @pytorch_test_common.xfail(
        error_message="'torch_nn_modules_activation_GELU_used_gelu_1' not found",
        reason="optimizer",
    )
    @common_utils.parametrize(
        "is_exported_program",
        [
            common_utils.subtest(
                True,
                name="exported_program",
            ),
            common_utils.subtest(
                False,
                name="nn_module",
            ),
        ],
    )
    def test_modularize_pass_succeeds_when_submodule_output_is_unused(
        self, is_exported_program
    ):
        # This is an ill-formed model, but exporter must not crash.
        # It is illegal for submodule to have zero output. For modularization pass it can happen
        # when the submodule output is unused, so no inner node is connected to any outer
        # nodes.
        # However, this also means the entire submodule should be erased by DCE. Hence
        # it should never occur.
        #
        # Minified repro from Background_Matting. https://github.com/pytorch/benchmark/issues/1768
        
        # 定义一个测试模块，用于验证模块化处理在子模块输出未使用时是否成功
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.unused_relu = torch.nn.ReLU()  # 创建一个未使用的 ReLU 激活函数对象
                self.used_gelu = torch.nn.GELU()    # 创建一个使用的 GELU 激活函数对象

            def forward(self, x, y):
                result = self.used_gelu(x + y)     # 对输入 x + y 应用 GELU 激活函数
                unused_relu_result = self.unused_relu(x)  # 对输入 x 应用未使用的 ReLU 激活函数
                return result

        if is_exported_program:
            model = torch.export.export(
                TestModule(), args=(torch.randn(3), torch.randn(3))
            )
        else:
            model = TestModule()

        onnx_program = torch.onnx.dynamo_export(model, torch.randn(3), torch.randn(3))  # 将模型导出为 ONNX 格式的程序
        model_proto = onnx_program.model_proto  # 获取导出模型的协议
        function_proto_names = [function.name for function in model_proto.functions]  # 提取模型中函数的名称列表
        self.assertIn(
            "torch_nn_modules_activation_GELU_used_gelu_1", function_proto_names  # 断言模型中是否包含特定的 GELU 函数
        )
        self.assertFalse(any("ReLU" in name for name in function_proto_names))  # 断言模型中不包含任何 ReLU 相关的函数

    @pytorch_test_common.xfail(
        error_message="'torch_nn_modules_activation_ReLU_relu_1' not found",
        reason="optimizer",
    )
    @common_utils.parametrize(
        "is_exported_program",
        [
            common_utils.subtest(
                True,
                name="exported_program",
            ),
            common_utils.subtest(
                False,
                name="nn_module",
            ),
        ],
    )
    def test_modularize_pass_succeeds_when_a_submodule_is_called_multiple_times(
        self, is_exported_program
    ):
    ):
        # 定义一个测试模块，继承自torch.nn.Module
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 初始化模块，添加一个ReLU激活函数作为模块的成员变量
                self.relu = torch.nn.ReLU()

            def forward(self, x, y):
                # 前向传播函数定义，接收输入x和y
                out = x + y
                out = self.relu(out)  # 使用ReLU激活函数处理输出
                out = out + x
                out = self.relu(out)  # 再次使用ReLU激活函数处理输出
                return out

        # 根据is_exported_program参数选择不同的模型导出方式
        if is_exported_program:
            model = torch.export.export(
                TestModule(), args=(torch.randn(3), torch.randn(3))
            )
        else:
            model = TestModule()

        # 使用torch.onnx.dynamo_export函数将模型导出为ONNX程序
        onnx_program = torch.onnx.dynamo_export(model, torch.randn(3), torch.randn(3))
        model_proto = onnx_program.model_proto
        # 获取导出的模型中的函数名称列表
        function_proto_names = [function.name for function in model_proto.functions]
        # 断言ReLU激活函数名称在模型的函数名称列表中
        self.assertIn("torch_nn_modules_activation_ReLU_relu_1", function_proto_names)
        self.assertIn("torch_nn_modules_activation_ReLU_relu_2", function_proto_names)

    @pytorch_test_common.xfail(
        error_message="'torch_nn_modules_activation_ReLU_inner_module_relu_1' not found",
        reason="optimizer",
    )
    @common_utils.parametrize(
        "is_exported_program",
        [
            common_utils.subtest(
                True,
                name="exported_program",
            ),
            common_utils.subtest(
                False,
                name="nn_module",
            ),
        ],
    )
    # 定义测试函数test_modularize_pass_succeeds_when_a_submodule_is_called_from_multiple_layers
    def test_modularize_pass_succeeds_when_a_submodule_is_called_from_multiple_layers(
        self, is_exported_program
        # Minified repro from basic_gnn_edgecnn.
        # 定义一个内部模块，继承自 torch.nn.Module
        class InnerModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 初始化一个 ReLU 激活函数层
                self.relu = torch.nn.ReLU()

            # 定义前向传播方法
            def forward(self, x):
                # 对输入 x 应用 ReLU 激活函数并返回结果
                return self.relu(x)

        # 定义一个测试模块，继承自 torch.nn.Module
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 初始化一个内部模块实例
                self.inner_module = InnerModule()

            # 定义前向传播方法，接受输入 x 和 y
            def forward(self, x, y):
                # 计算 x 和 y 的和
                out = x + y
                # 将和作为输入传递给内部模块，并获取输出
                out = self.inner_module(out)
                # 将内部模块的输出与 x 相加
                out = out + x
                # 对内部模块的输出再次应用 ReLU 激活函数
                out = self.inner_module.relu(out)
                # 返回最终的输出结果
                return out

        # 如果设置为导出程序模式，则导出 TestModule 实例
        if is_exported_program:
            model = torch.export.export(
                TestModule(), args=(torch.randn(3), torch.randn(3))
            )
        else:
            # 否则创建一个 TestModule 实例
            model = TestModule()

        # 使用 torch.onnx.dynamo_export 导出模型为 ONNX 程序
        onnx_program = torch.onnx.dynamo_export(model, torch.randn(3), torch.randn(3))
        # 获取导出的模型的 protobuf 定义
        model_proto = onnx_program.model_proto
        # 获取模型中所有函数的名称列表
        function_proto_names = [function.name for function in model_proto.functions]
        # 断言是否存在特定的函数名称
        self.assertIn(
            "torch_nn_modules_activation_ReLU_inner_module_relu_1", function_proto_names
        )
        self.assertIn(
            "torch_nn_modules_activation_ReLU_inner_module_relu_2", function_proto_names
        )
        # 在测试环境中，本地模块的限定名称可能会因不同的测试调用方法而不稳定
        self.assertTrue(
            any("InnerModule_inner_module_1" in name for name in function_proto_names)
        )
# 如果当前脚本作为主程序执行（而不是被导入到其他模块），则执行以下代码块
if __name__ == "__main__":
    # 调用 common_utils 模块中的 run_tests 函数，用于执行测试用例
    common_utils.run_tests()
```