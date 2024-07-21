# `.\pytorch\test\export\test_verifier.py`

```
# 引入单元测试框架 unittest
import unittest

# 引入 PyTorch 库
import torch

# 引入 functorch 中的控制流模块
from functorch.experimental import control_flow

# 引入 PyTorch 中的 Tensor 类型
from torch import Tensor

# 引入 Torch 的内部模块，用于检查是否支持动态图
from torch._dynamo.eval_frame import is_dynamo_supported

# 引入 Torch 的导出相关模块和类
from torch._export.verifier import SpecViolationError, Verifier
from torch.export import export
from torch.export.exported_program import InputKind, InputSpec, TensorArgument

# 引入 Torch 的测试工具函数和类
from torch.testing._internal.common_utils import IS_WINDOWS, run_tests, TestCase

# 如果不支持动态图，则跳过测试
@unittest.skipIf(not is_dynamo_supported(), "dynamo isn't supported")
class TestVerifier(TestCase):
    # 测试基本的 Verifier 功能
    def test_verifier_basic(self) -> None:
        # 定义一个简单的 Torch 模块 Foo
        class Foo(torch.nn.Module):
            def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                return x + y

        # 创建 Foo 的实例 f
        f = Foo()

        # 导出模型 f，传入随机生成的张量作为输入
        ep = export(f, (torch.randn(100), torch.randn(100)))

        # 创建 Verifier 实例
        verifier = Verifier()
        
        # 对导出的模型进行验证
        verifier.check(ep)

    # 测试调用模块时的 Verifier 功能
    def test_verifier_call_module(self) -> None:
        # 定义一个包含 Linear 层的 Torch 模块 M
        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(10, 10)

            def forward(self, x: Tensor) -> Tensor:
                return self.linear(x)

        # 对模块 M 进行符号跟踪
        gm = torch.fx.symbolic_trace(M())

        # 创建 Verifier 实例
        verifier = Verifier()

        # 验证 _check_graph_module 方法是否会引发 SpecViolationError 异常
        with self.assertRaises(SpecViolationError):
            verifier._check_graph_module(gm)

    # 测试不使用 functional 的 Verifier 功能
    def test_verifier_no_functional(self) -> None:
        # 定义一个简单的 Torch 模块 Foo
        class Foo(torch.nn.Module):
            def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                return x + y

        # 创建 Foo 的实例 f
        f = Foo()

        # 导出模型 f，传入随机生成的张量作为输入
        ep = export(f, (torch.randn(100), torch.randn(100)))

        # 修改导出图中特定操作的目标，使其不使用 functional
        for node in ep.graph.nodes:
            if node.target == torch.ops.aten.add.Tensor:
                node.target = torch.ops.aten.add_.Tensor

        # 创建 Verifier 实例
        verifier = Verifier()

        # 验证修改后的导出模型是否会引发 SpecViolationError 异常
        with self.assertRaises(SpecViolationError):
            verifier.check(ep)

    # 测试高阶函数的 Verifier 功能
    @unittest.skipIf(IS_WINDOWS, "Windows not supported for this test")
    def test_verifier_higher_order(self) -> None:
        # 定义一个包含条件控制流的 Torch 模块 Foo
        class Foo(torch.nn.Module):
            def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                def true_fn(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                    return x + y

                def false_fn(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                    return x - y

                return control_flow.cond(x.shape[0] > 2, true_fn, false_fn, [x, y])

        # 创建 Foo 的实例 f
        f = Foo()

        # 导出模型 f，传入特定形状的张量作为输入
        ep = export(f, (torch.randn(3, 3), torch.randn(3, 3)))

        # 创建 Verifier 实例
        verifier = Verifier()

        # 验证高阶函数调用的导出模型是否符合预期
        verifier.check(ep)
    def test_verifier_nested_invalid_module(self) -> None:
        # 定义一个内嵌的测试类，继承自torch.nn.Module
        class Foo(torch.nn.Module):
            # 重写forward方法，接受两个torch.Tensor类型参数，并返回一个torch.Tensor类型结果
            def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                # 定义一个内部函数true_fn，接受两个torch.Tensor类型参数，并返回它们的和
                def true_fn(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                    return x + y

                # 定义一个内部函数false_fn，接受两个torch.Tensor类型参数，并返回它们的差
                def false_fn(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                    return x - y

                # 调用控制流模块中的cond方法，根据条件判断选择调用true_fn或false_fn
                return control_flow.cond(x.shape[0] > 2, true_fn, false_fn, [x, y])

        # 创建Foo类的实例f
        f = Foo()

        # 导出Foo类实例f的计算图ep
        ep = export(f, (torch.randn(3, 3), torch.randn(3, 3)))

        # 遍历计算图ep中的true_graph_0的节点
        for node in ep.graph_module.true_graph_0.graph.nodes:
            # 如果节点的目标是torch.ops.aten.add.Tensor，则将其修改为torch.ops.aten.add_.Tensor
            if node.target == torch.ops.aten.add.Tensor:
                node.target = torch.ops.aten.add_.Tensor

        # 创建Verifier类的实例verifier
        verifier = Verifier()

        # 断言验证器verifier检查计算图ep时引发SpecViolationError异常
        with self.assertRaises(SpecViolationError):
            verifier.check(ep)

    def test_ep_verifier_basic(self) -> None:
        # 定义一个简单的torch.nn.Module类M
        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                # 在初始化方法中添加一个线性层
                self.linear = torch.nn.Linear(10, 10)

            # 重写forward方法，接受一个torch.Tensor类型参数x，并返回一个torch.Tensor类型结果
            def forward(self, x: Tensor) -> Tensor:
                return self.linear(x)

        # 创建M类的实例，并导出其计算图ep
        ep = export(M(), (torch.randn(10, 10),))

        # 调用计算图ep的_validate方法，验证其有效性
        ep._validate()

    def test_ep_verifier_invalid_param(self) -> None:
        # 定义一个包含参数的torch.nn.Module类M
        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                # 在初始化方法中注册一个名为"a"的参数
                self.register_parameter(
                    name="a", param=torch.nn.Parameter(torch.randn(100))
                )

            # 重写forward方法，接受两个torch.Tensor类型参数x和y，并返回一个torch.Tensor类型结果
            def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                return x + y + self.a

        # 创建M类的实例，并导出其计算图ep，传入两个随机生成的100维向量作为输入
        ep = export(M(), (torch.randn(100), torch.randn(100)))

        # 将输入规范中的第一个参数标记为InputKind.PARAMETER，使用名称"p_a"和目标"bad_param"
        ep.graph_signature.input_specs[0] = InputSpec(
            kind=InputKind.PARAMETER, arg=TensorArgument(name="p_a"), target="bad_param"
        )

        # 断言验证计算图ep时引发SpecViolationError异常，异常消息包含"not in the state dict"
        with self.assertRaisesRegex(SpecViolationError, "not in the state dict"):
            ep._validate()

        # 将状态字典ep.state_dict中添加一个名为"bad_param"的参数，值为一个随机生成的100维向量
        ep.state_dict["bad_param"] = torch.randn(100)

        # 断言验证计算图ep时引发SpecViolationError异常，异常消息包含"not an instance of torch.nn.Parameter"
        with self.assertRaisesRegex(
            SpecViolationError, "not an instance of torch.nn.Parameter"
        ):
            ep._validate()
    def test_ep_verifier_invalid_buffer(self) -> None:
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.a = torch.tensor(3.0)

            def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                return x + y + self.a

        # 导出模型 M 的推理图
        ep = export(M(), (torch.randn(100), torch.randn(100)))

        # 在推理图签名的输入规范中设置一个不存在于状态字典中的缓冲区
        ep.graph_signature.input_specs[0] = InputSpec(
            kind=InputKind.BUFFER,
            arg=TensorArgument(name="c_a"),
            target="bad_buffer",
            persistent=True,
        )
        # 断言引发特定异常，说明缓冲区不在状态字典中
        with self.assertRaisesRegex(SpecViolationError, "not in the state dict"):
            ep._validate()

    def test_ep_verifier_buffer_mutate(self) -> None:
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()

                self.my_parameter = torch.nn.Parameter(torch.tensor(2.0))

                # 注册两个缓冲区
                self.register_buffer("my_buffer1", torch.tensor(3.0))
                self.register_buffer("my_buffer2", torch.tensor(4.0))

            def forward(self, x1, x2):
                # 在前向方法中使用参数、缓冲区和两个输入
                output = (
                    x1 + self.my_parameter
                ) * self.my_buffer1 + x2 * self.my_buffer2

                # 修改一个缓冲区（例如，递增1）
                self.my_buffer2.add_(1.0)
                return output

        # 导出模型 M 的推理图
        ep = export(M(), (torch.tensor(5.0), torch.tensor(6.0)))
        # 验证推理图的有效性
        ep._validate()

    def test_ep_verifier_invalid_output(self) -> None:
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()

                self.my_parameter = torch.nn.Parameter(torch.tensor(2.0))

                # 注册两个缓冲区
                self.register_buffer("my_buffer1", torch.tensor(3.0))
                self.register_buffer("my_buffer2", torch.tensor(4.0))

            def forward(self, x1, x2):
                # 在前向方法中使用参数、缓冲区和两个输入
                output = (
                    x1 + self.my_parameter
                ) * self.my_buffer1 + x2 * self.my_buffer2

                # 修改一个缓冲区（例如，递增1）
                self.my_buffer2.add_(1.0)
                return output

        # 导出模型 M 的推理图
        ep = export(M(), (torch.tensor(5.0), torch.tensor(6.0)))

        # 获取推理图中的最后一个节点
        output_node = list(ep.graph.nodes)[-1]
        # 修改输出节点的参数
        output_node.args = (
            (
                output_node.args[0][0],
                next(iter(ep.graph.nodes)),
                output_node.args[0][1],
            ),
        )

        # 断言引发特定异常，说明输出节点的数量不符合预期
        with self.assertRaisesRegex(SpecViolationError, "Number of output nodes"):
            ep._validate()
# 如果当前脚本作为主程序运行（而不是被导入到其他脚本中执行），则执行 run_tests() 函数
if __name__ == "__main__":
    run_tests()
```