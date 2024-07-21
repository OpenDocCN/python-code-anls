# `.\pytorch\test\dynamo\test_backward_higher_order_ops.py`

```py
# Owner(s): ["module: dynamo"]
# flake8: noqa

# 导入 functools 模块，用于高阶函数支持
import functools

# 导入 PyTorch 库
import torch

# 导入 PyTorch Dynamo 模块下的测试相关内容
import torch._dynamo.test_case

# 导入 PyTorch Dynamo 模块下的 testing 工具
import torch._dynamo.testing

# 导入 PyTorch Dynamo 模块下的实用工具
import torch._dynamo.utils

# 从 torch 模块中导入 _inductor
from torch import _inductor as inductor

# 从 torch._dynamo 中导入 compiled_autograd 模块
from torch._dynamo import compiled_autograd

# 从 torch._dynamo._trace_wrapped_higher_order_op 模块中导入 trace_wrapped 函数
from torch._dynamo._trace_wrapped_higher_order_op import trace_wrapped

# 从 torch._dynamo.testing 模块中导入 normalize_gm 函数
from torch._dynamo.testing import normalize_gm

# 从 torch._dynamo.utils 模块中导入 counters 函数
from torch._dynamo.utils import counters

# 从 torch.fx.experimental.proxy_tensor 模块中导入 make_fx 函数
from torch.fx.experimental.proxy_tensor import make_fx


# 定义一个私有函数 _multiply，用于计算输入 x 的平方
def _multiply(x):
    return x * x


# 定义一个私有函数 _multiply_invoke，将输入 grad 传递给 _multiply 函数并进行跟踪
def _multiply_invoke(grad):
    return trace_wrapped(grad, fn=_multiply)


# 定义一个测试类 BackwardHigherOrderOpTests，继承自 torch._dynamo.test_case.TestCase
class BackwardHigherOrderOpTests(torch._dynamo.test_case.TestCase):

    # 定义测试方法 test_invoke_in_eager，测试在 eager 模式下的函数调用
    def test_invoke_in_eager(self):
        # 创建两个张量 x 和 y，并声明需要计算梯度
        x = torch.tensor([0.5, 0.5], requires_grad=True)
        y = torch.tensor([0.5, 0.5], requires_grad=True)

        # 定义一个函数 fn，注册 _multiply_invoke 作为 x 的钩子函数，并返回 x * y
        def fn(x, y):
            x.register_hook(_multiply_invoke)
            return x * y

        # 调用 fn 函数计算结果 out
        out = fn(x, y)

        # 定义梯度 grad_out
        grad_out = torch.tensor([2.0, 2.0])

        # 对 out 进行反向传播
        out.backward(grad_out)

        # 断言 x 的梯度与 y * grad_out 相等
        self.assertEqual(x.grad, y * grad_out)

    # 定义测试方法 test_invoke_in_pt2，测试在不同后端模式下的函数调用
    def test_invoke_in_pt2(self):
        # 遍历多个后端模式
        for backend in ["eager", "aot_eager", "inductor"]:
            # 重置 torch._dynamo 的状态
            torch._dynamo.reset()

            # 创建两个张量 x 和 y，并声明需要计算梯度
            x = torch.tensor([0.5, 0.5], requires_grad=True)
            y = torch.tensor([0.5, 0.5], requires_grad=True)

            # 定义一个函数 fn，注册 _multiply_invoke 作为 x 的钩子函数，并返回 x * y
            def fn(x, y):
                x.register_hook(_multiply_invoke)
                return x * y

            # 优化 fn 函数的后端模式，并重新赋值给 fn
            fn = torch._dynamo.optimize(backend)(fn)

            # 调用 fn 函数计算结果 out
            out = fn(x, y)

            # 定义梯度 grad_out
            grad_out = torch.tensor([2.0, 2.0])

            # 对 out 进行反向传播
            out.backward(grad_out)

            # 断言 x 的梯度与 grad_out * y 相等
            self.assertEqual(x.grad, grad_out * y)

    # 定义测试方法 test_invoke_make_fx_forward_contrived，测试使用 make_fx 的前向传播
    def test_invoke_make_fx_forward_contrived(self):
        # 创建一个张量 x，并声明需要计算梯度
        x = torch.tensor([0.5, 0.5], requires_grad=True)

        # 调用 make_fx 函数，并将 _multiply_invoke 作为参数传递给它
        out = make_fx(_multiply_invoke)(x)

        # 断言 out(x) 的结果与 torch.tensor([0.25, 0.25]) 相等
        self.assertEqual(out(x), torch.tensor([0.25, 0.25]))

        # 获取 normalize_gm(out.print_readable(False)) 的结果，并赋值给 actual
        actual = normalize_gm(out.print_readable(False))

        # 断言 actual 的输出与预期的字符串匹配
        self.assertExpectedInline(
            actual,
            """\
class _multiply_invoke(torch.nn.Module):
    def forward(self, grad_1: "f32[2]"):
        trace_wrapped: "f32[2]" = torch__dynamo__trace_wrapped_higher_order_op_self_invoke(grad_1);  grad_1 = None
        return trace_wrapped
""",
        )

    # 定义测试方法 test_invoke_make_bw，测试使用 make_fx 的反向传播
    def test_invoke_make_bw(self):
        # 创建一个张量 x，并声明需要计算梯度
        x = torch.tensor([0.5, 0.5], requires_grad=True)

        # 定义一个函数 fwd，计算 x * x，并返回其两倍的结果
        def fwd(x):
            z = x * x
            return z + z

        # 调用 fwd 函数计算结果 res
        res = fwd(x)

        # 对 res 进行反向传播
        res.backward(torch.tensor([1.0, 1.0]))

        # 调用 make_fx 函数，并将 _multiply_invoke 作为参数传递给它
        out = make_fx(_multiply_invoke)(x.grad)

        # 断言 out(x.grad) 的结果与 torch.tensor([4.0, 4.0]) 相等
        self.assertEqual(out(x.grad), torch.tensor([4.0, 4.0]))

        # 获取 normalize_gm(out.print_readable(False)) 的结果，并赋值给 actual
        actual = normalize_gm(out.print_readable(False))

        # 断言 actual 的输出与预期的字符串匹配
        self.assertExpectedInline(
            actual,
            """\
class _multiply_invoke(torch.nn.Module):
    def forward(self, grad_1: "f32[2]"):
        trace_wrapped: "f32[2]" = torch__dynamo__trace_wrapped_higher_order_op_self_invoke(grad_1);  grad_1 = None
        return trace_wrapped
""",
        )
        # 定义测试函数 test_invoke_in_pt2_compiled_autograd
        def test_invoke_in_pt2_compiled_autograd(self):
            # 初始化变量 graph 为 None
            graph = None

            # 定义编译器函数 compiler_fn，接受 gm 参数
            def compiler_fn(gm):
                # 定义内部编译器函数 inner_compiler，接受 gm_ 和 example_inputs_ 参数
                def inner_compiler(gm_, example_inputs_):
                    nonlocal graph  # 使用外部定义的 graph 变量
                    self.assertEqual(graph, None)  # 断言当前 graph 为 None
                    graph = gm_  # 将 gm_ 赋给外部变量 graph
                    return inductor.compile(gm_, example_inputs_)  # 编译 gm_ 并返回结果

                # 调用 torch.compile，使用 inner_compiler 作为 backend
                return torch.compile(
                    gm, backend=inner_compiler, fullgraph=True, dynamic=True
                )

            # 遍历不同的 backend
            for backend in ["eager", "aot_eager", "inductor"]:
                torch._dynamo.reset()  # 重置 torch._dynamo 状态
                x = torch.tensor([0.5, 0.5], requires_grad=True)  # 创建张量 x
                y = torch.tensor([0.5, 0.5], requires_grad=True)  # 创建张量 y

                # 定义函数 fn，注册 x 的 hook _multiply_invoke，并返回 x + y
                def fn(x, y):
                    x.register_hook(_multiply_invoke)
                    return x + y

                # 优化 fn 函数的执行，根据当前 backend 进行优化
                fn = torch._dynamo.optimize(backend)(fn)

                # 调用 fn 函数计算输出 out
                out = fn(x, y)

                # 设置梯度 grad_out
                grad_out = torch.tensor([2.0, 2.0])

                # 启用编译后自动求导环境
                with compiled_autograd.enable(compiler_fn):
                    out.backward(grad_out)  # 对 out 进行反向传播

                # 标准化 graph 的可读输出，并赋值给 actual
                actual = normalize_gm(graph.print_readable(False))

                # 断言计算得到的梯度 x.grad 与 grad_out * grad_out 相等
                self.assertEqual(x.grad, grad_out * grad_out)

                # 断言 actual 的内容符合预期的内联字符串输出
                self.assertExpectedInline(
                    actual,
                    """\
class GraphModule(torch.nn.Module):
    def forward(self, L_inputs_ : list):
        l_inputs_ = L_inputs_

        # 从输入列表中获取第一个元素，并赋值给getitem变量；清空l_inputs_
        getitem: "f32[s0]" = l_inputs_[0];  l_inputs_ = None

        # 使用torch.clone复制getitem的值，存储在new_grad变量中
        new_grad: "f32[s0]" = torch.clone(getitem)

        # 计算getitem的平方，并将结果存储在result变量中；清空getitem
        result: "f32[s0]" = getitem * getitem;  getitem = None

        # 使用torch.clone复制result的值，存储在new_grad_1变量中；清空result
        new_grad_1: "f32[s0]" = torch.clone(result);  result = None
        
        # 返回new_grad和new_grad_1作为结果
        return (new_grad, new_grad_1)
    def test_invoke_in_pt2_compiled_autograd_graph_breaks(self):
        # 定义一个会破坏自动求导图的函数
        def _graph_breaking_fn(x):
            # 打印 "Boo!"
            print("Boo!")
            # 调用 _multiply 函数处理输入 x
            return _multiply(x)

        # 定义一个调用图破坏函数的函数
        def _graph_break_invoke(grad):
            # 调用 trace_wrapped 函数来追踪 grad，并传递 _graph_breaking_fn 函数
            return trace_wrapped(grad, fn=_graph_breaking_fn)

        # 定义一个编译函数，使用 torch 的编译器将计算图编译为特定后端的形式
        def compiler_fn(gm):
            return torch.compile(gm, backend="inductor", fullgraph=True, dynamic=True)

        # 遍历不同的后端选项
        for backend in ["eager", "aot_eager", "inductor"]:
            # 重置 torch._dynamo 状态
            torch._dynamo.reset()
            # 创建两个张量 x 和 y，并设置 requires_grad=True，表示需要计算梯度
            x = torch.tensor([0.5, 0.5], requires_grad=True)
            y = torch.tensor([0.5, 0.5], requires_grad=True)

            # 定义一个函数 fn，注册 _graph_break_invoke 函数来处理 x 的梯度
            def fn(x, y):
                x.register_hook(_graph_break_invoke)
                return x + y

            # 使用 torch._dynamo.optimize 对 fn 进行优化，根据后端选择优化策略
            fn = torch._dynamo.optimize(backend, nopython=True)(fn)
            # 调用优化后的 fn 函数计算输出
            out = fn(x, y)
            # 定义梯度值 grad_out
            grad_out = torch.tensor([2.0, 2.0])
            # 使用断言确保在编译的自动求导中包含特定的异常信息
            with self.assertRaisesRegex(
                torch._dynamo.exc.Unsupported,
                "print",
            ):
                # 启用编译的自动求导，并调用 out 的 backward 方法传入 grad_out
                with compiled_autograd.enable(compiler_fn):
                    out.backward(grad_out)

            # 将 graph 设置为 None
            graph = None
# 如果这个脚本作为主程序被执行
if __name__ == "__main__":
    # 从torch._dynamo.test_case模块中导入run_tests函数
    from torch._dynamo.test_case import run_tests

    # 运行测试函数run_tests()
    run_tests()
```