# `.\pytorch\test\jit\test_async.py`

```py
# Owner(s): ["oncall: jit"]

# 导入必要的模块
import os  # 导入操作系统相关的功能
import sys  # 导入系统相关的功能

# 导入类型提示相关的模块
from typing import Any, Tuple

# 导入 PyTorch 相关模块
import torch  # 导入 PyTorch 主模块
import torch.nn as nn  # 导入 PyTorch 的神经网络模块

# 将 test/ 目录下的 helper 文件设为可导入
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)

# 导入类型提示相关的模块
from typing import List  # 导入列表类型提示

# 导入 PyTorch 的张量类型
from torch import Tensor  # 导入张量类型
from torch.jit import Future  # 导入异步 Future 类型
from torch.testing._internal.jit_utils import _inline_everything, JitTestCase  # 导入测试工具类


class TestAsync(JitTestCase):
    def test_async_python(self):
        # 定义一个 Torch 脚本函数 foo，对输入张量取负
        @torch.jit.script
        def foo(x):
            return torch.neg(x)

        # 创建一个随机张量 x
        x = torch.rand(3, 4)
        # 异步执行 foo 函数，返回一个 Future 对象
        fut = torch.jit.fork(foo, x)
        # 同步执行 foo 函数，得到 y_hat 张量
        y_hat = foo(x)
        # 等待 fut 对象的结果
        y = torch.jit.wait(fut)
        # 不做断言，仅用于验证假的 Python 路径是否可行

    def test_async_future_type_python(self):
        # 定义一个普通的 Python 函数 foo，接收输入，并返回多个 Future 对象的列表
        def foo(inp):
            futures = torch.jit.annotate(List[torch.jit.Future[torch.Tensor]], [])
            for i in range(5):
                futures.append(torch.jit.fork(lambda x: x, inp))
            all_outputs = []
            for future in futures:
                all_outputs.append(torch.jit.wait(future))
            return all_outputs

        # 不做断言，仅用于验证 Python 类型解析是否正常工作
        foo(torch.randn(3, 4))

    def test_async_parsing(self):
        # 定义一个 Torch 脚本函数 foo，接收一个张量 x，并返回两个张量组成的列表
        @torch.jit.script
        def foo(x: Tensor) -> List[Tensor]:
            return [torch.neg(x), x.t()]

        # 定义另一个 Torch 脚本函数 bar，接收一个输入 x，并返回多个 Future 对象的列表
        @torch.jit.script
        def bar(x):
            futures = torch.jit.annotate(List[Future[List[Tensor]]], [])
            for _ in range(3):
                # 使用 torch.jit.annotate 明确指定 Future 对象的类型
                future = torch.jit.annotate(
                    Future[List[Tensor]], torch.jit.fork(foo, x)
                )
                futures.append(future)

            output = torch.jit.annotate(List[List[Tensor]], [])
            for i in range(3):
                output.append(torch.jit.wait(futures[i]))
            return output

        # 创建一个随机张量 x
        x = torch.rand(3, 3)
        # 调用 bar 函数，得到结果
        result = bar(x)
        # 断言结果列表长度为 3
        self.assertEqual(len(result), 3)

    def test_async_script(self):
        # 定义一个 Torch 脚本函数 foo，接收一个张量 x，返回它的负值和自身
        @torch.jit.script
        def foo(x):
            return torch.neg(x), x

        # 创建一个随机张量 x
        x = torch.rand(3, 4)

        # 定义另一个 Torch 脚本函数 wait_script，执行 foo 的 fork 和 wait 操作
        @torch.jit.script
        def wait_script(x):
            fut = torch.jit.fork(foo, x)
            y_hat = foo(x)
            y = torch.jit.wait(fut)
            return y, y_hat

        # 调用 wait_script 函数，得到 y 和 y_hat
        y, y_hat = wait_script(x)

        # 断言 y 和 y_hat 相等
        self.assertEqual(y, y_hat)
    def test_async_script_capture(self):
        # 定义一个继承自 torch.jit.ScriptModule 的类 Mod
        class Mod(torch.jit.ScriptModule):
            # 定义常量列表 __constants__
            __constants__ = ["const"]

            # 初始化方法
            def __init__(self):
                # 调用父类初始化方法
                super().__init__()
                # 设定常量 const 的值为 42
                self.const = 42
                # 创建一个包含随机值的可训练参数 param
                self.param = nn.Parameter(torch.randn(2, 2))

            # 定义一个脚本方法 foo，接受两个参数 x1 和 x2
            @torch.jit.script_method
            def foo(self, x1, x2):
                # 返回 torch.neg(x1)、self.param、self.const、torch.neg(x2) 和 self.param
                return torch.neg(x1), self.param, self.const, torch.neg(x2), self.param

            # 定义一个脚本方法 forward，接受两个参数 x1 和 x2
            @torch.jit.script_method
            def forward(self, x1, x2):
                # 使用 torch.jit.fork 异步执行 foo 方法
                fut = torch.jit.fork(self.foo, x1, x2)
                # 同步调用 foo 方法
                y_hat = self.foo(x1, x2)
                # 等待 fut 的完成
                y = torch.jit.wait(fut)
                # 返回 y 和 y_hat
                return y, y_hat

        # 创建输入数据 x1 和 x2
        x1 = torch.rand(3, 4)
        x2 = torch.rand(5, 6)

        # 创建 Mod 的实例 m
        m = Mod()

        # 禁用优化执行
        with torch.jit.optimized_execution(False):
            # 调用 m 的 forward 方法，获取结果 y 和 y_hat
            y, y_hat = m.forward(x1, x2)

        # 断言 y 和 y_hat 相等
        self.assertEqual(y, y_hat)

    def test_async_script_nested(self):
        # 定义一个脚本函数 foo，接受一个参数 x
        @torch.jit.script
        def foo(x):
            # 返回 torch.neg(x) 和 x
            return torch.neg(x), x

        # 创建输入数据 x
        x = torch.rand(3, 4)

        # 定义一个脚本函数 wait_script，接受一个参数 x
        @torch.jit.script
        def wait_script(x):
            # 使用 torch.jit._fork 异步执行 foo 方法
            fut = torch.jit._fork(foo, x)
            # 同步调用 foo 方法
            y_hat = foo(x)
            # 等待 fut 的完成
            y = torch.jit._wait(fut)
            # 返回 y 和 y_hat
            return y, y_hat

        # 定义一个脚本函数 wait_script_nest，接受一个参数 x
        @torch.jit.script
        def wait_script_nest(x):
            # 使用 torch.jit._fork 异步执行 wait_script 方法
            fut = torch.jit._fork(wait_script, x)
            # 等待 fut 的完成
            return torch.jit._wait(fut)

        # 调用 wait_script_nest 方法，获取结果 y 和 y_hat
        y, y_hat = wait_script_nest(x)

        # 断言 y 和 y_hat 相等
        self.assertEqual(y, y_hat)

    def test_async_script_no_script_mod(self):
        # 创建输入数据 x
        x = torch.rand(3, 4)

        # 使用 assertRaisesRegexWithHighlight 检查是否引发 RuntimeError
        with self.assertRaisesRegexWithHighlight(
            RuntimeError, "cannot call a value", "torch.jit._fork(x"
        ):
            # 定义一个脚本函数 wait_script，接受一个参数 x
            @torch.jit.script
            def wait_script(x):
                # 使用 torch.jit._fork 异步执行 x
                fut = torch.jit._fork(x)
                # 返回 fut
                return fut

    def test_async_script_multi_waits(self):
        # 定义一个脚本函数 foo，接受一个参数 x
        @torch.jit.script
        def foo(x):
            # 返回 torch.neg(x).t() + x
            return torch.neg(x).t() + x

        # 定义一个脚本函数 wait_script，接受一个参数 x
        @torch.jit.script
        def wait_script(x):
            # 使用 torch.jit._fork 异步执行 foo 方法
            fut = torch.jit._fork(foo, x)

            # 对同一个 future 进行两次等待
            y1 = torch.jit._wait(fut)
            y2 = torch.jit._wait(fut)
            # 返回 y1 和 y2
            return y1, y2

        # 创建输入数据 x
        x = torch.rand(2, 2)
        # 调用 wait_script 方法，获取结果 y1 和 y2
        y1, y2 = wait_script(x)
        # 断言 y1 和 y2 相等
        self.assertEqual(y1, y2)
    # 定义一个测试方法，用于测试多个异步脚本的并发执行
    def test_async_script_multi_forks(self):
        # 定义一个 Torch Script 函数 foo1，对输入张量进行负值操作并转置，然后与原始张量相加
        @torch.jit.script
        def foo1(x):
            return torch.neg(x).t() + x

        # 定义一个 Torch Script 函数 foo2，对两个输入张量进行负值操作并转置，然后相加
        @torch.jit.script
        def foo2(x, y):
            return torch.neg(x).t() + x + torch.neg(y).t()

        # 定义一个 Torch Script 函数 foo3，对三个输入张量进行负值操作并转置，然后相加
        @torch.jit.script
        def foo3(x, y, z):
            return torch.neg(z).t() + y.t() + x

        # 创建三个随机张量
        x1 = torch.rand(10, 10)
        x2 = torch.rand(10, 10)
        x3 = torch.rand(10, 10)

        # 定义一个 Torch Script 函数 wait_script，通过 _fork 和 _wait 控制异步执行多个 Torch Script 函数
        @torch.jit.script
        def wait_script(x1, x2, x3):
            # 使用 _fork 启动多个 foo 函数的异步执行
            f1 = torch.jit._fork(foo1, x1)
            f2 = torch.jit._fork(foo2, x1, x2)
            f3 = torch.jit._fork(foo3, x1, x2, x3)
            f4 = torch.jit._fork(foo1, x2)
            f5 = torch.jit._fork(foo2, x2, x3)

            # 等待异步任务完成
            y1 = torch.jit._wait(f1)
            y2 = torch.jit._wait(f2)
            y3 = torch.jit._wait(f3)

            # 返回异步任务的结果
            return y1, y2, y3

        # 调用 wait_script 函数获取异步执行结果
        y1, y2, y3 = wait_script(x1, x2, x3)
        # 使用断言验证异步执行结果与预期的 foo 函数结果相等
        self.assertEqual(y1, foo1(x1))
        self.assertEqual(y2, foo2(x1, x2))
        self.assertEqual(y3, foo3(x1, x2, x3))

    # 定义一个测试方法，用于测试异步函数处理关键字参数的情况
    def test_async_kwargs(self):
        # 定义一个简单的函数 foo，接受两个输入张量并返回它们的线性组合
        def foo(x1, x2):
            return 2 * x1 + x2

        # 创建两个随机张量
        x1 = torch.rand(3, 4)
        x2 = torch.rand(3, 4)
        # 计算预期的函数输出结果
        y_hat = foo(x1, x2)

        # 遍历多个参数组合的函数包装方式，包括裸函数、使用不同参数排列的 Torch Script 包装
        for func in [
            lambda x1, x2: torch.jit._wait(torch.jit._fork(foo, x1, x2)),
            lambda x1, x2: torch.jit._wait(torch.jit._fork(foo, x1, x2=x2)),
            lambda x1, x2: torch.jit._wait(torch.jit._fork(foo, x1=x1, x2=x2)),
            lambda x1, x2: torch.jit._wait(torch.jit._fork(foo, x2=x2, x1=x1)),
        ]:
            # 遍历包装函数的方式，包括裸函数和使用 Torch Script 追踪的方式
            for wrapper in [
                func,
                torch.jit.trace(func, (x1, x2)),
            ]:
                # 使用断言验证每种参数组合方式的函数执行结果与预期输出 y_hat 相等
                self.assertEqual(wrapper(x1, x2), y_hat)
                self.assertEqual(wrapper(x1, x2=x2), y_hat)
                self.assertEqual(wrapper(x1=x1, x2=x2), y_hat)
                self.assertEqual(wrapper(x2=x2, x1=x1), y_hat)

        # 遍历 Torch Script 包装的函数方式，包括基于参数位置和关键字的两种封装方式
        @torch.jit.script
        def foo_script_args(x1, x2):
            return torch.jit._wait(torch.jit._fork(foo, x1, x2))

        @torch.jit.script
        def foo_script_kwargs(x1, x2):
            return torch.jit._wait(torch.jit._fork(foo, x1=x1, x2=x2))

        for wrapper in [
            foo_script_args,
            foo_script_kwargs,
        ]:
            # 使用断言验证每种 Torch Script 封装方式的函数执行结果与预期输出 y_hat 相等
            self.assertEqual(wrapper(x1, x2), y_hat)
            self.assertEqual(wrapper(x1, x2=x2), y_hat)
            self.assertEqual(wrapper(x1=x1, x2=x2), y_hat)
            self.assertEqual(wrapper(x2=x2, x1=x1), y_hat)

    # 定义一个装饰器，标记在此处但未提供具体实现
    @_inline_everything
    def test_async_script_trace(self):
        # 定义一个继承自 nn.Module 的内部类 Traced，重写 forward 方法
        class Traced(nn.Module):
            def forward(self, x):
                return (torch.neg(x), x)

        # 定义一个继承自 torch.jit.ScriptModule 的内部类 Mod
        class Mod(torch.jit.ScriptModule):
            def __init__(self):
                super().__init__()
                # 创建一个 3x3 的随机张量 x
                x = torch.rand(3, 3)
                # 使用 torch.jit.trace 方法将 Traced 类追踪为 ScriptModule
                self.traced = torch.jit.trace(Traced(), (x), _force_outplace=True)

            @torch.jit.script_method
            # 定义 forward 方法，输入为 Tensor x，输出为复杂的嵌套结构
            def forward(
                self, x: Tensor
            ) -> Tuple[List[Tensor], Tuple[Tensor, Tensor], Tensor]:
                # 使用 torch.jit._fork 方法异步执行 self.traced
                future1 = torch.jit._fork(self.traced, x)
                # 使用 torch.jit._fork 方法异步执行 torch.neg 函数
                future2 = torch.jit._fork(torch.neg, x)

                # 等待 future1 和 future2 执行完毕
                tensor_tuple = torch.jit._wait(future1)
                tensor_single = torch.jit._wait(future2)

                # 将结果组织成列表
                tensor_list = []
                tensor_list.append(tensor_tuple[0])
                tensor_list.append(tensor_single)

                # 返回一个嵌套结构的元组
                return (tensor_list, tensor_tuple, tensor_tuple[1])

        # 定义一个继承自 nn.Module 的内部类 TupleCl
        class TupleCl(nn.Module):
            def __init__(self):
                super().__init__()
                # 创建 Mod 的实例
                self.module = Mod()

            def forward(self, x):
                # 对输入张量 x 应用 torch.neg 函数
                z = torch.neg(x)
                # 调用 self.module 的 forward 方法
                y = self.module(x)
                # 将结果组织成列表 list
                list = [z, y[0][0], y[0][1], y[1][0], y[1][1], y[2]]
                # 返回一个元组
                return tuple(list)

        # 创建一个 3x3 的随机张量 x
        x = torch.rand(3, 3)
        # 使用 torch.jit.trace 方法将 TupleCl 类追踪为 ScriptModule
        module = torch.jit.trace(TupleCl(), (x), _force_outplace=True)

        # 断言模块的计算图包含两个 prim::fork 节点
        self.assertGraphContainsExactly(
            module.graph, kind="prim::fork", num_kind_nodes=2
        )
        # 断言模块的计算图中有一个 aten::neg 节点
        self.assertGraphContainsExactly(
            module.graph, kind="aten::neg", num_kind_nodes=1
        )
        # 断言模块的计算图中有三个 aten::neg 节点，包括子图
        self.assertGraphContainsExactly(
            module.graph, kind="aten::neg", num_kind_nodes=3, consider_subgraphs=True
        )

        # 对输入张量 x 应用 torch.neg 函数并断言模块的输出与预期一致
        y = torch.neg(x)
        self.assertEqual(module(x), (y, y, y, y, x, x))
    def test_async_script_error(self):
        # 创建一个 3x4 的随机张量
        x = torch.rand(3, 4)

        # 定义一个 Torch 脚本函数 foo，对输入张量进行转置并加上自身，但存在错误
        @torch.jit.script
        def foo(x):
            # error here
            return x.t() + x

        # 定义另一个 Torch 脚本函数 wait_script，使用 _fork 异步调用 foo，并返回 _wait 的结果
        @torch.jit.script
        def wait_script(x):
            fut = torch.jit._fork(foo, x)
            return torch.jit._wait(fut)

        # 定义第三个 Torch 脚本函数 wait_script_nest，嵌套调用 wait_script
        @torch.jit.script
        def wait_script_nest(x):
            fut = torch.jit._fork(wait_script, x)
            return torch.jit._wait(fut)

        # 在测试中验证对于 foo 函数调用的异常情况
        # 错误消息：The size.*must match the size of tensor，应该出现在 "x.t() + x" 中
        with self.assertRaisesRegexWithHighlight(Exception, error_msg, "x.t() + x"):
            foo(x)

        # 在测试中验证对于 wait_script 函数调用的异常情况
        # 错误消息：The size.*must match the size of tensor，应该出现在 "torch.jit._fork(foo, x)" 中
        with self.assertRaisesRegexWithHighlight(
            Exception, error_msg, "torch.jit._fork(foo, x"
        ):
            wait_script(x)

        # 在测试中验证对于 wait_script_nest 函数调用的异常情况
        x = torch.rand(3, 4, 5)
        # 错误消息：expects a tensor with <= 2 dimensions，应该出现在 "torch.jit._fork(wait_script, x)" 中
        with self.assertRaisesRegexWithHighlight(
            Exception,
            "expects a tensor with <= 2 dimensions",
            "torch.jit._fork(wait_script, x",
        ):
            wait_script_nest(x)

    def test_async_grad_guard_with_grad(self):
        # 定义一个 Torch 脚本函数 foo，对输入张量乘以2并返回其 requires_grad 属性
        @torch.jit.script
        def foo(x):
            y = x * 2
            return y.requires_grad

        # 定义一个 Torch 脚本函数 bar，使用 _fork 异步调用 foo，并返回 _wait 的结果
        @torch.jit.script
        def bar(x):
            fut = torch.jit._fork(foo, x)
            requires_grad_in_fork = torch.jit._wait(fut)
            z = x * 2
            return (requires_grad_in_fork, z.requires_grad)

        # 创建一个 requires_grad=True 的随机张量 x
        x = torch.randn(3, requires_grad=True)

        # 在 Torch 的 enable_grad 上下文中运行 bar 函数，并验证结果
        with torch.enable_grad():
            (inside_fork, after_wait) = bar(x)

        # 断言结果，期望 inside_fork 和 after_wait 都为 True
        self.assertEqual(inside_fork, True)
        self.assertEqual(after_wait, True)

    def test_async_grad_guard_no_grad(self):
        # 定义一个 Torch 脚本函数 foo，对输入张量乘以2并返回其 requires_grad 属性
        @torch.jit.script
        def foo(x):
            y = x * 2
            return y.requires_grad

        # 定义一个 Torch 脚本函数 bar，使用 _fork 异步调用 foo，并返回 _wait 的结果
        @torch.jit.script
        def bar(x):
            fut = torch.jit._fork(foo, x)
            requires_grad_in_fork = torch.jit._wait(fut)
            z = x * 2
            return (requires_grad_in_fork, z.requires_grad)

        # 创建一个 requires_grad=True 的随机张量 x
        x = torch.randn(3, requires_grad=True)

        # 在 Torch 的 no_grad 上下文中运行 bar 函数，并验证结果
        with torch.no_grad():
            (inside_fork, after_wait) = bar(x)

        # 断言结果，期望 inside_fork 和 after_wait 都为 False
        self.assertEqual(inside_fork, False)
        self.assertEqual(after_wait, False)
    # 定义测试函数 test_trace_fork_wait，用于测试 torch.jit._fork 和 torch.jit._wait 的功能
    def test_trace_fork_wait(self):
        # 定义内部函数 fork_body，对输入参数 x 执行负值操作和负值加一操作，并返回结果
        def fork_body(x):
            return x.neg(), x.neg() + 1

        # 定义内部函数 fn，接受参数 x，使用 torch.jit._fork 对 fork_body 函数进行并行执行，
        # 使用 torch.jit._wait 等待执行结果，同时返回三个值：第一个和第二个是 fork_body 函数的结果，
        # 第三个是 x 减一的结果
        def fn(x):
            fut = torch.jit._fork(fork_body, x)
            vals = torch.jit._wait(fut)
            return vals[0], vals[1], x - 1

        # 对 fn 函数进行追踪转换为 torch.jit.ScriptModule 对象 traced
        traced = torch.jit.trace(fn, (torch.rand(3, 4),))
        # 随机生成输入张量 x
        x = torch.rand(3, 4)
        # 断言 fn(x) 的结果与 traced(x) 的结果相等
        self.assertEqual(fn(x), traced(x))

        # 断言 traced.graph 中包含且仅包含一个 kind 类型为 "prim::fork" 的节点
        self.assertGraphContainsExactly(
            traced.graph, kind="prim::fork", num_kind_nodes=1
        )
        # 断言 traced.graph 中包含且仅包含一个 kind 类型为 "aten::wait" 的节点
        self.assertGraphContainsExactly(
            traced.graph, kind="aten::wait", num_kind_nodes=1
        )
        # 断言 traced.graph 中包含且仅包含两个 kind 类型为 "aten::neg" 的节点（考虑子图）
        self.assertGraphContainsExactly(
            traced.graph, kind="aten::neg", num_kind_nodes=2, consider_subgraphs=True
        )

    # 定义测试函数 test_trace_fork_wait_leaking，用于测试在 tracer 中发现数据依赖关系泄漏的情况
    def test_trace_fork_wait_leaking(self):
        # 定义空列表 my_list
        my_list = []

        # 定义内部函数 fork_body，对输入参数 x 执行加一操作，并将结果添加到 my_list 中，然后返回
        def fork_body(x):
            my_list.append(x + 1)
            return x + 1

        # 定义内部函数 fn，接受参数 x，使用 torch.jit._fork 对 fork_body 函数进行并行执行，
        # 使用 torch.jit._wait 等待执行结果，返回 my_list 中的第一个元素
        def fn(x):
            fut = torch.jit._fork(fork_body, x)
            val = torch.jit._wait(fut)
            return my_list[0]

        # 使用断言捕获 RuntimeError 异常，其消息中包含泄漏检测的关键信息
        with self.assertRaisesRegexWithHighlight(
            RuntimeError,
            "did not have observable data dependence with trace inputs; "
            "this probably indicates your program cannot be understood "
            "by the tracer.",
            "",
        ):
            # 对 fn 函数进行追踪转换为 torch.jit.ScriptModule 对象 traced，关闭检查追踪
            traced = torch.jit.trace(fn, (torch.rand(3, 4),), check_trace=False)

    # 定义测试函数 test_trace_fork_wait_inline，用于测试在追踪后内联处理 torch.jit._fork 和 torch.jit._wait 的功能
    def test_trace_fork_wait_inline(self):
        # 定义内部函数 fork_body，对输入参数 x 执行加一和加二操作，并返回结果
        def fork_body(x):
            return x + 1, x + 2

        # 定义内部函数 fn，接受参数 x，使用 torch.jit._fork 对 fork_body 函数进行并行执行，
        # 使用 torch.jit._wait 等待执行结果，返回 val 中的第二个元素
        def fn(x):
            fut = torch.jit._fork(fork_body, x)
            val = torch.jit._wait(fut)
            return val[1]

        # 对 fn 函数进行追踪转换为 torch.jit.ScriptModule 对象 traced
        traced = torch.jit.trace(fn, (torch.rand(3, 4),))
        # 在 traced.graph 中内联处理所有的 prim::fork 和 aten::wait 节点
        torch._C._jit_pass_inline_fork_wait(traced.graph)
        # 断言 traced.graph 中不包含任何 kind 类型为 "prim::fork" 的节点
        self.assertGraphContainsExactly(
            traced.graph, kind="prim::fork", num_kind_nodes=0
        )
        # 断言 traced.graph 中不包含任何 kind 类型为 "aten::wait" 的节点
        self.assertGraphContainsExactly(
            traced.graph, kind="aten::wait", num_kind_nodes=0
        )
        # 断言 traced.graph 中包含且仅包含两个 kind 类型为 "aten::add" 的节点
        self.assertGraphContainsExactly(
            traced.graph, kind="aten::add", num_kind_nodes=2
        )
    def test_trace_fork_wait_list_modulecalls(self):
        def add_one(input):
            return input + torch.ones(input.size())

        class TestListFutureModule(nn.Module):
            def forward(self, input):
                input_list = []
                for i in range(3):
                    input_list.append(input)

                fut_list: List[Future[torch.Tensor]] = []
                for input_tensor in input_list:
                    fut_list.append(torch.jit._fork(add_one, input_tensor))
                # return list[future[tensor]] here to ensure tracing
                # module calls return the correct types
                return fut_list

        class TestModuleWrapper(nn.Module):
            def __init__(self):
                super().__init__()
                self.list_fut_mod = TestListFutureModule()

            def forward(self, input):
                fut_list = self.list_fut_mod(input)
                res = input
                for fut in fut_list:
                    res = res + fut.wait()
                return res

        self.checkTrace(TestModuleWrapper(), (torch.randn(5, 5),))

    def test_trace_modulecalls_with_different_output_types(self):
        def add_one(input):
            return input + torch.ones(input.size())

        class DifferentOutputModule(nn.Module):
            def forward(self, input):
                fut_res = torch.jit._fork(add_one, (input))

                # return different types from module call
                return input, fut_res

        class TestModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.gen_output = DifferentOutputModule()

            def forward(self, input):
                res, fut_res = self.gen_output(input)
                res = res + fut_res.wait()
                return res

        self.checkTrace(TestModule(), (torch.randn(5, 5),))

    def test_no_future_subtype_message(self):
        with self.assertRaisesRegexWithHighlight(
            RuntimeError, "Future without a contained type", ""
        ):
            # Define a TorchScript function that expects a list of futures
            @torch.jit.script
            def forward(self, x):
                futs = torch.jit.annotate(List[torch.jit.Future], [])


注释：


            # 在测试中捕获期望的运行时错误，确保出现“Future without a contained type”错误信息
            # 定义一个 TorchScript 函数，期望接收一个 Future 列表作为参数
    def test_future_subtyping(self):
        """
        Test that futures subtype each other properly.
        """

        # Successful subtyping.
        # 定义一个函数，接受整数参数并返回整数结果
        def returns_int(x: int) -> int:
            return x + x + 1

        # 定义一个函数，接受整数参数并返回一个任意类型的 Torch Future 对象
        def returns_future_any(x: int) -> torch.jit.Future[Any]:
            return torch.jit._fork(returns_int, (x))

        # Torch JIT 脚本函数，接受整数参数并返回其等待的 Future 对象
        @torch.jit.script
        def fn_int(x: int) -> Any:
            fut = returns_future_any(x)
            return fut.wait()

        # Unsuccessful subtyping.
        # 使用断言检查是否引发特定类型的异常，同时包含自定义的错误消息部分
        with self.assertRaisesRegexWithHighlight(
            RuntimeError,
            r"was annotated as having type Future\[float\] but is actually of type Future\[int\]",
            "fut = returns_future_float(x",
        ):

            # 定义一个函数，接受整数参数并返回期望的浮点数类型的 Torch Future 对象
            def returns_future_float(x: int) -> torch.jit.Future[float]:
                return torch.jit._fork(returns_int, (x))

            # Torch JIT 脚本函数，接受整数参数并返回其等待的 Future 对象
            @torch.jit.script
            def fn_float(x: int) -> Any:
                fut = returns_future_float(x)
                return fut.wait()
# 如果当前文件作为主程序运行时，抛出运行时错误并显示相关提示信息。
if __name__ == "__main__":
    raise RuntimeError(
        "This test file is not meant to be run directly, use:\n\n"
        "\tpython test/test_jit.py TESTNAME\n\n"
        "instead."
    )
```