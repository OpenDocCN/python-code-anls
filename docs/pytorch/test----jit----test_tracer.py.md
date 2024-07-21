# `.\pytorch\test\jit\test_tracer.py`

```
# Owner(s): ["oncall: jit"]

# 导入必要的库和模块
import copy  # 导入 copy 模块，用于复制对象
import io  # 导入 io 模块，用于处理文件流
import os  # 导入 os 模块，用于操作系统相关功能
import sys  # 导入 sys 模块，用于系统相关的参数和功能
import unittest  # 导入 unittest 模块，用于编写和运行单元测试

import torch  # 导入 PyTorch 深度学习框架
import torch.nn as nn  # 导入神经网络模块
import torch.nn.functional as F  # 导入神经网络函数模块
from torch.autograd import Function, Variable  # 导入自动求导函数和变量
from torch.testing import FileCheck  # 导入用于测试的文件检查工具

# Make the helper files in test/ importable
# 设置使得 test/ 中的辅助文件可以被导入
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
import warnings  # 导入警告模块

# Standard library
# 标准库
from collections import namedtuple  # 导入命名元组
from itertools import chain  # 导入 itertools 中的链式迭代工具
from typing import Dict, List, Optional, Tuple  # 导入类型提示工具

from torch import Tensor  # 导入张量类型
from torch.testing._internal.common_cuda import with_tf32_off  # 导入 CUDA 相关工具
from torch.testing._internal.common_utils import (
    enable_profiling_mode_for_profiling_tests,  # 启用性能测试模式
    IS_SANDCASTLE,  # 是否运行在 Sandcastle 平台
    skipIfCompiledWithoutNumpy,  # 如果未编译 NumPy 则跳过
    skipIfCrossRef,  # 如果存在交叉引用则跳过
    skipIfTorchDynamo,  # 如果是 TorchDynamo 则跳过
    suppress_warnings,  # 抑制警告
    TemporaryFileName,  # 临时文件名
)
from torch.testing._internal.jit_utils import (
    _tmp_donotuse_dont_inline_everything,  # 临时使用，不要内联所有内容
    _trace,  # 进行跟踪
    enable_cpu_fuser,  # 启用 CPU 融合器
    JitTestCase,  # JIT 测试案例
    make_global,  # 设置全局变量
    RUN_CUDA,  # 运行 CUDA
    RUN_CUDA_MULTI_GPU,  # 运行多 GPU CUDA
)

if __name__ == "__main__":
    # 如果当前脚本作为主程序运行，则抛出运行时错误，建议通过指定的方式运行
    raise RuntimeError(
        "This test file is not meant to be run directly, use:\n\n"
        "\tpython test/test_jit.py TESTNAME\n\n"
        "instead."
    )


@skipIfTorchDynamo("Not a suitable test for TorchDynamo")
class TestTracer(JitTestCase):
    @unittest.skipIf(not RUN_CUDA, "requires CUDA")
    def test_large_nbr_kernel_args(self):
        # 定义一个继承自 nn.Module 的循环神经网络类
        class Recurrence(nn.Module):
            def __init__(self, seq_len):
                super().__init__()
                self.seq_len = seq_len

            def forward(self, input):
                # 调整输入张量的维度顺序
                input = input.transpose(0, 1)

                # 主循环
                output = []
                for i in range(self.seq_len):
                    # 对输入的每个元素乘以 2，并加入到输出列表中
                    b = input[i] * 2
                    output.append(b)

                # 将输出列表中的张量连接成一个张量，并调整维度
                output = torch.cat(output, 0).view(input.size(0), *output[0].size())
                output = output.transpose(0, 1)
                return output

        # 设置输入大小、批处理大小和序列长度
        input_size = 8
        batch_size = 2
        seq_len = 130

        # 创建 Recurrence 类的实例
        rec = Recurrence(seq_len)
        # 创建随机输入张量
        input = torch.rand(batch_size, seq_len, input_size)

        # 设置当前 CUDA 设备为第一个 GPU
        torch.cuda.set_device(0)
        # 将 Recurrence 实例移动到 CUDA 上
        rec = rec.cuda()
        # 将输入张量移动到 CUDA 上
        input = input.cuda()

        # 对 Recurrence 类进行跟踪
        traced_rec = torch.jit.trace(rec, (input))

    def test_trace_legacy_ctor(self):
        # 定义一个简单的 nn.Module 子类 MyModule
        class MyModule(nn.Module):
            def forward(self, x):
                # 返回输入 x 加 1 和一个包含单个浮点数 0 的张量
                return (x + 1, torch.FloatTensor([0]))

        # 对 MyModule 类进行跟踪
        traced_rec = torch.jit.trace(MyModule(), torch.randn(2, 2))

    def test_simple(self):
        # 定义两个 requires_grad=True 的张量
        x = torch.tensor([0.4], requires_grad=True)
        y = torch.tensor([0.7], requires_grad=True)

        # 定义一个函数 f，对 x 和 y 进行操作并返回结果
        def f(x, y):
            return torch.sigmoid(torch.tanh(x * (x + y)))

        # 调用 JitTestCase 类的 checkTrace 方法，验证函数 f 的跟踪
        self.checkTrace(f, (x, y))
    def test_trace_checking_with_global_name(self):
        class MyClass(torch.nn.Module):
            def forward(self, xs: List[Tensor]):
                y = torch.cat(xs, dim=0)  # 将输入张量列表在指定维度上进行拼接
                return y

        model = MyClass()
        # 模拟这些输入在全局变量中的情况，如在脚本的最外层定义的情况
        global input1, input2
        input1 = torch.ones(2, 2)  # 创建一个2x2的张量，所有元素初始化为1
        input2 = torch.ones(2, 2)  # 创建另一个2x2的张量，所有元素初始化为1
        m2 = torch.jit.trace(model, ((input1, input2),))  # 对模型进行追踪，传入输入元组(input1, input2)

    def test_trace_aliased_parameter(self):
        class M(nn.Module):
            def __init__(self, x):
                super().__init__()
                self.x = nn.Parameter(x)  # 将输入张量x封装为模型参数

            def forward(self, y):
                return self.x + y  # 返回模型参数x与输入张量y的和

        m = M(torch.rand(3, 4))  # 创建一个M类的实例，参数为一个3x4的随机张量
        r = torch.jit.trace(m, m.x)  # 对模型m的参数x进行追踪
        t2 = torch.rand(3, 4)  # 创建一个3x4的随机张量t2
        self.assertEqual(r(t2), m.x + t2)  # 断言追踪后的模型r对输入t2的计算结果与直接计算m.x + t2的结果相等

    def test_trace_nested_fn(self):
        class TracedInlineDecision(torch.nn.Module):
            def forward(self, x, flag):
                @torch.jit.script
                def make_decision(flag, x):
                    if flag:
                        return x
                    else:
                        return torch.zeros_like(x)  # 根据条件flag返回x或者形状与x相同的全0张量

                x = torch.neg(x)  # 对输入张量x逐元素取负
                return make_decision(flag, x)  # 调用内部定义的make_decision函数并返回结果

        decision = TracedInlineDecision()
        torch.jit.trace(
            decision,
            (torch.rand(3, 4), torch.tensor([True], dtype=torch.bool)),  # 传入追踪的模型decision的输入参数
            check_trace=True,  # 启用追踪时的检查
        )

    def test_trace_single_tuple(self):
        x = torch.tensor(2.0)  # 创建一个标量张量x，值为2.0

        def f2(x):
            return (x,)  # 返回包含输入张量x的元组

        jit_f2 = torch.jit.trace(f2, x)  # 对函数f2进行追踪，传入输入参数x
        assert f2(x) == jit_f2(x)  # 断言原函数f2和追踪后的jit_f2对输入x的输出结果相等，预期是失败的

    def test_trace_out_operator_with_two_output(self):
        example_input = torch.rand(2, 8)  # 创建一个2x8的随机张量example_input
        out_1, out_2 = torch.cummax(example_input, 1)  # 对example_input按行进行累积最大值计算，返回两个张量

        def run_cummax(example_input, out_1, out_2):
            output_1, output_2 = torch.cummax(example_input, 1, out=(out_1, out_2))  # 对example_input进行累积最大值计算，将结果存储在out_1和out_2中
            return output_1, output_2  # 返回计算得到的两个累积最大值张量

        trace_model = torch.jit.trace(run_cummax, (example_input, out_1, out_2))  # 对函数run_cummax进行追踪，传入输入参数

    def test_trace_namedtuple(self):
        Point = namedtuple("point", ["x", "y"])  # 创建一个命名元组Point，包含字段"x"和"y"

        def f(p):
            if type(p) is tuple:
                p = Point(*p)  # 如果输入p是元组，则转换为Point类型的命名元组
            return p.x + p.y  # 返回命名元组p的字段x和y的和

        p = Point(torch.randn(1), torch.randn(1))  # 创建一个Point类型的命名元组p，包含两个随机张量
        traced = torch.jit.trace(f, (p,))  # 对函数f进行追踪，传入输入参数p
        self.assertEqual(f(p), traced(p))  # 断言原函数f和追踪后的traced对输入p的输出结果相等
    # 定义一个测试函数 test_trace_topk，用于测试 topk 方法的追踪功能
    def test_trace_topk(self):
        # 定义一个简单的 PyTorch 模块 M，包含一个 forward 方法
        class M(torch.nn.Module):
            def forward(self, x, y):
                return x.topk(y, dim=1)[1]

        # 创建 M 类的实例 mod
        mod = M()
        # 准备输入数据 inputs
        inputs = (torch.randint(0, 10, (20, 20)), torch.tensor(17))
        # 使用 torch.jit.trace 方法对 mod 进行追踪，生成 traced_func
        traced_func = torch.jit.trace(mod, inputs)

        # 准备用于测试的输入数据 test_inputs
        test_inputs = (torch.randint(0, 9, (9, 9)), torch.tensor(8))
        # 直接调用 mod 的 forward 方法进行计算，得到 eager_out
        eager_out = mod(*test_inputs)
        # 使用追踪得到的 traced_func 对象进行计算，得到 traced_out
        traced_out = traced_func(*test_inputs)
        # 断言追踪后的输出与直接调用的输出相等
        self.assertNotWarn(
            lambda: traced_func(*test_inputs),
            "Shouldn't throw slicing related warn here",
        )
        # 断言 eager_out 和 traced_out 相等
        self.assertEqual(eager_out, traced_out)

        # 重新设置测试输入数据 test_inputs
        test_inputs = (torch.randint(0, 50, (50, 50)), torch.tensor(12))
        # 再次进行计算得到 eager_out 和 traced_out
        eager_out = mod(*test_inputs)
        traced_out = traced_func(*test_inputs)
        # 断言追踪后的输出与直接调用的输出相等
        self.assertNotWarn(
            lambda: traced_func(*test_inputs),
            "Shouldn't throw slicing related warn here",
        )
        # 断言 eager_out 和 traced_out 相等
        self.assertEqual(eager_out, traced_out)

    # 定义一个测试函数 test_typeas_trace_check，用于测试 type_as 方法的追踪
    def test_typeas_trace_check(self):
        # 创建两个张量 a 和 b
        a = torch.tensor([0.4], requires_grad=True)
        b = torch.tensor([0.7], requires_grad=True)

        # 定义一个简单的函数 f，用于测试 type_as 方法
        def f(x, y):
            return x.type_as(y)

        # 使用 torch.jit.trace 方法对函数 f 进行追踪，生成 trace 对象
        trace = torch.jit.trace(f, (a, b))

    # 定义一个测试函数 test_trace_index，用于测试索引操作的追踪
    def test_trace_index(self):
        # 创建一个张量 x 和一个索引张量 y
        x = torch.tensor([0.4], requires_grad=True)
        y = torch.tensor([0], dtype=torch.int64)

        # 定义一个简单的函数 fn，用于测试索引操作
        def fn(x, y):
            return x[y]

        # 使用 torch.jit.trace 方法对函数 fn 进行追踪，生成 fn_traced 函数
        fn_traced = torch.jit.trace(
            fn,
            (
                x,
                y,
            ),
        )

        # 断言原始函数 fn 和追踪后的函数 fn_traced 输出相等
        self.assertEqual(fn(x, y), fn_traced(x, y))

    # 对于索引常量的追踪测试函数
    def test_trace_index_constant(self):
        # 创建一个张量 x
        x = torch.tensor([0.4], requires_grad=True)

        # 定义一个简单的函数 fn，用于测试索引常量的追踪
        def fn(x):
            return x[0]

        # 定义一个运行函数 run，用于运行函数并获取梯度
        def run(f):
            y = f(x)
            grad = torch.autograd.grad(y, x)[0].clone()
            return y, grad

        # 使用 torch.jit.trace 方法对函数 fn 进行追踪，生成 traced_fn 函数
        traced_fn = torch.jit.trace(fn, torch.ones(1))
        # 断言原始函数 fn 和追踪后的函数 traced_fn 输出相等
        self.assertEqual(run(fn), run(traced_fn))

    # 定义一个测试函数 test_index_put，用于测试索引赋值操作的追踪
    def test_index_put(self):
        # 创建一个全零张量 ten 和一个布尔掩码 mask
        ten = torch.zeros(3, 3)
        mask = torch.tensor(
            [[True, True, True], [True, False, False], [True, True, False]]
        )

        # 定义一个简单的函数 test_fn，用于测试索引赋值操作
        def test_fn(ten, mask):
            ten[mask] = torch.ones(6)
            return ten

        # 使用 torch.jit.trace 方法对函数 test_fn 进行追踪，生成 traced_test_fn 函数
        traced_test_fn = torch.jit.trace(test_fn, (ten, mask))

        # 重新设置 ten 为随机张量
        ten = torch.rand(3, 3)
        # 断言原始函数 test_fn 和追踪后的函数 traced_test_fn 输出相等
        self.assertEqual(test_fn(ten, mask), traced_test_fn(ten, mask))
    def test_canonicalize_tensor_iterator(self):
        # 创建一个 4x4 的随机张量 x
        x = torch.randn(4, 4)

        def f(x):
            # 将 x 加 2
            x = x + 2
            # 从 x 减去 4
            x = x - 4
            # 将 x 乘以 6
            x = x * 6
            # 将 x 除以 8
            x = x / 8
            return x

        # 对函数 f 进行追踪
        traced = torch.jit.trace(f, (x,))
        # 调用 f(x)
        f(x)
        # 获取追踪后的图形对象，用于后续断言
        graph = traced.graph_for(x)
        # 断言图中应该有 4 个整数常数用于运算右侧，以及一个 alpha 参数用于 add 和 sub 操作
        self.assertTrue(str(traced.graph_for(x)).count(": int = prim::Constant") == 5)

    @suppress_warnings
    def test_constant(self):
        # 创建一个形状为 (2, 2) 的随机张量 x，要求计算梯度
        x = torch.randn(2, 2, requires_grad=True)

        def f(x):
            # 返回 x 与对角矩阵 [2.0, 2.0] 的乘积
            return x.matmul(torch.diag(torch.tensor([2.0, 2.0])))

        # 检查函数 f 的追踪情况
        self.checkTrace(f, (x,), (torch.ones(2, 2, requires_grad=True),))

    def test_wrapped_number(self):
        # 标量转换为默认张量类型的 'wrapped' 张量。
        # 在某些提升操作中，'wrapped' 张量的行为与普通张量不同：
        # float_tensor * double -> float，但 wrapped_float * double -> double。
        # 如果在 `aten::isclose()` 中未正确处理，这可能会在 check-trace 中引起问题。

        def foobar():
            # 创建一个标量 -10000.0
            x = -10000.0
            # 将 x 乘以一个形状为 (1,) 的浮点数张量
            result = x * torch.ones(1, dtype=torch.float)
            return result

        # 对函数 foobar 进行追踪，同时检查追踪情况
        scripted = torch.jit.trace(foobar, (), check_trace=True)

    def test_inplace_transplant(self):
        # 创建一个形状为 [0.0] 的张量 x，要求计算梯度
        x = torch.tensor([0.0], requires_grad=True)

        def fn(x):
            # 对 x 进行克隆操作，得到 y
            y = x.clone()
            # 对 y 执行原位加法操作 y.add_(2)
            y.add_(2)
            # 对 y 再次执行原位加法操作 y.add_(3)
            y.add_(3)
            return y

        # 获取 fn 函数的追踪图形和输出的依赖消除 (DCE) 之后的图形
        g, _ = torch.jit._get_trace_graph(fn, (x,))
        # 运行 DCE 优化
        self.run_pass("dce", g)
        # 使用 FileCheck 检查克隆和加法操作的次数
        FileCheck().check_count("aten::clone", 1, exactly=True).check_count(
            "aten::add_", 2, exactly=True
        ).check_next("return").run(str(g))
        # 对优化后的图形进行导出和导入测试
        self.assertExportImport(g, (x,))

    def test_inplace_flags(self):
        class InplaceFn(Function):
            @staticmethod
            def forward(ctx, x):
                ctx.mark_dirty(x)
                return x.add_(1)

            @staticmethod
            def backward(ctx, go):
                return go

        class RegularFn(Function):
            @staticmethod
            def forward(ctx, x):
                return x.add(1)

            @staticmethod
            def backward(ctx, go):
                return go

        # 创建一个形状为 [0.0] 的张量 x，要求计算梯度
        x = torch.tensor([0.0], requires_grad=True)

        def fn(x):
            # 使用 RegularFn 对 x 执行加法操作
            y = RegularFn.apply(x)
            # 使用 InplaceFn 对 y 执行加法操作（原位操作）
            y = InplaceFn.apply(y)
            # 再次使用 InplaceFn 对 y 执行加法操作（原位操作）
            y = InplaceFn.apply(y)
            # 使用 RegularFn 对 y 执行加法操作
            y = RegularFn.apply(y)
            return y

        # 获取 fn 函数的追踪图形和强制使用 outplace 方式
        trace_graph, _ = torch.jit._get_trace_graph(fn, (x,), _force_outplace=True)
        # 运行 DCE 优化
        self.run_pass("dce", trace_graph)
        # 获取图形中的操作节点
        ops = list(trace_graph.nodes())
        # 断言每个操作节点是否具有 "inplace" 属性
        for op in ops:
            self.assertTrue(op.hasAttribute("inplace"))
        # 预期的原位标志列表
        inplace_flags = [False, True, True, False]
        # 检查每个操作节点的原位标志是否正确
        for op, is_inplace in zip(ops, inplace_flags):
            self.assertEqual(op.i("inplace"), is_inplace)
    # 测试一个自定义的 inplace 函数的行为
    def test_inplace_check(self):
        # 定义一个继承自 Function 的类 MyInplaceFn
        class MyInplaceFn(Function):
            # 前向传播函数
            @staticmethod
            def forward(self, x):
                # 在输入张量 x 上执行 in-place 加法操作
                x.add_(1)
                # 标记张量 x 为脏数据，表示其内容已被修改
                self.mark_dirty(x)
                return x

            # 反向传播函数
            @staticmethod
            def backward(self, grad):
                # 反向传播仅返回梯度 grad，不做其他处理
                return grad

        # 定义一个函数 fn，调用 MyInplaceFn 的 apply 方法
        def fn(x):
            return MyInplaceFn.apply(x)

        # 生成一个形状为 (5, 5) 的随机张量 x
        x = torch.randn(5, 5)
        # 对 fn 函数进行追踪，设置强制 outplace 模式，不检查追踪
        ge = torch.jit.trace(fn, (x,), _force_outplace=True, check_trace=False)
        # 使用 assertRaisesRegex 断言捕获 RuntimeError，并检查其消息内容
        with self.assertRaisesRegex(RuntimeError, "inplace MyInplaceFn"):
            # 在 ge 上调用 x，触发 RuntimeError
            ge(x)

    # 测试填充操作的强制 outplace 行为
    def test_force_outplace_check_fill(self):
        # 定义函数 f，返回一个形状与输入张量 x 相同的空张量，填充值为 7
        def f(x):
            return torch.empty(x.shape).fill_(7)

        # 生成一个形状为 (10, 15) 的随机张量 x
        x = torch.randn(10, 15)
        # 对 f 函数进行追踪，设置强制 outplace 模式
        ft = torch.jit.trace(f, x, _force_outplace=True)
        # 断言追踪后的结果与原函数 f 在输入 x 上的结果相等
        self.assertEqual(f(x), ft(x))

    # 测试零填充操作的强制 outplace 行为
    def test_force_outplace_check_zero(self):
        # 定义函数 f，返回一个形状与输入张量 x 相同的空张量，并将其置零
        def f(x):
            return torch.empty(x.shape).zero_()

        # 生成一个形状为 (10, 15) 的随机张量 x
        x = torch.randn(10, 15)
        # 对 f 函数进行追踪，设置强制 outplace 模式
        ft = torch.jit.trace(f, x, _force_outplace=True)
        # 断言追踪后的结果与原函数 f 在输入 x 上的结果相等
        self.assertEqual(f(x), ft(x))

    # 执行张量大小调整的追踪测试
    def do_trace_size(self, requires_grad):
        # 定义函数 fn，对输入张量 x 执行 view 操作，调整其形状
        def fn(x):
            return x.view(x.shape[1] * 2, x.size(0), 2)

        # 生成一个形状为 (5, 2, 4) 的随机张量 x，并设置是否需要梯度
        x = torch.randn(5, 2, 4, requires_grad=requires_grad)
        # 生成一个形状为 (4, 8, 4) 的随机张量 y，并设置是否需要梯度
        y = torch.randn(4, 8, 4, requires_grad=requires_grad)

        # 对 fn 函数进行追踪
        traced_fn = torch.jit.trace(fn, x)
        # 断言追踪后的结果与原函数 fn 在输入 y 上的结果相等
        self.assertEqual(traced_fn(y), fn(y))
        # 断言追踪后的结果与原函数 fn 在输入 x 上的结果相等
        self.assertEqual(traced_fn(x), fn(x))

    # 测试不需要梯度的张量大小调整追踪
    def test_trace_size(self):
        self.do_trace_size(False)

    # 测试需要梯度的张量大小调整追踪
    def test_trace_size_with_grad(self):
        self.do_trace_size(True)

    # 测试张量元素数量计算的追踪
    def test_trace_numel(self):
        # 定义函数 fn，返回输入张量 x 的元素数量
        def fn(x):
            return x.numel()

        # 生成一个形状为 (2, 3, 4) 的随机张量 x
        x = torch.randn(2, 3, 4)
        # 生成一个形状为 (4, 5, 6) 的随机张量 y
        y = torch.randn(4, 5, 6)

        # 对 fn 函数进行追踪
        traced_fn = torch.jit.trace(fn, x)
        # 断言追踪后的结果与原函数 fn 在输入 y 上的结果相等
        self.assertEqual(traced_fn(y), fn(y))
        # 断言追踪后的结果与原函数 fn 在输入 x 上的结果相等
        self.assertEqual(traced_fn(x), fn(x))
    # 定义一个方法，用于对输入张量进行 arange 操作，返回索引值张量
    def do_trace_arange(self, requires_grad):
        # 定义内部函数 arange，对输入张量进行 arange 操作，返回索引值张量
        def arange(x):
            return torch.arange(x.shape[0])

        # 定义内部函数 arange_scalar，对固定形状的张量进行 arange 操作，返回固定长度的索引值张量
        def arange_scalar(x):
            return torch.arange(12)

        # 定义内部函数 arange_start_end，对输入张量的起始和结束位置进行 arange 操作，返回索引值张量
        def arange_start_end(x):
            return torch.arange(start=x.shape[0], end=x.shape[0] + 5)

        # 创建两个随机张量 x 和 y，用于后续的 trace 操作
        x = torch.randn(5, 3, 2, requires_grad=requires_grad)
        y = torch.randn(8, 2, 4, requires_grad=requires_grad)

        # 对 arange 函数进行追踪，生成一个追踪版本 traced_arange
        traced_arange = torch.jit.trace(arange, x)
        # 检查追踪后的函数调用结果是否与未追踪版本一致
        self.assertEqual(traced_arange(y), arange(y))
        self.assertEqual(traced_arange(x), arange(x))

        # 对 arange_scalar 函数进行追踪，生成一个追踪版本 traced_arange_scalar
        traced_arange_scalar = torch.jit.trace(arange_scalar, x)
        # 检查追踪后的函数调用结果是否与未追踪版本一致
        self.assertEqual(traced_arange_scalar(y), arange_scalar(y))
        self.assertEqual(traced_arange_scalar(x), arange_scalar(x))

        # 对 arange_start_end 函数进行追踪，生成一个追踪版本 traced_arange_start_end
        traced_arange_start_end = torch.jit.trace(arange_start_end, x)
        # 检查追踪后的函数调用结果是否与未追踪版本一致
        self.assertEqual(traced_arange_start_end(y), arange_start_end(y))
        self.assertEqual(traced_arange_start_end(x), arange_start_end(x))

    # 测试函数，对 do_trace_arange 方法进行测试，不要求梯度计算
    def test_trace_arange(self):
        self.do_trace_arange(False)

    # 测试函数，对 do_trace_arange 方法进行测试，要求梯度计算
    # 测试梯度计算时的不同路径
    def test_trace_arange_with_grad(self):
        self.do_trace_arange(True)

    # 测试函数，验证 torch.full 函数的追踪结果不会将形状信息存储为常量
    def test_trace_full_dynamic_shape(self):
        # 定义一个函数 full_with_shape_like，生成一个形状与输入张量 x 相同的全为常量的张量
        def full_with_shape_like(x):
            return torch.full(x.shape, 2.0)

        # 创建一个随机张量 x，用于后续的 trace 操作
        x = torch.randn(3, 4)
        # 对 full_with_shape_like 函数进行追踪，生成一个追踪版本 ge
        ge = torch.jit.trace(full_with_shape_like, example_inputs=x)
        # 创建一个随机张量 y，检查追踪后的函数调用结果形状与输入张量 y 的形状是否一致
        y = torch.randn(2, 7)
        self.assertEqual(ge(y).shape, y.shape)
        # 检查追踪后的函数调用结果形状与输入张量 x 的形状是否一致
        self.assertEqual(ge(x).shape, x.shape)

    # 测试函数，验证 setitem 操作的追踪结果不会将形状信息存储为常量
    # 修复 https://github.com/pytorch/pytorch/issues/43548
    def test_trace_slice_setitem_dynamic_shape(self):
        # 定义一个函数 slice_setitem，对输入张量 x 进行切片和赋值操作，返回修改后的张量
        def slice_setitem(x, y):
            x[:, 2] = y + 1
            return x

        # 创建一个随机张量 x，用于后续的 trace 操作
        x = torch.randn(3, 4)
        # 对 slice_setitem 函数进行追踪，生成一个追踪版本 traced
        traced = torch.jit.trace(slice_setitem, (x, x[:, 0]))
        # 创建一个随机张量 x_new，检查追踪后的函数调用结果与未追踪版本的结果是否一致
        x_new = torch.randn(10, 5)
        self.assertEqual(traced(x.clone(), x[:, 0]), slice_setitem(x.clone(), x[:, 0]))

    # Suppression: 我们有意对张量进行切片操作，不关心是否将其变为常量
    @suppress_warnings
    # 定义一个方法，用于执行带梯度需求的切片操作
    def do_trace_slice(self, requires_grad):
        # 定义一个切片函数，对输入张量进行不同维度的切片操作
        def slice(x):
            results = []
            # 对输入张量的第一个维度进行递减切片，同时对后两个维度进行固定切片
            for i in range(4):
                results.append(x[: x.size(0) - i, i : x.size(2), i:3])
            return tuple(results)

        # 定义一个选择切片函数，对输入张量进行不同维度的切片和选择操作
        def slice_select(x):
            results = []
            # 对输入张量的第二个维度和第三个维度进行动态切片和选择操作
            for i in range(4):
                results.append(x[:, i:, x.size(2) - 5])
            return tuple(results)

        # 创建随机张量 x 和 y，设置是否需要梯度计算
        x = torch.randn(5, 6, 7, requires_grad=requires_grad)
        y = torch.randn(7, 8, 9, requires_grad=requires_grad)

        # 使用 torch.jit.trace 对 slice 函数进行跟踪编译
        traced_slice = torch.jit.trace(slice, x)
        # 断言跟踪后的结果与直接调用 slice 函数的结果相等
        self.assertEqual(traced_slice(y), slice(y))
        self.assertEqual(traced_slice(x), slice(x))

        # 使用 torch.jit.trace 对 slice_select 函数进行跟踪编译
        traced_slice_select = torch.jit.trace(slice_select, x)
        # 断言跟踪后的结果与直接调用 slice_select 函数的结果相等
        self.assertEqual(traced_slice_select(y), slice_select(y))
        self.assertEqual(traced_slice_select(x), slice_select(x))

    # 测试不需要梯度的切片操作
    def test_trace_slice(self):
        self.do_trace_slice(False)

    # 测试需要梯度的切片操作
    # 测试不同情况下梯度路径的图执行器
    def test_trace_slice_with_grad(self):
        self.do_trace_slice(True)

    # 测试不同的类型转换操作
    def test_trace_casts(self):
        # 定义多个类型转换 lambda 函数
        casts = [
            lambda x: x.byte(),
            lambda x: x.float(),
            lambda x: x.cpu(),
            lambda x: x.to(device="cpu"),
            lambda x: x.to(dtype=torch.int64),
            lambda x: x.to(device="cpu", dtype=torch.float),
            lambda x: x.to(x),
        ]

        # 定义断言函数，验证跟踪的图中是否包含类型转换节点
        def assertContainsCast(trace):
            self.assertEqual(
                sum(n.kind() == "aten::to" for n in trace.graph.nodes()), 1
            )

        # 遍历所有类型转换 lambda 函数并进行跟踪编译
        for cast in casts:
            trace = torch.jit.trace(cast, torch.randn(2, 2))
            # 断言跟踪后的图中包含一次类型转换操作
            assertContainsCast(trace)
            x = torch.randn(2, 2)
            # 断言跟踪后的结果与直接调用类型转换函数的结果相等
            self.assertEqual(trace(x), cast(x))

        # 定义一个复杂类型转换函数，并进行跟踪编译
        def to_tensor(x, y):
            return x.to(y)

        to_tensor_trace = torch.jit.trace(
            to_tensor, (torch.randn(2, 2), torch.randn(1, 8))
        )
        # 断言跟踪后的图中包含一次类型转换操作
        assertContainsCast(to_tensor_trace)
        x, y = torch.randn(2, 2), torch.randn(1, 10)
        # 断言跟踪后的结果与直接调用复杂类型转换函数的结果相等
        self.assertEqual(to_tensor_trace(x, y), to_tensor(x, y))

    # 如果没有 numpy，则跳过测试
    @skipIfCompiledWithoutNumpy
    # 如果有交叉引用，则跳过测试
    @skipIfCrossRef
    def test_trace_warn(self):
        # 定义一个函数 fn，接受参数 x
        def fn(x):
            int(x)  # Warning 1. 警告1：尝试将 x 转换为整数
            y = x * 1  # 计算 x 的乘积，并赋值给 y
            if y:  # Warning 2. 警告2：使用 y 作为条件判断
                pass  # 如果条件为真，则执行空语句块
            q = [x, x * 4]  # 创建包含 x 和 x*4 的列表 q
            z = q[y]  # 使用 y 作为索引，获取列表 q 中的元素，并赋值给 z
            float(z)  # Warning 3. 警告3：尝试将 z 转换为浮点数
            z.tolist()  # Warning 4. 警告4：尝试调用 z 的 tolist() 方法
            z.numpy()  # Warning 5. 警告5：尝试调用 z 的 numpy() 方法
            for _ in torch.ones(4, 4):  # Warning 6. 警告6：迭代遍历 torch.ones(4, 4)
                pass  # 执行空的迭代循环
            return z + 4  # 返回 z 加上 4

        # 使用 warnings 模块捕获警告
        with warnings.catch_warnings(record=True) as warns:
            traced_fn = torch.jit.trace(fn, torch.tensor([1]))
        
        # 验证捕获的警告类型是否为 torch.jit.TracerWarning
        for warn in warns:
            self.assertIs(warn.category, torch.jit.TracerWarning)
        
        # 将警告消息转换为字符串列表
        warns = [str(w.message) for w in warns]
        
        # 检查特定警告消息是否包含在字符串列表中
        self.assertIn("a Python integer", warns[0])
        self.assertIn("a Python boolean", warns[1])
        self.assertIn("a Python float", warns[2])
        self.assertIn("a Python list", warns[3])
        self.assertIn("a NumPy array", warns[4])
        self.assertIn("Iterating over", warns[5])

    def test_trace_tuple(self):
        # 定义一个函数 fn，接受参数 x 和 y
        def fn(x, y):
            return x, (x * y[1], x * y[0])  # 返回 x 和一个元组，元组中包含 x 乘以 y[1] 和 x 乘以 y[0]

        # 初始化 x 和 y
        x, y = torch.randn(2, 2), (torch.ones(2, 2), torch.randn(2, 2))
        
        # 使用 torch.jit.trace 对 fn 进行跟踪
        traced_fn = torch.jit.trace(fn, (x, y))
        
        # 断言跟踪函数的输出与未跟踪函数的输出相等
        self.assertEqual(traced_fn(x, y), fn(x, y))
        
        # 检查是否在跟踪图中有两个 "prim::TupleConstruct" 节点
        FileCheck().check_count("prim::TupleConstruct", 2, exactly=True).check_next(
            "return"
        ).run(str(traced_fn.graph))
        
        # 检查导出和导入跟踪图是否成功
        self.assertExportImport(traced_fn.graph, (x, y))

    def test_trace_random(self):
        # 定义一个函数 f，接受平均值和标准差作为参数
        def f(mean, std):
            return torch.normal(mean, std)  # 返回服从正态分布的随机张量

        # 使用 torch.jit.trace 对 f 进行跟踪
        traced = torch.jit.trace(
            f, (torch.zeros(2, 3), torch.ones(2, 3)), check_trace=False
        )
        
        # 初始化 mean 和 std
        mean, std = torch.zeros(5, 5), torch.ones(5, 5)
        
        # 在随机数生成环境中执行 f 函数
        with torch.random.fork_rng(devices=[]):
            output = f(mean, std)
        
        # 在跟踪后的函数中执行 f 函数
        traced_output = traced(mean, std)
        
        # 断言未跟踪函数和跟踪后函数的输出是否相等
        self.assertEqual(output, traced_output)

    def test_trace_tensor_factory(self):
        # 定义一个函数 run，接受关键字参数 kwargs
        def run(**kwargs):
            # 从 kwargs 中弹出 "inputs_require_grads" 参数，默认为 True
            inputs_require_grads = kwargs.pop("inputs_require_grads", True)

            # 定义一个函数 fn，接受参数 x
            def fn(x):
                return x + torch.ones(2, 3, **kwargs)  # 返回 x 加上一个用 kwargs 创建的张量

            # 复制一份 kwargs，并删除其中的 "out" 键
            input_kwargs = kwargs.copy()
            if "out" in input_kwargs:
                del input_kwargs["out"]
            
            # 创建一个全为 1 的张量 input
            input = torch.ones(2, 3, **input_kwargs)
            
            # 检查跟踪结果，并验证是否记录了 'ones'，而不仅仅是一个常量
            self.checkTrace(fn, (input,), inputs_require_grads=inputs_require_grads)
            
            # 使用 torch.jit.trace 对 fn 进行跟踪
            tfn = torch.jit.trace(fn, input)
            
            # 断言跟踪函数的图中是否包含字符串 'ones'
            self.assertTrue("ones" in str(tfn.graph))

        # 运行 run 函数的不同配置
        run()
        run(dtype=torch.int, inputs_require_grads=False)
        run(out=torch.tensor([]))
        if RUN_CUDA:
            run(device="cuda:0")
        if RUN_CUDA_MULTI_GPU:
            run(device="cuda:1")
    def test_trace_indexed_assignment(self):
        # 定义内部函数 `stuff`，用于对输入张量进行克隆并进行索引赋值操作
        def stuff(x, y):
            # 对输入张量 x 进行克隆操作，确保不改变原始输入
            x = x.clone()
            # 将索引为 0 的元素赋值为 y
            x[0] = y
            # 返回赋值后的张量 x
            return x

        # 创建一个形状为 (3, 4) 的随机张量 example
        example = torch.rand(3, 4)
        # 使用自定义的 checkTrace 方法检查 stuff 函数的追踪结果
        self.checkTrace(stuff, (example, example[0] + 1))

    # TODO: implement
    @unittest.expectedFailure
    def test_output_unflatten(self):
        """检查追踪函数的输出是否保留原始的结构和嵌套"""

        # 定义函数 fn，接受一个输入 x，并返回一个包含多个元组的结构化输出
        def fn(x):
            return (
                x * 2,
                (
                    x**2,
                    x + 4,
                    (x + 2,),
                ),
                x * 4,
            )

        # 使用自定义的 checkTrace 方法检查 fn 函数的追踪结果
        self.checkTrace(fn, (torch.randn(2, 2),))

    def test_input_flatten(self):
        """检查追踪函数的输入是否被展平"""

        # 定义函数 fn，接受两个输入 x 和 t，其中 t 是一个元组
        def fn(x, t):
            # 解包元组 t 得到两个变量 y 和 z，并返回它们与 x 的乘积
            y, z = t
            return x * y * z

        # 定义输入 inputs，包含一个张量和一个元组
        inputs = (torch.randn(1), (torch.randn(1), torch.randn(1)))
        # 使用自定义的 checkTrace 方法检查 fn 函数的追踪结果
        self.checkTrace(fn, inputs)

    def test_input_dict_empty(self):
        # 定义函数 test，接受一个空字典作为输入
        def test(d):
            pass

        # 使用断言确保自定义的 checkTrace 方法在处理空字典输入时引发 RuntimeError
        with self.assertRaises(RuntimeError):
            self.checkTrace(test, {})

    def test_input_dict_remembers_keys(self):
        """检查追踪过程中是否记住了字典输入的键"""

        # 定义一个继承自 torch.nn.Module 的测试模块 TestModule
        class TestModule(torch.nn.Module):
            # 定义模块的前向传播方法，接受一个字典输入 dict_input
            def forward(self, dict_input):
                # 返回字典中键为 'x' 的值
                return dict_input["x"]

        # 创建一个包含键为 'x' 的输入字典 input_1
        input_1 = {"x": torch.tensor(1)}
        # 实例化 TestModule 类为 m
        m = TestModule()
        # 使用 torch.jit.trace 方法对 m 进行追踪，传入 input_1 作为参数
        m_traced = torch.jit.trace(m, (input_1,))
        # 使用断言验证追踪后的模块 m_traced 在输入 input_1 下的输出是否为 torch.tensor(1)
        self.assertEqual(m_traced(input_1), torch.tensor(1))

        # 创建一个包含键为 'x' 的不同值的输入字典 input_same_key_different_value
        input_same_key_different_value = {"x": torch.tensor(2)}
        # 使用断言验证追踪后的模块 m_traced 在输入 input_same_key_different_value 下的输出是否为 torch.tensor(2)
        self.assertEqual(m_traced(input_same_key_different_value), torch.tensor(2))

        # 创建一个包含键为 'y' 的输入字典 input_different_key
        input_different_key = {"y": torch.tensor(3)}
        # 使用断言确保在 input_different_key 下调用 m_traced 时引发 RuntimeError
        with self.assertRaises(RuntimeError):
            m_traced(input_different_key)

        # 创建一个包含键为 'x' 和额外键 'y' 的输入字典 input_additional_key
        input_additional_key = {"x": torch.tensor(4), "y": torch.tensor(3)}
        # 使用断言验证追踪后的模块 m_traced 在输入 input_additional_key 下的输出是否为 torch.tensor(4)
        self.assertEqual(m_traced(input_additional_key), torch.tensor(4))

    def test_input_dict_insertion_order(self):
        """检查字典访问是否不依赖于插入顺序"""

        # 定义一个继承自 torch.nn.Module 的测试模块 TestModule
        class TestModule(torch.nn.Module):
            # 定义模块的前向传播方法，接受一个字典输入 dict_input
            def forward(self, dict_input):
                # 返回字典中键 'x' 和 'y' 对应的值
                return dict_input["x"], dict_input["y"]

        # 创建一个空字典 input_x_then_y，并依次向其添加键 'x' 和 'y' 的值
        input_x_then_y = {}
        input_x_then_y["x"] = torch.tensor(1)
        input_x_then_y["y"] = torch.tensor(2)

        # 实例化 TestModule 类为 m
        m = TestModule()
        # 使用 torch.jit.trace 方法对 m 进行追踪，传入 input_x_then_y 作为参数
        m_traced = torch.jit.trace(m, (input_x_then_y,))
        # 使用断言验证追踪后的模块 m_traced 在输入 input_x_then_y 下的输出是否为 (torch.tensor(1), torch.tensor(2))
        self.assertEqual(m_traced(input_x_then_y), (torch.tensor(1), torch.tensor(2)))

        # 创建一个空字典 input_y_then_x，并依次向其添加键 'y' 和 'x' 的值
        input_y_then_x = {}
        input_y_then_x["y"] = torch.tensor(4)
        input_y_then_x["x"] = torch.tensor(3)

        # 使用断言验证追踪后的模块 m_traced 在输入 input_y_then_x 下的输出是否为 (torch.tensor(3), torch.tensor(4))
        self.assertEqual(m_traced(input_y_then_x), (torch.tensor(3), torch.tensor(4)))
    def test_input_dict_recursive(self):
        # 定义一个测试用的神经网络模块，其 forward 方法从输入字典中取出键"x"对应的值的第二个元素
        class TestModule(torch.nn.Module):
            def forward(self, dict_input):
                return dict_input["x"][1]

        # 定义输入字典 input_1，包含键"x"对应的值是包含 torch.tensor(1) 的字典
        input_1 = {"x": {1: torch.tensor(1)}}
        # 实例化神经网络模块
        m = TestModule()
        # 对模块进行追踪（trace），以获取追踪后的模块 m_traced
        m_traced = torch.jit.trace(m, (input_1,))

        # 定义输入字典 input_2，包含键"x"对应的值是包含 torch.tensor(2) 的字典
        input_2 = {"x": {1: torch.tensor(2)}}
        # 断言追踪后的模块 m_traced 对输入 input_2 的输出是否等于 torch.tensor(2)
        self.assertEqual(m_traced(input_2), torch.tensor(2))

    def test_input_dict_checkTrace_mut(self):
        # 定义一个函数 test，对输入字典 d 的"x"键对应的值进行 tanh_() 操作，并返回结果
        def test(d):
            d["x"].tanh_()
            return d["x"]

        # 定义输入字典 inputs，包含键"x"和"y"，对应的值是随机生成的张量
        inputs = {"x": torch.rand(3, 4), "y": torch.rand(3, 4)}
        # 调用 self.checkTrace 方法，对 test 函数进行追踪，并传入 inputs 字典作为参数
        self.checkTrace(test, (inputs,), inputs_require_grads=False)

    def test_input_dict_unify(self):
        # 定义一个函数 test，从输入字典 d 中取出键"int"和"float"对应的值，并返回
        def test(d):
            return d["int"], d["float"]

        # 定义输入字典 inputs，包含键"int"和"float"，对应的值是指定类型的张量
        inputs = {
            "int": torch.ones((2, 2), dtype=torch.int32),
            "float": torch.ones((2, 2), dtype=torch.float32),
        }
        # 调用 self.checkTrace 方法，对 test 函数进行追踪，并传入 inputs 字典作为参数
        self.checkTrace(test, (inputs,), inputs_require_grads=False)

    def test_input_tuple_of_dicts(self):
        # 定义一个函数 test，从元组 t 中取出第一个元素 d，再从 d 中取出"x"键对应的"y"键的值，并返回
        def test(t):
            d = t[0]
            return d["x"]["y"]

        # 定义输入字典 inputs，包含键"x"，对应的值是包含键"y"对应的随机生成张量的字典
        inputs = {"x": {"y": torch.rand(2, 3)}}
        # 调用 self.checkTrace 方法，对 test 函数进行追踪，并传入元组 (inputs, inputs) 作为参数
        self.checkTrace(test, ((inputs, inputs),), allow_unused=True)

    def test_input_dict_of_dicts(self):
        # 定义一个函数 test，从输入字典 d 中取出键"x"对应的值的"y"键的值，并返回
        def test(d):
            return d["x"]["y"]

        # 定义嵌套输入 nested_input，包含键"y"对应的随机生成张量的字典
        nested_input = {"y": torch.rand(2, 3)}
        # 定义统一的嵌套输入 unified_nested，包含键"y"对应的不同形状的随机生成张量的字典
        unified_nested = {"y": torch.rand(3, 2)}
        # 定义输入字典 inputs，包含键"x"对应的是 nested_input，"force_unify"对应的是 unified_nested
        inputs = {"x": nested_input, "force_unify": unified_nested}
        # 调用 self.checkTrace 方法，对 test 函数进行追踪，并传入 inputs 字典作为参数
        self.checkTrace(test, (inputs,), allow_unused=True)

    def test_input_dict_of_lists(self):
        # 定义一个函数 test，从输入字典 d 中取出键"x"对应的列表的第一个元素，并返回
        def test(d):
            return d["x"][0]

        # 定义输入字典 inputs，包含键"x"对应的是包含随机生成张量的列表
        inputs = {"x": [torch.rand(3, 2)]}
        # 调用 self.checkTrace 方法，对 test 函数进行追踪，并传入 inputs 字典作为参数
        self.checkTrace(test, (inputs,))

    def test_input_list_toplevel_flatten(self):
        # 定义一个函数 test，对输入的两个张量进行相加操作，并返回结果张量
        def test(t1, t2):
            return torch.add(t1, t2)

        # 定义输入列表 inputs，包含两个张量
        inputs = [torch.ones(2, 2), torch.rand(2, 2)]
        # 调用 self.checkTrace 方法，对 test 函数进行追踪，并传入 inputs 列表作为参数
        self.checkTrace(test, inputs)

    def test_input_list_toplevel_flatten_direct(self):
        # 定义一个神经网络模块 Test，其 forward 方法对输入的两个张量进行相加操作，并返回结果张量
        class Test(torch.nn.Module):
            def forward(self, t1, t2):
                return torch.add(t1, t2)

        # 定义输入列表 inputs，包含两个张量
        inputs = [torch.ones(2, 2), torch.rand(2, 2)]
        # 对 Test 模块进行追踪，传入 inputs 列表作为参数
        torch.jit.trace(Test(), inputs)

    def test_input_list_of_tuples(self):
        # 定义一个函数 test，从输入列表 l 中取出第一个元组，并返回该元组的第一个元素
        def test(l):
            return l[0][0]

        # 定义输入列表 inputs，包含一个元组，元组包含一个张量
        inputs = [(torch.ones(2, 2),)]
        # 调用 self.checkTrace 方法，对 test 函数进行追踪，并传入 inputs 列表作为参数
        self.checkTrace(test, (inputs,))

    def test_input_dict_empty_list(self):
        # 定义一个函数 test，未定义具体操作，即什么也不做
        def test(d):
            pass

        # 定义输入字典 inputs，包含一个键为整数1，对应的值是空列表
        inputs = {1: []}
        # 使用 self.assertRaisesRegex 断言，调用 self.checkTrace 方法时会抛出 RuntimeError 异常，
        # 异常信息包含 "List trace"
        with self.assertRaisesRegex(RuntimeError, "List trace"):
            self.checkTrace(test, (inputs,))

    def test_input_list_mixed_type(self):
        # 定义一个函数 test，未定义具体操作，即什么也不做
        def test(d):
            pass

        # 定义输入列表 inputs，包含一个随机生成的张量和一个元组，元组包含两个形状相同的张量
        inputs = [torch.rand(2, 3), (torch.ones(2), torch.ones(2))]
        # 使用 self.assertRaisesRegex 断言，调用 self.checkTrace 方法时会抛出 RuntimeError 异常，
        # 异常信息包含 "consistent"
        with self.assertRaisesRegex(RuntimeError, "consistent"):
            self.checkTrace(test, (inputs,))
    # 定义一个测试方法，用于测试卷积操作
    def test_conv(self):
        # 创建一个形状为 (20, 16, 50, 40) 的全1张量
        x = torch.ones(20, 16, 50, 40)
        # 使用 torch.jit._get_trace_graph 方法获取卷积操作的跟踪图
        g, outputs, inputs = torch.jit._get_trace_graph(
            nn.Conv2d(16, 13, 3, bias=False), x, return_inputs=True
        )
        # 根据跟踪图创建函数
        m = self.createFunctionFromGraph(g)
        # 断言输出结果与跟踪图的输出相等
        self.assertEqual(outputs, m(*inputs))

    # 定义一个测试方法，用于测试最大池化操作
    def test_max_pool(self):
        # 创建一个形状为 (20, 16, 10, 10) 的随机张量
        x = torch.rand(20, 16, 10, 10)

        # 定义一个进行最大池化并加2的函数
        def max_pool2d(x):
            return F.max_pool2d(x, 2) + 2

        # 使用 torch.jit.trace 方法对 max_pool2d 函数进行跟踪
        trace = torch.jit.trace(max_pool2d, (x))
        # 获取跟踪图
        graph = trace.graph_for(x)
        # 检查跟踪图中是否包含最大池化操作
        FileCheck().check("aten::max_pool2d(").run(graph)
        # 断言函数运行结果与跟踪结果一致
        self.assertEqual(max_pool2d(x), trace(x))

    # 定义一个测试方法，用于测试嵌套的原位操作
    def test_nested_inplace(self):
        # 创建一个形状为 (2, 2) 的随机张量
        x = torch.randn(2, 2)
        # 使用 torch.jit._get_trace_graph 方法获取原位阈值函数的跟踪图
        g, outputs, inputs = torch.jit._get_trace_graph(
            lambda x: F.threshold(x, 0, 0, inplace=True), (x,), return_inputs=True
        )
        # 根据跟踪图创建函数
        m = self.createFunctionFromGraph(g)
        # 断言输出结果与跟踪图的输出相等
        self.assertEqual(outputs, m(*inputs))
        # 检查跟踪图中是否包含 "threshold_" 操作
        FileCheck().check("threshold_").run(str(g))
        # 检查跟踪图的导出和导入
        self.assertExportImport(g, (x,))

    # 定义一个测试方法，用于测试重复输入
    def test_repeated_input(self):
        # 定义一个简单的函数 fn，对两个随机张量求和
        def fn(a, b):
            return a + b

        # 使用 self.checkTrace 方法对 fn 函数进行跟踪，输入为两个形状为 (2, 2) 的随机张量
        ge = self.checkTrace(fn, [torch.randn(2, 2)] * 2)
        # 获取跟踪图的输入节点集合
        inputs = set(ge.graph.inputs())
        # 断言输入节点集合长度为 3，因为 checkTrace 中的导出/导入添加了一个 `self` 模块参数
        self.assertTrue(len(inputs) == 3)

    # 定义一个测试方法，用于测试重复输出
    def test_repeated_output(self):
        # 定义一个函数 fn，对两个随机张量进行加法，返回相同的结果张量
        def fn(a, b):
            z = a + b
            return z, z

        # 使用 self.checkTrace 方法对 fn 函数进行跟踪，输入为两个形状为 (2, 2) 的随机张量
        ge = self.checkTrace(fn, [torch.randn(2, 2) for _ in range(2)])
        # 获取输出的元组
        tuple_output = list(ge.graph.outputs())[0]
        # 获取元组输出节点的输入节点列表
        tuple_inputs = list(tuple_output.node().inputs())
        # 断言元组的两个输入节点相等
        self.assertTrue(tuple_inputs[0] == tuple_inputs[1])

    # 定义一个测试方法，用于测试原位复制
    def test_inplace_copy(self):
        # 创建一个形状为 (4, 4) 的随机张量，并要求计算梯度
        x = torch.randn(4, 4, requires_grad=True)

        # 定义一个函数 f，将输入张量 x 复制到一个全零张量中并返回
        def f(x):
            out = torch.zeros(x.size())
            out.copy_(x)
            return out

        # 使用 torch.jit._get_trace_graph 方法获取 f 函数的跟踪图
        g, outputs, inputs = torch.jit._get_trace_graph(f, (x,), return_inputs=True)
        # 运行死代码消除（DCE）优化传递
        self.run_pass("dce", g)
        # 根据跟踪图创建函数
        m = self.createFunctionFromGraph(g)
        # 断言输出结果与跟踪图的输出相等
        self.assertEqual(outputs, m(*inputs))
        # 检查跟踪图的导出和导入
        self.assertExportImport(g, (x,))

    # 定义一个测试方法，用于测试强制输出复制
    def test_inplace_copy_force_outplace(self):
        # 创建一个形状为 (4, 4) 的随机张量，并要求计算梯度
        x = torch.randn(4, 4, requires_grad=True)

        # 定义一个函数 f，将输入张量 x 复制到一个全零张量中并返回
        def f(x):
            out = torch.zeros(x.size())
            out.copy_(x)
            return out

        # 使用 torch.jit._get_trace_graph 方法获取 f 函数的跟踪图，并强制使用 outplace 操作
        g, outputs, inputs = torch.jit._get_trace_graph(
            f, (x,), return_inputs=True, _force_outplace=True
        )
        # 运行死代码消除（DCE）优化传递
        self.run_pass("dce", g)
        # 根据跟踪图创建函数
        m = self.createFunctionFromGraph(g)
        # 断言输出结果与跟踪图的输出相等
        self.assertEqual(outputs, m(*inputs))
        # 检查跟踪图的导出和导入
        self.assertExportImport(g, (x,))
        # 检查跟踪图中是否包含 "expand_as" 操作
        FileCheck().check("expand_as").run(str(g))
    # 定义一个测试方法，用于测试共享参数的情况
    def test_shared_param(self):
        # 定义一个继承自torch.nn.Module的子类MyModule
        class MyModule(torch.nn.Module):
            # 初始化方法
            def __init__(self):
                super().__init__()
                # 创建一个参数 a 和 b，它们共享相同的值，都是一个2x2的随机张量
                self.b = self.a = nn.Parameter(torch.randn(2, 2))

            # 前向传播方法
            def forward(self, x):
                # 返回 x 与参数 a 的乘积加上参数 b
                return x * self.a + self.b

        # 创建 MyModule 的实例 m
        m = MyModule()
        # 获取模型 m 的跟踪图和一个与之相关的空输入张量
        g, _ = torch.jit._get_trace_graph(m, (torch.randn(2, 2),))
        # 运行指定的优化传递（此处为 dead code elimination）
        self.run_pass("dce", g)
        # 断言输入张量的数量为 2
        self.assertEqual(len(list(g.inputs())), 2)
        # 使用 FileCheck 检查跟踪图中是否包含 "mul" 和 "add" 操作
        FileCheck().check("mul").check("add").run(str(g))

    # 运行图执行器测试，根据传入的 optimize 和 use_cuda 参数进行优化和 CUDA 加速设置
    def run_ge_tests(self, optimize, use_cuda):
        # 在性能分析测试中启用性能模式
        with enable_profiling_mode_for_profiling_tests():
            # 在优化执行环境中启用 Torch 的优化执行
            with torch.jit.optimized_execution(optimize):

                # 定义一个生成随机张量的函数，根据 use_cuda 决定是否使用 CUDA
                def rand(*args):
                    t = torch.rand(*args).float()
                    if use_cuda:
                        t = t.cuda()
                    return t

                # 使用 self.checkTrace 方法对给定的 lambda 函数进行跟踪
                self.checkTrace(
                    lambda a, b: a * b + b, [rand(1), rand(1)], [rand(2, 3), rand(2, 3)]
                )
                # 检查一个简单的恒等变换
                self.checkTrace(lambda a, b: (b, a), [rand(1), rand(1)])

                # 定义一个函数 foo，对输入张量 a 进行操作
                def foo(a):
                    t = a * a
                    return t * t, 4 * t

                # 使用 self.checkTrace 方法对 foo 函数进行跟踪
                self.checkTrace(foo, [rand(1)])
                # 测试未使用的输入
                self.checkTrace(
                    lambda a, b: a * a, [rand(1), rand(1)], allow_unused=True
                )
                # 测试不参与梯度计算的输出
                self.checkTrace(foo, [rand(1)], drop=1)
                # 测试自动求导回退
                self.checkTrace(
                    lambda a, b: a * b / (a - 2 * b) + b, [rand(1), rand(1)]
                )

    # 测试未优化的图执行器
    def test_ge_unoptimized(self):
        self.run_ge_tests(False, False)

    # 在 Sandcastle 环境中跳过，因为暂不支持 fuser
    @unittest.skipIf(IS_SANDCASTLE, "NYI: fuser support for Sandcastle")
    @enable_cpu_fuser
    # 测试优化的图执行器
    def test_ge_optimized(self):
        # 在性能分析测试中启用性能模式
        with enable_profiling_mode_for_profiling_tests():
            self.run_ge_tests(True, False)

    # 如果不支持 CUDA，跳过 CUDA 测试
    @unittest.skipIf(not RUN_CUDA, "requires CUDA")
    # 测试 CUDA 加速的图执行器
    def test_ge_cuda(self):
        self.run_ge_tests(True, True)

    # 更多手动测试图执行器的方法，可以用作临时记录板
    def test_ge(self):
        # 定义一个函数 foo，接受两个输入参数 a 和 b
        def foo(a, b):
            return a * b / (a - b) + b

        # 创建 Variable 类的别名 V，并分别对 a 和 b 进行 Variable 封装
        V = Variable
        a, b = V(torch.rand(1)), V(torch.rand(1))
        # 使用 torch.jit.trace 方法对 foo 函数进行跟踪
        ge = torch.jit.trace(foo, (a, b))
        a, b = V(torch.rand(1), requires_grad=True), V(
            torch.rand(1), requires_grad=True
        )
        # 对 ge 的结果 r 进行自动求导，创建计算图
        (r,) = ge(a, b)
        da, db = torch.autograd.grad(r + 3, [a, b], create_graph=True)

        # 计算 l2
        l2 = da * db + db * db
        # 对 l2 进行自动求导，计算 g2result
        g2result = torch.autograd.grad(l2, [da, db])

        # 再次计算 foo(a, b) 的结果 r
        r = foo(a, b)
        # 再次对 r 进行自动求导，创建计算图
        da2, db2 = torch.autograd.grad(r + 3, [a, b], create_graph=True)
        # 断言两次求导的结果相等
        self.assertEqual(da, da2)
        self.assertEqual(db, db2)
        # 计算 l3
        l3 = da2 * db2 + db2 * db2
        # 再次对 l3 进行自动求导，计算 g2result2
        g2result2 = torch.autograd.grad(l3, [da2, db2])
        # 断言两次求导的结果相等
        self.assertEqual(g2result, g2result2)
    def test_trace_annotation(self):
        # 定义一个带有跟踪注释的函数装饰器，应用于 foo 函数
        @_trace(torch.rand(1))
        def foo(a):
            return a + a + a

        # 创建一个 5x5 的随机张量 x
        x = torch.randn(5, 5)
        # 断言 foo 函数对 x 的计算结果与 x + x + x 相等
        self.assertEqual(foo(x), x + x + x)

    @unittest.skipIf(not RUN_CUDA, "calls .cuda()")
    # 默认情况下，在 Ampere 或更新的 GPU 上，nn.Linear 使用 TF32 精度计算浮点张量。
    # 我们希望浮点张量在完整精度下计算，以便使用默认精度。
    @with_tf32_off
    def test_traced_module_cuda(self):
        # 定义一个简单的神经网络模型类 Model
        class Model(nn.Module):
            def __init__(self, num_features, num_layers):
                super().__init__()
                self.num_layers = num_layers
                # 创建包含多个层的子模块序列
                layers = [
                    [nn.Linear(num_features, num_features), nn.Sigmoid()]
                    for _ in range(num_layers)
                ]
                self.submodule = nn.Sequential(*chain(*layers))

            def forward(self, x):
                # 前向传播函数，依次对子模块进行处理
                for i in range(self.num_layers):
                    x = self.submodule[i](x) + x
                return x

        # 创建一个 Model 类的实例 model，包含 5 个特征和 3 层
        model = Model(5, 3)
        # 创建一个 2x5 的随机张量 x
        x = torch.randn(2, 5)
        # 对模型进行追踪，以便后续的 JIT 编译
        traced_model = torch.jit.trace(model, x)

        # 获取模型的 __repr__()，确保我们能够获取模型的字符串表示
        model.__repr__()

        # XXX: 索引子模块序列当前不可用
        # 获取跟踪模型的第一个线性子模块
        linear_submodule = next(iter(traced_model.submodule._modules.values()))

        # 所有非参数属性应引发 AttributeError
        with self.assertRaises(AttributeError):
            linear_submodule.in_features
        # 访问线性子模块的权重
        linear_submodule.weight
        # 设置线性子模块的权重为新的随机参数
        linear_submodule.weight = nn.Parameter(
            torch.randn(linear_submodule.weight.shape)
        )
        # 删除线性子模块的权重应引发 RuntimeError
        with self.assertRaises(RuntimeError):
            del linear_submodule.weight

        # 子模块不能直接调用
        with self.assertRaises(RuntimeError):
            linear_submodule(x)

        # 类型转换：将线性子模块移到 CUDA 设备
        linear_submodule.cuda()
        traced_model.float().cuda()
        cuda_out = traced_model(x.float().cuda())
        traced_model.cpu()
        cpu_out = traced_model(x.float())
        # 断言 CPU 和 CUDA 下的输出结果一致
        self.assertEqual(cpu_out, cuda_out)
        traced_model.to("cuda")
        cuda_out = traced_model(x.float().cuda())
        traced_model.to("cpu")
        cpu_out = traced_model(x.float())
        # 再次断言 CPU 和 CUDA 下的输出结果一致
        self.assertEqual(cpu_out, cuda_out)
        # 将跟踪模型切换到默认数据类型
        traced_model.to(torch.get_default_dtype())

        # state_dict + load_state_dict
        # 复制当前跟踪模型的状态字典
        state = {k: v.clone() for k, v in traced_model.state_dict().items()}
        # 创建一个新的状态字典，所有值都设置为 1
        new_state = {k: v.clone().fill_(1) for k, v in state.items()}
        # 对 x 运行跟踪模型，获取输出
        out = traced_model(x)
        # 加载新的状态字典到跟踪模型
        traced_model.load_state_dict(new_state)
        # 再次对 x 运行跟踪模型，获取输出
        out_ones = traced_model(x)
        # 重新加载之前的状态字典到跟踪模型
        traced_model.load_state_dict(state)
        # 第三次对 x 运行跟踪模型，获取输出
        out_state = traced_model(x)
        # 断言最初状态和当前状态下的输出结果相等
        self.assertEqual(out, out_state)
        # 断言不同状态下的输出结果不相等
        self.assertNotEqual(out, out_ones)

    @unittest.skipIf(not RUN_CUDA, "uses cuda")
    def test_type_same_device(self):
        # 定义一个继承自 torch.nn.Module 的模型类 Model
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 设置模型的数据类型为 torch.float16
                self.dtype = torch.float16

            def forward(self, x=None):
                # 将输入 x 的数据类型转换为 self.dtype 指定的类型
                h = x.type(self.dtype)
                return h

        # 创建 Model 的实例 a
        a = Model()
        # 使用 torch.jit.trace 对模型 a 进行追踪，并指定输入为在 CUDA 设备上的全1张量
        b = torch.jit.trace(
            a, example_inputs=(torch.ones([1], device=torch.device("cuda")),)
        )
        # 使用 FileCheck() 对追踪结果的代码进行检查，确保其中不含有 "device"
        FileCheck().check_not("device").run(b.code)

    def test_export_no_reorder(self):
        # 定义一个简单的数学函数 func，用于测试
        def func(a, b):
            return a * b / (a - 2 * b) + b

        # 准备用于记录输入的张量列表
        recording_inputs = [
            torch.tensor(
                [0.55619788169860839844], dtype=torch.float32, requires_grad=True
            ),
            torch.tensor(
                [0.25947844982147216797], dtype=torch.float32, requires_grad=True
            ),
        ]

        # 使用 torch.jit.trace 对函数 func 进行追踪
        ge1 = torch.jit.trace(func, recording_inputs)
        # 调用自定义方法 self.getExportImportCopy 对 ge1 追踪结果进行复制
        ge2 = self.getExportImportCopy(ge1)

        # 分别使用追踪结果 ge1 和 ge2 对 recording_inputs 进行计算
        outputs_ge1 = ge1(*recording_inputs)
        outputs_ge2 = ge2(*recording_inputs)

        # 计算 ge1 和 ge2 的梯度
        grad_ge1 = torch.autograd.grad(outputs_ge1, recording_inputs)
        grad_ge2 = torch.autograd.grad(outputs_ge2, recording_inputs)

        # 断言 ge1 和 ge2 的输出结果相等
        self.assertTrue(outputs_ge1 == outputs_ge2)
        # 断言 ge1 和 ge2 的梯度相等
        self.assertTrue(grad_ge1 == grad_ge2)

    def test_python_function(self):
        # 定义一个继承自 Function 的自定义函数 MyFn
        class MyFn(Function):
            @staticmethod
            def forward(ctx, x):
                return x + 1

            @staticmethod
            def backward(ctx, grad_output):
                return grad_output

        # 使用 @_trace(torch.zeros(2)) 对函数 fn 进行追踪
        @_trace(torch.zeros(2))
        def fn(x):
            # 调用 MyFn 的 forward 方法，并对输入 x 进行加法操作
            return MyFn.apply(x + 2) + 3

        # 创建两个张量 x 和 y
        x = torch.tensor([1.0, 2.0, 3.0])
        y = torch.randn(2, 2, requires_grad=True)
        # 调用函数 fn，分别传入 x 和 y
        fn(x)
        fn(y)

    def test_python_function_tup(self):
        # 定义一个继承自 Function 的自定义函数 MyFn
        class MyFn(Function):
            @staticmethod
            def forward(ctx, x):
                return x + 1, x - 1

            @staticmethod
            def backward(ctx, grad_output):
                return grad_output, grad_output

        # 使用 @_trace(torch.zeros(2)) 对函数 fn 进行追踪
        @_trace(torch.zeros(2))
        def fn(x):
            # 调用 MyFn 的 forward 方法，并对输入 x 进行加法和减法操作
            a, b = MyFn.apply(x + 2)
            return a + b + 3

        # 创建两个张量 x 和 y
        x = torch.tensor([1.0, 2.0, 3.0])
        y = torch.randn(2, 2, requires_grad=True)
        # 调用函数 fn，分别传入 x 和 y
        fn(x)
        fn(y)

    def test_trace_detach(self):
        # 定义一个简单的函数 foo，使用 torch.matmul 计算结果并进行 detach
        def foo(x, w):
            return torch.matmul(x, w).detach()

        # 使用 torch.jit.trace 对函数 foo 进行追踪
        traced = torch.jit.trace(foo, (torch.rand(3, 4), torch.rand(4, 5)))

        # 使用 FileCheck() 检查追踪结果的图中是否包含 "matmul" 和 "detach"
        FileCheck().check("matmul").check("detach").run(str(traced.graph))
        # 创建两个张量 x 和 w
        x, w = torch.rand(3, 4), torch.rand(4, 5, requires_grad=True)
        # 对追踪后的函数 traced 进行调用，并计算结果
        traced_result = traced(x, w)
        # 断言调用追踪函数和直接调用函数 foo 的结果相等
        self.assertEqual(foo(x, w), traced_result)
        # 断言追踪结果不需要梯度
        self.assertFalse(traced_result.requires_grad)
        # 断言追踪结果的梯度函数为 None
        self.assertIsNone(traced_result.grad_fn)
    # 定义一个测试函数，用于验证在函数内部使用 `detach()` 方法后，对计算图的影响
    def test_trace_detach_redispatch(self):
        # 定义一个简单的函数 foo，计算输入 x 和权重 w 的矩阵乘积 y，并确保 y 需要梯度
        def foo(x, w):
            y = torch.matmul(x, w)
            assert y.requires_grad
            # 对 y 调用 detach() 方法，使其从计算图中分离
            y = y.detach()
            # 确保 detach() 方法的调用使得 y 不再需要梯度
            assert not y.requires_grad
            return y

        # 生成随机的输入 x 和权重 w
        x, w = torch.rand(3, 4), torch.rand(4, 5, requires_grad=True)
        # 使用 torch.jit.trace 对函数 foo 进行追踪，check_trace=False 表示不检查追踪过程中的断言
        torch.jit.trace(foo, (x, w), check_trace=False)

    # 定义一个测试函数，验证在 inplace 操作后对函数追踪的影响
    def test_trace_detach_inplace(self):
        # 定义一个函数 foo，计算输入 x 和权重 w 的矩阵乘积，并对结果使用 detach_() 方法进行 inplace 操作
        def foo(x, w):
            y = torch.matmul(x, w)
            # 使用 detach_() 方法将 y 从计算图中分离
            y.detach_()
            return y

        # 对函数 foo 进行追踪，使用 torch.jit.trace 进行静态图追踪
        traced = torch.jit.trace(foo, (torch.rand(3, 4), torch.rand(4, 5)))

        # 使用 FileCheck 检查追踪后的图形中是否包含 "matmul" 和 "detach("，确保追踪的正确性
        FileCheck().check("matmul").check("detach(").run(str(traced.graph))

        # 生成随机的输入 x 和权重 w
        x, w = torch.rand(3, 4), torch.rand(4, 5, requires_grad=True)
        # 对追踪后的模型进行测试，确保追踪结果与原函数调用的结果一致
        traced_result = traced(x, w)
        self.assertEqual(foo(x, w), traced_result)
        # 确保追踪结果不需要梯度
        self.assertFalse(traced_result.requires_grad)
        # 确保追踪结果的梯度函数为 None
        self.assertIsNone(traced_result.grad_fn)

    # 定义一个测试函数，验证在 inplace 操作后对函数追踪的影响，包含断言以确保操作正确
    def test_trace_detach_inplace_redispatch(self):
        # 定义一个函数 foo，计算输入 x 和权重 w 的矩阵乘积，并在计算完成后使用 detach_() 方法进行 inplace 操作
        def foo(x, w):
            y = torch.matmul(x, w)
            assert y.requires_grad
            # 使用 detach_() 方法将 y 从计算图中分离
            y.detach_()
            # 确保 detach_() 方法的调用使得 y 不再需要梯度
            assert not y.requires_grad
            return y

        # 生成随机的输入 x 和权重 w
        x, w = torch.rand(3, 4), torch.rand(4, 5, requires_grad=True)
        # 使用 torch.jit.trace 对函数 foo 进行追踪，check_trace=False 表示不检查追踪过程中的断言
        torch.jit.trace(foo, (x, w), check_trace=False)

    # 定义一个测试函数，验证在追踪时对张量进行切片操作的正确性
    def test_trace_slice_full_dim(self):
        # 定义一个函数 foo，对输入张量 x 进行切片操作并加上常数 1.0，返回结果
        def foo(x):
            return x[0:5, 0] + 1.0

        # 使用 torch.jit.trace 对函数 foo 进行追踪，传入一个形状为 (5, 4) 的随机张量作为输入
        traced = torch.jit.trace(foo, (torch.rand(5, 4),))
        
        # 生成一个形状为 (6, 3) 的随机张量，用于测试 foo 函数的输出
        test_x = torch.rand(6, 3)
        # 断言追踪结果与原始函数调用的结果相同
        self.assertEqual(foo(test_x), traced(test_x))

    # 定义一个测试函数，验证在追踪过程中处理字典输入的模型的正确性
    def test_trace_dict_input(self):
        # 定义一个继承自 torch.nn.Module 的类 Bar，包含一个模块 Foo 的实例化对象
        class Bar(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.foo = Foo()

            # 定义 Bar 类的 forward 方法，接受两个输入 a 和 b，返回 foo 模块处理后的结果中的 "a" 键对应的值
            def forward(self, a, b):
                return self.foo({"a": a, "b": b})["a"]

        # 定义一个继承自 torch.nn.Module 的类 Foo，实现其 forward 方法，接受一个字典输入 x，返回处理后的结果
        class Foo(torch.nn.Module):
            def forward(self, x):
                return {"a": x["a"] * x["b"]}

        # 生成一个元组 x，包含两个形状为 (3,) 的随机张量，作为模型 Bar 的输入
        x = (torch.rand(3), torch.rand(3))
        # 创建 Bar 类的实例 model
        model = Bar()
        # 调用自定义的函数 self.checkTrace，检查模型 model 的追踪结果是否正确，传入 x 作为输入
        self.checkTrace(model, x)
    def test_trace_dict_output(self):
        # 定义一个输出为字典的模块 TraceDictStrTensor
        class TraceDictStrTensor(torch.nn.Module):
            # 模块的前向传播方法，返回包含键 "a" 和 "b" 的字典
            def forward(self, a, b):
                return {"a": a, "b": b}

        # 定义一个输出为字典的模块 TraceDictTensorTensor
        class TraceDictTensorTensor(torch.nn.Module):
            # 模块的前向传播方法，返回以输入参数 a 和 b 为键的字典
            def forward(self, a, b):
                return {a: b, b: a}

        # 创建输入数据 x，由两个随机张量组成
        x = (torch.rand(3), torch.rand(3))
        
        # 使用 torch.jit.trace 对 TraceDictStrTensor 模块进行追踪，预期抛出 RuntimeError 并包含指定错误信息
        with self.assertRaisesRegex(RuntimeError, r"Encountering a dict at the output"):
            torch.jit.trace(TraceDictStrTensor(), x)

        # 对 TraceDictStrTensor 模块进行追踪，strict=False 表示非严格模式
        traced_dict_str_mod = torch.jit.trace(TraceDictStrTensor(), x, strict=False)
        # 断言追踪后的模块输出与预期的字典相等
        self.assertEqual(traced_dict_str_mod(*x), {"a": x[0], "b": x[1]})

        # 对 TraceDictTensorTensor 模块进行追踪，strict=False 表示非严格模式
        traced_dict_tensor_mod = torch.jit.trace(
            TraceDictTensorTensor(), x, strict=False
        )
        # 断言追踪后的模块输出与预期的字典相等
        self.assertEqual(traced_dict_tensor_mod(*x), {x[0]: x[1], x[1]: x[0]})

    def test_trace_with_tensor_list_output(self):
        # 定义一个返回张量列表的函数 f
        def f():
            return [torch.zeros(1), torch.zeros(5)]

        # 使用 assertWarnsRegex 检测是否有 TracerWarning 警告信息
        with self.assertWarnsRegex(
            torch.jit.TracerWarning, "cause the trace to be incorrect"
        ):
            torch.jit.trace(f, [])
        # 使用非严格模式进行追踪
        traced_non_strict_f = torch.jit.trace(f, [], strict=False)
        # 断言追踪后的结果与原始函数的结果相等
        self.assertEqual(traced_non_strict_f(), f())

    def test_trace_with_number_list_output(self):
        # 定义一个返回数字列表的函数 f
        def f():
            return [1, 5]

        # 使用 assertRaisesRegex 检测是否有 RuntimeError，并包含指定错误信息
        with self.assertRaisesRegex(
            RuntimeError, r"Only tensors.+can be output from traced functions"
        ):
            traced_f = torch.jit.trace(f, [])

    def test_trace_with_nested_tensor_list_output(self):
        # 定义一个返回嵌套张量列表的函数 f
        def f():
            return [[torch.zeros(1)], [torch.zeros(5)]]

        # 使用 assertRaisesRegex 检测是否有 RuntimeError，并包含指定错误信息
        with self.assertRaisesRegex(
            RuntimeError, r"Only tensors.+can be output from traced functions"
        ):
            traced_f = torch.jit.trace(f, [])

    def test_trace_with_nested_strided_tensor_output(self):
        # 定义一个使用 Torch Script 的函数 nt_construct
        @torch.jit.script
        def nt_construct(values, kv_lengths):
            kv_lengths_list: List[int] = kv_lengths.tolist()
            return torch._nested_tensor_from_tensor_list(
                list(values.split(kv_lengths_list, dim=0)), None, None, None, None
            )

        # 定义一个函数 f，接受张量 x 和偏移量 offsets 作为参数
        def f(x, offsets):
            kv_lengths = offsets[1:] - offsets[:-1]
            # 调用 nt_construct 构建嵌套张量，并对其进行余弦运算
            return nt_construct(x, kv_lengths).cos()

        # 创建输入数据 x 和 offsets
        x = torch.rand(5, 4)
        offsets = torch.tensor([0, 2, 5])
        # 计算参考结果 ref
        ref = f(x, offsets)
        # 对函数 f 进行追踪
        f_t = torch.jit.trace(f, (x, offsets))
        # 计算追踪后的结果 res
        res = f_t(x, offsets)
        # 断言追踪后的结果与参考结果相等
        self.assertEqual(ref, res)
        # 创建另一组输入数据 x2 和 offsets2
        x2 = torch.rand((8, 4))
        offsets2 = torch.tensor([0, 2, 4, 8])
        # 断言在不同输入下，追踪后的结果与参考结果相等
        self.assertEqual(f(x2, offsets2), f_t(x2, offsets2))

    def test_trace_variable_instantiation(self):
        # 定义一个包含 Variable 的函数 random_foo
        def random_foo(x):
            return Variable(Variable(x) + 1.0)

        # 对 random_foo 函数进行追踪
        random_foo_traced = torch.jit.trace(random_foo, (torch.rand(3, 4),))

        # 创建输入数据 x
        x = torch.rand(5, 6)
        # 断言追踪后的结果与原始函数的结果相等
        self.assertEqual(random_foo(x), random_foo_traced(x))
    def test_trace_slice_expr_complete_type(self):
        # 定义一个函数 random_foo，将输入加上浮点数 1.0 并返回
        def random_foo(x):
            return x + 1.0
        
        # 使用 torch.jit.trace 对 random_foo 进行跟踪，输入为一个大小为 (3, 4) 的张量
        random_foo_traced = torch.jit.trace(random_foo, (torch.rand(3, 4),))
        
        # 定义一个 Torch 脚本函数 random_bar，接受输入 x，调用 random_foo_traced 处理 x 后返回其切片的前 1 行
        @torch.jit.script
        def random_bar(x):
            return random_foo_traced(x)[0:1]
        
        # 创建一个大小为 (3, 4) 的随机张量 x
        x = torch.rand(3, 4)
        # 断言 random_bar 处理 x 后的结果与 x + 1 的第一行切片相等
        self.assertEqual(random_bar(x), (x + 1)[0:1])

    def test_trace_inline_shape(self):
        # 测试 peephole 优化是否将 size 转换为常量在脚本函数中
        
        # 定义一个 Torch 脚本函数 tensor_size，接受输入 x，返回一个包含 x.size()[0] 的张量
        @torch.jit.script
        def tensor_size(x: torch.Tensor) -> torch.Tensor:
            return torch.tensor([x.size()[0]])
        
        # 断言 tensor_size 处理一个大小为 15 的随机张量后的结果是否为 torch.tensor([15])
        self.assertEqual(
            tensor_size(
                torch.rand(
                    15,
                )
            ),
            torch.tensor([15]),
        )
        
        # 使用 torch.jit.trace 对 tensor_size 进行跟踪，输入为一个大小为 7 的随机张量
        traced_tensor_size = torch.jit.trace(
            tensor_size,
            torch.rand(
                7,
            ),
        )
        
        # 断言 traced_tensor_size 处理一个大小为 15 的随机张量后的结果是否为 torch.tensor([15])
        self.assertEqual(
            traced_tensor_size(
                torch.rand(
                    15,
                )
            ),
            torch.tensor([15]),
        )
        
        # 定义一个 Torch 脚本函数 use_device，接受输入 x，返回一个与 x 设备相同的全零张量
        @torch.jit.script
        def use_device(x):
            return torch.zeros_like(x, device=x.device)
        
        # 定义一个函数 foo，调用 use_device 处理输入 x
        def foo(x):
            return use_device(x)
        
        # 使用 torch.jit.trace 对 foo 进行跟踪，输入为一个大小为 7 的随机张量
        traced_tensor_size = torch.jit.trace(
            foo,
            torch.rand(
                7,
            ),
        )
        
        # 运行 "inline" 优化传递给 traced_tensor_size 的图形，并检查是否存在 "prim::device"
        self.run_pass("inline", traced_tensor_size.graph)
        FileCheck().check("prim::device").run(traced_tensor_size.graph)

    def test_trace_save(self):
        # 定义一个函数 fn，对输入 x 加上 2 并返回
        def fn(x):
            return x + 2
        
        # 定义一个函数 check，用于检查保存和加载的函数的一致性
        def check(func):
            with TemporaryFileName() as fname:
                # 将 func 保存到临时文件中，加载后对输入为 torch.randn(2, 2) 的张量进行断言
                func.save(fname)
                loaded = torch.jit.load(fname)
                input = torch.randn(2, 2)
                self.assertEqual(func(input), loaded(input))
        
        # 使用 torch.jit.trace 对 fn 进行跟踪，输入为一个大小为 (2, 2) 的全 1 张量
        out = torch.jit.trace(fn, (torch.ones(2, 2),))
        # 调用 check 函数检查 out 的一致性
        check(out)

    def test_trace_optioanl_dtype(self):
        # 定义一个类 Test，继承自 torch.nn.Module，其 forward 方法返回一个包含 0 到 4 的张量
        class Test(torch.nn.Module):
            def forward(self):
                return torch.arange(5)
        
        # 使用 torch.jit.trace 对 Test 类进行跟踪，输入为空元组
        traced = torch.jit.trace(Test(), ())
        # 使用 torch.allclose 检查 traced() 和 Test()() 的结果是否相等
        torch.allclose(traced(), Test()())

    def test_trace_save_load_copy(self):
        # 定义一个类 Test，继承自 torch.nn.Module，包含一个卷积层 torch.nn.Conv2d(3, 3, 3)
        class Test(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 3, 3)

            def forward(self, x):
                return self.conv(x)
        
        # 使用 torch.jit.trace 对 Test 类进行跟踪，输入为一个大小为 (1, 3, 224, 224) 的随机张量
        traced = torch.jit.trace(Test(), torch.rand(1, 3, 224, 224))
        # 创建一个字节流缓冲区 buffer
        buffer = io.BytesIO()
        # 将 traced 保存到缓冲区中
        torch.jit.save(traced, buffer)
        buffer.seek(0)
        # 从缓冲区加载模型
        loaded = torch.jit.load(buffer)
        # 使用 copy.copy 和 copy.deepcopy 分别测试 loaded 的复制和深复制
        copy.copy(loaded)
        copy.deepcopy(loaded)
    def test_trace_export_fns(self):
        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.a = 3  # 初始化一个属性 'a' 为 3

            @torch.jit.export
            def __getstate__(self):
                return (3, self.training)  # 返回一个包含值 3 和当前训练状态的元组

            @torch.jit.export
            def __setstate__(self, state):
                self.a = state[0]  # 将属性 'a' 设置为元组中的第一个值
                self.training = state[1]  # 设置当前训练状态为元组中的第二个值

            def forward(self, x):
                return x + self.a  # 返回输入 x 加上属性 'a' 的结果

        f = Foo()  # 创建 Foo 类的实例 f

        traced = torch.jit.trace(f, (torch.rand(3, 4),))  # 对 f 进行追踪转换
        expected_names = ["__getstate__", "__setstate__"]  # 期望的方法名列表

        def check(mod):
            self.assertTrue(
                all(name in mod._c._method_names() for name in expected_names)
            )  # 检查模型 mod 中是否包含所有期望的方法名

        check(traced)  # 检查追踪后的模型 traced

        imported = self.getExportImportCopy(traced)  # 导出并导入追踪后的模型 traced
        check(imported)  # 检查导入后的模型 imported

    def test_trace_export_fns_recursive(self):
        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.a = 3  # 初始化属性 'a' 为 3

            @torch.jit.export
            def __getstate__(self):
                return (3, self.training)  # 返回一个包含值 3 和当前训练状态的元组

            @torch.jit.export
            def __setstate__(self, state):
                self.a = state[0]  # 将属性 'a' 设置为元组中的第一个值
                self.training = state[1]  # 设置当前训练状态为元组中的第二个值

            def forward(self, x):
                return x + self.a  # 返回输入 x 加上属性 'a' 的结果

        class Wrapper(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.foo = Foo()  # 创建 Foo 类的实例 foo

            def forward(self, x):
                return self.foo(x)  # 调用 foo 的 forward 方法并返回结果

        f = Wrapper()  # 创建 Wrapper 类的实例 f

        traced = torch.jit.trace(f, (torch.rand(3, 4),))  # 对 f 进行追踪转换
        expected_names = ["__getstate__", "__setstate__"]  # 期望的方法名列表

        def check(mod):
            self.assertTrue(
                all(name in mod._c._method_names() for name in expected_names)
            )  # 检查模型 mod 中是否包含所有期望的方法名

        check(traced.foo)  # 检查 Wrapper 实例中 foo 属性的方法

        imported = self.getExportImportCopy(traced)  # 导出并导入追踪后的模型 traced
        check(imported.foo)  # 检查导入后的模型 imported 中 foo 属性的方法

        # 注意：Bar 的 forward 方法只能进行追踪转换，而无法进行脚本化
        class Bar(nn.Module):
            @torch.jit.export
            def addTwo(self, x):
                return x + 2  # 返回输入 x 加上 2 的结果

            def forward(self, input):
                return (lambda a: a + 1)(input)  # 返回输入加上 1 的结果

        # 当作为子模块追踪 Bar 时，我们只希望脚本化导出的方法，并保持 forward 方法仍然追踪
        class WrapperExports(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.bar = Bar()  # 创建 Bar 类的实例 bar

            @torch.jit.export
            def addOne(self, x):
                return x + 1  # 返回输入 x 加上 1 的结果

            def forward(self, x):
                return self.bar(x)  # 调用 bar 的 forward 方法并返回结果

        f = WrapperExports()  # 创建 WrapperExports 类的实例 f

        traced = torch.jit.trace(f, (torch.rand(3, 4),))  # 对 f 进行追踪转换
        expected_names = ["addOne"]  # 期望的方法名列表
        check(traced)  # 检查追踪后的模型 traced
    def test_trace_autograd_function(self):
        # 定义一个自定义的 Torch 自动微分函数
        class TestFunc(torch.autograd.Function):
            @staticmethod
            # 前向传播方法，返回输入的负数
            def forward(ctx, input):
                return torch.neg(input)

            @staticmethod
            # 反向传播方法，返回梯度的负数
            def backward(ctx, grad_output):
                return torch.neg(grad_output)

        # 定义一个使用 TestFunc 的追踪模块
        class TracedModule(torch.nn.Module):
            def forward(self, x):
                return torch.relu(TestFunc.apply(x))

        # 定义一个包装器模块，将 TracedModule 包装起来
        class Wrapper(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.tm = TracedModule()

            def forward(self, x):
                return self.tm(x)

        # 使用 torch.jit.trace 方法对 Wrapper 进行追踪
        traced = torch.jit.trace(Wrapper(), (torch.rand(3, 4),))

    def test_trace_multi_output_function(self):
        # 定义一个具有两个输出的自动微分函数
        # 它交换输入，以便检查 TorchScript 中的形状处理是否正确
        class Foo(torch.autograd.Function):
            @staticmethod
            # 前向传播方法，交换输入并返回
            def forward(ctx, x, y):
                return y, x

            @staticmethod
            # 反向传播方法，交换梯度并返回
            def backward(ctx, du, dv):
                return dv, du

        # 定义一个使用 Foo 的模块
        class Bar(torch.nn.Module):
            def forward(self, x, y):
                x = x.relu()
                y = y.relu()
                z = Foo.apply(x, y)
                return z

        # 创建输入张量 x 和 y
        x = torch.rand(3, 2, dtype=torch.double)
        y = torch.rand(1, 2, dtype=torch.double)

        # 生成 JIT IR
        traced = torch.jit.trace(Bar(), (x, y))
        print(traced.graph)

        # 预期的自定义 autograd.Function 输出模式
        schema = (
            "(Double(1, 2, strides=[2, 1], requires_grad=0, device=cpu), "
            "Double(3, 2, strides=[2, 1], requires_grad=0, device=cpu)) "
            "= ^Foo"
        )

        # 检查生成的图形是否符合预期模式
        FileCheck().check(schema).run(traced.graph)

        # 还要检查图形是否可运行并产生正确的结果
        u, v = traced(x, y)
        self.assertEqual(u, y)
        self.assertEqual(v, x)

    def test_interpolate_trace(self):
        # 定义一个简单的 nn.Module，包含一个卷积层和插值操作
        class test(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(1, 32, kernel_size=3, padding=1)

            def forward(self, x):
                y = self.conv(x)
                w = nn.functional.interpolate(
                    y, mode="bilinear", align_corners=False, scale_factor=3
                )
                return w

        # 创建一个测试实例 f
        f = test()
        # 使用 torch.jit.trace 方法对 f 进行追踪
        g = torch.jit.trace(f, (torch.zeros(1, 1, 28, 28),))
        x = torch.zeros(1, 1, 14, 14)
        # 检查追踪的模型 g 是否能够产生与原始模型 f 相同的输出
        self.assertEqual(g(x), f(x))

    @_tmp_donotuse_dont_inline_everything
    def test_trace_optional(self):
        # 使用 torch.jit.script 装饰器将函数 test 转换为 TorchScript 脚本
        @torch.jit.script
        def test(x: Optional[Tensor]):
            # 如果输入参数 x 是 None，则返回一个形状为 (1,) 的零张量
            if x is None:
                return torch.zeros(1)
            else:
                # 否则，返回输入的张量 x
                return x

        # 定义函数 test_none，用来测试输入为 None 的情况
        def test_none():
            return test(None)

        # 定义函数 test_tensor，用来测试输入为一个全零张量的情况
        def test_tensor():
            return test(torch.zeros(2))

        # 对 test_none 进行 TorchScript 的跟踪
        f_none = torch.jit.trace(test_none, ())
        # 断言跟踪后的结果与预期的全零张量相等
        self.assertEqual(f_none(), torch.zeros(1))

        # 对 test_tensor 进行 TorchScript 的跟踪
        f_tensor = torch.jit.trace(test_tensor, ())
        # 断言跟踪后的结果与预期的全零张量相等
        self.assertEqual(f_tensor(), torch.zeros(2))

        # 获取跟踪后的张量图形表示
        graph = f_tensor.graph
        # 使用 FileCheck 验证图形中是否包含特定的操作和调用信息
        FileCheck().check('name="test"').check_next("prim::CallFunction").run(graph)

    def test_trace_nested_datatypes(self):
        # 使用 torch.jit.script 装饰器将函数 foo 转换为 TorchScript 脚本
        @torch.jit.script
        def foo(x):
            # 定义一个函数 foo，返回一个嵌套列表，每个元素为输入 x 加减不同值的结果
            return [[x + 1, x - 1], [x + 2, x - 2]]

        # 定义函数 bar，调用 foo 函数并返回嵌套列表的特定元素
        def bar(x):
            list_stuff = foo(x)
            return list_stuff[0][0], list_stuff[1][1]

        # 对函数 bar 进行 TorchScript 的跟踪，输入为一个随机张量
        traced = torch.jit.trace(bar, torch.rand(3, 4))
        # 定义一个新的随机张量 x
        x = torch.rand(5, 6)
        # 断言函数 bar 在输入 x 上的结果与跟踪后的结果相等
        self.assertEqual(bar(x), traced(x))

    @_tmp_donotuse_dont_inline_everything
    def test_call_traced_fn_from_traced_module(self):
        # 使用装饰器 @_trace(torch.rand(3, 4)) 对函数 traced_fn 进行跟踪
        @_trace(torch.rand(3, 4))
        def traced_fn(x):
            # 定义一个跟踪函数 traced_fn，对输入张量 x 执行 torch.neg 操作
            return torch.neg(x)

        # 定义一个 TorchScript 模块 TracedModule
        class TracedModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.param = torch.nn.Parameter(torch.rand(4, 5))

            def forward(self, x):
                # 在前向传播中调用跟踪函数 traced_fn，并对输入 x 与 self.param 执行矩阵乘法
                return traced_fn(torch.mm(x, self.param))

        # 对 TracedModule 进行 TorchScript 的跟踪，输入为一个随机张量
        tm = torch.jit.trace(TracedModule(), torch.rand(3, 4))

        # 注意：跟踪函数 traced_fn 中的 neg 操作应正确内联
        FileCheck().check("aten::mm").check('name="traced_fn"').check_next(
            "prim::CallFunction"
        ).run(str(tm.graph))

    @_tmp_donotuse_dont_inline_everything
    def test_call_traced_module_from_traced_module(self):
        # 定义一个 TorchScript 模块 TracedModule1
        class TracedModule1(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.param = torch.nn.Parameter(torch.rand(5, 7))

            def forward(self, x):
                # 在前向传播中对输入 x 与 self.param 执行矩阵乘法
                return torch.mm(x, self.param)

        # 定义一个 TorchScript 模块 TracedModule
        class TracedModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.param = torch.nn.Parameter(torch.rand(4, 5))
                # 在初始化中对 TracedModule1 进行 TorchScript 的跟踪，输入为一个随机张量
                self.mod = torch.jit.trace(TracedModule1(), torch.rand(3, 5))

            def forward(self, x):
                # 在前向传播中调用 TracedModule1 的前向方法，并对结果与 1.0 执行加法
                return self.mod(torch.mm(x, self.param)) + 1.0

        # 对 TracedModule 进行 TorchScript 的跟踪，输入为一个随机张量
        tm = torch.jit.trace(TracedModule(), torch.rand(3, 4))

        # 使用 FileCheck 验证图形中是否包含特定的操作和调用信息
        FileCheck().check("aten::mm").check("prim::CallMethod").check_same(
            "forward"
        ).check("aten::add").run(str(tm.graph))
    # 定义一个测试函数，用于测试带有视图的索引赋值操作的跟踪
    def test_index_put_trace_with_view(self):
        # 使用装饰器 _trace 对 test_index_put 进行跟踪
        @_trace(torch.rand(100), torch.tensor([1, 2, 3, 4]), torch.rand(1, 1, 1, 4))
        # 定义 test_index_put 函数，接受目标张量 target、索引 indices 和右手边的张量 rhs
        def test_index_put(target, indices, rhs):
            # 对目标张量 target 的指定索引处赋值为 rhs
            target[indices] = rhs
            # 返回赋值后的目标张量
            return target

        # 使用 FileCheck 对生成的图形式字符串进行检查，确保包含 "aten::view" 和 "index_put_" 字符串
        FileCheck().check("aten::view").check("index_put_").run(
            str(test_index_put.graph)
        )

    # 定义一个测试函数，用于测试不带视图的索引赋值操作的跟踪
    def test_index_put_trace_without_view(self):
        # 使用装饰器 _trace 对 test_index_put 进行跟踪
        @_trace(torch.rand(100), torch.tensor([1, 2, 3, 4]), torch.rand(4))
        # 定义 test_index_put 函数，接受目标张量 target、索引 indices 和右手边的张量 rhs
        def test_index_put(target, indices, rhs):
            # 对目标张量 target 的指定索引处赋值为 rhs
            target[indices] = rhs
            # 返回赋值后的目标张量
            return target

        # 使用 FileCheck 对生成的图形式字符串进行检查，确保不包含 "aten::view" 字符串但包含 "index_put_" 字符串
        FileCheck().check_not("aten::view").check("index_put_").run(
            str(test_index_put.graph)
        )

    # 定义一个测试函数，用于测试数据检查时抑制警告
    @suppress_warnings
    def test_trace_checker_dot_data(self):
        # 断言捕获 torch.jit.TracingCheckError 异常，其中异常消息包含特定文本
        with self.assertRaisesRegex(
            torch.jit.TracingCheckError,
            r"Tensor-valued Constant nodes differed in value " r"across invocations",
        ):
            # 使用装饰器 _trace 对 foo 函数进行跟踪
            @_trace(torch.rand(3, 4), check_inputs=[(torch.rand(3, 4),)])
            # 定义 foo 函数，接受输入张量 x，对其进行操作并返回结果
            def foo(x):
                # 访问输入张量 x 的数据属性
                y = x.data
                # 返回 x 与其数据属性 y 相加的结果
                return x + y

    # 定义一个测试函数，用于测试控制流程的跟踪检查
    @suppress_warnings
    def test_trace_checker_control_flow(self):
        # 定义 foo 函数，接受输入张量 x，在循环中对其进行操作并返回结果
        def foo(x):
            for _ in range(x.size(0)):
                x = torch.neg(x)
            return x

        # 断言捕获 torch.jit.TracingCheckError 异常，其中异常消息包含特定文本
        with self.assertRaisesRegex(
            torch.jit.TracingCheckError, r"Graphs differed across invocations!"
        ):
            # 对 foo 函数进行跟踪，使用指定的输入张量和检查输入
            torch.jit.trace(foo, torch.randn(3, 4), check_inputs=[torch.randn(4, 4)])

    # 定义一个测试函数，用于测试跟踪器的记忆功能
    @suppress_warnings
    def test_trace_checker_memoization(self):
        # 断言捕获 torch.jit.TracingCheckError 异常，其中异常消息包含特定文本
        with self.assertRaisesRegex(
            torch.jit.TracingCheckError, r"Graphs differed across invocations!"
        ):
            # 定义 foo 函数，接受输入张量 x，如果 foo 函数没有缓存属性，则将其取反后赋值给缓存属性，然后返回 x 加上缓存属性的结果
            def foo(x):
                if not hasattr(foo, "cache"):
                    foo.cache = torch.neg(x)
                return x + foo.cache

            # 对 foo 函数进行跟踪，使用指定的输入张量和检查输入
            traced = torch.jit.trace(
                foo, torch.rand(3, 4), check_inputs=[(torch.rand(3, 4),)]
            )

    # 定义一个测试函数，用于测试对左值切片操作的跟踪检查
    def test_trace_checker_slice_lhs(self):
        # 定义 foo 函数，接受输入张量 x，在循环中对其指定行进行赋值操作并返回结果
        def foo(x):
            for i in range(3):
                x[i, :] = torch.zeros(4)
            return x

        # 使用 self.checkTrace 方法检查对 foo 函数的跟踪结果，输入为指定形状的随机张量，不需要计算梯度
        self.checkTrace(foo, (torch.rand(3, 4),), inputs_require_grads=False)

    # 定义一个测试函数，用于测试视图上的原地操作的跟踪警告
    def test_trace_checker_inplace_on_view(self):
        # 定义 foo 函数，接受输入张量 x，在视图上进行原地操作并返回结果
        def foo(x):
            x.view(-1).add_(-x.view(-1))
            return x

        # 断言捕获 torch.jit.TracerWarning 警告，其中警告消息包含特定文本
        with self.assertWarnsRegex(
            torch.jit.TracerWarning,
            "Output nr 1. of the traced function does not match the "
            "corresponding output of the Python function",
        ):
            # 对 foo 函数进行跟踪，使用指定的输入张量和检查输入，并强制使用非原地操作
            torch.jit.trace(
                foo,
                torch.rand(3, 4),
                check_inputs=[torch.rand(5, 6)],
                _force_outplace=True,
            )
    # 定义一个测试方法，测试当左值索引失败时的情况
    def test_lhs_index_fails(self):
        # 定义内部函数foo，接受参数x，对x进行操作并返回
        def foo(x):
            # 尝试对x的二维索引(0, 1)进行赋值操作
            x[0, 1] = 4
            return x

        # 使用断言检查，预期会抛出torch.jit.TracerWarning警告，并包含特定字符串"cause the trace to be incorrect"
        with self.assertWarnsRegex(
            torch.jit.TracerWarning, "cause the trace to be incorrect"
        ):
            # 对foo函数进行追踪，输入参数为torch.rand(3, 4)，并强制使用outplace方式进行操作
            torch.jit.trace(foo, torch.rand(3, 4), _force_outplace=True)

    # 定义一个测试方法，测试左值索引的简单情况
    def test_lhs_index_trivial(self):
        # 定义内部函数foo，接受参数y和x，将x的值赋给y，并返回y
        def foo(y, x):
            # 使用"..."操作符将x的值赋给y
            y[...] = x
            return y

        # 调用自定义方法checkTrace，验证foo函数的追踪情况
        self.checkTrace(
            foo, (torch.rand(3, 4), torch.rand(4)), inputs_require_grads=False
        )

    # 定义一个测试方法，测试原地操作时的警告情况
    def test_inplace_warn(self):
        # 定义内部函数foo，接受参数x，对x进行原地操作，并返回x
        def foo(x):
            # 对x进行视图操作，然后对其结果进行inplace加法和乘法
            x.view(-1).add_(-x.view(-1))
            return x

        # 使用断言检查，预期会抛出torch.jit.TracerWarning警告，并包含特定字符串"cause the trace to be incorrect"
        with self.assertWarnsRegex(
            torch.jit.TracerWarning, "cause the trace to be incorrect"
        ):
            # 对foo函数进行追踪，输入参数为torch.rand(3, 4)，并强制使用outplace方式进行操作
            torch.jit.trace(foo, torch.rand(3, 4), _force_outplace=True)

    # 定义一个使用装饰器的测试方法，测试在训练中使用dropout的追踪检查情况
    @suppress_warnings
    def test_trace_checker_dropout_train(self):
        # 定义内部函数foo，接受参数x，对x进行dropout操作，并返回结果
        def foo(x):
            return torch.dropout(x, p=0.5, train=True)

        # 使用断言检查，预期会抛出torch.jit.TracerWarning警告，并包含特定字符串描述追踪函数输出不匹配的情况
        with self.assertWarnsRegex(
            torch.jit.TracerWarning,
            "Output nr 1. of the traced function does not match the "
            "corresponding output of the Python function",
        ):
            # 对foo函数进行追踪，输入参数为torch.rand(3, 4)，同时检查输入参数为torch.rand(5, 6)
            torch.jit.trace(foo, torch.rand(3, 4), check_inputs=[torch.rand(5, 6)])

        # 使用断言检查，预期会抛出torch.jit.TracerWarning警告，并包含特定字符串描述追踪过程中存在非确定性节点的情况
        with self.assertWarnsRegex(
            torch.jit.TracerWarning, "Trace had nondeterministic nodes"
        ):
            # 对foo函数进行追踪，输入参数为torch.rand(3, 4)，同时检查输入参数为torch.rand(5, 6)
            torch.jit.trace(foo, torch.rand(3, 4), check_inputs=[torch.rand(5, 6)])

    # 定义一个测试方法，测试在不训练中使用dropout的追踪情况
    def test_trace_checker_dropout_notrain(self):
        # 定义变量input，表示一个3x4的随机张量
        input = torch.rand(3, 4)

        # 使用装饰器对内部函数foo进行追踪，以input为输入参数，对input进行dropout操作，并返回结果
        @_trace(input)
        def foo(x):
            return torch.dropout(x, p=0.5, train=False)

        # 使用断言检查，验证foo函数对输入input的处理结果与input本身相等
        self.assertEqual(foo(input), input)

    # 定义一个测试方法，测试张量连续化的追踪情况
    def test_trace_contiguous(self):
        # 定义内部函数foo，接受参数x，对x进行切片、连续化和视图变换操作，并返回结果
        def foo(x):
            return x[:, :, ::2].contiguous().view(12)

        # 创建一个2x3x4的随机张量x
        x = torch.rand(2, 3, 4)
        # 对foo函数进行追踪，输入参数为x，并返回追踪后的结果
        traced = torch.jit.trace(foo, (x,))
        # 调用追踪后的函数traced，对输入x进行处理，并得到输出y
        y = traced(x)
        # 使用断言检查，验证处理后的y与原始输入x在存储上的数据指针不相同
        self.assertNotEqual(x.storage().data_ptr(), y.storage().data_ptr())

    # 定义一个测试方法，测试在连续化过程中的追踪情况
    # 这里还添加了额外的注释，说明了函数的测试目的和逻辑
    def test_trace_contiguous_short_circuit(self):
        """
        This tests the logic in THPVariable_contiguous. There is short-circuiting
        code that prevents us from even getting to VariableType::contiguous, since
        it is an optimization that prevents us from acquiring the GIL for touching
        the device. We needed to add the tracing logic directly into the
        THPVariable_contiguous function only for the path where we are skipping
        dispatch into contiguous. We should see an aten::contiguous in this trace!
        """
        # 定义内部函数foo，接受参数x，对x进行连续化操作，并返回结果
        def foo(x):
            return x.contiguous()

        # 创建一个2x3x4的随机张量x
        x = torch.rand(2, 3, 4)
        # 对foo函数进行追踪，输入参数为x，并返回追踪后的结果
        traced = torch.jit.trace(foo, (x,))
        # 使用FileCheck工具检查追踪图中是否包含"aten::contiguous"操作
        FileCheck().check("aten::contiguous").run(str(traced.graph))

    # 定义一个测试方法，测试按位取反操作的追踪情况
    def test_trace_inverse(self):
        # 定义内部函数foo，接受参数x，对x进行按位取反操作，并返回结果
        def foo(x):
            return ~x

        # 对foo函数进行追踪，输入参数为torch.zeros(3, 4, dtype=torch.uint8)，并得到追踪后的函数foo_traced
        foo_traced = torch.jit.trace(foo, torch.zeros(3, 4, dtype=torch.uint8))
        # 创建一个3长度的torch.uint8类型的零张量eg
        eg = torch.zeros(3, dtype=torch.uint8)
        # 使用断言检查，验证追踪后的foo_traced对输入eg的处理结果与foo函数本身处理结果相等
        self.assertEqual(foo_traced(eg), foo(eg))
    @skipIfCrossRef
    # 使用装饰器，跳过交叉引用的测试
    def test_trace_records_names(self):
        # 定义函数 foo，接受两个参数 bar 和 baz
        def foo(bar, baz):
            # 将 bar 加 3 赋给 baz
            baz = bar + 3
            # 对 baz 取负值，赋给 quick_brown_fox
            quick_brown_fox = torch.neg(baz)
            # 循环 20 次
            for _ in range(20):
                # yeet 被赋值为 quick_brown_fox 减 3.14
                yeet = quick_brown_fox - 3.14
            # 返回 yeet
            return yeet

        # 使用 torch.jit.trace 对 foo 进行追踪，传入两个 3x3 的随机张量作为参数
        traced = torch.jit.trace(foo, (torch.rand(3, 3), torch.rand(3, 3)))
        # 将追踪后的计算图转换为字符串形式
        graph_str = str(traced.graph)
        # 断言计算图中包含字符串 "bar"
        assert "bar" in graph_str
        # 断言计算图中包含字符串 "baz"
        assert "baz" in graph_str
        # 断言计算图中包含字符串 "quick_brown_fox"
        assert "quick_brown_fox" in graph_str

    @skipIfTorchDynamo("Not a suitable test for TorchDynamo")
    # 使用装饰器，跳过 TorchDynamo 测试，因为这不是 TorchDynamo 的适当测试
    def test_tracing_hooks(self):
        # 定义一个简单的神经网络模型类
        class Net(nn.Module):
            def forward(self, x):
                return x + x

        # 定义用于测试钩子函数的函数
        def test_hook(is_post_hook, hook, fc):
            # 创建一个网络实例
            n = Net()
            # 根据条件选择注册正向传播钩子或者正向传播前钩子
            if is_post_hook:
                n.register_forward_hook(hook)
            else:
                n.register_forward_pre_hook(hook)

            # 使用 torch.jit.trace 对模型进行跟踪
            module = torch.jit.trace(n, (torch.tensor(1.0),))

            # 使用非跟踪的输入进行前向计算
            eager_input = torch.tensor(1.0)
            eager_out = n(eager_input)

            # 运行 FileCheck 实例检查跟踪后的图形
            fc.run(module.forward.graph)

            # 使用跟踪后的模型进行输入输出的计算
            input = torch.tensor(1.0)
            output = module(input)

            # 断言跟踪前后的输入和输出相同
            self.assertEqual(input, eager_input)
            self.assertEqual(output, eager_out)

        # 定义一个没有返回值的正向传播钩子函数
        def hook_no_return(mod, input, output):
            input[0].add_(1)
            output.sub_(1)

        # 创建 FileCheck 实例进行多个检查
        fc = FileCheck().check("add(").check("add_(").check("sub_(")
        # 测试正向传播后钩子函数没有返回值的情况
        test_hook(True, hook_no_return, fc)

        # 定义一个有返回值的正向传播钩子函数
        def hook_return(mod, input, output):
            input[0].add_(1)
            return output - 3

        # 创建 FileCheck 实例进行多个检查
        fc = FileCheck().check("add(").check("add_(").check("sub(")
        # 测试正向传播后钩子函数有返回值的情况
        test_hook(True, hook_return, fc)

        # 定义一个捕获外部变量的正向传播钩子函数
        b = torch.tensor(3.0)
        def captured_hook(mod, input, output):
            return output - b

        # 创建 FileCheck 实例进行多个检查
        fc = FileCheck().check("add(").check("sub(")
        # 测试捕获外部变量的正向传播后钩子函数
        test_hook(True, captured_hook, fc)

        # 定义一个没有返回值的正向传播前钩子函数
        def pre_hook_no_ret(mod, input):
            input[0].add_(3)

        # 创建 FileCheck 实例进行多个检查
        fc = FileCheck().check("add_(").check("add(")
        # 测试正向传播前钩子函数没有返回值的情况
        test_hook(False, pre_hook_no_ret, fc)

        # 定义一个有返回值的正向传播前钩子函数
        def pre_hook_ret(mod, input):
            return input[0] - 4

        # 创建 FileCheck 实例进行多个检查
        fc = FileCheck().check("sub(").check("add(")
        # 测试正向传播前钩子函数有返回值的情况
        test_hook(False, pre_hook_ret, fc)

    def test_tracing_backward_hook_error(self):
        # 定义一个简单的神经网络模型类
        class Net(nn.Module):
            def forward(self, x):
                return x + x

        # 创建一个网络实例
        n = Net()

        # 定义一个空的反向传播钩子函数
        def backward_hook(module, grad_input, grad_output):
            pass

        # 注册空的反向传播钩子函数到网络实例中
        n.register_backward_hook(backward_hook)

        # 使用 torch.jit.trace 尝试跟踪模型
        # 预期会抛出异常，因为已经注册了反向传播钩子
        with self.assertRaisesRegex(Exception, "backward hooks assigned"):
            torch.jit.trace(n, (torch.tensor(1.0),))
    def test_tracing_multiple_methods(self):
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(1, 1, 3)  # 定义一个 2D 卷积层

            def forward(self, x):
                return self.conv(x)  # 执行前向传播，返回卷积层的输出

            def weighted_kernel_sum(self, weight):
                return weight * self.conv.weight  # 返回权重与卷积核权重的乘积

        example_weight = torch.rand(1, 1, 3, 3)  # 创建一个示例权重张量
        example_forward_input = torch.rand(1, 1, 3, 3)  # 创建一个示例前向输入张量
        inputs = {
            "forward": example_forward_input,
            "weighted_kernel_sum": example_weight,
        }
        n = Net()  # 创建一个 Net 实例
        module = torch.jit.trace_module(n, inputs)  # 对 Net 实例进行跟踪

        check_inputs = []
        for i in range(2):
            check_weight = torch.rand(1, 1, 3, 3)  # 创建用于检查的权重张量
            check_forward_input = torch.rand(1, 1, 3, 3)  # 创建用于检查的前向输入张量
            check_inputs.append(
                {"forward": check_forward_input, "weighted_kernel_sum": check_weight}
            )
        module = torch.jit.trace_module(
            n, inputs, check_trace=True, check_inputs=check_inputs
        )  # 对 Net 实例进行带检查的跟踪

        self.assertTrue(module._c._has_method("forward"))  # 断言模块包含 forward 方法
        self.assertTrue(module._c._has_method("weighted_kernel_sum"))  # 断言模块包含 weighted_kernel_sum 方法

        module = torch.jit.trace(n.forward, example_forward_input)  # 对 Net 的 forward 方法进行跟踪
        module = torch.jit.trace(
            n.forward,
            example_forward_input,
            check_trace=True,
            check_inputs=[example_forward_input],
        )  # 对 Net 的 forward 方法进行带检查的跟踪

        with self.assertRaisesRegex(
            AttributeError,
            "trace doesn't support compiling individual module's functions",
        ):
            module = torch.jit.trace(n.weighted_kernel_sum, inputs)  # 尝试对 Net 的 weighted_kernel_sum 方法进行跟踪，期望引发 AttributeError

    def test_tensor_with_grad_as_constant(self):
        param = torch.randn(3).requires_grad_()  # 创建一个带梯度的参数张量
        x = torch.randn(3)  # 创建一个张量 x

        def f(x):
            return x + param  # 定义一个函数 f，返回 x 加上 param

        with self.assertRaisesRegex(
            RuntimeError, "Cannot insert a Tensor that requires grad as a constant"
        ):
            torch.jit.trace(f, x)  # 尝试对函数 f 进行跟踪，期望引发 RuntimeError

    def test_non_tensor_tracing(self):
        def f(x):
            return x + param  # noqa: F821

        with self.assertRaisesRegex(
            RuntimeError, r"Type 'Tuple\[int\]' cannot be traced"
        ):
            torch.jit.trace(f, (1,))  # 尝试对函数 f 进行跟踪，期望引发 RuntimeError

    def test_trace_skip_none_submodule(self):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.submod = torch.nn.Linear(3, 4)  # 定义一个线性层
                self.submod = None  # 将子模块设置为 None

            def forward(self, inputs):
                return inputs

        m = TestModule()  # 创建 TestModule 实例
        tm = torch.jit.trace(m, torch.tensor(1.0))  # 对 TestModule 实例进行跟踪
        self.assertFalse(hasattr(tm, "submod"))  # 断言跟踪后的模块没有 submod 属性
    def test_trace_with_conditional_property(self):
        # 定义一个继承自 nn.Module 的神经网络类 Net
        class Net(nn.Module):
            def __init__(self, attr=None):
                super().__init__()
                # 如果有传入 attr 参数，则设置实例的 _attr 属性
                if attr is not None:
                    self._attr = attr
                # 设置实例的属性名为 "_attr"
                self.attr_name = "_attr"

            @property
            # 定义属性 attr，返回实例的 self.attr_name 属性值
            def attr(self):
                return getattr(self, self.attr_name)

            # 定义前向传播方法，简单地返回输入 x
            def forward(self, x):
                return x

        # 创建一个值为 1 的张量 x
        x = torch.ones(1)
        # 对 Net 类进行 Torch 的 JIT 跟踪
        torch.jit.trace(Net(), x)

    def test_trace_func_argument_names_captured(self):
        # 定义一个接受两个 torch.Tensor 类型参数并返回 torch.Tensor 类型的函数 fn
        def fn(first_arg: torch.Tensor, second_arg: torch.Tensor) -> torch.Tensor:
            return first_arg + second_arg

        # 对 fn 函数进行 Torch 的 JIT 跟踪，并检查生成的图形是否包含参数名 "first_arg" 和 "second_arg"
        traced_fn = torch.jit.trace(fn, (torch.ones(1), torch.ones(1)))
        FileCheck().check("first_arg").check_next("second_arg").run(
            str(traced_fn.graph)
        )

    def test_trace_partial_func_argument_names_captured(self):
        # 定义一个接受一个 torch.Tensor 类型参数和一个默认值为 1 的参数的函数 fn
        def fn(first_arg: torch.Tensor, second_arg=1) -> torch.Tensor:
            return first_arg + second_arg

        # 对 fn 函数进行 Torch 的 JIT 跟踪，并检查生成的图形是否包含参数名 "first_arg"，但不包含 "second_arg"
        traced_fn = torch.jit.trace(fn, (torch.ones(1),))
        FileCheck().check("first_arg").check_not("second_arg").run(str(traced_fn.graph))

    def test_trace_module_argument_names_captured(self):
        # 定义一个继承自 nn.Module 的测试模块类 TestModule
        class TestModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(1, 1, 3)

            # 定义前向传播方法，接受两个 torch.Tensor 类型参数，并使用 self.conv 对第一个参数进行卷积后与第二个参数相加
            def forward(self, first_arg: torch.Tensor, second_arg: torch.Tensor):
                return self.conv(first_arg) + second_arg

        # 创建 TestModule 类的实例 m
        m = TestModule()
        # 创建一个示例输入元组 example_input
        example_input = (torch.ones(1, 1, 3, 3), torch.ones(1, 1, 3, 3))

        # 显式地对模块的 forward 方法进行 Torch 的 JIT 跟踪，并检查生成的图形是否包含参数名 "first_arg" 和 "second_arg"
        traced_module_forward = torch.jit.trace(m.forward, example_input)
        FileCheck().check("first_arg").check_next("second_arg").run(
            str(traced_module_forward.graph)
        )

        # 对整个模块 m 进行 Torch 的 JIT 跟踪，并检查生成的图形是否包含参数名 "first_arg" 和 "second_arg"
        traced_module = torch.jit.trace(m, example_input)
        FileCheck().check("first_arg").check_next("second_arg").run(
            str(traced_module.graph)
        )

    def test_trace_checking_with_deprecated_name(self):
        # 定义一个继承自 torch.nn.Module 的 MyClass 类
        class MyClass(torch.nn.Module):
            def __init__(self):
                super(MyClass, self).__init__()

            # 定义前向传播方法，接受 x、y 和其他已弃用参数，并在存在已弃用参数时引发异常
            def forward(self, x, y, **deprecated_arguments):
                if len(deprecated_arguments) > 0:
                    raise RuntimeError(
                        f"Got unexpected arguments: {deprecated_arguments}"
                    )
                return x + y

        # 创建 MyClass 类的实例 model
        model = MyClass()
        # 对 model 进行 Torch 的 JIT 跟踪，并传入示例输入元组作为关键字参数输入，允许宽松模式
        m2 = torch.jit.trace(model, (torch.ones(1), torch.ones(1)))
        m3 = torch.jit.trace(
            model,
            example_kwarg_inputs={"x": torch.ones(1), "y": torch.ones(1)},
            strict=False,
        )
    def test_trace_no_duplicated_lifted_input_output(self):
        # 定义一个名为Normalize的神经网络模块类
        class Normalize(nn.Module):
            def __init__(self):
                super().__init__()
                # 使用GroupNorm规范化输入数据，32个组，每组32个通道
                self.norm = nn.GroupNorm(num_groups=32, num_channels=32)

            def forward(self, x, y):
                # 如果y为空，则将y设为x
                if y is None:
                    y = x
                else:
                    # 对y进行规范化处理
                    y = self.norm(y)
                # 对y乘以2
                y = y * 2
                return y

        # 定义一个名为G的神经网络模块类
        class G(nn.Module):
            def __init__(self):
                super().__init__()
                # 创建一个Normalize的实例
                self.norm = Normalize()

            def forward(self, x):
                # 使用Normalize模块处理输入x，y为空
                A = self.norm(x, None)
                # 对A使用ReLU激活函数
                B = F.relu(A)
                return A, B

        # 定义一个名为Net的神经网络模块类
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                # 创建一个G的实例
                self.g = G()
                # 创建一个Normalize的实例
                self.norm_1 = Normalize()

            def forward(self, x):
                # 使用G模块处理输入x，得到hs
                hs = self.g(x)
                # 将hs中的两个返回值分别赋值给A和B
                A, B = hs
                # 使用Normalize模块处理输入B和A
                h = self.norm_1(B, A)
                return h

        # 创建一个Net的实例net，并设置为评估模式
        net = Net()
        net = net.eval()
        # 创建一个形状为(1, 32, 16, 16)的随机张量x
        x = torch.randn(1, 32, 16, 16)
        # 对net进行追踪，并将结果赋给traced
        traced = torch.jit.trace(net, x)
        # 使用FileCheck检查traced图中是否存在"prim::TupleUnpack"，并确保不存在
        FileCheck().check_not("prim::TupleUnpack").run(str(traced.graph))
# 标记一个测试类，如果 TorchDynamo 不适合则跳过测试
@skipIfTorchDynamo("Not a suitable test for TorchDynamo")
class TestMixTracingScripting(JitTestCase):
    
    # 测试函数，用于跟踪脚本化函数 func1
    def test_trace_script(self):
        # 定义一个接受元组（包含两个张量）并返回它们相加的脚本化函数
        @torch.jit.script
        def func1(x: Tuple[Tensor, Tensor]) -> Tensor:
            return x[0] + x[1]

        # 定义一个接受列表（包含两个张量）并返回它们相加的脚本化函数
        @torch.jit.script
        def func2(x: List[Tensor]) -> Tensor:
            return x[0] + x[1]

        # 生成两个随机张量 a 和 b
        a = torch.randn(5)
        b = torch.randn(5)

        # 对 func1 和 func2 进行跟踪检查
        self.checkTrace(func1, ((a, b),))
        self.checkTrace(func2, ((a, b),))

        # 定义一个接受张量 x 和可选参数 method 和 align_corners 的脚本化函数
        @torch.jit.script
        def func3(
            x: Tensor, method: str = "bilinear", align_corners: bool = True
        ) -> Tensor:
            hw = x.shape[2:4]
            return F.interpolate(x, hw, mode=method, align_corners=align_corners)

        # 生成一个随机输入张量 inp
        inp = torch.rand(1, 3, 6, 6)
        # 对 func3 进行跟踪检查
        self.checkTrace(func3, (inp,))

        # 定义一个接受张量 x 和可选参数 a 的列表的脚本化函数
        @torch.jit.script
        def func4(x: Tensor, a: List[Optional[str]]) -> Tensor:
            if len(a) == 2:
                return x + 2
            else:
                return x

    # 测试混合脚本化函数并输出字典的跟踪
    def test_trace_mixed_by_script_with_dict_output(self):
        # 定义一个接受输入张量 input 并返回包含 "foo" 键的字典的脚本化函数
        @torch.jit.script
        def return_dict(input: torch.Tensor) -> Dict[str, torch.Tensor]:
            return {"foo": input + 1}

        # 定义一个继承自 torch.nn.Module 的类 TraceModule
        class TraceModule(torch.nn.Module):
            def forward(self, input):
                dict = return_dict(input)
                return dict["foo"] + dict["foo"]

        # 创建一个全为 1 的输入张量 x
        x = torch.ones(1)
        # 对 TraceModule 进行跟踪，验证其输出与预期相等
        tm = torch.jit.trace(TraceModule(), x)
        self.assertEqual(tm(x), x + 1 + x + 1)

    # 测试脚本化函数的跟踪
    def test_trace_of_script(self):
        # 定义一个接受 a 和 c 两个参数的脚本化函数 foo
        @torch.jit.script
        def foo(a, c):
            b = 0.0
            if bool(a == 0.0):
                b = 1.0
            return b + c

        # 创建一个全为 1 的张量 a
        a = torch.ones(1, dtype=torch.float)

        # 使用 @_trace 装饰器将函数 use 跟踪为 torch.zeros(1, dtype=torch.float)
        @_trace(torch.zeros(1, dtype=torch.float))
        def use(b):
            return foo(b - 1.0, a) + 1.0

        # 验证在函数中是否正确传播了形状信息
        self.assertTrue("Dynamic" not in str(use.graph))

        # 验证 use 函数对不同输入的输出是否符合预期
        self.assertEqual(3, use(torch.ones(1, dtype=torch.float)))
        self.assertEqual(2, use(torch.zeros(1, dtype=torch.float)))

    # 测试带有 size 的跟踪
    def test_trace_with_size(self):
        # 使用 @_trace 装饰器将函数 foo 跟踪为 torch.zeros(1, 1)
        @_trace(torch.zeros(1, 1))
        def foo(x):
            return x + 1

        # 定义一个接受 x 参数的脚本化函数 bar
        @torch.jit.script
        def bar(x):
            y = int(foo(x))
            if 1 == 1:
                y = 7
            return y + 1

        # 验证 bar 函数对给定输入的输出是否符合预期
        self.assertEqual(8, bar(torch.ones(1, 1)))

    # 测试切片的跟踪
    def test_tracing_slicing(self):
        # 使用 @_trace 装饰器将函数 foo_trace 跟踪为 torch.zeros(10)
        @_trace(torch.zeros(10))
        def foo_trace(x):
            return x[-5:-3]

        # 定义一个接受 x 参数的脚本化函数 foo_script
        @torch.jit.script
        def foo_script(x):
            return x[-5:-3]

        # 定义一个简单的函数 foo，返回 x 的切片 x[-5:-3]
        def foo(x):
            return x[-5:-3]

        # 创建两个张量 a 和 b，分别有不同的长度
        a = torch.arange(0, 8)
        b = torch.arange(0, 20)

        # 验证三种不同方式生成的切片结果是否相等或不等
        self.assertEqual(foo_trace(a), foo_script(a))
        self.assertEqual(foo_trace(a), foo(a))
        self.assertNotEqual(foo_trace(a), foo_trace(b))
    def test_tracing_indexing(self):
        # 定义装饰器函数，用于跟踪函数调用及其参数
        @_trace(torch.zeros(10))
        def foo_trace(x):
            # 返回张量 x 的倒数第二个元素
            return x[-2]

        # 使用 TorchScript 进行脚本化编译，返回张量 x 的倒数第二个元素
        @torch.jit.script
        def foo_script(x):
            return x[-2]

        # 普通的 Python 函数，返回张量 x 的倒数第二个元素
        def foo(x):
            return x[-2]

        # 创建两个张量 a 和 b
        a = torch.arange(0, 8)
        b = torch.arange(0, 20)
        # 断言 TorchScript 编译的结果与装饰器跟踪的结果相等
        self.assertEqual(foo_script(a), foo_trace(a))
        # 断言装饰器跟踪的结果与普通函数的结果相等
        self.assertEqual(foo_trace(a), foo(a))
        # 断言不同张量的装饰器跟踪结果不相等
        self.assertNotEqual(foo_trace(a), foo_trace(b))

    def test_trace_hierarchy(self):
        # 测试在 TorchScript 模块中保留子模块的模块层次结构
        # during tracing

        # 定义一个 TorchScript 脚本模块 AnotherScriptMod
        class AnotherScriptMod(torch.jit.ScriptModule):
            def __init__(self):
                super().__init__()
                # 添加一个参数 param，是一个随机初始化的张量
                self.param = torch.nn.Parameter(torch.rand(1, 2, 3))

            @torch.jit.script_method
            def bar(self):
                # 返回一个形状为 (4, 5) 的全零张量
                return torch.zeros(4, 5)

        # 定义一个 TorchScript 脚本模块 SomeScriptMod
        class SomeScriptMod(torch.jit.ScriptModule):
            def __init__(self):
                super().__init__()
                # 添加一个子模块 AnotherScriptMod
                self.asm = AnotherScriptMod()

            @torch.jit.script_method
            def foo(self):
                # 返回一个形状为 (3, 4) 的全零张量
                return torch.zeros(3, 4)

            @torch.jit.script_method
            def bar(self):
                # 返回一个形状为 (4, 3) 的全零张量
                return torch.zeros(4, 3)

        # 定义一个继承自 torch.nn.Module 的普通 Python 类 TraceMe
        class TraceMe(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 添加一个子模块 SomeScriptMod
                self.ssm = SomeScriptMod()

            def forward(self, x):
                # 返回子模块 SomeScriptMod 的 bar 方法返回值与输入张量 x 的和
                return self.ssm.bar() + x

        # 创建一个 TraceMe 类的实例 orig
        orig = TraceMe()
        # 对 orig 进行 TorchScript 追踪，传入一个随机张量作为输入
        traced = torch.jit.trace(orig, (torch.rand(4, 3),))

        # 对每个检查点进行验证，检查 TorchScript 模块对象和 Python 封装对象
        # 中是否都包含预期的方法和参数
        self.assertTrue(traced.ssm._c._has_method("foo"))
        self.assertTrue(hasattr(traced.ssm, "foo"))

        imported = self.getExportImportCopy(traced)

        self.assertTrue(imported.ssm._c._has_method("foo"))
        self.assertTrue(hasattr(imported.ssm, "foo"))

        self.assertTrue(imported.ssm.asm._c._has_method("bar"))
        self.assertTrue(hasattr(imported.ssm.asm, "bar"))

        self.assertTrue(hasattr(imported.ssm.asm, "param"))
    @_tmp_donotuse_dont_inline_everything
    # 使用装饰器禁止对函数进行优化和内联操作
    def test_call_script_fn_from_traced_module(self):
        # 定义一个 Torch 脚本函数，对输入 x 取负
        @torch.jit.script
        def scripted_fn(x):
            return torch.neg(x)

        # 定义一个继承自 torch.nn.Module 的类
        class TracedModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 添加一个名为 param 的参数，维度为 4x5
                self.param = torch.nn.Parameter(torch.rand(4, 5))

            def forward(self, x):
                # 调用之前定义的 Torch 脚本函数，对输入 x 和 self.param 进行矩阵乘法操作
                return scripted_fn(torch.mm(x, self.param))

        # 使用 torch.jit.trace 将 TracedModule 实例化为一个 Torch 脚本模块
        tm = torch.jit.trace(TracedModule(), torch.rand(3, 4))
        
        # 使用 FileCheck 对生成的图形进行验证，检查是否包含特定操作和函数调用
        FileCheck().check("aten::mm").check('name="scripted_fn"').check(
            "prim::CallFunction"
        ).run(str(tm.graph))

    @_tmp_donotuse_dont_inline_everything
    # 使用装饰器禁止对函数进行优化和内联操作
    def test_call_script_module_from_traced_module(self):
        # 定义一个继承自 torch.jit.ScriptModule 的脚本模块
        class ScriptMod(torch.jit.ScriptModule):
            def __init__(self):
                super().__init__()
                # 添加一个名为 param_foo 的参数，维度为 5x7
                self.param_foo = torch.nn.Parameter(torch.rand(5, 7))

            @torch.jit.script_method
            def forward(self, x):
                # 对输入 x 和 self.param_foo 进行矩阵乘法操作
                return torch.mm(x, self.param_foo)

        # 定义一个继承自 torch.nn.Module 的类
        class TracedModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 添加一个名为 param 的参数，维度为 4x5
                self.param = torch.nn.Parameter(torch.rand(4, 5))
                # 添加一个 ScriptMod 实例
                self.mod = ScriptMod()

            def forward(self, x):
                # 调用 self.mod 的 forward 方法，并加上常数 1.0
                return self.mod(torch.mm(x, self.param)) + 1.0

        # 使用 torch.jit.trace 将 TracedModule 实例化为一个 Torch 脚本模块
        tm = torch.jit.trace(TracedModule(), torch.rand(3, 4))

        # 使用 FileCheck 对生成的图形进行验证，检查是否包含特定操作和方法调用
        FileCheck().check("aten::mm").check("prim::CallMethod").check_same(
            "forward"
        ).check("aten::add").run(str(tm.graph))
    # 定义测试函数，用于测试从脚本函数调用跟踪函数
    def test_call_traced_fn_from_script_fn(self):
        # 装饰器，用于对函数进行追踪
        @_trace(torch.rand(3, 4))
        def traced_fn(x):
            # 返回输入张量的负值
            return torch.neg(x)

        # 使用 Torch 脚本装饰器声明的函数
        @torch.jit.script
        def script_fn(x):
            # 调用追踪函数，并对其结果加一
            return traced_fn(x) + 1

        # 运行文件检查，验证生成的 Torch 脚本图是否包含指定的操作序列
        FileCheck().check("prim::CallFunction").check("aten::add").run(
            str(script_fn.graph)
        )

    # 定义测试函数，用于测试从脚本函数调用跟踪模块时的行为
    def test_call_traced_mod_from_script_fn(self):
        # 断言捕获运行时错误，因为不能调用不是调用方子模块的 Torch 脚本模块
        with self.assertRaisesRegex(
            RuntimeError,
            "Cannot call a ScriptModule that is not a submodule of the caller",
        ):

            # 定义追踪模块，包含一个前向传播函数
            class TracedModule(torch.nn.Module):
                def forward(self, x):
                    # 返回输入张量与指定维度的零矩阵的矩阵乘积
                    return torch.mm(x, torch.zeros(4, 3))

            # 对追踪模块进行追踪，并传入随机生成的张量
            tm = torch.jit.trace(TracedModule(), torch.rand(3, 4))

            # 使用 Torch 脚本装饰器声明的函数
            @torch.jit.script
            def script_fn(x):
                # 调用追踪模块的前向传播函数，并对结果加一
                return tm(x) + 1

    # 装饰器，标记为临时，不要将所有内容内联化
    @_tmp_donotuse_dont_inline_everything
    # 定义测试函数，用于测试从脚本模块调用跟踪函数时的行为
    def test_call_tracing_fn_from_script_module(self):
        # 装饰器，用于对函数进行追踪
        @_trace(torch.rand(3, 3))
        def traced_fn(x):
            # 返回输入张量的负值
            return torch.neg(x)

        # 定义 Torch 脚本模块
        class ScriptMod(torch.jit.ScriptModule):
            def __init__(self):
                super().__init__()
                # 定义一个参数，是一个张量的参数对象
                self.param = torch.nn.Parameter(torch.rand(4, 3))

            @torch.jit.script_method
            def forward(self, x):
                # 调用追踪函数，对输入张量与参数张量进行矩阵乘法，然后取负值
                return traced_fn(torch.mm(x, self.param))

        # 创建 Torch 脚本模块的实例
        sm = ScriptMod()
        # 运行文件检查，验证生成的 Torch 脚本图是否包含指定的操作序列
        FileCheck().check("aten::mm").check("prim::CallFunction").run(
            str(sm.forward.graph)
        )

    # 装饰器，标记为临时，不要将所有内容内联化
    @_tmp_donotuse_dont_inline_everything
    # 定义测试函数，用于测试从脚本模块调用跟踪模块时的行为
    def test_call_tracing_mod_from_script_module(self):
        # 定义追踪模块，包含一个前向传播函数
        class TracedMod(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 定义一个参数，是一个张量的参数对象
                self.param = torch.nn.Parameter(torch.rand(3, 5))

            def forward(self, x):
                # 返回输入张量与参数张量的矩阵乘法结果
                return torch.mm(x, self.param)

        # 定义 Torch 脚本模块
        class ScriptMod(torch.jit.ScriptModule):
            def __init__(self):
                super().__init__()
                # 定义一个参数，是一个张量的参数对象
                self.param = torch.nn.Parameter(torch.rand(4, 3))
                # 对追踪模块进行追踪，并传入随机生成的张量
                self.tm = torch.jit.trace(TracedMod(), torch.rand(3, 3))

            @torch.jit.script_method
            def forward(self, x):
                # 调用追踪模块的前向传播函数，对输入张量与参数张量进行矩阵乘法
                return self.tm(torch.mm(x, self.param))

        # 创建 Torch 脚本模块的实例
        sm = ScriptMod()
        # 运行文件检查，验证生成的 Torch 脚本图是否包含指定的操作序列
        FileCheck().check("aten::mm").check("prim::CallMethod").run(str(sm.graph))

    # 定义测试函数，用于测试多参数情况下的脚本内联和追踪
    def test_script_inline_trace_multiple_args(self):
        # 定义一个普通的 PyTorch 模块，包含一个前向传播函数
        class M(torch.nn.Module):
            def forward(self, input, input2):
                # 返回两个输入张量的元素级加法结果
                return input + input2

        # 定义 Torch 脚本模块
        class M2(torch.jit.ScriptModule):
            def __init__(self):
                super().__init__()
                # 对普通模块进行追踪，并传入两个零矩阵张量作为参数
                self.m = torch.jit.trace(M(), (torch.zeros(4, 3), torch.zeros(4, 3)))

            @torch.jit.script_method
            def forward(self, inp):
                # 调用追踪得到的脚本模块，并传入一个输入张量
                return self.m(inp, inp)

        # 关闭优化执行
        with torch.jit.optimized_execution(False):
            # 创建 Torch 脚本模块的实例
            m2 = M2()
            # 调用脚本模块的前向传播函数，并传入一个零矩阵张量作为参数
            m2(torch.zeros(4, 3))
    def test_trace_dict_mix_script(self):
        # 定义一个内嵌的 PyTorch 模块 testB，继承自 nn.Module
        class testB(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 在 testB 中定义一个线性层，输入维度为 2，输出维度为 2
                self.linear = torch.nn.Linear(2, 2)

            # testB 的前向传播方法，接收一个特征映射的字典，值为 Tensor 列表，返回一个 Tensor
            def forward(self, feature_map: Dict[str, List[Tensor]]) -> Tensor:
                output = []
                # 遍历特征映射的值（Tensor 列表）
                for j in feature_map.values():
                    # 将每个列表中的第一个 Tensor 输入线性层并添加到输出列表中
                    output.append(self.linear(j[0]))

                # 将输出列表中的 Tensor 堆叠成一个 Tensor 返回
                return torch.stack(output)

        # 定义另一个内嵌的 PyTorch 模块 testA，继承自 nn.Module
        class testA(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 实例化 testB 并赋值给 testA 的成员变量 b，使用 torch.jit.script 对 testB 进行脚本化
                self.b = torch.jit.script(testB())

            # testA 的前向传播方法，接收一个输入映射的字典，值为 Tensor 列表，返回一个 Tensor
            def forward(self, input_map: Dict[str, List[Tensor]]) -> Tensor:
                feature_map = {}
                # 遍历输入映射的项目，将每个项目的第一个 Tensor 放入 feature_map 中
                for i, j in input_map.items():
                    feature_map[i] = [j[0]]

                # 调用脚本化的 testB 模块，传入 feature_map，返回其计算结果
                return self.b(feature_map)

        # 创建一个输入映射的字典，包含键为 "1" 和 "3" 的项目，每个项目有两个随机生成的 Tensor
        input_map = {
            "1": [torch.rand(2, 2), torch.rand(2, 2)],
            "3": [torch.rand(2, 2), torch.rand(2, 2)],
        }
        # 实例化 testA 模块为 model
        model = testA()
        # 使用 torch.jit.trace 对 model 进行跟踪，输入为 input_map
        traced_model = torch.jit.trace(model, input_map)
        # 创建一个新的输入映射的字典 new_input_map，键为 "1" 和 "3" 的项目，第一个 Tensor 不变，第二个使用 torch.randn 生成
        new_input_map = {
            "1": [torch.rand(2, 2), torch.randn(2, 2)],
            "3": [torch.rand(2, 2), torch.rand(2, 2)],
        }
        # 使用断言比较 model 和 traced_model 对新输入映射 new_input_map 的计算结果是否相同
        self.assertEqual(model(new_input_map), traced_model(new_input_map))
    def test_trace_script_returning_complex_dict(self):
        """测试跟踪返回复杂字典的脚本函数。
        字典应能递归地包含其他容器（如元组）。
        """

        class ReturnsDict(torch.nn.Module):
            def forward(
                self,
                id_score_list: Dict[
                    str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
                ],
            ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
                # 执行一些随机操作，然后返回相同结构的字典
                v = id_score_list["1000"]
                idx_keys = v[1] - 1500000
                weights = v[2]
                result = {"1000": (v[0], idx_keys, weights)}
                return result

        class ChecksDict(torch.nn.Module):
            def forward(
                self, input: Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
            ):
                v = input["1000"]
                return v[1] + 1

        class TestModule(torch.nn.Module):
            def __init__(self, checks_dict, returns_dict):
                super().__init__()
                self.checks_dict = checks_dict
                self.returns_dict = returns_dict

            def forward(
                self, input: Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
            ):
                foo = self.returns_dict(input)
                return self.checks_dict(foo)

        input1 = {
            "1000": (
                torch.tensor([0]),
                torch.tensor([], dtype=torch.int64),
                torch.tensor([]),
            )
        }

        input2 = {
            "1000": (
                torch.tensor([0]),
                torch.tensor([1500000, 1500004], dtype=torch.int64),
                torch.tensor([2.0, 3.0]),
            )
        }

        # 使用 Torch JIT 编译 ChecksDict 和 ReturnsDict 类
        checks_dict = torch.jit.script(ChecksDict())
        returns_dict = torch.jit.script(ReturnsDict())
        # 创建 TestModule 实例
        eager_module = TestModule(checks_dict, returns_dict)
        # 对 eager_module 进行跟踪编译
        traced_module = torch.jit.trace(eager_module, input1)
        # 断言跟踪模块和非跟踪模块在相同输入上的输出相等
        self.assertEqual(traced_module(input1), eager_module(input1))
        self.assertEqual(traced_module(input2), eager_module(input2))
    def test_trace_returning_dict_with_tensor_tuples(self):
        """对返回值为包含张量元组的字典的模块进行跟踪应当工作。

        这个测试函数包含两个内部定义的 PyTorch 模块类 ReturnsDict 和 ReturnsBadDict。
        ReturnsDict 类的 forward 方法返回一个字典，其值为张量元组。
        ReturnsBadDict 类的 forward 方法返回一个字典，其值包含一个张量和一个浮点数，这不符合预期类型。

        在测试中，我们首先创建 ReturnsDict 的实例 mod，并使用 torch.jit.trace 进行跟踪。
        然后调用跟踪后的模块 traced_module，并断言其输出与预期结果 expected 相等。

        如果模块的返回类型与预期不匹配，会引发 RuntimeError 异常。
        """
        
        class ReturnsDict(torch.nn.Module):
            def forward(
                self, k: torch.Tensor, v: torch.Tensor
            ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
                x = 2 * k
                y = 3 * v
                result = {"imakey": (x, y)}
                return result

        class ReturnsBadDict(torch.nn.Module):
            def forward(
                self, k: torch.Tensor, v: torch.Tensor
            ) -> Dict[str, Tuple[torch.Tensor, float]]:
                x = 2 * k
                result = {"imakey": (x, 1)}
                return result

        mod = ReturnsDict()
        traced_module = torch.jit.trace(
            mod, [torch.ones(1), torch.ones(1)], strict=False
        )
        out = traced_module(torch.ones(1), torch.ones(1))
        expected = {"imakey": (torch.tensor([2.0]), torch.tensor([3.0]))}
        self.assertEqual(out, expected)

        with self.assertRaisesRegex(
            RuntimeError, "cannot be understood by the tracer, only outputs matching"
        ):
            mod = ReturnsBadDict()
            traced_module = torch.jit.trace(
                mod, [torch.ones(1), torch.ones(1)], strict=False
            )

    def test_trace_linear(self):
        """测试线性层的跟踪功能。

        创建一个包含 20 个输入和 20 个输出的线性层 m。
        使用 checkTrace 函数验证线性层的跟踪结果。
        使用 torch.jit.trace 对线性层 m 进行跟踪，并获取其计算图 g。
        使用 FileCheck 来验证计算图中是否包含 "aten::linear" 操作。
        """
        m = torch.nn.Linear(20, 20)
        inp = torch.rand([20, 20])
        self.checkTrace(m, (inp,))
        g = torch.jit.trace(m, (inp,)).graph
        FileCheck().check("aten::linear").run(g)

    def test_traced_module_implements_interface(self):
        """测试跟踪模块是否实现了指定接口。

        定义一个名为 TestModuleInterface 的 PyTorch 接口，具有一个 forward 方法，
        接收两个张量参数并返回一个张量。
        创建一个 TestModule 类，实现 TestModuleInterface 接口，并包含一个卷积层 conv。
        定义一个函数 fn_takes_interface，接收一个 TestModuleInterface 类型的参数 x，
        并在函数体中调用 x 的 forward 方法。

        使用 torch.jit.script 对 TestModule 的实例进行脚本化。
        使用 checkScript 函数验证 fn_takes_interface 函数能够正确调用脚本化的 TestModule 实例。
        """
        @torch.jit.interface
        class TestModuleInterface(nn.Module):
            def forward(
                self, first_arg: torch.Tensor, second_arg: torch.Tensor
            ) -> torch.Tensor:
                pass

        make_global(TestModuleInterface)

        class TestModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(1, 1, 3)

            def forward(
                self, first_arg: torch.Tensor, second_arg: torch.Tensor
            ) -> torch.Tensor:
                return self.conv(first_arg) + second_arg

        def fn_takes_interface(x: TestModuleInterface):
            ones = torch.ones(1, 1, 3, 3)
            return x.forward(ones, ones)

        scripted_test_module = torch.jit.script(TestModule())
        self.checkScript(fn_takes_interface, (scripted_test_module,))
    def test_jit_trace_callfunction_return_shapes(self):
        # 定义一个 torch.jit.script 函数，将其作为 CallFunction 节点插入
        @torch.jit.script
        def inner_fn(x):
            return torch.cat((x, x))

        # 定义一个外部函数 outer_fn，它调用 inner_fn 并施加 relu 激活函数
        def outer_fn(x, y):
            return inner_fn(x + y).relu()

        # 创建两个随机张量 x 和 y
        x, y = [torch.rand((2, 2), dtype=torch.float) for _ in range(2)]
        # 对 outer_fn 进行追踪（tracing），生成追踪模块 fn_t
        fn_t = torch.jit.trace(outer_fn, (x, y))

        # 期望 CallFunction 节点的返回类型具有形状信息
        FileCheck().check("Float").check("4, 2").check("CallFunction").run(fn_t.graph)
        # 遍历 fn_t 图中的每个节点
        for n in fn_t.graph.nodes():
            # 如果节点的类型是 "prim::CallFunction"
            if n.kind() == "prim::CallFunction":
                # 断言节点的输出是完整的张量
                self.assertTrue(n.output().isCompleteTensor())
```