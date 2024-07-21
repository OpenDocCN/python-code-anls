# `.\pytorch\test\dynamo\test_aot_autograd.py`

```
# 导入必要的库和模块
import copy  # 导入copy模块，用于复制对象
import re  # 导入re模块，用于正则表达式匹配
import unittest  # 导入unittest模块，用于编写和运行测试
from textwrap import dedent  # 从textwrap模块导入dedent函数，用于移除多余的缩进
from unittest.mock import patch  # 从unittest.mock模块导入patch，用于在测试中模拟对象

import torch  # 导入PyTorch库

# 导入PyTorch内部模块和函数
import torch._dynamo
import torch._dynamo.test_case
import torch.fx.traceback as fx_traceback
import torch.utils._pytree as pytree
from torch._dynamo.testing import CompileCounter, expectedFailureDynamic, rand_strided
from torch._functorch.aot_autograd import _aot_export_function, create_functional_call
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import make_fx
from torch.profiler import profile
from torch.testing._internal.common_utils import compare_equal_outs_and_grads

# 定义一个可能会复制操作的函数
def maybe_dupe_op(x):
    y = x + 1  # 对输入张量加1，保存在y中
    z = x + 2  # 对输入张量加2，保存在z中
    if x.numel() < 5:
        return y, y  # 如果输入张量元素数量小于5，返回(y, y)
    else:
        return y, z  # 否则返回(y, z)

# 通过torch.ops.aten导入ATen运算符
aten = torch.ops.aten

# 创建一个自定义库对象lib，用于定义和实现函数
lib = torch.library.Library("custom", "DEF")  # noqa: TOR901
lib.define("maybe_dupe_op(Tensor a) -> (Tensor, Tensor)")  # 定义函数maybe_dupe_op的签名
lib.impl("maybe_dupe_op", maybe_dupe_op, "CPU")  # 在CPU环境下实现函数maybe_dupe_op
lib.impl("maybe_dupe_op", maybe_dupe_op, "Meta")  # 在Meta环境下实现函数maybe_dupe_op

# 创建一个测试类AotAutogradFallbackTests，继承自torch._dynamo.test_case.TestCase
class AotAutogradFallbackTests(torch._dynamo.test_case.TestCase):

    # 定义测试方法test_LSTM
    def test_LSTM(self):
        # 引用链接：https://github.com/pytorch/torchdynamo/issues/1147
        class Repro(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 创建一个双向LSTM模型self_mod_model_lstm_lstm
                self.self_mod_model_lstm_lstm = torch.nn.LSTM(
                    64, 64, num_layers=2, bidirectional=True
                )

            def forward(self, permute: torch.Tensor):
                # 执行LSTM模型self_mod_model_lstm_lstm的前向传播
                self_mod_model_lstm_lstm = self.self_mod_model_lstm_lstm(permute)
                return (self_mod_model_lstm_lstm,)

        mod = Repro()  # 创建Repro类的实例mod

        # 对模型mod进行AOT（Ahead-Of-Time）优化
        aot_mod = torch._dynamo.optimize("aot_eager")(mod)

        # 定义测试参数args
        args = [((92, 4, 64), (1, 5888, 92), torch.float32, "cpu", False)]
        args = [
            rand_strided(sh, st, dt, dev).requires_grad_(rg)
            for (sh, st, dt, dev, rg) in args
        ]

        # 在原模型上进行eager执行
        eager_result = mod(*args)
        # 在AOT优化后的模型上进行执行
        aot_result = aot_mod(*args)
        # 断言eager执行结果和AOT优化后执行结果一致
        self.assertTrue(torch._dynamo.testing.same(eager_result, aot_result))

    # 定义测试方法test_mutation
    def test_mutation(self):
        # 引用链接：https://github.com/pytorch/torchdynamo/issues/1301
        def fn(param, y):
            prev_grad = torch.is_grad_enabled()  # 获取当前梯度计算的开启状态
            try:
                torch.set_grad_enabled(False)  # 关闭梯度计算
                param.add_(y)  # 将参数param和y相加，不记录梯度
            finally:
                torch.set_grad_enabled(prev_grad)  # 恢复梯度计算的状态
            return y

        y = torch.randn(4)  # 创建一个形状为(4,)的随机张量y
        x = torch.nn.Parameter(torch.randn(4))  # 创建一个形状为(4,)的参数张量x
        aot_fn = torch._dynamo.optimize("aot_eager")(fn)  # 对函数fn进行AOT优化
        # 这应该不会出错：我们在no_grad模式下修改了一个autograd叶子节点。
        aot_fn(x, y)  # 调用AOT优化后的函数aot_fn，传入参数x和y
    # 定义一个测试方法 test_mutation1，用于测试函数 fn 的行为
    def test_mutation1(self):
        # 定义函数 fn，接受两个参数 _stack0 和 diagonal_chunked_attention_scores
        def fn(_stack0: torch.Tensor, diagonal_chunked_attention_scores: torch.Tensor):
            # 从 diagonal_chunked_attention_scores 中获取特定切片的数据，并赋给 getitem
            getitem = diagonal_chunked_attention_scores[
                (
                    slice(None, None, None),   # 第一维全取
                    slice(None, None, None),   # 第二维全取
                    slice(None, 256, None),    # 第三维从开始到256
                    slice(None, 257, None),    # 第四维从开始到257
                )
            ]
            # 将 getitem 的数据写入 _stack0 的特定切片区域
            _stack0[
                (
                    slice(None, None, None),   # 第一维全取
                    slice(None, -1, None),     # 第二维从开始到倒数第二个
                    slice(None, None, None),   # 第三维全取
                    slice(256, None, None),    # 第四维从256到结束
                )
            ] = getitem
            # 将 _stack0 重新视图为形状为 (1, 12, 1024, 513) 的张量 view
            view = _stack0.view(1, 12, 1024, 513)
            # 返回一个包含 view 的元组
            return (view,)

        # 生成形状为 [12, 4, 256, 513] 的随机张量 x
        x = torch.randn(torch.Size([12, 4, 256, 513]))
        # 生成形状为 [12, 3, 512, 513] 的随机张量 y
        y = torch.randn(torch.Size([12, 3, 512, 513]))
        # 对函数 fn 进行即时编译优化，生成 aot_fn
        aot_fn = torch._dynamo.optimize("aot_eager")(fn)
        # 调用优化后的函数 aot_fn，并传入 x 和 y 作为参数
        aot_fn(x, y)

    # 定义一个测试方法 test_negative_testing_mutation，用于测试负面场景下的函数 fn 的行为
    def test_negative_testing_mutation(self):
        # 定义函数 fn，接受两个参数 _stack0 和 diagonal_chunked_attention_scores
        def fn(_stack0: torch.Tensor, diagonal_chunked_attention_scores: torch.Tensor):
            # 从 diagonal_chunked_attention_scores 中获取特定切片的数据，并赋给 getitem
            getitem = diagonal_chunked_attention_scores[
                (
                    slice(None, None, None),   # 第一维全取
                    slice(None, None, None),   # 第二维全取
                    slice(None, 256, None),    # 第三维从开始到256
                    slice(None, 257, None),    # 第四维从开始到257
                )
            ]
            # 将 _stack0 的每个元素取正弦值后重新赋值给 _stack0
            _stack0 = torch.sin(_stack0)
            # 将 getitem 的数据写入 _stack0 的特定切片区域
            _stack0[
                (
                    slice(None, None, None),   # 第一维全取
                    slice(None, -1, None),     # 第二维从开始到倒数第二个
                    slice(None, None, None),   # 第三维全取
                    slice(256, None, None),    # 第四维从256到结束
                )
            ] = getitem
            # 将 _stack0 重新视图为形状为 (1, 12, 1024, 513) 的张量 view
            view = _stack0.view(1, 12, 1024, 513)
            # 返回一个包含 view 的元组
            return (view,)

        # 生成形状为 [12, 4, 256, 513] 的随机张量 x
        x = torch.randn(torch.Size([12, 4, 256, 513]))
        # 生成形状为 [12, 3, 512, 513] 的随机张量 y
        y = torch.randn(torch.Size([12, 3, 512, 513]))
        # 对函数 fn 进行即时编译优化，生成 aot_fn
        aot_fn = torch._dynamo.optimize("aot_eager")(fn)
        # 调用优化后的函数 aot_fn，并传入 x 和 y 作为参数
        aot_fn(x, y)

    # 定义一个测试方法 test_negative_testing，用于测试负面场景下的函数 fn 的行为
    def test_negative_testing(self):
        # 定义函数 fn，接受两个参数 x 和 y，返回对 x 取正弦后与 y 相加的结果
        def fn(x, y):
            return torch.sin(x).add_(y)

        # 生成形状为 [4] 的随机张量 y
        y = torch.randn(4)
        # 生成形状为 [4] 的随机张量 x
        x = torch.randn(4)
        # 对函数 fn 进行即时编译优化，生成 aot_fn
        aot_fn = torch._dynamo.optimize("aot_eager")(fn)
        # 调用优化后的函数 aot_fn，并传入 x 和 y 作为参数
        aot_fn(x, y)
    def test_call_fn_with_non_const_inputs_aot_safe(self):
        # 定义一个特殊的 Module 类，用于测试
        class ModuleSpecialFwd(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 创建一个二维卷积层，输入通道数为3，输出通道数为20，卷积核大小为(5, 5)
                self.conv = torch.nn.Conv2d(
                    in_channels=3, out_channels=20, kernel_size=(5, 5)
                )

            def _conv_forward(self, x):
                # 调用卷积层的前向方法，传入输入 x，权重和偏置
                return self.conv._conv_forward(x, self.conv.weight, self.conv.bias)

            def forward(self, x):
                # 调用 _conv_forward 方法进行前向传播
                return self._conv_forward(x)

        # 初始化 ModuleSpecialFwd 实例
        mod = ModuleSpecialFwd()
        # 生成一个形状为 [3, 10, 10] 的随机张量
        rx = torch.randn([3, 10, 10])

        # 实际运行模型
        real = mod(rx)

        # 导出模型的计算图
        graph, _ = torch._dynamo.export(mod)(rx)

        # 使用 AOT（Ahead-of-Time）模式运行导出的图
        self.assertTrue(torch._dynamo.testing.same(real, graph(rx)))

        # 对导出的图进行 AOT 优化
        aot_fn = torch._dynamo.optimize("aot_eager")(graph)
        # 运行优化后的图
        aot_fn(rx)

    def test_call_fn_with_non_const_inputs_aot_unsafe(self):
        # 定义一个特殊的 Module 类，用于测试
        class ModuleSpecialFwd(torch.nn.Module):
            def _some_bad_fwd(self, param, y):
                # 保存当前梯度计算状态并关闭梯度计算
                prev_grad = torch.is_grad_enabled()
                try:
                    torch.set_grad_enabled(False)
                    # 在禁用梯度计算的情况下修改参数 param
                    param.add_(y)
                finally:
                    # 恢复原来的梯度计算状态
                    torch.set_grad_enabled(prev_grad)
                return y

            def forward(self, x, y):
                # 调用 _some_bad_fwd 方法进行前向传播
                return self._some_bad_fwd(x, y)

        # 初始化 ModuleSpecialFwd 实例
        mod = ModuleSpecialFwd()
        # 创建一个形状为 [4] 的参数张量 x，并赋予随机值
        x = torch.nn.Parameter(torch.randn(4))
        # 创建一个形状为 [4] 的随机张量 y
        y = torch.randn([4])

        # 实际运行模型
        real = mod(x, y)

        # 导出模型的计算图
        graph, _ = torch._dynamo.export(mod)(x, y)

        # 断言两种运行方式的结果相等
        self.assertTrue(torch._dynamo.testing.same(real, graph(x, y)))

        # 使用 AOT（Ahead-of-Time）模式运行导出的图
        aot_fn = torch._dynamo.optimize("aot_eager")(graph)
        # 运行优化后的图，这里会在无梯度计算模式下修改自动求导的叶节点，不应该抛出错误
        aot_fn(x, y)
    # 定义一个测试函数，用于测试在非常量输入下使用 ahead-of-time（AOT）不安全的控制流
    def test_call_fn_with_non_const_inputs_aot_unsafe_control_flow(self):
        # 定义一个特殊的 PyTorch 模块 ModuleSpecialFwd
        class ModuleSpecialFwd(torch.nn.Module):
            # 定义一个不良的前向方法 _some_bad_fwd，接受参数 param 和 y
            def _some_bad_fwd(self, param, y):
                # 如果 y[0][0] 小于 3，返回 y + param
                if y[0][0] < 3:
                    return y + param
                # 否则返回 param * y
                return param * y

            # 定义正向传播方法 forward，接受输入 x 和 y
            def forward(self, x, y):
                # 计算 a = x * y
                a = x * y
                # 调用不良的前向方法 _some_bad_fwd 处理 a，并重新赋值给 a
                a = self._some_bad_fwd(a, a)
                # 计算 b = x + y
                b = x + y
                # 返回 a * b 的结果作为正向传播的输出
                return a * b

        # 初始化模块 mod
        mod = ModuleSpecialFwd()
        # 创建一个形状为 [2, 2] 的随机张量 x，并封装成 Parameter 类型
        x = torch.nn.Parameter(torch.randn([2, 2]))
        # 创建一个形状为 [2, 2] 的随机张量 y
        y = torch.randn([2, 2])

        # 运行实际的正向传播
        real = mod(x, y)

        # 准备运行优化，使用我们的捕获函数
        gms = []
        counter = CompileCounter()

        # 定义捕获函数 capturing_fn，接受图模块 gm 和输入 inputs
        def capturing_fn(gm, inputs):
            nonlocal gms
            # 将当前图模块 gm 添加到列表 gms 中
            gms.append(gm)
            # 返回 counter 对捕获的图模块 gm 和输入 inputs 进行处理后的结果
            return counter(gm, inputs)

        # 对模块 mod 进行优化，使用 torch._dynamo.optimize 包装 capturing_fn
        optimized_mod = torch._dynamo.optimize(capturing_fn)(mod)

        # 断言真实输出与优化输出相等
        self.assertTrue(torch._dynamo.testing.same(real, optimized_mod(x, y)))

        # 取消注释以重现下面注释掉的图形
        # for gm in gms:
        #     print("GM CODE", gm.code)

        # 断言 frame_count 和 op_count 的值分别为 4 和 7
        self.assertEqual(counter.frame_count, 4)
        self.assertEqual(counter.op_count, 7)

        # 图形 1
        # def forward(self, x : torch.nn.parameter.Parameter, y : torch.Tensor):
        #     mul = x * y;  x = y = None
        #     return (mul,)
        # BREAK
        # 图形 2
        # def forward(self, y : torch.Tensor):
        #     getitem = y[0];  y = None
        #     getitem_1 = getitem[0];  getitem = None
        #     lt = getitem_1 < 3;  getitem_1 = None
        #     return (lt,)
        # BREAK
        # 图形 3
        # def forward(self, param : torch.Tensor, y : torch.Tensor):
        #     add = y + param;  y = param = None
        #     return (add,)
        # BREAK
        # 图形 4
        # def forward(self, _stack0 : torch.Tensor, x : torch.nn.parameter.Parameter, y : torch.Tensor):
        #     add = x + y;  x = y = None
        #     mul = _stack0 * add;  _stack0 = add = None
        #     return (mul,)

        # 使用 AOT 运行函数
        torch._dynamo.reset()

        # 对优化后的模块 optimized_mod 应用 "aot_eager" 优化
        aot_fn = torch._dynamo.optimize("aot_eager")(optimized_mod)
        aot_fn(x, y)

    # 注意：Dynamo 重新编译以防止无效的梯度
    #
    # 此测试在精神上相当于 test_invalid_requires_grad_fake 在 test_autodispatch.py 中的测试
    # 此测试的目的是以一种通常会触发断言的方式调用 aot_autograd，证明我们正确重新编译 Dynamo 并保护此条件
    #
    # 子注意：test_invalid_requires_grad_fake 使用虚假张量的原因是因为 Dynamo 将虚假张量传递给 aot_autograd。
    @patch("torch._functorch.config.debug_assert", True)
    # 定义一个测试方法，验证通过 Dynamo 重新编译后是否能正确处理 requires_grad 属性
    def test_requires_grad_fake_via_dynamo_recompiles(self):
        
        # 定义一个简单的神经网络模块
        class F(torch.nn.Module):
            # 前向传播方法，接受两个参数 x 和 y，返回它们的和组成的元组
            def forward(self, x, y):
                return (x + y,)

        # 创建两个张量 x 和 y，形状为 (3, 3)，且都需要计算梯度
        x = torch.randn(3, 3, requires_grad=True)
        y = torch.randn(3, 3, requires_grad=True)
        
        # 创建一个张量 z，形状为 (3, 3)，不需要计算梯度
        z = torch.randn(3, 3, requires_grad=False)

        # 创建一个编译计数器对象 cc，用于记录编译次数，使用后端 "aot_eager"
        cc = torch._dynamo.testing.CompileCounterWithBackend("aot_eager")

        # 定义失败处理函数的变量 failure_reason
        failure_reason = None

        # 定义一个失败守卫函数，将失败原因记录到 failure_reason 中
        def guard_fail_fn(failure):
            nonlocal failure_reason
            failure_reason = failure[0]

        # 通过 Dynamo 优化模块 F()，得到优化后的模块 fxy，并比较其输出和梯度
        fxy = torch._dynamo.optimize(cc, guard_fail_fn=guard_fail_fn)(F())
        compare_equal_outs_and_grads(self, F(), fxy, (x, y))
        compare_equal_outs_and_grads(self, F(), fxy, (x, z))
        
        # 断言 failure_reason 中包含特定的消息，指出 'y' 张量的 requires_grad 属性不匹配
        self.assertIn(
            """tensor 'L['y']' requires_grad mismatch. expected requires_grad=1""",
            failure_reason,
        )

        # 重置 failure_reason
        failure_reason = None

        # 断言编译计数器 cc 的帧数为 2
        self.assertEqual(cc.frame_count, 2)

        # 重置 Dynamo 状态，为新后端做准备
        torch._dynamo.reset()  # for new backend
        cc = torch._dynamo.testing.CompileCounterWithBackend("aot_eager")

        # 通过 Dynamo 优化模块 F()，得到优化后的模块 fxz，并比较其输出和梯度
        fxz = torch._dynamo.optimize(cc, guard_fail_fn=guard_fail_fn)(F())
        compare_equal_outs_and_grads(self, F(), fxz, (x, z))
        compare_equal_outs_and_grads(self, F(), fxz, (x, z))
        
        # 断言编译计数器 cc 的帧数为 1
        self.assertEqual(cc.frame_count, 1)
        
        # 断言 failure_reason 为 None，表示没有失败原因
        self.assertTrue(failure_reason is None)
    # 定义一个测试方法，用于测试双向传播时的错误情况
    def test_double_backward_errors(self):
        # 在修复双向传播问题后，删除此测试
        # 遍历两种梯度输出情况：一个有梯度的张量，一个没有
        for grad_output in (torch.tensor(1.0, requires_grad=True), None):
            # 创建一个需要梯度的张量 x
            x = torch.tensor(1.0, requires_grad=True)
            # 错误消息，指出 torch.compile 与 aot_autograd 目前不支持双向传播
            err = "torch.compile with aot_autograd does not currently support double backward"

            # 下面这些情况应该是等价的:

            # (1) 双向传播完全在编译函数内部
            def f1(x):
                # 计算 y = sin(x).exp()
                y = x.sin().exp()
                # 计算 y 对 x 的梯度，并创建计算图以支持二阶梯度
                (gx,) = torch.autograd.grad(
                    y, x, create_graph=True, grad_outputs=grad_output
                )
                # 对 gx 再次进行梯度计算，进行二阶梯度计算
                torch.autograd.grad(gx, x)
                return gx

            # 使用 aot_eager 后端编译函数 f1
            compiled_f1 = torch.compile(backend="aot_eager")(f1)
            # 直接调用 f1 进行计算
            f1(x)
            # 预期捕获 RuntimeError 异常并检查其消息是否符合预期错误信息
            with self.assertRaisesRegex(RuntimeError, err):
                compiled_f1(x)

            # (2) 双向传播的后半部分在编译函数外部
            def f2(x):
                y = x.sin().exp()
                (gx,) = torch.autograd.grad(
                    y, x, create_graph=True, grad_outputs=grad_output
                )
                return gx

            compiled_f2 = torch.compile(backend="aot_eager")(f2)
            # 使用编译后的函数计算 gx
            gx = compiled_f2(x)
            # 预期捕获 RuntimeError 异常并检查其消息是否符合预期错误信息
            with self.assertRaisesRegex(RuntimeError, err):
                torch.autograd.grad(gx, x)

            # (3) 双向传播完全在编译函数外部
            def f3(x):
                y = x.sin().exp()
                return y

            compiled_f3 = torch.compile(backend="aot_eager")(f3)
            # 使用编译后的函数计算 y
            y = compiled_f3(x)
            # 计算 y 对 x 的梯度，并创建计算图以支持二阶梯度
            (gx,) = torch.autograd.grad(
                y, x, create_graph=True, grad_outputs=grad_output
            )
            # 预期捕获 RuntimeError 异常并检查其消息是否符合预期错误信息
            with self.assertRaisesRegex(RuntimeError, err):
                torch.autograd.grad(gx, x)

        # create_graph=False 的情况
        # 定义一个不需要创建计算图的函数 f4
        def f4(x):
            y = x.sin().exp()
            return y

        # 使用 aot_eager 后端编译函数 f4
        compiled_f4 = torch.compile(backend="aot_eager")(f4)
        # 创建一个需要梯度的张量 x
        x = torch.tensor(1.0, requires_grad=True)
        # 使用编译后的函数计算 y
        y = compiled_f4(x)
        # 计算 y 对 x 的梯度，不需要创建计算图
        (gx,) = torch.autograd.grad(y, x, create_graph=False, grad_outputs=grad_output)

    # 使用 patch 函数设置 torch._functorch.config.debug_assert 为 True
    @patch("torch._functorch.config.debug_assert", True)
    # 定义一个测试方法，用于测试动态编译器在重复参数时重新编译的行为
    def test_arg_dupe_via_dynamo_recompiles(self):
        # 定义一个继承自 torch.nn.Module 的内嵌类 F
        class F(torch.nn.Module):
            # 定义该类的前向传播方法，接受两个参数 x 和 y
            def forward(self, x, y):
                # 对输入 x 进行截断操作，并在原地修改
                x = x.trunc_()
                # 对输入 y 进行截断操作，并在原地修改
                y = y.trunc_()
                # 返回一个元组，包含 x + y 的结果
                return (x + y,)

        # 生成一个形状为 (3, 3) 的随机张量 x，并标记为需要梯度计算
        x = torch.randn(3, 3, requires_grad=True)
        # 克隆 x 成四个副本 x1, x2, x3, x4
        x1, x2, x3, x4 = x.clone(), x.clone(), x.clone(), x.clone()
        
        # 生成一个形状为 (3, 3) 的随机张量 y，并标记为需要梯度计算
        y = torch.randn(3, 3, requires_grad=True)
        # 克隆 y 成三个副本 y1, y2, y4
        y1, y2, y4 = y.clone(), y.clone(), y.clone()

        # 创建一个编译计数器对象 cc，使用 "aot_eager" 后端
        cc = torch._dynamo.testing.CompileCounterWithBackend("aot_eager")

        # 定义一个变量用于存储失败原因，默认为 None
        failure_reason = None

        # 定义一个函数 guard_fail_fn 用于处理失败时的回调
        def guard_fail_fn(failure):
            nonlocal failure_reason
            # 将失败原因记录为失败信息的第一个元素
            failure_reason = failure[0]

        # 对类 F 的实例进行动态优化，并返回优化后的函数 fxy
        fxy = torch._dynamo.optimize(cc, guard_fail_fn=guard_fail_fn)(F())
        
        # 第一次调用 fxy，并传入 x1 和 y1 作为参数
        fxy(x1, y1)
        # 第二次调用 fxy，并传入 x2 和 y2 作为参数
        fxy(x2, y2)

        # 断言失败原因为 None，即没有失败
        self.assertTrue(failure_reason is None)

        # 重置失败原因
        failure_reason = None

        # 断言编译计数器的帧数为 1
        self.assertEqual(cc.frame_count, 1)

        # 重置动态编译器状态，为新的后端做准备
        torch._dynamo.reset()

        # 使用 "aot_eager" 后端再次创建编译计数器对象 cc
        cc = torch._dynamo.testing.CompileCounterWithBackend("aot_eager")

        # 对类 F 的实例进行动态优化，并返回优化后的函数 fxx
        fxx = torch._dynamo.optimize(cc, guard_fail_fn=guard_fail_fn)(F())
        
        # 第一次调用 fxx，并传入 x3 作为两个参数
        fxx(x3, x3)
        # 第二次调用 fxx，并传入 x4 和 y4 作为参数
        fxx(x4, y4)

        # 断言编译计数器的帧数为 2
        self.assertEqual(cc.frame_count, 2)
        # 断言失败原因中包含字符串 """L['x'] is L['y']"""
        self.assertIn("""L['x'] is L['y']""", failure_reason)
    def test_arg_dupe_via_dynamo_recompiles_many_args_param_non_tensor_arg(self):
        # 定义一个测试方法，用于检测在动态编译中通过重新编译处理多个参数和非张量参数的情况

        class F(torch.nn.Module):
            # 定义一个继承自torch.nn.Module的类F
            def __init__(self):
                super().__init__()
                self.mean = torch.nn.Parameter(torch.randn(3, 3))
                # 初始化方法，创建一个形状为(3, 3)的随机张量，并封装为Parameter

            def forward(self, a, b, e, f):
                # 前向传播方法，接受四个参数a, b, e, f
                a.trunc_()
                # 对参数a进行截断操作
                b.trunc_()
                # 对参数b进行截断操作
                return (a + b + self.mean) * e * f
                # 返回一个张量表达式 (a + b + self.mean) * e * f

        a = torch.randn(3, 3, requires_grad=True)
        # 创建一个形状为(3, 3)的随机张量a，需要计算梯度
        b = torch.randn(3, 3, requires_grad=True)
        # 创建一个形状为(3, 3)的随机张量b，需要计算梯度
        a1, a2 = a.clone(), a.clone()
        # 克隆张量a，得到a1和a2
        b1, b2 = b.clone(), b.clone()
        # 克隆张量b，得到b1和b2

        failure_reason = None
        # 初始化失败原因为None

        def guard_fail_fn(failure):
            # 定义一个保护失败的函数guard_fail_fn，接受一个失败列表failure
            nonlocal failure_reason
            # 声明failure_reason为非局部变量
            failure_reason = failure[0]
            # 将failure的第一个元素赋值给failure_reason

        self.assertTrue(failure_reason is None)
        # 使用self.assertTrue断言确保failure_reason为None

        cc = torch._dynamo.testing.CompileCounterWithBackend("aot_eager")
        # 创建一个动态编译计数器cc，使用aot_eager作为后端

        f = torch._dynamo.optimize(cc, guard_fail_fn=guard_fail_fn)(F())
        # 对类F应用动态优化，使用cc作为编译计数器，并指定guard_fail_fn作为保护失败函数
        f(a1, a1, 2, 2)
        # 调用f函数，传入参数a1, a1, 2, 2
        f(a2, b2, 2, 2)
        # 再次调用f函数，传入参数a2, b2
        self.assertEqual(cc.frame_count, 2)
        # 使用self.assertEqual断言，检查编译计数器cc的帧数为2
        self.assertIn(
            """L['a'] is L['b']""",
            failure_reason,
        )
        # 使用self.assertIn断言，检查字符串"""L['a'] is L['b']"""是否在failure_reason中

        torch._dynamo.reset()
        # 重置动态编译环境

        cc = torch._dynamo.testing.CompileCounterWithBackend("aot_eager")
        # 再次创建一个动态编译计数器cc，使用aot_eager作为后端

        c = torch.randn(3, 3, requires_grad=True)
        # 创建一个形状为(3, 3)的随机张量c，需要计算梯度
        d = torch.randn(3, 3, requires_grad=True)
        # 创建一个形状为(3, 3)的随机张量d，需要计算梯度
        c3, c4 = c.clone(), c.clone()
        # 克隆张量c，得到c3和c4
        d3, d4 = d.clone(), d.clone()
        # 克隆张量d，得到d3和d4

        f = torch._dynamo.optimize(cc, guard_fail_fn=guard_fail_fn)(F())
        # 对类F应用动态优化，使用cc作为编译计数器，并指定guard_fail_fn作为保护失败函数
        f(c3, c3, 3, 3)
        # 调用f函数，传入参数c3, c3, 3, 3
        f(c4, d4, 3, 3)
        # 再次调用f函数，传入参数c4, d4
        self.assertEqual(cc.frame_count, 2)
        # 使用self.assertEqual断言，检查编译计数器cc的帧数为2
        self.assertIn("""L['a'] is L['b']""", failure_reason)
        # 使用self.assertIn断言，检查字符串"""L['a'] is L['b']"""是否在failure_reason中

    @patch("torch._functorch.config.debug_assert", True)
    # 使用patch装饰器，将torch._functorch.config.debug_assert设置为True
    def test_arg_dupe_via_dynamo_recompiles_many_with_global(self):
        # 定义一个测试方法，用于检测在动态编译中通过重新编译处理多个带全局变量的参数

        z = None
        # 初始化变量z为None

        class F(torch.nn.Module):
            # 定义一个继承自torch.nn.Module的类F
            def __init__(self):
                super().__init__()
                self.mean = torch.nn.Parameter(torch.randn(3, 3))
                # 初始化方法，创建一个形状为(3, 3)的随机张量，并封装为Parameter

            def forward(self, a, b, e, f):
                # 前向传播方法，接受四个参数a, b, e, f
                a.trunc_()
                # 对参数a进行截断操作
                b.trunc_()
                # 对参数b进行截断操作
                return (a + b + z + self.mean) * e * f
                # 返回一个张量表达式 (a + b + z + self.mean) * e * f，包含全局变量z

        a = torch.randn(3, 3, requires_grad=True)
        # 创建一个形状为(3, 3)的随机张量a，需要计算梯度
        b = torch.randn(3, 3, requires_grad=True)
        # 创建一个形状为(3, 3)的随机张量b，需要计算梯度
        z = a
        # 将z赋值为a
        a1, a2 = a.clone(), a.clone()
        # 克隆张量a，得到a1和a2
        b1, b2 = b.clone(), b.clone()
        # 克隆张量b，得到b1和b2

        failure_reason = None
        # 初始化失败原因为None

        def guard_fail_fn(failure):
            # 定义一个保护失败的函数guard_fail_fn，接受一个失败列表failure
            nonlocal failure_reason
            # 声明failure_reason为非局部变量
            failure_reason = failure[0]
            # 将failure的第一个元素赋值给failure_reason

        self.assertTrue(failure_reason is None)
        # 使用self.assertTrue断言确保failure_reason为None

        cc = torch._dynamo.testing.CompileCounterWithBackend("aot_eager")
        # 创建一个动态编译计数器cc，使用aot_eager作为后端

        f = torch._dynamo.optimize(cc, guard_fail_fn=guard_fail_fn)(F())
        # 对类F应用动态优化，使用cc作为编译计数器，并指定guard_fail_fn作为保护失败函数
        f(a1, a1, 2, 2)
        # 调用f函数，传入参数a1, a1, 2, 2
        f(a2, b2, 2, 2)
        # 再次调用f函数，传入参数a2, b2
        self.assertEqual(cc.frame_count, 2)
        # 使用self.assertEqual断言，检查编译计数器cc的帧数为2
        self.assertIn(
            """L['a'] is L['b']""",
            failure_reason,
        )
        # 使用self.assertIn断言，检查字符串"""L['a'] is L['b']"""是否在failure_reason中
    def test_arg_dupe_via_dynamo_recompiles_many_args_param_non_tensor_arg_list(self):
        # 定义一个测试方法，验证通过Dynamo重新编译时重复参数的处理
        class F(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.mean = torch.nn.Parameter(torch.randn(3, 3))

            def forward(self, e, f, a, b):
                # 修改a和b的值为其截断值
                a.trunc_()
                b.trunc_()
                # 返回计算结果
                return (a + b + self.mean) * e[0] * f[0]

        # 创建两个需要梯度的随机张量a和b
        a = torch.randn(3, 3, requires_grad=True)
        b = torch.randn(3, 3, requires_grad=True)
        # 克隆a和b，生成a1, a2和b1, b2
        a1, a2 = a.clone(), a.clone()
        b1, b2 = b.clone(), b.clone()

        # 失败原因初始化为None
        failure_reason = None

        # 定义一个用于处理失败的函数
        def guard_fail_fn(failure):
            nonlocal failure_reason
            failure_reason = failure[0]

        # 断言失败原因为None
        self.assertTrue(failure_reason is None)

        # 初始化编译计数器cc，并设置后端为"aot_eager"
        cc = torch._dynamo.testing.CompileCounterWithBackend("aot_eager")

        # 优化F模块，当有失败时调用guard_fail_fn函数
        f = torch._dynamo.optimize(cc, guard_fail_fn=guard_fail_fn)(F())
        # 调用f两次，参数为[3, 2, 1], [4, 5, 6], a1, a1 和 [3, 2, 1], [4, 5, 6], a2, b2
        f([3, 2, 1], [4, 5, 6], a1, a1)
        f([3, 2, 1], [4, 5, 6], a2, b2)
        # 断言编译计数为2
        self.assertEqual(cc.frame_count, 2)
        # 断言失败原因中包含"L['a'] is L['b']"
        self.assertIn(
            """L['a'] is L['b']""",
            failure_reason,
        )

        # 重置Dynamo状态
        torch._dynamo.reset()

        # 再次初始化编译计数器cc，并设置后端为"aot_eager"
        cc = torch._dynamo.testing.CompileCounterWithBackend("aot_eager")

        # 创建两个需要梯度的随机张量c和d
        c = torch.randn(3, 3, requires_grad=True)
        d = torch.randn(3, 3, requires_grad=True)
        # 克隆c和d，生成c3, c4和d3, d4
        c3, c4 = c.clone(), c.clone()
        d3, d4 = d.clone(), d.clone()

        # 优化F模块，当有失败时调用guard_fail_fn函数
        f = torch._dynamo.optimize(cc, guard_fail_fn=guard_fail_fn)(F())
        # 调用f两次，参数为[3, 2, 1], [4, 5, 6], c3, c3 和 [3, 2, 1], [4, 5, 6], c4, d4
        f([3, 2, 1], [4, 5, 6], c3, c3)
        f([3, 2, 1], [4, 5, 6], c4, d4)
        # 断言编译计数为2
        self.assertEqual(cc.frame_count, 2)
    # 定义一个测试函数，用于验证在使用动态编译优化时处理重复参数时的情况
    def test_arg_dupe_via_dynamo_recompiles_many_args_param(self):
        
        # 定义一个继承自torch.nn.Module的子类F
        class F(torch.nn.Module):
            # 构造函数，初始化mean为一个3x3的随机张量作为可学习参数
            def __init__(self):
                super().__init__()
                self.mean = torch.nn.Parameter(torch.randn(3, 3))

            # 前向传播函数，接受参数a和b，对a和b进行截断操作后返回a + b + self.mean
            def forward(self, a, b):
                a.trunc_()  # 截断操作，修改a的值
                b.trunc_()  # 截断操作，修改b的值
                return a + b + self.mean

        # 生成两个3x3的随机张量a和b，并标记为需要计算梯度
        a = torch.randn(3, 3, requires_grad=True)
        b = torch.randn(3, 3, requires_grad=True)
        # 克隆a和b，生成a1, a2和b1, b2
        a1, a2 = a.clone(), a.clone()
        b1, b2 = b.clone(), b.clone()

        # 初始化失败原因为None
        failure_reason = None

        # 定义一个用于设置失败原因的函数
        def guard_fail_fn(failure):
            nonlocal failure_reason  # 使用nonlocal关键字声明failure_reason为非局部变量
            failure_reason = failure[0]

        # 断言当前失败原因为None
        self.assertTrue(failure_reason is None)

        # 初始化编译计数器cc，并设置其后端为"aot_eager"
        cc = torch._dynamo.testing.CompileCounterWithBackend("aot_eager")

        # 使用动态编译优化f，并将guard_fail_fn作为失败处理函数，对F的实例进行优化
        f = torch._dynamo.optimize(cc, guard_fail_fn=guard_fail_fn)(F())
        f(a1, a1)  # 调用f，并传入a1和a1作为参数
        f(a2, b2)  # 再次调用f，并传入a2和b2作为参数
        # 断言编译帧数为2
        self.assertEqual(cc.frame_count, 2)
        # 断言failure_reason中包含特定的错误信息
        self.assertIn(
            """L['a'] is L['b']""",
            failure_reason,
        )

        # 重置动态编译环境
        torch._dynamo.reset()

        # 重新初始化编译计数器cc，并设置其后端为"aot_eager"
        cc = torch._dynamo.testing.CompileCounterWithBackend("aot_eager")

        # 生成两个3x3的随机张量c和d，并标记为需要计算梯度
        c = torch.randn(3, 3, requires_grad=True)
        d = torch.randn(3, 3, requires_grad=True)
        # 克隆c和d，生成c3, c4和d3, d4
        c3, c4 = c.clone(), c.clone()
        d3, d4 = d.clone(), d.clone()

        # 使用动态编译优化f，并将guard_fail_fn作为失败处理函数，对F的实例进行优化
        f = torch._dynamo.optimize(cc, guard_fail_fn=guard_fail_fn)(F())
        f(c3, c3)  # 调用f，并传入c3和c3作为参数
        f(c4, d4)  # 再次调用f，并传入c4和d4作为参数
        # 断言编译帧数为2
        self.assertEqual(cc.frame_count, 2)
        # 断言failure_reason中包含特定的错误信息
        self.assertIn("""L['a'] is L['b']""", failure_reason)
    # 定义一个测试方法，用于测试通过 Dynamo 重新编译带有多个参数的函数
    def test_arg_dupe_via_dynamo_recompiles_many_args(self):
        # 定义一个继承自 torch.nn.Module 的类 F
        class F(torch.nn.Module):
            # 定义 forward 方法，接受四个参数 a, b, c, d
            def forward(self, a, b, c, d):
                # 在每个参数上执行 trunc_() 方法，原地截断操作
                a.trunc_()
                b.trunc_()
                c.trunc_()
                d.trunc_()
                # 返回一个包含四个参数相加的元组
                return (a + b + c + d,)

        # 创建两个随机张量 a 和 b，要求梯度计算
        a = torch.randn(3, 3, requires_grad=True)
        b = torch.randn(3, 3, requires_grad=True)
        # 克隆张量 a 和 b，每个克隆版本分别命名为 a1, a2, a3, a4 和 b1, b2, b3, b4
        a1, a2, a3, a4 = a.clone(), a.clone(), a.clone(), a.clone()
        b1, b2, b3, b4 = b.clone(), b.clone(), b.clone(), b.clone()

        # 初始化失败原因为 None
        failure_reason = None

        # 定义一个函数 guard_fail_fn 作为失败回调，记录失败原因
        def guard_fail_fn(failure):
            nonlocal failure_reason
            failure_reason = failure[0]

        # 断言失败原因为 None
        self.assertTrue(failure_reason is None)

        # 创建一个编译计数器 cc，使用 "aot_eager" 后端进行测试
        cc = torch._dynamo.testing.CompileCounterWithBackend("aot_eager")

        # 使用 Dynamo 进行优化，并传入失败回调函数
        f = torch._dynamo.optimize(cc, guard_fail_fn=guard_fail_fn)(F())
        # 调用 f 函数，传入四个相同的张量 a1
        f(a1, a1, a1, a1)
        # 调用 f 函数，传入第一个参数为 a2，其余为 b2
        f(a2, b2, b2, b2)
        # 断言编译帧数为 2
        self.assertEqual(cc.frame_count, 2)
        # 断言失败原因中包含指定的字符串
        self.assertIn(
            """L['a'] is L['b']""",
            failure_reason,
        )

        # 重置 Dynamo 状态
        torch._dynamo.reset()

        # 重新创建编译计数器 cc，使用 "aot_eager" 后端进行测试
        cc = torch._dynamo.testing.CompileCounterWithBackend("aot_eager")

        # 创建两个新的随机张量 c 和 d，要求梯度计算
        c = torch.randn(3, 3, requires_grad=True)
        d = torch.randn(3, 3, requires_grad=True)
        # 克隆张量 c 和 d，分别命名为 c3, c4 和 d3, d4
        c3, c4 = c.clone(), c.clone()
        d3, d4 = d.clone(), d.clone()

        # 使用 Dynamo 进行优化，并传入失败回调函数
        f = torch._dynamo.optimize(cc, guard_fail_fn=guard_fail_fn)(F())
        # 调用 f 函数，传入参数 a3, b3, c3, c3
        f(a3, b3, c3, c3)
        # 调用 f 函数，传入参数 a4, b4, c4, d4
        f(a4, b4, c4, d4)
        # 断言编译帧数为 2
        self.assertEqual(cc.frame_count, 2)
        # 断言失败原因中包含指定的字符串
        self.assertIn("""L['c'] is L['d']""", failure_reason)

    # 定义一个测试方法，用于测试输入别名情况
    def test_alias_inputs(self):
        # 定义一个内部函数 fn
        def fn():
            # 创建包含单个元素的张量 a
            a = torch.tensor([1])
            # 从 a 中获取切片 [0:1]，得到长度为 1 的张量 a
            a = a[0:1]
            # 对 a 进行挤压操作，去除维度为 1 的轴
            b = a.squeeze()
            # 修改 a 的第一个元素为 0
            a[0] = 0
            # 如果 a 的第一个元素小于 1e5，执行占位语句
            if a[0] < 1e5:
                pass
            # 再次修改 a 的第一个元素为 2
            a[0] = 2
            # 返回经过挤压操作后的张量 b
            return b

        # 记录 fn 函数的参考输出
        ref_output = fn()
        # 使用 "aot_eager" 后端编译 fn 函数
        aot_fn = torch._dynamo.optimize("aot_eager")(fn)
        # 调用编译后的函数，并记录实际输出
        actual_output = aot_fn()
        # 断言参考输出与实际输出相等
        self.assertEqual(ref_output, actual_output)

    # 定义一个测试方法，用于测试梯度输入别名情况
    def test_grad_inputs_alias_inputs(self):
        # 定义一个继承自 torch.autograd.Function 的类 Test
        class Test(torch.autograd.Function):
            # 定义前向传播方法 forward，接受两个参数 x 和 y
            @staticmethod
            def forward(ctx, x, y):
                # 保存 x 到上下文 ctx 中
                ctx.save_for_backward(x)
                # 返回 y 作为输出
                return y

            # 定义反向传播方法 backward，接受上下文 ctx 和梯度 grad
            @staticmethod
            def backward(ctx, grad):
                # 从上下文 ctx 中取出保存的张量 x
                (x,) = ctx.saved_tensors
                # 返回 x 和 grad 作为反向传播的输出
                return x, grad

        # 定义一个函数 fn，接受两个参数 x 和 y
        def fn(x, y):
            # 调用 Test 类的 apply 方法，将 x 和 y 作为参数传入
            return Test.apply(x, y)

        # 创建一个值为 1 的张量 x，要求梯度计算
        x = torch.ones(1, requires_grad=True)
        # 创建一个值为 1 的张量 y，要求梯度计算
        y = torch.ones(1, requires_grad=True)
        # 使用 "aot_eager" 后端编译 fn 函数
        compiled_fn = torch.compile(fn, backend="aot_eager")
        # 调用编译后的函数，传入张量 x 和 y，并记录输出
        out = compiled_fn(x, y)
        # 对输出进行求和，并执行反向传播
        out.sum().backward()

    # 标记下面的测试为预期动态失败，链接到 GitHub 上的相关问题
    @expectedFailureDynamic  # https://github.com/pytorch/pytorch/issues/103539
    # 配置 _dynamo 中的 automatic_dynamic_shapes 参数为 False
    @torch._dynamo.config.patch(automatic_dynamic_shapes=False)
    # 使用 patch 方法将 debug_assert 方法配置为 True
    @patch("torch._functorch.config.debug_assert", True)
    def test_multiple_aot_autograd_calls_dupe_args(self):
        # 处理 aot_module_simplified 函数期望子模块始终返回元组或列表的情况
        class WrapperModule(torch.nn.Module):
            def __init__(self, mod):
                super().__init__()
                self.mod = mod

            def forward(self, *args):
                # 调用输入模块的 forward 方法
                out = self.mod(*args)
                # 如果返回值是列表或元组，则直接返回；否则将其封装成元组返回
                if isinstance(out, (list, tuple)):
                    return out
                return (out,)

        def compile_submod(input_mod, args):
            from functorch.compile import nop
            from torch._functorch.aot_autograd import aot_module_simplified

            class WrapperModule(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.original = input_mod
                    # 使用 aot_module_simplified 对输入模块进行编译，使用 nop 作为参数
                    self.submod = aot_module_simplified(input_mod, args, nop)

                def forward(self, *args):
                    # 调用经过编译后的子模块的 forward 方法
                    return self.submod(*args)

            return WrapperModule()

        def test_compile(fx_g, example_inps):
            # 将 fx_g 模块按照包含 "mul" 的节点进行分割
            split_gm = torch.fx.passes.split_module.split_module(
                fx_g, None, lambda node: 1 if "mul" in str(node) else 0
            )
            # 对第一个子模块使用 compile_submod 函数进行编译
            submod_1_inps = split_gm.submod_0(*example_inps)
            split_gm.submod_0 = compile_submod(
                WrapperModule(split_gm.submod_0), example_inps
            )
            # 对第二个子模块使用 compile_submod 函数进行编译
            split_gm.submod_1 = compile_submod(
                WrapperModule(split_gm.submod_1), submod_1_inps
            )
            return split_gm

        # 使用 torch._dynamo.optimize 装饰器对 f 函数进行优化编译
        @torch._dynamo.optimize(test_compile)
        def f(a):
            # 调用自定义的 maybe_dupe_op 操作符对输入 a 进行处理
            b, c = torch.ops.custom.maybe_dupe_op(a)
            # 返回经过乘法操作的结果元组
            return (b.mul_(c),)

        # 调用 f 函数两次，分别传入长度为 4 和 6 的全一张量
        f(torch.ones(4))
        f(torch.ones(6))

    def test_nn_parameter_construction(self):
        # 解决 GitHub 问题链接中描述的问题
        def fn(x):
            # 对输入张量 x 进行正弦函数计算
            y = x.sin()
            # 创建一个值为 1 的 torch.nn.Parameter 参数
            z = torch.nn.Parameter(torch.ones(1))
            # 返回正弦计算结果与参数 z 的和
            return y + z

        # 创建一个形状为 (4, 4) 的随机张量 x
        x = torch.rand((4, 4))

        # 使用 torch._dynamo.optimize("aot_eager") 对 fn 函数进行优化编译
        opt_fn = torch._dynamo.optimize("aot_eager")(fn)
        # 断言 fn(x) 和经过优化后的 opt_fn(x) 结果相同
        self.assertTrue(torch._dynamo.testing.same(fn(x), opt_fn(x)))
# 模拟一个CSV格式的数据，包含三列：SeqNr, OrigAten, SrcFn
SeqNr|OrigAten|SrcFn
# 第一行数据示例
0|aten.convolution.default|l__self___conv1
# 第二行数据示例
0|aten.add.Tensor|l__self___bn1
# 第三行数据示例
1|aten._native_batch_norm_legit_functional.default|l__self___bn1
# 第四行数据示例
2|aten.relu.default|l__self___relu1
# 第五行数据示例
2|aten.detach.default|l__self___relu1
# 第六行数据示例
2|aten.detach.default|l__self___relu1
# 第七行数据示例
3|aten.add.Tensor|add
# 第八行数据示例
4|aten.view.default|flatten
# 第九行数据示例
5|aten.view.default|l__self___fc1
# 第十行数据示例
6|aten.t.default|l__self___fc1
# 第十一行数据示例
7|aten.addmm.default|l__self___fc1
# 第十二行数据示例
8|aten.view.default|l__self___fc1
# 第十三行数据示例
9|aten.sub.Tensor|l__self___loss_fn
# 第十四行数据示例
10|aten.abs.default|l__self___loss_fn
# 第十五行数据示例
11|aten.mean.default|l__self___loss_fn
# 第十六行数据示例
11|aten.ones_like.default|
# 第十七行数据示例
11|aten.expand.default|
# 第十八行数据示例
11|aten.div.Scalar|
# 第十九行数据示例
10|aten.sgn.default|
# 第二十行数据示例
10|aten.mul.Tensor|
# 第二十一行数据示例
8|aten.view.default|
# 第二十二行数据示例
7|aten.t.default|
# 第二十三行数据示例
7|aten.mm.default|
# 第二十四行数据示例
7|aten.t.default|
# 第二十五行数据示例
7|aten.mm.default|
# 第二十六行数据示例
7|aten.t.default|
# 第二十七行数据示例
7|aten.sum.dim_IntList|
# 第二十八行数据示例
7|aten.view.default|
# 第二十九行数据示例
6|aten.t.default|
# 第三十行数据示例
5|aten.view.default|
# 第三十一行数据示例
4|aten.view.default|
# 第三十二行数据示例
2|aten.detach.default|
# 第三十三行数据示例
2|aten.detach.default|
# 第三十四行数据示例
2|aten.threshold_backward.default|
# 第三十五行数据示例
1|aten.native_batch_norm_backward.default|
# 第三十六行数据示例
0|aten.convolution_backward.default|
# 第三十七行数据示例
11|aten.add.Tensor|
# 以下为测试函数的定义，略过不注释
"""
            ),
        )

    def test_split_with_sizes_aot_autograd_cleans_up_traceback_meta(self):
        from torch._functorch.aot_autograd import setup_stacktrace_preservation_hooks

        def fn(result, split_sizes):
            rs = torch.ops.aten.split_with_sizes(result, split_sizes.tolist())
            return rs

        example_inputs = (
            torch.randn(32, requires_grad=True),
            torch.tensor((7, 16, 9)),
        )
        outs = fn(*example_inputs)
        setup_stacktrace_preservation_hooks([out.grad_fn for out in outs])
        with fx_traceback.preserve_node_meta():
            (outs[0].sum() + outs[1].sum() + outs[2].sum()).backward()

        self.assertNotIn("grad_fn_seq_nr", fx_traceback.current_meta)
        self.assertNotIn("in_grad_fn", fx_traceback.current_meta)

    # https://github.com/pytorch/pytorch/issues/110121
    def test_aot_export_joint_simple_repro(self):
        class Mod(torch.nn.Module):
            def __init__(self, *args, **kwargs) -> None:
                super().__init__(*args, **kwargs)
                self.linear = torch.nn.Linear(5, 7)

            def forward(self, x):
                return self.linear(x)

        def mini_backend(gm, sample_inputs):
            from torch._functorch.aot_autograd import aot_export_joint_simple

            fake_mode = torch._dynamo.utils.detect_fake_mode(sample_inputs)

            with patch.object(fake_mode, "allow_non_fake_inputs", True), fake_mode:
                return aot_export_joint_simple(gm, sample_inputs, trace_joint=False)

        sample_inputs = [torch.rand((3, 4, 5))]
        model = Mod()
        m_compiled = torch.compile(model, backend=mini_backend)

        out_ref = model(*sample_inputs)
        out_test = m_compiled(*sample_inputs)
        self.assertEqual(out_ref, out_test)
    def test_eager_sequence_nr(self):
        # 定义一个内部模型类，继承自torch.nn.Module
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 定义一个2维卷积层，输入和输出通道数均为16，卷积核大小为(1, 1)，步长为1，填充方式为"same"，包含偏置
                self.conv1 = torch.nn.Conv2d(
                    in_channels=16,
                    out_channels=16,
                    kernel_size=(1, 1),
                    stride=1,
                    padding="same",
                    bias=True,
                )
                # 定义一个批归一化层，特征数量为16
                self.bn1 = torch.nn.BatchNorm2d(num_features=16)
                # 定义一个ReLU激活函数层
                self.relu1 = torch.nn.ReLU()
                # 定义一个全连接层，输入特征数为1638400，输出特征数为1
                self.fc1 = torch.nn.Linear(in_features=1638400, out_features=1)
                # 定义一个L1损失函数
                self.loss_fn = torch.nn.L1Loss()

            # 定义前向传播方法，接受输入x和目标target
            def forward(self, x, target):
                y = x
                x = self.conv1(x)
                x = self.bn1(x)
                x = self.relu1(x)
                x = x + y
                x = torch.flatten(x)
                x = self.fc1(x)
                output = self.loss_fn(x, target)

                return (output,)

        # 定义一个函数grad_with_create_graph，接受模型mod、输入x和目标target
        def grad_with_create_graph(mod, x, target):
            y = mod(x, target)
            # 使用torch.autograd.grad计算梯度，设置create_graph=True以确保后向操作的sequence_nr继续递减
            (gx,) = torch.autograd.grad(
                y[0], x, create_graph=True, grad_outputs=grad_output
            )
            return gx

        # 创建一个大小为(100, 16, 32, 32)的随机张量x，设置requires_grad=True
        x = torch.rand(100, 16, 32, 32, requires_grad=True)
        # 创建一个大小为(1,)的随机张量target
        target = torch.rand(1)
        # 实例化模型类Model，得到模型实例mod
        mod = Model()
        # 构造参数列表args，包含mod、x和target
        args = [mod, x, target]
        # 创建一个大小为1.0的张量grad_output，设置requires_grad=True
        grad_output = torch.tensor(1.0, requires_grad=True)
        # 使用torch.compile(backend="aot_eager")编译函数grad_with_create_graph，得到编译后的模型实例compiled_f1
        compiled_f1 = torch.compile(backend="aot_eager")(grad_with_create_graph)
        # 将编译后的模型实例赋值给model_instance
        model_instance = compiled_f1
        # 使用profile进行性能分析，记录CPU活动并记录张量形状
        with profile(
            activities=[torch.profiler.ProfilerActivity.CPU],
            record_shapes=True,
        ) as kineto_prof:
            # 执行model_instance(*args)，得到结果res
            res = model_instance(*args)
        # 创建一个空集合bwd_set
        bwd_set = set()
        # 初始化性能分析结果字符串prof_str，包含列标题
        prof_str = "SeqNr|Thread|FwdThread|Name\n"
        # 遍历性能分析事件kineto_prof.events()
        for event in kineto_prof.events():
            # 如果事件的sequence_nr >= 0
            if event.sequence_nr >= 0:
                # 将事件信息添加到prof_str中
                prof_str = (
                    prof_str + f"{event.sequence_nr}|{event.thread}"
                    f"|{event.fwd_thread}|{event.name}|\n"
                )
                # 如果事件名称匹配"Backward[01]"，将其sequence_nr添加到bwd_set中
                if re.search(r"Backward[01]", event.name):
                    bwd_set.add(event.sequence_nr)
        # 使用self.assertTrue断言，验证bwd_set中元素数量为13
        self.assertTrue(len(bwd_set), 13)
    # 定义测试函数 test_aot_grad_mode_mutation，用于测试即时编译和自动求导模式的变异
    def test_aot_grad_mode_mutation(self):
        # 遍历编译器列表，分别进行测试
        for compiler in ["aot_eager", "inductor"]:

            # 定义函数 f(x)，计算输入 x 的平方并关闭梯度追踪
            def f(x):
                y = x * x
                torch.set_grad_enabled(False)
                return y.clone(), y

            # 使用指定的编译器编译函数 f
            f_compiled = torch.compile(f, backend=compiler, fullgraph=True)

            # 启用梯度追踪
            torch.set_grad_enabled(True)
            # 创建需要求导的张量 x
            x = torch.ones(3, requires_grad=True) * 3
            # 计算参考值 y_ref
            y_ref = f(x)
            # 断言当前是否关闭了梯度追踪
            self.assertEqual(torch.is_grad_enabled(), False)
            # 再次启用梯度追踪
            torch.set_grad_enabled(True)
            # 使用编译后的函数计算 y
            y = f_compiled(x)
            # 断言当前是否关闭了梯度追踪
            self.assertEqual(torch.is_grad_enabled(), False)
            # 再次启用梯度追踪
            torch.set_grad_enabled(True)
            # 断言两种计算方式得到的结果相等
            self.assertEqual(y_ref, y)

            # 断言参考值 y_ref 的第一个元素没有梯度函数
            self.assertIsNone(y_ref[0].grad_fn)
            # 断言编译后的结果 y 的第一个元素没有梯度函数
            self.assertIsNone(y[0].grad_fn)

            # 断言参考值 y_ref 的第二个元素有梯度函数
            self.assertIsNotNone(y_ref[1].grad_fn)
            # 断言编译后的结果 y 的第二个元素有梯度函数
            self.assertIsNotNone(y[1].grad_fn)

            # 检查对输入计算梯度时，给定输入时计算的梯度是否相同
            # y[0] 的切线对梯度无关紧要，因此 grad_required=False
            self.assertEqual(
                sum(y_ref[1].grad_fn(torch.tensor([-1.0, 2.0, 0.0]))),
                sum(
                    x
                    for x in y[1].grad_fn.apply(None, torch.tensor([-1.0, 2.0, 0.0]))
                    if x is not None
                ),
            )

    # 定义测试函数 test_aot_autograd_raises_invalid_leaf_set，用于测试自动求导中的不合法叶子节点设置
    def test_aot_autograd_raises_invalid_leaf_set(self):
        # 定义函数 f(x)，尝试在其内部执行不合法的原地操作
        @torch.compile
        def f(x):
            x.set_(torch.ones(2))

        # 确保调用函数 f(x) 会触发运行时异常，指出存在原地操作使用不合法的叶子节点
        x = torch.ones(2, requires_grad=True)
        with self.assertRaisesRegex(
            RuntimeError, "is being used in an in-place operation"
        ):
            f(x)

    # 定义测试函数 test_aot_autograd_expand_mutation_functionalizes，测试自动求导中 expand 操作的变异功能
    def test_aot_autograd_expand_mutation_functionalizes(self):
        # 定义函数 fn(x)，对输入 x 进行扩展并进行操作
        def fn(x):
            y = x.expand(3, *x.shape)
            y[0, 0].add_(5)
            return y

        # 使用 aot_eager 编译函数 fn
        opt_fn = torch.compile(fn, backend="aot_eager")

        # 创建张量 x，并克隆用于优化的副本 x_opt
        x = torch.arange(6)
        x_opt = x.clone().detach()
        # 断言未编译和编译后的结果相同
        self.assertEqual(fn(x), opt_fn(x_opt))
        # 断言 x 和 x_opt 相等
        self.assertEqual(x, x_opt)

    # 定义测试函数 test_aot_autograd_expand_mutation_backwards，测试自动求导中 expand 操作的反向传播功能
    def test_aot_autograd_expand_mutation_backwards(self):
        # 定义函数 fn(x, z)，对输入 x 和 z 进行扩展并进行操作
        def fn(x, z):
            y = x.expand(3, *x.shape)
            y[1, 1].mul_(5)
            ret = y * z
            return ret

        # 使用 aot_eager 编译函数 fn
        opt_fn = torch.compile(fn, backend="aot_eager")

        # 创建张量 x 和 z，并克隆用于优化的副本 x_opt 和 z_opt
        x = torch.arange(6, dtype=torch.float)
        z = x.clone().detach()
        x_opt = x.clone().detach()
        z_opt = x.clone().detach()

        # 设置 z 和 z_opt 需要计算梯度
        z.requires_grad = True
        z_opt.requires_grad = True

        # 计算未编译和编译后的结果，并进行比较
        res = fn(x, z)
        opt_res = opt_fn(x_opt, z_opt)
        self.assertEqual(res, opt_res)

        # 分别对 res 和 opt_res 进行求和后反向传播
        res.sum().backward()
        opt_res.sum().backward()

        # 断言 x 和 x_opt 相等
        self.assertEqual(x, x_opt)
        # 断言 z 和 z_opt 的梯度相等
        self.assertEqual(z.grad, z_opt.grad)
    # 定义测试函数，验证在禁止不安全数据指针访问的情况下复制张量
    def test_data_ptr_access_copy(self):
        # 导入 torch._functorch.config 模块，用于配置
        import torch._functorch.config as _config

        # 使用 _config.patch 上下文管理器，禁止不安全数据指针访问
        with _config.patch(fake_tensor_allow_unsafe_data_ptr_access=False):
            # 进入 FakeTensorMode 上下文
            with FakeTensorMode():
                # 创建一个随机张量 x
                x = torch.randn(3)
                # 使用 copy.copy 复制张量 x 到 y
                y = copy.copy(x)
        # 断言复制后的张量 y 的形状与原张量 x 相同
        self.assertEqual(y.shape, x.shape)

    # 测试在前向传播中数据指针访问失败的情况
    def test_data_ptr_access_fails_in_forward(self):
        # 使用 torch.library._scoped_library 上下文，设置库 "mylib" 片段模式
        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            # 定义 mylib::foo 函数签名和实现
            torch.library.define("mylib::foo", "(Tensor x) -> Tensor", lib=lib)

            # 定义 CompositeImplicitAutograd 下的 mylib::foo 实现
            @torch.library.impl("mylib::foo", "CompositeImplicitAutograd", lib=lib)
            def _(x):
                # 尝试访问张量 x 的数据指针
                x.data_ptr()
                # 返回张量 x 的克隆
                return x.clone()

            # 创建一个随机张量 x
            x = torch.randn(3)

            # 定义输入为图的数据指针访问函数
            def data_ptr_graph_input(x):
                r0 = torch.ops.mylib.foo(x)
                return r0

            # 定义中间值为图的数据指针访问函数
            def data_ptr_graph_intermediate(x):
                y = x.clone()
                r0 = torch.ops.mylib.foo(y)
                return r0

            # 测试用例集合
            tests = [data_ptr_graph_input, data_ptr_graph_intermediate]

            # 定义上下文管理器 ctx，用于捕获预期的 RuntimeError 异常
            def ctx():
                return self.assertRaisesRegex(
                    RuntimeError, "Cannot access data pointer"
                )

            # 对每个测试函数 f 执行以下操作
            for f in tests:
                # 使用 ctx 上下文，期望捕获 RuntimeError 异常
                with ctx():
                    make_fx(f, tracing_mode="fake")(x)
                with ctx():
                    make_fx(f, tracing_mode="symbolic")(x)
                with ctx():
                    torch.compile(f, backend="eager", fullgraph=True)(x)

    # 测试在反向传播中数据指针访问失败的情况
    def test_data_ptr_access_fails_in_backward(self):
        # 使用 torch.library._scoped_library 上下文，设置库 "mylib" 片段模式
        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            # 定义 mylib::foo 函数签名和实现
            torch.library.define("mylib::foo", "(Tensor x) -> Tensor", lib=lib)

            # 定义一个标志变量，用于检测反向传播是否调用
            backward_called = False

            # 定义一个自定义的 torch.autograd.Function Foo 类
            class Foo(torch.autograd.Function):
                @staticmethod
                def forward(ctx, x):
                    return x.clone()

                @staticmethod
                def backward(ctx, grad):
                    nonlocal backward_called
                    backward_called = True
                    # 尝试访问梯度 grad 的数据指针
                    grad.data_ptr()
                    return grad.clone()

            # 定义 CompositeImplicitAutograd 下的 mylib::foo 实现
            @torch.library.impl("mylib::foo", "CompositeImplicitAutograd", lib=lib)
            def _(x):
                return Foo.apply(x)

            # 定义函数 f，调用 mylib::foo
            def f(x):
                return torch.ops.mylib.foo(x)

            # 创建一个随机张量 x，要求梯度跟踪
            x = torch.randn(3, requires_grad=True)

            # 使用 self.assertRaisesRegex 上下文，期望捕获 RuntimeError 异常
            with self.assertRaisesRegex(RuntimeError, "Cannot access data pointer"):
                # 编译并执行函数 f，使用 AOT eager 后端和完整图形模式
                y = torch.compile(f, backend="aot_eager", fullgraph=True)(x)

            # 断言反向传播已被调用
            self.assertTrue(backward_called)

    # 预期此测试用例失败，因为无法捕获对同一内存位置的多次变异
    @unittest.expectedFailure
    # 定义一个测试方法，用于测试自动微分扩展时的变异错误
    def test_aot_autograd_expand_mutation_error(self):
        # 定义一个函数 fn，接受参数 x
        def fn(x):
            # 对 x 进行扩展操作，扩展为 3 行 * x 的形状
            y = x.expand(3, *x.shape)
            # 对 y 的第一列的前三行加上 5（in-place 操作）
            y[0:3, 0].add_(5)
            # 返回处理后的 y
            return y

        # 使用 Torch 编译函数 fn，使用 AOT eager 后端
        opt_fn = torch.compile(fn, backend="aot_eager")

        # 创建一个张量 x，包含 0 到 5 的整数
        x = torch.arange(6)
        # 克隆并分离张量 x，得到 x_opt
        x_opt = x.clone().detach()
        
        # 断言调用 fn(x) 会引发异常
        with self.assertRaises(Exception):
            fn(x)
        # 断言调用 opt_fn(x_opt) 会引发异常
        with self.assertRaises(Exception):
            opt_fn(x_opt)
# 如果当前脚本是作为主程序运行
if __name__ == "__main__":
    # 从 torch._dynamo.test_case 模块导入 run_tests 函数
    from torch._dynamo.test_case import run_tests

    # 运行测试函数 run_tests()
    run_tests()
```