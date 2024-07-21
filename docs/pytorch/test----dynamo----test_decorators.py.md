# `.\pytorch\test\dynamo\test_decorators.py`

```py
# 导入必要的模块和库
# Owner(s): ["module: dynamo"]
import functools  # 导入 functools 模块，用于装饰器功能
import os  # 导入 os 模块，提供操作系统相关功能
import unittest.mock as mock  # 导入 unittest.mock 模块，并命名为 mock，用于单元测试的模拟对象
from unittest.mock import patch  # 从 unittest.mock 中导入 patch 函数，用于单元测试中的模拟补丁

import torch  # 导入 PyTorch 库

import torch._dynamo.test_case  # 导入 PyTorch 内部测试用例相关模块
import torch._dynamo.testing  # 导入 PyTorch 内部测试相关模块
from torch._dynamo.exc import IncorrectUsage  # 从 torch._dynamo.exc 模块导入 IncorrectUsage 异常类


def my_custom_function(x):
    # 自定义函数，对输入 x 执行加一操作并返回结果
    return x + 1


class DecoratorTests(torch._dynamo.test_case.TestCase):
    def test_disallow_in_graph(self):
        cnts = torch._dynamo.testing.CompileCounter()  # 创建编译计数器对象 cnts

        @torch._dynamo.optimize(cnts)
        def fn(a):
            x = torch.add(a, 1)  # 在图优化装饰器下对 a 加 1
            x = torch.add(x, 1)  # 在图优化装饰器下再对 x 加 1
            x = torch.sub(x, 1)  # 在图优化装饰器下对 x 减 1
            x = torch.add(x, 1)  # 在图优化装饰器下再对 x 加 1
            x = torch.add(x, 1)  # 在图优化装饰器下再对 x 加 1
            return x

        torch._dynamo.disallow_in_graph(torch.sub)  # 在图优化装饰器下禁止对 torch.sub 的使用
        fn(torch.randn(10))  # 调用 fn 函数并传入随机生成的张量

        torch._dynamo.allow_in_graph(torch.sub)  # 在图优化装饰器下允许对 torch.sub 的使用

        # 检查图优化过程中对 torch.sub 操作的次数
        self.assertEqual(cnts.frame_count, 2)
        self.assertEqual(cnts.op_count, 4)

    def test_disable_for_custom_op(self):
        import torch.library  # 导入 torch.library 模块
        from torch.library import Library  # 从 torch.library 导入 Library 类

        foo = Library("foo", "DEF")  # 创建名为 "foo" 的 Library 对象，设置为 "DEF"
        foo.define("custom(Tensor self) -> Tensor")  # 定义名为 "custom" 的自定义操作

        # 定义名为 "foo_cpu" 的自定义 CPU 实现函数
        @torch.library.impl(foo, "custom", "CPU")
        def foo_cpu(x):
            return x.nonzero()

        # 禁用 torch.ops.foo.custom 操作，由于 torch.library Python API 中存在额外的 Python 帧
        torch.ops.foo.custom = torch._dynamo.disable(torch.ops.foo.custom)

        def fn(x):
            a = torch.nn.functional.relu(x)  # 对 x 执行 ReLU 激活函数
            b = torch.ops.foo.custom(a)  # 调用自定义操作 torch.ops.foo.custom
            c = torch.cos(b)  # 对 b 执行余弦函数
            return c

        x = torch.randint(2, (100,))  # 创建一个形状为 (100,) 的整数张量 x
        ref = fn(x)  # 调用 fn 函数并传入张量 x，得到参考结果 ref

        cnts = torch._dynamo.testing.CompileCounter()  # 创建新的编译计数器对象 cnts
        opt_fn = torch._dynamo.optimize(cnts)(fn)  # 对 fn 函数进行图优化
        res = opt_fn(x)  # 调用优化后的函数 opt_fn，并传入张量 x，得到优化结果 res
        self.assertEqual(cnts.frame_count, 2)  # 检查图优化过程中的帧数
        self.assertEqual(ref, res)  # 检查优化结果是否与参考结果一致

    def test_disable_ignores_outer_wraps(self):
        def orig_inner():
            pass

        def inner():
            pass

        inner._torchdynamo_orig_callable = orig_inner

        @functools.wraps(inner)
        def wrapper():
            raise AssertionError("wrapper called")

        # torch._dynamo.disable 函数不会影响外部装饰器包装的函数
        w = torch._dynamo.disable(fn=wrapper, recursive=True)
    # 定义一个测试用例，验证禁用神经网络模块前向钩子的功能
    def test_disable_nn_modules_forward_hook(self):
        
        # 定义一个简单的线性模块
        class SimpleLinear(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer0 = torch.nn.Linear(4, 4)

            # 线性模块的前向传播函数
            def forward(self, inp):
                # 对输入数据进行 sigmoid 激活后，通过第一个线性层处理
                return self.layer0(torch.sigmoid(inp))

        # 定义一个简单的模型
        class SimpleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer0 = SimpleLinear()  # 包含一个 SimpleLinear 模块
                self.layer1 = torch.nn.Linear(4, 4)

            # 模型的前向传播函数
            def forward(self, inp):
                # 对输入数据先进行 sin 函数处理，然后通过第一个线性层，最后经过第二个线性层
                z = self.layer0(torch.sin(inp))
                return self.layer1(z)

        # 定义一个前向钩子函数
        def hook(module, args):
            # 对输入数据进行 sigmoid 激活，并返回
            inp = args[0].sigmoid()
            return (inp,)

        model = SimpleModel()  # 创建一个 SimpleModel 实例

        # 注册前向钩子到 layer0 上
        model.layer0.register_forward_pre_hook(hook)

        # 禁用模块的 monkeypatching
        model.layer0 = torch._dynamo.disable(model.layer0)

        # 创建一个编译计数器，使用 "eager" 后端
        cnts = torch._dynamo.testing.CompileCounterWithBackend("eager")

        # 编译模型
        opt_model = torch.compile(model, backend=cnts)

        # 对随机输入数据进行推理
        opt_model(torch.randn(4))

        # 检查是否没有破坏计算图
        self.assertEqual(cnts.frame_count, 2)

        gm0 = cnts.graphs[0]
        # 检查第一个计算图中是否存在 sin 节点，且不存在 sigmoid 节点
        self.assertTrue(any(node.target is torch.sin for node in gm0.graph.nodes))
        self.assertTrue(
            all(node.target is not torch.sigmoid for node in gm0.graph.nodes)
        )

        gm1 = cnts.graphs[1]
        # 检查第二个计算图中是否不存在 sigmoid 节点。sigmoid 在前向钩子和禁用的模块中都被使用。
        self.assertTrue(
            all(node.target is not torch.sigmoid for node in gm1.graph.nodes)
        )
    # 定义测试方法：使用类装饰器禁用神经网络模块
    def test_disable_nn_module_with_class_decorator(self):
        # 创建一个编译计数器，指定后端为 "eager"
        cnts = torch._dynamo.testing.CompileCounterWithBackend("eager")

        # 使用装饰器禁用动态图的功能
        @torch._dynamo.disable
        # 定义简单的线性神经网络模块
        class SimpleLinear(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer0 = torch.nn.Linear(4, 4)

            def forward(self, inp):
                # 在前向传播中使用 torch.sigmoid 函数
                return self.layer0(torch.sigmoid(inp))

        # 使用编译装饰器编译模型，指定后端为 cnts
        @torch.compile(backend=cnts)
        # 定义简单的模型类
        class SimpleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer0 = SimpleLinear()
                self.layer1 = torch.nn.Linear(4, 4)

            def forward(self, inp):
                # 在模型的前向传播中，调用子模块 self.layer0
                z = self.layer0(torch.sin(inp))
                return self.layer1(z)

        # 定义钩子函数，修改输入的第一个参数为其 sigmoid 函数的输出
        def hook(module, args):
            inp = args[0].sigmoid()
            return (inp,)

        # 创建 SimpleModel 实例
        model = SimpleModel()
        # 注册前向传播钩子函数到 model.layer0
        model.layer0.register_forward_pre_hook(hook)

        # 对模型进行推理
        model(torch.randn(4))

        # 断言：检查帧计数是否为 2
        self.assertEqual(cnts.frame_count, 2)

        # 获取第一个图形 gm0
        gm0 = cnts.graphs[0]
        # 断言：检查第一个图形中是否存在 sin 节点，且不存在 sigmoid 节点
        self.assertTrue(any(node.target is torch.sin for node in gm0.graph.nodes))
        self.assertTrue(
            all(node.target is not torch.sigmoid for node in gm0.graph.nodes)
        )

        # 获取第二个图形 gm1
        gm1 = cnts.graphs[1]
        # 断言：检查第二个图形中是否不存在 sigmoid 节点。注意 sigmoid 函数在钩子函数和禁用的模块中都被使用。
        self.assertTrue(
            all(node.target is not torch.sigmoid for node in gm1.graph.nodes)
        )

    # 定义测试方法：允许在图形中使用自定义函数
    def test_allow_in_graph(self):
        # 创建一个编译计数器
        cnts = torch._dynamo.testing.CompileCounter()

        # 使用优化装饰器定义函数 fn
        @torch._dynamo.optimize(cnts)
        def fn(a):
            x = torch.add(a, 1)
            x = torch.add(x, 1)
            x = my_custom_function(x)
            x = torch.add(x, 1)
            x = torch.add(x, 1)
            return x

        # 允许在图形中使用自定义函数 my_custom_function
        torch._dynamo.allow_in_graph(my_custom_function)
        # 调用函数 fn，传入参数为 10 个随机数
        fn(torch.randn(10))
        # 禁止在图形中使用自定义函数 my_custom_function
        torch._dynamo.disallow_in_graph(my_custom_function)

        # 断言：检查帧计数是否为 1，操作计数是否为 5
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 5)

    # 定义测试方法：不正确的使用禁止在图形中使用函数
    def test_incorrect_usage_disallow_in_graph(self):
        # 使用断言检查是否抛出 IncorrectUsage 异常
        with self.assertRaises(IncorrectUsage):

            # 尝试使用禁止在图形中使用函数装饰器
            @torch._dynamo.disallow_in_graph
            def fn1(x):
                return x.cos()

    # 定义测试方法：测试图形中断
    def test_graph_break(self):
        # 创建一个编译计数器
        cnts = torch._dynamo.testing.CompileCounter()

        # 使用优化装饰器定义函数 fn
        @torch._dynamo.optimize(cnts)
        def fn(x):
            x = torch.cos(x)
            x = torch.cos(x)
            # 手动断开当前图形
            torch._dynamo.graph_break()
            x = torch.cos(x)
            x = torch.cos(x)
            # 再次手动断开当前图形
            torch._dynamo.graph_break()
            x = torch.cos(x)
            x = torch.cos(x)
            return x

        # 调用函数 fn，传入参数为大小为 (4, 5) 的随机张量
        fn(torch.randn(4, 5))
        # 断言：检查帧计数是否为 3，操作计数是否为 6
        self.assertEqual(cnts.frame_count, 3)
        self.assertEqual(cnts.op_count, 6)
    def test_skip(self):
        # 定义内部函数 fn2，计算输入张量 x 的正弦值
        def fn2(x):
            return x.sin()

        # 使用装饰器禁用递归优化，函数 fn1 对输入张量 x 执行 sigmoid 操作，然后调用 fn2 计算结果的余弦值
        @torch._dynamo.disable(recursive=False)
        def fn1(x):
            x = x.sigmoid()
            return fn2(x.cos())

        # 定义函数 fn，对输入张量 x 执行正切操作，并调用 fn1 处理结果
        def fn(x):
            return fn1(x.tan())

        # 创建编译计数器实例
        cnts = torch._dynamo.testing.CompileCounter()
        # 使用编译计数器优化函数 fn，然后调用优化后的函数并传入随机生成的张量
        opt_fn = torch._dynamo.optimize(cnts)(fn)
        opt_fn(torch.randn(4))
        # 断言编译帧数为 2
        self.assertEqual(cnts.frame_count, 2)

    @patch.object(torch._dynamo.config, "suppress_errors", True)
    def test_nested_disable_decorator(self):
        # 创建编译计数器实例
        cnts = torch._dynamo.testing.CompileCounter()

        # 使用装饰器禁用动态图优化，函数 fn1 计算输入张量 x 的正弦值并乘以 10
        @torch._dynamo.disable()
        def fn1(x):
            return torch.sin(x) * 10

        # 使用编译计数器优化函数 fn2，对输入张量 x 执行多个加法操作，并调用 fn1 处理结果
        @torch._dynamo.optimize(cnts)
        def fn2(x):
            x = x + 1
            x = x + 1
            x = fn1(x)  # 图结构断开点
            x = x + 1
            x = x + 1
            return x

        # 使用编译计数器优化函数 fn3，调用函数 fn2 处理输入张量 x
        @torch._dynamo.optimize(cnts, nopython=True)
        def fn3(x):
            return fn2(x)

        # 调用 fn2 处理随机生成的 4x5 张量，并断言编译帧数为 2，操作数为 4
        fn2(torch.randn(4, 5))
        self.assertEqual(cnts.frame_count, 2)
        self.assertEqual(cnts.op_count, 4)

        try:
            # 尝试调用 fn3 处理随机生成的 4x5 张量，期望抛出不支持的异常
            fn3(torch.randn(4, 5))
            self.assertFalse(True)
        except torch._dynamo.exc.Unsupported as e:
            # 断言异常消息中包含指定字符串
            self.assertIn("call torch._dynamo.disable() wrapped function", str(e))

    def test_disable_optimize(self):
        # 创建编译计数器实例
        cnt = torch._dynamo.testing.CompileCounter()

        # 使用装饰器禁用优化，函数 f1 对输入张量 x 执行加法操作
        @torch._dynamo.optimize(cnt, disable=True)
        def f1(x):
            return x + 1

        # 调用 f1 处理全为 1 的张量，并断言编译帧数为 0
        f1(torch.ones(6))
        self.assertEqual(cnt.frame_count, 0)

        # 使用装饰器禁用优化，函数 f2 对输入张量 x 执行加法操作
        @torch._dynamo.optimize(cnt, disable=True)
        def f2(x):
            return x + 1

        # 调用 f2 处理全为 1 的张量，并断言编译帧数为 0
        f2(torch.ones(6))
        self.assertEqual(cnt.frame_count, 0)

        # 使用环境变量禁用优化
        with patch.dict(os.environ, {"TORCHDYNAMO_DISABLE": "1"}):

            # 使用编译计数器优化函数 f3，对输入张量 x 执行加法操作
            @torch._dynamo.optimize(cnt)
            def f3(x):
                return x + 1

            # 调用 f3 处理全为 1 的张量，并断言编译帧数为 0
            f3(torch.ones(6))
        self.assertEqual(cnt.frame_count, 0)

    def test_torch_guards_stack_frame_register_inlining_disable(self):
        # 创建张量 x
        x = torch.tensor([0.5, 0.5])

        # 定义继承自 torch.nn.Module 的编码器类
        class encoder(torch.nn.Module):
            def __init__(self, y):
                super().__init__()
                self.a = y

            # 使用装饰器禁用优化，helper 方法计算输入张量 x 和参数 y 的乘积
            @torch._dynamo.disable
            def helper(self, x, y):
                return x * y

            # 前向传播方法，计算输入张量 a 的两倍，调用 helper 方法处理结果
            def forward(self, a, *args):
                x = a + a
                return self.helper(x, self.a)

        # 创建 encoder 类的实例 e，参数为 2.0
        e = encoder(2.0)

        seen_frames = []
        import contextlib

        # 定义全局上下文捕获函数
        @contextlib.contextmanager
        def global_context_capture_fn(frame_summary):
            if frame_summary is not None:
                seen_frames.append(frame_summary)
            yield

        # 使用 mock.patch 捕获 torch._guards.TracingContext.current_frame 的调用
        with mock.patch(
            "torch._guards.TracingContext.current_frame",
            side_effect=global_context_capture_fn,
        ):
            # 使用编译计数器优化 eager 模式下的编码器实例 e 处理张量 x
            torch._dynamo.optimize("eager")(e)(x)

        # 断言未捕获任何帧信息
        self.assertEqual(len(seen_frames), 0)
    # 定义一个测试方法，用于验证 Torch 的堆栈帧、注册、内联部分禁用功能
    def test_torch_guards_stack_frame_register_inlining_partially_disable(self):
        # 创建一个 torch 参数对象，包含两个元素的张量 [0.25, 0.25]
        y = torch.nn.Parameter(torch.tensor([0.25, 0.25]))
        # 创建一个张量 [0.5, 0.5]
        x = torch.tensor([0.5, 0.5])

        # 定义一个名为 encoder 的内部类，继承自 torch.nn.Module
        class encoder(torch.nn.Module):
            # 构造方法，初始化时接收参数 y
            def __init__(self, y):
                super().__init__()
                # 注册一个名为 "param" 的参数，值为传入的 y
                self.register_parameter("param", y)

            # 被 torch._dynamo.disable 装饰的方法，对输入的 x 和 y 进行 sin 和 cos 运算
            @torch._dynamo.disable
            def helper_disabled(self, x, y):
                return x.sin() * y.cos()

            # 辅助方法，对输入的 x 和 y 进行乘法运算
            def helper(self, x, y):
                return x * y

            # 前向传播方法，接收参数 a 和可变长度的 args
            def forward(self, a, *args):
                # 计算输入 a 的两倍
                x = a + a
                # 返回 helper 方法对 x 和 self.param 的运算结果，以及 helper_disabled 方法对 x 和 self.param 的运算结果
                return self.helper(x, self.param) + self.helper_disabled(x, self.param)

        # 创建 encoder 类的实例 e，传入参数 y
        e = encoder(y)

        # 创建一个 CompileCounter 对象 cnt
        cnt = torch._dynamo.testing.CompileCounter()
        # 编译 encoder 实例 e，使用 cnt 计数
        torch.compile(e, backend=cnt)(x)

        # 断言编译的堆栈帧数量为 2
        self.assertEqual(cnt.frame_count, 2)
        # 断言操作的数量为 3
        self.assertEqual(cnt.op_count, 3)

    # 定义一个名为 _test_mark_static_address 的方法，接收一个 guarded 参数
    def _test_mark_static_address(self, guarded):
        # 初始化两个计数器
        compiles_with_buffers = 0
        compiles = 0

        # 定义一个 debug_compiler 函数，用于调试编译过程
        def debug_compiler(gm, _):
            nonlocal compiles_with_buffers
            nonlocal compiles
            # 如果 gm 包含缓冲区，则增加 compiles_with_buffers 计数
            compiles_with_buffers += len(gm._buffers) > 0
            # 每调用一次增加 compiles 计数
            compiles += 1
            return gm

        # 使用 torch._dynamo.optimize 装饰的函数 fn，调用 debug_compiler 函数进行优化
        @torch._dynamo.optimize(backend=debug_compiler)
        def fn(x):
            return x + 1

        # 创建一个全为 1 的张量 inp
        inp = torch.ones(2)

        # 标记 inp 为静态地址，如果 guarded 为 True 则进行保护
        torch._dynamo.mark_static_address(inp, guard=guarded)

        # 调用 fn 函数，传入 inp 作为参数
        fn(inp)
        # 断言 compiles_with_buffers 应为 1
        self.assertEqual(compiles_with_buffers, 1)

        # 创建另一个全为 1 的张量 inp2
        inp2 = torch.ones(2)

        # 如果 guarded 为 True，应触发重新编译
        # 因为它未标记为静态，compiles_with_buffers 不应再增加
        fn(inp2)
        # 断言 compiles_with_buffers 仍为 1
        self.assertEqual(compiles_with_buffers, 1)
        # 断言 compiles 应为 2（如果 guarded 为 True），否则为 1
        self.assertEqual(compiles, 2 if guarded else 1)

    # 定义一个测试方法，用于测试 mark_static_address 方法中 guarded 参数为 True 的情况
    def test_mark_static_address_guarded(self):
        self._test_mark_static_address(guarded=True)

    # 定义一个测试方法，用于测试 mark_static_address 方法中 guarded 参数为 False 的情况
    def test_mark_static_address_unguarded(self):
        self._test_mark_static_address(guarded=False)
    def test_assume_constant_result_on_user_defined_fn(self):
        # 定义一个假设用户定义函数结果常量的装饰器
        @torch._dynamo.assume_constant_result
        def const_fn(n, s):
            return torch.full([n], s)

        # 定义一个函数，参数为 B
        def fn(B):
            # 调用常量函数 const_fn，生成一个由 13 填充的大小为 B.size(0) 的张量 B
            B = const_fn(B.size(0), 13)
            # 将张量 B 中的每个元素乘以 2，得到结果张量 X
            X = B * 2
            # 将结果张量 X 转换为 Python 列表并返回
            return X.tolist()

        # 创建一个包含 32 个元素的整数列表 B_list，每个元素为 8
        B_list = [8] * 32

        # 使用 B_list 创建一个 PyTorch 整数张量 B，数据类型为 torch.int32
        B = torch.tensor(B_list, dtype=torch.int32)
        # 标记张量 B 为静态
        torch._dynamo.decorators.mark_static(B, 0)

        # 设置配置使得标量输出捕获为真
        torch._dynamo.config.capture_scalar_outputs = True
        # 设置配置使得动态输出形状操作捕获为真
        torch._dynamo.config.capture_dynamic_output_shape_ops = True

        # 断言调用 fn(B) 和使用 eager 模式和完整图形以及动态模式编译 fn(B) 的结果相等
        self.assertEqual(
            fn(B), torch.compile(fn, backend="eager", fullgraph=True, dynamic=True)(B)
        )
    def test_assume_constant_result_on_computation_with_graph_input(self):
        # 定义一个装饰器函数，用于标记被装饰函数在计算时结果可假定为常量
        @torch._dynamo.assume_constant_result
        def check(y):
            # 检查 y 的第一个元素是否等于 1，返回布尔值
            return y[0].item() == 1

        def fn(x, y):
            # 调用 check 函数检查 y 的值
            if check(y):
                # 如果 check 返回 True，返回 x + 2
                return x + 2
            else:
                # 如果 check 返回 False，返回 x + 1
                return x + 1

        y = torch.tensor([1])  # 创建一个张量 y，包含单个值 1
        x = torch.tensor(1)    # 创建一个张量 x，包含值 1

        # 使用 torch.compile 编译 fn 函数，并验证其与 fn(x, y) 的输出是否相等
        self.assertEqual(fn(x, y), torch.compile(fn)(x, y))
if __name__ == "__main__":
    # 检查当前模块是否作为主程序执行
    from torch._dynamo.test_case import run_tests
    # 导入 run_tests 函数，用于运行测试用例

    # 执行测试用例
    run_tests()
```