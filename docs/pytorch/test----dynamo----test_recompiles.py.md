# `.\pytorch\test\dynamo\test_recompiles.py`

```
`
# 导入 unittest.mock 的 patch 函数，用于在测试中替换对象
from unittest.mock import patch

import torch

import torch._dynamo.test_case  # 导入 dynamo 测试框架的测试用例基类
import torch._dynamo.testing  # 导入 dynamo 测试工具库


# 定义 RecompileTests 类，继承自 torch._dynamo.test_case.TestCase，作为测试类
class RecompileTests(torch._dynamo.test_case.TestCase):
    # 定义测试方法，测试自动动态减少编译次数
    def test_automatic_dynamic_reduce_recompiles(self):
        # 定义一个函数 foo，接受两个参数 x 和 y，返回它们的乘积
        def foo(x, y):
            return x * y

        # 定义一个函数 run_foo_6_times_and_count_recompiles，运行 foo 函数六次，并计数编译次数
        def run_foo_6_times_and_count_recompiles(dynamic=None):
            cnt = torch._dynamo.testing.CompileCounter()  # 创建 CompileCounter 对象，用于计数编译次数

            x = torch.randn([2])  # 创建一个形状为 [2] 的随机张量 x
            y = torch.randn([2])  # 创建一个形状为 [2] 的随机张量 y
            opt = torch._dynamo.optimize(cnt, dynamic=dynamic)(foo)  # 使用 dynamo 优化 foo 函数，传入 CompileCounter 对象和 dynamic 参数
            opt(x, y)  # 调用优化后的函数 opt，传入 x 和 y 张量
            x = torch.randn([3])  # 创建一个形状为 [3] 的随机张量 x
            y = torch.randn([3])  # 创建一个形状为 [3] 的随机张量 y
            opt(x, y)  # 调用优化后的函数 opt，传入 x 和 y 张量
            x = torch.randn([4])  # 创建一个形状为 [4] 的随机张量 x
            y = torch.randn([4])  # 创建一个形状为 [4] 的随机张量 y
            opt(x, y)  # 调用优化后的函数 opt，传入 x 和 y 张量
            opt(x, y)  # 再次调用优化后的函数 opt，传入 x 和 y 张量
            x = torch.randn([5])  # 创建一个形状为 [5] 的随机张量 x
            y = torch.randn([5])  # 创建一个形状为 [5] 的随机张量 y
            opt(x, y)  # 调用优化后的函数 opt，传入 x 和 y 张量
            opt(x, y)  # 再次调用优化后的函数 opt，传入 x 和 y 张量
            x = torch.randn([6])  # 创建一个形状为 [6] 的随机张量 x
            y = torch.randn([6])  # 创建一个形状为 [6] 的随机张量 y
            opt(x, y)  # 调用优化后的函数 opt，传入 x 和 y 张量

            return cnt  # 返回计数器对象 cnt

        # 使用 patch 装饰器，设置 torch._dynamo.config.automatic_dynamic_shapes 为 False，并 assume_static_by_default 为 True
        @patch.object(torch._dynamo.config, "automatic_dynamic_shapes", False)
        @patch.object(torch._dynamo.config, "assume_static_by_default", True)
        def run_without_automatic():
            return run_foo_6_times_and_count_recompiles()  # 调用 run_foo_6_times_and_count_recompiles 函数，返回编译计数器

        # 使用 patch 装饰器，设置 torch._dynamo.config.automatic_dynamic_shapes 为 True，并 assume_static_by_default 为 True
        @patch.object(torch._dynamo.config, "automatic_dynamic_shapes", True)
        @patch.object(torch._dynamo.config, "assume_static_by_default", True)
        def run_with_automatic():
            return run_foo_6_times_and_count_recompiles()  # 调用 run_foo_6_times_and_count_recompiles 函数，返回编译计数器

        without = run_without_automatic()  # 执行 run_without_automatic 函数，得到计数器对象 without
        self.assertEqual(without.frame_count, 5)  # 断言 without.frame_count 等于 5
        self.assertEqual(without.op_count, 5)  # 断言 without.op_count 等于 5
        torch._dynamo.reset()  # 重置 dynamo 的状态
        without = run_foo_6_times_and_count_recompiles(dynamic=False)  # 执行 run_foo_6_times_and_count_recompiles 函数，dynamic 设置为 False，得到计数器对象 without
        self.assertEqual(without.frame_count, 5)  # 断言 without.frame_count 等于 5
        self.assertEqual(without.op_count, 5)  # 断言 without.op_count 等于 5
        torch._dynamo.reset()  # 重置 dynamo 的状态
        with_automatic = run_with_automatic()  # 执行 run_with_automatic 函数，得到计数器对象 with_automatic
        self.assertEqual(with_automatic.frame_count, 2)  # 断言 with_automatic.frame_count 等于 2
        self.assertEqual(with_automatic.op_count, 2)  # 断言 with_automatic.op_count 等于 2
        torch._dynamo.reset()  # 重置 dynamo 的状态
        with_automatic = run_foo_6_times_and_count_recompiles(dynamic=None)  # 执行 run_foo_6_times_and_count_recompiles 函数，dynamic 设置为 None，得到计数器对象 with_automatic
        self.assertEqual(with_automatic.frame_count, 2)  # 断言 with_automatic.frame_count 等于 2
        self.assertEqual(with_automatic.op_count, 2)  # 断言 with_automatic.op_count 等于 2
        torch._dynamo.reset()  # 重置 dynamo 的状态
        with_dynamic = run_foo_6_times_and_count_recompiles(dynamic=True)  # 执行 run_foo_6_times_and_count_recompiles 函数，dynamic 设置为 True，得到计数器对象 with_dynamic
        self.assertEqual(with_dynamic.frame_count, 1)  # 断言 with_dynamic.frame_count 等于 1
        self.assertEqual(with_dynamic.op_count, 1)  # 断言 with_dynamic.op_count 等于 1

    # 使用 patch 装饰器，设置 torch._dynamo.config.assume_static_by_default 为 True
    @patch.object(torch._dynamo.config, "assume_static_by_default", True)
    def test_recompiles_true_false_flop(self):
        # 定义测试函数，测试条件下重新编译的情况

        # 定义内部函数 foo，根据参数 x 和 y 返回不同的计算结果
        def foo(x, y):
            if x:
                return y * 2
            else:
                return y * y

        # 定义函数，运行 foo 函数六次并计算重新编译次数
        def run_foo_6_times_and_count_recompiles():
            # 创建编译计数器对象
            cnt = torch._dynamo.testing.CompileCounter()

            # 使用编译优化器处理 foo 函数
            opt = torch._dynamo.optimize(cnt, nopython=True)(foo)

            # 第一次调用 foo 函数
            x = True
            y = torch.randn([2])
            opt(x, y)
            # 第二次调用 foo 函数
            x = False
            y = torch.randn([2])
            opt(x, y)
            # 第三次调用 foo 函数
            x = True
            y = torch.randn([3])
            opt(x, y)
            # 第四次调用 foo 函数
            x = True
            y = torch.randn([4])
            opt(x, y)
            # 第五次调用 foo 函数
            x = True
            y = torch.randn([5])
            opt(x, y)

            return cnt  # 返回编译计数器对象

        # 使用 patch.object 装饰器设置自动动态形状为 False，静态假设为 True 的配置，并运行测试函数
        @patch.object(torch._dynamo.config, "automatic_dynamic_shapes", False)
        @patch.object(torch._dynamo.config, "assume_static_by_default", True)
        def run_without_automatic():
            return run_foo_6_times_and_count_recompiles()

        # 使用 patch.object 装饰器设置自动动态形状为 True，静态假设为 True 的配置，并运行测试函数
        @patch.object(torch._dynamo.config, "automatic_dynamic_shapes", True)
        @patch.object(torch._dynamo.config, "assume_static_by_default", True)
        def run_with_automatic():
            return run_foo_6_times_and_count_recompiles()

        # 运行不使用自动形状的测试并断言结果
        without = run_without_automatic()
        self.assertEqual(without.frame_count, 5)
        self.assertEqual(without.op_count, 5)

        # 重置动态模块状态
        torch._dynamo.reset()

        # 运行使用自动形状的测试并断言结果
        with_automatic = run_with_automatic()
        self.assertEqual(with_automatic.frame_count, 3)
        self.assertEqual(with_automatic.op_count, 3)
    # 定义一个测试方法，用于测试自动动态张量标量变化的情况
    def test_automatic_dynamic_tensor_scalar_change(self):
        # 定义一个简单的函数 foo，计算输入 x 和 y 的乘积
        def foo(x, y):
            return x * y

        # 定义一个函数，运行 foo 函数多次并计数重新编译和类型交换的次数
        def run_foo_6_times_and_count_recompiles_swap_types():
            # 创建一个编译计数器
            cnt = torch._dynamo.testing.CompileCounter()

            # 初始化输入张量 x 和 y，使用动态张量优化 foo 函数
            x = torch.randn([2])
            y = torch.randn([2])
            opt = torch._dynamo.optimize(cnt)(foo)
            opt(x, y)

            # 更改输入张量 x 和标量 y 的大小并重新调用 opt 函数
            x = torch.randn([3])
            y = 3
            opt(x, y)

            # 再次更改输入张量 x 和 y，并多次调用 opt 函数
            x = torch.randn([4])
            y = torch.randn([4])
            opt(x, y)
            opt(x, y)

            # 再次更改输入张量 x 和标量 y 的大小并重新调用 opt 函数
            x = torch.randn([5])
            y = 4
            opt(x, y)
            opt(x, y)

            # 最后一次更改输入张量 x 和 y，并重新调用 opt 函数
            x = torch.randn([6])
            y = torch.randn([6])
            opt(x, y)

            # 返回编译计数器
            return cnt

        # 使用 patch.object 装饰器，关闭自动动态形状优化并运行测试函数
        @patch.object(torch._dynamo.config, "automatic_dynamic_shapes", False)
        @patch.object(torch._dynamo.config, "assume_static_by_default", True)
        def run_without_automatic():
            return run_foo_6_times_and_count_recompiles_swap_types()

        # 使用 patch.object 装饰器，开启自动动态形状优化并运行测试函数
        @patch.object(torch._dynamo.config, "automatic_dynamic_shapes", True)
        @patch.object(torch._dynamo.config, "assume_static_by_default", True)
        def run_with_automatic():
            return run_foo_6_times_and_count_recompiles_swap_types()

        # 分别运行关闭和开启自动动态形状优化的测试，并断言结果
        without = run_without_automatic()
        self.assertEqual(without.frame_count, 5)
        self.assertEqual(without.op_count, 5)

        # 重置动态系统状态，并再次运行开启自动动态形状优化的测试，并断言结果
        torch._dynamo.reset()
        with_automatic = run_with_automatic()
        self.assertEqual(with_automatic.frame_count, 3)
        self.assertEqual(with_automatic.op_count, 3)
    def`
    def test_aliasing_guard_failures(self):
        # 定义测试函数，名称为 test_aliasing_guard_failures
        def foo(a, b, c):
            # 定义一个函数 foo，接受三个参数 a, b, c，并执行 a.add_(b) 和返回 c + 1
            a.add_(b)
            return c + 1

        # 创建一个 CompileCounter 实例，跟踪编译次数
        cnt = torch._dynamo.testing.CompileCounter()
        # 使用 torch._dynamo.optimize 对 foo 函数进行优化，指定 nopython=True 禁用 JIT 编译器的动态类型检查
        compiled_foo = torch._dynamo.optimize(cnt, nopython=True)(foo)

        # 创建三个张量 x, y, z，值随机初始化
        x = torch.randn([3])
        y = torch.randn([3])
        z = torch.randn([3])
        # 调用编译后的函数，并传入 x, y, z，得到结果 cmp_result
        cmp_result = compiled_foo(
            x.clone().detach(), y.clone().detach(), z.clone().detach()
        )
        # 调用原始函数 foo，得到结果 eager_result
        eager_result = foo(x.clone().detach(), y.clone().detach(), z.clone().detach())
        # 验证编译结果与原始结果相等
        self.assertEqual(cmp_result, eager_result)
        # 验证编译次数为 1
        self.assertEqual(cnt.frame_count, 1)

        # 调用编译后的函数，并传入 z, y, x，得到结果 cmp_result
        cmp_result = compiled_foo(
            z.clone().detach(), y.clone().detach(), x.clone().detach()
        )
        # 调用原始函数 foo，传入 z, y, x，得到结果 eager_result
        eager_result = foo(z.clone().detach(), y.clone().detach(), x.clone().detach())
        # 验证编译结果与原始结果相等
        self.assertEqual(cmp_result, eager_result)
        # 验证编译次数依旧为 1，因为没有重新编译
        self.assertEqual(cnt.frame_count, 1)

        # 克隆 x 为 x_clone，并调用编译后的函数，传入 x_clone, y, x_clone，得到结果 cmp_result
        x_clone = x.clone().detach()
        cmp_result = compiled_foo(x_clone, y.clone().detach(), x_clone)
        # 克隆 x 为 x_clone，重新调用编译后的函数，传入 x_clone, y, x_clone，得到结果 eager_result
        x_clone = x.clone().detach()
        eager_result = compiled_foo(x_clone, y.clone().detach(), x_clone)
        # 验证编译结果与原始结果相等
        self.assertEqual(cmp_result, eager_result)
        # 验证编译次数增加为 2，因为 alias 改变，导致重新编译
        self.assertEqual(cnt.frame_count, 2)

    def test_aliasing_guard_failures_with_globals(self):
        # 创建两个全局张量 g1 和 g2，值随机初始化
        g1 = torch.randn([3])
        g2 = torch.randn([3])

        def foo(a):
            # 定义函数 foo，接受一个参数 a，执行 a.add_(g1) 和返回 g2 + 1
            a.add_(g1)
            return g2 + 1

        # 创建一个 CompileCounter 实例，跟踪编译次数
        cnt = torch._dynamo.testing.CompileCounter()
        # 使用 torch._dynamo.optimize 对 foo 函数进行优化，指定 nopython=True 禁用 JIT 编译器的动态类型检查
        compiled_foo = torch._dynamo.optimize(cnt, nopython=True)(foo)

        # 创建一个张量 z，值随机初始化
        z = torch.randn([3])
        # 调用编译后的函数，并传入 z，得到结果 cmp_result
        cmp_result = compiled_foo(z.clone().detach())
        # 调用原始函数 foo，传入 z，得到结果 eager_result
        eager_result = foo(z.clone().detach())
        # 验证编译结果与原始结果相等
        self.assertEqual(cmp_result, eager_result)
        # 验证编译次数为 1
        self.assertEqual(cnt.frame_count, 1)

        # 克隆 g1 为 g1，调用编译后的函数，传入 g1，得到结果 cmp_result
        g1 = g1.clone().detach()
        cmp_result = compiled_foo(g1)
        # 克隆 g1 为 g1，重新调用编译后的函数，传入 g1，得到结果 eager_result
        g1 = g1.clone().detach()
        eager_result = compiled_foo(g1)
        # 验证编译结果与原始结果相等
        self.assertEqual(cmp_result, eager_result)
        # 验证编译次数增加为 2，因为 alias 改变，导致重新编译
        self.assertEqual(cnt.frame_count, 2)

    def test_simple_module_recompile(self):
        # 定义一个简单的 Dropout 模块，包含 Dropout 层和 Linear 层
        class SimpleDropout(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.dropout = torch.nn.Dropout(0.5)
                self.linear = torch.nn.Linear(10, 1)

            def forward(self, x):
                # 定义前向传播，先经过 Linear 层，然后经过 Dropout 层
                return self.dropout(self.linear(x))

        # 实例化 SimpleDropout 模型
        model = SimpleDropout()
        # 创建一个随机初始化的输入张量 x
        x = torch.randn(10)
        # 创建一个 CompileCounter 实例，跟踪编译次数
        counter = torch._dynamo.testing.CompileCounter()
        # 使用 torch.compile 编译模型，指定 backend 为 counter，fullgraph=True 表示使用完整图
        model = torch.compile(model, backend=counter, fullgraph=True)
        # 进行 20 次前向传播测试
        for _ in range(20):
            model.eval()  # 设置模型为评估模式
            model(x)      # 进行一次前向传播
            model.train()  # 设置模型为训练模式
            model(x)      # 进行一次前向传播
        # 验证编译次数为 2
        self.assertEqual(counter.frame_count, 2)
# 如果当前脚本被直接执行（而不是被导入到其他脚本中执行），则执行以下代码块
if __name__ == "__main__":
    # 从torch._dynamo.test_case模块中导入run_tests函数
    from torch._dynamo.test_case import run_tests
    
    # 运行导入的run_tests函数，用于执行测试用例
    run_tests()
```