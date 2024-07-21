# `.\pytorch\test\jit\test_generator.py`

```py
# Owner(s): ["oncall: jit"]

# 导入必要的库和模块
import io
import math
import unittest

import torch
from torch.nn import init
from torch.testing._internal.common_utils import skipIfLegacyJitExecutor
from torch.testing._internal.jit_utils import JitTestCase

# 如果直接运行该文件，则抛出运行时错误，提醒使用指定方式运行测试
if __name__ == "__main__":
    raise RuntimeError(
        "This test file is not meant to be run directly, use:\n\n"
        "\tpython test/test_jit.py TESTNAME\n\n"
        "instead."
    )

# 定义一个测试类 TestGenerator，继承自 JitTestCase
class TestGenerator(JitTestCase):
    # 装饰器标记，表明该测试在 legacy JIT executor 下会跳过，且预期会失败
    @skipIfLegacyJitExecutor("legacy JIT executor does not support Generator type")
    @unittest.expectedFailure
    # 定义测试方法 test_trace
    def test_trace(self):
        # 定义函数 f，该函数使用 torch 的随机数生成器 Generator
        def f():
            # 创建一个新的 Generator 对象
            generator = torch.Generator()
            # 使用默认随机种子种子生成器
            generator.seed()
            # 手动设置生成器的种子为 2023
            generator.manual_seed(2023)
            # 获取当前生成器的初始种子
            generator.initial_seed()
            # 创建一个 2x2 的空张量 tensor
            tensor = torch.empty(2, 2)
            # 在 [-1.0, 1.0] 的均匀分布中填充 tensor，使用指定的生成器
            tensor.uniform_(0, 1, generator=generator)
            return tensor

        # 对函数 f 进行追踪，生成一个追踪的函数 traced_f
        traced_f = torch.jit.trace(f, ())

        # 运行该测试 3 次，以确保每次运行追踪函数时生成器都被手动种子化
        for i in range(3):
            # 设置全局的随机种子为 1
            torch.manual_seed(1)
            # 直接调用函数 f，获取 eager_tensor
            eager_tensor = f()

            # 修改默认生成器的种子为 2，以确保我们使用了追踪的生成器
            torch.manual_seed(2)
            # 调用追踪函数 traced_f，获取 traced_tensor
            traced_tensor = traced_f()

            # 使用断言检查 eager_tensor 和 traced_tensor 是否相等
            self.assertEqual(eager_tensor, traced_tensor)

    # 定义测试方法 test_script
    def test_script(self):
        # 定义函数 f，该函数使用 torch 的随机数生成器 Generator
        def f():
            # 创建一个新的 Generator 对象
            generator = torch.Generator()
            # 使用默认随机种子种子生成器
            generator.seed()
            # 手动设置生成器的种子为 2023
            generator.manual_seed(2023)
            # 获取当前生成器的初始种子
            generator.initial_seed()
            # 创建一个 2x2 的空张量 tensor
            tensor = torch.empty(2, 2)
            # 在正态分布 N(-1.0, 1.0) 中填充 tensor，使用指定的生成器
            tensor.normal_(-1.0, 1.0, generator=generator)
            return tensor

        # 对函数 f 进行脚本化，生成一个脚本化的函数 script_f
        script_f = torch.jit.script(f, ())

        # 运行该测试 3 次，以确保每次运行脚本化函数时生成器都被手动种子化
        for i in range(3):
            # 设置全局的随机种子为 1
            torch.manual_seed(1)
            # 直接调用函数 f，获取 eager_tensor
            eager_tensor = f()

            # 修改默认生成器的种子为 2，以确保我们使用了脚本化的生成器
            torch.manual_seed(2)
            # 调用脚本化函数 script_f，获取 script_tensor
            script_tensor = script_f()

            # 使用断言检查 eager_tensor 和 script_tensor 是否相等
            self.assertEqual(eager_tensor, script_tensor)

    # 定义测试方法 test_default_generator
    def test_default_generator(self):
        # 定义函数 f，该函数使用默认的 torch 随机数生成器
        def f():
            # 检查手动设置默认生成器种子的功能
            torch.manual_seed(2023)
            # 创建一个 2x2 的空张量 tensor
            tensor = torch.empty(2, 2)
            # 在正态分布 N(-1.0, 1.0) 中填充 tensor
            tensor.normal_(-1.0, 1.0)
            return tensor

        # 设置全局的随机种子为 1
        torch.manual_seed(1)
        # 直接调用函数 f，获取 eager_tensor
        eager_tensor = f()

        # 修改默认生成器的种子为 2
        torch.manual_seed(2)

        # 对函数 f 进行脚本化，生成一个脚本化的函数 script_f
        script_f = torch.jit.script(f, ())
        # 调用脚本化函数 script_f，获取 script_tensor
        script_tensor = script_f()

        # 使用断言检查 eager_tensor 和 script_tensor 是否相等
        self.assertEqual(eager_tensor, script_tensor)
    # 定义一个测试生成器参数的测试方法
    def test_generator_arg(self):
        # 定义内部函数 f，接受一个 torch.Generator 对象作为参数
        def f(generator: torch.Generator):
            # 创建一个空的 2x2 的张量
            tensor = torch.empty(2, 2)
            # 使用正态分布填充张量，使用给定的生成器
            tensor.normal_(-1.0, 1.0, generator=generator)
            # 返回填充后的张量
            return tensor

        # 创建一个新的 torch.Generator 对象
        generator = torch.Generator()
        # 设置该生成器的种子为 2023
        generator.manual_seed(2023)

        # 对函数 f 进行脚本化，指定生成器作为输入参数
        script_f = torch.jit.script(f, (generator,))

        # 循环执行以下代码 3 次
        for i in range(3):
            # 创建一个新的 torch.Generator 对象
            generator = torch.Generator()
            # 设置该生成器的种子为 2023 + i
            generator.manual_seed(2023 + i)

            # 设置全局随机种子为 1 + i
            torch.manual_seed(1 + i)

            # 使用给定的生成器生成张量（急切模式）
            eager_tensor = f(generator)

            # 创建一个新的 torch.Generator 对象
            generator = torch.Generator()
            # 设置该生成器的种子为 2023 + i
            generator.manual_seed(2023 + i)

            # 设置全局随机种子为 1 + i
            torch.manual_seed(1 + i)

            # 使用脚本化的函数生成张量
            script_tensor = script_f(generator)

            # 断言急切模式生成的张量与脚本化生成的张量相等
            self.assertEqual(eager_tensor, script_tensor)
    def test_save_load(self):
        # 定义一个名为 Foo 的类，继承自 torch.nn.Module
        class Foo(torch.nn.Module):
            # 类的初始化方法
            def __init__(self):
                super().__init__()
                # 创建一个线性层 foo，输入大小为 2，输出大小为 2，无偏置
                self.foo = torch.nn.Linear(2, 2, bias=False)
                # 创建一个线性层 bar，输入大小为 2，输出大小为 2，无偏置
                self.bar = torch.nn.Linear(2, 2, bias=False)

                # 调用 reset_parameters 方法初始化各层参数
                self.reset_parameters()

            # 重置线性层参数的方法
            def reset_linear(self, module, generator):
                # 使用 kaiming_uniform_ 方法初始化权重
                init.kaiming_uniform_(
                    module.weight, a=math.sqrt(5), generator=generator
                )

            # 重置所有参数的方法
            def reset_parameters(self):
                # 创建一个随机数生成器 generator
                generator = torch.Generator()
                generator.manual_seed(1)
                # 对 foo 层进行参数重置
                self.reset_linear(self.foo, generator)

                # 创建一个新的随机数生成器 generator
                generator = torch.Generator()
                generator.manual_seed(2)
                # 对 bar 层进行参数重置
                self.reset_linear(self.bar, generator)

            # 前向传播方法
            def forward(self, x):
                # 对输入 x 应用 foo 层
                x = self.foo(x)
                # 对输出 x 应用 bar 层
                x = self.bar(x)

                # 创建一个新的随机数生成器 generator
                generator = torch.Generator()
                generator.manual_seed(3)
                # 创建一个和 x 形状相同的空张量 r，并进行标准正态分布初始化
                r = torch.empty_like(x)
                r.normal_(0.0, 1.0, generator=generator)

                # 返回两个张量 x 和 r
                return x, r

        # 创建一个 eager_foo 对象，用于直接执行前向传播等操作
        eager_foo = Foo()

        # 将 Foo 类实例化并转换为 Torch Script
        script_module = torch.jit.script(Foo())
        # 创建一个 BytesIO 对象 saved_module，用于保存模型的字节流表示
        saved_module = io.BytesIO()
        # 将 Torch Script 模型保存到 saved_module 中
        torch.jit.save(script_module, saved_module)
        saved_module.seek(0)

        # 从 saved_module 中加载 Torch Script 模型
        loaded_module = torch.jit.load(saved_module)

        # 使用断言检查两个模型的权重是否相等
        self.assertEqual(eager_foo.foo.weight, loaded_module.foo.weight)
        self.assertEqual(eager_foo.bar.weight, loaded_module.bar.weight)

        try:
            # 运行 3 次以确保每次调用 forward 方法时生成器种子被设置
            for i in range(3):
                # 创建一个全为 1 的张量 x
                x = torch.ones(2, 2)
                # 对 eager_foo 和 loaded_module 分别调用 forward 方法
                out1, r1 = eager_foo(x)
                out2, r2 = loaded_module(x)

                try:
                    # 使用断言检查输出 out1 和 out2 是否相等
                    self.assertEqual(out1, out2)
                except:  # noqa: B001, E722
                    # 输出异常信息以及相关变量的值
                    print(f"Iteration {i}:\n{out1=}\n{out2=}")
                    raise

                try:
                    # 使用断言检查随机张量 r1 和 r2 是否相等
                    self.assertEqual(r1, r2)
                except:  # noqa: B001, E722
                    # 输出异常信息以及相关变量的值
                    print(f"Iteration {i}:\n{r1=}\n{r2=}")
                    raise
        except:  # noqa: B001, E722
            # 输出加载模型的 forward 方法的源代码
            print(loaded_module.forward.code)
            raise
```