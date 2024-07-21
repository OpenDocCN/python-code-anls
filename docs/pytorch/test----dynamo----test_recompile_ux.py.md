# `.\pytorch\test\dynamo\test_recompile_ux.py`

```py
# Owner(s): ["module: dynamo"]
# 引入单元测试模块
import unittest
# 引入弱引用模块
import weakref

# 引入 PyTorch 主模块
import torch

# 引入 PyTorch 内部 dynamo 模块及其子模块
import torch._dynamo
import torch._dynamo.config
import torch._dynamo.test_case
import torch._dynamo.testing

# 引入 PyTorch 内部 logging 模块
import torch._logging
# 从 PyTorch 内部测试工具中引入特定函数
from torch.testing._internal.logging_utils import kwargs_to_settings, log_settings

# 定义一个测试类 RecompileUxTests，继承自 torch._dynamo.test_case.TestCase
class RecompileUxTests(torch._dynamo.test_case.TestCase):
    # TODO(whc) dynamo actually recompiles one more time than the cache limit
    # 定义一个类变量 cache_limit，设置为 1
    cache_limit = 1

    @classmethod
    def setUpClass(cls):
        # 调用父类的 setUpClass 方法
        super().setUpClass()
        # 在上下文中设置缓存大小限制为 cls.cache_limit
        cls._exit_stack.enter_context(
            torch._dynamo.config.patch("cache_size_limit", cls.cache_limit)
        )

    # 定义一个测试方法 test_drop_cache_on_skip
    def test_drop_cache_on_skip(self):
        # 定义一个简单的模型函数 model
        def model(x, i):
            return x + i

        attached = False
        triggered = False

        # 定义一个触发函数 trigger
        def trigger():
            nonlocal triggered
            triggered = True

        # 定义一个编译器函数 compiler
        def compiler(gm, input):
            nonlocal attached
            f = gm.forward
            assert not attached
            # 注意：将此处的 weakref.ref 用于引用不再及时进行垃圾回收的循环
            weakref.finalize(f, trigger)
            attached = True
            return f

        x = torch.randn(2)
        # 进行两次优化过程
        for i in range(2):
            # 使用 dynamo 中的 optimize 函数优化模型
            opt_model = torch._dynamo.optimize(compiler)(model)
            # 调用优化后的模型
            opt_model(x, i)

        # 断言触发函数已被调用
        self.assertTrue(triggered)

    # 定义一个测试方法 test_loop_torture
    def test_loop_torture(self):
        # 定义一个循环函数 loop_torture
        def loop_torture(input, iters):
            out = input
            # 进行指定次数的循环操作
            for _ in range(iters):
                out += input
            return out

        # 创建一个编译计数器对象 compile_counter
        compile_counter = torch._dynamo.testing.CompileCounter()
        # 进行 10 次循环测试
        for _ in range(10):
            x = torch.randn(3)
            # 随机生成循环次数 iters
            iters = torch.randint(low=0, high=1000, size=())
            # 使用 compile_counter 对 loop_torture 函数进行优化
            opt_loop_torture = torch._dynamo.optimize(compile_counter)(loop_torture)
            opt_loop_torture(x, iters)

        # 断言编译计数器的 frame_count 属性与 self.cache_limit 相等
        self.assertEqual(compile_counter.frame_count, self.cache_limit)

    # 使用 dynamo.config.patch 装饰器，禁用 automatic_dynamic_shapes 功能
    @torch._dynamo.config.patch("automatic_dynamic_shapes", False)
    # 定义一个测试函数，测试动态输入情况下的模型编译和执行
    def test_dynamic_input(self):
        # 定义一个简单的模型函数，将输入加倍返回
        def model(input):
            return input + input

        # 期望重新编译次数为2
        expected_recompiles = 2
        # 创建一个编译计数器对象
        compile_counter = torch._dynamo.testing.CompileCounter()
        
        # 在指定的上下文中，设置缓存大小限制为 expected_recompiles
        with torch._dynamo.config.patch("cache_size_limit", expected_recompiles):
            # 使用 self.assertLogs 检查日志输出
            with self.assertLogs(logger="torch._dynamo", level="WARNING") as logs:
                # 进行10次循环
                for _ in range(10):
                    # 随机生成一个大小在0到1000之间的整数
                    bsz = torch.randint(low=0, high=1000, size=())
                    # 生成一个形状为 (bsz, 3, 4) 的随机张量
                    x = torch.randn((bsz, 3, 4))
                    # 对模型应用优化，使用 compile_counter 计数
                    opt_model = torch._dynamo.optimize(compile_counter)(model)
                    # 使用优化后的模型进行计算
                    opt_model(x)

        # 断言编译计数的帧数等于期望的重新编译次数
        self.assertEqual(compile_counter.frame_count, expected_recompiles)
        # 断言日志记录的数量为1
        self.assertEqual(len(logs.records), 1)
        # 打印第一条日志记录
        print(logs.records[0])
        # 断言日志消息以指定字符串开头
        self.assertTrue(
            logs.records[0]
            .getMessage()
            .startswith("torch._dynamo hit config.cache_size_limit")
        )

    # 如果 CUDA 不可用，则跳过此测试函数
    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_nvfuser_guards(self):
        # 我们可能希望模仿 nvfuser 的 ProfilingExecutor guards
        # 确保 dynamo 负责顶层的所有重新编译，
        # 从而简化底层的 torchscript executor
        def func(a, b, c):
            return a + b * c

        # 在 CUDA 设备上生成随机张量 a, b, c
        a = torch.rand(3, 4, 5, device="cuda")
        b = torch.rand(3, 4, 5, device="cuda")
        b_v = torch.rand(3, 5, 4, device="cuda").view(3, 4, 5)
        b_p = torch.rand(3, 5, 4, device="cuda").permute(0, 2, 1)
        c = torch.rand(3, 4, 5, device="cuda")
        # 创建一个编译计数器对象
        compile_counter = torch._dynamo.testing.CompileCounter()

        # 在指定的上下文中，设置缓存大小限制为2
        with torch._dynamo.config.patch("cache_size_limit", 2):
            # 对函数 func 应用优化，使用 compile_counter 计数
            opt_func = torch._dynamo.optimize(compile_counter)(func)
            # 对函数进行预热
            opt_func(a, b, c)  # warmup
            # 断言编译计数的帧数为1
            self.assertEqual(compile_counter.frame_count, 1)

            # 再次调用函数，不应触发守护失败或重新编译
            opt_func(a, b, c)  # no guard fail or recompile
            # 断言编译计数的帧数为1
            self.assertEqual(compile_counter.frame_count, 1)

            # 使用视图不应触发 nvfuser 重新编译
            opt_func(a, b_v, c)  # a view should not cause nvfuser recompile
            # 断言编译计数的帧数为1
            self.assertEqual(compile_counter.frame_count, 1)

            # 使用排列操作应触发重新编译
            opt_func(a, b_p, c)  # a permutation should cause recompile
            # 断言编译计数的帧数为2
            self.assertEqual(compile_counter.frame_count, 2)

    # 断言日志记录中包含指定字符串
    def assert_single_log_contains(self, logs, contains_str):
        # 断言日志记录的数量为1
        self.assertEqual(len(logs.records), 1)
        # 断言日志消息中包含指定的字符串
        self.assertTrue(
            logs.records[0].getMessage().find(contains_str) > 0,
            msg=f'Expected to find "{contains_str}" in log "{logs.records[0].getMessage()}"',
        )
    def test_verbose_tensor_check(self):
        def func(a):
            # 定义一个函数，用于执行 torch.add 操作，将输入张量 a 和标量 4 相加
            # 注意：选择一个在 C++ 中完全实现的函数，避免混淆 torch._refs 的警告信息
            return torch.add(a, 4)

        def cache_fail_test(cached_input, missed_input, expected_failure):
            # 定义一个测试函数，用于测试缓存和未缓存输入的优化行为
            torch._dynamo.reset()  # 重置动态优化器状态
            torch._dynamo.utils.counters.clear()  # 清空计数器
            opt_func = torch._dynamo.optimize("eager")(func)  # 使用 eager 模式优化 func 函数
            # 预热阶段，运行一次 opt_func
            opt_func(cached_input)

            with self.assertLogs(logger="torch._dynamo", level="WARNING") as logs:
                opt_func = torch._dynamo.optimize("eager")(func)  # 再次使用 eager 模式优化 func 函数
                opt_func(missed_input)  # 使用未缓存的输入运行 opt_func
            self.assert_single_log_contains(logs, expected_failure)  # 断言日志中是否包含预期的警告信息

        a = torch.rand(3, 4, 5)  # 创建一个形状为 (3, 4, 5) 的随机张量 a
        cache_fail_test(
            a,
            a[0:2, :, :],
            "tensor 'L['a']' size mismatch at index 0. expected 3, actual 2",
        )
        cache_fail_test(
            a,
            a.clone().as_strided((3, 4, 5), stride=(1, 3, 12)),
            "tensor 'L['a']' stride mismatch at index 0. expected 20, actual 1",
        )
        cache_fail_test(
            a, a[0, :, :], "tensor 'L['a']' rank mismatch. expected 3, actual 2"
        )
        cache_fail_test(a, a.to("meta"), "tensor 'L['a']' dispatch key set mismatch.")
        cache_fail_test(
            a,
            a.to(torch.float16),
            "tensor 'L['a']' dtype mismatch. expected Float, actual Half",
        )
        a_grad = a.clone()  # 克隆张量 a，用于创建梯度张量
        a_grad.requires_grad = True  # 设置梯度张量的 requires_grad 属性为 True
        cache_fail_test(
            a,
            a_grad,
            "tensor 'L['a']' requires_grad mismatch. expected requires_grad=0",
        )

    def test_mismatched_type(self):
        a = torch.rand(3, 4, 5)  # 创建一个形状为 (3, 4, 5) 的随机张量 a
        b = torch.rand(3, 4, 5)  # 创建一个形状与 a 相同的随机张量 b

        def func(a, b):
            return a + b  # 执行张量 a 和 b 的加法操作

        opt_func = torch._dynamo.optimize("eager")(func)  # 使用 eager 模式优化 func 函数
        # 预热阶段，运行一次 opt_func
        opt_func(a, b)

        with self.assertLogs(logger="torch._dynamo", level="WARNING") as logs:
            opt_func = torch._dynamo.optimize("eager")(func)  # 再次使用 eager 模式优化 func 函数
            opt_func(a, 1)  # 使用类型不匹配的输入运行 opt_func
        self.assert_single_log_contains(
            logs,
            "expected type of 'L['b']' to be a tensor type, ' but found <class 'int'>",
        )

    @torch._dynamo.config.patch("cache_size_limit", 32)
    def test_multiple_guard_fails(self):
        # 初始化一个空列表，用于存储失败的原因
        failure_reasons = []

        # 定义一个用于处理守卫失败的函数，将失败的原因添加到列表中
        def guard_fail_fn(failure):
            failure_reasons.append(failure[0])

        # 定义一个简单的函数 f(x)，使用 PyTorch 的 relu 函数处理输入 x
        def f(x):
            return torch.relu(x)

        # 使用 torch._dynamo.optimize 函数对 f 进行优化，
        # 使用 "eager" 后端，指定 guard_fail_fn 处理守卫失败情况，dynamic=False 表示关闭动态特性
        opt_f = torch._dynamo.optimize(
            backend="eager", guard_fail_fn=guard_fail_fn, dynamic=False
        )(f)

        # 循环5次，测试优化后的函数 opt_f 的行为
        for i in range(5):
            # 每次循环前清空失败原因列表
            failure_reasons.clear()
            # 对 opt_f 输入随机生成的张量（维度为 8+i），记录可能的失败原因
            opt_f(torch.randn(8 + i))

        # 将失败原因列表转换为字符串，每个原因占一行
        failure_str = "\n".join(failure_reasons)
        
        # 遍历字符串中的每一行
        for line in """\
    def test_multiple_guard_fails_report_all(self):
        # 使用 torch._dynamo.config.patch 修改配置项 "cache_size_limit" 为 32
        @torch._dynamo.config.patch("cache_size_limit", 32)
        # 定义测试方法 test_multiple_guard_fails_report_all
        def test_multiple_guard_fails_report_all(self):
            # 设置日志参数，启用重新编译的详细信息
            with log_settings(kwargs_to_settings(recompiles_verbose=True)):
                # 初始化失败原因列表
                failure_reasons = []

                # 定义 guard_fail_fn 函数，将失败原因添加到列表中
                def guard_fail_fn(failure):
                    failure_reasons.append(failure[0])

                # 定义函数 f(x)，返回长度为 x[-1] 的全1张量
                def f(x):
                    return torch.ones(len(x), x[-1])

                # 使用 torch._dynamo.optimize 优化函数 f，设置后端为 "eager"，关闭动态计算
                opt_f = torch._dynamo.optimize(
                    backend="eager", guard_fail_fn=guard_fail_fn, dynamic=False
                )(f)

                # 调用 opt_f 函数，传入参数 [4, 5, 6]
                opt_f([4, 5, 6])

                # 定义过滤失败原因的函数 filter_reasons
                def filter_reasons():
                    return "\n".join(
                        [
                            line
                            for line in "\n".join(failure_reasons).splitlines()
                            if not line.startswith("___check_type_id")
                        ]
                    )

                # 清空失败原因列表
                failure_reasons.clear()

                # 再次调用 opt_f 函数，传入参数 [7, 8]
                opt_f([7, 8])

                # 清空失败原因列表
                failure_reasons.clear()

                # 再次调用 opt_f 函数，传入参数 [9]
                opt_f([9])

                # 遍历每行字符串 "len(L['x']) == 3"，验证其是否在过滤后的失败原因中
                for line in """\
len(L['x']) == 3""".split(
                    "\n"
                ):
                    self.assertIn(line, filter_reasons())

    if __name__ == "__main__":
        # 导入 torch._dynamo.test_case 模块中的 run_tests 函数，并运行测试
        from torch._dynamo.test_case import run_tests

        run_tests()
```