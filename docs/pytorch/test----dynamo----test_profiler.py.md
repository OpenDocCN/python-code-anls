# `.\pytorch\test\dynamo\test_profiler.py`

```py
# Owner(s): ["module: dynamo"]
from unittest.mock import patch  # 导入 patch 函数，用于模拟对象的替换

import torch  # 导入 PyTorch 库

import torch._dynamo.test_case  # 导入 Dynamo 测试用例
import torch._dynamo.testing  # 导入 Dynamo 测试模块
import torch._dynamo.utils  # 导入 Dynamo 实用工具函数

from torch._dynamo.utils import dynamo_timed  # 从 Dynamo 实用工具中导入 dynamo_timed 函数

from torch.testing._internal.common_utils import TemporaryFileName  # 导入临时文件名生成工具


class DynamoProfilerTests(torch._dynamo.test_case.TestCase):
    def test_dynamo_timed_profiling_isolated(self):
        # @dynamo_timed 修饰的函数应出现在性能分析结果中。

        @dynamo_timed  # 使用 dynamo_timed 修饰符进行性能分析
        def inner_fn(x):
            return x.sin()

        def outer_fn(x, y):
            return inner_fn(x) * y

        x, y = (torch.rand((2, 2)) for _ in range(2))

        with torch.profiler.profile(with_stack=False) as prof:
            outer_fn(x, y)

        self.assertTrue(
            any("inner_fn (dynamo_timed)" in evt.name for evt in prof.events())
        )  # 断言检查性能分析结果中是否包含 "inner_fn (dynamo_timed)" 字符串

    def test_dynamo_timed_profiling_backend_compile(self):
        # @dynamo_timed 修饰的函数应出现在性能分析结果中。
        # 这个测试用例验证它们是否实际出现在 Dynamo 执行中。
        # "backend_compile" 只是作为一个例子被选择；如果它被重命名了，
        # 这个测试可以被替换或删除。

        fn_name = "call_user_compiler"

        def fn(x, y):
            return x.sin() * y.cos()

        x, y = (torch.rand((2, 2)) for _ in range(2))

        with torch.profiler.profile(with_stack=False) as prof:
            torch._dynamo.optimize("aot_eager")(fn)(x, y)

        self.assertTrue(
            any(f"{fn_name} (dynamo_timed)" in evt.name for evt in prof.events())
        )  # 断言检查性能分析结果中是否包含 "call_user_compiler (dynamo_timed)" 字符串

    @patch.object(torch._dynamo.config, "assume_static_by_default", False)
    def test_profile_dynamic_shapes_runtime(self):
        def fn(x, y, z):
            return x @ y + z

        opt_fn = torch._dynamo.optimize("aot_eager", dynamic=True, nopython=True)(fn)

        inputs = [
            (torch.rand(a, b), torch.rand(b, c), torch.rand(a, c))
            for (a, b, c) in [(15, 16, 17), (15, 15, 16), (16, 16, 16)]
        ]

        opt_fn(*inputs[0])
        opt_fn(*inputs[1])

        with torch.profiler.profile(record_shapes=True):
            opt_fn(*inputs[2])

    @patch.object(torch._dynamo.config, "assume_static_by_default", False)
    def test_profile_dynamic_shapes_compilation(self):
        def fn(x, y, z):
            return x @ y + z

        opt_fn = torch._dynamo.optimize("aot_eager", dynamic=True, nopython=True)(fn)

        inputs = (torch.rand(15, 16), torch.rand(16, 17), torch.rand(15, 17))

        with torch.profiler.profile(record_shapes=True):
            opt_fn(*inputs)

    @patch.object(torch._dynamo.config, "assume_static_by_default", False)
    def test_profile_dynamic_shapes_list_compilation(self):
        def fn(x, y, z):
            # 定义函数，将输入张量x和y沿着0维度拼接，再加上张量z
            return torch.cat([x, y], dim=0) + z

        # 使用动态编译优化函数fn，启用AOT（Ahead of Time）模式和即时编译
        opt_fn = torch._dynamo.optimize("aot_eager", dynamic=True, nopython=True)(fn)

        # 准备输入数据，三个张量，分别为大小为(4, 16)，(12, 16)，(16, 16)
        inputs = (torch.rand(4, 16), torch.rand(12, 16), torch.rand(16, 16))

        # 使用torch.profiler.profile记录张量形状，进入性能分析器
        with torch.profiler.profile(record_shapes=True):
            # 调用优化后的函数opt_fn并传入输入数据
            opt_fn(*inputs)

    def test_execution_trace_dynamic_shapes(self):
        def fn(x, y, z):
            # 定义函数，计算输入张量x与y的矩阵乘积，再加上张量z
            return x @ y + z

        # 创建ExecutionTraceObserver对象et
        et = torch.profiler.ExecutionTraceObserver()
        # 使用动态编译优化函数fn，启用AOT（Ahead of Time）模式
        opt_fn = torch.compile(fn, dynamic=True, backend="aot_eager")
        # 准备输入数据，三个大小为(4, 4)的张量
        inputs = [torch.rand((4, 4)) for _ in range(3)]

        # 使用TemporaryFileName上下文管理器创建临时文件fname
        with TemporaryFileName() as fname:
            # 注册回调函数到ExecutionTraceObserver对象et
            et.register_callback(fname)
            # 开始记录执行轨迹
            et.start()
            # 调用优化后的函数opt_fn并传入输入数据
            out = opt_fn(*inputs)
            # 停止记录执行轨迹
            et.stop()
            # 注销回调函数
            et.unregister_callback()

    def test_profiler_cache_lookup(self):
        def fn(x):
            # 定义函数，计算输入张量x的平方，再加2，然后计算结果的立方
            y = x**2
            y = y + 2
            z = y**3
            return z

        # 遍历两种性能分析器，分别为torch.autograd.profiler.profile和torch.profiler.profiler.profile
        for profiler, get_events in (
            (torch.autograd.profiler.profile, lambda prof: prof.function_events),
            (torch.profiler.profiler.profile, lambda prof: prof.events()),
        ):
            # 创建大小为(2, 2)的随机张量x，并要求梯度计算
            x = torch.randn((2, 2), requires_grad=True)
            # 计算参考值ref，调用原始函数fn
            ref = fn(x)
            # 使用动态编译优化函数opt_fn
            opt_fn = torch.compile(fn, backend="aot_eager")

            # 预热运行一次优化后的函数opt_fn
            opt_fn(x)

            # 使用性能分析器prof记录优化后函数opt_fn的执行
            with profiler() as prof:
                res = opt_fn(x)
            # 获取性能分析事件
            events = list(
                filter(
                    lambda event: "TorchDynamo Cache Lookup" in event.name,
                    get_events(prof),
                )
            )

            # 断言参考值与优化后函数的结果相等
            self.assertEqual(ref, res)
            # 断言预期性能分析事件中有一次"TorchDynamo Cache Lookup"事件
            self.assertTrue(
                len(events) == 1,
                "Expected one lookup profiler event for one opt_fn run",
            )

    def test_profiler_cache_lookup_profiler_step(self):
        def fn(x, y, z):
            # 定义函数，计算输入张量x减去y，再加上z
            return torch.add(torch.sub(x, y), z)

        # 使用动态编译优化函数fn
        opt_fn = torch._dynamo.optimize("aot_eager")(fn)

        # 创建三个大小为(4, 4)的随机张量x, y, z
        (x, y, z,) = (torch.rand(4, 4) for _ in range(3))

        # 创建性能分析器prof，配置执行计划，预热2次，激活2次，重复1次
        prof = torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=2, warmup=2, active=2, repeat=1)
        )

        # 执行10次优化后的函数opt_fn，并在每次迭代后执行性能分析步骤
        for _ in range(10):
            opt_fn(x, y, z)
            prof.step()

        # 断言性能分析事件中存在"TorchDynamo Cache Lookup"事件
        self.assertTrue(
            any(e.name == "TorchDynamo Cache Lookup" for e in prof.events())
        )

    def test_profiler_dynamo_compiled_region(self):
        def fn(x, y, z):
            # 定义函数，计算输入张量x与y的矩阵乘积，再加上张量z
            return x @ y + z

        # 使用动态编译优化函数fn
        opt_fn = torch._dynamo.optimize("eager")(fn)

        # 创建三个大小为(4, 4)的随机张量输入
        inputs = [torch.rand(4, 4) for _ in range(3)]

        # 执行两次优化后的函数opt_fn
        for _ in range(2):
            opt_fn(*inputs)

        # 使用性能分析器prof记录优化后函数opt_fn的执行
        with torch.profiler.profile() as prof:
            opt_fn(*inputs)

        # 断言性能分析事件中存在"Torch-Compiled Region"事件
        self.assertTrue(any(e.name == "Torch-Compiled Region" for e in prof.events()))
# 如果当前脚本作为主程序运行
if __name__ == "__main__":
    # 导入 torch._dynamo.test_case 模块中的 run_tests 函数
    from torch._dynamo.test_case import run_tests

    # 运行测试函数，用于执行相关的测试用例
    run_tests()
```