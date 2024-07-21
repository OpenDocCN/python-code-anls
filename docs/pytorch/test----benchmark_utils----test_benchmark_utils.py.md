# `.\pytorch\test\benchmark_utils\test_benchmark_utils.py`

```
# 导入所需的模块和库
import collections  # 引入集合模块
import json  # 引入 JSON 操作模块
import os  # 引入操作系统接口模块
import re  # 引入正则表达式模块
import textwrap  # 引入文本包装模块
import timeit  # 引入计时模块
import unittest  # 引入单元测试模块
from typing import Any, List, Tuple  # 引入类型提示模块

import expecttest  # 引入 expecttest 库
import numpy as np  # 引入 NumPy 库

import torch  # 引入 PyTorch 库
import torch.utils.benchmark as benchmark_utils  # 引入 PyTorch benchmark 工具
from torch.testing._internal.common_utils import (  # 引入 PyTorch 测试工具
    IS_SANDCASTLE,  # 判断是否为 Sandcastle 环境
    IS_WINDOWS,  # 判断是否为 Windows 环境
    run_tests,  # 运行测试
    slowTest,  # 标记为慢速测试
    TEST_WITH_ASAN,  # 判断是否开启 AddressSanitizer 测试
    TestCase,  # 测试用例基类
)

# 定义生成的 Callgrind 统计文件路径常量
CALLGRIND_ARTIFACTS: str = os.path.join(
    os.path.split(os.path.abspath(__file__))[0], "callgrind_artifacts.json"
)

def generate_callgrind_artifacts() -> None:
    """重新生成 `callgrind_artifacts.json` 文件

    不同于期望测试，重新生成 Callgrind 计数将会产生较大的差异，因为构建目录和
    conda/pip 目录会包含在指令字符串中。由于 Python 的随机性，这个过程也不是
    100% 确定性的，并且运行时间超过一分钟。因此，手动运行这个函数是必要的。
    """
    print("重新生成 callgrind 统计文件.")

    # 使用 benchmark_utils.Timer 收集基准不含数据的 Callgrind 统计信息
    stats_no_data = benchmark_utils.Timer("y = torch.ones(())").collect_callgrind(
        number=1000
    )

    # 使用 benchmark_utils.Timer 收集基准包含数据的 Callgrind 统计信息
    stats_with_data = benchmark_utils.Timer("y = torch.ones((1,))").collect_callgrind(
        number=1000
    )

    user = os.getenv("USER")

    def to_entry(fn_counts):
        return [f"{c} {fn.replace(f'/{user}/', '/test_user/')}" for c, fn in fn_counts]

    # 构建 Callgrind 统计信息的字典
    artifacts = {
        "baseline_inclusive": to_entry(stats_no_data.baseline_inclusive_stats),
        "baseline_exclusive": to_entry(stats_no_data.baseline_exclusive_stats),
        "ones_no_data_inclusive": to_entry(stats_no_data.stmt_inclusive_stats),
        "ones_no_data_exclusive": to_entry(stats_no_data.stmt_exclusive_stats),
        "ones_with_data_inclusive": to_entry(stats_with_data.stmt_inclusive_stats),
        "ones_with_data_exclusive": to_entry(stats_with_data.stmt_exclusive_stats),
    }

    # 将生成的 Callgrind 统计信息写入到文件中
    with open(CALLGRIND_ARTIFACTS, "w") as f:
        json.dump(artifacts, f, indent=4)


def load_callgrind_artifacts() -> (
    Tuple[benchmark_utils.CallgrindStats, benchmark_utils.CallgrindStats]
):
    """加载 Callgrind 统计文件，并进行单元测试

    除了收集统计数据，这个函数还提供了一些用于操作和显示收集到的统计数据的工具。
    多次测量的结果将存储在 callgrind_artifacts.json 文件中。

    虽然 FunctionCounts 和 CallgrindStats 可以被 pickle 序列化，但测试的
    统计数据以原始字符串形式存储，以便更容易地检查，并避免将任何实现细节
    写入到生成的统计文件中。
    """
    # 打开 Callgrind 统计文件并加载内容
    with open(CALLGRIND_ARTIFACTS) as f:
        artifacts = json.load(f)

    # 编译正则表达式用于解析统计数据
    pattern = re.compile(r"^\s*([0-9]+)\s(.+)$")

    def to_function_counts(
        count_strings: List[str], inclusive: bool
        # 将收集到的统计数据转换为 FunctionCounts 对象，并指定是否包含
        # 所有调用的信息

        count_strings: List[str], inclusive: bool
    ) -> List[Tuple[int, str]]:
        return [(int(match.group(1)), match.group(2)) for line in count_strings
                for match in [pattern.match(line)] if match]

    # 解析各类 Callgrind 统计信息并返回包含两种统计数据的元组
    return (
        to_function_counts(artifacts["ones_no_data_inclusive"], inclusive=True),
        to_function_counts(artifacts["ones_no_data_exclusive"], inclusive=False),
    )
    ) -> benchmark_utils.FunctionCounts:
        data: List[benchmark_utils.FunctionCount] = []
        for cs in count_strings:
            # 将每个字符串中的计数和函数名提取出来，并存储为格式化字符串，使得生成的 JSON 更易读
            match = pattern.search(cs)
            assert match is not None
            c, fn = match.groups()
            data.append(benchmark_utils.FunctionCount(count=int(c), function=fn))

        # 返回一个包含按计数排序后的函数统计信息的 FunctionCounts 对象
        return benchmark_utils.FunctionCounts(
            tuple(sorted(data, reverse=True)), inclusive=inclusive
        )

    # 从 artifacts 中获取 baseline_inclusive 数据，并转换为 FunctionCounts 对象
    baseline_inclusive = to_function_counts(artifacts["baseline_inclusive"], True)
    # 从 artifacts 中获取 baseline_exclusive 数据，并转换为 FunctionCounts 对象
    baseline_exclusive = to_function_counts(artifacts["baseline_exclusive"], False)

    # 创建一个不包含详细数据的 CallgrindStats 对象
    stats_no_data = benchmark_utils.CallgrindStats(
        benchmark_utils.TaskSpec("y = torch.ones(())", "pass"),
        number_per_run=1000,
        built_with_debug_symbols=True,
        baseline_inclusive_stats=baseline_inclusive,
        baseline_exclusive_stats=baseline_exclusive,
        stmt_inclusive_stats=to_function_counts(
            artifacts["ones_no_data_inclusive"], True
        ),
        stmt_exclusive_stats=to_function_counts(
            artifacts["ones_no_data_exclusive"], False
        ),
        stmt_callgrind_out=None,
    )

    # 创建一个包含详细数据的 CallgrindStats 对象
    stats_with_data = benchmark_utils.CallgrindStats(
        benchmark_utils.TaskSpec("y = torch.ones((1,))", "pass"),
        number_per_run=1000,
        built_with_debug_symbols=True,
        baseline_inclusive_stats=baseline_inclusive,
        baseline_exclusive_stats=baseline_exclusive,
        stmt_inclusive_stats=to_function_counts(
            artifacts["ones_with_data_inclusive"], True
        ),
        stmt_exclusive_stats=to_function_counts(
            artifacts["ones_with_data_exclusive"], False
        ),
        stmt_callgrind_out=None,
    )

    # 返回包含两种不同详细数据的 CallgrindStats 对象
    return stats_no_data, stats_with_data
class MyModule(torch.nn.Module):
    # 自定义 PyTorch 模块，继承自 nn.Module

    def forward(self, x):
        # 定义前向传播函数，接受输入 x 并返回 x + 1
        return x + 1


class TestBenchmarkUtils(TestCase):
    # 测试类 TestBenchmarkUtils，继承自 unittest 的 TestCase

    def regularizeAndAssertExpectedInline(
        self, x: Any, expect: str, indent: int = 12
    ) -> None:
        # 标准化字符串 x 后，与期望字符串 expect 进行断言比较
        x_str: str = re.sub(
            "object at 0x[0-9a-fA-F]+>",
            "object at 0xXXXXXXXXXXXX>",
            x if isinstance(x, str) else repr(x),
        )
        
        if "\n" in x_str:
            # 如果 x_str 中包含换行符，则根据指定的缩进格式化字符串
            x_str = textwrap.indent(x_str, " " * indent)

        self.assertExpectedInline(x_str, expect, skip=1)

    def test_timer(self):
        # 测试计时器功能
        timer = benchmark_utils.Timer(
            stmt="torch.ones(())",
        )
        # 使用计时器测量语句执行时间的中位数，并断言其类型为 float
        sample = timer.timeit(5).median
        self.assertIsInstance(sample, float)

        # 使用计时器进行块自动范围测量，获取中位数，并断言其类型为 float
        median = timer.blocked_autorange(min_run_time=0.01).median
        self.assertIsInstance(median, float)

        # 设置一个非常高的阈值以避免持续集成中的不稳定性
        # 内部算法在 `test_adaptive_timer` 中进行测试
        median = timer.adaptive_autorange(threshold=0.5).median

        # 测试多行语句的正确性
        median = (
            benchmark_utils.Timer(
                stmt="""
                with torch.no_grad():
                    y = x + 1""",
                setup="""
                x = torch.ones((1,), requires_grad=True)
                for _ in range(5):
                    x = x + 1.0""",
            )
            .timeit(5)
            .median
        )
        self.assertIsInstance(sample, float)

    @slowTest
    @unittest.skipIf(IS_SANDCASTLE, "C++ timing is OSS only.")
    @unittest.skipIf(True, "Failing on clang, see 74398")
    def test_timer_tiny_fast_snippet(self):
        # 测试小型快速 C++ 片段的计时功能
        timer = benchmark_utils.Timer(
            "auto x = 1;(void)x;",
            timer=timeit.default_timer,
            language=benchmark_utils.Language.CPP,
        )
        # 使用计时器进行块自动范围测量，并获取中位数，断言其类型为 float
        median = timer.blocked_autorange().median
        self.assertIsInstance(median, float)

    @slowTest
    @unittest.skipIf(IS_SANDCASTLE, "C++ timing is OSS only.")
    @unittest.skipIf(True, "Failing on clang, see 74398")
    def test_cpp_timer(self):
        # 测试 C++ 计时功能
        timer = benchmark_utils.Timer(
            """
                #ifndef TIMER_GLOBAL_CHECK
                static_assert(false);
                #endif

                torch::Tensor y = x + 1;
            """,
            setup="torch::Tensor x = torch::empty({1});",
            global_setup="#define TIMER_GLOBAL_CHECK",
            timer=timeit.default_timer,
            language=benchmark_utils.Language.CPP,
        )
        # 使用计时器进行多次测量，并获取中位数，断言其类型为 float
        t = timer.timeit(10)
        self.assertIsInstance(t.median, float)
    # 定义一个用于模拟定时器的内部类 `_MockTimer`
    class _MockTimer:
        # 初始种子值
        _seed = 0
    
        # 定时器噪声级别
        _timer_noise_level = 0.05
        # 定时器基本成本，单位为秒，即 100 纳秒
        _timer_cost = 100e-9  # 100 ns
    
        # 函数执行噪声级别
        _function_noise_level = 0.05
        # 各个函数执行成本的元组列表
        _function_costs = (
            ("pass", 8e-9),  # 空操作的成本
            ("cheap_fn()", 4e-6),  # 便宜函数的成本
            ("expensive_fn()", 20e-6),  # 昂贵函数的成本
            ("with torch.no_grad():\n    y = x + 1", 10e-6),  # 使用 torch 进行操作的成本
        )
    
        # 初始化方法，接受语句、设置、定时器和全局变量作为参数
        def __init__(self, stmt, setup, timer, globals):
            # 使用预设的种子创建随机数生成器
            self._random_state = np.random.RandomState(seed=self._seed)
            # 从函数成本字典中获取给定语句的平均成本
            self._mean_cost = dict(self._function_costs)[stmt]
    
        # 模拟采样函数，返回带有噪声的采样时间
        def sample(self, mean, noise_level):
            return max(self._random_state.normal(mean, mean * noise_level), 5e-9)
    
        # 执行定时器，返回模拟的执行时间
        def timeit(self, number):
            return sum(
                [
                    # 第一次定时器调用
                    self.sample(self._timer_cost, self._timer_noise_level),
                    # 语句体执行的成本
                    self.sample(self._mean_cost * number, self._function_noise_level),
                    # 第二次定时器调用
                    self.sample(self._timer_cost, self._timer_noise_level),
                ]
            )
    
    # 标记为慢速测试，以及一些跳过条件的装饰器
    @slowTest
    @unittest.skipIf(IS_WINDOWS, "Valgrind is not supported on Windows.")
    @unittest.skipIf(IS_SANDCASTLE, "Valgrind is OSS only.")
    @unittest.skipIf(TEST_WITH_ASAN, "fails on asan")
    # 测试 collect_callgrind 方法的异常情况处理
    def test_collect_callgrind(self):
        # 断言捕获 ValueError 异常，并验证异常信息是否匹配特定正则表达式
        with self.assertRaisesRegex(
            ValueError,
            r"`collect_callgrind` requires that globals be wrapped "
            r"in `CopyIfCallgrind` so that serialization is explicit.",
        ):
            # 创建 Timer 对象，传入一个无效的 globals 参数，并调用 collect_callgrind 方法
            benchmark_utils.Timer("pass", globals={"x": 1}).collect_callgrind(
                collect_baseline=False
            )

        # 断言捕获 OSError 异常，并验证异常信息是否包含特定字符串
        with self.assertRaisesRegex(
            # 子进程引发 AttributeError（来自 pickle），_ValgrindWrapper 以通用 OSError 重新引发
            OSError,
            "AttributeError: Can't get attribute 'MyModule'",
        ):
            # 创建 Timer 对象，传入包含无效模块的 globals 参数，并调用 collect_callgrind 方法
            benchmark_utils.Timer(
                "model(1)",
                globals={"model": benchmark_utils.CopyIfCallgrind(MyModule())},
            ).collect_callgrind(collect_baseline=False)

        # 使用 torch.jit.script 装饰的函数定义，定义一个简单的 Torch 脚本函数
        @torch.jit.script
        def add_one(x):
            return x + 1

        # 创建 Timer 对象，测试基准工具的性能，包括多个 globals 参数的设置
        timer = benchmark_utils.Timer(
            "y = add_one(x) + k",
            setup="x = torch.ones((1,))",
            globals={
                "add_one": benchmark_utils.CopyIfCallgrind(add_one),
                "k": benchmark_utils.CopyIfCallgrind(5),
                "model": benchmark_utils.CopyIfCallgrind(
                    MyModule(),
                    setup=f"""\
                    import sys
                    sys.path.append({repr(os.path.split(os.path.abspath(__file__))[0])})
                    from test_benchmark_utils import MyModule
                    """,
                ),
            },
        )

        # 收集并分析 callgrind 数据，返回统计信息
        stats = timer.collect_callgrind(number=1000)

        # 对统计数据进行进一步的分析，计算指令计数
        counts = stats.counts(denoise=False)

        # 断言 counts 的类型为整数
        self.assertIsInstance(counts, int)
        # 断言 counts 大于 0
        self.assertGreater(counts, 0)

        # 分配器存在某些抖动，因此使用更简单的任务进行测试以验证可重复性
        timer = benchmark_utils.Timer(
            "x += 1",
            setup="x = torch.ones((1,))",
        )

        # 收集并分析多次 callgrind 数据，返回多个统计结果的元组
        stats = timer.collect_callgrind(number=1000, repeats=20)

        # 断言 stats 的类型为元组
        assert isinstance(stats, tuple)

        # 检查重复的结果是否至少在某种程度上可重复（每迭代不超过 10 指令）
        counts = collections.Counter(
            [s.counts(denoise=True) // 10_000 * 10_000 for s in stats]
        )
        
        # 断言 counts 中最大值大于 1，确保重复结果不完全相同
        self.assertGreater(
            max(counts.values()),
            1,
            f"Every instruction count total was unique: {counts}",
        )

        # 导入 Valgrind 包装器的单例对象，并断言其绑定模块为 None
        from torch.utils.benchmark.utils.valgrind_wrapper.timer_interface import (
            wrapper_singleton,
        )
        self.assertIsNone(
            wrapper_singleton()._bindings_module,
            "JIT'd bindings are only for back testing.",
        )
    # 定义一个测试方法，用于收集 C++ 程序在 Callgrind 下的性能统计信息
    def test_collect_cpp_callgrind(self):
        # 创建一个计时器对象，测试执行 C++ 代码 "x += 1;"
        timer = benchmark_utils.Timer(
            "x += 1;",
            setup="torch::Tensor x = torch::ones({1});",
            timer=timeit.default_timer,
            language="c++",
        )
        # 收集 Callgrind 数据，重复收集三次
        stats = [timer.collect_callgrind() for _ in range(3)]
        # 获取每次收集的指令计数
        counts = [s.counts() for s in stats]

        # 断言最小的指令计数大于 0，确保至少有一些统计信息被收集
        self.assertGreater(min(counts), 0, "No stats were collected")
        # 断言最小和最大的指令计数相等，表明 C++ Callgrind 是确定性的
        self.assertEqual(
            min(counts), max(counts), "C++ Callgrind should be deterministic"
        )

        # 对于每个统计数据对象，验证去噪后的指令计数与原始的指令计数相等
        for s in stats:
            self.assertEqual(
                s.counts(denoise=True),
                s.counts(denoise=False),
                "De-noising should not apply to C++.",
            )

        # 重新收集 Callgrind 数据，设置执行次数为 1000，重复次数为 20 次
        stats = timer.collect_callgrind(number=1000, repeats=20)
        # 断言 stats 是一个元组类型
        assert isinstance(stats, tuple)

        # 注意事项：与上面示例不同，这里不要求所有重复收集的结果都完全相同。
        # 使用 Counter 统计去噪后的指令计数，并将每个计数约束到以万为单位
        counts = collections.Counter(
            [s.counts(denoise=True) // 10_000 * 10_000 for s in stats]
        )
        # 断言计数中最大值大于 1，确保至少有多个不同的计数存在
        self.assertGreater(max(counts.values()), 1, repr(counts))

    # 如果在 Windows 平台并且环境变量 VC_YEAR 是 2019，则跳过该测试方法
    @unittest.skipIf(
        IS_WINDOWS and os.getenv("VC_YEAR") == "2019", "Random seed only accepts int32"
    )
    # 定义一个测试方法，用于测试模糊测试工具
    def test_fuzzer(self):
        # 创建一个模糊测试对象
        fuzzer = benchmark_utils.Fuzzer(
            # 指定模糊测试参数，包括一个 loguniform 分布的参数 n，取值范围为 1 到 16
            parameters=[
                benchmark_utils.FuzzedParameter(
                    "n", minval=1, maxval=16, distribution="loguniform"
                )
            ],
            # 模糊测试使用的张量，这里包含一个名为 "x" 的张量，其大小由参数 "n" 决定
            tensors=[benchmark_utils.FuzzedTensor("x", size=("n",))],
            # 设置模糊测试的随机种子为 0
            seed=0,
        )

        # 预期的模糊测试结果，每个元组包含一个预期的张量值
        expected_results = [
            (0.7821, 0.0536, 0.9888, 0.1949, 0.5242, 0.1987, 0.5094),
            (0.7166, 0.5961, 0.8303, 0.005),
        ]

        # 对于 fuzzer.take(2) 返回的两个元组，分别进行验证
        for i, (tensors, _, _) in enumerate(fuzzer.take(2)):
            # 获取张量 "x"
            x = tensors["x"]
            # 断言张量 "x" 的值与预期的结果相等，设置相对误差为 1e-3，绝对误差为 1e-3
            self.assertEqual(x, torch.tensor(expected_results[i]), rtol=1e-3, atol=1e-3)
# 如果这个脚本作为主程序运行，则执行 run_tests() 函数
if __name__ == "__main__":
    run_tests()
```