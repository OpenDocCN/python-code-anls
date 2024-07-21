# `.\pytorch\torch\utils\throughput_benchmark.py`

```py
# mypy: allow-untyped-defs  # 声明允许在类型检查中使用未标注类型的函数和变量

import torch._C  # 导入torch._C模块，通常用于访问底层C++ API

def format_time(time_us=None, time_ms=None, time_s=None):
    """定义时间格式化函数。"""
    # 确保只有一个时间单位参数被指定
    assert sum([time_us is not None, time_ms is not None, time_s is not None]) == 1

    US_IN_SECOND = 1e6  # 微秒到秒的换算常量
    US_IN_MS = 1e3  # 微秒到毫秒的换算常量

    if time_us is None:
        if time_ms is not None:
            time_us = time_ms * US_IN_MS  # 将毫秒转换为微秒
        elif time_s is not None:
            time_us = time_s * US_IN_SECOND  # 将秒转换为微秒
        else:
            raise AssertionError("Shouldn't reach here :)")  # 如果执行到这里，抛出断言错误

    if time_us >= US_IN_SECOND:
        return f'{time_us / US_IN_SECOND:.3f}s'  # 如果时间大于等于1秒，以秒为单位返回格式化字符串
    if time_us >= US_IN_MS:
        return f'{time_us / US_IN_MS:.3f}ms'  # 如果时间大于等于1毫秒，以毫秒为单位返回格式化字符串
    return f'{time_us:.3f}us'  # 否则以微秒为单位返回格式化字符串


class ExecutionStats:
    def __init__(self, c_stats, benchmark_config):
        self._c_stats = c_stats  # 保存传入的c_stats对象
        self.benchmark_config = benchmark_config  # 保存传入的benchmark_config对象

    @property
    def latency_avg_ms(self):
        return self._c_stats.latency_avg_ms  # 返回_c_stats对象中的平均延迟时间（毫秒）

    @property
    def num_iters(self):
        return self._c_stats.num_iters  # 返回_c_stats对象中的迭代次数

    @property
    def iters_per_second(self):
        """返回所有调用线程每秒的迭代次数总和。"""
        return self.num_iters / self.total_time_seconds  # 计算每秒的迭代次数

    @property
    def total_time_seconds(self):
        return self.num_iters * (self.latency_avg_ms / 1000.0) / self.benchmark_config.num_calling_threads
        # 计算总时间（秒），考虑迭代次数、平均延迟时间（转换为秒）、调用线程数

    def __str__(self):
        return '\n'.join([
            "Average latency per example: " + format_time(time_ms=self.latency_avg_ms),
            f"Total number of iterations: {self.num_iters}",
            f"Total number of iterations per second (across all threads): {self.iters_per_second:.2f}",
            "Total time: " + format_time(time_s=self.total_time_seconds)
        ])
        # 返回统计信息的格式化字符串，包括平均延迟时间、总迭代次数、每秒迭代次数以及总时间


class ThroughputBenchmark:
    """
    该类是对c++组件throughput_benchmark::ThroughputBenchmark的封装。

    这个类负责在推理服务器负载下执行PyTorch模块（nn.Module或ScriptModule）。
    它可以模拟多个调用线程对单个提供的模块进行调用。未来，我们计划增强该组件，
    支持跨操作并行性、内部操作并行性以及在单个进程中运行多个模型。

    请注意，尽管支持nn.Module，但在执行Python代码或传递输入作为Python对象时，
    可能会增加持有GIL的开销。一旦您有模型的ScriptModule版本用于推理部署，
    最好切换到使用它来进行基准测试。

    """
    def __init__(self, module):
        # 检查传入的模块是否为 torch.jit.ScriptModule 类型
        if isinstance(module, torch.jit.ScriptModule):
            # 如果是，使用 Torch 的 ThroughputBenchmark 初始化对象
            self._benchmark = torch._C.ThroughputBenchmark(module._c)
        else:
            # 否则，使用 Torch 的 ThroughputBenchmark 初始化对象
            self._benchmark = torch._C.ThroughputBenchmark(module)

    def run_once(self, *args, **kwargs):
        """
        给定输入 ID（input_idx），运行一次基准测试并返回预测结果。

        这对于测试基准实际运行你想要运行的模块很有用。这里的 input_idx 是一个索引，
        它指向通过调用 add_input() 方法填充的 inputs 数组中的元素。
        """
        # 调用 Torch 的 ThroughputBenchmark 对象的 run_once 方法
        return self._benchmark.run_once(*args, **kwargs)

    def add_input(self, *args, **kwargs):
        """
        将单个模块的输入存储到基准测试内存中，并保留在那里。

        在基准测试执行期间，每个线程都会从通过该函数向基准测试提供的所有输入中随机选择一个。
        """
        # 调用 Torch 的 ThroughputBenchmark 对象的 add_input 方法
        self._benchmark.add_input(*args, **kwargs)
    # 定义一个 benchmark 方法，用于运行模块的基准测试
    def benchmark(
            self,
            num_calling_threads=1,
            num_warmup_iters=10,
            num_iters=100,
            profiler_output_path=""):
        """
        Run a benchmark on the module.

        Args:
            num_calling_threads (int): 调用线程的数量，默认为 1
            num_warmup_iters (int): 用于确保在实际测量之前运行模块几次的预热迭代次数。
                这样我们可以避免冷缓存和类似问题。每个线程的预热迭代次数。
            num_iters (int): 基准测试应该运行的迭代次数。这个数字独立于预热迭代次数。
                所有线程共享此数。一旦所有线程的总迭代次数达到 num_iters，执行将停止。
                实际迭代次数可能稍多，这会在 stats.num_iters 中报告。
            profiler_output_path (str): 保存 Autograd Profiler 跟踪文件的位置。
                如果不为空，Autograd Profiler 将在主基准测试执行期间启用（但不在预热阶段）。
                完整的跟踪将保存在此参数提供的文件路径中。

        返回一个 BenchmarkExecutionStats 对象，该对象通过 pybind11 定义。
        目前有两个字段:
            - num_iters - 基准测试实际进行的迭代次数
            - avg_latency_ms - 每个输入示例推断所需的平均时间（以毫秒为单位）
        """
        # 创建 BenchmarkConfig 对象
        config = torch._C.BenchmarkConfig()
        # 设置调用线程数
        config.num_calling_threads = num_calling_threads
        # 设置预热迭代次数
        config.num_warmup_iters = num_warmup_iters
        # 设置基准测试迭代次数
        config.num_iters = num_iters
        # 设置 Autograd Profiler 跟踪文件路径
        config.profiler_output_path = profiler_output_path
        # 调用 self._benchmark 对象的 benchmark 方法进行基准测试，并获取 C++ 结果
        c_stats = self._benchmark.benchmark(config)
        # 返回 ExecutionStats 对象，将 C++ 结果和配置对象传递给它
        return ExecutionStats(c_stats, config)
```