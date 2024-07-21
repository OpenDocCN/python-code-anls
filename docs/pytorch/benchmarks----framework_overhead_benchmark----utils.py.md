# `.\pytorch\benchmarks\framework_overhead_benchmark\utils.py`

```py
#`
# 导入时间模块
import time
# 导入命名元组，便于创建带字段的元组
from collections import namedtuple

# 从 torch.utils 模块导入 ThroughputBenchmark 类，用于性能基准测试
from torch.utils import ThroughputBenchmark

# 定义循环迭代次数常量，设为 1000
NUM_LOOP_ITERS = 1000
# 定义 BenchmarkConfig 命名元组，包含 warmup 迭代次数和测试迭代次数字段
BenchmarkConfig = namedtuple("BenchmarkConfig", "num_warmup_iters num_iters")
# 定义 ModuleConfig 命名元组，包含 pytorch 函数、C2 操作、参数数量和图模式字段
ModuleConfig = namedtuple("ModuleConfig", "pt_fn c2_op num_params graph_mode")

# 定义将毫秒转换为微秒的函数
def ms_to_us(time_ms):
    return time_ms * 1e3

# 定义将秒转换为微秒的函数
def secs_to_us(time_s):
    return time_s * 1e6

# 定义将秒转换为毫秒的函数
def secs_to_ms(time_s):
    return time_s * 1e3

# 定义使用 ThroughputBenchmark 进行基准测试的函数
def benchmark_using_throughput_benchmark(config, module):
    print("Benchmarking via ThroughputBenchmark")
    # 初始化 ThroughputBenchmark 对象，传入模块的模块实例
    bench = ThroughputBenchmark(module.module)
    # 添加输入数据到基准测试中
    bench.add_input(*module.tensor_inputs)
    # 执行基准测试，返回测试结果，包括 warmup 和测试迭代次数
    stats = bench.benchmark(1, config.num_warmup_iters, config.num_iters)
    # 返回平均延迟时间，单位为毫秒，除以循环迭代次数
    return stats.latency_avg_ms / NUM_LOOP_ITERS

# 定义基准测试模块的函数
def benchmark_module(config, module, use_throughput_benchmark=False):
    # 根据 use_throughput_benchmark 标志，选择使用 ThroughputBenchmark 或自定义基准测试
    if use_throughput_benchmark:
        return benchmark_using_throughput_benchmark(config, module)
    # 执行 warmup 迭代
    module.forward(config.num_warmup_iters)
    print(f"Running module for {config.num_iters} iterations")
    # 记录开始时间
    start = time.time()
    # 执行测试迭代
    module.forward(config.num_iters)
    # 记录结束时间
    end = time.time()
    # 计算经过的时间，单位为秒
    time_elapsed_s = end - start
    # 返回每次迭代的平均时间，单位为毫秒，除以测试迭代次数和循环迭代次数
    return secs_to_ms(time_elapsed_s) / config.num_iters / NUM_LOOP_ITERS
```