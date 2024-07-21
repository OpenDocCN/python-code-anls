# `.\pytorch\benchmarks\framework_overhead_benchmark\framework_overhead_benchmark.py`

```py
# 导入必要的模块
import argparse  # 导入命令行参数解析模块

from pt_wrapper_module import WrapperModule  # 导入自定义的PyTorch模块包装器

from SimpleAddModule import add_tensors_loop, SimpleAddModule  # 导入简单加法模块和其函数
from utils import benchmark_module, BenchmarkConfig, ModuleConfig, ms_to_us  # 导入基准测试相关的工具函数和配置类

""" 框架开销基准测试脚本。
基准测试框架开销。
当前支持的操作：add。
目前仅运行前向传播。
支持图模式和急切模式。在图模式下，通过JIT跟踪跟踪模块。
调试选项在启用图模式时打印跟踪图。
可以通过保存选项保存图。保存在运行基准测试的目录中。
示例构建/运行：
要运行PT基准测试：
buck run @mode/opt <path-to-framework_overhead_benchmark>:framework_overhead_benchmark --
 --add-op --graph-mode --eager-mode（同时运行图模式和急切模式）
buck run @mode/opt <path-to-framework_overhead_benchmark>:framework_overhead_benchmark --
 --add-op --graph-mode（仅运行图模式）
"""

SUPPORTED_OPS = {"add_op"}  # 支持的操作集合，当前仅包含"add_op"


def parse_op_args(op):
    op_list = op.split(",")  # 将操作参数分割为列表


def print_results(result):
    print("===================================")
    for key, value in result.items():
        print(f"{key}, latency per iter (us):{ms_to_us(value)}")  # 打印每个迭代的延迟时间（以微秒为单位）
    print("===================================")


def benchmark_simple_fn(args, config, module_config, module_type, result):
    """对指定在配置中的PyTorch可追踪函数进行基准测试。
    实例化一个包装器对象，该对象包装module_type的对象，并使用benchmark_module运行前向传播方法。
    Args:
        config:         包含预热和基准迭代次数的配置。
        module_config:  包含操作、操作参数数量以及是否启用图模式的模块配置。
        module_type:    要包装的模块类型。例如，对于加法操作，可以是SimpleAddModule。
        result:         将用基准测试结果（每迭代延迟时间）填充的字典实例。
    """
    print(f"Benchmarking {module_type.__name__}")  # 打印正在基准测试的模块类型的名称
    f_name = (
        module_config.pt_fn.__name__ + ":Num Operands=" + str(module_config.num_params)
    )  # 根据模块配置构造函数名称字符串
    graph_mode_str = "Graph mode" + ":" + str(module_config.graph_mode)  # 构造图模式字符串
    result_key = ",".join((f_name, graph_mode_str))  # 构造结果字典的键
    module = WrapperModule(module_type, module_config, args.debug, args.save)  # 实例化包装器模块对象
    latency_per_iter_ms = benchmark_module(
        config, module, args.use_throughput_benchmark
    )  # 运行基准测试模块
    result[result_key] = latency_per_iter_ms  # 将每迭代延迟时间添加到结果字典中


def main():
    parser = argparse.ArgumentParser()  # 创建命令行参数解析器
    parser.add_argument("--op", default="add_op", dest="op", type=str)  # 添加操作参数选项
    parser.add_argument(
        "--use-throughput-benchmark",
        "--use_throughput_benchmark",
        default=False,
        dest="use_throughput_benchmark",
        action="store_true",
    )  # 添加是否使用吞吐量基准测试选项
    parser.add_argument("--debug", default=False, dest="debug", action="store_true")  # 添加调试选项
    parser.add_argument("--save", default=False, dest="save", action="store_true")  # 添加保存选项
    parser.add_argument(
        "--eager-mode",
        "--eager_mode",
        default=False,
        dest="eager_mode",
        action="store_true",
    )
    parser.add_argument(
        "--num-warmup-iters", "--num_warmup_iters", type=int, default=100
    )
    parser.add_argument("--num-iters", "--num_iters", type=int, default=1000)
    args = parser.parse_args()

    # 检查所选操作是否在支持的操作列表中
    if args.op not in SUPPORTED_OPS:
        print(f"Op {args.op} is not supported: Supported ops are:{SUPPORTED_OPS}")
        return

    # 设置预热迭代次数和总迭代次数
    num_warmup_iters = args.num_warmup_iters
    num_iters = args.num_iters

    # 根据设置的迭代次数创建基准配置对象
    config = BenchmarkConfig(num_warmup_iters, num_iters)

    # 默认以图模式执行
    graph_mode = True
    if args.eager_mode:
        # 如果启用急切执行模式，则以急切模式执行
        graph_mode = False

    # 初始化结果字典
    result = {}

    # 根据操作类型选择相应的模块配置和执行基准测试
    if args.op == "add_op":
        num_params = 2
        module_config = ModuleConfig(add_tensors_loop, None, num_params, graph_mode)
        benchmark_simple_fn(args, config, module_config, SimpleAddModule, result)

    # 打印执行结果
    print_results(result)
# 如果当前脚本作为主程序执行（而不是被导入为模块），则执行 main 函数
if __name__ == "__main__":
    main()
```