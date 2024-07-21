# `.\pytorch\benchmarks\operator_benchmark\benchmark_runner.py`

```py
# 导入必要的模块 argparse：用于命令行参数解析
import argparse

# 导入自定义的性能基准测试核心模块 benchmark_core 和实用工具模块 benchmark_utils
import benchmark_core
import benchmark_utils

# 导入 PyTorch 模块
import torch

"""Performance microbenchmarks's main binary.

This is the main function for running performance microbenchmark tests.
It also registers existing benchmark tests via Python module imports.
"""
# 创建一个命令行参数解析器对象 argparse.ArgumentParser
parser = argparse.ArgumentParser(
    description="Run microbenchmarks.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

# 定义解析命令行参数的函数
def parse_args():
    # 添加命令行参数 --tag-filter，用于根据标签过滤要运行的测试形状
    parser.add_argument(
        "--tag-filter",
        "--tag_filter",
        help="tag_filter can be used to run the shapes which matches the tag. (all is used to run all the shapes)",
        default="short",
    )

    # 添加命令行参数 --operators，用于根据逗号分隔的操作符列表来过滤要测试的操作
    # 这是一个可选参数
    # 这个选项用于过滤要运行的测试用例
    parser.add_argument(
        "--operators",
        help="Filter tests based on comma-delimited list of operators to test",
        default=None,
    )

    # 添加命令行参数 --operator-range，用于根据操作范围来过滤要测试的操作
    # 例如：a-c 或者 b,c-d
    # 这是一个可选参数
    parser.add_argument(
        "--operator-range",
        "--operator_range",
        help="Filter tests based on operator_range(e.g. a-c or b,c-d)",
        default=None,
    )

    # 添加命令行参数 --test-name，用于运行具有指定测试名称的测试
    # 这是一个可选参数
    parser.add_argument(
        "--test-name",
        "--test_name",
        help="Run tests that have the provided test_name",
        default=None,
    )

    # 添加命令行参数 --list-ops，用于列出所有操作符而不运行它们
    # 这是一个布尔类型的选项
    parser.add_argument(
        "--list-ops",
        "--list_ops",
        help="List operators without running them",
        action="store_true",
    )

    # 添加命令行参数 --list-tests，用于列出所有测试用例而不运行它们
    # 这是一个布尔类型的选项
    parser.add_argument(
        "--list-tests",
        "--list_tests",
        help="List all test cases without running them",
        action="store_true",
    )

    # 添加命令行参数 --iterations，用于设置每个操作符重复执行的次数
    parser.add_argument(
        "--iterations",
        help="Repeat each operator for the number of iterations",
        type=int,
    )

    # 添加命令行参数 --num-runs，用于设置每个测试运行的次数
    # 每次运行执行指定数量的 <--iterations>
    parser.add_argument(
        "--num-runs",
        "--num_runs",
        help="Run each test for num_runs. Each run executes an operator for number of <--iterations>",
        type=int,
        default=1,
    )

    # 添加命令行参数 --min-time-per-test，用于设置运行每个测试的最小时间（单位：秒）
    parser.add_argument(
        "--min-time-per-test",
        "--min_time_per_test",
        help="Set the minimum time (unit: seconds) to run each test",
        type=int,
        default=0,
    )

    # 添加命令行参数 --warmup-iterations，用于设置在测量性能之前要忽略的迭代次数
    parser.add_argument(
        "--warmup-iterations",
        "--warmup_iterations",
        help="Number of iterations to ignore before measuring performance",
        default=100,
        type=int,
    )

    # 添加命令行参数 --omp-num-threads，用于设置 PyTorch 运行时中使用的 OpenMP 线程数
    parser.add_argument(
        "--omp-num-threads",
        "--omp_num_threads",
        help="Number of OpenMP threads used in PyTorch runtime",
        default=None,
        type=int,
    )

    # 添加命令行参数 --mkl-num-threads，用于设置 PyTorch 运行时中使用的 MKL 线程数
    parser.add_argument(
        "--mkl-num-threads",
        "--mkl_num_threads",
        help="Number of MKL threads used in PyTorch runtime",
        default=None,
        type=int,
    )

    # 添加命令行参数 --report-aibench，用于在 AIBench 上运行时打印结果
    parser.add_argument(
        "--report-aibench",
        "--report_aibench",
        type=benchmark_utils.str2bool,  # 使用 benchmark_utils 模块中的 str2bool 函数进行类型转换
        nargs="?",
        const=True,
        default=False,
        help="Print result when running on AIBench",
    )
    # 添加一个名为 --use-jit 的命令行参数，类型为 benchmark_utils.str2bool
    # 如果命令行中没有提供该参数，则默认为 False
    parser.add_argument(
        "--use-jit",
        "--use_jit",
        type=benchmark_utils.str2bool,
        nargs="?",
        const=True,
        default=False,
        help="Run operators with PyTorch JIT mode",
    )

    # 添加一个名为 --forward-only 的命令行参数，类型为 benchmark_utils.str2bool
    # 如果命令行中没有提供该参数，则默认为 False
    parser.add_argument(
        "--forward-only",
        "--forward_only",
        type=benchmark_utils.str2bool,
        nargs="?",
        const=True,
        default=False,
        help="Only run the forward path of operators",
    )

    # 添加一个名为 --device 的命令行参数，默认值为 "None"
    # 用于指定测试运行的架构（cpu、cuda）
    parser.add_argument(
        "--device",
        help="Run tests on the provided architecture (cpu, cuda)",
        default="None",
    )

    # 解析命令行参数并返回结果，忽略未知的命令行参数
    args, _ = parser.parse_known_args()

    # 如果命令行参数中指定了 omp_num_threads
    if args.omp_num_threads:
        # 设置环境变量 OMP_NUM_THREADS，不过由于 C2 初始化逻辑已经调用过，
        # 设置该环境变量此时不会有任何影响
        # 参考 OpenMP 标准第四章节：
        # https://www.openmp.org/wp-content/uploads/openmp-4.5.pdf
        # 根据标准，程序启动后修改环境变量（包括程序自身修改）不会被 OpenMP 实现所采纳
        benchmark_utils.set_omp_threads(args.omp_num_threads)
        # 设置 PyTorch 的线程数为指定的 omp_num_threads
        torch.set_num_threads(args.omp_num_threads)

    # 如果命令行参数中指定了 mkl_num_threads
    if args.mkl_num_threads:
        # 设置 MKL 线程数为指定的 mkl_num_threads
        benchmark_utils.set_mkl_threads(args.mkl_num_threads)

    # 返回解析后的命令行参数对象 args
    return args
# 主程序入口点，Python 程序的开始执行处
def main():
    # 解析命令行参数，返回参数对象
    args = parse_args()
    # 创建 BenchmarkRunner 对象，传入参数对象，然后运行 benchmark
    benchmark_core.BenchmarkRunner(args).run()

# 如果当前脚本作为主程序运行，则执行 main 函数
if __name__ == "__main__":
    main()
```