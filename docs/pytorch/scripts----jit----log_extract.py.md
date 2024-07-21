# `.\pytorch\scripts\jit\log_extract.py`

```py
# 导入必要的模块和函数
import argparse                   # 用于解析命令行参数的模块
import functools                  # 提供了高阶函数和操作工具的模块
import traceback                  # 提供异常追踪功能的模块
from typing import Callable, List, Optional, Tuple   # 引入类型提示的必要工具

# 从 torch.utils.jit.log_extract 中导入所需的函数
from torch.utils.jit.log_extract import (
    extract_ir,                  # 提取 IR 的函数
    load_graph_and_inputs,       # 加载图和输入的函数
    run_baseline_no_fusion,      # 运行没有融合的基准的函数
    run_nnc,                     # 运行 nnc 的函数
    run_nvfuser,                 # 运行 nvfuser 的函数
)

"""
Usage:
1. Run your script and pipe into a log file
  PYTORCH_JIT_LOG_LEVEL=">>graph_fuser" python3 my_test.py &> log.txt
2. Run log_extract:
  log_extract.py log.txt --nvfuser --nnc-dynamic --nnc-static

You can also extract the list of extracted IR:
  log_extract.py log.txt --output

Passing in --graphs 0 2 will only run graphs 0 and 2
"""

# 定义一个函数用于运行测试
def test_runners(
    graphs: List[str],                     # 图形名称列表
    runners: List[Tuple[str, Callable]],   # 包含运行器名称和运行函数的元组列表
    graph_set: Optional[List[int]],        # 可选的图形集合列表
):
    # 遍历图形列表
    for i, ir in enumerate(graphs):
        # 加载图形及其输入
        _, inputs = load_graph_and_inputs(ir)
        # 如果图形集合存在且当前索引不在集合中，则跳过该图形
        if graph_set and i not in graph_set:
            continue

        # 打印当前正在运行的图形编号
        print(f"Running Graph {i}")
        prev_result = None
        prev_runner_name = None
        # 遍历运行器列表
        for runner in runners:
            runner_name, runner_fn = runner
            try:
                # 运行运行器函数，并记录结果
                result = runner_fn(ir, inputs)
                # 如果有上一个运行结果，则计算改进百分比并打印信息
                if prev_result:
                    improvement = (prev_result / result - 1) * 100
                    print(
                        f"{runner_name} : {result:.6f} ms improvement over {prev_runner_name}: improvement: {improvement:.2f}%"
                    )
                else:
                    # 如果是第一个运行器，则仅打印结果
                    print(f"{runner_name} : {result:.6f} ms")
                prev_result = result
                prev_runner_name = runner_name
            except RuntimeError:
                # 捕获运行时异常，并打印出错信息
                print(f"  Graph {i} failed for {runner_name} :", traceback.format_exc())


# 定义一个函数用于运行程序
def run():
    parser = argparse.ArgumentParser(
        description="Extracts torchscript IR from log files and, optionally, benchmarks it or outputs the IR"
    )
    parser.add_argument("filename", help="Filename of log file")
    parser.add_argument(
        "--nvfuser", dest="nvfuser", action="store_true", help="benchmark nvfuser"
    )
    parser.add_argument(
        "--no-nvfuser",
        dest="nvfuser",
        action="store_false",
        help="DON'T benchmark nvfuser",
    )
    parser.set_defaults(nvfuser=False)
    parser.add_argument(
        "--nnc-static",
        dest="nnc_static",
        action="store_true",
        help="benchmark nnc static",
    )
    parser.add_argument(
        "--no-nnc-static",
        dest="nnc_static",
        action="store_false",
        help="DON'T benchmark nnc static",
    )
    parser.set_defaults(nnc_static=False)

    parser.add_argument(
        "--nnc-dynamic",
        dest="nnc_dynamic",
        action="store_true",
        help="nnc with dynamic shapes",
    )
    parser.add_argument(
        "--no-nnc-dynamic",
        dest="nnc_dynamic",
        action="store_false",
        help="DONT't benchmark nnc with dynamic shapes",
    )
    parser.set_defaults(nnc_dynamic=False)
    # 添加一个命令行参数，用于启用基准测试
    parser.add_argument(
        "--baseline", dest="baseline", action="store_true", help="benchmark baseline"
    )
    # 添加一个命令行参数，用于禁用基准测试
    parser.add_argument(
        "--no-baseline",
        dest="baseline",
        action="store_false",
        help="DON'T benchmark baseline",
    )
    # 设置默认的基准测试选项为 False
    parser.set_defaults(baseline=False)

    # 添加一个命令行参数，用于输出图形 IR
    parser.add_argument(
        "--output", dest="output", action="store_true", help="Output graph IR"
    )
    # 添加一个命令行参数，用于禁止输出图形 IR
    parser.add_argument(
        "--no-output", dest="output", action="store_false", help="DON'T output graph IR"
    )
    # 设置默认的输出图形 IR 的选项为 False
    parser.set_defaults(output=False)

    # 添加一个命令行参数，用于指定要运行的图形索引列表
    parser.add_argument(
        "--graphs", nargs="+", type=int, help="Run only specified graph indices"
    )

    # 解析命令行参数
    args = parser.parse_args()
    # 从指定文件中提取图形 IR 数据
    graphs = extract_ir(args.filename)

    # 设置 graph_set 变量为命令行参数中指定的图形索引列表，若未指定则设为 None
    graph_set = args.graphs
    graph_set = graph_set if graph_set else None

    # 初始化选项列表
    options = []
    # 如果命令行参数中指定了 baseline 选项，则添加相应选项和运行函数到 options 列表中
    if args.baseline:
        options.append(("Baseline no fusion", run_baseline_no_fusion))
    # 如果命令行参数中指定了 nnc_dynamic 选项，则添加相应选项和部分应用的运行函数到 options 列表中
    if args.nnc_dynamic:
        options.append(("NNC Dynamic", functools.partial(run_nnc, dynamic=True)))
    # 如果命令行参数中指定了 nnc_static 选项，则添加相应选项和部分应用的运行函数到 options 列表中
    if args.nnc_static:
        options.append(("NNC Static", functools.partial(run_nnc, dynamic=False)))
    # 如果命令行参数中指定了 nvfuser 选项，则添加相应选项和运行函数到 options 列表中
    if args.nvfuser:
        options.append(("NVFuser", run_nvfuser))

    # 运行测试函数，传入图形 IR 数据、选项列表和图形索引集合
    test_runners(graphs, options, graph_set)

    # 如果命令行参数中指定了 output 选项为 True，则输出引号包裹的图形 IR 列表
    if args.output:
        quoted = []
        for i, ir in enumerate(graphs):
            # 如果 graph_set 存在且当前索引不在图形索引集合中，则跳过该图形 IR
            if graph_set and i not in graph_set:
                continue
            # 将图形 IR 加入引号包裹的 quoted 列表中
            quoted.append('"""' + ir + '"""')
        # 打印格式化的引号包裹的图形 IR 列表
        print("[" + ", ".join(quoted) + "]")
# 如果当前脚本作为主程序执行（而不是作为模块被导入执行）
if __name__ == "__main__":
    # 调用名为 run 的函数或方法，这通常是启动程序的入口点
    run()
```