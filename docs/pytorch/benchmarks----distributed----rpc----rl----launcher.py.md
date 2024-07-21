# `.\pytorch\benchmarks\distributed\rpc\rl\launcher.py`

```
# 导入必要的模块
import argparse  # 用于解析命令行参数

import json  # 用于 JSON 数据的处理
import os  # 用于操作系统相关的功能
import time  # 用于时间相关的操作

from coordinator import CoordinatorBase  # 导入 CoordinatorBase 类

import torch.distributed.rpc as rpc  # 导入 PyTorch 的分布式 RPC 模块
import torch.multiprocessing as mp  # 导入 PyTorch 的多进程模块

COORDINATOR_NAME = "coordinator"  # 协调器的名称
AGENT_NAME = "agent"  # 代理的名称
OBSERVER_NAME = "observer{}"  # 观察者名称的格式化字符串

TOTAL_EPISODES = 10  # 总的模拟周期数
TOTAL_EPISODE_STEPS = 100  # 每个模拟周期的步数


def str2bool(v):
    """
    将字符串转换为布尔值
    Args:
        v (str): 输入字符串

    Returns:
        bool: 对应的布尔值
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


parser = argparse.ArgumentParser(description="PyTorch RPC RL Benchmark")  # 创建命令行参数解析器
parser.add_argument("--world-size", "--world_size", type=str, default="10")  # 添加世界大小参数
parser.add_argument("--master-addr", "--master_addr", type=str, default="127.0.0.1")  # 添加主地址参数
parser.add_argument("--master-port", "--master_port", type=str, default="29501")  # 添加主端口参数
parser.add_argument("--batch", type=str, default="True")  # 添加是否批处理参数

parser.add_argument("--state-size", "--state_size", type=str, default="10-20-10")  # 添加状态大小参数
parser.add_argument("--nlayers", type=str, default="5")  # 添加层数参数
parser.add_argument("--out-features", "--out_features", type=str, default="10")  # 添加输出特征数参数
parser.add_argument(
    "--output-file-path",
    "--output_file_path",
    type=str,
    default="benchmark_report.json",
)  # 添加输出文件路径参数

args = parser.parse_args()  # 解析命令行参数
args = vars(args)  # 将解析结果转换为字典


def run_worker(
    rank,
    world_size,
    master_addr,
    master_port,
    batch,
    state_size,
    nlayers,
    out_features,
    queue,
):
    """
    初始化一个 RPC 工作进程
    Args:
        rank (int): RPC 工作进程的排名
        world_size (int): RPC 网络中的工作进程数量（观察者数 + 1 个代理 + 1 个协调器）
        master_addr (str): 协调器的主地址
        master_port (str): 协调器的主端口
        batch (bool): 代理是否使用批处理或逐个处理观察者请求
        state_size (str): 表示状态维度的数值字符串（例如：5-15-10）
        nlayers (int): 模型中的层数
        out_features (int): 模型的输出特征数
        queue (SimpleQueue): 用于保存基准运行结果的 Torch 多进程上下文中的 SimpleQueue
    """
    state_size = list(map(int, state_size.split("-")))  # 将状态大小参数转换为整数列表
    batch_size = world_size - 2  # 观察者的数量

    os.environ["MASTER_ADDR"] = master_addr  # 设置环境变量：主地址
    os.environ["MASTER_PORT"] = master_port  # 设置环境变量：主端口
    if rank == 0:
        rpc.init_rpc(COORDINATOR_NAME, rank=rank, world_size=world_size)  # 初始化协调器 RPC

        coordinator = CoordinatorBase(  # 创建 CoordinatorBase 实例
            batch_size, batch, state_size, nlayers, out_features
        )
        coordinator.run_coordinator(TOTAL_EPISODES, TOTAL_EPISODE_STEPS, queue)  # 运行协调器

    elif rank == 1:
        rpc.init_rpc(AGENT_NAME, rank=rank, world_size=world_size)  # 初始化代理 RPC
    else:
        rpc.init_rpc(OBSERVER_NAME.format(rank), rank=rank, world_size=world_size)  # 初始化观察者 RPC
    rpc.shutdown()  # 关闭 RPC
# 确定用户是否为单个参数指定了多个条目，如果是，则为每个条目运行基准测试。
# 给定参数中的逗号分隔值表示多个条目。
# 输出设计成让用户可以使用绘图库将结果绘制到 x 轴上的每个变量参数的条目。
# 根据这些条目修改 Args。
# 不允许有多于一个参数具有多个条目。
def find_graph_variable(args):
    # 变量类型映射，指定了每个参数的预期类型或转换函数
    var_types = {
        "world_size": int,
        "state_size": str,
        "nlayers": int,
        "out_features": int,
        "batch": str2bool,
    }
    # 遍历每个参数类型
    for arg in var_types.keys():
        # 检查参数值中是否包含逗号
        if "," in args[arg]:
            # 如果 x_axis_name 已经存在，则抛出异常
            if args.get("x_axis_name"):
                raise ValueError("Only 1 x axis graph variable allowed")
            # 将逗号分隔的字符串转换成列表，并根据参数类型进行相应转换
            args[arg] = list(map(var_types[arg], args[arg].split(",")))
            # 设置 x_axis_name 以便后续绘图使用
            args["x_axis_name"] = arg
        else:
            # 如果参数值中没有逗号，则将其转换成预期的数据类型
            args[arg] = var_types[arg](args[arg])


# 返回一个修改后的字符串，其中末尾附加了空格。如果字符串参数的长度大于或等于指定长度，则附加一个空格；否则附加相应数量的空格，该数量等于字符串长度与长度参数之间的差值。
def append_spaces(string, length):
    string = str(string)
    offset = length - len(string)
    if offset <= 0:
        offset = 1
    # 附加指定数量的空格
    string += " " * offset
    return string


# 打印基准测试结果
def print_benchmark_results(report):
    print("--------------------------------------------------------------")
    print("PyTorch distributed rpc benchmark reinforcement learning suite")
    print("--------------------------------------------------------------")
    # 遍历报告中的键值对，打印除了 "benchmark_results" 之外的所有信息
    for key, val in report.items():
        if key != "benchmark_results":
            print(f"{key} : {val}")

    # 获取 x 轴的名称，如果存在的话
    x_axis_name = report.get("x_axis_name")
    col_width = 7
    heading = ""
    if x_axis_name:
        # 设置 x 轴输出标签，通过附加空格确保输出格式对齐
        x_axis_output_label = f"{x_axis_name} |"
        heading += append_spaces(x_axis_output_label, col_width)
    
    # 定义度量指标的标题
    metric_headers = [
        "agent latency (seconds)",
        "agent throughput",
        "observer latency (seconds)",
        "observer throughput",
    ]
    # 定义百分位子标题
    percentile_subheaders = ["p50", "p75", "p90", "p95"]
    subheading = ""
    if x_axis_name:
        # 根据 x_axis_output_label 的长度附加相应数量的空格，确保子标题对齐
        subheading += append_spaces(" " * (len(x_axis_output_label) - 1), col_width)
    # 对于每个指标标题，根据百分位子标题的数量计算并添加空格，构建表头
    for header in metric_headers:
        heading += append_spaces(header, col_width * len(percentile_subheaders))
        # 对于每个百分位子标题，计算并添加空格，构建副标题
        for percentile in percentile_subheaders:
            subheading += append_spaces(percentile, col_width)
    # 打印表头
    print(heading)
    # 打印副标题
    print(subheading)

    # 遍历报告中的每个基准运行结果
    for benchmark_run in report["benchmark_results"]:
        # 初始化存储每个运行结果的字符串
        run_results = ""
        # 如果存在 x 轴名称，将其添加到运行结果字符串中，并计算所需空格
        if x_axis_name:
            run_results += append_spaces(
                benchmark_run[x_axis_name], max(col_width, len(x_axis_output_label))
            )
        # 对于每个指标名称，获取百分位结果
        for metric_name in metric_headers:
            percentile_results = benchmark_run[metric_name]
            # 对于每个百分位子标题，将其结果添加到运行结果字符串中，并计算所需空格
            for percentile in percentile_subheaders:
                run_results += append_spaces(percentile_results[percentile], col_width)
        # 打印当前运行结果字符串
        print(run_results)
# 主函数，程序的入口点
def main():
    r"""
    如果没有参数有多个条目，则运行一次 rpc 基准测试；否则为每个多个条目运行一次。
    多个条目由逗号分隔的值指示，只能用于单个参数。
    结果将被打印并保存到输出文件中。在单个参数有多个条目的情况下，图表库可以用于将结果基准化在 y 轴上，每个条目在 x 轴上。
    """

    # 查找并设置图形变量
    find_graph_variable(args)

    # 如果没有 x 轴变量，则运行一次基准测试
    x_axis_variables = args[args["x_axis_name"]] if args.get("x_axis_name") else [None]

    # 使用 spawn 上下文创建进程池
    ctx = mp.get_context("spawn")
    queue = ctx.SimpleQueue()
    benchmark_runs = []

    # 遍历每个 x 轴变量，为每个变量运行基准测试
    for i, x_axis_variable in enumerate(x_axis_variables):
        if len(x_axis_variables) > 1:
            args[args["x_axis_name"]] = x_axis_variable  # 设置当前基准测试迭代的 x 轴变量

        processes = []
        start_time = time.time()

        # 启动多个进程执行 worker 的运行函数
        for rank in range(args["world_size"]):
            prc = ctx.Process(
                target=run_worker,
                args=(
                    rank,
                    args["world_size"],
                    args["master_addr"],
                    args["master_port"],
                    args["batch"],
                    args["state_size"],
                    args["nlayers"],
                    args["out_features"],
                    queue,
                ),
            )
            prc.start()
            processes.append(prc)

        # 等待所有进程完成
        benchmark_run_results = queue.get()
        for process in processes:
            process.join()

        # 打印基准测试运行时间
        print(f"Time taken benchmark run {i} -, {time.time() - start_time}")

        if args.get("x_axis_name"):
            # 将当前迭代的 x 轴值保存到基准测试结果中
            benchmark_run_results[args["x_axis_name"]] = x_axis_variable

        benchmark_runs.append(benchmark_run_results)

    # 构建报告，包括基准测试结果
    report = args
    report["benchmark_results"] = benchmark_runs

    if args.get("x_axis_name"):
        # 如果 x_axis_name 是变量，则在报告中不保存该变量的常量值
        del report[args["x_axis_name"]]

    # 将报告以 JSON 格式写入输出文件
    with open(args["output_file_path"], "w") as f:
        json.dump(report, f)

    # 打印基准测试结果
    print_benchmark_results(report)


# 如果当前脚本作为主程序运行，则调用主函数
if __name__ == "__main__":
    main()
```