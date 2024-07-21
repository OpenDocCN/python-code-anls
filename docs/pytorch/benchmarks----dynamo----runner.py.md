# `.\pytorch\benchmarks\dynamo\runner.py`

```
"""
A wrapper over the benchmark infrastructure to generate commonly used commands,
parse results and generate csv/graphs.

The script works on manually written TABLE (see below). We can add more commands
in the future.

One example usage is
-> python benchmarks/runner.py --suites=torchbench --inference
This command will generate the commands for the default compilers (see DEFAULTS
below) for inference, run them and visualize the logs.

If you want to just print the commands, you could use the following command
-> python benchmarks/runner.py --print-run-commands --suites=torchbench --inference

Similarly, if you want to just visualize the already finished logs
-> python benchmarks/runner.py --visualize-logs --suites=torchbench --inference

If you want to test float16
-> python benchmarks/runner.py --suites=torchbench --inference --dtypes=float16
"""


import argparse  # 导入处理命令行参数的模块
import dataclasses  # 导入用于创建不可变数据类的模块
import functools  # 导入用于创建高阶函数的模块
import glob  # 导入用于查找符合特定模式的文件路径名的模块
import importlib  # 导入用于动态加载模块的模块
import io  # 导入用于处理流的模块
import itertools  # 导入用于创建迭代器的模块
import logging  # 导入用于记录日志的模块
import os  # 导入与操作系统交互的模块
import re  # 导入用于处理正则表达式的模块
import shutil  # 导入用于文件操作的高级模块
import subprocess  # 导入用于执行外部命令的模块
import sys  # 导入与解释器交互的模块
import tempfile  # 导入用于创建临时文件和目录的模块
from collections import defaultdict  # 导入默认字典类的模块
from datetime import datetime, timedelta, timezone  # 导入处理日期和时间的模块
from os.path import abspath, exists  # 导入处理文件路径的函数
from random import randint  # 导入生成随机数的函数

import matplotlib.pyplot as plt  # 导入绘图库matplotlib

import numpy as np  # 导入数值计算和科学计算的库
import pandas as pd  # 导入数据处理和分析的库
from matplotlib import rcParams  # 导入matplotlib的配置参数类
from scipy.stats import gmean  # 导入几何平均函数
from tabulate import tabulate  # 导入用于生成表格的模块

import torch  # 导入PyTorch深度学习库

import torch._dynamo  # 导入torch._dynamo模块，可能是自定义扩展模块

rcParams.update({"figure.autolayout": True})  # 更新matplotlib的参数，自动布局图形
plt.rc("axes", axisbelow=True)  # 设置matplotlib图形的坐标轴在底层

DEFAULT_OUTPUT_DIR = "benchmark_logs"  # 默认的输出日志目录


log = logging.getLogger(__name__)  # 获取当前模块的日志记录器


TABLE = {
    "training": {
        "ts_nnc": "--training --speedup-ts ",  # 训练模式下的ts_nnc命令
        "ts_nvfuser": "--training --nvfuser --speedup-dynamo-ts ",  # 训练模式下的ts_nvfuser命令
        "eager": "--training --backend=eager ",  # 训练模式下的eager命令
        "aot_eager": "--training --backend=aot_eager ",  # 训练模式下的aot_eager命令
        "cudagraphs": "--training --backend=cudagraphs ",  # 训练模式下的cudagraphs命令
        "aot_nvfuser": "--training --nvfuser --backend=aot_ts_nvfuser ",  # 训练模式下的aot_nvfuser命令
        "nvprims_nvfuser": "--training --backend=nvprims_nvfuser ",  # 训练模式下的nvprims_nvfuser命令
        "inductor": "--training --inductor ",  # 训练模式下的inductor命令
        "inductor_no_cudagraphs": "--training --inductor --disable-cudagraphs ",  # 训练模式下的inductor_no_cudagraphs命令
        "inductor_max_autotune": "--training --inductor --inductor-compile-mode max-autotune ",  # 训练模式下的inductor_max_autotune命令
        "inductor_max_autotune_no_cudagraphs": (
            "--training --inductor --inductor-compile-mode max-autotune-no-cudagraphs --disable-cudagraphs "
        ),  # 训练模式下的inductor_max_autotune_no_cudagraphs命令
    },
    # 定义了一个名为 "inference" 的字典，包含了各种推断模式的命令行参数
    "inference": {
        # AOT eager 模式的推断参数
        "aot_eager": "--inference --backend=aot_eager ",
        # Eager 模式的推断参数
        "eager": "--inference --backend=eager ",
        # 使用 TS 加速的推断参数
        "ts_nnc": "--inference --speedup-ts ",
        # 使用 TS 和 NVFuser 加速的推断参数
        "ts_nvfuser": "--inference -n100 --speedup-ts --nvfuser ",
        # 使用 TensorRT 加速的推断参数
        "trt": "--inference -n100 --speedup-trt ",
        # 使用 cudagraphs_ts 后端的推断参数
        "ts_nvfuser_cudagraphs": "--inference --backend=cudagraphs_ts ",
        # Inductor 模式的推断参数
        "inductor": "--inference -n50 --inductor ",
        # 禁用 cudagraphs 后使用 Inductor 模式的推断参数
        "inductor_no_cudagraphs": "--inference -n50 --inductor --disable-cudagraphs ",
        # 使用 max-autotune 编译模式的 Inductor 模式推断参数
        "inductor_max_autotune": "--inference -n50 --inductor --inductor-compile-mode max-autotune ",
        # 使用 max-autotune-no-cudagraphs 编译模式的 Inductor 模式推断参数
        "inductor_max_autotune_no_cudagraphs": (
            "--inference -n50 --inductor --inductor-compile-mode max-autotune-no-cudagraphs --disable-cudagraphs "
        ),
        # 使用 TorchScript ONNX 的推断参数
        "torchscript-onnx": "--inference -n5 --torchscript-onnx",
        # 使用 Dynamo ONNX 的推断参数
        "dynamo-onnx": "--inference -n5 --dynamo-onnx",
    },
}

# 从全局变量 TABLE 中获取推理编译器的键，转换为元组
INFERENCE_COMPILERS = tuple(TABLE["inference"].keys())
# 从全局变量 TABLE 中获取训练编译器的键，转换为元组
TRAINING_COMPILERS = tuple(TABLE["training"].keys())

# 定义默认配置字典 DEFAULTS
DEFAULTS = {
    "training": [
        "eager",
        "aot_eager",
        "inductor",
        "inductor_no_cudagraphs",
    ],
    "inference": [
        "eager",
        "aot_eager",
        "inductor",
        "inductor_no_cudagraphs",
    ],
    "flag_compilers": {
        "training": ["inductor", "inductor_no_cudagraphs"],
        "inference": ["inductor", "inductor_no_cudagraphs"],
    },
    "dtypes": [
        "float32",
    ],
    "suites": ["torchbench", "huggingface", "timm_models"],
    "devices": [
        "cuda",
    ],
    "quick": {
        "torchbench": '-k "resnet..$"',
        "huggingface": "-k Albert",
        "timm_models": ' -k "^resnet" -k "^inception"',
    },
}

# 定义仪表盘默认配置字典 DASHBOARD_DEFAULTS
DASHBOARD_DEFAULTS = {
    "dashboard_image_uploader": "/fsx/users/anijain/bin/imgur.sh",
    "dashboard_archive_path": "/data/home/anijain/cluster/cron_logs",
    "dashboard_gh_cli_path": "/data/home/anijain/miniconda/bin/gh",
}

# 定义用于评估“加速度”标志的函数
def flag_speedup(x):
    return x < 0.95

# 定义用于评估“编译延迟”标志的函数
def flag_compilation_latency(x):
    return x > 120

# 定义用于评估“压缩比”标志的函数
def flag_compression_ratio(x):
    return x < 0.9

# 定义用于评估“准确性”标志的函数
def flag_accuracy(x):
    return "pass" not in x

# FLAG_FNS 字典存储评估函数，用于各种标志
FLAG_FNS = {
    "speedup": flag_speedup,
    "compilation_latency": flag_compilation_latency,
    "compression_ratio": flag_compression_ratio,
    "accuracy": flag_accuracy,
}

# 定义计算百分比的函数，接收部分值和总体值，并返回百分比
def percentage(part, whole, decimals=2):
    if whole == 0:
        return 0
    return round(100 * float(part) / float(whole), decimals)

# 定义解析命令行参数的函数
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--devices", action="append", help="cpu or cuda")
    parser.add_argument("--dtypes", action="append", help="float16/float32/amp")
    parser.add_argument("--suites", action="append", help="huggingface/torchbench/timm")
    parser.add_argument(
        "--compilers",
        action="append",
        help=f"For --inference, options are {INFERENCE_COMPILERS}. For --training, options are {TRAINING_COMPILERS}",
    )

    parser.add_argument(
        "--flag-compilers",
        action="append",
        help="List of compilers to flag issues. Same format as --compilers.",
    )
    parser.add_argument(
        "--quick", action="store_true", help="Just runs one model. Helps in debugging"
    )
    parser.add_argument(
        "--output-dir",
        help="Choose the output directory to save the logs",
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--keep-output-dir",
        action="store_true",
        help="Do not cleanup the output directory before running",
    )

    # 创建互斥组，用于选择生成命令或端到端运行
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument(
        "--print-run-commands",
        "--print_run_commands",
        action="store_true",
        help="Generate commands and saves them to run.sh",
    )
    # 添加一个命令行参数组，用于可视化日志
    group.add_argument(
        "--visualize-logs",
        "--visualize_logs",
        action="store_true",
        help="Pretty print the log files and draw graphs",
    )
    
    # 添加一个命令行参数，用于执行并解析文件
    group.add_argument(
        "--run",
        action="store_true",
        default=True,
        help="Generate commands, run and parses the files",
    )

    # 添加一个命令行参数，用于记录操作符的输入
    parser.add_argument(
        "--log-operator-inputs",
        action="store_true",
        default=False,
        help="Log operator inputs",
    )
    
    # 添加一个命令行参数，用于在性能提升报告中包含减速信息
    parser.add_argument(
        "--include-slowdowns",
        "--include_slowdowns",
        action="store_true",
        default=False,
        help="Include slowdowns in geomean performance speedup report. By default, slowdowns are ignored. "
        "This is because one can always use eager if compile is not speeding things up",
    )

    # 添加一个命令行参数，用于在命令行后附加额外的参数
    parser.add_argument(
        "--extra-args", default="", help="Append commandline with these args"
    )

    # 创建一个互斥选项组，要求选择推理或训练模式之一
    group_mode = parser.add_mutually_exclusive_group(required=True)
    
    # 向互斥选项组中添加推理模式的命令行参数
    group_mode.add_argument(
        "--inference", action="store_true", help="Only run inference related tasks"
    )
    
    # 向互斥选项组中添加训练模式的命令行参数
    group_mode.add_argument(
        "--training", action="store_true", help="Only run training related tasks"
    )

    # 添加一个命令行参数，用于指定被测试的 PyTorch 提交 ID
    parser.add_argument(
        "--base-sha",
        help="commit id for the tested pytorch",
    )
    
    # 添加一个命令行参数，用于指定分区的总数，将传递给实际的基准测试脚本
    parser.add_argument(
        "--total-partitions",
        type=int,
        help="Total number of partitions, to be passed to the actual benchmark script",
    )
    
    # 添加一个命令行参数，用于指定分区的 ID，将传递给实际的基准测试脚本
    parser.add_argument(
        "--partition-id",
        type=int,
        help="ID of partition, to be passed to the actual benchmark script",
    )

    # 添加一个命令行参数，用于更新仪表板
    parser.add_argument(
        "--update-dashboard",
        action="store_true",
        default=False,
        help="Updates to dashboard",
    )
    
    # 添加一个命令行参数，用于禁止生成和上传度量图表
    parser.add_argument(
        "--no-graphs",
        action="store_true",
        default=False,
        help="Do not genenerate and upload metric graphs",
    )
    
    # 添加一个命令行参数，用于禁止更新查找表和日志存档
    parser.add_argument(
        "--no-update-archive",
        action="store_true",
        default=False,
        help="Do not update lookup.csv or the log archive",
    )
    
    # 添加一个命令行参数，用于禁止向 GitHub 写入评论
    parser.add_argument(
        "--no-gh-comment",
        action="store_true",
        default=False,
        help="Do not write a comment to github",
    )
    
    # 添加一个命令行参数，用于禁止检测回归或指标图表
    parser.add_argument(
        "--no-detect-regressions",
        action="store_true",
        default=False,
        help="Do not compare to previous runs for regressions or metric graphs.",
    )
    
    # 添加一个命令行参数，用于执行 --no-graphs、--no-update-archive 和 --no-gh-comment 三个操作
    parser.add_argument(
        "--update-dashboard-test",
        action="store_true",
        default=False,
        help="does all of --no-graphs, --no-update-archive, and --no-gh-comment",
    )
    
    # 添加一个命令行参数，用于指定仪表板图像上传命令
    parser.add_argument(
        "--dashboard-image-uploader",
        default=DASHBOARD_DEFAULTS["dashboard_image_uploader"],
        help="Image uploader command",
    )
    parser.add_argument(
        "--dashboard-archive-path",
        default=DASHBOARD_DEFAULTS["dashboard_archive_path"],
        help="Archived directory path",
    )
    # 添加一个命令行参数 --dashboard-archive-path，用于指定存档目录的路径，默认值为预设的 dashboard_archive_path
    parser.add_argument(
        "--archive-name",
        help="Directory name under dashboard-archive-path to copy output-dir to. "
        "If not provided, a generated name is used.",
    )
    # 添加一个命令行参数 --archive-name，用于指定在 dashboard-archive-path 下将输出目录复制到的目录名。
    # 如果未提供，将使用生成的名称。
    parser.add_argument(
        "--dashboard-gh-cli-path",
        default=DASHBOARD_DEFAULTS["dashboard_gh_cli_path"],
        help="Github CLI path",
    )
    # 添加一个命令行参数 --dashboard-gh-cli-path，用于指定 GitHub CLI 的路径，默认值为预设的 dashboard_gh_cli_path
    parser.add_argument(
        "--batch-size",
        "--batch_size",
        type=int,
        default=None,
        help="batch size for benchmarking",
    )
    # 添加一个命令行参数 --batch-size（或 --batch_size），用于指定用于基准测试的批量大小，默认为 None
    parser.add_argument(
        "--threads",
        "-t",
        type=int,
        default=None,
        help="number of threads to use for eager and inductor.",
    )
    # 添加一个命令行参数 --threads（或 -t），用于指定用于 eager 和 inductor 的线程数
    launcher_group = parser.add_argument_group("CPU Launcher Parameters")
    # 创建一个参数组 "CPU Launcher Parameters"
    launcher_group.add_argument(
        "--enable-cpu-launcher",
        "--enable_cpu_launcher",
        action="store_true",
        default=False,
        help="Use torch.backends.xeon.run_cpu to get the peak performance on Intel(R) Xeon(R) Scalable Processors.",
    )
    # 在参数组 "CPU Launcher Parameters" 中添加一个命令行参数 --enable-cpu-launcher（或 --enable_cpu_launcher），
    # 用于启用 CPU 启动器，以获取在 Intel(R) Xeon(R) Scalable 处理器上的最高性能
    launcher_group.add_argument(
        "--cpu-launcher-args",
        "--cpu_launcher_args",
        type=str,
        default="",
        help="Provide the args of torch.backends.xeon.run_cpu. "
        "To look up what optional arguments this launcher offers: python -m torch.backends.xeon.run_cpu --help",
    )
    # 在参数组 "CPU Launcher Parameters" 中添加一个命令行参数 --cpu-launcher-args（或 --cpu_launcher_args），
    # 用于提供 torch.backends.xeon.run_cpu 的参数
    parser.add_argument(
        "--no-cold-start-latency",
        action="store_true",
        default=False,
        help="Do not include --cold-start-latency on inductor benchmarks",
    )
    # 添加一个命令行参数 --no-cold-start-latency，用于在 inductor 基准测试中不包括 --cold-start-latency
    parser.add_argument(
        "--inductor-compile-mode",
        default=None,
        help="torch.compile mode argument for inductor runs.",
    )
    # 添加一个命令行参数 --inductor-compile-mode，用于 inductor 运行的 torch.compile 模式参数
    args = parser.parse_args()
    # 解析所有的命令行参数
    return args
    # 返回解析后的命令行参数对象 args
# 根据命令行参数 args 获取模式（推断或训练），并返回对应的模式字符串
def get_mode(args):
    if args.inference:
        return "inference"
    return "training"


def get_skip_tests(suite, device, is_training: bool):
    """
    生成一个以 -x 分隔的字符串，用于跳过异常设置的训练测试
    """
    skip_tests = set()
    original_dir = abspath(os.getcwd())  # 获取当前工作目录的绝对路径
    module = importlib.import_module(suite)  # 动态导入指定模块
    os.chdir(original_dir)  # 切换回原始工作目录

    if suite == "torchbench":
        # 更新跳过的测试集合，根据 TorchBenchmarkRunner 类中定义的规则
        skip_tests.update(module.TorchBenchmarkRunner().skip_models)
        if is_training:
            skip_tests.update(
                module.TorchBenchmarkRunner().skip_not_suitable_for_training_models
            )
        if device == "cpu":
            skip_tests.update(module.TorchBenchmarkRunner().skip_models_for_cpu)
        elif device == "cuda":
            skip_tests.update(module.TorchBenchmarkRunner().skip_models_for_cuda)
    else:
        # 如果模块具有 SKIP 属性，则更新跳过的测试集合
        if hasattr(module, "SKIP"):
            skip_tests.update(module.SKIP)
        # 如果正在训练并且模块具有 SKIP_TRAIN 属性，则更新跳过的测试集合
        if is_training and hasattr(module, "SKIP_TRAIN"):
            skip_tests.update(module.SKIP_TRAIN)

    # 生成跳过测试的命令字符串生成器，每个测试名称前加上 "-x "
    skip_tests = (f"-x {name}" for name in skip_tests)
    skip_str = " ".join(skip_tests)  # 将生成的跳过测试命令字符串连接成一个完整字符串
    return skip_str  # 返回最终的跳过测试命令字符串


def generate_csv_name(args, dtype, suite, device, compiler, testing):
    mode = get_mode(args)  # 获取模式（推断或训练）
    # 根据给定的参数生成 CSV 文件名
    return f"{compiler}_{suite}_{dtype}_{mode}_{device}_{testing}.csv"


def generate_commands(args, dtypes, suites, devices, compilers, output_dir):
    mode = get_mode(args)  # 获取模式（推断或训练）
    suites_str = "_".join(suites)  # 将测试套件列表连接成字符串
    devices_str = "_".join(devices)  # 将设备列表连接成字符串
    dtypes_str = "_".join(dtypes)  # 将数据类型列表连接成字符串
    compilers_str = "_".join(compilers)  # 将编译器列表连接成字符串
    # 生成命令文件名，包含模式、设备、数据类型、测试套件和编译器信息
    generated_file = (
        f"run_{mode}_{devices_str}_{dtypes_str}_{suites_str}_{compilers_str}.sh"
    )
    # 使用 'w' 模式打开生成的文件，在上下文结束后自动关闭文件
    with open(generated_file, "w") as runfile:
        # 初始化一个空列表来存储每行的命令
        lines = []
    
        # 添加一条指定使用 Bash 解释器的 shebang 行
        lines.append("#!/bin/bash")
        # 启用 shell 脚本的调试模式，会输出每条命令及其扩展后的结果
        lines.append("set -x")
        # 设置输出目录的相关信息
        lines.append("# Setup the output directory")
        # 如果不保留输出目录，则删除该目录及其内容
        if not args.keep_output_dir:
            lines.append(f"rm -rf {output_dir}")
        # 如果输出目录已存在，则什么也不做（允许已存在的目录）
        lines.append(f"mkdir -p {output_dir}")
        lines.append("")
    
        # 遍历两个循环：testing 和 iter 中的组合
        for testing in ["performance", "accuracy"]:
            for iter in itertools.product(suites, devices, dtypes):
                suite, device, dtype = iter
                # 添加用于显示命令用途的注释，包括 suite、device、dtype、mode 和 testing 的信息
                lines.append(
                    f"# Commands for {suite} for device={device}, dtype={dtype} for {mode} and for {testing} testing"
                )
                # 获取特定 mode 的信息
                info = TABLE[mode]
                # 针对每种编译器生成命令
                for compiler in compilers:
                    base_cmd = info[compiler]
                    # 生成输出文件名
                    output_filename = f"{output_dir}/{generate_csv_name(args, dtype, suite, device, compiler, testing)}"
                    launcher_cmd = "python"
                    # 如果启用 CPU 启动器，则更改 launcher_cmd
                    if args.enable_cpu_launcher:
                        launcher_cmd = f"python -m torch.backends.xeon.run_cpu {args.cpu_launcher_args}"
                    # 组装完整的命令
                    cmd = f"{launcher_cmd} benchmarks/dynamo/{suite}.py --{testing} --{dtype} -d{device} --output={output_filename}"
                    cmd = f"{cmd} {base_cmd} {args.extra_args} --no-skip --dashboard"
                    # 获取适用于当前 suite 和 device 的跳过测试的字符串
                    skip_tests_str = get_skip_tests(suite, device, args.training)
                    cmd = f"{cmd} {skip_tests_str}"
    
                    # 如果启用了操作符输入日志记录，则添加相应的选项
                    if args.log_operator_inputs:
                        cmd = f"{cmd} --log-operator-inputs"
    
                    # 如果启用了快速模式，则添加默认的快速过滤器
                    if args.quick:
                        filters = DEFAULTS["quick"][suite]
                        cmd = f"{cmd} {filters}"
    
                    # 如果编译器是 "inductor" 或 "inductor_no_cudagraphs"，并且未禁用冷启动延迟，则添加相应的选项
                    if (
                        compiler
                        in (
                            "inductor",
                            "inductor_no_cudagraphs",
                        )
                        and not args.no_cold_start_latency
                    ):
                        cmd = f"{cmd} --cold-start-latency"
    
                    # 如果指定了批量大小，则添加相应的选项
                    if args.batch_size is not None:
                        cmd = f"{cmd} --batch-size {args.batch_size}"
    
                    # 如果指定了线程数，则添加相应的选项
                    if args.threads is not None:
                        cmd = f"{cmd} --threads {args.threads}"
    
                    # 如果指定了总分区数，则添加相应的选项
                    if args.total_partitions is not None:
                        cmd = f"{cmd} --total-partitions {args.total_partitions}"
    
                    # 如果指定了分区 ID，则添加相应的选项
                    if args.partition_id is not None:
                        cmd = f"{cmd} --partition-id {args.partition_id}"
    
                    # 如果指定了 Inductor 编译模式，则添加相应的选项
                    if args.inductor_compile_mode is not None:
                        cmd = f"{cmd} --inductor-compile-mode {args.inductor_compile_mode}"
                    # 将生成的命令添加到列表中
                    lines.append(cmd)
                # 在每个 suite/device/dtype 的组合之后添加一个空行
                lines.append("")
    
        # 将所有行写入到运行文件中
        runfile.writelines([line + "\n" for line in lines])
    
    # 返回生成的文件名
    return generated_file
# 生成一个下拉注释框的内容，包括标题和正文
def generate_dropdown_comment(title, body):
    # 创建一个字符串IO对象
    str_io = io.StringIO()
    # 写入标题到字符串IO对象
    str_io.write(f"{title}\n")
    # 写入下拉详情框的起始标签
    str_io.write("<details>\n")
    # 写入下拉详情框的摘要标签
    str_io.write("<summary>see more</summary>\n")
    # 将正文内容写入字符串IO对象
    str_io.write(f"{body}")
    # 写入换行符
    str_io.write("\n")
    # 写入下拉详情框的结束标签
    str_io.write("</details>\n\n")
    # 返回字符串IO对象的内容作为字符串
    return str_io.getvalue()


# 构建概要信息的函数，将各部分信息写入到输出字符串IO对象中
def build_summary(args):
    # 创建一个字符串IO对象
    out_io = io.StringIO()

    # 打印提交哈希值的函数
    def print_commit_hash(path, name):
        # 如果传入的参数中有基准SHA值
        if args.base_sha is not None:
            # 如果名称是"pytorch"，则将其提交SHA值写入输出字符串IO对象
            if name == "pytorch":
                out_io.write(f"{name} commit: {args.base_sha}\n")
        # 如果路径存在
        elif exists(path):
            # 导入git模块
            import git
            # 在指定路径下初始化一个Git仓库对象
            repo = git.Repo(path, search_parent_directories=True)
            # 获取当前HEAD的十六进制SHA值和提交日期
            sha = repo.head.object.hexsha
            date = repo.head.object.committed_datetime
            # 将名称和其提交SHA值写入输出字符串IO对象
            out_io.write(f"{name} commit: {sha}\n")
            # 将名称和其提交日期写入输出字符串IO对象
            out_io.write(f"{name} commit date: {date}\n")
        else:
            # 如果路径不存在，则将名称写为"Absent"
            out_io.write(f"{name} Absent\n")

    # 输出环境变量值的函数
    def env_var(name):
        # 如果名称在系统环境变量中
        if name in os.environ:
            # 将环境变量名和其对应的值写入输出字符串IO对象
            out_io.write(f"{name} = {os.environ[name]}\n")
        else:
            # 如果环境变量不存在，则将其值写为"None"
            out_io.write(f"{name} = {None}\n")

    # 写入空行到输出字符串IO对象
    out_io.write("\n")
    # 写入运行名称的标题到输出字符串IO对象
    out_io.write("### Run name ###\n")
    # 调用get_archive_name函数获取存档名称，并将其写入输出字符串IO对象
    out_io.write(get_archive_name(args, args.dtypes[0]))
    # 写入空行到输出字符串IO对象
    out_io.write("\n")

    # 写入提交哈希值的标题到输出字符串IO对象
    out_io.write("\n")
    out_io.write("### Commit hashes ###\n")
    # 获取并打印pytorch仓库的提交哈希值
    print_commit_hash("../pytorch", "pytorch")
    # 获取并打印torchbenchmark仓库的提交哈希值
    print_commit_hash("../torchbenchmark", "torchbench")

    # 写入空行到输出字符串IO对象
    out_io.write("\n")
    # 写入TorchDynamo配置标志的标题到输出字符串IO对象
    out_io.write("### TorchDynamo config flags ###\n")
    # 遍历torch._dynamo.config模块中的所有属性
    for key in dir(torch._dynamo.config):
        # 获取属性值
        val = getattr(torch._dynamo.config, key)
        # 如果属性不是私有属性且是布尔类型，则将其写入输出字符串IO对象
        if not key.startswith("__") and isinstance(val, bool):
            out_io.write(f"torch._dynamo.config.{key} = {val}\n")

    # 写入空行到输出字符串IO对象
    out_io.write("\n")
    # 写入Torch版本信息的标题到输出字符串IO对象
    out_io.write("### Torch version ###\n")
    # 将torch的版本号写入输出字符串IO对象
    out_io.write(f"torch: {torch.__version__}\n")

    # 写入空行到输出字符串IO对象
    out_io.write("\n")
    # 写入环境变量信息的标题到输出字符串IO对象
    out_io.write("### Environment variables ###\n")
    # 获取并打印TORCH_CUDA_ARCH_LIST环境变量的值
    env_var("TORCH_CUDA_ARCH_LIST")
    # 获取并打印CUDA_HOME环境变量的值
    env_var("CUDA_HOME")
    # 获取并打印USE_LLVM环境变量的值
    env_var("USE_LLVM")

    # 如果参数中包含"cuda"设备
    if "cuda" in args.devices:
        # 写入空行到输出字符串IO对象
        out_io.write("\n")
        # 写入GPU详细信息的标题到输出字符串IO对象
        out_io.write("### GPU details ###\n")
        # 获取并打印当前CUDNN版本号
        out_io.write(f"CUDNN VERSION: {torch.backends.cudnn.version()}\n")
        # 获取并打印CUDA设备的数量
        out_io.write(f"Number CUDA Devices: {torch.cuda.device_count()}\n")
        # 获取并打印第一个CUDA设备的名称
        out_io.write(f"Device Name: {torch.cuda.get_device_name(0)}\n")
        # 获取并打印第一个CUDA设备的总内存大小（以GB为单位）
        out_io.write(
            f"Device Memory [GB]: {torch.cuda.get_device_properties(0).total_memory/1e9}\n"
        )

    # 设置标题文本
    title = "## Build Summary"
    # 调用generate_dropdown_comment函数生成下拉注释框的内容
    comment = generate_dropdown_comment(title, out_io.getvalue())
    # 打开gh_build_summary.txt文件，并将生成的注释内容写入文件
    with open(f"{output_dir}/gh_build_summary.txt", "w") as gh_fh:
        gh_fh.write(comment)
    # 如果给定了归档名称，则进行以下处理
    if archive_name is not None:
        # 在归档名称中查找匹配 "_performance" 前的单词字符序列
        prefix_match = re.search(r"\w+(?=_performance)", archive_name)
        # 如果找到匹配项，则将匹配到的内容作为前缀
        if prefix_match is not None:
            prefix = prefix_match.group(0)
        else:
            # 如果未找到匹配项，则前缀为空字符串
            prefix = ""
        # 在归档名称中查找匹配 "day_<数字>_" 的部分，提取其中的数字
        day_match = re.search(r"day_(\d+)_", archive_name)
        # 如果找到匹配项，则将匹配到的数字作为当天的代号
        if day_match is not None:
            day = day_match.group(1)
        else:
            # 如果未找到匹配项，则将日期设置为 "000"
            day = "000"
    else:
        # 如果未提供归档名称，则获取当前时间
        now = datetime.now(tz=timezone(timedelta(hours=-8)))
        # 将当前日期转换为年内的第几天的格式，并作为当天的代号
        day = now.strftime("%j")
        # 使用当前日期格式化字符串作为前缀
        prefix = now.strftime(f"day_{day}_%d_%m_%y")
    # 返回处理后的当天代号和前缀
    return day, prefix
# 使用 functools 模块的 lru_cache 装饰器，对函数进行结果缓存，不限制缓存大小
@functools.lru_cache(None)
# 默认的归档名称生成函数，接受一个数据类型参数 dtype
def default_archive_name(dtype):
    # 调用 archive_data 函数获取前缀，忽略第一个返回值
    _, prefix = archive_data(None)
    # 返回生成的归档名称，包括前缀、性能相关标识、随机数（100到999之间）
    return f"{prefix}_performance_{dtype}_{randint(100, 999)}"


# 获取归档名称的函数，根据参数决定返回默认生成的名称或用户指定的名称
def get_archive_name(args, dtype):
    return (
        default_archive_name(dtype) if args.archive_name is None else args.archive_name
    )


# 归档函数，将源目录的内容复制到目标目录下特定名称的归档文件夹中
def archive(src_dir, dest_dir_prefix, archive_name, dtype):
    # 如果未提供归档名称，使用默认生成的归档名称
    if archive_name is None:
        archive_name = default_archive_name(dtype)
    # 拼接目标目录和归档名称路径
    dest = os.path.join(dest_dir_prefix, archive_name)
    # 使用 shutil 模块复制整个目录树到目标路径，允许目标目录存在
    shutil.copytree(src_dir, dest, dirs_exist_ok=True)
    # 打印复制操作的信息
    print(f"copied contents of {src_dir} to {dest}")


# 根据指标名称返回对应的标题
def get_metric_title(metric):
    if metric == "speedup":
        return "Performance speedup"
    elif metric == "accuracy":
        return "Accuracy"
    elif metric == "compilation_latency":
        return "Compilation latency (sec)"
    elif metric == "compression_ratio":
        return "Peak Memory Compression Ratio"
    elif metric == "abs_latency":
        return "Absolute latency (ms)"
    # 如果未知指标名称，引发运行时异常
    raise RuntimeError("unknown metric")


# 解析器类，初始化时接受多个参数，用于处理性能日志的解析
class Parser:
    def __init__(
        self, suites, devices, dtypes, compilers, flag_compilers, mode, output_dir
    ):
        # 初始化实例变量，存储测试套件、设备列表、数据类型、编译器、编译器标志、模式和输出目录
        self.suites = suites
        self.devices = devices
        self.dtypes = dtypes
        self.compilers = compilers
        self.flag_compilers = flag_compilers
        self.output_dir = output_dir
        self.mode = mode

    # 检查输出文件是否具有头部信息的方法
    def has_header(self, output_filename):
        # 默认没有找到头部信息
        header_present = False
        # 打开输出文件，逐行读取
        with open(output_filename) as f:
            line = f.readline()
            # 如果某行包含 "dev" 字符串，认为找到了头部信息
            if "dev" in line:
                header_present = True
        # 返回是否找到头部信息的布尔值
        return header_present


# 继承 Parser 类，用于解析性能日志的子类
class ParsePerformanceLogs(Parser):
    def __init__(
        self,
        suites,
        devices,
        dtypes,
        compilers,
        flag_compilers,
        mode,
        output_dir,
        include_slowdowns=False,
    ):
        # 调用父类的初始化方法，设置基本参数
        super().__init__(
            suites,
            devices,
            dtypes,
            compilers,
            flag_compilers,
            mode,
            output_dir,
        )
        # 初始化实例变量，存储解析后的帧数据、未处理的解析帧数据、指标列表等
        self.parsed_frames = defaultdict(lambda: defaultdict(None))
        self.untouched_parsed_frames = defaultdict(lambda: defaultdict(None))
        self.metrics = [
            "speedup",
            "abs_latency",
            "compilation_latency",
            "compression_ratio",
        ]
        self.bottom_k = 50
        # 执行日志解析的方法
        self.parse()
        # 是否包括减速信息的标志，默认为 False
        self.include_slowdowns = include_slowdowns
    def plot_graph(self, df, title):
        # 获取数据框的列标签列表，排除前三列（假设是name, batch_size, speedup）
        labels = df.columns.values.tolist()
        labels = labels[3:]
        # 使用 pandas 的 plot 方法绘制条形图
        df.plot(
            x="name",  # x 轴为数据框的 name 列
            y=labels,  # y 轴为除了前三列的所有列
            kind="bar",  # 绘制条形图
            width=0.65,  # 条形宽度
            title=title,  # 图表标题
            ylabel="Speedup over eager",  # y 轴标签
            xlabel="",  # x 轴标签为空
            grid=True,  # 显示网格线
            figsize=(max(len(df.index) / 4, 5), 10),  # 图表尺寸
            edgecolor="black",  # 条形边缘颜色
        )
        plt.tight_layout()  # 调整布局以防止重叠
        plt.savefig(f"{self.output_dir}/{title}.png")  # 保存图表到指定路径

    def read_csv(self, output_filename):
        if self.has_header(output_filename):  # 如果文件包含头部信息
            return pd.read_csv(output_filename)  # 直接读取 CSV 文件
        else:
            return pd.read_csv(
                output_filename,  # 文件名
                names=[  # 列名列表
                    "dev",
                    "name",
                    "batch_size",
                    "speedup",
                    "abs_latency",
                    "compilation_latency",
                    "compression_ratio",
                ],
                header=None,  # 没有头部信息
                engine="python",  # 使用 Python 引擎解析
            )

    def parse(self):
        self.extract_df("accuracy", "accuracy")  # 提取 "accuracy" 数据框
        for metric in self.metrics:  # 对每个指标
            self.extract_df(metric, "performance")  # 提取 "performance" 数据框

    def clean_batch_sizes(self, frames):
        # 清理批次大小为 0 的情况
        if len(frames) == 1:  # 如果只有一个数据框
            return frames  # 直接返回
        batch_sizes = frames[0]["batch_size"].to_list()  # 获取第一个数据框的批次大小列表
        for frame in frames[1:]:  # 遍历剩余的数据框
            frame_batch_sizes = frame["batch_size"].to_list()  # 获取当前数据框的批次大小列表
            for idx, (batch_a, batch_b) in enumerate(
                zip(batch_sizes, frame_batch_sizes)
            ):
                assert batch_a == batch_b or batch_a == 0 or batch_b == 0, print(  # 断言条件
                    f"a={batch_a}, b={batch_b}"  # 打印错误信息
                )
                batch_sizes[idx] = max(batch_a, batch_b)  # 更新批次大小列表
        for frame in frames:  # 对每个数据框
            frame["batch_size"] = batch_sizes  # 更新批次大小列
        return frames  # 返回更新后的数据框列表

    def get_passing_entries(self, compiler, df):
        return df[compiler][df[compiler] > 0]  # 返回指定编译器列中大于 0 的条目

    def comp_time(self, compiler, df):
        df = self.get_passing_entries(compiler, df)  # 获取通过的条目
        if df.empty:  # 如果数据框为空
            return "0.0"  # 返回默认值
        return f"{df.mean():.2f}"  # 返回均值格式化字符串

    def geomean(self, compiler, df):
        cleaned_df = self.get_passing_entries(compiler, df)  # 获取通过的条目
        if not self.include_slowdowns:  # 如果不包含减速情况
            cleaned_df = cleaned_df.clip(1)  # 将数据框裁剪到最小值为 1
        if cleaned_df.empty:  # 如果数据框为空
            return "0.0x"  # 返回默认值
        return f"{gmean(cleaned_df):.2f}x"  # 返回几何平均值格式化字符串

    def passrate(self, compiler, df):
        total = len(df.index)  # 总条目数
        passing = df[df[compiler] > 0.0][compiler].count()  # 通过的条目数
        perc = int(percentage(passing, total, decimals=0))  # 计算通过率百分比
        return f"{perc}%, {passing}/{total}"  # 返回通过率字符串
    def memory(self, compiler, df):
        # 调用内部方法获取通过编译器筛选后的数据框
        df = self.get_passing_entries(compiler, df)
        # 将数据框中的NaN值填充为0
        df = df.fillna(0)
        # 从数据框中选择大于0的行
        df = df[df > 0]
        # 如果数据框为空，则返回字符串"0.0x"
        if df.empty:
            return "0.0x"
        # 返回数据框中平均值的格式化字符串，保留两位小数，并附加"x"
        return f"{df.mean():.2f}x"

    def exec_summary_df(self, fn, metric):
        """
        生成包含通过率和几何平均性能的表格
        """
        # 创建列字典，初始化"Compiler"键为self.compilers
        cols = {}
        cols["Compiler"] = self.compilers
        # 遍历self.suites中的每个测试套件
        for suite in self.suites:
            # 从self.parsed_frames[suite][metric]获取数据框df
            df = self.parsed_frames[suite][metric]
            # 为每个编译器调用fn函数，计算性能加速比，并存储为列表speedups
            speedups = [fn(compiler, df) for compiler in self.compilers]
            # 创建以编译器为索引的pd.Series对象col，数据为speedups
            col = pd.Series(data=speedups, index=self.compilers)
            # 将该列添加到cols字典中，以suite命名
            cols[suite] = col
        # 使用cols创建DataFrame对象df
        df = pd.DataFrame(cols)
        # 将数据框中的NaN值填充为0
        df = df.fillna(0)
        # 将df保存为CSV文件，文件名为函数fn的名称，存放在self.output_dir目录下
        df.to_csv(os.path.join(self.output_dir, f"{fn.__name__}.csv"))
        # 返回生成的DataFrame对象df
        return df

    def exec_summary_text(self, caption, fn, metric):
        # 调用exec_summary_df函数生成汇总数据的数据框df
        df = self.exec_summary_df(fn, metric)
        # 使用tabulate函数将df转换为漂亮的表格形式tabform，不显示索引
        tabform = tabulate(df, headers="keys", tablefmt="pretty", showindex="never")

        # 创建字符串IO对象str_io
        str_io = io.StringIO()
        # 将标题caption写入str_io
        str_io.write(f"{caption}")
        # 写入分隔符"~~~\n"
        str_io.write("~~~\n")
        # 将tabform写入str_io
        str_io.write(f"{tabform}\n")
        # 写入结束分隔符"~~~\n"
        str_io.write("~~~\n")
        # 返回str_io中的全部内容作为字符串
        return str_io.getvalue()
    # 定义生成执行摘要的方法，该方法属于类的一部分
    def generate_executive_summary(self):
        # 默认机器设备为"A100 GPUs"
        machine = "A100 GPUs"
        
        # 如果在self.devices中发现"cpu"，则获取本地机器的CPU型号信息
        if "cpu" in self.devices:
            get_machine_cmd = "lscpu| grep 'Model name' | awk -F':' '{print $2}'"
            # 执行获取CPU型号信息的命令，并从命令输出中提取信息并去除首尾空白字符
            machine = subprocess.getstatusoutput(get_machine_cmd)[1].strip()
        
        # 定义摘要描述信息，包括评估不同后端在三个基准测试套件上的性能
        description = (
            "We evaluate different backends "
            "across three benchmark suites - torchbench, huggingface and timm. We run "
            "these experiments on "
            + machine
            + ". Each experiment runs one iteration of forward pass "
            "and backward pass for training and forward pass only for inference. "
            "For accuracy, we check the numerical correctness of forward pass outputs and gradients "
            "by comparing with native pytorch. We measure speedup "
            "by normalizing against the performance of native pytorch. We report mean "
            "compilation latency numbers and peak memory footprint reduction ratio. \n\n"
            "Caveats\n"
            "1) Batch size has been reduced to workaround OOM errors. Work is in progress to "
            "reduce peak memory footprint.\n"
            "2) Experiments do not cover dynamic shapes.\n"
            "3) Experimental setup does not have optimizer.\n\n"
        )
        
        # 使用生成下拉式评论的函数，生成基于描述的评论文本
        comment = generate_dropdown_comment("", description)
        
        # 创建一个字符串IO对象，用于构建执行摘要的字符串内容
        str_io = io.StringIO()
        str_io.write("\n")
        str_io.write("## Executive Summary ##\n")
        str_io.write(comment)

        # 生成速度提升的总结文本
        speedup_caption = "Geometric mean speedup \n"
        speedup_summary = self.exec_summary_text(
            speedup_caption, self.geomean, "speedup"
        )

        # 生成通过率的总结文本
        passrate_caption = "Passrate\n"
        passrate_summary = self.exec_summary_text(
            passrate_caption, self.passrate, "speedup"
        )

        # 生成平均编译时间的总结文本
        comp_time_caption = "Mean compilation time (seconds)\n"
        comp_time_summary = self.exec_summary_text(
            comp_time_caption, self.comp_time, "compilation_latency"
        )

        # 生成峰值内存占用的总结文本
        peak_memory_caption = (
            "Peak memory footprint compression ratio (higher is better)\n"
        )
        peak_memory_summary = self.exec_summary_text(
            peak_memory_caption, self.memory, "compression_ratio"
        )

        # 写入关于性能、编译延迟和内存占用的描述信息
        str_io.write(
            "To measure performance, compilation latency and memory footprint reduction, "
            "we remove the models that fail accuracy checks.\n\n"
        )
        # 将各总结文本逐一写入字符串IO对象
        str_io.write(passrate_summary)
        str_io.write(speedup_summary)
        str_io.write(comp_time_summary)
        str_io.write(peak_memory_summary)
        
        # 将字符串IO对象中的内容作为执行摘要结果保存在self.executive_summary中
        self.executive_summary = str_io.getvalue()
    # 根据给定的suite和metric，从self.untouched_parsed_frames中获取DataFrame
    df = self.untouched_parsed_frames[suite][metric]
    # 删除DataFrame中名为"dev"的列
    df = df.drop("dev", axis=1)
    # 将DataFrame的列名"batch_size"重命名为"bs"
    df = df.rename(columns={"batch_size": "bs"})
    
    # 对self.flag_compilers列应用flag_fn函数，逐元素进行标记
    # 如果一行中有任何元素失败，则整行被标记
    flag = np.logical_or.reduce(
        df[self.flag_compilers].applymap(flag_fn),
        axis=1,
    )
    # 根据标记过滤DataFrame中的行
    df = df[flag]
    
    # 在DataFrame中新增一列'suite'，赋值为参数suite的值
    df = df.assign(suite=suite)
    
    # 重新索引DataFrame的列，只包括"suite", "name"和self.flag_compilers
    return df.reindex(columns=["suite", "name"] + self.flag_compilers)
    # 定义生成摘要文件的方法，该方法属于某个类的成员函数
    def gen_summary_files(self):
        # 调用对象的方法生成执行摘要
        self.generate_executive_summary()
        # 遍历所有测试套件
        for suite in self.suites:
            # 调用对象的方法，绘制速度提升图表，并写入文件
            self.plot_graph(
                self.untouched_parsed_frames[suite]["speedup"],  # 提取特定测试套件中的速度提升数据
                f"{suite}_{self.dtypes[0]}",  # 生成文件名，包含测试套件名称和数据类型
            )

        # 打开一个文件用于写入 GitHub 页面的标题
        with open(f"{self.output_dir}/gh_title.txt", "w") as gh_fh:
            # 创建一个字符串 IO 对象
            str_io = io.StringIO()
            # 写入标题内容，包括数据类型的精度信息
            str_io.write("\n")
            str_io.write(f"# Performance Dashboard for {self.dtypes[0]} precision ##\n")
            str_io.write("\n")
            # 将字符串 IO 对象的内容写入文件
            gh_fh.write(str_io.getvalue())

        # 打开一个文件用于写入 GitHub 页面的执行摘要
        with open(f"{self.output_dir}/gh_executive_summary.txt", "w") as gh_fh:
            # 将执行摘要内容直接写入文件
            gh_fh.write(self.executive_summary)

        # 打开一个文件用于写入 GitHub 页面的警告信息
        with open(f"{self.output_dir}/gh_warnings.txt", "w") as gh_fh:
            # 调用对象的方法生成警告信息，并将其写入文件
            warnings_body = self.generate_warnings()
            gh_fh.write(warnings_body)

        # 创建一个字符串 IO 对象，用于准备每个测试套件的消息内容
        str_io = io.StringIO()
        for suite in self.suites:
            # 调用对象的方法准备每个测试套件的消息内容，并写入字符串 IO 对象
            str_io.write(self.prepare_message(suite))
        str_io.write("\n")
        # 打开一个文件用于写入 GitHub 页面的模式相关信息
        with open(f"{self.output_dir}/gh_{self.mode}.txt", "w") as gh_fh:
            # 将字符串 IO 对象的内容写入文件
            gh_fh.write(str_io.getvalue())
def parse_logs(args, dtypes, suites, devices, compilers, flag_compilers, output_dir):
    # 获取运行模式（args 中的模式信息）
    mode = get_mode(args)
    # 生成构建摘要信息（输出到日志中）
    build_summary(args)
    # 是否包含性能变慢信息（根据参数判断）
    include_slowdowns = args.include_slowdowns

    # 使用 ParsePerformanceLogs 类解析性能日志
    parser_class = ParsePerformanceLogs
    # 实例化解析器对象
    parser = parser_class(
        suites,
        devices,
        dtypes,
        compilers,
        flag_compilers,
        mode,
        output_dir,
        include_slowdowns,
    )
    # 生成摘要文件
    parser.gen_summary_files()
    return


@dataclasses.dataclass
class LogInfo:
    # 日志生成的年中的第几天
    day: str

    # 所有日志所在的目录路径
    dir_path: str


def get_date(log_info):
    # 根据日志信息返回日期（从年中的第几天转换为日期格式）
    return datetime.strptime(f"{log_info.day}", "%j").strftime("%m-%d")


def find_last_2_with_filenames(lookup_file, dashboard_archive_path, dtype, filenames):
    # 从 lookup_file 中读取数据并筛选出性能模式相关的条目
    df = pd.read_csv(lookup_file, names=("day", "mode", "prec", "path"))
    df = df[df["mode"] == "performance"]
    df = df[df["prec"] == dtype]
    # 将 DataFrame 反转顺序
    df = df[::-1]
    last2 = []
    for path in df["path"]:
        # 构建完整的输出目录路径
        output_dir = os.path.join(dashboard_archive_path, path)
        # 构建完整的文件路径列表
        fullpaths = [
            os.path.join(dashboard_archive_path, path, name) for name in filenames
        ]
        # 如果所有文件都存在，则将路径添加到 last2 列表中
        if all(os.path.exists(fullpath) for fullpath in fullpaths):
            last2.append(output_dir)
        # 如果已经找到两个符合条件的路径，则返回结果
        if len(last2) >= 2:
            return last2
    # 如果找不到符合条件的路径，则返回 None
    return None


class SummaryStatDiffer:
    def __init__(self, args):
        # 初始化 SummaryStatDiffer 类的参数
        self.args = args
        # 构建 lookup 文件的完整路径
        self.lookup_file = os.path.join(self.args.dashboard_archive_path, "lookup.csv")
        # 确保 lookup 文件存在
        assert os.path.exists(self.lookup_file)

    def generate_diff(self, last2, filename, caption):
        # 从最近两个路径中读取当前和先前的数据文件，并合并
        df_cur, df_prev = (pd.read_csv(os.path.join(path, filename)) for path in last2)
        df_merge = df_cur.merge(df_prev, on="Compiler", suffixes=("_cur", "_prev"))
        # 准备用于存储差异数据的字典
        data = {col: [] for col in ("compiler", "suite", "prev_value", "cur_value")}
        # 遍历合并后的数据框的行
        for _, row in df_merge.iterrows():
            # 如果编译器在标志编译器列表中
            if row["Compiler"] in self.args.flag_compilers:
                # 遍历套件列表
                for suite in self.args.suites:
                    # 如果当前套件的先前值和当前值均存在
                    if suite + "_prev" not in row or suite + "_cur" not in row:
                        continue
                    # 将编译器、套件以及先前值和当前值添加到数据字典中
                    data["compiler"].append(row["Compiler"])
                    data["suite"].append(suite)
                    data["prev_value"].append(row[suite + "_prev"])
                    data["cur_value"].append(row[suite + "_cur"])

        # 创建包含差异数据的 DataFrame
        df = pd.DataFrame(data)
        # 使用 tabulate 格式化 DataFrame 并生成表格形式的字符串
        tabform = tabulate(df, headers="keys", tablefmt="pretty", showindex="never")
        # 创建字符串 IO 对象，用于输出表格和标题
        str_io = io.StringIO()
        str_io.write("\n")
        str_io.write(f"{caption}\n")
        str_io.write("~~~\n")
        str_io.write(f"{tabform}\n")
        str_io.write("~~~\n")
        # 返回包含表格和标题的字符串
        return str_io.getvalue()
    # 定义一个方法用于生成注释
    def generate_comment(self):
        # 创建标题字符串，指定了摘要统计差异
        title = "## Summary Statistics Diff ##\n"
        # 创建正文字符串，描述了每个相关编译器的摘要统计比较
        body = (
            "For each relevant compiler, we compare the summary statistics "
            "for the most 2 recent reports that actually run the compiler.\n\n"
        )
        # 从参数中获取数据类型列表的第一个数据类型
        dtype = self.args.dtypes[0]
        # 查找包含最近两个文件名的路径，根据文件名找到最新的两个相关报告
        last2 = find_last_2_with_filenames(
            self.lookup_file,
            self.args.dashboard_archive_path,
            dtype,
            ["geomean.csv", "passrate.csv"],
        )

        # 如果找不到最近的两个报告
        if last2 is None:
            # 添加一条消息说明未找到最近的两个报告
            body += "Could not find most 2 recent reports.\n\n"
        else:
            # 对于每个状态（当前和上一个），追加报告的名称
            for state, path in zip(("Current", "Previous"), last2):
                body += f"{state} report name: {path}\n\n"
            # 生成并追加“passrate.csv”的差异报告
            body += self.generate_diff(last2, "passrate.csv", "Passrate diff")
            # 生成并追加“geomean.csv”的差异报告
            body += self.generate_diff(
                last2, "geomean.csv", "Geometric mean speedup diff"
            )

        # 生成下拉菜单风格的注释
        comment = generate_dropdown_comment(title, body)

        # 将生成的注释写入指定的输出文件
        with open(f"{self.args.output_dir}/gh_summary_diff.txt", "w") as gh_fh:
            gh_fh.write(comment)
# 定义 RegressionDetector 类，用于比较最近的两个基准测试，找出之前未标记但现在标记的模型
class RegressionDetector:
    """
    Compares the most recent 2 benchmarks to find previously unflagged models
    that are now flagged.
    """

    # 初始化方法，接收参数 args
    def __init__(self, args):
        self.args = args
        # 拼接 lookup.csv 文件的路径
        self.lookup_file = os.path.join(self.args.dashboard_archive_path, "lookup.csv")
        # 断言确保 lookup.csv 文件存在
        assert os.path.exists(self.lookup_file)

# 定义 RegressionTracker 类，用于绘制不同指标随时间的进展，以检测回归
class RegressionTracker:
    """
    Plots progress of different metrics over time to detect regressions.
    """

    # 初始化方法，接收参数 args
    def __init__(self, args):
        self.args = args
        self.suites = self.args.suites  # 将参数 args 中的 suites 赋值给实例变量 suites
        # 拼接 lookup.csv 文件的路径
        self.lookup_file = os.path.join(self.args.dashboard_archive_path, "lookup.csv")
        # 断言确保 lookup.csv 文件存在
        assert os.path.exists(self.lookup_file)
        self.k = 10  # 初始化实例变量 k 为 10

    # 查找最后的 k 个 (day number, log_path) 对
    def find_last_k(self):
        """
        Find the last k pairs of (day number, log_path)
        """
        dtype = self.args.dtypes[0]  # 获取参数 args 中的第一个数据类型
        # 从 lookup.csv 中读取数据，并筛选出 mode 为 "performance"，prec 等于 dtype 的行
        df = pd.read_csv(self.lookup_file, names=("day", "mode", "prec", "path"))
        df = df[df["mode"] == "performance"]
        df = df[df["prec"] == dtype]
        log_infos = []
        # 根据筛选后的数据创建 LogInfo 对象的列表
        for day, path in zip(df["day"], df["path"]):
            log_infos.append(LogInfo(day, path))

        # 断言确保 log_infos 的长度大于等于 k
        assert len(log_infos) >= self.k
        # 取出 log_infos 中最后的 k 个元素作为结果
        log_infos = log_infos[len(log_infos) - self.k :]
        return log_infos

    # 生成评论信息
    def generate_comment(self):
        title = "## Metrics over time ##\n"  # 设置标题
        str_io = io.StringIO()  # 创建一个字符串流对象 str_io
        # 如果不需要更新仪表板测试且不禁用图形，则遍历输出目录中所有 "*over_time.png" 文件
        if not self.args.update_dashboard_test and not self.args.no_graphs:
            for name in glob.glob(self.args.output_dir + "/*over_time.png"):
                # 调用 dashboard_image_uploader 工具上传图片，并获取输出结果
                output = (
                    subprocess.check_output([self.args.dashboard_image_uploader, name])
                    .decode("ascii")
                    .rstrip()
                )
                # 将图片路径和上传后的输出信息写入 str_io
                str_io.write(f"\n{name} : ![]({output})\n")
        # 调用 generate_dropdown_comment 函数生成下拉评论框内容
        comment = generate_dropdown_comment(title, str_io.getvalue())

        # 将评论写入文件 gh_regression.txt 中
        with open(f"{self.args.output_dir}/gh_regression.txt", "w") as gh_fh:
            gh_fh.write(comment)
    # 定义一个名为 diff 的方法，用于执行指定操作
    def diff(self):
        # 调用实例方法 find_last_k() 获取日志信息
        log_infos = self.find_last_k()

        # 遍历指定的指标列表：几何平均值、通过率、比较时间、内存使用率
        for metric in ["geomean", "passrate", "comp_time", "memory"]:
            # 创建一个新的图形 fig 和轴数组 axes，以显示数据可视化结果
            fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
            
            # 遍历实例变量 self.suites 中的每个测试套件
            for idx, suite in enumerate(self.suites):
                # 初始化空的数据框列表
                dfs = []
                
                # 遍历 log_infos 中的日志信息
                for log_info in log_infos:
                    # 构建目录路径，其中包含 log_info.dir_path
                    dir_path = os.path.join(
                        self.args.dashboard_archive_path, log_info.dir_path
                    )
                    # 断言目录路径存在
                    assert os.path.exists(dir_path)
                    
                    # 构建完整的文件名 gmean_filename
                    gmean_filename = os.path.join(dir_path, f"{metric}.csv")
                    
                    # 如果 gmean_filename 文件不存在，则跳过当前循环
                    if not os.path.exists(gmean_filename):
                        continue
                    
                    # 从 gmean_filename 中读取 CSV 文件并创建数据框 df
                    df = pd.read_csv(gmean_filename)
                    
                    # 如果当前测试套件 suite 不在 df 的列中，则跳过当前循环
                    if suite not in df:
                        continue
                    
                    # 根据指标 metric 的不同类型进行处理 df[suite] 的值
                    if metric == "geomean" or metric == "memory":
                        df[suite] = df[suite].str.replace("x", "").astype(float)
                    elif metric == "passrate":
                        df[suite] = df[suite].str.split("%").str[0].astype(float)
                    
                    # 在数据框 df 中插入一列 "day"，其值为 get_date(log_info) 的结果
                    df.insert(0, "day", get_date(log_info))
                    
                    # 根据 "day" 和 "Compiler" 列，将 df 重新组织成新的数据框
                    df = df.pivot(index="day", columns="Compiler", values=suite)

                    # 将数据框 df 中的列名 "inductor_cudagraphs" 更名为 "inductor"
                    df = df.rename(columns={"inductor_cudagraphs": "inductor"})
                    
                    # 遍历数据框 df 的每一列名
                    for col_name in df.columns:
                        # 如果 col_name 不在 self.args.compilers 中，则从 df 中删除该列
                        if col_name not in self.args.compilers:
                            df = df.drop(columns=[col_name])
                    
                    # 将处理后的 df 添加到 dfs 列表中
                    dfs.append(df)

                # 将 dfs 列表中的所有数据框连接起来形成一个新的数据框 df
                df = pd.concat(dfs)
                
                # 对数据框 df 进行线性插值处理
                df = df.interpolate(method="linear")
                
                # 在当前轴 axes[idx] 上绘制 df 的线性图
                ax = df.plot(
                    ax=axes[idx],
                    kind="line",
                    ylabel=metric,
                    xlabel="Date",
                    grid=True,
                    ylim=0 if metric == "passrate" else 0.8,
                    title=suite,
                    style=".-",
                    legend=False,
                )
                
                # 在当前轴 ax 上添加图例，位置为右下角，列数为 2
                ax.legend(loc="lower right", ncol=2)

            # 调整子图的布局以适应显示需求
            plt.tight_layout()
            
            # 将当前图形保存为文件，文件名为 metric_over_time.png，保存在 output_dir 中
            plt.savefig(os.path.join(output_dir, f"{metric}_over_time.png"))

        # 调用实例方法 generate_comment()，完成 diff 方法的最后处理
        self.generate_comment()
class DashboardUpdater:
    """
    Aggregates the information and makes a comment to Performance Dashboard.
    https://github.com/pytorch/torchdynamo/issues/681
    """

    def __init__(self, args):
        # 初始化函数，接收参数并设置实例变量
        self.args = args  # 保存传入的参数对象
        self.output_dir = args.output_dir  # 设置输出目录
        self.lookup_file = os.path.join(self.args.dashboard_archive_path, "lookup.csv")  # 构建查找文件的路径
        assert os.path.exists(self.lookup_file)  # 断言查找文件存在
        try:
            # 如果不是更新测试并且不是禁止更新存档，调用更新查找文件方法
            if not self.args.update_dashboard_test and not self.args.no_update_archive:
                self.update_lookup_file()
        except subprocess.CalledProcessError:
            sys.stderr.write("failed to update lookup file\n")  # 如果更新查找文件失败，输出错误信息到标准错误流

    def update_lookup_file(self):
        # 更新查找文件方法
        dtype = self.args.dtypes[0]  # 获取数据类型
        day, _ = archive_data(self.args.archive_name)  # 调用archive_data函数获取日期和其他信息
        target_dir = get_archive_name(self.args, dtype)  # 调用get_archive_name函数获取存档名称
        # 更新查找文件，追加记录到lookup.csv文件中
        subprocess.check_call(
            f'echo "{day},performance,{dtype},{target_dir}" >> {self.lookup_file}',
            shell=True,
        )

    def archive(self):
        # 存档方法
        dtype = self.args.dtypes[0]  # 获取数据类型
        # 调用archive函数将输出目录中的内容复制到仪表板存档路径中
        archive(
            self.output_dir,
            self.args.dashboard_archive_path,
            self.args.archive_name,
            dtype,
        )

    def upload_graphs(self):
        # 上传图表方法
        title = "## Performance graphs ##\n"  # 设置标题
        str_io = io.StringIO()  # 创建内存文件对象
        if not self.args.update_dashboard_test and not self.args.no_graphs:
            # 遍历输出目录中所有png文件（非over_time的）
            for name in glob.glob(self.output_dir + "/*png"):
                if "over_time" not in name:
                    # 调用dashboard_image_uploader程序上传图片，并获取输出结果
                    output = (
                        subprocess.check_output(
                            [self.args.dashboard_image_uploader, name]
                        )
                        .decode("ascii")
                        .rstrip()
                    )
                    str_io.write(f"\n{name} : ![]({output})\n")  # 将上传结果写入内存文件对象
        comment = generate_dropdown_comment(title, str_io.getvalue())  # 生成下拉评论
        # 将评论写入gh_graphs.txt文件
        with open(f"{self.output_dir}/gh_graphs.txt", "w") as gh_fh:
            gh_fh.write(comment)

    def gen_comment(self):
        # 生成总评论方法
        files = [
            "gh_title.txt",
            "gh_executive_summary.txt",
            "gh_summary_diff.txt",
            "gh_warnings.txt",
            "gh_regression.txt",
            "gh_metric_regression.txt",
            "gh_training.txt" if self.args.training else "gh_inference.txt",
            "gh_graphs.txt",
            "gh_build_summary.txt",
        ]
        all_lines = []
        for f in files:
            try:
                with open(os.path.join(self.output_dir, f)) as fh:
                    all_lines.extend(fh.readlines())  # 逐行读取文件内容并添加到all_lines列表中
            except FileNotFoundError:
                pass  # 忽略文件未找到异常
        return "\n".join([x.rstrip() for x in all_lines])  # 返回所有文件内容组成的字符串，去除每行末尾的换行符
    # 向指定仓库的 GitHub 问题添加评论
    def comment_on_gh(self, comment):
        """
        Send a commment to dashboard
        """
        # 创建临时文件并写入评论内容
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write(comment)
            filename = f.name

        # 确定要评论的问题编号，默认为 "93794"，如果数据类型为 float32，则使用 "93518"
        issue_number = "93794"
        if self.args.dtypes[0] == "float32":
            issue_number = "93518"

        # 调用外部命令 subprocess.check_call 来执行 GitHub 命令行工具操作
        subprocess.check_call(
            [
                self.args.dashboard_gh_cli_path,  # GitHub 命令行工具路径
                "issue",  # 操作类型为问题
                "comment",  # 评论操作
                "--repo=https://github.com/pytorch/pytorch.git",  # 指定仓库地址
                issue_number,  # 问题编号
                "-F",  # 指定评论内容来源于文件
                filename,  # 评论内容所在的临时文件名
            ]
        )

        # 删除临时文件
        os.remove(filename)

    # 更新操作
    def update(self):
        # 上传图表数据
        self.upload_graphs()
        
        # 如果不禁用检测回归功能
        if not self.args.no_detect_regressions:
            # 生成总结统计差异的评论
            SummaryStatDiffer(self.args).generate_comment()
            # 生成回归检测器的评论
            RegressionDetector(self.args).generate_comment()
            try:
                # 尝试执行回归跟踪器的差异检测
                RegressionTracker(self.args).diff()
            except Exception as e:
                logging.exception("")  # 记录异常信息到日志
                # 写入空字符串到指定的 GitHub 回归文件
                with open(f"{self.args.output_dir}/gh_regression.txt", "w") as gh_fh:
                    gh_fh.write("")

        # 生成最终的评论内容
        comment = self.gen_comment()
        print(comment)  # 打印评论内容

        # 如果不是更新仪表板测试
        if not self.args.update_dashboard_test:
            # 如果不禁用 GitHub 评论
            if not self.args.no_gh_comment:
                self.comment_on_gh(comment)  # 调用评论 GitHub 问题的方法
            # 如果不禁用更新归档
            if not self.args.no_update_archive:
                self.archive()  # 执行归档操作
if __name__ == "__main__":
    # 解析命令行参数，获取程序运行所需参数
    args = parse_args()

    # 定义函数 extract，用于从命令行参数 args 中提取指定键的值，如果不存在则使用默认值
    def extract(key):
        return DEFAULTS[key] if getattr(args, key, None) is None else getattr(args, key)

    # 提取数据类型（dtypes）、测试套件（suites）、设备列表（devices）的值
    dtypes = extract("dtypes")
    suites = extract("suites")
    devices = extract("devices")

    # 根据参数 args.inference 的值决定使用推断（inference）或训练（training）的编译器和编译标志
    if args.inference:
        compilers = DEFAULTS["inference"] if args.compilers is None else args.compilers
        flag_compilers = (
            DEFAULTS["flag_compilers"]["inference"]
            if args.flag_compilers is None
            else args.flag_compilers
        )
    else:
        # 如果不是推断模式，则必须是训练模式
        assert args.training
        compilers = DEFAULTS["training"] if args.compilers is None else args.compilers
        flag_compilers = (
            DEFAULTS["flag_compilers"]["training"]
            if args.flag_compilers is None
            else args.flag_compilers
        )

    # 设置输出目录为 args.output_dir
    output_dir = args.output_dir

    # 更新命令行参数中的编译器、设备列表、数据类型、编译标志
    args.compilers = compilers
    args.devices = devices
    args.dtypes = dtypes
    flag_compilers = list(set(flag_compilers) & set(compilers))
    args.flag_compilers = flag_compilers
    args.suites = suites

    # 如果需要打印运行命令，则生成命令文件并提示用户运行
    if args.print_run_commands:
        generated_file = generate_commands(
            args, dtypes, suites, devices, compilers, output_dir
        )
        print(
            f"Running commands are generated in file {generated_file}. Please run (bash {generated_file})."
        )
    # 如果需要可视化日志，则解析日志数据
    elif args.visualize_logs:
        parse_logs(args, dtypes, suites, devices, compilers, flag_compilers, output_dir)
    # 如果需要运行命令，则生成命令文件，并尝试执行其中的命令
    elif args.run:
        generated_file = generate_commands(
            args, dtypes, suites, devices, compilers, output_dir
        )
        # 生成备忘录式的归档名称，以便反映运行开始的日期
        get_archive_name(args, dtypes[0])
        # TODO - 是否需要担心段错误

        # 尝试执行生成的命令文件，如果失败则捕获异常并提示用户手动运行
        try:
            os.system(f"bash {generated_file}")
        except Exception as e:
            print(
                f"Running commands failed. Please run manually (bash {generated_file}) and inspect the errors."
            )
            raise e

        # 如果不需要记录操作输入，则根据需要更新归档，并解析日志数据
        if not args.log_operator_inputs:
            if not args.no_update_archive:
                archive(
                    output_dir,
                    args.dashboard_archive_path,
                    args.archive_name,
                    dtypes[0],
                )
            parse_logs(
                args, dtypes, suites, devices, compilers, flag_compilers, output_dir
            )
            if not args.no_update_archive:
                archive(
                    output_dir,
                    args.dashboard_archive_path,
                    args.archive_name,
                    dtypes[0],
                )

    # 如果需要更新仪表盘，则调用 DashboardUpdater 类的 update 方法
    if args.update_dashboard:
        DashboardUpdater(args).update()
```