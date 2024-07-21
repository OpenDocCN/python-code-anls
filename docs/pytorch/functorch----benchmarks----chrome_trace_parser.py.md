# `.\pytorch\functorch\benchmarks\chrome_trace_parser.py`

```py
#!/usr/bin/env python3
# 导入必要的模块
import argparse  # 解析命令行参数的模块
import logging   # 记录日志的模块

import os        # 提供与操作系统交互的功能

import pandas as pd  # 数据分析库 Pandas

from torch._functorch.benchmark_utils import compute_utilization  # 导入计算工具函数

# 处理由 PyTorch Profiler 输出的 Chrome 跟踪数据
# 需要输入的 JSON 文件名格式为 {model_name}_chrome_trace_*.json
# 运行时间文件应具有格式 (model_name, runtime)


def get_model_name(filename):
    """
    从格式为 {model_name}_chrome_trace_*.json 的文件中获取模型名称
    """
    _, tail = os.path.split(filename)
    modelname = tail[: tail.find("_chrome_trace")]
    return modelname


def get_total_length(run_times_df, modelname):
    """
    获取给定模型名称的总运行时间长度
    """
    return float(run_times_df[run_times_df["name"] == modelname]["runtime"])


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    parser.add_argument(
        "--runtime", "-runf", help="运行时间文件的文件名", required=True
    )
    group.add_argument(
        "--filename",
        "-f",
        action="append",
        help="要处理的 JSON 文件的文件名",
    )
    group.add_argument("--folder", "-fd", help="要处理的包含 JSON 文件的文件夹")
    args = parser.parse_args()

    if args.filename:
        filenames = args.filename
    elif args.folder:
        filenames = []
        directory = args.folder
        for filename in os.listdir(directory):
            f = os.path.join(directory, filename)
            if os.path.isfile(f) and f.endswith(".json"):
                filenames.append(f)
    else:
        print("请提供文件名或文件夹名")

    print("模型名称, GPU 利用率, MM 和 Conv 时间")

    run_times_df = pd.read_csv(args.runtime)
    for filename in filenames:
        try:
            modelname = get_model_name(filename)
            total_length = get_total_length(run_times_df, modelname) * 1e6
            utilization, mm_conv_utilization = compute_utilization(
                filenames, total_length
            )
            print(f"{modelname}, {utilization}, {mm_conv_utilization}")
        except BaseException:
            logging.exception("%s, 错误", filename)
            print(f"{filename}, 错误")


if __name__ == "__main__":
    main()
```