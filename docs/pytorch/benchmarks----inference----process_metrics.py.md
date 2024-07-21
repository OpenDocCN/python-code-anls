# `.\pytorch\benchmarks\inference\process_metrics.py`

```
"""
This file will take the csv outputs from server.py, calculate the mean and
variance of the warmup_latency, average_latency, throughput and gpu_util
and write these to the corresponding `results/output_{batch_size}_{compile}.md`
file, appending to the file if it exists or creatng a new one otherwise.
"""

import argparse  # 导入解析命令行参数的模块
import os  # 导入操作系统相关功能的模块

import pandas as pd  # 导入处理数据的模块

if __name__ == "__main__":
    # 如果这个脚本是作为主程序执行
    parser = argparse.ArgumentParser(description="Parse output files")
    parser.add_argument("--csv", type=str, help="Path to csv file")
    parser.add_argument("--name", type=str, help="Name of experiment")
    args = parser.parse_args()

    # 构建输入的 CSV 文件路径
    input_csv = "./results/" + args.csv
    # 使用 pandas 读取 CSV 文件并存入 DataFrame 对象 df
    df = pd.read_csv(input_csv)

    # 从命令行参数中提取批处理大小（batch_size）和编译信息（compile）
    batch_size = int(os.path.basename(args.csv).split("_")[1])
    compile = os.path.basename(args.csv).split("_")[-1].split(".")[0]

    # 计算特定指标的平均值和标准差
    metrics = ["warmup_latency", "average_latency", "throughput", "gpu_util"]
    means = dict()
    stds = dict()

    for metric in metrics:
        means[metric] = df[metric].mean()  # 计算指标的平均值
        stds[metric] = df[metric].std()    # 计算指标的标准差

    # 输出的 Markdown 文件路径
    output_md = f"results/output_{batch_size}_{compile}.md"
    write_header = os.path.isfile(output_md) is False  # 判断是否需要写入表头

    with open(output_md, "a+") as f:
        if write_header:
            # 如果文件不存在则写入表头和格式
            f.write(f"## Batch Size {batch_size} Compile {compile}\n\n")
            f.write(
                "| Experiment | Warmup_latency (s) | Average_latency (s) | Throughput (samples/sec) | GPU Utilization (%) |\n"
            )
            f.write(
                "| ---------- | ------------------ | ------------------- | ------------------------ | ------------------- |\n"
            )

        # 构建写入的行数据
        line = f"| {args.name} |"
        for metric in metrics:
            line += f" {means[metric]:.3f} +/- {stds[metric]:.3f} |"
        f.write(line + "\n")
```