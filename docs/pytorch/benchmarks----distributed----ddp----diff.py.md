# `.\pytorch\benchmarks\distributed\ddp\diff.py`

```
#!/usr/bin/env python3
#
# Computes difference between measurements produced by ./benchmark.py.
#

import argparse  # 导入命令行参数解析模块
import json  # 导入处理 JSON 格式的模块

import numpy as np  # 导入用于数值计算的模块


def load(path):
    """
    从指定路径加载 JSON 文件并返回其内容。
    """
    with open(path) as f:
        return json.load(f)


def main():
    """
    主函数，用于执行比较两个 benchmark 结果文件的差异。
    """
    parser = argparse.ArgumentParser(description="PyTorch distributed benchmark diff")
    parser.add_argument("file", nargs=2)  # 解析命令行参数，期望输入两个文件路径
    args = parser.parse_args()

    if len(args.file) != 2:
        raise RuntimeError("Must specify 2 files to diff")  # 如果文件数不是两个，抛出运行时错误

    ja = load(args.file[0])  # 加载第一个 JSON 文件
    jb = load(args.file[1])  # 加载第二个 JSON 文件

    keys = (set(ja.keys()) | set(jb.keys())) - {"benchmark_results"}  # 获取所有键，排除 "benchmark_results"
    print(f"{'':20s} {'baseline':>20s}      {'test':>20s}")
    print(f"{'':20s} {'-' * 20:>20s}      {'-' * 20:>20s}")
    for key in sorted(keys):
        va = str(ja.get(key, "-"))  # 获取第一个文件中键对应的值，不存在则为 "-"
        vb = str(jb.get(key, "-"))  # 获取第二个文件中键对应的值，不存在则为 "-"
        print(f"{key + ':':20s} {va:>20s}  vs  {vb:>20s}")  # 输出键、第一个文件值、第二个文件值的比较结果
    print("")

    ba = ja["benchmark_results"]  # 获取第一个文件中的 benchmark_results
    bb = jb["benchmark_results"]  # 获取第二个文件中的 benchmark_results
    for ra, rb in zip(ba, bb):
        if ra["model"] != rb["model"]:
            continue
        if ra["batch_size"] != rb["batch_size"]:
            continue

        model = ra["model"]
        batch_size = int(ra["batch_size"])
        name = f"{model} with batch size {batch_size}"
        print(f"Benchmark: {name}")

        # Print header
        print("")
        print(f"{'':>10s}", end="")  # 输出表头
        for _ in [75, 95]:
            print(
                f"{'sec/iter':>16s}{'ex/sec':>10s}{'diff':>10s}", end=""
            )  # 输出每列的标题信息
        print("")

        # Print measurements
        for i, (xa, xb) in enumerate(zip(ra["result"], rb["result"])):
            # Ignore round without ddp
            if i == 0:
                continue
            # Sanity check: ignore if number of ranks is not equal
            if len(xa["ranks"]) != len(xb["ranks"]):
                continue

            ngpus = len(xa["ranks"])
            ma = sorted(xa["measurements"])
            mb = sorted(xb["measurements"])
            print(f"{ngpus:>4d} GPUs:", end="")  # 输出 GPU 数量信息
            for p in [75, 95]:
                va = np.percentile(ma, p)
                vb = np.percentile(mb, p)
                # We're measuring time, so lower is better (hence the negation)
                delta = -100 * ((vb - va) / va)
                print(
                    f"  p{p:02d}: {vb:8.3f}s {int(batch_size / vb):7d}/s {delta:+8.1f}%",
                    end="",
                )  # 输出每个百分位的测量结果
            print("")
        print("")


if __name__ == "__main__":
    main()
```