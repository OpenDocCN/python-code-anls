# `.\pytorch\benchmarks\dynamo\check_perf_csv.py`

```
# 导入必要的库
import argparse  # 用于解析命令行参数
import sys  # 用于访问系统相关功能
import textwrap  # 用于格式化文本输出

import pandas as pd  # 导入 pandas 库，用于处理 CSV 文件


def check_perf_csv(filename, threshold):
    """
    Basic performance checking function for CSV files.

    Args:
        filename (str): CSV 文件名
        threshold (float): 性能加速阈值，用于检查性能是否有回退
    """

    # 使用 pandas 读取 CSV 文件内容并创建 DataFrame 对象
    df = pd.read_csv(filename)

    failed = []
    # 遍历 DataFrame 的每一行
    for _, row in df.iterrows():
        model_name = row["name"]  # 获取模型名称
        speedup = row["speedup"]  # 获取速度提升比例
        # 如果速度提升比例低于阈值，将模型名称加入到失败列表中
        if speedup < threshold:
            failed.append(model_name)

        # 输出模型名称和其对应的速度提升比例
        print(f"{model_name:34} {speedup}")

    # 如果有模型性能回退，输出错误信息
    if failed:
        print(
            textwrap.dedent(
                f"""
                Error {len(failed)} models performance regressed
                    {' '.join(failed)}
                """
            )
        )
        # 退出程序并返回状态码 1
        sys.exit(1)


if __name__ == "__main__":
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser()
    # 添加文件名参数，用于指定要检查的 CSV 文件
    parser.add_argument("--file", "-f", type=str, help="csv file name")
    # 添加阈值参数，用于指定性能加速的阈值
    parser.add_argument(
        "--threshold", "-t", type=float, help="threshold speedup value to check against"
    )
    # 解析命令行参数
    args = parser.parse_args()
    # 调用性能检查函数，传入文件名和阈值参数
    check_perf_csv(args.file, args.threshold)
```