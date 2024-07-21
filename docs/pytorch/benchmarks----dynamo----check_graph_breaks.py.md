# `.\pytorch\benchmarks\dynamo\check_graph_breaks.py`

```
# 导入必要的库
import argparse  # 用于解析命令行参数
import os  # 提供与操作系统交互的功能
import sys  # 提供与系统相关的参数和功能
import textwrap  # 提供文本包装和填充功能

import pandas as pd  # 导入 pandas 库，用于处理和分析数据


# 从 CSV 文件中获取特定模型的指定字段值
def get_field(csv, model_name: str, field: str):
    try:
        # 在 CSV 数据中查找模型名称为 model_name 的行，并返回指定字段的值
        return csv.loc[csv["name"] == model_name][field].item()
    except Exception as e:
        return None  # 如果发生异常（如未找到模型或字段），则返回 None


# 比较实际 CSV 和期望 CSV 中每个模型的动态图断点情况，并输出比较结果
def check_graph_breaks(actual_csv, expected_csv, expected_filename):
    failed = []  # 存储未通过检查的模型列表
    improved = []  # 存储有改进的模型列表

    # 遍历实际 CSV 中的每个模型
    for model in actual_csv["name"]:
        # 获取实际和期望的动态图断点数
        graph_breaks = get_field(actual_csv, model, "graph_breaks")
        expected_graph_breaks = get_field(expected_csv, model, "graph_breaks")

        # 检查实际和期望的动态图断点数是否相等
        if graph_breaks == expected_graph_breaks:
            status = "PASS"  # 如果相等，则状态为通过
            print(f"{model:34}  {status}")  # 输出模型名称和状态
            continue  # 继续下一个模型的比较

        # 如果实际动态图断点数大于期望的，则标记为失败
        elif graph_breaks > expected_graph_breaks:
            status = "FAIL:"  # 状态为失败
            failed.append(model)  # 将模型添加到失败列表中
        # 如果实际动态图断点数小于期望的，则标记为改进
        elif graph_breaks < expected_graph_breaks:
            status = "IMPROVED:"  # 状态为改进
            improved.append(model)  # 将模型添加到改进列表中

        # 输出模型名称、状态及详细信息
        print(
            f"{model:34}  {status:9} graph_breaks={graph_breaks}, expected={expected_graph_breaks}"
        )

    msg = ""
    # 如果有失败或改进的模型
    if failed or improved:
        # 构建错误和改进信息的消息
        if failed:
            msg += textwrap.dedent(
                f"""
            Error: {len(failed)} models have new dynamo graph breaks:
                {' '.join(failed)}

            """
            )
        if improved:
            msg += textwrap.dedent(
                f"""
            Improvement: {len(improved)} models have fixed dynamo graph breaks:
                {' '.join(improved)}

            """
            )
        # 获取环境变量中的提交 SHA1 值，或使用占位符提示用户替换
        sha = os.getenv("SHA1", "{your CI commit sha}")
        # 添加更新期望 CSV 文件的指导信息
        msg += textwrap.dedent(
            f"""
        If this change is expected, you can update `{expected_filename}` to reflect the new baseline.
        from pytorch/pytorch root, run
        `python benchmarks/dynamo/ci_expected_accuracy/update_expected.py {sha}`
        and then `git add` the resulting local changes to expected CSVs to your commit.
        """
        )
    return failed or improved, msg  # 返回失败或改进的模型列表及相关消息


# 主函数，解析命令行参数并调用检查函数
def main():
    parser = argparse.ArgumentParser()  # 创建参数解析器
    parser.add_argument("--actual", type=str, required=True)  # 添加实际 CSV 文件路径参数
    parser.add_argument("--expected", type=str, required=True)  # 添加期望 CSV 文件路径参数
    args = parser.parse_args()  # 解析命令行参数

    actual = pd.read_csv(args.actual)  # 读取实际 CSV 文件并转换为 pandas DataFrame
    expected = pd.read_csv(args.expected)  # 读取期望 CSV 文件并转换为 pandas DataFrame

    failed, msg = check_graph_breaks(actual, expected, args.expected)  # 调用检查函数进行比较

    if failed:
        print(msg)  # 如果有失败模型，则打印相关消息
        sys.exit(1)  # 退出程序并返回非零状态码


if __name__ == "__main__":
    main()  # 当脚本直接运行时，调用主函数
```