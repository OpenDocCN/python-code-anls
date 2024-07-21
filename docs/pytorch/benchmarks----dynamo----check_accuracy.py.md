# `.\pytorch\benchmarks\dynamo\check_accuracy.py`

```py
# 导入必要的模块
import argparse  # 用于解析命令行参数
import os  # 提供与操作系统交互的功能
import sys  # 提供与 Python 解释器进行交互的变量和函数
import textwrap  # 提供文本格式化和填充功能

import pandas as pd  # 导入 pandas 库，用于处理和分析数据


# 模型列表，这些模型具有不稳定的测试结果
flaky_models = {
    "yolov3",
    "gluon_inception_v3",
    "XGLMForCausalLM",  # 在 https://github.com/pytorch/pytorch/pull/128148 中发现
}


# 从 CSV 中获取指定模型的字段值
def get_field(csv, model_name: str, field: str):
    try:
        return csv.loc[csv["name"] == model_name][field].item()
    except Exception as e:
        return None


# 检查实际和期望的 CSV 文件中模型的准确率
def check_accuracy(actual_csv, expected_csv, expected_filename):
    failed = []  # 存储准确率下降的模型列表
    improved = []  # 存储准确率提高的模型列表

    for model in actual_csv["name"]:
        accuracy = get_field(actual_csv, model, "accuracy")
        expected_accuracy = get_field(expected_csv, model, "accuracy")

        if accuracy == expected_accuracy:
            status = "PASS" if expected_accuracy == "pass" else "XFAIL"
            print(f"{model:34}  {status}")  # 输出模型名称及其状态
            continue
        elif model in flaky_models:
            if accuracy == "pass":
                # 模型通过但标记为失败
                status = "PASS_BUT_FLAKY:"
            else:
                # 模型失败但标记为通过
                status = "FAIL_BUT_FLAKY:"
        elif accuracy != "pass":
            status = "FAIL:"
            failed.append(model)
        else:
            status = "IMPROVED:"
            improved.append(model)
        
        # 输出模型名称、状态、准确率和期望准确率
        print(f"{model:34}  {status:9} accuracy={accuracy}, expected={expected_accuracy}")

    msg = ""
    # 构建消息，指出准确率下降或提高的模型数量及其名称
    if failed or improved:
        if failed:
            msg += textwrap.dedent(
                f"""
            Error: {len(failed)} models have accuracy status regressed:
                {' '.join(failed)}

            """
            )
        if improved:
            msg += textwrap.dedent(
                f"""
            Improvement: {len(improved)} models have accuracy status improved:
                {' '.join(improved)}

            """
            )
        sha = os.getenv("SHA1", "{your CI commit sha}")  # 获取环境变量中的 SHA1 或使用默认值
        # 添加更新预期 CSV 的指南
        msg += textwrap.dedent(
            f"""
        If this change is expected, you can update `{expected_filename}` to reflect the new baseline.
        from pytorch/pytorch root, run
        `python benchmarks/dynamo/ci_expected_accuracy/update_expected.py {sha}`
        and then `git add` the resulting local changes to expected CSVs to your commit.
        """
        )
    # 返回准确率下降或提高的模型列表和消息
    return failed or improved, msg


# 主函数，负责解析命令行参数并调用检查准确率的函数
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--actual", type=str, required=True)  # 实际的 CSV 文件路径参数
    parser.add_argument("--expected", type=str, required=True)  # 期望的 CSV 文件路径参数
    args = parser.parse_args()  # 解析命令行参数

    actual = pd.read_csv(args.actual)  # 读取实际结果 CSV 文件为 DataFrame
    expected = pd.read_csv(args.expected)  # 读取期望结果 CSV 文件为 DataFrame

    failed, msg = check_accuracy(actual, expected, args.expected)  # 调用检查准确率函数

    if failed:
        print(msg)  # 输出消息，指出准确率下降或提高的模型情况
        sys.exit(1)  # 如果有准确率下降的模型，则以非零状态退出程序


if __name__ == "__main__":
    main()  # 执行主函数
```