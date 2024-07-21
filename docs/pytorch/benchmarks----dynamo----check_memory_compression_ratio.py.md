# `.\pytorch\benchmarks\dynamo\check_memory_compression_ratio.py`

```
# 导入必要的模块
import argparse  # 解析命令行参数的库
import sys  # 提供对 Python 解释器的访问和一些系统相关的功能
import textwrap  # 提供文本包装和填充功能

import pandas as pd  # 导入 pandas 库，用于数据操作和分析


# 主函数，程序入口
def main(args):
    # 从命令行参数中读取实际数据文件和预期数据文件，分别存入 actual 和 expected 变量中
    actual = pd.read_csv(args.actual)  # 读取实际数据文件并转换为 pandas DataFrame
    expected = pd.read_csv(args.expected)  # 读取预期数据文件并转换为 pandas DataFrame
    failed = []  # 创建一个空列表，用于存储未通过检查的模型名称

    # 遍历实际数据 DataFrame 中的每个模型名称
    for name in actual["name"]:
        # 从实际数据中获取当前模型的压缩比例并转换为浮点数
        actual_memory_compression = float(
            actual.loc[actual["name"] == name]["compression_ratio"]
        )
        try:
            # 尝试从预期数据中获取当前模型的压缩比例并转换为浮点数
            expected_memory_compression = float(
                expected.loc[expected["name"] == name]["compression_ratio"]
            )
        except TypeError:
            # 如果出现 TypeError，说明当前模型在预期数据中不存在，打印相应信息并继续下一个模型的检查
            print(f"{name:34} is missing from {args.expected}")
            continue
        
        # 检查实际内存压缩比例是否大于等于预期比例的 95%
        if actual_memory_compression >= expected_memory_compression * 0.95:
            status = "PASS"  # 如果通过检查，则状态为 PASS
        else:
            status = "FAIL"  # 如果未通过检查，则状态为 FAIL
            failed.append(name)  # 将未通过检查的模型名称添加到 failed 列表中
        
        # 打印当前模型的名称、实际压缩比例、预期压缩比例以及状态信息
        print(
            f"""
            {name:34}:
                actual_memory_compression={actual_memory_compression:.2f},
                expected_memory_compression={expected_memory_compression:.2f},
                {status}
            """
        )

    # 如果有模型未通过检查，则打印相关错误信息和建议，并以非零状态退出程序
    if failed:
        print(
            textwrap.dedent(
                f"""
                Error: {len(failed)} models below expected memory compression ratio:
                    {' '.join(failed)}
                If this drop is expected, you can update `{args.expected}`.
                """
            )
        )
        sys.exit(1)  # 以状态码 1 退出程序，表示发生错误


# 创建命令行参数解析器
parser = argparse.ArgumentParser()
parser.add_argument("--actual", type=str, required=True)  # 添加必需的实际数据文件参数
parser.add_argument("--expected", type=str, required=True)  # 添加必需的预期数据文件参数
args = parser.parse_args()  # 解析命令行参数并将其存储在 args 变量中

# 如果脚本被直接执行，则调用主函数，并传入命令行参数
if __name__ == "__main__":
    main(args)  # 调用主函数，开始执行主要逻辑
```