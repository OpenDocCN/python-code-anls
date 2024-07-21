# `.\pytorch\benchmarks\dynamo\check_csv.py`

```
# 导入必要的模块：命令行参数解析、系统交互、文本包装、Pandas 数据处理库
import argparse
import sys
import textwrap

# 导入 Pandas 数据处理库并重命名为 pd
import pandas as pd


# 定义函数 check_csv，用于基本的 CSV 文件精度检查
def check_csv(filename):
    """
    Basic accuracy checking.
    """

    # 使用 Pandas 读取 CSV 文件并存储为 DataFrame 对象 df
    df = pd.read_csv(filename)

    # 初始化一个空列表，用于存储未通过检查的模型名称
    failed = []

    # 遍历 DataFrame 中的每一行
    for _, row in df.iterrows():
        # 获取当前行的模型名称和准确度状态
        model_name = row["name"]
        status = row["accuracy"]

        # 如果状态列中不包含 "pass"，则将模型名称添加到 failed 列表中
        if "pass" not in status:
            failed.append(model_name)

        # 打印当前模型名称和其准确度状态
        print(f"{model_name:34} {status}")

    # 如果有未通过检查的模型
    if failed:
        # 打印错误信息，包括未通过检查的模型数量和名称列表
        print(
            textwrap.dedent(
                f"""
                Error {len(failed)} models failed
                    {' '.join(failed)}
                """
            )
        )
        # 终止程序并返回非零退出码表示失败
        sys.exit(1)


# 如果当前脚本作为主程序运行
if __name__ == "__main__":
    # 创建参数解析器对象
    parser = argparse.ArgumentParser()

    # 添加命令行参数选项 --file 或 -f，用于指定要检查的 CSV 文件名
    parser.add_argument("--file", "-f", type=str, help="csv file name")

    # 解析命令行参数
    args = parser.parse_args()

    # 调用 check_csv 函数，传入命令行参数中指定的 CSV 文件名
    check_csv(args.file)
```