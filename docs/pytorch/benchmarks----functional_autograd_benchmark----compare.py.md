# `.\pytorch\benchmarks\functional_autograd_benchmark\compare.py`

```py
# 导入必要的模块 argparse 用于命令行参数解析，defaultdict 用于创建默认值字典
import argparse
from collections import defaultdict

# 导入自定义模块 utils 中的函数 from_markdown_table 和 to_markdown_table
from utils import from_markdown_table, to_markdown_table


# 主函数，用于比较基准测试结果
def main():
    # 创建命令行参数解析器，并设置描述信息
    parser = argparse.ArgumentParser(
        "Main script to compare results from the benchmarks"
    )
    
    # 添加命令行参数 --before，用于指定基准测试的文本文件名，默认为 "before.txt"
    parser.add_argument(
        "--before",
        type=str,
        default="before.txt",
        help="Text file containing the times to use as base",
    )
    
    # 添加命令行参数 --after，用于指定新版本测试的文本文件名，默认为 "after.txt"
    parser.add_argument(
        "--after",
        type=str,
        default="after.txt",
        help="Text file containing the times to use as new version",
    )
    
    # 添加命令行参数 --output，用于指定输出结果的文本文件名，默认为空字符串表示不写入文件
    parser.add_argument(
        "--output", type=str, default="", help="Text file where to write the output"
    )
    
    # 解析命令行参数
    args = parser.parse_args()

    # 读取 --before 参数指定的文件内容，并解析成字典格式
    with open(args.before) as f:
        content = f.read()
    res_before = from_markdown_table(content)

    # 读取 --after 参数指定的文件内容，并解析成字典格式
    with open(args.after) as f:
        content = f.read()
    res_after = from_markdown_table(content)

    # 创建默认值为字典的 defaultdict 对象，用于存储测试结果的差异
    diff = defaultdict(defaultdict)
    
    # 遍历基准测试结果字典 res_before
    for model in res_before:
        for task in res_before[model]:
            mean_before, var_before = res_before[model][task]
            # 如果当前任务在新版本测试结果中不存在，记录为 None
            if task not in res_after[model]:
                diff[model][task] = (None, mean_before, var_before, None, None)
            else:
                mean_after, var_after = res_after[model][task]
                # 计算速度提升比例，并记录相关数据
                diff[model][task] = (
                    mean_before / mean_after,
                    mean_before,
                    var_before,
                    mean_after,
                    var_after,
                )
    
    # 遍历新版本测试结果字典 res_after
    for model in res_after:
        for task in res_after[model]:
            # 如果当前任务在基准测试结果中不存在，记录新版本的数据
            if task not in res_before[model]:
                mean_after, var_after = res_after[model][task]
                diff[model][task] = (None, None, None, mean_after, var_after)

    # 定义表头
    header = (
        "model",
        "task",
        "speedup",
        "mean (before)",
        "var (before)",
        "mean (after)",
        "var (after)",
    )
    
    # 将差异数据 diff 转换为 Markdown 格式的表格，并生成字符串 out
    out = to_markdown_table(diff, header=header)

    # 打印输出结果
    print(out)
    
    # 如果指定了 --output 参数，则将结果写入到指定的文件中
    if args.output:
        with open(args.output, "w") as f:
            f.write(out)


# 如果当前脚本作为主程序运行，则调用主函数 main()
if __name__ == "__main__":
    main()
```