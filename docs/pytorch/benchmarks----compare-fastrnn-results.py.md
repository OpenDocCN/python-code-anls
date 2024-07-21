# `.\pytorch\benchmarks\compare-fastrnn-results.py`

```py
import argparse  # 导入处理命令行参数的模块
import json  # 导入处理 JSON 数据的模块
from collections import namedtuple  # 导入命名元组用于创建简单的对象结构

# 定义一个命名元组，用于表示比较结果的结构
Result = namedtuple("Result", ["name", "base_time", "diff_time"])


def construct_name(fwd_bwd, test_name):
    # 根据方向和测试名称构造一个唯一的名称
    bwd = "backward" in fwd_bwd  # 检查方向是否为反向
    suite_name = fwd_bwd.replace("-backward", "")  # 移除方向信息，获取测试套件名称
    return f"{suite_name}[{test_name}]:{'bwd' if bwd else 'fwd'}"


def get_times(json_data):
    r = {}  # 初始化一个空字典用于存储结果
    for fwd_bwd in json_data:  # 遍历 JSON 数据中的方向信息
        for test_name in json_data[fwd_bwd]:  # 遍历每个方向下的测试名称
            name = construct_name(fwd_bwd, test_name)  # 构建唯一的测试名称
            r[name] = json_data[fwd_bwd][test_name]  # 将测试结果存入字典中
    return r  # 返回存储了所有测试结果的字典


# 解析命令行参数
parser = argparse.ArgumentParser("compare two pytest jsons")
parser.add_argument("base", help="base json file")  # 基准 JSON 文件名参数
parser.add_argument("diff", help="diff json file")  # 比较 JSON 文件名参数
parser.add_argument(
    "--format", default="md", type=str, help="output format (csv, md, json, table)"
)  # 输出格式参数

args = parser.parse_args()  # 解析命令行参数并存储到 args 变量中

# 打开并读取基准 JSON 文件
with open(args.base) as base:
    base_times = get_times(json.load(base))  # 解析 JSON 数据并获取基准测试时间

# 打开并读取比较 JSON 文件
with open(args.diff) as diff:
    diff_times = get_times(json.load(diff))  # 解析 JSON 数据并获取比较测试时间

# 获取所有测试结果的键的并集
all_keys = set(base_times.keys()).union(diff_times.keys())

# 创建结果对象列表，存储每个测试名称及其基准时间和差异时间
results = [
    Result(name, base_times.get(name, float("nan")), diff_times.get(name, float("nan")))
    for name in sorted(all_keys)  # 按名称排序所有键
]

# 不同输出格式的格式化字符串定义
header_fmt = {
    "table": "{:48s} {:>13s} {:>15s} {:>10s}",
    "md": "| {:48s} | {:>13s} | {:>15s} | {:>10s} |",
    "csv": "{:s}, {:s}, {:s}, {:s}",
}
data_fmt = {
    "table": "{:48s} {:13.6f} {:15.6f} {:9.1f}%",
    "md": "| {:48s} | {:13.6f} | {:15.6f} | {:9.1f}% |",
    "csv": "{:s}, {:.6f}, {:.6f}, {:.2f}%",
}

# 根据用户指定的输出格式打印结果
if args.format in ["table", "md", "csv"]:
    header_fmt_str = header_fmt[args.format]
    data_fmt_str = data_fmt[args.format]
    print(header_fmt_str.format("name", "base time (s)", "diff time (s)", "% change"))
    if args.format == "md":
        print(header_fmt_str.format(":---", "---:", "---:", "---:"))  # Markdown 表格头部分隔符
    for r in results:
        print(
            data_fmt_str.format(
                r.name,
                r.base_time,
                r.diff_time,
                (r.diff_time / r.base_time - 1.0) * 100.0,  # 计算时间变化百分比
            )
        )
elif args.format == "json":
    print(json.dumps(results))  # 输出 JSON 格式的结果
else:
    raise ValueError("Unknown output format: " + args.format)  # 未知的输出格式异常处理
```