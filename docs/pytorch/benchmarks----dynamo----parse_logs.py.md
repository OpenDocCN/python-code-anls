# `.\pytorch\benchmarks\dynamo\parse_logs.py`

```
import csv  # 导入CSV模块，用于操作CSV文件
import os   # 导入os模块，提供了一些与操作系统相关的功能
import re   # 导入re模块，用于正则表达式操作
import sys  # 导入sys模块，提供了对Python解释器的访问

# This script takes the logs produced by the benchmark scripts (e.g.,
# torchbench.py) and parses it into a CSV file that summarizes what
# is failing and why.  It is kept separate from the benchmark script
# emitting a more structured output as it is often more convenient
# to iterate quickly on log files offline instead of having to make
# a change to the benchmark script and then do a full sweep to see
# the updates.
#
# This script is not very well written, feel free to rewrite it as necessary

assert len(sys.argv) == 2  # 确保命令行参数的数量为2，否则抛出异常

full_log = open(sys.argv[1]).read()  # 读取命令行指定的日志文件内容

# If the log contains a gist URL, extract it so we can include it in the CSV
gist_url = ""
m = re.search(r"https://gist.github.com/[a-f0-9]+", full_log)  # 在日志中查找GitHub Gist的URL
if m is not None:
    gist_url = m.group(0)  # 如果找到匹配的URL，则保存到gist_url变量中

# Split the log into an entry per benchmark
entries = re.split(
    r"(?:cuda (?:train|eval) +([^ ]+)|WARNING:root:([^ ]+) failed to load)", full_log
)[1:]
# 使用正则表达式将日志拆分成每个基准测试条目，存储在entries列表中
# Entries schema example:
# `['hf_Bert', None, '
#  PASS\nTIMING: entire_frame_compile:1.80925 backend_compile:6e-05\nDynamo produced 1 graph(s) covering 367 ops\n']`

def chunker(seq, size):
    return (seq[pos : pos + size] for pos in range(0, len(seq), size))

c = 0
i = 0

out = csv.DictWriter(
    sys.stdout,
    [
        "bench",
        "name",
        "result",
        "component",
        "context",
        "explain",
        "frame_time",
        "backend_time",
        "graph_count",
        "op_count",
        "graph_breaks",
        "unique_graph_breaks",
    ],
    dialect="excel",
)
out.writeheader()  # 写入CSV文件头部信息
out.writerow({"explain": gist_url})  # 将gist_url写入CSV文件的explain列

# Sometimes backtraces will be in third party code, which results
# in very long file names.  Delete the absolute path in this case.
def normalize_file(f):
    if "site-packages/" in f:
        return f.split("site-packages/", 2)[1]  # 删除绝对路径中的'site-packages/'部分
    else:
        return os.path.relpath(f)  # 返回相对路径

# Assume we run torchbench, huggingface, timm_models in that order
# (as output doesn't say which suite the benchmark is part of)
# TODO: make this more robust

bench = "torchbench"  # 默认基准测试名称为torchbench

# 3 = 1 + number of matches in the entries split regex
for name, name2, log in chunker(entries, 3):
    if name is None:
        name = name2
    if name.startswith("Albert"):
        bench = "huggingface"  # 如果name以"Albert"开头，则基准测试为huggingface
    elif name.startswith("adv_inc"):
        bench = "timm_models"  # 如果name以"adv_inc"开头，则基准测试为timm_models

    # Payload that will go into the csv
    r = "UNKNOWN"  # 默认结果为UNKNOWN
    explain = ""
    component = ""
    context = ""

    if "PASS" in log:
        r = "PASS"  # 如果log中包含"PASS"，则结果为PASS
    if "TIMEOUT" in log:
        r = "FAIL TIMEOUT"  # 如果log中包含"TIMEOUT"，则结果为FAIL TIMEOUT
    if "Accuracy failed" in log:
        r = "FAIL ACCURACY"  # 如果log中包含"Accuracy failed"，则结果为FAIL ACCURACY

    # Attempt to extract out useful information from the traceback

    log = log.split(
        "The above exception was the direct cause of the following exception"
    )[0]
    split = log.split("Traceback (most recent call last)", maxsplit=1)
    if len(split) == 2:
        log = split[1]
    log = log.split("Original traceback:")[0]  # 删除log中的"Original traceback:"部分
    # 使用正则表达式在日志中查找特定格式的错误信息
    m = re.search(
        r'File "([^"]+)", line ([0-9]+), in .+\n +(.+)\n([A-Za-z]+(?:Error|Exception|NotImplementedError): ?.*)',
        log,
    )

    # 如果匹配到错误信息
    if m is not None:
        # 设置结果为 "FAIL"
        r = "FAIL"
        # 组装组件信息，格式化文件路径和行号
        component = f"{normalize_file(m.group(1))}:{m.group(2)}"
        # 获取错误发生的上下文信息
        context = m.group(3)
        # 获取错误的解释信息
        explain = f"{m.group(4)}"
    else:
        # 如果未匹配到常规错误，尝试匹配断言错误信息
        m = re.search(
            r'File "([^"]+)", line ([0-9]+), in .+\n +(.+)\nAssertionError', log
        )
        # 如果匹配到断言错误
        if m is not None:
            # 设置结果为 "FAIL"
            r = "FAIL"
            # 组装组件信息，格式化文件路径和行号
            component = f"{normalize_file(m.group(1))}:{m.group(2)}"
            # 获取错误发生的上下文信息
            context = m.group(3)
            # 设置错误的解释信息为 "AssertionError"
            explain = "AssertionError"

    # 在日志中检查是否包含 "FAIL" 字符串，指示测试失败
    if "FAIL" in log:
        # 设置结果为 "FAIL"
        r = "FAIL"

    # 如果结果为 "UNKNOWN"，增加计数器 c
    if r == "UNKNOWN":
        c += 1

    # 初始化后端时间和帧时间
    backend_time = None
    frame_time = None

    # 如果日志中包含 "TIMING:" 标记
    if "TIMING:" in log:
        # 从日志中提取时间信息
        result = re.search("TIMING:(.*)\n", log).group(1)
        # 按照特定分隔符分割字符串
        split_str = result.split("backend_compile:")
        # 如果成功分割成两部分
        if len(split_str) == 2:
            # 提取后端编译时间
            backend_time = float(split_str[1])
            # 提取帧时间
            frame_time = float(split_str[0].split("entire_frame_compile:")[1])

    # 如果日志中包含 "STATS:" 标记
    if "STATS:" in log:
        # 从日志中提取统计信息
        result = re.search("STATS:(.*)\n", log).group(1)
        # 按竖线符号分割所有统计信息
        split_all = result.split("|")
        # TODO: 重新设计以处理任意数量的统计信息

    # 初始化图形计数、操作计数、图形中断和唯一图形中断
    graph_count = None
    op_count = None
    graph_breaks = None
    unique_graph_breaks = None

    # 如果日志中包含特定格式的信息，匹配并提取相关统计数据
    if m := re.search(
        r"Dynamo produced (\d+) graphs covering (\d+) ops with (\d+) graph breaks \((\d+) unique\)",
        log,
    ):
        # 提取图形数量、操作数量、图形中断和唯一图形中断的数据
        graph_count = m.group(1)
        op_count = m.group(2)
        graph_breaks = m.group(3)
        unique_graph_breaks = m.group(4)

    # 如果上下文字符串过长，不将其包含在 CSV 中
    if len(context) > 78:
        context = ""

    # 如果组件包含临时文件路径 "/tmp/"，将组件信息设置为 "generated code"
    if "/tmp/" in component:
        component = "generated code"
        # 清空上下文信息
        context = ""

    # 输出一行 CSV 记录，包括测试名称、结果、组件、上下文、解释、帧时间、后端时间、图形计数、操作计数、图形中断和唯一图形中断
    out.writerow(
        {
            "bench": bench,
            "name": name,
            "result": r,
            "component": component,
            "context": context,
            "explain": explain,
            "frame_time": frame_time,
            "backend_time": backend_time,
            "graph_count": graph_count,
            "op_count": op_count,
            "graph_breaks": graph_breaks,
            "unique_graph_breaks": unique_graph_breaks,
        }
    )
    # 增加计数器 i
    i += 1
# 如果条件变量 c 的值为真，则打印错误信息到标准错误输出流
if c:
    print(f"failed to classify {c} entries", file=sys.stderr)
```