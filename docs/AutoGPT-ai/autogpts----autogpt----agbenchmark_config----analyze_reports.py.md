# `.\AutoGPT\autogpts\autogpt\agbenchmark_config\analyze_reports.py`

```py
#!/usr/bin/env python3

# 导入所需的模块
import json
import logging
import re
import sys
from collections import defaultdict
from pathlib import Path

# 导入第三方库 tabulate
from tabulate import tabulate

# 根据命令行参数设置日志级别
info = "-v" in sys.argv
debug = "-vv" in sys.argv
granular = "--granular" in sys.argv

# 配置日志记录器
logging.basicConfig(
    level=logging.DEBUG if debug else logging.INFO if info else logging.WARNING
)
logger = logging.getLogger(__name__)

# 获取目录中所有的 JSON 文件
report_files = [
    report_file
    for dir in (Path(__file__).parent / "reports").iterdir()
    if re.match(r"^\d{8}T\d{6}_", dir.name)
    and (report_file := dir / "report.json").is_file()
]

# 初始化变量
labels = list[str]()
runs_per_label = defaultdict[str, int](lambda: 0)
suite_names = list[str]()
test_names = list[str]()

# 创建一个字典来存储按后缀和测试分组的成功值
grouped_success_values = defaultdict[str, list[str]](list[str])

# 遍历每个 JSON 文件以收集后缀和成功值
for report_file in sorted(report_files):
    if label not in labels:
        labels.append(label)

# 创建表头
headers = ["Test Name"] + list(labels)

# 准备表格数据
table_data = list[list[str]]()
for test_name in test_names:
    row = [test_name]
    for label in labels:
        results = grouped_success_values.get(f"{label}|{test_name}", ["❔"])
        if len(results) < runs_per_label[label]:
            results.extend(["❔"] * (runs_per_label[label] - len(results)))
        if len(results) > 1 and all(r == "❔" for r in results):
            results.clear()
        row.append(" ".join(results))
    table_data.append(row)

# 打印表格数据
print(tabulate(table_data, headers=headers, tablefmt="grid"))
```