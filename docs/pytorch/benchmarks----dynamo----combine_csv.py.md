# `.\pytorch\benchmarks\dynamo\combine_csv.py`

```py
# 导入必要的模块：ast 用于安全地将字符串转换为 Python 字面量，csv 用于 CSV 文件操作，sys 用于访问命令行参数，defaultdict 用于创建默认字典。
import ast
import csv
import sys
from collections import defaultdict

# 确保命令行参数数量为3，否则抛出 AssertionError
assert len(sys.argv) == 3

# 创建一个默认字典 RESULTS 用于存储处理后的数据，结构为 { (bench, name): { 'static': row, 'dynamic': row } }
RESULTS = defaultdict(dict)

# 遍历命令行参数中的每个文件，side 取值为 ["static", "dynamic"]
for side, f in zip(["static", "dynamic"], sys.argv[1:]):
    # 打开文件并创建 CSV 字典读取器对象 reader
    with open(f) as f:
        reader = csv.DictReader(f)
        # 遍历 CSV 文件中的每一行
        for row in reader:
            # 将当前行 row 存入 RESULTS 中对应的 bench 和 name 下的 static 或 dynamic 字典中
            RESULTS[(row["bench"], row["name"])][side] = row

# 指定输出的字段列表
fields = ["frame_time", "graph_breaks"]

# 创建 CSV 字典写入器对象 out，将结果写入到标准输出 sys.stdout
out = csv.DictWriter(
    sys.stdout,
    ["bench", "name"] + [f"delta_{n}" for n in fields] + ["static_url", "dynamic_url"],
    dialect="excel",
)
# 写入 CSV 文件头部信息
out.writeheader()

# 遍历 RESULTS 中的每个 (bench, name)，以及其对应的 static 和 dynamic 数据字典 sides
for (bench, name), sides in RESULTS.items():
    # 如果 static 或 dynamic 数据不存在，则跳过当前循环
    if "static" not in sides:
        continue
    if "dynamic" not in sides:
        continue
    # 如果 name 为空，则直接将 static_url 和 dynamic_url 写入输出
    if not name:
        out.writerow(
            {
                "static_url": sides["static"]["explain"],
                "dynamic_url": sides["dynamic"]["explain"],
            }
        )
        continue
    # 构建当前行的字典数据 row，包括 bench 和 name
    row = {"bench": bench, "name": name}
    # 遍历字段列表 fields
    for f in fields:
        try:
            # 尝试将 static 和 dynamic 中的字段 f 转换为 Python 字面量
            static = ast.literal_eval(sides["static"][f])
            dynamic = ast.literal_eval(sides["dynamic"][f])
        except SyntaxError:
            # 如果转换出错则跳过当前字段的处理
            continue
        # 计算 dynamic - static 的差值，存入 row 中的 delta_ 字段中
        row[f"delta_{f}"] = dynamic - static
    # 将当前行数据写入 CSV 输出
    out.writerow(row)
```