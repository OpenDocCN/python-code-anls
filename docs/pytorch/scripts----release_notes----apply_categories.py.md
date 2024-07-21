# `.\pytorch\scripts\release_notes\apply_categories.py`

```
# 导入处理 CSV 文件的标准库
import csv

# 导入自定义的 commitlist 模块，用于读取和写入 commitlist 数据
import commitlist

# 指定包含分类数据的 CSV 文件路径
category_csv = "results/category_data.csv"

# 指定包含 commitlist 数据的 CSV 文件路径
commitlist_csv = "results/commitlist.csv"

# 打开并读取分类数据 CSV 文件
with open(category_csv) as category_data:
    # 使用 DictReader 读取 CSV 文件内容，并根据 commit_fields 解析行数据
    reader = csv.DictReader(category_data, commitlist.commit_fields)
    # 将所有行转换为列表
    rows = list(reader)
    # 创建一个字典，将 commit_hash 映射到对应的 category
    category_map = {row["commit_hash"]: row["category"] for row in rows}

# 打开并读取 commitlist 数据 CSV 文件
with open(commitlist_csv) as commitlist_data:
    # 使用 DictReader 读取 CSV 文件内容，并根据 commit_fields 解析行数据
    reader = csv.DictReader(commitlist_data, commitlist.commit_fields)
    # 将所有行转换为列表
    commitlist_rows = list(reader)

# 遍历 commitlist 的每一行数据
for row in commitlist_rows:
    # 获取当前行的 commit_hash
    hash = row["commit_hash"]
    # 如果 commit_hash 在 category_map 中存在，并且其对应的 category 不是 "Uncategorized"
    if hash in category_map and category_map[hash] != "Uncategorized":
        # 更新当前行的 category 为 category_map 中对应的值
        row["category"] = category_map[hash]

# 打开 commitlist 数据 CSV 文件，准备写入更新后的数据
with open(commitlist_csv, "w") as commitlist_write:
    # 使用 DictWriter 准备写入 CSV 文件，并指定字段名
    writer = csv.DictWriter(commitlist_write, commitlist.commit_fields)
    # 写入 CSV 文件头部，即字段名
    writer.writeheader()
    # 将更新后的 commitlist_rows 写入 CSV 文件
    writer.writerows(commitlist_rows)
```