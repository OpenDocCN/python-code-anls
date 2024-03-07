# `.\PokeLLMon\poke_env\data\static\items\construct_item_json.py`

```py
# 导入 pandas 和 json 模块
import pandas as pd
import json

# 从 "raw.txt" 文件中读取数据，使用制表符作为分隔符
X = pd.read_csv("raw.txt", "\t")

# 获取 Name、Effect 和 Category 列的数值
name = list(X.Name.values)
effect = list(X.Effect.values)
category = list(X.Category.values)

# 创建空字典 item_dict
item_dict = {}

# 遍历 name 列的长度
for i in range(len(name)):
    # 将 name 列中的值按 " icon " 分割，取第一个部分作为新的名称
    new_name = name[i].split(" icon ")[0]

    # 如果 effect 列中的值不是 NaN
    if str(effect[i]) != "nan":
        # 将新名称转换为小写并去除空格，作为字典的键，值为包含名称和效果的字典
        item_dict[new_name.lower().replace(" ", "")] = {"name":new_name, "effect":effect[i]}

# 打印 "pause"
print("pause")

# 将 item_dict 写入到 "item_effect.json" 文件中，格式化输出，缩进为 4
with open("item_effect.json", "w") as f:
    json.dump(item_dict, f, indent=4)

# 打印 "pause"
print("pause")
```