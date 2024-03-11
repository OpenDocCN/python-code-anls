# `.\PokeLLMon\poke_env\data\static\abilities\construct_ability_json.py`

```py
# 导入 pandas 库，用于处理数据
import pandas as pd
# 导入 json 库，用于处理 JSON 数据

# 从 "raw.txt" 文件中读取数据，使用制表符作为分隔符
X = pd.read_csv("raw.txt", "\t")

# 从 X 中提取 Name 列的数值，转换为列表
name = list(X.Name.values)
# 从 X 中提取 Description 列的数值，转换为列表
description = list(X.Description.values)
# 将 name 列中的每个元素转换为小写，并去除空格后存储到 name_new 列表中
name_new = list(map(lambda x: x.lower().replace(" ", ""), name))

# 创建空字典 ability_dict
ability_dict = {}

# 遍历 name 列的长度
for i in range(len(name)):
    # 将 name_new[i] 作为键，name[i] 和 description[i] 组成的字典作为值，存储到 ability_dict 中
    ability_dict[name_new[i]] = {"name": name[i], "effect": description[i]}

# 打印 "pause"
print("pause")

# 打开 "ability_effect.json" 文件，以写入模式打开，文件对象为 f
with open("ability_effect.json", "w") as f:
    # 将 ability_dict 写入到 f 中，格式化输出，缩进为 4
    json.dump(ability_dict, f, indent=4)

# 打印 "pause"
print("pause")
```