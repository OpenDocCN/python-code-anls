# `.\PokeLLMon\poke_env\data\static\moves\extract_gen8_moves.py`

```py
# 导入 json 和 re 模块
import json
import re
import json

# 初始化存储招式名称和效果的列表
move_name_list = []
move_effect_list = []

# 打开文件 "gen8_raw.txt" 以只读模式
with open("gen8_raw.txt", "r") as f:
    idx = 0
    # 循环读取文件中的每一行数据
    for i in range(2184):
        data = f.readline()
        # 每三行数据为一组，分别提取招式名称和效果
        if idx % 3 == 0:
            move_name = data.split("    ")[0]
            move_name_list.append(move_name)
        elif idx % 3 == 1:
            effect = data[:-1]
            move_effect_list.append(effect)

        idx += 1

# 将招式名称和效果列表组合成字典
move2effect = dict(zip(move_name_list, move_effect_list))

# 打开文件 "gen8moves.json" 以只读模式
with open("gen8moves.json", "r") as f:
    # 加载 JSON 文件内容到 gen8moves 字典中
    gen8moves = json.load(f)

# 初始化新的招式名称到效果的字典
move2effect_new = dict()
# 遍历 gen8moves 字典中的每个招式和信息
for move, info in gen8moves.items():
    try:
        # 尝试从 move2effect 字典中获取招式对应的效果
        effect = move2effect[info['name']]
        # 将招式和效果添加到新的字典中
        move2effect_new[move] = effect
    except:
        # 如果出现异常则继续下一个招式
        continue

# 打开文件 "gen8moves_effect.json" 以写入模式
with open("gen8moves_effect.json", "w") as f:
    # 将新的招式名称到效果的字典以美观的格式写入到文件中
    json.dump(move2effect_new, f, indent=4)
```