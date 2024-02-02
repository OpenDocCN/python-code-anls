# `arknights-mower\arknights_mower\data\__init__.py`

```py
# 导入 json 模块
import json
# 从 pathlib 模块中导入 Path 类
from pathlib import Path
# 从当前包的根目录导入 __rootdir__ 变量
from .. import __rootdir__

# 从 agent.json 文件中加载干员列表数据，存储在 agent_list 变量中
agent_list = json.loads(
    Path(f'{__rootdir__}/data/agent.json').read_text('utf-8'))

# 从 agent-base.json 文件中加载干员基础技能配置数据，存储在 agent_base_config 变量中
# agent_base_config = json.loads(
#     Path(f'{__rootdir__}/data/agent-base.json').read_text('utf-8'))

# 从 base.json 文件中加载地下室中每个房间的名称数据，存储在 base_room_list 变量中
base_room_list = json.loads(
    Path(f'{__rootdir__}/data/base.json').read_text('utf-8'))

# 从 clue.json 文件中加载线索所属阵营的数据，存储在 clue_name 变量中
clue_name = json.loads(
    Path(f'{__rootdir__}/data/clue.json').read_text('utf-8'))

# 从 shop.json 文件中加载商店中出售的商品数据，存储在 shop_items 变量中
shop_items = json.loads(
    Path(f'{__rootdir__}/data/shop.json').read_text('utf-8'))

# 从 ocr.json 文件中加载获取的 OCR 错误数据，存储在 ocr_error 变量中
ocr_error = json.loads(
    Path(f'{__rootdir__}/data/ocr.json').read_text('utf-8'))

# 从 chapter.json 文件中加载章节名称的英文数据，存储在 chapter_list 变量中
chapter_list = json.loads(
    Path(f'{__rootdir__}/data/chapter.json').read_text('utf-8'))

# 从 level.json 文件中加载支持的关卡列表数据，存储在 level_list 变量中
level_list = json.loads(
    Path(f'{__rootdir__}/data/level.json').read_text('utf-8'))

# 从 zone.json 文件中加载开放区域列表数据，存储在 zone_list 变量中
zone_list = json.loads(
    Path(f'{__rootdir__}/data/zone.json').read_text('utf-8'))

# 从 weekly.json 文件中加载支持的每周关卡列表数据，存储在 weekly_zones 变量中
weekly_zones = json.loads(
    Path(f'{__rootdir__}/data/weekly.json').read_text('utf-8'))

# 从 scene.json 文件中加载定义的场景列表数据，存储在 scene_list 变量中
scene_list = json.loads(
    Path(f'{__rootdir__}/data/scene.json').read_text('utf-8'))

# 从 recruit.json 文件中加载招募数据库数据，存储在 recruit_agent 变量中
recruit_agent = json.loads(
    Path(f'{__rootdir__}/data/recruit.json').read_text('utf-8'))

# 初始化招募标签列表，包括 '资深干员' 和 '高级资深干员'
recruit_tag = ['资深干员', '高级资深干员']
# 遍历招募数据库中的值，将所有标签添加到招募标签列表中
for x in recruit_agent.values():
    recruit_tag += x['tags']
# 去除重复的标签，得到最终的招募标签列表
recruit_tag = list(set(recruit_tag))

'''
按tag分类组合干员
'''

# 初始化招募干员列表和稀有度标签列表
recruit_agent_list = {}
rarity_tags = []

# 遍历招募标签列表，为每个标签创建一个空的干员列表和最低等级要求，存储在招募干员列表中
for key in recruit_tag:
    recruit_agent_list[key] = {
        "min_level": 7,
        "agent": []
    }
    # 遍历招募代理商列表中的每个代理商
    for opeartors in recruit_agent:
        # 如果关键字在当前代理商的标签中
        if key in recruit_agent[opeartors]['tags']:
            # 如果当前代理商的星级低于已记录的最低星级
            if recruit_agent[opeartors]['stars'] < recruit_agent_list[key]["min_level"]:
                # 更新最低星级为当前代理商的星级
                recruit_agent_list[key]["min_level"] = recruit_agent[opeartors]['stars']

            # 将当前代理商的信息添加到代理商列表中
            recruit_agent_list[key]["agent"].append(
                {
                    "name": recruit_agent[opeartors]['name'],
                    "level": recruit_agent[opeartors]['stars'],
                })
# 创建一个空列表，用于存储最低等级为5的tag
rarity_tags = []
# 遍历招募特工列表中的每个键
for key in recruit_agent_list:
    # 如果特工的最低等级大于等于5
    if recruit_agent_list[key]['min_level'] >= 5:
        # 将该特工的tag添加到rarity_tags列表中
        rarity_tags.append(key)
```