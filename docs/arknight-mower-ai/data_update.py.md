# `arknights-mower\data_update.py`

```py
# 导入所需的模块
import json
import os
import fontforge
import requests

# 设置代理
proxies = {'http': 'http://localhost:11223'}
# 设置数据目录
datadir = 'arknights_mower/data/'
# 设置 GitHub 数据源的 URL
GitHubURL = 'https://raw.githubusercontent.com/Kengxxiao/ArknightsGameData/master/zh_CN/gamedata/'

# 定义函数，将数据转储到文件
def dump(data, filename):
    with open(datadir + filename, 'w') as f:
        json.dump(data, f, ensure_ascii=False, indent=4, default=str)

# 定义函数，发送 GET 请求获取数据
def requests_get(path):
    return requests.get(GitHubURL + path, proxies=proxies).text

# 初始化代理列表
agent = []
# 从 character_table.json 中获取角色信息，并将角色名添加到代理列表中
character_table = json.loads(requests_get('excel/character_table.json'))
for x in character_table.values():
    if x['displayNumber'] is not None:
        agent.append(x['name'].strip())
# 将代理列表转储到 agent.json 文件中
dump(agent, 'agent.json')

# 初始化代理字符集和非代理字符集
agent_charset = set(''.join(agent))
Chinese, unChinese = [], []
# 将字符根据是否为中文分别添加到代理字符集和非代理字符集中
for c in agent_charset:
    if ord(c) < 256:
        unChinese.append(c)
    else:
        Chinese.append(c)
# 将代理字符集和非代理字符集分别写入到 Chinese.txt 和 unChinese.txt 文件中
with open('build/Chinese.txt', 'w') as f:
    f.write(''.join(Chinese))
with open('build/unChinese.txt', 'w') as f:
    f.write(''.join(unChinese))

# 执行命令，生成新的字体文件
command = 'java.exe -jar FontPruner/sfnttool.jar -c build/Chinese.txt  build/unChinese.txt FontPruner/SourceHanSansSC-Bold.ttf build/SourceHanSansSC-Bold.ttf'
# 如果执行命令成功，则抛出异常
if os.system(command) is False:
    raise Exception('build new font error!' + command)

# 设置字体文件路径
ttf_file = 'build/SourceHanSansSC-Bold.ttf'
otf_file = 'arknights_mower/fonts/SourceHanSansSC-Bold.otf'

# 打开字体文件，生成新的字体文件，并关闭字体文件
font = fontforge.open(ttf_file)
font.generate(otf_file)
font.close()

# 初始化章节列表
chapter = []
# 从 chapter_table.json 中获取章节信息，并将章节名添加到章节列表中
chapter_table = json.loads(requests_get('excel/chapter_table.json'))
for x in chapter_table.values():
    chapter.append(x['chapterName2'])
# 将章节列表转储到 chapter.json 文件中
dump(chapter, 'chapter.json')

# 初始化关卡和区域字典
level = {}
zone = {}

# 从 zone_table.json 中获取区域信息，并处理数据
zone_table = json.loads(requests_get('excel/zone_table.json'))
chapterIndex = -1
for x in zone_table['zones'].values():
    # 在这里继续处理 zone_table 数据
    pass
    # 如果输入字典 x 的 type 键值为 'MAINLINE'，则执行以下操作
    if x['type'] == 'MAINLINE':
        # 如果输入字典 x 的 zoneIndex 键值为 0，则将 chapterIndex 值加一
        if x['zoneIndex'] == 0:
            chapterIndex += 1
        # 将 zone 字典中的 zoneID 键值设为一个新的字典，包含以下键值对
        zone[x['zoneID']] = {
            'type': x['type'],
            'name': x['zoneNameSecond'],
            'chapterIndex': chapterIndex,
            'zoneIndex': int(x['zoneID'].split('_')[1]),
        }
    # 如果输入字典 x 的 type 键值为 'WEEKLY'，则执行以下操作
    elif x['type'] == 'WEEKLY':
        # 将 zone 字典中的 zoneID 键值设为一个新的字典，包含以下键值对
        zone[x['zoneID']] = {
            'type': x['type'],
            'name': x['zoneNameSecond'],
            'chapterIndex': None,
            'zoneIndex': None,
        }
# 从请求中获取 JSON 数据并加载到 stage_table 变量中
stage_table = json.loads(requests_get('excel/stage_table.json'))
# 遍历 stage_table 中的 stages 字段的值
for x in stage_table['stages'].values():
    # 检查条件：x['zoneId'] 在 zone 字典的键中，并且 x['canBattleReplay'] 为真，并且 x['levelId'] 不以 'Activities' 开头
    if (
        x['zoneId'] in zone.keys()
        and x['canBattleReplay']
        and not x['levelId'].startswith('Activities')
    ):
        # 将符合条件的数据存入 level 字典中
        level[x['code']] = {
            'zone_id': x['zoneId'],
            'ap_cost': x['apCost'],
            'code': x['code'],
            'name': x['name'],
        }

# 从请求中获取 JSON 数据并加载到 retro_table 变量中
retro_table = json.loads(requests_get('excel/retro_table.json'))
# 遍历 retro_table 中的 retroActList 字段的值
for x in retro_table['retroActList'].values():
    # 检查条件：x['type'] 为 1
    if x['type'] == 1:
        # 将符合条件的数据存入 zone 字典中
        zone[x['retroId']] = {
            'type': 'BRANCHLINE',
            'name': x['name'],
            'chapterIndex': None,
            'zoneIndex': x['index'],
        }
    # 检查条件：x['type'] 为 0
    elif x['type'] == 0:
        # 将符合条件的数据存入 zone 字典中
        zone[x['retroId']] = {
            'type': 'SIDESTORY',
            'name': x['name'],
            'chapterIndex': None,
            'zoneIndex': x['index'],
        }

# 从 retro_table 中获取 zoneToRetro 字段的值
zoneToRetro = retro_table['zoneToRetro']
# 遍历 retro_table 中的 stageList 字段的值
for x in retro_table['stageList'].values():
    # 检查条件：x['hardStagedId'] 为 None，并且 x['canBattleReplay'] 为真，并且 x['zoneId'] 以 '1' 结尾，并且 x['zoneId'] 在 zoneToRetro 字典的键中
    if x['hardStagedId'] is None and x['canBattleReplay'] and x['zoneId'].endswith('1') and x['zoneId'] in zoneToRetro.keys():
        # 将符合条件的数据存入 level 字典中
        level[x['code']] = {
            'zone_id': zoneToRetro[x['zoneId']],
            'ap_cost': x['apCost'],
            'code': x['code'],
            'name': x['name'],
        }

# 将 zone 字典中的数据保存到 'zone.json' 文件中
dump(zone, 'zone.json')
# 将 level 字典中的数据保存到 'level.json' 文件中
dump(level, 'level.json')
```