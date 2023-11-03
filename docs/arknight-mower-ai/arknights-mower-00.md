# ArknightMower源码解析 0

# `/opt/arknights-mower/data_update.py`

这段代码的主要作用是获取并上传游戏数据到服务器，并将数据存储在本地文件夹中。以下是具体步骤解释：

1. 导入所需库：import json, os, fontforge, requests。

2. 设置代理服务器，用于发送网络请求。在代码中，使用了一个名为`proxies`的字典，其中包含一个HTTP代理服务器地址，为`http://localhost:11223`。

3. 设置数据存储目录，用于保存游戏数据。在代码中，使用了`datadir`变量，其值为`arknights_mower/data/`。

4. 设置GitHub游戏数据仓库的URL。在代码中，使用了`GitHubURL`变量，其值为从`master`分支下载的数据仓库的URL。

5. 定义了`dump`函数，用于将数据存储到本地文件夹中的指定文件。在函数中，使用了`with`语句和`open`函数，打开一个文件并写入数据。`w`参数表示写入模式，`json.dump`函数用于将数据序列化为JSON格式并保存到文件中。`ensure_ascii=False`和`indent=4`参数表示不对ASCII字符进行转义，并使用4 spaces对数据进行对齐。`default=str`参数表示如果数据类型不是字符串，则默认为字符串类型。

6. 在主程序中，调用`dump`函数将游戏数据保存到本地文件夹中。在程序中，首先导入`fontforge`库和`requests`库，然后使用`proxies`变量中的HTTP代理服务器地址和`datadir`变量中的数据存储目录，设置了一个名为`ARKNIGHTS_MLOR_DATA_DIR`的常量。接下来，定义了`dump`函数，并将`ARKNIGHTS_MLOR_DATA_DIR`和`dump`函数作为参数，将游戏数据保存到本地文件夹中。

完整代码如下：


import json
import os
import fontforge
import requests

proxies = {'http': 'http://localhost:11223'}
datadir = 'arknights_mower/data/'
GitHubURL = 'https://raw.githubusercontent.com/Kengxxiao/ArknightsGameData/master/zh_CN/gamedata/'

def dump(data, filename):
   with open(datadir + filename, 'w') as f:
       json.dump(data, f, ensure_ascii=False, indent=4, default=str)



def upload_data(data, filename, headers):
   with open(datadir + filename, 'w') as f:
       f.write(json.dumps(data, ensure_ascii=False, indent=4, default=str))
   


























































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































```
import json
import os

import fontforge
import requests

proxies = {'http': 'http://localhost:11223'}
datadir = 'arknights_mower/data/'
GitHubURL = 'https://raw.githubusercontent.com/Kengxxiao/ArknightsGameData/master/zh_CN/gamedata/'


def dump(data, filename):
    with open(datadir + filename, 'w') as f:
        json.dump(data, f, ensure_ascii=False, indent=4, default=str)


```

这段代码的作用是获取特定路径下的 Excel 文件内容，并将其存储在两个列表中。这两个列表分别为：所有列的名称和所有不包含显示编号的角色的名称。

具体来说，代码首先使用 requests.get() 方法向 GitHub URL 加上路径的 URL 发送请求，并使用代理（proxies）来防止 IP 被封。然后，它使用 requests.text() 方法获取返回的 JSON 数据，并将其转换为字符串。接着，代码遍历从 JSON 数据中获取的所有角色的名称，并将它们存储到 agent 列表中。最后，代码将 agent 列表和所有不包含显示编号的角色名称存储到两个名为 agent_charset 和 Chinese 的字典中，并将这些字典中的所有名称导出为 JSON 文件。


```
def requests_get(path):
    return requests.get(GitHubURL + path, proxies=proxies).text


agent = []
character_table = json.loads(requests_get('excel/character_table.json'))
for x in character_table.values():
    if x['displayNumber'] is not None:
        agent.append(x['name'].strip())
dump(agent, 'agent.json')

agent_charset = set(''.join(agent))
Chinese, unChinese = [], []
for c in agent_charset:
    if ord(c) < 256:
        unChinese.append(c)
    else:
        Chinese.append(c)
```

这段代码的作用是读取两个文本文件（build/Chinese.txt和build/unChinese.txt），并将它们中的内容合并为一个新的文件（build/newfont.txt）。合并的方式是将所有汉字字符串连接成一个大的字符串，并将其写入到一个名为“newfont.txt”的新文件中。

具体来说，代码首先打开两个文件，一个以写入模式打开（'w'），一个以写入模式打开（'w'），并将这两个文件中的所有内容存储到它们各自的变量中。接着，代码调用了一个名为“build.py”的Python脚本，其中包含一个命令行参数（command）。这个命令行参数是一个Java命令，通过调用fontforge库中的函数，将读取的两个ttf文件中的内容合并为一个新的ttf文件并将其保存到build/newfont.txt中。

最后，代码使用os.system函数执行了一个命令，如果该命令的返回值是False，则会引发一个异常并捕获异常信息。


```
with open('build/Chinese.txt', 'w') as f:
    f.write(''.join(Chinese))
with open('build/unChinese.txt', 'w') as f:
    f.write(''.join(unChinese))

command = 'java.exe -jar FontPruner/sfnttool.jar -c build/Chinese.txt  build/unChinese.txt FontPruner/SourceHanSansSC-Bold.ttf build/SourceHanSansSC-Bold.ttf'
if os.system(command) is False:
    raise Exception('build new font error!' + command)


ttf_file = 'build/SourceHanSansSC-Bold.ttf'
otf_file = 'arknights_mower/fonts/SourceHanSansSC-Bold.otf'

font = fontforge.open(ttf_file)
font.generate(otf_file)
```

这段代码的主要作用是读取和导出一个Excel文件中的数据。下面是具体的解释：

1. `font.close()`：关闭当前打开的字体对象。

2. `chapter = []`：创建一个空列表，用于存储章节的名称。

3. `chapter_table = json.loads(requests_get('excel/chapter_table.json'))`：使用`requests_get`函数从Excel文件中获取chapter_table数据，并使用`json.loads`将其转换为Python可读的JSON格式。

4. `for x in chapter_table.values():`：遍历chapter_table中的每个值，将其存储在变量`x`中。

5. `chapter.append(x['chapterName2'])`：将每个值中的`chapterName2`属性存储在`chapter`列表中。

6. `dump(chapter, 'chapter.json')`：使用`dump`函数将`chapter`列表中的所有元素导出为JSON格式，并将其保存为`chapter.json`文件。

7. `level = {}`：创建一个空字典，用于存储章节级别。

8. `zone = {}`：创建一个空字典，用于存储区域级别。

9. `zone_table = json.loads(requests_get('excel/zone_table.json'))`：使用`requests_get`函数从Excel文件中获取zone_table数据，并使用`json.loads`将其转换为Python可读的JSON格式。

10. `chapterIndex=-1`：将`zone_table`中的章节索引设置为-1，表示当前章节没有对应的级别。


```
font.close()


chapter = []
chapter_table = json.loads(requests_get('excel/chapter_table.json'))
for x in chapter_table.values():
    chapter.append(x['chapterName2'])
dump(chapter, 'chapter.json')


level = {}
zone = {}

zone_table = json.loads(requests_get('excel/zone_table.json'))
chapterIndex = -1
```

这段代码是一个for循环，遍历一个名为zone_table的字典中的'zones'键。

for循环会依次读取字典中的每个键，并执行其中的代码块。在这个代码块中，首先使用x['type']来判断当前遍历的键所属的类别，如果是'MAINLINE'，则执行紧接着的代码块，如果是'WEEKLY'，则执行下下的代码块。

如果是'MAINLINE'，则执行以下代码块：

python
   if x['zoneIndex'] == 0:
       chapterIndex += 1
       zone[x['zoneID']] = {
           'type': x['type'],
           'name': x['zoneNameSecond'],
           'chapterIndex': chapterIndex,
           'zoneIndex': int(x['zoneID'].split('_')[1]),
       }
   else:
       zone[x['zoneID']] = {
           'type': x['type'],
           'name': x['zoneNameSecond'],
           'chapterIndex': None,
           'zoneIndex': None,
       }


如果是'WEEKLY'，则执行以下代码块：

python
   zone[x['zoneID']] = {
       'type': x['type'],
       'name': x['zoneNameSecond'],
       'chapterIndex': None,
       'zoneIndex': None,
   }


在这段代码中，首先通过x['type']来判断当前遍历的键所属的类别，然后执行相应的代码块。如果是'MAINLINE'，则执行以下操作：

1. 如果x['zoneIndex'] == 0，则执行以下代码块：
   a. 计算chapterIndex：章节编号+1
   b. 将当前键值对的zoneID添加到zone字典中，并使用spa特别方法spa.ChapterIndex 来设置其chapterIndex
   c. 将当前键值对的zoneIndex设置为1 + zoneNameSecond除以_的数量
   d. 返回zone字典
   e. 如果chapterIndex为None，则执行next(pageNumber)

如果是'WEEKLY'，则执行以下操作：

1. 将当前键值对的zoneID设置为{'type': x['type'], 'name': x['zoneNameSecond']}
2. 如果chapterIndex为None，则执行next(pageNumber)


```
for x in zone_table['zones'].values():
    if x['type'] == 'MAINLINE':
        if x['zoneIndex'] == 0:
            chapterIndex += 1
        zone[x['zoneID']] = {
            'type': x['type'],
            'name': x['zoneNameSecond'],
            'chapterIndex': chapterIndex,
            'zoneIndex': int(x['zoneID'].split('_')[1]),
        }
    elif x['type'] == 'WEEKLY':
        zone[x['zoneID']] = {
            'type': x['type'],
            'name': x['zoneNameSecond'],
            'chapterIndex': None,
            'zoneIndex': None,
        }

```

这段代码的作用是读取并解析一个Excel文件中的数据，并将其存储在两个变量`stage_table`和`retro_table`中。

具体来说，这个代码首先使用Python的`requests`库发起HTTP GET请求，获取一个名为`stage_table.json`的JSON文件数据，并将其存储在变量`stage_table`中。

接下来，代码使用Python的`json`库将`stage_table`解析成一个Python字典类型。

然后，代码遍历`stage_table`中的`stages`字典的值，即每个`stage`对象。在遍历过程中，代码根据`zoneId`、`canBattleReplay`和`levelId`字段来筛选出符合条件的结果，并将它们存储在另一个变量`level`中。

最后，代码再次使用`requests`库发起HTTP GET请求，获取一个名为`retro_table.json`的JSON文件数据，并将其存储在变量`retro_table`中。


```
stage_table = json.loads(requests_get('excel/stage_table.json'))
for x in stage_table['stages'].values():
    if (
        x['zoneId'] in zone.keys()
        and x['canBattleReplay']
        and not x['levelId'].startswith('Activities')
    ):
        level[x['code']] = {
            'zone_id': x['zoneId'],
            'ap_cost': x['apCost'],
            'code': x['code'],
            'name': x['name'],
        }

retro_table = json.loads(requests_get('excel/retro_table.json'))
```

这段代码使用了两个for循环，第一个for循环遍历了名为“retro_table”的列表中的所有元素，第二个for循环遍历了列表中的每个元素。

在第二个for循环中，如果元素x的类型为1，则执行if语句。如果x的类型为1，则执行以下代码块。这个代码块创建了一个字典，该字典的键是x的retroId，值为{'type': 'BRANCHLINE', 'name': x['name'], 'chapterIndex': None, 'zoneIndex': x['index']}。

如果元素的类型为0，则执行以下代码块。这个代码块创建了一个字典，该字典的键是x的retroId，值为{'type': 'SIDESTORY', 'name': x['name'], 'chapterIndex': None, 'zoneIndex': x['index']}。

总结起来，这段代码的作用是根据元素的类型创建了一个字典，存储了每个元素的retroId键值对。其中，retroId是元素在retro_table列表中的编号，字典的键是retroId，值是该元素的type类型的字典。


```
for x in retro_table['retroActList'].values():
    if x['type'] == 1:
        zone[x['retroId']] = {
            'type': 'BRANCHLINE',
            'name': x['name'],
            'chapterIndex': None,
            'zoneIndex': x['index'],
        }
    elif x['type'] == 0:
        zone[x['retroId']] = {
            'type': 'SIDESTORY',
            'name': x['name'],
            'chapterIndex': None,
            'zoneIndex': x['index'],
        }
```

这段代码的作用是根据游戏中的阶段（stage）和区（zone）信息，将区与对应的代码（code）和等级（level）信息存储在 JSON 文件中。

首先，将 `zoneToRetro` 变量初始化为 `retro_table['zoneToRetro']`，即游戏中的区与区的相对位置。

接着，遍历 `retro_table['stageList'].values()`，即遍历游戏中的所有阶段。在遍历过程中，对于每个阶段 `x`，进行以下判断：

1. 如果 `x` 的 `hardStagedId` 为 `None`，`x` 的 `canBattleReplay` 为 `True`，`x` 的 `zoneId` 末尾为 `1`，并且 `x` 的 `zoneId` 存在于 `zoneToRetro` 字典中，那么将阶段 `x` 对应的代码 `x['code']` 和等级 `x['apCost']` 存储到 `level` 字典中。其中，`x['zoneId']` 存储为键，`x['apCost']` 存储为值。
2. 如果以上条件都不满足，则不做任何操作，仅仅是遍历阶段列表。

最后，将 `zone` 和 `level` 字典分别保存到 JSON 文件中。


```
zoneToRetro = retro_table['zoneToRetro']
for x in retro_table['stageList'].values():
    if x['hardStagedId'] is None and x['canBattleReplay'] and x['zoneId'].endswith('1') and x['zoneId'] in zoneToRetro.keys():
        level[x['code']] = {
            'zone_id': zoneToRetro[x['zoneId']],
            'ap_cost': x['apCost'],
            'code': x['code'],
            'name': x['name'],
        }

dump(zone, 'zone.json')
dump(level, 'level.json')

```