# ArknightMower源码解析 1

# `diy.py`

这段代码是一个用于解决Arknights Mower游戏的AI的脚本。它实现了以下功能：

1. 导入time模块以获取时间相关的数据和函数；
2. 从datetime模块中导入日期时间和日期函数；
3. 从atestyne.py中导入atixture；
4. 从json模块中导入JSON数据格式；
5. 从os模块中导入文件操作；
6. 从arknights_mower.solvers.base_schedule模块中导入基于任务的计划者；
7. 从arknights_mower.strategy模块中导入策略；
8. 从arknights_mower.utils.device模块中导入设备；
9. 从arknights_mower.utils.email模块中导入电子邮件模板；
10. 从arknights_mower.utils.log模块中导入日志记录；
11. 从arknights_mower.utils.config模块中导入游戏配置；
12. 从arknights_mower.utils.simulator模块中导入游戏模拟器；
13. 从arknights_mower.utils.operators模块中导入操作符；
14. 从arknights_mower.utils.exceptions模块中导入异常类；
15. 在脚本的开始时，使用初始化日志记录函数初始化日志记录；
16. 通过taskset任务集，使用solver解决游戏中的AI；
17. 使用电子邮件模板发送游戏中的提醒。


```py
import time
from datetime import datetime
import atexit
import json
import os

from arknights_mower.solvers.base_schedule import BaseSchedulerSolver
from arknights_mower.strategy import Solver
from arknights_mower.utils.device import Device
from arknights_mower.utils.email import task_template
from arknights_mower.utils.log import logger, init_fhlr
from arknights_mower.utils import config
from arknights_mower.utils.simulator import restart_simulator
# 下面不能删除
from arknights_mower.utils.operators import Operators, Operator, Dormitory
```

這是一個 Roblox 遊戲中的分身(squad)的設定。 Roblox 是一個線上多人遊戲平台，玩家可以在其上建立自己的遊戲世界。

這個字典中定義了一個名為 "maatouch" 的 squad，擁有以下特徵：

* "mode": 0，这意味着它不會進行积极主动的搜索，也不會自動組建聯繫，並且不會使用追蹤器。
* "investment_enabled": True，這表示該 squad 可以使用投資功能。
* "stop_when_investment_full": False，這表示在投資沒有完成時，該 squad 不會停止。
* "refresh_trader_with_dice": True，這表示投資器可以在他們的選擇中使用 Dice。
* "weekly_plan": 一個包含 7 天周期的 weekly_plan 列表，用於設置該 squad 在每個星期期的執行計劃。每個周期都有兩個 stage，一個包含主要腳本，一個包含次要腳本。
* "roles": "取長补短"，這表示該 squad 每個周期都有一個 "PR-C-2" 的 roles。這將導致該 squad 參與到 PR-C-2 的練習中。
* "use_support": False，這表示該 squad 不會使用支持角色。
* "use_nonfriend_support": False，這表示該 squad 不會使用非好友的支持。
* "senkou_dice_api_key": "dgSF43jRQefiGxRqjKcNwQ機制翻"，這表示該 squad 使用的是 Senkou Dice API 密鑰，並需要該 API 密鑰才能使用該 API。

這個字典也定義了該 squad 的其他屬性，如 "sleep_min" 和 "sleep_max"，以及 "weekly_plan"。weekly_plan 是一個列表，其中包含一個 s汽油 的 weekly_plan 屬性。每個周期都包含一個 stage 和一個 roles，並指定要執行的 task。


```py
from arknights_mower.utils.scheduler_task import SchedulerTask

email_config= {
    # 发信账户
    'account':"xxx@qq.com",
    # 在QQ邮箱“帐户设置-账户-开启SMTP服务”中，按照指示开启服务获得授权码
    'pass_code':'xxx',
    # 收件人邮箱
    'receipts':['任何邮箱'],
    # 是否启用邮件提醒
    'mail_enable':False,
    # 邮件主题
    'subject': '任务数据'
}
maa_config = {
    "maa_enable":True,
    # 请设置为存放 dll 文件及资源的路径
    "maa_path":'F:\\MAA-v4.10.5-win-x64',
    # 请设置为存放 dll 文件及资源的路径
    "maa_adb_path":"D:\\Program Files\\Nox\\bin\\adb.exe",
    # adb 地址
    "maa_adb":['127.0.0.1:62001'],
    # maa 运行的时间间隔，以小时计
    "maa_execution_gap":4,
    # 以下配置，第一个设置为true的首先生效
    # 是否启动肉鸽
    "roguelike":False,
    # 是否启动生息演算
    "reclamation_algorithm":False,
    # 是否启动保全派驻
    "stationary_security_service": True,
    # 保全派驻类别 1-2
    "sss_type": 2,
    # 导能单元类别 1-3
    "ec_type":1,
    "copilot_file_location": "F:\\MAA-v4.10.5-win-x64\\resource\\copilot\\SSS_雷神工业测试平台_浊蒂版.json",
    "copilot_loop_times":10,
    "last_execution": datetime.now(),
    "blacklist":"家具,碳,加急许可",
    "rogue_theme":"Sami",
    "buy_first":"招聘许可",
    "recruit_only_4": True,
    "credit_fight": False,
    "recruitment_time": None,
    'mall_ignore_when_full': True,
    "touch_option": "maatouch",
    "conn_preset": "General",
    "rogue": {
        "squad": "指挥分队",
        "roles": "取长补短",
        "use_support": False,
        "core_char":"",
        "use_nonfriend_support": False,
        "mode": 0,
        "investment_enabled": True,
        "stop_when_investment_full": False,
        "refresh_trader_with_dice": True
    },
    "sleep_min":"",
    "sleep_max":"",
    # "weekly_plan": [{"weekday": "周一", "stage": ['AP-5'], "medicine": 0},
    #                 {"weekday": "周二", "stage": ['PR-D-1'], "medicine": 0},
    #                 {"weekday": "周三", "stage": ['PR-C-2'], "medicine": 0},
    #                 {"weekday": "周四", "stage": ['AP-5'], "medicine": 0},
    #                 {"weekday": "周五", "stage": ['PR-A-2'], "medicine": 0},
    #                 {"weekday": "周六", "stage": ['AP-5'], "medicine": 0},
    #                 {"weekday": "周日", "stage": ['AP-5'], "medicine": 0}],
    "weekly_plan": [{"weekday": "周一", "stage": [''], "medicine": 0},
                    {"weekday": "周二", "stage": [''], "medicine": 0},
                    {"weekday": "周三", "stage": [''], "medicine": 0},
                    {"weekday": "周四", "stage": [''], "medicine": 0},
                    {"weekday": "周五", "stage": [''], "medicine": 0},
                    {"weekday": "周六", "stage": [''], "medicine": 0},
                    {"weekday": "周日", "stage": [''], "medicine": 0}]
}

```

这段代码是一个 Python 编程语言中的一个类，用于模拟器相关设置。

该类包含一个名为 "simulator" 的字典，其中包含模拟器的名称、多开编号、用于执行模拟器命令的文件路径等设置。

具体来说，代码中定义了一个名为 "simulator" 的字典，其中包含以下几个键值对：

- "name": 模拟器的名称，为 "夜神"。
- "index": 多开编号，用于在模拟器助手中最左侧显示。
- "simulator_folder": 用于执行模拟器命令的文件路径，为 "D:\\Program Files\\Nox\\bin"。

接着，代码中定义了一个名为 "free_blacklist" 的列表，用于干员宿舍回复阈值。

具体来说，代码中使用了 Python 内置的 "黑名单" 数据结构，将一些干员ID加入到了列表中。这里需要指出的是，黑名单中的干员在游戏中并不会真正被封禁，而仅仅是影响了该干员在某些活动中的参与度。

此外，代码中还定义了一个名为 "UpperLimit" 的变量，用于干员宿舍回复阈值。这里需要说明的是，这个阈值只是一个默认值，具体值会在 agent-base.json 文件中进行修改。


```py
# 模拟器相关设置
simulator= {
    "name":"夜神",
    # 多开编号，在模拟器助手最左侧的数字
    "index":2,
    # 用于执行模拟器命令
    "simulator_folder":"D:\\Program Files\\Nox\\bin"
}

# Free (宿舍填充)干员安排黑名单
free_blacklist= []

# 干员宿舍回复阈值
    # 高效组心情低于 UpperLimit  * 阈值 (向下取整)的时候才会会安排休息
    # UpperLimit：默认24，特殊技能干员如夕，令可能会有所不同(设置在 agent-base.json 文件可以自行更改)
```

这段代码是一个人工智能程序，它设置了一些变量和常量，然后进入了一个 if 语句。

if 语句的条件是：“如果 all in 贸易站则 不需要修改设置”，说明在程序运行时，会检查贸易站是否都已经被占领。如果贸易站已经被占领，则不需要修改设置。

接下来，程序设置了一个变量 drone_room，并将其设置为 None。然后，程序设置了一个变量 drone_execution_gap，并将其设置为 4。

程序还设置了一个变量 relay_room，并将其设置一个空列表。然后，程序定义了一个变量 room_list，并将其设置为 []。

接下来，程序进入了一个 for 循环，遍历一个名为 "relay_room" 的列表。

在循环中，程序执行以下操作：程序将变量 trade_room 的所有元素复制到列表中，并替换原来的 trade_room。然后，程序将列表所有元素（也就是房间名称）添加到列表中。最后，程序将 "relay_room" 的索引值加 1。

接下来，程序进入了一个 if 语句。如果列表为空，则说明没有找到 "贸易站"，程序将执行一个 if 语句，并将变量 trade_room 的值设置为 0。程序还会执行一个 if 语句，并将变量 drone_execution_gap 的值设置为 0。

if 语句的条件是：“如果需要无人机加速其他房间则可以修改成房间名字如 'room_1_1'”，说明在程序运行时，会检查是否需要无人机加速其他房间。如果需要无人机加速其他房间，则可以修改为房间名字，比如将变量 drone_room 设置为 'room_1_1'。

程序还设置了一个变量 timezone_offset，并将其设置为 0。然后，程序进入了一个 for 循环，遍历列表 "relay_room"。

在循环中，程序执行以下操作：程序将变量 current_room 的所有元素复制到列表中，并替换原来的 current_room。然后，程序将列表中所有元素的索引值（也就是房间编号）更改为（当前房间编号-1）。最后，程序将 "relay_room" 的索引值加 1。

接下来，程序进入了一个 if 语句。如果列表为空，则说明没有找到 "贸易站"，程序将执行一个 if 语句，并将变量 trade_room 的值设置为 0。程序还会执行一个 if 语句，并将变量 drone_execution_gap 的值设置为 0。


```py
resting_threshold = 0.5

# 跑单如果all in 贸易站则 不需要修改设置
# 如果需要无人机加速其他房间则可以修改成房间名字如 'room_1_1'
drone_room = None
# 无人机执行间隔时间 （小时）
drone_execution_gap = 4

reload_room = []

# 基地数据json文件保存名
state_file_name = 'state.json'

# 邮件时差调整
timezone_offset = 0

```

这段代码定义了一套全自动基建的排班计划，名为plan_1。其中，agent为常驻高效组的干员名，group为干员编队，replacement为替换组干员备选。程序的主要功能包括以下几点：

1. 暖机干员的自动换班：在每天开始时，自动让除agent以外的所有干员换班，同时将其他正在休息的干员赶出宿舍。

2. 尽量安排多的替换干员：尝试安排不同干员的替换人员，避免替换人员之间的冲突。

3. 龙舌兰和但书默认为插拔干员：必须放在replacement的第一位。

4. 请把你所安排的替换组写入replacement：否则程序可能报错。

5. 替换组会按照从左到右的优先级选择可以编排的干员：优先级高的干员可以被优先安排。

6. 宿舍常驻干员不会被替换：所以不需要设置替换组。

7. 宿舍空余位置请编写为Free：至少安排一个群补和一个单补以达到最大恢复效率。

8. 宿舍恢复速率务必1-4从高到低排列：从高到低排列，以保证宿舍能够尽快地恢复。

9. 如果有菲亚梅塔则需要安排replacement：建议干员至少为三。

10. 菲亚梅塔会从replacemt中找最低心情的进行充能：根据replacemt中最低心情的干员进行充能。


```py
# 全自动基建排班计划：
# 这里定义了一套全自动基建的排班计划 plan_1
# agent 为常驻高效组的干员名

# group 为干员编队，你希望任何编队的人一起上下班则给他们编一样的名字
# replacement 为替换组干员备选
    # 暖机干员的自动换班
        # 目前只支持一个暖机干员休息
        # ！！ 会吧其他正在休息的暖机干员赶出宿舍
    # 请尽量安排多的替换干员，且尽量不同干员的替换人员不冲突
    # 龙舌兰和但书默认为插拔干员 必须放在 replacement的第一位
    # 请把你所安排的替换组 写入replacement 否则程序可能报错
    # 替换组会按照从左到右的优先级选择可以编排的干员
    # 宿舍常驻干员不会被替换所以不需要设置替换组
        # 宿舍空余位置请编写为Free，请至少安排一个群补和一个单补 以达到最大恢复效率
        # 宿管必须安排靠左，后面为填充干员
        # 宿舍恢复速率务必1-4从高到低排列
    # 如果有菲亚梅塔则需要安排replacement 建议干员至少为三
        # 菲亚梅塔会从replacment里找最低心情的进行充能
```

这看起来像是一个匹配地图的AI程序。在这个地图中，房间1到4分别代表不同的区域。每个房间都有一个AI作为替代者，用于在玩家无法进入这个房间时代替玩家。此外，地图中还包括一个 substring，可能是正在寻找另一个AI 模型。


```py
plan = {
    # 阶段 1
    "default": "plan_1",
    "plan_1": {
        # 中枢
        'central': [{'agent': '焰尾', 'group': '红松骑士', 'replacement': [ "阿米娅","凯尔希",]},
                    {'agent': '琴柳', 'group': '夕', 'replacement': [ "阿米娅","凯尔希","玛恩纳"]},
                    {'agent': '重岳', 'group': '', 'replacement': ["玛恩纳", "清道夫", "凯尔希", "阿米娅", '坚雷']},
                    {'agent': '夕', 'group': '夕', 'replacement': ["阿米娅","凯尔希","玛恩纳", "清道夫", "阿米娅", '坚雷']},
                    {'agent': '令', 'group': '', 'replacement': ["玛恩纳", "清道夫", "凯尔希", "阿米娅", '坚雷']},
                    ],
        'contact': [{'agent': '桑葚', 'group': '', 'replacement': ['艾雅法拉']}],
        # 宿舍
        'dormitory_1': [{'agent': '流明', 'group': '', 'replacement': []},
                        {'agent': '闪灵', 'group': '', 'replacement': []},
                        {'agent': 'Free', 'group': '', 'replacement': []},
                        {'agent': 'Free', 'group': '', 'replacement': []},
                        {'agent': 'Free', 'group': '', 'replacement': []}
                        ],
        'dormitory_2': [{'agent': '杜林', 'group': '', 'replacement': []},
                        {'agent': '蜜莓', 'group': '', 'replacement': []},
                        {'agent': '褐果', 'group': '', 'replacement': []},
                        {'agent': 'Free', 'group': '', 'replacement': []},
                        {'agent': 'Free', 'group': '', 'replacement': []}
                        ],
        'dormitory_3': [{'agent': '车尔尼', 'group': '', 'replacement': []},
                        {'agent': '斥罪', 'group': '', 'replacement': []},
                        {'agent': '爱丽丝', 'group': '', 'replacement': []},
                        {'agent': '桃金娘', 'group': '', 'replacement': []},
                        {'agent': 'Free', 'group': '', 'replacement': []}
                        ],
        'dormitory_4': [{'agent': '波登可', 'group': '', 'replacement': []},
                        {'agent': '夜莺', 'group': '', 'replacement': []},
                        {'agent': '菲亚梅塔', 'group': '', 'replacement': ['重岳', '令', '乌有']},
                        {'agent': 'Free', 'group': '', 'replacement': []},
                        {'agent': 'Free', 'group': '', 'replacement': []}],
        'factory': [{'agent': '年', 'replacement': ['九色鹿', '芳汀'], 'group': '夕'}],
        # 会客室
        'meeting': [{'agent': '伊内丝', 'replacement': ['陈', '红', '远山'], 'group': ''},
                    {'agent': '见行者', 'replacement': ['陈', '红', '星极'], 'group': ''}],
        'room_1_1': [{'agent': '乌有', 'group': '', 'replacement': ['伺夜']},
                     {'agent': '空弦', 'group': '图耶', 'replacement': ['龙舌兰', '鸿雪']},
                     {'agent': '伺夜', 'group': '图耶', 'replacement': ['但书','图耶']},
                     # {'agent': '伺夜', 'group': '图耶', 'replacement': ['但书','能天使']},
                     # {'agent': '空弦', '鸿雪': '图耶', 'replacement': ['龙舌兰', '雪雉']}
                     ],
        'room_1_2': [{'agent': '槐琥', 'group': '槐琥', 'replacement': ['贝娜']},
                     {'agent': '砾', 'group': '槐琥', 'Type': '', 'replacement': ['泡泡']},
                     {'agent': '至简', 'group': '槐琥', 'replacement': ['火神']}],
        'room_1_3': [{'agent': '承曦格雷伊', 'group': '异客', 'replacement': ['炎狱炎熔', '格雷伊']}],
        'room_2_2': [{'agent': '温蒂', 'group': '异客', 'replacement': ['火神']},
                     # {'agent': '异客', 'group': '异客', 'Type': '', 'replacement': ['贝娜']},
                     {'agent': '异客', 'group': '异客', 'Type': '', 'replacement': ['贝娜']},
                     {'agent': '森蚺', 'group': '异客', 'replacement': ['泡泡']}],
        'room_3_1': [{'agent': '稀音', 'group': '稀音', 'replacement': ['贝娜']},
                     {'agent': '帕拉斯', 'group': '稀音', 'Type': '', 'replacement': ['泡泡']},
                     {'agent': '红云', 'group': '稀音', 'replacement': ['火神']}],
        'room_2_3': [{'agent': '澄闪', 'group': '澄闪', 'replacement': ['炎狱炎熔', '格雷伊']}],
        'room_2_1': [{'agent': '食铁兽', 'group': '食铁兽', 'replacement': ['泡泡']},
                     {'agent': '断罪者', 'group': '食铁兽', 'replacement': ['火神']},
                     {'agent': '截云', 'group': '夕', 'replacement': ['迷迭香']}],
        'room_3_2': [{'agent': '灰毫', 'group': '红松骑士', 'replacement': ['贝娜']},
                     {'agent': '远牙', 'group': '红松骑士', 'Type': '', 'replacement': ['泡泡']},
                     {'agent': '野鬃', 'group': '红松骑士', 'replacement': ['火神']}],
        'room_3_3': [{'agent': '雷蛇', 'group': '澄闪', 'replacement': ['炎狱炎熔', '格雷伊']}]
    }
}

```

This is a JSON object that represents a "Grade" system in Final Fantasy VII. It includes要素 such as the name, current HP, and exhaust requirements for each character. The characters are arranged in a specific order, as indicated by the "ArrangeOrder" key.

The object starts with a summary of the current HP and exhaust requirements for each character. Then, for each character, additional information is added including their name, the current HP, and the name of the character they are exhausting.

For example, "During Flaw, F竟可以突破重圆的技能攻击，直接咏唱破界辅助。", or "K最重要，突破L2后技能会给出超环的加成。", etc.

This object can be used for various purposes such as managing the team's strategy, preparing for boss fights, and understanding the current party's capabilities.


```py
# UpperLimit、LowerLimit：心情上下限
# ExhaustRequire：是否强制工作到红脸再休息
# ArrangeOrder：指定在宿舍外寻找干员的方式
# RestInFull：是否强制休息到24心情再工作，与ExhaustRequire一起帮助暖机类技能工作更长时间
# RestingPriority：休息优先级，低优先级不会使用单回技能。

agent_base_config = {
    "Default": {"UpperLimit": 24, "LowerLimit": 0, "ExhaustRequire": False, "ArrangeOrder": [2, "false"],
                "RestInFull": False,"Workaholic":False},
    "令": {"LowerLimit": 12,"ArrangeOrder": [2, "true"]},
    "夕": {"UpperLimit": 12, "ArrangeOrder": [2, "true"]},
    "稀音": {"ExhaustRequire": True, "ArrangeOrder": [2, "true"], "RestInFull": True},
    "巫恋": {"ArrangeOrder": [2, "true"]},
    "柏喙": {"ExhaustRequire": True, "ArrangeOrder": [2, "true"]},
    "龙舌兰": {"ArrangeOrder": [2, "true"]},
    "空弦": {"ArrangeOrder": [2, "true"], "RestingPriority": "low"},
    "伺夜": {"ArrangeOrder": [2, "true"], "RestingPriority": "low"},
    "绮良": {"ArrangeOrder": [2, "true"]},
    "但书": {"ArrangeOrder": [2, "true"]},
    "泡泡": {"ArrangeOrder": [2, "true"]},
    "火神": {"ArrangeOrder": [2, "true"]},
    "黑键": {"ArrangeOrder": [2, "true"]},
    "波登可": {"ArrangeOrder": [2, "false"]},
    "夜莺": {"ArrangeOrder": [2, "false"]},
    "菲亚梅塔": {"ArrangeOrder": [2, "false"]},
    "流明": {"ArrangeOrder": [2, "false"]},
    "蜜莓": {"ArrangeOrder": [2, "false"]},
    "闪灵": {"ArrangeOrder": [2, "false"]},
    "杜林": {"ArrangeOrder": [2, "false"]},
    "褐果": {"ArrangeOrder": [2, "false"]},
    "车尔尼": {"ArrangeOrder": [2, "false"]},
    "安比尔": {"ArrangeOrder": [2, "false"]},
    "爱丽丝": {"ArrangeOrder": [2, "false"]},
    "桃金娘": {"ArrangeOrder": [2, "false"]},
    "帕拉斯": {"RestingPriority": "low"},
    "红云": {"RestingPriority": "low", "ArrangeOrder": [2, "true"]},
    "承曦格雷伊": {"ArrangeOrder": [2, "true"], "RestInFull": True},
    "乌有": {"ArrangeOrder": [2, "true"], "RestingPriority": "low"},
    "图耶": {"ArrangeOrder": [2, "true"]},
    "鸿雪": {"ArrangeOrder": [2, "true"]},
    "孑": {"ArrangeOrder": [2, "true"]},
    "清道夫": {"ArrangeOrder": [2, "true"]},
    "临光": {"ArrangeOrder": [2, "true"]},
    "杜宾": {"ArrangeOrder": [2, "true"]},
    "焰尾": {"RestInFull": True},
    "重岳": {"ArrangeOrder": [2, "true"]},
    "坚雷": {"ArrangeOrder": [2, "true"]},
    "年": {"RestingPriority": "low"},
    "伊内丝": {"ExhaustRequire": True, "ArrangeOrder": [2, "true"], "RestInFull": True},
    "铅踝":{"LowerLimit": 8,"UpperLimit": 12},
}


```

这段代码是一个 Python 程序，主要包括两个函数：debuglog() 和 savelog()。

debuglog() 函数的作用是在屏幕上输出调试信息，方便调试和报错。具体来说，该函数通过调用 config.LOGFILE_PATH 配置项中的指定日志文件输出调试信息。

savelog() 函数的作用是指定日志和截屏的保存位置，方便调试和报错。具体来说，该函数指定了日志和截图保存的目录，以及截图的最大数量。此外，该函数还通过调用 config.PASSWORD 配置项中的指定用户密码来设置屏幕截图时需要输入的密码。

总的来说，这两个函数都是在为开发者在调试和报错过程中提供方便的工具和信息。


```py
def debuglog():
    '''
    在屏幕上输出调试信息，方便调试和报错
    '''
    logger.handlers[0].setLevel('DEBUG')


def savelog():
    '''
    指定日志和截屏的保存位置，方便调试和报错
    调试信息和截图默认保存在代码所在的目录下
    '''
    config.LOGFILE_PATH = './log'
    config.SCREENSHOT_PATH = './screenshot'
    config.SCREENSHOT_MAXNUM = 30
    config.ADB_DEVICE = maa_config['maa_adb']
    config.ADB_CONNECT = maa_config['maa_adb']
    config.MAX_RETRYTIME = 10
    config.PASSWORD = '你的密码'
    config.APPNAME = 'com.hypergryph.arknights'  # 官服
    config.TAP_TO_LAUNCH["enable"] = False
    config.TAP_TO_LAUNCH["x"], config.TAP_TO_LAUNCH["y"] = 0,0
    #  com.hypergryph.arknights.bilibili   # Bilibili 服
    config.ADB_BINARY = ['F:\\MAA-v4.20.0-win-x64\\adb\\platform-tools\\adb.exe']
    init_fhlr()


```

scheduler是 Dragoneer 的核心调度器，负责处理 Dragoneer 任务的调度。在本回答中，scheduler 的实现主要分为以下几个步骤：

1. 读取用户配置文件中的相关参数，包括 device、recog、tasks、free_blacklist 等。
2. 初始化 Dragoneer 的设备、recog 和任务列表。
3. 读取用户设置中的心情开关，以便在需要时自动切换到全自动模式。
4. 设置基础调度器的一些参数，包括最大休息时间、resting 列表等。
5. 初始化调度器，包括计划、全局计划和当前计划等。
6. 运行调度器，并定期检查是否有异常情况。
7. 返回调度器实例，以便其他程序使用。

调度器的核心实现是 `base_scheduler`，这个类继承自 `scheduler` 类，并覆盖了一些方法，如 `scheduler.device`、`scheduler.recog`、`scheduler.tasks` 等。具体的实现主要在 `base_scheduler.py` 文件中。


```py
def inialize(tasks, scheduler=None):
    device = Device()
    cli = Solver(device)
    if scheduler is None:
        base_scheduler = BaseSchedulerSolver(cli.device, cli.recog)
        base_scheduler.package_name = config.APPNAME
        base_scheduler.operators = {}
        base_scheduler.global_plan = plan
        base_scheduler.current_base = {}
        base_scheduler.resting = []
        base_scheduler.current_plan = base_scheduler.global_plan[base_scheduler.global_plan["default"]]
        # 同时休息最大人数
        base_scheduler.max_resting_count = 4
        base_scheduler.tasks = tasks
        # 读取心情开关，有菲亚梅塔或者希望全自动换班得设置为 true
        base_scheduler.read_mood = True
        base_scheduler.last_room = ''
        base_scheduler.free_blacklist = free_blacklist
        base_scheduler.resting_threshold = resting_threshold
        base_scheduler.MAA = None
        base_scheduler.email_config = email_config
        base_scheduler.ADB_CONNECT = config.ADB_CONNECT[0]
        base_scheduler.maa_config = maa_config
        base_scheduler.error = False
        base_scheduler.drone_count_limit = 92  # 无人机高于于该值时才使用
        base_scheduler.drone_room = drone_room
        base_scheduler.drone_execution_gap = drone_execution_gap
        base_scheduler.agent_base_config = agent_base_config
        base_scheduler.run_order_delay = 10  # 跑单提前10分钟运行
        base_scheduler.reload_room = reload_room
        return base_scheduler
    else:
        scheduler.device = cli.device
        scheduler.recog = cli.recog
        scheduler.handle_error(True)
        return scheduler

```

这段代码定义了两个函数，save_state()和load_state()。这两个函数都在一个名为state_file_name的文件中读写。save_state()函数首先检查base_scheduler对象是否为空，如果是，则执行以下操作并将base_scheduler.op_data对象的字符串表示的Python对象的变量值写入到state_file_name文件中。如果base_scheduler对象不为空，则执行以下操作：从base_scheduler.op_data对象中获取op_data字典，并将其中的Python对象的变量值写入到state_file_name文件中。

load_state()函数首先检查state_file_name文件是否存在。如果不存在，则返回None。如果文件存在，则将其读取并返回其中的state字典。接下来，遍历state字典中的所有键值对，并将其存储为operators字典。然后，对于每个字典中的键，如果其time_stamp字段未被设置为None，则将其转换为datetime.datetime类型，并将其时间戳设置为当前时间戳。如果time_stamp已被设置为None，则将该字典的time_stamp设置为None。最后，返回operators字典。


```py
def save_state():
    with open(state_file_name, 'w') as f:
        if base_scheduler is not None and base_scheduler.op_data is not None:
            json.dump(vars(base_scheduler.op_data), f, default=str)

def load_state():
    if not os.path.exists(state_file_name):
        return None

    with open(state_file_name, 'r') as f:
        state = json.load(f)
    operators = {k: eval(v) for k, v in state['operators'].items()}
    for k, v in operators.items():
        if not v.time_stamp == 'None':
            v.time_stamp = datetime.strptime(v.time_stamp, '%Y-%m-%d %H:%M:%S.%f')
        else:
            v.time_stamp = None
    return operators


```

This is a Python script that appears to be the command-line interface (CLI) for a simulated war Simulator. It has several tasks that the simulator can perform, including recruiting new soldiers, assigning them to different teams, and conducting battles.

The script has a main function that defines several helper functions and variables. The main function initializes the simulator and tasks several times.

The `tasks` variable is a list of tasks that the simulator can perform. The tasks are defined as a list of conditions that, if the condition is met, the simulator will perform the corresponding task.

The `base_scheduler` variable is an instance of a class that represents the core scheduler of the simulator. It is responsible for managing the tasks that the simulator can perform.

The `op


```py
def simulate():
    '''
    具体调用方法可见各个函数的参数说明
    '''
    global ope_list, base_scheduler
    # 第一次执行任务
    taskstr = "SchedulerTask(time='2023-06-11 21:39:15.108665',task_plan={'room_3_2': ['Current', '但书', '龙舌兰']},task_type='room_3_2',meta_flag=False)||SchedulerTask(time='2023-06-11 21:44:48.187074',task_plan={'room_2_1': ['砾', '槐琥', '斑点']},task_type='dorm0,dorm1,dorm2',meta_flag=False)||SchedulerTask(time='2023-06-11 22:17:53.720905',task_plan={'room_1_1': ['Current', '龙舌兰', '但书']},task_type='room_1_1',meta_flag=False)||SchedulerTask(time='2023-06-11 23:02:10.469026',task_plan={'meeting': ['Current', '见行者']},task_type='dorm3',meta_flag=False)||SchedulerTask(time='2023-06-11 23:22:15.236154',task_plan={},task_type='菲亚梅塔',meta_flag=False)||SchedulerTask(time='2023-06-12 11:25:55.925731',task_plan={},task_type='impart',meta_flag=False)||SchedulerTask(time='2023-06-12 11:25:55.926731',task_plan={},task_type='maa_Mall',meta_flag=False)"
    tasks = [eval(t) for t in taskstr.split("||")]
    for t in tasks:
        t.time = datetime.strptime(t.time, '%Y-%m-%d %H:%M:%S.%f')
    reconnect_max_tries = 10
    reconnect_tries = 0
    success = False
    while not success:
        try:
            base_scheduler = inialize(tasks)
            success = True
        except Exception as E:
            reconnect_tries+=1
            if reconnect_tries <3:
                restart_simulator(simulator)
                continue
            else:
                raise E
    if base_scheduler.recog.h!=1080 or base_scheduler.recog.w!=1920:
        logger.error("模拟器分辨率不为1920x1080")
        return
    validation_msg = base_scheduler.initialize_operators()
    if validation_msg is not None:
        logger.error(validation_msg)
        return
    _loaded_operators = load_state()
    if _loaded_operators is not None:
        for k, v in _loaded_operators.items():
            if k in base_scheduler.op_data.operators and not base_scheduler.op_data.operators[k].room.startswith(
                    "dorm"):
                # 只复制心情数据
                base_scheduler.op_data.operators[k].mood = v.mood
                base_scheduler.op_data.operators[k].time_stamp = v.time_stamp
                base_scheduler.op_data.operators[k].depletion_rate = v.depletion_rate
                base_scheduler.op_data.operators[k].current_room = v.current_room
                base_scheduler.op_data.operators[k].current_index = v.current_index
    while True:
        try:
            if len(base_scheduler.tasks) > 0:
                (base_scheduler.tasks.sort(key=lambda x: x.time, reverse=False))
                sleep_time = (base_scheduler.tasks[0].time - datetime.now()).total_seconds()
                logger.info('||'.join([str(t) for t in base_scheduler.tasks]))
                base_scheduler.send_email(task_template.render(tasks=[obj.time_offset(timezone_offset) for obj in base_scheduler.tasks]), '', 'html')
                # 如果任务间隔时间超过9分钟则启动MAA
                if sleep_time > 540:
                    base_scheduler.maa_plan_solver()
                elif sleep_time > 0:
                    time.sleep(sleep_time)
            if len(base_scheduler.tasks) > 0 and base_scheduler.tasks[0].type.split('_')[0] == 'maa':
                base_scheduler.maa_plan_solver((base_scheduler.tasks[0].type.split('_')[1]).split(','), one_time=True)
                continue
            base_scheduler.run()
            reconnect_tries = 0
        except ConnectionError or ConnectionAbortedError or AttributeError as e:
            reconnect_tries += 1
            if reconnect_tries < reconnect_max_tries:
                logger.warning(f'连接端口断开....正在重连....')
                connected = False
                while not connected:
                    try:
                        base_scheduler = inialize([], base_scheduler)
                        break
                    except RuntimeError or ConnectionError or ConnectionAbortedError as ce:
                        logger.error(ce)
                        restart_simulator(simulator)
                        continue
                continue
            else:
                raise Exception(e)
        except RuntimeError as re:
            restart_simulator(simulator)
        except Exception as E:
            logger.exception(f"程序出错--->{E}")

    # cli.credit()  # 信用
    # ope_lists = cli.ope(eliminate=True, plan=ope_lists)  # 行动，返回未完成的作战计划
    # cli.shop(shop_priority)  # 商店
    # cli.recruit()  # 公招
    # cli.mission()  # 任务


```

这段代码是一个使用Python的第三方库`atexit`的功能。

`atexit`是一个Python库，用于在程序中注册自定义运维操作，例如打印日志、保存状态等。

具体来说，这段代码的作用如下：

1. `# debuglog()`是一个函数，用于打印调试信息。函数内部并没有做任何实际的逻辑，只是打印出了一个调试信息，例如：`print("debuglog")`，这只是告诉我们这个函数在将来会被打印出来。
2. `atexit.register(save_state)`是一个Python函数，用于注册一个自定义运维操作`save_state`。这个函数接受两个参数：`save_state`是一个需要保存的数据，函数内部使用`atexit`提供的`register`方法将`save_state`注册到运维操作链中，这样当程序退出时，所有使用`atexit`注册的运维操作都会自动保存，例如：`print("save_state")`。
3. `savelog()`是一个Python函数，用于打印日志。函数内部使用`atexit`提供的`register`方法将`print`函数注册到运维操作链中，这样当程序退出时，所有使用`print`函数输出的信息都会自动保存，例如：`print("save_log")`。
4. `simulate()`是一个Python函数，用于模拟程序的运行。这个函数内部可能调用了`atexit`注册的运维操作，例如打印调试信息、保存状态等，但是这个函数的具体实现不在我的知识范围内，我无法解释它的作用。


```py
# debuglog()
atexit.register(save_state)
savelog()
simulate()

```

# `main.py`

这段代码是一个Python脚本，它的作用是定义了一个名为`main`的函数，该函数是`arknights_mower`包的唯一实例。

具体来说，这段代码执行以下操作：

1. 导入`os`和`traceback`模块。
2. 从`arknights_mower`包中导入`__main__`函数。
3. 在`__main__`函数内执行以下操作：
   a. 调用`main`函数（含参数`module=False`，表示不导入`module`模块中的函数和类）。
   b. 尝试执行`main`函数。
   c. 创建一个`try`块，用于处理可能发生的异常。
   d. 在`try`块内，调用`main`函数，并传递一个参数`module=False`（表示不导入`module`模块中的函数和类）。
   e. 在`try`块外，创建一个`finally`块。
   f. 在`finally`块内，执行一个`if`语句，判断是否创建了一个`SystemExit`异常。如果是，跳过`finally`块内的内容，否则执行`finally`块内的内容。
   g. 在`if`语句外，执行`os.system('pause')`，即暂停操作系统并等待一段时间（此操作通常是用于在终端窗口中等待用户输入以便取消提示）。


```py
import os
import traceback

from arknights_mower.__main__ import main
from arknights_mower import __cli__

if __name__ == '__main__':
    try:
        main(module=False)
    except Exception:
        print(traceback.format_exc())
    except SystemExit:
        pass
    finally:
        if not __cli__:
            os.system('pause')

```

# `menu.py`

这段代码使用了多个Python库，包括Multiprocessing、PySimpleGUI和ruamel.yaml库。它主要用于处理和处理机器人生成器和任务的数据，并执行一些操作，如加载和保存配置文件、生成计划和更新版本号等。

具体来说，这段代码的作用可以分为以下几个方面：

1. 导入需要使用的Python库，包括json、Multiprocessing、PySimpleGUI、os和ruamel.yaml库。
2. 从名为"conf"的配置文件中读取配置信息，并将其存储在内存中的一个名为"config"的变量中。
3. 通过多线程或多进程的方式，从"config"变量中读取机器人生成器和任务的列表，并将它们存储在内存中的一个名为"agents"的列表中。
4. 通过调用write_plan函数，将生成的计划存储在"plan"文件中。
5. 创建一个名为"agent_list"的函数，用于打印当前系统中所有的 agent 对象。
6. 定义本游戏的版本，并在需要时进行更新。
7. 通过使用compere_version和download_version函数，从远程服务器下载更新版本的游戏。


```py
import json
from multiprocessing import Pipe, Process, freeze_support
import time
import PySimpleGUI as sg
import os
from ruamel.yaml import YAML

from arknights_mower.utils.conf import load_conf, save_conf, load_plan, write_plan
from arknights_mower.utils.log import logger
from arknights_mower.data import agent_list
from arknights_mower.__main__ import main
from arknights_mower.__init__ import __version__
from arknights_mower.utils.update import compere_version , update_version,download_version

yaml = YAML()
```

加载计划
========

def load_plan(url):
   global plans
   try:
       plans = json.loads(url)
       for plan in plans:
           if plan['default'] == plan:
               return plan
           else:
               plans[plan['default']] = plan
       return plans
   except Exception as e:
       logger.error(e)
       println('json解析错误！')
       return None



获取当前计划
========

def get_current_plan(plans):
   return plans[plan['default']]



查看所有建筑物
========

def view_buildings(plans):
   window = Listbox(plans['name'])
   for building in plans['buildings']:
       window.append_column(building)
   window.print_column_headers = True
   window.print_grid_lines = True



查看所有计划
========

def view_plans(plans):
   window = TkVersion
   window.title(plans['name'])
   window.geometry('500x500')
   window.update_idletasks()
   window.print_idletasks_with_time = True
   window.print_functions_怀旧 = True
   window.run_default_widget_mode = True
   window.set_user_point('Simulate')
   window.print_dimensions_滑动所有经区.sort_open = True
   window.set_idle_color('black')
   window.set_idle_color_fps_values([1], 'white')
   window.set_idle_keyboard_监视器.pub_join.set_state(True)
   window.set_idle_keyboard_监视器.pub_separator.set_state(True)
   window.set_idle_fps.put('当前计划', 'black')
   window.print_header('TITLE', 'center top', 'black', 'white')
   window.print_idletasks.background_color('white')
   window.print_idletasks.link_command.set_state(True)
   window.print_idletasks.link_command.disable_advanced.set_state(True)
   window.print_idletasks.link_command.can_link_from_title.set_state(True)
   window.print_idletasks.link_command.can_link_to_title.set_state(True)
   window.print_idletasks.link_command.execute.set_state(True)
   window.print_idletasks.link_command.status_pressed.set_state(True)
   window.print_idletasks.link_command.current_pressed.set_state(True)
   window.print_idletasks.link_command.destroy.set_state(True)
   window.print_idletasks.link_command.window_with_help.set_state(True)
   window.print_idletasks.link_command.warn_name.set_state(True)
   window.print_idletasks.link_command.warn_level.set_state(True)
   window.print_idletasks.link_command.corner_1.set_state(True)
   window.print_idletasks.link_command.corner_3.set_state(True)
   window.print_idletasks.link_command.corner_4.set_state(True)
   window.print_idletasks.link_command.corner_1.set_state(True)
   window.print_idletasks.link_command.corner_3.set_state(True)
   window.print_idletasks.link_command.corner_4.set_state(True)
   window.print_idletasks.link_command.keyboard_print.set_state(True)
   window.print_idletasks.link_command.status_pressed.set_state(True)
   window.print_idletasks.link_command.current_pressed.set_state(True)
   window.print_idletasks.link_command.destroy.set_state(True)
   window.print_idletasks.link_command.window_with_help.set_state(True)
   window.print_idletasks.print_functions_all_graphics.set_state(True)
   window.print_idletasks.print_print_dialogue.set_state(True)
   window.print_idletasks.print_print_status_pressed.set_state(True)
   window.print_idletasks.print_print_status.set_state(True)
   window.print_idletasks.print_print_dialogue.set_state(True)
   window.print_idletasks.print_print_print.set_state(True)
   window.print_idletasks.print_print_print.set_state(True)
   window.print_idletasks.print_print_print.set_state(True)
   window.print_idletasks.print_print_print.set_state(True)
   window.print_idletasks.print_print_print.set_state(True)
   window.print_idletasks.print_print_print.set_state(True)
   window.print_idletasks.print_print_print.set_state(True)
   window.print_idletasks.print_print_print.set_state(True)
   window.print_idletasks.print_print_print.set_state(True)
   window.print_idletasks.print_print_Print.set_state(True)
   window.print_idletasks.print_print_status.set_state(True)
   window.print_idletasks.print_print_print.set_state(True)
   window.print_idletasks.print_print_print.set_state(True)
   window.print_idletasks.print_print_Print.set_state(True)
   window.print_idletasks.print_print_print.set_state(True)
   window.print_idletasks.print_print_print.set_state(True)
   window.print_idletasks.print_print_Print.set_state(True)
   window.print_id


```py
# confUrl = './conf.yml'
conf = {}
plan = {}
current_plan = {}
operators = {}
global window
buffer = ''
line = 0
half_line_index = 0


def build_plan(url):
    global plan
    global current_plan
    try:
        plan = load_plan(url)
        current_plan = plan[plan['default']]
        conf['planFile'] = url
        for i in range(1, 4):
            for j in range(1, 4):
                window[f'btn_room_{str(i)}_{str(j)}'].update('待建造', button_color=('white', '#4f4945'))
        for key in current_plan:
            if type(current_plan[key]).__name__ == 'list':  # 兼容旧版格式
                current_plan[key] = {'plans': current_plan[key], 'name': ''}
            elif current_plan[key]['name'] == '贸易站':
                window['btn_' + key].update('贸易站', button_color=('#4f4945', '#33ccff'))
            elif current_plan[key]['name'] == '制造站':
                window['btn_' + key].update('制造站', button_color=('#4f4945', '#ffcc00'))
            elif current_plan[key]['name'] == '发电站':
                window['btn_' + key].update('发电站', button_color=('#4f4945', '#ccff66'))
        window['plan_radio_ling_xi_' + str(plan['conf']['ling_xi'])].update(True)
        window['plan_int_max_resting_count'].update(plan['conf']['max_resting_count'])
        window['plan_conf_exhaust_require'].update(plan['conf']['exhaust_require'])
        window['plan_conf_workaholic'].update(plan['conf']['workaholic'])
        window['plan_conf_rest_in_full'].update(plan['conf']['rest_in_full'])
        window['plan_conf_resting_priority'].update(plan['conf']['resting_priority'])
    except Exception as e:
        logger.error(e)
        println('json格式错误！')


```

这是一段Python文字游戏脚本，文字游戏大致玩法是玩家通过选择不同选项，操控游戏中的不同角色，影响游戏进程并获取胜利。

这段文字游戏脚本的具体玩法，还需要您根据实际游戏环境进行调整。


```py
# 主页面
def menu():
    global window
    global buffer
    global conf
    global plan
    conf = load_conf()
    plan = load_plan(conf['planFile'])
    sg.theme('LightBlue2')
    # --------主页
    package_type_title = sg.Text('服务器:', size=10)
    package_type_1 = sg.Radio('官服', 'package_type', default=conf['package_type'] == 1,
                              key='radio_package_type_1', enable_events=True)
    package_type_2 = sg.Radio('BiliBili服', 'package_type', default=conf['package_type'] == 2,
                              key='radio_package_type_2', enable_events=True)
    adb_title = sg.Text('adb连接地址:', size=10)
    adb = sg.InputText(conf['adb'], size=60, key='conf_adb', enable_events=True)
    # 黑名单
    free_blacklist_title = sg.Text('宿舍黑名单:', size=10)
    free_blacklist = sg.InputText(conf['free_blacklist'], size=60, key='conf_free_blacklist', enable_events=True)
    # 排班表json
    plan_title = sg.Text('排班表:', size=10)
    plan_file = sg.InputText(conf['planFile'], readonly=True, size=60, key='planFile', enable_events=True)
    plan_select = sg.FileBrowse('...', size=(3, 1), file_types=(("JSON files", "*.json"),))
    # 总开关
    on_btn = sg.Button('开始执行', key='on')
    off_btn = sg.Button('立即停止', key='off', visible=False, button_color='red')
    # 日志栏
    output = sg.Output(size=(150, 25), key='log', text_color='#808069', font=('微软雅黑', 9))

    # --------排班表设置页面
    # 宿舍区
    central = sg.Button('控制中枢', key='btn_central', size=(18, 3), button_color='#303030')
    dormitory_1 = sg.Button('宿舍', key='btn_dormitory_1', size=(18, 2), button_color='#303030')
    dormitory_2 = sg.Button('宿舍', key='btn_dormitory_2', size=(18, 2), button_color='#303030')
    dormitory_3 = sg.Button('宿舍', key='btn_dormitory_3', size=(18, 2), button_color='#303030')
    dormitory_4 = sg.Button('宿舍', key='btn_dormitory_4', size=(18, 2), button_color='#303030')
    central_area = sg.Column([[central], [dormitory_1], [dormitory_2], [dormitory_3], [dormitory_4]])
    # 制造站区
    room_1_1 = sg.Button('待建造', key='btn_room_1_1', size=(12, 2), button_color='#4f4945')
    room_1_2 = sg.Button('待建造', key='btn_room_1_2', size=(12, 2), button_color='#4f4945')
    room_1_3 = sg.Button('待建造', key='btn_room_1_3', size=(12, 2), button_color='#4f4945')
    room_2_1 = sg.Button('待建造', key='btn_room_2_1', size=(12, 2), button_color='#4f4945')
    room_2_2 = sg.Button('待建造', key='btn_room_2_2', size=(12, 2), button_color='#4f4945')
    room_2_3 = sg.Button('待建造', key='btn_room_2_3', size=(12, 2), button_color='#4f4945')
    room_3_1 = sg.Button('待建造', key='btn_room_3_1', size=(12, 2), button_color='#4f4945')
    room_3_2 = sg.Button('待建造', key='btn_room_3_2', size=(12, 2), button_color='#4f4945')
    room_3_3 = sg.Button('待建造', key='btn_room_3_3', size=(12, 2), button_color='#4f4945')
    left_area = sg.Column([[room_1_1, room_1_2, room_1_3],
                           [room_2_1, room_2_2, room_2_3],
                           [room_3_1, room_3_2, room_3_3]])
    # 功能区
    meeting = sg.Button('会客室', key='btn_meeting', size=(24, 2), button_color='#303030')
    factory = sg.Button('加工站', key='btn_factory', size=(24, 2), button_color='#303030')
    contact = sg.Button('办公室', key='btn_contact', size=(24, 2), button_color='#303030')
    right_area = sg.Column([[meeting], [factory], [contact]])

    setting_layout = [
        [sg.Column([[sg.Text('设施类别:'), sg.InputCombo(['贸易站', '制造站', '发电站'], size=12, key='station_type')]],
                   key='station_type_col', visible=False)]]
    # 排班表设置标签
    for i in range(1, 6):
        set_area = sg.Column([[sg.Text('干员：'),
                               sg.Combo(['Free'] + agent_list, size=20, key='agent' + str(i), change_submits=True),
                               sg.Text('组：'),
                               sg.InputText('', size=15, key='group' + str(i)),
                               sg.Text('替换：'),
                               sg.InputText('', size=30, key='replacement' + str(i))
                               ]], key='setArea' + str(i), visible=False)
        setting_layout.append([set_area])
    setting_layout.append(
        [sg.Button('保存', key='savePlan', visible=False), sg.Button('清空', key='clearPlan', visible=False)])
    setting_area = sg.Column(setting_layout, element_justification="center",
                             vertical_alignment="bottom",
                             expand_x=True)

    # --------高级设置页面
    current_version = sg.Text('当前版本：' + __version__, size=25)
    btn_check_update = sg.Button('检测更新', key='check_update')
    update_msg = sg.Text('', key='update_msg')
    run_mode_title = sg.Text('运行模式：', size=25)
    run_mode_1 = sg.Radio('换班模式', 'run_mode', default=conf['run_mode'] == 1,
                          key='radio_run_mode_1', enable_events=True)
    run_mode_2 = sg.Radio('仅跑单模式', 'run_mode', default=conf['run_mode'] == 2,
                          key='radio_run_mode_2', enable_events=True)
    ling_xi_title = sg.Text('令夕模式（令夕上班时起作用）：', size=25)
    ling_xi_1 = sg.Radio('感知信息', 'ling_xi', default=plan['conf']['ling_xi'] == 1,
                         key='plan_radio_ling_xi_1', enable_events=True)
    ling_xi_2 = sg.Radio('人间烟火', 'ling_xi', default=plan['conf']['ling_xi'] == 2,
                         key='plan_radio_ling_xi_2', enable_events=True)
    ling_xi_3 = sg.Radio('均衡模式', 'ling_xi', default=plan['conf']['ling_xi'] == 3,
                         key='plan_radio_ling_xi_3', enable_events=True)
    enable_party_title = sg.Text('线索收集：', size=25)
    enable_party_1 = sg.Radio('启用', 'enable_party', default=conf['enable_party'] == 1,
                              key='radio_enable_party_1', enable_events=True)
    enable_party_0 = sg.Radio('禁用', 'enable_party', default=conf['enable_party'] == 0,
                              key='radio_enable_party_0', enable_events=True)
    max_resting_count_title = sg.Text('最大组人数：', size=25, key='max_resting_count_title')
    max_resting_count = sg.InputText(plan['conf']['max_resting_count'], size=5,
                                     key='plan_int_max_resting_count', enable_events=True)
    drone_count_limit_title = sg.Text('无人机使用阈值：', size=25, key='drone_count_limit_title')
    drone_count_limit = sg.InputText(conf['drone_count_limit'], size=5,
                                     key='int_drone_count_limit', enable_events=True)
    run_order_delay_title = sg.Text('跑单前置延时(分钟)：', size=25, key='run_order_delay_title')
    run_order_delay = sg.InputText(conf['run_order_delay'], size=5,
                                   key='float_run_order_delay', enable_events=True)
    drone_room_title = sg.Text('无人机使用房间（room_X_X）：', size=25, key='drone_room_title')
    reload_room_title = sg.Text('搓玉补货房间（逗号分隔房间名）：', size=25, key='reload_room_title')
    drone_room = sg.InputText(conf['drone_room'], size=15,
                              key='conf_drone_room', enable_events=True)
    reload_room = sg.InputText(conf['reload_room'], size=30,
                               key='conf_reload_room', enable_events=True)
    rest_in_full_title = sg.Text('需要回满心情的干员：', size=25)
    rest_in_full = sg.InputText(plan['conf']['rest_in_full'], size=60,
                                key='plan_conf_rest_in_full', enable_events=True)

    exhaust_require_title = sg.Text('需用尽心情的干员：', size=25)
    exhaust_require = sg.InputText(plan['conf']['exhaust_require'], size=60,
                                   key='plan_conf_exhaust_require', enable_events=True)

    workaholic_title = sg.Text('0心情工作的干员：', size=25)
    workaholic = sg.InputText(plan['conf']['workaholic'], size=60,
                              key='plan_conf_workaholic', enable_events=True)

    resting_priority_title = sg.Text('宿舍低优先级干员：', size=25)
    resting_priority = sg.InputText(plan['conf']['resting_priority'], size=60,
                                    key='plan_conf_resting_priority', enable_events=True)

    start_automatically = sg.Checkbox('启动mower时自动开始任务', default=conf['start_automatically'],
                                      key='conf_start_automatically', enable_events=True)
    # --------外部调用设置页面
    # mail
    mail_enable_1 = sg.Radio('启用', 'mail_enable', default=conf['mail_enable'] == 1,
                             key='radio_mail_enable_1', enable_events=True)
    mail_enable_0 = sg.Radio('禁用', 'mail_enable', default=conf['mail_enable'] == 0,
                             key='radio_mail_enable_0', enable_events=True)
    account_title = sg.Text('QQ邮箱', size=25)
    account = sg.InputText(conf['account'], size=60, key='conf_account', enable_events=True)
    pass_code_title = sg.Text('授权码', size=25)
    pass_code = sg.Input(conf['pass_code'], size=60, key='conf_pass_code', enable_events=True, password_char='*')
    mail_frame = sg.Frame('邮件提醒',
                          [[mail_enable_1, mail_enable_0], [account_title, account], [pass_code_title, pass_code]])
    # maa

    maa_enable_1 = sg.Radio('启用', 'maa_enable', default=conf['maa_enable'] == 1,
                            key='radio_maa_enable_1', enable_events=True)
    maa_enable_0 = sg.Radio('禁用', 'maa_enable', default=conf['maa_enable'] == 0,
                            key='radio_maa_enable_0', enable_events=True)
    maa_gap_title = sg.Text('MAA启动间隔(小时)：', size=15)
    maa_gap = sg.InputText(conf['maa_gap'], size=5, key='float_maa_gap', enable_events=True)
    maa_mall_buy_title = sg.Text('信用商店优先购买（逗号分隔）：', size=25, key='mall_buy_title')
    maa_mall_buy = sg.InputText(conf['maa_mall_buy'], size=30,
                               key='conf_maa_mall_buy', enable_events=True)
    maa_recruitment_time = sg.Checkbox('公招三星设置7:40而非9:00', default=conf['maa_recruitment_time'],
                                      key='conf_maa_recruitment_time', enable_events=True)
    maa_recruit_only_4 = sg.Checkbox('仅公招四星', default=conf['maa_recruit_only_4'],
                                       key='conf_maa_recruit_only_4', enable_events=True)
    maa_mall_blacklist_title = sg.Text('信用商店黑名单（逗号分隔）：', size=25, key='mall_blacklist_title')
    maa_mall_blacklist = sg.InputText(conf['maa_mall_blacklist'], size=30,
                                key='conf_maa_mall_blacklist', enable_events=True)
    maa_rg_title = sg.Text('肉鸽：', size=10)
    maa_rg_enable_1 = sg.Radio('启用', 'maa_rg_enable', default=conf['maa_rg_enable'] == 1,
                               key='radio_maa_rg_enable_1', enable_events=True)
    maa_rg_enable_0 = sg.Radio('禁用', 'maa_rg_enable', default=conf['maa_rg_enable'] == 0,
                               key='radio_maa_rg_enable_0', enable_events=True)
    maa_rg_sleep = sg.Text('肉鸽任务休眠时间(如8:30-23:30)', size=25)
    maa_rg_sleep_min = sg.InputText(conf['maa_rg_sleep_min'], size=5, key='conf_maa_rg_sleep_min', enable_events=True)
    maa_rg_sleep_max = sg.InputText(conf['maa_rg_sleep_max'], size=5, key='conf_maa_rg_sleep_max', enable_events=True)
    maa_path_title = sg.Text('MAA地址', size=25)
    maa_path = sg.InputText(conf['maa_path'], size=60, key='conf_maa_path', enable_events=True)
    maa_adb_path_title = sg.Text('adb地址', size=25)
    maa_adb_path = sg.InputText(conf['maa_adb_path'], size=60, key='conf_maa_adb_path', enable_events=True)
    maa_weekly_plan_title = sg.Text('周计划', size=25)
    maa_layout = [[maa_enable_1, maa_enable_0, maa_gap_title, maa_gap, maa_recruitment_time, maa_recruit_only_4],
                  [maa_mall_buy_title, maa_mall_buy, maa_mall_blacklist_title, maa_mall_blacklist],
                  [maa_rg_title, maa_rg_enable_1, maa_rg_enable_0, maa_rg_sleep, maa_rg_sleep_min, maa_rg_sleep_max],
                  [maa_path_title, maa_path], [maa_adb_path_title, maa_adb_path],
                  [maa_weekly_plan_title]]
    for i, v in enumerate(conf['maa_weekly_plan']):
        maa_layout.append([
            sg.Text(f"-- {v['weekday']}:", size=15),
            sg.Text('关卡:', size=5),
            sg.InputText(",".join(v['stage']), size=15, key='maa_weekly_plan_stage_' + str(i), enable_events=True),
            sg.Text('理智药:', size=10),
            sg.Spin([l for l in range(0, 999)], initial_value=v['medicine'], size=5,
                    key='maa_weekly_plan_medicine_' + str(i), enable_events=True, readonly=True)
        ])

    maa_frame = sg.Frame('MAA', maa_layout)
    # --------组装页面
    main_tab = sg.Tab('  主页  ', [[package_type_title, package_type_1, package_type_2],
                                 [adb_title, adb],
                                 [free_blacklist_title, free_blacklist],
                                 [plan_title, plan_file, plan_select],
                                 [output],
                                 [on_btn, off_btn]])

    plan_tab = sg.Tab('  排班表 ', [[left_area, central_area, right_area], [setting_area]], element_justification="center")
    setting_tab = sg.Tab('  高级设置 ',
                         [
                             [current_version, btn_check_update, update_msg],
                             [run_mode_title, run_mode_1, run_mode_2], [ling_xi_title, ling_xi_1, ling_xi_2, ling_xi_3],
                             [enable_party_title, enable_party_1, enable_party_0],
                             [max_resting_count_title, max_resting_count, sg.Text('', size=16), run_order_delay_title,
                              run_order_delay],
                             [drone_room_title, drone_room, sg.Text('', size=7), drone_count_limit_title,
                              drone_count_limit],
                             [reload_room_title, reload_room],
                             [rest_in_full_title, rest_in_full],
                             [exhaust_require_title, exhaust_require],
                             [workaholic_title, workaholic],
                             [resting_priority_title, resting_priority],
                             [start_automatically],
                         ], pad=((10, 10), (10, 10)))

    other_tab = sg.Tab('  外部调用 ',
                       [[mail_frame], [maa_frame]], pad=((10, 10), (10, 10)))
    window = sg.Window('Mower', [[sg.TabGroup([[main_tab, plan_tab, setting_tab, other_tab]], border_width=0,
                                              tab_border_width=0, focus_color='#bcc8e5',
                                              selected_background_color='#d4dae8', background_color='#aab6d3',
                                              tab_background_color='#aab6d3')]], font='微软雅黑', finalize=True,
                       resizable=False)
    build_plan(conf['planFile'])
    btn = None
    bind_scirpt()  # 为基建布局左边的站点排序绑定事件
    drag_task = DragTask()
    if conf['start_automatically']:
        start()
    while True:
        event, value = window.Read()
        if event == sg.WIN_CLOSED:
            break
        if event.endswith('-script'):  # 触发事件，进行处理
            run_script(event[:event.rindex('-')], drag_task)
            continue
        drag_task.clear()  # 拖拽事件连续不间断，若未触发事件，则初始化

        if event.startswith('plan_conf_'):  # plan_conf开头，为字符串输入的排班表配置
            key = event[10:]
            plan['conf'][key] = window[event].get().strip()
        elif event.startswith('plan_int_'):  # plan_int开头，为数值型输入的配置
            key = event[9:]
            try:
                plan['conf'][key] = int(window[event].get().strip())
            except ValueError:
                println(f'[{window[key + "_title"].get()}]需为数字')
        elif event.startswith('plan_radio_'):
            v_index = event.rindex('_')
            plan['conf'][event[11:v_index]] = int(event[v_index + 1:])
        elif event.startswith('conf_'):  # conf开头，为字符串输入的配置
            key = event[5:]
            value = window[event].get()
            conf[key] = value.strip() if isinstance(value, str) else value
        elif event.startswith('int_'):  # int开头，为数值型输入的配置
            key = event[4:]
            try:
                conf[key] = int(window[event].get().strip())
            except ValueError:
                println(f'[{window[key + "_title"].get()}]需为数字')
        elif event.startswith('float_'):  # float开头，为数值型输入的配置
            key = event[6:]
            try:
                conf[key] = float(window[event].get().strip())
            except ValueError:
                println(f'[{window[key + "_title"].get()}]需为数字')
        elif event.startswith('radio_'):
            v_index = event.rindex('_')
            conf[event[6:v_index]] = int(event[v_index + 1:])
        elif event == 'planFile' and plan_file.get() != conf['planFile']:  # 排班表
            write_plan(plan, conf['planFile'])
            build_plan(plan_file.get())
            plan_file.update(conf['planFile'])
        elif event.startswith('maa_weekly_plan_stage_'):  # 关卡名
            v_index = event.rindex('_')
            conf['maa_weekly_plan'][int(event[v_index + 1:])]['stage'] = [window[event].get()]
        elif event.startswith('maa_weekly_plan_medicine_'):  # 体力药
            v_index = event.rindex('_')
            conf['maa_weekly_plan'][int(event[v_index + 1:])]['medicine'] = int(window[event].get())
        elif event.startswith('btn_'):  # 设施按钮
            btn = event
            init_btn(event)
        elif event.endswith('-agent_change'):  # 干员填写
            input_agent = window[event[:event.rindex('-')]].get().strip()
            window[event[:event.rindex('-')]].update(value=input_agent, values=list(
                filter(lambda s: input_agent in s, ['Free'] + agent_list)))
        elif event.startswith('agent'):
            input_agent = window[event].get().strip()
            window[event].update(value=input_agent,
                                 values=list(filter(lambda s: input_agent in s, ['Free'] + agent_list)))
        elif event == 'check_update':
            window['update_msg'].update('正在检测更新...', text_color='black')
            window[event].update(disabled=True)
            window.perform_long_operation(check_update, 'check_update_value')
        elif event == 'check_update_value':
            if value[event] :
                b=sg.popup_yes_no("下载成功！是否立刻更新？")
                if b == 'Yes':
                    update_version()

        elif event == 'savePlan':  # 保存设施信息
            save_btn(btn)
        elif event == 'clearPlan':  # 清空当前设施信息
            clear_btn(btn)
        elif event == 'on':
            start()
        elif event == 'off':
            println('停止运行')
            child_conn.close()
            main_thread.terminate()
            on_btn.update(visible=True)
            off_btn.update(visible=False)

    window.close()
    save_conf(conf)
    write_plan(plan, conf['planFile'])


```

这段代码是一个Python脚本，它的作用是开始一个下棋游戏的客户端。它通过Pygame库和其他库来实现图形化用户界面和与服务器通信。

具体来说，这段代码实现了一个简单的命令行界面，用户可以通过点击“开始”按钮开始游戏。在游戏开始时，它会创建一个父进程和一个子进程，分别用于处理游戏逻辑和用户界面。父进程负责将游戏逻辑抽象为一些事件，子进程负责将这些事件与游戏逻辑进行绑定，从而实现游戏界面的交互效果。

此外，这段代码还实现了一个键盘绑定脚本，用于将用户的键盘输入与游戏逻辑进行绑定。这个脚本可以接受用户的输入，并将用户的输入与游戏逻辑进行交互，从而实现游戏的响应。


```py
def start():
    global main_thread, child_conn
    window['on'].update(visible=False)
    window['off'].update(visible=True)
    clear()
    parent_conn, child_conn = Pipe()
    main_thread = Process(target=main, args=(conf, plan, operators, child_conn), daemon=True)
    main_thread.start()
    window.perform_long_operation(lambda: recv(parent_conn), 'recv')


def bind_scirpt():
    for i in range(3):
        for j in range(3):
            event = f'btn_room_{str(i + 1)}_{str(j + 1)}'
            window[event].bind("<B1-Motion>", "-motion-script")
            window[event].bind("<ButtonRelease-1>", "-ButtonRelease-script")
            window[event].bind("<Enter>", "-Enter-script")
    for i in range(5):
        event = f'agent{str(i + 1)}'
        window[event].bind("<Key>", "-agent_change")


```

这段代码是一个Python函数，名为`run_script`，它接受两个参数：`event`和`drag_task`。

首先，函数内部先打印一条日志信息，然后判断`event`是否以`-motion`结尾，如果是，就执行以下操作：

1. 如果`drag_task`的`step`字段等于0或者2，说明拖拽结束未进入其他元素，函数将初始化一些变量，如`drag_task.btn`，`drag_task.step`。

2. 如果`event`以`-ButtonRelease`结尾，标志着拖拽结束，函数将根据拖拽结束的步骤，推进`drag_task`的步骤。

3. 如果`event`以`-Enter`结尾，进入元素事件，函数将根据拖拽结束的步骤，进入其他元素。

4. 如果`drag_task`的`step`等于2，标记需要交换的按钮，然后调用一个名为`switch_plan`的函数，将进入的步骤与离开的步骤进行切换，并清除之前创建的按钮。

5. 如果`drag_task`的`step`不等于2，直接清除之前创建的按钮。


```py
def run_script(event, drag_task):
    # logger.info(f"{event}:{drag_task}")
    if event.endswith('-motion'):  # 拖拽事件，标志拖拽开始
        if drag_task.step == 0 or drag_task.step == 2:  # 若为2说明拖拽结束未进入其他元素，则初始化
            drag_task.btn = event[:event.rindex('-')]  # 记录初始按钮
            drag_task.step = 1  # 初始化键位，并推进任务步骤
    elif event.endswith('-ButtonRelease'):  # 松开按钮事件，标志着拖拽结束
        if drag_task.step == 1:
            drag_task.step = 2  # 推进任务步骤
    elif event.endswith('-Enter'):  # 进入元素事件，拖拽结束鼠标若在其他元素，会进入此事件
        if drag_task.step == 2:
            drag_task.new_btn = event[:event.rindex('-')]  # 记录需交换的按钮
            switch_plan(drag_task)
            drag_task.clear()
        else:
            drag_task.clear()


```

这段代码是一个名为`switch_plan`的函数，其作用是根据用户的选择，将一个名为`drag_task`的任务根据当前计划中的某些按钮，生成新的计划并将其写入。

具体来说，代码首先获取用户选择的一个按钮，并检查该按钮是否在当前计划中。如果是，代码从当前计划中查找该按钮的值，并将它存储在变量`value1`中。如果不是，代码将不会执行任何操作。

然后，代码根据`value1`的值，进一步检查当前计划中是否包含该按钮。如果是，代码将该按钮的值存储在当前计划中对应的位置，并返回该位置。如果不是，代码将该按钮及其之后的元素删除，并继续执行下一次操作。

接下来，代码将`value2`的值存储在当前计划中对应的位置，并检查该按钮是否在当前计划中。如果是，代码将该按钮的值存储在当前计划中对应的位置，并继续执行下一次操作。如果不是，代码将该按钮及其之后的元素删除，并返回当前计划中所有按钮值的最新更改的位置。

最后，代码调用两个函数`write_plan`和`build_plan`，将用户选择的计划和当前计划写入文件中，以便将新的计划应用到任务中。


```py
def switch_plan(drag_task):
    key1 = drag_task.btn[4:]
    key2 = drag_task.new_btn[4:]
    value1 = current_plan[key1] if key1 in current_plan else None;
    value2 = current_plan[key2] if key2 in current_plan else None;
    if value1 is not None:
        current_plan[key2] = value1
    elif key2 in current_plan:
        current_plan.pop(key2)
    if value2 is not None:
        current_plan[key1] = value2
    elif key1 in current_plan:
        current_plan.pop(key1)
    write_plan(plan, conf['planFile'])
    build_plan(conf['planFile'])


```

这段代码是一个Python定义函数，名为`init_btn`。该函数的作用是在一个GUI窗口中，根据用户点击的按钮事件（事件名为`event`），对当前计划中的一个房间（通过房间的索引）进行更新。以下是该函数的详细解释：

1. `def init_btn(event):`：定义了函数，初始化了函数的参数。

2. `room_key = event[4:]`：从事件参数（第四个元素，即下划线之后）获取房间的索引。

3. `station_name = current_plan[room_key]['name']`：如果房间索引`room_key`在当前计划中，返回房间对应的可视化名称（通过`current_plan`字典中的键值）。

4. `plans = current_plan[room_key]['plans']`：如果房间索引`room_key`在当前计划中，返回房间对应的计划列表（通过`current_plan`字典中的键值）。

5. `if room_key.startswith('room'):`：如果是`'room'`开始，则执行以下操作：

  6. `window['station_type_col'].update(visible=True)`：设置设施干员需求数量为当前设施数量，并且设施图标的可见性为`True`。

  7. `window['station_type'].update('Room ' + station_name)`：设置设施图标的文本为房间名称（通过`current_plan`字典中的键值）。

  8. `visible_cnt = 3`：设置设施干员需求数量为3。

  9. `elif room_key == 'meeting'`：如果点击的是会议按钮，则执行以下操作：

     10. `visible_cnt = 2`：设置设施干员需求数量为2。

     11. `elif room_key == 'factory'`：如果点击的是工厂按钮，则执行以下操作：

       12. `visible_cnt = 1`：设置设施干员需求数量为1。

       13. `elif room_key == 'contact'`：如果点击的是联系按钮，则执行以下操作：

         14. `visible_cnt = 5`：设置设施干员需求数量为5。

  15. `else`：如果不是上面四种情况，则执行以下操作：

     16. `window['station_type_col'].update(visible=False)`：设置设施图标的可见性为`False`。

     17. `window['station_type'].update('')`：清除设施图标的文本。

     18. `window['savePlan'].update(visible=True)`：设置保存计划的可视化图标为`True`。

     19. `window['clearPlan'].update(visible=True)`：设置清除计划的可视化图标为`True`。

     20. `for i in range(1, 6):`：从1到5遍历设施干员需求数量（根据点击的按钮编号，设施干员需求数量为1到5）。

     21. `if i > visible_cnt:`：如果当前设施数量大于设施干员需求数量，执行以下操作：

       22. `window['setArea' + str(i)].update(visible=False)`：设置设施的可见性为`False`。

       23. `window['agent' + str(i)].update('')`：清除设施干员需求的文本描述。

       24. `window['group' + str(i)].update('')`：清除设施干员需求的组队描述。

       25. `window['replacement' + str(i)].update('')`：清除设施干员需求的替换描述。

     26. `else:`：否则，执行以下操作：

       27. `window['setArea' + str(i)].update(visible=True)`：设置设施的可见性为`True`。

       28. `window['agent' + str(i)].update(plans[i - 1]['agent'])`：设置设施干员需求的文本描述。

       29. `window['group' + str(i)].update(plans[i - 1]['group'])`：设置设施干员需求的组队描述。

       30. `window['replacement' + str(i)].update(', '.join(plans[i - 1]['replacement']))`：设置设施干员需求的替换描述。


```py
def init_btn(event):
    room_key = event[4:]
    station_name = current_plan[room_key]['name'] if room_key in current_plan.keys() else ''
    plans = current_plan[room_key]['plans'] if room_key in current_plan.keys() else []
    if room_key.startswith('room'):
        window['station_type_col'].update(visible=True)
        window['station_type'].update(station_name)
        visible_cnt = 3  # 设施干员需求数量
    else:
        if room_key == 'meeting':
            visible_cnt = 2
        elif room_key == 'factory' or room_key == 'contact':
            visible_cnt = 1
        else:
            visible_cnt = 5
        window['station_type_col'].update(visible=False)
        window['station_type'].update('')
    window['savePlan'].update(visible=True)
    window['clearPlan'].update(visible=True)
    for i in range(1, 6):
        if i > visible_cnt:
            window['setArea' + str(i)].update(visible=False)
            window['agent' + str(i)].update('')
            window['group' + str(i)].update('')
            window['replacement' + str(i)].update('')
        else:
            window['setArea' + str(i)].update(visible=True)
            window['agent' + str(i)].update(plans[i - 1]['agent'] if len(plans) >= i else '')
            window['group' + str(i)].update(plans[i - 1]['group'] if len(plans) >= i else '')
            window['replacement' + str(i)].update(','.join(plans[i - 1]['replacement']) if len(plans) >= i else '')


```

这段代码定义了一个名为 `save_btn` 的函数，用于在需要时保存指定的计划。

该函数接受一个名为 `btn` 的参数，用于指定保存的计划的类型。函数内部首先定义了一个名为 `plan1` 的字典，用于存储该计划的所有相关信息。

接着，函数使用一个循环来遍历该计划中所有相关的子计划。对于每个子计划，函数首先从 `window` 字典中获取与该子计划相关的组件(如 'agent' 和 'group')和一个用于存储子计划中所有替代人选的列表的函数。

接下来，函数使用 `filter` 函数和列表推导式来过滤掉所有与指定代理人或子计划不匹配的元素。如果子计划的代理不为空，函数将该代理添加到 `plan1` 计划的相应部分中。如果子计划的类型是 "btn_dormitory"，函数将 'Free' 添加到 `plan1` 计划的相应部分中。

接下来，函数使用 `current_plan` 字典将计划保存到指定的文件中。最后，函数调用 `write_plan` 和 `build_plan` 函数来将计划保存到实际文件中。


```py
def save_btn(btn):
    plan1 = {'name': window['station_type'].get(), 'plans': []}
    for i in range(1, 6):
        agent = window['agent' + str(i)].get()
        group = window['group' + str(i)].get()
        replacement = list(filter(None, window['replacement' + str(i)].get().replace('，', ',').split(',')))
        if agent != '':
            plan1['plans'].append({'agent': agent, 'group': group, 'replacement': replacement})
        elif btn.startswith('btn_dormitory'):  # 宿舍
            plan1['plans'].append({'agent': 'Free', 'group': '', 'replacement': []})
    current_plan[btn[4:]] = plan1
    write_plan(plan, conf['planFile'])
    build_plan(conf['planFile'])


```

这段代码是一个 Python 函数，名为 `clear_btn`，它有以下几个主要作用：

1. 检查当前按钮是否在当前计划中。如果是，则从当前计划中移除该按钮。
2. 初始化按钮，设置按钮状态为不可点击。
3. 将当前计划写入 `plan` 变量，并调用 `write_plan` 函数。
4. 使用 `build_plan` 函数构建计划，并将其保存到 `plan` 变量中。

另外，还包含以下辅助函数：

1. `comparer_version()` 函数用于比较当前版本和最新版本，返回最新版本。
2. `check_update()` 函数用于检测是否有新版本更新，下载新版本并更新界面。
3. `download_version(version)` 函数用于下载指定版本的更新包。
4. `logger.error(e)` 函数用于记录并输出异常信息。


```py
def clear_btn(btn):
    if btn[4:] in current_plan:
        current_plan.pop(btn[4:])
    init_btn(btn)
    write_plan(plan, conf['planFile'])
    build_plan(conf['planFile'])


def check_update():
    try:
        newest_version = compere_version()
        if newest_version:
            window['update_msg'].update('检测到有新版本'+newest_version+',正在下载...',text_color='black')
            download_version(newest_version)
            window['update_msg'].update('下载完毕！',text_color='green')
        else:
            window['update_msg'].update('已是最新版！',text_color='green')
    except Exception as e:
        logger.error(e)
        window['update_msg'].update('更新失败！',text_color='red')
        return None
    window['check_update'].update(disabled=False)
    return newest_version


```

这段代码的作用是接收用户推送消息并输出消息内容。具体来说，代码分为以下几个部分：

1. `recv` 函数：该函数的作用是接收推送消息。它尝试从传入的 `pipe` 对象中读取消息，并处理不同消息类型。如果是日志类型（`type == 'log'`），函数会打印消息内容；如果是运算符类型（`type == 'operators'`），函数会创建一个名为 `operators` 的全局变量，并将消息内容赋给它。
2. `println` 函数：该函数的作用是打印推送消息。它会根据消息类型打印消息，并使用缓冲区（`buffer`）来避免频繁输出。函数会根据当前行数和最大行数来调整输出策略，以保证输出不会太长或太短。
3. `try`/`except` 块：该块用于处理可能出现的 `EOFError` 异常。如果读取到结束-of-file（EOF）错误，函数会关闭传入的 `pipe` 对象。


```py
# 接收推送
def recv(pipe):
    try:
        while True:
            msg = pipe.recv()
            if msg['type'] == 'log':
                println(msg['data'])
            elif msg['type'] == 'operators':
                global operators
                operators = msg['data']
    except EOFError:
        pipe.close()


def println(msg):
    global buffer
    global line
    global half_line_index
    maxLen = 500  # 最大行数
    buffer = f'{buffer}\n{time.strftime("%m-%d %H:%M:%S")} {msg}'.strip()
    window['log'].update(value=buffer)
    if line == maxLen // 2:
        half_line_index = len(buffer)
    if line >= maxLen:
        buffer = buffer[half_line_index:]
        line = maxLen // 2
    else:
        line += 1


```

这段代码定义了一个名为 `clear` 的函数，以及一个名为 `DragTask` 的类。

首先，定义了一个名为 `buffer` 的全局变量，用于存储输出栏中的内容。接着，定义了一个名为 `line` 的全局变量，用于跟踪输出栏中的行数。然后，将全局变量 `buffer` 赋值为空，将全局变量 `line` 赋值为 0。

接下来，定义了一个名为 `DragTask` 的类，该类包含了一个名为 `clear` 的方法和一个 `__repr__` 的成员函数。

在 `DragTask` 的 `clear` 方法中，重置了全局变量 `buffer` 和 `line`，并更新了输出栏中的值。然后，创建了一个新的 `DragTask` 实例，并将其设置为当前实例。

最后，定义了一个名为 `__repr__` 的成员函数，用于返回该对象的引用。


```py
# 清空输出栏
def clear():
    global buffer
    global line
    buffer = ''
    window['log'].update(value=buffer)
    line = 0


class DragTask:
    def __init__(self):
        self.btn = None
        self.new_btn = None
        self.step = 0

    def clear(self):
        self.btn = None
        self.new_btn = None
        self.step = 0

    def __repr__(self):
        return f"btn:{self.btn},new_btn:{self.new_btn},step:{self.step}"


```

这段代码是一个Python脚本，尽管它没有明确的名称和作者，但我们可以根据代码的功能来推测它的作用。

这段代码包含两个主要部分：

1. if 语句：这是一个条件判断语句，它会判断当前脚本是否作为主程序（__main__）运行。如果条件为真，那么执行if语句块内的内容。

2. freeze_support() 函数：这个函数可能是用来在程序运行时动态地加载和解冻Python模块定义的。freeze_support() 函数在Python 2.4中被引入，一直持续到Python 3.8。由于在这段代码中，函数名称与实际功能无关，我们可以忽略其具体实现，而关注函数调用的目的。

3. menu() 函数（可能性不大）：这个函数在没有定义具体内容的情况下，我们也可以忽略。从函数名称上看，menu() 可能是一个帮助用户设置选项菜单的函数，但实际功能取决于menu() 函数内包含的具体操作。

综上所述，这段代码的作用是：在程序作为主程序运行时，动态地加载和使用freeze_support()函数。虽然函数的作用还不确定，但我们可以根据freeze_support()函数的特性，推测它可能与模块的加载和卸载有关。


```py
if __name__ == '__main__':
    freeze_support()
    menu()

```