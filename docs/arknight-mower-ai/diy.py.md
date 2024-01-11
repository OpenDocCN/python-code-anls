# `arknights-mower\diy.py`

```
# 导入时间模块
import time
# 从 datetime 模块中导入 datetime 类
from datetime import datetime
# 导入 atexit 模块
import atexit
# 导入 json 模块
import json
# 导入 os 模块
import os

# 从 arknights_mower.solvers.base_schedule 模块中导入 BaseSchedulerSolver 类
from arknights_mower.solvers.base_schedule import BaseSchedulerSolver
# 从 arknights_mower.strategy 模块中导入 Solver 类
from arknights_mower.strategy import Solver
# 从 arknights_mower.utils.device 模块中导入 Device 类
from arknights_mower.utils.device import Device
# 从 arknights_mower.utils.email 模块中导入 task_template 函数
from arknights_mower.utils.email import task_template
# 从 arknights_mower.utils.log 模块中导入 logger, init_fhlr 函数
from arknights_mower.utils.log import logger, init_fhlr
# 从 arknights_mower.utils 模块中导入 config 模块
from arknights_mower.utils import config
# 从 arknights_mower.utils.simulator 模块中导入 restart_simulator 函数
from arknights_mower.utils.simulator import restart_simulator
# 下面不能删除
# 从 arknights_mower.utils.operators 模块中导入 Operators, Operator, Dormitory 类
from arknights_mower.utils.operators import Operators, Operator, Dormitory
# 从 arknights_mower.utils.scheduler_task 模块中导入 SchedulerTask 类
from arknights_mower.utils.scheduler_task import SchedulerTask

# 邮件配置信息
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
# MAA 配置信息
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
}
    # 定义了一个名为 "rogue" 的字典，包含了指挥分队、取长补短等信息
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
    # 定义了两个空字符串变量 "sleep_min" 和 "sleep_max"
    "sleep_min":"",
    "sleep_max":"",
    # 定义了一个名为 "weekly_plan" 的列表，包含了每周的计划信息
    "weekly_plan": [{"weekday": "周一", "stage": [''], "medicine": 0},
                    {"weekday": "周二", "stage": [''], "medicine": 0},
                    {"weekday": "周三", "stage": [''], "medicine": 0},
                    {"weekday": "周四", "stage": [''], "medicine": 0},
                    {"weekday": "周五", "stage": [''], "medicine": 0},
                    {"weekday": "周六", "stage": [''], "medicine": 0},
                    {"weekday": "周日", "stage": [''], "medicine": 0}]
# 模拟器相关设置
simulator= {
    "name":"夜神",  # 模拟器名称
    "index":2,  # 多开编号，在模拟器助手最左侧的数字
    "simulator_folder":"D:\\Program Files\\Nox\\bin"  # 用于执行模拟器命令的文件夹路径
}

# Free (宿舍填充)干员安排黑名单
free_blacklist= []  # 宿舍填充干员的黑名单列表

# 干员宿舍回复阈值
    # 高效组心情低于 UpperLimit  * 阈值 (向下取整)的时候才会会安排休息
    # UpperLimit：默认24，特殊技能干员如夕，令可能会有所不同(设置在 agent-base.json 文件可以自行更改)
resting_threshold = 0.5  # 宿舍回复心情的阈值

# 跑单如果all in 贸易站则 不需要修改设置
# 如果需要无人机加速其他房间则可以修改成房间名字如 'room_1_1'
drone_room = None  # 无人机执行的房间名
# 无人机执行间隔时间 （小时）
drone_execution_gap = 4  # 无人机执行的间隔时间（小时）

reload_room = []  # 重新加载的房间列表

# 基地数据json文件保存名
state_file_name = 'state.json'  # 基地数据json文件的保存名

# 邮件时差调整
timezone_offset = 0  # 邮件时差的调整值

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
plan = {
    # 阶段 1
    "default": "plan_1",  # 默认的排班计划
    }
}

# UpperLimit、LowerLimit：心情上下限
# ExhaustRequire：是否强制工作到红脸再休息
# ArrangeOrder：指定在宿舍外寻找干员的方式
# RestInFull：是否强制休息到24心情再工作，与ExhaustRequire一起帮助暖机类技能工作更长时间
# RestingPriority：休息优先级，低优先级不会使用单回技能。

agent_base_config = {
    "Default": {"UpperLimit": 24, "LowerLimit": 0, "ExhaustRequire": False, "ArrangeOrder": [2, "false"],
                "RestInFull": False,"Workaholic":False},  # 默认的干员基础配置
    "令": {"LowerLimit": 12,"ArrangeOrder": [2, "true"]},  # 令的干员配置
    "夕": {"UpperLimit": 12, "ArrangeOrder": [2, "true"]},  # 夕的干员配置
    "稀音": {"ExhaustRequire": True, "ArrangeOrder": [2, "true"], "RestInFull": True},  # 稀音的干员配置
    "巫恋": {"ArrangeOrder": [2, "true"]},  # 巫恋的干员配置
    "柏喙": {"ExhaustRequire": True, "ArrangeOrder": [2, "true"]},  # 柏喙的干员配置
    "龙舌兰": {"ArrangeOrder": [2, "true"]},  # 龙舌兰的干员配置
    "空弦": {"ArrangeOrder": [2, "true"], "RestingPriority": "low"},  # 空弦的干员配置
    # 定义了一个名为"伺夜"的字典，包含了"ArrangeOrder"和"RestingPriority"两个键值对
    "伺夜": {"ArrangeOrder": [2, "true"], "RestingPriority": "low"},
    # 定义了一个名为"绮良"的字典，包含了"ArrangeOrder"键值对
    "绮良": {"ArrangeOrder": [2, "true"]},
    # 定义了一个名为"但书"的字典，包含了"ArrangeOrder"键值对
    "但书": {"ArrangeOrder": [2, "true"]},
    # 定义了一个名为"泡泡"的字典，包含了"ArrangeOrder"键值对
    "泡泡": {"ArrangeOrder": [2, "true"]},
    # 定义了一个名为"火神"的字典，包含了"ArrangeOrder"键值对
    "火神": {"ArrangeOrder": [2, "true"]},
    # 定义了一个名为"黑键"的字典，包含了"ArrangeOrder"键值对
    "黑键": {"ArrangeOrder": [2, "true"]},
    # 定义了一个名为"波登可"的字典，包含了"ArrangeOrder"键值对
    "波登可": {"ArrangeOrder": [2, "false"]},
    # 定义了一个名为"夜莺"的字典，包含了"ArrangeOrder"键值对
    "夜莺": {"ArrangeOrder": [2, "false"]},
    # 定义了一个名为"菲亚梅塔"的字典，包含了"ArrangeOrder"键值对
    "菲亚梅塔": {"ArrangeOrder": [2, "false"]},
    # 定义了一个名为"流明"的字典，包含了"ArrangeOrder"键值对
    "流明": {"ArrangeOrder": [2, "false"]},
    # 定义了一个名为"蜜莓"的字典，包含了"ArrangeOrder"键值对
    "蜜莓": {"ArrangeOrder": [2, "false"]},
    # 定义了一个名为"闪灵"的字典，包含了"ArrangeOrder"键值对
    "闪灵": {"ArrangeOrder": [2, "false"]},
    # 定义了一个名为"杜林"的字典，包含了"ArrangeOrder"键值对
    "杜林": {"ArrangeOrder": [2, "false"]},
    # 定义了一个名为"褐果"的字典，包含了"ArrangeOrder"键值对
    "褐果": {"ArrangeOrder": [2, "false"]},
    # 定义了一个名为"车尔尼"的字典，包含了"ArrangeOrder"键值对
    "车尔尼": {"ArrangeOrder": [2, "false"]},
    # 定义了一个名为"安比尔"的字典，包含了"ArrangeOrder"键值对
    "安比尔": {"ArrangeOrder": [2, "false"]},
    # 定义了一个名为"爱丽丝"的字典，包含了"ArrangeOrder"键值对
    "爱丽丝": {"ArrangeOrder": [2, "false"]},
    # 定义了一个名为"桃金娘"的字典，包含了"ArrangeOrder"键值对
    "桃金娘": {"ArrangeOrder": [2, "false"]},
    # 定义了一个名为"帕拉斯"的字典，包含了"RestingPriority"键值对
    "帕拉斯": {"RestingPriority": "low"},
    # 定义了一个名为"红云"的字典，包含了"RestingPriority"和"ArrangeOrder"两个键值对
    "红云": {"RestingPriority": "low", "ArrangeOrder": [2, "true"]},
    # 定义了一个名为"承曦格雷伊"的字典，包含了"ArrangeOrder"和"RestInFull"两个键值对
    "承曦格雷伊": {"ArrangeOrder": [2, "true"], "RestInFull": True},
    # 定义了一个名为"乌有"的字典，包含了"ArrangeOrder"和"RestingPriority"两个键值对
    "乌有": {"ArrangeOrder": [2, "true"], "RestingPriority": "low"},
    # 定义了一个名为"图耶"的字典，包含了"ArrangeOrder"键值对
    "图耶": {"ArrangeOrder": [2, "true"]},
    # 定义了一个名为"鸿雪"的字典，包含了"ArrangeOrder"键值对
    "鸿雪": {"ArrangeOrder": [2, "true"]},
    # 定义了一个名为"孑"的字典，包含了"ArrangeOrder"键值对
    "孑": {"ArrangeOrder": [2, "true"]},
    # 定义了一个名为"清道夫"的字典，包含了"ArrangeOrder"键值对
    "清道夫": {"ArrangeOrder": [2, "true"]},
    # 定义了一个名为"临光"的字典，包含了"ArrangeOrder"键值对
    "临光": {"ArrangeOrder": [2, "true"]},
    # 定义了一个名为"杜宾"的字典，包含了"ArrangeOrder"键值对
    "杜宾": {"ArrangeOrder": [2, "true"]},
    # 定义了一个名为"焰尾"的字典，包含了"RestInFull"键值对
    "焰尾": {"RestInFull": True},
    # 定义了一个名为"重岳"的字典，包含了"ArrangeOrder"键值对
    "重岳": {"ArrangeOrder": [2, "true"]},
    # 定义了一个名为"坚雷"的字典，包含了"ArrangeOrder"键值对
    "坚雷": {"ArrangeOrder": [2, "true"]},
    # 定义了一个名为"年"的字典，包含了"RestingPriority"键值对
    "年": {"RestingPriority": "low"},
    # 定义了一个名为"伊内丝"的字典，包含了"ExhaustRequire"、"ArrangeOrder"和"RestInFull"三个键值对
    "伊内丝": {"ExhaustRequire": True, "ArrangeOrder": [2, "true"], "RestInFull": True},
    # 定义了一个名为"铅踝"的字典，包含了"LowerLimit"和"UpperLimit"两个键值对
    "铅踝":{"LowerLimit": 8,"UpperLimit": 12},
# 调试日志函数，设置日志级别为 DEBUG
def debuglog():
    logger.handlers[0].setLevel('DEBUG')

# 保存日志和截图的位置信息，设置默认保存路径和一些配置参数
def savelog():
    config.LOGFILE_PATH = './log'  # 日志保存路径
    config.SCREENSHOT_PATH = './screenshot'  # 截图保存路径
    config.SCREENSHOT_MAXNUM = 30  # 最大截图数量
    config.ADB_DEVICE = maa_config['maa_adb']  # ADB 设备配置
    config.ADB_CONNECT = maa_config['maa_adb']  # ADB 连接配置
    config.MAX_RETRYTIME = 10  # 最大重试次数
    config.PASSWORD = '你的密码'  # 密码配置
    config.APPNAME = 'com.hypergryph.arknights'  # 应用名称配置
    config.TAP_TO_LAUNCH["enable"] = False  # 是否启用点击启动
    config.TAP_TO_LAUNCH["x"], config.TAP_TO_LAUNCH["y"] = 0,0  # 点击启动坐标
    config.ADB_BINARY = ['F:\\MAA-v4.20.0-win-x64\\adb\\platform-tools\\adb.exe']  # ADB 可执行文件路径
    init_fhlr()  # 初始化文件处理器

# 初始化函数，创建设备和求解器对象
def inialize(tasks, scheduler=None):
    device = Device()  # 创建设备对象
    cli = Solver(device)  # 创建求解器对象
    # 如果调度器为空，则创建一个基础调度器对象
    if scheduler is None:
        base_scheduler = BaseSchedulerSolver(cli.device, cli.recog)
        base_scheduler.package_name = config.APPNAME
        base_scheduler.operators = {}
        base_scheduler.global_plan = plan
        base_scheduler.current_base = {}
        base_scheduler.resting = []
        base_scheduler.current_plan = base_scheduler.global_plan[base_scheduler.global_plan["default"]]
        # 设置同时休息的最大人数
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
        # 返回基础调度器对象
        return base_scheduler
    # 如果调度器不为空，则更新调度器的设备和识别器，并处理错误
    else:
        scheduler.device = cli.device
        scheduler.recog = cli.recog
        scheduler.handle_error(True)
        # 返回更新后的调度器对象
        return scheduler
# 保存当前状态到文件
def save_state():
    # 以写入模式打开状态文件
    with open(state_file_name, 'w') as f:
        # 如果基础调度器存在并且操作数据不为空，则将操作数据转换成字典并写入文件
        if base_scheduler is not None and base_scheduler.op_data is not None:
            json.dump(vars(base_scheduler.op_data), f, default=str)

# 从文件加载状态
def load_state():
    # 如果状态文件不存在，则返回空
    if not os.path.exists(state_file_name):
        return None

    # 以读取模式打开状态文件
    with open(state_file_name, 'r') as f:
        # 从文件中加载状态数据
        state = json.load(f)
    # 将操作数据中的字符串转换成对应的对象
    operators = {k: eval(v) for k, v in state['operators'].items()}
    # 将时间戳字符串转换成 datetime 对象
    for k, v in operators.items():
        if not v.time_stamp == 'None':
            v.time_stamp = datetime.strptime(v.time_stamp, '%Y-%m-%d %H:%M:%S.%f')
        else:
            v.time_stamp = None
    return operators

# 模拟执行
def simulate():
    '''
    具体调用方法可见各个函数的参数说明
    '''
    global ope_list, base_scheduler
    # 第一次执行任务
    # 定义任务字符串并将其转换成任务对象列表
    taskstr = "SchedulerTask(time='2023-06-11 21:39:15.108665',task_plan={'room_3_2': ['Current', '但书', '龙舌兰']},task_type='room_3_2',meta_flag=False)||SchedulerTask(time='2023-06-11 21:44:48.187074',task_plan={'room_2_1': ['砾', '槐琥', '斑点']},task_type='dorm0,dorm1,dorm2',meta_flag=False)||SchedulerTask(time='2023-06-11 22:17:53.720905',task_plan={'room_1_1': ['Current', '龙舌兰', '但书']},task_type='room_1_1',meta_flag=False)||SchedulerTask(time='2023-06-11 23:02:10.469026',task_plan={'meeting': ['Current', '见行者']},task_type='dorm3',meta_flag=False)||SchedulerTask(time='2023-06-11 23:22:15.236154',task_plan={},task_type='菲亚梅塔',meta_flag=False)||SchedulerTask(time='2023-06-12 11:25:55.925731',task_plan={},task_type='impart',meta_flag=False)||SchedulerTask(time='2023-06-12 11:25:55.926731',task_plan={},task_type='maa_Mall',meta_flag=False)"
    tasks = [eval(t) for t in taskstr.split("||")]
    # 将任务对象中的时间字符串转换成 datetime 对象
    for t in tasks:
        t.time = datetime.strptime(t.time, '%Y-%m-%d %H:%M:%S.%f')
    # 定义重连最大尝试次数和当前尝试次数
    reconnect_max_tries = 10
    reconnect_tries = 0
    success = False
    # 当成功标志为假时，执行循环
    while not success:
        # 尝试初始化任务调度器
        try:
            base_scheduler = inialize(tasks)
            # 设置成功标志为真
            success = True
        # 捕获任何异常
        except Exception as E:
            # 重连尝试次数加一
            reconnect_tries+=1
            # 如果重连尝试次数小于3，重新启动模拟器并继续循环
            if reconnect_tries <3:
                restart_simulator(simulator)
                continue
            # 如果重连尝试次数大于等于3，抛出异常
            else:
                raise E
    # 如果基础调度器的高度不为1080或宽度不为1920，记录错误并返回
    if base_scheduler.recog.h!=1080 or base_scheduler.recog.w!=1920:
        logger.error("模拟器分辨率不为1920x1080")
        return
    # 初始化操作员，获取验证消息
    validation_msg = base_scheduler.initialize_operators()
    # 如果验证消息不为空，记录错误并返回
    if validation_msg is not None:
        logger.error(validation_msg)
        return
    # 加载状态，获取加载的操作员数据
    _loaded_operators = load_state()
    # 如果加载的操作员数据不为空
    if _loaded_operators is not None:
        # 遍历加载的操作员数据
        for k, v in _loaded_operators.items():
            # 如果操作员在基础调度器的操作员数据中，并且不是以"dorm"开头的房间
            if k in base_scheduler.op_data.operators and not base_scheduler.op_data.operators[k].room.startswith(
                    "dorm"):
                # 只复制心情数据
                base_scheduler.op_data.operators[k].mood = v.mood
                base_scheduler.op_data.operators[k].time_stamp = v.time_stamp
                base_scheduler.op_data.operators[k].depletion_rate = v.depletion_rate
                base_scheduler.op_data.operators[k].current_room = v.current_room
                base_scheduler.op_data.operators[k].current_index = v.current_index
    # 进入无限循环
    while True:
        # 尝试执行以下代码块，捕获可能出现的异常
        try:
            # 如果任务列表不为空
            if len(base_scheduler.tasks) > 0:
                # 对任务列表按照时间进行排序
                (base_scheduler.tasks.sort(key=lambda x: x.time, reverse=False))
                # 计算下一个任务的等待时间
                sleep_time = (base_scheduler.tasks[0].time - datetime.now()).total_seconds()
                # 记录任务列表中的任务信息
                logger.info('||'.join([str(t) for t in base_scheduler.tasks]))
                # 发送邮件通知，包括任务的时间信息
                base_scheduler.send_email(task_template.render(tasks=[obj.time_offset(timezone_offset) for obj in base_scheduler.tasks]), '', 'html')
                # 如果下一个任务的等待时间超过9分钟，则启动MAA
                if sleep_time > 540:
                    base_scheduler.maa_plan_solver()
                # 如果等待时间大于0，则进入休眠状态
                elif sleep_time > 0:
                    time.sleep(sleep_time)
            # 如果任务列表不为空，并且下一个任务的类型是MAA
            if len(base_scheduler.tasks) > 0 and base_scheduler.tasks[0].type.split('_')[0] == 'maa':
                # 解决MAA计划
                base_scheduler.maa_plan_solver((base_scheduler.tasks[0].type.split('_')[1]).split(','), one_time=True)
                # 继续下一次循环
                continue
            # 运行任务调度器
            base_scheduler.run()
            # 重连尝试次数归零
            reconnect_tries = 0
        # 捕获连接错误、连接中断错误、属性错误等异常
        except ConnectionError or ConnectionAbortedError or AttributeError as e:
            # 重连尝试次数加一
            reconnect_tries += 1
            # 如果重连尝试次数小于最大重连次数
            if reconnect_tries < reconnect_max_tries:
                # 记录警告信息
                logger.warning(f'连接端口断开....正在重连....')
                # 连接状态设为False
                connected = False
                # 在连接状态为False时循环
                while not connected:
                    # 尝试重新初始化任务调度器
                    try:
                        base_scheduler = inialize([], base_scheduler)
                        break
                    # 捕获运行时错误、连接错误、连接中断错误等异常
                    except RuntimeError or ConnectionError or ConnectionAbortedError as ce:
                        # 记录错误信息
                        logger.error(ce)
                        # 重启模拟器
                        restart_simulator(simulator)
                        # 继续下一次循环
                        continue
                # 继续下一次循环
                continue
            # 如果重连尝试次数超过最大重连次数
            else:
                # 抛出异常
                raise Exception(e)
        # 捕获运行时错误
        except RuntimeError as re:
            # 重启模拟器
            restart_simulator(simulator)
        # 捕获其他异常
        except Exception as E:
            # 记录异常信息
            logger.exception(f"程序出错--->{E}")

    # cli.credit()  # 信用
    # ope_lists = cli.ope(eliminate=True, plan=ope_lists)  # 行动，返回未完成的作战计划
    # 调用商店功能
    cli.shop(shop_priority)  # 商店
    # 调用公招功能
    cli.recruit()  # 公招
    # 调用任务功能
    cli.mission()  # 任务
# 注册在程序退出时保存状态的函数
atexit.register(save_state)
# 保存日志
savelog()
# 模拟程序执行
simulate()
```