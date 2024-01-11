# `arknights-mower\arknights_mower\__main__.py`

```
# 导入模块
import atexit
import os
import time
from datetime import datetime
from arknights_mower.utils.log import logger
import json

# 导入深拷贝函数
from copy import deepcopy

# 导入管道和模拟器重启函数
from arknights_mower.utils.pipe import Pipe
from arknights_mower.utils.simulator import restart_simulator
from arknights_mower.utils.email import task_template

# 初始化全局变量
conf = {}
plan = {}
operators = {}

# 执行自动排班
def main(c, p, o={}, child_conn=None):
    # 初始化参数
    __init_params__()
    # 导入日志模块和配置模块
    from arknights_mower.utils.log import init_fhlr
    from arknights_mower.utils import config
    # 声明全局变量
    global plan
    global conf
    global operators
    # 赋值参数
    conf = c
    plan = p
    operators = o
    # 配置日志文件路径和截图路径
    config.LOGFILE_PATH = './log'
    config.SCREENSHOT_PATH = './screenshot'
    config.SCREENSHOT_MAXNUM = conf['screenshot']
    config.ADB_DEVICE = [conf['adb']]
    config.ADB_CONNECT = [conf['adb']]
    config.ADB_CONNECT = [conf['adb']]
    config.APPNAME = 'com.hypergryph.arknights' if conf['package_type'] == 1 else 'com.hypergryph.arknights.bilibili'  # 服务器
    config.TAP_TO_LAUNCH = conf['tap_to_launch_game']
    # 初始化文件句柄
    init_fhlr(child_conn)
    # 设置管道连接
    Pipe.conn = child_conn
    # 根据计划配置调整代理基础配置
    if plan['conf']['ling_xi'] == 1:
        agent_base_config['令']['UpperLimit'] = 12
        agent_base_config['夕']['LowerLimit'] = 12
    elif plan['conf']['ling_xi'] == 2:
        agent_base_config['夕']['UpperLimit'] = 12
        agent_base_config['令']['LowerLimit'] = 12
    # 根据计划配置设置代理基础配置的休息状态
    for key in list(filter(None, plan['conf']['rest_in_full'].replace('，', ',').split(','))):
        if key in agent_base_config.keys():
            agent_base_config[key]['RestInFull'] = True
        else:
            agent_base_config[key] = {'RestInFull': True}
    # 根据计划配置设置代理基础配置的疲劳状态
    for key in list(filter(None, plan['conf']['exhaust_require'].replace('，', ',').split(','))):
        if key in agent_base_config.keys():
            agent_base_config[key]['ExhaustRequire'] = True
        else:
            agent_base_config[key] = {'ExhaustRequire': True}
    # 遍历计划中工作狂的配置列表，将中文逗号替换为英文逗号后分割成列表
    for key in list(filter(None, plan['conf']['workaholic'].replace('，', ',').split(','))):
        # 如果配置键存在于 agent_base_config 中，则将其对应的值设为 True
        if key in agent_base_config.keys():
            agent_base_config[key]['Workaholic'] = True
        # 如果配置键不存在于 agent_base_config 中，则将其添加到 agent_base_config 中，并设置其值为 {'Workaholic': True}
        else:
            agent_base_config[key] = {'Workaholic': True}
    # 遍历计划中休息优先级的配置列表，将中文逗号替换为英文逗号后分割成列表
    for key in list(filter(None, plan['conf']['resting_priority'].replace('，', ',').split(','))):
        # 如果配置键存在于 agent_base_config 中，则将其对应的值设为 'low'
        if key in agent_base_config.keys():
            agent_base_config[key]['RestingPriority'] = 'low'
        # 如果配置键不存在于 agent_base_config 中，则将其添加到 agent_base_config 中，并设置其值为 {'RestingPriority': 'low'}
        else:
            agent_base_config[key] = {'RestingPriority': 'low'}
    # 输出信息日志，表示开始运行 Mower
    logger.info('开始运行Mower')
    # 输出调试日志，打印 agent_base_config 的内容
    logger.debug(agent_base_config)
    # 调用 simulate() 函数开始模拟
    simulate()
# 定义一个函数，用于将秒数格式化为小时和分钟的字符串
def format_time(seconds):
    # 计算小时
    rest_hours = int(seconds / 3600)
    # 计算分钟
    rest_minutes = int((seconds % 3600) / 60)
    # 根据小时是否为零来决定返回的格式化字符串
    if rest_hours == 0: 
        return f"{rest_minutes} 分钟"
    else:
        return f"{rest_hours} 小时 {rest_minutes} 分钟"

# 定义一个函数，用于隐藏配置中的密码
def hide_password(conf):
    # 深拷贝配置
    hpconf = deepcopy(conf)
    # 将密码字段替换为相同长度的星号
    hpconf["pass_code"] = "*" * len(conf["pass_code"])
    return hpconf

# 定义一个函数，用于更新设置
def update_conf():
    # 记录调试信息
    logger.debug("运行中更新设置")
    # 如果管道不存在或连接已关闭，则记录错误并返回
    if not Pipe or not Pipe.conn:
        logger.error("管道关闭")
        logger.info(maa_config)
        return
    # 记录调试信息
    logger.debug("通过管道发送更新设置请求")
    # 通过管道发送更新设置请求
    Pipe.conn.send({"type": "update_conf"})
    # 记录调试信息
    logger.debug("开始通过管道读取设置")
    # 通过管道接收设置
    conf = Pipe.conn.recv()
    # 记录调试信息，隐藏密码后输出设置
    logger.debug(f"接收设置：{hide_password(conf)}")
    return conf

# 定义一个函数，用于设置 MAA 选项
def set_maa_options(base_scheduler):
    # 更新设置
    conf = update_conf()
    # 将全局变量 maa_config 更新为从设置中获取的值
    global maa_config
    maa_config['maa_enable'] = conf['maa_enable']
    maa_config['maa_path'] = conf['maa_path']
    maa_config['maa_adb_path'] = conf['maa_adb_path']
    maa_config['maa_adb'] = conf['adb']
    maa_config['weekly_plan'] = conf['maa_weekly_plan']
    maa_config['roguelike'] = conf['maa_rg_enable'] == 1
    maa_config['rogue_theme'] = conf['maa_rg_theme']
    maa_config['sleep_min'] = conf['maa_rg_sleep_min']
    maa_config['sleep_max'] = conf['maa_rg_sleep_max']
    maa_config['maa_execution_gap'] = conf['maa_gap']
    maa_config['buy_first'] = conf['maa_mall_buy']
    maa_config['blacklist'] = conf['maa_mall_blacklist']
    maa_config['recruitment_time'] = conf['maa_recruitment_time']
    maa_config['recruit_only_4'] = conf['maa_recruit_only_4']
    maa_config['conn_preset'] = conf['maa_conn_preset']
    maa_config['touch_option'] = conf['maa_touch_option']
    maa_config['mall_ignore_when_full'] = conf['maa_mall_ignore_blacklist_when_full']
    maa_config['credit_fight'] = conf['maa_credit_fight']
    maa_config['rogue'] = conf['rogue']
    # 将基础调度器的 maa_config 更新为最新的 maa_config
    base_scheduler.maa_config = maa_config
    # 使用 debug 级别的日志记录器记录更新 Maa 设置的操作，包括 base_scheduler.maa_config 的值
    logger.debug(f"更新Maa设置：{base_scheduler.maa_config}")
# 初始化函数，用于初始化任务和调度器
def initialize(tasks, scheduler=None):
    # 导入所需的模块和类
    from arknights_mower.solvers.base_schedule import BaseSchedulerSolver
    from arknights_mower.strategy import Solver
    from arknights_mower.utils.device import Device
    from arknights_mower.utils import config
    # 创建设备对象
    device = Device()
    # 创建求解器对象
    cli = Solver(device)
    # 如果传入了调度器对象，则设置调度器的设备和识别器，并返回调度器
    else:
        scheduler.device = cli.device
        scheduler.recog = cli.recog
        scheduler.handle_error(True)
        return scheduler

# 模拟函数
def simulate():
    '''
    具体调用方法可见各个函数的参数说明
    '''
    # 初始化任务列表和重连最大尝试次数
    tasks = []
    reconnect_max_tries = 10
    reconnect_tries = 0
    # 声明全局变量 base_scheduler 和 success 标志
    global base_scheduler
    success = False
    # 循环直到成功连接或达到最大重连次数
    while not success:
        try:
            # 初始化调度器
            base_scheduler = initialize(tasks)
            success = True
        except Exception as E:
            # 捕获异常并尝试重连
            reconnect_tries += 1
            if reconnect_tries < 3:
                restart_simulator(conf['simulator'])
                continue
            else:
                raise E
    # 检查模拟器分辨率是否为1920x1080，如果不是则记录错误并返回
    if base_scheduler.recog.h != 1080 or base_scheduler.recog.w != 1920:
        logger.error("模拟器分辨率不为1920x1080")
        return
    # 初始化操作员信息，如果有错误则记录并返回
    validation_msg = base_scheduler.initialize_operators()
    if validation_msg is not None:
        logger.error(validation_msg)
        return
    # 如果操作员信息不为空，则更新操作员数据
    if operators != {}:
        for k, v in operators.items():
            if k in base_scheduler.op_data.operators and not base_scheduler.op_data.operators[k].room.startswith("dorm"):
                # 只复制心情数据
                base_scheduler.op_data.operators[k].mood = v.mood
                base_scheduler.op_data.operators[k].time_stamp = v.time_stamp
                base_scheduler.op_data.operators[k].depletion_rate = v.depletion_rate
                base_scheduler.op_data.operators[k].current_room = v.current_room
                base_scheduler.op_data.operators[k].current_index = v.current_index
    # 如果计划中的配置中的灵犀值为1或2
    if plan['conf']['ling_xi'] in [1, 2]:
        # 对于夕和令两个名字
        for name in ["夕","令"]:
            # 如果名字在操作数据的操作员中，并且操作员的组不为空
            if name in base_scheduler.op_data.operators and base_scheduler.op_data.operators[name].group !="":
                # 对于操作数据中操作员的组中的每个组名
                for group_name in base_scheduler.op_data.groups[base_scheduler.op_data.operators[name].group]:
                    # 如果组名不是夕或令
                    if group_name not in ["夕","令"]:
                        # 设置组名对应的操作员的心情下限为12
                        base_scheduler.op_data.operators[group_name].lower_limit = 12
                        # 记录日志，自动设置组名的心情下限为12
                        logger.info(f"自动设置{group_name}心情下限为12")
# 保存状态数据到 JSON 文件
def save_state(op_data, file='state.json'):
    # 如果临时文件夹不存在，则创建
    if not os.path.exists('tmp'):
        os.makedirs('tmp')
    # 打开文件，将操作数据转换为 JSON 格式写入文件
    with open('tmp/' + file, 'w') as f:
        if op_data is not None:
            json.dump(vars(op_data), f, default=str)

# 从 JSON 文件中加载状态数据
def load_state(file='state.json'):
    # 如果指定文件不存在，则返回 None
    if not os.path.exists('tmp/' + file):
        return None
    # 打开文件，加载状态数据
    with open('tmp/' + file, 'r') as f:
        state = json.load(f)
    # 将操作数据转换为对应的对象
    operators = {k: eval(v) for k, v in state['operators'].items()}
    # 将时间戳字符串转换为 datetime 对象
    for k, v in operators.items():
        if not v.time_stamp == 'None':
            v.time_stamp = datetime.strptime(v.time_stamp, '%Y-%m-%d %H:%M:%S.%f')
        else:
            v.time_stamp = None
    # 记录日志信息
    logger.info("基建配置已加载！")
    # 返回操作数据
    return operators

# 初始化参数
agent_base_config = {}
maa_config = {}

def __init_params__():
    # 声明全局变量
    global agent_base_config
    global maa_config
    # 初始化 maa_config 字典
    maa_config = {
        # maa 运行的时间间隔，以小时计
        "maa_execution_gap": 4,
        # 以下配置，第一个设置为true的首先生效
        # 是否启动肉鸽
        "roguelike": False,
        # 是否启动生息演算
        "reclamation_algorithm": False,
        # 是否启动保全派驻
        "stationary_security_service": False,
        "last_execution": None
    }
```