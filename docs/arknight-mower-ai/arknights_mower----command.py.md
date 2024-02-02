# `arknights-mower\arknights_mower\command.py`

```py
# 导入未来版本的注解特性
from __future__ import annotations

# 导入当前包的版本信息
from . import __version__
# 导入解决方案模块
from .solvers import *
# 导入配置工具模块
from .utils import config
# 导入设备模块
from .utils.device import Device
# 导入日志模块
from .utils.log import logger
# 导入参数错误异常和解析操作参数的工具模块
from .utils.param import ParamError, parse_operation_params

# 定义邮件处理函数，接受参数列表和设备对象
def mail(args: list[str] = [], device: Device = None):
    """
    mail
        自动收取邮件
    """
    # 使用邮件解决方案对象运行
    MailSolver(device).run()

# 定义基建处理函数，接受参数列表和设备对象
def base(args: list[str] = [], device: Device = None):
    """
    base [plan] [-c] [-d[F][N]] [-f[F][N]]
        自动处理基建的信赖/货物/订单/线索/无人机
        plan 表示选择的基建干员排班计划（建议搭配配置文件使用, 也可命令行直接输入）
        -c 是否自动收集并使用线索
        -d 是否自动消耗无人机，F 表示第几层（1-3），N 表示从左往右第几个房间（1-3）
        -f 是否使用菲亚梅塔恢复特定房间干员心情，恢复后恢复原位且工作位置不变，F、N 含义同上
    """
    # 从数据模块中导入基建房间列表和干员列表
    from .data import base_room_list, agent_list
    
    # 初始化排班计划、线索收集标志、无人机房间、菲亚梅塔房间、任意房间列表和干员列表
    arrange = None
    clue_collect = False
    drone_room = None
    fia_room = None
    any_room = []
    agents = []

    try:
        # 遍历参数列表
        for p in args:
            # 判断参数类型
            if p[0] == '-':
                if p[1] == 'c':
                    clue_collect = True
                elif p[1] == 'd':
                    assert '1' <= p[2] <= '3'
                    assert '1' <= p[3] <= '3'
                    drone_room = f'room_{p[2]}_{p[3]}'
                elif p[1] == 'f':
                    assert '1' <= p[2] <= '3'
                    assert '1' <= p[3] <= '3'
                    fia_room = f'room_{p[2]}_{p[3]}'
            elif arrange is None:
                arrange = config.BASE_CONSTRUCT_PLAN.get(p)
                if arrange is None:
                    if p in base_room_list:
                        any_room.append(p)
                        agents.append([])
                    elif p in agent_list or 'free' == p.lower():
                        agents[-1].append(p)                
    except Exception:
        # 捕获异常并抛出参数错误
        raise ParamError
    
    # 如果排班计划为空且任意房间列表不为空且干员列表长度大于0，则构建排班计划
    if arrange is None and any_room is not None and len(agents) > 0:
        arrange = dict(zip(any_room, agents))

    # 使用基建解决方案对象运行
    BaseConstructSolver(device).run(arrange, clue_collect, drone_room, fia_room)
# 定义一个名为 credit 的函数，参数为一个字符串列表和一个设备对象，默认为空列表和 None
def credit(args: list[str] = [], device: Device = None):
    """
    credit
        自动访友获取信用点
    """
    # 调用 CreditSolver 类的 run 方法
    CreditSolver(device).run()


# 定义一个名为 shop 的函数，参数为一个字符串列表和一个设备对象，默认为空列表和 None
def shop(args: list[str] = [], device: Device = None):
    """
    shop [items ...]
        自动前往商店消费信用点
        items 优先考虑的物品，若不指定则使用配置文件中的优先级，默认为从上到下从左到右购买
    """
    # 如果参数列表为空
    if len(args) == 0:
        # 调用 ShopSolver 类的 run 方法，传入配置文件中的优先级
        ShopSolver(device).run(config.SHOP_PRIORITY)
    else:
        # 调用 ShopSolver 类的 run 方法，传入参数列表
        ShopSolver(device).run(args)


# 定义一个名为 recruit 的函数，参数为一个字符串列表、两个空字典和一个设备对象，默认为空列表、空字典和 None
def recruit(args: list[str] = [], email_config={}, maa_config={},device: Device = None):
    """
    recruit [agents ...]
        自动进行公共招募
        agents 优先考虑的公招干员，若不指定则使用配置文件中的优先级，默认为高稀有度优先
    """
    # 如果参数列表为空
    if len(args) == 0:
        # 调用 RecruitSolver 类的 run 方法，传入配置文件中的优先级和两个配置字典
        RecruitSolver(device).run(config.RECRUIT_PRIORITY,email_config,maa_config)
    else:
        # 调用 RecruitSolver 类的 run 方法，传入参数列表和一个配置字典
        RecruitSolver(device).run(args,email_config)


# 定义一个名为 mission 的函数，参数为一个字符串列表和一个设备对象，默认为空列表和 None
def mission(args: list[str] = [], device: Device = None):
    """
    mission
        收集每日任务和每周任务奖励
    """
    # 调用 MissionSolver 类的 run 方法
    MissionSolver(device).run()


# 定义一个名为 operation 的函数，参数为一个字符串列表和一个设备对象，默认为空列表和 None
def operation(args: list[str] = [], device: Device = None):
    """
    operation [level] [n] [-r[N]] [-R[N]] [-e|-E]
        自动进行作战，可指定次数或直到理智不足
        level 指定关卡名称，未指定则默认前往上一次关卡
        n 指定作战次数，未指定则默认作战直到理智不足
        -r 是否自动回复理智，最多回复 N 次，N 未指定则表示不限制回复次数
        -R 是否使用源石回复理智，最多回复 N 次，N 未指定则表示不限制回复次数
        -e 是否优先处理未完成的每周剿灭，优先使用代理卡；-E 表示只使用代理卡而不消耗理智
    operation --plan
        （使用配置文件中的参数以及计划）自动进行作战
    """
    # 如果参数列表中只有一个元素且为 "--plan"
    if len(args) == 1 and args[0] == "--plan":
        # 调用 OpeSolver 类的 run 方法，传入配置文件中的参数和计划
        remain_plan = OpeSolver(device).run(None, config.OPE_TIMES, config.OPE_POTION,
                                            config.OPE_ORIGINITE, config.OPE_ELIMINATE, config.OPE_PLAN)
        # 更新配置文件中的作战计划
        config.update_ope_plan(remain_plan)
        return
    # 解析操作参数
    level, times, potion, originite, eliminate = parse_operation_params(args)
    # 调用 OpeSolver 类的 run 方法，传入解析后的参数
    OpeSolver(device).run(level, times, potion, originite, eliminate)


# 定义一个名为 version 的函数，参数为一个字符串列表和一个设备对象，默认为空列表和 None
def version(args: list[str] = [], device: Device = None):
    """
    version
        输出版本信息
    """
    # 打印版本信息
    print(f'arknights-mower: version: {__version__}')
def help(args: list[str] = [], device: Device = None):
    """
    help
        输出本段消息
    """
    # 输出使用说明
    print(
        'usage: arknights-mower command [command args] [--config filepath] [--debug]')
    print(
        'commands (prefix abbreviation accepted):')
    # 遍历全局命令列表，输出命令的文档字符串或者命令名
    for cmd in global_cmds:
        if cmd.__doc__:
            print('    ' + str(cmd.__doc__.strip()))
        else:
            print('    ' + cmd.__name__)
    # 输出调试信息和配置文件路径
    print(f'    --debug\n        启用调试功能，调试信息将会输出到 {config.LOGFILE_PATH} 中')
    print(f'    --config filepath\n        指定配置文件，默认使用 {config.PATH}')


"""
commands for schedule
operation will be replaced by operation_one in ScheduleSolver
"""
# 定义 schedule 模块的命令列表
schedule_cmds = [base, credit, mail, mission, shop, recruit, operation]


def add_tasks(solver: ScheduleSolver = None, tag: str = ''):
    """
    为 schedule 模块添加任务
    """
    # 从配置文件中获取计划
    plan = config.SCHEDULE_PLAN.get(tag)
    if plan is not None:
        # 遍历计划，添加任务到解决器中
        for args in plan:
            args = args.split()
            if 'schedule' in args:
                # 如果计划中包含了 'schedule'，则抛出未实现的错误
                logger.error(
                    'Found `schedule` in `schedule`. Are you kidding me?')
                raise NotImplementedError
            try:
                # 匹配命令并添加任务
                target_cmd = match_cmd(args[0], schedule_cmds)
                if target_cmd is not None:
                    solver.add_task(tag, target_cmd, args[1:])
            except Exception as e:
                logger.error(e)


def schedule(args: list[str] = [], device: Device = None):
    """
    schedule
        执行配置文件中的计划任务
        计划执行时会自动存档至本地磁盘，启动时若检测到有存档，则会使用存档内容继续完成计划
        -n 忽略之前中断的计划任务，按照配置文件重新开始新的计划
    """
    # 初始化新计划标志
    new_schedule = False

    try:
        # 检查命令行参数，判断是否需要新计划
        for p in args:
            if p[0] == '-':
                if p[1] == 'n':
                    new_schedule = True
    except Exception:
        # 抛出参数错误
        raise ParamError

    # 创建计划解决器对象
    solver = ScheduleSolver(device)
    # 如果有新的调度计划或者加载失败，则执行以下代码
    if new_schedule or solver.load_from_disk(schedule_cmds, match_cmd) is False:
        # 如果配置中存在调度计划
        if config.SCHEDULE_PLAN is not None:
            # 遍历调度计划中的标签
            for tag in config.SCHEDULE_PLAN.keys():
                # 向求解器中添加任务
                add_tasks(solver, tag)
        # 如果配置中不存在调度计划
        else:
            # 记录警告信息
            logger.warning('empty plan')
        # 执行每次运行前的操作
        solver.per_run()
    # 执行求解器的运行方法
    solver.run()
# 定义全局可用的命令列表
global_cmds = [base, credit, mail, mission, shop,
               recruit, operation, version, help, schedule]

# 匹配命令函数，接受一个前缀字符串和可用命令列表作为参数
def match_cmd(prefix: str, avail_cmds: list[str] = global_cmds):
    """ match command """
    # 从可用命令列表中筛选出以指定前缀开头的命令
    target_cmds = [x for x in avail_cmds if x.__name__.startswith(prefix)]
    # 如果只有一个匹配的命令，则返回该命令
    if len(target_cmds) == 1:
        return target_cmds[0]
    # 如果没有匹配的命令，则打印错误信息并返回 None
    elif len(target_cmds) == 0:
        print('unrecognized command: ' + prefix)
        return None
    # 如果有多个匹配的命令，则打印错误信息和匹配的命令列表，并返回 None
    else:
        print('ambiguous command: ' + prefix)
        print('matched commands: ' + ','.join(x.__name__ for x in target_cmds))
        return None
```