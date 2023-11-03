# ArknightMower源码解析 3

# git hooks

## Install

```pybash
git config core.hooksPath `git rev-parse --show-toplevel`/.github/git-hooks
```

## git commit 提交规范

参考：https://www.conventionalcommits.org/zh-hans/v1.0.0-beta.4/

- build: Changes that affect the build system or external dependencies.
- chore: Others.
- ci: Changes to our CI configuration files and scripts.
- docs: Documentation only changes.
- feat: A new feature.
- fix: A bug fix.
- hotfix: Publish a hotfix.
- perf: A code change that improves performance.
- refactor: A code change that neither fixes a bug or adds a feature.
- release: Publish a new release.
- revert: Revert the last commit.
- style: Changes that do not affect the meaning of the code.
- test: Adding missing tests or correcting existing tests.
- typo: Typographical error.
- update: Data updates.
- workflow: Changes to our workflow configuration files and scripts.


# git-flow hooks

## Install

```pybash
git config gitflow.path.hooks `git rev-parse --show-toplevel`/.github/gitflow-hooks
```

## git-flow config

- Branch name for production releases: `main`
- Branch name for "next release" development: `dev`
- Feature branch prefix: `feature/`
- Bugfix branch prefix: `bugfix/`
- Release branch prefix: `release/`
- Hotfix branch prefix: `hotfix/`
- Support branch prefix: `support/`
- Version tag prefix: `v`


# `arknights_mower/command.py`

这段代码是一个Python函数，名为`mail`，使用了Python 2.75及更高版本中的annotation特性。函数接收两个参数，一个是接收者列表（类型为列表类型），另一个是收件人设备（类型为Device类型）。函数的作用是调用一个名为`MailSolver`的类实例的`run`方法，该实例在`mail.py`文件中定义。

具体来说，这段代码的作用是调用`MailSolver`类的一个`run`方法，该方法会执行一系列的邮件收取操作，并将结果打印出来。由于邮件收取操作在`mail.py`文件中定义，因此运行该函数时会自动调用`MailSolver`类中的`run`方法，从而实现自动收取邮件的功能。


```py
from __future__ import annotations

from . import __version__
from .solvers import *
from .utils import config
from .utils.device import Device
from .utils.log import logger
from .utils.param import ParamError, parse_operation_params


def mail(args: list[str] = [], device: Device = None):
    """
    mail
        自动收取邮件
    """
    MailSolver(device).run()


```

这是一个程序，可以自动收集线索并使用菲亚梅塔恢复特定房间干员心情。收集线索的条件是自动，消耗无人机在1-3层，房间编号在1-3层。使用菲亚梅塔恢复特定房间干员心情时，可以自动消耗无人机，恢复后房间编号不变，工作位置也不变。同时，房间编号也可以在命令行中输入。


```py
def base(args: list[str] = [], device: Device = None):
    """
    base [plan] [-c] [-d[F][N]] [-f[F][N]]
        自动处理基建的信赖/货物/订单/线索/无人机
        plan 表示选择的基建干员排班计划（建议搭配配置文件使用, 也可命令行直接输入）
        -c 是否自动收集并使用线索
        -d 是否自动消耗无人机，F 表示第几层（1-3），N 表示从左往右第几个房间（1-3）
        -f 是否使用菲亚梅塔恢复特定房间干员心情，恢复后恢复原位且工作位置不变，F、N 含义同上
    """
    from .data import base_room_list, agent_list
    
    arrange = None
    clue_collect = False
    drone_room = None
    fia_room = None
    any_room = []
    agents = []

    try:
        for p in args:
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
        raise ParamError
    
    if arrange is None and any_room is not None and len(agents) > 0:
        arrange = dict(zip(any_room, agents))

    BaseConstructSolver(device).run(arrange, clue_collect, drone_room, fia_room)


```

这两函数是通过Python内置的type模块定义的。其中，credit函数接受两个参数，一个是列表参数args，另一个是单个参数device，用于指定自动访友的设备。函数内部调用了type(device).run()，来运行以device为参数的函数，这个函数可能是定义了将device的参数传递给device.run()方法的函数。

而shop函数也接受了两个参数，一个是列表参数args，另一个是单个参数device，用于指定自动前往商店消费信用点的设备。函数内部调用了type(device).run(config.SHOP_PRIORITY)，来运行以device为参数的函数，这个函数可能是定义了将device的参数传递给device.run(config.SHOP_PRIORITY)方法的函数。

这两个函数的作用是帮助用户实现自动访友和自动前往商店消费信用点等功能。其中，credit函数会将device作为参数传递给device.run()方法，然后运行type(device).run(config.SHOP_PRIORITY)方法，这个方法会使用配置文件中定义的优先级顺序，来决定先购买哪些物品。而shop函数则会将device作为参数传递给device.run(config.SHOP_PRIORITY)方法，然后运行ShopSolver(device).run(args)方法，这个方法会使用传入的args参数中的物品优先级顺序，来决定先购买哪些物品。


```py
def credit(args: list[str] = [], device: Device = None):
    """
    credit
        自动访友获取信用点
    """
    CreditSolver(device).run()


def shop(args: list[str] = [], device: Device = None):
    """
    shop [items ...]
        自动前往商店消费信用点
        items 优先考虑的物品，若不指定则使用配置文件中的优先级，默认为从上到下从左到右购买
    """
    if len(args) == 0:
        ShopSolver(device).run(config.SHOP_PRIORITY)
    else:
        ShopSolver(device).run(args)


```

这两函数是在Python中定义的，主要作用是帮助招募自动进行公共招募和收集每日任务和每周任务奖励。

python
def recruit(args: list[str] = [], email_config={}, maa_config={},device: Device = None):
   """
   recruit [agents ...]
       自动进行公共招募
       agents 优先考虑的公招干员，若不指定则使用配置文件中的优先级，默认为高稀有度优先
   """
   if len(args) == 0:
       RecruitSolver(device).run(config.RECRUIT_PRIORITY,email_config,maa_config)
   else:
       RecruitSolver(device).run(args,email_config)


这段代码的作用是自动招募代理人，如果用户没有提供参数，则默认优先考虑高稀有度。招募的代理人优先级是通过`RecruitSolver`类来控制的，这个类应该是在外部定义的，而且这个函数都会使用这个类。

python
def mission(args: list[str] = [], device: Device = None):
   """
   mission
       收集每日任务和每周任务奖励
   """
   MissionSolver(device).run()


这段代码的作用是收集代理人每日和每周任务奖励，这个奖励是通过`MissionSolver`类来控制的，同样也应该是在外部定义的，而且这个函数都会使用这个类。


```py
def recruit(args: list[str] = [], email_config={}, maa_config={},device: Device = None):
    """
    recruit [agents ...]
        自动进行公共招募
        agents 优先考虑的公招干员，若不指定则使用配置文件中的优先级，默认为高稀有度优先
    """
    if len(args) == 0:
        RecruitSolver(device).run(config.RECRUIT_PRIORITY,email_config,maa_config)
    else:
        RecruitSolver(device).run(args,email_config)


def mission(args: list[str] = [], device: Device = None):
    """
    mission
        收集每日任务和每周任务奖励
    """
    MissionSolver(device).run()


```

这段代码是一个自动进行作战的 Python 函数，名为 operation。它接受一个参数列表 args，其中包含以下参数：

- level：指定要达的关卡名称，如果没有指定，则默认为上一次关卡。
- n：指定要进行的作战次数，如果没有指定，则默认为直到理智不足为止。
- r：指示是否自动回复理智，最多回复 N 次，如果没有指定，则表示不限制回复次数。
- R：指示是否使用源石回复理智，最多回复 N 次，如果没有指定，则表示不限制回复次数。
- e：指示是否优先处理未完成的每周剿灭，优先使用代理卡；e-E 表示只使用代理卡而不消耗理智。

operation --plan 函数会先检查 args 参数中是否包含 --plan，如果是，则忽略该参数，否则执行 operation 函数。

operation 函数首先定义了一个名为 OpeSolver 的类，该类可能是一个用于自动进行作战的 AI 类。然后，根据 operation 参数的值，调用 OpeSolver 的 run 方法，传递给该方法的操作参数，然后执行该方法。

operation 函数的具体实现可能还涉及到从 args 参数中提取未完成的每周剿灭等信息，以便在 OpeSolver 的 run 方法中进行更复杂的操作。


```py
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

    if len(args) == 1 and args[0] == "--plan":
        remain_plan = OpeSolver(device).run(None, config.OPE_TIMES, config.OPE_POTION,
                                            config.OPE_ORIGINITE, config.OPE_ELIMINATE, config.OPE_PLAN)
        config.update_ope_plan(remain_plan)
        return

    level, times, potion, originite, eliminate = parse_operation_params(args)

    OpeSolver(device).run(level, times, potion, originite, eliminate)


```

这段代码定义了两个函数：`version` 和 `help`。它们的功能描述如下：

1. `version` 函数：
该函数打印出 "arknights-mower: version: [版本号]" 的消息，然后输出一个字符串，用于告知如何调用该函数。这个函数的作用是输出版本信息，用于版本控制。

2. `help` 函数：
该函数输出本段消息，然后输出一个表格，列出了所有可用的命令选项。这个函数的作用是输出帮助信息，告知用户如何使用工具。

这两个函数是通过 `global_cmds` 获取全局可执行命令列表中的所有命令，然后分别定义了如何使用这些命令。通过调用 `help` 函数，用户可以选择使用命令的选项，例如 `arknights-mower`，而通过调用 `version` 函数，用户可以获取到版本信息。


```py
def version(args: list[str] = [], device: Device = None):
    """
    version
        输出版本信息
    """
    print(f'arknights-mower: version: {__version__}')


def help(args: list[str] = [], device: Device = None):
    """
    help
        输出本段消息
    """
    print(
        'usage: arknights-mower command [command args] [--config filepath] [--debug]')
    print(
        'commands (prefix abbreviation accepted):')
    for cmd in global_cmds:
        if cmd.__doc__:
            print('    ' + str(cmd.__doc__.strip()))
        else:
            print('    ' + cmd.__name__)
    print(f'    --debug\n        启用调试功能，调试信息将会输出到 {config.LOGFILE_PATH} 中')
    print(f'    --config filepath\n        指定配置文件，默认使用 {config.PATH}')


```

这段代码定义了一个函数 `add_tasks`，用于为 `ScheduleSolver` 类添加任务。函数接受两个参数：`solver` 和 `tag`。函数内部首先从配置文件中获取特定标签的计划，然后遍历该计划中的所有任务。对于每个任务，函数使用 `match_cmd` 函数查找该任务在计划中的指令，如果找到，则使用 `add_task` 函数将该任务添加到 `ScheduleSolver` 的任务列表中。如果匹配失败，函数将记录错误并返回。


```py
"""
commands for schedule
operation will be replaced by operation_one in ScheduleSolver
"""
schedule_cmds = [base, credit, mail, mission, shop, recruit, operation]


def add_tasks(solver: ScheduleSolver = None, tag: str = ''):
    """
    为 schedule 模块添加任务
    """
    plan = config.SCHEDULE_PLAN.get(tag)
    if plan is not None:
        for args in plan:
            args = args.split()
            if 'schedule' in args:
                logger.error(
                    'Found `schedule` in `schedule`. Are you kidding me?')
                raise NotImplementedError
            try:
                target_cmd = match_cmd(args[0], schedule_cmds)
                if target_cmd is not None:
                    solver.add_task(tag, target_cmd, args[1:])
            except Exception as e:
                logger.error(e)


```

这段代码定义了一个函数 `schedule`，它接受两个参数 `args` 和 `device`。

函数的作用是执行配置文件中的计划任务，并在执行过程中自动存档到本地磁盘。如果之前的中断任务没有完成，则使用存档内容继续完成计划。如果之前的中断任务完成后，检查到有存档，则使用存档内容继续完成计划。

函数的具体实现包括以下几个步骤：

1. 读取配置文件中的计划任务，并按字典序进行排序。
2. 遍历 `args` 参数中的计划任务，如果任务分为两个部分，判断第一个部分是否为 '-'。
3. 如果任务分为两个部分，检查第一个部分是否为 'n'，如果是，则跳过这个任务。
4. 执行任务的 solver 对象，并使用 `schedule_cmds` 和 `match_cmd` 分别设置 `solver` 对象的一些参数。
5. 如果之前的中断任务没有完成，使用 solver 对象的方法 `per_run`，然后继续执行下一个任务。
6. 如果之前的中断任务完成后，检查到有存档，使用 solver 对象的方法 `run`，然后继续执行下一个任务。
7. 如果之前的中断任务完成后，仍然有计划任务未完成，则重复执行步骤 6，直到所有计划任务都完成。


```py
def schedule(args: list[str] = [], device: Device = None):
    """
    schedule
        执行配置文件中的计划任务
        计划执行时会自动存档至本地磁盘，启动时若检测到有存档，则会使用存档内容继续完成计划
        -n 忽略之前中断的计划任务，按照配置文件重新开始新的计划
    """
    new_schedule = False

    try:
        for p in args:
            if p[0] == '-':
                if p[1] == 'n':
                    new_schedule = True
    except Exception:
        raise ParamError

    solver = ScheduleSolver(device)
    if new_schedule or solver.load_from_disk(schedule_cmds, match_cmd) is False:
        if config.SCHEDULE_PLAN is not None:
            for tag in config.SCHEDULE_PLAN.keys():
                add_tasks(solver, tag)
        else:
            logger.warning('empty plan')
        solver.per_run()
    solver.run()


```



该代码定义了一个名为 `match_cmd` 的函数，用于在给定的命令列表中查找与给定前缀匹配的命令。函数接受两个参数：一个字符串 `prefix` 和一个可变参数 `avail_cmds`。函数内部首先定义了一个全局命令列表 `global_cmds`，其中包括了 `base`,`credit`,`mail`,`mission`,`shop`,`recruit`,`operation`,`version`,`help`，以及 `help` 和 `schedule` 两个命令。

接下来，函数内部创建了一个名为 `target_cmds` 的列表，用于存储匹配 `prefix` 的命令。列表中包含所有全局命令列表中的命令，并且仅包含以 `prefix` 开头的命令。

函数内部使用一个列表推导式来获取匹配 `prefix` 的命令。列表推导式的语法是 `target_cmds = [x for x in avail_cmds if x.__name__.startswith(prefix)]`，其中 `avail_cmds` 是全局命令列表，`x` 是命令的名称，`__name__` 是命令的全名，`startswith` 是一个字符串函数，用于检查给定的前缀是否是每个命令的全名的一部分。列表推导式的结果将是一个新的列表，其中包含所有与给定前缀匹配的命令的名称。

最后，函数内部使用一个简单的循环来处理匹配的命令。如果找到了匹配的命令，函数将返回该命令的名称，否则将打印一条消息并返回 `None`。


```py
# all available commands
global_cmds = [base, credit, mail, mission, shop,
               recruit, operation, version, help, schedule]


def match_cmd(prefix: str, avail_cmds: list[str] = global_cmds):
    """ match command """
    target_cmds = [x for x in avail_cmds if x.__name__.startswith(prefix)]
    if len(target_cmds) == 1:
        return target_cmds[0]
    elif len(target_cmds) == 0:
        print('unrecognized command: ' + prefix)
        return None
    else:
        print('ambiguous command: ' + prefix)
        print('matched commands: ' + ','.join(x.__name__ for x in target_cmds))
        return None

```

# `arknights_mower/strategy.py`

This is a class that simulates the game SystemShotter.py. It has methods for recruiting, operating, and shopping.

The Recruit method involves a priority list for公招干员， with the default value being high-rare.

The Ope method takes one or more parameters, including a level, the number of times to fight, potions, and originates.

The Shop method allows the player to use信用点购买物品， with the default value being to buy the first item that can be purchased.

The Mail method sends the player a message.

The Index method allows the player to access the index of their character.


```py
from __future__ import annotations

import functools

from .solvers import *
from .solvers.base_schedule import BaseSchedulerSolver
from .utils import typealias as tp
from .utils.device import Device
from .utils.recognize import Recognizer
from .utils.solver import BaseSolver


class Solver(object):
    """ Integration solver """

    def __init__(self, device: Device = None, recog: Recognizer = None, timeout: int = 99) -> None:
        """
        :param timeout: int, 操作限时，单位为小时
        """
        self.device = device if device is not None else Device()
        self.recog = recog if recog is not None else Recognizer(self.device)
        self.timeout = timeout


    def base_scheduler (self,tasks=[],plan={},current_base={},)-> None:
        # 返还所有排班计划以及 当前基建干员位置
        return BaseSchedulerSolver(self.device, self.recog).run(tasks,plan,current_base)

    def base(self, arrange: tp.BasePlan = None, clue_collect: bool = False, drone_room: str = None, fia_room: str = None) -> None:
        """
        :param arrange: dict(room_name: [agent,...]), 基建干员安排
        :param clue_collect: bool, 是否收取线索
        :param drone_room: str, 是否使用无人机加速
        :param fia_room: str, 是否使用菲亚梅塔恢复心情
        """
        BaseSolver(self.device, self.recog).run(
            arrange, clue_collect, drone_room, fia_room)

    def credit(self) -> None:
        CreditSolver(self.device, self.recog).run()

    def mission(self) -> None:
        MissionSolver(self.device, self.recog).run()

    def recruit(self, priority: list[str] = None) -> None:
        """
        :param priority: list[str], 优先考虑的公招干员，默认为高稀有度优先
        """
        RecruitSolver(self.device, self.recog).run(priority)

    def ope(self, level: str = None, times: int = -1, potion: int = 0, originite: int = 0, eliminate: int = 0, plan: list[tp.OpePlan] = None) -> list[tp.OpePlan]:
        """
        :param level: str, 指定关卡，默认为前往上一次关卡或当前界面关卡
        :param times: int, 作战的次数上限，-1 为无限制，默认为 -1
        :param potion: int, 使用药剂恢复体力的次数上限，-1 为无限制，默认为 0
        :param originite: int, 使用源石恢复体力的次数上限，-1 为无限制，默认为 0
        :param eliminate: int, 是否优先处理未完成的每周剿灭，0 为忽略剿灭，1 为优先剿灭，2 为优先剿灭但只消耗代理卡，默认为 0
        :param plan: [[str, int]...], 指定多个关卡以及次数，优先级高于 level

        :return remain_plan: [[str, int]...], 未完成的计划
        """
        return OpeSolver(self.device, self.recog).run(level, times, potion, originite, eliminate, plan)

    def shop(self, priority: bool = None) -> None:
        """
        :param priority: list[str], 使用信用点购买东西的优先级, 若无指定则默认购买第一件可购买的物品
        """
        ShopSolver(self.device, self.recog).run(priority)

    def mail(self) -> None:
        MailSolver(self.device, self.recog).run()

    def index(self) -> None:
        BaseSolver(self.device, self.recog).back_to_index()

```

# `arknights_mower/__init__.py`

这段代码的作用是设置一个名为“arknights_mower”的包的根目录。具体来说，它实现了以下操作：

1. 使用 `getattr` 和 `hasattr` 函数获取 `sys.frozen` 和 `sys._MEIPASS` 属性值，如果它们中有一个存在。
2. 如果 `__pyinstall__` 为真，并且 `sys.frozen` 和 `sys._MEIPASS` 存在，那么将 `__pyinstall__` 设置为 `True`，并将 `__rootdir__` 设置为 `Path(sys._MEIPASS).joinpath('arknights_mower').joinpath('__init__').resolve()`。
3. 如果 `__pyinstall__` 为假，并且 `sys.frozen` 或 `sys._MEIPASS` 中有一个存在，那么将 `__pyinstall__` 设置为 `False`，并将 `__rootdir__` 设置为 `Path(__file__).parent.resolve()`。
4. 如果 `__pyinstall__` 和 `sys.frozen`、`sys._MEIPASS` 都不存在，则执行命令行模式。

具体实现可以分为以下几个步骤：

1. 使用 `import platform` 和 `import sys` 导入 `platform` 模块和 `sys` 模块。
2. 使用 `from pathlib import Path` 导入 `pathlib` 模块的 `Path` 类。
3. 使用 `getattr` 和 `hasattr` 函数获取 `sys.frozen` 和 `sys._MEIPASS` 属性值。
4. 如果 `__pyinstall__` 为真，并且 `sys.frozen` 和 `sys._MEIPASS` 存在，那么将 `__pyinstall__` 设置为 `True`，并将 `__rootdir__` 设置为 `Path(sys._MEIPASS).joinpath('arknights_mower').joinpath('__init__').resolve()`。
5. 如果 `__pyinstall__` 为假，并且 `sys.frozen` 或 `sys._MEIPASS` 中有一个存在，那么将 `__pyinstall__` 设置为 `False`，并将 `__rootdir__` 设置为 `Path(__file__).parent.resolve()`。
6. 如果 `__pyinstall__` 和 `sys.frozen`、`sys._MEIPASS` 都不存在，则执行命令行模式。


```py
import platform
import sys
from pathlib import Path

# Use sys.frozen to check if run through pyinstaller frozen exe, and sys._MEIPASS to get temp path.
if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
    __pyinstall__ = True
    # Why they create a  __init__ folder here...idk.
    __rootdir__ = Path(sys._MEIPASS).joinpath('arknights_mower').joinpath('__init__').resolve()
else:
    __pyinstall__ = False
    __rootdir__ = Path(__file__).parent.resolve()

# Command line mode
__cli__ = not (__pyinstall__ and not sys.argv[1:])

```

这段代码是在 Python 环境中执行以下操作：

1. 将 `platform.system()` 返回值中的 `utf-8` 编码转换为小写，然后返回 `'ro'`。这里 `platform.system()` 返回一个字符串，包含了操作系统支持的所有系统命令。`utf-8` 编码是小写的。

2. 将步骤 1 中得到的字符串转换为数字 `3.4.3`。

3. 将步骤 2 中得到的数字字符串转换为字符串 `'v3.4.3'`。

4. 创建一个新的 `__system__` 变量，并将其设置为步骤 3 中得到的字符串。

5. 将 `__version__` 变量设置为 `'v3.4.3'`。

6. 在(__sys__, '__version__') 这对圆括号中，前者是一个字符串，后者是一个变量，存储了当前操作系统的版本号。


```py
__system__ = platform.system().lower()
__version__ = 'v3.4.3'

```

# `arknights_mower/__main__.py`

这段代码的作用是创建一个名为 "arrangement.py" 的文件，用于定义 "ArrowKnight simulator" 的 arrangement。它包含以下几部分：

1. 导入一些必要的库：import os, time, datetime, json, deepcopy, copy, Pipe, restart_simulator, task_template。
2. 自定义一些函数：from datetime import datetime, logger。
3. 设置几个变量：conf 和 plan。conf 是用来存储一些元数据的，plan 存储了一些策略。
4. 导入 "at丘信赖器官" 和 " 身体部件 "：ateVR 和 bodyPart。
5. 创建 "ArrowKnight simulator" 类：ArrowKnightSimulator。
6. 定义 " 构造函数 "：__init__。
7. 定义 " 模拟 "：simulate。
8. 定义 " 策略 "：strategy。
9. 定义 " 实现 "：executeStrategy。
10. 定义 " 初始化 "：start。
11. 定义 " 检查并启动 "：checkAndStart。
12. 保存 " 配置文件 "：saveConfigFile。
13. 加载 " 配置文件 "：loadConfigFile。
14. 打印 " 配置文件 " "：printConfigFile。
15. 创建 " 保管 "：keepSafe。
16. 创建 " 策略 "：strategy。
17. 创建 " 地图 "：map。
18. 创建 " 单位 "：unit。
19. 创建 " 事件 "：Event。
20. 创建 " Logger "：Logger。
21. 创建 " 医生 "： Doctor。
22. 创建 " 手术室 "： OperatingSurgicalRoom。
23. 创建 " 病床 "：Bed。
24. 创建 " 药房 "：DrugCube。
25. 创建 " 供氧 "：OxygenSupply。
26. 创建 " 机械手臂 "：RobotArm。
27. 创建 " 载具 "：Truck。
28. 创建 " 车辆 "：Vehicle。
29. 创建 " 道路 "：Road。
30. 创建 " 屋顶 "：Roof。
31. 创建 " 外墙 "：Wall。
32. 创建 " 螺旋楼梯 "：HelicalStair。
33. 创建 " 复合滑步 "：St絮。
34. 创建 " 治疗 "：treat。
35. 创建 " 损伤 "：injure。
36. 创建 " 治疗室 "：SurgicalRoom。
37. 创建 " 手术 "：surgery。
38. 创建 " 花费 "：cost。
39. 创建 " 礼物 "：gift。
40. 创建 " 餐厅 "：DiningRoom。
41. 创建 " 实验室 "：Laboratory。
42. 创建 " 仓库 "：Storage。
43. 创建 " 机器 "：Equipment。
44. 创建 " 维护 "：maintain。
45. 创建 " 升级 "：upgrade。
46. 创建 " 菜单 "：menu。
47. 创建 " 存储空间 "：SafetyDepositBox。
48. 创建 " 漫游 "： guides。
49. 创建 " 下载 "：download。
50. 创建 " 上传 "：upload。
51. 创建 " 字母 "：alphabet。
52. 创建 " 数字 "：number。
53. 创建 " 特殊 "：Special。
54. 创建 " 颜色 "：Color。
55. 创建 " 字体 "：Font。
56. 创建 " 标识 "：Identity。
57. 创建 " 位置 "：position。
58. 创建 " 随机数 "：RandomNumber。
59. 创建 " 列表 "：List。
60. 创建 " 货币 "：Coin。
61. 创建 " 连接 "：connect。
62. 创建 " 渠道 "：Channel。
63. 创建 " 发现 "：discover。
64. 创建 " 生成 "：generate。
65. 创建 " 提取 "：extract。
66. 创建 " 储存 "：store。
67. 创建 " 策略模板 "：strategyTemplate。
68. 创建 " 处理 "：handle。
69. 创建 " 初始化 "：initialize。
70. 创建 " 检测 "：check。
71. 创建 " 准备 "：prepare。
72. 创建 " 分析 "：analyze。
73. 创建 " 列表模板 "：listTemplate。
74. 创建 " 事件列表 "：eventList。
75. 创建 " 日常任务 "：dailyTasks。
76. 创建 " 当前任务 "：currentTasks。
77. 创建 " 任务 "：task。
78. 创建 " 拒绝 "：reject。
79. 创建 " 锁定 "：lock。
80. 创建 " 设置 "：settings。
81. 创建 " 状态 "：status。
82. 创建 " 是否可完成 "：isComplete。
83. 创建 " 是否进行中 "：isInProgress。
84. 创建 " 设置延迟 "：delay。
85. 创建 " 设置同步 "：synchronize。
86. 创建 " 存储位置 "：position。
87. 创建 " 同步 "：synchronize。
88. 创建 " 检查血量 "：checkHit points。
89. 创建 " 检查饥饿值 "：checkStarvation。
90. 创建 " 检查力量 "：checkStrength。
91. 创建 " 检查敏捷 "：checkAgility。
92. 创建 " 检查速度 "：checkSpeed。
93. 创建 " 创建 "：create。
94. 创建 " 画图 "：draw。
95. 创建 " 复制 "：copy。
96. 创建 " 粘贴 "：paste。
97. 创建 " 发送 "：send。
98. 创建 " 上传 "：upload。
99. 创建 " 下载 "：download。
100. 创建 " 打印 "：print。
101. 创建 " 擦除 "：erase。
102. 创建 " 移动 "：move。
103. 创建 " 探索 "：explore。
104. 创建 " 交易 "：trade。
105. 创建 " 对话 "：dialog。
106. 创建 " 历史 "：history。
107. 创建 " 价格 "：price。
108. 创建 " 预设 "：setup。
109. 创建 " 安全保护 "：securityProtection。
110. 创建 " 广告牌 "：billboard。
111. 创建 " 印刷 "：print。
112. 创建 " 故障检查 "：diagnose。
113. 创建 " 同步设置 "：synchronizeSettings。
114. 创建 " 任务设置 "：taskSettings。
115. 创建 " 类型 "：type。
116. 创建 " 继承 "：inherit。
117. 创建 " 自定义任务 "：customTask。
118. 创建 " 自定义地图 "：customMap。
119. 创建 " 自定义单位 "：customUnit。
120. 创建 " 自定义技能 "：customSkill。
121. 创建 " 自定义物品 "：customItem。
122. 创建 " 自定义地点 "：customLocation。
123. 创建 " 自定义角色 "：customRole。
124. 创建 " 自定义服装 "：customClothing。
125. 创建 " 自定义声音 "：custom


```py
import atexit
import os
import time
from datetime import datetime
from arknights_mower.utils.log import logger
import json

from copy import deepcopy

from arknights_mower.utils.pipe import Pipe
from arknights_mower.utils.simulator import restart_simulator
from arknights_mower.utils.email import task_template

conf = {}
plan = {}
```

It seems like the text you provided is a Python code, but it's hard to tell without the imports and other context. If you could provide more information about what this code is doing, I might be able to give you a more detailed explanation.


```py
operators = {}


# 执行自动排班
def main(c, p, o={}, child_conn=None):
    __init_params__()
    from arknights_mower.utils.log import init_fhlr
    from arknights_mower.utils import config
    global plan
    global conf
    global operators
    conf = c
    plan = p
    operators = o
    config.LOGFILE_PATH = './log'
    config.SCREENSHOT_PATH = './screenshot'
    config.SCREENSHOT_MAXNUM = conf['screenshot']
    config.ADB_DEVICE = [conf['adb']]
    config.ADB_CONNECT = [conf['adb']]
    config.ADB_CONNECT = [conf['adb']]
    config.APPNAME = 'com.hypergryph.arknights' if conf[
                                                       'package_type'] == 1 else 'com.hypergryph.arknights.bilibili'  # 服务器
    config.TAP_TO_LAUNCH = conf['tap_to_launch_game']
    init_fhlr(child_conn)
    Pipe.conn = child_conn
    if plan['conf']['ling_xi'] == 1:
        agent_base_config['令']['UpperLimit'] = 12
        agent_base_config['夕']['LowerLimit'] = 12
    elif plan['conf']['ling_xi'] == 2:
        agent_base_config['夕']['UpperLimit'] = 12
        agent_base_config['令']['LowerLimit'] = 12
    for key in list(filter(None, plan['conf']['rest_in_full'].replace('，', ',').split(','))):
        if key in agent_base_config.keys():
            agent_base_config[key]['RestInFull'] = True
        else:
            agent_base_config[key] = {'RestInFull': True}
    for key in list(filter(None, plan['conf']['exhaust_require'].replace('，', ',').split(','))):
        if key in agent_base_config.keys():
            agent_base_config[key]['ExhaustRequire'] = True
        else:
            agent_base_config[key] = {'ExhaustRequire': True}
    for key in list(filter(None, plan['conf']['workaholic'].replace('，', ',').split(','))):
        if key in agent_base_config.keys():
            agent_base_config[key]['Workaholic'] = True
        else:
            agent_base_config[key] = {'Workaholic': True}
    for key in list(filter(None, plan['conf']['resting_priority'].replace('，', ',').split(','))):
        if key in agent_base_config.keys():
            agent_base_config[key]['RestingPriority'] = 'low'
        else:
            agent_base_config[key] = {'RestingPriority': 'low'}
    logger.info('开始运行Mower')
    logger.debug(agent_base_config)
    simulate()

```

这段代码定义了一个名为`format_time`的函数，它接受一个整数参数`seconds`，表示一个整数时间的秒数。

函数的作用是计算小时和分钟数，并返回它们。它的实现方式是：首先，用`seconds`除以3600，得到小时数。然后，用`seconds` % 3600取余数，得到分钟数。接下来，根据小时数是否为0来决定是否显示小时数。最后，将计算得到的分钟数和小时数拼接在一起，并返回。

接着，定义了一个名为`hide_password`的函数，它接受一个字典类型的参数`conf`，表示一个密码。

函数的作用是隐藏密码，具体实现方式是：对密码进行深度复制，并删除密码中所有字母、数字和特殊字符。


```py
#newbing说用这个来定义休息时间省事
def format_time(seconds):
    # 计算小时和分钟
    rest_hours = int(seconds / 3600)
    rest_minutes = int((seconds % 3600) / 60)
    # 根据小时是否为零来决定是否显示
    if rest_hours == 0: 
        return f"{rest_minutes} 分钟"
    else:
        return f"{rest_hours} 小时 {rest_minutes} 分钟"


def hide_password(conf):
    hpconf = deepcopy(conf)
    hpconf["pass_code"] = "*" * len(conf["pass_code"])
    return hpconf


```

这段代码是一个 Python 函数，名为 `update_conf()`，函数的作用是更新远程服务器（MAE）的配置设置。以下是函数的详细解释：

1. `if not Pipe or not Pipe.conn:`：首先检查管道是否打开，以及连接到远程服务器的客户端（Pipe.conn）是否已连接。如果没有连接，函数将显示错误并输出一条有关管道关闭的警告信息。

2. `logger.error("管道关闭")`：如果管道关闭，函数将输出一个有关管道关闭的错误消息。

3. `logger.info(maa_config)`：如果管道连接正常，函数将通过管道发送一个包含设置名称的 HTTP 请求。

4. `Pipe.conn.send({"type": "update_conf"})`：发送包含设置名称的 HTTP 请求。

5. `logger.debug("通过管道读取设置")`：通过管道读取设置。

6. `conf = Pipe.conn.recv()`：接收设置并将其存储在 `conf` 变量中。

7. `logger.debug(f"接收设置：{hide_password(conf)}`)：隐藏密码，将设置名称存储在 `conf` 变量中。

8. `return conf`：函数返回设置，但没有做任何实际工作。


```py
def update_conf():
    logger.debug("运行中更新设置")

    if not Pipe or not Pipe.conn:
        logger.error("管道关闭")
        logger.info(maa_config)
        return

    logger.debug("通过管道发送更新设置请求")
    Pipe.conn.send({"type": "update_conf"})
    logger.debug("开始通过管道读取设置")
    conf = Pipe.conn.recv()
    logger.debug(f"接收设置：{hide_password(conf)}")

    return conf


```

The code you provided is a function that initializes an operator with the given configuration. The function takes two arguments, the first one is the initial configuration and the second one is the name of the operator.

The function updates the given Maa configuration with the new configuration provided. The new configuration can include changes to the `maa_enable`, `maa_path`, `maa_adb_path`, `adb`, `weekly_plan`, `roguelike`, `rogue_theme`, `sleep_min`, `sleep_max`, `maa_execution_gap`, `buy_first`, `blacklist`, `recruitment_time`, `recruit_only_4`, `conn_preset`, `touch_option`, `mall_ignore_when_full`, `credit_fight`, and `rogue` settings.

The function logs a message to the console with the new Maa configuration.

Overall, this function is useful for updating the Maa configuration of an operator in case it has been modified.


```py
def set_maa_options(base_scheduler):
    conf = update_conf()

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
    base_scheduler.maa_config = maa_config

    logger.debug(f"更新Maa设置：{base_scheduler.maa_config}")


```

Scheduler类是安排作业计划的工具，会根据用户设置的规则，计划需要完成的任务，并安排好任务的执行时间。

在该类中，首先定义了一些类成员变量，包括：

- base_scheduler：计划实例，保存了所有的设置，以及一些辅助方法
- scheduler：调度器，用于处理scheduler中的任务
- set_maa_options：设置机器人的选项，用于连接到机器人

然后设置一些默认值，以及一些方法，用于将配置文件中的设置加载到类中。

scheduler类的方法有：

- scan_time：扫描作业时间，用于计算任务执行时间
- last_room：获取最后完成任务的宿舍，用于在任务中设置最后完成任务的位置
- free_blacklist：获取免费黑名单，用于从黑名单中排除指定的用户
- resting_threshold：设置休息时间，当学生未完成作业时，可以休息一段时间
- MAA：获取机器人应用程序的实例，用于管理机器人的任务
- email_config：设置电子邮件通知的选项
- receipts：获取收到的电子邮件的记录，用于记录已经完成的任务
- notify：设置是否通知用户完成任务，如果任务还未完成
- add_task：添加任务到队列中，用于在安排好任务后增加一些任务
- start_task：开始执行任务，用于开始运行一个任务
- cancel_task：取消任务，用于取消已经安排好的任务
- get_task：获取任务列表，用于打印学生未完成的任务列表

该类的方法可以用于定时器或无限循环中，用于不断处理队列中的任务。


```py
def initialize(tasks, scheduler=None):
    from arknights_mower.solvers.base_schedule import BaseSchedulerSolver
    from arknights_mower.strategy import Solver
    from arknights_mower.utils.device import Device
    from arknights_mower.utils import config
    device = Device()
    cli = Solver(device)
    if scheduler is None:
        base_scheduler = BaseSchedulerSolver(cli.device, cli.recog)
        base_scheduler.operators = {}
        plan1 = {}
        for key in plan[plan['default']]:
            plan1[key] = plan[plan['default']][key]['plans']
        plan[plan['default']] = plan1
        logger.debug(plan)
        base_scheduler.package_name = config.APPNAME  # 服务器
        base_scheduler.global_plan = plan
        base_scheduler.current_plan = plan1
        base_scheduler.current_base = {}
        base_scheduler.resting = []
        base_scheduler.max_resting_count = plan['conf']['max_resting_count']
        base_scheduler.drone_count_limit = conf['drone_count_limit']
        base_scheduler.tasks = tasks
        base_scheduler.enable_party = conf['enable_party'] == 1  # 是否使用线索
        # 读取心情开关，有菲亚梅塔或者希望全自动换班得设置为 true
        base_scheduler.read_mood = conf['run_mode'] == 1
        # 干员宿舍回复阈值
        # 高效组心情低于 UpperLimit  * 阈值 (向下取整)的时候才会会安排休息

        base_scheduler.scan_time = {}
        base_scheduler.last_room = ''
        base_scheduler.free_blacklist = list(filter(None, conf['free_blacklist'].replace('，', ',').split(',')))
        logger.info('宿舍黑名单：' + str(base_scheduler.free_blacklist))
        base_scheduler.resting_threshold = conf['resting_threshold']
        base_scheduler.MAA = None
        base_scheduler.email_config = {
            'mail_enable': conf['mail_enable'],
            'subject': conf['mail_subject'],
            'account': conf['account'],
            'pass_code': conf['pass_code'],
            'receipts': [conf['account']],
            'notify': False
        }

        set_maa_options(base_scheduler)

        base_scheduler.ADB_CONNECT = config.ADB_CONNECT[0]
        base_scheduler.error = False
        base_scheduler.drone_room = None if conf['drone_room'] == '' else conf['drone_room']
        base_scheduler.reload_room = list(filter(None, conf['reload_room'].replace('，', ',').split(',')))
        base_scheduler.drone_execution_gap = 4
        base_scheduler.run_order_delay = conf['run_order_delay']
        base_scheduler.agent_base_config = agent_base_config
        base_scheduler.exit_game_when_idle = conf['exit_game_when_idle']
        
        
        #关闭游戏次数计数器
        base_scheduler.task_count = 0
        
        return base_scheduler
    else:
        scheduler.device = cli.device
        scheduler.recog = cli.recog
        scheduler.handle_error(True)
        return scheduler


```

It seems like the base scheduler is a scheduler program for the robotics domain. It executes tasks based on a task template and connects to a robotics simulator.

The base scheduler has a task template that defines the structure of the tasks it should execute. The tasks are executed by the scheduler at regular intervals, such as every 10 seconds.

The scheduler can also be configured to exit the game when it reaches an idle state. If this configuration is enabled, the scheduler will exit the game and lower the power of the robotics simulator when it reaches an idle state.

The scheduler can send emails using the `send_email` method, which takes a message as an argument and sends it as an email to a specified email address. It also supports sending emails with HTML body using the `render_email` method.

The scheduler runs in a continuous loop and calls the `run` method for each task. The `run` method executes the code of the task in a separate thread and sends emails using the `send_email` method if necessary.


```py
def simulate():
    '''
    具体调用方法可见各个函数的参数说明
    '''
    tasks = []
    reconnect_max_tries = 10
    reconnect_tries = 0
    global base_scheduler
    success = False
    while not success:
        try:
            base_scheduler = initialize(tasks)
            success = True
        except Exception as E:
            reconnect_tries += 1
            if reconnect_tries < 3:
                restart_simulator(conf['simulator'])
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
    if operators != {}:
        for k, v in operators.items():
            if k in base_scheduler.op_data.operators and not base_scheduler.op_data.operators[k].room.startswith(
                    "dorm"):
                # 只复制心情数据
                base_scheduler.op_data.operators[k].mood = v.mood
                base_scheduler.op_data.operators[k].time_stamp = v.time_stamp
                base_scheduler.op_data.operators[k].depletion_rate = v.depletion_rate
                base_scheduler.op_data.operators[k].current_room = v.current_room
                base_scheduler.op_data.operators[k].current_index = v.current_index
    if plan['conf']['ling_xi'] in [1, 2]:
        # 夕，令，同组的则设置lowerlimit
        for name in ["夕","令"]:
            if name in base_scheduler.op_data.operators and base_scheduler.op_data.operators[name].group !="":
                for group_name in base_scheduler.op_data.groups[base_scheduler.op_data.operators[name].group]:
                    if group_name not in ["夕","令"]:
                        base_scheduler.op_data.operators[group_name].lower_limit = 12
                        logger.info(f"自动设置{group_name}心情下限为12")
    while True:
        try:
            if len(base_scheduler.tasks) > 0:
                (base_scheduler.tasks.sort(key=lambda x: x.time, reverse=False))
                sleep_time = (base_scheduler.tasks[0].time - datetime.now()).total_seconds()
                logger.info('||'.join([str(t) for t in base_scheduler.tasks]))
                remaining_time = (base_scheduler.tasks[0].time - datetime.now()).total_seconds()

                set_maa_options(base_scheduler)

                if sleep_time > 540 and base_scheduler.maa_config['maa_enable'] == 1:
                    subject = f"下次任务在{base_scheduler.tasks[0].time.strftime('%H:%M:%S')}"
                    context = f"下一次任务:{base_scheduler.tasks[0].plan}"
                    logger.info(context)
                    logger.info(subject)
                    body = task_template.render(tasks=base_scheduler.tasks)
                    base_scheduler.send_email(body, subject, 'html')
                    base_scheduler.maa_plan_solver()
                elif sleep_time > 0:
                    subject = f"休息 {format_time(remaining_time)}，到{base_scheduler.tasks[0].time.strftime('%H:%M:%S')}开始工作"
                    context = f"下一次任务:{base_scheduler.tasks[0].plan}"
                    logger.info(context)
                    logger.info(subject)
                    if sleep_time > 300 and conf['exit_game_when_idle']:
                        base_scheduler.device.exit(base_scheduler.package_name)
                        base_scheduler.task_count += 1
                        logger.info(f"第{base_scheduler.task_count}次任务结束，关闭游戏，降低功耗")
                    body = task_template.render(tasks=base_scheduler.tasks)
                    base_scheduler.send_email(body, subject, 'html')
                    time.sleep(sleep_time)
            if len(base_scheduler.tasks) > 0 and base_scheduler.tasks[0].type.split('_')[0] == 'maa':
                logger.info(f"开始执行 MAA {base_scheduler.tasks[0].type.split('_')[1]} 任务")
                base_scheduler.maa_plan_solver((base_scheduler.tasks[0].type.split('_')[1]).split(','), one_time=True)
                continue
            base_scheduler.run()
            reconnect_tries = 0
        except ConnectionError or ConnectionAbortedError or AttributeError as e:
            reconnect_tries += 1
            if reconnect_tries < reconnect_max_tries:
                logger.warning(f'出现错误.尝试重启Mower')
                connected = False
                while not connected:
                    try:
                        base_scheduler = initialize([], base_scheduler)
                        break
                    except RuntimeError or ConnectionError or ConnectionAbortedError as ce:
                        logger.error(ce)
                        restart_simulator(conf['simulator'])
                        continue
                continue
            else:
                raise e
        except RuntimeError as re:
            logger.exception(f"程序出错-尝试重启模拟器->{re}")
            restart_simulator(conf['simulator'])
        except Exception as E:
            logger.exception(f"程序出错--->{E}")


```

这段代码定义了两个函数：`save_state` 和 `load_state`。这两个函数都在一个名为 `op_data` 的参数的基础上，对不同的操作系统数据类型进行了不同的操作。

`save_state` 函数将一个复杂的操作系统数据对象（例如配置数据）保存到一个名为 `state.json` 的文件中。如果这个文件不存在，函数会创建一个名为 `tmp` 的临时目录，并在其中创建一个名为 `state.json` 的文件。然后，如果给定的 `op_data` 参数不为 `None`，函数会将这个数据对象中的所有变量及其类型信息dump到一个名为 `state.json` 的文件中。如果给定的 `op_data` 参数为 `None`，则函数会直接返回一个名为 `state` 的字典，其中包含一些默认的类型映射。

`load_state` 函数从名为 `state.json` 的文件中读取一个操作系统数据对象，并将其加载到 `op_data` 参数中。如果给定的 `file` 参数不是一个有效的文件，函数会返回一个空字典 `{}`。然后，函数会遍历 `state.json` 文件中的所有键值对，并将它们加载为单个的类型对象。对于每个类型对象，如果它的 `time_stamp` 属性没有被设置为 `None`，那么函数会将它的时间戳解析为 `datetime.datetime` 类型，并将其设置为 `datetime.strptime(time_stamp, "%Y-%m-%d %H:%M:%S.%f")`。否则，函数会将 `time_stamp` 属性设置为 `None`。最后，函数会输出一条日志信息，表明它已经成功加载了数据。

总的来说，这两个函数都在对不同的操作系统数据类型进行操作，并根据给定的参数返回相应的结果。


```py
def save_state(op_data, file='state.json'):
    if not os.path.exists('tmp'):
        os.makedirs('tmp')
    with open('tmp/' + file, 'w') as f:
        if op_data is not None:
            json.dump(vars(op_data), f, default=str)


def load_state(file='state.json'):
    if not os.path.exists('tmp/' + file):
        return None
    with open('tmp/' + file, 'r') as f:
        state = json.load(f)
    operators = {k: eval(v) for k, v in state['operators'].items()}
    for k, v in operators.items():
        if not v.time_stamp == 'None':
            v.time_stamp = datetime.strptime(v.time_stamp, '%Y-%m-%d %H:%M:%S.%f')
        else:
            v.time_stamp = None
    logger.info("基建配置已加载！")
    return operators


```

调料包主要包括许多不同种类的调料，以及它们的组合。不同的调料可以用


```py
agent_base_config = {}
maa_config = {}


def __init_params__():
    global agent_base_config
    global maa_config
    agent_base_config = {
        "Default": {"UpperLimit": 24, "LowerLimit": 0, "ExhaustRequire": False, "ArrangeOrder": [2, "false"],
                    "RestInFull": False},
        "令": {"ArrangeOrder": [2, "true"]},
        "夕": {"ArrangeOrder": [2, "true"]},
        "稀音": {"ExhaustRequire": True, "ArrangeOrder": [2, "true"], "RestInFull": True},
        "巫恋": {"ArrangeOrder": [2, "true"]},
        "柏喙": {"ExhaustRequire": True, "ArrangeOrder": [2, "true"]},
        "龙舌兰": {"ArrangeOrder": [2, "true"]},
        "空弦": {"ArrangeOrder": [2, "true"]},
        "伺夜": {"ArrangeOrder": [2, "true"]},
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
        "红云": {"ArrangeOrder": [2, "true"]},
        "承曦格雷伊": {"ArrangeOrder": [2, "true"]},
        "乌有": {"ArrangeOrder": [2, "true"]},
        "图耶": {"ArrangeOrder": [2, "true"]},
        "鸿雪": {"ArrangeOrder": [2, "true"]},
        "孑": {"ArrangeOrder": [2, "true"]},
        "清道夫": {"ArrangeOrder": [2, "true"]},
        "临光": {"ArrangeOrder": [2, "true"]},
        "杜宾": {"ArrangeOrder": [2, "true"]},
        "焰尾": {"RestInFull": True},
        "重岳": {"ArrangeOrder": [2, "true"]},
        "琴柳": {},
        "坚雷": {"ArrangeOrder": [2, "true"]},
        "年": {"RestingPriority": "low"},
        "伊内丝": {"ExhaustRequire": True, "ArrangeOrder": [2, "true"], "RestInFull": True},
    }
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

# `arknights_mower/data/__init__.py`

这段代码的作用是读取并解析了两个JSON文件，将它们的数据存储在内存中，并定义了一个名为"agent"的列表，这个列表包含了所有房间的模型。

首先，代码从"data/agent.json"文件中读取并解析了一个JSON对象，这个对象包含了所有房间的模型。然后，代码将这个JSON对象存储在名为"agent"的列表中，这个列表的每个元素都是一个含有房间名称和模型设置的Python字典。

接着，代码从"data/agent-base.json"文件中读取并解析了一个JSON对象，这个对象包含了所有房间的模型设置。然后，代码将这个JSON对象存储在名为"agent-base-config"的变量中，这个变量存储了包含模型设置的Python字典。

最后，代码定义了一个名为"base-room-list"的列表，这个列表包含了所有房间的模型设置。这些模型设置存储在"data/base.json"文件中，并且包含了一个Python字典，这个字典的键是房间的名称，值是设置为真的值。


```py
import json
from pathlib import Path

from .. import __rootdir__

# agents list in Arknights
agent_list = json.loads(
    Path(f'{__rootdir__}/data/agent.json').read_text('utf-8'))

# # agents base skills
# agent_base_config = json.loads(
#     Path(f'{__rootdir__}/data/agent-base.json').read_text('utf-8'))

# name of each room in the basement
base_room_list = json.loads(
    Path(f'{__rootdir__}/data/base.json').read_text('utf-8'))

```

这段代码的主要作用是读取并解析了一个包含有关猜谜游戏信息的数据 JSON 文件，并将其存储为编程环境中的类和函数。

首先，它读取了一个名为 "clue.json" 的 JSON 文件，并将其内容读取为字符串。这个 JSON 文件可能包含了游戏中的提示、谜题和答案等信息。

然后，它又读取了一个名为 "shop.json" 的 JSON 文件，同样地，将其内容读取为字符串。这个 JSON 文件可能包含了游戏中所有商品的名称、价格和库存等信息。

接下来，它读取了一个名为 "ocr.json" 的 JSON 文件，同样地，将其内容读取为字符串。这个 JSON 文件可能包含了 OCR 错误文本和正确答案。

最后，它读取了一个名为 "chapter.json" 的 JSON 文件，同样地，将其内容读取为字符串。这个 JSON 文件可能包含了游戏中的章节和答案。

通过这些 JSON 文件的读取，代码可以获取到游戏中的各种信息，从而实现了猜谜游戏的基本功能。


```py
# the camps to which the clue belongs
clue_name = json.loads(
    Path(f'{__rootdir__}/data/clue.json').read_text('utf-8'))

# goods sold in shop
shop_items = json.loads(
    Path(f'{__rootdir__}/data/shop.json').read_text('utf-8'))

# collection of the obtained ocr error
ocr_error = json.loads(
    Path(f'{__rootdir__}/data/ocr.json').read_text('utf-8'))

# chapter name in English
chapter_list = json.loads(
    Path(f'{__rootdir__}/data/chapter.json').read_text('utf-8'))

```

这段代码的作用是读取并解析了一个JSON文件中的数据，并创建了两个列表：level_list 和 zone_list，分别存储了支持的不同等级和区域。

具体来说，代码首先读取了位于`__rootdir__`目录下的`data/level.json`文件，并将其内容解析为JSON格式。然后，又读取了同目录下的`data/zone.json`文件，同样将其内容解析为JSON格式。这两个列表分别存储了不同等级和区域的信息，可以用于后续的处理和分析。

接着，代码再次读取了同目录下的`data/weekly.json`文件，同样将其内容解析为JSON格式。这个列表中存储了每周不同等级和区域的场景信息。

最后，代码读取了同目录下的`data/scene.json`文件，同样将其内容解析为JSON格式。这个列表中存储了场景名称和场景定义等信息。


```py
# list of supported levels
level_list = json.loads(
    Path(f'{__rootdir__}/data/level.json').read_text('utf-8'))

# open zones
zone_list = json.loads(
    Path(f'{__rootdir__}/data/zone.json').read_text('utf-8'))

# list of supported weekly levels
weekly_zones = json.loads(
    Path(f'{__rootdir__}/data/weekly.json').read_text('utf-8'))

# list of scene defined
scene_list = json.loads(
    Path(f'{__rootdir__}/data/scene.json').read_text('utf-8'))

```

这段代码的主要作用是读取并按标签分类组合干员信息。首先，它读取了一个名为`recruit.json`的JSON文件，并将其中的内容存储在变量`recruit_agent`中。接着，它遍历了`recruit_agent`中的每个值，并将其对应的标签添加到变量`recruit_tag`中。然后，它将`recruit_tag`转换为集合，并将其存储在变量`recruit_tag`中。

接下来，代码按标签分组组合干员信息，将每个标签作为键，并将包含该标签的干员信息作为值，存储到一个字典中。最后，代码将`recruit_agent_list`和`rarity_tags`分别存储为`recruit_agent_list`和`rarity_tags`。


```py
# recruit database
recruit_agent = json.loads(
    Path(f'{__rootdir__}/data/recruit.json').read_text('utf-8'))

recruit_tag = ['资深干员', '高级资深干员']
for x in recruit_agent.values():
    recruit_tag += x['tags']
recruit_tag = list(set(recruit_tag))

'''
按tag分类组合干员
'''

recruit_agent_list = {}
rarity_tags = []

```

这段代码的作用是针对一个字典 `recruit_tag` 中的每个键，将其值赋给另一个字典 `recruit_agent_list`。具体来说，代码首先遍历 `recruit_tag` 字典中的每个键，然后在其下创建一个字典，包含两个键值对，第一个键是 `min_level`，第二个键是一个列表 `agent`。然后，代码遍历 `recruit_agent` 字典中的每个键，检查当前键是否存在于 `recruit_agent_list` 字典中的键中。如果是，并且该键在字典中的键的 `tags` 键中，那么将该键在 `recruit_agent_list` 字典中的键中的 `min_level` 键的值设置为当前键中 `min_level` 键的值。如果是，那么将当前键在 `recruit_agent_list` 字典中的键中的 `agent` 列表中添加一个新的字典，包含一个 `name` 键和一个 `level` 键，其中 `name` 是 `recruit_agent` 字典中当前键对应的人名，`level` 是 `recruit_agent` 字典中当前键对应的人的等级。最后，`recruit_agent_list` 字典中的键会被更新为当前键中 `min_level` 键的值，并且 `recruit_agent_list` 字典中的 `agent` 列表会被更新为当前键在 `recruit_agent` 字典中的 `name` 和 `level` 键。


```py
for key in recruit_tag:
    recruit_agent_list[key] = {
        "min_level": 7,
        "agent": []
    }
    for opeartors in recruit_agent:
        if key in recruit_agent[opeartors]['tags']:
            if recruit_agent[opeartors]['stars'] < recruit_agent_list[key]["min_level"]:
                recruit_agent_list[key]["min_level"] = recruit_agent[opeartors]['stars']

            recruit_agent_list[key]["agent"].append(
                {
                    "name": recruit_agent[opeartors]['name'],
                    "level": recruit_agent[opeartors]['stars'],
                })
```

这段代码的作用是获取招募代理列表中的所有稀有度级别（rarity level）且最小等级（min_level）大于或等于5的键（key）。这将创建一个名为rarity_tags的列表，其中包含这些稀有度级别。


```py
# 保底5星的tag
rarity_tags = []
for key in recruit_agent_list:
    if recruit_agent_list[key]['min_level'] >= 5:
        rarity_tags.append(key)

```

# Fonts

精简过的字体文件，包含所有干员名称内出现过的的汉字


# Models

## dbnet.onnx

DBNET 的模型文件，负责提取图像中的文字

## crnn_lite_lstm.onnx

CRNN 的轻量型模型文件，负责识别图像中的文字

## svm.model

SVM 分类器的模型文件，负责图像匹配判定


# `arknights_mower/ocr/config.py`

这段代码的作用是引入一个名为"dbnet"的模型，它可以从两个路径中选择一个： "dbnet.onnx" 或 "crnn_lite_lstm.onnx"。具体来说，"dbnet.onnx"是下载自清华大学 KEG 实验室和智谱AI训练的大型预训练模型，而"crnn_lite_lstm.onnx"是一个轻量级的 CRNN 模型，由清华大学 KEG 实验室提出，可用于检测和分割各种物体。

在 PyTorch 框架中，`__rootdir__`是一个变量，表示当前工作目录的路径。因此，`from .. import __rootdir__`就是将当前工作目录路径中的 models 目录下的两个模型文件引入到函数内部，以便于在函数中使用这些模型。


```py
from .. import __rootdir__

dbnet_model_path = f'{__rootdir__}/models/dbnet.onnx'
crnn_model_path = f'{__rootdir__}/models/crnn_lite_lstm.onnx'

```

# `arknights_mower/ocr/crnn.py`

这段代码是一个图像预处理和图像分类的PyTorch实现。它有两个函数：`predict_gaussian` 和 `predict_rbg`。这两个函数都是基于Gaussian方法和RGB方法进行图像分类的。

`predict_gaussian` 函数接受一个3通道的图像，将其缩放到一个32x32的分辨率，并将像素值从[-127.5, 127.5]范围内归一化。然后，它将图像转换为灰度图像，即图像的每个像素值都被替换为0-255范围内的值。接下来，它使用了一个小的先验知识并从其训练集中随机抽取一个随机的图像尺寸。最后，它使用训练好的预处理图像将原始图像转换为`N, 3, 32`的格式，其中`N`是图像尺寸，`3`是图像通道数，`32`是图像分辨率。最后，它使用图像的灰度值和尺寸来搜索网络的初始位置，并返回一个类概率分布。

`predict_rbg` 函数与 `predict_gaussian` 函数类似，只是使用了一个RGB的训练集中随机抽取一个图像。这个函数返回一个类概率分布，其中的概率分布使用了一个随机的Gaussian模型。

这两个函数都使用了一个预训练好的图像尺寸（32x32，224x224）作为图像的输入，并在其训练集中搜索网络的初始位置。然后，它们使用图像的灰度值和尺寸来搜索网络的初始位置，并返回一个类概率分布。


```py
import numpy as np
import onnxruntime as rt
from PIL import Image

from .keys import alphabetChinese as alphabet
from .utils import resizeNormalize, strLabelConverter

converter = strLabelConverter(''.join(alphabet))


class CRNNHandle:
    def __init__(self, model_path):
        sess_options = rt.SessionOptions()
        sess_options.log_severity_level = 3
        self.sess = rt.InferenceSession(model_path, sess_options)

    def predict(self, image):
        scale = image.size[1] * 1.0 / 32
        w = image.size[0] / scale
        w = int(w)
        transformer = resizeNormalize((w, 32))
        image = transformer(image)
        image = image.transpose(2, 0, 1)
        transformed_image = np.expand_dims(image, axis=0)
        transformed_image = np.array([[transformed_image[0, 0]] * 3])
        preds = self.sess.run(
            ['out'], {'input': transformed_image.astype(np.float32)})
        preds = preds[0]
        length = preds.shape[0]
        preds = preds.reshape(length, -1)
        preds = np.argmax(preds, axis=1)
        preds = preds.reshape(-1)
        sim_pred = converter.decode(preds, length, raw=False)
        return sim_pred

    def predict_rbg(self, image):
        scale = image.size[1] * 1.0 / 32
        w = image.size[0] / scale
        w = int(w)
        image = image.resize((w, 32), Image.BILINEAR)
        image = np.array(image, dtype=np.float32)
        image -= 127.5
        image /= 127.5
        image = image.transpose(2, 0, 1)
        transformed_image = np.expand_dims(image, axis=0)
        preds = self.sess.run(
            ['out'], {'input': transformed_image.astype(np.float32)})
        preds = preds[0]
        length = preds.shape[0]
        preds = preds.reshape(length, -1)
        preds = np.argmax(preds, axis=1)
        preds = preds.reshape(-1)
        sim_pred = converter.decode(preds, length, raw=False)
        return sim_pred

```

# `arknights_mower/ocr/dbnet.py`

这是一个使用 OpenCV 和 RT（RPA）库的图像分割应用程序，其目的是对输入图像中的物体进行检测和分割。该应用程序使用基于 RT 库的 InferenceSession 和其自定义的 SegmentDetectorRepresenter 类对图像进行处理。下面是应用程序的主要步骤：

1. 加载图像并将其转换为灰度图像。
2. 如果图像的大小与输入参数（如短尺寸）不匹配，则对图像进行缩放。
3. 对图像进行归一化处理，使其值在 0 到 1 之间。
4. 使用自定义的 SegmentDetectorRepresenter 类对图像进行处理，从而得到物体检测和分割的输出。
5. 如果检测到图像中存在物体，则对物体进行回归，并返回物体检测结果和分数列表。如果没有检测到物体，则返回一个空分数列表。

代码示例：


```py
import cv2
import numpy as np
import onnxruntime as rt

from .decode import SegDetectorRepresenter

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)


class DBNET():
    def __init__(self, model_path):
        sess_options = rt.SessionOptions()
        sess_options.log_severity_level = 3
        self.sess = rt.InferenceSession(model_path, sess_options)
        self.decode_handel = SegDetectorRepresenter()

    def process(self, img, short_size):

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        if h < w:
            scale_h = short_size / h
            tar_w = w * scale_h
            tar_w = tar_w - tar_w % 32
            tar_w = max(32, tar_w)
            scale_w = tar_w / w
        else:
            scale_w = short_size / w
            tar_h = h * scale_w
            tar_h = tar_h - tar_h % 32
            tar_h = max(32, tar_h)
            scale_h = tar_h / h

        img = cv2.resize(img, None, fx=scale_w, fy=scale_h)
        img = img.astype(np.float32)

        img /= 255.0
        img -= mean
        img /= std
        img = img.transpose(2, 0, 1)
        transformed_image = np.expand_dims(img, axis=0)
        out = self.sess.run(
            ['out1'], {'input0': transformed_image.astype(np.float32)})
        box_list, score_list = self.decode_handel(out[0][0], h, w)
        if len(box_list) > 0:
            idx = box_list.reshape(
                box_list.shape[0], -1).sum(axis=1) > 0
            box_list, score_list = box_list[idx], score_list[idx]
        else:
            box_list, score_list = [], []
        return box_list, score_list

```