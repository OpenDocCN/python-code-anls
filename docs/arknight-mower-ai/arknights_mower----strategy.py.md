# `arknights-mower\arknights_mower\strategy.py`

```py
# 导入未来版本的注解特性
from __future__ import annotations

# 导入 functools 模块
import functools

# 从当前目录下的 solvers 模块中导入所有内容
from .solvers import *

# 从当前目录下的 solvers 模块中导入 BaseSchedulerSolver 类
from .solvers.base_schedule import BaseSchedulerSolver

# 从当前目录下的 utils 模块中导入 typealias 别名
from .utils import typealias as tp

# 从当前目录下的 utils 模块中导入 Device 类
from .utils.device import Device

# 从当前目录下的 utils 模块中导入 Recognizer 类
from .utils.recognize import Recognizer

# 从当前目录下的 utils 模块中导入 BaseSolver 类
from .utils.solver import BaseSolver


# 定义 Solver 类
class Solver(object):
    """ Integration solver """

    # 初始化方法
    def __init__(self, device: Device = None, recog: Recognizer = None, timeout: int = 99) -> None:
        """
        :param timeout: int, 操作限时，单位为小时
        """
        # 如果 device 参数不为 None，则使用传入的 device，否则使用默认的 Device 对象
        self.device = device if device is not None else Device()
        # 如果 recog 参数不为 None，则使用传入的 recog，否则使用基于 device 的 Recognizer 对象
        self.recog = recog if recog is not None else Recognizer(self.device)
        # 设置操作限时
        self.timeout = timeout

    # 定义 base_scheduler 方法
    def base_scheduler(self, tasks=[], plan={}, current_base={}) -> None:
        # 返回所有排班计划以及当前基建干员位置
        return BaseSchedulerSolver(self.device, self.recog).run(tasks, plan, current_base)

    # 定义 base 方法
    def base(self, arrange: tp.BasePlan = None, clue_collect: bool = False, drone_room: str = None, fia_room: str = None) -> None:
        """
        :param arrange: dict(room_name: [agent,...]), 基建干员安排
        :param clue_collect: bool, 是否收取线索
        :param drone_room: str, 是否使用无人机加速
        :param fia_room: str, 是否使用菲亚梅塔恢复心情
        """
        # 运行基建干员安排
        BaseSolver(self.device, self.recog).run(arrange, clue_collect, drone_room, fia_room)

    # 定义 credit 方法
    def credit(self) -> None:
        # 运行信用商店相关操作
        CreditSolver(self.device, self.recog).run()

    # 定义 mission 方法
    def mission(self) -> None:
        # 运行任务相关操作
        MissionSolver(self.device, self.recog).run()

    # 定义 recruit 方法
    def recruit(self, priority: list[str] = None) -> None:
        """
        :param priority: list[str], 优先考虑的公招干员，默认为高稀有度优先
        """
        # 运行公招相关操作
        RecruitSolver(self.device, self.recog).run(priority)
    # 执行作战操作
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
        # 调用 OpeSolver 类的 run 方法执行作战操作
        return OpeSolver(self.device, self.recog).run(level, times, potion, originite, eliminate, plan)

    # 执行购物操作
    def shop(self, priority: bool = None) -> None:
        """
        :param priority: list[str], 使用信用点购买东西的优先级, 若无指定则默认购买第一件可购买的物品
        """
        # 调用 ShopSolver 类的 run 方法执行购物操作
        ShopSolver(self.device, self.recog).run(priority)

    # 执行邮件操作
    def mail(self) -> None:
        # 调用 MailSolver 类的 run 方法执行邮件操作
        MailSolver(self.device, self.recog).run()

    # 返回到主界面操作
    def index(self) -> None:
        # 调用 BaseSolver 类的 back_to_index 方法返回到主界面
        BaseSolver(self.device, self.recog).back_to_index()
```