# ArknightMower源码解析 6

# `/opt/arknights-mower/arknights_mower/solvers/credit.py`

CreditSolver 是信元转译模式中的一个 solver，用于通过线索交换自动收集信用。CreditSolver 中使用了一个自定义的 Run 方法，在每次运行开始时会调用一次。在 Run 方法的体内，通过调用 super().run() 来运行 solver 的核心逻辑。

具体来说，在 transition 方法中，根据当前场景来判断是否进行过渡，并调用对应的方法。在 transition 方法中，如果场景是 Scene.INDEX，则调用 transition_request 方法，否则调用 transition_save 方法。在 transition_request 和 transition_save 方法中，会根据不同的场景进行相应的操作，例如在 Scene.FRIEND_LIST_OFF 场景中，调用 down 键的 tap 方法，在 Scene.FRIEND_LIST_ON 场景中，调用 friend_list 键的 tap 方法。

CreditSolver 通过循环调用 transition 方法来不断进行场景切换，并在场景结束时调用 back_to_index 方法返回索引。


```
from ..utils import detector
from ..utils.device import Device
from ..utils.log import logger
from ..utils.recognize import RecognizeError, Recognizer, Scene
from ..utils.solver import BaseSolver


class CreditSolver(BaseSolver):
    """
    通过线索交换自动收集信用
    """

    def __init__(self, device: Device = None, recog: Recognizer = None) -> None:
        super().__init__(device, recog)

    def run(self) -> None:
        logger.info('Start: 信用')
        super().run()

    def transition(self) -> bool:
        if self.scene() == Scene.INDEX:
            self.tap_element('index_friend')
        elif self.scene() == Scene.FRIEND_LIST_OFF:
            self.tap_element('friend_list')
        elif self.scene() == Scene.FRIEND_LIST_ON:
            down = self.find('friend_list_on', strict=True)[1][1]
            scope = [(0, 0), (100000, down)]
            if not self.tap_element('friend_visit', scope=scope, detected=True):
                self.sleep(1)
        elif self.scene() == Scene.FRIEND_VISITING:
            visit_limit = self.find('visit_limit')
            if visit_limit is not None:
                return True
            visit_next = detector.visit_next(self.recog.img)
            if visit_next is not None:
                self.tap(visit_next)
            else:
                return True
        elif self.scene() == Scene.LOADING:
            self.sleep(3)
        elif self.scene() == Scene.CONNECTING:
            self.sleep(3)
        elif self.get_navigation():
            self.tap_element('nav_social')
        elif self.scene() != Scene.UNKNOWN:
            self.back_to_index()
        else:
            raise RecognizeError('Unknown scene')

```

# `/opt/arknights-mower/arknights_mower/solvers/mail.py`

这段代码是一个名为 `MailSolver` 的类，它继承自 `BaseSolver` 类。这个类的目的是完成一个邮件的收取，包括收取邮件中的信息并将其显示给用户。

具体来说，这段代码做了以下几件事情：

1. 在 `__init__` 方法中，初始化了 `device` 和 `recog` 变量。设备和识别器是用来获取信息和进行交互的工具。

2. 在 `run` 方法中，处理了系统运行时的情况。在这里，代码会向用户收取一封邮件。如果已经收取了邮件，就显示给用户。

3. 在 `transition` 方法中，判断了当前场景。如果场景是 `Scene.INDEX`，那么发送一个收取邮件的指令。如果场景是 `Scene.MAIL`，那么处理收取邮件的情况。

4. 在 `__get_navigation` 方法中，如果当前场景是 `Scene.INDEX`，那么使用 `find_path` 方法找到从当前位置到 `Scene.MAIL` 路径上的元素，并使用 `tap` 方法到达该元素。如果当前场景是 `Scene.MAIL`，那么在已经获取到邮件的情况下，再次使用 `tap` 方法获取邮件中的信息。

5. 在 `__call__` 方法中，初始化邮件收取功能，并调用 `run` 方法开始收取邮件。


```
from ..utils.device import Device
from ..utils.log import logger
from ..utils.recognize import RecognizeError, Recognizer, Scene
from ..utils.solver import BaseSolver


class MailSolver(BaseSolver):
    """
    收取邮件
    """

    def __init__(self, device: Device = None, recog: Recognizer = None) -> None:
        super().__init__(device, recog)

    def run(self) -> None:
        # if it touched
        self.touched = False

        logger.info('Start: 邮件')
        super().run()

    def transition(self) -> bool:
        if self.scene() == Scene.INDEX:
            scope = ((0, 0), (100+self.recog.w//4, self.recog.h//10))
            nav = self.find('index_nav', thres=250, scope=scope)
            self.tap(nav, 0.625)
        elif self.scene() == Scene.MAIL:
            if self.touched:
                return True
            self.touched = True
            self.tap_element('read_mail')
        elif self.scene() == Scene.LOADING:
            self.sleep(3)
        elif self.scene() == Scene.CONNECTING:
            self.sleep(3)
        elif self.scene() == Scene.MATERIEL:
            self.tap_element('materiel_ico')
        elif self.get_navigation():
            self.tap_element('nav_index')
        elif self.scene() != Scene.UNKNOWN:
            self.back_to_index()
        else:
            raise RecognizeError('Unknown scene')

```

# `/opt/arknights-mower/arknights_mower/solvers/mission.py`



This is a class definition for an X games application. It defines the behavior of an X game in the mission mode.

The class inherits from the `游戏框架`类，并覆盖了以下类：

* `游戏框架`
* `场景`
* `索引场景`
* `主场景`
* `聊天场景`
* `主菜单场景`
* `次场景`
* `任务场景`
* `合成场景`
* `探索场景`
* `编辑场景`
* `音乐场景`
* `动画场景`

The class implements the following methods：

* `游戏框架`
	+ `__init__(self, *args, **kwargs)`
	+ `游戏初始化`
	+ `游戏循环`
	+ `游戏结束`
* `场景`
	+ `__init__(self)`
	+ `切换场景`
	+ `进入场景`
	+ `游戏设置`
	+ `元素查找`
	+ `背景图片`
	+ `切换场景图标`
	+ `提示信息`
	+ `获取焦点`
	+ `返回当前场景`
* `索引场景`
	+ `游戏索引`
	+ `返回场景`
	+ `进入场景`
	+ `游戏设置`
	+ `元素查找`
	+ `背景图片`
	+ `切换场景图标`
	+ `提示信息`
	+ `获取焦点`
	+ `返回当前场景`
* `主场景`
	+ `游戏初始化`
	+ `游戏设置`
	+ `进入场景`
	+ `游戏循环`
	+ `游戏结束`
* `聊天场景`
	+ `游戏初始化`
	+ `进入场景`
	+ `发送消息`
	+ `接收消息`
	+ `消息列表`
* `主菜单场景`
	+ `游戏初始化`
	+ `进入场景`
	+ `游戏设置`
	+ `元素查找`
	+ `背景图片`
	+ `切换场景图标`
	+ `提示信息`
	+ `获取焦点`
	+ `返回当前场景`
* `次场景`
	+ `游戏初始化`
	+ `进入场景`
	+ `游戏设置`
	+ `元素查找`
	+ `背景图片`
	+ `切换场景图标`
	+ `提示信息`
	+ `获取焦点`
	+ `返回当前场景`
* `任务场景`
	+ `游戏初始化`
	+ `进入场景`
	+ `游戏设置`
	+ `元素查找`
	+ `背景图片`
	+ `切换场景图标`
	+ `提示信息`
	+ `获取焦点`
	+ `返回当前场景`
* `合成场景`
	+ `游戏初始化`
	+ `进入场景`
	+ `游戏设置`
	+ `元素查找`
	+ `背景图片`
	+ `切换场景图标`
	+ `提示信息`
	+ `获取焦点`
	+ `返回当前场景`
* `探索场景`
	+ `游戏初始化`
	+ `进入场景`
	+ `游戏设置`
	+ `元素查找`
	+ `背景图片`
	+ `切换场景图标`
	+ `提示信息`
	+ `获取焦点`
	+ `返回当前场景`
* `编辑场景`
	+ `游戏初始化`
	+ `进入场景`
	+ `游戏设置`
	+ `元素查找`
	+ `背景图片`
	+ `切换场景图标`
	+ `提示信息`
	+ `获取焦点`
	+ `返回当前场景`
* `音乐场景`
	+ `游戏初始化`
	+ `进入场景`
	+ `音乐播放`
	+ `音量控制`
	+ `背景图片`
	+ `切换场景图标`
	+ `提示信息`
	+ `获取焦点`
	+ `返回当前场景`
* `动画场景`
	+ `游戏初始化`
	+ `进入场景`
	+ `动画播放`
	+ `动画控制`
	+ `背景图片`
	+ `切换场景图标`
	+ `提示信息`
	+ `获取焦点`
	+ `返回当前场景`

This class also contains methods for other game scenes, including some scenes that are specific to the mission mode.

Additionally, this class contains methods for integrating the game with the X Python library and the `游戏` module, as well as initializing the game with arguments and keyword arguments.


```
from ..utils.device import Device
from ..utils.log import logger
from ..utils.recognize import RecognizeError, Recognizer, Scene
from ..utils.solver import BaseSolver


class MissionSolver(BaseSolver):
    """
    点击确认完成每日任务和每周任务
    """

    def __init__(self, device: Device = None, recog: Recognizer = None) -> None:
        super().__init__(device, recog)

    def run(self) -> None:
        # status of mission completion, 1: daily, 2: weekly
        self.checked = 0

        logger.info('Start: 任务')
        super().run()

    def transition(self) -> bool:
        if self.scene() == Scene.INDEX:
            self.tap_element('index_mission')
        elif self.scene() == Scene.MISSION_TRAINEE:
            if self.checked & 1 == 0:
                self.tap_element('mission_daily')
            elif self.checked & 2 == 0:
                self.tap_element('mission_weekly')
            else:
                return True
        elif self.scene() == Scene.MISSION_DAILY:
            self.checked |= 1
            collect = self.find('mission_collect')
            if collect is None:
                self.sleep(1)
                collect = self.find('mission_collect')
            if collect is not None:
                logger.info('任务：一键收取任务')
                self.tap(collect)
            elif self.checked & 2 == 0:
                self.tap_element('mission_weekly')
            else:
                return True
        elif self.scene() == Scene.MISSION_WEEKLY:
            self.checked |= 2
            collect = self.find('mission_collect')
            if collect is None:
                self.sleep(1)
                collect = self.find('mission_collect')
            if collect is not None:
                logger.info('任务：一键收取任务')
                self.tap(collect)
            elif self.checked & 1 == 0:
                self.tap_element('mission_daily')
            else:
                return True
        elif self.scene() == Scene.MATERIEL:
            self.tap_element('materiel_ico')
        elif self.scene() == Scene.LOADING:
            self.sleep(3)
        elif self.scene() == Scene.CONNECTING:
            self.sleep(3)
        elif self.get_navigation():
            self.tap_element('nav_mission')
        elif self.scene() != Scene.UNKNOWN:
            self.back_to_index()
        else:
            raise RecognizeError('Unknown scene')

```

# `/opt/arknights-mower/arknights_mower/solvers/operation.py`

这段代码是一个 Python 程序，从未来时代的一个包中导入了一些注解（from __future__ import annotations）。

它从另一个包中导入了一些时间（import time）、从堆栈中打印出 traceback（import traceback）、从另一个包中导入了一些周期的章列表（from ..data import chapter_list, level_list, weekly_zones, zone_list）、从另一个包中导入了一些 OCR处理的实例（from ..ocr import ocrhandle）。

它从另一个包中导入了一些配置（from ..utils import config），一些用于显示错误信息的日志（from ..utils import logger），一些用于图像分割的库（from ..utils.image import scope2slice），一些用于求解问题（from ..utils.recognize import RecognizeError, Scene），一些用于求解策略的类（from ..utils.solver import BaseSolver, StrategyError）以及一个名为 BOTTOM_TAP_NUMER 的变量。

最后，这个程序似乎定义了一些常量（BOTTOM_TAP_NUMER）。


```
from __future__ import annotations

import time
import traceback

from ..data import chapter_list, level_list, weekly_zones, zone_list
from ..ocr import ocrhandle
from ..utils import config
from ..utils import typealias as tp
from ..utils.image import scope2slice
from ..utils.log import logger
from ..utils.recognize import RecognizeError, Scene
from ..utils.solver import BaseSolver, StrategyError

BOTTOM_TAP_NUMER = 8


```

This is a Python implementation of the OCR process, where the OCR handle is used to recognize the text in an image. The OCR handle uses the Tw超参数 with a learning rate of 0.001 and an initialization factor of 0.5.

The input image is preprocessed by resizing it to a smaller size and normalizing the pixel values. Then, a waterfall deconstruction is applied to the image, which is divided into four sub-images and processed on each sub-image individually.

The output image is then processed by resizing it to a larger size and converting it to a binary format. If the output image is not the same as the expected image, an error is raised.

The OCR process is repeated until the expected image is obtained or the Tw test indicates that the image is not an expected image.


```
class LevelUnopenError(Exception):
    pass


class OpeSolver(BaseSolver):
    """
    自动作战策略
    """

    def __init__(self, device=None, recog=None):
        super().__init__(device, recog)

    def run(self, level: str = None, times: int = -1, potion: int = 0, originite: int = 0, eliminate: int = 0, plan: list = None):
        """
        :param level: str, 指定关卡，默认为前往上一次关卡或当前界面关卡
        :param times: int, 作战的次数上限，-1 为无限制，默认为 -1
        :param potion: int, 使用药剂恢复体力的次数上限，-1 为无限制，默认为 0
        :param originite: int, 使用源石恢复体力的次数上限，-1 为无限制，默认为 0
        :param eliminate: int, 是否优先处理未完成的每周剿灭，0 为忽略剿灭，1 为优先剿灭，2 为优先剿灭但只消耗代理卡，默认为 0
        :param plan: [[str, int]...], 指定多个关卡以及次数，优先级高于 level

        :return remain_plan: [[str, int]...], 未完成的计划
        """
        if level is not None and plan is not None:
            logger.error('不可同时指定 level 和 plan')
            return
        if plan is not None:
            for x in plan:
                if x[0] != 'pre_ope' and (x[0] not in level_list.keys() or level_list[x[0]]['ap_cost'] == 0):
                    logger.error(f'不支持关卡 {x[0]}，请重新指定')
                    return
        if level is not None:
            if level not in level_list.keys() or level_list[level]['ap_cost'] == 0:
                logger.error(f'不支持关卡 {level}，请重新指定')
                return
            plan = [[level, times]]
        if plan is None:
            plan = [['pre_ope', times]]  # 上一次作战关卡

        self.level = level
        self.times = times
        self.potion = potion
        self.originite = originite
        self.eliminate = eliminate
        self.plan = plan

        self.recover_state = 0  # 有关体力恢复的状态，0 为未知，1 为体力药剂恢复中，2 为源石恢复中（防止网络波动）
        self.eliminate_state = 0  # 有关每周剿灭的状态，0 为未知，1 为未完成，2 为已完成，3 为未完成但无代理卡可用
        self.wait_pre = 10  # 作战时每次等待的时长，普通关卡为 10s，剿灭关卡为 60s
        self.wait_start = 0  # 作战时第一次等待的时长
        self.wait_total = 0  # 作战时累计等待的时长
        self.level_choosed = plan[0][0] == 'pre_ope'  # 是否已经选定关卡
        self.unopen = []  # 未开放的关卡
        self.failed = False  # 作战代理是否正常运作

        logger.info('Start: 作战')
        logger.debug(f'plan: {plan}')
        super().run()
        return self.plan + self.unopen

    def switch_plan(self) -> None:
        self.plan = self.plan[1:]
        self.wait_start = 0
        self.level_choosed = False

    def transition(self) -> bool:
        # 选择剩余次数不为 0 的任务
        while len(self.plan) > 0 and self.plan[0][1] == 0:
            self.switch_plan()
        # 如果任务列表为空则退出
        if len(self.plan) == 0:
            return True

        if self.scene() == Scene.INDEX:
            self.tap_element('index_terminal')
        elif self.scene() == Scene.TERMINAL_MAIN:
            return self.terminal_main()
        elif self.scene() == Scene.OPERATOR_BEFORE:
            return self.operator_before()
        elif self.scene() == Scene.OPERATOR_ELIMINATE:
            return self.operator_before_elimi()
        elif self.scene() == Scene.OPERATOR_ELIMINATE_AGENCY:
            self.tap_element('ope_elimi_agency_confirm')
        elif self.scene() == Scene.OPERATOR_SELECT:
            self.tap_element('ope_select_start')
        elif self.scene() == Scene.OPERATOR_ONGOING:
            self.ope_ongoing()
        elif self.scene() == Scene.OPERATOR_FINISH:
            self.ope_finish()
        elif self.scene() == Scene.OPERATOR_ELIMINATE_FINISH:
            self.ope_finish_elimi()
        elif self.scene() == Scene.OPERATOR_GIVEUP:  # TODO 得找个稳定复现代理三星变两星的地图
            logger.error('代理出现失误')
            return True
        elif self.scene() == Scene.OPERATOR_FAILED:
            logger.error('代理出现失误')
            self.failed = True
            self.tap((self.recog.w // 2, 10))
        elif self.scene() == Scene.OPERATOR_RECOVER_POTION:
            return self.recover_potion()
        elif self.scene() == Scene.OPERATOR_RECOVER_ORIGINITE:
            return self.recover_originite()
        elif self.scene() == Scene.LOADING:
            self.sleep(3)
        elif self.scene() == Scene.CONNECTING:
            self.sleep(3)
        elif self.scene() == Scene.UPGRADE:
            self.tap_element('upgrade')
        elif self.scene() == Scene.OPERATOR_DROP:
            self.tap_element('nav_button', 0.2)
        elif self.get_navigation():
            self.tap_element('nav_terminal')
        elif self.scene() != Scene.UNKNOWN:
            self.back_to_index()
        else:
            raise RecognizeError('Unknown scene')

    def terminal_main(self) -> bool:
        if self.eliminate_state != 3:
            eliminate_todo = self.find('terminal_eliminate')
            # 检查每周剿灭完成情况
            if eliminate_todo is not None:
                self.eliminate_state = 1
            else:
                self.eliminate_state = 2
            # 如果每周剿灭未完成且设定为优先处理
            if self.eliminate and eliminate_todo is not None:
                self.tap(eliminate_todo)
                return
        try:
            # 选择关卡
            self.choose_level(self.plan[0][0])
        except LevelUnopenError:
            logger.error(f'关卡 {self.plan[0][0]} 未开放，请重新指定')
            self.unopen.append(self.plan[0])
            self.switch_plan()
            return
        self.level_choosed = True

    def operator_before(self) -> bool:
        # 关卡未选定，退回到终端主界面选择关卡
        if not self.level_choosed:
            self.get_navigation()
            self.tap_element('nav_terminal')
            return
        # 代理出现过失误，终止作战
        if self.failed:
            return True
        # 激活代理作战
        agency = self.find('ope_agency')
        if agency is not None:
            self.tap(agency)
            return
        # 重置普通关卡等待时长
        if self.wait_pre != 10:
            self.wait_start = 0
            self.wait_pre = 10
        self.wait_total = 0
        # 点击开始作战
        # ope_start_SN 对应三周年活动愚人船的 UI
        ope_start = self.find('ope_start', judge=False) or self.find('ope_start_SN', judge=False)
        if ope_start:
            self.tap(ope_start)
            # 确定可以开始作战后扣除相应的消耗药剂或者源石
            if self.recover_state == 1:
                logger.info('use potion to recover sanity')
                self.potion -= 1
            elif self.recover_state == 2:
                logger.info('use originite to recover sanity')
                self.originite -= 1
            self.recover_state = 0
        else:
            # 与预期不符，等待一阵并重新截图
            self.sleep(1)

    def operator_before_elimi(self) -> bool:
        # 如果每周剿灭完成情况未知，退回到终端主界面选择关卡
        if self.eliminate_state == 0:
            self.get_navigation()
            self.tap_element('nav_terminal')
            return
        # 如果每周剿灭已完成但仍然在剿灭关卡前，则只可能是 pre_ope 为剿灭关卡，此时应该退出
        if self.eliminate_state == 2:
            logger.warning('检测到关卡为剿灭，但每周剿灭任务已完成')
            return True
        # 如果剿灭代理卡已经用完但仍然在剿灭关卡前，则只可能是 pre_ope 为剿灭关卡，此时应该退出
        if self.eliminate_state == 3:
            logger.warning('检测到关卡为剿灭，但剿灭代理卡已经用完')
            return True
        # 代理出现过失误，终止作战
        if self.failed:
            return True
        # 激活代理作战
        agency = self.find('ope_elimi_agency')
        if agency is not None:
            self.tap(agency)
            return
        agency = self.find('ope_agency')
        if agency is not None:
            self.tap(agency)
            return
        # 若只想用代理卡，但此时代理卡已经用光，则退回到终端主界面选择关卡
        if self.eliminate == 2 and self.find('ope_elimi_agenct_used') is None:
            self.eliminate_state = 3
            self.get_navigation()
            self.tap_element('nav_terminal')
            return
        # 重置剿灭关卡等待时长
        if self.wait_pre != 60:
            self.wait_start = 0
            self.wait_pre = 60
        self.wait_total = 0
        # 点击开始作战
        self.tap_element('ope_start')
        # 确定可以开始作战后扣除相应的消耗药剂或者源石
        if self.recover_state == 1:
            logger.info('use potion to recover sanity')
            self.potion -= 1
        elif self.recover_state == 2:
            logger.info('use originite to recover sanity')
            self.originite -= 1
        self.recover_state = 0

    def ope_ongoing(self) -> None:
        if self.wait_total < self.wait_start:
            if self.wait_total == 0:
                logger.info(f'等待 {self.wait_start} 秒')
            self.wait_total += self.wait_pre
            if self.wait_total == self.wait_start:
                self.sleep(self.wait_pre)
            else:
                time.sleep(self.wait_pre)
        else:
            logger.info(f'等待 {self.wait_pre} 秒')
            self.wait_total += self.wait_pre
            self.sleep(self.wait_pre)

    def ope_finish(self) -> None:
        # 更新 wait_start
        if self.wait_total > 0:
            if self.wait_start == 0:
                self.wait_start = self.wait_total - self.wait_pre
            else:
                self.wait_start = min(
                    self.wait_start + self.wait_pre, self.wait_total - self.wait_pre)
            self.wait_total = 0
        # 如果关卡选定则扣除任务次数
        if self.level_choosed:
            self.plan[0][1] -= 1
        # 若每周剿灭未完成则剿灭完成状态变为未知
        if self.eliminate_state == 1:
            self.eliminate_state = 0
        # 随便点击某处退出结算界面
        self.tap((self.recog.w // 2, 10))

    def ope_finish_elimi(self) -> None:
        # 每周剿灭完成情况变为未知
        self.eliminate_state = 0
        # 随便点击某处退出结算界面
        self.tap((self.recog.w // 2, 10))

    def recover_potion(self) -> bool:
        if self.potion == 0:
            if self.originite != 0:
                # 转而去使用源石恢复
                self.tap_element('ope_recover_originite')
                return
            # 关闭恢复界面
            self.back()
            return True
        elif self.recover_state:
            # 正在恢复中，防止网络波动
            self.sleep(3)
        else:
            # 选择药剂恢复体力
            if self.find('ope_recover_potion_empty') is not None:
                # 使用次数未归零但已经没有药剂可以恢复体力了
                logger.info('The potions have been used up.')
                self.potion = 0
                return
            self.tap_element('ope_recover_potion_choose', 0.9, 0.75, judge=False)
            # 修改状态
            self.recover_state = 1

    def recover_originite(self) -> bool:
        if self.originite == 0:
            if self.potion != 0:
                # 转而去使用药剂恢复
                self.tap_element('ope_recover_potion')
                return
            # 关闭恢复界面
            self.back()
            return True
        elif self.recover_state:
            # 正在恢复中，防止网络波动
            self.sleep(3)
        else:
            # 选择源石恢复体力
            if self.find('ope_recover_originite_empty') is not None:
                # 使用次数未归零但已经没有源石可以恢复体力了
                logger.info('The originites have been used up.')
                self.originite = 0
                return
            self.tap_element('ope_recover_originite_choose', 0.9, 0.85, judge=False)
            # 修改状态
            self.recover_state = 2

    def ocr_level(self) -> list:
        ocr = ocrhandle.predict(self.recog.img)
        ocr = list(filter(lambda x: x[1] in level_list.keys(), ocr))
        levels = sorted([x[1] for x in ocr])
        return ocr, levels

    def choose_level(self, level: str) -> None:
        """ 在终端主界面选择关卡 """
        if level == 'pre_ope':
            logger.info(f'前往上一次关卡')
            self.tap_element('terminal_pre')
            return

        zone_name = level_list[level]['zone_id']
        zone = zone_list[zone_name]
        logger.info(f'关卡：{level}')
        logger.info(f'章节：{zone["name"]}')

        # 识别导航栏，辅助识别章节
        scope = self.recog.nav_button()
        scope[1][1] = self.recog.h

        # 选择章节/区域
        if zone['type'] == 'MAINLINE':
            self.switch_bottom(1)
            self.choose_zone_theme(zone, scope)
        elif zone['type'] == 'BRANCHLINE':
            self.switch_bottom(2)
            self.choose_zone_supple(zone, scope)
        elif zone['type'] == 'SIDESTORY':
            self.switch_bottom(3)
            self.choose_zone_supple(zone, scope)
        elif zone['type'] == 'WEEKLY':
            self.switch_bottom(4)
            self.choose_zone_resource(zone)
        else:
            raise RecognizeError('Unknown zone')

        # 关卡选择核心逻辑
        ocr, levels = self.ocr_level()

        # 先向左滑动
        retry_times = 3
        while level not in levels:
            _levels = levels
            self.swipe_noinertia((self.recog.w // 2, self.recog.h // 4),
                                 (self.recog.w // 3, 0), 100)
            ocr, levels = self.ocr_level()
            if _levels == levels:
                retry_times -= 1
                if retry_times == 0:
                    break
            else:
                retry_times = 3

        # 再向右滑动
        retry_times = 3
        while level not in levels:
            _levels = levels
            self.swipe_noinertia((self.recog.w // 2, self.recog.h // 4),
                                 (-self.recog.w // 3, 0), 100)
            ocr, levels = self.ocr_level()
            if _levels == levels:
                retry_times -= 1
                if retry_times == 0:
                    break
            else:
                retry_times = 3

        # 如果正常运行则此时关卡已经出现在界面中
        for x in ocr:
            if x[1] == level:
                self.tap(x[2])
                return
        raise RecognizeError('Level recognition error')

    def switch_bottom(self, id: int) -> None:
        id = id * 2 + 1
        bottom = self.recog.h - 10
        self.tap((self.recog.w//BOTTOM_TAP_NUMER//2*id, bottom))

    def choose_zone_theme(self, zone: list, scope: tp.Scope) -> None:
        """ 识别主题曲区域 """
        # 定位 Chapter 编号
        ocr = []
        act_id = 999
        while act_id != zone['chapterIndex']:
            _act_id = act_id
            act_id = -1
            for x in ocr:
                if zone['chapterIndex'] < _act_id:
                    if x[1].upper().replace(' ', '') == chapter_list[_act_id-1].replace(' ', ''):
                        self.tap(x[2])
                        break
                else:
                    if x[1].upper().replace(' ', '') == chapter_list[_act_id+1].replace(' ', ''):
                        self.tap(x[2])
                        break
            ocr = ocrhandle.predict(self.recog.img[scope2slice(scope)])
            for x in ocr:
                if x[1][:7].upper() == 'EPISODE' and len(x[1]) == 9:
                    try:
                        episode = int(x[1][-2:])
                        act_id = zone_list[f'main_{episode}']['chapterIndex']
                        break
                    except Exception:
                        raise RecognizeError('Unknown episode')
            if act_id == -1 or _act_id == act_id:
                raise RecognizeError('Unknown error')

        # 定位 Episode 编号
        cover = self.find(f'main_{episode}')
        while zone['zoneIndex'] < episode:
            self.swipe_noinertia((cover[0][0], cover[0][1]),
                                 (cover[1][0] - cover[0][0], 0))
            episode -= 1
        while episode < zone['zoneIndex']:
            self.swipe_noinertia((cover[1][0], cover[0][1]),
                                 (cover[0][0] - cover[1][0], 0))
            episode += 1
        self.tap(cover)

    def choose_zone_supple(self, zone: list, scope: tp.Scope) -> None:
        """ 识别别传/插曲区域 """
        try_times = 5
        zoneIndex = {}
        for x in zone_list.values():
            zoneIndex[x['name'].replace('·', '')] = x['zoneIndex']
        while try_times:
            try_times -= 1
            ocr = ocrhandle.predict(self.recog.img[scope2slice(scope)])
            zones = set()
            for x in ocr:
                if x[1] in zoneIndex.keys():
                    zones.add(zoneIndex[x[1]])
            logger.debug(zones)
            if zone['zoneIndex'] in zones:
                for x in ocr:
                    if x[1] == zone['name'].replace('·', ''):
                        self.tap(x[2])
                        self.tap_element('enter')
                        return
                raise RecognizeError
            else:
                st, ed = None, None
                for x in ocr:
                    if x[1] in zoneIndex.keys() and zoneIndex[x[1]] == min(zones):
                        ed = x[2][0]
                    elif x[1] in zoneIndex.keys() and zoneIndex[x[1]] == max(zones):
                        st = x[2][0]
                logger.debug((st, ed))
                self.swipe_noinertia(st, (0, ed[1]-st[1]))

    def choose_zone_resource(self, zone: list) -> None:
        """ 识别资源收集区域 """
        ocr = ocrhandle.predict(self.recog.img)
        unable = list(filter(lambda x: x[1] in ['不可进入', '本日16:00开启'], ocr))
        ocr = list(filter(lambda x: x[1] in weekly_zones, ocr))
        weekly = sorted([x[1] for x in ocr])
        while zone['name'] not in weekly:
            _weekly = weekly
            self.swipe((self.recog.w // 4, self.recog.h // 4),
                       (self.recog.w // 16, 0))
            ocr = ocrhandle.predict(self.recog.img)
            unable = list(filter(lambda x: x[1] in ['不可进入', '本日16:00开启'], ocr))
            ocr = list(filter(lambda x: x[1] in weekly_zones, ocr))
            weekly = sorted([x[1] for x in ocr])
            if _weekly == weekly:
                break
        while zone['name'] not in weekly:
            _weekly = weekly
            self.swipe((self.recog.w // 4, self.recog.h // 4),
                       (-self.recog.w // 16, 0))
            ocr = ocrhandle.predict(self.recog.img)
            unable = list(filter(lambda x: x[1] in ['不可进入', '本日16:00开启'], ocr))
            ocr = list(filter(lambda x: x[1] in weekly_zones, ocr))
            weekly = sorted([x[1] for x in ocr])
            if _weekly == weekly:
                break
        if zone['name'] not in weekly:
            raise RecognizeError('Not as expected')
        for x in ocr:
            if x[1] == zone['name']:
                for item in unable:
                    if x[2][0][0] < item[2][0][0] < x[2][1][0]:
                        raise LevelUnopenError
                self.tap(x[2])
                ocr = ocrhandle.predict(self.recog.img)
                unable = list(filter(lambda x: x[1] == '关卡尚未开放', ocr))
                if len(unable):
                    raise LevelUnopenError
                break

```

# `/opt/arknights-mower/arknights_mower/solvers/record.py`

这是一个相对复杂的函数，因为它需要使用 SQLite 数据库来存储数据。在这个函数中，我们使用 SQLite3 库连接到一个数据库，并创建了一个 'agent_action' 表格来存储干员的行动。

首先，我们创建了一个 'agent\_action' 表格，并在其中插入了一些列，包括 'name'、'agent\_current\_room'、'current\_room'、'is\_high' 和 'agent\_group'。然后，我们在函数中使用了这些列来查询数据库中的数据。

接下来，我们需要检查给定的干员是否具有高优先级。为此，我们执行一个查询，该查询将在数据库中查找与给定干员相同的房间并且是高优先级的干员。

最后，我们将更新过的数据库记录保存到 SQLite 数据库中。如果更新时间(默认为 None)为真，则我们将更新记录保存到数据库中。

整个函数的逻辑非常复杂，因为它需要使用多个模块(包括 SQLite 数据库和其他辅助函数)来完成。此外，这个函数可能会在较复杂的系统中作为关键组件出现，因此需要进行充分的测试和调试以确保其可靠性。


```
# 用于记录Mower操作行为
import sqlite3
import os
from collections import defaultdict

from arknights_mower.utils.log import logger
from datetime import datetime, timezone


# 记录干员进出站以及心情数据，将记录信息存入agent_action表里
def save_action_to_sqlite_decorator(func):
    def wrapper(self, name, mood, current_room, current_index, update_time=False):
        agent = self.operators[name]  # 干员

        agent_current_room = agent.current_room  # 干员所在房间
        agent_is_high = agent.is_high()  # 是否高优先级

        # 调用原函数
        result = func(self, name, mood, current_room, current_index, update_time)
        if not update_time:
            return
        # 保存到数据库
        current_time = datetime.now()
        database_path = os.path.join('tmp', 'data.db')

        try:
            # Create 'tmp' directory if it doesn't exist
            os.makedirs('tmp', exist_ok=True)

            connection = sqlite3.connect(database_path)
            cursor = connection.cursor()

            # Create a table if it doesn't exist
            cursor.execute('CREATE TABLE IF NOT EXISTS agent_action ('
                           'name TEXT,'
                           'agent_current_room TEXT,'
                           'current_room TEXT,'
                           'is_high INTEGER,'
                           'agent_group TEXT,'
                           'mood REAL,'
                           'current_time TEXT'
                           ')')

            # Insert data
            cursor.execute('INSERT INTO agent_action VALUES (?, ?, ?, ?, ?, ?, ?)',
                           (name, agent_current_room, current_room, int(agent_is_high), agent.group, mood,
                            str(current_time)))

            connection.commit()
            connection.close()

            # Log the action
            logger.debug(
                f"Saved action to SQLite: Name: {name}, Agent's Room: {agent_current_room}, Agent's group: {agent.group}, "
                f"Current Room: {current_room}, Is High: {agent_is_high}, Current Time: {current_time}")

        except sqlite3.Error as e:
            logger.error(f"SQLite error: {e}")

        return result

    return wrapper


```

It looks like the code is written in Python and it is using a SQLite database as the data source. The purpose of the code is to get the room information from the database and process it according to the rules.

The code first opens the database connection and then reads the data from the table 'table_name'. The data is then processed according to the rules specified in the comment.

The processed data is then stored in the 'processed_data' dictionary. The 'grouped_data' dictionary is used to store the data that belongs to each room according to the rules.

Finally, the code returns the processed data.


```
def get_work_rest_ratios():
    # TODO 整理数据计算工休比
    database_path = os.path.join('tmp', 'data.db')

    try:
    # 连接到数据库
        conn = sqlite3.connect(database_path)
        # conn = sqlite3.connect('../../tmp/data.db')
        cursor = conn.cursor()
        # 查询数据
        cursor.execute("""
                        SELECT a.*
                        FROM agent_action a
                        JOIN (
                            SELECT DISTINCT b.name
                            FROM agent_action b
                            WHERE DATE(b.current_time) >= DATE('now', '-7 day', 'localtime')
                            AND b.is_high = 1 AND b.current_room NOT LIKE 'dormitory%'
                            UNION
                            SELECT '菲亚梅塔' AS name
                        ) AS subquery ON a.name = subquery.name
                        WHERE DATE(a.current_time) >= DATE('now', '-1 month', 'localtime')
                        ORDER BY a.current_time;
                       """)
        data = cursor.fetchall()
        # 关闭数据库连接
        conn.close()
    except sqlite3.Error as e:
        data = []
    processed_data = {}
    grouped_data = {}
    for row in data:
        name = row[0]
        current_room = row[2]
        current_time = row[6]  # Assuming index 6 is the current_time column
        agent = grouped_data.get(name, {
            'agent_data': [{'current_time': current_time,
                            'current_room': current_room}],
            'difference': []
        })
        difference = {'time_diff': calculate_time_difference(agent['agent_data'][-1]['current_time'], current_time),
                      'current_room': agent['agent_data'][-1]['current_room']}
        agent['agent_data'].append({'current_time': current_time,
                                    'current_room': current_room})
        agent['difference'].append(difference)
        grouped_data[name] = agent
    for name in grouped_data:
        work_time = 0
        rest_time = 0
        for difference in grouped_data[name]['difference']:
            if difference['current_room'].startswith('dormitory'):
                rest_time += difference['time_diff']
            else:
                work_time += difference['time_diff']
        processed_data[name] = {'labels':['休息时间','工作时间'],
                                'datasets':[{
                                    'data':[rest_time,work_time]
                                }]}

    return processed_data


```

mood_data = {}  # dictionaries to store the data with mood labels
mood_labels = []  # list to store the mood labels
grouped_work_rest_data = {}  # dictionary to store the data with mood labels grouped by group name

# Mood Data
mood_data['datasets'] = {}  # dictionary to store the data with mood labels
mood_data['labels'] = []  # list to store the mood labels

for row in current_group.cursor():
   current_time = row[6]  # assuming 'current_time' is at index 7
   current_date = row[8]  # assuming 'current_date' is at index 9
   current_user = row[10]  # assuming 'current_user' is at index 11
   current_role = row[12]  # assuming 'current_role' is at index 13
   current_permission = row[15]  # assuming 'current_permission' is at index 16
   current_channel = row[19]  # assuming 'current_channel' is at index 20
   current_mood = row[23]  # assuming 'current_mood' is at index 24
   current_label = row[25]  # assuming 'current_label' is at index 26
   current_date_with_time = row[27]  # assuming 'current_date_with_time' is at index 28
   current_time_interval = row[29]  # assuming 'current_time_interval' is at index 30
   current_group_name = row[33]  # assuming 'current_group_name' is at index 34
   current_user_id = row[36]  # assuming 'current_user_id' is at index 37
   current_user_name = row[38]  # assuming 'current_user_name' is at index 39
   current_user_email = row[40]  # assuming 'current_user_email' is at index 41
   current_user_role = row[43]  # assuming 'current_user_role' is at index 44
   current_user_permission = row[45]  # assuming 'current_user_permission' is at index 46
   current_user_channel = row[48]  # assuming 'current_user_channel' is at index 49
   current_user_mood = row[51]  # assuming 'current_user_mood' is at index 52
   current_user_mood_label = row[53]  # assuming 'current_user_mood_label' is at index 54
   current_user_channel_label = row[55]  # assuming 'current_user_channel_label' is at index 56
   current_user_channel = row[57]  # assuming 'current_user_channel' is at index 58
   current_user_channel_data = row[59]  # assuming 'current_user_channel_data' is at index 60
   current_user_channel_data_json = row[61]  # assuming 'current_user_channel_data_json' is at index 62
   current_user_channel_name = row[63]  # assuming 'current_user_channel_name' is at index 64
   current_user_channel_name_lowercase = row[65]  # assuming 'current_user_channel_name_lowercase' is at index 66
   current_user_channel_status = row[67]  # assuming 'current_user_channel_status' is at index 68
   current_user_channel_status_json = row[69]  # assuming 'current_user_channel_status_json' is at index 70
   current_user_channel_status_str = row[71]  # assuming 'current_user_channel_status_str' is at index 72
   current_user_channel_id = row[73]  # assuming 'current_user_channel_id' is at index 74
   current_user_channel_id_lowercase = row[75]  # assuming 'current_user_channel_id_lowercase' is at index 76
   current_user_channel_data_with_time = current_user_channel_data + current_time
   current_user_channel_data_json = current_user_channel_data_json.append(current_time)
   current_user_channel_name = current_user_channel_name.lowercase()
   current_user_channel_status = 'active'
   current_user_channel_status_json = current_user_channel_status_json.append(current_user_channel_status)
   current_user_channel_status_str = current_user_channel_status_str.append(current_user_channel_status)
   current_user_channel = current_user_channel + current_time + current_user_channel_status
   current_user_channel_data_with_time_json = current_user_channel_data_with_time.json()
   current_user_channel_data_with_time_array = current_user_channel_data_with_time_json.array()
   current_user_channel_data_with_time = current_user_channel_data_with_time_array.astype(int)[0]
   current_user_channel_data = current_user_channel_data + current_time
   current_user_channel_data_json = current_user_channel_data_json.append(current_user_channel_data)
   current_user_channel_name = current_user_channel_name.lowercase()
   current_user_channel_status = current_user_channel_status.lowercase()
   current_user_channel_status_json = current_user_channel_status_json.append(current_user_channel_status)
   current_user_channel_status_str = current_user_channel_status_str.append(current_user_channel_status)
   current_user_channel = current_user_channel + current_time + current_user_channel_status
   current_user_channel_data_with_time_json = current_user_channel_data_with_time.json()
   current_user_channel_data_with_time_array = current


```
# 整理心情曲线
def get_mood_ratios():
    database_path = os.path.join('tmp', 'data.db')

    try:
        # 连接到数据库
        conn = sqlite3.connect(database_path)
        cursor = conn.cursor()
        # 查询数据（筛掉宿管和替班组的数据）
        cursor.execute("""
                       SELECT a.*
                        FROM agent_action a
                        JOIN (
                            SELECT DISTINCT b.name
                            FROM agent_action b
                            WHERE DATE(b.current_time) >= DATE('now', '-7 day', 'localtime')
                            AND b.is_high = 1 AND b.current_room NOT LIKE 'dormitory%'
                            UNION
                            SELECT '菲亚梅塔' AS name
                        ) AS subquery ON a.name = subquery.name
                        WHERE DATE(a.current_time) >= DATE('now', '-7 day', 'localtime')
                        ORDER BY a.agent_group DESC, a.current_time;

        """)
        data = cursor.fetchall()
        # 关闭数据库连接
        conn.close()
    except sqlite3.Error as e:
        data = []

    work_rest_data_ratios = get_work_rest_ratios()
    grouped_data = {}
    grouped_work_rest_data = {}
    for row in data:
        group_name = row[4]  # Assuming 'agent_group' is at index 4
        if not group_name:
            group_name = row[0]
        mood_data = grouped_data.get(group_name, {
            'labels': [],
            'datasets': []
        })
        work_rest_data = grouped_work_rest_data.get(group_name,
            work_rest_data_ratios[row[0]]
        )
        grouped_work_rest_data[group_name]=work_rest_data


        timestamp_datetime = datetime.strptime(row[6], '%Y-%m-%d %H:%M:%S.%f')  # Assuming 'current_time' is at index 6
        # 创建 Luxon 格式的字符串
        current_time = f"{timestamp_datetime.year:04d}-{timestamp_datetime.month:02d}-{timestamp_datetime.day:02d}T{timestamp_datetime.hour:02d}:{timestamp_datetime.minute:02d}:{timestamp_datetime.second:02d}.{timestamp_datetime.microsecond:06d}+08:00"

        mood_label = row[0]  # Assuming 'name' is at index 0
        mood_value = row[5]  # Assuming 'mood' is at index 5

        if mood_label in [dataset['label'] for dataset in mood_data['datasets']]:
            # if mood_label == mood_data['datasets'][0]['label']:
            mood_data['labels'].append(current_time)
            # If mood label already exists, find the corresponding dataset
            for dataset in mood_data['datasets']:
                if dataset['label'] == mood_label:
                    dataset['data'].append({'x': current_time, 'y': mood_value})
                    break
        else:
            # If mood label doesn't exist, create a new dataset
            mood_data['labels'].append(current_time)
            mood_data['datasets'].append({
                'label': mood_label,
                'data': [{'x': current_time, 'y': mood_value}]
            })

        grouped_data[group_name] = mood_data
    print(grouped_work_rest_data)
    # 将数据格式整理为数组
    formatted_data = []
    for group_name, mood_data in grouped_data.items():
        formatted_data.append({
            'groupName': group_name,
            'moodData': mood_data,
            'workRestData':grouped_work_rest_data[group_name]
        })
    return formatted_data


```

这两段代码定义了一个名为 `calculate_time_difference` 的函数，它接受两个参数 `start_time` 和 `end_time`，分别表示开始时间和结束时间。函数使用 `datetime.strptime` 方法将开始时间和结束时间转换为 datetime 对象，并计算它们之间的差值。函数返回差值的时间以秒为单位。

在另一个名为 `__main__` 的函数中，函数调用 `calculate_time_difference` 函数并传入自己的参数，即开始时间和结束时间。函数的作用是计算程序运行时间，即 `calculate_time_difference` 函数返回的结果。

该代码还会调用另一个名为 `__main__` 的函数，该函数在代码中没有定义，因此不会对程序产生任何影响。


```
def calculate_time_difference(start_time, end_time):
    time_format = '%Y-%m-%d %H:%M:%S.%f'
    start_datetime = datetime.strptime(start_time, time_format)
    end_datetime = datetime.strptime(end_time, time_format)
    time_difference = end_datetime - start_datetime
    return time_difference.total_seconds()


def __main__():
    get_work_rest_ratios()

__main__()

```

# `/opt/arknights-mower/arknights_mower/solvers/recruit.py`

This code defines a class RecruitPoss, which appears to be a combination of RecruitAgent and RecruitTag objects. RecruitPoss似乎用于招募两栖动物(可能是青蛙)，并具有以下功能：

1. 从 `..data` 包中导入 `recruit_agent` 和 `recruit_tag` 对象。
2. 从 `..data.recruit_agent_list` 导出招募代理器的列表。
3. 从 `..ocr` 包中导入 `ocr_rectify` 和 `ocrhandle` 函数，对捕获到的文本进行 OCR 处理。
4. 从 `..utils` 包中导入 `segment` 和 `device` 函数，对文本进行分词和处理设备。
5. 从 `..utils.email` 包中导入 `recruit_template` 和 `recruit_rarity` 函数，创建招募邮件的主题和内容。
6. 从 `..utils.recognize` 包中导入 `RecognizeError`、`Recognizer` 和 `Scene` 类，对文本进行识别并处理。
7. 从 `..utils.solver` 包中导入 `BaseSolver` 类，用于求解问题。

总而言之， RecruitPoss 似乎是一个用于招募青蛙的 AI 系统，可以从 OCR 文本、分词、识别和求解问题等方面进行操作。


```
from __future__ import annotations

from itertools import combinations

from ..data import recruit_agent, recruit_tag, recruit_agent_list
from ..ocr import ocr_rectify, ocrhandle
from ..utils import segment
from ..utils.device import Device
from ..utils.email import recruit_template, recruit_rarity
from ..utils.log import logger
from ..utils.recognize import RecognizeError, Recognizer, Scene
from ..utils.solver import BaseSolver


# class RecruitPoss(object):
```

这段代码是一个 Python 类的实例，这个类记录了公招标签组合的可能性数据。

在这个类的实例中，有五个变量：choose、max、min、poss 和 ls。其中，choose 是标签选择（按位），第 6 个标志位表示是否选满招募时限，0 为选满，1 为选 03:50；max 是等级上限；min 是等级下限；poss 是可能性；ls 是可能的干员列表。

这个类的实例还定义了一个方法 __init__，用于初始化这些变量。在这个方法中，对这五个变量进行了赋值，并传入了初始值 03:50。

该类的实例还定义了一个方法 __lt__(self, another: RecruitPoss)，用于比较两个实例的大小，这个方法返回的是 True，如果当前实例的 Poss 变量值小于另一个实例的 Poss 变量值。

最后，该类的实例还定义了一个方法 __str__(self)，用于打印输出实例的对象，返回的信息包括当前实例的所有变量。


```
#     """ 记录公招标签组合的可能性数据 """
#
#     def __init__(self, choose: int, max: int = 0, min: int = 7) -> None:
#         self.choose = choose  # 标签选择（按位），第 6 个标志位表示是否选满招募时限，0 为选满，1 为选 03:50
#         self.max = max  # 等级上限
#         self.min = min  # 等级下限
#         self.poss = 0  # 可能性
#         self.lv2a3 = False  # 是否包含等级为 2 和 3 的干员
#         self.ls = []  # 可能的干员列表
#
#     def __lt__(self, another: RecruitPoss) -> bool:
#         return (self.poss) < (another.poss)
#
#     def __str__(self) -> str:
#         return "%s,%s,%s,%s,%s" % (self.choose, self.max, self.min, self.poss, self.ls)
```

This is a class definition for a robot that can recruit new agents based on certain conditions. The robot can recruit agents with different rarity levels and can only recruit agents that have a certain level of level. Additionally, depending on the recruitment conditions, the robot can choose to recruit agents of a specific type.

The robot can also send emails to the chat history，告知用户有哪些五星干员和四星干员是可以通过手动招募选择，并标记为已选择，以便用户更容易地查看哪些干员是可以手动招募的。

This class is in charge of the basic logic for the robot, and can be further developed to support more complex features.


```
#
#     def __repr__(self) -> str:
#         return "%s,%s,%s,%s,%s" % (self.choose, self.max, self.min, self.poss, self.ls)


class RecruitSolver(BaseSolver):
    """
    自动进行公招
    """

    def __init__(self, device: Device = None, recog: Recognizer = None) -> None:
        super().__init__(device, recog)

        self.result_agent = {}
        self.agent_choose = {}
        self.recruit_config = {}

        self.recruit_pos = -1

    def run(self, priority: list[str] = None, email_config={}, maa_config={}) -> None:
        """
        :param priority: list[str], 优先考虑的公招干员，默认为高稀有度优先
        """
        self.priority = priority
        self.recruiting = 0
        self.has_ticket = True  # 默认含有招募票
        self.can_refresh = True  # 默认可以刷新
        self.email_config = email_config

        # 调整公招参数
        self.add_recruit_param(maa_config)

        logger.info('Start: 公招')
        # 清空
        self.result_agent.clear()

        self.result_agent = {}
        self.agent_choose = {}

        # logger.info(f'目标干员：{priority if priority else "无，高稀有度优先"}')
        try:
            super().run()
        except Exception as e:
            logger.error(e)

        logger.debug(self.agent_choose)
        logger.debug(self.result_agent)

        if self.result_agent:
            logger.info(f"上次公招结果汇总{self.result_agent}")

        if self.agent_choose:
            logger.info(f'公招标签：{self.agent_choose}')
        if self.agent_choose or self.result_agent:
            self.send_email(recruit_template.render(recruit_results=self.agent_choose,
                                                    recruit_get_agent=self.result_agent,
                                                    title_text="公招汇总"), "公招汇总通知", "html")

    def add_recruit_param(self, maa_config):
        if not maa_config:
            raise Exception("招募设置为空")

        if maa_config['recruitment_time']:
            recruitment_time = 460
        else:
            recruitment_time = 540

        self.recruit_config = {
            "recruit_only_4": maa_config['recruit_only_4'],
            "recruitment_time": {
                "3": recruitment_time,
                "4": 540
            }
        }

    def transition(self) -> bool:
        if self.scene() == Scene.INDEX:
            self.tap_element('index_recruit')
        elif self.scene() == Scene.RECRUIT_MAIN:
            segments = segment.recruit(self.recog.img)
            tapped = False
            for idx, seg in enumerate(segments):
                # 在主界面重置为-1
                self.recruit_pos = -1
                if self.recruiting & (1 << idx) != 0:
                    continue
                if self.tap_element('recruit_finish', scope=seg, detected=True):
                    # 完成公招点击聘用候选人
                    self.recruit_pos = idx
                    tapped = True
                    break
                if not self.has_ticket and not self.can_refresh:
                    continue
                # 存在职业需求，说明正在进行招募
                required = self.find('job_requirements', scope=seg)
                if required is None:
                    # 不在进行招募的位置 （1、空位 2、人力办公室等级不够没升的位置）
                    # 下一次循环进入对应位置，先存入值
                    self.recruit_pos = idx
                    self.tap(seg)
                    tapped = True
                    self.recruiting |= (1 << idx)
                    break
            if not tapped:
                return True
        elif self.scene() == Scene.RECRUIT_TAGS:
            return self.recruit_tags()
        elif self.scene() == Scene.SKIP:
            self.tap_element('skip')
        elif self.scene() == Scene.RECRUIT_AGENT:
            return self.recruit_result()
        elif self.scene() == Scene.MATERIEL:
            self.tap_element('materiel_ico')
        elif self.scene() == Scene.LOADING:
            self.sleep(3)
        elif self.scene() == Scene.CONNECTING:
            self.sleep(3)
        elif self.get_navigation():
            self.tap_element('nav_recruit')
        elif self.scene() != Scene.UNKNOWN:
            self.back_to_index()
        else:
            raise RecognizeError('Unknown scene')

    def recruit_tags(self) -> bool:
        """ 识别公招标签的逻辑 """
        if self.find('recruit_no_ticket') is not None:
            self.has_ticket = False
        if self.find('recruit_no_refresh') is not None:
            self.can_refresh = False

        needs = self.find('career_needs', judge=False)
        avail_level = self.find('available_level', judge=False)
        budget = self.find('recruit_budget', judge=False)
        up = needs[0][1] - 80
        down = needs[1][1] + 60
        left = needs[1][0]
        right = avail_level[0][0]

        while True:
            # ocr the recruitment tags and rectify
            img = self.recog.img[up:down, left:right]
            ocr = ocrhandle.predict(img)
            for x in ocr:
                if x[1] not in recruit_tag:
                    x[1] = ocr_rectify(img, x, recruit_tag, '公招标签')

            # recruitment tags
            tags = [x[1] for x in ocr]
            logger.info(f'第{self.recruit_pos + 1}个位置上的公招标签：{tags}')

            # 计算招募标签组合结果
            best, need_choose = self.recruit_cal(tags, self.priority)

            # 刷新标签
            if need_choose is False:
                '''稀有tag或支援，不需要选'''
                self.send_email(recruit_rarity.render(recruit_results=best, title_text="稀有tag通知"), "出稀有标签辣", "html")
                logger.debug('稀有tag,发送邮件')
                self.back()
                return
            # best为空说明是三星，刷新标签
            if not best:
                # refresh
                if self.tap_element('recruit_refresh', detected=True):
                    self.tap_element('double_confirm', 0.8,
                                     interval=3, judge=False)
                    logger.info("刷新标签")
                    continue
            break

        # 如果没有招募券则只刷新标签不选人
        if not self.has_ticket:
            logger.info('无招募券 结束公招')
            self.back()
            return

        # best为空说明这次大概率三星
        if self.recruit_config["recruit_only_4"] and not best:
            logger.info('不招三星 结束公招')
            self.back()
            return

        choose = []
        if len(best) > 0:
            choose = (next(iter(best)))
            # tap selected tags

        logger.info(f'选择标签：{list(choose)}')
        for x in ocr:
            color = self.recog.img[up + x[2][0][1] - 5, left + x[2][0][0] - 5]
            if (color[2] < 100) != (x[1] not in choose):
                # 存在choose为空但是进行标签选择的情况
                logger.debug(f"tap{x}")
                self.device.tap((left + x[2][0][0] - 5, up + x[2][0][1] - 5))

        # 9h为True 3h50min为False
        recruit_time_choose = self.recruit_config["recruitment_time"]["3"]
        if len(best) > 0:
            if best[choose]['level'] == 1:
                recruit_time_choose = 230
            else:
                recruit_time_choose = self.recruit_config["recruitment_time"][str(best[choose]['level'])]

        if recruit_time_choose == 540:
            # 09:00
            logger.debug("时间9h")
            self.tap_element('one_hour', 0.2, 0.8, 0)
        elif recruit_time_choose == 230:
            # 03:50
            logger.debug("时间3h50min")
            [self.tap_element('one_hour', 0.2, 0.2, 0) for _ in range(2)]
            [self.tap_element('one_hour', 0.5, 0.2, 0) for _ in range(5)]
        elif recruit_time_choose == 460:
            # 07:40
            logger.debug("时间7h40min")
            [self.tap_element('one_hour', 0.2, 0.8, 0) for _ in range(2)]
            [self.tap_element('one_hour', 0.5, 0.8, 0) for _ in range(2)]

        # start recruit
        self.tap((avail_level[1][0], budget[0][1]), interval=3)
        if len(best) > 0:
            logger_result = best[choose]['agent']
            self.agent_choose[str(self.recruit_pos + 1)] = best

            logger.info(f'第{self.recruit_pos + 1}个位置上的公招预测结果：{logger_result}')
        else:
            self.agent_choose[str(self.recruit_pos + 1)] = {
                ('',): {
                    'isRarity': False,
                    'isRobot': False,
                    'level': 3,
                    'agent': [{'name': '随机三星干员', 'level': 3}]}
            }

            logger.info(f'第{self.recruit_pos + 1}个位置上的公招预测结果：{"随机三星干员"}')

    def recruit_result(self) -> bool:
        """ 识别公招招募到的干员 """
        """ 卡在首次获得 挖个坑"""
        agent = None
        ocr = ocrhandle.predict(self.recog.img)
        for x in ocr:
            if x[1][-3:] == '的信物':
                agent = x[1][:-3]
                agent_ocr = x
                break
        if agent is None:
            logger.warning('未能识别到干员名称')
        else:
            if agent not in recruit_agent.keys():
                agent_with_suf = [x + '的信物' for x in recruit_agent.keys()]
                agent = ocr_rectify(
                    self.recog.img, agent_ocr, agent_with_suf, '干员名称')[:-3]
            if agent in recruit_agent.keys():
                if 2 <= recruit_agent[agent]['stars'] <= 4:
                    logger.info(f'获得干员：{agent}')
                else:
                    logger.critical(f'获得干员：{agent}')

        if agent is not None:
            # 汇总开包结果
            self.result_agent[str(self.recruit_pos + 1)] = agent

        self.tap((self.recog.w // 2, self.recog.h // 2))

    def merge_agent_list(self, tags: [str], list_1: list[dict], list_2={}, list_3={}):
        """
        交叉筛选干员

        tags:组合出的标签
        list_x:每个单独标签对应的干员池

        return : 筛选出来的干员池，平均等级，是否稀有标签，是否支援机械
        """
        List1_name_dict = {}
        merge_list = []
        isRarity = False
        isRobot = False
        level = 7

        for operator in list_1:
            if operator['level'] == 6 and "高级资深干员" not in tags:
                continue
            elif operator['level'] == 1 and "支援机械" not in tags:
                continue
            elif operator['level'] == 2:
                continue
            List1_name_dict[operator['name']] = operator

        for key in List1_name_dict:
            if List1_name_dict[key]['level'] == 2:
                print(List1_name_dict[key])

        if len(tags) == 1 and not list_2 and not list_3:
            for key in List1_name_dict:
                if List1_name_dict[key]['level'] < level:
                    level = List1_name_dict[key]['level']
                merge_list.append(List1_name_dict[key])

        elif len(tags) == 2 and not list_3:
            for operator in list_2:
                if operator['name'] in List1_name_dict:
                    if operator['level'] < level:
                        level = operator['level']
                    merge_list.append(operator)


        elif len(tags) == 3 and list_3:
            List1and2_name_dict = {}
            for operator in list_2:
                if operator['name'] in List1_name_dict:
                    List1and2_name_dict[operator['name']] = operator

            for operator in list_3:
                if operator['name'] in List1and2_name_dict:
                    if operator['level'] < level:
                        level = operator['level']
                    merge_list.append(operator)

        if level >= 5:
            isRarity = True
        elif level == 1:
            isRobot = True
        logger.debug(f"merge_list{merge_list}")

        return merge_list, level, isRarity, isRobot

    def recruit_cal(self, tags: list[str], auto_robot=False, need_Robot=True) -> (dict, bool):
        possible_list = {}
        has_rarity = False
        has_robot = False

        for item in combinations(tags, 1):
            # 防止出现类似情况 ['重', '装', '干', '员']

            merge_temp, level, isRarity, isRobot = self.merge_agent_list(item, recruit_agent_list[item[0]]['agent'])
            if len(merge_temp) > 0:
                if has_rarity is False and isRarity:
                    has_rarity = isRarity
                if has_robot is False and isRobot:
                    has_robot = isRobot
                possible_list[item[0],] = {
                    "isRarity": isRarity,
                    "isRobot": isRobot,
                    "level": level,
                    "agent": merge_temp
                }

        for item in combinations(tags, 2):
            merge_temp, level, isRarity, isRobot = self.merge_agent_list(item, recruit_agent_list[item[0]]['agent'],
                                                                         recruit_agent_list[item[1]]['agent'])
            if len(merge_temp) > 0:
                if has_rarity is False and isRarity:
                    has_rarity = isRarity
                if has_robot is False and isRobot:
                    has_robot = isRobot
                possible_list[item[0], item[1]] = {
                    "isRarity": isRarity,
                    "isRobot": isRobot,
                    "level": level,
                    "agent": merge_temp
                }
        for item in combinations(tags, 3):
            merge_temp, level, isRarity, isRobot = self.merge_agent_list(item, recruit_agent_list[item[0]]['agent'],
                                                                         recruit_agent_list[item[1]]['agent'],
                                                                         recruit_agent_list[item[2]]['agent'])
            if len(merge_temp) > 0:
                if has_rarity is False and isRarity:
                    has_rarity = isRarity
                if has_robot is False and isRobot:
                    has_robot = isRobot

                possible_list[item[0], item[1], item[2]] = {
                    "isRarity": isRarity,
                    "isRobot": isRobot,
                    "level": level,
                    "agent": merge_temp
                }

        logger.debug(f"公招可能性:{self.recruit_str(possible_list)}")

        for key in list(possible_list.keys()):
            # 五星六星选择优先级大于支援机械
            if has_rarity:
                if possible_list[key]['isRarity'] is False:
                    del possible_list[key]
                    continue
                elif possible_list[key]['level'] < 6 and "高级资深干员" in tags:
                    del possible_list[key]
                    continue
            # 不要支援机械
            elif need_Robot is False and possible_list[key]['isRobot'] is True:
                del possible_list[key]
                continue
            # 支援机械手动选择
            elif has_robot and need_Robot is True and possible_list[key]['isRobot'] is False:
                del possible_list[key]
                continue

            '''只保留大概率能出的tag'''
            for i in range(len(possible_list[key]["agent"]) - 1, -1, -1):
                if possible_list[key]["agent"][i]['level'] != possible_list[key]["level"]:
                    possible_list[key]["agent"].remove(possible_list[key]["agent"][i])

        # 六星 五星 支援机械手动选择返回全部结果

        # 有支援机械 不需要自动点支援机械 并且需要支援机械的情况下，邮件提醒
        notice_robot = (has_robot and auto_robot == False and need_Robot)
        need_choose = True
        if notice_robot or has_rarity:
            need_choose = False
            logger.info(f"稀有tag:{self.recruit_str(possible_list)}")
            return possible_list, need_choose

        best = {}
        # 4星=需要选择tag返回选择的tag，3星不选
        for key in possible_list:
            if possible_list[key]['level'] >= 4:
                best[key] = possible_list[key]
                break

        return best, need_choose

    def recruit_str(self, recruit_result: dict):
        if not recruit_result:
            return "随机三星干员"
        result_str = "{"
        for key in recruit_result:
            temp_str = "{[" + ",".join(list(key))
            temp_str = temp_str + "],level:"
            temp_str = temp_str + str(recruit_result[key]["level"]) + ",agent:"
            temp_str = temp_str + str(recruit_result[key]["agent"]) + "},"
            result_str = result_str + temp_str

        return result_str

```