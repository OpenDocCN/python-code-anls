# `arknights-mower\arknights_mower\solvers\mission.py`

```py
# 从utils目录中导入Device类
from ..utils.device import Device
# 从utils目录中导入logger对象
from ..utils.log import logger
# 从utils目录中导入RecognizeError异常类、Recognizer类和Scene类
from ..utils.recognize import RecognizeError, Recognizer, Scene
# 从utils目录中导入BaseSolver类
from ..utils.solver import BaseSolver

# 定义MissionSolver类，继承自BaseSolver类
class MissionSolver(BaseSolver):
    """
    点击确认完成每日任务和每周任务
    """

    # 初始化方法，接受一个Device对象和一个Recognizer对象作为参数
    def __init__(self, device: Device = None, recog: Recognizer = None) -> None:
        super().__init__(device, recog)

    # 运行方法，不返回任何结果
    def run(self) -> None:
        # 任务完成状态，1代表每日任务，2代表每周任务
        self.checked = 0

        # 输出日志信息，表示开始执行任务
        logger.info('Start: 任务')
        # 调用父类的run方法
        super().run()
    # 定义状态转换方法，返回布尔值
    def transition(self) -> bool:
        # 如果当前场景是INDEX
        if self.scene() == Scene.INDEX:
            # 点击index_mission元素
            self.tap_element('index_mission')
        # 如果当前场景是MISSION_TRAINEE
        elif self.scene() == Scene.MISSION_TRAINEE:
            # 如果checked按位与1的结果为0
            if self.checked & 1 == 0:
                # 点击mission_daily元素
                self.tap_element('mission_daily')
            # 如果checked按位与2的结果为0
            elif self.checked & 2 == 0:
                # 点击mission_weekly元素
                self.tap_element('mission_weekly')
            else:
                # 返回True
                return True
        # 如果当前场景是MISSION_DAILY
        elif self.scene() == Scene.MISSION_DAILY:
            # 将checked按位或1
            self.checked |= 1
            # 查找mission_collect元素
            collect = self.find('mission_collect')
            # 如果未找到，等待1秒后再次查找
            if collect is None:
                self.sleep(1)
                collect = self.find('mission_collect')
            # 如果找到了collect元素
            if collect is not None:
                # 记录日志
                logger.info('任务：一键收取任务')
                # 点击collect元素
                self.tap(collect)
            # 如果checked按位与2的结果为0
            elif self.checked & 2 == 0:
                # 点击mission_weekly元素
                self.tap_element('mission_weekly')
            else:
                # 返回True
                return True
        # 如果当前场景是MISSION_WEEKLY
        elif self.scene() == Scene.MISSION_WEEKLY:
            # 将checked按位或2
            self.checked |= 2
            # 查找mission_collect元素
            collect = self.find('mission_collect')
            # 如果未找到，等待1秒后再次查找
            if collect is None:
                self.sleep(1)
                collect = self.find('mission_collect')
            # 如果找到了collect元素
            if collect is not None:
                # 记录日志
                logger.info('任务：一键收取任务')
                # 点击collect元素
                self.tap(collect)
            # 如果checked按位与1的结果为0
            elif self.checked & 1 == 0:
                # 点击mission_daily元素
                self.tap_element('mission_daily')
            else:
                # 返回True
                return True
        # 如果当前场景是MATERIEL
        elif self.scene() == Scene.MATERIEL:
            # 点击materiel_ico元素
            self.tap_element('materiel_ico')
        # 如果当前场景是LOADING
        elif self.scene() == Scene.LOADING:
            # 等待3秒
            self.sleep(3)
        # 如果当前场景是CONNECTING
        elif self.scene() == Scene.CONNECTING:
            # 等待3秒
            self.sleep(3)
        # 如果有导航
        elif self.get_navigation():
            # 点击nav_mission元素
            self.tap_element('nav_mission')
        # 如果当前场景不是UNKNOWN
        elif self.scene() != Scene.UNKNOWN:
            # 返回到INDEX场景
            self.back_to_index()
        else:
            # 抛出识别错误
            raise RecognizeError('Unknown scene')
```