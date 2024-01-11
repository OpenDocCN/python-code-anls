# `arknights-mower\arknights_mower\solvers\credit.py`

```
# 从工具包中导入检测器
from ..utils import detector
# 从工具包中导入设备类
from ..utils.device import Device
# 从工具包中导入日志记录器
from ..utils.log import logger
# 从工具包中导入识别器和识别错误类
from ..utils.recognize import RecognizeError, Recognizer, Scene
# 从工具包中导入基础解决方案类
from ..utils.solver import BaseSolver

# 定义信用解决方案类，继承自基础解决方案类
class CreditSolver(BaseSolver):
    """
    通过线索交换自动收集信用
    """

    # 初始化方法，接受设备和识别器作为参数
    def __init__(self, device: Device = None, recog: Recognizer = None) -> None:
        super().__init__(device, recog)

    # 运行方法
    def run(self) -> None:
        # 记录日志，表示开始执行信用任务
        logger.info('Start: 信用')
        # 调用父类的运行方法
        super().run()

    # 状态转换方法
    def transition(self) -> bool:
        # 如果当前场景是首页
        if self.scene() == Scene.INDEX:
            # 点击首页的好友元素
            self.tap_element('index_friend')
        # 如果当前场景是关闭的好友列表
        elif self.scene() == Scene.FRIEND_LIST_OFF:
            # 点击好友列表
            self.tap_element('friend_list')
        # 如果当前场景是打开的好友列表
        elif self.scene() == Scene.FRIEND_LIST_ON:
            # 获取好友列表底部的坐标
            down = self.find('friend_list_on', strict=True)[1][1]
            # 设置范围为整个屏幕
            scope = [(0, 0), (100000, down)]
            # 如果在指定范围内找到好友访问元素，则点击
            if not self.tap_element('friend_visit', scope=scope, detected=True):
                self.sleep(1)
        # 如果当前场景是正在访问好友
        elif self.scene() == Scene.FRIEND_VISITING:
            # 检测是否出现访问次数限制
            visit_limit = self.find('visit_limit')
            if visit_limit is not None:
                return True
            # 检测是否有下一个好友可以访问
            visit_next = detector.visit_next(self.recog.img)
            if visit_next is not None:
                self.tap(visit_next)
            else:
                return True
        # 如果当前场景是加载中
        elif self.scene() == Scene.LOADING:
            self.sleep(3)
        # 如果当前场景是连接中
        elif self.scene() == Scene.CONNECTING:
            self.sleep(3)
        # 如果存在导航栏
        elif self.get_navigation():
            self.tap_element('nav_social')
        # 如果当前场景不是未知场景
        elif self.scene() != Scene.UNKNOWN:
            # 返回到首页
            self.back_to_index()
        # 如果以上条件都不满足，则抛出识别错误
        else:
            raise RecognizeError('Unknown scene')
```