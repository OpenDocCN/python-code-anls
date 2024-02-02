# `arknights-mower\arknights_mower\solvers\mail.py`

```py
from ..utils.device import Device  # 从指定路径导入 Device 类
from ..utils.log import logger  # 从指定路径导入 logger 对象
from ..utils.recognize import RecognizeError, Recognizer, Scene  # 从指定路径导入 RecognizeError, Recognizer, Scene
from ..utils.solver import BaseSolver  # 从指定路径导入 BaseSolver 类

class MailSolver(BaseSolver):  # 定义一个名为 MailSolver 的类，继承自 BaseSolver 类
    """
    收取邮件
    """

    def __init__(self, device: Device = None, recog: Recognizer = None) -> None:  # 初始化方法，接受 device 和 recog 两个参数
        super().__init__(device, recog)  # 调用父类的初始化方法，传入 device 和 recog 参数

    def run(self) -> None:  # 定义一个名为 run 的方法，不返回任何值
        # if it touched
        self.touched = False  # 初始化一个名为 touched 的属性，并赋值为 False

        logger.info('Start: 邮件')  # 使用 logger 对象记录日志信息
        super().run()  # 调用父类的 run 方法

    def transition(self) -> bool:  # 定义一个名为 transition 的方法，返回布尔值
        if self.scene() == Scene.INDEX:  # 如果当前场景是 INDEX
            scope = ((0, 0), (100+self.recog.w//4, self.recog.h//10))  # 定义一个名为 scope 的变量
            nav = self.find('index_nav', thres=250, scope=scope)  # 在指定范围内查找名为 index_nav 的元素
            self.tap(nav, 0.625)  # 点击找到的元素，持续时间为 0.625 秒
        elif self.scene() == Scene.MAIL:  # 如果当前场景是 MAIL
            if self.touched:  # 如果 touched 属性为 True
                return True  # 返回 True
            self.touched = True  # 将 touched 属性设置为 True
            self.tap_element('read_mail')  # 点击名为 read_mail 的元素
        elif self.scene() == Scene.LOADING:  # 如果当前场景是 LOADING
            self.sleep(3)  # 休眠 3 秒
        elif self.scene() == Scene.CONNECTING:  # 如果当前场景是 CONNECTING
            self.sleep(3)  # 休眠 3 秒
        elif self.scene() == Scene.MATERIEL:  # 如果当前场景是 MATERIEL
            self.tap_element('materiel_ico')  # 点击名为 materiel_ico 的元素
        elif self.get_navigation():  # 如果存在导航
            self.tap_element('nav_index')  # 点击名为 nav_index 的元素
        elif self.scene() != Scene.UNKNOWN:  # 如果当前场景不是 UNKNOWN
            self.back_to_index()  # 返回到 INDEX 场景
        else:  # 如果以上条件都不满足
            raise RecognizeError('Unknown scene')  # 抛出 RecognizeError 异常，提示未知场景
```