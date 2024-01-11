# `arknights-mower\arknights_mower\utils\device\device.py`

```
# 导入未来的注解特性
from __future__ import annotations

# 导入时间模块
import time
# 导入可选类型模块
from typing import Optional

# 从上级目录中导入配置模块
from .. import config
# 从日志模块中导入日志和保存截图函数
from ..log import logger, save_screenshot
# 从 adb_client 模块中导入 ADBClient 类
from .adb_client import ADBClient
# 从 minitouch 模块中导入 MiniTouch 类
from .minitouch import MiniTouch
# 从 scrcpy 模块中导入 Scrcpy 类

class Device(object):
    """ Android Device """

    # 初始化方法，接受设备 ID、连接方式和触摸设备参数
    def __init__(self, device_id: str = None, connect: str = None, touch_device: str = None) -> None:
        self.device_id = device_id
        self.connect = connect
        self.touch_device = touch_device
        self.client = None
        self.control = None
        self.start()

    # 启动方法，初始化 ADBClient 对象和 Device.Control 对象
    def start(self) -> None:
        self.client = ADBClient(self.device_id, self.connect)
        self.control = Device.Control(self, self.client)

    # 运行命令方法，接受命令字符串，返回可选的字节流
    def run(self, cmd: str) -> Optional[bytes]:
        return self.client.run(cmd)

    # 启动应用方法
    def launch(self) -> None:
        """ launch the application """
        # 获取是否启用点击启动应用、点击坐标 x 和 y
        tap = config.TAP_TO_LAUNCH["enable"]
        x = config.TAP_TO_LAUNCH["x"]
        y = config.TAP_TO_LAUNCH["y"]

        # 如果启用点击启动应用，则发送点击命令，否则发送启动应用命令
        if tap:
            self.run(f'input tap {x} {y}')
        else:
            self.run(f'am start -n {config.APPNAME}/{config.APP_ACTIVITY_NAME}')

    # 退出应用方法，接受应用名称
    def exit(self, app: str) -> None:
        """ exit the application """
        # 发送强制停止应用命令
        self.run(f'am force-stop {app}')

    # 发送按键事件方法，接受按键码
    def send_keyevent(self, keycode: int) -> None:
        """ send a key event """
        # 记录按键事件并发送按键事件命令
        logger.debug(f'keyevent: {keycode}')
        command = f'input keyevent {keycode}'
        self.run(command)

    # 发送文本方法，接受文本字符串
    def send_text(self, text: str) -> None:
        """ send a text """
        # 记录发送的文本并发送输入文本命令
        logger.debug(f'text: {repr(text)}')
        text = text.replace('"', '\\"')
        command = f'input text "{text}"'
        self.run(command)

    # 截屏方法，接受是否保存截图参数
    def screencap(self, save: bool = False) -> bytes:
        """ get a screencap """
        # 发送截屏命令，获取截屏数据，如果需要保存则保存截图并返回截屏数据
        command = 'screencap -p 2>/dev/null'
        screencap = self.run(command)
        if save:
            save_screenshot(screencap)
        return screencap
    def current_focus(self) -> str:
        """ detect current focus app """
        # 定义命令，用于获取当前焦点的应用程序
        command = 'dumpsys window | grep mCurrentFocus'
        # 运行命令并将结果解码为UTF-8格式的字符串
        line = self.run(command).decode('utf8')
        # 去除字符串末尾的换行符，并按空格分割，返回最后一个元素
        return line.strip()[:-1].split(' ')[-1]

    def display_frames(self) -> tuple[int, int, int]:
        """ get display frames if in compatibility mode"""
        # 如果不在兼容模式下，则返回None
        if not config.MNT_COMPATIBILITY_MODE:
            return None

        # 定义命令，用于获取显示框架
        command = 'dumpsys window | grep DisplayFrames'
        # 运行命令并将结果解码为UTF-8格式的字符串
        line = self.run(command).decode('utf8')
        """ eg. DisplayFrames w=1920 h=1080 r=3 """
        # 去除字符串末尾的换行符，替换'='为' '，然后按空格分割，返回第3、5、7个元素转换为整数的元组
        res = line.strip().replace('=', ' ').split(' ')
        return int(res[2]), int(res[4]), int(res[6])

    def tap(self, point: tuple[int, int]) -> None:
        """ tap """
        # 记录日志，表示进行点击操作
        logger.debug(f'tap: {point}')
        # 调用控制对象的tap方法进行点击操作
        self.control.tap(point)

    def swipe(self, start: tuple[int, int], end: tuple[int, int], duration: int = 100) -> None:
        """ swipe """
        # 记录日志，表示进行滑动操作
        logger.debug(f'swipe: {start} -> {end}, duration={duration}')
        # 调用控制对象的swipe方法进行滑动操作
        self.control.swipe(start, end, duration)

    def swipe_ext(self, points: list[tuple[int, int]], durations: list[int], up_wait: int = 500) -> None:
        """ swipe_ext """
        # 记录日志，表示进行扩展滑动操作
        logger.debug(
            f'swipe_ext: points={points}, durations={durations}, up_wait={up_wait}')
        # 调用控制对象的swipe_ext方法进行扩展滑动操作
        self.control.swipe_ext(points, durations, up_wait)

    def check_current_focus(self):
        """ check if the application is in the foreground """
        # 如果当前焦点的应用程序不是指定的应用程序，则启动指定的应用程序
        if self.current_focus() != f"{config.APPNAME}/{config.APP_ACTIVITY_NAME}":
            self.launch()
            # 等待应用程序启动完成
            time.sleep(10)
```