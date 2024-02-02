# `arknights-mower\arknights_mower\utils\device\scrcpy\control.py`

```py
import functools  # 导入 functools 模块
import socket  # 导入 socket 模块
import struct  # 导入 struct 模块
from time import sleep  # 从 time 模块中导入 sleep 函数

from . import const  # 从当前包中导入 const 模块


def inject(control_type: int):
    """
    Inject control code, with this inject, we will be able to do unit test
    Args:
        control_type: event to send, TYPE_*
    """
    # 定义装饰器函数，接受控制类型参数
    def wrapper(f):
        @functools.wraps(f)
        def inner(*args, **kwargs):
            # 将控制类型和函数返回值打包成字节流
            package = struct.pack(">B", control_type) + f(*args, **kwargs)
            # 如果父对象的控制套接字不为空
            if args[0].parent.control_socket is not None:
                # 使用控制套接字锁，发送打包好的数据
                with args[0].parent.control_socket_lock:
                    args[0].parent.control_socket.send(package)
            return package  # 返回打包好的数据
        return inner  # 返回内部函数
    return wrapper  # 返回装饰器函数


class ControlSender:
    def __init__(self, parent):
        self.parent = parent  # 初始化父对象

    @inject(const.TYPE_INJECT_KEYCODE)  # 使用装饰器注入按键事件类型
    def keycode(
        self, keycode: int, action: int = const.ACTION_DOWN, repeat: int = 0
    ) -> bytes:
        """
        Send keycode to device
        Args:
            keycode: const.KEYCODE_*
            action: ACTION_DOWN | ACTION_UP
            repeat: repeat count
        """
        return struct.pack(">Biii", action, keycode, repeat, 0)  # 返回打包好的按键事件数据

    @inject(const.TYPE_INJECT_TEXT)  # 使用装饰器注入文本事件类型
    def text(self, text: str) -> bytes:
        """
        Send text to device
        Args:
            text: text to send
        """
        buffer = text.encode("utf-8")  # 将文本编码为 UTF-8 格式
        return struct.pack(">i", len(buffer)) + buffer  # 返回打包好的文本数据

    @inject(const.TYPE_INJECT_TOUCH_EVENT)  # 使用装饰器注入触摸事件类型
    def touch(
        self, x: int, y: int, action: int = const.ACTION_DOWN, touch_id: int = -1
    ):
        """
        Send touch event to device
        Args:
            x: x coordinate
            y: y coordinate
            action: ACTION_DOWN | ACTION_UP
            touch_id: touch id
        """
        return struct.pack(">iiii", action, x, y, touch_id)  # 返回打包好的触摸事件数据
    # 定义触摸屏幕的方法，返回一个字节流
    def touch_screen(self, x: int, y: int, action: int, touch_id: int = -1) -> bytes:
        """
        Touch screen
        Args:
            x: horizontal position
            y: vertical position
            action: ACTION_DOWN | ACTION_UP | ACTION_MOVE
            touch_id: Default using virtual id -1, you can specify it to emulate multi finger touch
        """
        # 确保 x, y 坐标不小于 0
        x, y = max(x, 0), max(y, 0)
        # 使用 struct.pack 方法打包数据，返回字节流
        return struct.pack(
            ">BqiiHHHi",
            action,
            touch_id,
            int(x),
            int(y),
            int(self.parent.resolution[0]),
            int(self.parent.resolution[1]),
            0xFFFF,
            1,
        )

    # 定义滚动屏幕的方法，返回一个字节流
    @inject(const.TYPE_INJECT_SCROLL_EVENT)
    def scroll(self, x: int, y: int, h: int, v: int) -> bytes:
        """
        Scroll screen
        Args:
            x: horizontal position
            y: vertical position
            h: horizontal movement
            v: vertical movement
        """
        # 确保 x, y 坐标不小于 0
        x, y = max(x, 0), max(y, 0)
        # 使用 struct.pack 方法打包数据，返回字节流
        return struct.pack(
            ">iiHHii",
            int(x),
            int(y),
            int(self.parent.resolution[0]),
            int(self.parent.resolution[1]),
            int(h),
            int(v),
        )

    # 定义返回或打开屏幕的方法，返回一个字节流
    @inject(const.TYPE_BACK_OR_SCREEN_ON)
    def back_or_turn_screen_on(self, action: int = const.ACTION_DOWN) -> bytes:
        """
        If the screen is off, it is turned on only on ACTION_DOWN
        Args:
            action: ACTION_DOWN | ACTION_UP
        """
        # 使用 struct.pack 方法打包数据，返回字节流
        return struct.pack(">B", action)

    # 定义展开通知面板的方法，返回一个空字节流
    @inject(const.TYPE_EXPAND_NOTIFICATION_PANEL)
    def expand_notification_panel(self) -> bytes:
        """
        Expand notification panel
        """
        return b""

    # 定义展开设置面板的方法，返回一个空字节流
    @inject(const.TYPE_EXPAND_SETTINGS_PANEL)
    def expand_settings_panel(self) -> bytes:
        """
        Expand settings panel
        """
        return b""

    # 定义折叠所有面板的方法，返回一个空字节流
    @inject(const.TYPE_COLLAPSE_PANELS)
    def collapse_panels(self) -> bytes:
        """
        Collapse all panels
        """
        return b""
    def get_clipboard(self, copy_key=const.COPY_KEY_NONE) -> str:
        """
        Get clipboard
        """
        # 获取剪贴板内容，需要通过套接字响应，无法自动注入
        s: socket.socket = self.parent.control_socket

        with self.parent.control_socket_lock:
            # 清空套接字缓冲区
            s.setblocking(False)
            while True:
                try:
                    s.recv(1024)
                except BlockingIOError:
                    break
            s.setblocking(True)

            # 读取数据包
            package = struct.pack(">BB", const.TYPE_GET_CLIPBOARD, copy_key)
            s.send(package)
            (code,) = struct.unpack(">B", s.recv(1))
            assert code == 0
            (length,) = struct.unpack(">i", s.recv(4))

            return s.recv(length).decode("utf-8")

    @inject(const.TYPE_SET_CLIPBOARD)
    def set_clipboard(self, text: str, paste: bool = False) -> bytes:
        """
        Set clipboard
        Args:
            text: 要设置的字符串
            paste: 是否立即粘贴
        """
        buffer = text.encode("utf-8")
        return struct.pack(">?i", paste, len(buffer)) + buffer

    @inject(const.TYPE_SET_SCREEN_POWER_MODE)
    def set_screen_power_mode(self, mode: int = const.POWER_MODE_NORMAL) -> bytes:
        """
        设置屏幕电源模式
        Args:
            mode: POWER_MODE_OFF | POWER_MODE_NORMAL
        """
        return struct.pack(">b", mode)

    @inject(const.TYPE_ROTATE_DEVICE)
    def rotate_device(self) -> bytes:
        """
        旋转设备
        """
        return b""

    def swipe(
        self,
        start_x: int,
        start_y: int,
        end_x: int,
        end_y: int,
        move_step_length: int = 5,
        move_steps_delay: float = 0.005,
    # 定义一个方法，用于在屏幕上滑动操作
    def swipe(self, start_x: int, start_y: int, end_x: int, end_y: int, move_step_length: int, move_steps_delay: float) -> None:
        """
        Swipe on screen
        Args:
            start_x: start horizontal position
            start_y: start vertical position
            end_x: start horizontal position
            end_y: end vertical position
            move_step_length: length per step
            move_steps_delay: sleep seconds after each step
        :return:
        """

        # 在起始位置进行按下操作
        self.touch(start_x, start_y, const.ACTION_DOWN)
        next_x = start_x
        next_y = start_y

        # 如果结束位置超出屏幕水平分辨率，则将结束位置设置为屏幕水平分辨率
        if end_x > self.parent.resolution[0]:
            end_x = self.parent.resolution[0]

        # 如果结束位置超出屏幕垂直分辨率，则将结束位置设置为屏幕垂直分辨率
        if end_y > self.parent.resolution[1]:
            end_y = self.parent.resolution[1]

        # 根据起始位置和结束位置的关系确定 x 轴和 y 轴的移动方向
        decrease_x = True if start_x > end_x else False
        decrease_y = True if start_y > end_y else False
        while True:
            # 根据移动方向和步长更新下一个 x 轴位置
            if decrease_x:
                next_x -= move_step_length
                if next_x < end_x:
                    next_x = end_x
            else:
                next_x += move_step_length
                if next_x > end_x:
                    next_x = end_x

            # 根据移动方向和步长更新下一个 y 轴位置
            if decrease_y:
                next_y -= move_step_length
                if next_y < end_y:
                    next_y = end_y
            else:
                next_y += move_step_length
                if next_y > end_y:
                    next_y = end_y

            # 在更新后的位置进行移动操作
            self.touch(next_x, next_y, const.ACTION_MOVE)

            # 如果已经到达结束位置，则进行抬起操作并结束循环
            if next_x == end_x and next_y == end_y:
                self.touch(next_x, next_y, const.ACTION_UP)
                break
            # 等待一定时间
            sleep(move_steps_delay)

    # 定义一个方法，用于在屏幕上进行点击操作
    def tap(self, x, y, hold_time: float = 0.07) -> None:
        """
        Tap on screen
        Args:
            x: horizontal position
            y: vertical position
            hold_time: hold time
        """
        # 在指定位置进行按下操作
        self.touch(x, y, const.ACTION_DOWN)
        # 等待一定时间
        sleep(hold_time)
        # 在指定位置进行抬起操作
        self.touch(x, y, const.ACTION_UP)
```