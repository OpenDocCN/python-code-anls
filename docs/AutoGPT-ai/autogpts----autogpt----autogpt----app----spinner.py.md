# `.\AutoGPT\autogpts\autogpt\autogpt\app\spinner.py`

```py
# 导入所需的模块
import itertools
import sys
import threading
import time

# 定义一个 Spinner 类
class Spinner:
    """A simple spinner class"""

    # 初始化 Spinner 类
    def __init__(
        self,
        message: str = "Loading...",
        delay: float = 0.1,
        plain_output: bool = False,
    ) -> None:
        """Initialize the spinner class

        Args:
            message (str): The message to display.
            delay (float): The delay between each spinner update.
            plain_output (bool): Whether to display the spinner or not.
        """
        # 设置属性
        self.plain_output = plain_output
        self.spinner = itertools.cycle(["-", "/", "|", "\\"])
        self.delay = delay
        self.message = message
        self.running = False
        self.spinner_thread = None

    # 启动旋转
    def spin(self) -> None:
        """Spin the spinner"""
        # 如果是普通输出模式，直接打印消息并返回
        if self.plain_output:
            self.print_message()
            return
        # 在运行状态下不断打印消息并休眠一段时间
        while self.running:
            self.print_message()
            time.sleep(self.delay)

    # 打印消息
    def print_message(self):
        sys.stdout.write(f"\r{' ' * (len(self.message) + 2)}\r")
        sys.stdout.write(f"{next(self.spinner)} {self.message}\r")
        sys.stdout.flush()

    # 启动旋转
    def start(self):
        self.running = True
        self.spinner_thread = threading.Thread(target=self.spin)
        self.spinner_thread.start()

    # 停止旋转
    def stop(self):
        self.running = False
        if self.spinner_thread is not None:
            self.spinner_thread.join()
        sys.stdout.write(f"\r{' ' * (len(self.message) + 2)}\r")
        sys.stdout.flush()

    # 进入上下文管理器时启动旋转
    def __enter__(self):
        """Start the spinner"""
        self.start()
        return self

    # 退出上下文管理器时停止旋转
    def __exit__(self, exc_type, exc_value, exc_traceback) -> None:
        """Stop the spinner

        Args:
            exc_type (Exception): The exception type.
            exc_value (Exception): The exception value.
            exc_traceback (Exception): The exception traceback.
        """
        self.stop()
```