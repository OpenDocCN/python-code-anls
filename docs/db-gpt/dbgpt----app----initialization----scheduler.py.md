# `.\DB-GPT-src\dbgpt\app\initialization\scheduler.py`

```py
import logging  # 导入日志模块
import threading  # 导入多线程模块
import time  # 导入时间模块

import schedule  # 导入调度模块

from dbgpt.component import BaseComponent, SystemApp  # 导入自定义模块

logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器对象


class DefaultScheduler(BaseComponent):
    """默认调度器类"""

    name = "dbgpt_default_scheduler"  # 设置调度器名称

    def __init__(
        self,
        system_app: SystemApp,
        scheduler_delay_ms: int = 5000,
        scheduler_interval_ms: int = 1000,
    ):
        super().__init__(system_app)
        self.system_app = system_app
        self._scheduler_interval_ms = scheduler_interval_ms  # 设置调度间隔时间（毫秒）
        self._scheduler_delay_ms = scheduler_delay_ms  # 设置调度延迟时间（毫秒）
        self._stop_event = threading.Event()  # 创建一个线程事件对象用于控制调度器停止

    def init_app(self, system_app: SystemApp):
        self.system_app = system_app  # 初始化应用系统对象

    def after_start(self):
        thread = threading.Thread(target=self._scheduler)  # 创建一个线程，目标函数为 _scheduler 方法
        thread.start()  # 启动线程
        self._stop_event.clear()  # 清除停止事件，确保调度器可以运行

    def before_stop(self):
        self._stop_event.set()  # 设置停止事件，通知调度器停止运行

    def _scheduler(self):
        time.sleep(self._scheduler_delay_ms / 1000)  # 延迟一定时间，单位为秒
        while not self._stop_event.is_set():  # 当停止事件未设置时循环执行
            try:
                schedule.run_pending()  # 执行调度模块的待定任务
            except Exception as e:
                logger.debug(f"Scheduler error: {e}")  # 捕获异常并记录调试信息
            finally:
                time.sleep(self._scheduler_interval_ms / 1000)  # 按设定的间隔时间再次休眠
```