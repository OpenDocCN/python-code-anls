# `ZeroNet\src\Debug\DebugReloader.py`

```py
# 导入日志、时间和操作系统模块
import logging
import time
import os

# 从 Config 模块导入 config 对象
from Config import config

# 如果处于调试模式并且执行动作是 "main"
if config.debug and config.action == "main":
    try:
        # 尝试导入 watchdog 模块
        import watchdog
        import watchdog.observers
        import watchdog.events
        # 记录调试信息：检测到 Watchdog 文件系统监听器，启用源代码自动重载
        logging.debug("Watchdog fs listener detected, source code autoreload enabled")
        # 设置 enabled 变量为 True
        enabled = True
    except Exception as err:
        # 记录调试信息：无法加载 Watchdog 文件系统监听器
        logging.debug("Watchdog fs listener could not be loaded: %s" % err)
        # 设置 enabled 变量为 False
        enabled = False
else:
    # 设置 enabled 变量为 False
    enabled = False

# 定义 DebugReloader 类
class DebugReloader:
    # 初始化方法
    def __init__(self, paths=None):
        # 如果未提供路径，则设置默认路径
        if not paths:
            paths = ["src", "plugins", config.data_dir + "/__plugins__"]
        # 获取名为 "DebugReloader" 的日志记录器
        self.log = logging.getLogger("DebugReloader")
        # 设置最后修改时间为 0
        self.last_chaged = 0
        # 初始化回调函数列表
        self.callbacks = []
        # 如果 enabled 变量为 True
        if enabled:
            # 创建 watchdog 观察者对象
            self.observer = watchdog.observers.Observer()
            # 创建文件系统事件处理程序对象
            event_handler = watchdog.events.FileSystemEventHandler()
            # 设置文件修改和删除事件的处理方法为 onChanged
            event_handler.on_modified = event_handler.on_deleted = self.onChanged
            # 设置文件创建和移动事件的处理方法为 onChanged
            event_handler.on_created = event_handler.on_moved = self.onChanged
            # 遍历路径列表
            for path in paths:
                # 如果路径不存在，则跳过
                if not os.path.isdir(path):
                    continue
                # 记录调试信息：添加自动重载路径
                self.log.debug("Adding autoreload: %s" % path)
                # 将事件处理程序对象和路径添加到观察者对象中
                self.observer.schedule(event_handler, path, recursive=True)
            # 启动观察者对象
            self.observer.start()

    # 添加回调函数到回调函数列表
    def addCallback(self, f):
        self.callbacks.append(f)
    # 当文件系统中的文件发生变化时触发的回调函数
    def onChanged(self, evt):
        # 获取文件路径
        path = evt.src_path
        # 获取文件扩展名
        ext = path.rsplit(".", 1)[-1]
        # 如果文件扩展名不是"py"或"json"，或者文件名中包含"Test"，或者距离上次修改时间不足1秒，则返回False
        if ext not in ["py", "json"] or "Test" in path or time.time() - self.last_chaged < 1.0:
            return False
        # 更新上次修改时间
        self.last_chaged = time.time()
        # 如果路径指向一个文件
        if os.path.isfile(path):
            # 获取文件的修改时间
            time_modified = os.path.getmtime(path)
        else:
            time_modified = 0
        # 记录文件变化的日志信息
        self.log.debug("File changed: %s reloading source code (modified %.3fs ago)" % (evt, time.time() - time_modified))
        # 如果距离文件修改时间超过5秒，则返回False，可能只是属性变化，忽略
        if time.time() - time_modified > 5:
            return False

        # 等待0.1秒，等待锁释放
        time.sleep(0.1)
        # 遍历回调函数列表，依次执行回调函数
        for callback in self.callbacks:
            try:
                callback()
            except Exception as err:
                # 记录异常信息
                self.log.exception(err)

    # 停止自动重新加载观察者
    def stop(self):
        if enabled:
            # 停止观察者
            self.observer.stop()
            # 记录停止观察者的日志信息
            self.log.debug("Stopped autoreload observer")
# 创建一个调试重新加载器对象
watcher = DebugReloader()
```