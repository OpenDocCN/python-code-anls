# `ZeroNet\src\Debug\DebugHook.py`

```py
# 导入系统模块
import sys
# 导入日志模块
import logging
# 导入信号模块
import signal
# 导入动态导入模块
import importlib

# 导入协程模块
import gevent
# 导入协程调度模块
import gevent.hub

# 从 Config 模块中导入 config 对象
from Config import config
# 从当前目录中导入 Debug 模块
from . import Debug

# 定义全局变量 last_error
last_error = None

# 定义关闭程序的函数，传入关闭原因参数
def shutdown(reason="Unknown"):
    # 记录日志，显示关闭原因
    logging.info("Shutting down (reason: %s)..." % reason)
    # 动态导入 main 模块
    import main
    # 如果 main 模块中存在 file_server 属性
    if "file_server" in dir(main):
        try:
            # 使用协程停止 file_server
            gevent.spawn(main.file_server.stop)
            # 如果 main 模块中存在 ui_server 属性
            if "ui_server" in dir(main):
                # 使用协程停止 ui_server
                gevent.spawn(main.ui_server.stop)
        except Exception as err:
            # 打印关闭错误信息
            print("Proper shutdown error: %s" % err)
            # 退出程序
            sys.exit(0)
    else:
        # 退出程序
        sys.exit(0)

# 定义处理错误的函数，接受任意参数和关键字参数
def handleError(*args, **kwargs):
    # 声明全局变量 last_error
    global last_error
    # 如果没有传入参数
    if not args:  # Manual called
        # 获取当前异常信息
        args = sys.exc_info()
        # 设置为静默模式
        silent = True
    else:
        # 非静默模式
        silent = False
    # 如果异常类型不是 "Notify"
    if args[0].__name__ != "Notify":
        # 记录最后的错误信息
        last_error = args

    # 如果异常类型是 "KeyboardInterrupt"
    if args[0].__name__ == "KeyboardInterrupt":
        # 调用关闭程序函数，传入键盘中断原因
        shutdown("Keyboard interrupt")
    # 如果不是静默模式且异常类型不是 "Notify"
    elif not silent and args[0].__name__ != "Notify":
        # 记录异常信息到日志
        logging.exception("Unhandled exception")
        # 如果异常信息不是在 greenlet.py 中，避免重复显示错误
        if "greenlet.py" not in args[2].tb_frame.f_code.co_filename:  # Don't display error twice
            # 调用系统默认的异常处理函数
            sys.__excepthook__(*args, **kwargs)

# 忽略 "Notify" 类型的错误
def handleErrorNotify(*args, **kwargs):
    # 获取第一个参数作为错误信息
    err = args[0]
    # 如果错误类型是 "KeyboardInterrupt"
    if err.__name__ == "KeyboardInterrupt":
        # 调用关闭程序函数，传入键盘中断原因
        shutdown("Keyboard interrupt")
    # 如果错误类型不是 "Notify"
    elif err.__name__ != "Notify":
        # 记录错误信息到日志
        logging.error("Unhandled exception: %s" % Debug.formatException(args))
        # 调用系统默认的异常处理函数
        sys.__excepthook__(*args, **kwargs)

# 如果配置为调试模式
if config.debug:  # Keep last error for /Debug
    # 设置系统异常处理函数为 handleError
    sys.excepthook = handleError
else:
    # 设置系统异常处理函数为 handleErrorNotify
    sys.excepthook = handleErrorNotify

# 如果协程调度模块中存在 handle_error 属性
if "handle_error" in dir(gevent.hub.Hub):
    # 保存原始的 handle_error 函数
    gevent.hub.Hub._original_handle_error = gevent.hub.Hub.handle_error
else:
    # 记录日志，显示未找到 gevent.hub.Hub.handle_error
    logging.debug("gevent.hub.Hub.handle_error not found using old gevent hooks")
    # 保存原始的 Greenlet 类
    OriginalGreenlet = gevent.Greenlet
    # 定义一个名为ErrorhookedGreenlet的类，继承自OriginalGreenlet类
    class ErrorhookedGreenlet(OriginalGreenlet):
        # 定义_report_error方法，用于报告错误信息
        def _report_error(self, exc_info):
            # 调用sys.excepthook方法，传入异常信息
            sys.excepthook(exc_info[0], exc_info[1], exc_info[2])
    
    # 将gevent.Greenlet和gevent.greenlet.Greenlet指向ErrorhookedGreenlet类
    gevent.Greenlet = gevent.greenlet.Greenlet = ErrorhookedGreenlet
    # 重新加载gevent模块
    importlib.reload(gevent)
# 定义处理 Greenlet 错误的函数，接受上下文、错误类型、错误值和回溯信息作为参数
def handleGreenletError(context, type, value, tb):
    # 如果上下文是元组并且第一个元素的类名是 "ThreadPool"
    if context.__class__ is tuple and context[0].__class__.__name__ == "ThreadPool":
        # 在 ThreadPool 中的异常将在主线程中处理
        return None

    # 如果错误值是字符串
    if isinstance(value, str):
        # Cython 可能会引发值为普通字符串的错误
        # 例如，AttributeError, "_semaphore.Semaphore has no attr", <traceback>
        value = type(value)

    # 如果错误类型不是 gevent.get_hub().NOT_ERROR 的子类
    if not issubclass(type, gevent.get_hub().NOT_ERROR):
        # 使用系统默认的异常处理函数处理错误
        sys.excepthook(type, value, tb)

# 将 gevent 的错误处理函数设置为 handleGreenletError
gevent.get_hub().handle_error = handleGreenletError

# 尝试设置信号 SIGTERM 的处理函数为 shutdown("SIGTERM")
try:
    signal.signal(signal.SIGTERM, lambda signum, stack_frame: shutdown("SIGTERM"))
# 如果出现异常，记录错误信息
except Exception as err:
    logging.debug("Error setting up SIGTERM watcher: %s" % err)

# 如果当前模块是主模块
if __name__ == "__main__":
    import time
    from gevent import monkey
    # 打补丁，禁用线程和 SSL
    monkey.patch_all(thread=False, ssl=False)
    from . import Debug

    # 定义一个函数 sleeper，接受一个参数 num
    def sleeper(num):
        print("started", num)
        time.sleep(3)
        # 抛出一个异常
        raise Exception("Error")
        print("stopped", num)
    # 创建两个 Greenlet 对象，分别执行 sleeper 函数
    thread1 = gevent.spawn(sleeper, 1)
    thread2 = gevent.spawn(sleeper, 2)
    time.sleep(1)
    print("killing...")
    # 终止 thread1，抛出 Debug.Notify("Worker stopped") 异常
    thread1.kill(exception=Debug.Notify("Worker stopped"))
    # 抛出 Debug.Notify("Throw") 异常
    #thread2.throw(Debug.Notify("Throw"))
    print("killed")
    # 等待所有 Greenlet 对象执行完毕
    gevent.joinall([thread1,thread2])
```