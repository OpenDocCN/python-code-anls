# `ZeroNet\src\Debug\Debug.py`

```py
# 导入系统、操作系统、正则表达式和配置模块
import sys
import os
import re
from Config import config

# 定义非致命异常类
class Notify(Exception):
    def __init__(self, message=None):
        if message:
            self.message = message

    def __str__(self):
        return self.message

# 创建自定义异常类型
def createNotifyType(message):
    return type("Notify", (Notify, ), {"message": message})

# 格式化异常消息
def formatExceptionMessage(err):
    err_type = err.__class__.__name__
    if err.args:
        err_message = err.args[-1]
    else:
        err_message = err.__str__()
    return "%s: %s" % (err_type, err_message)

# 获取 Python 库目录
python_lib_dirs = [path.replace("\\", "/") for path in sys.path if re.sub(r".*[\\/]", "", path) in ("site-packages", "dist-packages")]
python_lib_dirs.append(os.path.dirname(os.__file__).replace("\\", "/"))  # TODO: check if returns the correct path for PyPy

# 获取根目录
root_dir = os.path.realpath(os.path.dirname(__file__) + "/../../")
root_dir = root_dir.replace("\\", "/")

# 格式化回溯信息
def formatTraceback(items, limit=None, fold_builtin=True):
    back = []
    i = 0
    prev_file_title = ""
    is_prev_builtin = False
    return back

# 格式化异常
def formatException(err=None, format="text"):
    import traceback
    if type(err) == Notify:
        return err
    elif type(err) == tuple and err and err[0] is not None:  # Passed trackeback info
        exc_type, exc_obj, exc_tb = err
        err = None
    else:  # No trackeback info passed, get latest
        exc_type, exc_obj, exc_tb = sys.exc_info()

    if not err:
        if hasattr(err, "message"):
            err = exc_obj.message
        else:
            err = exc_obj

    tb = formatTraceback([[frame[0], frame[1]] for frame in traceback.extract_tb(exc_tb)])
    if format == "html":
        return "%s: %s<br><small class='multiline'>%s</small>" % (repr(err), err, " > ".join(tb))
    else:
        return "%s: %s in %s" % (exc_type.__name__, err, " > ".join(tb))

# 格式化堆栈信息
def formatStack(limit=None):
    import inspect
    # 根据调用堆栈信息格式化异常的回溯信息，限制回溯信息的深度为limit
    tb = formatTraceback([[frame[1], frame[2]] for frame in inspect.stack()[1:]], limit=limit)
    # 将格式化后的回溯信息以">"连接成字符串，并返回
    return " > ".join(tb)
# 导入日志和协程库
import logging
import gevent
import time

# 定义全局变量 num_block
num_block = 0

# 定义函数 testBlock
def testBlock():
    global num_block
    # 记录日志，表示开始检测协程是否阻塞
    logging.debug("Gevent block checker started")
    # 记录上次时间
    last_time = time.time()
    # 循环检测
    while 1:
        # 休眠1秒
        time.sleep(1)
        # 如果当前时间减去上次时间大于1.1秒
        if time.time() - last_time > 1.1:
            # 记录日志，表示检测到协程阻塞
            logging.debug("Gevent block detected: %.3fs" % (time.time() - last_time - 1))
            # num_block 加一
            num_block += 1
        # 更新上次时间
        last_time = time.time()

# 创建协程并执行 testBlock 函数
gevent.spawn(testBlock)

# 如果当前脚本为主程序
if __name__ == "__main__":
    try:
        # 抛出除零异常
        print(1 / 0)
    except Exception as err:
        # 打印异常类型
        print(type(err).__name__)
        # 打印异常信息
        print("1/0 error: %s" % formatException(err))

    # 定义函数 loadJson
    def loadJson():
        # 加载错误的 JSON 数据
        json.loads("Errr")

    # 导入 JSON 库
    import json
    try:
        # 调用 loadJson 函数
        loadJson()
    except Exception as err:
        # 打印异常信息
        print(err)
        # 打印异常信息
        print("Json load error: %s" % formatException(err))

    try:
        # 抛出 Notify 异常
        raise Notify("nothing...")
    except Exception as err:
        # 打印异常信息
        print("Notify: %s" % formatException(err))

    # 调用 loadJson 函数
    loadJson()
```