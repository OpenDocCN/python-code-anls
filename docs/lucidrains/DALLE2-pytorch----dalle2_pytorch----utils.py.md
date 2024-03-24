# `.\lucidrains\DALLE2-pytorch\dalle2_pytorch\utils.py`

```py
# 导入时间模块
import time
# 导入 importlib 模块
import importlib

# 辅助函数

# 检查值是否存在
def exists(val):
    return val is not None

# 时间辅助函数

# 计时器类
class Timer:
    def __init__(self):
        self.reset()

    # 重置计时器
    def reset(self):
        self.last_time = time.time()

    # 返回经过的时间
    def elapsed(self):
        return time.time() - self.last_time

# 打印辅助函数

# 打印带边框的字符串
def print_ribbon(s, symbol='=', repeat=40):
    flank = symbol * repeat
    return f'{flank} {s} {flank}'

# 导入辅助函数

# 尝试导入指定模块，如果失败则打印错误信息并退出程序
def import_or_print_error(pkg_name, err_str=None):
    try:
        return importlib.import_module(pkg_name)
    except ModuleNotFoundError as e:
        if exists(err_str):
            print(err_str)
        exit()
```