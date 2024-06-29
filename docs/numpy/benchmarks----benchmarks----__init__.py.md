# `.\numpy\benchmarks\benchmarks\__init__.py`

```
# 导入名为 common 的当前目录下的模块
from . import common
# 导入 sys 和 os 模块
import sys
import os

# 定义函数 show_cpu_features
def show_cpu_features():
    # 从 numpy 库中导入 _opt_info 函数
    from numpy.lib._utils_impl import _opt_info
    # 调用 _opt_info 函数获取信息
    info = _opt_info()
    # 构造 info 字符串，描述 NumPy 的 CPU 特性信息
    info = "NumPy CPU features: " + (info if info else 'nothing enabled')
    # 检查是否在环境变量中找到 'SHELL'，并且不在 Windows 平台上
    if 'SHELL' in os.environ and sys.platform != 'win32':
        # 在终端中输出带有黄色前景色的 info 字符串
        print(f"\033[33m{info}\033[0m")
    else:
        # 在终端中输出 info 字符串
        print(info)

# 定义函数 dirty_lock
def dirty_lock(lock_name, lock_on_count=1):
    # 检查当前操作系统是否具有 getppid 函数
    if not hasattr(os, "getppid"):
        return False
    # 获取当前进程的父进程 ID
    ppid = os.getppid()
    # 如果没有父进程或者父进程 ID 等于当前进程 ID，则返回 False
    if not ppid or ppid == os.getpid():
        return False
    # 构造锁文件的路径
    lock_path = os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "env", lock_name)
    )
    # 尝试在锁文件中进行加锁操作
    try:
        with open(lock_path, 'a+') as f:
            f.seek(0)
            # 读取锁文件中的计数和父进程 ID
            count, _ppid = (f.read().split() + [0, 0])[:2]
            count, _ppid = int(count), int(_ppid)
            # 如果父进程 ID 相同，则进行锁计数逻辑
            if _ppid == ppid:
                if count >= lock_on_count:
                    return True
                count += 1
            else:
                count = 0
            # 清空并更新锁文件内容为新的计数和父进程 ID
            f.seek(0)
            f.truncate()
            f.write(f"{str(count)} {str(ppid)}")
    except OSError:
        pass
    return False

# 如果未能获取到名为 "print_cpu_features.lock" 的脏锁，则调用 show_cpu_features 函数
if not dirty_lock("print_cpu_features.lock"):
    show_cpu_features()
```