# `.\lucidrains\RETRO-pytorch\retro_pytorch\utils.py`

```
# 导入 os 模块
import os
# 导入 numpy 模块并重命名为 np
import numpy as np

# 从 pathlib 模块中导入 Path 类
from pathlib import Path
# 从 shutil 模块中导入 rmtree 函数
from shutil import rmtree
# 从 contextlib 模块中导入 contextmanager 装饰器
from contextlib import contextmanager

# 检查环境变量是否为真
def is_true_env_flag(env_flag):
    return os.getenv(env_flag, 'false').lower() in ('true', '1', 't')

# 重置文件夹
def reset_folder_(p):
    # 创建 Path 对象
    path = Path(p)
    # 删除文件夹及其内容，如果文件夹不存在则忽略错误
    rmtree(path, ignore_errors = True)
    # 创建文件夹，如果文件夹已存在则忽略
    path.mkdir(exist_ok = True, parents = True)

# 创建内存映射对象的上下文管理器
@contextmanager
def memmap(*args, **kwargs):
    # 创建内存映射对象
    pointer = np.memmap(*args, **kwargs)
    # 通过 yield 将指针传递给调用者
    yield pointer
    # 在退出上下文管理器时删除内存映射对象
    del pointer
```