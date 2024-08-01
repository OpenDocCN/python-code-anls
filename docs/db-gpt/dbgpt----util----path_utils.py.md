# `.\DB-GPT-src\dbgpt\util\path_utils.py`

```py
# 导入操作系统模块
import os

# 检查给定文件名的路径是否存在
def has_path(filename):
    # 获取文件名所在的目录路径
    directory = os.path.dirname(filename)
    # 返回目录路径是否存在的布尔值，若目录存在则返回 True，否则返回 False
    return bool(directory)
```