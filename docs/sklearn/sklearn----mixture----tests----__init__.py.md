# `D:\src\scipysrc\scikit-learn\sklearn\mixture\tests\__init__.py`

```
# 导入必要的模块：os 和 random
import os
import random

# 定义一个函数，用于从指定目录中随机选择一个文件
def pick_random_file(directory):
    # 获取指定目录下的所有文件列表
    files = os.listdir(directory)
    # 从文件列表中随机选择一个文件名
    file = random.choice(files)
    # 返回随机选择的文件名
    return file
```