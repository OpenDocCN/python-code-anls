# `.\pytorch\tools\jit\test\__init__.py`

```py
# 导入所需的模块
import os
import sys

# 定义一个名为find_files的函数，接收目录路径dir和文件后缀suffix作为参数
def find_files(dir, suffix):
    # 使用os.walk遍历dir目录及其子目录下的所有文件和目录
    for root, dirs, files in os.walk(dir):
        # 遍历当前目录下的所有文件
        for file in files:
            # 如果文件名以给定的后缀suffix结尾
            if file.endswith(suffix):
                # 构建文件的完整路径
                filepath = os.path.join(root, file)
                # 打印文件的完整路径
                print(filepath)

# 调用find_files函数，传入当前目录'.'和后缀'.txt'作为参数
find_files('.', '.txt')
```