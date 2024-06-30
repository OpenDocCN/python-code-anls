# `D:\src\scipysrc\sympy\sympy\holonomic\tests\__init__.py`

```
# 导入所需的模块：os 模块提供了与操作系统交互的功能，re 模块提供了正则表达式操作功能
import os
import re

# 定义函数 find_files，接收一个目录路径和一个正则表达式作为参数
def find_files(directory, pattern):
    # 使用 os.walk 函数遍历目录及其子目录中的文件
    for root, dirs, files in os.walk(directory):
        # 遍历当前目录中的每个文件名
        for basename in files:
            # 使用正则表达式匹配文件名
            if re.match(pattern, basename):
                # 使用 yield 返回匹配的文件的完整路径
                yield os.path.join(root, basename)
```