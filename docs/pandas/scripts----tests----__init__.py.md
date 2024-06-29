# `D:\src\scipysrc\pandas\scripts\tests\__init__.py`

```
# 导入必要的模块：os（操作系统相关功能）、re（正则表达式操作）
import os
import re

# 定义函数 `find_files`
# 该函数接收两个参数：`dir`（目录路径）和 `pattern`（用于匹配文件名的正则表达式）
def find_files(dir, pattern):
    # 遍历指定目录下的所有文件和子目录
    for root, dirs, files in os.walk(dir):
        # 遍历当前目录下的所有文件
        for basename in files:
            # 如果文件名符合指定的正则表达式
            if re.match(pattern, basename):
                # 拼接当前文件的完整路径
                filename = os.path.join(root, basename)
                # 打印找到的文件路径
                print(filename)

# 调用 `find_files` 函数，查找指定目录下所有以 `.txt` 结尾的文件
find_files('.', r'.*\.txt$')
```