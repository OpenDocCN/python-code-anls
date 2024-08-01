# `.\DB-GPT-src\dbgpt\app\static\_next\static\chunks\6241.6a2e06c38a4a022b.js`

```py
# 导入需要使用的模块：os 模块用于操作文件系统，re 模块用于正则表达式处理
import os
import re

# 定义一个函数，用于遍历指定路径下的所有文件和文件夹
def list_files(directory):
    # 初始化一个空列表，用于存储所有找到的文件的完整路径
    files = []
    # 遍历指定目录下的所有项目，包括文件和文件夹
    for root, directories, filenames in os.walk(directory):
        # 使用正则表达式过滤掉隐藏文件夹（以点开头的文件夹）
        directories[:] = [d for d in directories if not re.match(r'\.', d)]
        # 将当前目录下的所有文件添加到 files 列表中
        for filename in filenames:
            # 拼接文件的完整路径，并添加到 files 列表中
            files.append(os.path.join(root, filename))
    # 返回存储所有文件路径的列表
    return files
```