# `.\DB-GPT-src\dbgpt\app\static\_next\static\chunks\pages\database-0428b7022de673a0.js`

```py
# 导入所需的模块
import os
import sys

# 定义一个函数，用于获取指定路径下的所有文件列表
def list_files(directory):
    # 初始化一个空列表，用于存储文件名
    files = []
    # 遍历指定路径下的所有文件和目录
    for entry in os.scandir(directory):
        # 如果当前条目是文件而不是目录
        if entry.is_file():
            # 将文件名添加到列表中
            files.append(entry.name)
    # 返回文件名列表
    return files

# 调用函数并打印结果，列出当前目录下的所有文件
print(list_files('.'))
```