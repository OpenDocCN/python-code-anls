# `.\DB-GPT-src\dbgpt\app\static\_next\static\chunks\1134.f746dab04f840183.js`

```py
# 导入必要的模块：os 模块提供了操作系统相关的功能，re 模块用于正则表达式操作
import os
import re

# 定义函数 search_files，接收一个目录路径和一个正则表达式作为参数
def search_files(directory, regex):
    # 初始化一个空列表，用于存储符合条件的文件路径
    files = []
    # 遍历指定目录下的所有文件和子目录
    for root, _, filenames in os.walk(directory):
        # 遍历当前目录下的所有文件名
        for filename in filenames:
            # 使用正则表达式匹配文件名
            if re.search(regex, filename):
                # 如果文件名匹配成功，将文件的完整路径添加到列表中
                files.append(os.path.join(root, filename))
    # 返回匹配的文件路径列表
    return files
```