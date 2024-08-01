# `.\DB-GPT-src\dbgpt\app\static\_next\static\ob-workers\oracle.js`

```py
# 导入所需模块：os 模块提供与操作系统交互的功能，sys 模块提供系统相关的功能
import os
import sys

# 定义函数：遍历指定目录下的所有文件和子目录，并返回一个包含所有文件的列表
def list_files(directory):
    # 初始化空列表，用于存储所有找到的文件路径
    files = []
    # 遍历指定目录下的所有文件和子目录
    for root, dirs, filenames in os.walk(directory):
        # 将当前目录下的所有文件路径加入到 files 列表中
        for filename in filenames:
            files.append(os.path.join(root, filename))
    # 返回包含所有文件路径的列表
    return files

# 打印出调用该脚本时所传递的所有参数（除去脚本名本身）
print("Arguments:", sys.argv[1:])
```