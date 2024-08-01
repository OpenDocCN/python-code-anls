# `.\DB-GPT-src\dbgpt\app\static\_next\static\chunks\8719.94582b395ce9745b.js`

```py
# 导入必要的模块：os 模块用于文件操作，re 模块用于正则表达式匹配
import os
import re

# 定义函数：获取指定目录下所有文件的文件名列表
def get_file_names(directory):
    # 使用 os 模块的 listdir 函数列出指定目录下的所有文件和文件夹，存储在 files 变量中
    files = os.listdir(directory)
    # 使用列表推导式筛选出文件名，忽略文件夹
    file_names = [f for f in files if os.path.isfile(os.path.join(directory, f))]
    # 返回文件名列表
    return file_names

# 定义函数：根据文件名筛选出符合特定模式的文件名列表
def filter_file_names(file_names, pattern):
    # 使用 re 模块的 compile 函数编译正则表达式模式
    regex = re.compile(pattern)
    # 使用列表推导式从 file_names 中筛选出符合正则表达式模式的文件名
    filtered_names = [f for f in file_names if regex.search(f)]
    # 返回筛选后的文件名列表
    return filtered_names
```