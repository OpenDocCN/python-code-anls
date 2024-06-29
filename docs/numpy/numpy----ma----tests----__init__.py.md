# `.\numpy\numpy\ma\tests\__init__.py`

```py
# 导入所需的模块：os 用于操作文件系统，re 用于正则表达式处理
import os
import re

# 定义一个函数，参数为目录路径
def find_files(directory):
    # 初始化一个空列表，用于存储符合条件的文件名
    file_list = []
    # 遍历指定目录下的所有文件和文件夹
    for root, dirs, files in os.walk(directory):
        # 使用正则表达式匹配文件名中包含数字的文件
        for file in files:
            if re.search(r'\d', file):
                # 如果文件名中包含数字，将其路径添加到列表中
                file_list.append(os.path.join(root, file))
    # 返回符合条件的文件路径列表
    return file_list
```