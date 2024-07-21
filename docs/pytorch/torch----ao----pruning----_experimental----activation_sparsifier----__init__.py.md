# `.\pytorch\torch\ao\pruning\_experimental\activation_sparsifier\__init__.py`

```py
# 导入需要使用的模块：os 模块用于操作文件系统，re 模块用于正则表达式的匹配
import os
import re

# 定义一个函数，用于遍历指定目录下的所有文件和子目录
def list_files(directory):
    # 初始化一个空列表，用于存储所有找到的文件路径
    files = []
    # 遍历指定目录下的所有文件和子目录
    for dirpath, _, filenames in os.walk(directory):
        # 遍历当前目录下的所有文件名
        for filename in filenames:
            # 构造文件的完整路径
            filepath = os.path.join(dirpath, filename)
            # 使用正则表达式匹配文件名是否以 ".txt" 结尾，并且不包含 "tmp" 字符串
            if re.search(r'\.txt$', filename) and 'tmp' not in filename:
                # 如果匹配成功，则将文件路径添加到列表中
                files.append(filepath)
    # 返回所有找到的符合条件的文件路径列表
    return files
```