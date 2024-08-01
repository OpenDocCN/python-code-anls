# `.\DB-GPT-src\dbgpt\serve\core\tests\__init__.py`

```py
# 导入必要的模块
import os
import sys
from collections import defaultdict

# 定义一个函数，接受一个参数作为目录路径
def find_duplicates(directory):
    # 创建一个空字典，用于存储文件名和对应的文件路径列表
    files = defaultdict(list)
    
    # 遍历指定目录下的所有文件和子目录
    for dirpath, dirnames, filenames in os.walk(directory):
        # 遍历当前目录下的所有文件名
        for filename in filenames:
            # 获取文件的完整路径
            full_path = os.path.join(dirpath, filename)
            # 获取文件大小
            file_size = os.path.getsize(full_path)
            # 将文件大小和文件路径添加到字典中
            files[(filename, file_size)].append(full_path)
    
    # 返回所有重复文件的字典，其中键是文件名和大小，值是路径列表
    return {key: paths for key, paths in files.items() if len(paths) > 1}
```