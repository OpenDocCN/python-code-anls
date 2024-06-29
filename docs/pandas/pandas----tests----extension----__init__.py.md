# `D:\src\scipysrc\pandas\pandas\tests\extension\__init__.py`

```
# 导入必要的模块：os 模块提供了与操作系统交互的功能
import os

# 定义一个函数，用于遍历指定目录下的所有文件和子目录
def list_files(directory):
    # 初始化一个空列表，用于存储所有文件的完整路径
    files = []
    # 遍历指定目录下的所有内容，包括文件和子目录
    for root, dirs, filenames in os.walk(directory):
        # 将当前目录下的所有文件名与其完整路径拼接，并添加到列表中
        for filename in filenames:
            files.append(os.path.join(root, filename))
    # 返回包含所有文件路径的列表
    return files
```