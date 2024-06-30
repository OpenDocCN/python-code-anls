# `D:\src\scipysrc\scikit-learn\sklearn\datasets\tests\data\openml\id_1590\__init__.py`

```
# 导入所需的模块
import os
import sys

# 定义一个函数，接收一个目录路径作为参数
def list_files(directory):
    # 初始化一个空列表，用于存储找到的文件名
    files = []
    # 遍历指定目录下的所有文件和子目录
    for root, _, filenames in os.walk(directory):
        # 将当前目录下的所有文件添加到列表中
        for filename in filenames:
            # 构建文件的完整路径
            filepath = os.path.join(root, filename)
            # 将完整路径添加到文件列表中
            files.append(filepath)
    # 返回包含所有文件路径的列表
    return files

# 调用 list_files 函数并打印结果
print(list_files(sys.argv[1]))
```