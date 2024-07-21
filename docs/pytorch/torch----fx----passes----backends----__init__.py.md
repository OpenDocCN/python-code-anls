# `.\pytorch\torch\fx\passes\backends\__init__.py`

```py
# 导入模块os，用于操作文件和目录
import os

# 定义函数get_filepaths，接收一个目录路径作为参数，返回该目录下所有文件的路径列表
def get_filepaths(directory):
    # 初始化一个空列表，用于存储文件路径
    file_paths = []

    # 遍历目录中的所有文件和子目录
    for root, directories, files in os.walk(directory):
        # 在当前目录下遍历所有文件
        for filename in files:
            # 将每个文件的完整路径添加到file_paths列表中
            filepath = os.path.join(root, filename)
            file_paths.append(filepath)

    # 返回所有文件路径的列表
    return file_paths
```