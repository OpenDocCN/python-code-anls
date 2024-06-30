# `D:\src\scipysrc\scikit-learn\sklearn\datasets\tests\__init__.py`

```
# 导入所需的模块
import os
import sys

# 定义一个名为find_files的函数，接收两个参数：dir_path为目录路径，suffix为文件后缀
def find_files(dir_path, suffix):
    # 初始化一个空列表用于存储符合条件的文件路径
    files = []
    # 遍历指定目录下的所有文件和子目录
    for root, dirs, filenames in os.walk(dir_path):
        # 遍历当前目录下的所有文件名
        for filename in filenames:
            # 如果文件名以指定后缀结尾，则将其路径加入到files列表中
            if filename.endswith(suffix):
                files.append(os.path.join(root, filename))
    # 返回包含所有符合条件的文件路径的列表
    return files

# 调用find_files函数，传入当前目录路径和'.py'作为参数，返回所有Python文件的路径列表
python_files = find_files('.', '.py')
# 打印输出所有找到的Python文件路径
print(python_files)
```