# `D:\src\scipysrc\sympy\sympy\integrals\tests\__init__.py`

```
# 导入需要的模块
import os
import sys

# 定义一个函数，用于遍历指定路径下的所有文件和文件夹
def list_files(directory):
    # 初始化一个空列表，用于存储所有文件的完整路径
    files = []
    # 遍历指定路径下的所有文件和文件夹
    for root, _, filenames in os.walk(directory):
        # 遍历当前文件夹中的所有文件名
        for filename in filenames:
            # 将每个文件的完整路径添加到列表中
            files.append(os.path.join(root, filename))
    # 返回包含所有文件完整路径的列表
    return files

# 如果当前运行的 Python 版本小于 3.5，抛出一个运行时错误
if sys.version_info < (3, 5):
    raise RuntimeError("需要 Python 3.5 或更高版本")

# 调用 list_files 函数，并打印返回的文件列表
print(list_files(sys.argv[1]))
```