# `.\pytorch\torch\ao\quantization\pt2e\__init__.py`

```
# 导入所需模块：os 模块用于操作文件系统，re 模块用于正则表达式操作
import os
import re

# 定义一个函数，接收一个目录路径作为参数，返回一个包含所有 Python 文件名的列表
def find_python_files(directory):
    # 使用 os 模块的 listdir 方法列出指定目录下的所有文件和目录，存储在 files 变量中
    files = os.listdir(directory)
    # 使用列表推导式，筛选出 files 中以 '.py' 结尾的文件名，存储在 py_files 列表中
    py_files = [f for f in files if f.endswith('.py')]
    # 返回 py_files 列表，其中包含了指定目录下所有的 Python 文件名
    return py_files
```