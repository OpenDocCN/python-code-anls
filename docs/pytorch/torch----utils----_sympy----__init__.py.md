# `.\pytorch\torch\utils\_sympy\__init__.py`

```py
# 导入必要的模块：os 模块用于操作文件系统，sys 模块用于访问系统相关信息
import os
import sys

# 定义一个名为 list_files 的函数，用于列出指定目录下的所有文件和文件夹
def list_files(directory):
    # 初始化一个空列表，用于存储所有文件和文件夹的名称
    files = []
    # 遍历指定目录下的所有文件和文件夹
    for filename in os.listdir(directory):
        # 拼接文件的完整路径
        filepath = os.path.join(directory, filename)
        # 判断当前路径是否为文件
        if os.path.isfile(filepath):
            # 如果是文件，则将文件名添加到列表中
            files.append(filename)
        # 如果当前路径是文件夹，则显示提示信息
        else:
            print(f"Ignoring directory: {filename}")
    # 返回存储文件名的列表
    return files

# 获取当前脚本所在目录
script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))

# 调用 list_files 函数，列出当前脚本所在目录下的所有文件和文件夹
script_files = list_files(script_dir)
```