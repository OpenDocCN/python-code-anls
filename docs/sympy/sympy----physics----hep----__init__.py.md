# `D:\src\scipysrc\sympy\sympy\physics\hep\__init__.py`

```
# 导入所需的模块
import os
import sys

# 定义一个函数，接收一个参数作为目录路径
def list_files(directory):
    # 初始化一个空列表，用于存储找到的文件名
    files = []
    # 遍历指定目录下的所有文件和文件夹
    for root, dirs, filenames in os.walk(directory):
        # 将找到的文件名添加到列表中
        files.extend(filenames)
    # 返回包含所有文件名的列表
    return files

# 如果脚本作为独立的执行文件运行
if __name__ == "__main__":
    # 调用函数，获取当前工作目录下的所有文件名
    files = list_files(os.getcwd())
    # 打印文件名列表
    print(files)
```