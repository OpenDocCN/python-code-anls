# `.\models\deprecated\__init__.py`

```
# 导入必要的模块：os 模块提供了与操作系统交互的功能，sys 模块提供了与 Python 解释器交互的功能
import os
import sys

# 定义函数：遍历指定路径下的所有文件和文件夹，返回它们的绝对路径列表
def list_files(directory):
    # 初始化一个空列表，用于存储所有文件的绝对路径
    files = []
    # 遍历指定路径下的所有文件和文件夹
    for dirpath, _, filenames in os.walk(directory):
        # 将每个文件的绝对路径添加到列表中
        for f in filenames:
            files.append(os.path.abspath(os.path.join(dirpath, f)))
    # 返回所有文件的绝对路径列表
    return files

# 如果脚本直接运行（而非被导入），执行以下操作
if __name__ == "__main__":
    # 如果未提供参数，则提示正确的用法
    if len(sys.argv) != 2:
        print("Usage: python list_files.py directory")
        # 退出程序，并返回非零状态码以指示错误
        sys.exit(1)
    
    # 从命令行参数获取目录路径
    directory = sys.argv[1]
    
    # 调用函数获取目录下所有文件的绝对路径列表
    files = list_files(directory)
    
    # 打印每个文件的绝对路径
    for f in files:
        print(f)
```