# `D:\src\scipysrc\sympy\sympy\sandbox\tests\__init__.py`

```
# 导入所需的模块
import os
import sys

# 定义一个函数，用于获取指定路径下的所有文件列表
def list_files(path):
    # 使用列表推导式，获取路径下所有文件的绝对路径
    return [os.path.join(path, f) for f in os.listdir(path)]

# 检查文件是否存在并打印相应的信息
if __name__ == '__main__':
    # 如果命令行参数少于2个，则打印错误信息并退出程序
    if len(sys.argv) < 2:
        print("Usage: python list_files.py <directory>")
        sys.exit(1)

    # 获取命令行参数中指定的目录路径
    directory = sys.argv[1]

    # 检查指定目录是否存在
    if not os.path.exists(directory):
        # 如果目录不存在，则打印错误信息并退出程序
        print(f"Error: '{directory}' does not exist.")
        sys.exit(1)

    # 调用 list_files 函数获取指定目录下的所有文件列表
    files = list_files(directory)

    # 打印获取到的文件列表
    for file in files:
        print(file)
```