# `D:\src\scipysrc\sympy\sympy\physics\tests\__init__.py`

```
# 导入必要的模块
import os
import sys

# 定义一个函数，参数为目录路径
def list_files(directory):
    # 初始化一个空列表，用于存储找到的文件名
    files = []
    # 遍历指定目录下的所有文件和文件夹
    for filename in os.listdir(directory):
        # 组合文件的完整路径
        filepath = os.path.join(directory, filename)
        # 判断当前路径是否为文件并且可读
        if os.path.isfile(filepath) and os.access(filepath, os.R_OK):
            # 将文件名添加到列表中
            files.append(filename)
    # 返回所有可读文件的文件名列表
    return files

# 如果脚本作为独立程序运行
if __name__ == '__main__':
    # 检查输入参数数量是否正确
    if len(sys.argv) != 2:
        # 打印使用方法
        print("Usage: python list_files.py <directory>")
        # 退出程序，返回错误状态码
        sys.exit(1)
    # 调用函数获取指定目录下的所有可读文件列表
    files = list_files(sys.argv[1])
    # 打印文件列表
    for file in files:
        print(file)
```