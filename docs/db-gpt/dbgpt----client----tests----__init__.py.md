# `.\DB-GPT-src\dbgpt\client\tests\__init__.py`

```py
# 导入所需模块
import os
import sys

# 定义一个函数，接收一个目录路径作为参数
def list_files(directory):
    # 初始化一个空列表，用于存储文件名
    files = []
    # 遍历指定目录下的所有文件和文件夹
    for filename in os.listdir(directory):
        # 构建文件或文件夹的完整路径
        filepath = os.path.join(directory, filename)
        # 检查路径是否为文件
        if os.path.isfile(filepath):
            # 如果是文件，则将文件名添加到列表中
            files.append(filename)
    # 返回存储文件名的列表
    return files

# 主程序入口点，检查脚本是否有参数
if __name__ == "__main__":
    # 检查参数数量是否为2，如果不是，则输出提示信息并退出
    if len(sys.argv) != 2:
        print("Usage: python list_files.py directory")
        sys.exit(1)
    
    # 从命令行参数获取目录路径
    directory = sys.argv[1]
    # 调用函数获取目录下所有文件的列表
    files = list_files(directory)
    # 打印列表中的每个文件名
    for file in files:
        print(file)
```