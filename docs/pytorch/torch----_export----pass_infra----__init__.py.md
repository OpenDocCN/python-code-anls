# `.\pytorch\torch\_export\pass_infra\__init__.py`

```py
# 导入必要的模块：os模块用于操作系统相关功能，sys模块提供了与Python解释器交互的变量和函数
import os
import sys

# 定义一个函数，接收一个目录路径作为参数
def list_files(directory):
    # 初始化一个空列表，用于存储找到的文件名
    files = []
    # 遍历指定目录及其子目录下的所有文件和文件夹
    for dirpath, _, filenames in os.walk(directory):
        # 遍历当前目录中的所有文件名
        for filename in filenames:
            # 将文件的完整路径加入到列表中
            files.append(os.path.join(dirpath, filename))
    # 返回包含所有文件路径的列表
    return files

# 如果这个脚本被直接运行，则执行以下代码块
if __name__ == "__main__":
    # 如果没有提供目录作为命令行参数，则打印使用方法并退出
    if len(sys.argv) != 2:
        print("Usage: python list_files.py <directory>")
        sys.exit(1)
    
    # 从命令行参数获取要搜索的目录路径
    directory = sys.argv[1]
    
    # 调用list_files函数，获取目录下所有文件的路径列表
    files = list_files(directory)
    
    # 打印找到的所有文件路径
    for file in files:
        print(file)
```