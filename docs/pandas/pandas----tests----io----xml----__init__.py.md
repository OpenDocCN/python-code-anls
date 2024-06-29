# `D:\src\scipysrc\pandas\pandas\tests\io\xml\__init__.py`

```
# 导入必要的模块：os（操作系统接口）、sys（系统参数和函数）、json（处理 JSON 格式数据）
import os
import sys
import json

# 定义函数：根据给定路径获取目录下所有文件名列表
def get_files(path):
    # 使用 os 模块的 listdir 函数获取指定路径下所有文件和目录的列表
    files = os.listdir(path)
    # 返回获取到的文件列表
    return files

# 主程序入口
if __name__ == "__main__":
    # 获取命令行参数列表
    args = sys.argv
    # 如果没有传入参数，输出提示信息并退出程序
    if len(args) < 2:
        print("Usage: python script.py <directory>")
        sys.exit(1)
    
    # 获取目录路径
    directory = args[1]
    # 调用函数获取目录下所有文件名列表
    files = get_files(directory)
    # 将文件列表转换为 JSON 格式字符串
    json_data = json.dumps(files)
    # 打印 JSON 格式字符串
    print(json_data)
```