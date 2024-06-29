# `D:\src\scipysrc\pandas\pandas\tests\scalar\timestamp\__init__.py`

```
# 导入模块：os，用于处理操作系统相关的功能
import os

# 定义函数：find_files
def find_files(directory, extension):
    # 初始化空列表，用于存储找到的文件路径
    file_list = []
    # 遍历指定目录下的所有文件和子目录
    for root, dirs, files in os.walk(directory):
        # 遍历当前目录中的所有文件
        for file in files:
            # 如果文件名以指定的扩展名结尾
            if file.endswith(extension):
                # 将文件的完整路径加入到列表中
                file_list.append(os.path.join(root, file))
    # 返回找到的所有文件路径列表
    return file_list
```