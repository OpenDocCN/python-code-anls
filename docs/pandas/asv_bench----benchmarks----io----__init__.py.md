# `D:\src\scipysrc\pandas\asv_bench\benchmarks\io\__init__.py`

```
# 导入需要的模块
import os
import sys

# 定义函数，接收一个参数作为文件路径
def process_directory(dir_path):
    # 初始化一个空列表，用于存储符合条件的文件名
    file_list = []
    
    # 遍历指定路径下的所有文件和文件夹
    for root, dirs, files in os.walk(dir_path):
        # 在当前路径下遍历所有文件
        for file in files:
            # 如果文件以".txt"结尾，则将其完整路径加入到列表中
            if file.endswith(".txt"):
                file_list.append(os.path.join(root, file))
    
    # 返回符合条件的文件路径列表
    return file_list

# 调用函数，传入指定路径，并将返回的文件列表赋值给变量
files = process_directory('/path/to/directory')
```