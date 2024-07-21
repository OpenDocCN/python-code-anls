# `.\pytorch\torch\utils\benchmark\op_fuzzers\__init__.py`

```
# 导入必要的模块：os 模块提供了与操作系统交互的功能，sys 模块提供了访问与 Python 解释器相关的变量和函数
import os
import sys

# 定义一个函数 find_files，接收两个参数：root_dir 表示要搜索的根目录，ext 表示要匹配的文件扩展名
def find_files(root_dir, ext):
    # 初始化一个空列表，用于存储找到的文件路径
    file_list = []
    # 遍历 root_dir 及其子目录下的所有文件和文件夹
    for root, dirs, files in os.walk(root_dir):
        # 遍历当前目录中的所有文件
        for file in files:
            # 检查文件的扩展名是否匹配 ext 参数
            if file.endswith(ext):
                # 构造文件的完整路径并添加到 file_list 中
                file_list.append(os.path.join(root, file))
    # 返回找到的所有匹配文件的路径列表
    return file_list
```