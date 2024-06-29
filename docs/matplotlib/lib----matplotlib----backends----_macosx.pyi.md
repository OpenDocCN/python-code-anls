# `D:\src\scipysrc\matplotlib\lib\matplotlib\backends\_macosx.pyi`

```py
# 导入所需的模块：os 模块用于操作文件系统，sys 模块用于处理系统相关的功能
import os
import sys

# 定义一个函数 create_directory，接收一个目录路径参数 dirname
def create_directory(dirname):
    # 如果目录不存在
    if not os.path.exists(dirname):
        # 创建目录
        os.mkdir(dirname)
        # 打印提示消息，指示已创建目录
        print(f"Directory {dirname} created.")
    else:
        # 否则，如果目录已存在，打印提示消息，指示目录已经存在
        print(f"Directory {dirname} already exists.")

# 调用函数 create_directory，传入参数为当前工作目录的名称
create_directory(os.getcwd())
```