# `D:\src\scipysrc\pandas\pandas\tests\scalar\timedelta\methods\__init__.py`

```
# 导入必要的模块：os 模块用于文件操作，shutil 模块用于高级文件操作
import os
import shutil

# 定义函数 deftree，用于递归地打印指定目录的目录结构
def deftree(root_dir):
    # 遍历 root_dir 目录下的所有文件和子目录
    for root, dirs, files in os.walk(root_dir):
        # 打印当前目录路径
        print(root)
        # 遍历当前目录下的所有子目录
        for d in dirs:
            # 打印子目录相对于 root_dir 的路径
            print(os.path.join(root, d))
        # 遍历当前目录下的所有文件
        for f in files:
            # 打印文件相对于 root_dir 的路径
            print(os.path.join(root, f))

# 调用函数 deftree，打印当前目录（'.' 表示当前目录）的目录结构
deftree('.')
```