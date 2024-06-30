# `D:\src\scipysrc\sympy\sympy\polys\benchmarks\__init__.py`

```
# 导入必要的模块：os（操作系统接口）、shutil（高级文件操作）、sys（系统特定的参数和函数）
import os
import shutil
import sys

# 定义函数 copy_files，接收两个参数：source_dir（源目录路径）和 dest_dir（目标目录路径）
def copy_files(source_dir, dest_dir):
    # 如果目标目录不存在，则创建目标目录
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    # 遍历源目录中的所有文件和子目录
    for item in os.listdir(source_dir):
        # 构建源文件或目录的完整路径
        source_item = os.path.join(source_dir, item)
        # 构建目标文件或目录的完整路径
        dest_item = os.path.join(dest_dir, item)
        
        # 如果当前项目是一个文件
        if os.path.isfile(source_item):
            # 使用 shutil 模块复制文件到目标目录
            shutil.copy(source_item, dest_item)
        # 如果当前项目是一个目录
        elif os.path.isdir(source_item):
            # 递归调用当前函数，复制整个子目录
            copy_files(source_item, dest_item)

# 调用 copy_files 函数，从源目录 '/path/to/source' 复制到目标目录 '/path/to/destination'
copy_files('/path/to/source', '/path/to/destination')
```