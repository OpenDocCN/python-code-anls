# `.\numpy\numpy\testing\_private\__init__.py`

```py
# 导入需要使用的模块：os（操作系统接口）、shutil（高级文件操作模块）
import os
import shutil

# 定义函数 move_files，接收源目录和目标目录作为参数
def move_files(source_dir, target_dir):
    # 获取源目录下的所有文件和子目录的列表
    files = os.listdir(source_dir)
    # 遍历源目录下的每一个文件或子目录
    for file in files:
        # 构建源文件或目录的完整路径
        src_file = os.path.join(source_dir, file)
        # 构建目标文件或目录的完整路径
        tgt_file = os.path.join(target_dir, file)
        # 如果当前遍历到的是一个目录，则使用 shutil.move() 移动整个目录到目标目录
        if os.path.isdir(src_file):
            shutil.move(src_file, tgt_file)
        # 如果当前遍历到的是一个文件，则使用 shutil.move() 移动文件到目标目录
        else:
            shutil.move(src_file, target_dir)
```