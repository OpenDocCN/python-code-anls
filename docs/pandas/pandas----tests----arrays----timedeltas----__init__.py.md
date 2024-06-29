# `D:\src\scipysrc\pandas\pandas\tests\arrays\timedeltas\__init__.py`

```
# 导入所需的模块：os 和 shutil
import os
import shutil

# 定义一个函数 move_files，接受两个参数 src_dir 和 dest_dir
def move_files(src_dir, dest_dir):
    # 使用 os 模块的 listdir 函数列出源目录 src_dir 中的所有文件和目录
    files = os.listdir(src_dir)
    
    # 遍历列出的每一个文件或目录
    for file in files:
        # 构建源文件的完整路径
        src_file = os.path.join(src_dir, file)
        # 如果当前文件是一个普通文件（非目录）
        if os.path.isfile(src_file):
            # 构建目标文件的完整路径
            dest_file = os.path.join(dest_dir, file)
            # 使用 shutil 模块的 move 函数移动源文件到目标文件
            shutil.move(src_file, dest_file)
```