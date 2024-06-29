# `D:\src\scipysrc\pandas\pandas\tests\tseries\offsets\__init__.py`

```
# 导入所需模块：os（操作系统接口）、shutil（高级文件操作工具）
import os
import shutil

# 定义函数 copy_files，接收两个参数：source_dir（源目录路径）、dest_dir（目标目录路径）
def copy_files(source_dir, dest_dir):
    # 使用 os 模块的 listdir 函数列出源目录下的所有文件和目录，并遍历它们
    for item in os.listdir(source_dir):
        # 构建源文件或目录的完整路径
        s = os.path.join(source_dir, item)
        # 构建目标文件或目录的完整路径
        d = os.path.join(dest_dir, item)
        # 如果当前项是一个文件并且不是一个链接文件（即非符号链接）
        if os.path.isfile(s) and not os.path.islink(s):
            # 复制文件从源路径到目标路径
            shutil.copy2(s, d)
        # 如果当前项是一个目录并且不是一个链接目录
        elif os.path.isdir(s) and not os.path.islink(s):
            # 使用 shutil 模块的 copytree 函数递归地复制目录及其内容到目标路径
            shutil.copytree(s, d)
```