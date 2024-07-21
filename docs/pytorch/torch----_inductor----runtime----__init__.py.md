# `.\pytorch\torch\_inductor\runtime\__init__.py`

```py
# 导入必要的模块：os（操作系统接口）、shutil（高级文件操作）、glob（文件名模式匹配）、random（生成伪随机数）
import os
import shutil
import glob
import random

# 定义函数 'organize_files'
def organize_files(source_dir, dest_dir):
    # 遍历源目录下的所有文件
    for filename in os.listdir(source_dir):
        # 拼接源目录和文件名，得到文件的完整路径
        source_file = os.path.join(source_dir, filename)
        # 如果文件是普通文件而不是目录
        if os.path.isfile(source_file):
            # 生成一个随机文件名
            random_name = str(random.randint(1, 1000)) + '_' + filename
            # 拼接目标目录和随机文件名，得到文件的目标路径
            dest_file = os.path.join(dest_dir, random_name)
            # 复制文件从源路径到目标路径
            shutil.copy(source_file, dest_file)
        # 如果文件是目录
        elif os.path.isdir(source_file):
            # 生成一个新的目标目录路径
            dest_subdir = os.path.join(dest_dir, filename)
            # 递归调用 'organize_files' 函数，将源子目录整理到目标子目录
            organize_files(source_file, dest_subdir)
```