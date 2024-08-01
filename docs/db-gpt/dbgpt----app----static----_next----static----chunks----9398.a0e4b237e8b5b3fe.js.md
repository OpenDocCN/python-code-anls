# `.\DB-GPT-src\dbgpt\app\static\_next\static\chunks\9398.a0e4b237e8b5b3fe.js`

```py
# 导入必要的模块：os 模块用于操作文件系统，shutil 模块用于高级文件操作
import os
import shutil

# 定义函数 copy_files(source_dir, dest_dir) 用于复制一个目录下的所有文件到另一个目录
def copy_files(source_dir, dest_dir):
    # 如果目标目录不存在，则创建目标目录
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    # 获取源目录下的所有文件和子目录
    files = os.listdir(source_dir)
    for file in files:
        # 构建源文件的完整路径
        source_file = os.path.join(source_dir, file)
        # 构建目标文件的完整路径
        dest_file = os.path.join(dest_dir, file)
        # 如果是文件，直接复制到目标目录
        if os.path.isfile(source_file):
            shutil.copy(source_file, dest_file)
        # 如果是目录，递归地调用本函数处理子目录
        elif os.path.isdir(source_file):
            copy_files(source_file, dest_file)
```