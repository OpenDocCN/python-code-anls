# `.\DB-GPT-src\dbgpt\app\static\_next\static\chunks\1147.71f2b582039c2418.js`

```py
# 导入必要的模块：os 模块用于操作系统相关功能，shutil 模块提供高级文件操作功能
import os
import shutil

# 定义函数 move_files，接受源目录和目标目录作为参数
def move_files(source_dir, dest_dir):
    # 遍历源目录中的所有文件和子目录
    for root, dirs, files in os.walk(source_dir):
        # 遍历当前目录下的所有文件
        for file in files:
            # 拼接文件的完整路径
            src_path = os.path.join(root, file)
            # 构造目标路径，使用 shutil.move 函数将文件移动到目标目录
            shutil.move(src_path, dest_dir)

# 调用 move_files 函数，将当前目录下的所有文件移动到目录 '/tmp/target/'
move_files('.', '/tmp/target/')
```