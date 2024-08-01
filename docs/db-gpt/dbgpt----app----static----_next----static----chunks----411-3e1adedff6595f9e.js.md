# `.\DB-GPT-src\dbgpt\app\static\_next\static\chunks\411-3e1adedff6595f9e.js`

```py
# 导入所需的模块：os（操作系统接口）、shutil（高级文件操作）、glob（文件名模式匹配）
import os
import shutil
import glob

# 定义一个函数，用于将指定目录下的所有文件移动到另一个目录中
def move_files(src_dir, dest_dir):
    # 使用 glob 模块查找源目录下所有的文件路径
    files = glob.glob(os.path.join(src_dir, '*'))
    # 遍历找到的每一个文件路径
    for f in files:
        # 使用 shutil 模块的 move 函数将文件移动到目标目录
        shutil.move(f, dest_dir)

# 调用 move_files 函数，将当前目录下的所有文件移动到目录 '/path/to/destination'
move_files('.', '/path/to/destination')
```