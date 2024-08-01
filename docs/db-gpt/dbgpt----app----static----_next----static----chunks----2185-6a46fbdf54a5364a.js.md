# `.\DB-GPT-src\dbgpt\app\static\_next\static\chunks\2185-6a46fbdf54a5364a.js`

```py
# 导入必要的模块：os 模块用于处理文件和目录，shutil 模块用于高级文件操作
import os
import shutil

# 定义函数 move_files，接收源目录和目标目录两个参数
def move_files(source_dir, dest_dir):
    # 获取源目录下所有文件和目录的列表
    files = os.listdir(source_dir)
    # 遍历每个文件或目录
    for f in files:
        # 构建源文件或目录的完整路径
        source = os.path.join(source_dir, f)
        # 构建目标文件或目录的完整路径
        destination = os.path.join(dest_dir, f)
        # 判断当前路径是否为文件夹
        if os.path.isdir(source):
            # 如果是文件夹，则使用 shutil.move 函数移动文件夹到目标路径
            shutil.move(source, destination)
        else:
            # 如果是文件，则使用 shutil.move 函数移动文件到目标路径
            shutil.move(source, destination)

# 调用 move_files 函数，将源目录 'source_folder' 下的所有文件和目录移动到目标目录 'destination_folder'
move_files('source_folder', 'destination_folder')
```