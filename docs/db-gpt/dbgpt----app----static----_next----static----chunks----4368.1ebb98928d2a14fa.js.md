# `.\DB-GPT-src\dbgpt\app\static\_next\static\chunks\4368.1ebb98928d2a14fa.js`

```py
# 导入必要的模块：os 模块提供了与操作系统交互的功能，shutil 模块提供了高级文件操作功能
import os
import shutil

# 定义函数 move_files(source_dir, dest_dir)，实现将 source_dir 目录下的所有文件移动到 dest_dir 目录
def move_files(source_dir, dest_dir):
    # 获取 source_dir 目录下的所有文件列表
    files = os.listdir(source_dir)
    
    # 遍历文件列表
    for file_name in files:
        # 构建源文件的完整路径
        source_file = os.path.join(source_dir, file_name)
        # 构建目标文件的完整路径
        destination_file = os.path.join(dest_dir, file_name)
        # 如果目标文件已存在，则删除
        if os.path.exists(destination_file):
            os.remove(destination_file)
        # 移动源文件到目标文件夹
        shutil.move(source_file, dest_dir)
```