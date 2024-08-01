# `.\DB-GPT-src\dbgpt\app\scene\chat_knowledge\v1\__init__.py`

```py
# 导入所需的模块：os 模块提供了与操作系统交互的功能，shutil 模块提供了高级的文件操作功能
import os
import shutil

# 定义一个函数 move_files，接收两个参数：source_dir 和 dest_dir
def move_files(source_dir, dest_dir):
    # 如果目标文件夹不存在，则创建它
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    
    # 遍历源文件夹中的所有文件和文件夹
    for item in os.listdir(source_dir):
        # 构建源文件或文件夹的完整路径
        source = os.path.join(source_dir, item)
        # 构建目标文件或文件夹的完整路径
        destination = os.path.join(dest_dir, item)
        # 如果当前项是文件夹，则递归调用 move_files 函数来移动文件夹及其内容
        if os.path.isdir(source):
            move_files(source, destination)
        # 如果当前项是文件，则使用 shutil.move 函数移动文件到目标文件夹
        else:
            shutil.move(source, destination)
```