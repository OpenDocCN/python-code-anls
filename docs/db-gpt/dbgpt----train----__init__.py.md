# `.\DB-GPT-src\dbgpt\train\__init__.py`

```py
# 导入所需的模块
import os
import sys
import shutil

# 定义一个函数，用于递归地复制文件夹及其内容
def copy_folder(src, dest):
    # 如果目标文件夹不存在，则创建它
    if not os.path.exists(dest):
        os.makedirs(dest)
    
    # 获取源文件夹中的所有文件和子文件夹列表
    items = os.listdir(src)
    
    # 遍历源文件夹中的每一个文件或子文件夹
    for item in items:
        # 构建完整的路径
        src_path = os.path.join(src, item)
        dest_path = os.path.join(dest, item)
        
        # 如果是文件，直接复制到目标文件夹中
        if os.path.isfile(src_path):
            shutil.copy(src_path, dest_path)
        # 如果是文件夹，递归调用本函数复制文件夹及其内容
        elif os.path.isdir(src_path):
            copy_folder(src_path, dest_path)
```