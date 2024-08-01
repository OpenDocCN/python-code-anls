# `.\DB-GPT-src\dbgpt\serve\rag\retriever\__init__.py`

```py
# 导入所需的模块：os 模块用于处理操作系统相关功能，shutil 模块用于高级文件操作
import os
import shutil

# 定义函数 copy_files，接收两个参数 src_dir 和 dest_dir，用于从源目录复制所有文件到目标目录
def copy_files(src_dir, dest_dir):
    # 如果目标目录不存在，则创建目标目录
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    
    # 获取源目录下的所有文件和文件夹列表
    files = os.listdir(src_dir)
    
    # 遍历源目录下的每个文件或文件夹
    for file in files:
        # 构建源文件的完整路径
        src_file = os.path.join(src_dir, file)
        
        # 构建目标文件的完整路径
        dest_file = os.path.join(dest_dir, file)
        
        # 如果是文件夹，则递归调用 copy_files 函数复制文件夹及其内容
        if os.path.isdir(src_file):
            copy_files(src_file, dest_file)
        else:
            # 如果是文件，则使用 shutil 模块的 copy2 函数复制文件（保留元数据）
            shutil.copy2(src_file, dest_file)
```