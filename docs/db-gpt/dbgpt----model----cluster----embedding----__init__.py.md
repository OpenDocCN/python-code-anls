# `.\DB-GPT-src\dbgpt\model\cluster\embedding\__init__.py`

```py
# 导入所需的模块
import os
import shutil

# 定义一个函数，接收源目录路径和目标目录路径作为参数
def copy_files(source_dir, target_dir):
    # 如果目标目录不存在，则创建目标目录
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # 遍历源目录中的所有文件和子目录
    for item in os.listdir(source_dir):
        # 构建完整的源文件或目录的路径
        source = os.path.join(source_dir, item)
        # 构建完整的目标文件或目录的路径
        target = os.path.join(target_dir, item)
        
        # 如果当前项是一个文件，直接复制到目标目录中
        if os.path.isfile(source):
            shutil.copy(source, target)
        # 如果当前项是一个子目录，递归调用本函数复制整个子目录
        elif os.path.isdir(source):
            copy_files(source, target)
```