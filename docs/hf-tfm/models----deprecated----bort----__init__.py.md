# `.\models\deprecated\bort\__init__.py`

```py
# 导入必要的模块：os 模块提供了操作系统相关的功能，shutil 模块提供了高级的文件操作功能
import os
import shutil

# 定义一个函数，用于复制指定目录下的所有文件到另一个目录中
def copy_all_files(src_dir, dst_dir):
    # 如果目标目录不存在，则创建目标目录
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    
    # 遍历源目录下的所有文件和子目录
    for item in os.listdir(src_dir):
        # 构造完整的源文件或子目录路径
        src = os.path.join(src_dir, item)
        # 构造完整的目标文件或子目录路径
        dst = os.path.join(dst_dir, item)
        
        # 如果是文件，则执行复制操作
        if os.path.isfile(src):
            shutil.copy(src, dst)
        # 如果是子目录，则递归调用本函数复制子目录及其内容
        elif os.path.isdir(src):
            copy_all_files(src, dst)
```