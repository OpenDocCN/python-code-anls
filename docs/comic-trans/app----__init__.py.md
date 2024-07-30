# `.\comic-translate\app\__init__.py`

```py
# 导入所需模块
import os
import shutil

# 定义函数，用于移动文件
def move_files(source_dir, dest_dir):
    # 获取源目录下的所有文件列表
    files = os.listdir(source_dir)
    # 遍历每个文件
    for file in files:
        # 构建完整的源文件路径
        source_file = os.path.join(source_dir, file)
        # 构建完整的目标文件路径
        dest_file = os.path.join(dest_dir, file)
        # 移动文件
        shutil.move(source_file, dest_file)
```