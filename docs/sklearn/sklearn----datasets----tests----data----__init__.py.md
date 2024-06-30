# `D:\src\scipysrc\scikit-learn\sklearn\datasets\tests\data\__init__.py`

```
# 导入所需的模块：os（操作系统功能）、shutil（高级文件操作）、glob（文件名模式匹配）
import os
import shutil
import glob

# 定义函数：复制指定目录下的所有文件到目标目录
def copy_files(source_dir, target_dir):
    # 确保目标目录存在，如果不存在则创建
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # 遍历源目录下的所有文件和子目录
    for file_path in glob.glob(os.path.join(source_dir, '*')):
        # 如果是文件，则复制到目标目录中
        if os.path.isfile(file_path):
            shutil.copy(file_path, target_dir)
```