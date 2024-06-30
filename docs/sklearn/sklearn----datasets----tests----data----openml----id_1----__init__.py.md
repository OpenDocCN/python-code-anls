# `D:\src\scipysrc\scikit-learn\sklearn\datasets\tests\data\openml\id_1\__init__.py`

```
# 导入所需的模块
import os
import shutil

# 定义函数，接收源目录路径和目标目录路径作为参数
def backup_files(source_dir, dest_dir):
    # 检查目标目录是否存在，如果不存在则创建
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    # 遍历源目录中的所有文件和子目录
    for item in os.listdir(source_dir):
        # 构建完整的文件或目录路径
        s = os.path.join(source_dir, item)
        d = os.path.join(dest_dir, item)
        
        # 如果当前项目是一个文件
        if os.path.isfile(s):
            # 使用 shutil 模块复制文件到目标目录
            shutil.copy2(s, d)
        # 如果当前项目是一个目录
        elif os.path.isdir(s):
            # 递归调用 backup_files 函数，备份子目录
            backup_files(s, d)

# 调用备份函数，传入源目录路径和目标目录路径
backup_files('/path/to/source', '/path/to/destination')
```