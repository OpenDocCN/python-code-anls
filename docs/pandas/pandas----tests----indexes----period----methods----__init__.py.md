# `D:\src\scipysrc\pandas\pandas\tests\indexes\period\methods\__init__.py`

```
# 导入必要的模块：os（操作系统相关功能）、glob（文件名模式匹配）、shutil（高级文件操作）
import os
import glob
import shutil

# 设置源目录和目标目录
source_dir = '/path/to/source/directory'
target_dir = '/path/to/target/directory'

# 如果目标目录不存在，则创建它
if not os.path.exists(target_dir):
    # 使用 os.makedirs 递归地创建目录
    os.makedirs(target_dir)

# 遍历源目录下的所有文件
for file_path in glob.glob(os.path.join(source_dir, '*')):
    # 如果文件是一个普通文件并且可读
    if os.path.isfile(file_path) and os.access(file_path, os.R_OK):
        # 将文件复制到目标目录中
        shutil.copy(file_path, target_dir)

# 打印操作完成的消息
print("Files copied successfully.")
```