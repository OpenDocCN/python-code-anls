# `.\numpy\numpy\typing\tests\__init__.py`

```
# 导入所需的模块
import os
import shutil

# 源文件夹路径
source_folder = '/path/to/source/folder'

# 目标文件夹路径
target_folder = '/path/to/target/folder'

# 如果目标文件夹不存在，则创建它
if not os.path.exists(target_folder):
    os.makedirs(target_folder)

# 遍历源文件夹中的文件和子文件夹
for item in os.listdir(source_folder):
    # 构建源文件或子文件夹的完整路径
    source_item = os.path.join(source_folder, item)
    # 构建目标文件或子文件夹的完整路径
    target_item = os.path.join(target_folder, item)
    # 如果当前项目是一个文件，就复制它到目标文件夹中
    if os.path.isfile(source_item):
        shutil.copy(source_item, target_item)
    # 如果当前项目是一个文件夹，就递归地复制整个文件夹及其内容到目标文件夹中
    elif os.path.isdir(source_item):
        shutil.copytree(source_item, target_item)
```