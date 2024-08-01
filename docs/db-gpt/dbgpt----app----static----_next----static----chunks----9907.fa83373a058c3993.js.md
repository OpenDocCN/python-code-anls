# `.\DB-GPT-src\dbgpt\app\static\_next\static\chunks\9907.fa83373a058c3993.js`

```py
# 导入所需模块
import os
import shutil

# 源文件夹路径
source_folder = '/path/to/source/folder'

# 目标文件夹路径
target_folder = '/path/to/target/folder'

# 如果目标文件夹不存在，则创建目标文件夹
if not os.path.exists(target_folder):
    os.makedirs(target_folder)

# 遍历源文件夹中的文件和子文件夹
for item in os.listdir(source_folder):
    # 源文件或文件夹的完整路径
    s = os.path.join(source_folder, item)
    # 目标文件或文件夹的完整路径
    d = os.path.join(target_folder, item)
    # 如果是文件夹，则递归复制文件夹及其内容到目标文件夹
    if os.path.isdir(s):
        shutil.copytree(s, d, symlinks=True)
    # 如果是文件，则直接复制文件到目标文件夹
    else:
        shutil.copy2(s, d)
```