# `.\DB-GPT-src\dbgpt\serve\__init__.py`

```py
# 导入所需的模块
import os
import sys
import shutil

# 定义一个函数，接受两个参数：源路径和目标路径
def backup_files(source, dest):
    # 检查目标路径是否存在，如果不存在则创建
    if not os.path.exists(dest):
        os.makedirs(dest)

    # 遍历源路径下的所有文件和文件夹
    for item in os.listdir(source):
        # 构建源文件或文件夹的完整路径
        s = os.path.join(source, item)
        # 构建目标文件或文件夹的完整路径
        d = os.path.join(dest, item)

        # 如果是文件，使用 shutil 库复制文件到目标路径
        if os.path.isfile(s):
            shutil.copy2(s, d)
        # 如果是文件夹，递归调用备份函数来复制文件夹及其内容
        elif os.path.isdir(s):
            backup_files(s, d)
```