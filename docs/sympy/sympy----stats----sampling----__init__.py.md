# `D:\src\scipysrc\sympy\sympy\stats\sampling\__init__.py`

```
# 导入所需模块
import os
import shutil

# 源目录
source_dir = '/path/to/source'

# 目标目录
target_dir = '/path/to/target'

# 遍历源目录中的文件和子目录
for item in os.listdir(source_dir):
    # 源文件或目录的完整路径
    s = os.path.join(source_dir, item)
    # 目标文件或目录的完整路径
    d = os.path.join(target_dir, item)
    # 如果是文件，使用 shutil 模块复制文件到目标目录
    if os.path.isfile(s):
        shutil.copy2(s, d)
    # 如果是目录，使用 shutil 模块递归地复制整个目录树到目标目录
    elif os.path.isdir(s):
        shutil.copytree(s, d)
```