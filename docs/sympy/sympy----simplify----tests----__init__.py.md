# `D:\src\scipysrc\sympy\sympy\simplify\tests\__init__.py`

```
# 导入所需的模块：os（操作系统功能）、shutil（高级文件操作功能）
import os
import shutil

# 定义一个函数，用于复制指定源目录下的所有文件和子目录到目标目录
def copytree(src, dst):
    # 获取源目录下的所有文件和子目录列表
    files = os.listdir(src)
    # 遍历列表中的每个文件或子目录
    for f in files:
        # 拼接源目录的路径和文件名/子目录名，得到完整路径
        src_path = os.path.join(src, f)
        # 拼接目标目录的路径和文件名/子目录名，得到完整路径
        dst_path = os.path.join(dst, f)
        # 如果是一个子目录，则递归调用 copytree 函数复制整个子目录及其内容
        if os.path.isdir(src_path):
            copytree(src_path, dst_path)
        # 如果是一个文件，则使用 shutil.copy2 函数复制文件到目标目录
        else:
            shutil.copy2(src_path, dst_path)
```