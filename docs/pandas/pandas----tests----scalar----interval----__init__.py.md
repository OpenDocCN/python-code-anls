# `D:\src\scipysrc\pandas\pandas\tests\scalar\interval\__init__.py`

```
# 导入必要的模块：os 用于与操作系统交互，shutil 用于高级文件操作
import os
import shutil

# 定义函数：递归地复制源目录下的所有文件和子目录到目标目录
def copytree(src, dst):
    # 获取源目录下的所有文件和子目录列表
    for item in os.listdir(src):
        # 构建源文件或目录的完整路径
        s = os.path.join(src, item)
        # 构建目标文件或目录的完整路径
        d = os.path.join(dst, item)
        # 如果是子目录，递归调用 copytree 函数复制整个子目录
        if os.path.isdir(s):
            copytree(s, d)
        else:
            # 如果是文件，使用 shutil.copy2 复制文件到目标位置
            shutil.copy2(s, d)
```