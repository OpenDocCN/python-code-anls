# `D:\src\scipysrc\pandas\pandas\tests\arrays\period\__init__.py`

```
# 导入需要使用的模块：os 模块提供了与操作系统交互的功能，shutil 模块提供了高级文件操作功能
import os
import shutil

# 定义一个函数，用于复制整个目录及其内容到指定位置
def copy_directory(src, dest):
    # 判断目标路径是否存在，如果不存在则创建
    if not os.path.exists(dest):
        os.makedirs(dest)
    
    # 遍历源目录下的所有文件和子目录
    for item in os.listdir(src):
        # 构建源文件或子目录的完整路径
        s = os.path.join(src, item)
        # 构建目标文件或子目录的完整路径
        d = os.path.join(dest, item)
        
        # 判断是否为文件夹，如果是则递归调用本函数复制文件夹及其内容
        if os.path.isdir(s):
            copy_directory(s, d)
        else:
            # 否则复制文件到目标路径
            shutil.copy2(s, d)
```