# `D:\src\scipysrc\pandas\pandas\tests\resample\__init__.py`

```
# 导入必要的模块：os模块用于操作文件系统，shutil模块用于高级文件操作
import os
import shutil

# 定义一个函数，用于复制目录及其内容
def copy_directory(source, destination):
    # 如果目标目录不存在，则创建它
    if not os.path.exists(destination):
        os.makedirs(destination)
    
    # 遍历源目录中的所有文件和子目录
    for item in os.listdir(source):
        # 构建源文件或目录的完整路径
        s = os.path.join(source, item)
        # 构建目标文件或目录的完整路径
        d = os.path.join(destination, item)
        
        # 如果是一个子目录，则递归调用copytree函数复制子目录及其内容
        if os.path.isdir(s):
            shutil.copytree(s, d)
        # 如果是一个文件，则使用shutil.copy2函数复制文件
        else:
            shutil.copy2(s, d)
```