# `.\numpy\numpy\random\tests\data\__init__.py`

```py
# 导入必要的模块：os模块用于与操作系统交互，shutil模块用于高级文件操作
import os
import shutil

# 定义一个函数，接受两个参数：源目录和目标目录
def copytree(src, dst):
    # 如果目标目录已经存在，则抛出一个异常
    if os.path.exists(dst):
        raise OSError(f"目标目录 '{dst}' 已经存在.")
    
    # 创建目标目录
    os.makedirs(dst)
    
    # 遍历源目录下的所有文件和目录
    for item in os.listdir(src):
        # 拼接源文件或目录的完整路径
        s = os.path.join(src, item)
        # 拼接目标文件或目录的完整路径
        d = os.path.join(dst, item)
        
        # 如果当前项目是一个目录，则递归地复制整个目录树
        if os.path.isdir(s):
            shutil.copytree(s, d)
        # 如果当前项目是一个文件，则直接复制文件
        else:
            shutil.copy2(s, d)

# 调用函数，复制源目录到目标目录
copytree('/path/to/source', '/path/to/destination')
```