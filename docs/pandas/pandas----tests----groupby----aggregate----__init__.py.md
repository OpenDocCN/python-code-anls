# `D:\src\scipysrc\pandas\pandas\tests\groupby\aggregate\__init__.py`

```
# 导入所需的模块：os（操作系统接口）、shutil（高级文件操作模块）
import os
import shutil

# 定义函数：复制源目录到目标目录
def copytree(src, dst, symlinks=False):
    # 获取源目录中所有文件和子目录的列表
    names = os.listdir(src)
    # 如果目标目录不存在，则创建
    os.makedirs(dst)
    
    # 遍历源目录中的每个文件或目录
    for name in names:
        # 拼接源文件或目录的完整路径
        srcname = os.path.join(src, name)
        # 拼接目标文件或目录的完整路径
        dstname = os.path.join(dst, name)
        
        # 如果是符号链接并允许复制符号链接，则创建符号链接
        if symlinks and os.path.islink(srcname):
            linkto = os.readlink(srcname)
            os.symlink(linkto, dstname)
        # 如果是目录，则递归调用本函数复制目录及其内容
        elif os.path.isdir(srcname):
            copytree(srcname, dstname, symlinks)
        # 如果是文件，则通过 shutil 模块复制文件
        else:
            shutil.copy2(srcname, dstname)
```