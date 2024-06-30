# `D:\src\scipysrc\scipy\scipy\interpolate\tests\__init__.py`

```
# 导入所需的模块：os（操作系统接口）、sys（系统特定的参数和函数）、shutil（高级文件操作）、glob（文件名模式匹配）、zipfile（ZIP 文件处理）
import os
import sys
import shutil
import glob
import zipfile

# 定义一个函数 unzip(source_filename, dest_dir)，用于解压缩 ZIP 文件
def unzip(source_filename, dest_dir):
    # 如果目标目录不存在，则创建它
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    
    # 打开 ZIP 文件为二进制读取模式
    with zipfile.ZipFile(source_filename, 'r') as zf:
        # 解压缩 ZIP 文件中的所有文件到目标目录
        zf.extractall(dest_dir)

# 定义一个函数 zip_dir(dirname, zipfilename)，用于将指定目录压缩成 ZIP 文件
def zip_dir(dirname, zipfilename):
    # 如果目标 ZIP 文件已存在，则先删除它
    if os.path.exists(zipfilename):
        os.remove(zipfilename)
    
    # 遍历指定目录下的所有文件和子目录
    files = glob.glob(os.path.join(dirname, '**'), recursive=True)
    
    # 创建一个新的 ZIP 文件并打开为写入模式
    with zipfile.ZipFile(zipfilename, 'w', zipfile.ZIP_DEFLATED) as zf:
        # 逐个将文件和子目录添加到 ZIP 文件中
        for f in files:
            zf.write(f, os.path.relpath(f, dirname))
```