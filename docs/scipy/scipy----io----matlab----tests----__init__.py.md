# `D:\src\scipysrc\scipy\scipy\io\matlab\tests\__init__.py`

```
# 导入所需的库：os（操作系统功能）、shutil（文件操作）和 glob（文件匹配）
import os
import shutil
import glob

# 定义函数 compress_files，接收参数 folder_path 和 dest_zip
def compress_files(folder_path, dest_zip):
    # 使用 shutil.make_archive 函数将文件夹压缩成 ZIP 文件
    shutil.make_archive(dest_zip, 'zip', folder_path)
    # 获取当前目录下所有的 ZIP 文件名列表
    zip_files = glob.glob('*.zip')
    # 遍历所有找到的 ZIP 文件名
    for zip_file in zip_files:
        # 输出每个 ZIP 文件的文件名
        print(f'Found ZIP file: {zip_file}')
    # 如果存在任何 ZIP 文件
    if zip_files:
        # 移动第一个找到的 ZIP 文件到指定的目标文件夹
        shutil.move(zip_files[0], dest_zip)
```