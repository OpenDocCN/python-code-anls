# `D:\src\scipysrc\scikit-learn\sklearn\cluster\_hdbscan\tests\__init__.py`

```
# 导入必要的模块：os（操作系统接口）、glob（用于查找文件路径名匹配模式）、shutil（用于高级文件操作）和zipfile（用于操作ZIP文件）
import os
import glob
import shutil
import zipfile

# 定义一个函数，用于将指定目录下的所有文件和子目录压缩成一个ZIP文件
def zip_directory(directory, zipname):
    # 创建一个ZIP文件，准备写入压缩内容
    with zipfile.ZipFile(zipname, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # 遍历指定目录及其所有子目录和文件
        for root, dirs, files in os.walk(directory):
            # 在ZIP文件中创建当前目录的相对路径
            for file in files:
                # 构造当前文件的绝对路径
                filepath = os.path.join(root, file)
                # 将文件添加到ZIP文件中，使用相对路径
                zipf.write(filepath, os.path.relpath(filepath, directory))

# 示例用法：将当前目录下的所有文件和子目录压缩为一个名为'archive.zip'的ZIP文件
zip_directory('.', 'archive.zip')
```