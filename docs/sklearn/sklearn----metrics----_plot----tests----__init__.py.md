# `D:\src\scipysrc\scikit-learn\sklearn\metrics\_plot\tests\__init__.py`

```
# 导入需要的模块：os（操作系统接口）、shutil（高级文件操作）、glob（文件名模式匹配）、re（正则表达式操作）
import os
import shutil
import glob
import re

# 定义函数：遍历指定目录下的所有文件和子目录，并返回它们的绝对路径列表
def list_files(directory):
    # 初始化空列表，用于存储所有文件的绝对路径
    files = []
    # 遍历指定目录下的所有文件和子目录
    for dirpath, dirnames, filenames in os.walk(directory):
        # 将当前目录下的所有文件添加到 files 列表中
        for file in filenames:
            # 使用 os.path.join 将当前文件的路径与目录路径拼接成完整的绝对路径
            files.append(os.path.join(dirpath, file))
    # 返回包含所有文件绝对路径的列表
    return files

# 定义函数：将指定目录下的所有文件复制到目标目录
def copy_files(src_dir, dest_dir):
    # 确保目标目录存在，如果不存在则创建它
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    # 获取源目录下所有文件的绝对路径列表
    files = list_files(src_dir)
    # 遍历文件列表，逐个复制文件到目标目录
    for file in files:
        # 使用 shutil.copy2 复制文件到目标目录，保留元数据（如文件权限、最后访问时间等）
        shutil.copy2(file, dest_dir)

# 定义函数：返回指定目录下所有以数字开头的文件或目录的列表
def find_numeric_dirs(directory):
    # 初始化空列表，用于存储以数字开头的文件或目录
    numeric_dirs = []
    # 获取指定目录下所有文件和目录的列表
    entries = os.listdir(directory)
    # 使用正则表达式匹配以数字开头的文件或目录名
    for entry in entries:
        if re.match(r'^\d+', entry):
            # 如果匹配成功，将其加入到 numeric_dirs 列表中
            numeric_dirs.append(entry)
    # 返回以数字开头的文件或目录名列表
    return numeric_dirs

# 定义函数：返回指定目录下所有以特定后缀名结尾的文件列表
def find_files_by_extension(directory, extension):
    # 使用 glob 模块匹配指定目录下所有以特定后缀名结尾的文件
    files = glob.glob(os.path.join(directory, f'*.{extension}'))
    # 返回符合条件的文件列表
    return files
```