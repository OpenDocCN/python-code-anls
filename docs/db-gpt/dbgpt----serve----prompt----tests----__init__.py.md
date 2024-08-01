# `.\DB-GPT-src\dbgpt\serve\prompt\tests\__init__.py`

```py
# 导入必要的模块：os（操作系统接口）、shutil（高级文件操作）、glob（文件名模式匹配）、zipfile（ZIP 文件处理）
import os
import shutil
import glob
import zipfile

# 定义一个函数，用于将指定目录下的所有文件和文件夹打包成一个 ZIP 文件
def backup_to_zip(folder, zip_file):
    # 确保传入的文件夹路径存在
    folder = os.path.abspath(folder)

    # 确保传入的 ZIP 文件路径所在的目录存在，如果不存在则创建
    os.makedirs(os.path.dirname(zip_file), exist_ok=True)

    # 创建一个新的 ZIP 文件，准备写入数据
    with zipfile.ZipFile(zip_file, 'w') as zipf:
        # 遍历传入的文件夹下的所有子文件和子文件夹
        for foldername, subfolders, filenames in os.walk(folder):
            # 添加当前文件夹到 ZIP 文件中
            zipf.write(foldername, os.path.relpath(foldername, folder))
            # 添加当前文件夹中的所有文件到 ZIP 文件中
            for filename in filenames:
                zipf.write(os.path.join(foldername, filename), os.path.relpath(os.path.join(foldername, filename), folder))

# 调用备份函数，将指定目录下的所有内容打包成一个 ZIP 文件
backup_to_zip('/path/to/folder', '/path/to/backup.zip')
```