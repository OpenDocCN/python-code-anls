# `.\DB-GPT-src\dbgpt\app\static\_next\static\chunks\1065.f6e9b646913a0d07.js`

```py
# 导入所需模块
import os
import zipfile

# 定义函数 unzip_files，接收 ZIP 文件路径和目标文件夹路径作为参数
def unzip_files(zip_file, dest_dir):
    # 如果目标文件夹不存在，则创建它
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    # 打开 ZIP 文件为二进制读取模式
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        # 解压 ZIP 文件中的所有文件到目标文件夹
        zip_ref.extractall(dest_dir)

# 调用函数 unzip_files，解压名为 'example.zip' 的 ZIP 文件到当前目录下的 'example_folder' 文件夹
unzip_files('example.zip', 'example_folder')
```