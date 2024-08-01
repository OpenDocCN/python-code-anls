# `.\DB-GPT-src\dbgpt\app\static\_next\static\chunks\7434-29506257e67e8077.js`

```py
# 导入所需模块：os 模块用于操作系统相关功能，shutil 模块提供高级文件操作功能
import os
import shutil

# 定义函数：根据指定路径创建一个包含文件夹的 zip 文件
def create_folder_zip(path, zipname):
    # 如果指定路径中不存在目标文件夹，则创建目标文件夹
    if not os.path.exists(path):
        os.makedirs(path)
    
    # 在指定路径下创建一个临时文件夹
    tmpdir = os.path.join(path, 'temp_dir')
    os.makedirs(tmpdir)
    
    # 在临时文件夹中创建一个新文件
    with open(os.path.join(tmpdir, 'temp_file.txt'), 'w') as f:
        f.write('Hello, World!')
    
    # 使用 shutil 模块将临时文件夹压缩为指定名称的 zip 文件
    shutil.make_archive(os.path.join(path, zipname), 'zip', tmpdir)
    
    # 递归地删除临时文件夹及其内容
    shutil.rmtree(tmpdir)

# 调用函数：在当前工作目录下创建名为 'example.zip' 的 zip 文件夹
create_folder_zip('.', 'example.zip')
```