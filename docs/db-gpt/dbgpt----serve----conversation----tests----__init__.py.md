# `.\DB-GPT-src\dbgpt\serve\conversation\tests\__init__.py`

```py
# 导入必要的模块：os 模块用于操作文件系统，shutil 模块用于高级文件操作
import os
import shutil

# 定义函数 compress_folder，接收一个文件夹路径作为参数
def compress_folder(folder_path):
    # 获取文件夹的基本名称作为压缩文件的名称
    base_name = os.path.basename(folder_path)
    # 使用 shutil 模块创建一个 ZIP 文件，命名为文件夹的基本名称，模式为写入
    shutil.make_archive(base_name, 'zip', folder_path)
    # 返回压缩文件的名称，包含完整路径
    return f"{base_name}.zip"
```