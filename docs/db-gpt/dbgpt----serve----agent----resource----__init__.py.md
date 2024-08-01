# `.\DB-GPT-src\dbgpt\serve\agent\resource\__init__.py`

```py
# 导入所需的模块：os 用于操作文件路径，shutil 用于文件操作
import os
import shutil

# 定义函数 compress(source, destination)，接受源目录路径 source 和目标压缩文件路径 destination
def compress(source, destination):
    # 使用 shutil.make_archive 函数创建 ZIP 压缩文件，传入目标文件名和目录路径
    shutil.make_archive(destination, 'zip', source)
    # 打印提示信息，表示压缩完成
    print(f'压缩完成：{destination}.zip')

# 示例调用 compress 函数，压缩当前目录下的 'data' 文件夹到 'backup' 文件夹中的 'data_backup' ZIP 文件
compress('data', 'backup/data_backup')
```