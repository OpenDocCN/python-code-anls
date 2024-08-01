# `.\DB-GPT-src\dbgpt\app\static\_next\static\chunks\9400.5d4350f695c4222f.js`

```py
# 导入必要的模块：os 用于操作文件路径，shutil 用于高级文件操作
import os
import shutil

# 定义函数：复制文件夹及其内容到目标位置
def copy_folder(src, dest):
    # 如果目标文件夹存在，则先删除
    if os.path.exists(dest):
        shutil.rmtree(dest)
    # 递归地复制源文件夹及其内容到目标位置
    shutil.copytree(src, dest)

# 调用函数，将文件夹 'source_folder' 复制到 'destination_folder'
copy_folder('source_folder', 'destination_folder')
```