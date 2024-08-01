# `.\DB-GPT-src\dbgpt\app\static\_next\static\chunks\3760.e077740dca3d7ba3.js`

```py
# 导入所需的模块：os 模块用于处理文件和目录，shutil 模块用于高级文件操作
import os
import shutil

# 定义函数 compress_folder，接收两个参数：folder_path（文件夹路径）和output_path（输出压缩文件路径）
def compress_folder(folder_path, output_path):
    # 使用 shutil 模块的 make_archive 函数将文件夹 folder_path 压缩到 output_path 指定的路径中
    shutil.make_archive(output_path, 'zip', folder_path)
    # 将压缩后的文件名修改为 output_path（去除 .zip 后缀），即将 output_path.zip 改名为 output_path
    os.rename(output_path + '.zip', output_path)
```