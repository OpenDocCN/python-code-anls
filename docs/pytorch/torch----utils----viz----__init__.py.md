# `.\pytorch\torch\utils\viz\__init__.py`

```py
# 导入所需的模块
import os
import shutil

# 定义函数 compress_folder，接收一个文件夹路径作为参数
def compress_folder(folder_path):
    # 获取当前工作目录
    current_dir = os.getcwd()
    # 切换到指定的文件夹路径
    os.chdir(folder_path)
    # 获取当前文件夹的名称
    folder_name = os.path.basename(folder_path)
    # 将当前文件夹压缩成 ZIP 文件，文件名为当前文件夹的名称 + '.zip'
    shutil.make_archive(folder_name, 'zip', folder_path)
    # 切换回原来的工作目录
    os.chdir(current_dir)
```