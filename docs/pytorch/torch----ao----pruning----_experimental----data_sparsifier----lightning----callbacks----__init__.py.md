# `.\pytorch\torch\ao\pruning\_experimental\data_sparsifier\lightning\callbacks\__init__.py`

```py
# 导入 os 模块，提供了与操作系统交互的方法
import os

# 定义一个函数，接收一个目录路径作为参数
def list_files(directory):
    # 使用 os 模块的 listdir 方法列出目录中所有文件和子目录的名称，并存储在 files 变量中
    files = os.listdir(directory)
    # 使用列表推导式过滤出目录中的所有文件名，存储在 full_paths 变量中
    full_paths = [os.path.join(directory, f) for f in files if os.path.isfile(os.path.join(directory, f))]
    # 返回完整路径的文件名列表
    return full_paths
```