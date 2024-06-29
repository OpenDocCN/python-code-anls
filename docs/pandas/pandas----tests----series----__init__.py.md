# `D:\src\scipysrc\pandas\pandas\tests\series\__init__.py`

```
# 导入必要的模块：os（操作系统功能）、glob（文件路径模式匹配）
import os
import glob

# 定义一个函数：获取指定目录下所有扩展名为 .txt 的文件列表
def get_txt_files(directory):
    # 使用 os 模块的 join 函数构建完整的目录路径
    dir_path = os.path.join(directory, '*.txt')
    # 使用 glob 模块的 glob 函数匹配目录下所有符合条件的文件路径，并返回一个列表
    files = glob.glob(dir_path)
    # 返回文件列表
    return files
```