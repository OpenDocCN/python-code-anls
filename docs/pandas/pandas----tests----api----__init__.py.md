# `D:\src\scipysrc\pandas\pandas\tests\api\__init__.py`

```
# 导入必要的模块：os 模块用于与操作系统交互，shutil 模块用于高级文件操作
import os
import shutil

# 定义函数：清空指定目录下的所有内容
def clear_folder(folder):
    # 检查指定的路径是否是一个目录
    if os.path.isdir(folder):
        # 使用 shutil 模块的 rmtree 函数删除目录及其内容
        shutil.rmtree(folder)
    # 如果路径不是目录，则抛出异常
    else:
        raise ValueError(f"{folder} is not a directory")

# 调用函数：清空当前目录下的 "temp" 子目录
clear_folder('temp')
```