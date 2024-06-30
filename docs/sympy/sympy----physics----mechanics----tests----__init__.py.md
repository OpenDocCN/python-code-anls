# `D:\src\scipysrc\sympy\sympy\physics\mechanics\tests\__init__.py`

```
# 导入所需的模块：os（操作系统相关功能）、shutil（高级文件操作）、glob（文件名模式匹配）
import os
import shutil
import glob

# 定义函数：删除指定目录下所有的文件和子目录
def cleanup(target_dir):
    # 使用 glob 模块结合目录通配符 * 查找指定目录下的所有文件和目录
    files = glob.glob(os.path.join(target_dir, '*'))
    # 遍历找到的所有文件和目录
    for f in files:
        # 如果是文件，则直接删除
        if os.path.isfile(f):
            os.remove(f)
        # 如果是目录，则递归删除整个目录及其内容
        elif os.path.isdir(f):
            shutil.rmtree(f)
```