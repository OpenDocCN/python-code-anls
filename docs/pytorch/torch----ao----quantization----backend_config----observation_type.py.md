# `.\pytorch\torch\ao\quantization\backend_config\observation_type.py`

```py
# 导入所需的模块：os 模块用于与操作系统交互，shutil 模块用于高级文件操作，tempfile 模块用于创建临时文件和目录
import os
import shutil
import tempfile

# 定义一个函数，该函数接收一个路径作为参数
def make_backup(sfile):
    # 获取 sfile 的绝对路径
    sfile = os.path.abspath(sfile)
    # 获取 sfile 的基本文件名（不含路径）
    basename = os.path.basename(sfile)
    # 获取 sfile 的目录名
    dirname = os.path.dirname(sfile)
    # 使用 tempfile 模块创建一个临时目录
    tmpdir = tempfile.mkdtemp()
    try:
        # 将 sfile 复制到临时目录中，目标文件名为原始文件的基本文件名
        shutil.copy(sfile, os.path.join(tmpdir, basename))
        # 返回备份文件的路径，格式为：临时目录路径/原始文件名
        return os.path.join(tmpdir, basename)
    finally:
        # 删除临时目录及其内容
        shutil.rmtree(tmpdir)
```