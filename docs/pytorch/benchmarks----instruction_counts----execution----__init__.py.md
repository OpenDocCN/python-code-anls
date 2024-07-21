# `.\pytorch\benchmarks\instruction_counts\execution\__init__.py`

```py
# 导入所需模块：os 模块用于操作文件和目录，shutil 模块用于高级文件操作，tempfile 模块用于创建临时文件和目录
import os
import shutil
import tempfile

# 定义一个函数，接受两个参数：源目录路径 src_dir 和目标目录路径 dst_dir
def backup(src_dir, dst_dir):
    # 使用 tempfile 模块创建一个临时目录，并将其路径保存在变量 temp_dir 中
    temp_dir = tempfile.mkdtemp()
    try:
        # 将源目录 src_dir 复制到临时目录 temp_dir 中，包括其所有内容
        shutil.copytree(src_dir, temp_dir)
        # 如果目标目录 dst_dir 存在，则先递归删除目标目录及其内容
        if os.path.exists(dst_dir):
            shutil.rmtree(dst_dir)
        # 将临时目录 temp_dir 移动或重命名为目标目录 dst_dir
        shutil.move(temp_dir, dst_dir)
    finally:
        # 无论如何都会执行的清理操作：递归删除临时目录 temp_dir 及其内容
        shutil.rmtree(temp_dir)
```