# `D:\src\scipysrc\pandas\pandas\tests\arrays\boolean\__init__.py`

```
# 导入需要的模块：os 用于操作系统相关功能，shutil 用于高级文件操作，tempfile 用于临时文件和目录的创建
import os
import shutil
import tempfile

# 定义函数 extract_and_move，接收两个参数：src_dir（源目录）和 dest_dir（目标目录）
def extract_and_move(src_dir, dest_dir):
    # 使用 tempfile 模块创建一个临时目录，并将路径存储在 temp_dir 变量中
    temp_dir = tempfile.mkdtemp()
    
    # 使用 os 模块的 walk 函数遍历 src_dir 目录下的所有文件和子目录
    for root, dirs, files in os.walk(src_dir):
        # 遍历当前目录下的文件列表
        for file in files:
            # 拼接文件的完整路径
            src_file = os.path.join(root, file)
            # 如果文件是一个普通文件（非目录），则进行复制操作
            if os.path.isfile(src_file):
                # 拼接目标目录下的文件路径
                dest_file = os.path.join(dest_dir, file)
                # 使用 shutil 模块的 copy2 函数复制文件到目标目录
                shutil.copy2(src_file, dest_file)
    
    # 使用 shutil 模块的 rmtree 函数删除临时目录及其内容
    shutil.rmtree(temp_dir)
```