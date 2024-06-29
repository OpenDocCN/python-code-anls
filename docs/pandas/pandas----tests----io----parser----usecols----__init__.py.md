# `D:\src\scipysrc\pandas\pandas\tests\io\parser\usecols\__init__.py`

```
# 导入必要的模块：os 模块用于操作系统功能，shutil 模块用于高级文件操作
import os
import shutil

# 定义函数 compress_files，接收两个参数：source_dir（源目录路径）和dest_zip（目标 ZIP 文件路径）
def compress_files(source_dir, dest_zip):
    # 创建一个新的 ZIP 文件，以写入模式打开
    with zipfile.ZipFile(dest_zip, 'w') as zipf:
        # 遍历源目录下的所有文件和子目录
        for root, dirs, files in os.walk(source_dir):
            # 遍历当前目录下的文件
            for file in files:
                # 构建文件的完整路径
                filepath = os.path.join(root, file)
                # 将文件添加到 ZIP 文件中，使用相对路径保存到 ZIP 中
                zipf.write(filepath, os.path.relpath(filepath, source_dir))
```