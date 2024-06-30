# `D:\src\scipysrc\scipy\scipy\cluster\tests\__init__.py`

```
# 导入必要的模块：os 模块提供了与操作系统交互的功能，shutil 模块提供了高级的文件操作功能
import os
import shutil

# 定义函数 move_files，接收两个参数：source_dir 源目录和 dest_dir 目标目录
def move_files(source_dir, dest_dir):
    # 如果目标目录不存在，则创建目标目录
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    
    # 遍历源目录下的所有文件和子目录
    for item in os.listdir(source_dir):
        # 构建源文件或目录的完整路径
        s = os.path.join(source_dir, item)
        # 构建目标文件或目录的完整路径
        d = os.path.join(dest_dir, item)
        
        # 如果 s 是文件并且 d 不存在或者 s 是目录并且 d 不是目录，则执行复制操作
        if os.path.isfile(s):
            if not os.path.exists(d) or not os.path.isdir(d):
                shutil.copy2(s, d)
        elif os.path.isdir(s):
            shutil.copytree(s, d, symlinks=True)
```