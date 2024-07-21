# `.\pytorch\torch\_export\serde\__init__.py`

```
# 导入所需的模块
import os
import sys

# 定义函数 `copy_files`，接受两个参数：源目录和目标目录
def copy_files(source_dir, dest_dir):
    # 如果目标目录不存在，则创建目标目录
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    
    # 遍历源目录下的所有文件和子目录
    for item in os.listdir(source_dir):
        # 拼接源文件或子目录的完整路径
        source_item = os.path.join(source_dir, item)
        # 拼接目标文件或子目录的完整路径
        dest_item = os.path.join(dest_dir, item)
        
        # 如果是文件，则进行复制操作
        if os.path.isfile(source_item):
            # 打开源文件
            with open(source_item, 'rb') as fsrc:
                # 打开目标文件
                with open(dest_item, 'wb') as fdest:
                    # 从源文件读取数据，写入目标文件
                    fdest.write(fsrc.read())
        # 如果是目录，则递归调用 `copy_files` 函数复制子目录及其文件
        elif os.path.isdir(source_item):
            copy_files(source_item, dest_item)

# 在命令行参数中读取源目录和目标目录
source_dir = sys.argv[1]
dest_dir = sys.argv[2]

# 调用 `copy_files` 函数，开始复制操作
copy_files(source_dir, dest_dir)
```