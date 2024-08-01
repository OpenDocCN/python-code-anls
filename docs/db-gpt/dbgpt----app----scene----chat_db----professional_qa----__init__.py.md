# `.\DB-GPT-src\dbgpt\app\scene\chat_db\professional_qa\__init__.py`

```py
# 导入所需的模块：os 模块用于操作文件路径，shutil 模块用于高级文件操作
import os
import shutil

# 定义一个函数，接受两个参数：源目录和目标目录
def backup_files(source_dir, dest_dir):
    # 如果目标目录不存在，则创建目标目录
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    
    # 遍历源目录下的所有文件和子目录
    for root, dirs, files in os.walk(source_dir):
        # 遍历当前目录下的文件
        for file in files:
            # 构建源文件的完整路径
            source_file = os.path.join(root, file)
            # 构建目标文件的完整路径，拼接目标目录和源文件的相对路径
            dest_file = os.path.join(dest_dir, os.path.relpath(source_file, source_dir))
            
            # 如果源文件是文件（不是目录）
            if os.path.isfile(source_file):
                # 如果目标文件已经存在且与源文件一致，则跳过
                if os.path.exists(dest_file) and os.path.samefile(source_file, dest_file):
                    continue
                # 否则，复制源文件到目标文件
                shutil.copy2(source_file, dest_file)
            # 如果源文件是目录
            elif os.path.isdir(source_file):
                # 如果目标目录已经存在，则跳过
                if os.path.exists(dest_file):
                    continue
                # 否则，递归地创建目录结构
                os.makedirs(dest_file)

# 备份文件的源目录和目标目录
source_dir = '/path/to/source'
dest_dir = '/path/to/destination'

# 调用备份函数，传入源目录和目标目录进行备份操作
backup_files(source_dir, dest_dir)
```