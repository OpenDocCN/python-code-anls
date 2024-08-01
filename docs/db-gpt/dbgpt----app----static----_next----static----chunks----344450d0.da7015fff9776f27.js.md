# `.\DB-GPT-src\dbgpt\app\static\_next\static\chunks\344450d0.da7015fff9776f27.js`

```py
# 导入需要的模块：os（操作系统接口）、shutil（高级文件操作）、datetime（日期和时间处理）
import os
import shutil
import datetime

# 定义一个函数，用于备份指定目录的文件到另一个目录
def backup_files(source_dir, dest_dir):
    # 使用当前日期作为备份文件夹的名称
    backup_dir = os.path.join(dest_dir, datetime.datetime.now().strftime('%Y-%m-%d'))
    
    # 如果目标备份目录不存在，则创建它
    if not os.path.exists(backup_dir):
        os.makedirs(backup_dir)
    
    # 遍历源目录下的所有文件和子目录
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            # 构建源文件的完整路径
            source_file = os.path.join(root, file)
            # 构建目标文件的完整路径（在备份目录下保持相同的相对路径）
            dest_file = os.path.join(backup_dir, os.path.relpath(source_file, source_dir))
            # 如果目标文件所在的目录不存在，则创建它
            if not os.path.exists(os.path.dirname(dest_file)):
                os.makedirs(os.path.dirname(dest_file))
            # 复制源文件到目标文件
            shutil.copy2(source_file, dest_file)
    
    # 返回备份文件夹的路径
    return backup_dir
```