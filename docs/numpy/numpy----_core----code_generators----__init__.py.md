# `.\numpy\numpy\_core\code_generators\__init__.py`

```py
# 导入必要的模块：os（操作系统接口）、shutil（高级文件操作）、datetime（日期和时间相关操作）
import os
import shutil
import datetime

# 定义一个函数，接收两个参数：source_dir（源目录）和dest_dir（目标目录）
def backup_files(source_dir, dest_dir):
    # 使用当前日期和时间生成一个唯一的备份文件夹名
    backup_dir = os.path.join(dest_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    
    # 创建备份目录
    os.makedirs(backup_dir)
    
    # 遍历源目录中的所有文件和文件夹
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            # 构建每个文件的源路径和目标路径
            source_file = os.path.join(root, file)
            dest_file = os.path.join(backup_dir, os.path.relpath(source_file, source_dir))
            
            # 如果目标路径不存在，创建它
            if not os.path.exists(os.path.dirname(dest_file)):
                os.makedirs(os.path.dirname(dest_file))
            
            # 复制源文件到目标位置
            shutil.copy(source_file, dest_file)
    
    # 返回备份文件夹的路径
    return backup_dir
```