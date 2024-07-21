# `.\pytorch\test\expect\__init__.py`

```
# 导入必要的模块：os（操作系统接口）、shutil（高级文件操作）、datetime（日期和时间相关操作）
import os
import shutil
import datetime

# 定义函数：备份指定目录的所有文件到目标目录
def backup_files(source_dir, dest_dir):
    # 构建备份文件名，格式为当前日期时间 + '_backup'
    backup_name = datetime.datetime.now().strftime('%Y%m%d%H%M%S') + '_backup'
    # 拼接目标目录和备份文件名构成完整的备份目录路径
    backup_dir = os.path.join(dest_dir, backup_name)
    # 创建目标备份目录
    os.makedirs(backup_dir)
    
    # 遍历源目录下的所有文件和目录
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            # 构建源文件的完整路径
            source_file = os.path.join(root, file)
            # 构建目标备份文件的完整路径
            dest_file = os.path.join(backup_dir, file)
            # 复制源文件到目标备份目录
            shutil.copy2(source_file, dest_file)
    
    # 返回备份的目录路径
    return backup_dir
```