# `.\pytorch\torch\_strobelight\__init__.py`

```py
# 导入所需模块：os（操作系统功能）、shutil（文件操作）、datetime（日期时间处理）
import os
import shutil
import datetime

# 定义函数：备份指定目录下的所有文件到目标目录
def backup_files(source_dir, target_dir):
    # 获取当前日期作为备份目录名
    backup_dir = os.path.join(target_dir, datetime.datetime.now().strftime('%Y-%m-%d'))
    
    # 如果目标备份目录不存在，则创建它
    if not os.path.exists(backup_dir):
        os.makedirs(backup_dir)
    
    # 获取源目录下的所有文件和子目录列表
    files = os.listdir(source_dir)
    
    # 遍历源目录下的所有文件和子目录
    for file_name in files:
        # 构造源文件的完整路径
        source_file = os.path.join(source_dir, file_name)
        
        # 如果当前项是文件
        if os.path.isfile(source_file):
            # 构造目标备份目录下的完整路径
            target_file = os.path.join(backup_dir, file_name)
            
            # 复制源文件到目标备份目录
            shutil.copy(source_file, target_file)
    
    # 返回备份目录的路径
    return backup_dir
```