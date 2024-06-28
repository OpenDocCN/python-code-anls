# `.\agent\Agent_OpenAI\scripts\__init__.py`

```
# 导入模块 'os'、'sys' 和 'shutil'
import os, sys, shutil

# 定义函数 'backup_files'
def backup_files(source_dir, dest_dir):
    # 如果目标目录不存在，则创建它
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    
    # 遍历源目录中的所有文件和文件夹
    for item in os.listdir(source_dir):
        # 构建源文件/文件夹的完整路径
        source = os.path.join(source_dir, item)
        # 构建目标文件/文件夹的完整路径
        destination = os.path.join(dest_dir, item)
        
        # 如果是文件，使用 'shutil.copy2' 复制文件到目标目录
        if os.path.isfile(source):
            shutil.copy2(source, destination)
        # 如果是文件夹，递归调用 'backup_files' 函数备份文件夹及其内容
        elif os.path.isdir(source):
            backup_files(source, destination)
```