# `.\pytorch\tools\github\__init__.py`

```py
# 导入所需的模块：os（操作系统接口）、shutil（高级文件操作工具）、datetime（处理日期和时间的标准库）
import os
import shutil
import datetime

# 定义函数：备份指定目录下的所有文件和子目录到目标目录
def backup(source_dir, target_dir):
    # 构造备份目标路径，以当前日期时间命名
    backup_name = os.path.basename(source_dir) + '_' + datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    backup_dir = os.path.join(target_dir, backup_name)

    # 使用 shutil 模块的 copytree 函数复制整个目录树
    shutil.copytree(source_dir, backup_dir)

    # 返回备份目录的路径，供用户确认
    return backup_dir
```