# `D:\src\scipysrc\scikit-learn\sklearn\datasets\tests\data\openml\id_561\__init__.py`

```
# 导入所需模块：os（操作系统接口）、shutil（高级文件操作）、datetime（日期和时间处理）、glob（文件名模式匹配）
import os
import shutil
import datetime
import glob

# 定义一个函数，用于移动文件并记录移动的操作日志
def move_files(source_dir, dest_dir):
    # 获取当前日期和时间
    today = datetime.date.today()
    # 格式化日期字符串
    today_str = today.strftime('%Y%m%d')
    # 构建日志文件名
    log_filename = f'move_log_{today_str}.txt'
    
    # 创建日志文件
    with open(log_filename, 'w') as log:
        # 获取源目录下所有扩展名为txt的文件列表
        files_to_move = glob.glob(os.path.join(source_dir, '*.txt'))
        
        # 遍历每个文件
        for file_path in files_to_move:
            try:
                # 构建目标文件路径
                dest_file = os.path.join(dest_dir, os.path.basename(file_path))
                # 移动文件到目标目录
                shutil.move(file_path, dest_file)
                # 记录移动操作到日志文件
                log.write(f'Moved {file_path} to {dest_file}\n')
            except Exception as e:
                # 若移动过程中出现异常，记录异常信息到日志文件
                log.write(f'Error moving {file_path}: {str(e)}\n')

# 示例调用函数：移动源目录中的txt文件到目标目录，并记录操作日志
move_files('/path/to/source/directory', '/path/to/destination/directory')
```