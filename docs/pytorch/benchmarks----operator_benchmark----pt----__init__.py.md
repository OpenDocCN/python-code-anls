# `.\pytorch\benchmarks\operator_benchmark\pt\__init__.py`

```py
# 导入必要的模块：os 模块提供了与操作系统交互的功能，shutil 模块提供了高级文件操作功能
import os
import shutil

# 定义函数 move_files，接受两个参数 src_dir 和 dest_dir
def move_files(src_dir, dest_dir):
    # 如果目标目录不存在，则创建它
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    
    # 获取源目录下的所有文件和子目录列表
    files = os.listdir(src_dir)
    
    # 遍历源目录下的每一个文件或目录
    for file_name in files:
        # 构建源文件的完整路径
        src_file = os.path.join(src_dir, file_name)
        
        # 如果是文件，则移动到目标目录
        if os.path.isfile(src_file):
            shutil.move(src_file, dest_dir)
            
        # 如果是目录，则递归调用 move_files 函数移动其下所有文件和子目录
        elif os.path.isdir(src_file):
            move_files(src_file, dest_dir)

# 调用示例：移动 '/source/dir' 下的所有文件和子目录到 '/destination/dir'
move_files('/source/dir', '/destination/dir')
```