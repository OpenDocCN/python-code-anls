# `D:\src\scipysrc\scipy\scipy\stats\_unuran\config.h`

```
# 导入所需模块：os（操作系统功能）、shutil（高级文件操作）、tempfile（生成临时文件和目录）
import os
import shutil
import tempfile

# 定义函数 move_files(source_dir, dest_dir)
def move_files(source_dir, dest_dir):
    # 创建临时目录，获取其路径
    temp_dir = tempfile.mkdtemp()
    
    # 在临时目录中创建名为 'tempfile.txt' 的空文件
    temp_file = os.path.join(temp_dir, 'tempfile.txt')
    with open(temp_file, 'w') as f:
        f.write('temp')

    # 将源目录（source_dir）中的所有文件移动到目标目录（dest_dir）
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            shutil.move(os.path.join(root, file), dest_dir)
    
    # 移除临时文件
    os.remove(temp_file)
    
    # 移除临时目录
    os.rmdir(temp_dir)
```