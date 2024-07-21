# `.\pytorch\test\jit\_imported_class_test\very\__init__.py`

```
# 导入必要的模块：os（操作系统功能）和 shutil（高级文件操作功能）
import os
import shutil

# 定义一个函数 move_files，接收两个参数：source_dir（源目录）和dest_dir（目标目录）
def move_files(source_dir, dest_dir):
    # 确保目标目录存在，如果不存在则创建
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    
    # 遍历源目录下的所有文件和子目录
    for root, dirs, files in os.walk(source_dir):
        # 遍历当前目录下的所有文件
        for file in files:
            # 构建源文件的完整路径
            source_file = os.path.join(root, file)
            # 构建目标文件的完整路径
            dest_file = os.path.join(dest_dir, file)
            # 如果目标文件已经存在，则先删除，以便移动文件
            if os.path.exists(dest_file):
                os.remove(dest_file)
            # 移动源文件到目标位置
            shutil.move(source_file, dest_file)

# 示例用法：将源目录 'source_folder' 下的所有文件移动到目标目录 'destination_folder'
move_files('source_folder', 'destination_folder')
```