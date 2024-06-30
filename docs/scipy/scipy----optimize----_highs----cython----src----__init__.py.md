# `D:\src\scipysrc\scipy\scipy\optimize\_highs\cython\src\__init__.py`

```
# 导入所需的模块：`os` 用于操作文件路径，`shutil` 用于执行文件移动和删除操作
import os
import shutil

# 定义函数 `organize_files`
def organize_files(source_dir, target_dir):
    # 使用 `os.listdir` 函数获取源目录下的所有文件和文件夹列表
    files = os.listdir(source_dir)

    # 遍历源目录中的每个文件或文件夹
    for file_name in files:
        # 构建源文件的完整路径
        source_file = os.path.join(source_dir, file_name)
        
        # 判断当前路径是否为文件，并且不是以 `.` 开头的隐藏文件
        if os.path.isfile(source_file) and not file_name.startswith('.'):
            # 构建目标文件的完整路径，加入目标目录和文件名
            target_file = os.path.join(target_dir, file_name)
            
            # 使用 `shutil.move` 函数将文件移动到目标路径
            shutil.move(source_file, target_file)
            
            # 打印移动文件的信息
            print(f'Moved: {source_file} -> {target_file}')
```