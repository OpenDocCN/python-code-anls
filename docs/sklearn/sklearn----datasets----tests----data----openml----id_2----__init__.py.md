# `D:\src\scipysrc\scikit-learn\sklearn\datasets\tests\data\openml\id_2\__init__.py`

```
# 导入所需模块：os（操作系统接口）、shutil（文件操作工具）、glob（文件名模式匹配）、zipfile（ZIP 文件处理）
import os
import shutil
import glob
import zipfile

# 定义函数 `backup_and_zip`，接收两个参数：`source_dir`（源目录路径）和 `dest_zip`（目标 ZIP 文件路径）
def backup_and_zip(source_dir, dest_zip):
    # 如果目标 ZIP 文件已存在，则删除它（重新创建）
    if os.path.exists(dest_zip):
        os.remove(dest_zip)
    # 创建一个新的 ZIP 文件对象，以写模式打开
    zipf = zipfile.ZipFile(dest_zip, 'w', zipfile.ZIP_DEFLATED)
    
    # 使用 glob 模块获取源目录下的所有文件和子目录
    for root, dirs, files in os.walk(source_dir):
        # 遍历当前目录下的所有文件
        for file in files:
            # 构建当前文件的完整路径
            full_path = os.path.join(root, file)
            # 将文件添加到 ZIP 文件中，使用相对路径存储
            zipf.write(full_path, os.path.relpath(full_path, source_dir))
    
    # 关闭 ZIP 文件对象
    zipf.close()
    
    # 返回目标 ZIP 文件的路径
    return dest_zip
```