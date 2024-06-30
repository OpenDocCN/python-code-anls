# `D:\src\scipysrc\scikit-learn\sklearn\tests\__init__.py`

```
# 导入所需的模块
import os
import sys
import zipfile

# 定义一个函数，用于解压缩指定的 ZIP 文件
def unzip_file(zip_file, dest_dir):
    # 如果目标目录不存在，则创建它
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    
    # 打开 ZIP 文件为二进制模式
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        # 解压缩 ZIP 文件中的所有文件到目标目录
        zip_ref.extractall(dest_dir)

# 测试函数调用
unzip_file('example.zip', 'destination_folder')
```