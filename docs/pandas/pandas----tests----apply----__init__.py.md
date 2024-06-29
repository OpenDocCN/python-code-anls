# `D:\src\scipysrc\pandas\pandas\tests\apply\__init__.py`

```
# 导入必要的模块
import os
import zipfile

# 定义一个函数，用于解压指定的 ZIP 文件
def unzip_file(zip_file, destination):
    # 如果目标文件夹不存在，则创建它
    if not os.path.exists(destination):
        os.makedirs(destination)
    
    # 打开指定的 ZIP 文件
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        # 将 ZIP 文件中的所有文件解压到目标文件夹中
        zip_ref.extractall(destination)

# 调用函数，解压名为 'archive.zip' 的文件到当前目录下的 'extracted' 文件夹
unzip_file('archive.zip', 'extracted')
```