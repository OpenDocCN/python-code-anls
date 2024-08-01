# `.\DB-GPT-src\tests\intetration_tests\datasource\__init__.py`

```py
# 导入必要的模块：os（操作系统接口）、shutil（高级文件操作）、zipfile（ZIP 文件处理）、tempfile（生成临时文件和目录的模块）
import os
import shutil
import zipfile
import tempfile

# 定义一个函数，用于解压缩 ZIP 文件到指定目录
def unzip_file(zip_file, dest_dir):
    # 如果目标目录不存在，则创建它
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    
    # 打开传入的 ZIP 文件
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        # 解压缩 ZIP 文件中的所有内容到目标目录
        zip_ref.extractall(dest_dir)

# 生成一个临时目录，用于存放解压后的文件
with tempfile.TemporaryDirectory() as tmp_dir:
    # 调用解压函数，将指定的 ZIP 文件解压到临时目录中
    unzip_file('example.zip', tmp_dir)
    # 打印消息，指示文件已解压
    print("Files extracted successfully to", tmp_dir)
```