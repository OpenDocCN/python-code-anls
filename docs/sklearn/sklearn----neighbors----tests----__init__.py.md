# `D:\src\scipysrc\scikit-learn\sklearn\neighbors\tests\__init__.py`

```
# 导入所需的模块
import os
import zipfile

# 定义函数 unzip_file，接受两个参数：zip_file（ZIP 文件路径）和 extract_dir（解压目标路径）
def unzip_file(zip_file, extract_dir):
    # 如果解压目标路径不存在，则创建该路径
    if not os.path.exists(extract_dir):
        os.makedirs(extract_dir)
    
    # 打开 ZIP 文件为二进制模式，创建一个 ZipFile 对象
    zip_ref = zipfile.ZipFile(zip_file, 'r')
    
    # 遍历 ZIP 文件中的每个文件
    for file in zip_ref.namelist():
        # 从 ZIP 文件中解压当前文件到指定的解压目标路径
        zip_ref.extract(file, extract_dir)
    
    # 关闭 ZipFile 对象，释放资源
    zip_ref.close()

# 调用 unzip_file 函数，解压名为 'example.zip' 的文件到当前目录下的 'extracted' 文件夹中
unzip_file('example.zip', 'extracted')
```