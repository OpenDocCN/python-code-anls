# `D:\src\scipysrc\pandas\pandas\tests\indexes\timedeltas\__init__.py`

```
# 导入必要的模块 BytesIO 和 zipfile
from io import BytesIO
import zipfile

# 定义一个函数 unzip_files，接收一个 ZIP 文件名作为参数
def unzip_files(zip_file):
    # 以二进制只读方式打开给定的 ZIP 文件，并读取其内容
    zip_data = open(zip_file, 'rb').read()
    # 使用 BytesIO 将 ZIP 文件内容封装成字节流对象
    bio = BytesIO(zip_data)
    # 使用 zipfile 模块打开这个字节流对象作为 ZIP 文件，模式为只读
    zip_archive = zipfile.ZipFile(bio, 'r')
    # 获取 ZIP 文件中所有文件的文件名列表
    file_list = zip_archive.namelist()
    # 遍历文件名列表，并逐个读取文件内容，存放在一个字典中
    file_contents = {fname: zip_archive.read(fname) for fname in file_list}
    # 关闭 ZIP 文件
    zip_archive.close()
    # 返回包含文件名和文件内容的字典
    return file_contents
```