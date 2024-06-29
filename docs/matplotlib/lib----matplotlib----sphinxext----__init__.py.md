# `D:\src\scipysrc\matplotlib\lib\matplotlib\sphinxext\__init__.py`

```py
# 导入必要的模块 BytesIO 和 zipfile
from io import BytesIO
import zipfile

# 定义函数 unzip_files，接收一个 ZIP 文件名作为参数
def unzip_files(zip_file):
    # 使用二进制模式打开 ZIP 文件，读取其内容，并创建 BytesIO 对象
    bio = BytesIO(open(zip_file, 'rb').read())
    
    # 使用 BytesIO 对象创建 ZipFile 对象，模式为只读 'r'
    zip = zipfile.ZipFile(bio, 'r')
    
    # 使用列表推导式遍历 ZipFile 对象中的所有文件名，并读取每个文件的内容到字典中
    file_contents = {name: zip.read(name) for name in zip.namelist()}
    
    # 关闭 ZipFile 对象
    zip.close()
    
    # 返回包含文件名和内容的字典
    return file_contents
```