# `.\pytorch\torch\testing\_internal\generated\__init__.py`

```py
# 导入所需的模块 BytesIO 和 zipfile
from io import BytesIO
import zipfile

# 定义一个函数 unzip_files，接收一个参数 fname 作为 ZIP 文件名
def unzip_files(fname):
    # 读取指定文件名的 ZIP 文件，模式为读取二进制数据，并将其包装成 BytesIO 对象
    bio = BytesIO(open(fname, 'rb').read())
    
    # 使用 BytesIO 对象创建 ZipFile 对象，模式为只读
    zip = zipfile.ZipFile(bio, 'r')
    
    # 使用列表推导式遍历 ZipFile 对象的所有文件名，读取每个文件的内容，形成文件名到数据的字典
    file_contents = {name: zip.read(name) for name in zip.namelist()}
    
    # 关闭 ZipFile 对象，释放资源
    zip.close()
    
    # 返回包含文件名到数据的字典的结果
    return file_contents
```