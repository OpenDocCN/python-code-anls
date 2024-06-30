# `D:\src\scipysrc\seaborn\tests\__init__.py`

```
# 导入所需的字节流操作和 ZIP 文件处理模块
from io import BytesIO
import zipfile

# 定义一个函数，接收一个文件名作为参数，用于读取 ZIP 文件中的内容并返回文件名到数据的字典
def read_zip(fname):
    # 读取指定文件名的文件，并以二进制方式读取其内容，然后将其封装成字节流对象
    bio = BytesIO(open(fname, 'rb').read())
    
    # 使用字节流创建一个 ZipFile 对象，打开该 ZIP 文件以供进一步操作
    zip = zipfile.ZipFile(bio, 'r')
    
    # 使用 zip.namelist() 获取 ZIP 文件中所有文件的文件名列表，并将每个文件名对应的文件内容读取到字典中
    fdict = {n:zip.read(n) for n in zip.namelist()}
    
    # 关闭 ZipFile 对象，释放资源
    zip.close()
    
    # 返回读取到的文件名及其对应数据的字典
    return fdict
```