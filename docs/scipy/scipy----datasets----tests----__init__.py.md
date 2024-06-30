# `D:\src\scipysrc\scipy\scipy\datasets\tests\__init__.py`

```
# 导入所需的字节流和 ZIP 文件处理模块
from io import BytesIO
import zipfile

# 定义一个函数，接收一个 ZIP 文件名作为参数，返回一个字典
def process_zip(fname):
    # 使用二进制模式打开指定文件，读取其中的内容并封装成字节流对象
    bio = BytesIO(open(fname, 'rb').read())
    
    # 使用字节流创建一个 ZIP 文件对象，模式为只读
    zip = zipfile.ZipFile(bio, 'r')
    
    # 创建一个空字典，用于存储文件名到文件内容的映射关系
    fdict = {n: zip.read(n) for n in zip.namelist()}
    
    # 关闭 ZIP 文件对象，释放资源
    zip.close()
    
    # 返回处理后的字典，其中包含了 ZIP 文件中每个文件名对应的内容
    return fdict
```