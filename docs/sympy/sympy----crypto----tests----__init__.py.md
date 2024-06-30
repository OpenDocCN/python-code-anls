# `D:\src\scipysrc\sympy\sympy\crypto\tests\__init__.py`

```
# 导入所需的字节流处理和 ZIP 文件处理模块
from io import BytesIO
import zipfile

# 定义一个名为 read_zip 的函数，接受一个文件名作为参数
def read_zip(fname):
    # 使用 'rb' 模式打开文件，读取其内容，并将内容封装成字节流对象
    bio = BytesIO(open(fname, 'rb').read())
    
    # 使用字节流对象创建一个 ZIP 文件对象，以只读模式 ('r') 打开
    zip = zipfile.ZipFile(bio, 'r')
    
    # 通过遍历 ZIP 文件对象的文件名列表，构建一个字典
    # 字典的键为文件名，值为对应文件的内容（使用 zip.read(n) 读取）
    fdict = {n: zip.read(n) for n in zip.namelist()}
    
    # 关闭 ZIP 文件对象，释放资源
    zip.close()
    
    # 返回构建的字典，其中包含了 ZIP 文件中每个文件的文件名及其内容
    return fdict
```