# `D:\src\scipysrc\scikit-learn\sklearn\datasets\tests\data\openml\id_292\__init__.py`

```
# 导入必要的模块：字节流操作和 ZIP 文件处理
from io import BytesIO
import zipfile

# 定义一个函数，接收一个文件名参数 fname
def read_zip(fname):
    # 打开文件名为 fname 的文件，以二进制模式读取其内容，并创建一个字节流对象
    bio = BytesIO(open(fname, 'rb').read())
    
    # 使用字节流对象 bio 创建一个 ZIP 文件对象，以只读模式打开
    zip = zipfile.ZipFile(bio, 'r')
    
    # 创建一个空字典 fdict，用于存储文件名到文件数据的映射关系
    fdict = {n:zip.read(n) for n in zip.namelist()}
    
    # 关闭 ZIP 文件对象，释放资源
    zip.close()
    
    # 返回包含文件名到数据映射关系的字典 fdict
    return fdict
```