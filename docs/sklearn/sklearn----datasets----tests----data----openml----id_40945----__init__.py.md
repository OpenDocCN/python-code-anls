# `D:\src\scipysrc\scikit-learn\sklearn\datasets\tests\data\openml\id_40945\__init__.py`

```
# 导入必要的模块：`BytesIO`用于创建二进制数据的内存缓冲区，`zipfile`用于操作ZIP文件
from io import BytesIO
import zipfile

# 定义一个函数，参数为ZIP文件名，返回一个包含文件名到数据字典的函数
def read_zip(fname):
    # 打开指定文件名的文件，使用二进制模式读取文件内容，并将内容封装到`BytesIO`对象中
    bio = BytesIO(open(fname, 'rb').read())
    
    # 使用`BytesIO`对象创建一个ZIP文件对象，打开模式为只读('r')
    zip = zipfile.ZipFile(bio, 'r')
    
    # 使用ZIP文件对象的`namelist`方法获取ZIP文件中所有文件的文件名列表，并根据每个文件名读取对应的文件数据，
    # 然后使用字典推导式将文件名和文件数据构成键值对，组成一个字典`fdict`
    fdict = {n:zip.read(n) for n in zip.namelist()}
    
    # 关闭ZIP文件对象，释放资源
    zip.close()
    
    # 返回由文件名到文件数据组成的字典
    return fdict
```