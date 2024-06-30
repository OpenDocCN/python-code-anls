# `D:\src\scipysrc\sympy\sympy\unify\tests\__init__.py`

```
# 导入必要的模块 BytesIO 和 zipfile
from io import BytesIO
import zipfile

# 定义一个函数 unzip_data，接收一个文件名参数 fname
def unzip_data(fname):
    # 读取二进制文件内容并创建一个 BytesIO 对象
    bio = BytesIO(open(fname, 'rb').read())
    # 使用 BytesIO 对象创建一个 ZipFile 对象，打开 ZIP 文件
    zip = zipfile.ZipFile(bio, 'r')
    # 使用字典推导式遍历 ZIP 文件中所有文件的文件名，并读取文件内容，存储在字典 fdict 中
    fdict = {n: zip.read(n) for n in zip.namelist()}
    # 关闭 ZipFile 对象，释放资源
    zip.close()
    # 返回包含文件名到数据的字典 fdict
    return fdict
```