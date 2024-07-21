# `.\pytorch\torch\testing\_internal\distributed\rpc\examples\__init__.py`

```py
# 导入必要的模块 BytesIO 和 zipfile
from io import BytesIO
import zipfile

# 定义一个函数，接收一个文件名作为参数，用于读取其中的内容并返回一个字典
def read_zip(fname):
    # 打开指定文件名的二进制文件，并将其内容读取到内存中的字节流中
    bio = BytesIO(open(fname, 'rb').read())
    # 使用字节流创建一个 ZipFile 对象，以便操作 ZIP 文件
    zip = zipfile.ZipFile(bio, 'r')
    # 创建一个空字典，用于存储文件名和对应的文件数据
    fdict = {n:zip.read(n) for n in zip.namelist()}
    # 关闭 ZipFile 对象，释放资源
    zip.close()
    # 返回存储文件名和对应数据的字典
    return fdict
```