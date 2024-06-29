# `D:\src\scipysrc\pandas\pandas\core\util\__init__.py`

```
# 导入必要的模块 BytesIO、zipfile
from io import BytesIO
import zipfile

# 定义一个函数 extract_zip，接收一个文件名参数 fname
def extract_zip(fname):
    # 打开文件 fname，以二进制只读模式读取文件内容，并将内容封装为 BytesIO 对象
    bio = BytesIO(open(fname, 'rb').read())
    # 使用 BytesIO 对象创建一个 zipfile.ZipFile 对象，以只读模式打开 ZIP 文件
    zip = zipfile.ZipFile(bio, 'r')
    # 初始化一个空字典 fdict，用于存储文件名到文件内容的映射关系
    fdict = {}
    # 遍历 ZIP 文件中所有的文件名列表
    for n in zip.namelist():
        # 将 ZIP 文件中的每个文件名作为键，对应的文件内容作为值存入 fdict 字典中
        fdict[n] = zip.read(n)
    # 关闭 zipfile.ZipFile 对象，释放资源
    zip.close()
    # 返回存储了 ZIP 文件中所有文件名及其内容的字典 fdict
    return fdict
```