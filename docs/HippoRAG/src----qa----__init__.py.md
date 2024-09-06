# `.\HippoRAG\src\qa\__init__.py`

```py
# 导入所需的模块
import zipfile
from io import BytesIO

# 定义函数，读取 ZIP 文件内容并返回文件名到数据的字典
def read_zip(fname):
    # 以二进制模式打开文件，读取文件内容并创建一个字节流对象
    bio = BytesIO(open(fname, 'rb').read())
    # 使用字节流对象创建一个 ZIP 文件对象
    zip = zipfile.ZipFile(bio, 'r')
    # 遍历 ZIP 文件中的每个文件名，并读取其内容，生成文件名到内容的字典
    fdict = {n:zip.read(n) for n in zip.namelist()}
    # 关闭 ZIP 文件对象
    zip.close()
    # 返回包含所有文件内容的字典
    return fdict
```