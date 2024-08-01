# `.\DB-GPT-src\dbgpt\util\tracer\tests\__init__.py`

```py
# 导入所需的模块：`BytesIO` 和 `zipfile`
from io import BytesIO
import zipfile

# 定义函数 `read_zip`，接收一个参数 `fname`，表示 ZIP 文件名
def read_zip(fname):
    # 读取指定文件名 `fname` 的二进制内容，并将其封装成字节流对象 `bio`
    bio = BytesIO(open(fname, 'rb').read())
    
    # 使用字节流 `bio` 创建一个 ZIP 文件对象 `zip`
    zip = zipfile.ZipFile(bio, 'r')
    
    # 通过遍历 ZIP 文件对象 `zip` 的文件名列表 `namelist()`，读取每个文件的数据，并组成字典 `fdict`
    fdict = {n:zip.read(n) for n in zip.namelist()}
    
    # 关闭 ZIP 文件对象 `zip`
    zip.close()
    
    # 返回包含文件名到数据映射的字典 `fdict`
    return fdict
```