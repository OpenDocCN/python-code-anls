# `.\pytorch\test\jit\fixtures_srcs\__init__.py`

```
# 导入所需模块：`BytesIO` 用于创建二进制数据的内存缓冲区，`zipfile` 用于处理 ZIP 文件
from io import BytesIO
import zipfile

# 定义函数 `read_zip`，接受一个文件名参数 `fname`
def read_zip(fname):
    # 打开指定文件名的文件，以二进制只读模式读取其内容，并封装成字节流对象 `bio`
    bio = BytesIO(open(fname, 'rb').read())
    
    # 使用字节流 `bio` 创建一个 ZIP 文件对象 `zip`
    zip = zipfile.ZipFile(bio, 'r')
    
    # 使用列表推导式遍历 ZIP 文件对象中的所有文件名，读取每个文件的内容，生成文件名到数据的字典 `fdict`
    fdict = {n:zip.read(n) for n in zip.namelist()}
    
    # 关闭 ZIP 文件对象 `zip`
    zip.close()
    
    # 返回包含文件名到数据映射的字典 `fdict`
    return fdict
```