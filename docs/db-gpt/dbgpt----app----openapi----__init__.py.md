# `.\DB-GPT-src\dbgpt\app\openapi\__init__.py`

```py
# 导入所需的模块 BytesIO 和 zipfile
from io import BytesIO
import zipfile

# 定义一个函数，接收一个 ZIP 文件名作为参数，返回一个字典
def process_zipfile(fname):
    # 使用二进制模式打开指定文件，并将其内容读取到内存中的字节流
    bio = BytesIO(open(fname, 'rb').read())
    
    # 使用字节流创建一个 ZIP 文件对象
    zip = zipfile.ZipFile(bio, 'r')
    
    # 使用列表推导式遍历 ZIP 文件中所有的文件名，并读取每个文件的内容，构建成字典
    fdict = {n: zip.read(n) for n in zip.namelist()}
    
    # 关闭 ZIP 文件对象，释放资源
    zip.close()
    
    # 返回包含文件名及其内容的字典
    return fdict
```