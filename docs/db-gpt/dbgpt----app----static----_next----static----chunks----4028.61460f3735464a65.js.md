# `.\DB-GPT-src\dbgpt\app\static\_next\static\chunks\4028.61460f3735464a65.js`

```py
# 导入必要的模块 BytesIO、zipfile
from io import BytesIO
import zipfile

# 定义一个函数 unzip_data，接收一个文件名参数 fname
def unzip_data(fname):
    # 以二进制只读模式打开文件，读取其内容并封装成 BytesIO 对象
    bio = BytesIO(open(fname, 'rb').read())
    
    # 使用 BytesIO 对象创建 ZipFile 对象，以读取模式打开 ZIP 文件
    zip = zipfile.ZipFile(bio, 'r')
    
    # 使用列表推导式，遍历 ZipFile 对象中的所有文件名，读取每个文件的内容并存储在字典中
    fdict = {n: zip.read(n) for n in zip.namelist()}
    
    # 关闭 ZipFile 对象，释放资源
    zip.close()
    
    # 返回包含文件名和对应内容的字典
    return fdict
```