# `.\DB-GPT-src\dbgpt\app\static\_next\static\chunks\1156.a6b2e0b4646513ee.js`

```py
# 导入所需的字节流和 ZIP 文件处理模块
from io import BytesIO
import zipfile

# 定义一个函数，接收一个文件名参数，用于读取并解析 ZIP 文件内容
def read_and_parse_zip(fname):
    # 读取指定文件名的二进制内容，并将其封装为字节流对象
    bio = BytesIO(open(fname, 'rb').read())
    
    # 使用字节流创建 ZIP 文件对象，以便后续操作
    zip = zipfile.ZipFile(bio, 'r')
    
    # 创建一个空字典，用于存储 ZIP 文件中的文件名和对应的数据
    fdict = {n: zip.read(n) for n in zip.namelist()}
    
    # 关闭已打开的 ZIP 文件对象，释放资源
    zip.close()
    
    # 返回解析后的文件名到数据的字典作为结果
    return fdict
```