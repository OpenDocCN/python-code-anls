# `.\DB-GPT-src\dbgpt\serve\datasource\service\__init__.py`

```py
# 导入必要的模块：io 用于处理字节流，zipfile 用于处理 ZIP 文件
import io
import zipfile

# 定义函数 unzip_file，接受一个文件名参数 fname
def unzip_file(fname):
    # 以二进制只读模式打开文件 fname，创建一个文件对象 bio
    bio = io.BytesIO(open(fname, 'rb').read())
    # 使用字节流创建一个 ZipFile 对象 zip，以只读模式打开 ZIP 文件
    zip = zipfile.ZipFile(bio, 'r')
    # 使用字典推导式遍历 zip 文件中的每个文件名，将文件名及其内容存储在字典 fdict 中
    fdict = {n: zip.read(n) for n in zip.namelist()}
    # 关闭 ZipFile 对象 zip
    zip.close()
    # 返回包含 ZIP 文件中所有文件名及其内容的字典 fdict
    return fdict
```