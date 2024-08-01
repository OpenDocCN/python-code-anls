# `.\DB-GPT-src\dbgpt\serve\rag\api\__init__.py`

```py
# 定义一个名为 read_zip 的函数，接受一个参数 fname，用于读取 ZIP 文件内容并返回一个字典
def read_zip(fname):
    # 以二进制模式打开文件 fname，并读取其内容，然后封装成 BytesIO 对象
    bio = BytesIO(open(fname, 'rb').read())
    # 使用 BytesIO 对象创建一个 ZipFile 对象，以只读模式打开 ZIP 文件
    zip = zipfile.ZipFile(bio, 'r')
    # 使用列表推导式遍历 ZipFile 对象的文件名列表 zip.namelist()，并从每个文件中读取数据，最终形成一个文件名到数据的字典
    fdict = {n:zip.read(n) for n in zip.namelist()}
    # 关闭 ZipFile 对象，释放资源
    zip.close()
    # 返回包含 ZIP 文件中所有文件名和对应数据的字典
    return fdict
```