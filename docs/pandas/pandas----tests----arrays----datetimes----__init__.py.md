# `D:\src\scipysrc\pandas\pandas\tests\arrays\datetimes\__init__.py`

```
# 定义一个名为 read_zip 的函数，接受一个参数 fname，用于读取 ZIP 文件并返回其内容的字典
def read_zip(fname):
    # 以二进制只读模式打开文件 fname，并读取其内容后封装成 BytesIO 对象
    bio = BytesIO(open(fname, 'rb').read())
    # 使用 BytesIO 对象创建一个 zipfile.ZipFile 对象，以便操作 ZIP 文件
    zip = zipfile.ZipFile(bio, 'r')
    # 使用字典推导式遍历 ZIP 文件中所有文件的文件名，并将每个文件名与其内容读取后的数据对应起来，形成字典 fdict
    fdict = {n: zip.read(n) for n in zip.namelist()}
    # 关闭 zipfile.ZipFile 对象，释放资源
    zip.close()
    # 返回包含 ZIP 文件中所有文件名及其对应数据的字典 fdict
    return fdict
```