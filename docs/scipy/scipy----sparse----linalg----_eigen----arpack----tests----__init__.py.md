# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_eigen\arpack\tests\__init__.py`

```
# 定义一个名为 read_zip 的函数，接收一个参数 fname，用于读取 ZIP 文件并返回文件名到数据的映射字典
def read_zip(fname):
    # 打开指定文件名的文件，并以二进制读取模式读取其内容，然后将内容封装到一个 BytesIO 对象中
    bio = BytesIO(open(fname, 'rb').read())
    # 使用 BytesIO 对象创建一个 ZipFile 对象，以便后续对 ZIP 文件内容进行操作
    zip = zipfile.ZipFile(bio, 'r')
    # 使用字典推导式遍历 ZipFile 对象的所有文件名，依次读取每个文件的内容，并将文件名和内容存储在字典 fdict 中
    fdict = {n:zip.read(n) for n in zip.namelist()}
    # 关闭 ZipFile 对象，释放资源
    zip.close()
    # 返回包含文件名和对应数据的字典 fdict
    return fdict
```