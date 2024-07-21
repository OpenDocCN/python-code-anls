# `.\pytorch\torch\testing\_internal\distributed\rpc\jit\__init__.py`

```py
# 定义一个函数，名称为 read_zip，接受一个参数 fname（表示文件名）
def read_zip(fname):
    # 以二进制模式打开文件 fname，并读取其内容，然后封装成 BytesIO 对象并赋给变量 bio
    bio = BytesIO(open(fname, 'rb').read())
    # 使用 BytesIO 对象 bio 创建一个 zipfile.ZipFile 对象，模式为只读 'r' 模式，并赋给变量 zip
    zip = zipfile.ZipFile(bio, 'r')
    # 使用 zip 对象的 namelist 方法获取 ZIP 文件中所有文件的文件名列表，然后通过字典推导式创建字典 fdict，
    # 键为文件名 n，值为 zip 对象读取文件 n 的数据
    fdict = {n:zip.read(n) for n in zip.namelist()}
    # 关闭 zip 对象，释放资源
    zip.close()
    # 返回包含文件名到数据的字典 fdict
    return fdict
```