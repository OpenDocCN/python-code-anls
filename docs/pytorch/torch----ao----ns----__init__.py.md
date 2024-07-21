# `.\pytorch\torch\ao\ns\__init__.py`

```
# 定义一个名为 read_zip 的函数，接收一个参数 fname，用于读取 ZIP 文件并返回文件名到数据的字典
def read_zip(fname):
    # 打开文件 fname 为二进制模式，并读取其内容，将内容封装为 BytesIO 对象
    bio = BytesIO(open(fname, 'rb').read())
    # 使用 BytesIO 对象创建一个 zipfile.ZipFile 对象，模式为只读 ('r')
    zip = zipfile.ZipFile(bio, 'r')
    # 使用列表推导式创建一个字典 fdict，字典的键是 ZIP 文件中的每个文件名，值是对应文件名的内容数据
    fdict = {n:zip.read(n) for n in zip.namelist()}
    # 关闭 zipfile.ZipFile 对象，释放资源
    zip.close()
    # 返回包含 ZIP 文件中所有文件名及其内容数据的字典
    return fdict
```