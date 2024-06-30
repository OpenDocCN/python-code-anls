# `D:\src\scipysrc\scikit-learn\sklearn\datasets\tests\data\openml\id_42585\__init__.py`

```
# 定义一个函数，名称为 read_zip，接收一个参数 fname，用于读取指定的 ZIP 文件并返回其内容
def read_zip(fname):
    # 使用二进制模式打开文件 fname，并读取其内容，然后将内容封装到 BytesIO 对象中
    bio = BytesIO(open(fname, 'rb').read())
    # 使用封装好的字节流创建一个 ZipFile 对象，以便操作 ZIP 文件
    zip = zipfile.ZipFile(bio, 'r')
    # 创建一个空字典 fdict，用于存储 ZIP 文件中每个文件名对应的文件内容
    fdict = {n:zip.read(n) for n in zip.namelist()}
    # 关闭 ZipFile 对象，释放资源
    zip.close()
    # 返回包含 ZIP 文件内容的字典 fdict
    return fdict
```