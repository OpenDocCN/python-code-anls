# `.\pytorch\test\quantization\bc\__init__.py`

```py
# 定义一个名为 read_zip 的函数，接受一个参数 fname，用于读取指定的 ZIP 文件并返回其中的文件名到数据的字典
def read_zip(fname):
    # 读取指定文件名 fname 的内容，并以二进制形式打开，然后将其封装成一个字节流对象
    bio = BytesIO(open(fname, 'rb').read())
    # 使用 BytesIO 对象 bio 创建一个 ZipFile 对象，以便后续操作
    zip = zipfile.ZipFile(bio, 'r')
    # 使用字典推导式从 ZipFile 对象中获取所有文件的文件名，并读取每个文件的内容，存储为文件名到数据的映射关系
    fdict = {n: zip.read(n) for n in zip.namelist()}
    # 关闭 ZipFile 对象，释放资源
    zip.close()
    # 返回包含文件名到数据映射关系的字典对象
    return fdict
```