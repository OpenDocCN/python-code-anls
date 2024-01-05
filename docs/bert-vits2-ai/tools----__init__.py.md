# `d:/src/tocomm/Bert-VITS2\tools\__init__.py`

```
# 根据 ZIP 文件名读取内容，返回其中文件名到数据的字典
def read_zip(fname):
    # 根据 ZIP 文件名读取其二进制，封装成字节流
    bio = BytesIO(open(fname, 'rb').read())  # 创建一个字节流对象，将指定文件的二进制数据封装到字节流中
    使用字节流里面内容创建 ZIP 对象  # 创建一个 ZIP 对象，使用字节流中的内容
    zip = zipfile.ZipFile(bio, 'r')  # 使用字节流中的内容创建一个 ZIP 对象，以只读模式打开
    遍历 ZIP 对象所包含文件的文件名，读取文件数据，组成文件名到数据的字典  # 遍历 ZIP 对象中包含的文件名列表，读取每个文件的数据，将文件名和数据组成字典
    fdict = {n:zip.read(n) for n in zip.namelist()}  # 使用字典推导式，将文件名和对应的数据组成字典
    # 关闭 ZIP 对象
    zip.close()  # 关闭 ZIP 对象
    # 返回结果字典
    return fdict  # 返回文件名到数据的字典
```