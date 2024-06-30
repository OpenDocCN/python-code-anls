# `D:\src\scipysrc\scipy\scipy\special\tests\data\__init__.py`

```
# 定义一个函数 named read_zip，接收一个参数 fname
def read_zip(fname):
    # 以二进制只读模式打开文件 fname，读取其内容并存入 BytesIO 对象 bio
    bio = BytesIO(open(fname, 'rb').read())
    # 使用 BytesIO 对象 bio 创建一个 ZipFile 对象 zip，以只读模式打开 ZIP 文件
    zip = zipfile.ZipFile(bio, 'r')
    # 使用字典推导式，遍历 ZipFile 对象 zip 中的所有文件名，并读取每个文件的内容，将结果存入字典 fdict
    fdict = {n:zip.read(n) for n in zip.namelist()}
    # 关闭 ZipFile 对象 zip，释放资源
    zip.close()
    # 返回包含文件名到数据映射的字典 fdict
    return fdict
```