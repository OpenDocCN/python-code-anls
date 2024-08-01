# `.\DB-GPT-src\dbgpt\app\static\_next\static\chunks\4508205f.e8ea8f62c46ba3d1.js`

```py
# 定义一个名为 read_zip 的函数，参数为 fname，用于读取 ZIP 文件内容并返回文件名到数据的字典
def read_zip(fname):
    # 打开指定文件名的文件，以二进制模式读取其内容，并将内容封装到 BytesIO 对象中
    bio = BytesIO(open(fname, 'rb').read())
    # 使用 BytesIO 对象创建一个 ZipFile 对象，以便操作 ZIP 文件
    zip = zipfile.ZipFile(bio, 'r')
    # 使用 ZipFile 对象的 namelist() 方法获取 ZIP 文件中的所有文件名列表，
    # 并以列表推导式的方式遍历每个文件名 n，并读取对应文件的内容，组成文件名到数据的字典 fdict
    fdict = {n:zip.read(n) for n in zip.namelist()}
    # 关闭 ZipFile 对象，释放资源
    zip.close()
    # 返回包含 ZIP 文件中所有文件名和对应数据的字典
    return fdict
```