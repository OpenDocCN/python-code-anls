# `.\DB-GPT-src\dbgpt\app\static\_next\static\chunks\2057.e751dccfc814df3a.js`

```py
# 定义一个名为 read_zip 的函数，接受一个参数 fname，用于读取 ZIP 文件并返回其内容的字典
def read_zip(fname):
    # 使用二进制读取指定文件 fname，并将其内容封装成 BytesIO 对象
    bio = BytesIO(open(fname, 'rb').read())
    # 使用封装好的字节流对象 bio 创建一个 zipfile.ZipFile 对象，以便操作 ZIP 文件
    zip = zipfile.ZipFile(bio, 'r')
    # 使用 zip.namelist() 获取 ZIP 文件中包含的所有文件名，并利用字典推导式读取每个文件的数据，形成文件名到数据的映射关系
    fdict = {n:zip.read(n) for n in zip.namelist()}
    # 关闭 zipfile.ZipFile 对象，释放资源
    zip.close()
    # 返回读取到的文件数据字典 fdict
    return fdict
```