# `.\DB-GPT-src\dbgpt\serve\datasource\tests\__init__.py`

```py
# 定义一个名为 read_zip 的函数，接收一个参数 fname，用于读取 ZIP 文件并返回其中文件名到数据的字典
def read_zip(fname):
    # 打开指定文件 fname，以二进制模式读取其内容，并将其封装成字节流对象
    bio = BytesIO(open(fname, 'rb').read())
    
    # 使用 BytesIO 对象 bio 创建一个 zipfile.ZipFile 对象，以便操作 ZIP 文件内容
    zip = zipfile.ZipFile(bio, 'r')
    
    # 使用 zip.namelist() 方法获取 ZIP 文件中所有文件的文件名列表，并创建一个字典
    # 字典的键是文件名，值是对应文件名在 ZIP 文件中的内容（二进制数据）
    fdict = {n: zip.read(n) for n in zip.namelist()}
    
    # 关闭 zipfile.ZipFile 对象，释放资源
    zip.close()
    
    # 返回包含文件名到数据的字典的结果
    return fdict
```