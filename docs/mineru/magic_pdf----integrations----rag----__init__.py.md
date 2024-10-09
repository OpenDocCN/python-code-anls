# `.\MinerU\magic_pdf\integrations\rag\__init__.py`

```
# 定义一个函数，接收一个文件名作为参数
def read_zip(fname):
    # 打开指定的 ZIP 文件，读取其内容，并将其封装为字节流
    bio = BytesIO(open(fname, 'rb').read())
    # 使用字节流内容创建一个 ZIP 文件对象，模式为只读
    zip = zipfile.ZipFile(bio, 'r')
    # 遍历 ZIP 对象中的文件名，读取每个文件的内容并组成一个字典
    fdict = {n:zip.read(n) for n in zip.namelist()}
    # 关闭 ZIP 文件对象，以释放资源
    zip.close()
    # 返回包含文件名及其对应数据的字典
    return fdict
```