# `.\pytorch\tools\code_coverage\package\tool\parser\__init__.py`

```py
# 导入所需的模块：io 用于处理字节流，zipfile 用于处理 ZIP 文件
import io
import zipfile

# 定义一个函数，接收一个文件名作为参数，读取该 ZIP 文件的内容并返回一个字典
def read_zip(fname):
    # 以只读二进制模式打开文件，并将其内容封装为字节流对象
    bio = io.BytesIO(open(fname, 'rb').read())
    # 使用字节流创建一个 ZipFile 对象，参数 'r' 表示只读模式
    zip = zipfile.ZipFile(bio, 'r')
    # 使用列表推导式，遍历 ZipFile 对象的文件列表，生成文件名到文件数据的映射字典
    fdict = {n: zip.read(n) for n in zip.namelist()}
    # 关闭 ZipFile 对象，释放资源
    zip.close()
    # 返回包含文件名和对应数据的字典
    return fdict
```