# `D:\src\scipysrc\sympy\sympy\physics\biomechanics\tests\__init__.py`

```
# 导入所需模块：io 模块用于处理字节流，zipfile 模块用于操作 ZIP 文件
import io
import zipfile

# 定义一个函数，接收一个文件名作为参数，用于读取 ZIP 文件并返回文件名到数据的字典
def read_zip(fname):
    # 打开指定文件名的二进制文件，并读取其内容后封装成字节流对象
    bio = io.BytesIO(open(fname, 'rb').read())
    # 使用字节流创建一个 ZIP 文件对象，以只读模式打开
    zip = zipfile.ZipFile(bio, 'r')
    # 使用 ZIP 文件对象的 namelist() 方法获取 ZIP 文件中所有文件的名称列表，
    # 并通过字典推导式创建一个字典，键为文件名，值为对应文件的内容数据
    fdict = {n: zip.read(n) for n in zip.namelist()}
    # 关闭 ZIP 文件对象，释放资源
    zip.close()
    # 返回包含文件名到数据的字典的结果
    return fdict
```