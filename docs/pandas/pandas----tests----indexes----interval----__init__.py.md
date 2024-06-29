# `D:\src\scipysrc\pandas\pandas\tests\indexes\interval\__init__.py`

```
# 导入必要的模块：io 用于处理文件流，zipfile 用于操作 ZIP 文件
import io
import zipfile

# 定义函数 unzip_data，接收一个文件名参数 fname
def unzip_data(fname):
    # 以只读二进制模式打开文件，并将内容读取到内存中的字节流
    bio = io.BytesIO(open(fname, 'rb').read())
    
    # 使用字节流创建一个 ZIP 文件对象
    zip = zipfile.ZipFile(bio, 'r')
    
    # 使用列表推导式遍历 ZIP 文件中的所有文件名，读取每个文件的内容，并存储在一个字典中
    fdict = {n: zip.read(n) for n in zip.namelist()}
    
    # 关闭 ZIP 文件对象，释放资源
    zip.close()
    
    # 返回包含文件名和数据的字典
    return fdict
```