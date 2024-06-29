# `D:\src\scipysrc\pandas\pandas\tests\indexes\__init__.py`

```
# 导入所需的模块：io模块用于处理IO操作，zipfile模块用于处理ZIP文件
import io
import zipfile

# 定义一个函数，接收一个ZIP文件名作为参数，并返回一个包含文件名到数据的字典
def read_zip(fname):
    # 以二进制只读模式打开文件，并将文件内容封装成一个字节流对象（BytesIO对象）
    bio = io.BytesIO(open(fname, 'rb').read())
    
    # 使用BytesIO对象创建一个ZipFile对象，以便操作ZIP文件内容
    zip = zipfile.ZipFile(bio, 'r')
    
    # 通过ZipFile对象的namelist()方法获取ZIP文件中所有文件的文件名列表，
    # 然后使用字典推导式读取每个文件的数据，生成文件名到数据的映射关系，并赋值给fdict变量
    fdict = {n: zip.read(n) for n in zip.namelist()}
    
    # 关闭ZipFile对象，释放资源
    zip.close()
    
    # 返回包含文件名到数据映射关系的字典
    return fdict
```