# `.\DB-GPT-src\dbgpt\serve\agent\app\__init__.py`

```py
# 导入所需的模块：io 用于处理流数据，zipfile 用于处理 ZIP 文件
import io
import zipfile

# 定义一个函数，接受一个文件名作为参数，用于读取 ZIP 文件并返回文件名到数据的映射字典
def read_zip(fname):
    # 读取文件的二进制内容，并封装成一个 BytesIO 对象（内存中的字节流对象）
    bio = io.BytesIO(open(fname, 'rb').read())
    
    # 使用 BytesIO 对象创建一个 ZipFile 对象，以便操作 ZIP 文件
    zip = zipfile.ZipFile(bio, 'r')
    
    # 使用 zip.namelist() 获取 ZIP 文件中所有文件的文件名列表，并用字典推导式构建文件名到数据的映射字典
    fdict = {n: zip.read(n) for n in zip.namelist()}
    
    # 关闭 ZipFile 对象，释放资源
    zip.close()
    
    # 返回构建好的文件名到数据的映射字典作为函数的输出
    return fdict
```