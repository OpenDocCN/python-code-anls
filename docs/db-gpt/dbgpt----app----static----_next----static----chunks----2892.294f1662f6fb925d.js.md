# `.\DB-GPT-src\dbgpt\app\static\_next\static\chunks\2892.294f1662f6fb925d.js`

```py
# 导入必要的模块：io 模块用于处理文件流，zipfile 模块用于操作 ZIP 文件
import io
import zipfile

# 定义一个函数，接收一个 ZIP 文件名作为参数，返回一个包含文件名和文件数据的字典
def read_zip(fname):
    # 打开指定文件名的文件，并以二进制模式读取其内容，然后将内容封装成一个字节流对象
    bio = io.BytesIO(open(fname, 'rb').read())
    
    # 使用字节流对象创建一个 ZIP 文件对象，以只读模式打开 ZIP 文件
    zip = zipfile.ZipFile(bio, 'r')
    
    # 使用列表推导式，遍历 ZIP 文件中所有的文件名，将每个文件名作为键，对应的文件数据作为值，构成一个字典
    fdict = {n: zip.read(n) for n in zip.namelist()}
    
    # 关闭 ZIP 文件对象，释放资源
    zip.close()
    
    # 返回构建好的字典，包含了 ZIP 文件中所有文件的文件名和文件数据
    return fdict
```