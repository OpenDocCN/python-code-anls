# `.\DB-GPT-src\dbgpt\app\static\_next\static\chunks\180.0c02744079cffa2a.js`

```py
# 导入必要的模块：io 模块中的 BytesIO 类和 zipfile 模块
from io import BytesIO
import zipfile

# 定义一个函数 read_zip，接收一个文件名作为参数
def read_zip(fname):
    # 打开文件名为 fname 的文件，使用二进制模式 'rb' 读取其内容，并封装成 BytesIO 对象 bio
    bio = BytesIO(open(fname, 'rb').read())
    
    # 使用封装好的字节流 bio 创建一个 ZipFile 对象 zip，模式为读取模式 'r'
    zip = zipfile.ZipFile(bio, 'r')
    
    # 使用字典推导式，遍历 zip 对象中所有的文件名（namelist() 方法返回一个列表），并读取每个文件的内容
    # 最终得到一个字典 fdict，键为文件名，值为对应文件的内容
    fdict = {n:zip.read(n) for n in zip.namelist()}
    
    # 关闭 ZipFile 对象 zip，释放资源
    zip.close()
    
    # 返回包含文件名到数据映射的字典 fdict
    return fdict
```