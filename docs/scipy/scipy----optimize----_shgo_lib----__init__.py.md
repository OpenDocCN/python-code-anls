# `D:\src\scipysrc\scipy\scipy\optimize\_shgo_lib\__init__.py`

```
`
# 导入必要的模块：io 用于处理字节流，zipfile 用于操作 ZIP 文件
import io
import zipfile

# 定义函数 unzip_files，接受一个参数 fname，表示 ZIP 文件名，返回一个字典
def unzip_files(fname):
    # 打开文件 fname，以二进制读取模式读取文件内容，并将内容封装成字节流对象
    bio = io.BytesIO(open(fname, 'rb').read())
    # 使用封装的字节流对象创建一个 ZipFile 对象，使用 'r' 模式打开 ZIP 文件
    zip = zipfile.ZipFile(bio, 'r')
    # 创建一个空字典 fdict，用于存储 ZIP 文件中每个文件名及其对应的内容
    fdict = {n: zip.read(n) for n in zip.namelist()}
    # 关闭 ZipFile 对象，释放资源
    zip.close()
    # 返回存储了 ZIP 文件中文件名及其内容的字典 fdict
    return fdict
```