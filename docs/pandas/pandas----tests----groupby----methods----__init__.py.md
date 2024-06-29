# `D:\src\scipysrc\pandas\pandas\tests\groupby\methods\__init__.py`

```
# 导入必要的模块：io 用于处理字节流，zipfile 用于处理 ZIP 文件
import io
import zipfile

# 定义一个函数 extract_zip，接受一个 ZIP 文件名作为参数
def extract_zip(zip_file):
    # 打开 ZIP 文件为二进制读取模式，创建一个字节流对象
    bio = io.BytesIO(open(zip_file, 'rb').read())
    # 用 BytesIO 对象创建一个 zipfile.ZipFile 对象，以读取模式打开
    zip_obj = zipfile.ZipFile(bio, 'r')
    # 获取 ZIP 文件中所有文件的名称列表，并以此创建一个空字典 fdict
    fdict = {name: zip_obj.read(name) for name in zip_obj.namelist()}
    # 关闭 zipfile.ZipFile 对象，释放资源
    zip_obj.close()
    # 返回包含 ZIP 文件中所有文件名及其数据的字典 fdict
    return fdict
```