# `.\DB-GPT-src\tests\intetration_tests\vector_store\__init__.py`

```py
# 导入必要的模块 BytesIO 和 zipfile
from io import BytesIO
import zipfile

# 定义一个函数 unzip_files，接收一个 ZIP 文件名作为参数
def unzip_files(zip_file):
    # 打开并读取 ZIP 文件，将其内容封装成字节流对象
    bio = BytesIO(open(zip_file, 'rb').read())
    # 使用字节流创建一个 ZipFile 对象，以读取模式打开
    zip_obj = zipfile.ZipFile(bio, 'r')
    # 使用列表推导式遍历 ZipFile 对象中的文件名列表，将文件名与其对应的内容读取出来形成字典
    files_dict = {name: zip_obj.read(name) for name in zip_obj.namelist()}
    # 关闭 ZipFile 对象，释放资源
    zip_obj.close()
    # 返回包含解压后文件名及其内容的字典
    return files_dict
```