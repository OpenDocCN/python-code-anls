# `.\DB-GPT-src\dbgpt\model\cluster\tests\registry_impl\__init__.py`

```py
# 导入所需的模块 BytesIO 和 zipfile
from io import BytesIO
import zipfile

# 定义一个函数 unzip_files，接受一个文件名作为参数
def unzip_files(zip_file):
    # 以二进制只读模式打开指定的 ZIP 文件，创建一个 BytesIO 对象并将文件内容读入其中
    bio = BytesIO(open(zip_file, 'rb').read())
    
    # 使用 BytesIO 对象创建一个 ZipFile 对象，以便操作 ZIP 文件内容
    zip_obj = zipfile.ZipFile(bio, 'r')
    
    # 通过调用 namelist() 方法获取 ZIP 文件中所有文件的名称列表，并以此构建一个字典
    files_dict = {name: zip_obj.read(name) for name in zip_obj.namelist()}
    
    # 关闭 ZipFile 对象，释放资源
    zip_obj.close()
    
    # 返回包含文件名和其对应数据的字典
    return files_dict
```