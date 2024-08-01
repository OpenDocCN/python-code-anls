# `.\DB-GPT-src\dbgpt\app\knowledge\request\__init__.py`

```py
# 导入必要的模块 BytesIO 和 zipfile
from io import BytesIO
import zipfile

# 定义一个函数 extract_zip，接收一个 ZIP 文件名作为参数，返回一个包含所有文件名和数据的字典
def extract_zip(zip_file):
    # 读取 ZIP 文件的二进制内容并创建 BytesIO 对象
    bio = BytesIO(open(zip_file, 'rb').read())
    
    # 使用 BytesIO 对象创建 ZipFile 对象，打开 ZIP 文件
    zip_obj = zipfile.ZipFile(bio, 'r')
    
    # 创建一个空字典，用于存储文件名和对应的数据
    file_data = {}
    
    # 遍历 ZipFile 对象中的每个文件名
    for file_name in zip_obj.namelist():
        # 读取当前文件名对应的数据，并存入字典中
        file_data[file_name] = zip_obj.read(file_name)
    
    # 关闭 ZipFile 对象，释放资源
    zip_obj.close()
    
    # 返回包含所有文件名和数据的字典
    return file_data
```