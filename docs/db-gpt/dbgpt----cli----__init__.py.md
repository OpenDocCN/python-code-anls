# `.\DB-GPT-src\dbgpt\cli\__init__.py`

```py
# 导入必要的模块 BytesIO 和 zipfile
from io import BytesIO
import zipfile

# 定义一个函数 extract_zip，接收一个 ZIP 文件的字节数据和要提取的文件名列表
def extract_zip(zip_data, filenames):
    # 创建一个 BytesIO 对象，用于将二进制数据封装成字节流
    bio = BytesIO(zip_data)
    # 使用字节流创建一个 ZIP 对象
    zip_file = zipfile.ZipFile(bio, 'r')
    
    # 初始化一个空字典，用于存放文件名和其对应的数据
    extracted_data = {}
    
    # 遍历传入的文件名列表
    for filename in filenames:
        # 读取 ZIP 文件中特定文件名的数据，并存入字典
        extracted_data[filename] = zip_file.read(filename)
    
    # 关闭 ZIP 对象
    zip_file.close()
    
    # 返回提取出的文件数据字典
    return extracted_data
```