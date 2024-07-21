# `.\pytorch\benchmarks\instruction_counts\core\__init__.py`

```py
# 导入 Python 内置的 zipfile 模块，用于处理 ZIP 文件
import zipfile

# 导入 BytesIO 类，用于创建一个在内存中操作二进制数据的流
from io import BytesIO

# 定义一个函数 unzip_files，接受一个 ZIP 文件名作为参数
def unzip_files(zip_file):
    # 打开指定的 ZIP 文件，'rb' 表示以二进制读取模式打开文件
    with open(zip_file, 'rb') as f:
        # 创建一个 BytesIO 对象，用来在内存中操作二进制数据
        bio = BytesIO(f.read())
    
    # 使用创建的 BytesIO 对象来创建一个 ZipFile 对象，'r' 表示读取模式
    zip_obj = zipfile.ZipFile(bio, 'r')
    
    # 获取 ZIP 文件中所有文件的文件名列表
    file_list = zip_obj.namelist()
    
    # 定义一个空字典，用于存储文件名和对应的文件内容
    file_contents = {}
    
    # 遍历文件名列表
    for file_name in file_list:
        # 使用 ZipFile 对象的 read 方法读取文件内容并存储到字典中
        file_contents[file_name] = zip_obj.read(file_name)
    
    # 关闭 ZipFile 对象
    zip_obj.close()
    
    # 返回存储文件名和文件内容的字典
    return file_contents
```