# `.\DB-GPT-src\dbgpt\app\static\_next\static\chunks\8748.dc83f2c9574fac22.js`

```py
# 导入模块 zipfile 和 io 中的 BytesIO 类
import zipfile
from io import BytesIO

# 定义函数 unzip_files，接受一个参数 fname（ZIP 文件名）
def unzip_files(fname):
    # 打开指定文件名的 ZIP 文件，模式为只读（'r'）
    with zipfile.ZipFile(fname, 'r') as zip:
        # 获取 ZIP 文件中所有文件的名称列表
        file_names = zip.namelist()
        # 遍历每个文件名
        for file_name in file_names:
            # 读取 ZIP 文件中当前文件名对应的内容数据
            data = zip.read(file_name)
            # 输出当前文件名和其对应的内容数据的长度
            print(f"File: {file_name}, Size: {len(data)} bytes")

# 调用函数 unzip_files，传入参数为 'example.zip'
unzip_files('example.zip')
```