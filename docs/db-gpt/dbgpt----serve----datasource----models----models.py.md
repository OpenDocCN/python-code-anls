# `.\DB-GPT-src\dbgpt\serve\datasource\models\models.py`

```py
# 导入必要的模块：io 用于处理文件流，zipfile 用于操作 ZIP 文件
import io
import zipfile

# 定义一个函数 unzip_files，接收一个 ZIP 文件的路径作为参数
def unzip_files(zip_file):
    # 打开 ZIP 文件为二进制文件流对象
    with open(zip_file, 'rb') as f:
        # 创建一个 BytesIO 对象，将 ZIP 文件内容读取到内存中
        bio = io.BytesIO(f.read())
    
    # 使用 BytesIO 对象创建一个 ZipFile 对象，以便操作 ZIP 文件内容
    with zipfile.ZipFile(bio, 'r') as zip:
        # 获取 ZIP 文件中所有文件的名称列表，存储在 namelist 中
        namelist = zip.namelist()
        
        # 遍历 namelist 列表中的每个文件名，逐个解压并保存文件内容到 results 字典中
        results = {name: zip.read(name) for name in namelist}
    
    # 返回解压后的文件内容字典 results
    return results
```