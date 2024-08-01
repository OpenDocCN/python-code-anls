# `.\DB-GPT-src\dbgpt\app\static\_next\static\chunks\5237-f8ce62e2a793a23a.js`

```py
# 导入必要的模块：io 和 zipfile
import io
import zipfile

# 定义函数 unzip_data，接受一个文件名作为参数
def unzip_data(filename):
    # 打开指定文件名的二进制文件，读取其内容并将其封装在内存中的字节流对象中
    with open(filename, 'rb') as f:
        bio = io.BytesIO(f.read())
    
    # 使用字节流对象创建一个 ZipFile 对象，以读取模式打开
    zip_file = zipfile.ZipFile(bio, 'r')
    
    # 初始化一个空的字典，用于存储解压缩后的文件名和内容对
    files = {}
    
    # 遍历 ZipFile 对象中所有的文件名列表
    for file_name in zip_file.namelist():
        # 读取当前文件名对应的文件内容，并将其存储在字典中
        files[file_name] = zip_file.read(file_name)
    
    # 关闭 ZipFile 对象，释放资源
    zip_file.close()
    
    # 返回包含解压缩文件名和内容对的字典
    return files
```