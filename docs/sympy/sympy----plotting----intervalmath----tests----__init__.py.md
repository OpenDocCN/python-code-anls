# `D:\src\scipysrc\sympy\sympy\plotting\intervalmath\tests\__init__.py`

```
# 导入所需的模块：`BytesIO`用于处理二进制数据，`zipfile`用于操作ZIP文件
from io import BytesIO
import zipfile

# 定义函数`extract_zip_contents`，接收一个ZIP文件路径作为参数
def extract_zip_contents(zip_path):
    # 使用二进制模式打开ZIP文件，创建一个字节流对象`bio`
    bio = BytesIO(open(zip_path, 'rb').read())
    
    # 使用`zipfile.ZipFile`打开字节流，创建ZIP文件对象`zipfile`
    zipf = zipfile.ZipFile(bio, 'r')
    
    # 初始化一个空字典`file_contents`，用于存储文件名和文件内容的对应关系
    file_contents = {}
    
    # 遍历ZIP文件中所有的文件名
    for file_name in zipf.namelist():
        # 将文件名作为键，使用`zipfile.read`方法读取文件内容作为对应的值，存入字典`file_contents`
        file_contents[file_name] = zipf.read(file_name)
    
    # 关闭ZIP文件对象
    zipf.close()
    
    # 返回包含ZIP文件中所有文件名和内容的字典`file_contents`
    return file_contents
```