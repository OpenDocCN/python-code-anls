# `D:\src\scipysrc\pandas\pandas\tests\copy_view\__init__.py`

```
# 导入所需模块：`BytesIO` 用于创建内存中的二进制数据流，`zipfile` 用于处理 ZIP 文件
from io import BytesIO
import zipfile

# 定义一个函数 `extract_zip_contents`，接受一个 ZIP 文件名作为参数
def extract_zip_contents(zip_file):
    # 使用二进制模式打开 ZIP 文件，读取其内容，并将其包装成一个字节流对象 `bio`
    bio = BytesIO(open(zip_file, 'rb').read())
    
    # 使用 `zipfile.ZipFile` 类打开 `bio`，并指定为只读模式 `r`
    zip_obj = zipfile.ZipFile(bio, 'r')
    
    # 使用 `zip_obj.namelist()` 获取 ZIP 文件中所有文件的名称列表，
    # 并通过字典推导式将每个文件名与其内容读取为一个字典 `file_contents`
    file_contents = {name: zip_obj.read(name) for name in zip_obj.namelist()}
    
    # 关闭 ZIP 文件对象，释放资源
    zip_obj.close()
    
    # 返回包含文件名到内容映射的字典 `file_contents`
    return file_contents
```