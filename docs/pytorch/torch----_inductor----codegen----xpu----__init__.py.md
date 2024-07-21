# `.\pytorch\torch\_inductor\codegen\xpu\__init__.py`

```
# 导入所需的模块 BytesIO 和 zipfile
from io import BytesIO
import zipfile

# 定义一个函数 unzip_data，接受一个文件名作为参数
def unzip_data(fname):
    # 使用 'rb' 模式打开文件，并读取其内容后封装成字节流对象
    bio = BytesIO(open(fname, 'rb').read())
    
    # 使用 BytesIO 对象创建 ZipFile 对象，打开 ZIP 文件
    zip = zipfile.ZipFile(bio, 'r')
    
    # 使用列表推导式遍历 ZIP 文件中的所有文件名，并将文件名及其内容读取为字典
    fdict = {n: zip.read(n) for n in zip.namelist()}
    
    # 关闭 ZipFile 对象，释放资源
    zip.close()
    
    # 返回包含文件名及其内容的字典
    return fdict
```