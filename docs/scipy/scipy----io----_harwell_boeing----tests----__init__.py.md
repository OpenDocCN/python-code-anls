# `D:\src\scipysrc\scipy\scipy\io\_harwell_boeing\tests\__init__.py`

```
# 导入必要的模块 BytesIO 和 zipfile
from io import BytesIO
import zipfile

# 定义函数 extract_zip，接收一个 ZIP 文件的字节流和目标文件名
def extract_zip(zip_bytes, filename):
    # 创建一个 BytesIO 对象，用于将 ZIP 文件的字节流封装成可读的文件对象
    bio = BytesIO(zip_bytes)
    
    # 创建一个 ZipFile 对象，打开这个 ZIP 文件
    zip_file = zipfile.ZipFile(bio, 'r')
    
    # 尝试从 ZIP 文件中读取指定文件名的内容
    try:
        # 读取目标文件的内容为一个字节串
        content = zip_file.read(filename)
        # 返回读取到的内容
        return content
    except KeyError:
        # 如果指定文件名不存在于 ZIP 文件中，返回 None
        return None
    finally:
        # 无论是否成功读取，都要关闭 ZipFile 对象
        zip_file.close()
```