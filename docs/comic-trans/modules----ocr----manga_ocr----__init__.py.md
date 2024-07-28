# `.\comic-translate\modules\ocr\manga_ocr\__init__.py`

```py
# 导入所需的模块：io 模块用于处理 IO 操作，zipfile 模块用于处理 ZIP 文件
import io
import zipfile

# 定义一个函数 extract_zip，接收一个 ZIP 文件名作为参数，返回一个字典，包含 ZIP 文件中所有文件名和对应数据的映射关系
def extract_zip(zip_file):
    # 打开并读取 ZIP 文件为二进制数据，使用 io.BytesIO 封装成字节流对象
    bio = io.BytesIO(open(zip_file, 'rb').read())
    # 使用 zipfile.ZipFile 对象打开字节流对象，模式为读取 ('r')
    zip_obj = zipfile.ZipFile(bio, 'r')
    # 使用字典推导式遍历 ZIP 文件中的所有文件名，对每个文件名读取其数据，构建文件名到数据的映射关系
    file_data = {name: zip_obj.read(name) for name in zip_obj.namelist()}
    # 关闭 ZIP 文件对象，释放资源
    zip_obj.close()
    # 返回包含所有文件名和对应数据的字典
    return file_data
```