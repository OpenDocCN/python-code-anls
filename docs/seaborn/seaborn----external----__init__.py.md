# `D:\src\scipysrc\seaborn\seaborn\external\__init__.py`

```
# 导入必要的模块：io用于处理字节流，zipfile用于解析ZIP文件
import io
import zipfile

# 定义一个函数extract_zip，接收一个ZIP文件路径作为参数，返回解压后的文件内容列表
def extract_zip(zip_file):
    # 打开ZIP文件为二进制模式，并将其内容读取到内存中的字节流对象
    bio = io.BytesIO(open(zip_file, 'rb').read())
    # 使用字节流创建一个ZIP文件对象
    zip = zipfile.ZipFile(bio, 'r')
    # 初始化一个空列表，用于存储解压后的文件内容
    file_contents = []
    # 遍历ZIP文件中的所有文件名
    for file_name in zip.namelist():
        # 读取ZIP文件中指定文件名的内容，并将其添加到文件内容列表中
        content = zip.read(file_name)
        file_contents.append(content)
    # 关闭ZIP文件对象
    zip.close()
    # 返回解压后的文件内容列表
    return file_contents
```