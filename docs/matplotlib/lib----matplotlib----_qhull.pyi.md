# `D:\src\scipysrc\matplotlib\lib\matplotlib\_qhull.pyi`

```py
# 导入必要的模块 BytesIO 用于处理二进制数据，zipfile 用于处理 ZIP 文件
from io import BytesIO
import zipfile

# 定义函数 unzip_files，接受一个 ZIP 文件路径作为参数
def unzip_files(zip_file):
    # 打开 ZIP 文件为二进制读取模式，创建字节流对象
    bio = BytesIO(open(zip_file, 'rb').read())
    # 使用字节流创建 ZIP 文件对象
    zip = zipfile.ZipFile(bio, 'r')
    # 初始化一个空列表来存储解压后的文件名列表
    file_list = []
    # 遍历 ZIP 文件中的每个文件
    for file_name in zip.namelist():
        # 读取 ZIP 文件中的每个文件的内容并添加到文件名列表中
        file_list.append(zip.read(file_name))
    # 关闭 ZIP 文件对象
    zip.close()
    # 返回解压后的文件内容列表
    return file_list
```