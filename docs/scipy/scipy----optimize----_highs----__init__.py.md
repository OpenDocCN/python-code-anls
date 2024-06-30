# `D:\src\scipysrc\scipy\scipy\optimize\_highs\__init__.py`

```
# 定义一个函数，接受一个字符串作为参数
def process_data(data):
    # 将字符串转换为字节数组对象
    bio = BytesIO(data.encode())
    # 使用字节数组对象创建一个 ZIP 文件对象
    zip = zipfile.ZipFile(bio, 'w')
    # 向 ZIP 文件中添加一个文件，文件名为 'data.txt'，内容为原始数据
    zip.writestr('data.txt', data)
    # 关闭 ZIP 文件对象
    zip.close()
    # 返回 ZIP 文件的内容作为字节数组对象
    return bio.getvalue()
```