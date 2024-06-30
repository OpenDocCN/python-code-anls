# `D:\src\scipysrc\scipy\scipy\optimize\_highs\cython\src\HConfig.h`

```
# 导入必要的模块：io 用于处理输入输出，zipfile 用于处理 ZIP 文件
import io
import zipfile

# 定义一个函数 read_zip，接收一个文件名参数 fname
def read_zip(fname):
    # 使用二进制模式打开文件 fname，并读取其内容，然后封装成字节流对象
    bio = io.BytesIO(open(fname, 'rb').read())
    # 使用 BytesIO 对象创建一个 ZipFile 对象，以便后续操作
    zip = zipfile.ZipFile(bio, 'r')
    # 使用字典推导式创建一个字典 fdict，键是 ZIP 文件中的文件名，值是对应文件名的内容数据
    fdict = {n: zip.read(n) for n in zip.namelist()}
    # 关闭 ZipFile 对象，释放资源
    zip.close()
    # 返回创建的字典 fdict，其中包含了 ZIP 文件中每个文件名及其内容的映射关系
    return fdict
```