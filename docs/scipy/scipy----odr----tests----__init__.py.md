# `D:\src\scipysrc\scipy\scipy\odr\tests\__init__.py`

```
# 导入所需的模块：io 用于处理文件流，zipfile 用于操作 ZIP 文件
import io
import zipfile

# 定义一个函数 extract_zip，接收两个参数：zip_filename 为 ZIP 文件名，extract_to 为解压目标路径
def extract_zip(zip_filename, extract_to):
    # 打开 ZIP 文件为二进制文件对象
    with open(zip_filename, 'rb') as f:
        # 创建一个 BytesIO 对象，将文件对象的内容读取到内存中
        bio = io.BytesIO(f.read())
    
    # 创建一个 ZipFile 对象，使用读取的内存中的二进制数据
    with zipfile.ZipFile(bio, 'r') as zip:
        # 遍历 ZIP 文件中的所有文件
        for file_info in zip.infolist():
            # 构建文件的完整路径
            file_path = extract_to / file_info.filename
            # 如果是文件夹，创建对应的目录
            if file_info.is_dir():
                file_path.mkdir(parents=True, exist_ok=True)
            else:
                # 是文件，则解压到指定路径
                with open(file_path, 'wb') as f_out:
                    f_out.write(zip.read(file_info.filename))
```