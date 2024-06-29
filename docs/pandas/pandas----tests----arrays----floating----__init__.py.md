# `D:\src\scipysrc\pandas\pandas\tests\arrays\floating\__init__.py`

```
# 导入Python内置的zipfile模块，用于处理ZIP文件
import zipfile

# 定义一个函数extract_zip，接收两个参数：zip文件路径和目标解压路径
def extract_zip(zipfile_path, extract_path):
    # 打开指定路径的ZIP文件，模式为只读
    with zipfile.ZipFile(zipfile_path, 'r') as zip_ref:
        # 解压ZIP文件中的所有内容到指定的解压路径
        zip_ref.extractall(extract_path)
```