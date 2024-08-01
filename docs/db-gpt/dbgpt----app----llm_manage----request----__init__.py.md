# `.\DB-GPT-src\dbgpt\app\llm_manage\request\__init__.py`

```py
# 导入所需的模块：io 用于处理文件流，zipfile 用于处理 ZIP 文件
import io
import zipfile

# 定义函数 unzip_files，接受两个参数：zip_filename（ZIP 文件名）和extract_to（解压到的目录）
def unzip_files(zip_filename, extract_to):
    # 打开指定的 ZIP 文件，'rb' 表示以二进制读取模式打开
    with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
        # 解压缩 ZIP 文件中的所有文件到指定目录 extract_to
        zip_ref.extractall(extract_to)
```