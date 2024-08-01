# `.\DB-GPT-src\dbgpt\util\network\__init__.py`

```py
# 导入所需的模块
import os
import zipfile

# 定义函数，接收一个目录路径参数
def zipdir(path):
    # 确保路径存在并且是一个目录
    assert os.path.isdir(path), '路径必须是一个存在的目录'
    # 获取目录的绝对路径
    abs_path = os.path.abspath(path)
    # 创建一个与目录同名的 ZIP 文件
    zipf = zipfile.ZipFile(abs_path + '.zip', 'w', zipfile.ZIP_DEFLATED)
    
    # 遍历目录中的每个文件和子目录
    for root, dirs, files in os.walk(path):
        # 将每个文件和子目录添加到 ZIP 文件中
        for file in files:
            absname = os.path.abspath(os.path.join(root, file))
            arcname = absname[len(abs_path) + 1:]
            zipf.write(absname, arcname)
    
    # 关闭 ZIP 文件
    zipf.close()

# 调用函数，将指定目录压缩为同名 ZIP 文件
zipdir('/path/to/your/directory')
```