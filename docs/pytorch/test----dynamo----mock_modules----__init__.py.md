# `.\pytorch\test\dynamo\mock_modules\__init__.py`

```
# 导入必要的模块：os（操作系统接口）、shutil（高级文件操作工具）、glob（Unix风格路径名模式扩展）。
import os
import shutil
import glob

# 定义一个函数：clean_dir，接收一个参数dir_path，用于清空指定目录下的所有文件和子目录。
def clean_dir(dir_path):
    # 使用 os.path.join 将 dir_path 和 '*'（代表所有文件和子目录）拼接成完整的目录路径。
    full_path = os.path.join(dir_path, '*')
    # 使用 glob.glob 函数获取目录中的所有文件和子目录的路径列表。
    all_files = glob.glob(full_path)
    # 遍历所有文件和子目录的路径列表。
    for f in all_files:
        # 如果是文件，使用 os.remove 函数删除文件。
        if os.path.isfile(f):
            os.remove(f)
        # 如果是目录，使用 shutil.rmtree 函数递归删除目录及其内容。
        elif os.path.isdir(f):
            shutil.rmtree(f)

# 定义一个变量：directory_path，存储要清空的目录路径。
directory_path = '/path/to/your/directory'

# 调用 clean_dir 函数，传入 directory_path 参数，清空指定目录下的所有文件和子目录。
clean_dir(directory_path)
```