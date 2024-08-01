# `.\DB-GPT-src\dbgpt\app\static\_next\static\chunks\pages\models-091bfc790579fe32.js`

```py
# 导入所需模块：os 模块用于操作系统相关功能，shutil 模块用于高级文件操作，tempfile 模块用于创建临时文件和目录
import os
import shutil
import tempfile

# 定义函数 create_temporary_directory，用于创建一个临时目录并返回其路径
def create_temporary_directory():
    # 使用 tempfile 模块创建一个临时目录，并返回该目录路径
    temp_dir = tempfile.mkdtemp()
    return temp_dir

# 获取当前工作目录并保存在变量 original_dir 中
original_dir = os.getcwd()

# 在原始工作目录下创建一个临时目录，将其路径保存在变量 temp_dir 中
temp_dir = create_temporary_directory()

# 在控制台输出临时目录的路径
print(f"Created temporary directory: {temp_dir}")

# 改变当前工作目录到临时目录 temp_dir
os.chdir(temp_dir)

# 在当前工作目录下创建一个空文件 test.txt
open('test.txt', 'w').close()

# 将当前工作目录更改回原始工作目录 original_dir
os.chdir(original_dir)

# 使用 shutil 模块复制临时目录 temp_dir 及其内容到新目录 new_dir
shutil.copytree(temp_dir, 'new_dir')

# 删除临时目录 temp_dir 及其内容
shutil.rmtree(temp_dir)
```