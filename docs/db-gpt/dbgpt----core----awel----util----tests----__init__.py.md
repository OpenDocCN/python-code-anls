# `.\DB-GPT-src\dbgpt\core\awel\util\tests\__init__.py`

```py
# 导入必要的模块：os 模块用于操作文件系统，shutil 模块用于高级文件操作，tempfile 模块用于生成临时文件和目录
import os
import shutil
import tempfile

# 定义一个函数 create_temporary_dir，用于创建一个临时目录并返回其路径
def create_temporary_dir():
    # 使用 tempfile 模块创建一个临时目录，函数返回临时目录的路径
    temp_dir = tempfile.mkdtemp()
    return temp_dir

# 在当前目录下创建一个名为 'temp' 的临时目录
temp_dir = create_temporary_dir()

# 打印创建的临时目录的路径
print(f"Created temporary directory: {temp_dir}")

# 使用 shutil 模块的 rmtree 函数删除已创建的临时目录及其内容
shutil.rmtree(temp_dir)

# 打印删除后的提示信息
print(f"Deleted temporary directory: {temp_dir}")
```