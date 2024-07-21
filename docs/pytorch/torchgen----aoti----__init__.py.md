# `.\pytorch\torchgen\aoti\__init__.py`

```py
# 导入所需的模块：os 模块用于文件路径操作，subprocess 模块用于执行外部命令
import os
import subprocess

# 定义函数：根据文件名和命令执行外部命令，并返回执行结果
def run_command(filename, command):
    # 构建完整的文件路径
    filepath = os.path.join('/path/to/dir', filename)
    # 使用 subprocess 模块执行给定的外部命令，并捕获输出
    result = subprocess.run(command, cwd='/path/to/dir', stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # 返回执行结果的标准输出
    return result.stdout.decode('utf-8')
```