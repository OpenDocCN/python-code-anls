# `.\DB-GPT-src\dbgpt\app\static\_next\static\chunks\971df74e-084169c6c09eb1fe.js`

```py
# 导入必要的模块：os 模块用于操作文件和目录，shutil 模块用于高级文件操作
import os
import shutil

# 指定要复制的源文件夹路径
source = '/path/to/source/folder'

# 指定目标文件夹路径，如果不存在则创建
destination = '/path/to/destination/folder'
if not os.path.exists(destination):
    os.makedirs(destination)

# 使用 shutil 模块的 copytree 函数进行文件夹的递归复制
shutil.copytree(source, destination)

# 打印复制完成的消息
print(f'Folder copied from {source} to {destination}')
```