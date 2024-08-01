# `.\DB-GPT-src\dbgpt\app\static\_next\static\chunks\7119-64fb8f0364433c24.js`

```py
# 导入所需的模块：os（操作系统接口）、shutil（文件操作工具）和 glob（路径模式匹配工具）
import os
import shutil
import glob

# 源目录路径，指定为当前目录下的 'source/' 文件夹
source_dir = './source/'

# 目标目录路径，指定为当前目录下的 'destination/' 文件夹
target_dir = './destination/'

# 如果目标目录不存在，则创建它
if not os.path.exists(target_dir):
    # 使用 os.makedirs 递归创建目标目录
    os.makedirs(target_dir)

# 查找源目录下所有扩展名为 '.txt' 的文件
for file_name in glob.glob(os.path.join(source_dir, '*.txt')):
    # 构建每个文件的目标路径，将其放置到目标目录下
    target_file = os.path.join(target_dir, os.path.basename(file_name))
    # 使用 shutil.copy2 复制文件到目标路径
    shutil.copy2(file_name, target_file)
```