# `D:\src\scipysrc\sympy\sympy\polys\agca\tests\__init__.py`

```
# 导入必要的模块：os（用于操作文件路径），shutil（用于高级文件操作）
import os
import shutil

# 源文件夹路径
src_dir = '/path/to/source/directory'

# 目标文件夹路径
dst_dir = '/path/to/destination/directory'

# 如果目标文件夹不存在，则创建它
if not os.path.exists(dst_dir):
    # 创建目标文件夹
    os.makedirs(dst_dir)

# 遍历源文件夹中的所有文件和子文件夹
for item in os.listdir(src_dir):
    # 构建源文件或子文件夹的完整路径
    s = os.path.join(src_dir, item)
    # 构建目标文件或子文件夹的完整路径
    d = os.path.join(dst_dir, item)
    # 如果是文件，直接复制到目标文件夹中
    if os.path.isfile(s):
        shutil.copy2(s, d)
    # 如果是文件夹，递归地复制整个文件夹及其内容到目标文件夹中
    elif os.path.isdir(s):
        shutil.copytree(s, d, symlinks=True)

# 复制完成后输出消息
print(f'Files from {src_dir} have been successfully copied to {dst_dir}.')
```