# `D:\src\scipysrc\sympy\sympy\algebras\tests\__init__.py`

```
# 导入所需的模块：os 模块用于与操作系统交互，shutil 模块提供高级文件操作功能
import os
import shutil

# 源文件夹路径
source_dir = '/path/to/source/directory'
# 目标文件夹路径
target_dir = '/path/to/target/directory'

# 如果目标文件夹不存在，则创建它
if not os.path.exists(target_dir):
    # 使用 os.makedirs 递归地创建目标文件夹及其父文件夹
    os.makedirs(target_dir)

# 遍历源文件夹中的文件和子文件夹
for item in os.listdir(source_dir):
    # 构建完整的源文件或子文件夹路径
    s = os.path.join(source_dir, item)
    # 构建完整的目标文件或子文件夹路径
    d = os.path.join(target_dir, item)
    # 如果当前项目是文件且不是文件夹，则复制文件
    if os.path.isfile(s):
        # 使用 shutil.copy2 复制文件，保留文件元数据（如权限和时间戳）
        shutil.copy2(s, d)
    # 如果当前项目是文件夹，则递归地复制文件夹及其内容
    elif os.path.isdir(s):
        # 使用 shutil.copytree 递归地复制文件夹及其内容到目标文件夹
        shutil.copytree(s, d)

# 打印完成消息
print(f"复制完成：从 '{source_dir}' 到 '{target_dir}'")
```