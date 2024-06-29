# `D:\src\scipysrc\pandas\pandas\_libs\window\__init__.py`

```
# 导入所需的模块：os（操作系统接口）、shutil（高级文件操作）、glob（文件名模式匹配）
import os
import shutil
import glob

# 定义函数 compress_files，接收一个目标文件夹路径作为参数
def compress_files(folder_path):
    # 获取目标文件夹下所有以'.txt'为后缀的文件列表
    files = glob.glob(os.path.join(folder_path, '*.txt'))
    # 创建名为 'archive.zip' 的 ZIP 文件，准备用来存档文件
    with zipfile.ZipFile('archive.zip', 'w') as zipf:
        # 遍历文件列表
        for file in files:
            # 将每个文件添加到 ZIP 文件中，使用文件名作为 ZIP 文件中的文件名
            zipf.write(file, os.path.basename(file))
    # 将 'archive.zip' 移动到当前工作目录下的 'archives' 文件夹中
    shutil.move('archive.zip', os.path.join(os.getcwd(), 'archives'))
```