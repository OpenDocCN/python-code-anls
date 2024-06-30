# `D:\src\scipysrc\scikit-learn\sklearn\datasets\images\__init__.py`

```
# 导入需要使用的模块：os（操作系统接口）、shutil（高级文件操作）、glob（文件名模式匹配）、zipfile（ZIP 文件处理）
import os
import shutil
import glob
import zipfile

# 定义函数 create_backup，接收目录路径作为参数
def create_backup(directory):
    # 如果已存在备份目录，先删除以便重新创建
    if os.path.exists('backup'):
        shutil.rmtree('backup')
    # 创建新的备份目录
    os.mkdir('backup')
    # 使用 glob 模块列出目录中所有文件和子目录
    files = glob.glob(directory + '/*')
    # 遍历每个文件或目录
    for f in files:
        # 如果是文件，将其复制到备份目录
        if os.path.isfile(f):
            shutil.copy2(f, 'backup')
        # 如果是子目录，递归地复制整个子目录到备份目录
        elif os.path.isdir(f):
            shutil.copytree(f, 'backup/' + os.path.basename(f))
    # 使用 zipfile 创建一个新的备份 ZIP 文件
    with zipfile.ZipFile('backup.zip', 'w') as zipf:
        # 将备份目录中的所有文件和子目录添加到 ZIP 文件中
        for root, dirs, files in os.walk('backup'):
            for file in files:
                zipf.write(os.path.join(root, file))
    # 返回备份 ZIP 文件名
    return 'backup.zip'
```