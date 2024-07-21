# `.\pytorch\torch\utils\jit\__init__.py`

```py
# 导入所需的模块：os（操作系统接口）、shutil（高级文件操作）、tempfile（生成临时文件和目录）、glob（根据模式匹配文件路径名）
import os
import shutil
import tempfile
import glob

# 使用 tempfile 模块生成一个临时目录，并返回该目录的路径
tmpdir = tempfile.mkdtemp()

# 获取当前工作目录，并保存在变量 cwd 中
cwd = os.getcwd()

# 将当前工作目录更改为临时目录 tmpdir
os.chdir(tmpdir)

# 将当前目录（tmpdir）下所有以 .txt 结尾的文件路径列表存储在变量 txtfiles 中
txtfiles = glob.glob('*.txt')

# 遍历 txtfiles 列表中的每个文件路径
for txtfile in txtfiles:
    # 将每个 txtfile 文件复制到当前工作目录（tmpdir）下
    shutil.copy(txtfile, cwd)

# 将当前工作目录更改回原来的工作目录（cwd）
os.chdir(cwd)

# 删除临时目录 tmpdir 及其内容
shutil.rmtree(tmpdir)
```