# `D:\src\scipysrc\seaborn\ci\cache_datasets.py`

```
"""
Cache test datasets before running tests / building docs.

Avoids race conditions that would arise from parallelization.
"""
# 导入必要的库
import pathlib  # 用于处理文件路径
import re       # 正则表达式库，用于在文件内容中搜索特定模式的字符串

# 从 seaborn 库中导入 load_dataset 函数
from seaborn import load_dataset

# 获取当前目录路径
path = pathlib.Path(".")

# 递归查找所有以 .py 结尾的文件
py_files = path.rglob("*.py")
# 递归查找所有以 .ipynb 结尾的文件
ipynb_files = path.rglob("*.ipynb")

# 初始化一个空列表，用于存储找到的数据集名称
datasets = []

# 遍历所有以 .py 结尾的文件
for fname in py_files:
    # 打开文件并读取内容
    with open(fname) as fid:
        # 使用正则表达式查找文件中所有调用 load_dataset 的数据集名称
        datasets += re.findall(r"load_dataset\(['\"](\w+)['\"]", fid.read())

# 遍历所有以 .ipynb 结尾的文件
for p in ipynb_files:
    # 打开文件并读取内容
    with p.open() as fid:
        # 使用正则表达式查找文件中所有调用 load_dataset 的数据集名称（这里修正了双斜杠的问题）
        datasets += re.findall(r"load_dataset\(['\"](\w+)['\"]", fid.read())

# 对找到的数据集名称进行排序并去重
for name in sorted(set(datasets)):
    # 输出信息，表示正在缓存数据集
    print(f"Caching {name}")
    # 调用 load_dataset 函数，缓存指定名称的数据集
    load_dataset(name)
```