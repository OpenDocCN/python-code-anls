# `.\numpy\tools\commitstats.py`

```
# 使用命令行执行 SVN log 命令，获取最近 2300 条提交日志，并将输出保存到 output.txt 文件中
command = 'svn log -l 2300 > output.txt'

# 导入正则表达式模块
import re
# 导入 NumPy 库，用于处理数组和数据
import numpy as np
# 导入操作系统接口模块
import os

# 定义正则表达式，匹配 SVN 日志中的提交者名字
names = re.compile(r'r\d+\s\|\s(.*)\s\|\s200')

# 定义函数，从指定文件中获取提交者名字出现的次数，并以列表形式返回
def get_count(filename, repo):
    # 读取指定文件的内容
    mystr = open(filename).read()
    # 使用正则表达式查找符合条件的提交者名字
    result = names.findall(mystr)
    # 使用 NumPy 的 unique 函数获取唯一的提交者名字
    u = np.unique(result)
    # 组装每个提交者名字、出现次数和仓库名为元组，并放入列表中
    count = [(x, result.count(x), repo) for x in u]
    return count

# 切换当前工作目录至上级目录
os.chdir('..')
# 使用操作系统接口执行 SVN log 命令，输出结果保存到 output.txt 文件
os.system(command)

# 获取 NumPy 仓库中提交者名字的出现次数统计
count = get_count('output.txt', 'NumPy')

# 切换当前工作目录至 scipy 目录
os.chdir('../scipy')
# 再次执行 SVN log 命令，更新 output.txt 文件
os.system(command)

# 将 SciPy 仓库中提交者名字的出现次数统计加入到 count 列表中
count.extend(get_count('output.txt', 'SciPy'))

# 切换当前工作目录至 scikits 目录
os.chdir('../scikits')
# 再次执行 SVN log 命令，更新 output.txt 文件
os.system(command)

# 将 SciKits 仓库中提交者名字的出现次数统计加入到 count 列表中
count.extend(get_count('output.txt', 'SciKits'))

# 对 count 列表进行排序，按提交者名字的字母顺序排序
count.sort()

# 输出标题，表示 SciPy 和 NumPy 仓库的统计结果
print("** SciPy and NumPy **")
print("=====================")

# 遍历 count 列表，输出每个元组的内容，包括提交者名字、出现次数和仓库名
for val in count:
    print(val)
```