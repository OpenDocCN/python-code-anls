# `D:\src\scipysrc\scipy\scipy\integrate\_ivp\tests\__init__.py`

```
# 导入CSV模块，用于处理CSV文件
import csv

# 定义函数read_csv，接收文件名参数fname
def read_csv(fname):
    # 打开CSV文件，模式为只读
    with open(fname, 'r', encoding='utf-8') as f:
        # 使用csv.reader读取文件内容，生成可迭代的行对象
        reader = csv.reader(f)
        # 将每一行数据转换为列表，并存储在rows列表中
        rows = [row for row in reader]
        # 返回包含所有行数据的列表
        return rows
```