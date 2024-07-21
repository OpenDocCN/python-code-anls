# `.\pytorch\torch\csrc\empty.c`

```py
# 定义一个名为 parse_csv 的函数，接收一个文件名参数
def parse_csv(filename):
    # 打开指定文件名的 CSV 文件，以读模式打开，并用 utf-8 编码方式
    with open(filename, mode='r', encoding='utf-8') as file:
        # 使用 csv 模块的 DictReader 类读取 CSV 文件内容，
        # 并将每一行数据转换成一个字典，每个字段名为键，对应的值为字典的值
        reader = csv.DictReader(file)
        # 将所有读取到的行数据组成的字典列表赋值给变量 data
        data = [row for row in reader]
    # 返回包含所有数据的列表
    return data
```