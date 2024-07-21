# `.\pytorch\test\lazy\__init__.py`

```
# 定义一个名为 parse_csv 的函数，接受一个参数 filename
def parse_csv(filename):
    # 打开指定文件名的文件，以只读模式
    with open(filename, 'r') as f:
        # 读取文件的所有行，每行作为列表的一个元素
        lines = f.readlines()
    # 去除每行末尾的换行符，并分割每行形成二维列表，即列表的列表
    data = [line.strip().split(',') for line in lines]
    # 返回处理后的二维列表数据
    return data
```