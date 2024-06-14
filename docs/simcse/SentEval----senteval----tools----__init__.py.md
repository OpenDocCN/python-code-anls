# `.\SentEval\senteval\tools\__init__.py`

```py
# 定义一个名为 read_csv 的函数，接受一个参数 filename
def read_csv(filename):
    # 打开文件并赋值给变量 f，'r' 表示以只读模式打开文件
    with open(filename, 'r') as f:
        # 读取文件的所有行，并使用列表推导式将每一行转换为去除空白字符的字符串
        data = [line.strip() for line in f.readlines()]
    # 返回处理后的数据列表
    return data
```