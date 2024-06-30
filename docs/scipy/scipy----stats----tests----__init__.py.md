# `D:\src\scipysrc\scipy\scipy\stats\tests\__init__.py`

```
# 定义一个名为 read_csv 的函数，接受一个文件名参数 fname
def read_csv(fname):
    # 打开文件 fname 作为文件对象 f，以只读模式打开
    with open(fname, 'r') as f:
        # 读取文件 f 中的所有行，并将其作为列表返回
        data = f.readlines()
        # 使用列表推导式，去除每行末尾的换行符
        data = [line.strip() for line in data]
        # 返回处理后的列表 data，其中包含了文件的所有行（去除了换行符）
        return data
```