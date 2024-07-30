# `.\comic-translate\app\ui\__init__.py`

```py
# 定义一个名为 parse_csv 的函数，接受一个文件名参数
def parse_csv(filename):
    # 打开指定文件名的文件，以只读方式
    with open(filename, 'r') as f:
        # 读取文件的所有行，并将每一行用逗号分割成列表项
        data = [line.strip().split(',') for line in f]
    # 返回处理后的数据列表
    return data
```