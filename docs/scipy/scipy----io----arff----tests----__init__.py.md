# `D:\src\scipysrc\scipy\scipy\io\arff\tests\__init__.py`

```
# 定义一个名为 read_csv 的函数，接收一个参数 filename
def read_csv(filename):
    # 尝试打开指定文件名的文件，'r' 表示读取模式，使用 utf-8 编码
    with open(filename, 'r', encoding='utf-8') as f:
        # 读取整个文件内容，返回一个字符串
        data = f.read()
        # 将文件内容按行分割，形成一个字符串列表
        lines = data.split('\n')
        # 使用列表推导式，将每一行按逗号分隔的数据转换为列表，形成一个二维列表
        csv_data = [line.split(',') for line in lines if line.strip()]
        # 返回处理后的 CSV 数据，是一个二维列表
        return csv_data
```