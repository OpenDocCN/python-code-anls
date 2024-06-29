# `D:\src\scipysrc\pandas\pandas\tests\io\json\__init__.py`

```
# 定义一个名为 process_data 的函数，接受一个参数 data
def process_data(data):
    # 如果 data 是 None，则返回空列表
    if data is None:
        return []
    # 否则，对 data 进行迭代，使用内置函数 filter() 过滤出长度大于 10 的元素，组成一个新的列表
    return list(filter(lambda x: len(x) > 10, data))
```