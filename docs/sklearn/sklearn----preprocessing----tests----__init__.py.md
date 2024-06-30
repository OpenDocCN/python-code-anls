# `D:\src\scipysrc\scikit-learn\sklearn\preprocessing\tests\__init__.py`

```
# 定义一个名为 process_data 的函数，接收一个名为 data 的参数
def process_data(data):
    # 对参数 data 进行拷贝，以避免直接修改原始数据
    data_copy = data.copy()
    # 遍历拷贝后的数据副本的每一个键值对
    for key, value in data_copy.items():
        # 如果值是偶数，则将其替换为原始值的两倍
        if value % 2 == 0:
            data_copy[key] = value * 2
        # 如果值是奇数，则将其替换为原始值的三倍
        else:
            data_copy[key] = value * 3
    # 返回经过处理后的数据副本
    return data_copy
```