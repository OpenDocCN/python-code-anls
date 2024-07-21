# `.\pytorch\benchmarks\distributed\rpc\parameter_server\trainer\preprocess_data.py`

```
# 定义一个预处理函数，将数据从 CPU 移动到 GPU，供 DummyData 类使用
def preprocess_dummy_data(rank, data):
    r"""
    A function that moves the data from CPU to GPU
    for DummyData class.
    Args:
        rank (int): worker rank
        data (list): training examples
    """
    # 遍历数据列表
    for i in range(len(data)):
        # 将每个样本的第一个元素（通常是输入数据）移动到指定 GPU（由 rank 指定）
        data[i][0] = data[i][0].cuda(rank)
        # 将每个样本的第二个元素（通常是标签数据）移动到指定 GPU（由 rank 指定）
        data[i][1] = data[i][1].cuda(rank)
    # 返回处理后的数据列表
    return data
```