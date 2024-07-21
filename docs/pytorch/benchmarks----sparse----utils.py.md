# `.\pytorch\benchmarks\sparse\utils.py`

```
# 导入 functools、operator、random 和 time 模块
import functools
import operator
import random
import time

# 导入 numpy 和 torch 库
import numpy as np
import torch


# 定义一个 CPU 上的 torch.cuda.Event 的替代类
class Event:
    def __init__(self, enable_timing):
        pass

    def record(self):
        # 记录当前时间作为事件记录的时间
        self.time = time.perf_counter()

    def elapsed_time(self, end_event):
        # 断言结束事件是 Event 类的实例
        assert isinstance(end_event, Event)
        # 返回当前事件记录与结束事件记录的时间差
        return end_event.time - self.time


# 生成稀疏的 CSR 格式的张量
def gen_sparse_csr(shape, nnz):
    # 填充值设为 0
    fill_value = 0
    # 计算张量中总共的元素个数
    total_values = functools.reduce(operator.mul, shape, 1)
    # 生成一个包含随机值的稠密数组
    dense = np.random.randn(total_values)
    # 随机选择要填充为 0 的位置
    fills = random.sample(list(range(total_values)), total_values - nnz)

    for f in fills:
        dense[f] = fill_value

    # 将稠密数组转换为稀疏的 CSR 格式的张量并返回
    dense = torch.from_numpy(dense.reshape(shape))
    return dense.to_sparse_csr()


# 生成稀疏的 COO 格式的张量
def gen_sparse_coo(shape, nnz):
    # 生成一个包含随机值的稠密数组
    dense = np.random.randn(*shape)
    values = []
    indices = [[], []]
    for n in range(nnz):
        # 随机选择非零元素的位置
        row = random.randint(0, shape[0] - 1)
        col = random.randint(0, shape[1] - 1)
        indices[0].append(row)
        indices[1].append(col)
        values.append(dense[row, col])

    # 使用 COO 格式的索引和值生成稀疏的 COO 格式的张量并返回
    return torch.sparse_coo_tensor(indices, values, size=shape)


# 生成同时包含 COO 和 CSR 格式的稀疏张量
def gen_sparse_coo_and_csr(shape, nnz):
    # 计算张量中总共的元素个数
    total_values = functools.reduce(operator.mul, shape, 1)
    # 生成一个包含随机值的稠密数组
    dense = np.random.randn(total_values)
    # 随机选择要填充为 0 的位置
    fills = random.sample(list(range(total_values)), total_values - nnz)

    for f in fills:
        dense[f] = 0

    # 将稠密数组分别转换为稀疏的 COO 和 CSR 格式的张量并返回
    dense = torch.from_numpy(dense.reshape(shape))
    return dense.to_sparse(), dense.to_sparse_csr()
```