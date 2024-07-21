# `.\pytorch\benchmarks\dynamo\microbenchmarks\inductor_cpu_atomic.py`

```
import itertools  # 导入 itertools 模块，用于生成迭代器的工具函数

from benchmark_helper import time_with_torch_timer  # 从 benchmark_helper 模块中导入 time_with_torch_timer 函数

import torch  # 导入 PyTorch 库
import torch._dynamo  # 导入 PyTorch 的 _dynamo 模块，用于动态优化

@torch._dynamo.optimize("inductor", nopython=True)  # 使用 PyTorch 的动态优化装饰器，优化名为 "inductor" 的函数，禁用 Python 解释器
def inductor_scatter_add(dst, src, index):
    return torch.scatter_add(dst, 1, index, src)  # 使用 PyTorch 的 scatter_add 函数进行张量操作

def torch_scatter_add(dst, src, index):
    return torch.scatter_add(dst, 1, index, src)  # 使用 PyTorch 的 scatter_add 函数进行张量操作

def test_total_time(shapes, types):
    print(
        "shape; type; torch scatter_add; inductor scatter_add; torch scatter_add (worst case); inductor scatter_add (worst case)"
    )
    for shape, dtype in itertools.product(shapes, types):  # 使用 itertools.product 生成 shapes 和 types 的笛卡尔积迭代
        print(shape, dtype, sep="; ", end="; ")

        torch.manual_seed(1)  # 设置 PyTorch 随机种子为 1
        if dtype.is_floating_point:  # 检查数据类型是否为浮点数
            src = torch.randn(shape, device="cpu", dtype=dtype)  # 生成服从标准正态分布的张量 src
            dst = torch.randn(shape, device="cpu", dtype=dtype)  # 生成服从标准正态分布的张量 dst
        else:
            src = torch.randint(0, shape[1], shape, device="cpu", dtype=dtype)  # 生成随机整数张量 src
            dst = torch.randint(0, shape[1], shape, device="cpu", dtype=dtype)  # 生成随机整数张量 dst
        index = torch.randint(0, shape[1], shape, device="cpu", dtype=torch.int64)  # 生成随机整数索引张量 index
        worst_index = torch.tensor([[0] * shape[1]], device="cpu", dtype=torch.int64)  # 生成最差情况下的索引张量 worst_index

        torch_result = torch_scatter_add(dst, src, index)  # 调用自定义的 torch_scatter_add 函数进行张量操作
        inductor_result = inductor_scatter_add(dst, src, index)  # 调用自定义的 inductor_scatter_add 函数进行张量操作
        torch.testing.assert_close(torch_result, inductor_result)  # 使用 PyTorch 的测试函数检查两种方法得到的结果是否相似

        torch_ms = (  # 计算执行时间并转换为毫秒
            time_with_torch_timer(torch_scatter_add, (dst, src, index)).mean * 1000
        )
        inductor_ms = (  # 计算执行时间并转换为毫秒
            time_with_torch_timer(inductor_scatter_add, (dst, src, index)).mean * 1000
        )
        torch_worst_ms = (  # 计算最差情况下的执行时间并转换为毫秒
            time_with_torch_timer(torch_scatter_add, (dst, src, worst_index)).mean * 1000
        )
        inductor_worst_ms = (  # 计算最差情况下的执行时间并转换为毫秒
            time_with_torch_timer(inductor_scatter_add, (dst, src, worst_index)).mean * 1000
        )

        print(torch_ms, inductor_ms, torch_worst_ms, inductor_worst_ms, sep="; ")  # 打印计时结果

        torch._dynamo.reset()  # 重置 PyTorch 的动态优化设置

if __name__ == "__main__":
    shapes = [  # 定义不同形状的张量
        ([1, 4096]),  # 形状为 [1, 4096] 的张量
        ([1, 65536]),  # 形状为 [1, 65536] 的张量
    ]
    types = [  # 定义不同数据类型的张量
        torch.float32,  # 浮点数类型
        torch.int32,  # 整数类型
    ]
    print("test total time")
    test_total_time(shapes, types)  # 执行测试函数，打印测试结果
```