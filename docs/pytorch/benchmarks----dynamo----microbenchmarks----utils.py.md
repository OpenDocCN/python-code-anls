# `.\pytorch\benchmarks\dynamo\microbenchmarks\utils.py`

```
# 导入数学库
import math

# 导入 PyTorch 库
import torch

# 创建一个函数，返回一个在给定范围内的、经过舍入处理的等间距数组
def rounded_linspace(low, high, steps, div):
    # 使用 PyTorch 的 linspace 函数生成在指定范围内的等间距张量
    ret = torch.linspace(low, high, steps)
    # 将张量转换为整数类型，然后进行舍入和分割处理
    ret = (ret.int() + div - 1) // div * div
    # 使用 PyTorch 提供的唯一化函数去除重复值
    ret = torch.unique(ret)
    # 将结果转换为整数列表并返回
    return list(map(int, ret))


# 创建一个函数，返回一个指数空间中的数值列表
def powspace(start, stop, pow, step):
    # 对起始值和停止值取以给定底数为底的对数
    start = math.log(start, pow)
    stop = math.log(stop, pow)
    # 计算需要的步数并转换为整数
    steps = int((stop - start + 1) // step)
    # 使用 PyTorch 的指数函数生成指数空间中的数值张量
    ret = torch.pow(pow, torch.linspace(start, stop, steps))
    # 使用 PyTorch 提供的唯一化函数去除重复值
    ret = torch.unique(ret)
    # 将结果转换为整数列表并返回
    return list(map(int, ret))
```