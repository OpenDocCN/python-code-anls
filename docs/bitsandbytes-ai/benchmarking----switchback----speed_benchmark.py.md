# `bitsandbytes\benchmarking\switchback\speed_benchmark.py`

```py
# 导入所需的库
import json
import time

import torch

# 导入自定义的函数
from bitsandbytes.triton.int8_matmul_mixed_dequantize import (
    int8_matmul_mixed_dequantize,
)
from bitsandbytes.triton.int8_matmul_rowwise_dequantize import (
    int8_matmul_rowwise_dequantize,
)
from bitsandbytes.triton.quantize_columnwise_and_transpose import (
    quantize_columnwise_and_transpose,
)
from bitsandbytes.triton.quantize_global import (
    quantize_global,
    quantize_global_transpose,
)
from bitsandbytes.triton.quantize_rowwise import quantize_rowwise

# 已知问题：当 embeddim 过大时，需要优化 "w_quantize_colwise_transpose"。

# 定义函数用于计算执行函数 fn 的时间
def get_time(k, fn, info_dict):

    # 预热 GPU，执行 fn 函数
    for _ in range(repeat // 2):
       fn()

    # 同步 GPU，开始计时
    torch.cuda.synchronize()
    start = time.time()
    # 执行 fn 函数，重复 repeat 次
    for _ in range(repeat):
       fn()

    # 同步 GPU，结束计时
    torch.cuda.synchronize()
    end = time.time()
    # 计算执行 fn 函数的平均时间
    ms = (end - start) / repeat * 1000
    # 打印执行 fn 函数的时间
    print(f"time {k}: {ms:.3f} ms")
    # 将执行 fn 函数的时间记录到 info_dict 中
    info_dict[k] = ms

# 主函数入口
if __name__ == '__main__':
    # 设置随机种子
    torch.manual_seed(0)
    # 初始化 wm 变量为 4
    wm = 4
```