# `.\pytorch\benchmarks\operator_benchmark\common\repeat_benchmark.py`

```py
import time
# 导入时间模块

import numpy as np
# 导入numpy模块

import torch
# 导入torch模块

"""Microbenchmarks for Tensor repeat operator. Supports PyTorch."""
# 微基准测试用于张量重复操作符。支持PyTorch。

input_shapes = (
    (4, 4, 1),
    (16, 1, 32),
    (64, 64, 1, 1),
    (8, 256, 128),
    (1, 64, 128, 32),
    (512, 512),
)
# 定义输入张量的形状

repeats = (
    (1, 1, 1, 64),
    (1, 4, 1, 2),
    (1, 2, 2, 15),
    (1, 1, 3, 2),
    (128, 1, 8, 1),
    (1, 1, 2, 16),
)
# 定义重复的次数

NUM_WARMUP_ITERS = 5
NUM_BENCHMARK_ITERS = 10
DTYPE_TO_BYTES = {"float": 4}
# 定义常量

def generate_data_for_repeat():
    input_tensors = [torch.randn(*input_shape) for input_shape in input_shapes]
    total_num_elements = 0
    for input_tensor, repeat in zip(input_tensors, repeats):
        total_num_elements += input_tensor.numel()
        total_num_elements += input_tensor.numel() * np.prod(repeat)
    return input_tensors, (total_num_elements * DTYPE_TO_BYTES["float"])
# 生成用于重复操作的数据

input_tensors, total_bytes = generate_data_for_repeat()
BYTES_TO_MB = 1.0 / 1000.0 / 1000.0
# 计算字节到MB的转换比例

def pt_repeat(input_tensor, repeat):
    return input_tensor.repeat(repeat)
# 定义PyTorch的重复操作函数

def pt_repeat_n_times(niters):
    for _ in range(niters):
        for input_tensor, repeat in zip(input_tensors, repeats):
            pt_repeat(input_tensor, repeat)
# 定义多次执行PyTorch的重复操作函数

if __name__ == "__main__":
    # Warm up runs.
    pt_repeat_n_times(NUM_WARMUP_ITERS)
    # 预热运行

    s = time.time()
    pt_repeat_n_times(NUM_BENCHMARK_ITERS)
    total_time_s = time.time() - s
    total_time_per_iter_s = total_time_s / NUM_BENCHMARK_ITERS
    achieved_bandwidth = (total_bytes * BYTES_TO_MB) / total_time_per_iter_s
    print(f"Time:{total_time_per_iter_s} Achieved Bandwidth:{achieved_bandwidth} MB/s")
    # 打印结果
```