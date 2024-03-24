# `.\lucidrains\se3-transformer-pytorch\tests\test_spherical_harmonics.py`

```py
# 导入必要的库
import time
import torch
import numpy as np

# 从 lie_learn 库中导入 spherical_harmonics 函数
from lie_learn.representations.SO3.spherical_harmonics import sh

# 从 se3_transformer_pytorch 库中导入 get_spherical_harmonics_element 和 benchmark 函数
from se3_transformer_pytorch.spherical_harmonics import get_spherical_harmonics_element
from se3_transformer_pytorch.utils import benchmark

# 定义测试 spherical_harmonics 函数
def test_spherical_harmonics():
    # 设置数据类型为 torch.float64
    dtype = torch.float64

    # 生成随机的 theta 和 phi 数据
    theta = 0.1 * torch.randn(32, 1024, 10, dtype=dtype)
    phi = 0.1 * torch.randn(32, 1024, 10, dtype=dtype)

    # 初始化变量
    s0 = s1 = 0
    max_error = -1.

    # 循环遍历 l 和 m 的取值范围
    for l in range(8):
        for m in range(-l, l + 1):
            # 记录开始时间
            start = time.time()

            # 使用 benchmark 函数计算 get_spherical_harmonics_element 函数的运行时间和输出
            diff, y = benchmark(get_spherical_harmonics_element)(l, m, theta, phi)
            # 将 y 转换为 torch.float32 类型
            y = y.type(torch.float32)
            s0 += diff

            # 使用 benchmark 函数计算 sh 函数的运行时间和输出
            diff, z = benchmark(sh)(l, m, theta, phi)
            s1 += diff

            # 计算误差
            error = np.mean(np.abs((y.cpu().numpy() - z) / z))
            max_error = max(max_error, error)
            print(f"l: {l}, m: {m} ", error)

    # 计算时间差异比率
    time_diff_ratio = s0 / s1

    # 断言最大误差小于 1e-4
    assert max_error < 1e-4, 'maximum error must be less than 1e-3'
    # 断言时间差异比率小于 1
    assert time_diff_ratio < 1., 'spherical harmonics must be faster than the one offered by lie_learn'

    # 打印最大误差和时间差异比率
    print(f"Max error: {max_error}")
    print(f"Time diff: {time_diff_ratio}")
```