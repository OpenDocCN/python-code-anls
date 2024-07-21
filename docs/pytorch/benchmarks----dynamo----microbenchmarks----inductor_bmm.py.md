# `.\pytorch\benchmarks\dynamo\microbenchmarks\inductor_bmm.py`

```py
from benchmark_helper import time_with_torch_timer  # 导入用于计时的辅助函数

import torch  # 导入PyTorch库

import torch._dynamo  # 导入PyTorch的内部模块
import torch._dynamo.config  # 导入PyTorch的内部配置模块
import torch._inductor.config as config  # 导入PyTorch的另一个配置模块


@torch._dynamo.optimize("inductor", nopython=True)
def inductor_aten_bmm(a, b):
    return torch.bmm(a, b)  # 执行PyTorch的批矩阵乘操作


@torch._dynamo.optimize("inductor", nopython=True)
def inductor_triton_bmm(a, b):
    return torch.bmm(a, b)  # 执行PyTorch的批矩阵乘操作


def torch_bmm(a, b):
    return torch.bmm(a, b)  # 执行PyTorch的批矩阵乘操作


def test_total_time(shapes):
    print("shape; torch bmm; inductor aten bmm; inductor triton bmm")
    for i in range(len(shapes)):
        a_shape, b_shape = shapes[i]
        print(a_shape, "x", b_shape, end="; ")

        # 使用CUDA设备生成随机张量a和b，指定数据类型为torch.float16
        a = torch.randn(a_shape, device="cuda", dtype=torch.float16)
        b = torch.randn(b_shape, device="cuda", dtype=a.dtype)

        # 禁用Triton加速并调用inductor_aten_bmm函数
        config.triton.use_bmm = False
        inductor_aten_bmm(a, b)

        # 启用Triton加速并调用inductor_triton_bmm函数
        config.triton.use_bmm = True
        inductor_triton_bmm(a, b)

        # 计算使用torch_bmm函数的执行时间并将结果转换为毫秒
        torch_ms = time_with_torch_timer(torch_bmm, (a, b)).mean * 1000

        # 禁用Triton加速并计时inductor_aten_bmm函数的执行时间
        config.triton.use_bmm = False
        ind_aten_ms = time_with_torch_timer(inductor_aten_bmm, (a, b)).mean * 1000

        # 启用Triton加速并计时inductor_triton_bmm函数的执行时间
        config.triton.use_bmm = True
        ind_triton_ms = time_with_torch_timer(inductor_triton_bmm, (a, b)).mean * 1000

        # 输出每个操作的执行时间
        print(torch_ms, ind_aten_ms, ind_triton_ms, sep="; ")


if __name__ == "__main__":
    shapes = [
        # BERT (全部形状)
        ([192, 128, 64], [192, 64, 128]),
        ([192, 128, 128], [192, 128, 64]),
        # hf_GPT2 (全部形状)
        ([12, 1024, 1024], [12, 1024, 64]),
        ([12, 1024, 64], [12, 64, 1024]),
        # hf_Albert (全部形状)
        ([12, 512, 64], [12, 64, 512]),
        ([12, 512, 512], [12, 512, 64]),
    ]

    # 执行测试函数，传入各种形状进行性能测试
    test_total_time(shapes)
```