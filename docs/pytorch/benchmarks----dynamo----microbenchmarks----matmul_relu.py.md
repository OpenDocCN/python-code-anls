# `.\pytorch\benchmarks\dynamo\microbenchmarks\matmul_relu.py`

```
from benchmark_helper import time_with_torch_timer  # 导入自定义的基准测试辅助函数

import torch  # 导入PyTorch库

import torch._dynamo  # 导入PyTorch私有模块
import torch._inductor.config as inductor_config  # 导入PyTorch私有模块中的配置模块

inductor_config.triton.mm = "triton"  # 设置私有配置变量，这里设置了模型优化器为“triton”

@torch._dynamo.optimize("inductor", nopython=True)
def inductor_mm(a, b):
    return torch.mm(a, b)  # 使用torch.mm函数实现矩阵乘法的优化版本

def torch_mm_relu(a, b):
    return torch.nn.functional.relu(torch.mm(a, b))  # 使用torch.mm进行矩阵乘法后，再应用ReLU函数

def torch_mm(a, b):
    return torch.mm(a, b)  # 使用torch.mm函数实现普通的矩阵乘法

if __name__ == "__main__":
    # Real shapes from torchbench
    a_shapes = [
        [2048, 768],
        [64, 1280],
        [2048, 768],
        [32, 2048],
        [1, 39200],
        [128, 3072],
        [16, 1280],
    ]
    b_shapes = [
        [768, 3072],
        [1280, 1000],
        [768, 768],
        [2048, 1000],
        [39200, 50],
        [3072, 1000],
        [1280, 1000],
    ]

    # Artificial larger shapes
    a_shapes += [[10240, 512], [10240, 1024]]
    b_shapes += [[512, 10240], [1024, 10240]]

    for i in range(len(a_shapes)):
        a_shape = a_shapes[i]
        b_shape = b_shapes[i]
        print("Shape:", a_shape, "x", b_shape)
        a = torch.randn(a_shape, device="cuda", dtype=torch.float16)  # 生成随机数据张量a，指定设备为cuda，数据类型为torch.float16
        b = torch.randn(b_shape, device="cuda", dtype=a.dtype)  # 生成随机数据张量b，设备和数据类型与a相同

        time_with_torch_timer(torch_mm, (a, b), string_id="torch mm")  # 使用基准测试函数测试torch_mm函数的性能
        time_with_torch_timer(torch_mm_relu, (a, b), string_id="torch mm + relu")  # 使用基准测试函数测试torch_mm_relu函数的性能
        time_with_torch_timer(inductor_mm, (a, b), string_id="inductor mm")  # 使用基准测试函数测试inductor_mm函数的性能
```