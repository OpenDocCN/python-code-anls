# `.\pytorch\benchmarks\dynamo\microbenchmarks\bench_mm_fusion.py`

```
# flake8: noqa
# 引入triton和PrettyTable库
import triton
from prettytable import PrettyTable

# 引入torch库及其内部模块
import torch

import torch._dynamo
import torch._inductor.config

# 设置随机种子
torch.manual_seed(0)

# 控制是否在矩阵乘法中允许使用TF32格式的标志
torch.backends.cuda.matmul.allow_tf32 = True


class Func(object):
    # 优化矩阵乘法的方法，不同变体根据需要进行优化
    @torch._dynamo.optimize("inductor")
    def mm(a, b, bias):
        y = torch.mm(a, b)
        return y

    @torch._dynamo.optimize("inductor")
    def mm_add(a, b, bias):
        y = torch.mm(a, b)
        return y + bias

    @torch._dynamo.optimize("inductor")
    def mm_relu(a, b, bias):
        y = torch.mm(a, b)
        return torch.relu(y)

    @torch._dynamo.optimize("inductor")
    def mm_add_relu(a, b, bias):
        y = torch.mm(a, b)
        y += bias
        return torch.relu(y)


def bench(shape, layer_id, p, fusion_types=[""]):
    dtype = torch.float16
    M, K = shape[0]
    _, N = shape[1]
    torch.manual_seed(0)
    
    # 在CUDA设备上分配输入张量
    a = torch.randn(shape[0], device="cuda", dtype=dtype)
    b = torch.randn(shape[1], device="cuda", dtype=dtype)

    # 计算每秒浮点操作数（TFLOPS）
    def tflops(ms):
        return M * K * N / ms * 1e-9

    # 构建表格行数据
    row = [layer_id]
    for fusion_type in fusion_types:
        if fusion_type == "":
            fn_mm = getattr(Func, "mm")
        else:
            fn_mm = getattr(Func, f"mm_{fusion_type}")

        if "add" in fusion_type:
            bias = torch.randn((M, N), dtype=dtype, device="cuda")
        else:
            bias = None

        args = (a, b, bias)

        # 定义执行函数
        def fn():
            return fn_mm(*args)

        # 设置torch._inductor.config.triton.mm以确定使用的矩阵乘法实现
        torch._inductor.config.triton.mm = "aten"
        torch_mm_ms, _, _ = triton.testing.do_bench(fn)

        # 切换至triton后端以评估性能
        torch._inductor.config.triton.mm = "triton"
        torch._dynamo.reset()  # 重置以生成新的Python代码
        torch._inductor.metrics.reset()
        triton_mm_ms, _, _ = triton.testing.do_bench(fn)

        # 断言确保生成的内核数量为1，用于代码生成
        assert torch._inductor.metrics.generated_kernel_count == 1, "codegen #kernel != 1"
        
        # 添加每种操作类型的TFLOPS值到行数据中
        row.extend([tflops(torch_mm_ms), tflops(triton_mm_ms)])

    # 向表格p添加完整的行数据
    p.add_row(row)


# 定义多个神经网络层次的形状
fusion_types = ["", "add", "relu", "add_relu"]
shapes = [
    # alexnet
    ([128, 9216], [9216, 4096]),
    ([128, 4096], [4096, 4096]),
    ([128, 4096], [4096, 1000]),
    # BERT
    ([2048, 768], [768, 768]),
    ([2048, 768], [768, 3072]),
    ([2048, 3072], [3072, 768]),
    # hf_GPT2
    ([1024, 768], [768, 768]),
    ([1024, 768], [768, 3072]),
    ([1024, 3072], [3072, 768]),
    ([1024, 768], [768, 2304]),
]

# 创建PrettyTable实例p，并设置列名
p = PrettyTable()
field_names = ["layer"]
for fusion_type in fusion_types:
    if fusion_type == "":
        field_names.append("torch mm")
        field_names.append("triton mm")
    else:
        field_names.append(f"torch mm+{fusion_type}")
        field_names.append(f"triton mm+{fusion_type}")

p.field_names = field_names
p.float_format = ".3"
# 对于给定的形状列表 `shapes`，使用 `enumerate` 函数迭代，返回索引和对应的形状 `shape`
for id, shape in enumerate(shapes):
    # 调用 bench 函数，对当前形状 shape 进行基准测试，传入索引 id、参数 p 和融合类型列表 fusion_types
    bench(shape, id, p, fusion_types)

# 打印参数 p 的当前值（假设是一个变量）
print(p)
```