# `.\pytorch\benchmarks\dynamo\microbenchmarks\inductor_mm.py`

```
import triton  # 导入 triton 库
from benchmark_helper import time_with_torch_timer  # 从 benchmark_helper 模块导入 time_with_torch_timer 函数

import torch  # 导入 PyTorch 库

import torch._dynamo  # 导入 PyTorch 的 _dynamo 模块
import torch._dynamo.config  # 导入 _dynamo 模块的 config 子模块
import torch._inductor.config as config  # 导入 _inductor 模块的 config 子模块

# 下面的标志控制是否允许在矩阵乘法中使用 TF32。此标志默认为 True。
torch.backends.cuda.matmul.allow_tf32 = True
# 下面的标志控制是否允许在 cuDNN 中使用 TF32。此标志默认为 True。
torch.backends.cudnn.allow_tf32 = True


@torch._dynamo.optimize("inductor", nopython=True)
def inductor_aten_mm(a, b):
    return torch.mm(a, b)  # 对输入张量 a 和 b 进行矩阵乘法运算


@torch._dynamo.optimize("inductor", nopython=True)
def inductor_triton_mm(a, b):
    return torch.mm(a, b)  # 对输入张量 a 和 b 进行矩阵乘法运算


def torch_mm(a, b):
    return torch.mm(a, b)  # 对输入张量 a 和 b 进行矩阵乘法运算


def triton_mm(a, b):
    return triton.ops.matmul(a, b)  # 使用 Triton 库执行输入张量 a 和 b 的矩阵乘法运算


def test_total_time(shapes):
    print("shape; torch mm; triton mm; inductor aten mm; inductor triton mm")
    for i in range(len(shapes)):
        a_shape, b_shape = shapes[i]
        print(a_shape, "x", b_shape, end="; ")
        a = torch.randn(a_shape, device="cuda", dtype=torch.float16)  # 在 CUDA 设备上创建形状为 a_shape 的随机张量 a
        b = torch.randn(b_shape, device="cuda", dtype=a.dtype)  # 在 CUDA 设备上创建形状为 b_shape 的随机张量 b

        config.triton.mm = "aten"  # 设置 Triton 配置中的矩阵乘法模式为 "aten"
        inductor_aten_mm(a, b)  # 调用经过优化的 ATen 矩阵乘法函数

        config.triton.mm = "triton"  # 设置 Triton 配置中的矩阵乘法模式为 "triton"
        inductor_triton_mm(a, b)  # 调用经过优化的 Triton 矩阵乘法函数

        torch_ms = time_with_torch_timer(torch_mm, (a, b)).mean * 1000  # 测量普通 Torch 矩阵乘法的执行时间，单位为毫秒

        triton_ms = time_with_torch_timer(triton_mm, (a, b)).mean * 1000  # 测量 Triton 矩阵乘法的执行时间，单位为毫秒

        config.triton.mm = "aten"  # 设置 Triton 配置中的矩阵乘法模式为 "aten"
        ind_aten_ms = time_with_torch_timer(inductor_aten_mm, (a, b)).mean * 1000  # 测量经过优化的 ATen 矩阵乘法的执行时间，单位为毫秒

        config.triton.mm = "triton"  # 设置 Triton 配置中的矩阵乘法模式为 "triton"
        ind_triton_ms = time_with_torch_timer(inductor_triton_mm, (a, b)).mean * 1000  # 测量经过优化的 Triton 矩阵乘法的执行时间，单位为毫秒

        print(torch_ms, triton_ms, ind_aten_ms, ind_triton_ms, sep="; ")  # 输出各个矩阵乘法的执行时间

        torch._dynamo.reset()  # 重置 PyTorch 动态编译器的状态


def test_GPU_time(shapes):
    print("shape; torch mm; triton mm; inductor aten mm; inductor triton mm")
    for i in range(len(shapes)):
        a_shape, b_shape = shapes[i]
        print(a_shape, "x", b_shape, end="; ")
        a = torch.randn(a_shape, device="cuda", dtype=torch.float16)  # 在 CUDA 设备上创建形状为 a_shape 的随机张量 a
        b = torch.randn(b_shape, device="cuda", dtype=a.dtype)  # 在 CUDA 设备上创建形状为 b_shape 的随机张量 b

        config.triton.mm = "aten"  # 设置 Triton 配置中的矩阵乘法模式为 "aten"
        inductor_aten_mm(a, b)  # 调用经过优化的 ATen 矩阵乘法函数

        config.triton.mm = "triton"  # 设置 Triton 配置中的矩阵乘法模式为 "triton"
        inductor_triton_mm(a, b)  # 调用经过优化的 Triton 矩阵乘法函数

        torch_ms, _, _ = triton.testing.do_bench(lambda: torch_mm(a, b))  # 测量普通 Torch 矩阵乘法的执行时间，单位为毫秒
        triton_ms, _, _ = triton.testing.do_bench(lambda: triton_mm(a, b))  # 测量 Triton 矩阵乘法的执行时间，单位为毫秒
        ind_aten_ms, _, _ = triton.testing.do_bench(lambda: inductor_aten_mm(a, b))  # 测量经过优化的 ATen 矩阵乘法的执行时间，单位为毫秒
        ind_triton_ms, _, _ = triton.testing.do_bench(lambda: inductor_triton_mm(a, b))  # 测量经过优化的 Triton 矩阵乘法的执行时间，单位为毫秒
        print(torch_ms, triton_ms, ind_aten_ms, ind_triton_ms, sep="; ")  # 输出各个矩阵乘法的执行时间

        torch._dynamo.reset()  # 重置 PyTorch 动态编译器的状态


if __name__ == "__main__":
    shapes = [
        # 定义不同神经网络模型的输入输出形状
        # AlexNet模型的输入输出形状
        ([128, 9216], [9216, 4096]),
        ([128, 4096], [4096, 4096]),
        ([128, 4096], [4096, 1000]),
        # BERT模型的输入输出形状
        ([2048, 768], [768, 768]),
        ([2048, 768], [768, 3072]),
        ([2048, 3072], [3072, 768]),
        # hf_GPT2模型的输入输出形状
        ([1024, 768], [768, 768]),
        ([1024, 768], [768, 3072]),
        ([1024, 3072], [3072, 768]),
        ([1024, 768], [768, 2304]),
    ]
    # 打印测试总时间
    print("test total time")
    # 调用测试总时间函数，传入模型形状列表
    test_total_time(shapes)

    # 打印测试GPU时间
    print("test GPU time")
    # 调用测试GPU时间函数，传入模型形状列表
    test_GPU_time(shapes)
# 显示在 AWS AI 集群上的测试结果预览

"""
以下是测试结果的总时间
各种形状下的计算时间：
torch mm（PyTorch 矩阵乘法）, triton mm（Triton 矩阵乘法）, inductor aten mm（Inductor Aten 矩阵乘法）, inductor triton mm（Inductor Triton 矩阵乘法）
"""

# 第一组测试数据
shape; torch mm; triton mm; inductor aten mm; inductor triton mm
[128, 9216] x [9216, 4096]; 0.07240759208798409; 0.10885953903198242; 0.20063146017491817; 0.20054904278367758
[128, 4096] x [4096, 4096]; 0.03640300128608942; 0.10960095096379519; 0.09948539081960917; 0.0996188772842288
[128, 4096] x [4096, 1000]; 0.02215010579675436; 0.12592008337378502; 0.031120930798351765; 0.0370654184371233
[2048, 768] x [768, 768]; 0.023501068353652954; 0.10804693214595318; 0.03004650119692087; 0.0276932492852211
[2048, 768] x [768, 3072]; 0.045639658346772194; 0.10883208829909563; 0.062736920081079; 0.06480381824076176
[2048, 3072] x [3072, 768]; 0.054093082435429096; 0.10804777964949608; 0.08744294755160809; 0.07766005117446184
[1024, 768] x [768, 768]; 0.021525858901441097; 0.10909941978752613; 0.02656651195138693; 0.02683836966753006
[1024, 768] x [768, 3072]; 0.027319076471030712; 0.10825308971107006; 0.040118801407516; 0.039282338693737984
[1024, 3072] x [3072, 768]; 0.034132059663534164; 0.10594133753329515; 0.05069758277386427; 0.04572632722556591
[1024, 768] x [768, 2304]; 0.02529360819607973; 0.10486091021448374; 0.03724239766597748; 0.036449190229177475

# 第二组测试数据
"""
测试 GPU 时间
各种形状下的计算时间：
torch mm（PyTorch 矩阵乘法）, triton mm（Triton 矩阵乘法）, inductor aten mm（Inductor Aten 矩阵乘法）, inductor triton mm（Inductor Triton 矩阵乘法）
"""

shape; torch mm; triton mm; inductor aten mm; inductor triton mm
[128, 9216] x [9216, 4096]; 0.09113600105047226; 0.09011200070381165; 0.21606400609016418; 0.21606400609016418
[128, 4096] x [4096, 4096]; 0.053247999399900436; 0.05222399905323982; 0.1157120019197464; 0.1157120019197464
[128, 4096] x [4096, 1000]; 0.026623999699950218; 0.02969600073993206; 0.04710400104522705; 0.05222399905323982
[2048, 768] x [768, 768]; 0.02457600086927414; 0.020479999482631683; 0.04095999896526337; 0.03993599861860275
[2048, 768] x [768, 3072]; 0.05119999870657921; 0.05222399905323982; 0.07475200295448303; 0.07577600330114365
[2048, 3072] x [3072, 768]; 0.05939200147986412; 0.05222399905323982; 0.09830400347709656; 0.0870399996638298
[1024, 768] x [768, 768]; 0.01945599913597107; 0.016383999958634377; 0.03276799991726875; 0.03276799991726875
[1024, 768] x [768, 3072]; 0.03174399957060814; 0.03276799991726875; 0.053247999399900436; 0.053247999399900436
[1024, 3072] x [3072, 768]; 0.04403200000524521; 0.03379200026392937; 0.06860800087451935; 0.062463998794555664
[1024, 768] x [768, 2304]; 0.02969600073993206; 0.02969600073993206; 0.04915200173854828; 0.048128001391887665
```