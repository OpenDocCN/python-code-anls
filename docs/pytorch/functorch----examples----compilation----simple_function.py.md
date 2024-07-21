# `.\pytorch\functorch\examples\compilation\simple_function.py`

```py
# 导入时间模块，用于性能测试时计时
import time

# 导入 PyTorch 库
import torch

# 导入函数式 Torch 扩展模块
from functorch import grad, make_fx
# 导入 Functorch 编译模块
from functorch.compile import nnc_jit


# 定义一个函数 f，计算输入张量的元素的正弦值的和
def f(x):
    return torch.sin(x).sum()


# 生成一个随机张量作为输入
inp = torch.randn(100)
# 使用 grad 函数对函数 f 进行自动求导，得到梯度函数 grad_pt
grad_pt = grad(f)
# 使用 make_fx 函数将 grad_pt 转换为可用于 FX（functionally eXpressed）的版本，得到 grad_fx
grad_fx = make_fx(grad_pt)(inp)
# 使用 nnc_jit 函数对 grad_pt 进行编译优化，得到 grad_nnc
grad_nnc = nnc_jit(grad_pt)


# 定义性能测试函数 bench，用于测试给定函数的执行时间
def bench(name, f, iters=10000, warmup=3):
    # 预热阶段，执行 warmup 次函数调用，忽略计时结果
    for _ in range(warmup):
        f()
    # 正式计时阶段，执行 iters 次函数调用并计时
    begin = time.time()
    for _ in range(iters):
        f()
    # 打印性能测试结果，输出名称及执行时间
    print(f"{name}: ", time.time() - begin)


# 使用 bench 函数分别测试 PyTorch 自动求导、FX 优化后的自动求导和 NNC 编译优化后的自动求导的执行时间
bench("Pytorch: ", lambda: grad_pt(inp))
bench("FX: ", lambda: grad_fx(inp))
bench("NNC: ", lambda: grad_nnc(inp))
```