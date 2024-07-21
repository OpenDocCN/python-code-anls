# `.\pytorch\functorch\examples\compilation\eager_fusion.py`

```
# 导入时间模块，用于性能测量
import time

# 导入 PyTorch 相关模块
import torch
import torch.utils

# 导入 functorch 的编译函数
from functorch.compile import aot_function, tvm_compile

# 创建随机张量 a 和 b，a 具有梯度信息
a = torch.randn(2000, 1, 4, requires_grad=True)
b = torch.randn(1, 2000, 4)

# 定义函数 f，计算 a 和 b 的点积并按第0维求和
def f(a):
    return (a * b).sum(dim=0)

# 编译函数 f 到 AOT（Ahead-of-Time）模型，使用 tvm 编译为 LLVM
fw_compiler = tvm_compile(target="llvm", tuning_logfile="fw_keops")
bw_compiler = tvm_compile(target="llvm", tuning_logfile="bw_keops")
compiled_f = aot_function(f, fw_compiler, bw_compiler)

# 设定迭代次数
iters = 10

# 对函数 compiled_f 进行计算
out = compiled_f(a)

# 计算输出的和，并进行反向传播
out.sum().backward()

# 定义性能测试函数 bench
def bench(func):
    # 记录开始时间
    begin = time.time()
    # 进行 iters 次循环
    for _ in range(iters):
        # 调用 func 计算结果并计算 sin 函数
        out = func(a).sin()
        # 对结果进行求和并反向传播
        out.sum().backward()
        # 清空梯度
        a.grad = None
    # 输出总耗时
    print(time.time() - begin)

# 定义基于 JAX 的性能测试函数 bench_jax
def bench_jax():
    # 导入 JAX 相关模块
    import jax
    import jax.numpy as jnp

    # 将 PyTorch 张量转换为 JAX 张量
    jax_a = jnp.array(a.detach().numpy())
    jax_b = jnp.array(b.detach().numpy())

    # 定义 JAX 下的函数 f，计算 a 和 b 的点积并按指定轴求和再求 sin 函数
    def f(a):
        return jnp.sin((a * jax_b).sum(axis=[0])).sum()

    # 使用 JAX 编译函数 f 并对其进行梯度计算
    jit_f = jax.jit(jax.grad(f))
    jit_f(jax_a)

    # 记录开始时间
    begin = time.time()
    # 进行 iters 次循环
    for _ in range(iters):
        # 调用 jit_f 计算结果
        out = jit_f(jax_a)
    # 等待所有操作完成
    out.block_until_ready()
    # 输出总耗时
    print(time.time() - begin)
    # for 循环结尾缺失

# 执行 bench 函数对 f 进行性能测试
bench(f)

# 执行 bench 函数对 compiled_f 进行性能测试
bench(compiled_f)

# bench_jax() 函数未被执行
# bench_jax()
```