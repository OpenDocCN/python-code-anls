# `.\pytorch\functorch\benchmarks\operator_authoring.py`

```py
# 导入计时器模块
import timeit
# 导入偏函数模块
from functools import partial

# 导入科学计算库NumPy
import numpy as np
# 导入数据处理库Pandas
import pandas as pd

# 导入PyTorch深度学习框架
import torch

# 导入functorch库中的点对点操作编译器
from functorch.compile import pointwise_operator

# 是否写入CSV文件的标志
WRITE_CSV = False
# 是否使用CUDA加速的标志
CUDA = False
# 待测试的数据大小列表
SIZES = [1, 512, 8192]
# 每个大小对应的测试次数列表
NUMBER = [100, 10, 1, 1]
# 测试重复次数
REPEAT = 20

# 定义用于非原地操作的加和归一化函数装饰器
@pointwise_operator
def nnc_add(a, b):
    return a + b

# 定义用于非原地操作的加和归一化函数装饰器，带有均值和标准差参数
@pointwise_operator
def nnc_addnorm(a, b, mean, std):
    return (a + b - mean) / std

# 原始的加和归一化函数，非装饰器形式
def eager_addnorm(a, b, mean, std):
    return (a + b - mean) / std

# 原地操作的加和归一化函数
def inplace_addnorm(a, b, mean, std, out):
    # 将a和b相加，并将结果存储在out中
    out = torch.add(a, b, out=out)
    # 从out中减去均值mean
    torch.sub(out, mean, out=out)
    # 将out除以标准差std
    torch.div(out, std, out=out)
    return out

# 将eager_addnorm函数编译为Torch脚本
ts_addnorm = torch.jit.script(eager_addnorm)
# 将inplace_addnorm函数编译为Torch脚本
ts_ip_addnorm = torch.jit.script(inplace_addnorm)

# 如果CUDA加速开启，则定义一个可能同步的函数装饰器
def maybe_synced(fn):
    if CUDA:
        synchronize = torch.cuda.synchronize
        synchronize()  # 预热GPU同步

        def _fn():
            result = fn()
            synchronize()
            return result

        return _fn
    return fn

# 定义基准测试循环函数，返回每个大小下的加速比数组
def benchmark_loop(setup):
    # 创建一个存储结果的数组，维度为(REPEAT, len(SIZES), 2)
    result = np.zeros((REPEAT, len(SIZES), 2), dtype=np.float64)
    # 遍历每个数据大小和测试次数
    for s, n in enumerate(SIZES):
        # 调用setup函数获取两种操作函数
        nnc, aten = setup(n)
        # 如果CUDA开启，则应用同步函数装饰器
        nnc = maybe_synced(nnc)
        aten = maybe_synced(aten)

        # 执行测试循环
        for r in range(result.shape[0]):
            # 测量nnc操作的执行时间
            result[r, s, 0] = timeit.timeit(nnc, number=NUMBER[s])
            # 测量aten操作的执行时间
            result[r, s, 1] = timeit.timeit(aten, number=NUMBER[s])

    # 计算结果的中位数
    result = np.median(result, axis=0)
    # 断言结果形状为(len(SIZES), 2)
    assert result.shape == (len(SIZES), 2)
    # 计算加速比
    result = result[:, 1] / result[:, 0]
    # 打印加速比结果
    print(result)
    return result

# 定义测试函数，用于测试非原地操作
def test(make_args, nnc=nnc_add, aten=torch.add):
    # 定义测试数据的设置函数
    def setup(n):
        args = make_args(n)
        result_aten = aten(*args)
        result_nnc = nnc(*args)
        assert result_nnc.dtype == result_aten.dtype
        assert result_nnc.size() == result_aten.size()
        assert result_nnc.stride() == result_aten.stride()
        torch.testing.assert_close(result_aten, result_nnc)
        return (lambda: nnc(*args), lambda: aten(*args))

    return benchmark_loop(setup)

# 定义测试函数，用于测试原地操作
def test_inplace(make_args, nnc=nnc_add, aten=torch.add):
    # 定义原地操作的测试数据设置函数
    def inplace_setup(n):
        a, b = make_args(n)
        result_aten = torch.clone(a)
        result_nnc = torch.clone(a)
        nnc(result_nnc, b, out=result_nnc)
        aten(result_aten, b, out=result_aten)
        torch.testing.assert_close(result_aten, result_nnc)
        return (lambda: nnc(a, b, out=a), lambda: aten(a, b, out=a))

    return benchmark_loop(inplace_setup)

# 定义测试函数，用于测试输出张量的操作
def test_out(make_args, out, nnc=nnc_add, aten=torch.add):
    # 定义输出张量操作的测试数据设置函数
    def out_setup(n):
        args = make_args(n)
        result_aten = out(n)
        result_nnc = out(n)
        aten(*args, out=result_aten)
        nnc(*args, out=result_nnc)
        torch.testing.assert_close(result_aten, result_nnc)
        result = out(n)
        return (lambda: nnc(*args, out=result), lambda: aten(*args, out=result))

    return benchmark_loop(out_setup)

# 定义测试函数，用于测试反向传播
def test_backwards(make_args, nnc=nnc_add, aten=torch.add):
    # 定义一个函数 `backwards_setup`，接受参数 `n`
    def backwards_setup(n):
        # 调用 `make_args` 函数生成参数列表 `args`
        args = make_args(n)
        # 在参数列表 `args` 中找到第一个需要梯度计算的变量 `grad_var`
        (grad_var,) = (a for a in args if a.requires_grad)
        # 调用 `aten` 函数对参数进行计算，并对结果求和后进行反向传播
        aten(*args).sum().backward()
        # 将第一个反向传播计算得到的梯度保存在 `correct` 中
        correct = grad_var.grad.clone()
        # 清零 `grad_var` 的梯度，以便进行下一次反向传播计算
        grad_var.grad.zero_()
        # 再次调用 `nnc` 函数对参数进行计算，并对结果求和后进行反向传播
        nnc(*args).sum().backward()
        # 使用 PyTorch 提供的测试函数检查第二次反向传播计算得到的梯度是否与 `correct` 相近
        torch.testing.assert_close(correct, grad_var.grad)
        # 返回两个函数对象，分别执行 `nnc` 和 `aten` 的计算和反向传播
        return (
            lambda: nnc(*args).sum().backward(),
            lambda: aten(*args).sum().backward(),
        )

    # 调用 `benchmark_loop` 函数，将 `backwards_setup` 函数作为参数传入，并返回其结果
    return benchmark_loop(backwards_setup)
# 主函数，程序的入口点
def main():
    # 设置 Torch 的线程数为1，这里可能会添加并行支持（TODO: jansel）
    torch.set_num_threads(1)
    # 允许在 CPU 上进行融合操作优化
    torch._C._jit_override_can_fuse_on_cpu(True)

    # 根据 CUDA 变量决定设备类型是 "cuda" 还是 "cpu"
    device = "cuda" if CUDA else "cpu"
    # 创建一个部分应用的函数 I，用于在指定设备上生成随机整数张量
    I = partial(torch.randint, 0, 100, device=device)
    # 创建一个部分应用的函数 R，用于在指定设备上生成随机正态分布张量
    R = partial(torch.randn, device=device)

    # 错误的代码片段，下面的代码应该是列表的开始，而不是右括号
    ]

    # 使用 results 列表中的数据创建 Pandas 的 DataFrame
    df = pd.DataFrame(
        np.stack([r for n, r in results]),  # 使用 results 中的数据创建堆叠数组
        columns=[f"{n}x{n}".rjust(9) for n in SIZES],  # 列名，以指定大小格式化
        index=[n for n, r in results],  # 行索引，使用 results 中的名称
    )

    # 如果 WRITE_CSV 为真，将 DataFrame 写入到 CSV 文件
    if WRITE_CSV:
        df.to_csv("../operator_authoring_results.csv")
        print("wrote ../operator_authoring_results.csv")  # 打印写入成功消息

    # 打印空行
    print()
    # 打印速度提升信息的标题
    print("Speedups over aten")
    # 设置 Pandas 显示浮点数格式，保留两位小数并追加 "x" 字符
    pd.options.display.float_format = "{:.2f}x".format
    # 打印 DataFrame，显示速度提升信息
    print(df)


if __name__ == "__main__":
    main()
```