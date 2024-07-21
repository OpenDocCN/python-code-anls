# `.\pytorch\benchmarks\dynamo\microbenchmarks\overheads.py`

```
# 导入时间相关模块
import time
import timeit

# 导入科学计算库 NumPy
import numpy as np

# 导入 PyTorch 深度学习框架
import torch

# 定义一个简单的函数，对输入 x 加 1 并返回结果
def add1(x):
    return x + 1

# 定义一个性能测试函数
def bench(name, fn, requires_grad):
    # 重置 Torch 的内部状态
    torch._dynamo.reset()
    # 生成一个随机张量 x，如果需要梯度则设置 requires_grad=True
    x = torch.randn(1, requires_grad=requires_grad)
    # 记录开始时间
    start = time.perf_counter()
    # 执行函数 fn(x) 3 次
    for _ in range(3):
        fn(x)
    # 记录结束时间
    end = time.perf_counter()

    # 使用 timeit.repeat 函数测试执行 fn(x) 1000 次的性能
    results = timeit.repeat(lambda: fn(x), number=1000, repeat=1000)
    # 打印性能测试结果，包括中位数和热身时间
    print(f"{name} {np.median(results)*1000:.1f}us (warmup={end-start:.1f}s)")

# 主函数
def main():
    # 输出 requires_grad=False 的测试结果
    print("requires_grad=False")
    bench("eager   ", add1, False)  # 测试普通的即时执行模式
    bench("compiled", torch.compile(add1), False)  # 测试编译后的模式
    print()
    
    # 输出 requires_grad=True 的测试结果
    print("requires_grad=True")
    bench("eager   ", add1, True)  # 测试普通的即时执行模式
    bench("compiled", torch.compile(add1), True)  # 测试编译后的模式
    print()
    
    # 进入推断模式的测试
    print("inference_mode()")
    with torch.inference_mode():
        bench("eager   ", add1, False)  # 测试普通的即时执行模式
        bench("compiled", torch.compile(add1), False)  # 测试编译后的模式

# 如果作为主程序运行，则执行主函数
if __name__ == "__main__":
    main()
```