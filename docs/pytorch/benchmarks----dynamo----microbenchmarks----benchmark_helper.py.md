# `.\pytorch\benchmarks\dynamo\microbenchmarks\benchmark_helper.py`

```
# 从 torch.utils.benchmark 模块中导入 Timer 类
from torch.utils.benchmark import Timer

# 使用 Torch 的 Timer 类来计时给定函数的执行时间
def time_with_torch_timer(fn, args, kwargs=None, iters=100):
    kwargs = kwargs or {}
    # 准备环境变量字典，包括函数参数和函数本身
    env = {"args": args, "kwargs": kwargs, "fn": fn}
    # 构造函数调用的字符串表示，用于计时器的语句参数
    fn_call = "fn(*args, **kwargs)"

    # 创建 Timer 对象并指定要计时的语句，使用上述环境变量
    timer = Timer(stmt=f"{fn_call}", globals=env)
    # 执行计时器多次，并返回平均时间
    tt = timer.timeit(iters)

    # 返回计时结果
    return tt
```