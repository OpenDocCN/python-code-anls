# `.\pytorch\functorch\benchmarks\pointwise_scorecard.py`

```
import inspect  # 导入inspect模块，用于获取和检查活动对象的信息
import itertools  # 导入itertools模块，用于创建和操作迭代器的函数
import sys  # 导入sys模块，提供与解释器交互的函数
import time  # 导入time模块，提供时间相关的函数

import torch  # 导入PyTorch库

from functorch import pointwise_operator  # 从functorch模块导入pointwise_operator函数

torch.set_num_threads(1)  # 设置PyTorch线程数为1
torch._C._debug_set_fusion_group_inlining(False)  # 设置PyTorch的融合组内联调试为False


def rand(*shape):
    return torch.rand(*shape).mul(16).add(1)  # 生成指定形状的随机张量，元素乘以16再加1


# ------------------------------------------------------------------------------
# Shape test cases
# ------------------------------------------------------------------------------
def scalar():
    return (rand(1), rand(1))  # 返回两个形状为(1,)的随机张量


def small():
    return (rand(32), rand(32))  # 返回两个形状为(32,)的随机张量


def small_2d():
    return (rand(1, 32), rand(1, 32))  # 返回两个形状为(1, 32)的随机张量


def small_broadcast():
    return (rand(4, 32), rand(32))  # 返回一个形状为(4, 32)，另一个形状为(32,)的随机张量


def medium():
    return (rand(32, 12, 64, 64), rand(32, 12, 64, 64))  # 返回两个形状为(32, 12, 64, 64)的随机张量


def medium_sliced():
    return (rand(32, 12, 64, 64)[..., ::2], rand(32, 12, 64, 64)[..., ::2])  # 返回两个形状为(32, 12, 64, 64)的切片随机张量


def medium_transpose():
    return (
        rand(32, 12, 64, 64).transpose(-1, -2),
        rand(32, 12, 64, 64).transpose(-1, -2),
    )  # 返回两个形状为(32, 12, 64, 64)的张量，其中最后两个维度对换


def medium2():
    return (rand(32, 3, 224, 224), rand(32, 3, 224, 224))  # 返回两个形状为(32, 3, 224, 224)的随机张量


def medium3d():
    return (rand(16, 32, 64), rand(16, 32, 64))  # 返回两个形状为(16, 32, 64)的随机张量


def medium_channels_last():
    return (
        rand(32, 3, 224, 224).to(memory_format=torch.channels_last),  # 返回形状为(32, 3, 224, 224)的随机张量，存储格式为通道优先
        rand(32, 3, 224, 224).to(memory_format=torch.channels_last),  # 返回形状为(32, 3, 224, 224)的随机张量，存储格式为通道优先
    )


def medium_broadcast():
    return (rand(32, 12, 64, 64), rand(64))  # 返回一个形状为(32, 12, 64, 64)，另一个形状为(64,)的随机张量


def medium_broadcast_channels_last():
    return (
        rand(32, 3, 223, 223).to(memory_format=torch.channels_last),  # 返回形状为(32, 3, 223, 223)的随机张量，存储格式为通道优先
        rand(3, 1, 1),  # 返回形状为(3, 1, 1)的随机张量
    )


def large():
    return (rand(8192, 8192), rand(8192, 8192))  # 返回两个形状为(8192, 8192)的随机张量


def large_transpose():
    return (rand(8192, 8192).transpose(0, 1), rand(8192, 8192).transpose(0, 1))  # 返回两个形状为(8192, 8192)的张量，其中第一、第二维度对换


def large_channels_last():
    return (
        rand(32, 32, 256, 256).to(memory_format=torch.channels_last),  # 返回形状为(32, 32, 256, 256)的随机张量，存储格式为通道优先
        rand(32, 32, 256, 256).to(memory_format=torch.channels_last),  # 返回形状为(32, 32, 256, 256)的随机张量，存储格式为通道优先
    )


def pathological_broadcast():
    return (rand(1, 32, 32, 2), rand(1024, 1, 1, 2))  # 返回一个形状为(1, 32, 32, 2)，另一个形状为(1024, 1, 1, 2)的随机张量


# ------------------------------------------------------------------------------
# Operator test cases
# ------------------------------------------------------------------------------
def add(a, b):
    return a + b  # 返回张量a和b的元素加法结果


def sub(a, b):
    return a - b  # 返回张量a和b的元素减法结果


def mul(a, b):
    return a * b  # 返回张量a和b的元素乘法结果


def div(a, b):
    return a / b  # 返回张量a和b的元素除法结果


def relu(a):
    return a.relu()  # 返回张量a的ReLU函数应用结果


def sigmoid(a):
    return a.sigmoid()  # 返回张量a的sigmoid函数应用结果


def tanh(a):
    return a.tanh()  # 返回张量a的双曲正切函数应用结果


def log(a):
    return a.log()  # 返回张量a的自然对数函数应用结果


def exp(a):
    return a.exp()  # 返回张量a的指数函数应用结果


def square(a):
    return a**2  # 返回张量a的平方


def fma(a, b):
    return a * b + b  # 返回张量a和b的乘积加上b的结果


def hardswish(a):
    return a * (a + 3.0).clamp(0.0, 6.0) / 6.0  # 返回张量a的hardswish函数应用结果


def native_hardswish(a):
    return torch._C._nn.hardswish(a)  # 返回张量a的原生hardswish函数应用结果


def softplus(a):
    return (a * 1.0).exp().log1p() / 1.0  # 返回张量a的softplus函数应用结果


def mish(a):
    return a * ((a * 1.0).exp().log1p() / 1.0).tanh()  # 返回张量a的mish函数应用结果


# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------
def time_cpu(fn, args, iters):
    s = time.perf_counter()  # 记录起始时间
    # 执行循环 `iters` 次，每次调用函数 `fn` 并传入参数 `args`
    for _ in range(iters):
        fn(*args)
    # 获取循环结束后的性能计数器时间
    e = time.perf_counter()
    # 返回循环执行所花费的时间，即结束时间减去开始时间 `s`
    return e - s
def time_cuda(fn, args, iters):
    # 创建 CUDA 计时事件对象，用于测量时间
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    # 记录开始时间
    start.record()
    # 执行指定次数的函数调用
    for _ in range(iters):
        fn(*args)
    # 记录结束时间
    end.record()
    # 等待 CUDA 所有任务完成
    torch.cuda.synchronize()
    # 计算并返回执行时间（秒）
    return start.elapsed_time(end) / 1e3


def benchmark_with_timer(fn, args, timer):
    # 使用给定的计时器函数执行函数多次，并返回平均时间
    timer(fn, args, 3)
    # 使用计时器函数进行一次校准，以确定执行时间
    calibration = timer(fn, args, 1)
    # 根据校准时间计算合适的迭代次数
    iters = int(1.0 / calibration)
    # 使用计时器函数执行函数，并返回平均时间
    return timer(fn, args, iters) / iters


def benchmark(fn, args):
    # 根据设备类型选择相应的计时器函数
    timer = time_cpu if args[0].device.type == "cpu" else time_cuda
    # 使用指定的函数和参数执行基准测试，并返回平均时间
    return benchmark_with_timer(fn, args, timer)


def micros(s):
    # 将秒转换为微秒，并格式化输出
    return f"{s * 1e6:.1f}"


shapes = [
    scalar,
    small,
    small_2d,
    small_broadcast,
    medium,
    medium2,
    medium3d,
    medium_sliced,
    medium_transpose,
    medium_channels_last,
    medium_broadcast,
    medium_broadcast_channels_last,
    large,
    large_transpose,
    large_channels_last,
    pathological_broadcast,
]

operators = [
    add,
    sub,
    mul,
    div,
    relu,
    sigmoid,
    tanh,
    log,
    exp,
    square,
    fma,
    hardswish,
    native_hardswish,
]

nope = set()
# 遍历形状和运算符的组合，执行基准测试和校验
for shape, operator in itertools.product(shapes, operators):
    # 确定运算符的参数数量
    nargs = len(inspect.signature(operator).parameters)
    # 获取对应形状的参数
    args = shape()[:nargs]

    try:
        # 如果是特定的形状，抛出运行时错误
        if shape == medium_transpose:
            raise RuntimeError("pointwise_operator hangs on medium_transpose")
        # 进行点对点操作并校验结果
        pw_op = pointwise_operator(operator)
        torch.testing.assert_close(operator(*args), pw_op(*args))
    except Exception:
        # 打印错误信息，记录失败的运算符和形状
        print(f"pointwise_operator failed on {operator.__name__}, {shape.__name__}")
        nope.add((operator, shape))

    # 对运算符进行 Torch 脚本编译
    ts_op = torch.jit.script(operator)
    # 执行脚本编译后的运算符，并校验结果
    torch.testing.assert_close(operator(*args), ts_op(*args))


# 输出基准测试结果的表头
print("fuser,device,operator,shape,time")
results = []
# 遍历形状和运算符的组合，执行基准测试并打印结果
for shape, operator in itertools.product(shapes, operators):
    # 确定运算符的参数数量
    nargs = len(inspect.signature(operator).parameters)
    # 获取对应形状的参数
    args = shape()[:nargs]

    # 执行基准测试，并输出结果
    result = benchmark(operator, args)
    print(
        ",".join(
            [
                "eager",
                args[0].device.type,
                operator.__name__,
                shape.__name__,
                micros(result),
            ]
        )
    )
    try:
        # 如果是特定的形状，抛出运行时错误
        if shape == medium_transpose:
            raise RuntimeError("pointwise_operator hangs on medium_transpose")
        # 如果该运算符和形状组合在失败集合中，抛出运行时错误
        if (operator, shape) in nope:
            raise RuntimeError("pointwise_operator fails on medium_transpose")
        # 进行点对点操作，并执行基准测试
        pw_op = pointwise_operator(operator)
        result = benchmark(pw_op, args)
        # 输出点对点操作的基准测试结果
        print(
            ",".join(
                [
                    "pointwise",
                    args[0].device.type,
                    operator.__name__,
                    shape.__name__,
                    micros(result),
                ]
            )
        )
    except Exception as e:
        # 打印错误信息
        print(f"Error: {e}")
    # 捕获任何异常并处理
    except Exception:
        # 将以下内容以逗号分隔的形式打印到标准输出
        print(
            ",".join(
                [
                    "pointwise",                   # 固定字符串 "pointwise"
                    args[0].device.type,           # 第一个参数的设备类型
                    operator.__name__,             # 运算符的名称
                    shape.__name__,                # 形状函数的名称
                    micros(float("nan")),          # 对 NaN 值进行微秒转换
                ]
            )
        )

    # 使用 Torch 的 JIT 脚本化操作符
    ts_op = torch.jit.script(operator)
    # 对脚本化操作符进行基准测试，返回结果
    result = benchmark(ts_op, args)
    # 将以下内容以逗号分隔的形式打印到标准输出
    print(
        ",".join(
            [
                "fuser",                        # 固定字符串 "fuser"
                args[0].device.type,            # 第一个参数的设备类型
                operator.__name__,              # 运算符的名称
                shape.__name__,                 # 形状函数的名称
                micros(result),                 # 对基准测试结果进行微秒转换
            ]
        )
    )
    # 刷新标准输出缓冲区
    sys.stdout.flush()
```