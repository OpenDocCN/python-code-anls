# `.\pytorch\benchmarks\fuser\run_benchmarks.py`

```
# 导入模块inspect、itertools、sys和time
import inspect
import itertools
import sys
import time

# 导入click模块
import click

# 导入torch模块
import torch

# 设置PyTorch的线程数为1
torch.set_num_threads(1)

# 关闭PyTorch中的融合组内联调试
torch._C._debug_set_fusion_group_inlining(False)


# 定义一个生成随机数张量的函数，形状由参数*shape指定
def rand(*shape):
    return torch.rand(*shape).mul(16).add(1)


# ------------------------------------------------------------------------------
# Shape test cases
# ------------------------------------------------------------------------------

# 返回包含两个形状为(1,)的随机数张量的元组
def scalar():
    return (rand(1), rand(1))


# 返回包含两个形状为(32,)的随机数张量的元组
def small():
    return (rand(32), rand(32))


# 返回包含两个形状为(1, 32)的随机数张量的元组
def small_2d():
    return (rand(1, 32), rand(1, 32))


# 返回包含一个形状为(4, 32)和一个形状为(32,)的随机数张量的元组
def small_broadcast():
    return (rand(4, 32), rand(32))


# 返回包含两个形状为(32, 12, 64, 64)的随机数张量的元组
def medium():
    return (rand(32, 12, 64, 64), rand(32, 12, 64, 64))


# 返回对形状为(32, 12, 64, 64)的随机数张量进行切片后的元组
def medium_sliced():
    return (rand(32, 12, 64, 64)[..., ::2], rand(32, 12, 64, 64)[..., ::2])


# 返回对形状为(32, 12, 64, 64)的随机数张量进行转置后的元组
def medium_transpose():
    return (
        rand(32, 12, 64, 64).transpose(-1, -2),
        rand(32, 12, 64, 64).transpose(-1, -2),
    )


# 返回包含两个形状为(32, 3, 224, 224)的随机数张量的元组
def medium2():
    return (rand(32, 3, 224, 224), rand(32, 3, 224, 224))


# 返回包含两个形状为(16, 32, 64)的随机数张量的元组
def medium3d():
    return (rand(16, 32, 64), rand(16, 32, 64))


# 返回包含两个形状为(32, 3, 224, 224)的随机数张量，其中第一个张量采用通道优先的内存格式
def medium_channels_last():
    return (
        rand(32, 3, 224, 224).to(memory_format=torch.channels_last),
        rand(32, 3, 224, 224).to(memory_format=torch.channels_last),
    )


# 返回包含一个形状为(32, 12, 64, 64)和一个形状为(64,)的随机数张量的元组
def medium_broadcast():
    return (rand(32, 12, 64, 64), rand(64))


# 返回包含一个形状为(32, 3, 223, 223)和一个形状为(3, 1, 1)的随机数张量的元组，第一个张量采用通道优先的内存格式
def medium_broadcast_channels_last():
    return (rand(32, 3, 223, 223).to(memory_format=torch.channels_last), rand(3, 1, 1))


# 返回包含两个形状为(8192, 8192)的随机数张量的元组
def large():
    return (rand(8192, 8192), rand(8192, 8192))


# 返回对形状为(8192, 8192)的随机数张量进行转置后的元组
def large_transpose():
    return (rand(8192, 8192).transpose(0, 1), rand(8192, 8192).transpose(0, 1))


# 返回包含两个形状为(32, 32, 256, 256)的随机数张量，其中第一个张量采用通道优先的内存格式
def large_channels_last():
    return (
        rand(32, 32, 256, 256).to(memory_format=torch.channels_last),
        rand(32, 32, 256, 256).to(memory_format=torch.channels_last),
    )


# 返回包含一个形状为(1, 32, 32, 2)和一个形状为(1024, 1, 1, 2)的随机数张量的元组
def broadcast_narrow_57611():
    return (rand(1, 32, 32, 2), rand(1024, 1, 1, 2))


# 返回包含一个形状为(64, 8, 256, 162)和一个形状为(256, 162)的随机数张量的元组
def large_broadcast_66816():
    return (rand(64, 8, 256, 162), rand(256, 162))


# ------------------------------------------------------------------------------
# Operator test cases
# ------------------------------------------------------------------------------

# 返回3 * a + b的结果
def add(a, b):
    return 3 * a + b


# 返回3 * a - b的结果
def sub(a, b):
    return 3 * a - b


# 返回3 * a * b的结果
def mul(a, b):
    return 3 * a * b


# 返回3 * a / b的结果
def div(a, b):
    return 3 * a / b


# 返回(3 * a).relu()的结果
def relu(a):
    return (3 * a).relu()


# 返回(3 * a).sigmoid()的结果
def sigmoid(a):
    return (3 * a).sigmoid()


# 返回(3 * a).tanh()的结果
def tanh(a):
    return (3 * a).tanh()


# 返回(3 * a).log()的结果
def log(a):
    return (3 * a).log()


# 返回(3 * a).exp()的结果
def exp(a):
    return (3 * a).exp()


# 返回(3 * a) ** 2的结果
def square(a):
    return (3 * a) ** 2


# 返回a * b + b的结果
def fma(a, b):
    return a * b + b


# 返回(a * b) + (a * c)的结果
def mul_mul_add_66816(a, b, c):
    return (a * b) + (a * c)


# 返回a * (a + 3).clamp(0, 6) / 6的结果
def hardswish_int(a):
    return a * (a + 3).clamp(0, 6) / 6


# 返回a * (a + 3).clamp(0.0, 6.0) / 6的结果
def hardswish(a):
    return a * (a + 3).clamp(0.0, 6.0) / 6


# 返回torch._C._nn.hardswish(a * 3)的结果
def native_hardswish(a):
    return torch._C._nn.hardswish(a * 3)


# 返回(a * 1.0).exp().log1p() / 1.0的结果
def softplus(a):
    return (a * 1.0).exp().log1p() / 1.0


# 返回a * ((a * 1.0).exp().log1p() / 1.0).tanh()的结果
def mish(a):
    return a * ((a * 1.0).exp().log1p() / 1.0).tanh()


# 定义测试用例的形状列表
SHAPES = [
    scalar,  # 标量变量，表示单个数值或对象

    small,  # 小型数据，可能是一个小数组或者单个对象

    small_2d,  # 小型二维数据，通常是一个小的二维数组

    small_broadcast,  # 小型数据，可能用于广播操作的格式

    medium,  # 中等大小的数据，可能是一个中等大小的数组或对象

    medium2,  # 第二个中等大小的数据，可能是另一个中等大小的数组或对象

    medium3d,  # 中等大小的三维数据，通常是一个中等大小的三维数组

    medium_sliced,  # 中等大小的数据，可能是通过切片操作获得的一部分数据

    medium_transpose,  # 中等大小的数据，可能是经过转置操作后的格式

    medium_channels_last,  # 中等大小的数据，通常是按通道顺序排列的格式

    medium_broadcast,  # 中等大小的数据，可能用于广播操作的格式

    medium_broadcast_channels_last,  # 中等大小的数据，可能按通道顺序排列且用于广播操作的格式

    large,  # 大型数据，通常是一个大数组或对象

    large_transpose,  # 大型数据，可能是经过转置操作后的格式

    large_channels_last,  # 大型数据，通常是按通道顺序排列的格式

    broadcast_narrow_57611,  # 窄广播数据，可能具有限制的广播格式

    large_broadcast_66816,  # 大型广播数据，通常是具有大范围广播的格式
# 全局变量，包含多个函数引用，用于操作符列表
OPERATORS = [
    add,  # 加法操作
    sub,  # 减法操作
    mul,  # 乘法操作
    div,  # 除法操作
    relu,  # ReLU 激活函数
    sigmoid,  # Sigmoid 激活函数
    tanh,  # 双曲正切函数
    log,  # 对数函数
    exp,  # 指数函数
    square,  # 平方函数
    fma,  # fused multiply-add 操作
    mul_mul_add_66816,  # 自定义函数 mul_mul_add_66816
    hardswish_int,  # 整数类型的 HardSwish 激活函数
    hardswish,  # HardSwish 激活函数
    native_hardswish,  # 原生实现的 HardSwish 激活函数
    softplus,  # Softplus 激活函数
    mish,  # Mish 激活函数
]


def time_cpu(fn, args, iters):
    # 计算 CPU 上执行函数 fn 的运行时间
    s = time.perf_counter()
    for _ in range(iters):
        fn(*args)
    e = time.perf_counter()
    return e - s


def time_cuda(fn, args, iters):
    # 计算 CUDA 上执行函数 fn 的运行时间
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn(*args)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / 1e3


def benchmark_with_timer(fn, args, timer):
    # 使用给定的计时器函数 timer 对 fn 执行性能进行评估
    timer(fn, args, 3)  # 运行 3 次以获取稳定的时间
    calibration = timer(fn, args, 1)  # 运行 1 次进行校准
    iters = int(1.0 / calibration)  # 根据校准时间计算迭代次数
    return timer(fn, args, iters) / iters  # 返回平均每次迭代的时间


def benchmark(fn, args):
    # 根据设备类型选择相应的计时器函数，并执行性能评估
    timer = time_cpu if args[0].device.type == "cpu" else time_cuda
    return benchmark_with_timer(fn, args, timer)


def micros(s):
    # 将秒转换为微秒，并以字符串形式返回
    return f"{s * 1e6:.1f}"


def with_nvfuser():
    # 启用 NVFuser，禁用其他优化
    torch._C._jit_override_can_fuse_on_cpu(False)
    torch._C._jit_override_can_fuse_on_gpu(False)
    torch._C._jit_set_texpr_fuser_enabled(False)
    torch._C._jit_set_nvfuser_enabled(True)
    torch._C._jit_set_profiling_executor(True)
    torch._C._jit_set_profiling_mode(True)


def with_nnc():
    # 启用 NNC，禁用其他优化
    torch._C._jit_override_can_fuse_on_cpu(True)
    torch._C._jit_override_can_fuse_on_gpu(True)
    torch._C._jit_set_texpr_fuser_enabled(True)
    torch._C._jit_set_nvfuser_enabled(False)
    torch._C._jit_set_profiling_executor(True)
    torch._C._jit_set_profiling_mode(True)


def with_legacy():
    # 启用传统优化模式，禁用新特性和优化
    torch._C._jit_override_can_fuse_on_cpu(True)
    torch._C._jit_override_can_fuse_on_gpu(True)
    torch._C._jit_set_texpr_fuser_enabled(False)
    torch._C._jit_set_nvfuser_enabled(False)
    torch._C._jit_set_profiling_executor(False)
    torch._C._jit_set_profiling_mode(False)


@click.command()
@click.option("--operators", default=None)
@click.option("--shapes", default=None)
def run_benchmarks(operators, shapes):
    if operators is None:
        operators = OPERATORS  # 使用默认操作符列表
    else:
        # 根据输入的操作符名称列表获取全局函数引用
        operators = [globals()[k] for k in operators.split(",")]
    if shapes is None:
        shapes = SHAPES  # 使用默认形状列表
    else:
        # 根据输入的形状名称列表获取全局函数引用
        shapes = [globals()[k] for k in shapes.split(",")]

    print("fuser,device,operator,shape,time")
    results = []
    # 使用itertools.product生成shapes和operators的所有组合，并遍历每个组合
    for shape, operator in itertools.product(shapes, operators):
        # 获取operator函数的参数个数
        nargs = len(inspect.signature(operator).parameters)
        # 调用shape函数获取参数，并根据需要扩展参数列表
        args = shape()
        if nargs > len(args):
            args = list(args)
            args += [args[-1]] * (nargs - len(args))
        args = args[:nargs]
        # 将参数列表中的每个元素转移到CUDA设备上
        args = [arg.to("cuda") for arg in args]

        # 使用benchmark函数对operator和args执行基准测试，并打印结果
        result = benchmark(operator, args)
        print(
            ",".join(
                [
                    "eager",  # 执行模式为eager
                    args[0].device.type,  # 第一个参数的设备类型
                    operator.__name__,  # 操作函数的名称
                    shape.__name__,  # 形状生成函数的名称
                    micros(result),  # 将基准测试结果转换为微秒单位
                ]
            )
        )

        # 定义bench函数，用于对operator函数使用torch.jit.trace进行追踪，再进行基准测试，并打印结果
        def bench(name):
            nnc_op = torch.jit.trace(operator, args)
            result = benchmark(nnc_op, args)
            print(
                ",".join(
                    [
                        name,  # 函数名称
                        args[0].device.type,  # 第一个参数的设备类型
                        operator.__name__,  # 操作函数的名称
                        shape.__name__,  # 形状生成函数的名称
                        micros(result),  # 将基准测试结果转换为微秒单位
                    ]
                )
            )
            sys.stdout.flush()

        # 使用NNC优化执行bench函数，并输出结果
        with_nnc()
        bench("nnc")
        # 使用NVFuser优化执行bench函数，并输出结果
        with_nvfuser()
        bench("nvfuser")
        # 使用传统方式执行bench函数，并输出结果
        with_legacy()
        bench("legacy")
# 如果当前脚本被直接执行（而不是被作为模块导入），则执行下面的代码块
if __name__ == "__main__":
    # 调用函数运行基准测试
    run_benchmarks()
```