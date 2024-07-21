# `.\pytorch\benchmarks\dynamo\microbenchmarks\microbench.py`

```py
#!/usr/bin/env python3
# 导入必要的模块
import argparse  # 解析命令行参数的模块
import inspect  # 提供检查源码的功能
import sys  # 提供与 Python 解释器交互的功能

import numpy as np  # 提供多维数组和矩阵操作的功能
import tabulate  # 用于生成漂亮的表格输出的模块

import torch  # PyTorch 深度学习库

import torch._inductor  # PyTorch 内部机制模块
from torch._dynamo.backends.cudagraphs import cudagraphs_inner  # 导入 cudagraphs_inner 函数
from torch._dynamo.testing import same  # 导入 same 函数
from torch._inductor.compile_fx import compile_fx  # 导入 compile_fx 函数
from torch._inductor.utils import timed  # 导入 timed 函数

try:
    import test.test_torchinductor as tti  # 尝试导入测试模块 test_torchinductor
except ImportError:
    tti = None  # 如果导入失败，则设置 tti 为 None


def compute_speedups(args, models, example_inputs):
    # 计算预期输出
    expected = models[0](*example_inputs)
    # 验证所有模型的输出是否相同
    for model in models[1:]:
        actual = model(*example_inputs)
        assert same(actual, expected), expected[0] - actual[0]

    # 初始化一个二维数组来存储多次运行的时间
    timings = np.zeros((args.repeat, len(models)), np.float64)
    for rep in range(args.repeat):
        # 交错运行以处理频率缩放和负载变化
        for m, model in enumerate(models):
            timings[rep, m] = timed(model, example_inputs)
    # 计算中位数
    median = np.median(timings, axis=0)
    # 返回性能提升比例的列表
    return (median[0] / median[1:]).tolist()


def microbenchmark(args, model, example_inputs):
    # 编译模型为 Torch Script
    compiled_fn = compile_fx(torch.fx.symbolic_trace(model), example_inputs)
    # 使用 cudagraphs_inner 生成 Eager 模式的图形
    cudagraphs_eager = cudagraphs_inner(model, example_inputs, copy_outputs=False)
    # 使用 torch.jit.trace 生成 JIT 编译的图形
    cudagraphs_jit = cudagraphs_inner(
        torch.jit.trace(model, example_inputs), example_inputs, copy_outputs=False
    )
    # 计算速度提升比例
    return compute_speedups(
        args,
        [cudagraphs_eager, cudagraphs_jit, compiled_fn],
        example_inputs,
    )


class MyModel1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 构建一个包含线性层和 ReLU 激活函数的序列模型
        self.model = torch.nn.Sequential(
            torch.nn.Linear(1024, 1024),
            torch.nn.ReLU(),
        )

    def forward(self, input):
        # 返回模型的输出
        return (self.model(input),)


class MyModel2(torch.nn.Module):
    def forward(self, x, y):
        # 返回输入 x 和 y 的和
        return (x + y,)


class MicroBenchmarks:
    @staticmethod
    def add(a, b):
        # 返回 a 和 b 的和
        return (a + b,)

    @staticmethod
    def scale(x, m, d):
        # 返回 (x - m) / torch.clip(d, 1e-4) 的结果
        return ((x - m) / torch.clip(d, 1e-4),)

    @staticmethod
    def abs_norm(x):
        # 返回 x / (torch.abs(x) + 1) 的结果
        return (x / (torch.abs(x) + 1),)

    @staticmethod
    def add_relu_softmax(x, a):
        # 返回经过 ReLU 和 Softmax 操作后的结果
        return (torch.softmax(torch.relu(x + a), -1),)

    @staticmethod
    def sum(a, b):
        # 返回 (a + b) 的和
        return ((a + b).sum(),)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--filter", "-k", action="append", help="filter benchmarks with regexp"
    )
    parser.add_argument(
        "--exclude", "-x", action="append", help="filter benchmarks with regexp"
    )
    parser.add_argument("--devices", "-d", action="append", help="cpu or cuda")
    parser.add_argument("--size", "-s", action="append", help="cpu or cuda")
    parser.add_argument(
        "--repeat", "-n", type=int, default=30, help="number of timing runs"
    )
    parser.add_argument(
        "--threads", "-t", type=int, help="number of threads to use for eager"
    )
    )
    # 添加一个布尔类型的命令行参数 '--verbose' 或 '-v'，用于启用详细的调试输出
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="enable verbose debug printouts"
    )
    # 添加一个布尔类型的命令行参数 '--nvfuser'，用于全局启用 nvfuser
    parser.add_argument(
        "--nvfuser", action="store_true", help="enable nvfuser globally"
    )
    # 添加一个布尔类型的命令行参数 '--transpose'，用于转置一个输入
    parser.add_argument("--transpose", action="store_true", help="transpose one input")
    # 添加一个布尔类型的命令行参数 '--broadcast'，用于广播一个输入
    parser.add_argument("--broadcast", action="store_true", help="broadcast one input")
    # 解析命令行参数，并将结果存储在 args 变量中
    args = parser.parse_args()

    # 默认值设定
    args.devices = args.devices or ["cpu", "cuda"]
    args.filter = args.filter or [r"."]
    args.exclude = args.exclude or [r"^$"]
    args.size = args.size or [64, 256, 1024, 4096, 8192]

    # 根据 '--nvfuser' 参数的值设置 torch 模块中的相关配置
    if args.nvfuser:
        torch._C._jit_override_can_fuse_on_cpu(False)
        torch._C._jit_override_can_fuse_on_gpu(False)
        torch._C._jit_set_texpr_fuser_enabled(False)
        torch._C._jit_set_nvfuser_enabled(True)
    else:
        torch._C._jit_override_can_fuse_on_cpu(torch._C._llvm_enabled())
        torch._C._jit_override_can_fuse_on_gpu(True)
        torch._C._jit_set_texpr_fuser_enabled(True)
        if torch.cuda.is_available():
            torch._C._jit_set_nvfuser_enabled(False)

    # 如果指定了 '--threads' 参数，设置 torch 的线程数
    if args.threads:
        torch.set_num_threads(args.threads)
        torch._inductor.config.cpp.threads = args.threads

    # 如果指定了 '--verbose' 参数，启用 torch 的调试输出
    if args.verbose:
        torch._inductor.config.debug = True

    # 设置 torch 的 triton.autotune_pointwise 配置为 True
    torch._inductor.config.triton.autotune_pointwise = True

    # 对 MicroBenchmarks.sum 中的模型进行迭代测试
    rows = []
    for model in (MicroBenchmarks.sum,):
        nargs = len(inspect.signature(model).parameters)
        # 对指定的设备列表进行迭代
        for device in args.devices:
            # 对指定的大小列表进行迭代
            for n in args.size:
                n = int(n)
                # 输出当前测试的模型名称、设备和大小信息
                sys.stdout.write(f"{model.__name__:10} {device:4} {n:5} ")
                sys.stdout.flush()
                # 生成指定大小的随机输入数据列表
                inputs = [torch.rand((n, n), device=device) for _ in range(nargs)]
                # 如果指定了 '--broadcast' 参数，调整最后一个输入数据的维度
                if args.broadcast:
                    inputs[-1] = torch.rand((1, n), device=device)
                # 如果指定了 '--transpose' 参数，对最后一个输入数据进行转置
                if args.transpose:
                    inputs[-1] = inputs[-1].transpose(0, 1)
                # 执行微基准测试，并记录结果
                result = microbenchmark(args, model, inputs)
                rows.append([model.__name__, device, str(n)] + result)
                # 输出每项结果的加速比信息
                print(" ".join(f"{v:.2f}x" for v in result))

    # 使用 tabulate 模块输出测试结果的表格
    print(
        tabulate.tabulate(
            rows,
            headers=[
                "model",
                "dev",
                "n",
                "ts",
                "inductor",
            ],
        )
    )
# 如果当前脚本被直接执行而非被导入作为模块，则执行 main() 函数
if __name__ == "__main__":
    main()
```