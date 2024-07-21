# `.\pytorch\benchmarks\nested\nested_bmm_bench.py`

```py
# 导入必要的库 argparse 和 random
import argparse
import random

# 导入 PyTorch 库
import torch


# 定义函数 bench，用于评估嵌套张量的乘积操作性能
def bench(nt_a, nt_b, niter):
    # 预热阶段，执行一次乘积操作
    nt_c = nt_a.bmm(nt_b)

    # 同步所有 CUDA 设备上的操作
    torch.cuda.synchronize()

    # 创建 CUDA 事件对象，用于记录运行时间
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    # 记录开始时间
    start_event.record()

    # 循环执行 niter 次乘积操作
    for iter in range(niter):
        nt_c = nt_a.bmm(nt_b)

    # 记录结束时间
    end_event.record()

    # 同步所有 CUDA 设备上的操作，确保记录的时间有效
    torch.cuda.synchronize()

    # 计算平均每次迭代的运行时间
    runtime = (start_event.elapsed_time(end_event)) / niter
    return runtime


# 定义函数 sweep_n，用于对不同数量的嵌套张量进行性能测试
def sweep_n(niter, dtype):
    # 遍历不同的张量数量
    for ntensor in [4, 8, 16, 32, 64, 128, 256]:
        # 生成随机张量列表
        tensors = [torch.randn(256, random.randint(100, 200)) for t in range(ntensor)]

        # 创建嵌套张量 nt_a，每个张量都在 CUDA 上，数据类型为 dtype
        nt_a = torch.nested.nested_tensor(
            tensors,
            dtype=dtype,
            device="cuda",
        )

        # 创建嵌套张量 nt_b，每个张量为其转置，也在 CUDA 上，数据类型为 dtype
        nt_b = torch.nested.nested_tensor(
            [t.t() for t in tensors],
            dtype=dtype,
            device="cuda",
        )

        # 对当前配置下的性能进行评估，并记录运行时间
        runtime = bench(nt_a, nt_b, niter)

        # 计算 nt_a 中各个张量的尺寸信息
        nt_a_size = torch.ops.aten._nested_tensor_size(nt_a)
        lengths = nt_a_size[:, 1]

        # 打印结果，包括张量数量、数据类型、最小长度、平均长度、最大长度和运行时间
        print(
            ",".join(
                map(
                    str,
                    [
                        ntensor,
                        dtype,
                        lengths.min().item(),
                        lengths.float().mean().item(),
                        lengths.max().item(),
                        runtime,
                    ],
                )
            )
        )


# 主程序入口，确保在单独运行时执行以下代码块
if __name__ == "__main__":
    # 设定随机数种子，以便结果可复现
    random.seed(123)

    # 创建参数解析器对象，用于处理命令行参数
    parser = argparse.ArgumentParser(description="Nested Tensor BMM Benchmark")

    # 添加命令行参数选项 --niter，指定迭代次数，默认为 10
    parser.add_argument("--niter", default="10", type=int)

    # 解析命令行参数
    args = parser.parse_args()
    niter = args.niter

    # 打印输出表头，描述各列的含义
    print("ntensor,dtype,min_length,mean_length,max_length,runtime")

    # 分别对 torch.float32 和 torch.float16 数据类型执行性能测试
    sweep_n(niter, torch.float32)
    sweep_n(niter, torch.float16)
```