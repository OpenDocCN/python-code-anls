# `.\pytorch\benchmarks\distributed\ddp\benchmark.py`

```py
#!/usr/bin/env python3
#
# Measure distributed training iteration time.
#
# This program performs a sweep over a) a number of model architectures, and
# b) an increasing number of processes. This produces a 1-GPU baseline,
# an 8-GPU baseline (if applicable), as well as measurements for however
# many processes can participate in training.
#

import argparse  # 导入用于解析命令行参数的模块
import itertools  # 导入用于创建迭代器的模块
import json  # 导入处理 JSON 数据的模块
import os  # 导入操作系统相关功能的模块
import shlex  # 导入用于解析命令行字符串的模块
import subprocess  # 导入执行外部命令的模块
import sys  # 导入与 Python 解释器交互的模块
import time  # 导入时间相关的模块

import numpy as np  # 导入数值计算库 NumPy
import torchvision  # 导入视觉处理工具包 torchvision

import torch  # 导入深度学习框架 PyTorch
import torch.distributed as dist  # 导入 PyTorch 分布式训练模块
import torch.nn as nn  # 导入神经网络模块
import torch.optim as optim  # 导入优化器模块


def allgather_object(obj):
    # 创建一个长度等于进程数的列表 out，并将 obj 分发到每个进程
    out = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(out, obj)
    return out


def allgather_run(cmd):
    # 运行给定的 shell 命令 cmd，并收集所有进程的输出
    proc = subprocess.run(shlex.split(cmd), capture_output=True)
    assert proc.returncode == 0  # 确保命令执行成功
    return allgather_object(proc.stdout.decode("utf-8"))


def allequal(iterator):
    iterator = iter(iterator)
    try:
        first = next(iterator)
    except StopIteration:
        return True
    # 检查迭代器中的所有元素是否相等
    return all(first == rest for rest in iterator)


def benchmark_process_group(pg, benchmark, use_ddp_for_single_rank=True):
    torch.manual_seed(pg.rank())  # 设置随机种子
    torch.cuda.manual_seed(pg.rank())  # 设置 CUDA 随机种子

    model = benchmark.create_model()  # 创建模型
    data = [(benchmark.generate_inputs(), benchmark.generate_target())]  # 生成输入数据和目标数据
    criterion = nn.CrossEntropyLoss()  # 定义损失函数
    optimizer = optim.SGD(model.parameters(), 0.001, momentum=0.9, weight_decay=1e-4)  # 定义优化器

    if use_ddp_for_single_rank or pg.size() > 1:
        # 如果使用 DDP 进行单进程训练或者进程数大于 1，使用 DistributedDataParallel
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            process_group=pg,
            bucket_cap_mb=benchmark.bucket_size,
        )

    measurements = []  # 初始化用于存储测量时间的列表
    warmup_iterations = 5  # 预热迭代次数
    measured_iterations = 10  # 测量迭代次数
    for inputs, target in data * (warmup_iterations + measured_iterations):
        start = time.time()  # 记录开始时间
        output = model(*inputs)  # 模型前向传播
        loss = criterion(output, target)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新模型参数
        torch.cuda.synchronize()  # 同步所有 CUDA 设备
        measurements.append(time.time() - start)  # 计算单次迭代时间并添加到列表中

    # 抛弃预热迭代的测量结果
    return measurements[warmup_iterations:]


def run_benchmark(benchmark, ranks, opts):
    group = dist.new_group(ranks=ranks, backend=benchmark.distributed_backend)  # 创建新的分布式进程组
    measurements = []

    if dist.get_rank() in set(ranks):
        if not opts:
            opts = {}  # 如果 opts 为空，则初始化为空字典
        measurements = benchmark_process_group(group, benchmark, **opts)  # 执行基准测试
    dist.destroy_process_group(group)  # 销毁进程组
    dist.barrier()  # 等待所有进程完成

    # 聚合测量结果以更好地估计百分位数
    return list(itertools.chain(*allgather_object(measurements)))


def sweep(benchmark):
    # 综合要运行的基准测试集合
    # benchmarks 是一个列表，包含了 ("string prefix", [rank...]) 的元组
    benchmarks = []
    # 将性能基准信息附加到 benchmarks 列表中
    def append_benchmark(prefix, ranks, opts=None):
        # 格式化前缀，显示 GPU 数量和描述信息
        prefix = f"{len(ranks):4} GPUs -- {prefix}"
        # 将 benchmark 元组添加到 benchmarks 列表中
        benchmarks.append((prefix, ranks, opts))

    # 在本地打印消息，仅当进程的 rank 为 0 时输出
    def local_print(msg):
        if dist.get_rank() == 0:
            # 打印消息，强制刷新输出缓冲区
            print(msg, end="", flush=True)  # noqa: E999

    # 打印表头信息
    def print_header():
        # 在本地打印空行
        local_print("\n")
        # 打印表头的列标题
        local_print("%22s" % "")
        for p in [50, 75, 90, 95]:
            # 打印每列的标题信息，包括 "sec/iter" 和 "ex/sec"
            local_print("%14s%10s" % ("sec/iter", "ex/sec"))
        # 在本地打印空行
        local_print("\n")

    # 打印测量结果
    def print_measurements(prefix, nelem, measurements):
        # 对测量结果进行排序
        measurements = sorted(measurements)
        # 打印前缀信息
        local_print("%8s:" % prefix)
        for p in [50, 75, 90, 95]:
            # 计算百分位数值
            v = np.percentile(measurements, p)
            # 打印百分位数值和每秒处理元素数
            local_print("  p%02d:  %1.3fs  %6d/s" % (p, v, nelem / v))
        # 在本地打印空行
        local_print("\n")

    # 每个进程单独运行一次以进行预热（CUDA 初始化等）
    append_benchmark("  warmup", [dist.get_rank()], {"use_ddp_for_single_rank": False})

    # 单机基线
    append_benchmark("  no ddp", range(1), {"use_ddp_for_single_rank": False})
    append_benchmark("   1M/1G", range(1))
    append_benchmark("   1M/2G", range(2))
    append_benchmark("   1M/4G", range(4))

    # 多机器性能基准
    for i in range(1, (dist.get_world_size() // 8) + 1):
        # 添加多机器性能基准到 benchmarks 列表中
        append_benchmark("   %dM/8G" % i, range(i * 8))

    # 按照 GPU 数量递增的顺序运行基准测试
    print_header()
    results = []
    for prefix, ranks, opts in sorted(benchmarks, key=lambda tup: len(tup[1])):
        # 将 ranks 转换为列表
        ranks = list(ranks)
        # 运行基准测试，获取测量结果
        measurements = run_benchmark(benchmark, ranks, opts)
        if "warmup" not in prefix:
            # 打印测量结果
            print_measurements(prefix, benchmark.batch_size, measurements)
            # 添加结果到 results 列表中
            results.append({"ranks": ranks, "measurements": measurements})

    # 返回所有结果
    return results
class Benchmark:
    # Benchmark 类，用于定义基准测试的基本结构和接口
    def __init__(self, device, distributed_backend, bucket_size):
        # 初始化函数，设定基准测试的设备、批量大小、分布式后端和桶大小
        self.device = device
        self.batch_size = 32
        self.distributed_backend = distributed_backend
        self.bucket_size = bucket_size

    def __str__(self):
        # 返回对象的字符串表示形式，由子类实现
        raise NotImplementedError

    def create_model(self):
        # 创建模型的抽象方法，由子类实现
        raise NotImplementedError

    def generate_inputs(self):
        # 生成输入数据的抽象方法，由子类实现
        raise NotImplementedError

    def generate_target(self):
        # 生成目标数据的抽象方法，由子类实现
        raise NotImplementedError


class TorchvisionBenchmark(Benchmark):
    # TorchvisionBenchmark 类，继承自 Benchmark，用于实现基于 torchvision 的基准测试
    def __init__(self, device, distributed_backend, bucket_size, model):
        # 初始化函数，调用父类构造函数并添加模型参数
        super().__init__(
            device,
            distributed_backend,
            bucket_size,
        )
        self.model = model  # 设置模型属性

    def __str__(self):
        # 返回基准测试对象的字符串表示形式，包含模型和批量大小信息
        return f"{self.model} with batch size {self.batch_size}"

    def create_model(self):
        # 创建模型实例并移动到指定设备上
        return torchvision.models.__dict__[self.model]().to(self.device)

    def generate_inputs(self):
        # 生成随机输入数据列表，大小为 [批量大小, 3, 224, 224]，并放置在指定设备上
        return [torch.rand([self.batch_size, 3, 224, 224], device=self.device)]

    def generate_target(self):
        # 生成目标张量，填充值为 1，数据类型为 long，并放置在指定设备上
        return torch.tensor([1] * self.batch_size, dtype=torch.long, device=self.device)


def main():
    # 主函数，用于执行分布式 PyTorch 基准测试套件
    parser = argparse.ArgumentParser(description="PyTorch distributed benchmark suite")
    parser.add_argument("--rank", type=int, default=os.environ["RANK"])  # 进程在全局组中的排名
    parser.add_argument("--world-size", type=int, required=True)  # 全局组的大小
    parser.add_argument("--distributed-backend", type=str, default="nccl")  # 分布式后端类型
    parser.add_argument("--bucket-size", type=int, default=25)  # 桶的大小
    parser.add_argument("--master-addr", type=str, required=True)  # 主节点地址
    parser.add_argument("--master-port", type=str, required=True)  # 主节点端口
    parser.add_argument("--model", type=str)  # 模型名称
    parser.add_argument(
        "--json", type=str, metavar="PATH", help="Write file with benchmark results"  # JSON 输出文件路径
    )
    args = parser.parse_args()  # 解析命令行参数

    num_gpus_per_node = torch.cuda.device_count()  # 获取当前节点上 GPU 的数量
    assert num_gpus_per_node == 8, "Expected 8 GPUs per machine"  # 断言当前节点上有 8 个 GPU

    # 用于传递基准测试元数据的全局进程组，如测量值，而不是用于实际基准测试
    dist.init_process_group(
        backend="gloo",
        init_method=f"tcp://{args.master_addr}:{args.master_port}",
        rank=args.rank,
        world_size=args.world_size,
    )

    output = allgather_run("nvidia-smi topo -m")  # 运行 nvidia-smi topo -m 命令并收集输出
    if not allequal(output):  # 检查所有节点的输出是否一致
        print('Output of "nvidia-smi topo -m" differs between machines')
        sys.exit(1)  # 输出不一致则退出程序
    # 如果当前进程的 rank 是 0，输出以下信息
    if args.rank == 0:
        print("-----------------------------------")
        print("PyTorch distributed benchmark suite")
        print("-----------------------------------")
        print("")
        print(f"* PyTorch version: {torch.__version__}")
        print(f"* CUDA version: {torch.version.cuda}")
        print(f"* Distributed backend: {args.distributed_backend}")
        print(f"* Maximum bucket size: {args.bucket_size}MB")
        print("")
        print("--- nvidia-smi topo -m ---")
        print("")
        # 打印来自 nvidia-smi 命令输出的 GPU 拓扑信息
        print(output[0])
        print("--------------------------")
        print("")

    # 设置当前进程的 CUDA 设备为 dist.get_rank() % 8 所得的 GPU
    torch.cuda.set_device(dist.get_rank() % 8)
    # 创建 Torch 设备对象，指定为当前进程的 CUDA 设备
    device = torch.device("cuda:%d" % (dist.get_rank() % 8))

    # 初始化一个空的性能基准测试列表
    benchmarks = []
    # 如果有指定模型，则添加对应模型的性能基准测试
    if args.model:
        benchmarks.append(
            TorchvisionBenchmark(
                device=device,
                distributed_backend=args.distributed_backend,
                bucket_size=args.bucket_size,
                model=args.model,
            )
        )
    else:
        # 否则，对于预定义的模型列表，分别添加性能基准测试
        for model in ["resnet50", "resnet101", "resnext50_32x4d", "resnext101_32x8d"]:
            benchmarks.append(
                TorchvisionBenchmark(
                    device=device,
                    distributed_backend=args.distributed_backend,
                    bucket_size=args.bucket_size,
                    model=model,
                )
            )

    # 初始化一个空的基准测试结果列表
    benchmark_results = []
    # 遍历每个性能基准测试对象
    for benchmark in benchmarks:
        # 如果当前进程的 rank 是 0，输出当前基准测试的信息
        if args.rank == 0:
            print(f"\nBenchmark: {str(benchmark)}")
        # 执行性能基准测试，获取其结果
        result = sweep(benchmark)
        # 将基准测试结果以字典形式添加到结果列表中
        benchmark_results.append(
            {
                "model": benchmark.model,
                "batch_size": benchmark.batch_size,
                "result": result,
            }
        )

    # 如果当前进程的 rank 是 0，并且指定了输出 JSON 文件
    # 则创建报告对象，包括 PyTorch 版本、CUDA 版本、分布式后端、桶大小和基准测试结果
    if args.rank == 0 and args.json:
        report = {
            "pytorch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "distributed_backend": args.distributed_backend,
            "bucket_size": args.bucket_size,
            "benchmark_results": benchmark_results,
        }
        # 将报告对象写入指定的 JSON 文件
        with open(args.json, "w") as f:
            json.dump(report, f)
# 如果当前脚本被直接运行而非被导入为模块，则执行 main() 函数
if __name__ == "__main__":
    main()
```