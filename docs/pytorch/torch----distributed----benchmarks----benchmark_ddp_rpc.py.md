# `.\pytorch\torch\distributed\benchmarks\benchmark_ddp_rpc.py`

```py
# mypy: allow-untyped-defs
# 引入必要的库和模块
import argparse  # 用于命令行参数解析
import io  # 用于字节流操作
import os  # 提供操作系统相关的功能
import random  # 提供生成随机数的功能
import shlex  # 用于解析 shell 命令字符串
import subprocess  # 用于执行外部命令
import time  # 提供时间相关的功能

import numpy as np  # 数组和数值计算库

import torch  # PyTorch 深度学习框架
import torch.distributed as dist  # PyTorch 分布式通信模块
import torch.distributed.autograd as dist_autograd  # 分布式自动求导模块
import torch.distributed.rpc as rpc  # 分布式远程过程调用模块
import torch.multiprocessing as mp  # 多进程管理模块
import torch.nn as nn  # 神经网络模块
import torch.optim as optim  # PyTorch 优化器模块
from torch.distributed.optim import DistributedOptimizer  # 分布式优化器
from torch.distributed.rpc import RRef, TensorPipeRpcBackendOptions  # 远程引用和 TensorPipe RPC 后端选项
from torch.distributed.rpc.backend_registry import BackendType  # 后端类型
from torch.nn.parallel import DistributedDataParallel as DDP  # 分布式数据并行模块


# Config
NUM_TRAINERS = 8  # 训练器的数量
NUM_PS = 8  # 参数服务器的数量

NUM_EMBEDDINGS = 300  # 嵌入的数量
EMBEDDING_DIM = 64  # 嵌入的维度

WARMUP_CYCLES = 5  # 热身循环的次数


class HybridModel(torch.nn.Module):
    r"""
    模型由稀疏部分和稠密部分组成。

    稠密部分是一个 nn.Linear 模块，使用 DistributedDataParallel 在所有训练器之间复制。
    稀疏部分包含在多个参数服务器上的 nn.EmbeddingBags。

    模型持有嵌入表在参数服务器上的远程引用。
    """

    def __init__(self, emb_rref_list, device):
        super().__init__()
        self.emb_rref_list = emb_rref_list
        fc1 = torch.nn.Linear(512, 256)  # 第一个全连接层
        fc2 = torch.nn.Linear(256, 128)  # 第二个全连接层
        relu = torch.nn.ReLU()  # ReLU 激活函数
        fc3 = torch.nn.Linear(128, 64)  # 第三个全连接层
        fc4 = torch.nn.Linear(64, 32)  # 第四个全连接层
        fc5 = torch.nn.Linear(32, 8)  # 第五个全连接层
        sec = nn.Sequential(fc1, fc2, relu, fc3, fc4, fc5)  # 创建网络的序列化层
        self.ddp = DDP(sec.to(device), device_ids=[device])  # 使用分布式数据并行封装序列化层
        self.device = device  # 设备

    def forward(self, indices, offsets):
        emb_lookups = []

        for emb_rref in self.emb_rref_list:
            emb_lookups.append(
                emb_rref.rpc_sync().forward(
                    indices, offsets
                )  # 调用远程过程：获取嵌入向量
            )
        emb_lookups_cat = torch.cat(emb_lookups, dim=1)  # 沿着指定维度拼接嵌入向量

        # 确保组合后的参数服务器维度总是大于或等于全连接层的输入
        assert NUM_PS * EMBEDDING_DIM >= 512
        dim_normalizer = int(NUM_PS * EMBEDDING_DIM / 512)  # 维度标准化因子
        emb_lookups_reshaped = emb_lookups_cat.reshape(
            [emb_lookups_cat.shape[0] * dim_normalizer, 512]
        )  # 重新塑造嵌入向量的形状

        return self.ddp(emb_lookups_reshaped)  # 返回分布式数据并行的结果


def _retrieve_embedding_parameters(emb_rref):
    return [RRef(p) for p in emb_rref.local_value().parameters()]  # 检索嵌入参数


def _print_header():
    _print_cont("\n")  # 打印换行
    _print_cont("%10s" % "")  # 打印空格
    for p in [50, 75, 90, 95]:
        _print_cont("%14s%10s" % ("sec/epoch", "epoch/sec"))  # 打印标题行
    _print_cont("\n")  # 打印换行


def _print_benchmark(prefix, nelem, measurements):
    measurements = sorted(measurements)  # 对测量结果进行排序
    _print_cont("%8s:" % prefix)  # 打印前缀
    for p in [50, 75, 90, 95]:
        v = np.percentile(measurements, p)  # 计算百分位数
        _print_cont("  p%02d:  %1.3fs  %6d/s" % (p, v, nelem / v))  # 打印百分位数和速率
    _print_cont("\n")  # 打印换行


def _print_cont(msg):
    # 打印消息，不换行，并立即刷新输出
    print(msg, end="", flush=True)
# 定义一个函数，用于执行给定的命令，并捕获其输出
def _run_printable(cmd):
    # 使用 shlex 将命令字符串解析为适合 subprocess 的列表格式，并执行命令
    proc = subprocess.run(shlex.split(cmd), capture_output=True, check=False)  # type: ignore[call-overload]
    # 断言命令执行成功，即返回码为 0
    assert proc.returncode == 0

    # 创建一个字节流缓冲区
    buffer = io.BytesIO()
    # 将命令执行的标准输出内容按 UTF-8 解码后保存到字节流缓冲区中
    torch.save(proc.stdout.decode("utf-8"), buffer)
    # 将缓冲区中的数据转换为 ByteTensor 类型的张量，并以列表形式存储
    input_tensor = torch.ByteTensor(list(buffer.getvalue()))
    # 创建一个 IntTensor 类型的张量，包含输入张量的长度
    input_length = torch.IntTensor([input_tensor.size(0)])

    # 创建一个空列表用于存储输出结果
    output = []
    # 将包含输入张量的字节流缓冲区加载为 Torch 对象，并添加到输出列表中
    output.append(torch.load(buffer))
    # 返回输出列表
    return output


# 定义一个函数，用于执行训练过程，接受分布式参数和排名作为输入
def _run_trainer(emb_rref_list, rank):
    r"""
    每个训练器执行前向传播，包括在 8 个参数服务器上进行嵌入查找，
    并在本地运行 nn.Linear。

    在反向传播过程中，DDP 负责聚合稠密部分（nn.Linear）的梯度，
    分布式自动求导确保梯度更新传播到参数服务器。
    """
    # 设置模型
    model = HybridModel(emb_rref_list, rank)

    # 为分布式优化器准备模型参数的远程引用
    model_parameter_rrefs = []
    for ind, emb_rref in enumerate(emb_rref_list):
        ps_name = f"ps{ind}"
        # 使用 RPC 同步调用从参数服务器获取嵌入表的参数，并扩展到模型参数远程引用列表中
        model_parameter_rrefs.extend(
            rpc.rpc_sync(ps_name, _retrieve_embedding_parameters, args=(emb_rref,))
        )

    # model.parameters() 只包括本地参数，需要将其也加入模型参数远程引用列表中
    for param in model.parameters():
        model_parameter_rrefs.append(RRef(param))

    # 设置分布式优化器，使用 SGD 作为优化器，学习率为 0.05
    opt = DistributedOptimizer(optim.SGD, model_parameter_rrefs, lr=0.05)

    # 设置交叉熵损失函数
    criterion = torch.nn.CrossEntropyLoss()

    # 定义获取下一个批次数据的函数，接受排名作为参数
    def get_next_batch(rank):
        for _ in range(10):
            # 随机生成一个数据集的大小
            num_indices = random.randint(20, 50)
            # 生成一个包含随机索引的 LongTensor
            indices = torch.LongTensor(num_indices).random_(0, NUM_EMBEDDINGS)

            # 生成偏移量
            offsets = []
            start = 0
            batch_size = 0

            while start < num_indices:
                offsets.append(start)
                start += random.randint(1, 10)
                batch_size += 1

            offsets_tensor = torch.LongTensor(offsets)
            # 生成随机目标张量，将其移动到指定的 GPU 设备上
            target = torch.LongTensor(batch_size).random_(8).cuda(rank)

            yield indices, offsets_tensor, target

    # 创建一个空列表，用于存储测量结果
    measurements = []
    # 在训练过程中包含热身周期
    # 执行包括预热周期在内的100个循环迭代
    for epoch in range(100 + WARMUP_CYCLES):
        # 记录每次迭代的开始时间
        start = time.time()
        # 批量大小初始化为0
        batch_size = 0

        # 创建分布式自动求导上下文
        for indices, offsets, target in get_next_batch(rank):
            # 计算当前批次的总样本数
            batch_size += len(target)

            # 在分布式自动求导上下文中执行以下操作
            with dist_autograd.context() as context_id:
                # 将输入索引和偏移量传递给模型进行计算
                output = model(indices, offsets)
                # 计算模型输出与目标之间的损失
                loss = criterion(output, target)

                # 执行分布式反向传播
                dist_autograd.backward(context_id, [loss])

                # 执行分布式优化器，梯度传播到参数服务器
                opt.step(context_id)

                # 每次迭代创建不同的分布式自动求导上下文，不需要手动清零梯度

        # 记录每次迭代的运行时间
        measurements.append(time.time() - start)
        # 打印当前迭代周期的训练完成信息（已注释掉的代码）

    # 丢弃预热周期的测量结果
    measurements = measurements[WARMUP_CYCLES:]
    # 返回进程的排名、测量时间列表和批量大小（类型为忽略未定义，针对类型检查的提示）
    return rank, measurements, batch_size  # type: ignore[possibly-undefined]
def run_worker(rank, world_size):
    r"""
    Initialize RPC, calls the function, and shuts down RPC.
    """
    # 使用不同的端口号在 TCP 的 init_method 中初始化 init_rpc 和 init_process_group，以避免端口冲突。
    rpc_backend_options = TensorPipeRpcBackendOptions()
    rpc_backend_options.init_method = "tcp://localhost:29500"

    # 如果 rank 等于 NUM_TRAINERS + NUM_PS，则执行以下代码
    if rank == (NUM_TRAINERS + NUM_PS):
        # 初始化 master 节点的 RPC
        rpc.init_rpc(
            "master",
            rank=rank,
            backend=BackendType.TENSORPIPE,  # 指定后端为 TensorPipe
            world_size=world_size,
        )

        # 在参数服务器上构建 Embedding 表
        emb_rref_list = []
        index = 0
        while index < NUM_PS:
            ps_name = f"ps{index}"
            # 在远程参数服务器上创建 EmbeddingBag 对象
            emb_rref = rpc.remote(
                ps_name,
                torch.nn.EmbeddingBag,
                args=(NUM_EMBEDDINGS, EMBEDDING_DIM),
                kwargs={"mode": "sum"},
            )
            emb_rref_list.append(emb_rref)
            index += 1

        # 在训练节点上运行训练循环
        futs = []
        for trainer_rank in range(NUM_TRAINERS):
            trainer_name = f"trainer{trainer_rank}"
            # 异步调用训练节点上的 _run_trainer 函数
            fut = rpc.rpc_async(
                trainer_name, _run_trainer, args=(emb_rref_list, trainer_rank)
            )
            futs.append(fut)

        # 打印表头信息
        _print_header()

        measurements_all_trainers = []
        batch_size_all_trainers = 0
        # 等待所有训练完成
        for fut in futs:
            rank, measurements, batch_size = fut.wait()
            # 打印每个训练节点的性能指标
            _print_benchmark(f"Trainer{rank}", batch_size, measurements)
            batch_size_all_trainers += batch_size
            measurements_all_trainers.append(measurements)

        # 打印所有训练节点的性能指标总和
        _print_benchmark("All", batch_size_all_trainers, measurements_all_trainers)

    # 如果 rank 在 0 到 NUM_PS 之间，则执行以下代码
    elif rank >= 0 and rank < NUM_PS:
        # 初始化分布式数据并行训练节点的进程组
        dist.init_process_group(
            backend=dist.Backend.GLOO,
            rank=rank,
            world_size=NUM_TRAINERS,
            init_method="tcp://localhost:29501",
        )

        # 初始化训练节点的 RPC，训练节点仅等待来自 master 节点的 RPC 调用
        trainer_name = f"trainer{rank}"
        rpc.init_rpc(
            trainer_name,
            rank=rank,
            world_size=world_size,
            rpc_backend_options=rpc_backend_options,
        )

    # 如果 rank 在 NUM_TRAINERS 到 NUM_TRAINERS + NUM_PS 之间，则执行以下代码
    elif rank >= NUM_TRAINERS and rank < NUM_TRAINERS + NUM_PS:
        ps_name = f"ps{rank - NUM_TRAINERS}"
        # 初始化参数服务器的 RPC
        rpc.init_rpc(
            ps_name,
            rank=rank,
            world_size=world_size,
            backend=BackendType.TENSORPIPE,  # 指定后端为 TensorPipe
            rpc_backend_options=rpc_backend_options,
        )
        # 参数服务器什么也不做
        pass

    # 阻塞直到所有 RPC 完成
    rpc.shutdown()


if __name__ == "__main__":
    """Initializing the distributed environment."""

    # 运行命令获取 GPU 拓扑结构信息
    output = _run_printable("nvidia-smi topo -m")
    # 打印分隔线
    print("-------------------------------------------")
    # 打印信息标题
    print("                  Info                     ")
    # 打印分隔线
    print("-------------------------------------------")
    print("")
    # 打印 PyTorch 版本信息
    print(f"* PyTorch version: {torch.__version__}")
    # 打印 CUDA 版本信息
    print(f"* CUDA version: {torch.version.cuda}")
    print("")
    # 打印 nvidia-smi 命令输出的 GPU 拓扑信息
    print("------------ nvidia-smi topo -m -----------")
    print("")
    print(output[0])  # 打印第一个输出结果
    print("-------------------------------------------")
    # 打印分布式训练的标题
    print("PyTorch Distributed Benchmark (DDP and RPC)")
    print("-------------------------------------------")

    # 解析命令行参数，用于配置分布式训练
    parser = argparse.ArgumentParser(description="PyTorch DDP and RPC Benchmark")
    parser.add_argument(
        "--master-addr", type=str, default="localhost", help="Address of master node."
    )
    parser.add_argument("--master-port", type=str, default="29500", help="Master port.")

    parser.add_argument(
        "--number-trainers",
        type=int,
        default=NUM_TRAINERS,
        help="Number of Trainer Nodes.",
    )
    parser.add_argument(
        "--number-ps", type=int, default=NUM_PS, help="Number of Parameter Servers."
    )
    parser.add_argument(
        "--number-embeddings",
        type=int,
        default=NUM_EMBEDDINGS,
        help="Number of test embeddings to be generated.",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=EMBEDDING_DIM,
        help="Number of embedding dimensions.",
    )
    parser.add_argument(
        "--warmup-cycles",
        type=int,
        default=WARMUP_CYCLES,
        help="Number of cycles to warm-up each process before running the benchmark.",
    )

    args = parser.parse_args()

    # 设置环境变量 MASTER_ADDR 和 MASTER_PORT
    os.environ["MASTER_ADDR"] = args.master_addr
    os.environ["MASTER_PORT"] = args.master_port

    # 更新全局变量以反映从命令行获取的参数值
    NUM_TRAINERS = args.number_trainers
    NUM_PS = args.number_ps
    NUM_EMBEDDINGS = args.number_embeddings
    EMBEDDING_DIM = args.embedding_dim
    WARMUP_CYCLES = args.warmup_cycles

    # 计算集群的总大小，包括训练节点、参数服务器和主节点
    world_size = NUM_TRAINERS + NUM_PS + 1  # Trainers + PS + Master
    # 使用多进程（mp）启动分布式训练的工作进程
    mp.spawn(run_worker, args=(world_size,), nprocs=world_size, join=True)
```