# `.\pytorch\benchmarks\distributed\rpc\parameter_server\launcher.py`

```py
# 导入需要的库
import argparse  # 解析命令行参数的库
import json  # 处理 JSON 格式数据的库
import os  # 提供与操作系统交互的功能
from pathlib import Path  # 处理文件路径的库

# 导入自定义模块
from data import data_map  # 导入数据映射模块
from metrics.ProcessedMetricsPrinter import ProcessedMetricsPrinter  # 导入处理后的指标打印类
from models import model_map  # 导入模型映射模块
from server import server_map  # 导入服务器映射模块
from trainer import (  # 导入训练器相关的模块
    criterion_map,  # 导入损失函数映射
    ddp_hook_map,  # 导入分布式数据并行的钩子映射
    ddp_model_map,  # 导入分布式数据并行的模型映射
    hook_state_map,  # 导入钩子状态映射
    iteration_step_map,  # 导入迭代步骤映射
    preprocess_data_map,  # 导入数据预处理映射
    trainer_map,  # 导入训练器映射
)

# 导入 PyTorch 库
import torch  # PyTorch 主库
import torch.distributed as c10d  # PyTorch 分布式库
import torch.distributed.rpc as rpc  # PyTorch 分布式 RPC 库
import torch.multiprocessing as mp  # 多进程处理库
from torch.distributed.rpc import TensorPipeRpcBackendOptions  # 导入 TensorPipe RPC 后端选项
from torch.futures import wait_all  # 等待所有任务完成的库
from torch.utils.data import DataLoader  # PyTorch 数据加载器

def get_name(rank, args):
    r"""
    A function that gets the name for the rank
    argument
    Args:
        rank (int): process number in the world
        args (parser): benchmark configurations
    """
    t_count = args.ntrainer + args.ncudatrainer  # 计算训练器和 CUDA 训练器的总数
    s_count = args.nserver + args.ncudaserver  # 计算服务器和 CUDA 服务器的总数
    if rank < t_count:
        return f"trainer{rank}"  # 返回训练器的名称
    elif rank < (t_count + s_count):
        return f"server{rank}"  # 返回服务器的名称
    else:
        return "master"  # 返回主节点的名称


def get_server_rank(args, rank):
    r"""
    A function that gets the server rank for
    the rank argument.
    Args:
        args (parser): benchmark configurations
        rank (int): trainer rank
    """
    s_offset = args.ntrainer + args.ncudatrainer  # 计算服务器的偏移量
    tps = args.ntrainer // args.nserver  # 计算每个服务器上的训练器数量
    return rank // tps + s_offset  # 返回服务器的排名


def get_cuda_server_rank(args, rank):
    r"""
    A function that gets the cudaserver rank for
    the rank argument.
    Args:
        args (parser): benchmark configurations
        rank (int): trainer rank
    """
    s_offset = args.ntrainer + args.ncudatrainer + args.nserver  # 计算 CUDA 服务器的偏移量
    t_index = rank - args.ntrainer  # 计算在 CUDA 训练器中的索引
    ctps = args.ncudatrainer // args.ncudaserver  # 计算每个 CUDA 服务器上的训练器数量
    return t_index // ctps + s_offset  # 返回 CUDA 服务器的排名


def get_server_rref(server_rank, args, extra_args):
    r"""
    A function that creates a RRef to the server.
    Args:
        server_rank (int): process number in the world
        args (parser): benchmark configurations
        extra_args (dict): configurations added by the user
    """
    server = server_map[args.server]  # 获取服务器映射中指定名称的服务器对象
    name = get_name(server_rank, args)  # 根据服务器排名和参数获取服务器名称
    if extra_args is not None:
        server_args = extra_args.values()  # 获取额外参数的值
    else:
        server_args = []  # 如果额外参数为空，则初始化为空列表
    if server_rank >= args.ntrainer + args.ncudatrainer + args.nserver:
        trainer_count = args.ncudatrainer / args.ncudaserver  # 计算每个 CUDA 服务器上的训练器数量
        use_cuda_rpc = True  # 设置使用 CUDA RPC
    else:
        trainer_count = args.ntrainer / args.nserver  # 计算每个服务器上的训练器数量
        use_cuda_rpc = False  # 设置不使用 CUDA RPC
    return rpc.remote(
        name,
        server,
        args=(
            server_rank,
            trainer_count,
            use_cuda_rpc,
            *server_args,
        ),
    )


def run_trainer(args, extra_args, data, rank, server_rref):
    r"""
    A function that runs obtains a trainer instance and calls
    the train method.
    Args:
        args (parser): benchmark configurations
        extra_args (dict): configurations added by the user
        data: training data
        rank (int): process number in the world
        server_rref: remote reference to the server
    """
    Args:
        args (parser): benchmark configurations
        extra_args (dict): configurations added by the user
        data (list): training samples
        rank (int): process number in the world
        server_rref (dict): a dictionary containing server RRefs
    """
    # 根据 args.trainer 选择合适的 Trainer 类
    trainer_class = trainer_map[args.trainer]
    
    # 如果 extra_args 不为空，则获取其中的值作为 trainer_args
    if extra_args is not None:
        trainer_args = extra_args.values()
    else:
        trainer_args = []
    
    # 计算 trainer 的数量，包括 ntrainer 和 ncudatrainer
    trainer_count = args.ntrainer + args.ncudatrainer
    
    # 创建一个基于文件存储的分布式存储器
    store = c10d.FileStore(args.filestore, trainer_count)
    
    # 根据 backend 类型选择相应的进程组通信方式
    if args.backend == "gloo":
        process_group = c10d.ProcessGroupGloo(store, rank, trainer_count)
    elif args.backend == "nccl":
        process_group = c10d.ProcessGroupNCCL(store, rank, trainer_count)
    elif args.backend == "multi":
        # 对于多后端，选择 NCCL 作为进程组通信方式
        process_group = c10d.ProcessGroupNCCL(store, rank, trainer_count)
        # 如果当前未初始化，使用 Gloo 后端进行初始化
        if c10d.is_initialized() is False:
            c10d.init_process_group(backend="gloo", rank=rank, world_size=trainer_count)
    
    # 载入模型
    model = load_model(args)
    
    # 获取预处理数据的函数
    preprocess_data = preprocess_data_map[args.preprocess_data]
    
    # 获取创建损失函数的函数
    create_criterion = criterion_map[args.create_criterion]
    
    # 获取创建 DDP 模型的函数
    create_ddp_model = ddp_model_map[args.create_ddp_model]
    
    # 获取迭代步骤处理的函数
    iteration_step = iteration_step_map[args.iteration_step]
    
    # 获取钩子状态的类
    hook_state_class = hook_state_map[args.hook_state]
    
    # 获取分布式数据并行钩子的类
    hook = ddp_hook_map[args.ddp_hook]
    
    # 检查是否是 cudatrainer
    use_cuda_rpc = rank >= args.ntrainer
    
    # 创建 Trainer 实例
    trainer = trainer_class(
        process_group,
        use_cuda_rpc,
        server_rref,
        args.backend,
        args.epochs,
        preprocess_data,
        create_criterion,
        create_ddp_model,
        hook_state_class,
        hook,
        iteration_step,
        *trainer_args,
    )
    
    # 使用 Trainer 对象进行训练
    trainer.train(model, data)
    
    # 获取训练过程中的度量指标
    metrics = trainer.get_metrics()
    
    # 返回包含进程号和度量指标的列表
    return [rank, metrics]
# 调用训练器函数，启动每个训练器的异步 RPC 请求
def call_trainers(args, extra_args, train_data, server_rrefs):
    # 存储每个异步 RPC 请求的 future 对象列表
    futs = []
    # 遍历所有训练器的排名
    for trainer_rank in range(0, args.ntrainer + args.ncudatrainer):
        # 获取当前训练器的名称
        trainer_name = get_name(trainer_rank, args)
        # 初始化服务器远程引用对象为 None
        server_rref = None
        # 如果有服务器远程引用对象存在
        if server_rrefs:
            # 根据训练器排名确定服务器排名
            if trainer_rank >= args.ntrainer:
                server_rank = get_cuda_server_rank(args, trainer_rank)
            else:
                server_rank = get_server_rank(args, trainer_rank)
            # 获取对应服务器的远程引用对象
            server_rref = server_rrefs[server_rank]
        # 发起异步 RPC 调用并将 future 对象添加到列表中
        fut = rpc.rpc_async(
            trainer_name,
            run_trainer,
            args=(
                args,
                extra_args,
                train_data[trainer_rank],
                trainer_rank,
                server_rref,
            ),
            timeout=args.rpc_timeout,
        )
        futs.append(fut)
    # 返回所有异步 RPC 调用的 future 对象列表
    return futs


# 运行训练算法的函数，用于 RPC 的预热和服务器状态重置
def benchmark_warmup(args, extra_args, data, server_rrefs):
    # 调用训练器函数，获取异步 RPC 调用的 future 对象列表
    futs = call_trainers(args, extra_args, data, server_rrefs)
    # 等待所有 future 对象完成
    wait_all(futs)
    # 重置所有服务器的状态
    for server_rref in server_rrefs.values():
        server_rref.rpc_sync().reset_state(server_rref)
    # 输出完成信息
    print("benchmark warmup done\n")


# 将一个列表分割成 n 个子列表的函数
def split_list(arr, n):
    # 返回分割后的子列表列表
    return [arr[i::n] for i in range(n)]


# 获取服务器的指标数据的函数，通过远程调用获取
def get_server_metrics(server_rrefs):
    # 存储排名和指标数据的列表
    rank_metrics = []
    # 遍历所有服务器的远程引用对象
    for rank, server_rref in server_rrefs.items():
        # 获取远程服务器的指标数据
        metrics = server_rref.rpc_sync().get_metrics(server_rref)
        # 将排名和指标数据组成的列表添加到结果列表中
        rank_metrics.append([rank, metrics])
    # 返回所有服务器的排名和指标数据组成的列表
    return rank_metrics


# 运行主进程的函数，获取初始化服务器的远程引用、分割数据、运行训练器并打印指标
def run_master(rank, data, args, extra_configs, rpc_backend_options):
    # 此函数的具体功能已超出代码片段，未完整提供，无法进行详细注释
    # 计算整个训练集的节点数量
    world_size = args.ntrainer + args.ncudatrainer + args.nserver + args.ncudaserver + 1
    # 初始化 RPC，设置当前节点的名称、排名、总节点数和 RPC 后端选项
    rpc.init_rpc(
        get_name(rank, args),
        rank=rank,
        world_size=world_size,
        rpc_backend_options=rpc_backend_options,
    )
    # 存储远程服务器的远程引用的字典
    server_rrefs = {}
    # 为所有服务器节点创建远程引用
    for i in range(args.ntrainer + args.ncudatrainer, world_size - 1):
        server_rrefs[i] = get_server_rref(i, args, extra_configs["server_config"])
    # 将训练数据分成多个部分，每个部分用于不同的训练节点
    train_data = split_list(
        list(DataLoader(data, batch_size=args.batch_size)),
        args.ntrainer + args.ncudatrainer,
    )

    # 执行基准测试的预热运行
    benchmark_warmup(args, extra_configs["trainer_config"], train_data, server_rrefs)
    # 执行训练过程
    trainer_futs = call_trainers(
        args, extra_configs["trainer_config"], train_data, server_rrefs
    )
    # 收集和打印训练节点的指标数据
    metrics_printer = ProcessedMetricsPrinter()
    rank_metrics_list = wait_all(trainer_futs)
    metrics_printer.print_metrics("trainer", rank_metrics_list)
    # 获取和打印服务器节点的指标数据
    rank_metrics_list = get_server_metrics(server_rrefs)
    metrics_printer.print_metrics("server", rank_metrics_list)
# 运行基准测试的函数，根据参数 rank、args 和 data 运行基准测试
def run_benchmark(rank, args, data):
    r"""
    A function that runs the benchmark.
    Args:
        rank (int): process number in the world
        args (parser): configuration args
        data (list): training samples
    """

    # 载入额外的配置信息
    config = load_extra_configs(args)

    # 设置随机种子
    torch.manual_seed(args.torch_seed)
    torch.cuda.manual_seed_all(args.cuda_seed)
    # 开启 cuDNN 的性能优化和确定性选项
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    # 计算总的进程数
    world_size = args.ntrainer + args.ncudatrainer + args.nserver + args.ncudaserver + 1
    # 设置环境变量
    os.environ["MASTER_ADDR"] = args.master_addr
    os.environ["MASTER_PORT"] = args.master_port
    # 设置 TensorPipe RPC 后端选项
    rpc_backend_options = TensorPipeRpcBackendOptions(rpc_timeout=args.rpc_timeout)
    
    if rank == world_size - 1:
        # 如果是主进程，调用 run_master 函数
        # master = [ntrainer + ncudatrainer + nserver + ncudaserver, ntrainer + ncudatrainer + nserver + ncudaserver]
        run_master(rank, data, args, config, rpc_backend_options)
    elif rank >= args.ntrainer + args.ncudatrainer:
        # 如果是参数服务器进程，初始化 RPC
        # parameter_servers = [ntrainer + ncudatrainer, ntrainer + ncudatrainer + nserver + ncudaserver)
        rpc.init_rpc(
            get_name(rank, args),
            rank=rank,
            world_size=world_size,
            rpc_backend_options=rpc_backend_options,
        )
    else:
        # 如果是训练进程，初始化 RPC
        # trainers = [0, ntrainer + ncudatrainer)
        if rank >= args.ntrainer:
            # 对于 GPU 训练进程，设置设备映射
            server_rank = get_cuda_server_rank(args, rank)
            server_name = get_name(server_rank, args)
            rpc_backend_options.set_device_map(server_name, {rank: server_rank})
        # 获取进程名并初始化 RPC
        trainer_name = get_name(rank, args)
        rpc.init_rpc(
            trainer_name,
            rank=rank,
            world_size=world_size,
            rpc_backend_options=rpc_backend_options,
        )
    
    # 关闭 RPC
    rpc.shutdown()


# 从文件中加载 JSON 格式的配置信息的函数
def get_json_config(file_name: str, id: str):
    r"""
    A function that loads a json configuration from a file.
    Args:
        file_name (str): name of configuration file to load
        id (str): configuration that will be loaded
    """
    with open(Path(__file__).parent / file_name) as f:
        # 从文件中加载 JSON 数据，并获取特定 id 的配置
        json_config = json.load(f)[id]
    return json_config


# 加载额外配置信息的函数，返回一个字典，包含 trainer_config 和 server_config 两个键，默认值为 None
def load_extra_configs(args):
    r"""
    A function that creates a dictionary that contains any extra configurations
    set by the user. The dictionary will contain two keys trainer_config and
    server_config, with default values None.
    Args:
        args (parser): launcher configurations
    """
    # 获取训练配置文件路径和服务器配置文件路径
    trainer_config_file = args.trainer_config_path
    server_config_file = args.server_config_path
    configurations = {"trainer_config": None, "server_config": None}
    # 如果指定了训练配置文件并且 args.trainer 不为空，加载 trainer_config
    if args.trainer is not None and trainer_config_file is not None:
        configurations["trainer_config"] = get_json_config(
            trainer_config_file, args.trainer
        )
    # 如果指定了服务器配置文件并且 args.server 不为空，加载 server_config
    if args.server is not None and server_config_file is not None:
        configurations["server_config"] = get_json_config(
            server_config_file, args.server
        )
    return configurations



    # 返回变量 configurations 所引用的数据结构
    return configurations
# 创建一个数据类的实例的函数，根据给定的配置文件路径和参数。
def load_data(args):
    # 获取数据配置文件的路径
    data_config_file = args.data_config_path
    # 使用指定的配置文件和数据类型，获取JSON格式的配置信息
    data_config = get_json_config(data_config_file, args.data)
    # 根据配置信息中的数据类，获取对应的数据类
    data_class = data_map[data_config["data_class"]]
    # 使用配置信息中的具体配置参数，创建数据类的实例并返回
    return data_class(**data_config["configurations"])


# 创建一个模型类的实例的函数，根据给定的配置文件路径和参数。
def load_model(args):
    # 获取模型配置文件的路径
    model_config_file = args.model_config_path
    # 使用指定的配置文件和模型类型，获取JSON格式的配置信息
    model_config = get_json_config(model_config_file, args.model)
    # 根据配置信息中的模型类，获取对应的模型类
    model_class = model_map[model_config["model_class"]]
    # 使用配置信息中的具体配置参数，创建模型类的实例并返回
    return model_class(**model_config["configurations"])


# 主函数，创建多个进程来运行基准测试。
def main(args):
    # 检查CPU和RPC训练器的数量设置
    if args.ntrainer > 0 and args.ncudatrainer > 0:
        # 如果同时有CPU训练器和CUDA训练器，要求至少有一个CPU服务器和一个CUDA服务器
        assert args.nserver > 0 and args.ncudaserver > 0
    if args.nserver > 0:
        # 如果有CPU服务器，要求至少有一个CPU训练器，并且CPU训练器数量应能整除CPU服务器数量
        assert args.ntrainer > 0
        assert args.ntrainer % args.nserver == 0
    if args.ncudaserver > 0:
        # 如果有CUDA服务器，要求至少有一个CUDA训练器，并且CUDA训练器数量应能整除CUDA服务器数量
        assert args.ncudatrainer > 0
        assert args.ncudatrainer % args.ncudaserver == 0

    # 计算所有角色（训练器、CUDA训练器、服务器和CUDA服务器）的总数
    world_size = args.ntrainer + args.ncudatrainer + args.nserver + args.ncudaserver + 1

    # 使用给定的参数加载数据
    data = load_data(args)

    # 使用多进程来执行基准测试函数，传入参数和数据，进程数量为world_size，等待所有进程结束
    mp.spawn(
        run_benchmark,
        args=(
            args,
            data,
        ),
        nprocs=world_size,
        join=True,
    )


if __name__ == "__main__":
    # 创建参数解析器
    parser = argparse.ArgumentParser(description="RPC server Benchmark")
    # 添加命令行参数：主服务器地址
    parser.add_argument(
        "--master-addr",
        "--master_addr",
        type=str,
        help="IP address of the machine that will host the process with rank 0",
    )
    # 添加命令行参数：主服务器端口
    parser.add_argument(
        "--master-port",
        "--master_port",
        type=str,
        help="A free port on the machine that will host the process with rank 0",
    )
    # 添加命令行参数：训练器的键，用于获取基准测试运行的训练器类
    parser.add_argument(
        "--trainer",
        type=str,
        help="trainer map key to get trainer class for benchmark run",
    )
    # 添加命令行参数：训练器数量，用于基准测试运行
    parser.add_argument("--ntrainer", type=int, help="trainer count for benchmark run")
    # 添加命令行参数：CUDA训练器数量，用于基准测试运行
    parser.add_argument(
        "--ncudatrainer", type=int, help="cudatrainer count for benchmark run"
    )
    # 添加命令行参数：文件存储位置，用于进程组的文件存储
    parser.add_argument(
        "--filestore", type=str, help="filestore location for process group"
    )
    # 添加命令行参数：服务器的键，用于获取基准测试运行的服务器类
    parser.add_argument(
        "--server",
        type=str,
        help="server map key to get trainer class for benchmark run",
    )
    # 添加命令行参数：服务器数量，用于基准测试运行
    parser.add_argument("--nserver", type=int, help="server count for benchmark run")
    # 添加命令行参数：CUDA服务器数量，用于基准测试运行
    parser.add_argument(
        "--ncudaserver", type=int, help="cudaserver count for benchmark run"
    )
    # 添加命令行参数：RPC超时时间，用于RPC调用的超时设置（秒）
    parser.add_argument(
        "--rpc-timeout",
        "--rpc_timeout",
        type=int,
        help="timeout in seconds to use for RPC",
    )
    parser.add_argument(
        "--backend",
        type=str,
        help="distributed communication backend to use for benchmark run",
    )
    # 添加命令行参数：指定分布式通信后端，用于基准测试运行

    parser.add_argument("--epochs", type=int, help="epoch count for training")
    # 添加命令行参数：指定训练的 epoch 数量

    parser.add_argument(
        "--batch-size",
        "--batch_size",
        type=int,
        help="number of training examples used in one iteration",
    )
    # 添加命令行参数：指定每次迭代中使用的训练样本数量

    parser.add_argument("--data", type=str, help="id for data configuration")
    # 添加命令行参数：指定数据配置的标识符

    parser.add_argument("--model", type=str, help="id for model configuration")
    # 添加命令行参数：指定模型配置的标识符

    parser.add_argument(
        "--data-config-path",
        "--data_config_path",
        type=str,
        help="path to data configuration file",
    )
    # 添加命令行参数：指定数据配置文件的路径

    parser.add_argument(
        "--model-config-path",
        "--model_config_path",
        type=str,
        help="path to model configuration file",
    )
    # 添加命令行参数：指定模型配置文件的路径

    parser.add_argument(
        "--server-config-path",
        "--server_config_path",
        type=str,
        help="path to server configuration file",
    )
    # 添加命令行参数：指定服务器配置文件的路径

    parser.add_argument(
        "--trainer-config-path",
        "--trainer_config_path",
        type=str,
        help="path to trainer configuration file",
    )
    # 添加命令行参数：指定训练器配置文件的路径

    parser.add_argument(
        "--torch-seed",
        "--torch_seed",
        type=int,
        help="seed for generating random numbers to a non-deterministic random number",
    )
    # 添加命令行参数：指定用于生成非确定性随机数的种子

    parser.add_argument(
        "--cuda-seed",
        "--cuda_seed",
        type=int,
        help="seed for generating random numbers to a random number for the current GPU",
    )
    # 添加命令行参数：指定用于当前 GPU 生成随机数的种子

    parser.add_argument(
        "--preprocess-data",
        "--preprocess_data",
        type=str,
        help="this function will be used to preprocess data before training",
    )
    # 添加命令行参数：指定用于在训练之前预处理数据的函数

    parser.add_argument(
        "--create-criterion",
        "--create_criterion",
        type=str,
        help="this function will be used to create the criterion used for model loss calculation",
    )
    # 添加命令行参数：指定用于创建模型损失计算所使用的标准的函数

    parser.add_argument(
        "--create-ddp-model",
        "--create_ddp_model",
        type=str,
        help="this function will be used to create the ddp model used during training",
    )
    # 添加命令行参数：指定用于在训练期间创建 DDP 模型的函数

    parser.add_argument(
        "--hook-state",
        "--hook_state",
        type=str,
        help="this will be the state class used when registering the ddp communication hook",
    )
    # 添加命令行参数：指定注册 DDP 通信钩子时使用的状态类

    parser.add_argument(
        "--ddp-hook",
        "--ddp_hook",
        type=str,
        default="allreduce_hook",
        help="ddp communication hook",
    )
    # 添加命令行参数：指定 DDP 通信钩子的名称，默认为 "allreduce_hook"

    parser.add_argument(
        "--iteration-step",
        "--iteration_step",
        type=str,
        help="this will be the function called for each iteration of training",
    )
    # 添加命令行参数：指定在每次训练迭代中调用的函数

    args = parser.parse_args()
    # 解析命令行参数并将其存储在 args 变量中

    print(f"{args}\n")
    # 打印解析后的参数信息

    main(args)
    # 调用主函数 main，并传递解析后的参数 args
```