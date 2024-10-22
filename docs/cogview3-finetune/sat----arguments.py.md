# `.\cogview3-finetune\sat\arguments.py`

```
# 导入所需的库
import argparse  # 处理命令行参数
import os  # 与操作系统交互的功能
import torch  # 深度学习框架
import json  # JSON 数据处理
import warnings  # 发出警告的功能
import omegaconf  # 处理配置文件的库
from omegaconf import OmegaConf  # 导入 OmegaConf 用于配置文件
from sat.helpers import print_rank0  # 导入打印函数，用于仅在主进程中输出信息
from sat import mpu  # 导入分布式计算相关的功能
from sat.arguments import set_random_seed  # 设置随机种子的功能
from sat.arguments import add_training_args, add_evaluation_args, add_data_args  # 导入参数添加功能

# 定义函数以添加模型配置参数
def add_model_config_args(parser):
    """Model arguments"""  # 函数说明：模型参数

    # 创建一个新的参数组，命名为 "model"
    group = parser.add_argument_group("model", "model configuration")
    # 添加基本配置参数，接收多个字符串作为输入
    group.add_argument("--base", type=str, nargs="*", help="config for input and saving")
    # 添加模型并行大小参数，默认为 1，仅供专家使用
    group.add_argument(
        "--model-parallel-size", type=int, default=1, help="size of the model parallel. only use if you are an expert."
    )
    # 添加强制预训练标志
    group.add_argument("--force-pretrain", action="store_true")
    # 添加设备参数，默认为 -1
    group.add_argument("--device", type=int, default=-1)

    # 返回修改后的解析器
    return parser

# 定义函数以添加采样配置参数
def add_sampling_config_args(parser):
    """Sampling configurations"""  # 函数说明：采样配置

    # 创建一个新的参数组，命名为 "sampling"
    group = parser.add_argument_group("sampling", "Sampling Configurations")
    # 添加输入目录参数，默认为 None
    group.add_argument("--input-dir", type=str, default=None)
    # 添加输出目录参数，默认为 "samples"
    group.add_argument("--output-dir", type=str, default="samples")
    # 添加输入类型参数，默认为 "cli"
    group.add_argument("--input-type", type=str, default="cli")
    # 添加中继模型参数，默认为 False
    group.add_argument("--relay-model", type=bool, default=False)
    # 添加输入文件参数，默认为 "input.txt"
    group.add_argument("--input-file", type=str, default="input.txt")
    # 添加采样图像大小参数，默认为 1024
    group.add_argument("--sampling-image-size", type=int, default=1024)
    # 添加采样潜在维度参数，默认为 4
    group.add_argument("--sampling-latent-dim", type=int, default=4)
    # 添加采样 F 参数，默认为 8
    group.add_argument("--sampling-f", type=int, default=8)
    # 添加采样图像宽度参数，默认为 None
    group.add_argument("--sampling-image-size-x", type=int, default=None)
    # 添加采样图像高度参数，默认为 None
    group.add_argument("--sampling-image-size-y", type=int, default=None)
    # 添加 SDEdit 标志
    group.add_argument("--sdedit", action="store_true")
    # 添加 IP2P 标志
    group.add_argument("--ip2p", action="store_true")
    # 添加网格列数参数，默认为 1
    group.add_argument("--grid-num-columns", type=int, default=1)
    # 添加强制推理标志
    group.add_argument("--force-inference", action="store_true")

    # 返回修改后的解析器
    return parser

# 定义函数以添加额外配置参数
def add_additional_config_args(parser):
    # 创建一个新的参数组，命名为 "additional"
    group = parser.add_argument_group("additional", "Additional Configurations")
    # 添加多方面训练标志
    group.add_argument("--multiaspect-training", action="store_true")
    # 添加多方面形状参数，接收多个整数
    group.add_argument("--multiaspect-shapes", nargs="+", default=None, type=int)

    # 返回修改后的解析器
    return parser

# 定义函数以获取所有参数
def get_args(args_list=None, parser=None):
    """Parse all the args."""  # 函数说明：解析所有参数
    # 如果未提供解析器，则创建一个新的 ArgumentParser 实例
    if parser is None:
        parser = argparse.ArgumentParser(description="sat")
    else:
        # 确保提供的解析器是 ArgumentParser 的实例
        assert isinstance(parser, argparse.ArgumentParser)
    # 添加模型配置参数
    parser = add_model_config_args(parser)
    # 添加采样配置参数
    parser = add_sampling_config_args(parser)
    # 添加训练参数
    parser = add_training_args(parser)
    # 添加评估参数
    parser = add_evaluation_args(parser)
    # 添加数据参数
    parser = add_data_args(parser)
    # 添加额外配置参数
    parser = add_additional_config_args(parser)

    # 导入 DeepSpeed 库
    import deepspeed
    # 包含 DeepSpeed 配置参数
    parser = deepspeed.add_config_arguments(parser)

    # 解析提供的参数列表
    args = parser.parse_args(args_list)
    # 处理配置并转换为参数
    args = process_config_to_args(args)

    # 如果没有指定训练数据，则发出警告
    if not args.train_data:
        print_rank0("No training data specified", level="WARNING")
    # 确保 train_iters 和 epochs 仅有一个被设置
        assert (args.train_iters is None) or (args.epochs is None), "only one of train_iters and epochs should be set."
        # 如果两个参数都没有设置
        if args.train_iters is None and args.epochs is None:
            # 默认设置为 10000 次迭代
            args.train_iters = 10000  # default 10k iters
            # 打印警告信息，使用默认的迭代次数
            print_rank0("No train_iters (recommended) or epochs specified, use default 10k iters.", level="WARNING")
    
        # 检查 CUDA 是否可用
        args.cuda = torch.cuda.is_available()
    
        # 从环境变量获取当前进程的排名
        args.rank = int(os.getenv("RANK", "0"))
        # 从环境变量获取世界大小
        args.world_size = int(os.getenv("WORLD_SIZE", "1"))
        # 如果本地排名未设置
        if args.local_rank is None:
            # 从环境变量获取本地排名
            args.local_rank = int(os.getenv("LOCAL_RANK", "0"))  # torchrun
    
        # 如果设备未手动设置
        if args.device == -1:  # not set manually
            # 如果没有可用的 CUDA 设备
            if torch.cuda.device_count() == 0:
                # 使用 CPU 作为设备
                args.device = "cpu"
            # 如果本地排名已设置
            elif args.local_rank is not None:
                # 将本地排名设置为设备
                args.device = args.local_rank
            else:
                # 使用当前排名与 CUDA 设备数量的余数作为设备
                args.device = args.rank % torch.cuda.device_count()
    
        # 本地排名在 DeepSpeed 中应与设备一致
        if args.local_rank != args.device and args.mode != "inference":
            # 抛出不一致错误
            raise ValueError(
                "LOCAL_RANK (default 0) and args.device inconsistent. "
                "This can only happens in inference mode. "
                "Please use CUDA_VISIBLE_DEVICES=x for single-GPU training. "
            )
    
        # args.model_parallel_size = min(args.model_parallel_size, args.world_size)
        # 如果当前进程为 0
        if args.rank == 0:
            # 打印世界大小
            print_rank0("using world size: {}".format(args.world_size))
        # if args.vocab_size > 0:
        #     _adjust_vocab_size(args)
    
        # 如果训练数据权重已设置
        if args.train_data_weights is not None:
            # 确保权重和训练数据的长度一致
            assert len(args.train_data_weights) == len(args.train_data)
    
        # 如果模式不是推理，则进行训练
        if args.mode != "inference":  # training with deepspeed
            # 启用 DeepSpeed
            args.deepspeed = True
            # 如果未指定 DeepSpeed 配置
            if args.deepspeed_config is None:  # not specified
                # 生成 DeepSpeed 配置路径
                deepspeed_config_path = os.path.join(
                    os.path.dirname(__file__), "training", f"deepspeed_zero{args.zero_stage}.json"
                )
                # 打开并加载 DeepSpeed 配置文件
                with open(deepspeed_config_path) as file:
                    args.deepspeed_config = json.load(file)
                # 标记为覆盖 DeepSpeed 配置
                override_deepspeed_config = True
            else:
                # 不覆盖 DeepSpeed 配置
                override_deepspeed_config = False
    
        # 确保不能同时指定 fp16 和 bf16
        assert not (args.fp16 and args.bf16), "cannot specify both fp16 and bf16."
    
        # 如果 zero_stage 大于 0 并且 fp16 和 bf16 均未设置
        if args.zero_stage > 0 and not args.fp16 and not args.bf16:
            # 自动设置 fp16 为 True
            print_rank0("Automatically set fp16=True to use ZeRO.")
            args.fp16 = True
            # 设置 bf16 为 False
            args.bf16 = False
    # 检查是否启用 DeepSpeed
        if args.deepspeed:
            # 检查是否启用检查点激活
            if args.checkpoint_activations:
                # 启用 DeepSpeed 激活检查点
                args.deepspeed_activation_checkpointing = True
            else:
                # 禁用 DeepSpeed 激活检查点
                args.deepspeed_activation_checkpointing = False
            # 检查是否指定了 DeepSpeed 配置
            if args.deepspeed_config is not None:
                # 将 DeepSpeed 配置赋值
                deepspeed_config = args.deepspeed_config
                # 注释掉的代码，读取 JSON 格式的 DeepSpeed 配置
                # with open(args.deepspeed_config) as file:
                #     deepspeed_config = json.load(file)
    
            # 如果覆盖 DeepSpeed 配置
            if override_deepspeed_config:  # not specify deepspeed_config, use args
                # 检查是否启用 FP16 精度
                if args.fp16:
                    deepspeed_config["fp16"]["enabled"] = True
                # 检查是否启用 BF16 精度
                elif args.bf16:
                    deepspeed_config["bf16"]["enabled"] = True
                    deepspeed_config["fp16"]["enabled"] = False
                else:
                    # 禁用 FP16 精度
                    deepspeed_config["fp16"]["enabled"] = False
                # 设置每个 GPU 的微批大小
                deepspeed_config["train_micro_batch_size_per_gpu"] = args.batch_size
                # 设置梯度累积步数
                deepspeed_config["gradient_accumulation_steps"] = args.gradient_accumulation_steps
                # 获取优化器参数配置
                optimizer_params_config = deepspeed_config["optimizer"]["params"]
                # 设置学习率
                optimizer_params_config["lr"] = args.lr
                # 设置权重衰减
                optimizer_params_config["weight_decay"] = args.weight_decay
            else:  # override args with values in deepspeed_config
                # 如果当前进程为主进程，输出提示信息
                if args.rank == 0:
                    print_rank0("Will override arguments with manually specified deepspeed_config!")
                # 检查 FP16 配置并更新参数
                if "fp16" in deepspeed_config and deepspeed_config["fp16"]["enabled"]:
                    args.fp16 = True
                else:
                    args.fp16 = False
                # 检查 BF16 配置并更新参数
                if "bf16" in deepspeed_config and deepspeed_config["bf16"]["enabled"]:
                    args.bf16 = True
                else:
                    args.bf16 = False
                # 更新每个 GPU 的微批大小
                if "train_micro_batch_size_per_gpu" in deepspeed_config:
                    args.batch_size = deepspeed_config["train_micro_batch_size_per_gpu"]
                # 更新梯度累积步数，如果没有则设为 None
                if "gradient_accumulation_steps" in deepspeed_config:
                    args.gradient_accumulation_steps = deepspeed_config["gradient_accumulation_steps"]
                else:
                    args.gradient_accumulation_steps = None
                # 更新优化器参数
                if "optimizer" in deepspeed_config:
                    optimizer_params_config = deepspeed_config["optimizer"].get("params", {})
                    args.lr = optimizer_params_config.get("lr", args.lr)
                    args.weight_decay = optimizer_params_config.get("weight_decay", args.weight_decay)
            # 将 DeepSpeed 配置存储到 args 中
            args.deepspeed_config = deepspeed_config
    
        # 注释掉的代码，处理 sandwich 层归一化（在 v0.3 中移除）
        # if args.sandwich_ln: # removed in v0.3
        #     args.layernorm_order = 'sandwich'
    
        # 初始化分布式环境和随机种子
        initialize_distributed(args)
        # 设置种子，增加当前进程的排名
        args.seed = args.seed + torch.distributed.get_rank()
        # 设置随机种子
        set_random_seed(args.seed)
        # 返回更新后的参数
        return args
# 初始化分布式训练的设置
def initialize_distributed(args):
    """Initialize torch.distributed."""
    # 检查分布式训练是否已初始化
    if torch.distributed.is_initialized():
        # 检查模型并行是否已初始化
        if mpu.model_parallel_is_initialized():
            # 检查模型并行大小是否与之前的配置一致
            if args.model_parallel_size != mpu.get_model_parallel_world_size():
                raise ValueError(
                    "model_parallel_size is inconsistent with prior configuration."
                    "We currently do not support changing model_parallel_size."
                )
            return False
        else:
            # 如果模型并行大小大于1且未通过SAT初始化分布式
            if args.model_parallel_size > 1:
                warnings.warn(
                    "model_parallel_size > 1 but torch.distributed is not initialized via SAT."
                    "Please carefully make sure the correctness on your own."
                )
            # 初始化模型并行
            mpu.initialize_model_parallel(args.model_parallel_size)
        return True
    # 将设备的自动分配移至arguments.py
    if args.device == "cpu":
        pass
    else:
        # 设置当前CUDA设备
        torch.cuda.set_device(args.device)
    # 设置初始化方法
    init_method = "tcp://"
    # 获取主节点IP，默认为localhost
    args.master_ip = os.getenv("MASTER_ADDR", "localhost")

    # 如果世界规模为1，获取一个可用的端口
    if args.world_size == 1:
        from sat.helpers import get_free_port

        default_master_port = str(get_free_port())
    else:
        # 否则使用默认端口6000
        default_master_port = "6000"
    # 获取主节点端口，优先使用环境变量
    args.master_port = os.getenv("MASTER_PORT", default_master_port)
    # 构建初始化方法字符串
    init_method += args.master_ip + ":" + args.master_port
    # 初始化进程组
    torch.distributed.init_process_group(
        backend=args.distributed_backend, world_size=args.world_size, rank=args.rank, init_method=init_method
    )

    # 设置模型并行和数据并行的通信器
    # mpu.initialize_model_parallel(args.model_parallel_size)
    mpu.initialize_model_parallel(1)
    # 可选的DeepSpeed激活检查点功能
    if args.deepspeed:
        import deepspeed

        # 初始化DeepSpeed分布式设置
        deepspeed.init_distributed(
            dist_backend=args.distributed_backend, world_size=args.world_size, rank=args.rank, init_method=init_method
        )
        # # 配置检查点，即使未使用也似乎没有负面影响
        # deepspeed.checkpointing.configure(mpu, deepspeed_config=args.deepspeed_config, num_checkpoints=args.num_layers)
    else:
        # 在仅模型模式下，不想初始化DeepSpeed，但仍需初始化rng追踪器，以便在丢弃时保存种子
        try:
            import deepspeed
            from deepspeed.runtime.activation_checkpointing.checkpointing import (
                _CUDA_RNG_STATE_TRACKER,
                _MODEL_PARALLEL_RNG_TRACKER_NAME,
            )

            # 默认种子为1，添加到RNG状态追踪器
            _CUDA_RNG_STATE_TRACKER.add(_MODEL_PARALLEL_RNG_TRACKER_NAME, 1)  # default seed 1
        except Exception as e:
            from sat.helpers import print_rank0

            # 打印调试级别的错误信息
            print_rank0(str(e), level="DEBUG")

    # 返回初始化成功
    return True


# 从--base中提取参数
def process_config_to_args(args):
    """Fetch args from only --base"""
    # 从给定的基本配置路径加载每个配置文件，并将它们组成一个列表
        configs = [OmegaConf.load(cfg) for cfg in args.base]
        # 合并所有加载的配置，形成一个单一的配置对象
        config = OmegaConf.merge(*configs)
    
        # 从合并后的配置中提取 "args" 部分，若不存在则创建一个空的 OmegaConf 对象
        args_config = config.pop("args", OmegaConf.create())
        # 遍历 args_config 中的每个键
        for key in args_config:
            # 检查值是否为字典或列表配置，若是则转换为普通对象
            if isinstance(args_config[key], omegaconf.DictConfig) or isinstance(args_config[key], omegaconf.ListConfig):
                arg = OmegaConf.to_object(args_config[key])
            else:
                # 否则直接获取其值
                arg = args_config[key]
            # 如果 args 中有该键，则设置其属性为对应的值
            if hasattr(args, key):
                setattr(args, key, arg)
    
        # 检查配置中是否包含 "model" 键
        if "model" in config:
            # 从配置中提取 "model" 部分，若不存在则创建一个空的 OmegaConf 对象
            model_config = config.pop("model", OmegaConf.create())
            # 将提取的模型配置赋值给 args 的 model_config 属性
            args.model_config = model_config
        # 检查配置中是否包含 "deepspeed" 键
        if "deepspeed" in config:
            # 从配置中提取 "deepspeed" 部分，若不存在则创建一个空的 OmegaConf 对象
            deepspeed_config = config.pop("deepspeed", OmegaConf.create())
            # 将提取的深度学习加速配置转换为对象并赋值给 args 的 deepspeed_config 属性
            args.deepspeed_config = OmegaConf.to_object(deepspeed_config)
        # 检查配置中是否包含 "data" 键
        if "data" in config:
            # 从配置中提取 "data" 部分，若不存在则创建一个空的 OmegaConf 对象
            data_config = config.pop("data", OmegaConf.create())
            # 将提取的数据配置赋值给 args 的 data_config 属性
            args.data_config = data_config
    
        # 返回更新后的 args 对象
        return args
```