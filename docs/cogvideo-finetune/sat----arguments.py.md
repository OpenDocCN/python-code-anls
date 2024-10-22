# `.\cogvideo-finetune\sat\arguments.py`

```py
# 导入所需的库和模块
import argparse  # 用于处理命令行参数
import os  # 提供与操作系统交互的功能
import torch  # PyTorch库，用于深度学习
import json  # 用于处理JSON数据
import warnings  # 用于发出警告
import omegaconf  # 用于配置管理
from omegaconf import OmegaConf  # 从omegaconf导入OmegaConf类
from sat.helpers import print_rank0  # 从sat.helpers导入print_rank0函数
from sat import mpu  # 导入sat模块中的mpu部分
from sat.arguments import set_random_seed  # 从sat.arguments导入设置随机种子的函数
from sat.arguments import add_training_args, add_evaluation_args, add_data_args  # 导入添加参数的函数
import torch.distributed  # 导入PyTorch分布式训练功能

def add_model_config_args(parser):
    """Model arguments"""  # 函数说明：添加模型参数配置

    group = parser.add_argument_group("model", "model configuration")  # 创建模型参数组
    group.add_argument("--base", type=str, nargs="*", help="config for input and saving")  # 添加基本配置参数
    group.add_argument(
        "--model-parallel-size", type=int, default=1, help="size of the model parallel. only use if you are an expert."
    )  # 添加模型并行大小参数
    group.add_argument("--force-pretrain", action="store_true")  # 添加强制预训练标志
    group.add_argument("--device", type=int, default=-1)  # 添加设备参数，默认为-1
    group.add_argument("--debug", action="store_true")  # 添加调试标志
    group.add_argument("--log-image", type=bool, default=True)  # 添加日志图像参数，默认为True

    return parser  # 返回更新后的解析器

def add_sampling_config_args(parser):
    """Sampling configurations"""  # 函数说明：添加采样配置参数

    group = parser.add_argument_group("sampling", "Sampling Configurations")  # 创建采样参数组
    group.add_argument("--output-dir", type=str, default="samples")  # 添加输出目录参数
    group.add_argument("--input-dir", type=str, default=None)  # 添加输入目录参数，默认为None
    group.add_argument("--input-type", type=str, default="cli")  # 添加输入类型参数，默认为"cli"
    group.add_argument("--input-file", type=str, default="input.txt")  # 添加输入文件参数，默认为"input.txt"
    group.add_argument("--final-size", type=int, default=2048)  # 添加最终尺寸参数，默认为2048
    group.add_argument("--sdedit", action="store_true")  # 添加sdedit标志
    group.add_argument("--grid-num-rows", type=int, default=1)  # 添加网格行数参数，默认为1
    group.add_argument("--force-inference", action="store_true")  # 添加强制推理标志
    group.add_argument("--lcm_steps", type=int, default=None)  # 添加最小公倍数步骤参数，默认为None
    group.add_argument("--sampling-num-frames", type=int, default=32)  # 添加采样帧数参数，默认为32
    group.add_argument("--sampling-fps", type=int, default=8)  # 添加采样帧率参数，默认为8
    group.add_argument("--only-save-latents", type=bool, default=False)  # 添加仅保存潜变量标志，默认为False
    group.add_argument("--only-log-video-latents", type=bool, default=False)  # 添加仅记录视频潜变量标志，默认为False
    group.add_argument("--latent-channels", type=int, default=32)  # 添加潜变量通道数参数，默认为32
    group.add_argument("--image2video", action="store_true")  # 添加图像转视频标志

    return parser  # 返回更新后的解析器

def get_args(args_list=None, parser=None):
    """Parse all the args."""  # 函数说明：解析所有参数
    if parser is None:  # 检查解析器是否为None
        parser = argparse.ArgumentParser(description="sat")  # 创建新的解析器
    else:
        assert isinstance(parser, argparse.ArgumentParser)  # 确保解析器是argparse.ArgumentParser的实例
    parser = add_model_config_args(parser)  # 添加模型参数配置
    parser = add_sampling_config_args(parser)  # 添加采样参数配置
    parser = add_training_args(parser)  # 添加训练参数
    parser = add_evaluation_args(parser)  # 添加评估参数
    parser = add_data_args(parser)  # 添加数据参数

    import deepspeed  # 导入DeepSpeed库

    parser = deepspeed.add_config_arguments(parser)  # 添加DeepSpeed配置参数

    args = parser.parse_args(args_list)  # 解析命令行参数
    args = process_config_to_args(args)  # 处理配置并转换为参数

    if not args.train_data:  # 检查是否指定训练数据
        print_rank0("No training data specified", level="WARNING")  # 打印警告信息

    assert (args.train_iters is None) or (args.epochs is None), "only one of train_iters and epochs should be set."  # 确保train_iters和epochs只能有一个被设置
    # 检查训练迭代次数和周期是否为 None
        if args.train_iters is None and args.epochs is None:
            # 如果两者均为 None，设置默认训练迭代次数为 10000
            args.train_iters = 10000  # default 10k iters
            # 输出警告信息，提示使用默认的 10k 迭代
            print_rank0("No train_iters (recommended) or epochs specified, use default 10k iters.", level="WARNING")
    
        # 检查 CUDA 是否可用，并设置 args.cuda
        args.cuda = torch.cuda.is_available()
    
        # 从环境变量获取 RANK，并转为整数，默认为 0
        args.rank = int(os.getenv("RANK", "0"))
        # 从环境变量获取 WORLD_SIZE，并转为整数，默认为 1
        args.world_size = int(os.getenv("WORLD_SIZE", "1"))
        # 如果 local_rank 为 None，从环境变量获取 LOCAL_RANK，并转为整数，默认为 0
        if args.local_rank is None:
            args.local_rank = int(os.getenv("LOCAL_RANK", "0"))  # torchrun
    
        # 如果 device 设置为 -1，进行设备选择
        if args.device == -1:
            # 如果没有可用的 CUDA 设备，设置为 CPU
            if torch.cuda.device_count() == 0:
                args.device = "cpu"
            # 如果 local_rank 不为 None，使用 local_rank 作为设备
            elif args.local_rank is not None:
                args.device = args.local_rank
            # 否则，使用 rank 对设备数量取模
            else:
                args.device = args.rank % torch.cuda.device_count()
    
        # 如果 local_rank 不等于 device，且模式不是推理，则抛出错误
        if args.local_rank != args.device and args.mode != "inference":
            raise ValueError(
                "LOCAL_RANK (default 0) and args.device inconsistent. "
                "This can only happens in inference mode. "
                "Please use CUDA_VISIBLE_DEVICES=x for single-GPU training. "
            )
    
        # 如果 rank 为 0，输出当前的 world size
        if args.rank == 0:
            print_rank0("using world size: {}".format(args.world_size))
    
        # 如果训练数据权重不为 None，检查其长度是否与训练数据一致
        if args.train_data_weights is not None:
            assert len(args.train_data_weights) == len(args.train_data)
    
        # 如果模式不是推理，进行 DeepSpeed 训练配置
        if args.mode != "inference":  # training with deepspeed
            args.deepspeed = True
            # 如果 DeepSpeed 配置未指定，构造配置路径
            if args.deepspeed_config is None:  # not specified
                deepspeed_config_path = os.path.join(
                    os.path.dirname(__file__), "training", f"deepspeed_zero{args.zero_stage}.json"
                )
                # 打开 DeepSpeed 配置文件并加载内容
                with open(deepspeed_config_path) as file:
                    args.deepspeed_config = json.load(file)
                # 标记需要覆盖配置
                override_deepspeed_config = True
            else:
                override_deepspeed_config = False
    
        # 确保不能同时指定 fp16 和 bf16
        assert not (args.fp16 and args.bf16), "cannot specify both fp16 and bf16."
    
        # 如果 zero_stage 大于 0，且未指定 fp16 或 bf16，则自动设置 fp16 为 True
        if args.zero_stage > 0 and not args.fp16 and not args.bf16:
            print_rank0("Automatically set fp16=True to use ZeRO.")
            args.fp16 = True
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
        # 如果 DeepSpeed 配置不为 None，获取配置
        if args.deepspeed_config is not None:
            deepspeed_config = args.deepspeed_config

        # 如果覆盖 DeepSpeed 配置，则使用 args
        if override_deepspeed_config:  # not specify deepspeed_config, use args
            # 如果启用 FP16
            if args.fp16:
                deepspeed_config["fp16"]["enabled"] = True
            # 如果启用 BF16
            elif args.bf16:
                deepspeed_config["bf16"]["enabled"] = True
                deepspeed_config["fp16"]["enabled"] = False
            else:
                # 禁用 FP16
                deepspeed_config["fp16"]["enabled"] = False
            # 设置每个 GPU 的微批大小
            deepspeed_config["train_micro_batch_size_per_gpu"] = args.batch_size
            # 设置梯度累积步骤
            deepspeed_config["gradient_accumulation_steps"] = args.gradient_accumulation_steps
            # 获取优化器参数配置
            optimizer_params_config = deepspeed_config["optimizer"]["params"]
            # 设置学习率
            optimizer_params_config["lr"] = args.lr
            # 设置权重衰减
            optimizer_params_config["weight_decay"] = args.weight_decay
        else:  # override args with values in deepspeed_config
            # 如果当前是主进程，打印信息
            if args.rank == 0:
                print_rank0("Will override arguments with manually specified deepspeed_config!")
            # 检查 FP16 配置
            if "fp16" in deepspeed_config and deepspeed_config["fp16"]["enabled"]:
                args.fp16 = True
            else:
                # 禁用 FP16
                args.fp16 = False
            # 检查 BF16 配置
            if "bf16" in deepspeed_config and deepspeed_config["bf16"]["enabled"]:
                args.bf16 = True
            else:
                # 禁用 BF16
                args.bf16 = False
            # 获取每个 GPU 的微批大小
            if "train_micro_batch_size_per_gpu" in deepspeed_config:
                args.batch_size = deepspeed_config["train_micro_batch_size_per_gpu"]
            # 获取梯度累积步骤
            if "gradient_accumulation_steps" in deepspeed_config:
                args.gradient_accumulation_steps = deepspeed_config["gradient_accumulation_steps"]
            else:
                # 如果没有设置，梯度累积步骤为 None
                args.gradient_accumulation_steps = None
            # 检查优化器配置
            if "optimizer" in deepspeed_config:
                optimizer_params_config = deepspeed_config["optimizer"].get("params", {})
                # 设置学习率
                args.lr = optimizer_params_config.get("lr", args.lr)
                # 设置权重衰减
                args.weight_decay = optimizer_params_config.get("weight_decay", args.weight_decay)
        # 更新 DeepSpeed 配置到 args
        args.deepspeed_config = deepspeed_config

    # 初始化分布式和随机种子，因为这似乎总是必要的
    initialize_distributed(args)
    # 为当前进程设置种子
    args.seed = args.seed + mpu.get_data_parallel_rank()
    # 设置随机种子
    set_random_seed(args.seed)
    # 返回更新后的 args
    return args
# 初始化分布式训练，使用 torch.distributed
def initialize_distributed(args):
    """Initialize torch.distributed."""
    # 检查分布式是否已初始化
    if torch.distributed.is_initialized():
        # 检查模型并行是否已初始化
        if mpu.model_parallel_is_initialized():
            # 如果模型并行大小与先前配置不一致，抛出错误
            if args.model_parallel_size != mpu.get_model_parallel_world_size():
                raise ValueError(
                    "model_parallel_size is inconsistent with prior configuration."
                    "We currently do not support changing model_parallel_size."
                )
            # 如果一致，返回 False
            return False
        else:
            # 如果模型并行大小大于 1，发出警告
            if args.model_parallel_size > 1:
                warnings.warn(
                    "model_parallel_size > 1 but torch.distributed is not initialized via SAT."
                    "Please carefully make sure the correctness on your own."
                )
            # 初始化模型并行
            mpu.initialize_model_parallel(args.model_parallel_size)
        # 返回 True，表示初始化成功
        return True
    # 自动设备分配已转移到 arguments.py
    if args.device == "cpu":
        # 如果设备是 CPU，什么也不做
        pass
    else:
        # 设置当前 CUDA 设备
        torch.cuda.set_device(args.device)
    # 设置初始化方法
    init_method = "tcp://"
    # 获取主节点 IP，默认是 localhost
    args.master_ip = os.getenv("MASTER_ADDR", "localhost")

    # 如果世界大小为 1，获取一个可用端口
    if args.world_size == 1:
        from sat.helpers import get_free_port

        default_master_port = str(get_free_port())
    else:
        # 否则设置默认端口为 6000
        default_master_port = "6000"
    # 获取主节点端口，优先使用环境变量
    args.master_port = os.getenv("MASTER_PORT", default_master_port)
    # 构造初始化方法的完整地址
    init_method += args.master_ip + ":" + args.master_port
    # 初始化进程组，设置后端、世界大小、当前进程的排名和初始化方法
    torch.distributed.init_process_group(
        backend=args.distributed_backend, world_size=args.world_size, rank=args.rank, init_method=init_method
    )

    # 设置模型并行和数据并行的通信器
    mpu.initialize_model_parallel(args.model_parallel_size)

    # 将 VAE 上下文并行组设置为模型并行组
    from sgm.util import set_context_parallel_group, initialize_context_parallel

    # 如果模型并行大小小于等于 2，设置上下文并行组
    if args.model_parallel_size <= 2:
        set_context_parallel_group(args.model_parallel_size, mpu.get_model_parallel_group())
    else:
        # 否则初始化上下文并行
        initialize_context_parallel(2)
    # mpu.initialize_model_parallel(1)
    # 可选的 DeepSpeed 激活检查点功能
    if args.deepspeed:
        import deepspeed

        # 初始化 DeepSpeed 分布式
        deepspeed.init_distributed(
            dist_backend=args.distributed_backend, world_size=args.world_size, rank=args.rank, init_method=init_method
        )
        # # 配置检查点，即使不使用也似乎没有负面影响
        # deepspeed.checkpointing.configure(mpu, deepspeed_config=args.deepspeed_config, num_checkpoints=args.num_layers)
    # 否则分支，表示处于模型仅模式，不初始化 deepspeed，但仍需初始化随机数跟踪器
        else:
            # 在模型仅模式下，不初始化 deepspeed，但需要初始化 rng 跟踪器，以便在 dropout 时保存种子
            try:
                # 尝试导入 deepspeed 模块
                import deepspeed
                # 从 deepspeed 导入激活检查点相关的 RNG 跟踪器
                from deepspeed.runtime.activation_checkpointing.checkpointing import (
                    _CUDA_RNG_STATE_TRACKER,
                    _MODEL_PARALLEL_RNG_TRACKER_NAME,
                )
    
                # 将默认种子 1 添加到 CUDA RNG 状态跟踪器
                _CUDA_RNG_STATE_TRACKER.add(_MODEL_PARALLEL_RNG_TRACKER_NAME, 1)  # default seed 1
            except Exception as e:
                # 如果发生异常，从 sat.helpers 导入打印函数
                from sat.helpers import print_rank0
    
                # 输出异常信息，级别为 DEBUG
                print_rank0(str(e), level="DEBUG")
    
        # 返回 True，表示执行成功
        return True
# 处理配置文件，将参数提取到 args 中
def process_config_to_args(args):
    """Fetch args from only --base"""  # 文档字符串，说明该函数从 --base 参数中获取配置

    # 加载每个配置文件并将其转换为 OmegaConf 对象，形成列表
    configs = [OmegaConf.load(cfg) for cfg in args.base]
    # 合并多个配置对象，生成一个综合配置
    config = OmegaConf.merge(*configs)

    # 从合并的配置中提取 "args" 部分，默认为一个空的 OmegaConf 对象
    args_config = config.pop("args", OmegaConf.create())
    # 遍历 args_config 中的每一个键
    for key in args_config:
        # 检查当前键的值是否为字典配置或列表配置
        if isinstance(args_config[key], omegaconf.DictConfig) or isinstance(args_config[key], omegaconf.ListConfig):
            # 将 OmegaConf 对象转换为普通 Python 对象
            arg = OmegaConf.to_object(args_config[key])
        else:
            # 否则直接获取该键的值
            arg = args_config[key]
        # 如果 args 对象中有该键，则设置其值
        if hasattr(args, key):
            setattr(args, key, arg)

    # 如果配置中包含 "model" 键，提取其值并设置到 args 中
    if "model" in config:
        model_config = config.pop("model", OmegaConf.create())  # 从配置中移除 "model"，默认为空 OmegaConf 对象
        args.model_config = model_config  # 将模型配置存储到 args 中
    # 如果配置中包含 "deepspeed" 键，提取其值并转换为对象
    if "deepspeed" in config:
        deepspeed_config = config.pop("deepspeed", OmegaConf.create())  # 从配置中移除 "deepspeed"，默认为空 OmegaConf 对象
        args.deepspeed_config = OmegaConf.to_object(deepspeed_config)  # 转换为普通对象并存储
    # 如果配置中包含 "data" 键，提取其值
    if "data" in config:
        data_config = config.pop("data", OmegaConf.create())  # 从配置中移除 "data"，默认为空 OmegaConf 对象
        args.data_config = data_config  # 将数据配置存储到 args 中

    # 返回更新后的 args 对象
    return args
```