# `.\pytorch\benchmarks\dynamo\distributed.py`

```
import argparse  # 导入用于解析命令行参数的模块
import logging  # 导入日志记录模块
import os  # 导入操作系统相关的功能
from functools import partial  # 导入 functools 模块中的 partial 函数

import torch  # 导入 PyTorch 深度学习库
import torch._dynamo as dynamo  # 导入 Torch 内部使用的 dynamo 模块
import torch.utils._pytree as pytree  # 导入 Torch 内部使用的 _pytree 模块
from torch._dynamo.testing import reduce_to_scalar_loss  # 从 Torch 内部的 _dynamo.testing 模块导入函数
from torch.nn.parallel import DistributedDataParallel as DDP  # 导入分布式数据并行处理模块的别名 DDP
from torch.profiler import profile, ProfilerActivity, record_function  # 导入性能分析相关模块

try:
    from .common import timed  # 尝试从当前目录下的 common 模块导入 timed 函数
    from .dist_util import apply_fsdp, cleanup, get_model, model_iter_fn, setup  # 尝试从当前目录下的 dist_util 模块导入多个函数
except ImportError:
    from common import timed  # 如果导入失败，则从全局的 common 模块导入 timed 函数
    from dist_util import apply_fsdp, cleanup, get_model, model_iter_fn, setup  # 如果导入失败，则从全局的 dist_util 模块导入多个函数

log = logging.getLogger(__name__)  # 获取当前模块的日志记录器


def torchviz_model(args, model, inputs, rank):
    from torchviz import make_dot  # 从 torchviz 模块导入 make_dot 函数

    outputs = model(*inputs)  # 使用模型对输入进行前向传播，得到输出
    loss = reduce_to_scalar_loss(outputs)  # 将输出转换为标量损失值
    parameter_names = dict(model.named_parameters())  # 获取模型中的所有参数名称
    dot = make_dot(loss, params=parameter_names, show_attrs=True, show_saved=True)  # 创建用于可视化的计算图
    if rank == 0:
        dot.render("torchviz.dot")  # 如果是主进程（rank == 0），则将计算图渲染为文件


def profile_model(args, model, inputs, rank):
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:  # 使用性能分析器，记录 CPU 和 CUDA 活动
        for i in range(args.repeat):
            with record_function("Forward"):  # 记录前向传播函数的执行
                outputs = model(*inputs)  # 对输入进行模型的前向传播
                loss = reduce_to_scalar_loss(outputs)  # 将输出转换为标量损失值
            with record_function("Backward"):  # 记录反向传播函数的执行
                loss.backward()  # 执行反向传播
    if rank == 0:
        prof.export_chrome_trace(args.trace_file)  # 如果是主进程（rank == 0），将性能分析结果导出为 Chrome 跟踪文件


def run_model(args, model, inputs, key):
    rank = int(os.getenv("RANK", 0))  # 获取环境变量 RANK 的值，表示当前进程的排名，默认为 0
    world_size = int(os.getenv("WORLD_SIZE", 1))  # 获取环境变量 WORLD_SIZE 的值，表示总的进程数，默认为 1
    # result_q = []  # 创建一个空列表（被注释掉的行）

    setup(rank, world_size)  # 设置当前进程的环境和总体进程数量
    if args.device == "cuda":
        torch.cuda.set_device(rank)  # 设置当前 CUDA 设备的编号为当前进程的排名

    dev_rank = f"{args.device}:{rank}"  # 根据设备类型和排名生成设备标识符
    model = model.to(dev_rank)  # 将模型移动到指定的设备

    def move_tensor(maybe_tensor):
        if torch.is_tensor(maybe_tensor):  # 判断输入是否为张量
            return maybe_tensor.to(dev_rank)  # 如果是张量，则将其移动到指定设备
        return maybe_tensor  # 如果不是张量，则直接返回

    inputs = pytree.tree_map(move_tensor, inputs)  # 使用 pytree 库中的函数将输入数据结构中的所有张量移动到指定设备

    if args.fsdp:
        model = apply_fsdp(
            args,
            model,
            use_checkpointing=args.fsdp_checkpoint,
            use_wrap_policy=args.fsdp_wrap,
        )  # 如果启用 FSDP，则对模型应用 FSDP 策略
    elif args.ddp:
        model = DDP(model)  # 如果启用 DDP，则使用 DDP 对模型进行分布式数据并行处理

    if args.verbose:
        print(model)  # 如果设置了 verbose 标志，则打印模型信息
    # 如果参数中包含 dynamo
    if args.dynamo:
        # 重置 dynamo
        dynamo.reset()
        
        # 如果参数中包含 verbose，则设置 dynamo 的 verbose 和 log_level
        if args.verbose:
            dynamo.config.verbose = True
            dynamo.config.log_level = logging.DEBUG
        
        # 如果参数中包含 dynamo_no_optimize_ddp，则设置 dynamo 的 optimize_ddp 为 False
        if args.dynamo_no_optimize_ddp:
            dynamo.config.optimize_ddp = False
        
        # 如果参数中同时包含 dynamo 是 "inductor" 且 fsdp 也存在
        if args.dynamo == "inductor" and args.fsdp:
            # 设置 triton 的 cudagraphs 为 False，用于与 FSDP 兼容性
            torch._inductor.config.triton.cudagraphs = False
            # 发出警告信息，提示 cudagraphs 被禁用以确保兼容性
            log.warning("disabling inductor cudagraphs for compatibility with FSDP")

        # 定义一个函数 print_compile，如果 dynamo 参数是 "print" 则打印模型的计算图
        def print_compile(gm, ex):
            print(
                f"print_compile:\n{str(gm.graph)}\n-----------------------------------------"
            )
            return gm
        
        # 根据 dynamo 的优化策略进行优化，根据参数不同传入不同的优化函数或者 print_compile 函数
        dynamo_ctx = dynamo.optimize(
            print_compile if args.dynamo == "print" else args.dynamo
        )
        
        # 使用优化后的 dynamo 上下文来优化模型
        model = dynamo_ctx(model)

    # 执行模型的预热运行，运行 3 次，并且不返回结果
    _ = timed(model, model_iter_fn, inputs, times=3, return_result=False)
    
    # 测量模型运行的总时间，运行 args.repeat 次，并且不返回结果
    t_total = timed(
        model, model_iter_fn, inputs, times=args.repeat, return_result=False
    )
    
    # 如果参数中包含 torchviz，则对模型进行可视化
    if args.torchviz:
        torchviz_model(args, model, inputs, rank)
    
    # 如果参数中包含 profile，则对模型进行性能分析
    if args.profile:
        profile_model(args, model, inputs, rank)
    
    # 执行清理操作
    cleanup()
    
    # 返回总时间 t_total
    return t_total
if __name__ == "__main__":
    # 如果当前脚本作为主程序运行，则执行以下代码块

    parser = argparse.ArgumentParser()
    # 创建参数解析器对象

    parser.add_argument("--device", default="cuda")
    # 添加名为 --device 的命令行参数，默认值为 "cuda"

    parser.add_argument(
        "--dynamo",
        default=None,
        help="if set to a str, uses dynamo[str] backend. else, eager",
    )
    # 添加名为 --dynamo 的命令行参数，若设置为字符串，则使用对应的 dynamo[str] 后端，否则为 eager 模式

    parser.add_argument("--verbose", action="store_true")
    # 添加一个布尔类型的命令行参数 --verbose，若设置则为 True

    parser.add_argument("--batch-size", "--batch_size", default=None)
    # 添加名为 --batch-size 或 --batch_size 的命令行参数，默认值为 None

    parser.add_argument(
        "--torchviz", action="store_true", help="Dump autograd graph with torchviz"
    )
    # 添加一个布尔类型的命令行参数 --torchviz，若设置则使用 torchviz 输出 autograd 图

    parser.add_argument("--profile", action="store_true", help="Run the profiler")
    # 添加一个布尔类型的命令行参数 --profile，若设置则运行分析器（profiler）

    parser.add_argument(
        "--trace-file", "--trace_file", default="profile.json", help="Run the profiler"
    )
    # 添加名为 --trace-file 或 --trace_file 的命令行参数，默认为 "profile.json"

    parser.add_argument("--repeat", default=10, help="Repeats for timing run")
    # 添加名为 --repeat 的命令行参数，默认值为 10，表示运行的重复次数

    parser.add_argument(
        "--dynamo-no-optimize-ddp",
        "--dynamo_no_optimize_ddp",
        action="store_true",
        help="Disable dynamo's ddp optimizer (enabled by default)",
    )
    # 添加一个布尔类型的命令行参数 --dynamo-no-optimize-ddp 或 --dynamo_no_optimize_ddp，
    # 若设置则禁用 dynamo 的 ddp 优化器（默认启用）

    parser.add_argument(
        "--fsdp-checkpoint",
        "--fsdp_checkpoint",
        action="store_true",
        help="Use gradient checkpointing via model-specific policy",
    )
    # 添加一个布尔类型的命令行参数 --fsdp-checkpoint 或 --fsdp_checkpoint，
    # 若设置则使用模型特定策略的梯度检查点

    parser.add_argument(
        "--fsdp-wrap",
        "--fsdp_wrap",
        action="store_true",
        help="Apply fsdp to submodules via model-specific policy",
    )
    # 添加一个布尔类型的命令行参数 --fsdp-wrap 或 --fsdp_wrap，
    # 若设置则通过模型特定策略将 fsdp 应用于子模块

    dist_arg = parser.add_mutually_exclusive_group()
    # 创建互斥组 dist_arg，用于处理互斥的命令行参数

    dist_arg.add_argument("--ddp", action="store_true")
    # 在 dist_arg 组中添加一个布尔类型的命令行参数 --ddp

    dist_arg.add_argument("--fsdp", action="store_true")
    # 在 dist_arg 组中添加一个布尔类型的命令行参数 --fsdp

    model_arg = parser.add_mutually_exclusive_group(required=True)
    # 创建必须选择一个的互斥组 model_arg

    model_arg.add_argument(
        "--torchbench-model",
        "--torchbench_model",
        help="name of torchbench model, e.g. hf_Bert",
    )
    # 在 model_arg 组中添加一个参数 --torchbench-model 或 --torchbench_model，
    # 用于指定 torchbench 模型的名称，例如 hf_Bert

    model_arg.add_argument(
        "--toy-model", "--toy_model", action="store_true", help="use toy model instead"
    )
    # 在 model_arg 组中添加一个布尔类型的参数 --toy-model 或 --toy_model，
    # 若设置则使用玩具模型而不是真实模型

    args = parser.parse_args()
    # 解析命令行参数并将结果存储在 args 对象中

    model_name = args.torchbench_model
    # 将 args 中的 torchbench_model 参数值赋给 model_name
    if args.toy_model:
        model_name = "ToyModel"
    # 若命令行参数中设置了 --toy-model，则将 model_name 设为 "ToyModel"

    model, inputs = get_model(args)
    # 调用函数 get_model，并传入 args 作为参数，获取模型和输入数据

    fn = partial(run_model, args, model, inputs)
    # 创建函数 fn，部分应用 run_model 函数，并传入 args、model 和 inputs 作为参数

    world_size = os.getenv("WORLD_SIZE", 1)
    # 从环境变量中获取 WORLD_SIZE 的值，默认为 1

    t_total = fn(f"{model_name}_{world_size}")
    # 调用 fn 函数，传入模型名称及 world_size 构成的字符串作为参数，并将结果赋给 t_total

    print(f"mean latency {t_total / args.repeat} across {args.repeat} runs")
    # 输出平均延迟及运行的总次数
```