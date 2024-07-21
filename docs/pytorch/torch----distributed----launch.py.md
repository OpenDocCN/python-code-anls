# `.\pytorch\torch\distributed\launch.py`

```
# mypy: allow-untyped-defs
r"""
Module ``torch.distributed.launch``.

``torch.distributed.launch`` is a module that spawns up multiple distributed
training processes on each of the training nodes.

.. warning::

    This module is going to be deprecated in favor of :ref:`torchrun <launcher-api>`.

The utility can be used for single-node distributed training, in which one or
more processes per node will be spawned. The utility can be used for either
CPU training or GPU training. If the utility is used for GPU training,
each distributed process will be operating on a single GPU. This can achieve
well-improved single-node training performance. It can also be used in
multi-node distributed training, by spawning up multiple processes on each node
for well-improved multi-node distributed training performance as well.
This will especially be beneficial for systems with multiple Infiniband
interfaces that have direct-GPU support, since all of them can be utilized for
aggregated communication bandwidth.

In both cases of single-node distributed training or multi-node distributed
training, this utility will launch the given number of processes per node
(``--nproc-per-node``). If used for GPU training, this number needs to be less
or equal to the number of GPUs on the current system (``nproc_per_node``),
and each process will be operating on a single GPU from *GPU 0 to
GPU (nproc_per_node - 1)*.

**How to use this module:**

1. Single-Node multi-process distributed training

::

    python -m torch.distributed.launch --nproc-per-node=NUM_GPUS_YOU_HAVE
               YOUR_TRAINING_SCRIPT.py (--arg1 --arg2 --arg3 and all other
               arguments of your training script)

    This command starts distributed training on a single node, spawning multiple
    processes according to the number of GPUs available.

2. Multi-Node multi-process distributed training: (e.g. two nodes)

Node 1: *(IP: 192.168.1.1, and has a free port: 1234)*

::

    python -m torch.distributed.launch --nproc-per-node=NUM_GPUS_YOU_HAVE
               --nnodes=2 --node-rank=0 --master-addr="192.168.1.1"
               --master-port=1234 YOUR_TRAINING_SCRIPT.py (--arg1 --arg2 --arg3
               and all other arguments of your training script)

    This command launches distributed training across multiple nodes, where each
    node has multiple processes running.

Node 2:

::

    python -m torch.distributed.launch --nproc-per-node=NUM_GPUS_YOU_HAVE
               --nnodes=2 --node-rank=1 --master-addr="192.168.1.1"
               --master-port=1234 YOUR_TRAINING_SCRIPT.py (--arg1 --arg2 --arg3
               and all other arguments of your training script)

    Similar to Node 1, this command launches distributed training on another node
    in the cluster.

3. To look up what optional arguments this module offers:

::

    python -m torch.distributed.launch --help

    This command provides information on additional arguments that can be used
    with the torch distributed launcher.

**Important Notices:**

1. This utility and multi-process distributed (single-node or
multi-node) GPU training currently only achieves the best performance using
the NCCL distributed backend. Thus NCCL backend is the recommended backend to
use for GPU training.

2. In your training program, you must parse the command-line argument:
``--local-rank=LOCAL_PROCESS_RANK``, which will be provided by this module.

"""
# 如果你的训练程序使用了GPU，确保代码只在LOCAL_PROCESS_RANK指定的GPU设备上运行。
# 这可以通过以下步骤实现：

# 解析 local_rank 参数
>>> import argparse
>>> parser = argparse.ArgumentParser()
>>> parser.add_argument("--local-rank", "--local_rank", type=int)
>>> args = parser.parse_args()

# 在你的代码运行之前将设备设置为 local_rank
>>> torch.cuda.set_device(args.local_rank)

# 或者使用以下方式之一将设备设置为 local_rank
>>> with torch.cuda.device(args.local_rank):
>>>     # 这里是你的代码运行部分
>>>     ...

# 从 PyTorch 2.0.0 开始，启动器将传递 "--local-rank=<rank>" 参数到你的脚本。
# 推荐使用中划线 "--local-rank"，而不是之前使用的下划线 "--local_rank"。
# 为了向后兼容，用户需要在参数解析代码中同时处理 "--local-rank" 和 "--local_rank"。
# 如果仅提供 "--local_rank"，启动器将触发错误："error: unrecognized arguments: --local-rank=<rank>"。

# 在你的训练程序中，建议在开头调用以下函数来启动分布式后端。
# 强烈建议使用 init_method='env://'。其他的 init 方法（如 'tcp://'）可能也能工作，
# 但是 'env://' 是官方支持的方法。

>>> torch.distributed.init_process_group(backend='YOUR BACKEND',
>>>                                      init_method='env://')

# 在你的训练程序中，你可以使用常规的分布式函数，也可以使用 torch.nn.parallel.DistributedDataParallel 模块。
# 如果你的训练程序使用GPU进行训练，并且想要使用 DistributedDataParallel 模块，
# 下面是配置方法。

>>> model = torch.nn.parallel.DistributedDataParallel(model,
>>>                                                   device_ids=[args.local_rank],
>>>                                                   output_device=args.local_rank)

# 确保 "device_ids" 参数设置为代码将操作的唯一 GPU 设备 ID。通常是进程的本地排名（local rank）。
# 换句话说，"device_ids" 需要是 `[args.local_rank]`，而 "output_device" 需要是 `args.local_rank`，
# 才能使用这个实用程序。

# 另一种将 local_rank 传递给子进程的方法是通过环境变量 "LOCAL_RANK"。
# 当使用 `--use-env=True` 启动脚本时，会启用此行为。
# 你必须调整上面子进程示例中的 "args.local_rank" 为 "os.environ['LOCAL_RANK']"；
# 当指定此标志时，启动器将不会传递 "--local-rank"。

# 注意：
    # `local_rank`并非全局唯一：它仅在单个进程内唯一，在一台机器上不是。因此，不要使用它来决定是否应该写入网络文件系统等操作。
    # 参考 https://github.com/pytorch/pytorch/issues/12042 了解如果不正确处理这个问题可能会导致的错误示例。
# 从 typing_extensions 模块中导入 deprecated 装饰器，并重命名为 _deprecated
from typing_extensions import deprecated as _deprecated

# 从 torch.distributed.run 模块中导入 get_args_parser 和 run 函数
from torch.distributed.run import get_args_parser, run

# 定义函数 parse_args，接受参数 args
def parse_args(args):
    # 调用 get_args_parser 函数创建参数解析器对象 parser
    parser = get_args_parser()
    # 向 parser 添加命令行参数：
    # --use-env 或 --use_env：用于通过环境变量传递 'local rank'，默认为 False
    # 若设置为 True，脚本将不会传递 --local-rank 参数，而是设置 LOCAL_RANK
    parser.add_argument(
        "--use-env",
        "--use_env",
        default=False,
        action="store_true",
        help="Use environment variable to pass "
        "'local rank'. For legacy reasons, the default value is False. "
        "If set to True, the script will not pass "
        "--local-rank as argument, and will instead set LOCAL_RANK.",
    )
    # 使用 parser 对传入的 args 进行解析并返回解析结果
    return parser.parse_args(args)

# 定义函数 launch，接受参数 args
def launch(args):
    # 如果参数 args 中 no_python 为 True 且 use_env 为 False，则抛出 ValueError 异常
    if args.no_python and not args.use_env:
        raise ValueError(
            "When using the '--no-python' flag,"
            " you must also set the '--use-env' flag."
        )
    # 调用 run 函数，执行分布式任务
    run(args)

# 使用 _deprecated 装饰器装饰 main 函数，提供相关警告信息和建议
@_deprecated(
    "The module torch.distributed.launch is deprecated\n"
    "and will be removed in future. Use torchrun.\n"
    "Note that --use-env is set by default in torchrun.\n"
    "If your script expects `--local-rank` argument to be set, please\n"
    "change it to read from `os.environ['LOCAL_RANK']` instead. See \n"
    "https://pytorch.org/docs/stable/distributed.html#launch-utility for \n"
    "further instructions\n",
    category=FutureWarning,
)
# 定义主函数 main，接受参数 args，默认为 None
def main(args=None):
    # 调用 parse_args 函数解析参数，返回解析结果赋值给 args
    args = parse_args(args)
    # 调用 launch 函数，启动分布式任务
    launch(args)

# 如果当前脚本作为主程序运行，则调用 main 函数
if __name__ == "__main__":
    main()
```