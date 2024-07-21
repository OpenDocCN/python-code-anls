# `.\pytorch\benchmarks\dynamo\dist_util.py`

```
`
# 导入必要的库和模块
import argparse   # 用于解析命令行参数的库
import functools  # 提供高阶函数操作的库
import importlib  # 提供模块导入功能的库
import os         # 提供与操作系统交互的功能

import torch                      # PyTorch 深度学习框架
import torch.distributed as dist  # PyTorch 分布式训练模块
import torch.nn as nn             # PyTorch 神经网络模块
from torch._dynamo.testing import reduce_to_scalar_loss  # 减少到标量损失的测试工具
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing,
    checkpoint_wrapper,
    CheckpointImpl,
)  # 检查点相关的功能和模块导入
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP  # 全分片数据并行模块
from torch.distributed.fsdp.wrap import ModuleWrapPolicy  # 模块包装策略

try:
    from .torchbench import setup_torchbench_cwd  # 尝试从当前目录导入 torchbench 相关设置
except ImportError:
    from torchbench import setup_torchbench_cwd  # 如果导入失败，则从 torchbench 库中导入相关设置

from transformers.models.bert.modeling_bert import BertLayer, BertLMPredictionHead  # 导入 BERT 模型相关的类
from transformers.models.t5.modeling_t5 import T5Block  # 导入 T5 模型相关的类


def setup(rank, world_size):
    # 设置环境变量
    os.environ["MASTER_ADDR"] = os.getenv("MASTER_ADDR", "localhost")  # 主节点地址，默认为 localhost
    os.environ["MASTER_PORT"] = os.getenv("MASTER_PORT", "12355")     # 主节点端口，默认为 12355
    os.environ["RANK"] = os.getenv("RANK", "0")                       # 当前进程的排名，默认为 0
    os.environ["WORLD_SIZE"] = os.getenv("WORLD_SIZE", "1")           # 分布式训练中的进程总数，默认为 1
    dist.init_process_group("nccl")  # 使用 NCCL 初始化进程组


def cleanup():
    dist.destroy_process_group()  # 清理并销毁进程组


class CustomLinear(torch.nn.Module):
    def __init__(self, a, b):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(a, b))  # 定义一个可学习的参数 weight

    def forward(self, x):
        return torch.mm(x, self.weight)  # 前向传播函数，执行矩阵乘法操作


class MyModule(torch.nn.Module):
    def __init__(self, a, b):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(a, b),  # 添加一个线性层
            nn.ReLU(),         # 添加 ReLU 激活函数
        )

    def forward(self, x):
        return self.net(x)  # 前向传播函数，通过网络执行输入到输出的转换


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            *[nn.Linear(10, 10000), nn.ReLU()]  # 添加线性层和 ReLU 激活函数
            + [nn.Linear(10000, 10000), nn.ReLU()]  # 继续添加线性层和 ReLU 激活函数
            + [MyModule(10000, 10000)]  # 添加自定义模块 MyModule
            + [MyModule(10000, 1000)]   # 继续添加自定义模块 MyModule
            + [MyModule(1000, 1000)]    # 继续添加自定义模块 MyModule
            + [MyModule(1000, 1000)]    # 继续添加自定义模块 MyModule
            + [MyModule(1000, 1000)]    # 继续添加自定义模块 MyModule
            + [MyModule(1000, 1000)]    # 继续添加自定义模块 MyModule
            + [MyModule(1000, 1000)]    # 继续添加自定义模块 MyModule
            + [MyModule(1000, 1000)]    # 继续添加自定义模块 MyModule
            + [MyModule(1000, 1000)]    # 继续添加自定义模块 MyModule
            + [nn.Linear(1000, 5)]      # 最后添加线性层
        )

    def forward(self, x):
        return self.net(x)  # 前向传播函数，通过网络执行输入到输出的转换


def model_iter_fn(model, example_inputs, collect_outputs=False):
    outputs = model(*example_inputs)  # 使用模型进行前向传播
    loss = reduce_to_scalar_loss(outputs)  # 将输出降维到标量损失
    loss.backward()  # 计算损失的梯度
    if collect_outputs:
        return outputs  # 如果需要收集输出，则返回输出结果


def get_model(args):
    if args.torchbench_model:
        old_cwd = setup_torchbench_cwd()  # 设置 TorchBench 的当前工作目录
        module = importlib.import_module(
            f"torchbenchmark.models.{args.torchbench_model}"
        )  # 动态导入指定的 TorchBench 模型
        benchmark_cls = getattr(module, "Model", None)  # 获取模型类
        bm = benchmark_cls(test="train", device=args.device, batch_size=args.batch_size)  # 创建 TorchBench 模型实例
        model, inputs = bm.get_module()  # 获取 TorchBench 模型和输入数据
    elif args.toy_model:
        model = ToyModel()  # 创建 ToyModel 实例
        inputs = (torch.randn(20, 10),)  # 创建输入数据
    else:
        # 如果未提供模型参数，则抛出参数错误异常
        raise argparse.ArgumentError(
            args.torchbench_model, message="Must specify a model"
        )
    
    # 返回模型和输入数据
    return model, inputs
# 定义函数，为模型应用激活检查点技术
# 直接更新模型，因此返回None
def fsdp_checkpointing_base(model, blocks):
    # 使用 functools.partial 创建一个非重入包装器，用于激活检查点
    non_reentrant_wrapper = functools.partial(
        checkpoint_wrapper,
        offload_to_cpu=False,
        checkpoint_impl=CheckpointImpl.NO_REENTRANT,
    )

    # 定义用于检查子模块是否属于指定类型 blocks 的函数
    def check_fn(submodule):
        return isinstance(submodule, blocks)

    # 调用 apply_activation_checkpointing 函数，为模型应用激活检查点
    apply_activation_checkpointing(
        model, checkpoint_wrapper_fn=non_reentrant_wrapper, check_fn=check_fn
    )


# 定义模型与包装策略之间的映射关系
MODEL_FSDP_WRAP = {
    "toy_model": (MyModule,),
    "hf_Bert": (BertLayer, BertLMPredictionHead),
    "hf_T5": (T5Block,),
}


# 应用 FSDP 策略于给定的模型
def apply_fsdp(args, model, use_checkpointing=False, use_wrap_policy=True):
    wrap_policy = None
    # 根据模型的类型选择相应的 blocks 类型
    blocks = MODEL_FSDP_WRAP[
        "toy_model" if model.__class__ is ToyModel else args.torchbench_model
    ]
    # 如果使用包装策略，则创建 ModuleWrapPolicy 对象
    if use_wrap_policy:
        wrap_policy = ModuleWrapPolicy(blocks)

    # 应用 FSDP 策略于模型，使用原始参数
    model = FSDP(model, auto_wrap_policy=wrap_policy, use_orig_params=True)
    # 如果需要使用检查点技术，则调用 fsdp_checkpointing_base 函数
    if use_checkpointing:
        fsdp_checkpointing_base(model, blocks)
    # 返回经过 FSDP 策略处理后的模型
    return model
```