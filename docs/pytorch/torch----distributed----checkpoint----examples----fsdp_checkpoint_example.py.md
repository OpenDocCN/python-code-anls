# `.\pytorch\torch\distributed\checkpoint\examples\fsdp_checkpoint_example.py`

```
# mypy: allow-untyped-defs
# Copyright (c) Meta Platforms, Inc. and affiliates

"""
The following example demonstrates how to use Pytorch Distributed Checkpoint to save a FSDP model.

This is the current recommended way to checkpoint FSDP.
torch.save() and torch.load() is not recommended when checkpointing sharded models.
"""

import os
import shutil

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dist_cp
import torch.multiprocessing as mp
from torch.distributed.checkpoint.optimizer import load_sharded_optimizer_state_dict
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType

CHECKPOINT_DIR = f"/scratch/{os.environ['LOGNAME']}/checkpoint"

# 定义一个函数，用于返回优化器状态的指定索引的值
def opt_at(opt, idx):
    return list(opt.state.values())[idx]

# 初始化模型和优化器
def init_model():
    # 创建一个包含 FSDP 的线性模型，部署到当前进程的 CUDA 设备上
    model = FSDP(torch.nn.Linear(4, 4).cuda(dist.get_rank()))
    # 使用 Adam 优化器初始化模型参数
    optim = torch.optim.Adam(model.parameters(), lr=0.1)
    # 对模型进行前向传播、反向传播和优化步骤
    model(torch.rand(4, 4)).sum().backward()
    optim.step()

    return model, optim

# 打印模型参数
def print_params(stage, model_1, model_2, optim_1, optim_2):
    # 使用 FSDP 来打印模型参数
    with FSDP.summon_full_params(model_1):
        with FSDP.summon_full_params(model_2):
            print(
                f"{stage} --- rank: {dist.get_rank()}\n"
                f"model.weight: {model_1.weight}\n"
                f"model_2.weight:{model_2.weight}\n"
                f"model.bias: {model_1.bias}\n"
                f"model_2.bias: {model_2.bias}\n"
            )

    # 打印优化器的状态信息
    print(
        f"{stage} --- rank: {dist.get_rank()}\n"
        f"optim exp_avg:{opt_at(optim_1, 0)['exp_avg']}\n"
        f"optim_2 exp_avg:{opt_at(optim_2, 0)['exp_avg']}\n"
        f"optim exp_avg_sq:{opt_at(optim_1, 0)['exp_avg_sq']}\n"
        f"optim_2 exp_avg_sq:{opt_at(optim_2, 0)['exp_avg_sq']}\n"
    )

# 运行 FSDP 模型检查点示例
def run_fsdp_checkpoint_example(rank, world_size):
    # 设置分布式训练的环境变量
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # 初始化进程组
    dist.init_process_group("cpu:gloo,cuda:nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    # 创建第一个模型及其优化器
    model_1, optim_1 = init_model()

    # 将第一个模型保存到 CHECKPOINT_DIR
    with FSDP.state_dict_type(model_1, StateDictType.SHARDED_STATE_DICT):
        state_dict = {
            "model": model_1.state_dict(),
            "optim": FSDP.optim_state_dict(model_1, optim_1),
        }

        dist_cp.save_state_dict(
            state_dict=state_dict,
            storage_writer=dist_cp.FileSystemWriter(CHECKPOINT_DIR),
        )

    # 创建第二个模型及其优化器
    model_2, optim_2 = init_model()

    # 打印两个模型的参数。在加载之前，这些参数应该是不同的。
    print_params("Before loading", model_1, model_2, optim_1, optim_2)

    # 加载 CHECKPOINT_DIR 中保存的模型参数到 model_2
    # 使用 FSDP 状态管理器加载模型2的状态字典，指定为 SHARDED_STATE_DICT 类型
    with FSDP.state_dict_type(model_2, StateDictType.SHARDED_STATE_DICT):
        # 构建模型的状态字典，只包括模型的状态，不包括优化器的状态
        state_dict = {
            "model": model_2.state_dict(),
            # 无法同时加载模型状态字典和优化器状态字典
        }

        # 使用分布式检查点 (dist_cp) 加载模型的状态字典
        dist_cp.load_state_dict(
            state_dict=state_dict,
            storage_reader=dist_cp.FileSystemReader(CHECKPOINT_DIR),
        )
        # 从加载的状态字典中恢复模型2的状态
        model_2.load_state_dict(state_dict["model"])

        # 加载分片优化器状态字典
        optim_state = load_sharded_optimizer_state_dict(
            model_state_dict=state_dict["model"],
            optimizer_key="optim",
            storage_reader=dist_cp.FileSystemReader(CHECKPOINT_DIR),
        )

        # 将优化器状态字典扁平化以便加载
        flattened_osd = FSDP.optim_state_dict_to_load(
            model_2, optim_2, optim_state["optim"]
        )
        # 加载扁平化后的优化器状态字典到优化器2
        optim_2.load_state_dict(flattened_osd)

    # 打印两个模型的参数，用于确认加载后两个模型的参数一致
    print_params("After loading", model_1, model_2, optim_1, optim_2)

    # 关闭分布式进程组，结束分布式训练
    dist.destroy_process_group()
# 如果当前脚本作为主程序运行
if __name__ == "__main__":
    # 获取可用的 CUDA 设备数量
    world_size = torch.cuda.device_count()
    # 打印运行的 fsdp checkpoint 示例将在多少个设备上运行
    print(f"Running fsdp checkpoint example on {world_size} devices.")
    # 清空指定目录下的所有文件和文件夹，如果目录不存在则忽略错误
    shutil.rmtree(CHECKPOINT_DIR, ignore_errors=True)
    # 使用多进程方式在多个设备上运行指定的函数
    mp.spawn(
        run_fsdp_checkpoint_example,  # 要运行的函数
        args=(world_size,),           # 函数的参数元组
        nprocs=world_size,            # 使用的进程数量
        join=True,                    # 等待所有进程结束后再继续执行
    )
```