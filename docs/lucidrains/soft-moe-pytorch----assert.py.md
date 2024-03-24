# `.\lucidrains\soft-moe-pytorch\assert.py`

```
# 导入必要的库
import os
from copy import deepcopy

import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from soft_moe_pytorch.soft_moe import Experts, FeedForward as Expert
from soft_moe_pytorch.distributed import all_gather_variable_dim

# 设置初始化函数，用于初始化分布式进程组
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

# 清理函数，用于销毁进程组
def cleanup():
    dist.destroy_process_group()

# 主函数，启动分布式训练
def start(
    rank,
    world_size,
    batch_size,
    batch_size_var_len,
    num_experts,
    tokens_per_expert,
    dim,
):
    # 初始化分布式进程组
    setup(rank, world_size)

    # 创建专家网络
    net = Experts([Expert(dim) for _ in range(num_experts)])

    # 根据是否变长批次设置批次大小
    if batch_size_var_len:
        batch_size = batch_size + rank

    # 生成随机输入序列
    seq = torch.randn(batch_size, num_experts, tokens_per_expert, dim)

    # 分布式训练

    # 使用分布式数据并行包装模型
    model = DDP(net)
    out = model(seq)
    out.mean().backward()

    # 所有进程收集输出
    ddp_all_out, _ = all_gather_variable_dim(out)

    # 单设备上

    # 所有进程收集输入
    all_inputs, _ = all_gather_variable_dim(seq)
    copied_net = deepcopy(net)

    # 在单设备上进行前向传播
    single_out = copied_net(
        all_inputs,
        is_distributed=False
    )

    single_out.mean().backward()

    if rank == 0:
        # 验证输出是否相同
        # 如果在单台机器上和多台机器上进行

        assert torch.allclose(single_out, ddp_all_out), 'output is not the same'

        # 验证梯度和grad是否相同

        get_first_expert_grad = lambda t: t.experts[0][0].weight.grad

        assert torch.allclose(
            get_first_expert_grad(net),
            get_first_expert_grad(copied_net),
            atol=1e-2
        ), 'grad is not the same'

        print('✅')

    # 清理进程组
    cleanup()

if __name__ == '__main__':
    # 设置参数
    world_size = 9
    num_experts = 8
    batch_size = 2
    batch_size_var_len = False

    seq_len = 32
    dim = 8

    # 多进程启动
    mp.spawn(
        start,
        args=(
            world_size,
            batch_size,
            batch_size_var_len,
            num_experts,
            seq_len,
            dim
        ),
        nprocs=world_size,
        join=True
    )
```