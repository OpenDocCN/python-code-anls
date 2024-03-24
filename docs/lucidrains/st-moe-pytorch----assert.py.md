# `.\lucidrains\st-moe-pytorch\assert.py`

```
# 导入必要的库
import os
from copy import deepcopy
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from st_moe_pytorch.st_moe_pytorch import Experts, Expert
from st_moe_pytorch.distributed import all_gather_variable_dim

# 设置初始化函数，用于初始化分布式训练环境
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

# 清理函数，用于销毁进程组
def cleanup():
    dist.destroy_process_group()

# 主函数，用于启动分布式训练
def start(
    rank,
    world_size,
    batch_size,
    batch_size_var_len,
    num_experts,
    tokens_per_expert,
    dim,
    use_cuda
):
    # 初始化分布式训练环境
    setup(rank, world_size)

    # 创建专家网络
    net = Experts([Expert(dim) for _ in range(num_experts)])

    # 根据是否变长批次设置批次大小
    if batch_size_var_len:
        batch_size = batch_size + rank

    # 生成随机输入序列
    seq = torch.randn(batch_size, num_experts, tokens_per_expert, dim)

    # 本地计算

    # 深拷贝专家网络
    local_net = deepcopy(net)

    # 聚合所有进程的输入数据
    local_inputs, _ = all_gather_variable_dim(seq)

    # 在本地网络上进行前向传播
    local_out = local_net(
        local_inputs,
        is_distributed=False
    )

    # 计算本地输出的均值并进行反向传播
    local_out.mean().backward()

    # 分布式计算

    # 使用分布式数据并行模型
    model = DDP(net)
    ddp_inputs = seq

    # 如果使用CUDA，则将模型和输入数据移动到对应设备
    if use_cuda:
        model.cuda(rank)
        ddp_inputs = seq.cuda(rank)

    # 在分布式模型上进行前向传播
    out = model(ddp_inputs)
    out.mean().backward()

    # 聚合所有进程的输出数据
    ddp_all_out, _ = all_gather_variable_dim(out)

    if rank == 0:
        # 验证本地和分布式输出是否一致

        # 将模型和输出数据移回CPU
        model.cpu()
        ddp_all_out.cpu()

        # 使用assert检查本地和分布式输出是否一致
        assert torch.allclose(local_out, ddp_all_out.cpu(), atol=1e-3), 'output is not the same'

        # 验证本地和分布式第一个专家的梯度是否一致

        # 定义获取第一个专家梯度的函数
        get_first_expert_grad = lambda t: t.experts[0].net[0].weight.grad

        # 使用assert检查本地和分布式第一个专家的梯度是否一致
        assert torch.allclose(
            get_first_expert_grad(net).cpu(),
            get_first_expert_grad(local_net),
            atol=1e-2
        ), 'grad is not the same'

        # 输出验证结果
        print('✅ outputs and gradients are same between local and ddp')

    # 清理环境
    cleanup()

# 主程序入口
if __name__ == '__main__':
    # 设置参数
    world_size = 8
    num_experts = 3
    batch_size = 2
    batch_size_var_len = True
    use_cuda = False

    # 检查是否使用CUDA并且设备数量小于等于进程数量
    assert not use_cuda or torch.cuda.device_count() <= world_size

    seq_len = 32
    dim = 8

    # 使用多进程启动分布式训练
    mp.spawn(
        start,
        args=(
            world_size,
            batch_size,
            batch_size_var_len,
            num_experts,
            seq_len,
            dim,
            use_cuda
        ),
        nprocs=world_size,
        join=True
    )
```