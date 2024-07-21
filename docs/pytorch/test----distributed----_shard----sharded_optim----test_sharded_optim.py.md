# `.\pytorch\test\distributed\_shard\sharded_optim\test_sharded_optim.py`

```
# Owner(s): ["oncall: distributed"]

from copy import deepcopy  # 导入深拷贝函数

import torch  # 导入PyTorch库
import torch.optim as optim  # 导入PyTorch优化器模块
from torch.distributed._shard import shard_parameter, sharded_tensor  # 导入分片参数和分片张量模块
from torch.distributed._shard.sharded_optim import ShardedOptimizer  # 导入分片优化器模块
from torch.distributed._shard.sharding_spec import ChunkShardingSpec  # 导入分片规格模块
from torch.testing._internal.common_distributed import requires_nccl, skip_if_lt_x_gpu  # 导入测试相关模块和装饰器
from torch.testing._internal.common_utils import run_tests  # 导入测试运行函数

from torch.testing._internal.distributed._shard.sharded_tensor import (
    ShardedTensorTestBase,  # 导入分片张量测试基类
    with_comms,  # 导入通信上下文装饰器
)


class MyShardedModel(torch.nn.Module):
    def __init__(self, spec=None, group=None):
        super().__init__()
        # 设置随机种子
        torch.manual_seed(0)
        self.param = torch.nn.Parameter(torch.rand(5, 10))  # 创建参数张量
        if spec is not None:
            # 创建分片参数张量，根据规格和进程组
            self.sharded_param = torch.nn.Parameter(
                sharded_tensor.rand(
                    spec, 20, 10, requires_grad=True, process_group=group
                )
            )
        else:
            self.sharded_param = torch.nn.Parameter(torch.rand(5, 10))  # 创建普通参数张量

    def forward(self, input):
        if isinstance(self.sharded_param, sharded_tensor.ShardedTensor):
            # 如果是分片张量，则返回参数、本地分片的张量和输入的和
            return self.param + self.sharded_param.local_shards()[0].tensor + input
        else:
            return self.sharded_param + self.param + input  # 否则返回参数、本身和输入的和


class MyShardedLinear(torch.nn.Module):
    def __init__(self, rank=None):
        super().__init__()
        # 设置随机种子
        torch.manual_seed(0)
        self.linear1 = torch.nn.Linear(17, 12)  # 创建线性层1
        self.linear2 = torch.nn.Linear(12, 29)  # 创建线性层2
        self.gelu = torch.nn.GELU()  # 创建GELU激活函数

        if rank:
            self.linear1.cuda(rank)  # 将线性层1放置在指定的GPU上
            self.linear2.cuda(rank)  # 将线性层2放置在指定的GPU上

    def shard_parameter(self):
        # 按行切片规格
        rowwise_sharding_spec = ChunkShardingSpec(
            dim=0,
            placements=[
                "rank:0/cuda:0",
                "rank:1/cuda:1",
                "rank:2/cuda:2",
                "rank:3/cuda:3",
            ],
        )

        # 按列切片规格
        colwise_sharding_spec = ChunkShardingSpec(
            dim=1,
            placements=[
                "rank:0/cuda:0",
                "rank:1/cuda:1",
                "rank:2/cuda:2",
                "rank:3/cuda:3",
            ],
        )

        shard_parameter(self.linear1, "weight", rowwise_sharding_spec)  # 对线性层1的权重进行切片
        shard_parameter(self.linear2, "weight", colwise_sharding_spec)  # 对线性层2的权重进行切片

    def forward(self, inp):
        return self.linear2(self.gelu(self.linear1(inp)))  # 神经网络正向传播过程


class TestShardedOptimizer(ShardedTensorTestBase):
    @with_comms(init_rpc=False)  # 使用通信上下文装饰器，初始化RPC
    @skip_if_lt_x_gpu(4)  # 如果GPU数少于4，则跳过测试
    @requires_nccl()  # 需要NCCL库支持
    def test_sharded_optim(self):
        # 定义行分片规范，按照第0维度进行分片
        rowwise_spec = ChunkShardingSpec(
            dim=0,
            placements=[
                "rank:0/cuda:0",
                "rank:1/cuda:1",
                "rank:2/cuda:2",
                "rank:3/cuda:3",
            ],
        )
        # 创建本地模型，并移到GPU上
        local_model = MyShardedModel().cuda()
        # 创建分片模型，并根据分片规范移到GPU上
        sharded_model = MyShardedModel(spec=rowwise_spec).cuda()

        # 从本地模型复制参数到分片模型的本地分片
        sharded_model.sharded_param.local_shards()[0].tensor = (
            local_model.sharded_param.detach().clone().requires_grad_()
        )

        # 创建本地优化器，针对本地模型的参数
        local_optim = optim.SGD(local_model.parameters(), lr=0.1)
        # 获取分片模型的所有参数，并用它们创建分片优化器
        sharded_model_params = dict(sharded_model.named_parameters())
        sharded_optim = ShardedOptimizer(sharded_model_params, optim.SGD, lr=0.1)

        # 将本地优化器和分片优化器的梯度清零
        local_optim.zero_grad()
        sharded_optim.zero_grad()

        # 复制优化前的分片参数状态
        before_update = deepcopy(sharded_optim.named_params)

        # 创建一个随机输入，并移到当前 GPU（self.rank）
        inp = torch.rand([5, 10]).cuda(self.rank).requires_grad_()

        # 运行前向传播
        local_output = local_model(inp)
        sharded_output = sharded_model(inp)
        # 反向传播
        local_output.sum().backward()
        sharded_output.sum().backward()

        # 更新优化器
        local_optim.step()
        sharded_optim.step()

        # 确保参数（包括分片参数）被优化器更新，并且更新后的本地参数与分片参数相同
        for key, val in before_update.items():
            new_val = sharded_optim.named_params[key]
            if isinstance(val, sharded_tensor.ShardedTensor):
                self.assertNotEqual(
                    val.local_shards()[0].tensor, new_val.local_shards()[0].tensor
                )
                self.assertEqual(
                    new_val.local_shards()[0].tensor, local_model.sharded_param
                )
            else:
                self.assertNotEqual(val, new_val)
                self.assertEqual(new_val, local_model.param)

    @with_comms(init_rpc=False)
    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    # 测试使用具名参数和分片张量
    def test_named_params_with_sharded_tensor(self):
        # 定义行分片规范
        rowwise_spec = ChunkShardingSpec(
            dim=0,
            placements=[
                "rank:0/cuda:0",
                "rank:1/cuda:1",
                "rank:2/cuda:2",
                "rank:3/cuda:3",
            ],
        )
        # 创建具有行分片规范的分片模型，并将其移到 GPU 上
        sharded_model = MyShardedModel(spec=rowwise_spec).cuda()
        # 获取分片模型的参数字典
        sharded_model_params = dict(sharded_model.named_parameters())
        # 获取参数字典的键列表
        param_keys = list(sharded_model_params.keys())
        # 断言参数字典的长度为2
        self.assertEqual(len(param_keys), 2)
        # 断言参数字典中包含"param"键
        self.assertTrue("param" in param_keys)
        # 断言参数字典中包含"sharded_param"键

        self.assertTrue("sharded_param" in param_keys)

        # 创建具有给定等级的分片线性模型，并将其移到 GPU 上
        sharded_linear = MyShardedLinear(rank=self.rank).cuda()
        # 对分片线性模型进行参数分片
        sharded_linear.shard_parameter()
        # 获取分片线性模型的参数字典
        sharded_linear_params = dict(sharded_linear.named_parameters())
        # 获取参数字典的键列表
        param_keys = list(sharded_linear_params.keys())
        # 断言参数字典的长度为4
        self.assertEqual(len(param_keys), 4)
        # 断言参数字典中包含"linear1.bias"键
        self.assertTrue("linear1.bias" in param_keys)
        # 断言参数字典中包含"linear2.bias"键
        self.assertTrue("linear2.bias" in param_keys)
        # 断言参数字典中包含"linear1.weight"键
        self.assertTrue("linear1.weight" in param_keys)
        # 断言参数字典中包含"linear2.weight"键
        self.assertTrue("linear2.weight" in param_keys)
        # 断言参数字典中不包含"bias"键
        self.assertFalse("bias" in param_keys)
# 如果当前脚本作为主程序执行（而非被导入其他模块），则执行 run_tests 函数
if __name__ == "__main__":
    run_tests()
```