# `.\pytorch\test\distributed\_spmd\test_data_parallel.py`

```
# Owner(s): ["oncall: distributed"]

# 导入必要的模块和类
from copy import deepcopy  # 导入深拷贝函数
import torch  # 导入PyTorch库
import torch.nn as nn  # 导入神经网络模块

# 导入分布式相关的模块和函数
from torch.distributed._spmd.api import compile  # 导入编译函数
from torch.distributed._spmd.parallel_mode import DataParallel  # 导入数据并行模块
from torch.distributed._tensor import Replicate  # 导入复制操作相关类
from torch.nn.parallel import DistributedDataParallel as DDP  # 导入分布式数据并行类
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu  # 导入条件跳过测试函数
from torch.testing._internal.common_utils import run_tests  # noqa: TCH001

# 导入分布式相关测试工具和类
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)

# 定义一个简单的多层感知机模型类
class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net1 = nn.Linear(50, 32)  # 第一个全连接层
        self.relu = nn.ReLU()  # ReLU 激活函数
        self.net2 = nn.Linear(32, 8)  # 第二个全连接层

    def forward(self, x):
        return torch.sigmoid(self.net2(self.relu(self.net1(x))))  # 前向传播函数

    def reset_parameters(self, *args, **kwargs):
        self.net1.reset_parameters()  # 重置第一个全连接层的参数
        self.net2.reset_parameters()  # 重置第二个全连接层的参数

# 简单的训练步骤函数，用于示例
def train_step(model, optim, train_batch):
    def loss_fn(out, labels):
        return (out - labels).sum()  # 计算损失函数

    optim.zero_grad()  # 梯度清零
    inputs, labels = train_batch  # 获取输入和标签

    out = model(inputs)  # 前向传播
    loss = loss_fn(out, labels)  # 计算损失
    loss.backward()  # 反向传播
    optim.step()  # 更新优化器参数
    return loss  # 返回损失值

# 测试数据并行的类，继承自 DTensorTestBase
class TestDataParallel(DTensorTestBase):
    @property
    def world_size(self):
        return 2  # 设置测试的世界大小为2

    # 测试数据并行函数
    def _test_data_parallel(
        self,
        mod,
        ddp_mod,
        opt,
        ddp_opt,
        inp,
        train_step,
        data_parallel_mode,
        data_parallel_options=None,
    ):
        ddp_inp = deepcopy(inp)  # 深拷贝输入数据

        # 需要一步来热身优化器
        train_step(mod, opt, inp)
        opt.zero_grad()  # 优化器梯度清零

        # DDP 运行完整的训练步骤一次，以便与热身对齐
        train_step(ddp_mod, ddp_opt, ddp_inp)
        ddp_opt.zero_grad()  # DDP优化器梯度清零

        # 手动训练一次 DDP 模型，因为 DDP 的梯度不同
        torch.sum(ddp_mod(ddp_inp[0]) - ddp_inp[1]).backward()  # 计算梯度
        ddp_opt.step()  # 更新优化器参数

        # 使用复制操作编译并运行一次步骤
        data_parallel_options = data_parallel_options or {}
        compiled_fn = compile(
            parallel_mode=DataParallel(data_parallel_mode, **data_parallel_options)
        )(train_step)  # 编译并运行步骤
        compiled_fn(mod, opt, inp)  # 执行编译后的函数

        # 比较模型参数
        for p1, p2 in zip(mod.parameters(), ddp_mod.parameters()):
            # 如果数据并行模式是 "fully_shard"，则将 p1 转换为本地张量后再进行比较
            if data_parallel_mode == "fully_shard":
                # 收集分片以进行比较
                p1_replica = p1.redistribute(placements=[Replicate()])
                p1_local_param = p1_replica.to_local()
            else:
                p1_local_param = p1.to_local()
            self.assertEqual(p1_local_param, p2)  # 断言两个模型参数相等

    @skip_if_lt_x_gpu(2)  # 如果 GPU 数量少于 2，则跳过测试
    @with_comms
    @skip_if_lt_x_gpu(2)
    @with_comms
    def test_replicate_sgd(self):
        # 定义一系列 SGD 配置列表，包括不同的学习率和动量选项
        sgd_configs = [
            {"lr": 0.1},
            {"lr": 0.1, "momentum": 0.9},
            {"lr": 0.1, "momentum": 0.9, "foreach": True},
        ]

        # 遍历每个配置
        for config in sgd_configs:
            # 创建一个在 CUDA 设备上运行的简单 MLP 模型实例
            mod = SimpleMLP().cuda(self.rank)
            # 使用当前配置创建 SGD 优化器
            opt = torch.optim.SGD(mod.parameters(), **config)

            # 创建一个随机的训练批次，指定在当前设备上
            train_batch = (
                torch.randn((128, 50), device=torch.device(self.rank)),
                torch.randn((128, 8), device=torch.device(self.rank)),
            )

            # 使用深拷贝创建 DDP 模型，设备 ID 为当前设备
            ddp_mod = DDP(deepcopy(mod), device_ids=[self.rank])
            # 使用当前配置创建 DDP SGD 优化器
            ddp_opt = torch.optim.SGD(ddp_mod.parameters(), **config)

            # 执行数据并行测试，比较模型、优化器和训练批次在单GPU和分布式数据并行模式下的表现
            self._test_data_parallel(
                mod, ddp_mod, opt, ddp_opt, train_batch, train_step, "replicate"
            )

    @skip_if_lt_x_gpu(2)
    @with_comms
    def test_replicate_adam_fused(self):
        # 创建一个在 CUDA 设备上运行的简单 MLP 模型实例
        mod = SimpleMLP().cuda(self.rank)
        # 使用 Adam 优化器，启用 fused 参数，学习率为 0.1
        opt = torch.optim.Adam(mod.parameters(), lr=0.1, fused=True)

        # 创建一个随机的训练批次，指定在当前设备上
        train_batch = (
            torch.randn((128, 50), device=torch.device(self.rank)),
            torch.randn((128, 8), device=torch.device(self.rank)),
        )

        # 使用深拷贝创建 DDP 模型，设备 ID 为当前设备
        ddp_mod = DDP(deepcopy(mod), device_ids=[self.rank])
        # 使用 Adam 优化器，启用 fused 参数，学习率为 0.1
        ddp_opt = torch.optim.Adam(ddp_mod.parameters(), lr=0.1, fused=True)

        # 执行数据并行测试，比较模型、优化器和训练批次在单GPU和分布式数据并行模式下的表现
        self._test_data_parallel(
            mod, ddp_mod, opt, ddp_opt, train_batch, train_step, "replicate"
        )

    @skip_if_lt_x_gpu(2)
    @with_comms
    def test_fully_shard_sgd(self):
        # 定义一系列 SGD 配置列表，包括不同的学习率和动量选项
        sgd_configs = [
            {"lr": 0.1},
            {"lr": 0.1, "momentum": 0.9},
            {"lr": 0.1, "momentum": 0.9, "foreach": True},
        ]

        # 遍历每个配置
        for config in sgd_configs:
            # 创建一个在 CUDA 设备上运行的简单 MLP 模型实例
            mod = SimpleMLP().cuda(self.rank)
            # 使用当前配置创建 SGD 优化器
            opt = torch.optim.SGD(mod.parameters(), **config)

            # 创建一个随机的训练批次，指定在当前设备上
            train_batch = (
                torch.randn((128, 50), device=torch.device(self.rank)),
                torch.randn((128, 8), device=torch.device(self.rank)),
            )

            # 使用深拷贝创建 DDP 模型，设备 ID 为当前设备
            ddp_mod = DDP(deepcopy(mod), device_ids=[self.rank])
            # 使用当前配置创建 DDP SGD 优化器
            ddp_opt = torch.optim.SGD(ddp_mod.parameters(), **config)

            # 执行数据并行测试，比较模型、优化器和训练批次在单GPU和分布式数据并行模式下的表现
            self._test_data_parallel(
                mod, ddp_mod, opt, ddp_opt, train_batch, train_step, "fully_shard"
            )

    @skip_if_lt_x_gpu(2)
    @with_comms
    def test_fully_shard_adam_fused(self):
        # 创建一个在 CUDA 设备上运行的简单 MLP 模型实例
        mod = SimpleMLP().cuda(self.rank)
        # 使用 Adam 优化器，启用 fused 参数，学习率为 0.1
        opt = torch.optim.Adam(mod.parameters(), lr=0.1, fused=True)

        # 创建一个随机的训练批次，指定在当前设备上
        train_batch = (
            torch.randn((128, 50), device=torch.device(self.rank)),
            torch.randn((128, 8), device=torch.device(self.rank)),
        )

        # 使用深拷贝创建 DDP 模型，设备 ID 为当前设备
        ddp_mod = DDP(deepcopy(mod), device_ids=[self.rank])
        # 使用 Adam 优化器，启用 fused 参数，学习率为 0.1
        ddp_opt = torch.optim.Adam(ddp_mod.parameters(), lr=0.1, fused=True)

        # 执行数据并行测试，比较模型、优化器和训练批次在单GPU和分布式数据并行模式下的表现
        self._test_data_parallel(
            mod, ddp_mod, opt, ddp_opt, train_batch, train_step, "fully_shard"
        )

    @skip_if_lt_x_gpu(2)
    @with_comms
    def test_data_parallel_batch_dim_analysis(self):
        # test batch dim analysis by adding a few ops that changes
        # the batch dim in non-trival ways
        
        # 定义一个包裹模块，继承自 nn.Module
        class WrapperModule(nn.Module):
            def __init__(self, *args, **kwargs) -> None:
                super().__init__(*args, **kwargs)
                # 创建一个 SimpleMLP 实例作为模块的子模块
                self.mlp = SimpleMLP()

            def forward(self, x):
                # 前向传播函数
                output = self.mlp(x)
                # 克隆输出并展平为一维向量
                new_output = output.clone().view(-1)
                # 压缩张量维度
                squeezed_out = new_output.squeeze()
                # 展开张量维度
                unsqueezed_out = squeezed_out.unsqueeze(0)
                # 对输出进行操作，增加一个较小的张量到展平的输出上
                output = output + 0.1 * unsqueezed_out.view(output.shape[0], -1)

                # 使用数据并行扩展测试工厂操作
                arange = torch.arange(output.shape[1], device=output.device)
                ones = torch.ones(output.shape, device=output.device)
                added_arange = arange.unsqueeze(0) + ones

                # 测试重复逻辑
                zeros = torch.zeros(output.shape[1], device=output.device)
                repeated_zeros = zeros.unsqueeze(0).repeat(output.shape[0], 1)

                # 对输出进行进一步的操作，结合上述的操作结果
                output = output + added_arange + repeated_zeros

                return output

        # 使用两种并行模式进行测试
        for parallel_mode in ["replicate", "fully_shard"]:
            # 在指定的 GPU 设备上创建 WrapperModule 实例
            mod = WrapperModule().cuda(self.rank)
            # 使用 SGD 优化器
            opt = torch.optim.SGD(mod.parameters(), lr=0.1)

            # 创建训练批次数据
            train_batch = (
                torch.randn((128, 50), device=torch.device(self.rank)),
                torch.randn((128, 8), device=torch.device(self.rank)),
            )

            # 使用分布式数据并行（DDP）模式
            ddp_mod = DDP(deepcopy(mod), device_ids=[self.rank])
            ddp_opt = torch.optim.SGD(ddp_mod.parameters(), lr=0.1)

            # 调用测试函数以验证数据并行处理
            self._test_data_parallel(
                mod, ddp_mod, opt, ddp_opt, train_batch, train_step, parallel_mode
            )

    @skip_if_lt_x_gpu(2)
    @with_comms
    def test_fully_shard_non_0_batch_dim(self):
        # 定义一个包裹模块，继承自 nn.Module
        class WrapperModule(nn.Module):
            def __init__(self):
                super().__init__()
                # 创建一个 SimpleMLP 实例作为模块的子模块
                self.mlp = SimpleMLP()

            def forward(self, x):
                # 前向传播函数，对输入进行转置和连续化操作
                reshaped_x = x.t().contiguous()
                return self.mlp(reshaped_x).t()

        # 在指定的 GPU 设备上创建 WrapperModule 实例
        mod = WrapperModule().cuda(self.rank)
        # 使用 Adam 优化器，并启用融合（fused）操作
        opt = torch.optim.Adam(mod.parameters(), lr=0.1, fused=True)

        # 创建训练批次数据
        train_batch = (
            torch.randn((50, 128), device=torch.device(self.rank)),
            torch.randn((8, 128), device=torch.device(self.rank)),
        )

        # 使用分布式数据并行（DDP）模式
        ddp_mod = DDP(deepcopy(mod), device_ids=[self.rank])
        ddp_opt = torch.optim.Adam(ddp_mod.parameters(), lr=0.1, fused=True)

        # 调用测试函数以验证数据并行处理，使用全分片模式（fully_shard）
        self._test_data_parallel(
            mod,
            ddp_mod,
            opt,
            ddp_opt,
            train_batch,
            train_step,
            "fully_shard",
            {"input_batch_dim": 1},
        )
# 如果当前脚本作为主程序执行（而不是被导入其他模块），则执行以下代码块
if __name__ == "__main__":
    # 判断条件永远为假，因此不会执行这个分支下的代码
    if False:
        # 调用一个名为 run_tests() 的函数，用于运行测试（但由于条件为 False，实际上不会执行）
        run_tests()
```