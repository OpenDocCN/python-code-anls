# `.\pytorch\test\distributed\fsdp\test_fsdp_multiple_forward.py`

```
# Owner(s): ["oncall: distributed"]

import sys  # 导入 sys 模块，用于处理系统相关的功能

import torch  # 导入 PyTorch 库
from torch import distributed as dist  # 导入 PyTorch 分布式模块
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP  # 导入 FSDP 模块
from torch.nn import Linear, Module  # 导入 PyTorch 中的线性层和模块基类
from torch.nn.parallel import DistributedDataParallel  # 导入 PyTorch 分布式数据并行模块
from torch.optim import SGD  # 导入 PyTorch SGD 优化器
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu  # 导入测试工具函数
from torch.testing._internal.common_fsdp import FSDPTest, get_full_params  # 导入 FSDP 测试相关函数和参数获取函数
from torch.testing._internal.common_utils import run_tests, TEST_WITH_DEV_DBG_ASAN  # 导入测试运行函数和调试环境标记

if not dist.is_available():  # 如果分布式环境不可用
    print("Distributed not available, skipping tests", file=sys.stderr)  # 输出信息到标准错误流
    sys.exit(0)  # 退出程序

if TEST_WITH_DEV_DBG_ASAN:  # 如果处于调试环境下
    print(
        "Skip dev-asan as torch + multiprocessing spawn have known issues",
        file=sys.stderr,
    )  # 输出信息到标准错误流
    sys.exit(0)  # 退出程序


class Model(Module):
    def __init__(self, wrap_fsdp):
        super().__init__()  # 调用父类构造函数
        # keep everything deterministic for model initialization
        torch.manual_seed(0)  # 设置随机数种子以确保模型初始化的确定性
        self.inner = Linear(4, 4)  # 创建一个输入输出维度为 4 的线性层
        if wrap_fsdp:
            self.inner = FSDP(self.inner)  # 如果使用 FSDP 包装，则将内部线性层替换为 FSDP
        self.outer = Linear(4, 5)  # 创建一个输入维度为 4，输出维度为 5 的线性层

    def forward(self, x):
        # Forward twice.
        i = self.inner(x)  # 使用内部层处理输入 x
        j = self.inner(x)  # 再次使用内部层处理输入 x
        return self.outer(i + j)  # 返回外部层对 i + j 的处理结果


class TestMultiForward(FSDPTest):
    def _dist_train(self, wrap_fsdp):
        # keep everything deterministic for input data
        torch.manual_seed(0)  # 设置随机数种子以确保输入数据的确定性

        model = Model(wrap_fsdp).cuda()  # 创建模型并将其移动到 GPU 上
        if wrap_fsdp:
            model = FSDP(model)  # 如果使用 FSDP 包装，则将模型替换为 FSDP 模型
        else:
            model = DistributedDataParallel(model, device_ids=[self.rank])  # 使用 DDP 对模型进行分布式数据并行处理
        optim = SGD(model.parameters(), lr=0.1)  # 使用 SGD 优化器优化模型参数

        in_data = torch.rand(64, 4).cuda()  # 创建输入数据，大小为 [64, 4]，并将其移动到 GPU 上
        in_data.requires_grad = True  # 设置输入数据需要梯度计算
        for _ in range(3):
            out = model(in_data)  # 对模型进行前向传播计算
            out.sum().backward()  # 计算损失函数关于参数的梯度
            optim.step()  # 执行优化步骤
            optim.zero_grad()  # 清空梯度

        if wrap_fsdp:
            return get_full_params(model)  # 如果使用 FSDP，返回完整的模型参数

        return list(model.parameters())  # 返回模型的参数列表

    @skip_if_lt_x_gpu(2)  # 如果 GPU 数量小于 2，则跳过该测试
    def test_multi_forward(self):
        # DDP
        ddp_state = self._dist_train(wrap_fsdp=False)  # 运行不使用 FSDP 的分布式训练

        # FSDP
        fsdp_state = self._dist_train(wrap_fsdp=True)  # 运行使用 FSDP 的分布式训练

        self.assertEqual(ddp_state, fsdp_state)  # 断言两种模式下的模型状态一致


if __name__ == "__main__":
    run_tests()  # 运行测试
```