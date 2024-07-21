# `.\pytorch\test\distributed\fsdp\test_fsdp_freezing_weights.py`

```
# Owner(s): ["oncall: distributed"]

import contextlib                  # 引入上下文管理模块，用于管理上下文，如禁用自动求导
import sys                         # 系统相关功能模块，用于与系统交互
from enum import Enum              # 引入枚举类型，用于定义一组命名的常数

import torch                       # 引入PyTorch深度学习框架
import torch.nn as nn              # PyTorch中的神经网络模块
import torch.optim as optim        # PyTorch中的优化算法模块
from torch import distributed as dist  # PyTorch分布式模块
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP  # 引入FSDP分布式训练工具
from torch.nn.parallel import DistributedDataParallel  # 分布式数据并行模块
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu  # 测试相关的分布式功能
from torch.testing._internal.common_fsdp import FSDPTest, get_full_params  # 测试相关的FSDP功能
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,  # 测试工具：实例化参数化测试
    parametrize,                     # 测试工具：参数化装饰器
    run_tests,                       # 测试工具：运行测试
    TEST_WITH_DEV_DBG_ASAN,          # 是否在开发者ASAN环境下运行测试
)

if not dist.is_available():         # 如果分布式不可用
    print("Distributed not available, skipping tests", file=sys.stderr)  # 输出信息到标准错误
    sys.exit(0)                     # 程序退出

if TEST_WITH_DEV_DBG_ASAN:          # 如果在开发者ASAN环境下运行测试
    print(
        "Skip dev-asan as torch + multiprocessing spawn have known issues",
        file=sys.stderr,            # 输出信息到标准错误
    )
    sys.exit(0)                     # 程序退出


class Model(nn.Module):             # 定义模型类，继承自nn.Module
    def __init__(                    # 初始化方法
        self,
        with_fsdp,                   # 是否使用FSDP
        freeze_after_wrap_fsdp,      # 包装后是否冻结
        disable_autograd,            # 是否禁用自动求导
        fsdp_kwargs,                 # FSDP的参数
    ):
        super().__init__()           # 调用父类初始化方法
        self.trunk = nn.Sequential(  # 定义模型主干网络，包含卷积、ReLU激活、自适应平均池化和展平层
            nn.Conv2d(3, 64, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(),
        )
        self.device = torch.cuda.current_device()  # 获取当前CUDA设备
        self.head = nn.Linear(64, 10)             # 定义模型头部网络，线性层
        if with_fsdp and freeze_after_wrap_fsdp:  # 如果使用FSDP并且包装后冻结
            self.fsdp_wrap(fsdp_kwargs)           # 对模型进行FSDP包装
        self.autograd_ctx = (                      # 设置自动求导上下文管理器
            torch.no_grad if disable_autograd else contextlib.nullcontext
        )

    def fsdp_wrap(self, fsdp_kwargs):             # 定义FSDP包装方法
        self.trunk = FSDP(self.trunk, **fsdp_kwargs)  # 对主干网络进行FSDP包装
        self.head = FSDP(self.head, **fsdp_kwargs)    # 对头部网络进行FSDP包装

    def forward(self, x):                         # 定义前向传播方法
        with self.autograd_ctx():                 # 使用自动求导上下文管理器
            x = self.trunk(x)                     # 将输入数据通过主干网络
        return self.head(x)                       # 返回头部网络的输出


class NestedTrunkModel(nn.Module):                # 定义嵌套主干网络的模型类，继承自nn.Module
    def __init__(                                # 初始化方法
        self,
        with_fsdp,                               # 是否使用FSDP
        freeze_after_wrap_fsdp,                  # 包装后是否冻结
        disable_autograd,                        # 是否禁用自动求导
        fsdp_kwargs,                             # FSDP的参数
    ):
        super().__init__()                       # 调用父类初始化方法
        self.trunk = nn.Sequential(              # 定义嵌套的主干网络，包含多个块
            self._create_block(3, 64, with_fsdp, freeze_after_wrap_fsdp),  # 创建第一个块
            self._create_block(64, 64, with_fsdp, freeze_after_wrap_fsdp),  # 创建第二个块
        )
        self.head = nn.Sequential(               # 定义头部网络，包含自适应平均池化、展平层和线性层
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(),
            nn.Linear(64, 10),
        )
        if with_fsdp and freeze_after_wrap_fsdp:  # 如果使用FSDP并且包装后冻结
            self.fsdp_wrap(fsdp_kwargs)           # 对模型进行FSDP包装
        self.autograd_ctx = (                      # 设置自动求导上下文管理器
            torch.no_grad if disable_autograd else contextlib.nullcontext
        )

    def fsdp_wrap(self, fsdp_kwargs):             # 定义FSDP包装方法
        for name, child in self.trunk.named_children():  # 遍历嵌套主干网络的子模块
            wrapped_child = FSDP(child, **fsdp_kwargs)   # 对每个子模块进行FSDP包装
            setattr(self.trunk, name, wrapped_child)     # 将包装后的子模块设置回原来的位置
        self.trunk = FSDP(self.trunk, **fsdp_kwargs)    # 对整体嵌套主干网络进行FSDP包装
        self.head = FSDP(self.head, **fsdp_kwargs)      # 对头部网络进行FSDP包装
    # 定义一个前向传播方法，接收输入张量 x
    def forward(self, x):
        # 使用自动微分的上下文处理
        with self.autograd_ctx():
            # 将输入张量 x 传递给网络的主干部分 trunk
            x = self.trunk(x)
        # 将 trunk 的输出作为输入传递给网络的头部部分 head，并返回结果
        return self.head(x)

    # 定义一个创建网络块的方法，接收输入通道数、输出通道数以及一些控制标志
    def _create_block(
        self, in_channels, out_channels, with_fsdp, freeze_after_wrap_fsdp
    ):
        # 创建一个包含卷积层和激活函数 ReLU 的序列化块
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3),  # 创建一个2D卷积层
            nn.ReLU(inplace=True),  # 使用 inplace 激活函数 ReLU
        )
        return block  # 返回创建的网络块
class FreezingMethod(str, Enum):
    GradToNone = "grad_to_none"
    RequiresGrad = "requires_grad"

class TestFreezingWeights(FSDPTest):
    def _create_model(
        self,
        with_fsdp,
        with_nested_trunk,
        freeze_after_wrap_fsdp,
        disable_autograd,
        fsdp_kwargs,
    ):
        # 根据参数创建模型，可以选择嵌套的或非嵌套的模型
        if with_nested_trunk:
            model = NestedTrunkModel(
                with_fsdp, freeze_after_wrap_fsdp, disable_autograd, fsdp_kwargs
            )
        else:
            model = Model(
                with_fsdp, freeze_after_wrap_fsdp, disable_autograd, fsdp_kwargs
            )
        return model

    def _dist_train(
        self,
        with_nested_trunk,
        freezing_method,
        freeze_after_wrap_fsdp,
        with_fsdp,
        disable_autograd,
        forward_prefetch,
    ):
        torch.manual_seed(0)
        # 创建一个大小为 (2, 3, 224, 224) 的张量，并放置在 CUDA 设备上
        batch = torch.randn(size=(2, 3, 224, 224)).cuda()

        fsdp_kwargs = {
            "device_id": self.rank,  # 设备 ID，用于分布式训练
            "forward_prefetch": forward_prefetch,  # 是否启用前向预取
        }

        ddp_kwargs = {
            "device_ids": [self.rank],  # 设备 ID 列表，用于分布式数据并行
            "find_unused_parameters": True if disable_autograd else False,  # 是否找到未使用的参数
        }

        # 创建模型对象
        model = self._create_model(
            with_fsdp,
            with_nested_trunk,
            freeze_after_wrap_fsdp,
            disable_autograd,
            fsdp_kwargs,
        )
        model = model.cuda()  # 将模型移动到 CUDA 设备上

        # 使用 requires_grad 方法冻结模型的主干部分
        if freezing_method == FreezingMethod.RequiresGrad:
            for param in model.trunk.parameters():
                param.requires_grad = False

        # 如果使用 FSDP
        if with_fsdp:
            if not freeze_after_wrap_fsdp:
                model.fsdp_wrap(fsdp_kwargs)  # 将模型进行 FSDP 封装
            model = FSDP(model, **fsdp_kwargs)  # 使用 FSDP 进行模型并行
        else:
            model = DistributedDataParallel(model, **ddp_kwargs)  # 使用 DDP 进行模型并行

        target = torch.tensor([0, 1], dtype=torch.long).cuda()  # 创建目标张量，并放置在 CUDA 设备上
        criterion = nn.CrossEntropyLoss()  # 使用交叉熵损失函数
        optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)  # 使用 SGD 优化器

        for iteration in range(3):
            out = model(batch)  # 前向传播
            fake_loss = criterion(out, target)  # 计算损失
            optimizer.zero_grad()  # 梯度清零
            fake_loss.backward()  # 反向传播
            # 如果使用 GradToNone 方法冻结主干部分
            if freezing_method == FreezingMethod.GradToNone:
                for param in model.module.trunk.parameters():
                    param.grad = None  # 将梯度置为 None
            optimizer.step()  # 执行优化步骤

        if with_fsdp:
            return get_full_params(model)  # 返回完整的模型参数

        return list(model.parameters())  # 返回模型参数列表

    @skip_if_lt_x_gpu(2)  # 如果 GPU 数量小于 2，则跳过测试
    @parametrize("with_nested_trunk", [True, False])  # 参数化测试：是否使用嵌套的主干部分
    @parametrize(
        "freezing_method", [FreezingMethod.RequiresGrad, FreezingMethod.GradToNone]  # 参数化测试：冻结方法
    )
    @parametrize("freeze_after_wrap_fsdp", [True, False])  # 参数化测试：是否在封装 FSDP 后冻结
    @parametrize("disable_autograd", [True, False])  # 参数化测试：是否禁用自动梯度
    @parametrize("forward_prefetch", [True, False])  # 参数化测试：是否启用前向预取
    # 定义一个测试方法，用于验证权重冻结的行为
    def test_freezing_weights(
        self,
        with_nested_trunk,
        freezing_method,
        freeze_after_wrap_fsdp,
        disable_autograd,
        forward_prefetch,
    ):
        # 使用分布式数据并行（DDP）进行训练
        ddp_state = self._dist_train(
            with_nested_trunk,
            freezing_method,
            freeze_after_wrap_fsdp,
            with_fsdp=False,  # 不启用 FullyShardedDataParallel
            disable_autograd=disable_autograd,
            forward_prefetch=False,  # 对于 DDP 不适用前向预取
        )

        # 使用全分片数据并行（FSDP）进行训练
        fsdp_state = self._dist_train(
            with_nested_trunk,
            freezing_method,
            freeze_after_wrap_fsdp,
            with_fsdp=True,  # 启用 FullyShardedDataParallel
            disable_autograd=disable_autograd,
            forward_prefetch=forward_prefetch,
        )

        # 断言两种训练方式的状态应该完全一致
        self.assertEqual(
            ddp_state,
            fsdp_state,
            exact_device=True,  # 精确比较设备位置
            msg="FullyShardedDataParallel states didn't match PyTorch DDP states",  # 错误消息
        )

        # 如果冻结方法是 RequiresGrad
        if freezing_method == FreezingMethod.RequiresGrad:
            # 逐个比较 DDP 和 FSDP 参数是否需要梯度
            for ddp_param, fsdp_param in zip(ddp_state, fsdp_state):
                self.assertEqual(ddp_param.requires_grad, fsdp_param.requires_grad)
# 实例化参数化测试，使用 TestFreezingWeights 类作为参数
instantiate_parametrized_tests(TestFreezingWeights)

# 如果当前脚本被直接执行，则运行测试
if __name__ == "__main__":
    run_tests()
```