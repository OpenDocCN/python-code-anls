# `.\pytorch\test\distributed\_composable\fsdp\test_fully_shard_frozen.py`

```py
# 导入所需的模块和库

import copy  # 导入copy模块，用于深拷贝对象
import functools  # 导入functools模块，用于高阶函数（higher-order functions）操作
import itertools  # 导入itertools模块，用于高效的迭代工具

from typing import List, Union  # 导入类型提示相关内容

import torch  # 导入PyTorch库
import torch.distributed as dist  # 导入PyTorch分布式模块
import torch.nn as nn  # 导入PyTorch神经网络模块
import torch.nn.functional as F  # 导入PyTorch函数式模块
from torch.distributed._composable import checkpoint, replicate  # 导入分布式相关函数
from torch.distributed._composable.fsdp import fully_shard  # 导入完全分片的相关函数
from torch.distributed._composable.fsdp._fsdp_param_group import (
    RegisterPostBackwardFunction,  # 导入注册后向函数相关内容
)
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu  # 导入分布式测试相关内容
from torch.testing._internal.common_fsdp import (
    check_sharded_parity,  # 导入检查分片一致性相关内容
    FSDPTest,  # 导入FSDP测试类
    MLP,  # 导入多层感知机（MLP）模型
    patch_reduce_scatter,  # 导入补丁化的reduce_scatter函数
    patch_register_post_backward_hook_backward,  # 导入注册后向钩子函数相关内容
    reduce_scatter_with_assert,  # 导入带有断言的reduce_scatter函数
)
from torch.testing._internal.common_utils import run_tests  # 导入运行测试相关工具

# 定义一个测试类，继承自FSDPTest类，用于测试完全分片的冻结参数情况
class TestFullyShardFrozen(FSDPTest):

    # 定义一个属性方法，返回GPU数量和4的较小值，用于设定分布式测试的全局大小
    @property
    def world_size(self) -> int:
        return min(4, torch.cuda.device_count())

    # 装饰器，如果GPU数小于2，则跳过测试
    @skip_if_lt_x_gpu(2)
    def test_train_mixed_requires_grad_per_group(self):
        """
        测试在同一个FSDP通信组中混合冻结和非冻结参数时，与DDP的训练一致性。
        检查reduce-scatter是否减少了预期的元素数量，并且它们通过自定义的反向传播函数（backward）调用。
        这里验证它们不会延迟到反向传播结束。
        """
        self.run_subtests(
            {
                "reshard_after_forward": [False, True, 2],
                "use_activation_checkpointing": [False, True],
                "freeze_after_init": [False, True],
            },
            self._test_train_mixed_requires_grad_per_group,
        )

    # 实际执行混合梯度需求组测试的方法
    def _test_train_mixed_requires_grad_per_group(
        self,
        reshard_after_forward: Union[bool, int],
        use_activation_checkpointing: bool,
        freeze_after_init: bool,
    ):
        pass  # 实际测试逻辑未提供，因此此处占位符

    # 装饰器，如果GPU数小于2，则跳过测试
    @skip_if_lt_x_gpu(2)
    def test_train_mixed_requires_grad_across_groups(self):
        """
        测试在不同的FSDP通信组中混合冻结和非冻结参数时，与DDP的训练一致性。
        包括可能解冻参数的情况。
        """
        self.run_subtests(
            {
                "reshard_after_forward": [False, True, 2],
                "unfreeze_params": [False, True],
            },
            self._test_train_mixed_requires_grad_across_groups,
        )

    # 实际执行跨组混合梯度需求测试的方法
    def _test_train_mixed_requires_grad_across_groups(
        self,
        reshard_after_forward: Union[bool, int],
        unfreeze_params: bool,
    ):
        pass  # 实际测试逻辑未提供，因此此处占位符
        ):
        # 设置随机种子以保证结果的可重复性
        torch.manual_seed(42)
        # 定义线性层的数量和维度
        num_linears, lin_dim = (6, 32)
        # 初始化模块列表
        modules: List[nn.Module] = []
        # 构建包含多个线性层和ReLU激活函数的模型
        for _ in range(num_linears):
            modules += [nn.Linear(lin_dim, lin_dim), nn.ReLU()]
        # 创建模型对象
        model = nn.Sequential(*modules)
        # 复制模型并放置到指定的CUDA设备上，用于并行训练
        ref_model = replicate(
            copy.deepcopy(model).cuda(),
            device_ids=[self.rank],
            find_unused_parameters=True,
        )
        # 遍历模型中的所有模块，对线性层进行分片处理
        for module in model.modules():
            if isinstance(module, nn.Linear):
                fully_shard(module, reshard_after_forward=reshard_after_forward)
        # 使用Adam优化器初始化原始模型和参考模型
        ref_optim = torch.optim.Adam(ref_model.parameters(), lr=1e-2)
        optim = torch.optim.Adam(model.parameters(), lr=1e-2)
        # 保存原始的RegisterPostBackwardFunction.backward函数
        orig_backward = RegisterPostBackwardFunction.backward
        # 初始化反向传播计数器
        backward_count = 0

        # 定义一个函数，用于设置模型中各线性层的梯度需求
        def _set_requires_grad(seq: nn.Module, requires_grad: bool):
            for i in range(num_linears):
                # 交替冻结和解冻线性层参数
                if i % 2 == 0:
                    for param in seq[i % 2].parameters():
                        param.requires_grad_(requires_grad)

        # 定义一个带计数功能的反向传播函数
        def backward_with_count(*args, **kwargs):
            nonlocal backward_count
            backward_count += 1
            return orig_backward(*args, **kwargs)

        # 初始时冻结模型中的参数
        _set_requires_grad(model, False)
        _set_requires_grad(ref_model, False)
        # 定义迭代次数和不允许梯度传播的迭代索引
        num_iters, no_grad_iter_idx = (3, 1)
        # 设置本地随机种子以确保结果的可重复性
        torch.manual_seed(42 + self.rank)
        # 生成输入数据并放置到CUDA设备上
        inp = torch.randn((8, lin_dim), device="cuda")
        # 使用patch_register_post_backward_hook_backward装饰器，注册带计数功能的反向传播钩子
        with patch_register_post_backward_hook_backward(backward_with_count):
            for iter_idx in range(num_iters):
                losses: List[torch.Tensor] = []
                for _model, _optim in ((ref_model, ref_optim), (model, optim)):
                    # 在最后一步解冻参数以模拟某些微调
                    if unfreeze_params and iter_idx == num_iters - 1:
                        _set_requires_grad(model, True)
                    # 在指定迭代步骤中不允许梯度传播
                    if iter_idx == no_grad_iter_idx:
                        with torch.no_grad():
                            losses.append(_model(inp).sum())
                    else:
                        # 计算损失并执行反向传播
                        losses.append(_model(inp).sum())
                        losses[-1].backward()
                        _optim.step()
                        # 清空梯度
                        _optim.zero_grad(set_to_none=(iter_idx % 2 == 0))
            # 断言两个模型的损失相等
            self.assertEqual(losses[0], losses[1])
            # 检查带计数功能的反向传播钩子是否在自动求导的反向传播中运行，
            # 而不是在最终回调中（除了可能的第一个线性层，它没有需要梯度的输入）
            self.assertTrue(backward_count >= num_linears - 1)

    @skip_if_lt_x_gpu(2)
    def test_multi_forward_mixed_requires_grad(self):
        """
        Tests training parity with DDP when having trainable and frozen modules
        that participate multiple times in forward.
        """
        # 运行子测试，测试在有可训练和冻结模块并且在前向传播中多次参与时，使用DDP的训练一致性
        self.run_subtests(
            {"reshard_after_forward": [True, False, 2]},
            self._test_multi_forward_mixed_requires_grad,
        )

    def _test_multi_forward_mixed_requires_grad(
        self,
        reshard_after_forward: Union[bool, int],
    ):
        class MultiForwardModule(nn.Module):
            def __init__(self, device: torch.device):
                super().__init__()
                # 初始化模块，包括一个有梯度的线性层和一个无梯度的线性层
                self.layer_0 = nn.Linear(5, 5, device=device)
                self.layer_no_grad = nn.Linear(5, 5, device=device)
                self.layer_with_grad = nn.Linear(5, 5, device=device)
                self.layer_no_grad.requires_grad_(False)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # 执行前向传播
                x = self.layer_0(x)
                for _ in range(3):
                    # 多次调用同一层，无论是否启用梯度都能正常工作
                    x = self.layer_no_grad(F.relu(self.layer_with_grad(x)))
                    # 使用torch.no_grad()确保即使梯度被启用，也能正确计算结果
                    with torch.no_grad():
                        x += F.relu(self.layer_with_grad(x))
                return x

        torch.manual_seed(42)
        # 创建模型实例并复制到指定设备上
        model = MultiForwardModule(torch.device("cpu"))
        ref_model = replicate(copy.deepcopy(model).cuda(), device_ids=[self.rank])
        # 为模型的每个线性层进行分片操作
        for module in model.modules():
            if isinstance(module, nn.Linear):
                fully_shard(module, reshard_after_forward=reshard_after_forward)
        fully_shard(model, reshard_after_forward=reshard_after_forward)
        # 使用Adam优化器来优化模型参数
        ref_optim = torch.optim.Adam(ref_model.parameters(), lr=1e-2)
        optim = torch.optim.Adam(model.parameters(), lr=1e-2)
        # 执行迭代训练
        for iter_idx in range(10):
            inp = torch.randn((8, 5), device="cuda")
            losses: List[torch.Tensor] = []
            # 针对每个模型实例和优化器，执行优化步骤并计算损失
            for _model, _optim in ((ref_model, ref_optim), (model, optim)):
                _optim.zero_grad(set_to_none=(iter_idx % 2 == 0))
                losses.append(_model(inp).sum())
                losses[-1].backward()
                _optim.step()
            # 断言两个模型的损失值相等
            self.assertEqual(losses[0], losses[1])
# 如果当前脚本作为主程序运行（而不是被导入为模块），则执行 run_tests() 函数
if __name__ == "__main__":
    run_tests()
```