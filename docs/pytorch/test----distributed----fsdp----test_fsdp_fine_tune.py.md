# `.\pytorch\test\distributed\fsdp\test_fsdp_fine_tune.py`

```
# Owner(s): ["oncall: distributed"]

import copy  # 导入 copy 模块，用于对象的深复制操作
import sys  # 导入 sys 模块，用于系统相关操作
from unittest import mock  # 导入 mock 模块，用于模拟测试

import torch  # 导入 PyTorch 主模块
import torch.distributed as dist  # 导入 PyTorch 分布式模块
import torch.nn as nn  # 导入 PyTorch 神经网络模块
from torch.distributed.fsdp import BackwardPrefetch, CPUOffload, MixedPrecision  # 导入 FSDP 相关模块
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    FullyShardedDataParallel as FSDP,  # 导入 FSDP 并命名为 FSDP
    ShardingStrategy,  # 导入分片策略
)
from torch.distributed.fsdp.wrap import ModuleWrapPolicy  # 导入模块包装策略
from torch.nn.parallel import DistributedDataParallel as DDP  # 导入分布式数据并行模块并命名为 DDP
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu  # 导入测试用的 GPU 数量判断装饰器
from torch.testing._internal.common_fsdp import FSDPTest  # 导入 FSDP 测试基类
from torch.testing._internal.common_utils import run_tests, TEST_WITH_DEV_DBG_ASAN  # 导入测试运行和调试标志

if not dist.is_available():  # 如果分布式不可用
    print("Distributed not available, skipping tests", file=sys.stderr)  # 输出错误信息到标准错误流
    sys.exit(0)  # 退出程序

if TEST_WITH_DEV_DBG_ASAN:  # 如果设置了 dev-asan 调试标志
    print(
        "Skip dev-asan as torch + multiprocessing spawn have known issues",
        file=sys.stderr,  # 输出警告信息到标准错误流
    )
    sys.exit(0)  # 退出程序


class LinearUnusedInput(nn.Linear):  # 定义继承自 nn.Linear 的线性层子类 LinearUnusedInput
    def forward(self, frozen_input, learnable_input):  # 重写 forward 方法，接收冻结输入和可学习输入
        return super().forward(frozen_input)  # 调用父类 nn.Linear 的 forward 方法


class ModelUnusedInput(nn.Module):  # 定义继承自 nn.Module 的模型类 ModelUnusedInput
    def __init__(self, freeze: bool):  # 初始化方法，接收一个冻结标志
        super().__init__()  # 调用父类的初始化方法
        self.layer0 = LinearUnusedInput(4, 4, device="cuda")  # 创建一个 LinearUnusedInput 实例作为层0
        self.layer1_frozen = LinearUnusedInput(4, 4, device="cuda")  # 创建一个冻结的 LinearUnusedInput 实例作为层1
        if freeze:  # 如果需要冻结
            for param in self.layer1_frozen.parameters():  # 遍历层1的参数
                param.requires_grad = False  # 设置参数不需要梯度计算
        self.layer2 = LinearUnusedInput(4, 4, device="cuda")  # 创建一个 LinearUnusedInput 实例作为层2

    def forward(self, frozen_input, learnable_input):  # 前向传播方法，接收冻结输入和可学习输入
        x = self.layer0(frozen_input, learnable_input)  # 使用层0进行前向计算
        y = self.layer1_frozen(frozen_input, learnable_input)  # 使用层1进行前向计算
        z = self.layer2(frozen_input, learnable_input)  # 使用层2进行前向计算
        return torch.concat([x, y, z, learnable_input])  # 返回拼接后的张量


class TestFSDPFineTune(FSDPTest):  # 定义继承自 FSDPTest 的测试类 TestFSDPFineTune
    """Tests fine-tuning cases where some parameters are frozen."""  # 类的文档字符串，描述测试的目的

    NUM_LINEARS = 6  # 类属性，线性层的数量为6

    @property
    def world_size(self) -> int:  # 定义 world_size 属性，返回 GPU 数量和2的较小值
        return min(torch.cuda.device_count(), 2)

    def _init_seq_module(self) -> nn.Module:  # 初始化方法，返回一个包含多个线性层的序列模型
        torch.manual_seed(42)  # 设置随机种子为42
        modules = []  # 创建模块列表
        for _ in range(self.NUM_LINEARS):  # 循环创建指定数量的线性层和ReLU激活函数
            modules += [nn.Linear(5, 5, device="cuda"), nn.ReLU()]
        seq = nn.Sequential(*modules)  # 创建序列模型
        self._set_seq_module_requires_grad(seq, False)  # 调用方法设置模型的 requires_grad 属性
        return seq  # 返回创建的序列模型

    def _set_seq_module_requires_grad(self, seq: nn.Module, requires_grad: bool):  # 设置模型 requires_grad 属性的方法
        # 假设线性层是叶子模块，可以传递 recurse=True 以支持 FSDP 包装前后的设置
        for i in range(self.NUM_LINEARS):  # 遍历线性层
            # 仅设置每隔一个线性层的 requires_grad 以测试混合冻结/非冻结参数
            if i % 2 == 0:  # 如果是偶数索引的线性层
                for param in seq[i * 2].parameters(recurse=True):  # 遍历线性层的参数
                    param.requires_grad = requires_grad  # 设置参数的 requires_grad 属性

    @skip_if_lt_x_gpu(2)  # 装饰器，如果 GPU 数量小于2则跳过测试
    def test_backward_reshard_hooks(self):
        """
        Tests that the post-backward reshard happens even for flat parameters
        that do not require gradients.
        """
        # 运行子测试，验证即使对不需要梯度的平坦参数也会发生后向重分片
        self.run_subtests(
            {
                "sharding_strategy": [
                    ShardingStrategy.FULL_SHARD,
                    ShardingStrategy.SHARD_GRAD_OP,
                    ShardingStrategy.NO_SHARD,
                ],
                "use_orig_params": [False, True],
                "inp_requires_grad": [False, True],
                "unfreeze_params": [False, True],
            },
            self._test_backward_reshard_hooks,
        )

    def _test_backward_reshard_hooks(
        self,
        sharding_strategy: ShardingStrategy,
        use_orig_params: bool,
        inp_requires_grad: bool,
        unfreeze_params: bool,
    ):
        """
        Actual test function for testing backward reshard hooks with various parameters.
        """
        # 实际测试函数，用于测试带有各种参数的后向重分片钩子

    def _init_multi_traversal_module(self) -> nn.Module:
        torch.manual_seed(42)

        class TestModule(nn.Module):
            def __init__(self):
                super().__init__()
                # 初始化测试模块，包括三个线性层，其中一个无梯度
                self.layer_0 = nn.Linear(5, 5, device="cuda")
                self.layer_no_grad = nn.Linear(5, 5, device="cuda")
                self.layer_with_grad = nn.Linear(5, 5, device="cuda")
                self.layer_no_grad.requires_grad_(False)

            def forward(self, x):
                # 在前向传播过程中，多次调用`layer_no_grad`和`layer_with_grad`
                # 即它们的参数在前向传播中被多次使用
                x = self.layer_0(x)
                for _ in range(10):
                    x = self.layer_no_grad(self.layer_with_grad(x))
                    # 确保多次调用同一层在启用梯度时也能正常工作
                    with torch.no_grad():
                        x += self.layer_with_grad(x)
                return x

        return TestModule()

    @skip_if_lt_x_gpu(2)
    def test_hooks_multi_traversal(self):
        """
        Tests that the hooks do reshard / unshard correctly in the case of same
        parameters being used multiple times during forward pass.
        """
        # 运行子测试，验证在前向传播过程中多次使用相同参数时，钩子能正确进行重分片/取消分片
        self.run_subtests(
            {
                "sharding_strategy": [
                    ShardingStrategy.FULL_SHARD,
                    ShardingStrategy.SHARD_GRAD_OP,
                    ShardingStrategy.NO_SHARD,
                ],
                "use_orig_params": [False, True],
                "inp_requires_grad": [False, True],
                "forward_prefetch": [False, True],
            },
            self._test_hooks_multi_traversal,
        )

    def _test_hooks_multi_traversal(
        self,
        sharding_strategy: ShardingStrategy,
        use_orig_params: bool,
        inp_requires_grad: bool,
        forward_prefetch: bool,
    ):
        """
        Actual test function for testing hooks' behavior during multi-traversal scenarios.
        """
        # 实际测试函数，用于测试多次遍历场景下钩子的行为
    ):
        # 初始化多遍历模块，并返回序列
        seq = self._init_multi_traversal_module()
        # 创建模块包装策略，仅包含 nn.Linear 模块
        policy = ModuleWrapPolicy({nn.Linear})
        # 使用 FSDP 对象封装深拷贝的 seq，设置自动包装策略、分片策略、使用原始参数、前向预取
        fsdp_seq = FSDP(
            copy.deepcopy(seq),
            auto_wrap_policy=policy,
            sharding_strategy=sharding_strategy,
            use_orig_params=use_orig_params,
            forward_prefetch=forward_prefetch,
        )
        # 使用 DDP 对象封装深拷贝的 seq，设备 ID 为 self.rank
        ddp_seq = DDP(copy.deepcopy(seq), device_ids=[self.rank])
        # 使用 Adam 优化器初始化 FSDP 对象的参数优化器
        fsdp_optim = torch.optim.Adam(fsdp_seq.parameters(), lr=1e-2)
        # 使用 Adam 优化器初始化 DDP 对象的参数优化器
        ddp_optim = torch.optim.Adam(ddp_seq.parameters(), lr=1e-2)
        # 设定随机种子
        torch.manual_seed(self.rank + 1)
        # 初始化损失列表
        losses = []
        # 执行 6 次循环
        for _ in range(6):
            # 在 CUDA 设备上生成随机张量 inp，根据参数决定是否需要梯度
            inp = torch.randn((8, 5), device="cuda", requires_grad=inp_requires_grad)
            # 对 fsdp_seq 和 ddp_seq 应用相同的操作
            for seq, optim in ((fsdp_seq, fsdp_optim), (ddp_seq, ddp_optim)):
                # 计算序列的输出，求和得到损失
                loss = seq(inp).sum()
                # 将损失值添加到损失列表
                losses.append(loss)
                # 反向传播
                loss.backward()
                # 执行优化步骤
                optim.step()
                # 清空梯度
                optim.zero_grad()
            # 断言损失列表中的第一个元素与第二个元素接近
            torch.testing.assert_close(losses[0], losses[1])
            # 清空损失列表
            losses.clear()

    @skip_if_lt_x_gpu(2)
    def test_parity_with_ddp(self):
        """
        Tests parity with DDP when mixing flat parameters that require and do
        not require gradients.
        """
        # 运行子测试，测试与 DDP 的一致性
        self.run_subtests(
            {
                "sharding_strategy": [
                    ShardingStrategy.FULL_SHARD,
                    ShardingStrategy.SHARD_GRAD_OP,
                    ShardingStrategy.NO_SHARD,
                ],
                "use_orig_params": [False, True],
            },
            self._test_parity_with_ddp,
        )

    def _test_parity_with_ddp(
        self,
        sharding_strategy: ShardingStrategy,
        use_orig_params: bool,
    ):
        # 初始化序列模块
        seq = self._init_seq_module()
        # 创建模块包装策略，仅包含 nn.Linear 模块
        policy = ModuleWrapPolicy({nn.Linear})
        # 使用 FSDP 对象封装深拷贝的 seq，设置自动包装策略、分片策略、使用原始参数
        fsdp_seq = FSDP(
            copy.deepcopy(seq),
            auto_wrap_policy=policy,
            sharding_strategy=sharding_strategy,
            use_orig_params=use_orig_params,
        )
        # 使用 DDP 对象封装深拷贝的 seq，设备 ID 为 self.rank
        ddp_seq = DDP(copy.deepcopy(seq), device_ids=[self.rank])
        # 使用 Adam 优化器初始化 FSDP 对象的参数优化器
        fsdp_optim = torch.optim.Adam(fsdp_seq.parameters(), lr=1e-2)
        # 使用 Adam 优化器初始化 DDP 对象的参数优化器
        ddp_optim = torch.optim.Adam(ddp_seq.parameters(), lr=1e-2)
        # 设定随机种子
        torch.manual_seed(self.rank + 1)
        # 初始化损失列表
        losses = []
        # 执行 6 次循环
        for _ in range(6):
            # 在 CUDA 设备上生成随机张量 inp
            inp = torch.randn((8, 5), device="cuda")
            # 对 fsdp_seq 和 ddp_seq 应用相同的操作
            for seq, optim in ((fsdp_seq, fsdp_optim), (ddp_seq, ddp_optim)):
                # 计算序列的输出，求和得到损失
                loss = seq(inp).sum()
                # 将损失值添加到损失列表
                losses.append(loss)
                # 反向传播
                loss.backward()
                # 执行优化步骤
                optim.step()
                # 清空梯度
                optim.zero_grad()
            # 断言损失列表中的第一个元素与第二个元素接近
            torch.testing.assert_close(losses[0], losses[1])
            # 清空损失列表
            losses.clear()

    @skip_if_lt_x_gpu(2)
    def test_parity_with_non_frozen_fsdp(self):
        """
        For frozen modules with unused input, reshard could happen without unshard
        Verify numerical parity between `_post_backward_reshard_only_hook` and
        `_post_backward_hook` path
        """
        # 运行子测试，传入不同的参数组合进行验证
        self.run_subtests(
            {
                "sharding_strategy": [
                    ShardingStrategy.FULL_SHARD,
                    ShardingStrategy.SHARD_GRAD_OP,
                ],
                "use_orig_params": [True, False],
                "offload_params": [True, False],
                "mixed_precision": [
                    MixedPrecision(),
                    MixedPrecision(
                        param_dtype=torch.float16,
                        buffer_dtype=torch.float16,
                        reduce_dtype=torch.float16,
                    ),
                ],
                "backward_prefetch": [
                    BackwardPrefetch.BACKWARD_PRE,
                    BackwardPrefetch.BACKWARD_POST,
                ],
            },
            # 将测试方法 `_test_parity_with_non_frozen_fsdp` 作为参数传入
            self._test_parity_with_non_frozen_fsdp,
        )

    def _test_parity_with_non_frozen_fsdp(
        self,
        sharding_strategy: ShardingStrategy,
        use_orig_params: bool,
        offload_params: bool,
        mixed_precision: MixedPrecision,
        backward_prefetch: BackwardPrefetch,
        ):
        # 设置随机种子为42，以保证结果的可重复性
        torch.manual_seed(42)
        # 创建一个冻结的模型实例
        model = ModelUnusedInput(freeze=True)
        # 再次设置随机种子为42，确保两个模型初始状态相同
        torch.manual_seed(42)
        # 创建一个不冻结的参考模型实例
        ref_model = ModelUnusedInput(freeze=False)
        # 定义FSDP的配置参数字典
        fsdp_kwargs = {
            "auto_wrap_policy": ModuleWrapPolicy({LinearUnusedInput}),
            "sharding_strategy": sharding_strategy,
            "use_orig_params": use_orig_params,
            "cpu_offload": CPUOffload(offload_params=offload_params),
            "mixed_precision": mixed_precision,
            "backward_prefetch": backward_prefetch,
        }
        # 将模型包装进FSDP，使用指定的配置参数
        model = FSDP(model, **fsdp_kwargs)
        # 将参考模型包装进FSDP，使用相同的配置参数
        ref_model = FSDP(ref_model, **fsdp_kwargs)
        # 创建模型的优化器，使用Adam优化器，学习率为1e-2
        model_optim = torch.optim.Adam(model.parameters(), lr=1e-2)
        # 创建参考模型的优化器，只优化未冻结层的参数，学习率为1e-2
        ref_model_optim = torch.optim.Adam(
            [
                param
                for name, param in ref_model.named_parameters()
                if not name.startswith("_fsdp_wrapped_module.layer1_frozen")
            ],
            lr=1e-2,
        )
        # 设置随机种子，以确保每次运行时生成相同的输入数据
        torch.manual_seed(self.rank + 1)
        # 初始化空列表用于存储损失值
        losses = []
        # 执行6次迭代
        for idx in range(6):
            # 生成在CUDA上的随机张量作为冻结输入，不需要梯度信息
            frozen_input = torch.randn((4, 4), device="cuda", requires_grad=False)
            # 生成在CUDA上的随机张量作为可学习输入，需要梯度信息
            learnable_input = torch.randn((4, 4), device="cuda", requires_grad=True)
            # 对模型和参考模型执行前向传播计算损失，并对损失进行求和
            for _model, _optim in ((model, model_optim), (ref_model, ref_model_optim)):
                loss = _model(frozen_input, frozen_input).sum()
                # 将损失值加入列表
                losses.append(loss)
                # 反向传播，计算梯度
                loss.backward()
                # 执行优化步骤
                _optim.step()
                # 清空梯度
                _optim.zero_grad()
            # 断言两个模型的第一个损失值相等，用于验证模型逻辑是否一致
            self.assertEqual(losses[0], losses[1])
            # 清空损失列表，为下一轮迭代做准备
            losses.clear()
        # 使用FSDP.summon_full_params上下文管理器，确保模型处于完整参数状态
        with FSDP.summon_full_params(model):
            # 使用FSDP.summon_full_params上下文管理器，确保参考模型处于完整参数状态
            with FSDP.summon_full_params(ref_model):
                # 逐个比较模型和参考模型的每个参数，确保它们完全相同
                for param, ref_param in zip(model.parameters(), ref_model.parameters()):
                    # 断言模型参数与参考模型参数相等
                    self.assertEqual(param, ref_param)
# 如果这个脚本是作为主程序运行
if __name__ == "__main__":
    # 调用函数 run_tests()，用于执行测试代码或测试套件
    run_tests()
```