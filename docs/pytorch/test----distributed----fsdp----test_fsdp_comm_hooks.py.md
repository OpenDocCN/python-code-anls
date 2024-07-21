# `.\pytorch\test\distributed\fsdp\test_fsdp_comm_hooks.py`

```
# Owner(s): ["oncall: distributed"]

import sys  # 导入sys模块，用于系统相关操作
from typing import Optional  # 导入Optional类型提示

import torch  # 导入PyTorch库
import torch.nn as nn  # 导入PyTorch的神经网络模块
import torch.nn.functional as F  # 导入PyTorch的函数式模块
from torch import distributed as dist  # 导入PyTorch分布式通信模块
from torch.distributed.algorithms._comm_hooks import default_hooks  # 导入通信钩子模块
from torch.distributed.distributed_c10d import _get_default_group  # 导入默认分布式组模块
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision  # 导入全分片数据并行模块和混合精度模块
from torch.distributed.fsdp.fully_sharded_data_parallel import ShardingStrategy  # 导入分片策略模块
from torch.distributed.fsdp.wrap import ModuleWrapPolicy  # 导入模块包装策略模块
from torch.testing._internal.common_distributed import (
    requires_nccl,  # 导入NCCL需求检测
    requires_nccl_version,  # 导入NCCL版本需求检测
    skip_but_pass_in_sandcastle_if,  # 导入测试跳过逻辑，但在沙堡中通过的检测
    skip_if_lt_x_gpu,  # 导入小于指定GPU数量跳过的检测
)
from torch.testing._internal.common_fsdp import FSDPTest  # 导入FSDP测试模块
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,  # 导入实例化参数化测试函数
    parametrize,  # 导入参数化装饰器
    run_tests,  # 导入运行测试函数
)

if not dist.is_available():  # 检测分布式是否可用
    print("Distributed not available, skipping tests", file=sys.stderr)  # 输出分布式不可用信息到标准错误流
    sys.exit(0)  # 退出程序

# bfloat16仅在CUDA 11+支持
BFLOAT16_AVAILABLE = torch.cuda.is_available() and (
    torch.version.cuda is not None or torch.version.hip is not None
)


class Net(nn.Module):
    def __init__(self, has_wrapping, sharding_strategy, mixed_precision=None):
        # 保证结果的确定性
        torch.manual_seed(0)  # 设置随机种子
        torch.cuda.manual_seed(0)  # 设置CUDA随机种子
        super().__init__()  # 调用父类的初始化方法

        if has_wrapping:
            # 如果有包装策略
            self.net = FSDP(  # 使用全分片数据并行模块封装
                nn.Sequential(  # 创建神经网络序列
                    nn.Linear(8, 16),  # 添加线性层，输入8个特征，输出16个特征
                    nn.ReLU(),  # ReLU激活函数
                    FSDP(  # 再次使用全分片数据并行模块封装
                        nn.Linear(16, 8),  # 添加线性层，输入16个特征，输出8个特征
                        device_id=torch.cuda.current_device(),  # 当前CUDA设备ID
                        sharding_strategy=sharding_strategy,  # 分片策略
                        mixed_precision=mixed_precision,  # 混合精度
                    ),
                ),
                device_id=torch.cuda.current_device(),  # 当前CUDA设备ID
                sharding_strategy=sharding_strategy,  # 分片策略
                mixed_precision=mixed_precision,  # 混合精度
            )
        else:
            # 如果没有包装策略
            self.net = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 8))  # 创建神经网络序列

        self.out = nn.Linear(8, 4)  # 输出层，线性层，输入8个特征，输出4个特征

    def forward(self, x):
        return self.out(F.relu(self.net(x)))  # 网络前向传播


class DummyState:
    __slots__ = ["process_group", "noise"]  # 仅允许有"process_group"和"noise"两个属性

    def __init__(self, process_group: dist.ProcessGroup, noise: int):
        self.process_group = process_group  # 初始化分布式进程组
        self.noise = noise  # 初始化噪声值


class DummyHook:
    def dummy_hook_for_no_shard_fsdp(self, state: DummyState, grad: torch.Tensor):
        """
        This communication hook is for illustration and testing purpose only.
        This communication hook is used during FSDP ``NO_SHARD`` training. It adds some noise to
        the provided ``grad`` parameter and uses ``all_reduce`` to communicate full, flattened,
        unsharded gradient.
        """
        grad.add_(state.noise)  # 添加噪声到梯度张量
        dist.all_reduce(grad, group=state.process_group)  # 使用all_reduce通信全局梯度，使用给定的分布式进程组
    # 定义一个方法 `custom_reduce_scatter`，用于实现一个定制的 reduce-scatter 操作
    # 将扁平化的张量按某种方式分散到组中的所有进程中
    def custom_reduce_scatter(self, output, input, group=None):
        """
        This function is for illustrative purpose only.
        It is meant to implement a custom reduce-scatter
        of a flattened tensor to all processes in a group.
        Currently a no-op.
        """
        # 这里当前没有实现任何操作，函数体为空
        pass

    # 定义一个方法 `dummy_hook_for_sharded_fsdp`，用于分片 FSDP （Fully Sharded Data Parallelism） 的虚拟钩子
    # 该钩子用于测试和说明目的，用于 FSDP 中的 FULL_SHARD 或 SHARD_GRAD_OP 训练
    def dummy_hook_for_sharded_fsdp(
        self, state: DummyState, grad: torch.Tensor, output: torch.Tensor
    ):
        """
        This communication hook is for illustration and testing purposes only.
        This communication hook is used during FSDP ``FULL_SHARD`` or ``SHARD_GRAD_OP`` training.
        It adds some noise to the provided ``grad`` parameter, uses
        ``reduce_scatter`` for gradient communication and stores a sharded gradient in ``output``.
        """
        # 将噪声添加到梯度张量中
        grad.add_(state.noise)
        # 调用自定义的 reduce_scatter 方法，用于梯度通信
        self.custom_reduce_scatter(output, grad, group=state.process_group)
class TestCommunicationHooks(FSDPTest):
    # 跳过如果 GPU 数量小于 2 的测试
    @skip_if_lt_x_gpu(2)
    # 参数化测试，测试不同的分片策略
    @parametrize(
        "sharding_strategy",
        [
            ShardingStrategy.NO_SHARD,
            ShardingStrategy.FULL_SHARD,
            ShardingStrategy.SHARD_GRAD_OP,
        ],
    )
    # 测试默认通信钩子行为
    def test_default_communication_hook_behavior(
        self, sharding_strategy: Optional[ShardingStrategy]
    ):
        """
        Tests FSDP's default communication hook's behavior and correctness.
        This test creates a simple linear net with weight shape  ``1 X N``,
        where ``N`` is the number of workers.
        For sharded cases, each worker gets 1 element of the weight parameter. This test
        checks that after backward, each worker has a proper value in its chunk of
        the gradient, or the whole gradient on every worker is equal to an expected value.

        Arguments:
            sharding_strategy (Optional[ShardingStrategy]): Configures the FSDP algorithm.
        """
        # 输出维度等于全局进程数
        out_dim = self.world_size
        # 创建一个简单的线性网络，权重形状为 ``1 X N``
        net = torch.nn.Linear(1, out_dim, bias=False)
        # 输入为当前进程的秩，转换为 CUDA 张量
        inpt = torch.tensor([self.rank]).float().cuda(self.rank)

        # 使用 FSDP 封装网络，配置当前设备和分片策略
        net_default_hook = FSDP(
            net,
            device_id=torch.cuda.current_device(),
            sharding_strategy=sharding_strategy,
        ).to(self.rank)

        # 检查默认情况下 `_comm_hook` 是否为 None
        for entry in FSDP.fsdp_modules(net_default_hook):
            self.assertEqual(entry._comm_hook, None)

        # 执行四次迭代
        for _ in range(4):
            # 清空梯度
            net_default_hook.zero_grad()
            # 计算损失并反向传播
            loss = net_default_hook(inpt).sum()
            loss.backward()

            # 对于每个 worker，权重的梯度应为 worker_rank
            grad = net_default_hook.params[0].grad
            expected_grad = (
                sum(i for i in range(dist.get_world_size())) / dist.get_world_size()
            )
            # 验证默认钩子产生预期的梯度
            self.assertEqual(
                grad[0].item(),
                expected_grad,
                msg=f"Expected hook grad of {expected_grad} but got {grad[0].item()}",
            )

    # 获取 FSDP 网络的子模块，排除根模块
    def _get_submodules(self, fsdp_net):
        return [
            submodule
            for submodule in FSDP.fsdp_modules(fsdp_net)
            if not submodule.check_is_root()
        ]

    # 初始化模型函数，配置设备、分片策略和混合精度选项
    def _init_model(self, core, sharding_strategy, mixed_precision=None):
        device = torch.device("cuda")
        return FSDP(
            core,
            device_id=torch.cuda.current_device(),
            sharding_strategy=sharding_strategy,
            mixed_precision=mixed_precision,
        ).to(device)

    # 跳过如果 GPU 数量小于 2 的测试
    @skip_if_lt_x_gpu(2)
    # 参数化测试，测试是否有包装和不同的分片策略
    @parametrize("has_wrapping", [True, False])
    @parametrize(
        "sharding_strategy",
        [
            ShardingStrategy.NO_SHARD,
            ShardingStrategy.FULL_SHARD,
            ShardingStrategy.SHARD_GRAD_OP,
        ],
    )
    `
        def test_default_communication_hook_initialization(
            self, has_wrapping: bool, sharding_strategy: Optional[ShardingStrategy]
        ):
            """
            Tests FSDP's communication hook interface behavior.
    
            Arguments:
                has_wrapping (bool): Configures wrapping of a module.
                sharding_strategy (Optional[ShardingStrategy]): Configures the FSDP algorithm.
            """
    
            # 初始化一个模型，调用初始化函数，传入包含是否包装和分片策略的网络实例
            fsdp_model_with_hook = self._init_model(
                Net(has_wrapping=has_wrapping, sharding_strategy=sharding_strategy),
                sharding_strategy=sharding_strategy,
            )
    
            # 检查默认情况下，`_comm_hook` 是否为 None
            for fsdp_module in FSDP.fsdp_modules(fsdp_model_with_hook):
                self.assertEqual(fsdp_module._comm_hook, None)
    
            # 创建一个 dummy 状态对象，process_group 为 None，noise 为 1234
            dummy_state = DummyState(process_group=None, noise=1234)
            # 根据分片策略选择合适的钩子
            dummy_hook = (
                DummyHook.dummy_hook_for_no_shard_fsdp
                if sharding_strategy != ShardingStrategy.NO_SHARD
                else DummyHook.dummy_hook_for_sharded_fsdp
            )
    
            # 注册通讯钩子
            fsdp_model_with_hook.register_comm_hook(dummy_state, dummy_hook)
    
            # 检查是否不能重复注册通讯钩子，期望抛出断言错误
            with self.assertRaisesRegex(
                AssertionError, "^A communication hook is already registered$"
            ):
                fsdp_model_with_hook.register_comm_hook(dummy_state, dummy_hook)
    
            # 检查是否成功将 dummy_hook 注册到根模块及其所有子模块
            for fsdp_module in FSDP.fsdp_modules(fsdp_model_with_hook):
                self.assertEqual(fsdp_module._comm_hook, dummy_hook)
                self.assertEqual(fsdp_module._comm_hook_state, dummy_state)
    
        @skip_if_lt_x_gpu(2)
        @parametrize(
            "sharding_strategy",
            [
                ShardingStrategy.NO_SHARD,
                ShardingStrategy.FULL_SHARD,
                ShardingStrategy.SHARD_GRAD_OP,
            ],
        )
        def test_registering_hook_non_root(
            self, sharding_strategy: Optional[ShardingStrategy]
        ):
    ):
        """
        Tests FSDP's communication hook registering for submodules.
        Make sure it can't be registered for non-root submodules.
        Currently tests only ``NO_SHARD`` strategy.

        Arguments:
            sharding_strategy (Optional[ShardingStrategy]): Configures the FSDP algorithm.
        """

        fsdp_model_with_hook = self._init_model(
            Net(has_wrapping=True, sharding_strategy=sharding_strategy),
            sharding_strategy=sharding_strategy,
        )
        dummy_state = DummyState(process_group=None, noise=1234)
        dummy_hook = (
            DummyHook.dummy_hook_for_no_shard_fsdp
            if sharding_strategy != ShardingStrategy.NO_SHARD
            else DummyHook.dummy_hook_for_sharded_fsdp
        )
        # Creating a list of non-root submodules to test
        submodules = self._get_submodules(fsdp_model_with_hook)
        # Check that assertion is raised for registering a comm hook on a non-root
        with self.assertRaisesRegex(
            AssertionError,
            "^register_comm_hook can only be called on a root instance.$",
        ):
            submodules[1].register_comm_hook(dummy_state, dummy_hook)
    ):
        """
        Tests FSDP's communication hook registering for submodules.
        Checks behavior if a hook was registered for a non-root submodule
        Currently tests only ``NO_SHARD`` strategy.

        Arguments:
            sharding_strategy (Optional[ShardingStrategy]): Configures the FSDP algorithm.
        """

        # 初始化带有通信钩子的模型
        fsdp_model_with_hook = self._init_model(
            Net(has_wrapping=True, sharding_strategy=sharding_strategy),
            sharding_strategy=sharding_strategy,
        )
        # 创建一个虚拟状态和钩子
        dummy_state = DummyState(process_group=None, noise=1234)
        dummy_hook = (
            DummyHook.dummy_hook_for_no_shard_fsdp
            if sharding_strategy != ShardingStrategy.NO_SHARD
            else DummyHook.dummy_hook_for_sharded_fsdp
        )
        # 获取模型的子模块列表
        submodules = self._get_submodules(fsdp_model_with_hook)

        # 模拟在子模块上注册钩子
        submodules[1]._comm_hook = dummy_hook
        # 检查当某些子模块已分配非默认钩子时是否引发错误
        with self.assertRaisesRegex(
            AssertionError, "^A communication hook is already registered$"
        ):
            fsdp_model_with_hook.register_comm_hook(dummy_state, dummy_hook)

    def _check_low_precision_hook(
        self, state, hook, sharding_strategy, dtype, has_wrapping
    ):
        # keep everything deterministic for input data
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)

        # 初始化带有低精度钩子的模型
        fsdp_with_hook = self._init_model(
            Net(has_wrapping=has_wrapping, sharding_strategy=sharding_strategy),
            sharding_strategy=sharding_strategy,
        )
        # 在模型中注册通信钩子
        fsdp_with_hook.register_comm_hook(state, hook)

        # 创建仅混合精度的 MixedPrecision 对象
        mp_only_grad = MixedPrecision(reduce_dtype=dtype)
        # 初始化带有混合精度的模型
        fsdp_with_mp = self._init_model(
            Net(
                has_wrapping=has_wrapping,
                sharding_strategy=sharding_strategy,
                mixed_precision=mp_only_grad,
            ),
            sharding_strategy=sharding_strategy,
            mixed_precision=mp_only_grad,
        )

        # 定义优化器
        optim_hook = torch.optim.SGD(fsdp_with_hook.parameters(), lr=0.1)
        optim_mp = torch.optim.SGD(fsdp_with_mp.parameters(), lr=0.1)

        # 创建输入数据，并在 CUDA 上执行
        in_data = torch.rand(16, 8).cuda()
        fsdp_with_hook.train()
        fsdp_with_mp.train()
        # 计算钩子模型和混合精度模型的损失
        loss_hook = fsdp_with_hook(in_data).sum()
        loss_mp = fsdp_with_mp(in_data).sum()
        # 计算钩子模型的梯度
        loss_hook.backward()
        # 确保梯度转换为参数的精度
        self.assertEqual(fsdp_with_hook.params[0].grad.dtype, state.parameter_type)
        # 计算混合精度模型的梯度
        loss_mp.backward()
        # 执行钩子模型的优化步骤
        optim_hook.step()
        # 执行混合精度模型的优化步骤
        optim_mp.step()

        # 等待所有进程执行完毕
        dist.barrier()

        # 检查钩子模型和混合精度模型的梯度是否一致
        for hook_param, mp_param in zip(
            fsdp_with_hook.parameters(), fsdp_with_mp.parameters()
        ):
            self.assertEqual(hook_param.grad, mp_param.grad)

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    @parametrize("has_wrapping", [True, False])
    # 参数化测试装饰器，用于多次运行同一测试方法，每次传入不同的参数
    @parametrize(
        "sharding_strategy",
        [
            ShardingStrategy.NO_SHARD,        # 不进行分片的策略
            ShardingStrategy.FULL_SHARD,      # 完全分片的策略
            ShardingStrategy.SHARD_GRAD_OP,   # 梯度操作分片的策略
        ],
    )
    # 测试方法，用于测试 fp16_hook 功能
    def test_fp16_hook(
        self, has_wrapping: bool, sharding_strategy: Optional[ShardingStrategy]
    ):
        # 创建默认的低精度状态，使用默认进程组
        state = default_hooks.LowPrecisionState(process_group=_get_default_group())
        # 设置 fp16_compress_hook 作为测试钩子
        hook = default_hooks.fp16_compress_hook
    
        # 调用私有方法，检查低精度钩子的行为是否符合预期
        self._check_low_precision_hook(
            state, hook, sharding_strategy, torch.float16, has_wrapping
        )
    
    # 要求 NCCL 的装饰器，确保 NCCL 在当前环境中可用
    @requires_nccl()
    # 要求特定版本的 NCCL 装饰器，此处需要 NCCL 2.10+ 版本才能支持 BF16_COMPRESS
    @requires_nccl_version((2, 10), "Need NCCL 2.10+ for BF16_COMPRESS")
    # 如果 BFLOAT16_AVAILABLE 不可用，则在 Sandcastle 中跳过此测试
    @skip_but_pass_in_sandcastle_if(
        not BFLOAT16_AVAILABLE,
        "BFloat16 is only supported by CUDA 11+",
    )
    # 如果 GPU 数量少于 2，跳过此测试
    @skip_if_lt_x_gpu(2)
    # 参数化测试装饰器，用于多次运行同一测试方法，每次传入不同的参数
    @parametrize("has_wrapping", [True, False])
    @parametrize(
        "sharding_strategy",
        [
            ShardingStrategy.NO_SHARD,        # 不进行分片的策略
            ShardingStrategy.FULL_SHARD,      # 完全分片的策略
            ShardingStrategy.SHARD_GRAD_OP,   # 梯度操作分片的策略
        ],
    )
    # 测试方法，用于测试 bf16_hook 功能
    def test_bf16_hook(
        self, has_wrapping: bool, sharding_strategy: Optional[ShardingStrategy]
    ):
        # 创建默认的低精度状态，使用默认进程组
        state = default_hooks.LowPrecisionState(process_group=_get_default_group())
        # 设置 bf16_compress_hook 作为测试钩子
        hook = default_hooks.bf16_compress_hook
    
        # 调用私有方法，检查低精度钩子的行为是否符合预期
        self._check_low_precision_hook(
            state, hook, sharding_strategy, torch.bfloat16, has_wrapping
        )
# 实例化参数化测试用例，并传入 TestCommunicationHooks 类型作为参数
instantiate_parametrized_tests(TestCommunicationHooks)

# 如果当前脚本作为主程序执行，则运行测试
if __name__ == "__main__":
    run_tests()
```