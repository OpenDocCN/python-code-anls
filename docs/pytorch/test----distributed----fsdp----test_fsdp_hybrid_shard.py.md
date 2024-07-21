# `.\pytorch\test\distributed\fsdp\test_fsdp_hybrid_shard.py`

```
# Owner(s): ["oncall: distributed"]

import contextlib  # 导入上下文管理器相关模块
import sys  # 导入系统模块
from collections import Counter  # 导入计数器集合模块
from enum import auto, Enum  # 导入枚举相关模块
from functools import partial  # 导入偏函数模块
from typing import List, Optional, Tuple  # 导入类型提示相关模块

import torch  # 导入PyTorch深度学习库
import torch.distributed as dist  # 导入PyTorch分布式训练模块
import torch.distributed.fsdp._traversal_utils as traversal_utils  # 导入FSDP模块的遍历工具
import torch.nn as nn  # 导入PyTorch神经网络模块
from torch.distributed.device_mesh import init_device_mesh  # 导入设备网格初始化模块
from torch.distributed.distributed_c10d import _rank_not_in_group  # 导入分布式C10D模块中的排名不在组中判断
from torch.distributed.fsdp import (  # 导入FSDP相关模块
    FullyShardedDataParallel as FSDP,  # 导入全分片数据并行模块
    ShardingStrategy,  # 导入分片策略模块
    StateDictType,  # 导入状态字典类型模块
)
from torch.distributed.fsdp._init_utils import (  # 导入FSDP初始化工具模块
    _init_intra_and_inter_node_groups,  # 导入内部和节点间组初始化函数
    HYBRID_SHARDING_STRATEGIES,  # 导入混合分片策略列表
)

from torch.distributed.fsdp.wrap import ModuleWrapPolicy  # 导入模块封装策略
from torch.nn import TransformerDecoderLayer, TransformerEncoderLayer  # 导入Transformer编解码层模块
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu  # 导入分布式测试模块中的GPU数量判断函数
from torch.testing._internal.common_fsdp import (  # 导入FSDP常用功能测试模块
    CUDAInitMode,  # 导入CUDA初始化模式
    FSDPInitMode,  # 导入FSDP初始化模式
    FSDPTest,  # 导入FSDP测试类
    TransformerWithSharedParams,  # 导入具有共享参数的Transformer模块
)
from torch.testing._internal.common_utils import (  # 导入通用测试内部工具模块
    instantiate_parametrized_tests,  # 导入实例化参数化测试函数
    run_tests,  # 导入运行测试函数
    TEST_WITH_DEV_DBG_ASAN,  # 导入是否使用开发者调试ASAN的标志
)

if not dist.is_available():  # 如果分布式不可用
    print("Distributed not available, skipping tests", file=sys.stderr)  # 打印消息并输出到标准错误流
    sys.exit(0)  # 退出程序，返回状态码0

if TEST_WITH_DEV_DBG_ASAN:  # 如果设置了开发者调试ASAN标志
    print(
        "Skip dev-asan as torch + multiprocessing spawn have known issues",
        file=sys.stderr,
    )  # 打印消息并输出到标准错误流
    sys.exit(0)  # 退出程序，返回状态码0


@contextlib.contextmanager
def patch_allreduce(new_allreduce):
    """
    Patches dist.all_reduce with a new all_reduce and
    restores upon exiting.
    """
    orig_ar = dist.all_reduce  # 备份原始的dist.all_reduce函数
    dist.all_reduce = new_allreduce  # 使用新的all_reduce函数替换dist.all_reduce
    try:
        yield  # 执行代码块
    finally:
        dist.all_reduce = orig_ar  # 恢复原始的dist.all_reduce函数


@contextlib.contextmanager
def patch_reduce_scatter(new_reduce_scatter):
    """
    Patches dist.reduce_scatter_tensor with a new reduce_scatter_tensor and
    restores upon exiting.
    """
    orig_reduce_scatter = dist.reduce_scatter_tensor  # 备份原始的dist.reduce_scatter_tensor函数
    dist.reduce_scatter_tensor = new_reduce_scatter  # 使用新的reduce_scatter_tensor函数替换dist.reduce_scatter_tensor
    try:
        yield  # 执行代码块
    finally:
        dist.reduce_scatter_tensor = orig_reduce_scatter  # 恢复原始的dist.reduce_scatter_tensor函数


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()  # 调用父类构造函数
        self.lin1 = nn.Linear(10, 10)  # 定义线性层1
        self.lin2 = nn.Linear(10, 10)  # 定义线性层2
        self.lin3 = nn.Linear(10, 10)  # 定义线性层3

    def forward(self, x):
        return self.lin3(self.lin2(self.lin1(x)))  # 模型的前向传播逻辑


class ShardingStrategyMode(Enum):
    ALL_HYBRID_SHARD = auto()  # 所有混合分片模式枚举
    MIXED_HYBRID_FULL_SHARD = auto()  # 混合和完全分片模式枚举


class TestFSDPHybridShard(FSDPTest):
    @property
    def world_size(self):
        return max(torch.cuda.device_count(), 2)  # 返回当前设备的GPU数量或最小为2

    @property
    def process_group(self):
        return dist.distributed_c10d._get_default_group()  # 返回默认的进程组

    @skip_if_lt_x_gpu(2)  # 如果GPU数量小于2，则跳过测试
    def test_raises_manual_wrap_hybrid_shard_when_none_policy(self):
        # 创建一个在 CUDA 上的模型实例
        model = MyModel().cuda()
        # 设置错误上下文，期望捕获 ValueError 异常，并验证错误消息内容
        err_ctx = self.assertRaisesRegex(
            ValueError,
            "requires explicit specification of process group or device_mesh.",
        )

        # 测试用例：捕获 ValueError 异常，确保在指定混合分片策略时会触发异常
        with err_ctx:
            model = FSDP(model, sharding_strategy=ShardingStrategy.HYBRID_SHARD)

        # 再次使用相同的错误上下文，验证另一种混合分片策略是否也会触发相同异常
        with err_ctx:
            model = FSDP(model, sharding_strategy=ShardingStrategy._HYBRID_SHARD_ZERO2)

    @skip_if_lt_x_gpu(4)
    def test_hsdp_save_load_state_dict(self):
        # 创建一个在 CUDA 上的模型实例
        model = MyModel().cuda()
        # 获取当前 CUDA 设备数量
        num_node_devices = torch.cuda.device_count()
        # 计算每个分片组的设备列表，以及分片组的分组
        shard_rank_lists = list(range(0, num_node_devices // 2)), list(
            range(num_node_devices // 2, num_node_devices)
        )
        shard_groups = (
            dist.new_group(shard_rank_lists[0]),
            dist.new_group(shard_rank_lists[1]),
        )
        # 确定当前进程所属的分片组
        my_shard_group = (
            shard_groups[0] if self.rank in shard_rank_lists[0] else shard_groups[1]
        )
        my_replicate_group = None
        my_rank = self.rank
        # 创建分片因子，并为每个分片组分配复制组
        shard_factor = len(shard_rank_lists[0])
        for i in range(num_node_devices // 2):
            replicate_group_ranks = list(range(i, num_node_devices, shard_factor))
            replicate_group = dist.new_group(replicate_group_ranks)
            if my_rank in replicate_group_ranks:
                my_replicate_group = replicate_group

        # 部分应用构造函数，创建具有特定参数的 FSDP 实例
        fsdp_ctor = partial(
            FSDP,
            sharding_strategy=ShardingStrategy.HYBRID_SHARD,
            use_orig_params=True,
            process_group=(my_shard_group, my_replicate_group),
        )
        model = fsdp_ctor(model)
        # 创建 AdamW 优化器
        optim = torch.optim.AdamW(model.parameters())
        # 初始化优化器状态
        model(torch.randn(2, 10)).sum().backward()
        optim.step()
        # 获取分片组和复制组
        shard_g = model.process_group
        replicate_g = model._inter_node_pg
        # 断言分片组和复制组与预期相符
        assert shard_g == my_shard_group
        assert replicate_g == my_replicate_group
        # 使用分片状态字典类型处理模型和优化器状态
        with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
            msd = model.state_dict()
            osd = FSDP.optim_state_dict(model, optim)

        # 加载模型并创建新的优化器
        load_model = fsdp_ctor(MyModel().cuda())
        load_optim = torch.optim.AdamW(load_model.parameters())
        # 使用分片状态字典类型加载模型状态和优化器状态
        with FSDP.state_dict_type(load_model, StateDictType.SHARDED_STATE_DICT):
            load_model.load_state_dict(msd)
            FSDP.optim_state_dict_to_load(load_model, load_optim, osd)
        load_optim.load_state_dict(osd)

    @skip_if_lt_x_gpu(4)
    # 定义测试函数，用于测试模型状态同步功能
    def test_hsdp_sync_module_state(self):
        # 创建模型实例并将其移动到 CUDA 设备上
        model = MyModel().cuda()
        # 获取当前 CUDA 设备数量
        num_node_devices = torch.cuda.device_count()
        # 分片的排名列表，分为两组
        shard_rank_lists = list(range(0, num_node_devices // 2)), list(
            range(num_node_devices // 2, num_node_devices)
        )
        # 创建分片组
        shard_groups = (
            dist.new_group(shard_rank_lists[0]),
            dist.new_group(shard_rank_lists[1]),
        )
        # 根据当前进程的排名确定所属的分片组
        my_shard_group = (
            shard_groups[0] if self.rank in shard_rank_lists[0] else shard_groups[1]
        )
        # 初始化复制组变量
        my_replicate_group = None
        # 获取当前进程的排名
        my_rank = self.rank
        # 创建复制组，每个设备创建一个复制组
        shard_factor = len(shard_rank_lists[0])
        for i in range(num_node_devices // 2):
            replicate_group_ranks = list(range(i, num_node_devices, shard_factor))
            replicate_group = dist.new_group(replicate_group_ranks)
            # 如果当前进程在复制组的排名列表中，则将其设为当前进程的复制组
            if my_rank in replicate_group_ranks:
                my_replicate_group = replicate_group

        # 初始化模型的权重参数为当前进程的排名
        nn.init.constant_(model.lin1.weight, self.rank)
        nn.init.constant_(model.lin2.weight, self.rank)
        nn.init.constant_(model.lin3.weight, self.rank)

        # 配置 FSDP 的构造函数
        fsdp_ctor = partial(
            FSDP,
            sharding_strategy=ShardingStrategy.HYBRID_SHARD,
            use_orig_params=True,
            sync_module_states=True,
            process_group=(my_shard_group, my_replicate_group),
        )
        # 使用 FSDP 包装模型
        model = fsdp_ctor(model)

        # 进入 FSDP 状态字典管理器，验证模型的权重参数是否全部为 0
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT):
            self.assertTrue((model.lin1.weight == 0).all())
            self.assertTrue((model.lin2.weight == 0).all())
            self.assertTrue((model.lin3.weight == 0).all())

    # 根据 GPU 数量条件跳过测试
    @skip_if_lt_x_gpu(2)
    def test_invalid_pg_specification_raises(self):
        # 创建模型包装策略
        pol = ModuleWrapPolicy({nn.Linear})
        # 创建模型实例并将其移动到 CUDA 设备上
        model = MyModel().cuda()
        # 验证异常是否被正确地抛出
        with self.assertRaisesRegex(
            ValueError, "Expected process_group to be passed in"
        ):
            # 使用 FSDP 包装模型，验证是否能正确捕获异常
            model = FSDP(
                model,
                auto_wrap_policy=pol,
                process_group=self.process_group,
                sharding_strategy=ShardingStrategy.HYBRID_SHARD,
            )

    # TODO - add test for ZeRO-2 style sharding ensure params are not
    # resharded after forward.

    # 根据 GPU 数量条件跳过测试
    @skip_if_lt_x_gpu(2)
    def test_fsdp_hybrid_shard_basic_setup(self):
        """
        Tests basic functionality of HYBRID_SHARD and _HYBRID_SHARD_ZERO2:
            1. Inter and intra-node process groups are correctly setup
            2. Process groups are the same across FSDP wrapped instances
            3. reduce_scatter and allreduce called the expected no. of times
        """
        # 运行子测试，测试 HYBRID_SHARD 和 _HYBRID_SHARD_ZERO2 的基本功能
        self.run_subtests(
            {
                "hsdp_sharding_strategy": [
                    ShardingStrategy.HYBRID_SHARD,
                    ShardingStrategy._HYBRID_SHARD_ZERO2,
                ],
                "sharding_strategy_mode": [
                    ShardingStrategyMode.ALL_HYBRID_SHARD,
                    ShardingStrategyMode.MIXED_HYBRID_FULL_SHARD,
                ],
                "use_orig_params": [False, True],
                "use_device_mesh": [False, True],
            },
            self._test_fsdp_hybrid_shard_basic_setup,
        )

    def _test_fsdp_hybrid_shard_basic_setup(
        self,
        hsdp_sharding_strategy: ShardingStrategy,
        sharding_strategy_mode: ShardingStrategyMode,
        use_orig_params: bool,
        use_device_mesh: bool,
    ):
        # 跳过如果 GPU 少于 4 个的情况
        @skip_if_lt_x_gpu(4)
        def test_fsdp_hybrid_shard_parity(self):
            self.run_subtests(
                {
                    "hsdp_sharding_strategy": [
                        ShardingStrategy.HYBRID_SHARD,
                        ShardingStrategy._HYBRID_SHARD_ZERO2,
                    ],
                    "use_orig_params": [False, True],
                },
                self._test_fsdp_hybrid_shard_parity,
            )

        def _test_fsdp_hybrid_shard_parity(
            self, hsdp_sharding_strategy: ShardingStrategy, use_orig_params: bool
        ):
            # 初始化 FSDP 模型
            fsdp_model = self._init_fsdp_model(use_orig_params)
            # 获取全局进程组
            global_pg = dist.distributed_c10d._get_default_group()
            # 初始化 HYBRID_SHARD 的内部和跨节点组
            hsdp_pgs = _init_intra_and_inter_node_groups(global_pg, 2)
            # 初始化 HSDP 模型
            hsdp_model = self._init_hsdp_model(
                hsdp_sharding_strategy,
                ShardingStrategyMode.ALL_HYBRID_SHARD,
                use_orig_params,
                hsdp_process_groups=hsdp_pgs,
            )
            # 断言 HSDP 模型是否初始化为非单个副本
            assert (
                hsdp_model._inter_node_pg.size() > 1
            ), "HSDP model initialized without replication"
            # 初始化优化器
            fsdp_optim = torch.optim.Adam(fsdp_model.parameters(), lr=1e-2)
            hsdp_optim = torch.optim.Adam(hsdp_model.parameters(), lr=1e-2)
            # 设定随机种子
            torch.manual_seed(global_pg.rank() + 1)
            # 执行五次训练迭代
            for _ in range(5):
                # 获取输入数据
                inp = fsdp_model.module.get_input(torch.device("cuda"))
                # 存储损失值列表
                losses: List[torch.Tensor] = []
                # 遍历模型和优化器对
                for model, optim in ((fsdp_model, fsdp_optim), (hsdp_model, hsdp_optim)):
                    # 清空梯度
                    optim.zero_grad()
                    # 计算损失值
                    loss = model(*inp).sum()
                    losses.append(loss)
                    # 反向传播
                    loss.backward()
                    # 执行优化步骤
                    optim.step()
                # 断言两个模型的损失值是否相等
                self.assertEqual(losses[0], losses[1])
    # 初始化一个使用 FSDP（Fully Sharded Data Parallel）的 Transformer 模型
    def _init_fsdp_model(self, use_orig_params: bool) -> nn.Module:
        # 定义自动包装策略，适用于 TransformerEncoderLayer 和 TransformerDecoderLayer
        auto_wrap_policy = ModuleWrapPolicy(
            {TransformerEncoderLayer, TransformerDecoderLayer},
        )
        # 设置 FSDP 的初始化参数
        hsdp_kwargs = {
            "auto_wrap_policy": auto_wrap_policy,  # 自动包装策略
            "device_id": torch.cuda.current_device(),  # 当前 CUDA 设备 ID
            "use_orig_params": use_orig_params,  # 是否使用原始参数
        }
        # 使用 TransformerWithSharedParams 类初始化 FSDP 模型
        fsdp_model = TransformerWithSharedParams.init(
            self.process_group,
            FSDPInitMode.RECURSIVE,  # 递归初始化模式
            CUDAInitMode.CUDA_BEFORE,  # CUDA 初始化模式
            hsdp_kwargs,
            deterministic=True,  # 确定性初始化
        )
        return fsdp_model

    # 初始化一个使用 HSDP（Hybrid Sharded Data Parallel）的 Transformer 模型
    def _init_hsdp_model(
        self,
        hsdp_sharding_strategy: ShardingStrategy,
        sharding_strategy_mode: str,
        use_orig_params: bool,
        hsdp_process_groups: Optional[
            Tuple[dist.ProcessGroup, dist.ProcessGroup]
        ] = None,
        hsdp_device_mesh: Optional = None,
    ):
        # 断言条件：HSDP 的处理组和设备网格只能选择一个为 None
        assert hsdp_process_groups is None or hsdp_device_mesh is None
        # 定义自动包装策略，适用于 TransformerEncoderLayer 和 TransformerDecoderLayer
        auto_wrap_policy = ModuleWrapPolicy(
            {TransformerEncoderLayer, TransformerDecoderLayer},
        )
        # 设置 HSDP 的初始化参数
        hsdp_kwargs = {
            "device_id": torch.cuda.current_device(),  # 当前 CUDA 设备 ID
            "auto_wrap_policy": auto_wrap_policy,  # 自动包装策略
            "sharding_strategy": hsdp_sharding_strategy,  # 分片策略
            "use_orig_params": use_orig_params,  # 是否使用原始参数
            "device_mesh": hsdp_device_mesh,  # 设备网格
        }
        # 根据不同的分片策略模式选择初始化模型
        if sharding_strategy_mode == ShardingStrategyMode.ALL_HYBRID_SHARD:
            # 使用 TransformerWithSharedParams 类初始化 HSDP 模型
            hsdp_model = TransformerWithSharedParams.init(
                hsdp_process_groups or self.process_group,
                FSDPInitMode.RECURSIVE,  # 递归初始化模式
                CUDAInitMode.CUDA_BEFORE,  # CUDA 初始化模式
                hsdp_kwargs,
                deterministic=True,  # 确定性初始化
            )
        elif sharding_strategy_mode == ShardingStrategyMode.MIXED_HYBRID_FULL_SHARD:
            # 使用 TransformerWithSharedParams 类初始化模型
            model = TransformerWithSharedParams.init(
                hsdp_process_groups or self.process_group,
                FSDPInitMode.NO_FSDP,  # 不使用 FSDP 初始化模式
                CUDAInitMode.CUDA_BEFORE,  # CUDA 初始化模式
                {},  # 空的初始化参数字典
                deterministic=True,  # 确定性初始化
            )
            # 将 transformer 模块替换为 FSDP 封装后的模块
            model.transformer = FSDP(model.transformer, **hsdp_kwargs)
            # 使用 `FULL_SHARD` 策略对嵌入和输出投影进行全分片
            hsdp_model = FSDP(
                model,
                device_id=torch.cuda.current_device(),  # 当前 CUDA 设备 ID
                sharding_strategy=ShardingStrategy.FULL_SHARD,  # 全分片策略
                use_orig_params=use_orig_params,  # 是否使用原始参数
            )
        return hsdp_model
# 实例化一个参数化测试，使用 TestFSDPHybridShard 类作为参数
instantiate_parametrized_tests(TestFSDPHybridShard)

# 如果当前脚本作为主程序执行，则运行测试函数
if __name__ == "__main__":
    run_tests()
```