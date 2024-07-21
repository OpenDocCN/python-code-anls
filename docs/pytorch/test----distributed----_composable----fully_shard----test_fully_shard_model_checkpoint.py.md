# `.\pytorch\test\distributed\_composable\fully_shard\test_fully_shard_model_checkpoint.py`

```
# Owner(s): ["oncall: distributed"]

# 导入必要的库和模块
import copy  # 导入深拷贝模块
import itertools  # 导入迭代工具模块
import sys  # 导入系统模块
from typing import Dict  # 导入类型提示中的字典类型

import torch  # 导入PyTorch库
import torch.distributed as dist  # 导入PyTorch分布式模块
import torch.nn as nn  # 导入PyTorch神经网络模块
from torch.distributed._composable import fully_shard  # 导入完全分片函数
from torch.distributed._state_dict_utils import _gather_state_dict  # 导入状态字典工具函数
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType  # 导入完全分片数据并行和状态字典类型
from torch.distributed.fsdp.api import ShardingStrategy  # 导入分片策略
from torch.distributed.fsdp.wrap import ModuleWrapPolicy  # 导入模块包装策略
from torch.nn import TransformerDecoderLayer, TransformerEncoderLayer  # 导入Transformer编码器和解码器层
from torch.testing._internal.common_dist_composable import (
    CompositeParamModel,  # 导入组合参数模型
    UnitModule,  # 导入单元模块
)
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu  # 导入如果GPU数小于x则跳过测试函数
from torch.testing._internal.common_fsdp import (
    _zero_model,  # 导入零模型函数
    CUDAInitMode,  # 导入CUDA初始化模式
    FSDPInitMode,  # 导入FSDP初始化模式
    FSDPTest,  # 导入FSDP测试类
    TransformerWithSharedParams,  # 导入具有共享参数的Transformer类
)
from torch.testing._internal.common_utils import run_tests, TEST_WITH_DEV_DBG_ASAN  # 导入运行测试和开发调试ASAN测试标志

# 如果分布式不可用，则打印跳过测试信息并退出程序
if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)

# 如果开启了开发调试ASAN测试标志，打印相关信息并退出程序
if TEST_WITH_DEV_DBG_ASAN:
    print(
        "Skip dev-asan as torch + multiprocessing spawn have known issues",
        file=sys.stderr,
    )
    sys.exit(0)


class TestModelCheckpointing(FSDPTest):
    """Tests ``fully_shard`` model checkpointing."""

    @property
    def world_size(self) -> int:
        return 2  # 返回世界大小为2

    @skip_if_lt_x_gpu(2)
    def test_state_dict_save_load_root_fully_shard(self):
        """
        Tests that the full state dict saved from a module with ``fully_shard``
        applied to the global root matches that of an equivalent local module. Also
        ensure that this state_dict can be reloaded into a composable module and
        is equivalent to the original composable module.
        """
        local_model = CompositeParamModel(device=torch.device("cuda"))  # 创建CUDA设备上的组合参数模型
        save_composable = copy.deepcopy(local_model)  # 深度复制本地模型
        fully_shard(save_composable, policy=ModuleWrapPolicy({UnitModule}))  # 对复制的模型进行完全分片
        local_sd = local_model.state_dict()  # 获取本地模型的状态字典
        composable_sd = save_composable.state_dict()  # 获取完全分片后模型的状态字典
        self._check_state_dict_parity(local_sd, composable_sd)  # 检查状态字典的一致性

        # 验证加载
        load_composable = fully_shard(
            copy.deepcopy(local_model), policy=ModuleWrapPolicy({UnitModule})
        )  # 对深度复制的本地模型再次进行完全分片
        _zero_model(load_composable, summon_full=False)  # 对加载的模型进行零初始化
        for p in load_composable.parameters():  # 遍历加载模型的参数
            self.assertEqual(p.sum(), 0)  # 断言参数的和为0

        sd = {k: v.clone() for k, v in composable_sd.items()}  # 克隆状态字典的每一项
        load_composable.load_state_dict(sd)  # 加载克隆后的状态字典到加载的模型
        self._check_model_parity(load_composable, save_composable)  # 检查加载后模型的一致性
    def test_state_dict_save_load_submodule_fully_shard(self):
        """
        Tests that the full state dict saved from a module with ``fully_shard``
        applied on submodules matches that of an equivalent local module. Also
        ensures that this state_dict can be reloaded into a composable module and
        is equivalent to the original composable module.
        """
        # 创建一个本地的 CompositeParamModel 模型，设备为 CUDA
        local_model = CompositeParamModel(device=torch.device("cuda"))

        def _create_fully_shard_on_submodules(mod: nn.Module):
            # 对模块中的 u1 和 u2 子模块应用 fully_shard
            fully_shard(mod.u1)
            fully_shard(mod.u2)
            return mod

        # 深拷贝 local_model 到 save_composable
        save_composable = copy.deepcopy(local_model)
        # 在 save_composable 上应用 fully_shard
        save_composable = _create_fully_shard_on_submodules(save_composable)
        # 获取 local_model 和 save_composable 的状态字典
        local_sd = local_model.state_dict()
        composable_sd = save_composable.state_dict()
        # 检查两个状态字典的一致性
        self._check_state_dict_parity(local_sd, composable_sd)

        # Validate load
        # 再次深拷贝 local_model 到 load_composable
        load_composable = copy.deepcopy(local_model)
        # 在 load_composable 上应用 fully_shard
        load_composable = _create_fully_shard_on_submodules(load_composable)
        # 将 load_composable 中的参数全部置零，summon_full 设为 False
        _zero_model(load_composable, summon_full=False)
        # 遍历 load_composable 的所有参数，确保它们的总和为零
        for p in load_composable.parameters():
            self.assertEqual(0, p.sum())

        # 深拷贝 composable_sd 到 sd，每个张量都进行克隆
        sd = {k: v.clone() for k, v in composable_sd.items()}
        # 加载 sd 到 load_composable 中
        load_composable.load_state_dict(sd)
        # 检查 load_composable 和 save_composable 的模型结构和参数是否完全一致
        self._check_model_parity(load_composable, save_composable)

    @skip_if_lt_x_gpu(2)
    def test_state_dict_save_load_flow(self):
        """
        E2E test of save + load with rank0_only + CPU offload for TransformerWithSharedParams
        on the composable path.
        """
        # 运行子测试，测试 ignore_modules 和 sharded_state_dict 的组合
        self.run_subtests(
            {"ignore_modules": [False, True], "sharded_state_dict": [False, True]},
            self._test_save_dict_save_load_flow,
        )

    def _test_save_dict_save_load_flow(
        self, ignore_modules: bool, sharded_state_dict: bool
    ):
        # 使用 TransformerWithSharedParams 类初始化本地模型，设定参数共享模式为 NO_FSDP，CUDA 初始化模式为 CUDA_BEFORE，确保确定性
        local_model = TransformerWithSharedParams.init(
            self.process_group,
            FSDPInitMode.NO_FSDP,
            CUDAInitMode.CUDA_BEFORE,
            deterministic=True,
        )

        # 强制模型参数和缓冲区非零
        for tensor in itertools.chain(local_model.parameters(), local_model.buffers()):
            if torch.count_nonzero(tensor) == 0:
                with torch.no_grad():
                    tensor.add_(torch.ones_like(tensor))

        # 深拷贝本地模型，以备保存
        save_model = copy.deepcopy(local_model)

        # 对保存的模型进行完全分片
        fully_shard(
            save_model,
            policy=ModuleWrapPolicy({TransformerEncoderLayer, TransformerDecoderLayer}),
            ignored_modules=(
                save_model.get_ignored_modules() if ignore_modules else []
            ),
        )

        # TODO: 在 https://github.com/pytorch/pytorch/issues/90954 解决之后，测试 state_dict_type
        # 设置保存模型的 state_dict 类型，根据 sharded_state_dict 的值选择 FULL_STATE_DICT 或 SHARDED_STATE_DICT
        if not sharded_state_dict:
            FSDP.set_state_dict_type(save_model, StateDictType.FULL_STATE_DICT)
        else:
            FSDP.set_state_dict_type(save_model, StateDictType.SHARDED_STATE_DICT)

        # 获取保存模型的 state_dict
        state_dict = save_model.state_dict()
        # 获取本地模型的 state_dict
        local_state_dict = local_model.state_dict()
        # 检查本地模型和保存模型 state_dict 的一致性
        self._check_state_dict_parity(local_state_dict, _gather_state_dict(state_dict))

        # 使用 TransformerWithSharedParams 类初始化加载模型，设定参数共享模式为 NO_FSDP，CUDA 初始化模式为 CUDA_BEFORE
        load_model = TransformerWithSharedParams.init(
            self.process_group,
            FSDPInitMode.NO_FSDP,
            CUDAInitMode.CUDA_BEFORE,
        )

        # 对加载的模型进行零初始化，包括缓冲区
        _zero_model(load_model, zero_buffers=True, summon_full=False)

        # 对加载的模型进行完全分片
        fully_shard(
            load_model,
            policy=ModuleWrapPolicy({TransformerDecoderLayer, TransformerEncoderLayer}),
            ignored_modules=(
                load_model.get_ignored_modules() if ignore_modules else []
            ),
        )

        # 设置加载模型的 state_dict 类型，根据 sharded_state_dict 的值选择 FULL_STATE_DICT 或 SHARDED_STATE_DICT
        if not sharded_state_dict:
            FSDP.set_state_dict_type(load_model, StateDictType.FULL_STATE_DICT)
        else:
            FSDP.set_state_dict_type(load_model, StateDictType.SHARDED_STATE_DICT)

        # 加载保存模型的 state_dict 到加载模型
        load_model.load_state_dict(state_dict)

        # 检查加载模型和保存模型的一致性
        self._check_model_parity(load_model, save_model)

    @skip_if_lt_x_gpu(2)
    def test_full_state_dict_save_load_mixed_sharding(self):
        """
        Tests that the full state dict saved from a module with ``fully_shard``
        and ``no_shard`` applied on the module matches that of an equivalent
        local module. Also ensures that this state_dict can be reloaded into
        a composable module and is equivalent to the original composable module.
        """
        # 创建一个 CUDA 设备上的 CompositeParamModel 实例
        local_model = CompositeParamModel(device=torch.device("cuda"))

        def _create_mixed_shard_on_model(mod: nn.Module):
            # 在模块的 u1 参数上应用 fully_shard
            fully_shard(mod.u1)
            # 在模块上应用 NO_SHARD 策略
            fully_shard(mod, strategy=ShardingStrategy.NO_SHARD)
            return mod

        # 深拷贝 local_model，保存初始状态
        save_composable = copy.deepcopy(local_model)
        # 在拷贝的模型上应用混合分片
        save_composable = _create_mixed_shard_on_model(save_composable)
        # 获取本地模型的状态字典
        local_sd = local_model.state_dict()
        # 获取保存后模型的状态字典
        composable_sd = save_composable.state_dict()
        # 检查状态字典的一致性
        self._check_state_dict_parity(local_sd, composable_sd)

        # 验证加载
        # 深拷贝 local_model 以便加载
        load_composable = copy.deepcopy(local_model)
        # 在加载的模型上应用混合分片
        load_composable = _create_mixed_shard_on_model(load_composable)
        # 将加载的模型所有参数置零，除非 summon_full=False
        _zero_model(load_composable, summon_full=False)
        # 检查加载后模型各参数的和是否为零
        for p in load_composable.parameters():
            self.assertEqual(0, p.sum())

        # 深拷贝保存后的状态字典
        sd = {k: v.clone() for k, v in composable_sd.items()}
        # 加载状态字典到加载的模型
        load_composable.load_state_dict(sd)
        # 检查加载后模型与保存模型的一致性
        self._check_model_parity(load_composable, save_composable)

    def _check_state_dict_parity(self, local_sd: Dict, composable_sd: Dict):
        """Checks that ``local_sd`` and ``composable_sd`` are the same."""
        # 检查所有键是否匹配
        self.assertEqual(set(composable_sd.keys()), set(local_sd.keys()))
        # 检查值的形状是否匹配
        for k in composable_sd.keys():
            v1 = composable_sd[k]
            v2 = local_sd[k]
            self.assertEqual(
                v1.shape, v2.shape, f"Shape mismatch for {k} {v1.shape} vs {v2.shape}"
            )
        # 检查实际数值是否匹配
        for k in composable_sd.keys():
            v1 = composable_sd[k]
            v2 = local_sd[k]
            self.assertEqual(v1, v2, f"Param mismatch for {k}: {v1} vs {v2}")

    def _check_model_parity(self, m1: nn.Module, m2: nn.Module):
        """
        Checks that ``m1`` and ``m2`` have equal ``named_parameters()``.
        """
        # 检查两个模型的命名参数是否相同
        for (n1, p1), (n2, p2) in zip(m1.named_parameters(), m2.named_parameters()):
            self.assertEqual(n1, n2)
            self.assertEqual(p1, p2)
# 如果当前脚本作为主程序运行（而不是作为模块被导入执行），则执行 run_tests() 函数
if __name__ == "__main__":
    run_tests()
```