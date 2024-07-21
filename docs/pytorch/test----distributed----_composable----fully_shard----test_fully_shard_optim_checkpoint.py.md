# `.\pytorch\test\distributed\_composable\fully_shard\test_fully_shard_optim_checkpoint.py`

```
# 导入必要的库
import copy
import itertools
import sys

import torch
import torch.distributed as dist
from torch.distributed._composable import fully_shard
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from torch.testing._internal.common_dist_composable import (
    CompositeParamModel,
    UnitModule,
)
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import FSDPTest
from torch.testing._internal.common_utils import run_tests, TEST_WITH_DEV_DBG_ASAN

# 检查是否支持分布式训练，如果不支持则跳过测试
if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)

# 检查是否在进行 ASAN 调试，如果是则跳过测试
if TEST_WITH_DEV_DBG_ASAN:
    print(
        "Skip dev-asan as torch + multiprocessing spawn have known issues",
        file=sys.stderr,
    )
    sys.exit(0)

# 设置优化器类和学习率
_optim_cls = torch.optim.Adam
_optim_lr = 1e-2

# 定义测试类 TestOptimStateCheckpointing，用于测试优化器状态检查点
class TestOptimStateCheckpointing(FSDPTest):
    """Tests ``fully_shard`` optimizer state checkpointing."""

    # 定义属性 world_size，返回值为 2，表示测试的世界大小为 2
    @property
    def world_size(self) -> int:
        return 2
    # 定义测试函数，用于测试优化器状态的保存和加载功能
    def _test_optim_state_save_load(self, model1, optim1, model2, optim2) -> None:
        # 创建一个随机数据张量作为输入批次，存储在 GPU 上
        batch = torch.randn(2, 100, device="cuda")
        # 遍历每对模型和优化器
        for model, optim in (
            (model1, optim1),
            (model2, optim2),
        ):
            # 清零梯度，设置为None
            optim.zero_grad(set_to_none=True)
            # 将模型对批次的输出求和并反向传播
            model(batch).sum().backward()
            # 执行优化步骤
            optim.step()

        # 获取优化器的状态字典
        o1_sd, o2_sd = optim1.state_dict(), optim2.state_dict()
        # 使用FSDP扩展类获取模型和优化器的优化器状态字典
        optim_state_dict1 = FSDP.optim_state_dict(model1, optim1)
        optim_state_dict2 = FSDP.optim_state_dict(model2, optim2)

        # 断言两个优化器状态字典的长度相等
        self.assertEqual(
            len(optim_state_dict1["state"]), len(optim_state_dict2["state"])
        )
        # 遍历并断言每个完全限定名称和其状态在两个优化器状态字典中相等
        for fqn, state in optim_state_dict1["state"].items():
            self.assertEqual(state, optim_state_dict2["state"][fqn], fqn)

        # 使用itertools进行并行遍历，对比参数组中的每个键值对是否相等
        for group1, group2 in itertools.zip_longest(
            optim_state_dict1["param_groups"], optim_state_dict2["param_groups"]
        ):
            for key, value in group1.items():
                self.assertEqual(value, group2[key])

        # 使用给定的学习率创建重新加载的优化器实例
        reload_o1 = _optim_cls(model1.parameters(), lr=_optim_lr)
        reload_o2 = _optim_cls(model2.parameters(), lr=_optim_lr)
        # 从FSDP优化器状态字典加载模型和优化器的状态
        fsdp_o1_load = FSDP.optim_state_dict_to_load(
            model1, optim1, optim_state_dict1, is_named_optimizer=False
        )
        reload_o1.load_state_dict(fsdp_o1_load)
        fsdp_o2_load = FSDP.optim_state_dict_to_load(
            model2, optim2, optim_state_dict2, is_named_optimizer=False
        )
        reload_o2.load_state_dict(fsdp_o2_load)
        # 获取重新加载后的优化器的状态字典
        reload_o1_sd, reload_o2_sd = reload_o1.state_dict(), reload_o2.state_dict()
        # 遍历两个原始状态字典和重新加载后的状态字典，断言它们的键值对应相等
        for sd_pair in [(o1_sd, reload_o1_sd), (o2_sd, reload_o2_sd)]:
            sd1, sd2 = sd_pair
            for (k1, v1), (k2, v2) in zip(sd1.items(), sd2.items()):
                self.assertEqual(k1, k2, f"Mismatched keys: {k1} vs {k2}")
                self.assertEqual(v1, v2, f"Mismatched values {v1} vs {v2}")

    # 如果 GPU 数量小于 2，则跳过测试
    @skip_if_lt_x_gpu(2)
    def test_optim_state_dict_save_load(self):
        # 创建原始模型并进行深度复制
        orig_model = CompositeParamModel(device=torch.device("cuda"))
        composable_model = copy.deepcopy(orig_model)
        # 将模型完全分片，并使用给定的策略包装模块
        fully_shard(composable_model, policy=ModuleWrapPolicy({UnitModule}))
        # 使用给定的学习率创建优化器实例
        composable_optim = _optim_cls(composable_model.parameters(), lr=_optim_lr)
        # 将原始模型转换为FSDP模型
        orig_model = FSDP(orig_model)
        # 使用给定的学习率创建优化器实例
        orig_optim = _optim_cls(orig_model.parameters(), lr=_optim_lr)

        # 调用测试函数，测试优化器状态的保存和加载
        self._test_optim_state_save_load(
            orig_model, orig_optim, composable_model, composable_optim
        )
    # 定义测试函数，用于测试优化器状态字典在完全分片子模块上的行为
    def test_optim_state_dict_submodule_fully_shard(self):
        # 创建原始模型实例，使用 CUDA 设备
        orig_model = CompositeParamModel(device=torch.device("cuda"))
        # 使用深拷贝创建可组合模型实例，复制原始模型的所有属性和子模块
        composable_model = copy.deepcopy(orig_model)
        # 对可组合模型的第一个子模块进行完全分片
        fully_shard(composable_model.u1)
        # 对可组合模型的第二个子模块进行完全分片
        fully_shard(composable_model.u2)
        # 使用给定的优化器类和学习率创建可组合模型的优化器实例
        composable_optim = _optim_cls(composable_model.parameters(), lr=_optim_lr)
        # 将原始模型包装为 FSDP 模型（Fully Sharded DataParallel），以支持模型参数的完全分片
        orig_model = FSDP(orig_model)
        # 使用给定的优化器类和学习率创建原始模型的优化器实例
        orig_optim = _optim_cls(orig_model.parameters(), lr=_optim_lr)

        # 调用测试方法，验证优化器状态保存和加载在原始模型和可组合模型之间的正确性
        self._test_optim_state_save_load(
            orig_model, orig_optim, composable_model, composable_optim
        )
# 如果当前脚本被直接执行（而不是被导入作为模块），则执行下面的代码块
if __name__ == "__main__":
    # 调用名为 run_tests 的函数，用于执行测试或程序的主要功能
    run_tests()
```