# `.\pytorch\test\distributed\checkpoint\fsdp\test_fsdp_dsd.py`

```py
# Owner(s): ["oncall: distributed"]

import copy  # 导入 copy 模块，用于对象的深拷贝操作

import torch  # 导入 PyTorch 库
import torch.distributed.checkpoint as dcp  # 导入分布式检查点模块
import torch.nn as nn  # 导入神经网络模块
from torch.distributed._composable.fsdp import fully_shard  # 导入分布式数据并行模块中的 fully_shard 函数
from torch.distributed._tensor import DTensor  # 导入分布式张量模块中的 DTensor 类
from torch.distributed.checkpoint.state_dict import (  # 导入检查点状态字典模块中的特定函数和类
    get_model_state_dict,
    get_optimizer_state_dict,
    StateDictOptions,
)

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType  # 导入完全分片数据并行模块
from torch.distributed.fsdp.wrap import always_wrap_policy  # 导入包装策略函数
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu  # 导入分布式测试通用函数
from torch.testing._internal.common_fsdp import FSDPTest, MLP  # 导入 FSDP 测试通用模块和 MLP 模型
from torch.testing._internal.common_utils import run_tests  # 导入测试通用函数
from torch.testing._internal.distributed.checkpoint_utils import with_temp_dir  # 导入用于临时目录处理的函数
from torch.utils._pytree import tree_all_only  # 导入 pytree 模块中的特定函数

class TestFullyShardWithDistributedStateDict(FSDPTest):
    @property
    def world_size(self) -> int:
        return min(4, torch.cuda.device_count())  # 返回当前设备数量和4的最小值

    def _get_base_model(self, mlp_dim: int = 2):
        base_model = nn.Sequential(  # 创建基础模型，使用 nn.Sequential 定义的模型序列
            MLP(mlp_dim),  # 添加一个 MLP 模块
            nn.Sequential(MLP(mlp_dim), nn.Linear(mlp_dim, mlp_dim)),  # 添加一个 MLP 模块和线性层
            MLP(mlp_dim),  # 添加另一个 MLP 模块
        )
        return base_model  # 返回定义好的基础模型

    @skip_if_lt_x_gpu(2)
    def test_1d_fsdp_get_model_state_dict(self):
        self.run_subtests(  # 执行子测试，测试不同参数组合下的函数
            {"mlp_dim": [2, 3, 4, 5]},  # 定义参数字典
            self._test_1d_fsdp_get_model_state_dict,  # 指定要测试的函数
        )

    def _test_1d_fsdp_get_model_state_dict(self, mlp_dim: int):
        """
        Test model.state_dict() and distributed_state_dict parity.
        """
        base_model = self._get_base_model(mlp_dim)  # 获取指定维度的基础模型
        # Default is `reshard_after_forward=True`
        model1 = copy.deepcopy(base_model)  # 深拷贝基础模型，创建第一个模型对象
        for module in model1:
            fully_shard(module)  # 对模型中的每个模块进行完全分片
        fully_shard(model1)  # 对整个模型进行完全分片

        # osd: original state dict, dsd: distributed state dict
        osd = model1.state_dict()  # 获取第一个模型的原始状态字典
        dsd = get_model_state_dict(model1)  # 获取第一个模型的分布式状态字典
        self.assertEqual(osd, dsd)  # 断言原始状态字典和分布式状态字典相等

        # Check `reshard_after_forward=False` after a forward
        model2 = copy.deepcopy(base_model)  # 深拷贝基础模型，创建第二个模型对象
        for module in model2:
            fully_shard(module, reshard_after_forward=False)  # 对模型中的每个模块进行完全分片，但不在前向后重分片
        fully_shard(model2, reshard_after_forward=False)  # 对整个模型进行完全分片，但不在前向后重分片
        inp = torch.randn((2, mlp_dim), device="cuda")  # 创建一个指定设备上的随机张量
        model2(inp)  # 对模型输入进行前向传播，参数在此次前向后不会重分片
        # Check that state dict hooks reshard
        osd_2 = model2.state_dict()  # 获取第二个模型的状态字典
        dsd_2 = get_model_state_dict(model2)  # 获取第二个模型的分布式状态字典
        self.assertEqual(osd_2, dsd_2)  # 断言第二个模型的状态字典和分布式状态字典相等

    @skip_if_lt_x_gpu(2)
    def test_1d_fsdp_cpu_offload_full_model_state_dict(self):
        """
        Test full_state_dict and cpu_offload works for FSDP2 state_dict.
        """
        # 获取基础模型的副本
        orig_model = self._get_base_model()
        # 深拷贝原始模型以创建FSDP模型
        fsdp_model = copy.deepcopy(orig_model)
        # 遍历FSDP模型的每个模块并分片化
        for module in fsdp_model:
            fully_shard(module)
        # 对整个FSDP模型进行分片化
        fully_shard(fsdp_model)

        # 获取原始模型的状态字典
        osd = orig_model.state_dict()
        # 获取FSDP模型的状态字典，使用全状态和CPU卸载选项
        dsd = get_model_state_dict(
            fsdp_model, options=StateDictOptions(full_state_dict=True, cpu_offload=True)
        )

        # 定义CPU设备
        cpu_device = torch.device("cpu")

        # 判断是否为CPU张量
        def is_cpu(v):
            if isinstance(v, DTensor):
                return v.device == torch.device("cpu")
            else:
                return v.device == cpu_device

        # 如果当前进程的等级是0
        if self.rank == 0:
            # 断言原始模型的状态字典与FSDP模型的状态字典相等
            self.assertEqual(osd, dsd)
            # 确保状态字典中所有的张量类型为torch.Tensor或DTensor，并且在CPU上
            self.assertTrue(tree_all_only((torch.Tensor, DTensor), is_cpu, osd))
        else:
            # 断言FSDP模型的状态字典为空字典
            self.assertEqual(dsd, {})

    @skip_if_lt_x_gpu(2)
    def test_save_with_fsdp1_and_load_with_fsdp2(self):
        # 运行子测试，测试在FSDP1保存并在FSDP2加载
        self.run_subtests(
            {
                "state_dict_type": [
                    StateDictType.FULL_STATE_DICT,
                    StateDictType.SHARDED_STATE_DICT,
                ]
            },
            self._test_save_with_fsdp1_and_load_with_fsdp2,
        )

    @skip_if_lt_x_gpu(2)
    @with_temp_dir
    def _test_save_with_fsdp1_and_load_with_fsdp2(self, state_dict_type: StateDictType):
        """
        Test that we can save a model with FSDP1 and load it with FSDP2.
        """

        # Save state dict with model wrapped with FSDP1
        fsdp1_model = FSDP(
            self._get_base_model().cuda(),
            use_orig_params=True,
            auto_wrap_policy=always_wrap_policy,
        )
        # 使用 FSDP1 封装模型，并将其移动到 GPU 上

        fsdp1_optim = torch.optim.Adam(fsdp1_model.parameters(), lr=0.1)
        # 创建基于 FSDP1 模型参数的 Adam 优化器

        fsdp1_model(torch.randn((2,), device=self.rank)).sum().backward()
        # 对 FSDP1 模型进行前向传播、求和和反向传播

        fsdp1_optim.step()
        # 使用优化器更新 FSDP1 模型的参数

        with FSDP.state_dict_type(fsdp1_model, state_dict_type):
            fsdp1_state_dict = {
                "model": fsdp1_model.state_dict(),
                "optim": FSDP.sharded_optim_state_dict(fsdp1_model, fsdp1_optim),
            }
            # 使用指定的状态字典类型保存 FSDP1 模型的状态字典和优化器状态

            dcp.save(
                fsdp1_state_dict,
                checkpoint_id=self.temp_dir,
            )

        fsdp1_full_msd = get_model_state_dict(
            fsdp1_model,
            options=StateDictOptions(full_state_dict=True, cpu_offload=True),
        )
        # 获取 FSDP1 模型的完整状态字典，包括 CPU 卸载选项

        fsdp1_full_osd = get_optimizer_state_dict(
            fsdp1_model,
            fsdp1_optim,
            options=StateDictOptions(full_state_dict=True, cpu_offload=True),
        )
        # 获取 FSDP1 模型关联的优化器的完整状态字典，包括 CPU 卸载选项

        # Load state dict into model with FSDP2 applied
        fsdp2_model = self._get_base_model()
        for module in fsdp2_model:
            fully_shard(module)
        fully_shard(fsdp2_model)
        # 将 FSDP2 应用到模型的所有模块和模型本身

        fsdp2_optim = torch.optim.Adam(fsdp2_model.parameters(), lr=0.1)
        # 创建基于 FSDP2 模型参数的 Adam 优化器

        fsdp2_state_dict = {
            "model": get_model_state_dict(fsdp2_model),
            "optim": get_optimizer_state_dict(fsdp2_model, fsdp2_optim),
        }
        # 获取 FSDP2 模型和优化器的状态字典

        dcp.load(
            fsdp2_state_dict,
            checkpoint_id=self.temp_dir,
        )
        # 加载状态字典到带有 FSDP2 的模型中

        fsdp2_model.load_state_dict(fsdp2_state_dict["model"])
        fsdp2_optim.load_state_dict(fsdp2_state_dict["optim"])
        # 加载 FSDP2 模型和优化器的状态字典

        fsdp2_full_msd = get_model_state_dict(
            fsdp2_model,
            options=StateDictOptions(full_state_dict=True, cpu_offload=True),
        )
        # 获取 FSDP2 模型的完整状态字典，包括 CPU 卸载选项

        fsdp2_full_osd = get_optimizer_state_dict(
            fsdp2_model,
            fsdp2_optim,
            options=StateDictOptions(full_state_dict=True, cpu_offload=True),
        )
        # 获取 FSDP2 模型关联的优化器的完整状态字典，包括 CPU 卸载选项

        # Compare full state dict to make sure they are the same.
        self.assertEqual(fsdp2_full_msd, fsdp1_full_msd)
        self.assertEqual(fsdp1_full_osd, fsdp2_full_osd)
        # 比较完整的状态字典以确保它们相同
# 如果当前脚本作为主程序运行
if __name__ == "__main__":
    # 调用函数运行测试函数
    run_tests()
```