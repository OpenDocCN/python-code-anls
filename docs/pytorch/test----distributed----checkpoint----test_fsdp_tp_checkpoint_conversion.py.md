# `.\pytorch\test\distributed\checkpoint\test_fsdp_tp_checkpoint_conversion.py`

```
# Owner(s): ["oncall: distributed"]
# 导入PyTorch库和分布式检查点模块
import torch
import torch.distributed.checkpoint as dist_cp
# 导入分片张量（ShardedTensor）相关模块
from torch.distributed._shard.sharded_tensor import ShardedTensor

# 导入分布式状态字典工具函数和分布式张量类（DTensor）、设备网格初始化函数及复制类（Replicate）
from torch.distributed._state_dict_utils import _all_gather_sharded_tensor
from torch.distributed._tensor import DTensor, init_device_mesh, Replicate
# 导入全分片数据并行（Fully Sharded Data Parallel, FSDP）模块及其状态字典类型
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType

# 导入张量并行模块（ColwiseParallel、RowwiseParallel）及模块并行函数（parallelize_module）
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    parallelize_module,
    RowwiseParallel,
)
# 导入测试相关工具函数和类（run_tests、DTensorTestBase、MLPModule等）
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    MLPModule,
    skip_if_lt_x_gpu,
    with_comms,
)
# 导入临时目录管理工具函数
from torch.testing._internal.distributed.checkpoint_utils import with_temp_dir


# TODO: modularize this test and add test for checkpoint conversion in both direction.
# 定义一个测试类 TestFsdpTpCheckpointConversion，继承自 DTensorTestBase 类
class TestFsdpTpCheckpointConversion(DTensorTestBase):
    # 使用装饰器定义测试方法 with_comms 和 skip_if_lt_x_gpu
    @with_comms
    @skip_if_lt_x_gpu(2)
    @with_temp_dir
    def test_fsdp_to_tp(self):
        CHECKPOINT_DIR = self.temp_dir

        # 创建一个 MLPModule 模型并将其移到 GPU 上指定的设备
        model = MLPModule(self.device_type).cuda(self.rank)
        
        # 使用 FSDP 包装模型
        fsdp_model = FSDP(model, use_orig_params=True)

        # 设置 FSDP 模型的状态字典类型为 SHARDED_STATE_DICT
        FSDP.set_state_dict_type(
            fsdp_model,
            StateDictType.SHARDED_STATE_DICT,
        )
        fsdp_state_dict = fsdp_model.state_dict()

        # 将 fsdp_state_dict 保存到文件系统
        dist_cp.save_state_dict(
            state_dict=fsdp_state_dict,
            storage_writer=dist_cp.FileSystemWriter(CHECKPOINT_DIR),
        )

        # 创建一个 TP 包装模型
        mesh_shape = (self.world_size,)
        device_mesh = init_device_mesh(self.device_type, mesh_shape)
        model = MLPModule(self.device_type).cuda(self.rank)
        
        # 根据给定的并行化计划对模块进行并行化
        parallelize_plan = {
            "net1": ColwiseParallel(),
            "net2": RowwiseParallel(),
        }
        tp_model = parallelize_module(model, device_mesh, parallelize_plan)
        optimizer = torch.optim.SGD(tp_model.parameters(), lr=0.25)

        # 更新参数以确保 tp_model.state_dict() 与 fsdp_model.state_dict() 不同
        torch.manual_seed(0)
        inp = torch.rand(20, 10).cuda(self.rank)
        output = tp_model(inp)
        output.sum().backward()
        optimizer.step()
        tp_state_dict = tp_model.state_dict()

        # 检查在加载之前参数确实不同
        for fsdp_item, tp_item in zip(fsdp_state_dict.items(), tp_state_dict.items()):
            fsdp_k, fsdp_v = fsdp_item
            tp_k, tp_v = tp_item

            self.assertEqual(fsdp_k, tp_k)

            # 如果 fsdp_v 是 ShardedTensor 并且 tp_v 是 DTensor，则分布式数据不相等
            if isinstance(fsdp_v, ShardedTensor) and isinstance(tp_v, DTensor):
                fsdp_redistributed = _all_gather_sharded_tensor(fsdp_v)
                tp_redistributed = tp_v.redistribute(
                    device_mesh, placements=[Replicate()]
                ).to_local()
                self.assertNotEqual(fsdp_redistributed, tp_redistributed)

        # 从文件系统加载 tp_state_dict
        dist_cp.load_state_dict(
            state_dict=tp_state_dict,
            storage_reader=dist_cp.FileSystemReader(CHECKPOINT_DIR),
        )
        tp_model.load_state_dict(tp_state_dict)

        # 检查加载后参数是否相等
        tp_state_dict_after_load = tp_model.state_dict()
        for fsdp_item, tp_item in zip(fsdp_state_dict.items(), tp_state_dict.items()):
            fsdp_k, fsdp_v = fsdp_item
            tp_k, tp_v = tp_item

            self.assertEqual(fsdp_k, tp_k)

            # 如果 fsdp_v 是 ShardedTensor 并且 tp_v 是 DTensor，则分布式数据相等
            if isinstance(fsdp_v, ShardedTensor) and isinstance(tp_v, DTensor):
                fsdp_redistributed = _all_gather_sharded_tensor(fsdp_v)
                tp_redistributed = tp_v.redistribute(
                    device_mesh, placements=[Replicate()]
                ).to_local()
                self.assertEqual(fsdp_redistributed, tp_redistributed)
# 如果当前脚本作为主程序执行（而不是被导入到其他模块中执行），则执行 run_tests() 函数
if __name__ == "__main__":
    run_tests()
```