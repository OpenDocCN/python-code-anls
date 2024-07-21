# `.\pytorch\test\distributed\checkpoint\test_fsdp_model_state.py`

```py
# Owner(s): ["oncall: distributed"]

import torch  # 导入 PyTorch 库
import torch.distributed as dist  # 导入分布式通信模块
import torch.distributed.checkpoint as dist_cp  # 导入分布式检查点模块

from torch.distributed.checkpoint.default_planner import (  # 导入默认的检查点计划器
    DefaultLoadPlanner,
    DefaultSavePlanner,
)

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP  # 导入全分片数据并行模块
from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType  # 导入状态字典类型
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu  # 导入 GPU 数量检查装饰器
from torch.testing._internal.common_utils import run_tests  # 导入测试运行函数

from torch.testing._internal.distributed._tensor.common_dtensor import (  # 导入分布式张量测试基类和通信装饰器
    DTensorTestBase,
    with_comms,
)
from torch.testing._internal.distributed.checkpoint_utils import with_temp_dir  # 导入临时目录装饰器


class FsdpModelStateCheckpoint(DTensorTestBase):  # 定义继承自 DTensorTestBase 的类 FsdpModelStateCheckpoint
    @property
    def backend(self):
        return "cpu:gloo,cuda:nccl"  # 返回分布式后端设置为 "cpu:gloo,cuda:nccl"

    def _test_fsdp_model_state(self, process_group) -> None:
        CHECKPOINT_DIR = self.temp_dir  # 获取临时目录路径

        model = FSDP(torch.nn.Linear(8, 8, device="meta"))  # 创建一个 FSDP 模型
        model(torch.rand(8, 8, device=dist.get_rank())).sum().backward()  # 在模型上执行前向和反向传播

        with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):  # 使用分片状态字典类型包装模型
            state_dict = {
                "model": model.state_dict(),  # 获取模型的状态字典
            }

            dist_cp.save_state_dict(
                state_dict=state_dict,  # 保存模型状态字典
                storage_writer=dist_cp.FileSystemWriter(CHECKPOINT_DIR),  # 使用文件系统写入器保存到指定目录
                planner=DefaultSavePlanner(),  # 使用默认保存计划器
            )

        model_2 = FSDP(
            torch.nn.Linear(8, 8, device="meta"), process_group=process_group  # 创建另一个 FSDP 模型
        )

        with FSDP.summon_full_params(model):  # 恢复模型的完整参数
            with FSDP.summon_full_params(model_2):  # 恢复模型2的完整参数
                self.assertNotEqual(model.weight, model_2.weight)  # 断言模型权重不相等
                self.assertNotEqual(model.bias, model_2.bias)  # 断言模型偏置不相等

        # 现在加载模型并确保值相同
        with FSDP.state_dict_type(model_2, StateDictType.SHARDED_STATE_DICT):  # 使用分片状态字典类型包装模型2
            state_dict = {
                "model": model_2.state_dict(),  # 获取模型2的状态字典
            }

            dist_cp.load_state_dict(
                state_dict=state_dict,  # 加载模型2的状态字典
                storage_reader=dist_cp.FileSystemReader(CHECKPOINT_DIR),  # 使用文件系统读取器从指定目录读取
                planner=DefaultLoadPlanner(),  # 使用默认加载计划器
            )
            model_2.load_state_dict(state_dict["model"])  # 加载模型2的状态字典

        with FSDP.summon_full_params(model):  # 恢复模型的完整参数
            with FSDP.summon_full_params(model_2):  # 恢复模型2的完整参数
                self.assertEqual(model.weight, model_2.weight)  # 断言模型权重相等
                self.assertEqual(model.bias, model_2.bias)  # 断言模型偏置相等

    @with_comms  # 使用通信装饰器
    @skip_if_lt_x_gpu(2)  # 如果 GPU 数量小于2，则跳过测试
    @with_temp_dir  # 使用临时目录装饰器
    def test_fsdp_model_state_no_resharding(self):
        self._test_fsdp_model_state(process_group=None)  # 执行模型状态测试，传入 None 作为进程组参数
    # 创建一个新的分布式组
    def _create_new_dist_group(self):
        # 获取当前分布式环境的总大小
        world_size = dist.get_world_size()
        # 从所有进程中筛选出索引为偶数的进程列表，作为第一个组
        group1 = [i for i in range(world_size) if i % 2 == 0]
        # 从所有进程中筛选出索引为奇数的进程列表，作为第二个组
        group2 = [i for i in range(world_size) if i % 2 != 0]

        # 创建用于重新分片的新的 fsdp 组
        fsdp_0 = dist.new_group(ranks=group1)
        fsdp_1 = dist.new_group(ranks=group2)
        
        # 根据当前进程的索引选择对应的 fsdp 组
        if dist.get_rank() % 2 == 0:
            my_fsdp = fsdp_0
        else:
            my_fsdp = fsdp_1

        # 返回当前进程所属的 fsdp 组
        return my_fsdp

    # 使用装饰器定义测试函数，测试 fsdp 模型状态的重分片
    @with_comms
    @skip_if_lt_x_gpu(4)
    @with_temp_dir
    def test_fsdp_model_state_with_resharding(self):
        # 调用测试函数，并传入用于处理组的新创建的分布式组
        self._test_fsdp_model_state(process_group=self._create_new_dist_group())
# 如果当前脚本被直接执行（而非被导入为模块），则执行 run_tests() 函数
if __name__ == "__main__":
    run_tests()
```