# `.\pytorch\test\distributed\checkpoint\test_dtensor_checkpoint.py`

```py
# 所有权归["oncall: distributed"]，导入所需的模块和类型注解
from typing import Dict, Union

import torch
import torch.distributed as dist  # 导入分布式操作模块
import torch.distributed.checkpoint as dist_cp  # 导入分布式checkpoint相关模块
from torch.distributed._tensor import (  # 导入分布式张量相关模块
    DeviceMesh,
    distribute_tensor,
    DTensor,
    Replicate,
    Shard,
    zeros,
)
from torch.testing._internal.common_utils import run_tests  # 导入用于测试的通用工具函数
from torch.testing._internal.distributed._tensor.common_dtensor import (  # 导入分布式张量测试相关模块
    DTensorTestBase,
    skip_if_lt_x_gpu,
    with_comms,
)
from torch.testing._internal.distributed.checkpoint_utils import with_temp_dir  # 导入临时目录相关工具函数


SUBMESH_TENSOR_SIZE = 6  # 定义子网格张量的大小为6


class MyTestModule(torch.nn.Module):
    def __init__(
        self,
        sdt: DTensor,
        rdt: DTensor,
        submesh_sdt: DTensor,
        submesh_rdt: DTensor,
        extra_state: int = 1,
        extra_state_tensor: torch.Tensor = torch.zeros(1),
    ) -> None:
        super().__init__()
        # 初始化模块，设置属性为张量的参数
        self.sdt = torch.nn.Parameter(sdt)
        self.rdt = torch.nn.Parameter(rdt)
        self.submesh_sdt = torch.nn.Parameter(submesh_sdt)
        self.submesh_rdt = torch.nn.Parameter(submesh_rdt)
        self._extra_state = extra_state  # 设置额外状态变量
        self._extra_state_tensor = extra_state_tensor  # 设置额外状态张量

    @property
    def extra_state(self) -> int:
        return self._extra_state  # 返回额外状态

    @extra_state.setter
    def extra_state(self, new_extra_state: int) -> None:
        self._extra_state = new_extra_state  # 设置额外状态

    @property
    def extra_state_tensor(self) -> torch.Tensor:
        return self._extra_state_tensor  # 返回额外状态张量

    @extra_state_tensor.setter
    def extra_state_tensor(self, new_extra_state_tensor: torch.Tensor) -> None:
        self._extra_state_tensor = new_extra_state_tensor  # 设置额外状态张量

    def get_extra_state(self) -> Dict[str, Union[int, torch._tensor.Tensor]]:
        return {
            "extra_state": self._extra_state,
            "extra_state_tensor": self._extra_state_tensor,
        }  # 返回包含额外状态和额外状态张量的字典

    def set_extra_state(
        self, state: Dict[str, Union[int, torch._tensor.Tensor]]
    ) -> None:
        self._extra_state = state["extra_state"]  # 设置额外状态，忽略静态类型检查
        self._extra_state_tensor = state["extra_state_tensor"]  # 设置额外状态张量，忽略静态类型检查


class DTensorPlanner(DTensorTestBase):
    def create_dtensor_model(
        self,
        tensor_to_shard: torch.tensor,
        tensor_to_replicate: torch.tensor,
    ) -> torch.nn.Module:
        # 创建一个设备网格对象，包含整个分布式环境中所有设备
        mesh = DeviceMesh(
            device_type=self.device_type,
            mesh=range(dist.get_world_size()),
        )
        # 将要分片的张量分发到设备网格中，使用单个分片
        sharded_dt = distribute_tensor(tensor_to_shard, mesh, placements=[Shard(0)])
        # 将要复制的张量分发到设备网格中，使用复制方式
        replicated_dt = distribute_tensor(
            tensor_to_replicate, mesh, placements=[Replicate()]
        )

        # 只有偶数排位的设备会成为子网格的一部分
        submesh = DeviceMesh(
            device_type=self.device_type,
            mesh=[i for i in range(dist.get_world_size()) if i % 2 == 0],
        )
        # 定义子网格张量的大小
        submesh_tensor_size = [SUBMESH_TENSOR_SIZE]
        # 在子网格上创建一个全零张量，使用单个分片
        submesh_sharded_dt = zeros(
            submesh_tensor_size,
            device_mesh=submesh,
            placements=[Shard(0)],
        )
        # 在子网格上创建一个全零张量，使用复制方式
        submesh_replicated_dt = zeros(
            submesh_tensor_size, device_mesh=submesh, placements=[Replicate()]
        )

        # 实例化自定义测试模块，将分片和复制的张量作为参数，然后放到 CUDA 设备上
        model = MyTestModule(
            sharded_dt,
            replicated_dt,
            submesh_sharded_dt,
            submesh_replicated_dt,
        ).cuda()

        # 返回模型及其所需的分片和复制的张量
        return (
            model,
            sharded_dt,
            replicated_dt,
        )

    @with_comms
    @with_temp_dir
    @skip_if_lt_x_gpu(2)
# 如果当前脚本作为主程序运行（而不是作为模块被导入），则执行 run_tests 函数
if __name__ == "__main__":
    run_tests()
```