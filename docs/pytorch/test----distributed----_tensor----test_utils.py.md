# `.\pytorch\test\distributed\_tensor\test_utils.py`

```
# Owner(s): ["oncall: distributed"]

import itertools  # 导入 itertools 库，用于生成迭代器的工具函数

import torch  # 导入 PyTorch 库
from torch.distributed._tensor import distribute_tensor, DTensor  # 导入分布式张量相关模块和类
from torch.distributed._tensor._utils import (
    compute_local_shape,  # 导入计算本地形状的函数
    compute_local_shape_and_global_offset,  # 导入计算本地形状和全局偏移的函数
)

from torch.distributed._tensor.debug import CommDebugMode  # 导入通信调试模式相关模块
from torch.distributed._tensor.placement_types import (  # 导入张量放置类型相关模块
    DTensorSpec,
    Replicate,
    Shard,
    TensorMeta,
)
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh  # 导入设备网格相关模块

from torch.testing._internal.common_utils import run_tests  # 导入测试相关工具函数
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,  # 导入分布式张量测试基类
    with_comms,  # 导入带通信的装饰器函数
)

c10d_functional = torch.ops.c10d_functional  # 定义 c10d_functional 作为 C10D 功能的操作接口


class UtilTest(DTensorTestBase):  # 定义 UtilTest 类，继承自 DTensorTestBase

    @property
    def world_size(self):  # 定义属性 world_size，返回集群的大小，这里设定为 8
        return 8

    @with_comms  # 使用带通信的装饰器，用于测试方法中的通信

    def test_compute_local_shape_2d_uneven(self):  # 定义测试方法 test_compute_local_shape_2d_uneven

        # mesh: 4 * 2
        mesh_tensor = torch.arange(self.world_size).reshape(4, 2)  # 创建一个 4x2 的网格张量
        mesh = DeviceMesh(self.device_type, mesh_tensor)  # 使用设备类型和网格张量创建 DeviceMesh 对象
        size = torch.Size([7, 7])  # 定义一个大小为 7x7 的张量形状
        rank_coordinates = mesh.get_coordinate()  # 获取当前进程在网格中的坐标

        # replicate, shard
        placements2 = [Replicate(), Shard(0)]  # 创建放置类型列表
        local_size2 = compute_local_shape(size, mesh, placements2)  # 计算使用给定放置类型的本地形状
        if rank_coordinates[1] < 1:
            self.assertEqual(local_size2, torch.Size([4, 7]))  # 如果在第二维度小于 1，则断言本地形状为 [4, 7]
        else:
            self.assertEqual(local_size2, torch.Size([3, 7]))  # 否则断言本地形状为 [3, 7]

        # shard, shard
        placements3 = [Shard(0), Shard(1)]  # 创建放置类型列表
        local_size3 = compute_local_shape(size, mesh, placements3)  # 计算使用给定放置类型的本地形状
        # first dim
        if rank_coordinates[0] < 3:
            self.assertEqual(local_size3[0], 2)  # 如果在第一维度小于 3，则断言第一维度的本地形状为 2
        else:
            self.assertEqual(local_size3[0], 1)  # 否则断言第一维度的本地形状为 1
        # second dim
        if rank_coordinates[1] < 1:
            self.assertEqual(local_size3[1], 4)  # 如果在第二维度小于 1，则断言第二维度的本地形状为 4
        else:
            self.assertEqual(local_size3[1], 3)  # 否则断言第二维度的本地形状为 3
    # 定义一个测试函数，用于测试计算 1D 数据的本地形状和全局偏移
    def test_compute_local_shape_and_global_offset_1D(self):
        # 定义一维数据的放置方式，包括每个测试场景下的放置策略
        one_d_placements = [[Shard(0)], [Replicate()]]

        # 遍历不同的放置策略
        for placements in one_d_placements:
            # 当放置策略是 [Shard(0)] 时，测试三种不同的情况：
            # 1) 分片导致所有或某些等级的空分片
            # 2) 分片导致不同等级间分片大小不同
            # 3) 分片导致所有等级上非空分片大小相同
            for size in range(self.world_size * 2 + 1):
                # 创建一个包含全球设备标识的张量
                mesh_tensor = torch.arange(self.world_size)
                # 创建设备网格对象
                device_mesh = DeviceMesh(self.device_type, mesh_tensor)
                # 创建全局张量
                global_tensor = torch.arange(size)
                # 获取全局张量的形状
                global_shape = global_tensor.size()

                # 分发全局张量到设备上，根据给定的放置策略
                dtensor = distribute_tensor(global_tensor, device_mesh, placements)
                # 计算本地形状和全局偏移量
                local_size, global_offset = compute_local_shape_and_global_offset(
                    global_shape, device_mesh, placements
                )

                # 获取第一维度的起始和结束位置
                dim0_start = global_offset[0]
                dim0_end = global_offset[0] + local_size[0]

                # 检查 dtensor 的本地张量是否与通过全局大小和全局偏移切片得到的 global_tensor 相同
                self.assertEqual(
                    dtensor.to_local(),
                    global_tensor[dim0_start:dim0_end],
                )
    def test_compute_local_shape_and_global_offset_2D(self):
        two_d_placements_options = [Shard(0), Shard(1), Replicate()]
        # 生成6种二维放置组合
        two_d_placements = list(
            itertools.combinations_with_replacement(two_d_placements_options, 2)
        )

        # 遍历所有二维放置组合
        for placements in two_d_placements:
            # 对于不同的 dim_0_size 进行测试
            for dim_0_size in (1, 2, 4, 8):
                # 创建一个 2 * 4 的网格张量
                mesh_tensor = torch.arange(self.world_size).reshape(2, 4)
                device_mesh = DeviceMesh(self.device_type, mesh_tensor)
                # 创建一个全局张量，形状为 dim_0_size * (总元素数除以 dim_0_size)
                global_tensor = torch.arange(64).view(dim_0_size, -1)
                global_shape = global_tensor.size()

                # 将全局张量在设备网格和放置方式下进行分发
                dtensor = distribute_tensor(global_tensor, device_mesh, placements)
                # 计算局部形状和全局偏移量
                local_size, global_offset = compute_local_shape_and_global_offset(
                    global_shape, device_mesh, placements
                )

                # TODO: make this test cleaner and work for nD
                # 获取局部张量的切片起始和结束位置
                dim0_start = global_offset[0]
                dim0_end = global_offset[0] + local_size[0]
                dim1_start = global_offset[1]
                dim1_end = global_offset[1] + local_size[1]

                # 检查 dtensor 的局部张量是否与使用 local_size 和 global_offset 切片的 global_tensor 完全相同
                self.assertEqual(
                    dtensor.to_local(),
                    global_tensor[dim0_start:dim0_end, dim1_start:dim1_end],
                )
class Test2DStridedLocalShard(DTensorTestBase):
    @property
    def world_size(self):
        # 返回当前测试环境的全局进程数，这里是模拟返回数字 4
        return 4

    @with_comms
    def test_fsdp1_tp_2d_dtensor_local_shards_and_offsets(self):
        # 我们在模拟 FSDP1 + TP 的行为。
        # 当前，2D DTensor 的本地分片是正确的，因为从 from_local + redistribute 后面会发生 all_gather。
        # 当我们有一个全局张量 [0, 1, 2, 3, 4, 5, 6, 7] 时，2D DTensor 的本地分片如下：
        # rank0: [0, 1], rank1: [2, 3], rank2: [4, 5], rank3: [6, 7]
        with CommDebugMode() as comm_mode:
            # 创建一个全局张量 [0, 1, 2, 3, 4, 5, 6, 7]
            global_tensor = torch.arange(8).view(4, 2)
            # 初始化一个2D设备网格，用于存储设备信息，指定网格维度名称为 ("DP", "TP")
            mesh_2d = init_device_mesh(
                self.device_type, (2, 2), mesh_dim_names=("DP", "TP")
            )
            # 获取 TP 维度的网格信息
            tp_mesh = mesh_2d["TP"]
            # 在 TP 维度上分发张量
            dtensor_tp = distribute_tensor(
                global_tensor, tp_mesh, placements=[Shard(0)]
            )
            # 使用 from_local 方法创建 2D DTensor，并进行重分布
            dtensor_2d = DTensor.from_local(
                dtensor_tp.to_local(), mesh_2d, [Replicate(), Shard(0)], run_check=False
            ).redistribute(mesh_2d, [Shard(0), Shard(0)])
            # 断言调试模式下 all_gather 操作的次数为 1
            self.assertEqual(
                comm_mode.get_comm_counts()[c10d_functional.all_gather_into_tensor], 1
            )

        # 断言 dtensor_2d 的本地值等于全局张量中当前进程的分片
        self.assertEqual(
            dtensor_2d.to_local(), global_tensor[self.rank : self.rank + 1]
        )
        # 计算本地形状和全局偏移量，当前考虑到分片是有步长的
        local_size, global_offset = compute_local_shape_and_global_offset(
            global_tensor.shape, mesh_2d, [Shard(0), Shard(0)]
        )
        # 断言本地形状与预期的 torch.Size([1, 2]) 相符
        self.assertEqual(local_size, torch.Size([1, 2]))
        # 断言全局偏移量与预期的 torch.Size([self.rank, 0]) 相符
        self.assertEqual(global_offset, torch.Size([self.rank, 0]))

    @with_comms
    # 定义一个测试方法，用于测试FSDP2 + TP的行为
    def test_fsdp2_tp_2d_dtensor_local_shards_and_offsets(self):
        # 模拟FSDP2 + TP的行为
        # 当前，2D DTensor的本地分片对于重分片是不正确的，因为我们希望避免额外的通信。
        # 它对于重分片来说是不正确的，因为`compute_local_shape_and_global_offset`
        # 不知道重分片的正确偏移量。
        # 当我们有一个全局张量[0, 1, 2, 3, 4, 5, 6, 7]时，2D DTensor的本地分片将是：
        # 本地张量 -- rank0: [0, 1], rank1: [4, 5], rank2: [2, 3], rank3: [6, 7]
        # 当前的偏移量 -- rank0: [0, 0], rank1: [1, 0], rank2: [2, 0], rank3: [3, 0]
        # 理想情况下，使用分段分片，偏移应为 rank0: [0, 0], rank1: [2, 0], rank2: [1, 0], rank3: [3, 0]
        # TODO: 要使FSDP2 + TP的本地分片在重分片时正确，需要使用分段分片
        # 并且让compute_local_shape_and_global_offset考虑分段分片。
        
        # 进入CommDebugMode上下文
        with CommDebugMode() as comm_mode:
            # 创建一个全局张量，包含值为0到7的8个元素，reshape为4行2列
            global_tensor = torch.arange(8).view(4, 2)
            # 初始化设备网格，使用2x2的网格，指定网格维度名称为("DP", "TP")
            mesh_2d = init_device_mesh(
                self.device_type, (2, 2), mesh_dim_names=("DP", "TP")
            )
            # 获取TP维度的网格
            tp_mesh = mesh_2d["TP"]
            # 分发张量到本地，使用tp_mesh，并指定放置策略为Shard(0)
            dtensor_tp = distribute_tensor(
                global_tensor, tp_mesh, placements=[Shard(0)]
            )
            # 将dtensor_tp沿第0维度分成两个块
            chunks = list(torch.chunk(dtensor_tp.to_local(), 2, dim=0))
            # 确定当前分片的排名
            shard_rank = 0 if self.rank // 2 == 0 else 1
            # 获取对应分片的参数
            sharded_param = chunks[shard_rank]
            # 定义2D DTensor的规范，包括网格、放置策略和张量元数据
            spec_2d = DTensorSpec(
                mesh=mesh_2d,
                placements=(Shard(0), Shard(0)),
                tensor_meta=TensorMeta(
                    global_tensor.size(),
                    global_tensor.stride(),
                    global_tensor.dtype,
                ),
            )
            # 创建2D DTensor对象，使用分片参数、规范和不需要梯度
            dtensor_2d = DTensor(
                sharded_param,
                spec_2d,
                requires_grad=False,
            )
            # 断言通信模式中的通信次数为0
            self.assertEqual(
                comm_mode.get_comm_counts()[c10d_functional.all_gather_into_tensor], 0
            )
# 如果当前脚本被作为主程序执行，则运行测试函数
if __name__ == "__main__":
    run_tests()
```