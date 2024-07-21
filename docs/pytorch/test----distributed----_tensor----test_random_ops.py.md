# `.\pytorch\test\distributed\_tensor\test_random_ops.py`

```
# 导入所需模块和函数
import itertools

import torch
import torch.distributed._functional_collectives as funcol
import torch.distributed._tensor.random as random

from torch.distributed._tensor import DeviceMesh, DTensor
from torch.distributed._tensor._utils import compute_local_shape_and_global_offset
from torch.distributed._tensor.api import distribute_tensor
from torch.distributed._tensor.placement_types import Replicate, Shard
from torch.distributed._tensor.random import is_rng_supported_mesh, manual_seed

from torch.distributed.distributed_c10d import broadcast_object_list

from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    skip_if_lt_x_gpu,
    skip_unless_torch_gpu,
    with_comms,
)

# 定义一个测试类，继承自 DTensorTestBase
class DistTensorRandomInitTest(DTensorTestBase):
    
    # 定义一个私有方法用于运行初始化操作
    def _run_init_op(self, init_op, *args, **kwargs):
        # 构建设备网格对象
        device_mesh = self.build_device_mesh()
        # 指定分片规格，此处为单个分片
        shard_spec = [Shard(0)]
        # 定义输入张量的大小
        input_size = (8, 4)

        # 如果当前设备网格不支持随机数生成，则使用普通设备的随机初始化方式
        if not is_rng_supported_mesh(device_mesh):
            # 在当前设备类型上生成一个随机张量
            input_tensor = torch.randn(*input_size, device=self.device_type)
            # 使用本地张量创建 DTensor 对象
            dtensor = DTensor.from_local(input_tensor, device_mesh, shard_spec)
            # 克隆本地张量
            local_tensor_clone = torch.clone(input_tensor)
            # 设置随机数种子为当前进程的排名
            torch.manual_seed(self.rank)
            # 对克隆的本地张量应用初始化操作
            local_tensor_clone = init_op(local_tensor_clone, *args, **kwargs)
            # 再次设置随机数种子为当前进程的排名
            torch.manual_seed(self.rank)
            # 对 DTensor 应用初始化操作
            dtensor = init_op(dtensor, *args, **kwargs)
            # 断言克隆的本地张量与 DTensor 转换为本地后的结果相等
            self.assertEqual(local_tensor_clone, dtensor.to_local())
        else:
            # 在 CUDA 设备上创建一个空张量
            _tensor = torch.empty(*input_size, device="cuda")
            # 使用设备网格和分片规格分发张量，创建 DTensor 对象
            dtensor = distribute_tensor(_tensor, device_mesh, [Shard(1)])

            # 对 DTensor 进行随机初始化操作
            dtensor = init_op(dtensor, *args, **kwargs)
            # 将 DTensor 转换为本地张量
            local_tensor = dtensor.to_local()

            # 与其他排名的本地张量进行比较
            for other_rank in range(self.world_size):
                if self.rank != other_rank:
                    # 计算切片索引
                    slice_idx = [
                        slice(input_size[0]),
                        slice(
                            other_rank * input_size[1], (other_rank + 1) * input_size[1]
                        ),
                    ]
                    # 断言 DTensor 的切片结果与本地张量不相等
                    self.assertNotEqual(dtensor.full_tensor()[slice_idx], local_tensor)

    # 装饰器，确保在通信环境中运行测试
    @with_comms
    # 定义一个测试函数，用于初始化操作的测试
    def test_init_ops(self):
        # 运行初始化操作，使用 kaiming_uniform_ 方法
        self._run_init_op(
            torch.nn.init.kaiming_uniform_,
            a=0,
            mode="fan_in",
            nonlinearity="leaky_relu",
        )
        # 运行初始化操作，使用 normal_ 方法，指定均值和标准差
        self._run_init_op(torch.nn.init.normal_, mean=1.5, std=0.8)
        # 运行初始化操作，使用 uniform_ 方法，指定上下界
        self._run_init_op(torch.nn.init.uniform_, a=0, b=1.2)

        # 遍历浮点数数据类型列表，针对每一种数据类型执行初始化操作
        for dtype in (torch.float32, torch.float16):
            # 运行初始化操作，使用 rand_like 方法，指定数据类型
            self._run_init_op(torch.rand_like, dtype=dtype)
            # 运行初始化操作，使用 randn_like 方法，指定数据类型
            self._run_init_op(torch.randn_like, dtype=dtype)
            # 运行初始化操作，使用 randint_like 方法，指定数据类型、上下界
            self._run_init_op(torch.randint_like, low=0, high=100, dtype=dtype)
    class DistTensorRandomOpTest(DTensorTestBase):
        # 测试类，用于测试分布式张量的随机操作

        @with_comms
        @skip_unless_torch_gpu
        def test_rng_tracker_init(self):
            # 初始化随机数生成器追踪器的测试方法

            # 设置当前 GPU 的随机种子
            torch.cuda.manual_seed(self.rank)

            # 创建包含当前 GPU 初始种子的对象列表
            object_list = [torch.cuda.initial_seed()]

            # 广播对象列表到所有节点
            broadcast_object_list(object_list)

            # 从对象列表中获取来自rank 0的种子值
            seed_from_rank_0 = int(object_list[0])

            # 创建设备网格对象，用于指定设备类型和节点列表
            device_mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))

            # 首次调用 `distribute_tensor` 后同步种子
            dtensor = distribute_tensor(
                torch.empty([self.world_size], device="cuda"), device_mesh, [Shard(0)]
            )

            # 断言当前节点的种子值与追踪器中的并行随机数生成器种子相同
            self.assertEqual(seed_from_rank_0, random._rng_tracker.get_seed("parallel-rng"))

        @with_comms
        @skip_unless_torch_gpu
        def test_manual_seed(self):
            # 手动设置种子值的测试方法

            # 创建设备网格对象，用于指定设备类型和节点列表
            device_mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))

            # 手动设置种子值为1234
            manual_seed(1234, device_mesh)

            # 断言追踪器中的并行随机数生成器种子为1234
            self.assertEqual(1234, random._rng_tracker.get_seed("parallel-rng"))

            # 使用相同的设备网格再次调用手动设置种子，预期引发RuntimeError
            with self.assertRaisesRegex(RuntimeError, "different seed values"):
                manual_seed(self.rank, device_mesh)

        @with_comms
        @skip_unless_torch_gpu
        def test_deterministic_dropout_1d(self):
            # 测试一维确定性Dropout的方法

            # 设置当前 GPU 的随机种子
            torch.cuda.manual_seed(self.rank)

            # 创建设备网格对象，用于指定设备类型和节点列表
            device_mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))

            # 定义张量的大小
            size = [4, 4]

            # 分布张量到设备网格上，使用Shard(1)策略
            dtensor = distribute_tensor(
                torch.empty(*size, device="cuda"), device_mesh, [Shard(1)]
            )

            # 执行随机操作以改变偏移量
            dtensor.uniform_(0, 1)

            # 将分布的张量重新分发到所有设备网格节点上
            dtensor = dtensor.redistribute(device_mesh, [Replicate()])

            # 创建Dropout层实例
            dropout = torch.nn.Dropout(p=0.2)

            # 在张量上应用Dropout
            dtensor = dropout(dtensor)

            # 使用funcol.all_gather_tensor方法收集所有本地张量
            local_tensor = funcol.all_gather_tensor(
                dtensor.to_local(), gather_dim=0, group=(device_mesh, 0)
            )

            # 将本地张量与其他节点的本地张量比较
            self_slice = slice(4 * self.rank, 4 * self.rank + 4)
            for other_rank in range(self.world_size):
                if self.rank != other_rank:
                    # 其他节点应该有相同的本地张量
                    other_slice = slice(4 * other_rank, 4 * other_rank + 4)
                    self.assertEqual(
                        local_tensor[self_slice, :],
                        local_tensor[other_slice, :],
                    )
    # 定义一个单元测试方法，用于测试确定性随机生成一维张量的行为
    def test_deterministic_rand_1d(self):
        # 创建设备网格对象，使用给定的设备类型和全局大小的范围
        device_mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))
        # 定义张量大小
        size = [4, 4 * self.world_size]

        # 遍历以下随机生成函数列表
        for fn in [
            torch.distributed._tensor.rand,
            torch.distributed._tensor.randn,
        ]:
            # 调用随机生成函数生成分布式张量
            dtensor = fn(size, device_mesh=device_mesh, placements=[Shard(1)])
            # 将生成的分布式张量转换为本地张量并进行全局聚合
            local_tensor = funcol.all_gather_tensor(
                dtensor.to_local(), gather_dim=0, group=(device_mesh, 0)
            )

            # 与其他进程的本地张量进行比较
            self_slice = slice(4 * self.rank, 4 * self.rank + 4)
            for other_rank in range(self.world_size):
                if self.rank != other_rank:
                    # 其他进程应该有一个与本地张量相同的本地张量
                    other_slice = slice(4 * other_rank, 4 * other_rank + 4)
                    self.assertNotEqual(
                        local_tensor[self_slice, :],
                        local_tensor[other_slice, :],
                    )

            # 设置当前 CUDA 设备的随机种子为进程的排名
            torch.cuda.manual_seed(self.rank)
            # 再次调用随机生成函数生成分布式张量
            dtensor = fn(size, device_mesh=device_mesh, placements=[Replicate()])
            # 将生成的分布式张量转换为本地张量并进行全局聚合
            local_tensor = funcol.all_gather_tensor(
                dtensor.to_local(), gather_dim=0, group=(device_mesh, 0)
            )

            # 与其他进程的本地张量进行比较
            self_slice = slice(4 * self.rank, 4 * self.rank + 4)
            for other_rank in range(self.world_size):
                if self.rank != other_rank:
                    # 其他进程应该有一个与本地张量相同的本地张量
                    other_slice = slice(4 * other_rank, 4 * other_rank + 4)
                    self.assertEqual(
                        local_tensor[self_slice, :],
                        local_tensor[other_slice, :],
                    )

    # 使用装饰器指定测试环境，跳过 GPU 数量小于 4 的情况
    @with_comms
    @skip_if_lt_x_gpu(4)
    @with_comms
    @skip_if_lt_x_gpu(4)
    # 定义一个测试函数，用于测试元数据张量的初始化
    def test_meta_tensor_init(self):
        # 在测试套件中，将每个进程的随机种子设置为相同的值，但实际执行中默认的随机种子会有所不同（随机值）。
        # DTensor 的随机操作将使用相同的随机种子，尽管 torch 随机生成器在不同进程中保持不同的种子。
        # 这确保了复制的 DTensor 在各个进程中具有相同的初始化结果。
        torch.cuda.manual_seed(self.rank)
        
        # 创建设备网格对象，根据设备类型和全局大小创建
        device_mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))
        
        # 设置张量的大小
        size = [1024, 2048]
        
        # 分发张量，使用空张量作为元数据，分布到设备网格上，并使用 Replicate 策略
        meta_dtensor = distribute_tensor(
            torch.empty(*size, device="meta"), device_mesh, [Replicate()]
        )
        
        # 断言元数据张量确实是元数据
        self.assertTrue(meta_dtensor.is_meta)
        
        # 创建一个与 meta_dtensor 相同大小的空张量，使用设备类型指定设备
        dtensor = torch.empty_like(meta_dtensor, device=self.device_type)

        # 禁用 RNG 的分发区域
        random._rng_tracker.distribute_region_enabled = False
        
        # 对 dtensor 进行均匀分布的随机初始化
        dtensor.uniform_()

        # 收集所有本地张量
        local_tensor = funcol.all_gather_tensor(
            dtensor.to_local(), gather_dim=0, group=(device_mesh, 0)
        )

        # 与其他进程的本地张量进行比较
        self_slice = slice(1024 * self.rank, 1024 * self.rank + 1024)
        for other_rank in range(self.world_size):
            # 即使预期是复制的，每个进程上的 RNG 结果仍然不同
            if self.rank != other_rank:
                other_slice = slice(1024 * other_rank, 1024 * other_rank + 1024)
                self.assertNotEqual(
                    local_tensor[self_slice, :], local_tensor[other_slice, :]
                )

        # 启用 RNG 的分发区域
        random._rng_tracker.distribute_region_enabled = True
        
        # 再次断言元数据张量确实是元数据
        self.assertTrue(meta_dtensor.is_meta)
        
        # 使用与 meta_dtensor 相同大小的空张量，再次在指定设备上进行随机初始化
        dtensor = torch.empty_like(meta_dtensor, device=self.device_type)
        dtensor.uniform_()

        # 再次收集所有本地张量
        local_tensor = funcol.all_gather_tensor(
            dtensor.to_local(), gather_dim=0, group=(device_mesh, 0)
        )

        # 与其他进程的本地张量进行比较
        for other_rank in range(self.world_size):
            # 由于复制的原因，每个进程上的 RNG 结果应该是相同的
            if self.rank != other_rank:
                # 其他进程应该有相同的本地张量
                other_slice = slice(1024 * other_rank, 1024 * other_rank + 1024)
                self.assertEqual(
                    local_tensor[self_slice, :], local_tensor[other_slice, :]
                )
# 如果这个脚本被直接执行（而不是作为模块被导入），则运行 run_tests() 函数
if __name__ == "__main__":
    run_tests()
```