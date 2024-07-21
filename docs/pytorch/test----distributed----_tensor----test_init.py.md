# `.\pytorch\test\distributed\_tensor\test_init.py`

```py
# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]

# 导入 PyTorch 库
import torch
# 导入分布式张量相关模块
from torch.distributed._tensor import DeviceMesh, DTensor, Replicate, Shard, zeros
# 导入测试工具函数
from torch.testing._internal.common_utils import run_tests
# 导入分布式张量测试相关模块
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)


class DTensorInitOpsTest(DTensorTestBase):
    # 定义运行初始化操作的方法
    def _run_init_op(self, init_op, *args, **kwargs):
        # 构建设备网格
        device_mesh = self.build_device_mesh()
        # 定义分片规格，这里仅使用第一个分片
        shard_spec = [Shard(0)]
        # 定义输入张量的大小
        input_size = (8, 4)
        # 在指定设备上创建随机张量
        input_tensor = torch.randn(*input_size, device=self.device_type)
        # 使用本地张量创建分布式张量对象
        dtensor = DTensor.from_local(input_tensor, device_mesh, shard_spec)
        # 克隆本地张量用于后续的对比
        local_tensor_clone = torch.clone(input_tensor)
        # 设置随机种子
        torch.manual_seed(self.rank)
        # 在克隆的本地张量上执行初始化操作
        local_tensor_clone = init_op(local_tensor_clone, *args, **kwargs)
        # 再次设置随机种子
        torch.manual_seed(self.rank)
        # 在分布式张量上执行初始化操作
        dtensor = init_op(dtensor, *args, **kwargs)
        # 断言本地克隆张量与分布式张量的本地表示是否一致
        self.assertEqual(local_tensor_clone, dtensor.to_local())

    # 使用通信环境装饰器定义测试初始化操作的方法
    @with_comms
    def test_init_ops(self):
        # 注意：随机初始化测试已迁移到 test_random_ops.py
        # 运行初始化操作方法，使用常量初始化
        self._run_init_op(torch.nn.init.constant_, 2.4)


class DTensorConstructorTest(DTensorTestBase):
    # 定义世界大小属性
    @property
    def world_size(self):
        return 4
    # 运行初始化操作的私有方法，用于测试不同的分布式初始化情况
    def _run_init_op(self, init_op, dist_init_op, eq_op, *args, **kwargs):
        # 1d mesh test
        # 创建一个包含设备网格的 DeviceMesh 对象，设备类型为 self.device_type，包含从 0 到 self.world_size-1 的列表
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        
        # 定义放置列表，包括 Shard(0), Shard(1), Shard(2), 和 Replicate() 对象
        placements_list = [[Shard(0)], [Shard(1)], [Shard(2)], [Replicate()]]
        
        # even sharding
        # 定义一个 tensor_size 列表 [4, 8, 12]
        tensor_size = [4, 8, 12]
        for placements in placements_list:
            # 复制 tensor_size 到 local_tensor_size
            local_tensor_size = tensor_size.copy()
            # 如果 placements[0] 是 Shard 类型，则计算 sharding 维度并更新 local_tensor_size
            if isinstance(placements[0], Shard):
                shard_dim = placements[0].dim
                local_tensor_size[shard_dim] //= self.world_size
            
            # 使用 dist_init_op 初始化分布式 tensor，传入 tensor_size 和其他参数及设备网格和放置列表
            dist_tensor = dist_init_op(
                tensor_size,
                *args,
                **kwargs,
                device_mesh=device_mesh,
                placements=placements,
            )
            # 使用 init_op 初始化本地 tensor，传入 local_tensor_size 和其他参数
            ones_expected = init_op(local_tensor_size, *args, **kwargs)
            # 断言两个 tensor 相等，调用 eq_op
            eq_op(ones_expected, dist_tensor.to_local())

        # uneven sharding
        # 定义一个新的 tensor_size 列表 [5, 10, 15]
        tensor_size = [5, 10, 15]
        for placements in placements_list:
            # 使用 dist_init_op 初始化分布式 tensor，传入 tensor_size 和其他参数及设备网格和放置列表
            dist_tensor = dist_init_op(
                tensor_size,
                *args,
                **kwargs,
                device_mesh=device_mesh,
                placements=placements,
            )
            # 如果 placements[0] 是 Shard 类型
            if isinstance(placements[0], Shard):
                shard_dim = placements[0].dim
                # 将 init_op 初始化的 tensor 按照 shard_dim 维度分块，分成 self.world_size 份
                exp_tensor_list = list(
                    torch.chunk(
                        init_op(tensor_size, *args, **kwargs),
                        self.world_size,
                        dim=shard_dim,
                    )
                )
                # 如果当前进程 rank 小于 exp_tensor_list 的长度，断言 exp_tensor_list[self.rank] 和 dist_tensor.to_local() 相等
                if self.rank < len(exp_tensor_list):
                    eq_op(exp_tensor_list[self.rank], dist_tensor.to_local())
            else:
                # 如果 placements[0] 不是 Shard 类型，则直接使用 init_op 初始化 tensor
                exp_tensor = init_op(tensor_size, *args, **kwargs)
                # 断言 exp_tensor 和 dist_tensor.to_local() 相等
                eq_op(exp_tensor, dist_tensor.to_local())

        # empty shape
        # 使用 dist_init_op 初始化一个空形状的 tensor，传入 [] 和其他参数及设备网格和放置列表为 Replicate()
        local_tensor = dist_init_op(
            [], *args, **kwargs, device_mesh=device_mesh, placements=[Replicate()]
        ).to_local()
        # 使用 init_op 初始化一个空形状的 tensor，传入 [] 和其他参数
        expected_tensor = init_op([], *args, **kwargs)
        # 断言 expected_tensor 和 local_tensor 相等
        eq_op(expected_tensor, local_tensor)

    @with_comms
    # 使用 with_comms 装饰器定义测试函数 test_ones
    def test_ones(self):
        # 调用 _run_init_op 方法，初始化全为 1 的 tensor，使用 torch.distributed._tensor.ones 进行分布式初始化
        self._run_init_op(
            torch.ones,
            torch.distributed._tensor.ones,
            self.assertEqual,
            requires_grad=True,
        )

    @with_comms
    # 使用 with_comms 装饰器定义测试函数 test_empty
    def test_empty(self):
        # 调用 _run_init_op 方法，初始化空 tensor，使用 torch.distributed._tensor.empty 进行分布式初始化
        self._run_init_op(
            torch.empty,
            torch.distributed._tensor.empty,
            lambda x, y: (x.shape == y.shape)
            and (x.dtype == y.dtype)
            and (x.layout == y.layout),
            requires_grad=True,
        )

    @with_comms
    # 使用 with_comms 装饰器定义测试函数 test_full
    def test_full(self):
        # 调用 _run_init_op 方法，初始化全为给定值的 tensor，使用 torch.distributed._tensor.full 进行分布式初始化
        self._run_init_op(
            torch.full,
            torch.distributed._tensor.full,
            self.assertEqual,
            123.4,
            requires_grad=True,
        )

    @with_comms
    # 定义名为 test_zeros 的测试方法，属于类 self 对象
    def test_zeros(self):
        # 调用 _run_init_op 方法，传入 torch.zeros、torch.distributed._tensor.zeros 和 self.assertEqual 作为参数
        # requires_grad=True 作为额外参数传递给 _run_init_op 方法
        self._run_init_op(
            torch.zeros,
            torch.distributed._tensor.zeros,
            self.assertEqual,
            requires_grad=True,
        )
    
    # 应用装饰器 @with_comms，装饰在 test_zeros 方法上
    @with_comms
    def test_zeros_full_mesh(self):
        # 构建一个 CUDA 设备上的 1 维网格
        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))
        placements = [Shard(0)]
        size = [32, 3]
        # 创建一个分布式张量，所有元素设为零，设备网格为 mesh，放置方案为 placements
        dist_tensor = zeros(size, device_mesh=mesh, placements=placements)
        self.assertEqual(dist_tensor.size(), torch.Size(size))
        # 转换为本地张量
        local_tensor = dist_tensor.to_local()
        self.assertEqual(local_tensor.size(), torch.Size([8, 3]))

        # 创建一个本地的全零张量
        local_tensor = torch.zeros(8, 3)
        self.assertEqual(dist_tensor.to_local(), local_tensor)

        self.assertEqual(dist_tensor.device.type, self.device_type)

        # 1 维不均匀分片
        size = [31, 3]
        dist_tensor = zeros(size, device_mesh=mesh, placements=placements)
        self.assertEqual(dist_tensor.size(), torch.Size(size))
        local_tensor = dist_tensor.to_local()
        if self.rank <= 2:
            self.assertEqual(local_tensor.size(), torch.Size([8, 3]))
            self.assertEqual(torch.zeros(8, 3), local_tensor)
        else:
            self.assertEqual(local_tensor.size(), torch.Size([7, 3]))
            self.assertEqual(torch.zeros(7, 3), local_tensor)

        # 构建一个具有 2 维的 CUDA 设备网格：分片，复制
        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size).reshape(2, 2))
        placements = [Shard(0), Replicate()]
        size = [32, 4]
        dist_tensor = zeros(size, device_mesh=mesh, placements=placements)

        self.assertEqual(dist_tensor.size(), torch.Size(size))
        local_tensor = dist_tensor.to_local()
        self.assertEqual(local_tensor.size(), torch.Size([16, 4]))
        self.assertEqual(local_tensor, torch.zeros([16, 4]))

        # 构建一个具有 2 维的 CUDA 设备网格：分片，分片
        placements = [Shard(0), Shard(1)]
        size = [32, 4]
        dist_tensor = zeros(size, device_mesh=mesh, placements=placements)

        self.assertEqual(dist_tensor.size(), torch.Size(size))
        local_tensor = dist_tensor.to_local()
        self.assertEqual(local_tensor.size(), torch.Size([16, 2]))
        self.assertEqual(local_tensor, torch.zeros([16, 2]))

        # 2 维不均匀分片
        placements = [Shard(0), Shard(1)]
        size = [31, 3]
        dist_tensor = zeros(size, device_mesh=mesh, placements=placements)

        self.assertEqual(dist_tensor.size(), torch.Size(size))
        local_tensor = dist_tensor.to_local()
        if self.rank == 0:
            self.assertEqual(local_tensor, torch.zeros([16, 2]))
        elif self.rank == 1:
            self.assertEqual(local_tensor, torch.zeros([16, 1]))
        elif self.rank == 2:
            self.assertEqual(local_tensor, torch.zeros([15, 2]))
        elif self.rank == 3:
            self.assertEqual(local_tensor, torch.zeros([15, 1]))
    # 测试在没有子网格初始化的情况下创建全零分布张量
    def test_zeros_submesh(self):
        # 默认的 world_size 是 4
        # 构建一个 CUDA 设备的一维网格，没有初始化子网格
        sub_mesh_list = [0, 3]
        mesh = DeviceMesh(self.device_type, sub_mesh_list)
        placements = [Shard(0)]
        size = [32, 3]
        dist_tensor = zeros(size, device_mesh=mesh, placements=placements)
        self.assertEqual(dist_tensor.size(), torch.Size(size))
        local_tensor = dist_tensor.to_local()

        if self.rank in sub_mesh_list:
            self.assertEqual(local_tensor.size(), torch.Size([16, 3]))
            self.assertEqual(local_tensor, torch.zeros([16, 3]))
        else:
            self.assertEqual(local_tensor.size(), torch.Size([0]))
            self.assertEqual(local_tensor, torch.zeros(0))

        # 构建一个 CUDA 设备的一维网格：不均匀，带有初始化的子网格
        sub_mesh_list = [0, 1, 3]
        mesh = DeviceMesh(self.device_type, sub_mesh_list)
        placements = [Shard(0)]
        size = [32, 3]
        dist_tensor = zeros(size, device_mesh=mesh, placements=placements)
        self.assertEqual(dist_tensor.size(), torch.Size(size))
        local_tensor = dist_tensor.to_local()

        if self.rank in sub_mesh_list:
            if self.rank != 3:
                self.assertEqual(local_tensor.size(), torch.Size([11, 3]))
                self.assertEqual(local_tensor, torch.zeros([11, 3]))
            else:
                self.assertEqual(local_tensor.size(), torch.Size([10, 3]))
                self.assertEqual(local_tensor, torch.zeros([10, 3]))
        else:
            self.assertEqual(local_tensor.size(), torch.Size([0]))
            self.assertEqual(local_tensor, torch.tensor([]))

        # 构建一个 CUDA 设备的二维网格，没有初始化子网格
        sub_mesh_list = [[0], [3]]
        mesh = DeviceMesh(self.device_type, sub_mesh_list)
        placements = [Shard(0), Shard(1)]
        size = [32, 3]
        dist_tensor = zeros(size, device_mesh=mesh, placements=placements)
        self.assertEqual(dist_tensor.size(), torch.Size(size))
        local_tensor = dist_tensor.to_local()

        if self.rank in [0, 3]:
            self.assertEqual(local_tensor.size(), torch.Size([16, 3]))
            self.assertEqual(local_tensor, torch.zeros([16, 3]))
        else:
            self.assertEqual(local_tensor.size(), torch.Size([0]))
            self.assertEqual(local_tensor, torch.tensor([]))
# 如果当前脚本作为主程序运行，则执行 run_tests() 函数
if __name__ == "__main__":
    run_tests()
```