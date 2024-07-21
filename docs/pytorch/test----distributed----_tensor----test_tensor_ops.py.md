# `.\pytorch\test\distributed\_tensor\test_tensor_ops.py`

```
# 导入 PyTorch 库及相关模块和函数
import torch
from torch.distributed._tensor import DeviceMesh, distribute_tensor, DTensor
from torch.distributed._tensor.debug import CommDebugMode
from torch.distributed._tensor.placement_types import Partial, Replicate, Shard
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorConverter,
    DTensorTestBase,
    with_comms,
)

# 创建一个测试类 DistTensorOpsTest，继承自 DTensorTestBase 类
class DistTensorOpsTest(DTensorTestBase):

    # 装饰器函数，用于在测试方法中进行通信设置
    @with_comms
    def test_aten_contiguous(self):
        # 创建设备网格对象，指定设备类型和世界大小范围
        mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        # 调用测试操作函数 _test_op，测试 torch.ops.aten.contiguous 方法
        self._test_op(
            mesh,
            lambda x: torch.ops.aten.contiguous(x),
            torch.randn(16, 32),
        )

    @with_comms
    def test_detach(self):
        # 创建设备网格对象，指定设备类型和世界大小范围
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        # 定义分片规格，这里只包含 Shard(0)
        shard_spec = [Shard(0)]
        # 创建一个带梯度的随机张量
        tensor_to_detach = torch.randn(12, 8, requires_grad=True)
        # 将张量在设备网格上分发
        mat = distribute_tensor(tensor_to_detach, device_mesh, shard_spec)
        # 对分发后的张量执行 detach 操作
        detached_mat = mat.detach()
        # 断言 detached_mat 不是原始 mat 张量对象
        self.assertFalse(detached_mat is mat)

    @with_comms
    def test_clone(self):
        # 创建设备网格对象，指定设备类型和世界大小范围
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        # 定义多种分片规格，包括 Replicate() 和 Shard(0)
        specs = [[Replicate()], [Shard(0)]]
        # 创建一个带梯度的随机张量
        tensor_to_clone = torch.randn(12, 8, requires_grad=True)
        # 对每种分片规格进行测试
        for spec in specs:
            # 将张量在设备网格上分发
            mat = distribute_tensor(tensor_to_clone, device_mesh, spec)
            # 对分发后的张量执行 clone 操作
            cloned_mat = mat.clone()
            # 断言 cloned_mat 不是原始 mat 张量对象
            self.assertFalse(cloned_mat is mat)
            # 断言 cloned_mat 和 mat 的本地表示相等
            self.assertEqual(cloned_mat.to_local(), mat.to_local())

    @with_comms
    def test_contiguous(self):
        # 创建设备网格对象，指定设备类型和世界大小范围
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        # 创建一个带梯度的随机张量
        tensor = torch.rand(3, 5, 6, requires_grad=True)
        # 定义分片规格，这里只包含 Shard(0)
        sharding = [Shard(0)]
        # 将本地张量转换为分布式张量对象
        dist_tensor = DTensor.from_local(tensor, device_mesh, sharding)
        # 断言 dist_tensor 是连续的
        self.assertTrue(dist_tensor.is_contiguous())
        # 断言分布式张量的步长与原始张量的步长相同
        self.assertEqual(dist_tensor.stride(), tensor.stride())

        # 对分布式张量执行维度转置操作
        new_dt = dist_tensor.transpose(0, 2)
        # 断言 new_dt 不是连续的
        self.assertFalse(new_dt.is_contiguous())
        # 断言 new_dt 的本地表示不是连续的
        self.assertFalse(new_dt.to_local().is_contiguous())
        # 检查步长
        self.assertEqual(new_dt.stride(), (1, 6, 30))

        # 对 new_dt 执行连续性操作
        new_dt = new_dt.contiguous()
        # 断言 new_dt 是连续的
        self.assertTrue(new_dt.is_contiguous())
        # 断言 new_dt 的本地表示是连续的
        self.assertTrue(new_dt.to_local().is_contiguous())
        # 检查步长
        self.assertEqual(dist_tensor.stride(), tensor.stride())

        # 检查反向传播
        new_dt.to_local().sum().backward()
        # 断言张量的梯度与预期值相等
        self.assertEqual(tensor.grad, torch.ones(3, 5, 6))

    @with_comms
    # 定义一个测试方法，用于测试就地操作（inplace operation）
    def test_inplace_op(self):
        # 创建一个设备网格对象，并指定设备类型和全球大小
        mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        # 创建一个随机张量，设备类型与测试中指定的设备类型一致
        input_tensor = torch.randn((12, 3), device=self.device_type)
        # 将输入张量在指定网格上分发，并应用在Shard(0)上
        dt_to_add = distribute_tensor(input_tensor, mesh, [Shard(0)])
        # 克隆分发后的张量，用于加法操作
        dt_to_mul = dt_to_add.clone()
        # 创建期望的加法操作后的张量
        expected_add_dt = dt_to_add.clone() + 3
        # 执行就地加法操作，并接收返回值
        add_res = dt_to_add.add_(3)
        # 创建期望的乘法操作后的张量
        expected_mul_dt = dt_to_mul.clone() * 3
        # 执行就地乘法操作，并接收返回值
        mul_res = dt_to_mul.mul_(3)
        # 断言就地加法操作后的返回值与原对象相同
        self.assertTrue(add_res is dt_to_add)
        # 断言加法操作后的结果与期望值在本地张量上相等
        self.assertEqual(add_res.to_local(), expected_add_dt.to_local())

        # 断言就地乘法操作后的返回值与原对象相同
        self.assertTrue(mul_res is dt_to_mul)
        # 断言乘法操作后的结果与期望值在本地张量上相等
        self.assertEqual(mul_res.to_local(), expected_mul_dt.to_local())

        # 测试就地加法操作对自身和其他数据张量与其他规格的影响
        # 并确保输出规格不变
        shard_spec = [Shard(0)]
        partial_spec = [Partial()]
        # 在指定网格上分发输入张量，并应用在Shard(0)上
        dt_to_inplace_add = distribute_tensor(input_tensor, mesh, shard_spec)
        # 创建局部梯度张量，使用本地随机张量与网格和部分规格
        partial_grad = DTensor.from_local(torch.randn(12, 3), mesh, partial_spec)
        # 执行就地加法操作，并接收返回值
        res = dt_to_inplace_add.add_(partial_grad)
        # 断言就地加法操作后的返回值与原对象相同
        self.assertTrue(res is dt_to_inplace_add)
        # 断言就地加法操作后的位置信息与Shard(0)相同
        self.assertTrue(res.placements == tuple(shard_spec))

    # 使用通信装饰器定义测试方法
    @with_comms
    def test_op_out_variant(self):
        # 创建一个设备网格对象，并指定设备类型和全球大小
        mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        # 创建一个随机张量，设备类型与测试中指定的设备类型一致
        input_tensor = torch.randn((12, 3), device=self.device_type)
        # 将输入张量在指定网格上分发，并应用在Shard(0)上
        sharded_dt_input = distribute_tensor(input_tensor, mesh, [Shard(0)])
        # 创建期望的加法操作后的张量
        expected_dt = sharded_dt_input.clone() + 3
        # 克隆分发后的张量，用于输出变体操作
        sharded_dt_out = sharded_dt_input.clone()
        # 执行带有输出变体的加法操作，并接收返回值
        res = torch.add(sharded_dt_input, 3, out=sharded_dt_out)
        # 断言输出变体操作后的返回值与原对象相同
        self.assertTrue(res is sharded_dt_out)
        # 断言输出变体操作后的结果与期望值在本地张量上相等
        self.assertEqual(sharded_dt_out.to_local(), expected_dt.to_local())

        # 测试带有输出变体的加法操作对其他规格的影响
        # 并确保输出规格不变
        replica_spec = [Replicate()]
        # 在指定网格上分发输入张量，并应用在Replicate()上
        replicate_out = distribute_tensor(input_tensor, mesh, replica_spec)
        # 创建期望的加法操作后的张量
        expected_dt = replicate_out.clone() + 3
        # 执行带有输出变体的加法操作，并接收返回值
        res = torch.add(sharded_dt_input, 3, out=replicate_out)
        # 断言输出变体操作后的返回值与原对象相同
        self.assertTrue(res is replicate_out)
        # 断言输出变体操作后的位置信息与Replicate()相同
        self.assertTrue(res.placements == tuple(replica_spec))
        # 断言输出变体操作后的结果与期望值在本地张量上相等
        self.assertEqual(replicate_out.to_local(), expected_dt.to_local())

    # 使用通信装饰器定义测试方法
    @with_comms
    def test_empty_like(self):
        # 创建一个设备网格对象，并指定设备类型和全球大小
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        # 指定在Shard(0)上的分片规格
        shard_spec = [Shard(0)]

        # 创建一个随机张量，并标记需要梯度计算
        input_tensor = torch.randn(4, 8, requires_grad=True)
        # 从本地创建分发张量，使用设备网格和分片规格
        dist_tensor = DTensor.from_local(input_tensor, device_mesh, shard_spec)
        # 创建与dist_tensor相同形状的空张量
        empty_like_dt = torch.empty_like(dist_tensor)
        # 空张量不确定性，所以我们只检查分片传播是否正常工作
        self.assertEqual((4, 8), empty_like_dt.to_local().shape)
    def test_fill_inplace(self):
        # 创建设备网格对象，使用设备类型和全球大小的范围
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        # 创建分片规格，这里仅包含一个Shard对象
        shard_spec = [Shard(0)]

        # 生成一个形状为(4, 8)的随机张量，要求梯度计算
        input_tensor = torch.randn(4, 8, requires_grad=True)
        # 将本地张量转换为分布式张量，使用给定的设备网格和分片规格
        dist_tensor = DTensor.from_local(input_tensor, device_mesh, shard_spec)
        # 在分布式张量上填充值为42.0，返回填充后的分布式张量
        full_like_dt = torch.fill_(dist_tensor, 42.0)
        # 生成一个形状为(4, 8)，元素值为42.0的张量
        full_expected = torch.full((4, 8), 42.0)
        # 断言填充后的分布式张量与期望的本地张量相等
        self.assertEqual(full_expected, full_like_dt.to_local())
        # 断言原始分布式张量也与期望的本地张量相等
        self.assertEqual(full_expected, dist_tensor.to_local())

    @with_comms
    def test_full_like(self):
        # 创建设备网格对象，使用设备类型和全球大小的范围
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        # 创建分片规格，这里仅包含一个Shard对象
        shard_spec = [Shard(0)]

        # 生成一个形状为(4, 8)的随机张量，要求梯度计算
        input_tensor = torch.randn(4, 8, requires_grad=True)
        # 将本地张量转换为分布式张量，使用给定的设备网格和分片规格
        dist_tensor = DTensor.from_local(input_tensor, device_mesh, shard_spec)
        # 根据分布式张量生成一个形状相同，元素值为42.0的张量
        full_like_dt = torch.full_like(dist_tensor, 42.0)
        # 生成一个形状为(4, 8)，元素值为42.0的张量
        full_expected = torch.full((4, 8), 42.0)
        # 断言生成的分布式张量与期望的本地张量相等
        self.assertEqual(full_expected, full_like_dt.to_local())

    @with_comms
    def test_ones_like(self):
        # 创建设备网格对象，使用设备类型和全球大小的范围
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        # 创建分片规格，这里仅包含一个Shard对象
        shard_spec = [Shard(0)]

        # 生成一个形状为(4, 8)的随机张量，要求梯度计算
        input_tensor = torch.randn(4, 8, requires_grad=True)
        # 将本地张量转换为分布式张量，使用给定的设备网格和分片规格
        dist_tensor = DTensor.from_local(input_tensor, device_mesh, shard_spec)
        # 根据分布式张量生成一个形状相同，元素值为1的张量
        ones_like_dt = torch.ones_like(dist_tensor)
        # 生成一个形状为(4, 8)，元素值为1的张量
        ones_expected = torch.ones(4, 8)
        # 断言生成的分布式张量与期望的本地张量相等
        self.assertEqual(ones_expected, ones_like_dt.to_local())

    @with_comms
    def test_ones_like_partial_sum(self):
        # 创建设备网格对象，使用设备类型和全球大小的范围
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        # 创建部分分片规格，这里包含一个Partial对象
        shard_spec = [Partial()]

        # 生成一个形状为(4, 8)的随机张量，要求梯度计算
        input_tensor = torch.randn(4, 8, requires_grad=True)
        # 将本地张量转换为分布式张量，使用给定的设备网格和部分分片规格
        dist_tensor = DTensor.from_local(input_tensor, device_mesh, shard_spec)
        # 断言分布式张量的形状为(4, 8)
        assert dist_tensor.shape == (4, 8)

        # 根据分布式张量生成一个形状相同，元素值为1的张量
        ones_like_dt = torch.ones_like(dist_tensor)
        # 生成一个形状为(4, 8)，元素值为1的张量
        ones_expected = torch.ones(dist_tensor.shape)
        # 断言生成的分布式张量与期望的全1张量相等
        self.assertEqual(ones_expected, ones_like_dt.full_tensor())

    @with_comms
    def test_fill_inplace_partial_sum(self):
        # 创建设备网格对象，使用设备类型和全球大小的范围
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        # 创建部分分片规格，这里包含一个Partial对象
        shard_spec = [Partial()]

        # 生成一个形状为(4, 8)的随机张量，要求梯度计算
        input_tensor = torch.randn(4, 8, requires_grad=True)
        # 将本地张量转换为分布式张量，使用给定的设备网格和部分分片规格
        dist_tensor = DTensor.from_local(input_tensor, device_mesh, shard_spec)
        # 断言分布式张量的形状为(4, 8)
        assert dist_tensor.shape == (4, 8)

        # 在分布式张量上就地填充值为8
        torch.fill_(dist_tensor, 8)
        # 生成一个形状与分布式张量相同，元素值为8*全球大小的张量
        fill_expected = torch.full(
            dist_tensor.shape, 8 * self.world_size, dtype=input_tensor.dtype
        )
        # 断言填充后的分布式张量与期望的全8张量相等
        self.assertEqual(fill_expected, dist_tensor.full_tensor())
    # 定义一个测试方法，用于测试部分求和的零张量操作
    def test_zeros_like_partial_sum(self):
        # 创建设备网格对象，指定设备类型和全局设备索引列表
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        # 创建一个部分分片规格列表，包含一个空的部分对象
        shard_spec = [Partial()]

        # 创建一个随机张量，形状为 (4, 8)，需要计算梯度
        input_tensor = torch.randn(4, 8, requires_grad=True)
        # 使用本地数据创建分布式张量对象
        dist_tensor = DTensor.from_local(input_tensor, device_mesh, shard_spec)
        # 断言分布式张量的形状为 (4, 8)
        assert dist_tensor.shape == (4, 8)

        # 创建一个与 dist_tensor 形状相同的零张量
        zeros_like_dt = torch.zeros_like(dist_tensor)
        # 创建一个期望的全零张量，形状为 (4, 8)
        zeros_expected = torch.zeros(dist_tensor.shape)
        # 断言两个张量是否相等
        self.assertEqual(zeros_expected, zeros_like_dt.full_tensor())

    # 使用通信装饰器标记的测试方法，测试零值张量的原地操作
    @with_comms
    def test_zero_inplace(self):
        # 创建设备网格对象，指定设备类型和全局设备索引列表
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        # 创建一个分片规格列表，包含一个索引为 0 的分片对象
        shard_spec = [Shard(0)]

        # 创建一个随机张量，形状为 (4, 8)，需要计算梯度
        input_tensor = torch.randn(4, 8, requires_grad=True)
        # 使用本地数据创建分布式张量对象
        dist_tensor = DTensor.from_local(input_tensor, device_mesh, shard_spec)
        # 原地将 dist_tensor 的值置零
        zeros_like_dt = torch.zero_(dist_tensor)
        # 创建一个期望的全零张量，形状为 (4, 8)
        zeros_expected = torch.zeros(4, 8)
        # 断言两个张量是否相等
        self.assertEqual(zeros_expected, zeros_like_dt.to_local())
        self.assertEqual(zeros_expected, dist_tensor.to_local())

    # 使用通信装饰器标记的测试方法，测试零值张量的创建操作
    @with_comms
    def test_zeros_like(self):
        # 创建设备网格对象，指定设备类型和全局设备索引列表
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        # 创建一个分片规格列表，包含一个索引为 0 的分片对象
        shard_spec = [Shard(0)]

        # 创建一个随机张量，形状为 (4, 8)，需要计算梯度
        input_tensor = torch.randn(4, 8, requires_grad=True)
        # 使用本地数据创建分布式张量对象
        dist_tensor = DTensor.from_local(input_tensor, device_mesh, shard_spec)
        # 创建一个与 dist_tensor 形状相同的零张量
        zeros_like_dt = torch.zeros_like(dist_tensor)
        # 创建一个期望的全零张量，形状为 (4, 8)
        zeros_expected = torch.zeros(4, 8)
        # 断言两个张量是否相等
        self.assertEqual(zeros_expected, zeros_like_dt.to_local())

    # 使用通信和 GPU 数量检查装饰器标记的测试方法
    @with_comms
    @skip_if_lt_x_gpu(4)
    # 定义一个测试方法，用于测试栈操作
    def test_stack(self):
        # 创建一个二维设备网格对象，并用指定的设备类型和张量初始化
        mesh_2d = DeviceMesh(
            self.device_type, torch.arange(self.world_size).reshape(2, 2)
        )
        # 创建部分复制和复制放置的列表
        partial_replicate_placement = [Partial(), Replicate()]
        # 创建部分放置的列表
        partial_placement = [Partial(), Partial()]

        # 使用局部复制和复制放置创建一个从本地张量到张量对象的映射
        partial_replicate_dt = DTensor.from_local(
            torch.randn(4, 8), mesh_2d, partial_replicate_placement
        )
        # 使用部分放置创建一个从本地张量到张量对象的映射
        partial_dt = DTensor.from_local(torch.randn(4, 8), mesh_2d, partial_placement)

        # 在维度0上堆叠两个张量对象
        stack_dt = torch.stack([partial_replicate_dt, partial_dt])
        # 断言堆叠后张量对象的放置与部分放置列表相同
        self.assertEqual(stack_dt.placements, tuple(partial_placement))
        # 断言堆叠后张量对象的形状为 (2, 4, 8)
        self.assertEqual(stack_dt.shape, (2, 4, 8))

        # 创建一个一维设备网格对象，并用指定的设备类型和张量初始化
        mesh_1d = DeviceMesh(self.device_type, torch.arange(self.world_size))
        # 创建一个全局输入张量
        global_input = torch.randn(8, 8)
        # 在一维网格上分发全局输入张量，并用指定的分片列表创建分布张量对象
        shard1_input = distribute_tensor(global_input, mesh_1d, [Shard(1)])
        # 克隆分片1的输入张量
        cloned_shard1_input = shard1_input.clone()
        # 在维度0上堆叠两个分片1的输入张量
        stack_shard1_dt = torch.stack([shard1_input, cloned_shard1_input])
        # 断言堆叠后张量对象的放置为 (Shard(2),)
        self.assertEqual(stack_shard1_dt.placements, (Shard(2),))
        # 断言堆叠后张量对象的形状为 (2, 8, 8)
        self.assertEqual(stack_shard1_dt.shape, (2, 8, 8))
        # 断言堆叠后张量对象的完整张量与全局输入张量在维度0上的堆叠结果相同
        self.assertEqual(
            stack_shard1_dt.full_tensor(), torch.stack([global_input, global_input])
        )

        # 在维度1上堆叠两个分片1的输入张量
        stack_dim1_shard1_dt = torch.stack([shard1_input, cloned_shard1_input], dim=1)
        # 断言堆叠后张量对象的放置为 (Shard(2),)
        self.assertEqual(stack_dim1_shard1_dt.placements, (Shard(2),))
        # 断言堆叠后张量对象的形状为 (8, 2, 8)
        self.assertEqual(stack_dim1_shard1_dt.shape, (8, 2, 8))
        # 断言堆叠后张量对象的完整张量与全局输入张量在维度1上的堆叠结果相同
        self.assertEqual(
            stack_dim1_shard1_dt.full_tensor(),
            torch.stack([global_input, global_input], dim=1),
        )
    # 定义测试方法，用于验证分布式张量的相等性操作
    def test_equal(self):
        # 创建设备网格对象，指定设备类型和全局设备 ID 列表
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        # 定义分片规范，这里仅包含一个分片
        shard_spec = [Shard(0)]

        # 创建本地输入张量，填充为全1
        input_tensor_1 = torch.ones(4, 4)
        # 使用本地输入张量创建分布式张量，基于给定的设备网格和分片规范
        dist_tensor_1 = DTensor.from_local(input_tensor_1, device_mesh, shard_spec)

        # 第一个测试用例：创建另一个全1张量，比较两个分布式张量的相等性
        input_tensor_2 = torch.ones(4, 4)
        dist_tensor_2 = DTensor.from_local(input_tensor_2, device_mesh, shard_spec)
        eq_result = dist_tensor_1.equal(dist_tensor_2)
        # 断言：两个张量应当相等
        self.assertTrue(eq_result)

        # 第二个测试用例：在某些分片上，创建不同的张量，再次比较相等性
        if self.rank == 0:
            input_tensor_2 = torch.ones(4, 4)  # 在第一个分片上，张量保持全1
        else:
            input_tensor_2 = torch.randn(4, 4)  # 在其它分片上，张量为随机值
        dist_tensor_2 = DTensor.from_local(input_tensor_2, device_mesh, shard_spec)
        eq_result = dist_tensor_1.equal(dist_tensor_2)
        # 断言：预期这两个张量不相等
        self.assertFalse(eq_result)
        # 断言：两个张量应当具有相同的尺寸
        self.assertTrue(dist_tensor_1.is_same_size(dist_tensor_2))

        # 第三个测试用例：测试不同的复制规格，创建全1张量并比较
        replica_spec = [Replicate()]
        global_input = torch.ones(4 * self.world_size, 4)  # 创建全1张量
        dist_tensor_3 = DTensor.from_local(
            global_input, device_mesh, replica_spec, run_check=False
        )
        eq_result = dist_tensor_1.equal(dist_tensor_3)
        # 断言：预期这两个张量相等
        self.assertTrue(eq_result)
        # 断言：两个张量应当具有相同的尺寸
        self.assertTrue(dist_tensor_1.is_same_size(dist_tensor_3))

        # 第四个测试用例：测试具有部分分片内容差异的情况下的相等性
        eq_result = dist_tensor_2.equal(dist_tensor_3)
        # 断言：预期这两个张量不相等
        self.assertFalse(eq_result)
        # 断言：两个张量应当具有相同的尺寸
        self.assertTrue(dist_tensor_1.is_same_size(dist_tensor_3))
        # 断言：输入张量与分布式张量的尺寸应不相等
        self.assertFalse(input_tensor_2.is_same_size(dist_tensor_3))

    # 定义测试操作的私有方法，验证操作调用的结果与预期输出是否一致
    def _test_op(self, mesh, op_call, *args, **kwargs):
        # 执行操作调用，获取操作结果
        out = op_call(*args, **kwargs)
        # 创建分布式张量转换器对象，用于分析参数和关键字参数
        dtc = DTensorConverter(mesh, args, kwargs)
        # 遍历转换器，分别验证每个分布式操作的结果是否与预期的输出一致
        for d_args, d_kwargs in dtc:
            # 断言：分布式操作转换成功
            self.assertTrue(dtc.successful())
            # 执行分布式操作，获取结果张量
            d_out = op_call(*d_args, **d_kwargs)
            # 断言：分布式操作的完整张量结果应当与预期输出一致
            self.assertEqual(d_out.full_tensor(), out)

    # 使用通信装饰器的方法调用
    @with_comms
    # 定义一个名为 test_new_full 的测试方法
    def test_new_full(self):
        # 创建一个 DeviceMesh 对象，用给定的设备类型和世界大小列表初始化
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        # 创建一个 CommDebugMode 对象作为通信模式

        # 生成一个 12x8 的随机张量
        global_tensor = torch.randn(12, 8)
        # 定义不同的放置策略列表
        placements = [[Shard(0)], [Replicate()]]
        # 遍历每种放置策略
        for placement in placements:
            # 将全局张量 global_tensor 按照给定的设备网格和放置策略进行分布
            input_dt = distribute_tensor(global_tensor, device_mesh, placement)
            # 进入通信调试模式
            with comm_mode:
                # 使用 new_full 方法创建一个形状为 (4, 8)，值为 42.0 的新张量 new_full_diff_dt
                new_full_diff_dt = input_dt.new_full((4, 8), 42.0)
                # new_full_diff_dt 创建了一个复制的张量，不论 input_dt 的放置策略如何，
                # 不应触发任何通信。
                self.assertEqual(comm_mode.get_total_counts(), 0)
            # 创建预期的 new_full_diff_dt 张量
            new_full_diff_expected = torch.full((4, 8), 42.0)
            # 断言 new_full_diff_dt 的第一个放置策略是复制
            self.assertTrue(new_full_diff_dt.placements[0].is_replicate())
            # 断言 new_full_diff_dt 的本地化结果与预期相等
            self.assertEqual(new_full_diff_expected, new_full_diff_dt.to_local())

            # 再次进入通信调试模式
            with comm_mode:
                # 使用 new_full 方法创建一个形状为 (12, 8)，值为 42.0 的新张量 new_full_same_dt
                new_full_same_dt = input_dt.new_full((12, 8), 42.0)
                # new_full_same_dt 创建了一个与 input_dt 相同放置策略的张量，
                # 不应触发任何通信。
                self.assertEqual(comm_mode.get_total_counts(), 0)
            # 创建预期的 new_full_same_dt 张量
            new_full_same_expected = torch.full((12, 8), 42.0)
            # 断言 new_full_same_dt 的放置策略与预期相等
            self.assertEqual(new_full_same_dt.placements, placement)
            # 断言 new_full_same_dt 的张量与预期相等
            self.assertEqual(new_full_same_expected, new_full_same_dt.full_tensor())

    # 使用装饰器指定测试方法需要通信支持
    @with_comms
    # 定义一个测试方法，用于测试在新建空数据分布时的不同情况
    def test_new_empty_strided(self):
        # 创建设备网格对象，使用指定的设备类型和世界大小范围
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        # 创建用于调试的通信模式对象
        comm_mode = CommDebugMode()

        # 设置分片维度为1，创建包含一个分片对象的放置元组
        shard_dim = 1
        placement = (Shard(shard_dim),)

        # 创建一个形状为(12, 8)的随机张量
        global_tensor = torch.randn(12, 8)
        # 将全局张量分布到设备网格上指定的位置
        input_dt = distribute_tensor(global_tensor, device_mesh, placement)
        # 断言分片维度可以整除世界大小
        self.assertTrue(input_dt.shape[shard_dim] % self.world_size == 0)
        
        # 进入通信模式，创建一个新的空的按步进方式分布的数据分布张量
        with comm_mode:
            new_empty_strided_dt = input_dt.new_empty_strided((12, 8), (8, 1))
            # 断言通信模式的总计数为0
            self.assertEqual(comm_mode.get_total_counts(), 0)
        
        # 断言新建的空按步进方式分布的数据分布张量的放置方式与预期相同
        self.assertEqual(new_empty_strided_dt.placements, placement)
        # 断言新建的空按步进方式分布的数据分布张量的本地张量大小为(12, 8 // self.world_size)
        self.assertEqual(new_empty_strided_dt._local_tensor.size(), (12, 8 // self.world_size))
        # 断言新建的空按步进方式分布的数据分布张量的步进为(8 // self.world_size, 1)
        self.assertEqual(new_empty_strided_dt._local_tensor.stride(), (8 // self.world_size, 1))
        # 断言新建的空按步进方式分布的数据分布张量是连续的
        self.assertTrue(new_empty_strided_dt.contiguous() is new_empty_strided_dt)

        # 创建一个形状为(12, 7)的随机张量
        global_tensor = torch.randn(12, 7)
        # 将全局张量分布到设备网格上指定的位置
        input_dt = distribute_tensor(global_tensor, device_mesh, placement)
        # 断言分片维度不能整除世界大小
        self.assertTrue(input_dt.shape[shard_dim] % self.world_size != 0)
        
        # 进入通信模式，创建一个新的空的按步进方式分布的数据分布张量
        with comm_mode:
            new_empty_strided_dt = input_dt.new_empty_strided((12, 7), (7, 1))
            # 断言通信模式的总计数为0
            self.assertEqual(comm_mode.get_total_counts(), 0)
        
        # 断言新建的空按步进方式分布的数据分布张量的放置方式为复制
        self.assertEqual(new_empty_strided_dt.placements, (Replicate(),))
        # 断言新建的空按步进方式分布的数据分布张量的本地张量大小为(12, 7)
        self.assertEqual(new_empty_strided_dt._local_tensor.size(), (12, 7))
        # 断言新建的空按步进方式分布的数据分布张量的步进为(7, 1)
        self.assertEqual(new_empty_strided_dt._local_tensor.stride(), (7, 1))

        # 创建一个形状为(12, 8)的随机张量
        global_tensor = torch.randn(12, 8)
        # 将全局张量分布到设备网格上指定的位置
        input_dt = distribute_tensor(global_tensor, device_mesh, placement)
        
        # 进入通信模式，创建一个新的空的按步进方式分布的数据分布张量
        with comm_mode:
            new_empty_strided_dt = input_dt.new_empty_strided((12, 4), (4, 1))
            # 断言通信模式的总计数为0
            self.assertEqual(comm_mode.get_total_counts(), 0)
        
        # 断言新建的空按步进方式分布的数据分布张量的放置方式为复制
        self.assertEqual(new_empty_strided_dt.placements, (Replicate(),))
        # 断言新建的空按步进方式分布的数据分布张量的本地张量大小为(12, 4)
        self.assertEqual(new_empty_strided_dt._local_tensor.size(), (12, 4))
        # 断言新建的空按步进方式分布的数据分布张量的步进为(4, 1)
        self.assertEqual(new_empty_strided_dt._local_tensor.stride(), (4, 1))

    @with_comms
    # 定义一个名为 test_scatter 的测试函数，用于测试 scatter 操作的分布式行为
    def test_scatter(self):
        # 创建一个设备网格对象，指定设备类型和全局设备索引列表
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        # 创建一个调试通信模式对象
        comm_mode = CommDebugMode()

        # 第一种情况：所有内容都进行复制：输入复制、索引/源复制、输出复制
        global_indexs = [
            torch.tensor([[0, 1, 2, 0]]),  # 全局索引张量1
            torch.tensor([[0, 1, 2], [0, 1, 4]]),  # 全局索引张量2
        ]
        # 遍历 scatter 维度：0 和 1
        for scatter_dim in [0, 1]:
            # 源数据列表，包含张量和标量
            srcs = [torch.arange(1, 11).reshape((2, 5)), 4]
            # 遍历源数据列表
            for global_src in srcs:
                # 创建全局输入张量，全零，数据类型为 int64
                global_input = torch.zeros(3, 5, dtype=torch.int64)
                # 获取当前 scatter 维度的全局索引张量
                global_index = global_indexs[scatter_dim]

                # 将全局输入张量分发到设备网格上，使用复制策略
                input_dt = distribute_tensor(global_input.clone(), device_mesh, [Replicate()])
                # 将全局索引张量分发到设备网格上，使用复制策略
                index_dt = distribute_tensor(global_index, device_mesh, [Replicate()])
                # 如果全局源数据是张量，则将其分发到设备网格上，使用复制策略；否则直接使用
                if isinstance(global_src, torch.Tensor):
                    src_dt = distribute_tensor(global_src, device_mesh, [Replicate()])
                else:
                    src_dt = global_src

                # 在全局输入张量上进行 scatter 操作，按照 scatter_dim 维度进行分散，使用全局索引和全局源数据
                global_output = torch.scatter(global_input, scatter_dim, global_index, global_src)
                # 使用通信调试模式执行 scatter 操作，将分布式张量 input_dt 的 scatter 结果保存在 output_dt 中
                with comm_mode:
                    output_dt = torch.scatter(input_dt, scatter_dim, index_dt, src_dt)

                # 断言：通信模式的总计数应为 0
                self.assertEqual(comm_mode.get_total_counts(), 0)
                # 断言：输出分布式张量的部署应为 [Replicate()]，即复制策略
                self.assertEqual(output_dt.placements, [Replicate()])
                # 断言：将 output_dt 转换为本地张量后应与 global_output 相等
                self.assertEqual(output_dt.to_local(), global_output)

    @with_comms
    # 定义一个单元测试方法 `test_gather`
    def test_gather(self):
        # 创建一个 DeviceMesh 对象，表示设备类型和设备列表
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        # 创建一个 CommDebugMode 对象，用于通信调试模式

        # case 1: 全部复制情况下的测试
        global_input = torch.randn(12, 8, 16)  # 创建一个随机张量作为全局输入数据
        global_index = torch.randint(8, (4, 4, 8))  # 创建一个随机索引张量
        # 将全局输入数据进行分发，使用 Replicate 方式进行复制
        input_dt = distribute_tensor(global_input, device_mesh, [Replicate()])
        # 将全局索引进行分发，使用 Replicate 方式进行复制
        index_dt = distribute_tensor(global_index, device_mesh, [Replicate()])
        # 遍历 gather_dim，进行 gather 操作，并进行验证
        for gather_dim in [0, 1, 2]:
            global_output = torch.gather(global_input, gather_dim, global_index)
            # 进入通信模式，进行分布式 gather 操作
            with comm_mode:
                output_dt = torch.gather(input_dt, gather_dim, index_dt)
                # 验证通信计数为零
                self.assertEqual(comm_mode.get_total_counts(), 0)
            # 验证输出的数据分布方式为 Replicate
            self.assertEqual(output_dt.placements, [Replicate()])
            # 验证分布式张量的本地数据与全局输出一致
            self.assertEqual(output_dt.to_local(), global_output)

        # case 2: 输入数据分片，索引复制，输出部分遮罩的情况
        # 只在 gather 维度上索引大小为1且输入在 gather 维度上分片时有效
        from torch.distributed._tensor.ops.embedding_ops import _MaskPartial

        gather_dim = 1
        global_input = torch.randn(12, 8, 16)  # 创建一个随机张量作为全局输入数据
        global_index = torch.randint(8, (4, 1, 8))  # 创建一个随机索引张量
        global_output = torch.gather(global_input, gather_dim, global_index)
        # 将全局输入数据在 gather 维度上进行分片
        input_dt = distribute_tensor(global_input, device_mesh, [Shard(gather_dim)])
        # 将全局索引进行复制
        index_dt = distribute_tensor(global_index, device_mesh, [Replicate()])
        # 进入通信模式，进行分布式 gather 操作
        with comm_mode:
            output_dt = torch.gather(input_dt, gather_dim, index_dt)
            # 验证通信计数为零
            self.assertEqual(comm_mode.get_total_counts(), 0)
        # 验证输出数据分布方式为 _MaskPartial 类型
        self.assertIsInstance(output_dt.placements[0], _MaskPartial)
        # 验证分布式张量的完整数据与全局输出一致
        self.assertEqual(output_dt.full_tensor(), global_output)

        # case 3: 索引分片，输入复制，输出分片的情况
        # 只在分片维度为 gather 维度时有效
        global_input = torch.randn(12, 8, 16)  # 创建一个随机张量作为全局输入数据
        global_index = torch.randint(8, (4, 4, 8))  # 创建一个随机索引张量
        # 遍历 gather_dim，进行 gather 操作，并进行验证
        for gather_dim in range(len(global_index.shape)):
            # 将全局输入数据进行复制
            input_dt = distribute_tensor(global_input, device_mesh, [Replicate()])
            # 将全局索引在 gather 维度上进行分片
            index_dt = distribute_tensor(global_index, device_mesh, [Shard(gather_dim)])
            global_output = torch.gather(global_input, gather_dim, global_index)
            # 进入通信模式，进行分布式 gather 操作
            with comm_mode:
                output_dt = torch.gather(input_dt, gather_dim, index_dt)
                # 验证通信计数为零
                self.assertEqual(comm_mode.get_total_counts(), 0)
            # 验证输出的数据分布方式为 Shard(gather_dim)
            self.assertEqual(output_dt.placements, [Shard(gather_dim)])
            # 验证分布式张量的完整数据与全局输出一致
            self.assertEqual(output_dt.full_tensor(), global_output)

    @with_comms
    @with_comms
    # 测试方法，用于验证类型提升功能
    def test_where_type_promotion(self):
        # 创建一个设备网格对象，使用给定的设备类型和世界大小列表（一维网格）
        mesh = DeviceMesh(self.device_type, list(range(self.world_size)))  # 1D mesh

        # 定义分布规格列表
        specs = [[Shard(0)], [Replicate()]]
        # 遍历分布规格列表
        for spec in specs:
            # 创建一个随机张量
            global_tensor = torch.randn(12, 8)
            # 将全局张量分布到网格上，使用给定的分布规格
            mat = distribute_tensor(global_tensor, mesh, spec)
            # 使用 torch.where 进行条件检索，大于0的位置置为1，否则置为0
            res = torch.where(mat > 0, 1, 0)
            # 对比全局张量进行条件检索的结果
            ref = torch.where(global_tensor > 0, 1, 0)
            # 断言检查结果张量是否与参考张量相等
            self.assertEqual(res.full_tensor(), ref)

    # 测试方法，用于验证张量数据类型转换功能
    @with_comms
    def test_dtensor_dtype_conversion(self):
        # 创建设备网格对象，使用给定的设备类型和世界大小列表
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        # 定义分片规格列表，包含一个分片对象
        shard_spec = [Shard(0)]
        
        # 创建一个随机张量，数据类型为 torch.bfloat16
        local_tenor = torch.randn(2, 8, dtype=torch.bfloat16)
        # 将本地张量转换为分布式张量对象，使用给定的设备网格和分片规格
        bf16_sharded_dtensor = DTensor.from_local(local_tenor, device_mesh, shard_spec)
        # 断言检查分布式张量对象的数据类型是否为 torch.bfloat16
        self.assertEqual(bf16_sharded_dtensor.dtype, torch.bfloat16)
        # 将分布式张量对象转换回本地张量，再次断言检查数据类型是否为 torch.bfloat16
        self.assertEqual(bf16_sharded_dtensor.to_local().dtype, torch.bfloat16)

        # 将分布式张量对象转换为 torch.float32 数据类型
        fp32_sharded_dtensor = bf16_sharded_dtensor.float()
        # 断言检查转换后的数据类型是否为 torch.float32
        self.assertEqual(fp32_sharded_dtensor.dtype, torch.float32)
        # 将转换后的分布式张量对象再次转换回本地张量，断言检查数据类型是否为 torch.float32
        self.assertEqual(fp32_sharded_dtensor.to_local().dtype, torch.float32)

        # 将 torch.float32 类型的分布式张量对象转换回 torch.bfloat16 类型
        bf16_sharded_dtensor1 = fp32_sharded_dtensor.type_as(bf16_sharded_dtensor)
        # 断言检查转换后的数据类型是否为 torch.bfloat16
        self.assertEqual(bf16_sharded_dtensor1.dtype, torch.bfloat16)
        # 将转换后的分布式张量对象再次转换回本地张量，断言检查数据类型是否为 torch.bfloat16
        self.assertEqual(bf16_sharded_dtensor1.to_local().dtype, torch.bfloat16)

        # 导入获取分片属性缓存信息的函数
        from torch.distributed._tensor.debug import get_sharding_prop_cache_info

        # 在这一步，应当只有缓存未命中的情况
        hits, misses, _, _ = get_sharding_prop_cache_info()
        # 断言检查缓存命中次数是否为 0
        self.assertEqual(hits, 0)
        # 断言检查缓存未命中次数是否为 2
        self.assertEqual(misses, 2)

        # 再次将分布式张量对象转换为 torch.float32 类型，并检查缓存命中情况
        fp32_sharded_dtensor1 = bf16_sharded_dtensor1.float()
        # 断言检查缓存命中次数是否为 1
        self.assertEqual(hits, 1)
        # 断言检查缓存未命中次数是否为 2
        self.assertEqual(misses, 2)
# 如果当前脚本被作为主程序执行（而非被导入其他模块），则执行以下代码
if __name__ == "__main__":
    # 调用运行测试的函数
    run_tests()
```