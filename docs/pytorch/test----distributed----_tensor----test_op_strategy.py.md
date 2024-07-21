# `.\pytorch\test\distributed\_tensor\test_op_strategy.py`

```
# Owner(s): ["oncall: distributed"]

from itertools import chain  # 导入 chain 函数，用于扁平化多个可迭代对象成单个迭代器

import torch  # 导入 PyTorch 模块
from torch.distributed._tensor import DeviceMesh, DTensor  # 导入分布式相关的数据结构 DeviceMesh 和 DTensor
from torch.distributed._tensor._collective_utils import redistribute_cost  # 导入用于数据重分布成本计算的函数 redistribute_cost
from torch.distributed._tensor._op_schema import OpSchema, OpStrategy, PlacementStrategy  # 导入操作和策略相关的类
from torch.distributed._tensor.ops.basic_strategy import (
    EinsumDims,  # 导入处理 Einsum 操作维度的类 EinsumDims
    gen_einsum_strategies,  # 导入生成 Einsum 策略的函数 gen_einsum_strategies
)
from torch.distributed._tensor.placement_types import (
    DTensorSpec,  # 导入描述 DTensor 的规范的类 DTensorSpec
    Partial, Replicate, Shard, TensorMeta,  # 导入分布式张量的放置类型
)

from torch.testing._internal.common_utils import run_tests, TestCase  # 导入用于测试的通用函数和 TestCase 类
from torch.testing._internal.distributed._tensor.common_dtensor import DTensorOpTestBase  # 导入用于分布式张量操作测试的基类 DTensorOpTestBase


class TestEinsumDims(TestCase):
    def test_batch_dims(self):
        equation = "abc,abc->abc"
        input_dims, output_dim = EinsumDims.parse_equation(equation)  # 解析 Einsum 方程，获取输入维度和输出维度
        edims = EinsumDims.parse_dims(input_dims, output_dim)  # 根据输入和输出维度解析 Einsum 操作的维度信息

        self.assertEqual(edims.batch_dims, ["a", "b", "c"])  # 断言批处理维度与预期相符
        self.assertEqual(edims.contracting_dims, [])  # 断言收缩维度为空
        self.assertEqual(edims.lhs_out_only_dims, [])  # 断言左侧仅输出维度为空
        self.assertEqual(edims.rhs_out_only_dims, [])  # 断言右侧仅输出维度为空

    def test_mm_dims(self):
        equation = "mk,kn->mn"
        input_dims, output_dim = EinsumDims.parse_equation(equation)  # 解析 Einsum 方程，获取输入维度和输出维度
        edims = EinsumDims.parse_dims(input_dims, output_dim)  # 根据输入和输出维度解析 Einsum 操作的维度信息

        self.assertEqual(edims.batch_dims, [])  # 断言批处理维度为空
        self.assertEqual(edims.contracting_dims, ["k"])  # 断言收缩维度为 ["k"]
        self.assertEqual(edims.lhs_out_only_dims, ["m"])  # 断言左侧仅输出维度为 ["m"]
        self.assertEqual(edims.rhs_out_only_dims, ["n"])  # 断言右侧仅输出维度为 ["n"]

    def test_bmm_dims(self):
        equation = "bmk,bkn->bmn"
        input_dims, output_dim = EinsumDims.parse_equation(equation)  # 解析 Einsum 方程，获取输入维度和输出维度
        edims = EinsumDims.parse_dims(input_dims, output_dim)  # 根据输入和输出维度解析 Einsum 操作的维度信息

        self.assertEqual(edims.batch_dims, ["b"])  # 断言批处理维度为 ["b"]
        self.assertEqual(edims.contracting_dims, ["k"])  # 断言收缩维度为 ["k"]
        self.assertEqual(edims.lhs_out_only_dims, ["m"])  # 断言左侧仅输出维度为 ["m"]
        self.assertEqual(edims.rhs_out_only_dims, ["n"])  # 断言右侧仅输出维度为 ["n"]

        equation = "bcmk,bckn->bcmn"
        input_dims, output_dim = EinsumDims.parse_equation(equation)  # 解析 Einsum 方程，获取输入维度和输出维度
        edims = EinsumDims.parse_dims(input_dims, output_dim)  # 根据输入和输出维度解析 Einsum 操作的维度信息

        self.assertEqual(edims.batch_dims, ["b", "c"])  # 断言批处理维度为 ["b", "c"]
        self.assertEqual(edims.contracting_dims, ["k"])  # 断言收缩维度为 ["k"]
        self.assertEqual(edims.lhs_out_only_dims, ["m"])  # 断言左侧仅输出维度为 ["m"]
        self.assertEqual(edims.rhs_out_only_dims, ["n"])  # 断言右侧仅输出维度为 ["n"]
    # 定义一个单元测试方法，测试自由维度的解析和处理
    def test_free_dims(self):
        # 定义第一个测试用例的爱因斯坦求和表达式
        equation = "abc,ab->abc"
        # 解析输入维度和输出维度
        input_dims, output_dim = EinsumDims.parse_equation(equation)
        # 根据解析得到的维度信息，进一步解析维度对象
        edims = EinsumDims.parse_dims(input_dims, output_dim)

        # 断言批处理维度应为 ["a", "b"]
        self.assertEqual(edims.batch_dims, ["a", "b"])
        # 断言收缩维度为空列表
        self.assertEqual(edims.contracting_dims, [])
        # 断言左侧仅输出维度为 ["c"]
        self.assertEqual(edims.lhs_out_only_dims, ["c"])
        # 断言右侧仅输出维度为空列表
        self.assertEqual(edims.rhs_out_only_dims, [])

        # 定义第二个测试用例的爱因斯坦求和表达式
        equation = "abd,bf->abfd"
        # 解析输入维度和输出维度
        input_dims, output_dim = EinsumDims.parse_equation(equation)
        # 根据解析得到的维度信息，进一步解析维度对象
        edims = EinsumDims.parse_dims(input_dims, output_dim)

        # 断言批处理维度应为 ["b"]
        self.assertEqual(edims.batch_dims, ["b"])
        # 断言收缩维度为空列表
        self.assertEqual(edims.contracting_dims, [])
        # 断言左侧仅输出维度为 ["a", "d"]
        self.assertEqual(edims.lhs_out_only_dims, ["a", "d"])
        # 断言右侧仅输出维度为 ["f"]
        self.assertEqual(edims.rhs_out_only_dims, ["f"])
# 定义一个测试类 TestEinsumStrategies，继承自 DTensorOpTestBase 类
class TestEinsumStrategies(DTensorOpTestBase):
    
    # 定义一个属性方法 world_size，返回整数 4
    @property
    def world_size(self) -> int:
        return 4

    # 定义一个测试方法 test_mm_1d_mesh
    def test_mm_1d_mesh(self):
        # 调用父类方法 build_device_mesh()，创建设备网格
        mesh = self.build_device_mesh()

        # 调用 gen_einsum_strategies 函数，生成 "mk,kn->mn" 的所有策略
        all_strats = gen_einsum_strategies("mk,kn->mn", mesh)
        # 断言所有策略的数量等于 4
        self.assertEqual(len(all_strats.strategies), 4)

    # 定义一个测试方法 test_mm_2d_mesh
    def test_mm_2d_mesh(self):
        # 创建一个设备网格，使用 self.device_type 和 torch.arange(self.world_size) 生成
        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size).reshape(2, 2))

        # 调用 gen_einsum_strategies 函数，生成 "mk,kn->mn" 的所有策略
        all_strats = gen_einsum_strategies("mk,kn->mn", mesh)
        # 断言所有策略的数量等于 16
        self.assertEqual(len(all_strats.strategies), 16)

    # 定义一个测试方法 test_bmm_1d_mesh
    def test_bmm_1d_mesh(self):
        # 调用父类方法 build_device_mesh()，创建设备网格
        mesh = self.build_device_mesh()

        # 调用 gen_einsum_strategies 函数，生成 "bmk,bkn->bmn" 的所有策略
        all_strats = gen_einsum_strategies("bmk,bkn->bmn", mesh)
        # 断言所有策略的数量等于 5
        self.assertEqual(len(all_strats.strategies), 5)

    # 定义一个测试方法 test_bmm_2d_mesh
    def test_bmm_2d_mesh(self):
        # 创建一个设备网格，使用 self.device_type 和 torch.arange(self.world_size) 生成
        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size).reshape(2, 2))

        # 调用 gen_einsum_strategies 函数，生成 "bmk,bkn->bmn" 的所有策略
        all_strats = gen_einsum_strategies("bmk,bkn->bmn", mesh)
        # 断言所有策略的数量等于 25
        self.assertEqual(len(all_strats.strategies), 25)

    # 定义一个测试方法 test_pointwise_1d_mesh
    def test_pointwise_1d_mesh(self):
        # 调用父类方法 build_device_mesh()，创建设备网格
        mesh = self.build_device_mesh()

        # 调用 gen_einsum_strategies 函数，生成 "abcd,abcd->abcd" 的简单策略
        simple_strats = gen_einsum_strategies("abcd,abcd->abcd", mesh)
        # 断言简单策略的数量等于 5
        self.assertEqual(len(simple_strats.strategies), 5)

        # 调用 gen_einsum_strategies 函数，生成 "bcd,abcd->abcd" 的广播策略
        broadcast_strats = gen_einsum_strategies("bcd,abcd->abcd", mesh)
        # 断言广播策略的数量等于 5
        self.assertEqual(len(broadcast_strats.strategies), 5)

    # 定义一个测试方法 test_linearity_1d_mesh
    def test_linearity_1d_mesh(self):
        # 调用父类方法 build_device_mesh()，创建设备网格
        mesh = self.build_device_mesh()

        # 调用 gen_einsum_strategies 函数，生成 "abcd,abcd->abcd" 的所有线性策略
        all_strats = gen_einsum_strategies("abcd,abcd->abcd", mesh, linearity=True)
        # 断言所有线性策略的数量等于 6
        self.assertEqual(len(all_strats.strategies), 6)
    # 定义测试方法，测试在一维网格上重新分配成本
    def test_redistribute_cost_mesh_1d(self):
        # 构建设备网格的一维表示
        mesh_1d = self.build_device_mesh()
        # 定义分片位置，这里只有一个分片
        shard_placement = (Shard(0),)
        # 定义复制位置，这里使用复制策略
        replica_placement = (Replicate(),)
        # 定义部分位置，使用部分复制策略
        partial_placement = (Partial(),)

        # 创建一个大小为 10x10 的随机张量
        global_tensor = torch.randn(10, 10)
        # 提取全局张量的元数据
        global_tensor_meta = self._extract_tensor_meta(global_tensor)

        # 创建分片规范
        shard_spec = DTensorSpec(mesh_1d, shard_placement, global_tensor_meta)
        # 创建复制规范
        replica_spec = DTensorSpec(mesh_1d, replica_placement, global_tensor_meta)
        # 创建部分复制规范
        partial_spec = DTensorSpec(mesh_1d, partial_placement, global_tensor_meta)

        # 确保相同规范的重新分配成本为 0
        for spec in [shard_spec, replica_spec, partial_spec]:
            cost = redistribute_cost(spec, spec)
            self.assertEqual(cost, 0)

        # 计算从分片到复制的全聚合成本
        allgather_cost = redistribute_cost(shard_spec, replica_spec)
        # 计算从部分到分片的减少散射成本
        reduce_scatter_cost = redistribute_cost(partial_spec, shard_spec)
        # 计算从部分到复制的全归约成本
        allreduce_cost = redistribute_cost(partial_spec, replica_spec)
        # 断言全聚合成本与减少散射成本相等
        self.assertEqual(allgather_cost, reduce_scatter_cost)
        # 断言全归约成本比全聚合成本与减少散射成本的和要小至少 1
        self.assertTrue(allreduce_cost + 1 < allgather_cost + reduce_scatter_cost)
        # 计算从分片到部分的成本
        cost = redistribute_cost(shard_spec, partial_spec)
        # 断言分片到部分的成本为无穷大
        self.assertEqual(cost, float("inf"))
    def test_redistribute_cost_latency(self):
        # 定义测试函数，验证在addmm操作上的成本模型
        from torch.distributed._tensor.ops.matrix_ops import addmm_strategy
        
        # 构建设备网格
        mesh = self.build_device_mesh()
        
        # 定义第一个分片的放置策略和张量元数据
        shard0_placement = (Shard(0),)
        shard0_tensor_meta = self._extract_tensor_meta(torch.randn(8))
        
        # 定义部分放置策略和张量元数据
        partial_placement = (Partial(),)
        partial_tensor_meta = self._extract_tensor_meta(torch.randn(50, 6))
        
        # 定义第二个分片的放置策略和张量元数据
        shard1_placement = (Shard(1),)
        shard1_tensor_meta = self._extract_tensor_meta(torch.randn(6, 8))
        
        # 创建第一个分片的张量规格
        shard0_spec = DTensorSpec(mesh, shard0_placement, shard0_tensor_meta)
        
        # 创建部分张量的规格
        partial_spec = DTensorSpec(mesh, partial_placement, partial_tensor_meta)
        
        # 创建第二个分片的张量规格
        shard1_spec = DTensorSpec(mesh, shard1_placement, shard1_tensor_meta)
        
        # 定义操作模式的模式图谱
        op_schema = OpSchema(
            torch.ops.aten.addmm.default,
            (
                OpStrategy([PlacementStrategy(shard0_spec)]),
                OpStrategy([PlacementStrategy(partial_spec)]),
                OpStrategy([PlacementStrategy(shard1_spec)]),
            ),
            {},
        )
        
        # 根据操作模式计算输出策略
        output_strategy = addmm_strategy(mesh, op_schema)
        
        # 初始化策略成本的空字典
        strategy_costs = {}
        
        # 计算每种策略的重新分配成本，并存储在字典中
        for strategy in output_strategy.strategies:
            redistribute_cost = sum(chain.from_iterable(strategy.redistribute_cost))
            strategy_costs[str(strategy)] = redistribute_cost
        
        # 断言：验证成本模型考虑了集体延迟（即多次通信惩罚）
        self.assertTrue(
            strategy_costs["(S(0), R, S(1)) -> S(1)"]
            < strategy_costs["(R, S(0), R) -> S(0)"]
        )
        
        # 断言：验证单一全局归约是最佳选择
        self.assertEqual(
            strategy_costs["(S(0), R, S(1)) -> S(1)"], min(strategy_costs.values())
        )
    # 定义测试方法，用于测试在二维设备网格上重新分配成本的函数
    def test_redistribute_cost_mesh_2d(self):
        # 创建一个二维设备网格对象，使用给定设备类型和全局大小进行初始化
        mesh_2d = DeviceMesh(
            self.device_type, torch.arange(self.world_size).reshape(2, 2)
        )
        # 定义数据分片的放置方式为 (Shard(0), Shard(0))
        shard_placement = (Shard(0), Shard(0))
        # 定义数据复制的放置方式为 (Replicate(), Replicate())
        replica_placement = (Replicate(), Replicate())
        # 定义数据部分复制的放置方式为 (Partial(), Partial())
        partial_placement = (Partial(), Partial())

        # 创建一个8x8的随机张量
        global_tensor = torch.randn(8, 8)
        # 提取全局张量的元数据信息
        global_tensor_meta = self._extract_tensor_meta(global_tensor)

        # 创建 Shard 策略的张量规格对象
        shard_spec = DTensorSpec(mesh_2d, shard_placement, global_tensor_meta)
        # 创建 Replicate 策略的张量规格对象
        replica_spec = DTensorSpec(mesh_2d, replica_placement, global_tensor_meta)
        # 创建 Partial 策略的张量规格对象
        partial_spec = DTensorSpec(mesh_2d, partial_placement, global_tensor_meta)

        # 确保对于相同的规格，重新分配成本为0
        for spec in [shard_spec, replica_spec, partial_spec]:
            # 计算给定规格到其自身的重新分配成本
            cost = redistribute_cost(spec, spec)
            # 断言重新分配成本为0
            self.assertEqual(cost, 0)

        # 计算从 Shard 到 Replicate 的全部聚集成本
        allgather_cost = redistribute_cost(shard_spec, replica_spec)
        # 计算从 Partial 到 Replicate 的全部减少成本
        allreduce_cost = redistribute_cost(partial_spec, replica_spec)
        # 计算从 Partial 到 Shard 的全部减少散开成本
        reduce_scatter_cost = redistribute_cost(partial_spec, shard_spec)
        # 断言部分到全部减少的成本大于全部到全部聚集的成本
        self.assertTrue(allreduce_cost > allgather_cost)
        # 断言部分到全部减少的成本大于部分到部分减少散开的成本
        self.assertTrue(allreduce_cost > reduce_scatter_cost)
    def test_mm_strategies(self):
        # 导入矩阵乘法策略函数
        from torch.distributed._tensor.ops.matrix_ops import mm_strategy

        # 构建设备网格
        mesh = self.build_device_mesh()
        # 创建随机张量
        lhs_tensor = torch.randn(6, 8)
        rhs_tensor = torch.randn(8, 12)
        # 提取左右张量的元数据
        lhs_tensor_meta = self._extract_tensor_meta(lhs_tensor)
        rhs_tensor_meta = self._extract_tensor_meta(rhs_tensor)

        # 定义矩阵乘法组合
        mm_combs = (
            (Shard(0), Replicate()),
            (Replicate(), Shard(1)),
            (Shard(1), Shard(0)),
            (Replicate(), Replicate()),
        )
        # 遍历每一种组合
        for lhs, rhs in mm_combs:
            # 创建左右张量的规格对象
            lhs_spec = DTensorSpec(mesh, (lhs,), lhs_tensor_meta)
            rhs_spec = DTensorSpec(mesh, (rhs,), rhs_tensor_meta)

            # 创建操作模式架构对象，指定乘法操作和每个操作的策略
            op_schema = OpSchema(
                torch.ops.aten.mm.default,
                (
                    OpStrategy([PlacementStrategy(lhs_spec)]),
                    OpStrategy([PlacementStrategy(rhs_spec)]),
                ),
                {},
            )
            # 测试策略
            res_strategies = mm_strategy(mesh, op_schema)

            # 遍历返回的策略列表，查找与当前规格匹配的策略
            for strtgy in res_strategies.strategies:
                if strtgy.input_specs == (lhs_spec, rhs_spec):
                    # 断言重新分发成本为零
                    self.assertEqual(strtgy.redistribute_cost, [[0.0], [0.0]])
                    break

            # 创建操作模式架构对象，指定乘法操作和每个操作的规格
            op_schema = OpSchema(
                torch.ops.aten.mm.default,
                (lhs_spec, rhs_spec),
                {},
            )
            # 测试分片属性传播
            output_sharding = DTensor._op_dispatcher.sharding_propagator.propagate_op_sharding_non_cached(
                op_schema
            )
            # 断言不需要重新分发
            self.assertFalse(output_sharding.needs_redistribute)
    # 定义一个测试方法，用于测试矩阵乘法的策略
    def test_bmm_strategies(self):
        # 导入矩阵乘法的策略函数
        from torch.distributed._tensor.ops.matrix_ops import bmm_strategy

        # 构建设备网格
        mesh = self.build_device_mesh()
        
        # 随机生成两个张量，分别表示左操作数和右操作数
        lhs_tensor = torch.randn(8, 6, 8)
        rhs_tensor = torch.randn(8, 8, 12)
        
        # 提取左右操作数的元信息
        lhs_tensor_meta = self._extract_tensor_meta(lhs_tensor)
        rhs_tensor_meta = self._extract_tensor_meta(rhs_tensor)

        # 定义矩阵乘法的各种组合方式
        bmm_combs = (
            (Shard(0), Shard(0)),
            (Shard(1), Replicate()),
            (Replicate(), Shard(2)),
            (Shard(2), Shard(1)),
            (Replicate(), Replicate()),
        )
        
        # 遍历每种组合方式
        for lhs, rhs in bmm_combs:
            # 根据当前组合方式创建左右操作数的规格
            lhs_spec = DTensorSpec(mesh, (lhs,), lhs_tensor_meta)
            rhs_spec = DTensorSpec(mesh, (rhs,), rhs_tensor_meta)

            # 创建操作模式（OpSchema），使用默认的 torch.ops.aten.bmm.default 函数
            op_schema = OpSchema(
                torch.ops.aten.bmm.default,
                (
                    OpStrategy([PlacementStrategy(lhs_spec)]),
                    OpStrategy([PlacementStrategy(rhs_spec)]),
                ),
                {},
            )
            
            # 测试矩阵乘法的策略
            res_strategies = bmm_strategy(mesh, op_schema)

            # 遍历策略结果
            for strtgy in res_strategies.strategies:
                # 查找匹配当前左右操作数规格的策略
                if strtgy.input_specs == (lhs_spec, rhs_spec):
                    # 断言策略的重分配成本为零
                    self.assertEqual(strtgy.redistribute_cost, [[0.0], [0.0]])
                    break

            # 创建操作模式（OpSchema），用于测试分片属性
            op_schema = OpSchema(
                torch.ops.aten.bmm.default,
                (lhs_spec, rhs_spec),
                {},
            )
            
            # 测试分片属性的传播，确保不需要重新分配
            output_sharding = DTensor._op_dispatcher.sharding_propagator.propagate_op_sharding_non_cached(
                op_schema
            )
            self.assertFalse(output_sharding.needs_redistribute)
# 如果当前脚本作为主程序执行（而不是被导入到其他模块中），则执行 run_tests() 函数
if __name__ == "__main__":
    run_tests()
```