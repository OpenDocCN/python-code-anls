# `.\pytorch\test\distributed\_tensor\test_math_ops.py`

```
# 导入必要的模块和类
import copy  # 导入 copy 模块用于复制对象
import itertools  # 导入 itertools 模块用于迭代工具

import torch  # 导入 PyTorch 库

# 导入分布式相关模块和函数
from torch.distributed._tensor import DeviceMesh, distribute_module, distribute_tensor
from torch.distributed._tensor.debug import CommDebugMode  # 导入通信调试模式类
from torch.distributed._tensor.ops.utils import is_tensor_partial, normalize_dim  # 导入工具函数
from torch.distributed._tensor.placement_types import Replicate, Shard  # 导入数据放置类型
from torch.testing._internal.common_utils import run_tests  # 导入测试运行函数
from torch.testing._internal.distributed._tensor.common_dtensor import (  # 导入分布式张量测试基类和相关函数
    DTensorTestBase,
    skip_unless_torch_gpu,
    with_comms,
)

# 使用 torch.ops.c10d_functional 别名
funcol = torch.ops.c10d_functional

# DistMathOpsTest 类，继承自 DTensorTestBase
class DistMathOpsTest(DTensorTestBase):

    # 线性操作规约方法
    def linear_op_reductions(self, op_str):
        # 构建设备网格
        device_mesh = self.build_device_mesh()
        # 定义分片规范
        shard_spec = [Shard(0)]

        # 创建形状为 (12, 8, 8) 的随机张量
        tensor = torch.randn(12, 8, 8)
        # 将张量分发到设备网格上，并应用分片规范
        dtensor = distribute_tensor(tensor, device_mesh, shard_spec)

        # 获取张量的操作函数和分布式张量的操作函数
        op = getattr(tensor, op_str)
        op_dt = getattr(dtensor, op_str)

        # 是否保持维度的可能性：True, False, None
        keep_dim_or_not = [True, False, None]
        # 对张量的每个维度执行测试
        for dim in range(tensor.ndim):
            for keep_dim in keep_dim_or_not:
                args = (dim, keep_dim) if keep_dim is not None else (dim,)
                # 对于 max 和 min 操作，当指定维度时返回元组
                if op_str in ("max", "min"):
                    dim_reduced_tensor, _ = op(*args)
                    dt_reduced, _ = op_dt(*args)
                else:
                    dim_reduced_tensor = op(*args)
                    dt_reduced = op_dt(*args)
                # 获取分布式张量的维度规约后的完整张量并断言相等
                dt_dim_reduced_tensor = dt_reduced.full_tensor()
                self.assertEqual(dt_dim_reduced_tensor, dim_reduced_tensor)

        # 执行全局操作并断言分布式张量的结果与普通张量相等
        full_reduced_tensor = op()
        dt_full_reduced = op_dt().full_tensor()
        self.assertEqual(dt_full_reduced, full_reduced_tensor)

    # 测试线性操作规约的方法
    @with_comms
    def test_linear_op_reductions(self):
        for op_str in ("all", "sum", "prod", "max", "min"):
            self.linear_op_reductions(op_str)

    # 测试 mean 方法的方法
    @with_comms
    @skip_unless_torch_gpu  # 仅在存在 GPU 的情况下执行测试
    def test_mean(self):
        self.linear_op_reductions("mean")

    # TODO: forward test can be removed once test_softmax_with_bwd passes on CPU
    @with_comms
    # 定义一个测试 softmax 前向传播的方法，用于测试分布式环境下的计算
    def test_softmax_fwd(self):
        # 构建设备网格，用于模拟分布式计算的设备布局
        device_mesh = self.build_device_mesh()

        # 生成一个随机张量 x，形状为 (8, 12, 16)，在指定设备上进行操作
        x = torch.rand(8, 12, 16, device=self.device_type)
        
        # dims 用于将 -1 转换为实际维度
        dims = range(3)
        
        # softmax_dims 和 shard_dims 是用于测试的维度列表
        softmax_dims = [-1, 0, 1, 2]
        shard_dims = [-1, 0, 1, 2]
        
        # 生成 softmax_dim 和 shard_dim 的所有可能组合
        test_list = list(itertools.product(softmax_dims, shard_dims))

        # 遍历所有组合进行测试
        for softmax_dim, shard_dim in test_list:
            # 在指定维度 softmax_dim 上进行 softmax 操作，返回本地结果 local_y
            local_y = torch.nn.functional.softmax(
                x, dim=softmax_dim, dtype=torch.float32
            )
            
            # 将张量 x 在 device_mesh 上分发，以模拟分布式计算
            dist_x = distribute_tensor(x, device_mesh, [Shard(shard_dim)])
            
            # 在分布式环境下，在 softmax_dim 维度上进行 softmax 操作，返回分布式结果 dist_y
            dist_y = torch.nn.functional.softmax(
                dist_x, dim=softmax_dim, dtype=torch.float32
            )
            
            # 根据维度信息，规范化 shard_dim
            shard_dim = normalize_dim(shard_dim, dist_x.ndim)
            
            # 如果 dims[shard_dim] 等于 dims[softmax_dim]
            if dims[shard_dim] == dims[softmax_dim]:
                # 断言分布式结果 dist_y 是复制的结果
                self.assertTrue(dist_y.placements[0].is_replicate())
                # 断言分布式结果 dist_y 转为本地结果与 local_y 相等
                self.assertEqual(dist_y.to_local(), local_y)
            else:
                # 断言分布式结果 dist_y 在 shard_dim 维度上是分片的
                self.assertTrue(dist_y.placements[0].is_shard(dim=shard_dim))
                # 断言分布式结果 dist_y 的完整张量与 local_y 相等
                self.assertEqual(dist_y.full_tensor(), local_y)

    # TODO: get test_softmax_with_bwd pass on CPU
    # DTensor's _softmax_backward_data produces wrong result on CPU on certain dimension.
    # fail_on_cpu_list = [(0, -1), (1, -1)]
    @with_comms
    @skip_unless_torch_gpu
    # 定义一个测试方法，用于测试带有反向传播的 softmax 函数
    def test_softmax_with_bwd(self):
        # 构建设备网格
        device_mesh = self.build_device_mesh()

        # 定义维度范围，用于将 -1 转换为实际的维度
        dims = range(3)
        # softmax 可用的维度
        softmax_dims = [-1, 0, 1, 2]
        # 分片维度
        shard_dims = [-1, 0, 1, 2]
        # 使用 softmax_dims 和 shard_dims 生成所有组合的列表
        test_list = list(itertools.product(softmax_dims, shard_dims))

        # 遍历所有组合
        for params in test_list:
            softmax_dim, shard_dim = params
            # 创建一个随机张量 x，并标记需要计算梯度
            x = torch.rand(8, 12, 16, device=self.device_type, requires_grad=True)
            self.assertTrue(x.requires_grad)
            # 在本地计算 softmax，并对结果求和
            local_y = torch.nn.functional.softmax(
                x, dim=softmax_dim, dtype=torch.float32
            ).sum()
            # 执行反向传播
            local_y.backward()

            # 在设备网格上分发张量 x
            dist_x = distribute_tensor(x, device_mesh, [Shard(shard_dim)])
            self.assertTrue(dist_x.requires_grad)
            # 计算分布式 softmax
            dist_softmax = dist_x.softmax(dim=softmax_dim)
            # 标准化分片维度
            shard_dim = normalize_dim(shard_dim, dist_x.ndim)
            # 根据维度是否相同判断分布是否为复制
            if dims[softmax_dim] == dims[shard_dim]:
                self.assertTrue(dist_softmax.placements[0].is_replicate())
            else:
                self.assertTrue(dist_softmax.placements[0].is_shard(dim=shard_dim))
            # 计算分布式 softmax 的和
            dist_y = dist_softmax.sum()
            # 根据维度是否相同判断分布是否为复制或部分
            if dims[softmax_dim] == dims[shard_dim]:
                self.assertTrue(dist_y.placements[0].is_replicate())
            else:
                self.assertTrue(dist_y.placements[0].is_partial())
                # 重新分发分布式结果到设备网格
                dist_y = dist_y.redistribute(device_mesh, [Replicate()])
            # 断言分布式结果与本地结果相等
            self.assertEqual(dist_y.to_local(), local_y)
            # 确认分布式张量的梯度为 None
            self.assertIsNone(dist_x.grad)
            # 对分布式结果进行反向传播
            dist_y.backward()
            # 确认分布式张量的梯度不为 None
            self.assertIsNotNone(dist_x.grad)
            # 根据维度是否相同判断梯度的分布是否为复制或分片
            if dims[softmax_dim] == dims[shard_dim]:
                self.assertTrue(dist_x.grad.placements[0].is_replicate())
            else:
                self.assertTrue(dist_x.grad.placements[0].is_shard(dim=shard_dim))
            # 断言分布式张量的梯度与本地张量的梯度相等
            self.assertEqual(dist_x.grad.full_tensor(), x.grad)

    # 标记一个方法，用于测试分片数学运算
    @with_comms
    @skip_unless_torch_gpu
    @with_comms
    def test_shard_math_ops(self):
        # 定义设备网格的形状
        mesh_shape = (2, self.world_size // 2)
        # 创建设备网格对象
        mesh = DeviceMesh(
            self.device_type,
            torch.arange(self.world_size).reshape(*mesh_shape),
        )
        # 创建全局张量
        global_tensor = torch.ones(4, 4)
        # 将全局张量分布到双份分片张量
        double_shard_tensor = distribute_tensor(
            global_tensor, mesh, [Shard(0), Shard(0)]
        )
        # 将全局张量分布到完全分片张量
        fully_shard_tensor = distribute_tensor(
            global_tensor, mesh, [Shard(0), Shard(1)]
        )

        # 遍历四种数学运算
        for op in [torch.add, torch.sub, torch.mul, torch.div]:
            # 预期的结果张量
            expect_rs = op(global_tensor, 2)
            # 计算双份分片张量经过数学运算后的全张量结果
            double_shard_full_tensor = op(double_shard_tensor, 2).full_tensor()
            # 断言结果张量与预期结果相等
            self.assertEqual(double_shard_full_tensor, expect_rs)

            # 计算完全分片张量经过数学运算后的全张量结果
            fully_shard_full_tensor = op(fully_shard_tensor, 2).full_tensor()
            # 断言结果张量与预期结果相等
            self.assertEqual(fully_shard_full_tensor, expect_rs)
    def test_layer_norm_fwd(self):
        # 构建设备网格
        device_mesh = self.build_device_mesh()

        # NLP example from pytorch docs
        # https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
        # 设置示例中的批量大小、句子长度和嵌入维度
        batch, sentence_length, embedding_dim = 20, 5, 10
        # 创建随机张量 x，表示文本处理的输入数据，维度为 batch x sentence_length x embedding_dim
        x = torch.rand(batch, sentence_length, embedding_dim, device=self.device_type)
        # 构建用于测试的不同配置列表
        norm_shape_idx_list = list(range(x.ndim))
        shard_dims = [-1, 0, 1, 2]
        elementwise_affine_list = [False, True]
        test_config_list = list(
            itertools.product(shard_dims, norm_shape_idx_list, elementwise_affine_list)
        )

        # 对每个配置进行测试
        for shard_dim, norm_idx, elementwise_affine in test_config_list:
            # 根据当前配置获取被归一化的形状
            normalized_shape = x.shape[norm_idx:]
            # 创建 LayerNorm 对象，用于对输入数据进行归一化处理
            layer_norm = torch.nn.LayerNorm(
                normalized_shape,
                elementwise_affine=elementwise_affine,
                device=self.device_type,
            )
            # 在当前设备类型上进行深度复制
            layer_norm_local = copy.deepcopy(layer_norm).to(self.device_type)

            # 定义用于复制参数的函数，确保权重和偏置被正确分发到设备网格上
            def _replicate_fn(name, module, device_mesh):
                for name, param in module.named_parameters():
                    if name in ["weight", "bias"]:
                        param_dist = torch.nn.Parameter(
                            distribute_tensor(param, device_mesh, [Replicate()])
                        )
                        module.register_parameter(name, param_dist)

            # 在设备网格上分发 LayerNorm 模块
            layer_norm_dist = distribute_module(layer_norm, device_mesh, _replicate_fn)

            # 本地和分布式环境下对输入数据进行处理
            x_local = x
            x_dist = distribute_tensor(x, device_mesh, [Shard(shard_dim)])

            # 在本地执行 layer normalization
            y_local = layer_norm_local(x_local)
            
            # 确保前向传播中的 layer norm 不会引入额外的集合操作
            comm_mode = CommDebugMode()
            with comm_mode:
                y_dist = layer_norm_dist(x_dist)

            # 断言集合操作的次数不超过一个
            self.assertLessEqual(
                comm_mode.get_total_counts(),
                1,  # TODO: This should be 0!
                f"comm count={comm_mode.get_total_counts()}, "
                f"shard_dim={shard_dim}, norm_shape={normalized_shape}, elem_affine={elementwise_affine}",
            )

            # 导入用于分布式张量的元数据
            from torch.distributed._tensor.placement_types import TensorMeta

            # 获取分布式结果张量的元数据，并断言其正确性
            dtensor_meta = y_dist._spec.tensor_meta
            assert isinstance(dtensor_meta, TensorMeta)
            # 确保分片属性中的形状正确
            self.assertEqual(y_local.shape, dtensor_meta.shape)
            # 断言本地和分布式结果张量相等
            self.assertEqual(y_local, y_dist.full_tensor())

    @with_comms
    @with_comms
    # 定义测试函数 `test_topk`，用于测试 topk 操作
    def test_topk(self):
        # 构建设备网格
        device_mesh = self.build_device_mesh()
        # 定义不同的放置组合：Shard(0), Shard(1), Shard(2), Replicate()
        placement_combs = [Shard(0), Shard(1), Shard(2), Replicate()]

        # 创建调试通信模式对象
        comm_mode = CommDebugMode()

        # 生成一个随机张量 tensor，大小为 12x8x8，需要计算梯度
        tensor = torch.randn(12, 8, 8, requires_grad=True)
        # 在全局上执行 topk 操作，返回前三个元素及其索引
        global_topk = tensor.topk(3, dim=0)

        # 遍历每种放置组合
        for placement in placement_combs:
            # 将 tensor 在设备网格上分布，使用当前放置组合 (placement,)
            dtensor = distribute_tensor(tensor, device_mesh, (placement,))
            # 使用通信调试模式进行下列操作
            with comm_mode:
                # 在分布的 tensor 上执行 topk 操作，返回前三个元素及其索引
                out_dt = dtensor.topk(3, dim=0)
            
            # 如果当前放置是 Shard(0)
            if placement.is_shard(0):
                # 断言通信调试模式的总计数为 1
                self.assertEqual(comm_mode.get_total_counts(), 1)
                # 断言通信调试模式中 all_gather_into_tensor 的通信计数为 1
                self.assertEqual(
                    comm_mode.get_comm_counts()[funcol.all_gather_into_tensor],
                    1,
                )
            
            # 获取分布后的 tensor 的所有值，组成完整的张量
            out_full_values = out_dt.values.full_tensor()
            # 断言全局 topk 操作的值与分布后的完整值相等
            self.assertEqual(global_topk.values, out_full_values)

            # TODO: 支持反向散射（暂时注释掉的代码）
            # global_topk.values.sum().backward()
            # out_full_values.sum().backward()

    # 使用通信装饰器的测试函数 `test_shard0_svd`
    @with_comms
    def test_shard0_svd(self):
        # 构建设备网格
        device_mesh = self.build_device_mesh()
        # 设置随机数种子为 42
        torch.manual_seed(42)
        # 在给定设备类型上生成一个随机张量 replicated_x
        replicated_x = torch.randn((8, 8), device=self.device_type)
        # 将 replicated_x 在设备网格上分布，仅使用 Shard(0) 放置
        sharded_x = distribute_tensor(replicated_x, device_mesh, (Shard(0),))
        
        # 使用通信调试模式进行下列操作
        with CommDebugMode() as comm_mode:
            # 在分布的 sharded_x 上执行奇异值分解 SVD，不生成完整的 U, S, V
            U, S, V = torch.linalg.svd(sharded_x, full_matrices=False)
        
        # 在本地计算参考的完整 SVD 结果
        ref_U, ref_S, ref_V = torch.linalg.svd(replicated_x, full_matrices=False)
        
        # 断言分布后的 U, S, V 与参考的本地结果一致
        self.assertEqual(U.to_local(), ref_U)
        self.assertEqual(S.to_local(), ref_S)
        self.assertEqual(V.to_local(), ref_V)
        
        # 获取通信计数
        comm_counts = comm_mode.get_comm_counts()
        # 断言仅存在一个通信计数项
        self.assertEqual(len(comm_counts), 1)
        # 断言 all_gather_into_tensor 的通信计数为 1
        self.assertEqual(comm_counts[funcol.all_gather_into_tensor], 1)
# 如果当前脚本作为主程序执行（而不是被导入为模块），则执行 run_tests() 函数
if __name__ == "__main__":
    run_tests()
```