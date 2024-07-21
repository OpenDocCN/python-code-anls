# `.\pytorch\test\distributed\_tensor\test_matrix_ops.py`

```py
    @with_comms
    # 使用装饰器确保在通信环境中执行测试函数
    def test_addmm(self):
        # 创建设备网格，包含当前设备类型和整个世界大小的设备索引列表
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        # 定义分片规范，这里选择在第一个分片上操作
        shard_spec = [Shard(0)]
        # 定义复制规范，表示在所有设备上进行复制
        replica_spec = [Replicate()]

        # 创建一个大小为 (12, 8) 的随机张量，并将其分布到设备网格上的指定分片
        tensor_to_shard = torch.randn(12, 8)
        mat1 = distribute_tensor(tensor_to_shard, device_mesh, shard_spec)
        # 创建一个大小为 (8, 4) 的随机张量，并将其复制到设备网格上的所有设备
        tensor_to_replicate = torch.randn(8, 4)
        mat2 = distribute_tensor(tensor_to_replicate, device_mesh, replica_spec)
        # 创建一个大小为 (4,) 的随机张量，并将其复制到设备网格上的所有设备
        input_tensor = torch.randn(4)
        input = distribute_tensor(input_tensor, device_mesh, replica_spec)

        # 在分布式环境中执行 torch.addmm 操作，计算 dist_res
        dist_res = torch.addmm(input, mat1, mat2)
        # 在本地执行 torch.addmm 操作，计算 local_res
        local_res = torch.addmm(input_tensor, tensor_to_shard, tensor_to_replicate)
        # 断言分布式计算结果 dist_res 与本地计算结果 local_res 相等
        self.assertEqual(dist_res.full_tensor(), local_res)

    @with_comms
    # 使用装饰器确保在通信环境中执行测试函数
    def test_addmm_empty_operand(self):
        # 创建设备网格，包含当前设备类型和整个世界大小的设备索引列表
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        # 定义分片规范，这里选择在第一个分片上操作
        shard_spec = [Shard(0)]
        # 定义复制规范，表示在所有设备上进行复制
        replica_spec = [Replicate()]

        # 创建一个大小为 (12, 0) 的随机张量，并将其分布到设备网格上的指定分片
        tensor_to_shard = torch.randn(12, 0)
        mat1 = distribute_tensor(tensor_to_shard, device_mesh, shard_spec)
        # 创建一个大小为 (0, 4) 的随机张量，并将其复制到设备网格上的所有设备
        tensor_to_replicate = torch.randn(0, 4)
        mat2 = distribute_tensor(tensor_to_replicate, device_mesh, replica_spec)
        # 创建一个大小为 (4,) 的随机张量，并将其复制到设备网格上的所有设备
        input_tensor = torch.randn(4)
        inp = distribute_tensor(input_tensor, device_mesh, replica_spec)

        # 在分布式环境中执行 torch.addmm 操作，计算 dist_res
        dist_res = torch.addmm(inp, mat1, mat2)
        # 在本地执行 torch.addmm 操作，计算 local_res
        local_res = torch.addmm(input_tensor, tensor_to_shard, tensor_to_replicate)
        # 断言分布式计算结果 dist_res 与本地计算结果 local_res 相等
        self.assertEqual(dist_res.full_tensor(), local_res)
    # 定义测试方法，用于验证自动重分配功能
    def test_addmm_auto_redistribute(self):
        # 创建设备网格对象，指定设备类型和全局设备编号列表
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        # 定义仅在第一个分片上运行的分片规格
        shard0_spec = [Shard(0)]
        # 定义仅在第二个分片上运行的分片规格
        shard1_spec = [Shard(1)]
        # 定义在所有副本上复制数据的复制规格
        replica_spec = [Replicate()]

        # 创建一个随机张量，用于第二个分片上的操作，并分发到相应设备
        tensor_to_shard1 = torch.randn(12, 8, requires_grad=True)
        mat1 = distribute_tensor(tensor_to_shard1, device_mesh, shard1_spec)
        # 创建一个随机张量，用于第一个分片上的操作，并分发到相应设备
        tensor_to_shard0 = torch.randn(8, 4, requires_grad=True)
        mat2 = distribute_tensor(tensor_to_shard0, device_mesh, shard0_spec)
        # 创建一个随机张量，并复制到所有副本
        input_tensor = torch.randn(4, requires_grad=True)
        input = distribute_tensor(input_tensor, device_mesh, replica_spec)

        # 在本地计算局部结果
        local_res = torch.addmm(input_tensor, tensor_to_shard1, tensor_to_shard0)
        # 在分布式环境中计算分布式结果
        dist_res = torch.addmm(input, mat1, mat2)

        # 测试是否分布式结果为 DTensor 类型
        self.assertIsInstance(dist_res, DTensor)
        # 测试分布式结果的第一个位置是否为 Partial 类型
        self.assertIsInstance(dist_res.placements[0], Partial)

        # 测试分布式结果是否与本地结果相同
        dist_local_res = dist_res.full_tensor()
        self.assertEqual(local_res, dist_local_res)

        # 反向传播检查
        dist_local_res.sum().backward()
        local_res.sum().backward()
        # 断言 mat2 的梯度不为空，并且与 tensor_to_shard0 的梯度相同
        self.assertIsNotNone(mat2.grad)
        self.assertEqual(mat2.grad.full_tensor(), tensor_to_shard0.grad)

    @with_comms
    # 定义带有通信装饰器的测试方法
    def test_mm(self):
        # 创建设备网格对象，指定设备类型和全局设备编号列表
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        # 定义只在第一个分片上运行的分片规格
        shard0_spec = Shard(0)
        # 定义只在第二个分片上运行的分片规格
        shard1_spec = Shard(1)
        # 定义在所有副本上复制数据的复制规格
        replica_spec = Replicate()

        # 创建两个随机张量，并在本地计算矩阵乘积结果
        t1 = torch.randn(12, 8, requires_grad=True)
        t2 = torch.randn(8, 16, requires_grad=True)
        local_res = torch.mm(t1, t2)

        # 定义测试分布式张量分发和矩阵乘积的方法
        def test_placement_comb(
            placements1: List[Placement], placements2: List[Placement]
        ) -> None:
            # 分布式分发两个张量到指定设备网格上的指定位置
            dt1 = distribute_tensor(t1, device_mesh, placements1)
            dt2 = distribute_tensor(t2, device_mesh, placements2)
            # 在设备网格上执行矩阵乘积，并使用复制规格重分配结果
            dist_res: DTensor = cast(DTensor, torch.mm(dt1, dt2)).redistribute(
                device_mesh, [replica_spec]
            )
            # 断言分布式结果转换为本地结果后与本地计算结果相同
            self.assertEqual(dist_res.to_local(), local_res)
            # 反向传播检查
            grad_dist_res = torch.ones_like(dist_res)
            dist_res.backward(grad_dist_res)
            # 断言 dt1 的梯度不为空
            self.assertIsNotNone(dt1.grad)

        # 生成所有可能的分片规格组合
        placement_specs = [shard0_spec, shard1_spec, replica_spec]
        shard_specs_comb = list(itertools.product(placement_specs, placement_specs))
        # 对每一种分片规格组合执行测试方法
        for spec in shard_specs_comb:
            test_placement_comb([spec[0]], [spec[1]])
    # 定义测试方法 test_t，测试张量转置和分布式计算功能
    def test_t(self):
        # 创建设备网格对象，使用给定的设备类型和世界大小范围
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        # 创建分片规范，包含单个分片 0
        shard_spec = [Shard(0)]

        # 创建一个大小为 12x8 的随机张量，要求计算梯度
        tensor_to_transpose = torch.randn(12, 8, requires_grad=True)
        # 将张量分布到设备网格上，并根据分片规范分发
        mat = distribute_tensor(tensor_to_transpose, device_mesh, shard_spec)
        # 对分布式张量进行转置操作
        tranposed_mat = mat.t()
        # 断言转置后的张量尺寸为 [8, 12]
        self.assertEqual(tranposed_mat.size(), torch.Size([8, 12]))
        # 断言转置后的张量分布位置为分片 1
        self.assertEqual(tranposed_mat.placements, [Shard(1)])
        
        # 再次对转置后的张量进行转置操作
        tranposed_mat2 = tranposed_mat.t()
        # 断言再次转置后的张量尺寸为 [12, 8]
        self.assertEqual(tranposed_mat2.size(), torch.Size([12, 8]))
        # 断言再次转置后的张量分布位置与初始分片规范相同
        self.assertEqual(tranposed_mat2.placements, shard_spec)

    # 使用装饰器 with_comms 标记为带通信的测试方法
    @with_comms
    # 定义测试方法 test_t_partial，测试部分转置和分布式计算功能
    def test_t_partial(self):
        # 创建设备网格对象，使用给定的设备类型和世界大小范围
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))

        # 创建大小为 12x8 的随机张量 a 和大小为 8x4 的随机张量 b
        a = torch.randn(12, 8)
        b = torch.randn(8, 4)
        # 计算张量 a 和 b 的矩阵乘积，并对结果进行转置
        c = torch.mm(a, b).t()

        # 将张量 a 和 b 分布到设备网格上，并根据分片规范 [Shard(1)] 和 [Shard(0)] 进行分发
        da = distribute_tensor(a, device_mesh, [Shard(1)])
        db = distribute_tensor(b, device_mesh, [Shard(0)])

        # 计算分布式张量 da 和 db 的矩阵乘积，并对结果进行转置
        dc = torch.mm(da, db).t()

        # 断言 dc 的第一个分布位置是 Partial 类型的对象
        self.assertTrue(isinstance(dc.placements[0], Partial))

        # 检查局部和分布式操作结果是否匹配
        self.assertEqual(
            c,
            dc.redistribute(device_mesh, [Replicate()]).to_local(),
        )

    # 使用装饰器 with_comms 和 skip_unless_torch_gpu 标记为带通信且仅在使用 GPU 的情况下执行的测试方法
    def test_baddbmm(self):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        # 创建一个形状为 (4, 4, 8) 的随机张量，指定设备和梯度属性
        tensor = torch.rand(4, 4, 8, device=self.device_type, requires_grad=True)
        # 创建另一个形状相同的随机张量，指定设备和梯度属性
        batch_1 = torch.rand(4, 4, 8, device=self.device_type, requires_grad=True)
        # 创建形状为 (4, 8, 8) 的随机张量，指定设备和梯度属性
        batch_2 = torch.rand(4, 8, 8, device=self.device_type, requires_grad=True)

        def test_placement_comb(
            tensor_placements: List[Placement],
            batch_1_placements: List[Placement],
            batch_2_placements: List[Placement],
            beta: int,
            alpha: int,
            batch_1_grad: Optional[torch.Tensor],
        ) -> None:
            # 将 tensor 在设备网格上分发到指定位置
            tensor_dt = distribute_tensor(tensor, device_mesh, tensor_placements)
            # 将 batch_1 在设备网格上分发到指定位置
            batch_1_dt = distribute_tensor(batch_1, device_mesh, batch_1_placements)
            # 将 batch_2 在设备网格上分发到指定位置
            batch_2_dt = distribute_tensor(batch_2, device_mesh, batch_2_placements)
            # 在分布式环境中执行 baddbmm 操作，beta 和 alpha 是权重参数
            dist_res = cast(
                DTensor,
                torch.baddbmm(
                    tensor_dt, batch_1_dt, batch_2_dt, beta=beta, alpha=alpha
                ),
            ).redistribute(device_mesh, [Replicate()])
            # 将分布式结果转换为本地结果
            dist_local_res = dist_res.to_local()
            # 断言本地结果中没有 NaN 值
            assert not torch.isnan(local_result).any()
            assert not torch.isnan(dist_local_res).any()
            # 断言分布式结果与本地结果相等
            self.assertEqual(dist_local_res.detach(), local_result.detach())

            # TODO: add test backward
            # 创建全 1 张量作为 dist_res 的梯度，并执行反向传播
            # grad_dist_res = torch.ones_like(dist_res)
            # dist_res.backward(grad_dist_res)
            # 断言 batch_1_dt 的梯度不为 None
            # self.assertIsNotNone(batch_1_dt.grad)
            # 将 batch_1_dt 的梯度在设备网格上复制，并转换为本地梯度
            # batch_1_grad_local = batch_1_dt.grad.redistribute(
            #     device_mesh, [Replicate()]
            # ).to_local()
            # 断言本地梯度与预期的 batch_1_grad 相等
            # self.assertEqual(batch_1_grad_local, batch_1_grad)

        # 定义不同的设备分片和复制策略
        shard0_spec = Shard(0)
        shard1_spec = Shard(1)
        shard2_spec = Shard(2)
        replica_spec = Replicate()
        shard_specs = [shard0_spec, shard1_spec, shard2_spec, replica_spec]
        # 生成所有可能的分片组合
        shard_specs_comb = list(
            itertools.product(shard_specs, shard_specs, shard_specs)
        )
        # 如果 beta 是 0，则输入张量将被忽略
        numeric_params_comb = [
            (0.0, 0.5),  # zero-beta
            (0.8, 0.5),  # non-zero-beta
        ]

        # 对每组 beta 和 alpha 参数进行测试
        for beta, alpha in numeric_params_comb:
            # 在本地执行 baddbmm 操作，计算本地结果
            local_result = torch.baddbmm(
                tensor, batch_1, batch_2, beta=beta, alpha=alpha
            )
            # 创建全 1 张量作为本地结果的梯度，并执行反向传播
            grad_local_res = torch.ones_like(local_result)
            local_result.backward(grad_local_res)
            # 对所有分片组合进行测试
            for spec in shard_specs_comb:
                test_placement_comb(
                    [spec[0]], [spec[1]], [spec[2]], beta, alpha, batch_1.grad
                )

    @with_comms
    # 定义测试方法 test_bmm，用于测试矩阵乘法操作
    def test_bmm(self):
        # 创建设备网格对象，指定设备类型和世界大小范围
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        # 随机生成两个张量 mat1 和 mat2，形状分别为 (4, 8, 4) 和 (4, 4, 8)，在指定设备上，需要梯度计算
        mat1 = torch.rand(4, 8, 4, device=self.device_type, requires_grad=True)
        mat2 = torch.rand(4, 4, 8, device=self.device_type, requires_grad=True)
        # 对 mat1 和 mat2 执行批量矩阵乘法
        local_result = torch.bmm(mat1, mat2)
        # 创建一个与 local_result 形状相同的全 1 张量，用于反向传播
        grad_local_res = torch.ones_like(local_result)
        # 执行 local_result 的反向传播
        local_result.backward(grad_local_res)

        # 定义内部测试方法 test_placement_comb，用于测试分布式张量操作
        def test_placement_comb(
            placements1: List[Placement],
            placements2: List[Placement],
        ) -> None:
            # 将 mat1 和 mat2 分布到指定的设备网格上，根据 placements1 和 placements2
            mat1_dt = distribute_tensor(mat1, device_mesh, placements1)
            mat2_dt = distribute_tensor(mat2, device_mesh, placements2)
            # 执行分布式矩阵乘法，并进行数据重分配操作
            dist_res = cast(DTensor, torch.bmm(mat1_dt, mat2_dt)).redistribute(
                device_mesh, [Replicate()]
            )
            # 将分布式结果转换为本地结果
            dist_local_res = dist_res.to_local()
            # 断言分布式结果与本地结果相等
            self.assertEqual(dist_local_res, local_result)

            # 测试反向传播
            # TODO: figure out (replicate, shard1) fail on backward
            # it generates a different grad shape
            # 创建一个与 dist_res 形状相同的全 1 张量，用于反向传播
            grad_dist_res = torch.ones_like(dist_res)
            # 对 dist_res 执行反向传播
            dist_res.backward(grad_dist_res)
            # 断言 mat1_dt 的梯度不为空
            self.assertIsNotNone(mat1_dt.grad)
            # 将 mat1_dt 的梯度转换为本地张量，并进行数据重分配
            mat1_dt_grad = cast(DTensor, mat1_dt.grad).redistribute(
                device_mesh, [Replicate()]
            ).to_local()
            # 断言分布式梯度与本地梯度相等
            self.assertEqual(mat1_grad_local, mat1.grad)

        # 定义不同的分片规格和复制规格的组合列表
        shard0_spec = Shard(0)
        shard1_spec = Shard(1)
        shard2_spec = Shard(2)
        replica_spec = Replicate()
        placement_specs = [shard0_spec, shard1_spec, shard2_spec, replica_spec]
        shard_specs_comb = list(itertools.product(placement_specs, placement_specs))

        # 遍历所有可能的分片规格和复制规格组合，进行测试
        for spec in shard_specs_comb:
            test_placement_comb([spec[0]], [spec[1]])

    # 使用装饰器声明测试方法需要通信环境
    @with_comms
    # 使用装饰器声明仅在 Torch GPU 环境下运行测试方法
    @skip_unless_torch_gpu
# 如果当前脚本作为主程序运行（而非被导入），则执行 run_tests 函数
if __name__ == "__main__":
    run_tests()
```