# `.\pytorch\test\distributed\_tensor\test_dtensor.py`

```py
    @with_comms
    # 使用装饰器确保在测试中启用通信模拟环境
    def test_dtensor_constructor(self):
        # 创建设备网格，用于分布式张量的放置
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        # 定义张量的放置策略为单个分片
        placements = [Shard(0)]
        # 创建一个本地张量，形状为3x3，并标记需要梯度
        local_tensor = torch.randn(3, 3, requires_grad=True)

        # 创建分布式张量的规格，包括设备网格、放置策略和张量元数据
        spec = DTensorSpec(
            device_mesh,
            tuple(placements),
            tensor_meta=TensorMeta(
                torch.Size([self.world_size * 3, 3]),
                local_tensor.stride(),
                local_tensor.dtype,
            ),
        )

        # 使用本地张量和规格创建分布式张量，标记需要梯度
        dist_tensor = DTensor(
            local_tensor,
            spec,
            requires_grad=True,
        )
        # 断言分布式张量的形状为 (self.world_size * 3, 3)
        self.assertEqual(dist_tensor.size(), torch.Size((self.world_size * 3, 3)))

        # 测试在特定条件下，使用相同规格但不需要梯度来创建分布式张量会触发警告
        with self.assertWarnsRegex(UserWarning, "To construct"):
            DTensor(
                local_tensor,
                spec,
                requires_grad=False,
            )
    # 测试 meta tensor 的分布式操作
    def test_meta_dtensor(self):
        # 构建设备网格
        device_mesh = self.build_device_mesh()
        # 定义分布规格
        dist_specs = [[Shard(0)], [Replicate()]]
        # 创建一个随机的 meta tensor，设备为 "meta"
        meta_tensor = torch.randn(1024, 2048, device="meta")
        
        # 遍历分布规格进行测试
        for dist_spec in dist_specs:
            # 在设备网格上进行 meta tensor 的分布式操作
            meta_dtensor = distribute_tensor(meta_tensor, device_mesh, dist_spec)
            # 断言 meta_dtensor 是 meta tensor
            self.assertTrue(meta_dtensor.is_meta)
            
            # 重新创建一个与 meta_dtensor 相同大小的 tensor，设备为 self.device_type
            meta_dtensor = torch.empty_like(meta_dtensor, device=self.device_type)
            # 用常数初始化 meta_dtensor
            torch.nn.init.constant_(meta_dtensor, 1.2)
            # 创建一个与 meta_dtensor.to_local() 相同大小的 tensor，并填充值为 1.2
            value_tensor = torch.empty_like(meta_dtensor.to_local()).fill_(1.2)
            # 断言 meta_dtensor 不再是 meta tensor
            self.assertFalse(meta_dtensor.is_meta)
            # 断言 meta_dtensor 的设备类型与 self.device_type 相同
            self.assertEqual(meta_dtensor.device.type, self.device_type)
            # 断言 meta_dtensor.to_local() 结果与 value_tensor 相同
            
            self.assertEqual(meta_dtensor.to_local(), value_tensor)
            
            # 测试在 meta tensor 上执行 from_local 操作
            meta_dtensor = DTensor.from_local(meta_tensor, device_mesh, dist_spec)
            # 创建一个与 meta_dtensor 相同大小的 tensor，设备为 self.device_type
            meta_dtensor = torch.empty_like(meta_dtensor, device=self.device_type)
            # 用常数初始化 meta_dtensor
            torch.nn.init.constant_(meta_dtensor, 1.5)
            # 断言 meta_dtensor 的设备类型与 self.device_type 相同
            self.assertEqual(meta_dtensor.device.type, self.device_type)
            # 创建一个与 meta_dtensor.to_local() 相同大小的 tensor，并填充值为 1.5
            value_tensor = torch.empty_like(meta_dtensor.to_local()).fill_(1.5)
            # 断言 meta_dtensor.to_local() 结果与 value_tensor 相同
            self.assertEqual(meta_dtensor.to_local(), value_tensor)

    # 使用通信装饰器的模块测试
    @with_comms
    def test_modules_w_meta_dtensor(self):
        # 创建一个 "meta" 类型的虚拟 MLP 模型
        model = DummyMLP("meta")
        # 构建设备网格
        device_mesh = self.build_device_mesh()
        # 并行化计划
        parallelize_plan = {
            "net1": ColwiseParallel(),
            "net2": RowwiseParallel(),
        }
        # 将模型并行化
        model_tp = parallelize_module(model, device_mesh, parallelize_plan)
        # 将模型转移到指定设备
        model_tp.to_empty(device=self.device_type)
        # 重置模型参数
        model_tp.reset_parameters()
        # 使用 SGD 优化器
        optim = torch.optim.SGD(model_tp.parameters(), lr=0.1)
        
        # 创建一个普通设备类型的虚拟 MLP 模型
        model_regular = DummyMLP(self.device_type)
        # 将普通模型并行化
        model_regular_tp = parallelize_module(
            model_regular, device_mesh, parallelize_plan
        )
        # 使用 SGD 优化器
        optim_regular = torch.optim.SGD(model_regular_tp.parameters(), lr=0.1)
        # 重置普通模型参数
        model_regular_tp.reset_parameters()
        
        # 设置随机种子
        torch.manual_seed(0)
        # 创建输入 tensor
        inp = torch.randn(20, 5, device=self.device_type)

        # 对并行化模型和普通模型执行前向传播
        output = model_tp(inp)
        output_regular = model_regular_tp(inp)
        # 断言两个输出结果相等
        self.assertEqual(output, output_regular)

        # 对输出结果进行求和并进行反向传播
        output.sum().backward()
        output_regular.sum().backward()

        # 使用优化器更新参数
        optim.step()
        optim_regular.step()

        # 设置随机种子
        torch.manual_seed(1)
        # 创建输入 tensor
        inp = torch.randn(20, 5, device=self.device_type)
        # 断言并行化模型和普通模型的输出结果相等
        self.assertEqual(model_tp(inp), model_regular_tp(inp))
    # 定义一个测试方法，用于测试 DTensor 类的 stride 方法
    def test_dtensor_stride(self):
        # 创建一个设备网格对象，包含给定的设备类型和世界大小的范围列表
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        # 定义一个仅包含第一个分片的分片规范列表
        shard0_spec = [Shard(0)]
        # 创建一个大小为 4x8 的随机张量
        local_tensor = torch.randn(4, 8)
        # 创建一个全局形状为 (self.world_size * 4, 8) 的 torch.Size 对象
        global_shape = torch.Size([self.world_size * 4, 8])
        # 从本地张量创建一个分布张量对象，使用给定的设备网格和分片规范
        dist_tensor = DTensor.from_local(local_tensor, device_mesh, shard0_spec)
        # 断言分布张量的步幅为 (8, 1)
        self.assertEqual(dist_tensor.stride(), (8, 1))

        # 定义一个仅包含第二个分片的分片规范列表
        shard1_spec = [Shard(1)]
        # 创建一个大小为 8x4 的随机张量
        local_tensor = torch.randn(8, 4)
        # 创建一个全局形状为 (8, self.world_size * 4) 的 torch.Size 对象
        global_shape = torch.Size([8, self.world_size * 4])
        # 从本地张量创建一个分布张量对象，使用给定的设备网格和分片规范
        dist_tensor = DTensor.from_local(local_tensor, device_mesh, shard1_spec)
        # 断言分布张量的步幅为 (4 * self.world_size, 1)
        self.assertEqual(dist_tensor.stride(), (4 * self.world_size, 1))

        # 如果从转置后的矩阵初始化
        # 创建一个大小为 8x4x8 的随机张量
        local_tensor = torch.randn(8, 4, 8)
        # 将本地张量进行维度置换，得到转置后的张量
        local_tensor_t = local_tensor.permute(1, 2, 0)
        # 创建一个全局形状为 (4, self.world_size * 8, 8) 的 torch.Size 对象
        global_shape = torch.Size([4, self.world_size * 8, 8])
        # 断言转置后张量的步幅为 (8, 1, 32)
        self.assertEqual(local_tensor_t.stride(), (8, 1, 32))
        # 从转置后的本地张量创建一个分布张量对象，使用给定的设备网格和分片规范
        dist_tensor = DTensor.from_local(local_tensor_t, device_mesh, shard1_spec)
        # 定义全局步幅为 (8 * self.world_size, 1, 32 * self.world_size)
        global_stride = (8 * self.world_size, 1, 32 * self.world_size)
        # 断言分布张量的步幅与全局步幅相等
        self.assertEqual(dist_tensor.stride(), global_stride)
    # 定义一个测试方法，用于测试从本地张量创建分布式张量的功能
    def test_from_local(self):
        # 创建设备网格对象，使用设备类型和全局大小初始化
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        # 定义一个放置方案列表，这里只包含一个 Shard(0) 对象
        placements = [Shard(0)]
        # 创建一个本地张量，形状为 (3, 3)，数据是随机生成的
        local_tensor = torch.randn(3, 3)
        # 使用 DTensor 类的 from_local 方法，将本地张量分布到设备网格上
        sharded_tensor = DTensor.from_local(local_tensor, device_mesh, placements)
        # 断言分布式张量的大小为 (全局大小 * 3, 3)
        self.assertEqual(sharded_tensor.size(), torch.Size([self.world_size * 3, 3]))

        # 定义一个复制规范列表，包含一个 Replicate() 对象
        replica_spec = [Replicate()]
        # 使用相同的本地张量创建另一个分布式张量，使用复制规范
        ddp_tensor = DTensor.from_local(local_tensor, device_mesh, replica_spec)
        # 断言分布式张量的大小与本地张量的大小相同
        self.assertEqual(ddp_tensor.size(), local_tensor.size())

        # 定义一个部分规范列表，包含一个 Partial() 对象
        partial_spec = [Partial()]
        # 使用相同的本地张量创建另一个分布式张量，使用部分规范
        partial_tensor = DTensor.from_local(local_tensor, device_mesh, partial_spec)
        # 断言分布式张量的大小与本地张量的大小相同
        self.assertEqual(partial_tensor.size(), local_tensor.size())

        # 在具有梯度的本地张量上进行测试，形状为 (3, 3)
        local_tensor_with_grad = torch.randn(3, 3, requires_grad=True)
        # 对本地张量进行一些操作
        local_tensor_temp = local_tensor_with_grad * 3
        # 使用非叶子本地张量创建分布式张量，预期分布式张量也是非叶子节点
        dist_tensor = DTensor.from_local(local_tensor_temp, device_mesh, placements)
        # 断言分布式张量不是叶子节点
        self.assertFalse(dist_tensor.is_leaf)
        # 对分布式张量进行一些随机操作
        output = dist_tensor * 3
        # 断言输出对象是 DTensor 类型的实例
        self.assertIsInstance(output, DTensor)
        # 直接在分布式张量上触发反向传播
        local_grad = torch.ones(3, 3)
        grad_output = DTensor.from_local(local_grad, device_mesh, placements)
        # 执行分布式张量上的反向传播
        output.backward(grad_output)
        # 检查梯度是否正确流回原始的 torch.Tensor
        self.assertIsNotNone(local_tensor_with_grad.grad)
        expected_grad = torch.ones(3, 3) * 9
        # 断言期望的梯度值与计算得到的梯度值相等
        self.assertEqual(local_tensor_with_grad.grad, expected_grad)

    @with_comms
    # 定义一个测试方法，用于测试不均匀分片的本地张量创建分布式张量的功能
    def test_from_local_uneven_sharding(self):
        # 初始化设备网格对象，使用设备类型和网格形状 (self.world_size,)
        mesh_shape = (self.world_size,)
        device_mesh = init_device_mesh(self.device_type, mesh_shape)

        # 定义一个不均匀维度大小，比 self.world_size 多一
        uneven_dim0_size = self.world_size + 1
        # 创建一个全局张量，形状为 (uneven_dim0_size, 2)，数据随机生成
        global_tensor = torch.randn(uneven_dim0_size, 2)
        # 创建一个 Shard(0) 对象，用于分片放置
        shard_placement = Shard(0)
        # 使用分片放置对象的 _split_tensor 方法，将全局张量分割成多个本地张量
        tensor_list, _ = shard_placement._split_tensor(
            global_tensor,
            device_mesh.size(mesh_dim=0),
            with_padding=False,
            contiguous=True,
        )

        # 使用 DTensor 类的 from_local 方法，将本地张量列表中的第 self.rank 个张量创建成分布式张量
        dtensor = DTensor.from_local(
            tensor_list[self.rank],
            device_mesh,
            (Shard(0),),
            shape=global_tensor.size(),
            stride=global_tensor.stride(),
        )

        # 断言分布式张量的大小与全局张量的大小相同
        self.assertEqual(dtensor.size(), global_tensor.size())
        # 断言分布式张量的步幅与全局张量的步幅相同
        self.assertEqual(dtensor.stride(), global_tensor.stride())
    # 定义测试函数，测试当分片不均匀时是否会引发错误
    def test_from_local_uneven_sharding_raise_error(self):
        # 设置网格形状，这里是一个元组，只有一个维度，维度大小为 self.world_size
        mesh_shape = (self.world_size,)
        # 使用给定设备类型和网格形状初始化设备网格
        device_mesh = init_device_mesh(self.device_type, mesh_shape)

        # 计算一个不均匀的第零维大小，比 self.world_size 大一
        uneven_dim0_size = self.world_size + 1
        # 创建一个大小为 (uneven_dim0_size, 2) 的随机张量
        global_tensor = torch.randn(uneven_dim0_size, 2)
        # 创建一个 Shard 对象，表示数据分片的位置
        shard_placement = Shard(0)
        # 将全局张量分片，返回分片后的张量列表和填充信息（这里用 _ 表示忽略）
        tensor_list, _ = shard_placement._split_tensor(
            global_tensor,
            device_mesh.size(mesh_dim=0),
            with_padding=False,
            contiguous=True,
        )

        # 断言捕获 RuntimeError 异常，并检查异常信息是否包含特定文本
        with self.assertRaisesRegex(
            RuntimeError, "Please pass both shape and stride at the same time."
        ):
            # 使用 DTensor 类的 from_local 方法，传入分片后的张量、设备网格、分片位置和全局张量的形状
            dtensor = DTensor.from_local(
                tensor_list[self.rank],
                device_mesh,
                (Shard(0),),
                shape=global_tensor.size(),
            )

        # 再次进行断言，捕获 RuntimeError 异常，并检查异常信息是否包含特定文本
        with self.assertRaisesRegex(
            RuntimeError, "Please pass both shape and stride at the same time."
        ):
            # 使用 DTensor 类的 from_local 方法，传入分片后的张量、设备网格、分片位置和全局张量的步长
            dtensor = DTensor.from_local(
                tensor_list[self.rank],
                device_mesh,
                (Shard(0),),
                stride=global_tensor.stride(),
            )

    # 使用装饰器 with_comms 标记的测试函数
    @with_comms
    # 测试当维度为负数时的情况
    def test_from_local_negative_dim(self):
        # 创建设备网格对象，包括设备类型和从 0 到 self.world_size 的设备列表
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        # 创建一个 Shard 列表，其中包含一个负一的 Shard 对象
        placements = [Shard(-1)]
        # 创建一个大小为 (3, 3) 的随机局部张量
        local_tensor = torch.randn(3, 3)
        # 使用 DTensor 类的 from_local 方法，传入局部张量、设备网格和分片位置列表
        # 返回一个分布式张量对象
        sharded_tensor = DTensor.from_local(local_tensor, device_mesh, placements)
        # 断言分布式张量的第一个分片的维度是否为 1
        self.assertEqual(sharded_tensor.placements[0].dim, 1)

    # 下面是另一个使用装饰器 with_comms 标记的测试函数
    # 定义一个测试方法，用于测试将分布式张量转换为本地张量的功能
    def test_to_local(self):
        # 创建设备网格对象，指定设备类型和世界大小的列表
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        # 指定分片的位置，这里只有一个分片 Shard(0)
        placements = (Shard(0),)
        # 创建一个带梯度的随机张量，设备类型由 self.device_type 指定
        local_tensor_with_grad = torch.randn(
            3, 3, device=self.device_type, requires_grad=True
        )
        # 定义分布式张量的形状
        dist_tensor_shape = torch.Size([self.world_size * 3, 3])
        # 创建分布式张量的规格对象 DTensorSpec
        spec = DTensorSpec(
            mesh=device_mesh,
            placements=placements,
            tensor_meta=TensorMeta(
                dist_tensor_shape,
                local_tensor_with_grad.stride(),
                local_tensor_with_grad.dtype,
            ),
        )
        # 使用 DTensor 类创建分布式张量对象
        sharded_tensor = DTensor(
            local_tensor_with_grad,
            spec,
            requires_grad=True,
        )
        # 断言分布式张量的大小与预期的 dist_tensor_shape 相同
        self.assertEqual(sharded_tensor.size(), dist_tensor_shape)
        # 断言分布式张量转换为本地张量后与原始带梯度张量相同
        self.assertEqual(sharded_tensor.to_local(), local_tensor_with_grad)

        # 在分布式张量上进行一些操作，例如乘以 3
        temp_st = sharded_tensor * 3

        # 在分布式张量的本地张量上进行一些操作，创建一个新的带梯度的本地张量
        new_tensor_with_grad = torch.randn(
            3, 3, device=self.device_type, requires_grad=True
        )
        # 将分布式张量转换为本地张量并与新张量相加
        res = temp_st.to_local() + new_tensor_with_grad
        # 直接在 torch.Tensor 上调用 backward，检查梯度是否正确传播到分布式张量
        res.sum().backward()
        # 断言分布式张量的梯度不为空
        self.assertIsNotNone(sharded_tensor.grad)

        # 断言分布式张量的梯度转换为本地张量后与预期的值相同
        self.assertEqual(sharded_tensor.grad.to_local(), torch.ones(3, 3) * 3)

        # 测试当梯度步幅与前向输入不同时的情况
        res = sharded_tensor.to_local()
        # 创建一个简单的模型和目标张量
        model = torch.nn.ReLU()
        # 注册一个钩子函数，用于修改梯度的步幅
        res.register_hook(lambda grad: grad.t())
        target = torch.randn(3, 3, device=self.device_type)
        mae_loss = torch.nn.L1Loss()
        # 计算模型的输出并计算损失
        output = mae_loss(model(res), target)
        # 尝试进行反向传播，预期会引发 RuntimeError
        try:
            output.backward()
        except RuntimeError:
            # 断言分布式张量的梯度步幅变化后为 [1, 3 * self.world_size]
            self.assertEqual(sharded_tensor.grad.stride(), [1, 3 * self.world_size])

        # 测试在无梯度情况下直接返回本地张量的情况
        with torch.no_grad():
            local_no_grad = sharded_tensor.to_local()
            # 使用断言验证本地无梯度张量与分布式张量的本地张量相同
            assert local_no_grad is sharded_tensor._local_tensor

    @with_comms
    @with_comms
    # 使用装饰器标记此测试函数需要启用通信模拟

    def test_to_local_grad_hint(self):
        # 创建设备网格对象，包括设备类型和世界大小的列表
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        # 设置分片位置为第一个分片
        placements = (Shard(0),)
        # 创建一个全局张量，所有元素为1，并需要梯度
        global_tensor = torch.ones(8, 3, requires_grad=True)

        # 将全局张量分布到设备网格上，并使用指定的分片位置
        sharded_dtensor = distribute_tensor(global_tensor, device_mesh, placements)
        # 创建一个通信调试模式对象
        comm_mode = CommDebugMode()

        # 进入通信调试模式
        with comm_mode:
            # 将分片后的张量重新分发到本地，使用复制策略，并指定梯度分片策略为部分
            local_out = sharded_dtensor.redistribute(placements=[Replicate()]).to_local(
                grad_placements=[Partial()]
            )
            # 对本地张量求反向传播
            local_out.backward(torch.ones_like(local_out))

        # 断言使用全聚合到张量的通信次数为1
        self.assertEqual(
            comm_mode.comm_counts[c10d_functional.all_gather_into_tensor], 1
        )
        # 断言使用张量的规约分散通信次数为1
        self.assertEqual(
            comm_mode.comm_counts[c10d_functional.reduce_scatter_tensor], 1
        )

        # 获取分片后的张量的梯度，并断言其值等于全局张量乘以世界大小
        replica_grad = sharded_dtensor.grad.full_tensor()
        self.assertEqual(replica_grad, global_tensor * self.world_size)

    @with_comms
    # 使用装饰器标记此测试函数需要启用通信模拟

    def test_full_tensor_sync(self):
        # 创建设备网格对象，包括设备类型和世界大小的列表
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        # 设置分片位置为第一个分片
        placements = (Shard(0),)
        # 创建一个全局张量，所有元素为1，并需要梯度
        global_tensor = torch.ones(8, 3, requires_grad=True)

        # 将全局张量分布到设备网格上，并使用指定的分片位置
        sharded_dtensor = distribute_tensor(global_tensor, device_mesh, placements)
        # 获取完整的分片后的张量
        full_out = sharded_dtensor.full_tensor()
        # 断言完整的分片后的张量不是异步集体张量对象
        self.assertFalse(isinstance(full_out, AsyncCollectiveTensor))
        # 断言完整的分片后的张量与全局张量相等
        self.assertEqual(full_out, global_tensor)

    @with_comms
    # 使用装饰器标记此测试函数需要启用通信模拟

    def test_full_tensor_grad_hint(self):
        # 创建设备网格对象，包括设备类型和世界大小的列表
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        # 设置分片位置为第一个分片
        placements = (Shard(0),)
        # 创建一个全局张量，所有元素为1，并需要梯度
        global_tensor = torch.ones(8, 3, requires_grad=True)

        # 将全局张量分布到设备网格上，并使用指定的分片位置
        sharded_dtensor = distribute_tensor(global_tensor, device_mesh, placements)
        # 获取完整的分片后的张量，并指定梯度分片策略为部分
        local_out = sharded_dtensor.full_tensor(grad_placements=[Partial()])
        # 对完整的分片后的张量求和，并对结果进行反向传播
        local_out.sum().backward()

        # 获取分片后的张量的梯度，并断言其值等于全局张量乘以世界大小
        replica_grad = sharded_dtensor.grad.full_tensor()
        self.assertEqual(replica_grad, global_tensor * self.world_size)
    # 定义测试方法 `test_dtensor_new_empty_strided`
    def test_dtensor_new_empty_strided(self):
        # 创建设备网格对象，使用给定设备类型和全局大小创建
        device_mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))
        # 创建本地张量，大小为 8x8，包含梯度信息，使用指定设备类型
        local_tensor = torch.randn(8, 8, requires_grad=True, device=self.device_type)
        # 将本地张量分布到设备网格上，仅包含 Shard(0) 的部分
        my_dtensor = distribute_tensor(local_tensor, device_mesh, [Shard(0)])
        # 使用 my_dtensor 创建一个新的空的分块张量，大小为 (8, 8)，步长为 (8, 1)，包含梯度信息
        new_strided_dtensor = my_dtensor.new_empty_strided(
            (8, 8), (8, 1), requires_grad=True
        )
        # 测试操作是否生成了新的分块张量，并且自动求导正常工作
        self.assertEqual(new_strided_dtensor.shape, my_dtensor.shape)
        # 对新张量求和并进行反向传播
        new_strided_dtensor.sum().backward()
        # 确保新张量的梯度不为空
        self.assertIsNotNone(new_strided_dtensor.grad)
        # 确保新张量的梯度类型为 DTensor
        self.assertIsInstance(new_strided_dtensor.grad, DTensor)

        # 测试带有分块的新空张量反向传播是否正常工作
        my_dtensor.to_local().sum().backward()
        local_tensor.sum().backward()
        # 确保 my_dtensor 和 new_strided_dtensor 的梯度相等
        self.assertEqual(my_dtensor.grad, new_strided_dtensor.grad)
        # 确保 my_dtensor 的分布与本地张量的梯度相等
        self.assertEqual(
            my_dtensor.grad.redistribute(placements=[Replicate()]).to_local(),
            local_tensor.grad,
        )

    @with_comms
    # 定义测试方法 `test_dtensor_async_output`
    def test_dtensor_async_output(self):
        # 测试 dtensor 操作的输出如果没有用于任何计算，则应该是 AsyncCollectiveTensor
        # 表示尚未同步集合操作
        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))

        # 定义一个函数 fn，接受一个 dtensor 对象并返回本地张量
        def fn(dt):
            # 对 dt 执行重分布操作，使用设备网格和 Replicate() 方案，异步操作设置为 True
            dt_out_redistribute = dt.redistribute(mesh, [Replicate()], async_op=True)
            # 确保尚未同步
            # TODO: figure out why this is returning None
            # self.assertTrue(_tensor_needs_wait(dt_out_redistribute))
            # 将重分布后的 dtensor 视图化为其形状
            dt_out_redistribute_view = dt_out_redistribute.view(
                dt_out_redistribute.shape
            )
            # 将视图化后的张量转换为本地张量
            local_tensor = dt_out_redistribute_view.to_local()
            return local_tensor

        # 创建一个大小为 (4, 2) 的张量 x，使用指定设备类型
        x = torch.ones((4, 2), device=self.device_type)
        # 将张量 x 分布到设备网格上，仅包含 Shard(0) 的部分
        dt = distribute_tensor(x, mesh, [Shard(0)])
        # 调用 fn 函数，传入 dt，得到输出 out
        out = fn(dt)
        # 确保输出类型为 AsyncCollectiveTensor，并且尚未完成同步
        self.assertEqual(type(out), AsyncCollectiveTensor)
        self.assertFalse(out.completed)
        # 对输出进行视图化，确保输出类型为 AsyncCollectiveTensor，并且尚未完成同步
        out_view = out.view(-1)

        # 确保输出类型为 AsyncCollectiveTensor，并且尚未完成同步
        self.assertEqual(type(out_view), AsyncCollectiveTensor)
        self.assertFalse(out.completed)

        # 使用数据进行计算，需要同步操作
        ref = torch.ones((4, 2), device=self.device_type) + 1
        ref = ref.view(-1)
        out_data = out_view + 1
        # 确保输出数据类型为 torch.Tensor，并且与参考值 ref 相等
        self.assertEqual(type(out_data), torch.Tensor)
        self.assertEqual(out_data, ref)

        # 测试 async_op 默认为 False 的情况
        sync_out = dt.redistribute(mesh, [Replicate()])
        # 确保 sync_out 不是 AsyncCollectiveTensor
        self.assertFalse(isinstance(sync_out, AsyncCollectiveTensor))
        # 确保 sync_out 转换为本地张量与原始张量 x 相等
        self.assertEqual(sync_out.to_local(), x)
    def test_from_local_then_to_local(self):
        # 确保从 torch.Tensor -> dist tensor -> torch.Tensor 的端到端操作正常工作的测试
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        placements = [Shard(0)]

        # step 1. 从本地构造带梯度的张量
        local_tensor_with_grad = torch.randn(
            3, 3, device=self.device_type, requires_grad=True
        )
        # 在本地张量上执行一些操作
        local_tensor_temp = local_tensor_with_grad + 8
        # step 2. 使用非叶子本地张量创建分布式张量，创建的分布式张量也应为非叶子节点
        dist_tensor = DTensor.from_local(local_tensor_temp, device_mesh, placements)
        self.assertFalse(dist_tensor.is_leaf)
        # 在分布式张量上进行一些随机操作
        output = dist_tensor * 6
        self.assertIsInstance(output, DTensor)

        # step 3. 在分布式张量的本地张量上执行一些操作
        new_tensor_with_grad = torch.randn(
            3, 3, device=self.device_type, requires_grad=True
        )
        res = output.to_local() + new_tensor_with_grad
        # 直接在 torch.Tensor 上调用 backward，检查是否通过将梯度传播回原始 torch.Tensor
        res.sum().backward()
        self.assertIsNotNone(local_tensor_with_grad.grad)

        expected_grad = torch.ones(3, 3) * 6
        self.assertEqual(local_tensor_with_grad.grad, expected_grad)

    @with_comms
    def test_dtensor_spec_read_only_after_set(self):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        placements = [Shard(0)]
        local_tensor = torch.randn(3, 3)
        sharded_tensor = DTensor.from_local(local_tensor, device_mesh, placements)

        # 修改 placements，但不应更改 dist_tensor 的规范
        placements[0] = Replicate()
        self.assertTrue(sharded_tensor.placements is not placements)
        self.assertNotEqual(sharded_tensor.placements, placements)

    @with_comms
    def test_dtensor_spec_hash(self):
        # 创建一个设备网格对象，使用指定的设备类型和全局大小范围
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        # 定义一个包含单个分片的放置列表
        placements = [Shard(0)]
        # 创建一个随机填充的本地张量
        local_tensor = torch.randn(3, 3)
        # 创建另一个随机填充的本地张量
        local_tensor2 = torch.randn(3, 3)
        # 使用本地张量和设备信息创建分布式张量对象
        sharded_tensor = DTensor.from_local(local_tensor, device_mesh, placements)
        # 使用第二个本地张量和相同的设备信息创建另一个分布式张量对象
        sharded_tensor2 = DTensor.from_local(local_tensor2, device_mesh, placements)
        
        # 断言：由于 DTensorSpec 没有真实的张量数据，所以哈希值相同，
        # 只要设备网格、放置方式和张量属性相同
        self.assertEqual(hash(sharded_tensor._spec), hash(sharded_tensor2._spec))

        # 修改放置方式会改变哈希值
        local_tensor3 = torch.ones(3, 3)
        # 创建一个包含复制副本的规范列表
        replica_spec = [Replicate()]
        # 使用新的本地张量、设备信息和复制规范创建分布式张量对象
        replica_tensor = DTensor.from_local(
            local_tensor3, device_mesh, replica_spec, run_check=False
        )
        # 断言：哈希值不相等
        self.assertNotEqual(hash(sharded_tensor._spec), hash(replica_tensor._spec))

    @with_comms
    def test_dtensor_properties(self):
        # 创建一个设备网格对象，使用指定的设备类型和全局大小范围
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        # 定义一个包含单个分片的放置列表
        placements = [Shard(0)]
        # 创建一个随机填充的本地张量
        local_tensor = torch.randn(3, 3)
        # 使用本地张量和设备信息创建分布式张量对象
        sharded_tensor = DTensor.from_local(local_tensor, device_mesh, placements)
        # 断言：验证张量的设备类型与预期的设备类型相同
        self.assertEqual(sharded_tensor.device.type, self.device_type)

    @with_comms
    def test_dtensor_save_load(self):
        import io

        # 创建设备网格对象
        device_mesh = self.build_device_mesh()
        # 定义一个包含单个分片的放置列表
        placements = [Shard(0)]
        # 创建一个随机填充的本地张量
        local_tensor = torch.randn(3, 3)
        # 使用本地张量和设备信息创建分布式张量对象
        sharded_tensor = DTensor.from_local(local_tensor, device_mesh, placements)
        
        # 创建一个字节流缓冲区
        buffer = io.BytesIO()
        # 将分布式张量对象保存到缓冲区
        torch.save(sharded_tensor, buffer)
        buffer.seek(0)
        # 重新加载保存的分布式张量对象
        reloaded_st = torch.load(buffer)
        # 断言：验证重新加载的张量与原始张量相等
        self.assertEqual(sharded_tensor, reloaded_st)
        
        # 测试仅加载权重的情况
        try:
            # 向安全全局变量中添加需要序列化的类
            torch.serialization.add_safe_globals(
                [DTensor, DeviceMesh, Shard, DTensorSpec, TensorMeta]
            )
            buffer.seek(0)
            # 使用仅加载权重的方式重新加载分布式张量对象
            reloaded_st = torch.load(buffer, weights_only=True)
            # 断言：验证重新加载的张量与原始张量相等
            self.assertEqual(sharded_tensor, reloaded_st)
        finally:
            # 清除安全全局变量
            torch.serialization.clear_safe_globals()
class DTensorMeshTest(DTensorTestBase):
    @property
    def world_size(self):
        return 8

    def sub_mesh_assert_equal(self, mesh, exp_in_mesh, exp_out_of_mesh, tensor):
        # 如果当前测试的维度在 mesh 中
        if self.rank in mesh:
            # 断言 tensor 等于预期在 mesh 中的值
            self.assertEqual(tensor, exp_in_mesh)
        else:
            # 断言 tensor 等于预期不在 mesh 中的值
            self.assertEqual(tensor, exp_out_of_mesh)

    @with_comms
    def test_dtensor_device_mesh_device_conversion(self):
        # 构建一个 CUDA 设备的 mesh
        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))

        # 使用 CPU 上的本地 tensor 构建带 CUDA 设备 mesh 的 dist tensor
        # 应自动将 dist tensor 转换为 CUDA
        placements = [Shard(0)]
        local_tensor = torch.randn(3, 3)
        dist_tensor = DTensor.from_local(local_tensor, mesh, placements)
        self.assertEqual(dist_tensor.device.type, self.device_type)
        self.assertEqual(dist_tensor.to_local().device.type, self.device_type)

    @with_comms
    def test_dtensor_api_device_mesh_context_manager(self):
        # 使用设备类型和范围构建一个设备 mesh 上下文管理器
        with DeviceMesh(self.device_type, list(range(self.world_size))) as mesh:
            placements = [Shard(0)]
            local_tensor = torch.randn(3, 3)
            # 使用本地 tensor 构建分布在 mesh 上的 sharded tensor
            sharded_tensor = DTensor.from_local(
                local_tensor, device_mesh=mesh, placements=placements
            )

        # 在一个设备类型和范围构建的设备 mesh 上下文管理器内
        with DeviceMesh(self.device_type, list(range(self.world_size))):
            placements = [Shard(0)]
            local_tensor = torch.randn(3, 3)
            # 使用本地 tensor 构建 sharded tensor，不指定 device_mesh
            sharded_tensor = DTensor.from_local(local_tensor, placements=placements)
            replica_spec = [Replicate()]
            # 重新分布 sharded tensor 到 replica 的位置
            replica_tensor = sharded_tensor.redistribute(placements=replica_spec)
            self.assertEqual(
                replica_tensor.size(), torch.Size([3 * self.world_size, 3])
            )

        # 使用设备类型和范围构建一个设备 mesh 上下文管理器
        with DeviceMesh(self.device_type, torch.arange(self.world_size)):
            placements = [Shard(0)]
            global_shape = torch.Size([3 * self.world_size, 3])
            global_tensor = torch.randn(global_shape)
            # 在给定的 placements 上分布全局 tensor
            sharded_tensor = distribute_tensor(global_tensor, placements=placements)
            self.assertEqual(sharded_tensor.to_local().shape, torch.Size([3, 3]))

            # 构建一个二维的设备 mesh
            mesh_2d = DeviceMesh(
                self.device_type, torch.arange(self.world_size).reshape(2, 4)
            )

            # 在二维设备 mesh 上下文中
            with mesh_2d:
                shard_2d_spec = [Shard(0), Replicate()]
                # 在给定的 placements 上分布全局 tensor
                tensor_2d = distribute_tensor(global_tensor, placements=shard_2d_spec)

                self.assertEqual(tensor_2d.to_local().shape, torch.Size([3 * 4, 3]))

            # 在指定的 placements 上重新分布全局 tensor
            sharded_after_2d = distribute_tensor(global_tensor, placements=placements)
            self.assertEqual(sharded_after_2d.to_local().shape, torch.Size([3, 3]))

    @with_comms
    # 定义一个测试方法，用于测试二维张量的设备网格分布
    def test_dtensor_2d_mesh(self):
        # 创建一个包含世界大小元素的张量，并将其重塑为 2x4 的形状
        mesh_tensor = torch.arange(self.world_size).reshape(2, 4)
        # 使用给定的设备类型和张量创建一个设备网格对象
        mesh = DeviceMesh(self.device_type, mesh_tensor)

        # 在二维设备网格上构建分布张量，并测试其是否正常工作
        placements = [Shard(0), Shard(1)]
        # 创建一个本地随机张量
        local_tensor = torch.randn(3, 3)
        # 根据本地张量、设备网格和分片位置创建分布张量
        dist_tensor = DTensor.from_local(local_tensor, mesh, placements)
        # 断言分布张量的尺寸是否正确
        self.assertEqual(
            dist_tensor.size(), torch.Size([3 * mesh.size(0), 3 * mesh.size(1)])
        )
        # 断言分布张量所在设备的类型是否正确
        self.assertEqual(dist_tensor.device.type, self.device_type)
        # 断言将分布张量转换为本地张量后，所在设备的类型是否正确
        self.assertEqual(dist_tensor.to_local().device.type, self.device_type)

        # 如果分片在同一张量维度上
        # 我们应该正确构建全局张量的尺寸
        shard_same_dim_spec = [Shard(0), Shard(0)]
        # 创建一个新的本地随机张量
        local_tensor = torch.randn(3, 3)
        # 根据本地张量、设备网格和相同维度分片规格创建分布张量
        dist_tensor = DTensor.from_local(local_tensor, mesh, shard_same_dim_spec)
        # 断言分布张量的尺寸是否正确
        self.assertEqual(dist_tensor.size(), torch.Size([3 * self.world_size, 3]))

    @with_comms
    # 定义一个使用通信功能装饰器的测试方法
    def test_device_mesh_nd(self):
        # 创建一个包含世界大小元素的张量，并将其重塑为 2x2x2 的形状
        mesh_tensor = torch.arange(self.world_size).reshape(2, 2, 2)
        # 使用给定的设备类型和张量创建一个设备网格对象
        mesh = DeviceMesh(self.device_type, mesh_tensor)
        # 在三维设备网格上构建分布张量，并测试其是否正常工作
        placements = [Shard(0), Shard(1), Shard(2)]
        # 创建一个本地随机张量
        local_tensor = torch.randn(3, 3, 3)
        # 根据本地张量、设备网格和分片位置创建分布张量
        dist_tensor = DTensor.from_local(local_tensor, mesh, placements)
        # 断言分布张量的尺寸是否正确
        self.assertEqual(dist_tensor.size(), torch.Size([6, 6, 6]))
        # 断言分布张量所在设备的类型是否正确
        self.assertEqual(dist_tensor.device.type, self.device_type)
        # 断言将分布张量转换为本地张量后，所在设备的类型是否正确
        self.assertEqual(dist_tensor.to_local().device.type, self.device_type)

        # 在三维设备网格上构建分布张量，并测试部分分片在相同维度上的情况
        placements = [Shard(0), Shard(0), Shard(2)]
        # 创建一个新的本地随机张量
        local_tensor = torch.randn(3, 3, 3)
        # 根据本地张量、设备网格和分片位置创建分布张量
        dist_tensor = DTensor.from_local(local_tensor, mesh, placements)
        # 断言分布张量的尺寸是否正确
        self.assertEqual(dist_tensor.size(), torch.Size([12, 3, 6]))
        # 断言分布张量所在设备的类型是否正确
        self.assertEqual(dist_tensor.device.type, self.device_type)
        # 断言将分布张量转换为本地张量后，所在设备的类型是否正确
        self.assertEqual(dist_tensor.to_local().device.type, self.device_type)
    # 测试函数：test_dtensor_spec_local_shard_offset
    def test_dtensor_spec_local_shard_offset(self):
        # 创建设备网格对象，使用 self.device_type 和一个由 0 到 world_size-1 组成的张量
        device_mesh = DeviceMesh(
            self.device_type, torch.arange(self.world_size).reshape(2, 4)
        )
        # 定义张量的形状
        tensor_shape = (3 * self.world_size, 3 * self.world_size)
        
        # 定义分片规格和对应的本地分片偏移
        shard_spec_and_offsets = [
            (
                [Shard(0), Replicate()],
                (3 * (self.world_size // 2) * (self.rank // 4), 0),
            ),
            (
                [Shard(1), Replicate()],
                (0, 3 * (self.world_size // 2) * (self.rank // 4)),
            ),
            (
                [Replicate(), Shard(0)],
                (3 * (self.world_size // 4) * (self.rank % 4), 0),
            ),
            (
                [Replicate(), Shard(1)],
                (0, 3 * (self.world_size // 4) * (self.rank % 4)),
            ),
        ]
        
        # 导入计算本地形状和全局偏移的函数
        from torch.distributed._tensor._utils import (
            compute_local_shape_and_global_offset,
        )
        
        # 创建随机张量
        logical_tensor = torch.randn(tensor_shape)
        
        # 遍历所有分片规格并检查本地分片偏移
        for placements, expected_shard_offsets in shard_spec_and_offsets:
            # 分发张量到设备网格上
            dtensor = distribute_tensor(logical_tensor, device_mesh, placements)
            # 计算本地形状和全局偏移
            _, offset = compute_local_shape_and_global_offset(
                dtensor.shape, device_mesh, dtensor.placements
            )
            # 断言本地分片偏移是否与期望值相等
            self.assertEqual(expected_shard_offsets, offset)

    # 测试函数：test_from_local_sub_mesh
    @with_comms
    def test_from_local_sub_mesh(self):
        # 创建设备网格对象，使用 self.device_type 和列表 [0, 2]
        mesh = DeviceMesh(self.device_type, [0, 2])
        # 创建本地张量，全为 1 的张量形状为 (3, 4)
        local_tensor = torch.ones(3, 4)
        
        # 在子网格中创建分布式张量
        dtensor = DTensor.from_local(local_tensor, mesh, [Shard(0)])
        # 断言分布式张量的大小为 [6, 4]
        self.assertEqual(dtensor.size(), torch.Size([6, 4]))
        
        # 使用辅助函数 sub_mesh_assert_equal 断言子网格的状态
        self.sub_mesh_assert_equal(
            mesh.mesh,
            torch.ones(3, 4),
            torch.tensor([]),
            dtensor.to_local(),
        )
        
        # 测试在子网格中创建的张量进行加法操作，只应用于子网格内的本地分片
        dtensor = dtensor + 2
        
        # 使用辅助函数 sub_mesh_assert_equal 再次断言子网格的状态
        self.sub_mesh_assert_equal(
            mesh.mesh,
            torch.ones(3, 4) + 2,
            torch.tensor([]),
            dtensor.to_local(),
        )
    def test_default_value_sub_mesh(self):
        mesh = DeviceMesh(self.device_type, [0, 2])

        # test scalar return value
        local_tensor1 = torch.ones(4, 3)
        local_tensor2 = torch.ones(4, 3)
        # 使用 DeviceMesh 和 Shard 对象创建 DTensor 对象，存储到 dtensor1 和 dtensor2
        dtensor1 = DTensor.from_local(local_tensor1, mesh, [Shard(0)])
        dtensor2 = DTensor.from_local(local_tensor2, mesh, [Shard(0)])
        # 调用 equal 方法比较 dtensor1 和 dtensor2 的相等性，返回本地结果
        local_res = dtensor1.equal(dtensor2)  # equal returns local result
        # 调用自定义断言函数检查子网格的相等性
        self.sub_mesh_assert_equal(
            mesh.mesh,
            True,
            True,
            local_res,
        )

        # test 0-d tensor return value
        local_tensor = torch.ones(4, 3)
        # 使用 DeviceMesh 和 Shard 对象创建 DTensor 对象，然后计算其和
        dtensor = DTensor.from_local(local_tensor, mesh, [Shard(0)]).sum()
        # 调用自定义断言函数检查子网格的相等性
        self.sub_mesh_assert_equal(
            mesh.mesh,
            torch.tensor(12.0),
            torch.tensor(0.0),
            dtensor.to_local(),
        )

        # test List[torch.Tensor] return value
        local_tensor = torch.ones(3, 4)
        # 使用 DeviceMesh 和 Shard 对象创建 DTensor 对象，并按指定维度拆分
        dtensor = DTensor.from_local(local_tensor, mesh, [Shard(0)])
        dtensor_list = dtensor.split([2, 2], dim=1)
        # 调用自定义断言函数检查子网格的相等性，对每个 DTensor 进行本地化处理
        self.sub_mesh_assert_equal(
            mesh.mesh,
            [torch.ones(3, 2)] * 2,
            [torch.tensor([])] * 2,
            [dt.to_local() for dt in dtensor_list],
        )

    @with_comms
    def test_redistribute_sub_mesh(self):
        mesh = DeviceMesh(self.device_type, [0, 2])

        # test redistribute on a submesh
        local_tensor1 = torch.ones(4, 3)
        # 使用 DeviceMesh 和 Shard 对象创建 DTensor 对象
        sharded_dtensor = DTensor.from_local(local_tensor1, mesh, [Shard(0)])
        # 将 DTensor 对象复制到多个位置
        replicated_dtensor = sharded_dtensor.redistribute(placements=[Replicate()])
        # 调用自定义断言函数检查子网格的相等性，将复制后的 DTensor 对象本地化处理
        self.sub_mesh_assert_equal(
            mesh.mesh, torch.ones(8, 3), torch.tensor([]), replicated_dtensor.to_local()
        )
        # 再次在指定的位置上分配 DTensor 对象
        sharded_again = replicated_dtensor.redistribute(placements=[Shard(0)])
        # 调用自定义断言函数检查子网格的相等性，将重新分配后的 DTensor 对象本地化处理
        self.sub_mesh_assert_equal(
            mesh.mesh, torch.ones(4, 3), torch.tensor([]), sharded_again.to_local()
        )

    @with_comms
    def test_implicit_replication(self):
        mesh = init_device_mesh(self.device_type, (self.world_size,))
        local_tensor1 = torch.ones(4, 3)
        # 使用 DeviceMesh 和 Shard 对象创建 DTensor 对象
        sharded_dtensor = DTensor.from_local(local_tensor1, mesh, [Shard(0)])

        from torch.distributed._tensor.experimental import implicit_replication

        with implicit_replication():
            # 将标量张量作为左操作数，测试当 args 列表中存在非 DTensor 时的情况
            out_dt = torch.ones(3, device=self.device_type) + sharded_dtensor
            # 使用自定义断言函数检查子网格的分布情况
            self.assertEqual(out_dt.placements, [Shard(0)])
            # 检查输出张量的形状
            self.assertEqual(out_dt.shape, (4 * self.world_size, 3))
            # 将分布在本地的数据进行本地化处理
            local_shard = out_dt.to_local()
            # 检查本地数据的形状和值
            self.assertEqual(local_shard.shape, (4, 3))
            self.assertEqual(local_shard, torch.ones(4, 3) + torch.ones(3))
    # 定义一个测试方法，用于测试自动隐式复制功能
    def test_auto_implicit_replication(self):
        # 初始化设备网格，根据设备类型和世界大小创建网格对象
        mesh = init_device_mesh(self.device_type, (self.world_size,))

        # 创建一个本地张量，每个设备上都是全1的张量
        local_tensor = torch.ones(self.world_size, 3, device=self.device_type)
        
        # 使用本地张量创建一个分布式张量，仅在指定分片上存在
        sharded_dtensor = DTensor.from_local(local_tensor, mesh, [Shard(0)])

        # 创建一个标量张量，仅包含一个元素，根据设备类型存储
        ndim_0_tensor = torch.tensor(1, device=self.device_type)

        # 定义一个函数，将标量张量与分布式张量相加
        def add_scalar_tensor_with_dtensor():
            return ndim_0_tensor + sharded_dtensor

        # 将相加后的结果转换回本地张量，并与预期的本地张量加上标量张量进行比较
        result = add_scalar_tensor_with_dtensor().to_local()
        self.assertEqual(result, local_tensor + ndim_0_tensor)
        
        # 确保不会产生警告，因为找到了一个元素数量为1且维度不为0的非标量张量
        self.assertNotWarn(
            add_scalar_tensor_with_dtensor,
            "Found a non-scalar tensor with numel=1 and ndim!=0",
        )

        # 创建一个包含一个元素的张量，根据设备类型存储
        numel_1_tensor = torch.tensor([1], device=self.device_type)
        
        # 将一个包含一个元素的张量与分布式张量相加，然后转换回本地张量，并与预期的本地张量加上numel_1_tensor进行比较
        self.assertEqual(
            (numel_1_tensor + sharded_dtensor).to_local(), numel_1_tensor + local_tensor
        )
class TestDTensorPlacementTypes(DTensorTestBase):
    @property
    def world_size(self):
        return 8  # 返回一个整数，表示测试用例的并行运行环境中的设备数量

    def _create_tensor(self, size):
        # 保持所有操作的确定性，使用固定的随机种子
        torch.manual_seed(0)
        tensor = torch.rand(size)  # 创建一个指定大小的随机张量
        if self.device_type == "cuda":
            return tensor.cuda()  # 如果设备类型为cuda，返回在GPU上的张量
        else:
            return tensor  # 否则返回在CPU上的张量

    @with_comms
    def test_split_tensor_1D(self) -> None:
        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))  # 创建一个设备网格对象
        shard_placement = Shard(0)  # 创建一个分片位置对象

        for size in range(8):
            tensor = self._create_tensor(size)  # 创建一个指定大小的张量
            splitted_tensor_list, pad_sizes = shard_placement._split_tensor(
                tensor,
                mesh.size(),
                with_padding=True,
                contiguous=True,
            )
            if size == 0:
                # 当张量大小为0时，所有排名不需要填充
                expected_pad_sizes = []
                assert_array_equal(expected_pad_sizes, pad_sizes)

                is_tensor_empty = [
                    False if splitted_tensor.numel() > 0 else True
                    for splitted_tensor in splitted_tensor_list
                ]
                expected_is_tensor_empty = [True] * self.world_size
                assert_array_equal(expected_is_tensor_empty, is_tensor_empty)
            else:
                expected_pad_sizes = [
                    0 if idx < size else 1
                    for idx, _ in enumerate(range(self.world_size))
                ]
                assert_array_equal(expected_pad_sizes, pad_sizes)

                from torch.distributed._tensor._collective_utils import unpad_tensor

                # 对分割后的张量进行去填充处理
                unpadded_list = [
                    unpad_tensor(tensor, shard_placement.dim, pad_sizes[i])
                    if pad_sizes[i] > 0
                    else tensor
                    for i, tensor in enumerate(splitted_tensor_list)
                ]
                expected_is_tensor_empty = [
                    False if idx < size else True
                    for idx, _ in enumerate(range(self.world_size))
                ]
                is_tensor_empty = [
                    False if unpadded_tensor.numel() > 0 else True
                    for unpadded_tensor in unpadded_list
                ]
                assert_array_equal(expected_is_tensor_empty, is_tensor_empty)


if __name__ == "__main__":
    run_tests()  # 执行测试用例
```