# `.\pytorch\test\distributed\_tensor\test_redistribute.py`

```py
    @property
    def world_size(self):
        # 返回集群的大小，这里设定为4
        return 4

    @with_comms
    def test_shard_to_replicate_forward_backward(self):
        # 1) test shard -> replicate forward
        # 创建一个包含所有设备的设备网格，使用当前设备类型和集群中的设备范围
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        # 定义复制（replicate）的放置规格
        replica_spec = [Replicate()]

        # 定义多个输入大小和分片维度的组合
        input_sizes_and_shard_dim = [
            ((self.world_size * 3, 3), 0),          # 第一个维度分片，总大小是集群大小的3倍
            ((self.world_size * 3 + 1, 3), 0),      # 同上，但大小增加1
            ((self.world_size * 3 + 2, 3), 0),      # 同上，但大小增加2
            ((3, self.world_size * 3), 1),          # 第二个维度分片，总大小是集群大小的3倍
            ((3, self.world_size * 3 + 1), 1),      # 同上，但大小增加1
            ((3, self.world_size * 3 + 2), 1),      # 同上，但大小增加2
        ]

        # 创建通信调试模式实例
        comm_mode = CommDebugMode()
        for input_size, shard_dim in input_sizes_and_shard_dim:
            # 根据输入大小和分片维度定义分片（shard）放置规格
            shard_spec = [Shard(shard_dim)]
            # 创建具有指定大小和设备类型的随机张量，并启用梯度计算
            expected_tensor = torch.randn(
                input_size, device=self.device_type, requires_grad=True
            )
            # 将张量分布到设备网格上，按照给定的分片和放置规格
            dtensor = distribute_tensor(expected_tensor, device_mesh, shard_spec)
            # 使用通信模式进行操作
            with comm_mode:
                # 重新分布张量到新的放置规格（replicate_spec）
                reshard_dtensor = dtensor.redistribute(device_mesh, replica_spec)
            # 断言重新分布后张量的大小与预期相符
            self.assertEqual(reshard_dtensor.size(), torch.Size(input_size))
            # 断言重新分布后的张量内容与原始张量的本地表示相同
            self.assertEqual(expected_tensor, reshard_dtensor.to_local())
            # 断言使用的通信操作计数为1次
            self.assertEqual(
                comm_mode.get_comm_counts()[funcol.all_gather_into_tensor], 1
            )

            # 2) test shard -> replicate backward:
            # 应该将梯度作为分片（shard）返回
            grad_output = torch.ones_like(reshard_dtensor)
            # 使用通信模式进行操作
            with comm_mode:
                # 执行反向传播，计算梯度
                reshard_dtensor.backward(grad_output)
            # 获取梯度输入
            grad_input = dtensor.grad
            # 断言梯度的放置规格与分片规格相同
            self.assertEqual(grad_input.placements, shard_spec)
            # 断言梯度的本地表示与张量本地表示形状相同
            self.assertEqual(
                grad_input.to_local(), torch.ones(dtensor.to_local().size())
            )
            # 断言通信操作的总计数为0
            self.assertEqual(comm_mode.get_total_counts(), 0)
    # 定义测试方法，用于验证复制到复制的前向和反向传播
    def test_replicate_to_replicate_forward_backward(self):
        # 创建设备网格对象，包含设备类型和世界大小的列表
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        # 指定复制规格为单一复制
        replica_spec = [Replicate()]
        # 创建在指定设备上的随机张量，需要梯度计算
        local_tensor = torch.randn(12, 3, device=self.device_type, requires_grad=True)

        # 创建通信调试模式对象
        comm_mode = CommDebugMode()

        # 1) 测试复制 -> 复制前向传播
        # 将本地张量分发到设备网格上的复制副本张量
        replica_tensor = distribute_tensor(local_tensor, device_mesh, replica_spec)
        # 启用通信模式
        with comm_mode:
            # 使用设备网格和复制规格重新分配复制副本张量
            reshard_replica_tensor = replica_tensor.redistribute(
                device_mesh, replica_spec
            )
        # 断言复制副本张量的大小与本地张量相同
        self.assertEqual(replica_tensor.size(), local_tensor.size())
        # 断言复制副本张量与重新分配后的副本张量相等
        self.assertEqual(replica_tensor, reshard_replica_tensor)
        # 断言通信模式的总计数为0
        self.assertEqual(comm_mode.get_total_counts(), 0)

        # 2) 测试复制 -> 复制反向传播:
        # 应该得到复制形式的梯度
        # 创建与重新分配后的复制副本张量相同大小的梯度输出张量
        grad_output = torch.ones_like(reshard_replica_tensor)
        # 启用通信模式
        with comm_mode:
            # 对重新分配后的复制副本张量执行反向传播
            reshard_replica_tensor.backward(grad_output)
        # 获取复制副本张量的梯度输入
        grad_input = replica_tensor.grad
        # 断言梯度输入的放置方式与复制规格相同
        self.assertEqual(grad_input.placements, replica_spec)
        # 断言梯度输入的本地值为全1张量
        self.assertEqual(grad_input.to_local(), torch.ones(12, 3))
        # 断言通信模式的总计数为0
        self.assertEqual(comm_mode.get_total_counts(), 0)

    @with_comms
    # 定义带有通信装饰器的测试方法
    def test_replicate_to_local_partial_grad(self):
        # 创建设备网格对象，包含设备类型和世界大小的列表
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        # 指定复制规格为单一复制
        replica_spec = [Replicate()]
        # 创建在指定设备上的随机张量，需要梯度计算
        local_tensor = torch.randn(12, 3, device=self.device_type, requires_grad=True)

        # 将本地张量分发到设备网格上的复制副本张量
        replica_tensor = distribute_tensor(local_tensor, device_mesh, replica_spec)

        # 创建通信调试模式对象
        comm_mode = CommDebugMode()

        # 启用通信模式
        with comm_mode:
            # 使用复制规格为复制的重新分配方法，将张量转换为本地梯度
            out = replica_tensor.redistribute(placements=[Replicate()]).to_local(
                grad_placements=[Partial()]
            )
            # 对输出执行反向传播，使用全1张量作为梯度输出
            out.backward(torch.ones_like(out))

        # 断言通信模式的总计数为1
        self.assertEqual(comm_mode.get_total_counts(), 1)
        # 断言通信模式中函数all_reduce的通信计数为1
        self.assertEqual(comm_mode.get_comm_counts()[funcol.all_reduce], 1)
    # 定义一个测试方法，用于测试在分片前后的复制操作
    def test_replicate_to_shard_forward_backward(self):
        # 创建设备网格对象，指定设备类型和全局设备列表
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        # 定义复制规范列表，这里只包含一个复制对象
        replica_spec = [Replicate()]

        # 定义输入大小和分片维度的组合列表
        input_sizes_and_shard_dim = [
            ((self.world_size * 3, 3), 0),
            ((self.world_size * 3 + 1, 3), 0),
            ((self.world_size * 3 + 2, 3), 0),
            ((3, self.world_size * 3), 1),
            ((3, self.world_size * 3 + 1), 1),
            ((3, self.world_size * 3 + 2), 1),
        ]

        # 创建通信调试模式对象
        comm_mode = CommDebugMode()

        # 遍历输入大小和分片维度的组合
        for input_size, shard_dim in input_sizes_and_shard_dim:
            # 根据输入大小和设备类型生成一个随机张量，并标记为需要梯度计算
            local_replica = torch.randn(
                input_size, device=self.device_type, requires_grad=True
            )
            # 按照指定的分片维度将本地复制的张量分块
            splitted_list = list(
                torch.chunk(local_replica, self.world_size, dim=shard_dim)
            )
            
            # 将本地张量作为对应分块列表的元素
            local_tensor = splitted_list[self.rank]

            # 将本地复制的张量在设备网格上进行分发，使用复制规范列表
            replica_tensor = distribute_tensor(local_replica, device_mesh, replica_spec)
            
            # 在通信调试模式下，对复制后的张量进行重新分片操作
            with comm_mode:
                reshard_tensor = replica_tensor.redistribute(device_mesh, shard_spec)
            
            # 断言重新分片后张量的大小与复制前一致
            self.assertEqual(reshard_tensor.size(), replica_tensor.size())
            # 断言重新分片后张量的部署规范与分片规范一致
            self.assertEqual(reshard_tensor.placements, shard_spec)
            # 断言重新分片后的本地张量与本地复制的张量一致
            self.assertEqual(reshard_tensor.to_local(), local_tensor)
            # 断言通信调试模式的总计数为0
            self.assertEqual(comm_mode.get_total_counts(), 0)

            # 测试复制 -> 分片反向传播:
            # 应该给出与复制相同的梯度
            grad_output = torch.ones_like(reshard_tensor)
            with comm_mode:
                reshard_tensor.backward(grad_output)
            
            # 获取复制张量的梯度
            grad_input = replica_tensor.grad
            
            # 断言复制张量梯度的部署规范与复制规范一致
            self.assertEqual(grad_input.placements, replica_spec)
            # 断言复制张量梯度的本地值为全1张量
            self.assertEqual(grad_input.to_local(), torch.ones(input_size))
            # 断言通信调试模式的总计数为1
            self.assertEqual(comm_mode.get_total_counts(), 1)
            # 断言在通信调试模式中，all_gather_into_tensor的通信计数为1
            self.assertEqual(
                comm_mode.get_comm_counts()[funcol.all_gather_into_tensor], 1
            )
    # 定义一个测试方法，用于验证部分到复制的前向和后向传播是否正常工作
    def test_partial_to_replicate_forward_backward(self):
        # 尽管我们不允许用户重新分片以产生部分放置（即用户不能将分片重新分片为部分），但我们允许内部从复制到部分的转换，
        # 并且反向的部分到复制应该按预期工作

        # 创建设备网格对象，使用给定设备类型和整数列表作为参数
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))

        # 创建一个张量，大小为(12, 3)，所有元素为1，位于指定设备上，并且需要梯度
        partial_local = torch.ones(12, 3, device=self.device_type, requires_grad=True)

        # 创建部分放置的规范列表，其中只有一个部分对象
        partial_spec = [Partial()]

        # 创建复制的规范列表，其中只有一个复制对象
        replica_spec = [Replicate()]

        # 创建通信调试模式对象
        comm_mode = CommDebugMode()

        # 测试从部分到复制的转换，这会触发全局归约操作
        # 从本地张量创建分布式张量对象，使用设备网格和部分放置规范作为参数
        partial_tensor = DTensor.from_local(partial_local, device_mesh, partial_spec)

        # 进入通信模式，执行从部分到复制的重分布操作
        with comm_mode:
            global_partial_tensor = partial_tensor.redistribute(
                device_mesh, replica_spec
            )

        # 断言部分张量的大小与本地张量的大小相同
        self.assertEqual(partial_tensor.size(), partial_local.size())

        # 断言全局部分张量与本地部分张量乘以世界大小后的结果相同
        self.assertEqual(
            partial_local * self.world_size, global_partial_tensor.to_local()
        )

        # 断言通信模式中执行全归约操作的次数为1
        self.assertEqual(comm_mode.get_comm_counts()[funcol.all_reduce], 1)

        # 测试反向传播，确保部分张量的梯度是复制梯度
        # 对于从本地张量的反向传播，我们希望复制() -> 部分() 被正常传递
        with comm_mode:
            global_partial_tensor.backward(torch.ones_like(global_partial_tensor))

        # 断言部分本地张量的梯度不为空
        self.assertIsNotNone(partial_local.grad)

        # 断言部分本地张量的梯度大小与本地张量的大小相同
        self.assertEqual(partial_local.grad.size(), partial_local.size())

        # 断言部分本地张量的梯度值等于所有元素为1的张量
        self.assertEqual(partial_local.grad, torch.ones_like(partial_local))

        # 断言通信模式中总计执行的通信次数为0
        self.assertEqual(comm_mode.get_total_counts(), 0)

    # 与通信相关的装饰器函数，用于标记测试方法使用通信功能
    @with_comms
    def test_replicate_to_partial(self):
        # 创建一个设备网格对象，包含指定设备类型和全球大小的列表
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        # 在指定设备上生成一个带有梯度信息的随机张量
        local_tensor = torch.randn(12, 3, device=self.device_type, requires_grad=True)
        # 创建一个 Partial 规范对象
        partial_spec = Partial()
        # 创建一个 Replicate 规范对象
        replica_spec = Replicate()

        # 1) 测试从复制到部分的前向传播
        # 将本地张量分发到设备网格上，使用复制规范进行复制
        replica_tensor = distribute_tensor(local_tensor, device_mesh, [replica_spec])
        # 使用断言检查是否抛出了预期的运行时错误，错误信息为 "Can not redistribute to Partial"
        with self.assertRaisesRegex(RuntimeError, "Can not redistribute to Partial"):
            # 尝试将复制的张量重新分布到部分规范
            partial_tensor = replica_tensor.redistribute(device_mesh, [partial_spec])

        # 导入 Redistribute 类
        from torch.distributed._tensor._redistribute import Redistribute

        # 创建通信调试模式对象
        comm_mode = CommDebugMode()

        # 在通信调试模式下执行以下代码块
        with comm_mode:
            # 使用 Redistribute.apply 方法将复制的张量按部分规范重新分布
            partial_tensor = Redistribute.apply(
                replica_tensor, device_mesh, [partial_spec]
            )
        # 使用断言验证部分张量的尺寸与本地张量的尺寸相同
        self.assertEqual(partial_tensor.size(), local_tensor.size())
        # 验证成功将其他进程上的内容置零
        self.assertEqual(
            replica_tensor.to_local() / self.world_size, partial_tensor.to_local()
        )
        # 验证通信模式的总计数为 0
        self.assertEqual(comm_mode.get_total_counts(), 0)

        # 在子组上进行复制到部分的操作
        # 重新生成本地张量
        local_tensor = torch.randn(12, 3, device=self.device_type)
        # 创建一个设备网格对象，包含设备类型和重新排列为二维网格的世界大小张量
        device_mesh = DeviceMesh(
            self.device_type,
            torch.arange(self.world_size).reshape(self.world_size // 2, 2),
        )
        # 1) 测试在二维网格子组上的复制到部分
        # 将本地张量分发到设备网格上，使用两个复制规范进行复制
        replica_tensor = distribute_tensor(
            local_tensor, device_mesh, [replica_spec, replica_spec]
        )
        # 在通信调试模式下执行以下代码块
        with comm_mode:
            # 使用 Redistribute.apply 方法将复制的张量按两个部分规范重新分布
            partial_tensor = Redistribute.apply(
                replica_tensor, device_mesh, [partial_spec, partial_spec]
            )
        # 使用断言验证部分张量的尺寸与本地张量的尺寸相同
        self.assertEqual(partial_tensor.size(), local_tensor.size())

        # 验证成功将其他进程上的内容置零
        self.assertEqual(
            replica_tensor.to_local() / self.world_size,
            partial_tensor.to_local(),
        )
        # 验证通信模式的总计数为 0
        self.assertEqual(comm_mode.get_total_counts(), 0)
    # 定义一个测试方法，用于测试部分数据到分片的转换
    def test_partial_to_shard(self):
        # 创建设备网格对象，使用给定设备类型和整数列表作为设备 ID
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        # 创建一个包含一个空部分对象的部分规范列表
        partial_spec = [Partial()]
        # 获取当前进程在设备网格中的排名
        my_rank = device_mesh.get_rank()

        # 定义输入尺寸和分片维度的列表
        input_sizes_and_shard_dim = [
            ((self.world_size * 3, 3), 0),
            ((self.world_size * 3 + 1, 3), 0),
            ((self.world_size * 3 + 2, 3), 0),
            ((3, self.world_size * 3), 1),
            ((3, self.world_size * 3 + 1), 1),
            ((3, self.world_size * 3 + 2), 1),
        ]

        # 创建调试通信模式对象
        comm_mode = CommDebugMode()

        # 遍历输入尺寸和分片维度的列表
        for input_size, shard_dim in input_sizes_and_shard_dim:
            # 创建全为1的局部张量，并指定设备类型
            partial_local = torch.ones(input_size, device=self.device_type)
            # 使用本地创建的局部张量、设备网格和部分规范，创建分布式张量
            partial_tensor = DTensor.from_local(
                partial_local, device_mesh, partial_spec, run_check=False
            )

            # 计算全块大小，用于计算每个分片的大小
            full_chunk_size = (
                input_size[shard_dim] + self.world_size - 1
            ) // self.world_size
            # 计算每个分片的大小列表
            chunk_sizes = [
                max(
                    min(input_size[shard_dim], full_chunk_size * (idx + 1))
                    - full_chunk_size * idx,
                    0,
                )
                for idx in range(self.world_size)
            ]

            # 复制输入尺寸，然后替换指定分片维度的大小为当前进程的分片大小
            local_shape = list(input_size)
            local_shape[shard_dim] = chunk_sizes[my_rank]

            # 测试部分到分片的转换，触发 reduce_scatter 操作
            with comm_mode:
                scatter_shard_tensor = partial_tensor.redistribute(
                    device_mesh, shard_spec
                )

            # 断言分布式张量的大小与局部张量的大小相同
            self.assertEqual(scatter_shard_tensor.size(), partial_tensor.size())
            # 断言分布式张量的放置规范与分片规范相同
            self.assertEqual(scatter_shard_tensor.placements, shard_spec)
            # 断言分布式张量转换为本地张量后的值为全为世界大小的张量
            self.assertEqual(
                scatter_shard_tensor.to_local(),
                torch.ones(local_shape) * self.world_size,
            )
            # 断言通信模式对象的通信计数中 reduce_scatter_tensor 函数调用次数为1
            self.assertEqual(
                comm_mode.get_comm_counts()[funcol.reduce_scatter_tensor], 1
            )

    # 使用 comms 装饰器标记的测试方法
    @with_comms
    # 定义一个测试方法，用于测试负分片维度的分布操作
    def test_redistribute_negative_shard_dim(self):
        # 创建设备网格对象，使用给定设备类型和整数列表作为设备 ID
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        # 创建一个形状为 (12, 3) 的本地张量，元素为随机数，并启用梯度计算
        local_tensor = torch.randn(12, 3, device=self.device_type, requires_grad=True)
        # 创建一个包含一个分片对象的分片规范列表
        shard_spec = [Shard(1)]
        # 创建一个包含一个负分片维度的分片规范列表
        shard_minus_spec = [Shard(-1)]

        # 将本地张量分布到设备网格上，使用给定的分片规范
        shard_tensor = distribute_tensor(local_tensor, device_mesh, shard_spec)
        # 断言分布式张量的放置规范的第一个维度为1
        self.assertEqual(shard_tensor.placements[0].dim, 1)
        # 将分布式张量重新分布到设备网格上，使用负分片维度的分片规范
        reshard_tensor = shard_tensor.redistribute(device_mesh, shard_minus_spec)
        # 断言分布式张量的放置规范的第一个维度仍为1
        self.assertEqual(shard_tensor.placements[0].dim, 1)
    # 定义测试方法，用于测试不均匀分片的情况
    def test_redistribute_uneven_sharding(self):
        # 创建一个设备网格对象，使用给定设备类型和二维排列的设备索引
        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size).reshape(2, 2))
        
        # 准备待测试的数据列表
        data_to_test = [
            # 最后一个维度不均匀
            torch.randn((10, 5), device=self.device_type),
            # 两个维度都不均匀
            torch.randn((9, 5), device=self.device_type),
            # 小于网格维度的形状
            torch.randn((3, 5), device=self.device_type),
            torch.randn((1, 3), device=self.device_type),
        ]

        # 准备不同的分片组合列表
        sharding_to_tests = [
            [Shard(0), Shard(0)],
            [Shard(0), Shard(1)],
        ]

        # 遍历每个输入张量
        for input_tensor in data_to_test:
            # 遍历每种分片组合
            for placements in sharding_to_tests:
                # 对输入张量进行分布式分片操作
                dt = distribute_tensor(input_tensor, mesh, placements)
                # 获取分布式张量的完整张量表示
                dt_full_tensor = dt.full_tensor()
                # 断言分布式张量的完整张量表示与原始输入张量相等
                self.assertEqual(dt_full_tensor, input_tensor)

    @with_comms
class MultiDimRedistributeTest(DTensorTestBase):
    # 定义一个测试类，继承自DTensorTestBase

    @property
    def world_size(self) -> int:
        # 返回并发运行的设备数量，这里固定为8
        return 8

    @with_comms
    def test_multi_dim_mesh(self):
        # 定义测试方法，测试多维网格分布

        devices = torch.arange(self.world_size)
        # 创建一个包含self.world_size个设备索引的张量

        for mesh_shape in [devices, devices.view(4, 2), devices.view(2, 2, 2)]:
            # 遍历不同的网格形状: [一维网格, 二维网格(4x2), 三维网格(2x2x2)]
            
            mesh_shape = torch.arange(self.world_size).view(-1, 2)
            # 重新定义mesh_shape为一个包含两列的张量

            device_mesh = DeviceMesh(self.device_type, mesh_shape)
            # 创建一个DeviceMesh对象，指定设备类型和网格形状

            tensor_shape = (16, 24)
            # 定义张量的形状为(16, 24)

            if torch.distributed.get_rank() == 0:
                # 如果当前进程的分布式排名为0

                full_tensor = torch.randn(*tensor_shape)
                # 创建一个随机填充的张量full_tensor
            else:
                # 否则（排名不为0）

                # these should be entirely ignored
                # because distribute_tensor is expected to override shards in ranks != 0
                full_tensor = torch.ones(*tensor_shape)
                # 创建一个全为1的张量full_tensor，但在实际运行中应完全忽略

            possibilities = [Replicate()] + [Shard(i) for i in range(full_tensor.ndim)]
            # 创建一个可能性列表，包含Replicate()和每个维度索引的Shard(i)

            all_outputs = list(itertools.product(*(mesh_shape.ndim * [possibilities])))
            # 生成所有输出组合，每个维度对应一个可能性列表的笛卡尔积

            all_inputs = list(
                itertools.product(*(mesh_shape.ndim * [possibilities + [Partial()]]))
            )
            # 生成所有输入组合，每个维度对应一个可能性列表与Partial()的并集的笛卡尔积

            for inputs in all_inputs:
                # 遍历所有输入组合

                # if partial, temporarily make it Replicated, then replace replicated with partial afterwards
                repl_inputs = [Replicate() if s.is_partial() else s for s in inputs]
                # 如果输入是Partial()，则暂时将其替换为Replicate()，然后在后续替换回Partial()

                dt = distribute_tensor(full_tensor, device_mesh, repl_inputs)
                # 使用distribute_tensor函数将full_tensor分发到device_mesh上，并使用repl_inputs控制分发方式

                if repl_inputs != inputs:
                    # 如果repl_inputs与inputs不同

                    # create a new DTensor reinterpreting some of the replicated entires as "Partial"
                    dt = DTensor.from_local(
                        dt.to_local(), device_mesh, inputs, run_check=False
                    )
                    # 创建一个新的DTensor对象，重新解释部分Replicated为"Partial"

                for outputs in all_outputs:
                    # 遍历所有输出组合

                    dt2 = dt.redistribute(device_mesh, outputs)
                    # 使用dt.redistribute方法在目标输出上重新分发

                    local_full = dt2.full_tensor()
                    # 获取重新分发后的张量的本地副本

                    if torch.distributed.get_rank() == 0:
                        # 如果当前进程的分布式排名为0

                        self.assertEqual(local_full.shape, full_tensor.shape)
                        # 断言本地副本的形状与原始张量的形状相同

                        num_sums = 1
                        for idx, input in enumerate(inputs):
                            if input.is_partial():
                                num_sums *= mesh_shape.size(idx)
                        expected = num_sums * full_tensor
                        # 计算预期的结果，根据Partial()输入的数目乘以原始张量

                        self.assertEqual(local_full, expected)
                        # 断言本地副本与预期结果相等


if __name__ == "__main__":
    run_tests()
    # 运行所有测试
```