# `.\pytorch\test\distributed\_tensor\test_api.py`

```
    @with_comms
    def test_distribute_tensor(self):
        # 创建一个设备网格对象，指定设备类型和参与的设备列表
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        # 定义分片规范，这里只包含一个分片对象
        shard_spec = [Shard(0)]

        # 遍历 requires_grad 取值为 True 和 False 的情况
        for requires_grad in [True, False]:
            # 创建一个形状为 (3 * self.world_size, 3) 的随机张量
            tensor_to_shard = torch.randn(
                3 * self.world_size, 3, requires_grad=requires_grad
            )
            # 将张量分发到设备网格上，并按照分片规范进行分片
            dist_tensor = distribute_tensor(tensor_to_shard, device_mesh, shard_spec)
            # 断言分发后张量的形状符合预期
            self.assertEqual(dist_tensor.size(), torch.Size([3 * self.world_size, 3]))
            # 将分发后的张量转回本地张量
            local_tensor = dist_tensor.to_local()
            # 断言本地张量的形状符合预期
            self.assertEqual(local_tensor.size(), torch.Size([3, 3]))
            # 如果 requires_grad 为 True，则断言分发后张量需要梯度并且是叶子节点
            if requires_grad:
                self.assertTrue(dist_tensor.requires_grad)
                self.assertTrue(dist_tensor.is_leaf)

        # 测试负数维度情况
        shard_minus_spec = [Shard(-1)]
        # 创建一个形状为 (3, 3 * self.world_size) 的随机张量
        tensor_to_shard = torch.randn(3, 3 * self.world_size)
        # 将张量分发到设备网格上，按照负数维度的分片规范进行分片
        dist_tensor = distribute_tensor(tensor_to_shard, device_mesh, shard_minus_spec)
        # 断言分发后张量的第一个分片的维度为 1
        self.assertEqual(dist_tensor.placements[0].dim, 1)
    # 定义一个测试函数，用于测试分发张量时的错误情况
    def test_distribute_tensor_errors(self):
        # 创建设备网格对象，使用给定的设备类型和编号网格
        device_mesh = DeviceMesh(
            self.device_type, torch.arange(self.world_size).reshape(2, 2)
        )
        # 定义张量的形状，维度为 3*world_size × 3*world_size
        tensor_shape = [3 * self.world_size, 3 * self.world_size]
        # 创建一个随机张量，形状为 tensor_shape
        tensor_to_distribute = torch.randn(*tensor_shape)

        # 测试：验证在分片规范 shard_spec 下分发张量时是否抛出 ValueError 异常
        with self.assertRaisesRegex(ValueError, "must have the same length"):
            shard_spec = [Shard(0)]
            distribute_tensor(tensor_to_distribute, device_mesh, shard_spec)

        # 测试：验证在分片规范 shard_spec 下分发带梯度的全局张量时是否抛出 RuntimeError 异常
        with self.assertRaisesRegex(RuntimeError, "distribute leaf tensor"):
            shard_spec = [Shard(0)]
            # 创建带梯度的随机全局张量
            global_tensor = torch.randn(*tensor_shape, requires_grad=True)
            global_tensor_to_distribute = global_tensor + 2
            distribute_tensor(global_tensor_to_distribute, device_mesh, shard_spec)

        # 定义分片规范 spec，包含两个 Shard 对象
        spec = [Shard(0), Shard(1)]
        # 将 tensor_to_distribute 按照 spec 规范分发，并获取分布后的张量
        dtensor = distribute_tensor(tensor_to_distribute, device_mesh, spec)

        # 测试：验证当分发的张量尝试应用于不同设备网格时是否抛出 ValueError 异常
        with self.assertRaisesRegex(ValueError, "to a different device mesh"):
            new_mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))
            distribute_tensor(dtensor, new_mesh, [Shard(0)])

        # 测试：验证当分发的张量尝试应用于不同分布规范时是否抛出 ValueError 异常
        with self.assertRaisesRegex(ValueError, "to a different placements"):
            new_spec = [Shard(0), Replicate()]
            distribute_tensor(dtensor, device_mesh, new_spec)

    # 用于测试不均匀分片情况的分发张量函数
    @with_comms
    def test_distribute_tensor_uneven_sharding(self):
        # 创建设备网格对象，使用给定的设备类型和编号网格
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        # 定义输入大小和分片维度的列表
        input_sizes_and_shard_dims = [
            ((self.world_size * 3 + 1, 3, 3), 0),
            ((self.world_size * 3 + 2, 3, 3), 0),
            ((3, self.world_size * 3 + 1, 3), 1),
            ((3, self.world_size * 3 + 2, 3), 1),
            ((3, 3, self.world_size * 3 + 1), 2),
            ((3, 3, self.world_size * 3 + 2), 2),
        ]
        # 遍历输入大小和分片维度的组合
        for input_size, shard_dim in input_sizes_and_shard_dims:
            shard_spec = [Shard(shard_dim)]
            # 创建形状为 input_size 的随机张量
            tensor_to_shard = torch.randn(input_size)
            # 在给定维度 shard_dim 上将 tensor_to_shard 分成 self.world_size 份
            splitted_tensor_list = list(
                torch.chunk(tensor_to_shard, self.world_size, dim=shard_dim)
            )
            # 将 tensor_to_shard 按照 shard_spec 规范分发，并获取分布后的张量
            dist_tensor = distribute_tensor(tensor_to_shard, device_mesh, shard_spec)
            # 断言分发后张量的形状与 input_size 相同
            self.assertEqual(dist_tensor.size(), torch.Size(input_size))
            # 将分发后的张量转为本地张量
            local_tensor = dist_tensor.to_local()
            # 断言本地张量与原始张量在当前进程的分片结果相同
            self.assertEqual(local_tensor, splitted_tensor_list[self.rank])

    # 使用通信装饰器，用于测试分发张量的函数
    @with_comms
    def test_distribute_module_input_fn_output_fn(self):
        # 创建设备网格对象，指定设备类型和世界大小范围
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))

        # 创建需要完全复制的线性模型实例
        module_to_replicate = MyModel(20, 1, device=self.device_type)

        # 定义输入函数，将输入张量在维度0上进行分片标记
        def input_fn(mod, inputs, device_mesh):
            return DTensor.from_local(inputs[0], device_mesh, [Shard(0)])

        # 定义输出函数，确保输出是 DTensor 类型，并将其转换为本地格式
        def output_fn(mod, outputs, device_mesh):
            assert isinstance(outputs, DTensor)
            return outputs.to_local()

        # 将模型复制到设备网格上，并应用定义的输入和输出函数
        replica_module = distribute_module(
            module_to_replicate,
            device_mesh,
            input_fn=input_fn,
            output_fn=output_fn,
        )

        # 创建输入张量并对复制后的模型进行推理
        input_tensor = torch.randn(5, 20, device=self.device_type)
        local_out = replica_module(input_tensor)
        self.assertIsInstance(local_out, torch.Tensor)
        self.assertNotIsInstance(local_out, DTensor)

        # 创建另一个需要完全复制的模型
        model = MyModel(10, 10, device=self.device_type)

        # 定义输入函数，将输入张量完全复制到设备网格上
        def replicate_input_fn(mod, inputs, device_mesh):
            return DTensor.from_local(inputs[0], device_mesh, [Replicate()])

        # 将模型复制到设备网格上，仅应用定义的输入函数
        replica_model = distribute_module(
            model,
            device_mesh,
            input_fn=replicate_input_fn,
        )

        # 创建输入张量并对复制后的模型进行推理和反向传播
        input = torch.randn(10, 10, requires_grad=True)
        output = replica_model(input)
        output.sum().backward()
        param_grad = next(iter(replica_model.parameters())).grad
        self.assertTrue(isinstance(param_grad, DTensor))
        self.assertTrue(isinstance(param_grad.placements[0], Replicate))

    @with_comms
    def test_distribute_module_input_fn_output_fn_warning(self):
        # 创建设备网格对象，指定设备类型和世界大小范围
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))

        # 创建需要完全复制的线性模型实例
        module_to_replicate = MyModel(20, 1, device=self.device_type)

        # 标记输入函数，将输入张量在维度0上进行分片标记
        def input_fn(inputs, device_mesh):
            return DTensor.from_local(inputs[0], device_mesh, [Shard(0)])

        # 标记输出函数，确保输出是 DTensor 类型，并将其转换为本地格式
        def output_fn(outputs, device_mesh):
            assert isinstance(outputs, DTensor)
            return outputs.to_local()

        # 断言在使用过程中会发出 FutureWarning 警告信息
        with self.assertWarnsRegex(FutureWarning, "Deprecating"):
            replica_module = distribute_module(
                module_to_replicate,
                device_mesh,
                input_fn=input_fn,
                output_fn=output_fn,
            )

        # 创建输入张量并对复制后的模型进行推理
        input_tensor = torch.randn(5, 20, device=self.device_type)
        local_out = replica_module(input_tensor)
        self.assertIsInstance(local_out, torch.Tensor)
        self.assertNotIsInstance(local_out, DTensor)
    def test_distribute_module_casting(self):
        # 创建设备网格对象，用于模拟设备之间的通信
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))

        # 检查 DTensor 的数据类型转换
        dt = DTensor.from_local(torch.rand(10), device_mesh, [Replicate()])
        dt = dt.to(torch.bfloat16)
        self.assertEqual(dt.dtype, torch.bfloat16)
        self.assertEqual(dt._local_tensor.dtype, torch.bfloat16)

        # 检查 distribute_tensor 的数据类型转换
        dt = distribute_tensor(torch.rand(10), device_mesh, [Replicate()])
        dt = dt.to(torch.bfloat16)
        self.assertEqual(dt.dtype, torch.bfloat16)
        self.assertEqual(dt._local_tensor.dtype, torch.bfloat16)

        # 检查 distribute_module 的数据类型转换
        model = MyModel(10, 10, device=self.device_type)
        replica_model = distribute_module(
            model,
            device_mesh,
        )
        replica_model = replica_model.to(torch.bfloat16)
        self.assertEqual(replica_model.seq[0].weight.dtype, torch.bfloat16)
        self.assertEqual(
            replica_model.seq[0].weight._local_tensor.dtype, torch.bfloat16
        )

        # 检查自动混合精度（autocast）
        dt = distribute_tensor(torch.rand(10), device_mesh, [Replicate()])
        replica_model = distribute_module(
            model,
            device_mesh,
        )
        with torch.autocast(device_type=self.device_type, dtype=torch.bfloat16):
            output = replica_model(dt)
        self.assertEqual(output.dtype, torch.bfloat16)

    @with_comms
    def test_distribute_module_meta(self):
        # 如果模型过大，用户可能首先在元设备上创建整个模型，然后在分区函数中初始化它到设备上。
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))

        # 在维度 0 上完全分片所有参数
        module_to_shard = MyModel(5 * self.world_size, 20, device="meta")

        shard_spec = [Shard(0)]

        def shard_fn(name, module, device_mesh):
            for param_name, param in module._parameters.items():
                # 分发参数到设备网格上的特定分片
                dist_param = distribute_tensor(param, device_mesh, shard_spec)
                dist_param = torch.empty_like(
                    dist_param, device=device_mesh.device_type
                )
                module.register_parameter(param_name, torch.nn.Parameter(dist_param))

        # 分发模型并应用分片函数
        sharded_module = distribute_module(module_to_shard, device_mesh, shard_fn)
        for param in sharded_module.parameters():
            self.assertIsInstance(param, DTensor)
            self.assertFalse(param.is_meta)
            self.assertTrue(param.device.type == device_mesh.device_type)
# 如果当前模块被直接执行（而不是被导入到其它模块中），则执行以下代码块
if __name__ == "__main__":
    # 调用名为 run_tests 的函数来执行测试
    run_tests()
```