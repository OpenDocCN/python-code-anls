# `.\pytorch\test\distributed\test_device_mesh.py`

```
    @skip_if_lt_x_gpu(4)
    def test_assert_invalid_mesh_tensor(self):
        # 创建一个张量 mesh，用于设备间通信
        mesh = torch.arange(self.world_size).to(self.rank)
        # 当设备数量小于 4 时，断言会引发 ValueError 异常
        with self.assertRaises(ValueError):
            # 尝试使用 mesh 创建 DeviceMesh 对象，应该会失败
            device_mesh = DeviceMesh(self.device_type, mesh)

    @with_comms
    # 定义测试方法：测试获取组和获取所有组功能
    def test_get_group_and_get_all_groups(self):
        # 定义网格形状为二维，第一维度为2，第二维度为总体大小除以2
        mesh_shape = (2, self.world_size // 2)
        # 使用给定设备类型和网格形状初始化设备网格对象
        mesh_2d = init_device_mesh(
            self.device_type, mesh_shape, mesh_dim_names=("dp", "tp")
        )

        # 获取 'tp' 维度的网格组
        tp_mesh = mesh_2d["tp"]
        # 获取 'dp' 维度的网格组
        dp_mesh = mesh_2d["dp"]

        # 断言：获取网格对象中第一个组与 'dp' 维度的网格组相等
        self.assertEqual(mesh_2d.get_group(0), mesh_2d.get_group("dp"))
        # 断言：获取网格对象中第二个组与 'tp' 维度的网格组相等
        self.assertEqual(mesh_2d.get_group(1), mesh_2d.get_group("tp"))

        # 断言：获取 'dp' 维度的网格组与 dp_mesh 对象的组相等
        self.assertEqual(mesh_2d.get_group("dp"), dp_mesh.get_group())
        # 断言：获取 'tp' 维度的网格组与 tp_mesh 对象的组相等
        self.assertEqual(mesh_2d.get_group("tp"), tp_mesh.get_group())

        # 获取所有网格组
        groups = mesh_2d.get_all_groups()
        # 断言：网格组的数量为2
        self.assertEqual(len(groups), 2)
        # 断言：'tp' 维度的网格组在所有网格组中
        self.assertTrue(tp_mesh.get_group() in groups)
        # 断言：'dp' 维度的网格组在所有网格组中
        self.assertTrue(dp_mesh.get_group() in groups)

    # 装饰器：使用通信功能
    @with_comms
    # 定义测试方法：测试在未指定 mesh_dim 时获取本地排名是否引发异常
    def test_get_local_rank_raises_exception(self):
        # 定义网格形状为二维，第一维度为2，第二维度为总体大小除以2
        mesh_shape = (2, self.world_size // 2)
        # 使用给定设备类型和网格形状初始化设备网格对象
        mesh_2d = init_device_mesh(
            self.device_type, mesh_shape, mesh_dim_names=("dp", "tp")
        )

        # 断言：调用 get_local_rank 方法时引发 RuntimeError 异常，异常信息需包含指定文本
        with self.assertRaisesRegex(
            RuntimeError,
            "Optional kwarg `mesh_dim` needs to be specified when device_mesh.ndim > 1.",
        ):
            local_rank = mesh_2d.get_local_rank()

    # 装饰器：使用通信功能
    @with_comms
    # 定义测试方法：测试获取指定维度的本地排名功能
    def test_get_local_rank(self):
        # 定义网格形状为二维，第一维度为2，第二维度为总体大小除以2
        mesh_shape = (2, self.world_size // 2)
        # 使用给定设备类型和网格形状初始化设备网格对象
        mesh_2d = init_device_mesh(
            self.device_type, mesh_shape, mesh_dim_names=("dp", "tp")
        )
        # 断言：获取 'dp' 维度的本地排名与索引为0的本地排名相等
        self.assertEqual(mesh_2d.get_local_rank("dp"), mesh_2d.get_local_rank(0))
        # 断言：获取 'tp' 维度的本地排名与索引为1的本地排名相等
        self.assertEqual(mesh_2d.get_local_rank("tp"), mesh_2d.get_local_rank(1))

        # 获取 'dp' 维度的网格组
        dp_mesh = mesh_2d["dp"]
        # 获取 'tp' 维度的网格组
        tp_mesh = mesh_2d["tp"]
        # 断言：dp_mesh 对象的本地排名与 'dp' 维度的本地排名相等
        self.assertEqual(dp_mesh.get_local_rank(), mesh_2d.get_local_rank("dp"))
        # 断言：tp_mesh 对象的本地排名与 'tp' 维度的本地排名相等
        self.assertEqual(tp_mesh.get_local_rank(), mesh_2d.get_local_rank("tp"))

    # 装饰器：使用通信功能
    @with_comms
    # 定义测试方法：测试二维设备网格对象的功能
    def test_device_mesh_2d(self):
        # 创建一个包含4个元素的张量，并重塑为2x2的形状
        mesh_tensor = torch.arange(4).reshape(2, 2)
        # 使用给定设备类型和张量初始化设备网格对象
        mesh = DeviceMesh(self.device_type, mesh_tensor)

        # 获取所有维度的子组
        dim_to_subgroups = mesh.get_all_groups()

        # 预期每个维度的排名
        expected_ranks_by_dim = [[[0, 2], [1, 3]], [[0, 1], [2, 3]]]
        # 遍历每个维度及其组
        for dim, dim_group in enumerate(dim_to_subgroups):
            # 断言：维度索引应小于2
            self.assertTrue(dim < 2)
            # 获取当前维度的预期排名
            dim_ranks = expected_ranks_by_dim[dim]

            # 获取维度组的大小
            dim_group_size = get_world_size(dim_group)
            # 断言：维度组类型为 ProcessGroup
            self.assertIsInstance(dim_group, ProcessGroup)
            # 断言：维度组大小为2
            self.assertEqual(dim_group_size, 2)
            # 获取全局排名列表
            global_ranks = [
                get_global_rank(dim_group, i) for i in range(dim_group_size)
            ]
            # 获取当前排名的预期组排名
            current_rank_expected_group_ranks = (
                dim_ranks[0] if self.rank in dim_ranks[0] else dim_ranks[1]
            )
            # 断言：全局排名与预期组排名相等
            self.assertEqual(global_ranks, current_rank_expected_group_ranks)
    def test_device_mesh_init_backend(self):
        # 创建一个 DeviceMesh 对象，设备类型为 self.device_type，包含一个单元素的列表 [1]，不初始化后端
        mesh = DeviceMesh(self.device_type, [1], _init_backend=False)

        # 使用断言检查是否抛出 RuntimeError 异常，异常消息为 "process groups not initialized!"
        with self.assertRaisesRegex(RuntimeError, "process groups not initialized!"):
            mesh.get_group()

        # 当 init_backend 为 False 时，应始终填充 coordinates（坐标），因为每次调用 init_backend 时，应确保默认的进程组已创建
        mesh.get_coordinate()

    def test_fake_pg_device_mesh(self):
        # 创建一个 FakeStore 对象作为存储
        fake_store = FakeStore()
        # 使用 fake store 初始化一个进程组，设备类型为 "cuda"（如果可用则为 CUDA，否则为 CPU），排名为 0，世界大小为 self.world_size
        init_process_group("fake", store=fake_store, rank=0, world_size=self.world_size)
        # 确定设备类型为 "cuda"（如果可用）
        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        # 创建一个 DeviceMesh 对象，设备类型为 device_type，包含一个从 0 到 self.world_size-1 的张量
        mesh = DeviceMesh(device_type, torch.arange(self.world_size))

        # 创建一个本地张量
        local_tensor = torch.randn(2, 8)
        # 对 local_tensor 进行全局收集，按维度 0 收集，使用 mesh 的第一个进程组（group=(mesh, 0)）
        global_tensor = funcol.all_gather_tensor(
            local_tensor, gather_dim=0, group=(mesh, 0)
        )
        # 断言全局张量的形状为 (self.world_size * 2, 8)
        self.assertEqual(global_tensor.shape, (self.world_size * 2, 8))

    @with_comms
    def test_from_group_with_global_pg(self):
        # 简单测试：检查使用全局进程组（global_pg）与直接通过 init_device_mesh 初始化的区别
        global_pg = _get_default_group()
        # 使用 init_device_mesh 初始化一个参考的全局 Mesh，设备类型为 "cuda"，大小为 (self.world_size,)
        ref_global_mesh = init_device_mesh("cuda", (self.world_size,))
        # 使用 DeviceMesh.from_group 从全局进程组创建一个 global_mesh，设备类型为 "cuda"
        global_mesh = DeviceMesh.from_group(global_pg, "cuda")
        # 断言 ref_global_mesh 与 global_mesh 相等
        self.assertEqual(ref_global_mesh, global_mesh)
        # 断言 ref_global_mesh 的维度组信息与 global_mesh 相同
        self.assertEqual(ref_global_mesh._dim_group_infos, global_mesh._dim_group_infos)
        # 断言 ref_global_mesh 的每个维度上的坐标信息与 global_mesh 相同
        self.assertEqual(
            ref_global_mesh._coordinate_on_dim, global_mesh._coordinate_on_dim
        )

    @with_comms
    def test_from_group_with_invalid_mesh(self):
        # 获取默认的全局进程组
        global_pg = _get_default_group()
        # 获取全局进程组的大小
        global_pg_size = global_pg.size()
        # 断言全局进程组大小为 4，用于测试假设全局世界大小为 4
        assert global_pg_size == 4, "Test assumes global world size of 4"
        # 创建一个无效的 mesh，期望的是 1D，但提供了一个 2D 的 mesh
        invalid_mesh = [[0, 1], [2, 3]]  # 2D mesh when we need 1D
        # 构建用于匹配的正则表达式
        regex = r"Invalid mesh \[\[0, 1\], \[2, 3\]\] for ProcessGroup with ranks \[0, 1, 2, 3\]"
        # 使用断言检查是否抛出 ValueError 异常，异常消息应匹配 regex
        with self.assertRaisesRegex(ValueError, regex):
            DeviceMesh.from_group(global_pg, "cuda", invalid_mesh)

        # 使用 init_device_mesh 初始化一个设备 mesh，设备类型为 self.device_type，大小为 (2, 2)
        device_mesh = init_device_mesh(self.device_type, (2, 2))
        # 获取设备 mesh 的所有进程组
        groups = device_mesh.get_all_groups()
        # 创建一个无效的 mesh，期望的是 2D，但提供了一个 1D 的 mesh
        invalid_mesh = (0, 1, 2, 3)  # 1D mesh when we need 2D
        # 构建用于匹配的正则表达式
        regex = r"Expects mesh with ndim equal to number of ProcessGroups but got mesh \[0, 1, 2, 3\] and 2 ProcessGroups"
        # 使用断言检查是否抛出 ValueError 异常，异常消息应匹配 regex
        with self.assertRaisesRegex(ValueError, regex):
            DeviceMesh.from_group(groups, self.device_type, invalid_mesh)

    def test_raises_invalid_device_type(self):
        # 使用断言检查是否抛出 RuntimeError 异常，异常消息为 "Device type with GPU index is not supported"
        with self.assertRaisesRegex(
            RuntimeError,
            "Device type with GPU index is not supported",
        ):
            # 使用一个包含 GPU 索引的无效设备类型测试 init_device_mesh
            mesh_shape = (2, self.world_size // 2)
            mesh_2d = init_device_mesh(
                "cuda:0", mesh_shape=mesh_shape, mesh_dim_names=("dp", "tp")
            )

    @with_comms
    # 定义测试方法 test_set_mesh_dim_group_options
    def test_set_mesh_dim_group_options(self):
        # 根据是否支持 CUDA 设置设备类型为 "cuda" 或者 "cpu"
        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 调用 _set_mesh_dim_group_options 方法，设置 mesh 的维度组选项
        _mesh_resources._set_mesh_dim_group_options(1, "fake", None)
        
        # 创建一个 2x2 的张量 mesh_tensor，元素为 [0, 1, 2, 3]
        mesh_tensor = torch.arange(4).reshape(2, 2)
        
        # 根据设备类型和张量创建 DeviceMesh 对象
        mesh = DeviceMesh(device_type, mesh_tensor)
        
        # 断言：获取组索引为 1 的组，检查其后端名称为 "fake"
        self.assertEqual(mesh.get_group(1)._get_backend_name(), "fake")
class DeviceMeshTestNDim(DTensorTestBase):
    @property
    def world_size(self):
        # 返回测试中设定的虚拟世界大小，这里固定为 8
        return 8

    @with_comms
    def test_device_mesh_nd(self):
        # 构建一个 CUDA 设备网格
        mesh_tensor = torch.arange(8).reshape(2, 2, 2)
        # 使用设备网格张量创建设备网格对象
        mesh = DeviceMesh(self.device_type, mesh_tensor)

        # 检查所有维度分组
        dim_to_subgroups = mesh.get_all_groups()

        for dim, dim_group in enumerate(dim_to_subgroups):
            # 断言当前维度小于设备网格张量的维度数
            self.assertTrue(dim < mesh_tensor.ndim)
            # 重新排列维度，获取当前维度对应的 ranks
            dim_ranks = mesh_tensor.swapdims(-1, dim).reshape(-1, 2)

            # 获取维度分组的大小
            dim_group_size = get_world_size(dim_group)
            # 断言维度分组是进程组的实例
            self.assertIsInstance(dim_group, ProcessGroup)
            # 断言维度分组的大小为 2
            self.assertEqual(dim_group_size, 2)

            # 获取全局 ranks
            global_ranks = [
                get_global_rank(dim_group, i) for i in range(dim_group_size)
            ]
            for ranks in dim_ranks:
                if self.rank in ranks:
                    # 断言全局 ranks 和当前 ranks 一致
                    self.assertEqual(global_ranks, ranks.tolist())

    @with_comms
    def test_device_mesh_hash(self):
        # 创建一个二维设备网格张量
        mesh_tensor_2d = torch.arange(8).reshape(4, 2)
        # 使用设备类型创建设备网格对象
        mesh = DeviceMesh(self.device_type, mesh_tensor_2d)
        mesh2 = DeviceMesh(self.device_type, mesh_tensor_2d)
        # 断言两个相同网格对象的哈希值相等
        self.assertEqual(hash(mesh), hash(mesh2))

        # 创建一个不同的三维设备网格张量
        mesh_tensor_3d = torch.arange(8).reshape(2, 2, 2)
        mesh3 = DeviceMesh(self.device_type, mesh_tensor_3d)
        # 断言不同网格对象的哈希值不相等
        self.assertNotEqual(hash(mesh), hash(mesh3))
        self.assertNotEqual(hash(mesh2), hash(mesh3))

    @with_comms
    def test_get_local_rank_3d(self):
        """
        如果我们有一个三维网格，想要对其应用 dp、pp、tp，网格维度名称为 ["dp", "pp", "tp"]，
        网格张量如下：
        mesh_3d_tensor = [
            [
                [0, 1],
                [2, 3],
            ],
            [
                [4, 5],
                [6, 7],
            ]
        ]
        """
        mesh_shape = (2, 2, 2)
        # 使用设备类型和网格维度名称初始化设备网格对象
        mesh_3d = init_device_mesh(
            self.device_type, mesh_shape, mesh_dim_names=("dp", "pp", "tp")
        )

        # 计算在 "tp" 维度上的本地 rank
        tp_rank = mesh_3d.get_local_rank("tp")
        print(f"{self.rank=}, {tp_rank=}")
        expected_tp_rank = self.rank % 2
        self.assertEqual(tp_rank, expected_tp_rank)

        # 计算在 "pp" 维度上的本地 rank
        pp_rank = mesh_3d.get_local_rank("pp")
        expected_pp_rank = 0 if self.rank % 4 <= 1 else 1
        self.assertEqual(pp_rank, expected_pp_rank)

        # 计算在 "dp" 维度上的本地 rank
        dp_rank = mesh_3d.get_local_rank("dp")
        expected_dp_rank = self.rank // 4
        self.assertEqual(dp_rank, expected_dp_rank)
    def test_device_mesh_parent_child_hash(self):
        # 初始化一个二维设备网格，根据设备类型和世界大小分配网格维度名称为("DP", "TP")
        mesh_2d = init_device_mesh(
            self.device_type, (2, self.world_size // 2), mesh_dim_names=("DP", "TP")
        )

        # 创建两个设备组，分别对应前半部分和后半部分的设备索引
        mesh_group_1 = torch.arange(0, self.world_size // 2)
        mesh_group_2 = torch.arange(self.world_size // 2, self.world_size)

        # 创建两个设备网格对象，ep_mesh_1 对应前半部分设备组，ep_mesh_2 对应后半部分设备组
        ep_mesh_1 = DeviceMesh(self.device_type, mesh_group_1)
        ep_mesh_2 = DeviceMesh(self.device_type, mesh_group_2)

        # 根据当前进程的排名确定使用哪个设备网格对象
        ep_mesh = ep_mesh_1 if self.rank < self.world_size // 2 else ep_mesh_2
        
        # 检查 ep_mesh 和 mesh_2d["TP"] 是否不同，因为 mesh_2d["TP"] 有一个父网格而 ep_mesh 没有
        self.assertEqual(mesh_2d["TP"]._flatten_mesh_list, ep_mesh._flatten_mesh_list)
        self.assertEqual(mesh_2d["TP"].mesh.shape, ep_mesh.mesh.shape)
        self.assertEqual(mesh_2d["TP"].device_type, ep_mesh.device_type)
        self.assertNotEqual(mesh_2d["TP"].mesh_dim_names, ep_mesh.mesh_dim_names)
        self.assertEqual(mesh_2d["TP"]._thread_id, ep_mesh._thread_id)
        self.assertNotEqual(mesh_2d["TP"]._parent_mesh, ep_mesh._parent_mesh)
        self.assertNotEqual(hash(mesh_2d["TP"]), hash(ep_mesh))
        self.assertNotEqual(mesh_2d["TP"], ep_mesh)

        # 创建另外一个设备网格对象，与 ep_mesh 具有相同的网格和没有父网格
        another_mesh_1 = DeviceMesh(self.device_type, mesh_group_1)
        another_mesh_2 = DeviceMesh(self.device_type, mesh_group_2)
        another_mesh = (
            another_mesh_1 if self.rank < self.world_size // 2 else another_mesh_2
        )

        # 检查 another_mesh 和 ep_mesh 是否相同，因为它们具有相同的网格和没有父网格
        self.assertEqual(ep_mesh._flatten_mesh_list, another_mesh._flatten_mesh_list)
        self.assertEqual(ep_mesh.mesh.shape, another_mesh.mesh.shape)
        self.assertEqual(ep_mesh.device_type, another_mesh.device_type)
        self.assertEqual(ep_mesh.mesh_dim_names, another_mesh.mesh_dim_names)
        self.assertEqual(ep_mesh._thread_id, another_mesh._thread_id)
        self.assertEqual(ep_mesh._parent_mesh, another_mesh._parent_mesh)
        self.assertEqual(hash(ep_mesh), hash(another_mesh))
        self.assertEqual(ep_mesh, another_mesh)

    @with_comms
    def test_from_group_with_mesh_shape(self):
        """Tests ``from_group`` when passing ``mesh_shape`` as 2D."""
        # 定义一个三维的网格形状
        mesh_shape = (2, 2, 2)
        # 定义网格维度名称
        mesh_dim_names = ("dp_replicate", "dp_shard", "tp")
        # 初始化参考网格对象
        ref_mesh = init_device_mesh(
            self.device_type, mesh_shape, mesh_dim_names=mesh_dim_names
        )

        # 获取 dp_shard 维度的分组信息
        dp_shard_group = ref_mesh["dp_shard"].get_group()
        # 获取 dp_replicate 维度的分组信息
        dp_replicate_group = ref_mesh["dp_replicate"].get_group()

        # 使用 DeviceMesh 的 from_group 方法构建 dp 维度的网格对象
        dp_mesh = DeviceMesh.from_group(
            [dp_replicate_group, dp_shard_group],
            self.device_type,
            mesh=ref_mesh.mesh[:, :, ref_mesh.get_local_rank(2)],
            mesh_dim_names=mesh_dim_names[:2],
        )

        # 获取参考网格对象的前两个维度的分组信息
        ref_mesh_dp_dim_group_infos = ref_mesh._dim_group_infos[:2]
        # 检查构建的 dp_mesh 对象的分组信息与参考网格对象的是否一致
        for (_, ref_ranks, _), (_, ranks, _) in zip(
            ref_mesh_dp_dim_group_infos, dp_mesh._dim_group_infos
        ):
            self.assertEqual(ref_ranks, ranks)
        
        # 由于父网格不同，无法直接比较网格对象的相等性
        self.assertEqual(dp_mesh["dp_replicate"].mesh, ref_mesh["dp_replicate"].mesh)
        # 检查 dp_replicate 维度的分组信息是否与参考网格对象一致
        for (_, ref_ranks, _), (_, ranks, _) in zip(
            dp_mesh["dp_replicate"]._dim_group_infos,
            ref_mesh["dp_replicate"]._dim_group_infos,
        ):
            self.assertEqual(ref_ranks, ranks)
        
        # 检查 dp_shard 维度的网格是否与参考网格对象一致
        self.assertEqual(dp_mesh["dp_shard"].mesh, ref_mesh["dp_shard"].mesh)
        # 检查 dp_shard 维度的分组信息是否与参考网格对象一致
        for (_, ref_ranks, _), (_, ranks, _) in zip(
            dp_mesh["dp_shard"]._dim_group_infos, ref_mesh["dp_shard"]._dim_group_infos
        ):
            self.assertEqual(ref_ranks, ranks)
class InitDeviceMeshTest(DTensorTestBase):
    @property
    def world_size(self):
        return 8

    @with_comms
    def test_init_device_mesh(self):
        mesh_shape = (2, 4)  # 定义设备网格的形状为二维，2行4列
        mesh_dim_names = ("DP", "TP")  # 定义设备网格的维度名称为 "DP" 和 "TP"
        ref_mesh = DeviceMesh(  # 创建参考的设备网格对象
            self.device_type,  # 使用当前测试类的设备类型
            torch.arange(8).view(mesh_shape),  # 使用torch.arange生成0到7的张量，并reshape为设定的网格形状
            mesh_dim_names=mesh_dim_names,  # 设置设备网格的维度名称
        )

        # test init_device_mesh with mesh_dim_names
        mesh_2d = init_device_mesh(  # 调用初始化设备网格函数，使用指定的设备类型、网格形状和维度名称
            self.device_type, mesh_shape, mesh_dim_names=mesh_dim_names
        )
        self.assertEqual(mesh_2d, ref_mesh)  # 断言初始化的设备网格与参考网格对象相等
        self.assertEqual(mesh_2d.mesh_dim_names, mesh_dim_names)  # 断言初始化的设备网格的维度名称与设定的一致

    @with_comms
    def test_raises_duplicate_mesh_dim_names(self):
        with self.assertRaisesRegex(
            RuntimeError,
            "Each mesh_dim_name must be unique.",
        ):
            mesh = init_device_mesh(  # 调用初始化设备网格函数，传入设备类型和重复的网格维度名称列表
                self.device_type,
                (2, 4),
                mesh_dim_names=["dp", "dp"],
            )

    @with_comms
    def test_raises_mesh_shape_mesh_dim_names_mismatch(self):
        with self.assertRaisesRegex(
            RuntimeError,
            "mesh_shape and mesh_dim_names should have same length!",
        ):
            mesh = init_device_mesh(  # 调用初始化设备网格函数，传入设备类型、不匹配的网格形状和维度名称列表
                self.device_type,
                (8,),
                mesh_dim_names=["dp", "tp"],
            )


class TestDeviceMeshGetItem(DTensorTestBase):
    @property
    def world_size(self):
        return 8

    @with_comms
    def test_raises_no_mesh_dim_found(self):
        with self.assertRaisesRegex(
            RuntimeError, "Cannot slice a DeviceMesh without mesh_dim_names!"
        ):
            mesh = init_device_mesh(self.device_type, (2, 4))  # 调用初始化设备网格函数，未提供网格维度名称
            child_mesh = mesh["DP"]  # 尝试使用网格维度名称索引子网格

    @with_comms
    def test_raises_invalid_mesh_dim_name(self):
        child_mesh_dim_name = ("PP",)  # 设置一个无效的子网格维度名称
        with self.assertRaisesRegex(KeyError, "Invalid mesh_dim_name"):
            mesh_dim_names = ("DP", "TP")  # 定义设备网格的有效维度名称
            mesh = init_device_mesh(  # 调用初始化设备网格函数，传入设备类型、网格形状和有效的网格维度名称
                self.device_type, (2, 4), mesh_dim_names=mesh_dim_names
            )
            child_mesh = mesh[child_mesh_dim_name]  # 尝试使用无效的网格维度名称索引子网格
    def test_get_item_2d(self):
        # 定义网格形状为 (2, 4)
        mesh_shape = (2, 4)
        # 定义网格维度名称为 ("DP", "TP")
        mesh_dim_names = ("DP", "TP")
        # 初始化二维设备网格
        mesh_2d = init_device_mesh(
            self.device_type, mesh_shape, mesh_dim_names=mesh_dim_names
        )

        # 初始化用于存储不同维度名称对应的网格 ranks 的字典
        pg_ranks_by_dim_name = {}
        # 遍历网格维度名称列表
        for mesh_dim_name in mesh_dim_names:
            # 获取当前维度名称在列表中的索引
            mesh_dim = mesh_dim_names.index(mesh_dim_name)
            # 将当前维度名称对应的网格 ranks 存储到字典中
            pg_ranks_by_dim_name[mesh_dim_name] = mesh_2d.mesh.swapdims(
                -1, mesh_dim
            ).reshape(-1, mesh_2d.mesh.size(mesh_dim))

        # 获取 "TP" 维度的网格
        tp_mesh = mesh_2d["TP"]
        # 计算当前进程的 TP 组索引
        tp_group_idx = self.rank // 4
        # 断言 TP 网格与预期的 ranks 相等
        self.assertEqual(tp_mesh.mesh, pg_ranks_by_dim_name["TP"][tp_group_idx])

        # 获取 "DP" 维度的网格
        dp_mesh = mesh_2d["DP"]
        # 计算当前进程的 DP 组索引
        dp_group_idx = self.rank % 4
        # 断言 DP 网格与预期的 ranks 相等
        self.assertEqual(mesh_2d["DP"].mesh, pg_ranks_by_dim_name["DP"][dp_group_idx])

    @with_comms
    def test_get_item_1d(self):
        # 初始化一维设备网格
        mesh = init_device_mesh(self.device_type, (8,), mesh_dim_names=("dp",))
        # 确保从一维网格中切片出一维网格的操作正常工作
        # 这里只是返回一个没有父网格的虚拟值
        dp_mesh = mesh["dp"]
        # 断言切片后的一维网格与原网格相等
        self.assertEqual(dp_mesh, mesh)

        # 断言捕获 KeyError 异常，提示 "Invalid mesh_dim_name"
        with self.assertRaisesRegex(KeyError, "Invalid mesh_dim_name"):
            dp_mesh = mesh["dim0"]

    @with_comms
    def test_get_item_3d(self):
        # 定义三维网格形状为 (2, 2, 2)
        mesh_shape = (2, 2, 2)
        # 定义三维网格维度名称为 ("Replicate", "Shard", "TP")
        mesh_dim_names = ("Replicate", "Shard", "TP")
        # 初始化三维设备网格
        mesh_3d = init_device_mesh(
            self.device_type, mesh_shape, mesh_dim_names=mesh_dim_names
        )

        # 定义 TP 组列表
        tp_group = [[0, 1], [2, 3], [4, 5], [6, 7]]
        # 计算当前进程的 TP 组索引
        tp_group_idx = int(self.rank / 2)
        # 断言 TP 网格与预期的组相等
        self.assertEqual(mesh_3d["TP"].mesh.tolist(), tp_group[tp_group_idx])

        # 定义 Shard 组列表
        shard_group = [[0, 2], [1, 3], [4, 6], [5, 7]]
        # 计算当前进程的 Shard 组索引
        shard_group_idx = self.rank % 2 + self.rank // 4 * 2
        # 断言 Shard 网格与预期的组相等
        self.assertEqual(mesh_3d["Shard"].mesh.tolist(), shard_group[shard_group_idx])

        # 定义 Replicate 组列表
        replicate_group = [[0, 4], [1, 5], [2, 6], [3, 7]]
        # 计算当前进程的 Replicate 组索引
        replicate_group_idx = self.rank % 4
        # 断言 Replicate 网格与预期的组相等
        self.assertEqual(
            mesh_3d["Replicate"].mesh.tolist(), replicate_group[replicate_group_idx]
        )

        # 支持 nD 切片的 UX（用户体验）
        # mesh_3d[["Replicate", "Shard"]] 或 mesh_3d["Replicate", "Shard"]
        # 获取同时包含 "Replicate" 和 "Shard" 维度的网格
        hsdp_mesh_1 = mesh_3d[["Replicate", "Shard"]]
        hsdp_mesh_2 = mesh_3d["Replicate", "Shard"]
        # 定义多维切片组
        hsdp_group = [[[0, 2], [4, 6]], [[1, 3], [5, 7]]]
        # 计算当前进程的切片组索引
        hsdp_group_idx = self.rank % 2
        # 断言两种切片方式得到的网格相等
        self.assertEqual(hsdp_mesh_1.mesh.tolist(), hsdp_group[hsdp_group_idx])
        self.assertEqual(hsdp_mesh_2.mesh.tolist(), hsdp_group[hsdp_group_idx])
        # 断言两种切片方式得到的网格对象完全相等
        self.assertEqual(hsdp_mesh_1, hsdp_mesh_2)
    def test_cache_and_reuse_submesh_slice_result(self):
        # 初始化一个设备上的网格，指定维度为 (2, 4)，并命名为 "dp" 和 "tp"
        mesh = init_device_mesh(self.device_type, (2, 4), mesh_dim_names=("dp", "tp"))

        # 从网格中获取 "dp" 维度的子网格
        dp_mesh = mesh["dp"]

        # 获取当前世界中的群组数量作为参考值
        ref_pg_count = _world.group_count

        # 当第二次调用 "dp" 切片时，不应创建新的群组。
        # 因为此时应该只是使用缓存的结果，所以群组数量应该与参考值相同。
        dp_mesh_2 = mesh["dp"]
        self.assertEqual(ref_pg_count, _world.group_count)

        # 当调用 "tp" 切片时，不应创建新的群组，因为 "tp" 切片应该只是重用父网格的群组。
        tp_mesh = mesh["tp"]
        self.assertEqual(_world.group_count, ref_pg_count)
class TestMeshEnv(DTensorTestBase):
    # Test class for testing mesh environment related functionalities

    @with_comms
    def test_get_parent_mesh(self):
        # Test function to verify parent mesh retrieval

        # Define mesh shape and dimension names
        mesh_shape = (2, self.world_size // 2)
        mesh_dim_names = ("DP", "TP")

        # Initialize a 2D device mesh
        mesh_2d = init_device_mesh(
            self.device_type, mesh_shape, mesh_dim_names=mesh_dim_names
        )

        # Assert parent mesh retrieval for DP and TP dimensions
        self.assertEqual(_mesh_resources.get_parent_mesh(mesh_2d["DP"]), mesh_2d)
        self.assertEqual(_mesh_resources.get_parent_mesh(mesh_2d["TP"]), mesh_2d)

        # Create different device meshes
        mesh_0_2 = DeviceMesh(self.device_type, [0, 2])
        mesh_1_3 = DeviceMesh(self.device_type, [1, 3])

        # Assert parent mesh retrieval for DP and TP dimensions
        self.assertEqual(_mesh_resources.get_parent_mesh(mesh_2d["DP"]), mesh_2d)
        self.assertEqual(_mesh_resources.get_parent_mesh(mesh_2d["TP"]), mesh_2d)
        # Assert parent mesh is None for newly created meshes
        self.assertEqual(_mesh_resources.get_parent_mesh(mesh_0_2), None)
        self.assertEqual(_mesh_resources.get_parent_mesh(mesh_1_3), None)

    @with_comms
    def test_get_parent_mesh_dim_exist(self):
        # Test function to verify parent mesh dimension retrieval when dimensions exist

        # Define mesh shape and dimension names
        mesh_shape = (2, self.world_size // 2)
        mesh_dim_names = ("DP", "TP")

        # Initialize a 2D device mesh
        mesh_2d = init_device_mesh(
            self.device_type, mesh_shape, mesh_dim_names=mesh_dim_names
        )

        # Assert parent mesh dimension retrieval for DP and TP dimensions
        self.assertEqual(_mesh_resources.get_parent_mesh_dim(mesh_2d["DP"]), 0)
        self.assertEqual(_mesh_resources.get_parent_mesh_dim(mesh_2d["TP"]), 1)

    @with_comms
    def test_get_parent_mesh_dim_not_exist(self):
        # Test function to verify parent mesh dimension retrieval when dimensions do not exist

        # Define mesh shape
        mesh_shape = (self.world_size,)

        # Initialize a device mesh
        mesh = init_device_mesh(self.device_type, mesh_shape)

        # Assert parent mesh dimension retrieval is None
        self.assertEqual(_mesh_resources.get_parent_mesh_dim(mesh), None)

    @with_comms
    def test_get_mesh_dim_by_name(self):
        # Test function to verify mesh dimension retrieval by name

        # Define mesh shape and dimension names
        mesh_shape = (2, self.world_size // 2)
        mesh_dim_names = ("DP", "TP")

        # Initialize a 2D device mesh
        mesh_2d = init_device_mesh(
            self.device_type, mesh_shape, mesh_dim_names=mesh_dim_names
        )

        # Assert mesh dimension retrieval by name for DP and TP dimensions
        self.assertEqual(_mesh_resources.get_mesh_dim_by_name(mesh_2d, "DP"), 0)
        self.assertEqual(_mesh_resources.get_mesh_dim_by_name(mesh_2d, "TP"), 1)


class DeviceMeshCollectiveTest(DTensorTestBase):
    @property
    def world_size(self):
        return 8

    @with_comms
    def test_broadcast_1d(self):
        # Test function to verify 1D broadcast operation on device mesh

        # Create a device mesh with indices ranging from 0 to 7
        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))

        # Create a local tensor filled with ones multiplied by rank
        local_tensor = torch.ones(3, 3, device=self.device_type) * self.rank

        # Perform mesh broadcast along dimension 0
        mesh_broadcast(local_tensor, mesh, mesh_dim=0)

        # Assert local tensor is broadcasted to zeros
        self.assertEqual(local_tensor, torch.zeros(3, 3))

    @with_comms
    def test_scatter_1d(self):
        # 创建一个设备网格，使用给定设备类型和从0到world_size的序列
        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))
        # 定义scatter_tensor_shape作为三维张量的形状
        scatter_tensor_shape = [3, 3, 3]
        # 遍历scatter_tensor_shape的每个维度
        for scatter_dim in range(len(scatter_tensor_shape)):
            # 创建一个Shard对象，用于scatter_dim维度的分片
            shard_placement = Shard(scatter_dim)
            # 更新scatter_tensor_shape的当前维度，使其乘以world_size
            scatter_tensor_shape[scatter_dim] *= self.world_size
            # 设置随机种子，确保跨rank的随机数生成相同
            torch.manual_seed(0)
            # 创建一个在指定设备上随机初始化的全局张量
            global_tensor = torch.randn(scatter_tensor_shape, device=self.device_type)
            # 使用shard_placement对象的_split_tensor方法，将global_tensor分割为多个片段
            splitted_list, _ = shard_placement._split_tensor(
                global_tensor, mesh.size(), with_padding=True, contiguous=True
            )
            # 创建一个与splitted_list当前rank对应的接收张量
            recv_tensor = torch.empty_like(splitted_list[mesh.get_rank()])
            # 在dim > 0的维度上进行scatter，可能生成非连续的张量，验证其是否有效
            mesh_scatter(recv_tensor, splitted_list, mesh, mesh_dim=0)
            # 断言接收到的张量与splitted_list中当前rank对应的片段相等
            self.assertEqual(recv_tensor, splitted_list[mesh.get_rank()])

    @with_comms
    def test_scatter_uneven(self):
        # 创建一个设备网格，使用给定设备类型和从0到world_size的列表
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        # 获取当前rank
        my_rank = device_mesh.get_rank()
        # 创建一个在指定设备上随机初始化的张量，其形状为device_mesh.size()+3和device_mesh.size()+1
        tensor_to_split = torch.randn(
            device_mesh.size() + 3, device_mesh.size() + 1, device=self.device_type
        )

        # 遍历tensor_to_split的每个维度
        for shard_dim in range(tensor_to_split.ndim):
            # 创建一个Shard对象，用于shard_dim维度的分片
            shard_placement = Shard(shard_dim)

            # 克隆tensor_to_split以防止污染，然后将其分割为self.world_size个块列表
            tensor_to_scatter = tensor_to_split.clone()
            tensor_splitted_list = list(
                torch.chunk(tensor_to_split, self.world_size, dim=shard_dim)
            )
            # 如果分割后的列表长度少于self.world_size，则在末尾添加空张量
            for _ in range(self.world_size - len(tensor_splitted_list)):
                tensor_splitted_list.append(torch.tensor([], device=self.device_type))

            # 使用shard_placement对象的_split_tensor方法，将tensor_to_scatter分割为带填充的列表和填充大小
            padded_tensor_list, pad_sizes = shard_placement._split_tensor(
                tensor_to_scatter,
                device_mesh.size(),
                with_padding=True,
                contiguous=True,
            )

            # 创建一个与padded_tensor_list中当前rank对应的分散张量
            scattered_tensor = torch.empty_like(padded_tensor_list[my_rank])
            # 在dim=0的网格维度上执行scatter操作
            mesh_scatter(scattered_tensor, padded_tensor_list, device_mesh, mesh_dim=0)

            # 如果pad_sizes中当前rank对应的填充大小不为0，则进行反填充操作
            if pad_sizes[my_rank] != 0:
                scattered_tensor = unpad_tensor(
                    scattered_tensor, shard_dim, pad_sizes[my_rank]
                )

            # 如果scattered_tensor的元素数为0，使用numel()方法验证是否与tensor_splitted_list中当前rank对应的张量的元素数相等
            if scattered_tensor.numel() == 0:
                self.assertEqual(
                    scattered_tensor.numel(), tensor_splitted_list[my_rank].numel()
                )
            else:
                # 否则，使用size()方法验证scattered_tensor的大小是否与tensor_splitted_list中当前rank对应的张量的大小相等
                self.assertEqual(
                    scattered_tensor.size(), tensor_splitted_list[my_rank].size()
                )
                # 并使用equal()方法验证scattered_tensor与tensor_splitted_list中当前rank对应的张量是否相等
                self.assertEqual(scattered_tensor, tensor_splitted_list[my_rank])
    # 定义一个测试方法，用于测试在不均匀分布情况下的全局聚合操作
    def test_all_gather_uneven(self):
        # 创建一个设备网格对象，包含指定设备类型和世界大小的设备
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        # 获取当前进程的排名
        my_rank = device_mesh.get_rank()
        # 创建一个全为1的张量，其形状比设备网格大小各增加3和1
        tensor_to_split = torch.ones(
            device_mesh.size() + 3,
            device_mesh.size() + 1,
            device=self.device_type,
        )

        # 遍历张量的每一个维度
        for shard_dim in range(tensor_to_split.ndim):
            # 创建一个分片对象，用于处理当前维度的分片操作
            shard_placement = Shard(shard_dim)
            # 使用分片对象对张量进行分片操作，返回分片后的张量列表和填充尺寸
            tensor_padded_list, pad_sizes = shard_placement._split_tensor(
                tensor_to_split,
                device_mesh.size(),
                with_padding=True,
                contiguous=True,
            )
            # 获取当前进程的本地分片张量
            local_tensor = tensor_padded_list[my_rank]
            # 在指定维度上对本地分片张量进行全局聚合，使用指定的设备网格和组0
            big_tensor = funcol.all_gather_tensor(
                local_tensor, gather_dim=shard_dim, group=(device_mesh, 0)
            )
            # 将大张量按指定维度分割成设备网格大小份
            big_tensor_chunks = list(
                torch.chunk(big_tensor, device_mesh.size(), dim=shard_dim)
            )
            # 创建未填充列表，根据填充尺寸对大张量进行解填充操作
            unpadded_list = [
                (
                    unpad_tensor(big_tensor, shard_dim, pad_sizes[i])
                    if pad_sizes[i] > 0  # 如果存在填充尺寸大于0
                    else big_tensor  # 否则保持大张量不变
                )
                for i, big_tensor in enumerate(big_tensor_chunks)
            ]
            # 将所有解填充后的张量按指定维度进行拼接，形成全局聚合后的张量
            all_gathered_tensor = torch.cat(unpadded_list, dim=shard_dim)

            # 断言全局聚合后的张量形状与原始张量相同
            self.assertEqual(all_gathered_tensor.size(), tensor_to_split.size())
            # 断言全局聚合后的张量与原始张量相等
            self.assertEqual(all_gathered_tensor, tensor_to_split)

    @with_comms
    def test_reduce_scatter_contiguous(self):
        # 创建一个 DeviceMesh 对象，指定设备类型和全局大小的列表，获取当前进程的排名
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        my_rank = device_mesh.get_rank()

        # 初始化一个张量
        step = self.world_size * 2
        total_elem = step**2
        tensor = torch.arange(0, total_elem).view(step, -1).to(device=self.device_type)
        tensor = tensor * (my_rank + 1)

        # 通过切片操作获取一个非连续的张量
        tensor_to_reduce = tensor[::2, :2]
        # 克隆并确保连续性的张量
        tensor_contiguous = tensor_to_reduce.clone().contiguous()

        # 将非连续张量转换为 DTensor 对象，并通过本地操作构造成片段以触发 reduce_scatter
        tensor_to_reduce = DTensor.from_local(
            tensor_to_reduce, device_mesh, [_Partial()]
        )
        # 将连续张量转换为 DTensor 对象，并通过本地操作构造成片段以触发 reduce_scatter
        tensor_contiguous = DTensor.from_local(
            tensor_contiguous, device_mesh, [_Partial()]
        )
        # 对张量进行重新分配，仅保留分片 0，以触发 reduce_scatter
        new_tensor = tensor_to_reduce.redistribute(device_mesh, [Shard(0)])
        # 对连续张量进行重新分配，仅保留分片 0，以触发 reduce_scatter
        new_tensor_contiguous = tensor_contiguous.redistribute(device_mesh, [Shard(0)])

        # 验证非连续和连续张量的 reduce_scatter 值应相同
        new_tensor_local = new_tensor._local_tensor
        new_tensor_contiguous_local = new_tensor_contiguous._local_tensor
        self.assertEqual(new_tensor_local, new_tensor_contiguous_local)
        self.assertEqual(list(new_tensor_local.size()), [1, 2])

        # 检查 reduce 操作后的数值结果
        sum_base = (1 + self.world_size) * self.world_size / 2
        first_elem = my_rank * sum_base * step * 2
        expected_tensor = torch.tensor(
            [[first_elem, first_elem + sum_base]],
            dtype=new_tensor_local.dtype,
            device=self.device_type,
        )
        self.assertEqual(new_tensor_local, expected_tensor)
    # 定义一个测试方法，用于测试不均匀的 reduce scatter 操作
    def test_reduce_scatter_uneven(self):
        # 创建设备网格对象，包括设备类型和整数列表表示的世界大小
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        # 获取当前进程的排名
        my_rank = device_mesh.get_rank()
        # 创建一个张量 tensor_to_split，其形状为设备网格大小加3和加1，并填充为当前进程的排名
        tensor_to_split = (
            torch.ones(
                device_mesh.size() + 3,
                device_mesh.size() + 1,
                device=self.device_type,
            )
            * self.rank
        )

        # 遍历 tensor_to_split 的维度
        for shard_dim in range(tensor_to_split.ndim):
            # 创建一个 Shard 对象，指定切片的维度
            shard_placement = Shard(shard_dim)
            # 克隆 tensor_to_split 到 tensor_to_scatter
            tensor_to_scatter = tensor_to_split.clone()

            # 将 tensor_to_split 沿着 shard_dim 维度分块成 self.world_size 份
            tensor_splitted_list = list(
                torch.chunk(tensor_to_split, self.world_size, dim=shard_dim)
            )
            # 补充空张量，使列表长度达到 self.world_size
            for _ in range(self.world_size - len(tensor_splitted_list)):
                tensor_splitted_list.append(torch.tensor([], device=self.device_type))

            # 使用 shard_placement 的 _split_tensor 方法对 tensor_to_scatter 进行分割，
            # 返回填充后的张量列表和填充大小
            padded_tensor_list, pad_sizes = shard_placement._split_tensor(
                tensor_to_scatter,
                device_mesh.size(),
                with_padding=True,
                contiguous=True,
            )

            # 沿着 shard_dim 维度拼接填充后的张量列表，形成 tensor_to_reduce
            tensor_to_reduce = torch.cat(padded_tensor_list, shard_dim)

            # 计算预期的结果数量 res_num
            res_num = ((0 + self.world_size - 1) * self.world_size) / 2

            # 调用 funcol.reduce_scatter_tensor 函数，对 tensor_to_reduce 执行 reduce scatter 操作
            scattered_tensor = funcol.reduce_scatter_tensor(
                tensor_to_reduce,
                reduceOp="sum",
                scatter_dim=shard_dim,
                group=(device_mesh, 0),
            )

            # 如果当前进程的 pad_sizes 大于 0，则对 scattered_tensor 进行去填充操作
            if pad_sizes[my_rank] > 0:
                scattered_tensor = unpad_tensor(
                    scattered_tensor, shard_dim, pad_sizes[my_rank]
                )

            # 如果 scattered_tensor 的元素数量为 0，则进行断言检查，比较它和 tensor_splitted_list 中对应进程的张量的元素数量
            if scattered_tensor.numel() == 0:
                self.assertEqual(
                    scattered_tensor.numel(), tensor_splitted_list[my_rank].numel()
                )
            else:
                # 否则，比较 scattered_tensor 的形状和 tensor_splitted_list 中对应进程的张量的形状
                self.assertEqual(
                    scattered_tensor.size(), tensor_splitted_list[my_rank].size()
                )
                # 比较 scattered_tensor 和 tensor_splitted_list 中对应进程的张量是否相等，
                # 其值为 torch.ones_like(tensor_splitted_list[my_rank]) * res_num
                self.assertEqual(
                    scattered_tensor,
                    torch.ones_like(tensor_splitted_list[my_rank]) * res_num,
                )

    @with_comms
    def test_broadcast_nd(self):
        # 创建一个3维张量并填充0到7的值，形状为(2, 2, 2)
        mesh_tensor = torch.arange(8).reshape(2, 2, 2)
        # 使用DeviceMesh类创建一个网格对象，并指定设备类型和张量作为网格的数据
        mesh = DeviceMesh(self.device_type, mesh_tensor)
        # 创建一个本地张量，形状为(3, 3)，每个元素为当前进程的排名
        local_tensor = torch.ones(3, 3, device=self.device_type) * self.rank

        # 检查所有维度的分组
        dim_to_subgroups = mesh.get_all_groups()
        # 遍历每个维度及其对应的子组
        for dim, dim_group in enumerate(dim_to_subgroups):
            # 获取当前维度子组的大小
            dim_group_size = get_world_size(dim_group)
            # 获取全局排名列表，包含当前维度子组的所有进程的全局排名
            global_ranks = [
                get_global_rank(dim_group, i) for i in range(dim_group_size)
            ]
            # 克隆本地张量
            cloned_local_tensor = local_tensor.clone()
            # 在网格上进行广播操作，将克隆的本地张量广播到整个维度的子组
            mesh_broadcast(cloned_local_tensor, mesh, mesh_dim=dim)
            # 计算结果的排名
            res_num = global_ranks[0]
            # 断言广播后的克隆本地张量等于全1乘以结果排名的张量
            self.assertEqual(cloned_local_tensor, torch.ones(3, 3) * res_num)

    @with_comms
    def test_scatter_nd(self):
        # 创建一个3维张量并填充0到7的值，形状为(2, 2, 2)
        mesh_tensor = torch.arange(8).reshape(2, 2, 2)
        # 使用DeviceMesh类创建一个网格对象，并指定设备类型和张量作为网格的数据
        mesh = DeviceMesh(self.device_type, mesh_tensor)

        # 检查所有维度的分组
        dim_to_subgroups = mesh.get_all_groups()
        # 遍历每个维度及其对应的子组
        for dim, dim_group in enumerate(dim_to_subgroups):
            # 获取当前维度子组的大小
            dim_group_size = get_world_size(dim_group)
            # 获取全局排名列表，包含当前维度子组的所有进程的全局排名
            global_ranks = [
                get_global_rank(dim_group, i) for i in range(dim_group_size)
            ]
            # 创建一个列表，包含在当前维度子组内每个全局排名对应的全1张量
            scattered_tensors = [
                torch.ones(3, 3, device=self.device_type) * global_rank
                for global_rank in global_ranks
            ]
            # 创建一个空张量，形状与当前进程的坐标对应的分散张量相同
            received_tensor = torch.empty_like(
                scattered_tensors[mesh.get_coordinate()[dim]]
            )
            # 在网格上进行分散操作，将分散张量集合分散到每个维度的子组
            mesh_scatter(received_tensor, scattered_tensors, mesh, mesh_dim=dim)
            # 断言接收的张量等于全1乘以当前进程的排名
            self.assertEqual(received_tensor, torch.ones(3, 3) * self.rank)
# 如果这个模块是直接被运行的（而不是被导入到其他模块中执行），则执行 run_tests() 函数
if __name__ == "__main__":
    run_tests()
```