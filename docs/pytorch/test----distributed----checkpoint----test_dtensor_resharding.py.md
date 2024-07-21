# `.\pytorch\test\distributed\checkpoint\test_dtensor_resharding.py`

```py
# 引入PyTorch库
import torch
# 导入分布式检查点模块
import torch.distributed.checkpoint as dist_cp
# 导入分布式张量相关模块和函数
from torch.distributed._tensor import (
    distribute_tensor,
    init_device_mesh,
    Replicate,
    Shard,
    zeros,
)
# 导入测试工具函数
from torch.testing._internal.common_utils import run_tests
# 导入分布式张量测试相关模块
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    skip_if_lt_x_gpu,
    with_comms,
)
# 导入临时目录管理工具
from torch.testing._internal.distributed.checkpoint_utils import with_temp_dir

# 检查点保存目录
CHECKPOINT_DIR = "checkpoint"

# 一维张量的放置策略列表
ONE_D_PLACEMENTS = [
    [Shard(0)],  # 使用Shard(0)放置
    [Replicate()],  # 复制放置
]

# 一维到一维张量的放置策略对列表
ONE_D_TO_ONE_D_PLACEMENTS = [
    ([Replicate()], [Shard(0)]),  # 从复制放置到Shard(0)放置
    ([Shard(0)], [Replicate()]),  # 从Shard(0)放置到复制放置
]

# 二维张量的放置策略列表
TWO_D_PLACEMENTS = [
    [Replicate(), Replicate()],  # 双重复制放置
    [Replicate(), Shard(0)],  # 一重复制放置一重Shard(0)放置
    [Shard(0), Replicate()],  # 一重Shard(0)放置一重复制放置
    [Shard(0), Shard(0)],  # 双重Shard(0)放置
]

# 二维到二维张量的放置策略对列表
TWO_D_TO_TWO_D_PLACEMENTS = []
for p1 in TWO_D_PLACEMENTS:
    for p2 in TWO_D_PLACEMENTS:
        if p1 != p2:
            TWO_D_TO_TWO_D_PLACEMENTS.append((p1, p2))


class TestDTensorReshardPlacementChange(DTensorTestBase):
    """
    Test DCP reshard for DTensor with placements changes and without world_size change and mesh_tensor change.
    """

    @with_comms  # 使用通信上下文装饰器
    @skip_if_lt_x_gpu(2)  # 如果GPU数目少于2，则跳过测试
    @with_temp_dir  # 使用临时目录装饰器
    # 定义测试方法：测试一维到一维重分布的位置变更
    def test_1d_to_1d_reshard_placement_change(self) -> None:
        # 设置检查点目录为临时目录
        CHECKPOINT_DIR = self.temp_dir

        # 遍历定义的一维到一维重分布的不同位置组合
        for one_d_to_one_d_placements in ONE_D_TO_ONE_D_PLACEMENTS:
            # 获取原始位置和新位置
            original_placement, new_placement = one_d_to_one_d_placements

            # 创建一个全局张量，值为 0 到 15，浮点类型，形状为 4x4
            global_tensor = torch.arange(16, dtype=torch.float).view(4, 4)
            # 设置网格形状为当前世界大小，设备网格初始化
            mesh_shape = (self.world_size,)
            device_mesh = init_device_mesh(self.device_type, mesh_shape)
            # 将全局张量分布到设备网格上，使用给定的原始位置进行分布
            dtensor = distribute_tensor(
                global_tensor, device_mesh, placements=original_placement
            )
            # 准备要保存的状态字典，仅包含分布后的张量
            state_dict_to_save = {"dtensor": dtensor}

            # 使用分布式检查点保存状态字典到文件系统
            dist_cp.save_state_dict(
                state_dict=state_dict_to_save,
                storage_writer=dist_cp.FileSystemWriter(path=CHECKPOINT_DIR),
                planner=dist_cp.DefaultSavePlanner(),
            )

            # 创建一个与全局张量形状相同的零张量，使用新位置分布
            zero_dtensor = zeros(
                [4, 4], device_mesh=device_mesh, placements=new_placement
            )
            # 准备要加载的状态字典，仅包含零张量
            state_dict_to_load = {"dtensor": zero_dtensor}

            # 使用分布式检查点从文件系统加载状态字典
            dist_cp.load_state_dict(
                state_dict=state_dict_to_load,
                storage_reader=dist_cp.FileSystemReader(CHECKPOINT_DIR),
                planner=dist_cp.DefaultLoadPlanner(),
            )

            # 将加载的张量整体实例化到本地，以便与原始全局张量进行比较
            state_dict_to_load["dtensor"] = state_dict_to_load["dtensor"].redistribute(
                device_mesh,
                placements=[Replicate()],
            )
            # 使用断言比较原始全局张量与加载后的本地张量
            self.assertEqual(global_tensor, state_dict_to_load["dtensor"].to_local())

            # 将张量重新分布到其原始位置以便进行比较
            state_dict_to_load["dtensor"] = state_dict_to_load["dtensor"].redistribute(
                device_mesh,
                placements=original_placement,
            )
            # 使用断言比较保存前后的张量是否一致
            self.assertEqual(
                state_dict_to_save["dtensor"].to_local(),
                state_dict_to_load["dtensor"].to_local(),
            )

    @with_comms
    @skip_if_lt_x_gpu(4)
    @with_temp_dir
    # 定义测试方法，用于测试 2D 到 2D 张量重分布和放置变化的情况
    def test_2d_to_2d_reshard_placement_change(self) -> None:
        # 设置检查点目录为临时目录
        CHECKPOINT_DIR = self.temp_dir
        # 遍历预定义的 2D 到 2D 张量放置方案列表
        for two_d_to_two_d_placements in TWO_D_TO_TWO_D_PLACEMENTS:
            # 获取原始和新的张量放置方案
            original_placement, new_placement = two_d_to_two_d_placements

            # 创建全局张量，形状为 4x4，值为从 0 到 15 的浮点数
            global_tensor = torch.arange(16, dtype=torch.float).view(4, 4)
            # 定义网格形状为 (2, self.world_size // 2)，初始化设备网格
            mesh_shape = (2, self.world_size // 2)
            mesh_2d = init_device_mesh(self.device_type, mesh_shape)
            # 在设备网格上分布张量，使用原始放置方案
            dtensor = distribute_tensor(
                global_tensor,
                mesh_2d,
                placements=original_placement,
            )
            # 准备要保存的状态字典，仅包含分布后的张量
            state_dict_to_save = {"dtensor": dtensor}

            # 使用分布检查点库保存状态字典到文件系统，使用指定的检查点目录和默认保存策划
            dist_cp.save_state_dict(
                state_dict=state_dict_to_save,
                storage_writer=dist_cp.FileSystemWriter(path=CHECKPOINT_DIR),
                planner=dist_cp.DefaultSavePlanner(),
            )

            # 创建与零填充的新张量，形状为 [4, 4]，使用新的放置方案
            zero_dtensor = zeros([4, 4], device_mesh=mesh_2d, placements=new_placement)
            # 准备要加载的状态字典，仅包含零填充的新张量
            state_dict_to_load = {"dtensor": zero_dtensor}

            # 使用分布检查点库从文件系统加载状态字典，使用指定的检查点目录和默认加载策划
            dist_cp.load_state_dict(
                state_dict=state_dict_to_load,
                storage_reader=dist_cp.FileSystemReader(CHECKPOINT_DIR),
                planner=dist_cp.DefaultLoadPlanner(),
            )

            # 将加载的张量重新分布到设备网格，使用复制的放置方案 [Replicate(), Replicate()]
            state_dict_to_load["dtensor"] = state_dict_to_load["dtensor"].redistribute(
                mesh_2d,
                placements=[Replicate(), Replicate()],
            )
            # 断言全局张量与加载后的张量在本地设备上的值相等
            self.assertEqual(global_tensor, state_dict_to_load["dtensor"].to_local())

            # 将加载的张量重新分布到设备网格，使用原始放置方案
            state_dict_to_load["dtensor"] = state_dict_to_load["dtensor"].redistribute(
                mesh_2d,
                placements=original_placement,
            )
            # 断言保存前后的张量在本地设备上的值相等
            self.assertEqual(
                state_dict_to_save["dtensor"].to_local(),
                state_dict_to_load["dtensor"].to_local(),
            )
class TestDTensorReshardMeshChange(DTensorTestBase):
    """
    Test DCP reshard for DTensor with placements changes and mesh_tensor change.
    """

    @with_comms
    @with_temp_dir
    @skip_if_lt_x_gpu(2)
    def test_1d_to_2d_reshard_mesh_change(self) -> None:
        # 设置检查点目录为临时目录
        CHECKPOINT_DIR = self.temp_dir
        # 遍历所有一维放置方式
        for placements_1d in ONE_D_PLACEMENTS:
            # 创建一个全局的四行四列的浮点数张量
            global_tensor = torch.arange(16, dtype=torch.float).view(4, 4)
            # 设置网格形状为当前世界尺寸
            mesh_shape = (self.world_size,)
            # 初始化一维设备网格
            mesh_1d = init_device_mesh(self.device_type, mesh_shape)
            # 将全局张量分发到指定的设备上
            dtensor = distribute_tensor(
                global_tensor, mesh_1d, placements=placements_1d
            )
            # 准备要保存的状态字典，包含分布式张量对象
            state_dict_to_save = {"dtensor": dtensor}

            # 使用分布式检查点保存状态字典
            dist_cp.save_state_dict(
                state_dict=state_dict_to_save,
                storage_writer=dist_cp.FileSystemWriter(path=CHECKPOINT_DIR),
                planner=dist_cp.DefaultSavePlanner(),
            )

            # 遍历所有二维放置方式
            for placements_2d in TWO_D_PLACEMENTS:
                # 设置二维网格形状为2行，当前世界尺寸的一半列
                mesh_shape = (2, self.world_size // 2)
                # 初始化二维设备网格
                mesh_2d = init_device_mesh(self.device_type, mesh_shape)

                # 创建一个与全局张量相同形状的零张量
                zero_dtensor = zeros(
                    [4, 4], device_mesh=mesh_2d, placements=placements_2d
                )
                # 准备要加载的状态字典，包含零张量对象
                state_dict_to_load = {"dtensor": zero_dtensor}

                # 使用分布式检查点加载状态字典
                dist_cp.load_state_dict(
                    state_dict=state_dict_to_load,
                    storage_reader=dist_cp.FileSystemReader(CHECKPOINT_DIR),
                    planner=dist_cp.DefaultLoadPlanner(),
                )

                # 将加载的张量在新的二维网格上重新分布，使用复制策略
                state_dict_to_load["dtensor"] = state_dict_to_load[
                    "dtensor"
                ].redistribute(
                    mesh_2d,
                    placements=[Replicate(), Replicate()],
                )
                # 断言重新分布后的张量与原始全局张量相等
                self.assertEqual(
                    global_tensor, state_dict_to_load["dtensor"].to_local()
                )
    # 定义测试方法：test_2d_to_1d_reshard_mesh_change，用于测试二维到一维重分片网格变化
    def test_2d_to_1d_reshard_mesh_change(self) -> None:
        # 设置检查点目录为临时目录
        CHECKPOINT_DIR = self.temp_dir
        # 对于二维放置方式中的每一种方式
        for placements_2d in TWO_D_PLACEMENTS:
            # 创建一个全局的4x4浮点数张量
            global_tensor = torch.arange(16, dtype=torch.float).view(4, 4)
            # 设置网格形状为(2, self.world_size // 2)
            mesh_shape = (2, self.world_size // 2)
            # 根据设备类型和网格形状初始化二维网格
            mesh_2d = init_device_mesh(self.device_type, mesh_shape)
            # 将全局张量在二维网格上分布
            dtensor = distribute_tensor(
                global_tensor, mesh_2d, placements=placements_2d
            )
            # 准备要保存的状态字典，只包含分布的张量
            state_dict_to_save = {"dtensor": dtensor}

            # 使用分布式检查点保存状态字典到文件系统中
            dist_cp.save_state_dict(
                state_dict=state_dict_to_save,
                storage_writer=dist_cp.FileSystemWriter(path=CHECKPOINT_DIR),
                planner=dist_cp.DefaultSavePlanner(),
            )

            # 对于一维放置方式中的每一种方式
            for placements_1d in ONE_D_PLACEMENTS:
                # 设置网格形状为(self.world_size,)
                mesh_shape = (self.world_size,)
                # 根据设备类型和网格形状初始化一维网格
                mesh_1d = init_device_mesh(self.device_type, mesh_shape)
                # 在一维网格上创建全零张量
                zero_dtensor = zeros(
                    [4, 4], device_mesh=mesh_1d, placements=placements_1d
                )
                # 准备要加载的状态字典，只包含全零张量
                state_dict_to_load = {"dtensor": zero_dtensor}

                # 使用分布式检查点加载状态字典从文件系统中
                dist_cp.load_state_dict(
                    state_dict=state_dict_to_load,
                    storage_reader=dist_cp.FileSystemReader(CHECKPOINT_DIR),
                    planner=dist_cp.DefaultLoadPlanner(),
                )

                # 将加载的张量重新分布到一维网格上，使用复制放置方式
                state_dict_to_load["dtensor"] = state_dict_to_load[
                    "dtensor"
                ].redistribute(
                    mesh_1d,
                    placements=[Replicate()],
                )
                # 断言加载的张量与原始全局张量相等（转换为本地张量后）
                self.assertEqual(
                    global_tensor, state_dict_to_load["dtensor"].to_local()
                )

    @with_comms
    @with_temp_dir
    @skip_if_lt_x_gpu(2)
    def test_dtensor_checkpoint_resharding_with_empty_shard(self):
        """
        Test dtensor checkpoint resharding with dtensor containing empty shards.
        """
        # 创建一个在GPU上随机初始化的张量
        tensor = torch.rand(1).cuda()
        # 根据网格形状(self.world_size,)初始化设备网格
        mesh = init_device_mesh(self.device_type, (self.world_size,))
        # 将张量在设备网格上分布，只有一个Shard(0)分片
        dtensor = distribute_tensor(tensor, mesh, [Shard(0)])
        # 准备参考状态字典，只包含分布的张量
        ref_state_dict = {"dtensor": dtensor}

        # 使用分布式检查点保存参考状态字典到临时目录中
        dist_cp.save_state_dict(
            state_dict=ref_state_dict,
            storage_writer=dist_cp.FileSystemWriter(path=self.temp_dir),
        )

        # 创建另一个在GPU上随机初始化的张量
        tensor = torch.rand(1).cuda()
        # 根据网格形状(2, self.world_size // 2)初始化设备网格
        mesh_2 = init_device_mesh(self.device_type, (2, self.world_size // 2))
        # 将张量在设备网格上分布，包含两个Shard(0)分片
        dtensor = distribute_tensor(tensor, mesh_2, [Shard(0), Shard(0)])
        # 准备状态字典，只包含分布的张量
        state_dict = {"dtensor": dtensor}
        
        # 使用分布式检查点加载状态字典从临时目录中
        dist_cp.load_state_dict(
            state_dict=state_dict,
            storage_reader=dist_cp.FileSystemReader(self.temp_dir),
        )

    # TODO: Add a assertEqual for ref_state_dict["dtensor"].full_tensor()
    # and state_dict["dtensor"].full_tensor() after we fix the size mismatch
    # issue for un-even sharding dtensor.
# 如果当前脚本作为主程序执行（而不是被导入到其他模块中执行），则运行测试函数。
if __name__ == "__main__":
    # 调用名为 `run_tests()` 的函数来执行测试。
    run_tests()
```