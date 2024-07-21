# `.\pytorch\test\distributed\checkpoint\e2e\test_fsdp_ep.py`

```py
# Owner(s): ["oncall: distributed"]

# 导入PyTorch库中的相关模块和类
import torch
import torch.nn as nn
from torch.distributed._tensor import DTensor
from torch.distributed.checkpoint.state_dict import get_state_dict
from torch.distributed.device_mesh import _mesh_resources, init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    skip_if_lt_x_gpu,
    with_comms,
)
from torch.testing._internal.distributed.checkpoint_utils import with_temp_dir
from torch.testing._internal.distributed.common_state_dict import VerifyStateDictMixin


class Dummymodel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        raise NotImplementedError


class EPModel(nn.Module):
    def __init__(self, rank):
        super().__init__()
        # 创建两层神经网络，每层包含线性变换和ReLU激活函数
        self.net1 = nn.Sequential(nn.Linear(16, 16), nn.ReLU())
        self.net2 = nn.Sequential(nn.Linear(16, 16), nn.ReLU())

    def forward(self, x):
        raise NotImplementedError


class SecondTier(nn.Module):
    def __init__(self, rank):
        super().__init__()
        # 创建一个包含多个EPModel或Dummymodel的模块列表
        self.ep_layers = nn.ModuleList(
            [EPModel(rank) if rank % 4 == i else Dummymodel() for i in range(4)]
        )
        # 创建一个包含线性变换和ReLU激活函数的神经网络序列
        self.net = nn.Sequential(nn.Linear(16, 16), nn.ReLU())

    def forward(self, x):
        raise NotImplementedError


class TopModel(nn.Module):
    def __init__(self, rank):
        super().__init__()
        torch.manual_seed(0)

        # 创建一个SecondTier对象
        self.second = SecondTier(rank)
        # 创建一个包含线性变换和ReLU激活函数的神经网络序列
        self.net = nn.Sequential(nn.Linear(16, 16), nn.ReLU())

    def forward(self, x):
        raise NotImplementedError


class TestFSDPWithEP(DTensorTestBase, VerifyStateDictMixin):
    @property
    def world_size(self) -> int:
        # 返回GPU数量和8的最小值作为全局大小
        return min(8, torch.cuda.device_count())

    @with_comms
    @skip_if_lt_x_gpu(8)
    @with_temp_dir
    # 定义一个端对端测试方法，用于测试模型在 CUDA 上的运行
    def test_e2e(self):
        # 创建一个在指定排名上的 TopModel 模型对象，并将其放置在 CUDA 上
        model = TopModel(self.rank).cuda()

        # 初始化一个包含两个维度 (2, 4) 的设备网格 mesh_fsdp_tp，并命名其维度为 ("dp", "tp")
        mesh_fsdp_tp = init_device_mesh(
            self.device_type, (2, 4), mesh_dim_names=("dp", "tp")
        )
        
        # 创建一个子网格 mesh_fsdp_ep，作为 mesh_fsdp_tp 的子网格，仅保留维度 "dp"
        # 注意：这里使用了一个内部 API，未来需要改为公共 API
        mesh_fsdp_ep = _mesh_resources.create_child_mesh(mesh_fsdp_tp, ("dp",))
        
        # 删除 _mesh_resources.child_to_parent_mapping 中的 mesh_fsdp_ep 对应的映射关系
        del _mesh_resources.child_to_parent_mapping[mesh_fsdp_ep]

        # 使用设备类型和维度 (8,) 初始化一个设备网格 mesh_fsdp
        mesh_fsdp = init_device_mesh(self.device_type, (8,))
        
        # 遍历模型中的每个 ep 层，并将其转换为 FSDP 层，使用原始参数和设备网格 mesh_fsdp_ep
        for i, l in enumerate(model.second.ep_layers):
            model.second.ep_layers[i] = FSDP(
                l, use_orig_params=True, device_mesh=mesh_fsdp_ep
            )
        
        # 将 model.second 转换为 FSDP 模型，使用原始参数和设备网格 mesh_fsdp
        model.second = FSDP(model.second, use_orig_params=True, device_mesh=mesh_fsdp)
        
        # 将整个 model 转换为 FSDP 模型，使用原始参数和设备网格 mesh_fsdp
        model = FSDP(model, use_orig_params=True, device_mesh=mesh_fsdp)
        
        # 使用 Adam 优化器优化模型的参数，学习率为 0.1
        optim = torch.optim.Adam(model.parameters(), lr=0.1)
        
        # 获取模型和优化器的状态字典 msd 和 osd
        msd, osd = get_state_dict(model, optim)

        # 验证 FSDP 模型特定参数的设备网格和数据类型
        for key in (
            "net.0.weight",
            "net.0.bias",
            "second.net.0.weight",
            "second.net.0.bias",
        ):
            # 获取 msd 和 osd 中特定键的值
            msd_v = msd[key]
            osd_v = osd["state"][key]["exp_avg"]
            # 断言这些值的类型是 DTensor
            self.assertTrue(isinstance(msd_v, DTensor))
            self.assertTrue(isinstance(osd_v, DTensor))
            # 断言这些值的设备网格是 (0, 1, 2, 3, 4, 5, 6, 7) 的元组
            self.assertEqual(tuple(msd_v.device_mesh.mesh), tuple(range(8)))
            self.assertEqual(tuple(osd_v.device_mesh.mesh), tuple(range(8)))

        # 验证 FSDP/EP 模型特定参数的设备网格和数据类型
        layer = self.rank % 4
        ranks = (layer, layer + 4)
        for i in range(4):
            for key in (
                f"second.ep_layers.{i}.net1.0.weight",
                f"second.ep_layers.{i}.net1.0.bias",
                f"second.ep_layers.{i}.net2.0.weight",
                f"second.ep_layers.{i}.net2.0.bias",
            ):
                if layer != i:
                    # 如果当前层不是目标层，则断言 msd 中不包含该键
                    self.assertTrue(key not in msd)
                else:
                    # 否则获取 msd 和 osd 中特定键的值
                    msd_v = msd[key]
                    osd_v = osd["state"][key]["exp_avg"]
                    # 断言这些值的类型是 DTensor
                    self.assertTrue(isinstance(msd_v, DTensor))
                    self.assertTrue(isinstance(osd_v, DTensor))
                    # 断言这些值的设备网格是 ranks 的元组
                    self.assertEqual(tuple(msd_v.device_mesh.mesh), ranks)
                    self.assertEqual(tuple(osd_v.device_mesh.mesh), ranks)

        # 断言 osd 的状态键集合与 msd 的键集合相同
        self.assertEqual(set(osd["state"].keys()), set(msd.keys()))
# 如果这个脚本是作为主程序运行，执行 run_tests() 函数来运行测试
if __name__ == "__main__":
    run_tests()
```