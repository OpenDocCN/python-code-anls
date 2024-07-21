# `.\pytorch\test\distributed\checkpoint\test_save_load_api.py`

```
# Owner(s): ["oncall: distributed"]

# 导入必要的模块
import os  # 导入操作系统模块
from unittest.mock import patch  # 导入 mock 模块

import torch.distributed.checkpoint as dcp  # 导入分布式检查点模块
import torch.nn as nn  # 导入 PyTorch 神经网络模块
from torch.distributed._tensor.device_mesh import init_device_mesh  # 导入设备网格初始化函数
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP  # 导入 FullyShardedDataParallel 分布式模块
from torch.testing._internal.common_utils import run_tests  # 导入测试运行函数
from torch.testing._internal.distributed._tensor.common_dtensor import (  # 导入分布式测试相关模块
    DTensorTestBase,
    skip_if_lt_x_gpu,
    with_comms,
)
from torch.testing._internal.distributed.checkpoint_utils import with_temp_dir  # 导入临时目录装饰器函数


class MyTestModule(nn.Module):
    def __init__(self):
        super().__init__()
        # 定义多层神经网络
        self.net1 = nn.Sequential(nn.Linear(8, 16), nn.ReLU())
        self.net2 = nn.Sequential(nn.Linear(16, 32), nn.ReLU())
        self.net3 = nn.Linear(32, 64)
        self.net4 = nn.Sequential(nn.ReLU(), nn.Linear(64, 8))

    def forward(self, x):
        # 神经网络前向传播
        return self.net4(self.net3(self.net2(self.net1(x))))


class TestSaveAndLoadAPI(DTensorTestBase):
    @property
    def world_size(self) -> int:
        # 返回测试用例的分布式设置中的进程数
        return 2

    @with_comms  # 使用通信装饰器
    @skip_if_lt_x_gpu(4)  # 如果 GPU 少于 4 个，则跳过测试
    @with_temp_dir  # 使用临时目录装饰器
    def test_auto_detect(self):
        # 创建并初始化分片数据并行模型
        model = FSDP(MyTestModule().cuda())
        # 初始化设备网格
        device_mesh = init_device_mesh(self.device_type, (self.world_size,))
        # 使用设备网格再次初始化分片数据并行模型
        model = FSDP(model, device_mesh=device_mesh)
        # 保存模型状态字典到第一个检查点
        dcp.save(model.state_dict(), checkpoint_id=os.path.join(self.temp_dir, "first"))
        # 从第一个检查点加载模型状态字典
        sd = dcp.load(
            model.state_dict(), checkpoint_id=os.path.join(self.temp_dir, "first")
        )

        # 使用 mock 对象模拟文件系统读取器验证不通过
        with patch.object(
            dcp.FileSystemReader, "validate_checkpoint_id", return_value=False
        ) as m1:
            # 使用 mock 对象模拟文件系统写入器验证不通过
            with patch.object(
                dcp.FileSystemWriter, "validate_checkpoint_id", return_value=False
            ) as m2:
                # 保存模型状态字典到第二个检查点
                dcp.save(
                    model.state_dict(),
                    checkpoint_id=os.path.join(self.temp_dir, "second"),
                )
                # 从第二个检查点加载模型状态字典
                sd = dcp.load(
                    model.state_dict(),
                    checkpoint_id=os.path.join(self.temp_dir, "second"),
                )

        # 检查是否抛出预期的运行时错误（无法检测到指定的检查点）
        with self.assertRaisesRegex(RuntimeError, "Cannot detect"):
            dcp.save(model.state_dict(), checkpoint_id="abc://abc.abc")
        with self.assertRaisesRegex(RuntimeError, "Cannot detect"):
            sd = dcp.load(model.state_dict(), checkpoint_id="abc://abc.abc")


if __name__ == "__main__":
    run_tests()  # 运行测试用例
```