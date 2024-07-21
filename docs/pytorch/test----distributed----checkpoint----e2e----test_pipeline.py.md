# `.\pytorch\test\distributed\checkpoint\e2e\test_pipeline.py`

```
# Owner(s): ["oncall: distributed"]

# 导入必要的库
import os
import sys

import torch
import torch.distributed as dist  # 导入分布式相关模块
import torch.distributed.checkpoint as dcp  # 导入分布式checkpoint相关模块
import torch.nn as nn
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict  # 导入状态字典相关函数
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP  # 导入全分片数据并行模块
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu  # 导入分布式测试相关函数
from torch.testing._internal.common_fsdp import FSDPTest  # 导入FSDP测试类
from torch.testing._internal.common_utils import run_tests, TEST_WITH_DEV_DBG_ASAN  # 导入测试工具函数和测试标志
from torch.testing._internal.distributed.checkpoint_utils import with_temp_dir  # 导入临时目录处理函数

# 如果分布式不可用，则打印消息并退出
if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)

# 如果测试标志 TEST_WITH_DEV_DBG_ASAN 为真，则打印相关消息并退出
if TEST_WITH_DEV_DBG_ASAN:
    print(
        "Skip dev-asan as torch + multiprocessing spawn have known issues",
        file=sys.stderr,
    )
    sys.exit(0)

# 定义模型维度
DIM = 500


class PipelineModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # 定义模型层次结构
        self.layer1 = nn.Linear(DIM, DIM)
        self.layer2 = nn.Linear(DIM, DIM)
        self.layer3 = nn.Linear(DIM, DIM)
        self.layer4 = nn.Linear(DIM, DIM)
        self.relu = nn.ReLU()

    def forward(self, batch):
        # 定义前向传播逻辑
        x = self.relu(self.layer1(batch))
        x = self.relu(self.layer2(x))
        x = self.relu(self.layer3(x))
        x = self.relu(self.layer4(x))
        return x


class TestPipeline(FSDPTest):
    @property
    def world_size(self) -> int:
        # 返回GPU数量和4的较小值作为world_size
        return min(4, torch.cuda.device_count())

    def save_with_pipeline(self, pipeline_dir: str) -> None:
        # 使用元设备上下文创建PipelineModel对象
        with torch.device("meta"):
            model = PipelineModel()

        # 定义管道模块列表
        pipeline_modules = [model.layer1, model.layer2, model.layer3, model.layer4]

        # 将当前进程的模块移到空设备上
        submodule = pipeline_modules[self.rank]
        submodule.to_empty(device=torch.device("cuda"))
        # submodule.reset_parameters()  # 注释掉的重置参数操作

        # 创建优化器
        optim = torch.optim.Adam(submodule.parameters(), lr=1e-3)

        # 忽略训练过程，因为没有真正的管道并行

        # 保存状态字典
        model_state_dict, optim_state_dict = get_state_dict(model, optimizers=optim)
        saved_state_dict = {"model": model_state_dict, "optim": optim_state_dict}
        dcp.save_state_dict(
            state_dict=saved_state_dict,
            storage_writer=dcp.FileSystemWriter(pipeline_dir),
        )
    # 使用 FSDP 加载管道目录中的模型
    def load_with_fsdp(self, pipeline_dir: str) -> None:
        # 使用 FSDP 包装 PipelineModel 并移至 GPU
        model = FSDP(PipelineModel().cuda())
        # 使用 Adam 优化器优化模型参数
        optim = torch.optim.Adam(model.parameters(), lr=1e-3)

        # 加载检查点
        # 获取模型和优化器的状态字典
        model_state_dict, optim_state_dict = get_state_dict(model, optimizers=optim)
        # 使用文件系统读取器从 pipeline_dir 中加载状态字典
        dcp.load_state_dict(
            {"model": model_state_dict, "optim": optim_state_dict},
            storage_reader=dcp.FileSystemReader(pipeline_dir),
        )
        # 设置模型和优化器的状态字典
        set_state_dict(
            model,
            optimizers=optim,
            model_state_dict=model_state_dict,
            optim_state_dict=optim_state_dict,
        )

    # 如果 GPU 数量小于 4，则跳过该测试
    @skip_if_lt_x_gpu(4)
    # 使用临时目录执行测试
    @with_temp_dir
    def test_pipeline(self) -> None:
        # 确保临时目录存在
        self.assertTrue(os.path.exists(self.temp_dir))
        pipeline_dir = os.path.join(self.temp_dir, "pipeline")
        if self.rank == 0:
            # 如果当前进程的排名为 0，则创建 pipeline_dir 目录
            os.mkdir(pipeline_dir)
        os.sync()
        # 同步所有进程
        dist.barrier()
        # 确保 pipeline_dir 目录存在
        self.assertTrue(os.path.exists(pipeline_dir))
        # 在 pipeline_dir 中保存管道
        self.save_with_pipeline(pipeline_dir)
        # 使用 FSDP 加载 pipeline_dir 中的管道
        self.load_with_fsdp(pipeline_dir)
# 如果这个脚本被直接运行（而不是被导入为模块），则执行 run_tests() 函数
if __name__ == "__main__":
    run_tests()
```