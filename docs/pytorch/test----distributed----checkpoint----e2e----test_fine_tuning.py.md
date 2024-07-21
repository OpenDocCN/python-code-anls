# `.\pytorch\test\distributed\checkpoint\e2e\test_fine_tuning.py`

```
# Owner(s): ["oncall: distributed"]

# 导入标准库模块
import os
import sys

# 导入PyTorch相关模块
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dist_cp
import torch.nn as nn
from torch.distributed._tensor import init_device_mesh
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    get_state_dict,
    set_model_state_dict,
    set_state_dict,
    StateDictOptions,
)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_utils import run_tests, TEST_WITH_DEV_DBG_ASAN
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)
from torch.testing._internal.distributed.checkpoint_utils import with_temp_dir

# 如果分布式环境不可用，输出消息并退出
if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)

# 如果测试设置为使用 dev-asan，则输出消息并退出，因为 torch + multiprocessing spawn 存在已知问题
if TEST_WITH_DEV_DBG_ASAN:
    print(
        "Skip dev-asan as torch + multiprocessing spawn have known issues",
        file=sys.stderr,
    )
    sys.exit(0)

# 定义维度常量
DIM = 500


class PreTrainedModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # 初始化模型的各层
        self.layer1 = nn.Linear(DIM, DIM)
        self.layer2 = nn.Linear(DIM, DIM)
        self.layer3 = nn.Linear(DIM, DIM)
        self.sequential = nn.Sequential(nn.Linear(DIM, DIM), nn.ReLU())
        self.module_list = nn.ModuleList([nn.Linear(DIM, DIM), nn.ReLU()])
        self.relu = nn.ReLU()

    # 前向传播函数定义
    def forward(self, batch):
        x = self.relu(self.layer1(batch))
        x = self.relu(self.layer2(x))
        x = self.relu(self.layer3(x))
        x = self.sequential(x)
        x = self.module_list[1](self.module_list[0](x))
        return x


class FineTuningModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # 初始化预训练模型及冻结其参数
        self.pretrain = PreTrainedModel()
        for p in self.pretrain.parameters():
            p.requires_grad = False

        # 初始化本地模型的各层
        self.layer1 = nn.Linear(DIM, DIM)
        self.layer2 = nn.Linear(DIM, DIM)
        self.layer3 = nn.Linear(DIM, DIM)
        self.relu = nn.ReLU()

    # 本地模型的前向传播函数定义
    def forward(self, batch):
        x = self.relu(self.pretrain(batch))
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.relu(self.layer3(x))
        return x


class TestFineTuning(DTensorTestBase):
    # 返回分布式测试的进程数
    @property
    def world_size(self) -> int:
        return min(4, torch.cuda.device_count())

    # 返回分布式后端设置
    @property
    def backend(self):
        return "cpu:gloo,cuda:nccl"
    # 定义一个用于预训练的方法，接受一个预训练目录路径作为参数，无返回值
    def pretrain(self, pretrain_dir: str) -> None:
        # 初始化设备网格，根据设备类型和世界大小（self.world_size）
        device_mesh = init_device_mesh(self.device_type, (self.world_size,))

        # 创建一个在 CUDA 设备上的预训练模型实例
        model = PreTrainedModel().cuda()
        
        # 使用混合精度分布式训练（FSDP），将模型放置在设备网格上
        model = FSDP(model, device_mesh=device_mesh)
        
        # 使用 Adam 优化器，设置学习率为 0.001，优化模型的参数
        optim = torch.optim.Adam(model.parameters(), lr=1e-3)

        # 训练过程
        for i in range(3):
            # 生成一个在 CUDA 设备上的随机张量批次，大小为 (32, DIM)
            batch = torch.rand(32, DIM, device="cuda")
            
            # 前向传播，计算模型对批次数据的预测值，并求和作为损失
            loss = model(batch).sum()
            
            # 反向传播，计算损失对模型参数的梯度
            loss.backward()
            
            # 根据梯度更新优化器的参数
            optim.step()
            
            # 清空优化器中的梯度，为下一个迭代做准备
            optim.zero_grad()

        # 保存模型和优化器的状态字典
        model_state_dict, optim_state_dict = get_state_dict(model, optimizers=optim)
        
        # 组装要保存的状态字典，包括模型和优化器的状态
        saved_state_dict = {"model": model_state_dict, "optim": optim_state_dict}
        
        # 使用分布式文件复制工具（dist_cp）保存状态字典到预训练目录
        dist_cp.save_state_dict(
            state_dict=saved_state_dict,
            # 使用文件系统写入器（FileSystemWriter）将状态字典写入预训练目录
            storage_writer=dist_cp.FileSystemWriter(pretrain_dir),
        )
    def finetune(self, pretrain_dir: str, finetune_dir: str) -> None:
        # 初始化设备网格，根据设备类型和世界大小
        device_mesh = init_device_mesh(self.device_type, (self.world_size,))

        # 创建 FineTuningModel 模型实例并将其移到 CUDA 设备上
        model = FineTuningModel().cuda()
        
        # 使用 FSDP 封装模型，增加并行性，使用原始参数，指定设备网格
        model = FSDP(model, use_orig_params=True, device_mesh=device_mesh)
        
        # 使用 Adam 优化器优化模型参数，学习率设为 1e-3
        optim = torch.optim.Adam(model.parameters(), lr=1e-3)

        # 模拟在前三次迭代后重新启动微调
        for i in range(2):
            # 加载预训练子模块的检查点
            pretrain_state_dict = get_model_state_dict(
                model,
                submodules={model.pretrain},
                options=StateDictOptions(keep_submodule_prefixes=False),
            )
            dist_cp.load_state_dict(
                {"model": pretrain_state_dict},
                storage_reader=dist_cp.FileSystemReader(pretrain_dir),
            )
            set_model_state_dict(
                model,
                model_state_dict={model.pretrain: pretrain_state_dict},
                options=StateDictOptions(strict=False),
            )

            try:
                # 加载训练子模块的检查点
                model_state_dict, optim_state_dict = get_state_dict(
                    model,
                    optimizers=optim,
                    options=StateDictOptions(ignore_frozen_params=True),
                )
                dist_cp.load_state_dict(
                    {"model": model_state_dict, "optim": optim_state_dict},
                    storage_reader=dist_cp.FileSystemReader(pretrain_dir),
                )
                set_state_dict(
                    model,
                    optimizers=optim,
                    model_state_dict=model_state_dict,
                    optim_state_dict=optim_state_dict,
                    options=StateDictOptions(strict=False),
                )
            except KeyError:
                # 如果是微调的第一轮，则什么都没有保存
                # 如果是微调的重新启动，则应该存在检查点
                self.assertEqual(i, 0)

            # 训练过程
            for j in range(3):
                # 创建随机数据批次并将其移到 CUDA 设备上
                batch = torch.rand(32, DIM, device="cuda")
                # 计算模型损失并求和
                loss = model(batch).sum()
                # 反向传播损失
                loss.backward()
                # 优化器执行一步优化
                optim.step()
                # 清空梯度
                optim.zero_grad()

            # 保存模型状态字典
            model_state_dict, optim_state_dict = get_state_dict(
                model,
                optimizers=optim,
                options=StateDictOptions(ignore_frozen_params=True),
            )
            saved_state_dict = {"model": model_state_dict, "optim": optim_state_dict}
            dist_cp.save_state_dict(
                state_dict=saved_state_dict,
                storage_writer=dist_cp.FileSystemWriter(finetune_dir),
            )

    @skip_if_lt_x_gpu(4)
    @with_comms
    @with_temp_dir
    # 定义一个测试方法，用于测试模型微调功能，不返回任何值
    def test_fine_tuning(self) -> None:
        # 断言临时目录存在
        self.assertTrue(os.path.exists(self.temp_dir))
        # 在临时目录中创建一个预训练目录路径
        pretrain_dir = os.path.join(self.temp_dir, "pretrain")
        # 在临时目录中创建一个微调目录路径
        finetune_dir = os.path.join(self.temp_dir, "finetune")
        # 打印预训练和微调目录路径
        print(pretrain_dir, finetune_dir)
        # 如果当前进程是主进程（rank == 0）
        if self.rank == 0:
            # 创建预训练目录
            os.mkdir(pretrain_dir)
            # 创建微调目录
            os.mkdir(finetune_dir)
        # 同步所有进程的执行，等待所有进程执行到此处
        dist.barrier()
        # 刷新文件系统缓冲区
        os.sync()
        # 断言预训练目录存在
        self.assertTrue(os.path.exists(pretrain_dir))
        # 断言微调目录存在
        self.assertTrue(os.path.exists(finetune_dir))

        # 调用预训练方法，传入预训练目录路径
        self.pretrain(pretrain_dir)
        # 调用微调方法，传入预训练目录路径和微调目录路径
        self.finetune(pretrain_dir, finetune_dir)
# 如果当前脚本作为主程序执行（而非被导入为模块），则执行 run_tests() 函数
if __name__ == "__main__":
    run_tests()
```