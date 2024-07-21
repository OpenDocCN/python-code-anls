# `.\pytorch\test\distributed\fsdp\test_fsdp_memory.py`

```py
# Owner(s): ["oncall: distributed"]

import sys  # 导入系统模块

import torch  # 导入PyTorch库
import torch.nn as nn  # 导入PyTorch的神经网络模块
import torch.optim as optim  # 导入PyTorch的优化器模块
from torch import distributed as dist  # 导入PyTorch的分布式模块
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP  # 导入FullyShardedDataParallel（FSDP）分布式训练模块
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu  # 导入测试相关的分布式函数
from torch.testing._internal.common_fsdp import FSDPTest  # 导入测试相关的FSDP函数
from torch.testing._internal.common_utils import (  # 导入通用测试函数
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TEST_WITH_DEV_DBG_ASAN,
)
from torch.utils.checkpoint import checkpoint  # 导入模型检查点函数

if not dist.is_available():  # 检查分布式环境是否可用
    print("Distributed not available, skipping tests", file=sys.stderr)  # 输出分布式不可用的提示信息
    sys.exit(0)  # 退出程序

if TEST_WITH_DEV_DBG_ASAN:  # 检查是否需要跳过开发调试ASAN
    print(
        "Skip dev-asan as torch + multiprocessing spawn have known issues",
        file=sys.stderr,
    )  # 输出跳过开发ASAN的提示信息
    sys.exit(0)  # 退出程序


def get_cur_mem(rank, result, prefix):
    """Collect memory allocated values in a result dict in MB"""
    torch._C._cuda_clearCublasWorkspaces()  # 清除CUDA的Cublas工作空间
    result[prefix] = round(torch.cuda.memory_allocated() / 1024 / 1024)  # 记录当前GPU内存占用，并保存到结果字典中


class Model(nn.Module):
    def __init__(self, hidden_dim, with_fsdp=False, with_checkpoint=False):
        super().__init__()  # 调用父类的初始化方法
        if with_fsdp:
            self.stem = nn.Sequential(  # 设置模型的stem部分，包含卷积、批归一化和ReLU激活函数
                nn.Conv2d(3, 64, kernel_size=3),  # 3通道到64通道的卷积层
                FSDP(nn.BatchNorm2d(64)),  # 使用FSDP对BatchNorm2d进行分片
                nn.ReLU(inplace=True),  # 使用inplace方式的ReLU激活函数
            )
        else:
            self.stem = nn.Sequential(  # 设置模型的stem部分，包含卷积、批归一化和ReLU激活函数
                nn.Conv2d(3, 64, kernel_size=3),  # 3通道到64通道的卷积层
                nn.BatchNorm2d(64),  # BatchNorm2d归一化层
                nn.ReLU(inplace=True),  # 使用inplace方式的ReLU激活函数
            )
        if with_fsdp:
            self.blocks = nn.Sequential(  # 设置模型的blocks部分，包含多个卷积、批归一化、ReLU和池化层
                nn.Conv2d(64, hidden_dim, kernel_size=5, padding=2),  # 64通道到hidden_dim通道的卷积层
                FSDP(nn.BatchNorm2d(hidden_dim)),  # 使用FSDP对BatchNorm2d进行分片
                nn.ReLU(inplace=True),  # 使用inplace方式的ReLU激活函数
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=5, padding=2),  # hidden_dim通道到hidden_dim通道的卷积层
                FSDP(nn.BatchNorm2d(hidden_dim)),  # 使用FSDP对BatchNorm2d进行分片
                nn.ReLU(inplace=True),  # 使用inplace方式的ReLU激活函数
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=5, padding=2),  # hidden_dim通道到hidden_dim通道的卷积层
                FSDP(nn.BatchNorm2d(hidden_dim)),  # 使用FSDP对BatchNorm2d进行分片
                nn.ReLU(inplace=True),  # 使用inplace方式的ReLU激活函数
                nn.AdaptiveAvgPool2d(output_size=(1, 1)),  # 自适应平均池化层
                nn.Flatten(),  # 展平层，将多维数据压平成一维
            )
        else:
            self.blocks = nn.Sequential(  # 设置模型的blocks部分，包含多个卷积、批归一化、ReLU和池化层
                nn.Conv2d(64, hidden_dim, kernel_size=5, padding=2),  # 64通道到hidden_dim通道的卷积层
                nn.BatchNorm2d(hidden_dim),  # BatchNorm2d归一化层
                nn.ReLU(inplace=True),  # 使用inplace方式的ReLU激活函数
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=5, padding=2),  # hidden_dim通道到hidden_dim通道的卷积层
                nn.BatchNorm2d(hidden_dim),  # BatchNorm2d归一化层
                nn.ReLU(inplace=True),  # 使用inplace方式的ReLU激活函数
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=5, padding=2),  # hidden_dim通道到hidden_dim通道的卷积层
                nn.BatchNorm2d(hidden_dim),  # BatchNorm2d归一化层
                nn.ReLU(inplace=True),  # 使用inplace方式的ReLU激活函数
                nn.AdaptiveAvgPool2d(output_size=(1, 1)),  # 自适应平均池化层
                nn.Flatten(),  # 展平层，将多维数据压平成一维
            )

        self.head = nn.Linear(hidden_dim, 10)  # 设置模型的头部，全连接层将hidden_dim维度映射到10维度
        self.with_checkpoint = with_checkpoint  # 标志是否使用检查点功能
    # 定义神经网络模型的前向传播函数，接收输入张量 x
    def forward(self, x):
        # 如果指定使用检查点技术
        if self.with_checkpoint:
            # 执行带检查点的前向传播：通过检查点函数处理模块列表和输入 x 的结果，并传递给头部处理
            return self.head(checkpoint(self.blocks, self.stem(x), use_reentrant=True))
        else:
            # 否则，执行普通的前向传播：将输入 x 经过模块列表 blocks 处理，然后传递给头部处理
            return self.head(self.blocks(self.stem(x)))
# 定义一个函数，用于创建模型对象，并设置随机种子为0
def create_model(with_fsdp, with_checkpoint, model_hidden_dim):
    torch.manual_seed(0)
    # 根据参数创建模型对象
    model = Model(model_hidden_dim, with_fsdp, with_checkpoint)
    # 如果指定启用 FSDP，对模型的 stem、blocks 和 head 层应用 FSDP
    if with_fsdp:
        model.stem = FSDP(model.stem)
        model.blocks = FSDP(model.blocks)
        model.head = FSDP(model.head)

    return model


class TestFSDPMemory(FSDPTest):
    @property
    def world_size(self):
        return 2

    # 分布式训练方法，参数包括是否使用 checkpoint、期望结果、模型隐藏层维度、迭代次数
    def _dist_train(self, with_checkpoint, expected, model_hidden_dim, iterations):
        gpu_id = self.rank  # 获取当前 GPU 的 ID
        world_size = self.world_size  # 获取世界大小（即 GPU 数量）

        # 生成一个大小为 (2, 3, 224, 224) 的随机张量，并移到 GPU 上
        batch = torch.randn(size=(2, 3, 224, 224)).cuda()

        # 创建模型对象，启用 FSDP 并移到 GPU 上
        model = create_model(
            with_fsdp=True,
            with_checkpoint=with_checkpoint,
            model_hidden_dim=model_hidden_dim,
        )
        model = model.cuda()
        model = FSDP(model)

        # 设置损失函数为均方误差损失
        criterion = nn.MSELoss()
        # 设置优化器为带有动量的 SGD，学习率为 1e-4
        optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)

        results = {}  # 用于存储内存统计结果的字典
        # 迭代训练循环
        for iteration in range(iterations):
            # 记录当前 GPU 内存使用情况，并添加到结果字典中
            get_cur_mem(gpu_id, results, f"iter {iteration}: start")

            # 模型前向传播
            out = model(batch)
            get_cur_mem(gpu_id, results, f"iter {iteration}: after fwd")

            # 计算输出的和，并生成一个虚假的损失
            out = sum(o.sum() for o in out[0])
            fake_loss = criterion(out, torch.tensor(0.0).cuda())
            get_cur_mem(gpu_id, results, f"iter {iteration}: after loss")

            # 反向传播计算梯度
            fake_loss.backward()
            get_cur_mem(gpu_id, results, f"iter {iteration}: after bwd")

            # 执行优化器的一步参数更新
            optimizer.step()
            get_cur_mem(gpu_id, results, f"iter {iteration}: after step")

            # 使用 `set_to_none` 方法清除模型梯度，释放内存
            model.zero_grad(set_to_none=True)
            get_cur_mem(gpu_id, results, f"iter {iteration}: done")

        # 定义一个比较函数，用于比较实际内存使用情况和预期结果
        def cmp(results, expected):
            ret = ""
            self.assertEqual(results.keys(), expected.keys())
            for k, v in results.items():
                exp = expected[k]
                if abs(exp - v) > 1:  # 允许 1MB 的舍入差异
                    ret += f"{k}: got {v}, expected {exp}\n"
            return ret

        # 进行比较，并断言输出为空，表示测试通过
        output = cmp(results, expected)
        self.assertEqual(output, "")

    # 跳过 GPU 小于 2 的测试用例
    @skip_if_lt_x_gpu(2)
    # 参数化测试用例，包括是否有检查点的两种情况
    @parametrize("ckpt", ["no_ckpt", "ckpt"])
    instantiate_parametrized_tests(TestFSDPMemory)

# 如果运行在主程序中，则执行测试
if __name__ == "__main__":
    run_tests()
```