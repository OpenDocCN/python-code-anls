# `.\pytorch\test\distributed\test_c10d_spawn_gloo.py`

```py
# Owner(s): ["oncall: distributed"]

# 导入必要的库和模块
import copy
import os
import sys
import tempfile

# 导入测试相关模块
import test_c10d_spawn
from test_c10d_spawn import _torch_dist_nn_available, TestDistributedNNFunctions

# 导入PyTorch相关模块
import torch
import torch.distributed as c10d
import torch.nn as nn
from torch.testing._internal.common_cuda import TEST_CUDA, TEST_MULTIGPU
from torch.testing._internal.common_distributed import (
    create_device,
    requires_gloo,
    skip_if_lt_x_gpu,
)
from torch.testing._internal.common_utils import (
    run_tests,
    skip_but_pass_in_sandcastle_if,
    TEST_WITH_DEV_DBG_ASAN,
    TestCase,
)

# 在Python版本低于3.9时，避免执行某些测试用例，详情见GitHub问题链接
if sys.version_info < (3, 9):

    # 单进程分布式数据并行测试类
    class ProcessGroupShareTensorTest(
        test_c10d_spawn.AbstractProcessGroupShareTensorTest, TestCase
    ):

        # 单元测试初始化设置
        def setUp(self):
            self.rank = 0
            self.world_size = 1
            # 创建一个临时文件对象，用于存储测试数据，delete=False表示测试结束后不删除
            self.file = tempfile.NamedTemporaryFile(delete=False)  # noqa: P201

        # 单元测试清理工作
        def tearDown(self):
            try:
                # 尝试删除临时文件
                os.remove(self.file.name)
            except OSError:
                pass

        # 测试基本功能方法
        def _test_base(self, net, inp, check_allclose=True):
            # 使用文件存储创建进程组
            store = c10d.FileStore(self.file.name, self.world_size)
            # 初始化进程组
            c10d.init_process_group(
                backend="gloo", store=store, rank=self.rank, world_size=self.world_size
            )
            # 获取默认的分布式进程组
            process_group = c10d.distributed_c10d._get_default_group()
            # 如果输入数据在CUDA上，获取当前设备ID
            if inp[0].is_cuda:
                device_ids = [torch.cuda.current_device()]
            else:
                device_ids = None

            # 创建分布式数据并行模型
            ddp = nn.parallel.DistributedDataParallel(
                copy.deepcopy(net), device_ids=device_ids, process_group=process_group
            )

            # 定义普通模型和分布式模型的优化器
            net_opt = torch.optim.Adam(net.parameters(), lr=0.001)
            ddp_opt = torch.optim.Adam(ddp.parameters(), lr=0.001)

            # 断言普通模型和分布式模型的参数是否相等
            for i, j in zip(ddp.parameters(), net.parameters()):
                self.assertTrue(i.allclose(j))

            # 进行多次训练迭代
            for _ in range(10):
                # 普通模型和分布式模型的前向传播
                net_out = net(*inp)
                ddp_out = ddp(*inp)

                # 计算普通模型和分布式模型的损失并进行反向传播
                net_out.sum().backward()
                ddp_out.sum().backward()

                # 普通模型和分布式模型的优化器执行优化步骤
                net_opt.step()
                ddp_opt.step()

            # 如果需要检查所有参数的接近性，则再次断言普通模型和分布式模型的参数是否接近
            if check_allclose:
                for i, j in zip(ddp.parameters(), net.parameters()):
                    self.assertTrue(i.allclose(j))

    # 测试在CPU上运行分布式数据并行模型
    @requires_gloo()
    def test_cpu(self):
        self._test_base(nn.Linear(2, 2), [torch.randn(30, 2)])

    # 测试在CUDA上运行分布式数据并行模型
    @requires_gloo()
    @skip_but_pass_in_sandcastle_if(not TEST_CUDA, "At least 1 CUDA GPUS needed")
    def test_cuda(self):
        self._test_base(nn.Linear(2, 2).to(0), [torch.randn(30, 2).to(0)])
    def test_rnn(self):
        # This test is inspired by the bug reported in
        # https://github.com/pytorch/pytorch/issues/36268
        BATCH_SIZE = 12  # Divisible by 2, 3, 4
        INPUT_DIM = 256
        OUTPUT_DIM = 256
        HIDDEN_DIM = 256
        N_LAYERS = 3
        SEQ_LEN = 100

        class Net(nn.Module):
            def __init__(self, input_dim, hidden_dim, output_dim, hidden_layers):
                super().__init__()
                self.input_dim = input_dim  # 设置输入维度
                self.hidden_dim = hidden_dim  # 设置隐藏层维度
                self.output_dim = output_dim  # 设置输出维度
                self.hidden_layers = hidden_layers  # 设置隐藏层数量

                # 创建一个 LSTM 模型
                self.lstm = nn.LSTM(
                    input_dim, hidden_dim, hidden_layers, batch_first=True
                )
                # 创建一个线性层，将隐藏层维度映射到输出维度
                self.h2o = nn.Linear(hidden_dim, output_dim)

            def forward(self, x, y):
                self.lstm.flatten_parameters()  # 将LSTM层的参数扁平化，提高效率
                h_t, _ = self.lstm(x)  # LSTM前向传播，得到隐藏状态h_t
                output = self.h2o(h_t)  # 将隐藏状态映射到输出空间
                loss = nn.functional.mse_loss(output, y)  # 计算均方误差损失
                return loss

        net = Net(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS).to(0)  # 创建一个Net对象，并移动到GPU 0上
        inp = [
            torch.randn((BATCH_SIZE, SEQ_LEN, INPUT_DIM)).to(0),  # 创建一个随机输入张量，并移动到GPU 0上
            torch.rand((BATCH_SIZE, SEQ_LEN, OUTPUT_DIM)).to(0),  # 创建一个随机输出张量，并移动到GPU 0上
        ]

        # 不检查结果的allclose，因为在这次更改之前存在参数不一致的情况。参见＃37079
        self._test_base(net, inp, check_allclose=False)  # 调用基本测试函数，不检查是否全等
# 如果不是在开发调试模式下（即非 dev-asan），则执行以下操作
if not TEST_WITH_DEV_DBG_ASAN:
    # 如果当前脚本是作为主程序执行的
    if __name__ == "__main__":
        # 运行测试函数
        run_tests()
```