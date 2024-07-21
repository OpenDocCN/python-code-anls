# `.\pytorch\test\distributed\_composable\test_replicate.py`

```
# Owner(s): ["oncall: distributed"]

# 导入必要的库和模块
import os
from copy import deepcopy

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.distributed._composable.replicate import replicate
from torch.testing._internal.common_distributed import (
    MultiProcessTestCase,
    skip_if_lt_x_gpu,
)
from torch.testing._internal.common_utils import run_tests

# 定义神经网络模型
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 2)
        self.fc2 = nn.Linear(2, 2)
        self.fc3 = nn.Linear(2, 2)

    def forward(self, x):
        return self.fc3(self.fc2(self.fc1(x)))

# 测试类，继承自MultiProcessTestCase
class ReplicateStateDictTest(MultiProcessTestCase):

    # 初始化测试环境
    def setUp(self) -> None:
        super().setUp()
        self._spawn_processes()

    # 清理测试环境
    def tearDown(self):
        super().tearDown()
        try:
            os.remove(self.file_name)
        except OSError:
            pass

    # 初始化分布式进程组
    def _init_pg(self):
        dist.init_process_group(
            backend="gloo",
            rank=self.rank,
            world_size=self.world_size,
            store=dist.FileStore(self.file_name, self.world_size),
        )

    # 检查两个状态字典的一致性
    def _check_state_dict_parity(self, sd_1, sd_2):
        for k1, k2 in zip(sd_1.keys(), sd_2.keys()):
            self.assertEqual(k1, k2)

        for v1, v2 in zip(sd_1.values(), sd_2.values()):
            self.assertEqual(v1, v2)

    # 测试单个模块状态字典的复制与加载
    def test_replicate_single_module_save_load(self):
        """
        Tests that replicate() on a single module state_dict
        matches local module state_dict.
        """
        self._init_pg()
        model = Net()
        replicate_model = replicate(deepcopy(model))
        local_sd = model.state_dict()
        ddp_sd = replicate_model.state_dict()
        self._check_state_dict_parity(local_sd, ddp_sd)

    # 测试多个子模块状态字典的复制与加载
    def test_replicate_non_root_multiple_save_load(self):
        """
        Tests tha replicate() on multiple submodules matches
        local module state_dict.
        """
        self._init_pg()
        model = Net()
        replicate_model = deepcopy(model)
        replicate(replicate_model.fc1)
        replicate(replicate_model.fc2)
        replicate(replicate_model.fc3)

        local_sd = model.state_dict()
        ddp_sd = replicate_model.state_dict()
        self._check_state_dict_parity(local_sd, ddp_sd)


class ReplicateTest(MultiProcessTestCase):

    # 定义分布式测试的进程数为2
    @property
    def world_size(self) -> int:
        return 2

    # 初始化测试环境
    def setUp(self) -> None:
        super().setUp()
        self._spawn_processes()

    # 清理测试环境
    def tearDown(self):
        super().tearDown()
        try:
            os.remove(self.file_name)
        except OSError:
            pass

    # 初始化分布式进程组
    def _init_pg(self):
        dist.init_process_group(
            backend="gloo",
            rank=self.rank,
            world_size=self.world_size,
            store=dist.FileStore(self.file_name, self.world_size),
        )
    # 定义一个方法，用于比较两个模型的训练过程
    def _compare_module(self, mod, replicate_mod):
        # 本地批量大小为1
        local_batch_size = 1
        # 全局批量大小为世界大小乘以本地批量大小
        global_batch_size = self.world_size * local_batch_size
        # 生成一个形状为(global_batch_size, 2)的随机张量作为输入
        input = torch.randn(global_batch_size, 2)
        # 生成一个形状为(global_batch_size, 2)的随机张量作为目标输出
        target = torch.randn(global_batch_size, 2)

        # 定义一个内部函数，用于执行模型训练的单步操作
        def step_model(model, input, target):
            # 设置模型为训练模式
            model.train()
            # 使用模型进行前向传播计算输出
            output = model(input)
            # 计算输出与目标之间的均方误差损失
            loss = F.mse_loss(output, target.to(output.device))
            # 反向传播计算梯度
            loss.backward()
            # 对模型的每个参数执行参数更新，使用torch.no_grad()上下文管理器确保不记录梯度信息
            for param in model.parameters():
                with torch.no_grad():
                    param -= param.grad
                param.grad = None

        # 迭代两次进行模型训练
        for iteration in range(2):
            # 使用主模型进行一次训练步骤
            step_model(mod, input, target)
            # 使用复制模型进行一次训练步骤，仅处理本地批量大小的数据
            step_model(
                replicate_mod,
                input[
                    self.rank * local_batch_size : (self.rank + 1) * local_batch_size
                ],
                target[
                    self.rank * local_batch_size : (self.rank + 1) * local_batch_size
                ],
            )

            # 断言主模型和复制模型的参数数量相等
            self.assertEqual(
                len(list(mod.parameters())),
                len(list(replicate_mod.parameters())),
            )
            # 逐个比较主模型和复制模型的每个参数，使用指定的相对和绝对误差容忍度
            for i, j in zip(mod.parameters(), replicate_mod.parameters()):
                self.assertEqual(i, j, rtol=1.3e-06, atol=5e-5)

            # 对输入进行随机重排列，以确保分布式数据并行训练时输入数据不同
            torch.manual_seed(iteration)
            input = input[torch.randperm(global_batch_size)]

    # 测试复制单个模型的功能
    def test_replicate_single_module(self):
        # 初始化进程组
        self._init_pg()
        # 创建一个网络模型
        model = Net()
        # 深度复制模型，得到复制模型
        replicate_model = replicate(deepcopy(model))
        # 比较主模型和复制模型的训练过程
        self._compare_module(model, replicate_model)

    # 跳过如果GPU数小于2的测试装饰器
    @skip_if_lt_x_gpu(2)
    def test_replicate_move_args_kwargs_to_device(self):
        # 定义一个简单的网络模型
        class MyNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.a = nn.Linear(2, 2)

            def forward(self, inp, *, kwarg=None):
                if kwarg is not None:
                    inp = inp @ kwarg
                return self.a(inp)

        # 初始化进程组
        self._init_pg()
        # 设置当前CUDA设备为当前进程的设备
        torch.cuda.set_device(self.rank)
        # 创建一个MyNet模型实例，并将其部署到CUDA设备上
        model = MyNet().cuda()
        # 使用replicate函数将模型复制到指定设备
        replicate(model, device_id=torch.cuda.current_device())
        # 创建两个形状为(2, 2)的随机张量a和b
        a, b = torch.randn(2, 2), torch.randn(2, 2)
        # 调用模型进行前向传播，并计算输出张量的和，然后执行反向传播
        model(a, kwarg=b).sum().backward()

    # 跳过如果GPU数小于2的测试装饰器
    @skip_if_lt_x_gpu(2)
    # 定义一个测试方法，用于验证 replicate 函数在忽略模块的情况下的行为
    def test_replicate_ignore_module(self):
        # 初始化模拟的参数组
        self._init_pg()
        # 设置当前 CUDA 设备为指定的设备
        torch.cuda.set_device(self.rank)
        # 设置随机种子，以确保不同的输入数据在各个进程中具有不同的局部梯度
        torch.manual_seed(self.rank)
        torch.cuda.manual_seed(self.rank)
        # 创建一个 Net 模型，并将其移到 GPU 上
        model = Net().cuda()
        # 使用 replicate 函数复制模型，指定忽略的模块为 model.fc1
        replicate(model, ignored_modules=[model.fc1])
        # 创建一个在 GPU 上的随机输入，其形状为 (5, 2)
        inp = torch.randn(5, 2, device="cuda") * (self.rank + 1)
        # 对模型进行前向传播，并对输出乘以 10
        out = model(inp) * 10
        # 对输出进行求和并计算反向传播
        out.sum().backward()
        # 检查是否 FC1 的梯度不会被同步，但 FC2 和 FC3 应该被同步
        fc1_grad = model.fc1.weight.grad
        # 创建一个与 fc1_grad 形状相同的张量列表
        tensor_list = [torch.zeros_like(fc1_grad) for _ in range(dist.get_world_size())]
        # 使用 all_gather 函数收集各个进程中的 fc1_grad
        dist.all_gather(tensor_list, fc1_grad)
        grad, rest = tensor_list[0], tensor_list[1:]
        # 检查除了第一个进程外的其它进程中的梯度与第一个进程中的梯度不相等
        for g in rest:
            self.assertNotEqual(grad, g)

        # 对于 model.fc2.weight.grad 和 model.fc3.weight.grad，进行相似的操作
        for dp_grad in [model.fc2.weight.grad, model.fc3.weight.grad]:
            # 创建一个与 dp_grad 形状相同的张量列表
            tensor_list = [
                torch.zeros_like(dp_grad) for _ in range(dist.get_world_size())
            ]
            # 使用 all_gather 函数收集各个进程中的 dp_grad
            dist.all_gather(tensor_list, dp_grad)
            grad, rest = tensor_list[0], tensor_list[1:]
            # 检查除了第一个进程外的其它进程中的梯度与第一个进程中的梯度相等
            for g in rest:
                self.assertEqual(grad, g)

    # 定义一个测试方法，用于验证 replicate 函数在复制多个模块的情况下的行为
    def test_replicate_multi_module(self):
        # 初始化模拟的参数组
        self._init_pg()
        # 创建一个 Net 模型
        model = Net()
        # 深拷贝原始模型
        replicate_model = deepcopy(model)
        # 分别使用 replicate 函数复制模型的 fc1、fc2 和 fc3 模块
        replicate(replicate_model.fc1)
        replicate(replicate_model.fc2)
        replicate(replicate_model.fc3)
        # 比较原始模型和复制后的模型
        self._compare_module(model, replicate_model)

    # 定义一个测试方法，用于验证 replicate 函数在使用关键字参数的情况下的行为
    def test_replicate_with_kwargs(self):
        # 初始化模拟的参数组
        self._init_pg()
        # 创建一个 Net 模型
        model = Net()
        # 使用 replicate 函数复制模型，并指定 bucket_cap_mb 和 gradient_as_bucket_view 参数
        replicate_model = replicate(
            deepcopy(model), bucket_cap_mb=1, gradient_as_bucket_view=True
        )
        # 比较原始模型和复制后的模型
        self._compare_module(model, replicate_model)

    # 如果 GPU 数量小于 2，跳过当前测试用例
    @skip_if_lt_x_gpu(2)
    # 定义测试方法，用于测试设备复制功能
    def test_replicate_device_id(self):
        # 初始化测试环境
        self._init_pg()
        # 创建模型对象
        model = Net()
        # 深拷贝模型，并将其移到 CUDA 设备上
        model_cuda = deepcopy(model).cuda()
        # 再次深拷贝 CUDA 模型
        model_cuda2 = deepcopy(model_cuda)
        # 在 CPU 上复制模型
        replicate(model, device_id=torch.device("cpu"))
        # 在第一次前向传播中附加 DDP 实例
        model(torch.randn(2, 2))
        # 获取模型的 DDP 弱引用状态
        replicate_ddp_weakref = replicate.state(model)._ddp_weakref()
        # 对于 CPU 训练，device_ids 应为 None
        self.assertEqual(None, replicate_ddp_weakref.device_ids)

        # 在指定 CUDA 设备上复制 CUDA 模型
        replicate(model_cuda, device_id=torch.device(torch.cuda.current_device()))
        # 在第一次前向传播中附加 DDP 实例
        model_cuda(torch.randn(2, 2))
        # 获取模型的 DDP 弱引用状态
        replicate_ddp_weakref = replicate.state(model_cuda)._ddp_weakref()
        # 验证 device_ids 是否为 [0]
        self.assertEqual([0], replicate_ddp_weakref.device_ids)
        # 以整数形式传入 device_id
        replicate(model_cuda2, device_id=int(torch.cuda.current_device()))
        # 在第一次前向传播中附加 DDP 实例
        model_cuda2(torch.randn(2, 2))
        # 获取模型的 DDP 弱引用状态
        replicate_ddp_weakref = replicate.state(model_cuda2)._ddp_weakref()
        # 验证 device_ids 是否为 [0]
        self.assertEqual([0], replicate_ddp_weakref.device_ids)

    # 测试错误的设备 ID 类型
    def test_replicate_wrong_device_id_type(self):
        # 初始化测试环境
        self._init_pg()
        # 创建模型对象
        model = Net()
        # 断言捕获 RuntimeError，期望 device_id 是 int 或 torch.device 类型
        with self.assertRaisesRegex(
            RuntimeError, "Expected device_id to be int or torch.device"
        ):
            # 使用错误的 device_id 类型进行模型复制
            replicate(model, device_id=[torch.device("cpu")])
# 如果这个脚本是作为主程序执行（而不是被导入到其他脚本中），则运行测试函数
if __name__ == "__main__":
    run_tests()
```