# `.\pytorch\test\distributed\_composable\fsdp\test_fully_shard_training.py`

```
# Owner(s): ["oncall: distributed"]

import contextlib  # 导入上下文管理器模块
import copy  # 导入复制模块
import functools  # 导入函数工具模块
import itertools  # 导入迭代器模块
import unittest  # 导入单元测试模块
from typing import Iterable, List, Tuple, Type, Union  # 导入类型提示模块

import torch  # 导入PyTorch库
import torch.distributed as dist  # 导入分布式通信模块
import torch.distributed.checkpoint as dcp  # 导入分布式检查点模块
import torch.nn as nn  # 导入神经网络模块
from torch.distributed._composable import checkpoint, replicate  # 导入分布式可组合模块
from torch.distributed._composable.fsdp import (
    CPUOffloadPolicy,
    FSDPModule,
    fully_shard,
    OffloadPolicy,
    register_fsdp_forward_method,
)  # 导入分布式弹性数据并行模块
from torch.distributed._tensor import DTensor, init_device_mesh  # 导入分布式张量模块
from torch.distributed._tensor.debug.comm_mode import CommDebugMode  # 导入通信调试模式模块
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    _CHECKPOINT_PREFIX,
    apply_activation_checkpointing,
    CheckpointWrapper,
)  # 导入检查点封装模块
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    get_optimizer_state_dict,
)  # 导入状态字典检查点模块
from torch.distributed.device_mesh import DeviceMesh  # 导入设备网格模块
from torch.testing._internal.common_cuda import TEST_CUDA  # 导入CUDA测试模块
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu  # 导入分布式测试模块
from torch.testing._internal.common_fsdp import (
    check_sharded_parity,
    FSDPTest,
    FSDPTestMultiThread,
    MLP,
    MLPStack,
    patch_all_gather,
    patch_reduce_scatter,
    test_compiled_fsdp,
)  # 导入分布式弹性数据并行测试模块
from torch.testing._internal.common_utils import (
    get_cycles_per_ms,
    run_tests,
    skipIfRocm,
    wrapSwapTensorsTest,
)  # 导入常用测试工具模块
from torch.testing._internal.distributed._tensor.common_dtensor import (
    ModelArgs,
    Transformer,
    TransformerBlock,
)  # 导入常见分布式张量模块
from torch.testing._internal.distributed.checkpoint_utils import with_temp_dir  # 导入临时目录装饰器

c10d_ops = torch.ops.c10d  # 设置c10d操作的别名
funcol = torch.ops.c10d_functional  # 设置c10d函数操作的别名


class TestFullyShardForwardInputs(FSDPTestMultiThread):
    @property
    def world_size(self) -> int:
        return 2  # 返回测试的世界大小为2

    @unittest.skipIf(not TEST_CUDA, "no cuda")
    def test_root_move_forward_input_to_device(self):
        device = torch.device("cuda", 0)  # 定义使用的CUDA设备

        class ParamlessModule(nn.Module):
            def forward(self, x: torch.Tensor, ys: Tuple[torch.Tensor, ...]):
                # 检查FSDP是否将输入移动到GPU，包括递归进入元组数据结构
                assert x.device == device, f"Expects {device} but got {x.device}"
                assert (
                    ys[0].device == device
                ), f"Expects {device} but got {ys[0].device}"
                assert (
                    ys[1].device == device
                ), f"Expects {device} but got {ys[1].device}"
                y = ys[0] + ys[1]
                return x + y + 1

        model = ParamlessModule()  # 创建ParamlessModule的实例
        fully_shard(model)  # 对模型进行全面分片
        x = torch.randn((3,))  # 创建一个形状为(3,)的随机张量x
        ys = (torch.randn((3,)), torch.randn((3,)))  # 创建两个形状为(3,)的随机张量组成的元组ys
        self.assertEqual(x.device, torch.device("cpu"))  # 断言x张量在CPU上
        self.assertEqual(ys[0].device, torch.device("cpu"))  # 断言ys元组中第一个张量在CPU上
        self.assertEqual(ys[1].device, torch.device("cpu"))  # 断言ys元组中第二个张量在CPU上
        model(x, ys)  # 调用模型的forward方法，传入x和ys作为输入
# 定义一个名为 TestFullyShardRegisteredParams 的类，它是 FSDPTestMultiThread 的子类
class TestFullyShardRegisteredParams(FSDPTestMultiThread):

    # 定义一个属性 world_size，返回整数值 4
    @property
    def world_size(self) -> int:
        return 4

    # 使用 unittest 模块的 skipIf 装饰器，如果 TEST_CUDA 不为真，则跳过此测试
    @unittest.skipIf(not TEST_CUDA, "no cuda")
    def test_param_registration_after_forward(self):
        """Tests the parameter registration after forward."""
        device = torch.device("cuda", 0)
        # Single FSDP group
        for reshard_after_forward in (True, False, 2):
            torch.manual_seed(42)
            model = MLP(3, device)
            # Since seed is per process, not per thread, we broadcast to ensure
            # the same parameters across ranks
            for param in model.parameters():
                dist.broadcast(param, src=0)
            ref_model = copy.deepcopy(model)
            # Fully shard the model with the specified resharding option
            fully_shard(model, reshard_after_forward=reshard_after_forward)  # root only
            inp = torch.randn((2, 3), device="cuda")
            self._assert_dtensor_params(model.parameters())
            self._assert_same_params(model.parameters(), ref_model.parameters())
            # Perform a forward pass through the model
            model(inp)  # root does not reshard after forward
            self._assert_tensor_params(model.parameters())
            self._assert_same_params(model.parameters(), ref_model.parameters())
            # Manually trigger resharding of the model
            model.reshard()  # however, we can manually reshard
            self._assert_dtensor_params(model.parameters())
            self._assert_same_params(model.parameters(), ref_model.parameters())

        # Multiple FSDP groups
        for reshard_after_forward in (True, False, 2):
            torch.manual_seed(42)
            model = nn.Sequential(MLP(3, device), MLP(3, device))
            for param in model.parameters():
                dist.broadcast(param, src=0)
            ref_model = copy.deepcopy(model)
            # Fully shard each sub-module and the entire model with the specified resharding option
            fully_shard(model[0].in_proj, reshard_after_forward=reshard_after_forward)
            fully_shard(model[0].out_proj, reshard_after_forward=reshard_after_forward)
            fully_shard(model, reshard_after_forward=reshard_after_forward)

            self._assert_dtensor_params(model.parameters())
            self._assert_same_params(model.parameters(), ref_model.parameters())
            model(inp)
            # Differentiate root and non-root parameters for assertion
            non_root_params = list(model[0].in_proj.parameters()) + list(
                model[0].out_proj.parameters()
            )
            root_params = list(set(model.parameters()) - set(non_root_params))
            if reshard_after_forward is False:
                self._assert_tensor_params(non_root_params)
            else:
                self._assert_dtensor_params(non_root_params)
            self._assert_tensor_params(root_params)
            self._assert_same_params(model.parameters(), ref_model.parameters())
            # Manually trigger resharding for each FSDP module in the model
            for module in model.modules():
                if isinstance(module, FSDPModule):
                    module.reshard()  # however, we can manually reshard
            self._assert_dtensor_params(model.parameters())
            self._assert_same_params(model.parameters(), ref_model.parameters())

    @unittest.skipIf(not TEST_CUDA, "no cuda")
    # 定义测试方法，用于验证反向传播后参数注册的行为
    def test_param_registration_after_backward(self):
        """Tests the parameter registration after backward."""
        # 指定使用 CUDA 设备 0
        device = torch.device("cuda", 0)
        # 单个 FSDP 组
        for reshard_after_forward in (True, False, 2):
            # 创建一个包含8个输入节点的MLP模型，并指定设备
            model = MLP(8, device)
            # 对模型进行完全分片，仅在前向传播后重分片
            fully_shard(model, reshard_after_forward=reshard_after_forward)  # root only
            # 生成一个随机张量作为输入
            inp = torch.randn((2, 8), device="cuda")
            # 验证模型参数不包含DTensor类，只包含torch.Tensor
            self._assert_dtensor_params(model.parameters())
            # 对输入进行模型前向传播，并对输出求和进行反向传播
            model(inp).sum().backward()
            # 再次验证模型参数不包含DTensor类，只包含torch.Tensor
            self._assert_dtensor_params(model.parameters())

        # 多个 FSDP 组
        for reshard_after_forward in (True, False, 2):
            # 创建一个包含8个输入节点的MLP模型，并指定设备
            model = MLP(8, device)
            # 对模型的in_proj和out_proj分别进行完全分片，仅在前向传播后重分片
            fully_shard(model.in_proj, reshard_after_forward=reshard_after_forward)
            fully_shard(model.out_proj, reshard_after_forward=reshard_after_forward)
            fully_shard(model, reshard_after_forward=reshard_after_forward)
            # 验证模型参数不包含DTensor类，只包含torch.Tensor
            self._assert_dtensor_params(model.parameters())
            # 对输入进行模型前向传播，并对输出求和进行反向传播
            model(inp).sum().backward()
            # 再次验证模型参数不包含DTensor类，只包含torch.Tensor
            self._assert_dtensor_params(model.parameters())

    # 辅助方法：验证参数列表中至少包含一个torch.Tensor
    def _assert_tensor_params(self, params: Iterable[nn.Parameter]):
        self.assertGreater(len(list(params)), 0)
        for param in params:
            # 断言参数不是DTensor的实例，而是torch.Tensor的实例
            self.assertNotIsInstance(param, DTensor)
            self.assertIsInstance(param, torch.Tensor)

    # 辅助方法：验证参数列表中至少包含一个DTensor
    def _assert_dtensor_params(self, params: Iterable[nn.Parameter]):
        self.assertGreater(len(list(params)), 0)
        for param in params:
            # 断言参数是DTensor的实例
            self.assertIsInstance(param, DTensor)

    # 辅助方法：验证两个参数列表中的每个参数形状和数值是否相等
    def _assert_same_params(
        self, params: Iterable[nn.Parameter], ref_params: Iterable[nn.Parameter]
    ):
        params, ref_params = list(params), list(ref_params)
        self.assertEqual(len(params), len(ref_params))
        for param, ref_param in zip(params, ref_params):
            if isinstance(param, DTensor):
                param = param.full_tensor()
            # 断言两个参数形状和数值完全相同
            self.assertEqual(param.shape, ref_param.shape)
            self.assertEqual(param, ref_param)
# 定义一个测试类 TestFullyShardCastAfterInit，继承自 FSDPTestMultiThread 类
class TestFullyShardCastAfterInit(FSDPTestMultiThread):

    # 定义一个属性方法 world_size，返回整数值 2，指定测试的并行进程数
    @property
    def world_size(self) -> int:
        return 2

    # 使用 unittest 的装饰器 skipIf，如果没有 CUDA 支持则跳过测试
    # wrapSwapTensorsTest 装饰器用于测试特定功能
    @unittest.skipIf(not TEST_CUDA, "no cuda")
    @wrapSwapTensorsTest(True)
    # 定义测试方法 test_to_float64_after_init，测试在初始化后将模型转换为 float64
    def test_to_float64_after_init(self):
        """Tests that the user can cast the module to float64 after init."""
        # 注意：测试选择 fp64 而不是像 bf16 这样的低精度数据类型，以获得更好的数值精度。
        # 关键在于改变数据类型。
        
        # 设置随机种子
        torch.manual_seed(42)
        # 定义 MLP 模型的维度、设备和数据类型
        mlp_dim, device, dtype = 4, torch.device("cuda"), torch.float64
        # 创建一个 MLP 模型对象
        model = MLP(mlp_dim, device=device)
        
        # 对模型的每个参数进行分布式广播，源设备为 0
        for param in model.parameters():
            dist.broadcast(param, src=0)
        
        # 深度复制模型，并将其转换为指定的数据类型 dtype
        ref_model = copy.deepcopy(model).to(dtype)
        # 在模型复制的基础上进行复制
        replicate(ref_model)
        
        # 使用 Adam 优化器初始化复制后的模型的参数
        ref_optim = torch.optim.Adam(ref_model.parameters(), lr=1e-2)
        
        # 对模型的每个模块进行完全分片
        for module in (model.in_proj, model.out_proj, model):
            fully_shard(module)
        
        # 将原始模型转换为指定的数据类型 dtype
        model.to(dtype)
        
        # 验证模型每个参数的数据类型是否为 dtype
        for param in model.parameters():
            self.assertEqual(param.dtype, dtype)
        
        # 使用 foreach=True 参数初始化模型的 Adam 优化器
        optim = torch.optim.Adam(model.parameters(), lr=1e-2, foreach=True)
        
        # 检查分片后模型与参考模型的一致性
        check_sharded_parity(self, ref_model, model)
        
        # 设置随机种子，使其与当前进程的排名相关
        torch.manual_seed(42 + self.rank + 1)
        
        # 在 CUDA 设备上创建指定维度和数据类型的输入张量
        inp = torch.randn((2, mlp_dim), device="cuda", dtype=dtype)
        
        # 进行多次迭代
        for iter_idx in range(10):
            # 初始化损失列表
            losses: List[torch.Tensor] = []
            
            # 遍历参考模型和当前模型
            for _model in (ref_model, model):
                # 计算模型的输出并累加到损失列表中
                losses.append(_model(inp).sum())
                # 反向传播损失
                losses[-1].backward()
            
            # 断言两个模型的损失是否相等
            self.assertEqual(losses[0], losses[1])
            
            # 检查分片后模型与参考模型的一致性
            check_sharded_parity(self, ref_model, model)
            
            # 遍历参考优化器和当前优化器
            for _optim in (ref_optim, optim):
                # 执行优化步骤
                _optim.step()
                # 清空梯度
                _optim.zero_grad(set_to_none=(iter_idx % 2 == 0))


# 定义一个测试类 TestFullyShard1DTrainingCore，继承自 FSDPTest 类
class TestFullyShard1DTrainingCore(FSDPTest):

    # 定义一个属性方法 world_size，返回最小值为 8 和当前 CUDA 设备数的较小值
    @property
    def world_size(self) -> int:
        return min(8, torch.cuda.device_count())

    # 使用 skip_if_lt_x_gpu 装饰器，如果 CUDA 设备数小于 2 则跳过测试
    @skip_if_lt_x_gpu(2)
    # 定义测试方法 test_train_parity_single_group，测试单个 FSDP 组的训练一致性
    def test_train_parity_single_group(self):
        """Tests train parity with DDP for a single FSDP group."""
        # 运行子测试，测试线性层形状
        self.run_subtests(
            {
                "lin_shapes": [[(16, 15), (15, 8)], [(7, 15), (15, 3)]],
            },
            self._test_train_parity_single_group,
        )
    # 定义一个测试函数，用于验证单个参数组的训练一致性
    def _test_train_parity_single_group(self, lin_shapes: List[Tuple[int, int]]):
        # 设置随机种子，以确保结果可重复
        torch.manual_seed(42)
        # 创建包含线性层和ReLU激活函数的神经网络模型
        model = nn.Sequential(
            nn.Linear(*lin_shapes[0]), nn.ReLU(), nn.Linear(*lin_shapes[1])
        )
        # 深度复制模型，并移动到GPU上进行参考
        ref_model = copy.deepcopy(model).cuda()
        # 在指定设备上复制模型
        replicate(ref_model, device_ids=[self.rank])
        # 使用Adam优化器来优化参考模型的参数
        ref_optim = torch.optim.Adam(ref_model.parameters(), lr=1e-2)
        # 对模型进行完全分片
        fully_shard(model)
        # 使用Adam优化器来优化模型的参数
        optim = torch.optim.Adam(model.parameters(), lr=1e-2)
        # 重新设置随机种子，以便每个进程获得不同的输入数据
        torch.manual_seed(42 + self.rank + 1)
        # 准备输入数据，在GPU上创建张量
        inp = (torch.randn((4, lin_shapes[0][0]), device="cuda"),)
        # 迭代多次进行训练
        for iter_idx in range(10):
            # 存储损失的列表
            losses: List[torch.Tensor] = []
            # 对参考模型和当前模型分别进行优化
            for _model, _optim in ((ref_model, ref_optim), (model, optim)):
                # 在每次迭代中，根据条件清零优化器的梯度
                _optim.zero_grad(set_to_none=(iter_idx % 2 == 0))
                # 计算模型的输出并求和，作为损失
                losses.append(_model(*inp).sum())
                # 反向传播损失
                losses[-1].backward()
                # 更新优化器的参数
                _optim.step()
            # 断言两个模型的损失是否相等
            self.assertEqual(losses[0], losses[1])

    # 跳过如果GPU数量小于2的装饰器，用于测试多组参数的训练一致性
    @skip_if_lt_x_gpu(2)
    @test_compiled_fsdp(compile_compute_on_module=Transformer)
    def test_train_parity_multi_group(self):
        """
        Tests train parity against DDP when using multiple parameter groups for
        communication (for communication and computation overlap plus memory
        reduction).
        """
        # 运行子测试，测试多组参数的训练一致性
        self.run_subtests(
            {
                "reshard_after_forward": [True, False, 2],
                "device_type": ["cuda"],
                "offload_policy": [OffloadPolicy()],
                "delay_after_forward": [False, True],
                "delay_before_all_gather": [False, True],
                "delay_before_reduce_scatter": [False, True],
                "delay_before_optim": [False, True],
            },
            self._test_train_parity_multi_group,
        )

    # 跳过如果GPU数量小于2的装饰器，用于测试多组参数的训练一致性和CPU卸载策略
    @skip_if_lt_x_gpu(2)
    def test_train_parity_multi_group_cpu_offload_eager(self):
        """
        Tests train parity against DDP when using multiple parameter groups for
        communication and CPU offloading.
        """
        # 运行子测试，测试多组参数的训练一致性和CPU卸载策略
        self.run_subtests(
            {
                "reshard_after_forward": [True],  # 保存CI时间
                "offload_policy": [
                    CPUOffloadPolicy(pin_memory=True),
                    CPUOffloadPolicy(pin_memory=False),
                ],
                "device_type": ["cuda"],
                "delay_after_forward": [False, True],
                "delay_before_all_gather": [False, True],
                "delay_before_reduce_scatter": [False, True],
                "delay_before_optim": [False, True],
            },
            self._test_train_parity_multi_group,
        )

    # 定义一个测试函数，用于验证多组参数的训练一致性
    def _test_train_parity_multi_group(
        self,
        reshard_after_forward: Union[bool, int],
        offload_policy: OffloadPolicy,
        device_type: str,
        delay_after_forward: bool,
        delay_before_all_gather: bool,
        delay_before_reduce_scatter: bool,
        delay_before_optim: bool,
    # 使用装饰器 `skip_if_lt_x_gpu` 来标记此测试函数，以确保只有 GPU 数量不小于 2 时才运行
    @skip_if_lt_x_gpu(2)
    # 定义一个测试函数，测试在根节点和非根节点上进行前向和后向传播
    def test_non_root_forward_backward(self):
        """
        Tests running forward/backward through the root and then through a
        non-root. The non-root needs to synchronize streams/queue the callback.
        """
        # 设置随机种子为 42
        torch.manual_seed(42)
        # 定义线性层的维度
        lin_dim = 32
        # 创建包含三个 MLP 模型的序列模型，并指定在 CPU 设备上运行
        model = nn.Sequential(*[MLP(lin_dim, torch.device("cpu")) for _ in range(3)])
        # 深拷贝模型到 CUDA 设备上，并作为参考模型
        ref_model = copy.deepcopy(model).cuda()
        # 使用 Adam 优化器优化参考模型的参数
        ref_optim = torch.optim.Adam(ref_model.parameters(), lr=1e-2)
        # 对每个 MLP 模型进行分片处理
        for mlp in model:
            fully_shard(mlp)
        # 对整个模型进行分片处理
        fully_shard(model)
        # 使用 foreach=True 创建 Adam 优化器以优化整个模型的参数
        optim = torch.optim.Adam(model.parameters(), lr=1e-2, foreach=True)
        # 在随机种子的基础上再增加当前进程的排名作为随机种子
        torch.manual_seed(42 + self.rank)
        # 创建在 CUDA 设备上的随机输入张量
        inp = torch.randn((8, lin_dim), device=torch.device("cuda"))
    
        # 计算参考模型在根节点上的损失值并进行反向传播
        ref_root_loss = ref_model(inp).sum()
        ref_root_loss.backward()
        # 对参考模型的参数梯度进行全局归约操作，并调整梯度值
        for param in ref_model.parameters():
            dist.all_reduce(param.grad)
            param.grad.detach().div_(self.world_size)
        # 使用参考模型的优化器执行优化步骤并清零梯度
        ref_optim.step()
        ref_optim.zero_grad()
        
        # 计算参考模型在非根节点上的损失值并进行反向传播
        ref_nonroot_loss = ref_model[0](inp).sum()
        ref_nonroot_loss.backward()
        # 对参考模型的参数梯度进行全局归约操作，并调整梯度值
        for param in ref_model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad)
                param.grad.detach().div_(self.world_size)
        # 使用参考模型的优化器执行优化步骤
        ref_optim.step()
    
        # 计算模型在根节点上的损失值并进行反向传播
        root_loss = model(inp).sum()
        root_loss.backward()
        # 等待一段时间，以确保 CUDA 内核完成当前任务
        torch.cuda._sleep(int(100 * get_cycles_per_ms()))
        # 使用模型的优化器执行优化步骤并清零梯度
        optim.step()
        optim.zero_grad()
    
        # 计算模型在非根节点上的损失值并进行反向传播
        nonroot_loss = model[0](inp).sum()
        nonroot_loss.backward()
        # 使用模型的优化器执行优化步骤
        optim.step()
    
        # 断言参考模型在根节点上的损失值与模型在根节点上的损失值相等
        self.assertEqual(ref_root_loss, root_loss)
        # 断言参考模型在非根节点上的损失值与模型在非根节点上的损失值相等
        self.assertEqual(ref_nonroot_loss, nonroot_loss)
        # 断言参考模型在输入上的损失值与模型在输入上的损失值相等
        self.assertEqual(ref_model(inp).sum(), model(inp).sum())
    
    # 使用装饰器 `skip_if_lt_x_gpu` 标记此测试函数，确保只有 GPU 数量不小于 2 时才运行
    @skip_if_lt_x_gpu(2)
    # 定义一个测试函数，测试运行一个多次参与前向传播的模块时与 DDP 的一致性
    def test_multi_forward_module(self):
        """
        Tests parity with DDP when running a module that participates multiple
        times in forward.
        """
        # 运行子测试，传递测试参数 `reshard_after_forward` 和测试函数 `_test_multi_forward_module`
        self.run_subtests(
            {"reshard_after_forward": [True, False, 2]},
            self._test_multi_forward_module,
        )
    def _test_multi_forward_module(self, reshard_after_forward: Union[bool, int]):
        # 定义一个内部类 MultiForwardModule，继承自 nn.Module，用于测试多次前向传播
        class MultiForwardModule(nn.Module):
            def __init__(self, device: torch.device):
                super().__init__()
                # 创建一个在指定设备上的线性层，输入和输出都是4维
                self.inner = nn.Linear(4, 4, device=device)
                # 创建另一个在指定设备上的线性层，输入为4维，输出为5维
                self.outer = nn.Linear(4, 5, device=device)

            def forward(self, x):
                # 执行两次内部线性层的前向传播
                i = self.inner(x)
                j = self.inner(x)
                # 将两次内部线性层输出的结果相加，并传递给外部线性层进行前向传播
                return self.outer(i + j)

        # 设置随机种子
        torch.manual_seed(42)
        # 创建一个 MultiForwardModule 的实例，使用 CUDA 设备
        model = MultiForwardModule(device="cuda")
        # 使用深度拷贝创建一个参考模型
        ref_model = copy.deepcopy(model)
        # 将参考模型复制到当前设备上
        replicate(ref_model, device_ids=[self.rank])
        # 使用 Adam 优化器初始化参考模型的参数
        ref_optim = torch.optim.Adam(ref_model.parameters(), lr=1e-2)
        # 将模型的内部线性层进行全面分片
        fully_shard(model.inner)
        # 将整个模型进行全面分片
        fully_shard(model)
        # 使用 Adam 优化器初始化模型的参数
        optim = torch.optim.Adam(model.parameters(), lr=1e-2)

        # 设置带有当前 GPU 设备编号的新随机种子
        torch.manual_seed(42 + self.rank)
        # 创建一个输入张量，形状为 (32, 4)，并使用 CUDA 设备
        inp = torch.randn((32, 4), device="cuda")
        # 进行10次迭代
        for iter_idx in range(10):
            # 用于保存损失的列表，每次迭代重新初始化为空列表
            losses: List[torch.Tensor] = []
            # 遍历 (ref_model, ref_optim) 和 (model, optim) 这两对模型和优化器
            for _model, _optim in ((ref_model, ref_optim), (model, optim)):
                # 每次迭代如果 iter_idx 是偶数，则将优化器的梯度清零
                _optim.zero_grad(set_to_none=(iter_idx % 2 == 0))
                # 计算当前模型对输入的预测并求和，将结果添加到损失列表中
                losses.append(_model(inp).sum())
                # 对损失值进行反向传播
                losses[-1].backward()
                # 根据梯度更新模型的参数
                _optim.step()
            # 使用断言确保两个模型在同一次迭代中的损失相等
            self.assertEqual(losses[0], losses[1])

    @skip_if_lt_x_gpu(2)
    # 定义一个测试方法，用于验证显式预取功能
    def test_explicit_prefetching(self):
        # 设置随机种子，确保可复现性
        torch.manual_seed(42)
        # 创建模型参数对象，指定层数和 dropout 概率
        model_args = ModelArgs(n_layers=8, dropout_p=0.0)
        # 创建 Transformer 模型实例
        model = Transformer(model_args)
        # 创建模型的深层副本，并将其复制到 CUDA 设备上
        ref_model = replicate(copy.deepcopy(model).cuda())
        # 创建 AdamW 优化器，并传入深层副本的参数
        ref_optim = torch.optim.AdamW(ref_model.parameters(), lr=1e-2)
        
        # 对模型的每一层及模型本身进行完全分片
        for layer in itertools.chain(model.layers, [model]):
            fully_shard(layer)
        
        # 创建 AdamW 优化器，并传入原始模型的参数
        optim = torch.optim.AdamW(model.parameters(), lr=1e-2)

        # 定义需要向前预取和向后预取的层数
        num_to_forward_prefetch = num_to_backward_prefetch = 2
        
        # 向前预取的层次设置
        for i, layer in enumerate(model.layers):
            if i >= len(model.layers) - num_to_forward_prefetch:
                break
            layers_to_prefetch = [
                model.layers[i + j] for j in range(1, num_to_forward_prefetch + 1)
            ]
            layer.set_modules_to_forward_prefetch(layers_to_prefetch)
        
        # 向后预取的层次设置
        for i, layer in enumerate(model.layers):
            if i < num_to_backward_prefetch:
                continue
            layers_to_prefetch = [
                model.layers[i - j] for j in range(1, num_to_backward_prefetch + 1)
            ]
            layer.set_modules_to_backward_prefetch(layers_to_prefetch)

        # 使用带有特定随机种子和 GPU 设备的随机整数填充输入
        torch.manual_seed(42 + self.rank)
        inp = torch.randint(0, model_args.vocab_size, (2, 8), device="cuda")
        
        # 执行 10 次迭代
        for iter_idx in range(10):
            losses: List[torch.Tensor] = []
            # 针对参考模型和当前模型进行优化步骤
            for _model, _optim in ((ref_model, ref_optim), (model, optim)):
                _optim.zero_grad()
                # 计算损失并添加到列表中
                losses.append(_model(inp).sum())
                # 计算损失的梯度
                losses[-1].backward()
                # 执行优化步骤
                _optim.step()
            # 断言两个模型的损失值相等
            self.assertEqual(losses[0], losses[1])

    # 如果 GPU 数量小于 2，则跳过该测试
    @skip_if_lt_x_gpu(2)
    # 定义测试方法，用于测试后优化事件
    def test_post_optim_event(self):
        # 设置随机种子以确保可重复性
        torch.manual_seed(42)
        # 初始化模型参数
        model_args = ModelArgs(dropout_p=0.0)
        # 创建Transformer模型实例
        model = Transformer(model_args)
        # 深度复制模型并在GPU上复制
        ref_model = replicate(copy.deepcopy(model).cuda())
        # 在复制模型的参数上使用AdamW优化器
        ref_optim = torch.optim.AdamW(ref_model.parameters(), lr=1e-2)
        # 对模型的每一层进行分片处理
        for layer in itertools.chain(model.layers, [model]):
            fully_shard(layer)
        # 使用AdamW优化器来优化模型参数
        optim = torch.optim.AdamW(model.parameters(), lr=1e-2)

        # 定义后优化事件的钩子函数
        def step_post_hook(
            fsdp_module: FSDPModule, opt: torch.optim.Optimizer, args, kwargs
        ) -> None:
            # 记录当前CUDA流的后优化事件
            post_optim_event = torch.cuda.current_stream().record_event()
            fsdp_module.set_post_optim_event(post_optim_event)

        # 在优化器上注册后优化事件的钩子函数
        optim.register_step_post_hook(functools.partial(step_post_hook, model))

        # 重新设置随机种子以确保在不同GPU上运行时结果一致
        torch.manual_seed(42 + self.rank)
        # 在GPU上生成指定范围内的随机整数张量作为输入
        inp = torch.randint(0, model_args.vocab_size, (2, 8), device="cuda")
        # 跟踪所有损失并在结束时检查它们的相等性，以避免在每次迭代后都进行CPU同步点
        ref_losses: List[torch.Tensor] = []
        losses: List[torch.Tensor] = []
        # 执行10次迭代
        for iter_idx in range(10):
            # 清空参考模型优化器的梯度
            ref_optim.zero_grad()
            # 计算参考模型在输入上的预测并累加损失
            ref_losses.append(ref_model(inp).sum())
            # 反向传播损失
            ref_losses[-1].backward()
            # 参考模型优化器执行优化步骤
            ref_optim.step()
        # 执行10次迭代
        for iter_idx in range(10):
            # 清空模型优化器的梯度
            optim.zero_grad()
            # 计算模型在输入上的预测并累加损失
            losses.append(model(inp).sum())
            # 反向传播损失
            losses[-1].backward()
            # 模型优化器执行优化步骤
            optim.step()
            # 在优化器步骤后睡眠，以允许CPU在下一次迭代的前向计算中运行，从而练习后优化流同步
            torch.cuda._sleep(int(25 * get_cycles_per_ms()))
        # 检查参考损失和损失的相等性
        for ref_loss, loss in zip(ref_losses, losses):
            self.assertEqual(ref_loss, loss)
# 定义一个测试类，用于测试完全分片一维训练组合
class TestFullyShard1DTrainingCompose(FSDPTest):
    
    # 返回GPU数量和2的最小值作为世界大小
    @property
    def world_size(self) -> int:
        # 由于这些测试使用更大的Transformer模型，可能在超过2个GPU时出现数值漂移
        return min(torch.cuda.device_count(), 2)

    # 在至少有2个GPU时才运行该测试
    @skip_if_lt_x_gpu(2)
    
    # 使用编译后的FSDP来测试，传入Transformer模块作为计算模块
    @test_compiled_fsdp(compile_compute_on_module=Transformer)
    def test_train_parity_with_activation_checkpointing(self):
        """
        Tests train parity against DDP when composing with activation
        checkpointing.
        """
        # 运行子测试，测试训练时与激活检查点结合的效果
        self.run_subtests(
            {
                "reshard_after_forward": [True, False, 2],
                "checkpoint_impl": ["composable", "utils", "wrapper"],
            },
            self._test_train_parity_with_activation_checkpointing,
        )

    # 测试训练时与激活检查点结合的效果
    def _test_train_parity_with_activation_checkpointing(
        self, reshard_after_forward: Union[bool, int], checkpoint_impl: str
    ):
        pass  # 在后续实现中会定义这个函数

# 定义另一个测试类，用于测试完全分片共享参数的情况
class TestFullyShardSharedParams(FSDPTest):
    
    # 返回4和GPU数量的最小值作为世界大小
    @property
    def world_size(self) -> int:
        return min(4, torch.cuda.device_count())

    # 在至少有2个GPU时才运行该测试
    @skip_if_lt_x_gpu(2)
    def test_train_parity_with_shared_params(self):
        # 运行子测试，测试训练时与共享参数的效果
        self.run_subtests(
            {
                "reshard_after_forward": [False, True],
                "use_activation_checkpointing": [False, True],
            },
            self._test_train_shared_params,
        )

    # 测试训练时与共享参数的效果
    def _test_train_shared_params(
        self,
        reshard_after_forward: bool,
        use_activation_checkpointing: bool,
    ):
        torch.manual_seed(42)
        model_args = ModelArgs(n_layers=3, dropout_p=0.0, weight_tying=True)
        model = Transformer(model_args)
        ref_model = copy.deepcopy(model).cuda()
        replicate(ref_model, device_ids=[self.rank])
        ref_optim = torch.optim.Adam(ref_model.parameters(), lr=1e-2)
        
        # 对每个模块进行迭代，如果是TransformerBlock类型，则根据参数设置是否使用激活检查点
        for module in model.modules():
            if isinstance(module, TransformerBlock):
                if use_activation_checkpointing:
                    checkpoint(module)
                fully_shard(module, reshard_after_forward=reshard_after_forward)
        
        # 对整个模型进行完全分片，根据参数设置是否在前向后重整分片
        fully_shard(model, reshard_after_forward=reshard_after_forward)
        optim = torch.optim.Adam(model.parameters(), lr=1e-2)

        torch.manual_seed(42 + self.rank + 1)
        
        # 进行10次迭代训练
        for iter_idx in range(10):
            inp = torch.randint(0, model_args.vocab_size, (2, 16), device="cuda")
            losses: List[torch.Tensor] = []
            
            # 对每个模型和优化器进行迭代，计算损失并进行反向传播
            for _model, _optim in ((ref_model, ref_optim), (model, optim)):
                _optim.zero_grad(set_to_none=(iter_idx % 2 == 0))
                losses.append(_model(inp).sum())
                losses[-1].backward()
                _optim.step()
            
            # 断言两个模型的损失应该相等
            self.assertEqual(losses[0], losses[1])

# 定义另一个测试类，用于测试完全分片梯度累积的情况
class TestFullyShardGradientAccumulation(FSDPTest):
    
    # 返回4和GPU数量的最小值作为世界大小
    @property
    def world_size(self) -> int:
        return min(4, torch.cuda.device_count())

    # 在至少有2个GPU时才运行该测试
    @skip_if_lt_x_gpu(2)
    def test_gradient_accumulation(self):
        """
        Tests gradient accumulation with/without gradient reduction and
        with/without resharding after backward.
        """
        # 初始化设备网格列表，始终测试 FSDP
        meshes = [init_device_mesh("cuda", (self.world_size,))]
        # 如果有足够的 GPU，也测试 HSDP
        if self.world_size == 4:
            shard_size, replicate_size = 2, 2
            meshes.append(init_device_mesh("cuda", (replicate_size, shard_size)))
        # 运行子测试
        self.run_subtests(
            {
                "mesh": meshes,
                "reshard_after_forward": [True, False, 2],
                # "all": 对所有模块禁用 reduce-scatter
                # "root_only": 仅对根节点的线性层禁用 reduce-scatter
                # "some_mlps": 仅对部分 MLPs 禁用 reduce-scatter
                "mode": ["all", "root_only", "some_mlps"],
                "reshard_after_backward": [False, True],
                "offload_policy": [OffloadPolicy(), CPUOffloadPolicy()],
                # 仅针对 HSDP：
                # `True`: 每个微批次仅使用 reduce-scatter（没有 all-reduce），直到最后一个微批次
                # `False`: 每个微批次既不使用 reduce-scatter 也不使用 all-reduce，直到最后一个微批次
                "reduce_scatter_only": [False, True],
            },
            self._test_gradient_accumulation,
        )

    @skip_if_lt_x_gpu(2)
    def test_1f1b_microbatching(self):
        # 运行子测试
        self.run_subtests(
            {
                "use_explicit_unshard": [False, True],
                "reshard_after_backward": [False, True],
            },
            self._test_1f1b_microbatching,
        )

    def _test_gradient_accumulation(
        self,
        mesh: DeviceMesh,
        reshard_after_forward: Union[bool, int],
        mode: str,
        reshard_after_backward: bool,
        offload_policy: OffloadPolicy,
        reduce_scatter_only: bool,  # 用于 HSDP
    ):
        pass

    def _test_1f1b_microbatching(
        self, use_explicit_unshard: bool, reshard_after_backward: bool
    ):
        pass
        ):
        # 设置随机种子，确保可重复性
        torch.manual_seed(42)
        # 定义模型参数对象，设置 dropout 概率为 0.0
        model_args = ModelArgs(dropout_p=0.0)
        # 创建 Transformer 模型对象
        model = Transformer(model_args)
        # 深拷贝模型，并将其移到 GPU 上
        ref_model = copy.deepcopy(model).cuda()
        # 使用 AdamW 优化器初始化深拷贝模型的参数
        ref_optim = torch.optim.AdamW(ref_model.parameters(), lr=1e-2)
        # 遍历模型中的每个模块，如果是 TransformerBlock 类型则进行分片操作
        for module in model.modules():
            if isinstance(module, TransformerBlock):
                fully_shard(module, reshard_after_forward=False)
        # 对整个模型进行分片操作，不在前向传播后重新分片
        fully_shard(model, reshard_after_forward=False)
        # 使用 AdamW 优化器初始化模型的参数
        optim = torch.optim.AdamW(model.parameters(), lr=1e-2)

        # 定义微批次数量和本地批次大小，并生成输入数据列表
        num_microbatches = 3
        local_batch_size = 2
        # 设置新的随机种子，以确保每个进程有不同的种子
        torch.manual_seed(42 + self.rank + 1)
        inps = [
            torch.randint(
                0, model_args.vocab_size, (local_batch_size, 16), device="cuda"
            )
            for _ in range(num_microbatches)
        ]

        # 如果使用显式取消分片，则遍历模型中的每个模块，取消分片操作
        if use_explicit_unshard:
            for module in model.modules():
                if isinstance(module, FSDPModule):
                    module.unshard(async_op=True)

        # 初始化损失列表和深拷贝模型的损失列表
        losses: List[torch.Tensor] = []
        ref_losses: List[torch.Tensor] = []
        # 遍历输入数据列表中的每个输入数据
        for inp_idx, inp in enumerate(inps):
            # 判断当前是否是最后一个微批次
            is_last_microbatch = inp_idx == num_microbatches - 1
            # 设置模型是否需要同步梯度
            model.set_requires_gradient_sync(is_last_microbatch)
            # 设置是否是最后一个反向传播
            model.set_is_last_backward(is_last_microbatch)
            # 如果不在反向传播后重新分片，则设置不在最后一个反向传播后重新分片
            if not reshard_after_backward:
                model.set_reshard_after_backward(is_last_microbatch)
            # 计算模型对当前输入数据的预测结果的损失并求和
            losses.append(model(inp).sum())
            # 反向传播损失
            losses[-1].backward()
            # 计算深拷贝模型对当前输入数据的预测结果的损失并求和
            ref_losses.append(ref_model(inp).sum())
            # 深拷贝模型反向传播损失
            ref_losses[-1].backward()
        # 对深拷贝模型的所有参数进行全局平均梯度归约
        for param in ref_model.parameters():
            dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)

        # 对损失列表和深拷贝模型的损失列表进行逐一比较
        for loss, ref_loss in zip(losses, ref_losses):
            self.assertEqual(loss, ref_loss)
        # 优化当前模型的参数
        optim.step()
        # 优化深拷贝模型的参数
        ref_optim.step()
        # 检查深拷贝模型和当前模型的分片的一致性
        check_sharded_parity(self, ref_model, model)
class TestFullyShard2DTraining(FSDPTest):
    @property
    def world_size(self) -> int:
        return min(4, torch.cuda.device_count())
    # 定义一个属性，返回当前系统中 GPU 数量和 4 的较小值作为全局尺寸

    def init_global_mesh(self) -> DeviceMesh:
        # 初始化全局网格，根据 GPU 数量选择数据并行尺寸
        dp_size = 2 if self.world_size > 2 else 1
        return init_device_mesh(
            "cuda", (dp_size, self.world_size // dp_size), mesh_dim_names=("dp", "tp")
        )
    
    @skip_if_lt_x_gpu(2)
    @skipIfRocm
    def test_train_parity_2d_mlp(self):
        # 初始化全局网格
        global_mesh = self.init_global_mesh()
        # 运行子测试，测试不同配置下的 MLP 训练一致性
        self.run_subtests(
            {
                "reshard_after_forward": [False, True],
                "use_activation_checkpointing": [False, True],
                "mlp_dim": [3, 16, 17],
            },
            functools.partial(self._test_train_parity_2d_mlp, global_mesh),
        )

    def _test_train_parity_2d_mlp(
        self,
        global_mesh: DeviceMesh,
        reshard_after_forward: bool,
        use_activation_checkpointing: bool,
        mlp_dim: int,
    ):
        # 获取数据并行和模型并行的网格
        dp_mesh, tp_mesh = global_mesh["dp"], global_mesh["tp"]
        # 获取数据并行的过程组，用于复制模型
        dp_pg = dp_mesh.get_group()  # used for `replicate()`

        # 设置随机种子并创建模型及其参考模型
        torch.manual_seed(42)
        model = MLPStack(mlp_dim)
        ref_model = copy.deepcopy(model).cuda()
        # 在数据并行过程组中复制参考模型
        replicate(ref_model, device_ids=[self.rank], process_group=dp_pg)
        # 创建参考模型的优化器
        ref_optim = torch.optim.Adam(ref_model.parameters(), lr=1e-2, foreach=False)
        
        # 并行化模型，使用指定的网格配置和激活检查点选项
        model.parallelize(
            tp_mesh,
            dp_mesh,
            use_activation_checkpointing,
            reshard_after_forward=reshard_after_forward,
        )
        # 创建模型的优化器
        optim = torch.optim.Adam(model.parameters(), lr=1e-2, foreach=False)

        # 设置随机种子并选择设备
        torch.manual_seed(42 + dp_pg.rank() + 1)
        device = torch.device("cuda")
        # 迭代训练过程
        for iter_idx in range(10):
            inp = torch.randn((8, mlp_dim), device=device)
            losses: List[torch.Tensor] = []
            # 对参考模型和并行化后的模型执行训练步骤
            for _model, _optim in ((ref_model, ref_optim), (model, optim)):
                _optim.zero_grad(set_to_none=(iter_idx % 2 == 0))
                losses.append(_model(inp).sum())
                losses[-1].backward()
                _optim.step()
            # 断言两个模型的损失相等
            self.assertEqual(losses[0], losses[1])

    @skip_if_lt_x_gpu(2)
    @skipIfRocm
    # 定义测试函数，用于测试在 FSDP 离线加载时的 TP 并行化
    def test_tp_with_fsdp_offloading(self):
        # 初始化设备网格，使用 CUDA，在 dp 和 tp 维度上定义网格
        global_mesh = init_device_mesh(
            "cuda", (1, self.world_size), mesh_dim_names=("dp", "tp")
        )
        dp_mesh, tp_mesh = global_mesh["dp"], global_mesh["tp"]
        # 设置随机种子
        torch.manual_seed(42)
        # 定义 MLP 的维度
        mlp_dim = 16
        # 创建 MLPStack 模型
        model = MLPStack(mlp_dim)
        # 深拷贝模型，并移动到 CUDA 设备上
        ref_model = copy.deepcopy(model).cuda()
        # 使用 Adam 优化器初始化参考模型的参数
        ref_optim = torch.optim.Adam(ref_model.parameters(), lr=1e-2, foreach=False)
        
        # 在多线程 TP 和单线程 FSDP 的情况下并行化模型
        model.parallelize(
            tp_mesh,
            dp_mesh,
            use_activation_checkpointing=False,
            reshard_after_forward=True,
            offload_policy=CPUOffloadPolicy(),
        )
        
        # 验证模型参数是否在 CPU 上
        for param in model.parameters():
            self.assertEqual(param.device.type, "cpu")
        
        # 统计模型中 MLP 模块的数量
        num_mlps = sum(isinstance(module, MLP) for module in model.modules())
        
        # 使用 Adam 优化器初始化模型的参数
        optim = torch.optim.Adam(model.parameters(), lr=1e-2, foreach=False)

        # NOTE: 我们仍然看到 FSDP 中的所有 gather/reduce-scatter c10d 操作被调用，
        # 但它们只会是无操作，不会发出任何内核。
        # 我们更喜欢在 c10d 层面保持无操作检查，而不是在 FSDP 中。
        
        # 创建输入张量，尺寸为 (4, mlp_dim)，位于 CUDA 设备上
        inp = torch.randn((4, mlp_dim), device="cuda")  # 在所有的 ranks 上都相同
        
        # 执行 10 次迭代
        for iter_idx in range(10):
            # 清空参考模型的梯度
            ref_optim.zero_grad()
            # 清空模型的梯度
            optim.zero_grad()

            # 进入通信调试模式
            with CommDebugMode() as fwd_comm_mode:
                # 计算模型的输出并求和，得到损失值
                loss = model(inp).sum()

            # 获取前向通信统计信息
            fwd_comm_counts = fwd_comm_mode.get_comm_counts()
            self.assertEqual(len(fwd_comm_counts), 2)
            self.assertEqual(fwd_comm_counts[funcol.all_reduce], num_mlps)
            self.assertEqual(fwd_comm_counts[c10d_ops._allgather_base_], num_mlps)
            
            # 计算参考模型的损失值
            ref_loss = ref_model(inp).sum()
            self.assertEqual(loss, ref_loss)

            # 进入通信调试模式
            with CommDebugMode() as bwd_comm_mode:
                # 反向传播计算梯度
                loss.backward()

            # 获取反向通信统计信息
            bwd_comm_counts = bwd_comm_mode.get_comm_counts()
            self.assertEqual(len(bwd_comm_counts), 3)
            # 第一个 MLP 的输入梯度不需要进行 all-reduce 操作
            self.assertEqual(bwd_comm_counts[funcol.all_reduce], num_mlps - 1)
            self.assertEqual(bwd_comm_counts[c10d_ops._allgather_base_], num_mlps)
            self.assertEqual(bwd_comm_counts[c10d_ops._reduce_scatter_base_], num_mlps)
            
            # 对优化器进行更新
            optim.step()
            ref_optim.step()

    @skip_if_lt_x_gpu(2)
    @with_temp_dir
    def test_train_parity_2d_transformer_checkpoint_resume(self):
        """
        Tests train parity of a 2D transformer without checkpointing against a
        2D transformer with a checkpoint save/load.
        """
        # 运行子测试，传入测试参数和测试函数
        self.run_subtests(
            {
                "use_seq_parallel": [False, True],
                # 如果重用模型和优化器，则加载到同一个实例中，否则构建新的实例（需要急切优化状态初始化）
                "reuse_model_optim": [False, True],
                "optimizer_class": [torch.optim.Adam, torch.optim.AdamW],
                # TODO: 在测试之前需要更新 `parallelize`，然后可以包括 foreach=True 进行测试
                "foreach": [False],
            },
            self._test_train_parity_2d_transformer_checkpoint_resume,
        )

    def _test_train_parity_2d_transformer_checkpoint_resume(
        self,
        use_seq_parallel: bool,
        reuse_model_optim: bool,
        optimizer_class: Type[torch.optim.Optimizer],
        foreach: bool,
    ):
        # 实际测试函数，接受参数用于控制测试的不同方面
        # 在这里会执行实际的测试任务，比如比较有/无检查点的训练结果的一致性
        pass
class TestFullyShardNDTraining(FSDPTest):
    # 定义一个测试类，继承自FSDPTest类，用于测试完全分片的ND训练

    @property
    def world_size(self) -> int:
        # 返回GPU数量和8的较小值作为world_size
        return min(8, torch.cuda.device_count())

    def init_global_mesh(self) -> DeviceMesh:
        # 初始化全局设备网格，优先测试使用>=8个GPU，但是对于2个GPU，使用2-way TP
        dp_size = 2 if self.world_size > 2 else 1
        pp_size = 2 if self.world_size > 4 else 1
        return init_device_mesh(
            "cuda",
            (pp_size, dp_size, self.world_size // (dp_size * pp_size)),
            mesh_dim_names=("pp", "dp", "tp"),
        )

    @skip_if_lt_x_gpu(4)
    def test_2d_mlp_with_nd_mesh(self):
        # 如果GPU数量小于4，则跳过测试
        global_mesh = self.init_global_mesh()
        self.run_subtests(
            {
                "reshard_after_forward": [False, True],
                "use_activation_checkpointing": [False, True],
                "mlp_dim": [3, 16, 17],
                "foreach": [False],
            },
            functools.partial(self._test_2d_mlp_with_nd_mesh, global_mesh),
        )

    def _test_2d_mlp_with_nd_mesh(
        self,
        global_mesh: DeviceMesh,
        reshard_after_forward: bool,
        use_activation_checkpointing: bool,
        mlp_dim: int,
        foreach: bool,
    ):
        # 测试2D MLP模型在ND网格上的表现
        global_mesh = self.init_global_mesh()
        pp_mesh, dp_mesh, tp_mesh = (
            global_mesh["pp"],
            global_mesh["dp"],
            global_mesh["tp"],
        )
        dp_pg = dp_mesh.get_group()  # 用于`replicate()`的进程组

        torch.manual_seed(42)
        model = MLPStack(mlp_dim)
        ref_model = copy.deepcopy(model).cuda()
        replicate(ref_model, device_ids=[self.rank], process_group=dp_pg)
        ref_optim = torch.optim.Adam(ref_model.parameters(), lr=1e-2, foreach=foreach)
        model.parallelize(
            tp_mesh,
            dp_mesh,
            use_activation_checkpointing,
            reshard_after_forward=reshard_after_forward,
        )
        optim = torch.optim.Adam(model.parameters(), lr=1e-2, foreach=foreach)

        torch.manual_seed(42 + dp_pg.rank() + 1)
        device = torch.device("cuda")
        for iter_idx in range(10):
            inp = torch.randn((8, mlp_dim), device=device)
            losses: List[torch.Tensor] = []
            for _model, _optim in ((ref_model, ref_optim), (model, optim)):
                _optim.zero_grad(set_to_none=(iter_idx % 2 == 0))
                losses.append(_model(inp).sum())
                losses[-1].backward()
                _optim.step()
            self.assertEqual(losses[0], losses[1])

        for n, p in model.named_parameters():
            self.assertIsInstance(p, DTensor)
            self.assertEqual(p.device_mesh.ndim, 2)
            self.assertEqual(len(p.placements), 2)
            self.assertEqual(p.device_mesh.mesh_dim_names, ("dp", "tp"))


class TestFullyShardHSDPTraining(FSDPTest):
    @property
    def world_size(self) -> int:
        # 返回GPU数量和4的较小值作为world_size
        return min(4, torch.cuda.device_count())

    @skip_if_lt_x_gpu(2)
    # 定义测试函数，用于验证训练的分布式同步和一致性
    def test_train_parity_hsdp(self):
        # 根据世界大小确定每个分片的大小
        shard_size = 2 if self.world_size > 2 else 1
        # 计算复制的数量，即世界大小除以分片大小
        replicate_size = self.world_size // shard_size
        # 初始化全局设备网格，使用 CUDA，设置复制和分片的维度名称
        global_mesh = init_device_mesh(
            "cuda", (replicate_size, shard_size), mesh_dim_names=("replicate", "shard")
        )
        # 运行子测试，传入不同参数组合和部分测试函数
        self.run_subtests(
            {
                "reshard_after_forward": [False, True],  # 是否在前向传播后重新分片
                "use_activation_checkpointing": [False, True],  # 是否使用激活检查点
                "mlp_dim": [3, 16, 17],  # MLP模型的维度
                "sync_gradients_at_last_batch": [True, False],  # 是否在最后一批次同步梯度
            },
            functools.partial(self._test_train_parity_hsdp, global_mesh),  # 部分应用测试函数
        )

    # 实际的测试函数，用于验证训练的分布式同步和一致性
    def _test_train_parity_hsdp(
        self,
        global_mesh: DeviceMesh,
        reshard_after_forward: bool,
        use_activation_checkpointing: bool,
        mlp_dim: int,
        sync_gradients_at_last_batch: bool,
    ):
        # 设置随机种子确保可复现性
        torch.manual_seed(42)
        # 构建模型，包括LayerNorm层和多层感知机（MLP）模型
        model = nn.Sequential(
            nn.LayerNorm(mlp_dim, bias=False),
            MLP(mlp_dim, dim_multiplier=3),
            MLP(mlp_dim),
            MLP(mlp_dim, dim_multiplier=3),
        )
        # 创建模型的深层副本，并复制到指定设备
        ref_model = copy.deepcopy(model).cuda()
        replicate(ref_model, device_ids=[self.rank])  # 复制模型到指定设备
        ref_optim = torch.optim.Adam(ref_model.parameters(), lr=1e-2)  # 创建优化器
        # 对每个MLP模块进行操作
        for mlp in model:
            if use_activation_checkpointing:
                checkpoint(mlp)  # 如果使用激活检查点，则应用检查点
            fully_shard(
                mlp, mesh=global_mesh, reshard_after_forward=reshard_after_forward
            )  # 对MLP模块进行完全分片操作
        fully_shard(
            model, mesh=global_mesh, reshard_after_forward=reshard_after_forward
        )  # 对整个模型进行完全分片操作
        optim = torch.optim.Adam(model.parameters(), lr=1e-2)  # 创建模型的优化器
        check_sharded_parity(self, ref_model, model)  # 检查分片一致性
        # 设置新的随机种子，确保在每个进程中使用不同的种子
        torch.manual_seed(42 + self.rank + 1)
        device = torch.device("cuda")  # 设置设备为CUDA
        num_microbatches = 3  # 定义微批次的数量
        # 迭代进行多次训练循环
        for iter_idx in range(5):
            # 遍历每个微批次
            for microbatch_idx in range(num_microbatches):
                is_last_microbatch = microbatch_idx == num_microbatches - 1  # 判断是否是最后一个微批次
                if sync_gradients_at_last_batch:
                    model.set_requires_gradient_sync(is_last_microbatch)  # 根据参数设置是否在最后一批次同步梯度
                inp = torch.randn((8, mlp_dim), device=device)  # 生成输入张量
                losses: List[torch.Tensor] = []  # 存储损失值的列表
                # 对参考模型和当前模型进行迭代优化
                for _model, _optim in ((ref_model, ref_optim), (model, optim)):
                    losses.append(_model(inp).sum())  # 计算模型输出的总和作为损失
                    losses[-1].backward()  # 反向传播损失
                self.assertEqual(losses[0], losses[1])  # 断言两个模型的损失值相等
            check_sharded_parity(self, ref_model, model)  # 检查分片一致性
            # 对参考模型和当前模型的优化器进行迭代步骤
            for _model, _optim in ((ref_model, ref_optim), (model, optim)):
                _optim.step()  # 执行优化步骤
                _optim.zero_grad(set_to_none=(iter_idx % 2 == 0))  # 清零梯度，根据迭代次数的奇偶性设置是否设置为None
            check_sharded_parity(self, ref_model, model)  # 检查分片一致性
class TestFullyShardCustomForwardMethod(FSDPTest):
    @property
    def world_size(self) -> int:
        # 返回当前可用的 GPU 数量和 2 的较小值，作为世界大小
        return min(torch.cuda.device_count(), 2)

    @skip_if_lt_x_gpu(2)
    def test_register_fsdp_forward_method(self):
        """Based on https://github.com/pytorch/pytorch/issues/109385"""

        class VisionTransformer(nn.Module):
            def __init__(self):
                super().__init__()
                # 定义一个将输入 3 通道图像转换为 1024 通道的卷积层
                self.patch_proj = nn.Conv2d(3, 1024, kernel_size=14, stride=14)

            def forward_features(self, imgs: torch.Tensor) -> torch.Tensor:
                # 对输入图像执行 patch projection，并将结果展平并转置
                return self.patch_proj(imgs).flatten(2).transpose(1, 2)

            def forward(self, imgs: torch.Tensor) -> torch.Tensor:
                # 调用 forward_features 方法并对结果进行求和
                return self.forward_features(imgs).sum(dim=1)

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                # 创建 VisionTransformer 实例和一个线性投影层实例
                self.vit, self.projector = VisionTransformer(), nn.Linear(1024, 256)

            def forward(self, imgs: torch.Tensor) -> torch.Tensor:
                # 运行 VisionTransformer 的 forward_features 方法，注意这不是 forward 方法！
                patch_embeddings = self.vit.forward_features(imgs)
                return self.projector(patch_embeddings)

        torch.manual_seed(42)
        # 创建模型实例并复制到 GPU，以备参考
        model = Model()
        ref_model = copy.deepcopy(model).cuda()
        # 对模型的 vit 和 projector 属性进行分片
        fully_shard(model.vit)
        fully_shard(model.projector)
        fully_shard(model)
        # 注册 fsdp forward 方法到 model.vit 的 forward_features 方法
        register_fsdp_forward_method(model.vit, "forward_features")

        torch.manual_seed(42 + self.rank + 1)
        # 创建输入张量，并将其移动到 GPU
        inp = torch.randn(4, 3, 224, 224, device="cuda")
        # 在参考模型和当前模型上计算损失并求和
        ref_loss = ref_model(inp).sum()
        loss = model(inp).sum()
        # 断言损失值相等
        self.assertEqual(ref_loss, loss)
        # 对参考模型和当前模型的损失进行反向传播
        ref_loss.backward()
        loss.backward()
        # 对所有参数的梯度进行全局平均汇总
        for param in ref_model.parameters():
            dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)
        # 检查分片后的模型是否一致
        check_sharded_parity(self, ref_model, model)


if __name__ == "__main__":
    run_tests()
```