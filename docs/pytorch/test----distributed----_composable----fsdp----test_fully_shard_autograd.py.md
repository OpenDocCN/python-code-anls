# `.\pytorch\test\distributed\_composable\fsdp\test_fully_shard_autograd.py`

```
# Owner(s): ["oncall: distributed"]

# 导入标准库模块
import collections
import copy
import functools
import itertools
import unittest

# 导入第三方库模块
from typing import Any, List, Optional, Type, Union

# 导入 PyTorch 相关模块
import torch
import torch.distributed as dist
import torch.nn as nn

# 导入 PyTorch 分布式训练中使用的特定模块
from torch.distributed._composable.fsdp import fully_shard
from torch.nn.parallel.scatter_gather import _is_namedtuple
from torch.testing._internal.common_cuda import TEST_CUDA
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import (
    check_sharded_parity,
    DoubleLinear,
    FSDPTest,
    FSDPTestMultiThread,
    MLP,
)
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import (
    ModelArgs,
    Transformer,
)

# 定义一个测试类，继承自 FSDPTest 类
class TestFullyShardAutograd(FSDPTest):
    
    # 定义一个属性方法，返回 GPU 数量与 4 的较小值作为 world_size
    @property
    def world_size(self) -> int:
        return min(4, torch.cuda.device_count())

    # 定义一个方法，用于在给定分组中对模型的参数梯度进行归约操作
    def _reduce_1d_partial_grads(
        self, module: nn.Module, group: Optional[dist.ProcessGroup] = None
    ) -> None:
        # 如果分组未提供，则使用默认的分组
        group = group or dist.distributed_c10d._get_default_group()
        # 遍历模型的参数列表，对存在梯度的参数进行均分操作
        for param in module.parameters():
            if param.grad is not None:
                param.grad.div_(group.size())

    # 装饰器函数，如果 GPU 数量小于 2，则跳过测试
    @skip_if_lt_x_gpu(2)
    def test_unused_forward_output(self):
        """
        Tests that gradients propagate when running a backward where some
        forward output is not used to compute the loss, motivated by:
        https://github.com/pytorch/pytorch/pull/83195
        """
        # 运行子测试，测试不同的参数组合
        self.run_subtests(
            {"reshard_after_forward": [True, False, 2]},
            self._test_unused_forward_output,
        )
    # 定义一个测试方法，用于测试未使用前向模块输出的情况
    def _test_unused_forward_output(self, reshard_after_forward: Union[bool, int]):
        # 设定随机种子，保证可重复性
        torch.manual_seed(42)
        # 设置本地批次大小
        local_batch_size = 2
        # 计算全局批次大小和特征维度
        global_batch_size, dim = (self.world_size * local_batch_size, 24)
        # 创建一个双线性模型实例，包含第二个线性层
        model = DoubleLinear(dim=dim, use_second_linear=True)
        # 深度复制模型，移到 CUDA 设备上
        ref_model = copy.deepcopy(model).cuda()
        # 对模型的第一线性层进行全分片处理
        fully_shard(model.lin1, reshard_after_forward=reshard_after_forward)
        # 对整个模型进行全分片处理
        fully_shard(model, reshard_after_forward=reshard_after_forward)
        # 创建参考优化器，针对深度复制的模型参数
        ref_optim = torch.optim.Adam(ref_model.parameters(), lr=1e-2)
        # 创建优化器，针对当前模型参数
        optim = torch.optim.Adam(model.parameters(), lr=1e-2)

        # 设定随机种子，确保每个排名上相同
        torch.manual_seed(1)
        # 执行10次迭代
        for iter_idx in range(10):
            # 在前半个迭代中使用所有前向输出计算损失/反向传播
            # 在后半个迭代中仅使用第一个前向输出计算损失/反向传播
            global_inp = torch.rand((global_batch_size, dim), device="cuda")
            local_inp = global_inp[
                self.rank * local_batch_size : (self.rank + 1) * local_batch_size
            ].detach()
            # 调用模型进行前向传播
            out1, out2 = model(local_inp)
            # 根据迭代索引选择损失计算方式
            loss = (out1 * out2).sum() if iter_idx < 3 else out1.sum()
            # 执行反向传播
            loss.backward()
            # 更新优化器
            optim.step()
            # 调用参考模型进行前向传播
            ref_out1, ref_out2 = ref_model(global_inp)
            # 根据迭代索引选择参考损失计算方式
            ref_loss = (ref_out1 * ref_out2).sum() if iter_idx < 3 else ref_out1.sum()
            # 执行参考模型的反向传播
            ref_loss.backward()
            # 对参考模型进行一维部分梯度规约
            self._reduce_1d_partial_grads(ref_model)
            # 更新参考优化器
            ref_optim.step()
            # 对损失进行全局规约
            dist.all_reduce(loss)  # partial -> replicated
            # 断言当前模型的损失与参考模型的损失相等
            self.assertEqual(loss, ref_loss)
            # 清空优化器的梯度，根据迭代索引选择设置为 None
            optim.zero_grad(set_to_none=(iter_idx % 2))
            ref_optim.zero_grad(set_to_none=(iter_idx % 2))
            # 检查全分片处理后模型的一致性
            check_sharded_parity(self, ref_model, model)

    # 跳过测试条件小于两个 GPU 的情况
    @skip_if_lt_x_gpu(2)
    def test_unused_forward_module(self):
        """
        测试当运行反向传播时，某些前向模块未用于计算损失时梯度是否正确传播。
        受启发于：https://github.com/pytorch/pytorch/pull/80245
        """
        # 运行子测试，测试不同的 reshard_after_forward 参数
        self.run_subtests(
            {"reshard_after_forward": [True, False, 2]},
            self._test_unused_forward_module,
        )
    # 定义一个测试方法，用于测试未使用的前向模块，并接受一个布尔型或整型参数，指示是否在前向传播后重新分片
    def _test_unused_forward_module(self, reshard_after_forward: Union[bool, int]):
        # 设置随机种子，确保结果可重复
        torch.manual_seed(42)
        # 定义本地批大小和维度
        local_batch_size, dim = (2, 24)
        # 计算全局批大小
        global_batch_size = self.world_size * local_batch_size
        # 创建一个 DoubleLinear 模型对象，不使用第二个线性层
        model = DoubleLinear(dim=dim, use_second_linear=False)
        # 深度复制模型并将其移到 CUDA 设备上
        ref_model = copy.deepcopy(model).cuda()
        # 对模型的第一个线性层进行完全分片
        fully_shard(model.lin1, reshard_after_forward=reshard_after_forward)
        # 对模型的第二个线性层进行完全分片
        fully_shard(model.lin2, reshard_after_forward=reshard_after_forward)
        # 对整个模型进行完全分片
        fully_shard(model, reshard_after_forward=reshard_after_forward)
        # 创建 Adam 优化器并绑定到深度复制模型的参数上
        ref_optim = torch.optim.Adam(ref_model.parameters(), lr=1e-2)
        # 创建 Adam 优化器并绑定到模型的参数上
        optim = torch.optim.Adam(model.parameters(), lr=1e-2)

        # 设置随机种子，确保结果可重复，同样适用于所有排名
        torch.manual_seed(1)
        # 执行 10 次迭代
        for iter_idx in range(10):
            # 创建一个 CUDA 设备上的随机输入张量
            global_inp = torch.rand((global_batch_size, dim), device="cuda")
            # 提取本地输入张量，根据当前进程的排名
            local_inp = global_inp[
                self.rank * local_batch_size : (self.rank + 1) * local_batch_size
            ].detach()
            # 初始化损失列表
            losses: List[torch.Tensor] = []
            # 针对深度复制模型和本地输入执行前向传播并计算损失
            for _model, inp in ((ref_model, global_inp), (model, local_inp)):
                losses.append(_model(inp).sum())
                # 反向传播损失
                losses[-1].backward()
            # 在深度复制模型上执行一维部分梯度的归约操作
            self._reduce_1d_partial_grads(ref_model)
            # 在所有进程上执行全部梯度的归约操作
            dist.all_reduce(losses[1])  # partial -> replicated
            # 断言两个损失张量相等
            self.assertEqual(losses[0], losses[1])
            # 检查分片的一致性
            check_sharded_parity(self, ref_model, model)
            # 在优化器上执行一步优化，并在迭代索引为偶数时将梯度清零
            for _optim in (optim, ref_optim):
                _optim.step()
                _optim.zero_grad(set_to_none=(iter_idx % 2))

    # 如果 GPU 数量小于 2，则跳过该测试
    @skip_if_lt_x_gpu(2)
    def test_nontensor_activations(self):
        """
        Tests that gradients propagate when running forward with nontensor
        data structures wrapping the activations. This is mainly to test the
        hook registration.
        """
        # 执行子测试，测试在使用非张量数据结构包装激活函数时，梯度是否正确传播
        self.run_subtests(
            {"container_type": [list, collections.namedtuple, tuple, dict]},
            self._test_nontensor_activations,
        )
# 定义一个测试类，继承自FSDPTestMultiThread类，用于测试多线程环境下的完全分片后累积梯度钩子
class TestFullyShardPostAccGradHookMultiThread(FSDPTestMultiThread):

    # 返回当前测试的进程数量，这里设定为2
    @property
    def world_size(self) -> int:
        return 2

    # 使用unittest.skipIf装饰器，如果未启用CUDA，则跳过该测试
    @unittest.skipIf(not TEST_CUDA, "no cuda")
    # 定义测试方法，验证累积梯度后钩子的运行情况
    def test_post_acc_grad_hook_runs(self):
        # 创建一个字典，用于记录参数名到钩子运行次数的映射关系
        param_name_to_hook_count = collections.defaultdict(int)

        # 定义钩子函数，用于在梯度累积后更新钩子计数
        def hook(param_name: str, param: torch.Tensor) -> None:
            nonlocal param_name_to_hook_count
            param_name_to_hook_count[param_name] += 1

        # 创建一个MLP模型实例
        model = MLP(8)
        # 针对模型的每个子模块（in_proj, out_proj, model本身），应用完全分片
        for module in (model.in_proj, model.out_proj, model):
            fully_shard(module)
        
        # 遍历模型的所有参数，并为每个参数注册累积梯度后的钩子函数
        for param_name, param in model.named_parameters():
            param_hook = functools.partial(hook, param_name)
            param.register_post_accumulate_grad_hook(param_hook)

        # 创建一个CUDA上的随机输入
        inp = torch.randn((2, 8), device="cuda")
        # 执行模型前向传播、反向传播，并累积梯度
        model(inp).sum().backward()
        
        # 获取模型中所有参数的名称集合
        param_names = {param_name for param_name, _ in model.named_parameters()}
        # 断言：模型中的所有参数名称应该与钩子计数字典的键一致
        self.assertEqual(param_names, set(param_name_to_hook_count.keys()))
        
        # 遍历钩子计数字典，断言每个参数的钩子运行次数为1
        for param_name, count in param_name_to_hook_count.items():
            self.assertEqual(count, 1)


# 定义一个测试类，继承自FSDPTest类，用于测试多进程环境下的完全分片后累积梯度钩子
class TestFullyShardPostAccGradHookMultiProcess(FSDPTest):

    # 返回当前测试的进程数量，这里设定为不超过当前CUDA设备数量的最小值与2的较小值
    @property
    def world_size(self) -> int:
        return min(torch.cuda.device_count(), 2)

    # 使用skip_if_lt_x_gpu装饰器，如果CUDA设备少于2个，则跳过该测试
    @skip_if_lt_x_gpu(2)
    def test_post_acc_grad_hook_optim_parity(self):
        """
        Tests parity of running the optimizer via the post-accumulate-grad
        hook vs. normally.
        """
        # 设置随机种子，以便结果可重复
        torch.manual_seed(42)
        # 创建模型参数对象，设置 dropout 概率为 0.0
        model_args = ModelArgs(dropout_p=0.0)
        # 创建 Transformer 模型实例
        model = Transformer(model_args)

        # 深度复制模型用于参考，并将其移至 CUDA 设备
        ref_model = copy.deepcopy(model).cuda()
        # 对参考模型的所有层及模型本身进行全面分片处理
        for module in itertools.chain(ref_model.layers, [ref_model]):
            fully_shard(module)
        
        # 设置优化器的参数
        optim_kwargs = {"lr": 1e-2, "foreach": False}
        # 创建参考模型的优化器
        ref_optim = torch.optim.AdamW(ref_model.parameters(), **optim_kwargs)
        # 设置学习率调度器的参数
        lr_scheduler_kwargs = {"step_size": 5}
        # 创建参考模型的学习率调度器
        ref_lr_scheduler = torch.optim.lr_scheduler.StepLR(
            ref_optim, **lr_scheduler_kwargs
        )

        # 对当前模型的所有层及模型本身进行全面分片处理
        for module in itertools.chain(model.layers, [model]):
            fully_shard(module)
        
        # 初始化一个空字典，用于存储每个参数对应的优化器和学习率调度器
        param_to_optim = {}
        param_to_lr_scheduler = {}
        # 遍历当前模型的所有参数
        for param in model.parameters():
            # 创建当前参数对应的优化器，并将其存储在字典中
            param_to_optim[param] = torch.optim.AdamW([param], **optim_kwargs)
            # 创建当前参数对应的学习率调度器，并将其存储在字典中
            param_to_lr_scheduler[param] = torch.optim.lr_scheduler.StepLR(
                param_to_optim[param], **lr_scheduler_kwargs
            )

        # 定义优化钩子函数，用于在参数累积梯度后执行优化步骤和梯度清零操作，并更新学习率
        def optim_hook(param: nn.Parameter) -> None:
            param_to_optim[param].step()
            param_to_optim[param].zero_grad()
            param_to_lr_scheduler[param].step()

        # 为当前模型的每个参数注册优化钩子函数
        for param in model.parameters():
            param.register_post_accumulate_grad_hook(optim_hook)

        # 重新设置随机种子以确保结果可重复性
        torch.manual_seed(42 + self.rank)
        # 创建随机整数输入张量，形状为 (2, 16)，放置在 CUDA 设备上
        inp = torch.randint(0, model_args.vocab_size, (2, 16), device="cuda")
        # 迭代执行 10 次
        for _ in range(10):
            # 计算参考模型的损失和梯度
            ref_loss = ref_model(inp).sum()
            ref_loss.backward()
            ref_optim.step()
            ref_optim.zero_grad()
            ref_lr_scheduler.step()
            # 计算当前模型的损失和梯度
            loss = model(inp).sum()
            loss.backward()
            # 断言两个模型的损失值相等
            self.assertTrue(torch.equal(ref_loss, loss))
            # 断言两个模型的参数值相等
            for ref_param, param in zip(ref_model.parameters(), model.parameters()):
                self.assertTrue(torch.equal(ref_param, param))
# 如果当前脚本被直接执行（而不是被导入为模块），则执行以下代码块
if __name__ == "__main__":
    # 调用运行测试函数，此函数应该定义在当前脚本或导入的模块中
    run_tests()
```