# `.\pytorch\test\distributed\_composable\fsdp\test_fully_shard_mixed_precision.py`

```py
# Owner(s): ["oncall: distributed"]

# 导入必要的库和模块
import copy  # 导入copy模块，用于深拷贝对象
import functools  # 导入functools模块，用于函数式编程工具

from typing import Dict, List, Optional, Union  # 导入类型提示模块

import torch  # 导入PyTorch库
import torch.distributed as dist  # 导入PyTorch分布式模块
import torch.distributed._functional_collectives as funcol  # 导入PyTorch分布式函数集合模块
import torch.nn as nn  # 导入PyTorch神经网络模块
from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy  # 导入分布式训练相关模块
from torch.testing._internal.common_distributed import (
    requires_nccl_version,  # 导入版本需求检查函数
    SaveForwardInputsModel,  # 导入模型保存输入的辅助工具类
    skip_if_lt_x_gpu,  # 导入GPU数量检查的装饰器
)
from torch.testing._internal.common_fsdp import (
    check_sharded_parity,  # 导入检查分片一致性的辅助函数
    FSDPTest,  # 导入分布式训练单元测试基类
    FSDPTestMultiThread,  # 导入多线程分布式训练测试基类
    MLP,  # 导入多层感知机模型类
    patch_reduce_scatter,  # 导入改进reduce_scatter的辅助函数
    reduce_scatter_with_assert,  # 导入带有断言的reduce_scatter辅助函数
)
from torch.testing._internal.common_utils import run_tests  # 导入运行测试的工具函数


class TestFullyShardMixedPrecisionTraining(FSDPTest):
    @property
    def world_size(self) -> int:
        return min(4, torch.cuda.device_count())  # 返回GPU数量和4的最小值作为world_size

    def _init_models_and_optims(
        self,
        reshard_after_forward: Union[bool, int],  # 参数：前向后重分片标志
        param_dtype: Optional[torch.dtype],  # 参数：参数数据类型
        reduce_dtype: Optional[torch.dtype],  # 参数：减少数据类型
    ):
        torch.manual_seed(42)  # 设置随机种子为42
        model = nn.Sequential(*[MLP(16, torch.device("cpu")) for _ in range(3)])  # 创建包含3个MLP模型的序列模型
        ref_model = copy.deepcopy(model).cuda()  # 深拷贝模型并将其移到GPU上
        ref_optim = torch.optim.Adam(ref_model.parameters(), lr=1e-2)  # 创建Adam优化器用于参考模型
        mp_policy = MixedPrecisionPolicy(
            param_dtype=param_dtype, reduce_dtype=reduce_dtype
        )  # 创建混合精度策略对象
        fully_shard_fn = functools.partial(
            fully_shard,
            reshard_after_forward=reshard_after_forward,
            mp_policy=mp_policy,
        )  # 使用functools.partial创建部分应用的fully_shard函数
        for mlp in model:
            fully_shard_fn(mlp)  # 对每个MLP模型应用完全分片
        fully_shard_fn(model)  # 对整个模型应用完全分片
        optim = torch.optim.Adam(model.parameters(), lr=1e-2, foreach=True)  # 创建Adam优化器用于模型的每个参数
        return ref_model, ref_optim, model, optim  # 返回参考模型、参考优化器、模型和优化器

    @skip_if_lt_x_gpu(2)  # 如果GPU数量小于2，则跳过测试
    @requires_nccl_version((2, 10), "Need NCCL 2.10+ for bf16 collectives")
    def test_compute_dtype(self):
        self.run_subtests(
            {"reshard_after_forward": [False, True, 2]},  # 在不同条件下运行子测试
            self._test_compute_dtype,  # 使用_test_compute_dtype方法运行测试
        )
    # 定义一个测试方法来验证计算中的数据类型处理
    def _test_compute_dtype(self, reshard_after_forward: Union[bool, int]):
        # 设置参数的数据类型为 bfloat16
        param_dtype = torch.bfloat16
        # 初始化参考模型、优化器和当前模型、优化器
        ref_model, ref_optim, model, optim = self._init_models_and_optims(
            reshard_after_forward, param_dtype=param_dtype, reduce_dtype=None
        )
        # 深拷贝参考模型，并转换其参数为 bfloat16 数据类型
        ref_model_bf16 = copy.deepcopy(ref_model).to(param_dtype)
        # 缓存原始的 reduce_scatter_tensor 函数
        orig_reduce_scatter = dist.reduce_scatter_tensor

        # 定义一个断言函数，用于验证输出的数据类型是否为 param_dtype
        def assert_fn(output: torch.Tensor):
            self.assertEqual(output.dtype, param_dtype)

        # 使用 functools.partial 创建一个 reduce_scatter_with_assert 函数，
        # 其中包含了原始的 reduce_scatter_tensor 函数和断言函数 assert_fn
        reduce_scatter = functools.partial(
            reduce_scatter_with_assert, self, orig_reduce_scatter, assert_fn
        )
        # 设置随机种子
        torch.manual_seed(42 + self.rank + 1)
        # 创建一个在 CUDA 设备上的随机输入张量，数据类型为 param_dtype
        inp = torch.randn((4, 16), device="cuda", dtype=param_dtype)

        # 开始迭代测试
        for iter_idx in range(10):
            # 每次迭代前先将优化器的梯度清零，根据迭代次数决定是否保留梯度计算图
            optim.zero_grad(set_to_none=(iter_idx % 2 == 0))
            # 对模型进行前向计算并计算损失，将结果求和
            fsdp_loss = model(inp).sum()
            # 使用 patch_reduce_scatter 函数修改 reduce_scatter 的行为
            with patch_reduce_scatter(reduce_scatter):
                # 反向传播计算梯度
                fsdp_loss.backward()
            # 根据梯度更新优化器状态
            optim.step()

            # 同时更新参考模型的优化器状态，以进行比较
            ref_optim.zero_grad(set_to_none=(iter_idx % 2 == 0))
            # 使用 bfloat16 数据类型对输入进行前向计算，并计算损失，将结果求和
            ref_loss = ref_model_bf16(inp.to(param_dtype)).sum()
            # 对损失进行反向传播计算梯度
            ref_loss.backward()
            # 对每个参数的梯度进行 reduce-scatter 和 all-gather 操作，实现类似 all-reduce 的效果
            for param in ref_model_bf16.parameters():
                output = torch.zeros_like(torch.chunk(param.grad, self.world_size)[0])
                dist.reduce_scatter_tensor(output, param.grad)
                dist.all_gather_into_tensor(param.grad, output)
                param.grad.div_(self.world_size)
            # 将参考模型的 bfloat16 参数梯度转换为 fp32 数据类型
            for param_fp32, param_bf16 in zip(
                ref_model.parameters(), ref_model_bf16.parameters()
            ):
                param_fp32.grad = param_bf16.grad.to(param_fp32.dtype)
                param_bf16.grad = None
            # 使用 fp32 优化器进行参数更新
            ref_optim.step()
            # 将 fp32 模型的参数值复制到 bfloat16 模型，保持同步
            for param_fp32, param_bf16 in zip(
                ref_model.parameters(), ref_model_bf16.parameters()
            ):
                param_bf16.detach().copy_(param_fp32)

            # 断言当前模型和参考模型的损失是否相等
            self.assertEqual(fsdp_loss, ref_loss)
            # 检查分片的一致性，确保模型同步
            check_sharded_parity(self, ref_model, model)

    # 跳过 GPU 数量小于 2 的测试
    @skip_if_lt_x_gpu(2)
    # 要求 NCCL 版本为 2.10+，用于 bfloat16 通信
    @requires_nccl_version((2, 10), "Need NCCL 2.10+ for bf16 collectives")
    # 测试 reduce_dtype 的功能
    def test_reduce_dtype(self):
        # 对 _test_reduce_dtype_fp32_reduce 方法进行子测试
        self.run_subtests(
            {"reshard_after_forward": [False, True, 2]},
            self._test_reduce_dtype_fp32_reduce,
        )
        # 对 _test_reduce_dtype_bf16_reduce 方法进行子测试
        self.run_subtests(
            {"reshard_after_forward": [False, True, 2]},
            self._test_reduce_dtype_bf16_reduce,
        )
    # 定义一个测试方法，用于验证减少数据类型为 fp32 的 reduce 操作
    def _test_reduce_dtype_fp32_reduce(self, reshard_after_forward: Union[bool, int]):
        # 设置参数数据类型为 bfloat16，减少数据类型为 float32
        param_dtype, reduce_dtype = torch.bfloat16, torch.float32
        # 初始化参考模型、优化器、当前模型、当前优化器
        ref_model, ref_optim, model, optim = self._init_models_and_optims(
            reshard_after_forward, param_dtype=param_dtype, reduce_dtype=reduce_dtype
        )
        # 深拷贝参考模型并转换为 param_dtype 类型
        ref_model_bf16 = copy.deepcopy(ref_model).to(param_dtype)
        # 保存原始的 reduce_scatter_tensor 函数
        orig_reduce_scatter = dist.reduce_scatter_tensor

        # 定义断言函数，验证输出的数据类型是否为 reduce_dtype
        def assert_fn(output: torch.Tensor):
            self.assertEqual(output.dtype, reduce_dtype)

        # 部分应用 reduce_scatter_with_assert 函数，传入 orig_reduce_scatter 和 assert_fn 函数
        reduce_scatter = functools.partial(
            reduce_scatter_with_assert, self, orig_reduce_scatter, assert_fn
        )
        # 设置随机种子
        torch.manual_seed(42 + self.rank + 1)
        # 生成随机输入数据，使用 param_dtype 类型，在 GPU 上进行计算
        inp = torch.randn((4, 16), device="cuda", dtype=param_dtype)
        # 迭代 10 次
        for iter_idx in range(10):
            # 梯度归零，根据迭代次数设置是否将梯度置为 None
            optim.zero_grad(set_to_none=(iter_idx % 2 == 0))
            # 对模型进行前向传播，计算损失
            fsdp_loss = model(inp).sum()
            # 使用 patch_reduce_scatter 修饰 reduce_scatter 函数的上下文
            with patch_reduce_scatter(reduce_scatter):
                # 反向传播计算梯度
                fsdp_loss.backward()
            # 更新优化器参数
            optim.step()

            # 参考模型的优化过程
            ref_optim.zero_grad(set_to_none=(iter_idx % 2 == 0))
            # 使用 param_dtype 类型进行模型前向传播，计算损失
            ref_loss = ref_model_bf16(inp.to(param_dtype)).sum()
            # 反向传播计算梯度
            ref_loss.backward()
            # 将参考模型的梯度转换为 float32 类型，并进行全局 reduce 操作
            for param in ref_model_bf16.parameters():
                param.grad.data = param.grad.to(torch.float32)
                dist.all_reduce(param.grad)  # fp32 reduction
                param.grad.div_(self.world_size)
            # 将 float32 类型的梯度更新到参考模型的参数上
            for param_fp32, param_bf16 in zip(
                ref_model.parameters(), ref_model_bf16.parameters()
            ):
                param_fp32.grad = param_bf16.grad
                param_bf16.grad = None
            # 参考模型优化器执行一步 fp32 优化
            ref_optim.step()
            # 将参考模型的参数复制回当前模型的参数中
            for param_fp32, param_bf16 in zip(
                ref_model.parameters(), ref_model_bf16.parameters()
            ):
                param_bf16.detach().copy_(param_fp32)

            # 验证当前模型的损失与参考模型的损失是否相等
            self.assertEqual(fsdp_loss, ref_loss)
            # 检查分片的一致性
            check_sharded_parity(self, ref_model, model)
    # 定义一个方法 _test_reduce_dtype_bf16_reduce，接受一个名为 reshard_after_forward 的参数，
    # 可以是布尔型或整型。这个方法用于测试在使用 bf16 计算和 fp32 减少时的梯度累积行为。
    def _test_reduce_dtype_bf16_reduce(self, reshard_after_forward: Union[bool, int]):
        # 设置两种数据类型：param_dtype 为 torch.float32，reduce_dtype 为 torch.bfloat16
        param_dtype, reduce_dtype = torch.float32, torch.bfloat16
        # 初始化参考模型、优化器、当前模型、当前优化器，并返回这些对象
        ref_model, ref_optim, model, optim = self._init_models_and_optims(
            reshard_after_forward, param_dtype=param_dtype, reduce_dtype=reduce_dtype
        )
        # 获取默认的分布式组
        group = dist.distributed_c10d._get_default_group()
        # 保存原始的 reduce_scatter_tensor 方法
        orig_reduce_scatter = dist.reduce_scatter_tensor

        # 定义一个断言函数 assert_fn，用于检查输出的张量的数据类型是否为 reduce_dtype
        def assert_fn(output: torch.Tensor):
            self.assertEqual(output.dtype, reduce_dtype)

        # 使用 functools.partial 创建一个新的 reduce_scatter 函数，将 assert_fn 作为参数传入
        reduce_scatter = functools.partial(
            reduce_scatter_with_assert, self, orig_reduce_scatter, assert_fn
        )
        # 设置随机数种子
        torch.manual_seed(42 + self.rank + 1)
        # 在 CUDA 设备上生成一个随机张量 inp，数据类型为 param_dtype
        inp = torch.randn((4, 16), device="cuda", dtype=param_dtype)
        
        # 执行 10 次迭代
        for iter_idx in range(10):
            # 每次迭代前将优化器的梯度归零，根据条件选择是否置为 None
            optim.zero_grad(set_to_none=(iter_idx % 2 == 0))
            # 对模型输入 inp 进行前向传播并计算损失，求和作为 fsdp_loss
            fsdp_loss = model(inp).sum()
            # 使用 patch_reduce_scatter 上下文管理器替换 reduce_scatter 方法
            with patch_reduce_scatter(reduce_scatter):
                # 反向传播计算梯度
                fsdp_loss.backward()
            # 优化器执行一步更新参数
            optim.step()

            # 参考模型的优化器也进行相同的步骤
            ref_optim.zero_grad(set_to_none=(iter_idx % 2 == 0))
            ref_loss = ref_model(inp).sum()
            ref_loss.backward()
            # 对参考模型的每个参数执行梯度更新
            for param in ref_model.parameters():
                # 将参数的梯度转换为 reduce_dtype
                param_grad = param.grad.to(reduce_dtype)
                # 使用 reduce_scatter_tensor 函数对梯度进行 reduce-scatter 操作，scatter_dim=0，
                # reduceOp="avg" 表示平均值操作，group 表示通信组
                sharded_grad = funcol.reduce_scatter_tensor(
                    param_grad, scatter_dim=0, reduceOp="avg", group=group
                )  # bf16 reduction
                # 使用 all_gather_tensor 函数对 sharded_grad 进行 all-gather 操作，
                # gather_dim=0 表示沿着第一个维度进行收集，group 表示通信组
                param.grad = funcol.all_gather_tensor(
                    sharded_grad, gather_dim=0, group=group
                ).to(
                    param.dtype
                )  # upcast to fp32
            # 参考模型的优化器执行一步更新参数，使用 fp32 类型进行优化
            ref_optim.step()  # fp32 optimizer step

            # 断言当前模型计算得到的损失与参考模型的损失相等
            self.assertEqual(fsdp_loss, ref_loss)
            # 检查当前模型和参考模型的参数是否一致
            check_sharded_parity(self, ref_model, model)

    # 装饰器函数，用于检查当前 GPU 数量是否大于等于 2，若不满足条件则跳过测试
    @skip_if_lt_x_gpu(2)
    # 测试方法 test_grad_acc_with_reduce_dtype，验证在使用 bf16 计算和 fp32 减少时梯度累积行为的情况
    def test_grad_acc_with_reduce_dtype(self):
        """
        Tests that gradient accumulation without reduce-scatter when using
        bf16 compute and fp32 reduction accumulates the unsharded gradients in
        fp32.
        """
        # 运行子测试，测试参数 reshard_after_forward 取值为 True 和 False 时的情况，
        # 使用 _test_grad_acc_with_reduce_dtype 方法作为测试函数
        self.run_subtests(
            {"reshard_after_forward": [True, False]},
            self._test_grad_acc_with_reduce_dtype,
        )
class TestFullyShardMixedPrecisionCasts(FSDPTestMultiThread):
    @property
    def world_size(self) -> int:
        return 2

    @skip_if_lt_x_gpu(1)
    def test_float16_on_one_submodule(self):
        x = torch.zeros(2, 100, device="cuda")

        # Subtest 1: use fp16 on the second child submodule -- does not require
        # any additional casting logic
        forward_inputs: Dict[str, nn.Module] = {}
        # 创建 SaveForwardInputsModel 实例，保存前向输入，无需额外的类型转换逻辑
        model = SaveForwardInputsModel(
            forward_inputs,
            cast_forward_inputs=False,
        ).cuda()
        # 在 model.c2 上使用 MixedPrecisionPolicy 将参数类型设置为 float16
        fully_shard(model.c2, mp_policy=MixedPrecisionPolicy(param_dtype=torch.float16))
        # 在整个模型上应用混合精度策略
        fully_shard(model)
        # 对模型进行前向传播、求和及反向传播
        model(x).sum().backward()
        # 断言前向输入的数据类型
        self.assertEqual(forward_inputs[model].dtype, torch.float32)
        self.assertEqual(forward_inputs[model.c1].dtype, torch.float32)
        self.assertEqual(forward_inputs[model.c2].dtype, torch.float16)

        # Subtest 2: use fp16 on the second child module, where the user module
        # owns the cast
        forward_inputs: Dict[nn.Module, torch.Tensor] = {}
        # 创建 SaveForwardInputsModel 实例，保存前向输入，并由用户模块进行类型转换
        model = SaveForwardInputsModel(
            forward_inputs=forward_inputs, cast_forward_inputs=True
        ).cuda()
        # 在 model.c2 上使用 MixedPrecisionPolicy 将参数类型设置为 float16，但不进行前向输入的转换
        fully_shard(
            model.c2,
            mp_policy=MixedPrecisionPolicy(
                param_dtype=torch.float16, cast_forward_inputs=False
            ),
        )
        # 在整个模型上应用混合精度策略
        fully_shard(model)
        # 对模型进行前向传播、求和及反向传播
        model(x).sum().backward()
        # 断言前向输入的数据类型
        self.assertEqual(forward_inputs[model].dtype, torch.float32)
        self.assertEqual(forward_inputs[model.c1].dtype, torch.float32)
        self.assertEqual(forward_inputs[model.c2].dtype, torch.float32)

        # Subtest 3: use fp16 on the first child module and specify its output
        # dtype so that the second child module does not need to cast
        forward_inputs: Dict[nn.Module, torch.Tensor] = {}
        # 创建 SaveForwardInputsModel 实例，保存前向输入，且不进行前向输入的类型转换
        model = SaveForwardInputsModel(
            forward_inputs=forward_inputs, cast_forward_inputs=False
        ).cuda()
        # 在 model.c1 上使用 MixedPrecisionPolicy 将参数类型设置为 float16，并指定输出类型为 float32
        fully_shard(
            model.c1,
            mp_policy=MixedPrecisionPolicy(
                param_dtype=torch.float16, output_dtype=torch.float32
            ),
        )
        # 在整个模型上应用混合精度策略
        fully_shard(model)
        # 对模型进行前向传播、求和及反向传播
        model(x).sum().backward()
        # 断言前向输入的数据类型
        self.assertEqual(forward_inputs[model].dtype, torch.float32)
        self.assertEqual(forward_inputs[model.c1].dtype, torch.float16)
        self.assertEqual(forward_inputs[model.c2].dtype, torch.float32)

    @skip_if_lt_x_gpu(1)
    def test_submodules_with_external_inputs(self):
        # 运行带有外部输入的子模块的测试用例
        self.run_subtests(
            {"enable_submodule_cast": [False, True]},
            self._test_submodules_with_external_inputs,
        )
    # 定义测试函数 _test_submodules_with_external_inputs，接受一个布尔型参数 enable_submodule_cast
    def _test_submodules_with_external_inputs(self, enable_submodule_cast: bool):
        # 定义一个名为 ToyModule 的内部类，继承自 nn.Module
        class ToyModule(nn.Module):
            # 初始化函数，接受一个字典类型的参数 forward_inputs，返回空值
            def __init__(self, forward_inputs: Dict[str, torch.Tensor]) -> None:
                super().__init__()
                # 创建一个线性层，输入维度为 100，输出维度为 100
                self.l = nn.Linear(100, 100)
                # 将输入参数 forward_inputs 存储在实例属性中
                self.forward_inputs = forward_inputs

            # 前向传播函数，接受两个张量 x 和 y，返回一个张量
            def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                # 将张量 x 存储在 forward_inputs 字典中的键 "l2_input_x"
                self.forward_inputs["l2_input_x"] = x
                # 将张量 y 存储在 forward_inputs 字典中的键 "l2_input_y"
                self.forward_inputs["l2_input_y"] = y
                # 返回线性层 self.l 对输入 x 进行的计算结果
                return self.l(x)

        # 定义一个名为 ToyModel 的内部类，继承自 nn.Module
        class ToyModel(nn.Module):
            # 初始化函数，接受一个字典类型的参数 forward_inputs，返回空值
            def __init__(self, forward_inputs: Dict[str, torch.Tensor]) -> None:
                super().__init__()
                # 创建一个线性层，输入维度为 100，输出维度为 100
                self.l1 = nn.Linear(100, 100)
                # 创建一个 ToyModule 实例，传入参数 forward_inputs
                self.l2 = ToyModule(forward_inputs)
                # 将输入参数 forward_inputs 存储在实例属性中
                self.forward_inputs = forward_inputs

            # 前向传播函数，接受一个张量 x，返回一个张量
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # 将张量 x 存储在 forward_inputs 字典中的键 "model_input_x"
                self.forward_inputs["model_input_x"] = x
                # 创建一个与指定大小的张量 y，元素均为 1，位于 GPU 上，数据类型为 float32
                y = torch.ones(
                    2, 100, device="cuda", dtype=torch.float32
                )  # external input
                # 调用 ToyModule 实例 self.l2 的前向传播函数，传入参数 self.l1(x) 和 y，并返回结果
                return self.l2(self.l1(x), y)

        # 创建一个空字典 forward_inputs
        forward_inputs: Dict[str, torch.Tensor] = {}
        # 创建一个 ToyModel 实例 model，传入参数 forward_inputs，并将其移到 GPU 上
        model = ToyModel(forward_inputs).cuda()
        # 创建一个与指定大小的张量 x，元素均为 0，位于 GPU 上，数据类型为 float32
        x = torch.zeros(2, 100, device="cuda", dtype=torch.float32)
        # 调用 fully_shard 函数，对 model.l2 进行特定策略的分片操作
        fully_shard(
            model.l2,
            mp_policy=MixedPrecisionPolicy(
                param_dtype=torch.float16, cast_forward_inputs=enable_submodule_cast
            ),
        )
        # 调用 fully_shard 函数，对 model 进行特定策略的分片操作
        fully_shard(model, mp_policy=MixedPrecisionPolicy(param_dtype=torch.float16))
        # 对 model(x) 进行前向传播计算，然后对结果进行求和，并进行反向传播
        model(x).sum().backward()

        # 如果 enable_submodule_cast 为 True，则 model.l2 被设为转换为 fp16，否则保持为 fp32
        self.assertEqual(forward_inputs["model_input_x"].dtype, torch.float16)
        # 断言 forward_inputs 字典中 "l2_input_x" 键对应的张量数据类型为 torch.float16
        self.assertEqual(forward_inputs["l2_input_x"].dtype, torch.float16)
        # 断言 forward_inputs 字典中 "l2_input_y" 键对应的张量数据类型根据 enable_submodule_cast 的值而定，为 torch.float16 或 torch.float32
        self.assertEqual(
            forward_inputs["l2_input_y"].dtype,
            torch.float16 if enable_submodule_cast else torch.float32,
        )

    # 标记测试函数 test_norm_modules_bf16，在 GPU 数量不小于 1 时运行
    @skip_if_lt_x_gpu(1)
    @requires_nccl_version((2, 10), "Need NCCL 2.10+ for bf16 collectives")
    def test_norm_modules_bf16(self):
        # 创建一个 MixedPrecisionPolicy 实例 mp_policy，参数数据类型为 torch.bfloat16
        mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16)
        # 调用 _test_norm_modules 函数，传入 mp_policy 参数进行测试
        self._test_norm_modules(mp_policy)

    # 标记测试函数 test_norm_modules_fp16，在 GPU 数量不小于 1 时运行
    @skip_if_lt_x_gpu(1)
    def test_norm_modules_fp16(self):
        # 创建一个 MixedPrecisionPolicy 实例 mp_policy，参数数据类型为 torch.float16
        mp_policy = MixedPrecisionPolicy(param_dtype=torch.float16)
        # 调用 _test_norm_modules 函数，传入 mp_policy 参数进行测试
        self._test_norm_modules(mp_policy)
    # 定义测试函数 _test_norm_modules，接受一个混合精度策略参数 mp_policy
    def _test_norm_modules(self, mp_policy: MixedPrecisionPolicy):
        
        # 定义内部函数 inner，接受模型 model 和输入张量 x，运行前向和反向传播检查类型匹配性
        def inner(model: nn.Module, x: torch.Tensor):
            # 运行模型前向传播得到输出 z
            z = model(x)
            # 断言输出 z 的数据类型与 mp_policy.param_dtype 一致
            self.assertEqual(z.dtype, mp_policy.param_dtype)
            # 对输出 z 求和并进行反向传播
            z.sum().backward()

        # Layer norm
        # 创建包含线性层、LayerNorm 层和线性层的序列模型
        model = nn.Sequential(nn.Linear(32, 32), nn.LayerNorm(32), nn.Linear(32, 32))
        # 对序列模型的每个模块（包括整个模型本身）应用 fully_shard 函数，使用给定的混合精度策略 mp_policy
        for module in (model[0], model[1], model[2], model):
            fully_shard(module, mp_policy=mp_policy)
        # 调用 inner 函数，传入模型和形状为 (4, 32) 的随机张量作为输入
        inner(model, torch.randn((4, 32)))

        # Batch norm 1D
        # 创建包含线性层、BatchNorm1d 层和线性层的序列模型
        model = nn.Sequential(nn.Linear(32, 32), nn.BatchNorm1d(32), nn.Linear(32, 32))
        # 对序列模型的每个模块应用 fully_shard 函数，使用给定的混合精度策略 mp_policy
        for module in (model[0], model[1], model[2], model):
            fully_shard(module, mp_policy=mp_policy)
        # 调用 inner 函数，传入模型和形状为 (4, 32) 的随机张量作为输入
        inner(model, torch.randn((4, 32)))

        # Batch norm 2D: error in backward from buffer dtype mismatch
        # 创建包含卷积层、BatchNorm2d 层和卷积层的序列模型
        model = nn.Sequential(nn.Conv2d(1, 5, 3), nn.BatchNorm2d(5), nn.Conv2d(5, 4, 3))
        # 对序列模型的每个模块应用 fully_shard 函数，使用给定的混合精度策略 mp_policy
        for module in (model[0], model[1], model[2], model):
            fully_shard(module, mp_policy=mp_policy)
        # 使用断言检查在 BatchNorm2d 层反向传播时出现的错误
        with self.assertRaisesRegex(RuntimeError, "Expected running_mean to have type"):
            # 调用 inner 函数，传入模型和形状为 (3, 1, 9, 9) 的随机张量作为输入，预期触发错误
            inner(model, torch.randn((3, 1, 9, 9)))

        # Batch norm 2D: cast buffers down to lower precision
        # 创建包含卷积层、BatchNorm2d 层和卷积层的序列模型
        model = nn.Sequential(nn.Conv2d(1, 5, 3), nn.BatchNorm2d(5), nn.Conv2d(5, 4, 3))
        # 对序列模型的每个模块应用 fully_shard 函数，使用给定的混合精度策略 mp_policy
        for module in (model[0], model[1], model[2], model):
            fully_shard(module, mp_policy=mp_policy)
        # 将 BatchNorm2d 层的 running_mean 和 running_var 张量转换为更低精度以支持反向传播
        model[1].running_mean = model[1].running_mean.to(mp_policy.param_dtype)
        model[1].running_var = model[1].running_var.to(mp_policy.param_dtype)
        # 调用 inner 函数，传入模型和形状为 (3, 1, 9, 9) 的随机张量作为输入
        inner(model, torch.randn((3, 1, 9, 9)))

        # Batch norm 2D: use special mixed precision policy
        # 创建包含卷积层、BatchNorm2d 层和卷积层的序列模型
        model = nn.Sequential(nn.Conv2d(1, 5, 3), nn.BatchNorm2d(5), nn.Conv2d(5, 4, 3))
        # 创建特定的 BatchNorm2d 层混合精度策略 bn_mp_policy
        bn_mp_policy = MixedPrecisionPolicy(output_dtype=mp_policy.param_dtype)
        # 对 BatchNorm2d 层应用 fully_shard 函数，使用 bn_mp_policy 混合精度策略
        fully_shard(model[1], mp_policy=bn_mp_policy)
        # 对序列模型的每个模块（不包括 BatchNorm2d 层）应用给定的混合精度策略 mp_policy
        for module in (model[0], model[2], model):
            fully_shard(module, mp_policy=mp_policy)
        # 调用 inner 函数，传入模型和形状为 (3, 1, 9, 9) 的随机张量作为输入
        inner(model, torch.randn((3, 1, 9, 9)))
# 如果当前模块被直接执行（而不是被导入到其他模块中执行），则执行以下代码
if __name__ == "__main__":
    # 调用名为 run_tests 的函数，用于执行测试或其他主要操作
    run_tests()
```