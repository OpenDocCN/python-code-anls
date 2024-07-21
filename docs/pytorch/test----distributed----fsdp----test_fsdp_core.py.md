# `.\pytorch\test\distributed\fsdp\test_fsdp_core.py`

```py
# Owner(s): ["oncall: distributed"]

# 导入必要的模块和库
import contextlib  # 上下文管理工具
import functools  # 函数工具
import itertools  # 迭代工具
import sys  # 系统相关功能
from typing import Any, Callable, Dict, List, Optional  # 类型提示
from unittest import mock  # 单元测试的模拟对象

import torch  # PyTorch 主库
import torch.distributed as dist  # 分布式训练模块
import torch.nn as nn  # PyTorch 神经网络模块
from torch.distributed.fsdp import CPUOffload, MixedPrecision  # FSDP CPU 离载和混合精度
from torch.distributed.fsdp._flat_param import FlatParamHandle  # 扁平参数处理
from torch.distributed.fsdp.fully_sharded_data_parallel import (  # 完全分片数据并行
    BackwardPrefetch,  # 后向预取
    FullyShardedDataParallel as FSDP,  # 完全分片数据并行别名
    ShardingStrategy,  # 分片策略
)
from torch.distributed.fsdp.wrap import ModuleWrapPolicy  # 模块封装策略
from torch.distributed.utils import _p_assert  # 分布式实用工具中的断言函数
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu  # 分布式测试中的 GPU 数量检查
from torch.testing._internal.common_fsdp import (  # FSDP 测试相关模块
    AlwaysWrapNestedWrappedModule,  # 总是封装嵌套模块
    CUDAInitMode,  # CUDA 初始化模式
    DummyDDP,  # 虚拟的 DDP
    FSDPInitMode,  # FSDP 初始化模式
    FSDPTest,  # FSDP 测试基类
    MixtureOfExperts,  # 专家混合
    NestedWrappedModule,  # 嵌套封装模块
    NestedWrappedModuleWithDelay,  # 带延迟的嵌套封装模块
    subtest_name,  # 子测试名称生成器
    TransformerWithSharedParams,  # 具有共享参数的 Transformer
)
from torch.testing._internal.common_utils import (  # 通用测试工具
    instantiate_parametrized_tests,  # 实例化参数化测试
    parametrize,  # 参数化装饰器
    run_tests,  # 运行测试
    TEST_WITH_DEV_DBG_ASAN,  # 是否在 dev-asan 模式下运行测试
)

# 如果分布式训练不可用，输出错误信息并退出
if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)

# 如果处于 dev-asan 模式，输出相关信息并退出
if TEST_WITH_DEV_DBG_ASAN:
    print(
        "Skip dev-asan as torch + multiprocessing spawn have known issues",
        file=sys.stderr,
    )
    sys.exit(0)

# 定义测试参数和配置
params = "cpu_offload,sharding_strategy"
cpu_offload_config = [CPUOffload(offload_params=True), CPUOffload(offload_params=False)]
sharding_strategy_config = [
    None,
    ShardingStrategy.SHARD_GRAD_OP,
    ShardingStrategy.NO_SHARD,
]
configs = list(itertools.product(cpu_offload_config, sharding_strategy_config))

# 定义测试名称映射
test_name_mapping = {
    str(CPUOffload(offload_params=True)): "offload_true",
    str(CPUOffload(offload_params=False)): "offload_false",
    str(ShardingStrategy.SHARD_GRAD_OP): "shard_grad_op",
    str(ShardingStrategy.NO_SHARD): "no_shard",
}

# 定义子测试名称生成函数
subtest_name = functools.partial(subtest_name, test_name_mapping)


class TestParityWithDDP(FSDPTest):
    """
    Compare losses and parameter values after several updates when using
    PyTorch DDP vs. FullyShardedDataParallel.
    """

    def _get_cuda_init_modes(self, cpu_offload: CPUOffload) -> List[CUDAInitMode]:
        # 返回 CUDA 初始化模式列表，根据 CPU 离载参数确定支持的模式
        modes = [
            CUDAInitMode.CUDA_AFTER,
            CUDAInitMode.CUDA_BEFORE,
        ]
        # 注意，CUDAInitMode.CUDA_NEVER 目前仅在启用 CPU 离载时工作，
        # 因为我们明确将参数带回 CUDA 设备。一般情况下，它不起作用，
        # 因为我们试图 all_gather 在 CPU 上的 p.data，但 NCCL 仅支持 GPU。
        if cpu_offload.offload_params:
            modes.append(CUDAInitMode.CUDA_NEVER)

        return modes
    def _get_subtest_config(self, cpu_offload: CPUOffload) -> Dict[str, List[Any]]:
        """Returns a subtest configuration that subtests CUDA initialization
        modes and prefetching settings together."""
        # 构建并返回一个子测试配置字典，用于测试 CUDA 初始化模式和预取设置
        return {
            "cuda_init_mode": self._get_cuda_init_modes(cpu_offload),
            # 获取 CUDA 初始化模式的列表
            "backward_prefetch": [
                None,
                BackwardPrefetch.BACKWARD_PRE,
                BackwardPrefetch.BACKWARD_POST,
            ],
            # 定义后向预取选项列表
            "forward_prefetch": [False, True],
            # 定义前向预取选项列表
            "use_orig_params": [False, True],
            # 定义使用原始参数选项列表
        }

    @skip_if_lt_x_gpu(2)
    @parametrize(params, configs, subtest_name)
    def test_nested_wrapped_model(
        self,
        cpu_offload: CPUOffload,
        sharding_strategy: Optional[ShardingStrategy],
    ):
        # 运行子测试，用于测试嵌套包装模型的功能
        self.run_subtests(
            self._get_subtest_config(cpu_offload),
            self._test_fsdp_parity,
            NestedWrappedModule,
            FSDPInitMode.RECURSIVE,
            cpu_offload=cpu_offload,
            sharding_strategy=sharding_strategy,
        )

    @skip_if_lt_x_gpu(2)
    @parametrize(params, configs, subtest_name)
    def test_nested_wrapped_model_single_iteration_mixed_precision(
        self,
        cpu_offload: CPUOffload,
        sharding_strategy: Optional[ShardingStrategy],
    ):
        mixed_precision = MixedPrecision(
            param_dtype=torch.float16,
            buffer_dtype=torch.float16,
            reduce_dtype=torch.float16,
        )
        # 运行子测试，测试混合精度下的单次迭代嵌套包装模型
        self.run_subtests(
            self._get_subtest_config(cpu_offload),
            self._test_fsdp_parity,
            NestedWrappedModule,
            FSDPInitMode.RECURSIVE,
            cpu_offload=cpu_offload,
            sharding_strategy=sharding_strategy,
            num_iters=1,
            mixed_precision=mixed_precision,
        )

    @skip_if_lt_x_gpu(2)
    @parametrize(params, configs, subtest_name)
    def test_nested_always_wrap_model(
        self,
        cpu_offload: CPUOffload,
        sharding_strategy: Optional[ShardingStrategy],
    ):
        # 运行子测试，测试始终包装的嵌套包装模型
        self.run_subtests(
            self._get_subtest_config(cpu_offload),
            self._test_fsdp_parity,
            AlwaysWrapNestedWrappedModule,
            FSDPInitMode.RECURSIVE,
            cpu_offload=cpu_offload,
            sharding_strategy=sharding_strategy,
        )

    @skip_if_lt_x_gpu(2)
    @parametrize(params, configs, subtest_name)
    def test_transformer(
        self,
        cpu_offload: CPUOffload,
        sharding_strategy: Optional[ShardingStrategy],
    ):
        # 运行子测试，测试具有共享参数的 Transformer 模型
        self.run_subtests(
            self._get_subtest_config(cpu_offload),
            self._test_fsdp_parity,
            TransformerWithSharedParams,
            FSDPInitMode.RECURSIVE,
            cpu_offload=cpu_offload,
            sharding_strategy=sharding_strategy,
        )
    def test_delayed_optim_step(
        self,
        cpu_offload: CPUOffload,
        sharding_strategy: Optional[ShardingStrategy],
    ):
        """Tests the FSDP forward, backward, and optimizer step runtime by
        using a model with a long CUDA delay after the loss computation/before
        the optimizer step to exercise the internal CUDA stream usage in that
        the forward pass all-gathers do not start until after the optimizer
        step completes."""
        # 运行子测试，测试延迟优化步骤的性能
        self.run_subtests(
            # 获取子测试配置
            self._get_subtest_config(cpu_offload),
            # 测试 FSDP 的一致性
            self._test_fsdp_parity,
            # 使用带延迟的嵌套包装模块
            NestedWrappedModuleWithDelay,
            # 递归初始化 FSDP 模式
            FSDPInitMode.RECURSIVE,
            cpu_offload=cpu_offload,
            sharding_strategy=sharding_strategy,
            # 初始化参数，延迟 250 毫秒
            init_kwargs={"delay_after_loss_ms": 250},
        )

    @skip_if_lt_x_gpu(2)
    @parametrize(params, configs, subtest_name)
    def test_delayed_reduce_scatter(
        self,
        cpu_offload: CPUOffload,
        sharding_strategy: Optional[ShardingStrategy],
    ):
        """Tests the FSDP forward, backward, and optimizer step runtime by
        using a model with a long CUDA delay before the gradient reduce-scatter
        to exercise the internal CUDA stream usage in that the backward pass
        waits for those reductions to finish."""
        # 运行子测试，测试延迟的梯度 reduce-scatter
        self.run_subtests(
            # 获取子测试配置
            self._get_subtest_config(cpu_offload),
            # 测试 FSDP 的一致性
            self._test_fsdp_parity,
            # 使用带延迟的嵌套包装模块
            NestedWrappedModuleWithDelay,
            # 递归初始化 FSDP 模式
            FSDPInitMode.RECURSIVE,
            cpu_offload=cpu_offload,
            sharding_strategy=sharding_strategy,
            # 初始化参数，延迟 250 毫秒
            init_kwargs={"delay_before_reduction_ms": 250},
        )

    def _dummy_ddp_fn(self, model):
        # `MixtureOfExperts`` implements custom gradient reduction logic, so
        # the reference behavior should follow that logic instead of DDP
        # 返回一个 DummyDDP 对象，用于模拟 DDP 的行为
        return DummyDDP(model)

    @skip_if_lt_x_gpu(2)
    @parametrize(params, configs, subtest_name)
    def test_mixture_of_experts(
        self,
        cpu_offload: CPUOffload,
        sharding_strategy: Optional[ShardingStrategy],
    ):
        # 运行子测试，测试专家混合模型的性能
        self.run_subtests(
            # 获取子测试配置
            self._get_subtest_config(cpu_offload),
            # 测试 FSDP 的一致性
            self._test_fsdp_parity,
            # 使用专家混合模型
            MixtureOfExperts,
            # 递归初始化 FSDP 模式
            FSDPInitMode.RECURSIVE,
            # 设置参考初始化函数为 _dummy_ddp_fn
            ref_init_fn=self._dummy_ddp_fn,
            cpu_offload=cpu_offload,
            sharding_strategy=sharding_strategy,
        )

    @skip_if_lt_x_gpu(2)
    @parametrize(params, configs, subtest_name)
    def test_mixture_of_experts_with_delay_before_free(
        self,
        cpu_offload: CPUOffload,
        sharding_strategy: Optional[ShardingStrategy],
    ):
        # 运行子测试，测试在释放之前延迟的专家混合模型的性能
        self.run_subtests(
            # 获取子测试配置
            self._get_subtest_config(cpu_offload),
            # 测试 FSDP 的一致性
            self._test_fsdp_parity,
            # 使用专家混合模型
            MixtureOfExperts,
            # 递归初始化 FSDP 模式
            FSDPInitMode.RECURSIVE,
            cpu_offload=cpu_offload,
            sharding_strategy=sharding_strategy,
        )
        ):
        # 调用当前对象的 run_subtests 方法，执行一组子测试，传入以下参数：
        # - self._get_subtest_config(cpu_offload): 获取用于子测试的配置信息，根据是否使用 CPU offload 决定
        # - self._test_fsdp_parity: 执行 FSDP（Fully Sharded Data Parallel） 的 parity 测试方法
        # - MixtureOfExperts: 专家混合模型（Mixture of Experts）
        # - FSDPInitMode.RECURSIVE: 使用递归方式进行 FSDP 初始化模式
        # - ref_init_fn=self._dummy_ddp_fn: 参考初始化函数使用 self._dummy_ddp_fn
        # - cpu_offload=cpu_offload: 是否进行 CPU offload 的参数
        # - sharding_strategy=sharding_strategy: 分片策略参数
        # - init_kwargs={"delay_before_free_ms": 250}: 初始化参数，设置延迟释放时间为 250 毫秒
# 定义一个测试类 TestParamInit，继承自 FSDPTest，用于测试参数初始化
class TestParamInit(FSDPTest):
    # 装饰器：如果 GPU 小于 2 个则跳过该测试
    @skip_if_lt_x_gpu(2)
    # 参数化装饰器：测试混合精度是否为 True 和 False 时的情况
    @parametrize("mixed_precision", [True, False])
    # 测试方法：测试初始化后改变 FSDP 模型参数值是否持久化
    def test_param_change_after_init(self, mixed_precision):
        """
        Tests that changing FSDP model parameter values in-place after FSDP
        initialization persist.
        """
        # 建立参考行为
        fsdp_kwargs = {}
        # 如果 mixed_precision 为 True，则设置 "mixed_precision" 键为 MixedPrecision 对象
        if mixed_precision:
            fsdp_kwargs["mixed_precision"] = MixedPrecision()
        # 使用 TransformerWithSharedParams.init 初始化 FSDP 模型
        fsdp_model = TransformerWithSharedParams.init(
            self.process_group,
            FSDPInitMode.RECURSIVE,
            CUDAInitMode.CUDA_AFTER,
            fsdp_kwargs,
            deterministic=True,
        )
        # 获取 CUDA 设备上的输入数据
        input = fsdp_model.module.get_input(torch.device("cuda"))
        # 计算参考输出
        ref_output = fsdp_model(*input)
        # 初始化一个相同的模型，但在 FSDP 初始化后改变其第一个参数的值
        new_fsdp_model = TransformerWithSharedParams.init(
            self.process_group,
            FSDPInitMode.RECURSIVE,
            CUDAInitMode.CUDA_AFTER,
            fsdp_kwargs,
            deterministic=True,
        )
        # 获取新模型的第一个参数并使用正态分布初始化其数据
        first_param = next(new_fsdp_model.parameters())
        nn.init.normal_(first_param.data)
        # 计算新模型的输出
        new_output = new_fsdp_model(*input)
        # 断言：验证 new_output 反映了初始化后参数的变化
        self.assertNotEqual(
            ref_output,
            new_output,
            msg="new_output did not reflect change to param after init",
        )


# 定义一个测试类 TestHooks，继承自 FSDPTest，用于测试钩子函数
class TestHooks(FSDPTest):
    # 装饰器：如果 GPU 小于 2 个则跳过该测试
    @skip_if_lt_x_gpu(2)
    # 参数化装饰器：测试 cuda_first 参数为 False 和 True 时的情况
    @parametrize("cuda_first", [False, True])
    # 测试方法：测试 FSDP 前向传播钩子是否在前向传播输出时注册
    def test_pre_backward_hook_registration(self, cuda_first: bool):
        """Tests that FSDP pre-backward hooks are registered on forward pass
        outputs."""
        # 使用 TransformerWithSharedParams.init 初始化 FSDP 模型
        fsdp_model = TransformerWithSharedParams.init(
            self.process_group,
            FSDPInitMode.RECURSIVE,
            CUDAInitMode.CUDA_BEFORE if cuda_first else CUDAInitMode.CUDA_AFTER,
        )
        # 调用 _test_pre_backward_hook_registration 方法验证钩子函数是否注册
        self._test_pre_backward_hook_registration(fsdp_model)

    # 装饰器：如果 GPU 小于 2 个则跳过该测试
    @skip_if_lt_x_gpu(2)
    # 测试方法：测试保存和加载模型后，FSDP 前向传播钩子是否注册
    def test_pre_backward_hook_registration_after_state_dict(self):
        """Tests that FSDP pre-backward hooks are registered on forward pass
        outputs after saving and loading the model from a checkpoint."""
        # 使用 TransformerWithSharedParams.init 初始化 FSDP 模型
        fsdp_model = TransformerWithSharedParams.init(
            self.process_group,
            FSDPInitMode.RECURSIVE,
            CUDAInitMode.CUDA_AFTER,
        )
        # 训练模型多个步骤，关闭自动类型转换
        self._train_for_several_steps(fsdp_model, num_steps=2, autocast=False)
        # 获取模型的状态字典并加载到模型中
        state_dict = fsdp_model.state_dict()
        fsdp_model.load_state_dict(state_dict)
        # 调用 _test_pre_backward_hook_registration 方法验证钩子函数是否注册
        self._test_pre_backward_hook_registration(fsdp_model)
    # 定义测试方法，用于验证预向后钩子的注册是否正确
    def _test_pre_backward_hook_registration(self, model):
        # 使用 SGD 优化器，学习率为 0.01，动量为 0.9
        optim = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        # 将梯度清零
        optim.zero_grad()
        # 获取模型输入，始终在 CUDA 设备上进行计算
        input = model.module.get_input(torch.device("cuda"))
        # 调用模型进行前向传播
        output = model(*input)
        # 检查是否已注册了一个预向后钩子
        self.assertEqual(len(output._backward_hooks), 1)
        # 计算模型损失，并将其转移到 CUDA 设备上
        loss = model.module.get_loss(input, output).cuda()
        # 执行反向传播
        loss.backward()
        # 检查预向后钩子是否仍然存在
        self.assertEqual(len(output._backward_hooks), 1)
        # 执行优化步骤
        optim.step()
        # 再次检查预向后钩子是否仍然存在
        self.assertEqual(len(output._backward_hooks), 1)

    # 装饰器，如果 GPU 数量小于 2，则跳过测试
    @skip_if_lt_x_gpu(2)
    # 参数化装饰器，测试两种情况：cuda_first 为 False 或 True
    @parametrize("cuda_first", [False, True])
    # 参数化装饰器，测试两种情况：mixed_precision 为 True 或 False
    @parametrize("mixed_precision", [True, False])
    # 测试函数，验证 _register_{pre|post}_backward_hooks() 是否被调用
    def test_register_functions_called(self, cuda_first: bool, mixed_precision: bool):
        """Tests that ``_register_{pre|post}_backward_hooks()`` are called
        during the FSDP forward."""
        # 设置 FSDP 参数
        fsdp_kwargs = {}
        # 如果 mixed_precision 为 True，则设置 mixed_precision 为 MixedPrecision 对象
        if mixed_precision:
            fsdp_kwargs["mixed_precision"] = MixedPrecision()
        # 初始化 FSDP 模型
        fsdp_model = TransformerWithSharedParams.init(
            self.process_group,
            FSDPInitMode.RECURSIVE,
            CUDAInitMode.CUDA_BEFORE if cuda_first else CUDAInitMode.CUDA_AFTER,
            fsdp_kwargs,
        )
        # 获取模型输入，并放置在 CUDA 设备上
        input = fsdp_model.module.get_input(torch.device("cuda"))

        # 由于 `_register_pre_backward_hooks()` 修改了前向输出，
        # 我们无法直接模拟它，因此我们自己实现计数器。
        # 这里实现了一个计数器来统计 _register_pre_backward_hooks() 的调用次数
        orig_register_pre_backward_hooks = (
            torch.distributed.fsdp._runtime_utils._register_pre_backward_hooks
        )
        register_pre_backward_hooks_call_count = 0

        def _register_pre_backward_hooks_with_count(*args, **kwargs):
            nonlocal register_pre_backward_hooks_call_count
            register_pre_backward_hooks_call_count += 1
            return orig_register_pre_backward_hooks(*args, **kwargs)

        # 使用 mock.patch 替换 _register_pre_backward_hooks() 方法，
        # 并将替换后的方法用于计数
        with mock.patch(
            "torch.distributed.fsdp._runtime_utils._register_pre_backward_hooks",
            _register_pre_backward_hooks_with_count,
        ), mock.patch(
            "torch.distributed.fsdp._runtime_utils._register_post_backward_hook"
        ) as register_post_bwd_mock:
            # 断言在开始时，_register_pre_backward_hooks() 尚未被调用
            self.assertEqual(register_pre_backward_hooks_call_count, 0)
            # 断言在开始时，_register_post_backward_hook() 未被调用
            self.assertFalse(register_post_bwd_mock.called)
            # 执行 FSDP 模型的前向传播
            fsdp_model(*input)
            # 断言 _register_pre_backward_hooks() 至少被调用了一次
            self.assertTrue(register_pre_backward_hooks_call_count > 0)
            # 断言 _register_post_backward_hook() 被调用
            self.assertTrue(register_post_bwd_mock.called)
class TestNoGrad(FSDPTest):
    @skip_if_lt_x_gpu(2)
    @parametrize("mixed_precision", [True, False])
    def test_transformer_no_grad(self, mixed_precision):
        """Tests that for an FSDP-wrapped transformer model with shared
        parameters, after training for one iteration, running a forward pass in
        ``eval()`` mode gives the same output as running a forward pass in
        ``torch.no_grad()``."""
        
        # 准备 FSDP 参数设置字典
        fsdp_kwargs = {}
        if mixed_precision:
            fsdp_kwargs["mixed_precision"] = MixedPrecision(
                param_dtype=torch.float16,
                reduce_dtype=torch.float16,
                buffer_dtype=torch.float16,
            )
        else:
            fsdp_kwargs["mixed_precision"] = None
        
        # 初始化 FSDP 模型
        fsdp_model = TransformerWithSharedParams.init(
            self.process_group,
            FSDPInitMode.RECURSIVE,
            CUDAInitMode.CUDA_AFTER,
            fsdp_kwargs,
        )
        
        # 对 FSDP 模型进行一定步数的训练
        self._train_for_several_steps(
            fsdp_model,
            num_steps=1,
            autocast=False,
            mixed_precision=fsdp_kwargs["mixed_precision"],
        )
        
        # 获取模型输入数据
        input = fsdp_model.module.get_input(torch.device("cuda"))
        
        # 运行 eval 模式下的前向传播
        fsdp_model.eval()
        ref_output = fsdp_model(*input)
        
        # 使用 `no_grad()` 运行前向传播，并进行比较
        with torch.no_grad():
            no_grad_output = fsdp_model(*input)
        
        # 断言两种方式的输出结果应该一致
        self.assertEqual(ref_output, no_grad_output)


class TestAutograd(FSDPTest):
    @skip_if_lt_x_gpu(2)
    def test_unshard_params_as_tensors(
        self,
    ):
        """
        Tests that FSDP always unshards the logical parameters as ``Tensor``
        views during forward and backward computation even when forward and/or
        backward prefetching.
        """
        
        # 运行子测试，针对不同的参数组合
        self.run_subtests(
            {
                "sharding_strategy": [
                    ShardingStrategy.FULL_SHARD,
                    ShardingStrategy.SHARD_GRAD_OP
                    # 跳过 `NO_SHARD` 的测试，因为它使用了 `_use_unsharded_views()` 来测试分片视图。
                    # 测试 `FULL_SHARD` 和 `SHARD_GRAD_OP` 能够充分确认 `as_params` 逻辑的正确性。
                ],
                "use_orig_params": [False, True],
                "forward_prefetch": [False, True],
                "backward_prefetch": [
                    BackwardPrefetch.BACKWARD_PRE,
                    BackwardPrefetch.BACKWARD_POST,
                    None,
                ],
            },
            self._test_unshard_params_as_tensors,
        )

    def _test_unshard_params_as_tensors(
        self,
        sharding_strategy: ShardingStrategy,
        use_orig_params: bool,
        forward_prefetch: bool,
        backward_prefetch: Optional[BackwardPrefetch],
    ):
        # 实际测试函数，测试 FSDP 在前向和反向计算中始终将逻辑参数作为 `Tensor` 视图解析。
        pass
    ):
        orig_use_unsharded_views = FlatParamHandle._use_unsharded_views
        # 保存原始的 _use_unsharded_views 方法引用，以便后续恢复使用

        def _use_unsharded_views_assert_as_tensors(
            self: FlatParamHandle, as_params: bool
        ) -> None:
            # 定义一个新的方法 _use_unsharded_views_assert_as_tensors，用于断言是否使用张量视图而非参数视图
            _p_assert(
                not as_params, "Expects to use Tensor views but using parameter views"
            )
            return orig_use_unsharded_views(self, as_params)
            # 调用原始方法 _use_unsharded_views，并返回其结果

        fsdp_kwargs = {
            "sharding_strategy": sharding_strategy,
            "use_orig_params": use_orig_params,
            "forward_prefetch": forward_prefetch,
            "backward_prefetch": backward_prefetch,
            "auto_wrap_policy": ModuleWrapPolicy({nn.Linear}),
        }
        # 定义 FSDP 参数的关键字参数

        device = torch.device("cuda")
        # 创建一个 CUDA 设备对象

        # 定义一个包含足够多 FSDP 实例以进行预取的模型
        NUM_LINEARS = 5
        model = nn.Sequential(
            *[nn.Linear(3, 3, device=device) for _ in range(NUM_LINEARS)]
        )
        fsdp_model = FSDP(model, **fsdp_kwargs)
        # 使用定义好的参数初始化 FSDP 模型

        self.assertEqual(len(list(FSDP.fsdp_modules(fsdp_model))), NUM_LINEARS + 1)
        # 断言 FSDP 模型中包含的模块数量是否为 NUM_LINEARS + 1

        for _ in range(3):
            inp = torch.randn((2, 3), device=device)
            with self._patch_use_unsharded_views(
                _use_unsharded_views_assert_as_tensors
            ):
                # 使用自定义的 _use_unsharded_views_assert_as_tensors 方法进行上下文管理
                loss = fsdp_model(inp).sum()
                loss.backward()

    @contextlib.contextmanager
    def _patch_use_unsharded_views(self, new_use_unsharded_views: Callable):
        orig_use_unsharded_views = FlatParamHandle._use_unsharded_views
        # 保存原始的 _use_unsharded_views 方法引用

        FlatParamHandle._use_unsharded_views = new_use_unsharded_views
        # 设置新的 _use_unsharded_views 方法作为当前类的 _use_unsharded_views 方法

        try:
            yield
            # 执行上下文管理器中的代码块
        finally:
            FlatParamHandle._use_unsharded_views = orig_use_unsharded_views
            # 恢复原始的 _use_unsharded_views 方法引用
# 使用函数 instantiate_parametrized_tests 实例化参数化测试，针对 TestHooks 类
instantiate_parametrized_tests(TestHooks)
# 使用函数 instantiate_parametrized_tests 实例化参数化测试，针对 TestParityWithDDP 类
instantiate_parametrized_tests(TestParityWithDDP)
# 使用函数 instantiate_parametrized_tests 实例化参数化测试，针对 TestNoGrad 类
instantiate_parametrized_tests(TestNoGrad)
# 使用函数 instantiate_parametrized_tests 实例化参数化测试，针对 TestParamInit 类
instantiate_parametrized_tests(TestParamInit)

# 检查当前模块是否作为主程序运行，如果是则执行 run_tests 函数
if __name__ == "__main__":
    run_tests()
```