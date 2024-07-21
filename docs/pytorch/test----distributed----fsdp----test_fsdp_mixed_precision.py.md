# `.\pytorch\test\distributed\fsdp\test_fsdp_mixed_precision.py`

```
# Owner(s): ["oncall: distributed"]

# 导入必要的库和模块
import contextlib                    # 用于上下文管理的工具库
import itertools                     # 用于创建迭代器的工具库
import os                            # 提供与操作系统交互的功能
import sys                           # 提供与 Python 解释器交互的功能
from functools import partial        # 提供函数部分应用的功能
from itertools import product        # 提供迭代器操作的功能
from typing import Any, Dict, List   # 提供类型注解支持

import torch                         # PyTorch 深度学习库
import torch.cuda.nccl as nccl       # PyTorch CUDA NCCL 包
import torch.nn as nn                # PyTorch 神经网络模块
import torch.nn.functional as F      # PyTorch 神经网络函数库
from torch import distributed as dist  # PyTorch 分布式通信模块
from torch.distributed._composable import fully_shard  # PyTorch 可组合的分片模块
from torch.distributed.fsdp import (
    BackwardPrefetch,               # FSDP 模块：反向预取
    CPUOffload,                     # FSDP 模块：CPU 卸载
    FullyShardedDataParallel as FSDP,  # FSDP 模块：完全分片数据并行
    MixedPrecision,                 # FSDP 模块：混合精度
    ShardingStrategy,               # FSDP 模块：分片策略
)
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler  # FSDP 模块：分片梯度缩放器
from torch.distributed.fsdp.wrap import ModuleWrapPolicy, size_based_auto_wrap_policy  # FSDP 模块：模块包装策略
from torch.nn import TransformerDecoderLayer, TransformerEncoderLayer  # PyTorch 神经网络模块：Transformer 解码器层、编码器层
from torch.nn.modules.batchnorm import _BatchNorm  # PyTorch 神经网络模块：批归一化类
from torch.optim.swa_utils import AveragedModel  # PyTorch 优化模块：平均模型
from torch.testing._internal.common_distributed import (
    SaveForwardInputsModel,         # 测试用例：保存前向输入模型
    skip_if_lt_x_gpu,               # 测试用例：如果 GPU 数量小于 x，则跳过
)
from torch.testing._internal.common_fsdp import (
    CUDAInitMode,                   # FSDP 测试工具：CUDA 初始化模式
    FSDPInitMode,                   # FSDP 测试工具：FSDP 初始化模式
    FSDPTest,                       # FSDP 测试工具：FSDP 测试基类
    subtest_name,                   # FSDP 测试工具：子测试名称
    TransformerWithSharedParams,    # FSDP 测试工具：共享参数的 Transformer 模型
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,  # 测试工具：实例化参数化测试
    parametrize,                    # 测试工具：参数化装饰器
    run_tests,                      # 测试工具：运行测试
    skip_but_pass_in_sandcastle_if,  # 测试工具：在沙堡环境中跳过但通过
    TEST_WITH_DEV_DBG_ASAN,         # 测试工具：是否在开发调试 ASAN 模式下运行
)

try:
    import torchvision              # 导入 torchvision 库
    HAS_TORCHVISION = True          # 如果导入成功，则为 True
except ImportError:
    HAS_TORCHVISION = False         # 如果导入失败，则为 False

# 在没有分布式环境时，输出消息并退出
skipIfNoTorchVision = skip_but_pass_in_sandcastle_if(
    not HAS_TORCHVISION, "no torchvision"
)

# 如果分布式不可用，则输出消息并退出
if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)

# 如果设置了开发调试 ASAN 模式，则输出消息并退出
if TEST_WITH_DEV_DBG_ASAN:
    print(
        "Skip dev-asan as torch + multiprocessing spawn have known issues",
        file=sys.stderr,
    )
    sys.exit(0)

# 不同的混合精度配置
# 默认配置：参数、缓冲区和减少操作都使用 float16 精度
default_mp = MixedPrecision(
    param_dtype=torch.float16,
    buffer_dtype=torch.float16,
    reduce_dtype=torch.float16,
)

# 仅使用减少操作的配置：仅缓冲区和参数在 float16 精度下通信
mp_only_reduce = MixedPrecision(reduce_dtype=torch.float16)

# 仅使用参数和缓冲区的配置：参数在 float16 精度下通信，缓冲区使用 float16 精度
mp_only_param_and_buf = MixedPrecision(
    param_dtype=torch.float16, buffer_dtype=torch.float16
)

# 不使用混合精度的配置：所有操作都使用全精度
mp_no_mixed_precision = MixedPrecision()

# 如果 NCCL 支持 bfloat16 精度，则增加一个使用不同缓冲区和减少操作精度的配置
nccl_supports_bf16 = dist.is_nccl_available() and nccl.version() >= (2, 10)
mp_configs = [default_mp, mp_only_reduce, mp_only_param_and_buf, mp_no_mixed_precision]
if nccl_supports_bf16:
    mp_diff_buffer_and_reduce = MixedPrecision(
        param_dtype=torch.float16,
        buffer_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
    )
    mp_configs.extend([mp_diff_buffer_and_reduce])

# 缓冲区的原始数据类型，默认为 float64
_BUFFER_ORIG_DTYPE = torch.float64
params = "mp_config,cpu_offload,full_precision_param_dtype,enable_sharded_grad_scaler"
cpu_offload_config = [CPUOffload(offload_params=True), CPUOffload(offload_params=False)]
full_precision_param_dtype_config = [torch.float32, torch.float64]
enable_sharded_grad_scaler = ["enable_sharded_grad_scaler", None]

configs = list(
    product(
        mp_configs,  # 所有的 mp_configs 组合
        cpu_offload_config,  # CPU offload 配置
        full_precision_param_dtype_config,  # 全精度参数数据类型配置
        enable_sharded_grad_scaler,  # 启用分片梯度缩放器配置
    )
)

test_name_mapping = {
    str(CPUOffload(offload_params=True)): "offload_true",  # CPU offload 参数为 True 的映射
    str(CPUOffload(offload_params=False)): "offload_false",  # CPU offload 参数为 False 的映射
    str(default_mp): "mp_fp16",  # 默认的混合精度映射为 fp16
    str(mp_only_reduce): "mp_only_reduce",  # 仅减少映射
    str(mp_only_param_and_buf): "mp_only_param_and_buf",  # 仅参数和缓冲区映射
    str(mp_no_mixed_precision): "mp_no_mp",  # 没有混合精度映射
    str(torch.float32): "fp32",  # torch.float32 映射为 fp32
    str(torch.float64): "fp64",  # torch.float64 映射为 fp64
    "enable_sharded_grad_scaler": "enable_sharded_grad_scaler",  # 启用分片梯度缩放器映射
}

if nccl_supports_bf16:
    test_name_mapping.update(
        {
            str(mp_diff_buffer_and_reduce): "mp_diff_buffer_reduce",  # 不同缓冲区和减少的映射
        }
    )

subtest_name = partial(subtest_name, test_name_mapping)

_CURRENT_FULL_PRECISION_PARAM_DTYPE = None


@contextlib.contextmanager
def patch_reduce_scatter(new_reduce_scatter, full_precision_param_dtype):
    """
    Patches ``dist.reduce_scatter_tensor`` with ``new_reduce_scatter`` and
    restores upon exiting. Used for validation of mixed precision.
    """
    orig_reduce_scatter = dist.reduce_scatter_tensor  # 保存原始的 reduce scatter 张量函数
    dist.reduce_scatter_tensor = new_reduce_scatter  # 替换为新的 reduce scatter 张量函数
    global _CURRENT_FULL_PRECISION_PARAM_DTYPE
    _CURRENT_FULL_PRECISION_PARAM_DTYPE = full_precision_param_dtype  # 设置当前全精度参数数据类型
    try:
        yield  # 执行相关代码块
    finally:
        dist.reduce_scatter_tensor = orig_reduce_scatter  # 恢复原始的 reduce scatter 张量函数
        _CURRENT_FULL_PRECISION_PARAM_DTYPE = None  # 重置当前全精度参数数据类型为 None


class LinearMixedPrecision(nn.Module):
    """
    A linear module with extra checks for mixed precision training.
    """

    def __init__(self, param_dtype, buffer_name="buffer", run_checks=True):
        super().__init__()
        self.lin = nn.Linear(10, 10, bias=False).to(param_dtype)  # 使用给定的参数数据类型创建线性层
        # 使用可配置的缓冲区名称，避免所有子模块共享相同的缓冲区名称，可能隐藏前缀与非前缀名称的错误
        self.buffer_name = buffer_name  # 设置缓冲区名称
        self.register_buffer(buffer_name, torch.randn((1, 2), dtype=_BUFFER_ORIG_DTYPE))  # 注册缓冲区并初始化为随机数
        self._orig_param_type = param_dtype  # 原始参数数据类型
        self._orig_buffer_dtype = _BUFFER_ORIG_DTYPE  # 原始缓冲区数据类型
        self.run_checks = run_checks  # 是否运行检查标志


class TestFSDPMixedPrecision(FSDPTest):
    @property
    def world_size(self):
        raise ValueError("To be implemented by child classes")  # 由子类实现的抽象方法

    def _get_simple_nested_model(
        self, param_dtype, run_checks, *fsdp_args, **fsdp_kwargs
    ):
        # 根据参数创建一个复杂的嵌套模型，使用FSDP进行分布式训练和混合精度计算
        model = FSDP(
            nn.Sequential(
                # 使用线性混合精度计算，并将其放置在CUDA设备上
                FSDP(
                    LinearMixedPrecision(
                        param_dtype, buffer_name="buffer0", run_checks=run_checks
                    ).cuda(),
                    *fsdp_args,
                    **fsdp_kwargs,
                ),
                # 使用线性混合精度计算，并将其放置在CUDA设备上
                LinearMixedPrecision(
                    param_dtype, buffer_name="buffer1", run_checks=run_checks
                ).cuda(),
            ),
            *fsdp_args,
            **fsdp_kwargs,
        )
        # 返回构建好的模型
        return model

    def _get_simple_nested_model_composable(
        self, param_dtype, run_checks, *fsdp_args, **fsdp_kwargs
    ):
        # 创建一个简单的嵌套模型，每个层级都使用线性混合精度计算，并放置在CUDA设备上
        model = nn.Sequential(
            LinearMixedPrecision(
                param_dtype, buffer_name="buffer0", run_checks=run_checks
            ).cuda(),
            LinearMixedPrecision(
                param_dtype, buffer_name="buffer1", run_checks=run_checks
            ).cuda(),
        )
        # 对第一个层级进行全局分片处理
        fully_shard(model[0], *fsdp_args, **fsdp_kwargs)
        # 对整个模型进行全局分片处理
        fully_shard(model, *fsdp_args, **fsdp_kwargs)
        # 返回构建好的模型
        return model

    def _get_simple_model(self, param_dtype, *fsdp_args, **fsdp_kwargs):
        # 创建一个简单的模型，使用线性混合精度计算，并放置在CUDA设备上
        model = FSDP(
            LinearMixedPrecision(param_dtype).cuda(), *fsdp_args, **fsdp_kwargs
        )
        # 返回构建好的模型
        return model

    def _validate_no_mp_shard(self, fsdp_model):
        """
        Validates that there is no mixed precision _mp_shard allocated
        when it is not expected to be.
        """
        # 获取FSDP单元并验证没有混合精度_mp_shard分配
        fsdp_units = FSDP.fsdp_modules(fsdp_model)
        for fsdp in fsdp_units:
            for param in fsdp.params:
                self.assertFalse(hasattr(param, "_mp_shard"))

    def _validate_mp_shard_freed(self, fsdp_model):
        """
        Ensures that the mixed precision shard is freed for all FSDP units.
        """
        # 获取FSDP单元并验证所有FSDP单元的混合精度_shard是否被释放
        fsdp_units = FSDP.fsdp_modules(fsdp_model)
        for fsdp in fsdp_units:
            for param in fsdp.params:
                # 确保混合精度_shard的存储大小为0
                self.assertEqual(0, param._mp_shard.untyped_storage().size())

    def _reduce_scatter_validate_mp(
        self, orig_reduce_scatter, mp_config, should_run_low_prec, *args, **kwargs
        ):
        """
        Runs reduce-scatter but verifies mixed precision settings before. This
        is to test mixed precision is working as expected during backward pass.
        In particular it ensures that the gradients were cast to the right type
        and comm. is going to happen in the right type.
        """
        tensors = []
        for x in args:
            if isinstance(x, torch.Tensor):
                tensors.append(x)
        for x in kwargs.values():
            if isinstance(x, torch.Tensor):
                tensors.append(x)

        # reduce_dtype has higher priority than param_dtype, because mixed_precision
        # supports overriding param_dtype with reduce_dtype to control the
        # reduction precision. In the case where reduce_dtype == param_dtype
        # this tests that gradients are in the expected precision as well.
        # If reduce_dtype is not specified (is None) we comm. in the param_dtype
        # if that is specified, otherwise full precision dtype.
        if should_run_low_prec:
            expected_dtype = (
                mp_config.reduce_dtype
                if mp_config.reduce_dtype is not None
                else (
                    mp_config.param_dtype
                    if mp_config.param_dtype is not None
                    else _CURRENT_FULL_PRECISION_PARAM_DTYPE
                )
            )
        else:
            expected_dtype = _CURRENT_FULL_PRECISION_PARAM_DTYPE

        for t in tensors:
            # Ensure each tensor in tensors has the expected_dtype
            self.assertEqual(
                expected_dtype,
                t.dtype,
                f"Expected to reduce in {expected_dtype} but got tensors in {t.dtype}",
            )

        # Call the original reduce_scatter function with the provided arguments
        return orig_reduce_scatter(*args, **kwargs)

    def _test_grads_reduced_precision(
        self, offload_params: bool, use_orig_params: bool
    ):
        class MyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.lin1 = nn.Linear(10, 10)
                self.lin2 = nn.Linear(10, 10)

            def forward(self, x):
                return self.lin2(self.lin1(x))

        m = MyModel().cuda()
        mp = MixedPrecision(
            param_dtype=torch.float16,
            reduce_dtype=torch.float16,
            buffer_dtype=torch.float16,
            keep_low_precision_grads=True,
        )
        fsdp_kwargs = {
            "mixed_precision": mp,
            "cpu_offload": CPUOffload(offload_params=offload_params),
            "use_orig_params": use_orig_params,
        }
        # Apply FSDP (Fully Sharded Data Parallel) to m.lin1 and m with given kwargs
        m.lin1 = FSDP(m.lin1, **fsdp_kwargs)
        m = FSDP(m, **fsdp_kwargs)

        # Perform backward pass 6 times with torch.float16 precision
        for _ in range(6):
            inp = torch.ones(1, 10)
            # Forward pass and compute gradients
            m(inp).sum().backward()
            # Check that all gradients have torch.float16 dtype
            for param in m.parameters():
                if param.grad is not None:
                    self.assertEqual(torch.float16, param.grad.dtype)

        # Synchronize all processes in distributed training
        dist.barrier()
    # 定义一个私有方法 `_run_test_mixed_precision_e2e`，用于执行混合精度端到端测试
    # 参数说明:
    # - mp_config: 混合精度配置，包括精度位数等设置
    # - cpu_offload: 是否开启 CPU 卸载
    # - backward_prefetch: 是否开启反向传播预取
    # - forward_prefetch: 是否开启前向传播预取
    # - full_precision_param_dtype: 完全精度参数的数据类型
    # - sharding_strategy: 分片策略，用于模型或梯度的分片处理
    # - enable_sharded_grad_scaler: 是否启用分片梯度缩放器
# 定义一个测试类 `TestFSDPMixedPrecisionSharded`，继承自 `TestFSDPMixedPrecision`
class TestFSDPMixedPrecisionSharded(TestFSDPMixedPrecision):

    # 定义 `world_size` 属性，返回值为 2
    @property
    def world_size(self):
        return 2

    # 返回一个包含子测试配置的字典，用于测试预取设置
    def _get_subtest_config(self) -> Dict[str, List[Any]]:
        """Returns a subtest configuration that subtests prefetching settings
        together."""
        return {
            "forward_prefetch": [False, True],
            "backward_prefetch": [
                None,
                BackwardPrefetch.BACKWARD_PRE,
                BackwardPrefetch.BACKWARD_POST,
            ],
        }

    # 如果 GPU 数量小于 2，则跳过测试
    @skip_if_lt_x_gpu(2)
    def test_mixed_precision_no_reshard_after_forward(self):
        # 注意：为了不增加测试时间太多，这里并未测试所有可能的不同配置
        # 根据 nccl 支持情况选择混合精度配置
        mp = default_mp if not nccl_supports_bf16 else mp_diff_buffer_and_reduce
        # 运行混合精度端到端测试
        self._run_test_mixed_precision_e2e(
            mp_config=mp,
            cpu_offload=CPUOffload(offload_params=True),
            backward_prefetch=None,
            forward_prefetch=False,
            full_precision_param_dtype=torch.float64,
            sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,
            enable_sharded_grad_scaler=False,
        )

    # 如果 GPU 数量小于 2，则跳过测试；参数化测试函数
    @skip_if_lt_x_gpu(2)
    @parametrize(params, configs, subtest_name)
    def test_mixed_precision_e2e_full_shard(
        self,
        mp_config,
        cpu_offload,
        full_precision_param_dtype,
        enable_sharded_grad_scaler,
    ):
        # 运行子测试
        self.run_subtests(
            self._get_subtest_config(),
            self._run_test_mixed_precision_e2e,
            mp_config=mp_config,
            cpu_offload=cpu_offload,
            full_precision_param_dtype=full_precision_param_dtype,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            enable_sharded_grad_scaler=enable_sharded_grad_scaler,
        )
    def _test_mixed_precision_embedding_table(self, mp_config):
        # Basic test to ensure int inputs are not casted which would break
        # modules such as embedding tables.
        
        # 确定参数的数据类型，默认为 torch.float32，如果 mp_config 中指定了则使用指定的数据类型
        param_dtype = mp_config.param_dtype or torch.float32
        
        # 保存原始的 dist.reduce_scatter_tensor 函数
        orig_reduce_scatter = dist.reduce_scatter_tensor
        
        # 创建一个部分应用函数 test_reduce_scatter，用于验证混合精度模式下的 reduce_scatter_tensor 函数
        test_reduce_scatter = partial(
            self._reduce_scatter_validate_mp,
            orig_reduce_scatter,
            mp_config,
            True,
        )
        
        # 使用 patch_reduce_scatter 上下文管理器来替换 reduce_scatter_tensor 函数，
        # 参数使用 param_dtype 指定的数据类型
        with patch_reduce_scatter(test_reduce_scatter, param_dtype):
            # TODO: 如果不对整个 `TransformerWithSharedParams` 使用顶层的 FSDP，
            # 则 `test_mp_embedding_reduce()` 将失败
            # 使用指定的 mp_config 创建 TransformerWithSharedParams 模型
            model = TransformerWithSharedParams.init(
                self.process_group,
                FSDPInitMode.NO_FSDP,
                CUDAInitMode.CUDA_BEFORE,
                {"mixed_precision": mp_config},
            )
            
            # 使用 FSDP 将模型包装起来，指定混合精度配置
            fsdp_model = FSDP(model, mixed_precision=mp_config)
            
            # 使用 SGD 优化器来优化 fsdp_model 的参数
            optim = torch.optim.SGD(fsdp_model.parameters(), lr=0.1)
            
            # 执行 6 次迭代
            for _ in range(6):
                # 获取模型输入，并将其传输到 CUDA 设备上
                inp = fsdp_model.module.get_input(torch.device("cuda"))
                
                # 对模型进行前向计算
                output = fsdp_model(*inp)
                
                # 计算损失，并将其转移到 CUDA 设备上
                loss = fsdp_model.module.get_loss(inp, output).cuda()
                
                # 断言损失的数据类型为 param_dtype 指定的数据类型
                self.assertEqual(loss.dtype, param_dtype)
                
                # 对损失进行反向传播
                fsdp_model.module.run_backward(loss)
                
                # 更新优化器的参数
                optim.step()

    @skip_if_lt_x_gpu(2)
    def test_mp_embedding_reduce(self):
        # 调用 _test_mixed_precision_embedding_table 方法，使用指定的混合精度配置
        self._test_mixed_precision_embedding_table(
            mp_config=MixedPrecision(reduce_dtype=torch.float16)
        )

    @skip_if_lt_x_gpu(2)
    def test_mp_embedding_only_params_and_bufs(self):
        # 调用 _test_mixed_precision_embedding_table 方法，使用指定的混合精度配置
        self._test_mixed_precision_embedding_table(
            mp_config=MixedPrecision(
                param_dtype=torch.float16,
                buffer_dtype=torch.float16,
            )
        )

    @skip_if_lt_x_gpu(2)
    def test_mp_embedding_default(self):
        # 创建一个默认的混合精度配置
        default_mp_config = MixedPrecision(
            param_dtype=torch.float16,
            buffer_dtype=torch.float16,
            reduce_dtype=torch.float16,
        )
        # 调用 _test_mixed_precision_embedding_table 方法，使用默认的混合精度配置
        self._test_mixed_precision_embedding_table(mp_config=default_mp_config)

    @skip_if_lt_x_gpu(2)
    def test_mp_embedding_params_and_reduce_diff(self):
        # 创建一个包含不同参数和 reduce 数据类型的混合精度配置
        params_and_reduce_different = MixedPrecision(
            param_dtype=torch.float16,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.float16,
        )
        # 调用 _test_mixed_precision_embedding_table 方法，使用指定的混合精度配置
        self._test_mixed_precision_embedding_table(
            mp_config=params_and_reduce_different
        )

    @skip_if_lt_x_gpu(2)
    @skipIfNoTorchVision
    def test_mixed_precision_resnet(self):
        """
        End to end test to ensure mixed precision + auto_wrap works
        for ResNet model.
        """
        # 创建一个 ResNet50 模型并移动到 GPU
        resnet_model = torchvision.models.resnet50().cuda()
        # 将模型中的 BatchNorm 转换为同步 BatchNorm
        resnet_model = nn.SyncBatchNorm.convert_sync_batchnorm(
            resnet_model, process_group=dist.distributed_c10d._get_default_group()
        )
        # 统计模型中的 BatchNorm 层数量
        n_bn = sum(
            1 if isinstance(x, _BatchNorm) else 0 for x in resnet_model.modules()
        )
        # 创建一个在 GPU 上的输入张量
        inp = torch.ones(1, 3, 1000, 1000, device="cuda")
        # 定义混合精度配置
        mp_config = MixedPrecision(
            param_dtype=torch.float16,
            reduce_dtype=torch.float16,
            buffer_dtype=torch.float16,
        )
        # 使用 FSDP 封装 ResNet 模型，启用基于大小的自动封装策略和混合精度配置
        fsdp = FSDP(
            resnet_model,
            auto_wrap_policy=size_based_auto_wrap_policy,
            mixed_precision=mp_config,
        )
        
        # 遍历 FSDP 中的模块，统计 BatchNorm 的数量
        fsdp_bn = 0
        for module in fsdp.fsdp_modules(fsdp):
            wrapped_module = module.module
            if isinstance(wrapped_module, _BatchNorm):
                fsdp_bn += 1
        
        # 断言 FSDP 中的 BatchNorm 数量与原始 ResNet 模型中的一致
        self.assertEqual(fsdp_bn, n_bn)
        
        # 在没有混合精度自动封装的情况下，会引发类型不匹配的问题
        loss = fsdp(inp).sum()
        loss.backward()

    @skip_if_lt_x_gpu(2)
    def test_grads_reduced_precision(self):
        # 执行子测试，针对不同的参数组合
        self.run_subtests(
            {
                "offload_params": [False, True],
                "use_orig_params": [False, True],
            },
            self._test_grads_reduced_precision,
        )

    @skip_if_lt_x_gpu(2)
    @parametrize("convert_sync_bn", [True, False])
    # 定义一个测试方法，用于测试带有批量归一化的神经网络模型
    def test_mp_batchnorm(self, convert_sync_bn):
        # 定义一个简单的神经网络模型，包含线性层、批量归一化层、Layer归一化层和线性层
        class BatchNormNet(nn.Module):
            def __init__(self, affine=True):
                super().__init__()
                self.fc1 = nn.Linear(2, 40, bias=False)
                self.bn = nn.BatchNorm1d(4, affine=affine)  # 添加批量归一化层
                self.fc2 = nn.Linear(40, 4, bias=False)
                self.ln = nn.LayerNorm(4)  # 添加Layer归一化层
                self.fc3 = nn.Linear(4, 4, bias=False)

            def forward(self, x):
                x = torch.reshape(self.fc1(x), (-1, 4, 10))  # 对输入数据进行形状变换
                x = self.bn(x)  # 对数据应用批量归一化
                x = torch.reshape(x, (-1, 40))  # 再次对数据进行形状变换
                x = self.fc2(x)  # 应用第二个线性层
                x = self.ln(x)  # 对输出数据应用Layer归一化
                x = self.fc3(x)  # 应用最后一个线性层
                return F.softmax(x, dim=1)  # 返回经过softmax处理的输出

        # 定义一个函数，始终返回False，用于设置自动包装策略
        def never_wrap_policy(*args, **kwargs):
            return False

        # 创建一个BatchNormNet实例，并将其放到GPU上
        net = BatchNormNet().cuda()

        # 如果指定需要转换同步批量归一化，则调用convert_sync_batchnorm函数进行转换
        if convert_sync_bn:
            net = nn.SyncBatchNorm.convert_sync_batchnorm(net)

        # 配置混合精度参数
        mp_config = MixedPrecision(
            param_dtype=torch.float16,
            reduce_dtype=torch.float16,
            buffer_dtype=torch.float16,
            _module_classes_to_ignore=[_BatchNorm, nn.LayerNorm],
        )

        # 使用FSDP对模型进行包装，以解决混合精度与批量归一化结合可能引起的问题，并测试警告输出
        with self.assertWarnsRegex(
            expected_warning=UserWarning,
            expected_regex="These modules will be wrapped as separate FSDP",
        ):
            model = FSDP(
                net,
                mixed_precision=mp_config,
                auto_wrap_policy=never_wrap_policy,
            )

        # 创建一个无混合精度设置的MixedPrecision实例
        no_mp = MixedPrecision()

        # 检查模型中的Layer归一化和批量归一化是否都被正确地包装为FSDP单元
        for mod in [model.ln, model.bn]:
            self.assertTrue(isinstance(mod, FSDP))
            self.assertEqual(no_mp, mod.mixed_precision)

        # 检查模型中的线性层是否未被包装为FSDP单元
        for mod in [model.fc1, model.fc2, model.fc3]:
            self.assertFalse(isinstance(mod, FSDP))

        # 验证整体模型是否仍启用了混合精度
        self.assertEqual(mp_config, model.mixed_precision)

        # 创建一个GPU上的随机输入张量
        inp = torch.randn((1, 2), device="cuda")

        # 调用模型进行前向传播，并对输出进行求和和反向传播
        # 如果没有FSDP批量归一化混合精度修复，这里可能会引发运行时错误
        model(inp).sum().backward()

    @skip_if_lt_x_gpu(2)
    def test_eval_root_cast_inputs(self):
        """
        In a case where root module does not manage FSDP parameters,
        ensure that we don't cast forward inputs which could potentially
        cause a dtype mismatch. Check that FSDP_USE_FULL_PREC_IN_EVAL controls
        this.
        """

        low_prec_dtype = torch.float16  # 设置低精度数据类型为 torch.float16

        class MyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.a = nn.Linear(5, 5)  # 创建一个线性层，输入输出维度为 5

            def forward(self, x, expect_use_full_prec_in_eval):
                if expect_use_full_prec_in_eval:
                    assert x.dtype == torch.float32, f"Expected fp32, got {x.dtype}"  # 如果期望使用全精度，在评估时输入应为 torch.float32
                else:
                    assert (
                        x.dtype == low_prec_dtype
                    ), f"Expected {low_prec_dtype}, got {x.dtype}"  # 否则，输入应为预设的低精度类型

                return self.a(x)  # 返回线性层的输出

        mp_config = MixedPrecision(
            param_dtype=low_prec_dtype,
            reduce_dtype=low_prec_dtype,
            buffer_dtype=low_prec_dtype,
        )  # 创建混合精度配置对象，设置参数、缓存和规约的数据类型都为低精度类型

        for use_full_prec_in_eval in [True, False]:
            os.environ["FSDP_USE_FULL_PREC_IN_EVAL"] = (
                "1" if use_full_prec_in_eval else "0"
            )  # 根据 use_full_prec_in_eval 的值设置环境变量 FSDP_USE_FULL_PREC_IN_EVAL

            m = MyModel().cuda()  # 创建 MyModel 实例并移动到 GPU 上
            m.a = FSDP(m.a, mixed_precision=mp_config)  # 使用 FSDP 封装线性层 a，设置混合精度配置
            model = FSDP(m, mixed_precision=mp_config)  # 使用 FSDP 封装整个模型，设置混合精度配置
            model.eval()  # 设置模型为评估模式
            inp = torch.randn(5, 5)  # 创建一个 5x5 的随机输入张量
            model(inp, use_full_prec_in_eval).sum().backward()  # 将输入传递给模型并进行反向传播
    def test_full_precision_in_eval(self):
        """
        Tests that eval runs in full precision if FSDP_USE_FULL_PREC_IN_EVAL is set.
        """
        # 使用 itertools 的 product 函数生成所有可能的组合
        for (
            use_composable,  # 是否使用可组合模式
            cast_forward_inputs,  # 是否将输入向前转换
            use_full_prec_in_eval,  # 是否在评估时使用完整精度
        ) in itertools.product([True, False], [True, False], [True, False]):
            # 定义混合精度配置 MixedPrecision 对象
            mp_config = MixedPrecision(
                param_dtype=torch.float16,  # 参数数据类型为 float16
                reduce_dtype=torch.float16,  # 缩减操作数据类型为 float16
                buffer_dtype=torch.float16,  # 缓冲区数据类型为 float16
                cast_forward_inputs=cast_forward_inputs,  # 是否将输入向前转换
            )
            # 设置环境变量 FSDP_USE_FULL_PREC_IN_EVAL
            os.environ["FSDP_USE_FULL_PREC_IN_EVAL"] = (
                "1" if use_full_prec_in_eval else "0"
            )
            # 初始化 TransformerWithSharedParams 模型
            model = TransformerWithSharedParams.init(
                self.process_group,  # 进程组
                FSDPInitMode.NO_FSDP if use_composable else FSDPInitMode.RECURSIVE,  # 初始化模式
                CUDAInitMode.CUDA_BEFORE,  # CUDA 初始化模式
                {"mixed_precision": mp_config},  # 混合精度配置
            )
            # 如果使用可组合模式，自动包装策略为 ModuleWrapPolicy
            if use_composable:
                auto_wrap_policy = ModuleWrapPolicy(
                    {
                        TransformerEncoderLayer,
                        TransformerDecoderLayer,
                    }
                )
                fully_shard(model, policy=auto_wrap_policy, mixed_precision=mp_config)
            # 设置模块访问器
            module_accessor = model if use_composable else model
            # 获取模型输入并将其移到 CUDA 设备上
            inp = module_accessor.get_input(torch.device("cuda"))
            # 执行模型前向传播
            output = model(*inp)
            # 计算损失
            loss = module_accessor.get_loss(inp, output).cuda()
            # 断言损失数据类型为 float16
            self.assertEqual(torch.float16, loss.dtype)
            # 执行模型反向传播
            module_accessor.run_backward(loss)
            # 断言参数梯度数据类型为 float32，因为在反向传播时会向上转型为 float32
            for p in model.parameters():
                if p.grad is not None:
                    self.assertEqual(torch.float32, p.grad.dtype)

            # 进入评估模式，如果设置了 use_full_prec_in_eval，则损失应为 fp32
            model.eval()
            # 再次获取输入并移到 CUDA 设备上
            inp = module_accessor.get_input(torch.device("cuda"))
            # 再次执行模型前向传播
            output = model(*inp)
            # 计算损失
            loss = module_accessor.get_loss(inp, output).cuda()
            # 预期损失数据类型为 torch.float32 或 torch.float16
            expected_dtype = torch.float32 if use_full_prec_in_eval else torch.float16
            self.assertEqual(expected_dtype, loss.dtype)

    @skip_if_lt_x_gpu(2)
    def test_full_precision_in_eval_buffers(self):
        """
        Tests that when model.eval() and FSDP_USE_FULL_PREC_IN_EVAL is set,
        buffers are in the full precision.
        """
        # 遍历三个布尔变量的组合：use_composable, cast_forward_inputs, use_full_prec_in_eval
        for (
            use_composable,
            cast_forward_inputs,
            use_full_prec_in_eval,
        ) in itertools.product([True, False], [True, False], [True, False]):
            # 根据 use_full_prec_in_eval 的值设置环境变量 FSDP_USE_FULL_PREC_IN_EVAL
            os.environ["FSDP_USE_FULL_PREC_IN_EVAL"] = (
                "1" if use_full_prec_in_eval else "0"
            )
            # 创建 MixedPrecision 对象 mp_config
            mp_config = MixedPrecision(
                param_dtype=torch.float16,
                reduce_dtype=torch.float16,
                buffer_dtype=torch.float16,
                cast_forward_inputs=cast_forward_inputs,
            )
            # 根据 use_composable 决定获取哪种模型构造函数
            model_getter = (
                self._get_simple_nested_model_composable
                if use_composable
                else self._get_simple_nested_model
            )
            # 根据参数创建 fsdp_model
            fsdp_model = model_getter(
                param_dtype=torch.float32,
                run_checks=False,
                mixed_precision=mp_config,
            )

            # 创建输入张量 inp，设备为 cuda
            inp = torch.randn(3, 10, device="cuda")
            # 调用 fsdp_model 进行前向传播
            fsdp_model((inp, self, fsdp_model, mp_config, torch.float32))
            # 遍历 fsdp_model 的所有缓冲区，并检查它们的数据类型是否为 torch.float16
            for buf in fsdp_model.buffers():
                self.assertEqual(torch.float16, buf.dtype)

            # 设置前向传播前的钩子函数，验证评估模式下缓冲区的数据类型
            def verify_eval_buffer_dtype(module, input):
                expected_dtype = (
                    _BUFFER_ORIG_DTYPE if use_full_prec_in_eval else torch.float16
                )
                for buf in module.buffers():
                    self.assertEqual(expected_dtype, buf.dtype)

            # 定义获取底层模块的函数
            def _get_underlying_module(m):
                return m.module if isinstance(m, FSDP) else m

            hook_handles = []
            # 注册前向传播前的钩子函数，用于验证评估模式下缓冲区的数据类型
            hook_handles.append(
                _get_underlying_module(fsdp_model[0]).register_forward_pre_hook(
                    verify_eval_buffer_dtype
                )
            )
            hook_handles.append(
                _get_underlying_module(fsdp_model[1]).register_forward_pre_hook(
                    verify_eval_buffer_dtype
                )
            )

            # 将模型设置为评估模式
            fsdp_model.eval()
            # 再次进行前向传播
            fsdp_model((inp, self, fsdp_model, mp_config, torch.float32))
            # 移除钩子函数
            for hook_handle in hook_handles:
                hook_handle.remove()

            # 验证评估后缓冲区的数据类型是否符合预期
            expected_dtype = (
                _BUFFER_ORIG_DTYPE if use_full_prec_in_eval else torch.float16
            )
            for buf in fsdp_model.buffers():
                self.assertEqual(expected_dtype, buf.dtype)

            # 将模型设置为训练模式
            fsdp_model.train()
            # 再次进行前向传播
            fsdp_model((inp, self, fsdp_model, mp_config, torch.float32))
            # 验证训练后缓冲区的数据类型是否为 torch.float16
            for buf in fsdp_model.buffers():
                self.assertEqual(torch.float16, buf.dtype)
    @skip_if_lt_x_gpu(2)
    # 标记为跳过测试，如果 GPU 少于两个
    def test_full_precision_in_eval_comm(self):
        # 对以下参数的所有组合进行迭代测试
        for (
            use_composable,
            cast_forward_inputs,
            use_full_prec_in_eval,
        ) in itertools.product([True, False], [True, False], [True, False]):
            # 设置环境变量，控制是否在评估过程中使用全精度
            os.environ["FSDP_USE_FULL_PREC_IN_EVAL"] = (
                "1" if use_full_prec_in_eval else "0"
            )
            # 定义混合精度配置
            mp_config = MixedPrecision(
                param_dtype=torch.float32,
                reduce_dtype=torch.float16,
                buffer_dtype=torch.float32,
                cast_forward_inputs=cast_forward_inputs,
                # 在本测试中仅为批归一化强制转换为减少精度，以简化验证过程。
                _module_classes_to_ignore=[],
            )
            # 初始化模型
            model = TransformerWithSharedParams.init(
                self.process_group,
                FSDPInitMode.NO_FSDP if use_composable else FSDPInitMode.RECURSIVE,
                CUDAInitMode.CUDA_BEFORE,
                {"mixed_precision": mp_config},
            )
            # 如果使用可组合模型，自动封装策略
            if use_composable:
                auto_wrap_policy = ModuleWrapPolicy(
                    {
                        TransformerEncoderLayer,
                        TransformerDecoderLayer,
                    }
                )
                fully_shard(model, policy=auto_wrap_policy, mixed_precision=mp_config)
            # 获取模型访问器
            model_accessor = model if use_composable else model.module
            # 用于添加混合精度类型验证的 reduce_scatter 修补程序
            orig_reduce_scatter = dist.reduce_scatter_tensor
            test_reduce_scatter = partial(
                self._reduce_scatter_validate_mp,
                orig_reduce_scatter,
                mp_config,
                not use_full_prec_in_eval,
            )
            # 将模型设置为评估模式
            model.eval()
            # 使用 patch_reduce_scatter 函数进行 reduce_scatter 的补丁，使用 torch.float32 进行验证
            with patch_reduce_scatter(test_reduce_scatter, torch.float32):
                # 获取模型输入
                inp = model_accessor.get_input(torch.device("cuda"))
                # 运行模型
                output = model(*inp)
                # 计算损失
                loss = model_accessor.get_loss(inp, output).cuda()
                # 执行反向传播
                model_accessor.run_backward(loss)

    @skip_if_lt_x_gpu(2)
    # 标记为跳过测试，如果 GPU 少于两个
    def test_input_grads_with_param_mixed_precision(self):
        """
        Tests that input tensors that require gradients do get their gradients
        even after being cast to a low precision (when parameter mixed
        precision is enabled).
        """
        # 运行子测试，测试输入梯度与参数混合精度
        self.run_subtests(
            {
                "sharding_strategy": [
                    ShardingStrategy.FULL_SHARD,
                    ShardingStrategy.SHARD_GRAD_OP,
                    ShardingStrategy.NO_SHARD,
                ],
                "use_orig_params": [False, True],
            },
            self._test_input_grads_with_param_mixed_precision,
        )

    def _test_input_grads_with_param_mixed_precision(
        self,
        sharding_strategy: ShardingStrategy,
        use_orig_params: bool,
        # 继续定义测试的具体实现
        ):
            # 创建一个具有1024个输入和1024个输出的线性模型，无偏置
            model = nn.Linear(1024, 1024, bias=False)
            # 创建一个混合精度训练的实例，指定参数精度为torch.float16，梯度和缓冲区精度为torch.float32
            mixed_precision = MixedPrecision(
                param_dtype=torch.float16,
                reduce_dtype=torch.float32,
                buffer_dtype=torch.float32,
            )
            # 使用FSDP进行模型分片和混合精度训练配置
            fsdp_model = FSDP(
                model,
                sharding_strategy=sharding_strategy,
                mixed_precision=mixed_precision,
                device_id=torch.cuda.current_device(),
                use_orig_params=use_orig_params,
            )
            # 创建一个在CUDA设备上的随机张量作为输入，指定数据类型为torch.float32
            # 这会触发类型转换，因为模型使用的是torch.float16
            x_float = torch.randn(
                (32, 1024),
                device="cuda",
                dtype=torch.float32,
                requires_grad=True,
            )
            # 将输入数据传递给模型，计算输出并对输出进行求和，然后反向传播梯度
            fsdp_model(x_float).sum().backward()
            # 断言输入张量x_float有梯度信息
            self.assertTrue(x_float.grad is not None)
            # 检查输入张量x_float的数据类型保持不变，表明梯度通过ToCopyBackward0正确传播
            self.assertEqual(x_float.grad.dtype, torch.float32)
class TestFSDPMixedPrecisionUnsharded(TestFSDPMixedPrecision):
    """
    Smaller test suite for unshared param (i.e. world_size == 1) case.
    """

    @property
    def world_size(self):
        # 返回测试中的世界大小，此处为 1
        return 1

    @skip_if_lt_x_gpu(1)
    def test_grads_reduced_precision(self):
        # 运行子测试，测试梯度在减少精度后的情况
        self.run_subtests(
            {"offload_params": [False, True], "use_orig_params": [False, True]},
            self._test_grads_reduced_precision,
        )

    @skip_if_lt_x_gpu(1)
    def test_mixed_precision_no_reshard_after_forward(self):
        # 注意，为了不增加测试时间太多，我们并未尝试所有可能的配置。
        # 运行端到端混合精度测试，验证前向传播后不重新分片的情况
        mp = default_mp if not nccl_supports_bf16 else mp_diff_buffer_and_reduce
        self._run_test_mixed_precision_e2e(
            mp_config=mp,
            cpu_offload=CPUOffload(offload_params=True),
            backward_prefetch=None,
            forward_prefetch=False,
            full_precision_param_dtype=torch.float64,
            sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,
            enable_sharded_grad_scaler=False,
        )

    @skip_if_lt_x_gpu(1)
    def test_mixed_precision_e2e_full_shard(self):
        # 运行端到端混合精度测试，验证完全分片的情况
        mp = default_mp if not nccl_supports_bf16 else mp_diff_buffer_and_reduce
        self._run_test_mixed_precision_e2e(
            mp_config=mp,
            cpu_offload=CPUOffload(offload_params=True),
            backward_prefetch=None,
            forward_prefetch=False,
            full_precision_param_dtype=torch.float64,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            enable_sharded_grad_scaler=False,
        )


instantiate_parametrized_tests(TestFSDPMixedPrecisionSharded)


class IgnoredModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.l = nn.Linear(100, 100)

    def forward(self, x):
        return self.l(x)


class ModelWithIgnoredModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(100, 100)
        self.ignored = IgnoredModule()  # 创建一个忽略模块的实例
        self.l2 = nn.Linear(100, 100)

    def forward(self, x):
        return self.l2(self.ignored(self.l1(x)))


class TestFSDPMixedPrecisionIgnoredModules(FSDPTest):
    @property
    def world_size(self):
        # 返回测试中的世界大小，此处为 1
        return 1

    @skip_if_lt_x_gpu(1)
    def test_mixed_precision_with_ignored_module(self):
        # 创建一个包含被忽略模块的模型，并将其移到 GPU 上
        model = ModelWithIgnoredModule().cuda()
        float16 = MixedPrecision(param_dtype=torch.float16)
        # 使用 FSDP 包装模型，设置混合精度和忽略模块
        model = FSDP(
            model,
            ignored_modules=[model.ignored],
            mixed_precision=float16,
        )

        x = torch.ones(2, 100, device=torch.cuda.current_device())

        with self.assertRaisesRegex(RuntimeError, "must have the same dtype"):
            # 断言运行模型时会抛出异常，要求张量具有相同的数据类型
            model(x).sum().backward()


class TestFSDPDifferentSubmodulePrecision(FSDPTest):
    @property
    def world_size(self):
        # 返回测试中的世界大小，此处为 2
        return 2

    @skip_if_lt_x_gpu(2)
    def test_float16_on_one_submodule(self):
        # 定义一个空的字典，用于存储前向输入模块
        forward_inputs: Dict[str, nn.Module] = {}
        # 创建一个混合精度对象，使用 float16 参数，并将前向输入转换为 float16
        float16 = MixedPrecision(param_dtype=torch.float16, cast_forward_inputs=True)

        # 创建 SaveForwardInputsModel 模型实例，将前向输入模块和不转换前向输入参数传入，并放置在 GPU 上
        model = SaveForwardInputsModel(
            forward_inputs,
            cast_forward_inputs=False,
        ).cuda()
        # 分别获取模型中的 c1 和 c2 子模块
        c1, c2 = model.c1, model.c2
        # 创建一个在 GPU 上的形状为 (2, 100) 的全零张量
        x = torch.zeros(2, 100, device="cuda")

        # 在一个子模块上使用 float16 精度，其余使用 float32 精度
        model.c2 = FSDP(model.c2, mixed_precision=float16)
        # 将整个模型包装在 FSDP 中
        fsdp = FSDP(model)

        # 对模型进行前向传播、求和、反向传播
        fsdp(x).sum().backward()

        # 断言前向输入模型、c1 和 c2 的数据类型
        self.assertEqual(forward_inputs[model].dtype, torch.float32)
        self.assertEqual(forward_inputs[c1].dtype, torch.float32)
        self.assertEqual(forward_inputs[c2].dtype, torch.float16)

    @skip_if_lt_x_gpu(2)
    def test_float16_on_one_submodule_skip_inputs(self):
        # 定义一个空的字典，用于存储前向输入模块和输入张量
        forward_inputs: Dict[nn.Module, torch.Tensor] = {}
        # 创建一个混合精度对象，使用 float16 参数，不转换前向输入参数
        float16 = MixedPrecision(param_dtype=torch.float16, cast_forward_inputs=False)

        # 创建 SaveForwardInputsModel 模型实例，将前向输入模块和转换前向输入参数传入，并放置在 GPU 上
        model = SaveForwardInputsModel(
            forward_inputs=forward_inputs, cast_forward_inputs=True
        ).cuda()
        # 分别获取模型中的 c1 和 c2 子模块
        c1, c2 = model.c1, model.c2
        # 创建一个在 GPU 上的形状为 (2, 100) 的全零张量
        x = torch.zeros(2, 100, device="cuda")

        # 在一个子模块上使用 float16 精度，其余使用 float32 精度
        model.c2 = FSDP(model.c2, mixed_precision=float16)
        # 将整个模型包装在 FSDP 中
        fsdp = FSDP(model)

        # 对模型进行前向传播、求和、反向传播
        fsdp(x).sum().backward()

        # 断言前向输入模型、c1 和 c2 的数据类型，期望都是 float32
        self.assertEqual(forward_inputs[model].dtype, torch.float32)
        self.assertEqual(forward_inputs[c1].dtype, torch.float32)
        self.assertEqual(forward_inputs[c2].dtype, torch.float32)

    @skip_if_lt_x_gpu(2)
    def test_float16_on_one_submodule_skip_inputs_error(self):
        # 定义一个空的字典，用于存储前向输入模块和输入张量
        forward_inputs: Dict[nn.Module, torch.Tensor] = {}
        # 创建一个混合精度对象，使用 float16 参数，不转换前向输入参数
        float16 = MixedPrecision(param_dtype=torch.float16, cast_forward_inputs=False)

        # 创建 SaveForwardInputsModel 模型实例，将前向输入模块和不转换前向输入参数传入，并放置在 GPU 上
        model = SaveForwardInputsModel(
            forward_inputs=forward_inputs, cast_forward_inputs=False
        ).cuda()
        # 分别获取模型中的 c1 和 c2 子模块
        c1, c2 = model.c1, model.c2
        # 创建一个在 GPU 上的形状为 (2, 100) 的全零张量
        x = torch.zeros(2, 100, device="cuda")

        # 在一个子模块上使用 float16 精度，其余使用 float32 精度
        model.c2 = FSDP(model.c2, mixed_precision=float16)
        # 将整个模型包装在 FSDP 中
        fsdp = FSDP(model)

        # 断言捕获到的异常类型和消息
        with self.assertRaisesRegex(
            RuntimeError, "mat1 and mat2 must have the same dtype"
        ):
            # 对模型进行前向传播、求和、反向传播
            fsdp(x).sum().backward()
    @skip_if_lt_x_gpu(2)
    def test_submodules_with_different_precisions_error(self):
        # 定义一个空的字典，用于存储模型和其对应的输入张量
        forward_inputs: Dict[nn.Module, torch.Tensor] = {}
        
        # 创建两个 MixedPrecision 对象，分别对应于 float16 和 float32 的精度
        float16 = MixedPrecision(param_dtype=torch.float16, cast_forward_inputs=True)
        float32 = MixedPrecision(param_dtype=torch.float32, cast_forward_inputs=True)

        # 在 CUDA 上初始化 SaveForwardInputsModel 模型
        model = SaveForwardInputsModel(
            forward_inputs=forward_inputs, cast_forward_inputs=False
        ).cuda()
        
        # 创建一个 CUDA 上的全零张量 x
        x = torch.zeros(2, 100, device="cuda")

        # 对于不同精度的子模块，当前设计不支持根 FSDP 实例包装一个不是第一个执行的子模块的情况。
        # 因为对于那个子模块，其输入（或前一个子模块的输出）无法被转换，相反，
        # 在进入根模块的前向传播之前，会提前转换根模块的输入。
        model.c1 = FSDP(model.c1, mixed_precision=float16)
        
        # 创建一个 FSDP 包装的模型，采用 float32 的混合精度
        fsdp = FSDP(model, mixed_precision=float32)
        
        # 断言异常，期望抛出 RuntimeError，错误信息为 "mat1 and mat2 must have the same dtype"
        with self.assertRaisesRegex(
            RuntimeError, "mat1 and mat2 must have the same dtype"
        ):
            # 执行前向传播、求和、反向传播操作
            fsdp(x).sum().backward()

    @skip_if_lt_x_gpu(2)
    def test_submodules_with_different_precisions(self):
        # 定义一个空的字典，用于存储模型和其对应的输入张量
        forward_inputs: Dict[nn.Module, torch.Tensor] = {}
        
        # 创建两个 MixedPrecision 对象，分别对应于 float16 和 float32 的精度
        float16 = MixedPrecision(param_dtype=torch.float16, cast_forward_inputs=True)
        float32 = MixedPrecision(param_dtype=torch.float32, cast_forward_inputs=True)

        # 在 CUDA 上初始化 SaveForwardInputsModel 模型
        model = SaveForwardInputsModel(
            forward_inputs=forward_inputs, cast_forward_inputs=False
        ).cuda()
        
        # 将模型的 c1 和 c2 子模块分别赋给 c1 和 c2 变量
        c1, c2 = model.c1, model.c2
        
        # 创建一个 CUDA 上的全零张量 x
        x = torch.zeros(2, 100, device="cuda")

        # 将模型的 c2 子模块用 float16 精度的 FSDP 包装
        model.c2 = FSDP(model.c2, mixed_precision=float16)
        
        # 创建一个 FSDP 包装的模型，采用 float32 的混合精度
        fsdp = FSDP(model, mixed_precision=float32)

        # 执行前向传播、求和、反向传播操作
        fsdp(x).sum().backward()

        # 断言 forward_inputs 中模型的 dtype 分别为 torch.float32 和 torch.float32，而 c2 的 dtype 为 torch.float16
        self.assertEqual(forward_inputs[model].dtype, torch.float32)
        self.assertEqual(forward_inputs[c1].dtype, torch.float32)
        self.assertEqual(forward_inputs[c2].dtype, torch.float16)
    def test_submodules_with_external_inputs(self):
        # 定义一个名为 test_submodules_with_external_inputs 的测试方法
        class ToyModule(nn.Module):
            # ToyModule 类，继承自 nn.Module
            def __init__(self, forward_inputs: Dict[str, torch.Tensor]) -> None:
                super().__init__()
                # 在 ToyModule 初始化中，创建一个线性层，输入和输出都是 100 维
                self.l = nn.Linear(100, 100)
                # 将外部传入的输入字典保存到实例属性中
                self.forward_inputs = forward_inputs

            def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                # 在 forward 方法中，将输入 x 和 y 分别存储到 forward_inputs 字典中的不同键
                self.forward_inputs["l2_input_x"] = x
                self.forward_inputs["l2_input_y"] = y
                # 返回线性层的输出结果
                return self.l(x)

        class ToyModel(nn.Module):
            # ToyModel 类，继承自 nn.Module
            def __init__(self, forward_inputs: Dict[str, torch.Tensor]) -> None:
                super().__init__()
                # 在 ToyModel 初始化中，创建两个线性层
                self.l1 = nn.Linear(100, 100)
                # 创建一个 ToyModule 实例，传入外部输入字典
                self.l2 = ToyModule(forward_inputs)
                # 将外部传入的输入字典保存到实例属性中
                self.forward_inputs = forward_inputs

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # 在 forward 方法中，将输入 x 存储到 forward_inputs 字典中的特定键
                self.forward_inputs["model_input_x"] = x
                # 创建一个 CUDA 设备上的全 1 张量 y，数据类型为 torch.float32
                y = torch.ones(2, 100, device="cuda", dtype=torch.float32)
                # 调用 ToyModule 实例的 forward 方法，并传入 l1(x) 和 y 作为参数
                return self.l2(self.l1(x), y)

        forward_inputs: Dict[str, torch.Tensor] = {}

        # 创建一个 MixedPrecision 实例，设置参数数据类型为 torch.float16
        float16 = MixedPrecision(param_dtype=torch.float16)
        # 创建一个 ToyModel 实例，并将其移动到 CUDA 设备上
        model = ToyModel(forward_inputs).cuda()
        # 创建一个形状为 (2, 100) 的全零张量 x，数据类型为 torch.float32，放在 CUDA 设备上
        x = torch.zeros(2, 100, device="cuda", dtype=torch.float32)
        # 将 ToyModel 实例中的 l2 属性替换为 FSDP 包装后的模型，使用混合精度 float16
        model.l2 = FSDP(model.l2, mixed_precision=float16)
        # 对整个模型 model 应用 FSDP，使用混合精度 float16
        fsdp = FSDP(model, mixed_precision=float16)

        # 对模型进行前向传播，并对输出结果求和，然后进行反向传播
        fsdp(x).sum().backward()

        # 断言验证模型中各输入的数据类型是否符合预期
        # 在默认情况下，根模块中的输入会被转换为 float16 类型，但子模块中的外部输入 y 不会被显式转换
        self.assertEqual(forward_inputs["model_input_x"].dtype, torch.float16)
        self.assertEqual(forward_inputs["l2_input_x"].dtype, torch.float16)
        self.assertEqual(forward_inputs["l2_input_y"].dtype, torch.float32)
# 定义一个名为 TestFSDPTrainEval 的测试类，继承自 FSDPTest 类
class TestFSDPTrainEval(FSDPTest):

    # 定义一个名为 world_size 的属性方法，返回值为 2
    @property
    def world_size(self):
        return 2

    # 定义一个名为 test_train_ema_eval_flow 的测试方法，使用 skip_if_lt_x_gpu 装饰器，
    # 要求至少有 2 个 GPU 才能执行该测试
    def test_train_ema_eval_flow(self):
        """
        Tests a train -> EMA update -> eval flow with mixed precision enabled.
        测试训练 -> 更新EMA -> 评估流程，并启用混合精度。
        """
        # 运行子测试，传入一个字典和一个测试方法
        self.run_subtests(
            {
                "sharding_strategy": [
                    # 我们主要测试 `SHARD_GRAD_OP`，因为它暴露了
                    # 一个原始的 bug，即在评估时没有使用正确的EMA参数，
                    # 但为了完整性，我们也测试其他策略
                    ShardingStrategy.SHARD_GRAD_OP,
                    ShardingStrategy.FULL_SHARD,
                    ShardingStrategy.NO_SHARD,
                ]
            },
            self._test_train_ema_eval_flow,  # 测试方法参数
        )
    def _test_train_ema_eval_flow(self, sharding_strategy: ShardingStrategy):
        # 定义一个带有指数移动平均 (EMA) 的 Transformer 模型类
        class TransformerWithEMA(nn.Module):
            def __init__(self, device: torch.device):
                super().__init__()
                # 创建主要使用的 Transformer 模型
                self.module = nn.Transformer(device=device)
                # 创建用于 EMA 的 Transformer 模型，使用平均函数和缓冲区
                self.ema_module = AveragedModel(
                    nn.Transformer(device=device),
                    multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(),
                    use_buffers=True,
                )

            def forward(self, *args, **kwargs):
                # 如果处于训练状态，则返回主要模型的前向传播结果
                if self.training:
                    return self.module(*args, **kwargs)
                # 否则返回 EMA 模型的前向传播结果
                return self.ema_module(*args, **kwargs)

        device = torch.device("cuda")
        # 创建使用 EMA 的 TransformerWithEMA 模型实例
        model = TransformerWithEMA(device=device)
        # 定义模型包装策略，适用于 Transformer 和其编码器、解码器层
        policy = ModuleWrapPolicy(
            {nn.Transformer, nn.TransformerEncoderLayer, nn.TransformerDecoderLayer}
        )
        # 定义混合精度设置，使用浮点16位参数
        mixed_precision = MixedPrecision(param_dtype=torch.float16)
        # 使用 FSDP 进行模型并行训练，配置包括自动包装策略、混合精度和分片策略
        fsdp_model = FSDP(
            model,
            auto_wrap_policy=policy,
            mixed_precision=mixed_precision,
            sharding_strategy=sharding_strategy,
        )
        # 使用 Adam 优化器对主要模型的参数进行优化
        optim = torch.optim.Adam(fsdp_model.module.parameters(), lr=1e-2)
        # 如果当前进程的排名为0，打印模型信息
        if self.rank == 0:
            print(fsdp_model)
        # 设置随机种子，以确保结果的可重复性
        torch.manual_seed(1 + self.rank)
        # 创建用于评估的随机输入数据
        eval_src = torch.randn((8, 1, 512), device=device)
        eval_tgt = torch.randn((16, 1, 512), device=device)
        # 存储评估输出的和
        eval_out_sums: List[torch.Tensor] = []
        # 进行多次迭代，每次迭代包括训练、更新 EMA 模型、以及评估
        for _ in range(3):
            fsdp_model.train()
            # 创建用于训练的随机输入数据
            train_src = torch.randn((8, 4, 512), device=device)
            train_tgt = torch.randn((16, 4, 512), device=device)
            # 对模型进行训练，计算输出并进行反向传播
            train_out = fsdp_model(train_src, train_tgt)
            train_out.sum().backward()
            # 执行优化步骤，清除梯度
            optim.step()
            optim.zero_grad()
            # 通过全参数召唤，更新 EMA 模型的参数
            with FSDP.summon_full_params(fsdp_model):
                fsdp_model.ema_module.update_parameters(fsdp_model.module)
            # 切换到评估模式，使用无梯度的上下文评估模型
            fsdp_model.eval()
            with torch.no_grad():
                eval_out = fsdp_model(eval_src, eval_tgt)
            # 记录评估输出的和
            eval_out_sums.append(eval_out.sum())
        # 检查评估输出是否在迭代中有所不同，以确保使用正确的 EMA 参数进行评估
        for i in range(len(eval_out_sums) - 1):
            self.assertNotEqual(eval_out_sums[i], eval_out_sums[i + 1])
        self.assertNotEqual(eval_out_sums[0], eval_out_sums[-1])
# 如果这个脚本是作为主程序执行（而不是被导入到其他程序中），则运行测试函数
if __name__ == "__main__":
    run_tests()
```