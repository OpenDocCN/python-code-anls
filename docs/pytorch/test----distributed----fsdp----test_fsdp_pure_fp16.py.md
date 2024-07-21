# `.\pytorch\test\distributed\fsdp\test_fsdp_pure_fp16.py`

```
# Owner(s): ["oncall: distributed"]

# 导入系统模块
import sys

# 导入PyTorch相关模块
import torch
import torch.distributed.fsdp._traversal_utils as traversal_utils
from torch import distributed as dist
from torch.distributed.fsdp import (
    CPUOffload,
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
)
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import (
    CUDAInitMode,
    FSDPInitMode,
    FSDPTest,
    NestedWrappedModule,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    run_tests,
    TEST_WITH_DEV_DBG_ASAN,
)

# 如果分布式环境不可用，则跳过测试
if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)

# 如果启用了开发调试ASAN，则跳过测试
if TEST_WITH_DEV_DBG_ASAN:
    print(
        "Skip dev-asan as torch + multiprocessing spawn have known issues",
        file=sys.stderr,
    )
    sys.exit(0)


class TestPureFP16(FSDPTest):
    @property
    def world_size(self):
        # 限制世界大小，因为使用超过4个GPU会导致测试失败
        return min(4, super().world_size)

    @skip_if_lt_x_gpu(2)
    def test_pure_fp16_training(self):
        """Tests pure FP16 training, including when the parameter's dtype is
        changed after FSDP initialization and before training."""
        # 运行子测试，测试纯FP16训练
        self.run_subtests(
            {
                "cpu_offload": [
                    CPUOffload(offload_params=True),
                    CPUOffload(offload_params=False),
                ]
            },
            self._test_pure_fp16_training,
        )

    def _test_pure_fp16_training(self, cpu_offload: CPUOffload):
        # 测试纯FP16训练的函数，验证FSDP初始化后和训练前参数dtype的变化
        self._test_fsdp_parity(
            NestedWrappedModule,
            FSDPInitMode.RECURSIVE,
            cuda_init_mode=CUDAInitMode.CUDA_BEFORE,
            # 只运行一次迭代，避免梯度缩放时出现NaN
            num_iters=1,
            cpu_offload=cpu_offload,
            use_pure_fp16=True,
        )

    @skip_if_lt_x_gpu(2)
    def test_fp16_dtypes(self):
        """
        Tests that both user-facing parameter/gradient dtypes and internal
        saved dtype attributes are as expected when using an FP16 model
        possibly with explicit mixed precision enabled.
        """
        # 运行子测试，验证在使用FP16模型时，用户参数/梯度的dtype和内部保存的dtype属性是否符合预期
        self.run_subtests(
            {
                "to_half_before_fsdp_init": [False, True],
                "use_orig_params": [False, True],
                "mixed_precision": [
                    MixedPrecision(),
                    MixedPrecision(
                        param_dtype=torch.float16,
                        reduce_dtype=torch.float32,
                    ),
                    MixedPrecision(
                        param_dtype=torch.float32,
                    ),
                ],
            },
            self._test_fp16_dtypes,
        )

    def _test_fp16_dtypes(
        self,
        to_half_before_fsdp_init: bool,
        use_orig_params: bool,
        mixed_precision: MixedPrecision,
        ):
            # 使用 NestedWrappedModule 类的静态方法初始化模型，传入参数包括进程组、FSDP 初始化模式、CUDA 初始化模式和空字典作为额外参数
            model = NestedWrappedModule.init(
                self.process_group,
                FSDPInitMode.NO_FSDP,
                CUDAInitMode.CUDA_NEVER,
                {},
            )
            # 准备 FSDP 的关键字参数字典
            fsdp_kwargs = {
                "use_orig_params": use_orig_params,  # 是否使用原始参数
                "device_id": torch.cuda.current_device(),  # 当前 CUDA 设备 ID
                "mixed_precision": mixed_precision,  # 是否使用混合精度
            }
            # 如果在 FSDP 初始化之前需要将模型转换为半精度
            if to_half_before_fsdp_init:
                model = model.half()
            # 创建 FSDP 模型对象
            fsdp_model = FSDP(model, **fsdp_kwargs)
            # 如果不在 FSDP 初始化之前将模型转换为半精度
            if not to_half_before_fsdp_init:
                fsdp_model = fsdp_model.half()
            # 遍历 FSDP 模型的参数，确保其数据类型为 torch.float16
            for param in fsdp_model.parameters():
                self.assertEqual(param.dtype, torch.float16)
            # 准备输入数据
            inp = tuple(
                t.half() if torch.is_tensor(t) else t
                for t in fsdp_model.module.get_input(torch.device("cuda"))
            )
            # 使用 FSDP 模型进行前向传播
            out = fsdp_model(*inp)
            # 计算输出的和，并进行反向传播
            out.sum().backward()

            # 检查 FSDP 模型中各个 handle 的数据类型属性
            for handle in traversal_utils._get_fsdp_handles(fsdp_model):
                self.assertEqual(handle.flat_param.dtype, torch.float16)
                self.assertEqual(handle.flat_param.grad.dtype, torch.float16)
                self.assertEqual(handle._orig_param_dtype, torch.float16)
                # 指定 `mixed_precision` 优先于模型的 dtype，适用于 `param_dtype` 和 `reduce_dtype`
                if mixed_precision.param_dtype is not None:
                    self.assertEqual(
                        handle._fwd_bwd_param_dtype, mixed_precision.param_dtype
                    )
                else:
                    self.assertEqual(handle._fwd_bwd_param_dtype, torch.float16)
                if mixed_precision.reduce_dtype is not None:
                    self.assertEqual(handle._reduce_dtype, mixed_precision.reduce_dtype)
                elif (
                    mixed_precision.reduce_dtype is None
                    and mixed_precision.param_dtype is not None
                ):
                    # 特殊情况：从参数 dtype 推断 reduce dtype
                    self.assertEqual(handle._reduce_dtype, mixed_precision.param_dtype)
                else:
                    self.assertEqual(handle._reduce_dtype, torch.float16)

            # 再次检查 FSDP 模型中参数和梯度的数据类型
            for param in fsdp_model.parameters():
                self.assertEqual(param.dtype, torch.float16)
                if param.grad is not None:
                    self.assertEqual(param.grad.dtype, torch.float16)
# 实例化参数化测试对象 TestPureFP16
instantiate_parametrized_tests(TestPureFP16)

# 如果当前脚本作为主程序运行，则执行测试
if __name__ == "__main__":
    # 运行测试函数
    run_tests()
```