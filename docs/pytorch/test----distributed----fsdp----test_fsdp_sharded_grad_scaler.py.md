# `.\pytorch\test\distributed\fsdp\test_fsdp_sharded_grad_scaler.py`

```py
# Owner(s): ["oncall: distributed"]

# 导入所需的模块和类
import copy
import functools
import itertools
import sys
import unittest
from typing import List, Optional

import torch
from torch import distributed as dist
from torch.cuda.amp.common import amp_definitely_not_available
from torch.distributed.fsdp import CPUOffload, MixedPrecision
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
)
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from torch.nn import TransformerDecoderLayer, TransformerEncoderLayer
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import (
    CUDAInitMode,
    DummyProcessGroup,
    FSDPInitMode,
    FSDPTest,
    NestedWrappedModule,
    NonUniformReqGradNWM,
    subtest_name,
    TransformerWithSharedParams,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TEST_WITH_DEV_DBG_ASAN,
    TestCase,
)

# 检查是否支持分布式训练，若不支持则跳过测试
if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)

# 检查是否开启了开发调试模式，如果是，则跳过测试
if TEST_WITH_DEV_DBG_ASAN:
    print(
        "Skip dev-asan as torch + multiprocessing spawn have known issues",
        file=sys.stderr,
    )
    sys.exit(0)

# 定义用于参数化测试的参数
params = "cpu_offload,sharding_strategy,mixed_precision,use_orig_params"
cpu_offload_config = [CPUOffload(offload_params=True), CPUOffload(offload_params=False)]
sharding_strategy_config = [ShardingStrategy.SHARD_GRAD_OP, None]
mixed_precision = ["enable_mixed_precision", None]
use_orig_params = ["enable_use_orig_params", None]

# 生成参数组合
configs = list(
    itertools.product(
        cpu_offload_config, sharding_strategy_config, mixed_precision, use_orig_params
    )
)

# 定义测试名称映射表，用于显示测试结果时的标签
test_name_mapping = {
    str(CPUOffload(offload_params=True)): "offload_true",
    str(CPUOffload(offload_params=False)): "offload_false",
    str(ShardingStrategy.SHARD_GRAD_OP): "shard_grad_op",
    "enable_mixed_precision": "mixed_precision",
    "enable_use_orig_params": "use_orig_params",
}

# 为子测试定义一个局部函数，用于添加标签
subtest_name = functools.partial(subtest_name, test_name_mapping)

# 定义测试类 TestShardGradScaler
class TestShardGradScaler(TestCase):
    # 如果不支持 amp，跳过测试
    @unittest.skipIf(
        amp_definitely_not_available(), "no supported device (cuda, xla) found"
    )
    # 定义用于测试梯度缩放的测试方法
    def test_grad_scaling(self):
        # 创建虚拟的进程组对象
        pg = DummyProcessGroup(0, 1)
        # 初始化梯度缩放器对象
        scaler = ShardedGradScaler(init_scale=2.0, process_group=pg, enabled=True)
        # 创建包含单一元素的张量 t0 和 t1，值分别为 4.0 和 8.0
        t0 = torch.full((1,), 4.0, dtype=torch.float32, device="cpu")
        t1 = torch.full((1,), 8.0, dtype=torch.float32, device="cpu")
        # 创建包含复制的张量 outputs，包括 t1 的克隆，元组 (t0 的克隆, t1 的克隆)，列表 [t0 的克隆, t1 的克隆]
        outputs = [t1.clone(), (t0.clone(), t1.clone()), [t0.clone(), t1.clone()]]
        # 对 outputs 进行梯度缩放
        outputs = scaler.scale(outputs)
        # 断言 outputs 中的值符合预期
        self.assertTrue(
            outputs[0] == 16.0 and outputs[1][0] == 8.0 and outputs[1][1] == 16.0
        )
        self.assertTrue(outputs[2][0] == 8.0 and outputs[2][1] == 16.0)
        # 断言 scaler 的缩放设备与 t1 的设备相同
        self.assertTrue(scaler._scale.device == t1.device)

    # 如果 amp_definitely_not_available() 返回 True，则跳过该测试用例
    @unittest.skipIf(
        amp_definitely_not_available(), "no supported device (cuda, xla) found"
    )
    # 定义测试稀疏张量的缩放和反缩放的方法
    def test_scaling_unscaling_sparse(self):
        # 创建虚拟的进程组对象
        pg = DummyProcessGroup(0, 1)
        # 初始化梯度缩放器对象
        scaler = ShardedGradScaler(init_scale=2.0, process_group=pg, enabled=True)
        # 创建用于反缩放的逆缩放比例张量和发现无穷大张量
        inv_scale = torch.full((1,), 0.5, dtype=torch.float, device="cpu")
        found_inf = torch.full((1,), 0, dtype=torch.float, device="cpu")

        # 创建稀疏张量 s，包含非零元素的索引 i 和值 v
        i = torch.tensor([[0, 1, 1], [2, 0, 2]], device="cpu", dtype=torch.int64)
        v = torch.tensor([16.0, 32.0, 64.0], dtype=torch.float, device="cpu")
        s = torch.sparse_coo_tensor(
            i, v, torch.Size([2, 3]), device="cpu", dtype=torch.float
        )

        # 对稀疏张量 s 进行反缩放
        s1 = s.clone()
        s1.grad = s.clone()
        opt = torch.optim.SGD([s1], lr=1.0)
        found_inf.zero_()
        found_inf = scaler._unscale_grads_(opt, inv_scale, found_inf)[s1.device]
        # 断言 found_inf 的值为 0.0
        self.assertEqual(found_inf, 0.0)
        # 断言 s1 的梯度密集表示与 s 的一半相等
        self.assertEqual(s1.grad.to_dense(), (s / 2).to_dense())

        # 创建包含无穷大元素的稀疏张量 s1，对其进行反缩放
        v = torch.tensor([16.0, 32.0, float("inf")], dtype=torch.float, device="cpu")
        s1.grad = torch.sparse_coo_tensor(
            i, v, torch.Size([2, 3]), device="cpu", dtype=torch.float
        )
        found_inf.zero_()
        found_inf = scaler._unscale_grads_(opt, inv_scale, found_inf)[s1.device]
        # 断言 found_inf 的值为 1.0
        self.assertEqual(found_inf, 1.0)

        # 创建包含可能溢出的稀疏张量 s1，对其进行反缩放
        i = torch.tensor([[1, 1, 1], [0, 0, 2]], device="cpu", dtype=torch.int64)
        v = torch.tensor([2**15, 2**15, 1.0], dtype=torch.float16, device="cpu")
        s1 = torch.sparse_coo_tensor(
            i, v, torch.Size([2, 3]), device="cpu", dtype=torch.float16
        )
        s1.grad = s1.clone()
        found_inf.zero_()
        found_inf = scaler._unscale_grads_(opt, inv_scale, found_inf)[s1.device]
        # 断言 found_inf 的值为 1.0
        self.assertEqual(found_inf, 1.0)
    # 定义一个测试方法，用于验证在梯度计算中跳过优化步骤的情况
    def test_inf_gradients_skip_optim_step(self):
        # 创建一个虚拟的进程组对象，仅用于测试目的
        pg = DummyProcessGroup(0, 1)
        # 创建一个分片梯度缩放器对象，设置初始缩放比例为2.0，指定进程组为pg，启用状态为True
        scaler = ShardedGradScaler(init_scale=2.0, process_group=pg, enabled=True)
        # 创建一个包含单个元素的张量，值为4.0，数据类型为float32，在CPU上
        loss = torch.full((1,), 4.0, dtype=torch.float32, device="cpu")
        # 创建一个包含正无穷大的张量t0，数据类型为float32，在CPU上
        t0 = torch.tensor([float("inf")], dtype=torch.float32, device="cpu")
        # 克隆t0并将其设为梯度值
        t0.grad = t0.clone()
        # 使用随机梯度下降算法创建一个优化器opt，优化目标为张量t0，学习率为1.0
        opt = torch.optim.SGD([t0], lr=1.0)
        # 使用缩放器对损失值进行缩放
        scaler.scale(loss)
        # 执行优化步骤，返回值为None，因为在此处梯度值为正无穷大，跳过了优化步骤
        ret_val = scaler.step(opt)
        # 断言返回值为None，验证跳过优化步骤的行为
        self.assertTrue(ret_val is None)
class TestShardedGradScalerParityWithDDP(FSDPTest):
    # 继承自 FSDPTest 的测试类，用于测试分片梯度缩放器和分布式数据并行（DDP）的一致性

    def _get_init_modes_for_test(self, cpu_offload):
        # 返回用于测试的初始化模式列表，根据 CPU 是否卸载参数决定是否包含 CUDAInitMode.CUDA_NEVER
        modes = [CUDAInitMode.CUDA_AFTER, CUDAInitMode.CUDA_BEFORE]
        if cpu_offload.offload_params:
            modes.append(CUDAInitMode.CUDA_NEVER)
        return modes

    @skip_if_lt_x_gpu(2)
    @parametrize(params, configs, subtest_name)
    def test_fsdp_ddp_parity_with_grad_scaler(
        self,
        cpu_offload: CPUOffload,
        sharding_strategy: Optional[ShardingStrategy],
        mixed_precision: Optional[str],
        use_orig_params: Optional[str],
    ):
        # 测试函数，验证 FSDP 和 DDP 之间的一致性，使用分片梯度缩放器

        # 获取初始化模式列表
        init_modes = self._get_init_modes_for_test(cpu_offload)

        # 如果需要使用混合精度，则创建 MixedPrecision 对象
        mp = (
            MixedPrecision(
                param_dtype=torch.float16,
                reduce_dtype=torch.float16,
                buffer_dtype=torch.float16,
            )
            if mixed_precision is not None
            else None
        )

        # 根据 use_orig_params 参数选择模型类和分片梯度缩放器的参数
        if use_orig_params == "enable_use_orig_params":
            use_orig = True
            model_cls = NonUniformReqGradNWM  # 使用特定的模型类 NonUniformReqGradNWM
            sharded_grad_scaler_kwargs = {"init_scale": 2.0**11}  # 设置特定的分片梯度缩放器参数
        else:
            use_orig = False
            model_cls = NestedWrappedModule  # 使用默认的模型类 NestedWrappedModule
            sharded_grad_scaler_kwargs = None

        # 遍历所有的初始化模式进行测试
        for cuda_init_mode in init_modes:
            self._test_fsdp_parity(
                model_cls,
                FSDPInitMode.RECURSIVE,
                cuda_init_mode=cuda_init_mode,
                cpu_offload=cpu_offload,
                sharding_strategy=sharding_strategy,
                mixed_precision=mp,
                enable_sharded_grad_scaler=True,
                use_orig_params=use_orig,
                sharded_grad_scaler_kwargs=sharded_grad_scaler_kwargs,
            )

    def _build_model_and_optim(
        self,
        cpu_offload: CPUOffload = CPUOffload(offload_params=False),
        use_orig_params: bool = False,
    ):
        # 使用 TransformerWithSharedParams 类初始化模型，指定参数初始化模式和CUDA初始化模式，并设为确定性运行
        model = TransformerWithSharedParams.init(
            self.process_group,
            FSDPInitMode.NO_FSDP,
            CUDAInitMode.CUDA_BEFORE,
            deterministic=True,
        )
        # 使用深拷贝创建模型的分布式数据并行 (DDP) 版本，仅在当前设备上运行
        ref_model = DDP(
            copy.deepcopy(model),
            device_ids=[self.rank],
        )
        # 为参考模型创建 Adam 优化器
        ref_optim = torch.optim.Adam(ref_model.parameters(), lr=1e-2)
        # 配置 FSDP 所需的参数
        fsdp_kwargs = {
            "use_orig_params": use_orig_params,  # 是否使用原始参数
            "cpu_offload": cpu_offload,  # CPU 上的参数卸载设置
            "auto_wrap_policy": ModuleWrapPolicy(  # 自动封装策略
                {
                    TransformerEncoderLayer,  # Transformer 编码器层
                    TransformerDecoderLayer,  # Transformer 解码器层
                }
            ),
        }
        # 使用 FSDP 对模型进行包装，传入模型和配置参数
        model = FSDP(model, **fsdp_kwargs)
        # 创建模型的 Adam 优化器
        optim = torch.optim.Adam(model.parameters(), lr=1e-2)
        # 返回模型、优化器、参考模型和参考模型的优化器
        return model, optim, ref_model, ref_optim

    @skip_if_lt_x_gpu(2)
    def test_sharded_grad_scaler_found_inf(self):
        # 运行子测试，测试不同参数组合下的 _test_sharded_grad_scaler_found_inf 方法
        self.run_subtests(
            {
                "use_orig_params": [False, True],  # 使用原始参数的两种选择
                "cpu_offload": [
                    CPUOffload(offload_params=True),  # CPU 参数卸载为真
                    CPUOffload(offload_params=False),  # CPU 参数卸载为假
                ],
            },
            self._test_sharded_grad_scaler_found_inf,  # 待测试的方法
        )

    def _test_sharded_grad_scaler_found_inf(
        self,
        use_orig_params: bool,  # 是否使用原始参数
        cpu_offload: CPUOffload,  # CPU 参数卸载设置
# 调用函数 instantiate_parametrized_tests，参数为 TestShardGradScaler 类，实例化参数化测试
instantiate_parametrized_tests(TestShardGradScaler)

# 调用函数 instantiate_parametrized_tests，参数为 TestShardedGradScalerParityWithDDP 类，实例化参数化测试
instantiate_parametrized_tests(TestShardedGradScalerParityWithDDP)

# 如果当前脚本作为主程序执行，则调用函数 run_tests() 运行测试
if __name__ == "__main__":
    run_tests()
```