# `.\pytorch\test\distributed\fsdp\test_fsdp_checkpoint.py`

```
# Owner(s): ["oncall: distributed"]

# 导入必要的库和模块
import contextlib               # 上下文管理模块，用于创建上下文管理器
import sys                      # 系统模块，提供对 Python 解释器的访问
from copy import deepcopy       # 导入深拷贝函数 deepcopy
from functools import partial   # 导入偏函数模块中的 partial 函数

import torch                    # PyTorch 深度学习框架
import torch.distributed as dist  # PyTorch 分布式模块的别名 dist
import torch.nn as nn           # PyTorch 神经网络模块的别名 nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,         # 导入分布式算法的检查点包装器
    offload_wrapper,            # 导入分布式算法的卸载包装器
)
from torch.distributed.fsdp import ShardingStrategy  # 导入 FSDP 模块的分片策略
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    CPUOffload,                 # 导入 CPU 卸载类
    FullyShardedDataParallel as FSDP,  # 导入全分片数据并行类，并使用 FSDP 别名
)

from torch.testing._internal.common_distributed import skip_if_lt_x_gpu  # 导入分布式测试工具中的 GPU 数量检测函数
from torch.testing._internal.common_fsdp import _maybe_wrap_fsdp, FSDPTest  # 导入 FSDP 的测试工具函数和测试基类
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,  # 导入实例化参数化测试函数
    parametrize,              # 导入参数化装饰器函数
    run_tests,                # 导入运行测试函数
    TEST_WITH_DEV_DBG_ASAN,  # 导入是否开启 Dev Debug ASAN 测试的标志
)
from torch.utils.checkpoint import checkpoint  # 导入 PyTorch 的检查点函数

# 如果分布式不可用，则打印警告信息并退出
if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)

# 如果开启了 Dev Debug ASAN 测试，则打印警告信息并退出
if TEST_WITH_DEV_DBG_ASAN:
    print(
        "Skip dev-asan as torch + multiprocessing spawn have known issues",
        file=sys.stderr,
    )
    sys.exit(0)

# 全局变量，用于标记是否调用了 save_on_cpu 函数
_save_on_cpu_called = False


# 获取被修补后的 save_on_cpu 函数的版本
def get_patched_save_on_cpu():
    orig_save_on_cpu = (
        torch.distributed.algorithms._checkpoint.checkpoint_wrapper.save_on_cpu
    )

    def patched_save_on_cpu(*args, **kwargs):
        global _save_on_cpu_called
        _save_on_cpu_called = True
        return orig_save_on_cpu(*args, **kwargs)

    return patched_save_on_cpu


# 上下文管理器，用于临时替换 save_on_cpu 函数
@contextlib.contextmanager
def patch_save_on_cpu(new_save_on_cpu):
    orig_save_on_cpu = (
        torch.distributed.algorithms._checkpoint.checkpoint_wrapper.save_on_cpu
    )
    torch.distributed.algorithms._checkpoint.checkpoint_wrapper.save_on_cpu = (
        new_save_on_cpu
    )
    try:
        yield
    finally:
        torch.distributed.algorithms._checkpoint.checkpoint_wrapper.save_on_cpu = (
            orig_save_on_cpu
        )


# FSDPCheckpoint 测试类，继承自 FSDPTest
class TestFSDPCheckpoint(FSDPTest):
    # 定义一个继承自 nn.Module 的 SequentialModule 类
    class SequentialModule(nn.Module):
        # 初始化函数，接受一些参数用于配置模块
        def __init__(
            self,
            checkpoint_layer=False,  # 是否使用检查点层
            offload_activations=False,  # 是否卸载激活函数
            wrap_fsdp=False,  # 是否包装 FSDP
            *fsdp_args,  # FSDP 参数
            **fsdp_kwargs,  # FSDP 关键字参数
        ):
            # 设置随机种子
            torch.manual_seed(0)
            torch.cuda.manual_seed(0)
            # 调用父类的初始化函数
            super().__init__()
            # 创建三个线性层对象，并将其放到 GPU 上
            l1 = nn.Linear(3, 3).cuda()
            l2 = nn.Linear(3, 3).cuda()
            l3 = nn.Linear(3, 3).cuda()

            # 如果需要使用检查点层
            if checkpoint_layer:
                # 根据是否卸载激活函数选择合适的包装器
                if offload_activations:
                    ckpt_wrapper = offload_wrapper
                else:
                    ckpt_wrapper = checkpoint_wrapper

                # 对每个线性层应用检查点包装器
                l1 = ckpt_wrapper(l1)
                l2 = ckpt_wrapper(l2)
                l3 = ckpt_wrapper(l3)

            # 创建一个部分函数，用于包装 FSDP
            fsdp_wrapper = partial(
                _maybe_wrap_fsdp, *fsdp_args, wrap_fsdp=wrap_fsdp, **fsdp_kwargs
            )
            # 创建一个序列模块，包含三个线性层
            self.ffn = nn.Sequential(
                fsdp_wrapper(l1),
                fsdp_wrapper(l2),
                fsdp_wrapper(l3),
            )

        # 前向传播函数
        def forward(self, x):
            return self.ffn(x)

    # 验证损失、输出和模型之间的一致性
    def _verify_parity(self, losses, outputs, models):
        # 断言损失、输出和模型都存在
        assert losses
        assert outputs
        assert models

        # 遍历损失和输出，验证其与第一个元素的一致性
        for l, o in zip(losses[1:], outputs[1:]):
            self.assertEqual(losses[0], l)
            self.assertEqual(outputs[0], o)

        # 验证梯度
        ref_model = models[0]
        ref_grads = [p.grad for p in ref_model.parameters()]
        for m in models[1:]:
            grads = [p.grad for p in m.parameters()]
            for ref_g, g in zip(ref_grads, grads):
                self.assertEqual(ref_g, g)

    # 测试检查点和 FSDP 包装
    @skip_if_lt_x_gpu(2)
    @parametrize(
        "cpu_offload",
        [CPUOffload(offload_params=True), CPUOffload(offload_params=False)],
    )
    @parametrize("offload_activations", [True, False])
    @parametrize("use_orig_params", [False, True])
    def test_checkpoint_fsdp_wrapping(
        self,
        cpu_offload: CPUOffload,
        offload_activations: bool,
        use_orig_params: bool,
        # Test checkpoint(FSDP(layer1), FSDP(layer2), ....)
        # 如果启用激活值离线处理，选择相应的包装器；否则使用检查点包装器
        if offload_activations:
            wrapper_to_use = offload_wrapper
        else:
            wrapper_to_use = checkpoint_wrapper

        # 准备 FSDP 需要的参数字典
        fsdp_kwargs = {"cpu_offload": cpu_offload, "use_orig_params": use_orig_params}

        # 使用选定的包装器包装顺序模块，生成带有 FSDP 的检查点顺序模块
        ckpt_sequential_wrapped_fsdp = wrapper_to_use(
            TestFSDPCheckpoint.SequentialModule(
                wrap_fsdp=True,
                **fsdp_kwargs,
            ),
        )

        # 创建内部检查点顺序模块，每个层都使用检查点
        inner_ckpt = TestFSDPCheckpoint.SequentialModule(
            checkpoint_layer=True,
            offload_activations=offload_activations,
            wrap_fsdp=True,
            **fsdp_kwargs,
        )

        # 创建基线顺序模块，未使用检查点和离线处理
        baseline = TestFSDPCheckpoint.SequentialModule(
            wrap_fsdp=True,
            **fsdp_kwargs,
        )

        # 注意，基于重入的检查点需要设置输入的梯度标志
        inp = torch.randn(10, 3, device=torch.cuda.current_device(), requires_grad=True)

        # 初始化保存在 CPU 上的调用计数器
        global _save_on_cpu_called

        # 将所有模型放入列表中，以便迭代测试
        models = [ckpt_sequential_wrapped_fsdp, inner_ckpt, baseline]

        # 使用修补的保存在 CPU 上的函数进行上下文管理
        with patch_save_on_cpu(get_patched_save_on_cpu()):
            # 迭代两次以测试不同的条件
            for i in range(2):
                losses = []
                outputs = []
                for m in models:
                    # 检查是否需要离线处理，并且不是基线模型以及是第一次迭代
                    check_offload = m != baseline and i == 0 and offload_activations
                    if check_offload:
                        # 在进行离线处理时，应当保证没有保存在 CPU 上的调用
                        self.assertFalse(_save_on_cpu_called)
                    out = m(inp)
                    if check_offload:
                        # 在离线处理之后应当有保存在 CPU 上的调用
                        self.assertTrue(_save_on_cpu_called)
                        _save_on_cpu_called = False  # 重置调用状态
                    loss = out.sum()
                    loss.backward()
                    losses.append(loss)
                    outputs.append(out)

                # 验证所有模型的输出结果和损失
                self._verify_parity(losses, outputs, models)

        # 等待所有进程同步
        dist.barrier()
        ):
            fsdp_kwargs = {"cpu_offload": cpu_offload, "use_orig_params": use_orig_params}
            global _save_on_cpu_called
            with patch_save_on_cpu(get_patched_save_on_cpu()):
                seq = TestFSDPCheckpoint.SequentialModule().to(torch.cuda.current_device())
                # Runs FSDP with no checkpointing
                fsdp_only_seq = FSDP(deepcopy(seq), **fsdp_kwargs)
                # Runs checkpoint-wrapped FSDP
                if offload_activations:
                    wrapper_to_use = offload_wrapper
                else:
                    wrapper_to_use = checkpoint_wrapper

                checkpointed_fsdp = wrapper_to_use(
                    FSDP(deepcopy(seq), **fsdp_kwargs),
                )
                # Runs FSDP-wrapped checkpointed module
                fsdp_wrapped_checkpoint = FSDP(
                    wrapper_to_use(deepcopy(seq)),
                    **fsdp_kwargs,
                )
                # Runs FSDP with manual calls to checkpoint.
                fsdp_call_checkpoint = FSDP(deepcopy(seq), **fsdp_kwargs)
                # note that reentrant-based checkpointing requires inputs to have grad
                # flag set.

                inp = torch.randn(
                    10, 3, device=torch.cuda.current_device(), requires_grad=True
                )

                models = [
                    fsdp_only_seq,
                    checkpointed_fsdp,
                    fsdp_wrapped_checkpoint,
                    fsdp_call_checkpoint,
                ]
                # Ensure _save_on_cpu is not yet called
                self.assertFalse(_save_on_cpu_called)
                for i in range(6):
                    losses = []
                    outputs = []
                    for m in models:
                        check_offload = (
                            m != fsdp_only_seq and i == 0 and offload_activations
                        )
                        if m == fsdp_call_checkpoint:
                            # _save_on_cpu should not be called yet
                            self.assertFalse(_save_on_cpu_called)
                            offload_ctx = (
                                get_patched_save_on_cpu()(pin_memory=True)
                                if offload_activations
                                else contextlib.nullcontext()
                            )
                            with offload_ctx:
                                out = checkpoint(m, inp, use_reentrant=True)
                        else:
                            # _save_on_cpu should not be called yet
                            self.assertFalse(_save_on_cpu_called)
                            out = m(inp)

                        if check_offload:
                            self.assertTrue(_save_on_cpu_called)
                        loss = out.sum()
                        loss.backward()
                        losses.append(loss)
                        outputs.append(out)
                        _save_on_cpu_called = False

                    self._verify_parity(losses, outputs, models)

            dist.barrier()


注释：


                ):

这里开始定义一个方法，没有参数。


                    fsdp_kwargs = {"cpu_offload": cpu_offload, "use_orig_params": use_orig_params}
                    global _save_on_cpu_called

定义 `fsdp_kwargs` 字典，用于传递给 FSDP 模块的参数；声明全局变量 `_save_on_cpu_called`。


                    with patch_save_on_cpu(get_patched_save_on_cpu()):

使用 `patch_save_on_cpu` 修饰器来临时修改 `_save_on_cpu` 的行为。


                        seq = TestFSDPCheckpoint.SequentialModule().to(torch.cuda.current_device())

创建 `SequentialModule` 实例，并将其移动到当前 CUDA 设备。


                        # Runs FSDP with no checkpointing
                        fsdp_only_seq = FSDP(deepcopy(seq), **fsdp_kwargs)

使用 FSDP 包装 `seq`，不使用检查点功能。


                        # Runs checkpoint-wrapped FSDP
                        if offload_activations:
                            wrapper_to_use = offload_wrapper
                        else:
                            wrapper_to_use = checkpoint_wrapper

                        checkpointed_fsdp = wrapper_to_use(
                            FSDP(deepcopy(seq), **fsdp_kwargs),
                        )

根据 `offload_activations` 的值选择使用 `offload_wrapper` 或 `checkpoint_wrapper` 对 FSDP 进行包装，启用检查点功能。


                        # Runs FSDP-wrapped checkpointed module
                        fsdp_wrapped_checkpoint = FSDP(
                            wrapper_to_use(deepcopy(seq)),
                            **fsdp_kwargs,
                        )

使用 FSDP 包装使用了检查点的 `seq`。


                        # Runs FSDP with manual calls to checkpoint.
                        fsdp_call_checkpoint = FSDP(deepcopy(seq), **fsdp_kwargs)

使用 FSDP 包装 `seq`，手动调用检查点。


                        # note that reentrant-based checkpointing requires inputs to have grad
                        # flag set.

注意，基于重入的检查点需要设置输入的 `grad` 标志。


                        inp = torch.randn(
                            10, 3, device=torch.cuda.current_device(), requires_grad=True
                        )

创建一个张量 `inp`，形状为 `(10, 3)`，在当前 CUDA 设备上，需要计算梯度。


                        models = [
                            fsdp_only_seq,
                            checkpointed_fsdp,
                            fsdp_wrapped_checkpoint,
                            fsdp_call_checkpoint,
                        ]

将四个不同配置的模型放入列表 `models`。


                        # Ensure _save_on_cpu is not yet called
                        self.assertFalse(_save_on_cpu_called)

确保 `_save_on_cpu` 尚未被调用。


                        for i in range(6):

循环六次。


                            losses = []
                            outputs = []
                            for m in models:

初始化 `losses` 和 `outputs` 列表，遍历 `models` 列表。


                                check_offload = (
                                    m != fsdp_only_seq and i == 0 and offload_activations
                                )
                                if m == fsdp_call_checkpoint:
                                    # _save_on_cpu should not be called yet
                                    self.assertFalse(_save_on_cpu_called)
                                    offload_ctx = (
                                        get_patched_save_on_cpu()(pin_memory=True)
                                        if offload_activations
                                        else contextlib.nullcontext()
                                    )
                                    with offload_ctx:
                                        out = checkpoint(m, inp, use_reentrant=True)
                                else:
                                    # _save_on_cpu should not be called yet
                                    self.assertFalse(_save_on_cpu_called)
                                    out = m(inp)

根据 `m` 的不同执行不同的操作：如果 `m` 是 `fsdp_call_checkpoint`，使用 `checkpoint` 函数并启用重入；否则直接调用 `m`。


                                if check_offload:
                                    self.assertTrue(_save_on_cpu_called)

如果 `check_offload` 为真，则确保 `_save_on_cpu` 已经被调用。


                                loss = out.sum()
                                loss.backward()
                                losses.append(loss)
                                outputs.append(out)
                                _save_on_cpu_called = False

计算输出 `out` 的总和作为损失，计算损失的梯度并添加到 `losses` 列表中，将 `_save_on_cpu_called` 设为假。


                            self._verify_parity(losses, outputs, models)

调用 `_verify_parity` 方法验证模型之间的一致性。


            dist.barrier()

在分布式设置中同步所有进程。
# 实例化一个带有参数化测试的 TestFSDPCheckpoint 类的对象
instantiate_parametrized_tests(TestFSDPCheckpoint)


class CheckpointModule(nn.Module):
    def __init__(self, checkpoint: bool = False, use_reentrant: bool = True):
        super().__init__()
        # 创建一个包含4个线性层的序列
        self.seq = nn.Sequential(*[nn.Linear(100, 100) for _ in range(4)])
        # 是否使用检查点技术的标志
        self.checkpoint = checkpoint
        # 是否使用可重入（reentrant）的标志
        self.use_reentrant = use_reentrant

    def forward(self, x):
        # 如果启用检查点，则调用 checkpoint 函数，否则直接执行序列操作
        return (
            checkpoint(self.seq, x, use_reentrant=self.use_reentrant)
            if self.checkpoint
            else self.seq(x)
        )


class ModelWithCheckpointSubmodule(nn.Module):
    def __init__(self, checkpoint: bool = False, use_reentrant: bool = True):
        super().__init__()
        # 创建第一层线性层
        self.l1 = nn.Linear(100, 100)
        # 创建两个 CheckpointModule 实例作为子模块
        self.s1 = CheckpointModule(checkpoint, use_reentrant)
        self.s2 = CheckpointModule(checkpoint, use_reentrant)
        # 创建 ReLU 激活函数层
        self.relu = nn.ReLU()
        # 创建第二层线性层
        self.l2 = nn.Linear(100, 100)

    def forward(self, x):
        # 前向传播过程：先 s1，再 s2，最后经过 ReLU 和 l2 层处理
        return self.l2(self.relu(self.s2(self.s1(self.l1(x)))))


class TestModel(nn.Module):
    def __init__(self, checkpoint: bool = False, use_reentrant: bool = True):
        super().__init__()
        # 创建第一层线性层
        self.l1 = nn.Linear(100, 100)
        # 创建 ReLU 激活函数层
        self.relu = nn.ReLU()
        # 创建两个 ModelWithCheckpointSubmodule 实例作为子模块
        self.checkpoint1 = ModelWithCheckpointSubmodule(checkpoint, use_reentrant)
        self.checkpoint2 = ModelWithCheckpointSubmodule(checkpoint, use_reentrant)
        # 创建第二层线性层
        self.l2 = nn.Linear(100, 100)

    def forward(self, x):
        # 前向传播过程：先 checkpoint1，再 checkpoint2，最后经过 ReLU 和 l2 层处理
        return self.l2(self.relu(self.checkpoint2(self.checkpoint1(self.l1(x)))))


class TestFSDPCheckpointSubmodule(FSDPTest):
    # TODO: 当 use_reentrant = True 时，梯度值检查偶尔失败
    # 标记：仅在 GPU 数量不小于 2 时执行
    @skip_if_lt_x_gpu(2)
    # 参数化测试：设置 use_reentrant 参数为 False，进行测试
    @parametrize("use_reentrant", [False])
    # 定义一个测试方法，测试模型的子模块是否正常工作，接受一个布尔类型参数用于选择是否使用可重入模式
    def test_checkpoint_submodule(self, use_reentrant: bool):
        # 创建一个 TestModel 对象，使用 CUDA 运行，并进行深拷贝
        model = TestModel(use_reentrant=use_reentrant).cuda()
        model_ac = deepcopy(model)

        # 遍历模型的所有子模块，如果是 CheckpointModule 类型，则设置其 checkpoint 属性为 True
        for _, m in model_ac.named_modules():
            if isinstance(m, CheckpointModule):
                m.checkpoint = True

        # 断言检查，确保模型的特定子模块的 checkpoint 属性为 True
        self.assertTrue(model_ac.checkpoint1.s1.checkpoint)
        self.assertTrue(model_ac.checkpoint2.s2.checkpoint)

        # 定义 FSDP 配置参数
        fsdp_kwargs = {
            "device_id": torch.cuda.current_device(),
            "sharding_strategy": ShardingStrategy.NO_SHARD,
        }

        # 使用 FSDP 封装不需要检查点的模型子模块
        model.checkpoint1 = FSDP(module=model.checkpoint1, **fsdp_kwargs)
        model.checkpoint2 = FSDP(module=model.checkpoint2, **fsdp_kwargs)

        # 使用 FSDP 封装需要检查点的模型子模块
        model_ac.checkpoint1 = FSDP(module=model_ac.checkpoint1, **fsdp_kwargs)
        model_ac.checkpoint2 = FSDP(module=model_ac.checkpoint2, **fsdp_kwargs)

        # 创建一个 CUDA 设备上的随机张量 x
        x = torch.randn(2, 100, device="cuda")

        # 对模型进行前向传播、求和、反向传播操作
        model(x).sum().backward()
        model_ac(x).sum().backward()

        # 比较两个模型的同名参数是否相等，并且比较它们的梯度是否在数值上接近
        for (n1, p1), (n2, p2) in zip(
            model.named_parameters(), model_ac.named_parameters()
        ):
            self.assertEqual(n1, n2)
            self.assertTrue(p1.grad.allclose(p2.grad))
# 实例化一个带参数的测试，参数为 TestFSDPCheckpointSubmodule 类
instantiate_parametrized_tests(TestFSDPCheckpointSubmodule)

# 如果当前脚本作为主程序执行
if __name__ == "__main__":
    # 运行测试函数
    run_tests()
```