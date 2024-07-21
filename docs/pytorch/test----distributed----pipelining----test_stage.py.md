# `.\pytorch\test\distributed\pipelining\test_stage.py`

```
# 引入必要的标准库和第三方库模块
import os
import sys
import tempfile

# 引入 PyTorch 库及其分布式模块
import torch
import torch.distributed as dist
from torch.distributed.pipelining import (
    build_stage,
    pipeline,
    PipelineStage,
    ScheduleGPipe,
)
from torch.distributed.pipelining._utils import PipeliningShapeError

# 引入测试相关的辅助模块和函数
from torch.testing._internal.common_cuda import TEST_MULTIGPU
from torch.testing._internal.common_distributed import (
    MultiProcContinousTest,
    requires_nccl,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    skip_but_pass_in_sandcastle_if,
)
from torch.utils._pytree import tree_map_only

# 设置隐藏层维度、批量大小和分块数
d_hid = 512
batch_size = 256
chunks = 4

# 设置随机种子
torch.manual_seed(0)

# 定义用于模型混合精度仿真的钩子函数
def get_dtype_change_hook(new_dtype):
    """A simple hook for simulating mixed precision"""
    def dtype_change_hook(module, input, output):
        def f(x):
            return x.to(new_dtype)
        return tree_map_only(torch.Tensor, f, output)
    return dtype_change_hook

# 定义用于模型输出形状错误仿真的钩子函数
def get_flatten_hook():
    """A simple hook for simulating wrong model output shape"""
    def flatten_hook(module, input, output):
        def f(x):
            return x.flatten()
        return tree_map_only(torch.Tensor, f, output)
    return flatten_hook

# 定义 StageTest 类，继承自 MultiProcContinousTest 类
class StageTest(MultiProcContinousTest):
    @classmethod
    def backend_str(cls) -> str:
        # 返回使用的分布式后端类型
        return "nccl"

    @classmethod
    def setUpClass(cls):
        """
        Class-scope test fixture. Run once for entire test class, before any test starts.
        Set up the device.
        """
        super().setUpClass()
        # 根据当前进程的排名选择对应的 CUDA 设备
        dev_id = cls.rank % torch.cuda.device_count()
        cls.device = torch.device(f"cuda:{dev_id}")

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "NCCL test requires 2+ GPUs")
    @parametrize("ModelClass", [ExampleCode, MultiMLP])
    # 定义一个测试方法，用于测试管道模型的行为
    def test_tracer(self, ModelClass):
        # 根据提供的隐藏层维度创建模型实例
        mod = ModelClass(d_hid)
        # 将模型移动到指定的设备上
        mod.to(self.device)

        # 创建一个随机张量作为输入数据，并按照指定的方式进行分块
        x = torch.randn(batch_size, d_hid, device=self.device)
        x_mb = x.chunk(chunks)[0]

        # 如果模型有分割规格属性，则获取其值，否则设为None
        split_spec = mod.split_spec if hasattr(mod, "split_spec") else None
        # 创建一个管道对象，用于处理输入数据
        pipe = pipeline(
            mod,
            mb_args=(x_mb,),
            split_spec=split_spec,
        )

        # 构建管道的执行阶段，指定当前进程的排名和设备
        stage = pipe.build_stage(
            self.rank,
            self.device,
        )

        # 将阶段与一个调度器关联起来
        schedule = ScheduleGPipe(stage, chunks)

        # 运行管道的一个步骤
        def _run_step(x):
            # 如果当前进程是第一个进程，传入输入数据并执行步骤
            if self.rank == 0:
                return schedule.step(x)
            # 否则，只执行步骤，不传入输入数据
            else:
                return schedule.step()

        # 执行管道的一个步骤，并将结果保存在 out 变量中
        out = _run_step(x)

        # 如果当前进程是最后一个进程，进行结果检查
        if self.rank == self.world_size - 1:
            # 计算模型在输入数据上的期望输出
            ref_out = mod(x)
            # 使用测试框架验证实际输出和期望输出的接近程度
            torch.testing.assert_close(out, ref_out, atol=1e-3, rtol=5e-2)

        # 测试模块的限定名称映射
        submod_keys = stage.submod.state_dict().keys()
        # 确保子模块的所有键在原始模型的键中都存在
        old_keys = mod.state_dict().keys()
        assert all(k in old_keys for k in submod_keys)

        # 如果当前进程是第一个进程
        if self.rank == 0:
            # 下面的代码本意是在所有进程上运行，但是如果第一个进程出错，
            # 将不会执行发送操作来解锁第二个进程。

            # TODO(whc) can't test this until fixing args/kwargs issue
            # with self.assertRaisesRegex(PipeliningShapeError, "shape mismatch"):
            #     _run_step(torch.randn(batch_size + 1, d_hid, device=self.device))

            # 预期会抛出 PipeliningShapeError 异常，提示数据类型不匹配
            with self.assertRaisesRegex(PipeliningShapeError, "dtype mismatch"):
                _run_step(x.to(torch.int32))

            # 阶段的 MLP 层的输出将通过此钩子被扁平化，
            # 预期会抛出 PipeliningShapeError 异常，提示形状不匹配
            handle = stage.submod.register_forward_hook(get_flatten_hook())
            with self.assertRaisesRegex(PipeliningShapeError, "shape mismatch"):
                _run_step(x)
            handle.remove()

            # 注册一个钩子，将阶段的数据类型更改为 torch.bfloat16
            stage.submod.register_forward_hook(get_dtype_change_hook(torch.bfloat16))
            # 预期会抛出 PipeliningShapeError 异常，提示数据类型不匹配
            with self.assertRaisesRegex(PipeliningShapeError, "dtype mismatch"):
                _run_step(x)

    # 标记需要使用 NCCL 的测试方法，用于多 GPU 测试
    @requires_nccl()
    # 如果不测试多 GPU，则跳过测试但在沙盒中继续执行
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "NCCL test requires 2+ GPUs")
    # 使用参数化装饰器，指定测试的模型类为 ModelWithKwargs
    @parametrize("ModelClass", [ModelWithKwargs])
    # 定义一个测试函数，用于测试模型管道中的追踪关键字参数
    def test_tracer_kwargs(self, ModelClass):
        # 使用给定的隐藏层维度创建模型实例
        mod = ModelClass(d_hid)
        # 将模型移动到指定的设备上
        mod.to(self.device)

        # 创建随机输入张量 x 和 y，形状为 (batch_size, d_hid)，并且放置在指定的设备上
        x = torch.randn(batch_size, d_hid, device=self.device)
        y = torch.randn(batch_size, d_hid, device=self.device)

        # 将 x 和 y 分割为多个小块，选择第一个小块作为 x_mb 和 y_mb
        x_mb = x.chunk(chunks)[0]
        y_mb = y.chunk(chunks)[0]

        # 创建管道对象，传入模型和小块参数 x_mb 和关键字参数 y_mb
        pipe = pipeline(
            mod,
            mb_args=(x_mb,),
            mb_kwargs={"y": y_mb},
        )

        # 获取指定排名的管道阶段模块
        stage_mod = pipe.get_stage_module(self.rank)

        # 测试 build_stage 函数，构建阶段对象 stage
        stage = build_stage(
            stage_mod,
            self.rank,
            pipe.info(),
            self.device,
        )

        # 将阶段对象 stage 与调度器 ScheduleGPipe 结合
        schedule = ScheduleGPipe(stage, chunks)

        # 运行管道
        def _run_step(x):
            # 如果当前进程排名为 0，则传递 y 参数给 schedule.step；否则仅调用 schedule.step()
            if self.rank == 0:
                return schedule.step(x, y=y)
            else:
                return schedule.step()

        # 最后一个进程排名检查结果
        out = _run_step(x)
        if self.rank == self.world_size - 1:
            # 如果当前进程排名是世界大小减一，则比较 out 和模型直接调用的结果 ref_out
            ref_out = mod(x, y=y)
            torch.testing.assert_close(out, ref_out, atol=1e-3, rtol=5e-2)

        # 测试 qualname 映射
        submod_keys = stage.submod.state_dict().keys()
        # 确认 submod_keys 中的所有键都存在于原始模型 mod 的状态字典中
        old_keys = mod.state_dict().keys()
        assert all(k in old_keys for k in submod_keys)

        # 如果当前进程排名为 0
        if self.rank == 0:
            # 测试输入形状不匹配的情况，期望捕获 PipeliningShapeError 异常并输出 "shape mismatch" 提示信息
            with self.assertRaisesRegex(PipeliningShapeError, "shape mismatch"):
                _run_step(torch.randn(batch_size + 1, d_hid, device=self.device))

            # 测试输入数据类型不匹配的情况，期望捕获 PipeliningShapeError 异常并输出 "dtype mismatch" 提示信息
            with self.assertRaisesRegex(PipeliningShapeError, "dtype mismatch"):
                _run_step(x.to(torch.int32))

            # 注册一个钩子函数，用于在 stage 的 mlp 层输出前扁平化数据，预期捕获 PipeliningShapeError 异常并输出 "shape mismatch" 提示信息
            # 然后移除这个钩子函数
            handle = stage.submod.register_forward_hook(get_flatten_hook())
            with self.assertRaisesRegex(PipeliningShapeError, "shape mismatch"):
                _run_step(x)
            handle.remove()

            # 注册一个钩子函数，用于在 stage 的子模块输出前改变数据类型为 torch.bfloat16
            stage.submod.register_forward_hook(get_dtype_change_hook(torch.bfloat16))
            # 预期捕获 PipeliningShapeError 异常并输出 "dtype mismatch" 提示信息
            with self.assertRaisesRegex(PipeliningShapeError, "dtype mismatch"):
                _run_step(x)
    def test_manual(self):
        # 创建具有指定隐藏层维度和层数的多层感知机模型
        full_mod = MultiMLP(d_hid, n_layers=self.world_size)
        # 将模型移动到指定的设备上
        full_mod.to(self.device)
        # 获取指定子模块（根据当前进程的排名）
        stage_mod = full_mod.get_submodule(f"layers.{self.rank}")

        # 创建指定维度和设备上的随机张量作为输入
        x = torch.randn(batch_size, d_hid, device=self.device)

        # 创建流水线阶段对象，传入阶段模块、当前进程的排名、总进程数、设备和输入参数
        stage = PipelineStage(
            stage_mod,
            self.rank,
            self.world_size,
            self.device,
            input_args=x.chunk(chunks)[0],
        )

        # 将阶段对象和指定的切块数创建为调度器对象
        schedule = ScheduleGPipe(stage, chunks)

        # 定义内部函数来执行单步操作
        def _run_step(x):
            if self.rank == 0:
                return schedule.step(x)
            else:
                return schedule.step()

        # 执行单步操作，获取输出结果
        out = _run_step(x)

        # 如果当前进程是最后一个进程，检查输出结果与整体模型在相同输入下的结果是否接近
        if self.rank == self.world_size - 1:
            ref_out = full_mod(x)
            torch.testing.assert_close(out, ref_out)

        # 如果当前进程是第一个进程，进行以下错误情况的测试
        if self.rank == 0:
            # 测试输入形状不匹配时是否抛出异常
            with self.assertRaisesRegex(PipeliningShapeError, "shape mismatch"):
                _run_step(torch.randn(batch_size + 1, d_hid, device=self.device))

            # 测试输入数据类型不匹配时是否抛出异常
            with self.assertRaisesRegex(PipeliningShapeError, "dtype mismatch"):
                _run_step(x.to(torch.int32))

            # 注册前向钩子以对阶段模块的输出进行扁平化，此时阶段应该引发错误
            handle = stage_mod.register_forward_hook(get_flatten_hook())
            with self.assertRaisesRegex(PipeliningShapeError, "shape mismatch"):
                _run_step(x)
            handle.remove()

            # 注册前向钩子以将输入数据类型更改为torch.bfloat16，测试是否抛出异常
            stage_mod.register_forward_hook(get_dtype_change_hook(torch.bfloat16))
            with self.assertRaisesRegex(PipeliningShapeError, "dtype mismatch"):
                _run_step(x)
    def test_custom_dw_with_fb_schedule(self):
        """Tests that separate weight grad function 'dw_runner' gets run under a schedule that's only aware of F/B."""
        # 创建一个 MultiMLP 对象 full_mod，包含指定隐藏层维度和全局层数
        full_mod = MultiMLP(d_hid, n_layers=self.world_size)
        # 将 full_mod 移动到指定设备上
        full_mod.to(self.device)
        # 获取指定层级的子模块 stage_mod
        stage_mod = full_mod.get_submodule(f"layers.{self.rank}")

        # 创建输入张量 x 和目标张量 target，形状为 batch_size x d_hid，位于指定设备上
        x = torch.randn(batch_size, d_hid, device=self.device)
        target = torch.randn(batch_size, d_hid, device=self.device)

        # 定义 CustomState 类，用于创建自定义状态和相关方法
        class CustomState:
            def __init__(self):
                self.i = 0

            # 定义 dw_builder 方法，返回一个自定义的 dw_runner 函数
            def dw_builder(self):
                """This simulates a function attached to a model with a custom backward.
                Each call to builder gives a new dw_runner that has some updated state to compute the latest dw.
                """
                def dw_runner():
                    # 内部函数，在 `backward_weight_one_chunk` 期间由 PipelineStage 调用
                    print(f"dw called {self.i}th time")
                    self.i += 1

                return dw_runner

        # 创建 CustomState 实例 cs
        cs = CustomState()

        # 创建 PipelineStage 实例 stage，用于管理单个管道阶段的执行
        stage = PipelineStage(
            stage_mod,
            self.rank,
            self.world_size,
            self.device,
            input_args=x.chunk(chunks)[0],
            dw_builder=cs.dw_builder,
        )

        # 创建 ScheduleGPipe 实例 schedule，用于管理 GPipe 计划和损失函数
        schedule = ScheduleGPipe(
            stage, chunks, loss_fn=torch.nn.MSELoss(reduction="sum")
        )

        # 定义 _run_step 函数，根据排名和世界大小选择执行步骤
        def _run_step(x):
            if self.rank == 0:
                return schedule.step(x)
            elif self.rank == self.world_size - 1:
                return schedule.step(target=target)
            else:
                return schedule.step()

        # 运行 _run_step 函数，并验证 CustomState 中的计数是否为 chunks
        out = _run_step(x)
        self.assertEqual(cs.i, chunks)

        # 如果当前排名为世界大小减一，检查输出结果是否与全模型 full_mod(x) 接近
        if self.rank == self.world_size - 1:
            ref_out = full_mod(x)
            torch.testing.assert_close(out, ref_out)

        # 如果当前排名为0，验证在输入形状错误时是否引发 PipeliningShapeError 异常
        if self.rank == 0:
            with self.assertRaisesRegex(PipeliningShapeError, "shape mismatch"):
                _run_step(torch.randn(batch_size + 1, d_hid, device=self.device))

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "NCCL test requires 2+ GPUs")
    # 定义测试方法，用于测试期望的错误是否被引发
    def test_custom_dw_errors(self):
        """Tests expected errors are raised"""
        # 创建完整的多层感知机模型对象，设置隐藏层维度和层数
        full_mod = MultiMLP(d_hid, n_layers=self.world_size)
        # 将完整模型移动到指定设备上
        full_mod.to(self.device)
        # 获取指定子模块，子模块路径为"layers.{self.rank}"
        stage_mod = full_mod.get_submodule(f"layers.{self.rank}")

        # 生成具有随机标准正态分布数据的张量，设备为self.device
        x = torch.randn(batch_size, d_hid, device=self.device)
        # 生成具有随机标准正态分布数据的目标张量，设备为self.device
        target = torch.randn(batch_size, d_hid, device=self.device)

        # 创建具有数据流管道阶段特征的PipelineStage对象，包括：
        # - stage_mod: 子模块对象
        # - self.rank: 当前进程的排名
        # - self.world_size: 总进程数
        # - self.device: 设备
        # - input_args: 根据x分块后的第一个分块数据作为输入
        # - dw_builder: 空的数据流构建器
        stage_with_dw_builder = PipelineStage(
            stage_mod,
            self.rank,
            self.world_size,
            self.device,
            input_args=x.chunk(chunks)[0],
            dw_builder=lambda: None,
        )
        # 断言特定异常（AssertionError）以给定字符串（"backward_one_chunk"）抛出
        with self.assertRaisesRegex(AssertionError, "backward_one_chunk"):
            # 调用具有数据流构建器的管道阶段对象的方法，并期望抛出异常
            stage_with_dw_builder.backward_weight_one_chunk(bwd_chunk_id=0)

        # 创建另一个具有数据流管道阶段特征的PipelineStage对象，包括：
        # - stage_mod: 子模块对象
        # - self.rank: 当前进程的排名
        # - self.world_size: 总进程数
        # - self.device: 设备
        # - input_args: 根据x分块后的第一个分块数据作为输入
        # - dw_builder: None，即没有数据流构建器
        stage_without_dw_builder = PipelineStage(
            stage_mod,
            self.rank,
            self.world_size,
            self.device,
            input_args=x.chunk(chunks)[0],
            dw_builder=None,
        )

        # 断言特定异常（AssertionError）以给定字符串（"dw_builder"）抛出
        with self.assertRaisesRegex(AssertionError, "dw_builder"):
            # 调用没有数据流构建器的管道阶段对象的方法，并期望抛出异常
            stage_without_dw_builder.backward_one_chunk(
                bwd_chunk_id=0, full_backward=False
            )
# 实例化参数化测试，将 StageTest 作为参数传递
instantiate_parametrized_tests(StageTest)

if __name__ == "__main__":
    # 检查是否存在 GPU 和 NCCL
    if not (
        dist.is_available()
        and dist.is_nccl_available()
        and torch.cuda.device_count() > 1
    ):
        # 如果 c10d NCCL 不可用或者 GPU 数量不足，跳过测试并打印错误信息到标准错误输出
        print(
            "c10d NCCL not available or not enough GPUs, skipping tests",
            file=sys.stderr,
        )
        # 退出程序，返回状态码 0 表示正常退出
        sys.exit(0)

    rank = int(os.getenv("RANK", -1))
    world_size = int(os.getenv("WORLD_SIZE", 2))

    if rank != -1:
        # 使用 torchrun 或其他多进程启动器启动。直接运行测试。
        StageTest.run_rank(rank, world_size)
    else:
        # 作为单进程启动。生成子进程来运行测试。
        # 同时需要一个用于 `init_process_group` 的会合文件。
        rdvz_file = tempfile.NamedTemporaryFile(delete=False).name
        # 使用 torch.multiprocessing.spawn 启动多个进程来运行 StageTest.run_rank
        torch.multiprocessing.spawn(
            StageTest.run_rank,
            nprocs=world_size,  # 进程数为 world_size
            args=(world_size, rdvz_file),  # 传递参数 world_size 和 rendezvous 文件名
        )
```