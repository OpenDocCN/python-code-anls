# `.\pytorch\test\distributed\pipelining\test_schedule.py`

```
# 版权声明和所有者信息
# 所有者：["oncall: distributed"]
import copy  # 导入深拷贝模块
import logging  # 导入日志模块
import os  # 导入操作系统相关模块
import sys  # 导入系统相关模块
import tempfile  # 导入临时文件模块
import unittest  # 导入单元测试模块
from typing import Dict, List, Optional, Tuple  # 导入类型提示模块

from model_registry import ModelWithKwargs, MultiMLP  # 导入模型注册模块
from schedule_registry import ScheduleUnbalanced, ScheduleVShaped, ScheduleWithW  # 导入调度注册模块

import torch  # 导入PyTorch深度学习库
import torch.distributed as dist  # 导入分布式训练模块
from torch.distributed.pipelining import (  # 导入分布式流水线相关模块
    pipeline,
    PipelineStage,
    Schedule1F1B,
    ScheduleGPipe,
    ScheduleInterleaved1F1B,
    ScheduleLoopedBFS,
)
from torch.distributed.pipelining.schedules import _Action, _ComputationType  # 导入流水线调度相关模块
from torch.distributed.pipelining.stage import _PipelineStageBase  # 导入流水线阶段基类
from torch.testing._internal.common_cuda import TEST_MULTIGPU  # 导入CUDA通用测试工具
from torch.testing._internal.common_distributed import (  # 导入分布式通用测试工具
    MultiProcContinousTest,
    requires_nccl,
)
from torch.testing._internal.common_utils import (  # 导入通用工具函数
    instantiate_parametrized_tests,
    parametrize,
    skip_but_pass_in_sandcastle_if,
)

logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器

d_hid = 512  # 定义隐藏层维度
batch_size = 256  # 定义批量大小

torch.manual_seed(0)  # 设置随机种子以保证结果的可重复性


class MockPipelineStage(_PipelineStageBase):
    def __init__(self, *args, **kwargs):
        # 模拟必要的属性
        self.num_stages = kwargs.get("num_stages", 1)  # 初始化阶段数，默认为1
        self.group_size = kwargs.get("group_size", 1)  # 初始化组大小，默认为1
        self.group_rank = kwargs.get("group_rank", 0)  # 初始化组排名，默认为0
        self.group = kwargs.get("group", None)  # 初始化组对象，默认为空
        self.stage_index_to_group_rank = kwargs.get("stage_index_to_group_rank", None)  # 初始化阶段索引到组排名的映射表

    def _create_grad_recv_info(self, *args, **kwargs):
        return None  # 返回空，模拟创建梯度接收信息

    def _prepare_forward_infra(self, n_microbatches):
        pass  # 准备前向传播基础设施，不执行任何操作

    def _prepare_backward_infra(self, n_microbatches):
        pass  # 准备反向传播基础设施，不执行任何操作


class ScheduleTest(MultiProcContinousTest):
    @classmethod
    def backend_str(cls) -> str:
        # 使用NCCL后端进行测试
        return "nccl"

    @classmethod
    def setUpClass(cls):
        """
        类级别的测试夹具。整个测试类执行前执行一次，用于设定设备。
        """
        super().setUpClass()  # 调用父类的setUpClass方法
        dev_id = cls.rank % torch.cuda.device_count()  # 计算设备ID，以当前进程排名除以GPU数量取余数
        cls.device = torch.device(f"cuda:{dev_id}")  # 设置设备为对应的CUDA设备

    @requires_nccl()  # 标记需要使用NCCL
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "NCCL test requires 2+ GPUs")
    @parametrize("ScheduleClass", [ScheduleGPipe, Schedule1F1B])
    # 定义一个测试方法，用于测试多迭代情况下的模型行为
    def test_multi_iter(self, ScheduleClass):
        # 创建一个多层感知机模型，隐藏层维度为 d_hid，层数为 self.world_size
        mod = MultiMLP(d_hid, n_layers=self.world_size)
        # 将模型移动到指定设备上
        mod.to(self.device)

        # 创建输入数据 x 和目标数据 target，均为随机张量，维度为 batch_size x d_hid
        x = torch.randn(batch_size, d_hid, device=self.device)
        target = torch.randn(batch_size, d_hid, device=self.device)
        # 定义损失函数为均方误差损失
        loss_fn = torch.nn.MSELoss(reduction="sum")

        # 将输入数据 x 拆分为多个小块，取第一个块作为 x_mb
        chunks = 4
        x_mb = x.chunk(chunks)[0]

        # 创建一个数据处理流水线
        # 如果模型有 split_spec 属性，则使用它作为拆分规范
        split_spec = mod.split_spec if hasattr(mod, "split_spec") else None
        pipe = pipeline(
            mod,
            mb_args=(x_mb,),
            split_spec=split_spec,
        )

        # 构建流水线的执行阶段
        stage = pipe.build_stage(
            self.rank,
            self.device,
        )

        # 将执行阶段与调度器关联，使用给定的 ScheduleClass 和损失函数 loss_fn
        schedule = ScheduleClass(stage, chunks, loss_fn=loss_fn)

        # 运行模型迭代
        for _ in range(20):
            # 如果当前进程的 rank 为 0，则执行 schedule.step(x)
            if self.rank == 0:
                schedule.step(x)
            # 如果当前进程的 rank 等于 self.world_size - 1，则执行 schedule.step(target=target, losses=losses)
            elif self.rank == self.world_size - 1:
                losses = []
                out = schedule.step(target=target, losses=losses)
            # 其它情况下，执行 schedule.step()，即不传递任何参数
            else:
                schedule.step()

    # 标记需要使用 NCCL 的测试方法
    @requires_nccl()
    # 如果不满足 TEST_MULTIGPU 条件，则在沙盒模式下跳过测试
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "NCCL test requires 2+ GPUs")
    # 参数化测试，使用 ScheduleGPipe 和 Schedule1F1B 两种调度器类
    @parametrize("ScheduleClass", [ScheduleGPipe, Schedule1F1B])
    def test_kwargs_with_tracer(self, ScheduleClass):
        # 创建带有关键字参数的模型
        mod = ModelWithKwargs(d_hid)
        # 将模型移动到指定设备上
        mod.to(self.device)

        # 创建输入数据 x, y 和目标数据 target，均为随机张量，维度为 batch_size x d_hid
        x = torch.randn(batch_size, d_hid, device=self.device)
        y = torch.randn(batch_size, d_hid, device=self.device)
        target = torch.randn(batch_size, d_hid, device=self.device)
        # 定义损失函数为均方误差损失
        loss_fn = torch.nn.MSELoss(reduction="sum")

        # 将输入数据 x 和 y 拆分为多个小块，取第一个块作为 x_mb 和 y_mb
        chunks = 4
        x_mb = x.chunk(chunks)[0]
        y_mb = y.chunk(chunks)[0]

        # 创建数据处理流水线，传入 x_mb 作为位置参数，y_mb 作为关键字参数
        pipe = pipeline(
            mod,
            mb_args=(x_mb,),
            mb_kwargs={"y": y_mb},
        )

        # 构建流水线的执行阶段
        stage = pipe.build_stage(
            self.rank,
            self.device,
        )

        # 将执行阶段与调度器关联，使用给定的 ScheduleClass 和损失函数 loss_fn
        schedule = ScheduleClass(stage, chunks, loss_fn=loss_fn)

        # 运行模型迭代
        if self.rank == 0:
            # 如果当前进程的 rank 为 0，则执行 schedule.step(x, y=y)
            schedule.step(x, y=y)
        elif self.rank == self.world_size - 1:
            # 如果当前进程的 rank 等于 self.world_size - 1，则执行 schedule.step(target=target, losses=losses)
            losses = []
            out = schedule.step(target=target, losses=losses)
        else:
            # 其它情况下，执行 schedule.step()，即不传递任何参数
            schedule.step()

        # 进程间同步
        dist.barrier()

        # 最后一个进程检查结果
        if self.rank == self.world_size - 1:
            # 计算模型在输入数据 x, y 上的参考输出和损失值
            ref_out = mod(x, y=y)
            ref_loss = loss_fn(ref_out, target)
            pipe_loss = sum(losses)
            # 使用测试工具验证模型输出 out 与参考输出 ref_out 的接近程度
            torch.testing.assert_close(out, ref_out, rtol=1e-2, atol=5e-3)
            # 使用测试工具验证流水线损失 pipe_loss 与参考损失 ref_loss 的接近程度
            torch.testing.assert_close(pipe_loss, ref_loss)

    # 标记需要使用 NCCL 的测试方法
    @requires_nccl()
    # 如果不满足 TEST_MULTIGPU 条件，则在沙盒模式下跳过测试
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "NCCL test requires 2+ GPUs")
    # 参数化测试，使用 ScheduleGPipe 和 Schedule1F1B 两种调度器类
    @parametrize("ScheduleClass", [ScheduleGPipe, Schedule1F1B])
    # 参数化测试，使用 MultiMLP 模型类
    @parametrize("ModelClass", [MultiMLP])
    # 定义一个测试方法，用于测试具有追踪器的梯度计算
    def test_grad_with_tracer(self, ScheduleClass, ModelClass):
        # 创建模型对象，输入参数为隐藏层维度
        mod = ModelClass(d_hid)
        # 将模型移动到指定设备上
        mod.to(self.device)

        # 深拷贝模型对象作为参考模型
        ref_mod = copy.deepcopy(mod)
        # 生成一个随机张量作为输入数据
        x = torch.randn(batch_size, d_hid, device=self.device)

        # 在不追踪梯度的上下文中执行以下操作
        with torch.no_grad():
            # 使用参考模型计算输出
            y = ref_mod(x)
            # 添加一个小的扰动作为目标值
            target = y + torch.randn(batch_size, d_hid, device=self.device)

        # 定义损失函数为均方误差损失
        loss_fn = torch.nn.MSELoss(reduction="sum")

        # 运行参考模型的梯度计算
        for _ in range(2):
            ref_mod.zero_grad()
            ref_out = ref_mod(x)
            ref_loss = loss_fn(ref_out, target)
            ref_loss.backward()

        # 创建一个流水线对象
        chunks = 4
        x_mb = x.chunk(chunks)[0]
        split_spec = mod.split_spec if hasattr(mod, "split_spec") else None
        # 构建流水线
        pipe = pipeline(
            mod,
            mb_args=(x_mb,),
            split_spec=split_spec,
        )

        # 构建流水线阶段
        stage = pipe.build_stage(
            self.rank,
            self.device,
        )

        # 将阶段对象与调度器关联
        schedule = ScheduleClass(stage, chunks, loss_fn=loss_fn)

        # 运行流水线阶段
        stage_module = pipe.get_stage_module(self.rank)
        for _ in range(2):
            # 清零梯度
            stage_module.zero_grad()
            if self.rank == 0:
                # 如果是第一个进程，执行调度器的步骤
                schedule.step(x)
            elif self.rank == self.world_size - 1:
                # 如果是最后一个进程，执行带有目标值和损失列表的调度器步骤
                losses = []
                out = schedule.step(target=target, losses=losses)
            else:
                # 其他进程执行调度器的步骤
                schedule.step()

        # 等待所有进程执行完毕
        dist.barrier()

        # 最后一个进程检查结果
        if self.rank == self.world_size - 1:
            # 检查输出是否接近参考输出
            torch.testing.assert_close(out, ref_out)
            # 检查损失是否接近参考损失
            # 由于上面损失函数的减少使用了 "sum"，这里也使用 "sum" 将微批次损失减少为单个值
            pipe_loss = sum(losses)
            torch.testing.assert_close(pipe_loss, ref_loss)

        # 每个进程检查梯度
        for name, p in stage_module.named_parameters():
            ref_p = ref_mod.get_parameter(name)
            try:
                # 检查参数的梯度是否接近参考模型的梯度
                torch.testing.assert_close(p.grad, ref_p.grad, rtol=1e-5, atol=4e-5)
            except AssertionError:
                # 如果检查失败，打印错误消息
                print(f"Gradient test failed for {name}: {p.grad} vs {ref_p.grad}")
                raise
   `
    # 定义一个测试函数，用于测试带有手动梯度更新的情况
    def test_grad_with_manual(self, ScheduleClass):
        # 创建一个具有多层隐藏层的多层感知机模型
        full_mod = MultiMLP(d_hid, n_layers=self.world_size)
        # 将模型移动到指定设备上
        full_mod.to(self.device)

        # 深度复制完整模型以备份
        ref_mod = copy.deepcopy(full_mod)
        # 生成一个随机输入张量
        x = torch.randn(batch_size, d_hid, device=self.device)
        # 使用 torch.no_grad() 上下文管理器，确保在该区域内不计算梯度
        with torch.no_grad():
            # 对复制模型进行前向传播
            y = ref_mod(x)
            # 添加一个小扰动
            target = y + torch.randn(batch_size, d_hid, device=self.device)

        # 定义损失函数为均方误差损失函数
        loss_fn = torch.nn.MSELoss(reduction="sum")

        # 运行参考模型
        for _ in range(2):
            # 梯度归零
            ref_mod.zero_grad()
            # 对参考模型进行前向传播
            ref_out = ref_mod(x)
            # 计算参考模型的损失
            ref_loss = loss_fn(ref_out, target)
            # 反向传播损失
            ref_loss.backward()

        # 获取子模块的名称，例如 `layers.0` 或 `layers.1`
        submod_name = f"layers.{self.rank}"
        # 获取完整模型中指定子模块
        stage_module = full_mod.get_submodule(submod_name)
        # 将输入张量分块，创建一个管道阶段以包装该子模块
        chunks = 4
        stage = PipelineStage(
            stage_module,
            self.rank,
            self.world_size,
            self.device,
            input_args=x.chunk(chunks)[0],
        )

        # 将管道阶段附加到调度策略中
        schedule = ScheduleClass(stage, chunks, loss_fn=loss_fn)

        # 运行
        for _ in range(2):
            # 梯度归零
            stage_module.zero_grad()
            if self.rank == 0:
                # 如果是第一个进程，执行调度策略的步骤
                schedule.step(x)
            elif self.rank == self.world_size - 1:
                # 如果是最后一个进程，初始化损失列表，执行调度策略的步骤并接收损失值
                losses = []
                out = schedule.step(target=target, losses=losses)
            else:
                # 其他进程只需执行调度策略的步骤
                schedule.step()

        # 同步所有进程
        dist.barrier()

        # 最后一个进程检查结果
        if self.rank == self.world_size - 1:
            # 检查输出是否接近参考输出
            torch.testing.assert_close(out, ref_out)
            # 检查损失值
            # 由于上面损失函数使用的是 "sum" 归约方式，因此这里也使用 "sum" 将微批次的损失值归约为一个单一值
            pipe_loss = sum(losses)
            torch.testing.assert_close(pipe_loss, ref_loss)

        # 每个进程检查梯度
        ref_submod = ref_mod.get_submodule(submod_name)
        for name, p in stage_module.named_parameters():
            ref_p = ref_submod.get_parameter(name)
            try:
                # 断言当前进程的参数梯度与参考模型的参数梯度接近
                torch.testing.assert_close(p.grad, ref_p.grad, rtol=1e-5, atol=4e-5)
            except AssertionError:
                # 如果断言失败，打印错误消息并抛出异常
                print(f"Gradient test failed for {name}: {p.grad} vs {ref_p.grad}")
                raise
# 实例化带参数的测试，使用ScheduleTest作为参数
instantiate_parametrized_tests(ScheduleTest)


def format_pipeline_order(pipeline_order: Dict[int, List[Optional[_Action]]]):
    import itertools

    # 计算所有排程中步骤的最大数量
    num_steps = max(len(actions) for actions in pipeline_order.values())
    # 生成步骤标签列表，格式为"Step X"
    step_labels = [
        "Step " + str(i).zfill(len(str(num_steps - 1))) for i in range(num_steps)
    ]
    # 按键排序字典，并按排序后的键顺序获取值列表
    rank_actions = [
        pipeline_order.get(key, [""] * num_steps) for key in sorted(pipeline_order)
    ]
    # 转置列表中的列表（行变列）
    transposed_actions = list(itertools.zip_longest(*rank_actions, fillvalue=""))
    # 生成每个排程的列标签，格式为"Rank X"
    num_ranks = len(pipeline_order)
    rank_labels = ["Rank " + str(i) for i in range(num_ranks)]
    # 计算每列的最大长度，考虑标签的影响
    max_lengths = [
        max(len(str(item)) if item is not None else 0 for item in col)
        for col in zip(step_labels, *transposed_actions)
    ]
    # 格式化表头行，包括排程标签
    header_row = " " * (len(step_labels[0]) + 2) + " ".join(
        f"{label:<{max_lengths[i]}}" for i, label in enumerate(rank_labels)
    )
    # 格式化每一行及其相应标签
    formatted_rows = [
        f"{label}: "
        + " ".join(f"{str(item):<{max_lengths[i]}}" for i, item in enumerate(row))
        for label, row in zip(step_labels, transposed_actions)
    ]
    # 拼接所有行成为单个字符串表格
    formatted_table = (
        "=========== ALL_RANK_ACTIONS ===========\n"
        + header_row
        + "\n"
        + "\n".join(formatted_rows)
        + "\n"
    )
    return formatted_table


class TestSchedulePlan(unittest.TestCase):
    def _validate_pipeline_order(
        self,
        pipeline_order: Dict[int, List[Optional[_Action]]],
        num_microbatches: int,
        num_stages: int,
    ):
        @parametrize("ScheduleClass", [ScheduleInterleaved1F1B, ScheduleLoopedBFS])
    def test_pipeline_order(self, ScheduleClass):
        # 定义一组测试用例，其中包括不同的 num_local_stages、num_microbatches 和 group_size
        # 这些测试用例应该成功，因为 num_microbatches % group_size == 0
        test_cases = [
            # 少量阶段
            (2, 2, 2),
            (2, 4, 4),
            (2, 8, 2),
            (2, 8, 4),
            (2, 8, 8),
            (4, 4, 4),
            (4, 8, 4),
            (4, 8, 8),
            # 大量微批次
            (4, 16, 4),
            (4, 32, 4),
            (4, 64, 4),
            # 大组
            (4, 16, 16),
            (4, 32, 32),
            (4, 128, 64),
            # 奇数管道阶段
            (3, 2, 2),
            (3, 8, 2),
            (3, 12, 4),
            # 奇数 group_sizes
            (4, 6, 3),
            (4, 10, 5),
        ]
        # 对于每个测试用例，执行以下操作
        for num_local_stages, num_microbatches, group_size in test_cases:
            with self.subTest(
                num_local_stages=num_local_stages,
                num_microbatches=num_microbatches,
                group_size=group_size,
            ):
                # 打印当前测试用例的参数
                print(f"{num_local_stages=} {num_microbatches=} {group_size=}")
                # 计算总阶段数
                num_stages = num_local_stages * group_size
                # 创建包含多个 MockPipelineStage 实例的列表
                stages = [
                    MockPipelineStage(group_size=group_size, num_stages=num_stages)
                    for i in range(num_local_stages)
                ]

                # 创建 ScheduleClass 的实例，传入阶段列表和 num_microbatches
                schedule = ScheduleClass(stages, num_microbatches)
                # 验证管道顺序的有效性
                self._validate_pipeline_order(
                    schedule.pipeline_order, num_microbatches, num_stages
                )
# 实例化参数化测试，针对 TestSchedulePlan 进行测试准备
instantiate_parametrized_tests(TestSchedulePlan)

if __name__ == "__main__":
    # 创建一个 unittest 的加载器
    loader = unittest.TestLoader()
    # 从 TestSchedulePlan 类中加载测试用例到测试套件中
    suite = loader.loadTestsFromTestCase(TestSchedulePlan)
    # 创建一个文本形式的测试运行器
    runner = unittest.TextTestRunner()
    # 运行指定的测试套件
    runner.run(suite)

    # 检查是否有 GPU 和 NCCL 可用
    if not (
        dist.is_available()
        and dist.is_nccl_available()
        and torch.cuda.device_count() > 1
    ):
        # 如果 c10d NCCL 不可用或者 GPU 数量不足，跳过测试
        print(
            "c10d NCCL not available or not enough GPUs, skipping tests",
            file=sys.stderr,
        )
        # 退出程序运行，返回状态码 0 表示正常退出
        sys.exit(0)

    rank = int(os.getenv("RANK", -1))
    world_size = int(os.getenv("WORLD_SIZE", 2))

    if rank != -1:
        # 使用 torchrun 或其他多进程启动器启动。直接运行测试。
        ScheduleTest.run_rank(rank, world_size)
    else:
        # 作为单一进程启动。生成子进程来运行测试。
        # 同时为 `init_process_group` 创建一个会合文件。
        rdvz_file = tempfile.NamedTemporaryFile(delete=False).name
        # 使用 torch.multiprocessing.spawn 启动多进程来运行 ScheduleTest.run_rank
        torch.multiprocessing.spawn(
            ScheduleTest.run_rank,
            nprocs=world_size,
            args=(world_size, rdvz_file),
        )
```