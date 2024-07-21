# `.\pytorch\test\distributed\algorithms\test_join.py`

```py
# Owner(s): ["oncall: distributed"]

# 导入需要的模块
import contextlib  # 上下文管理工具库
import os  # 系统操作模块
import sys  # 系统相关的功能模块
from typing import Any, Optional  # 类型提示，用于声明参数和返回值类型

import torch  # PyTorch 深度学习框架
import torch.distributed as dist  # PyTorch 分布式模块

# 检查分布式功能是否可用，不可用则跳过测试
if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)

# 导入需要的分布式模块和测试工具
from torch.distributed.algorithms.join import Join, Joinable, JoinHook  # 导入分布式 join 相关模块
from torch.testing._internal.common_distributed import (
    MultiProcessTestCase,
    require_n_gpus_for_nccl_backend,
)  # 导入多进程测试用例和 GPU 数目需求函数
from torch.testing._internal.common_utils import run_tests, TEST_WITH_DEV_DBG_ASAN  # 导入运行测试函数和开发者调试模式标记

# 如果开发者调试模式为真，则跳过 dev-asan 测试，因为 torch + multiprocessing spawn 存在已知问题
if TEST_WITH_DEV_DBG_ASAN:
    print(
        "Skip dev-asan as torch + multiprocessing spawn have known issues",
        file=sys.stderr,
    )
    sys.exit(0)

# 根据 CUDA 是否可用选择分布式后端
BACKEND = dist.Backend.NCCL if torch.cuda.is_available() else dist.Backend.GLOO

# 设置全局进程数量，最小为 2，最大为当前系统上可用的 CUDA 设备数量（如果 CUDA 可用）
WORLD_SIZE = min(4, max(2, torch.cuda.device_count()))

# 用于测试后钩子的常量
BEFORE_CONSTANT = 41  # 测试前常量
AFTER_CONSTANT = 42  # 测试后常量


class AllReducerJoinHook(JoinHook):
    r"""
    Join hook for :class:`AllReducer`.

    Arguments:
        allreducer (AllReducer): the :class:`AllReducer` object using this
            hook.
        num_allreduces (int): the number of all-reduces to shadow per
            iteration.
        run_post_hook (bool): a flag enabling the post-hook logic.
    """

    def __init__(self, allreducer, num_allreduces, run_post_hook):
        # 初始化 join 钩子对象
        self.allreducer = allreducer  # 存储 AllReducer 实例
        self.num_allreduces = num_allreduces  # 每次迭代要模拟的 all-reduces 数量
        self.run_post_hook = run_post_hook  # 是否启用后钩子逻辑标志

    def main_hook(self):
        r"""
        Shadows each all-reduce; the number of all-reduces is passed into the
        constructor as ``num_allreduces``.
        """
        device = self.allreducer.device  # 获取设备信息
        # 模拟多次 all-reduce 操作
        for _ in range(self.num_allreduces):
            t = torch.zeros(1, device=device)  # 在指定设备上创建一个张量
            dist.all_reduce(t)  # 执行全局 all-reduce 操作

    def post_hook(self, is_last_joiner: bool):
        r"""
        Broadcasts a tensor containing a magic constant ``AFTER_CONSTANT`` from
        the last joiner to all other processes.
        """
        if not self.run_post_hook:  # 如果禁用了后钩子逻辑，则直接返回
            return
        rank = dist.get_rank(self.allreducer.process_group)  # 获取当前进程的排名
        common_rank = self.allreducer.find_common_rank(rank, is_last_joiner)  # 找到所有参与者的共同排名
        device = self.allreducer.device  # 获取设备信息
        if rank == common_rank:  # 如果当前进程是共同排名
            # 创建包含 AFTER_CONSTANT 的张量并存储在对象中
            self.allreducer.post_hook_tensor = torch.tensor(
                [AFTER_CONSTANT], device=device
            )
        dist.broadcast(self.allreducer.post_hook_tensor, src=common_rank)  # 将后钩子张量广播到所有进程


class AllReducer(Joinable):
    r"""
    Example :class:`Joinable` that performs some number of all-reduces as its
    per-iteration collective communication.
    """

    def __init__(self, device, process_group):
        super().__init__()  # 调用父类构造函数
        self.device = device  # 存储设备信息
        self.process_group = process_group  # 存储进程组信息
        # 初始化前钩子张量为 BEFORE_CONSTANT
        self.post_hook_tensor = torch.tensor([BEFORE_CONSTANT], device=self.device)
    def __call__(self, num_allreduces=1):
        r"""
        All-reduces a dim-1 one tensor ``num_allreduces``-many times, and
        returns the total result.
        """
        # 通知加入上下文
        Join.notify_join_context(self)
        # 获取设备信息
        device = self.device
        # 初始化总和
        total = 0
        # 循环执行 num_allreduces 次
        for _ in range(num_allreduces):
            # 创建一个在指定设备上的单位张量
            t = torch.ones(1, device=device)
            # 对张量 t 进行全局归约操作
            dist.all_reduce(t)
            # 累加归约结果到总和
            total += t.item()
        # 返回总和
        return total

    def join_hook(self, **kwargs) -> JoinHook:
        r"""
        Returns a join hook that shadows some number of all-reduces; by default,
        this number is 1.
        """
        # 获取关键字参数中的 num_allreduces，默认为 1
        num_allreduces = kwargs.get("num_allreduces", 1)
        # 获取关键字参数中的 run_post_hooks，默认为 False
        run_post_hook = kwargs.get("run_post_hooks", False)
        # 返回一个 AllReducerJoinHook 实例，用于监视指定数量的全局归约操作
        return AllReducerJoinHook(self, num_allreduces, run_post_hook)

    @property
    def join_device(self) -> torch.device:
        # 返回对象所在的设备信息
        return self.device

    @property
    def join_process_group(self) -> Any:
        # 返回对象所在的进程组信息
        return self.process_group

    def find_common_rank(self, rank, to_consider):
        r"""
        Returns the max rank of the ones to consider over the process group.
        """
        # 创建一个张量，其值为 rank（如果 to_consider 为真）或 -1
        common_rank = torch.tensor([rank if to_consider else -1], device=self.device)
        # 对 common_rank 进行全局归约操作，使用 MAX 运算符
        dist.all_reduce(common_rank, op=dist.ReduceOp.MAX, group=self.process_group)
        # 获取全局归约后的 common_rank 的数值，并确保其大于等于 0
        common_rank = common_rank.item()
        assert common_rank >= 0
        # 返回全局归约后的 common_rank
        return common_rank
    # 定义一个测试类，继承自 MultiProcessTestCase，用于测试通用的加入上下文
    class TestJoin(MultiProcessTestCase):
        """Test cases for the generic join context."""

        # 设置测试环境
        def setUp(self):
            super().setUp()  # 调用父类的 setUp 方法，准备测试环境
            # 设置环境变量 WORLD_SIZE 为当前进程数量
            os.environ["WORLD_SIZE"] = str(self.world_size)
            # 设置环境变量 BACKEND 为预定义的后端类型
            os.environ["BACKEND"] = BACKEND
            # 启动多个子进程来模拟多进程测试环境
            self._spawn_processes()

        # 返回设备对象
        @property
        def device(self):
            return (
                torch.device(self.rank)  # 如果使用 NCCL 后端，则返回当前进程的设备对象
                if BACKEND == dist.Backend.NCCL
                else torch.device("cpu")  # 否则返回 CPU 设备对象
            )

        # 返回世界大小，即当前测试环境中的进程数量
        @property
        def world_size(self):
            return WORLD_SIZE

        # 返回进程组对象，这里使用全局的 WORLD 组
        @property
        def process_group(self):
            return dist.group.WORLD

        # 清理测试环境
        def tearDown(self):
            try:
                dist.destroy_process_group()  # 销毁当前进程组
            except AssertionError:
                pass
            try:
                os.remove(self.file_name)  # 删除临时文件
            except OSError:
                pass

        # 初始化分布式进程组
        def dist_init(self, rank, world_size, backend=BACKEND):
            # 使用文件存储创建分布式存储对象
            store = dist.FileStore(self.file_name, world_size)
            # 初始化分布式进程组，指定后端、存储、当前进程的等级和总进程数量
            return dist.init_process_group(
                backend=backend, store=store, rank=rank, world_size=world_size
            )

        # 构建不均匀的输入数据
        def construct_uneven_inputs(self, base, offset, device=None):
            """
            Returns uneven inputs: rank i gets ``base`` + i * ``offset`` inputs.
            """
            if device is None:
                device = self.device  # 如果未指定设备，则使用默认设备
            # 返回列表，包含当前进程需要的不均匀数量的零张量
            return [torch.zeros(1, device=device) for _ in range(base + self.rank * offset)]

        # 构建均匀的输入数据
        def construct_even_inputs(self, base, device=None):
            """
            Returns even inputs: each rank gets ``base`` inputs.
            """
            if device is None:
                device = self.device  # 如果未指定设备，则使用默认设备
            # 返回列表，包含当前进程需要的均匀数量的零张量
            return [torch.zeros(1, device=device) for _ in range(base)]

        # 返回所有进程共享的基础输入数量
        @property
        def base_num_inputs(self):
            """
            Base number of inputs to be used by all ranks.
            """
            return 3

        # 返回当前进程的偏移量，用于计算不均匀输入的数量
        @property
        def offset(self):
            """
            Rank i gets i * ``offset`` additional inputs.
            """
            return 1

        # 私有方法，用于测试基本的加入操作
        def _test_join_base(
            self,
            uneven_inputs: bool,
            num_joinables: int,
            enable: bool,
            throw_on_early_termination: bool,
            num_allreduces: int,
            run_post_hooks: bool,
            expected_total: Optional[int] = None,
        ):
    ):
        r"""
        Skeleton for all :class:`Join` tests.

        Arguments:
            uneven_inputs (bool): ``True`` to use uneven inputs; ``False``
                otherwise.
            num_joinables (int): number of :class:`AllReducer` s to construct.
            enable (bool): ``True`` to enable the join context manager;
                ``False`` otherwise.
            throw_on_early_termination (bool): ``True`` to raise an exception
                upon detecting uneven inputs; ``False`` otherwise.
            num_allreduces (int): number of all-reduces to perform per input.
            run_post_hooks (bool): ``True`` to run post-hooks; ``False``
                otherwise.
            expected_total (Optional[int]): ``None`` to not check the expected
                all-reduce total; otherwise, the expected total; default is
                ``None``.
        """
        # 初始化分布式环境，设置当前进程的 rank 和总的进程数
        self.dist_init(self.rank, self.world_size)

        # 创建指定数量的 AllReducer 实例列表
        allreducers = [
            AllReducer(self.device, self.process_group) for _ in range(num_joinables)
        ]
        # 断言每个 AllReducer 实例的 post_hook_tensor 的初始值为 BEFORE_CONSTANT
        for allreducer in allreducers:
            self.assertEqual(allreducer.post_hook_tensor.item(), BEFORE_CONSTANT)

        # 根据 uneven_inputs 参数决定生成输入数据
        inputs = (
            self.construct_uneven_inputs(self.base_num_inputs, self.offset)
            if uneven_inputs
            else self.construct_even_inputs(self.base_num_inputs)
        )
        # 初始化 allreduce_total 用于统计所有执行的 all-reduce 操作次数
        allreduce_total = 0

        # 如果 throw_on_early_termination=True，则期望抛出 RuntimeError
        # 当 rank 0 先耗尽其输入时
        expected_msg = (
            "Rank 0 exhausted all inputs."
            if self.rank == 0
            else "Detected at least one rank that exhausted inputs. "
            "Throwing across all ranks."
        )
        # 在满足 throw_on_early_termination 条件时，使用 assertRaisesRegex 断言抛出指定异常
        # 否则使用 contextlib.nullcontext() 来进行上下文管理
        with self.assertRaisesRegex(
            RuntimeError, expected_msg
        ) if throw_on_early_termination else contextlib.nullcontext():
            # 使用 Join 上下文管理器，执行 all-reduce 操作并捕获可能的异常
            with Join(
                allreducers,
                enable=enable,
                throw_on_early_termination=throw_on_early_termination,
                num_allreduces=num_allreduces,
                run_post_hooks=run_post_hooks,
            ):
                # 遍历输入数据，对每个输入执行指定次数的 all-reduce 操作
                for _ in inputs:
                    for allreducer in allreducers:
                        allreduce_total += allreducer(num_allreduces)

        # 如果 throw_on_early_termination=True，则直接返回，不进行后续的断言和检查
        if throw_on_early_termination:
            return

        # 如果指定了 expected_total，则检查 allreduce_total 是否等于 expected_total
        if expected_total:
            self.assertEqual(allreduce_total, expected_total)

        # 当 run_post_hooks=True 时，断言所有 AllReducer 实例的 post_hook_tensor 已更新为 AFTER_CONSTANT
        if run_post_hooks:
            for allreducer in allreducers:
                self.assertEqual(allreducer.post_hook_tensor.item(), AFTER_CONSTANT)

    @require_n_gpus_for_nccl_backend(WORLD_SIZE, BACKEND)
    def test_single_joinable_main_hooks(self):
        r"""Tests the main hooks of a single :class:`Joinable`."""
        num_joinables = 1  # 定义一个变量，表示可加入的对象数量为1
        num_allreduces = 1  # 定义一个变量，表示进行全局归约操作的次数为1
        run_post_hooks = False  # 定义一个变量，表示是否运行后处理钩子为False

        # 非加入的进程进行全局归约1次，因此该进程的全局归约总数应该正好等于之前它加入之前处理的输入总数
        expected_total = self.world_size * self.base_num_inputs

        # 每个进程i运行额外i次迭代
        for num_joined in range(1, self.rank + 1):
            expected_total += (self.world_size - num_joined) * self.offset

        # 调用内部方法 _test_join_base 进行测试
        self._test_join_base(
            uneven_inputs=True,  # 使用不均匀输入进行测试
            num_joinables=num_joinables,  # 设置可加入的对象数量
            enable=True,  # 启用测试
            throw_on_early_termination=False,  # 不在早期终止时抛出异常
            num_allreduces=num_allreduces,  # 设置全局归约操作的次数
            run_post_hooks=run_post_hooks,  # 设置是否运行后处理钩子
            expected_total=expected_total,  # 设置预期的全局归约总数
        )

    @require_n_gpus_for_nccl_backend(WORLD_SIZE, BACKEND)
    def test_single_joinable_post_hooks(self):
        r"""Tests the post-hooks of a single :class:`Joinable`."""
        num_joinables = 1  # 定义一个变量，表示可加入的对象数量为1
        num_allreduces = 0  # 将全局归约次数设置为0，以跳过主钩子
        run_post_hooks = False  # 定义一个变量，表示是否运行后处理钩子为False

        # 调用内部方法 _test_join_base 进行测试
        self._test_join_base(
            uneven_inputs=True,  # 使用不均匀输入进行测试
            num_joinables=num_joinables,  # 设置可加入的对象数量
            enable=True,  # 启用测试
            throw_on_early_termination=False,  # 不在早期终止时抛出异常
            num_allreduces=num_allreduces,  # 设置全局归约操作的次数
            run_post_hooks=run_post_hooks,  # 设置是否运行后处理钩子
            expected_total=None,  # 设置预期的全局归约总数为None
        )

    @require_n_gpus_for_nccl_backend(WORLD_SIZE, BACKEND)
    def test_single_joinable(self):
        r"""
        Tests the main hooks and post-hooks of a single :class:`Joinable`
        together.

        This combines ``test_single_joinable_main_hooks()`` and
        ``test_single_joinable_post_hooks()`` into a single test to ensure that
        main hooks and post-hooks operate correctly together.
        """
        num_joinables = 1  # 定义一个变量，表示可加入的对象数量为1
        num_allreduces = 1  # 定义一个变量，表示进行全局归约操作的次数为1
        run_post_hooks = True  # 定义一个变量，表示是否运行后处理钩子为True

        expected_total = self.world_size * self.base_num_inputs

        # 每个进程i运行额外i次迭代
        for num_joined in range(1, self.rank + 1):
            expected_total += (self.world_size - num_joined) * self.offset

        # 调用内部方法 _test_join_base 进行测试
        self._test_join_base(
            uneven_inputs=True,  # 使用不均匀输入进行测试
            num_joinables=num_joinables,  # 设置可加入的对象数量
            enable=True,  # 启用测试
            throw_on_early_termination=False,  # 不在早期终止时抛出异常
            num_allreduces=num_allreduces,  # 设置全局归约操作的次数
            run_post_hooks=run_post_hooks,  # 设置是否运行后处理钩子
            expected_total=expected_total,  # 设置预期的全局归约总数
        )

    @require_n_gpus_for_nccl_backend(WORLD_SIZE, BACKEND)
    def test_multiple_joinables(self):
        r"""
        Tests the main hooks and post-hooks of multiple :class:`Joinable` s
        together.

        This generalizes ``test_single_joinable()`` to multiple
        :class:`Joinable` s.
        """
        # 定义测试中的变量：Joinable 的数量和所有归约操作的数量
        num_joinables = 3
        num_allreduces = 1
        run_post_hooks = True

        # 计算预期的总数，基于当前进程的排名和世界大小
        expected_total = self.world_size * self.base_num_inputs
        for num_joined in range(1, self.rank + 1):
            expected_total += (self.world_size - num_joined) * self.offset
        # 将预期总数乘以 Joinable 的数量
        expected_total *= num_joinables

        # 调用测试基础方法 _test_join_base，传入各个参数进行测试
        self._test_join_base(
            uneven_inputs=True,
            num_joinables=num_joinables,
            enable=True,
            throw_on_early_termination=False,
            num_allreduces=num_allreduces,
            run_post_hooks=run_post_hooks,
            expected_total=expected_total,
        )

    @require_n_gpus_for_nccl_backend(WORLD_SIZE, BACKEND)
    def test_single_joinable_disable(self):
        r"""Tests ``enable=False`` for a single :class:`Joinable`."""
        # 定义测试中的变量：Joinable 的数量和所有归约操作的数量
        num_joinables = 1
        num_allreduces = 1
        uneven_inputs = False
        enable = False
        run_post_hooks = False

        # 计算预期的总数，基于当前进程的排名和世界大小
        expected_total = self.world_size * self.base_num_inputs

        # 调用测试基础方法 _test_join_base，传入各个参数进行测试
        self._test_join_base(
            uneven_inputs=uneven_inputs,
            num_joinables=num_joinables,
            enable=enable,
            throw_on_early_termination=False,
            num_allreduces=num_allreduces,
            run_post_hooks=run_post_hooks,
            expected_total=expected_total,
        )

    @require_n_gpus_for_nccl_backend(WORLD_SIZE, BACKEND)
    def test_multiple_joinable_disable(self):
        r"""
        Tests ``enable=False`` for multiple :class:`Joinable` s.

        This generalizes ``test_single_joinable_disable`` to multiple
        :class:`Joinable` s.
        """
        # 定义测试中的变量：Joinable 的数量和所有归约操作的数量
        num_joinables = 3
        num_allreduces = 1
        uneven_inputs = False
        enable = False
        run_post_hooks = False

        # 计算预期的总数，基于当前进程的排名、世界大小和 Joinable 的数量
        expected_total = self.world_size * self.base_num_inputs * num_joinables

        # 调用测试基础方法 _test_join_base，传入各个参数进行测试
        self._test_join_base(
            uneven_inputs=uneven_inputs,
            num_joinables=num_joinables,
            enable=enable,
            throw_on_early_termination=False,
            num_allreduces=num_allreduces,
            run_post_hooks=run_post_hooks,
            expected_total=expected_total,
        )

    @require_n_gpus_for_nccl_backend(WORLD_SIZE, BACKEND)
    def test_single_joinable_throw(self):
        r"""
        Tests ``throw_on_early_termination=True`` for a single
        :class:`Joinable`.
        """
        # 定义单个 Joinable 的测试用例
        num_joinables = 1
        num_allreduces = 1
        throw_on_early_termination = True
        run_post_hooks = False

        # 调用基础测试方法 _test_join_base 进行测试
        self._test_join_base(
            uneven_inputs=True,  # 使用不均匀输入进行测试
            num_joinables=num_joinables,  # 设置 Joinable 的数量
            enable=True,  # 启用测试
            throw_on_early_termination=throw_on_early_termination,  # 是否在早期终止时抛出异常
            num_allreduces=num_allreduces,  # 设置 AllReduce 操作的数量
            run_post_hooks=run_post_hooks,  # 是否运行后续钩子函数
            expected_total=None,  # 预期的总数为 None
        )

    @require_n_gpus_for_nccl_backend(WORLD_SIZE, BACKEND)
    def test_multiple_joinables_throw(self):
        r"""
        Tests ``throw_on_early_termination=True`` for multiple
        :class:`Joinable` s together.

        This generalizes ``test_single_joinable_throw`` to multiple
        :class:`Joinable` s.
        """
        # 定义多个 Joinable 一起测试的用例
        num_joinables = 3
        num_allreduces = 1
        throw_on_early_termination = True
        run_post_hooks = False

        # 调用基础测试方法 _test_join_base 进行测试
        self._test_join_base(
            uneven_inputs=True,  # 使用不均匀输入进行测试
            num_joinables=num_joinables,  # 设置 Joinable 的数量
            enable=True,  # 启用测试
            throw_on_early_termination=throw_on_early_termination,  # 是否在早期终止时抛出异常
            num_allreduces=num_allreduces,  # 设置 AllReduce 操作的数量
            run_post_hooks=run_post_hooks,  # 是否运行后续钩子函数
            expected_total=None,  # 预期的总数为 None
        )

    @require_n_gpus_for_nccl_backend(WORLD_SIZE, BACKEND)
    def test_join_kwargs(self):
        r"""
        Tests passing keyword arguments to the context manager.
        """
        # 测试将关键字参数传递给上下文管理器

        num_joinables = 1
        num_allreduces = 2
        run_post_hooks = False

        # 计算期望的总数
        expected_total = self.world_size * self.base_num_inputs
        for num_joined in range(1, self.rank + 1):
            expected_total += (self.world_size - num_joined) * self.offset
        # 现在将预期总数乘以 NUM_ALLREDUCES 的因子
        expected_total *= num_allreduces

        # 调用基础测试方法 _test_join_base 进行测试
        self._test_join_base(
            uneven_inputs=True,  # 使用不均匀输入进行测试
            num_joinables=num_joinables,  # 设置 Joinable 的数量
            enable=True,  # 启用测试
            throw_on_early_termination=False,  # 在早期终止时不抛出异常
            num_allreduces=num_allreduces,  # 设置 AllReduce 操作的数量
            run_post_hooks=run_post_hooks,  # 是否运行后续钩子函数
            expected_total=expected_total,  # 设置预期的总数
        )
# 如果当前脚本作为主程序执行（而不是被导入到其他模块中），则执行 run_tests() 函数
if __name__ == "__main__":
    run_tests()
```