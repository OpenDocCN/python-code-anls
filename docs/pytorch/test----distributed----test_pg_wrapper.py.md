# `.\pytorch\test\distributed\test_pg_wrapper.py`

```py
# Owner(s): ["oncall: distributed"]
# 导入必要的模块和库
import os
import sys
from datetime import timedelta
from unittest.mock import patch

import torch  # 导入PyTorch库
import torch.distributed as c10d  # 导入PyTorch分布式模块
from torch._C._distributed_c10d import _ProcessGroupWrapper  # 导入_ProcessGroupWrapper类

# 如果c10d不可用，则打印错误信息并退出
if not c10d.is_available():
    print("c10d not available, skipping tests", file=sys.stderr)
    sys.exit(0)

# 导入测试用例相关的模块和函数
from test_c10d_common import LOOPBACK

from torch.testing._internal.common_distributed import (
    create_device,
    MultiProcessTestCase,
    requires_gloo,
    requires_nccl,
    skip_if_lt_x_gpu,
    with_dist_debug_levels,
)
from torch.testing._internal.common_utils import run_tests, TEST_WITH_DEV_DBG_ASAN


class AbstractProcessGroupWrapperTest(MultiProcessTestCase):
    def setUp(self):
        super().setUp()
        self._spawn_processes()  # 生成多进程用于测试

    def _validate_error(self, exception, op_type, rank, tensor, verify_diff=True):
        err = str(exception)
        # 检查错误消息中是否包含操作类型
        self.assertTrue(
            op_type in err, f"Got {err} but expected {op_type} to be in error."
        )
        # 如果操作类型不是"BARRIER"，则验证张量形状和设备类型
        if op_type != "BARRIER":
            # 检查错误消息中是否包含张量的形状信息
            self.assertTrue(
                f"{list(tensor.shape)}" in err,
                f"Did not find shapes {list(tensor.shape)} in error {err}",
            )
            # 对于CUDA设备，只检查设备类型而不是具体的索引
            if "cuda" in str(tensor.device):
                self.assertTrue(
                    "cuda" in err, f"Did not find cuda device in error {err}"
                )
            else:
                # 检查错误消息中是否包含张量的设备信息
                self.assertTrue(
                    str(tensor.device) in err,
                    f"Did not find tensor device {str(tensor.device)} in error {err}",
                )
            # 检查错误消息中是否包含正确的数据类型信息（Float或Long）
            if "float" in str(tensor.dtype):
                self.assertTrue("Float" in err, "Expected Float type")
            elif "int" in str(tensor.dtype):
                self.assertTrue("Long" in err, "Expected Long type")
            else:
                self.fail(f"Unexpected dtype {str(tensor.dtype)} for error {err}")

            # 检查错误消息中是否包含SequenceNumber信息
            self.assertTrue("SequenceNumber" in err)
            # 如果需要验证diff，检查错误消息中是否包含collectives diff相关信息
            if verify_diff:
                self.assertTrue(
                    "Collectives differ in the following" in err, f"Got error {err}"
                )
    # 测试函数：测试在除了 rank 1 之外的所有进程调用 allreduce 后，wrapper_pg 是否能够检测到 hang，并报告 rank 1 的问题。
    def _test_collective_hang(self, wrapper_pg, use_cuda=False):
        # 如果当前进程的 rank 不是 faulty_rank，则执行以下操作
        faulty_rank = 1
        if self.rank != faulty_rank:
            # 创建一个大小为 (20, 10) 的随机张量
            tensor = torch.randn(20, 10)
            # 如果使用 CUDA，则将张量移动到当前进程的 GPU 上
            if use_cuda:
                tensor = tensor.to(self.rank)

            if self.rank == 0:
                # 如果当前进程的 rank 是 0，则报告 faulty_rank 的错误
                err = f"Ranks {faulty_rank} failed to pass monitoredBarrier"
            else:
                # 如果当前进程的 rank 不是 0，则提示查看 rank 0 的日志以获取错误信息
                err = "Please check rank 0 logs for faulty rank"

            # 在错误信息中追加可能由于 rank 在 rank 0 调用 allreduce 之前提前退出而导致的 Gloo 错误
            err += "|Connection closed by peer|Connection reset by peer"
            # 断言在执行 wrapper_pg.allreduce([tensor]) 时会抛出 RuntimeError 异常，异常信息为 err
            with self.assertRaisesRegex(RuntimeError, err):
                wrapper_pg.allreduce([tensor])

    # 测试函数：测试集体操作的匹配问题，包括 op 类型不一致的情况
    def _test_collectives_op_mismatch(self, wrapper_pg, use_cuda=False):
        # 创建一个大小为 (20, 10) 的随机张量
        tensor = torch.randn(20, 10)
        # 如果使用 CUDA，则将张量移动到当前进程的 GPU 上
        if use_cuda:
            tensor = tensor.to(self.rank)
        
        works = []
        # 执行 500 次成功的 collective 操作
        for _ in range(500):
            work = wrapper_pg.allreduce([tensor])
            works.append(work)

        # 等待所有 collective 操作完成
        for w in works:
            w.wait()

        # 模拟 op 类型不匹配的情况：在 rank 0 执行 allreduce，其他 rank 执行 reduce
        # 异常信息应包含关于不一致 collective、rank、张量形状、设备和数据类型的信息
        with self.assertRaisesRegex(RuntimeError, ".*") as cm:
            if self.rank == 0:
                wrapper_pg.allreduce([tensor])
            else:
                wrapper_pg.reduce([tensor])
        # 验证捕获的异常是否符合预期
        self._validate_error(
            exception=cm.exception,
            op_type="ALLREDUCE" if self.rank == 0 else "REDUCE",
            rank=self.rank,
            tensor=tensor,
        )

        # 捕获 op 类型不匹配异常：在 rank 0 执行 reduce，其他 rank 执行 barrier
        with self.assertRaisesRegex(RuntimeError, ".*") as cm:
            if self.rank == 0:
                wrapper_pg.reduce([tensor])
            else:
                wrapper_pg.barrier()
        # 验证捕获的异常是否符合预期
        self._validate_error(
            exception=cm.exception,
            op_type="REDUCE" if self.rank == 0 else "BARRIER",
            rank=self.rank,
            tensor=tensor,
        )

        # 捕获 op 类型不匹配异常：在 rank 0 执行 broadcast，其他 rank 执行 allgather
        with self.assertRaisesRegex(RuntimeError, ".*") as cm:
            if self.rank == 0:
                wrapper_pg.broadcast(tensor, 0)
            else:
                # 创建与 self.world_size 大小相同的 output_tensors 列表，并执行 allgather 操作
                output_tensors = [
                    torch.zeros_like(tensor) for _ in range(self.world_size)
                ]
                wrapper_pg.allgather([output_tensors], [tensor])
        # 验证捕获的异常是否符合预期
        self._validate_error(
            exception=cm.exception,
            op_type="BROADCAST" if self.rank == 0 else "ALLGATHER",
            rank=self.rank,
            tensor=tensor,
        )
    # 测试集体操作中的形状不匹配情况

    # 在分布式进程组中，确保所有进程都到达这一点
    wrapper_pg.barrier()

    # 根据进程的 rank 确定 tensor 的维度，rank 为 0 时维度为 2，否则为 10
    dim = 2 if self.rank == 0 else 10

    # 创建一个随机张量，形状为 (20, dim)
    tensor = torch.randn(20, dim)

    # 如果使用 CUDA，则将 tensor 移动到对应的 GPU 设备上
    if use_cuda:
        tensor = tensor.to(self.rank)

    # 使用断言检查是否会抛出指定异常（这里是 RuntimeError），并捕获异常上下文
    with self.assertRaisesRegex(RuntimeError, ".*") as cm:
        # 执行分布式所有约简操作，传入 tensor 的列表
        wrapper_pg.allreduce([tensor])

    # 调用 _validate_error 方法验证捕获的异常，检查异常类型、rank 和 tensor 的内容
    self._validate_error(
        exception=cm.exception,
        op_type="ALLREDUCE",
        rank=self.rank,
        tensor=tensor,
    )

    # 检查当张量的维度不匹配时是否会引发错误

    # 根据进程的 rank 决定 tensor 的形状，rank 为 0 时形状为 (20, 10, 2)，否则为 (20, 10)
    tensor = torch.randn(20, 10, 2) if self.rank == 0 else torch.randn(20, 10)

    # 如果使用 CUDA，则将 tensor 移动到对应的 GPU 设备上
    if use_cuda:
        tensor = tensor.to(self.rank)

    # 使用断言检查是否会抛出指定异常（这里是 RuntimeError），并捕获异常上下文
    with self.assertRaisesRegex(RuntimeError, ".*") as cm:
        # 执行分布式所有约简操作，传入 tensor 的列表
        wrapper_pg.allreduce([tensor])

    # 调用 _validate_error 方法验证捕获的异常，检查异常类型、rank 和 tensor 的内容
    self._validate_error(
        exception=cm.exception,
        op_type="ALLREDUCE",
        rank=self.rank,
        tensor=tensor,
    )

    # 检查 scatter 操作中的形状错误

    # 创建输入张量列表，根据 rank 的不同，张量包含不同数量的元素
    input = [
        torch.tensor(
            [self.rank] if self.rank == 0 else [self.rank, self.rank],
            device=self.rank if use_cuda else "cpu",
        )
        for _ in range(self.world_size)
    ]

    # 创建输出张量列表，根据 rank 的不同，张量包含不同数量的元素
    outputs = [
        torch.tensor(
            [-1] if self.rank == 0 else [-1, -1],
            device=self.rank if use_cuda else "cpu",
        )
        for _ in range(self.world_size)
    ]

    # 指定根进程的 rank
    root_rank = 0

    # 创建 scatter 操作的选项对象
    opts = c10d.ScatterOptions()
    opts.rootRank = root_rank

    # 使用断言检查是否会抛出指定异常（这里是 RuntimeError），并捕获异常上下文
    with self.assertRaisesRegex(RuntimeError, ".*") as cm:
        # 如果当前进程是根进程，执行 scatter 操作，并等待操作完成
        if self.rank == root_rank:
            wrapper_pg.scatter([outputs[self.rank]], [input], opts).wait()
        else:
            # 如果当前进程不是根进程，执行 scatter 操作，并等待操作完成
            wrapper_pg.scatter([outputs[self.rank]], [], opts).wait()

    # 调用 _validate_error 方法验证捕获的异常，检查异常类型、rank 和 tensor 的内容
    self._validate_error(
        exception=cm.exception,
        op_type="SCATTER",
        rank=self.rank,
        tensor=outputs[self.rank],
    )
# ASAN is not safe since we are spawning processes.
if not TEST_WITH_DEV_DBG_ASAN:
    # 如果未设置 ASAN 调试标志，则执行以下代码块

    @requires_gloo()
    @requires_nccl()
# 标记需要 Gloo 支持
@requires_gloo()
class ProcessGroupGlooWrapperTest(AbstractProcessGroupWrapperTest):
    # 设置选项函数，配置线程数和超时时间
    def opts(self, threads=2, timeout=10.0):
        # 创建 Gloo 进程组选项对象
        opts = c10d.ProcessGroupGloo._Options()
        opts._timeout = timeout  # 设置超时时间
        opts._devices = [create_device(interface=LOOPBACK)]  # 设置设备为环回接口
        opts._threads = threads  # 设置线程数
        return opts

    # 创建包装后的 Gloo 进程组对象
    def _create_wrapper_pg(self, with_new_group=False, timeout=10.0):
        # 使用文件存储创建 FileStore 对象
        store = c10d.FileStore(self.file_name, self.world_size)
        # 初始化 Gloo 进程组
        c10d.init_process_group(
            backend="gloo", rank=self.rank, world_size=self.world_size, store=store
        )
        if with_new_group:
            # 如果需要创建新进程组，则调用 new_group 函数
            pg = c10d.new_group(backend="gloo")
        else:
            # 否则创建新的 Gloo 进程组对象
            _pg = c10d.ProcessGroupGloo(
                store, self.rank, self.world_size, self.opts(timeout=timeout)
            )
            pg = c10d._create_process_group_wrapper(
                _pg,
                "unused",
                store,
                self.rank,
                self.world_size,
                timeout=timeout,
            )
        return pg

    # 测试集体操作是否挂起
    def test_collective_hang(self):
        pg = self._create_wrapper_pg(timeout=2.0)
        self._test_collective_hang(pg)

    # NOTE: these tests are separated by debug level instead of combined into
    # one due to https://github.com/pytorch/pytorch/issues/55967, they can be
    # combined after that is resolved.
    # 设置分布式调试级别为 DETAIL 的集体操作不匹配调试模式测试
    @with_dist_debug_levels(levels=["DETAIL"])
    def test_collectives_op_mismatch_debug_mode(self):
        pg = self._create_wrapper_pg(with_new_group=True)
        self._test_collectives_op_mismatch(pg)

    # 设置分布式调试级别为 OFF 的集体操作不匹配测试
    @with_dist_debug_levels(levels=["OFF"])
    def test_collectives_op_mismatch(self):
        pg = self._create_wrapper_pg(with_new_group=False)
        self._test_collectives_op_mismatch(pg)

    # 设置分布式调试级别为 DETAIL 的集体形状不匹配调试模式测试
    @with_dist_debug_levels(levels=["DETAIL"])
    def test_collective_shape_mismatch_debug_mode(self):
        pg = self._create_wrapper_pg(with_new_group=True)
        self._test_collective_shape_mismatch(pg)

    # 设置分布式调试级别为 OFF 的集体形状不匹配调试模式关闭测试
    @with_dist_debug_levels(levels=["OFF"])
    def test_collective_shape_mismatch_debug_mode_off(self):
        pg = self._create_wrapper_pg(with_new_group=False)
        self._test_collective_shape_mismatch(pg)

    # 如果 GPU 数量小于 4，则跳过测试
    @skip_if_lt_x_gpu(4)
    # 设置分布式调试级别为 DETAIL 的 CUDA 集体操作不匹配调试模式测试
    @with_dist_debug_levels(levels=["DETAIL"])
    def test_collectives_op_mismatch_cuda_debug_mode(self):
        pg = self._create_wrapper_pg(with_new_group=True)
        self._test_collectives_op_mismatch(pg, use_cuda=True)

    # 如果 GPU 数量小于 4，则跳过测试
    @skip_if_lt_x_gpu(4)
    # 设置分布式调试级别为 OFF 的 CUDA 集体操作不匹配测试
    @with_dist_debug_levels(levels=["OFF"])
    def test_collectives_op_mismatch_cuda(self):
        pg = self._create_wrapper_pg(with_new_group=False)
        self._test_collectives_op_mismatch(pg, use_cuda=True)

    # 设置分布式调试级别为 DETAIL 的 CUDA 集体形状不匹配调试模式测试
    @skip_if_lt_x_gpu(4)
    @with_dist_debug_levels(levels=["DETAIL"])
    def test_collective_shape_mismatch_cuda_debug_mode(self):
        pg = self._create_wrapper_pg(with_new_group=True)
        self._test_collective_shape_mismatch(pg)
    # 定义一个测试方法，用于测试在 CUDA 调试模式下的集合形状不匹配情况
    def test_collective_shape_mismatch_cuda_debug_mode(self):
        # 创建一个新的进程组，并获取其包装器
        pg = self._create_wrapper_pg(with_new_group=True)
        # 调用测试方法，测试集合操作中的形状不匹配情况，使用 CUDA 加速
        self._test_collective_shape_mismatch(pg, use_cuda=True)

    @skip_if_lt_x_gpu(4)
    @with_dist_debug_levels(levels=["OFF"])
    # 装饰器指定条件，如果 GPU 数量小于 4，则跳过该测试方法
    # 装饰器设置分布式调试级别为 "OFF"
    def test_collective_shape_mismatch_cuda(self):
        # 创建一个不包含新进程组的包装器
        pg = self._create_wrapper_pg(with_new_group=False)
        # 调用测试方法，测试集合操作中的形状不匹配情况，使用 CUDA 加速
        self._test_collective_shape_mismatch(pg, use_cuda=True)
# 如果当前脚本作为主程序运行（而不是被导入到其他脚本中），则执行以下代码块
if __name__ == "__main__":
    # 使用断言检查，确保当前 CUDA 上下文未被初始化，以避免在主进程上初始化 CUDA 上下文
    assert (
        not torch.cuda._initialized
    ), "test_pg_wrapper must not have initialized CUDA context on main process"

    # 运行测试函数
    run_tests()
```